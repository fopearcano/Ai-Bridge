from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

from pydantic import ValidationError

from aibridge_houdini.config import Settings
from aibridge_houdini.providers.base import ProviderError
from aibridge_houdini.types import LLMPlan, UserRequest


log = logging.getLogger("aibridge.providers.lmstudio")


SYSTEM_INSTRUCTION = (
    "You are the reasoning engine of Ai-Bridge_Houdini. "
    "Return ONLY valid JSON. No markdown."
)

DEFAULT_TIMEOUT_SECONDS = 60.0

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class LMStudioProvider:
    """Adapter for LM Studio's OpenAI-compatible local server.

    Uses chat.completions (LM Studio does not implement the Responses API)
    with response_format=json_object, plus a system instruction that mandates
    JSON-only output. Does NOT execute any returned Houdini code.
    """

    name = "lmstudio"

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        max_retries: int = 3,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.lmstudio_model:
            raise ProviderError("LMStudioProvider requires LMSTUDIO_MODEL")
        self._model = settings.lmstudio_model
        self._max_retries = max_retries
        self._timeout = timeout
        self._sleep = sleep
        self._client = client if client is not None else _build_default_client(
            settings, timeout
        )

    def generate(self, request: UserRequest) -> LLMPlan:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": request.text},
        ]
        last_err: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                log.info(
                    "chat.completions attempt=%d/%d model=%s timeout=%.1fs len=%d",
                    attempt,
                    self._max_retries,
                    self._model,
                    self._timeout,
                    len(request.text),
                )
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=self._timeout,
                )
                payload = _extract_text(response)
                payload = _strip_code_fence(payload)
                log.debug("lmstudio output text (%d chars)", len(payload))
                data = json.loads(payload)
                plan = LLMPlan.model_validate(data)
                log.info(
                    "plan ok intent=%s risk=%s requires_houdini=%s",
                    plan.intent,
                    plan.risk_level,
                    plan.requires_houdini,
                )
                return plan
            except _retryable_errors() as e:
                last_err = e
                if attempt == self._max_retries:
                    break
                wait = 2**attempt
                log.warning(
                    "transient LM Studio error %s (attempt %d/%d); retry in %ds",
                    type(e).__name__,
                    attempt,
                    self._max_retries,
                    wait,
                )
                self._sleep(wait)
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                log.warning(
                    "invalid LM Studio payload (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    e,
                )
                if attempt == self._max_retries:
                    break
                self._sleep(1)
            except Exception as e:
                log.exception("non-retryable LM Studio error")
                raise ProviderError(f"LM Studio request failed: {e}") from e

        raise ProviderError(
            f"LM Studio provider failed after {self._max_retries} attempts: {last_err}"
        )


def _build_default_client(settings: Settings, timeout: float) -> Any:
    from openai import OpenAI  # imported lazily so tests don't require the SDK

    return OpenAI(
        base_url=settings.lmstudio_base_url,
        api_key=settings.lmstudio_api_key.get_secret_value(),
        timeout=timeout,
    )


def _retryable_errors() -> tuple[type[BaseException], ...]:
    """OpenAI-SDK exceptions worth retrying when talking to LM Studio."""
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
    except Exception:
        return ()
    return (APIConnectionError, APITimeoutError, RateLimitError)


def _extract_text(response: Any) -> str:
    """Pull the assistant's text out of a chat.completions result."""
    try:
        choices = response.choices
        if not choices:
            raise ProviderError("LM Studio response had no choices")
        content = choices[0].message.content
    except AttributeError as e:
        raise ProviderError(f"unexpected LM Studio response shape: {e}") from e
    if not content:
        raise ProviderError("LM Studio response had empty content")
    return content


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text
