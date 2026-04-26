from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from pydantic import ValidationError

from aibridge_houdini.config import Settings
from aibridge_houdini.providers.base import ProviderError
from aibridge_houdini.types import LLMPlan, UserRequest


log = logging.getLogger("aibridge.providers.openai")


SYSTEM_INSTRUCTION = (
    "You are the reasoning engine of Ai-Bridge_Houdini. "
    "Return ONLY valid JSON. No markdown."
)

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent",
        "summary",
        "risk_level",
        "requires_houdini",
        "houdini_python",
        "explanation",
        "expected_result",
    ],
    "properties": {
        "intent": {"type": "string"},
        "summary": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "requires_houdini": {"type": "boolean"},
        "houdini_python": {"type": "string"},
        "explanation": {"type": "string"},
        "expected_result": {"type": "string"},
    },
}


class OpenAIProvider:
    """OpenAI Responses API adapter producing a strict-JSON LLMPlan.

    Does NOT execute any generated code — that's the execution layer's job.
    """

    name = "openai"

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if settings.openai_api_key is None:
            raise ProviderError("OpenAIProvider requires OPENAI_API_KEY")
        self._model = settings.openai_model
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = client if client is not None else _build_default_client(settings)

    def generate(self, request: UserRequest) -> LLMPlan:
        last_err: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                log.info(
                    "responses.create attempt=%d/%d model=%s len=%d",
                    attempt,
                    self._max_retries,
                    self._model,
                    len(request.text),
                )
                response = self._client.responses.create(
                    model=self._model,
                    instructions=SYSTEM_INSTRUCTION,
                    input=request.text,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "houdini_plan",
                            "schema": JSON_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                payload = _extract_text(response)
                log.debug("openai output text (%d chars)", len(payload))
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
                    "transient OpenAI error %s (attempt %d/%d); retry in %ds",
                    type(e).__name__,
                    attempt,
                    self._max_retries,
                    wait,
                )
                self._sleep(wait)
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                log.warning(
                    "invalid LLM payload (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    e,
                )
                if attempt == self._max_retries:
                    break
                self._sleep(1)
            except Exception as e:
                # Non-retryable (auth, bad request, schema rejection, etc.)
                log.exception("non-retryable OpenAI error")
                raise ProviderError(f"OpenAI request failed: {e}") from e

        raise ProviderError(
            f"OpenAI provider failed after {self._max_retries} attempts: {last_err}"
        )


def _build_default_client(settings: Settings) -> Any:
    from openai import OpenAI  # imported lazily so tests don't require the SDK

    assert settings.openai_api_key is not None
    return OpenAI(api_key=settings.openai_api_key.get_secret_value())


def _retryable_errors() -> tuple[type[BaseException], ...]:
    """Return the OpenAI exception classes we should retry on, if SDK is installed."""
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
    except Exception:
        return ()
    return (APIConnectionError, APITimeoutError, RateLimitError)


def _extract_text(response: Any) -> str:
    """Pull the assistant's text payload out of a Responses API result."""
    text = getattr(response, "output_text", None)
    if text:
        return text
    output = getattr(response, "output", None) or []
    for item in output:
        for content in getattr(item, "content", []) or []:
            t = getattr(content, "text", None)
            if t:
                return t
    raise ProviderError("OpenAI response had no text output")
