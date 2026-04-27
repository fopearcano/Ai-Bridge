from __future__ import annotations

import json
import logging
from typing import Callable, Mapping

from aibridge_houdini.config import Settings
from aibridge_houdini.providers.base import LLMProvider, ProviderError
from aibridge_houdini.types import Command, UserRequest


log = logging.getLogger("aibridge.providers.router")


SceneContextFn = Callable[[], dict | None]


class RouterError(RuntimeError):
    """Raised when the router cannot produce a valid Command."""


REPAIR_HINT = (
    "IMPORTANT: Your previous response was not valid JSON matching the "
    "required schema. Reply with ONLY a JSON object containing exactly these "
    "fields: intent (string), summary (string), risk_level "
    '("low"|"medium"|"high"), requires_houdini (boolean), houdini_python '
    "(string), explanation (string), expected_result (string). "
    "No markdown, no commentary."
)


SCENE_CONTEXT_HEADER = (
    "Live Houdini scene context (JSON snapshot from the running session — "
    "use this to ground your plan; do NOT echo it back):"
)


def _augment_with_context(text: str, context: dict | None) -> str:
    """Append a JSON scene-context block to the user prompt if non-empty."""
    if not context:
        return text
    blob = json.dumps(context, indent=2, ensure_ascii=False, default=str)
    return f"{text}\n\n{SCENE_CONTEXT_HEADER}\n```json\n{blob}\n```"


class ProviderRouter:
    """Selects an LLM provider, sends prompts, and normalizes the response.

    On the first failure (invalid JSON / validation / provider-side error) the
    router makes ONE auto-repair attempt by re-issuing the request with a
    strengthened instruction prepended to the original prompt.
    """

    def __init__(
        self,
        settings: Settings,
        providers: Mapping[str, LLMProvider] | None = None,
        scene_context_fn: SceneContextFn | None = None,
    ) -> None:
        self._settings = settings
        self._cache: dict[str, LLMProvider] = dict(providers) if providers else {}
        if settings.default_provider not in settings.providers:
            # Defensive: config validation should already prevent this.
            raise RouterError(
                f"DEFAULT_PROVIDER={settings.default_provider} is not in "
                f"PROVIDERS={','.join(settings.providers)}"
            )
        self._active: str = settings.default_provider
        self._scene_context_fn = scene_context_fn

    @property
    def active(self) -> str:
        return self._active

    @property
    def enabled(self) -> list[str]:
        return list(self._settings.providers)

    def set_provider(self, name: str) -> None:
        name = name.strip().lower()
        if name not in self._settings.providers:
            raise RouterError(
                f"provider '{name}' is not enabled "
                f"(PROVIDERS={','.join(self._settings.providers)})"
            )
        self._build(name)  # surface construction errors immediately
        self._active = name
        log.info("active provider set to %s", name)

    def route(self, request: UserRequest) -> Command:
        provider = self._build(self._active)

        scene_context = self._safe_fetch_context()
        augmented_text = _augment_with_context(request.text, scene_context)
        augmented = (
            request if augmented_text is request.text else UserRequest(text=augmented_text)
        )
        log.info(
            "route -> %s (len=%d, scene_context=%s)",
            provider.name,
            len(augmented.text),
            "yes" if scene_context else "no",
        )

        try:
            plan = provider.generate(augmented)
            return Command.from_plan(provider.name, request, plan)
        except ProviderError as e:
            log.warning("router: %s failed (%s); attempting auto-repair", provider.name, e)
        except Exception as e:
            log.warning("router: %s raised %s; attempting auto-repair", provider.name, e)

        repair_request = UserRequest(text=f"{augmented.text}\n\n{REPAIR_HINT}")
        try:
            plan = provider.generate(repair_request)
            log.info("router: auto-repair succeeded for %s", provider.name)
            return Command.from_plan(provider.name, request, plan)
        except Exception as e:
            raise RouterError(
                f"{provider.name} failed after auto-repair: {e}"
            ) from e

    def _safe_fetch_context(self) -> dict | None:
        if self._scene_context_fn is None:
            return None
        try:
            ctx = self._scene_context_fn()
        except Exception as e:
            log.warning("scene_context_fn raised %s; sending request without context", e)
            return None
        if ctx and not isinstance(ctx, dict):
            log.warning("scene_context_fn returned %r; expected dict", type(ctx).__name__)
            return None
        return ctx or None

    def _build(self, name: str) -> LLMProvider:
        if name in self._cache:
            return self._cache[name]
        from aibridge_houdini.providers import make_provider

        # Build a per-provider Settings copy so make_provider's dispatch picks
        # the requested adapter even when it differs from default_provider.
        # model_copy on a frozen model returns a new instance without re-running
        # the cross-field validators, which is what we want here.
        provider_settings = self._settings.model_copy(update={"default_provider": name})
        try:
            provider = make_provider(provider_settings)
        except ProviderError as e:
            raise RouterError(f"cannot enable provider '{name}': {e}") from e
        self._cache[name] = provider
        return provider
