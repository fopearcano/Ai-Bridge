from __future__ import annotations

import logging
from typing import Mapping

from aibridge_houdini.config import Settings
from aibridge_houdini.providers.base import LLMProvider, ProviderError
from aibridge_houdini.types import Command, UserRequest


log = logging.getLogger("aibridge.providers.router")


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
        log.info("route -> %s (len=%d)", provider.name, len(request.text))

        try:
            plan = provider.generate(request)
            return Command.from_plan(provider.name, request, plan)
        except ProviderError as e:
            log.warning("router: %s failed (%s); attempting auto-repair", provider.name, e)
        except Exception as e:
            log.warning("router: %s raised %s; attempting auto-repair", provider.name, e)

        repair_request = UserRequest(text=f"{request.text}\n\n{REPAIR_HINT}")
        try:
            plan = provider.generate(repair_request)
            log.info("router: auto-repair succeeded for %s", provider.name)
            return Command.from_plan(provider.name, request, plan)
        except Exception as e:
            raise RouterError(
                f"{provider.name} failed after auto-repair: {e}"
            ) from e

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
