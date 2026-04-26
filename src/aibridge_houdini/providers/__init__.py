"""LLM provider adapters."""

from aibridge_houdini.config import Settings
from aibridge_houdini.providers.base import LLMProvider, ProviderError
from aibridge_houdini.providers.placeholder import PlaceholderProvider


def make_provider(settings: Settings) -> LLMProvider:
    """Construct the provider selected by `DEFAULT_PROVIDER`."""
    if settings.default_provider == "openai":
        from aibridge_houdini.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    if settings.default_provider == "anthropic":
        # Anthropic adapter not implemented yet; fall back to placeholder
        # so the loop stays usable until that lands.
        return PlaceholderProvider()
    if settings.default_provider == "lmstudio":
        from aibridge_houdini.providers.lmstudio_provider import LMStudioProvider

        return LMStudioProvider(settings)
    raise ProviderError(f"unknown provider: {settings.default_provider}")


__all__ = ["LLMProvider", "ProviderError", "PlaceholderProvider", "make_provider"]
