from __future__ import annotations

from typing import Protocol

from aibridge_houdini.types import LLMPlan, UserRequest


class ProviderError(RuntimeError):
    """Raised when a provider can't produce a valid plan."""


class LLMProvider(Protocol):
    name: str

    def generate(self, request: UserRequest) -> LLMPlan: ...
