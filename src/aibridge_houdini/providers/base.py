from __future__ import annotations

from typing import Protocol

from aibridge_houdini.types import BridgeResponse, UserRequest


class LLMProvider(Protocol):
    name: str

    def generate(self, request: UserRequest) -> BridgeResponse: ...
