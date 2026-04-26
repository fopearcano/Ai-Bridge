from __future__ import annotations

from aibridge_houdini.types import BridgeResponse, HoudiniAction, UserRequest


class PlaceholderProvider:
    """Stand-in provider used until real OpenAI/Anthropic adapters land."""

    name = "placeholder"

    def generate(self, request: UserRequest) -> BridgeResponse:
        action = HoudiniAction(
            intent="unknown",
            summary=f"(stub) would translate: {request.text!r}",
            python_code="# TODO: generated Houdini Python will go here",
        )
        return BridgeResponse(
            request=request,
            actions=[action],
            explanation="Placeholder response. No LLM call performed yet.",
            executed=False,
        )
