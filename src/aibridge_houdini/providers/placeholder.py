from __future__ import annotations

from aibridge_houdini.types import LLMPlan, UserRequest


class PlaceholderProvider:
    """Stand-in provider used until a real LLM adapter is configured."""

    name = "placeholder"

    def generate(self, request: UserRequest) -> LLMPlan:
        return LLMPlan(
            intent="unknown",
            summary=f"(stub) would translate: {request.text!r}",
            risk_level="low",
            requires_houdini=False,
            houdini_python="# TODO: generated Houdini Python will go here",
            explanation="Placeholder response. No LLM call performed yet.",
            expected_result="No scene change.",
        )
