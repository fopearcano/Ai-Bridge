from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


RiskLevel = Literal["low", "medium", "high"]


class UserRequest(BaseModel):
    text: str


class LLMPlan(BaseModel):
    """Strict structured output produced by an LLM provider."""

    intent: str
    summary: str
    risk_level: RiskLevel
    requires_houdini: bool
    houdini_python: str
    explanation: str
    expected_result: str


class Command(BaseModel):
    """Normalized command produced by the router for one user turn.

    This is the bridge's single output unit: an executable plan plus the
    metadata needed to display, log, or hand off to the execution layer.
    """

    provider: str
    request: str
    intent: str
    summary: str
    risk_level: RiskLevel
    requires_houdini: bool
    houdini_python: str
    explanation: str
    expected_result: str

    @classmethod
    def from_plan(cls, provider: str, request: UserRequest, plan: LLMPlan) -> "Command":
        return cls(
            provider=provider,
            request=request.text,
            intent=plan.intent,
            summary=plan.summary,
            risk_level=plan.risk_level,
            requires_houdini=plan.requires_houdini,
            houdini_python=plan.houdini_python,
            explanation=plan.explanation,
            expected_result=plan.expected_result,
        )
