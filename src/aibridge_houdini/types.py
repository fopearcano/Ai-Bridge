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


class BridgeResponse(BaseModel):
    """One full bridge turn: the user request and the plan returned by the LLM."""

    request: UserRequest
    plan: LLMPlan
    executed: bool = False
