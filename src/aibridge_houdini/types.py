from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


RiskLevel = Literal["low", "medium", "high"]
CLARIFICATION_INTENT = "clarification"


class UserRequest(BaseModel):
    text: str


class LLMPlan(BaseModel):
    """Structured output produced by an LLM provider.

    Two valid shapes:

    * **Plan** — the normal case. All seven of summary, risk_level,
      requires_houdini, houdini_python, explanation, expected_result are
      filled; ``question`` is null/absent.
    * **Clarification** — when the request is ambiguous. ``intent`` is
      ``"clarification"``, ``requires_houdini`` is ``False``, and
      ``question`` carries the follow-up question. The other text fields
      may be empty strings (clients should ignore them).
    """

    intent: str
    summary: str = ""
    risk_level: RiskLevel = "low"
    requires_houdini: bool = True
    houdini_python: str = ""
    explanation: str = ""
    expected_result: str = ""
    question: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "LLMPlan":
        if self.intent == CLARIFICATION_INTENT:
            if not (self.question and self.question.strip()):
                raise ValueError(
                    "clarification plans must include a non-empty 'question'"
                )
            if self.requires_houdini:
                raise ValueError(
                    "clarification plans must have requires_houdini=False"
                )
        else:
            missing = [
                name
                for name in (
                    "summary",
                    "houdini_python",
                    "explanation",
                    "expected_result",
                )
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"plan missing required fields: {missing}")
        return self


class Command(BaseModel):
    """Normalized command produced by the router for one user turn.

    Mirrors LLMPlan and adds the provider name + the user's original prompt.
    Carries ``question`` when the plan was a clarification.
    """

    provider: str
    request: str
    intent: str
    summary: str = ""
    risk_level: RiskLevel = "low"
    requires_houdini: bool = True
    houdini_python: str = ""
    explanation: str = ""
    expected_result: str = ""
    question: str | None = None

    @property
    def is_clarification(self) -> bool:
        return self.intent == CLARIFICATION_INTENT

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
            question=plan.question,
        )
