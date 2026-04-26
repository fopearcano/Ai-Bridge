from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Intent = Literal["create", "modify", "query", "unknown"]


class UserRequest(BaseModel):
    text: str


class HoudiniAction(BaseModel):
    """A single planned operation against the Houdini scene."""

    intent: Intent = "unknown"
    summary: str = ""
    python_code: str = ""


class BridgeResponse(BaseModel):
    """Structured response returned by the bridge for one user turn."""

    request: UserRequest
    actions: list[HoudiniAction] = Field(default_factory=list)
    explanation: str = ""
    executed: bool = False
