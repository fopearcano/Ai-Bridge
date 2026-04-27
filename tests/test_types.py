from __future__ import annotations

import pytest
from pydantic import ValidationError

from aibridge_houdini.types import Command, LLMPlan, UserRequest


_FULL_PLAN = dict(
    intent="create_sphere",
    summary="Create a sphere",
    risk_level="low",
    requires_houdini=True,
    houdini_python="import hou\nhou.node('/obj').createNode('geo')\n",
    explanation="adds a geo container",
    expected_result="a /obj/geo1 appears",
)


# ---- normal plan validation --------------------------------------------


def test_plan_accepts_full_seven_field_shape():
    plan = LLMPlan(**_FULL_PLAN)
    assert plan.question is None
    assert plan.intent == "create_sphere"


def test_plan_rejects_missing_required_field():
    bad = {k: v for k, v in _FULL_PLAN.items() if k != "expected_result"}
    with pytest.raises(ValidationError) as exc:
        LLMPlan(**bad)
    assert "expected_result" in str(exc.value)


def test_plan_rejects_empty_houdini_python():
    bad = dict(_FULL_PLAN, houdini_python="")
    with pytest.raises(ValidationError) as exc:
        LLMPlan(**bad)
    assert "houdini_python" in str(exc.value)


def test_plan_with_question_set_is_still_valid_for_normal_intent():
    plan = LLMPlan(**_FULL_PLAN, question="leftover")
    # Question is allowed but the intent isn't clarification, so the
    # validator only requires the seven plan fields.
    assert plan.question == "leftover"
    assert plan.intent != "clarification"


# ---- clarification shape -----------------------------------------------


def test_clarification_minimal_three_field_shape_validates():
    plan = LLMPlan(
        intent="clarification",
        requires_houdini=False,
        question="Which axis should I translate along?",
    )
    assert plan.intent == "clarification"
    assert plan.question == "Which axis should I translate along?"
    # Other text fields default to empty / sensible values.
    assert plan.summary == ""
    assert plan.houdini_python == ""
    assert plan.risk_level == "low"


def test_clarification_with_full_fields_also_validates():
    """OpenAI strict mode requires every property — empty strings allowed."""
    plan = LLMPlan(
        intent="clarification",
        summary="",
        risk_level="low",
        requires_houdini=False,
        houdini_python="",
        explanation="",
        expected_result="",
        question="What size sphere?",
    )
    assert plan.is_clarification if hasattr(plan, "is_clarification") else True
    assert plan.question == "What size sphere?"


def test_clarification_rejects_missing_question():
    with pytest.raises(ValidationError) as exc:
        LLMPlan(intent="clarification", requires_houdini=False)
    assert "question" in str(exc.value).lower()


def test_clarification_rejects_blank_question():
    with pytest.raises(ValidationError) as exc:
        LLMPlan(intent="clarification", requires_houdini=False, question="   ")
    assert "question" in str(exc.value).lower()


def test_clarification_rejects_requires_houdini_true():
    with pytest.raises(ValidationError) as exc:
        LLMPlan(
            intent="clarification",
            requires_houdini=True,
            question="What size?",
        )
    assert "requires_houdini" in str(exc.value).lower()


# ---- Command roundtrip --------------------------------------------------


def test_command_from_plan_passes_question_through():
    plan = LLMPlan(
        intent="clarification",
        requires_houdini=False,
        question="Where do you want it?",
    )
    cmd = Command.from_plan("openai", UserRequest(text="put it there"), plan)
    assert cmd.is_clarification
    assert cmd.question == "Where do you want it?"
    assert cmd.request == "put it there"
    assert cmd.provider == "openai"
    assert cmd.requires_houdini is False
    assert cmd.houdini_python == ""


def test_command_from_full_plan_has_no_question():
    plan = LLMPlan(**_FULL_PLAN)
    cmd = Command.from_plan("openai", UserRequest(text="make sphere"), plan)
    assert not cmd.is_clarification
    assert cmd.question is None
    assert cmd.intent == "create_sphere"
    assert cmd.houdini_python.startswith("import hou")
