from __future__ import annotations

import json
import textwrap
from typing import Any

from aibridge_houdini.execution.safety import SafetyReport
from aibridge_houdini.houdini.transport import ExecutionResult
from aibridge_houdini.types import Command


PROMPT = "ai-bridge> "
CONFIRM_PROMPT = "execute? [y/N]: "
EXIT_WORDS = {"exit", "quit", ":q"}
_INDENT = "    "


def read_user_input() -> str | None:
    try:
        return input(PROMPT)
    except (EOFError, KeyboardInterrupt):
        return None


def confirm_execute() -> bool:
    try:
        answer = input(CONFIRM_PROMPT)
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in {"y", "yes"}


def render_response(obj: Any) -> str:
    return json.dumps(obj.model_dump(), indent=2, ensure_ascii=False)


def render_clarification(command: Command) -> str:
    """Render a clarification turn (no code, no safety, no execution)."""
    lines = [
        f"provider: {command.provider}",
        "intent:   clarification",
    ]
    question = (command.question or "").strip()
    if not question:
        # Defensive: a well-formed clarification always has a question.
        question = "(no question provided)"
    lines.append(f"question: {question}")
    return "\n".join(lines)


def render_turn(
    command: Command,
    safety: SafetyReport,
    exec_result: ExecutionResult | None,
    *,
    declined: bool = False,
    bridge_available: bool = True,
) -> str:
    lines: list[str] = []
    lines.append(f"provider: {command.provider}")
    lines.append(f"summary:  {command.summary}")
    lines.append("code:")
    lines.append(textwrap.indent(command.houdini_python.rstrip("\n"), _INDENT))

    lines.append(f"safety:   {safety.verdict} (mode={safety.mode})")
    for f in safety.findings:
        lines.append(f"{_INDENT}- [{f.severity}] {f.rule} (line {f.line}): {f.message}")

    lines.append("result:")
    lines.extend(_render_result_block(safety, exec_result, declined, bridge_available))

    return "\n".join(lines)


def _render_result_block(
    safety: SafetyReport,
    exec_result: ExecutionResult | None,
    declined: bool,
    bridge_available: bool,
) -> list[str]:
    if safety.verdict == "blocked":
        return [f"{_INDENT}BLOCKED — code not executed."]
    if declined:
        return [f"{_INDENT}DECLINED — user did not confirm."]
    if not bridge_available:
        return [
            f"{_INDENT}NOT EXECUTED — no Houdini transport configured "
            "(set HYTHON_PATH to enable)."
        ]
    if exec_result is None:
        # Safety allowed it, bridge exists, but no result was produced.
        return [f"{_INDENT}NOT EXECUTED."]

    status = "ok" if exec_result.success else "FAILED"
    rc = exec_result.return_code
    duration = exec_result.duration_seconds
    head = f"{_INDENT}{status}"
    if rc is not None:
        head += f" (rc={rc}"
        if duration is not None:
            head += f", {duration:.2f}s"
        head += ")"
    elif duration is not None:
        head += f" ({duration:.2f}s)"
    out = [head]
    if exec_result.error:
        out.append(f"{_INDENT}error:")
        out.append(textwrap.indent(exec_result.error.rstrip("\n"), _INDENT * 2))
    out.extend(_render_stream("stdout", exec_result.stdout))
    out.extend(_render_stream("stderr", exec_result.stderr))
    return out


def _render_stream(name: str, value: str) -> list[str]:
    if not value:
        return [f"{_INDENT}{name}: (empty)"]
    return [
        f"{_INDENT}{name}:",
        textwrap.indent(value.rstrip("\n"), _INDENT * 2),
    ]
