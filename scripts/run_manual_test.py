#!/usr/bin/env python3
"""Run the manual prompt catalog against a real LLM provider.

This is **not** a pytest target — it makes real network/LLM calls and
optionally executes generated code via the Houdini bridge. Use it to
sanity-check a new model, prompt, or pipeline change end-to-end.

The PROMPTS list below is the source of truth; the markdown reference at
``tests/manual_houdini_prompts.md`` mirrors it. ``tests/test_manual_runner.py``
asserts they stay in sync.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make this script runnable from a fresh checkout without `pip install -e .`
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aibridge_houdini.config import ConfigError, Settings
from aibridge_houdini.execution.safety import SafetyReport, evaluate
from aibridge_houdini.houdini.bridge import HoudiniBridge
from aibridge_houdini.houdini.scene_client import fetch_scene_context
from aibridge_houdini.houdini.transport import ExecutionResult, TransportError
from aibridge_houdini.providers import ProviderRouter, RouterError
from aibridge_houdini.types import Command, UserRequest


@dataclass(frozen=True)
class PromptCase:
    id: str
    prompt: str
    expected_intents: tuple[str, ...]
    expected_risks: tuple[str, ...]


# Source of truth. Keep in sync with tests/manual_houdini_prompts.md.
PROMPTS: tuple[PromptCase, ...] = (
    PromptCase(
        id="sphere",
        prompt="Create a sphere",
        expected_intents=("create_sphere", "create_geometry"),
        expected_risks=("low",),
    ),
    PromptCase(
        id="procedural_rock",
        prompt="Make a procedural rock",
        expected_intents=("create_rock", "create_procedural_rock"),
        expected_risks=("low", "medium"),
    ),
    PromptCase(
        id="desert_scatter",
        prompt="Scatter a few rocks across a desert ground plane",
        expected_intents=(
            "create_scatter",
            "create_desert_scatter",
            "scatter_rocks",
        ),
        expected_risks=("medium",),
    ),
    PromptCase(
        id="explosion_setup",
        prompt="Set up an explosion without running the sim",
        expected_intents=("create_explosion_placeholder", "create_pyro_setup"),
        expected_risks=("medium",),
    ),
    PromptCase(
        id="camera",
        prompt="Add a camera looking at the origin",
        expected_intents=("create_camera", "create_camera_and_light"),
        expected_risks=("low",),
    ),
    PromptCase(
        id="scale_selected",
        prompt="Scale the selected nodes by 2",
        expected_intents=("modify_selected", "scale_selected"),
        expected_risks=("medium",),
    ),
    PromptCase(
        id="assign_material",
        prompt="Assign a default principled shader to the selected geometry",
        expected_intents=(
            "assign_material",
            "create_material",
            "apply_principled",
        ),
        expected_risks=("medium",),
    ),
    PromptCase(
        id="network_organization",
        prompt=(
            "Lay out and color-code the /obj network so rendering nodes are "
            "blue and geometry is yellow"
        ),
        expected_intents=("organize_network", "layout_network", "color_nodes"),
        expected_risks=("low", "medium"),
    ),
)


@dataclass
class CaseOutcome:
    id: str
    prompt: str
    provider: str = ""
    intent: str = ""
    risk_level: str = ""
    summary: str = ""
    code_lines: int = 0
    is_clarification: bool = False
    question: str | None = None
    intent_match: bool | None = None
    risk_match: bool | None = None
    safety_verdict: str | None = None
    safety_findings: int = 0
    executed: bool = False
    exec_success: bool | None = None
    exec_return_code: int | None = None
    exec_duration_s: float | None = None
    error: str | None = None


# ---- formatting helpers ------------------------------------------------


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _short(text: str, width: int = 72) -> str:
    return textwrap.shorten(text, width=width, placeholder="…")


def _print_case_header(case: PromptCase) -> None:
    print()
    print(_hr("═"))
    print(f"[{case.id}]  {case.prompt}")
    print(
        f"expected: intent ∈ {{{', '.join(case.expected_intents)}}}, "
        f"risk ∈ {{{', '.join(case.expected_risks)}}}"
    )
    print(_hr())


def _print_command(command: Command) -> None:
    print(f"provider:   {command.provider}")
    print(f"intent:     {command.intent}")
    if command.is_clarification:
        print(f"question:   {(command.question or '').strip()}")
        return
    print(f"risk_level: {command.risk_level}")
    print(f"summary:    {_short(command.summary, 120)}")
    print(f"explain:    {_short(command.explanation, 120)}")
    print("code:")
    print(textwrap.indent(command.houdini_python.rstrip(), "    "))


def _print_safety(report: SafetyReport) -> None:
    print(f"safety:     {report.verdict} (mode={report.mode})")
    for f in report.findings:
        print(f"    - [{f.severity}] {f.rule} (line {f.line}): {f.message}")


def _print_exec(result: ExecutionResult) -> None:
    head = "ok" if result.success else "FAILED"
    if result.return_code is not None:
        head += f" rc={result.return_code}"
    if result.duration_seconds is not None:
        head += f" ({result.duration_seconds:.2f}s)"
    print(f"result:     {head}")
    if result.error:
        print("error:")
        print(textwrap.indent(result.error.rstrip(), "    "))
    if result.stdout:
        print("stdout:")
        print(textwrap.indent(result.stdout.rstrip(), "    "))
    if result.stderr:
        print("stderr:")
        print(textwrap.indent(result.stderr.rstrip(), "    "))


# ---- run logic ---------------------------------------------------------


def _build_router(settings: Settings, provider: str | None) -> ProviderRouter:
    router = ProviderRouter(
        settings,
        scene_context_fn=lambda: fetch_scene_context(
            settings.houdini_host, settings.houdini_port, timeout=2.0
        ),
    )
    if provider:
        router.set_provider(provider)
    return router


def _try_build_bridge(settings: Settings) -> HoudiniBridge | None:
    try:
        return HoudiniBridge(settings, transport="hython")
    except TransportError as e:
        logging.getLogger("manual").warning(
            "no Houdini transport: %s — execution will be skipped", e
        )
        return None


def _run_case(
    case: PromptCase,
    router: ProviderRouter,
    bridge: HoudiniBridge | None,
    *,
    mode: str,
    do_execute: bool,
) -> CaseOutcome:
    outcome = CaseOutcome(id=case.id, prompt=case.prompt)
    _print_case_header(case)

    try:
        command = router.route(UserRequest(text=case.prompt))
    except RouterError as e:
        outcome.error = f"router: {e}"
        print(f"ERROR  {outcome.error}")
        return outcome
    except Exception as e:  # safety net for SDK weirdness
        outcome.error = f"unexpected: {e}"
        print(f"ERROR  {outcome.error}")
        return outcome

    outcome.provider = command.provider
    outcome.intent = command.intent
    outcome.is_clarification = command.is_clarification
    outcome.question = command.question
    outcome.risk_level = command.risk_level
    outcome.summary = command.summary
    outcome.code_lines = command.houdini_python.count("\n") if command.houdini_python else 0
    outcome.intent_match = command.intent in case.expected_intents
    outcome.risk_match = (
        None if command.is_clarification else command.risk_level in case.expected_risks
    )

    _print_command(command)

    if command.is_clarification:
        return outcome

    safety = evaluate(command.houdini_python, mode=mode)
    outcome.safety_verdict = safety.verdict
    outcome.safety_findings = len(safety.findings)
    _print_safety(safety)

    if not do_execute:
        return outcome
    if bridge is None:
        print("execution: skipped (no transport)")
        return outcome
    if safety.verdict == "blocked":
        print("execution: skipped (safety blocked)")
        return outcome
    if safety.verdict == "needs_confirmation":
        print("execution: skipped (needs_confirmation; re-run with --mode direct)")
        return outcome

    try:
        result = bridge.execute(command.houdini_python)
    except NotImplementedError as e:
        outcome.error = f"transport: {e}"
        print(f"execution: ERROR {e}")
        return outcome

    outcome.executed = True
    outcome.exec_success = result.success
    outcome.exec_return_code = result.return_code
    outcome.exec_duration_s = result.duration_seconds
    _print_exec(result)
    return outcome


def _print_summary(outcomes: list[CaseOutcome], do_execute: bool) -> None:
    print()
    print(_hr("═"))
    print("SUMMARY")
    print(_hr("─"))
    print(f"{'id':<22} {'intent_ok':<10} {'risk_ok':<8} {'safety':<18} {'exec':<8}")
    for o in outcomes:
        intent = "—" if o.intent_match is None else ("yes" if o.intent_match else "no")
        risk = "—" if o.risk_match is None else ("yes" if o.risk_match else "no")
        safety = o.safety_verdict or "—"
        if not do_execute:
            execcol = "—"
        elif o.error:
            execcol = "ERR"
        elif not o.executed:
            execcol = "skip"
        else:
            execcol = "ok" if o.exec_success else "FAIL"
        print(f"{o.id:<22} {intent:<10} {risk:<8} {safety:<18} {execcol:<8}")
    counts_intent = sum(1 for o in outcomes if o.intent_match)
    print(_hr("─"))
    print(
        f"plans matching expected intent: {counts_intent}/{len(outcomes)}"
        + (
            "" if not do_execute
            else f" | executed: {sum(1 for o in outcomes if o.executed)}"
        )
    )


def _emit_json(outcomes: list[CaseOutcome], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": [asdict(o) for o in outcomes],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


# ---- entrypoint --------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the manual Houdini prompt catalog against the configured "
            "LLM provider. Generates plans by default; pass --execute to "
            "run them through the Houdini bridge."
        )
    )
    parser.add_argument("--env-file", default=".env",
                        help="Path to .env (default: .env)")
    parser.add_argument("--provider",
                        help="Override DEFAULT_PROVIDER for this run.")
    parser.add_argument("--mode", choices=("dev", "safe", "direct"),
                        help="Override safety MODE for this run.")
    parser.add_argument("--execute", action="store_true",
                        help="Run generated code via the Houdini bridge.")
    parser.add_argument("--only", metavar="ID",
                        help="Run a single prompt by id.")
    parser.add_argument("--output", type=Path, metavar="PATH",
                        help="Write a JSON transcript of the run to PATH.")
    parser.add_argument("--list", action="store_true",
                        help="List prompt ids and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list:
        for case in PROMPTS:
            print(f"{case.id:<22} {case.prompt}")
        return 0

    cases: tuple[PromptCase, ...]
    if args.only:
        cases = tuple(c for c in PROMPTS if c.id == args.only)
        if not cases:
            print(f"unknown prompt id: {args.only}", file=sys.stderr)
            print("available ids:", ", ".join(c.id for c in PROMPTS),
                  file=sys.stderr)
            return 2
    else:
        cases = PROMPTS

    try:
        settings = Settings.load(env_file=args.env_file or None)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("manual")

    try:
        router = _build_router(settings, args.provider)
    except RouterError as e:
        print(f"router error: {e}", file=sys.stderr)
        return 2

    bridge = _try_build_bridge(settings) if args.execute else None
    mode = args.mode or settings.mode

    log.info(
        "starting manual run: provider=%s, mode=%s, execute=%s, cases=%d",
        router.active, mode, args.execute, len(cases),
    )

    outcomes: list[CaseOutcome] = []
    for case in cases:
        try:
            outcomes.append(_run_case(
                case, router, bridge, mode=mode, do_execute=args.execute,
            ))
        except KeyboardInterrupt:
            print("\ninterrupted; partial results below")
            break

    _print_summary(outcomes, do_execute=args.execute)

    if args.output:
        _emit_json(outcomes, args.output)
        print(f"\ntranscript written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
