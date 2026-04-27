from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import get_args

from aibridge_houdini import __version__
from aibridge_houdini.config import ConfigError, Mode, Settings
from aibridge_houdini.execution.safety import SafetyReport, evaluate
from aibridge_houdini.houdini.bridge import HoudiniBridge
from aibridge_houdini.houdini.transport import ExecutionResult, TransportError
from aibridge_houdini.logging_setup import setup_logging
from aibridge_houdini.providers import ProviderRouter, RouterError
from aibridge_houdini.types import Command, UserRequest
from aibridge_houdini.ui.cli import (
    EXIT_WORDS,
    confirm_execute,
    read_user_input,
    render_turn,
)


VALID_MODES: tuple[Mode, ...] = get_args(Mode)


@dataclass
class Session:
    """Runtime state — mutable mode lives here so /mode can flip it."""

    settings: Settings
    router: ProviderRouter
    bridge: HoudiniBridge | None
    log: logging.Logger
    mode: Mode = field(init=False)

    def __post_init__(self) -> None:
        self.mode = self.settings.mode


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aibridge-houdini",
        description="Bridge: natural language -> Houdini Python.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--env-file", default=".env", help="Path to .env file (default: .env)"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration, print a sanitized summary, and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        settings = Settings.load(env_file=args.env_file)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    log = setup_logging(settings.log_level, settings.log_dir)
    log.info(
        "aibridge-houdini %s starting (mode=%s, provider=%s)",
        __version__,
        settings.mode,
        settings.default_provider,
    )
    log.debug("settings: %s", settings.safe_summary())

    if args.check_config:
        for k, v in settings.safe_summary().items():
            print(f"{k}: {v}")
        return 0

    try:
        router = ProviderRouter(settings)
    except RouterError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    bridge = _try_build_bridge(settings, log)
    session = Session(settings=settings, router=router, bridge=bridge, log=log)

    bridge_note = (
        "transport: hython"
        if bridge is not None
        else "transport: NONE (set HYTHON_PATH to execute)"
    )
    print(
        f"Ai-Bridge Houdini v{__version__} "
        f"(active: {router.active}, mode: {session.mode}, {bridge_note}). "
        "Type '/help' for commands, '/quit' to leave."
    )

    while True:
        text = read_user_input()
        if text is None:
            print()
            break
        text = text.strip()
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            break

        if text.startswith("/"):
            if handle_slash_command(text, session):
                break
            continue

        process_request(text, session)

    log.info("shutting down")
    return 0


def process_request(text: str, session: Session) -> None:
    request = UserRequest(text=text)
    try:
        command = session.router.route(request)
    except RouterError as e:
        session.log.error("router failed: %s", e)
        print(f"error: {e}", file=sys.stderr)
        return
    except Exception:
        session.log.exception("unexpected router error")
        print("error: unexpected router error (see log)", file=sys.stderr)
        return

    safety = evaluate(command.houdini_python, mode=session.mode)
    exec_result, declined = _gate_and_execute(command, safety, session)

    session.log.debug(
        "turn: provider=%s verdict=%s executed=%s",
        command.provider,
        safety.verdict,
        exec_result is not None,
    )
    print(
        render_turn(
            command,
            safety,
            exec_result,
            declined=declined,
            bridge_available=session.bridge is not None,
        )
    )


def _gate_and_execute(
    command: Command,
    safety: SafetyReport,
    session: Session,
) -> tuple[ExecutionResult | None, bool]:
    if safety.verdict == "blocked":
        session.log.warning(
            "safety blocked code from %s (%d findings)",
            command.provider,
            len(safety.findings),
        )
        return None, False

    if safety.verdict == "needs_confirmation":
        if not confirm_execute():
            session.log.info("user declined execution")
            return None, True

    if session.bridge is None:
        return None, False

    try:
        result = session.bridge.execute(command.houdini_python)
    except NotImplementedError as e:
        session.log.error("transport not implemented: %s", e)
        return ExecutionResult(
            transport=session.bridge.transport.name,
            success=False,
            error=str(e),
        ), False
    except Exception as e:
        session.log.exception("transport failed unexpectedly")
        return ExecutionResult(
            transport=session.bridge.transport.name,
            success=False,
            error=str(e),
        ), False
    return result, False


def handle_slash_command(text: str, session: Session) -> bool:
    """Returns True iff the loop should exit (e.g. /quit)."""
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/provider":
        _handle_provider(arg, session)
        return False

    if cmd == "/mode":
        _handle_mode(arg, session)
        return False

    if cmd in ("/quit", "/exit"):
        return True

    if cmd in ("/help", "/?"):
        _print_help()
        return False

    print(f"unknown command: {cmd} (try /help)", file=sys.stderr)
    return False


def _handle_provider(arg: str, session: Session) -> None:
    if not arg:
        print(f"active: {session.router.active}")
        print(f"enabled: {', '.join(session.router.enabled)}")
        return
    try:
        session.router.set_provider(arg)
    except RouterError as e:
        print(f"error: {e}", file=sys.stderr)
        return
    print(f"switched to: {session.router.active}")


def _handle_mode(arg: str, session: Session) -> None:
    if not arg:
        print(f"mode: {session.mode}")
        print(f"allowed: {', '.join(VALID_MODES)}")
        return
    new_mode = arg.lower()
    if new_mode not in VALID_MODES:
        print(
            f"error: unknown mode '{arg}'. allowed: {', '.join(VALID_MODES)}",
            file=sys.stderr,
        )
        return
    session.mode = new_mode  # type: ignore[assignment]
    session.log.info("mode set to %s", new_mode)
    print(f"mode: {session.mode}")


def _print_help() -> None:
    print("commands:")
    print("  /provider                       show active and enabled providers")
    print("  /provider <openai|anthropic|lmstudio>   switch active provider")
    print("  /mode                           show current safety mode")
    print("  /mode <safe|dev|direct>         change safety mode")
    print("  /help                           show this list")
    print("  /quit                           leave (alias: /exit, exit, quit, :q)")


def _try_build_bridge(settings: Settings, log: logging.Logger) -> HoudiniBridge | None:
    try:
        return HoudiniBridge(settings, transport="hython")
    except TransportError as e:
        log.warning(
            "Houdini bridge unavailable: %s — code will be displayed but not executed",
            e,
        )
        return None
