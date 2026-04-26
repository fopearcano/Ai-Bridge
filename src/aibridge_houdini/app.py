from __future__ import annotations

import argparse
import sys

from aibridge_houdini import __version__
from aibridge_houdini.config import ConfigError, Settings
from aibridge_houdini.logging_setup import setup_logging
from aibridge_houdini.providers import ProviderRouter, RouterError
from aibridge_houdini.types import UserRequest
from aibridge_houdini.ui.cli import EXIT_WORDS, read_user_input, render_response


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aibridge-houdini",
        description="Bridge: natural language -> Houdini Python (skeleton).",
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

    print(
        f"Ai-Bridge Houdini v{__version__} (active: {router.active}). "
        "Type 'exit' to quit, '/help' for commands."
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
            handle_slash_command(text, router)
            continue

        request = UserRequest(text=text)
        try:
            command = router.route(request)
        except RouterError as e:
            log.error("router failed: %s", e)
            print(f"error: {e}", file=sys.stderr)
            continue
        except Exception:
            log.exception("unexpected router error")
            continue

        log.debug("command: %s", command.model_dump())
        print(render_response(command))

    log.info("shutting down")
    return 0


def handle_slash_command(text: str, router: ProviderRouter) -> None:
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/provider":
        if not arg:
            print(f"active: {router.active}")
            print(f"enabled: {', '.join(router.enabled)}")
            return
        try:
            router.set_provider(arg)
        except RouterError as e:
            print(f"error: {e}", file=sys.stderr)
            return
        print(f"switched to: {router.active}")
        return

    if cmd in ("/help", "/?"):
        print("commands:")
        print("  /provider                 show active and enabled providers")
        print("  /provider <openai|anthropic|lmstudio>")
        print("                            switch the active provider")
        print("  /help                     show this list")
        print("  exit | quit | :q          leave")
        return

    print(f"unknown command: {cmd} (try /help)", file=sys.stderr)
