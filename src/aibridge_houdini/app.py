from __future__ import annotations

import argparse
import sys

from aibridge_houdini import __version__
from aibridge_houdini.config import ConfigError, Settings
from aibridge_houdini.logging_setup import setup_logging
from aibridge_houdini.providers import ProviderError, make_provider
from aibridge_houdini.types import BridgeResponse, UserRequest
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
        provider = make_provider(settings)
    except ProviderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(
        f"Ai-Bridge Houdini v{__version__} ({provider.name}). Type 'exit' to quit."
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

        request = UserRequest(text=text)
        try:
            plan = provider.generate(request)
        except ProviderError as e:
            log.error("provider failed: %s", e)
            continue
        except Exception:
            log.exception("unexpected provider error")
            continue

        response = BridgeResponse(request=request, plan=plan, executed=False)
        log.debug("response: %s", response.model_dump())
        print(render_response(response))

    log.info("shutting down")
    return 0
