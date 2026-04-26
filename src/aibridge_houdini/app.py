from __future__ import annotations

import argparse
import logging

from aibridge_houdini import __version__
from aibridge_houdini.config import Settings
from aibridge_houdini.logging_setup import setup_logging
from aibridge_houdini.providers.placeholder import PlaceholderProvider
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings.load(env_file=args.env_file)
    log = setup_logging(settings.log_level, settings.log_dir)
    log.info("aibridge-houdini %s starting (provider=%s)", __version__, settings.provider)

    provider = PlaceholderProvider()
    print(f"Ai-Bridge Houdini v{__version__} (skeleton). Type 'exit' to quit.")

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

        try:
            response = provider.generate(UserRequest(text=text))
        except Exception:
            log.exception("provider failed")
            continue

        log.debug("response: %s", response.model_dump())
        print(render_response(response))

    log.info("shutting down")
    return 0
