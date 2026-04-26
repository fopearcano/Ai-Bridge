from __future__ import annotations

import json

from aibridge_houdini.types import BridgeResponse


PROMPT = "ai-bridge> "
EXIT_WORDS = {"exit", "quit", ":q"}


def read_user_input() -> str | None:
    try:
        return input(PROMPT)
    except (EOFError, KeyboardInterrupt):
        return None


def render_response(response: BridgeResponse) -> str:
    return json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
