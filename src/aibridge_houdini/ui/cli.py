from __future__ import annotations

import json
from typing import Any


PROMPT = "ai-bridge> "
EXIT_WORDS = {"exit", "quit", ":q"}


def read_user_input() -> str | None:
    try:
        return input(PROMPT)
    except (EOFError, KeyboardInterrupt):
        return None


def render_response(obj: Any) -> str:
    return json.dumps(obj.model_dump(), indent=2, ensure_ascii=False)
