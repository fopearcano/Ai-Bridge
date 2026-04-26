from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from aibridge_houdini import app
from aibridge_houdini.providers.placeholder import PlaceholderProvider
from aibridge_houdini.types import UserRequest


def test_placeholder_provider_returns_structured_response():
    provider = PlaceholderProvider()
    response = provider.generate(UserRequest(text="make a pyro sim"))
    assert response.request.text == "make a pyro sim"
    assert len(response.actions) == 1
    assert response.executed is False
    assert response.actions[0].intent == "unknown"


def test_main_loop_handles_exit(monkeypatch, tmp_path):
    inputs = iter(["", "create a sphere", "exit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
    monkeypatch.setenv("AIBRIDGE_LOG_DIR", str(tmp_path))

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = app.main(["--env-file", str(tmp_path / "missing.env")])

    assert rc == 0
    out = buf.getvalue()
    assert "Ai-Bridge Houdini" in out
    assert "create a sphere" in out
