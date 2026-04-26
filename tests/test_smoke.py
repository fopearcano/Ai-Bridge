from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from aibridge_houdini import app
from aibridge_houdini.providers.placeholder import PlaceholderProvider
from aibridge_houdini.types import UserRequest


@pytest.fixture
def env_with_anthropic(monkeypatch, tmp_path):
    """Minimum env that satisfies validation for the default provider."""
    monkeypatch.setenv("DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic-1234567890")
    monkeypatch.setenv("AIBRIDGE_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


def test_placeholder_provider_returns_structured_response():
    provider = PlaceholderProvider()
    response = provider.generate(UserRequest(text="make a pyro sim"))
    assert response.request.text == "make a pyro sim"
    assert len(response.actions) == 1
    assert response.executed is False
    assert response.actions[0].intent == "unknown"


def test_main_loop_handles_exit(monkeypatch, env_with_anthropic):
    inputs = iter(["", "create a sphere", "exit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = app.main(["--env-file", str(env_with_anthropic / "missing.env")])

    assert rc == 0
    out = buf.getvalue()
    assert "Ai-Bridge Houdini" in out
    assert "create a sphere" in out
