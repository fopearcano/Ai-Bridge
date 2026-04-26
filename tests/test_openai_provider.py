from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aibridge_houdini.config import Settings
from aibridge_houdini.providers.base import ProviderError
from aibridge_houdini.providers.openai_provider import (
    JSON_SCHEMA,
    SYSTEM_INSTRUCTION,
    OpenAIProvider,
)
from aibridge_houdini.types import UserRequest


SPHERE_PROMPT = "Create a sphere in Houdini."

SPHERE_PLAN = {
    "intent": "create_geometry",
    "summary": "Create a sphere SOP at /obj/geo1.",
    "risk_level": "low",
    "requires_houdini": True,
    "houdini_python": (
        "import hou\n"
        "geo = hou.node('/obj').createNode('geo', 'geo1')\n"
        "geo.createNode('sphere', 'sphere1')\n"
    ),
    "explanation": "Adds a geo container with a sphere SOP inside.",
    "expected_result": "A new /obj/geo1/sphere1 node visible in the network.",
}


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in (
        "DEFAULT_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "MODE",
        "HOUDINI_HOST",
        "HOUDINI_PORT",
        "HYTHON_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-testkey-1234567890")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    return Settings.load(env_file=None)


def _fake_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(output_text=json.dumps(payload), output=[])


def test_generate_calls_responses_api_with_strict_schema(monkeypatch):
    settings = _settings(monkeypatch)
    client = MagicMock()
    client.responses.create.return_value = _fake_response(SPHERE_PLAN)

    provider = OpenAIProvider(settings, client=client, sleep=lambda _s: None)
    plan = provider.generate(UserRequest(text=SPHERE_PROMPT))

    client.responses.create.assert_called_once()
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["instructions"] == SYSTEM_INSTRUCTION
    assert kwargs["input"] == SPHERE_PROMPT
    fmt = kwargs["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["schema"] == JSON_SCHEMA

    assert plan.intent == "create_geometry"
    assert plan.risk_level == "low"
    assert plan.requires_houdini is True
    assert "sphere" in plan.houdini_python.lower()


def test_generate_does_not_execute_returned_code(monkeypatch):
    """Sanity: provider must only return the plan, never run the code."""
    settings = _settings(monkeypatch)
    payload = dict(SPHERE_PLAN, houdini_python="raise RuntimeError('should not run')")
    client = MagicMock()
    client.responses.create.return_value = _fake_response(payload)

    provider = OpenAIProvider(settings, client=client, sleep=lambda _s: None)
    plan = provider.generate(UserRequest(text=SPHERE_PROMPT))

    assert "RuntimeError" in plan.houdini_python  # we got it back as text only


def test_retries_on_rate_limit_then_succeeds(monkeypatch):
    settings = _settings(monkeypatch)
    rate_err = _make_openai_error("RateLimitError")
    client = MagicMock()
    client.responses.create.side_effect = [rate_err, rate_err, _fake_response(SPHERE_PLAN)]
    sleeps: list[float] = []

    provider = OpenAIProvider(settings, client=client, max_retries=3, sleep=sleeps.append)
    plan = provider.generate(UserRequest(text=SPHERE_PROMPT))

    assert client.responses.create.call_count == 3
    assert sleeps == [2, 4]  # backoff before retries 2 and 3
    assert plan.intent == "create_geometry"


def test_raises_after_max_retries(monkeypatch):
    settings = _settings(monkeypatch)
    err = _make_openai_error("APIConnectionError")
    client = MagicMock()
    client.responses.create.side_effect = err

    provider = OpenAIProvider(settings, client=client, max_retries=2, sleep=lambda _s: None)
    with pytest.raises(ProviderError) as exc:
        provider.generate(UserRequest(text=SPHERE_PROMPT))

    assert "after 2 attempts" in str(exc.value)
    assert client.responses.create.call_count == 2


def test_invalid_json_payload_raises_provider_error(monkeypatch):
    settings = _settings(monkeypatch)
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="not json at all", output=[]
    )

    provider = OpenAIProvider(settings, client=client, max_retries=2, sleep=lambda _s: None)
    with pytest.raises(ProviderError):
        provider.generate(UserRequest(text=SPHERE_PROMPT))


def test_missing_field_in_payload_raises(monkeypatch):
    settings = _settings(monkeypatch)
    bad = {k: v for k, v in SPHERE_PLAN.items() if k != "expected_result"}
    client = MagicMock()
    client.responses.create.return_value = _fake_response(bad)

    provider = OpenAIProvider(settings, client=client, max_retries=2, sleep=lambda _s: None)
    with pytest.raises(ProviderError):
        provider.generate(UserRequest(text=SPHERE_PROMPT))


def test_constructor_requires_openai_key(monkeypatch):
    monkeypatch.setenv("DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    settings = Settings.load(env_file=None)
    # Swap to openai-style settings without an OpenAI key
    settings = settings.model_copy(update={"default_provider": "openai", "openai_api_key": None})
    with pytest.raises(ProviderError):
        OpenAIProvider(settings, client=MagicMock())


def _make_openai_error(name: str) -> Exception:
    """Instantiate a real openai exception class for retry testing."""
    import openai  # type: ignore

    cls = getattr(openai, name)
    # Different SDK versions accept different signatures; try the safest forms.
    for args in ((), ("transient", ), ):
        try:
            return cls(*args)  # type: ignore[arg-type]
        except TypeError:
            continue
    # Last resort: bypass __init__
    return cls.__new__(cls)
