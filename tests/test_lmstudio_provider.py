from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aibridge_houdini.config import Settings
from aibridge_houdini.providers.base import ProviderError
from aibridge_houdini.providers.lmstudio_provider import (
    DEFAULT_TIMEOUT_SECONDS,
    SYSTEM_INSTRUCTION,
    LMStudioProvider,
    _strip_code_fence,
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
        "PROVIDERS",
        "DEFAULT_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "LMSTUDIO_BASE_URL",
        "LMSTUDIO_MODEL",
        "LMSTUDIO_API_KEY",
        "MODE",
        "HOUDINI_HOST",
        "HOUDINI_PORT",
        "HYTHON_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(monkeypatch, **overrides) -> Settings:
    monkeypatch.setenv("DEFAULT_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder-7b")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return Settings.load(env_file=None)


def _fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_generate_calls_chat_completions_with_strict_shape(monkeypatch):
    settings = _settings(monkeypatch)
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(json.dumps(SPHERE_PLAN))

    provider = LMStudioProvider(settings, client=client, sleep=lambda _s: None)
    plan = provider.generate(UserRequest(text=SPHERE_PROMPT))

    client.chat.completions.create.assert_called_once()
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "qwen2.5-coder-7b"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS
    assert kwargs["messages"][0] == {"role": "system", "content": SYSTEM_INSTRUCTION}
    assert kwargs["messages"][1] == {"role": "user", "content": SPHERE_PROMPT}

    assert plan.intent == "create_geometry"
    assert plan.risk_level == "low"
    assert plan.requires_houdini is True
    assert "sphere" in plan.houdini_python.lower()


def test_does_not_execute_returned_code(monkeypatch):
    settings = _settings(monkeypatch)
    payload = dict(SPHERE_PLAN, houdini_python="raise RuntimeError('should not run')")
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(json.dumps(payload))

    provider = LMStudioProvider(settings, client=client, sleep=lambda _s: None)
    plan = provider.generate(UserRequest(text=SPHERE_PROMPT))

    assert "RuntimeError" in plan.houdini_python


def test_strips_markdown_code_fence(monkeypatch):
    settings = _settings(monkeypatch)
    fenced = "```json\n" + json.dumps(SPHERE_PLAN) + "\n```"
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(fenced)

    provider = LMStudioProvider(settings, client=client, sleep=lambda _s: None)
    plan = provider.generate(UserRequest(text=SPHERE_PROMPT))

    assert plan.intent == "create_geometry"


def test_strip_code_fence_helper():
    raw = json.dumps({"a": 1})
    assert _strip_code_fence(raw) == raw
    assert _strip_code_fence(f"```\n{raw}\n```") == raw
    assert _strip_code_fence(f"```json\n{raw}\n```") == raw
    assert _strip_code_fence(f"   ```json\n{raw}\n```   ") == raw


def test_custom_timeout_is_propagated(monkeypatch):
    settings = _settings(monkeypatch)
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(json.dumps(SPHERE_PLAN))

    provider = LMStudioProvider(
        settings, client=client, timeout=12.5, sleep=lambda _s: None
    )
    provider.generate(UserRequest(text=SPHERE_PROMPT))

    assert client.chat.completions.create.call_args.kwargs["timeout"] == 12.5


def test_retries_on_timeout_then_succeeds(monkeypatch):
    settings = _settings(monkeypatch)
    timeout_err = _make_openai_error("APITimeoutError")
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        timeout_err,
        timeout_err,
        _fake_response(json.dumps(SPHERE_PLAN)),
    ]
    sleeps: list[float] = []

    provider = LMStudioProvider(
        settings, client=client, max_retries=3, sleep=sleeps.append
    )
    plan = provider.generate(UserRequest(text=SPHERE_PROMPT))

    assert client.chat.completions.create.call_count == 3
    assert sleeps == [2, 4]
    assert plan.intent == "create_geometry"


def test_raises_after_max_retries(monkeypatch):
    settings = _settings(monkeypatch)
    err = _make_openai_error("APIConnectionError")
    client = MagicMock()
    client.chat.completions.create.side_effect = err

    provider = LMStudioProvider(
        settings, client=client, max_retries=2, sleep=lambda _s: None
    )
    with pytest.raises(ProviderError) as exc:
        provider.generate(UserRequest(text=SPHERE_PROMPT))
    assert "after 2 attempts" in str(exc.value)
    assert client.chat.completions.create.call_count == 2


def test_invalid_json_raises_provider_error(monkeypatch):
    settings = _settings(monkeypatch)
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response("this is not json")

    provider = LMStudioProvider(
        settings, client=client, max_retries=2, sleep=lambda _s: None
    )
    with pytest.raises(ProviderError):
        provider.generate(UserRequest(text=SPHERE_PROMPT))


def test_missing_field_raises(monkeypatch):
    settings = _settings(monkeypatch)
    bad = {k: v for k, v in SPHERE_PLAN.items() if k != "expected_result"}
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(json.dumps(bad))

    provider = LMStudioProvider(
        settings, client=client, max_retries=2, sleep=lambda _s: None
    )
    with pytest.raises(ProviderError):
        provider.generate(UserRequest(text=SPHERE_PROMPT))


def test_constructor_requires_model(monkeypatch):
    settings = _settings(monkeypatch)
    settings = settings.model_copy(update={"lmstudio_model": None})
    with pytest.raises(ProviderError):
        LMStudioProvider(settings, client=MagicMock())


def test_default_client_uses_base_url_and_api_key(monkeypatch):
    settings = _settings(
        monkeypatch,
        LMSTUDIO_BASE_URL="http://192.168.1.50:1234/v1",
        LMSTUDIO_API_KEY="my-secret-key",
    )
    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = MagicMock()
            self.chat.completions.create.return_value = _fake_response(
                json.dumps(SPHERE_PLAN)
            )

    monkeypatch.setattr("openai.OpenAI", _FakeClient)

    provider = LMStudioProvider(settings, timeout=7.0, sleep=lambda _s: None)
    provider.generate(UserRequest(text=SPHERE_PROMPT))

    assert captured["base_url"] == "http://192.168.1.50:1234/v1"
    assert captured["api_key"] == "my-secret-key"
    assert captured["timeout"] == 7.0


def test_factory_returns_lmstudio_provider(monkeypatch):
    from aibridge_houdini.providers import make_provider

    settings = _settings(monkeypatch)
    monkeypatch.setattr(
        "aibridge_houdini.providers.lmstudio_provider._build_default_client",
        lambda *_a, **_k: MagicMock(),
    )

    provider = make_provider(settings)
    assert provider.name == "lmstudio"
    assert isinstance(provider, LMStudioProvider)


def _make_openai_error(name: str) -> Exception:
    import openai  # type: ignore

    cls = getattr(openai, name)
    for args in ((), ("transient",)):
        try:
            return cls(*args)  # type: ignore[arg-type]
        except TypeError:
            continue
    return cls.__new__(cls)
