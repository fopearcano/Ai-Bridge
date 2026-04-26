from __future__ import annotations

import io
import json
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aibridge_houdini import app
from aibridge_houdini.config import Settings
from aibridge_houdini.providers import ProviderRouter, RouterError
from aibridge_houdini.providers.base import ProviderError
from aibridge_houdini.types import Command, LLMPlan, UserRequest


SPHERE_PROMPT = "Create a sphere in Houdini."


def _plan(intent: str, py: str = "import hou\n") -> LLMPlan:
    return LLMPlan(
        intent=intent,
        summary=f"Plan for {intent}",
        risk_level="low",
        requires_houdini=True,
        houdini_python=py,
        explanation="explained",
        expected_result="a sphere appears",
    )


class FakeProvider:
    """Minimal LLMProvider stand-in for router tests."""

    def __init__(self, name: str, *plans_or_exc):
        self.name = name
        self._queue = list(plans_or_exc)
        self.calls: list[UserRequest] = []

    def generate(self, request: UserRequest) -> LLMPlan:
        self.calls.append(request)
        if not self._queue:
            raise AssertionError(f"FakeProvider({self.name}) ran out of responses")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


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
        "AIBRIDGE_LOG_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def _all_providers_settings(monkeypatch, default: str = "openai") -> Settings:
    monkeypatch.setenv("PROVIDERS", "openai,anthropic,lmstudio")
    monkeypatch.setenv("DEFAULT_PROVIDER", default)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-abcdefgh12345678")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder-7b")
    return Settings.load(env_file=None)


# ---- core routing -------------------------------------------------------


def test_route_returns_normalized_command_from_active(monkeypatch):
    s = _all_providers_settings(monkeypatch, default="openai")
    fakes = {
        "openai": FakeProvider("openai", _plan("create_sphere_oai")),
        "anthropic": FakeProvider("anthropic", _plan("create_sphere_ant")),
        "lmstudio": FakeProvider("lmstudio", _plan("create_sphere_lms")),
    }
    router = ProviderRouter(s, providers=fakes)

    cmd = router.route(UserRequest(text=SPHERE_PROMPT))

    assert isinstance(cmd, Command)
    assert cmd.provider == "openai"
    assert cmd.request == SPHERE_PROMPT
    assert cmd.intent == "create_sphere_oai"
    assert cmd.risk_level == "low"
    assert cmd.requires_houdini is True
    # only the active fake was called
    assert len(fakes["openai"].calls) == 1
    assert fakes["anthropic"].calls == []
    assert fakes["lmstudio"].calls == []


def test_set_provider_switches_active_and_routes(monkeypatch):
    s = _all_providers_settings(monkeypatch, default="openai")
    fakes = {
        "openai": FakeProvider("openai", _plan("oai-1")),
        "anthropic": FakeProvider("anthropic", _plan("ant-1")),
        "lmstudio": FakeProvider("lmstudio", _plan("lms-1")),
    }
    router = ProviderRouter(s, providers=fakes)

    router.set_provider("anthropic")
    assert router.active == "anthropic"
    cmd = router.route(UserRequest(text=SPHERE_PROMPT))
    assert cmd.provider == "anthropic"
    assert cmd.intent == "ant-1"

    router.set_provider("lmstudio")
    cmd = router.route(UserRequest(text=SPHERE_PROMPT))
    assert cmd.provider == "lmstudio"
    assert cmd.intent == "lms-1"


def test_set_provider_normalizes_case(monkeypatch):
    s = _all_providers_settings(monkeypatch)
    router = ProviderRouter(s, providers={"openai": FakeProvider("openai", _plan("x"))})
    router.set_provider("OpenAI")
    assert router.active == "openai"


def test_set_provider_rejects_unknown(monkeypatch):
    s = _all_providers_settings(monkeypatch)
    router = ProviderRouter(s, providers={"openai": FakeProvider("openai", _plan("x"))})
    with pytest.raises(RouterError) as exc:
        router.set_provider("groq")
    assert "groq" in str(exc.value)
    assert "PROVIDERS" in str(exc.value)


def test_set_provider_rejects_disabled(monkeypatch):
    monkeypatch.setenv("PROVIDERS", "openai,anthropic")
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-abcdefgh12345678")
    s = Settings.load(env_file=None)
    router = ProviderRouter(s, providers={"openai": FakeProvider("openai", _plan("x"))})
    with pytest.raises(RouterError):
        router.set_provider("lmstudio")


def test_enabled_lists_providers_from_config(monkeypatch):
    s = _all_providers_settings(monkeypatch)
    router = ProviderRouter(s, providers={"openai": FakeProvider("openai", _plan("x"))})
    assert router.enabled == ["openai", "anthropic", "lmstudio"]


# ---- auto-repair --------------------------------------------------------


def test_route_auto_repairs_after_one_failure(monkeypatch):
    s = _all_providers_settings(monkeypatch, default="openai")
    fake = FakeProvider(
        "openai",
        ProviderError("invalid JSON"),
        _plan("create_sphere_repaired"),
    )
    router = ProviderRouter(s, providers={"openai": fake})

    cmd = router.route(UserRequest(text=SPHERE_PROMPT))

    assert cmd.intent == "create_sphere_repaired"
    assert len(fake.calls) == 2
    # Original prompt preserved on the Command.
    assert cmd.request == SPHERE_PROMPT
    # Second call carried the repair hint appended to the original prompt.
    assert fake.calls[1].text.startswith(SPHERE_PROMPT)
    assert "JSON" in fake.calls[1].text
    assert "no markdown" in fake.calls[1].text.lower()


def test_route_raises_router_error_after_repair_fails(monkeypatch):
    s = _all_providers_settings(monkeypatch, default="anthropic")
    fake = FakeProvider(
        "anthropic",
        ProviderError("boom-1"),
        ProviderError("boom-2"),
    )
    router = ProviderRouter(s, providers={"anthropic": fake})

    with pytest.raises(RouterError) as exc:
        router.route(UserRequest(text=SPHERE_PROMPT))
    assert "anthropic" in str(exc.value)
    assert "auto-repair" in str(exc.value)
    assert len(fake.calls) == 2


# ---- end-to-end through real provider classes ---------------------------


def _openai_responses_payload(plan: dict) -> SimpleNamespace:
    return SimpleNamespace(output_text=json.dumps(plan), output=[])


def _lmstudio_chat_payload(plan: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(plan)))]
    )


SPHERE_JSON = {
    "intent": "create_geometry",
    "summary": "Create a sphere SOP at /obj/geo1.",
    "risk_level": "low",
    "requires_houdini": True,
    "houdini_python": "import hou\nhou.node('/obj').createNode('geo')\n",
    "explanation": "Adds a geo container with a sphere SOP.",
    "expected_result": "A new /obj/geo1/sphere1 node.",
}


def test_router_with_real_openai_provider_mocked(monkeypatch):
    from aibridge_houdini.providers.openai_provider import OpenAIProvider

    s = _all_providers_settings(monkeypatch, default="openai")
    fake_client = MagicMock()
    fake_client.responses.create.return_value = _openai_responses_payload(SPHERE_JSON)
    real_openai = OpenAIProvider(s, client=fake_client, sleep=lambda _: None)

    router = ProviderRouter(s, providers={"openai": real_openai})
    cmd = router.route(UserRequest(text=SPHERE_PROMPT))

    assert cmd.provider == "openai"
    assert cmd.intent == "create_geometry"
    fake_client.responses.create.assert_called_once()


def test_router_with_real_lmstudio_provider_mocked(monkeypatch):
    from aibridge_houdini.providers.lmstudio_provider import LMStudioProvider

    s = _all_providers_settings(monkeypatch, default="lmstudio")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _lmstudio_chat_payload(SPHERE_JSON)
    real_lmstudio = LMStudioProvider(s, client=fake_client, sleep=lambda _: None)

    router = ProviderRouter(s, providers={"lmstudio": real_lmstudio})
    cmd = router.route(UserRequest(text=SPHERE_PROMPT))

    assert cmd.provider == "lmstudio"
    assert cmd.intent == "create_geometry"
    fake_client.chat.completions.create.assert_called_once()


def test_router_anthropic_falls_back_to_placeholder_via_factory(monkeypatch):
    """No real Anthropic adapter yet — factory returns the placeholder."""
    s = _all_providers_settings(monkeypatch, default="anthropic")
    router = ProviderRouter(s)  # no injected providers

    cmd = router.route(UserRequest(text=SPHERE_PROMPT))

    # PlaceholderProvider keeps name="placeholder" for now.
    assert cmd.provider == "placeholder"
    assert cmd.request == SPHERE_PROMPT
    assert cmd.requires_houdini is False  # placeholder default


def test_router_lazily_constructs_providers_on_switch(monkeypatch):
    """Switching to a provider that isn't pre-cached must build it lazily."""
    s = _all_providers_settings(monkeypatch, default="openai")
    fake_openai = FakeProvider("openai", _plan("oai"))
    # Only OpenAI is pre-cached. Switching to anthropic should pull from factory.
    router = ProviderRouter(s, providers={"openai": fake_openai})

    router.set_provider("anthropic")
    cmd = router.route(UserRequest(text=SPHERE_PROMPT))
    assert cmd.provider == "placeholder"  # factory's anthropic fallback


# ---- /provider slash command --------------------------------------------


def test_slash_provider_switches_via_app_handler(monkeypatch, capsys):
    s = _all_providers_settings(monkeypatch, default="openai")
    router = ProviderRouter(
        s,
        providers={
            "openai": FakeProvider("openai", _plan("o")),
            "anthropic": FakeProvider("anthropic", _plan("a")),
            "lmstudio": FakeProvider("lmstudio", _plan("l")),
        },
    )

    app.handle_slash_command("/provider lmstudio", router)
    out = capsys.readouterr().out
    assert "switched to: lmstudio" in out
    assert router.active == "lmstudio"


def test_slash_provider_no_arg_shows_state(monkeypatch, capsys):
    s = _all_providers_settings(monkeypatch, default="anthropic")
    router = ProviderRouter(
        s, providers={"anthropic": FakeProvider("anthropic", _plan("a"))}
    )

    app.handle_slash_command("/provider", router)
    out = capsys.readouterr().out
    assert "active: anthropic" in out
    assert "openai" in out and "lmstudio" in out


def test_slash_provider_unknown_prints_error(monkeypatch, capsys):
    s = _all_providers_settings(monkeypatch)
    router = ProviderRouter(
        s, providers={"openai": FakeProvider("openai", _plan("x"))}
    )

    app.handle_slash_command("/provider groq", router)
    err = capsys.readouterr().err
    assert "error" in err
    assert "groq" in err
    assert router.active == "openai"  # unchanged


def test_slash_help_prints_command_list(monkeypatch, capsys):
    s = _all_providers_settings(monkeypatch)
    router = ProviderRouter(
        s, providers={"openai": FakeProvider("openai", _plan("x"))}
    )

    app.handle_slash_command("/help", router)
    out = capsys.readouterr().out
    assert "/provider" in out
    assert "openai" in out and "anthropic" in out and "lmstudio" in out


def test_slash_unknown_command_reports_error(monkeypatch, capsys):
    s = _all_providers_settings(monkeypatch)
    router = ProviderRouter(
        s, providers={"openai": FakeProvider("openai", _plan("x"))}
    )

    app.handle_slash_command("/whatever", router)
    err = capsys.readouterr().err
    assert "/whatever" in err


def test_main_loop_supports_slash_provider(monkeypatch, tmp_path):
    """End-to-end through main(): /provider switches then a request is routed."""
    monkeypatch.setenv("PROVIDERS", "openai,anthropic,lmstudio")
    monkeypatch.setenv("DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-abcdefgh12345678")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder-7b")
    monkeypatch.setenv("AIBRIDGE_LOG_DIR", str(tmp_path))

    inputs = iter(["/provider lmstudio", SPHERE_PROMPT, "exit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))

    # Patch the LM Studio adapter's HTTP client builder so no network is touched.
    from aibridge_houdini.providers import lmstudio_provider as lm_mod

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _lmstudio_chat_payload(SPHERE_JSON)
    monkeypatch.setattr(lm_mod, "_build_default_client", lambda *_a, **_k: fake_client)

    out_buf, err_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        rc = app.main(["--env-file", str(tmp_path / "missing.env")])

    assert rc == 0
    out = out_buf.getvalue()
    assert "switched to: lmstudio" in out
    assert '"provider": "lmstudio"' in out
    assert SPHERE_PROMPT in out  # echoed back via Command.request
    fake_client.chat.completions.create.assert_called_once()
