from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock

import pytest

from aibridge_houdini import app
from aibridge_houdini.config import Settings
from aibridge_houdini.houdini.transport import ExecutionResult
from aibridge_houdini.providers import ProviderRouter
from aibridge_houdini.providers.base import ProviderError
from aibridge_houdini.types import LLMPlan, UserRequest


# ---- helpers ------------------------------------------------------------


SAFE_PLAN = LLMPlan(
    intent="create_geometry",
    summary="Create a sphere SOP at /obj/geo1.",
    risk_level="low",
    requires_houdini=True,
    houdini_python=(
        "import hou\n"
        "geo = hou.node('/obj').createNode('geo', 'geo1')\n"
        "geo.createNode('sphere', 'sphere1')\n"
    ),
    explanation="Adds a geo container with a sphere SOP.",
    expected_result="/obj/geo1/sphere1 visible.",
)

WARN_PLAN = SAFE_PLAN.model_copy(
    update={
        "summary": "Render to /tmp/render.exr",
        "houdini_python": "open('/tmp/render.exr', 'w').close()\n",
    }
)

BLOCKED_PLAN = SAFE_PLAN.model_copy(
    update={
        "summary": "Wipe /tmp/render",
        "houdini_python": "import shutil\nshutil.rmtree('/tmp/render')\n",
    }
)


class FakeProvider:
    def __init__(self, name: str, plan: LLMPlan):
        self.name = name
        self._plan = plan
        self.calls: list[UserRequest] = []

    def generate(self, request: UserRequest) -> LLMPlan:
        self.calls.append(request)
        return self._plan


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch, tmp_path):
    for name in (
        "PROVIDERS", "DEFAULT_PROVIDER", "MODE",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "LMSTUDIO_MODEL", "HYTHON_PATH",
        "AIBRIDGE_LOG_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("AIBRIDGE_LOG_DIR", str(tmp_path))


def _settings(monkeypatch, **extra) -> Settings:
    for k, v in extra.items():
        monkeypatch.setenv(k, v)
    return Settings.load(env_file=None)


def _session(monkeypatch, *, plan: LLMPlan = SAFE_PLAN, bridge=None, **env) -> app.Session:
    s = _settings(monkeypatch, **env)
    router = ProviderRouter(s, providers={s.default_provider: FakeProvider(s.default_provider, plan)})
    return app.Session(
        settings=s,
        router=router,
        bridge=bridge,
        log=logging.getLogger("aibridge.test.main_loop"),
    )


# ---- /mode --------------------------------------------------------------


def test_mode_no_arg_shows_current(monkeypatch, capsys):
    sess = _session(monkeypatch)
    app.handle_slash_command("/mode", sess)
    out = capsys.readouterr().out
    assert "mode: safe" in out
    assert "allowed" in out
    assert all(m in out for m in ("safe", "dev", "direct"))


def test_mode_switches(monkeypatch, capsys):
    sess = _session(monkeypatch)
    assert sess.mode == "safe"
    app.handle_slash_command("/mode direct", sess)
    out = capsys.readouterr().out
    assert sess.mode == "direct"
    assert "mode: direct" in out


def test_mode_normalizes_case(monkeypatch):
    sess = _session(monkeypatch)
    app.handle_slash_command("/mode DEV", sess)
    assert sess.mode == "dev"


def test_mode_unknown_value_rejected(monkeypatch, capsys):
    sess = _session(monkeypatch)
    app.handle_slash_command("/mode yolo", sess)
    err = capsys.readouterr().err
    assert "yolo" in err
    assert sess.mode == "safe"


# ---- /quit --------------------------------------------------------------


def test_quit_returns_true_to_break_loop(monkeypatch):
    sess = _session(monkeypatch)
    assert app.handle_slash_command("/quit", sess) is True


def test_exit_alias_returns_true(monkeypatch):
    sess = _session(monkeypatch)
    assert app.handle_slash_command("/exit", sess) is True


def test_other_commands_do_not_exit(monkeypatch):
    sess = _session(monkeypatch)
    assert app.handle_slash_command("/help", sess) is False
    assert app.handle_slash_command("/mode dev", sess) is False
    assert app.handle_slash_command("/provider", sess) is False


# ---- safety gating + render --------------------------------------------


def test_blocked_plan_is_not_executed_and_renders_blocked(monkeypatch, capsys):
    fake_bridge = MagicMock()
    fake_bridge.transport.name = "hython"
    sess = _session(monkeypatch, plan=BLOCKED_PLAN, bridge=fake_bridge)

    app.process_request("delete render dir", sess)

    out = capsys.readouterr().out
    fake_bridge.execute.assert_not_called()
    assert "provider:" in out and sess.router.active in out
    assert BLOCKED_PLAN.summary in out
    assert "shutil.rmtree" in out
    assert "safety:   blocked" in out
    assert "BLOCKED" in out


def test_warn_plan_in_safe_mode_prompts_and_executes_on_yes(monkeypatch, capsys):
    fake_bridge = MagicMock()
    fake_bridge.transport.name = "hython"
    fake_bridge.execute.return_value = ExecutionResult(
        transport="hython", success=True, return_code=0,
        stdout="ok\n", stderr="", duration_seconds=0.05,
    )
    sess = _session(monkeypatch, plan=WARN_PLAN, bridge=fake_bridge)

    monkeypatch.setattr("builtins.input", lambda *_: "y")
    app.process_request("save render", sess)

    out = capsys.readouterr().out
    fake_bridge.execute.assert_called_once_with(WARN_PLAN.houdini_python)
    assert "needs_confirmation" in out
    assert "ok (rc=0, 0.05s)" in out
    assert "stdout:" in out


def test_warn_plan_in_safe_mode_declined_renders_declined(monkeypatch, capsys):
    fake_bridge = MagicMock()
    fake_bridge.transport.name = "hython"
    sess = _session(monkeypatch, plan=WARN_PLAN, bridge=fake_bridge)

    monkeypatch.setattr("builtins.input", lambda *_: "n")
    app.process_request("save render", sess)

    out = capsys.readouterr().out
    fake_bridge.execute.assert_not_called()
    assert "DECLINED" in out


def test_warn_plan_in_direct_mode_executes_without_prompt(monkeypatch, capsys):
    fake_bridge = MagicMock()
    fake_bridge.transport.name = "hython"
    fake_bridge.execute.return_value = ExecutionResult(
        transport="hython", success=True, return_code=0, duration_seconds=0.01,
    )
    sess = _session(monkeypatch, plan=WARN_PLAN, bridge=fake_bridge)
    sess.mode = "direct"

    # If the loop tried to prompt, this would raise StopIteration.
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("should not prompt"))
    app.process_request("save render", sess)

    fake_bridge.execute.assert_called_once_with(WARN_PLAN.houdini_python)
    out = capsys.readouterr().out
    assert "safety:   safe (mode=direct)" in out


def test_safe_plan_executes_directly_in_safe_mode(monkeypatch, capsys):
    fake_bridge = MagicMock()
    fake_bridge.transport.name = "hython"
    fake_bridge.execute.return_value = ExecutionResult(
        transport="hython", success=True, return_code=0,
        stdout="created\n", stderr="", duration_seconds=0.42,
    )
    sess = _session(monkeypatch, plan=SAFE_PLAN, bridge=fake_bridge)

    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("should not prompt"))
    app.process_request("Create a sphere in Houdini.", sess)

    fake_bridge.execute.assert_called_once_with(SAFE_PLAN.houdini_python)
    out = capsys.readouterr().out
    assert f"provider: {sess.router.active}" in out
    assert SAFE_PLAN.summary in out
    assert "import hou" in out
    assert "ok (rc=0, 0.42s)" in out


def test_no_bridge_renders_not_executed(monkeypatch, capsys):
    sess = _session(monkeypatch, plan=SAFE_PLAN, bridge=None)
    app.process_request("Create a sphere in Houdini.", sess)
    out = capsys.readouterr().out
    assert "NOT EXECUTED" in out
    assert "HYTHON_PATH" in out


def test_router_failure_reported(monkeypatch, capsys):
    s = _settings(monkeypatch)
    failing = MagicMock()
    failing.name = "openai"
    failing.generate.side_effect = ProviderError("model down")
    router = ProviderRouter(s, providers={s.default_provider: failing})
    sess = app.Session(
        settings=s,
        router=router,
        bridge=None,
        log=logging.getLogger("aibridge.test"),
    )

    app.process_request("anything", sess)
    err = capsys.readouterr().err
    # Two attempts (auto-repair) then the router gives up.
    assert "auto-repair" in err.lower() or "after auto-repair" in err.lower()


# ---- end-to-end main() with bridge mocked at construction --------------


def test_main_full_flow_with_mode_switch_and_execution(monkeypatch, tmp_path):
    """User: /mode direct -> request -> /quit. Bridge executes once."""
    monkeypatch.setenv("PROVIDERS", "openai,anthropic,lmstudio")
    monkeypatch.setenv("DEFAULT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("AIBRIDGE_LOG_DIR", str(tmp_path))

    # Patch bridge construction so no real hython is needed.
    fake_bridge = MagicMock()
    fake_bridge.transport.name = "hython"
    fake_bridge.execute.return_value = ExecutionResult(
        transport="hython", success=True, return_code=0,
        stdout="created /obj/geo1/sphere1\n", stderr="", duration_seconds=0.7,
    )
    monkeypatch.setattr(app, "_try_build_bridge", lambda settings, log: fake_bridge)

    # The factory uses the PlaceholderProvider symbol bound in
    # aibridge_houdini.providers — patch that binding directly.
    class PlanProvider:
        name = "placeholder"

        def generate(self, request):
            return SAFE_PLAN

    monkeypatch.setattr(
        "aibridge_houdini.providers.PlaceholderProvider", PlanProvider
    )

    inputs = iter(["/mode direct", "Create a sphere.", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))

    out_buf, err_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        rc = app.main(["--env-file", str(tmp_path / "missing.env")])

    assert rc == 0
    out = out_buf.getvalue()
    assert "mode: direct" in out
    assert "provider: placeholder" in out
    assert SAFE_PLAN.summary in out
    assert "import hou" in out
    assert "ok (rc=0, 0.70s)" in out
    fake_bridge.execute.assert_called_once_with(SAFE_PLAN.houdini_python)


def test_help_lists_every_command(monkeypatch, capsys):
    sess = _session(monkeypatch)
    app.handle_slash_command("/help", sess)
    out = capsys.readouterr().out
    for token in ("/provider", "/mode", "/quit", "openai", "anthropic", "lmstudio", "safe", "dev", "direct"):
        assert token in out, token


def test_startup_banner_mentions_mode_and_transport(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("AIBRIDGE_LOG_DIR", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda *_: "/quit")

    out_buf = io.StringIO()
    with redirect_stdout(out_buf):
        rc = app.main(["--env-file", str(tmp_path / "missing.env")])
    out = out_buf.getvalue()

    assert rc == 0
    assert "mode: safe" in out
    assert "transport: NONE" in out  # no HYTHON_PATH in this fixture
