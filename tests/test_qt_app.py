"""Smoke tests for the PySide6 UI.

The Qt platform plugin defaults to ``offscreen`` so these run headless on
CI / containers without an X server. If PySide6 isn't installed at all
the whole module is skipped.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

# Force a headless platform before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")  # also pulls in Qt platform deps

from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from aibridge_houdini.config import Settings  # noqa: E402
from aibridge_houdini.execution.safety import SafetyReport  # noqa: E402
from aibridge_houdini.houdini.transport import ExecutionResult  # noqa: E402
from aibridge_houdini.providers import ProviderRouter, RouterError  # noqa: E402
from aibridge_houdini.providers.base import ProviderError  # noqa: E402
from aibridge_houdini.types import Command, LLMPlan, UserRequest  # noqa: E402
from aibridge_houdini.ui import qt_app  # noqa: E402


# ---- shared fixtures ---------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for name in (
        "PROVIDERS",
        "DEFAULT_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LMSTUDIO_MODEL",
        "MODE",
        "HOUDINI_HOST",
        "HOUDINI_PORT",
        "HYTHON_PATH",
        "AIBRIDGE_LOG_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _settings(monkeypatch, *, default="openai") -> Settings:
    monkeypatch.setenv("PROVIDERS", "openai,anthropic,lmstudio")
    monkeypatch.setenv("DEFAULT_PROVIDER", default)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-abcdefgh12345678")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen2.5-coder-7b")
    return Settings.load(env_file=None)


def _full_plan(intent: str = "create_geometry") -> LLMPlan:
    return LLMPlan(
        intent=intent,
        summary="Create a sphere",
        risk_level="low",
        requires_houdini=True,
        houdini_python="import hou\nhou.node('/obj').createNode('geo')\n",
        explanation="Adds a geo container.",
        expected_result="A new /obj/geo1 appears.",
    )


def _full_command() -> Command:
    return Command.from_plan("openai", UserRequest(text="make a sphere"), _full_plan())


def _clarification_command() -> Command:
    plan = LLMPlan(
        intent="clarification",
        requires_houdini=False,
        question="Polygon or NURBS?",
    )
    return Command.from_plan("openai", UserRequest(text="make sphere"), plan)


# ---- formatters --------------------------------------------------------


def test_format_plan_includes_provider_and_summary():
    out = qt_app._format_plan(_full_command())
    assert "provider: openai" in out
    assert "summary:  Create a sphere" in out
    assert "risk:     low" in out


def test_format_clarification_uses_question_field():
    out = qt_app._format_clarification(_clarification_command())
    assert "intent:   clarification" in out
    assert "question: Polygon or NURBS?" in out


def test_format_safety_lists_findings_in_order():
    from aibridge_houdini.execution.safety import Finding
    rep = SafetyReport(
        verdict="needs_confirmation",
        mode="safe",
        findings=(
            Finding(rule="forbidden-import", severity="medium",
                    line=1, message="os.remove forbidden"),
        ),
    )
    out = qt_app._format_safety(rep)
    assert "safety:   needs_confirmation (mode=safe)" in out
    assert "[medium] forbidden-import (line 1): os.remove forbidden" in out


def test_format_exec_block():
    r = ExecutionResult(
        transport="hython", success=True, return_code=0,
        stdout="hello\n", stderr="", duration_seconds=0.42,
    )
    out = qt_app._format_exec(r)
    assert "result:   ok (rc=0, 0.42s)" in out
    assert "stdout:" in out
    assert "    hello" in out


def test_format_exec_failed_with_error():
    r = ExecutionResult(
        transport="hython", success=False, return_code=None,
        stdout="", stderr="", error="hython binary not found: /nope",
    )
    out = qt_app._format_exec(r)
    assert "FAILED" in out
    assert "hython binary not found" in out


# ---- workers (run in a real QThread) -----------------------------------


def _drain_event_loop(timeout_ms: int = 2000) -> None:
    """Spin the Qt event loop briefly until pending signals settle."""
    loop = QEventLoop()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()


def test_route_worker_emits_command_on_success(qapp, monkeypatch):
    s = _settings(monkeypatch)

    class FakeProvider:
        name = "openai"
        def generate(self, request):
            return _full_plan()

    router = ProviderRouter(s, providers={"openai": FakeProvider()})
    worker = qt_app.RouteWorker(router, "make a sphere")
    received: list = []
    worker.finished.connect(lambda cmd, err: received.append((cmd, err)))
    worker.run()

    assert len(received) == 1
    cmd, err = received[0]
    assert err == ""
    assert cmd.intent == "create_geometry"


def test_route_worker_emits_error_on_router_failure(qapp, monkeypatch):
    s = _settings(monkeypatch)

    class BoomProvider:
        name = "openai"
        def generate(self, request):
            raise ProviderError("boom")

    router = ProviderRouter(s, providers={"openai": BoomProvider()})
    worker = qt_app.RouteWorker(router, "make a sphere")
    received: list = []
    worker.finished.connect(lambda cmd, err: received.append((cmd, err)))
    worker.run()

    assert len(received) == 1
    cmd, err = received[0]
    assert cmd is None
    assert "openai" in err and "auto-repair" in err


def test_execute_worker_forwards_to_bridge(qapp):
    bridge = MagicMock()
    bridge.execute.return_value = ExecutionResult(
        transport="hython", success=True, return_code=0,
        stdout="ok\n", stderr="",
    )
    worker = qt_app.ExecuteWorker(bridge, "import hou\n")
    received: list = []
    worker.finished.connect(lambda res, err: received.append((res, err)))
    worker.run()

    bridge.execute.assert_called_once_with("import hou\n")
    assert len(received) == 1
    res, err = received[0]
    assert err == ""
    assert res.success is True
    assert res.stdout == "ok\n"


def test_execute_worker_emits_error_on_not_implemented(qapp):
    bridge = MagicMock()
    bridge.execute.side_effect = NotImplementedError("socket transport not ready")
    worker = qt_app.ExecuteWorker(bridge, "import hou\n")
    received: list = []
    worker.finished.connect(lambda res, err: received.append((res, err)))
    worker.run()

    assert len(received) == 1
    res, err = received[0]
    assert res is None
    assert "socket transport not ready" in err


# ---- MainWindow construction + interactions ----------------------------


def _make_window(qapp, monkeypatch, *, with_bridge=False):
    s = _settings(monkeypatch, default="openai")

    class StubProvider:
        name = "openai"
        def generate(self, request):
            return _full_plan()

    router = ProviderRouter(s, providers={
        "openai": StubProvider(),
        "anthropic": StubProvider(),
        "lmstudio": StubProvider(),
    })

    bridge = None
    if with_bridge:
        bridge = MagicMock()
        bridge.transport = MagicMock(name="hython")
        bridge.execute.return_value = ExecutionResult(
            transport="hython", success=True, return_code=0,
            stdout="ran\n", stderr="", duration_seconds=0.1,
        )
    import logging
    session = qt_app._Session(
        settings=s, router=router, bridge=bridge,
        log=logging.getLogger("aibridge.test"), mode=s.mode,
    )
    return qt_app.MainWindow(session), session


def test_window_initializes_with_combos_populated(qapp, monkeypatch):
    win, _ = _make_window(qapp, monkeypatch)
    try:
        items = [win.provider_combo.itemText(i) for i in range(win.provider_combo.count())]
        assert items == ["openai", "anthropic", "lmstudio"]
        modes = [win.mode_combo.itemText(i) for i in range(win.mode_combo.count())]
        assert set(modes) == {"dev", "safe", "direct"}
        assert win.send_button.isEnabled()
    finally:
        win.close()


def test_provider_combo_change_calls_router(qapp, monkeypatch):
    win, session = _make_window(qapp, monkeypatch)
    try:
        win.provider_combo.setCurrentText("lmstudio")
        # Drain pending signals
        QCoreApplication.processEvents()
        assert session.router.active == "lmstudio"
    finally:
        win.close()


def test_provider_combo_reverts_on_router_error(qapp, monkeypatch):
    win, session = _make_window(qapp, monkeypatch)
    try:
        # Patch the router to refuse the switch.
        original = session.router.set_provider
        def boom(name):
            raise RouterError("denied")
        session.router.set_provider = boom  # type: ignore[method-assign]

        # Patch QMessageBox to avoid actually opening a modal dialog.
        from PySide6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)

        win.provider_combo.setCurrentText("anthropic")
        QCoreApplication.processEvents()
        # Active provider unchanged, combo reverted.
        assert session.router.active == "openai"
        assert win.provider_combo.currentText() == "openai"
        session.router.set_provider = original  # type: ignore[method-assign]
    finally:
        win.close()


def test_mode_combo_updates_session_mode(qapp, monkeypatch):
    win, session = _make_window(qapp, monkeypatch)
    try:
        win.mode_combo.setCurrentText("direct")
        QCoreApplication.processEvents()
        assert session.mode == "direct"
    finally:
        win.close()


def test_send_with_empty_input_is_a_noop(qapp, monkeypatch):
    win, _ = _make_window(qapp, monkeypatch)
    try:
        win.input_box.setPlainText("   ")
        win._on_send()
        # Send button stays enabled, no busy state.
        assert win.send_button.isEnabled()
        assert not win._busy
    finally:
        win.close()


def test_route_done_renders_clarification_without_safety(qapp, monkeypatch):
    win, _ = _make_window(qapp, monkeypatch)
    try:
        win._on_route_done(_clarification_command(), "")
        text = win.response_panel.edit.toPlainText()
        assert "intent:   clarification" in text
        assert "question: Polygon or NURBS?" in text
        assert win.code_panel.edit.toPlainText() == ""
        assert "safety:" not in text
    finally:
        win.close()


def test_route_done_with_no_bridge_skips_execution(qapp, monkeypatch):
    win, _ = _make_window(qapp, monkeypatch)  # with_bridge=False
    try:
        win._on_route_done(_full_command(), "")
        text = win.response_panel.edit.toPlainText()
        assert "summary:  Create a sphere" in text
        assert "NOT EXECUTED" in text
        assert win.code_panel.edit.toPlainText().startswith("import hou")
    finally:
        win.close()


def test_route_done_blocked_by_safety_does_not_execute(qapp, monkeypatch):
    win, _ = _make_window(qapp, monkeypatch, with_bridge=True)
    try:
        evil = Command(
            provider="openai",
            request="rm",
            intent="bad_intent",
            summary="bad",
            risk_level="high",
            requires_houdini=True,
            houdini_python="import os\nos.system('echo hi')\n",
            explanation="dangerous",
            expected_result="never",
        )
        win._on_route_done(evil, "")
        text = win.response_panel.edit.toPlainText()
        assert "BLOCKED" in text
        # Bridge must not have been called.
        assert not win._session.bridge.execute.called
    finally:
        win.close()


def test_route_done_error_path_renders_error(qapp, monkeypatch):
    win, _ = _make_window(qapp, monkeypatch)
    try:
        win._on_route_done(None, "router exploded")
        text = win.response_panel.edit.toPlainText()
        assert "error:" in text
        assert "router exploded" in text
        assert win._busy is False
    finally:
        win.close()


def test_clear_button_resets_panels(qapp, monkeypatch):
    win, _ = _make_window(qapp, monkeypatch)
    try:
        win.input_box.setPlainText("something")
        win.response_panel.edit.setPlainText("prior plan")
        win.code_panel.edit.setPlainText("import hou")
        win._on_clear()
        assert win.input_box.toPlainText() == ""
        assert win.response_panel.edit.toPlainText() == ""
        assert win.code_panel.edit.toPlainText() == ""
    finally:
        win.close()


# ---- CLI flag delegation -----------------------------------------------


def test_cli_ui_qt_dispatches_to_qt_run(monkeypatch, tmp_path):
    """`python -m aibridge_houdini --ui qt` must hand off to qt_app.run."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("AIBRIDGE_LOG_DIR", str(tmp_path))

    from aibridge_houdini import app as cli_app

    called: dict = {}

    def fake_run(env_file: str | None = ".env") -> int:
        called["env_file"] = env_file
        return 0

    monkeypatch.setattr("aibridge_houdini.ui.qt_app.run", fake_run)

    rc = cli_app.main(["--ui", "qt", "--env-file", str(tmp_path / "missing.env")])
    assert rc == 0
    assert called["env_file"] == str(tmp_path / "missing.env")
