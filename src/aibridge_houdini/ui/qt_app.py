"""PySide6 desktop UI for Ai-Bridge_Houdini.

Wraps the same primitives the CLI uses (Settings, ProviderRouter,
HoudiniBridge, evaluate, fetch_scene_context) so behavior stays
identical: scene context is fetched before each turn, the safety gate
runs per-mode, and execution goes through the configured transport.

Threading model
---------------
* The LLM call (router.route) runs on a QThread via RouteWorker.
* Execution (bridge.execute) runs on a separate QThread via ExecuteWorker.
* The GUI thread only does cheap work (safety analysis, rendering).

CLI keeps working: this module is import-only unless launched explicitly,
either via ``python -m aibridge_houdini.ui.qt_app`` or
``python -m aibridge_houdini --ui qt``.
"""

from __future__ import annotations

import logging
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, get_args

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from aibridge_houdini import __version__
from aibridge_houdini.config import ConfigError, Mode, Settings
from aibridge_houdini.execution.safety import SafetyReport, evaluate
from aibridge_houdini.houdini.bridge import HoudiniBridge
from aibridge_houdini.houdini.scene_client import fetch_scene_context
from aibridge_houdini.houdini.transport import ExecutionResult, TransportError
from aibridge_houdini.logging_setup import setup_logging
from aibridge_houdini.providers import ProviderRouter, RouterError
from aibridge_houdini.types import Command, UserRequest


VALID_MODES: tuple[Mode, ...] = get_args(Mode)


# ---- workers -----------------------------------------------------------


class RouteWorker(QObject):
    """Calls ``router.route(UserRequest)`` on a worker thread."""

    finished = Signal(object, str)  # (Command | None, error_message)

    def __init__(self, router: ProviderRouter, text: str) -> None:
        super().__init__()
        self._router = router
        self._text = text

    def run(self) -> None:
        try:
            command = self._router.route(UserRequest(text=self._text))
        except RouterError as e:
            self.finished.emit(None, str(e))
            return
        except Exception as e:  # safety net
            logging.getLogger("aibridge.ui.qt").exception("router crashed")
            self.finished.emit(None, f"unexpected error: {e}")
            return
        self.finished.emit(command, "")


class ExecuteWorker(QObject):
    """Calls ``bridge.execute(code)`` on a worker thread."""

    finished = Signal(object, str)  # (ExecutionResult | None, error_message)

    def __init__(self, bridge: HoudiniBridge, code: str) -> None:
        super().__init__()
        self._bridge = bridge
        self._code = code

    def run(self) -> None:
        try:
            result = self._bridge.execute(self._code)
        except NotImplementedError as e:
            self.finished.emit(None, str(e))
            return
        except Exception as e:
            logging.getLogger("aibridge.ui.qt").exception("transport crashed")
            self.finished.emit(None, f"transport error: {e}")
            return
        self.finished.emit(result, "")


# ---- log handler -------------------------------------------------------


class _GuiLogHandler(logging.Handler, QObject):
    """Logging handler that emits a Qt signal per record (thread-safe)."""

    record = Signal(str)

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.record.emit(self.format(record))
        except Exception:  # pragma: no cover
            self.handleError(record)


# ---- session state -----------------------------------------------------


@dataclass
class _Session:
    settings: Settings
    router: ProviderRouter
    bridge: HoudiniBridge | None
    log: logging.Logger
    mode: Mode


# ---- main window -------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self, session: _Session) -> None:
        super().__init__()
        self._session = session
        self._busy = False
        self._route_thread: QThread | None = None
        self._exec_thread: QThread | None = None

        self.setWindowTitle(f"Ai-Bridge Houdini v{__version__}")
        self.resize(1100, 760)

        self._build_ui()
        self._wire_log_handler()
        self._update_status()
        self._log_welcome()

    # -- construction ---------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top selector row
        top = QHBoxLayout()
        top.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        for name in self._session.router.enabled:
            self.provider_combo.addItem(name)
        self.provider_combo.setCurrentText(self._session.router.active)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        top.addWidget(self.provider_combo)

        top.addSpacing(20)
        top.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        for m in VALID_MODES:
            self.mode_combo.addItem(m)
        self.mode_combo.setCurrentText(self._session.mode)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        top.addWidget(self.mode_combo)

        top.addStretch(1)
        root.addLayout(top)

        # Input box
        root.addWidget(QLabel("Prompt:"))
        self.input_box = QPlainTextEdit()
        self.input_box.setPlaceholderText("Describe what you want in Houdini…")
        self.input_box.setFixedHeight(110)
        root.addWidget(self.input_box)

        # Send / Clear row
        send_row = QHBoxLayout()
        send_row.addStretch(1)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._on_clear)
        send_row.addWidget(self.clear_button)
        self.send_button = QPushButton("Send")
        self.send_button.setDefault(True)
        self.send_button.clicked.connect(self._on_send)
        send_row.addWidget(self.send_button)
        root.addLayout(send_row)

        # Splitter: response (top) / code / log
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.response_panel = self._make_panel("Response", monospace=False)
        splitter.addWidget(self.response_panel.container)

        self.code_panel = self._make_panel("Code (houdini_python)", monospace=True)
        splitter.addWidget(self.code_panel.container)

        self.log_panel = self._make_panel("Log", monospace=True)
        splitter.addWidget(self.log_panel.container)

        splitter.setSizes([220, 220, 200])
        root.addWidget(splitter, 1)

        self.setStatusBar(QStatusBar())

    def _make_panel(self, title: str, *, monospace: bool):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(QLabel(title))
        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        if monospace:
            font = QFont("Monospace")
            font.setStyleHint(QFont.StyleHint.TypeWriter)
            edit.setFont(font)
        layout.addWidget(edit)
        return _Panel(container=container, edit=edit)

    def _wire_log_handler(self) -> None:
        self._log_handler = _GuiLogHandler()
        self._log_handler.record.connect(self._append_log)
        logging.getLogger().addHandler(self._log_handler)

    def _log_welcome(self) -> None:
        self._session.log.info(
            "Qt UI started: provider=%s, mode=%s, transport=%s",
            self._session.router.active,
            self._session.mode,
            "hython" if self._session.bridge is not None else "none",
        )

    # -- handlers -------------------------------------------------------

    def _on_provider_changed(self, name: str) -> None:
        if not name or name == self._session.router.active:
            return
        try:
            self._session.router.set_provider(name)
        except RouterError as e:
            QMessageBox.warning(self, "Provider error", str(e))
            # Revert combo without re-triggering the slot.
            self.provider_combo.blockSignals(True)
            self.provider_combo.setCurrentText(self._session.router.active)
            self.provider_combo.blockSignals(False)
            return
        self._update_status()

    def _on_mode_changed(self, name: str) -> None:
        if name in VALID_MODES:
            self._session.mode = name  # type: ignore[assignment]
            self._session.log.info("mode set to %s", name)
            self._update_status()

    def _on_clear(self) -> None:
        self.input_box.clear()
        self.response_panel.edit.clear()
        self.code_panel.edit.clear()

    def _on_send(self) -> None:
        if self._busy:
            return
        text = self.input_box.toPlainText().strip()
        if not text:
            return
        self._set_busy(True)
        self.response_panel.edit.setPlainText("…thinking…")
        self.code_panel.edit.clear()

        worker = RouteWorker(self._session.router, text)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_route_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._route_thread = thread
        thread.start()

    def _on_route_done(self, command: Command | None, err: str) -> None:
        if err or command is None:
            self.response_panel.edit.setPlainText(f"error: {err or 'unknown'}")
            self._set_busy(False)
            return

        if command.is_clarification:
            self.response_panel.edit.setPlainText(_format_clarification(command))
            self.code_panel.edit.clear()
            self._set_busy(False)
            return

        # Show plan + code immediately so the user sees what's about to run.
        self.response_panel.edit.setPlainText(_format_plan(command))
        self.code_panel.edit.setPlainText(command.houdini_python)

        safety = evaluate(command.houdini_python, mode=self._session.mode)
        self._session.log.info(
            "safety verdict=%s mode=%s findings=%d",
            safety.verdict, safety.mode, len(safety.findings),
        )

        if safety.verdict == "blocked":
            self._append_result(_format_safety(safety) + "\n\nBLOCKED — code not executed.")
            self._set_busy(False)
            return

        if self._session.bridge is None:
            self._append_result(
                _format_safety(safety) +
                "\n\nNOT EXECUTED — no Houdini transport configured "
                "(set HYTHON_PATH to enable).")
            self._set_busy(False)
            return

        if safety.verdict == "needs_confirmation":
            allow = self._confirm_execute(command, safety)
            if not allow:
                self._append_result(_format_safety(safety) + "\n\nDECLINED — user did not confirm.")
                self._set_busy(False)
                return

        self._spawn_executor(command.houdini_python, safety)

    def _confirm_execute(self, command: Command, safety: SafetyReport) -> bool:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Execute generated code?")
        body = [
            f"Provider: {command.provider}",
            f"Mode: {self._session.mode}",
            f"Safety: {safety.verdict} ({len(safety.findings)} findings)",
            "",
            "Code preview:",
            textwrap.shorten(command.houdini_python.replace("\n", " ⏎ "), 200),
        ]
        msg.setText("\n".join(body))
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        return msg.exec() == QMessageBox.StandardButton.Yes

    def _spawn_executor(self, code: str, safety: SafetyReport) -> None:
        worker = ExecuteWorker(self._session.bridge, code)
        thread = QThread(self)
        worker.moveToThread(thread)

        def _done(result: ExecutionResult | None, err: str) -> None:
            if err or result is None:
                self._append_result(_format_safety(safety) + f"\n\nFAILED — {err or 'unknown error'}")
            else:
                self._append_result(_format_safety(safety) + "\n\n" + _format_exec(result))
            self._set_busy(False)

        thread.started.connect(worker.run)
        worker.finished.connect(_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._exec_thread = thread
        thread.start()

    # -- utility --------------------------------------------------------

    def _append_log(self, line: str) -> None:
        edit = self.log_panel.edit
        edit.appendPlainText(line)
        edit.moveCursor(QTextCursor.MoveOperation.End)

    def _append_result(self, text: str) -> None:
        # Append result/safety section under the existing plan summary so the
        # response panel ends up with: plan summary, then verdict, then exec.
        existing = self.response_panel.edit.toPlainText()
        sep = "\n\n" if existing else ""
        self.response_panel.edit.setPlainText(existing + sep + text)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.send_button.setEnabled(not busy)
        self.input_box.setReadOnly(busy)
        self._update_status()

    def _update_status(self) -> None:
        bridge_label = "hython" if self._session.bridge is not None else "no transport"
        state = "busy…" if self._busy else "ready"
        self.statusBar().showMessage(
            f"{state} | provider: {self._session.router.active} "
            f"| mode: {self._session.mode} | {bridge_label}"
        )

    # -- shutdown -------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 (Qt API)
        for t in (self._route_thread, self._exec_thread):
            if t is not None and t.isRunning():
                t.quit()
                t.wait(2000)
        try:
            logging.getLogger().removeHandler(self._log_handler)
        except Exception:
            pass
        super().closeEvent(event)


@dataclass
class _Panel:
    container: QWidget
    edit: QPlainTextEdit


# ---- formatting helpers (also used by tests) ---------------------------


def _format_plan(c: Command) -> str:
    return (
        f"provider: {c.provider}\n"
        f"intent:   {c.intent}\n"
        f"risk:     {c.risk_level}\n"
        f"summary:  {c.summary}\n"
        f"explain:  {c.explanation}\n"
        f"expects:  {c.expected_result}"
    )


def _format_clarification(c: Command) -> str:
    return (
        f"provider: {c.provider}\n"
        f"intent:   clarification\n"
        f"question: {(c.question or '').strip() or '(empty)'}"
    )


def _format_safety(s: SafetyReport) -> str:
    lines = [f"safety:   {s.verdict} (mode={s.mode})"]
    for f in s.findings:
        lines.append(f"    - [{f.severity}] {f.rule} (line {f.line}): {f.message}")
    return "\n".join(lines)


def _format_exec(r: ExecutionResult) -> str:
    head = "ok" if r.success else "FAILED"
    if r.return_code is not None:
        head += f" (rc={r.return_code}"
        if r.duration_seconds is not None:
            head += f", {r.duration_seconds:.2f}s"
        head += ")"
    out = [f"result:   {head}"]
    if r.error:
        out.append("error:")
        out.append(textwrap.indent(r.error.rstrip("\n"), "    "))
    if r.stdout:
        out.append("stdout:")
        out.append(textwrap.indent(r.stdout.rstrip("\n"), "    "))
    if r.stderr:
        out.append("stderr:")
        out.append(textwrap.indent(r.stderr.rstrip("\n"), "    "))
    return "\n".join(out)


# ---- entry points ------------------------------------------------------


def _build_session(env_file: str | None) -> _Session:
    settings = Settings.load(env_file=env_file or None)
    log = setup_logging(settings.log_level, settings.log_dir)
    router = ProviderRouter(
        settings,
        scene_context_fn=_make_scene_context_fn(settings, log),
    )
    bridge: HoudiniBridge | None
    try:
        bridge = HoudiniBridge(settings, transport="hython")
    except TransportError as e:
        log.warning("Houdini bridge unavailable: %s", e)
        bridge = None
    return _Session(
        settings=settings, router=router, bridge=bridge, log=log, mode=settings.mode
    )


def _make_scene_context_fn(settings: Settings, log: logging.Logger):
    host, port = settings.houdini_host, settings.houdini_port

    def _fetch() -> dict | None:
        ctx = fetch_scene_context(host, port, timeout=2.0)
        if ctx is None:
            log.debug("no scene context (receiver unreachable at %s:%d)", host, port)
        return ctx

    return _fetch


def run(env_file: str | None = ".env") -> int:
    """Launch the Qt UI. Returns a Unix-style exit code."""
    try:
        session = _build_session(env_file)
    except (ConfigError, RouterError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(session)
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="aibridge-houdini-qt")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)
    return run(env_file=args.env_file)


if __name__ == "__main__":
    raise SystemExit(main())
