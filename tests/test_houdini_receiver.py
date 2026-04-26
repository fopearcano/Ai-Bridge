from __future__ import annotations

import json
import socket
import struct
from types import SimpleNamespace
from typing import Any

import pytest

from aibridge_houdini.houdini.houdini_receiver import (
    LOOPBACK_HOSTS,
    MAX_MESSAGE_BYTES,
    HoudiniReceiver,
    _read_message,
    _send_message,
)


# ---- direct execute() tests (no socket) ---------------------------------


def _direct_receiver(hou_module: Any | None = None) -> HoudiniReceiver:
    """Bind to an ephemeral port; we won't start the serve loop here."""
    return HoudiniReceiver(host="127.0.0.1", port=0, hou_module=hou_module)


def test_execute_captures_stdout_and_stderr():
    rx = _direct_receiver()
    try:
        result = rx.execute(
            "import sys\nprint('hello')\nsys.stderr.write('warn\\n')",
            request_id="r-1",
        )
    finally:
        rx.server_close()

    assert result["request_id"] == "r-1"
    assert result["success"] is True
    assert result["stdout"] == "hello\n"
    assert result["stderr"] == "warn\n"
    assert result["error"] is None


def test_execute_returns_traceback_on_exception():
    rx = _direct_receiver()
    try:
        result = rx.execute("raise RuntimeError('boom')", request_id="r-2")
    finally:
        rx.server_close()

    assert result["request_id"] == "r-2"
    assert result["success"] is False
    assert "RuntimeError: boom" in (result["error"] or "")
    assert "Traceback" in (result["error"] or "")


def test_execute_returns_syntax_error_without_running():
    rx = _direct_receiver()
    try:
        result = rx.execute("def (:", request_id="r-3")
    finally:
        rx.server_close()

    assert result["success"] is False
    assert "SyntaxError" in (result["error"] or "")
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_execute_treats_systemexit_as_failure_not_crash():
    rx = _direct_receiver()
    try:
        result = rx.execute("raise SystemExit(2)", request_id="r-4")
    finally:
        rx.server_close()

    assert result["success"] is False
    assert "SystemExit" in (result["error"] or "")


def test_execute_injects_hou_when_available():
    fake_hou = SimpleNamespace(
        applicationVersionString=lambda: "20.5.test",
        node=lambda path: f"node({path})",
    )
    rx = _direct_receiver(hou_module=fake_hou)
    try:
        result = rx.execute(
            "print(hou.node('/obj'))", request_id="r-5"
        )
    finally:
        rx.server_close()

    assert result["success"] is True
    assert result["stdout"].strip() == "node(/obj)"


def test_execute_runs_without_hou():
    rx = _direct_receiver()  # hou_module=None
    try:
        result = rx.execute("print(1 + 2)", request_id="r-6")
    finally:
        rx.server_close()

    assert result["success"] is True
    assert result["stdout"].strip() == "3"


# ---- bind safety --------------------------------------------------------


def test_refuses_to_bind_to_non_loopback():
    with pytest.raises(ValueError) as exc:
        HoudiniReceiver(host="0.0.0.0", port=0)
    msg = str(exc.value)
    assert "loopback" in msg
    assert "0.0.0.0" in msg


def test_loopback_hosts_allowlist():
    assert "127.0.0.1" in LOOPBACK_HOSTS
    assert "localhost" in LOOPBACK_HOSTS


# ---- socket round-trip tests --------------------------------------------


@pytest.fixture
def running_receiver():
    """Start a receiver on an ephemeral port and yield (host, port)."""
    rx = HoudiniReceiver(host="127.0.0.1", port=0)
    rx.start_in_background()
    host, port = rx.server_address
    try:
        yield host, port
    finally:
        rx.stop()


def _roundtrip(host: str, port: int, payload: dict, *, timeout: float = 5.0) -> dict:
    with socket.create_connection((host, port), timeout=timeout) as s:
        _send_message(s, json.dumps(payload))
        raw = _read_message(s)
    assert raw is not None, "server closed without responding"
    return json.loads(raw)


def test_roundtrip_success(running_receiver):
    host, port = running_receiver
    response = _roundtrip(
        host, port,
        {"code": "print('round-trip')", "request_id": "rt-ok"},
    )
    assert response["request_id"] == "rt-ok"
    assert response["success"] is True
    assert response["stdout"] == "round-trip\n"
    assert response["error"] is None


def test_roundtrip_returns_error_on_exception(running_receiver):
    host, port = running_receiver
    response = _roundtrip(
        host, port,
        {"code": "raise ValueError('nope')", "request_id": "rt-err"},
    )
    assert response["request_id"] == "rt-err"
    assert response["success"] is False
    assert "ValueError: nope" in (response["error"] or "")


def test_roundtrip_request_id_is_echoed(running_receiver):
    host, port = running_receiver
    rid = "abc-123-XYZ"
    response = _roundtrip(host, port, {"code": "x = 1", "request_id": rid})
    assert response["request_id"] == rid
    assert response["success"] is True


def test_roundtrip_bad_json(running_receiver):
    host, port = running_receiver
    with socket.create_connection(running_receiver, timeout=5.0) as s:
        body = b"not even json"
        s.sendall(struct.pack("!I", len(body)) + body)
        raw = _read_message(s)
    assert raw is not None
    response = json.loads(raw)
    assert response["success"] is False
    assert "bad request" in (response["error"] or "").lower()
    assert response["request_id"] == ""


def test_roundtrip_missing_code_field(running_receiver):
    host, port = running_receiver
    # `code` defaults to "" — that's a no-op exec, which should succeed.
    response = _roundtrip(host, port, {"request_id": "no-code"})
    assert response["success"] is True
    assert response["stdout"] == ""


def test_roundtrip_code_wrong_type(running_receiver):
    host, port = running_receiver
    response = _roundtrip(
        host, port, {"code": 42, "request_id": "wrong-type"}
    )
    assert response["success"] is False
    assert "code" in (response["error"] or "").lower()


def test_oversized_message_rejected_by_helper():
    # Helper-level guard: caller would never normally send this.
    fake_sock = SimpleNamespace(
        recv=lambda n: struct.pack("!I", MAX_MESSAGE_BYTES + 1)[:n]
    )
    with pytest.raises(ValueError) as exc:
        _read_message(fake_sock)
    assert "exceeds limit" in str(exc.value)
