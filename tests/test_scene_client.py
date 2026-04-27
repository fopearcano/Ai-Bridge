from __future__ import annotations

import json
import socket
import struct
import threading

import pytest

from aibridge_houdini.houdini.houdini_receiver import HoudiniReceiver
from aibridge_houdini.houdini.scene_client import (
    SceneClientError,
    fetch_scene_context,
    request,
)


# ---- end-to-end against the real receiver ------------------------------


def _running_receiver(hou_module=None):
    rx = HoudiniReceiver(host="127.0.0.1", port=0, hou_module=hou_module)
    rx.start_in_background()
    return rx


def test_fetch_scene_context_against_running_receiver_without_hou():
    rx = _running_receiver(hou_module=None)
    host, port = rx.server_address
    try:
        ctx = fetch_scene_context(host, port, timeout=2.0)
    finally:
        rx.stop()
    # No hou available -> data still returned (with available=False).
    assert isinstance(ctx, dict)
    assert ctx["available"] is False
    assert ctx["objects"] == []


def test_fetch_scene_context_returns_none_when_no_server_listening():
    # Bind+release a port to find one that's almost certainly closed.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
    # `port` is now closed.
    ctx = fetch_scene_context("127.0.0.1", port, timeout=0.5)
    assert ctx is None


def test_request_adds_request_id_when_missing():
    rx = _running_receiver()
    host, port = rx.server_address
    try:
        response = request(host, port, {"type": "inspect_scene"}, timeout=2.0)
    finally:
        rx.stop()
    assert response["success"] is True
    assert response["request_id"]  # non-empty


def test_request_preserves_caller_request_id():
    rx = _running_receiver()
    host, port = rx.server_address
    try:
        response = request(
            host, port,
            {"type": "inspect_scene", "request_id": "MY-ID"},
            timeout=2.0,
        )
    finally:
        rx.stop()
    assert response["request_id"] == "MY-ID"


def test_request_raises_on_connection_failure():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
    with pytest.raises(SceneClientError) as exc:
        request("127.0.0.1", port, {"type": "inspect_scene"}, timeout=0.5)
    assert "transport error" in str(exc.value).lower()


# ---- malformed-server scenarios ----------------------------------------


def _serve_one_raw(reply_bytes: bytes, port_holder: list):
    """Run a one-shot server that returns reply_bytes verbatim then closes."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port_holder.append(srv.getsockname()[1])

    def _accept():
        conn, _ = srv.accept()
        with conn:
            try:
                conn.recv(8192)  # discard request
                if reply_bytes:
                    conn.sendall(reply_bytes)
            finally:
                pass
        srv.close()

    t = threading.Thread(target=_accept, daemon=True)
    t.start()
    return t


def test_request_raises_on_malformed_json_reply():
    holder: list = []
    body = b"not json"
    reply = struct.pack("!I", len(body)) + body
    _serve_one_raw(reply, holder)
    while not holder:
        pass
    port = holder[0]
    with pytest.raises(SceneClientError) as exc:
        request("127.0.0.1", port, {"type": "inspect_scene"}, timeout=2.0)
    assert "invalid json" in str(exc.value).lower()


def test_fetch_returns_none_when_receiver_reports_failure():
    holder: list = []
    body = json.dumps(
        {"request_id": "x", "success": False,
         "stdout": "", "stderr": "",
         "error": "boom", "data": None}
    ).encode("utf-8")
    reply = struct.pack("!I", len(body)) + body
    _serve_one_raw(reply, holder)
    while not holder:
        pass
    port = holder[0]
    ctx = fetch_scene_context("127.0.0.1", port, timeout=2.0)
    assert ctx is None


def test_fetch_returns_none_when_data_missing():
    holder: list = []
    body = json.dumps(
        {"request_id": "x", "success": True,
         "stdout": "", "stderr": "", "error": None, "data": None}
    ).encode("utf-8")
    reply = struct.pack("!I", len(body)) + body
    _serve_one_raw(reply, holder)
    while not holder:
        pass
    port = holder[0]
    ctx = fetch_scene_context("127.0.0.1", port, timeout=2.0)
    assert ctx is None
