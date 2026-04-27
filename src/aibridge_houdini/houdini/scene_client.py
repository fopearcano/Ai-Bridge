"""Host-side client for the in-Houdini receiver.

Used by the bridge to fetch ``{"type": "inspect_scene"}`` snapshots from a
running Houdini instance and forward them to the LLM as context.

The client speaks the same length-prefixed JSON wire as the receiver:
    [4 bytes big-endian uint32 length][UTF-8 JSON bytes]

All errors (no receiver listening, timeout, malformed reply, server-side
inspect failure) collapse to a returned ``None`` plus a warning log line so
the bridge can degrade gracefully — sending the prompt without scene
context is always better than failing the user's turn.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import uuid
from typing import Any


log = logging.getLogger("aibridge.houdini.scene_client")


DEFAULT_TIMEOUT = 5.0
LENGTH_HEADER_BYTES = 4
MAX_REPLY_BYTES = 16 * 1024 * 1024  # mirrors receiver's MAX_MESSAGE_BYTES


class SceneClientError(RuntimeError):
    """Raised by ``request`` for direct callers; ``fetch_scene_context``
    converts these into a logged warning + None."""


def fetch_scene_context(
    host: str,
    port: int,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict | None:
    """Best-effort: ask the receiver for a scene snapshot, return its data.

    Returns the ``data`` dict from ``inspect_scene`` on success, or ``None``
    if anything goes wrong (no server, timeout, server-side error). Never
    raises.
    """
    try:
        response = request(host, port, {"type": "inspect_scene"}, timeout=timeout)
    except SceneClientError as e:
        log.warning("could not fetch scene context from %s:%d: %s", host, port, e)
        return None

    if not response.get("success"):
        log.warning(
            "receiver reported inspect failure: %s",
            response.get("error") or "<no message>",
        )
        return None
    data = response.get("data")
    if not isinstance(data, dict):
        log.warning("receiver reply missing 'data' dict: %r", data)
        return None
    return data


def request(
    host: str,
    port: int,
    payload: dict,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Send one request to the receiver and return its parsed response.

    Adds a ``request_id`` if the caller didn't supply one.
    """
    if "request_id" not in payload:
        payload = dict(payload, request_id=uuid.uuid4().hex[:12])
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(struct.pack("!I", len(body)) + body)
            reply = _read_reply(s)
    except OSError as e:
        raise SceneClientError(f"transport error: {e}") from e

    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError as e:
        raise SceneClientError(f"invalid JSON from receiver: {e}") from e
    if not isinstance(parsed, dict):
        raise SceneClientError("receiver reply was not a JSON object")
    return parsed


def _read_reply(sock: socket.socket) -> bytes:
    header = _recv_exact(sock, LENGTH_HEADER_BYTES)
    if header is None:
        raise SceneClientError("receiver closed before sending a reply")
    (length,) = struct.unpack("!I", header)
    if length == 0:
        return b"{}"
    if length > MAX_REPLY_BYTES:
        raise SceneClientError(f"reply length {length} exceeds limit {MAX_REPLY_BYTES}")
    body = _recv_exact(sock, length)
    if body is None:
        raise SceneClientError("receiver closed mid-reply")
    return body


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
