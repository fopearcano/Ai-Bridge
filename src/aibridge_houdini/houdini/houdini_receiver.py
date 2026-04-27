"""Ai-Bridge_Houdini in-process receiver.

Runs INSIDE Houdini (e.g. from a shelf tool, Python source editor, or
``hython houdini_receiver.py``). Spawns a loopback-only TCP server that
accepts JSON requests, exec()s the supplied Python in this process — where
``hou`` is importable — and returns a JSON result.

Wire protocol (length-prefixed JSON, both directions)
-----------------------------------------------------
    [4 bytes big-endian uint32 length][UTF-8 JSON bytes]

Request:
    {"code": "<python source>", "request_id": "<opaque string>"}

Response:
    {"request_id": "...", "success": bool,
     "stdout": "...", "stderr": "...", "error": "..." | null}

SECURITY WARNING
----------------
This server runs UNTRUSTED CODE via exec() in the Houdini process. It is
intentionally not sandboxed — that's the whole point of the bridge. The
server therefore refuses to bind to anything but a loopback address, and
every accepted connection is double-checked against the loopback list
before any code is read. Even so:

  * Do NOT forward this port across the network or into containers.
  * Do NOT run this in a shared user session you don't control.
  * Anyone who can reach the port can do anything Houdini can do —
    delete files, save over your scene, run shell commands, etc.

The module is dependency-free on purpose so it can be dropped into Houdini's
PYTHONPATH or 123.py without pulling in the rest of aibridge_houdini.
"""

from __future__ import annotations

import io
import json
import logging
import socketserver
import struct
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any


log = logging.getLogger("aibridge.houdini.receiver")


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18861
LENGTH_HEADER_BYTES = 4
MAX_MESSAGE_BYTES = 16 * 1024 * 1024  # 16 MiB hard cap per message
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost"})
LOOPBACK_CLIENTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})


# ---- wire helpers --------------------------------------------------------


def _recv_exact(sock, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _read_message(sock) -> bytes | None:
    header = _recv_exact(sock, LENGTH_HEADER_BYTES)
    if header is None:
        return None
    (length,) = struct.unpack("!I", header)
    if length == 0:
        return b""
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(
            f"message length {length} exceeds limit {MAX_MESSAGE_BYTES}"
        )
    return _recv_exact(sock, length)


def _send_message(sock, payload: str) -> None:
    data = payload.encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


# ---- server --------------------------------------------------------------


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        client_ip = self.client_address[0]
        if client_ip not in LOOPBACK_CLIENTS:
            log.warning("rejecting non-loopback client %s", client_ip)
            return

        try:
            raw = _read_message(self.request)
        except (OSError, ValueError) as e:
            log.warning("read error from %s: %s", client_ip, e)
            return
        if raw is None:
            return  # client closed before sending a complete message

        request_id = ""
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise TypeError("request must be a JSON object")
            request_id = str(request.get("request_id", ""))
            msg_type = (request.get("type") or "exec").lower()
            if msg_type == "exec":
                code = request.get("code", "")
                if not isinstance(code, str):
                    raise TypeError("'code' must be a string")
            elif msg_type == "inspect_scene":
                code = None
            else:
                raise ValueError(f"unknown request type: {msg_type!r}")
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self._reply(
                request_id,
                success=False,
                stdout="",
                stderr="",
                error=f"bad request: {e}",
                data=None,
            )
            return

        if msg_type == "inspect_scene":
            log.info("inspect_scene request_id=%s", request_id)
            result = self.server.inspect(request_id)  # type: ignore[attr-defined]
        else:
            log.info("exec request_id=%s len=%d", request_id, len(code))
            result = self.server.execute(code, request_id)  # type: ignore[attr-defined]
        self._reply(**result)

    def _reply(
        self,
        request_id: str,
        *,
        success: bool,
        stdout: str,
        stderr: str,
        error: str | None,
        data: Any = None,
    ) -> None:
        payload = json.dumps(
            {
                "request_id": request_id,
                "success": success,
                "stdout": stdout,
                "stderr": stderr,
                "error": error,
                "data": data,
            },
            ensure_ascii=False,
        )
        try:
            _send_message(self.request, payload)
        except OSError as e:
            log.warning("could not send reply for request_id=%s: %s", request_id, e)


class HoudiniReceiver(socketserver.ThreadingTCPServer):
    """Loopback-only socket server that exec()s Python in the Houdini process."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        hou_module: Any = None,
    ) -> None:
        if host not in LOOPBACK_HOSTS:
            raise ValueError(
                f"refusing to bind to non-loopback host {host!r}; "
                "this server runs untrusted exec() and must stay local. "
                f"Allowed: {sorted(LOOPBACK_HOSTS)}"
            )
        super().__init__((host, port), _Handler)
        self._serve_thread: threading.Thread | None = None
        self._hou = hou_module if hou_module is not None else _try_import_hou()

        log.warning(
            "Ai-Bridge receiver listening on %s:%d. SECURITY: this server "
            "exec()s arbitrary Python in the Houdini process. Loopback only — "
            "do NOT expose this port off-host.",
            host,
            self.server_address[1],
        )

    # -- lifecycle ---------------------------------------------------------

    def start_in_background(self) -> None:
        """Run serve_forever() on a daemon thread (typical Houdini usage)."""
        if self._serve_thread is not None and self._serve_thread.is_alive():
            return
        self._serve_thread = threading.Thread(
            target=self.serve_forever,
            name="aibridge-receiver",
            daemon=True,
        )
        self._serve_thread.start()

    def stop(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
        try:
            self.server_close()
        except Exception:
            pass
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=2)
            self._serve_thread = None

    # -- exec --------------------------------------------------------------

    def execute(self, code: str, request_id: str) -> dict:
        out_buf, err_buf = io.StringIO(), io.StringIO()
        namespace: dict[str, Any] = {"__name__": "__aibridge__"}
        if self._hou is not None:
            namespace["hou"] = self._hou

        try:
            compiled = compile(code, "<aibridge-receiver>", "exec")
        except SyntaxError:
            tb = traceback.format_exc()
            log.warning("syntax error in request_id=%s", request_id)
            return {
                "request_id": request_id,
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": tb,
                "data": None,
            }

        try:
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                exec(compiled, namespace)
        except SystemExit as e:
            return {
                "request_id": request_id,
                "success": False,
                "stdout": out_buf.getvalue(),
                "stderr": err_buf.getvalue(),
                "error": f"SystemExit({e.code!r})",
                "data": None,
            }
        except BaseException:
            tb = traceback.format_exc()
            log.warning("exec failed for request_id=%s", request_id)
            return {
                "request_id": request_id,
                "success": False,
                "stdout": out_buf.getvalue(),
                "stderr": err_buf.getvalue(),
                "error": tb,
                "data": None,
            }

        return {
            "request_id": request_id,
            "success": True,
            "stdout": out_buf.getvalue(),
            "stderr": err_buf.getvalue(),
            "error": None,
            "data": None,
        }

    # -- inspect_scene -----------------------------------------------------

    def inspect(self, request_id: str) -> dict:
        """Snapshot the current scene via houdini.inspection."""
        # Imported lazily so the receiver still loads if inspection.py is
        # missing (defensive — both files normally ship together).
        try:
            from aibridge_houdini.houdini.inspection import inspect_scene
        except ImportError as e:
            return {
                "request_id": request_id,
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"inspection module unavailable: {e}",
                "data": None,
            }

        try:
            scene = inspect_scene(self._hou)
        except BaseException:
            tb = traceback.format_exc()
            log.warning("inspect failed for request_id=%s", request_id)
            return {
                "request_id": request_id,
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": tb,
                "data": None,
            }
        return {
            "request_id": request_id,
            "success": True,
            "stdout": "",
            "stderr": "",
            "error": None,
            "data": scene,
        }


# ---- helpers -------------------------------------------------------------


def _try_import_hou() -> Any | None:
    try:
        import hou  # type: ignore[import-not-found]
    except ImportError:
        log.info("hou module not importable; running outside Houdini")
        return None
    try:
        log.info("running inside Houdini %s", hou.applicationVersionString())
    except Exception:
        # hou may be a stub or partially loaded — that's fine
        pass
    return hou


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    server = HoudiniReceiver(host=host, port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("interrupted; shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
