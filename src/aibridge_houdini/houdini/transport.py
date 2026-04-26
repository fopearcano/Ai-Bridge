from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from aibridge_houdini.config import Settings


log = logging.getLogger("aibridge.houdini.transport")


class TransportError(RuntimeError):
    """Raised when a transport cannot be constructed (e.g. missing config)."""


class ExecutionResult(BaseModel):
    """Outcome of one Houdini Python execution."""

    transport: str
    success: bool
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float | None = None
    error: str | None = None  # bridge-level error (binary missing, timeout, …)


class Transport(Protocol):
    name: str

    def execute(self, code: str) -> ExecutionResult: ...


class HythonTransport:
    """Run code by writing it to a temp file and invoking `hython script.py`.

    Does NOT depend on Houdini being importable in this process — it shells out
    to the user-configured hython binary and reads back stdout/stderr.
    """

    name = "hython"

    def __init__(self, settings: Settings, timeout: float = 120.0) -> None:
        if settings.hython_path is None:
            raise TransportError(
                "HYTHON_PATH is not set — required for the hython transport"
            )
        self._hython = Path(settings.hython_path)
        self._timeout = timeout

    def execute(self, code: str) -> ExecutionResult:
        # delete=False so the subprocess can re-open the path on Windows.
        # We always remove it ourselves in the finally block.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="aibridge_",
            delete=False,
            encoding="utf-8",
        ) as fp:
            fp.write(code)
            script_path = Path(fp.name)

        cmd = [str(self._hython), str(script_path)]
        log.info("hython exec: %s (timeout=%.1fs)", cmd, self._timeout)
        start = time.monotonic()
        try:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
            except FileNotFoundError:
                duration = time.monotonic() - start
                log.error("hython binary not found at %s", self._hython)
                return ExecutionResult(
                    transport=self.name,
                    success=False,
                    error=f"hython binary not found: {self._hython}",
                    duration_seconds=duration,
                )
            except subprocess.TimeoutExpired as e:
                duration = time.monotonic() - start
                log.error("hython timed out after %.1fs", self._timeout)
                return ExecutionResult(
                    transport=self.name,
                    success=False,
                    error=f"hython timed out after {self._timeout}s",
                    stdout=_decode(e.stdout),
                    stderr=_decode(e.stderr),
                    duration_seconds=duration,
                )

            duration = time.monotonic() - start
            log.info(
                "hython rc=%d duration=%.2fs stdout=%dB stderr=%dB",
                proc.returncode,
                duration,
                len(proc.stdout or ""),
                len(proc.stderr or ""),
            )
            return ExecutionResult(
                transport=self.name,
                success=proc.returncode == 0,
                return_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                duration_seconds=duration,
            )
        finally:
            try:
                script_path.unlink(missing_ok=True)
            except OSError as e:
                log.warning("could not remove temp script %s: %s", script_path, e)


class SocketTransport:
    """Placeholder for a future RPC transport against an in-Houdini server.

    Interface only — `execute()` raises NotImplementedError. The real adapter
    will speak to a small server inside Houdini bound to HOUDINI_HOST/PORT.
    """

    name = "socket"

    def __init__(self, settings: Settings) -> None:
        self._host = settings.houdini_host
        self._port = settings.houdini_port

    @property
    def endpoint(self) -> str:
        return f"{self._host}:{self._port}"

    def execute(self, code: str) -> ExecutionResult:
        raise NotImplementedError(
            f"SocketTransport({self.endpoint}) is not implemented yet — "
            "use HythonTransport via HoudiniBridge until the in-Houdini "
            "RPC server lands."
        )


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
