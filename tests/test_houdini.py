from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aibridge_houdini.config import Settings
from aibridge_houdini.houdini import (
    ExecutionResult,
    HoudiniBridge,
    HythonTransport,
    SocketTransport,
    TransportError,
)


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
    ):
        monkeypatch.delenv(name, raising=False)


def _settings(monkeypatch, *, hython: str | None = None, **extra) -> Settings:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    if hython is not None:
        monkeypatch.setenv("HYTHON_PATH", hython)
    for k, v in extra.items():
        monkeypatch.setenv(k, v)
    return Settings.load(env_file=None)


# ---- HythonTransport ---------------------------------------------------


def test_hython_transport_writes_temp_file_and_invokes_binary(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, hython=str(tmp_path / "hython"))
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["script"] = Path(cmd[1])
        captured["script_text"] = captured["script"].read_text(encoding="utf-8")
        captured["timeout"] = kwargs.get("timeout")
        captured["text"] = kwargs.get("text")
        captured["capture_output"] = kwargs.get("capture_output")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="hi\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    transport = HythonTransport(settings, timeout=5.0)
    code = "import hou\nprint('hi')\n"
    result = transport.execute(code)

    # Invocation
    assert captured["cmd"][0] == str(tmp_path / "hython")
    assert captured["cmd"][1].endswith(".py")
    assert "aibridge_" in captured["cmd"][1]
    assert captured["timeout"] == 5.0
    assert captured["text"] is True
    assert captured["capture_output"] is True

    # Temp file contained the exact code, then was removed.
    assert captured["script_text"] == code
    assert not captured["script"].exists()

    # Result shape
    assert isinstance(result, ExecutionResult)
    assert result.transport == "hython"
    assert result.success is True
    assert result.return_code == 0
    assert result.stdout == "hi\n"
    assert result.stderr == ""
    assert result.duration_seconds is not None
    assert result.duration_seconds >= 0
    assert result.error is None


def test_hython_transport_reports_nonzero_return_code(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, hython=str(tmp_path / "hython"))

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = HythonTransport(settings).execute("raise SystemExit(1)")

    assert result.success is False
    assert result.return_code == 1
    assert result.stderr == "boom\n"
    assert result.error is None


def test_hython_transport_handles_missing_binary(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, hython=str(tmp_path / "nope" / "hython"))
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["script"] = Path(cmd[1])
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = HythonTransport(settings).execute("print('x')")

    assert result.success is False
    assert result.return_code is None
    assert "not found" in (result.error or "").lower()
    assert str(settings.hython_path) in (result.error or "")
    # temp file is still cleaned up even when the binary isn't there
    assert not captured["script"].exists()


def test_hython_transport_handles_timeout(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, hython=str(tmp_path / "hython"))

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=cmd, timeout=0.5, output="partial-out", stderr="partial-err"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = HythonTransport(settings, timeout=0.5).execute("print('x')")

    assert result.success is False
    assert "timed out" in (result.error or "").lower()
    assert result.stdout == "partial-out"
    assert result.stderr == "partial-err"


def test_hython_transport_requires_hython_path(monkeypatch):
    settings = _settings(monkeypatch)  # no HYTHON_PATH
    with pytest.raises(TransportError) as exc:
        HythonTransport(settings)
    assert "HYTHON_PATH" in str(exc.value)


def test_hython_transport_cleans_up_temp_file_on_unexpected_error(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, hython=str(tmp_path / "hython"))
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["script"] = Path(cmd[1])
        raise RuntimeError("unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        HythonTransport(settings).execute("print('x')")

    assert not captured["script"].exists()


# ---- SocketTransport ---------------------------------------------------


def test_socket_transport_is_placeholder_only(monkeypatch):
    settings = _settings(
        monkeypatch, HOUDINI_HOST="10.0.0.5", HOUDINI_PORT="42424"
    )
    transport = SocketTransport(settings)

    assert transport.name == "socket"
    assert transport.endpoint == "10.0.0.5:42424"
    with pytest.raises(NotImplementedError) as exc:
        transport.execute("print('x')")
    assert "10.0.0.5:42424" in str(exc.value)


# ---- HoudiniBridge -----------------------------------------------------


def test_bridge_defaults_to_hython(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, hython=str(tmp_path / "hython"))
    bridge = HoudiniBridge(settings)
    assert bridge.transport.name == "hython"


def test_bridge_can_select_socket(monkeypatch):
    settings = _settings(monkeypatch)
    bridge = HoudiniBridge(settings, transport="socket")
    assert bridge.transport.name == "socket"


def test_bridge_unknown_transport(monkeypatch):
    settings = _settings(monkeypatch)
    with pytest.raises(TransportError):
        HoudiniBridge(settings, transport="carrier-pigeon")  # type: ignore[arg-type]


def test_bridge_forwards_code_to_transport(monkeypatch):
    settings = _settings(monkeypatch)
    fake = MagicMock()
    fake.name = "fake"
    fake.execute.return_value = ExecutionResult(transport="fake", success=True)

    bridge = HoudiniBridge(settings, transport=fake)
    result = bridge.execute("print('hello')")

    fake.execute.assert_called_once_with("print('hello')")
    assert result.success is True
    assert result.transport == "fake"


def test_bridge_rejects_empty_code(monkeypatch):
    settings = _settings(monkeypatch)
    fake = MagicMock()
    fake.name = "fake"

    bridge = HoudiniBridge(settings, transport=fake)
    result = bridge.execute("   \n  ")

    fake.execute.assert_not_called()
    assert result.success is False
    assert "empty" in (result.error or "").lower()


def test_bridge_propagates_socket_not_implemented(monkeypatch):
    settings = _settings(monkeypatch)
    bridge = HoudiniBridge(settings, transport="socket")
    with pytest.raises(NotImplementedError):
        bridge.execute("print('x')")


def test_bridge_does_not_import_hou(monkeypatch):
    """Sanity: the module graph must not require an embedded Houdini."""
    import sys

    # Wipe any potentially-loaded hou and verify imports still work.
    for name in [k for k in sys.modules if k == "hou" or k.startswith("hou.")]:
        del sys.modules[name]

    import importlib

    # Re-import to confirm no top-level import of `hou` was added.
    importlib.import_module("aibridge_houdini.houdini.bridge")
    importlib.import_module("aibridge_houdini.houdini.transport")
    assert "hou" not in sys.modules
