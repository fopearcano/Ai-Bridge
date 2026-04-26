from __future__ import annotations

import textwrap

import pytest

from aibridge_houdini.execution.safety import (
    Finding,
    SafetyReport,
    analyze,
    evaluate,
    evaluate_with_settings,
)


def src(s: str) -> str:
    return textwrap.dedent(s).strip()


def _rules(report: SafetyReport) -> set[str]:
    return {f.rule for f in report.findings}


# ---- mode policy --------------------------------------------------------


def test_clean_hou_code_is_safe_in_every_mode():
    code = src(
        """
        import hou
        geo = hou.node('/obj').createNode('geo', 'geo1')
        sphere = geo.createNode('sphere', 'sphere1')
        sphere.parm('rad').set(1.5)
        """
    )
    for mode in ("dev", "safe", "direct"):
        report = evaluate(code, mode=mode)  # type: ignore[arg-type]
        assert report.verdict == "safe", (mode, report.findings)
        assert report.findings == ()


def test_safe_mode_warn_becomes_needs_confirmation():
    code = src(
        """
        with open('/tmp/render.exr', 'w') as f:
            f.write('x')
        """
    )
    r_safe = evaluate(code, mode="safe")
    assert r_safe.verdict == "needs_confirmation"
    assert "fs.write_outside_allowed" in _rules(r_safe)
    assert r_safe.warnings and not r_safe.critical


def test_direct_mode_warn_is_allowed_but_critical_blocks():
    warn_code = "open('/tmp/x.txt', 'w')"
    crit_code = "import subprocess"

    assert evaluate(warn_code, mode="direct").verdict == "safe"
    assert evaluate(crit_code, mode="direct").verdict == "blocked"


def test_dev_mode_warn_is_allowed_but_critical_blocks():
    warn_code = "open('/tmp/x.txt', 'w')"
    crit_code = "import subprocess"

    assert evaluate(warn_code, mode="dev").verdict == "safe"
    assert evaluate(crit_code, mode="dev").verdict == "blocked"


def test_critical_blocks_in_safe_mode_too():
    assert evaluate("import subprocess", mode="safe").verdict == "blocked"


# ---- import bans --------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess",
        "from subprocess import run",
        "import socket",
        "import requests",
        "import httpx",
        "import aiohttp",
        "import paramiko",
        "import ftplib",
        "import smtplib",
        "import telnetlib",
        "import pickle",
        "import urllib.request",
        "from urllib.request import urlopen",
        "import http.client",
        "from http.server import HTTPServer",
        "import xmlrpc.client",
    ],
)
def test_import_blocked(code):
    report = evaluate(code, mode="direct")
    assert report.verdict == "blocked", code
    assert "import.blocked" in _rules(report)


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import os.path",
        "from os.path import join",
        "import sys",
        "import math",
        "import json",
        "import urllib.parse",
        "import hou",
    ],
)
def test_benign_imports_pass(code):
    report = evaluate(code, mode="direct")
    assert report.verdict == "safe", code


# ---- destructive calls --------------------------------------------------


@pytest.mark.parametrize(
    "code,rule",
    [
        ("import os\nos.remove('/tmp/foo')", "call.blocked"),
        ("import os\nos.unlink('/tmp/foo')", "call.blocked"),
        ("import os\nos.rmdir('/tmp/foo')", "call.blocked"),
        ("import os\nos.system('rm -rf /')", "call.blocked"),
        ("import os\nos.popen('ls')", "call.blocked"),
        ("import shutil\nshutil.rmtree('/tmp/foo')", "call.blocked"),
    ],
)
def test_destructive_calls_blocked(code, rule):
    report = evaluate(code, mode="direct")
    assert report.verdict == "blocked", code
    assert rule in _rules(report)


def test_method_call_unlink_blocked():
    code = src(
        """
        from pathlib import Path
        p = Path('/tmp/foo')
        p.unlink()
        """
    )
    report = evaluate(code, mode="direct")
    assert report.verdict == "blocked"
    assert "call.blocked" in _rules(report)


def test_method_call_rmtree_blocked():
    code = "tree.rmtree()"
    report = evaluate(code, mode="direct")
    assert report.verdict == "blocked"


# ---- meta-programming ---------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "eval('1+1')",
        "exec('x = 1')",
        "compile('x', '<s>', 'exec')",
        "__import__('os')",
    ],
)
def test_eval_exec_blocked(code):
    report = evaluate(code, mode="direct")
    assert report.verdict == "blocked", code
    assert "meta.exec" in _rules(report)


# ---- file writes --------------------------------------------------------


def test_open_read_is_safe():
    report = evaluate("with open('/anywhere/foo.txt') as f: data = f.read()", mode="safe")
    assert report.verdict == "safe"
    assert report.findings == ()


def test_open_write_outside_allowed_is_warn():
    report = evaluate("open('/etc/passwd', 'w')", mode="safe")
    assert report.verdict == "needs_confirmation"
    assert "fs.write_outside_allowed" in _rules(report)


@pytest.mark.parametrize(
    "path",
    [
        "$HIP/render.exr",
        "$JOB/cache/sim.bgeo.sc",
        "$TEMP/scratch.txt",
        "$HOME/Desktop/out.png",
    ],
)
def test_open_write_inside_allowed_prefix_is_safe(path):
    code = f"open({path!r}, 'w')"
    report = evaluate(code, mode="safe")
    assert report.verdict == "safe", path


def test_open_write_with_dynamic_path_is_warn():
    code = src(
        """
        import hou
        path = hou.expandString('$HIP/out.txt')
        open(path, 'w')
        """
    )
    report = evaluate(code, mode="safe")
    assert report.verdict == "needs_confirmation"
    assert "fs.write_dynamic_path" in _rules(report)


def test_open_with_keyword_mode_detected():
    report = evaluate("open('/etc/host', mode='a')", mode="safe")
    assert report.verdict == "needs_confirmation"


def test_pathlib_write_text_outside_allowed_is_warn():
    code = "from pathlib import Path\nPath('/etc/x').write_text('y')"
    report = evaluate(code, mode="safe")
    assert report.verdict == "needs_confirmation"
    assert "fs.write_outside_allowed" in _rules(report)


def test_pathlib_write_text_inside_allowed_is_safe():
    code = "from pathlib import Path\nPath('$HIP/x.txt').write_text('y')"
    report = evaluate(code, mode="safe")
    assert report.verdict == "safe"


def test_custom_allowed_prefixes_overrides_default():
    code = "open('/work/render.exr', 'w')"
    # default: not allowed -> needs_confirmation
    assert evaluate(code, mode="safe").verdict == "needs_confirmation"
    # custom: /work allowed -> safe
    report = evaluate(code, mode="safe", allowed_path_prefixes=["/work"])
    assert report.verdict == "safe"


# ---- syntax error -------------------------------------------------------


def test_syntax_error_is_blocked():
    report = evaluate("def (:\n", mode="direct")
    assert report.verdict == "blocked"
    rules = _rules(report)
    assert rules == {"syntax_error"}
    assert report.findings[0].severity == "critical"
    assert report.findings[0].line >= 1


# ---- analyze() pure structural -----------------------------------------


def test_analyze_returns_findings_without_mode():
    findings = analyze("import subprocess\nopen('/etc/x', 'w')")
    severities = {f.severity for f in findings}
    assert "critical" in severities
    assert "warn" in severities
    assert {f.rule for f in findings} == {"import.blocked", "fs.write_outside_allowed"}


def test_analyze_records_line_numbers():
    code = src(
        """
        x = 1
        y = 2
        import subprocess
        """
    )
    findings = analyze(code)
    assert len(findings) == 1
    assert findings[0].line == 3


# ---- mixed findings -----------------------------------------------------


def test_critical_plus_warn_returns_blocked():
    code = src(
        """
        import subprocess
        open('/etc/x', 'w')
        """
    )
    report = evaluate(code, mode="safe")
    assert report.verdict == "blocked"
    assert len(report.critical) == 1
    assert len(report.warnings) == 1


# ---- evaluate_with_settings --------------------------------------------


def test_evaluate_with_settings_uses_settings_mode(monkeypatch):
    from aibridge_houdini.config import Settings

    for name in ("PROVIDERS", "DEFAULT_PROVIDER", "ANTHROPIC_API_KEY", "MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abcdefgh12345678")
    monkeypatch.setenv("MODE", "direct")
    settings = Settings.load(env_file=None)

    report = evaluate_with_settings("open('/etc/x', 'w')", settings)
    assert report.mode == "direct"
    assert report.verdict == "safe"  # warn is allowed in direct


# ---- spot checks for hou patterns the spec requires to PASS -------------


def test_node_creation_and_parm_edits_are_safe():
    code = src(
        """
        import hou
        obj = hou.node('/obj')
        geo = obj.createNode('geo', 'geo1')
        sphere = geo.createNode('sphere')
        sphere.parm('rad').set(2.0)
        sphere.parmTuple('t').set((0, 1, 0))
        sphere.setName('mysphere')
        """
    )
    report = evaluate(code, mode="safe")
    assert report.verdict == "safe"
    assert report.findings == ()
