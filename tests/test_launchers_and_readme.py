"""Sanity checks for the launcher scripts and README."""

from __future__ import annotations

import os
import platform
import re
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---- launchers ---------------------------------------------------------


@pytest.mark.parametrize("name", ["run_cli.sh", "run_cli.bat", "run_ui.sh", "run_ui.bat"])
def test_launcher_exists(name):
    path = REPO_ROOT / name
    assert path.exists(), f"missing launcher: {name}"


@pytest.mark.parametrize("name", ["run_cli.sh", "run_ui.sh"])
def test_shell_launchers_are_executable(name):
    path = REPO_ROOT / name
    mode = path.stat().st_mode
    assert mode & stat.S_IXUSR, f"{name} is not executable by owner"


def test_run_cli_sh_invokes_aibridge_module():
    text = (REPO_ROOT / "run_cli.sh").read_text()
    assert "PYTHONPATH" in text and "src" in text, "must bootstrap PYTHONPATH"
    assert "-m aibridge_houdini" in text
    assert "--ui qt" not in text  # CLI launcher should not force the UI


def test_run_ui_sh_passes_ui_qt_flag():
    text = (REPO_ROOT / "run_ui.sh").read_text()
    assert "PYTHONPATH" in text and "src" in text
    assert "--ui qt" in text


def test_run_cli_bat_invokes_aibridge_module():
    text = (REPO_ROOT / "run_cli.bat").read_text()
    assert "PYTHONPATH" in text and "src" in text
    assert "-m aibridge_houdini" in text
    assert "--ui qt" not in text


def test_run_ui_bat_passes_ui_qt_flag():
    text = (REPO_ROOT / "run_ui.bat").read_text()
    assert "PYTHONPATH" in text and "src" in text
    assert "--ui qt" in text


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="bash launchers are tested on POSIX only")
@pytest.mark.parametrize("name", ["run_cli.sh", "run_ui.sh"])
def test_shell_launcher_runs_help(name):
    """The launcher must bootstrap a fresh PYTHONPATH and invoke the
    package without needing `pip install -e .`."""
    env = dict(os.environ)
    # Strip any pre-existing PYTHONPATH so we exercise the bootstrap.
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [str(REPO_ROOT / name), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert "aibridge-houdini" in proc.stdout
    # Both launchers go through the same argparse, so --ui must appear.
    assert "--ui" in proc.stdout


# ---- README ------------------------------------------------------------


@pytest.fixture(scope="module")
def readme_text() -> str:
    path = REPO_ROOT / "README.md"
    assert path.exists(), "README.md is missing"
    return path.read_text()


def test_readme_has_required_tagline(readme_text):
    """User explicitly asked for this exact framing."""
    assert "A natural-language bridge for Houdini automation" in readme_text


def test_readme_does_not_overstate_scope(readme_text):
    """Must NOT describe itself as a full AI artist."""
    text = readme_text.lower()
    # Allow the explicit denial sentence ("not an AI artist") but reject
    # any positive claim.
    bad_phrases = [
        "is an ai artist",
        "is a full ai artist",
        "ai artist that",
        "creative ai",
    ]
    for phrase in bad_phrases:
        assert phrase not in text, f"README should not claim to be {phrase!r}"
    # Positive: it should explicitly disclaim the role. Strip markdown
    # emphasis so "**not** an AI artist" still counts as a match.
    bare = re.sub(r"[*_`]", "", text)
    assert "not an ai artist" in bare


@pytest.mark.parametrize(
    "section_title",
    [
        "## Install",
        "## Configure `.env`",
        "## Run the Houdini receiver",
        "## CLI usage",
        "## UI usage",
        "## Safety notes",
    ],
)
def test_readme_has_section(readme_text, section_title):
    # Match as a markdown heading at start of line.
    pattern = rf"(?m)^{re.escape(section_title)}"
    assert re.search(pattern, readme_text), f"README missing section: {section_title}"


def test_readme_documents_provider_envs(readme_text):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LMSTUDIO_MODEL",
                "DEFAULT_PROVIDER", "PROVIDERS", "MODE", "HYTHON_PATH",
                "HOUDINI_HOST", "HOUDINI_PORT"):
        assert var in readme_text, f"README does not mention {var}"


def test_readme_documents_slash_commands(readme_text):
    for cmd in ("/provider", "/mode", "/help", "/quit"):
        assert cmd in readme_text, f"README missing command reference: {cmd}"


def test_readme_documents_safety_verdicts(readme_text):
    for verdict in ("safe", "needs_confirmation", "blocked"):
        assert verdict in readme_text


def test_readme_links_launchers(readme_text):
    for name in ("run_cli.sh", "run_cli.bat", "run_ui.sh", "run_ui.bat"):
        assert name in readme_text, f"README does not mention {name}"


def test_readme_warns_about_loopback_only(readme_text):
    """Receiver section must mention loopback / not-off-host."""
    assert "loopback" in readme_text.lower()
    assert "off-host" in readme_text.lower() or "off host" in readme_text.lower()
