"""Sanity tests for ``scripts/run_manual_test.py``.

These do NOT call any LLM — they only verify the script imports cleanly,
its prompt catalog stays in sync with ``manual_houdini_prompts.md``, and
the simple ``--list`` / ``--only`` plumbing works.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_manual_test.py"
MARKDOWN = REPO_ROOT / "tests" / "manual_houdini_prompts.md"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("manual_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["manual_runner"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner_module()


# ---- structural checks -------------------------------------------------


def test_prompts_exist_and_are_well_formed(runner):
    cases = runner.PROMPTS
    assert len(cases) == 8
    ids = [c.id for c in cases]
    assert len(set(ids)) == len(ids), "prompt ids must be unique"
    for case in cases:
        assert case.id and case.prompt
        assert isinstance(case.expected_intents, tuple) and case.expected_intents
        assert isinstance(case.expected_risks, tuple) and case.expected_risks
        for risk in case.expected_risks:
            assert risk in {"low", "medium", "high"}, risk


def test_prompts_cover_required_categories(runner):
    """The user explicitly asked for these 8 categories."""
    expected = {
        "sphere",
        "procedural_rock",
        "desert_scatter",
        "explosion_setup",
        "camera",
        "scale_selected",
        "assign_material",
        "network_organization",
    }
    actual = {c.id for c in runner.PROMPTS}
    assert actual == expected


def test_markdown_documents_every_prompt(runner):
    md = MARKDOWN.read_text()
    assert MARKDOWN.exists()
    # Collapse all internal whitespace so we can match against the
    # human-formatted (line-wrapped) markdown.
    md_collapsed = re.sub(r"\s+", " ", md)
    for case in runner.PROMPTS:
        # The markdown lists each id as `**Id:** \`<id>\``.
        pattern = rf"\*\*Id:\*\*\s*`{re.escape(case.id)}`"
        assert re.search(pattern, md), (
            f"manual_houdini_prompts.md is missing a section for id={case.id!r}"
        )
        # And the verbatim prompt text must appear (whitespace-normalized).
        prompt_collapsed = re.sub(r"\s+", " ", case.prompt).strip()
        assert prompt_collapsed in md_collapsed, (
            f"prompt text missing in markdown for id={case.id!r}: {case.prompt!r}"
        )


def test_markdown_mentions_runner_script(runner):
    md = MARKDOWN.read_text()
    assert "scripts/run_manual_test.py" in md


# ---- CLI plumbing ------------------------------------------------------


def test_list_prints_all_ids(runner, capsys):
    rc = runner.main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    for case in runner.PROMPTS:
        assert case.id in out


def test_only_unknown_id_is_an_error(runner, capsys):
    rc = runner.main(["--only", "no-such-id"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown prompt id" in err
    assert "sphere" in err  # the available list should mention real ids


# ---- formatters --------------------------------------------------------


def test_short_truncates_long_text(runner):
    out = runner._short("a" * 200, width=40)
    assert len(out) <= 41  # "…" appended
    assert out.endswith("…")


def test_short_passes_short_text_unchanged(runner):
    assert runner._short("hello") == "hello"
