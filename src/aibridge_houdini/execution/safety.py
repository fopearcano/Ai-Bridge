"""Static safety check for LLM-produced Houdini Python.

This module never executes code. It parses the source via :mod:`ast` and
classifies each suspicious construct as either ``warn`` or ``critical``,
then applies the global ``MODE`` to derive a verdict:

    safe                  - go ahead and execute
    needs_confirmation    - run only after the user explicitly approves
    blocked               - never run

Policy:
    * ``critical`` findings always escalate to ``blocked``.
    * ``warn`` findings escalate based on MODE:
        - ``safe``   -> needs_confirmation
        - ``dev``    -> safe (warnings are logged)
        - ``direct`` -> safe (warnings are logged)

Because Houdini scripts legitimately use ``hou`` for node creation, parm
edits, etc., none of the ``hou.*`` surface is flagged. Findings target
host-machine side effects: filesystem destruction, shells, network
connections, and meta-programming inside generated code.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Literal, Sequence

from aibridge_houdini.config import Mode, Settings


log = logging.getLogger("aibridge.execution.safety")


Verdict = Literal["safe", "needs_confirmation", "blocked"]
Severity = Literal["warn", "critical"]


# ---- policy data --------------------------------------------------------


# Always-critical imports (any submodule too).
BLOCKED_IMPORTS: frozenset[str] = frozenset(
    {
        "subprocess",
        "socket",
        "requests",
        "httpx",
        "aiohttp",
        "paramiko",
        "ftplib",
        "smtplib",
        "telnetlib",
        "pickle",  # arbitrary code execution on load
    }
)

# Critical sub-paths (matched as prefix; "urllib.parse" stays allowed).
BLOCKED_IMPORT_PREFIXES: frozenset[str] = frozenset(
    {
        "urllib.request",
        "urllib.error",
        "http.client",
        "http.server",
        "xmlrpc",
    }
)

# Fully-qualified calls that are always critical.
BLOCKED_QUALIFIED_CALLS: frozenset[str] = frozenset(
    {
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.removedirs",
        "os.system",
        "os.popen",
        "os.execv",
        "os.execvp",
        "os.execvpe",
        "os.spawnl",
        "os.spawnv",
        "os.spawnvp",
        "os.kill",
        "shutil.rmtree",
        "pathlib.Path.unlink",  # rare AST shape, kept for completeness
    }
)

# Bare-name builtins that are always critical when called.
BLOCKED_BUILTIN_CALLS: frozenset[str] = frozenset(
    {"eval", "exec", "compile", "__import__"}
)

# Method names that, regardless of receiver, indicate destructive intent.
BLOCKED_METHOD_NAMES: frozenset[str] = frozenset(
    {"unlink", "rmdir", "rmtree"}
)

# File writes whose path doesn't start with one of these are a `warn`.
DEFAULT_ALLOWED_PATH_PREFIXES: tuple[str, ...] = (
    "$HIP",
    "$JOB",
    "$TEMP",
    "$HOME",
    "$HOUDINI_TEMP_DIR",
)

# Path-write methods on file/path-like objects (write_text, write_bytes, write).
WRITE_METHOD_NAMES: frozenset[str] = frozenset(
    {"write_text", "write_bytes"}
)


# ---- result types -------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    severity: Severity
    rule: str
    message: str
    line: int


@dataclass(frozen=True)
class SafetyReport:
    verdict: Verdict
    mode: Mode
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def critical(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "critical")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "warn")


# ---- public api ---------------------------------------------------------


def analyze(
    code: str,
    allowed_path_prefixes: Sequence[str] | None = None,
) -> tuple[Finding, ...]:
    """Run the structural analysis. Mode policy is NOT applied here."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (
            Finding(
                severity="critical",
                rule="syntax_error",
                message=f"could not parse code: {e.msg}",
                line=e.lineno or 0,
            ),
        )
    visitor = _Visitor(
        allowed_path_prefixes=tuple(
            allowed_path_prefixes
            if allowed_path_prefixes is not None
            else DEFAULT_ALLOWED_PATH_PREFIXES
        )
    )
    visitor.visit(tree)
    return tuple(visitor.findings)


def evaluate(
    code: str,
    mode: Mode,
    allowed_path_prefixes: Sequence[str] | None = None,
) -> SafetyReport:
    """Analyze ``code`` and apply ``mode`` policy to derive a Verdict."""
    findings = analyze(code, allowed_path_prefixes)
    verdict = _apply_mode(findings, mode)
    if verdict != "safe":
        log.info(
            "safety: verdict=%s mode=%s findings=%d",
            verdict,
            mode,
            len(findings),
        )
    for f in findings:
        if f.severity == "critical":
            log.warning("safety: critical %s (line %d): %s", f.rule, f.line, f.message)
        else:
            log.info("safety: warn %s (line %d): %s", f.rule, f.line, f.message)
    return SafetyReport(verdict=verdict, mode=mode, findings=findings)


def evaluate_with_settings(code: str, settings: Settings) -> SafetyReport:
    return evaluate(code, mode=settings.mode)


# ---- internals ----------------------------------------------------------


def _apply_mode(findings: Sequence[Finding], mode: Mode) -> Verdict:
    has_critical = any(f.severity == "critical" for f in findings)
    has_warn = any(f.severity == "warn" for f in findings)
    if has_critical:
        return "blocked"
    if not has_warn:
        return "safe"
    if mode == "safe":
        return "needs_confirmation"
    # dev / direct: warn but allow
    return "safe"


class _Visitor(ast.NodeVisitor):
    def __init__(self, allowed_path_prefixes: tuple[str, ...]) -> None:
        self.findings: list[Finding] = []
        self._allowed = allowed_path_prefixes

    # -- imports ------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        # `from urllib.request import urlopen` -> mod="urllib.request"
        self._check_import(mod, node.lineno)
        for alias in node.names:
            full = f"{mod}.{alias.name}" if mod else alias.name
            self._check_import(full, node.lineno)
        self.generic_visit(node)

    def _check_import(self, mod: str, line: int) -> None:
        if not mod:
            return
        if mod in BLOCKED_IMPORTS or any(
            mod == p or mod.startswith(p + ".") for p in BLOCKED_IMPORTS
        ):
            self._add(
                "critical",
                "import.blocked",
                f"import of {mod!r} is forbidden (host-side I/O / shell / network)",
                line,
            )
            return
        if any(mod == p or mod.startswith(p + ".") for p in BLOCKED_IMPORT_PREFIXES):
            self._add(
                "critical",
                "import.blocked",
                f"import of {mod!r} is forbidden (network)",
                line,
            )

    # -- calls --------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        qualified = _qualified_name(node.func)

        # eval / exec / compile / __import__
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTIN_CALLS:
            self._add(
                "critical",
                "meta.exec",
                f"{node.func.id}() is forbidden in generated code",
                node.lineno,
            )

        # os.remove(...), shutil.rmtree(...), …
        if qualified and qualified in BLOCKED_QUALIFIED_CALLS:
            self._add(
                "critical",
                "call.blocked",
                f"{qualified}(...) is forbidden",
                node.lineno,
            )

        # Method calls like x.unlink(), p.rmdir(), tree.rmtree()
        if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_METHOD_NAMES:
            self._add(
                "critical",
                "call.blocked",
                f".{node.func.attr}() is forbidden (filesystem destruction)",
                node.lineno,
            )

        # Path("...").write_text(...) / write_bytes(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr in WRITE_METHOD_NAMES:
            self._flag_write(_first_arg(node.func.value), node.lineno, f".{node.func.attr}()")

        # open("...", "w")
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self._check_open(node)

        self.generic_visit(node)

    def _check_open(self, node: ast.Call) -> None:
        mode_str = _open_mode(node)
        if mode_str is None:
            # Could not statically determine -> assume read; emit only if write modes likely.
            return
        if any(c in mode_str for c in "wax+"):
            target = node.args[0] if node.args else None
            self._flag_write(target, node.lineno, "open()")

    def _flag_write(self, path_node: ast.AST | None, line: int, fn: str) -> None:
        if path_node is None:
            self._add(
                "warn",
                "fs.write_dynamic_path",
                f"{fn} write with no static path",
                line,
            )
            return
        if isinstance(path_node, ast.Constant) and isinstance(path_node.value, str):
            value = path_node.value
            if any(value.startswith(p) for p in self._allowed):
                return
            self._add(
                "warn",
                "fs.write_outside_allowed",
                f"{fn} write to {value!r} (outside allowed prefixes "
                f"{list(self._allowed)})",
                line,
            )
            return
        self._add(
            "warn",
            "fs.write_dynamic_path",
            f"{fn} write with non-literal path",
            line,
        )

    def _add(self, severity: Severity, rule: str, message: str, line: int) -> None:
        self.findings.append(Finding(severity, rule, message, line))


def _qualified_name(node: ast.AST) -> str | None:
    """Return e.g. 'os.remove' or 'shutil.rmtree' for chained Name.Attribute paths."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _open_mode(node: ast.Call) -> str | None:
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        v = node.args[1].value
        return v if isinstance(v, str) else None
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            v = kw.value.value
            return v if isinstance(v, str) else None
    # Default open() mode is "r".
    return "r"


def _first_arg(node: ast.AST) -> ast.AST | None:
    """Best-effort recovery of the path argument from a Path("...")-style chain."""
    if isinstance(node, ast.Call) and node.args:
        return node.args[0]
    return None
