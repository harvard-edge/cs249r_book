"""Audit Check: Enforce usage of centralized notation macros over raw LaTeX.

Scans QMD content for raw mathematical variables (like D_{\\text{vol}}, R_{\\text{peak}})
and flags them, requiring the author/LLM to use the semantic macro (\\DataVolume, \\PeakRate)
defined in mlsysim.Notation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from audit.ledger import Issue, make_issue_id
except ImportError:
    from binder.tools.audit.ledger import Issue, make_issue_id

CATEGORY = "notation-macros"
RULE = "All defined notations must use their respective LaTeX macros"
RULE_TEXT = (
    "Raw mathematical notation variables defined in mlsysim.Notation "
    "MUST be referenced using their semantic \\macro equivalent. "
    "Do not type the raw string directly; use the macro to prevent drift."
)

project_root = Path(os.environ.get("BOOK_ROOT", "."))
if not (project_root / "mlsysim").exists():
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "mlsysim"))

try:
    from mlsysim.notation import Notation
except ImportError:
    Notation = None

_BANNED_RAW: dict[str, str] = {}
if Notation:
    for var in Notation.list():
        latex = var.latex.strip()
        macro = var.macro.strip()
        # Target complex expressions (subscripts, text macros, Greek) to prevent false positives on single letters
        if "\\text{" in latex or "_" in latex or len(latex) > 2 or "\\" in latex:
            _BANNED_RAW[latex] = macro


def check(file_path: Path, content: str) -> list[Issue]:
    """Scan content for raw LaTeX strings that should be notation macros."""
    issues = []
    if not _BANNED_RAW:
        return issues

    lines = content.splitlines()
    in_python = False

    for i, line in enumerate(lines):
        # Ignore python execution cells — matplotlib uses raw latex strings
        if line.startswith("```{python}"):
            in_python = True
            continue
        if line.startswith("```") and in_python:
            in_python = False
            continue
        if in_python:
            continue

        # Ignore HTML comments or YAML frontmatter markers
        if line.strip().startswith("<!--") or line.strip() == "---":
            continue

        for raw_latex, macro in _BANNED_RAW.items():
            if raw_latex in line:
                # Ensure the macro itself isn't already used
                if macro in line:
                    continue
                issues.append(
                    Issue(
                        id=make_issue_id(CATEGORY, file_path, i + 1),
                        category=CATEGORY,
                        rule=RULE,
                        rule_text=RULE_TEXT,
                        file=str(file_path),
                        line=i + 1,
                        before=line.strip()[:100],
                        suggested_after=line.replace(raw_latex, macro).strip()[:100],
                        reason=f"Found raw notation '{raw_latex}'. Use the canonical macro '{macro}' instead.",
                        auto_fixable=True,
                        status="open",
                        confidence="high",
                    )
                )
    return issues
