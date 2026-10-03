"""Audit Check: Enforce usage of centralized notation macros over raw LaTeX.

Scans QMD content for raw mathematical variables (like D_{\\text{vol}}, R_{\\text{peak}})
and flags them, requiring the author/LLM to use the semantic macro (\\DataVolume, \\PeakRate)
defined in mlsysim.Notation.
"""
from __future__ import annotations

import os
import re
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

# Map volume -> list of (compiled_pattern, raw_latex, macro)
_VOLUME_BANNED: dict[str, list[tuple[re.Pattern, str, str]]] = {
    "core": [],
    "vol1": [],
    "vol2": [],
    "vol3": [],
    "vol4": [],
}

if Notation:
    for var in Notation.list():
        latex = var.latex.strip()
        macro = var.macro.strip()
        vol = getattr(var, "volume", "core")
        if vol not in _VOLUME_BANNED:
            vol = "core"
        # Target composite expressions (subscripts, text macros, bold vectors, superscripts)
        if "_" in latex or "\\text{" in latex or "\\mathbf{" in latex or "^" in latex:
            # Match raw latex bounded by non-alphanumeric and not preceded by backslash
            pattern = re.compile(r"(?<![A-Za-z0-9\\])" + re.escape(latex) + r"(?![A-Za-z0-9])")
            _VOLUME_BANNED[vol].append((pattern, latex, macro))


def _extract_math_spans(text: str) -> list[str]:
    """Extract math strings from inline ($...$) and display ($$...$$) math."""
    spans = []
    # Display math
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL):
        spans.append(m.group(1))
    # Remove display math so inline math regex does not clash
    clean = re.sub(r"\$\$(.+?)\$\$", "", text, flags=re.DOTALL)
    for m in re.finditer(r"\$([^$\n]+?)\$", clean):
        spans.append(m.group(1))
    return spans


def check(file_path: Path, content: str) -> list[Issue]:
    """Scan content for raw LaTeX strings in math mode that should be notation macros."""
    issues = []
    
    # Determine which volume rules apply
    p_str = str(file_path)
    applicable_rules: list[tuple[re.Pattern, str, str]] = list(_VOLUME_BANNED["core"])
    for vol in ["vol1", "vol2", "vol3", "vol4"]:
        if f"/{vol}/" in p_str or p_str.startswith(f"{vol}/"):
            applicable_rules.extend(_VOLUME_BANNED[vol])
            break
    else:
        # If unknown (e.g. test.qmd), include all rules
        for vol in ["vol1", "vol2", "vol3", "vol4"]:
            applicable_rules.extend(_VOLUME_BANNED[vol])

    if not applicable_rules:
        return issues

    lines = content.splitlines()
    in_python = False
    in_yaml = False

    for i, line in enumerate(lines):
        # Handle YAML frontmatter
        if i == 0 and line.strip() == "---":
            in_yaml = True
            continue
        if in_yaml:
            if line.strip() == "---":
                in_yaml = False
            continue

        # Ignore python execution cells — matplotlib uses raw latex strings
        if line.startswith("```{python}"):
            in_python = True
            continue
        if line.startswith("```") and in_python:
            in_python = False
            continue
        if in_python:
            continue

        # Ignore HTML comments
        if line.strip().startswith("<!--"):
            continue

        math_spans = _extract_math_spans(line)
        if not math_spans:
            continue

        for math_str in math_spans:
            for pattern, raw_latex, macro in applicable_rules:
                if pattern.search(math_str):
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
