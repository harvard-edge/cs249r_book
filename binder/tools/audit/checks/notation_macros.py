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
    from audit.protected_contexts import LineWalker
except ImportError:
    from binder.tools.audit.ledger import Issue, make_issue_id
    from binder.tools.audit.protected_contexts import LineWalker

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


_LATEX_MATH_OPEN = re.compile(r"\\begin\{(gather\*?|align\*?|equation\*?|multline\*?|aligned)\}")
_LATEX_MATH_CLOSE = re.compile(r"\\end\{(gather\*?|align\*?|equation\*?|multline\*?|aligned)\}")


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
    if (
        file_path.name.startswith("_")
        or "/frontmatter/" in p_str
        or "/_includes/" in p_str
        or "/shared/" in p_str
    ):
        return issues

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

    walker = LineWalker(content)
    in_latex_math_env = False

    for line, state, line_num in walker:
        if state.in_yaml or state.in_code_fence or state.in_html_comment or state.in_html_style_block:
            continue

        if _LATEX_MATH_OPEN.search(line):
            in_latex_math_env = True

        is_display_math = state.in_display_math or in_latex_math_env
        closing_latex_math = bool(_LATEX_MATH_CLOSE.search(line))

        if is_display_math:
            # Inside display math environment: macros are expected and required
            # Check 2: Flag raw notation in display math
            for pattern, raw_latex, macro in applicable_rules:
                if pattern.search(line):
                    issues.append(
                        Issue(
                            id=make_issue_id(CATEGORY, file_path, line_num),
                            category=CATEGORY,
                            rule=RULE,
                            rule_text=RULE_TEXT,
                            file=str(file_path),
                            line=line_num,
                            before=line.strip()[:100],
                            suggested_after=line.replace(raw_latex, macro).strip()[:100],
                            reason=f"Found raw notation '{raw_latex}'. Use the canonical macro '{macro}' instead.",
                            auto_fixable=True,
                            status="open",
                            confidence="high",
                        )
                    )
        else:
            # Outside display math: prose, table rows, captions
            # Check 1: Flag any macros sneaking directly into prose/tables outside math mode
            clean_line = re.sub(r"\\\$", "DOLLAR", line)
            clean_line = re.sub(r"\$\$[^\$\n]+?\$\$", "", clean_line)
            clean_line = re.sub(r"\$[^\$\n]+?\$", "", clean_line)
            clean_line = re.sub(r"\\\(.*?\\\)", "", clean_line)
            clean_line = re.sub(r"`[^`\n]+`", "", clean_line)
            for _, _, macro in applicable_rules:
                m_pat = re.compile(r"(?<![A-Za-z0-9\\])" + re.escape(macro) + r"(?![A-Za-z0-9])")
                if m_pat.search(clean_line):
                    issues.append(
                        Issue(
                            id=make_issue_id(CATEGORY, file_path, line_num),
                            category=CATEGORY,
                            rule="No notation macros outside math mode",
                            rule_text="LaTeX macros must be enclosed within math mode ($...$ or $$...$$). Bare macros in prose or tables will be dropped by Pandoc in HTML builds.",
                            file=str(file_path),
                            line=line_num,
                            before=line.strip()[:100],
                            suggested_after=f"${macro}$",
                            reason=f"Found macro '{macro}' outside math delimiters. Enclose in $...$ or use plain text.",
                            auto_fixable=False,
                            status="open",
                            confidence="high",
                        )
                    )

            # Check 2: Flag raw notation in inline math spans on this line
            math_spans = _extract_math_spans(line)
            for math_str in math_spans:
                for pattern, raw_latex, macro in applicable_rules:
                    if pattern.search(math_str):
                        issues.append(
                            Issue(
                                id=make_issue_id(CATEGORY, file_path, line_num),
                                category=CATEGORY,
                                rule=RULE,
                                rule_text=RULE_TEXT,
                                file=str(file_path),
                                line=line_num,
                                before=line.strip()[:100],
                                suggested_after=line.replace(raw_latex, macro).strip()[:100],
                                reason=f"Found raw notation '{raw_latex}'. Use the canonical macro '{macro}' instead.",
                                auto_fixable=True,
                                status="open",
                                confidence="high",
                            )
                        )

        if closing_latex_math:
            in_latex_math_env = False

    return issues
