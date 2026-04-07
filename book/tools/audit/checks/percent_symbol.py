"""Check: '%' symbol in body prose should be 'percent'.

Rule: book-prose-merged.md section 2 + section 10.2
    Spell out "percent" in body prose; '%' is OK inside tables, equations
    (`$60\\%$`), code blocks, and figure captions.

Two forms to detect:
  1. Hard-coded: '94%' -> '94 percent'
  2. Inline Python: '`{python} val_str`%' -> '`{python} val_str` percent'

Auto-fixable: yes. Mirrors book/tools/scripts/content/fix_percent.py.

Protected contexts this check skips (all via the scanner framework):
- Code fences, YAML, display math, HTML comments
- Inline math (`$...$`), inline code, \\index{}, attribute values
- Table rows (the % convention is "OK inside tables")
- fig-cap / fig-alt attribute values (part of protected attrs)
- LaTeX \\% escapes

Note: The pass 03 sweep already did vol1 (1,275) and vol2 (1,375). This
check exists to catch any stragglers that escaped or were re-introduced.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    default_line_skip,
    inline_protected_spans,
    is_table_row,
    position_in_spans,
)


def _is_html_line(line: str) -> bool:
    """True if this line is HTML markup (not body prose).

    Mirrors fix_percent.py:39-45. HTML tags like <td width="20%"> appear
    in acknowledgements and some content-visible blocks. Percent symbols
    inside HTML attribute values are NOT body prose and must not be
    converted.
    """
    stripped = line.strip()
    if stripped.startswith("<") and not stripped.startswith("<!--"):
        return True
    # HTML attributes with quoted or unquoted values: width="70%", width=100%
    if re.search(r'(?:width|height|style)\s*=\s*"?[^"\s]*%', line):
        return True
    # Quarto image/figure width hints: ![](path){width="70%"} or {width=100%}
    if re.search(r'\{[^}]*width\s*=\s*"?[^"\s}]*%', line):
        return True
    return False

CATEGORY = "percent-symbol"
RULE = "book-prose-merged.md section 2 + 10.2"
RULE_TEXT = "Spell out 'percent' in body prose; '%' only in tables/equations/code"

# Pattern 1: inline Python followed by %
#   `{python} something`% -> `{python} something` percent
# The backtick-python expression is itself an inline-code span, so the
# protected_spans detector already covers the expression. We need to match
# the trailing % that follows the closing backtick.
_INLINE_PY_PCT_RE = re.compile(r"(`\{python\}[^`\n]+`)%")

# Pattern 2: hard-coded number + %
#   94% -> 94 percent ; 94.5% -> 94.5 percent
# Negative lookbehind for \\ avoids \% (LaTeX escaped percent).
# Negative lookbehind for word-char avoids URL fragments like "#abc%".
_HARDCODED_PCT_RE = re.compile(r"(?<![\\\w])(\d+(?:\.\d+)?)%")


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for bare '%' symbols in body prose.

    Returns (issues, next_counter).
    """
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue
        if is_table_row(line):
            continue
        if _is_html_line(line):
            continue
        if "%" not in line:
            continue

        spans = inline_protected_spans(line)

        # Pass 1: inline-python-followed-by-% matches
        for m in _INLINE_PY_PCT_RE.finditer(line):
            # The `{python}...` span is protected; the % immediately after
            # is NOT protected, so we check the % position.
            pct_pos = m.end() - 1  # position of the '%' char
            if position_in_spans(pct_pos, spans):
                continue
            # Build the suggested replacement for the full match
            before_match = line[: m.start()]
            after_match = line[m.end() :]
            # Replace `{python} ...`% with `{python} ...` percent
            py_expr = m.group(1)
            new_line = before_match + py_expr + " percent" + after_match

            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=m.start(),
                    before=line,
                    suggested_after=new_line,
                    auto_fixable=True,
                    reason="inline-python %",
                )
            )
            counter += 1

        # Pass 2: hard-coded digit% matches
        for m in _HARDCODED_PCT_RE.finditer(line):
            pct_pos = m.end() - 1
            if position_in_spans(pct_pos, spans):
                continue
            # Replace "N%" with "N percent"
            num = m.group(1)
            new_line = line[: m.start()] + num + " percent" + line[m.end() :]

            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=m.start(),
                    before=line,
                    suggested_after=new_line,
                    auto_fixable=True,
                    reason=f"{num}% -> {num} percent",
                )
            )
            counter += 1

    return issues, counter
