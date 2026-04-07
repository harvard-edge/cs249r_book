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

        # Collect all matches first, then build a single fully-fixed
        # suggested_after for the whole line. This is required because
        # the applier replaces lines wholesale: emitting multiple issues
        # with partial suggested_after values would only apply ONE of
        # them per line. Use the compound_prefix.py pattern: collect
        # edits as (start, end, replacement) tuples, apply in reverse
        # order to keep indices stable, emit one Issue per match all
        # sharing the same fully-fixed line.

        edits: list[tuple[int, int, str, str]] = []  # (start, end, before, after)

        # Pass 1: inline-python-followed-by-% matches
        for m in _INLINE_PY_PCT_RE.finditer(line):
            pct_pos = m.end() - 1  # position of the '%' char
            if position_in_spans(pct_pos, spans):
                continue
            py_expr = m.group(1)
            before_text = m.group(0)
            after_text = py_expr + " percent"
            edits.append((m.start(), m.end(), before_text, after_text))

        # Pass 2: hard-coded digit% matches
        for m in _HARDCODED_PCT_RE.finditer(line):
            pct_pos = m.end() - 1
            if position_in_spans(pct_pos, spans):
                continue
            num = m.group(1)
            before_text = m.group(0)
            after_text = f"{num} percent"
            edits.append((m.start(), m.end(), before_text, after_text))

        if not edits:
            continue

        # Sort by position (so the reverse-application order is stable)
        edits.sort(key=lambda e: e[0])

        # Build the fully-fixed line by applying all edits in reverse order
        new_line = line
        for start, end, _b, after in reversed(edits):
            new_line = new_line[:start] + after + new_line[end:]

        # Emit one Issue per edit, all sharing the same suggested_after
        for start, end, before_text, after_text in edits:
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=line_num,
                    col=start,
                    before=line,
                    suggested_after=new_line,
                    auto_fixable=True,
                    reason=f"{before_text} -> {after_text}",
                )
            )
            counter += 1

    return issues, counter
