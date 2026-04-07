"""Check: bare 'vs' should be 'vs.' (with period).

Rule: book-prose-merged.md section 1 + section 10.10
    "vs. always with period" — never "versus" or bare "vs".

Auto-fixable: yes. The proven regex from pass 10b (commit c3e328c0b):
    only replace when bracketed by single space + alphabetic word on both
    sides, to avoid touching hyphenated compounds like "latency-vs-
    architecture" or anchor IDs.

Protected contexts this check skips:
- Code fences, YAML, display math (via LineWalker)
- Inline math, inline code, \\index{}, attribute values, @refs, citations,
  footnote refs, anchor IDs (via inline_protected_spans)
- Table rows (tables use bold + pipe formatting that doesn't need vs. fix)

Notes:
- "VS" (all caps) is NOT flagged — likely an acronym, not a comparison word
- "vs." (already correct) is NOT flagged
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

CATEGORY = "vs-period"
RULE = "book-prose-merged.md section 1 + 10.10"
RULE_TEXT = "'vs.' always with period — never 'versus' or bare 'vs'"

# Match " vs " with single spaces on both sides, with alphabetic word before
# and after. Uses look-behind/look-ahead so the captured span is exactly "vs".
# - (?<= [a-zA-Z]) : preceded by alphabetic char then space then "vs"
# - \bvs\b         : the literal "vs" as a whole word
# - (?= [a-zA-Z])  : followed by space then alphabetic char
# The actual match starts at the "v" so we know the exact replacement position.
_VS_RE = re.compile(r"(?<=[a-zA-Z] )\bvs\b(?= [a-zA-Z])")


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for bare 'vs' in body prose.

    Returns (issues, next_counter).
    """
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue
        # Tables also get skipped — pipe-table cells that contain "foo vs bar"
        # are very rare, and if present, they are edge cases for manual review.
        if is_table_row(line):
            continue
        if "vs" not in line:
            continue

        spans = inline_protected_spans(line)

        for m in _VS_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue

            # Build the suggested_after: replace "vs" with "vs."
            new_line = line[: m.start()] + "vs." + line[m.end() :]

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
                )
            )
            counter += 1

    return issues, counter
