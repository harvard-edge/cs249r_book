"""Check: binary units (GiB, TiB, KiB, MiB) in body prose.

Rule: book-prose-merged.md section 1
    "Never binary units in prose. Write `GB` and `TB`, not `GiB` or `TiB`."

Auto-fixable: NO. The fix depends on whether the source value was a binary
or decimal quantity. A GiB -> GB conversion changes the number by ~7%. The
correct fix requires either re-reading the spec source or leaving the
value as-is and just changing the unit label (which produces a slightly
incorrect number). We emit as 'needs_subagent=False, auto_fixable=False'
so the operator has to hand-review each case.

Protected contexts: code fences, YAML, math, HTML comments, inline code,
and tables (tables sometimes legitimately show GiB in spec-citation rows).
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

CATEGORY = "binary-units-in-prose"
RULE = "book-prose-merged.md section 1"
RULE_TEXT = "Never use binary units (GiB, TiB, KiB, MiB) in body prose"

# Match KiB/MiB/GiB/TiB/PiB/EiB when preceded by a digit, possibly with a space
_BINARY_UNIT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*([KMGTPE]iB)\b")


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for binary unit suffixes in body prose."""
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue
        if is_table_row(line):
            continue
        if "iB" not in line:
            continue

        spans = inline_protected_spans(line)

        for m in _BINARY_UNIT_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            unit = m.group(1)

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
                    suggested_after="",  # No automatic fix; needs review
                    auto_fixable=False,
                    needs_subagent=False,  # Needs human judgment, not subagent
                    reason=f"{unit} in body prose (needs manual review)",
                )
            )
            counter += 1

    return issues, counter
