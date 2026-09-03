"""Check: 'Acknowledgements' -> 'Acknowledgments' (American spelling).

Rule: book-prose-merged.md section 10.7 / 10.15

Per Webster's 11th: Acknowledgments (American) is the canonical form.
The QMD filename can stay as acknowledgements.qmd (path not rendered);
only the rendered heading text needs to change.

Auto-fixable: yes. Trivial one-line fix per file.

Round-1 completion status: vol1 + vol2 done in pass 10b.
This check exists to prevent regression and to catch any stragglers in
other files (dedication, about, chapter-connection callouts, etc.)
that reference the word.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    default_line_skip,
    inline_protected_spans,
    position_in_spans,
)

CATEGORY = "acknowledgements-spelling"
RULE = "book-prose-merged.md section 10.7 + 10.15"
RULE_TEXT = "Use 'Acknowledgments' (American) not 'Acknowledgements' (British)"

_WORD_RE = re.compile(r"\bAcknowledgements\b")


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for the British spelling in any non-protected context."""
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue
        if "Acknowledgements" not in line:
            continue

        spans = inline_protected_spans(line)

        for m in _WORD_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            new_line = line[: m.start()] + "Acknowledgments" + line[m.end() :]

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
                    reason="British -> American spelling",
                )
            )
            counter += 1

    return issues, counter
