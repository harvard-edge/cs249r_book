"""Check: capitalized 'Chapter 12', 'Section 3.2' etc. in body prose.

Rule: book-prose-merged.md section 10.4
    Lowercase chapter/section/figure/table words when written by hand.
    Quarto's @sec-/@fig-/@tbl-/@eq- references handle this automatically.

Detection: "Chapter 12", "Section 3.2", "Figure 1.1", "Table 5",
"Equation 2.4" — written BY HAND (capitalized), not via @ref.

Auto-fixable: yes. The replacement is lowercase the first letter of the
word while keeping the number.

Protected contexts this check skips:
- All default-line skips (code fences, YAML, math, HTML comments)
- Inline math, inline code, attribute values, \\index{}, @refs, citations
- Section headers (the rule does not apply to the heading words themselves,
  only to references IN body prose to other numbered items)
- Sentence start (would require capital anyway)
- Any bare @sec-/@fig-/@tbl-/@eq- reference — those are already
  lowercased by Quarto rendering.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    default_line_skip,
    heading_level,
    inline_protected_spans,
    is_sentence_start,
    position_in_spans,
)

CATEGORY = "lowercase-prose-references"
RULE = "book-prose-merged.md section 10.4"
RULE_TEXT = (
    "Lowercase 'chapter/section/figure/table/equation' when writing the "
    "word by hand (not via @sec-/@fig-/@tbl-/@eq- references)"
)

# Words that should be lowercase in prose when followed by a number.
# Map of capitalized form -> lowercase form.
_WORDS = {
    "Chapter": "chapter",
    "Section": "section",
    "Figure": "figure",
    "Table": "table",
    "Equation": "equation",
}

# Match "<Word> <number>" where number may be "12", "3.2", "1.1.2", etc.
# Use lookahead to avoid consuming the number so position is exact.
_REF_RE = re.compile(
    r"\b(" + "|".join(_WORDS.keys()) + r")\b(?=\s+\d)"
)


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for capitalized prose references like 'Chapter 12'.

    Returns (issues, next_counter).
    """
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue
        # Skip section headers — the rule applies to references IN body prose,
        # not to headings that might themselves contain "Section 3.2" as a
        # title (which is headline case anyway at H1/H2).
        if heading_level(line) is not None:
            continue

        # Quick bail
        if not any(w in line for w in _WORDS):
            continue

        spans = inline_protected_spans(line)
        edits: list[tuple[int, int, str, str]] = []

        for m in _REF_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            if is_sentence_start(line, m.start()):
                continue  # "Section 3.2 describes..." at sentence start is fine
            matched = m.group(1)
            edits.append((m.start(), m.end(), matched, _WORDS[matched]))

        if not edits:
            continue

        # Apply all edits in reverse to build suggested_after
        new_line = line
        for start, end, _b, after in reversed(edits):
            new_line = new_line[:start] + after + new_line[end:]

        for start, end, before_word, after_word in edits:
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
                    reason=f"{before_word} -> {after_word}",
                )
            )
            counter += 1

    return issues, counter
