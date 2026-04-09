"""Check: §10.12 alt-text must follow §10.3 concept-term rules.

Rule: book-prose-merged.md section 10.12

    "Alt-text strings live inside `fig-alt="..."` attributes and are
    easy to forget during prose passes — but they ARE body prose for
    screen readers. The same Section 1, 2, and 10 prose rules apply:

      - Concept terms lowercase (matching the body prose lowercase
        list in §10.3): 'machine learning' not 'Machine Learning';
        'memory wall' not 'Memory Wall'; 'data-centric AI' not
        'Data-Centric AI'.
      - Hyphenation matches the body-prose compound rules (§10.8)
      - `vs.` with period
      - Numbers: spell out 1-9, digits for 10+, units always digits"

This first implementation of the alt-text check focuses on the
HIGHEST-VALUE §10.12 sub-rule: concept-term lowercasing. The term
list is imported from
`audit.checks.concept_term_capitalization._TERMS` — sharing the
canonical §10.3 list keeps the two checks in sync.

The check scans every `fig-alt="..."` attribute value in the
source. Any capitalized concept term inside alt-text is flagged
EXCEPT the first word of the alt-text (which is correctly
capitalized as a sentence start) and per-term special cases
ported from `concept_term_capitalization`:

  - "Iron Law of Processor Performance" (H&P canonical reference)
  - "Bitter Lesson" when inside quotes or near "Sutton"

Other §10.12 sub-rules (hyphenation, numbers, `vs.`) are left as
follow-up items — they each require distinct regex patterns and
the book's existing alt-text pass (round 1 pass 7) already
addressed many of them.

Auto-fixable? Technically yes — the substitution is deterministic.
But fixing alt-text is delicate (wrong substitutions break screen-
reader output) so the fix is marked `auto_fixable=False,
needs_subagent=True` for conservative editorial review.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.checks.concept_term_capitalization import (
    _TERMS as _CONCEPT_TERMS,
    _is_hp_reference,
    _is_sutton_essay_title,
)
from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import LineWalker

CATEGORY = "alt-text-style"
RULE = "book-prose-merged.md section 10.12"
RULE_TEXT = "Alt-text strings must follow §10.3 concept-term lowercase rule"


# Matches `fig-alt="<contents>"` allowing escaped inner double-quotes.
# Group 1 is the contents between the quotes. The start position of the
# match equals the start of `fig-alt="`; the start of group 1 is the
# first character of the attribute value. Non-greedy.
_FIG_ALT_RE = re.compile(r'fig-alt="((?:[^"\\]|\\.)*?)"')


def _skip_line_for_alt(line: str, state) -> bool:
    """Line-level filter. We do NOT skip div fences (`:::`) because
    `fig-alt="..."` typically appears ON a div opener line. But we do
    skip block-level protected contexts."""
    if state.in_yaml or state.in_code_fence or state.in_display_math:
        return True
    if state.in_html_style_block or state.in_html_comment:
        return True
    return False


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan a file for §10.12 alt-text concept-term violations.

    For each `fig-alt="..."` attribute value, find every capitalized
    §10.3 concept term inside the alt-text content. Skip matches at
    the first-word position (sentence start) and matches that are
    protected by the per-term special cases (H&P reference, Sutton
    essay title).
    """
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if _skip_line_for_alt(line, state):
            continue

        for alt_match in _FIG_ALT_RE.finditer(line):
            alt_content = alt_match.group(1)
            alt_value_start = alt_match.start(1)

            # Leading whitespace inside the attribute value: the
            # "first word position" is the first non-whitespace char.
            first_word_pos = len(alt_content) - len(alt_content.lstrip())

            for term, replacement in _CONCEPT_TERMS.items():
                offset = 0
                while True:
                    idx = alt_content.find(term, offset)
                    if idx == -1:
                        break
                    end = idx + len(term)
                    # Skip if this is the first word of the alt-text
                    # (sentence-start exception).
                    if idx == first_word_pos:
                        offset = end
                        continue
                    # Per-term special cases (use same predicates as
                    # concept_term_capitalization for consistency).
                    if term == "Iron Law" and _is_hp_reference(alt_content, end):
                        offset = end
                        continue
                    if term == "Bitter Lesson" and _is_sutton_essay_title(alt_content):
                        offset = end
                        continue
                    # Absolute column in the full source line.
                    line_col = alt_value_start + idx
                    issues.append(
                        Issue(
                            id=make_issue_id(scope, CATEGORY, counter),
                            category=CATEGORY,
                            rule=RULE,
                            rule_text=RULE_TEXT,
                            file=str(file_path),
                            line=line_num,
                            col=line_col,
                            before=line,
                            suggested_after="",  # subagent fills
                            auto_fixable=False,
                            needs_subagent=True,
                            reason=f"alt-text: {term} -> {replacement}",
                        )
                    )
                    counter += 1
                    offset = end

    return issues, counter


# ── Adversarial self-test ──────────────────────────────────────────────────
#
# Each test case is (name, one_line_source, expected_hit_count).

_TESTS = [
    # Positive: concept term in alt-text body
    (
        "memory wall mid-sentence",
        '::: {#fig-x fig-alt="Diagram of the Memory Wall constraint."}',
        1,
    ),
    (
        "multiple terms in one alt-text",
        '::: {#fig-y fig-alt="The Iron Law and Memory Wall together."}',
        2,  # Iron Law + Memory Wall (plus "Iron" at pos 4 is not first-word)
    ),
    # Negative: concept term at sentence start of alt-text
    (
        "memory wall at sentence start — protected",
        '::: {#fig-a fig-alt="Memory Wall constraint in the roofline model."}',
        0,
    ),
    # Negative: alt-text contains the correct lowercase form
    (
        "correctly lowercased alt-text",
        '::: {#fig-b fig-alt="Diagram of the iron law decomposition."}',
        0,
    ),
    # Negative: the H&P canonical reference must stay capitalized
    (
        "Iron Law of Processor Performance preserved",
        '::: {#fig-c fig-alt="Comparison with the Iron Law of Processor Performance."}',
        0,
    ),
    # Negative: no fig-alt attribute on this line
    (
        "no fig-alt on line",
        'Regular body prose with Iron Law that should not be flagged here.',
        0,
    ),
    # Negative: fig-cap has the term (only fig-alt is checked in this item)
    (
        "term in fig-cap is not flagged by this check",
        '::: {#fig-d fig-cap="**Memory Wall**: the constraint." fig-alt="Diagram only."}',
        0,
    ),
]


def _self_test() -> int:
    from audit.protected_contexts import LineState

    state = LineState()
    pos_fail: list[str] = []

    def count_hits(line: str) -> int:
        if _skip_line_for_alt(line, state):
            return 0
        hits = 0
        for alt_match in _FIG_ALT_RE.finditer(line):
            alt_content = alt_match.group(1)
            first_word_pos = len(alt_content) - len(alt_content.lstrip())
            for term in _CONCEPT_TERMS:
                offset = 0
                while True:
                    idx = alt_content.find(term, offset)
                    if idx == -1:
                        break
                    end = idx + len(term)
                    if idx == first_word_pos:
                        offset = end
                        continue
                    if term == "Iron Law" and _is_hp_reference(alt_content, end):
                        offset = end
                        continue
                    if term == "Bitter Lesson" and _is_sutton_essay_title(alt_content):
                        offset = end
                        continue
                    hits += 1
                    offset = end
        return hits

    for name, line, expected in _TESTS:
        got = count_hits(line)
        if got != expected:
            pos_fail.append(f"{name}: expected {expected}, got {got}")

    total = len(_TESTS)
    passed = total - len(pos_fail)
    print(f"alt_text_style self-test: {passed}/{total} passed")
    for f in pos_fail:
        print(f"  - {f}")
    return 0 if not pos_fail else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_self_test())
