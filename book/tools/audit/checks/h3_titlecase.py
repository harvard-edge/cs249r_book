"""Check: H3+ headings in title case (should be sentence case).

Rule: book-prose-merged.md section 10.9
    H1 and H2: headline style (capitalize principal words)
    H3 and below: sentence style (first word + proper nouns only)

Detection: any heading at H3/H4/H5/H6 where more than one word starts with
a capital letter (excluding the first word). Proper-noun detection is NOT
safe to automate — "ResNet Architecture" should become "ResNet architecture"
but "Hardware Balance" should become "Hardware balance". A subagent must
review each one.

Auto-fixable: NO. Every issue is marked needs_subagent=True.

The section header slug (e.g. {#sec-foo-bar-1234}) is not considered part
of the heading text for case-analysis purposes; we strip it before
counting caps.

Protected contexts this check skips:
- Code fences, YAML, display math, HTML comments (default line skip)
- H1 and H2 headings (the rule only applies to H3+)
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    default_line_skip,
    heading_level,
)

CATEGORY = "h3-titlecase"
RULE = "book-prose-merged.md section 10.9"
RULE_TEXT = "H3+ headings use sentence case (first word + proper nouns only)"

# Extract the heading text (without the leading ### and without any
# trailing {#sec-...} anchor ID or {.unnumbered} class.
_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}\s+")
_HEADING_SUFFIX_RE = re.compile(r"\s*\{[^}]*\}\s*$")


def _heading_text(line: str) -> str:
    """Return the heading text with prefix hashes and trailing attrs stripped."""
    text = _HEADING_PREFIX_RE.sub("", line)
    text = _HEADING_SUFFIX_RE.sub("", text)
    return text.rstrip()


# Words that should stay lowercase in sentence case (articles, conjunctions,
# short prepositions). These are the "minor words" that title case would
# also lowercase, so they are neutral for title-case detection.
_TITLE_CASE_SKIP = {
    "a", "an", "and", "the", "of", "in", "on", "at", "to", "for",
    "or", "but", "nor", "so", "yet", "as", "by", "vs", "vs.",
    "from", "with", "into", "onto", "upon", "over", "under",
    "via", "per",
}

# Word pattern: starts with a letter, may contain letters/digits/hyphens.
# Apostrophes split words (e.g. "Sutton's" is "Sutton" + "'s" — we treat the
# whole thing as one word since the apostrophe is inside).
_WORD_RE = re.compile(r"[A-Za-z][\w'-]*")


def _looks_titlecase(text: str) -> bool:
    """Return True if the heading text appears to be in title case.

    Heuristic: after the first word, count how many remaining words start
    with a capital. Skip "minor words" (articles, conjunctions, short
    prepositions) — those are lowercase in both title and sentence case.
    If two or more of the remaining content words are capitalized, this
    is title case.

    This is the H3 detection heuristic from the Pass 15 plan — a
    subagent then reviews each flagged heading to decide whether the
    capitals are proper nouns (preserve) or title-case (lowercase).
    """
    words = _WORD_RE.findall(text)
    if len(words) < 2:
        return False  # single-word headings are always "correct" case-wise

    cap_count = 0
    content_count = 0
    for w in words[1:]:  # skip first word (always capitalized in sentence case)
        if w.lower() in _TITLE_CASE_SKIP:
            continue
        content_count += 1
        if w[0].isupper():
            cap_count += 1

    # Two or more content words capitalized after the first word = title case
    return content_count >= 2 and cap_count >= 2


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for H3+ headings that look like title case.

    Returns (issues, next_counter). Every issue has needs_subagent=True
    because proper-noun detection is not safe to automate.
    """
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue

        level = heading_level(line)
        if level is None or level < 3:
            continue

        heading_text = _heading_text(line)
        if not _looks_titlecase(heading_text):
            continue

        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY, counter),
                category=CATEGORY,
                rule=RULE,
                rule_text=RULE_TEXT,
                file=str(file_path),
                line=line_num,
                col=0,
                before=line,
                suggested_after="",  # Subagent fills this in
                auto_fixable=False,
                needs_subagent=True,
                reason=f"H{level} heading in title case: {heading_text!r}",
            )
        )
        counter += 1

    return issues, counter
