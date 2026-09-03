"""Check: pre-/non- compound prefix close-up.

Rule: book-prose-merged.md section 10.8

Close up to remove the hyphen — STRICT 6-term list (no extrapolation):

    pre-training   -> pretraining
    pre-trained    -> pretrained
    pre-deployment -> predeployment
    pre-learning   -> prelearning
    pre-processing -> preprocessing
    non-zero       -> nonzero

The discarded bulk-edit run extrapolated this list to dozens of additional
terms (pretransformer, nontensor, noncompute, preallocate, ...). Per
section 10.8 those are explicitly out of scope: "Would a senior ML/systems
engineer pause when reading the closed-up form? If yes, keep the hyphen."

Domain compounds that MUST keep their hyphens (also section 10.8):
    multi-layer, multi-chip, multi-server, multi-model, multi-stream,
    multi-scale, semi-supervised, anti-pattern

This check does NOT flag those.

Always-keep-hyphen contexts (section 10.8):
- Before an acronym or proper noun: multi-GPU, multi-NUMA, pre-CUDA
- Before a number or symbol: pre-2010, multi-$100$
- When close-up would be unreadable: re-cover, re-create, pre-eminent

Auto-fixable: yes. The proven pass 10b implementation is 6 literal
substring replacements with the same protected-context skip list as
vs_period.

Protected contexts this check skips:
- Code fences, YAML, display math, HTML comments (via LineWalker)
- Inline math, inline code, \\index{}, attribute values, @refs, citations,
  footnote refs, anchor IDs (via inline_protected_spans)
- Section header SLUGS (e.g. {#sec-pre-training-bd45}) — anchor IDs are
  permanent and protected separately by inline spans
- Table rows
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

CATEGORY = "compound-prefix-closeup"
RULE = "book-prose-merged.md section 10.8"
RULE_TEXT = (
    "Close up only the 6 listed pre-/non- forms; "
    "domain compounds (multi-/semi-/anti-) keep their hyphens"
)

# The strict 6-term list. Each entry is (hyphenated, closed-up).
# Pattern matches case-insensitively for the prefix but preserves the case
# of the matched word so "Pre-Training" -> "Pretraining" and
# "PRE-TRAINING" -> "PRETRAINING".
_TERMS = [
    ("pre-training", "pretraining"),
    ("pre-trained", "pretrained"),
    ("pre-deployment", "predeployment"),
    ("pre-learning", "prelearning"),
    ("pre-processing", "preprocessing"),
    ("non-zero", "nonzero"),
]


def _build_regex() -> re.Pattern[str]:
    """Build a single regex matching any of the 6 terms case-insensitively.

    The pattern is anchored with \\b on both sides so we don't match
    "pre-training" inside "warm-pre-training-warm" or similar. The match
    captures the matched text so we can preserve case in the replacement.
    """
    # Sort by length descending so longer matches win (pre-training before pre-train).
    sorted_terms = sorted(_TERMS, key=lambda t: -len(t[0]))
    alt = "|".join(re.escape(h) for h, _ in sorted_terms)
    return re.compile(rf"\b({alt})\b", re.IGNORECASE)


_TERMS_RE = _build_regex()
_LOWER_TO_CLOSED = {h.lower(): c for h, c in _TERMS}

# Bold span detector: any **...** pair on a line. Captures the inner text.
# Used to skip matches inside bold first-definition spans, which by §10.3
# convention contain proper-noun expansions like
# **CLIP (Contrastive Language-Image Pre-training)**.
_BOLD_SPAN_RE = re.compile(r"\*\*[^*\n]+?\*\*")


def _bold_spans(line: str) -> list[tuple[int, int]]:
    """Return (start, end) ranges of all **...** spans on a line."""
    return [(m.start(), m.end()) for m in _BOLD_SPAN_RE.finditer(line)]


def _is_proper_noun_continuation(line: str, end: int) -> bool:
    """True if the matched form is followed by a capitalized word (proper noun).

    Pattern: `Pre-trained Transformer`, `Pre-training Corpus`, etc.
    The hyphenated form is part of a multi-word proper-name phrase.

    Heuristic: if the next non-whitespace, non-punctuation character is an
    uppercase letter, treat the match as part of a proper-noun chain and
    keep the hyphen.

    Edge cases handled:
    - End of line: returns False (no continuation)
    - Followed by ')' then capitalized: still True (e.g. "Pre-training)" inside
      a parenthesized expansion is ambiguous; we conservatively treat it as
      proper-noun if the bold-span check above didn't catch it)
    - Followed by lowercase: returns False (regular prose)
    """
    rest = line[end:]
    # Skip whitespace and a single punctuation char like ',' or ':'
    stripped = rest.lstrip(" \t)")
    if not stripped:
        return False
    return stripped[0].isupper()


def _preserve_case(matched: str, closed: str) -> str:
    """Apply the case style of `matched` to `closed`.

    Examples:
        ("pre-training", "pretraining") -> "pretraining"
        ("Pre-training", "pretraining") -> "Pretraining"
        ("Pre-Training", "pretraining") -> "Pretraining"
        ("PRE-TRAINING", "pretraining") -> "PRETRAINING"
    """
    if matched.isupper():
        return closed.upper()
    if matched[0].isupper():
        return closed[0].upper() + closed[1:]
    return closed


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for the 6 pre-/non- compound prefixes that should be closed up.

    Returns (issues, next_counter).
    """
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue
        # Skip table rows — bold table headers and pipe content are handled
        # separately and the close-up rule doesn't apply there.
        if is_table_row(line):
            continue
        if "-" not in line:
            continue

        spans = inline_protected_spans(line)
        bolds = _bold_spans(line)
        edits: list[tuple[int, int, str, str]] = []  # (start, end, before, after)

        for m in _TERMS_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            # Skip matches inside **bold** first-definition spans —
            # those are proper-noun expansions per §10.3 convention
            # (e.g. **CLIP (Contrastive Language-Image Pre-training)**).
            if position_in_spans(m.start(), bolds):
                continue
            # Skip matches that are part of a proper-noun phrase, where
            # the next word is capitalized (e.g. "Pre-trained Transformer",
            # "Pre-training Corpus", "Pre-training)" inside an expansion).
            if _is_proper_noun_continuation(line, m.end()):
                continue
            matched = m.group(1)
            closed = _LOWER_TO_CLOSED[matched.lower()]
            replacement = _preserve_case(matched, closed)
            edits.append((m.start(), m.end(), matched, replacement))

        if not edits:
            continue

        # Apply edits in reverse to keep indices stable
        new_line = line
        for start, end, _before, after in reversed(edits):
            new_line = new_line[:start] + after + new_line[end:]

        # Emit one Issue per match (not one per line) so the ledger is
        # at the granularity of edits, matching the script-lane fixer's
        # expectation.
        for start, end, before, after in edits:
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
                    reason=f"{before} -> {after}",
                )
            )
            counter += 1

    return issues, counter
