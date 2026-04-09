"""Check: Latin abbreviations in running text (§10.6).

Rule: book-prose-merged.md section 10.6

    "In running text, prefer English to Latin abbreviations":

      - "for example"  not e.g.
      - "that is"      not i.e.
      - "and so on"    not etc.

    The Latin abbreviations are acceptable inside parentheses,
    footnotes, and notes where space is constrained.

This check flags `e.g.`, `i.e.`, and `etc.` when they appear in
body prose OUTSIDE a parenthesized clause. Parenthesized uses are
fine because the author has already committed to a parenthetical
aside where the Latin form is conventional and terse.

Not flagged:
  - Inside `(...)`                — explicitly permitted
  - Inside inline code `...`      — code or citation key
  - Inside inline math $...$
  - Inside `\\index{...}`         — index entry
  - Inside `title=`/`fig-cap=`/`fig-alt=`/`lst-cap=`/`tbl-cap=`
  - Inside citations `[@key]`, footnote refs `[^fn-...]`
  - Inside section headings (headings are terse, Latin OK)
  - Inside table rows (space-constrained)
  - In YAML, code fences, display math, HTML blocks

Auto-fixable? Not deterministically. `e.g.` → "for example" is a
mechanical substitution, BUT the result may need case adjustment
at sentence start, comma handling, and surrounding rewording.
Mark `auto_fixable=False`; `needs_subagent=True`.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import (
    LineWalker,
    heading_level,
    inline_protected_spans,
    is_div_attribute_line,
    is_inside_index_entry,
    is_inside_protected_attribute,
    is_python_chunk_option,
    is_table_row,
    position_in_spans,
)

CATEGORY = "latin-running-text"
RULE = "book-prose-merged.md section 10.6"
RULE_TEXT = "Latin abbreviations (e.g., i.e., etc.) belong in parentheses; prefer English in running text"


# ── Latin abbreviation patterns ────────────────────────────────────────────
#
# Each pattern matches the abbreviation with a trailing period and bounded
# by non-word characters on both sides. We deliberately include the comma
# that usually follows these abbreviations as part of the "classic running-
# text use" — but the regex match itself is just the abbreviation.
#
# Case-sensitive: "e.g." and "i.e." are always lowercase by convention,
# and matching "E.g." at sentence start is handled separately (we do NOT
# flag at sentence start because the English substitute would then have
# to be capitalized, which the fix doesn't know how to do automatically).

_LATIN_ABBREVS = [
    ("e.g.", "for example"),
    ("i.e.", "that is"),
    ("etc.", "and so on"),
]

_PATTERN = re.compile(
    r"(?<![\w.])(e\.g\.|i\.e\.|etc\.)(?![\w])"
)


# ── Parenthetical span detection ───────────────────────────────────────────
#
# The only permitted use of Latin abbreviations is inside a parenthesized
# clause. We compute parenthesis spans per line (matching open `(` to the
# next close `)`) and skip matches inside them.
#
# The implementation is intentionally a simple stack-based scan, not a
# regex. Nested parens are handled correctly (a match is inside *any*
# enclosing pair). We only look at parens on the same line — multi-line
# parenthetical asides are rare and produce at most one false positive.

def _paren_spans(
    line: str, protected: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return a list of (open_pos, close_pos+1) spans for parenthetical
    ranges on this line. Unclosed `(` is ignored.

    `protected` is the output of `inline_protected_spans(line)`. Paren
    characters inside protected spans (inline math, inline code, index
    entries, citations) are NOT counted because they often contain
    unbalanced parens — e.g. `$[0, 0.5) \\to 0$` has a `)` with no
    matching `(` inside the math span, which would otherwise pop an
    outer parenthetical and corrupt the stack.
    """
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for i, ch in enumerate(line):
        if ch == "(" or ch == ")":
            if position_in_spans(i, protected):
                continue
            if ch == "(":
                stack.append(i)
            else:  # ch == ")"
                if stack:
                    start = stack.pop()
                    spans.append((start, i + 1))
    return spans


def _inside_parens(pos: int, spans: list[tuple[int, int]]) -> bool:
    """True if position `pos` is inside any parenthetical span."""
    for s, e in spans:
        if s <= pos < e:
            return True
    return False


# ── Line-level filter ──────────────────────────────────────────────────────

def _skip_line(line: str, state) -> bool:
    """Return True for lines where Latin abbreviations are always OK."""
    if state.in_yaml or state.in_code_fence or state.in_display_math:
        return True
    if state.in_html_style_block or state.in_html_comment:
        return True
    if is_python_chunk_option(line):
        return True
    if is_div_attribute_line(line):
        return True
    # Section headings are terse; Latin abbreviations are acceptable there.
    if heading_level(line) is not None:
        return True
    # Table rows are space-constrained; Latin abbreviations are acceptable.
    if is_table_row(line):
        return True
    return False


# ── Main check entry point ─────────────────────────────────────────────────

def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for Latin abbreviations in running text outside parentheses."""
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if _skip_line(line, state):
            continue

        spans = inline_protected_spans(line)
        parens = _paren_spans(line, spans)

        for m in _PATTERN.finditer(line):
            start, end = m.start(), m.end()
            # Inside an inline protected span? (code, math, index,
            # attribute, citation, footnote ref, etc.)
            if position_in_spans(start, spans):
                continue
            if is_inside_index_entry(line, start):
                continue
            if is_inside_protected_attribute(line, start):
                continue
            # Inside a parenthetical clause? (permitted by §10.6)
            if _inside_parens(start, parens):
                continue
            latin = m.group(1)
            english = next(
                eng for lat, eng in _LATIN_ABBREVS if lat == latin
            )
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
                    suggested_after="",
                    auto_fixable=False,
                    needs_subagent=True,
                    reason=f"{latin!r} in running text — prefer {english!r}",
                )
            )
            counter += 1

    return issues, counter


# ── Adversarial self-test ──────────────────────────────────────────────────

_POSITIVE_LINES = [
    # Latin abbreviation in body prose, outside parens
    "Models are trained on large datasets, e.g., ImageNet for vision tasks.",
    "The optimizer has several variants, i.e., Adam, SGD, and RMSProp.",
    "Common operations include matrix multiply, etc., across GPUs.",
    "Modern systems, e.g., H100 accelerators, run at peak efficiency.",
    "The two approaches differ, i.e., one is asynchronous and the other is synchronous.",
]

_NEGATIVE_LINES = [
    # Inside parentheses — permitted
    "Models are trained on large datasets (e.g., ImageNet).",
    "The optimizer has several variants (i.e., Adam and SGD).",
    "Common operations (matrix multiply, convolution, etc.) run on GPUs.",
    # Nested: parenthetical with Latin abbreviation nested inside
    "A family of optimizers (including Adam, SGD, etc.) is available.",
    # Inside inline code
    "Set `flags.etc.` in the YAML config file.",
    # Inside citation — footnote/bracketed citation is protected
    "The dataset is well known [@deng2009imagenet].",
    # Inside section header
    "### Common optimizers, e.g., Adam and SGD",
    # Inside \index{}
    "Later discussion \\index{example, e.g.} expands on this.",
    # Inside a table row
    "| **Optimizer** | **Use Case** | **Notes (e.g., memory)** |",
]


def _self_test() -> int:
    from audit.protected_contexts import LineState
    state = LineState()

    pos_fail: list[str] = []
    neg_fail: list[str] = []

    def has_hit(line: str) -> bool:
        if _skip_line(line, state):
            return False
        spans = inline_protected_spans(line)
        parens = _paren_spans(line, spans)
        for m in _PATTERN.finditer(line):
            s = m.start()
            if position_in_spans(s, spans):
                continue
            if is_inside_index_entry(line, s):
                continue
            if is_inside_protected_attribute(line, s):
                continue
            if _inside_parens(s, parens):
                continue
            return True
        return False

    for line in _POSITIVE_LINES:
        if not has_hit(line):
            pos_fail.append(line)
    for line in _NEGATIVE_LINES:
        if has_hit(line):
            neg_fail.append(line)

    total = len(_POSITIVE_LINES) + len(_NEGATIVE_LINES)
    failures = len(pos_fail) + len(neg_fail)
    print(f"latin_running_text self-test: {total - failures}/{total} passed")
    if pos_fail:
        print(f"\n{len(pos_fail)} POSITIVE NOT flagged:")
        for line in pos_fail:
            print(f"  - {line}")
    if neg_fail:
        print(f"\n{len(neg_fail)} NEGATIVE flagged (FP):")
        for line in neg_fail:
            print(f"  - {line}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_self_test())
