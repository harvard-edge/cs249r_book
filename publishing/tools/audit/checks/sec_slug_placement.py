"""Check: Quarto crossref ID ({#sec-...}, {#fig-...}, etc.) at end of body
prose, where it does not attach to anything.

Background. In Quarto/Pandoc, the `{#sec-foo}` syntax is a header attribute.
It is recognized when it appears:

    - immediately after an ATX heading            `# Title {#sec-foo}`
    - on a fenced div                              `::: {#sec-foo}`
    - on a single-line display-math block          `$$ ... $$ {#eq-foo}`
    - on a single-line figure                      `![alt](img.png){#fig-foo}`
    - on a single-line table caption (Quarto)      `: **Caption**: ... {#tbl-foo}`

If the same syntax sits at the end of a body-prose paragraph (after sentence-
final punctuation, on a non-heading line), Pandoc treats it as a span/bracketed-
text attribute that has nothing to bind to. In practice it either silently
disappears from the rendered output or — worse — leaves a literal `{#sec-foo}`
string visible in the rendered chapter. Cross-references (`@sec-foo`) that
point at this slug then resolve to "??".

This check detects the broken placement and suggests two safe fixes:

    1. If the previous non-blank line is an ATX heading without an ID,
       move the `{#sec-foo}` to the end of that heading.
    2. Otherwise, drop the broken trailing slug and emit a `reason` that
       tells the operator to relocate it.

Auto-fixable: NO. The fix involves either relocating the slug to a heading
that may or may not exist or deciding whether the cross-reference itself
should be removed — both require human judgment. Every issue is emitted as
needs_subagent=False (manual review) with a suggested_after that the
operator can copy verbatim if (1) applies.

Protected contexts this check honors:
- YAML frontmatter, code fences, display math (LineWalker via default_line_skip)
- HTML comments and style/script blocks
- TikZ blocks (which are code fences with ```{.tikz})
- Quarto div fences (`::: {#sec-foo}` is valid)
- ATX heading lines (`#...` is valid)
- Quarto table caption lines (`: **Caption**: ...` is valid)
- Image/figure lines (`![alt](path){#fig-foo}` is valid)
- Single-line display math closers (`$$ ... $$ {#eq-foo}`)
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id
from audit.protected_contexts import LineWalker, default_line_skip


CATEGORY = "sec-slug-end-of-paragraph"
RULE = "Quarto crossref ID placement (Pandoc header-attribute syntax)"
RULE_TEXT = (
    "Crossref IDs ({#sec-...}, {#fig-...}, {#tbl-...}, {#eq-...}, {#lst-...}) "
    "must attach to a heading, fenced div, single-line math/figure/caption — "
    "not float at the end of a body-prose paragraph"
)

# Recognized crossref ID prefixes per Quarto.
_PREFIXES = ("sec", "fig", "tbl", "eq", "lst")
_PREFIXES_ALT = "|".join(_PREFIXES)

# Body-prose case A: `... period {#sec-foo}` at end of line.
# We accept any sentence-final punctuation (`.`, `!`, `?`) and also a
# closing bracket/paren after punctuation (e.g. `).` or `].`). The leading
# anchor `^(?!#)` rejects ATX headings; we also reject the table-caption
# colon, the div fence, the image bang, and the display-math `$$`.
_TRAILING_ID_PUNCT_RE = re.compile(
    r"^(?![\s]*[#:>])"               # not heading, table caption, blockquote
    r".+?"                            # any preceding content (non-greedy)
    r"[.!?][)\]\"']?"                # sentence-final punctuation (opt close brkt)
    r"\s*"                            # optional whitespace
    r"(\{#(?:" + _PREFIXES_ALT + r")-[^}\n]+\})"
    r"\s*$"
)

# Body-prose case B: `... non-period content {#sec-foo}` at end of line, when
# the line is clearly body prose (no leading `#`, no `:::`, no `![`, no
# leading `$$`, no leading `:` table caption). This catches the case where
# someone tacked the ID on after a clause that doesn't end in a period — also
# wrong. We keep the pattern conservative: the line must contain at least one
# word character before the slug, and must NOT be a recognized valid host.
_TRAILING_ID_GENERIC_RE = re.compile(
    r"^(?![\s]*[#:>])"
    r"(?!\s*:::)"
    r"(?!\s*!\[)"
    r"(?!\s*\$\$)"
    r"(?!\s*`)"
    r".*?\w.*?"                       # at least one word char before the slug
    r"(\{#(?:" + _PREFIXES_ALT + r")-[^}\n]+\})"
    r"\s*$"
)

# Sanity check: is the line a Quarto table caption? Those start with `:` and
# legitimately end with `{#tbl-foo}`. We treat any line whose first non-blank
# char is `:` (but not `:::` which is a div) as a caption candidate.
_CAPTION_LINE_RE = re.compile(r"^\s*:(?!::)")

# Image/figure host line: `![alt](path){#fig-foo}`.
_IMAGE_LINE_RE = re.compile(r"^\s*!\[")

# Display-math single-line closer: `$$ ... $$ {#eq-foo}`. Two `$$` on one line.
_DISPLAY_MATH_LINE_RE = re.compile(r"^\s*\$\$.*\$\$\s*\{#")

# Quarto div line: `::: {#sec-foo}` or `::: {.callout-tip}` etc.
_DIV_LINE_RE = re.compile(r"^\s*:::")

# ATX heading line.
_HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s")


def _is_valid_host_line(line: str) -> bool:
    """Return True if the line is a recognized valid host for a crossref ID."""
    if _HEADING_LINE_RE.match(line):
        return True
    if _CAPTION_LINE_RE.match(line):
        return True
    if _IMAGE_LINE_RE.match(line):
        return True
    if _DISPLAY_MATH_LINE_RE.match(line):
        return True
    if _DIV_LINE_RE.match(line):
        return True
    return False


def _suggested_after(line: str, slug: str, prev_heading_line: str | None) -> str:
    """Build a suggested fix.

    If the previous non-blank line is an ATX heading that does NOT already
    carry its own {#...} attribute, suggest moving the slug onto that heading.
    Otherwise, suggest dropping the slug and relocating it manually.
    """
    if (
        prev_heading_line is not None
        and _HEADING_LINE_RE.match(prev_heading_line)
        and "{#" not in prev_heading_line
    ):
        # Suggest the heading-attached form. We return only the corrected
        # body-line (with the slug stripped); the operator must also add the
        # slug to the heading. The reason field describes the two-step fix.
        body_fixed = line[: line.rfind(slug)].rstrip()
        return body_fixed
    # No safe heading host — just strip and let the operator decide.
    return line[: line.rfind(slug)].rstrip()


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for crossref IDs misplaced at the end of body-prose paragraphs."""
    issues: list[Issue] = []
    counter = start_counter

    # First pass: remember the most recent non-blank, non-protected line so
    # we can reference it when building suggestions. We iterate via
    # LineWalker so block-level state (code fences, math, YAML, HTML
    # comments, TikZ) is honored automatically.
    prev_nonblank: str | None = None

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            # Inside a protected block — don't touch, don't update prev.
            continue

        # Fast reject: must contain a {#prefix- pattern.
        if "{#" not in line:
            if line.strip():
                prev_nonblank = line
            continue

        # If the line is a recognized valid host, skip it but update prev.
        if _is_valid_host_line(line):
            if line.strip():
                prev_nonblank = line
            continue

        # Try the strict (period-bounded) pattern first; fall back to the
        # generic pattern for non-period clause endings.
        m = _TRAILING_ID_PUNCT_RE.match(line)
        confidence = "high"
        if not m:
            m = _TRAILING_ID_GENERIC_RE.match(line)
            confidence = "medium"
        if not m:
            if line.strip():
                prev_nonblank = line
            continue

        slug = m.group(1)
        prefix = slug[2 : slug.index("-")]  # e.g. "sec", "fig"
        slug_col = line.rfind(slug)

        suggested = _suggested_after(line, slug, prev_nonblank)
        # Build a reason that explains both the diagnosis and the two-step
        # fix when the previous line is a moveable heading.
        if (
            prev_nonblank is not None
            and _HEADING_LINE_RE.match(prev_nonblank)
            and "{#" not in prev_nonblank
        ):
            reason = (
                f"`{slug}` is misplaced at end of body prose; "
                f"move it onto the preceding heading "
                f"({prev_nonblank.rstrip()!r}) instead"
            )
        else:
            reason = (
                f"`{slug}` is at end of body prose with no valid host "
                f"(heading, div, caption, image, or display math); "
                f"Pandoc will drop it or leave it as literal text"
            )

        issues.append(
            Issue(
                id=make_issue_id(scope, CATEGORY, counter),
                category=CATEGORY,
                rule=RULE,
                rule_text=RULE_TEXT,
                file=str(file_path),
                line=line_num,
                col=slug_col,
                before=line,
                suggested_after=suggested,
                auto_fixable=False,
                needs_subagent=False,
                confidence=confidence,
                reason=reason,
            )
        )
        counter += 1

        if line.strip():
            prev_nonblank = line

    return issues, counter


# ── Adversarial self-test ───────────────────────────────────────────────────
#
# Run with:
#     python3 book/tools/audit/checks/sec_slug_placement.py
#
# Each case is a (corpus, expected_hits) tuple. The detector is exercised
# against a small in-memory document; hits are counted and compared.

_POSITIVE_CASES = [
    # End-of-paragraph period + {#sec-foo}: the canonical failure.
    (
        "Some prose that ends a paragraph here. {#sec-foo}\n\nNext para.\n",
        1,
    ),
    # End-of-paragraph with {#fig-...} — same failure mode, different prefix.
    (
        "Body text closing out the discussion. {#fig-roofline}\n",
        1,
    ),
    # End-of-paragraph with {#tbl-...} on a non-caption non-heading line.
    (
        "More body prose ending in a period. {#tbl-results}\n",
        1,
    ),
    # End-of-paragraph with {#eq-...} on a non-math line.
    (
        "We refer to this equation later in the chapter. {#eq-foo}\n",
        1,
    ),
    # Non-period clause ending — the generic pattern catches it (medium conf).
    (
        "A clause that has no period {#sec-foo}\n",
        1,
    ),
]

_NEGATIVE_CASES = [
    # ATX heading with attached ID — valid.
    ("## Heading text {#sec-foo}\n", 0),
    # Quarto div fence — valid.
    ("::: {#sec-foo}\n\nContent.\n\n:::\n", 0),
    # Table caption — valid (Quarto pattern).
    (": **Table Caption**: explanation here. {#tbl-foo}\n", 0),
    # Single-line display math — valid.
    ("$$ x = y + z $$ {#eq-foo}\n", 0),
    # Image with figure ID — valid.
    ("![Alt text](path/to.png){#fig-bar}\n", 0),
    # Inside code fence — must be skipped.
    (
        "```python\n"
        "# this is just code: not a problem. {#sec-foo}\n"
        "```\n",
        0,
    ),
    # Inside TikZ — must be skipped (TikZ is a code fence).
    (
        "```{.tikz}\n"
        "\\node at (0,0) {label}; % {#fig-foo}\n"
        "```\n",
        0,
    ),
    # Inline cross-reference, not a slug definition.
    ("See @sec-foo for details.\n", 0),
]


def _run_corpus(corpus: str) -> int:
    """Run the detector on `corpus` and return the number of hits."""
    issues, _ = check(Path("<test>"), corpus, "test", 0)
    return len(issues)


def _self_test() -> int:
    failures: list[str] = []

    for i, (corpus, expected) in enumerate(_POSITIVE_CASES):
        got = _run_corpus(corpus)
        if got != expected:
            failures.append(
                f"POSITIVE[{i}] expected {expected}, got {got}: {corpus!r}"
            )

    for i, (corpus, expected) in enumerate(_NEGATIVE_CASES):
        got = _run_corpus(corpus)
        if got != expected:
            failures.append(
                f"NEGATIVE[{i}] expected {expected}, got {got}: {corpus!r}"
            )

    total = len(_POSITIVE_CASES) + len(_NEGATIVE_CASES)
    passed = total - len(failures)
    print(f"sec_slug_placement self-test: {passed}/{total} passed")
    for f in failures:
        print(f"  {f}")
    return 0 if not failures else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_self_test())
