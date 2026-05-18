r"""Check: math-notation rendering issues that break Pandoc/Quarto output.

Five known failure modes from prior copyedit findings (commits bcc4527f4a
"Normalize math rendering across QMD files" and 42209c08b0 "wave 4 — audit
backlog cleanup"):

1. **Adjacent math spans** (`$\sim$$15%$`): two abutting inline-math spans
   render with a visible seam at print scale and the closing-`$` plus the
   opening-`$` collapse into a literal `$$` in some renderers. Should be a
   single span: `$\sim 15\%$`.
   Confidence: high.

2. **`100s W` literal**: "100s W" or "10s W" in body prose reads as a
   plural-s genitive ("hundreds of watts" → "100s W") instead of the
   intended approximation ("~100 watts"). The print convention is
   `$\sim 100$~W` or `\mathord{\sim}100`~W.
   Confidence: medium — could legitimately mean "hundreds" in some
   colloquial passages, so this needs operator review.

3. **Missing `\mu`**: literal `u` in `uJ`/`us`/`uW`/`uA`/`uF`/`uV`/`um` unit
   contexts in body prose (these should be `\mu`J etc.). Skip code blocks
   and Python cells (where `u`-prefixed names are computed values).
   Confidence: medium — `us` can be the English word in unrelated
   passages; the regex anchors on digit-adjacency to reduce false hits.

4. **Spacing around `\times`**: `5 \times 5` is fine, `5\times 5` is wrong;
   also `\times5` (no space after) is wrong. Per book-prose-merged.md §2
   the canonical dimension form is `$N{\times}M$` (with braces) and the
   canonical multiplier form is `N$\times$` (number outside math). Both
   bare `\d\\times` and `\\times\d` patterns escape those forms.
   Confidence: high.

5. **Bare `%` inside math** (`$15%$`): a literal `%` inside `$...$` should
   be escaped as `\%`. Pandoc renders `%` as LaTeX comment start, so
   everything to end-of-line silently disappears in PDF.
   Confidence: high.

Auto-fixable: NO for all five. Each pattern has a context-dependent fix
(merge spans, choose `\sim` form, decide on `\mu` vs literal `u`, decide
on dimension vs multiplier form, escape `%`). Operator review required.

Protected contexts: code fences (```), inline-Python (`{python}`)
chunks, TikZ blocks (```{.tikz}), `.callout-code` divs, YAML, display
math, HTML comments, inline code, `\index{}`, protected attributes.
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

CATEGORY = "math-notation-render"
RULE = "book-prose-merged.md section 2 (Quantitative Rigor & Math)"
RULE_TEXT = (
    "Math notation must render correctly: single math span per expression, "
    "use \\sim/\\mu for ~/u, space and brace \\times per house style, "
    "escape % as \\% inside math."
)


# ── Pattern regexes ─────────────────────────────────────────────────────────

# Pattern 1: Adjacent inline math spans. The literal collision `$A$$B$`
# is parsed as `$A$` then `$B$` but renders as `∼$15%$` in some Pandoc
# paths (and breaks the display-math heuristic in others). We look for
# `$...$` immediately followed (no whitespace) by `$...$` on the same
# line. Both spans must be non-empty and not contain `$` (so we don't
# match display math `$$...$$`).
_ADJACENT_MATH_RE = re.compile(
    r"(\$(?!\$)[^$\n]+?\$)(\$(?!\$)[^$\n]+?\$)"
)

# Pattern 2: `100s W` / `10s W` / `1000s W` literal. A digit run followed
# by lowercase `s`, whitespace, then `W` (uppercase, often the watts
# unit). Anchored to word boundaries so we don't catch e.g. "items W..."
# (no leading digit) or "1sW" (no space).
_LITERAL_S_W_RE = re.compile(r"\b(\d+s)\s+W\b")

# Pattern 3: Literal `u` in place of `\mu` for unit prefixes. We match a
# digit run (optional decimal), optional whitespace or tilde, then `u`
# immediately followed by one of the SI-prefixable units: J, s, W, A, F,
# V, C, m. Word boundary after to avoid `100us`+`er` etc. The literal
# `us` is the most ambiguous (could be English "us"); we require the
# digit-adjacency, which eliminates most false hits. Negative lookbehind
# on `\\` skips `\u...` LaTeX commands (none in our corpus but safe).
_MISSING_MU_RE = re.compile(
    r"(?<![\\a-zA-Z])(\d+(?:\.\d+)?)\s*~?\s*(u)(J|s|W|A|F|V|C|m)\b"
)

# Pattern 4: `\times` without space on either side. Two sub-cases:
#   4a. `\d\times`  — digit immediately before `\times` (no space)
#   4b. `\times\d`  — digit immediately after `\times` (no space)
# Per book-prose-merged.md §2 (`Multiplication ($\times$) — Dimension vs
# Multiplier`), the canonical forms are:
#   - dimension: `$N{\times}M$` (braces around \times, no spaces)
#   - multiplier: `N$\times$` (digit outside math, no spaces around $)
#   - math-anchored multiplier (exception): `$10^N\times$ more energy`
#     — \times sits at the END of a math span, no digit follows
# A bare `\d\\times\d` or `\d\\times M` (where M is a token mid-math) is
# non-canonical. The braced form `{\times}` doesn't match because the
# digit is separated from `\times` by `{`.
#
# We exclude the math-anchored-multiplier exception via a lookahead:
# if `\times` is immediately followed by `$` (end of math span), the
# usage is canonical and we don't flag.
_TIMES_NO_SPACE_RE = re.compile(
    r"(\d\\times(?!\$)|\\times\d)"
)

# Pattern 5: Bare `%` inside inline math. We match `$...%...$` where the
# `%` is NOT escaped as `\%`. Non-greedy span, must not contain another
# `$` (so display math `$$...$$` is handled separately by the walker's
# display-math state). Negative lookbehind on `\\` ensures we skip the
# already-correct `\%` escape.
_BARE_PERCENT_IN_MATH_RE = re.compile(
    r"\$(?!\$)([^$\n]*?(?<!\\)%[^$\n]*?)\$(?!\$)"
)


# ── Additional protected-context detection ──────────────────────────────────


def _in_callout_code_div(state) -> bool:
    """True if we're inside a `.callout-code` div.

    The LineWalker tracks tip/checkpoint/definition callouts by name but
    not `.callout-code`. We approximate by checking callout_depth > 0
    AND none of the known protection flags being set — i.e. we're inside
    some Quarto div that the walker is counting but not specifically
    flagging. This is a loose check; the canonical signal is the code
    fence inside the div, which the walker's `in_code_fence` already
    catches. Kept here for documentation.
    """
    return False  # rely on in_code_fence; .callout-code wrapping isn't
    # itself protective — its contents are protected via the code fence.


def _is_python_chunk_line(line: str) -> bool:
    """True if this line is a Quarto inline-Python expression line.

    Matches lines that contain `` `{python} ... ` `` — these are
    protected because the expression body is dynamic content (computed
    values), and patterns like `1uJ` in a Python f-string variable name
    are legitimate code, not prose typos.

    The inline-code span detector in `inline_protected_spans` already
    covers per-character protection within the line; this is a coarser
    line-level skip for lines that are predominantly inline-Python.
    Currently unused — we rely on per-span protection — but documented
    for future tuning.
    """
    return "`{python}" in line


# ── Main check ──────────────────────────────────────────────────────────────


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for math-notation rendering issues in body prose.

    Returns (issues, next_counter).
    """
    issues: list[Issue] = []
    counter = start_counter

    walker = LineWalker(text)
    for line, state, line_num in walker:
        if default_line_skip(line, state):
            continue

        # Compute protected spans once per line — used by all five
        # patterns to filter out matches inside inline code, inline math
        # (for non-math patterns), index entries, protected attributes,
        # and citations.
        spans = inline_protected_spans(line)

        # ── Pattern 1: Adjacent inline math spans ──
        # `$A$$B$` — should be one span. High confidence; mechanical.
        # We allow the match position to be inside a protected attr or
        # inline code, in which case we skip — those are body-prose-
        # adjacent contexts (fig-cap with `$A$$B$` is also a bug, but
        # the prose pattern catches the same source line if it's
        # rendered text).
        for m in _ADJACENT_MATH_RE.finditer(line):
            # Skip if the match starts inside inline code or an index
            # entry; protected attributes are still in-scope (a caption
            # with `$A$$B$` renders just as poorly).
            if position_in_spans(m.start(), [
                s for s in spans if line[s[0]:s[1]].startswith("`")
                or line[s[0]:s[1]].startswith("\\index")
            ]):
                continue
            before_text = m.group(0)
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
                    suggested_after="",
                    auto_fixable=False,
                    needs_subagent=False,
                    confidence="high",
                    reason=(
                        f"adjacent math spans {before_text!r} should be a "
                        "single span (e.g. `$\\sim 15\\%$`)"
                    ),
                )
            )
            counter += 1

        # ── Pattern 2: `100s W` / `10s W` literal ──
        # Medium confidence: could be intentional in colloquial passages.
        for m in _LITERAL_S_W_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            before_text = m.group(0)
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
                    suggested_after="",
                    auto_fixable=False,
                    needs_subagent=False,
                    confidence="medium",
                    reason=(
                        f"{before_text!r} reads as plural-s, not "
                        "approximation; use `$\\sim N$~W` or "
                        "`\\mathord{\\sim}N`~W"
                    ),
                )
            )
            counter += 1

        # ── Pattern 3: Missing `\mu` (literal `u` for micro-) ──
        # Medium confidence: skip code/Python lines (per default_line_skip
        # which already filters code fences and `#|` chunk options).
        # Inline code spans are also skipped via `spans`.
        for m in _MISSING_MU_RE.finditer(line):
            if position_in_spans(m.start(), spans):
                continue
            num, u, unit = m.group(1), m.group(2), m.group(3)
            before_text = m.group(0)
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
                    suggested_after="",
                    auto_fixable=False,
                    needs_subagent=False,
                    confidence="medium",
                    reason=(
                        f"{before_text!r} likely missing \\mu; should be "
                        f"`{num} $\\mu${unit}` or `${num}\\,\\mu\\text{{{unit}}}$`"
                    ),
                )
            )
            counter += 1

        # ── Pattern 4: `\times` without space ──
        # High confidence: `\d\times` and `\times\d` both escape the
        # canonical `$N{\times}M$` (braced dimension) and `N$\times$`
        # (multiplier-outside-math) forms.
        for m in _TIMES_NO_SPACE_RE.finditer(line):
            # Skip if the match falls inside inline code (e.g. a `4x` in
            # a model-option label that got typed as `4\times` for
            # rendering reasons).
            if position_in_spans(m.start(), [
                s for s in spans if line[s[0]:s[1]].startswith("`")
                or line[s[0]:s[1]].startswith("\\index")
            ]):
                continue
            # Skip the canonical braced form `{\times}` — the byte
            # before `\times` is `{`, not a digit. Our regex already
            # requires `\d\\times` so the brace form doesn't match;
            # this is documentation.
            before_text = m.group(0)
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
                    suggested_after="",
                    auto_fixable=False,
                    needs_subagent=False,
                    confidence="high",
                    reason=(
                        f"{before_text!r} lacks space around \\times; "
                        "use `$N{\\times}M$` for dimensions or "
                        "`N$\\times$` for multipliers"
                    ),
                )
            )
            counter += 1

        # ── Pattern 5: Bare `%` inside math ──
        # High confidence: `%` inside `$...$` starts a LaTeX comment.
        for m in _BARE_PERCENT_IN_MATH_RE.finditer(line):
            # Skip if the match starts inside inline code or an index
            # entry; otherwise we want to flag (including matches inside
            # protected attributes like fig-cap, where math still renders).
            if position_in_spans(m.start(), [
                s for s in spans if line[s[0]:s[1]].startswith("`")
                or line[s[0]:s[1]].startswith("\\index")
            ]):
                continue
            body = m.group(1)
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
                    suggested_after="",
                    auto_fixable=False,
                    needs_subagent=False,
                    confidence="high",
                    reason=(
                        f"bare `%` inside math span `${body}$` starts a "
                        "LaTeX comment; escape as `\\%`"
                    ),
                )
            )
            counter += 1

    return issues, counter
