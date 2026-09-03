"""Check: a `: caption {#tbl-...}` line sandwiched between two `:::` close fences.

Rule: EPUB build integrity — Pandoc's EPUB writer treats a caption as orphan
when it is separated from its table by a closing div fence and then followed
by another closing div fence. The label never registers, `@tbl-` xrefs go
unresolved, and the EPUB postprocess silently rewrites them to dead HTML
URLs (e.g. `nn_computation.html#tbl-…` that doesn't exist inside the EPUB).
This produces epubcheck ERRORs that block the build.

The pattern that broke vol1 EPUB on 2026-05-18 (commit 54ca22d7d7):

    ::: {.content-visible when-format="html"}
    ::: {tbl-colwidths="[25,25,25,25]"}

    | **Header** | … |
    | row | … |

    :::                                   ← inner close

    : **Caption**: …. {#tbl-foo}          ← orphaned caption

    :::                                   ← outer close

The PDF block (with its own raw-LaTeX `tabular`) is unaffected because LaTeX
does not bind captions through Pandoc; only HTML/EPUB Markdown rendering
breaks.

The safe form moves `tbl-colwidths` onto the caption attribute and drops the
inner nested div, keeping caption and table inside the same flow:

    ::: {.content-visible unless-format="pdf"}

    | **Header** | … |
    | row | … |

    : **Caption**: …. {#tbl-foo tbl-colwidths="[25,25,25,25]"}

    :::

Auto-fixable: NO. The reshape requires moving the `tbl-colwidths` attribute
onto the caption line and widening `when-format="html"` to
`unless-format="pdf"` if the author intends EPUB to render the markdown
table. A human (or editor agent) must do that.

How false-positive-resistant this is:
  - Only flags caption lines that are BOTH preceded AND followed by a
    closing `:::` fence (skipping blanks). Captions inside a single div
    flow (one close after, no close before) do not match.
  - Skips code fences and raw-LaTeX/TikZ blocks.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id

CATEGORY = "caption-orphan"
RULE = "EPUB build integrity — caption sandwiched between two `:::` closes"
RULE_TEXT = (
    "A `: …{#tbl-…}` caption line must not sit between two `:::` "
    "close fences (Pandoc EPUB orphans the caption and the @tbl- ref "
    "goes unresolved). Move `tbl-colwidths` onto the caption attribute "
    "and drop the inner nested div."
)

_CAPTION_RE = re.compile(r"^\s*:\s+.*\{#tbl-[A-Za-z0-9_-]+[\s}]")
_CLOSE_FENCE_RE = re.compile(r"^\s*:{3,}\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_RAW_FENCE_RE = re.compile(r"^\s*```\s*\{\.?tikz|^\s*```\s*\{=latex\}|^\s*\{=tex\}")


def _prev_nonblank(lines: list[str], idx: int) -> tuple[int, str] | None:
    """Return (index, line) of the closest non-blank line before idx, or None."""
    j = idx - 1
    while j >= 0:
        if lines[j].strip():
            return j, lines[j]
        j -= 1
    return None


def _next_nonblank(lines: list[str], idx: int) -> tuple[int, str] | None:
    """Return (index, line) of the closest non-blank line after idx, or None."""
    j = idx + 1
    while j < len(lines):
        if lines[j].strip():
            return j, lines[j]
        j += 1
    return None


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Flag caption lines that sit between two `:::` close fences."""
    issues: list[Issue] = []
    counter = start_counter

    lines = text.split("\n")
    n = len(lines)

    in_code_fence = False
    in_raw_block = False

    for i, line in enumerate(lines):
        if _CODE_FENCE_RE.match(line):
            if _RAW_FENCE_RE.match(line):
                in_raw_block = not in_raw_block
            else:
                in_code_fence = not in_code_fence
            continue

        if in_code_fence or in_raw_block:
            continue

        if not _CAPTION_RE.match(line):
            continue

        prev = _prev_nonblank(lines, i)
        nxt = _next_nonblank(lines, i)
        if not prev or not nxt:
            continue

        prev_is_close = _CLOSE_FENCE_RE.match(prev[1]) is not None
        next_is_close = _CLOSE_FENCE_RE.match(nxt[1]) is not None

        if prev_is_close and next_is_close:
            preview = line.strip()
            if len(preview) > 100:
                preview = preview[:97] + "..."
            issues.append(
                Issue(
                    id=make_issue_id(scope, CATEGORY, counter),
                    category=CATEGORY,
                    rule=RULE,
                    rule_text=RULE_TEXT,
                    file=str(file_path),
                    line=i + 1,
                    col=0,
                    before=preview,
                    suggested_after="",
                    auto_fixable=False,
                    needs_subagent=False,
                    reason=(
                        f"Caption at line {i + 1} is sandwiched between two "
                        f"closing `:::` fences (prev close at line {prev[0] + 1}, "
                        f"next close at line {nxt[0] + 1}). Pandoc's EPUB writer "
                        "will orphan the caption and `@tbl-` refs to this label "
                        "will go unresolved. Move `tbl-colwidths` onto the "
                        "caption attribute and drop the inner nested div."
                    ),
                )
            )
            counter += 1

    return issues, counter
