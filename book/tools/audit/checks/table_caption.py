"""Check: every markdown pipe table has a caption + {#tbl-...} ID.

Rule: book-prose-merged.md section 6 (Visuals & Assets) — Table Formatting

    Tables that carry quantitative content (numbers, multipliers, comparisons)
    MUST have a caption + `@tbl-` ID, regardless of whether they sit in body
    text or inside a callout / `tbl-colwidths` div. The convention:

        | **Col A** | **Col B** |
        |:----------|:----------|
        | row 1     | row 2     |

        : **Bold Title**: Explanation. {#tbl-id}

    The caption line is `: <bold> : <explanation> {#tbl-<id>}` placed
    IMMEDIATELY after the table's last row (no blank line required).

Auto-fixable: NO. Generating a sensible caption needs human judgment.
We emit `needs_subagent=False, auto_fixable=False` so a human (or editor
agent) must add the caption.

Protected contexts:
  - Tables inside ```` ``` ```` code fences (rendered as code, not tables)
  - Tables inside `{.tikz}` blocks (LaTeX `tabular`, not markdown pipes)
  - Tables inside YAML front matter (extremely rare; out of scope)

How false-positive-resistant this is:
  - Only flags blocks that have BOTH a pipe row AND a separator row (the
    Quarto/CommonMark definition of a pipe table).
  - Allows the caption to appear on the line immediately after the last
    row, OR after one blank line, OR after a closing div fence (`:::`).
  - Will NOT flag ASCII-art tables, decorative dividers, or anything that
    lacks a `|----|` separator row.

Drift this catches:
  - The energy-movement table in vol1/data_engineering (caught manually
    this release pass — would have been auto-caught by this rule).
  - Any future table authored inside a callout without a caption.
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id

CATEGORY = "table-without-caption"
RULE = "book-prose-merged.md section 6 (Visuals & Assets)"
RULE_TEXT = (
    "Markdown pipe tables must have a caption line `: **Title**: …` "
    "followed by `{#tbl-…}` ID, even inside callouts and tbl-colwidths divs."
)

# A pipe-table content row: at least one `|` separator, non-empty cells.
# Tolerates trailing `\index{...}` tags appended to the last cell (a common
# pattern when a table-summary index entry is anchored to the last row).
_PIPE_ROW_RE = re.compile(r"^\s*\|.+\|(\s*\\index\{[^}]*\})*\s*$")
# A pipe-table separator row: only |, :, -, spaces between the pipes.
_PIPE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
# A caption line: starts with `: ` and contains `{#tbl-...}` or
# `{#tbl-id tbl-colwidths="..."}` style Quarto attribute lists.
_CAPTION_RE = re.compile(r"^\s*:\s+.*\{#tbl-[A-Za-z0-9_-]+[\s}]")
# Fence delimiters we treat as code blocks.
_CODE_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
# Quarto raw tikz/latex fences.
_RAW_FENCE_RE = re.compile(r"^\s*```\s*\{\.?tikz|^\s*```\s*\{=latex\}|^\s*\{=tex\}")


def _is_caption_candidate(line: str) -> bool:
    """Caption lines may carry the `{#tbl-...}` on the same line or wrap."""
    return _CAPTION_RE.match(line) is not None


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for markdown pipe tables that lack a caption line."""
    issues: list[Issue] = []
    counter = start_counter

    lines = text.split("\n")
    n = len(lines)

    in_code_fence = False  # generic ``` / ~~~ blocks
    in_raw_block = False   # `{.tikz}` and `{=latex}` blocks
    i = 0

    while i < n:
        line = lines[i]

        # Toggle code-fence state.
        if _CODE_FENCE_RE.match(line):
            # Distinguish raw-LaTeX / TikZ fences (still code, but specifically not markdown).
            if _RAW_FENCE_RE.match(line):
                in_raw_block = not in_raw_block
            else:
                in_code_fence = not in_code_fence
            i += 1
            continue

        if in_code_fence or in_raw_block:
            i += 1
            continue

        # Look for a markdown pipe table start: a pipe row followed by a separator row.
        if _PIPE_ROW_RE.match(line) and i + 1 < n and _PIPE_SEP_RE.match(lines[i + 1]):
            table_start = i + 1  # 1-indexed line number of the header row
            # Walk forward until we leave the table.
            j = i + 2  # skip header + separator
            while j < n and _PIPE_ROW_RE.match(lines[j]):
                j += 1
            # `j` now points at the first line AFTER the table body.
            # Caption may be on this line, on the next line (after one blank),
            # or after a closing `:::` of an enclosing colwidths div.
            caption_found = False
            lookahead_limit = min(j + 4, n)  # ~3 lines of slack
            for k in range(j, lookahead_limit):
                stripped = lines[k].strip()
                if _is_caption_candidate(lines[k]):
                    caption_found = True
                    break
                # An intervening closing `:::` (end of `{tbl-colwidths=...}`
                # div) is fine — caption may still come after it.
                if stripped == "":
                    continue
                if stripped.startswith(":::"):
                    continue
                # `\index{...}` tags often trail the last data row or sit
                # between the table and its caption; skip them, keep scanning.
                if stripped.startswith("\\index{"):
                    continue
                # Otherwise we've hit non-blank, non-caption content —
                # the caption opportunity is gone.
                break

            if not caption_found:
                # Build a short preview of the header row for the issue
                # message so the operator can grep to find this table fast.
                header_preview = lines[table_start - 1].strip()
                if len(header_preview) > 100:
                    header_preview = header_preview[:97] + "..."
                issues.append(
                    Issue(
                        id=make_issue_id(scope, CATEGORY, counter),
                        category=CATEGORY,
                        rule=RULE,
                        rule_text=RULE_TEXT,
                        file=str(file_path),
                        line=table_start,
                        col=0,
                        before=header_preview,
                        suggested_after="",  # No automatic fix.
                        auto_fixable=False,
                        needs_subagent=False,
                        reason=(
                            "Pipe table at line "
                            f"{table_start} has no `: **Title**: … {{#tbl-…}}` "
                            "caption within four lines of its last row."
                        ),
                    )
                )
                counter += 1

            # Resume scanning after the table body.
            i = j
            continue

        i += 1

    return issues, counter
