"""Check: long pipe / HTML tables that should declare longtable handling.

Rule: PDF-output integrity — the print build (LuaLaTeX) needs `longtable`
handling on any table that exceeds one printed page. Without it:

  1. The table-caption counter restarts on the continuation page, so the
     same table is numbered twice (e.g., "Table 4.3" and "Table 4.4" both
     for what is conceptually one table).
  2. The header row is missing on subsequent pages, leaving the
     continuation rows context-free.

Quarto exposes this via `{tbl-colwidths="[...]"}` *plus* explicit
`longtable` handling (either a `tbl-colwidths`-bearing div or an inline
LaTeX `\begin{longtable}` block). A table that is "likely too long for
one page" but lacks either is a candidate defect for the release gate.

Heuristic (source-level):
  A pipe table is flagged when ANY of:
    (a) >= 25 data rows (header + separator excluded);
    (b) >= 8 columns;
    (c) has a `{tbl-colwidths=...}` attribute AND row count would exceed
        30 lines of body text (estimated as total wrapped width / column
        width — proxied by row count > 20 with row text > 80 chars
        average).

Additionally:
  Any HTML `<table>` block (typically inside a callout) with >= 25 `<tr>`
  rows is flagged.

A flagged table is treated as a candidate only if it does NOT already
sit inside a div carrying a `{tbl-colwidths=...}` attribute. Quarto's
default behavior is to apply `longtable` to any table inside a
`tbl-colwidths` div, so those are already safe. Tables without that
attribute that exceed the page-length heuristic are the defect class.

Auto-fixable: NO. The fix requires deciding column widths and rewriting
the caption attribute list. We emit `needs_subagent=False,
auto_fixable=False` so a human (or editor agent) makes the call.

Confidence: medium. A 25-row pipe table is *likely* to overflow a
printed page but not guaranteed — depends on row height, column count,
and font size. The operator should sanity-check by rendering the chapter
to PDF.

PDF-render verification:
  `verify_rendered_pdf` is a stub. Detecting a "table counter reset" in
  a rendered PDF would require parsing PDF text streams, mapping each
  "Table X.Y" caption to its source `{#tbl-...}` ID, and checking for
  duplicate IDs that span page boundaries. That is materially more work
  than the source-level heuristic and yields the same defect signal
  (the source-level check catches the cause; the PDF check catches the
  symptom). See the function docstring for the design we'd pursue if
  this were promoted out of "optional".
"""

from __future__ import annotations

import re
from pathlib import Path

from audit.ledger import Issue, make_issue_id


CATEGORY = "longtable-candidate"
RULE = "PDF-output integrity — longtable handling for multi-page tables"
RULE_TEXT = (
    "Tables that exceed one printed page must declare `{tbl-colwidths=...}` "
    "(or use an explicit longtable environment) so the print build repeats "
    "header rows and continues the caption counter."
)

# ── Detection regexes ───────────────────────────────────────────────────────

# A pipe-table content row: at least one `|` separator, non-empty cells.
# Mirrors table_caption.py's pattern so we agree on what "a pipe row" is.
_PIPE_ROW_RE = re.compile(r"^\s*\|.+\|(\s*\\index\{[^}]*\})*\s*$")
# A pipe-table separator row: only |, :, -, spaces between the pipes.
_PIPE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
# Generic code-fence delimiter.
_CODE_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
# Quarto raw tikz/latex fences (still code, but explicitly not markdown).
_RAW_FENCE_RE = re.compile(r"^\s*```\s*\{\.?tikz|^\s*```\s*\{=latex\}|^\s*\{=tex\}")
# tbl-colwidths attribute anywhere on a line (in a caption attr list or div fence).
_TBL_COLWIDTHS_RE = re.compile(r"tbl-colwidths\s*=")
# Opening of an HTML table block.
_HTML_TABLE_OPEN_RE = re.compile(r"<table\b", re.IGNORECASE)
_HTML_TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_HTML_TR_RE = re.compile(r"<tr\b", re.IGNORECASE)

# Thresholds (heuristics — see module docstring).
ROW_THRESHOLD = 25            # >= this many data rows → flag
COL_THRESHOLD = 8             # >= this many columns → flag
WIDE_ROW_THRESHOLD = 20       # paired with colwidths + wide rows → flag
WIDE_AVG_CHARS = 80           # average row width considered "wide"


def _count_columns(header_line: str) -> int:
    """Estimate column count from a pipe-table header row.

    A pipe row `| A | B | C |` has 4 pipes and 3 columns. We strip a
    trailing `\\index{...}` (which is not a column separator) before
    counting.
    """
    stripped = re.sub(r"\\index\{[^}]*\}\s*$", "", header_line.strip())
    # Strip leading/trailing pipes so internal pipes count as separators.
    inner = stripped.strip().strip("|")
    return inner.count("|") + 1


def _nearby_has_colwidths(lines: list[str], table_start: int, table_end: int) -> bool:
    """Check whether this table already carries a tbl-colwidths attribute.

    Quarto allows the attribute on the caption line just below the table
    OR on a `:::` div fence surrounding the table. We look up to 6 lines
    before the table start and 6 lines after the table end.
    """
    n = len(lines)
    lo = max(0, table_start - 6)
    hi = min(n, table_end + 6)
    for k in range(lo, hi):
        if _TBL_COLWIDTHS_RE.search(lines[k]):
            return True
    return False


def check(
    file_path: Path,
    text: str,
    scope: str,
    start_counter: int = 0,
) -> tuple[list[Issue], int]:
    """Scan for long tables that should declare longtable handling.

    Args:
      file_path: source file path (for issue records).
      text: file content as a single string.
      scope: scope tag (vol1 / vol2 / both) used for issue IDs.
      start_counter: monotonic counter for issue-id generation.

    Returns:
      (issues, next_counter)
    """
    issues: list[Issue] = []
    counter = start_counter
    lines = text.split("\n")

    n = len(lines)
    in_code_fence = False
    in_raw_block = False
    i = 0

    while i < n:
        line = lines[i]

        # Toggle code-fence state.
        if _CODE_FENCE_RE.match(line):
            if _RAW_FENCE_RE.match(line):
                in_raw_block = not in_raw_block
            else:
                in_code_fence = not in_code_fence
            i += 1
            continue

        if in_code_fence or in_raw_block:
            i += 1
            continue

        # ── Pipe-table detection ───────────────────────────────────────
        if (
            _PIPE_ROW_RE.match(line)
            and i + 1 < n
            and _PIPE_SEP_RE.match(lines[i + 1])
        ):
            table_start = i  # 0-indexed line of header row
            # Walk forward across the data rows.
            j = i + 2
            data_row_count = 0
            total_chars = 0
            while j < n and _PIPE_ROW_RE.match(lines[j]):
                data_row_count += 1
                total_chars += len(lines[j])
                j += 1
            table_end = j  # one past the last data row

            header_line = lines[table_start]
            col_count = _count_columns(header_line)
            avg_chars = (total_chars / data_row_count) if data_row_count else 0
            has_colwidths = _nearby_has_colwidths(lines, table_start, table_end)

            # Apply heuristics.
            reasons: list[str] = []
            if data_row_count >= ROW_THRESHOLD:
                reasons.append(f"{data_row_count} data rows (>= {ROW_THRESHOLD})")
            if col_count >= COL_THRESHOLD:
                reasons.append(f"{col_count} columns (>= {COL_THRESHOLD})")
            if (
                has_colwidths
                and data_row_count >= WIDE_ROW_THRESHOLD
                and avg_chars >= WIDE_AVG_CHARS
            ):
                reasons.append(
                    f"wide rows ({data_row_count} rows, avg {avg_chars:.0f} chars) "
                    "with tbl-colwidths set"
                )

            # Only flag if the table is "likely too long" AND does NOT
            # already carry the colwidths attribute. The colwidths case
            # in `reasons` is the explicit-wide-with-colwidths heuristic
            # (c) and is itself the signal that we may still need
            # longtable handling even with the attribute.
            should_flag = bool(reasons) and (not has_colwidths or reasons[-1].startswith("wide rows"))

            if should_flag:
                preview = header_line.strip()
                if len(preview) > 100:
                    preview = preview[:97] + "..."
                reason_text = "; ".join(reasons)
                suggested_after = (
                    "Add `{tbl-colwidths=\"[...]\"}` to the caption "
                    "attribute list and verify the rendered PDF repeats "
                    "the header row across page breaks."
                )
                issues.append(
                    Issue(
                        id=make_issue_id(scope, CATEGORY, counter),
                        category=CATEGORY,
                        rule=RULE,
                        rule_text=RULE_TEXT,
                        file=str(file_path),
                        line=table_start + 1,  # 1-indexed
                        col=0,
                        before=preview,
                        suggested_after=suggested_after,
                        auto_fixable=False,
                        needs_subagent=False,
                        confidence="medium",
                        reason=(
                            f"Long pipe table at line {table_start + 1}: "
                            f"{reason_text}. Needs explicit longtable handling "
                            "or the print build risks header drop / counter reset."
                        ),
                    )
                )
                counter += 1

            i = j
            continue

        # ── HTML <table> detection ─────────────────────────────────────
        if _HTML_TABLE_OPEN_RE.search(line):
            html_start = i
            tr_count = _HTML_TR_RE.findall(line).__len__()
            j = i + 1
            closed = _HTML_TABLE_CLOSE_RE.search(line) is not None
            while j < n and not closed:
                tr_count += len(_HTML_TR_RE.findall(lines[j]))
                if _HTML_TABLE_CLOSE_RE.search(lines[j]):
                    closed = True
                    break
                j += 1
            html_end = j + 1 if closed else j
            # Subtract 1 for the header <tr> if there's a <th> nearby —
            # we don't bother; the threshold accounts for it.
            if tr_count >= ROW_THRESHOLD:
                has_colwidths = _nearby_has_colwidths(lines, html_start, html_end)
                if not has_colwidths:
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
                            line=html_start + 1,
                            col=0,
                            before=preview,
                            suggested_after=(
                                "Convert to a markdown pipe table with "
                                "`{tbl-colwidths=\"[...]\"}` or wrap in a "
                                "longtable-aware div so the PDF build "
                                "repeats header rows."
                            ),
                            auto_fixable=False,
                            needs_subagent=False,
                            confidence="medium",
                            reason=(
                                f"HTML <table> at line {html_start + 1} has "
                                f"~{tr_count} <tr> rows — likely overflows one "
                                "printed page without longtable handling."
                            ),
                        )
                    )
                    counter += 1
            i = html_end
            continue

        i += 1

    return issues, counter


# ── Optional: PDF-render verification ───────────────────────────────────────


def verify_rendered_pdf(pdf_path: Path) -> list[Issue]:
    r"""Verify the rendered PDF for table-counter resets (optional, stub).

    Design note (deferred — see module docstring):

    A faithful implementation would:

      1. Extract text from the rendered PDF with page boundaries
         preserved (pdfplumber, PyMuPDF, or `pdftotext -layout`).
      2. Find every "Table N.M" or "Table N.M (continued)" caption.
      3. Map each caption to its source `{#tbl-...}` ID via the
         intermediate `.tex` (Quarto emits `\caption{... \label{tbl-...}}`).
      4. Flag any source ID that appears with two non-adjacent numeric
         labels (counter reset) OR any caption that spans two pages
         without a "(continued)" suffix on the second page.

    This is materially more work than the source-level heuristic and
    requires the PDF to already be built — which means it can only run
    after the print build, not in the pre-render scan stage. The
    source-level heuristic above catches the *cause* (missing
    `tbl-colwidths` on a likely-long table); this function would catch
    the *symptom* (an actual counter reset in the rendered output). The
    source-level catch is strictly more useful for the release gate, so
    we ship the source-level check and leave the PDF verifier as a stub
    with this design note.

    Returns:
      Empty list. (Stub.)
    """
    return []
