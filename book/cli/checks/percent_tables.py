#!/usr/bin/env python3
"""Percent-symbol-in-tables: single source of truth for the check and the fixer.

House style inverts between prose and tables:

  - In PROSE and CAPTIONS, "percent" is spelled out (MIT Press §10.2); the
    `%` symbol is banned there.
  - In TABLES, cells are dense tabular data and the conventional form is the
    `%` symbol: `86.4%`, not `86.4 percent`.

This module enforces the table side. It is imported by BOTH:

  - the validator scope ``_run_mitpress_percent_in_tables`` in
    ``book/cli/commands/validate.py`` (the check that blocks commits), and
  - the formatter target ``_run_percent_tables`` in
    ``book/cli/commands/formatting.py`` (``./book/binder format percent-tables``).

Keeping the detection (`find_in_text`) and the rewrite (`fix_text`) here — and
having both the check and the fixer import them — means the two can never
drift: the fixer fixes exactly what the check flags, by construction.

A pipe table is detected the way every other table check in the binder does
it: a header row immediately followed by a ``|---|`` separator row, then the
body rows. Only those rows (excluding the separator) are scanned, and only a
number-token immediately before "percent" matches, so header labels like
"Percent of total" are never touched. ``\\bpercent\\b`` deliberately excludes
"percentage" / "percentage points", which are a different unit (pp) and stay
spelled out.

Run standalone:
  python3 -m cli.checks.percent_tables --check book/quarto/contents/
  python3 -m cli.checks.percent_tables book/quarto/contents/   # apply
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

# A pipe-table data/header row and the alignment separator row that marks the
# table header. Identical to validate.py's _PIPE_ROW_RE / _PIPE_SEP_RE.
PIPE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
PIPE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

# <number-token> <space> percent
#   number-token = digit run (decimal/comma, optional hyphen/en/em-dash range)
#                  OR an inline code span `...`
# Case-insensitive on the word; the leading number is what gates the match.
PCT_PAT = re.compile(
    r"(\d[\d.,]*(?:\s*[\-–—]\s*\d[\d.,]*)?|`[^`]*`)\s+percent\b",
    re.IGNORECASE,
)

CODE_FENCE_RE = re.compile(r"^\s*```")


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    code: str
    message: str
    context: str = ""
    suggestion: str = ""


@dataclass(frozen=True)
class Hit:
    """One `<number> percent` occurrence inside a table row."""

    line: int            # 1-based line number
    match: str           # the matched text, e.g. "50 percent"
    replacement: str     # the symbol form, e.g. "50%"
    context: str         # trimmed surrounding text


def _table_row_indices(lines: List[str]) -> List[int]:
    """Return 0-based indices of lines that are table body/header rows.

    Excludes separator rows and anything inside a fenced code block. A table
    is a header row immediately followed by a ``|---|`` separator.
    """
    n = len(lines)
    rows: List[int] = []
    in_code = False
    i = 0
    while i < n:
        if CODE_FENCE_RE.match(lines[i]):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue
        if (PIPE_ROW_RE.match(lines[i])
                and i + 1 < n
                and PIPE_SEP_RE.match(lines[i + 1])):
            block_end = i + 2
            while block_end < n and PIPE_ROW_RE.match(lines[block_end]):
                block_end += 1
            for j in range(i, block_end):
                if not PIPE_SEP_RE.match(lines[j]):
                    rows.append(j)
            i = block_end
        else:
            i += 1
    return rows


def find_in_text(text: str) -> List[Hit]:
    """Find every `<number> percent` inside a pipe table in *text*."""
    lines = text.splitlines()
    hits: List[Hit] = []
    for idx in _table_row_indices(lines):
        line = lines[idx]
        for m in PCT_PAT.finditer(line):
            ctx = line[max(0, m.start() - 5): min(len(line), m.end() + 10)].strip()
            hits.append(Hit(
                line=idx + 1,
                match=m.group().strip(),
                replacement=f"{m.group(1)}%",
                context=ctx,
            ))
    return hits


def fix_text(text: str) -> Tuple[str, int]:
    """Rewrite `<number> percent` -> `<number>%` in every table row.

    Returns ``(new_text, n_lines_changed)``. Whitespace/EOL outside table
    rows is preserved exactly.
    """
    lines = text.splitlines(keepends=True)
    raw = [ln.rstrip("\n") for ln in lines]
    eols = [ln[len(r):] for ln, r in zip(lines, raw)]

    changed = 0
    for idx in _table_row_indices(raw):
        new = PCT_PAT.sub(lambda m: f"{m.group(1)}%", raw[idx])
        if new != raw[idx]:
            raw[idx] = new
            changed += 1

    if not changed:
        return text, 0
    return "".join(r + e for r, e in zip(raw, eols)), changed


# ----------------------------------------------------------------------------
# Standalone / audit surface (mirrors cli/checks/currency_style.py shape)
# ----------------------------------------------------------------------------

def iter_target_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(p for p in path.rglob("*.qmd") if p.is_file())
        elif path.is_file() and path.suffix == ".qmd":
            files.append(path)
    return sorted(dict.fromkeys(files))


def audit(paths: Iterable[Path]) -> List[Violation]:
    violations: List[Violation] = []
    for path in iter_target_files(paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        for hit in find_in_text(text):
            violations.append(Violation(
                file=str(path),
                line=hit.line,
                code="percent_in_table",
                message=(
                    "Use the % symbol, not the word 'percent', inside tables: "
                    f"'{hit.match}' → '{hit.replacement}'. Run "
                    "'./book/binder format percent-tables' to auto-fix."
                ),
                context=hit.context,
                suggestion=hit.replacement,
            ))
    return violations


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in argv
    targets = [a for a in argv if not a.startswith("--")]
    if not targets:
        print(__doc__)
        return 2

    paths = [Path(t) for t in targets]
    total = 0
    touched = 0
    for path in iter_target_files(paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        new_text, n = fix_text(text)
        if not n:
            continue
        touched += 1
        total += n
        for hit in find_in_text(text):
            print(f"{path}:{hit.line}  '{hit.match}' → '{hit.replacement}'")
        if not check:
            path.write_text(new_text, encoding="utf-8")

    verb = "would fix" if check else "fixed"
    print(f"\n{verb} {total} row(s) across {touched} file(s).")
    return 1 if (check and total) else 0


if __name__ == "__main__":
    sys.exit(main())
