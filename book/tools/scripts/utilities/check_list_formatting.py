#!/usr/bin/env python3
"""
Ensure markdown lists are preceded by a blank line.

Markdown requires a blank line before the first item of a list for it to
render as a proper list block.  Without it, Quarto / Pandoc treat the items
as continuation of the preceding paragraph.

Detected patterns
-----------------
1. A non-blank, non-list line immediately followed by ``1. `` (numbered list start).
2. A non-blank, non-list line immediately followed by ``- `` (bullet list start).

Lines that are themselves list items, blank lines, code fences, Quarto
divs (``:::``), table rows (``|``), and YAML front-matter are excluded
from triggering a violation.

Usage (from book/):
    python3 tools/scripts/utilities/check_list_formatting.py --check quarto/contents/vol2/
    python3 tools/scripts/utilities/check_list_formatting.py --fix   file.qmd
    python3 tools/scripts/utilities/check_list_formatting.py --fix   --recursive quarto/contents/

Pre-commit (configured in .pre-commit-config.yaml):
    entry: python tools/scripts/utilities/check_list_formatting.py --fix

Exit codes:
    0  No issues (or all fixed in --fix mode)
    1  Issues found (--check mode only)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ── Patterns ────────────────────────────────────────────────────────────────

# First item of a numbered list (must start with "1." to be a list start)
RE_NUMBERED_START = re.compile(r"^\s*1\.\s")
# Any numbered continuation (2., 3., …)
RE_NUMBERED_ANY = re.compile(r"^\s*\d+\.\s")
# Bullet list item
RE_BULLET = re.compile(r"^\s*- \S")

# Lines that should never trigger a "missing blank line" violation
RE_SKIP = re.compile(
    r"^\s*$"          # blank
    r"|^\s*\d+\.\s"   # numbered list item
    r"|^\s*- "         # bullet list item
    r"|^```"           # code fence
    r"|^:::"           # Quarto div
    r"|^\s*\|"         # table row
    r"|^---"           # YAML delimiter / horizontal rule
    r"|^#"             # heading or YAML comment
    r"|^\[\^fn-"       # footnote definition
)


def _in_code_block(lines: list[str], idx: int) -> bool:
    """Return True if *idx* falls inside a fenced code block."""
    fences = 0
    for i in range(idx):
        if lines[i].startswith("```"):
            fences += 1
    return fences % 2 == 1


def check_file(path: Path, *, fix: bool = False) -> list[tuple[int, str]]:
    """Return list of (line_number, offending_line) violations.

    If *fix* is True, rewrite the file with blank lines inserted.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    violations: list[tuple[int, str]] = []
    insert_before: set[int] = set()

    for i in range(1, len(lines)):
        prev = lines[i - 1]
        curr = lines[i]

        # Current line must be the *start* of a list
        is_list_start = bool(RE_NUMBERED_START.match(curr) or RE_BULLET.match(curr))
        if not is_list_start:
            continue

        # Previous line must NOT be skippable
        if RE_SKIP.match(prev):
            continue

        # Don't flag inside code blocks
        if _in_code_block(lines, i):
            continue

        violations.append((i + 1, prev.strip()))  # 1-indexed
        insert_before.add(i)

    if fix and insert_before:
        new_lines: list[str] = []
        for i, line in enumerate(lines):
            if i in insert_before:
                new_lines.append("")
            new_lines.append(line)
        path.write_text("\n".join(new_lines), encoding="utf-8")

    return violations


def collect_files(paths: list[str], recursive: bool) -> list[Path]:
    """Expand CLI paths into a list of .qmd files."""
    result: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_file():
            result.append(pp)
        elif pp.is_dir():
            glob = "**/*.qmd" if recursive else "*.qmd"
            result.extend(sorted(pp.glob(glob)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check / fix blank lines before markdown lists in QMD files."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Report without fixing (exit 1 if issues)")
    mode.add_argument("--fix", action="store_true", help="Auto-insert blank lines")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recurse into directories")
    parser.add_argument("files", nargs="*", default=[], help="Files or directories to scan")
    args = parser.parse_args()

    # Default to --check if neither flag given
    do_fix = args.fix

    files = collect_files(args.files, recursive=args.recursive or do_fix)
    if not files:
        return 0

    total = 0
    for f in files:
        violations = check_file(f, fix=do_fix)
        if violations:
            total += len(violations)
            rel = f
            for lineno, context in violations:
                action = "Fixed" if do_fix else "Missing blank line before list"
                print(f"{rel}:{lineno}: {action} — after: {context[:80]}")

    if total:
        if do_fix:
            print(f"\n✅ Fixed {total} list formatting issue(s).")
            return 0
        else:
            print(f"\n❌ Found {total} list formatting issue(s). Run with --fix to auto-repair.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
