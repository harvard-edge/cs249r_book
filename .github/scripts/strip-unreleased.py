#!/usr/bin/env python3
"""Remove unreleased-volume blocks from site sources before a live build.

Usage: strip-unreleased.py <file> [<file> ...]

A live publish workflow runs this on a fresh checkout, before `quarto render`,
so the live site never builds pages, navigation entries, or landing cards for
volumes that are still staging-only. Dev previews skip it and keep everything.

Mark a block with two lines, each containing the marker text in whatever
comment syntax the file uses:

    # unreleased:start            (YAML)
    <!-- unreleased:start -->     (Markdown / Quarto)
    ...
    # unreleased:end

Both marker lines and everything between them are removed. An unbalanced or
nested marker fails the run, so a typo cannot ship a half-stripped page.

To release a volume, delete its markers (keeping the content) and drop it from
the workflow's list of unreleased paths.
"""

import sys
from pathlib import Path

START = "unreleased:start"
END = "unreleased:end"


def strip(path: Path) -> int:
    kept, removed, inside, start_line = [], 0, False, 0
    for number, line in enumerate(path.read_text().splitlines(keepends=True), 1):
        if START in line:
            if inside:
                sys.exit(f"{path}:{number}: nested '{START}' (block opened at line {start_line})")
            inside, start_line = True, number
        if inside:
            removed += 1
        else:
            kept.append(line)
        if END in line:
            if not inside:
                sys.exit(f"{path}:{number}: '{END}' without a matching '{START}'")
            inside = False
    if inside:
        sys.exit(f"{path}:{start_line}: '{START}' is never closed")
    path.write_text("".join(kept))
    return removed


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for name in sys.argv[1:]:
        removed = strip(Path(name))
        print(f"   {name}: removed {removed} unreleased line(s)")


if __name__ == "__main__":
    main()
