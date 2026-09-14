#!/usr/bin/env python3
"""
Block scratch files, OS junk, and machine-local paths from entering the repo.

Why this exists
---------------
A 2026-09-14 audit found five build scripts in the Volume III and IV roots.
Two of them read chapter-opener art from an AI tool's session directory under
a personal home folder, so nothing but one laptop could run them. Three macOS
AppleDouble files rode along in Volume III.

What this blocks
----------------
- AppleDouble resource forks (``._name``) and other OS junk files.
- Scripts placed directly in a volume root, such as ``books/vol4/build.py``.
  Generators belong under ``binder/tools/scripts/``.
- Machine-local paths in file contents: working directories inside a personal
  home folder (``/Users/<name>/GitHub``, ``Documents``, ``Desktop``, and so on)
  and AI tool session state.

Usage
-----
  python3 shared/scripts/check-stray-files.py FILE [FILE ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

JUNK_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
VOLUME_ROOT_SCRIPT = re.compile(r"^books/vol\d+/[^/]+\.(?:py|sh|ipynb|js|mjs|rb)$")

# Split literals keep this file from matching its own patterns.
MACHINE_LOCAL = re.compile(
    "|".join([
        r"/Users/[A-Za-z0-9._-]+/(?:GitHub|Desktop|Documents|Downloads|Library|Projects|Dropbox"
        r"|\.gemini|\.claude|\.codex|\.cursor)/",
        "antigravity" + r"-cli/brain",
        r"\.claude/" + "projects/",
        "/private/tmp/" + "claude-",
    ])
)


def problems(rel: str) -> list[str]:
    found = []
    name = Path(rel).name
    if name.startswith("._") or name in JUNK_NAMES:
        found.append(f"{rel}: OS junk file; delete it")
    if VOLUME_ROOT_SCRIPT.match(rel):
        found.append(f"{rel}: script in a volume root; move it under binder/tools/scripts/")
    path = Path(rel)
    if path.is_file() and not path.is_symlink():
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            return found
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = MACHINE_LOCAL.search(line)
            if match:
                found.append(f"{rel}:{lineno}: machine-local path {match.group(0)!r}; use a repo-relative path")
    return found


def main(argv: list[str]) -> int:
    all_problems = [p for rel in argv for p in problems(rel)]
    for problem in all_problems:
        print(problem)
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
