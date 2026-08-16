#!/usr/bin/env python3
"""Catch inline `{python}` references that run before the cell that defines them.

Quarto executes a chapter's cells top to bottom, so a prose reference to
`SomeClass.value_str` fails at render time unless the LEGO cell defining
`SomeClass` appears earlier in the file. The failure is a bare NameError from
deep inside the render, hundreds of lines from the real cause, and it does not
reproduce when the cells are executed on their own -- only a full render finds
it, and a full render costs ~20 minutes per volume.

This finds the same defect in about a second.

Added 2026-08-16 after a prose paragraph was reordered above its own LEGO cell
in vol2/responsible_ai and broke the Volume II HTML build at chapter 28 of 38.

Usage:
    check_inline_ref_order.py [path ...]      # default: book/quarto/contents
Exit status is 1 when any violation is found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CELL = re.compile(r"^```\{python\}\s*$(.*?)^```\s*$", re.M | re.S)
CLASSDEF = re.compile(r"^\s*class\s+(\w+)", re.M)
ASSIGN = re.compile(r"^\s*(\w+)\s*=", re.M)
INLINE = re.compile(r"`\{python\}\s*([A-Za-z_]\w*)(?:\.\w+)*\s*`")


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")

    # (offset, kind, payload) for every cell body and every inline reference
    events: list[tuple[int, str, object]] = []
    for m in CELL.finditer(text):
        events.append((m.start(), "cell", m.group(1)))
    for m in INLINE.finditer(text):
        # skip references that live inside a python cell (they are code, not prose)
        events.append((m.start(), "ref", (m.group(1), m.group(0))))

    cell_spans = [(m.start(), m.end()) for m in CELL.finditer(text)]

    def inside_cell(pos: int) -> bool:
        return any(a <= pos < b for a, b in cell_spans)

    defined: set[str] = set()
    problems: list[str] = []
    for pos, kind, payload in sorted(events, key=lambda e: e[0]):
        if kind == "cell":
            body = payload
            defined.update(CLASSDEF.findall(body))
            defined.update(ASSIGN.findall(body))
            continue
        name, raw = payload
        if inside_cell(pos):
            continue
        if name not in defined:
            line = text.count("\n", 0, pos) + 1
            problems.append(f"{path}:{line}: `{name}` referenced before it is defined  ({raw})")
    return problems


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [Path("book/quarto/contents")]
    files: list[Path] = []
    for r in roots:
        files.extend(sorted(r.rglob("*.qmd")) if r.is_dir() else [r])
    files = [f for f in files if "_shelved" not in str(f)]

    all_problems: list[str] = []
    for f in files:
        all_problems.extend(check_file(f))

    if all_problems:
        print(f"inline references used before definition: {len(all_problems)}")
        for p in all_problems:
            print("  " + p)
        return 1
    print(f"OK: {len(files)} files, every inline reference is defined before use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
