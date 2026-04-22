#!/usr/bin/env python3
"""
Wrap README HTML tables in a wide, GitHub-safe frame (98% width, outer
`#cfd6dd` border via cellspacing, white inner panel, padded cells).

Handles:
  - ``<table>`` + newline + ``<thead>`` data tables (adds header cell styling)
  - ``<table>`` + newline + ``<tbody>`` body-only tables (skips ALL-CONTRIBUTORS blocks)

Does not match early-release banners (`<table border="0"`).

Usage:
  python3 wrap_readme_data_tables.py path/to/README.md [...]
"""

from __future__ import annotations

import pathlib
import sys

NEEDLE = "<table>\n  <thead>"
NEEDLE_TBODY = "<table>\n  <tbody>"
CLOSE = "</tbody>\n</table>"

OPEN_OUTER = """<div align="center">
<table width="98%" border="0" cellspacing="0" cellpadding="1" bgcolor="#cfd6dd" role="presentation"><tr><td bgcolor="#ffffff" align="left">
"""

INNER_OPEN = '<table width="100%" border="0" cellspacing="0" cellpadding="14" bgcolor="#ffffff">\n  <thead>'

INNER_TBODY = '<table width="100%" border="0" cellspacing="0" cellpadding="14" bgcolor="#ffffff">\n  <tbody>'

CLOSE_WRAPPER = """</td></tr>
</table>
</div>"""

MARK_S = "<!-- ALL-CONTRIBUTORS-LIST:START"
MARK_E = "<!-- ALL-CONTRIBUTORS-LIST:END"


def _in_contributors_block(text: str, pos: int) -> bool:
    s = text.rfind(MARK_S, 0, pos)
    if s == -1:
        return False
    e = text.find(MARK_E, s)
    return e != -1 and s < pos < e


def _style_th(line: str) -> str:
    if "<th " in line and "bgcolor=" not in line:
        return line.replace("<th ", '<th bgcolor="#eef2f7" align="left" valign="top" ', 1)
    return line


def _wrap_pass(text: str, needle: str, inner_replace: tuple[str, str]) -> tuple[str, int]:
    old, new_open = inner_replace
    pos = 0
    parts: list[str] = []
    n = 0
    while True:
        start = text.find(needle, pos)
        if start == -1:
            parts.append(text[pos:])
            break
        if _in_contributors_block(text, start):
            parts.append(text[pos : start + len(needle)])
            pos = start + len(needle)
            continue
        end = text.find(CLOSE, start)
        if end == -1:
            parts.append(text[pos:])
            break
        end += len(CLOSE)
        chunk = text[start:end]
        inner = chunk.replace(old, new_open, 1)
        if needle == NEEDLE:
            inner = "\n".join(_style_th(line) for line in inner.split("\n"))
        wrapped = OPEN_OUTER + inner + CLOSE_WRAPPER
        parts.append(text[pos:start])
        parts.append(wrapped)
        pos = end
        n += 1
    return "".join(parts), n


def process(text: str) -> tuple[str, int]:
    text, n1 = _wrap_pass(text, NEEDLE, ("<table>\n  <thead>", INNER_OPEN))
    text, n2 = _wrap_pass(text, NEEDLE_TBODY, ("<table>\n  <tbody>", INNER_TBODY))
    return text, n1 + n2


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: wrap_readme_data_tables.py README.md [...]", file=sys.stderr)
        return 2
    total = 0
    for p in argv[1:]:
        path = pathlib.Path(p)
        raw = path.read_text()
        new, count = process(raw)
        if new != raw:
            path.write_text(new)
            print(f"{path}: wrapped {count} table(s)")
            total += count
        else:
            print(f"{path}: no changes")
    print(f"Total wrapped: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
