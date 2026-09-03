#!/usr/bin/env python3
"""Move tbl-colwidths from wrapper divs onto table caption lines."""

from __future__ import annotations

import re
import sys
from pathlib import Path

WRAPPER_RE = re.compile(r'^::: \{tbl-colwidths="(\[[^\]]+\])"\}\s*$')
CAPTION_RE = re.compile(r'^(: .* \{#tbl-[^\s\}]+)(.*\})\s*$')

def normalize_content(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    changes = 0

    while i < len(lines):
        m = WRAPPER_RE.match(lines[i].rstrip("\n"))
        if not m:
            out.append(lines[i])
            i += 1
            continue

        colwidths = m.group(1)
        i += 1
        block: list[str] = []
        while i < len(lines):
            if lines[i].strip() == ":::":
                i += 1
                break
            block.append(lines[i])
            i += 1

        caption_idx = None
        for j, line in enumerate(block):
            cm = CAPTION_RE.match(line.rstrip("\n"))
            if not cm:
                continue
            prefix, suffix = cm.group(1), cm.group(2)
            if "tbl-colwidths=" in line:
                caption_idx = j
                break
            block[j] = f'{prefix} tbl-colwidths="{colwidths}"{suffix}\n'
            caption_idx = j
            changes += 1
            break

        if caption_idx is None:
            raise ValueError("tbl-colwidths wrapper without table caption")

        out.extend(block)

    return "".join(out), changes

def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv[1:]] if len(argv) > 1 else list(
        Path("book/quarto/contents").rglob("*.qmd")
    )
    total = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "::: {tbl-colwidths=" not in text:
            continue
        new_text, n = normalize_content(text)
        if n:
            path.write_text(new_text, encoding="utf-8")
            print(f"{path}: {n} table(s)")
            total += n
    print(f"Normalized {total} table(s)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
