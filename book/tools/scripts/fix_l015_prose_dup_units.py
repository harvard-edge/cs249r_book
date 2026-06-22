#!/usr/bin/env python3
"""Remove prose unit words duplicated after closed-fixed {python} exports (L015)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Unit token immediately after a closed-fixed export ref.
PROSE_DUP = re.compile(
    r"( `\{python\}\s*[\w.]+\.\w+_(?:w|kw|mw|wh|kwh|mwh|gb|tb|ms|s|kg)_str`) "
    r"(?:W|MW|kWh|MWh|GB|TB|ms|s|kg)\b",
    re.I,
)


def fix_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    new_text, n = PROSE_DUP.subn(r"\1", text)
    if n:
        path.write_text(new_text, encoding="utf-8")
    return n


def main() -> int:
    root = Path(__file__).resolve().parents[3] / "book" / "quarto" / "contents"
    grand = 0
    for path in sorted(root.rglob("*.qmd")):
        n = fix_file(path)
        if n:
            print(f"{n:4d}  {path.relative_to(root.parent.parent.parent)}")
            grand += n
    print(f"Removed {grand} duplicate unit tokens from prose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
