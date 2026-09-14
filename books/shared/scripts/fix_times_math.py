#!/usr/bin/env python3
"""
Fix inline math for dimension-like N×M only (two numbers) in QMD files.

When a digit is adjacent to $ (e.g. 224$\\times$224), some Quarto/Pandoc
configurations fail to open math mode. Wrapping the full expression in one
math span (e.g. $224\\times224$) fixes rendering. Only dimension patterns
(two numbers) are fixed here; multiplier "N×" is left as N$\\times$ per
LaTeX/typography convention (number in text, symbol in math).
"""
from __future__ import annotations

import re
from pathlib import Path


def fix_times_math(text: str) -> str:
    # Malformed dimension: $N\times$ M (space before second number) -> $N\times M$
    text = re.sub(
        r"[\$](\d[\d,.]*)[\\]times[\$] +(\d[\d,.]*)",
        r"$\g<1>\\times\g<2>$",
        text,
    )
    # Two numbers: 4096$\times$4096 -> $4096\times4096$
    text = re.sub(
        r"(\d[\d,.]*)[\$][\\]times[\$](\d[\d,.]*)",
        r"$\1\\times\2$",
        text,
    )
    return text


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "contents"
    if not root.is_dir():
        raise SystemExit(f"Contents root not found: {root}")
    qmd_files = list(root.rglob("*.qmd"))
    total_replacements = 0
    for path in sorted(qmd_files):
        content = path.read_text(encoding="utf-8")
        new_content = fix_times_math(content)
        if new_content != content:
            path.write_text(new_content, encoding="utf-8")
            total_replacements += 1
            print(path.relative_to(root.parent))
    print(f"Updated {total_replacements} files.")


if __name__ == "__main__":
    main()
