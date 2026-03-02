#!/usr/bin/env python3
"""
One-off: fix malformed dimensions and revert multiplier/range × to N$\\times$ form.

Run order:
  1. Fix malformed dimensions: $N\\times$ M -> $N\\times M$ (so both numbers in one math span).
  2. Revert range: $N\\text{--}M\\times$ -> N--M$\\times$
  3. Revert single number: $N\\times$ -> N$\\times$

After this, only true dimensions (two numbers) remain as $N\\times M$; multipliers
use N$\\times$ per LaTeX/typography convention.
"""
from __future__ import annotations

import re
from pathlib import Path


def run(text: str) -> str:
    # 1. Malformed dimension: $N\times$ M (space before second number) -> $N\times M$
    # Use [\\] to match one backslash so we avoid escape-level confusion
    text = re.sub(
        r"[\$](\d[\d,.]*)[\\]times[\$] +(\d[\d,.]*)",
        r"$\g<1>\\times\g<2>$",
        text,
    )
    # 2. Revert range: $N\text{--}M\times$ -> N--M$\times$
    text = re.sub(
        r"[\$]([\d,.]+)[\\]text\{--\}([\d,.]+)[\\]times[\$]",
        r"\g<1>--\g<2>$\\times$",
        text,
    )
    # 3. Revert single number: $N\times$ -> N$\times$ (negative lookahead avoids dimension case)
    text = re.sub(
        r"[\$](\d[\d,.]*)[\\]times[\$](?!\s*\d)",
        r"\g<1>$\\times$",
        text,
    )
    return text


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "contents"
    if not root.is_dir():
        raise SystemExit(f"Contents root not found: {root}")
    qmd_files = list(root.rglob("*.qmd"))
    updated = 0
    for path in sorted(qmd_files):
        content = path.read_text(encoding="utf-8")
        new_content = run(content)
        if new_content != content:
            path.write_text(new_content, encoding="utf-8")
            updated += 1
            print(path.relative_to(root.parent))
    print(f"Updated {updated} files.")


if __name__ == "__main__":
    main()
