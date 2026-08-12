#!/usr/bin/env python3
"""Split closing ``` fences glued to the last line of a {python} cell."""

from __future__ import annotations

import argparse
from pathlib import Path


def fix_text(text: str) -> tuple[str, int]:
    fixes = 0
    out: list[str] = []
    for line in text.splitlines():
        if line.endswith("```") and line.strip() != "```" and not line.lstrip().startswith("```{"):
            idx = line.rfind("```")
            out.append(line[:idx])
            out.append("```")
            fixes += 1
        else:
            out.append(line)
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + trailing, fixes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("book/quarto/contents"),
        help="Root directory of QMD files",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = 0
    for path in sorted(args.root.rglob("*.qmd")):
        original = path.read_text(encoding="utf-8")
        fixed, n = fix_text(original)
        if n:
            total += n
            print(f"{path}: {n}")
            if not args.dry_run:
                path.write_text(fixed, encoding="utf-8")
    print(f"{'Would fix' if args.dry_run else 'Fixed'} {total} glued fence(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
