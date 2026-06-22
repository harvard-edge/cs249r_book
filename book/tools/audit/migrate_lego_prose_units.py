#!/usr/bin/env python3
"""Strip redundant unit tokens after ``{python} *_str`` refs when fmt suffix exists."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENTS = REPO_ROOT / "book" / "quarto" / "contents"

STRIP_AFTER_REF = re.compile(
    r"(`\{python\}\s+[A-Za-z_][\w.]*_str`)\s+"
    r"(ms|mW|MW|GW|kW|Wh|kWh|MWh|GWh|"
    r"GB|MB|KB|GiB|TiB|TB|Gb|"
    r"seconds?|secs?|minutes?|mins?|hours?|hrs?|weeks?|months?|years?|"
    r"percent|GPUs?|QPS|FLOPS|TFLOP/?s|PFLOP/?s|"
    r"flights?|tokens?|images?|nodes?|servers?|"
    r"USD|\$|%)\b",
    re.I,
)


def migrate_line(line: str) -> tuple[str, int]:
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return m.group(1)

    return STRIP_AFTER_REF.sub(repl, line), count


def migrate_file(path: Path, *, dry_run: bool) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    total = 0
    new_lines = []
    for line in lines:
        new_line, n = migrate_line(line)
        total += n
        new_lines.append(new_line)
    if total and not dry_run:
        path.write_text("".join(new_lines), encoding="utf-8")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.paths:
        paths = []
        for p in args.paths:
            p = p if p.is_absolute() else REPO_ROOT / p
            if p.is_dir():
                paths.extend(sorted(p.rglob("*.qmd")))
            elif p.suffix == ".qmd":
                paths.append(p)
    else:
        paths = sorted(CONTENTS.rglob("*.qmd"))
    grand = 0
    for path in paths:
        n = migrate_file(path, dry_run=args.dry_run)
        if n:
            grand += n
            print(f"{path.relative_to(REPO_ROOT)}: {n}")
    print(f"{'Would strip' if args.dry_run else 'Stripped'} {grand} prose unit tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
