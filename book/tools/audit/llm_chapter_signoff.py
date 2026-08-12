#!/usr/bin/env python3
"""Pre-commit chapter sign-off for registry-migrated QMD LEGO cells."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
ALIAS_BANNED = re.compile(
    r"Systems\.Tiers\b|"
    r"\bHardware\.(H100|A100|V100|B200|ESP32|Jetson|iPhone)\b|\bInfra\."
)
CONSTANTS_BANNED = re.compile(r"from\s+mlsysim\.core\.constants\s+import")


def python_cells(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if CELL_START.match(lines[i]):
            j = i + 1
            while j < len(lines) and not CELL_END.match(lines[j]):
                j += 1
            blocks.append("\n".join(lines[i + 1 : j]))
            i = j + 1
        else:
            i += 1
    return blocks


def check_chapter(path: Path, *, check_constants: bool) -> list[str]:
    issues: list[str] = []
    for idx, block in enumerate(python_cells(path), start=1):
        if ALIAS_BANNED.search(block):
            issues.append(f"cell {idx}: banned legacy alias pattern")
        if check_constants and CONSTANTS_BANNED.search(block):
            issues.append(f"cell {idx}: banned constants import")
    return issues


def exec_cells(path: Path) -> None:
    for block in python_cells(path):
        if not block.strip():
            continue
        code = block + "\n"
        subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qmd", type=Path)
    parser.add_argument("--exec", action="store_true", help="Execute LEGO cells")
    parser.add_argument(
        "--require-no-constants-import",
        action="store_true",
        help="Fail if cells still import mlsysim.core.units",
    )
    args = parser.parse_args()
    path = args.qmd if args.qmd.is_absolute() else REPO_ROOT / args.qmd
    issues = check_chapter(path, check_constants=args.require_no_constants_import)
    if issues:
        for issue in issues:
            print(f"FAIL {path}: {issue}")
        return 1
    if args.exec:
        exec_cells(path)
    print(f"OK {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
