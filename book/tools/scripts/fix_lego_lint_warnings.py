#!/usr/bin/env python3
"""Bulk-fix common LEGO lint warnings (L007, L008, L009) in QMD python blocks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

UREG_REPLACEMENTS = [
    (r"ureg\.millijoule\b", "mJ"),
    (r"ureg\.megawatt\b", "MW"),
    (r"ureg\.joule\b", "joule"),
    (r"ureg\.kilowatt_hour\b", "kWh"),
    (r"ureg\.megawatt_hour\b", "MWh"),
    (r"ureg\.watt_hour\b", "Wh"),
    (r"ureg\.kilogram\b", "kg"),
    (r"ureg\.millisecond\b", "ms"),
    (r"ureg\.microsecond\b", "microsecond"),
    (r"ureg\.minute\b", "minute"),
    (r"ureg\.metric_ton\b", "metric_ton"),
    (r"ureg\.picojoule\b", "pJ"),
    (r"ureg\.kibibyte\b", "KiB"),
    (r"ureg\.mebibyte\b", "MiB"),
    (r"ureg\.terabit\s*/\s*second\b", "terabit / second"),
]

TO_UNIT_REPLACEMENTS = [
    (".to(US)", ".to(microsecond)"),
    (".to(NS)", ".to(nanosecond)"),
    (".to(MS)", ".to(ms)"),
]

# Unparenthesized rate: N * TB / second -> N * (TB / second)
RATE_FIX = re.compile(
    r"(\d+(?:\.\d+)?)\s*\*\s*(TB|GB|TFLOP|MB|PFLOPs|TFLOPs)\s*/\s*second\b"
)


def fix_python_block(block: str) -> tuple[str, int]:
    changes = 0
    lines: list[str] = []
    for line in block.splitlines():
        orig = line
        if line.lstrip().startswith("#"):
            lines.append(line)
            continue
        for old, new in TO_UNIT_REPLACEMENTS:
            if old in line:
                line = line.replace(old, new)
        for pat, repl in UREG_REPLACEMENTS:
            line = re.sub(pat, repl, line)
        line = RATE_FIX.sub(r"\1 * (\2 / second)", line)
        if line != orig:
            changes += 1
        lines.append(line)
    return "\n".join(lines), changes


def fix_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    total = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal total
        body, n = fix_python_block(match.group(1))
        total += n
        return f"```{{python}}{body}```"

    new_text = re.sub(r"```\{python\}(.*?)```", repl, text, flags=re.S)
    if total:
        path.write_text(new_text, encoding="utf-8")
    return total


def main() -> int:
    root = Path(__file__).resolve().parents[3] / "book" / "quarto" / "contents"
    paths = sorted(root.rglob("*.qmd"))
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    grand = 0
    for path in paths:
        n = fix_file(path)
        if n:
            print(f"{n:4d}  {path.relative_to(root.parent.parent.parent)}")
            grand += n
    print(f"Fixed {grand} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
