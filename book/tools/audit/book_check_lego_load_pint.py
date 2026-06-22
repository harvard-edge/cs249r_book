#!/usr/bin/env python3
"""Static lint: physical ``*_value`` assignments must not use bare floats.

Policy
------
In LEGO ``{python}`` cells, quantities with physical units should load from
``ureg``, ``Hardware.*``, ``Systems.*``, ``Infrastructure.*``, etc. — not
bare numeric literals.

Skip dimensionless names containing: count, ratio, pct, speedup, factor,
multiplier, fraction, util, nodes, gpus, samples, queries, users, requests,
epochs, steps, layers, heads, batch, rank, slot, index, and similar counts.

Also skip when the RHS uses registry paths, ``ureg``, ``.m_as(``, or non-literal
expressions (other identifiers, calls, arithmetic).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTENTS = REPO_ROOT / "book" / "quarto" / "contents"

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
LEGO_MARK = re.compile(
    r"#\s*[│┌].*LEGO"
    r"|#\s*│\s*(?:Context|Goal|Scope|Show|How):"
    r"|#.*\b4\.\s*OUTPUT\b"
)
VALUE_ASSIGN = re.compile(r"^(\s+)(\w+_value)\s*=\s*(.+?)\s*(?:#.*)?$", re.M)
BARE_NUMERIC = re.compile(r"^[\d_]+(?:\.\d+)?(?:e[+-]?\d+)?$", re.I)

DIMENSIONLESS = re.compile(
    r"(?:^|_)(?:count|ratio|pct|percent|speedup|factor|multiplier|fraction"
    r"|util|utilization|qps|photos|images|queries|tokens|samples|users|"
    r"requests|servers|workers|gpus|nodes|flights|employees|incidents"
    r")(?:_|$|\d)",
    re.I,
)

REGISTRY_RHS = re.compile(
    r"(?:"
    r"\b(?:Hardware|Systems|Infrastructure|Literature|Ops|Datasets|Models|Monitoring|calibration)\."
    r"|\bureg\b|\.m_as\(|\.to\(|Quantity\b"
    r")",
    re.I,
)


def _is_bare_physical_value(name: str, rhs: str) -> bool:
    if DIMENSIONLESS.search(name):
        return False
    if re.search(r"(?:^|_)rate(?:_|$)", name, re.I):
        if re.search(r"(?:cost|price|usd|hour|kwh|watt|mw|gb|mb|bps|flops|carbon|grid)", name, re.I):
            return BARE_NUMERIC.match(rhs.strip()) is not None
        return False
    if REGISTRY_RHS.search(rhs):
        return False
    if BARE_NUMERIC.match(rhs.strip()) is None:
        return False
    return True


def _lego_cell_blocks(path: Path) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        if CELL_START.match(lines[i]):
            start = i + 1
            j = i + 1
            while j < len(lines) and not CELL_END.match(lines[j]):
                j += 1
            code = "\n".join(lines[start:j])
            if LEGO_MARK.search(code):
                blocks.append((i + 1, code))
            i = j + 1
        else:
            i += 1
    return blocks


def check_file(path: Path) -> list[tuple[int, str, list[str]]]:
    issues: list[tuple[int, str, list[str]]] = []
    for cell_line, code in _lego_cell_blocks(path):
        for m in VALUE_ASSIGN.finditer(code):
            name = m.group(2)
            rhs = m.group(3).strip()
            if not _is_bare_physical_value(name, rhs):
                continue
            line_no = cell_line + code[: m.start()].count("\n")
            issues.append(
                (
                    line_no,
                    m.group(0).strip()[:120],
                    [f"bare float for physical *_value {name!r} — use ureg/registry"],
                )
            )
    issues.sort(key=lambda item: item[0])
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="QMD files (default: all contents)")
    args = parser.parse_args()
    if args.paths:
        expanded: list[Path] = []
        for path in args.paths:
            p = path if path.is_absolute() else REPO_ROOT / path
            if p.is_dir():
                expanded.extend(sorted(p.rglob("*.qmd")))
            elif p.suffix == ".qmd":
                expanded.append(p)
        paths = expanded
    else:
        paths = sorted(CONTENTS.rglob("*.qmd"))
    failures = 0
    total = 0
    for path in paths:
        p = path if path.is_absolute() else REPO_ROOT / path
        if not p.exists() or p.suffix != ".qmd":
            continue
        issues = check_file(p)
        total += 1
        if not issues:
            continue
        failures += 1
        print(f"\n{p.relative_to(REPO_ROOT)}")
        for lineno, snippet, labels in issues:
            uniq = ", ".join(dict.fromkeys(labels))
            print(f"  L{lineno}: {uniq}")
            print(f"    {snippet}")
    if failures:
        print(f"\n{failures} file(s) with LEGO load/pint violations")
        return 1
    print(f"OK LEGO load/pint ({total} QMD files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
