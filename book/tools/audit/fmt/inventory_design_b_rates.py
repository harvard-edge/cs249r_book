#!/usr/bin/env python3
"""Inventory Design B multiplier and rate-name migration targets.

This is a coverage tool, not an edit tool. It freezes the exact export/ref
scope for the fmt_multiple Design B migration and the rate-name normalization
pass so manual review can work from a complete list instead of sampling.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

from audit_fmt_usage import extract_python_cells

INLINE_PY = re.compile(r"`\{python\}\s+([A-Za-z_][\w.]*)`")
CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
COMPACT_TIMES_AFTER = re.compile(r"^\s*(\$\\times\$|\\times|×|x\b)")

MULT_FUNCS = {"fmt_multiple", "fmt_multiple_range"}

RATE_SUFFIX_GROUPS = {
    "byte_per_s_old": re.compile(r"(?:_gb_s|_gbs|_tb_s|_tbs|_mb_s|_mbs)_str$"),
    "bit_per_s_old": re.compile(r"(?:_gbps|_mbps|_gbit_s|_mbit_s)_str$"),
    "compute_per_s_old": re.compile(r"(?:_tflops|_tflop_s|_flop_s)_str$"),
    "tokens_per_s_old": re.compile(r"_tokens_s_str$"),
    "acronym_rate": re.compile(r"(?:^|_)(?:qps|rps|tps)_str$"),
}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return None


def _classes(tree: ast.AST) -> list[tuple[str, int, int]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            out.append((node.name, node.lineno, node.end_lineno or node.lineno))
    return out


def _qualifier(classes: list[tuple[str, int, int]], lineno: int) -> str:
    best = None
    for name, start, end in classes:
        if start <= lineno <= end and (best is None or start > best[1]):
            best = (name, start, end)
    return f"{best[0]}." if best else ""


def _target_names(node: ast.AST) -> list[str]:
    targets = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    out = []
    for target in targets:
        if isinstance(target, ast.Name):
            out.append(target.id)
        elif isinstance(target, ast.Attribute):
            out.append(target.attr)
    return out


def _line(lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _scan_python(qmd: Path, *, include_context: bool) -> tuple[list[dict], list[dict]]:
    text = qmd.read_text(encoding="utf-8", errors="replace")
    raw_lines = text.splitlines()
    mult_exports: list[dict] = []
    rate_exports: list[dict] = []

    for fence_line, source in extract_python_cells(text):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        classes = _classes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            fname = _call_name(value)
            if not fname:
                continue
            file_line = fence_line + (node.lineno or 0)
            for name in _target_names(node):
                qual_name = _qualifier(classes, node.lineno) + name
                record = {
                    "file": str(qmd),
                    "line": file_line,
                    "name": name,
                    "qualified": qual_name,
                    "formatter": fname,
                }
                if include_context:
                    record["context"] = _line(raw_lines, file_line)
                if fname in MULT_FUNCS:
                    record["has_mult_token"] = "_mult" in name
                    mult_exports.append(record)
                for group, pattern in RATE_SUFFIX_GROUPS.items():
                    if name.endswith("_str") and pattern.search(name):
                        r = dict(record)
                        r["group"] = group
                        rate_exports.append(r)
    return mult_exports, rate_exports


def _scan_refs(
    qmd: Path, mult_by_qualified: set[str], *, include_context: bool
) -> list[dict]:
    text = qmd.read_text(encoding="utf-8", errors="replace")
    refs: list[dict] = []
    in_cell = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if in_cell:
            continue
        for match in INLINE_PY.finditer(line):
            ref = match.group(1)
            if ref not in mult_by_qualified:
                continue
            after = line[match.end(): match.end() + 36]
            times_match = COMPACT_TIMES_AFTER.match(after)
            record = {
                "file": str(qmd),
                "line": lineno,
                "ref": ref,
                "has_compact_times_after": bool(times_match),
                "times_token": times_match.group(1) if times_match else "",
            }
            if include_context:
                record["after"] = after
                record["context"] = line.strip()
            refs.append(
                record
            )
    return refs


def inventory(root: Path, *, include_context: bool = False) -> dict:
    qmds = sorted(root.rglob("*.qmd"))
    mult_exports: list[dict] = []
    rate_exports: list[dict] = []
    for qmd in qmds:
        mult, rates = _scan_python(qmd, include_context=include_context)
        mult_exports.extend(mult)
        rate_exports.extend(rates)

    mult_by_qualified = {row["qualified"] for row in mult_exports}
    mult_refs: list[dict] = []
    for qmd in qmds:
        mult_refs.extend(
            _scan_refs(qmd, mult_by_qualified, include_context=include_context)
        )

    return {
        "root": str(root),
        "summary": {
            "qmd_files": len(qmds),
            "mult_exports": len(mult_exports),
            "mult_refs": len(mult_refs),
            "mult_refs_with_times_after": sum(
                1 for row in mult_refs if row["has_compact_times_after"]
            ),
            "mult_exports_without_mult_token": sum(
                1 for row in mult_exports if not row["has_mult_token"]
            ),
            "rate_exports": len(rate_exports),
            "rate_exports_by_group": dict(
                Counter(row["group"] for row in rate_exports)
            ),
        },
        "mult_exports": mult_exports,
        "mult_refs": mult_refs,
        "rate_exports": rate_exports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("book/quarto/contents"))
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include source-line context in JSON for local/manual review.",
    )
    args = parser.parse_args()

    payload = inventory(args.root, include_context=args.include_context)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"[inventory] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
