#!/usr/bin/env python3
"""Check Stage 4 (OUTPUT) purity and formatting discipline in LEGO cells.

This powers:
    ./binder/binder check code --scope lego-stage4-purity

Rules:
1. Stage 4 Isolation: Stage 4 must contain strictly display assignments.
   Intermediate variables (e.g. arithmetic calculations, helper closures, loops)
   belong in Stage 2 (EXECUTE).
2. Display Suffix Discipline: Every variable defined in Stage 4 must terminate in
   an approved suffix: '_str', '_math', '_eq', '_frac', or '_mult_str'.
3. Formatter Helper Usage: Format outputs using canonical 'mlsysim.fmt' helpers.
   Avoid redundant double-wrapping like 'MarkdownStr(fmt(...))'.
4. Semantic Guidance: Percent outputs should use '*_pct_str' to communicate that
   the '%' glyph is already closed.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
STAGE4_COMMENT = re.compile(r"^\s*#.*?\b4\.\s*(OUTPUT|EXPORT)\b", re.M | re.I)
STAGE_ANY = re.compile(r"^\s*#.*?\b([1-4])\.\s*(LOAD|EXECUTE|GUARD|OUTPUT|ASSERT|COMPUTE|EXPORT)\b", re.M | re.I)

VALID_SUFFIXES = ("_str", "_math", "_eq", "_frac", "_mult_str")


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    code: str
    message: str


def _python_cells(lines: list[str]) -> Iterable[tuple[int, list[tuple[int, str]]]]:
    in_cell = False
    start_line = 0
    cell: list[tuple[int, str]] = []

    for lineno, line in enumerate(lines, 1):
        if not in_cell and CELL_START.match(line):
            in_cell = True
            start_line = lineno
            cell = []
            continue
        if in_cell and CELL_END.match(line):
            yield start_line, cell
            in_cell = False
            start_line = 0
            cell = []
            continue
        if in_cell:
            cell.append((lineno, line))


def audit_file(path: Path) -> list[Violation]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    violations: list[Violation] = []

    for start_line, cell in _python_cells(lines):
        clean_lines = []
        for _, line in cell:
            if line.strip().startswith("#|"):
                clean_lines.append("")
            else:
                clean_lines.append(line)
        exec_code = "\n".join(clean_lines)

        try:
            tree = ast.parse(exec_code)
        except SyntaxError:
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue

            # Find line where Stage 4 starts inside cell
            stage4_rel_lineno = None
            for idx, (_, line) in enumerate(cell, 1):
                if STAGE4_COMMENT.search(line):
                    stage4_rel_lineno = idx
                    break

            if stage4_rel_lineno is None:
                continue

            # Check statements inside class body
            for item in node.body:
                if item.lineno < stage4_rel_lineno:
                    continue

                abs_line = cell[item.lineno - 1][0] if item.lineno <= len(cell) else start_line

                # Check assignments in Stage 4
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if not isinstance(target, ast.Name):
                            continue
                        var_name = target.id
                        if var_name.startswith("_"):
                            continue

                        # Check 1: Suffix discipline
                        if not any(var_name.endswith(s) for s in VALID_SUFFIXES):
                            violations.append(
                                Violation(
                                    file=str(path),
                                    line=abs_line,
                                    code="lego_stage4_intermediate_var",
                                    message=(
                                        f"Intermediate variable '{var_name}' defined in Stage 4 (OUTPUT). "
                                        "Stage 4 is reserved strictly for prose-facing formatted strings "
                                        "ending in '_str', '_math', '_eq', '_frac', or '_mult_str'. "
                                        "Move calculations to Stage 2 (EXECUTE)."
                                    ),
                                )
                            )
                            continue

                        # Check 2: Raw arithmetic in Stage 4
                        if isinstance(item.value, (ast.BinOp, ast.UnaryOp)):
                            violations.append(
                                Violation(
                                    file=str(path),
                                    line=abs_line,
                                    code="lego_stage4_raw_arithmetic",
                                    message=(
                                        f"Raw arithmetic calculation assigned to '{var_name}' in Stage 4. "
                                        "Compute numeric values in Stage 2 (EXECUTE) and format using "
                                        "canonical mlsysim.fmt helpers in Stage 4."
                                    ),
                                )
                            )

                        # Check 3: Double wrapping MarkdownStr(fmt(...)) or MarkdownStr(f"...{fmt(...)}...")
                        if isinstance(item.value, ast.Call):
                            func = item.value.func
                            fn_name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
                            if fn_name == "MarkdownStr":
                                for sub in ast.walk(item.value):
                                    if sub is item.value:
                                        continue
                                    if isinstance(sub, ast.Call):
                                        sub_f = sub.func
                                        sub_fn = sub_f.id if isinstance(sub_f, ast.Name) else (sub_f.attr if isinstance(sub_f, ast.Attribute) else "")
                                        if sub_fn.startswith("fmt") or sub_fn in ("sci_latex",):
                                            violations.append(
                                                Violation(
                                                    file=str(path),
                                                    line=abs_line,
                                                    code="lego_stage4_double_wrapping",
                                                    message=(
                                                        f"Redundant wrapper '{var_name} = MarkdownStr(...{sub_fn}(...)...)'. "
                                                        "Canonical formatters already return MarkdownStr. Use typed formatters "
                                                        "(e.g. fmt_qty, fmt_time, fmt_range) directly."
                                                    ),
                                                )
                                            )
                                            break

    return violations


def audit_paths(paths: Iterable[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for p in paths:
        if p.is_file() and p.suffix == ".qmd":
            violations.extend(audit_file(p))
        elif p.is_dir():
            for qmd in sorted(p.rglob("*.qmd")):
                violations.extend(audit_file(qmd))
    return violations


def qmd_files(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".qmd":
            result.append(p)
        elif p.is_dir():
            result.extend(sorted(p.rglob("*.qmd")))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Stage 4 (OUTPUT) purity in LEGO cells.")
    parser.add_argument("files", nargs="*", help="Files or directories to check")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.files] if args.files else [Path("books")]
    violations = audit_paths(paths)

    if violations:
        by_file: dict[str, list[Violation]] = {}
        for v in violations:
            by_file.setdefault(v.file, []).append(v)
        for f, f_viols in by_file.items():
            print(f"\n{f}")
            for v in f_viols:
                print(f"  - L{v.line} [{v.code}]: {v.message}")
        print(f"\nTotal violations: {len(violations)}")
        return 1

    print("All Stage 4 purity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
