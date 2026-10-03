#!/usr/bin/env python3
"""Check LEGO cell architecture and structural encapsulation invariants.

This powers:
    ./binder/binder check code --scope lego-cell-architecture

Rules:
1. Exactly one class per LEGO cell (no multi-class cells).
2. Proper class placement: 'class ClassName:' must be declared before Stage 1
   (no inverted cells where computational stages run at notebook module level).
3. Echo suppression: every Python cell must specify '#| echo: false'.
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
CLASS_DEF = re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M)
STAGE_COMMENT = re.compile(r"^\s*#.*?\b([1-4])\.\s*(LOAD|EXECUTE|GUARD|OUTPUT|ASSERT|COMPUTE)\b", re.M | re.I)


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    code: str
    message: str


def _python_cells(lines: list[str]) -> Iterable[tuple[int, list[tuple[int, str]]]]:
    """Yield Python cell line spans as (start_line, [(lineno, text), ...])."""
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
    """Audit one QMD file for LEGO architecture and encapsulation invariants."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    violations: list[Violation] = []

    for start_line, cell in _python_cells(lines):
        cell_text = "\n".join(line for _, line in cell)
        has_lego_banner = bool(re.search(r"#\s*┌──\s*LEGO", cell_text, re.I))
        is_figure_cell = bool(re.search(r"#\|\s*label:\s*fig-", cell_text) or re.search(r"plt\.show\(|fig,\s*ax\s*=", cell_text))

        # Check echo suppression on all python cells
        has_echo_false = any("#| echo: false" in line for _, line in cell)
        if not has_echo_false and not is_figure_cell:
            violations.append(
                Violation(
                    file=str(path),
                    line=start_line,
                    code="lego_missing_echo_false",
                    message="Python block missing '#| echo: false' directive.",
                )
            )

        # Parse AST to find classes
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

        classes: list[ast.ClassDef] = [n for n in tree.body if isinstance(n, ast.ClassDef)]

        # If it's a figure or setup cell without classes, skip class checks
        if not classes and not has_lego_banner:
            continue

        # Rule 1: Cell defining multiple classes
        if len(classes) > 1:
            cls_names = ", ".join(f"'{c.name}'" for c in classes)
            first_cls_line = cell[classes[1].lineno - 1][0] if classes[1].lineno <= len(cell) else start_line
            violations.append(
                Violation(
                    file=str(path),
                    line=first_cls_line,
                    code="lego_multi_class",
                    message=(
                        f"Cell defines {len(classes)} classes ({cls_names}). "
                        "MLSysBook LEGO contract requires exactly one class per cell to preserve "
                        "the 1-cell = 1-concept mental model. Split into separate cells."
                    ),
                )
            )

        # Rule 2: Inverted cell architecture (class declared after Stage 1, 2, or 3)
        if classes:
            first_class = classes[0]
            # Check if any Stage 1, 2, or 3 comment appears before the class definition in the cell
            for lineno, line in cell[:first_class.lineno - 1]:
                m = STAGE_COMMENT.search(line)
                if m and int(m.group(1)) in (1, 2, 3):
                    stage_num = m.group(1)
                    stage_name = m.group(2)
                    violations.append(
                        Violation(
                            file=str(path),
                            line=cell[first_class.lineno - 1][0],
                            code="lego_inverted_cell",
                            message=(
                                f"Inverted cell architecture: 'class {first_class.name}' is declared after Stage {stage_num} ({stage_name}). "
                                "The class must open immediately after the box comment header to encapsulate all computational "
                                "stages inside class scope and prevent notebook namespace pollution."
                            ),
                        )
                    )
                    break

        # Rule 3: Banner marked LEGO but defines no class
        if has_lego_banner and not classes and not is_figure_cell:
            violations.append(
                Violation(
                    file=str(path),
                    line=start_line,
                    code="lego_banner_no_class",
                    message=(
                        "Cell carries '# ┌── LEGO' banner but defines no class. Wrap logic in a "
                        "dedicated LEGO class or replace with appropriate block header."
                    ),
                )
            )

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
    parser = argparse.ArgumentParser(description="Check LEGO cell architecture invariants.")
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

    print("All LEGO cell architecture checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
