#!/usr/bin/env python3
"""LEGO exported-variable liveness check.

This powers::

    ./book/binder check code --scope lego-dead-code

Binder owns the implementation; pre-commit reaches it through ``binder check``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CELL_START = re.compile(r"^```\{python\}")
CELL_END = re.compile(r"^```\s*$")
CLASS_DEF = re.compile(r"\bclass\s+([A-Za-z_]\w*)")
OUTPUT_HEADER = re.compile(r"#.*\b4\.\s*OUTPUT\b")
SECTION_HEADER = re.compile(r"^\s*#.*\b\d+\.\s+[A-Z]")
ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=")


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    code: str
    message: str
    context: str = ""


def _python_cells(lines: list[str]) -> Iterable[tuple[int, list[tuple[int, str]]]]:
    """Yield Python cell line spans as ``(start_line, [(lineno, text), ...])``."""
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
    """Return dead-code / structure violations for one QMD file."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[Violation] = []

    for start_line, cell in _python_cells(lines):
        cell_text = "\n".join(line for _, line in cell)
        if "LEGO" not in cell_text:
            continue

        class_match = CLASS_DEF.search(cell_text)
        if not class_match:
            continue
        class_name = class_match.group(1)

        output_idx = None
        for idx, (_lineno, line) in enumerate(cell):
            if OUTPUT_HEADER.search(line):
                output_idx = idx
                break

        if output_idx is None:
            out.append(
                Violation(
                    file=str(path),
                    line=start_line,
                    code="lego_output_missing",
                    message=(
                        f"LEGO block class `{class_name}` is missing a "
                        "`# ... 4. OUTPUT` section. Every exported prose "
                        "value should live in that section."
                    ),
                )
            )
            continue

        output_lines: list[tuple[int, str]] = []
        for idx in range(output_idx + 1, len(cell)):
            lineno, line = cell[idx]
            if SECTION_HEADER.search(line):
                break
            output_lines.append((lineno, line))

        output_text = "\n".join(line for _, line in output_lines)
        for lineno, line in output_lines:
            assign_match = ASSIGN.match(line)
            if not assign_match:
                continue

            var = assign_match.group(1)
            if var.startswith("_") or var == "RS":
                continue

            full_ref = f"{class_name}.{var}"
            if full_ref in text:
                continue

            alias_pattern = re.compile(r"\b[A-Za-z0-9_]+\." + re.escape(var) + r"\b")
            if alias_pattern.search(text):
                continue

            # If the output variable is only assigned once and never used to
            # construct another exported value, it is dead prose-facing output.
            if output_text.count(var) > 1:
                continue

            out.append(
                Violation(
                    file=str(path),
                    line=lineno,
                    code="lego_dead_code",
                    message=f"Exported LEGO variable `{full_ref}` is never used in prose.",
                    context=line.strip()[:160],
                )
            )

    return out


def qmd_files(paths: Iterable[Path]) -> list[Path]:
    """Expand files/directories into QMD files."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.qmd")))
        elif path.suffix == ".qmd" and path.exists():
            files.append(path)
    return sorted(dict.fromkeys(files))


def audit_paths(paths: Iterable[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for path in qmd_files(paths):
        try:
            violations.extend(audit_file(path))
        except (OSError, UnicodeDecodeError):
            continue
    return violations


def _default_paths() -> list[Path]:
    base = Path("book/quarto/contents")
    if base.exists():
        return [base]
    return [Path(".")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check for unused LEGO variables.")
    parser.add_argument("files", nargs="*", help="Files or directories to check")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.files] if args.files else _default_paths()
    violations = audit_paths(paths)

    if violations:
        by_file: dict[str, list[Violation]] = {}
        for violation in violations:
            by_file.setdefault(violation.file, []).append(violation)
        for file, file_violations in by_file.items():
            print(f"\n{file}")
            for violation in file_violations:
                loc = f":{violation.line}" if violation.line else ""
                print(f"  - {violation.code}{loc}: {violation.message}")
        print(
            "\nFix these by deleting unused exports from the 4. OUTPUT section "
            "or using them in prose."
        )
        return 1

    print("LEGO variable checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
