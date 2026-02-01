#!/usr/bin/env python3
"""
validate_inline_refs.py
Pre-render guardrail: ensures every `{python} var_name` inline reference
in QMD files resolves to a variable defined in that chapter's compute cell.

Usage:
    python3 book/quarto/calc/validate_inline_refs.py [--verbose]

Exit codes:
    0 = all references resolve
    1 = unresolved references found
"""

import re
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parent.parent  # book/quarto/
CONTENTS = BOOK_ROOT / "contents"

# Pattern for inline Python references: `{python} var_name`
INLINE_REF = re.compile(r'`\{python\}\s+(\w+)`')

# Pattern for Python compute cell blocks
CELL_START = re.compile(r'^```\{python\}')
CELL_END = re.compile(r'^```\s*$')

# Pattern for variable assignments in compute cells
ASSIGNMENT = re.compile(r'^(\w+)\s*=')


def extract_compute_vars(lines):
    """Extract all variable names assigned in ```{python} compute cells."""
    variables = set()
    in_cell = False
    for line in lines:
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            continue
        if in_cell:
            m = ASSIGNMENT.match(line.strip())
            if m:
                variables.add(m.group(1))
    return variables


def extract_inline_refs(lines):
    """Extract all inline `{python} var` references with line numbers."""
    refs = []
    for i, line in enumerate(lines, 1):
        for m in INLINE_REF.finditer(line):
            refs.append((i, m.group(1)))
    return refs


def validate_file(qmd_path, verbose=False):
    """Validate one QMD file. Returns list of (file, line, var) errors."""
    text = qmd_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    inline_refs = extract_inline_refs(lines)
    if not inline_refs:
        return []  # No inline refs, nothing to validate

    compute_vars = extract_compute_vars(lines)
    errors = []
    for lineno, var in inline_refs:
        if var not in compute_vars:
            errors.append((str(qmd_path.relative_to(BOOK_ROOT)), lineno, var))
            if verbose:
                print(f"  ✗ {qmd_path.name}:{lineno} — `{{python}} {var}` not defined in compute cell")

    return errors


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    qmd_files = sorted(CONTENTS.rglob("*.qmd"))
    total_files = 0
    total_refs = 0
    all_errors = []

    for qmd in qmd_files:
        errors = validate_file(qmd, verbose=verbose)
        text = qmd.read_text(encoding="utf-8")
        refs = INLINE_REF.findall(text)
        if refs:
            total_files += 1
            total_refs += len(refs)
        all_errors.extend(errors)

    print(f"\n{'='*60}")
    print(f"Inline Reference Validation Report")
    print(f"{'='*60}")
    print(f"Files with inline refs:  {total_files}")
    print(f"Total inline references: {total_refs}")
    print(f"Unresolved references:   {len(all_errors)}")

    if all_errors:
        print(f"\n{'─'*60}")
        print("ERRORS:")
        for filepath, lineno, var in all_errors:
            print(f"  {filepath}:{lineno} — `{{python}} {var}` undefined")
        print(f"{'─'*60}")
        print("FAIL: Fix unresolved references before rendering.")
        return 1
    else:
        print("\n✓ All inline references resolve to compute cell variables.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
