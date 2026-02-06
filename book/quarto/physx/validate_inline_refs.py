#!/usr/bin/env python3
"""
validate_inline_refs.py
Pre-render guardrail for inline Python in QMD files.

Checks:
1. Every `{python} var_name` resolves to a defined variable
2. No inline Python inside LaTeX math mode (causes decimal stripping)
3. No inline Python adjacent to LaTeX symbols like $\\times$
4. No grid tables with inline Python (use pipe tables instead)

Usage:
    python3 book/quarto/physx/validate_inline_refs.py [--verbose] [--check-patterns]

Exit codes:
    0 = all checks pass
    1 = issues found
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

# Problematic patterns that cause rendering issues
# Pattern 1: Inline Python directly inside LaTeX math: $`{python}`$ or $..`{python}`$
# Only matches when $ is immediately followed by backtick-python (within short distance)
# This avoids false positives from {python} appearing between two separate $...$ pairs
# EXCLUDES _str variables which are pre-formatted strings (no decimals to strip)
LATEX_INLINE_PYTHON = re.compile(r'(?<!\\)\$\s*`\{python\}\s+(?!\w+_str)[^`]+`|`\{python\}\s+(?!\w+_str)[^`]+`\s*(?<!\\)\$')

# Pattern 2: Inline Python adjacent to LaTeX symbols (will strip decimals)
# EXCLUDES _str variables which are pre-formatted strings (safe to use adjacent to LaTeX)
LATEX_ADJACENT = re.compile(r'`\{python\}\s+(?!\w+_str)[^`]+`\s*\$\\(times|approx|ll|gg|mu)\$')

# Pattern 3: Grid table row separator (indicates grid table format)
GRID_TABLE_SEP = re.compile(r'^\+[-:=+]+\+$')

# Pattern 4: Inline f-string formatting (should be pre-computed as _str)
INLINE_FSTRING = re.compile(r'`\{python\}\s*f"[^`]+`')

# Pattern 5: Inline function calls (should be pre-computed as _str)
INLINE_FUNC_CALL = re.compile(r'`\{python\}\s*\w+\([^`]+\)`')


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


def check_rendering_patterns(qmd_path, verbose=False):
    """Check for patterns that cause rendering issues. Returns list of warnings."""
    text = qmd_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    warnings = []
    filepath = str(qmd_path.relative_to(BOOK_ROOT))
    
    # Track if we're in a grid table
    in_grid_table = False
    grid_table_start = 0
    
    for i, line in enumerate(lines, 1):
        # Check for inline Python inside LaTeX math
        if LATEX_INLINE_PYTHON.search(line):
            warnings.append((filepath, i, "LATEX_MATH", 
                "Inline Python inside $...$ - will strip decimal points"))
            if verbose:
                print(f"  ⚠ {qmd_path.name}:{i} — Python inside LaTeX math")
        
        # Check for inline Python adjacent to LaTeX symbols
        if LATEX_ADJACENT.search(line):
            warnings.append((filepath, i, "LATEX_ADJACENT",
                "Inline Python adjacent to $\\\\times$ etc - use Unicode × instead"))
            if verbose:
                print(f"  ⚠ {qmd_path.name}:{i} — Python adjacent to LaTeX symbol")
        
        # Track grid tables
        if GRID_TABLE_SEP.match(line.strip()):
            if not in_grid_table:
                in_grid_table = True
                grid_table_start = i
        elif in_grid_table and not line.strip().startswith('|') and line.strip():
            in_grid_table = False
        
        # Check for inline Python in grid tables
        if in_grid_table and '`{python}' in line:
            warnings.append((filepath, grid_table_start, "GRID_TABLE",
                "Grid table with inline Python - convert to pipe table"))
            if verbose:
                print(f"  ⚠ {qmd_path.name}:{i} — Python in grid table")
        
        # Check for inline f-string formatting (should be pre-computed)
        if INLINE_FSTRING.search(line):
            warnings.append((filepath, i, "INLINE_FSTRING",
                "Inline f-string - pre-compute as _str variable in Python block"))
            if verbose:
                print(f"  ⚠ {qmd_path.name}:{i} — Inline f-string")
        
        # Check for inline function calls (should be pre-computed)
        if INLINE_FUNC_CALL.search(line):
            warnings.append((filepath, i, "INLINE_FUNC",
                "Inline function call - pre-compute as _str variable in Python block"))
            if verbose:
                print(f"  ⚠ {qmd_path.name}:{i} — Inline function call")
    
    return warnings


def validate_file(qmd_path, verbose=False, check_patterns=False):
    """Validate one QMD file. Returns (errors, warnings)."""
    text = qmd_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    inline_refs = extract_inline_refs(lines)
    if not inline_refs:
        return [], []  # No inline refs, nothing to validate

    compute_vars = extract_compute_vars(lines)
    errors = []
    for lineno, var in inline_refs:
        if var not in compute_vars:
            errors.append((str(qmd_path.relative_to(BOOK_ROOT)), lineno, var))
            if verbose:
                print(f"  ✗ {qmd_path.name}:{lineno} — `{{python}} {var}` not defined")

    warnings = []
    if check_patterns:
        warnings = check_rendering_patterns(qmd_path, verbose)
    
    return errors, warnings


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    check_patterns = "--check-patterns" in sys.argv or "-p" in sys.argv

    qmd_files = sorted(CONTENTS.rglob("*.qmd"))
    total_files = 0
    total_refs = 0
    all_errors = []
    all_warnings = []

    for qmd in qmd_files:
        errors, warnings = validate_file(qmd, verbose=verbose, check_patterns=check_patterns)
        text = qmd.read_text(encoding="utf-8")
        refs = INLINE_REF.findall(text)
        if refs:
            total_files += 1
            total_refs += len(refs)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    print(f"\n{'='*60}")
    print(f"Inline Python Validation Report")
    print(f"{'='*60}")
    print(f"Files with inline refs:  {total_files}")
    print(f"Total inline references: {total_refs}")
    print(f"Unresolved references:   {len(all_errors)}")
    if check_patterns:
        print(f"Rendering warnings:      {len(all_warnings)}")

    exit_code = 0

    if all_errors:
        print(f"\n{'─'*60}")
        print("ERRORS (will break render):")
        for filepath, lineno, var in all_errors:
            print(f"  {filepath}:{lineno} — `{{python}} {var}` undefined")
        exit_code = 1

    if all_warnings:
        print(f"\n{'─'*60}")
        print("WARNINGS (may cause incorrect rendering):")
        # Group by type
        by_type = {}
        for filepath, lineno, wtype, msg in all_warnings:
            by_type.setdefault(wtype, []).append((filepath, lineno, msg))
        
        for wtype, items in sorted(by_type.items()):
            print(f"\n  [{wtype}] ({len(items)} issues)")
            for filepath, lineno, msg in items[:5]:  # Show first 5
                print(f"    {filepath}:{lineno} — {msg}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
        exit_code = 1

    if exit_code == 0:
        print("\n✓ All checks passed!")
    else:
        print(f"\n{'─'*60}")
        print("Fix issues before rendering to ensure correct output.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
