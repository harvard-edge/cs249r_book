#!/usr/bin/env python3
"""
validate_inline_refs.py
Pre-render guardrail for inline Python in QMD files.

Checks:
1. Every `{python} var_name` resolves to a defined variable
2. Every `{python} var_name` appears AFTER its definition (Locality)
3. No inline Python inside LaTeX math mode (causes decimal stripping)
4. No inline Python adjacent to LaTeX symbols like $\\times$
5. No grid tables with inline Python (use pipe tables instead)

Usage:
    python3 book/quarto/mlsys/validate_inline_refs.py [--verbose] [--check-patterns]

Exit codes:
    0 = all checks pass
    1 = issues found
"""

import re
import sys
from pathlib import Path

DEPRECATION_MSG = (
    "DEPRECATION: use Binder instead of direct script invocation:\n"
    "  ./book/binder validate inline-refs [--path <file-or-dir>] [--check-patterns]"
)

BOOK_ROOT = Path(__file__).resolve().parent.parent  # book/quarto/
CONTENTS = BOOK_ROOT / "contents"

# Pattern for inline Python references: `{python} var_name`
INLINE_REF = re.compile(r'`\{python\}\s+(\w+)`')

# Pattern for Python compute cell blocks
CELL_START = re.compile(r'^```\{python\}')
CELL_END = re.compile(r'^```\s*$')

# Pattern for variable assignments in compute cells (handles tuple unpacking)
ASSIGNMENT = re.compile(r'^([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)\s*=')

# Pattern for Exports: in header block
EXPORTS_SECTION = re.compile(r'#\s*.\s*[Ee]xports?:\s*(.*)')

# Problematic patterns that cause rendering issues
# Pattern 1: Inline Python directly inside LaTeX math: $`{python}`$ or $..`{python}`$
# Only matches when $ is immediately followed by backtick-python (within short distance)
# This avoids false positives from {python} appearing between two separate $...$ pairs
# EXCLUDES _str variables which are pre-formatted strings (no decimals to strip)
LATEX_INLINE_PYTHON = re.compile(r'(?<!\\)\$\s*`\{python\}\s+(?!\w+_str)[^`]+`|`\{python\}\s+(?!\w+_str)[^`]+`\s*(?<!\\)\$')

# Pattern 2: Inline Python adjacent to LaTeX symbols (decimal stripping risk)
# Only flags NON-_str variables. Using _str variables adjacent to $\times$ etc. is the
# PREFERRED convention — see book-prose.md "Multiplication and Times Notation".
LATEX_ADJACENT = re.compile(r'`\{python\}\s+(?!\w+_str)[^`]+`\s*\$\\(times|approx|ll|gg|mu)\$')

# Pattern 3: Grid table row separator (indicates grid table format)
GRID_TABLE_SEP = re.compile(r'^\+[-:=+]+\+$')

# Pattern 4: Inline f-string formatting (should be pre-computed as _str)
INLINE_FSTRING = re.compile(r'`\{python\}\s*f"[^`]+`')

# Pattern 5: Inline function calls (should be pre-computed as _str)
INLINE_FUNC_CALL = re.compile(r'`\{python\}\s*\w+\([^`]+\)`')

# Pattern 6: Inline Python in YAML chunk options (fig-cap, tbl-cap, fig-alt, lst-cap)
# These NEVER render — Quarto uses the option value as a literal string (verified by
# rendering _test_inline_captions.qmd: body and ": Caption {#tbl-...}" run inline Python;
# #| fig-alt and #| fig-cap do not).
YAML_OPTION_INLINE = re.compile(r'^#\|\s*(fig-cap|tbl-cap|lst-cap|fig-alt):\s*.*`\{python\}')


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
        
        # Check for non-_str inline Python adjacent to LaTeX symbols (decimal stripping risk)
        # NOTE: _str variables adjacent to $\times$ is the PREFERRED convention.
        if LATEX_ADJACENT.search(line):
            warnings.append((filepath, i, "LATEX_ADJACENT",
                "Non-_str inline Python adjacent to $\\\\times$ (decimal stripping risk)"))
            if verbose:
                print(f"  ⚠ {qmd_path.name}:{i} — Non-_str Python adjacent to LaTeX symbol")
        
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

        # Check for inline Python in YAML chunk options (fig-cap, fig-alt, tbl-cap, lst-cap) — NEVER renders
        if YAML_OPTION_INLINE.search(line):
            warnings.append((filepath, i, "YAML_OPTION",
                "Inline Python in #| fig-alt/fig-cap/tbl-cap/lst-cap - NEVER renders! Use hardcoded value or set caption in code."))
            if verbose:
                print(f"  ✗ {qmd_path.name}:{i} — Python in YAML option (will appear literally)")

    return warnings


def validate_file(qmd_path, verbose=False, check_patterns=False):
    """Validate one QMD file. Returns (errors, warnings)."""
    text = qmd_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    errors = []
    defined_vars = set()
    in_cell = False
    in_exports = False

    for i, line in enumerate(lines, 1):
        # 1. Track variable definitions in cells
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            in_exports = False
            continue
        
        if in_cell:
            # Check for assignments: var = ... or var1, var2 = ...
            m = ASSIGNMENT.match(line.strip())
            if m:
                vars_part = m.group(1)
                for v in re.split(r'[,\s]+', vars_part):
                    if v.strip():
                        defined_vars.add(v.strip())
            
            # Check for Exports: in header
            m = EXPORTS_SECTION.match(line.strip())
            if m:
                in_exports = True
                vars_raw = m.group(1)
                # Remove unit parentheticals like (MB, GB)
                vars_raw = re.sub(r'\(.*?\)', '', vars_raw)
                for v in re.split(r'[,\s]+', vars_raw):
                    v = v.strip().rstrip(',')
                    if v:
                        defined_vars.add(v)
            elif in_exports:
                # Continuation of exports
                m = re.match(r'#\s*.\s*(.*)', line.strip())
                if m:
                    content = m.group(1).strip()
                    # If content starts with a section like 'Goal:', stop
                    if re.match(r'^[A-Z][a-z]+:', content):
                        in_exports = False
                    elif content == "" or "──" in content:
                        in_exports = False
                    else:
                        vars_raw = re.sub(r'\(.*?\)', '', content)
                        for v in re.split(r'[,\s]+', vars_raw):
                            v = v.strip().rstrip(',')
                            if v:
                                defined_vars.add(v)
                else:
                    in_exports = False
            continue # Don't check for refs inside compute cells

        # 2. Check inline references for Locality
        for m in INLINE_REF.finditer(line):
            var = m.group(1)
            if var not in defined_vars:
                errors.append((str(qmd_path.relative_to(BOOK_ROOT)), i, var))
                if verbose:
                    print(f"  ✗ {qmd_path.name}:{i} — `{{python}} {var}` used before definition (Locality Violation)")

    warnings = []
    if check_patterns:
        warnings = check_rendering_patterns(qmd_path, verbose)
    
    return errors, warnings


def main():
    print(DEPRECATION_MSG, file=sys.stderr)

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    check_patterns = "--check-patterns" in sys.argv or "-p" in sys.argv
    
    # Check for path argument
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        target_path = Path(args[0]).resolve()
        if target_path.is_file():
            qmd_files = [target_path]
        else:
            qmd_files = sorted(target_path.rglob("*.qmd"))
    else:
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
            print(f"  {filepath}:{lineno} — `{{python}} {var}` undefined/locality violation")
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
