#!/usr/bin/env python3
"""
validate_inline_refs.py
Pre-render guardrail for inline Python in QMD files.

Checks:
1. Every `{python} ref` resolves to a defined variable or ClassName.attr
2. Every `{python} ref` appears AFTER its definition (Locality)
3. No inline Python inside LaTeX math mode (causes decimal stripping)
4. No inline Python adjacent to LaTeX symbols like $\\times$
5. No grid tables with inline Python (use pipe tables instead)
6. (--check-lego) callout-notebook blocks without a preceding LEGO cell
7. (--check-lego) Hardcoded derived results in display math inside callout-notebooks
8. (--check-scope) Bare variable references in class bodies that need ClassName.attr
9. (--check-scope) Python 3 class-scope comprehension issues

Inline refs may be simple identifiers (`{python} var_name`) or dotted class
attribute access (`{python} ClassName.attr`).

Usage:
    python3 book/quarto/mlsysim/validate_inline_refs.py [--verbose] [--check-patterns] [--check-lego] [--check-scope]

Exit codes:
    0 = all checks pass
    1 = issues found
"""

import re
import sys
from pathlib import Path

DEPRECATION_MSG = (
    "DEPRECATION: use Binder instead of direct script invocation:\n"
    "  ./book/binder validate inline-refs [--path <file-or-dir>] [--check-patterns] [--check-lego]"
)

BOOK_ROOT = Path(__file__).resolve().parent.parent  # book/quarto/
CONTENTS = BOOK_ROOT / "contents"

# Pattern for inline Python references: `{python} var_name` or `{python} Name.attr`
INLINE_REF = re.compile(r'`\{python\}\s+(\w+(?:\.\w+)?)`')

# Pattern for Python compute cell blocks
CELL_START = re.compile(r'^```\{python\}')
CELL_END = re.compile(r'^```\s*$')

# Pattern for variable assignments in compute cells (handles tuple unpacking)
ASSIGNMENT = re.compile(r'^([a-zA-Z_]\w*(?:\s*,\s*[a-zA-Z_]\w*)*)\s*=')

# Pattern for class definitions in compute cells
CLASS_DEF = re.compile(r'^class\s+(\w+)\s*[:(]')

# Pattern for Exports: in header block
EXPORTS_SECTION = re.compile(r'#\s*.\s*[Ee]xports?:\s*(.*)')

# Problematic patterns that cause rendering issues
# Pattern 1: Inline Python directly inside LaTeX math: $`{python}`$ or $..`{python}`$
# Only matches when $ is immediately followed by backtick-python (within short distance)
# This avoids false positives from {python} appearing between two separate $...$ pairs
# EXCLUDES _str variables which are pre-formatted strings (no decimals to strip)
# Handles both simple `var_str` and dotted `Name.attr_str`
LATEX_INLINE_PYTHON = re.compile(r'(?<!\\)\$\s*`\{python\}\s+(?!\w+(?:\.\w+)?_str)[^`]+`|`\{python\}\s+(?!\w+(?:\.\w+)?_str)[^`]+`\s*(?<!\\)\$')

# Pattern 2: Inline Python adjacent to LaTeX symbols (decimal stripping risk)
# Only flags NON-_str variables. Using _str variables adjacent to $\times$ etc. is the
# PREFERRED convention — see book-prose.md "Multiplication and Times Notation".
# Handles both simple `var_str` and dotted `Name.attr_str`
LATEX_ADJACENT = re.compile(r'`\{python\}\s+(?!\w+(?:\.\w+)?_str)[^`]+`\s*\$\\(times|approx|ll|gg|mu)\$')

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


# ---------------------------------------------------------------------------
# LEGO compliance patterns (--check-lego)
# ---------------------------------------------------------------------------

CALLOUT_NOTEBOOK_START = re.compile(r'^:{2,}\s*\{\.callout-notebook')
CALLOUT_DIV_END = re.compile(r'^:{3,}\s*$')

# Hardcoded numeric result after = or \approx in LaTeX math.
# Matches patterns like: = 0.046, = 31.25, \approx 12.4, = 14,400
# Also handles: = \mathbf{6.9}, = \$398,131, = **0.30**
HARDCODED_RESULT_RE = re.compile(
    r'(?:=\s*|\\approx\s*)'
    r'(?:\\mathbf\{|\\textbf\{|\*\*)?'
    r'-?\d[\d,]*\.?\d*'
)

# A line with `{python}` reference — used to check if a callout uses computed values
# Matches both `{python} var` and `{python} Name.attr`
HAS_PYTHON_REF = re.compile(r'`\{python\}\s+\w+(?:\.\w+)?`')

# Suppression comment
LEGO_OK = re.compile(r'<!--\s*lego-ok\s*-->')

# Display math delimiters
DISPLAY_MATH_LINE = re.compile(r'^\$\$')

# Lines that contain numeric content worth flagging (digits with decimals or arithmetic)
HAS_NUMERIC_CONTENT = re.compile(r'\d+\.\d+|\\times|\\frac|\\approx')


def check_lego_compliance(qmd_path, verbose=False):
    """Check LEGO principle compliance in callout-notebook blocks.

    Returns list of (filepath, lineno, check_type, message) tuples.
    """
    text = qmd_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    warnings = []
    filepath = str(qmd_path.relative_to(BOOK_ROOT))

    # Pre-scan: record lines that are Python cell starts or ends
    python_cell_end_lines = set()
    in_cell = False
    cell_start_line = 0
    for i, line in enumerate(lines):
        if CELL_START.match(line):
            in_cell = True
            cell_start_line = i
        elif in_cell and CELL_END.match(line):
            in_cell = False
            python_cell_end_lines.add(i)  # 0-based

    # Main scan: find callout-notebook blocks
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for lego-ok suppression on this line or the next
        if LEGO_OK.search(line):
            i += 1
            continue

        if CALLOUT_NOTEBOOK_START.match(line):
            callout_start = i  # 0-based
            callout_title = line.strip()

            # Check suppression on the callout start line itself
            if LEGO_OK.search(line):
                # Skip to end of callout
                i += 1
                depth = 1
                while i < len(lines) and depth > 0:
                    if re.match(r'^:{3,}\s*\{', lines[i]):
                        depth += 1
                    elif CALLOUT_DIV_END.match(lines[i]):
                        depth -= 1
                    i += 1
                continue

            # --- Check 1: MISSING_LEGO_CELL ---
            # Look backwards up to 15 lines for a Python cell end (```)
            has_preceding_cell = False
            lookback = min(callout_start, 15)
            for j in range(callout_start - 1, callout_start - lookback - 1, -1):
                if j < 0:
                    break
                if j in python_cell_end_lines:
                    has_preceding_cell = True
                    break
                # Stop looking if we hit another callout or heading
                if re.match(r'^#{1,4}\s', lines[j]) or CALLOUT_NOTEBOOK_START.match(lines[j]):
                    break

            # --- Collect callout body ---
            callout_body_lines = []
            i += 1
            depth = 1
            while i < len(lines) and depth > 0:
                if LEGO_OK.search(lines[i]):
                    # Suppression inside callout: skip rest
                    while i < len(lines) and depth > 0:
                        if re.match(r'^:{3,}\s*\{', lines[i]):
                            depth += 1
                        elif CALLOUT_DIV_END.match(lines[i]):
                            depth -= 1
                        i += 1
                    break
                if re.match(r'^:{3,}\s*\{', lines[i]):
                    depth += 1
                elif CALLOUT_DIV_END.match(lines[i]):
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                callout_body_lines.append((i, lines[i]))
                i += 1

            # Check if callout has numeric content (display math, decimals, etc.)
            has_numerics = False
            has_python_refs = False
            for _, bline in callout_body_lines:
                if HAS_NUMERIC_CONTENT.search(bline):
                    has_numerics = True
                if HAS_PYTHON_REF.search(bline):
                    has_python_refs = True

            if not has_preceding_cell and has_numerics and not has_python_refs:
                warnings.append((filepath, callout_start + 1, "MISSING_LEGO_CELL",
                    f"callout-notebook has no preceding Python LEGO cell"))
                if verbose:
                    print(f"  ⚠ {qmd_path.name}:{callout_start + 1} — "
                          f"callout-notebook missing LEGO cell")

            # --- Check 2: HARDCODED_RESULT ---
            in_display_math = False
            display_math_start = 0
            display_math_buf = []

            for line_idx, bline in callout_body_lines:
                # Per-line suppression
                if LEGO_OK.search(bline):
                    continue

                # Track display math blocks ($$...$$)
                if DISPLAY_MATH_LINE.match(bline.strip()):
                    if not in_display_math:
                        in_display_math = True
                        display_math_start = line_idx
                        display_math_buf = [bline]
                    else:
                        # End of display math
                        display_math_buf.append(bline)
                        math_text = ' '.join(display_math_buf)
                        if (HARDCODED_RESULT_RE.search(math_text)
                                and not HAS_PYTHON_REF.search(math_text)):
                            warnings.append((filepath, display_math_start + 1,
                                "HARDCODED_RESULT",
                                "display math has hardcoded numeric result (no {python} ref)"))
                            if verbose:
                                snippet = math_text[:80].replace('\n', ' ')
                                print(f"  ⚠ {qmd_path.name}:{display_math_start + 1} — "
                                      f"hardcoded result: {snippet}…")
                        in_display_math = False
                        display_math_buf = []
                    continue

                if in_display_math:
                    display_math_buf.append(bline)
                    continue

                # Single-line display math: $$ ... $$
                stripped = bline.strip()
                if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
                    if (HARDCODED_RESULT_RE.search(stripped)
                            and not HAS_PYTHON_REF.search(stripped)):
                        warnings.append((filepath, line_idx + 1, "HARDCODED_RESULT",
                            "display math has hardcoded numeric result (no {python} ref)"))
                        if verbose:
                            print(f"  ⚠ {qmd_path.name}:{line_idx + 1} — hardcoded result")
                    continue

                # Inline math with results: $...= NUMBER...$
                # Only flag if line has = or \approx followed by a number, no {python}
                if ('$' in bline and not HAS_PYTHON_REF.search(bline)
                        and HARDCODED_RESULT_RE.search(bline)):
                    # Confirm it's inside $...$ math, not prose
                    dollar_count = bline.count('$') - bline.count('\\$')
                    if dollar_count >= 2:
                        warnings.append((filepath, line_idx + 1, "HARDCODED_RESULT",
                            "inline math has hardcoded numeric result (no {python} ref)"))
                        if verbose:
                            print(f"  ⚠ {qmd_path.name}:{line_idx + 1} — "
                                  f"hardcoded inline result")

            continue  # already advanced i in the callout body loop

        i += 1

    return warnings


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


import ast
import importlib

PYTHON_BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
PYTHON_BUILTINS |= {
    'int', 'float', 'str', 'list', 'dict', 'set', 'tuple', 'bool', 'bytes',
    'range', 'len', 'print', 'type', 'isinstance', 'enumerate', 'zip', 'map',
    'filter', 'sorted', 'reversed', 'min', 'max', 'sum', 'abs', 'round',
    'True', 'False', 'None', 'super', 'property', 'staticmethod', 'classmethod',
    'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
    'RuntimeError', 'AttributeError', 'NameError', 'ImportError',
    'NotImplementedError', 'StopIteration', 'AssertionError',
    'open', 'getattr', 'setattr', 'hasattr', 'delattr', 'callable',
    'any', 'all', 'id', 'hash', 'hex', 'oct', 'bin', 'chr', 'ord',
    'format', 'repr', 'input', 'breakpoint', 'object', 'complex',
    'frozenset', 'bytearray', 'memoryview', 'vars', 'dir', 'globals', 'locals',
    'exec', 'eval', 'compile', 'iter', 'next',
    'ZeroDivisionError', 'FileNotFoundError', 'OSError', 'IOError',
    'OverflowError', 'ArithmeticError', 'SystemError', 'Warning',
    'DeprecationWarning', 'UserWarning', 'FutureWarning',
}

_star_import_cache: dict = {}
_sys_path_patched = False


def _ensure_book_on_path():
    """Add book/quarto to sys.path so mlsysim.* imports resolve for star-import analysis."""
    global _sys_path_patched
    if _sys_path_patched:
        return
    book_quarto = str(BOOK_ROOT)
    if book_quarto not in sys.path:
        sys.path.insert(0, book_quarto)
    _sys_path_patched = True


def _resolve_star_import(module_name):
    """Try to resolve names exported by a `from X import *` statement.
    Returns a set of names, or empty set on failure."""
    if module_name in _star_import_cache:
        return _star_import_cache[module_name]
    _ensure_book_on_path()
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, '__all__'):
            names = set(mod.__all__)
        else:
            names = {n for n in dir(mod) if not n.startswith('_')}
        _star_import_cache[module_name] = names
        return names
    except Exception:
        _star_import_cache[module_name] = set()
        return set()


def _extract_cell_blocks(qmd_path):
    """Extract Python code cells from a QMD file.

    Returns list of (start_line_1based, source_code) tuples.
    """
    text = qmd_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    cells = []
    in_cell = False
    cell_start = 0
    cell_lines = []

    for i, line in enumerate(lines):
        if CELL_START.match(line):
            in_cell = True
            cell_start = i + 1
            cell_lines = []
        elif in_cell and CELL_END.match(line):
            in_cell = False
            cells.append((cell_start, '\n'.join(cell_lines)))
        elif in_cell:
            cell_lines.append(line)

    return cells


def _collect_all_assignments(node):
    """Recursively collect all names assigned anywhere under an AST node."""
    assigned = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        assigned.add(t.id)
        elif isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
            assigned.add(child.target.id)
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            assigned.add(child.target.id)
        elif isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
            assigned.add(child.target.id)
    return assigned


def _names_loaded_in_node(node):
    """Collect all Name nodes in Load context from an AST node,
    excluding names locally bound by function params/locals, lambda params,
    comprehension vars, for-loop vars, and exception handlers."""
    names = set()
    locally_bound = set()

    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in child.args.args + child.args.posonlyargs + child.args.kwonlyargs:
                locally_bound.add(arg.arg)
            if child.args.vararg:
                locally_bound.add(child.args.vararg.arg)
            if child.args.kwarg:
                locally_bound.add(child.args.kwarg.arg)
            locally_bound |= _collect_all_assignments(child)
        elif isinstance(child, ast.Lambda):
            for arg in child.args.args:
                locally_bound.add(arg.arg)
        elif isinstance(child, ast.comprehension):
            if isinstance(child.target, ast.Name):
                locally_bound.add(child.target.id)
            elif isinstance(child.target, ast.Tuple):
                for elt in child.target.elts:
                    if isinstance(elt, ast.Name):
                        locally_bound.add(elt.id)
        elif isinstance(child, ast.For):
            if isinstance(child.target, ast.Name):
                locally_bound.add(child.target.id)
            elif isinstance(child.target, ast.Tuple):
                for elt in child.target.elts:
                    if isinstance(elt, ast.Name):
                        locally_bound.add(elt.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            locally_bound.add(child.name)

    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id not in locally_bound:
                names.add(child.id)
        elif isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Load):
            if isinstance(child.value, ast.Name) and child.value.id not in locally_bound:
                names.add(child.value.id)
    return names


def _names_defined_in_class(class_node):
    """Collect names defined (assigned, imported, function-defined) in a class body.

    In Python, assignments inside for-loops, if-blocks, with-blocks, and try-blocks
    at the class level all define class-level attributes (they share the class scope).
    """
    defined = set()

    def _collect(stmts):
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    for t in ast.walk(target):
                        if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                            defined.add(t.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                defined.add(stmt.target.id)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(stmt.name)
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    defined.add(alias.asname if alias.asname else alias.name.split('.')[0])
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                defined.add(stmt.target.id)
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name):
                    defined.add(stmt.target.id)
                elif isinstance(stmt.target, ast.Tuple):
                    for elt in stmt.target.elts:
                        if isinstance(elt, ast.Name):
                            defined.add(elt.id)
                _collect(stmt.body)
                _collect(stmt.orelse)
            elif isinstance(stmt, ast.If):
                _collect(stmt.body)
                _collect(stmt.orelse)
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        defined.add(item.optional_vars.id)
                _collect(stmt.body)
            elif isinstance(stmt, ast.Try):
                _collect(stmt.body)
                for handler in stmt.handlers:
                    _collect(handler.body)
                _collect(stmt.orelse)
                _collect(stmt.finalbody)

    _collect(class_node.body)
    return defined


def _find_comprehension_scope_issues(class_node):
    """Detect class-scope variables referenced inside list/dict/set/generator comprehensions.

    In Python 3, comprehensions inside class bodies cannot access class-level names
    (except the iterable expression). This is a well-known scoping gotcha.
    Returns list of (lineno, name) tuples for problematic references.
    """
    issues = []
    class_local_names = _names_defined_in_class(class_node)

    for stmt in ast.walk(class_node):
        if isinstance(stmt, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            comp_iter_names = set()
            for gen in stmt.generators:
                if isinstance(gen.target, ast.Name):
                    comp_iter_names.add(gen.target.id)
                elif isinstance(gen.target, ast.Tuple):
                    for elt in gen.target.elts:
                        if isinstance(elt, ast.Name):
                            comp_iter_names.add(elt.id)

            if isinstance(stmt, ast.ListComp):
                inner_node = stmt.elt
            elif isinstance(stmt, ast.SetComp):
                inner_node = stmt.elt
            elif isinstance(stmt, ast.GeneratorExp):
                inner_node = stmt.elt
            elif isinstance(stmt, ast.DictComp):
                inner_node = stmt  # check key and value

            for child in ast.walk(inner_node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    name = child.id
                    if (name in class_local_names
                            and name not in comp_iter_names
                            and name not in PYTHON_BUILTINS):
                        if isinstance(child, ast.Name) and hasattr(child, 'lineno'):
                            issues.append((child.lineno, name))
    return issues


def check_scope(qmd_path, verbose=False):
    """Check for bare variable references inside class bodies that need ClassName. prefix.

    Detects two classes of bugs:
    1. BARE_CLASS_REF: A class body references a name that isn't locally defined,
       imported, or a builtin — likely needs a ClassName.attr prefix from a prior class.
    2. COMPREHENSION_SCOPE: A list comprehension inside a class body references a
       class-local function/variable, which fails in Python 3 due to implicit scope.

    Returns list of (filepath, lineno, check_type, message) tuples.
    """
    cells = _extract_cell_blocks(qmd_path)
    try:
        filepath = str(qmd_path.relative_to(BOOK_ROOT))
    except ValueError:
        filepath = str(qmd_path)
    warnings = []

    module_scope_names = set()
    known_classes = {}  # class_name -> set of attribute names

    for cell_start, source in cells:
        try:
            tree = ast.parse(source, filename=str(qmd_path))
        except SyntaxError:
            continue

        cell_imports = set()
        cell_top_level_names = set()
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                if isinstance(stmt, ast.ImportFrom) and any(
                    a.name == '*' for a in stmt.names
                ):
                    star_names = _resolve_star_import(stmt.module or '')
                    cell_imports |= star_names
                    cell_top_level_names |= star_names
                else:
                    for alias in stmt.names:
                        name = alias.asname if alias.asname else alias.name.split('.')[0]
                        cell_imports.add(name)
                        cell_top_level_names.add(name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        cell_top_level_names.add(target.id)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cell_top_level_names.add(stmt.name)

        for stmt in tree.body:
            if not isinstance(stmt, ast.ClassDef):
                continue

            class_name = stmt.name
            local_defs = _names_defined_in_class(stmt)
            known_classes[class_name] = local_defs

            all_loaded = set()
            for child_stmt in stmt.body:
                if isinstance(child_stmt, (ast.Expr,)) and isinstance(child_stmt.value, ast.Constant):
                    continue
                all_loaded |= _names_loaded_in_node(child_stmt)

            unresolved = (
                all_loaded
                - local_defs
                - cell_imports
                - cell_top_level_names
                - module_scope_names
                - PYTHON_BUILTINS
                - {class_name}
            )

            for name in sorted(unresolved):
                candidates = [
                    cn for cn, attrs in known_classes.items()
                    if name in attrs and cn != class_name
                ]
                qmd_line = cell_start + _find_name_line(source, name)
                if candidates:
                    suggestion = f"{candidates[0]}.{name}"
                    warnings.append((filepath, qmd_line, "BARE_CLASS_REF",
                        f"bare `{name}` in class `{class_name}` — did you mean `{suggestion}`?"))
                else:
                    warnings.append((filepath, qmd_line, "BARE_CLASS_REF",
                        f"bare `{name}` in class `{class_name}` — not defined locally or in prior classes"))
                if verbose:
                    if candidates:
                        print(f"  ⚠ {qmd_path.name}:{qmd_line} — bare `{name}` in `{class_name}` "
                              f"(try `{candidates[0]}.{name}`)")
                    else:
                        print(f"  ⚠ {qmd_path.name}:{qmd_line} — bare `{name}` in `{class_name}` (undefined)")

            comp_issues = _find_comprehension_scope_issues(stmt)
            for lineno, name in comp_issues:
                qmd_line = cell_start + lineno - 1
                warnings.append((filepath, qmd_line, "COMPREHENSION_SCOPE",
                    f"`{name}` in comprehension inside class `{class_name}` — "
                    f"Python 3 class-scope comprehensions cannot access class-level names"))
                if verbose:
                    print(f"  ⚠ {qmd_path.name}:{qmd_line} — `{name}` in comprehension "
                          f"(Python 3 class scope issue)")

        module_scope_names |= cell_top_level_names
        for cn in known_classes:
            module_scope_names.add(cn)

    return warnings


def _find_name_line(source, name):
    """Find approximate line number of a name reference in source code."""
    for i, line in enumerate(source.splitlines()):
        if re.search(rf'\b{re.escape(name)}\b', line):
            if not line.strip().startswith('#'):
                return i
    return 0


def validate_file(qmd_path, verbose=False, check_patterns=False, check_lego=False,
                  check_scope_flag=False):
    """Validate one QMD file. Returns (errors, warnings)."""
    text = qmd_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    errors = []
    defined_vars = set()
    defined_classes = set()
    in_cell = False
    in_exports = False

    for i, line in enumerate(lines, 1):
        # 1. Track variable and class definitions in cells
        if CELL_START.match(line):
            in_cell = True
            continue
        if in_cell and CELL_END.match(line):
            in_cell = False
            in_exports = False
            continue
        
        if in_cell:
            # Check for class definitions: class ClassName:
            cm = CLASS_DEF.match(line.strip())
            if cm:
                defined_classes.add(cm.group(1))

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
            ref = m.group(1)
            if '.' in ref:
                cls_name = ref.split('.', 1)[0]
                resolved = cls_name in defined_classes or cls_name in defined_vars
            else:
                resolved = ref in defined_vars
            if not resolved:
                errors.append((str(qmd_path.relative_to(BOOK_ROOT)), i, ref))
                if verbose:
                    print(f"  ✗ {qmd_path.name}:{i} — `{{python}} {ref}` used before definition (Locality Violation)")

    warnings = []
    if check_patterns:
        warnings = check_rendering_patterns(qmd_path, verbose)
    if check_lego:
        warnings.extend(check_lego_compliance(qmd_path, verbose))
    if check_scope_flag:
        warnings.extend(check_scope(qmd_path, verbose))
    
    return errors, warnings


def main():
    print(DEPRECATION_MSG, file=sys.stderr)

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    check_patterns = "--check-patterns" in sys.argv or "-p" in sys.argv
    check_lego = "--check-lego" in sys.argv or "-l" in sys.argv
    check_scope_flag = "--check-scope" in sys.argv or "-s" in sys.argv
    
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
        errors, warnings = validate_file(
            qmd, verbose=verbose,
            check_patterns=check_patterns, check_lego=check_lego,
            check_scope_flag=check_scope_flag,
        )
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
    if check_patterns or check_lego:
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
            for filepath, lineno, msg in items:
                print(f"    {filepath}:{lineno} — {msg}")
        exit_code = 1

    if exit_code == 0:
        print("\n✓ All checks passed!")
    else:
        print(f"\n{'─'*60}")
        print("Fix issues before rendering to ensure correct output.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
