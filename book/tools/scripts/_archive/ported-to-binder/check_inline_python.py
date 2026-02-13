#!/usr/bin/env python3
r"""
Check inline Python references in QMD files.

This book uses Quarto's `{python} var` syntax to embed computed values in prose.
Malformed references render raw {python} text in the final PDF — a critical bug
that is invisible until build time.

Checks:
  1. Missing opening backtick:  {python} var`  (no backtick before {python})
  2. Dollar-as-backtick:        ${python} var` ($ used where ` should be)
  3. Python inside $...$ math:  $..{python}..$  (needs md_math() instead)
  4. Python inside $$...$$ math: display math with inline Python
  5. Python in grid tables:     grid tables don't support inline Python

Usage:
  python check_inline_python.py quarto/contents/vol1/
  python check_inline_python.py quarto/contents/vol1/training/training.qmd
  python check_inline_python.py --errors-only quarto/contents/

Exit codes:
  0 - No issues found
  1 - Issues found
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass

DEPRECATION_MSG = (
    "DEPRECATION: use Binder instead of direct script invocation:\n"
    "  ./book/binder validate inline-python [--path <file-or-dir>]"
)


@dataclass
class Issue:
    """A single inline Python problem."""
    line_num: int
    check: str
    severity: str  # 'error' or 'warning'
    message: str
    context: str


# ---------------------------------------------------------------------------
# Checks — ordered by severity / likelihood
# ---------------------------------------------------------------------------
CHECKS = {
    # --- ERRORS (will produce broken output) ---

    'missing_backtick': {
        'regex': re.compile(r'(?<!`)(\{python\}\s+\w+`)'),
        'severity': 'error',
        'message': 'Missing opening backtick — will render raw {python} text in PDF',
    },
    'dollar_as_backtick': {
        'regex': re.compile(r'\$\{python\}\s+\w+`'),
        'severity': 'error',
        'message': 'Dollar sign used instead of backtick before {python}',
    },
    # python_in_math is handled by _check_python_in_math() below, not a regex
    'python_in_display_math': {
        'regex': re.compile(r'\$\$[^$]*`?\{python\}'),
        'severity': 'error',
        'message': 'Inline Python inside $$...$$ display math — use md_math() instead',
    },

    # --- WARNINGS (may cause PDF issues) ---

    'latex_after_python': {
        'regex': re.compile(
            r'`\{python\}[^`]+`\s*\$\\(times|approx|ll|gg|mu|le|ge|neq|pm|cdot|div)'
        ),
        'severity': 'warning',
        'message': 'LaTeX symbol immediately after inline Python — may break in PDF',
    },
    'latex_before_python': {
        'regex': re.compile(
            r'\$\\(times|approx|ll|gg|mu|le|ge|neq|pm|cdot|div)\$\s*`\{python\}'
        ),
        'severity': 'warning',
        'message': 'LaTeX symbol immediately before inline Python — may break in PDF',
    },
}


def _check_python_in_math(line: str) -> list[str]:
    """Find {python} references inside properly-paired $...$ math blocks.

    Parses the line character-by-character to correctly pair opening and
    closing $ delimiters, avoiding false positives when inline Python sits
    *between* two separate math expressions on the same line.
    """
    hits: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == '$' and (i == 0 or line[i - 1] != '\\'):
            # Skip $$ display-math delimiters
            if i + 1 < n and line[i + 1] == '$':
                i += 2
                continue
            # Skip currency pattern $` (dollar immediately before backtick)
            if i + 1 < n and line[i + 1] == '`':
                i += 2
                continue
            # We have an opening $  — find its matching closing $
            start = i
            j = i + 1
            while j < n:
                if line[j] == '$' and (j == 0 or line[j - 1] != '\\'):
                    # Skip $$ inside the span (shouldn't happen, but be safe)
                    if j + 1 < n and line[j + 1] == '$':
                        j += 2
                        continue
                    # Found the closing $
                    content = line[start + 1 : j]
                    if '{python}' in content:
                        ctx = line[max(0, start) : min(n, j + 1)]
                        if len(ctx) > 60:
                            ctx = ctx[:57] + '...'
                        hits.append(ctx)
                    i = j + 1
                    break
                j += 1
            else:
                i += 1  # unclosed $, move on
        else:
            i += 1
    return hits


def check_file(filepath: Path) -> list[Issue]:
    """Check a single QMD file for inline Python issues."""
    content = filepath.read_text()
    lines = content.split('\n')
    issues: list[Issue] = []

    in_code_block = False

    for i, line in enumerate(lines):
        # Skip fenced code blocks (```python, ```{.python}, ```.tikz, etc.)
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Regex-based checks
        for check_name, check_info in CHECKS.items():
            for match in check_info['regex'].finditer(line):
                ctx = match.group(0)
                if len(ctx) > 60:
                    ctx = ctx[:57] + '...'
                issues.append(Issue(
                    line_num=i + 1,
                    check=check_name,
                    severity=check_info['severity'],
                    message=check_info['message'],
                    context=ctx,
                ))

        # Parser-based check: {python} inside $...$ math
        for ctx in _check_python_in_math(line):
            issues.append(Issue(
                line_num=i + 1,
                check='python_in_math',
                severity='error',
                message='Inline Python inside $...$ math — use md_math() instead',
                context=ctx,
            ))

    # Grid tables with inline Python (needs multiline context)
    in_grid = False
    for i, line in enumerate(lines):
        if re.match(r'^\+[-:=+]+\+$', line):
            in_grid = True
        elif in_grid and not line.startswith('|') and not re.match(r'^\+[-:=+]+\+$', line):
            in_grid = False
        if in_grid and '`{python}' in line:
            issues.append(Issue(
                line_num=i + 1,
                check='grid_table_python',
                severity='error',
                message='Inline Python in grid table — convert to pipe table',
                context=line.strip()[:60],
            ))

    return issues


def main():
    import argparse

    print(DEPRECATION_MSG, file=sys.stderr)

    parser = argparse.ArgumentParser(
        description='Validate inline Python references in QMD files',
    )
    parser.add_argument('paths', nargs='*', help='Files or directories to check')
    parser.add_argument('--errors-only', '-e', action='store_true',
                        help='Only report errors (skip warnings)')
    parser.add_argument('--summary', '-s', action='store_true',
                        help='Print counts only')

    args = parser.parse_args()
    if not args.paths:
        parser.print_help()
        return 0

    all_issues: list[tuple[Path, Issue]] = []

    for path_str in args.paths:
        path = Path(path_str)
        files = [path] if path.is_file() else sorted(path.rglob('*.qmd'))
        for f in files:
            if f.suffix != '.qmd':
                continue
            for issue in check_file(f):
                all_issues.append((f, issue))

    if args.errors_only:
        all_issues = [(p, i) for p, i in all_issues if i.severity == 'error']

    # --- Output ---
    files_hit = sorted(set(p for p, _ in all_issues))
    errors = sum(1 for _, i in all_issues if i.severity == 'error')
    warnings = sum(1 for _, i in all_issues if i.severity == 'warning')

    if args.summary:
        print(f"Files with issues: {len(files_hit)}")
        print(f"Errors: {errors}  Warnings: {warnings}")
    else:
        cur = None
        for filepath, issue in all_issues:
            if filepath != cur:
                if cur is not None:
                    print()
                print(f"{filepath}:")
                cur = filepath
            icon = '❌' if issue.severity == 'error' else '⚠️'
            print(f"  {icon} L{issue.line_num}: {issue.message}")
            print(f"     → {issue.context}")

    if all_issues:
        print(f"\n{'❌' if errors else '⚠️'} {errors} error(s), {warnings} warning(s) in {len(files_hit)} file(s).")
        return 1

    print("✓ All inline Python references look correct.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
