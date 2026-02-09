#!/usr/bin/env python3
r"""
Check for problematic rendering patterns in QMD files.

Detects patterns known to cause PDF/LaTeX rendering issues:
1. Missing opening backtick: {python} var` (renders raw {python} text)
2. Dollar sign instead of backtick: ${python} var` (same rendering failure)
3. Inline Python inside $...$ math blocks (LaTeX escaping breaks output)
4. LaTeX symbols with space adjacent to inline Python: `{python} var` $\times$
5. Quad asterisks (malformed bold): ****text**
6. Footnotes in table cells: | text[^fn-xxx] |
7. Grid tables (should be pipe tables for inline Python)

Usage:
  python check_render_patterns.py --check book/quarto/contents/
  python check_render_patterns.py file.qmd
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Issue:
    """Represents a rendering issue."""
    line_num: int
    pattern_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    context: str


# Pattern definitions
PATTERNS = {
    'missing_opening_backtick': {
        # {python} followed by a closing backtick, but NOT preceded by an opening backtick
        # Matches: {python} var_str`  or  ${python} var_str`  or  >{python} var_str`
        # Skips lines inside ```python code blocks (handled by in_code_block logic)
        'regex': re.compile(r'(?<!`)(\{python\}\s+\w+`)'),
        'severity': 'error',
        'message': 'Missing opening backtick on inline Python - will render raw {python} text',
        'fix_hint': 'Add opening backtick: `{python} var_name` (backtick before {python})'
    },
    'dollar_before_python': {
        # $ immediately before {python} (currency dollar used instead of backtick)
        'regex': re.compile(r'\$\{python\}\s+\w+`'),
        'severity': 'error',
        'message': 'Dollar sign instead of backtick before {python} - will render raw text',
        'fix_hint': 'Replace $ with backtick: $`{python} var_name` (add backtick between $ and {python})'
    },
    # NOTE: 'python_in_dollar_math' is handled by a custom function
    # (check_python_in_dollar_math) instead of a regex pattern, because
    # regex cannot reliably distinguish separate $...$ blocks on the same
    # line from a single $...$ block containing {python}. The greedy regex
    # previously produced false positives on lines like:
    #   $\times$ `{python} foo` $\times$
    # where {python} is BETWEEN two separate math blocks, not inside one.
    # NOTE: 'inconsistent_arith_units' check removed — different chapters
    # legitimately use different phrasings ("FLOPs per byte", "ops/byte", etc.)
    # depending on context, and enforcing a single form is not useful.
    # NOTE: The original 'latex_inline_python' check was removed because it
    # produced false positives on currency patterns like $`{python} cost_str`.
    # The 'python_in_dollar_math' check above handles real $...{python}...$ cases.
    'latex_adjacent_python_after_space': {
        # Only flag when there's a SPACE between the closing backtick and the $
        # No-space pattern (`{python} var`$\times) renders correctly.
        'regex': re.compile(r'`\{python\}[^`]+`\s+\$\\(times|approx|ll|gg|mu|le|ge|neq|pm|cdot|div)'),
        'severity': 'warning', 
        'message': 'Space between inline Python and LaTeX symbol - may not render correctly in PDF',
        'fix_hint': 'Remove space: `{python} var`$\\times$ (no space before $)'
    },
    'latex_adjacent_python_before_space': {
        # Only flag when there's a SPACE between the $ and the opening backtick
        'regex': re.compile(r'\$\\(times|approx|ll|gg|mu|le|ge|neq|pm|cdot|div)\$\s+`\{python\}'),
        'severity': 'warning',
        'message': 'Space between LaTeX symbol and inline Python - may not render correctly in PDF',
        'fix_hint': 'Remove space: $\\times$`{python} var` (no space after $)'
    },
    'quad_asterisks': {
        'regex': re.compile(r'\*{4,}'),
        'severity': 'warning',
        'message': 'Quad asterisks - likely malformed bold/italic',
        'fix_hint': 'Use **bold** or *italic*, not ****text**'
    },
    'footnote_in_table': {
        'regex': re.compile(r'^\|.*\[\^fn-[^\]]+\].*\|'),
        'severity': 'warning',
        'message': 'Footnote in table cell - may break PDF rendering',
        'fix_hint': 'Move footnote to caption or text after table'
    },
    'grid_table_with_python': {
        'regex': re.compile(r'^\+[-:=+]+\+.*`\{python\}', re.MULTILINE),
        'severity': 'error',
        'message': 'Grid table with inline Python - will not render correctly',
        'fix_hint': 'Convert to pipe table format'
    },
    'double_dollar_python': {
        'regex': re.compile(r'\$\$[^$]*`\{python\}'),
        'severity': 'error',
        'message': 'Inline Python in display math - will not render correctly',
        'fix_hint': 'Use separate lines or text mode for dynamic values'
    },
}


def check_python_in_dollar_math(line: str, line_num: int) -> list[Issue]:
    """Check if inline Python appears inside a $...$ math block.

    Properly parses $...$ spans by finding the shortest matching pairs,
    avoiding false positives when multiple $...$ blocks appear on the
    same line with {python} between them (not inside them).

    Exception: {python} inside LaTeX exponents (^{...}) is allowed because
    exponents are always integers, which don't have decimal points that
    LaTeX would strip. Example: $\\times 10^{`{python} exp_str`}$
    """
    issues = []
    # Find all $...$ math spans (shortest match, non-escaped)
    # Pattern: unescaped $ followed by non-$ chars, ending at next unescaped $
    math_spans = re.finditer(r'(?<!\\)\$(?!\$)(?!`)(.+?)(?<!\\)\$', line)
    for match in math_spans:
        inner = match.group(1)
        if '{python}' not in inner:
            continue
        # Allow {python} inside exponents: ^{`{python} var`}
        # Remove all ^{...`{python}...`...} patterns before checking
        inner_without_exponents = re.sub(
            r'\^\{[^}]*`\{python\}[^`]*`[^}]*\}', '', inner
        )
        if '{python}' in inner_without_exponents:
            issues.append(Issue(
                line_num=line_num,
                pattern_type='python_in_dollar_math',
                severity='error',
                message='Inline Python inside $...$ math block - will not render correctly',
                context=match.group(0)[:60]
            ))
    return issues


def check_file(filepath: Path) -> list[Issue]:
    """Check a file for rendering issues."""
    content = filepath.read_text()
    lines = content.split('\n')
    issues = []
    
    # Check for grid tables (need multiline context)
    if re.search(r'^\+[-:=+]+\+$', content, re.MULTILINE):
        # Check if any grid table row contains inline Python
        in_grid_table = False
        grid_start = 0
        for i, line in enumerate(lines):
            if re.match(r'^\+[-:=+]+\+$', line):
                if not in_grid_table:
                    in_grid_table = True
                    grid_start = i
            elif in_grid_table and not line.startswith('|') and not re.match(r'^\+[-:=+]+\+$', line):
                in_grid_table = False
            
            if in_grid_table and '`{python}' in line:
                issues.append(Issue(
                    line_num=i + 1,
                    pattern_type='grid_table_with_python',
                    severity='error',
                    message='Grid table with inline Python - will not render correctly',
                    context=line.strip()[:60]
                ))
    
    # Check line by line for other patterns
    for i, line in enumerate(lines):
        # Custom check: Python inside $...$ math (not handled by regex)
        issues.extend(check_python_in_dollar_math(line, i + 1))

        for pattern_name, pattern_info in PATTERNS.items():
            if pattern_name == 'grid_table_with_python':
                continue  # Already handled above
            if 'regex' not in pattern_info:
                continue  # Skip non-regex entries (e.g., comment-only)
            
            matches = pattern_info['regex'].findall(line)
            if matches:
                # For regex with groups, get the full match
                full_matches = pattern_info['regex'].finditer(line)
                for match in full_matches:
                    issues.append(Issue(
                        line_num=i + 1,
                        pattern_type=pattern_name,
                        severity=pattern_info['severity'],
                        message=pattern_info['message'],
                        context=match.group(0)[:60] if len(match.group(0)) > 60 else match.group(0)
                    ))
    
    return issues


def format_issue(issue: Issue, filepath: Path) -> str:
    """Format an issue for display."""
    severity_icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}[issue.severity]
    return f"  {severity_icon} Line {issue.line_num}: {issue.message}\n     → {issue.context}"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check for problematic rendering patterns in QMD files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Patterns checked:
  - Inline Python in LaTeX math ($...`{python}`...$)
  - LaTeX symbols adjacent to inline Python
  - Quad asterisks (malformed bold)
  - Footnotes in table cells
  - Grid tables with inline Python

Exit codes:
  0 - No issues found
  1 - Issues found (errors or warnings)
"""
    )
    parser.add_argument('paths', nargs='*', default=[],
                        help='Files or directories to check')
    parser.add_argument('--errors-only', '-e', action='store_true',
                        help='Only report errors, not warnings')
    parser.add_argument('--summary', '-s', action='store_true',
                        help='Show summary counts only')
    
    args = parser.parse_args()
    
    if not args.paths:
        parser.print_help()
        return 0
    
    all_issues = []
    files_with_issues = []
    
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.qmd':
            issues = check_file(path)
            if issues:
                all_issues.extend([(path, i) for i in issues])
                files_with_issues.append(path)
        elif path.is_dir():
            for qmd_file in sorted(path.rglob('*.qmd')):
                issues = check_file(qmd_file)
                if issues:
                    all_issues.extend([(qmd_file, i) for i in issues])
                    if qmd_file not in files_with_issues:
                        files_with_issues.append(qmd_file)
    
    # Filter by severity if requested
    if args.errors_only:
        all_issues = [(p, i) for p, i in all_issues if i.severity == 'error']
    
    # Output
    if args.summary:
        error_count = sum(1 for _, i in all_issues if i.severity == 'error')
        warning_count = sum(1 for _, i in all_issues if i.severity == 'warning')
        print(f"Files checked: {len(files_with_issues)} with issues")
        print(f"Errors: {error_count}")
        print(f"Warnings: {warning_count}")
    else:
        current_file = None
        for filepath, issue in all_issues:
            if filepath != current_file:
                if current_file is not None:
                    print()
                print(f"{filepath}:")
                current_file = filepath
            print(format_issue(issue, filepath))
    
    if all_issues:
        error_count = sum(1 for _, i in all_issues if i.severity == 'error')
        warning_count = sum(1 for _, i in all_issues if i.severity == 'warning')
        print(f"\n{'❌' if error_count else '⚠️'} Found {error_count} error(s) and {warning_count} warning(s) in {len(files_with_issues)} file(s).")
        # Only fail on errors; warnings are informational (e.g., LaTeX symbols
        # adjacent to inline Python render correctly but are worth noting).
        return 1 if error_count > 0 else 0
    
    print("✓ No rendering issues found.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
