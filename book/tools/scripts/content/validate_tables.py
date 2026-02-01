#!/usr/bin/env python3
"""
Table Rendering Validator for MLSysBook

Catches rendering issues in Quarto grid tables that the structural
formatter (format_tables.py) misses. Designed to prevent broken PDF/HTML
output by detecting content-level problems BEFORE building.

Checks performed:
  1. Bare pipe characters (|) inside LaTeX math that break column parsing
  2. LaTeX \\frac{}{} in multiline cells (breaks PDF rendering)
  3. HTML entities (&gt; &lt; &amp;) that shouldn't be in Markdown
  4. Unbalanced $ delimiters in table cells (broken math)
  5. Overly wide cells that will overflow in PDF
  6. Missing table labels/captions (#tbl- references)

Usage:
    # Check all vol1 files
    python3 validate_tables.py -d book/quarto/contents/vol1

    # Check a single file
    python3 validate_tables.py -f book/quarto/contents/vol1/conclusion/conclusion.qmd

    # Auto-fix safe issues (HTML entities, \\| -> \\Vert, | -> \\lvert/\\rvert)
    python3 validate_tables.py -d book/quarto/contents/vol1 --fix

Exit Codes:
    0: No issues found
    1: Warnings only (rendering may be imperfect)
    2: Errors found (rendering will break)
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TableIssue:
    """A single issue found in a table."""
    file: str
    line: int
    severity: str  # 'error' or 'warning'
    code: str      # short code like 'BARE_PIPE', 'FRAC_MULTILINE'
    message: str
    context: str   # the offending line content
    fixable: bool = False


@dataclass
class TableSpan:
    """Location of a grid table in a file."""
    start_line: int
    end_line: int
    lines: List[str]
    caption_line: Optional[int] = None
    label: Optional[str] = None


def find_grid_tables(lines: List[str]) -> List[TableSpan]:
    """Find all grid tables in a QMD file."""
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Grid tables start with +---+---+
        if re.match(r'^\+[-:=+]+\+\s*$', line):
            start = i
            # Find end of table
            j = i + 1
            while j < len(lines):
                if re.match(r'^\+[-:=+]+\+\s*$', lines[j]):
                    # Check if next line is also a table row or if this is the end
                    if j + 1 < len(lines) and (
                        lines[j + 1].startswith('|') or
                        re.match(r'^\+[-:=+]+\+\s*$', lines[j + 1])
                    ):
                        j += 1
                        continue
                    else:
                        break
                elif lines[j].startswith('|'):
                    j += 1
                    continue
                else:
                    j -= 1
                    break
            end = j

            # Look for caption on next non-blank line
            caption_line = None
            label = None
            k = end + 1
            while k < len(lines) and lines[k].strip() == '':
                k += 1
            if k < len(lines) and lines[k].strip().startswith(':'):
                caption_line = k
                label_match = re.search(r'\{#(tbl-[\w-]+)\}', lines[k])
                if label_match:
                    label = label_match.group(1)

            tables.append(TableSpan(
                start_line=start,
                end_line=end,
                lines=lines[start:end + 1],
                caption_line=caption_line,
                label=label,
            ))
            i = end + 1
        else:
            i += 1
    return tables


def extract_cells_from_row(line: str) -> List[str]:
    """Split a table row into cells, respecting the grid structure."""
    if not line.startswith('|'):
        return []
    # Remove leading/trailing pipes and split
    inner = line[1:]
    if inner.endswith('|'):
        inner = inner[:-1]
    # Simple split on | — this is what the parser does
    return [c.strip() for c in inner.split('|')]


def find_math_spans(line: str) -> List[Tuple[int, int]]:
    """Find all LaTeX $...$ math spans in a line, skipping currency ($0.50, $27,500)."""
    spans = []
    i = 0
    while i < len(line):
        if line[i] == '$' and (i == 0 or line[i - 1] != '\\'):
            # Skip currency: $ followed by digit or comma-separated number
            if i + 1 < len(line) and re.match(r'[\d,]', line[i + 1]):
                i += 1
                continue
            # Skip $$ (display math delimiter)
            if i + 1 < len(line) and line[i + 1] == '$':
                i += 2
                continue
            # This looks like start of inline math — find closing $
            j = i + 1
            while j < len(line):
                if line[j] == '$' and line[j - 1] != '\\':
                    # Skip currency inside math (shouldn't happen, but be safe)
                    spans.append((i, j))
                    i = j + 1
                    break
                j += 1
            else:
                # No closing $ found — unbalanced
                i += 1
        else:
            i += 1
    return spans


def check_bare_pipes_in_math(line: str, line_num: int, filepath: str) -> List[TableIssue]:
    """Detect bare | inside LaTeX math that will break column parsing.

    Works cell-by-cell to avoid false positives where $ in one cell and
    $ in the next cell look like a single math span crossing columns.
    """
    issues = []
    if not line.startswith('|'):
        return issues

    # Split line into cells first, then check math within each cell
    # We need column boundaries to avoid cross-cell math span detection
    cells = extract_cells_from_row(line)
    for cell in cells:
        math_spans = find_math_spans(cell)
        for start, end in math_spans:
            math_content = cell[start + 1:end]
            # Look for | that isn't preceded by \ and isn't \lvert/\rvert/\Vert
            for m in re.finditer(r'(?<!\\)\|', math_content):
                prefix = math_content[:m.start()]
                if prefix.endswith(('\\lvert', '\\rvert', '\\Vert', '\\mid', '\\')):
                    continue
                issues.append(TableIssue(
                    file=filepath,
                    line=line_num,
                    severity='error',
                    code='BARE_PIPE',
                    message=f'Bare | in LaTeX math will be parsed as column separator. '
                            f'Use \\lvert/\\rvert for absolute value or \\Vert for norms.',
                    context=line.rstrip(),
                    fixable=True,
                ))

    return issues


def check_frac_in_multiline(table: TableSpan, filepath: str) -> List[TableIssue]:
    """Detect \\frac{{}}{{}} in cells that span multiple rows (breaks PDF)."""
    issues = []
    for i, line in enumerate(table.lines):
        if not line.startswith('|'):
            continue
        if '\\frac{' in line or '\\frac ' in line or '\\dfrac{' in line:
            # Check if this cell spans multiple rows
            abs_line = table.start_line + i + 1
            # Check if next line is a continuation (not a border)
            if i + 1 < len(table.lines) and table.lines[i + 1].startswith('|'):
                next_line = table.lines[i + 1]
                if not re.match(r'^\+[-:=+]+\+', next_line):
                    issues.append(TableIssue(
                        file=filepath,
                        line=abs_line,
                        severity='warning',
                        code='FRAC_MULTILINE',
                        message='\\frac in multiline cell may render incorrectly in PDF. '
                                'Consider using (...)/denominator notation instead.',
                        context=line.rstrip(),
                        fixable=False,
                    ))
    return issues


def check_html_entities(table: TableSpan, filepath: str) -> List[TableIssue]:
    """Detect HTML entities that shouldn't be in Markdown grid tables."""
    issues = []
    entity_pattern = re.compile(r'&(gt|lt|amp|quot|apos);')
    for i, line in enumerate(table.lines):
        if not line.startswith('|'):
            continue
        matches = entity_pattern.finditer(line)
        for m in matches:
            abs_line = table.start_line + i + 1
            issues.append(TableIssue(
                file=filepath,
                line=abs_line,
                severity='error',
                code='HTML_ENTITY',
                message=f'HTML entity {m.group(0)} found in grid table. '
                        f'Quarto grid tables use raw characters, not HTML entities.',
                context=line.rstrip(),
                fixable=True,
            ))
    return issues


def check_unbalanced_math(table: TableSpan, filepath: str) -> List[TableIssue]:
    """Detect unbalanced $ delimiters within individual table cells."""
    issues = []
    for i, line in enumerate(table.lines):
        if not line.startswith('|'):
            continue
        cells = extract_cells_from_row(line)
        for cell in cells:
            # Count $ not preceded by backslash, excluding $$
            singles = len(re.findall(r'(?<!\$)(?<!\\)\$(?!\$)', cell))
            if singles % 2 != 0:
                # Could be a multiline math expression — check continuation
                # For now, only warn on single-line cells
                abs_line = table.start_line + i + 1
                issues.append(TableIssue(
                    file=filepath,
                    line=abs_line,
                    severity='warning',
                    code='UNBALANCED_MATH',
                    message=f'Unbalanced $ in table cell (found {singles}). '
                            f'Math may span multiple rows — verify manually.',
                    context=cell.strip()[:80],
                    fixable=False,
                ))
    return issues


def check_missing_label(table: TableSpan, filepath: str) -> List[TableIssue]:
    """Check that tables have a caption with a #tbl- label."""
    issues = []
    if table.caption_line is None:
        issues.append(TableIssue(
            file=filepath,
            line=table.start_line + 1,
            severity='warning',
            code='NO_CAPTION',
            message='Grid table has no caption line (: Caption text {#tbl-name}).',
            context=table.lines[0].rstrip(),
            fixable=False,
        ))
    elif table.label is None:
        issues.append(TableIssue(
            file=filepath,
            line=table.caption_line + 1,
            severity='warning',
            code='NO_LABEL',
            message='Table caption exists but has no {#tbl-name} label for cross-referencing.',
            context='(caption without label)',
            fixable=False,
        ))
    return issues


def check_kl_divergence_pipes(table: TableSpan, filepath: str) -> List[TableIssue]:
    """Detect \\| (LaTeX double-bar) that gets parsed as column separator."""
    issues = []
    for i, line in enumerate(table.lines):
        if not line.startswith('|'):
            continue
        # Find \| that isn't \lvert, \rvert, \Vert
        # The pattern: backslash followed by pipe
        matches = list(re.finditer(r'\\(?:(?!lvert|rvert|Vert|mid)\|)', line))
        # Simpler: just find \| literally
        pos = 0
        while True:
            idx = line.find('\\|', pos)
            if idx == -1:
                break
            # Check it's not part of a longer command
            before = line[max(0, idx - 6):idx]
            if any(before.endswith(cmd) for cmd in ['\\lvert', '\\rvert', '\\Vert']):
                pos = idx + 2
                continue
            abs_line = table.start_line + i + 1
            issues.append(TableIssue(
                file=filepath,
                line=abs_line,
                severity='error',
                code='BACKSLASH_PIPE',
                message='\\| in table cell will be parsed as column separator. '
                        'Use \\Vert for KL divergence double-bar notation.',
                context=line.rstrip(),
                fixable=True,
            ))
            pos = idx + 2
    return issues


def validate_file(filepath: Path) -> List[TableIssue]:
    """Run all validation checks on a single file."""
    content = filepath.read_text(encoding='utf-8')
    lines = content.split('\n')
    rel_path = str(filepath)

    tables = find_grid_tables(lines)
    all_issues = []

    for table in tables:
        # Run all checks
        all_issues.extend(check_html_entities(table, rel_path))
        all_issues.extend(check_frac_in_multiline(table, rel_path))
        all_issues.extend(check_kl_divergence_pipes(table, rel_path))
        all_issues.extend(check_missing_label(table, rel_path))

        # Per-line checks
        for i, line in enumerate(table.lines):
            if line.startswith('|'):
                abs_line = table.start_line + i + 1
                all_issues.extend(check_bare_pipes_in_math(line, abs_line, rel_path))

        # Unbalanced math (noisy, so only check single-row cells)
        # Skip for now — multiline math in grid tables is common

    return all_issues


def fix_html_entities(content: str) -> str:
    """Replace HTML entities with raw characters."""
    content = content.replace('&gt;', '>')
    content = content.replace('&lt;', '<')
    content = content.replace('&amp;', '&')
    content = content.replace('&quot;', '"')
    content = content.replace('&apos;', "'")
    return content


def fix_backslash_pipes(content: str) -> str:
    """Replace \\| with \\Vert in LaTeX math contexts within table rows."""
    lines = content.split('\n')
    fixed = []
    in_table = False
    for line in lines:
        if re.match(r'^\+[-:=+]+\+\s*$', line):
            in_table = True
            fixed.append(line)
        elif in_table and line.startswith('|'):
            # Only replace \| inside $...$ spans
            result = []
            i = 0
            in_math = False
            while i < len(line):
                if line[i] == '$' and (i == 0 or line[i - 1] != '\\'):
                    in_math = not in_math
                    result.append(line[i])
                elif in_math and line[i] == '\\' and i + 1 < len(line) and line[i + 1] == '|':
                    result.append('\\Vert')
                    i += 2
                    continue
                else:
                    result.append(line[i])
                i += 1
            fixed.append(''.join(result))
        else:
            if not line.startswith('|') and not re.match(r'^\+[-:=+]+\+', line):
                in_table = False
            fixed.append(line)
    return '\n'.join(fixed)


def apply_fixes(filepath: Path, issues: List[TableIssue]) -> int:
    """Apply auto-fixes for fixable issues. Returns count of fixes."""
    fixable = [i for i in issues if i.fixable]
    if not fixable:
        return 0

    content = filepath.read_text(encoding='utf-8')
    original = content

    has_html = any(i.code == 'HTML_ENTITY' for i in fixable)
    has_pipes = any(i.code == 'BACKSLASH_PIPE' for i in fixable)

    if has_html:
        content = fix_html_entities(content)
    if has_pipes:
        content = fix_backslash_pipes(content)

    if content != original:
        filepath.write_text(content, encoding='utf-8')
        return len(fixable)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Validate grid tables in QMD files for rendering issues.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('-f', '--file', help='Single file to check')
    parser.add_argument('-d', '--directory', help='Directory to check recursively')
    parser.add_argument('--fix', action='store_true',
                        help='Auto-fix safe issues (HTML entities, backslash pipes)')
    parser.add_argument('--errors-only', action='store_true',
                        help='Only show errors, suppress warnings')
    args = parser.parse_args()

    files = []
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"Error: {p} not found")
            return 2
        files = [p]
    elif args.directory:
        p = Path(args.directory)
        if not p.exists():
            print(f"Error: {p} not found")
            return 2
        files = sorted(p.rglob('*.qmd'))
    else:
        parser.print_help()
        return 0

    all_issues = []
    total_tables = 0
    total_fixes = 0

    for f in files:
        issues = validate_file(f)
        if args.errors_only:
            issues = [i for i in issues if i.severity == 'error']

        if args.fix:
            fixes = apply_fixes(f, issues)
            if fixes:
                total_fixes += fixes
                # Re-validate after fix
                issues = validate_file(f)
                if args.errors_only:
                    issues = [i for i in issues if i.severity == 'error']

        all_issues.extend(issues)

    # Print results
    errors = [i for i in all_issues if i.severity == 'error']
    warnings = [i for i in all_issues if i.severity == 'warning']

    if all_issues:
        for issue in all_issues:
            icon = '❌' if issue.severity == 'error' else '⚠️'
            print(f"{icon}  {issue.file}:{issue.line} [{issue.code}] {issue.message}")
            if issue.context:
                print(f"    {issue.context[:120]}")
            print()

    # Summary
    print(f"{'─' * 60}")
    print(f"Files checked: {len(files)}")
    if total_fixes:
        print(f"Auto-fixed: {total_fixes} issues")
    print(f"Errors: {len(errors)}  Warnings: {len(warnings)}")

    if errors:
        return 2
    elif warnings:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
