"""
Table Rendering Validator for MLSysBook.

Catches rendering issues in Quarto grid tables that the structural
formatter misses. Designed to prevent broken PDF/HTML output by detecting
content-level problems BEFORE building.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
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
        if re.match(r'^\+[-:=+]+\+\s*$', line):
            start = i
            j = i + 1
            while j < len(lines):
                if re.match(r'^\+[-:=+]+\+\s*$', lines[j]):
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
    inner = line[1:]
    if inner.endswith('|'):
        inner = inner[:-1]
    return [c.strip() for c in inner.split('|')]


def find_math_spans(line: str) -> List[Tuple[int, int]]:
    """Find all LaTeX $...$ math spans in a line, skipping currency ($0.50, $27,500)."""
    spans = []
    i = 0
    while i < len(line):
        if line[i] == '$' and (i == 0 or line[i - 1] != '\\'):
            if i + 1 < len(line) and re.match(r'[\d,]', line[i + 1]):
                i += 1
                continue
            if i + 1 < len(line) and line[i + 1] == '$':
                i += 2
                continue
            j = i + 1
            while j < len(line):
                if line[j] == '$' and line[j - 1] != '\\':
                    spans.append((i, j))
                    i = j + 1
                    break
                j += 1
            else:
                i += 1
        else:
            i += 1
    return spans


def check_bare_pipes_in_math(line: str, line_num: int, filepath: str) -> List[TableIssue]:
    """Detect bare | inside LaTeX math that will break column parsing."""
    issues = []
    if not line.startswith('|'):
        return issues

    cells = extract_cells_from_row(line)
    for cell in cells:
        math_spans = find_math_spans(cell)
        for start, end in math_spans:
            math_content = cell[start + 1:end]
            for m in re.finditer(r'(?<!\\)\|', math_content):
                prefix = math_content[:m.start()]
                if prefix.endswith(('\\lvert', '\\rvert', '\\Vert', '\\mid', '\\')):
                    continue
                issues.append(TableIssue(
                    file=filepath,
                    line=line_num,
                    severity='error',
                    code='BARE_PIPE',
                    message='Bare | in LaTeX math will be parsed as column separator. '
                            'Use \\lvert/\\rvert for absolute value or \\Vert for norms.',
                    context=line.rstrip(),
                    fixable=True,
                ))

    return issues


def check_frac_in_multiline(table: TableSpan, filepath: str) -> List[TableIssue]:
    """Detect \\frac{}{} in cells that span multiple rows (breaks PDF)."""
    issues = []
    for i, line in enumerate(table.lines):
        if not line.startswith('|'):
            continue
        if '\\frac{' in line or '\\frac ' in line or '\\dfrac{' in line:
            abs_line = table.start_line + i + 1
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
        pos = 0
        while True:
            idx = line.find('\\|', pos)
            if idx == -1:
                break
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
        all_issues.extend(check_html_entities(table, rel_path))
        all_issues.extend(check_frac_in_multiline(table, rel_path))
        all_issues.extend(check_kl_divergence_pipes(table, rel_path))
        all_issues.extend(check_missing_label(table, rel_path))

        for i, line in enumerate(table.lines):
            if line.startswith('|'):
                abs_line = table.start_line + i + 1
                all_issues.extend(check_bare_pipes_in_math(line, abs_line, rel_path))

    return all_issues
