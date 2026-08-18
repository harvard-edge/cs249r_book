"""
Convert grid tables to pipe tables in Quarto/Markdown files.

Grid tables use + for corners and borders. Pipe tables are the book's canonical standard.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple


def parse_alignment(separator_line: str) -> list[str]:
    """Parse alignment from grid table separator line."""
    alignments = []
    parts = [p for p in separator_line.split('+') if p]
    for part in parts:
        part = part.strip()
        if not part:
            continue
        starts_colon = part.startswith(':')
        ends_colon = part.endswith(':')
        if starts_colon and ends_colon:
            alignments.append(':---:')
        elif ends_colon:
            alignments.append('---:')
        elif starts_colon:
            alignments.append(':---')
        else:
            alignments.append('---')
    return alignments


def parse_grid_table(table_lines: list[str]) -> tuple[list[str], list[list[str]], list[str]]:
    """Parse a grid table into headers, rows, and alignments."""
    headers = []
    rows = []
    alignments = []
    in_header = True
    current_row = None

    for line in table_lines:
        line = line.rstrip()
        if '=' in line and line.startswith('+'):
            alignments = parse_alignment(line)
            if current_row is not None:
                headers = current_row
                current_row = None
            in_header = False
            continue

        if line.startswith('+') and '-' in line:
            if current_row is not None and not in_header:
                rows.append(current_row)
                current_row = None
            continue

        if line.startswith('|'):
            cells = line.strip('|').split('|')
            cells = [c.strip() for c in cells]
            if current_row is None:
                current_row = cells
            else:
                if cells[0] == '':
                    for i, cell in enumerate(cells):
                        if i < len(current_row) and cell:
                            if current_row[i]:
                                current_row[i] += ' ' + cell
                            else:
                                current_row[i] = cell
                else:
                    if not in_header:
                        rows.append(current_row)
                    else:
                        headers = current_row
                        in_header = False
                    current_row = cells

    if current_row is not None:
        if in_header:
            headers = current_row
        else:
            rows.append(current_row)

    if not alignments and headers:
        alignments = ['---'] * len(headers)

    return headers, rows, alignments


def grid_table_to_pipe(table_lines: list[str]) -> str:
    """Convert grid table lines to pipe table format."""
    headers, rows, alignments = parse_grid_table(table_lines)
    if not headers:
        return '\n'.join(table_lines)

    while len(alignments) < len(headers):
        alignments.append('---')
    alignments = alignments[:len(headers)]

    result = []
    result.append('| ' + ' | '.join(headers) + ' |')
    result.append('|' + '|'.join(alignments) + '|')
    for row in rows:
        while len(row) < len(headers):
            row.append('')
        row = row[:len(headers)]
        result.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(result)


def find_grid_tables(content: str) -> list[tuple[int, int, list[str]]]:
    """Find all grid tables in content."""
    lines = content.split('\n')
    tables = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if re.match(r'^\+[-:=+]+\+\s*$', line):
            start = i
            table_lines = [line]
            i += 1
            while i < len(lines):
                current = lines[i]
                if re.match(r'^\+[-:=+]+\+\s*$', current) or current.strip().startswith('|'):
                    table_lines.append(current)
                    i += 1
                else:
                    break
            if any('=' in line for line in table_lines):
                tables.append((start, i - 1, table_lines))
        else:
            i += 1

    return tables


def check_grid_tables(content: str) -> list[tuple[int, str]]:
    """Check for grid tables in content."""
    tables = find_grid_tables(content)
    issues = []
    for start, end, table_lines in tables:
        data_line = next((l for l in table_lines if l.startswith('|')), '')
        preview = data_line[:60] + '...' if len(data_line) > 60 else data_line
        issues.append((start + 1, f"Grid table ({end - start + 1} lines): {preview}"))
    return issues


def convert_grid_tables(content: str) -> tuple[str, int]:
    """Convert all grid tables to pipe tables."""
    tables = find_grid_tables(content)
    if not tables:
        return content, 0
    lines = content.split('\n')
    for start, end, table_lines in reversed(tables):
        pipe_table = grid_table_to_pipe(table_lines)
        lines[start:end + 1] = pipe_table.split('\n')
    return '\n'.join(lines), len(tables)
