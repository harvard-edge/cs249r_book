#!/usr/bin/env python3
"""
Production Table Formatter for MLSysBook

This script formats and validates grid tables in Quarto .qmd files.
Designed for use in CI/CD pipelines and pre-commit hooks.

Key Features:
- Smart column header bolding (always, including multiline headers)
- Intelligent first column bolding (based on content analysis)
- Proper spacing calculation accounting for bold markers
- Handles multiline headers, multiline cells, empty cells, and Unicode
- Comprehensive validation with detailed error reporting
- Exit codes suitable for CI/CD integration

Usage:
    # Check single file
    python format_tables.py --check -f quarto/contents/core/efficient_ai/efficient_ai.qmd

    # Fix single file
    python format_tables.py --fix -f quarto/contents/core/efficient_ai/efficient_ai.qmd

    # Check all files in a directory
    python format_tables.py --check -d quarto/contents/core/optimizations

    # Fix all chapter files
    python format_tables.py --fix --all

    # With text wrapping
    python format_tables.py --fix --all --max-width 60

Exit Codes:
    0: Success (all tables properly formatted)
    1: Formatting issues found
    2: Validation errors (structural problems)
    3: File errors
"""

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ExitCode(Enum):
    """Exit codes for CI/CD integration."""
    SUCCESS = 0
    FORMATTING_ISSUES = 1
    VALIDATION_ERRORS = 2
    FILE_ERRORS = 3


@dataclass
class TableIssue:
    """Represents an issue found in a table."""
    line_num: int
    severity: str  # 'error' or 'warning'
    message: str


class GridTableParser:
    """Parser for grid-style markdown tables."""

    def __init__(self, lines: List[str], start_line: int = 0):
        self.lines = lines
        self.start_line = start_line
        self.issues: List[TableIssue] = []

        # Parsed components
        self.header_rows: List[List[str]] = []  # Changed to support multiline headers
        self.header_cells: List[str] = []  # Kept for backward compatibility (first row)
        self.data_rows: List[List[str]] = []
        self.alignments: List[str] = []
        self.num_columns = 0
        self.row_has_border_after: List[bool] = []  # Track which data rows have borders after them

    def parse(self) -> bool:
        """
        Parse the table. Returns True if successful, False otherwise.
        Issues are stored in self.issues.
        """
        if not self.lines or not self.lines[0].strip().startswith('+'):
            self.issues.append(TableIssue(
                self.start_line, 'error',
                'Table must start with border line (+----+...)'
            ))
            return False

        try:
            # Skip first border
            idx = 1

            # Parse header rows (may be multiline)
            if idx >= len(self.lines) or not self.lines[idx].strip().startswith('|'):
                self.issues.append(TableIssue(
                    self.start_line + idx, 'error',
                    'Expected header row after top border'
                ))
                return False

            # Read all header rows until we hit the separator
            while idx < len(self.lines) and self.lines[idx].strip().startswith('|'):
                header_row = self._parse_row(self.lines[idx])
                if not self.header_rows:  # First header row
                    self.num_columns = len(header_row)
                    self.header_cells = header_row  # For backward compatibility
                elif len(header_row) != self.num_columns:
                    self.issues.append(TableIssue(
                        self.start_line + idx, 'error',
                        f'Header row has {len(header_row)} columns, expected {self.num_columns}'
                    ))
                    return False
                self.header_rows.append(header_row)
                idx += 1

            if not self.header_rows:
                self.issues.append(TableIssue(
                    self.start_line + idx, 'error',
                    'No header rows found'
                ))
                return False

            # Parse separator with alignments
            if idx >= len(self.lines) or not self.lines[idx].strip().startswith('+'):
                self.issues.append(TableIssue(
                    self.start_line + idx, 'error',
                    'Expected separator with alignment markers (+:===+...)'
                ))
                return False

            self.alignments = self._extract_alignments(self.lines[idx])
            if len(self.alignments) != self.num_columns:
                self.issues.append(TableIssue(
                    self.start_line + idx, 'error',
                    f'Alignment count ({len(self.alignments)}) != column count ({self.num_columns})'
                ))
                return False
            idx += 1

            # Parse data rows
            while idx < len(self.lines):
                line = self.lines[idx].strip()
                if line.startswith('|'):
                    cells = self._parse_row(line)
                    if len(cells) != self.num_columns:
                        self.issues.append(TableIssue(
                            self.start_line + idx, 'error',
                            f'Row has {len(cells)} columns, expected {self.num_columns}'
                        ))
                        return False
                    self.data_rows.append(cells)
                    idx += 1

                    # Check if next line is a border
                    if idx < len(self.lines) and self.lines[idx].strip().startswith('+'):
                        self.row_has_border_after.append(True)
                        idx += 1  # Skip the border
                        if idx >= len(self.lines) or not self.lines[idx].strip().startswith('|'):
                            # End of table
                            break
                    else:
                        self.row_has_border_after.append(False)
                elif line.startswith('+'):
                    # Unexpected border (shouldn't happen with the logic above, but just in case)
                    idx += 1
                    if idx >= len(self.lines) or not self.lines[idx].strip().startswith('|'):
                        # End of table
                        break
                else:
                    # End of table
                    break

            return True

        except Exception as e:
            self.issues.append(TableIssue(
                self.start_line, 'error',
                f'Parsing error: {str(e)}'
            ))
            return False

    def _parse_row(self, row: str) -> List[str]:
        """Parse a table row into cells."""
        row = row.strip()
        if row.startswith('|'):
            row = row[1:]
        if row.endswith('|'):
            row = row[:-1]
        return [cell.strip() for cell in row.split('|')]

    def _extract_alignments(self, separator: str) -> List[str]:
        """Extract alignment from separator line."""
        parts = separator.strip().split('+')[1:-1]
        alignments = []

        for part in parts:
            part = part.strip()
            if not part:
                continue
            has_left = part.startswith(':')
            has_right = part.endswith(':')

            if has_left and has_right:
                alignments.append('center')
            elif has_left:
                alignments.append('left')
            elif has_right:
                alignments.append('right')
            else:
                alignments.append('left')

        return alignments


def display_width(text: str) -> int:
    """
    Calculate display width of text.
    Bold markers (**) don't count toward width.
    Unicode wide characters count as 2.
    """
    # Remove bold markers
    clean_text = text.replace('**', '')

    width = 0
    for char in clean_text:
        ea_width = unicodedata.east_asian_width(char)
        if ea_width in ('F', 'W'):  # Fullwidth or Wide
            width += 2
        else:
            width += 1

    return width


def is_bolded(text: str) -> bool:
    """Check if text is already bolded."""
    text = text.strip()
    return (text.startswith('**') and text.endswith('**') and len(text) > 4)


def add_bold(text: str) -> str:
    """Add bold markers to text if not already bolded. Returns empty string for empty text."""
    text = text.strip()
    if not text:
        return ''
    if is_bolded(text):
        return text
    return f"**{text}**"


def remove_bold(text: str) -> str:
    """Remove bold markers from text."""
    text = text.strip()
    if is_bolded(text):
        return text[2:-2]
    return text


def detect_column_alignments(header_rows: List[List[str]], data_rows: List[List[str]]) -> List[str]:
    """
    Detect optimal alignment for each column based on content.

    Rules:
    - Numeric columns (>70% numbers): right-aligned
    - Text columns: left-aligned
    - Mixed: left-aligned (default)
    """
    if not data_rows or not header_rows:
        return ['left'] * len(header_rows[0]) if header_rows else []

    num_columns = len(header_rows[0])
    alignments = []

    for col_idx in range(num_columns):
        # Collect all values in this column (skip empty cells)
        column_values = []
        for row in data_rows:
            if col_idx < len(row):
                cell = row[col_idx].strip()
                # Remove bold markers for analysis
                if cell.startswith('**') and cell.endswith('**'):
                    cell = cell[2:-2].strip()
                if cell:  # Skip empty cells
                    column_values.append(cell)

        if not column_values:
            alignments.append('left')
            continue

        # Count numeric cells
        numeric_count = 0
        for value in column_values:
            # Remove common formatting: commas, spaces, currency symbols
            clean_value = value.replace(',', '').replace(' ', '').replace('$', '')
            # Remove units (W, mW, µW, KB, MB, GB, etc.)
            clean_value = ''.join(c for c in clean_value if c.isdigit() or c in '.-+<>~')

            # Check if it's primarily numeric
            if clean_value and any(c.isdigit() for c in clean_value):
                numeric_count += 1

        # If >70% of cells are numeric, right-align
        if numeric_count / len(column_values) > 0.7:
            alignments.append('right')
        else:
            alignments.append('left')

    return alignments


def should_bold_first_column(header_cells: List[str], data_rows: List[List[str]]) -> bool:
    """
    Determine if first column should be bolded based on intelligent analysis.

    Returns True for comparison/definition tables where first column contains:
    - Category names (Aspect, Technique, Category, Architecture, etc.)
    - Descriptive multi-word phrases

    Returns False for data tables where first column contains:
    - Numbers, IDs, years
    - Simple enumeration
    """
    if not header_cells:
        return False

    first_header = remove_bold(header_cells[0]).lower()

    # Keywords that indicate first column should be bolded
    bold_indicators = [
        'aspect', 'technique', 'category', 'architecture', 'challenge',
        'criterion', 'criteria', 'feature', 'characteristic', 'dimension',
        'metric', 'property', 'attribute', 'method', 'approach', 'strategy',
        'type', 'principle', 'factor', 'component', 'element', 'term',
        'concept', 'deployment context', 'system aspect', 'design pattern',
        'era', 'role', 'threat type', 'mechanism', 'resource type',
        'storage tier', 'stage', 'characteristic'
    ]

    # Check if header matches bold indicators
    if any(indicator in first_header for indicator in bold_indicators):
        return True

    # Keywords that indicate DON'T bold
    no_bold_indicators = [
        'id', '#', 'number', 'index', 'rank', 'year', 'date', 'time',
        'count', 'order'
    ]

    if any(indicator in first_header for indicator in no_bold_indicators):
        return False

    # Analyze first column content
    if not data_rows:
        return True  # Default to bolding if no data

    first_col_values = [row[0] for row in data_rows if row and row[0].strip()]

    if not first_col_values:
        return True

    # Check if mostly numeric (data table)
    numeric_count = 0
    for value in first_col_values:
        clean = remove_bold(value).replace('%', '').replace('$', '').replace(',', '').strip()
        try:
            float(clean)
            numeric_count += 1
        except ValueError:
            pass

    if numeric_count > len(first_col_values) * 0.7:
        return False

    # Check if descriptive (multi-word = comparison table)
    descriptive_count = 0
    for value in first_col_values:
        clean = remove_bold(value)
        words = clean.replace('/', ' ').replace('-', ' ').replace('(', ' ').split()
        # Filter out empty words
        words = [w for w in words if w.strip()]
        if len(words) >= 2:
            descriptive_count += 1

    if descriptive_count > len(first_col_values) * 0.4:
        return True

    # Default: bold for comparison-style tables
    return True


def calculate_column_widths(parser: GridTableParser,
                            bold_headers: bool = True,
                            bold_first_col: bool = False) -> List[int]:
    """
    Calculate required width for each column, accounting for bolding.
    """
    widths = [0] * parser.num_columns

    # Header widths (with potential bolding)
    for i, cell in enumerate(parser.header_cells):
        text = cell
        if bold_headers and not is_bolded(cell) and cell.strip():
            text = add_bold(cell)
        widths[i] = max(widths[i], display_width(text))

    # Data row widths
    for row in parser.data_rows:
        for i, cell in enumerate(row):
            text = cell
            # First column might need bolding (but not if empty - multiline cells)
            if i == 0 and bold_first_col and cell.strip() and not is_bolded(cell):
                text = add_bold(cell)
            widths[i] = max(widths[i], display_width(text))

    return widths


def build_border(widths: List[int]) -> str:
    """Build border line: +----+----+----+"""
    parts = ['-' * (w + 2) for w in widths]  # +2 for padding spaces
    return '+' + '+'.join(parts) + '+'


def build_separator(widths: List[int], alignments: List[str]) -> str:
    """Build separator line: +:===+:===:+====:+

    The separator must match the border length exactly.
    Border segment for width W: '-' * (W + 2)  [+2 for padding spaces]
    So separator segment must also be length (W + 2).
    """
    parts = []
    for width, align in zip(widths, alignments):
        if align == 'center':
            # :===: format - colon + equals + colon = W+2
            parts.append(':' + '=' * width + ':')
        elif align == 'left':
            # :==== format - colon + equals = W+2
            parts.append(':' + '=' * (width + 1))
        elif align == 'right':
            # ====: format - equals + colon = W+2
            parts.append('=' * (width + 1) + ':')
        else:
            # ===== format - no alignment markers = W+2
            parts.append('=' * (width + 2))
    return '+' + '+'.join(parts) + '+'


def escape_html_entities(content: str) -> str:
    r"""Convert bare < and > to HTML entities (&lt; and &gt;).

    Preserves:
    - Already-escaped sequences like \> and \<
    - HTML tags like <li>, </li>, <ul>, etc.
    """
    import re

    # Temporarily protect escaped sequences and HTML tags
    content = content.replace('\\>', '\x00ESCAPED_GT\x00')  # Protect \>
    content = content.replace('\\<', '\x00ESCAPED_LT\x00')  # Protect \<

    # Protect HTML tags (e.g., <li>, </li>, <ul>, <p>, etc.)
    # Match opening tags: <tagname> or <tagname attr="value">
    # Match closing tags: </tagname>
    # Match self-closing: <tagname />
    html_tag_pattern = r'</?[a-zA-Z][a-zA-Z0-9]*(?:\s+[^>]*)?/?>'
    tags = re.findall(html_tag_pattern, content)
    for i, tag in enumerate(tags):
        content = content.replace(tag, f'\x00TAG_{i}\x00', 1)

    # Now convert bare < and >
    content = content.replace('>', '&gt;')
    content = content.replace('<', '&lt;')

    # Restore HTML tags
    for i, tag in enumerate(tags):
        content = content.replace(f'\x00TAG_{i}\x00', tag)

    # Restore escaped sequences
    content = content.replace('\x00ESCAPED_GT\x00', '\\>')
    content = content.replace('\x00ESCAPED_LT\x00', '\\<')

    return content


def format_cell(content: str, width: int, alignment: str = 'left') -> str:
    """Format cell content with proper padding.

    Width is the LITERAL character count (including ** markers and HTML entities).
    Always left-aligns content within cells (the alignment parameter only
    affects column alignment markers in the separator row).
    """
    content = content.strip()
    content_len = len(content)  # Literal length including ** and HTML entities
    padding = width - content_len

    if padding < 0:
        padding = 0

    # Always left-align cell content (padding on right only)
    return content + ' ' * padding


def wrap_cell_text(text: str, max_width: int) -> List[str]:
    """
    Wrap text to fit within max_width, breaking at natural points.

    Returns list of lines (wrapped text).
    """
    text = text.strip()

    # If text fits, no wrapping needed
    if len(text) <= max_width:
        return [text]

    # Find good break points: commas, semicolons, " and ", " or "
    lines = []
    current_line = ""

    # Split by commas first (most common)
    parts = text.split(',')

    for i, part in enumerate(parts):
        part = part.strip()

        # Add comma back except for last part
        if i < len(parts) - 1:
            part_with_comma = part + ','
        else:
            part_with_comma = part

        # Check if adding this part would exceed max_width
        if not current_line:
            # First part of line
            current_line = part_with_comma
        elif len(current_line + ' ' + part_with_comma) <= max_width:
            # Fits on current line
            current_line = current_line + ' ' + part_with_comma
        else:
            # Need to start new line
            lines.append(current_line)
            current_line = part_with_comma

    # Add remaining text
    if current_line:
        lines.append(current_line)

    return lines


def wrap_table_rows(data_rows: List[List[str]], max_width: int) -> List[List[str]]:
    """
    Wrap cells in data rows that exceed max_width.

    Creates continuation rows where needed.
    """
    if max_width is None:
        return data_rows

    wrapped_rows = []

    for row in data_rows:
        # Check if any cell needs wrapping
        needs_wrapping = False
        wrapped_cells = []
        max_lines = 1

        for cell in row:
            wrapped = wrap_cell_text(cell, max_width)
            wrapped_cells.append(wrapped)
            max_lines = max(max_lines, len(wrapped))
            if len(wrapped) > 1:
                needs_wrapping = True

        if not needs_wrapping:
            # No wrapping needed, keep original row
            wrapped_rows.append(row)
        else:
            # Create multiple rows (one per line)
            for line_idx in range(max_lines):
                new_row = []
                for col_idx, cell_lines in enumerate(wrapped_cells):
                    if line_idx < len(cell_lines):
                        new_row.append(cell_lines[line_idx])
                    else:
                        # Empty cell for continuation
                        new_row.append('')
                wrapped_rows.append(new_row)

    return wrapped_rows


def format_table_lines(parser: GridTableParser, max_width: Optional[int] = None) -> List[str]:
    """Format a parsed table into properly formatted lines."""
    # Determine formatting rules
    bold_headers = True  # Always bold headers
    bold_first_col = should_bold_first_column(parser.header_cells, parser.data_rows)

    # Auto-detect optimal alignments (text=left, numbers=right)
    optimal_alignments = detect_column_alignments(parser.header_rows, parser.data_rows)

    # Apply text wrapping FIRST (before bolding)
    wrapped_data = wrap_table_rows(parser.data_rows, max_width)

    # Prepare ALL header rows (support multiline headers)
    formatted_header_rows = []
    for header_row in parser.header_rows:
        formatted_row = []
        for cell in header_row:
            # First escape HTML entities
            cell = escape_html_entities(cell)
            # Then apply bolding if needed
            if bold_headers and cell.strip() and not is_bolded(cell):
                formatted_row.append(add_bold(cell))
            else:
                formatted_row.append(cell)
        formatted_header_rows.append(formatted_row)

    # Prepare data rows (with wrapping applied)
    formatted_data = []
    for row in wrapped_data:
        new_row = []
        for i, cell in enumerate(row):
            # First escape HTML entities
            cell = escape_html_entities(cell)
            # Bold first column only if it has content (preserve empty cells for multiline)
            if i == 0 and bold_first_col and cell.strip() and not is_bolded(cell):
                new_row.append(add_bold(cell))
            else:
                new_row.append(cell)
        formatted_data.append(new_row)

    # Calculate widths based on formatted content
    # IMPORTANT: Use len() not display_width() because restructuredText counts literal chars including **
    widths = [0] * parser.num_columns

    # Header widths (literal length of formatted/bolded text) - check ALL header rows
    for header_row in formatted_header_rows:
        for i, cell in enumerate(header_row):
            widths[i] = max(widths[i], len(cell.strip()))

    # Data widths (literal length of formatted/bolded text)
    for row in formatted_data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell.strip()))

    # Build formatted table
    lines = []

    # Top border
    lines.append(build_border(widths))

    # ALL Header rows (support multiline)
    for header_row in formatted_header_rows:
        header_cells_formatted = []
        for cell, width, align in zip(header_row, widths, optimal_alignments):
            header_cells_formatted.append(format_cell(cell, width, align))
        lines.append('| ' + ' | '.join(header_cells_formatted) + ' |')

    # Separator (use optimal alignments)
    lines.append(build_separator(widths, optimal_alignments))

    # Data rows with borders between them
    for i, row in enumerate(formatted_data):
        row_cells_formatted = []
        for cell, width, align in zip(row, widths, optimal_alignments):
            row_cells_formatted.append(format_cell(cell, width, align))
        lines.append('| ' + ' | '.join(row_cells_formatted) + ' |')

        # Add border after this row if the original table had one
        if i < len(parser.row_has_border_after) and parser.row_has_border_after[i]:
            lines.append(build_border(widths))

    # Footer border (only if last row didn't already have a border)
    if not (formatted_data and len(parser.row_has_border_after) > 0 and parser.row_has_border_after[-1]):
        lines.append(build_border(widths))

    return lines


def validate_table(parser: GridTableParser, max_width: Optional[int] = None) -> Tuple[bool, List[str]]:
    """
    Validate table formatting.

    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []

    # Check header bolding (check ALL header rows for multiline headers)
    unbolded_headers = []
    for row_idx, header_row in enumerate(parser.header_rows):
        for col_idx, cell in enumerate(header_row):
            if cell.strip() and not is_bolded(cell):
                # Track column index (1-based for human readability)
                if (col_idx + 1) not in unbolded_headers:
                    unbolded_headers.append(col_idx + 1)

    if unbolded_headers:
        warnings.append(f"Headers not bolded in columns: {', '.join(map(str, sorted(unbolded_headers)))}")

    # Check first column bolding
    if should_bold_first_column(parser.header_cells, parser.data_rows):
        unbolded_first = []
        for i, row in enumerate(parser.data_rows):
            if row[0].strip() and not is_bolded(row[0]):
                unbolded_first.append(i + 1)

        if unbolded_first:
            warnings.append(f"First column not bolded in rows: {', '.join(map(str, unbolded_first[:5]))}")
            if len(unbolded_first) > 5:
                warnings.append(f"  ... and {len(unbolded_first) - 5} more rows")

    # Check spacing
    formatted_lines = format_table_lines(parser, max_width)
    original_borders = [line for line in parser.lines if line.strip().startswith('+')]
    formatted_borders = [line for line in formatted_lines if line.startswith('+')]

    if original_borders and formatted_borders:
        if original_borders[0].strip() != formatted_borders[0].strip():
            warnings.append("Table spacing is incorrect (column widths don't match content)")

    # Check if any data lines differ (catches alignment changes)
    original_data_lines = [l.strip() for l in parser.lines if l.strip().startswith('|') and '|' in l[1:]]
    formatted_data_lines = [l.strip() for l in formatted_lines if l.startswith('|') and '|' in l[1:]]

    if original_data_lines and formatted_data_lines:
        if original_data_lines != formatted_data_lines:
            warnings.append("Cell content alignment needs updating")

    return len(warnings) == 0, warnings


def extract_tables_from_file(file_path: Path) -> List[Tuple[int, List[str], int]]:
    """
    Extract all tables from a file.

    Returns:
        List of (start_line, table_lines, end_line) tuples
    """
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')

    tables = []
    i = 0

    while i < len(lines):
        if lines[i].strip().startswith('+') and '---' in lines[i]:
            # Potential table start
            start_line = i
            table_lines = []

            while i < len(lines):
                line = lines[i].rstrip()
                if line.startswith(('+', '|')):
                    table_lines.append(line)
                    i += 1
                elif not line and table_lines:
                    # Empty line after table
                    break
                else:
                    break

            if len(table_lines) >= 5:  # Minimum valid table
                tables.append((start_line, table_lines, i))
            else:
                i = start_line + 1
        else:
            i += 1

    return tables


def process_file(file_path: Path, mode: str, verbose: bool = False, max_width: Optional[int] = None) -> Tuple[int, int, int]:
    """
    Process a single file.

    Args:
        file_path: Path to file
        mode: 'check' or 'format'
        verbose: Print detailed info

    Returns:
        (tables_found, tables_with_issues, tables_with_errors)
    """
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 0, 0, 1

    try:
        tables = extract_tables_from_file(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0, 1

    tables_found = len(tables)
    tables_with_issues = 0
    tables_with_errors = 0

    if mode == 'check':
        # Validate tables
        for start_line, table_lines, _ in tables:
            parser = GridTableParser(table_lines, start_line)

            if not parser.parse():
                tables_with_errors += 1
                print(f"\n{file_path}:{start_line + 1}: Table validation errors:")
                for issue in parser.issues:
                    print(f"  {issue.severity.upper()}: {issue.message}")
            else:
                is_valid, warnings = validate_table(parser, max_width)
                if not is_valid:
                    tables_with_issues += 1
                    if verbose or True:  # Always show in check mode
                        print(f"\n{file_path}:{start_line + 1}: Table formatting issues:")
                        for warning in warnings:
                            print(f"  - {warning}")

    elif mode == 'format':
        # Format tables
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        new_lines = []

        processed_lines = set()

        for start_line, table_lines, end_line in tables:
            parser = GridTableParser(table_lines, start_line)

            if not parser.parse():
                # Can't format invalid tables
                tables_with_errors += 1
                print(f"\n{file_path}:{start_line + 1}: Cannot format (validation errors):")
                for issue in parser.issues:
                    print(f"  {issue.severity.upper()}: {issue.message}")
                continue

            is_valid, warnings = validate_table(parser)
            if not is_valid:
                tables_with_issues += 1
                if verbose:
                    print(f"\n{file_path}:{start_line + 1}: Formatting table...")
                    for warning in warnings:
                        print(f"  Fixing: {warning}")

            # Mark these lines as processed
            for line_idx in range(start_line, end_line):
                processed_lines.add(line_idx)

        # Rebuild file with formatted tables
        i = 0
        for start_line, table_lines, end_line in tables:
            # Copy lines before table
            while i < start_line:
                new_lines.append(lines[i])
                i += 1

            # Parse and format table
            parser = GridTableParser(table_lines, start_line)
            if parser.parse():
                formatted = format_table_lines(parser, max_width)
                new_lines.extend(formatted)
            else:
                # Keep original if can't parse
                new_lines.extend(table_lines)

            i = end_line

        # Copy remaining lines
        while i < len(lines):
            new_lines.append(lines[i])
            i += 1

        # Write back
        if tables_with_issues > 0:
            file_path.write_text('\n'.join(new_lines), encoding='utf-8')
            print(f"{file_path}: Formatted {tables_with_issues} tables")

    return tables_found, tables_with_issues, tables_with_errors


def main():
    parser = argparse.ArgumentParser(
        description='Production table formatter for MLSysBook',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # File/directory selection (consistent with other scripts)
    file_group = parser.add_mutually_exclusive_group()
    file_group.add_argument('-f', '--file', type=str,
                           help='Process a specific .qmd file')
    file_group.add_argument('-d', '--directory', type=str,
                           help='Process all .qmd files in a directory recursively')
    file_group.add_argument('--all', action='store_true',
                           help='Process all .qmd files in quarto/contents/core')

    # Action selection
    action_group = parser.add_mutually_exclusive_group(required=False)
    action_group.add_argument('--check', action='store_true',
                             help='Check formatting only (default)')
    action_group.add_argument('--fix', action='store_true',
                             help='Fix table formatting in place')

    # Options
    parser.add_argument('--max-width', type=int, default=None,
                       help='Maximum cell width before wrapping (default: no wrapping)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Determine mode
    if args.fix:
        mode = 'format'
    else:
        mode = 'check'  # Default to check

    # Determine files to process
    script_path = Path(__file__).resolve()
    # Script is at book/tools/scripts/content/format_tables.py, need 5 parents to get to repo root
    workspace_root = script_path.parent.parent.parent.parent.parent
    files_to_process = []

    if args.file:
        # Single file
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = workspace_root / file_path
        files_to_process = [file_path]

    elif args.directory:
        # Directory
        dir_path = Path(args.directory)
        if not dir_path.is_absolute():
            dir_path = workspace_root / dir_path

        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            return ExitCode.FILE_ERRORS.value

        files_to_process = sorted(dir_path.rglob('*.qmd'))

    elif args.all:
        # All chapter files
        core_path = workspace_root / 'quarto' / 'contents' / 'core'

        if not core_path.exists():
            print(f"Error: {core_path} does not exist")
            return ExitCode.FILE_ERRORS.value

        files_to_process = sorted(core_path.rglob('*.qmd'))

    else:
        parser.print_help()
        return ExitCode.SUCCESS.value

    # Process files
    total_tables = 0
    total_issues = 0
    total_errors = 0

    for file_path in files_to_process:
        tables, issues, errors = process_file(file_path, mode, args.verbose, args.max_width)
        total_tables += tables
        total_issues += issues
        total_errors += errors

    # Print summary
    print(f"\nSummary: {total_tables} tables, {total_issues} with formatting issues, {total_errors} with errors")

    # Determine exit code
    if total_errors > 0:
        return ExitCode.VALIDATION_ERRORS.value
    elif total_issues > 0 and mode == 'check':
        return ExitCode.FORMATTING_ISSUES.value
    else:
        return ExitCode.SUCCESS.value


if __name__ == '__main__':
    sys.exit(main())
