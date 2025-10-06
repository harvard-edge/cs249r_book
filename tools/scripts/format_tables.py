#!/usr/bin/env python3
"""
Table Formatter for MLSysBook

This script formats markdown grid tables to ensure:
1. All column headers (first row) are bolded
2. All first column entries are bolded
3. Column widths are properly calculated based on content
4. Alignment bars match the actual column widths
5. Content is left-aligned within cells

Usage:
    python format_tables.py --check <file>      # Check if tables are formatted correctly
    python format_tables.py --fix <file>        # Format tables in place
    python format_tables.py --check-all         # Check all .qmd files in contents/core
    python format_tables.py --fix-all           # Format all .qmd files in contents/core
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import unicodedata


def display_width(text: str) -> int:
    """
    Calculate the display width of text, accounting for Unicode characters.
    
    Bold markers (**text**) are not counted in display width.
    East Asian Wide and Fullwidth characters count as 2.
    """
    # Remove bold markers for width calculation
    text = text.replace('**', '')
    
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width


def parse_table(lines: List[str]) -> Optional[dict]:
    """
    Parse a markdown grid table into structured data.
    
    Returns:
        dict with 'start_line', 'end_line', 'header', 'separator', 'rows', 'caption'
        or None if not a valid table
    """
    if not lines or not lines[0].startswith('+'):
        return None
    
    table_data = {
        'start_line': 0,
        'end_line': 0,
        'header_border': '',
        'header': '',
        'separator': '',
        'rows': [],
        'footer_border': '',
        'caption': ''
    }
    
    i = 0
    
    # First line should be top border
    if not lines[i].startswith('+'):
        return None
    table_data['header_border'] = lines[i]
    i += 1
    
    # Next should be header row
    if i >= len(lines) or not lines[i].startswith('|'):
        return None
    table_data['header'] = lines[i]
    i += 1
    
    # Next should be separator with := for alignment
    if i >= len(lines) or not lines[i].startswith('+'):
        return None
    table_data['separator'] = lines[i]
    i += 1
    
    # Parse data rows
    while i < len(lines):
        line = lines[i]
        if line.startswith('|'):
            table_data['rows'].append(line)
            i += 1
        elif line.startswith('+'):
            # Row separator or footer
            if i + 1 < len(lines) and lines[i + 1].startswith('|'):
                # This is a row separator, include it with the row
                table_data['rows'].append(line)
                i += 1
            else:
                # This is the footer border
                table_data['footer_border'] = line
                i += 1
                break
        else:
            break
    
    # Check for caption (starts with : after table)
    if i < len(lines) and lines[i].strip().startswith(':'):
        table_data['caption'] = lines[i]
        i += 1
    
    table_data['end_line'] = i
    
    return table_data


def parse_row(row: str) -> List[str]:
    """Parse a table row into individual cell contents."""
    # Remove leading and trailing pipes
    row = row.strip()
    if row.startswith('|'):
        row = row[1:]
    if row.endswith('|'):
        row = row[:-1]
    
    # Split by pipes and strip whitespace
    cells = [cell.strip() for cell in row.split('|')]
    return cells


def bold_text(text: str) -> str:
    """Add bold markers to text if not already bolded. Returns empty string if text is empty."""
    text = text.strip()
    # Don't bold empty strings
    if not text:
        return ''
    # Don't double-bold
    if text.startswith('**') and text.endswith('**'):
        return text
    return f"**{text}**"


def is_bolded(text: str) -> bool:
    """Check if text is already bolded."""
    text = text.strip()
    return text.startswith('**') and text.endswith('**')


def calculate_column_widths(header: str, rows: List[str]) -> List[int]:
    """
    Calculate the width needed for each column based on content.
    
    Returns list of widths for each column.
    """
    # Parse all rows to get cell contents
    all_rows = [parse_row(header)]
    for row in rows:
        if row.startswith('|'):
            all_rows.append(parse_row(row))
    
    # Find number of columns
    num_cols = len(all_rows[0])
    
    # Calculate max width for each column
    widths = [0] * num_cols
    for row in all_rows:
        for col_idx, cell in enumerate(row):
            if col_idx < num_cols:
                width = display_width(cell)
                widths[col_idx] = max(widths[col_idx], width)
    
    return widths


def extract_alignment(separator: str) -> List[str]:
    """
    Extract alignment information from separator line.
    
    Returns list of alignments: 'left', 'center', or 'right'
    """
    # Split by + to get each column separator
    parts = separator.split('+')[1:-1]  # Remove empty first and last
    
    alignments = []
    for part in parts:
        part = part.strip()
        if part.startswith(':') and part.endswith(':'):
            alignments.append('center')
        elif part.startswith(':'):
            alignments.append('left')
        elif part.endswith(':'):
            alignments.append('right')
        else:
            alignments.append('left')  # Default
    
    return alignments


def build_border(widths: List[int]) -> str:
    """Build a border line like +----+----+----+"""
    parts = ['-' * (w + 2) for w in widths]  # +2 for spaces around content
    return '+' + '+'.join(parts) + '+'


def build_separator(widths: List[int], alignments: List[str]) -> str:
    """Build separator line like +:===+:===:+====:+"""
    parts = []
    for width, align in zip(widths, alignments):
        if align == 'center':
            parts.append(':' + '=' * width + ':')
        elif align == 'left':
            parts.append(':' + '=' * width)
        elif align == 'right':
            parts.append('=' * width + ':')
        else:
            parts.append('=' * width)
    return '+' + '+'.join(parts) + '+'


def format_cell(content: str, width: int, alignment: str = 'left') -> str:
    """
    Format cell content to fit within the specified width.
    
    Pads content to match width, accounting for display width.
    """
    content = content.strip()
    display_w = display_width(content)
    padding_needed = width - display_w
    
    if alignment == 'center':
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        return ' ' * left_pad + content + ' ' * right_pad
    elif alignment == 'right':
        return ' ' * padding_needed + content
    else:  # left
        return content + ' ' * padding_needed


def format_row(cells: List[str], widths: List[int], alignments: List[str], bold_first: bool = False) -> str:
    """Format a row with proper cell widths and optional bolding of first column."""
    formatted_cells = []
    for idx, (cell, width, align) in enumerate(zip(cells, widths, alignments)):
        # Bold first column if requested
        if idx == 0 and bold_first and not is_bolded(cell):
            cell = bold_text(cell)
        formatted = format_cell(cell, width, align)
        formatted_cells.append(formatted)
    
    return '| ' + ' | '.join(formatted_cells) + ' |'


def format_table(table_data: dict) -> List[str]:
    """
    Format a complete table with proper bolding and column widths.
    
    Returns formatted table as list of lines.
    """
    # Parse header and rows
    header_cells = parse_row(table_data['header'])
    alignments = extract_alignment(table_data['separator'])
    
    # Bold all header cells
    header_cells = [bold_text(cell) for cell in header_cells]
    
    # Parse and prepare data rows (exclude border lines)
    data_rows = []
    for row in table_data['rows']:
        if row.startswith('|'):
            cells = parse_row(row)
            # Bold first column only if it's not empty
            if cells and cells[0].strip() and not is_bolded(cells[0]):
                cells[0] = bold_text(cells[0])
            data_rows.append(cells)
    
    # Calculate column widths based on all content
    all_cells = [header_cells] + data_rows
    num_cols = len(header_cells)
    widths = [0] * num_cols
    
    for row in all_cells:
        for col_idx, cell in enumerate(row):
            if col_idx < num_cols:
                width = display_width(cell)
                widths[col_idx] = max(widths[col_idx], width)
    
    # Build formatted table
    formatted = []
    
    # Top border
    formatted.append(build_border(widths))
    
    # Header row
    formatted.append(format_row(header_cells, widths, alignments, bold_first=False))
    
    # Separator
    formatted.append(build_separator(widths, alignments))
    
    # Data rows with borders
    for i, row_cells in enumerate(data_rows):
        formatted.append(format_row(row_cells, widths, alignments, bold_first=False))
        # Add row separator (border) after each data row except the last
        if i < len(data_rows) - 1:
            formatted.append(build_border(widths))
    
    # Footer border
    formatted.append(build_border(widths))
    
    # Caption
    if table_data.get('caption'):
        formatted.append('')  # Empty line before caption
        formatted.append(table_data['caption'])
    
    return formatted


def check_table_format(table_data: dict) -> List[str]:
    """
    Check if a table is properly formatted.
    
    Returns list of issues found (empty if table is correct).
    """
    issues = []
    
    # Parse header
    header_cells = parse_row(table_data['header'])
    
    # Check if all headers are bolded
    for idx, cell in enumerate(header_cells):
        if not is_bolded(cell):
            issues.append(f"Header column {idx + 1} is not bolded: '{cell}'")
    
    # Parse data rows and check first column (skip empty cells)
    row_num = 1
    for row in table_data['rows']:
        if row.startswith('|'):
            cells = parse_row(row)
            # Only check non-empty first column cells
            if cells and cells[0].strip() and not is_bolded(cells[0]):
                issues.append(f"First column in row {row_num} is not bolded: '{cells[0]}'")
            row_num += 1
    
    # Check column width consistency
    alignments = extract_alignment(table_data['separator'])
    header_cells_bolded = [bold_text(cell) for cell in header_cells]
    
    data_rows = []
    for row in table_data['rows']:
        if row.startswith('|'):
            cells = parse_row(row)
            # Bold first column only if it's not empty
            if cells and cells[0].strip() and not is_bolded(cells[0]):
                cells[0] = bold_text(cells[0])
            data_rows.append(cells)
    
    # Calculate expected widths
    all_cells = [header_cells_bolded] + data_rows
    num_cols = len(header_cells)
    expected_widths = [0] * num_cols
    
    for row in all_cells:
        for col_idx, cell in enumerate(row):
            if col_idx < num_cols:
                width = display_width(cell)
                expected_widths[col_idx] = max(expected_widths[col_idx], width)
    
    # Check if current borders match expected widths
    expected_border = build_border(expected_widths)
    if table_data['header_border'].strip() != expected_border.strip():
        issues.append(f"Border widths don't match content widths")
    
    return issues


def process_file(file_path: Path, fix: bool = False) -> Tuple[int, int]:
    """
    Process a single file to check or fix table formatting.
    
    Returns (tables_checked, tables_with_issues)
    """
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    tables_checked = 0
    tables_with_issues = 0
    modified = False
    
    i = 0
    new_lines = []
    
    while i < len(lines):
        # Check if this might be a table start
        if lines[i].startswith('+'):
            # Try to parse table
            table_lines = []
            j = i
            while j < len(lines):
                if lines[j].strip() == '' and table_lines and not lines[j - 1].startswith(':'):
                    break
                if lines[j].startswith(':') and table_lines:
                    table_lines.append(lines[j])
                    j += 1
                    break
                if lines[j].startswith('+') or lines[j].startswith('|'):
                    table_lines.append(lines[j])
                    j += 1
                else:
                    break
            
            table_data = parse_table(table_lines)
            
            if table_data:
                tables_checked += 1
                issues = check_table_format(table_data)
                
                if issues:
                    tables_with_issues += 1
                    if not fix:
                        print(f"  Issues found in table at line {i + 1}:")
                        for issue in issues:
                            print(f"    - {issue}")
                    else:
                        # Format the table
                        formatted = format_table(table_data)
                        new_lines.extend(formatted)
                        modified = True
                else:
                    # Table is already correct
                    new_lines.extend(table_lines)
                
                i = j
                continue
        
        # Not a table, keep line as is
        new_lines.append(lines[i])
        i += 1
    
    if fix and modified:
        # Write back to file
        file_path.write_text('\n'.join(new_lines), encoding='utf-8')
        print(f"  Fixed {tables_with_issues} tables")
    
    return tables_checked, tables_with_issues


def find_qmd_files(base_path: Path) -> List[Path]:
    """Find all .qmd files in contents/core directory."""
    core_path = base_path / "quarto" / "contents" / "core"
    if not core_path.exists():
        print(f"Error: {core_path} does not exist")
        return []
    
    qmd_files = list(core_path.rglob("*.qmd"))
    return sorted(qmd_files)


def main():
    parser = argparse.ArgumentParser(
        description="Format markdown grid tables in MLSysBook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a single file
  python format_tables.py --check quarto/contents/core/optimizations/optimizations.qmd
  
  # Fix a single file
  python format_tables.py --fix quarto/contents/core/optimizations/optimizations.qmd
  
  # Check all files
  python format_tables.py --check-all
  
  # Fix all files
  python format_tables.py --fix-all
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', metavar='FILE', help='Check table formatting in a file')
    group.add_argument('--fix', metavar='FILE', help='Fix table formatting in a file')
    group.add_argument('--check-all', action='store_true', help='Check all .qmd files')
    group.add_argument('--fix-all', action='store_true', help='Fix all .qmd files')
    
    args = parser.parse_args()
    
    # Determine workspace root (assume script is in tools/scripts/)
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parent.parent.parent
    
    if args.check or args.fix:
        # Process single file
        file_path = Path(args.check or args.fix)
        if not file_path.is_absolute():
            file_path = workspace_root / file_path
        
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist")
            return 1
        
        fix_mode = bool(args.fix)
        print(f"{'Fixing' if fix_mode else 'Checking'} {file_path.relative_to(workspace_root)}")
        
        tables_checked, tables_with_issues = process_file(file_path, fix=fix_mode)
        
        print(f"  Found {tables_checked} tables, {tables_with_issues} with issues")
        
        if not fix_mode and tables_with_issues > 0:
            return 1
        
    else:
        # Process all files
        qmd_files = find_qmd_files(workspace_root)
        
        if not qmd_files:
            print("No .qmd files found")
            return 1
        
        fix_mode = args.fix_all
        print(f"{'Fixing' if fix_mode else 'Checking'} {len(qmd_files)} files...")
        print()
        
        total_tables = 0
        total_issues = 0
        files_with_issues = []
        
        for qmd_file in qmd_files:
            tables_checked, tables_with_issues = process_file(qmd_file, fix=fix_mode)
            
            if tables_checked > 0:
                rel_path = qmd_file.relative_to(workspace_root)
                print(f"{rel_path}: {tables_checked} tables, {tables_with_issues} with issues")
                
                total_tables += tables_checked
                total_issues += tables_with_issues
                
                if tables_with_issues > 0:
                    files_with_issues.append(rel_path)
        
        print()
        print(f"Total: {total_tables} tables checked, {total_issues} with issues")
        
        if not fix_mode and total_issues > 0:
            print()
            print("Files with formatting issues:")
            for file_path in files_with_issues:
                print(f"  - {file_path}")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
