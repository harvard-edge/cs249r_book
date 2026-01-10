#!/usr/bin/env python3
"""
Convert simple pipe-style markdown tables to restructuredText grid-style tables.

This script finds tables like:
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |

And converts them to grid format:
+----------+----------+----------+
| Header 1 | Header 2 | Header 3 |
+==========+==========+==========+
| Cell 1   | Cell 2   | Cell 3   |
+----------+----------+----------+

Usage:
    python3 convert_pipe_to_grid_tables.py --file path/to/file.qmd
    python3 convert_pipe_to_grid_tables.py --directory path/to/dir
    python3 convert_pipe_to_grid_tables.py --all  # Process all volumes
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def detect_pipe_table(lines: List[str], start_idx: int) -> Optional[Tuple[int, int]]:
    """
    Detect if a pipe table starts at the given index.
    Returns (start_idx, end_idx) if found, None otherwise.

    A pipe table consists of:
    - Header row: | col1 | col2 | col3 |
    - Separator row: |------|------|------|
    - Data rows: | val1 | val2 | val3 |
    """
    if start_idx >= len(lines):
        return None

    # Check if this line looks like a pipe table header
    line = lines[start_idx].strip()
    if not line.startswith('|') or not line.endswith('|'):
        return None

    # Must have at least 2 pipe characters (start and end, plus at least one separator)
    if line.count('|') < 3:
        return None

    # Check if next line is a separator (contains mostly dashes and pipes)
    if start_idx + 1 >= len(lines):
        return None

    separator = lines[start_idx + 1].strip()
    if not separator.startswith('|') or not separator.endswith('|'):
        return None

    # Separator should contain dashes and colons (for alignment)
    separator_content = separator[1:-1]  # Remove outer pipes
    if not re.match(r'^[\s|:\-]+$', separator_content):
        return None

    # Find the end of the table
    end_idx = start_idx + 2  # Start after separator
    while end_idx < len(lines):
        line = lines[end_idx].strip()
        if not line:  # Empty line ends table
            break
        if not line.startswith('|') or not line.endswith('|'):
            break
        end_idx += 1

    # Must have at least one data row
    if end_idx <= start_idx + 2:
        return None

    return (start_idx, end_idx)


def parse_pipe_table(lines: List[str], start_idx: int, end_idx: int) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Parse a pipe table into headers, alignments, and data rows.

    Returns:
        (headers, alignments, data_rows)
        where alignments are ['left', 'center', 'right']
    """
    # Parse header
    header_line = lines[start_idx].strip()[1:-1]  # Remove outer pipes
    headers = [cell.strip() for cell in header_line.split('|')]

    # Parse separator to get alignments
    separator = lines[start_idx + 1].strip()[1:-1]  # Remove outer pipes
    separator_parts = [part.strip() for part in separator.split('|')]

    alignments = []
    for part in separator_parts:
        has_left = part.startswith(':')
        has_right = part.endswith(':')

        if has_left and has_right:
            alignments.append('center')
        elif has_right:
            alignments.append('right')
        else:
            alignments.append('left')

    # Parse data rows
    data_rows = []
    for i in range(start_idx + 2, end_idx):
        row_line = lines[i].strip()[1:-1]  # Remove outer pipes
        cells = [cell.strip() for cell in row_line.split('|')]
        data_rows.append(cells)

    return headers, alignments, data_rows


def convert_to_grid_table(headers: List[str], alignments: List[str], data_rows: List[List[str]]) -> List[str]:
    """
    Convert parsed table data to grid format.

    This creates the basic grid structure. The format_tables.py script will
    handle proper formatting, bolding, and alignment.
    """
    # Calculate column widths (basic - formatter will refine)
    num_cols = len(headers)
    col_widths = [len(h) for h in headers]

    for row in data_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    # Build border line
    def build_border():
        parts = ['-' * (w + 2) for w in col_widths]
        return '+' + '+'.join(parts) + '+'

    # Build separator line (with alignment markers)
    def build_separator():
        parts = []
        for width, align in zip(col_widths, alignments):
            if align == 'center':
                parts.append(':' + '=' * width + ':')
            elif align == 'left':
                parts.append(':' + '=' * (width + 1))
            elif align == 'right':
                parts.append('=' * (width + 1) + ':')
            else:
                parts.append('=' * (width + 2))
        return '+' + '+'.join(parts) + '+'

    # Build data row
    def build_row(cells):
        formatted_cells = []
        for cell, width in zip(cells, col_widths):
            # Left-align content (formatter will handle proper alignment)
            formatted_cells.append(' ' + cell.ljust(width) + ' ')
        return '|' + '|'.join(formatted_cells) + '|'

    # Assemble table
    lines = []
    lines.append(build_border())
    lines.append(build_row(headers))
    lines.append(build_separator())

    for row in data_rows:
        lines.append(build_row(row))

    lines.append(build_border())

    return lines


def process_file(file_path: Path, dry_run: bool = False) -> int:
    """
    Process a single file, converting pipe tables to grid format.

    Returns:
        Number of tables converted
    """
    content = file_path.read_text(encoding='utf-8')
    lines = content.split('\n')

    converted_count = 0
    new_lines = []
    i = 0

    while i < len(lines):
        table_info = detect_pipe_table(lines, i)

        if table_info:
            start_idx, end_idx = table_info

            # Parse the pipe table
            try:
                headers, alignments, data_rows = parse_pipe_table(lines, start_idx, end_idx)

                # Convert to grid format
                grid_lines = convert_to_grid_table(headers, alignments, data_rows)

                # Add to new content
                new_lines.extend(grid_lines)

                converted_count += 1
                print(f"  Converted table at line {start_idx + 1}")

                # Skip past the original table
                i = end_idx
            except Exception as e:
                print(f"  Error converting table at line {start_idx + 1}: {e}")
                # Keep original lines
                new_lines.append(lines[i])
                i += 1
        else:
            new_lines.append(lines[i])
            i += 1

    if converted_count > 0 and not dry_run:
        file_path.write_text('\n'.join(new_lines), encoding='utf-8')
        print(f"✓ {file_path}: Converted {converted_count} tables")
    elif converted_count > 0:
        print(f"[DRY RUN] Would convert {converted_count} tables in {file_path}")

    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description='Convert pipe-style tables to grid-style tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument('-f', '--file', type=str,
                           help='Process a specific .qmd file')
    file_group.add_argument('-d', '--directory', type=str,
                           help='Process all .qmd files in a directory recursively')
    file_group.add_argument('--all', action='store_true',
                           help='Process all .qmd files in both volumes')

    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be converted without making changes')

    args = parser.parse_args()

    # Determine workspace root
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parent.parent.parent.parent.parent

    # Collect files to process
    files_to_process = []

    if args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = workspace_root / file_path
        files_to_process = [file_path]

    elif args.directory:
        dir_path = Path(args.directory)
        if not dir_path.is_absolute():
            dir_path = workspace_root / dir_path
        files_to_process = sorted(dir_path.rglob('*.qmd'))

    elif args.all:
        contents_path = workspace_root / 'book' / 'quarto' / 'contents'
        vol1_path = contents_path / 'vol1'
        vol2_path = contents_path / 'vol2'

        files_to_process = []
        if vol1_path.exists():
            files_to_process.extend(sorted(vol1_path.rglob('*.qmd')))
        if vol2_path.exists():
            files_to_process.extend(sorted(vol2_path.rglob('*.qmd')))

    # Process files
    total_converted = 0
    files_modified = 0

    for file_path in files_to_process:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        converted = process_file(file_path, args.dry_run)
        if converted > 0:
            files_modified += 1
            total_converted += converted

    # Summary
    print(f"\nSummary: {files_modified} files modified, {total_converted} tables converted")

    return 0


if __name__ == '__main__':
    sys.exit(main())
