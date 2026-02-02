#!/usr/bin/env python3
"""
Convert grid tables to pipe tables in Quarto/Markdown files.

Grid tables use + for corners and borders:
+---------------+------------------+
| **Paradigm**  | **Where**        |
+:==============+:=================+   <- alignment markers here
| **Cloud ML**  | Data centers     |
+---------------+------------------+

Alignment markers in header separator:
  :===  = left align
  ===:  = right align
  :===: = center align
  ====  = default (left)

Pipe tables (output):
| **Paradigm** | **Where** |
|:-------------|:----------|
| **Cloud ML** | Data centers |

Usage:
  # Check mode (warn only, for pre-commit --check):
  python convert_grid_to_pipe_tables.py --check file1.qmd file2.qmd

  # Fix mode (auto-convert):
  python convert_grid_to_pipe_tables.py file1.qmd file2.qmd
"""

import re
import sys
from pathlib import Path


def parse_alignment(separator_line: str) -> list[str]:
    """
    Parse alignment from grid table separator line.
    
    Example: +:==============+:=================+============:+
    Returns: [':--', ':--', '--:', ...]
    """
    alignments = []
    # Split by + and remove empty strings at start/end
    parts = [p for p in separator_line.split('+') if p]
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        starts_colon = part.startswith(':')
        ends_colon = part.endswith(':')
        
        if starts_colon and ends_colon:
            alignments.append(':---:')  # center
        elif ends_colon:
            alignments.append('---:')   # right
        elif starts_colon:
            alignments.append(':---')   # left
        else:
            alignments.append('---')    # default
    
    return alignments


def parse_grid_table(table_lines: list[str]) -> tuple[list[str], list[list[str]], list[str]]:
    """
    Parse a grid table into headers, rows, and alignments.
    
    Returns: (headers, data_rows, alignments)
    """
    headers = []
    rows = []
    alignments = []
    
    in_header = True
    
    for line in table_lines:
        line = line.rstrip()
        
        # Header separator (with =) - extract alignment
        if '=' in line and line.startswith('+'):
            alignments = parse_alignment(line)
            in_header = False
            continue
        
        # Row separator (just -)
        if line.startswith('+') and '-' in line:
            continue
        
        # Data line
        if line.startswith('|'):
            # Split by | and clean up cells
            cells = [c.strip() for c in line.split('|')]
            # Remove empty first/last from split
            cells = [c for c in cells if c or cells.index(c) not in (0, len(cells)-1)]
            # Actually just use a cleaner approach
            cells = line.strip('|').split('|')
            cells = [c.strip() for c in cells]
            
            if in_header:
                headers = cells
            else:
                rows.append(cells)
    
    # Default alignments if none found
    if not alignments and headers:
        alignments = ['---'] * len(headers)
    
    return headers, rows, alignments


def grid_table_to_pipe(table_lines: list[str]) -> str:
    """Convert grid table lines to pipe table format."""
    headers, rows, alignments = parse_grid_table(table_lines)
    
    if not headers:
        return '\n'.join(table_lines)  # Return unchanged if parsing failed
    
    # Ensure alignments match header count
    while len(alignments) < len(headers):
        alignments.append('---')
    alignments = alignments[:len(headers)]
    
    # Build pipe table
    result = []
    
    # Header row
    result.append('| ' + ' | '.join(headers) + ' |')
    
    # Alignment row
    result.append('|' + '|'.join(alignments) + '|')
    
    # Data rows
    for row in rows:
        # Pad row if needed
        while len(row) < len(headers):
            row.append('')
        row = row[:len(headers)]  # Trim if too many
        result.append('| ' + ' | '.join(row) + ' |')
    
    return '\n'.join(result)


def find_grid_tables(content: str) -> list[tuple[int, int, list[str]]]:
    """
    Find all grid tables in content.
    
    Returns: list of (start_line, end_line, table_lines)
    """
    lines = content.split('\n')
    tables = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this starts a grid table (line of +---+)
        if re.match(r'^\+[-:=+]+\+\s*$', line):
            start = i
            table_lines = [line]
            i += 1
            
            # Collect table lines until we hit non-table content
            while i < len(lines):
                current = lines[i]
                # Table continues if it's a separator or data row
                if re.match(r'^\+[-:=+]+\+\s*$', current) or current.strip().startswith('|'):
                    table_lines.append(current)
                    i += 1
                else:
                    break
            
            # Valid table needs at least header separator with =
            if any('=' in line for line in table_lines):
                tables.append((start, i - 1, table_lines))
        else:
            i += 1
    
    return tables


def check_grid_tables(content: str) -> list[tuple[int, str]]:
    """
    Check for grid tables in content.
    
    Returns: list of (line_number, description) for each table found
    """
    tables = find_grid_tables(content)
    issues = []
    
    for start, end, table_lines in tables:
        # Get first data line for context
        data_line = next((l for l in table_lines if l.startswith('|')), '')
        preview = data_line[:60] + '...' if len(data_line) > 60 else data_line
        issues.append((start + 1, f"Grid table ({end - start + 1} lines): {preview}"))
    
    return issues


def convert_grid_tables(content: str) -> tuple[str, int]:
    """
    Convert all grid tables to pipe tables.
    
    Returns: (converted_content, number_of_conversions)
    """
    tables = find_grid_tables(content)
    
    if not tables:
        return content, 0
    
    lines = content.split('\n')
    
    # Process tables in reverse order to preserve line numbers
    for start, end, table_lines in reversed(tables):
        pipe_table = grid_table_to_pipe(table_lines)
        lines[start:end + 1] = pipe_table.split('\n')
    
    return '\n'.join(lines), len(tables)


def process_file(filepath: Path, check_only: bool = False) -> int:
    """Process a single file. Returns number of issues/conversions."""
    content = filepath.read_text()
    
    if check_only:
        issues = check_grid_tables(content)
        if issues:
            print(f"{filepath}:")
            for line_num, desc in issues:
                print(f"  Line {line_num}: {desc}")
        return len(issues)
    else:
        converted, count = convert_grid_tables(content)
        if count > 0:
            filepath.write_text(converted)
            print(f"Converted {count} grid table(s) in {filepath}")
        return count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert grid tables to pipe tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check for grid tables (CI/validation):
  python convert_grid_to_pipe_tables.py --check book/quarto/contents/

  # Convert all grid tables:
  python convert_grid_to_pipe_tables.py book/quarto/contents/vol1/ml_systems/ml_systems.qmd
"""
    )
    parser.add_argument('paths', nargs='*', default=[],
                        help='Files or directories to process')
    parser.add_argument('--check', '-c', action='store_true',
                        help='Check only, do not convert (exit 1 if tables found)')
    
    args = parser.parse_args()
    
    if not args.paths:
        parser.print_help()
        return 0
    
    total = 0
    
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.qmd':
            total += process_file(path, args.check)
        elif path.is_dir():
            for qmd_file in path.rglob('*.qmd'):
                total += process_file(qmd_file, args.check)
    
    if total > 0:
        if args.check:
            print(f"\n❌ Found {total} grid table(s). Run without --check to convert.")
        else:
            print(f"\n✓ Converted {total} grid table(s) to pipe format.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
