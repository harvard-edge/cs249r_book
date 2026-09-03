#!/usr/bin/env python3
"""
Prettify pipe-style markdown tables in Quarto/Markdown files.

Aligns columns for readability while preserving pipe table format.

Before:
| **Layer Type** | **Output Shape** | **Parameters** |
|:--|:--|--:|
| **Linear** | $(B, N_{out})$ | $(N_{in} + 1) \times N_{out}$ |

After:
| **Layer Type** | **Output Shape** | **Parameters**                  |
|:---------------|:-----------------|--------------------------------:|
| **Linear**     | $(B, N_{out})$   | $(N_{in} + 1) \times N_{out}$   |

Usage:
  # Check mode (for pre-commit --check or CI):
  python prettify_pipe_tables.py --check file1.qmd file2.qmd

  # Fix mode (auto-format):
  python prettify_pipe_tables.py file1.qmd file2.qmd

  # Process entire directory:
  python prettify_pipe_tables.py book/quarto/contents/
"""

import re
import sys
from pathlib import Path


def is_separator_row(line: str) -> bool:
    """Check if a line is a table separator row (|---|---|)."""
    stripped = line.strip()
    if not stripped.startswith('|') or not stripped.endswith('|'):
        return False
    # Remove pipes and check if only dashes, colons, and spaces remain
    content = stripped[1:-1]  # Remove outer pipes
    cells = content.split('|')
    for cell in cells:
        cell = cell.strip()
        if not cell:
            continue
        # Valid separator cell: optional colons around dashes
        if not re.match(r'^:?-+:?$', cell):
            return False
    return True


def parse_alignment(separator_line: str) -> list[str]:
    """Extract alignment markers from separator row."""
    cells = separator_line.strip().strip('|').split('|')
    alignments = []
    for cell in cells:
        cell = cell.strip()
        if cell.startswith(':') and cell.endswith(':'):
            alignments.append('center')
        elif cell.endswith(':'):
            alignments.append('right')
        elif cell.startswith(':'):
            alignments.append('left')
        else:
            alignments.append('default')
    return alignments


def make_separator_cell(alignment: str, width: int) -> str:
    """Create a separator cell with proper alignment markers."""
    # Minimum 3 dashes for valid markdown
    dash_width = max(3, width)
    if alignment == 'center':
        return ':' + '-' * (dash_width - 2) + ':'
    elif alignment == 'right':
        return '-' * (dash_width - 1) + ':'
    elif alignment == 'left':
        return ':' + '-' * (dash_width - 1)
    else:  # default
        return '-' * dash_width


def parse_table_row(line: str) -> list[str]:
    """Parse a table row into cells, handling edge cases."""
    stripped = line.strip()
    if not stripped.startswith('|'):
        return []
    
    # Remove leading/trailing pipes
    content = stripped[1:]  # Remove leading pipe
    if content.endswith('|'):
        content = content[:-1]  # Remove trailing pipe
    
    # Split by | but be careful with escaped pipes or pipes in code
    # For simplicity, we do basic split (works for most tables)
    cells = content.split('|')
    return [cell.strip() for cell in cells]


def prettify_table(table_lines: list[str]) -> list[str]:
    """
    Prettify a pipe table by aligning columns.
    
    Returns the prettified table lines.
    """
    if len(table_lines) < 2:
        return table_lines
    
    # Find separator row (usually line 2, index 1)
    separator_idx = None
    for i, line in enumerate(table_lines):
        if is_separator_row(line):
            separator_idx = i
            break
    
    if separator_idx is None:
        return table_lines  # Not a valid pipe table
    
    # Parse alignments from separator
    alignments = parse_alignment(table_lines[separator_idx])
    
    # Parse all rows (excluding separator)
    rows = []
    for i, line in enumerate(table_lines):
        if i == separator_idx:
            continue
        cells = parse_table_row(line)
        if cells:
            rows.append(cells)
    
    if not rows:
        return table_lines
    
    # Determine number of columns (max across all rows)
    num_cols = max(len(row) for row in rows)
    num_cols = max(num_cols, len(alignments))
    
    # Pad rows and alignments to have consistent columns
    for row in rows:
        while len(row) < num_cols:
            row.append('')
    while len(alignments) < num_cols:
        alignments.append('default')
    
    # Calculate column widths (max content width per column)
    col_widths = []
    for col_idx in range(num_cols):
        max_width = 3  # Minimum width for separator dashes
        for row in rows:
            if col_idx < len(row):
                max_width = max(max_width, len(row[col_idx]))
        col_widths.append(max_width)
    
    # Build prettified table
    result = []
    row_idx = 0
    
    for i, line in enumerate(table_lines):
        if i == separator_idx:
            # Build separator row
            # Data rows are rendered with one space on each side of cell
            # content, so size separator cells to the same visual width.
            sep_cells = [make_separator_cell(alignments[j], col_widths[j] + 2)
                        for j in range(num_cols)]
            # Keep separator compact to avoid unnecessary spacing churn.
            result.append('|' + '|'.join(sep_cells) + '|')
        else:
            # Build data row with padding
            if row_idx < len(rows):
                row = rows[row_idx]
                padded_cells = []
                for j in range(num_cols):
                    cell = row[j] if j < len(row) else ''
                    # Pad based on alignment
                    if alignments[j] == 'right':
                        padded_cells.append(' ' + cell.rjust(col_widths[j]) + ' ')
                    elif alignments[j] == 'center':
                        padded_cells.append(' ' + cell.center(col_widths[j]) + ' ')
                    else:  # left or default
                        padded_cells.append(' ' + cell.ljust(col_widths[j]) + ' ')
                result.append('|' + '|'.join(padded_cells) + '|')
                row_idx += 1
    
    return result


def find_pipe_tables(content: str) -> list[tuple[int, int, list[str]]]:
    """
    Find all pipe tables in content, skipping code blocks.
    
    Returns: list of (start_line, end_line, table_lines)
    """
    lines = content.split('\n')
    tables = []
    i = 0
    in_code_block = False
    code_fence_pattern = re.compile(r'^(`{3,}|~{3,})')
    
    while i < len(lines):
        line = lines[i]
        
        # Track code blocks to skip tables inside them
        fence_match = code_fence_pattern.match(line.strip())
        if fence_match:
            in_code_block = not in_code_block
            i += 1
            continue
        
        if in_code_block:
            i += 1
            continue
        
        # Check if this starts a pipe table
        stripped = line.strip()
        if stripped.startswith('|') and '|' in stripped[1:]:
            start = i
            table_lines = [line]
            i += 1
            
            # Collect consecutive table lines
            while i < len(lines):
                current = lines[i].strip()
                # Continue if it's a pipe table row
                if current.startswith('|') and current.endswith('|'):
                    table_lines.append(lines[i])
                    i += 1
                else:
                    break
            
            # Valid table needs at least header + separator + 1 data row
            # and must have a separator row
            if len(table_lines) >= 2:
                has_separator = any(is_separator_row(tl) for tl in table_lines)
                if has_separator:
                    tables.append((start, start + len(table_lines) - 1, table_lines))
        else:
            i += 1
    
    return tables


def tables_are_equal(original: list[str], prettified: list[str]) -> bool:
    """Check if two tables are textually identical for formatting purposes."""
    if len(original) != len(prettified):
        return False
    for orig, pretty in zip(original, prettified):
        # Ignore trailing whitespace-only differences, but keep internal
        # spacing significant so alignment changes are detected.
        if orig.rstrip() != pretty.rstrip():
            return False
    return True


def check_tables(content: str) -> list[tuple[int, str]]:
    """
    Check for unprettified tables in content.
    
    Returns: list of (line_number, description) for each table needing formatting
    """
    tables = find_pipe_tables(content)
    issues = []
    
    for start, end, table_lines in tables:
        prettified = prettify_table(table_lines)
        if not tables_are_equal(table_lines, prettified):
            # Get first data line for context
            preview = table_lines[0][:50] + '...' if len(table_lines[0]) > 50 else table_lines[0]
            issues.append((start + 1, f"Table needs formatting ({end - start + 1} lines): {preview}"))
    
    return issues


def prettify_all_tables(content: str) -> tuple[str, int]:
    """
    Prettify all pipe tables in content.
    
    Returns: (prettified_content, number_of_tables_modified)
    """
    tables = find_pipe_tables(content)
    
    if not tables:
        return content, 0
    
    lines = content.split('\n')
    modified_count = 0
    
    # Process tables in reverse order to preserve line numbers
    for start, end, table_lines in reversed(tables):
        prettified = prettify_table(table_lines)
        if not tables_are_equal(table_lines, prettified):
            lines[start:end + 1] = prettified
            modified_count += 1
    
    return '\n'.join(lines), modified_count


def process_file(filepath: Path, check_only: bool = False) -> int:
    """Process a single file. Returns number of issues/modifications."""
    content = filepath.read_text()
    
    if check_only:
        issues = check_tables(content)
        if issues:
            print(f"{filepath}:")
            for line_num, desc in issues:
                print(f"  Line {line_num}: {desc}")
        return len(issues)
    else:
        prettified, count = prettify_all_tables(content)
        if count > 0:
            filepath.write_text(prettified)
            print(f"Prettified {count} table(s) in {filepath}")
        return count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prettify pipe-style markdown tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check for unprettified tables (CI/pre-commit):
  python prettify_pipe_tables.py --check book/quarto/contents/

  # Prettify all tables in a file:
  python prettify_pipe_tables.py book/quarto/contents/vol1/ml_systems/ml_systems.qmd

  # Prettify all tables in a directory:
  python prettify_pipe_tables.py book/quarto/contents/
"""
    )
    parser.add_argument('paths', nargs='*', default=[],
                        help='Files or directories to process')
    parser.add_argument('--check', '-c', action='store_true',
                        help='Check only, do not modify (exit 1 if formatting needed)')
    
    args = parser.parse_args()
    
    if not args.paths:
        parser.print_help()
        return 0
    
    total = 0
    
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix in ('.qmd', '.md'):
            total += process_file(path, args.check)
        elif path.is_dir():
            for qmd_file in path.rglob('*.qmd'):
                total += process_file(qmd_file, args.check)
            for md_file in path.rglob('*.md'):
                total += process_file(md_file, args.check)
    
    if args.check and total > 0:
        print(f"\n❌ Found {total} table(s) needing formatting. Run without --check to fix.")
        return 1
    elif total > 0:
        print(f"\n✓ Prettified {total} table(s).")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
