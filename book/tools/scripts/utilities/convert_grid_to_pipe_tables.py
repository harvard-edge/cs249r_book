#!/usr/bin/env python3
"""
Convert grid tables to pipe tables in Quarto/Markdown files.

Grid tables:
+------+------+
| Col1 | Col2 |
+======+======+
| data | data |
+------+------+

Pipe tables:
| Col1 | Col2 |
|------|------|
| data | data |
"""

import re
import sys
from pathlib import Path


def parse_grid_table(lines: list[str]) -> tuple[list[str], list[list[str]], list[str]]:
    """Parse a grid table into headers, rows, and alignment info."""
    headers = []
    rows = []
    alignments = []
    
    in_header = True
    current_row = []
    
    for line in lines:
        line = line.rstrip()
        
        # Separator line (determines alignment)
        if line.startswith('+') and ('=' in line or '-' in line):
            # Parse alignment from separator
            if '=' in line:  # Header separator
                parts = re.split(r'\+', line)[1:-1]  # Remove empty first/last
                for part in parts:
                    if part.startswith(':') and part.endswith(':'):
                        alignments.append(':--:')
                    elif part.endswith(':'):
                        alignments.append('--:')
                    elif part.startswith(':'):
                        alignments.append(':--')
                    else:
                        alignments.append('--')
                in_header = False
            continue
        
        # Data line
        if line.startswith('|'):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if in_header:
                headers = cells
            else:
                rows.append(cells)
    
    # Default alignments if not found
    if not alignments and headers:
        alignments = ['--'] * len(headers)
    
    return headers, rows, alignments


def grid_table_to_pipe(table_text: str) -> str:
    """Convert a grid table to pipe table format."""
    lines = table_text.strip().split('\n')
    headers, rows, alignments = parse_grid_table(lines)
    
    if not headers:
        return table_text  # Return unchanged if parsing failed
    
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
        result.append('| ' + ' | '.join(row) + ' |')
    
    return '\n'.join(result)


def find_and_convert_tables(content: str) -> str:
    """Find all grid tables in content and convert to pipe tables."""
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this starts a grid table
        if re.match(r'^\+[-:=+]+\+$', line.strip()):
            # Collect the entire table
            table_lines = [line]
            i += 1
            
            while i < len(lines):
                current = lines[i]
                if re.match(r'^\+[-:=+]+\+$', current.strip()) or current.strip().startswith('|'):
                    table_lines.append(current)
                    # Check if this is the closing line
                    if re.match(r'^\+[-]+\+$', current.strip()) and len(table_lines) > 2:
                        # Peek ahead to see if table continues
                        if i + 1 < len(lines) and not lines[i + 1].strip().startswith('|'):
                            i += 1
                            break
                    i += 1
                else:
                    break
            
            # Convert the table
            table_text = '\n'.join(table_lines)
            pipe_table = grid_table_to_pipe(table_text)
            result.append(pipe_table)
        else:
            result.append(line)
            i += 1
    
    return '\n'.join(result)


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """Process a single file, converting grid tables to pipe tables."""
    content = filepath.read_text()
    
    # Check if file has grid tables
    if not re.search(r'^\+[-:=+]+\+$', content, re.MULTILINE):
        print(f"  No grid tables found in {filepath.name}")
        return False
    
    converted = find_and_convert_tables(content)
    
    if converted != content:
        if dry_run:
            print(f"  Would convert tables in {filepath.name}")
        else:
            filepath.write_text(converted)
            print(f"  Converted tables in {filepath.name}")
        return True
    
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert grid tables to pipe tables')
    parser.add_argument('files', nargs='+', help='Files to process')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Show what would be done')
    
    args = parser.parse_args()
    
    for filepath in args.files:
        path = Path(filepath)
        if path.exists():
            process_file(path, args.dry_run)
        else:
            print(f"  File not found: {filepath}")


if __name__ == '__main__':
    main()
