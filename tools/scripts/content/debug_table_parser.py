#!/usr/bin/env python3
"""
Debug script to understand table parsing issues
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from format_grid_tables import GridTableFormatter

def debug_table_parsing():
    """Debug the table parsing with a simple example."""
    
    table = """
+------------------+-----------+------------------+
| Technique        | Bit-Width | Storage Reduction|
+:=================+==========:+=================:+
| FP32             |    32-bit |        Baseline  |
+------------------+-----------+------------------+
| FP16             |    16-bit |             2×   |
+------------------+-----------+------------------+
| INT8             |     8-bit |             4×   |
+------------------+-----------+------------------+
""".strip()
    
    print("Original table:")
    print(table)
    print("\n" + "="*50 + "\n")
    
    formatter = GridTableFormatter()
    
    # Split into lines and analyze
    lines = [line.rstrip() for line in table.split('\n') if line.strip()]
    print("Lines:")
    for i, line in enumerate(lines):
        print(f"{i:2d}: {repr(line)}")
    
    print("\n" + "="*50 + "\n")
    
    # Find header separator
    header_sep_idx = -1
    for i, line in enumerate(lines):
        if '+' in line and '=' in line:
            header_sep_idx = i
            print(f"Header separator found at line {i}: {repr(line)}")
            break
    
    # Find border lines
    border_lines = []
    for i, line in enumerate(lines):
        if '+' in line and ('-' in line or '=' in line):
            border_lines.append(i)
            print(f"Border line {i}: {repr(line)}")
    
    print(f"\nBorder lines: {border_lines}")
    print(f"Header separator index: {header_sep_idx}")
    
    try:
        rows, alignments = formatter._parse_table(table)
        print(f"\nParsed {len(rows)} rows:")
        for i, row in enumerate(rows):
            print(f"  Row {i}: {row}")
        print(f"Alignments: {alignments}")
    except Exception as e:
        print(f"\nParsing failed: {e}")

if __name__ == "__main__":
    debug_table_parsing() 