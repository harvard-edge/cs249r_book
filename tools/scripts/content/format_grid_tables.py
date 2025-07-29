#!/usr/bin/env python3
"""
Grid Table Formatter for MLSysBook

This script finds and reformats Pandoc grid tables in .qmd files with appropriate
column alignment based on content type:
- Left align: Text descriptions, names, use cases
- Center align: Short categorical values (High/Low, bit-widths, etc.)
- Right align: Numerical data and comparisons (2×, 4×, percentages)
"""

import re
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict


class GridTableFormatter:
    def __init__(self):
        # Patterns to identify content types
        self.numeric_patterns = [
            r'^\d+[×x]\s*',  # 2×, 4×, etc.
            r'^\d+\.\d+[×x]\s*',  # 2.5×, etc.
            r'^\d+%$',  # percentages
            r'^\d+(\.\d+)?$',  # pure numbers
            r'^[\d.,]+\s*(ms|MB|GB|KB|seconds?|minutes?|hours?)$',  # measurements
            r'^[\d.,]+\s*bit$',  # bit measurements
            r'^\$[\d.,]+$',  # currency
            r'^[\d.,]+\s*W$',  # watts
        ]
        
        self.categorical_patterns = [
            r'^(High|Low|Medium|Very High|Very Low|Minimal|Moderate|Extreme)$',
            r'^(Yes|No|True|False|On|Off|Enabled|Disabled)$',
            r'^(Fast|Slow|Faster|Slower|Fastest|Slowest)$',
            r'^\d+-?bit$',  # 32-bit, 16-bit, etc.
            r'^[A-Z]{2,6}$',  # Short acronyms like FP32, INT8
        ]

    def detect_alignment(self, column_values: List[str]) -> str:
        """
        Detect appropriate alignment for a column based on its content.
        Returns 'left', 'center', or 'right'
        """
        if not column_values:
            return 'left'
        
        # Remove empty values and strip whitespace
        non_empty_values = [v.strip() for v in column_values if v.strip()]
        if not non_empty_values:
            return 'left'
        
        # Count matches for each type
        numeric_matches = 0
        categorical_matches = 0
        
        for value in non_empty_values:
            # Check for numeric patterns
            if any(re.match(pattern, value, re.IGNORECASE) for pattern in self.numeric_patterns):
                numeric_matches += 1
            # Check for categorical patterns
            elif any(re.match(pattern, value, re.IGNORECASE) for pattern in self.categorical_patterns):
                categorical_matches += 1
        
        total_values = len(non_empty_values)
        numeric_ratio = numeric_matches / total_values
        categorical_ratio = categorical_matches / total_values
        
        # Decision logic
        if numeric_ratio >= 0.6:  # 60% or more numeric -> right align
            return 'right'
        elif categorical_ratio >= 0.6:  # 60% or more categorical -> center align
            return 'center'
        else:
            # Check average length - short columns often categorical
            avg_length = sum(len(v) for v in non_empty_values) / total_values
            if avg_length <= 8 and categorical_ratio >= 0.3:
                return 'center'
            else:
                return 'left'  # Default to left for descriptive text

    def parse_grid_table(self, table_text: str) -> Tuple[List[List[str]], List[str]]:
        """
        Parse a grid table and return the data and current alignments.
        """
        lines = table_text.strip().split('\n')
        
        # Find header separator line (contains =)
        header_sep_idx = -1
        for i, line in enumerate(lines):
            if '=' in line and '+' in line:
                header_sep_idx = i
                break
        
        if header_sep_idx == -1:
            raise ValueError("Could not find header separator in grid table")
        
        # Extract current alignments from header separator
        header_sep = lines[header_sep_idx]
        alignments = self.extract_alignments(header_sep)
        
        # Find all row separators (lines with + and -)
        row_separators = []
        for i, line in enumerate(lines):
            if '+' in line and '-' in line:
                row_separators.append(i)
        
        # Parse header row (between first separator and header separator)
        rows = []
        if len(row_separators) > 0:
            # Header row
            header_content = []
            for line_idx in range(row_separators[0] + 1, header_sep_idx):
                line = lines[line_idx].strip()
                if line.startswith('|') and line.endswith('|'):
                    header_content.append(line)
            
            if header_content:
                header_cells = []
                first_line_cells = [cell.strip() for cell in header_content[0].split('|')[1:-1]]
                header_cells = first_line_cells[:]
                
                # Merge multi-line header content
                for line in header_content[1:]:
                    line_cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    for j, cell_content in enumerate(line_cells):
                        if j < len(header_cells) and cell_content:
                            if header_cells[j]:
                                header_cells[j] += ' ' + cell_content
                            else:
                                header_cells[j] = cell_content
                
                if header_cells:
                    rows.append(header_cells)
        
        # Parse data rows (between row separators, excluding header section)
        for i in range(len(row_separators) - 1):
            start_sep = row_separators[i]
            end_sep = row_separators[i + 1]
            
            # Skip if this is the header section (before header separator)
            if end_sep <= header_sep_idx:
                continue
                
            # Collect content lines between separators
            content_lines = []
            for line_idx in range(start_sep + 1, end_sep):
                line = lines[line_idx].strip()
                if line.startswith('|') and line.endswith('|'):
                    content_lines.append(line)
            
            if content_lines:
                # Parse the cell content across multiple lines
                row_cells = []
                
                # Split first line to get number of columns
                first_line_cells = [cell.strip() for cell in content_lines[0].split('|')[1:-1]]
                row_cells = first_line_cells[:]
                
                # Merge multi-line content
                for line in content_lines[1:]:
                    line_cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    for j, cell_content in enumerate(line_cells):
                        if j < len(row_cells) and cell_content:
                            if row_cells[j]:
                                row_cells[j] += ' ' + cell_content
                            else:
                                row_cells[j] = cell_content
                
                if row_cells:
                    rows.append(row_cells)
        
        return rows, alignments

    def extract_alignments(self, separator_line: str) -> List[str]:
        """Extract current column alignments from separator line."""
        # Split by + to get column parts
        parts = separator_line.split('+')[1:-1]  # Remove first and last empty
        alignments = []
        
        for part in parts:
            if part.startswith(':') and part.endswith(':'):
                alignments.append('center')
            elif part.startswith(':'):
                alignments.append('left')
            elif part.endswith(':'):
                alignments.append('right')
            else:
                alignments.append('left')  # Default
        
        return alignments

    def create_alignment_string(self, alignment: str, width: int) -> str:
        """Create alignment string for grid table separator."""
        if alignment == 'center':
            return ':' + '=' * (width - 2) + ':'
        elif alignment == 'right':
            return '=' * (width - 1) + ':'
        else:  # left
            return ':' + '=' * (width - 1)

    def format_grid_table(self, table_text: str) -> str:
        """
        Reformat a grid table with improved alignment.
        """
        try:
            rows, current_alignments = self.parse_grid_table(table_text)
            
            if not rows:
                return table_text
            
            # Determine optimal alignments based on content
            num_cols = len(rows[0])
            new_alignments = []
            
            for col_idx in range(num_cols):
                column_values = [row[col_idx] if col_idx < len(row) else '' for row in rows]
                alignment = self.detect_alignment(column_values)
                new_alignments.append(alignment)
            
            # Calculate column widths
            col_widths = []
            for col_idx in range(num_cols):
                max_width = 0
                for row in rows:
                    if col_idx < len(row):
                        max_width = max(max_width, len(row[col_idx].strip()))
                col_widths.append(max(max_width + 2, 4))  # Minimum width of 4
            
            # Build the formatted table
            result_lines = []
            
            # Top border
            border_line = '+'
            for width in col_widths:
                border_line += '-' * width + '+'
            result_lines.append(border_line)
            
            # Format each row
            for row_idx, row in enumerate(rows):
                # Content line
                data_line = '|'
                for col_idx, cell in enumerate(row):
                    if col_idx < len(col_widths):
                        cell_content = cell.strip()
                        cell_width = col_widths[col_idx] - 2  # Account for spaces
                        alignment = new_alignments[col_idx] if col_idx < len(new_alignments) else 'left'
                        
                        # Apply alignment
                        if alignment == 'center':
                            formatted_cell = f' {cell_content:^{cell_width}} '
                        elif alignment == 'right':
                            formatted_cell = f' {cell_content:>{cell_width}} '
                        else:  # left
                            formatted_cell = f' {cell_content:<{cell_width}} '
                        
                        data_line += formatted_cell + '|'
                result_lines.append(data_line)
                
                # Add separator after header row (index 0)
                if row_idx == 0:
                    # Header separator with alignments
                    sep_line = '+'
                    for col_idx, width in enumerate(col_widths):
                        if col_idx < len(new_alignments):
                            alignment_str = self.create_alignment_string(new_alignments[col_idx], width)
                            sep_line += alignment_str + '+'
                        else:
                            sep_line += ':' + '=' * (width - 1) + '+'
                    result_lines.append(sep_line)
                elif row_idx < len(rows) - 1:
                    # Regular row separator
                    result_lines.append(border_line)
            
            # Bottom border
            result_lines.append(border_line)
            
            return '\n'.join(result_lines)
            
        except Exception as e:
            print(f"Error formatting table: {e}")
            return table_text

    def find_grid_tables(self, content: str) -> List[Tuple[int, int, str]]:
        """
        Find all grid tables in content.
        Returns list of (start_pos, end_pos, table_text) tuples.
        """
        tables = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for table start (line with + and -)
            if line and '+' in line and '-' in line:
                table_start = i
                table_lines = [lines[i]]
                i += 1
                
                # Collect all table lines
                while i < len(lines):
                    line = lines[i].strip()
                    if not line:
                        # Empty line might be end of table, but check next line
                        if i + 1 < len(lines) and lines[i + 1].strip():
                            next_line = lines[i + 1].strip()
                            if not ('+' in next_line or '|' in next_line):
                                break
                        table_lines.append(lines[i])
                    elif '+' in line or '|' in line:
                        table_lines.append(lines[i])
                    else:
                        break
                    i += 1
                
                table_end = i
                table_text = '\n'.join(table_lines)
                
                # Verify it's actually a grid table (has header separator)
                if '=' in table_text and table_text.count('+') >= 4:
                    start_pos = sum(len(lines[j]) + 1 for j in range(table_start))
                    end_pos = sum(len(lines[j]) + 1 for j in range(table_end))
                    tables.append((start_pos, end_pos, table_text))
                
                continue
            
            i += 1
        
        return tables

    def process_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """
        Process a single file and format its grid tables.
        Returns True if any changes were made.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tables = self.find_grid_tables(content)
            if not tables:
                return False
            
            print(f"Found {len(tables)} grid table(s) in {file_path}")
            
            # Process tables in reverse order to maintain positions
            modified_content = content
            changes_made = False
            
            for start_pos, end_pos, table_text in reversed(tables):
                formatted_table = self.format_grid_table(table_text)
                if formatted_table != table_text:
                    if not dry_run:
                        modified_content = (modified_content[:start_pos] + 
                                          formatted_table + 
                                          modified_content[end_pos:])
                    changes_made = True
                    print(f"  ✅ Formatted table at position {start_pos}")
                else:
                    print(f"  ℹ️  Table at position {start_pos} already properly formatted")
            
            if changes_made and not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                print(f"✅ Updated {file_path}")
            
            return changes_made
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False

    def process_directory(self, directory: Path, dry_run: bool = False) -> int:
        """
        Process all .qmd files in a directory recursively.
        Returns the number of files that were modified.
        """
        modified_count = 0
        
        for qmd_file in directory.rglob("*.qmd"):
            if self.process_file(qmd_file, dry_run):
                modified_count += 1
        
        return modified_count


def main():
    parser = argparse.ArgumentParser(
        description="Format grid tables in markdown files with appropriate column alignment"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to file or directory to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making modifications"
    )
    
    args = parser.parse_args()
    
    formatter = GridTableFormatter()
    
    if args.path.is_file():
        if args.path.suffix == '.qmd':
            changed = formatter.process_file(args.path, args.dry_run)
            if args.dry_run:
                print(f"\n🔍 Dry run complete. File {'would be' if changed else 'would not be'} modified.")
            else:
                print(f"\n✅ Processing complete. File {'was' if changed else 'was not'} modified.")
        else:
            print("❌ Please provide a .qmd file")
            return 1
    elif args.path.is_dir():
        modified_count = formatter.process_directory(args.path, args.dry_run)
        if args.dry_run:
            print(f"\n🔍 Dry run complete. {modified_count} file(s) would be modified.")
        else:
            print(f"\n✅ Processing complete. {modified_count} file(s) were modified.")
    else:
        print(f"❌ Path not found: {args.path}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 