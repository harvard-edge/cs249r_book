#!/usr/bin/env python3
"""
Grid Table Formatter for MLSysBook

A robust tool for formatting Pandoc grid tables in .qmd files with intelligent
content-based alignment. Suitable for use in pre-commit hooks and CI/CD pipelines.

Usage:
    python3 format_grid_tables.py -f file.qmd [--dry-run]
    python3 format_grid_tables.py -d directory/ [--dry-run]
    python3 format_grid_tables.py --check-all [--dry-run]
"""

import re
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TableInfo:
    """Information about a detected table."""
    start_pos: int
    end_pos: int
    content: str
    file_path: Path
    table_type: str
    confidence: float


class GridTableAnalyzer:
    """Analyzes table content to determine optimal formatting."""
    
    def __init__(self):
        # Enhanced patterns for content type detection
        self.numeric_patterns = [
            (r'^\d+[×x]\s*', 'multiplication'),  # 2×, 4×
            (r'^\d+\.\d+[×x]\s*', 'decimal_multiplication'),  # 2.5×
            (r'^\d+%$', 'percentage'),  # 50%
            (r'^\d+(\.\d+)?$', 'pure_number'),  # 42, 3.14
            (r'^[\d.,]+\s*(ms|MB|GB|KB|TB|bit|W|Hz|MHz|GHz)$', 'measurement'),
            (r'^\$[\d.,]+$', 'currency'),  # $1,000
            (r'^[\d.,]+\s*(seconds?|minutes?|hours?)$', 'time'),
            (r'^Baseline\s*\(1×\)$', 'baseline'),  # Baseline (1×)
            (r'^\d+–\d+×', 'range_multiplication'),  # 4–8×
        ]
        
        self.categorical_patterns = [
            (r'^(High|Low|Medium|Very High|Very Low|Minimal|Moderate|Extreme)$', 'level'),
            (r'^(Yes|No|True|False|On|Off|Enabled|Disabled)$', 'boolean'),
            (r'^(Fast|Slow|Faster|Slower|Fastest|Slowest)$', 'speed'),
            (r'^\d+-?bit$', 'bit_width'),  # 32-bit, 16-bit
            (r'^[A-Z]{2,6}\d*$', 'acronym'),  # FP32, INT8, TPU
            (r'^(CPU|GPU|TPU|NPU|FPGA)$', 'hardware'),
            (r'^(Training|Inference|Both)$', 'usage_type'),
        ]
        
        self.descriptive_patterns = [
            (r'^[A-Z][a-z].*', 'description'),  # Starts with capital
            (r'.*\s(learning|training|inference|optimization).*', 'ml_description'),
            (r'.*\s(device|system|platform|framework).*', 'system_description'),
        ]

    def analyze_column_content(self, column_values: List[str]) -> Dict[str, any]:
        """Analyze a column's content to determine its characteristics."""
        non_empty = [v.strip() for v in column_values if v.strip()]
        if not non_empty:
            return {'type': 'empty', 'alignment': 'left', 'confidence': 0.0}
        
        total = len(non_empty)
        type_counts = {
            'numeric': 0,
            'categorical': 0,
            'descriptive': 0
        }
        
        # Count matches for each pattern type
        for value in non_empty:
            # Check numeric patterns
            for pattern, subtype in self.numeric_patterns:
                if re.match(pattern, value, re.IGNORECASE):
                    type_counts['numeric'] += 1
                    break
            else:
                # Check categorical patterns
                for pattern, subtype in self.categorical_patterns:
                    if re.match(pattern, value, re.IGNORECASE):
                        type_counts['categorical'] += 1
                        break
                else:
                    # Check descriptive patterns
                    for pattern, subtype in self.descriptive_patterns:
                        if re.match(pattern, value, re.IGNORECASE):
                            type_counts['descriptive'] += 1
                            break
                    else:
                        # Default to descriptive for unmatched content
                        type_counts['descriptive'] += 1
        
        # Determine primary type and confidence
        max_type = max(type_counts, key=type_counts.get)
        confidence = type_counts[max_type] / total
        
        # Determine alignment based on type and confidence
        if max_type == 'numeric' and confidence >= 0.6:
            alignment = 'right'
        elif max_type == 'categorical' and confidence >= 0.6:
            alignment = 'center'
        elif max_type == 'categorical' and confidence >= 0.3:
            # Check average length for borderline categorical
            avg_length = sum(len(v) for v in non_empty) / total
            alignment = 'center' if avg_length <= 10 else 'left'
        else:
            alignment = 'left'
        
        return {
            'type': max_type,
            'alignment': alignment,
            'confidence': confidence,
            'type_counts': type_counts,
            'avg_length': sum(len(v) for v in non_empty) / total
        }

    def classify_table(self, rows: List[List[str]]) -> str:
        """Classify the overall table type based on content."""
        if not rows:
            return 'unknown'
        
        # Look for common table patterns
        header = rows[0] if rows else []
        header_text = ' '.join(header).lower()
        
        if any(term in header_text for term in ['precision', 'format', 'bit']):
            return 'precision_comparison'
        elif any(term in header_text for term in ['model', 'architecture', 'network']):
            return 'model_comparison'
        elif any(term in header_text for term in ['performance', 'benchmark', 'latency']):
            return 'performance_table'
        elif any(term in header_text for term in ['technique', 'method', 'approach']):
            return 'technique_comparison'
        elif any(term in header_text for term in ['requirement', 'overhead', 'cost']):
            return 'requirements_table'
        else:
            return 'general_table'


class GridTableFormatter:
    """Formats grid tables with intelligent alignment."""
    
    def __init__(self):
        self.analyzer = GridTableAnalyzer()
    
    def find_grid_tables(self, content: str, file_path: Path) -> List[TableInfo]:
        """Find all grid tables in content using robust pattern matching."""
        tables = []
        
        # Split content into lines for analysis
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for table start (line with + and -)
            if self._is_table_border(line):
                table_start_line = i
                table_lines = [lines[i]]
                i += 1
                
                # Collect table content
                in_table = True
                has_header_sep = False
                
                while i < len(lines) and in_table:
                    line = lines[i].strip()
                    
                    if not line:
                        # Empty line - check if table continues
                        if i + 1 < len(lines) and self._is_table_line(lines[i + 1].strip()):
                            table_lines.append(lines[i])
                        else:
                            in_table = False
                            break
                    elif self._is_table_line(line):
                        table_lines.append(lines[i])
                        if '=' in line:
                            has_header_sep = True
                    else:
                        in_table = False
                        break
                    
                    i += 1
                
                # Validate as grid table
                if has_header_sep and len(table_lines) >= 3:
                    table_content = '\n'.join(table_lines)
                    start_pos = sum(len(lines[j]) + 1 for j in range(table_start_line))
                    end_pos = sum(len(lines[j]) + 1 for j in range(min(i, len(lines))))
                    
                    # Analyze table
                    try:
                        rows, _ = self._parse_table(table_content)
                        table_type = self.analyzer.classify_table(rows)
                        confidence = self._calculate_confidence(rows)
                        
                        tables.append(TableInfo(
                            start_pos=start_pos,
                            end_pos=end_pos,
                            content=table_content,
                            file_path=file_path,
                            table_type=table_type,
                            confidence=confidence
                        ))
                    except Exception as e:
                        print(f"Warning: Could not parse table at line {table_start_line}: {e}")
                
                continue
            
            i += 1
        
        return tables
    
    def _is_table_border(self, line: str) -> bool:
        """Check if line is a table border."""
        return '+' in line and '-' in line and len(line) > 5
    
    def _is_table_line(self, line: str) -> bool:
        """Check if line belongs to a table."""
        return ('+' in line) or (line.startswith('|') and line.endswith('|'))
    
    def _parse_table(self, table_text: str) -> Tuple[List[List[str]], List[str]]:
        """Parse table content into rows and extract current alignments."""
        lines = [line.rstrip() for line in table_text.split('\n') if line.strip()]
        
        # Find header separator
        header_sep_idx = -1
        for i, line in enumerate(lines):
            if '+' in line and '=' in line:
                header_sep_idx = i
                break
        
        if header_sep_idx == -1:
            raise ValueError("No header separator found")
        
        # Extract alignments
        alignments = self._extract_alignments(lines[header_sep_idx])
        
        # Extract data rows
        data_lines = []
        for line in lines:
            if line.startswith('|') and line.endswith('|') and '=' not in line and '+' not in line:
                data_lines.append(line)
        
        if not data_lines:
            raise ValueError("No data rows found")
        
        # Parse cells
        rows = []
        for line in data_lines:
            cells = [cell.strip() for cell in line[1:-1].split('|')]
            if cells:
                rows.append(cells)
        
        return rows, alignments
    
    def _extract_alignments(self, sep_line: str) -> List[str]:
        """Extract current alignments from header separator."""
        segments = sep_line.split('+')[1:-1]
        alignments = []
        
        for segment in segments:
            segment = segment.strip()
            if segment.startswith(':') and segment.endswith(':'):
                alignments.append('center')
            elif segment.startswith(':'):
                alignments.append('left')
            elif segment.endswith(':'):
                alignments.append('right')
            else:
                alignments.append('left')
        
        return alignments
    
    def _calculate_confidence(self, rows: List[List[str]]) -> float:
        """Calculate confidence in table parsing."""
        if not rows:
            return 0.0
        
        # Check consistency
        row_lengths = [len(row) for row in rows]
        if len(set(row_lengths)) > 1:
            return 0.5  # Inconsistent row lengths
        
        return 0.9  # High confidence for consistent tables
    
    def format_table(self, table_info: TableInfo) -> str:
        """Format a table with optimal alignment."""
        try:
            rows, current_alignments = self._parse_table(table_info.content)
            
            if not rows:
                return table_info.content
            
            # Analyze each column
            num_cols = len(rows[0]) if rows else 0
            column_analyses = []
            new_alignments = []
            
            for col_idx in range(num_cols):
                column_values = [row[col_idx] if col_idx < len(row) else '' 
                               for row in rows[1:]]  # Skip header for analysis
                analysis = self.analyzer.analyze_column_content(column_values)
                column_analyses.append(analysis)
                new_alignments.append(analysis['alignment'])
            
            # Calculate optimal column widths
            col_widths = []
            for col_idx in range(num_cols):
                max_width = 0
                for row in rows:
                    if col_idx < len(row):
                        max_width = max(max_width, len(row[col_idx]))
                col_widths.append(max(max_width + 2, 6))  # Minimum width
            
            # Generate formatted table
            return self._build_formatted_table(rows, new_alignments, col_widths)
            
        except Exception as e:
            print(f"Error formatting table in {table_info.file_path}: {e}")
            return table_info.content
    
    def _build_formatted_table(self, rows: List[List[str]], alignments: List[str], 
                              col_widths: List[int]) -> str:
        """Build the formatted table string."""
        result = []
        
        # Top border
        border = '+' + '+'.join('-' * w for w in col_widths) + '+'
        result.append(border)
        
        # Process each row
        for row_idx, row in enumerate(rows):
            # Format row content
            formatted_row = '|'
            for col_idx, cell in enumerate(row):
                if col_idx < len(col_widths):
                    width = col_widths[col_idx] - 2
                    alignment = alignments[col_idx] if col_idx < len(alignments) else 'left'
                    
                    if alignment == 'center':
                        formatted_cell = f' {cell:^{width}} '
                    elif alignment == 'right':
                        formatted_cell = f' {cell:>{width}} '
                    else:
                        formatted_cell = f' {cell:<{width}} '
                    
                    formatted_row += formatted_cell + '|'
            
            result.append(formatted_row)
            
            # Add separator after header
            if row_idx == 0:
                sep = '+'
                for col_idx, width in enumerate(col_widths):
                    alignment = alignments[col_idx] if col_idx < len(alignments) else 'left'
                    if alignment == 'center':
                        sep += ':' + '=' * (width - 2) + ':+'
                    elif alignment == 'right':
                        sep += '=' * (width - 1) + ':+'
                    else:
                        sep += ':' + '=' * (width - 1) + '+'
                result.append(sep)
            elif row_idx < len(rows) - 1:
                result.append(border)
        
        # Bottom border
        result.append(border)
        
        return '\n'.join(result)


def process_file(file_path: Path, dry_run: bool = False, verbose: bool = False) -> bool:
    """Process a single .qmd file."""
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    if file_path.suffix != '.qmd':
        if verbose:
            print(f"⏭️  Skipping non-qmd file: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        formatter = GridTableFormatter()
        tables = formatter.find_grid_tables(content, file_path)
        
        if not tables:
            if verbose:
                print(f"ℹ️  No tables found in {file_path}")
            return False
        
        print(f"📋 Found {len(tables)} table(s) in {file_path}")
        
        # Process tables in reverse order to maintain positions
        modified_content = content
        changes_made = False
        
        for table_info in reversed(tables):
            if verbose:
                print(f"   📊 {table_info.table_type} (confidence: {table_info.confidence:.1f})")
            
            formatted_table = formatter.format_table(table_info)
            if formatted_table != table_info.content:
                if not dry_run:
                    modified_content = (modified_content[:table_info.start_pos] + 
                                      formatted_table + 
                                      modified_content[table_info.end_pos:])
                changes_made = True
                print(f"   ✅ Formatted {table_info.table_type} table")
            else:
                if verbose:
                    print(f"   ✨ Table already optimally formatted")
        
        if changes_made:
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                print(f"💾 Updated {file_path}")
            else:
                print(f"🔍 Would update {file_path}")
        
        return changes_made
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def process_directory(directory: Path, dry_run: bool = False, verbose: bool = False) -> int:
    """Process all .qmd files in a directory."""
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return 0
    
    qmd_files = list(directory.rglob("*.qmd"))
    if not qmd_files:
        print(f"ℹ️  No .qmd files found in {directory}")
        return 0
    
    print(f"📁 Processing {len(qmd_files)} .qmd files in {directory}")
    
    modified_count = 0
    for qmd_file in sorted(qmd_files):
        if process_file(qmd_file, dry_run, verbose):
            modified_count += 1
    
    return modified_count


def main():
    parser = argparse.ArgumentParser(
        description="Format grid tables in Quarto markdown files with intelligent alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f book/contents/core/ml_systems.qmd --dry-run
  %(prog)s -d book/contents/ --verbose
  %(prog)s --check-all
        """
    )
    
    # Mutually exclusive group for input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--file', type=Path, 
                           help='Process a single .qmd file')
    input_group.add_argument('-d', '--directory', type=Path,
                           help='Process all .qmd files in directory (recursive)')
    input_group.add_argument('--check-all', action='store_true',
                           help='Process all .qmd files in book/contents/')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed processing information')
    parser.add_argument('--pre-commit', action='store_true',
                       help='Run in pre-commit mode (exit 1 if changes needed)')
    
    args = parser.parse_args()
    
    # Determine what to process
    if args.file:
        changed = process_file(args.file, args.dry_run, args.verbose)
        total_files = 1 if args.file.exists() and args.file.suffix == '.qmd' else 0
        modified_count = 1 if changed else 0
    elif args.directory:
        modified_count = process_directory(args.directory, args.dry_run, args.verbose)
        total_files = len(list(args.directory.rglob("*.qmd"))) if args.directory.exists() else 0
    elif args.check_all:
        book_contents = Path('book/contents')
        if not book_contents.exists():
            print("❌ book/contents directory not found")
            return 1
        modified_count = process_directory(book_contents, args.dry_run, args.verbose)
        total_files = len(list(book_contents.rglob("*.qmd")))
    
    # Summary
    if args.dry_run:
        print(f"\n🔍 Dry run complete: {modified_count}/{total_files} files would be modified")
    else:
        print(f"\n✅ Processing complete: {modified_count}/{total_files} files modified")
    
    # Pre-commit mode: exit with error if changes are needed
    if args.pre_commit and modified_count > 0:
        if args.dry_run:
            print("💡 Run without --dry-run to format tables")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 