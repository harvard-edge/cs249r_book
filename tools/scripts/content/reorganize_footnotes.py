#!/usr/bin/env python3
"""
Footnote Reorganizer for QMD Files

This script reorganizes footnotes in Quarto Markdown (.qmd) files by moving
footnote definitions to immediately after the paragraphs where they are referenced.

The script:
1. Parses QMD files to find footnote references [^footnote-id] and definitions [^footnote-id]: content
2. Moves each footnote definition to appear right after the paragraph containing its reference
3. Removes the original batched footnote definitions
4. Preserves all other content and formatting
5. Only processes files that actually need reorganization (have batched footnotes)

Usage:
    python reorganize_footnotes.py <file_or_directory>
    python reorganize_footnotes.py --dry-run <file_or_directory>  # Preview changes only
    python reorganize_footnotes.py --backup <file_or_directory>   # Create .bak files

Author: MLSysBook Team
"""

import re
import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class FootnoteReorganizer:
    def __init__(self, dry_run: bool = False, create_backup: bool = False):
        self.dry_run = dry_run
        self.create_backup = create_backup
        
        # Regex patterns
        self.footnote_ref_pattern = re.compile(r'\[\^([^]]+)\](?!:)')  # [^fn-name] but not [^fn-name]:
        self.footnote_def_pattern = re.compile(r'^\[\^([^]]+)\]:\s*(.+)$', re.MULTILINE)
        
    def find_qmd_files(self, path: str) -> List[Path]:
        """Find all .qmd files in the given path."""
        path_obj = Path(path)
        
        if path_obj.is_file() and path_obj.suffix == '.qmd':
            return [path_obj]
        elif path_obj.is_dir():
            return list(path_obj.rglob('*.qmd'))
        else:
            print(f"Warning: {path} is not a valid file or directory")
            return []
    
    def parse_content(self, content: str) -> Tuple[List[str], Dict[str, str], Dict[str, List[int]]]:
        """
        Parse content to extract lines, footnote definitions, and reference locations.
        
        Returns:
            lines: List of content lines
            footnote_defs: Dict mapping footnote IDs to their definitions
            footnote_refs: Dict mapping footnote IDs to line numbers where they're referenced
        """
        lines = content.split('\n')
        footnote_defs = {}
        footnote_refs = {}
        
        # Find all footnote definitions
        for match in self.footnote_def_pattern.finditer(content):
            footnote_id = match.group(1)
            footnote_content = match.group(2)
            footnote_defs[footnote_id] = footnote_content
        
        # Find all footnote references and their line numbers
        for line_num, line in enumerate(lines):
            for match in self.footnote_ref_pattern.finditer(line):
                footnote_id = match.group(1)
                if footnote_id not in footnote_refs:
                    footnote_refs[footnote_id] = []
                footnote_refs[footnote_id].append(line_num)
        
        return lines, footnote_defs, footnote_refs
    
    def find_paragraph_end(self, lines: List[str], start_line: int) -> int:
        """
        Find the end of the paragraph containing the given line.
        A paragraph ends at an empty line, heading, or special block.
        """
        for i in range(start_line + 1, len(lines)):
            line = lines[i].strip()
            
            # Empty line ends paragraph
            if not line:
                return i - 1
            
            # Heading ends paragraph
            if line.startswith('#'):
                return i - 1
            
            # Special blocks end paragraph
            if line.startswith(':::') or line.startswith('```') or line.startswith('|'):
                return i - 1
            
            # Footnote definition ends paragraph
            if self.footnote_def_pattern.match(line):
                return i - 1
        
        # If we reach the end, return the last line
        return len(lines) - 1
    
    def needs_reorganization(self, lines: List[str], footnote_defs: Dict[str, str], footnote_refs: Dict[str, List[int]]) -> bool:
        """
        Check if the file needs footnote reorganization.
        Returns True if there are footnote definitions that are not immediately after their references.
        """
        if not footnote_defs or not footnote_refs:
            return False
        
        for footnote_id, def_content in footnote_defs.items():
            if footnote_id not in footnote_refs:
                continue
            
            # Find where this footnote is currently defined
            def_line = None
            for i, line in enumerate(lines):
                if line.startswith(f'[^{footnote_id}]:'):
                    def_line = i
                    break
            
            if def_line is None:
                continue
            
            # Find where it should be (after the first reference)
            first_ref_line = min(footnote_refs[footnote_id])
            paragraph_end = self.find_paragraph_end(lines, first_ref_line)
            
            # If the definition is not immediately after the paragraph, reorganization is needed
            if def_line != paragraph_end + 2:  # +2 for empty line + definition line
                return True
        
        return False
    
    def reorganize_footnotes(self, content: str) -> str:
        """
        Reorganize footnotes in the content.
        """
        lines, footnote_defs, footnote_refs = self.parse_content(content)
        
        if not self.needs_reorganization(lines, footnote_defs, footnote_refs):
            return content
        
        # Create a new list of lines
        new_lines = []
        processed_footnotes = set()
        skip_lines = set()  # Lines to skip (original footnote definitions)
        
        # Mark original footnote definition lines for removal
        for i, line in enumerate(lines):
            if self.footnote_def_pattern.match(line):
                skip_lines.add(i)
        
        # Process each line
        for i, line in enumerate(lines):
            # Skip original footnote definition lines
            if i in skip_lines:
                continue
            
            new_lines.append(line)
            
            # Check if this line contains footnote references
            refs_in_line = []
            for match in self.footnote_ref_pattern.finditer(line):
                footnote_id = match.group(1)
                if footnote_id in footnote_defs and footnote_id not in processed_footnotes:
                    refs_in_line.append(footnote_id)
            
            # If this is the end of a paragraph with footnote references, add the definitions
            if refs_in_line:
                paragraph_end = self.find_paragraph_end(lines, i)
                
                # If we're at the paragraph end, add footnote definitions
                if i == paragraph_end:
                    # Only add empty line if the last line isn't already empty
                    if new_lines and new_lines[-1].strip():
                        new_lines.append('')
                    
                    for footnote_id in refs_in_line:
                        if footnote_id in footnote_defs:
                            footnote_def_line = f'[^{footnote_id}]: {footnote_defs[footnote_id]}'
                            new_lines.append(footnote_def_line)
                            processed_footnotes.add(footnote_id)
                    
                    # Add empty line after footnotes only if next content isn't empty
                    if i + 1 < len(lines) and lines[i + 1].strip():
                        new_lines.append('')
        
        # Remove excessive empty lines (more than 2 consecutive)
        final_lines = []
        empty_count = 0
        
        for line in new_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    final_lines.append(line)
            else:
                empty_count = 0
                final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def process_file(self, file_path: Path) -> bool:
        """
        Process a single QMD file.
        Returns True if the file was modified, False otherwise.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            reorganized_content = self.reorganize_footnotes(original_content)
            
            # Check if content actually changed
            if original_content == reorganized_content:
                print(f"No changes needed: {file_path}")
                return False
            
            if self.dry_run:
                print(f"Would modify: {file_path}")
                return True
            
            # Create backup if requested
            if self.create_backup:
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                shutil.copy2(file_path, backup_path)
                print(f"Created backup: {backup_path}")
            
            # Write the reorganized content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(reorganized_content)
            
            print(f"Reorganized footnotes: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def process_path(self, path: str) -> Tuple[int, int]:
        """
        Process all QMD files in the given path.
        Returns (files_processed, files_modified).
        """
        qmd_files = self.find_qmd_files(path)
        
        if not qmd_files:
            print(f"No .qmd files found in {path}")
            return 0, 0
        
        files_processed = 0
        files_modified = 0
        
        for file_path in qmd_files:
            files_processed += 1
            if self.process_file(file_path):
                files_modified += 1
        
        return files_processed, files_modified


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize footnotes in QMD files to appear after their references",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reorganize_footnotes.py file.qmd
  python reorganize_footnotes.py --dry-run contents/
  python reorganize_footnotes.py --backup --dry-run contents/core/
        """
    )
    
    parser.add_argument('path', help='Path to QMD file or directory containing QMD files')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview changes without modifying files')
    parser.add_argument('--backup', action='store_true',
                       help='Create .bak backup files before modifying')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist")
        sys.exit(1)
    
    reorganizer = FootnoteReorganizer(dry_run=args.dry_run, create_backup=args.backup)
    files_processed, files_modified = reorganizer.process_path(args.path)
    
    print(f"\nSummary:")
    print(f"Files processed: {files_processed}")
    print(f"Files modified: {files_modified}")
    
    if args.dry_run and files_modified > 0:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
