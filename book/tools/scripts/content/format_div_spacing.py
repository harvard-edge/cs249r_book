#!/usr/bin/env python3
"""
Script to ensure proper blank line spacing inside Quarto div blocks (callouts).

Problem: When content inside a div (like a callout) has a paragraph immediately
followed by a list without a blank line between them, Pandoc may render them
incorrectly in PDF output (content gets "mushed" together on one line).

Additionally, blank lines BETWEEN list items create "loose lists" with extra
spacing, which is usually undesirable.

Solution: This script detects and fixes these patterns by ensuring proper
blank lines exist between block elements inside divs, while removing
unnecessary blank lines between list items.

Patterns fixed:
1. Paragraph → List: Ensures blank line between a paragraph and a list
2. List → Paragraph: Ensures blank line between end of list and new paragraph
3. Bold header → List: Ensures blank line after standalone bold headers before lists
4. List item → blank → List item: Removes blank line to create tight list

Usage:
    python format_div_spacing.py -f <file.qmd>           # Process single file
    python format_div_spacing.py -d <directory>          # Process directory
    python format_div_spacing.py -f <file.qmd> --check   # Check only (no changes)
    python format_div_spacing.py -f <file.qmd> --verbose # Show details
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LineType(Enum):
    """Classification of line types for spacing analysis."""
    BLANK = "blank"
    DIV_START = "div_start"
    DIV_END = "div_end"
    LIST_ITEM = "list_item"
    PARAGRAPH = "paragraph"
    CODE_FENCE = "code_fence"
    HEADER = "header"
    BOLD_HEADER = "bold_header"  # Lines that are just **Bold Text**
    OTHER = "other"


@dataclass
class SpacingIssue:
    """Represents a spacing issue found in a file."""
    line_number: int
    issue_type: str
    before_line: str
    after_line: str
    context: str


def classify_line(line: str) -> LineType:
    """Classify a line by its markdown type."""
    stripped = line.strip()
    
    if stripped == "":
        return LineType.BLANK
    
    # Div boundaries
    if stripped.startswith(":::"):
        if stripped == ":::" or stripped == "::::":
            return LineType.DIV_END
        return LineType.DIV_START
    
    # Code fences
    if stripped.startswith("```"):
        return LineType.CODE_FENCE
    
    # Headers
    if stripped.startswith("#"):
        return LineType.HEADER
    
    # List items (-, *, +, or numbered)
    if re.match(r'^[-*+]\s', stripped) or re.match(r'^\d+\.\s', stripped):
        return LineType.LIST_ITEM
    
    # Bold header lines (entire line is just bold text, possibly with colon)
    # Matches: **Some Header** or **Some Header**:
    if re.match(r'^\*\*[^*]+\*\*:?\s*$', stripped):
        return LineType.BOLD_HEADER
    
    # Regular paragraph
    return LineType.PARAGRAPH


def find_spacing_issues(content: str) -> list[SpacingIssue]:
    """Find spacing issues inside div blocks."""
    lines = content.split('\n')
    issues = []
    
    div_depth = 0
    in_code_block = False
    
    for i, line in enumerate(lines):
        line_type = classify_line(line)
        
        # Track code blocks
        if line_type == LineType.CODE_FENCE:
            in_code_block = not in_code_block
            continue
        
        # Skip processing inside code blocks
        if in_code_block:
            continue
        
        # Track div depth
        if line_type == LineType.DIV_START:
            div_depth += 1
            continue
        elif line_type == LineType.DIV_END:
            div_depth = max(0, div_depth - 1)
            continue
        
        # Only check inside divs
        if div_depth == 0:
            continue
        
        # Check for issues: need to look at current line and previous non-blank line
        if i > 0:
            prev_idx = i - 1
            prev_type = classify_line(lines[prev_idx])
            
            # Issue 1: Bold header or paragraph immediately followed by list item
            # (no blank line between)
            if line_type == LineType.LIST_ITEM and prev_type in (LineType.BOLD_HEADER, LineType.PARAGRAPH):
                issues.append(SpacingIssue(
                    line_number=i + 1,  # 1-indexed
                    issue_type="missing_blank_before_list",
                    before_line=lines[prev_idx].strip(),
                    after_line=line.strip(),
                    context=f"Line {prev_idx + 1}: {lines[prev_idx].strip()[:50]}..."
                ))
            
            # Issue 2: List item immediately followed by paragraph (no blank line)
            if line_type == LineType.PARAGRAPH and prev_type == LineType.LIST_ITEM:
                issues.append(SpacingIssue(
                    line_number=i + 1,
                    issue_type="missing_blank_after_list",
                    before_line=lines[prev_idx].strip(),
                    after_line=line.strip(),
                    context=f"Line {prev_idx + 1}: {lines[prev_idx].strip()[:50]}..."
                ))
            
            # Issue 3: Loose list - blank line between list items
            if i >= 2 and line_type == LineType.LIST_ITEM and prev_type == LineType.BLANK:
                prev_prev_type = classify_line(lines[i - 2])
                if prev_prev_type == LineType.LIST_ITEM:
                    issues.append(SpacingIssue(
                        line_number=i,  # The blank line
                        issue_type="loose_list_blank_line",
                        before_line=lines[i - 2].strip(),
                        after_line=line.strip(),
                        context=f"Line {i - 1}: (blank line between list items)"
                    ))
    
    return issues


def fix_spacing(content: str) -> str:
    """Fix spacing issues inside div blocks."""
    lines = content.split('\n')
    result = []
    
    div_depth = 0
    in_code_block = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_type = classify_line(line)
        
        # Track code blocks
        if line_type == LineType.CODE_FENCE:
            in_code_block = not in_code_block
            result.append(line)
            i += 1
            continue
        
        # Preserve content inside code blocks
        if in_code_block:
            result.append(line)
            i += 1
            continue
        
        # Track div depth
        if line_type == LineType.DIV_START:
            div_depth += 1
            result.append(line)
            i += 1
            continue
        elif line_type == LineType.DIV_END:
            div_depth = max(0, div_depth - 1)
            result.append(line)
            i += 1
            continue
        
        # Only fix inside divs
        if div_depth == 0:
            result.append(line)
            i += 1
            continue
        
        # Fix loose lists: skip blank lines between list items
        if line_type == LineType.BLANK and i + 1 < len(lines) and len(result) > 0:
            next_type = classify_line(lines[i + 1])
            prev_type = classify_line(result[-1])
            if prev_type == LineType.LIST_ITEM and next_type == LineType.LIST_ITEM:
                # Skip this blank line (don't add to result)
                i += 1
                continue
        
        # Check if we need to insert a blank line before this line
        if len(result) > 0:
            prev_type = classify_line(result[-1])
            
            # Insert blank line before list if previous was bold header or paragraph
            if line_type == LineType.LIST_ITEM and prev_type in (LineType.BOLD_HEADER, LineType.PARAGRAPH):
                result.append('')  # Insert blank line
            
            # Insert blank line before paragraph if previous was list item
            elif line_type == LineType.PARAGRAPH and prev_type == LineType.LIST_ITEM:
                result.append('')  # Insert blank line
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)


def process_file(filepath: str, check_only: bool = False, verbose: bool = False) -> tuple[bool, list[SpacingIssue]]:
    """Process a single file. Returns (was_modified, issues_found)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = find_spacing_issues(content)
    
    if verbose and issues:
        print(f"\n{filepath}:")
        for issue in issues:
            print(f"  Line {issue.line_number}: {issue.issue_type}")
            print(f"    Before: {issue.before_line[:60]}...")
            print(f"    After:  {issue.after_line[:60]}...")
    
    if check_only:
        return False, issues
    
    if not issues:
        return False, issues
    
    # Apply fixes
    new_content = fix_spacing(content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True, issues
    
    return False, issues


def process_directory(directory: str, check_only: bool = False, verbose: bool = False) -> tuple[list[str], int]:
    """Process all .qmd files in a directory. Returns (modified_files, total_issues)."""
    modified_files = []
    total_issues = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.qmd'):
                filepath = os.path.join(root, file)
                was_modified, issues = process_file(filepath, check_only, verbose)
                total_issues += len(issues)
                if was_modified:
                    modified_files.append(filepath)
                    print(f"Modified: {filepath} ({len(issues)} issues fixed)")
                elif issues and check_only:
                    print(f"Issues found: {filepath} ({len(issues)} issues)")
    
    return modified_files, total_issues


def main():
    parser = argparse.ArgumentParser(
        description="Fix spacing inside Quarto div blocks (callouts) for proper PDF rendering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fix a single file
    python format_div_spacing.py -f chapter.qmd
    
    # Check all files in a directory (no changes)
    python format_div_spacing.py -d contents/ --check
    
    # Fix all files with verbose output
    python format_div_spacing.py -d contents/ --verbose
        """
    )
    parser.add_argument('-f', '--file', help='Process a single .qmd file')
    parser.add_argument('-d', '--directory', help='Process all .qmd files in directory')
    parser.add_argument('--check', action='store_true', 
                       help='Check only, do not modify files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output for each issue')
    
    args = parser.parse_args()
    
    if not args.file and not args.directory:
        parser.print_help()
        sys.exit(1)
    
    total_issues = 0
    modified_files = []
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        was_modified, issues = process_file(args.file, args.check, args.verbose)
        total_issues = len(issues)
        if was_modified:
            modified_files.append(args.file)
            print(f"Modified: {args.file} ({len(issues)} issues fixed)")
        elif issues:
            if args.check:
                print(f"Issues found: {args.file} ({len(issues)} issues)")
            else:
                print(f"No changes needed: {args.file}")
        else:
            print(f"No issues found: {args.file}")
    
    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
            sys.exit(1)
        modified_files, total_issues = process_directory(args.directory, args.check, args.verbose)
    
    # Summary
    print(f"\n{'=' * 50}")
    if args.check:
        print(f"Total issues found: {total_issues}")
        if total_issues > 0:
            sys.exit(1)
    else:
        print(f"Total files modified: {len(modified_files)}")
        print(f"Total issues fixed: {total_issues}")
        if modified_files:
            sys.exit(1)  # Pre-commit convention: exit 1 if files were modified
    
    sys.exit(0)


if __name__ == '__main__':
    main()
