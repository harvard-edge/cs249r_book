#!/usr/bin/env python3
"""
Fix bullet/numbered list spacing in QMD files.

Two fixes:
1. Add blank line BEFORE lists (text: followed by bullet needs blank line)
2. Remove blank lines BETWEEN consecutive list items (items should be consecutive)

Usage:
  # Check mode (warn only, for CI):
  python fix_bullet_spacing.py --check file1.qmd file2.qmd

  # Fix mode (auto-fix, default for pre-commit):
  python fix_bullet_spacing.py file1.qmd file2.qmd

  # Process entire directory:
  python fix_bullet_spacing.py book/quarto/contents/vol1/
"""

import re
import sys
from pathlib import Path


def check_bullet_spacing(content: str) -> list[tuple[int, str, str]]:
    """
    Check for bullet list spacing issues:
    1. Missing blank line before lists
    2. Extra blank lines between consecutive list items
    
    Returns: list of (line_number, issue_type, context) tuples
    """
    issues = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for missing blank line before list
        # (Only for paragraph text ending with colon, NOT for list items ending with colon)
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            if (line.rstrip().endswith(':') and 
                not line.strip().startswith('```') and
                not line.strip().startswith('#|') and
                not '://' in line and
                not line.strip().startswith('def ') and
                not line.strip().startswith('class ') and
                not is_list_item(line) and  # Don't flag list items ending with :
                is_list_item(next_line)):
                issues.append((i + 1, 'missing_before', line.strip()))
        
        # Check for extra blank lines between list items
        if is_list_item(line):
            j = i + 1
            blank_count = 0
            while j < len(lines) and lines[j].strip() == '':
                blank_count += 1
                j += 1
            if blank_count > 0 and j < len(lines) and is_list_item(lines[j]):
                issues.append((i + 1, 'extra_between', f"{blank_count} blank line(s) before item"))
        
        i += 1
    
    return issues


def is_list_item(line: str) -> bool:
    """Check if a line is a list item (bullet or numbered)."""
    return (line.startswith('*   ') or 
            line.startswith('-   ') or 
            line.startswith('- ') or
            bool(re.match(r'^\d+\.  ', line)))


def fix_bullet_spacing(content: str) -> tuple[str, int]:
    """
    Fix bullet list spacing:
    1. Add blank line BEFORE lists that are missing it
    2. Remove extra blank lines BETWEEN consecutive list items
    
    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split('\n')
    result = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if we need to add blank line before a list
        # (Only for paragraph text ending with colon, NOT for list items ending with colon)
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            if (line.rstrip().endswith(':') and 
                not line.strip().startswith('```') and
                not line.strip().startswith('#|') and
                not '://' in line and
                not line.strip().startswith('def ') and
                not line.strip().startswith('class ') and
                not is_list_item(line) and  # Don't add blank after list items ending with :
                is_list_item(next_line)):
                result.append(line)
                result.append('')
                fixes += 1
                i += 1
                continue
        
        # Check if this is a list item followed by blank line(s) then another list item
        if is_list_item(line):
            result.append(line)
            # Look ahead for blank lines followed by another list item
            j = i + 1
            blank_count = 0
            while j < len(lines) and lines[j].strip() == '':
                blank_count += 1
                j += 1
            
            # If there's a list item after the blank(s), skip the blanks
            if blank_count > 0 and j < len(lines) and is_list_item(lines[j]):
                fixes += blank_count
                i = j  # Skip to the next list item
                continue
        else:
            result.append(line)
        
        i += 1
    
    return '\n'.join(result), fixes


def process_file(filepath: Path, check_only: bool = False) -> int:
    """Process a single file. Returns number of issues/fixes."""
    content = filepath.read_text()
    
    if check_only:
        issues = check_bullet_spacing(content)
        if issues:
            print(f"{filepath}:")
            for line_num, issue_type, context in issues:
                if issue_type == 'missing_before':
                    print(f"  Line {line_num}: Missing blank line before bullet list")
                    print(f"    → {context[:60]}{'...' if len(context) > 60 else ''}")
                else:  # extra_between
                    print(f"  Line {line_num}: Extra blank line(s) between list items")
                    print(f"    → {context}")
        return len(issues)
    else:
        fixed_content, fixes = fix_bullet_spacing(content)
        if fixes > 0:
            filepath.write_text(fixed_content)
            print(f"Fixed {fixes} list spacing issue(s) in {filepath}")
        return fixes


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Fix bullet list spacing in QMD files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check mode (CI/validation):
  python fix_bullet_spacing.py --check book/quarto/contents/

  # Fix mode (pre-commit default):
  python fix_bullet_spacing.py file1.qmd file2.qmd
"""
    )
    parser.add_argument('paths', nargs='*', default=['.'], 
                        help='Files or directories to process')
    parser.add_argument('--check', '-c', action='store_true', 
                        help='Check only, do not fix (exit 1 if issues found)')
    parser.add_argument('--fix', action='store_true',
                        help='Auto-fix issues (default behavior)')
    args = parser.parse_args()
    
    check_only = args.check and not args.fix
    total_issues = 0
    
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.qmd':
            total_issues += process_file(path, check_only)
        elif path.is_dir():
            for qmd_file in path.rglob('*.qmd'):
                total_issues += process_file(qmd_file, check_only)
    
    if total_issues > 0:
        if check_only:
            print(f"\n❌ Found {total_issues} bullet list(s) missing blank line before them.")
            print("   Run without --check to auto-fix, or add blank line before bullet lists.")
        else:
            print(f"\n✓ Fixed {total_issues} bullet list(s).")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
