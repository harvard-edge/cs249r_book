#!/usr/bin/env python3
"""
Fix bullet list spacing in QMD files.

Ensures there's a blank line before bullet lists start.
Pattern: Text ending with colon followed directly by bullet should have blank line.

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


def check_bullet_spacing(content: str) -> list[tuple[int, str]]:
    """
    Check for bullet lists missing blank line before them.
    
    Returns: list of (line_number, line_content) tuples for issues found
    """
    issues = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            # Line ends with : (but not in code block markers or URLs)
            if (line.rstrip().endswith(':') and 
                not line.strip().startswith('```') and
                not line.strip().startswith('#|') and
                not '://' in line and
                not line.strip().startswith('def ') and
                not line.strip().startswith('class ') and
                # Next line is a bullet
                (next_line.startswith('*   ') or 
                 next_line.startswith('-   ') or
                 next_line.startswith('- ') or
                 re.match(r'^\d+\.  ', next_line))):
                issues.append((i + 1, line.strip()))  # 1-indexed line number
    
    return issues


def fix_bullet_spacing(content: str) -> tuple[str, int]:
    """
    Fix bullet lists that are missing blank line before them.
    
    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split('\n')
    result = []
    
    for i, line in enumerate(lines):
        result.append(line)
        
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            if (line.rstrip().endswith(':') and 
                not line.strip().startswith('```') and
                not line.strip().startswith('#|') and
                not '://' in line and
                not line.strip().startswith('def ') and
                not line.strip().startswith('class ') and
                (next_line.startswith('*   ') or 
                 next_line.startswith('-   ') or
                 next_line.startswith('- ') or
                 re.match(r'^\d+\.  ', next_line))):
                result.append('')
                fixes += 1
    
    return '\n'.join(result), fixes


def process_file(filepath: Path, check_only: bool = False) -> int:
    """Process a single file. Returns number of issues/fixes."""
    content = filepath.read_text()
    
    if check_only:
        issues = check_bullet_spacing(content)
        if issues:
            print(f"{filepath}:")
            for line_num, line_content in issues:
                print(f"  Line {line_num}: Missing blank line before bullet list")
                print(f"    → {line_content[:60]}{'...' if len(line_content) > 60 else ''}")
        return len(issues)
    else:
        fixed_content, fixes = fix_bullet_spacing(content)
        if fixes > 0:
            filepath.write_text(fixed_content)
            print(f"Fixed {fixes} bullet list(s) in {filepath}")
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
