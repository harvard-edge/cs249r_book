#!/usr/bin/env python3
"""
Check and fix markdown list formatting issues.

This script ensures that bullet lists in markdown files are preceded by an empty line,
which is required for proper markdown rendering.

Usage:
    # Check files without fixing
    python check_list_formatting.py --check path/to/file.qmd

    # Fix issues automatically
    python check_list_formatting.py --fix path/to/file.qmd

    # Check all .qmd files in a directory
    python check_list_formatting.py --check --recursive quarto/contents/core/
"""

import argparse
import os
import sys
from pathlib import Path


def find_list_formatting_issues(filepath):
    """
    Find lines ending with : followed immediately by bullet lists without empty line.

    Args:
        filepath: Path to the markdown file to check

    Returns:
        List of tuples (line_number, line_content) for each issue found
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return []

    issues = []
    for i in range(len(lines) - 1):
        current = lines[i].rstrip()
        next_line = lines[i + 1].rstrip()

        # Check if current line ends with : (not within code blocks or other special contexts)
        # and next line is a bullet list
        if (current and
            current.endswith(':') and
            not current.startswith('```') and
            not current.startswith(':::') and
            not current.startswith('#') and
            not current.startswith('|') and  # Skip tables
            next_line.startswith('- ')):
            issues.append((i + 1, current))

    return issues


def fix_list_formatting(filepath, dry_run=False):
    """
    Add empty line before bullet lists that follow lines ending with colon.

    Args:
        filepath: Path to the markdown file to fix
        dry_run: If True, only report issues without fixing

    Returns:
        Number of fixes made
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return 0

    fixed_lines = []
    fixes_made = 0
    i = 0

    while i < len(lines):
        current = lines[i].rstrip()

        # Look ahead to see if next line is a bullet without empty line before it
        if i < len(lines) - 1:
            next_line = lines[i + 1].rstrip()

            if (current and
                current.endswith(':') and
                not current.startswith('```') and
                not current.startswith(':::') and
                not current.startswith('#') and
                not current.startswith('|') and
                next_line.startswith('- ')):

                # Add current line and an empty line
                fixed_lines.append(lines[i])
                fixed_lines.append('\n')
                fixes_made += 1
                i += 1
                continue

        fixed_lines.append(lines[i])
        i += 1

    if fixes_made > 0 and not dry_run:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
        except Exception as e:
            print(f"Error writing {filepath}: {e}", file=sys.stderr)
            return 0

    return fixes_made


def process_file(filepath, check_only=True):
    """
    Process a single file.

    Args:
        filepath: Path to the file to process
        check_only: If True, only check for issues. If False, fix them.

    Returns:
        Tuple of (issues_found, issues_fixed)
    """
    issues = find_list_formatting_issues(filepath)

    if issues:
        print(f"\n{filepath}: {len(issues)} issue(s)")
        for line_num, content in issues[:5]:  # Show first 5
            print(f"  Line {line_num}: {content}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")

        if not check_only:
            fixes = fix_list_formatting(filepath, dry_run=False)
            if fixes > 0:
                print(f"  ✅ Fixed {fixes} issue(s)")
            return len(issues), fixes

        return len(issues), 0

    return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description='Check and fix markdown list formatting issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Files or directories to check'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Fix issues automatically (default: only check)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check for issues without fixing (default behavior)'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Recursively process directories'
    )

    args = parser.parse_args()

    check_only = not args.fix or args.check

    files_to_process = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            if path.suffix == '.qmd':
                files_to_process.append(path)
        elif path.is_dir():
            if args.recursive:
                files_to_process.extend(path.rglob('*.qmd'))
            else:
                files_to_process.extend(path.glob('*.qmd'))
        else:
            print(f"Warning: {path} not found", file=sys.stderr)

    if not files_to_process:
        print("No .qmd files found to process", file=sys.stderr)
        return 1

    total_issues = 0
    total_fixed = 0

    for filepath in sorted(files_to_process):
        issues, fixed = process_file(str(filepath), check_only)
        total_issues += issues
        total_fixed += fixed

    print(f"\n{'=' * 60}")
    if check_only:
        if total_issues == 0:
            print("✅ No list formatting issues found!")
            return 0
        else:
            print(f"❌ Found {total_issues} issue(s) across {len(files_to_process)} file(s)")
            print("\nRun with --fix to automatically fix these issues")
            return 1
    else:
        if total_fixed > 0:
            print(f"✅ Fixed {total_fixed} issue(s) across {len(files_to_process)} file(s)")
            return 0
        else:
            print("✅ No issues to fix")
            return 0


if __name__ == '__main__':
    sys.exit(main())
