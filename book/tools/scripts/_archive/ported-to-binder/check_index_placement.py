#!/usr/bin/env python3
r"""
Check for improper index placement in QMD files.

This script identifies LaTeX \index{} commands that appear on the same line as
structural elements (headings, callouts, divs), which breaks rendering.

Correct pattern:
    ### Heading {#id}
    
    \index{...}Content starts...

Incorrect patterns caught:
    ### Heading {#id}\index{...}
    \index{...}::: {.callout-...}
    ::: {.callout-...} \index{...}
    \index{...}[^fn-name]: footnote definition

Usage:
    python check_index_placement.py file1.qmd file2.qmd ...
    python check_index_placement.py -d path/to/directory/
    python check_index_placement.py -d path/to/directory/ --fix

Exit codes:
    0 - No issues found (or all fixed with --fix)
    1 - Issues found
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


# Issue types for categorization
ISSUE_HEADING = "heading"
ISSUE_INDEX_BEFORE_DIV = "index_before_div"
ISSUE_INDEX_AFTER_DIV = "index_after_div"
ISSUE_INDEX_BEFORE_FOOTNOTE = "index_before_footnote"


def check_file(filepath: Path) -> List[Tuple[int, str, str, str]]:
    """
    Check a single file for index placement issues.
    
    Returns:
        List of (line_number, line_content, issue_description, issue_type) tuples
    """
    issues = []
    in_code_block = False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return issues
    
    for i, line in enumerate(lines, start=1):
        # Track code block boundaries
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        
        # Skip lines inside code blocks
        if in_code_block:
            continue
        
        # Pattern 1: Heading with inline index
        # Matches: ### Heading {#id}\index{...}
        if re.search(r'^#{1,6}\s+.*\\index\{', line):
            issues.append((
                i,
                line.rstrip(),
                "Index command on same line as heading",
                ISSUE_HEADING
            ))
        
        # Pattern 2: Index directly before div/callout opening
        # Matches: \index{...}::: {.callout-...}
        elif re.search(r'\\index\{[^}]*\}:::', line):
            issues.append((
                i,
                line.rstrip(),
                "Index command directly before ::: (div/callout opening)",
                ISSUE_INDEX_BEFORE_DIV
            ))
        
        # Pattern 3: Div/callout opening with inline index at end
        # Matches: ::: {.callout-...} \index{...}
        # Note: Skip if it's a figure caption (fig-cap=) - those are OK
        elif re.search(r'^::+\s+\{[^}]*\}\s*\\index\{', line) and 'fig-cap=' not in line:
            issues.append((
                i,
                line.rstrip(),
                "Index command on same line as div/callout opening",
                ISSUE_INDEX_AFTER_DIV
            ))
        
        # Pattern 4: Index at start of footnote definition
        # Matches: \index{...}[^fn-name]: content
        # Footnote definitions must start with [^name]: at line beginning
        elif re.search(r'^\\index\{[^}]*\}.*\[\^[^\]]+\]:', line):
            issues.append((
                i,
                line.rstrip(),
                "Index command before footnote definition (move to footnote reference)",
                ISSUE_INDEX_BEFORE_FOOTNOTE
            ))
    
    return issues


def fix_line(line: str, issue_type: str) -> str:
    """
    Fix a single line based on the issue type.
    
    Returns the fixed line(s) as a string (may include newlines).
    """
    if issue_type == ISSUE_HEADING:
        # Extract all \index{...} from the line and move to after heading
        indexes = re.findall(r'\\index\{[^}]*\}', line)
        # Remove indexes from original line
        cleaned = re.sub(r'\\index\{[^}]*\}', '', line).rstrip()
        # Return heading + blank line + indexes on their own line
        return cleaned + '\n\n' + ''.join(indexes) + '\n'
    
    elif issue_type == ISSUE_INDEX_BEFORE_DIV:
        # Split at ::: and add newline between
        # \index{...}\index{...}::: {.callout} -> \index{...}\index{...}\n\n::: {.callout}
        match = re.match(r'^(.*\\index\{[^}]*\})(:::.*)', line)
        if match:
            indexes_part = match.group(1).rstrip()
            div_part = match.group(2)
            return indexes_part + '\n\n' + div_part + '\n'
        return line
    
    elif issue_type == ISSUE_INDEX_AFTER_DIV:
        # Move index after div opener to its own line
        # ::: {.callout} \index{...} -> ::: {.callout}\n\n\index{...}
        match = re.match(r'^(::+\s+\{[^}]*\})\s*(\\index\{.*)', line)
        if match:
            div_part = match.group(1)
            indexes_part = match.group(2).rstrip()
            return div_part + '\n\n' + indexes_part + '\n'
        return line
    
    elif issue_type == ISSUE_INDEX_BEFORE_FOOTNOTE:
        # Move index commands to their own line before footnote definition
        # \index{...}\index{...}[^fn]: content -> \index{...}\index{...}\n\n[^fn]: content
        match = re.match(r'^((?:\\index\{[^}]*\})+)(\[\^[^\]]+\]:.*)', line)
        if match:
            indexes_part = match.group(1).rstrip()
            footnote_part = match.group(2)
            return indexes_part + '\n\n' + footnote_part + '\n'
        return line
    
    return line


def fix_file(filepath: Path, issues: List[Tuple[int, str, str, str]], dry_run: bool = False) -> int:
    """
    Fix all issues in a file.
    
    Returns the number of fixes applied.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return 0
    
    # Sort issues by line number in reverse order so we can fix from bottom to top
    # (this prevents line number shifts from affecting subsequent fixes)
    sorted_issues = sorted(issues, key=lambda x: x[0], reverse=True)
    
    fixes_applied = 0
    for line_num, _, _, issue_type in sorted_issues:
        idx = line_num - 1  # Convert to 0-based index
        if 0 <= idx < len(lines):
            original = lines[idx]
            fixed = fix_line(original, issue_type)
            if fixed != original:
                lines[idx] = fixed
                fixes_applied += 1
    
    if fixes_applied > 0 and not dry_run:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        except Exception as e:
            print(f"Error writing {filepath}: {e}", file=sys.stderr)
            return 0
    
    return fixes_applied


def format_issue(filepath: Path, line_num: int, line_content: str, description: str, show_line: bool = True) -> str:
    """Format an issue for display."""
    output = f"\n{filepath}:{line_num}: {description}"
    if show_line:
        # Truncate long lines
        display_line = line_content if len(line_content) <= 100 else line_content[:97] + "..."
        output += f"\n  {display_line}"
        # Highlight the \index part
        index_match = re.search(r'\\index\{[^}]*\}', display_line)
        if index_match:
            pos = index_match.start()
            output += f"\n  {' ' * pos}{'~' * len(index_match.group())}"
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Check for improper \\index{} placement in QMD files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file1.qmd file2.qmd
  %(prog)s -d book/quarto/contents/vol1/
  %(prog)s -d book/quarto/contents/ --quiet
  %(prog)s -d book/quarto/contents/ --fix
  %(prog)s -d book/quarto/contents/ --fix --dry-run
        """
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='QMD files to check'
    )
    parser.add_argument(
        '-d', '--directory',
        type=Path,
        help='Directory to recursively search for QMD files'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show summary, not individual issues'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically fix issues by moving \\index{} to its own line'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without making changes (use with --fix)'
    )
    
    args = parser.parse_args()
    
    # Collect files to check
    files_to_check: List[Path] = []
    
    if args.directory:
        files_to_check.extend(args.directory.rglob('*.qmd'))
    
    if args.files:
        files_to_check.extend(Path(f) for f in args.files)
    
    if not files_to_check:
        parser.print_help()
        return 0
    
    # Check each file
    total_issues = 0
    total_fixes = 0
    files_with_issues = []
    
    for filepath in sorted(files_to_check):
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist", file=sys.stderr)
            continue
        
        issues = check_file(filepath)
        
        if issues:
            files_with_issues.append(filepath)
            total_issues += len(issues)
            
            if not args.quiet:
                for line_num, line_content, description, _ in issues:
                    print(format_issue(filepath, line_num, line_content, description))
            
            if args.fix:
                fixes = fix_file(filepath, issues, dry_run=args.dry_run)
                total_fixes += fixes
                if fixes > 0:
                    action = "Would fix" if args.dry_run else "Fixed"
                    print(f"  → {action} {fixes} issue(s) in {filepath}")
                # Warn about footnote fixes needing review
                footnote_issues = [i for i in issues if i[3] == ISSUE_INDEX_BEFORE_FOOTNOTE]
                if footnote_issues:
                    print(f"    ⚠ {len(footnote_issues)} footnote fix(es) - consider moving \\index{{}} to where footnote is referenced")
    
    # Print summary
    if total_issues > 0:
        print(f"\n{'=' * 70}")
        if args.fix and not args.dry_run:
            print(f"✓ Fixed {total_fixes} index placement issue(s) in {len(files_with_issues)} file(s)")
        elif args.fix and args.dry_run:
            print(f"Would fix {total_fixes} index placement issue(s) in {len(files_with_issues)} file(s)")
        else:
            print(f"❌ Found {total_issues} index placement issue(s) in {len(files_with_issues)} file(s)")
            print(f"{'=' * 70}")
            print("\nFix by moving \\index{{}} to its own line after the heading/callout:")
            print("  ### Heading {{#id}}")
            print("  ")
            print("  \\index{{...}}Content starts here...")
            print("\nOr run with --fix to automatically fix these issues.")
            return 1
        print(f"{'=' * 70}")
        return 0
    else:
        if not args.quiet:
            print(f"✓ No index placement issues found in {len(files_to_check)} file(s)")
        return 0


if __name__ == '__main__':
    sys.exit(main())
