#!/usr/bin/env python3
"""
Detect self-referential or circular section references in Quarto files.

This script identifies cases where:
1. A section refers to itself
2. A section refers to its immediate parent
3. A section refers to its immediate child

These patterns usually indicate writing issues that should be reviewed.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def parse_heading_structure(content: str, filepath: Path) -> List[Dict]:
    """
    Parse all headings from a Quarto file with their IDs, levels, and positions.

    Returns:
        List of dicts with keys: level, title, id, line_num, parent_id
    """
    headings = []
    lines = content.split('\n')

    # Stack to track parent sections at each level
    parent_stack = {}  # level -> heading dict

    for line_num, line in enumerate(lines, start=1):
        # Match Markdown headings with optional IDs
        # Format: ### Heading {#sec-id-here}
        heading_match = re.match(r'^(#{1,6})\s+(.+?)(?:\s+\{#([^\}]+)\})?$', line)

        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            section_id = heading_match.group(3)

            # Find parent (closest heading with lower level)
            parent_id = None
            for parent_level in range(level - 1, 0, -1):
                if parent_level in parent_stack:
                    parent_id = parent_stack[parent_level]['id']
                    break

            heading_dict = {
                'level': level,
                'title': title,
                'id': section_id,
                'line_num': line_num,
                'parent_id': parent_id,
                'filepath': filepath
            }

            headings.append(heading_dict)

            # Update parent stack
            parent_stack[level] = heading_dict
            # Clear deeper levels
            parent_stack = {k: v for k, v in parent_stack.items() if k <= level}

    return headings


def extract_cross_references(content: str, filepath: Path) -> List[Dict]:
    """
    Extract all section cross-references from content.

    Returns:
        List of dicts with keys: ref_id, line_num, context
    """
    references = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, start=1):
        # Find all @sec- references
        matches = re.finditer(r'@(sec-[a-zA-Z0-9\-]+)', line)

        for match in matches:
            ref_id = match.group(1)
            references.append({
                'ref_id': ref_id,
                'line_num': line_num,
                'context': line.strip(),
                'filepath': filepath
            })

    return references


def build_section_hierarchy(headings: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Build mappings for section relationships.

    Returns:
        (section_map, children_map)
        - section_map: id -> heading dict
        - children_map: id -> list of child section ids
    """
    section_map = {}
    children_map = defaultdict(list)

    for heading in headings:
        if heading['id']:
            section_map[heading['id']] = heading

            # Track parent-child relationships
            if heading['parent_id']:
                children_map[heading['parent_id']].append(heading['id'])

    return section_map, children_map


def find_section_for_reference(ref_line_num: int, headings: List[Dict]) -> Optional[Dict]:
    """
    Find which section a reference belongs to based on line number.
    """
    current_section = None

    for heading in headings:
        if heading['line_num'] <= ref_line_num:
            current_section = heading
        else:
            break

    return current_section


def check_self_referential_issues(filepath: Path) -> List[Dict]:
    """
    Check a single file for self-referential section issues.

    Returns:
        List of issue dicts with keys: type, section, reference, line_num, message
    """
    issues = []

    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return issues

    headings = parse_heading_structure(content, filepath)
    references = extract_cross_references(content, filepath)
    section_map, children_map = build_section_hierarchy(headings)

    for ref in references:
        ref_id = ref['ref_id']
        ref_line_num = ref['line_num']

        # Find which section this reference is in
        current_section = find_section_for_reference(ref_line_num, headings)

        if not current_section or not current_section['id']:
            continue

        current_id = current_section['id']

        # Check for self-reference (exact match)
        if ref_id == current_id:
            issues.append({
                'type': 'self_reference',
                'section': current_section['title'],
                'section_id': current_id,
                'reference_id': ref_id,
                'line_num': ref_line_num,
                'filepath': filepath,
                'message': f"Section refers to itself: '{current_section['title']}' references @{ref_id}"
            })

        # Check for parent reference
        elif current_section['parent_id'] == ref_id:
            parent_section = section_map.get(ref_id)
            parent_title = parent_section['title'] if parent_section else 'Unknown'
            issues.append({
                'type': 'parent_reference',
                'section': current_section['title'],
                'section_id': current_id,
                'reference_id': ref_id,
                'line_num': ref_line_num,
                'filepath': filepath,
                'message': f"Section refers to its parent: '{current_section['title']}' references parent '{parent_title}' (@{ref_id})"
            })

        # Check for immediate child reference
        elif ref_id in children_map.get(current_id, []):
            child_section = section_map.get(ref_id)
            child_title = child_section['title'] if child_section else 'Unknown'
            issues.append({
                'type': 'child_reference',
                'section': current_section['title'],
                'section_id': current_id,
                'reference_id': ref_id,
                'line_num': ref_line_num,
                'filepath': filepath,
                'message': f"Section refers to its immediate child: '{current_section['title']}' references child '{child_title}' (@{ref_id})"
            })

    return issues


def scan_directory(directory: Path, pattern: str = "**/*.qmd") -> List[Dict]:
    """
    Scan all Quarto files in a directory for self-referential issues.
    """
    all_issues = []

    for filepath in directory.glob(pattern):
        if filepath.is_file():
            issues = check_self_referential_issues(filepath)
            all_issues.extend(issues)

    return all_issues


def print_report(issues: List[Dict], verbose: bool = False):
    """
    Print a formatted report of issues found.
    """
    if not issues:
        print("✅ No self-referential section issues found.")
        return

    # Group by type
    by_type = defaultdict(list)
    for issue in issues:
        by_type[issue['type']].append(issue)

    print(f"\n🔍 Found {len(issues)} self-referential section issue(s):\n")

    for issue_type in ['self_reference', 'parent_reference', 'child_reference']:
        if issue_type not in by_type:
            continue

        type_name = issue_type.replace('_', ' ').title()
        type_issues = by_type[issue_type]

        print(f"\n{type_name} ({len(type_issues)} issue(s)):")
        print("=" * 80)

        for issue in type_issues:
            rel_path = issue['filepath'].relative_to(Path.cwd()) if issue['filepath'].is_relative_to(Path.cwd()) else issue['filepath']
            print(f"\n  File: {rel_path}")
            print(f"  Line: {issue['line_num']}")
            print(f"  {issue['message']}")

            if verbose:
                print(f"  Section ID: {issue['section_id']}")
                print(f"  Reference ID: {issue['reference_id']}")

    print("\n" + "=" * 80)
    print(f"Total: {len(issues)} issue(s) found\n")


def main():
    """
    Main entry point for the script.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect self-referential section references in Quarto files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a specific file
  python check_self_referential_sections.py book/quarto/contents/core/frameworks/frameworks.qmd

  # Check all files in a directory
  python check_self_referential_sections.py book/quarto/contents/

  # Check with verbose output
  python check_self_referential_sections.py book/quarto/contents/ --verbose
        """
    )

    parser.add_argument(
        'path',
        type=Path,
        nargs='?',
        default=Path('book/quarto/contents'),
        help='Path to file or directory to check (default: book/quarto/contents)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output including section and reference IDs'
    )

    parser.add_argument(
        '--pattern',
        default='**/*.qmd',
        help='Glob pattern for files to check (default: **/*.qmd)'
    )

    args = parser.parse_args()

    path = args.path.resolve()

    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    # Scan files
    if path.is_file():
        issues = check_self_referential_issues(path)
    else:
        issues = scan_directory(path, args.pattern)

    # Print report
    print_report(issues, verbose=args.verbose)

    # Exit with error code if issues found
    sys.exit(1 if issues else 0)


if __name__ == '__main__':
    main()
