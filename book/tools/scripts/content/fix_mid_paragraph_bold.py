#!/usr/bin/env python3
"""
Find and remove bold formatting from the middle of paragraphs.
Excludes footnotes and intentional bold at start of lines (like captions).
"""

import re
from pathlib import Path
from typing import List, Tuple


def find_mid_paragraph_bold(content: str, filepath: str) -> List[Tuple[int, str, str]]:
    """
    Find lines with bold text in the middle of paragraphs.

    Returns list of (line_number, original_line, fixed_line) tuples.
    """
    lines = content.split('\n')
    fixes = []

    for i, line in enumerate(lines, 1):
        # Skip footnote lines
        if line.strip().startswith('[^'):
            continue

        # Skip lines that start with bold (captions, list items, etc.)
        if line.strip().startswith('**'):
            continue

        # Skip figure/table captions
        if line.strip().startswith(':'):
            continue

        # Check if line has bold text preceded by lowercase letter or punctuation
        # This indicates bold in the middle of a sentence/paragraph
        pattern = r'([a-z,;:)])\s+\*\*([^*]+)\*\*'
        if re.search(pattern, line):
            # Remove the bold formatting
            fixed_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            fixes.append((i, line, fixed_line))

    return fixes


def process_file(filepath: Path, dry_run: bool = True) -> List[Tuple[int, str, str]]:
    """Process a single file and optionally apply fixes."""
    try:
        content = filepath.read_text(encoding='utf-8')
        fixes = find_mid_paragraph_bold(content, str(filepath))

        if fixes and not dry_run:
            # Apply all fixes
            lines = content.split('\n')
            for line_num, original, fixed in fixes:
                lines[line_num - 1] = fixed

            filepath.write_text('\n'.join(lines), encoding='utf-8')

        return fixes
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []


def main():
    # Find all .qmd files in contents/core only
    base_path = Path('/Users/VJ/GitHub/MLSysBook/quarto/contents/core')

    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return

    qmd_files = list(base_path.rglob('*.qmd'))

    print(f"Found {len(qmd_files)} .qmd files to check\n")

    # First pass: dry run to show what would be changed
    print("=" * 80)
    print("DRY RUN - Finding bold text in middle of paragraphs")
    print("=" * 80)

    all_fixes = {}
    for qmd_file in sorted(qmd_files):
        fixes = process_file(qmd_file, dry_run=True)
        if fixes:
            all_fixes[qmd_file] = fixes

    if not all_fixes:
        print("\nNo bold text found in middle of paragraphs!")
        return

    # Display findings
    for filepath, fixes in all_fixes.items():
        rel_path = filepath.relative_to(Path('/Users/VJ/GitHub/MLSysBook'))
        print(f"\n{rel_path}")
        print("-" * 80)
        for line_num, original, fixed in fixes:
            print(f"Line {line_num}:")
            print(f"  BEFORE: {original[:120]}{'...' if len(original) > 120 else ''}")
            print(f"  AFTER:  {fixed[:120]}{'...' if len(fixed) > 120 else ''}")

    # Summary
    total_fixes = sum(len(fixes) for fixes in all_fixes.values())
    print("\n" + "=" * 80)
    print(f"SUMMARY: Found {total_fixes} instances across {len(all_fixes)} files")
    print("=" * 80)

    # Ask for confirmation to apply
    response = input("\nApply these fixes? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        print("\nApplying fixes...")
        for filepath in all_fixes.keys():
            process_file(filepath, dry_run=False)
        print("✓ All fixes applied!")
    else:
        print("No changes made.")


if __name__ == '__main__':
    main()
