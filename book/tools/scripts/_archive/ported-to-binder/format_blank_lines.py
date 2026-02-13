#!/usr/bin/env python3
"""
Script to collapse multiple consecutive blank lines into single blank lines.
This is a cross-platform solution that works on both macOS and Linux.
"""

import sys
import re


def collapse_blank_lines(content):
    """Replace multiple consecutive blank lines with a single blank line.

    Preserves content inside code blocks (```...```) to avoid interfering
    with language-specific formatters like Black.
    """
    lines = content.split('\n')
    result = []
    in_code_block = False
    blank_count = 0

    for line in lines:
        # Detect code block boundaries
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            # Flush accumulated blank lines before code block
            if blank_count > 0:
                result.append('')  # Add single blank line
                blank_count = 0
            result.append(line)
            continue

        # Inside code blocks, preserve all content including blank lines
        if in_code_block:
            result.append(line)
            continue

        # Outside code blocks, collapse excessive blank lines
        if line.strip() == '':
            blank_count += 1
        else:
            # Add at most one blank line
            if blank_count > 0:
                result.append('')
                blank_count = 0
            result.append(line)

    # Handle trailing blank lines
    if blank_count > 0:
        result.append('')

    return '\n'.join(result)


def main():
    """Process files and collapse extra blank lines."""
    if len(sys.argv) < 2:
        print("Usage: python collapse_blank_lines.py <file1> [file2] ...")
        sys.exit(1)

    modified_files = []

    for filename in sys.argv[1:]:
        try:
            # Read the file
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            # Process the content
            new_content = collapse_blank_lines(content)

            # Write back if modified
            if new_content != content:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                modified_files.append(filename)
                print(f"Modified: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    # Exit with code 1 if files were modified (pre-commit convention)
    if modified_files:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
