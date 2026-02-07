#!/usr/bin/env python3
"""
validate_dropcap_compat.py
Pre-commit hook to validate drop cap compatibility in QMD chapters.

The dropcap.lua filter applies \\lettrine{X}{rest} to the first paragraph after
the first NUMBERED H2 in each chapter. This fails when that paragraph starts
with a cross-reference (@sec-...), link, or other non-text element because
Pandoc converts these to Link elements rather than Str (string) elements.

This script validates that opening paragraphs start with plain text.

Usage:
    python3 book/tools/scripts/content/validate_dropcap_compat.py [files...]

Exit codes:
    0 = all checks pass
    1 = issues found
"""

import re
import sys
from pathlib import Path

# Pattern for chapter header: # Title {#sec-...}
CHAPTER_HEADER = re.compile(r'^#\s+[^#].*\{#sec-')

# Pattern for numbered H2 (any H2 without .unnumbered)
NUMBERED_H2 = re.compile(r'^##\s+[^#]')

# Pattern for unnumbered H2
UNNUMBERED_H2 = re.compile(r'^##\s+.*\{.*\.unnumbered.*\}')

# Pattern for cross-reference at start of paragraph
STARTS_WITH_CROSSREF = re.compile(r'^\s*@(sec|fig|tbl|lst|eq)-')

# Pattern for markdown link at start of paragraph
STARTS_WITH_LINK = re.compile(r'^\s*\[')

# Pattern for inline code/python at start of paragraph
STARTS_WITH_INLINE = re.compile(r'^\s*`')

# Pattern for code block start
CODE_BLOCK = re.compile(r'^```')

# Pattern for div fence
DIV_FENCE = re.compile(r'^:::')

# Pattern for YAML frontmatter
YAML_FENCE = re.compile(r'^---\s*$')

# Pattern for callout blocks (need to skip these)
CALLOUT = re.compile(r'^:::\s*\{\.callout')

# Pattern for blank line
BLANK_LINE = re.compile(r'^\s*$')

# Pattern for list item
LIST_ITEM = re.compile(r'^\s*[-*+]|\s*\d+\.')

# Pattern for HTML comment
HTML_COMMENT = re.compile(r'^\s*<!--')

# Pattern for raw LaTeX
RAW_LATEX = re.compile(r'^\s*\\')


def find_first_numbered_h2_para(lines):
    """
    Find the first paragraph after the first numbered H2 in a chapter.

    Returns: (line_number, line_content) or None if not found
    """
    in_frontmatter = False
    in_code_block = False
    in_div = 0  # Track nested divs
    found_chapter = False
    found_numbered_h2 = False

    for i, line in enumerate(lines, 1):
        # Track frontmatter
        if i == 1 and YAML_FENCE.match(line):
            in_frontmatter = True
            continue
        if in_frontmatter:
            if YAML_FENCE.match(line):
                in_frontmatter = False
            continue

        # Track code blocks
        if CODE_BLOCK.match(line):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Track div nesting
        if DIV_FENCE.match(line):
            # Check if it's opening or closing
            stripped = line.strip()
            if stripped == ':::':
                in_div = max(0, in_div - 1)
            elif stripped.startswith(':::'):
                in_div += 1
            continue

        # Skip content inside divs (like callouts, column-margin, etc.)
        if in_div > 0:
            continue

        # Look for chapter header
        if CHAPTER_HEADER.match(line):
            found_chapter = True
            found_numbered_h2 = False  # Reset for this chapter
            continue

        if not found_chapter:
            continue

        # Look for first numbered H2 after chapter header
        if NUMBERED_H2.match(line) and not UNNUMBERED_H2.match(line):
            if not found_numbered_h2:
                found_numbered_h2 = True
            continue

        if not found_numbered_h2:
            continue

        # Skip blank lines, comments, raw LaTeX, list items
        if (BLANK_LINE.match(line) or HTML_COMMENT.match(line) or
            RAW_LATEX.match(line) or LIST_ITEM.match(line)):
            continue

        # Skip H3+ headers
        if line.strip().startswith('#'):
            continue

        # This should be the first paragraph after the numbered H2
        return (i, line)

    return None


def validate_file(filepath):
    """
    Validate a single QMD file for drop cap compatibility.

    Returns: list of (line_number, issue_description) tuples
    """
    issues = []

    try:
        content = Path(filepath).read_text(encoding='utf-8')
    except Exception as e:
        return [(0, f"Could not read file: {e}")]

    lines = content.splitlines()
    result = find_first_numbered_h2_para(lines)

    if result is None:
        # No numbered H2 found or no paragraph after it - that's fine
        return []

    line_num, line_content = result

    # Check for problematic starting patterns
    if STARTS_WITH_CROSSREF.match(line_content):
        match = STARTS_WITH_CROSSREF.match(line_content)
        ref = match.group(0).strip()
        issues.append((
            line_num,
            f"Drop cap paragraph starts with cross-reference '{ref}'. "
            f"The dropcap filter cannot apply lettrine to cross-references. "
            f"Restructure the sentence to start with plain text, e.g., "
            f"'The introduction (@sec-...) posed...' or 'Chapter 1 (@sec-...) posed...'"
        ))
    elif STARTS_WITH_LINK.match(line_content):
        issues.append((
            line_num,
            f"Drop cap paragraph starts with a markdown link. "
            f"The dropcap filter cannot apply lettrine to links. "
            f"Restructure the sentence to start with plain text."
        ))
    elif STARTS_WITH_INLINE.match(line_content):
        issues.append((
            line_num,
            f"Drop cap paragraph starts with inline code/Python. "
            f"The dropcap filter cannot apply lettrine to inline elements. "
            f"Restructure the sentence to start with plain text."
        ))

    return issues


def main():
    """Main entry point."""
    files = sys.argv[1:] if len(sys.argv) > 1 else []

    # If no files provided, scan all QMD files in contents/
    if not files:
        book_root = Path(__file__).resolve().parent.parent.parent.parent / "quarto"
        contents = book_root / "contents"
        if contents.exists():
            files = [str(f) for f in contents.rglob("*.qmd")]

    all_issues = []

    for filepath in files:
        if not filepath.endswith('.qmd'):
            continue
        issues = validate_file(filepath)
        for line_num, description in issues:
            all_issues.append((filepath, line_num, description))

    if all_issues:
        print("❌ Drop cap compatibility issues found:\n")
        for filepath, line_num, description in all_issues:
            # Get relative path for cleaner output
            try:
                rel_path = Path(filepath).relative_to(Path.cwd())
            except ValueError:
                rel_path = filepath
            print(f"  {rel_path}:{line_num}")
            print(f"    {description}\n")
        print(f"Found {len(all_issues)} issue(s). Fix before rendering PDF.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
