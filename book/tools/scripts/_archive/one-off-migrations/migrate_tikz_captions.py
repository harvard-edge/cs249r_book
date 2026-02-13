#!/usr/bin/env python3
"""
Migrate TikZ figure captions from markdown format to fig-cap attribute.

Converts:
    ::: {#fig-id fig-env="figure" fig-pos="htb"}
    ```{.tikz}
    [tikz code]
    ```
    **Caption text here**
    :::

To:
    ::: {#fig-id fig-cap="Caption text here" fig-env="figure" fig-pos="htb"}
    ```{.tikz}
    [tikz code]
    ```
    :::
"""

import re
import sys
from pathlib import Path
from typing import Tuple, List


def extract_caption_text(caption_line: str) -> str:
    """
    Extract caption text from markdown bold format.
    Handles: **Caption**: Rest of text
    """
    # Remove leading/trailing whitespace
    caption_line = caption_line.strip()

    # Pattern 1: **Bold Title**: Rest of caption
    if caption_line.startswith("**") and "**:" in caption_line:
        # Extract full caption including bold title
        match = re.match(r'\*\*(.+?)\*\*:\s*(.+)', caption_line)
        if match:
            title = match.group(1)
            rest = match.group(2)
            return f"**{title}**: {rest}"

    # Pattern 2: Just **Bold text**
    elif caption_line.startswith("**") and caption_line.endswith("**"):
        return caption_line

    # Pattern 3: Mixed bold and regular text
    else:
        return caption_line


def escape_for_attribute(text: str) -> str:
    """
    Escape text for use in a Quarto attribute value.
    Handle quotes, special chars, but preserve markdown formatting.
    """
    # Escape double quotes
    text = text.replace('"', '\\"')
    # Don't escape single quotes inside double-quoted attribute
    return text


def migrate_tikz_caption(content: str) -> Tuple[str, int]:
    """
    Migrate TikZ captions from markdown to fig-cap attribute.

    Returns:
        Tuple of (modified_content, number_of_migrations)
    """
    migrations = 0

    # Pattern to match TikZ figures with captions
    # Captures:
    # 1. Opening ::: with attributes
    # 2. The tikz code block
    # 3. The caption line (markdown bold text)
    # 4. Any content before closing :::
    pattern = re.compile(
        r'(:::[ \t]+\{#fig-[^\}]+)\}'  # Opening with attributes (no closing })
        r'(.*?```\{\.tikz\}.*?```)'     # TikZ code block
        r'\s*\n'                         # Newline after code block
        r'\s*(\*\*.+?\*\*[^\n]*)'       # Caption line starting with **
        r'(.*?)'                         # Any content before closing
        r'\n:::',                        # Closing :::
        re.DOTALL
    )

    def replace_caption(match):
        nonlocal migrations

        opening_attrs = match.group(1)  # ::: {#fig-id ...
        tikz_code = match.group(2)       # ```{.tikz}...```
        caption_line = match.group(3)    # **Caption**
        between_content = match.group(4) # Content between caption and :::

        # Extract caption text
        caption_text = extract_caption_text(caption_line)

        # Escape for attribute
        caption_escaped = escape_for_attribute(caption_text)

        # Build new structure
        # Add fig-cap right after the #fig-id
        new_opening = f'{opening_attrs} fig-cap="{caption_escaped}"}}'

        # Check if there's any content between caption and :::
        # If it's just whitespace, ignore it
        if between_content.strip():
            # Keep the content (might be important)
            result = f'{new_opening}{tikz_code}\n{between_content}\n:::'
        else:
            # Clean output - just code block
            result = f'{new_opening}{tikz_code}\n:::'

        migrations += 1
        return result

    modified_content = pattern.sub(replace_caption, content)

    return modified_content, migrations


def process_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Process a single file.

    Returns:
        True if file was modified, False otherwise
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        modified_content, migrations = migrate_tikz_caption(content)

        if migrations > 0:
            print(f"  Found {migrations} TikZ caption(s) to migrate")

            if not dry_run:
                file_path.write_text(modified_content, encoding='utf-8')
                print(f"  ✓ Updated {file_path}")
            else:
                print(f"  [DRY RUN] Would update {file_path}")

            return True

        return False

    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False


def find_tikz_figures(content: str) -> List[str]:
    """Find all TikZ figure IDs in content for validation."""
    pattern = re.compile(r':::[ \t]+\{#(fig-[^\s\}]+)')
    return pattern.findall(content)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Migrate TikZ captions to fig-cap attribute')
    parser.add_argument('files', nargs='*', help='Files to process (default: all .qmd files)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    parser.add_argument('--test', metavar='FILE', help='Test on a single file and show diff')

    args = parser.parse_args()

    # Test mode: process one file and show changes
    if args.test:
        test_file = Path(args.test)
        if not test_file.exists():
            print(f"Error: Test file {test_file} does not exist")
            return 1

        print(f"Testing on: {test_file}")
        print("=" * 60)

        original = test_file.read_text(encoding='utf-8')
        modified, migrations = migrate_tikz_caption(original)

        if migrations == 0:
            print("No TikZ captions found to migrate in this file.")
            return 0

        print(f"\nFound {migrations} caption(s) to migrate\n")

        # Show diff
        import difflib
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=str(test_file),
            tofile=f"{test_file} (modified)",
            lineterm=''
        )

        print("Diff preview:")
        print("=" * 60)
        for line in diff:
            print(line)

        return 0

    # Determine files to process
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        # Process all .qmd files in vol1 and vol2
        repo_root = Path(__file__).resolve().parents[3]
        vol1_dir = repo_root / "book" / "quarto" / "contents" / "vol1"
        vol2_dir = repo_root / "book" / "quarto" / "contents" / "vol2"

        files = []
        for vol_dir in [vol1_dir, vol2_dir]:
            if vol_dir.exists():
                files.extend(vol_dir.glob("**/*.qmd"))

    # Filter to only content chapters (exclude frontmatter, backmatter)
    content_files = [
        f for f in files
        if f.exists() and 'frontmatter' not in str(f) and 'backmatter' not in str(f)
    ]

    if not content_files:
        print("No files to process")
        return 1

    print(f"Processing {len(content_files)} file(s)...")
    print("=" * 60)

    modified_count = 0
    for file_path in sorted(content_files):
        # Convert to absolute path if needed
        file_path = file_path.resolve()
        try:
            relative_path = file_path.relative_to(Path.cwd())
        except ValueError:
            relative_path = file_path
        print(f"\n{relative_path}:")
        if process_file(file_path, dry_run=args.dry_run):
            modified_count += 1

    print("\n" + "=" * 60)
    print(f"Summary: Modified {modified_count}/{len(content_files)} file(s)")

    if args.dry_run:
        print("\nThis was a dry run. No files were actually modified.")
        print("Run without --dry-run to apply changes.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
