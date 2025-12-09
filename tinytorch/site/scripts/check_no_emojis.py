#!/usr/bin/env python3
"""
Pre-commit hook: Check that markdown files don't contain emojis.

Emojis cause rendering issues in PDF builds. Keep content professional
by using text descriptions instead.

Usage:
    python3 check_no_emojis.py [files...]

Exit codes:
    0 - No emojis found
    1 - Emojis found (lists files and emojis)
"""

import sys
import re
from pathlib import Path

# Emoji pattern - matches most common emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # Misc Symbols, Emoticons, Dingbats, etc.
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001FA00-\U0001FAFF"  # Extended symbols
    "]",
    flags=re.UNICODE
)

# Allowed characters:
# - 🔥 Fire emoji for Tiny🔥Torch branding
# - ✓ Checkmark (renders fine in most fonts, used in code examples)
# - ✗ X mark (renders fine in most fonts)
ALLOWED_EMOJIS = {'🔥', '✓', '✗', '×'}

def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a file for emojis. Returns list of (line_num, emoji, line_content)."""
    issues = []
    try:
        content = filepath.read_text(encoding='utf-8')
        for line_num, line in enumerate(content.splitlines(), 1):
            for match in EMOJI_PATTERN.finditer(line):
                emoji = match.group()
                if emoji not in ALLOWED_EMOJIS:
                    issues.append((line_num, emoji, line.strip()[:60]))
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
    return issues

def main():
    if len(sys.argv) < 2:
        print("Usage: check_no_emojis.py <file1> [file2] ...")
        sys.exit(0)

    files = [Path(f) for f in sys.argv[1:]]
    all_issues = {}

    for filepath in files:
        if filepath.suffix in ('.md', '.qmd'):
            issues = check_file(filepath)
            if issues:
                all_issues[filepath] = issues

    if all_issues:
        print("❌ Emojis found in markdown files (not allowed for PDF compatibility):\n")
        for filepath, issues in all_issues.items():
            print(f"  {filepath}:")
            for line_num, emoji, context in issues:
                print(f"    Line {line_num}: {emoji} - \"{context}...\"")
            print()
        print("Fix: Remove emojis or replace with text descriptions.")
        print("Note: 🔥 is allowed only for Tiny🔥Torch branding.")
        sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()
