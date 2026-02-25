#!/usr/bin/env python3
"""
One-shot script: Convert Unicode × to $\\times$ in QMD prose.

Skips:
- Python code blocks (```{python} ... ```)
- TikZ/LaTeX blocks (```{=tex} ... ``` and ```latex ... ```)
- Raw blocks (```{=html} etc.)
- fig-alt attributes (plain text, LaTeX doesn't render)
- Lines that are purely comments (% prefix in TikZ context)

Converts:
- Unicode × (U+00D7) in prose → $\\times$

Usage:
    python3 book/tools/scripts/_archive/convert_unicode_times.py [--dry-run]
"""

import re
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[3]  # book/
CONTENTS = BOOK_ROOT / "quarto" / "contents"

# Patterns
CODE_BLOCK_START = re.compile(r'^```')
PYTHON_BLOCK = re.compile(r'^```\{python\}|^```python')
TIKZ_BLOCK = re.compile(r'^```\{=tex\}|^```latex|^```\{=html\}|^```\{=typst\}')
CODE_BLOCK_END = re.compile(r'^```\s*$')
FIG_ALT = re.compile(r'fig-alt\s*=\s*"')

# Unicode × character
UNICODE_TIMES = '×'


def convert_file(qmd_path: Path, dry_run: bool = False) -> list:
    """Convert Unicode × to $\\times$ in prose lines of a QMD file."""
    text = qmd_path.read_text(encoding='utf-8')
    lines = text.split('\n')
    changes = []
    in_code = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Track code blocks
        if not in_code and CODE_BLOCK_START.match(stripped):
            in_code = True
            continue
        if in_code:
            if CODE_BLOCK_END.match(stripped) and not PYTHON_BLOCK.match(stripped):
                in_code = False
            continue
        
        # Skip if no × on this line
        if UNICODE_TIMES not in line:
            continue
        
        # Handle lines with fig-alt: only preserve × inside fig-alt="..." portion
        if FIG_ALT.search(line):
            # Find the fig-alt="..." span and protect it
            alt_match = re.search(r'(fig-alt\s*=\s*"[^"]*")', line)
            if alt_match:
                before = line[:alt_match.start()]
                alt_text = alt_match.group(1)  # preserve as-is
                after = line[alt_match.end():]
                new_line = (
                    before.replace(UNICODE_TIMES, '$\\times$')
                    + alt_text
                    + after.replace(UNICODE_TIMES, '$\\times$')
                )
            else:
                # fig-alt present but couldn't parse span — skip to be safe
                continue
        else:
            # Normal prose line: replace all ×
            new_line = line.replace(UNICODE_TIMES, '$\\times$')
        
        if new_line != line:
            changes.append((i + 1, line.rstrip(), new_line.rstrip()))
            lines[i] = new_line
    
    if changes and not dry_run:
        qmd_path.write_text('\n'.join(lines), encoding='utf-8')
    
    return changes


def main():
    dry_run = '--dry-run' in sys.argv
    
    qmd_files = sorted(CONTENTS.rglob('*.qmd'))
    total_changes = 0
    
    for qmd in qmd_files:
        changes = convert_file(qmd, dry_run=dry_run)
        if changes:
            rel = qmd.relative_to(BOOK_ROOT.parent)
            print(f"\n{'[DRY RUN] ' if dry_run else ''}{rel} ({len(changes)} lines changed):")
            for line_no, old, new in changes:
                # Show a short context around the change
                print(f"  L{line_no}")
            total_changes += len(changes)
    
    mode = "would change" if dry_run else "changed"
    print(f"\nTotal: {mode} {total_changes} lines across {len(qmd_files)} files scanned.")


if __name__ == '__main__':
    main()
