#!/usr/bin/env python3
"""Convert '%' to 'percent' in body prose per MIT Press style.

Handles three contexts:
  1. Hard-coded: '94%' → '94 percent'
  2. Inline Python followed by %: '`{python} x.pct_str`%' → '`{python} x.pct_str` percent'
  3. Leaves alone: tables, code fences, LaTeX math, YAML, Python cells, fig-cap, fig-alt

Usage:
    python3 fix_percent.py --dry-run book/quarto/contents/vol1/
    python3 fix_percent.py book/quarto/contents/vol1/
"""
import argparse
import re
import sys
from pathlib import Path


def is_table_line(line: str) -> bool:
    """True if this line is a markdown table row."""
    stripped = line.strip()
    return stripped.startswith("|")


def is_protected_line(line: str, in_code_fence: bool, in_yaml: bool) -> bool:
    """Return True if this line should NOT be modified."""
    if in_code_fence or in_yaml:
        return True
    stripped = line.lstrip()
    # Python cell directives
    if stripped.startswith("#|"):
        return True
    # Div attributes
    if stripped.startswith("::: {") or stripped.startswith(":::"):
        return True
    return False


def is_html_line(line: str) -> bool:
    """True if this line contains HTML markup (not body prose)."""
    stripped = line.strip()
    return (stripped.startswith("<") or
            'width="' in line or
            'style="' in line or
            stripped.startswith("<!--"))


def fix_percent_in_line(line: str) -> tuple[str, int]:
    """Fix percent symbols in a single body prose line.

    Returns (new_line, replacement_count).
    Skips:
      - Inside inline math $...$
      - Inside LaTeX \\%
      - Inside fig-cap= or fig-alt= attributes
      - Table lines (handled separately)
      - HTML lines
    """
    if is_table_line(line):
        return line, 0

    if is_html_line(line):
        return line, 0

    # Check for fig-cap= or fig-alt= — skip the whole line
    if "fig-cap=" in line or "fig-alt=" in line:
        return line, 0

    count = 0

    # Split by inline math to protect it
    # Also protect display math (lines starting with $$)
    if line.strip().startswith("$$"):
        return line, 0

    parts = re.split(r"(\$[^$]+\$)", line)
    result_parts = []

    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside inline math — leave alone
            result_parts.append(part)
        else:
            # Body prose segment
            new_part = part

            # Pattern 1: inline Python followed by %
            # `{python} something`%  →  `{python} something` percent
            pattern_inline = r"(`\{python\}[^`]+`)%"
            matches = re.findall(pattern_inline, new_part)
            count += len(matches)
            new_part = re.sub(pattern_inline, r"\1 percent", new_part)

            # Pattern 2: hard-coded number followed by %
            # But NOT \% (LaTeX escaped percent)
            # 94% → 94 percent
            # 94.5% → 94.5 percent
            pattern_hard = r"(?<!\\)(\d+\.?\d*)%"
            matches2 = re.findall(pattern_hard, new_part)
            count += len(matches2)
            new_part = re.sub(pattern_hard, r"\1 percent", new_part)

            result_parts.append(new_part)

    return "".join(result_parts), count


def process_file(filepath: Path, dry_run: bool = False) -> int:
    """Process a single QMD file. Returns count of replacements."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")
    new_lines = []
    total = 0

    in_code_fence = False
    in_yaml = False
    in_html_block = False
    yaml_seen = 0

    for line in lines:
        stripped = line.strip()

        # Track YAML frontmatter
        if stripped == "---":
            if yaml_seen == 0:
                in_yaml = True
                yaml_seen = 1
            elif yaml_seen == 1 and in_yaml:
                in_yaml = False
                yaml_seen = 2

        # Track code fences
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence

        # Track HTML blocks (<style>, <script>, etc.)
        if re.match(r"<(style|script)\b", stripped, re.IGNORECASE):
            in_html_block = True
        if re.match(r"</(style|script)>", stripped, re.IGNORECASE):
            in_html_block = False
            new_lines.append(line)
            continue

        if is_protected_line(line, in_code_fence, in_yaml) or in_html_block:
            new_lines.append(line)
            continue

        new_line, count = fix_percent_in_line(line)
        total += count
        new_lines.append(new_line)

    if total > 0:
        if dry_run:
            print(f"  {filepath}: {total} replacements (dry run)")
        else:
            filepath.write_text("\n".join(new_lines), encoding="utf-8")
            print(f"  {filepath}: {total} replacements applied")

    return total


def main():
    parser = argparse.ArgumentParser(description="Convert % to 'percent' in body prose")
    parser.add_argument("path", type=Path, help="Directory or file to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview without applying")
    args = parser.parse_args()

    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.rglob("*.qmd"))

    total = 0
    for f in files:
        total += process_file(f, dry_run=args.dry_run)

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"\n{mode}: {total} percent replacements across {len(files)} files")


if __name__ == "__main__":
    main()
