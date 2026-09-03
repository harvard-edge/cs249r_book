#!/usr/bin/env python3
"""Close up spaced em dashes per MIT Press style.

Replaces ' — ' (space-emdash-space) with '—' (closed) in body prose,
skipping code fences, YAML frontmatter, LaTeX math blocks, Python cells,
and table separator lines.

Usage:
    python3 fix_emdash.py --dry-run book/quarto/contents/vol1/
    python3 fix_emdash.py book/quarto/contents/vol1/
"""
import argparse
import re
import sys
from pathlib import Path


def is_protected_line(line: str, in_code_fence: bool, in_yaml: bool) -> bool:
    """Return True if this line should NOT be modified."""
    if in_code_fence or in_yaml:
        return True
    stripped = line.lstrip()
    # Python cell directives
    if stripped.startswith("#|"):
        return True
    # All table lines (data + separator) — pipe table prettifier re-spaces cells
    if stripped.startswith("|"):
        return True
    return False


def process_file(filepath: Path, dry_run: bool = False) -> int:
    """Process a single QMD file. Returns count of replacements."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")
    new_lines = []
    count = 0

    in_code_fence = False
    in_yaml = False
    yaml_seen = 0  # track opening/closing ---

    for line in lines:
        stripped = line.strip()

        # Track YAML frontmatter (first --- opens, second --- closes)
        if stripped == "---":
            if yaml_seen == 0:
                in_yaml = True
                yaml_seen = 1
            elif yaml_seen == 1 and in_yaml:
                in_yaml = False
                yaml_seen = 2
            # After frontmatter, --- is just a horizontal rule, not YAML

        # Track code fences (``` with optional language)
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence

        if is_protected_line(line, in_code_fence, in_yaml):
            new_lines.append(line)
            continue

        # Replace spaced em dashes in body prose
        # But skip if inside inline math $...$
        # Simple approach: replace ' — ' with '—' globally on the line,
        # then check we didn't break any inline math
        new_line = line
        if " — " in line:
            # Split by inline math to protect it
            parts = re.split(r"(\$[^$]+\$)", line)
            result_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    # Inside inline math — leave alone
                    result_parts.append(part)
                else:
                    # Body prose — close up em dashes
                    replacements = part.count(" — ")
                    count += replacements
                    result_parts.append(part.replace(" — ", "—"))
            new_line = "".join(result_parts)

        new_lines.append(new_line)

    if count > 0:
        if dry_run:
            print(f"  {filepath}: {count} replacements (dry run)")
        else:
            filepath.write_text("\n".join(new_lines), encoding="utf-8")
            print(f"  {filepath}: {count} replacements applied")

    return count


def main():
    parser = argparse.ArgumentParser(description="Close up spaced em dashes")
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
    print(f"\n{mode}: {total} em dash replacements across {len(files)} files")


if __name__ == "__main__":
    main()
