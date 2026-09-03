#!/usr/bin/env python3
"""Lowercase concept terms per MIT Press style.

Handles specific terms the copy editor flagged, with careful context
awareness to avoid breaking:
- Sentence starts
- Bold definitions (**Term**)
- Triple bold definitions (***Term***)
- Section headers (## ...)
- Callout titles (title="...")
- Index entries (\index{...})
- Table headers
- Code fences, Python cells, YAML, LaTeX math

Usage:
    python3 fix_capitalization.py --dry-run book/quarto/contents/vol1/
    python3 fix_capitalization.py book/quarto/contents/vol1/
"""
import argparse
import re
import sys
from pathlib import Path

# Terms to lowercase and their lowercase forms
# Only terms explicitly flagged by the copy editor
TERMS = {
    "Iron Law": "iron law",
    "Degradation Equation": "degradation equation",
    "Verification Gap": "verification gap",
    "ML Node": "ML node",
    "Bitter Lesson": "bitter lesson",
    "Data Wall": "data wall",
    "Compute Wall": "compute wall",
    "Memory Wall": "memory wall",
    "Power Wall": "power wall",
    "Energy Corollary": "energy corollary",
}

# Contexts where capitalization should be preserved
def should_skip_line(line: str, in_code_fence: bool, in_yaml: bool, in_display_math: bool) -> bool:
    """Return True if this line should not be modified."""
    if in_code_fence or in_yaml or in_display_math:
        return True
    stripped = line.lstrip()
    # Display math line (standalone $$)
    if stripped.startswith("$$"):
        return True
    # Python cell directives
    if stripped.startswith("#|"):
        return True
    # Section headers (headline style)
    if stripped.startswith("## ") or stripped.startswith("# "):
        return True
    # Div attributes
    if stripped.startswith(":::"):
        return True
    # Table header rows (bold)
    if stripped.startswith("|") and "**" in stripped:
        return True
    return False


def should_skip_match(line: str, match_start: int, match_end: int, term: str) -> bool:
    """Return True if this specific match should be preserved."""
    # Check if inside bold definition: **Term** or ***Term***
    before = line[:match_start]
    after = line[match_end:]
    if before.endswith("**") or before.endswith("***"):
        return True
    if after.startswith("**") or after.startswith("***"):
        return True

    # Check if at start of sentence
    # Look backwards for sentence boundary
    before_stripped = before.rstrip()
    if not before_stripped:
        return True  # Start of line
    last_char = before_stripped[-1]
    if last_char in ".!?:":
        return True  # After sentence-ending punctuation

    # Check if inside inline math $...$
    # Find all math spans and check if our match falls inside one
    in_math = False
    math_depth = 0
    for i, ch in enumerate(line):
        if ch == '$' and (i == 0 or line[i-1] != '\\'):
            if not in_math:
                in_math = True
                math_start = i
            else:
                # End of math span
                if math_start < match_start and i >= match_end:
                    return True  # Inside inline math
                in_math = False

    # Check if inside \index{...}
    idx_start = line.rfind("\\index{", 0, match_start)
    if idx_start >= 0:
        idx_end = line.find("}", idx_start)
        if idx_end >= match_end:
            return True  # Inside \index{...}

    # Check if "Iron Law of Processor Performance" (H&P reference — keep caps)
    if term == "Iron Law":
        # Look at surrounding text — if it's "Iron Law of Processor Performance", keep it
        after_text = line[match_end:match_end+30]
        if after_text.startswith(" of Processor Performance"):
            return True

    # Check if inside title="..."
    title_start = line.rfind('title="', 0, match_start)
    if title_start >= 0:
        title_end = line.find('"', title_start + 7)
        if title_end >= match_end:
            return True  # Inside title="..."

    # Check if inside fig-cap="..." or fig-alt="..."
    for attr in ['fig-cap="', 'fig-alt="']:
        attr_start = line.rfind(attr, 0, match_start)
        if attr_start >= 0:
            attr_end = line.find('"', attr_start + len(attr))
            if attr_end >= match_end:
                return True

    # Check if referring to Sutton's essay title "The Bitter Lesson"
    if term == "Bitter Lesson":
        # If preceded by "The " or "Sutton's " or followed by citation, keep capitalized
        if '"The Bitter Lesson"' in line or '"Bitter Lesson"' in line:
            return True
        if "Sutton" in line:
            return True

    return False


def process_file(filepath: Path, dry_run: bool = False) -> int:
    """Process a single QMD file. Returns count of replacements."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")
    new_lines = []
    total = 0

    in_code_fence = False
    in_yaml = False
    in_display_math = False
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

        # Track display math blocks ($$...$$)
        if stripped.startswith("$$") and not in_code_fence:
            # Count $$ occurrences on this line
            dd_count = stripped.count("$$")
            if dd_count >= 2:
                # Single-line display math — skip this line but don't change state
                pass
            else:
                in_display_math = not in_display_math

        if should_skip_line(line, in_code_fence, in_yaml, in_display_math):
            new_lines.append(line)
            continue

        new_line = line
        for term, replacement in TERMS.items():
            # Find all occurrences
            offset = 0
            while True:
                idx = new_line.find(term, offset)
                if idx == -1:
                    break

                # Check if this match should be preserved
                if should_skip_match(new_line, idx, idx + len(term), term):
                    offset = idx + len(term)
                    continue

                # Apply the replacement
                new_line = new_line[:idx] + replacement + new_line[idx + len(term):]
                total += 1
                offset = idx + len(replacement)

        new_lines.append(new_line)

    if total > 0:
        if dry_run:
            print(f"  {filepath}: {total} replacements (dry run)")
        else:
            filepath.write_text("\n".join(new_lines), encoding="utf-8")
            print(f"  {filepath}: {total} replacements applied")

    return total


def main():
    parser = argparse.ArgumentParser(description="Lowercase concept terms")
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
    print(f"\n{mode}: {total} capitalization replacements across {len(files)} files")


if __name__ == "__main__":
    main()
