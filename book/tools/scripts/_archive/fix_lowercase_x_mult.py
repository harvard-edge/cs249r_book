#!/usr/bin/env python3
"""Fix lowercase 'x' used as multiplication in prose.

Converts patterns like:
  - "1000x faster"  -> "1000$\\times$ faster"
  - "`str`x speedup" -> "`str` $\\times$ speedup"
  - "`str`x)"        -> "`str` $\\times$)"

Preserves 'x' in:
  - Code blocks (``` ... ```)
  - fig-alt attributes
  - \\index entries
  - Hardware counts: "8x A100", "8x H100", "4x RAID", "2x AA"

Usage:
  python3 fix_lowercase_x_mult.py --dry-run   # preview changes
  python3 fix_lowercase_x_mult.py              # apply changes
"""

import re
import sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv

# --- False positive exclusions ---
# Hardware counts and configurations where "Nx" means "N units of"
FALSE_POSITIVE_AFTER = re.compile(
    r"^x\s+(?:A100|H100|A10|V100|T4|L4|L40|B200|GB200|TPU|RAID|AA\b)",
    re.IGNORECASE,
)

# --- Replacement patterns ---

# Pattern 1: digit(s) + x + space + word
# e.g. "1000x faster" -> "1000$\times$ faster"
DIGIT_X_WORD = re.compile(r"(\d)(x)(\s+\w)")

# Pattern 2: backtick + x + space + word
# e.g. "`x speedup" -> "` $\times$ speedup"  (from `str`x speedup)
BACKTICK_X_WORD = re.compile(r"(`)(x)(\s+\w)")

# Pattern 3: backtick + x + closing paren
# e.g. "`x)" -> "` $\times$)"  (from `str`x))
BACKTICK_X_PAREN = re.compile(r"(`)(x)(\))")

FIG_ALT = re.compile(r'fig-alt\s*=\s*"')

QMD_ROOT = Path(__file__).resolve().parents[3] / "quarto" / "contents"


def _replace_digit_x(m: re.Match) -> str:
    """Replace digit-x-word, checking for false positives."""
    # Check what follows the digit: is the rest a false positive?
    full_after = m.group(2) + m.group(3)  # "x word..."
    # Get more context from the original string
    after_start = m.start(2)
    line = m.string
    after_text = line[after_start:]
    if FALSE_POSITIVE_AFTER.match(after_text):
        return m.group(0)  # keep as-is
    return m.group(1) + "$\\times$" + m.group(3)


def _replace_backtick_x_word(m: re.Match) -> str:
    """Replace backtick-x-word."""
    return m.group(1) + " $\\times$" + m.group(3)


def _replace_backtick_x_paren(m: re.Match) -> str:
    """Replace backtick-x-paren."""
    return m.group(1) + " $\\times$" + m.group(3)


def fix_line(line: str) -> str:
    """Apply all replacement patterns to a single line."""
    result = DIGIT_X_WORD.sub(_replace_digit_x, line)
    result = BACKTICK_X_WORD.sub(_replace_backtick_x_word, result)
    result = BACKTICK_X_PAREN.sub(_replace_backtick_x_paren, result)
    return result


def process_file(filepath: Path) -> int:
    """Process a single .qmd file. Returns number of lines changed."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    changes = 0
    in_code = False

    new_lines = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track code blocks
        if stripped.startswith("```"):
            in_code = not in_code
            new_lines.append(line)
            continue
        if in_code:
            new_lines.append(line)
            continue

        # Skip fig-alt lines and \index entries
        if FIG_ALT.search(line) or stripped.startswith("\\index"):
            new_lines.append(line)
            continue

        new_line = fix_line(line)
        if new_line != line:
            changes += 1
            rel = filepath.relative_to(QMD_ROOT)
            print(f"  {rel}:{i}")
            print(f"    OLD: {line.rstrip()[:140]}")
            print(f"    NEW: {new_line.rstrip()[:140]}")

        new_lines.append(new_line)

    if changes > 0 and not DRY_RUN:
        filepath.write_text("".join(new_lines), encoding="utf-8")

    return changes


def main():
    mode = "DRY RUN" if DRY_RUN else "APPLYING"
    print(f"=== {mode}: Fix lowercase 'x' multiplication ===\n")

    total_changes = 0
    files_changed = 0

    for f in sorted(QMD_ROOT.rglob("*.qmd")):
        n = process_file(f)
        if n > 0:
            files_changed += 1
            total_changes += n

    print(f"\n{'Would change' if DRY_RUN else 'Changed'}: "
          f"{total_changes} lines in {files_changed} files")


if __name__ == "__main__":
    main()
