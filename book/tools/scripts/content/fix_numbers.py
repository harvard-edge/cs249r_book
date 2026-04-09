#!/usr/bin/env python3
"""Spell out single digits 1–9 in body prose per MIT Press style.

Replaces bare digits 1–9 with their spelled-out equivalents in body prose,
skipping protected contexts: code fences, YAML frontmatter, LaTeX math,
Python cells, table rows, HTML tags, figure/div attributes, index entries,
version strings, numbers with units, ranges, decimals, and multi-digit numbers.

Usage:
    python3 fix_numbers.py --dry-run book/quarto/contents/vol1/
    python3 fix_numbers.py book/quarto/contents/vol1/
"""
import argparse
import re
import sys
from pathlib import Path

DIGIT_WORDS = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

# Unit abbreviations that, when following a digit + space, mean "do not spell out"
UNIT_PATTERN = (
    r"(?:GB|TB|MB|KB|kB|PB|EB|GiB|TiB|MiB|KiB|"
    r"ms|ns|μs|us|MHz|GHz|THz|Hz|"
    r"TFLOPS|TFLOP|GFLOPS|GFLOP|PFLOPS|PFLOP|FLOPS|FLOP|"
    r"FLOPs|TOPS|TOPs|"
    r"fps|FPS|"
    r"W|mW|kW|MW|"
    r"nm|mm|cm|m|km|"
    r"bps|Gbps|Mbps|Kbps|"
    r"px|dpi|dB|"
    r"bits?|bytes?|B)\b"
)

# Capitalized words that, when immediately preceding a digit, indicate a numbered
# label/identifier (e.g. "Phase 1", "Layer 3", "Step 5") — do NOT spell out.
NUMBERED_LABEL_WORDS = {
    "figure", "fig", "table", "tbl", "equation", "eq", "chapter", "section",
    "step", "phase", "stage", "layer", "level", "tier", "type", "class",
    "category", "group", "mode", "part", "rule", "principle", "property",
    "case", "option", "variant", "version", "item", "entry", "row",
    "column", "col", "dimension", "dim", "axis", "index", "rank",
    "epoch", "iteration", "round", "pass", "batch", "block",
    "node", "worker", "gpu", "core", "slot", "port", "channel",
    # Product/brand names that take numeric identifiers
    "geforce", "snapdragon", "exynos", "dimensity", "cortex", "neoverse",
    "gen", "pixel", "iphone", "galaxy", "jetson", "orin", "xavier",
    "threadripper", "xeon", "epyc", "ryzen", "adreno", "mali",
    "wifi", "bluetooth", "usb", "pcie", "thunderbolt", "displayport",
    "sdxl", "llama", "gpt", "palm", "gemini", "claude", "mixtral",
    "wave", "zone", "area", "region", "sector", "quadrant",
    "arm", "v", "mark", "revision", "rev", "release", "r",
}

# Pattern to detect version-like contexts: digit preceded by hyphen (GPT-3, ResNet-5, v2)
# or followed by a dot and another digit (3.14, 0.5)


def is_protected_line(line: str, in_code_fence: bool, in_yaml: bool,
                      in_math_block: bool) -> bool:
    """Return True if this line should NOT be modified."""
    if in_code_fence or in_yaml or in_math_block:
        return True
    stripped = line.lstrip()
    # Python cell directives
    if stripped.startswith("#|"):
        return True
    # Table rows (pipe tables)
    if stripped.startswith("|"):
        return True
    # HTML tags
    if stripped.startswith("<") and not stripped.startswith("<!-"):
        return True
    # Div attributes
    if stripped.startswith(":::"):
        return True
    # LaTeX commands (marginfigure, begin, end, etc.)
    if stripped.startswith("\\"):
        return True
    # fig-cap, fig-alt, tbl-cap attributes (sometimes on their own line)
    if re.match(r'^(fig-cap|fig-alt|tbl-cap)\s*[:=]', stripped):
        return True
    # Formula/calculation lines (bulleted items that are clearly math)
    if re.match(r'^-\s*\*{0,2}(FLOPs|Bytes|Operations|Arithmetic Intensity|Energy|Data Movement)\*{0,2}\s*:', stripped):
        return True
    return False


def replace_digits_in_line(line: str) -> tuple[str, int]:
    """Replace single digits 1-9 in prose contexts within a line.

    Returns (new_line, count_of_replacements).
    """
    count = 0

    # We process the line character by character, tracking inline math ($...$)
    # and other protected inline contexts.

    # First, identify all protected spans in the line
    protected_spans = []

    # Inline math: $...$ (but not $$)
    # Find all $...$ spans (non-greedy)
    for m in re.finditer(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', line):
        protected_spans.append((m.start(), m.end()))

    # Display math on single line: $$...$$
    for m in re.finditer(r'\$\$.+?\$\$', line):
        protected_spans.append((m.start(), m.end()))

    # LaTeX math with \(...\) notation
    for m in re.finditer(r'\\\(.+?\\\)', line):
        protected_spans.append((m.start(), m.end()))

    # Inline code: `...`
    for m in re.finditer(r'`[^`]+`', line):
        protected_spans.append((m.start(), m.end()))

    # \index{...} entries
    for m in re.finditer(r'\\index\{[^}]*\}', line):
        protected_spans.append((m.start(), m.end()))

    # HTML tags: <...>
    for m in re.finditer(r'<[^>]+>', line):
        protected_spans.append((m.start(), m.end()))

    # Cross-references: @sec-..., @fig-..., @tbl-..., @eq-..., @lst-...
    for m in re.finditer(r'@(?:sec|fig|tbl|eq|lst)-[\w-]+', line):
        protected_spans.append((m.start(), m.end()))

    # Citation keys: [@...] or @key
    for m in re.finditer(r'\[@[^\]]+\]', line):
        protected_spans.append((m.start(), m.end()))

    # {python} inline expressions
    for m in re.finditer(r'`\{python\}[^`]*`', line):
        protected_spans.append((m.start(), m.end()))

    # Curly-brace attributes: {#sec-..., .callout-..., etc.}
    for m in re.finditer(r'\{[^}]*\}', line):
        protected_spans.append((m.start(), m.end()))

    # Tuple/shape notation: (3, 3, 3) or (Height, Width, 3)
    for m in re.finditer(r'\([^)]*\d[^)]*,\s*[^)]*\)', line):
        protected_spans.append((m.start(), m.end()))

    # fig-cap="..." and fig-alt="..." attribute values
    for m in re.finditer(r'(?:fig-cap|fig-alt|tbl-cap)\s*=\s*"[^"]*"', line):
        protected_spans.append((m.start(), m.end()))

    # Sort and merge spans
    protected_spans.sort()

    def is_protected_pos(pos: int) -> bool:
        for start, end in protected_spans:
            if start <= pos < end:
                return True
            if start > pos:
                break
        return False

    # Now find all single digits and decide whether to replace
    # Pattern: a single digit 1-9 that is:
    # - not preceded by another digit, letter, hyphen, period, $, @, #, or backslash
    # - not followed by another digit, period, hyphen+letter (version), $, %, ×, or \times
    result = []
    i = 0
    while i < len(line):
        if is_protected_pos(i):
            result.append(line[i])
            i += 1
            continue

        # Check if we're at a single digit 1-9
        ch = line[i]
        if ch in DIGIT_WORDS:
            # Check preceding character
            if i > 0:
                prev = line[i - 1]
                # Skip if preceded by digit, letter, hyphen, period, $, @, #, \, /, =, [, ], +, ~, ^
                if prev.isalnum() or prev in '-.$@#\\/=[]+~^/':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if preceded by en-dash or em-dash (range like 3–5, the 5 part)
                if prev in '–—':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if preceded by -- (double hyphen range)
                if prev == '-' and i >= 2 and line[i - 2] == '-':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if preceded by hyphen and that hyphen is preceded by a digit (range like 3-4, the 4 part)
                if prev == '-' and i >= 2 and line[i - 2].isdigit():
                    result.append(ch)
                    i += 1
                    continue
                # Skip if preceded by "= " or "+ " or "$ " (formula/arithmetic/math context)
                if prev == ' ' and i >= 2 and line[i - 2] in '=+$':
                    result.append(ch)
                    i += 1
                    continue

            # Check following character
            if i + 1 < len(line):
                nxt = line[i + 1]
                # Skip if followed by digit (multi-digit number)
                if nxt.isdigit():
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by period and then digit (decimal: 3.14)
                if nxt == '.' and i + 2 < len(line) and line[i + 2].isdigit():
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by just period (e.g. ordered list "1. item")
                if nxt == '.':
                    # Check if this looks like a markdown ordered list item
                    after_dot = line[i + 2:i + 3] if i + 2 < len(line) else ''
                    if after_dot == ' ' or after_dot == '' or after_dot == '\t':
                        result.append(ch)
                        i += 1
                        continue
                # Skip if followed by $ (math context)
                if nxt == '$':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by % (percentage like 5%)
                if nxt == '%':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by × or \times (multiplier)
                if nxt == '×':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by \times
                if line[i + 1:i + 7] == '\\times':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by – or — (en-dash/em-dash range)
                if nxt in '–—':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by -- (double-hyphen, used as en-dash in LaTeX)
                if nxt == '-' and i + 2 < len(line) and line[i + 2] == '-':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by hyphen then digit (range like 3-4)
                if nxt == '-' and i + 2 < len(line) and line[i + 2].isdigit():
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by a letter directly (part of identifier like H100)
                if nxt.isalpha():
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by ], [, =, or / (array indexing / formula / fraction)
                if nxt in ']=[]/' :
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by : then digit (ratio like 3:1)
                if nxt == ':' and i + 2 < len(line) and line[i + 2].isdigit():
                    result.append(ch)
                    i += 1
                    continue
                # Skip if preceded by : (second part of ratio like 3:1)
                if i > 0 and line[i - 1] == ':':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by ) — parenthetical numbering like (3)
                if nxt == ')':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if preceded by ( — parenthetical numbering like (3)
                if i > 0 and line[i - 1] == '(':
                    result.append(ch)
                    i += 1
                    continue
                # Skip if followed by space then unit abbreviation
                if nxt == ' ' and i + 2 < len(line):
                    rest = line[i + 2:]
                    if re.match(UNIT_PATTERN, rest):
                        result.append(ch)
                        i += 1
                        continue
                    # Skip if followed by " /" (division, formula like "1 / ((1-p)...")
                    if rest[0] == '/':
                        result.append(ch)
                        i += 1
                        continue
                # Skip if followed by comma then digits (like 1,000)
                if nxt == ',' and i + 2 < len(line) and line[i + 2].isdigit():
                    result.append(ch)
                    i += 1
                    continue
            else:
                # Digit at end of line with nothing after — likely a reference number
                # or standalone; skip to be safe
                result.append(ch)
                i += 1
                continue

            # Check if preceded by a numbered label word (e.g. "Layer 3", "Step 5")
            if i > 1 and line[i - 1] == ' ':
                # Walk back to find the preceding word
                j = i - 2
                while j >= 0 and line[j].isalpha():
                    j -= 1
                prev_word = line[j + 1:i - 1].lower()
                if prev_word in NUMBERED_LABEL_WORDS:
                    result.append(ch)
                    i += 1
                    continue

            # All checks passed — spell it out
            word = DIGIT_WORDS[ch]
            result.append(word)
            count += 1
            i += 1
        else:
            result.append(ch)
            i += 1

    return ''.join(result), count


def process_file(filepath: Path, dry_run: bool = False) -> int:
    """Process a single QMD file. Returns count of replacements."""
    text = filepath.read_text(encoding="utf-8")
    lines = text.split("\n")
    new_lines = []
    count = 0

    in_code_fence = False
    in_yaml = False
    in_math_block = False
    yaml_seen = 0  # track opening/closing ---

    for line in lines:
        stripped = line.strip()

        # Track YAML frontmatter (first --- opens, second --- closes)
        if stripped == "---":
            if yaml_seen == 0:
                in_yaml = True
                yaml_seen = 1
                new_lines.append(line)
                continue
            elif yaml_seen == 1 and in_yaml:
                in_yaml = False
                yaml_seen = 2
                new_lines.append(line)
                continue

        # Track code fences (``` with optional language)
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            new_lines.append(line)
            continue

        # Track LaTeX environment blocks (\begin{...} / \end{...})
        if re.match(r'\\begin\{', stripped):
            in_math_block = True
            new_lines.append(line)
            continue
        if re.match(r'\\end\{', stripped):
            in_math_block = False
            new_lines.append(line)
            continue

        # Track display math blocks ($$)
        if stripped == "$$" or stripped.startswith("$$") and not stripped.endswith("$$"):
            # Toggle math block if line is just $$ or starts with $$ but doesn't close
            if stripped == "$$":
                in_math_block = not in_math_block
                new_lines.append(line)
                continue

        # Skip protected lines entirely
        if is_protected_line(line, in_code_fence, in_yaml, in_math_block):
            new_lines.append(line)
            continue

        # Process this line
        new_line, line_count = replace_digits_in_line(line)
        new_lines.append(new_line)
        count += line_count

    if count > 0 and not dry_run:
        filepath.write_text("\n".join(new_lines), encoding="utf-8")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Spell out single digits 1-9 in body prose (MIT Press style)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="QMD files or directories to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without modifying files.",
    )
    args = parser.parse_args()

    # Collect all QMD files
    qmd_files: list[Path] = []
    for p in args.paths:
        if p.is_file() and p.suffix == ".qmd":
            qmd_files.append(p)
        elif p.is_dir():
            qmd_files.extend(sorted(p.rglob("*.qmd")))
        else:
            print(f"Warning: skipping {p}", file=sys.stderr)

    total = 0
    for f in qmd_files:
        n = process_file(f, dry_run=args.dry_run)
        if n > 0:
            label = "would change" if args.dry_run else "changed"
            print(f"  {f.name}: {n} replacements {label}")
            total += n

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"\n[{mode}] Total: {total} replacements across {len(qmd_files)} files.")


if __name__ == "__main__":
    main()
