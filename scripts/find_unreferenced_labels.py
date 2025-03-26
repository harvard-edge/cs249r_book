#!/usr/bin/env python3
"""
🔍 Find Unreferenced Labels in Quarto Files

This script identifies defined labels (e.g., {#fig-xyz}) in .qmd files that are
never referenced (e.g., @fig-xyz). It's useful for cleaning up figures, tables,
sections, equations, and code listings that are defined but not used.

🏗️ Supported Label Types:
  - Figures   : {#fig-...}  → referenced as @fig-...
  - Tables    : {#tbl-...}  → referenced as @tbl-...
  - Sections  : {#sec-...}  → referenced as @sec-...
  - Equations : {#eq-...}   → referenced as @eq-...
  - Listings  : {#lst-...}  → referenced as @lst-...

📦 Usage:
  python find_unreferenced_labels.py path/to/file_or_dir [options]

🔧 Options:
  -f, --figures     Check figures only
  -t, --tables      Check tables only
  -s, --sections    Check sections only
  -e, --equations   Check equations only
  -l, --listings    Check code listings only

If no flags are provided, all types are checked.
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

# Full set of supported label types and their regex patterns
ALL_LABEL_TYPES = {
    "Figure":   r'\{#(fig-[\w-]+)',
    "Table":    r'\{#(tbl-[\w-]+)',
    "Section":  r'\{#(sec-[\w-]+)',
    "Equation": r'\{#(eq-[\w-]+)',
    "Listing":  r'\{#(lst-[\w-]+)',
}

REFERENCE_PATTERN = r'@((?:fig|tbl|sec|eq|lst)-[\w-]+)'

def get_files_to_process(path: Path):
    """Return a list of QMD files from a file or directory path."""
    if path.is_file() and path.suffix == ".qmd":
        return [path]
    elif path.is_dir():
        return list(path.rglob("*.qmd"))
    else:
        return []

def collect_labels_and_references(files, label_types):
    defined = defaultdict(dict)  # {label_type: {label_name: (file, line_number)}}
    referenced = set()

    for file in files:
        lines = file.read_text(encoding="utf-8").splitlines()

        for i, line in enumerate(lines, 1):
            # Check all label types
            for label_type, pattern in label_types.items():
                for match in re.finditer(pattern, line):
                    label = match.group(1)
                    defined[label_type][label] = (file, i)

            # Find references
            for match in re.finditer(REFERENCE_PATTERN, line):
                referenced.add(match.group(1))

    return defined, referenced

def report_unreferenced(defined, referenced):
    print("🔍 Unreferenced labels:\n")
    total = 0
    label_width = 30  # Adjust for padding

    for label_type, label_map in defined.items():
        for label, (file, line) in sorted(label_map.items()):
            if label not in referenced:
                try:
                    rel_path = file.relative_to(Path.cwd())
                except ValueError:
                    rel_path = file
                label_str = f"@{label}"
                print(f"❌ {label_type:<10}: {label_str:<{label_width}} ({rel_path}:{line})")
                total += 1

    if total == 0:
        print("✅ All defined labels are referenced!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Find unreferenced figures, tables, sections, equations, and listings in .qmd files."
    )
    parser.add_argument("path", type=Path, help="Path to a .qmd file or a directory of .qmd files")

    # Label-type filters
    parser.add_argument("-f", "--figures", action="store_true", help="Check figures only")
    parser.add_argument("-t", "--tables", action="store_true", help="Check tables only")
    parser.add_argument("-s", "--sections", action="store_true", help="Check sections only")
    parser.add_argument("-e", "--equations", action="store_true", help="Check equations only")
    parser.add_argument("-l", "--listings", action="store_true", help="Check listings only")

    return parser.parse_args()

def main():
    args = parse_args()

    # Determine which label types to check
    if not (args.figures or args.tables or args.sections or args.equations or args.listings):
        label_types = ALL_LABEL_TYPES  # default: all
    else:
        label_types = {}
        if args.figures:   label_types["Figure"] = ALL_LABEL_TYPES["Figure"]
        if args.tables:    label_types["Table"] = ALL_LABEL_TYPES["Table"]
        if args.sections:  label_types["Section"] = ALL_LABEL_TYPES["Section"]
        if args.equations: label_types["Equation"] = ALL_LABEL_TYPES["Equation"]
        if args.listings:  label_types["Listing"] = ALL_LABEL_TYPES["Listing"]

    files = get_files_to_process(args.path)
    if not files:
        print(f"❌ No .qmd files found in {args.path}")
        return

    defined, referenced = collect_labels_and_references(files, label_types)
    report_unreferenced(defined, referenced)

if __name__ == "__main__":
    main()
