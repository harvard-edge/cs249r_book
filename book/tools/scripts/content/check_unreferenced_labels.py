#!/usr/bin/env python3
"""
🔍 Find Unreferenced Labels in Quarto Files

This script identifies defined labels (e.g., {#fig-xyz}) in .qmd files that are
never referenced (e.g., @fig-xyz). It's useful for cleaning up figures, tables,
sections, equations, and code listings that are defined but not used.

Exits with 0 if all labels are referenced; exits with 1 otherwise.
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

ALL_LABEL_TYPES = {
    "Figure":   r'(?:\{#|#\|\s*label:\s*|%%\|\s*label:\s*)(fig-[\w-]+)',
    "Table":    r'\{#(tbl-[\w-]+)',
    "Section":  r'\{#(sec-[\w-]+)',
    "Equation": r'\{#(eq-[\w-]+)',
    "Listing":  r'\{[^}]*#(lst-[\w-]+)',
}

REFERENCE_PATTERN = r'@((?:fig|tbl|sec|eq|lst)-[\w-]+)'

def get_files_to_process(path: Path):
    if path.is_file() and path.suffix == ".qmd":
        return [path]
    elif path.is_dir():
        return list(path.rglob("*.qmd"))
    else:
        return []

def collect_labels_and_references(files, label_types):
    defined = defaultdict(dict)
    referenced = {}  # Changed to dict to track file and line number

    for file in files:
        lines = file.read_text(encoding="utf-8").splitlines()

        for i, line in enumerate(lines, 1):
            for label_type, pattern in label_types.items():
                for match in re.finditer(pattern, line):
                    label = match.group(1)
                    defined[label_type][label] = (file, i)

            for match in re.finditer(REFERENCE_PATTERN, line):
                ref = match.group(1)
                if ref not in referenced:
                    referenced[ref] = []
                referenced[ref].append((file, i))

    return defined, referenced

def report_unreferenced(defined, referenced):
    unreferenced_labels = []

    # Get all referenced labels (keys of the referenced dict)
    referenced_labels = set(referenced.keys())

    for label_type, label_map in defined.items():
        # Skip checking sections for unreferenced labels - it's normal for textbook sections to not be cross-referenced
        if label_type == "Section":
            continue

        for label, (file, line) in sorted(label_map.items()):
            if label not in referenced_labels:
                try:
                    rel_path = file.relative_to(Path.cwd())
                except ValueError:
                    rel_path = file.resolve()
                unreferenced_labels.append((label_type, label, rel_path, line))

    if unreferenced_labels:
        print("🔍 Unreferenced figures, tables, equations, or listings found:\n")
        for label_type, label, rel_path, line in unreferenced_labels:
            print(f"❌ {label_type:<10}: @{label:<30} ({rel_path}:{line})")
        return False
    else:
        print("All figures, tables, equations, and listings are properly referenced!")
        return True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find unreferenced figures, tables, sections, equations, and listings in .qmd files."
    )
    parser.add_argument("path", type=Path, help="Path to .qmd file or directory")

    parser.add_argument("-f", "--figures", action="store_true", help="Check figures only")
    parser.add_argument("-t", "--tables", action="store_true", help="Check tables only")
    parser.add_argument("-s", "--sections", action="store_true", help="Check sections only")
    parser.add_argument("-e", "--equations", action="store_true", help="Check equations only")
    parser.add_argument("-l", "--listings", action="store_true", help="Check listings only")

    return parser.parse_args()

def report_unresolved_references(referenced, defined):
    all_defined_labels = set()
    for label_map in defined.values():
        all_defined_labels.update(label_map.keys())

    unresolved = []
    for ref in sorted(referenced.keys()):
        if ref not in all_defined_labels:
            # Get all locations where this reference appears
            locations = referenced[ref]
            unresolved.append((ref, locations))

    if unresolved:
        print("\n🚫 References without matching definitions:\n")
        for ref, locations in unresolved:
            print(f"❌ @{ref} has no matching {{#{ref}}} label")
            print(f"   Referenced in:")
            for file, line in locations[:5]:  # Show first 5 occurrences
                try:
                    rel_path = file.relative_to(Path.cwd())
                except ValueError:
                    rel_path = file.resolve()
                print(f"     - {rel_path}:{line}")
            if len(locations) > 5:
                print(f"     ... and {len(locations) - 5} more locations")
        return False
    else:
        print("All references have corresponding labels!")
        return True

def main():
    args = parse_args()

    if not (args.figures or args.tables or args.sections or args.equations or args.listings):
        label_types = ALL_LABEL_TYPES
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
        sys.exit(1)

    defined, referenced = collect_labels_and_references(files, label_types)

    # Print summary statistics
    total_defined = sum(len(labels) for labels in defined.values())
    total_referenced = len(referenced)
    print(f"\n📊 Summary: Found {total_defined} defined labels and {total_referenced} unique references across {len(files)} files")

    success1 = report_unreferenced(defined, referenced)
    success2 = report_unresolved_references(referenced, defined)
    success = success1 and success2
    if success:
        print("\n✅ All labels and references are correctly defined!")
    else:
        print("\n❌ Some labels or references are not correctly defined!")
    print("\n Finished checking labels and references.")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
