#!/usr/bin/env python3
"""
Fixes image reference case mismatches in Quarto markdown files.

This script reads the output of the `validate-image-references` pre-commit hook,
extracts the case mismatch errors, and corrects the references in the .qmd files.
"""

import argparse
import re
from pathlib import Path
import sys

def parse_pre_commit_output(output_file: Path) -> list[tuple[str, str]]:
    """
    Parses the pre-commit output to find case mismatch errors.

    Args:
        output_file: Path to the file containing the pre-commit output.

    Returns:
        A list of tuples, where each tuple contains the incorrect and correct filenames.
    """
    mismatches = []
    with open(output_file, 'r') as f:
        content = f.read()

    pattern = r"Case mismatch: expected '([^']*)' but found '([^']*)'"
    matches = re.findall(pattern, content)

    for expected, found in matches:
        mismatches.append((expected, found))

    return mismatches

def find_qmd_files(base_dir: Path) -> list[Path]:
    """Finds all .qmd files in the base directory."""
    return list(base_dir.rglob("*.qmd"))

def fix_references_in_file(qmd_file: Path, mismatches: list[tuple[str, str]]):
    """
    Fixes the image references in a single .qmd file.

    Args:
        qmd_file: Path to the .qmd file to fix.
        mismatches: A list of tuples with incorrect and correct filenames.
    """
    try:
        with open(qmd_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return # Skip if file not found

    original_content = content
    for incorrect, correct in mismatches:
        content = content.replace(incorrect, correct)

    if content != original_content:
        with open(qmd_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Corrected references in {qmd_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Fix image reference case mismatches in Quarto files."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the file containing the pre-commit output."
    )
    parser.add_argument(
        "-d", "--directory",
        type=Path,
        default=Path("quarto/contents"),
        help="The directory to search for .qmd files."
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"❌ Error: Input file not found at {args.input_file}")
        sys.exit(1)

    mismatches = parse_pre_commit_output(args.input_file)
    if not mismatches:
        print("No case mismatches found in the input file.")
        return

    qmd_files = find_qmd_files(args.directory)
    for qmd_file in qmd_files:
        fix_references_in_file(qmd_file, mismatches)

    print("\nDone.")

if __name__ == "__main__":
    main()
