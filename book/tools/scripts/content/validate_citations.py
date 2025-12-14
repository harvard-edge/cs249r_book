#!/usr/bin/env python3
"""
Validate that all citations in .qmd files have corresponding entries in .bib files.

This script checks that every @key citation reference in a .qmd file has a matching
BibTeX entry in the associated bibliography file. It's designed to run as a pre-commit
hook to catch missing citations before they cause build failures.

Usage:
    python validate_citations.py file1.qmd file2.qmd ...
    python validate_citations.py -d quarto/contents/

Exit codes:
    0: All citations are valid
    1: Missing citations found or other errors
"""

import re
import argparse
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional
from collections import defaultdict


def extract_bibliography_file(qmd_file: Path) -> Optional[str]:
    """Extract the bibliography file name from a .qmd file's YAML frontmatter."""
    try:
        with open(qmd_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for bibliography field in YAML frontmatter
        match = re.search(r'^bibliography:\s*([^\s]+\.bib)\s*$', content, re.MULTILINE)
        return match.group(1) if match else None
    except Exception as e:
        print(f"❌ Error reading {qmd_file}: {e}", file=sys.stderr)
        return None


def extract_citation_keys(qmd_file: Path) -> Set[str]:
    """
    Extract all citation keys from a .qmd file.

    Handles various citation formats:
    - [@key]
    - @key
    - [@key1; @key2]
    - [-@key]
    - [@key, p. 123]
    """
    try:
        with open(qmd_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove code blocks to avoid false positives
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content = re.sub(r'`[^`]+`', '', content)

        # Extract all citation keys
        # Pattern matches @word with letters, numbers, hyphens, underscores, colons, dots
        citation_keys = set(re.findall(r'@([\w\-_:.]+)', content))

        # Strip trailing punctuation (periods, commas) that might be captured
        citation_keys = {key.rstrip('.,;:') for key in citation_keys}

        # Filter out DOI-style citations (start with numbers like 10.1109)
        citation_keys = {key for key in citation_keys if not re.match(r'^\d+\.\d+', key)}

        # Filter out common false positives that aren't citations
        filtered_keys = {
            key for key in citation_keys
            if not key.startswith(('fig-', 'tbl-', 'lst-', 'sec-', 'eq-'))
        }

        return filtered_keys
    except Exception as e:
        print(f"❌ Error reading {qmd_file}: {e}", file=sys.stderr)
        return set()


def extract_bib_keys(bib_file: Path) -> Set[str]:
    """Extract all entry keys from a .bib file."""
    try:
        with open(bib_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Match BibTeX entry keys: @article{key, @book{key, etc.
        bib_keys = set(re.findall(r'@\w+\{([^,\s]+)', content))
        return bib_keys
    except FileNotFoundError:
        print(f"❌ Bibliography file not found: {bib_file}", file=sys.stderr)
        return set()
    except Exception as e:
        print(f"❌ Error reading {bib_file}: {e}", file=sys.stderr)
        return set()


def validate_qmd_file(qmd_file: Path) -> Tuple[bool, List[str]]:
    """
    Validate that all citations in a .qmd file have corresponding .bib entries.

    Returns:
        (is_valid, missing_keys): Tuple of validation status and list of missing keys
    """
    bib_filename = extract_bibliography_file(qmd_file)

    if not bib_filename:
        # No bibliography specified, skip validation
        return True, []

    # Get the .bib file path (should be in the same directory as .qmd)
    bib_file = qmd_file.parent / bib_filename

    if not bib_file.exists():
        print(f"❌ {qmd_file.name}: Bibliography file not found: {bib_filename}", file=sys.stderr)
        return False, []

    citation_keys = extract_citation_keys(qmd_file)
    bib_keys = extract_bib_keys(bib_file)

    # Find missing citations
    missing_keys = sorted(citation_keys - bib_keys)

    if missing_keys:
        return False, missing_keys

    return True, []


def find_qmd_files(directory: Path) -> List[Path]:
    """Recursively find all .qmd files in a directory."""
    return list(directory.rglob("*.qmd"))


def main():
    parser = argparse.ArgumentParser(
        description="Validate that all citations in .qmd files have corresponding .bib entries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s file1.qmd file2.qmd              # Validate specific files
  %(prog)s -d quarto/contents/              # Validate all .qmd files in directory
  %(prog)s -d quarto/contents/ --quiet      # Only show errors
        """
    )
    parser.add_argument('files', nargs='*', help='Specific .qmd files to validate')
    parser.add_argument('-d', '--directory', type=Path,
                       help='Directory to recursively search for .qmd files')
    parser.add_argument('--quiet', action='store_true',
                       help='Only output errors, suppress success messages')

    args = parser.parse_args()

    # Collect files to validate
    files_to_check: List[Path] = []

    if args.files:
        files_to_check.extend(Path(f) for f in args.files)

    if args.directory:
        files_to_check.extend(find_qmd_files(args.directory))

    if not files_to_check:
        parser.print_help()
        return 0

    # Validate each file
    all_valid = True
    files_with_issues: Dict[str, List[str]] = {}
    total_files = 0
    files_with_citations = 0

    for qmd_file in files_to_check:
        if not qmd_file.exists():
            print(f"❌ File not found: {qmd_file}", file=sys.stderr)
            all_valid = False
            continue

        total_files += 1
        is_valid, missing_keys = validate_qmd_file(qmd_file)

        if not is_valid:
            all_valid = False
            if missing_keys:
                files_with_citations += 1
                files_with_issues[str(qmd_file)] = missing_keys

    # Report results
    if files_with_issues:
        print("\n❌ CITATION VALIDATION FAILED\n", file=sys.stderr)
        print("The following .qmd files reference citations that are missing from their .bib files:\n", file=sys.stderr)

        for qmd_file, missing_keys in sorted(files_with_issues.items()):
            print(f"📄 {qmd_file}:", file=sys.stderr)
            for key in missing_keys:
                print(f"   ❌ @{key}", file=sys.stderr)
            print(file=sys.stderr)

        print("To fix these issues:", file=sys.stderr)
        print("1. Find the citation entry in another chapter's .bib file", file=sys.stderr)
        print("2. Copy the BibTeX entry to the appropriate .bib file", file=sys.stderr)
        print("3. Or remove the citation reference if it's no longer needed\n", file=sys.stderr)

        return 1

    if not args.quiet:
        if total_files > 0:
            print(f"✅ All citations validated successfully ({total_files} files checked)")

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
