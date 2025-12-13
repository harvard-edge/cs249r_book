#!/usr/bin/env python3
"""
🚫 Find Duplicate Labels in Quarto Files

This script recursively finds all .qmd files in a directory and identifies duplicate labels
(e.g., {#fig-xyz}) that can cause ambiguous cross-reference links in Quarto.

By default checks: figures, tables, sections, listings (the most common types)
Use flags to check other types or restrict to specific types.

Any duplicate label definition (same-file or cross-file) can cause reference confusion.

DESIGN PHILOSOPHY FOR PRE-COMMIT:
- FAIL on any duplicate labels (--strict mode)
- Fast execution for CI/CD workflows
- Clear exit codes: 0 = no duplicates, 1 = duplicates found
- Minimal output for automation (--quiet)

Exits with 0 if no duplicates found; exits with 1 if duplicates exist.
"""

import argparse
import re
import sys
import json
from pathlib import Path
from collections import defaultdict

# Label patterns for DEFINITIONS only (not references)
LABEL_PATTERNS = {
    "Figure":   [
        r'\{#(fig-[\w-]+)',  # {#fig-xxx}
        r'#\|\s*(?:label|fig-label):\s*(fig-[\w-]+)',  # #| label: fig-xxx or #| fig-label: fig-xxx
    ],
    "Table":    [
        r'\{#(tbl-[\w-]+)',  # {#tbl-xxx}
        r'#\|\s*(?:label|tbl-label):\s*(tbl-[\w-]+)',  # #| label: tbl-xxx
    ],
    "Section":  [r'\{#(sec-[\w-]+)'],  # {#sec-xxx}
    "Equation": [r'\{#(eq-[\w-]+)'],   # {#eq-xxx}
    "Listing":  [
        r'\{#(lst-[\w-]+)',  # {#lst-xxx}
        r'#\|\s*(?:label|lst-label):\s*(lst-[\w-]+)',  # #| lst-label: lst-xxx
    ],
    "Video":    [r'\{#(vid-[\w-]+)'],    # {#vid-xxx}
    "Exercise": [r'\{#(exr-[\w-]+)'],    # {#exr-xxx}
}

def find_qmd_files(directory: Path):
    """Recursively find all .qmd files in directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    qmd_files = list(directory.rglob("*.qmd"))
    return sorted(qmd_files)  # Sort for consistent output

def is_in_code_block(lines, line_index):
    """Check if the current line is inside a code block."""
    in_code_block = False
    for i in range(line_index):
        line = lines[i].strip()
        if line.startswith('```'):
            in_code_block = not in_code_block
    return in_code_block

def get_format_context(lines, line_index):
    """Get the format context (html/pdf) for a line if it's in a conditional block.

    Returns:
        str: 'html', 'pdf', or 'default' if not in a conditional block
    """
    current_format = 'default'
    div_level = 0

    for i in range(line_index):
        line = lines[i].strip()

        if line.startswith(':::'):
            if 'when-format="html"' in line:
                current_format = 'html'
                div_level += 1
            elif 'when-format="pdf"' in line:
                current_format = 'pdf'
                div_level += 1
            elif line == ':::':
                div_level -= 1
                if div_level == 0:
                    current_format = 'default'

    return current_format

def build_label_map(files, label_types):
    """Build a complete map of all label DEFINITIONS found across all files.

    Handles conditional format blocks (HTML/PDF) properly - same label in different
    format blocks is considered one logical definition.

    Returns:
        dict: label -> [(file, line_num, label_type, format_context), ...]
    """
    label_map = defaultdict(list)  # label -> [(file, line_num, label_type, format_context), ...]
    file_count = 0
    total_labels = 0

    for file in files:
        try:
            content = file.read_text(encoding="utf-8")
            lines = content.splitlines()
            file_count += 1
        except Exception as e:
            print(f"⚠️  Warning: Could not read {file}: {e}", file=sys.stderr)
            continue

        file_labels = 0
        for line_num, line in enumerate(lines, 1):
            # Skip lines in code blocks
            if is_in_code_block(lines, line_num - 1):
                continue

            # Get format context (html/pdf/default)
            format_context = get_format_context(lines, line_num - 1)

            for label_type, patterns in label_types.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, line):
                        label = match.group(1)
                        label_map[label].append((file, line_num, label_type, format_context))
                        file_labels += 1
                        total_labels += 1

    return label_map, {"files_processed": file_count, "total_labels": total_labels}

def find_duplicates(label_map):
    """Find labels that have true duplicate definitions.

    Same label in HTML and PDF format blocks is NOT considered a duplicate
    (it's the same logical definition for different output formats).

    Args:
        label_map: Dictionary of label -> [(file, line_num, label_type, format_context), ...]

    Returns:
        Dictionary of duplicate labels and their locations
    """
    duplicates = {}

    for label, locations in label_map.items():
        if len(locations) <= 1:
            continue

        # Group by (file, format_context) to identify true duplicates
        unique_definitions = set()
        for file, line_num, label_type, format_context in locations:
            unique_definitions.add((file, format_context))

        # Check for true duplicates
        true_duplicates = []

        # Case 1: Same file, same format context -> duplicate
        file_format_groups = defaultdict(list)
        for file, line_num, label_type, format_context in locations:
            file_format_groups[(file, format_context)].append((file, line_num, label_type, format_context))

        for (file, format_context), group_locations in file_format_groups.items():
            if len(group_locations) > 1:
                # Multiple definitions in same file with same format context = duplicate
                true_duplicates.extend(group_locations)

        # Case 2: Different files (regardless of format) -> duplicate
        files_involved = set(loc[0] for loc in locations)
        if len(files_involved) > 1:
            # Add all cross-file occurrences as duplicates
            true_duplicates = locations

        # Case 3: Same file but BOTH have 'default' format (not in conditional blocks) -> duplicate
        default_in_same_file = [loc for loc in locations if loc[3] == 'default']
        file_groups = defaultdict(list)
        for loc in default_in_same_file:
            file_groups[loc[0]].append(loc)

        for file, file_locs in file_groups.items():
            if len(file_locs) > 1:
                true_duplicates.extend(file_locs)

        if true_duplicates:
            # Remove duplicates from the list while preserving order
            seen = set()
            unique_true_duplicates = []
            for loc in true_duplicates:
                loc_key = (loc[0], loc[1])  # (file, line_num)
                if loc_key not in seen:
                    seen.add(loc_key)
                    unique_true_duplicates.append(loc)

            if len(unique_true_duplicates) > 1:
                duplicates[label] = unique_true_duplicates

    return duplicates

def report_duplicates(duplicates, stats=None, quiet=False, format_type="text"):
    """Report duplicate labels found.

    Args:
        duplicates: Dictionary of duplicate labels
        stats: Statistics about processing
        quiet: If True, minimal output
        format_type: "text", "json", or "summary"

    Returns:
        True if no duplicates, False if duplicates found
    """
    if not duplicates:
        if not quiet and format_type == "text":
            if stats:
                print(f"✅ No duplicate labels found! Processed {stats['files_processed']} files, {stats['total_labels']} labels.")
            else:
                print("✅ No duplicate labels found!")
        return True

    if format_type == "json":
        # JSON output for automation
        result = {
            "status": "error",
            "duplicate_count": len(duplicates),
            "stats": stats or {},
            "duplicates": {}
        }

        for label, locations in duplicates.items():
            result["duplicates"][label] = []
            for file, line_num, label_type, format_context in locations:
                result["duplicates"][label].append({
                    "file": str(file),
                    "line": line_num,
                    "type": label_type,
                    "format_context": format_context
                })

        print(json.dumps(result, indent=2))
        return False

    elif format_type == "summary":
        # Brief summary for pre-commit
        cross_file_count = sum(1 for label, locs in duplicates.items()
                              if len(set(loc[0] for loc in locs)) > 1)
        same_file_count = len(duplicates) - cross_file_count

        print(f"❌ DUPLICATE LABELS DETECTED:")
        print(f"   • {cross_file_count} cross-file duplicates")
        print(f"   • {same_file_count} same-file duplicates")
        print(f"   • Total: {len(duplicates)} duplicate labels")
        if stats:
            print(f"   • Processed: {stats['files_processed']} files, {stats['total_labels']} labels")
        print(f"\n💡 Run: python3 scripts/find_duplicate_labels.py -d <directory> --details")
        print(f"   to see specific locations and fix suggestions.")
        return False

    else:  # text format (default)
        if quiet:
            # Minimal output: just warnings with icons for problematic labels and files
            for label, locations in sorted(duplicates.items()):
                print(f"🚫 {label}")
                for file, line_num, label_type, format_context in sorted(locations):
                    try:
                        rel_path = file.relative_to(Path.cwd())
                    except ValueError:
                        rel_path = file.resolve()
                    context_info = f" ({format_context})" if format_context != 'default' else ""
                    print(f"   📍 {rel_path}:{line_num}{context_info}")
            return False

        else:
            # Detailed output (default)
            print("🚫 Duplicate labels detected:\n")

            for label, locations in sorted(duplicates.items()):
                files_involved = set(loc[0] for loc in locations)

                if len(files_involved) > 1:
                    print(f"❌ Label '{label}' appears in {len(files_involved)} different files:")
                else:
                    print(f"❌ Label '{label}' appears {len(locations)} times in same file:")

                for file, line_num, label_type, format_context in sorted(locations):
                    try:
                        rel_path = file.relative_to(Path.cwd())
                    except ValueError:
                        rel_path = file.resolve()
                    context_info = f" ({format_context})" if format_context != 'default' else ""
                    print(f"   📍 {label_type:<10}: {rel_path}:{line_num}{context_info}")

                print()  # Empty line for readability

            print(f"💥 Found {len(duplicates)} duplicate labels!")
            print(f"⚠️  These duplicates can cause ambiguous cross-reference links!")
            if stats:
                print(f"📊 Processed {stats['files_processed']} files with {stats['total_labels']} total labels")

            print("\n🔧 To fix these issues:")
            print("   1. Rename one of the duplicate labels in each conflict")
            print("   2. Update any cross-references (@label) to use the new names")
            print("   3. Ensure each label is unique across your entire project")

            return False

def generate_suggestions(duplicates):
    """Generate suggestions for fixing duplicate labels."""
    if not duplicates:
        return

    print("\n💡 Suggested fixes:")
    print("=" * 50)

    for label, locations in sorted(duplicates.items()):
        print(f"\nFor label '{label}':")

        for i, (file, line_num, label_type, format_context) in enumerate(sorted(locations)):
            chapter_name = file.parent.name if file.parent.name != 'core' else file.stem
            suggested_label = f"{label}-{chapter_name}"

            try:
                rel_path = file.relative_to(Path.cwd())
            except ValueError:
                rel_path = file.resolve()

            context_info = f" ({format_context})" if format_context != 'default' else ""
            print(f"   📝 In {rel_path}:{line_num}{context_info}")
            print(f"      Change: {{#{label}}} → {{#{suggested_label}}}")
            print(f"      Update references: @{label} → @{suggested_label}")

def create_precommit_config():
    """Generate a sample pre-commit configuration."""
    config = """
# Add to .pre-commit-config.yaml

repos:
  - repo: local
    hooks:
      - id: check-duplicate-labels
        name: Check for duplicate Quarto labels
        entry: python3 scripts/find_duplicate_labels.py
        args: ['-d', 'contents/core/', '--figures', '--tables', '--listings', '--quiet', '--strict']
        language: system
        files: '\\.qmd$'
        pass_filenames: false
"""
    return config.strip()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find duplicate labels across .qmd files that could cause wrong cross-reference links.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-commit usage (focused on critical types)
  python3 find_duplicate_labels.py -d contents/core/ --figures --tables --listings --quiet --strict

  # Check only figures and tables
  python3 find_duplicate_labels.py -d contents/core/ --figures --tables

  # Check only figures
  python3 find_duplicate_labels.py -d contents/core/ --figures-only

  # Check all label types
  python3 find_duplicate_labels.py -d contents/core/ --all-types

  # Development usage with suggestions
  python3 find_duplicate_labels.py -d contents/core/ --suggestions

  # JSON output for automation
  python3 find_duplicate_labels.py -d contents/core/ --format json

PRE-COMMIT INTEGRATION:
  python3 find_duplicate_labels.py -d contents/core/ --figures --tables --listings --quiet --strict
  Exit code 0 = no duplicates, 1 = duplicates found

  Add to .pre-commit-config.yaml:
    - repo: local
      hooks:
        - id: check-duplicate-labels
          name: Check duplicate Quarto labels
          entry: python3 scripts/find_duplicate_labels.py
          args: ['-d', 'contents/core/', '--figures', '--tables', '--listings', '--quiet', '--strict']
          language: system
          files: '\\.qmd$'
          pass_filenames: false

Duplicate Label Issues Fixed:
  - Multiple files with {#fig-architecture} → Wrong @fig-architecture links
  - Duplicate {#tbl-results} across chapters → Ambiguous table references
  - Same {#sec-introduction} in multiple files → Broken section links
        """
    )

    # Main input argument
    parser.add_argument("-d", "--dir", type=Path, required=False,
                       help="Directory to search for .qmd files (searches recursively)")

    # Type-specific checks (by default: figures, tables, sections, listings)
    parser.add_argument("--figures", action="store_true", help="Check figures (default: enabled)")
    parser.add_argument("--tables", action="store_true", help="Check tables (default: enabled)")
    parser.add_argument("--sections", action="store_true", help="Check sections (default: enabled)")
    parser.add_argument("--listings", action="store_true", help="Check listings (default: enabled)")
    parser.add_argument("--equations", action="store_true", help="Check equations (default: disabled)")
    parser.add_argument("--videos", action="store_true", help="Check videos (default: disabled)")
    parser.add_argument("--exercises", action="store_true", help="Check exercises (default: disabled)")

    # Convenience flags
    parser.add_argument("--all-types", action="store_true", help="Check all label types")
    parser.add_argument("--figures-only", action="store_true", help="Check figures only")
    parser.add_argument("--tables-only", action="store_true", help="Check tables only")
    parser.add_argument("--sections-only", action="store_true", help="Check sections only")
    parser.add_argument("--listings-only", action="store_true", help="Check listings only")

    # Detection mode
    parser.add_argument("--strict", action="store_true", default=True,
                       help="FAIL on any duplicates (exit code 1) - recommended for pre-commit")

    # Output options
    parser.add_argument("--format", choices=["text", "json", "summary"], default="text",
                       help="Output format: text (detailed), json (machine-readable), summary (brief)")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output: just print warnings with icons for problematic labels and files")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed output (opposite of --quiet)")
    parser.add_argument("--suggestions", action="store_true",
                       help="Generate suggested fixes for duplicate labels")
    parser.add_argument("--precommit-config", action="store_true",
                       help="Show sample pre-commit configuration")

    return parser.parse_args()

def main():
    args = parse_args()

    # Handle special cases
    if args.precommit_config:
        print(create_precommit_config())
        return 0

    # Directory is required for all other operations
    if not args.dir:
        parser = argparse.ArgumentParser()
        parser.error("the following arguments are required: -d/--dir")

    # Check for all duplicate label definitions

    # Determine output settings
    quiet = args.quiet and not args.details

    # Determine which label types to check
    label_types = {}

    # Handle convenience flags first
    if args.figures_only:
        label_types["Figure"] = LABEL_PATTERNS["Figure"]
    elif args.tables_only:
        label_types["Table"] = LABEL_PATTERNS["Table"]
    elif args.sections_only:
        label_types["Section"] = LABEL_PATTERNS["Section"]
    elif args.listings_only:
        label_types["Listing"] = LABEL_PATTERNS["Listing"]
    elif args.all_types:
        label_types = LABEL_PATTERNS
    else:
        # Default behavior or explicit type selection
        default_types = ["figures", "tables", "sections", "listings"]

        # Check if any explicit type flags were used
        explicit_flags = any([args.figures, args.tables, args.sections, args.listings,
                             args.equations, args.videos, args.exercises])

        if explicit_flags:
            # Use only explicitly enabled types
            if args.figures:   label_types["Figure"] = LABEL_PATTERNS["Figure"]
            if args.tables:    label_types["Table"] = LABEL_PATTERNS["Table"]
            if args.sections:  label_types["Section"] = LABEL_PATTERNS["Section"]
            if args.listings:  label_types["Listing"] = LABEL_PATTERNS["Listing"]
            if args.equations: label_types["Equation"] = LABEL_PATTERNS["Equation"]
            if args.videos:    label_types["Video"] = LABEL_PATTERNS["Video"]
            if args.exercises: label_types["Exercise"] = LABEL_PATTERNS["Exercise"]
        else:
            # Use defaults: figures, tables, sections, listings
            label_types = {
                "Figure": LABEL_PATTERNS["Figure"],
                "Table": LABEL_PATTERNS["Table"],
                "Section": LABEL_PATTERNS["Section"],
                "Listing": LABEL_PATTERNS["Listing"]
            }

    # Find all .qmd files in directory
    try:
        qmd_files = find_qmd_files(args.dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not qmd_files:
        if not quiet:
            print(f"❌ No .qmd files found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    if not quiet and args.format == "text":
        checked_types = ", ".join(label_types.keys())
        print(f"🔍 Scanning {len(qmd_files)} .qmd files in {args.dir}")
        print(f"🏷️  Checking: {checked_types}")

    # Build complete label map across all files
    label_map, stats = build_label_map(qmd_files, label_types)

    if not quiet and args.format == "text":
        print(f"📊 Found {stats['total_labels']} labels across {stats['files_processed']} files")

    # Find and report duplicates
    duplicates = find_duplicates(label_map)
    success = report_duplicates(duplicates, stats=stats, quiet=quiet, format_type=args.format)

    # Generate suggestions if requested
    if args.suggestions and duplicates and args.format == "text":
        generate_suggestions(duplicates)

    # Print final status for text format
    if not quiet and args.format == "text":
        if success:
            print("\n✅ All labels are unique! No duplicate label conflicts found.")
        else:
            print(f"\n❌ Found duplicate labels that could cause wrong cross-reference links!")
            if not args.suggestions:
                print("   Run with --suggestions flag to get fix recommendations.")

    # Exit with appropriate code for pre-commit
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
