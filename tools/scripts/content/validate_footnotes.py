#!/usr/bin/env python3
"""
📝 Validate and Clean Footnotes in Quarto Files

This script validates footnote consistency in .qmd files and can optionally clean up issues:
- Finds undefined footnote references
- Finds unused footnote definitions  
- Detects duplicate footnote definitions
- Can automatically remove unused definitions with --clean flag

DESIGN PHILOSOPHY:
- Clear visual output with emoji indicators
- Fast execution for CI/CD workflows
- Exit codes: 0 = valid, 1 = issues found
- Cleanup mode for automatic fixing

Usage:
    python validate_footnotes.py -d quarto/contents/  # Check all files
    python validate_footnotes.py -f chapter.qmd       # Check single file
    python validate_footnotes.py -d quarto/ --clean   # Clean unused footnotes
    python validate_footnotes.py -d quarto/ --quiet   # Minimal output for CI
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def find_footnote_references(content: str) -> Set[str]:
    """Find all footnote references [^fn-name] in the content."""
    # Match [^fn-name] but not [^fn-name]:
    pattern = r'\[\^(fn-[a-zA-Z0-9-_]+)\](?!:)'
    return set(re.findall(pattern, content))

def find_footnote_definitions(content: str) -> Dict[str, List[int]]:
    """Find all footnote definitions [^fn-name]: and their line numbers."""
    pattern = r'^\[\^(fn-[a-zA-Z0-9-_]+)\]:'
    definitions = defaultdict(list)
    
    for i, line in enumerate(content.split('\n'), 1):
        match = re.match(pattern, line)
        if match:
            fn_name = match.group(1)
            definitions[fn_name].append(i)
    
    return dict(definitions)

def validate_footnotes_in_file(filepath: Path, quiet: bool = False) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Validate footnotes in a single file.
    Returns (errors, unused_definitions_with_lines)
    """
    errors = []
    unused_with_lines = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return [f"Error reading {filepath}: {e}"], {}
    
    references = find_footnote_references(content)
    definitions = find_footnote_definitions(content)
    
    # Get relative path for cleaner output
    try:
        rel_path = filepath.relative_to(Path.cwd())
    except ValueError:
        rel_path = filepath
    
    # Check for undefined references
    undefined = references - set(definitions.keys())
    for ref in sorted(undefined):
        errors.append(f"{Colors.RED}❌ Undefined reference{Colors.ENDC}: [{ref}] in {rel_path}")
    
    # Check for unused definitions
    unused = set(definitions.keys()) - references
    for def_name in sorted(unused):
        line_nums = definitions[def_name]
        unused_with_lines[def_name] = line_nums
        for line_num in line_nums:
            errors.append(f"{Colors.YELLOW}⚠️  Unused definition{Colors.ENDC}: [{def_name}] at {rel_path}:{line_num}")
    
    # Check for duplicate definitions
    for def_name, line_nums in definitions.items():
        if len(line_nums) > 1:
            for line_num in line_nums[1:]:  # Report all but the first as duplicates
                errors.append(f"{Colors.RED}❌ Duplicate definition{Colors.ENDC}: [{def_name}] at {rel_path}:{line_num}")
    
    return errors, unused_with_lines

def clean_unused_footnotes(filepath: Path, unused_definitions: Dict[str, List[int]], dry_run: bool = False) -> int:
    """Remove unused footnote definitions from a file."""
    if not unused_definitions:
        return 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"{Colors.RED}❌ Error reading {filepath}: {e}{Colors.ENDC}")
        return 0
    
    # Sort line numbers in reverse to remove from bottom to top
    lines_to_remove = set()
    for def_name, line_nums in unused_definitions.items():
        for line_num in line_nums:
            lines_to_remove.add(line_num - 1)  # Convert to 0-based index
    
    if dry_run:
        rel_path = filepath.relative_to(Path.cwd()) if filepath.is_relative_to(Path.cwd()) else filepath
        print(f"{Colors.CYAN}Would remove {len(lines_to_remove)} unused footnote(s) from {rel_path}{Colors.ENDC}")
        return len(lines_to_remove)
    
    # Remove lines (in reverse order to maintain indices)
    for line_idx in sorted(lines_to_remove, reverse=True):
        # Also remove the content lines that follow the footnote definition
        # until we hit another footnote or a blank line
        del lines[line_idx]
        
        # Continue removing following lines that are part of the footnote content
        while line_idx < len(lines):
            if not lines[line_idx].strip() or lines[line_idx].startswith('[^'):
                break
            del lines[line_idx]
    
    # Write back the cleaned content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    return len(lines_to_remove)

def find_qmd_files(path: Path) -> List[Path]:
    """Find all .qmd files in a directory or return single file."""
    if path.is_file():
        if path.suffix == '.qmd':
            return [path]
        else:
            return []
    elif path.is_dir():
        # Recursively find all .qmd files
        qmd_files = []
        for qmd_file in path.rglob("*.qmd"):
            # Skip certain directories
            if any(skip in qmd_file.parts for skip in ["_freeze", "_site", "_book", ".venv", "__pycache__"]):
                continue
            qmd_files.append(qmd_file)
        return sorted(qmd_files)
    else:
        return []

def print_summary(total_files: int, total_errors: int, total_cleaned: int = 0, quiet: bool = False):
    """Print a summary of the validation/cleaning results."""
    if quiet and total_errors == 0:
        return
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.ENDC}")
    
    if total_cleaned > 0:
        print(f"{Colors.GREEN}✨ Cleaned {total_cleaned} unused footnote definition(s){Colors.ENDC}")
    
    if total_errors == 0:
        print(f"{Colors.GREEN}✅ All footnotes validated successfully!{Colors.ENDC}")
        print(f"   {total_files} file(s) checked - no issues found")
    else:
        print(f"{Colors.RED}❌ Footnote validation found {total_errors} issue(s){Colors.ENDC}")
        print(f"   {total_files} file(s) checked")
    
    print(f"{Colors.BOLD}{'='*60}{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(
        description="Validate and optionally clean footnotes in Quarto files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d quarto/contents/          # Check all files in directory
  %(prog)s -f chapter.qmd              # Check single file
  %(prog)s -d quarto/ --clean          # Remove unused footnotes
  %(prog)s -d quarto/ --quiet          # Minimal output for CI
  %(prog)s -d quarto/ --clean --dry-run # Preview what would be cleaned
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-f", "--file",
        type=Path,
        help="Single .qmd file to validate"
    )
    input_group.add_argument(
        "-d", "--directory",
        type=Path,
        help="Directory to recursively search for .qmd files"
    )
    
    # Action options
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove unused footnote definitions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without making changes"
    )
    
    # Output options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (for CI/CD pipelines)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code 1 if any issues found"
    )
    
    args = parser.parse_args()
    
    # Determine input path
    input_path = args.file if args.file else args.directory
    
    # Find all .qmd files
    qmd_files = find_qmd_files(input_path)
    
    if not qmd_files:
        print(f"{Colors.RED}❌ No .qmd files found in {input_path}{Colors.ENDC}")
        return 1
    
    if not args.quiet:
        print(f"{Colors.BOLD}📝 Validating footnotes in {len(qmd_files)} file(s)...{Colors.ENDC}\n")
    
    # Validate each file
    all_errors = []
    total_cleaned = 0
    files_with_issues = []
    
    for qmd_file in qmd_files:
        errors, unused_defs = validate_footnotes_in_file(qmd_file, args.quiet)
        
        if errors:
            files_with_issues.append(qmd_file)
            if not args.quiet:
                rel_path = qmd_file.relative_to(Path.cwd()) if qmd_file.is_relative_to(Path.cwd()) else qmd_file
                print(f"{Colors.BOLD}📄 {rel_path}:{Colors.ENDC}")
                for error in errors:
                    print(f"   {error}")
                print()
            all_errors.extend(errors)
        
        # Clean unused definitions if requested
        if args.clean and unused_defs:
            cleaned = clean_unused_footnotes(qmd_file, unused_defs, args.dry_run)
            total_cleaned += cleaned
            if not args.quiet and cleaned > 0:
                action = "Would remove" if args.dry_run else "Removed"
                rel_path = qmd_file.relative_to(Path.cwd()) if qmd_file.is_relative_to(Path.cwd()) else qmd_file
                print(f"{Colors.GREEN}✨ {action} {cleaned} unused footnote(s) from {rel_path}{Colors.ENDC}")
    
    # Print summary
    print_summary(len(qmd_files), len(all_errors), total_cleaned, args.quiet)
    
    # Determine exit code
    if args.strict and all_errors:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())