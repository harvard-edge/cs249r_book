#!/usr/bin/env python3
"""
Validate that all footnotes are properly defined and referenced in Quarto markdown files.

This script checks:
1. All footnote references [^fn-name] have corresponding definitions
2. All footnote definitions [^fn-name]: are actually referenced in the text
3. No duplicate footnote definitions exist
"""

import re
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict

def find_footnote_references(content: str) -> Set[str]:
    """Find all footnote references in the content."""
    # Match [^fn-name] but not [^fn-name]:
    pattern = r'\[\^(fn-[a-zA-Z0-9-_]+)\](?!:)'
    return set(re.findall(pattern, content))

def find_footnote_definitions(content: str) -> Dict[str, int]:
    """Find all footnote definitions and their line numbers."""
    pattern = r'^\[\^(fn-[a-zA-Z0-9-_]+)\]:'
    definitions = {}
    for i, line in enumerate(content.split('\n'), 1):
        match = re.match(pattern, line)
        if match:
            fn_name = match.group(1)
            if fn_name in definitions:
                # Duplicate definition found
                definitions[fn_name] = -i  # Negative to indicate duplicate
            else:
                definitions[fn_name] = i
    return definitions

def validate_footnotes_in_file(filepath: Path) -> List[str]:
    """Validate footnotes in a single file."""
    errors = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return [f"Error reading {filepath}: {e}"]
    
    references = find_footnote_references(content)
    definitions = find_footnote_definitions(content)
    
    # Check for undefined references
    undefined = references - set(definitions.keys())
    for ref in sorted(undefined):
        errors.append(f"{filepath}: Undefined footnote reference [{ref}]")
    
    # Check for unused definitions
    unused = set(definitions.keys()) - references
    for def_name in sorted(unused):
        line_num = abs(definitions[def_name])
        errors.append(f"{filepath}:{line_num}: Unused footnote definition [{def_name}]")
    
    # Check for duplicate definitions
    for def_name, line_num in definitions.items():
        if line_num < 0:  # Negative indicates duplicate
            errors.append(f"{filepath}:{-line_num}: Duplicate footnote definition [{def_name}]")
    
    return errors

def find_qmd_files(root_dir: Path) -> List[Path]:
    """Find all .qmd files in the quarto directory."""
    qmd_files = []
    quarto_dir = root_dir / "quarto"
    
    if not quarto_dir.exists():
        return []
    
    # Recursively find all .qmd files
    for qmd_file in quarto_dir.rglob("*.qmd"):
        # Skip certain directories if needed
        if "_freeze" in qmd_file.parts or "_site" in qmd_file.parts:
            continue
        qmd_files.append(qmd_file)
    
    return sorted(qmd_files)

def main():
    """Main validation function."""
    # Find the repository root (where .git directory is)
    current_path = Path.cwd()
    repo_root = current_path
    
    # Search for .git directory to find repo root
    while repo_root.parent != repo_root:
        if (repo_root / ".git").exists():
            break
        repo_root = repo_root.parent
    else:
        # If we can't find .git, use current directory
        repo_root = current_path
    
    # Find all .qmd files
    qmd_files = find_qmd_files(repo_root)
    
    if not qmd_files:
        print("No .qmd files found in quarto directory")
        return 0
    
    # Validate each file
    all_errors = []
    for qmd_file in qmd_files:
        errors = validate_footnotes_in_file(qmd_file)
        all_errors.extend(errors)
    
    # Report results
    if all_errors:
        print("❌ Footnote validation failed:\n")
        for error in all_errors:
            print(f"  {error}")
        print(f"\nTotal errors: {len(all_errors)}")
        return 1
    else:
        print(f"✓ All footnotes validated successfully ({len(qmd_files)} files checked)")
        return 0

if __name__ == "__main__":
    sys.exit(main())