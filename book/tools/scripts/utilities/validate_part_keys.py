#!/usr/bin/env python3
"""
Part Key Validation Script
==========================

This script scans all .qmd files for \\part{key:xxx} commands and validates them
against the part_summaries.yml file. It provides a comprehensive report of any
issues before you even start building.

Usage:
    python3 scripts/validate_part_keys.py
"""

import os
import re
import yaml
import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple

def load_part_summaries() -> Dict:
    """Load part summaries from YAML file."""
    yaml_path = Path("book/quarto/contents/parts/summaries.yml")
    if not yaml_path.exists():
        print("❌ Error: book/quarto/contents/parts/summaries.yml not found")
        return {}

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'parts' not in data:
                print("❌ Error: No 'parts' section in part_summaries.yml")
                return {}

            # Create a mapping of normalized keys to entries
            summaries = {}
            for part in data['parts']:
                if 'key' in part:
                    key = part['key'].lower().replace('_', '').replace('-', '')
                    summaries[key] = part

            return summaries
    except Exception as e:
        print(f"❌ Error loading part_summaries.yml: {e}")
        return {}

def find_qmd_files() -> List[Path]:
    """Find all .qmd files in the quarto directory."""
    qmd_files = []
    book_dir = Path("book/quarto")

    if not book_dir.exists():
        print("❌ Error: book/quarto directory not found")
        return []

    # Find all .qmd files recursively
    for qmd_file in book_dir.rglob("*.qmd"):
        qmd_files.append(qmd_file)

    return qmd_files

def extract_part_keys(content: str) -> List[Tuple[str, int]]:
    """Extract all \\part{key:xxx} commands from content."""
    pattern = r'\\part\{key:([^}]+)\}'
    matches = []

    for match in re.finditer(pattern, content):
        key = match.group(1)
        line_num = content[:match.start()].count('\n') + 1
        matches.append((key, line_num))

    return matches

def normalize_key(key: str) -> str:
    """Normalize key for comparison (lowercase, no underscores/hyphens)."""
    return key.lower().replace('_', '').replace('-', '')

def validate_keys() -> Tuple[Dict, List[Tuple[Path, str, int, str]]]:
    """Validate all part keys in .qmd files against part_summaries.yml."""

    # Load available keys
    summaries = load_part_summaries()
    if not summaries:
        return {}, []

    print(f"📚 Loaded {len(summaries)} keys from part_summaries.yml:")
    for key, part in summaries.items():
        title = part.get('title', 'Unknown')
        print(f"   - '{key}' -> '{title}'")

    # Find all .qmd files
    qmd_files = find_qmd_files()
    print(f"\n📄 Found {len(qmd_files)} .qmd files to scan")

    # Scan each file for part keys
    issues = []
    all_found_keys = set()

    for qmd_file in qmd_files:
        try:
            with open(qmd_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract part keys
            part_keys = extract_part_keys(content)

            for key, line_num in part_keys:
                normalized_key = normalize_key(key)
                all_found_keys.add(normalized_key)

                if normalized_key not in summaries:
                    issues.append((qmd_file, key, line_num, normalized_key))

        except Exception as e:
            print(f"❌ Error reading {qmd_file}: {e}")

    return summaries, issues

def main():
    """Main validation function."""
    print("🔍 Part Key Validation Script")
    print("=" * 40)

    # Validate keys
    summaries, issues = validate_keys()

    if not summaries:
        print("\n❌ Cannot proceed without valid part_summaries.yml")
        return 1

    # Report results
    print(f"\n📊 Validation Results:")
    print(f"   - Available keys: {len(summaries)}")
    print(f"   - Issues found: {len(issues)}")

    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for file_path, original_key, line_num, normalized_key in issues:
            print(f"   📄 {file_path}:{line_num}")
            print(f"      - Key: '{original_key}' (normalized: '{normalized_key}')")
            print(f"      - Status: NOT FOUND in part_summaries.yml")
            print()

        print("💡 To fix these issues:")
        print("   1. Add the missing keys to book/part_summaries.yml")
        print("   2. Or correct the key names in the .qmd files")
        print("   3. Or remove the \\part{key:xxx} commands if not needed")

        return 1
    else:
        print("\n✅ All part keys are valid!")
        print("🚀 You can proceed with building the book.")
        return 0

if __name__ == "__main__":
    exit(main())
