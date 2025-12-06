#!/usr/bin/env python3
"""
Validate difficulty rating consistency across LEARNING_PATH.md and module ABOUT.md files.
"""

import re
import sys
from pathlib import Path


def normalize_difficulty(difficulty_str):
    """Normalize difficulty rating to star count"""
    if not difficulty_str:
        return None

    # Count stars
    star_count = difficulty_str.count("⭐")
    if star_count > 0:
        return star_count

    # Handle numeric format
    if difficulty_str.isdigit():
        return int(difficulty_str)

    # Handle "X/4" format
    match = re.match(r"(\d+)/4", difficulty_str)
    if match:
        return int(match.group(1))

    return None


def extract_difficulty_from_learning_path(module_num):
    """Extract difficulty rating for a module from LEARNING_PATH.md"""
    learning_path = Path("modules/LEARNING_PATH.md")
    if not learning_path.exists():
        return None

    content = learning_path.read_text()

    # Pattern: **Module XX: Name** (X-Y hours, ⭐...)
    pattern = rf"\*\*Module {module_num:02d}:.*?\*\*\s*\([^,]+,\s*([⭐]+)\)"
    match = re.search(pattern, content)

    return normalize_difficulty(match.group(1)) if match else None


def extract_difficulty_from_about(module_path):
    """Extract difficulty rating from module ABOUT.md"""
    about_file = module_path / "ABOUT.md"
    if not about_file.exists():
        return None

    content = about_file.read_text()

    # Pattern: difficulty: "⭐..." or difficulty: X
    pattern = r'difficulty:\s*["\']?([⭐\d/]+)["\']?'
    match = re.search(pattern, content)

    return normalize_difficulty(match.group(1)) if match else None


def main():
    """Validate difficulty ratings across all modules"""
    modules_dir = Path("modules")
    errors = []
    warnings = []

    print("⭐ Validating Difficulty Rating Consistency")
    print("=" * 60)

    # Find all module directories
    module_dirs = sorted([d for d in modules_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])

    for module_dir in module_dirs:
        module_num = int(module_dir.name.split("_")[0])
        module_name = module_dir.name

        learning_path_diff = extract_difficulty_from_learning_path(module_num)
        about_diff = extract_difficulty_from_about(module_dir)

        if not about_diff:
            warnings.append(f"⚠️  {module_name}: Missing difficulty in ABOUT.md")
            continue

        if not learning_path_diff:
            warnings.append(f"⚠️  {module_name}: Not found in LEARNING_PATH.md")
            continue

        if learning_path_diff != about_diff:
            errors.append(
                f"❌ {module_name}: Difficulty mismatch\n"
                f"   LEARNING_PATH.md: {'⭐' * learning_path_diff}\n"
                f"   ABOUT.md: {'⭐' * about_diff}"
            )
        else:
            print(f"✅ {module_name}: {'⭐' * about_diff}")

    print("\n" + "=" * 60)

    # Print warnings
    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  {warning}")

    # Print errors
    if errors:
        print("\n❌ Errors Found:")
        for error in errors:
            print(f"  {error}\n")
        print(f"\n{len(errors)} difficulty rating inconsistencies found!")
        sys.exit(1)
    else:
        print("\n✅ All difficulty ratings are consistent!")
        sys.exit(0)


if __name__ == "__main__":
    main()
