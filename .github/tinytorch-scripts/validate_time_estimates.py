#!/usr/bin/env python3
"""
Validate time estimate consistency across LEARNING_PATH.md and module ABOUT.md files.
"""

import re
import sys
from pathlib import Path


def extract_time_from_learning_path(module_num):
    """Extract time estimate for a module from LEARNING_PATH.md"""
    learning_path = Path("modules/LEARNING_PATH.md")
    if not learning_path.exists():
        return None

    content = learning_path.read_text()

    # Pattern: **Module XX: Name** (X-Y hours, ⭐...)
    pattern = rf"\*\*Module {module_num:02d}:.*?\*\*\s*\((\d+-\d+\s+hours)"
    match = re.search(pattern, content)

    return match.group(1) if match else None


def extract_time_from_about(module_path):
    """Extract time estimate from module ABOUT.md"""
    about_file = module_path / "ABOUT.md"
    if not about_file.exists():
        return None

    content = about_file.read_text()

    # Pattern: time_estimate: "X-Y hours"
    pattern = r'time_estimate:\s*"(\d+-\d+\s+hours)"'
    match = re.search(pattern, content)

    return match.group(1) if match else None


def main():
    """Validate time estimates across all modules"""
    modules_dir = Path("modules")
    errors = []
    warnings = []

    print("⏱️  Validating Time Estimate Consistency")
    print("=" * 60)

    # Find all module directories
    module_dirs = sorted([d for d in modules_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])

    for module_dir in module_dirs:
        module_num = int(module_dir.name.split("_")[0])
        module_name = module_dir.name

        learning_path_time = extract_time_from_learning_path(module_num)
        about_time = extract_time_from_about(module_dir)

        if not about_time:
            warnings.append(f"⚠️  {module_name}: Missing time_estimate in ABOUT.md")
            continue

        if not learning_path_time:
            warnings.append(f"⚠️  {module_name}: Not found in LEARNING_PATH.md")
            continue

        if learning_path_time != about_time:
            errors.append(
                f"❌ {module_name}: Time mismatch\n"
                f"   LEARNING_PATH.md: {learning_path_time}\n"
                f"   ABOUT.md: {about_time}"
            )
        else:
            print(f"✅ {module_name}: {about_time}")

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
        print(f"\n{len(errors)} time estimate inconsistencies found!")
        sys.exit(1)
    else:
        print("\n✅ All time estimates are consistent!")
        sys.exit(0)


if __name__ == "__main__":
    main()
