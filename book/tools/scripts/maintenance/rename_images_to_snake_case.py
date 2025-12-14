import os
import re
import argparse
import sys
from pathlib import Path

def to_snake_case(name: str) -> str:
    """
    Converts a string to snake_case.
    - Converts CamelCase to snake_case.
    - Replaces hyphens and spaces with underscores.
    - Handles multiple uppercase letters together.
    - Converts to lowercase.
    - Removes consecutive underscores.
    """
    if not name:
        return ""
    # Add underscore before a capital letter if it's preceded by a lowercase letter or digit
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # Add underscore before a capital letter if it's followed by a lowercase letter
    s2 = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', s1)
    # Replace spaces and hyphens with underscores
    s3 = s2.replace(' ', '_').replace('-', '_')
    # Consolidate multiple underscores and convert to lower case
    return re.sub(r'_+', '_', s3).lower()

def get_files_to_check(root_dir: str) -> list[Path]:
    """
    Finds all image files in a directory that are not in ignored directories.
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
    ignored_dirs = ['.git', '.quarto', '_build', 'mediabag']
    files_to_check = []
    for subdir, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to prune the search
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        for file in files:
            file_path = Path(subdir) / file
            if file_path.suffix.lower() in image_extensions:
                files_to_check.append(file_path)
    return files_to_check

def find_all_text_files(root_dir: str) -> list[Path]:
    """
    Returns a list of all files that are likely to be text files.
    """
    all_files = []
    ignored_dirs = ['.git', '.quarto', '_build']
    allowed_extensions = ['.qmd', '.md', '.txt', '.py', '.yml', '.yaml', '.html', '.tex']

    for subdir, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            file_path = Path(subdir) / file
            if file_path.suffix.lower() in allowed_extensions and 'mediabag' not in str(file_path):
                all_files.append(file_path)
    return all_files

def main():
    parser = argparse.ArgumentParser(
        description="Finds and fixes image filenames that are not in snake_case and updates their references."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the renaming and replacement. Without this flag, it's a dry run.",
    )
    args = parser.parse_args()

    root_dir = Path('.')
    image_files = get_files_to_check(str(root_dir))

    files_to_rename = {}
    for path in image_files:
        original_stem = path.stem
        correct_stem = to_snake_case(original_stem)

        if original_stem != correct_stem:
            correct_filename = f"{correct_stem}{path.suffix}"
            new_path = path.with_name(correct_filename)
            files_to_rename[path] = new_path

    if not files_to_rename:
        print("All image filenames are already compliant. Nothing to do.")
        return

    print("The following files will be renamed:")
    for old, new in files_to_rename.items():
        print(f"  - {old} -> {new}")

    if not args.execute:
        print("\nThis was a dry run. Run with --execute to apply these changes.")
        return

    print("\nStarting replacement and renaming...")

    all_text_files = find_all_text_files(str(root_dir))

    total_updates = 0
    # Process replacements for all files
    for old_path, new_path in files_to_rename.items():
        old_filename = old_path.name
        new_filename = new_path.name

        for text_file in all_text_files:
            try:
                # Read file content
                with text_file.open('r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Perform replacement if old filename is found
                if old_filename in content:
                    # Simple substring replacement as requested
                    new_content = content.replace(old_filename, new_filename)
                    # Write updated content back
                    with text_file.open('w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"  Updated: {text_file}")
                    total_updates += 1
            except Exception as e:
                print(f"  Skipping file {text_file}: {e}", file=sys.stderr)

    # Rename the actual files after all references are updated
    for old_path, new_path in files_to_rename.items():
        if old_path.exists():
            try:
                # Ensure parent directory exists for the new path
                new_path.parent.mkdir(parents=True, exist_ok=True)
                old_path.rename(new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            except Exception as e:
                print(f"  ERROR renaming {old_path} to {new_path}: {e}", file=sys.stderr)
        else:
            # This can happen if a parent directory was renamed.
            # The logic doesn't handle directory renames, so this shouldn't be an issue for now.
            print(f"  Skipping rename for {old_path}, as it no longer exists.", file=sys.stderr)

    print(f"\nProcess complete. Found {len(files_to_rename)} files to rename.")
    print(f"Updated {total_updates} references across the project.")

if __name__ == "__main__":
    main()
