import os
import re
import argparse
import sys
import shutil
from pathlib import Path

def to_snake_case(name):
    """
    Converts a string to snake_case.
    """
    name = name.replace('-', '_').replace(' ', '_')
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', name)
    return re.sub(r'_+', '_', name).lower()

def find_non_compliant_items(root_dir):
    """
    Finds all image files in a directory that are not in snake_case.
    This can be extended to find other non-compliant items.
    """
    non_compliant_files = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.gif']

    search_dirs = [root_dir]
    if os.path.isdir('quarto/_build'):
        search_dirs.append('quarto/_build')
    if os.path.isdir('quarto/.quarto/_freeze'):
        search_dirs.append('quarto/.quarto/_freeze')

    for s_dir in search_dirs:
        for subdir, _, files in os.walk(s_dir):
            if 'mediabag' in subdir:
                continue

            for file in files:
                file_path = Path(subdir) / file
                if file_path.suffix.lower() in image_extensions:
                    stem = file_path.stem
                    snake_case_stem = to_snake_case(stem)

                    if stem != snake_case_stem or not stem.islower():
                        new_filename = f"{snake_case_stem}{file_path.suffix}"
                        non_compliant_files.append({
                            "original_path": str(file_path),
                            "new_filename": new_filename
                        })

    return non_compliant_files

def find_references(root_dir, filenames):
    """
    Finds all files that reference any of the given filenames.
    """
    references = {filename: [] for filename in filenames}
    search_extensions = ['.qmd', '.py', '.yml', '.yaml', '.html', '.tex']

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = Path(subdir) / file
            if file_path.suffix.lower() in search_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for filename in filenames:
                            if Path(filename).name.lower() in content.lower():
                                references[filename].append(str(file_path))
                except Exception as e:
                    print(f"Could not read {file_path}: {e}", file=sys.stderr)

    return references

def update_references(references, original_basename, new_filename):
    """
    Updates all references to a file.
    """
    for file_path in references:
        try:
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Case-insensitive replacement
            pattern = re.compile(re.escape(original_basename), re.IGNORECASE)
            content = pattern.sub(new_filename, content)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Could not update {file_path}: {e}", file=sys.stderr)

def clean_backups(root_dir):
    """
    Deletes all .bak files in a directory.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.bak'):
                os.remove(os.path.join(subdir, file))

def main():
    parser = argparse.ArgumentParser(description="Enforce snake_case for filenames.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="The root directory to scan."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for non-compliant filenames and exit with a non-zero status if any are found."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the renaming and update all references."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up all backup files."
    )
    args = parser.parse_args()

    if args.clean:
        clean_backups(args.root_dir)
        print("Backup files cleaned.")
        return

    non_compliant_items = find_non_compliant_items(args.root_dir)

    if not non_compliant_items:
        print("All filenames are compliant.")
        sys.exit(0)

    if args.check:
        print("Found non-compliant filenames:")
        for item in non_compliant_items:
            print(f"  - {item['original_path']}")
        sys.exit(1)

    elif args.execute:
        original_filenames = [item['original_path'] for item in non_compliant_items]
        references = find_references(args.root_dir, original_filenames)

        for item in non_compliant_items:
            original_path = Path(item['original_path'])
            new_filename = item['new_filename']
            new_path = original_path.parent / new_filename

            try:
                if original_path.exists():
                    os.rename(original_path, new_path)
                    print(f"Renamed {original_path} to {new_path}")

                    original_basename = original_path.name
                    if item['original_path'] in references:
                        update_references(references[item['original_path']], original_basename, new_filename)
                else:
                    print(f"File {original_path} not found, skipping rename.")

            except Exception as e:
                print(f"Could not rename {original_path}: {e}", file=sys.stderr)

        print("\nFixes complete. Remember to delete the .bak files after verifying the changes.")
    else:
        print("Found non-compliant filenames. Use --check for pre-commit or --execute to fix them.")
        for item in non_compliant_items:
            print(f"  - Original: {item['original_path']}, Proposed: {item['new_filename']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
