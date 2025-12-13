import os
import re
import argparse
import sys
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

def get_all_files(root_dir):
    """
    Gets a list of all files in a directory.
    """
    all_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            all_files.append(Path(subdir) / file)
    return all_files

def main():
    parser = argparse.ArgumentParser(description="Enforce snake_case for filenames.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the renaming and update all references."
    )
    args = parser.parse_args()

    if not args.execute:
        print("This script is designed to be run with the --execute flag.")
        sys.exit(1)

    all_files = get_all_files('.')
    image_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.gif']

    non_compliant_images = []
    for file_path in all_files:
        if file_path.suffix.lower() in image_extensions:
            if 'mediabag' not in str(file_path):
                stem = file_path.stem
                snake_case_stem = to_snake_case(stem)
                if stem != snake_case_stem or not stem.islower():
                    non_compliant_images.append(file_path)

    for image_path in non_compliant_images:
        snake_case_stem = to_snake_case(image_path.stem)
        new_filename = f"{snake_case_stem}{image_path.suffix}"
        new_path = image_path.parent / new_filename

        # Rename the image
        if image_path.exists():
            os.rename(image_path, new_path)
            print(f"Renamed {image_path} to {new_path}")

        # Update references in all files
        for file_to_update in all_files:
            if file_to_update.is_file() and file_to_update.suffix not in image_extensions:
                try:
                    with open(file_to_update, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(image_path.name), re.IGNORECASE)
                    new_content = pattern.sub(new_filename, content)

                    if new_content != content:
                        with open(file_to_update, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                except Exception:
                    pass

if __name__ == "__main__":
    main()
