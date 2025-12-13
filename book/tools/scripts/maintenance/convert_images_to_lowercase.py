import os
import re
from pathlib import Path
import argparse

def get_image_files(root_dir: str) -> list[Path]:
    """Finds all image files (png, jpg, jpeg, gif, svg) in a directory."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg']
    image_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = Path(subdir) / file
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
    return image_files

def rename_image_to_lowercase(image_path: Path):
    """Renames an image file to all lowercase."""
    lower_case_filename = image_path.name.lower()
    if image_path.name != lower_case_filename:
        new_path = image_path.with_name(lower_case_filename)
        try:
            image_path.rename(new_path)
            print(f"Renamed: {image_path.name} -> {new_path.name}")
            return image_path, new_path
        except FileExistsError:
            print(f"Warning: {new_path} already exists. Skipping rename for {image_path.name}.")
        except Exception as e:
            print(f"Error renaming {image_path}: {e}")
    return None, None

def get_files_to_check(root_dir: str) -> list[Path]:
    """Finds all Markdown and Quarto files."""
    extensions = ['.qmd', '.md']
    files_to_check = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = Path(subdir) / file
            if file_path.suffix.lower() in extensions:
                files_to_check.append(file_path)
    return files_to_check

def update_references_in_file(file_path: Path, original_path: Path, new_path: Path):
    """Updates all references to an old image path with the new path in a file."""
    try:
        content = file_path.read_text(encoding='utf-8')

        # We need to construct a relative path for the reference update
        original_ref = os.path.relpath(original_path, file_path.parent)
        new_ref = os.path.relpath(new_path, file_path.parent)

        # To handle both markdown and HTML style references, we perform a case-insensitive replacement.
        # This is a simple string replacement, which should be effective for most cases.
        if content.lower().count(str(original_path.name).lower()) > 0:
            updated_content = re.sub(re.escape(original_path.name), new_path.name, content, flags=re.IGNORECASE)

            if content != updated_content:
                file_path.write_text(updated_content, encoding='utf-8')
                print(f"Updated reference in: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Converts image filenames to lowercase and updates their references in .qmd and .md files."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="quarto/contents",
        help="The root directory to search for images and content files. Defaults to 'quarto/contents'."
    )
    args = parser.parse_args()

    content_root = args.path

    print(f"Scanning for images in: {content_root}")
    image_files = get_image_files(content_root)

    print(f"Found {len(image_files)} image files to process.")

    files_to_check = get_files_to_check(content_root)

    for image_path in image_files:
        original_path, new_path = rename_image_to_lowercase(image_path)
        if original_path and new_path:
            for file_path in files_to_check:
                update_references_in_file(file_path, original_path, new_path)

    print("\nImage processing and reference updates complete.")

if __name__ == "__main__":
    main()
