import os
import re
from pathlib import Path

def to_snake_case(name: str) -> str:
    """Converts a string to snake_case."""
    if not name:
        return ""
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    s2 = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', s1)
    s3 = s2.replace(' ', '_').replace('-', '_')
    return re.sub(r'_+', '_', s3).lower()

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

def correct_image_references(file_path: Path):
    """Corrects the case of all image references in a file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        # This regex is designed to find Markdown image syntaxes like `![...](...)` or `![](...)`.
        # It is intentionally broad to capture various forms of image links.
        img_ref_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')

        def replacer(match):
            alt_text = match.group(1)
            path_str = match.group(2)

            # Skip http links
            if path_str.startswith('http'):
                return match.group(0)

            # A simple check for something that looks like a filepath.
            # This is not perfect, but it will prevent the script from
            # trying to process things that are not filepaths.
            if not re.search(r'\.(png|jpg|jpeg|gif|svg)$', path_str, re.IGNORECASE):
                return match.group(0)

            path = Path(path_str)
            original_filename = path.name

            # Deconstruct the filename to apply snake_case to the stem only.
            stem = path.stem
            suffix = path.suffix

            # Convert the stem to snake_case and then to lowercase.
            correct_stem = to_snake_case(stem)
            correct_filename = f"{correct_stem}{suffix}"

            # If the filename was already correct, no change is needed.
            if original_filename == correct_filename:
                return match.group(0)

            # Reconstruct the path with the corrected filename.
            new_path = path.with_name(correct_filename)
            return f"![{alt_text}]({new_path})"

        new_content, num_subs = img_ref_pattern.subn(replacer, content)

        if num_subs > 0:
            print(f"Corrected {num_subs} references in {file_path}")
            file_path.write_text(new_content, encoding='utf-8')

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    root_dir = '.'
    files_to_process = get_files_to_check(root_dir)

    for file_path in files_to_process:
        correct_image_references(file_path)

    print("\nFinished processing all files.")

if __name__ == "__main__":
    main()
