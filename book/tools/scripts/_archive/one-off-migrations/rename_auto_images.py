#!/usr/bin/env python3
"""
Rename auto-* images to use original filenames from git history.
"""

import os
import re
import subprocess
from pathlib import Path
import sys

def extract_clean_filename(url: str) -> str:
    """Extract clean filename from URL."""
    # Get filename from URL path
    filename = url.split('/')[-1].split('?')[0]

    # Remove extension to get base name
    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename

    # Remove trailing hash patterns (e.g., "_abc123xyz")
    base_name = re.sub(r'_[a-zA-Z0-9]{8,}$', '', base_name)

    # Sanitize
    base_name = re.sub(r'[^a-zA-Z0-9_-]', '-', base_name)
    base_name = re.sub(r'-+', '-', base_name).strip('-')

    # Get extension
    ext = filename.rsplit('.', 1)[-1].split('?')[0] if '.' in filename else 'png'

    return f"{base_name}.{ext}"

def get_url_mappings(commit_hash: str):
    """Get mapping of current filenames to original URLs."""
    print("📜 Extracting URL mappings from git history...")

    # Get the diff to see what changed
    result = subprocess.run(
        ['git', 'show', commit_hash],
        capture_output=True,
        text=True
    )

    diff_content = result.stdout

    # Find all URL -> local path mappings
    mappings = {}

    # Pattern to match changes like:
    # -![](https://url.com/image.png)
    # +![](images/png/auto-xxx.png)

    lines = diff_content.split('\n')
    for i in range(len(lines) - 1):
        if lines[i].startswith('-![') and 'http' in lines[i]:
            next_line = lines[i + 1] if i + 1 < len(lines) else ''

            if next_line.startswith('+![') and 'images/' in next_line:
                # Extract URL from old line
                url_match = re.search(r'https?://[^\s)]+', lines[i])
                # Extract local path from new line
                path_match = re.search(r'images/[^\s)]+', next_line)

                if url_match and path_match:
                    url = url_match.group(0)
                    local_path = path_match.group(0)
                    mappings[local_path] = url

    print(f"   Found {len(mappings)} URL mappings")
    return mappings

def rename_images(mappings: dict, dry_run: bool = False):
    """Rename images and update references."""
    print(f"\n{'🧪 DRY RUN' if dry_run else '✏️  RENAMING'} images...")

    renamed_count = 0

    # Group by .qmd file
    files_to_update = {}

    # Track used names per directory to avoid collisions
    used_names_per_dir = {}

    for local_path, url in mappings.items():
        # Find the actual file
        files = list(Path('book/quarto/contents/labs').rglob(local_path))

        if not files:
            print(f"   ⚠️  File not found: {local_path}")
            continue

        old_file = files[0]
        base_new_name = extract_clean_filename(url)

        # Handle duplicates by adding a number
        new_name = base_new_name
        dir_key = str(old_file.parent)

        if dir_key not in used_names_per_dir:
            used_names_per_dir[dir_key] = set()

        # If name is already used in this directory, add a number
        if new_name in used_names_per_dir[dir_key]:
            base, ext = new_name.rsplit('.', 1) if '.' in new_name else (new_name, 'png')
            counter = 2
            while f"{base}_{counter}.{ext}" in used_names_per_dir[dir_key]:
                counter += 1
            new_name = f"{base}_{counter}.{ext}"

        used_names_per_dir[dir_key].add(new_name)
        new_file = old_file.parent / new_name

        print(f"   {old_file.name} -> {new_name}")

        if not dry_run:
            # Rename the file
            old_file.rename(new_file)

        # Track which .qmd files need updating
        qmd_files = list(old_file.parent.parent.parent.glob('*.qmd'))
        for qmd_file in qmd_files:
            if qmd_file not in files_to_update:
                files_to_update[qmd_file] = []
            files_to_update[qmd_file].append((local_path, f"images/{old_file.parent.name}/{new_name}"))

        renamed_count += 1

    # Update .qmd files
    if not dry_run and files_to_update:
        print(f"\n📝 Updating {len(files_to_update)} .qmd files...")
        for qmd_file, replacements in files_to_update.items():
            with open(qmd_file, 'r', encoding='utf-8') as f:
                content = f.read()

            for old_ref, new_ref in replacements:
                content = content.replace(old_ref, new_ref)

            with open(qmd_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"   ✅ {qmd_file.name}")

    return renamed_count

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Rename auto-* images to use original names')
    parser.add_argument('--commit', default='8d8317806', help='Commit hash (default: 8d8317806)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')

    args = parser.parse_args()

    print("🔄 Image Renaming Tool")
    print("=" * 60)

    # Get URL mappings from git
    mappings = get_url_mappings(args.commit)

    if not mappings:
        print("❌ No mappings found in git history")
        return 1

    # Rename images
    renamed = rename_images(mappings, args.dry_run)

    print(f"\n📊 Summary: {renamed} images {'would be' if args.dry_run else ''} renamed")

    if args.dry_run:
        print("\n💡 Run without --dry-run to apply changes")

    return 0

if __name__ == '__main__':
    sys.exit(main())
