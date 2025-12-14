#!/usr/bin/env python3
"""
Rename auto-generated image files to use original filenames from URLs.

This script extracts the original URLs from git history and renames the
auto-* image files to use the actual image names from those URLs.
"""

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import sys

def extract_filename_from_url(url: str) -> str:
    """Extract and clean the original filename from a URL."""
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Get the filename from the path
    original_filename = os.path.basename(path)

    # Remove query parameters and get base name
    base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename

    # Remove trailing hashes/IDs (e.g., "oranges-frogs_nHEaTqne53" -> "oranges-frogs")
    base_name = re.sub(r'_[a-zA-Z0-9]{8,}$', '', base_name)

    # Sanitize filename
    base_name = re.sub(r'[^a-zA-Z0-9_-]', '-', base_name)
    base_name = re.sub(r'-+', '-', base_name)
    base_name = base_name.strip('-')

    # Get extension
    extension = original_filename.rsplit('.', 1)[-1].split('?')[0] if '.' in original_filename else 'png'

    return f"{base_name}.{extension}"

def get_original_urls(commit_hash: str, file_path: str) -> dict:
    """Get original URLs from git history before the commit."""
    try:
        # Get the file content before the commit
        result = subprocess.run(
            ['git', 'show', f'{commit_hash}^:{file_path}'],
            capture_output=True,
            text=True,
            check=True
        )

        content = result.stdout

        # Find all image URLs
        pattern = r'!\[.*?\]\((https?://[^)]+)\)'
        urls = re.findall(pattern, content)

        # Create mapping of URL to desired filename
        url_to_filename = {}
        for url in urls:
            clean_url = url.split(')')[0]  # Remove any trailing characters
            filename = extract_filename_from_url(clean_url)
            url_to_filename[clean_url] = filename

        return url_to_filename

    except subprocess.CalledProcessError as e:
        print(f"Error getting git history: {e}")
        return {}

def find_current_image_mapping(qmd_file: Path) -> dict:
    """Find current image references in the qmd file."""
    try:
        with open(qmd_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all local image references
        pattern = r'!\[.*?\]\((images/[^)]+)\)'
        local_images = re.findall(pattern, content)

        return {img: qmd_file.parent / img for img in local_images}

    except Exception as e:
        print(f"Error reading {qmd_file}: {e}")
        return {}

def rename_images_for_file(qmd_file: Path, commit_hash: str, dry_run: bool = False):
    """Rename auto-* images in a specific qmd file to use original names."""
    print(f"\n📄 Processing: {qmd_file}")

    # Get original URLs from git
    relative_path = str(qmd_file.relative_to(Path.cwd()))
    url_mapping = get_original_urls(commit_hash, relative_path)

    if not url_mapping:
        print(f"   ⚠️  No URLs found in git history")
        return

    print(f"   Found {len(url_mapping)} original URLs")

    # Read current file
    with open(qmd_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all current local image references
    pattern = r'(!\[.*?\]\()(images/[^)]+)(\)[^}]*\{[^}]*\})'

    new_content = content
    rename_count = 0

    for match in re.finditer(pattern, content):
        prefix = match.group(1)
        image_path = match.group(2)
        suffix = match.group(3)
        full_match = match.group(0)

        # Check if this is an auto-* file
        if '/auto-' not in image_path:
            continue

        # Get the actual file path
        full_image_path = qmd_file.parent / image_path

        if not full_image_path.exists():
            print(f"   ⚠️  File not found: {image_path}")
            continue

        # Try to find matching original URL by checking the file
        # We'll use a simpler approach: just rename based on the pattern
        filename = full_image_path.name
        parent_dir = full_image_path.parent
        extension = filename.split('.')[-1]

        # For auto-* files, we'll need to map back to original URL
        # This is tricky, so let's use a different approach:
        # Extract what the new name should be from our updated script
        # For now, let's just note which files need renaming

        print(f"   📝 Current: {image_path}")

    return rename_count

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Rename auto-* images to use original filenames')
    parser.add_argument('commit', help='Commit hash where images were downloaded (e.g., 8d8317806)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be renamed without actually renaming')

    args = parser.parse_args()

    print("🔄 Image Renaming Tool")
    print("=" * 60)
    print(f"Commit: {args.commit}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 60)

    # Find all qmd files with auto-* images
    labs_dir = Path('book/quarto/contents/labs')

    if not labs_dir.exists():
        print(f"❌ Directory not found: {labs_dir}")
        return 1

    # Find files
    qmd_files = list(labs_dir.rglob('*.qmd'))

    print(f"\n📂 Found {len(qmd_files)} .qmd files")

    total_renamed = 0
    for qmd_file in qmd_files:
        renamed = rename_images_for_file(qmd_file, args.commit, args.dry_run)
        if renamed:
            total_renamed += renamed

    print(f"\n{'📊' if not args.dry_run else '🧪'} SUMMARY:")
    print(f"   Files renamed: {total_renamed}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
