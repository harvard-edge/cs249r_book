#!/usr/bin/env python3
import os
import re
import requests
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import argparse

def find_qmd_files(directory):
    return list(Path(directory).rglob("*.qmd"))

def process_file(qmd_file, dry_run=False):
    try:
        with open(qmd_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Failed to read {qmd_file}: {e}")
        return

    # A simpler regex to find any markdown image with an external URL
    pattern = r'!\[(.*?)\]\((https?://[^\)]+)\)'
    
    matches = list(re.finditer(pattern, content))

    if not matches:
        return
        
    print(f"📄 Processing {qmd_file}")
    images_dir = qmd_file.parent / "images"
    
    new_content = content
    
    for match in matches:
        caption = match.group(1)
        url = match.group(2)
        
        print(f"  🔍 Found external image: {url}")
        
        try:
            image_name = Path(urlparse(url).path).name
            if not image_name:
                # If the URL path ends in a slash, there's no name, so we'll make one
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                image_name = f"image_{url_hash}.png" # Assume png, or we can try to guess

            local_path = images_dir / image_name
            relative_path = os.path.join("images", image_name)
            
            if dry_run:
                print(f"  🧪 [DRY RUN] Would download to {local_path}")
                print(f"  🧪 [DRY RUN] Would replace with {relative_path}")
                continue

            images_dir.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"  ✅ Downloaded to {local_path}")
            
            # Replace the old URL with the new relative path
            original_md_image = f"![{caption}]({url})"
            replacement_md_image = f"![{caption}]({relative_path})"
            new_content = new_content.replace(original_md_image, replacement_md_image)

        except Exception as e:
            print(f"  ❌ Failed to process {url}: {e}")

    if not dry_run and new_content != content:
        try:
            with open(qmd_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✅ Updated {qmd_file}")
        except Exception as e:
            print(f"  ❌ Failed to write updated file {qmd_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download external images from Quarto markdown files")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without actually downloading")
    args = parser.parse_args()

    qmd_files = find_qmd_files(args.directory)
    print(f"🔍 Found {len(qmd_files)} .qmd files to process")

    for qmd_file in qmd_files:
        process_file(qmd_file, args.dry_run)

if __name__ == "__main__":
    main()