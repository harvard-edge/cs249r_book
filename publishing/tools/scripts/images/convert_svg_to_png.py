#!/usr/bin/env python3
"""
convert_svg_to_png.py

Helper tool to convert SVG files to PNG format and find QMD file references.
This tool helps ensure consistency when converting SVG files to PNG.

Usage:
  python convert_svg_to_png.py path/to/file.svg
  python convert_svg_to_png.py --dry-run path/to/file.svg
  python convert_svg_to_png.py --find-references path/to/file.svg

Features:
- Converts SVG to PNG using ImageMagick
- Finds and shows QMD file references that need updating
- Supports dry-run mode to preview changes
- Provides detailed logging of all operations
- User controls all changes - nothing happens automatically
"""

import os
import sys
import argparse
import subprocess
import re
from pathlib import Path
from typing import List, Tuple, Optional

def find_qmd_references(svg_path: str, search_root: str = "book/quarto/contents") -> List[Tuple[str, int, str]]:
    """
    Find all QMD files that reference the given SVG file.

    Returns:
        List of tuples: (file_path, line_number, line_content)
    """
    references = []
    svg_filename = os.path.basename(svg_path)
    svg_relative_path = os.path.relpath(svg_path, start=search_root)

    # Search patterns for different ways the SVG might be referenced
    patterns = [
        svg_filename,  # Just the filename
        svg_relative_path,  # Relative path from search root
        svg_path,  # Full path
    ]

    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.endswith('.qmd'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for line_num, line in enumerate(lines, 1):
                        for pattern in patterns:
                            if pattern in line and '.svg' in line:
                                references.append((file_path, line_num, line.strip()))
                                break
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")

    return references

def convert_svg_to_png(svg_path: str, png_path: Optional[str] = None) -> bool:
    """
    Convert SVG file to PNG using ImageMagick.

    Returns:
        True if conversion successful, False otherwise
    """
    if png_path is None:
        png_path = svg_path.replace('.svg', '.png')

    try:
        # Use magick (ImageMagick 7) or convert (ImageMagick 6)
        cmd = ['magick', svg_path, png_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Try with convert command (older ImageMagick)
            cmd = ['convert', svg_path, png_path]
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Converted: {svg_path} → {png_path}")
            return True
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ ImageMagick not found. Please install ImageMagick:")
        print("  brew install imagemagick  # macOS")
        print("  apt-get install imagemagick  # Ubuntu/Debian")
        return False

def update_qmd_references(references: List[Tuple[str, int, str]], svg_path: str, png_path: str, dry_run: bool = False) -> int:
    """
    Update QMD file references from SVG to PNG.

    Returns:
        Number of files updated
    """
    files_updated = 0
    svg_filename = os.path.basename(svg_path)
    png_filename = os.path.basename(png_path)

    # Group references by file
    files_to_update = {}
    for file_path, line_num, line_content in references:
        if file_path not in files_to_update:
            files_to_update[file_path] = []
        files_to_update[file_path].append((line_num, line_content))

    for file_path, file_references in files_to_update.items():
        if dry_run:
            print(f"📝 Would update {file_path}:")
            for line_num, line_content in file_references:
                new_line = line_content.replace('.svg', '.png')
                print(f"   Line {line_num}: {line_content}")
                print(f"   →          {new_line}")
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Update lines
                updated = False
                for line_num, _ in file_references:
                    if line_num <= len(lines):
                        old_line = lines[line_num - 1]
                        new_line = old_line.replace('.svg', '.png')
                        if old_line != new_line:
                            lines[line_num - 1] = new_line
                            updated = True

                if updated:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    print(f"📝 Updated references in: {file_path}")
                    files_updated += 1

            except Exception as e:
                print(f"❌ Failed to update {file_path}: {e}")

    return files_updated

def main():
    parser = argparse.ArgumentParser(description="Helper tool for SVG to PNG conversion")
    parser.add_argument('svg_file', help="Path to SVG file to analyze/convert")
    parser.add_argument('--png-file', help="Output PNG file path (default: replace .svg with .png)")
    parser.add_argument('--search-root', default="book/quarto/contents", help="Root directory to search for QMD files")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be done without making changes")
    parser.add_argument('--find-references', action='store_true', help="Only find and show QMD references, don't convert")
    parser.add_argument('--convert-only', action='store_true', help="Only convert SVG to PNG, don't update references or remove SVG")
    parser.add_argument('--update-references', action='store_true', help="Update QMD references (use with caution)")
    parser.add_argument('--remove-svg', action='store_true', help="Remove original SVG file after conversion")

    args = parser.parse_args()

    svg_path = args.svg_file
    png_path = args.png_file or svg_path.replace('.svg', '.png')

    # Validate input
    if not os.path.exists(svg_path):
        print(f"❌ SVG file not found: {svg_path}")
        sys.exit(1)

    if not svg_path.endswith('.svg'):
        print(f"❌ File is not an SVG: {svg_path}")
        sys.exit(1)

    print(f"🔍 Analyzing SVG file: {svg_path}")
    print(f"📁 Searching for references in: {args.search_root}")

    # Find QMD references
    references = find_qmd_references(svg_path, args.search_root)

    if references:
        print(f"📋 Found {len(references)} reference(s) in QMD files:")
        for file_path, line_num, line_content in references:
            print(f"   {file_path}:{line_num} - {line_content}")
        print(f"\n⚠️  You will need to manually update these references from .svg to .png")
    else:
        print("📋 No QMD references found")

    # Handle different modes
    if args.find_references:
        print("✅ Reference search complete!")
        return

    if args.dry_run:
        print("\n🔍 DRY RUN - No changes will be made")
        print(f"🔄 Would convert: {svg_path} → {png_path}")
        if args.update_references and references:
            update_qmd_references(references, svg_path, png_path, dry_run=True)
        if args.remove_svg:
            print(f"🗑️  Would remove: {svg_path}")
        return

    # Convert SVG to PNG
    if not convert_svg_to_png(svg_path, png_path):
        sys.exit(1)

    # Update QMD references only if explicitly requested
    if args.update_references and references:
        print(f"\n⚠️  Updating QMD references (as requested)...")
        files_updated = update_qmd_references(references, svg_path, png_path)
        print(f"📝 Updated {files_updated} QMD file(s)")
    elif references and not args.convert_only:
        print(f"\n⚠️  Found {len(references)} QMD reference(s) that need manual updating:")
        for file_path, line_num, line_content in references:
            print(f"   {file_path}:{line_num}")
        print("   Use --update-references flag if you want the tool to update them automatically")

    # Remove original SVG file only if explicitly requested
    if args.remove_svg:
        try:
            os.remove(svg_path)
            print(f"🗑️  Removed original SVG: {svg_path}")
        except Exception as e:
            print(f"⚠️  Could not remove SVG file: {e}")
    elif not args.convert_only:
        print(f"📁 Original SVG file kept: {svg_path}")
        print("   Use --remove-svg flag to remove it after verifying the conversion")

    print("✅ Conversion complete!")

if __name__ == "__main__":
    main()
