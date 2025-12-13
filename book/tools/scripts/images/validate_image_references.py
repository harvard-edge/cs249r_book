#!/usr/bin/env python3
"""
Image Reference Validator for Quarto Markdown Files

This script validates that all images referenced in markdown files actually exist
on disk. It finds missing image references and reports broken links to help
maintain build reliability.

DESCRIPTION:
    Scans .qmd files for markdown image references and verifies that the
    referenced image files exist on the filesystem. Useful for catching
    missing images before build failures occur.

FEATURES:
    - Detects all markdown image syntax: ![caption](path) and ![caption](path){attributes}
    - Resolves relative paths correctly based on the .qmd file location
    - Checks both simple images and figure references with #fig- IDs
    - Provides clear reporting of missing files with file paths
    - Exit codes suitable for CI/pre-commit integration
    - Summary statistics and detailed missing file reports

USAGE:
    # Check single file
    python3 validate_image_references.py -f path/to/file.qmd

    # Check all files in directory (recursive)
    python3 validate_image_references.py -d book/contents/

    # Quiet mode (only show missing images)
    python3 validate_image_references.py -d book/contents/ --quiet

EXIT CODES:
    0: All image references are valid
    1: Missing image references found
    2: Error reading files

AUTHORS: AI Assistant
VERSION: 1.0.0
"""

import os
import re
import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Set
import logging

# Set up logging - only show warnings and errors by default
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_project_root(marker: str = 'quarto/_quarto.yml') -> Path:
    """Find the project root by looking for a specific file/directory."""
    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent
    return Path.cwd() # Fallback to current working directory if marker not found


def realcase(path):
    dirname, basename = os.path.split(path)
    if dirname == path:
        return dirname
    dirname = realcase(dirname)
    basename = os.path.normcase(basename)
    for child in os.listdir(dirname):
        if os.path.normcase(child) == basename:
            return os.path.join(dirname, child)
    raise FileNotFoundError(f'{path} not found')


class ImageReferenceValidator:
    """
    Validates that all image references in markdown files exist on disk.
    """

    def __init__(self, base_dir: str):
        """
        Initialize the validator.

        Args:
            base_dir (str): Base directory to search for .qmd files
        """
        self.project_root = find_project_root()
        self.base_dir = self.project_root / base_dir
        if not self.base_dir.exists():
            raise ValueError(f"Directory does not exist: {self.base_dir}")

    def find_qmd_files(self) -> List[Path]:
        """Find all .qmd files in the base directory recursively."""
        qmd_files = list(self.base_dir.rglob("*.qmd"))
        return sorted(qmd_files)

    def extract_image_references(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract all image references from markdown content.

        Args:
            content (str): Markdown content to parse

        Returns:
            List of tuples: (full_match, image_path)
        """
        # This captures both simple images and figure references, and handles nested brackets in captions.
        pattern = r'!\[(?:[^\]]|\[[^\]]*\])*\]\(([^)]+)\)(?:\{[^}]*\})?'

        matches = []
        for match in re.finditer(pattern, content):
            full_match = match.group(0)
            image_path = match.group(1).strip()

            # Check for valid image extension
            if not any(image_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']):
                continue

            # Skip external URLs (already handled by external image validator)
            if image_path.startswith(('http://', 'https://')):
                continue

            matches.append((full_match, image_path))

        return matches

    def validate_file(self, qmd_file: Path, quiet: bool = False) -> List[Tuple[str, str]]:
        """
        Validate image references in a single .qmd file.

        Args:
            qmd_file (Path): Path to the .qmd file to validate
            quiet (bool): If True, suppress progress messages

        Returns:
            List of missing image references: (reference_text, resolved_path_or_error)
        """
        try:
            with open(qmd_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"❌ Error reading {qmd_file}: {e}")
            return [("ERROR_READING_FILE", str(qmd_file))]

        image_references = self.extract_image_references(content)
        missing_images = []

        if not quiet and image_references:
            print(f"📄 Checking {len(image_references)} image references in {qmd_file}")

        for full_match, image_path in image_references:
            # Skip external URLs
            if image_path.startswith(('http://', 'https://')):
                continue

            # Resolve path relative to the .qmd file's directory
            qmd_dir = qmd_file.parent
            referenced_path_str = str((qmd_dir / image_path).resolve(strict=False))

            try:
                true_path_str = realcase(referenced_path_str)

                if referenced_path_str != true_path_str:
                    relative_qmd_path = qmd_file.relative_to(self.project_root)
                    error_message = (
                        f"Case mismatch in {relative_qmd_path}: "
                        f"reference is '{Path(referenced_path_str).name}', but on disk it is '{Path(true_path_str).name}'"
                    )
                    missing_images.append((full_match, error_message))
                    if not quiet:
                        print(f"❌ {error_message}")
                else:
                    if not quiet:
                        print(f"✅ Found: {image_path}")

            except FileNotFoundError:
                missing_images.append((full_match, referenced_path_str))
                if not quiet:
                    print(f"❌ Missing image: {image_path}")
                    print(f"   📍 Expected at: {referenced_path_str}")

        return missing_images

    def validate_all_files(self, quiet: bool = False) -> Tuple[int, List[Tuple[Path, str, str]]]:
        """
        Validate image references in all .qmd files.

        Args:
            quiet (bool): If True, suppress progress messages

        Returns:
            Tuple of (files_processed, list_of_missing_images)
            Each missing image is (qmd_file, reference_text, resolved_path)
        """
        qmd_files = self.find_qmd_files()

        if not quiet:
            print(f"🔍 Found {len(qmd_files)} .qmd files")
            print(f"📁 Validating image references in {self.base_dir}")
            print()

        all_missing_images = []
        files_with_issues = 0

        for qmd_file in qmd_files:
            missing_in_file = self.validate_file(qmd_file, quiet)

            if missing_in_file:
                files_with_issues += 1
                for reference_text, resolved_path in missing_in_file:
                    all_missing_images.append((qmd_file, reference_text, resolved_path))

            if not quiet and missing_in_file:
                print()  # Add spacing between files with issues

        return len(qmd_files), all_missing_images

def main():
    parser = argparse.ArgumentParser(
        description="Validate that all image references in Quarto markdown files exist on disk",
        epilog="Exit codes: 0=success, 1=missing images found, 2=file read errors"
    )

    # Mutually exclusive mode group
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-f", "--file",
        type=str,
        metavar="FILE",
        help="Validate image references in a specific .qmd file"
    )
    mode_group.add_argument(
        "-d", "--directory",
        type=str,
        metavar="DIR",
        help="Validate image references in all .qmd files recursively in specified directory"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode: only show summary and missing images"
    )

    args = parser.parse_args()

    if not args.quiet:
        print("🔍 Image Reference Validator")
        print("=" * 40)

    try:
        if args.file:
            # Single file mode
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ File not found: {args.file}")
                return 2

            validator = ImageReferenceValidator(file_path.parent)
            missing_images = validator.validate_file(file_path, args.quiet)

            # Summary for single file
            if not args.quiet:
                print()
                print("📊 VALIDATION SUMMARY:")
                print(f"   📄 File: {file_path}")
                print(f"   🖼️  Missing images: {len(missing_images)}")

            if missing_images:
                if args.quiet:
                    print(f"❌ Missing images in {file_path}:")
                    for reference_text, resolved_path in missing_images:
                        print(f"   {resolved_path}")
                else:
                    print(f"   ❌ Status: FAILED")
                return 1
            else:
                if not args.quiet:
                    print(f"   ✅ Status: PASSED")
                elif args.quiet:
                    print(f"✅ All image references valid in {file_path}")
                return 0

        elif args.directory:
            # Directory mode
            validator = ImageReferenceValidator(args.directory)
            files_processed, all_missing_images = validator.validate_all_files(args.quiet)

            # Group missing images by file for better reporting
            files_with_missing = {}
            for qmd_file, reference_text, resolved_path in all_missing_images:
                if qmd_file not in files_with_missing:
                    files_with_missing[qmd_file] = []
                files_with_missing[qmd_file].append((reference_text, resolved_path))

            # Summary
            if not args.quiet:
                print("📊 VALIDATION SUMMARY:")
                print(f"   📁 Directory: {args.directory}")
                print(f"   📄 Files processed: {files_processed}")
                print(f"   🖼️  Missing images: {len(all_missing_images)}")
                print(f"   📋 Files with issues: {len(files_with_missing)}")

            if all_missing_images:
                if args.quiet:
                    print(f"❌ Found {len(all_missing_images)} missing image references:")
                    for qmd_file, reference_text, resolved_path in all_missing_images:
                        print(f"   {resolved_path}")
                else:
                    print(f"   ❌ Status: FAILED")
                    print()
                    print("💡 Missing image details:")
                    for qmd_file, missing_images in files_with_missing.items():
                        print(f"     📄 {qmd_file}: {len(missing_images)} missing")
                        for reference_text, resolved_path in missing_images:
                            print(f"       - {resolved_path}")

                return 1
            else:
                if not args.quiet:
                    print(f"   ✅ Status: PASSED")
                elif args.quiet:
                    print(f"✅ All image references valid in {files_processed} files")
                return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
