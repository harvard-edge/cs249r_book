#!/usr/bin/env python3
"""
External Image Downloader for Quarto Markdown Files

This script automatically downloads external images referenced in markdown files
and organizes them locally according to the project's directory structure.

DESCRIPTION:
    Processes .qmd files to find markdown images with #fig references that use
    external URLs (http/https). Downloads these images and organizes them by
    file type in subdirectories, then updates the markdown to reference local paths.

FEATURES:
    - Smart pattern recognition for markdown figures
    - Automatic file type detection and organization
    - Unique filename generation to prevent conflicts
    - Preserves original figure captions and IDs
    - Safe dry-run mode for previewing changes
    - Handles nested brackets/parentheses in captions
    - Error handling and progress logging

DIRECTORY STRUCTURE:
    Images are organized as: chapter/images/{file_type}/filename
    - chapter/images/png/ for PNG files
    - chapter/images/jpeg/ for JPEG files
    - chapter/images/pdf/ for PDF files
    - etc.

EXAMPLE TRANSFORMATION:
    Before: ![caption](https://example.com/image.png){#fig-id}
    After:  ![caption](images/png/id_hash123.png){#fig-id}

USAGE:
    # Process all files recursively from current directory
    python3 download_external_images.py -d .

    # Process all files in specific directory
    python3 download_external_images.py -d book/contents/core

    # Process single file
    python3 download_external_images.py -f path/to/file.qmd

    # Preview changes without downloading
    python3 download_external_images.py -d . --dry-run

REQUIREMENTS:
    - Python 3.6+
    - requests library
    - Internet connection for downloads
"""

import os
import re
import requests
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import argparse
from typing import List, Tuple, Optional
import logging

# Set up logging - only show warnings and errors
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageDownloader:
    """
    Main class for downloading and organizing external images from Quarto markdown files.

    This class handles the entire workflow from finding .qmd files to downloading
    images and updating markdown references.

    Attributes:
        base_dir (Path): Base directory containing .qmd files to process
        session (requests.Session): HTTP session with browser-like headers
    """

    def __init__(self, base_dir: str):
        """
        Initialize the ImageDownloader.

        Args:
            base_dir (str): Base directory to search for .qmd files
        """
        self.base_dir = Path(base_dir)
        self.session = requests.Session()
        # Use browser-like headers to avoid being blocked by some servers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def find_qmd_files(self) -> List[Path]:
        """
        Find all .qmd files recursively in the base directory.

        Recursively searches through all subdirectories to find any .qmd files,
        regardless of naming patterns or directory structure.

        Returns:
            List[Path]: List of paths to .qmd files found
        """
        qmd_files = []
        if not self.base_dir.exists():
            logger.warning(f"❌ Base directory does not exist: {self.base_dir}")
            return qmd_files

        # Recursively find all .qmd files
        for qmd_file in self.base_dir.rglob("*.qmd"):
            if qmd_file.is_file():
                qmd_files.append(qmd_file)
                logger.debug(f"Found .qmd file: {qmd_file}")

        print(f"📄 Found {len(qmd_files)} .qmd files")
        return qmd_files

    def extract_figure_images(self, content: str) -> List[Tuple[str, str, str, str, str]]:
        r"""
        Extract markdown images with or without #fig references.
        Returns list of tuples: (full_match, caption, url, fig_id, attributes)

        Simple strategy: Find ![, then find ALL ](url) patterns up to optional {attrs}.
        Take the LAST ](url) as the image URL (others are citations).
        """
        matches = []

        # Simple approach: find ![, capture everything until } or end of line
        # Then parse the ](url) patterns
        lines = content.split('\n')

        for line in lines:
            if '![' not in line:
                continue

            # Find all image patterns on this line
            idx = 0
            while idx < len(line):
                start = line.find('![', idx)
                if start == -1:
                    break

                # Find the end: either {...} or next ![
                end_brace = line.find('}', start)
                next_img = line.find('![', start + 2)

                if end_brace != -1 and (next_img == -1 or end_brace < next_img):
                    end = end_brace + 1
                elif next_img != -1:
                    end = next_img
                else:
                    end = len(line)

                full_match = line[start:end]

                # Find ALL ](url) patterns in this match
                url_patterns = list(re.finditer(r'\]\(([^)]+)\)', full_match))

                if url_patterns:
                    # Take the LAST one - that's the image URL
                    url = url_patterns[-1].group(1).strip()

                    # Parse fig_id and attributes
                    fig_id = None
                    attributes = ""
                    attrs_match = re.search(r'\{([^}]+)\}', full_match)
                    if attrs_match:
                        attrs_block = attrs_match.group(1)
                        fig_match = re.search(r'#(fig-[^\s}]+)', attrs_block)
                        if fig_match:
                            fig_id = fig_match.group(1)
                        attributes = attrs_block.strip()

                    # Extract caption
                    caption_match = re.search(r'!\[([^\]]+)\]', full_match)
                    caption = caption_match.group(1) if caption_match else ""

                    # Only flag external URLs
                    if url.lower().startswith(('http://', 'https://')):
                        if not fig_id:
                            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                            fig_id = f"fig-auto-{url_hash}"

                        matches.append((full_match, caption, url, fig_id, attributes))

                idx = end

        return matches

    def get_file_extension(self, url: str, response_headers: dict) -> str:
        """Determine file extension from URL or content type."""
        # First try to get extension from URL
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        if path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.webp')):
            return path.split('.')[-1]

        # Try to get from content type
        content_type = response_headers.get('content-type', '').lower()
        content_type_map = {
            'image/png': 'png',
            'image/jpeg': 'jpg',
            'image/jpg': 'jpg',
            'image/gif': 'gif',
            'image/svg+xml': 'svg',
            'application/pdf': 'pdf',
            'image/webp': 'webp'
        }

        for ct, ext in content_type_map.items():
            if ct in content_type:
                return ext

        # Default to png if we can't determine
        logger.warning(f"⚠️ Could not determine file type for {url}, defaulting to png")
        return 'png'

    def generate_filename(self, url: str, fig_id: str, extension: str) -> str:
        """
        Generate a filename based on the original URL filename.

        Extracts the original filename from the URL and uses it, adding a hash
        suffix only if needed for uniqueness. This makes filenames more descriptive.

        Args:
            url (str): Original URL of the image
            fig_id (str): Figure identifier from markdown (e.g., 'fig-example')
            extension (str): File extension (e.g., 'png', 'jpg')

        Returns:
            str: Generated filename (e.g., 'oranges-frogs.png')

        Example:
            URL: https://example.com/path/oranges-frogs_abc123.png?params
            Result: oranges-frogs.png
        """
        # Parse the URL to get the path
        parsed_url = urlparse(url)
        path = parsed_url.path

        # Get the filename from the path
        original_filename = os.path.basename(path)

        # Remove any existing extension and parameters
        base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename

        # Clean up the filename - remove trailing hashes or IDs that are common in CDN URLs
        # e.g., "oranges-frogs_nHEaTqne53" -> "oranges-frogs"
        import re
        # Remove trailing underscore + hash patterns (common in CDN URLs)
        base_name = re.sub(r'_[a-zA-Z0-9]{8,}$', '', base_name)

        # If base_name is empty or too short, fall back to fig_id
        if not base_name or len(base_name) < 3:
            base_name = fig_id.replace('fig-', '').replace('auto-', '')
            if not base_name:
                # Last resort: use URL hash
                base_name = hashlib.md5(url.encode()).hexdigest()[:12]

        # Sanitize the filename - keep only alphanumeric, hyphens, and underscores
        base_name = re.sub(r'[^a-zA-Z0-9_-]', '-', base_name)
        # Remove multiple consecutive hyphens
        base_name = re.sub(r'-+', '-', base_name)
        # Remove leading/trailing hyphens
        base_name = base_name.strip('-')

        return f"{base_name}.{extension}"

    def download_image(self, url: str, output_path: Path) -> bool:
        """Download image from URL to output path."""
        try:
            logger.info(f"📦 Downloading {url}")
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"✅ Successfully downloaded to {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to download {url}: {e}")
            return False

    def process_file(self, qmd_file: Path, dry_run: bool = False, confirm: bool = False) -> int:
        """Process a single .qmd file and download its external images."""
        logger.info(f"📄 Processing {qmd_file}")

        try:
            with open(qmd_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"❌ Failed to read {qmd_file}: {e}")
            return 0

        figure_images = self.extract_figure_images(content)

        if not figure_images:
            logger.info(f"📄 No external figure images found in {qmd_file}")
            return 0

        logger.info(f"🔍 Found {len(figure_images)} external figure images")

        downloaded_count = 0
        new_content = content

        for full_match, caption, url, fig_id, attributes in figure_images:
            logger.info(f"🔍 Processing {fig_id}: {url}")

            if dry_run:
                logger.info(f"🧪 [DRY RUN] Would download {url} for {fig_id}")
                continue

            # Confirmation mode
            if confirm:
                print(f"\n🤔 Download external image?")
                print(f"   📄 File: {qmd_file}")
                print(f"   🔍 Figure: {fig_id}")
                print(f"   🌐 URL: {url}")
                response = input("   Download? [y/N]: ").lower().strip()
                if response not in ['y', 'yes']:
                    logger.info(f"⏭️ Skipped {fig_id}")
                    continue

            # Get file extension by making a HEAD request
            try:
                head_response = self.session.head(url, timeout=10)
                extension = self.get_file_extension(url, head_response.headers)
            except:
                # If HEAD request fails, try GET and determine extension
                extension = 'png'  # fallback

            # Determine chapter directory and images subdirectory
            chapter_name = qmd_file.parent.name
            images_dir = qmd_file.parent / "images" / extension

            # Generate filename
            filename = self.generate_filename(url, fig_id, extension)
            output_path = images_dir / filename

            # Create replacement markdown with preserved attributes
            local_path = f"images/{extension}/{filename}"

            # Only include fig_id if it's not auto-generated
            if fig_id.startswith("fig-auto-"):
                # Image didn't have a fig-id originally, don't add one
                if attributes:
                    replacement = f"![{caption}]({local_path}){{{attributes}}}"
                else:
                    replacement = f"![{caption}]({local_path})"
            else:
                # Image had a fig-id, preserve it
                if attributes:
                    replacement = f"![{caption}]({local_path}){{#{fig_id} {attributes}}}"
                else:
                    replacement = f"![{caption}]({local_path}){{#{fig_id}}}"

            # Check if file already exists
            if output_path.exists():
                logger.info(f"📁 File already exists: {output_path}")
                # Update the markdown anyway in case the reference is wrong
                new_content = new_content.replace(full_match, replacement)
                continue

            # Download the image
            if self.download_image(url, output_path):
                # Update the markdown content to use local path
                new_content = new_content.replace(full_match, replacement)
                downloaded_count += 1
            else:
                logger.warning(f"⚠️ Skipping update for failed download: {fig_id}")

        # Write updated content back to file if we made changes
        if new_content != content:
            try:
                with open(qmd_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                total_updates = len(figure_images)  # Count all external images that were processed
                logger.info(f"✅ Updated {qmd_file} with {total_updates} local image references ({downloaded_count} downloaded)")
            except Exception as e:
                logger.error(f"❌ Failed to write updated content to {qmd_file}: {e}")

        return downloaded_count

    def process_all_files(self, dry_run: bool = False, confirm: bool = False) -> Tuple[int, int]:
        """
        Process all .qmd files in the base directory.

        Returns:
            Tuple of (files_processed, images_downloaded)
        """
        qmd_files = self.find_qmd_files()
        print(f"🔍 Found {len(qmd_files)} .qmd files to process")

        total_downloaded = 0
        files_processed = 0

        for qmd_file in qmd_files:
            try:
                downloaded = self.process_file(qmd_file, dry_run, confirm)
                total_downloaded += downloaded
                files_processed += 1
            except Exception as e:
                logger.error(f"❌ Error processing {qmd_file}: {e}")

        return files_processed, total_downloaded

    def validate_external_images(self, ignore_external: bool = False) -> Tuple[int, List[Tuple[Path, str, str]]]:
        """
        Validate mode: Find external images without downloading.

        Args:
            ignore_external (bool): If True, only warn about external images

        Returns:
            Tuple of (total_files_processed, list_of_external_images)
            Each external image is (file_path, fig_id, url)
        """
        qmd_files = self.find_qmd_files()
        print(f"🔍 Validating {len(qmd_files)} .qmd files for external images")

        all_external_images = []

        for qmd_file in qmd_files:
            try:
                with open(qmd_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                figure_images = self.extract_figure_images(content)

                for full_match, caption, url, fig_id, attributes in figure_images:
                    all_external_images.append((qmd_file, fig_id, url))

                    if ignore_external:
                        logger.warning(f"⚠️ External image in {qmd_file}: {fig_id} → {url}")
                    else:
                        logger.error(f"❌ External image found in {qmd_file}: {fig_id} → {url}")

            except Exception as e:
                logger.error(f"❌ Error reading {qmd_file}: {e}")

        return len(qmd_files), all_external_images

def main():
    parser = argparse.ArgumentParser(
        description="Download external images from Quarto markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d .                         # Process all files recursively from current directory
  %(prog)s -d book/contents/core        # Process all files in specific directory
  %(prog)s -f chapter/file.qmd          # Process single file
  %(prog)s -d . --dry-run               # Preview what would be downloaded
  %(prog)s -d . --confirm               # Ask for confirmation before each download
  %(prog)s --validate book/contents/core # Validate mode: fail if external images found (pre-commit)
  %(prog)s --validate . --ignore-external # Validate mode: warn only (allow external images)
        """)

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-f", "--file", type=str, metavar="FILE",
                           help="Process only a specific file")
    mode_group.add_argument("-d", "--directory", type=str, metavar="DIR",
                           help="Process all .qmd files recursively in specified directory")
    mode_group.add_argument("--validate", type=str, metavar="DIR",
                           help="Validate mode: check for external images and fail if found (for pre-commit)")

    # Options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be downloaded without actually downloading")
    parser.add_argument("--confirm", action="store_true",
                       help="Ask for confirmation before downloading each image")
    parser.add_argument("--ignore-external", action="store_true",
                       help="Allow external images in validate mode (warning only)")

    args = parser.parse_args()

    print("🔍 External Image Downloader")
    print("=" * 40)

    if args.validate:
        # Validation mode (for pre-commit hooks)
        downloader = ImageDownloader(args.validate)
        files_processed, external_images = downloader.validate_external_images(args.ignore_external)

        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   📁 Directory: {args.validate}")
        print(f"   📄 Files processed: {files_processed}")
        print(f"   🌐 External images found: {len(external_images)}")

        if external_images:
            if args.ignore_external:
                print(f"   ⚠️ Mode: WARN ONLY (external images allowed)")
                print(f"\n💡 To fix external images, run:")
                print(f"   python3 {Path(__file__).name} -d {args.validate}")
                return 0
            else:
                print(f"   ❌ Mode: STRICT (external images not allowed)")
                print(f"\n💡 Found {len(external_images)} external images that need to be downloaded:")
                for file_path, fig_id, url in external_images[:5]:  # Show first 5
                    print(f"     📄 {file_path}: {fig_id}")
                if len(external_images) > 5:
                    print(f"     ... and {len(external_images) - 5} more")
                print(f"\n💡 To fix, run:")
                print(f"   python3 {Path(__file__).name} -d {args.validate}")
                print(f"\n💡 Or to allow external images, use --ignore-external flag")
                return 1
        else:
            print(f"   ✅ No external images found")
            return 0

    elif args.file:
        # Process single file
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return 1

        # Use the file's parent directory for organizing images
        downloader = ImageDownloader(file_path.parent)
        downloaded = downloader.process_file(file_path, args.dry_run, args.confirm)

        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   📄 File: {file_path}")
        print(f"   📦 Images downloaded: {downloaded}")
        if args.dry_run:
            print(f"   🧪 Mode: DRY RUN (no files changed)")
        elif args.confirm:
            print(f"   🤔 Mode: CONFIRM (user approval required)")
        elif downloaded > 0:
            print(f"   ✅ File updated successfully")
        else:
            print(f"   ℹ️  No external images found in file")

    elif args.directory:
        # Process all files in specified directory
        downloader = ImageDownloader(args.directory)
        files_processed, downloaded = downloader.process_all_files(args.dry_run, args.confirm)

        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   📁 Directory: {args.directory}")
        print(f"   📄 Files processed: {files_processed}")
        print(f"   📦 Images downloaded: {downloaded}")
        if args.dry_run:
            print(f"   🧪 Mode: DRY RUN (no files changed)")
        elif args.confirm:
            print(f"   🤔 Mode: CONFIRM (user approval required)")
        elif downloaded > 0:
            print(f"   ✅ Operation completed successfully")
        else:
            print(f"   ℹ️  No external images found in {files_processed} files")

    return 0

if __name__ == "__main__":
    exit(main())
