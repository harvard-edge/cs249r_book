#!/usr/bin/env python3
"""
EPUB Compression Tool for MLSysBook

This tool compresses EPUB files by optimizing embedded images while maintaining
EPUB format compliance. It extracts the EPUB, compresses images, and repacks
the archive following EPUB specifications.

Usage:
    python compress_epub.py --input input.epub --output output.epub [options]

Author: MLSysBook Team
License: MIT
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple

try:
    from PIL import Image
except ImportError:
    print("❌ Error: Pillow library is required. Install with: pip install Pillow")
    sys.exit(1)


class EPUBCompressor:
    """
    A class for compressing EPUB files by optimizing embedded images.

    This compressor maintains EPUB format compliance while reducing file size
    through image optimization techniques including quality reduction, resizing,
    and format optimization.
    """

    SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    def __init__(self, quality: int = 50, max_size: int = 1000, verbose: bool = False):
        """
        Initialize the EPUB compressor.

        Args:
            quality: JPEG compression quality (1-100, higher = better quality)
            max_size: Maximum dimension for image resizing (pixels)
            verbose: Enable verbose logging output
        """
        self.quality = quality
        self.max_size = max_size
        self.verbose = verbose
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging based on verbosity level."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(levelname)s: %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_inputs(self, input_path: Path, output_path: Path) -> None:
        """
        Validate input parameters and file paths.

        Args:
            input_path: Path to input EPUB file
            output_path: Path for output EPUB file

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If parameters are invalid
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input EPUB file not found: {input_path}")

        if not input_path.suffix.lower() == '.epub':
            raise ValueError(f"Input file must be an EPUB: {input_path}")

        if not 1 <= self.quality <= 100:
            raise ValueError(f"Quality must be between 1-100, got: {self.quality}")

        if not 100 <= self.max_size <= 5000:
            raise ValueError(f"Max size must be between 100-5000 pixels, got: {self.max_size}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"📖 Input EPUB: {input_path}")
        self.logger.info(f"📦 Output EPUB: {output_path}")
        self.logger.info(f"🎨 Image quality: {self.quality}%")
        self.logger.info(f"📏 Max image size: {self.max_size}px")

    def _compress_image(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Compress a single image file in place.

        Args:
            image_path: Path to the image file to compress

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        try:
            original_size = image_path.stat().st_size

            with Image.open(image_path) as img:
                img_format = img.format
                original_dimensions = img.size

                # Resize if image is too large
                if max(img.size) > self.max_size:
                    # Use backward-compatible resampling for older Pillow versions
                    try:
                        # Pillow >= 10.0.0
                        resample = Image.Resampling.LANCZOS
                    except AttributeError:
                        # Pillow < 10.0.0
                        resample = Image.LANCZOS

                    img.thumbnail((self.max_size, self.max_size), resample)
                    self.logger.debug(f"  📏 Resized {original_dimensions} → {img.size}")

                # Optimize based on format
                if img_format in ('JPEG', 'JPG'):
                    # Convert RGBA to RGB if needed (simpler approach)
                    if img.mode in ('RGBA', 'LA'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img

                    img.save(image_path, 'JPEG', quality=self.quality, optimize=True)

                elif img_format == 'PNG':
                    # Always try aggressive palette conversion for maximum compression
                    try:
                        img = img.convert('P', palette=Image.ADAPTIVE)
                        img.save(image_path, 'PNG', optimize=True)
                    except Exception:
                        # Fallback to original PNG optimization if palette conversion fails
                        img.save(image_path, 'PNG', optimize=True)

                else:
                    # For other formats, convert to JPEG if RGB, PNG if has transparency
                    if img.mode in ('RGBA', 'LA'):
                        img.save(image_path, 'PNG', optimize=True)
                    else:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(image_path, 'JPEG', quality=self.quality, optimize=True)

            new_size = image_path.stat().st_size
            compression_ratio = (1 - new_size / original_size) * 100 if original_size > 0 else 0

            self.logger.debug(f"  💾 {original_size:,} → {new_size:,} bytes ({compression_ratio:.1f}% reduction)")
            return True, None

        except Exception as e:
            error_msg = f"Failed to compress {image_path.name}: {str(e)}"
            self.logger.warning(f"  ⚠️ {error_msg}")
            return False, error_msg

    def _extract_epub(self, epub_path: Path, extract_dir: Path) -> None:
        """
        Extract EPUB contents to temporary directory.

        Args:
            epub_path: Path to EPUB file
            extract_dir: Directory to extract contents to
        """
        self.logger.info("📂 Extracting EPUB contents...")

        try:
            with zipfile.ZipFile(epub_path, 'r') as zip_file:
                zip_file.extractall(extract_dir)

            self.logger.debug(f"  ✅ Extracted to: {extract_dir}")

        except zipfile.BadZipFile:
            raise ValueError(f"Invalid EPUB file (not a valid ZIP): {epub_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract EPUB: {str(e)}")

    def _compress_images_in_directory(self, directory: Path) -> Tuple[int, int]:
        """
        Find and compress all images in the extracted EPUB directory.

        Args:
            directory: Root directory to search for images

        Returns:
            Tuple of (total_images: int, compressed_images: int)
        """
        self.logger.info("🎨 Compressing images...")

        image_files = []
        for ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(directory.rglob(f'*{ext}'))
            image_files.extend(directory.rglob(f'*{ext.upper()}'))

        total_images = len(image_files)
        compressed_images = 0

        if total_images == 0:
            self.logger.info("  ℹ️ No images found to compress")
            return 0, 0

        self.logger.info(f"  📊 Found {total_images} images to process")

        for i, image_path in enumerate(image_files, 1):
            self.logger.debug(f"  🖼️ [{i}/{total_images}] {image_path.name}")
            success, error = self._compress_image(image_path)
            if success:
                compressed_images += 1

        self.logger.info(f"  ✅ Successfully compressed {compressed_images}/{total_images} images")
        return total_images, compressed_images

    def _repack_epub(self, source_dir: Path, output_path: Path) -> None:
        """
        Repack the directory contents into a new EPUB file.

        Args:
            source_dir: Directory containing extracted and processed EPUB contents
            output_path: Path for the output EPUB file
        """
        self.logger.info("📦 Repacking EPUB...")

        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # First, add mimetype uncompressed (EPUB specification requirement)
                mimetype_path = source_dir / 'mimetype'
                if mimetype_path.exists():
                    zip_file.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
                    self.logger.debug("  📄 Added mimetype (uncompressed)")

                # Add all other files with compression
                files_added = 0
                for file_path in source_dir.rglob('*'):
                    if file_path.is_file() and file_path.name != 'mimetype':
                        arcname = file_path.relative_to(source_dir)
                        zip_file.write(file_path, arcname)
                        files_added += 1

                self.logger.debug(f"  ✅ Added {files_added} files to EPUB")

        except Exception as e:
            raise RuntimeError(f"Failed to repack EPUB: {str(e)}")

    def compress(self, input_path: Path, output_path: Path) -> dict:
        """
        Compress an EPUB file by optimizing embedded images.

        Args:
            input_path: Path to input EPUB file
            output_path: Path for compressed output EPUB file

        Returns:
            Dictionary with compression statistics

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input parameters are invalid
            RuntimeError: If compression process fails
        """
        # Validate inputs
        self._validate_inputs(input_path, output_path)

        # Get original file size
        original_size = input_path.stat().st_size
        self.logger.info(f"📊 Original EPUB size: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory(prefix='epub_compress_') as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Extract EPUB
                self._extract_epub(input_path, temp_path)

                # Compress images
                total_images, compressed_images = self._compress_images_in_directory(temp_path)

                # Repack EPUB
                self._repack_epub(temp_path, output_path)

            except Exception as e:
                # Clean up output file if it was partially created
                if output_path.exists():
                    output_path.unlink()
                raise e

        # Calculate final statistics
        final_size = output_path.stat().st_size
        compression_ratio = (1 - final_size / original_size) * 100 if original_size > 0 else 0

        stats = {
            'original_size': original_size,
            'final_size': final_size,
            'compression_ratio': compression_ratio,
            'size_saved': original_size - final_size,
            'total_images': total_images,
            'compressed_images': compressed_images
        }

        self.logger.info(f"✅ Compression complete!")
        self.logger.info(f"📊 Final size: {final_size:,} bytes ({final_size/1024/1024:.1f} MB)")
        self.logger.info(f"💾 Size reduction: {compression_ratio:.1f}% ({stats['size_saved']:,} bytes saved)")

        return stats


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compress EPUB files by optimizing embedded images while maintaining format compliance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input input.epub --output output.epub
  %(prog)s -i input.epub -o output.epub
  %(prog)s -i input.epub -o output.epub --quality 60 --max-size 1200
  %(prog)s -i input.epub -o output.epub --verbose
  %(prog)s -i input.epub -o output.epub -q 40 -s 800 -v

Quality Guidelines:
  90-100: Highest quality, larger files
  50-89:  Good quality, balanced size (recommended)
  35-49:  Acceptable quality, smaller files
  1-34:   Lower quality, smallest files

Max Size Guidelines:
  1000px: Default, optimized balance of quality and size
  1200px: Higher quality for detailed images
  800px:  Compact, suitable for basic readers
  600px:  Maximum compression for size-critical applications
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        metavar='EPUB_FILE',
        help='Path to the input EPUB file to compress'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        metavar='EPUB_FILE',
        help='Path for the compressed output EPUB file'
    )

    parser.add_argument(
        '--quality', '-q',
        type=int,
        default=50,
        metavar='N',
        help='JPEG compression quality (1-100, default: 50)'
    )

    parser.add_argument(
        '--max-size', '-s',
        type=int,
        default=1000,
        metavar='PIXELS',
        help='Maximum image dimension in pixels (default: 1000)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed progress information'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    return parser


def main() -> int:
    """
    Main entry point for the EPUB compression tool.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Create compressor instance
        compressor = EPUBCompressor(
            quality=args.quality,
            max_size=args.max_size,
            verbose=args.verbose
        )

        # Perform compression
        stats = compressor.compress(args.input, args.output)

        # Success message
        print(f"\n🎉 EPUB compression successful!")
        print(f"📁 Output: {args.output}")
        print(f"💾 Size reduction: {stats['compression_ratio']:.1f}%")
        print(f"🖼️ Images processed: {stats['compressed_images']}/{stats['total_images']}")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
