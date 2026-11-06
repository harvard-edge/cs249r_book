#!/usr/bin/env python3
"""
PDF Compression Tool for MLSysBook

This tool offers verified lossless stream optimization and Ghostscript quality
presets for academic textbooks. Lossless mode checks every rendered page and
document navigation before publishing the candidate. An already optimized input
may grow; reported size statistics retain that result rather than claiming savings.

Usage:
    python compress_pdf.py --input input.pdf --output output.pdf [options]
    python compress_pdf.py --input input.pdf --output output.pdf --quality lossless

Author: MLSysBook Team
License: MIT
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def verify_lossless_pdf(source, candidate) -> int:
    """Reject a rewrite that changes page content, navigation, or page geometry."""
    if len(source) != len(candidate):
        raise RuntimeError("Lossless compression changed the page count")
    if source.get_toc() != candidate.get_toc():
        raise RuntimeError("Lossless compression changed the document outline")
    if source.metadata != candidate.metadata or source.get_xml_metadata() != candidate.get_xml_metadata():
        raise RuntimeError("Lossless compression changed document metadata")
    if source.get_page_labels() != candidate.get_page_labels():
        raise RuntimeError("Lossless compression changed page labels")
    if source.embfile_names() != candidate.embfile_names():
        raise RuntimeError("Lossless compression changed attachment names")
    for name in source.embfile_names():
        if source.embfile_get(name) != candidate.embfile_get(name):
            raise RuntimeError(f"Lossless compression changed attachment {name}")
    for number, (before, after) in enumerate(zip(source, candidate), start=1):
        for box in ("mediabox", "cropbox", "trimbox", "bleedbox", "artbox", "rotation"):
            if getattr(before, box) != getattr(after, box):
                raise RuntimeError(f"Lossless compression changed page {number} {box}")
        if before.get_text() != after.get_text():
            raise RuntimeError(f"Lossless compression changed page {number} text")
        # Object numbers may change during garbage collection; link destinations must not.
        def links(page):
            return [{k: v for k, v in link.items() if k not in {"xref", "id"}}
                    for link in page.get_links()]
        if links(before) != links(after):
            raise RuntimeError(f"Lossless compression changed page {number} links")
        # Inspect every rendered page, including diagrams, rather than relying on text alone.
        if before.get_pixmap(dpi=72, alpha=True).samples != after.get_pixmap(dpi=72, alpha=True).samples:
            raise RuntimeError(f"Lossless compression changed page {number} rendering")
    return len(source)


def compress_lossless(input_path: Path, output_path: Path) -> dict:
    """Deflate and deduplicate PDF streams without resampling images or rewriting fonts.

    PyMuPDF's lossless save avoids the Ghostscript conversion that stalls on the
    Volume II manuscript. The candidate is verified before atomically replacing
    the destination; failed verification leaves any existing destination intact.
    """
    import pymupdf

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different PDF files")
    if input_path.suffix.lower() != ".pdf" or output_path.suffix.lower() != ".pdf":
        raise ValueError("Input and output must be PDF files")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_size = input_path.stat().st_size
    with pymupdf.open(input_path) as source:
        if not source.is_pdf or source.needs_pass or source.is_repaired:
            raise ValueError("Lossless compression requires an intact, unencrypted PDF")
        with tempfile.TemporaryDirectory(prefix=".pdf-lossless-", dir=output_path.parent) as scratch:
            candidate_path = Path(scratch) / "candidate.pdf"
            source.save(candidate_path, garbage=4, deflate=True,
                        deflate_images=True, deflate_fonts=True)
            with pymupdf.open(candidate_path) as candidate:
                verified_pages = verify_lossless_pdf(source, candidate)
            final_size = candidate_path.stat().st_size
            os.replace(candidate_path, output_path)
        pdf_format = source.metadata.get("format", "PDF")
    return {
        "original_size": original_size,
        "final_size": final_size,
        "compression_ratio": (1 - final_size / original_size) * 100,
        "size_saved": original_size - final_size,
        "quality_preset": "lossless",
        "pdf_compatibility": pdf_format,
        "verified_pages": verified_pages,
    }

class PDFCompressor:
    """
    A class for compressing PDF files using Ghostscript.

    This compressor uses Ghostscript with optimized settings for academic
    textbooks, balancing file size reduction with quality preservation
    for educational content.
    """

    # Ghostscript quality presets
    QUALITY_PRESETS = {
        'screen': '/screen',      # Lowest quality, smallest files (72 dpi)
        'ebook': '/ebook',        # Good for e-readers (150 dpi) - DEFAULT
        'printer': '/printer',    # Good for printing (300 dpi)
        'prepress': '/prepress',  # Highest quality (300+ dpi)
        'default': '/default',    # Ghostscript default settings
        'minimal': '/ebook'       # Minimal mode - matches original workflow exactly
    }

    def __init__(self, quality: str = 'ebook', compatibility: str = '1.4', verbose: bool = False):
        """
        Initialize the PDF compressor.

        Args:
            quality: Compression quality preset (screen, ebook, printer, prepress, default)
            compatibility: PDF compatibility level (1.3, 1.4, 1.5, 1.6, 1.7)
            verbose: Enable verbose logging output
        """
        self.quality = quality
        self.compatibility = compatibility
        self.verbose = verbose
        self._setup_logging()
        self._validate_dependencies()

    def _setup_logging(self) -> None:
        """Configure logging based on verbosity level."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(levelname)s: %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_dependencies(self) -> None:
        """Check if Ghostscript is available and determine the correct executable."""
        # Determine platform-specific Ghostscript executable
        if platform.system() == 'Windows':
            # On Windows, try gswin64c first, then gs
            gs_candidates = ['gswin64c', 'gs']
        else:
            # On Linux/macOS, use gs
            gs_candidates = ['gs']

        self.gs_executable = None
        for gs_cmd in gs_candidates:
            try:
                result = subprocess.run([gs_cmd, '--version'],
                                      capture_output=True, text=True, check=True)
                gs_version = result.stdout.strip()
                self.gs_executable = gs_cmd
                self.logger.debug(f"Found Ghostscript executable: {gs_cmd}")
                self.logger.debug(f"Ghostscript version: {gs_version}")
                break
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        if not self.gs_executable:
            raise RuntimeError(
                "Ghostscript is not installed or not in PATH. "
                f"Tried: {', '.join(gs_candidates)}. "
                "Please install Ghostscript to use this tool."
            )

    def _validate_inputs(self, input_path: Path, output_path: Path) -> None:
        """
        Validate input parameters and file paths.

        Args:
            input_path: Path to input PDF file
            output_path: Path for output PDF file

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If parameters are invalid
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input PDF file not found: {input_path}")

        if not input_path.suffix.lower() == '.pdf':
            raise ValueError(f"Input file must be a PDF: {input_path}")

        if self.quality not in self.QUALITY_PRESETS:
            raise ValueError(f"Quality must be one of {list(self.QUALITY_PRESETS.keys())}, got: {self.quality}")

        if self.compatibility not in ['1.3', '1.4', '1.5', '1.6', '1.7']:
            raise ValueError(f"Compatibility must be 1.3-1.7, got: {self.compatibility}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"📄 Input PDF: {input_path}")
        self.logger.info(f"📦 Output PDF: {output_path}")
        self.logger.info(f"🎨 Quality preset: {self.quality}")
        self.logger.info(f"📋 PDF compatibility: {self.compatibility}")

    def _format_file_size(self, size_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _build_ghostscript_command(self, input_path: Path, output_path: Path) -> list[str]:
        """
        Build the Ghostscript command with optimized parameters.

        Args:
            input_path: Path to input PDF file
            output_path: Path for output PDF file

        Returns:
            List of command arguments for subprocess
        """
        quality_setting = self.QUALITY_PRESETS[self.quality]

        if self.quality == 'minimal':
            # Minimal mode: exactly match original workflow commands
            command = [
                self.gs_executable,
                '-sDEVICE=pdfwrite',
                f'-dCompatibilityLevel={self.compatibility}',
                f'-dPDFSETTINGS={quality_setting}',
                '-dNOPAUSE',
                '-dQUIET' if not self.verbose else '-dNOQUIET',
                '-dBATCH',
                f'-sOutputFile={output_path}',
                str(input_path)
            ]
        else:
            # Enhanced mode: with additional quality improvements
            command = [
                self.gs_executable,  # Use platform-specific executable
                '-sDEVICE=pdfwrite',
                f'-dCompatibilityLevel={self.compatibility}',
                f'-dPDFSETTINGS={quality_setting}',
                '-dNOPAUSE',
                '-dQUIET' if not self.verbose else '-dNOQUIET',
                '-dBATCH',
                '-dSAFER',  # Security setting
                '-dAutoRotatePages=/None',  # Preserve page orientation
                '-dColorImageDownsampleType=/Bicubic',  # Better image quality
                '-dGrayImageDownsampleType=/Bicubic',
                '-dMonoImageDownsampleType=/Bicubic',
                f'-sOutputFile={output_path}',
                str(input_path)
            ]

        return command

    def compress(self, input_path: Path, output_path: Path) -> dict:
        """
        Compress a PDF file using Ghostscript.

        Args:
            input_path: Path to input PDF file
            output_path: Path for compressed output PDF file

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
        self.logger.info(f"📊 Original PDF size: {original_size:,} bytes ({self._format_file_size(original_size)})")

        # Build Ghostscript command
        command = self._build_ghostscript_command(input_path, output_path)

        self.logger.info("🔄 Compressing PDF with Ghostscript...")
        self.logger.debug(f"Command: {' '.join(command)}")

        try:
            # Run Ghostscript compression
            result = subprocess.run(
                command,
                check=True,
                capture_output=not self.verbose,
                text=True
            )

            self.logger.debug(f"Ghostscript return code: {result.returncode}")

        except subprocess.CalledProcessError as e:
            # Clean up output file if it was partially created
            if output_path.exists():
                output_path.unlink()

            error_msg = f"Ghostscript compression failed (exit code {e.returncode})"
            if e.stderr:
                error_msg += f": {e.stderr.strip()}"

            raise RuntimeError(error_msg)

        except Exception as e:
            # Clean up output file if it was partially created
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"PDF compression failed: {str(e)}")

        # Verify output file was created
        if not output_path.exists():
            raise RuntimeError("Ghostscript completed but output file was not created")

        # Calculate final statistics
        final_size = output_path.stat().st_size
        compression_ratio = (1 - final_size / original_size) * 100 if original_size > 0 else 0

        stats = {
            'original_size': original_size,
            'final_size': final_size,
            'compression_ratio': compression_ratio,
            'size_saved': original_size - final_size,
            'quality_preset': self.quality,
            'pdf_compatibility': self.compatibility
        }

        self.logger.info(f"✅ Compression complete!")
        self.logger.info(f"📊 Final size: {final_size:,} bytes ({self._format_file_size(final_size)})")
        self.logger.info(f"💾 Size reduction: {compression_ratio:.1f}% ({self._format_file_size(stats['size_saved'])} saved)")

        return stats


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compress PDFs with verified lossless optimization or Ghostscript quality presets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input input.pdf --output output.pdf
  %(prog)s -i input.pdf -o output.pdf
  %(prog)s -i input.pdf -o output.pdf --quality printer
  %(prog)s -i input.pdf -o output.pdf --verbose
  %(prog)s -i input.pdf -o output.pdf -q screen -c 1.5 -v

Quality Presets:
  screen:    Lowest quality, smallest files (72 dpi) - for web viewing
  ebook:     Good for e-readers (150 dpi) - DEFAULT, balanced size/quality
  printer:   Good for printing (300 dpi) - higher quality
  prepress:  Highest quality (300+ dpi) - for professional printing
  default:   Ghostscript default settings - no optimization
  minimal:   Exact match to original workflow commands - for compatibility
  lossless:  Deflate/deduplicate streams; verify every page, link, and outline

PDF Compatibility:
  1.3: Oldest, most compatible (Acrobat 4.0+)
  1.4: Good compatibility (Acrobat 5.0+) - DEFAULT
  1.5: Modern features (Acrobat 6.0+)
  1.6: Advanced features (Acrobat 7.0+)
  1.7: Latest features (Acrobat 8.0+)
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        metavar='PDF_FILE',
        help='Path to the input PDF file to compress'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        metavar='PDF_FILE',
        help='Path for the compressed output PDF file'
    )

    parser.add_argument(
        '--quality', '-q',
        choices=['screen', 'ebook', 'printer', 'prepress', 'default', 'minimal', 'lossless'],
        default='ebook',
        help='Compression quality preset (default: ebook)'
    )

    parser.add_argument(
        '--compatibility', '-c',
        choices=['1.3', '1.4', '1.5', '1.6', '1.7'],
        default='1.4',
        metavar='VERSION',
        help='Ghostscript PDF compatibility level; lossless preserves the source format (default: 1.4)'
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
    Main entry point for the PDF compression tool.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        if args.quality == 'lossless':
            stats = compress_lossless(args.input, args.output)
            print(f"Verified identical rendering and content on all {stats['verified_pages']} pages.")
        else:
            compressor = PDFCompressor(
                quality=args.quality,
                compatibility=args.compatibility,
                verbose=args.verbose
            )
            stats = compressor.compress(args.input, args.output)

        # Success message
        print(f"\n🎉 PDF compression successful!")
        print(f"📁 Output: {args.output}")
        print(f"💾 Size reduction: {stats['compression_ratio']:.1f}%")
        print(f"🎨 Quality preset: {stats['quality_preset']}")

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
