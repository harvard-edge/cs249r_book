#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVG Cleanup Script - Cross-Platform Version
============================================
Removes control characters from SVG files that can cause rendering issues
in browsers and other tools. Control characters are often introduced by
LaTeX -> SVG conversion tools.

This is a cross-platform Python replacement for clean-svgs.sh that works
on Windows, macOS, and Linux.
"""

import argparse
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import re


class Colors:
    """ANSI color codes for terminal output."""

    def __init__(self):
        # Only use colors if stdout is a terminal
        if sys.stdout.isatty():
            self.RED = '\033[0;31m'
            self.GREEN = '\033[0;32m'
            self.YELLOW = '\033[1;33m'
            self.BLUE = '\033[0;34m'
            self.PURPLE = '\033[0;35m'
            self.CYAN = '\033[0;36m'
            self.NC = '\033[0m'  # No Color
        else:
            self.RED = self.GREEN = self.YELLOW = ''
            self.BLUE = self.PURPLE = self.CYAN = self.NC = ''


class SVGCleaner:
    """Cross-platform SVG file cleaner that removes control characters."""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.colors = Colors()
        self.script_name = Path(__file__).name

        # Control characters to remove (same as bash version)
        # \000-\010: NULL through BACKSPACE
        # \013: VERTICAL TAB
        # \014: FORM FEED
        # \016-\037: SHIFT OUT through UNIT SEPARATOR
        self.control_chars_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

    def log_info(self, message: str) -> None:
        """Log info message."""
        print(f"{self.colors.CYAN}[INFO]{self.colors.NC} {self.script_name}: {message}")

    def log_success(self, message: str) -> None:
        """Log success message."""
        print(f"{self.colors.GREEN}[SUCCESS]{self.colors.NC} {self.script_name}: {message}")

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        print(f"{self.colors.YELLOW}[WARNING]{self.colors.NC} {self.script_name}: {message}")

    def log_error(self, message: str) -> None:
        """Log error message."""
        print(f"{self.colors.RED}[ERROR]{self.colors.NC} {self.script_name}: {message}")

    def log_file(self, message: str) -> None:
        """Log file-specific message."""
        print(f"{self.colors.PURPLE}  → {self.colors.NC}{message}")

    def log_debug(self, message: str) -> None:
        """Log debug message if debug mode is enabled."""
        if self.debug_mode:
            print(f"{self.colors.BLUE}[DEBUG]{self.colors.NC} {self.script_name}: {message}")

    def find_svg_files(self, build_dir: Path) -> List[Path]:
        """Find all SVG files in the build directory."""
        svg_files = []
        try:
            # Use pathlib's glob for cross-platform file finding
            svg_files = list(build_dir.rglob('*.svg'))
            self.log_debug(f"Found {len(svg_files)} SVG files")
        except Exception as e:
            self.log_error(f"Error finding SVG files: {e}")

        return svg_files

    def has_control_characters(self, file_path: Path) -> bool:
        """Check if file contains control characters."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return bool(self.control_chars_pattern.search(content))
        except Exception as e:
            self.log_debug(f"Error reading file {file_path}: {e}")
            return False

    def clean_svg_file(self, svg_file: Path) -> str:
        """
        Clean control characters from an SVG file.

        Returns:
            str: 'cleaned' if file was cleaned, 'no_action' if no cleaning needed, 'error' if failed
        """
        self.log_debug(f"Processing: {svg_file}")

        # Check if file contains control characters
        if not self.has_control_characters(svg_file):
            self.log_debug(f"No control characters found in: {svg_file}")
            return 'no_action'

        self.log_file(f"Cleaning: {svg_file}")
        self.log_debug(f"Control characters detected in: {svg_file}")

        # Check write permissions
        if not os.access(svg_file, os.W_OK):
            self.log_error(f"No write permission for: {svg_file}")
            return 'error'

        # Create temporary file for safe processing
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                           suffix='.svg.tmp',
                                           dir=svg_file.parent) as temp_file:
                temp_path = Path(temp_file.name)

            self.log_debug(f"Created temporary file: {temp_path}")

            # Read original file and clean control characters
            with open(svg_file, 'r', encoding='utf-8', errors='ignore') as input_file:
                content = input_file.read()

            # Remove control characters
            cleaned_content = self.control_chars_pattern.sub('', content)

            # Write cleaned content to temporary file
            with open(temp_path, 'w', encoding='utf-8') as output_file:
                output_file.write(cleaned_content)

            # Create backup of original file
            backup_path = svg_file.with_suffix(f'{svg_file.suffix}.bak.{os.getpid()}')
            self.log_debug(f"Creating backup: {backup_path}")

            shutil.copy2(svg_file, backup_path)

            # Replace original with cleaned version
            shutil.move(str(temp_path), str(svg_file))

            # Remove backup on success
            backup_path.unlink()
            self.log_debug(f"Successfully cleaned: {svg_file}")
            self.log_debug(f"Removed backup: {backup_path}")

            return 'cleaned'

        except Exception as e:
            self.log_error(f"Failed to clean {svg_file}: {e}")

            # Clean up temporary file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass

            # Restore from backup if it exists
            if 'backup_path' in locals() and backup_path.exists():
                try:
                    shutil.move(str(backup_path), str(svg_file))
                    self.log_info("Restored original file from backup")
                except Exception as restore_error:
                    self.log_error(f"Failed to restore backup for {svg_file}: {restore_error}")

            return 'error'

    def clean_directory(self, build_dir: str) -> Tuple[int, int]:
        """
        Clean all SVG files in the specified directory.

        Returns:
            Tuple[int, int]: (cleaned_count, error_count)
        """
        build_path = Path(build_dir)

        self.log_info(f"Starting SVG cleanup in directory: {build_path}")

        if self.debug_mode:
            self.log_debug("Debug mode enabled")
            quarto_output_dir = os.environ.get('QUARTO_PROJECT_OUTPUT_DIR')
            if quarto_output_dir:
                self.log_debug("Running as Quarto post-render script")
                self.log_debug(f"QUARTO_PROJECT_OUTPUT_DIR={quarto_output_dir}")
                quarto_render_all = os.environ.get('QUARTO_PROJECT_RENDER_ALL')
                if quarto_render_all:
                    self.log_debug(f"Full project render detected (QUARTO_PROJECT_RENDER_ALL={quarto_render_all})")
                else:
                    self.log_debug("Incremental or preview render")
            else:
                self.log_debug("Running as standalone script")

        # Check if build directory exists
        if not build_path.exists():
            self.log_warning(f"Build directory '{build_path}' does not exist")
            self.log_info("Script completed (nothing to clean)")
            return 0, 0

        if not build_path.is_dir():
            self.log_error(f"'{build_path}' is not a directory")
            return 0, 1

        # Find all SVG files
        svg_files = self.find_svg_files(build_path)

        if not svg_files:
            self.log_info(f"No SVG files found in {build_path}")
            self.log_info("Script completed (nothing to clean)")
            return 0, 0

        # Process each SVG file
        cleaned_count = 0
        error_count = 0

        for svg_file in svg_files:
            result = self.clean_svg_file(svg_file)
            if result == 'cleaned':
                cleaned_count += 1
            elif result == 'error':
                error_count += 1
            # 'no_action' doesn't increment any counter

        return cleaned_count, error_count


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean control characters from SVG files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script can be run in two modes:

1. Quarto Post-Render Script (automatic):
   When run by Quarto, uses QUARTO_PROJECT_OUTPUT_DIR environment variable
   Usage: %(prog)s [--debug]

2. Standalone Script:
   Usage: %(prog)s [BUILD_DIR] [--debug]

Examples:
  %(prog)s                    # Clean SVGs in _build directory (or Quarto output dir)
  %(prog)s --debug            # Enable debug mode (when run by Quarto)
  %(prog)s output             # Clean SVGs in output directory (standalone)
  %(prog)s _build --debug     # Clean with debug output enabled (standalone)
        """
    )

    parser.add_argument(
        'build_dir',
        nargs='?',
        help='Directory to clean SVG files in (default: _build or QUARTO_PROJECT_OUTPUT_DIR)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Determine build directory
    if args.build_dir:
        build_dir = args.build_dir
    else:
        # Use Quarto's output directory if available, otherwise default
        build_dir = os.environ.get('QUARTO_PROJECT_OUTPUT_DIR', '_build')

    # Create cleaner and process files
    cleaner = SVGCleaner(debug_mode=args.debug)
    cleaned_count, error_count = cleaner.clean_directory(build_dir)

    # Show summary
    if cleaned_count > 0:
        cleaner.log_success(f"Cleaned {cleaned_count} SVG file(s)")

    if error_count > 0:
        cleaner.log_warning(f"Failed to clean {error_count} SVG file(s) - see error messages above")
        cleaner.log_info("Script completed with some errors but continuing")
        return 1
    else:
        cleaner.log_info("Script completed successfully")
        return 0


if __name__ == '__main__':
    sys.exit(main())
