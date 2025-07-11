#!/usr/bin/env python3
"""
check_images.py

Validates image files by inspecting their actual content using Pillow.
Supports .png, .jpg, .jpeg, .gif formats.

Usage:
  - Single file:    python check_images.py -f image.png
  - Directory scan: python check_images.py -d ./assets
  - CI hooks:       python check_images.py image1.png image2.jpg

Returns:
  - Exit code 1 if invalid image files are found.
  - Exit code 2 if files are unreadable.
  - Exit code 0 if all images are valid.
"""

import os
import sys
import argparse
from PIL import Image
from rich.console import Console
from rich.table import Table

VALID_EXTENSIONS = {
    '.png': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.gif': 'GIF',
}

console = Console()

def is_valid_image(filepath, expected_format):
    try:
        with Image.open(filepath) as img:
            actual_format = img.format.upper()
            return actual_format == expected_format, actual_format
    except Exception as e:
        return f"Unreadable: {e}", None

def check_file(filepath, strict=False, verbose=False):
    ext = os.path.splitext(filepath)[1].lower()
    expected_format = VALID_EXTENSIONS.get(ext)

    if not expected_format:
        msg = f"Unsupported extension (.{ext})"
        if strict:
            return [(filepath, msg, None, None)]
        if verbose:
            console.print(f"⚠️  [yellow]Skipping unsupported file:[/yellow] {filepath}")
        return []

    if verbose:
        console.print(f"🔍 Checking [cyan]{filepath}[/cyan] (expected: {expected_format})")

    result, actual_format = is_valid_image(filepath, expected_format)
    if result is True:
        if verbose:
            console.print(f"✅ [green]{filepath}[/green]: valid ({actual_format})")
        return []
    elif isinstance(result, str):
        return [(filepath, result, None, expected_format)]
    else:
        return [(filepath, "Format mismatch", actual_format, expected_format)]

def check_directory(root_dir, strict=False, verbose=False):
    invalid_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            invalid_files.extend(check_file(fpath, strict=strict, verbose=verbose))
    return invalid_files

def print_invalid_files(invalid):
    table = Table(title="❌ Invalid Image Files", show_lines=True)

    table.add_column("File", style="cyan", overflow="fold")
    table.add_column("Reason", style="red")
    table.add_column("Actual Format", style="yellow")
    table.add_column("Expected Format", style="green")

    for fpath, reason, actual, expected in invalid:
        table.add_row(fpath, reason, str(actual or "—"), str(expected or "—"))

    console.print(table)

def main():
    parser = argparse.ArgumentParser(
        description="Validate image files by checking actual format using Pillow."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', '--file', type=str, help="Path to a single image file")
    group.add_argument('-d', '--dir', type=str, help="Directory to scan recursively")
    parser.add_argument('files', nargs='*', help="Files passed directly (e.g., via pre-commit)")
    parser.add_argument('--strict', action='store_true', help="Fail on unsupported file extensions")
    parser.add_argument('--verbose', '-v', action='store_true', help="Print each file being checked")

    args = parser.parse_args()

    invalid = []
    if args.file:
        invalid = check_file(args.file, strict=args.strict, verbose=args.verbose)
    elif args.dir:
        invalid = check_directory(args.dir, strict=args.strict, verbose=args.verbose)
    elif args.files:
        for fpath in args.files:
            invalid.extend(check_file(fpath, strict=args.strict, verbose=args.verbose))
    else:
        parser.print_help()
        sys.exit(0)

    if invalid:
        print_invalid_files(invalid)
        unreadable = any("Unreadable" in reason for _, reason, _, _ in invalid)
        sys.exit(2 if unreadable else 1)
    else:
        console.print("[bold green]✅ All image files are valid.[/bold green]")
        sys.exit(0)

if __name__ == "__main__":
    main()
