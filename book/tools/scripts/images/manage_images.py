#!/usr/bin/env python3
"""
check_images.py

Validates image files by inspecting their actual content.
Supports .png, .jpg, .jpeg, .gif, .svg, .webp formats.

Usage:
  - Single file:    python check_images.py -f image.png
  - Directory scan: python check_images.py -d ./assets
  - CI hooks:       python check_images.py image1.png image2.jpg
  - Auto-fix:       python check_images.py -d ./assets --fix
  - Show progress:  python check_images.py -d ./assets --verbose

By default, only shows summary. Use --verbose (-v) to see progress
for each file with ✅/❌ indicators. Use --debug for detailed info.

Validation methods:
  - Raster formats (PNG, JPEG, GIF, WebP): Uses Pillow to verify format
  - Vector formats (SVG): Validates XML structure and SVG namespace

Returns:
  - Exit code 1 if invalid image files are found.
  - Exit code 2 if files are unreadable.
  - Exit code 0 if all images are valid or fixed.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from PIL import Image, UnidentifiedImageError
from rich.console import Console
from rich.table import Table

VALID_EXTENSIONS = {
    '.png': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.gif': 'GIF',
    '.svg': 'SVG',
    '.webp': 'WEBP',  # Modern web format
}

console = Console()

def is_valid_svg(filepath):
    """Validate SVG file by checking if it's valid XML with SVG root."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        # Check if it has SVG namespace or is an SVG element
        if 'svg' in root.tag.lower() or root.tag.endswith('}svg'):
            return True, 'SVG'
        else:
            return False, f"Not valid SVG (root: {root.tag})"
    except ET.ParseError as e:
        return f"Invalid XML: {e}", None
    except Exception as e:
        return f"Unreadable: {e}", None

def is_valid_image(filepath, expected_format):
    """Validate image files using PIL for raster formats, custom logic for SVG."""
    if expected_format == 'SVG':
        return is_valid_svg(filepath)

    try:
        with Image.open(filepath) as img:
            actual_format = img.format.upper()
            return actual_format == expected_format, actual_format
    except Exception as e:
        return f"Unreadable: {e}", None

def fix_image(filepath, expected_format):
    """Fix image format mismatches. SVG files cannot be auto-fixed."""
    if expected_format == 'SVG':
        console.print(f"⚠️  [yellow]Cannot fix SVG files:[/yellow] {filepath}")
        return False

    try:
        with Image.open(filepath) as img:
            img = img.convert('RGBA') if expected_format == 'PNG' else img.convert('RGB')
            img.save(filepath, format=expected_format)
            console.print(f"🔧 [blue]Fixed:[/blue] {filepath} → {expected_format}")
            return True
    except Exception as e:
        console.print(f"❌ [red]Failed to fix:[/red] {filepath} ({e})")
        return False

def check_file(filepath, strict=False, verbose=False, fix=False, show_progress=True):
    ext = os.path.splitext(filepath)[1].lower()
    expected_format = VALID_EXTENSIONS.get(ext)

    if not expected_format:
        msg = f"Unsupported extension (.{ext})"
        if strict:
            if show_progress:
                console.print(f"⚠️  [yellow]{filepath}[/yellow] - Unsupported extension")
            return [(filepath, msg, None, None)]
        if verbose and show_progress:
            console.print(f"⚠️  [dim]Skip:[/dim] {filepath} (unsupported extension)")
        return []

    result, actual_format = is_valid_image(filepath, expected_format)
    if result is True:
        if show_progress:
            console.print(f"✅ [green]{filepath}[/green] ({actual_format})")
        elif verbose:
            console.print(f"✅ [green]{filepath}[/green]: valid ({actual_format})")
        return []
    elif isinstance(result, str):
        if show_progress:
            console.print(f"❌ [red]{filepath}[/red] - {result}")
        return [(filepath, result, None, expected_format)]
    else:
        if fix:
            fixed = fix_image(filepath, expected_format)
            if not fixed and show_progress:
                console.print(f"❌ [red]{filepath}[/red] - Fix failed")
            return [] if fixed else [(filepath, "Fix failed", actual_format, expected_format)]
        else:
            if show_progress:
                console.print(f"❌ [red]{filepath}[/red] - Format mismatch ({actual_format} != {expected_format})")
            return [(filepath, "Format mismatch", actual_format, expected_format)]

def check_directory(root_dir, strict=False, verbose=False, fix=False, show_progress=True):
    invalid_files = []
    total_files = 0
    image_files = 0
    format_stats = {}  # Track stats by format

    if show_progress:
        console.print(f"\n🔍 [bold cyan]Scanning directory:[/bold cyan] {root_dir}")
        console.print()

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            total_files += 1
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()

            # Only process image files
            if ext in VALID_EXTENSIONS:
                image_files += 1
                expected_format = VALID_EXTENSIONS[ext]

                # Initialize format stats if not exists
                if expected_format not in format_stats:
                    format_stats[expected_format] = {'total': 0, 'valid': 0, 'invalid': 0}

                format_stats[expected_format]['total'] += 1

                file_invalid = check_file(fpath, strict=strict, verbose=verbose, fix=fix, show_progress=show_progress)
                if file_invalid:
                    format_stats[expected_format]['invalid'] += 1
                    invalid_files.extend(file_invalid)
                else:
                    format_stats[expected_format]['valid'] += 1

    return invalid_files, total_files, image_files, format_stats

def print_invalid_files(invalid):
    table = Table(title="❌ Invalid Image Files", show_lines=True)
    table.add_column("File", style="cyan", overflow="fold")
    table.add_column("Reason", style="red")
    table.add_column("Actual Format", style="yellow")
    table.add_column("Expected Format", style="red")
    for fpath, reason, actual, expected in invalid:
        table.add_row(fpath, reason, str(actual or "—"), str(expected or "—"))
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Validate image files by checking actual format using Pillow.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f', '--file', type=str, help="Path to a single image file")
    group.add_argument('-d', '--dir', type=str, help="Directory to scan recursively")
    parser.add_argument('files', nargs='*', help="Files passed directly (e.g., via pre-commit)")
    parser.add_argument('--strict', action='store_true', help="Fail on unsupported file extensions")
    parser.add_argument('--verbose', '-v', action='store_true', help="Show progress for each file checked")
    parser.add_argument('--debug', action='store_true', help="Show detailed debug information")
    parser.add_argument('--fix', action='store_true', help="Attempt to fix format mismatches in place")

    args = parser.parse_args()
    invalid = []
    total_files = 0
    image_files = 0
    format_stats = {}
    show_progress = args.verbose
    debug_mode = args.debug

    if args.file:
        image_files = 1
        total_files = 1
        if show_progress:
            console.print(f"\n🔍 [bold cyan]Checking single file:[/bold cyan] {args.file}")
            console.print()
        invalid = check_file(args.file, strict=args.strict, verbose=debug_mode, fix=args.fix, show_progress=show_progress)

        # Track format stats for single file
        ext = os.path.splitext(args.file)[1].lower()
        if ext in VALID_EXTENSIONS:
            expected_format = VALID_EXTENSIONS[ext]
            format_stats[expected_format] = {'total': 1, 'valid': 0 if len(invalid) > 0 else 1, 'invalid': len(invalid)}

    elif args.dir:
        invalid, total_files, image_files, format_stats = check_directory(args.dir, strict=args.strict, verbose=debug_mode, fix=args.fix, show_progress=show_progress)
        # Debug print to see what invalid actually is
        print(f"DEBUG: invalid type: {type(invalid)}, value: {invalid}")  # temporary debug

    elif args.files:
        if show_progress:
            console.print(f"\n🔍 [bold cyan]Checking {len(args.files)} files...[/bold cyan]")
            console.print()
        image_files = len(args.files)
        total_files = len(args.files)

        for fpath in args.files:
            ext = os.path.splitext(fpath)[1].lower()
            if ext in VALID_EXTENSIONS:
                expected_format = VALID_EXTENSIONS[ext]

                # Initialize format stats if not exists
                if expected_format not in format_stats:
                    format_stats[expected_format] = {'total': 0, 'valid': 0, 'invalid': 0}

                format_stats[expected_format]['total'] += 1

                file_invalid = check_file(fpath, strict=args.strict, verbose=debug_mode, fix=args.fix, show_progress=show_progress)
                if file_invalid:
                    format_stats[expected_format]['invalid'] += 1
                    invalid.extend(file_invalid)
                else:
                    format_stats[expected_format]['valid'] += 1
    else:
        parser.print_help()
        sys.exit(0)

    # Print summary
    console.print()
    console.print("[bold]📊 Summary:[/bold]")
    if args.dir:
        console.print(f"   Total files scanned: [cyan]{total_files}[/cyan]")
    console.print(f"   Image files found: [cyan]{image_files}[/cyan]")
    console.print(f"   Valid images: [green]{image_files - len(invalid)}[/green]")
    console.print(f"   Invalid images: [red]{len(invalid)}[/red]")

    # Show format breakdown
    if format_stats:
        console.print()
        console.print("[bold]📋 Format Breakdown:[/bold]")

        # Sort by total count (descending)
        sorted_formats = sorted(format_stats.items(), key=lambda x: x[1]['total'], reverse=True)

        for format_name, stats in sorted_formats:
            total = stats['total']
            valid = stats['valid']
            invalid = stats['invalid']

            status_color = "green" if invalid == 0 else "yellow" if invalid < total else "red"
            console.print(f"   {format_name}: [cyan]{total}[/cyan] total ([{status_color}]{valid} valid, {invalid} invalid[/{status_color}])")

    if invalid:
        console.print()
        print_invalid_files(invalid)
        unreadable = any("Unreadable" in reason for _, reason, _, _ in invalid)
        sys.exit(2 if unreadable else 1)
    else:
        console.print("\n[bold green]✅ All image files are valid[/bold green]")
        sys.exit(0)

if __name__ == "__main__":
    main()
