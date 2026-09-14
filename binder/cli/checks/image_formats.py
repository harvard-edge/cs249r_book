"""
Image format validator for MLSysBook.

Validates image files by inspecting their actual binary/XML content using PIL/ElementTree.
Supports .png, .jpg, .jpeg, .gif, .svg, .webp formats.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

VALID_EXTENSIONS = {
    '.png': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.gif': 'GIF',
    '.svg': 'SVG',
    '.webp': 'WEBP',
}


def is_valid_svg(filepath: Path | str) -> Tuple[bool | str, Optional[str]]:
    """Validate SVG file by checking if it's valid XML with SVG root."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        if 'svg' in root.tag.lower() or root.tag.endswith('}svg'):
            return True, 'SVG'
        else:
            return False, f"Not valid SVG (root: {root.tag})"
    except ET.ParseError as e:
        return f"Invalid XML: {e}", None
    except Exception as e:
        return f"Unreadable: {e}", None


def is_valid_image(filepath: Path | str, expected_format: str) -> Tuple[bool | str, Optional[str]]:
    """Validate image files using PIL for raster formats, custom logic for SVG."""
    if expected_format == 'SVG':
        return is_valid_svg(filepath)

    try:
        with Image.open(filepath) as img:
            actual_format = img.format.upper() if img.format else ""
            return actual_format == expected_format, actual_format
    except Exception as e:
        return f"Unreadable: {e}", None


def check_file(
    filepath: Path | str,
    strict: bool = False,
    verbose: bool = False,
    fix: bool = False,
    show_progress: bool = False,
) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
    """Check a single image file for format validity."""
    fpath_str = str(filepath)
    ext = os.path.splitext(fpath_str)[1].lower()
    expected_format = VALID_EXTENSIONS.get(ext)

    if not expected_format:
        msg = f"Unsupported extension (.{ext})"
        if strict:
            return [(fpath_str, msg, None, None)]
        return []

    result, actual_format = is_valid_image(filepath, expected_format)
    if result is True:
        return []
    elif isinstance(result, str):
        return [(fpath_str, result, None, expected_format)]
    else:
        return [(fpath_str, "Format mismatch", actual_format, expected_format)]
