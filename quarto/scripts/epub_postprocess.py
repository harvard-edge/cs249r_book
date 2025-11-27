#!/usr/bin/env python3
"""
Cross-platform EPUB post-processor wrapper.
Extracts EPUB, fixes cross-references, and re-packages it.
Works on Windows, macOS, and Linux.
"""

import sys
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


# Import the fix_cross_references module functions directly
# This avoids subprocess complications and works cross-platform
sys.path.insert(0, str(Path(__file__).parent))
from fix_cross_references import (
    build_epub_section_mapping,
    process_html_file
)


def extract_epub(epub_path, temp_dir):
    """Extract EPUB to temporary directory."""
    print("   Extracting EPUB...")
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)


def fix_cross_references_in_extracted_epub(temp_dir):
    """Fix cross-references in extracted EPUB directory."""
    print("   Fixing cross-references...")

    # Build EPUB section mapping
    epub_mapping = build_epub_section_mapping(temp_dir)
    print(f"      Found {len(epub_mapping)} section IDs across chapters")

    # Find all XHTML files
    epub_text_dir = temp_dir / "EPUB" / "text"
    if not epub_text_dir.exists():
        print(f"      ⚠️ No EPUB/text directory found")
        return 0

    xhtml_files = list(epub_text_dir.glob("*.xhtml"))
    print(f"      Scanning {len(xhtml_files)} XHTML files...")

    # Process each file
    files_fixed = []
    total_refs_fixed = 0
    all_unmapped = set()

    skip_patterns = ['nav.xhtml', 'cover.xhtml', 'title_page.xhtml']

    for xhtml_file in xhtml_files:
        # Skip certain files
        if any(skip in xhtml_file.name for skip in skip_patterns):
            continue

        rel_path, fixed_count, unmapped = process_html_file(
            xhtml_file,
            temp_dir,  # base_dir for relative paths
            epub_mapping
        )

        if fixed_count > 0:
            files_fixed.append((rel_path or xhtml_file.name, fixed_count))
            total_refs_fixed += fixed_count
        all_unmapped.update(unmapped)

    if files_fixed:
        print(f"      ✅ Fixed {total_refs_fixed} cross-references in {len(files_fixed)} files")
        for path, count in files_fixed:
            print(f"         📄 {path}: {count} refs")
    else:
        print(f"      ✅ No unresolved cross-references found")

    if all_unmapped:
        print(f"      ⚠️ Unmapped references: {', '.join(sorted(list(all_unmapped)[:5]))}")

    return total_refs_fixed


def repackage_epub(temp_dir, output_path):
    """Re-package EPUB from temporary directory."""
    print("   Re-packaging EPUB...")

    # Create new EPUB zip file
    with zipfile.ZipFile(output_path, 'w') as epub_zip:
        # EPUB requires mimetype to be first and uncompressed
        mimetype_path = temp_dir / "mimetype"
        if mimetype_path.exists():
            epub_zip.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)

        # Add all other files recursively
        for item in ["META-INF", "EPUB"]:
            item_path = temp_dir / item
            if item_path.exists():
                if item_path.is_dir():
                    for file_path in item_path.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(temp_dir)
                            epub_zip.write(file_path, arcname, compress_type=zipfile.ZIP_DEFLATED)
                else:
                    epub_zip.write(item_path, item, compress_type=zipfile.ZIP_DEFLATED)


def main():
    """Main entry point."""
    # Determine EPUB file path
    if len(sys.argv) > 1:
        epub_file = Path(sys.argv[1])
    else:
        # Running as post-render hook - find the EPUB
        epub_file = Path("_build/epub/Machine-Learning-Systems.epub")

    if not epub_file.exists():
        print(f"⚠️  EPUB file not found: {epub_file}")
        return 0

    print(f"📚 Post-processing EPUB: {epub_file}")

    # Get absolute path to EPUB file
    epub_abs = epub_file.resolve()

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Extract EPUB
        extract_epub(epub_abs, temp_dir)

        # Fix cross-references
        fixes = fix_cross_references_in_extracted_epub(temp_dir)

        # Create a temporary output file
        fixed_epub = temp_dir / "fixed.epub"

        # Re-package EPUB
        repackage_epub(temp_dir, fixed_epub)

        # Replace original with fixed version
        shutil.move(str(fixed_epub), str(epub_abs))

        print("✅ EPUB post-processing complete")
        return 0

    except Exception as e:
        print(f"❌ Error during EPUB post-processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
