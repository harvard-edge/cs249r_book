#!/usr/bin/env python3
"""
Cross-platform EPUB post-processor wrapper.
Extracts EPUB, fixes cross-references, and re-packages it.
Works on Windows, macOS, and Linux.
"""

import re
import sys
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


# Matches C0 control chars that are illegal in XML 1.0 attribute values.
# XML 1.0 permits only \t (0x09), \n (0x0A), \r (0x0D) from the C0 range.
_C0_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

# Matches an HTML comment (DOTALL so it spans lines).
# Used to sanitize "--" inside TikZ-source comment blocks that violate XML.
_HTML_COMMENT = re.compile(r'<!--(.*?)-->', re.DOTALL)

# Matches aria-label="..." with a single-line (non-quote) value.
_SVG_ARIA_LABEL = re.compile(r'aria-label="([^"]*)"')

# Matches a bare <br> tag (not already self-closing).
# Quarto/Pandoc sometimes emits HTML5 <br> into XHTML, which is a fatal
# well-formedness error under strict XML parsers like Kindle / epubcheck.
_BARE_BR = re.compile(r'<br(\s*)>')


def _sanitize_comment_body(match):
    """Replace -- inside an HTML comment body with `- -` (XML-safe)."""
    body = match.group(1)
    if '--' not in body:
        return match.group(0)
    # Loop handles runs of 3+ dashes.
    while '--' in body:
        body = body.replace('--', '- -')
    return f'<!--{body}-->'


def _strip_c0_controls(match):
    """Strip C0 control chars from an aria-label value."""
    value = match.group(1)
    if _C0_CONTROL_CHARS.search(value) is None:
        return match.group(0)
    clean = _C0_CONTROL_CHARS.sub('', value)
    return f'aria-label="{clean}"'


def _sanitize_href_url(match):
    """Normalize an href URL to be strict-URI compliant for epubcheck.

    Fixes three BibTeX/citeproc-leak patterns that produce RSC-020:
      1. `\\_` → `_`    (BibTeX underscore escape in URLs)
      2. `\\%` → `%`    (BibTeX percent escape in URLs)
      3. raw `<`/`>`    (legal in DOI path segments like SICI DOIs, but
                         strict URI syntax forbids them — percent-encode)

    The match group is the raw URL inside the href attribute.
    """
    url = match.group(1)
    original = url
    if r'\_' in url:
        url = url.replace(r'\_', '_')
    if r'\%' in url:
        url = url.replace(r'\%', '%')
    # Angle brackets may appear raw or already XML-entity-escaped in the
    # attribute value. Epubcheck decodes entities before URL validation,
    # so both forms must be percent-encoded to satisfy RSC-020.
    if '<' in url:
        url = url.replace('<', '%3C')
    if '>' in url:
        url = url.replace('>', '%3E')
    if '&lt;' in url:
        url = url.replace('&lt;', '%3C')
    if '&gt;' in url:
        url = url.replace('&gt;', '%3E')
    if url == original:
        return match.group(0)
    return f'href="{url}"'


# Match any href="..." value. We invoke the sanitizer on every href and let
# it short-circuit when no fix is needed, so we catch all three RSC-020
# patterns in one pass (backslash-underscore, backslash-percent, raw <>).
_HREF_ATTR = re.compile(r'href="([^"]+)"')


def sanitize_xml_for_epubcheck(temp_dir):
    """Run post-render passes that make the EPUB strict-XML-clean.

    Fixes three FATAL(RSC-016) classes that Kindle / ClearView rejection
    is triggered by, plus the RSC-020 URL backslash-escape class that
    affects bibliography entries. All four are mechanical string fixes.

    Returns a dict with counts of fixes applied per category.
    """
    print("   Sanitizing XHTML/SVG for strict XML validity...")

    counts = {
        'comment_dashes': 0,   # -- inside HTML comments (RSC-016 FATAL)
        'bare_br': 0,          # <br> not self-closed       (RSC-016 FATAL)
        'svg_aria_c0': 0,      # C0 chars in aria-label     (RSC-016 FATAL)
        'href_rewritten': 0,   # href URLs needing sanitization (RSC-020)
    }

    def sanitize_xhtml(text):
        """Apply all XHTML-level passes. Returns (new_text, deltas_dict)."""
        out = text
        deltas = {'comment_dashes': 0, 'bare_br': 0, 'href_rewritten': 0}

        new_out, _ = _HTML_COMMENT.subn(_sanitize_comment_body, out)
        if new_out != out:
            # Approximate count: how many "- -" tokens the substitution
            # introduced. This undercounts when a run of 4+ dashes is
            # split in stages, but the number is for reporting only.
            deltas['comment_dashes'] = new_out.count('- -') - out.count('- -')
            out = new_out

        new_out, n = _BARE_BR.subn(r'<br\1/>', out)
        if n:
            deltas['bare_br'] = n
            out = new_out

        # Count href rewrites by counting matches where the sanitizer
        # actually returned a different value. Do this by walking matches.
        rewrites = 0

        def count_rewrite(m):
            nonlocal rewrites
            replacement = _sanitize_href_url(m)
            if replacement != m.group(0):
                rewrites += 1
            return replacement

        new_out = _HREF_ATTR.sub(count_rewrite, out)
        if rewrites:
            deltas['href_rewritten'] = rewrites
            out = new_out

        return out, deltas

    # --- XHTML pass ---------------------------------------------------------
    epub_text_dir = temp_dir / "EPUB" / "text"
    if epub_text_dir.exists():
        for xhtml_file in epub_text_dir.glob("*.xhtml"):
            original = xhtml_file.read_text(encoding='utf-8')
            modified, deltas = sanitize_xhtml(original)
            for k, v in deltas.items():
                counts[k] += v
            if modified != original:
                xhtml_file.write_text(modified, encoding='utf-8')

    # --- nav.xhtml is a sibling of EPUB/text, handle separately -------------
    nav_path = temp_dir / "EPUB" / "nav.xhtml"
    if nav_path.exists():
        original = nav_path.read_text(encoding='utf-8')
        modified, deltas = sanitize_xhtml(original)
        for k, v in deltas.items():
            counts[k] += v
        if modified != original:
            nav_path.write_text(modified, encoding='utf-8')

    # --- SVG pass -----------------------------------------------------------
    epub_media_dir = temp_dir / "EPUB" / "media"
    if epub_media_dir.exists():
        for svg_file in epub_media_dir.glob("*.svg"):
            original = svg_file.read_text(encoding='utf-8')
            modified, n = _SVG_ARIA_LABEL.subn(_strip_c0_controls, original)
            # Count files where stripping actually happened.
            if modified != original:
                # How many aria-label values contained C0 chars?
                c0_before = len(_C0_CONTROL_CHARS.findall(original))
                c0_after = len(_C0_CONTROL_CHARS.findall(modified))
                counts['svg_aria_c0'] += (c0_before - c0_after)
                svg_file.write_text(modified, encoding='utf-8')

    print(f"      ✅ XHTML comment `--` sanitized:  {counts['comment_dashes']}")
    print(f"      ✅ Bare <br> → <br/>:             {counts['bare_br']}")
    print(f"      ✅ SVG aria-label C0 chars:       {counts['svg_aria_c0']}")
    print(f"      ✅ href URLs normalized:          {counts['href_rewritten']}")

    return counts


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


def declare_nav_mathml_property(temp_dir):
    """Add `mathml` to the nav item's properties in the OPF manifest.

    Quarto emits `<math>` elements into `nav.xhtml` (from the TOC entries of
    sections whose titles contain math), but does not declare the `mathml`
    property on the nav item. Epubcheck flags this as OPF-014. Fix: extend
    the `properties=` attribute on the nav manifest entry from "nav" to
    "nav mathml".
    """
    opf_path = temp_dir / "EPUB" / "content.opf"
    if not opf_path.exists():
        return 0

    original = opf_path.read_text(encoding='utf-8')

    # Only patch if the nav item exists and doesn't already declare mathml.
    # The attribute form Quarto produces is `properties="nav"` on the item
    # whose href ends with nav.xhtml.
    pattern = re.compile(
        r'(<item\b[^>]*href="[^"]*nav\.xhtml"[^>]*\bproperties=")([^"]*)(")'
    )

    def patch(m):
        props = m.group(2).split()
        if 'mathml' in props:
            return m.group(0)
        props.append('mathml')
        return f'{m.group(1)}{" ".join(props)}{m.group(3)}'

    modified = pattern.sub(patch, original)

    if modified != original:
        opf_path.write_text(modified, encoding='utf-8')
        print("      ✅ OPF nav item: added `mathml` property")
        return 1
    return 0


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


def find_default_epub() -> Path | None:
    """Find the most recently generated EPUB in common build locations."""
    build_root = Path("_build")
    if not build_root.exists():
        return None

    # Support legacy path (_build/epub) and per-volume outputs (_build/epub-vol*).
    candidates = sorted(build_root.glob("**/*.epub"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    """Main entry point."""
    # Determine EPUB file path
    if len(sys.argv) > 1:
        epub_file = Path(sys.argv[1])
    else:
        # Running as post-render hook - auto-detect output location.
        detected = find_default_epub()
        if detected is None:
            print("⚠️  EPUB file not found under _build/")
            return 0
        epub_file = detected

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

        # Sanitize XHTML/SVG for strict XML validity (unblocks Kindle
        # rejection + ClearView / Tolino load failures reported in
        # issues #1014, #1052, #1148).
        sanitize_xml_for_epubcheck(temp_dir)

        # Ensure OPF declares the mathml property on the nav item
        # (issue: epubcheck OPF-014).
        declare_nav_mathml_property(temp_dir)

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
