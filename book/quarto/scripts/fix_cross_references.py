#!/usr/bin/env python3
"""
Post-process ALL HTML files to fix unresolved cross-reference links.

WHY THIS SCRIPT EXISTS:
-----------------------
When using selective rendering (only building specific chapters like index + introduction),
Quarto cannot resolve cross-references to chapters that aren't being built. These show up
in the HTML output as unresolved references like: ?@sec-ml-systems

This is a problem because:
1. The glossary has 800+ cross-references to all chapters
2. The introduction references many other chapters
3. We want fast builds during development (only building 2-3 files instead of 20+)
4. But we still want all cross-reference links to work properly

WHAT THIS SCRIPT DOES:
----------------------
1. Scans QMD source files to dynamically build a mapping of section IDs → HTML paths and titles
2. Scans ALL HTML files in the build directory after Quarto finishes
3. Finds unresolved references that appear as: <strong>?@sec-xxx</strong>
4. Converts them to proper HTML links: <strong><a href="../path/to/chapter.html#sec-xxx">Title</a></strong>

The dynamic approach means you never need to update this script when adding chapters or
renaming sections — it reads the source QMDs directly.

WHEN IT RUNS:
-------------
This script runs as a post-render hook in the Quarto configuration:
  post-render:
    - scripts/clean_svgs.py
    - scripts/fix_cross_references.py  # <-- Runs after all HTML is generated

HOW TO USE:
-----------
1. Automatic: Runs automatically during `quarto render` as a post-render hook
2. Manual: python3 scripts/fix_cross_references.py [specific-file.html]
3. Test all: python3 scripts/fix_cross_references.py  # processes all HTML files
"""

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Dynamic mapping: built by scanning QMD sources at runtime
# ---------------------------------------------------------------------------

def _extract_heading_text(line: str) -> str:
    """Strip Markdown heading markers, {#...} anchors, and attributes from a line."""
    # Remove leading # chars and space
    text = re.sub(r'^#+\s*', '', line)
    # Remove {#id ...} block (Quarto heading attribute)
    text = re.sub(r'\{[^}]*\}', '', text)
    return text.strip()


def build_qmd_mapping(qmd_roots: list[Path]) -> tuple[dict, dict]:
    """
    Scan QMD source trees to build section ID → (html_path, title) mappings.

    For each QMD file we:
      - Find heading-level IDs:  ## Heading Text {#sec-foo}
      - Find div-level IDs:      :::: {#pri-foo .callout-principle title="The Title"}
    The HTML output path is derived by replacing .qmd → .html, keeping the same
    relative path from the project root (which mirrors the build output layout).

    Each qmd_root is a (scan_dir, path_prefix) pair — scan_dir is where we recurse,
    path_prefix is prepended to every output path so it matches the Quarto build tree.

    Args:
        qmd_roots: list of (scan_dir, path_prefix) tuples, or plain Path objects
                   (plain Path → scan_dir == path_prefix parent == scan_dir)

    Returns:
        (chapter_mapping, chapter_titles) — same shape as the old hardcoded dicts
    """
    chapter_mapping: dict[str, str] = {}
    chapter_titles: dict[str, str] = {}

    # Regex patterns to find IDs in QMD source
    # Heading: # Title {#sec-foo} or # Title {#sec-foo .class}
    heading_id_re = re.compile(r'^(#+)\s+(.+?)\s*\{#([\w-]+)[^}]*\}')
    # Div / callout — match any attribute block containing #id, regardless of position
    # Handles both:
    #   :::: {#pri-foo .callout-principle title="..."}
    #   :::: {.callout-principle #pri-foo title="..."}
    div_id_re = re.compile(r'^\s*:{2,}\s*\{([^}]*)\}')
    id_attr_re = re.compile(r'#([\w-]+)')
    title_attr_re = re.compile(r'title="([^"]*)"')

    # Normalise qmd_roots: accept plain Path (scan from project root)
    # We need paths relative to the *project* root, not the contents/ subdir
    normalised: list[tuple[Path, Path]] = []
    for entry in qmd_roots:
        if isinstance(entry, tuple):
            normalised.append(entry)
        else:
            # entry is a Path like Path("contents"); its parent is the project root
            normalised.append((entry, entry.parent))

    for scan_dir, path_root in normalised:
        if not scan_dir.exists():
            continue

        for qmd_file in sorted(scan_dir.rglob("*.qmd")):
            # Derive the relative HTML path from the project root
            rel_qmd = qmd_file.relative_to(path_root)
            rel_html = rel_qmd.with_suffix(".html")
            html_path_str = str(rel_html).replace("\\", "/")

            try:
                lines = qmd_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            for line in lines:
                # Try heading pattern first
                m = heading_id_re.match(line)
                if m:
                    hashes, raw_title, sec_id = m.group(1), m.group(2), m.group(3)
                    title = _extract_heading_text(raw_title)
                    depth = len(hashes)
                    # Only add if not already registered (first occurrence wins)
                    if sec_id not in chapter_mapping:
                        chapter_mapping[sec_id] = f"{html_path_str}#{sec_id}"
                        chapter_titles[sec_id] = title
                    continue

                # Try div/callout pattern (handles any attribute order inside { })
                m = div_id_re.match(line)
                if m:
                    attrs = m.group(1)
                    im = id_attr_re.search(attrs)
                    if not im:
                        continue
                    sec_id = im.group(1)
                    tm = title_attr_re.search(attrs)
                    title = tm.group(1) if tm else sec_id  # fallback to ID
                    if sec_id not in chapter_mapping:
                        chapter_mapping[sec_id] = f"{html_path_str}#{sec_id}"
                        chapter_titles[sec_id] = title

    return chapter_mapping, chapter_titles


def _find_qmd_roots() -> list[tuple[Path, Path]]:
    """
    Locate the Quarto project source tree from common run locations.

    Quarto post-render hooks run from the project root (where _quarto.yml lives).
    Manual runs may happen from the scripts/ directory or the repo root.

    Returns list of (scan_dir, path_root) tuples for build_qmd_mapping.
    """
    candidates = [
        Path("."),           # run from project root (most common)
        Path(".."),          # run from scripts/
        Path("book/quarto"), # run from repo root
    ]
    for c in candidates:
        if (c / "contents").exists():
            return [(c / "contents", c)]
    return []


# Build the mapping once at import time (lazy cache via module-level variable)
_CHAPTER_MAPPING: dict | None = None
_CHAPTER_TITLES: dict | None = None


def get_mappings() -> tuple[dict, dict]:
    global _CHAPTER_MAPPING, _CHAPTER_TITLES
    if _CHAPTER_MAPPING is None:
        roots = _find_qmd_roots()
        if roots:
            _CHAPTER_MAPPING, _CHAPTER_TITLES = build_qmd_mapping(roots)
        else:
            _CHAPTER_MAPPING, _CHAPTER_TITLES = {}, {}
    return _CHAPTER_MAPPING, _CHAPTER_TITLES


# ---------------------------------------------------------------------------
# EPUB support
# ---------------------------------------------------------------------------

def build_epub_section_mapping(epub_dir: Path) -> dict:
    """
    Build mapping from section IDs to EPUB chapter files by scanning actual chapters.

    Args:
        epub_dir: Path to EPUB build directory (_build/epub or extracted EPUB root)

    Returns:
        Dictionary mapping section IDs to chapter filenames (e.g., {"sec-xxx": "ch004.xhtml"})
    """
    mapping: dict[str, str] = {}

    possible_text_dirs = [
        epub_dir / "text",
        epub_dir / "EPUB" / "text",
    ]

    text_dir = None
    for dir_path in possible_text_dirs:
        if dir_path.exists():
            text_dir = dir_path
            break

    if not text_dir:
        return mapping

    for xhtml_file in sorted(text_dir.glob("ch*.xhtml")):
        try:
            content = xhtml_file.read_text(encoding="utf-8")
            for sec_id in re.findall(r'id="(sec-[^"]+)"', content):
                mapping[sec_id] = xhtml_file.name
        except Exception:
            continue

    return mapping


# ---------------------------------------------------------------------------
# Path calculation
# ---------------------------------------------------------------------------

def calculate_relative_path(
    from_file: Path,
    to_path: str,
    build_dir: Path,
    epub_mapping: dict | None = None,
) -> str:
    """
    Calculate relative path from one file to another.

    Args:
        from_file: Path object of the source file
        to_path: String path from build root (e.g., "contents/vol1/chapter/file.html#anchor")
        build_dir: Path object of the build directory root
        epub_mapping: Optional dict mapping section IDs to EPUB chapter files

    Returns:
        Relative path string from from_file to to_path
    """
    if epub_mapping is not None:
        if "#" in to_path:
            _, sec_id = to_path.split("#", 1)
            target_chapter = epub_mapping.get(sec_id)
            if target_chapter:
                return f"{target_chapter}#{sec_id}"
        return to_path

    if "#" in to_path:
        target_path_str, anchor = to_path.split("#", 1)
        anchor = f"#{anchor}"
    else:
        target_path_str = to_path
        anchor = ""

    target_abs = build_dir / target_path_str
    source_abs = from_file

    try:
        rel_path = Path(target_abs).relative_to(source_abs.parent)
        result = str(rel_path).replace("\\", "/")
    except ValueError:
        source_parts = source_abs.parent.parts
        target_parts = target_abs.parts

        common_length = 0
        for s, t in zip(source_parts, target_parts):
            if s == t:
                common_length += 1
            else:
                break

        up_levels = len(source_parts) - common_length
        down_parts = target_parts[common_length:]
        rel_parts = [".."] * up_levels + list(down_parts)
        result = "/".join(rel_parts)

    return result + anchor


# ---------------------------------------------------------------------------
# Cross-reference fixing
# ---------------------------------------------------------------------------

def fix_cross_reference_link(match, from_file, build_dir, epub_mapping=None):
    """Replace a single cross-reference link with proper HTML link."""
    full_match = match.group(0)
    sec_ref = match.group(1)

    chapter_mapping, chapter_titles = get_mappings()
    abs_path = chapter_mapping.get(sec_ref)
    title = chapter_titles.get(sec_ref)

    if abs_path and title:
        rel_path = calculate_relative_path(from_file, abs_path, build_dir, epub_mapping)
        return f'<a href="{rel_path}">{title}</a>'
    else:
        print(f"   ⚠️  No mapping found for: {sec_ref}")
        return full_match


def fix_cross_references(
    html_content: str,
    from_file: Path,
    build_dir: Path,
    epub_mapping: dict | None = None,
) -> tuple[str, int, list]:
    """
    Fix all cross-reference links in HTML/XHTML content.

    Quarto generates three types of unresolved references when chapters aren't built:
    1. Full unresolved links: <a href="#sec-xxx" class="quarto-xref"><span class="quarto-unresolved-ref">...</span></a>
    2. Simple unresolved refs: <strong>?@sec-xxx</strong> (common in selective builds)
    3. EPUB unresolved refs: <a href="@sec-xxx">Link Text</a>
    """
    chapter_mapping, chapter_titles = get_mappings()

    # Pattern 1: Quarto full unresolved cross-reference links
    pattern1 = r'<a href="#(sec-[a-zA-Z0-9-]+)" class="quarto-xref"><span class="quarto-unresolved-ref">[^<]*</span></a>'

    # Pattern 2: Simple unresolved references (?@sec-xxx or ?@pri-xxx)
    pattern2 = r'<strong>\?\@(sec-[a-zA-Z0-9-]+|pri-[a-zA-Z0-9-]+)</strong>'

    # Pattern 3: EPUB-specific unresolved references
    pattern3 = r'<a href="@(sec-[a-zA-Z0-9-]+)"([^>]*)>([^<]*)</a>'

    matches1 = re.findall(pattern1, html_content)
    matches2 = re.findall(pattern2, html_content)
    matches3 = re.findall(pattern3, html_content)
    total_matches = len(matches1) + len(matches2) + len(matches3)

    # Fix Pattern 1
    fixed_content = re.sub(
        pattern1,
        lambda m: fix_cross_reference_link(m, from_file, build_dir, epub_mapping),
        html_content,
    )

    # Fix Pattern 2
    unmapped_refs = []

    def fix_simple_reference(match):
        sec_ref = match.group(1)
        abs_path = chapter_mapping.get(sec_ref)
        title = chapter_titles.get(sec_ref)
        if abs_path and title:
            rel_path = calculate_relative_path(from_file, abs_path, build_dir, epub_mapping)
            return f'<strong><a href="{rel_path}">{title}</a></strong>'
        else:
            unmapped_refs.append(sec_ref)
            return match.group(0)

    fixed_content = re.sub(pattern2, fix_simple_reference, fixed_content)

    # Fix Pattern 3 (EPUB)
    def fix_epub_reference(match):
        sec_ref = match.group(1)
        attrs = match.group(2)
        link_text = match.group(3)

        if epub_mapping:
            target_chapter = epub_mapping.get(sec_ref)
            if target_chapter:
                return f'<a href="{target_chapter}#{sec_ref}"{attrs}>{link_text}</a>'
            else:
                unmapped_refs.append(sec_ref)
                return match.group(0)
        else:
            abs_path = chapter_mapping.get(sec_ref)
            if abs_path:
                rel_path = calculate_relative_path(from_file, abs_path, build_dir, None)
                return f'<a href="{rel_path}"{attrs}>{link_text}</a>'
            else:
                unmapped_refs.append(sec_ref)
                return match.group(0)

    fixed_content = re.sub(pattern3, fix_epub_reference, fixed_content)

    remaining1 = re.findall(pattern1, fixed_content)
    remaining2 = re.findall(pattern2, fixed_content)
    remaining3 = re.findall(pattern3, fixed_content)
    fixed_count = total_matches - len(remaining1) - len(remaining2) - len(remaining3)

    return fixed_content, fixed_count, unmapped_refs


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_html_file(html_file: Path, base_dir: Path, epub_mapping: dict | None = None):
    """Process a single HTML/XHTML file to fix cross-references."""
    try:
        html_content = html_file.read_text(encoding="utf-8")
    except Exception:
        return None, 0, []

    fixed_content, fixed_count, unmapped = fix_cross_references(
        html_content, html_file, base_dir, epub_mapping
    )

    if fixed_count > 0:
        try:
            html_file.write_text(fixed_content, encoding="utf-8")
            return html_file.relative_to(base_dir), fixed_count, unmapped
        except Exception:
            return None, 0, []

    return None, 0, []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point. Runs in three modes:
    1. Post-render hook (no args): Processes HTML or EPUB builds from _build/
    2. Directory mode (dir arg): Processes extracted EPUB directory
    3. Manual mode (file arg): Processes a specific file
    """
    # Pre-load mappings and report count
    chapter_mapping, chapter_titles = get_mappings()
    if chapter_mapping:
        print(f"📖 Loaded {len(chapter_mapping)} section IDs from QMD sources")
    else:
        print("⚠️  No QMD sources found — cross-reference mapping will be empty")

    skip_patterns = [
        "search.html", "404.html", "site_libs",
        "nav.xhtml", "cover.xhtml", "title_page.xhtml",
    ]

    if len(sys.argv) == 1:
        # MODE 1: Running as Quarto post-render hook
        build_root = Path("_build")
        html_candidates = sorted(
            [p for p in build_root.glob("html*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        epub_candidates = sorted(
            [p for p in build_root.glob("epub*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        epub_mapping = None
        if html_candidates:
            build_dir = html_candidates[0]
            file_pattern = "*.html"
            file_type = "HTML"
        elif epub_candidates and list(epub_candidates[0].glob("*.xhtml")):
            build_dir = epub_candidates[0]
            file_pattern = "*.xhtml"
            file_type = "XHTML (EPUB)"
            print("📚 Building EPUB section mapping...")
            epub_mapping = build_epub_section_mapping(build_dir)
            print(f"   Found {len(epub_mapping)} section IDs across chapters")
        elif Path("EPUB").exists() and list(Path("EPUB").rglob("*.xhtml")):
            build_dir = Path(".")
            file_pattern = "*.xhtml"
            file_type = "XHTML (EPUB - extracted)"
            print("📚 Building EPUB section mapping...")
            epub_mapping = build_epub_section_mapping(Path("."))
            print(f"   Found {len(epub_mapping)} section IDs across chapters")
        else:
            print("⚠️  No HTML or EPUB build directory found — skipping")
            sys.exit(0)

        files = list(build_dir.rglob(file_pattern))
        print(f"🔗 [Cross-Reference Fix] Scanning {len(files)} {file_type} files...")

        files_fixed = []
        total_refs_fixed = 0
        all_unmapped: set[str] = set()

        for file in files:
            if any(skip in str(file) for skip in skip_patterns):
                continue

            rel_path, fixed_count, unmapped = process_html_file(file, build_dir, epub_mapping)
            if fixed_count > 0:
                files_fixed.append((rel_path, fixed_count))
                total_refs_fixed += fixed_count
            all_unmapped.update(unmapped)

        if files_fixed:
            print(f"✅ Fixed {total_refs_fixed} cross-references in {len(files_fixed)} files:")
            for path, count in files_fixed:
                print(f"   📄 {path}: {count} refs")
        else:
            print("✅ No unresolved cross-references found")

        if all_unmapped:
            print(f"⚠️  Unmapped references: {', '.join(sorted(all_unmapped))}")

    elif len(sys.argv) == 2:
        # MODE 2: Running with explicit file argument
        html_file = Path(sys.argv[1])
        if not html_file.exists():
            print(f"❌ File not found: {html_file}")
            sys.exit(1)

        epub_mapping = None
        if "text" in html_file.parts and html_file.suffix == ".xhtml":
            epub_base = html_file.parent.parent
            print("📚 Building EPUB section mapping...")
            epub_mapping = build_epub_section_mapping(epub_base)
            print(f"   Found {len(epub_mapping)} section IDs across chapters")

        print(f"🔗 Fixing cross-reference links in: {html_file}")
        rel_path, fixed_count, unmapped = process_html_file(
            html_file, html_file.parent, epub_mapping
        )
        if fixed_count > 0:
            print(f"✅ Fixed {fixed_count} cross-references")
            if unmapped:
                print(f"⚠️  Unmapped references: {', '.join(sorted(unmapped))}")
        else:
            print("✅ No cross-reference fixes needed")

    else:
        print("Usage: python3 fix_cross_references.py [<html-or-xhtml-file>]")
        sys.exit(1)


if __name__ == "__main__":
    main()
