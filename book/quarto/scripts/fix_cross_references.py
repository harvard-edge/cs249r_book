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
1. Scans ALL HTML files in the build directory after Quarto finishes
2. Finds unresolved references that appear as: <strong>?@sec-xxx</strong>
3. Converts them to proper HTML links: <strong><a href="../path/to/chapter.html#sec-xxx">Chapter N: Title</a></strong>
4. Uses a hardcoded mapping of section IDs to HTML paths and titles
5. Adds "Chapter N:" prefix to HTML links for better readability (PDF rendering adds this automatically)

WHEN IT RUNS:
-------------
This script runs as a post-render hook in the Quarto configuration:
  post-render:
    - scripts/clean_svgs.py
    - scripts/fix_cross_references.py  # <-- Runs after all HTML is generated

HOW TO USE:
-----------
1. Automatic: Runs automatically during `quarto render` as a post-render hook
2. Manual: python3 scripts/fix-glossary-html.py [specific-file.html]
3. Test all: python3 scripts/fix-glossary-html.py  # processes all HTML files

MAINTENANCE:
------------
If you add new chapters, update the CHAPTER_MAPPING and CHAPTER_TITLES dictionaries below.
The section IDs must match what's used in the .qmd files (e.g., {#sec-new-chapter}).
"""

import re
import sys
from pathlib import Path

# Chapter mapping from section IDs to their absolute paths from build root
# These are stored as absolute paths from the build root (e.g., contents/core/...)
# and will be converted to relative paths based on each file's location
# Update this when adding new chapters to the book
CHAPTER_MAPPING = {
    # Top-level chapters
    "sec-introduction": "contents/core/introduction/introduction.html#sec-introduction",
    "sec-ml-systems": "contents/core/ml_systems/ml_systems.html#sec-ml-systems",
    "sec-dl-primer": "contents/core/dl_primer/dl_primer.html#sec-dl-primer",
    "sec-dnn-architectures": "contents/core/dnn_architectures/dnn_architectures.html#sec-dnn-architectures",
    "sec-ai-workflow": "contents/core/workflow/workflow.html#sec-ai-workflow",
    "sec-data-engineering": "contents/core/data_engineering/data_engineering.html#sec-data-engineering",
    "sec-ai-frameworks": "contents/core/frameworks/frameworks.html#sec-ai-frameworks",
    "sec-ai-training": "contents/core/training/training.html#sec-ai-training",
    "sec-efficient-ai": "contents/core/efficient_ai/efficient_ai.html#sec-efficient-ai",
    "sec-model-optimizations": "contents/core/optimizations/optimizations.html#sec-model-optimizations",
    "sec-ai-acceleration": "contents/core/hw_acceleration/hw_acceleration.html#sec-ai-acceleration",
    "sec-benchmarking-ai": "contents/core/benchmarking/benchmarking.html#sec-benchmarking-ai",
    "sec-ml-operations": "contents/core/ops/ops.html#sec-ml-operations",
    "sec-ondevice-learning": "contents/core/ondevice_learning/ondevice_learning.html#sec-ondevice-learning",
    "sec-security-privacy": "contents/core/privacy_security/privacy_security.html#sec-security-privacy",
    "sec-robust-ai": "contents/core/robust_ai/robust_ai.html#sec-robust-ai",
    "sec-responsible-ai": "contents/core/responsible_ai/responsible_ai.html#sec-responsible-ai",
    "sec-sustainable-ai": "contents/core/sustainable_ai/sustainable_ai.html#sec-sustainable-ai",
    "sec-ai-good": "contents/core/ai_for_good/ai_for_good.html#sec-ai-good",
    "sec-agi-systems": "contents/core/frontiers/frontiers.html#sec-agi-systems",
    "sec-conclusion": "contents/core/conclusion/conclusion.html#sec-conclusion",

    # Subsections - AI Training chapter
    "sec-ai-training-distributed-systems-8fe8": "contents/core/training/training.html#sec-ai-training-distributed-systems-8fe8",
    "sec-ai-training-neural-network-computation-73f5": "contents/core/training/training.html#sec-ai-training-neural-network-computation-73f5",
    "sec-ai-training-optimization-algorithms-506e": "contents/core/training/training.html#sec-ai-training-optimization-algorithms-506e",

    # Subsections - Efficient AI chapter
    "sec-efficient-ai-ai-scaling-laws-a043": "contents/core/efficient_ai/efficient_ai.html#sec-efficient-ai-ai-scaling-laws-a043",

    # Subsections - Model Optimizations chapter
    "sec-model-optimizations-neural-architecture-search-3915": "contents/core/optimizations/optimizations.html#sec-model-optimizations-neural-architecture-search-3915",
    "sec-model-optimizations-numerical-precision-a93d": "contents/core/optimizations/optimizations.html#sec-model-optimizations-numerical-precision-a93d",
    "sec-model-optimizations-pruning-3f36": "contents/core/optimizations/optimizations.html#sec-model-optimizations-pruning-3f36",

    # Lab sections - Arduino Nicla Vision
    "sec-setup-overview-dcdd": "contents/labs/arduino/nicla_vision/setup/setup.html#sec-setup-overview-dcdd",
    "sec-image-classification-overview-7420": "contents/labs/arduino/nicla_vision/image_classification/image_classification.html#sec-image-classification-overview-7420",
    "sec-object-detection-overview-9d59": "contents/labs/arduino/nicla_vision/object_detection/object_detection.html#sec-object-detection-overview-9d59",
    "sec-keyword-spotting-kws-overview-0ae6": "contents/labs/arduino/nicla_vision/kws/kws.html#sec-keyword-spotting-kws-overview-0ae6",
    "sec-motion-classification-anomaly-detection-overview-b1a8": "contents/labs/arduino/nicla_vision/motion_classification/motion_classification.html#sec-motion-classification-anomaly-detection-overview-b1a8",

    # Lab sections - Seeed XIAO ESP32S3
    "sec-setup-overview-d638": "contents/labs/seeed/xiao_esp32s3/setup/setup.html#sec-setup-overview-d638",
    "sec-image-classification-overview-9a37": "contents/labs/seeed/xiao_esp32s3/image_classification/image_classification.html#sec-image-classification-overview-9a37",
    "sec-object-detection-overview-d035": "contents/labs/seeed/xiao_esp32s3/object_detection/object_detection.html#sec-object-detection-overview-d035",
    "sec-keyword-spotting-kws-overview-4373": "contents/labs/seeed/xiao_esp32s3/kws/kws.html#sec-keyword-spotting-kws-overview-4373",
    "sec-motion-classification-anomaly-detection-overview-cb1f": "contents/labs/seeed/xiao_esp32s3/motion_classification/motion_classification.html#sec-motion-classification-anomaly-detection-overview-cb1f",

    # Lab sections - Grove Vision AI V2
    "sec-setup-nocode-applications-introduction-b740": "contents/labs/seeed/grove_vision_ai_v2/setup_and_no_code_apps/setup_and_no_code_apps.html#sec-setup-nocode-applications-introduction-b740",
    "sec-image-classification-introduction-59d5": "contents/labs/seeed/grove_vision_ai_v2/image_classification/image_classification.html#sec-image-classification-introduction-59d5",

    # Lab sections - Raspberry Pi
    "sec-setup-overview-0ec9": "contents/labs/raspi/setup/setup.html#sec-setup-overview-0ec9",
    "sec-image-classification-overview-3e02": "contents/labs/raspi/image_classification/image_classification.html#sec-image-classification-overview-3e02",
    "sec-object-detection-overview-1133": "contents/labs/raspi/object_detection/object_detection.html#sec-object-detection-overview-1133",
    "sec-small-language-models-slm-overview-ef83": "contents/labs/raspi/llm/llm.html#sec-small-language-models-slm-overview-ef83",
    "sec-visionlanguage-models-vlm-introduction-4272": "contents/labs/raspi/vlm/vlm.html#sec-visionlanguage-models-vlm-introduction-4272"
}

# Chapter titles for readable link text
# Now includes "Chapter" prefix for better HTML readability
CHAPTER_TITLES = {
    # Top-level chapters
    "sec-introduction": "Chapter 1: Introduction",
    "sec-ml-systems": "Chapter 2: ML Systems",
    "sec-dl-primer": "Chapter 3: Deep Learning Primer",
    "sec-dnn-architectures": "Chapter 4: DNN Architectures",
    "sec-ai-workflow": "Chapter 5: AI Workflow",
    "sec-data-engineering": "Chapter 6: Data Engineering",
    "sec-ai-frameworks": "Chapter 7: AI Frameworks",
    "sec-ai-training": "Chapter 8: AI Training",
    "sec-efficient-ai": "Chapter 9: Efficient AI",
    "sec-model-optimizations": "Chapter 10: Model Optimizations",
    "sec-ai-acceleration": "Chapter 11: AI Acceleration",
    "sec-benchmarking-ai": "Chapter 12: Benchmarking AI",
    "sec-ml-operations": "Chapter 13: ML Operations",
    "sec-ondevice-learning": "Chapter 14: On-Device Learning",
    "sec-security-privacy": "Chapter 15: Security & Privacy",
    "sec-robust-ai": "Chapter 16: Robust AI",
    "sec-responsible-ai": "Chapter 17: Responsible AI",
    "sec-sustainable-ai": "Chapter 18: Sustainable AI",
    "sec-ai-good": "Chapter 19: AI for Good",
    "sec-agi-systems": "Chapter 20: AGI Systems",
    "sec-conclusion": "Chapter 21: Conclusion",

    # Subsections - AI Training chapter
    "sec-ai-training-distributed-systems-8fe8": "Distributed Systems",
    "sec-ai-training-neural-network-computation-73f5": "Neural Network Computation",
    "sec-ai-training-optimization-algorithms-506e": "Optimization Algorithms",

    # Subsections - Efficient AI chapter
    "sec-efficient-ai-ai-scaling-laws-a043": "AI Scaling Laws",

    # Subsections - Model Optimizations chapter
    "sec-model-optimizations-neural-architecture-search-3915": "Neural Architecture Search",
    "sec-model-optimizations-numerical-precision-a93d": "Numerical Precision",
    "sec-model-optimizations-pruning-3f36": "Pruning",

    # Lab sections - Arduino Nicla Vision
    "sec-setup-overview-dcdd": "Setup Nicla Vision",
    "sec-image-classification-overview-7420": "Image Classification",
    "sec-object-detection-overview-9d59": "Object Detection",
    "sec-keyword-spotting-kws-overview-0ae6": "Keyword Spotting",
    "sec-motion-classification-anomaly-detection-overview-b1a8": "Motion Classification and Anomaly Detection",

    # Lab sections - Seeed XIAO ESP32S3
    "sec-setup-overview-d638": "Setup the XIAOML Kit",
    "sec-image-classification-overview-9a37": "Image Classification",
    "sec-object-detection-overview-d035": "Object Detection",
    "sec-keyword-spotting-kws-overview-4373": "Keyword Spotting",
    "sec-motion-classification-anomaly-detection-overview-cb1f": "Motion Classification and Anomaly Detection",

    # Lab sections - Grove Vision AI V2
    "sec-setup-nocode-applications-introduction-b740": "Setup and No-Code Apps",
    "sec-image-classification-introduction-59d5": "Image Classification",

    # Lab sections - Raspberry Pi
    "sec-setup-overview-0ec9": "Setup Raspberry Pi",
    "sec-image-classification-overview-3e02": "Image Classification",
    "sec-object-detection-overview-1133": "Object Detection",
    "sec-small-language-models-slm-overview-ef83": "Small Language Models",
    "sec-visionlanguage-models-vlm-introduction-4272": "Visual-Language Models"
}

def build_epub_section_mapping(epub_dir):
    """
    Build mapping from section IDs to EPUB chapter files by scanning actual chapters.

    Args:
        epub_dir: Path to EPUB build directory (_build/epub or extracted EPUB root)

    Returns:
        Dictionary mapping section IDs to chapter filenames (e.g., {"sec-xxx": "ch004.xhtml"})
    """
    mapping = {}

    # Try different possible text directory locations
    possible_text_dirs = [
        epub_dir / "text",  # For _build/epub structure
        epub_dir / "EPUB" / "text",  # For extracted EPUB structure
    ]

    text_dir = None
    for dir_path in possible_text_dirs:
        if dir_path.exists():
            text_dir = dir_path
            break

    if not text_dir:
        return mapping

    # Scan all chapter files
    for xhtml_file in sorted(text_dir.glob("ch*.xhtml")):
        try:
            content = xhtml_file.read_text(encoding='utf-8')
            # Find all section IDs in this file using regex
            section_ids = re.findall(r'id="(sec-[^"]+)"', content)
            for sec_id in section_ids:
                # Map section ID to chapter filename only (no path, since we're in same dir)
                mapping[sec_id] = xhtml_file.name
        except Exception as e:
            continue

    return mapping

def calculate_relative_path(from_file, to_path, build_dir, epub_mapping=None):
    """
    Calculate relative path from one file to another.

    Args:
        from_file: Path object of the source file
        to_path: String path from build root (e.g., "contents/core/chapter/file.html#anchor")
        build_dir: Path object of the build directory root
        epub_mapping: Optional dict mapping section IDs to EPUB chapter files

    Returns:
        Relative path string from from_file to to_path
    """
    # For EPUB builds, use chapter-to-chapter mapping
    if epub_mapping is not None:
        # Extract section ID from to_path
        if '#' in to_path:
            _, anchor_with_hash = to_path.split('#', 1)
            sec_id = anchor_with_hash  # This is already just the section ID

            # Look up which chapter file contains this section
            target_chapter = epub_mapping.get(sec_id)
            if target_chapter:
                # All chapters are in same directory (text/), so just use filename
                return f"{target_chapter}#{sec_id}"

        # Fallback: if no mapping found, return original
        return to_path

    # Original HTML logic for non-EPUB builds
    # Split anchor from path
    if '#' in to_path:
        target_path_str, anchor = to_path.split('#', 1)
        anchor = f'#{anchor}'
    else:
        target_path_str = to_path
        anchor = ''

    # Convert to absolute paths
    target_abs = build_dir / target_path_str
    source_abs = from_file

    # Calculate relative path
    try:
        rel_path = Path(target_abs).relative_to(source_abs.parent)
        # If in same directory or subdirectory, use as is
        result = str(rel_path).replace('\\', '/')
    except ValueError:
        # Need to go up directories
        # Count how many levels up we need to go
        source_parts = source_abs.parent.parts
        target_parts = target_abs.parts

        # Find common prefix
        common_length = 0
        for s, t in zip(source_parts, target_parts):
            if s == t:
                common_length += 1
            else:
                break

        # Calculate relative path
        up_levels = len(source_parts) - common_length
        down_parts = target_parts[common_length:]

        rel_parts = ['..'] * up_levels + list(down_parts)
        result = '/'.join(rel_parts)

    return result + anchor

def fix_cross_reference_link(match, from_file, build_dir, epub_mapping=None):
    """Replace a single cross-reference link with proper HTML link."""
    full_match = match.group(0)
    sec_ref = match.group(1)

    abs_path = CHAPTER_MAPPING.get(sec_ref)
    title = CHAPTER_TITLES.get(sec_ref)

    if abs_path and title:
        # Calculate relative path from current file to target
        rel_path = calculate_relative_path(from_file, abs_path, build_dir, epub_mapping)
        # Create clean HTML link
        return f'<a href="{rel_path}">{title}</a>'
    else:
        # Keep original if no mapping found
        print(f"⚠️ No mapping found for: {sec_ref}")
        return full_match

def fix_cross_references(html_content, from_file, build_dir, epub_mapping=None, verbose=False):
    """
    Fix all cross-reference links in HTML/XHTML content.

    Quarto generates two types of unresolved references when chapters aren't built:
    1. Full unresolved links: <a href="#sec-xxx" class="quarto-xref"><span class="quarto-unresolved-ref">...</span></a>
    2. Simple unresolved refs: <strong>?@sec-xxx</strong> (more common in selective builds)
    3. EPUB unresolved refs: <a href="@sec-xxx">Link Text</a> (EPUB-specific)
    """
    # Pattern 1: Match Quarto's full unresolved cross-reference links
    # Example: <a href="#sec-xxx" class="quarto-xref"><span class="quarto-unresolved-ref">sec-xxx</span></a>
    pattern1 = r'<a href="#(sec-[a-zA-Z0-9-]+)" class="quarto-xref"><span class="quarto-unresolved-ref">[^<]*</span></a>'

    # Pattern 2: Match simple unresolved references (what we see in selective builds)
    # Example: <strong>?@sec-ml-systems</strong>
    # This is what Quarto outputs when it can't resolve a reference to an unbuilt chapter
    pattern2 = r'<strong>\?\@(sec-[a-zA-Z0-9-]+)</strong>'

    # Pattern 3: Match EPUB-specific unresolved references
    # Example: <a href="@sec-xxx">Link Text</a>
    # This is what Quarto outputs in EPUB when it can't resolve a reference
    pattern3 = r'<a href="@(sec-[a-zA-Z0-9-]+)"([^>]*)>([^<]*)</a>'

    # Count matches before replacement
    matches1 = re.findall(pattern1, html_content)
    matches2 = re.findall(pattern2, html_content)
    matches3 = re.findall(pattern3, html_content)
    total_matches = len(matches1) + len(matches2) + len(matches3)

    # Fix Pattern 1 matches
    fixed_content = re.sub(pattern1, lambda m: fix_cross_reference_link(m, from_file, build_dir, epub_mapping), html_content)

    # Fix Pattern 2 matches with proper relative path calculation
    unmapped_refs = []
    def fix_simple_reference(match):
        sec_ref = match.group(1)
        abs_path = CHAPTER_MAPPING.get(sec_ref)
        title = CHAPTER_TITLES.get(sec_ref)
        if abs_path and title:
            rel_path = calculate_relative_path(from_file, abs_path, build_dir, epub_mapping)
            return f'<strong><a href="{rel_path}">{title}</a></strong>'
        else:
            unmapped_refs.append(sec_ref)
            return match.group(0)

    fixed_content = re.sub(pattern2, fix_simple_reference, fixed_content)

    # Fix Pattern 3 matches (EPUB-specific)
    def fix_epub_reference(match):
        sec_ref = match.group(1)
        attrs = match.group(2)  # Additional attributes
        link_text = match.group(3)  # Original link text

        # For EPUB with mapping, use direct chapter lookup
        if epub_mapping:
            target_chapter = epub_mapping.get(sec_ref)
            if target_chapter:
                return f'<a href="{target_chapter}#{sec_ref}"{attrs}>{link_text}</a>'
            else:
                unmapped_refs.append(sec_ref)
                return match.group(0)
        else:
            # Fallback to HTML path resolution
            abs_path = CHAPTER_MAPPING.get(sec_ref)
            title = CHAPTER_TITLES.get(sec_ref)
            if abs_path:
                rel_path = calculate_relative_path(from_file, abs_path, build_dir, None)
                return f'<a href="{rel_path}"{attrs}>{link_text}</a>'
            else:
                unmapped_refs.append(sec_ref)
                return match.group(0)

    fixed_content = re.sub(pattern3, fix_epub_reference, fixed_content)

    # Count successful replacements
    remaining1 = re.findall(pattern1, fixed_content)
    remaining2 = re.findall(pattern2, fixed_content)
    remaining3 = re.findall(pattern3, fixed_content)
    fixed_count = total_matches - len(remaining1) - len(remaining2) - len(remaining3)

    # Return info about what was fixed
    return fixed_content, fixed_count, unmapped_refs

def process_html_file(html_file, base_dir, epub_mapping=None):
    """Process a single HTML/XHTML file to fix cross-references."""
    # Read file content
    try:
        html_content = html_file.read_text(encoding='utf-8')
    except Exception as e:
        return None, 0, []

    # Fix cross-reference links with proper relative path calculation
    fixed_content, fixed_count, unmapped = fix_cross_references(html_content, html_file, base_dir, epub_mapping)

    # Write back fixed content if changes were made
    if fixed_count > 0:
        try:
            html_file.write_text(fixed_content, encoding='utf-8')
            return html_file.relative_to(base_dir), fixed_count, unmapped
        except Exception as e:
            return None, 0, []
    return None, 0, []

def main():
    """
    Main entry point. Runs in three modes:
    1. Post-render hook (no args): Processes HTML or EPUB builds from _build/
    2. Directory mode (dir arg): Processes extracted EPUB directory
    3. Manual mode (file arg): Processes a specific file

    This allows both automatic fixing during builds and manual testing/debugging.
    """
    if len(sys.argv) == 1:
        # MODE 1: Running as Quarto post-render hook
        # Detect if this is HTML or EPUB build
        html_dir = Path("_build/html")
        epub_dir = Path("_build/epub")

        # Determine build type
        epub_mapping = None
        if html_dir.exists() and (html_dir / "index.html").exists():
            build_dir = html_dir
            file_pattern = "*.html"
            file_type = "HTML"
        elif epub_dir.exists() and list(epub_dir.glob("*.xhtml")):
            build_dir = epub_dir
            file_pattern = "*.xhtml"
            file_type = "XHTML (EPUB)"
            # Build EPUB section mapping for dynamic chapter references
            print("📚 Building EPUB section mapping...")
            epub_mapping = build_epub_section_mapping(epub_dir)
            print(f"   Found {len(epub_mapping)} section IDs across chapters")
        # Check for extracted EPUB structure (EPUB/ directory at current level)
        elif Path("EPUB").exists() and list(Path("EPUB").rglob("*.xhtml")):
            build_dir = Path(".")
            file_pattern = "*.xhtml"
            file_type = "XHTML (EPUB - extracted)"
            # Build EPUB section mapping
            print("📚 Building EPUB section mapping...")
            epub_mapping = build_epub_section_mapping(Path("."))
            print(f"   Found {len(epub_mapping)} section IDs across chapters")
        else:
            print("⚠️ No HTML or EPUB build directory found - skipping")
            sys.exit(0)

        # Find all files
        files = list(build_dir.rglob(file_pattern))
        print(f"🔗 [Cross-Reference Fix] Scanning {len(files)} {file_type} files...")

        files_fixed = []
        total_refs_fixed = 0
        all_unmapped = set()

        for file in files:
            # Skip certain files that don't need processing
            skip_patterns = ['search.html', '404.html', 'site_libs', 'nav.xhtml', 'cover.xhtml', 'title_page.xhtml']
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
            print(f"✅ No unresolved cross-references found")

        if all_unmapped:
            print(f"⚠️ Unmapped references: {', '.join(sorted(all_unmapped))}")

    elif len(sys.argv) == 2:
        # MODE 2: Running with explicit file argument
        html_file = Path(sys.argv[1])
        if not html_file.exists():
            print(f"❌ File not found: {html_file}")
            sys.exit(1)

        # Detect if this is an EPUB file (in text/ directory)
        epub_mapping = None
        if 'text' in html_file.parts and html_file.suffix == '.xhtml':
            # This is an EPUB chapter file, build mapping
            epub_base = html_file.parent.parent  # Go up from text/ to EPUB/
            print("📚 Building EPUB section mapping...")
            epub_mapping = build_epub_section_mapping(epub_base)
            print(f"   Found {len(epub_mapping)} section IDs across chapters")

        print(f"🔗 Fixing cross-reference links in: {html_file}")
        rel_path, fixed_count, unmapped = process_html_file(html_file, html_file.parent, epub_mapping)
        if fixed_count > 0:
            print(f"✅ Fixed {fixed_count} cross-references")
            if unmapped:
                print(f"⚠️ Unmapped references: {', '.join(sorted(unmapped))}")
        else:
            print(f"✅ No cross-reference fixes needed")
    else:
        print("Usage: python3 fix_cross_references.py [<html-or-xhtml-file>]")
        sys.exit(1)

if __name__ == "__main__":
    main()
