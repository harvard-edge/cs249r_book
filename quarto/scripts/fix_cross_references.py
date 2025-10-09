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
    "sec-model-optimizations-pruning-3f36": "contents/core/optimizations/optimizations.html#sec-model-optimizations-pruning-3f36"
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
    "sec-model-optimizations-pruning-3f36": "Pruning"
}

def calculate_relative_path(from_file, to_path, build_dir):
    """
    Calculate relative path from one HTML file to another.
    
    Args:
        from_file: Path object of the source HTML file
        to_path: String path from build root (e.g., "contents/core/chapter/file.html#anchor")
        build_dir: Path object of the build directory root
    
    Returns:
        Relative path string from from_file to to_path
    """
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

def fix_cross_reference_link(match, from_file, build_dir):
    """Replace a single cross-reference link with proper HTML link."""
    full_match = match.group(0)
    sec_ref = match.group(1)
    
    abs_path = CHAPTER_MAPPING.get(sec_ref)
    title = CHAPTER_TITLES.get(sec_ref)
    
    if abs_path and title:
        # Calculate relative path from current file to target
        rel_path = calculate_relative_path(from_file, abs_path, build_dir)
        # Create clean HTML link
        return f'<a href="{rel_path}">{title}</a>'
    else:
        # Keep original if no mapping found
        print(f"⚠️ No mapping found for: {sec_ref}")
        return full_match

def fix_cross_references(html_content, from_file, build_dir, verbose=False):
    """
    Fix all cross-reference links in HTML content.
    
    Quarto generates two types of unresolved references when chapters aren't built:
    1. Full unresolved links: <a href="#sec-xxx" class="quarto-xref"><span class="quarto-unresolved-ref">...</span></a>
    2. Simple unresolved refs: <strong>?@sec-xxx</strong> (more common in selective builds)
    """
    # Pattern 1: Match Quarto's full unresolved cross-reference links
    # Example: <a href="#sec-xxx" class="quarto-xref"><span class="quarto-unresolved-ref">sec-xxx</span></a>
    pattern1 = r'<a href="#(sec-[a-zA-Z0-9-]+)" class="quarto-xref"><span class="quarto-unresolved-ref">[^<]*</span></a>'
    
    # Pattern 2: Match simple unresolved references (what we see in selective builds)
    # Example: <strong>?@sec-ml-systems</strong>
    # This is what Quarto outputs when it can't resolve a reference to an unbuilt chapter
    pattern2 = r'<strong>\?\@(sec-[a-zA-Z0-9-]+)</strong>'
    
    # Count matches before replacement
    matches1 = re.findall(pattern1, html_content)
    matches2 = re.findall(pattern2, html_content)
    total_matches = len(matches1) + len(matches2)
    
    # Fix Pattern 1 matches
    fixed_content = re.sub(pattern1, lambda m: fix_cross_reference_link(m, from_file, build_dir), html_content)
    
    # Fix Pattern 2 matches with proper relative path calculation
    unmapped_refs = []
    def fix_simple_reference(match):
        sec_ref = match.group(1)
        abs_path = CHAPTER_MAPPING.get(sec_ref)
        title = CHAPTER_TITLES.get(sec_ref)
        if abs_path and title:
            rel_path = calculate_relative_path(from_file, abs_path, build_dir)
            return f'<strong><a href="{rel_path}">{title}</a></strong>'
        else:
            unmapped_refs.append(sec_ref)
            return match.group(0)
    
    fixed_content = re.sub(pattern2, fix_simple_reference, fixed_content)
    
    # Count successful replacements
    remaining1 = re.findall(pattern1, fixed_content)
    remaining2 = re.findall(pattern2, fixed_content)
    fixed_count = total_matches - len(remaining1) - len(remaining2)
    
    # Return info about what was fixed
    return fixed_content, fixed_count, unmapped_refs

def process_html_file(html_file, base_dir):
    """Process a single HTML file to fix cross-references."""
    # Read HTML content
    try:
        html_content = html_file.read_text(encoding='utf-8')
    except Exception as e:
        return None, 0, []
    
    # Fix cross-reference links with proper relative path calculation
    fixed_content, fixed_count, unmapped = fix_cross_references(html_content, html_file, base_dir)
    
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
    Main entry point. Runs in two modes:
    1. Post-render hook (no args): Processes ALL HTML files in _build/html/
    2. Manual mode (with file arg): Processes a specific HTML file
    
    This allows both automatic fixing during builds and manual testing/debugging.
    """
    if len(sys.argv) == 1:
        # MODE 1: Running as Quarto post-render hook
        # This happens automatically after `quarto render`
        # We process ALL HTML files because unresolved refs can appear anywhere
        build_dir = Path("_build/html")
        if not build_dir.exists():
            print("⚠️ Build directory not found - skipping")
            sys.exit(0)
        
        # Find all HTML files recursively
        html_files = list(build_dir.rglob("*.html"))
        print(f"🔗 [Cross-Reference Fix] Scanning {len(html_files)} HTML files...")
        
        files_fixed = []
        total_refs_fixed = 0
        all_unmapped = set()
        
        for html_file in html_files:
            # Skip certain files that don't need processing
            if any(skip in str(html_file) for skip in ['search.html', '404.html', 'site_libs']):
                continue
            
            rel_path, fixed_count, unmapped = process_html_file(html_file, build_dir)
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
        # Running with explicit file argument
        html_file = Path(sys.argv[1])
        if not html_file.exists():
            print(f"❌ File not found: {html_file}")
            sys.exit(1)
        
        print(f"🔗 Fixing cross-reference links in: {html_file}")
        rel_path, fixed_count, unmapped = process_html_file(html_file, html_file.parent)
        if fixed_count > 0:
            print(f"✅ Fixed {fixed_count} cross-references")
            if unmapped:
                print(f"⚠️ Unmapped references: {', '.join(sorted(unmapped))}")
        else:
            print(f"✅ No cross-reference fixes needed")
    else:
        print("Usage: python3 fix-glossary-html.py [<html-file>]")
        sys.exit(1)

if __name__ == "__main__":
    main()