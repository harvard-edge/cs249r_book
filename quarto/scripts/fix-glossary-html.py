#!/usr/bin/env python3
"""
Post-process HTML glossary files to fix cross-reference links.
Converts #sec- internal links to proper chapter HTML links for website builds.
"""

import re
import sys
from pathlib import Path

# Chapter mapping from section IDs to their HTML file paths with anchors
CHAPTER_MAPPING = {
    "sec-introduction": "../core/introduction/introduction.html#sec-introduction",
    "sec-ml-systems": "../core/ml_systems/ml_systems.html#sec-ml-systems", 
    "sec-dl-primer": "../core/dl_primer/dl_primer.html#sec-dl-primer",
    "sec-dnn-architectures": "../core/dnn_architectures/dnn_architectures.html#sec-dnn-architectures",
    "sec-ai-frameworks": "../core/frameworks/frameworks.html#sec-ai-frameworks",
    "sec-ai-training": "../core/training/training.html#sec-ai-training",
    "sec-benchmarking-ai": "../core/benchmarking/benchmarking.html#sec-benchmarking-ai",
    "sec-data-engineering": "../core/data_engineering/data_engineering.html#sec-data-engineering",
    "sec-ai-acceleration": "../core/hw_acceleration/hw_acceleration.html#sec-ai-acceleration",
    "sec-efficient-ai": "../core/efficient_ai/efficient_ai.html#sec-efficient-ai",
    "sec-model-optimizations": "../core/optimizations/optimizations.html#sec-model-optimizations",
    "sec-ml-operations": "../core/ops/ops.html#sec-ml-operations",
    "sec-ondevice-learning": "../core/ondevice_learning/ondevice_learning.html#sec-ondevice-learning",
    "sec-robust-ai": "../core/robust_ai/robust_ai.html#sec-robust-ai",
    "sec-security-privacy": "../core/privacy_security/privacy_security.html#sec-security-privacy",
    "sec-responsible-ai": "../core/responsible_ai/responsible_ai.html#sec-responsible-ai",
    "sec-sustainable-ai": "../core/sustainable_ai/sustainable_ai.html#sec-sustainable-ai",
    "sec-ai-good": "../core/ai_for_good/ai_for_good.html#sec-ai-good",
    "sec-ai-workflow": "../core/workflow/workflow.html#sec-ai-workflow",
    "sec-conclusion": "../core/conclusion/conclusion.html#sec-conclusion",
    "sec-frontiers-ml-systems": "../core/frontiers/frontiers.html#sec-frontiers-ml-systems",
    "sec-generative-ai": "../core/generative_ai/generative_ai.html#sec-generative-ai"
}

# Chapter titles for readable link text
CHAPTER_TITLES = {
    "sec-introduction": "Introduction",
    "sec-ml-systems": "ML Systems", 
    "sec-dl-primer": "Deep Learning Primer",
    "sec-dnn-architectures": "DNN Architectures",
    "sec-ai-frameworks": "AI Frameworks",
    "sec-ai-training": "AI Training",
    "sec-benchmarking-ai": "Benchmarking AI",
    "sec-data-engineering": "Data Engineering",
    "sec-ai-acceleration": "AI Acceleration",
    "sec-efficient-ai": "Efficient AI",
    "sec-model-optimizations": "Model Optimizations",
    "sec-ml-operations": "ML Operations",
    "sec-ondevice-learning": "On-Device Learning",
    "sec-robust-ai": "Robust AI",
    "sec-security-privacy": "Security & Privacy",
    "sec-responsible-ai": "Responsible AI",
    "sec-sustainable-ai": "Sustainable AI",
    "sec-ai-good": "AI for Good",
    "sec-ai-workflow": "AI Workflow",
    "sec-conclusion": "Conclusion",
    "sec-frontiers-ml-systems": "Frontiers",
    "sec-generative-ai": "Generative AI"
}

def fix_cross_reference_link(match):
    """Replace a single cross-reference link with proper HTML link."""
    full_match = match.group(0)
    sec_ref = match.group(1)
    
    html_path = CHAPTER_MAPPING.get(sec_ref)
    title = CHAPTER_TITLES.get(sec_ref)
    
    if html_path and title:
        # Create clean HTML link
        return f'<a href="{html_path}">{title}</a>'
    else:
        # Keep original if no mapping found
        print(f"⚠️ No mapping found for: {sec_ref}")
        return full_match

def fix_glossary_html(html_content):
    """Fix all cross-reference links in HTML content."""
    # Pattern to match Quarto's cross-reference links
    # <a href="#sec-xxx" class="quarto-xref"><span class="quarto-unresolved-ref">sec-xxx</span></a>
    pattern = r'<a href="#(sec-[a-zA-Z0-9-]+)" class="quarto-xref"><span class="quarto-unresolved-ref">[^<]*</span></a>'
    
    # Count matches before replacement
    matches = re.findall(pattern, html_content)
    print(f"🔍 Found {len(matches)} cross-reference links to fix")
    
    # Replace all matches
    fixed_content = re.sub(pattern, fix_cross_reference_link, html_content)
    
    # Count successful replacements
    remaining_matches = re.findall(pattern, fixed_content)
    fixed_count = len(matches) - len(remaining_matches)
    
    print(f"✅ Fixed {fixed_count} cross-reference links")
    if remaining_matches:
        print(f"⚠️ {len(remaining_matches)} links remain unfixed")
    
    return fixed_content

def main():
    # When run as post-render hook, find the glossary file automatically
    # Otherwise, accept file path as argument
    if len(sys.argv) == 1:
        # Running as post-render hook - find glossary file
        glossary_paths = [
            Path("_build/html/contents/backmatter/glossary/glossary.html"),
            Path("_build/pdf/contents/backmatter/glossary/glossary.html"),
        ]
        
        html_file = None
        for path in glossary_paths:
            if path.exists():
                html_file = path
                break
        
        if not html_file:
            print("⚠️ No glossary file found in standard locations - skipping")
            sys.exit(0)
    elif len(sys.argv) == 2:
        # Running with explicit file argument
        html_file = Path(sys.argv[1])
        if not html_file.exists():
            print(f"❌ File not found: {html_file}")
            sys.exit(1)
    else:
        print("Usage: python3 fix-glossary-html.py [<html-file>]")
        sys.exit(1)
    
    print(f"🔗 Fixing cross-reference links in: {html_file}")
    
    # Read HTML content
    try:
        html_content = html_file.read_text(encoding='utf-8')
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)
    
    # Fix cross-reference links
    fixed_content = fix_glossary_html(html_content)
    
    # Write back fixed content
    try:
        html_file.write_text(fixed_content, encoding='utf-8')
        print(f"✅ Successfully updated: {html_file}")
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()