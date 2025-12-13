#!/usr/bin/env python3
"""
Generate glossary.qmd file from master glossary JSON.

This script reads the master glossary JSON file and generates a properly
formatted Quarto markdown file for the comprehensive glossary page.
"""

import json
import re
from pathlib import Path
from datetime import datetime

def load_global_glossary():
    """Load the master glossary JSON file."""
    project_root = Path(__file__).parent.parent.parent.parent
    glossary_path = project_root / "quarto/contents/data/global_glossary.json"
    with open(glossary_path) as f:
        return json.load(f)

def extract_section_id_from_qmd(qmd_path):
    """Extract the main chapter section ID from a QMD file."""
    try:
        with open(qmd_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for main chapter heading with section ID
        # Pattern: # Chapter Title {#sec-something}
        pattern = r'^#\s+[^{]*\{#(sec-[^}]+)\}'
        match = re.search(pattern, content, re.MULTILINE)

        if match:
            return f"@{match.group(1)}"

        return None

    except Exception as e:
        print(f"Error reading {qmd_path}: {e}")
        return None

def build_simple_chapter_mapping():
    """Build mapping from chapter names to their actual section IDs."""
    project_root = Path(__file__).parent.parent.parent.parent
    chapters_dir = project_root / "quarto/contents/core"

    chapter_mapping = {}

    # List of chapter directories to scan
    chapter_dirs = [
        "introduction", "ml_systems", "dl_primer", "dnn_architectures",
        "frameworks", "training", "benchmarking", "data_engineering",
        "hw_acceleration", "efficient_ai", "optimizations", "ops",
        "ondevice_learning", "robust_ai", "privacy_security", "responsible_ai",
        "sustainable_ai", "ai_for_good", "workflow", "conclusion",
        "frontiers", "generative_ai"
    ]

    for chapter_name in chapter_dirs:
        chapter_path = chapters_dir / chapter_name / f"{chapter_name}.qmd"

        if chapter_path.exists():
            section_id = extract_section_id_from_qmd(chapter_path)
            if section_id:
                chapter_mapping[chapter_name] = section_id

    return chapter_mapping

def format_chapter_link(chapter):
    """Format chapter name as Quarto cross-reference link."""
    if not chapter:
        return ""

    # Use the simple section ID discovery to get actual section IDs from chapter files
    try:
        chapter_mapping = build_simple_chapter_mapping()
        return chapter_mapping.get(chapter, f"@sec-{chapter.replace('_', '-')}")
    except Exception as e:
        print(f"Warning: Could not build chapter mapping, using fallback: {e}")
        # Fallback to manual mapping if section ID discovery fails
        fallback_mapping = {
            "introduction": "@sec-introduction",
            "ml_systems": "@sec-ml-systems",
            "dl_primer": "@sec-dl-primer",
            "dnn_architectures": "@sec-dnn-architectures",
            "frameworks": "@sec-ai-frameworks",
            "training": "@sec-ai-training",
            "benchmarking": "@sec-benchmarking-ai",
            "data_engineering": "@sec-data-engineering",
            "hw_acceleration": "@sec-ai-acceleration",
            "efficient_ai": "@sec-efficient-ai",
            "optimizations": "@sec-model-optimizations",
            "ops": "@sec-ml-operations",
            "ondevice_learning": "@sec-ondevice-learning",
            "robust_ai": "@sec-robust-ai",
            "privacy_security": "@sec-security-privacy",
            "responsible_ai": "@sec-responsible-ai",
            "sustainable_ai": "@sec-sustainable-ai",
            "ai_for_good": "@sec-ai-good",
            "workflow": "@sec-ai-workflow",
            "conclusion": "@sec-conclusion",
            "frontiers": "@sec-frontiers-ml-systems",
            "generative_ai": "@sec-generative-ai"
        }
        return fallback_mapping.get(chapter, f"@sec-{chapter.replace('_', '-')}")

def generate_glossary_qmd(glossary_data):
    """Generate the glossary QMD content."""
    terms = glossary_data["terms"]
    total_terms = len(terms)

    # Header content
    content = [
        "---",
        "number-sections: false",
        "---",
        "",
        "# Glossary {.unnumbered}",
        "",
        "This comprehensive glossary contains definitions of key terms used throughout the ML Systems textbook. Terms are organized alphabetically and include references to the chapters where they appear.",
        "",
        "::: {.callout-note}",
        "## Using the Glossary",
        "",
        "- **Terms are alphabetically ordered** for easy reference",
        "- **Chapter references** show where terms are introduced or discussed",
        "- **Cross-references** help you explore related concepts",
        "- **Interactive tooltips** appear when you hover over glossary terms throughout the book",
        ":::",
        ""
    ]

    # Group terms by first letter
    terms_by_letter = {}
    for term in terms:
        first_letter = term["term"][0].upper()
        if first_letter not in terms_by_letter:
            terms_by_letter[first_letter] = []
        terms_by_letter[first_letter].append(term)

    # Generate glossary entries organized by letter
    for letter in sorted(terms_by_letter.keys()):
        content.append(f"## {letter}")
        content.append("")

        for term in sorted(terms_by_letter[letter], key=lambda x: x["term"]):
            term_name = term["term"]
            definition = term["definition"]

            # Create the term entry with proper formatting
            content.append(f"**{term_name}**")
            content.append(f": {definition}")

            # Handle chapter references consistently
            appears_in = term.get("appears_in", [])
            chapter_source = term.get("chapter_source", "")

            if appears_in and len(appears_in) > 1:
                # Multiple chapters - list them all
                formatted_chapters = [format_chapter_link(ch) for ch in appears_in]
                content.append(f"  *Appears in: {', '.join(formatted_chapters)}*")
            elif chapter_source:
                # Single primary chapter - use consistent "Appears in:" label
                formatted_chapter = format_chapter_link(chapter_source)
                content.append(f"  *Appears in: {formatted_chapter}*")

            # Add cross-references if available
            if term.get("see_also"):
                see_also = ", ".join(term["see_also"])
                content.append(f"  *See also: {see_also}*")

            content.append("")  # Add spacing between terms

    # Footer content
    content.extend([
        "---",
        "",
        "## About This Glossary",
        "",
        "This glossary was automatically generated from chapter-specific glossaries throughout the textbook, ensuring consistency and completeness. Each term is defined in the context of machine learning systems and includes references to help you explore related concepts.",
        "",
        "**Coverage**: {{< meta title >}} covers the full spectrum of ML systems from foundational concepts to cutting-edge applications, and this glossary reflects that comprehensive scope.",
        "",
        "**Updates**: The glossary is maintained alongside the textbook content to ensure definitions remain current and accurate.",
        "",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*"
    ])

    return "\n".join(content)

def main():
    """Main function to generate the glossary."""
    print("🔧 Generating Glossary QMD File")
    print("=" * 50)

    # Load glossary data
    print("📚 Loading master glossary...")
    glossary_data = load_global_glossary()
    total_terms = len(glossary_data["terms"])
    print(f"  → Found {total_terms} terms")

    # Generate QMD content
    print("📝 Generating QMD content...")
    qmd_content = generate_glossary_qmd(glossary_data)

    # Write to file
    project_root = Path(__file__).parent.parent.parent.parent
    output_path = project_root / "quarto/contents/backmatter/glossary/glossary.qmd"
    print(f"💾 Writing to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(qmd_content)

    print("✅ Glossary QMD file generated successfully!")
    print(f"  → Output: {output_path}")
    print(f"  → Terms: {total_terms}")

    # Count terms by letter for summary
    terms_by_letter = {}
    for term in glossary_data["terms"]:
        first_letter = term["term"][0].upper()
        terms_by_letter[first_letter] = terms_by_letter.get(first_letter, 0) + 1

    print(f"  → Sections: {len(terms_by_letter)} letter sections (A-Z)")
    print("  → Letter distribution:")
    for letter in sorted(terms_by_letter.keys()):
        print(f"     {letter}: {terms_by_letter[letter]} terms")

if __name__ == "__main__":
    main()
