#!/usr/bin/env python3
"""
Generate glossary.qmd files from volume-specific glossary JSONs.

This script reads volume-specific glossary JSON files and generates properly
formatted Quarto markdown files for each volume's glossary page.
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime

# Chapter to section ID mappings by volume
VOL1_CHAPTER_MAPPING = {
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
    "serving": "@sec-serving",
    "responsible_engr": "@sec-responsible-engr",
    "workflow": "@sec-ai-workflow",
    "conclusion": "@sec-conclusion-vol1",
}

VOL2_CHAPTER_MAPPING = {
    "introduction": "@sec-vol2-introduction",
    "infrastructure": "@sec-infrastructure",
    "distributed_training": "@sec-distributed-training",
    "inference": "@sec-inference",
    "communication": "@sec-communication",
    "storage": "@sec-storage",
    "ops_scale": "@sec-ops-scale",
    "fault_tolerance": "@sec-fault-tolerance",
    "edge_intelligence": "@sec-edge-intelligence",
    "robust_ai": "@sec-robust-ai",
    "privacy_security": "@sec-privacy-security",
    "responsible_ai": "@sec-responsible-ai",
    "sustainable_ai": "@sec-sustainable-ai",
    "ai_for_good": "@sec-ai-for-good",
    "frontiers": "@sec-frontiers",
    "conclusion": "@sec-conclusion-vol2",
}

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

def build_chapter_mapping(volume):
    """Build mapping from chapter names to their actual section IDs."""
    project_root = Path(__file__).parent.parent.parent.parent
    chapters_dir = project_root / f"quarto/contents/{volume}"

    chapter_mapping = {}

    # Try to discover section IDs from actual files
    if chapters_dir.exists():
        for chapter_dir in chapters_dir.iterdir():
            if chapter_dir.is_dir() and chapter_dir.name not in ['frontmatter', 'backmatter']:
                chapter_name = chapter_dir.name
                chapter_path = chapter_dir / f"{chapter_name}.qmd"

                if chapter_path.exists():
                    section_id = extract_section_id_from_qmd(chapter_path)
                    if section_id:
                        chapter_mapping[chapter_name] = section_id

    # Fall back to predefined mappings for any missing chapters
    fallback = VOL1_CHAPTER_MAPPING if volume == "vol1" else VOL2_CHAPTER_MAPPING
    for chapter, section_id in fallback.items():
        if chapter not in chapter_mapping:
            chapter_mapping[chapter] = section_id

    return chapter_mapping

def format_chapter_link(chapter, volume):
    """Format chapter name as Quarto cross-reference link."""
    if not chapter:
        return ""

    chapter_mapping = build_chapter_mapping(volume)
    return chapter_mapping.get(chapter, f"@sec-{chapter.replace('_', '-')}")

def load_volume_glossary(volume):
    """Load the volume-specific glossary JSON file."""
    project_root = Path(__file__).parent.parent.parent.parent
    glossary_path = project_root / f"quarto/contents/{volume}/backmatter/glossary/{volume}_glossary.json"
    
    if not glossary_path.exists():
        print(f"⚠️  Glossary not found: {glossary_path}")
        return None
        
    with open(glossary_path) as f:
        return json.load(f)


def generate_glossary_qmd(glossary_data, volume=None, volume_title=None):
    """Generate the glossary QMD content."""
    terms = glossary_data["terms"]
    total_terms = len(terms)

    # Determine volume description
    if volume_title:
        title_suffix = f" ({volume_title})"
        intro_text = f"This glossary contains definitions of key terms used in {volume_title}."
    else:
        title_suffix = ""
        intro_text = "This comprehensive glossary contains definitions of key terms used throughout the ML Systems textbook."

    # Header content
    content = [
        "---",
        "number-sections: false",
        "---",
        "",
        f"# Glossary{title_suffix} {{.unnumbered}}",
        "",
        f"{intro_text} Terms are organized alphabetically and include references to the chapters where they appear.",
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
                # Multiple chapters
                formatted_chapters = [format_chapter_link(ch, volume or "vol1") for ch in appears_in]
                content.append(f"  *Appears in: {', '.join(formatted_chapters)}*")
            elif chapter_source:
                # Single primary chapter
                formatted_chapter = format_chapter_link(chapter_source, volume or "vol1")
                content.append(f"  *Appears in: {formatted_chapter}*")

            # Add cross-references if available
            if term.get("see_also"):
                see_also = ", ".join(term["see_also"])
                content.append(f"  *See also: {see_also}*")

            content.append("")  # Add spacing between terms

    # Footer content
    if volume_title:
        about_text = f"This glossary was automatically generated from chapter glossaries in {volume_title}."
    else:
        about_text = "This glossary was automatically generated from chapter glossaries throughout the textbook."

    content.extend([
        "---",
        "",
        "## About This Glossary",
        "",
        f"{about_text} Each term is defined in the context of machine learning systems and includes references to help you explore related concepts.",
        "",
        "**Updates**: The glossary is maintained alongside the textbook content to ensure definitions remain current and accurate.",
        "",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*"
    ])

    return "\n".join(content)

def generate_volume_glossary(volume, volume_title):
    """Generate glossary QMD for a specific volume."""
    print(f"\n📚 Generating glossary for {volume_title}...")
    
    glossary_data = load_volume_glossary(volume)
    if not glossary_data:
        return False

    total_terms = len(glossary_data["terms"])
    print(f"  → Found {total_terms} terms")

    # Generate QMD content
    qmd_content = generate_glossary_qmd(glossary_data, volume, volume_title)

    # Write to file
    project_root = Path(__file__).parent.parent.parent.parent
    output_path = project_root / f"quarto/contents/{volume}/backmatter/glossary/glossary.qmd"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(qmd_content)

    print(f"✅ {volume_title} glossary generated: {total_terms} terms")
    return True

def main():
    """Main function to generate glossaries."""
    parser = argparse.ArgumentParser(description="Generate glossary QMD files")
    parser.add_argument("--volume", choices=["vol1", "vol2", "all"], 
                       default="all", help="Which volume(s) to generate")
    args = parser.parse_args()

    print("🔧 Generating Glossary QMD Files")
    print("=" * 50)

    if args.volume == "vol1":
        generate_volume_glossary("vol1", "Volume I: Foundations")
    elif args.volume == "vol2":
        generate_volume_glossary("vol2", "Volume II: Scalable Systems")
    else:  # all
        generate_volume_glossary("vol1", "Volume I: Foundations")
        generate_volume_glossary("vol2", "Volume II: Scalable Systems")

    print("\n✅ Glossary generation complete!")

if __name__ == "__main__":
    main()
