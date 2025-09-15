#!/usr/bin/env python3
"""
Generate glossary.qmd file from master glossary JSON.

This script reads the master glossary JSON file and generates a properly
formatted Quarto markdown file for the comprehensive glossary page.
"""

import json
from pathlib import Path
from datetime import datetime

def load_master_glossary():
    """Load the master glossary JSON file."""
    glossary_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/data/master_glossary.json")
    with open(glossary_path) as f:
        return json.load(f)

def format_chapter_name(chapter):
    """Format chapter name for display."""
    if not chapter:
        return ""
    # Convert underscores to spaces and title case
    return chapter.replace("_", " ").title()

def generate_glossary_qmd(glossary_data):
    """Generate the glossary QMD content."""
    terms = glossary_data["terms"]
    total_terms = len(terms)
    
    # Header content
    content = [
        "---",
        "title: \"Glossary\"",
        "number-sections: false",
        "---",
        "",
        "# Glossary {.unnumbered}",
        "",
        "This comprehensive glossary contains definitions of key terms used throughout the ML Systems textbook. Terms are organized alphabetically and include cross-references to the chapters where they are primarily discussed.",
        "",
        "::: {.callout-note}",
        "## Using the Glossary",
        "",
        "- **Terms are alphabetically ordered** for easy reference",
        "- **Chapter sources** indicate where each term is primarily discussed", 
        "- **Cross-references** help you explore related concepts",
        "- **Interactive tooltips** appear when you hover over glossary terms throughout the book",
        ":::",
        "",
        f"*{total_terms} terms defined across all chapters*",
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
            chapter = term.get("chapter_source", "")
            
            # Create the term entry with proper formatting
            content.append(f"**{term_name}**")
            content.append(f": {definition}")
            
            if chapter:
                formatted_chapter = format_chapter_name(chapter)
                content.append(f"  *→ Chapter: {formatted_chapter}*")
            
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
    glossary_data = load_master_glossary()
    total_terms = len(glossary_data["terms"])
    print(f"  → Found {total_terms} terms")
    
    # Generate QMD content
    print("📝 Generating QMD content...")
    qmd_content = generate_glossary_qmd(glossary_data)
    
    # Write to file
    output_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/backmatter/glossary.qmd")
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