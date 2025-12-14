#!/usr/bin/env python3
"""
Standardize and deduplicate glossary terms across all chapters.

This script:
1. Standardizes term casing (lowercase for consistency)
2. Merges duplicate terms with different definitions
3. Tracks which chapters use each term
4. Ensures consistent formatting
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_json_glossary(json_path):
    """Load a JSON glossary file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_glossary(data, output_path):
    """Save JSON glossary with proper formatting."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def standardize_term(term):
    """Standardize term format (lowercase, stripped)."""
    return term.lower().strip()

def merge_definitions(definitions):
    """Merge multiple definitions for the same term."""
    if len(definitions) == 1:
        return definitions[0]

    # If definitions are very similar (>80% overlap), use the longest one
    # Otherwise, combine them
    base_def = max(definitions, key=len)

    # Check for significant differences
    unique_info = []
    for d in definitions:
        if d != base_def and len(d) > 50:  # Skip very short duplicates
            # Check if this adds unique information
            words_d = set(d.lower().split())
            words_base = set(base_def.lower().split())
            overlap = len(words_d & words_base) / len(words_d)

            if overlap < 0.8:  # Less than 80% overlap
                unique_info.append(d)

    if unique_info:
        # Combine definitions
        combined = base_def
        for info in unique_info:
            if not combined.endswith('.'):
                combined += '.'
            combined += f" Alternative definition: {info}"
        return combined

    return base_def

def standardize_all_glossaries():
    """Standardize all glossary files."""
    base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
    json_files = list(base_dir.glob("**/*_glossary.json"))

    print(f"📚 Standardizing {len(json_files)} glossary files...")

    # First pass: collect all terms and their definitions
    all_terms = defaultdict(lambda: {
        'definitions': [],
        'chapters': [],
        'original_casings': set()
    })

    for json_path in sorted(json_files):
        data = load_json_glossary(json_path)
        chapter = data['metadata']['chapter']

        for term_entry in data['terms']:
            original_term = term_entry['term']
            std_term = standardize_term(original_term)

            all_terms[std_term]['definitions'].append(term_entry['definition'])
            all_terms[std_term]['chapters'].append(chapter)
            all_terms[std_term]['original_casings'].add(original_term)

    print(f"  Found {len(all_terms)} unique terms (after standardization)")

    # Report on case variations
    case_variations = {term: info['original_casings']
                      for term, info in all_terms.items()
                      if len(info['original_casings']) > 1}

    if case_variations:
        print(f"\n⚠️  Terms with case variations: {len(case_variations)}")
        for term, casings in list(case_variations.items())[:5]:
            print(f"    '{term}': {casings}")

    # Second pass: update individual chapter glossaries with standardized terms
    for json_path in sorted(json_files):
        data = load_json_glossary(json_path)
        chapter = data['metadata']['chapter']

        # Standardize terms in this chapter
        standardized_terms = []
        seen_terms = set()

        for term_entry in data['terms']:
            std_term = standardize_term(term_entry['term'])

            # Skip if we've already added this term (duplicate in same chapter)
            if std_term in seen_terms:
                continue

            seen_terms.add(std_term)

            # Use standardized term name
            term_entry['term'] = std_term

            # Keep the definition from this chapter
            # (each chapter keeps its own definition)
            standardized_terms.append(term_entry)

        # Update data
        data['terms'] = sorted(standardized_terms, key=lambda x: x['term'])
        data['metadata']['total_terms'] = len(standardized_terms)
        data['metadata']['standardized'] = True
        data['metadata']['last_updated'] = datetime.now().isoformat()

        # Save updated chapter glossary
        save_json_glossary(data, json_path)
        print(f"  ✓ {chapter}: {len(standardized_terms)} terms")

    # Create improved master glossary
    create_improved_master_glossary(all_terms)

def create_improved_master_glossary(all_terms):
    """Create an improved master glossary with deduplication."""
    print("\n📚 Creating improved master glossary...")

    master_terms = []

    for std_term, info in sorted(all_terms.items()):
        # Get unique definitions
        unique_defs = list(set(info['definitions']))

        # Merge definitions if multiple exist
        final_definition = merge_definitions(unique_defs)

        # Determine primary chapter (first occurrence)
        primary_chapter = info['chapters'][0]

        # Get additional chapters (if any)
        additional_chapters = list(set(info['chapters'][1:]))

        term_entry = {
            "term": std_term,
            "definition": final_definition,
            "chapter_source": primary_chapter,
            "aliases": [],  # Could be populated with case variations
            "see_also": []
        }

        # Add additional chapters if term appears in multiple places
        if additional_chapters:
            term_entry["appears_in"] = sorted(set(info['chapters']))

        # Add case variations as aliases if they exist
        if len(info['original_casings']) > 1:
            # Use the most common casing as primary, others as aliases
            casings = list(info['original_casings'])
            term_entry["case_variations"] = sorted(casings)

        master_terms.append(term_entry)

    # Build master glossary
    master_json = {
        "metadata": {
            "type": "master_glossary",
            "version": "2.0.0",
            "generated": datetime.now().isoformat(),
            "total_terms": len(master_terms),
            "standardized": True,
            "description": "Standardized master glossary with merged definitions and cross-references"
        },
        "terms": master_terms
    }

    # Save master glossary
    master_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/data/master_glossary.json")
    master_path.parent.mkdir(exist_ok=True)
    save_json_glossary(master_json, master_path)

    print(f"✓ Master glossary created: {master_path}")
    print(f"  → {len(master_terms)} standardized terms")

    # Report statistics
    multi_chapter_terms = [t for t in master_terms if 'appears_in' in t]
    case_variation_terms = [t for t in master_terms if 'case_variations' in t]

    print(f"  → {len(multi_chapter_terms)} terms appear in multiple chapters")
    print(f"  → {len(case_variation_terms)} terms have case variations")

    return master_json

def main():
    """Main standardization process."""
    print("🔧 Standardizing Glossary Terms")
    print("=" * 50)

    standardize_all_glossaries()

    print("\n✨ Standardization complete!")
    print("\nImprovements made:")
    print("• All terms now use consistent lowercase format")
    print("• Duplicate definitions merged intelligently")
    print("• Cross-chapter references tracked")
    print("• Case variations preserved as metadata")

if __name__ == "__main__":
    main()
