#!/usr/bin/env python3
"""
Clean up the master glossary to fix inconsistencies and duplicates.

This script:
1. Merges duplicate terms (e.g., "large language models" vs "large_language_model")
2. Standardizes term formatting (removes underscores, fixes casing)
3. Consolidates multiple definitions
4. Fixes chapter mapping issues
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def standardize_term_name(term):
    """Standardize term name by removing underscores and normalizing."""
    # Replace underscores with spaces
    normalized = term.replace("_", " ")
    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    # Strip and lowercase
    return normalized.strip().lower()

def find_best_definition(definitions):
    """Find the best definition from multiple alternatives."""
    if len(definitions) == 1:
        return definitions[0]

    # Split on "Alternative definition:" to get individual definitions
    all_defs = []
    for def_text in definitions:
        if "Alternative definition:" in def_text:
            parts = def_text.split("Alternative definition:")
            all_defs.extend([part.strip() for part in parts if part.strip()])
        else:
            all_defs.append(def_text.strip())

    # Choose the longest, most complete definition
    best_def = max(all_defs, key=len)

    # Clean up any remaining artifacts
    best_def = re.sub(r'^[.:\s]+', '', best_def)
    best_def = re.sub(r'\s+', ' ', best_def)

    return best_def.strip()

def merge_chapter_info(appears_in_lists, chapter_sources):
    """Merge chapter information from multiple entries."""
    all_chapters = set()

    # Collect all chapters from appears_in fields
    for appears_in in appears_in_lists:
        if appears_in:
            all_chapters.update(appears_in)

    # Collect all chapter sources
    for source in chapter_sources:
        if source:
            all_chapters.add(source)

    # Remove any invalid chapters
    valid_chapters = {ch for ch in all_chapters if ch and not ch.startswith('?')}

    if len(valid_chapters) == 1:
        return list(valid_chapters)[0], []
    elif len(valid_chapters) > 1:
        # Use the first alphabetically as primary, rest as appears_in
        sorted_chapters = sorted(valid_chapters)
        return sorted_chapters[0], sorted_chapters
    else:
        return "", []

def clean_global_glossary():
    """Clean up the master glossary."""
    print("🧹 Cleaning Master Glossary")
    print("=" * 50)

    # Load master glossary
    glossary_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/data/global_glossary.json")
    with open(glossary_path) as f:
        data = json.load(f)

    terms = data["terms"]
    print(f"📚 Starting with {len(terms)} terms")

    # Group terms by standardized name
    term_groups = defaultdict(list)

    for term in terms:
        std_name = standardize_term_name(term["term"])
        term_groups[std_name].append(term)

    print(f"📊 Found {len(term_groups)} unique standardized terms")

    # Process each group
    cleaned_terms = []
    duplicates_found = 0

    for std_name, group in term_groups.items():
        if len(group) > 1:
            duplicates_found += len(group) - 1
            print(f"🔄 Merging {len(group)} duplicates for: '{std_name}'")
            for term in group:
                print(f"   - {term['term']}")

        # Merge the group
        definitions = [term["definition"] for term in group]
        appears_in_lists = [term.get("appears_in", []) for term in group]
        chapter_sources = [term.get("chapter_source", "") for term in group]

        # Get best definition and chapter info
        best_definition = find_best_definition(definitions)
        primary_chapter, all_chapters = merge_chapter_info(appears_in_lists, chapter_sources)

        # Use the cleanest term name from the group
        best_term_name = min(group, key=lambda x: (len(x["term"]), x["term"]))["term"]
        if "_" in best_term_name:
            # Prefer version without underscores if available
            no_underscore = [t for t in group if "_" not in t["term"]]
            if no_underscore:
                best_term_name = min(no_underscore, key=lambda x: (len(x["term"]), x["term"]))["term"]

        # Create cleaned term entry
        cleaned_term = {
            "term": best_term_name.lower(),
            "definition": best_definition,
            "chapter_source": primary_chapter,
            "aliases": [],
            "see_also": []
        }

        # Add appears_in if multiple chapters
        if len(all_chapters) > 1:
            cleaned_term["appears_in"] = all_chapters

        cleaned_terms.append(cleaned_term)

    # Sort terms alphabetically
    cleaned_terms.sort(key=lambda x: x["term"])

    # Update metadata
    cleaned_data = {
        "metadata": {
            "type": "global_glossary",
            "version": "3.0.0",
            "generated": datetime.now().isoformat(),
            "total_terms": len(cleaned_terms),
            "standardized": True,
            "cleaned": True,
            "description": "Cleaned and standardized master glossary with merged duplicates and consistent formatting"
        },
        "terms": cleaned_terms
    }

    # Save cleaned glossary
    backup_path = glossary_path.with_suffix('.backup.json')
    if not backup_path.exists():
        print(f"💾 Creating backup: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saving cleaned glossary: {glossary_path}")
    with open(glossary_path, 'w') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    # Report results
    print(f"\n✅ Cleaning complete!")
    print(f"  → Original terms: {len(terms)}")
    print(f"  → Cleaned terms: {len(cleaned_terms)}")
    print(f"  → Duplicates merged: {duplicates_found}")
    print(f"  → Terms saved: {len(cleaned_terms)}")

    return cleaned_data

def main():
    """Main cleaning function."""
    cleaned_data = clean_global_glossary()

    print(f"\n📈 Final Statistics:")
    multi_chapter = [t for t in cleaned_data["terms"] if "appears_in" in t]
    print(f"  → Multi-chapter terms: {len(multi_chapter)}")

    # Show some examples of cleaned terms
    print(f"\n🔍 Sample cleaned terms:")
    for term in cleaned_data["terms"][:5]:
        print(f"   - {term['term']}")
        if "appears_in" in term:
            print(f"     Appears in: {', '.join(term['appears_in'])}")
        else:
            print(f"     Chapter: {term['chapter_source']}")

if __name__ == "__main__":
    main()
