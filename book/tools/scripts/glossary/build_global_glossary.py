#!/usr/bin/env python3
"""
Build clean master glossary from individual chapter glossaries.

This script:
1. Reads all individual chapter glossary JSON files (source of truth)
2. Standardizes and deduplicates terms during aggregation
3. Identifies cross-chapter terms and consolidates definitions
4. Outputs clean master glossary for use by glossary page generation

This is the proper data flow:
chapter glossaries (clean) → aggregation (smart) → master glossary (clean)
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_chapter_glossaries():
    """Load all individual chapter glossary files."""
    # Get project root (4 levels up from this script)
    project_root = Path(__file__).parent.parent.parent.parent
    base_dir = project_root / "quarto/contents/core"
    json_files = list(base_dir.glob("**/*_glossary.json"))

    print(f"📚 Found {len(json_files)} chapter glossary files")

    chapter_data = {}
    total_raw_terms = 0

    for json_path in sorted(json_files):
        try:
            with open(json_path) as f:
                data = json.load(f)

            chapter = data['metadata']['chapter']
            terms = data['terms']
            chapter_data[chapter] = terms
            total_raw_terms += len(terms)

            print(f"  → {chapter}: {len(terms)} terms")

        except Exception as e:
            print(f"  ❌ Error loading {json_path}: {e}")

    print(f"📊 Total raw terms across all chapters: {total_raw_terms}")
    return chapter_data

def standardize_term_name(term):
    """Standardize term name consistently."""
    # Remove underscores, normalize spacing, lowercase
    normalized = re.sub(r'[_\s]+', ' ', term.strip().lower())
    return normalized

def find_best_definition(definitions_with_chapters):
    """Find the best definition from multiple chapters."""
    if len(definitions_with_chapters) == 1:
        return definitions_with_chapters[0]['definition']

    # Prefer definitions that are:
    # 1. From primary/core chapters (training, dl_primer, etc.)
    # 2. Longer and more comprehensive
    # 3. Don't have "Alternative definition:" artifacts

    priority_chapters = ['dl_primer', 'training', 'ml_systems', 'dnn_architectures']

    # First try priority chapters
    for chapter in priority_chapters:
        for item in definitions_with_chapters:
            if item['chapter'] == chapter:
                definition = item['definition']
                if not definition.startswith('Alternative definition:'):
                    return definition

    # Otherwise, pick the longest clean definition
    clean_definitions = []
    for item in definitions_with_chapters:
        def_text = item['definition']
        # Take the first part before "Alternative definition:" if it exists
        if 'Alternative definition:' in def_text:
            def_text = def_text.split('Alternative definition:')[0].strip()
        clean_definitions.append((def_text, item['chapter']))

    # Return the longest definition
    best_def = max(clean_definitions, key=lambda x: len(x[0]))
    return best_def[0].rstrip('.')

def aggregate_terms(chapter_data):
    """Aggregate terms from all chapters with smart deduplication."""
    print("\n🔄 Aggregating and deduplicating terms...")

    # Group terms by standardized name
    term_groups = defaultdict(list)

    for chapter, terms in chapter_data.items():
        for term_entry in terms:
            std_name = standardize_term_name(term_entry['term'])
            term_groups[std_name].append({
                'original_term': term_entry['term'],
                'definition': term_entry['definition'],
                'chapter': chapter
            })

    print(f"📊 Found {len(term_groups)} unique standardized terms")

    # Process each group
    clean_terms = []
    duplicates_merged = 0

    for std_name, group in term_groups.items():
        if len(group) > 1:
            duplicates_merged += len(group) - 1
            chapters = [item['chapter'] for item in group]
            print(f"🔄 Merging {len(group)} occurrences of '{std_name}' from: {', '.join(chapters)}")

        # Determine best term name (prefer clean versions)
        term_names = [item['original_term'] for item in group]
        best_term_name = min(term_names, key=lambda x: (len(x), '_' in x, x.lower()))

        # Get best definition
        best_definition = find_best_definition(group)

        # Determine chapter attribution
        chapters = [item['chapter'] for item in group]
        unique_chapters = sorted(set(chapters))

        if len(unique_chapters) == 1:
            chapter_source = unique_chapters[0]
            appears_in = []
        else:
            # Use first chapter as primary source
            chapter_source = unique_chapters[0]
            appears_in = unique_chapters

        # Create clean term entry
        clean_term = {
            "term": best_term_name.lower(),
            "definition": best_definition,
            "chapter_source": chapter_source,
            "aliases": [],
            "see_also": []
        }

        if appears_in:
            clean_term["appears_in"] = appears_in

        clean_terms.append(clean_term)

    # Sort alphabetically
    clean_terms.sort(key=lambda x: x["term"])

    print(f"✅ Aggregation complete:")
    print(f"  → Unique terms: {len(clean_terms)}")
    print(f"  → Duplicates merged: {duplicates_merged}")

    return clean_terms

def build_global_glossary():
    """Build the master glossary from individual chapter files."""
    print("🔧 Building Master Glossary from Chapter Sources")
    print("=" * 60)

    # Load all chapter glossaries
    chapter_data = load_chapter_glossaries()

    # Aggregate and deduplicate
    clean_terms = aggregate_terms(chapter_data)

    # Build master glossary structure
    global_glossary = {
        "metadata": {
            "type": "global_glossary",
            "version": "3.0.0",
            "generated": datetime.now().isoformat(),
            "total_terms": len(clean_terms),
            "source": "aggregated_from_chapter_glossaries",
            "standardized": True,
            "description": "Master glossary built by aggregating and deduplicating individual chapter glossaries"
        },
        "terms": clean_terms
    }

    # Save master glossary
    project_root = Path(__file__).parent.parent.parent.parent
    output_path = project_root / "quarto/contents/data/global_glossary.json"

    # Create backup if exists
    if output_path.exists():
        backup_path = output_path.with_suffix('.backup.json')
        if not backup_path.exists():
            print(f"💾 Creating backup: {backup_path}")
            output_path.rename(backup_path)

    print(f"💾 Saving master glossary: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(global_glossary, f, indent=2, ensure_ascii=False)

    # Report statistics
    multi_chapter_terms = [t for t in clean_terms if "appears_in" in t]

    print(f"\n📈 Final Master Glossary Statistics:")
    print(f"  → Total terms: {len(clean_terms)}")
    print(f"  → Multi-chapter terms: {len(multi_chapter_terms)}")
    print(f"  → Single-chapter terms: {len(clean_terms) - len(multi_chapter_terms)}")

    # Show examples
    print(f"\n🔍 Example multi-chapter terms:")
    for term in multi_chapter_terms[:5]:
        chapters = ', '.join(term['appears_in'])
        print(f"  → {term['term']}: {chapters}")

    return global_glossary

def main():
    """Main function."""
    global_glossary = build_global_glossary()

    print(f"\n✅ Master glossary successfully built!")
    print(f"Next steps:")
    print(f"  1. Review individual chapter glossaries for any needed cleanup")
    print(f"  2. Run generate_glossary.py to create the glossary page")
    print(f"  3. Individual chapter glossaries remain the source of truth")

if __name__ == "__main__":
    main()
