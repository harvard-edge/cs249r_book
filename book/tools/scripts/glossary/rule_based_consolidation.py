#!/usr/bin/env python3
"""
Rule-based Glossary Consolidation following academic best practices.

This script implements standard consolidation rules for academic glossaries:
1. Merge singular/plural forms (use singular)
2. Merge acronym variations (include full form with acronym)
3. Standardize formatting (apostrophes, hyphens, capitalization)
4. Apply first-appearance rule for chapter attribution
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def apply_consolidation_rules() -> Dict[str, Dict]:
    """Define comprehensive consolidation rules based on academic best practices."""
    return {
        # Exact duplicates with formatting differences
        "application-specific integrated circuit": {
            "canonical": "application-specific integrated circuit (ASIC)",
            "reason": "add_acronym"
        },
        "field-programmable gate array": {
            "canonical": "field-programmable gate array (FPGA)",
            "reason": "add_acronym"
        },
        "graphics processing unit": {
            "canonical": "graphics processing unit (GPU)",
            "reason": "add_acronym"
        },
        "neural processing unit": {
            "canonical": "neural processing unit (NPU)",
            "reason": "add_acronym"
        },
        "tensor processing unit": {
            "canonical": "tensor processing unit (TPU)",
            "reason": "add_acronym"
        },
        "compute unified device architecture": {
            "canonical": "CUDA (Compute Unified Device Architecture)",
            "reason": "add_acronym"
        },

        # Singular/plural consolidations (use singular)
        "foundation models": {
            "canonical": "foundation model",
            "reason": "singular_form"
        },
        "large language models": {
            "canonical": "large language model (LLM)",
            "reason": "singular_form_acronym"
        },
        "machine learning frameworks": {
            "canonical": "machine learning framework",
            "reason": "singular_form"
        },
        "membership inference attacks": {
            "canonical": "membership inference attack",
            "reason": "singular_form"
        },

        # Formatting standardization
        "moores law": {
            "canonical": "Moore's law",
            "reason": "apostrophe"
        },
        "self attention": {
            "canonical": "self-attention",
            "reason": "hyphenation"
        },
        "self supervised learning": {
            "canonical": "self-supervised learning",
            "reason": "hyphenation"
        },
        "in context learning": {
            "canonical": "in-context learning",
            "reason": "hyphenation"
        },
        "zero shot learning": {
            "canonical": "zero-shot learning",
            "reason": "hyphenation"
        },
        "fine tuning": {
            "canonical": "fine-tuning",
            "reason": "hyphenation"
        },
        "text to image": {
            "canonical": "text-to-image",
            "reason": "hyphenation"
        },
        "end to end": {
            "canonical": "end-to-end",
            "reason": "hyphenation"
        },
        "state of the art": {
            "canonical": "state-of-the-art",
            "reason": "hyphenation"
        },

        # Capitalization standardization
        "data centric ai": {
            "canonical": "data-centric AI",
            "reason": "capitalization_hyphenation"
        },
        "generative ai": {
            "canonical": "generative AI",
            "reason": "capitalization"
        },
        "responsible ai": {
            "canonical": "responsible AI",
            "reason": "capitalization"
        },
        "explainable ai": {
            "canonical": "explainable AI",
            "reason": "capitalization"
        },

        # Subset/superset relationships
        "dynamic quantization": {
            "canonical": "quantization",
            "reason": "subset_to_general"
        },
        "dynamic pruning": {
            "canonical": "pruning",
            "reason": "subset_to_general"
        },
        "batch inference": {
            "canonical": "inference",
            "reason": "subset_to_general"
        },

        # Redundant prefixes
        "ml benchmarking": {
            "canonical": "benchmarking",
            "reason": "remove_redundant_prefix"
        },
        "cloud machine learning": {
            "canonical": "machine learning",
            "reason": "remove_redundant_prefix"
        },
        "edge machine learning": {
            "canonical": "machine learning",
            "reason": "remove_redundant_prefix"
        },

        # Technical term standardization
        "variational autoencoder": {
            "canonical": "variational autoencoder (VAE)",
            "reason": "add_acronym"
        },
        "principal component analysis": {
            "canonical": "principal component analysis (PCA)",
            "reason": "add_acronym"
        },
        "support vector machines": {
            "canonical": "support vector machine (SVM)",
            "reason": "singular_form_acronym"
        },
        "long short-term memory": {
            "canonical": "long short-term memory (LSTM)",
            "reason": "add_acronym"
        },
        "convolutional neural networks": {
            "canonical": "convolutional neural network (CNN)",
            "reason": "singular_form_acronym"
        },
        "recurrent neural networks": {
            "canonical": "recurrent neural network (RNN)",
            "reason": "singular_form_acronym"
        }
    }

def find_terms_to_consolidate(terms: List[Dict]) -> Dict[str, List[Dict]]:
    """Find terms that need consolidation based on rules."""
    consolidation_rules = apply_consolidation_rules()
    consolidation_map = {}

    for term_entry in terms:
        term = term_entry['term']
        normalized = term.lower()

        if normalized in consolidation_rules:
            canonical = consolidation_rules[normalized]['canonical']
            if canonical not in consolidation_map:
                consolidation_map[canonical] = []
            consolidation_map[canonical].append(term_entry)

    return consolidation_map

def merge_term_definitions(term_entries: List[Dict], canonical_term: str) -> Dict:
    """Merge multiple term entries into a single canonical entry."""
    # Sort by chapter appearance order (introduction first, etc.)
    chapter_priority = {
        'introduction': 0, 'ml_systems': 1, 'dl_primer': 2, 'dnn_architectures': 3,
        'frameworks': 4, 'training': 5, 'benchmarking': 6, 'data_engineering': 7,
        'hw_acceleration': 8, 'efficient_ai': 9, 'optimizations': 10, 'ops': 11,
        'ondevice_learning': 12, 'robust_ai': 13, 'privacy_security': 14,
        'responsible_ai': 15, 'sustainable_ai': 16, 'ai_for_good': 17,
        'workflow': 18, 'conclusion': 19, 'frontiers': 20, 'generative_ai': 21
    }

    # Sort entries by chapter priority
    sorted_entries = sorted(term_entries,
                           key=lambda x: chapter_priority.get(x.get('chapter_source', ''), 999))

    # Use the definition from the first (most foundational) chapter
    primary_entry = sorted_entries[0]

    # Collect all chapters where this term appears
    all_chapters = []
    for entry in term_entries:
        chapter = entry.get('chapter_source')
        if chapter and chapter not in all_chapters:
            all_chapters.append(chapter)

        # Also check appears_in field
        appears_in = entry.get('appears_in', [])
        for ch in appears_in:
            if ch not in all_chapters:
                all_chapters.append(ch)

    # Create merged entry
    merged_entry = {
        'term': canonical_term,
        'definition': primary_entry['definition'],
        'chapter_source': all_chapters[0] if all_chapters else primary_entry.get('chapter_source', ''),
        'aliases': [],
        'see_also': primary_entry.get('see_also', [])
    }

    # Set appears_in based on number of chapters
    if len(all_chapters) > 1:
        merged_entry['appears_in'] = all_chapters

    return merged_entry

def apply_rule_based_consolidation(global_glossary_path: Path) -> Tuple[int, List[Dict]]:
    """Apply rule-based consolidation to the master glossary."""

    # Load current master glossary
    with open(global_glossary_path, 'r') as f:
        data = json.load(f)

    original_count = len(data['terms'])
    print(f"📚 Original terms: {original_count}")

    # Find terms to consolidate
    consolidation_map = find_terms_to_consolidate(data['terms'])

    if not consolidation_map:
        print("✅ No terms found that need rule-based consolidation")
        return 0, []

    print(f"🔄 Found {len(consolidation_map)} canonical terms to consolidate:")

    # Apply consolidations
    consolidated_terms = []
    terms_to_remove = set()
    consolidation_log = []

    for canonical_term, term_entries in consolidation_map.items():
        if len(term_entries) > 1:
            print(f"  • {canonical_term}: merging {len(term_entries)} variants")
            for entry in term_entries:
                print(f"    - '{entry['term']}' (from {entry.get('chapter_source', 'unknown')})")
                terms_to_remove.add(entry['term'])

            # Create merged entry
            merged_entry = merge_term_definitions(term_entries, canonical_term)
            consolidated_terms.append(merged_entry)

            # Log the consolidation
            consolidation_log.append({
                'canonical_term': canonical_term,
                'merged_terms': [e['term'] for e in term_entries],
                'chapters': merged_entry.get('appears_in', [merged_entry['chapter_source']]),
                'action': 'merged'
            })

    # Add remaining terms that weren't consolidated
    for term in data['terms']:
        if term['term'] not in terms_to_remove:
            consolidated_terms.append(term)

    # Update the master glossary
    data['terms'] = consolidated_terms
    data['metadata']['total_terms'] = len(consolidated_terms)
    data['metadata']['last_updated'] = 'rule_based_consolidation'

    # Save updated glossary
    with open(global_glossary_path, 'w') as f:
        json.dump(data, f, indent=2)

    reduction = original_count - len(consolidated_terms)
    print(f"✅ Consolidation complete: {original_count} → {len(consolidated_terms)} terms (-{reduction})")

    return reduction, consolidation_log

def main():
    """Main function for rule-based consolidation."""
    print("🔧 Rule-Based Glossary Consolidation")
    print("=" * 60)

    project_root = Path(__file__).parent.parent.parent.parent
    master_path = project_root / "quarto/contents/data/global_glossary.json"

    # Apply consolidation
    reduction, log = apply_rule_based_consolidation(master_path)

    if reduction > 0:
        print(f"\n📊 Consolidation Summary:")
        print(f"  → Terms reduced: {reduction}")
        print(f"  → Consolidations applied: {len(log)}")

        print(f"\n📋 Next steps:")
        print(f"  1. Run generate_glossary.py to update the glossary page")
        print(f"  2. Review the consolidated terms for accuracy")
        print(f"  3. Test cross-references in the full website build")

    # Save consolidation log
    if log:
        log_path = master_path.parent / 'rule_based_consolidation_log.json'
        with open(log_path, 'w') as f:
            json.dump({
                'timestamp': 'automated_rule_based',
                'total_consolidations': len(log),
                'terms_reduced': reduction,
                'consolidations': log
            }, f, indent=2)
        print(f"📋 Consolidation log saved: {log_path}")

if __name__ == "__main__":
    main()
