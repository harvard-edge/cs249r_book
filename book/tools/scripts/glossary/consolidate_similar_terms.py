#!/usr/bin/env python3
"""
Consolidate similar and duplicate terms in the glossary.

This script identifies and merges similar terms in chapter glossaries:
- Singular vs plural forms (e.g., "adversarial example" vs "adversarial examples")
- Terms with and without acronyms (e.g., "application-specific integrated circuit" vs "application-specific integrated circuit (asic)")
- Subset terms that should be merged into the more general term
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

def normalize_term(term: str) -> str:
    """Normalize a term for comparison."""
    # Remove acronyms in parentheses for comparison
    normalized = term.lower()
    # Remove content in parentheses
    import re
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)
    # Remove trailing 's' for plural
    if normalized.endswith('s') and not normalized.endswith('ss'):
        normalized = normalized[:-1]
    return normalized.strip()

def find_consolidation_rules() -> Dict[str, str]:
    """Define rules for consolidating similar terms."""
    return {
        # Singular/plural consolidations (use singular form)
        "adversarial examples": "adversarial example",
        "foundation models": "foundation model",

        # Add acronym to term
        "application-specific integrated circuit": "application-specific integrated circuit (asic)",
        "field-programmable gate array": "field-programmable gate array (fpga)",
        "graphics processing unit": "graphics processing unit (gpu)",
        "neural processing unit": "neural processing unit (npu)",
        "tensor processing unit": "tensor processing unit (tpu)",

        # Remove redundant prefixes
        "cloud machine learning": "machine learning",
        "edge machine learning": "machine learning",
        "ml benchmarking": "benchmarking",

        # Keep more specific term
        "autoregressive": "autoregressive model",
        "bandwidth": "memory bandwidth",
        "inference": "batch inference",
        "parallelism": "data parallelism",
        "versioning": "data versioning",
        "pruning": "dynamic pruning",
        "quantization": "dynamic quantization",
        "model theft": "exact model theft",

        # Remove duplicates with different capitalization
        "generative ai": "generative AI",
        "responsible ai": "responsible AI",

        # Fix inconsistent naming
        "co design": "co-design",
        "moores law": "Moore's law",
        "in context learning": "in-context learning",
        "zero shot learning": "zero-shot learning",
        "test time compute": "test-time compute",
        "self attention": "self-attention",
        "text to image": "text-to-image",
        "fine tuning": "fine-tuning",
        "data centric ai": "data-centric AI",
        "generative ai": "generative AI",
        "support vector machines": "support vector machine",
        "uci machine learning repository": "UCI machine learning repository",
        "principal component analysis": "principal component analysis (PCA)",
        "variational autoencoder": "variational autoencoder (VAE)",
        "large language model": "large language model (LLM)",
        "neural language model": "language model",
        "parameter efficient finetuning": "parameter-efficient fine-tuning (PEFT)",
        "retrieval augmented generation": "retrieval-augmented generation (RAG)",
        "sequence to sequence": "sequence-to-sequence",
    }

def consolidate_chapter_glossaries():
    """Consolidate similar terms in all chapter glossary files."""
    chapters_dir = Path('quarto/contents/core')
    consolidation_rules = find_consolidation_rules()
    total_consolidated = 0

    for json_file in chapters_dir.glob('*/*_glossary.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)

        original_count = len(data['terms'])
        consolidated_terms = {}
        terms_to_remove = set()

        # First pass: apply consolidation rules
        for term_entry in data['terms']:
            term = term_entry['term']

            # Check if this term should be consolidated
            if term in consolidation_rules:
                canonical_term = consolidation_rules[term]
                print(f"{json_file.parent.name}: Consolidating '{term}' → '{canonical_term}'")
                term_entry['term'] = canonical_term
                term = canonical_term

            # Check for duplicates after consolidation
            if term in consolidated_terms:
                # Merge definitions if they're different
                existing = consolidated_terms[term]
                if existing['definition'] != term_entry['definition']:
                    # Keep the longer definition
                    if len(term_entry['definition']) > len(existing['definition']):
                        consolidated_terms[term] = term_entry
                    print(f"{json_file.parent.name}: Duplicate '{term}' found, keeping better definition")
            else:
                consolidated_terms[term] = term_entry

        # Update the terms list
        data['terms'] = list(consolidated_terms.values())

        if len(data['terms']) < original_count:
            consolidated_count = original_count - len(data['terms'])
            total_consolidated += consolidated_count
            print(f"{json_file.parent.name}: Consolidated {consolidated_count} terms ({original_count} → {len(data['terms'])})")

            # Save the updated file
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)

    return total_consolidated

def main():
    """Main function to consolidate similar terms."""
    print("🔧 Consolidating Similar Terms in Glossaries")
    print("=" * 60)

    total = consolidate_chapter_glossaries()

    print(f"\n✅ Consolidation complete!")
    print(f"  → Total terms consolidated: {total}")
    print("\nNext steps:")
    print("  1. Run build_global_glossary.py to rebuild the master")
    print("  2. Run generate_glossary.py to regenerate the glossary page")

if __name__ == "__main__":
    main()
