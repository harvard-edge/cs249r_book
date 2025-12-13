#!/usr/bin/env python3
"""
Smart Glossary Consolidation using LLM-based similarity detection.

This script automates the process of finding and consolidating similar terms
in the glossary using intelligent similarity detection and LLM-based decisions.

Workflow:
1. Detect similar terms using multiple similarity metrics
2. Group potential duplicates for LLM review
3. Use LLM to decide which terms to merge and how
4. Apply consolidation decisions automatically
5. Generate clean master glossary
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher
from collections import defaultdict

def calculate_similarity(term1: str, term2: str) -> float:
    """Calculate similarity between two terms using multiple metrics."""
    # Normalize terms for comparison
    norm1 = normalize_for_comparison(term1)
    norm2 = normalize_for_comparison(term2)

    # Exact match after normalization
    if norm1 == norm2:
        return 1.0

    # Sequence similarity
    seq_sim = SequenceMatcher(None, norm1, norm2).ratio()

    # Check for subset relationships
    if norm1 in norm2 or norm2 in norm1:
        return 0.9

    # Check for word overlap
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if words1 and words2:
        word_overlap = len(words1 & words2) / len(words1 | words2)
        if word_overlap > 0.5:
            return max(seq_sim, word_overlap)

    return seq_sim

def normalize_for_comparison(term: str) -> str:
    """Normalize term for similarity comparison."""
    normalized = term.lower()
    # Remove common variations
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)  # Remove parentheses
    normalized = re.sub(r'[^\w\s]', '', normalized)       # Remove punctuation
    normalized = re.sub(r'\s+', ' ', normalized).strip()  # Normalize whitespace
    # Handle plurals
    if normalized.endswith('s') and not normalized.endswith('ss'):
        singular = normalized[:-1]
        return singular
    return normalized

def find_similar_terms(terms: List[Dict]) -> List[List[Dict]]:
    """Find groups of similar terms that might need consolidation."""
    similarity_threshold = 0.7
    groups = []
    processed = set()

    for i, term1 in enumerate(terms):
        if i in processed:
            continue

        current_group = [term1]
        processed.add(i)

        for j, term2 in enumerate(terms[i+1:], i+1):
            if j in processed:
                continue

            similarity = calculate_similarity(term1['term'], term2['term'])
            if similarity >= similarity_threshold:
                current_group.append(term2)
                processed.add(j)

        # Only include groups with multiple terms
        if len(current_group) > 1:
            groups.append(current_group)

    return groups

def generate_consolidation_prompt(term_group: List[Dict]) -> str:
    """Generate a prompt for LLM to decide on term consolidation."""
    terms_info = []
    for term in term_group:
        info = f"- '{term['term']}': {term['definition'][:100]}..."
        if term.get('chapter_source'):
            info += f" (from {term['chapter_source']})"
        terms_info.append(info)

    prompt = f"""I have found these potentially similar glossary terms that might need consolidation:

{chr(10).join(terms_info)}

Please analyze these terms and provide a JSON response with your consolidation decision:

{{
  "action": "merge|keep_separate",
  "reasoning": "brief explanation of your decision",
  "preferred_term": "the term name to keep if merging",
  "preferred_definition": "the best definition if merging",
  "appears_in": ["list", "of", "chapters", "if", "merging"]
}}

Guidelines:
- MERGE if terms refer to the same concept (e.g., "adversarial example" vs "adversarial examples")
- MERGE if one term is a clear subset/superset of another
- KEEP_SEPARATE if terms have meaningfully different definitions or contexts
- For merged terms, prefer the most comprehensive definition
- Use singular form for merged terms unless plural is more standard
- Include all source chapters in appears_in for merged terms

Respond with only the JSON, no other text."""

    return prompt

def apply_consolidation_decisions(decisions: List[Dict], original_terms: List[Dict]) -> List[Dict]:
    """Apply LLM consolidation decisions to the original terms."""
    consolidated_terms = []
    terms_to_remove = set()

    # Process merge decisions
    for decision in decisions:
        if decision['action'] == 'merge':
            # Find all terms in this merge group
            group_terms = decision.get('original_terms', [])
            for term in group_terms:
                terms_to_remove.add(term['term'])

            # Add the merged term
            merged_term = {
                'term': decision['preferred_term'],
                'definition': decision['preferred_definition'],
                'appears_in': decision['appears_in'],
                'chapter_source': decision['appears_in'][0] if decision['appears_in'] else '',
                'aliases': [],
                'see_also': []
            }
            consolidated_terms.append(merged_term)

    # Add remaining terms that weren't merged
    for term in original_terms:
        if term['term'] not in terms_to_remove:
            consolidated_terms.append(term)

    return consolidated_terms

def process_with_claude(term_group: List[Dict]) -> Dict:
    """Process a term group using Claude API for consolidation decision."""
    import anthropic

    # You would need to set your API key
    # client = anthropic.Anthropic(api_key="your-api-key")

    prompt = generate_consolidation_prompt(term_group)

    # For now, return a mock decision - you'd replace this with actual API call
    # message = client.messages.create(
    #     model="claude-3-sonnet-20240229",
    #     max_tokens=1000,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    #
    # response = message.content[0].text
    # return json.loads(response)

    # Mock decision for demonstration
    return {
        "action": "keep_separate",
        "reasoning": "Mock decision - would be replaced with actual Claude API call",
        "preferred_term": term_group[0]['term'],
        "preferred_definition": term_group[0]['definition'],
        "appears_in": [term_group[0].get('chapter_source', '')]
    }

def save_consolidation_log(decisions: List[Dict], output_path: Path):
    """Save consolidation decisions for review."""
    log_data = {
        'timestamp': 'generated_automatically',
        'total_decisions': len(decisions),
        'merge_count': len([d for d in decisions if d['action'] == 'merge']),
        'keep_separate_count': len([d for d in decisions if d['action'] == 'keep_separate']),
        'decisions': decisions
    }

    log_path = output_path.parent / 'consolidation_log.json'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"📋 Consolidation log saved: {log_path}")

def main():
    """Main function for smart glossary consolidation."""
    print("🔧 Smart Glossary Consolidation")
    print("=" * 50)

    # Load current master glossary
    project_root = Path(__file__).parent.parent.parent.parent
    master_path = project_root / "quarto/contents/data/global_glossary.json"

    print("📚 Loading current master glossary...")
    with open(master_path, 'r') as f:
        data = json.load(f)

    original_count = len(data['terms'])
    print(f"  → Found {original_count} terms")

    # Find similar terms
    print("🔍 Detecting similar terms...")
    similar_groups = find_similar_terms(data['terms'])

    if not similar_groups:
        print("✅ No similar terms found that need consolidation!")
        return

    print(f"📊 Found {len(similar_groups)} groups of similar terms:")
    for i, group in enumerate(similar_groups, 1):
        terms = [t['term'] for t in group]
        print(f"  {i:2d}. {terms}")

    print(f"\n🤖 This would require {len(similar_groups)} LLM calls to decide consolidations.")
    print("📝 Each group would be analyzed for:")
    print("  • Semantic similarity")
    print("  • Definition overlap")
    print("  • Context appropriateness")
    print("  • Standard glossary practices")

    print(f"\n🎯 Potential outcomes:")
    print(f"  • Merge similar terms (e.g., 'example' + 'examples' → 'example')")
    print(f"  • Keep distinct terms (e.g., 'training' vs 'training data')")
    print(f"  • Standardize definitions across chapters")
    print(f"  • Maintain chapter attribution")

    # For demonstration, show what the first prompt would look like
    if similar_groups:
        print(f"\n📋 Example prompt for group 1:")
        print("-" * 40)
        prompt = generate_consolidation_prompt(similar_groups[0])
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    print(f"\n💡 To implement:")
    print(f"  1. Add LLM API integration (OpenAI/Anthropic)")
    print(f"  2. Process each group with LLM decision")
    print(f"  3. Apply consolidation automatically")
    print(f"  4. Regenerate master glossary and QMD file")
    print(f"  5. Log all decisions for review")

if __name__ == "__main__":
    main()
