#!/usr/bin/env python3
"""
Enhance all cross-reference explanations systematically
"""

import json
import requests
from pathlib import Path
import time

def create_focused_explanation(source_chapter, target_chapter, connection_type, concepts):
    """Create a focused, meaningful explanation"""

    # Clean concepts - take the most specific part
    clean_concepts = []
    for c in concepts[:2]:  # Focus on top 2 concepts
        if '~' in c:
            parts = c.split('~')
            clean_concepts.append(parts[-1])
        else:
            clean_concepts.append(c)

    # Chapter display names
    chapter_names = {
        'ml_systems': 'ML Systems',
        'dl_primer': 'Deep Learning',
        'frameworks': 'Frameworks',
        'training': 'Training',
        'workflow': 'Workflow',
        'data_engineering': 'Data Engineering',
        'efficient_ai': 'Efficient AI',
        'optimizations': 'Optimizations',
        'hw_acceleration': 'Hardware',
        'benchmarking': 'Benchmarking',
        'ops': 'MLOps',
        'ondevice_learning': 'On-Device',
        'privacy_security': 'Privacy',
        'responsible_ai': 'Ethics',
        'sustainable_ai': 'Sustainability',
        'ai_for_good': 'AI4Good',
        'robust_ai': 'Robustness',
        'generative_ai': 'GenAI',
        'frontiers': 'Frontiers'
    }

    target_name = chapter_names.get(target_chapter, target_chapter)

    # Create very focused prompts
    if connection_type == 'foundation':
        prompt = f"{target_name} provides {clean_concepts[0]} foundation. Explain value in 8 words:"
    elif connection_type == 'prerequisite':
        prompt = f"{target_name} teaches essential {clean_concepts[0]}. Why needed first? 8 words:"
    elif connection_type == 'extends':
        prompt = f"{target_name} advances {clean_concepts[0]} further. How? 8 words:"
    elif connection_type == 'complements':
        prompt = f"{target_name} offers alternative {clean_concepts[0]} perspective. What? 8 words:"
    else:
        prompt = f"{target_name} connects via {clean_concepts[0]}. How? 8 words:"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma2:27b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 15
                }
            },
            timeout=10
        )

        if response.status_code == 200:
            explanation = response.json().get('response', '').strip()
            explanation = explanation.replace('\n', ' ').replace('  ', ' ').strip()

            # Clean up common prefixes
            explanation = explanation.replace('Explanation:', '').strip()
            explanation = explanation.replace('Answer:', '').strip()

            # Ensure reasonable length
            words = explanation.split()
            if len(words) > 12:
                explanation = ' '.join(words[:10]) + '...'

            if explanation and len(explanation) > 5:
                return explanation

    except:
        pass

    # Better fallbacks
    fallbacks = {
        'foundation': f"Core {clean_concepts[0]} concepts for building understanding",
        'prerequisite': f"Essential {clean_concepts[0]} knowledge required first",
        'extends': f"Advanced {clean_concepts[0]} implementations and techniques",
        'complements': f"Alternative approach to {clean_concepts[0]}",
        'applies': f"Real-world {clean_concepts[0]} applications"
    }

    return fallbacks.get(connection_type, f"Connected through {clean_concepts[0]}")

def enhance_chapter(chapter_path):
    """Enhance all explanations in a chapter"""

    chapter_name = chapter_path.parent.name
    print(f"\n📖 Processing {chapter_name}...")

    with open(chapter_path, 'r') as f:
        data = json.load(f)

    enhanced = 0
    total = 0

    for section_id, refs in data.get('cross_references', {}).items():
        for ref in refs:
            total += 1

            # Skip if already has a good explanation
            current = ref.get('explanation', '')
            if current and not any(bad in current for bad in [
                'Builds on foundational', 'Essential prerequisite',
                'Advanced extension', 'Complementary perspective',
                'Real-world applications'
            ]):
                continue

            # Generate better explanation
            new_explanation = create_focused_explanation(
                chapter_name,
                ref.get('target_chapter', ''),
                ref.get('connection_type', ''),
                ref.get('concepts', [])
            )

            if new_explanation != current:
                ref['explanation'] = new_explanation
                enhanced += 1

                if enhanced % 10 == 0:
                    print(f"  ✨ Enhanced {enhanced} refs...")

    # Save
    with open(chapter_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  ✅ Enhanced {enhanced}/{total} explanations")
    return enhanced

def main():
    print("🚀 Enhancing All Cross-Reference Explanations")
    print("=" * 50)

    base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")

    # Process all chapters
    chapters_to_process = [
        'introduction', 'ml_systems', 'dl_primer', 'workflow',
        'data_engineering', 'frameworks', 'training', 'efficient_ai',
        'optimizations', 'hw_acceleration', 'benchmarking', 'ops',
        'ondevice_learning', 'privacy_security', 'responsible_ai',
        'sustainable_ai', 'ai_for_good', 'robust_ai', 'generative_ai',
        'frontiers', 'emerging_topics', 'conclusion'
    ]

    total_enhanced = 0

    for chapter in chapters_to_process:
        xref_file = base_dir / chapter / f"{chapter}_xrefs.json"
        if xref_file.exists():
            total_enhanced += enhance_chapter(xref_file)
            time.sleep(0.5)  # Be nice to Ollama

    print(f"\n✅ Total: Enhanced {total_enhanced} explanations across all chapters!")
    print("📝 Explanations are now concise and meaningful!")

if __name__ == "__main__":
    main()
