#!/usr/bin/env python3
"""
Enhance cross-reference explanations using Gemma 2 27B
Replaces template-based explanations with meaningful, context-aware ones
"""

import json
import requests
from pathlib import Path
import time
import sys

def generate_better_explanation(source_chapter: str, target_chapter: str,
                               connection_type: str, concepts: list) -> str:
    """Generate a meaningful explanation using Gemma 2 27B"""

    # Format chapter names nicely
    chapter_display_names = {
        'introduction': 'Introduction',
        'ml_systems': 'ML Systems',
        'dl_primer': 'Deep Learning Primer',
        'workflow': 'ML Workflow',
        'data_engineering': 'Data Engineering',
        'frameworks': 'ML Frameworks',
        'training': 'Training Systems',
        'efficient_ai': 'Efficient AI',
        'optimizations': 'Model Optimizations',
        'hw_acceleration': 'Hardware Acceleration',
        'benchmarking': 'Benchmarking',
        'ondevice_learning': 'On-Device Learning',
        'ops': 'ML Operations',
        'privacy_security': 'Privacy & Security',
        'responsible_ai': 'Responsible AI',
        'sustainable_ai': 'Sustainable AI',
        'ai_for_good': 'AI for Good',
        'robust_ai': 'Robust AI',
        'generative_ai': 'Generative AI',
        'frontiers': 'Research Frontiers',
        'emerging_topics': 'Emerging Topics',
        'conclusion': 'Conclusion'
    }

    source_name = chapter_display_names.get(source_chapter, source_chapter)
    target_name = chapter_display_names.get(target_chapter, target_chapter)

    # Extract key concepts (clean up the ~ notation)
    clean_concepts = []
    for c in concepts[:3]:  # Limit to 3 most important
        if '~' in c:
            # Take the more specific part after ~
            parts = c.split('~')
            clean_concepts.append(parts[1] if len(parts) > 1 else parts[0])
        else:
            clean_concepts.append(c)

    # Create context-aware prompts based on connection type
    prompts = {
        'foundation': f"""A student reading {source_name} needs foundation from {target_name}.
Key concepts: {', '.join(clean_concepts)}
Write a 10-15 word explanation of why this foundation helps understanding.
Focus on the learning benefit, not just listing concepts.""",

        'prerequisite': f"""Before reading {source_name}, students need {target_name}.
Key concepts: {', '.join(clean_concepts)}
Write a 10-15 word explanation of what essential knowledge this provides.
Be specific about why it's a prerequisite.""",

        'extends': f"""{target_name} extends concepts from {source_name}.
Key concepts: {', '.join(clean_concepts)}
Write a 10-15 word explanation of how these concepts advance further.
Focus on the progression, not repetition.""",

        'complements': f"""{target_name} complements {source_name} with different perspective.
Key concepts: {', '.join(clean_concepts)}
Write a 10-15 word explanation of what additional insights this provides.
Emphasize the complementary value.""",

        'applies': f"""{target_name} shows applications of {source_name} concepts.
Key concepts: {', '.join(clean_concepts)}
Write a 10-15 word explanation of real-world usage.
Focus on practical application."""
    }

    prompt = prompts.get(connection_type, prompts['complements'])

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma2:27b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 30
                }
            },
            timeout=15
        )

        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            # Clean up the response
            result = result.replace('\n', ' ').replace('  ', ' ')
            # Remove any leading "Explanation:" or similar
            result = result.replace('Explanation:', '').strip()

            # Ensure reasonable length
            words = result.split()
            if len(words) > 20:
                result = ' '.join(words[:17]) + "..."

            if result and len(result) > 10:
                return result

    except Exception as e:
        print(f"  ⚠️  LLM error: {e}")

    # Fallback to slightly better templates than the original
    fallbacks = {
        'foundation': f"Provides {clean_concepts[0]} foundation for advanced topics",
        'prerequisite': f"Essential {clean_concepts[0]} knowledge required first",
        'extends': f"Advances to {clean_concepts[0]} implementations",
        'complements': f"Alternative view on {clean_concepts[0]}",
        'applies': f"Practical {clean_concepts[0]} applications"
    }

    return fallbacks.get(connection_type, f"Connected through {clean_concepts[0]}")

def enhance_chapter_xrefs(chapter_path: Path):
    """Enhance explanations in a single chapter's xrefs file"""

    if not chapter_path.exists():
        return False

    chapter_name = chapter_path.parent.name
    print(f"\n📖 Enhancing {chapter_name}...")

    with open(chapter_path, 'r') as f:
        data = json.load(f)

    enhanced_count = 0
    total_refs = 0

    for section_id, refs in data.get('cross_references', {}).items():
        for ref in refs:
            total_refs += 1

            # Skip if explanation seems good already
            current_explanation = ref.get('explanation', '')
            if current_explanation and not any(x in current_explanation for x in
                ['Builds on foundational', 'Essential prerequisite', 'Advanced extension',
                 'Complementary perspective', 'Real-world applications']):
                continue

            # Generate better explanation
            new_explanation = generate_better_explanation(
                chapter_name,
                ref.get('target_chapter', ''),
                ref.get('connection_type', 'complements'),
                ref.get('concepts', [])
            )

            if new_explanation != current_explanation:
                ref['explanation'] = new_explanation
                enhanced_count += 1

                # Show progress
                if enhanced_count % 5 == 0:
                    print(f"  ✨ Enhanced {enhanced_count} explanations...")

    # Save enhanced version
    with open(chapter_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  ✅ Enhanced {enhanced_count}/{total_refs} explanations")
    return True

def main():
    print("🚀 Cross-Reference Explanation Enhancer")
    print("=" * 50)

    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            if 'gemma2:27b' not in models:
                print("❌ gemma2:27b not available in Ollama")
                print("   Run: ollama pull gemma2:27b")
                return 1
            print("✅ Ollama running with gemma2:27b")
    except:
        print("❌ Ollama is not running")
        print("   Please start Ollama first")
        return 1

    # Find all xrefs files
    base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
    xref_files = list(base_dir.glob("**/*_xrefs.json"))

    print(f"\n📚 Found {len(xref_files)} chapters to enhance")

    # Process each chapter
    success_count = 0
    for xref_file in sorted(xref_files):
        if enhance_chapter_xrefs(xref_file):
            success_count += 1
            time.sleep(0.5)  # Be nice to Ollama

    print(f"\n✅ Successfully enhanced {success_count}/{len(xref_files)} chapters")
    print("\n📝 Explanations are now more meaningful and context-aware!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
