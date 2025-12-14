#!/usr/bin/env python3
"""
Generate academic-style cross-reference explanations for textbook margins.
Focuses on WHAT readers will find, not marketing language.
"""

import json
import re
import requests
from pathlib import Path
from typing import Dict

def generate_academic_explanation(
    source_section_title: str,
    target_chapter: str,
    connection_type: str,
    model: str = "gemma2:27b"
) -> str:
    """Generate concise, academic explanations for margin notes"""

    # Map connection types to their purpose
    connection_logic = {
        "prerequisite": "essential background knowledge",
        "foundation": "core concepts being applied",
        "extends": "advanced techniques building on this",
        "complements": "alternative approaches to consider",
        "applies": "practical implementation methods",
        "optimizes": "performance improvement strategies",
        "considers": "important constraints and tradeoffs",
        "explores": "deeper theoretical treatment",
        "anticipates": "future directions and trends",
        "specializes": "specific use cases and variants"
    }

    prompt = f"""
    Write a brief margin note for an academic ML textbook. The reader is in the "{source_section_title}" section
    and we're pointing them to the {target_chapter} chapter for {connection_logic.get(connection_type, 'related material')}.

    STRICT REQUIREMENTS:
    1. Maximum 8-10 words
    2. Use noun phrases or technical descriptions
    3. Be specific to ML/systems content
    4. Academic tone (no marketing language)
    5. Never start with: Discover, Learn, Explore, See, Find

    GOOD EXAMPLES by connection type:

    prerequisite:
    - "Mathematical foundations for gradient descent"
    - "Core algorithms and complexity analysis"
    - "Required background in probability theory"

    foundation:
    - "System architecture patterns discussed here"
    - "Core ML pipeline components"
    - "Fundamental training algorithms"

    extends:
    - "Distributed training for large models"
    - "Advanced optimization techniques"
    - "Scaling strategies for production"

    complements:
    - "Alternative approach using streaming data"
    - "Graph-based methods for same problem"
    - "Cloud-native architecture patterns"

    applies:
    - "Production deployment strategies"
    - "Real-world implementation examples"
    - "Industry best practices"

    optimizes:
    - "Performance tuning for these models"
    - "Hardware acceleration techniques"
    - "Memory optimization strategies"

    BAD EXAMPLES (avoid these):
    - "Discover how to implement..."
    - "Learn about advanced..."
    - "Explore the possibilities..."
    - "Essential concepts for understanding..."
    - "Practical insights into..."

    Source section: {source_section_title}
    Target chapter: {target_chapter}
    Connection type: {connection_type}

    Generate only the 8-10 word explanation:
    """

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 30
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=data, timeout=30)
        if response.status_code == 200:
            explanation = response.json()['response'].strip()
            # Remove any quotes if the model added them
            explanation = explanation.strip('"\'')
            # Ensure no marketing words snuck in
            marketing_words = ['Discover', 'Learn', 'Explore', 'Find', 'See how']
            for word in marketing_words:
                if explanation.startswith(word):
                    # Generate a fallback
                    return get_fallback_explanation(target_chapter, connection_type)
            return explanation
    except Exception as e:
        print(f"Error: {e}")

    return get_fallback_explanation(target_chapter, connection_type)

def get_fallback_explanation(target_chapter: str, connection_type: str) -> str:
    """Fallback explanations that are academic and specific"""

    chapter_topics = {
        "ml_systems": "System architecture and design patterns",
        "dl_primer": "Neural network fundamentals",
        "workflow": "ML pipeline orchestration",
        "data_engineering": "Data pipeline architecture",
        "frameworks": "Framework selection and deployment",
        "training": "Training algorithms and optimization",
        "efficient_ai": "Model compression techniques",
        "optimizations": "Performance optimization strategies",
        "hw_acceleration": "GPU and TPU acceleration",
        "benchmarking": "Performance evaluation metrics",
        "ops": "Production deployment and monitoring",
        "ondevice_learning": "Edge computing for ML",
        "privacy_security": "Privacy-preserving ML techniques",
        "responsible_ai": "Ethical AI considerations",
        "robust_ai": "Adversarial robustness methods",
        "generative_ai": "Generative model architectures",
        "sustainable_ai": "Energy-efficient ML systems",
        "ai_for_good": "Social impact applications",
        "frontiers": "Emerging research directions",
        "emerging_topics": "Latest developments in ML"
    }

    fallbacks_by_type = {
        "prerequisite": {
            "default": "Essential background concepts"
        },
        "foundation": {
            "default": "Core principles applied here"
        },
        "extends": {
            "default": "Advanced techniques for scale"
        },
        "complements": {
            "default": "Alternative implementation approaches"
        },
        "applies": {
            "default": "Production deployment patterns"
        },
        "optimizes": {
            "default": "Performance optimization methods"
        }
    }

    # Try to get chapter-specific topic
    topic = chapter_topics.get(target_chapter, "Related techniques")

    # Get connection-specific template
    template = fallbacks_by_type.get(connection_type, {}).get("default", topic)

    # Ensure it's short
    words = template.split()
    if len(words) > 10:
        template = ' '.join(words[:8])

    return template

def regenerate_all_explanations(xref_path: Path):
    """Regenerate all explanations with academic style"""

    with open(xref_path) as f:
        data = json.load(f)

    chapter_name = xref_path.stem.replace('_xrefs', '')
    improvements = 0

    print(f"Processing {chapter_name}...")

    for section_id, refs in data.get('cross_references', {}).items():
        # Extract clean section title
        section_title = section_id.replace('sec-', '').replace('-', ' ')
        section_title = section_title.split()[1] if section_title.split() else "introduction"

        for ref in refs:
            old_explanation = ref.get('explanation', '')

            # Skip if already good (no marketing words)
            if not any(word in old_explanation for word in ['Discover', 'Learn', 'Explore', 'practical insights']):
                continue

            # Generate new academic explanation
            new_explanation = generate_academic_explanation(
                section_title,
                ref.get('target_chapter'),
                ref.get('connection_type', 'related')
            )

            if new_explanation and new_explanation != old_explanation:
                print(f"  {ref['target_chapter']}: {new_explanation}")
                ref['explanation'] = new_explanation
                improvements += 1

    return data, improvements

def main():
    base_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
    intro_xrefs = base_path / "introduction/introduction_xrefs.json"

    if intro_xrefs.exists():
        print("="*60)
        print("Regenerating with academic style (no marketing language)")
        print("="*60)

        data, count = regenerate_all_explanations(intro_xrefs)

        with open(intro_xrefs, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Fixed {count} explanations")
        print("\nExplanation style:")
        print("  • Noun phrases (e.g., 'Distributed training techniques')")
        print("  • Technical descriptions (e.g., 'GPU memory optimization')")
        print("  • No action verbs or marketing language")
    else:
        print(f"File not found: {intro_xrefs}")

if __name__ == "__main__":
    main()
