#!/usr/bin/env python3
"""Quick enhancement of introduction chapter explanations"""

import json
import requests
from pathlib import Path

def enhance_intro():
    """Enhance just the introduction chapter as a test"""

    intro_file = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core/introduction/introduction_xrefs.json")

    with open(intro_file, 'r') as f:
        data = json.load(f)

    # Quick test with first few references
    count = 0
    for section_id, refs in data.get('cross_references', {}).items():
        for ref in refs[:2]:  # Just first 2 refs per section for testing
            if count >= 10:  # Limit to 10 total for quick test
                break

            old_explanation = ref.get('explanation', '')
            connection_type = ref.get('connection_type', 'extends')
            target = ref.get('target_chapter', '')
            concepts = ref.get('concepts', [])

            # Clean concepts
            clean_concepts = []
            for c in concepts[:2]:
                if '~' in c:
                    parts = c.split('~')
                    clean_concepts.append(parts[-1])
                else:
                    clean_concepts.append(c)

            # Simple, focused prompt
            prompt_map = {
                'foundation': f"Why does {target} provide foundation? Focus: {', '.join(clean_concepts)}. Answer in 10 words:",
                'prerequisite': f"Why is {target} essential first? Focus: {', '.join(clean_concepts)}. Answer in 10 words:",
                'extends': f"How does {target} extend this? Focus: {', '.join(clean_concepts)}. Answer in 10 words:",
                'complements': f"What perspective does {target} add? Focus: {', '.join(clean_concepts)}. Answer in 10 words:"
            }

            prompt = prompt_map.get(connection_type, prompt_map['extends'])

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "gemma2:27b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 20
                        }
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    new_explanation = response.json().get('response', '').strip()
                    new_explanation = new_explanation.replace('\n', ' ').strip()

                    if new_explanation and len(new_explanation) > 5:
                        ref['explanation'] = new_explanation
                        count += 1
                        print(f"✓ {connection_type}: {new_explanation[:50]}")

            except Exception as e:
                print(f"Error: {e}")
                break

        if count >= 10:
            break

    # Save updated version
    with open(intro_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Updated {count} explanations in introduction")

if __name__ == "__main__":
    enhance_intro()
