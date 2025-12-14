#!/usr/bin/env python3
"""
Generate context-aware explanations that explain WHY a connection matters
for the specific section content.
"""

import json
import re
import requests
from pathlib import Path
from typing import Dict, List

def extract_section_content(qmd_path: Path, section_id: str) -> Dict:
    """Extract section title and content from QMD file"""

    with open(qmd_path) as f:
        content = f.read()

    # Find the section
    pattern = rf'##.*{{#{section_id}}}.*?(?=\n##|\Z)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return {"title": "", "content": ""}

    section_text = match.group(0)

    # Extract title
    title_match = re.match(r'##\s*([^{]+)', section_text)
    title = title_match.group(1).strip() if title_match else ""

    # Get first 500 chars of content for context
    lines = section_text.split('\n')[1:20]  # Skip title, get first 20 lines
    content_preview = '\n'.join(lines)

    return {
        "title": title,
        "content": content_preview[:500]
    }

def generate_contextual_explanation(
    source_section: Dict,
    target_chapter: str,
    target_section_id: str,
    connection_type: str,
    model: str = "gemma2:27b"
) -> str:
    """Generate explanation based on actual section content"""

    prompt = f"""
    Write a brief cross-reference note for a textbook margin. This will appear next to the source section
    to guide readers to related material.

    SOURCE SECTION: {source_section['title']}
    TARGET CHAPTER: {target_chapter}
    CONNECTION TYPE: {connection_type}

    Requirements:
    1. Maximum 8-10 words (strict limit for margin notes)
    2. Focus on WHAT the reader will find, not generic actions
    3. Be specific and practical
    4. Avoid starting with: Discover, Learn, Explore, See
    5. Use noun phrases when possible (e.g., "Distributed training for these architectures")

    Good examples based on connection type:
    - foundation: "Core algorithms behind these techniques"
    - extends: "Advanced optimization strategies for production"
    - complements: "Alternative approaches using streaming data"
    - prerequisite: "Mathematical foundations for these methods"
    - applies: "Production deployment of these models"

    Bad examples (too generic):
    - "Discover practical insights"
    - "Learn foundational concepts"
    - "Explore related topics"

    Return only the 8-10 word explanation, nothing else.
    """

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 100
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=data, timeout=30)
        if response.status_code == 200:
            explanation = response.json()['response'].strip()
            # Clean up
            explanation = explanation.rstrip('.')
            return explanation
    except Exception as e:
        print(f"Error generating explanation: {e}")

    # Fallback
    return f"Explore {connection_type} concepts in {target_chapter}"

def regenerate_explanations_with_context(xref_path: Path):
    """Regenerate all explanations using actual section content"""

    with open(xref_path) as f:
        data = json.load(f)

    chapter_name = xref_path.stem.replace('_xrefs', '')
    chapter_path = xref_path.parent / f"{chapter_name}.qmd"

    if not chapter_path.exists():
        print(f"Chapter file not found: {chapter_path}")
        return data, 0

    improvements = 0

    for section_id, refs in data.get('cross_references', {}).items():
        # Get source section content
        source_section = extract_section_content(chapter_path, section_id)

        if not source_section['content']:
            continue

        print(f"\nProcessing section: {source_section['title'][:50]}...")

        for ref in refs:
            target_chapter = ref.get('target_chapter')
            target_section = ref.get('target_section', '')
            connection_type = ref.get('connection_type', 'related')

            # Generate new contextual explanation
            new_explanation = generate_contextual_explanation(
                source_section,
                target_chapter,
                target_section,
                connection_type
            )

            if new_explanation != ref.get('explanation'):
                print(f"  {target_chapter}: {new_explanation}")
                ref['explanation'] = new_explanation
                improvements += 1

    return data, improvements

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--chapter', default='introduction', help='Chapter to process')
    parser.add_argument('--all', action='store_true', help='Process all chapters')

    args = parser.parse_args()

    base_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")

    if args.all:
        for chapter_dir in base_path.iterdir():
            if chapter_dir.is_dir():
                xref_file = chapter_dir / f"{chapter_dir.name}_xrefs.json"
                if xref_file.exists():
                    print(f"\n{'='*60}")
                    print(f"Processing {chapter_dir.name}")
                    print('='*60)

                    data, count = regenerate_explanations_with_context(xref_file)

                    with open(xref_file, 'w') as f:
                        json.dump(data, f, indent=2)

                    print(f"\n✅ Improved {count} explanations")
    else:
        xref_file = base_path / args.chapter / f"{args.chapter}_xrefs.json"

        if xref_file.exists():
            print(f"Regenerating contextual explanations for {args.chapter}")
            print('='*60)

            data, count = regenerate_explanations_with_context(xref_file)

            with open(xref_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"\n✅ Improved {count} explanations")
        else:
            print(f"File not found: {xref_file}")

if __name__ == "__main__":
    main()
