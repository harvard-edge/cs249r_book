#!/usr/bin/env python3
"""
Add human-readable section titles to cross-references for better navigation.
"""

import json
import re
from pathlib import Path

def get_section_title_from_qmd(chapter_path: Path, section_id: str) -> str:
    """Extract section title from QMD file"""

    with open(chapter_path) as f:
        content = f.read()

    # Look for the section with this ID
    pattern = rf'##.*{{#{section_id}}}'
    match = re.search(pattern, content)

    if match:
        # Extract title, removing the {#id} part
        line = match.group(0)
        title = re.sub(r'\s*{#[^}]+}', '', line)
        title = title.replace('##', '').strip()
        # Clean up the title
        title = re.sub(r'^\d+\.\d+\s+', '', title)  # Remove section numbers like "1.2"
        return title

    return ""

def add_section_titles_to_xrefs(xref_path: Path):
    """Add section titles to all cross-references"""

    with open(xref_path) as f:
        data = json.load(f)

    base_dir = xref_path.parent.parent
    updates_made = 0

    for section_id, refs in data.get('cross_references', {}).items():
        for ref in refs:
            if ref.get('target_section') and not ref.get('target_section_title'):
                target_chapter = ref.get('target_chapter')
                if target_chapter:
                    # Find the target chapter's QMD file
                    chapter_file = base_dir / target_chapter / f"{target_chapter}.qmd"
                    if chapter_file.exists():
                        title = get_section_title_from_qmd(chapter_file, ref['target_section'])
                        if title:
                            ref['target_section_title'] = title
                            updates_made += 1

    return data, updates_made

def main():
    base_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")

    # Process introduction
    intro_xrefs = base_path / "introduction/introduction_xrefs.json"

    if intro_xrefs.exists():
        print(f"Adding section titles to {intro_xrefs.name}")
        data, count = add_section_titles_to_xrefs(intro_xrefs)

        with open(intro_xrefs, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Added {count} section titles")

        # Show a few examples
        print("\nExamples:")
        shown = 0
        for refs in data['cross_references'].values():
            for ref in refs:
                if ref.get('target_section_title') and shown < 3:
                    print(f"  {ref['target_chapter']} → {ref['target_section_title']}")
                    shown += 1

    else:
        print(f"File not found: {intro_xrefs}")

if __name__ == "__main__":
    main()
