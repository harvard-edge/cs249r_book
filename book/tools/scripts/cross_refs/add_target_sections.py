#!/usr/bin/env python3
"""
Add target_section references to existing xref JSON files.
Maps each target chapter to its overview section.
"""

import json
from pathlib import Path

def add_target_sections(xref_path: Path):
    """Add target_section to all cross-references"""

    with open(xref_path) as f:
        data = json.load(f)

    updates = 0

    for section_id, refs in data.get('cross_references', {}).items():
        for ref in refs:
            target_chapter = ref.get('target_chapter')

            # Only add if missing
            if target_chapter and not ref.get('target_section'):
                # Standard section ID pattern for overview sections
                # Most chapters have an overview section with this pattern
                ref['target_section'] = f"sec-{target_chapter.replace('_', '-')}-overview"
                updates += 1

    return data, updates

def main():
    base_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")

    # Process introduction first
    intro_xrefs = base_path / "introduction/introduction_xrefs.json"

    if intro_xrefs.exists():
        print(f"Adding target sections to {intro_xrefs.name}")
        data, count = add_target_sections(intro_xrefs)

        with open(intro_xrefs, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Added {count} target sections")

        # Show a sample
        print("\nSample references with sections:")
        shown = 0
        for section_id, refs in data['cross_references'].items():
            for ref in refs:
                if shown < 3:
                    print(f"  {ref['target_chapter']} → {ref.get('target_section', 'missing')}")
                    shown += 1

    else:
        print(f"File not found: {intro_xrefs}")

if __name__ == "__main__":
    main()
