#!/usr/bin/env python3
"""
Improve cross-reference explanations to be more specific and actionable.
"""

import json
from pathlib import Path
import re

def improve_explanation(chapter, section_title, target_chapter, connection_type, explanation):
    """Generate more specific, actionable explanations"""

    # Map of improved explanation templates
    improvements = {
        # Introduction chapter improvements
        ("introduction", "benchmarking", "optimizes"):
            "Compare performance metrics for system design choices",
        ("introduction", "hw_acceleration", "optimizes"):
            "Hardware options for scaling ML workloads",
        ("introduction", "privacy_security", "considers"):
            "Data protection requirements in ML pipelines",
        ("introduction", "responsible_ai", "considers"):
            "Ethical design principles for ML systems",

        # Default improvements by connection type
        ("prerequisite",): "Core concepts needed before this section",
        ("foundation",): "Foundational principles applied here",
        ("extends",): "Advanced techniques building on these concepts",
        ("complements",): "Related approaches and alternatives",
        ("applies",): "Practical implementation patterns",
        ("optimizes",): "Performance optimization strategies",
        ("considers",): "Important considerations and constraints",
        ("explores",): "Deeper exploration of these topics",
        ("anticipates",): "Future directions and trends",
        ("specializes",): "Specialized applications and use cases"
    }

    # Look for specific improvement
    key = (chapter, target_chapter, connection_type)
    if key in improvements:
        return improvements[key]

    # Look for connection type default
    key = (connection_type,)
    if key in improvements:
        return improvements[key]

    # Make existing explanation more specific
    if "Production considerations" in explanation:
        return explanation.replace("Production considerations for", "Deploy").replace(".", "")
    if "Essential concepts" in explanation:
        return explanation.replace("Essential concepts for", "Learn").replace(".", "")
    if "Practical patterns" in explanation:
        return explanation.replace("Practical patterns for", "Implement").replace(".", "")
    if "Advanced techniques" in explanation:
        return explanation.replace("Advanced techniques in", "Master").replace(".", "")

    return explanation

def process_xrefs(file_path):
    """Process and improve explanations in xref file"""

    with open(file_path) as f:
        data = json.load(f)

    chapter_name = file_path.stem.replace('_xrefs', '')
    improvements_made = 0

    for section_id, refs in data.get('cross_references', {}).items():
        # Extract section title from ID
        section_title = section_id.replace('sec-', '').replace('-', ' ')

        for ref in refs:
            old_explanation = ref.get('explanation', '')
            new_explanation = improve_explanation(
                chapter_name,
                section_title,
                ref.get('target_chapter'),
                ref.get('connection_type'),
                old_explanation
            )

            if new_explanation != old_explanation:
                ref['explanation'] = new_explanation
                improvements_made += 1

    return data, improvements_made

def main():
    base_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")

    # Process introduction first as example
    intro_xrefs = base_path / "introduction/introduction_xrefs.json"

    if intro_xrefs.exists():
        print(f"Improving explanations in {intro_xrefs.name}")
        data, count = process_xrefs(intro_xrefs)

        with open(intro_xrefs, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Improved {count} explanations")
    else:
        print(f"File not found: {intro_xrefs}")

if __name__ == "__main__":
    main()
