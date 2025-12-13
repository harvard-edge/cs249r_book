#!/usr/bin/env python3
"""
Fix cross-reference issues:
1. Remove bold markdown from explanations
2. Consolidate multiple refs to same chapter
3. Generate section-specific explanations
"""

import json
import requests
from pathlib import Path
from collections import defaultdict

def remove_bold_markdown(text):
    """Remove ** markdown from text"""
    return text.replace('**', '').strip()

def get_section_context(section_id):
    """Extract context from section ID"""
    contexts = {
        'ai-pervasiveness': 'ubiquitous AI applications',
        'ai-ml-basics': 'fundamental concepts',
        'ai-evolution': 'historical development',
        'ml-systems-engineering': 'engineering practices',
        'defining-ml-systems': 'system definitions',
        'lifecycle': 'development lifecycle',
        'ml-systems-wild': 'real-world systems',
        'impact-lifecycle': 'system impact',
        'practical-applications': 'applications',
        'challenges': 'system challenges',
        'looking-ahead': 'future directions',
        'book-structure': 'learning path'
    }

    for key, context in contexts.items():
        if key in section_id:
            return context
    return 'general concepts'

def generate_context_aware_explanation(source_section, target_chapter, connection_type, section_context):
    """Generate explanation specific to section context"""

    templates = {
        ('ml_systems', 'foundation'): {
            'ubiquitous AI applications': 'Systems architecture enabling widespread AI deployment',
            'fundamental concepts': 'Core systems principles for ML implementation',
            'engineering practices': 'Systems engineering foundations for ML',
            'system definitions': 'Concrete systems implementation examples',
            'development lifecycle': 'Systems perspective on ML lifecycle',
            'real-world systems': 'Production systems architecture patterns',
            'system challenges': 'Systems solutions to ML challenges',
            'future directions': 'Next-generation ML systems design',
            'default': 'ML systems foundations and architecture'
        },
        ('dl_primer', 'prerequisite'): {
            'fundamental concepts': 'Neural network basics for understanding systems',
            'historical development': 'Evolution from classical to deep learning',
            'engineering practices': 'DL concepts essential for engineering',
            'practical applications': 'Deep learning in production applications',
            'system challenges': 'DL-specific computational challenges',
            'default': 'Deep learning fundamentals'
        },
        ('frameworks', 'extends'): {
            'engineering practices': 'Framework selection and deployment strategies',
            'practical applications': 'Framework-specific implementation patterns',
            'real-world systems': 'Production framework considerations',
            'development lifecycle': 'Framework integration in ML pipeline',
            'default': 'ML frameworks for production'
        },
        ('training', 'extends'): {
            'fundamental concepts': 'Scaling training from concepts to systems',
            'engineering practices': 'Distributed training engineering',
            'system challenges': 'Training infrastructure optimization',
            'development lifecycle': 'Training pipeline automation',
            'default': 'Training systems and optimization'
        },
        ('workflow', 'complements'): {
            'development lifecycle': 'End-to-end ML workflow patterns',
            'engineering practices': 'Workflow automation and CI/CD',
            'practical applications': 'Production workflow examples',
            'system challenges': 'Workflow debugging and monitoring',
            'default': 'ML workflow and pipeline design'
        }
    }

    template_key = (target_chapter, connection_type)
    if template_key in templates:
        return templates[template_key].get(section_context, templates[template_key]['default'])

    return f"{connection_type.title()} connection to {target_chapter}"

def consolidate_chapter_refs(refs):
    """Consolidate multiple refs to same chapter"""
    by_chapter = defaultdict(list)

    for ref in refs:
        chapter = ref.get('target_chapter')
        by_chapter[chapter].append(ref)

    consolidated = []
    for chapter, chapter_refs in by_chapter.items():
        if len(chapter_refs) == 1:
            consolidated.append(chapter_refs[0])
        else:
            # Merge multiple refs to same chapter
            sections = []
            for r in chapter_refs:
                section = r.get('target_section', '').replace('sec-', '').split('-')[0]
                if section:
                    sections.append(section)

            # Take the highest priority ref as base
            base_ref = min(chapter_refs, key=lambda x: x.get('priority', 99))

            # Update with consolidated info
            if len(sections) > 1:
                base_ref['multiple_sections'] = True
                base_ref['section_count'] = len(sections)

            consolidated.append(base_ref)

    return consolidated

def fix_chapter_xrefs(chapter_path):
    """Fix all issues in a chapter's xrefs"""

    chapter_name = chapter_path.parent.name
    print(f"\n📖 Fixing {chapter_name}...")

    with open(chapter_path, 'r') as f:
        data = json.load(f)

    fixed_count = 0

    for section_id, refs in data.get('cross_references', {}).items():
        section_context = get_section_context(section_id)

        # First, remove bold markdown from all explanations
        for ref in refs:
            if 'explanation' in ref:
                ref['explanation'] = remove_bold_markdown(ref['explanation'])

        # Consolidate refs to same chapter
        consolidated_refs = consolidate_chapter_refs(refs)

        # Generate section-specific explanations
        for ref in consolidated_refs:
            old_explanation = ref.get('explanation', '')

            # Generate new context-aware explanation
            new_explanation = generate_context_aware_explanation(
                section_id,
                ref.get('target_chapter'),
                ref.get('connection_type'),
                section_context
            )

            # Only update if it's generic or repetitive
            if any(generic in old_explanation for generic in [
                'Faster, cheaper', 'Tracks progress', 'Workflows reveal',
                'Standardization, automation', 'Efficient ML inference'
            ]):
                ref['explanation'] = new_explanation
                fixed_count += 1

        # Update with consolidated refs
        data['cross_references'][section_id] = consolidated_refs

    # Save fixed version
    with open(chapter_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  ✅ Fixed {fixed_count} explanations")
    return fixed_count

def main():
    print("🔧 Fixing Cross-Reference Issues")
    print("=" * 50)

    base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")

    # Fix all chapters
    total_fixed = 0
    for chapter_path in base_dir.glob("*/*_xrefs.json"):
        total_fixed += fix_chapter_xrefs(chapter_path)

    print(f"\n✅ Total: Fixed {total_fixed} issues!")
    print("\nImprovements made:")
    print("  • Removed bold markdown from explanations")
    print("  • Generated section-specific explanations")
    print("  • Consolidated multiple refs to same chapter")

if __name__ == "__main__":
    main()
