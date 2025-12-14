#!/usr/bin/env python3
"""
Concept-Driven Cross-Reference Generator
========================================

Generates intelligent cross-references between textbook chapters using structured concept maps.
Creates *_xrefs.json files for each chapter that can be consumed by Quarto Lua filters.

Unlike text similarity approaches, this uses the structured concept maps (YAML files) to find:
- Exact concept matches between chapters
- Educational progressions (prerequisites → applications)
- Methodological connections (theory → implementation)
- Hierarchical relationships (primary → secondary concepts)

USAGE:
    python3 concept_xrefs_generator.py -d /path/to/chapters/
    python3 concept_xrefs_generator.py -d /path/to/chapters/ --max-refs 3 --min-overlap 2

OUTPUT:
    Creates *_xrefs.json files in each chapter directory:
    - introduction/introduction_xrefs.json
    - ml_systems/ml_systems_xrefs.json
    - training/training_xrefs.json
    etc.

CONCEPT RELATIONSHIPS:
    1. Exact Matches: Same technical terms across chapters
    2. Educational Flow: Prerequisites → Advanced topics
    3. Application Links: Theory chapters → Implementation chapters
    4. Methodology Connections: Techniques → Use cases

REQUIREMENTS:
    pip install pyyaml

The generated JSON files follow this structure:
{
  "cross_references": {
    "sec-training-distributed": [
      {
        "target_chapter": "hw_acceleration",
        "target_section": "sec-gpu-computing",
        "connection_type": "implementation",
        "concepts": ["Distributed Training", "GPU Computing"],
        "explanation": "Distributed training benefits from GPU acceleration",
        "strength": 0.87
      }
    ]
  }
}
"""

import json
import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

@dataclass
class ConceptMap:
    """Represents a chapter's concept map."""
    chapter: str
    file_path: str
    source_file: str
    generated_date: str
    primary_concepts: List[str]
    secondary_concepts: List[str]
    technical_terms: List[str]
    methodologies: List[str]
    applications: List[str]
    keywords: List[str]
    topics_covered: List[Dict]

@dataclass
class CrossReference:
    """Represents a cross-reference between sections."""
    target_chapter: str
    target_section: str
    connection_type: str
    concepts: List[str]
    explanation: str
    strength: float

class ConceptXRefGenerator:
    def __init__(self, chapters_dir: str, max_refs: int = 5, min_overlap: int = 2):
        self.chapters_dir = Path(chapters_dir)
        self.max_refs = max_refs
        self.min_overlap = min_overlap
        self.concept_maps: Dict[str, ConceptMap] = {}
        self.chapter_order = []

    def load_concept_maps(self) -> None:
        """Load all concept maps from chapter directories."""
        print("🔍 Loading concept maps...")

        for chapter_dir in sorted(self.chapters_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            chapter_name = chapter_dir.name
            concept_file = chapter_dir / f"{chapter_name}_concepts.yml"

            if not concept_file.exists():
                print(f"⚠️  No concept file found: {concept_file}")
                continue

            try:
                with open(concept_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                if 'concept_map' not in data:
                    print(f"⚠️  Invalid concept file format: {concept_file}")
                    continue

                concept_map_data = data['concept_map']

                concept_map = ConceptMap(
                    chapter=chapter_name,
                    file_path=str(concept_file),
                    source_file=concept_map_data.get('source', f'{chapter_name}.qmd'),
                    generated_date=concept_map_data.get('generated_date', ''),
                    primary_concepts=concept_map_data.get('primary_concepts', []),
                    secondary_concepts=concept_map_data.get('secondary_concepts', []),
                    technical_terms=concept_map_data.get('technical_terms', []),
                    methodologies=concept_map_data.get('methodologies', []),
                    applications=concept_map_data.get('applications', []),
                    keywords=concept_map_data.get('keywords', []),
                    topics_covered=concept_map_data.get('topics_covered', [])
                )

                self.concept_maps[chapter_name] = concept_map
                self.chapter_order.append(chapter_name)
                print(f"  ✅ {chapter_name} ({len(concept_map.primary_concepts)} primary concepts)")

            except Exception as e:
                print(f"❌ Error loading {concept_file}: {e}")

        print(f"📚 Loaded {len(self.concept_maps)} concept maps")

    def find_concept_overlaps(self, source_chapter: str, target_chapter: str) -> Tuple[List[str], float]:
        """Find overlapping concepts between two chapters with fuzzy matching."""
        source_map = self.concept_maps[source_chapter]
        target_map = self.concept_maps[target_chapter]

        # Collect all concepts from each chapter with normalization
        def normalize_concept(concept: str) -> str:
            """Normalize concepts for better matching."""
            return concept.lower().strip()

        def collect_concepts(concept_map: ConceptMap) -> Set[str]:
            concepts = set()
            concepts.update([normalize_concept(c) for c in concept_map.primary_concepts])
            concepts.update([normalize_concept(c) for c in concept_map.secondary_concepts])
            concepts.update([normalize_concept(c) for c in concept_map.technical_terms])
            concepts.update([normalize_concept(c) for c in concept_map.methodologies])
            concepts.update([normalize_concept(c) for c in concept_map.applications])
            concepts.update([normalize_concept(c) for c in concept_map.keywords])
            return concepts

        source_concepts = collect_concepts(source_map)
        target_concepts = collect_concepts(target_map)

        # Find exact overlaps
        exact_overlaps = source_concepts & target_concepts

        # Find fuzzy overlaps (substring matching)
        fuzzy_overlaps = set()
        for source_concept in source_concepts:
            if source_concept in exact_overlaps:
                continue
            for target_concept in target_concepts:
                if target_concept in exact_overlaps:
                    continue
                # Check for substring matches (both ways)
                if (len(source_concept) > 4 and source_concept in target_concept) or \
                   (len(target_concept) > 4 and target_concept in source_concept):
                    fuzzy_overlaps.add(f"{source_concept}~{target_concept}")

        # Combine overlaps
        all_overlaps = list(exact_overlaps) + [match.split('~')[0] for match in fuzzy_overlaps]

        # Calculate weighted strength
        # Exact matches get full weight, fuzzy matches get 0.7 weight
        exact_weight = len(exact_overlaps)
        fuzzy_weight = len(fuzzy_overlaps) * 0.7
        total_weight = exact_weight + fuzzy_weight

        # Use smaller set for denominator to boost strength
        min_concepts = min(len(source_concepts), len(target_concepts))
        strength = total_weight / min_concepts if min_concepts > 0 else 0.0

        return all_overlaps, strength

    def determine_connection_type(self, source_chapter: str, target_chapter: str, overlaps: List[str]) -> str:
        """Determine the type of connection between chapters."""
        source_map = self.concept_maps[source_chapter]
        target_map = self.concept_maps[target_chapter]

        # Educational progression patterns
        educational_flow = {
            'introduction': 'foundation',
            'dl_primer': 'foundation',
            'ml_systems': 'architecture',
            'dnn_architectures': 'architecture',
            'frameworks': 'tools',
            'training': 'implementation',
            'efficient_ai': 'optimization',
            'optimizations': 'optimization',
            'hw_acceleration': 'implementation',
            'benchmarking': 'evaluation',
            'ops': 'deployment',
            'ondevice_learning': 'specialization',
            'robust_ai': 'specialization',
            'privacy_security': 'specialization',
            'responsible_ai': 'ethics',
            'sustainable_ai': 'ethics',
            'ai_for_good': 'applications',
            'workflow': 'process',
            'emerging_topics': 'frontier',
            'generative_ai': 'frontier',
            'frontiers': 'frontier',
            'conclusion': 'summary'
        }

        source_type = educational_flow.get(source_chapter, 'general')
        target_type = educational_flow.get(target_chapter, 'general')

        # Determine connection type based on chapter types and overlaps
        if source_type == 'foundation' and target_type in ['architecture', 'implementation']:
            return 'prerequisite'
        elif source_type == 'architecture' and target_type == 'implementation':
            return 'implementation'
        elif source_type == 'implementation' and target_type == 'optimization':
            return 'enhancement'
        elif source_type in ['foundation', 'architecture'] and target_type == 'applications':
            return 'application'
        elif any(concept in source_map.methodologies for concept in overlaps):
            return 'methodological'
        elif any(concept in source_map.technical_terms for concept in overlaps):
            return 'technical'
        elif any(concept in source_map.applications for concept in overlaps):
            return 'practical'
        else:
            return 'conceptual'

    def generate_explanation(self, source_chapter: str, target_chapter: str,
                           connection_type: str, overlaps: List[str]) -> str:
        """Generate explanation for the cross-reference."""
        overlap_text = ", ".join(overlaps[:3])
        if len(overlaps) > 3:
            overlap_text += f" and {len(overlaps) - 3} more"

        explanations = {
            'prerequisite': f"Builds on foundational concepts: {overlap_text}",
            'implementation': f"Shows practical implementation of: {overlap_text}",
            'enhancement': f"Optimizes and enhances: {overlap_text}",
            'application': f"Demonstrates real-world applications of: {overlap_text}",
            'methodological': f"Shares methodological approaches: {overlap_text}",
            'technical': f"Uses related technical concepts: {overlap_text}",
            'practical': f"Provides practical examples of: {overlap_text}",
            'conceptual': f"Explores related concepts: {overlap_text}"
        }

        return explanations.get(connection_type, f"Related through: {overlap_text}")

    def get_section_mapping(self, chapter: str) -> Dict[str, str]:
        """Get section ID mapping for a chapter (simplified for now)."""
        # In a full implementation, this would parse the .qmd file to extract actual section IDs
        # For now, we'll use common patterns based on chapter topics
        section_patterns = {
            'introduction': ['sec-introduction-ai-pervasiveness', 'sec-introduction-ai-evolution'],
            'ml_systems': ['sec-ml-systems-overview', 'sec-ml-systems-cloud', 'sec-ml-systems-edge'],
            'training': ['sec-training-distributed', 'sec-training-optimization'],
            'hw_acceleration': ['sec-hw-acceleration-gpu-computing', 'sec-hw-acceleration-tpu'],
            'frameworks': ['sec-frameworks-tensorflow', 'sec-frameworks-pytorch'],
            # Add more as needed
        }

        sections = section_patterns.get(chapter, [f'sec-{chapter}-overview'])
        return {section: section for section in sections}

    def generate_cross_references(self) -> Dict[str, Dict]:
        """Generate cross-references for all chapters."""
        print("\n🔗 Generating cross-references...")

        all_xrefs = {}

        for source_chapter in self.chapter_order:
            chapter_xrefs = {}
            source_sections = self.get_section_mapping(source_chapter)

            print(f"  📝 Processing {source_chapter}...")

            # Find connections to other chapters
            connections = []

            for target_chapter in self.chapter_order:
                if source_chapter == target_chapter:
                    continue

                overlaps, strength = self.find_concept_overlaps(source_chapter, target_chapter)

                if len(overlaps) >= self.min_overlap and strength > 0.05:
                    connection_type = self.determine_connection_type(source_chapter, target_chapter, overlaps)
                    explanation = self.generate_explanation(source_chapter, target_chapter, connection_type, overlaps)
                    target_sections = self.get_section_mapping(target_chapter)

                    # Create cross-reference
                    xref = CrossReference(
                        target_chapter=target_chapter,
                        target_section=list(target_sections.keys())[0],  # Use first section for now
                        connection_type=connection_type,
                        concepts=overlaps[:5],  # Limit to top 5 concepts
                        explanation=explanation,
                        strength=strength
                    )

                    connections.append((xref, strength))

            # Sort by strength and take top connections
            connections.sort(key=lambda x: x[1], reverse=True)
            top_connections = connections[:self.max_refs]

            # Group by source sections (for now, assign to first section)
            if source_sections and top_connections:
                first_section = list(source_sections.keys())[0]
                chapter_xrefs[first_section] = []

                for xref, _ in top_connections:
                    chapter_xrefs[first_section].append({
                        'target_chapter': xref.target_chapter,
                        'target_section': xref.target_section,
                        'connection_type': xref.connection_type,
                        'concepts': xref.concepts,
                        'explanation': xref.explanation,
                        'strength': round(xref.strength, 3)
                    })

            all_xrefs[source_chapter] = {
                'cross_references': chapter_xrefs,
                'generated_date': '2025-01-12',
                'generator': 'concept_xrefs_generator.py',
                'total_connections': len([xref for xrefs in chapter_xrefs.values() for xref in xrefs])
            }

            print(f"    ✅ Generated {len([xref for xrefs in chapter_xrefs.values() for xref in xrefs])} connections")

        return all_xrefs

    def save_xref_files(self, all_xrefs: Dict[str, Dict]) -> None:
        """Save cross-reference files for each chapter."""
        print("\n💾 Saving cross-reference files...")

        for chapter, xrefs in all_xrefs.items():
            chapter_dir = self.chapters_dir / chapter
            xref_file = chapter_dir / f"{chapter}_xrefs.json"

            try:
                with open(xref_file, 'w', encoding='utf-8') as f:
                    json.dump(xrefs, f, indent=2, ensure_ascii=False)
                print(f"  ✅ {xref_file}")

            except Exception as e:
                print(f"  ❌ Failed to save {xref_file}: {e}")

    def run(self) -> None:
        """Run the complete cross-reference generation process."""
        print("🚀 Concept-Driven Cross-Reference Generator")
        print("=" * 50)

        self.load_concept_maps()
        all_xrefs = self.generate_cross_references()
        self.save_xref_files(all_xrefs)

        total_connections = sum(xref['total_connections'] for xref in all_xrefs.values())
        print(f"\n🎉 Generated {total_connections} cross-references across {len(all_xrefs)} chapters")
        print("✅ Cross-reference generation complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Generate concept-driven cross-references for ML Systems textbook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '-d', '--chapters-dir',
        required=True,
        help='Directory containing chapter folders with concept maps'
    )

    parser.add_argument(
        '--max-refs',
        type=int,
        default=5,
        help='Maximum cross-references per chapter (default: 5)'
    )

    parser.add_argument(
        '--min-overlap',
        type=int,
        default=2,
        help='Minimum concept overlap to create connection (default: 2)'
    )

    args = parser.parse_args()

    if not Path(args.chapters_dir).exists():
        print(f"❌ Chapters directory does not exist: {args.chapters_dir}")
        sys.exit(1)

    generator = ConceptXRefGenerator(
        args.chapters_dir,
        max_refs=args.max_refs,
        min_overlap=args.min_overlap
    )

    generator.run()

if __name__ == "__main__":
    main()
