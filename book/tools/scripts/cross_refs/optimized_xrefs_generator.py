#!/usr/bin/env python3
"""
Optimized Concept-Driven Cross-Reference Generator
==================================================

Final optimized version that combines:
- Better concept matching with weighted categories
- Improved connection type classification
- Optional LLM enhancement for explanations
- Balanced coverage across all chapters
- Educational progression awareness

USAGE:
    python3 optimized_xrefs_generator.py -d /path/to/chapters/ [--llm] [--model llama3.1:8b]
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

# Import LLM enhancer if available
try:
    from llm_enhanced_xrefs import LLMEnhancer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

@dataclass
class ConceptMap:
    """Represents a chapter's concept map with weighted categories."""
    chapter: str
    primary_concepts: List[str]
    secondary_concepts: List[str]
    technical_terms: List[str]
    methodologies: List[str]
    applications: List[str]
    keywords: List[str]
    topics_covered: List[Dict]

class OptimizedXRefGenerator:
    def __init__(self, chapters_dir: str, max_refs: int = 4, use_llm: bool = False, llm_model: str = "llama3.1:8b"):
        self.chapters_dir = Path(chapters_dir)
        self.max_refs = max_refs
        self.concept_maps: Dict[str, ConceptMap] = {}
        self.chapter_order = []
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_enhancer = LLMEnhancer(llm_model) if self.use_llm else None

        # Educational flow hierarchy (for better connection types)
        self.educational_hierarchy = {
            'foundation': ['introduction', 'dl_primer'],
            'core_systems': ['ml_systems', 'dnn_architectures', 'data_engineering'],
            'implementation': ['frameworks', 'training', 'workflow'],
            'optimization': ['efficient_ai', 'optimizations', 'hw_acceleration'],
            'evaluation': ['benchmarking'],
            'deployment': ['ops'],
            'specialization': ['ondevice_learning', 'robust_ai', 'privacy_security'],
            'ethics': ['responsible_ai', 'sustainable_ai', 'ai_for_good'],
            'frontier': ['emerging_topics', 'generative_ai', 'frontiers'],
            'synthesis': ['conclusion']
        }

        # Concept category weights (higher = more important for matching)
        self.category_weights = {
            'primary_concepts': 1.0,
            'technical_terms': 0.8,
            'methodologies': 0.7,
            'secondary_concepts': 0.6,
            'applications': 0.5,
            'keywords': 0.3
        }

    def load_concept_maps(self) -> None:
        """Load all concept maps."""
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

                concept_map_data = data['concept_map']

                concept_map = ConceptMap(
                    chapter=chapter_name,
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

            except Exception as e:
                print(f"❌ Error loading {concept_file}: {e}")

        print(f"📚 Loaded {len(self.concept_maps)} concept maps")

    def find_weighted_concept_overlaps(self, source_chapter: str, target_chapter: str) -> Tuple[List[str], float]:
        """Find overlapping concepts with category weighting."""
        source_map = self.concept_maps[source_chapter]
        target_map = self.concept_maps[target_chapter]

        def normalize_concept(concept: str) -> str:
            return concept.lower().strip()

        def get_weighted_concepts(concept_map: ConceptMap) -> Dict[str, float]:
            """Get all concepts with their weights."""
            weighted = {}

            for category, weight in self.category_weights.items():
                concepts = getattr(concept_map, category, [])
                for concept in concepts:
                    normalized = normalize_concept(concept)
                    # Use maximum weight if concept appears in multiple categories
                    weighted[normalized] = max(weighted.get(normalized, 0), weight)

            return weighted

        source_weighted = get_weighted_concepts(source_map)
        target_weighted = get_weighted_concepts(target_map)

        # Find overlaps and calculate weighted strength
        overlaps = []
        total_weight = 0

        for concept in set(source_weighted.keys()) & set(target_weighted.keys()):
            weight = min(source_weighted[concept], target_weighted[concept])
            overlaps.append(concept)
            total_weight += weight

        # Find fuzzy overlaps (substring matching)
        fuzzy_weight = 0
        for source_concept in source_weighted:
            if source_concept in overlaps or len(source_concept) <= 4:
                continue
            for target_concept in target_weighted:
                if target_concept in overlaps or len(target_concept) <= 4:
                    continue
                if source_concept in target_concept or target_concept in source_concept:
                    weight = min(source_weighted[source_concept], target_weighted[target_concept]) * 0.7
                    fuzzy_weight += weight
                    overlaps.append(f"{source_concept}~{target_concept}")

        # Calculate strength (normalized by average concept set size)
        avg_concept_count = (len(source_weighted) + len(target_weighted)) / 2
        strength = (total_weight + fuzzy_weight) / avg_concept_count if avg_concept_count > 0 else 0

        return overlaps, strength

    def get_educational_level(self, chapter: str) -> str:
        """Get the educational level/category of a chapter."""
        for level, chapters in self.educational_hierarchy.items():
            if chapter in chapters:
                return level
        return 'core_systems'  # Default

    def determine_enhanced_connection_type(self, source_chapter: str, target_chapter: str, overlaps: List[str]) -> str:
        """Enhanced connection type classification."""
        source_level = self.get_educational_level(source_chapter)
        target_level = self.get_educational_level(target_chapter)

        # Educational progression patterns
        level_order = ['foundation', 'core_systems', 'implementation', 'optimization', 'evaluation', 'deployment', 'specialization', 'ethics', 'frontier', 'synthesis']

        try:
            source_idx = level_order.index(source_level)
            target_idx = level_order.index(target_level)

            # Prerequisite: foundation → advanced
            if source_idx < target_idx and target_idx - source_idx <= 2:
                return 'prerequisite'

            # Implementation: theory → practice
            if (source_level in ['foundation', 'core_systems'] and
                target_level in ['implementation', 'optimization']):
                return 'implementation'

            # Enhancement: basic → optimized
            if (source_level in ['implementation', 'core_systems'] and
                target_level == 'optimization'):
                return 'enhancement'

            # Application: theory → real-world
            if (source_level in ['foundation', 'core_systems', 'implementation'] and
                target_level in ['specialization', 'ethics']):
                return 'application'

        except ValueError:
            pass

        # Analyze actual concept overlaps for more specific types
        source_map = self.concept_maps[source_chapter]
        target_map = self.concept_maps[target_chapter]

        # Check for methodology sharing
        source_methods = set(c.lower() for c in source_map.methodologies)
        target_methods = set(c.lower() for c in target_map.methodologies)
        method_overlap = len(source_methods & target_methods)

        if method_overlap >= 2:
            return 'methodological'

        # Check for technical term sharing
        source_tech = set(c.lower() for c in source_map.technical_terms)
        target_tech = set(c.lower() for c in target_map.technical_terms)
        tech_overlap = len(source_tech & target_tech)

        if tech_overlap >= 3:
            return 'technical'

        # Default
        return 'conceptual'

    def generate_enhanced_explanation(self, source_chapter: str, target_chapter: str,
                                    connection_type: str, overlaps: List[str]) -> str:
        """Generate explanation with optional LLM enhancement."""
        overlap_text = ", ".join([o.split('~')[0] for o in overlaps[:3]])
        if len(overlaps) > 3:
            overlap_text += f" and {len(overlaps) - 3} more"

        base_explanations = {
            'prerequisite': f"Builds on foundational concepts: {overlap_text}",
            'implementation': f"Shows practical implementation of: {overlap_text}",
            'enhancement': f"Optimizes and enhances: {overlap_text}",
            'application': f"Demonstrates real-world applications of: {overlap_text}",
            'methodological': f"Shares methodological approaches: {overlap_text}",
            'technical': f"Uses related technical concepts: {overlap_text}",
            'conceptual': f"Explores related concepts: {overlap_text}"
        }

        base_explanation = base_explanations.get(connection_type, f"Connected through: {overlap_text}")

        # Use LLM enhancement if available
        if self.use_llm and self.llm_enhancer and self.llm_enhancer.available:
            try:
                enhanced = self.llm_enhancer.enhance_explanation(
                    source_chapter, target_chapter, connection_type,
                    [o.split('~')[0] for o in overlaps[:5]]
                )
                return enhanced
            except:
                pass

        return base_explanation

    def ensure_balanced_coverage(self, all_connections: Dict[str, List]) -> Dict[str, List]:
        """Ensure all chapters have at least some connections."""
        min_connections = 1

        for chapter in self.chapter_order:
            if len(all_connections.get(chapter, [])) < min_connections:
                # Find best available connections for this chapter
                candidates = []

                for target_chapter in self.chapter_order:
                    if chapter == target_chapter:
                        continue

                    overlaps, strength = self.find_weighted_concept_overlaps(chapter, target_chapter)
                    if len(overlaps) >= 1 and strength > 0.03:  # Lower threshold
                        connection_type = self.determine_enhanced_connection_type(chapter, target_chapter, overlaps)
                        explanation = self.generate_enhanced_explanation(chapter, target_chapter, connection_type, overlaps)

                        candidates.append({
                            'target_chapter': target_chapter,
                            'target_section': f'sec-{target_chapter}-overview',
                            'connection_type': connection_type,
                            'concepts': [o.split('~')[0] for o in overlaps[:5]],
                            'explanation': explanation,
                            'strength': strength
                        })

                # Add best candidates
                candidates.sort(key=lambda x: x['strength'], reverse=True)
                needed = min_connections - len(all_connections.get(chapter, []))
                if chapter not in all_connections:
                    all_connections[chapter] = []
                all_connections[chapter].extend(candidates[:needed])

        return all_connections

    def generate_cross_references(self) -> Dict[str, Dict]:
        """Generate optimized cross-references."""
        print("\n🔗 Generating optimized cross-references...")

        if self.use_llm:
            print(f"🤖 Using LLM enhancement with {self.llm_enhancer.model}")

        all_chapter_connections = {}

        # First pass: generate connections with higher thresholds
        for source_chapter in self.chapter_order:
            connections = []
            print(f"  📝 Processing {source_chapter}...")

            for target_chapter in self.chapter_order:
                if source_chapter == target_chapter:
                    continue

                overlaps, strength = self.find_weighted_concept_overlaps(source_chapter, target_chapter)

                # Use adaptive threshold based on chapter type
                source_level = self.get_educational_level(source_chapter)
                threshold = 0.08 if source_level in ['foundation', 'synthesis'] else 0.06

                if len(overlaps) >= 2 and strength > threshold:
                    connection_type = self.determine_enhanced_connection_type(source_chapter, target_chapter, overlaps)
                    explanation = self.generate_enhanced_explanation(source_chapter, target_chapter, connection_type, overlaps)

                    connections.append(({
                        'target_chapter': target_chapter,
                        'target_section': f'sec-{target_chapter}-overview',
                        'connection_type': connection_type,
                        'concepts': [o.split('~')[0] for o in overlaps[:5]],
                        'explanation': explanation,
                        'strength': round(strength, 3)
                    }, strength))

            # Sort by strength and take top connections
            connections.sort(key=lambda x: x[1], reverse=True)
            all_chapter_connections[source_chapter] = [conn[0] for conn in connections[:self.max_refs]]

        # Second pass: ensure balanced coverage
        all_chapter_connections = self.ensure_balanced_coverage(all_chapter_connections)

        # Build final structure
        all_xrefs = {}
        for source_chapter in self.chapter_order:
            connections = all_chapter_connections.get(source_chapter, [])

            chapter_xrefs = {}
            if connections:
                first_section = f'sec-{source_chapter}-overview'
                chapter_xrefs[first_section] = connections

            all_xrefs[source_chapter] = {
                'cross_references': chapter_xrefs,
                'generated_date': '2025-01-12',
                'generator': 'optimized_xrefs_generator.py',
                'llm_enhanced': self.use_llm,
                'total_connections': len(connections)
            }

            print(f"    ✅ Generated {len(connections)} connections")

        return all_xrefs

    def save_xref_files(self, all_xrefs: Dict[str, Dict]) -> None:
        """Save optimized cross-reference files."""
        print("\n💾 Saving optimized cross-reference files...")

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
        """Run the complete optimized cross-reference generation."""
        print("🚀 Optimized Concept-Driven Cross-Reference Generator")
        print("=" * 60)

        self.load_concept_maps()
        all_xrefs = self.generate_cross_references()
        self.save_xref_files(all_xrefs)

        total_connections = sum(xref['total_connections'] for xref in all_xrefs.values())
        print(f"\n🎉 Generated {total_connections} optimized cross-references across {len(all_xrefs)} chapters")
        print("✅ Optimized cross-reference generation complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized concept-driven cross-references",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('-d', '--chapters-dir', required=True)
    parser.add_argument('--max-refs', type=int, default=4)
    parser.add_argument('--llm', action='store_true', help='Use LLM enhancement')
    parser.add_argument('--model', default='llama3.1:8b', help='LLM model for enhancement')

    args = parser.parse_args()

    if not Path(args.chapters_dir).exists():
        print(f"❌ Chapters directory does not exist: {args.chapters_dir}")
        sys.exit(1)

    if args.llm and not LLM_AVAILABLE:
        print("❌ LLM enhancement requested but llm_enhanced_xrefs.py not available")
        sys.exit(1)

    generator = OptimizedXRefGenerator(
        args.chapters_dir,
        max_refs=args.max_refs,
        use_llm=args.llm,
        llm_model=args.model
    )

    generator.run()

if __name__ == "__main__":
    main()
