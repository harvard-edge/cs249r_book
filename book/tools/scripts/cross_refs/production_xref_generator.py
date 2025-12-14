#!/usr/bin/env python3
"""
Production Cross-Reference Generator

Based on comprehensive experimental findings, this is the final production-ready
cross-reference generator incorporating best practices from educational research.
"""

import os
import json
import yaml
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict, Counter
import time

class ProductionXRefGenerator:

    def __init__(self):
        self.base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
        self.chapters = [
            'introduction', 'ml_systems', 'dl_primer', 'workflow', 'data_engineering',
            'frameworks', 'training', 'efficient_ai', 'optimizations', 'hw_acceleration',
            'benchmarking', 'ondevice_learning', 'ops', 'privacy_security', 'responsible_ai',
            'sustainable_ai', 'ai_for_good', 'robust_ai', 'generative_ai', 'frontiers',
            'emerging_topics', 'conclusion'
        ]

        # Educational progression levels and types
        self.chapter_progression = {
            'introduction': {'level': 1, 'type': 'foundation', 'difficulty': 'basic'},
            'ml_systems': {'level': 1, 'type': 'foundation', 'difficulty': 'basic'},
            'dl_primer': {'level': 2, 'type': 'theory', 'difficulty': 'intermediate'},
            'workflow': {'level': 2, 'type': 'methodology', 'difficulty': 'intermediate'},
            'data_engineering': {'level': 2, 'type': 'methodology', 'difficulty': 'intermediate'},
            'frameworks': {'level': 3, 'type': 'implementation', 'difficulty': 'intermediate'},
            'training': {'level': 3, 'type': 'implementation', 'difficulty': 'advanced'},
            'efficient_ai': {'level': 4, 'type': 'optimization', 'difficulty': 'advanced'},
            'optimizations': {'level': 4, 'type': 'optimization', 'difficulty': 'advanced'},
            'hw_acceleration': {'level': 4, 'type': 'optimization', 'difficulty': 'advanced'},
            'benchmarking': {'level': 3, 'type': 'methodology', 'difficulty': 'intermediate'},
            'ondevice_learning': {'level': 5, 'type': 'specialization', 'difficulty': 'advanced'},
            'ops': {'level': 4, 'type': 'implementation', 'difficulty': 'advanced'},
            'privacy_security': {'level': 5, 'type': 'specialization', 'difficulty': 'expert'},
            'responsible_ai': {'level': 5, 'type': 'specialization', 'difficulty': 'expert'},
            'sustainable_ai': {'level': 5, 'type': 'specialization', 'difficulty': 'expert'},
            'ai_for_good': {'level': 5, 'type': 'application', 'difficulty': 'expert'},
            'robust_ai': {'level': 5, 'type': 'specialization', 'difficulty': 'expert'},
            'generative_ai': {'level': 4, 'type': 'specialization', 'difficulty': 'advanced'},
            'frontiers': {'level': 6, 'type': 'advanced', 'difficulty': 'expert'},
            'emerging_topics': {'level': 6, 'type': 'advanced', 'difficulty': 'expert'},
            'conclusion': {'level': 6, 'type': 'synthesis', 'difficulty': 'expert'}
        }

        # Connection types based on educational research
        self.connection_types = {
            'prerequisite': {
                'description': 'Essential background needed before this chapter',
                'placement': 'section_start',
                'max_per_section': 2,
                'priority': 1
            },
            'foundation': {
                'description': 'Fundamental concepts this chapter builds upon',
                'placement': 'chapter_start',
                'max_per_section': 3,
                'priority': 1
            },
            'extends': {
                'description': 'Advanced topics that build on this chapter',
                'placement': 'section_end',
                'max_per_section': 2,
                'priority': 2
            },
            'applies': {
                'description': 'Real-world applications of these concepts',
                'placement': 'contextual',
                'max_per_section': 2,
                'priority': 2
            },
            'complements': {
                'description': 'Related topics that provide additional perspective',
                'placement': 'sidebar',
                'max_per_section': 1,
                'priority': 3
            },
            'contrasts': {
                'description': 'Alternative approaches or trade-offs',
                'placement': 'inline',
                'max_per_section': 1,
                'priority': 3
            }
        }

        # Optimal threshold from experiments
        self.optimal_threshold = 0.01

    def load_concept_map(self, chapter: str) -> Dict:
        """Load concept map for a chapter"""
        concept_file = self.base_dir / chapter / f"{chapter}_concepts.yml"
        if not concept_file.exists():
            return {}

        with open(concept_file, 'r') as f:
            return yaml.safe_load(f)

    def extract_sections(self, chapter: str) -> List[Dict]:
        """Extract all sections from a chapter QMD file"""
        qmd_file = self.base_dir / chapter / f"{chapter}.qmd"
        if not qmd_file.exists():
            return []

        sections = []
        with open(qmd_file, 'r') as f:
            content = f.read()

        # Find all section headers with IDs
        section_pattern = r'^## (.+?) \{#(sec-[\w-]+)\}'
        for match in re.finditer(section_pattern, content, re.MULTILINE):
            title, section_id = match.groups()
            # Clean up title
            clean_title = re.sub(r'\s*\{\.unnumbered\}', '', title).strip()
            sections.append({
                'title': clean_title,
                'id': section_id,
                'chapter': chapter
            })

        return sections

    def generate_cross_references(self):
        """Generate production-quality cross-references for all chapters"""
        print("Generating production cross-references...")

        all_cross_references = {}

        for source_chapter in self.chapters:
            print(f"Processing {source_chapter}...")

            source_concepts = self.load_concept_map(source_chapter)
            source_sections = self.extract_sections(source_chapter)
            source_progression = self.chapter_progression[source_chapter]

            chapter_xrefs = {}

            # Generate cross-references for each section in the chapter
            for section in source_sections:
                section_id = section['id']

                section_connections = []

                # Find connections to other chapters
                for target_chapter in self.chapters:
                    if source_chapter == target_chapter:
                        continue

                    target_concepts = self.load_concept_map(target_chapter)
                    target_progression = self.chapter_progression[target_chapter]

                    overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                    if strength > self.optimal_threshold:
                        connection_type = self._classify_connection(
                            source_chapter, target_chapter,
                            source_progression, target_progression,
                            overlap, strength
                        )

                        # Calculate connection quality
                        quality = self._calculate_connection_quality(
                            source_chapter, target_chapter,
                            source_progression, target_progression,
                            connection_type, strength, overlap
                        )

                        if quality > 0.6:  # High-quality connections only
                            target_sections = self.extract_sections(target_chapter)
                            target_section = target_sections[0] if target_sections else {'id': f'sec-{target_chapter}-overview', 'title': 'Overview'}

                            section_connections.append({
                                'target_chapter': target_chapter,
                                'target_section': target_section['id'],
                                'connection_type': connection_type,
                                'concepts': overlap[:5],
                                'strength': strength,
                                'quality': quality,
                                'explanation': self._generate_explanation(
                                    connection_type, overlap, source_chapter, target_chapter
                                ),
                                'placement': self.connection_types[connection_type]['placement'],
                                'priority': self.connection_types[connection_type]['priority']
                            })

                # Sort by quality and limit by priority and placement type
                section_connections.sort(key=lambda x: (x['priority'], -x['quality']))

                # Apply limits based on connection type and placement
                final_connections = self._apply_connection_limits(section_connections)

                if final_connections:
                    chapter_xrefs[section_id] = final_connections

            # If no section-specific connections, create chapter-level connections
            if not chapter_xrefs and source_sections:
                overview_section = source_sections[0]['id']
                chapter_connections = []

                for target_chapter in self.chapters:
                    if source_chapter == target_chapter:
                        continue

                    target_concepts = self.load_concept_map(target_chapter)
                    target_progression = self.chapter_progression[target_chapter]

                    overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                    if strength > self.optimal_threshold * 1.2:  # Higher threshold for overview
                        connection_type = self._classify_connection(
                            source_chapter, target_chapter,
                            source_progression, target_progression,
                            overlap, strength
                        )

                        target_sections = self.extract_sections(target_chapter)
                        target_section = target_sections[0] if target_sections else {'id': f'sec-{target_chapter}-overview', 'title': 'Overview'}

                        chapter_connections.append({
                            'target_chapter': target_chapter,
                            'target_section': target_section['id'],
                            'connection_type': connection_type,
                            'concepts': overlap[:3],
                            'strength': strength,
                            'explanation': self._generate_explanation(
                                connection_type, overlap, source_chapter, target_chapter
                            )
                        })

                if chapter_connections:
                    chapter_connections.sort(key=lambda x: -x['strength'])
                    chapter_xrefs[overview_section] = chapter_connections[:3]  # Top 3 for overview

            all_cross_references[source_chapter] = {
                'cross_references': chapter_xrefs,
                'generated_date': time.strftime('%Y-%m-%d'),
                'generator': 'production_xref_generator.py',
                'version': '1.0',
                'total_connections': sum(len(conns) for conns in chapter_xrefs.values())
            }

        # Save all cross-reference files
        self._save_cross_reference_files(all_cross_references)

        # Generate summary statistics
        self._generate_summary_statistics(all_cross_references)

    def _find_concept_overlap(self, concepts1: Dict, concepts2: Dict) -> Tuple[List[str], float]:
        """Find overlapping concepts with production-tuned scoring"""
        if not concepts1 or not concepts2:
            return [], 0.0

        # Production-tuned weights based on experimental results
        weights = {
            'primary_concepts': 1.0,
            'methodologies': 0.9,
            'technical_terms': 0.8,
            'applications': 0.7,
            'secondary_concepts': 0.6,
            'keywords': 0.3
        }

        all_concepts1 = []
        all_concepts2 = []

        for category, weight in weights.items():
            concepts_cat1 = concepts1.get('concept_map', {}).get(category, [])
            concepts_cat2 = concepts2.get('concept_map', {}).get(category, [])

            if isinstance(concepts_cat1, list):
                all_concepts1.extend([(c.lower().strip(), weight) for c in concepts_cat1])
            if isinstance(concepts_cat2, list):
                all_concepts2.extend([(c.lower().strip(), weight) for c in concepts_cat2])

        # Find overlaps with fuzzy matching for similar terms
        concepts1_dict = {c[0]: c[1] for c in all_concepts1}
        concepts2_dict = {c[0]: c[1] for c in all_concepts2}

        overlapping = []
        total_strength = 0

        # Exact matches
        for concept, weight1 in concepts1_dict.items():
            if concept in concepts2_dict:
                weight2 = concepts2_dict[concept]
                overlapping.append(concept)
                total_strength += (weight1 + weight2) / 2

        # Fuzzy matches for similar terms
        for concept1, weight1 in concepts1_dict.items():
            if concept1 not in overlapping:
                for concept2, weight2 in concepts2_dict.items():
                    if self._concepts_similar(concept1, concept2):
                        overlapping.append(f"{concept1}~{concept2}")
                        total_strength += ((weight1 + weight2) / 2) * 0.8  # Discount for fuzzy match
                        break

        # Enhanced normalization
        total_concepts = max(len(concepts1_dict), len(concepts2_dict))
        normalized_strength = total_strength / total_concepts if total_concepts > 0 else 0

        return overlapping[:10], normalized_strength  # Limit to top 10 concepts

    def _concepts_similar(self, concept1: str, concept2: str) -> bool:
        """Check if two concepts are similar enough to be considered related"""
        # Simple similarity checks
        if len(concept1) < 4 or len(concept2) < 4:
            return False

        # Check for substring relationships
        if concept1 in concept2 or concept2 in concept1:
            return True

        # Check for common roots (simple heuristic)
        if concept1[:4] == concept2[:4] or concept1[-4:] == concept2[-4:]:
            return True

        return False

    def _classify_connection(self, source: str, target: str,
                           source_prog: Dict, target_prog: Dict,
                           concepts: List[str], strength: float) -> str:
        """Classify connection type using production rules"""

        level_diff = target_prog['level'] - source_prog['level']
        source_type = source_prog['type']
        target_type = target_prog['type']

        # Foundation connections (what this builds on)
        if level_diff < 0 or (level_diff == 0 and source_prog['difficulty'] == 'basic'):
            return 'foundation'

        # Prerequisite connections (immediate next step)
        if level_diff == 1 and target_type in ['theory', 'implementation']:
            return 'prerequisite'

        # Extension connections (advanced topics)
        if level_diff > 1 or (level_diff == 1 and target_type in ['optimization', 'specialization']):
            return 'extends'

        # Application connections
        if target_type == 'application' or 'ai_for_good' in target:
            return 'applies'

        # Complementary connections (same level, different approach)
        if level_diff == 0 and source_type != target_type:
            return 'complements'

        # Default to complementary
        return 'complements'

    def _calculate_connection_quality(self, source: str, target: str,
                                    source_prog: Dict, target_prog: Dict,
                                    connection_type: str, strength: float,
                                    concepts: List[str]) -> float:
        """Calculate connection quality score"""
        quality = 0.0

        # Base strength score (40%)
        quality += min(strength * 4, 0.4)

        # Educational progression score (30%)
        level_diff = target_prog['level'] - source_prog['level']
        if connection_type == 'foundation' and level_diff <= 0:
            quality += 0.3
        elif connection_type == 'prerequisite' and level_diff == 1:
            quality += 0.3
        elif connection_type == 'extends' and level_diff > 0:
            quality += 0.25
        else:
            quality += 0.15

        # Concept overlap score (20%)
        concept_score = min(len(concepts) / 5, 1.0) * 0.2
        quality += concept_score

        # Connection type priority score (10%)
        type_priority = self.connection_types[connection_type]['priority']
        priority_score = (4 - type_priority) / 3 * 0.1
        quality += priority_score

        return min(quality, 1.0)

    def _generate_explanation(self, connection_type: str, concepts: List[str],
                            source: str, target: str) -> str:
        """Generate human-readable explanation for the connection"""

        concept_text = ", ".join(concepts[:3])
        if len(concepts) > 3:
            concept_text += f" and {len(concepts) - 3} more"

        explanations = {
            'foundation': f"Builds on foundational concepts: {concept_text}",
            'prerequisite': f"Essential prerequisite covering: {concept_text}",
            'extends': f"Advanced extension exploring: {concept_text}",
            'applies': f"Real-world applications of: {concept_text}",
            'complements': f"Complementary perspective on: {concept_text}",
            'contrasts': f"Alternative approach to: {concept_text}"
        }

        return explanations.get(connection_type, f"Related concepts: {concept_text}")

    def _apply_connection_limits(self, connections: List[Dict]) -> List[Dict]:
        """Apply limits based on connection type and placement"""
        limited_connections = []
        placement_counts = defaultdict(int)

        for conn in connections:
            conn_type = conn['connection_type']
            placement = conn['placement']
            max_limit = self.connection_types[conn_type]['max_per_section']

            if placement_counts[placement] < max_limit:
                limited_connections.append(conn)
                placement_counts[placement] += 1

        return limited_connections

    def _save_cross_reference_files(self, all_cross_references: Dict):
        """Save cross-reference JSON files for each chapter"""
        for chapter, xref_data in all_cross_references.items():
            output_file = self.base_dir / chapter / f"{chapter}_xrefs.json"

            with open(output_file, 'w') as f:
                json.dump(xref_data, f, indent=2)

            print(f"Saved {xref_data['total_connections']} connections to {output_file}")

    def _generate_summary_statistics(self, all_cross_references: Dict):
        """Generate summary statistics for the cross-reference system"""
        total_connections = sum(data['total_connections'] for data in all_cross_references.values())
        chapters_with_connections = len([ch for ch, data in all_cross_references.items() if data['total_connections'] > 0])

        # Analyze connection types
        connection_type_counts = defaultdict(int)
        for chapter_data in all_cross_references.values():
            for section_connections in chapter_data['cross_references'].values():
                for connection in section_connections:
                    connection_type_counts[connection['connection_type']] += 1

        print(f"\n=== PRODUCTION CROSS-REFERENCE SUMMARY ===")
        print(f"Total connections: {total_connections}")
        print(f"Chapters with connections: {chapters_with_connections}/{len(self.chapters)}")
        print(f"Average connections per chapter: {total_connections / len(self.chapters):.1f}")

        print("\nConnection type distribution:")
        for conn_type, count in sorted(connection_type_counts.items()):
            percentage = (count / total_connections) * 100
            print(f"  {conn_type}: {count} ({percentage:.1f}%)")

        # Save summary
        summary_file = "/Users/VJ/GitHub/MLSysBook/tools/scripts/cross_refs/production_summary.json"
        summary_data = {
            'total_connections': total_connections,
            'chapters_with_connections': chapters_with_connections,
            'connection_type_distribution': dict(connection_type_counts),
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'generator_version': '1.0'
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    generator = ProductionXRefGenerator()
    generator.generate_cross_references()
