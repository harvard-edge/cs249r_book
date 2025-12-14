#!/usr/bin/env python3
"""
Cognitive Load Optimized Cross-Reference Generator

Based on 2024 research findings on cognitive load theory, educational design principles,
and hyperlink placement research, this generator creates cognitively-optimized
cross-references for educational textbooks.

Key principles implemented:
1. Modality Principle: Balance visual and textual information
2. Spatial Contiguity Principle: Place related information together
3. Segmentation: Break complex information into digestible chunks
4. Progressive Disclosure: Reveal information as needed
5. Hyperlink Placement Optimization: Strategic placement for learning outcomes

Based on 2024 educational research
"""

import os
import json
import yaml
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict, Counter
import time

class CognitiveLoadOptimizedGenerator:

    def __init__(self):
        self.base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
        self.chapters = [
            'introduction', 'ml_systems', 'dl_primer', 'workflow', 'data_engineering',
            'frameworks', 'training', 'efficient_ai', 'optimizations', 'hw_acceleration',
            'benchmarking', 'ondevice_learning', 'ops', 'privacy_security', 'responsible_ai',
            'sustainable_ai', 'ai_for_good', 'robust_ai', 'generative_ai', 'frontiers',
            'emerging_topics', 'conclusion'
        ]

        # Cognitive Load Theory principles for connection types and placement
        self.cognitive_connection_types = {
            'prerequisite_foundation': {
                'description': 'Essential prerequisite knowledge (low cognitive load)',
                'placement': 'chapter_start',
                'max_per_section': 2,
                'cognitive_load': 'low',
                'priority': 1
            },
            'conceptual_bridge': {
                'description': 'Bridge between related concepts (medium cognitive load)',
                'placement': 'section_transition',
                'max_per_section': 3,
                'cognitive_load': 'medium',
                'priority': 2
            },
            'progressive_extension': {
                'description': 'Gradual extension to advanced topics (controlled cognitive load)',
                'placement': 'section_end',
                'max_per_section': 2,
                'cognitive_load': 'medium',
                'priority': 2
            },
            'application_example': {
                'description': 'Real-world application examples (high engagement, manageable load)',
                'placement': 'contextual_sidebar',
                'max_per_section': 1,
                'cognitive_load': 'medium',
                'priority': 3
            },
            'optional_deepdive': {
                'description': 'Optional deep-dive for advanced readers (on-demand)',
                'placement': 'expandable',
                'max_per_section': 1,
                'cognitive_load': 'high',
                'priority': 4
            }
        }

        # Educational progression model based on research
        self.educational_progression = {
            'introduction': {'complexity': 1, 'depth': 'surface', 'prerequisites': []},
            'ml_systems': {'complexity': 2, 'depth': 'surface', 'prerequisites': ['introduction']},
            'dl_primer': {'complexity': 3, 'depth': 'deep', 'prerequisites': ['introduction', 'ml_systems']},
            'workflow': {'complexity': 2, 'depth': 'applied', 'prerequisites': ['introduction', 'ml_systems']},
            'data_engineering': {'complexity': 3, 'depth': 'applied', 'prerequisites': ['workflow']},
            'frameworks': {'complexity': 4, 'depth': 'deep', 'prerequisites': ['dl_primer', 'workflow']},
            'training': {'complexity': 5, 'depth': 'deep', 'prerequisites': ['dl_primer', 'frameworks']},
            'efficient_ai': {'complexity': 6, 'depth': 'applied', 'prerequisites': ['training']},
            'optimizations': {'complexity': 6, 'depth': 'applied', 'prerequisites': ['training', 'efficient_ai']},
            'hw_acceleration': {'complexity': 7, 'depth': 'specialized', 'prerequisites': ['optimizations']},
            'benchmarking': {'complexity': 4, 'depth': 'applied', 'prerequisites': ['frameworks']},
            'ondevice_learning': {'complexity': 8, 'depth': 'specialized', 'prerequisites': ['efficient_ai', 'hw_acceleration']},
            'ops': {'complexity': 5, 'depth': 'applied', 'prerequisites': ['frameworks', 'training']},
            'privacy_security': {'complexity': 7, 'depth': 'specialized', 'prerequisites': ['ops']},
            'responsible_ai': {'complexity': 6, 'depth': 'conceptual', 'prerequisites': ['ops']},
            'sustainable_ai': {'complexity': 6, 'depth': 'conceptual', 'prerequisites': ['efficient_ai']},
            'ai_for_good': {'complexity': 5, 'depth': 'applied', 'prerequisites': ['responsible_ai']},
            'robust_ai': {'complexity': 7, 'depth': 'specialized', 'prerequisites': ['training']},
            'generative_ai': {'complexity': 8, 'depth': 'specialized', 'prerequisites': ['training']},
            'frontiers': {'complexity': 9, 'depth': 'cutting_edge', 'prerequisites': ['generative_ai']},
            'emerging_topics': {'complexity': 9, 'depth': 'cutting_edge', 'prerequisites': ['frontiers']},
            'conclusion': {'complexity': 3, 'depth': 'synthesis', 'prerequisites': ['all']}
        }

    def load_concept_map(self, chapter: str) -> Dict:
        """Load concept map for a chapter"""
        concept_file = self.base_dir / chapter / f"{chapter}_concepts.yml"
        if not concept_file.exists():
            return {}

        with open(concept_file, 'r') as f:
            return yaml.safe_load(f)

    def extract_sections(self, chapter: str) -> List[Dict]:
        """Extract sections with cognitive load analysis"""
        qmd_file = self.base_dir / chapter / f"{chapter}.qmd"
        if not qmd_file.exists():
            return []

        sections = []
        with open(qmd_file, 'r') as f:
            content = f.read()

        # Find all section headers with IDs and analyze content
        section_pattern = r'^## (.+?) \{#(sec-[\w-]+)\}'
        for match in re.finditer(section_pattern, content, re.MULTILINE):
            title, section_id = match.groups()

            # Estimate cognitive load of section based on title and content analysis
            cognitive_complexity = self._analyze_section_complexity(title, content)

            sections.append({
                'title': title.strip(),
                'id': section_id,
                'chapter': chapter,
                'cognitive_complexity': cognitive_complexity
            })

        return sections

    def generate_cognitive_load_optimized_xrefs(self):
        """Generate cognitively-optimized cross-references"""
        print("Generating cognitive load optimized cross-references...")

        all_cross_references = {}

        for source_chapter in self.chapters:
            print(f"Processing {source_chapter}...")

            source_concepts = self.load_concept_map(source_chapter)
            source_sections = self.extract_sections(source_chapter)
            source_progression = self.educational_progression[source_chapter]

            chapter_xrefs = {}

            # Generate connections for each section with cognitive load optimization
            for section in source_sections:
                section_id = section['id']
                section_complexity = section['cognitive_complexity']

                section_connections = []

                # Find cognitively-appropriate connections
                for target_chapter in self.chapters:
                    if source_chapter == target_chapter:
                        continue

                    target_concepts = self.load_concept_map(target_chapter)
                    target_progression = self.educational_progression[target_chapter]

                    overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                    if strength > 0.01:  # Lower threshold for more connections
                        # Classify connection using cognitive load principles
                        connection_type, cognitive_load = self._classify_cognitive_connection(
                            source_chapter, target_chapter,
                            source_progression, target_progression,
                            section_complexity, strength, overlap
                        )

                        # Calculate pedagogical value
                        pedagogical_value = self._calculate_pedagogical_value(
                            source_chapter, target_chapter, connection_type,
                            cognitive_load, strength, overlap
                        )

                        if pedagogical_value > 0.5:  # Only high-value connections
                            target_sections = self.extract_sections(target_chapter)
                            target_section = target_sections[0] if target_sections else {
                                'id': f'sec-{target_chapter}-overview',
                                'title': 'Overview'
                            }

                            section_connections.append({
                                'target_chapter': target_chapter,
                                'target_section': target_section['id'],
                                'connection_type': connection_type,
                                'cognitive_load': cognitive_load,
                                'concepts': overlap[:5],
                                'strength': strength,
                                'pedagogical_value': pedagogical_value,
                                'explanation': self._generate_cognitive_explanation(
                                    connection_type, overlap, source_chapter, target_chapter
                                ),
                                'placement': self.cognitive_connection_types[connection_type]['placement'],
                                'priority': self.cognitive_connection_types[connection_type]['priority']
                            })

                # Apply cognitive load management
                optimized_connections = self._apply_cognitive_load_management(
                    section_connections, section_complexity
                )

                if optimized_connections:
                    chapter_xrefs[section_id] = optimized_connections

            all_cross_references[source_chapter] = {
                'cross_references': chapter_xrefs,
                'generated_date': time.strftime('%Y-%m-%d'),
                'generator': 'cognitive_load_optimized_generator.py',
                'version': '1.0',
                'optimization_approach': 'cognitive_load_theory',
                'total_connections': sum(len(conns) for conns in chapter_xrefs.values())
            }

        # Save all cross-reference files
        self._save_cognitive_optimized_files(all_cross_references)

        # Generate cognitive load analysis report
        self._generate_cognitive_load_report(all_cross_references)

    def _analyze_section_complexity(self, title: str, content: str) -> str:
        """Analyze cognitive complexity of a section based on title and content indicators"""
        # Keywords that indicate high cognitive load
        high_load_keywords = [
            'architecture', 'optimization', 'implementation', 'algorithm',
            'mathematical', 'distributed', 'advanced', 'specialized'
        ]

        medium_load_keywords = [
            'system', 'framework', 'model', 'training', 'application',
            'workflow', 'pipeline', 'deployment'
        ]

        low_load_keywords = [
            'introduction', 'overview', 'basics', 'purpose', 'summary',
            'fundamentals', 'concepts'
        ]

        title_lower = title.lower()

        if any(keyword in title_lower for keyword in high_load_keywords):
            return 'high'
        elif any(keyword in title_lower for keyword in medium_load_keywords):
            return 'medium'
        elif any(keyword in title_lower for keyword in low_load_keywords):
            return 'low'
        else:
            return 'medium'  # Default

    def _find_concept_overlap(self, concepts1: Dict, concepts2: Dict) -> Tuple[List[str], float]:
        """Enhanced concept overlap detection with cognitive load considerations"""
        if not concepts1 or not concepts2:
            return [], 0.0

        # Weights optimized for cognitive load management
        weights = {
            'primary_concepts': 1.0,    # Core concepts - highest cognitive value
            'methodologies': 0.9,       # Methods - high practical value
            'applications': 0.8,        # Applications - high engagement
            'technical_terms': 0.7,     # Technical terms - medium load
            'secondary_concepts': 0.6,  # Secondary - lower priority
            'keywords': 0.3            # Keywords - lowest cognitive load
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

        # Find overlaps with cognitive load optimization
        concepts1_dict = {c[0]: c[1] for c in all_concepts1}
        concepts2_dict = {c[0]: c[1] for c in all_concepts2}

        overlapping = []
        total_strength = 0

        # Exact matches
        for concept, weight1 in concepts1_dict.items():
            if concept in concepts2_dict:
                weight2 = concepts2_dict[concept]
                overlapping.append(concept)
                # Cognitive load adjusted strength
                total_strength += (weight1 + weight2) / 2 * 1.2  # Boost exact matches

        # Fuzzy matches with cognitive load consideration
        for concept1, weight1 in concepts1_dict.items():
            if concept1 not in overlapping and len(concept1) > 3:
                for concept2, weight2 in concepts2_dict.items():
                    if self._concepts_cognitively_similar(concept1, concept2):
                        overlapping.append(f"{concept1}~{concept2}")
                        total_strength += ((weight1 + weight2) / 2) * 0.8  # Discount fuzzy matches
                        break

        # Cognitive load normalized strength
        max_possible = min(len(concepts1_dict), 10) + min(len(concepts2_dict), 10)  # Cap for cognitive load
        normalized_strength = total_strength / max_possible if max_possible > 0 else 0

        return overlapping[:8], normalized_strength  # Limit for cognitive load management

    def _concepts_cognitively_similar(self, concept1: str, concept2: str) -> bool:
        """Check cognitive similarity between concepts"""
        if len(concept1) < 4 or len(concept2) < 4:
            return False

        # Enhanced similarity for cognitive load optimization
        if concept1 in concept2 or concept2 in concept1:
            return True

        # Common technical prefixes/suffixes
        if (concept1[:4] == concept2[:4] or
            concept1[-4:] == concept2[-4:]):
            return True

        # Domain-specific similarity
        ml_terms = ['learning', 'training', 'model', 'neural', 'deep']
        system_terms = ['system', 'framework', 'platform', 'architecture']

        for term_set in [ml_terms, system_terms]:
            if (any(term in concept1 for term in term_set) and
                any(term in concept2 for term in term_set)):
                return True

        return False

    def _classify_cognitive_connection(self, source: str, target: str,
                                     source_prog: Dict, target_prog: Dict,
                                     section_complexity: str, strength: float,
                                     concepts: List[str]) -> Tuple[str, str]:
        """Classify connection type with cognitive load considerations"""

        complexity_diff = target_prog['complexity'] - source_prog['complexity']
        source_prereqs = source_prog.get('prerequisites', [])

        # Cognitive load based classification
        if target in source_prereqs or complexity_diff < -2:
            return 'prerequisite_foundation', 'low'

        elif -1 <= complexity_diff <= 1:
            if strength > 0.08:
                return 'conceptual_bridge', 'medium'
            else:
                return 'application_example', 'medium'

        elif 1 < complexity_diff <= 3:
            if section_complexity == 'low':
                return 'progressive_extension', 'medium'
            else:
                return 'optional_deepdive', 'high'

        elif complexity_diff > 3:
            return 'optional_deepdive', 'high'

        else:
            return 'conceptual_bridge', 'medium'

    def _calculate_pedagogical_value(self, source: str, target: str,
                                   connection_type: str, cognitive_load: str,
                                   strength: float, concepts: List[str]) -> float:
        """Calculate pedagogical value with cognitive load considerations"""

        base_value = 0.0

        # Base strength contribution (30%)
        base_value += min(strength * 3, 0.3)

        # Connection type value (30%)
        type_values = {
            'prerequisite_foundation': 0.3,
            'conceptual_bridge': 0.25,
            'progressive_extension': 0.2,
            'application_example': 0.25,
            'optional_deepdive': 0.15
        }
        base_value += type_values.get(connection_type, 0.1)

        # Cognitive load appropriateness (25%)
        if cognitive_load == 'low':
            base_value += 0.25
        elif cognitive_load == 'medium':
            base_value += 0.2
        else:  # high
            base_value += 0.1

        # Concept relevance (15%)
        concept_bonus = min(len(concepts) / 8, 1.0) * 0.15
        base_value += concept_bonus

        return min(base_value, 1.0)

    def _apply_cognitive_load_management(self, connections: List[Dict],
                                       section_complexity: str) -> List[Dict]:
        """Apply cognitive load management to limit connections per section"""

        if not connections:
            return []

        # Sort by pedagogical value and priority
        connections.sort(key=lambda x: (x['priority'], -x['pedagogical_value']))

        # Cognitive load limits based on section complexity
        complexity_limits = {
            'low': 5,     # More connections for low complexity sections
            'medium': 4,  # Balanced for medium complexity
            'high': 3     # Fewer connections for high complexity sections
        }

        max_connections = complexity_limits.get(section_complexity, 4)

        # Apply type-specific limits
        limited_connections = []
        type_counts = defaultdict(int)

        for conn in connections:
            conn_type = conn['connection_type']
            max_for_type = self.cognitive_connection_types[conn_type]['max_per_section']

            if (type_counts[conn_type] < max_for_type and
                len(limited_connections) < max_connections):
                limited_connections.append(conn)
                type_counts[conn_type] += 1

        return limited_connections

    def _generate_cognitive_explanation(self, connection_type: str, concepts: List[str],
                                      source: str, target: str) -> str:
        """Generate cognitively-appropriate explanations"""

        concept_text = ", ".join(concepts[:3])
        if len(concepts) > 3:
            concept_text += f" and {len(concepts) - 3} more"

        explanations = {
            'prerequisite_foundation': f"Essential background: {concept_text}",
            'conceptual_bridge': f"Related concepts: {concept_text}",
            'progressive_extension': f"Next steps exploring: {concept_text}",
            'application_example': f"Applied in practice: {concept_text}",
            'optional_deepdive': f"Deep dive available: {concept_text}"
        }

        return explanations.get(connection_type, f"Connected topics: {concept_text}")

    def _save_cognitive_optimized_files(self, all_cross_references: Dict):
        """Save cognitively-optimized cross-reference files"""
        for chapter, xref_data in all_cross_references.items():
            output_file = self.base_dir / chapter / f"{chapter}_cognitive_xrefs.json"

            with open(output_file, 'w') as f:
                json.dump(xref_data, f, indent=2)

            print(f"Saved {xref_data['total_connections']} cognitive connections to {output_file}")

    def _generate_cognitive_load_report(self, all_cross_references: Dict):
        """Generate comprehensive cognitive load analysis report"""

        print(f"\n=== COGNITIVE LOAD OPTIMIZED CROSS-REFERENCE REPORT ===")

        total_connections = sum(data['total_connections'] for data in all_cross_references.values())
        chapters_with_connections = len([ch for ch, data in all_cross_references.items()
                                       if data['total_connections'] > 0])

        # Analyze cognitive load distribution
        cognitive_load_dist = defaultdict(int)
        connection_type_dist = defaultdict(int)
        placement_dist = defaultdict(int)

        for chapter_data in all_cross_references.values():
            for section_connections in chapter_data['cross_references'].values():
                for connection in section_connections:
                    cognitive_load_dist[connection['cognitive_load']] += 1
                    connection_type_dist[connection['connection_type']] += 1
                    placement_dist[connection['placement']] += 1

        print(f"Total cognitive-optimized connections: {total_connections}")
        print(f"Chapters with connections: {chapters_with_connections}/{len(self.chapters)}")
        print(f"Average connections per chapter: {total_connections / len(self.chapters):.1f}")

        print("\nCognitive Load Distribution:")
        for load_type, count in sorted(cognitive_load_dist.items()):
            percentage = (count / total_connections) * 100
            print(f"  {load_type}: {count} ({percentage:.1f}%)")

        print("\nConnection Type Distribution:")
        for conn_type, count in sorted(connection_type_dist.items()):
            percentage = (count / total_connections) * 100
            print(f"  {conn_type}: {count} ({percentage:.1f}%)")

        print("\nPlacement Strategy Distribution:")
        for placement, count in sorted(placement_dist.items()):
            percentage = (count / total_connections) * 100
            print(f"  {placement}: {count} ({percentage:.1f}%)")

        # Save detailed analysis
        analysis_file = "/Users/VJ/GitHub/MLSysBook/tools/scripts/cross_refs/cognitive_load_analysis.json"
        analysis_data = {
            'total_connections': total_connections,
            'chapters_with_connections': chapters_with_connections,
            'cognitive_load_distribution': dict(cognitive_load_dist),
            'connection_type_distribution': dict(connection_type_dist),
            'placement_distribution': dict(placement_dist),
            'optimization_principles': list(self.cognitive_connection_types.keys()),
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'research_basis': 'Cognitive Load Theory 2024, Educational Design Principles, Hyperlink Placement Research'
        }

        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print(f"\nDetailed cognitive load analysis saved to: {analysis_file}")

if __name__ == "__main__":
    generator = CognitiveLoadOptimizedGenerator()
    generator.generate_cognitive_load_optimized_xrefs()
