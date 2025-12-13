#!/usr/bin/env python3
"""
Refined Cross-Reference Experiments

Based on initial experimental findings, these focused experiments address:
1. Section-level granularity (cross-chapter only, not within-chapter)
2. Fine-tuned threshold optimization
3. Improved pedagogical classification
4. Asymmetric bidirectional connections
"""

import os
import json
import yaml
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict, Counter
import time

class RefinedCrossRefExperiments:

    def __init__(self):
        self.base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
        self.chapters = [
            'introduction', 'ml_systems', 'dl_primer', 'workflow', 'data_engineering',
            'frameworks', 'training', 'efficient_ai', 'optimizations', 'hw_acceleration',
            'benchmarking', 'ondevice_learning', 'ops', 'privacy_security', 'responsible_ai',
            'sustainable_ai', 'ai_for_good', 'robust_ai', 'generative_ai', 'frontiers',
            'emerging_topics', 'conclusion'
        ]

        # Educational progression map
        self.chapter_progression = {
            'introduction': {'level': 1, 'type': 'foundation'},
            'ml_systems': {'level': 1, 'type': 'foundation'},
            'dl_primer': {'level': 2, 'type': 'theory'},
            'workflow': {'level': 2, 'type': 'methodology'},
            'data_engineering': {'level': 2, 'type': 'methodology'},
            'frameworks': {'level': 3, 'type': 'implementation'},
            'training': {'level': 3, 'type': 'implementation'},
            'efficient_ai': {'level': 4, 'type': 'optimization'},
            'optimizations': {'level': 4, 'type': 'optimization'},
            'hw_acceleration': {'level': 4, 'type': 'optimization'},
            'benchmarking': {'level': 3, 'type': 'methodology'},
            'ondevice_learning': {'level': 5, 'type': 'specialization'},
            'ops': {'level': 4, 'type': 'implementation'},
            'privacy_security': {'level': 5, 'type': 'specialization'},
            'responsible_ai': {'level': 5, 'type': 'specialization'},
            'sustainable_ai': {'level': 5, 'type': 'specialization'},
            'ai_for_good': {'level': 5, 'type': 'application'},
            'robust_ai': {'level': 5, 'type': 'specialization'},
            'generative_ai': {'level': 4, 'type': 'specialization'},
            'frontiers': {'level': 6, 'type': 'advanced'},
            'emerging_topics': {'level': 6, 'type': 'advanced'},
            'conclusion': {'level': 6, 'type': 'synthesis'}
        }

        self.results = {}

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
            sections.append({
                'title': title.strip(),
                'id': section_id,
                'chapter': chapter
            })

        return sections

    def experiment_a_refined_sections(self):
        """Experiment A: Section-level connections (cross-chapter only)"""
        print("\n=== EXPERIMENT A: Refined Section-Level Connections ===")
        start_time = time.time()

        # Collect sections for each chapter
        chapter_sections = {}
        for chapter in self.chapters:
            chapter_sections[chapter] = self.extract_sections(chapter)

        connections = defaultdict(list)

        # Only create cross-chapter connections
        for source_chapter in self.chapters:
            source_concepts = self.load_concept_map(source_chapter)
            source_sections = chapter_sections[source_chapter]

            for target_chapter in self.chapters:
                if source_chapter == target_chapter:
                    continue  # Skip within-chapter connections

                target_concepts = self.load_concept_map(target_chapter)
                target_sections = chapter_sections[target_chapter]

                overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                if strength > 0.02:  # Reasonable threshold
                    # Create connections between relevant sections
                    for source_section in source_sections[:2]:  # Top 2 sections per chapter
                        for target_section in target_sections[:1]:  # Top 1 target section
                            connections[f"{source_chapter}:{source_section['id']}"].append({
                                'target_chapter': target_chapter,
                                'target_section': target_section['id'],
                                'target_title': target_section['title'],
                                'strength': strength,
                                'concepts': overlap[:3]
                            })

        total_connections = sum(len(conns) for conns in connections.values())
        connected_sections = len([k for k, v in connections.items() if v])
        total_sections = sum(len(sections) for sections in chapter_sections.values())

        self.results['experiment_a'] = {
            'total_sections': total_sections,
            'connected_sections': connected_sections,
            'total_connections': total_connections,
            'avg_connections_per_section': total_connections / connected_sections if connected_sections > 0 else 0,
            'section_coverage': connected_sections / total_sections,
            'sample_connections': dict(list(connections.items())[:2]),
            'execution_time': time.time() - start_time
        }

        print(f"Cross-chapter section connections: {total_connections}")
        print(f"Connected sections: {connected_sections}/{total_sections} ({connected_sections/total_sections:.1%})")
        print(f"Avg connections per section: {total_connections / connected_sections if connected_sections > 0 else 0:.1f}")

    def experiment_b_fine_threshold(self):
        """Experiment B: Fine-tuned threshold optimization"""
        print("\n=== EXPERIMENT B: Fine-Tuned Threshold Optimization ===")
        start_time = time.time()

        # Test more granular thresholds in the promising range
        thresholds = [0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03]
        threshold_results = {}

        for threshold in thresholds:
            connections = defaultdict(list)
            quality_scores = []

            for source_chapter in self.chapters:
                source_concepts = self.load_concept_map(source_chapter)

                for target_chapter in self.chapters:
                    if source_chapter == target_chapter:
                        continue

                    target_concepts = self.load_concept_map(target_chapter)
                    overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                    if strength > threshold:
                        # Evaluate connection quality
                        quality = self._evaluate_connection_quality(
                            source_chapter, target_chapter, strength, overlap
                        )

                        connections[source_chapter].append({
                            'target': target_chapter,
                            'strength': strength,
                            'quality': quality,
                            'concepts': overlap[:3]
                        })
                        quality_scores.append(quality)

            total_connections = sum(len(conns) for conns in connections.values())
            coverage = len([ch for ch in self.chapters if connections[ch]]) / len(self.chapters)
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

            # Composite score: balance quantity, coverage, and quality
            composite_score = (coverage * 0.4) + (min(total_connections/50, 1.0) * 0.3) + (avg_quality * 0.3)

            threshold_results[threshold] = {
                'total_connections': total_connections,
                'coverage': coverage,
                'avg_quality': avg_quality,
                'composite_score': composite_score,
                'connections_per_chapter': total_connections / len(self.chapters)
            }

        # Find optimal threshold
        best_threshold = max(threshold_results.keys(),
                           key=lambda t: threshold_results[t]['composite_score'])

        self.results['experiment_b'] = {
            'threshold_analysis': threshold_results,
            'optimal_threshold': best_threshold,
            'optimal_stats': threshold_results[best_threshold],
            'execution_time': time.time() - start_time
        }

        print("Fine-tuned threshold analysis:")
        for threshold, stats in threshold_results.items():
            print(f"  {threshold}: {stats['total_connections']:2d} conn, "
                  f"{stats['coverage']:.1%} cov, qual={stats['avg_quality']:.2f}, "
                  f"score={stats['composite_score']:.3f}")
        print(f"Optimal threshold: {best_threshold} (composite score: {threshold_results[best_threshold]['composite_score']:.3f})")

    def experiment_c_improved_classification(self):
        """Experiment C: Improved pedagogical classification"""
        print("\n=== EXPERIMENT C: Improved Pedagogical Classification ===")
        start_time = time.time()

        connections_by_type = defaultdict(list)

        for source_chapter in self.chapters:
            source_concepts = self.load_concept_map(source_chapter)
            source_progression = self.chapter_progression[source_chapter]

            for target_chapter in self.chapters:
                if source_chapter == target_chapter:
                    continue

                target_concepts = self.load_concept_map(target_chapter)
                target_progression = self.chapter_progression[target_chapter]
                overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                if strength > 0.02:
                    # Enhanced classification using progression map and content analysis
                    conn_type = self._enhanced_classify_connection(
                        source_chapter, target_chapter,
                        source_progression, target_progression,
                        overlap, strength
                    )

                    connections_by_type[conn_type].append({
                        'source': source_chapter,
                        'target': target_chapter,
                        'strength': strength,
                        'concepts': overlap[:3],
                        'source_level': source_progression['level'],
                        'target_level': target_progression['level']
                    })

        # Analyze improved classification
        type_distribution = {t: len(conns) for t, conns in connections_by_type.items()}
        total = sum(type_distribution.values())

        # Quality metrics for classification
        level_consistency = self._analyze_level_consistency(connections_by_type)

        self.results['experiment_c'] = {
            'connection_types_found': list(type_distribution.keys()),
            'type_distribution': type_distribution,
            'type_percentages': {t: (count/total)*100 for t, count in type_distribution.items()},
            'total_connections': total,
            'level_consistency': level_consistency,
            'sample_by_type': {t: conns[:2] for t, conns in connections_by_type.items()},
            'execution_time': time.time() - start_time
        }

        print("Enhanced connection type distribution:")
        for conn_type, count in type_distribution.items():
            percentage = (count/total)*100
            print(f"  {conn_type}: {count} ({percentage:.1f}%)")
        print(f"Level consistency score: {level_consistency:.2f}")

    def experiment_d_asymmetric_bidirectional(self):
        """Experiment D: Asymmetric bidirectional connections"""
        print("\n=== EXPERIMENT D: Asymmetric Bidirectional Connections ===")
        start_time = time.time()

        forward_map = defaultdict(list)  # What each chapter leads to
        backward_map = defaultdict(list)  # What leads to each chapter

        for source_chapter in self.chapters:
            source_concepts = self.load_concept_map(source_chapter)
            source_progression = self.chapter_progression[source_chapter]

            for target_chapter in self.chapters:
                if source_chapter == target_chapter:
                    continue

                target_concepts = self.load_concept_map(target_chapter)
                target_progression = self.chapter_progression[target_chapter]
                overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                if strength > 0.015:
                    # Asymmetric scoring: different perspectives
                    forward_strength = self._calculate_forward_strength(
                        source_progression, target_progression, strength
                    )
                    backward_strength = self._calculate_backward_strength(
                        source_progression, target_progression, strength
                    )

                    # Forward connection (what this leads to)
                    if forward_strength > 0.02:
                        forward_map[source_chapter].append({
                            'target': target_chapter,
                            'strength': forward_strength,
                            'type': 'leads_to',
                            'concepts': overlap[:3]
                        })

                    # Backward connection (what this builds on)
                    if backward_strength > 0.02:
                        backward_map[target_chapter].append({
                            'source': source_chapter,
                            'strength': backward_strength,
                            'type': 'builds_on',
                            'concepts': overlap[:3]
                        })

        # Analyze asymmetric patterns
        forward_total = sum(len(conns) for conns in forward_map.values())
        backward_total = sum(len(conns) for conns in backward_map.values())

        # Find chapters with interesting asymmetric patterns
        asymmetric_examples = []
        for chapter in self.chapters:
            forward_count = len(forward_map[chapter])
            backward_count = len(backward_map[chapter])
            if forward_count > 0 or backward_count > 0:
                ratio = forward_count / (backward_count + 0.1)  # Avoid div by zero
                asymmetric_examples.append({
                    'chapter': chapter,
                    'forward_count': forward_count,
                    'backward_count': backward_count,
                    'asymmetry_ratio': ratio
                })

        self.results['experiment_d'] = {
            'forward_connections': forward_total,
            'backward_connections': backward_total,
            'asymmetry_ratio': forward_total / backward_total if backward_total > 0 else float('inf'),
            'asymmetric_examples': sorted(asymmetric_examples,
                                        key=lambda x: abs(x['asymmetry_ratio'] - 1.0),
                                        reverse=True)[:5],
            'sample_forward': {k: v[:2] for k, v in list(forward_map.items())[:3]},
            'sample_backward': {k: v[:2] for k, v in list(backward_map.items())[:3]},
            'execution_time': time.time() - start_time
        }

        print(f"Forward connections: {forward_total}")
        print(f"Backward connections: {backward_total}")
        print(f"Asymmetry ratio: {forward_total / backward_total if backward_total > 0 else 'inf':.2f}")

    def _find_concept_overlap(self, concepts1: Dict, concepts2: Dict) -> Tuple[List[str], float]:
        """Find overlapping concepts between two concept maps with enhanced scoring"""
        if not concepts1 or not concepts2:
            return [], 0.0

        # Enhanced category weights
        weights = {
            'primary_concepts': 1.0,
            'technical_terms': 0.8,
            'methodologies': 0.9,  # Higher weight for methods
            'secondary_concepts': 0.6,
            'applications': 0.7,   # Higher weight for applications
            'keywords': 0.3
        }

        all_concepts1 = []
        all_concepts2 = []

        for category, weight in weights.items():
            concepts_cat1 = concepts1.get('concept_map', {}).get(category, [])
            concepts_cat2 = concepts2.get('concept_map', {}).get(category, [])

            if isinstance(concepts_cat1, list):
                all_concepts1.extend([(c.lower(), weight) for c in concepts_cat1])
            if isinstance(concepts_cat2, list):
                all_concepts2.extend([(c.lower(), weight) for c in concepts_cat2])

        # Find overlaps with enhanced scoring
        concepts1_dict = {c[0]: c[1] for c in all_concepts1}
        concepts2_dict = {c[0]: c[1] for c in all_concepts2}

        overlapping = []
        total_strength = 0

        for concept, weight1 in concepts1_dict.items():
            if concept in concepts2_dict:
                weight2 = concepts2_dict[concept]
                overlapping.append(concept)
                # Enhanced strength calculation
                total_strength += (weight1 + weight2) / 2

        # Better normalization
        total_concepts = len(set(concepts1_dict.keys()) | set(concepts2_dict.keys()))
        normalized_strength = total_strength / total_concepts if total_concepts > 0 else 0

        return overlapping, normalized_strength

    def _evaluate_connection_quality(self, source: str, target: str, strength: float, concepts: List[str]) -> float:
        """Evaluate the pedagogical quality of a connection"""
        source_prog = self.chapter_progression[source]
        target_prog = self.chapter_progression[target]

        quality_score = 0.0

        # Level progression bonus (learning should build progressively)
        level_diff = target_prog['level'] - source_prog['level']
        if 0 <= level_diff <= 2:
            quality_score += 0.3  # Good progression
        elif level_diff > 2:
            quality_score += 0.1  # Too big a jump
        else:
            quality_score += 0.2  # Backward reference

        # Type compatibility bonus
        compatible_transitions = {
            'foundation': ['theory', 'methodology'],
            'theory': ['implementation', 'methodology'],
            'methodology': ['implementation', 'optimization'],
            'implementation': ['optimization', 'specialization'],
            'optimization': ['specialization', 'application'],
            'specialization': ['application', 'advanced'],
            'application': ['advanced', 'synthesis'],
            'advanced': ['synthesis'],
        }

        if target_prog['type'] in compatible_transitions.get(source_prog['type'], []):
            quality_score += 0.4

        # Concept overlap bonus
        if len(concepts) >= 3:
            quality_score += 0.2
        elif len(concepts) >= 1:
            quality_score += 0.1

        # Strength bonus
        quality_score += min(strength * 2, 0.3)

        return min(quality_score, 1.0)

    def _enhanced_classify_connection(self, source: str, target: str,
                                    source_prog: Dict, target_prog: Dict,
                                    concepts: List[str], strength: float) -> str:
        """Enhanced pedagogical connection classification"""

        level_diff = target_prog['level'] - source_prog['level']
        source_type = source_prog['type']
        target_type = target_prog['type']

        # Level-based classification
        if level_diff == 1:
            if source_type == 'foundation' and target_type == 'theory':
                return 'foundation_to_theory'
            elif source_type == 'theory' and target_type == 'implementation':
                return 'theory_to_practice'
            elif source_type == 'implementation' and target_type == 'optimization':
                return 'practice_to_optimization'
            else:
                return 'sequential_progression'

        elif level_diff == 0:
            if source_type == target_type:
                return 'peer_concept'
            else:
                return 'complementary_approach'

        elif level_diff < 0:
            return 'builds_on_foundation'

        elif level_diff > 2:
            return 'advanced_application'

        else:
            # Content-based fallback
            if strength > 0.08:
                return 'strong_conceptual_link'
            elif any('optimization' in c for c in concepts):
                return 'optimization_related'
            elif any('system' in c for c in concepts):
                return 'systems_related'
            else:
                return 'topical_connection'

    def _calculate_forward_strength(self, source_prog: Dict, target_prog: Dict, base_strength: float) -> float:
        """Calculate forward connection strength (what this leads to)"""
        level_diff = target_prog['level'] - source_prog['level']

        # Forward connections stronger for natural progressions
        if level_diff == 1:
            return base_strength * 1.2
        elif level_diff > 1:
            return base_strength * 0.8
        else:
            return base_strength * 0.6

    def _calculate_backward_strength(self, source_prog: Dict, target_prog: Dict, base_strength: float) -> float:
        """Calculate backward connection strength (what this builds on)"""
        level_diff = target_prog['level'] - source_prog['level']

        # Backward connections stronger for prerequisite relationships
        if level_diff == -1:
            return base_strength * 1.3
        elif level_diff < -1:
            return base_strength * 1.0
        else:
            return base_strength * 0.5

    def _analyze_level_consistency(self, connections_by_type: Dict) -> float:
        """Analyze how well connection types align with level progression"""
        consistent_connections = 0
        total_connections = 0

        for conn_type, connections in connections_by_type.items():
            for conn in connections:
                total_connections += 1

                # Check if connection type makes sense given level progression
                level_diff = conn['target_level'] - conn['source_level']

                if (('foundation' in conn_type and level_diff >= 0) or
                    ('builds_on' in conn_type and level_diff <= 0) or
                    ('progression' in conn_type and 0 <= level_diff <= 2) or
                    ('peer' in conn_type and level_diff == 0)):
                    consistent_connections += 1

        return consistent_connections / total_connections if total_connections > 0 else 0.0

    def generate_refined_report(self):
        """Generate report on refined experiments"""
        print("\n" + "="*70)
        print("REFINED CROSS-REFERENCE EXPERIMENTAL RESULTS")
        print("="*70)

        total_time = sum(result.get('execution_time', 0)
                        for result in self.results.values()
                        if isinstance(result, dict))

        print(f"\nTotal refined experimental time: {total_time:.1f} seconds")

        # Key findings
        findings = []

        if 'experiment_a' in self.results:
            exp_a = self.results['experiment_a']
            findings.append(f"Cross-chapter section connections: {exp_a['total_connections']} "
                          f"({exp_a['section_coverage']:.1%} section coverage)")

        if 'experiment_b' in self.results:
            exp_b = self.results['experiment_b']
            optimal = exp_b['optimal_threshold']
            score = exp_b['optimal_stats']['composite_score']
            findings.append(f"Optimal threshold: {optimal} (composite quality score: {score:.3f})")

        if 'experiment_c' in self.results:
            exp_c = self.results['experiment_c']
            types_found = len(exp_c['connection_types_found'])
            consistency = exp_c['level_consistency']
            findings.append(f"Enhanced classification: {types_found} connection types "
                          f"(level consistency: {consistency:.2f})")

        if 'experiment_d' in self.results:
            exp_d = self.results['experiment_d']
            asymmetry = exp_d['asymmetry_ratio']
            findings.append(f"Bidirectional asymmetry ratio: {asymmetry:.2f}")

        print("\nKEY REFINED FINDINGS:")
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding}")

        # Save results
        results_file = "/Users/VJ/GitHub/MLSysBook/tools/scripts/cross_refs/refined_experimental_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nRefined results saved to: {results_file}")

    def run_refined_experiments(self):
        """Run all refined experiments"""
        print("Starting refined cross-reference experiments...")

        self.experiment_a_refined_sections()
        self.experiment_b_fine_threshold()
        self.experiment_c_improved_classification()
        self.experiment_d_asymmetric_bidirectional()

        self.generate_refined_report()

if __name__ == "__main__":
    experiments = RefinedCrossRefExperiments()
    experiments.run_refined_experiments()
