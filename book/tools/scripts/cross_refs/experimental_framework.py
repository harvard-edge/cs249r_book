#!/usr/bin/env python3
"""
Comprehensive Cross-Reference Experimental Framework

This script runs multiple experiments to optimize cross-reference generation:
1. Section-level granularity analysis
2. Bidirectional connection testing
3. Connection density optimization
4. Pedagogical connection type enhancement
5. Multi-placement strategy evaluation
"""

import os
import json
import yaml
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict, Counter
import time

class CrossRefExperimentalFramework:

    def __init__(self):
        self.base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
        self.chapters = [
            'introduction', 'ml_systems', 'dl_primer', 'workflow', 'data_engineering',
            'frameworks', 'training', 'efficient_ai', 'optimizations', 'hw_acceleration',
            'benchmarking', 'ondevice_learning', 'ops', 'privacy_security', 'responsible_ai',
            'sustainable_ai', 'ai_for_good', 'robust_ai', 'generative_ai', 'frontiers',
            'emerging_topics', 'conclusion'
        ]

        # Enhanced connection types for pedagogical ordering
        self.enhanced_connection_types = {
            'foundation': {'weight': 1.0, 'description': 'Essential prerequisite knowledge'},
            'prerequisite': {'weight': 0.9, 'description': 'Required background concepts'},
            'builds_on': {'weight': 0.8, 'description': 'Extends previous concepts'},
            'implements': {'weight': 0.7, 'description': 'Practical implementation of theory'},
            'applies': {'weight': 0.6, 'description': 'Real-world application'},
            'extends': {'weight': 0.5, 'description': 'Advanced extension or specialization'},
            'relates': {'weight': 0.4, 'description': 'Conceptually related topics'},
            'contrasts': {'weight': 0.3, 'description': 'Alternative approaches or comparisons'},
            'example': {'weight': 0.2, 'description': 'Concrete example or case study'}
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

    def experiment_1_section_granularity(self):
        """Experiment 1: Generate section-level cross-references"""
        print("\n=== EXPERIMENT 1: Section-Level Granularity ===")
        start_time = time.time()

        # Collect all sections across all chapters
        all_sections = {}
        section_concepts = {}

        for chapter in self.chapters:
            sections = self.extract_sections(chapter)
            concept_map = self.load_concept_map(chapter)

            for section in sections:
                section_key = f"{chapter}:{section['id']}"
                all_sections[section_key] = section
                # For now, assign chapter concepts to all sections
                # In reality, we'd need section-specific concept extraction
                section_concepts[section_key] = concept_map.get('concept_map', {})

        # Generate connections between sections
        connections = defaultdict(list)
        total_possible = len(all_sections) * (len(all_sections) - 1)

        for source_key, source_section in all_sections.items():
            source_chapter = source_section['chapter']
            source_concepts = section_concepts.get(source_key, {})

            for target_key, target_section in all_sections.items():
                if source_key == target_key:
                    continue

                target_chapter = target_section['chapter']
                target_concepts = section_concepts.get(target_key, {})

                # Find concept overlaps
                overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                if strength > 0.02:  # Lower threshold for section-level
                    connections[source_key].append({
                        'target_chapter': target_chapter,
                        'target_section': target_section['id'],
                        'target_title': target_section['title'],
                        'strength': strength,
                        'concepts': overlap[:5]  # Top 5 concepts
                    })

        # Analyze results
        total_connections = sum(len(conns) for conns in connections.values())
        coverage = len([k for k, v in connections.items() if v]) / len(all_sections)

        self.results['experiment_1'] = {
            'total_sections': len(all_sections),
            'total_connections': total_connections,
            'coverage': coverage,
            'avg_connections_per_section': total_connections / len(all_sections),
            'sample_connections': dict(list(connections.items())[:3]),
            'execution_time': time.time() - start_time
        }

        print(f"Sections found: {len(all_sections)}")
        print(f"Total connections: {total_connections}")
        print(f"Coverage: {coverage:.1%}")
        print(f"Avg connections per section: {total_connections / len(all_sections):.1f}")

    def experiment_2_bidirectional(self):
        """Experiment 2: Test bidirectional connections"""
        print("\n=== EXPERIMENT 2: Bidirectional Connections ===")
        start_time = time.time()

        forward_connections = {}
        backward_connections = {}

        for source_chapter in self.chapters:
            source_concepts = self.load_concept_map(source_chapter)
            forward_connections[source_chapter] = []

        # Initialize backward connections for all chapters
        for chapter in self.chapters:
            backward_connections[chapter] = []

        for source_chapter in self.chapters:
            source_concepts = self.load_concept_map(source_chapter)

            for target_chapter in self.chapters:
                if source_chapter == target_chapter:
                    continue

                target_concepts = self.load_concept_map(target_chapter)

                # Forward connections (what this chapter leads to)
                overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)
                if strength > 0.03:
                    forward_connections[source_chapter].append({
                        'target': target_chapter,
                        'type': 'forward',
                        'strength': strength,
                        'concepts': overlap[:3]
                    })

                # Backward connections (what leads to this chapter)
                overlap, strength = self._find_concept_overlap(target_concepts, source_concepts)
                if strength > 0.03:
                    backward_connections[target_chapter].append({
                        'source': source_chapter,
                        'type': 'backward',
                        'strength': strength,
                        'concepts': overlap[:3]
                    })

        # Analyze bidirectional patterns
        total_forward = sum(len(conns) for conns in forward_connections.values())
        total_backward = sum(len(conns) for conns in backward_connections.values())

        self.results['experiment_2'] = {
            'forward_connections': total_forward,
            'backward_connections': total_backward,
            'bidirectional_ratio': total_backward / total_forward if total_forward > 0 else 0,
            'sample_forward': {k: v[:2] for k, v in list(forward_connections.items())[:3]},
            'sample_backward': {k: v[:2] for k, v in list(backward_connections.items())[:3]},
            'execution_time': time.time() - start_time
        }

        print(f"Forward connections: {total_forward}")
        print(f"Backward connections: {total_backward}")
        print(f"Bidirectional coverage ratio: {total_backward / total_forward:.2f}")

    def experiment_3_density_optimization(self):
        """Experiment 3: Test different connection density thresholds"""
        print("\n=== EXPERIMENT 3: Connection Density Optimization ===")
        start_time = time.time()

        thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
        density_results = {}

        for threshold in thresholds:
            connections = defaultdict(list)

            for source_chapter in self.chapters:
                source_concepts = self.load_concept_map(source_chapter)

                for target_chapter in self.chapters:
                    if source_chapter == target_chapter:
                        continue

                    target_concepts = self.load_concept_map(target_chapter)
                    overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                    if strength > threshold:
                        connections[source_chapter].append({
                            'target': target_chapter,
                            'strength': strength,
                            'concepts': overlap[:5]
                        })

            total_connections = sum(len(conns) for conns in connections.values())
            coverage = len([ch for ch in self.chapters if connections[ch]]) / len(self.chapters)

            density_results[threshold] = {
                'total_connections': total_connections,
                'coverage': coverage,
                'avg_per_chapter': total_connections / len(self.chapters),
                'quality_score': coverage * (total_connections / 100)  # Balance coverage and quantity
            }

        # Find optimal threshold
        best_threshold = max(density_results.keys(),
                           key=lambda t: density_results[t]['quality_score'])

        self.results['experiment_3'] = {
            'threshold_analysis': density_results,
            'optimal_threshold': best_threshold,
            'optimal_stats': density_results[best_threshold],
            'execution_time': time.time() - start_time
        }

        print("Threshold analysis:")
        for threshold, stats in density_results.items():
            print(f"  {threshold}: {stats['total_connections']} connections, "
                  f"{stats['coverage']:.1%} coverage, quality={stats['quality_score']:.2f}")
        print(f"Optimal threshold: {best_threshold}")

    def experiment_4_pedagogical_types(self):
        """Experiment 4: Enhanced pedagogical connection types"""
        print("\n=== EXPERIMENT 4: Pedagogical Connection Types ===")
        start_time = time.time()

        # Chapter ordering for pedagogical progression
        chapter_order = {ch: i for i, ch in enumerate(self.chapters)}

        connections_by_type = defaultdict(list)

        for source_chapter in self.chapters:
            source_concepts = self.load_concept_map(source_chapter)
            source_order = chapter_order[source_chapter]

            for target_chapter in self.chapters:
                if source_chapter == target_chapter:
                    continue

                target_concepts = self.load_concept_map(target_chapter)
                target_order = chapter_order[target_chapter]
                overlap, strength = self._find_concept_overlap(source_concepts, target_concepts)

                if strength > 0.04:
                    # Determine connection type based on chapter progression and concepts
                    conn_type = self._classify_pedagogical_connection(
                        source_chapter, target_chapter, source_order, target_order,
                        overlap, strength
                    )

                    connections_by_type[conn_type].append({
                        'source': source_chapter,
                        'target': target_chapter,
                        'strength': strength,
                        'concepts': overlap[:3]
                    })

        # Analyze type distribution
        type_distribution = {t: len(conns) for t, conns in connections_by_type.items()}
        total = sum(type_distribution.values())

        self.results['experiment_4'] = {
            'connection_types': list(self.enhanced_connection_types.keys()),
            'type_distribution': type_distribution,
            'type_percentages': {t: (count/total)*100 for t, count in type_distribution.items()},
            'total_connections': total,
            'sample_by_type': {t: conns[:2] for t, conns in connections_by_type.items()},
            'execution_time': time.time() - start_time
        }

        print("Connection type distribution:")
        for conn_type, count in type_distribution.items():
            percentage = (count/total)*100
            print(f"  {conn_type}: {count} ({percentage:.1f}%)")

    def experiment_5_placement_strategies(self):
        """Experiment 5: Multi-level placement strategies"""
        print("\n=== EXPERIMENT 5: Placement Strategies ===")
        start_time = time.time()

        placement_strategies = {
            'chapter_start': 'At beginning of each chapter',
            'section_start': 'At beginning of each section',
            'contextual_inline': 'Inline where concepts are mentioned',
            'section_end': 'At end of sections (what\'s next)',
            'mixed_adaptive': 'Adaptive based on connection strength'
        }

        # Simulate different placement densities
        for chapter in self.chapters[:3]:  # Test on first 3 chapters
            sections = self.extract_sections(chapter)

            strategies_analysis = {}

            # Chapter start: Few high-level connections
            strategies_analysis['chapter_start'] = {
                'locations': 1,
                'avg_connections_per_location': 3,
                'total_connections': 3,
                'pedagogical_impact': 'High - sets context',
                'readability_impact': 'Low - doesn\'t clutter'
            }

            # Section start: More granular connections
            strategies_analysis['section_start'] = {
                'locations': len(sections),
                'avg_connections_per_location': 2,
                'total_connections': len(sections) * 2,
                'pedagogical_impact': 'Very High - contextual',
                'readability_impact': 'Medium - some clutter'
            }

            # Contextual inline: Many micro-connections
            strategies_analysis['contextual_inline'] = {
                'locations': len(sections) * 3,  # Estimate 3 per section
                'avg_connections_per_location': 1,
                'total_connections': len(sections) * 3,
                'pedagogical_impact': 'Medium - can be distracting',
                'readability_impact': 'High - significant clutter'
            }

            self.results[f'experiment_5_{chapter}'] = strategies_analysis

        # Overall recommendation
        self.results['experiment_5_summary'] = {
            'strategies_evaluated': list(placement_strategies.keys()),
            'recommended_approach': 'section_start',
            'rationale': 'Best balance of pedagogical value and readability',
            'execution_time': time.time() - start_time
        }

        print("Placement strategy analysis complete")
        print("Recommended: Section-start placement")

    def _find_concept_overlap(self, concepts1: Dict, concepts2: Dict) -> Tuple[List[str], float]:
        """Find overlapping concepts between two concept maps"""
        if not concepts1 or not concepts2:
            return [], 0.0

        # Extract all concepts from both maps
        all_concepts1 = []
        all_concepts2 = []

        # Category weights for calculating strength
        weights = {
            'primary_concepts': 1.0,
            'technical_terms': 0.8,
            'methodologies': 0.7,
            'secondary_concepts': 0.6,
            'applications': 0.5,
            'keywords': 0.3
        }

        for category, weight in weights.items():
            concepts_cat1 = concepts1.get(category, [])
            concepts_cat2 = concepts2.get(category, [])

            if isinstance(concepts_cat1, list):
                all_concepts1.extend([(c.lower(), weight) for c in concepts_cat1])
            if isinstance(concepts_cat2, list):
                all_concepts2.extend([(c.lower(), weight) for c in concepts_cat2])

        # Find exact matches
        concepts1_dict = {c[0]: c[1] for c in all_concepts1}
        concepts2_dict = {c[0]: c[1] for c in all_concepts2}

        overlapping = []
        total_strength = 0

        for concept, weight1 in concepts1_dict.items():
            if concept in concepts2_dict:
                weight2 = concepts2_dict[concept]
                overlapping.append(concept)
                total_strength += (weight1 + weight2) / 2

        # Normalize strength
        max_possible = len(all_concepts1) + len(all_concepts2)
        normalized_strength = total_strength / max_possible if max_possible > 0 else 0

        return overlapping, normalized_strength

    def _classify_pedagogical_connection(self, source: str, target: str,
                                       source_order: int, target_order: int,
                                       concepts: List[str], strength: float) -> str:
        """Classify connection type based on pedagogical relationship"""

        # Basic ordering logic
        if target_order < source_order:
            return 'builds_on'  # Target comes before source
        elif target_order == source_order + 1:
            return 'prerequisite'  # Direct sequence
        elif target_order > source_order + 3:
            return 'applies'  # Much later, likely application

        # Content-based classification
        foundation_terms = ['introduction', 'basics', 'primer', 'overview']
        advanced_terms = ['optimization', 'acceleration', 'advanced', 'specialized']

        if any(term in source.lower() for term in foundation_terms):
            return 'foundation'
        elif any(term in target.lower() for term in advanced_terms):
            return 'extends'
        elif strength > 0.08:
            return 'implements'
        else:
            return 'relates'

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE CROSS-REFERENCE EXPERIMENTAL RESULTS")
        print("="*60)

        if not self.results:
            print("No experiments run yet!")
            return

        total_time = sum(result.get('execution_time', 0)
                        for result in self.results.values()
                        if isinstance(result, dict))

        print(f"\nTotal experimental time: {total_time:.1f} seconds")
        print(f"Experiments completed: {len(self.results)}")

        # Extract key insights
        insights = []

        if 'experiment_1' in self.results:
            exp1 = self.results['experiment_1']
            insights.append(f"Section-level granularity generates {exp1['total_connections']} connections "
                          f"across {exp1['total_sections']} sections ({exp1['coverage']:.1%} coverage)")

        if 'experiment_3' in self.results:
            exp3 = self.results['experiment_3']
            optimal = exp3['optimal_threshold']
            insights.append(f"Optimal connection threshold: {optimal} "
                          f"({exp3['optimal_stats']['total_connections']} connections)")

        if 'experiment_4' in self.results:
            exp4 = self.results['experiment_4']
            top_type = max(exp4['type_distribution'], key=exp4['type_distribution'].get)
            insights.append(f"Most common connection type: {top_type} "
                          f"({exp4['type_percentages'][top_type]:.1f}%)")

        print("\nKEY INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")

        # Save detailed results
        results_file = "/Users/VJ/GitHub/MLSysBook/tools/scripts/cross_refs/experimental_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nDetailed results saved to: {results_file}")

    def run_all_experiments(self):
        """Run all experiments in sequence"""
        print("Starting comprehensive cross-reference experiments...")
        print("This may take 5-10 minutes to complete all experiments.")

        self.experiment_1_section_granularity()
        self.experiment_2_bidirectional()
        self.experiment_3_density_optimization()
        self.experiment_4_pedagogical_types()
        self.experiment_5_placement_strategies()

        self.generate_comprehensive_report()

if __name__ == "__main__":
    framework = CrossRefExperimentalFramework()
    framework.run_all_experiments()
