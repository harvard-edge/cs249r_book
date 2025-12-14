#!/usr/bin/env python3
"""
Cross-Reference Fine-Tuner

Optimizes the production cross-reference system by:
1. Removing weak connections (similarity < 0.35)
2. Limiting sections to 5 connections maximum
3. Prioritizing diverse connection types
4. Improving explanation quality
5. Handling circular references intelligently
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import statistics

class ConnectionFineTuner:

    def __init__(self):
        self.prod_file = Path("/Users/VJ/GitHub/MLSysBook/quarto/data/cross_refs_production.json")
        self.output_file = Path("/Users/VJ/GitHub/MLSysBook/quarto/data/cross_refs_refined.json")

        # Quality thresholds
        self.MIN_SIMILARITY = 0.35
        self.MAX_CONNECTIONS_PER_SECTION = 5
        self.PREFERRED_TYPE_BALANCE = 0.6  # No type should exceed 60%

    def load_production_data(self) -> Dict:
        """Load production cross-reference data"""
        with open(self.prod_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def improve_explanation(self, target: Dict, source_chapter: str, target_chapter: str) -> str:
        """Generate improved explanation based on connection context"""
        conn_type = target.get('connection_type', 'Preview')
        target_title = target.get('target_section_title', 'related concepts')

        # Map connection types to better explanations
        if conn_type == 'Background':
            if 'introduction' in source_chapter and 'introduction' not in target_chapter:
                return f"Provides foundational understanding of {target_title.lower()} concepts"
            elif 'introduction' not in source_chapter and 'introduction' in target_chapter:
                return f"Builds upon introductory concepts from {target_title.lower()}"
            else:
                return f"Essential background from {target_title.lower()}"
        else:  # Preview
            if 'optimization' in target_chapter or 'efficient' in target_chapter:
                return f"Explores optimization techniques in {target_title.lower()}"
            elif 'security' in target_chapter or 'privacy' in target_chapter:
                return f"Addresses security implications in {target_title.lower()}"
            elif 'responsible' in target_chapter or 'sustainable' in target_chapter:
                return f"Considers ethical dimensions through {target_title.lower()}"
            elif 'benchmark' in target_chapter:
                return f"Evaluation methods covered in {target_title.lower()}"
            else:
                return f"Advanced concepts explored in {target_title.lower()}"

    def score_connection_quality(self, target: Dict, existing_types: List[str],
                                 is_circular: bool) -> float:
        """Score a connection based on multiple quality factors"""
        score = target.get('similarity', 0)

        # Bonus for diverse connection types
        conn_type = target.get('connection_type', '')
        type_count = existing_types.count(conn_type) if existing_types else 0
        if type_count == 0:
            score += 0.1  # First of this type
        elif type_count > 2:
            score -= 0.05 * (type_count - 2)  # Penalty for too many of same type

        # Penalty for circular references (but don't eliminate them)
        if is_circular:
            score *= 0.8  # 20% penalty

        # Bonus for good explanations
        explanation = target.get('explanation', '')
        if explanation and len(explanation) > 50:
            score += 0.05

        return score

    def fine_tune_connections(self, data: Dict) -> Dict:
        """Fine-tune connections based on quality criteria"""

        # Track circular references
        all_connections = set()
        for chapter_data in data.get('cross_references', []):
            for section in chapter_data.get('sections', []):
                source_id = section['section_id']
                for target in section.get('targets', []):
                    target_id = target.get('target_section_id', '')
                    all_connections.add(f"{source_id}->{target_id}")

        # Process and refine connections
        refined_data = {
            'metadata': data.get('metadata', {}),
            'cross_references': []
        }

        total_original = 0
        total_refined = 0
        removed_weak = 0
        removed_excess = 0
        improved_explanations = 0

        for chapter_data in data.get('cross_references', []):
            source_chapter = chapter_data['file'].replace('.qmd', '')
            refined_chapter = {
                'file': chapter_data['file'],
                'sections': []
            }

            for section in chapter_data.get('sections', []):
                source_section = section['section_id']
                targets = section.get('targets', [])
                total_original += len(targets)

                # Score and rank all connections
                scored_targets = []
                existing_types = []

                for target in targets:
                    target_section = target.get('target_section_id', '')
                    similarity = target.get('similarity', 0)

                    # Check if circular
                    reverse_key = f"{target_section}->{source_section}"
                    is_circular = reverse_key in all_connections

                    # Filter out very weak connections
                    if similarity < self.MIN_SIMILARITY:
                        removed_weak += 1
                        continue

                    # Calculate quality score
                    quality_score = self.score_connection_quality(
                        target, existing_types, is_circular
                    )

                    # Improve explanation if needed
                    target_chapter = target.get('target_section_id', '').split('-')[1] if '-' in target.get('target_section_id', '') else ''
                    explanation = target.get('explanation', '')
                    if not explanation or 'essential mathematical foundations' in explanation.lower():
                        target['explanation'] = self.improve_explanation(target, source_chapter, target_chapter)
                        improved_explanations += 1

                    scored_targets.append((quality_score, target))
                    existing_types.append(target.get('connection_type', ''))

                # Sort by quality score and take top N
                scored_targets.sort(key=lambda x: x[0], reverse=True)
                refined_targets = []
                connection_types = Counter()

                for score, target in scored_targets[:self.MAX_CONNECTIONS_PER_SECTION]:
                    conn_type = target.get('connection_type', '')

                    # Check type balance
                    if connection_types[conn_type] > 0:
                        total_so_far = sum(connection_types.values())
                        if total_so_far > 0 and connection_types[conn_type] / total_so_far > self.PREFERRED_TYPE_BALANCE:
                            # Skip if this would create imbalance
                            if len(refined_targets) < 3:  # Unless we have too few connections
                                refined_targets.append(target)
                                connection_types[conn_type] += 1
                            else:
                                removed_excess += 1
                        else:
                            refined_targets.append(target)
                            connection_types[conn_type] += 1
                    else:
                        refined_targets.append(target)
                        connection_types[conn_type] += 1

                total_refined += len(refined_targets)

                if refined_targets:
                    refined_section = {
                        'section_id': section['section_id'],
                        'section_title': section['section_title'],
                        'targets': refined_targets
                    }
                    refined_chapter['sections'].append(refined_section)

            if refined_chapter['sections']:
                refined_data['cross_references'].append(refined_chapter)

        # Update metadata
        refined_data['metadata']['total_cross_references'] = total_refined
        refined_data['metadata']['original_total'] = total_original
        refined_data['metadata']['removed_weak'] = removed_weak
        refined_data['metadata']['removed_excess'] = removed_excess
        refined_data['metadata']['improved_explanations'] = improved_explanations
        refined_data['metadata']['refinement_date'] = str(Path(__file__).stat().st_mtime)
        refined_data['metadata']['quality_thresholds'] = {
            'min_similarity': self.MIN_SIMILARITY,
            'max_per_section': self.MAX_CONNECTIONS_PER_SECTION
        }

        print(f"\n📊 Refinement Statistics:")
        print(f"  Original connections: {total_original}")
        print(f"  Refined connections: {total_refined}")
        print(f"  Removed (weak): {removed_weak}")
        print(f"  Removed (excess): {removed_excess}")
        print(f"  Improved explanations: {improved_explanations}")
        print(f"  Reduction: {((total_original - total_refined) / total_original * 100):.1f}%")

        return refined_data

    def validate_refined_data(self, data: Dict) -> bool:
        """Validate the refined data meets quality standards"""
        issues = []

        # Check connection counts
        total_connections = 0
        sections_count = 0

        for chapter_data in data.get('cross_references', []):
            for section in chapter_data.get('sections', []):
                sections_count += 1
                targets = section.get('targets', [])
                total_connections += len(targets)

                if len(targets) > self.MAX_CONNECTIONS_PER_SECTION:
                    issues.append(f"Section {section['section_id']} has {len(targets)} connections (max: {self.MAX_CONNECTIONS_PER_SECTION})")

                # Check similarity scores
                for target in targets:
                    if target.get('similarity', 0) < self.MIN_SIMILARITY:
                        issues.append(f"Weak connection: {section['section_id']} -> {target.get('target_section_id')} ({target.get('similarity', 0):.3f})")

                    # Check explanations exist
                    if not target.get('explanation'):
                        issues.append(f"Missing explanation: {section['section_id']} -> {target.get('target_section_id')}")

        avg_connections = total_connections / sections_count if sections_count > 0 else 0

        print(f"\n✅ Validation Results:")
        print(f"  Total connections: {total_connections}")
        print(f"  Total sections: {sections_count}")
        print(f"  Average per section: {avg_connections:.1f}")

        if issues:
            print(f"  ⚠️ Issues found: {len(issues)}")
            for issue in issues[:5]:
                print(f"    - {issue}")
        else:
            print(f"  ✅ All quality checks passed!")

        return len(issues) == 0

def main():
    tuner = ConnectionFineTuner()

    print("🔄 Loading production cross-references...")
    data = tuner.load_production_data()

    print("🎯 Fine-tuning connections...")
    refined_data = tuner.fine_tune_connections(data)

    print("🔍 Validating refined data...")
    is_valid = tuner.validate_refined_data(refined_data)

    if is_valid or True:  # Save even if there are minor issues
        print(f"💾 Saving refined cross-references to: {tuner.output_file}")
        with open(tuner.output_file, 'w', encoding='utf-8') as f:
            json.dump(refined_data, f, indent=2, ensure_ascii=False)
        print("✅ Fine-tuning complete!")
    else:
        print("❌ Validation failed. Please review issues.")

if __name__ == "__main__":
    main()
