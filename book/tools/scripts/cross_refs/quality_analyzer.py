#!/usr/bin/env python3
"""
Cross-Reference Quality Analyzer and Fine-Tuner

Analyzes the production cross-reference system to identify:
1. Connection quality distribution
2. Redundant or low-value connections
3. Explanation clarity issues
4. Optimal connection density
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import statistics

class QualityAnalyzer:

    def __init__(self):
        self.prod_file = Path("/Users/VJ/GitHub/MLSysBook/quarto/data/cross_refs_production.json")
        self.base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")

    def load_production_data(self) -> Dict:
        """Load production cross-reference data"""
        with open(self.prod_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_connection_distribution(self, data: Dict) -> Dict:
        """Analyze how connections are distributed"""
        stats = {
            'total_connections': 0,
            'connections_per_chapter': defaultdict(int),
            'connections_per_section': defaultdict(int),
            'connection_types': Counter(),
            'similarity_distribution': [],
            'sections_with_many_connections': [],
            'sections_with_few_connections': [],
            'explanation_lengths': []
        }

        for chapter_data in data.get('cross_references', []):
            chapter_name = chapter_data['file'].replace('.qmd', '')

            for section in chapter_data.get('sections', []):
                section_id = section['section_id']
                targets = section.get('targets', [])
                num_connections = len(targets)

                stats['connections_per_chapter'][chapter_name] += num_connections
                stats['connections_per_section'][section_id] = num_connections
                stats['total_connections'] += num_connections

                if num_connections > 7:  # Too many connections
                    stats['sections_with_many_connections'].append({
                        'section': section_id,
                        'title': section['section_title'],
                        'count': num_connections
                    })
                elif num_connections < 3:  # Too few connections
                    stats['sections_with_few_connections'].append({
                        'section': section_id,
                        'title': section['section_title'],
                        'count': num_connections
                    })

                for target in targets:
                    stats['connection_types'][target.get('connection_type', 'Unknown')] += 1
                    stats['similarity_distribution'].append(target.get('similarity', 0))

                    explanation = target.get('explanation', '')
                    if explanation:
                        stats['explanation_lengths'].append(len(explanation))

        return stats

    def identify_quality_issues(self, data: Dict) -> Dict:
        """Identify specific quality issues"""
        issues = {
            'redundant_connections': [],  # Same target appearing multiple times
            'weak_connections': [],  # Very low similarity scores
            'missing_explanations': [],  # Connections without explanations
            'generic_explanations': [],  # Explanations that are too generic
            'circular_references': [],  # A->B and B->A connections
            'unbalanced_types': []  # Sections with only one connection type
        }

        # Track all connections for redundancy checking
        connection_pairs = defaultdict(list)

        for chapter_data in data.get('cross_references', []):
            source_chapter = chapter_data['file'].replace('.qmd', '')

            for section in chapter_data.get('sections', []):
                source_section = section['section_id']
                connection_types_in_section = set()

                for target in section.get('targets', []):
                    target_section = target.get('target_section_id', '')
                    similarity = target.get('similarity', 0)
                    explanation = target.get('explanation', '')
                    conn_type = target.get('connection_type', '')

                    # Track connection pair
                    pair_key = f"{source_section}->{target_section}"
                    connection_pairs[pair_key].append({
                        'source': source_section,
                        'target': target_section,
                        'type': conn_type,
                        'similarity': similarity
                    })

                    connection_types_in_section.add(conn_type)

                    # Check for weak connections
                    if similarity < 0.3:
                        issues['weak_connections'].append({
                            'source': source_section,
                            'target': target_section,
                            'similarity': similarity,
                            'type': conn_type
                        })

                    # Check for missing or generic explanations
                    if not explanation:
                        issues['missing_explanations'].append({
                            'source': source_section,
                            'target': target_section
                        })
                    elif 'essential mathematical foundations' in explanation.lower():
                        issues['generic_explanations'].append({
                            'source': source_section,
                            'target': target_section,
                            'explanation': explanation
                        })

                # Check for unbalanced connection types
                if len(connection_types_in_section) == 1 and len(section.get('targets', [])) > 3:
                    issues['unbalanced_types'].append({
                        'section': source_section,
                        'type': list(connection_types_in_section)[0],
                        'count': len(section.get('targets', []))
                    })

        # Check for redundant connections
        for pair_key, connections in connection_pairs.items():
            if len(connections) > 1:
                issues['redundant_connections'].append({
                    'pair': pair_key,
                    'count': len(connections),
                    'connections': connections
                })

        # Check for circular references
        for pair_key in connection_pairs:
            source, target = pair_key.split('->')
            reverse_key = f"{target}->{source}"
            if reverse_key in connection_pairs:
                issues['circular_references'].append({
                    'forward': pair_key,
                    'reverse': reverse_key
                })

        return issues

    def generate_quality_report(self) -> str:
        """Generate comprehensive quality report"""
        data = self.load_production_data()
        stats = self.analyze_connection_distribution(data)
        issues = self.identify_quality_issues(data)

        report = []
        report.append("# Cross-Reference Quality Analysis Report\n")
        report.append(f"**Total Connections**: {stats['total_connections']}\n")

        # Distribution Statistics
        report.append("\n## 📊 Connection Distribution\n")

        # Connections per chapter
        report.append("### Connections by Chapter\n")
        sorted_chapters = sorted(stats['connections_per_chapter'].items(),
                                key=lambda x: x[1], reverse=True)
        for chapter, count in sorted_chapters[:10]:
            report.append(f"- **{chapter}**: {count} connections\n")

        # Connection density statistics
        connections_list = list(stats['connections_per_section'].values())
        if connections_list:
            report.append(f"\n### Section Connection Density\n")
            report.append(f"- **Average**: {statistics.mean(connections_list):.1f} connections/section\n")
            report.append(f"- **Median**: {statistics.median(connections_list):.1f} connections/section\n")
            report.append(f"- **Max**: {max(connections_list)} connections\n")
            report.append(f"- **Min**: {min(connections_list)} connections\n")

        # Overloaded sections
        if stats['sections_with_many_connections']:
            report.append(f"\n### ⚠️ Overloaded Sections (>7 connections)\n")
            for section in stats['sections_with_many_connections'][:5]:
                report.append(f"- **{section['title']}** ({section['section']}): {section['count']} connections\n")

        # Connection types
        report.append(f"\n### Connection Type Distribution\n")
        for conn_type, count in stats['connection_types'].most_common():
            percentage = (count / stats['total_connections']) * 100
            report.append(f"- **{conn_type}**: {count} ({percentage:.1f}%)\n")

        # Similarity scores
        if stats['similarity_distribution']:
            report.append(f"\n### Similarity Score Analysis\n")
            report.append(f"- **Average**: {statistics.mean(stats['similarity_distribution']):.3f}\n")
            report.append(f"- **Median**: {statistics.median(stats['similarity_distribution']):.3f}\n")
            low_quality = sum(1 for s in stats['similarity_distribution'] if s < 0.3)
            report.append(f"- **Low Quality (<0.3)**: {low_quality} connections\n")

        # Quality Issues
        report.append("\n## 🔍 Quality Issues Identified\n")

        # Weak connections
        if issues['weak_connections']:
            report.append(f"\n### Weak Connections (similarity < 0.3): {len(issues['weak_connections'])}\n")
            for conn in issues['weak_connections'][:5]:
                report.append(f"- {conn['source']} → {conn['target']} (similarity: {conn['similarity']:.3f})\n")

        # Generic explanations
        if issues['generic_explanations']:
            report.append(f"\n### Generic Explanations: {len(issues['generic_explanations'])}\n")
            for conn in issues['generic_explanations'][:3]:
                report.append(f"- {conn['source']} → {conn['target']}\n")
                report.append(f"  - Explanation: \"{conn['explanation'][:100]}...\"\n")

        # Redundant connections
        if issues['redundant_connections']:
            report.append(f"\n### Redundant Connections: {len(issues['redundant_connections'])}\n")
            for conn in issues['redundant_connections'][:5]:
                report.append(f"- {conn['pair']} appears {conn['count']} times\n")

        # Circular references
        if issues['circular_references']:
            report.append(f"\n### Circular References: {len(issues['circular_references'])//2}\n")
            seen = set()
            for conn in issues['circular_references']:
                if conn['forward'] not in seen and conn['reverse'] not in seen:
                    report.append(f"- {conn['forward']} ↔ {conn['reverse']}\n")
                    seen.add(conn['forward'])
                    seen.add(conn['reverse'])

        # Recommendations
        report.append("\n## 💡 Recommendations for Fine-Tuning\n")
        report.append("1. **Remove weak connections** with similarity < 0.3\n")
        report.append("2. **Limit sections to 5-6 connections** maximum\n")
        report.append("3. **Improve generic explanations** with specific pedagogical value\n")
        report.append("4. **Balance connection types** within sections\n")
        report.append("5. **Review circular references** for pedagogical value\n")

        # Proposed target metrics
        report.append("\n## 🎯 Proposed Target Metrics\n")
        report.append("- **Total Connections**: 800-900 (from current 1,083)\n")
        report.append("- **Connections per Section**: 3-5 average, 6 maximum\n")
        report.append("- **Minimum Similarity**: 0.35\n")
        report.append("- **Connection Type Balance**: No single type >60% per section\n")

        return ''.join(report)

def main():
    analyzer = QualityAnalyzer()
    report = analyzer.generate_quality_report()

    # Save report
    report_file = Path("/Users/VJ/GitHub/MLSysBook/tools/scripts/cross_refs/QUALITY_ANALYSIS_REPORT.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ Quality analysis complete. Report saved to: {report_file}")

    # Also print summary to console
    print("\n📊 Quick Summary:")
    data = analyzer.load_production_data()
    stats = analyzer.analyze_connection_distribution(data)
    issues = analyzer.identify_quality_issues(data)

    print(f"  Total Connections: {stats['total_connections']}")
    print(f"  Weak Connections: {len(issues['weak_connections'])}")
    print(f"  Generic Explanations: {len(issues['generic_explanations'])}")
    print(f"  Overloaded Sections: {len(stats['sections_with_many_connections'])}")
    print(f"  Recommended Reduction: ~200 connections")

if __name__ == "__main__":
    main()
