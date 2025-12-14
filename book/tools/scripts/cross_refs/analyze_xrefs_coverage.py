#!/usr/bin/env python3
"""
Cross-Reference Coverage Analysis
=================================

Analyzes the generated cross-references to identify patterns and improvement opportunities.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict, Counter

def load_xrefs(chapters_dir: Path) -> Dict[str, Dict]:
    """Load all cross-reference files."""
    xrefs = {}
    for chapter_dir in sorted(chapters_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue

        xref_file = chapter_dir / f"{chapter_dir.name}_xrefs.json"
        if xref_file.exists():
            with open(xref_file) as f:
                xrefs[chapter_dir.name] = json.load(f)
    return xrefs

def analyze_coverage(xrefs: Dict[str, Dict]) -> Dict:
    """Analyze cross-reference coverage patterns."""
    stats = {
        'total_chapters': len(xrefs),
        'total_connections': 0,
        'chapters_with_connections': 0,
        'connection_types': Counter(),
        'target_frequency': Counter(),
        'source_frequency': Counter(),
        'strength_distribution': [],
        'concepts_per_connection': [],
        'bidirectional_connections': 0
    }

    all_connections = []

    for source_chapter, data in xrefs.items():
        chapter_connections = 0

        for section, connections in data.get('cross_references', {}).items():
            for conn in connections:
                chapter_connections += 1
                stats['total_connections'] += 1
                stats['connection_types'][conn['connection_type']] += 1
                stats['target_frequency'][conn['target_chapter']] += 1
                stats['source_frequency'][source_chapter] += 1
                stats['strength_distribution'].append(conn['strength'])
                stats['concepts_per_connection'].append(len(conn['concepts']))

                all_connections.append({
                    'source': source_chapter,
                    'target': conn['target_chapter'],
                    'type': conn['connection_type'],
                    'strength': conn['strength']
                })

        if chapter_connections > 0:
            stats['chapters_with_connections'] += 1

    # Calculate bidirectional connections
    connection_pairs = set()
    for conn in all_connections:
        pair = tuple(sorted([conn['source'], conn['target']]))
        connection_pairs.add(pair)

    # Count how many are truly bidirectional
    bidirectional = 0
    for pair in connection_pairs:
        forward = any(c['source'] == pair[0] and c['target'] == pair[1] for c in all_connections)
        backward = any(c['source'] == pair[1] and c['target'] == pair[0] for c in all_connections)
        if forward and backward:
            bidirectional += 1

    stats['bidirectional_connections'] = bidirectional

    return stats, all_connections

def find_missing_connections(xrefs: Dict[str, Dict], chapters_dir: Path) -> List[str]:
    """Identify potential missing connections based on concept overlap."""
    # Load concept maps
    concept_maps = {}
    for chapter_dir in sorted(chapters_dir.iterdir()):
        if not chapter_dir.is_dir():
            continue

        concept_file = chapter_dir / f"{chapter_dir.name}_concepts.yml"
        if concept_file.exists():
            with open(concept_file) as f:
                data = yaml.safe_load(f)
                concept_maps[chapter_dir.name] = data['concept_map']

    # Find chapters with high concept overlap but no connections
    missing = []

    for source in concept_maps:
        source_concepts = set()
        source_map = concept_maps[source]
        source_concepts.update([c.lower() for c in source_map.get('keywords', [])])
        source_concepts.update([c.lower() for c in source_map.get('primary_concepts', [])])

        existing_targets = set()
        if source in xrefs:
            for section_conns in xrefs[source].get('cross_references', {}).values():
                for conn in section_conns:
                    existing_targets.add(conn['target_chapter'])

        for target in concept_maps:
            if source == target or target in existing_targets:
                continue

            target_concepts = set()
            target_map = concept_maps[target]
            target_concepts.update([c.lower() for c in target_map.get('keywords', [])])
            target_concepts.update([c.lower() for c in target_map.get('primary_concepts', [])])

            overlap = source_concepts & target_concepts
            if len(overlap) >= 3:  # Significant overlap
                overlap_strength = len(overlap) / len(source_concepts | target_concepts)
                if overlap_strength > 0.1:
                    missing.append(f"{source} → {target} (overlap: {len(overlap)}, strength: {overlap_strength:.3f})")

    return missing

def print_analysis(stats: Dict, all_connections: List, missing: List):
    """Print comprehensive analysis results."""
    print("🔍 CROSS-REFERENCE COVERAGE ANALYSIS")
    print("=" * 50)

    print(f"\n📊 BASIC STATISTICS")
    print(f"Total chapters: {stats['total_chapters']}")
    print(f"Chapters with connections: {stats['chapters_with_connections']}")
    print(f"Coverage: {stats['chapters_with_connections']/stats['total_chapters']*100:.1f}%")
    print(f"Total connections: {stats['total_connections']}")
    print(f"Average connections per chapter: {stats['total_connections']/stats['total_chapters']:.1f}")
    print(f"Bidirectional connections: {stats['bidirectional_connections']}")

    print(f"\n🎯 CONNECTION TYPES")
    for conn_type, count in stats['connection_types'].most_common():
        pct = count / stats['total_connections'] * 100
        print(f"  {conn_type}: {count} ({pct:.1f}%)")

    print(f"\n⭐ STRENGTH DISTRIBUTION")
    strengths = stats['strength_distribution']
    if strengths:
        print(f"  Average strength: {sum(strengths)/len(strengths):.3f}")
        print(f"  Median strength: {sorted(strengths)[len(strengths)//2]:.3f}")
        print(f"  Min strength: {min(strengths):.3f}")
        print(f"  Max strength: {max(strengths):.3f}")

    print(f"\n🎪 MOST CONNECTED CHAPTERS (as targets)")
    for chapter, count in stats['target_frequency'].most_common(10):
        print(f"  {chapter}: {count} incoming connections")

    print(f"\n📤 MOST ACTIVE CHAPTERS (as sources)")
    for chapter, count in stats['source_frequency'].most_common(10):
        print(f"  {chapter}: {count} outgoing connections")

    print(f"\n🔍 CONCEPTS PER CONNECTION")
    concepts = stats['concepts_per_connection']
    if concepts:
        print(f"  Average concepts: {sum(concepts)/len(concepts):.1f}")
        print(f"  Max concepts: {max(concepts)}")
        print(f"  Min concepts: {min(concepts)}")

    if missing:
        print(f"\n🚫 POTENTIAL MISSING CONNECTIONS ({len(missing)})")
        for miss in missing[:10]:  # Show top 10
            print(f"  {miss}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    print(f"\n🎯 RECOMMENDATIONS")

    # Coverage recommendations
    coverage_pct = stats['chapters_with_connections']/stats['total_chapters']*100
    if coverage_pct < 80:
        print(f"  • Low coverage ({coverage_pct:.1f}%) - consider lowering thresholds")

    # Connection balance
    avg_connections = stats['total_connections']/stats['total_chapters']
    if avg_connections < 2:
        print(f"  • Low connection density ({avg_connections:.1f}) - increase max_refs or lower min_overlap")

    # Bidirectional balance
    bidirectional_pct = stats['bidirectional_connections'] / (stats['total_connections']/2) * 100
    if bidirectional_pct < 30:
        print(f"  • Low bidirectional connections ({bidirectional_pct:.1f}%) - concepts may be too specific")

    # Type diversity
    if len(stats['connection_types']) < 4:
        print(f"  • Limited connection types ({len(stats['connection_types'])}) - improve type classification")

    print(f"\n✅ Analysis complete!")

def main():
    chapters_dir = Path('/Users/VJ/GitHub/MLSysBook/quarto/contents/core')

    print("Loading cross-references...")
    xrefs = load_xrefs(chapters_dir)

    print("Analyzing coverage...")
    stats, all_connections = analyze_coverage(xrefs)

    print("Finding missing connections...")
    missing = find_missing_connections(xrefs, chapters_dir)

    print_analysis(stats, all_connections, missing)

if __name__ == "__main__":
    main()
