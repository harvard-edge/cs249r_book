#!/usr/bin/env python3
"""
extract_concepts.py

Extracts key concepts and topics from .qmd chapters by analyzing:
1. Section headers
2. Bold terms (**term**)
3. Footnote definitions ([^fn-name])
4. Terms in definition blocks
5. Figure and table captions

This helps build an accurate knowledge map of what each chapter actually covers.

Usage:
    python extract_concepts.py -f path/to/chapter.qmd
    python extract_concepts.py -d path/to/core/
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict

def extract_concepts_from_file(file_path):
    """
    Extracts key concepts from a .qmd file.

    Returns:
        dict with:
        - headers: list of (level, text) tuples
        - bold_terms: list of bolded terms
        - footnotes: list of footnote names/topics
        - definitions: list of defined terms
        - figures: list of figure topics
    """
    concepts = {
        'headers': [],
        'bold_terms': set(),
        'footnotes': [],
        'definitions': [],
        'figures': [],
        'introduces': set()  # Key introduced concepts
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    # Extract headers
    for line in lines:
        match = re.match(r'^(#{1,6})\s+(.*)', line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            # Remove {#sec-...} labels
            text = re.sub(r'\{#.*?\}', '', text).strip()
            concepts['headers'].append((level, text))

    # Extract bold terms (often definitions)
    bold_pattern = r'\*\*([^*]+)\*\*'
    for match in re.finditer(bold_pattern, content):
        term = match.group(1).strip()
        if len(term) > 2 and not term.startswith('Note'):
            concepts['bold_terms'].add(term)

    # Extract footnote definitions
    footnote_pattern = r'\[\^fn-([^\]]+)\]:\s*(.+?)(?=\n\n|\[\^|\Z)'
    for match in re.finditer(footnote_pattern, content, re.DOTALL):
        name = match.group(1)
        definition = match.group(2).strip()[:100]  # First 100 chars
        concepts['footnotes'].append(f"{name}: {definition}")

    # Extract definition blocks (common patterns)
    # Pattern: "X is defined as..." or "X refers to..."
    definition_patterns = [
        r'(\w[\w\s]+?)\s+is defined as',
        r'(\w[\w\s]+?)\s+refers to',
        r'(\w[\w\s]+?)\s+is a (?:type|kind|form) of',
        r'We define\s+(\w[\w\s]+?)\s+as',
    ]

    for pattern in definition_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            term = match.group(1).strip()
            if len(term) < 50:  # Reasonable length for a term
                concepts['definitions'].append(term)

    # Extract figure captions
    figure_pattern = r'!\[([^\]]+)\]'
    for match in re.finditer(figure_pattern, content):
        caption = match.group(1).strip()
        if caption:
            concepts['figures'].append(caption[:100])

    # Identify key introduced concepts (heuristic)
    # Look for phrases like "introduce", "present", "explore"
    intro_patterns = [
        r'we (?:will |now )?introduce\s+(\w[\w\s,]+)',
        r'introduces?\s+(\w[\w\s,]+)',
        r'explore\s+(\w[\w\s,]+)',
        r'present\s+(\w[\w\s,]+)',
        r'discuss\s+(\w[\w\s,]+)',
    ]

    for pattern in intro_patterns:
        for match in re.finditer(pattern, content[:5000], re.IGNORECASE):  # Check first part
            concepts['introduces'].add(match.group(1).strip())

    return concepts

def process_chapter(file_path):
    """Process a single chapter and return formatted summary."""
    concepts = extract_concepts_from_file(file_path)
    chapter_name = Path(file_path).stem

    summary = []
    summary.append(f"\n### {chapter_name.replace('_', ' ').title()}")

    # Main topics from level 2 headers
    main_topics = [text for level, text in concepts['headers'] if level == 2 and not text.startswith('Purpose')]
    if main_topics:
        summary.append("**Main Topics:**")
        for topic in main_topics[:10]:  # Limit to 10
            summary.append(f"- {topic}")

    # Key concepts from bold terms
    key_terms = sorted(list(concepts['bold_terms']))[:15]  # Top 15 terms
    if key_terms:
        summary.append("\n**Key Terms:**")
        summary.append(", ".join(key_terms))

    # Introduced concepts
    if concepts['introduces']:
        summary.append("\n**Introduces:**")
        for concept in sorted(list(concepts['introduces']))[:10]:
            summary.append(f"- {concept}")

    return "\n".join(summary)

def main():
    parser = argparse.ArgumentParser(description="Extract concepts from .qmd files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='Path to a single .qmd file')
    group.add_argument('-d', '--directory', help='Directory containing .qmd files')
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        # Get chapters in order
        chapter_order = [
            'introduction', 'ml_systems', 'dl_primer', 'dnn_architectures',
            'workflow', 'data_engineering', 'frameworks', 'training',
            'efficient_ai', 'optimizations', 'hw_acceleration', 'benchmarking',
            'ops', 'ondevice_learning', 'robust_ai', 'privacy_security',
            'responsible_ai', 'sustainable_ai', 'ai_for_good', 'conclusion'
        ]

        files = []
        base_dir = Path(args.directory)
        for chapter in chapter_order:
            chapter_file = base_dir / chapter / f"{chapter}.qmd"
            if chapter_file.exists():
                files.append(chapter_file)

    print("# Knowledge Map v2 - Extracted from Actual Content\n")

    for i, file_path in enumerate(files, 1):
        print(f"\n## Chapter {i}: {process_chapter(file_path)}")

if __name__ == "__main__":
    main()
