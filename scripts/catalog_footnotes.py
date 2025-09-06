#!/usr/bin/env python3
"""
Catalog all footnotes in Quarto markdown (.qmd) files.

This script:
1. Scans all qmd files for footnotes
2. Collects inline references and their contexts
3. Collects footnote definitions
4. Generates a comprehensive report for the footnote agent
"""

import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def extract_inline_references(content: str, file_path: Path) -> List[Dict]:
    """Extract all inline footnote references with their surrounding context."""
    references = []
    lines = content.splitlines()
    
    for line_num, line in enumerate(lines, 1):
        # Find all footnote references in this line
        matches = re.finditer(r'\[\^([^\]]+)\]', line)
        for match in matches:
            footnote_id = match.group(1)
            
            # Get context (the sentence containing the footnote)
            # Find sentence boundaries
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(line), match.end() + 100)
            context = line[start_pos:end_pos].strip()
            
            # Clean up context
            if start_pos > 0:
                context = "..." + context
            if end_pos < len(line):
                context = context + "..."
            
            references.append({
                'footnote_id': footnote_id,
                'file': str(file_path),
                'line': line_num,
                'context': context,
                'full_line': line.strip()
            })
    
    return references


def extract_footnote_definitions(content: str, file_path: Path) -> List[Dict]:
    """Extract all footnote definitions."""
    definitions = []
    lines = content.splitlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a footnote definition
        match = re.match(r'^\[\^([^\]]+)\]:\s*(.*)$', line)
        if match:
            footnote_id = match.group(1)
            definition_text = match.group(2)
            line_num = i + 1
            
            # Collect continuation lines
            i += 1
            while i < len(lines):
                next_line = lines[i]
                # Continuation lines are indented or empty
                if next_line and (next_line[0] == ' ' or next_line[0] == '\t'):
                    definition_text += '\n' + next_line
                    i += 1
                elif not next_line.strip():
                    # Empty line might be part of the footnote
                    if i + 1 < len(lines) and lines[i + 1] and (lines[i + 1][0] == ' ' or lines[i + 1][0] == '\t'):
                        definition_text += '\n'
                        i += 1
                    else:
                        break
                else:
                    break
            
            # Clean up the definition
            definition_text = definition_text.strip()
            
            # Extract bold term if it exists (common pattern: **Term**: Definition)
            term_match = re.match(r'\*\*([^*]+)\*\*:\s*(.+)', definition_text)
            term = term_match.group(1) if term_match else None
            
            definitions.append({
                'footnote_id': footnote_id,
                'file': str(file_path),
                'line': line_num,
                'definition': definition_text,
                'term': term,
                'length': len(definition_text)
            })
        else:
            i += 1
    
    return definitions


def analyze_footnote_patterns(all_definitions: List[Dict]) -> Dict:
    """Analyze patterns in footnote definitions."""
    patterns = {
        'total_definitions': len(all_definitions),
        'with_bold_terms': 0,
        'average_length': 0,
        'common_prefixes': defaultdict(int),
        'terms_used': set()
    }
    
    total_length = 0
    for defn in all_definitions:
        total_length += defn['length']
        if defn['term']:
            patterns['with_bold_terms'] += 1
            patterns['terms_used'].add(defn['term'].lower())
        
        # Extract common ID prefixes (e.g., 'fn-', 'note-', etc.)
        id_parts = defn['footnote_id'].split('-')
        if len(id_parts) > 1:
            patterns['common_prefixes'][id_parts[0]] += 1
    
    if all_definitions:
        patterns['average_length'] = total_length // len(all_definitions)
    
    patterns['terms_used'] = list(patterns['terms_used'])
    patterns['common_prefixes'] = dict(patterns['common_prefixes'])
    
    return patterns


def find_duplicates(all_references: List[Dict], all_definitions: List[Dict]) -> Dict:
    """Find duplicate footnotes across chapters."""
    duplicates = {
        'duplicate_ids': defaultdict(list),
        'duplicate_terms': defaultdict(list),
        'undefined_references': [],
        'unused_definitions': []
    }
    
    # Track footnote IDs by file
    for ref in all_references:
        file_name = Path(ref['file']).stem
        duplicates['duplicate_ids'][ref['footnote_id']].append(file_name)
    
    # Track terms across files
    for defn in all_definitions:
        if defn['term']:
            file_name = Path(defn['file']).stem
            duplicates['duplicate_terms'][defn['term'].lower()].append({
                'file': file_name,
                'footnote_id': defn['footnote_id']
            })
    
    # Find undefined references
    defined_ids = {d['footnote_id'] for d in all_definitions}
    referenced_ids = {r['footnote_id'] for r in all_references}
    
    for ref in all_references:
        if ref['footnote_id'] not in defined_ids:
            duplicates['undefined_references'].append({
                'footnote_id': ref['footnote_id'],
                'file': Path(ref['file']).stem,
                'line': ref['line']
            })
    
    # Find unused definitions
    for defn in all_definitions:
        if defn['footnote_id'] not in referenced_ids:
            duplicates['unused_definitions'].append({
                'footnote_id': defn['footnote_id'],
                'file': Path(defn['file']).stem,
                'line': defn['line']
            })
    
    # Clean up duplicates - only keep actual duplicates
    duplicates['duplicate_ids'] = {
        k: list(set(v)) for k, v in duplicates['duplicate_ids'].items() 
        if len(set(v)) > 1
    }
    
    duplicates['duplicate_terms'] = {
        k: v for k, v in duplicates['duplicate_terms'].items() 
        if len(v) > 1
    }
    
    return duplicates


def generate_chapter_summary(file_path: Path, references: List[Dict], definitions: List[Dict]) -> Dict:
    """Generate a summary for a specific chapter."""
    return {
        'file': str(file_path),
        'chapter_name': file_path.stem,
        'total_references': len(references),
        'total_definitions': len(definitions),
        'footnote_ids': sorted(list({r['footnote_id'] for r in references})),
        'terms_defined': sorted([d['term'] for d in definitions if d['term']])
    }


def generate_agent_context(all_data: Dict, target_chapter: str = None) -> str:
    """Generate context information for the footnote agent."""
    context = []
    
    context.append("# FOOTNOTE CATALOG AND CONTEXT\n")
    context.append("## Book-Wide Footnote Statistics\n")
    
    patterns = all_data['patterns']
    context.append(f"- Total footnotes defined: {patterns['total_definitions']}")
    context.append(f"- Footnotes with bold terms: {patterns['with_bold_terms']}")
    context.append(f"- Average definition length: {patterns['average_length']} characters")
    context.append(f"- Common ID prefixes: {patterns['common_prefixes']}")
    context.append(f"- Total unique terms: {len(patterns['terms_used'])}\n")
    
    if all_data['duplicates']['duplicate_terms']:
        context.append("## ⚠️ IMPORTANT: Terms Already Defined\n")
        context.append("These terms have already been defined in other chapters. DO NOT redefine them:\n")
        for term, locations in all_data['duplicates']['duplicate_terms'].items():
            context.append(f"- **{term}**: defined in {', '.join([l['file'] for l in locations])}")
        context.append("")
    
    if target_chapter:
        # Find chapter data
        chapter_data = None
        for chapter in all_data['by_chapter']:
            if chapter['chapter_name'] == target_chapter or target_chapter in chapter['file']:
                chapter_data = chapter
                break
        
        if chapter_data:
            context.append(f"## Current Chapter: {chapter_data['chapter_name']}\n")
            context.append(f"- Existing footnotes: {chapter_data['total_references']}")
            context.append(f"- Footnote IDs used: {', '.join(chapter_data['footnote_ids'])}")
            if chapter_data['terms_defined']:
                context.append(f"- Terms already defined: {', '.join(chapter_data['terms_defined'])}")
            context.append("")
    
    context.append("## Footnote Style Guidelines\n")
    context.append("Based on existing footnotes, follow these patterns:")
    context.append("1. Use ID format: [^fn-term-name] (lowercase, hyphens)")
    context.append("2. Definition format: **Bold Term**: Clear definition. Optional analogy.")
    context.append("3. Keep definitions concise (avg ~200 characters)")
    context.append("4. Avoid redefining terms from other chapters")
    context.append("5. Focus on technical terms that need clarification\n")
    
    context.append("## All Terms Currently Defined in Book\n")
    if patterns['terms_used']:
        for i in range(0, len(patterns['terms_used']), 5):
            batch = patterns['terms_used'][i:i+5]
            context.append(f"- {', '.join(batch)}")
    
    return '\n'.join(context)


def main():
    """Main function to catalog all footnotes."""
    # Determine root directory
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path('/Users/VJ/GitHub/MLSysBook/quarto')
    
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} does not exist")
        sys.exit(1)
    
    print(f"Cataloging footnotes in: {root_dir}")
    print("-" * 60)
    
    # Find all .qmd files
    qmd_files = sorted(root_dir.rglob('*.qmd'))
    
    all_references = []
    all_definitions = []
    by_chapter = []
    
    for qmd_file in qmd_files:
        try:
            with open(qmd_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip files with no content
            if not content.strip():
                continue
            
            # Extract footnotes
            references = extract_inline_references(content, qmd_file)
            definitions = extract_footnote_definitions(content, qmd_file)
            
            if references or definitions:
                relative_path = qmd_file.relative_to(root_dir.parent)
                print(f"✓ {relative_path}")
                print(f"  - {len(references)} inline references")
                print(f"  - {len(definitions)} definitions")
                
                all_references.extend(references)
                all_definitions.extend(definitions)
                
                chapter_summary = generate_chapter_summary(qmd_file, references, definitions)
                by_chapter.append(chapter_summary)
                
        except Exception as e:
            print(f"Error processing {qmd_file}: {e}")
    
    # Analyze patterns and duplicates
    patterns = analyze_footnote_patterns(all_definitions)
    duplicates = find_duplicates(all_references, all_definitions)
    
    # Create comprehensive report
    report = {
        'total_files': len(qmd_files),
        'total_references': len(all_references),
        'total_definitions': len(all_definitions),
        'patterns': patterns,
        'duplicates': duplicates,
        'by_chapter': by_chapter,
        'all_references': all_references,
        'all_definitions': all_definitions
    }
    
    # Save JSON report
    report_file = root_dir.parent / 'footnote_catalog.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("FOOTNOTE CATALOG SUMMARY")
    print("=" * 60)
    print(f"Total files scanned: {len(qmd_files)}")
    print(f"Total inline references: {len(all_references)}")
    print(f"Total definitions: {len(all_definitions)}")
    print(f"Unique footnote IDs: {len(set(r['footnote_id'] for r in all_references))}")
    print(f"Terms defined: {len(patterns['terms_used'])}")
    
    if duplicates['undefined_references']:
        print(f"\n⚠️  Undefined references: {len(duplicates['undefined_references'])}")
        for ref in duplicates['undefined_references'][:5]:
            print(f"   - [{ref['footnote_id']}] in {ref['file']} line {ref['line']}")
    
    if duplicates['unused_definitions']:
        print(f"\n⚠️  Unused definitions: {len(duplicates['unused_definitions'])}")
        for defn in duplicates['unused_definitions'][:5]:
            print(f"   - [{defn['footnote_id']}] in {defn['file']} line {defn['line']}")
    
    print(f"\n✓ Full report saved to: {report_file}")
    
    # Generate agent context file
    agent_context = generate_agent_context(report)
    context_file = root_dir.parent / '.claude' / 'footnote_context.md'
    context_file.parent.mkdir(exist_ok=True)
    with open(context_file, 'w', encoding='utf-8') as f:
        f.write(agent_context)
    print(f"✓ Agent context saved to: {context_file}")


if __name__ == "__main__":
    main()