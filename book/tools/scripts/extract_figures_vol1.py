#!/usr/bin/env python3
"""
Extract figures and captions/alt-text from Volume 1 chapters for MIT Press.

This script reads the chapter configuration from the YAML file, extracts
all figures from each chapter, and outputs a Markdown file listing each
chapter with its figures numbered sequentially (Figure 1, Figure 2, etc.).

Usage:
    python extract_figures_vol1.py [--label | --alt-text | --caption]

Options:
    --label     Extract just the figure title (bold text from fig-cap)
    --alt-text  Extract fig-alt (visual descriptions for accessibility)
    --caption   Extract full fig-cap (explanatory caption text) - default

Output:
    Creates FIGURE_LIST_VOL1.md in the book/quarto directory.
"""

import argparse
import re
import yaml
from pathlib import Path


def parse_yaml_chapters(yaml_path: Path) -> list[dict]:
    """
    Parse the YAML configuration to extract the ordered list of chapters.
    
    Returns a list of dicts with 'path' and 'is_part' keys.
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse YAML
    config = yaml.safe_load(content)
    
    chapters = []
    if 'book' in config and 'chapters' in config['book']:
        for item in config['book']['chapters']:
            if isinstance(item, str):
                # Skip non-content files like index.qmd
                if item == 'index.qmd':
                    continue
                chapters.append({
                    'path': item,
                    'is_part': '/parts/' in item
                })
            elif isinstance(item, dict) and 'part' in item:
                # Handle part definitions with chapters
                chapters.append({
                    'path': item.get('file', ''),
                    'is_part': True,
                    'part_title': item.get('part', '')
                })
                if 'chapters' in item:
                    for ch in item['chapters']:
                        if isinstance(ch, str):
                            chapters.append({
                                'path': ch,
                                'is_part': False
                            })
    
    return chapters


def read_all_chapters_from_yaml(yaml_content: str) -> list[str]:
    """
    Read both commented and uncommented chapter entries from YAML.
    This captures the full intended chapter order.
    """
    chapters = []
    
    # Pattern to match chapter entries (both commented and uncommented)
    # Matches lines like:
    #   - contents/vol1/introduction/introduction.qmd
    #   # - contents/vol1/introduction/introduction.qmd
    pattern = re.compile(r'^\s*#?\s*-\s*(contents/vol1/[^\s#]+\.qmd)\s*$', re.MULTILINE)
    
    for match in pattern.finditer(yaml_content):
        path = match.group(1)
        # Skip part files and backmatter
        if '/parts/' not in path and '/backmatter/' not in path and '/frontmatter/' not in path:
            chapters.append(path)
    
    return chapters


def extract_chapter_title(qmd_path: Path) -> str:
    """
    Extract the chapter title from a .qmd file.
    
    Looks for the first line starting with '# ' that is the chapter heading.
    """
    with open(qmd_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip YAML frontmatter
    if content.startswith('---'):
        end_yaml = content.find('---', 3)
        if end_yaml != -1:
            content = content[end_yaml + 3:]
    
    # Find first heading
    match = re.search(r'^#\s+([^{#\n]+)', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    return qmd_path.stem.replace('_', ' ').title()


def extract_figures(qmd_path: Path, extract_type: str = 'caption') -> list[dict]:
    """
    Extract all figures from a .qmd file.
    
    Args:
        qmd_path: Path to the .qmd file
        extract_type: 'label' for title only, 'alt-text' for fig-alt, 'caption' for full fig-cap
    
    Returns a list of dicts with 'id' and 'text' keys.
    """
    with open(qmd_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    figures = []
    
    # Choose the attribute to extract
    attr = 'fig-alt' if extract_type == 'alt-text' else 'fig-cap'
    
    # Pattern to match figure definitions with the chosen attribute
    # Handles escaped quotes: \"text\"
    # Format: {#fig-NAME ... fig-alt="TEXT" ...} or {#fig-NAME ... fig-cap="TEXT" ...}
    fig_pattern = re.compile(
        rf'\{{#(fig-[a-zA-Z0-9_-]+)[^}}]*{attr}="((?:[^"\\]|\\.)*)"',
        re.DOTALL
    )
    
    for match in fig_pattern.finditer(content):
        fig_id = match.group(1)
        text = match.group(2)
        
        # Unescape quotes
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        
        # Clean up text - normalize whitespace
        text = ' '.join(text.split())
        
        if extract_type == 'label':
            # Extract just the bold title portion: **Title**: description -> Title
            label_match = re.match(r'\*\*([^*]+)\*\*', text)
            if label_match:
                text = label_match.group(1).rstrip(':')
            else:
                # If no bold markers, take text up to first colon or period
                colon_pos = text.find(':')
                period_pos = text.find('.')
                if colon_pos > 0 and (period_pos < 0 or colon_pos < period_pos):
                    text = text[:colon_pos]
                elif period_pos > 0:
                    text = text[:period_pos]
        elif extract_type == 'caption':
            # Remove bold markers but keep full text
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        # For 'alt-text', keep as-is
        
        figures.append({
            'id': fig_id,
            'text': text
        })
    
    return figures


def generate_markdown_output(chapters_data: list[dict], output_path: Path, extract_type: str) -> None:
    """
    Generate the Markdown output file with chapter and figure listings.
    """
    type_labels = {
        'label': "Labels (Figure Titles Only)",
        'alt-text': "Alt Text (Visual Descriptions)",
        'caption': "Captions (Full Explanatory Text)"
    }
    type_label = type_labels.get(extract_type, extract_type)
    
    lines = [
        "# Figure List - Volume I: Introduction",
        "",
        "_Machine Learning Systems_",
        "",
        f"**Extraction Type**: {type_label}",
        "",
        "---",
        "",
    ]
    
    chapter_num = 0
    total_figures = 0
    
    for chapter in chapters_data:
        if not chapter.get('figures'):
            continue
        
        chapter_num += 1
        
        lines.append(f"## Chapter {chapter_num}: {chapter['title']}")
        lines.append("")
        
        for i, fig in enumerate(chapter['figures'], 1):
            lines.append(f"**Figure {chapter_num}.{i}**: {fig['text']}")
            lines.append("")
            total_figures += 1
        
        lines.append("---")
        lines.append("")
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Output written to: {output_path}")
    print(f"Total chapters processed: {chapter_num}")
    print(f"Total figures extracted: {total_figures}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract figure information from Volume 1 chapters for MIT Press.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_figures_vol1.py --label     # Extract just figure titles
    python extract_figures_vol1.py --alt-text  # Extract fig-alt (visual descriptions)
    python extract_figures_vol1.py --caption   # Extract full fig-cap (default)
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--label', 
        action='store_true',
        help='Extract just the figure title (bold text from fig-cap)'
    )
    group.add_argument(
        '--alt-text', 
        action='store_true',
        help='Extract fig-alt (visual descriptions for accessibility)'
    )
    group.add_argument(
        '--caption', 
        action='store_true',
        help='Extract full fig-cap (explanatory caption text) - default'
    )
    
    args = parser.parse_args()
    
    # Determine extraction type (default to 'caption')
    if args.label:
        extract_type = 'label'
    elif args.alt_text:
        extract_type = 'alt-text'
    else:
        extract_type = 'caption'
    
    mode_descriptions = {
        'label': 'fig-cap titles only',
        'alt-text': 'fig-alt (visual descriptions)',
        'caption': 'fig-cap (full captions)'
    }
    print(f"Extraction mode: {mode_descriptions[extract_type]}")
    print()
    
    # Determine paths
    script_dir = Path(__file__).parent
    book_root = script_dir.parent.parent  # book/quarto
    yaml_path = book_root / 'quarto' / 'config' / '_quarto-pdf-vol1.yml'
    
    # Also check for a more complete YAML or use the current one
    if not yaml_path.exists():
        yaml_path = book_root / 'config' / '_quarto-pdf-vol1.yml'
    
    if not yaml_path.exists():
        # Try from workspace root
        yaml_path = Path('/Users/VJ/GitHub/mlsysbook-vols/book/quarto/config/_quarto-pdf-vol1.yml')
    
    print(f"Reading YAML config from: {yaml_path}")
    
    # Read YAML content to extract all chapters (including commented ones)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_content = f.read()
    
    # Get all chapter paths from the YAML (preserving intended order)
    chapter_paths = read_all_chapters_from_yaml(yaml_content)
    
    print(f"Found {len(chapter_paths)} chapters to process")
    
    # Process each chapter
    chapters_data = []
    
    for rel_path in chapter_paths:
        # Construct full path
        qmd_path = yaml_path.parent.parent / rel_path
        
        if not qmd_path.exists():
            print(f"  Warning: File not found: {qmd_path}")
            continue
        
        print(f"  Processing: {rel_path}")
        
        title = extract_chapter_title(qmd_path)
        figures = extract_figures(qmd_path, extract_type)
        
        if figures:  # Only include chapters with figures
            chapters_data.append({
                'path': rel_path,
                'title': title,
                'figures': figures
            })
            print(f"    Found {len(figures)} figures")
    
    # Generate output
    output_path = yaml_path.parent.parent / 'FIGURE_LIST_VOL1.md'
    generate_markdown_output(chapters_data, output_path, extract_type)


if __name__ == '__main__':
    main()
