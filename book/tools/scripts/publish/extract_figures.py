#!/usr/bin/env python3
"""
Extract figures and captions/alt-text from book chapters for MIT Press.

This script reads the chapter configuration from the YAML file, extracts
all figures from each chapter, and outputs a Markdown file listing each
chapter with its figures numbered sequentially (Figure 1, Figure 2, etc.).

Usage:
    python extract_figures.py --vol 1                    # All types for vol1
    python extract_figures.py --vol 2 --label            # Labels only for vol2
    python extract_figures.py --vol 1 --alt-text         # Alt text only for vol1
    python extract_figures.py --vol 1 --caption          # Captions only for vol1

Options:
    --vol N     Required. Volume number (1, 2, etc.)
    --label     Extract just the figure title (bold text from fig-cap)
    --alt-text  Extract fig-alt (visual descriptions for accessibility)
    --caption   Extract full fig-cap (explanatory caption text)
    
    If no output type is specified, generates all three grouped by figure.

Output:
    Creates FIGURE_LIST_VOL{N}.md in the book/quarto directory.
"""

import argparse
import re
import sys
import yaml
from pathlib import Path


def read_all_chapters_from_yaml(yaml_content: str, vol: str) -> list[str]:
    """
    Read both commented and uncommented chapter entries from YAML.
    This captures the full intended chapter order.
    """
    chapters = []
    
    # Pattern to match chapter entries (both commented and uncommented)
    # Matches lines like:
    #   - contents/vol1/introduction/introduction.qmd
    #   # - contents/vol1/introduction/introduction.qmd
    pattern = re.compile(
        rf'^\s*#?\s*-\s*(contents/vol{vol}/[^\s#]+\.qmd)\s*$', 
        re.MULTILINE
    )
    
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


def extract_figures_all(qmd_path: Path) -> list[dict]:
    """
    Extract all figures from a .qmd file with all three attributes.
    
    Args:
        qmd_path: Path to the .qmd file
    
    Returns a list of dicts with 'id', 'label', 'caption', and 'alt_text' keys.
    """
    with open(qmd_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    figures = []
    
    # First, find all figure IDs with their full attribute blocks
    # Pattern matches the entire {#fig-ID ...} block
    fig_block_pattern = re.compile(
        r'\{#(fig-[a-zA-Z0-9_-]+)([^}]*)\}',
        re.DOTALL
    )
    
    for match in fig_block_pattern.finditer(content):
        fig_id = match.group(1)
        attrs = match.group(2)
        
        # Extract fig-cap
        cap_match = re.search(r'fig-cap="((?:[^"\\]|\\.)*)"', attrs)
        caption_raw = cap_match.group(1) if cap_match else ''
        
        # Extract fig-alt
        alt_match = re.search(r'fig-alt="((?:[^"\\]|\\.)*)"', attrs)
        alt_text = alt_match.group(1) if alt_match else ''
        
        # Skip if neither caption nor alt-text
        if not caption_raw and not alt_text:
            continue
        
        # Clean up caption
        caption_raw = caption_raw.replace('\\"', '"').replace("\\'", "'")
        caption_raw = ' '.join(caption_raw.split())
        
        # Extract label from caption (bold title)
        label = ''
        if caption_raw:
            label_match = re.match(r'\*\*([^*]+)\*\*', caption_raw)
            if label_match:
                label = label_match.group(1).rstrip(':')
            else:
                # If no bold markers, take text up to first colon or period
                colon_pos = caption_raw.find(':')
                period_pos = caption_raw.find('.')
                if colon_pos > 0 and (period_pos < 0 or colon_pos < period_pos):
                    label = caption_raw[:colon_pos]
                elif period_pos > 0:
                    label = caption_raw[:period_pos]
                else:
                    label = caption_raw[:50] + '...' if len(caption_raw) > 50 else caption_raw
        
        # Clean caption (remove bold markers)
        caption = re.sub(r'\*\*([^*]+)\*\*', r'\1', caption_raw)
        
        # Clean alt text
        alt_text = alt_text.replace('\\"', '"').replace("\\'", "'")
        alt_text = ' '.join(alt_text.split())
        
        figures.append({
            'id': fig_id,
            'label': label,
            'caption': caption,
            'alt_text': alt_text
        })
    
    return figures


def generate_markdown_output(chapters_data: list[dict], output_path: Path, 
                            extract_types: list[str], vol: str) -> None:
    """
    Generate the Markdown output file with chapter and figure listings.
    Figures are grouped together with all their attributes.
    """
    # Get volume title from YAML if possible
    vol_titles = {
        '1': 'Volume I: Introduction',
        '2': 'Volume II: Advanced Topics',
    }
    vol_title = vol_titles.get(vol, f'Volume {vol}')
    
    type_descriptions = {
        'label': 'Title',
        'caption': 'Caption', 
        'alt-text': 'Alt Text'
    }
    
    lines = [
        f"# Figure List - {vol_title}",
        "",
        "_Machine Learning Systems_",
        "",
        f"**Volume**: {vol}",
        f"**Includes**: {', '.join(type_descriptions[t] for t in extract_types)}",
        "",
        "---",
        "",
    ]
    
    chapter_num = 0
    total_figures = 0
    
    for chapter in chapters_data:
        figures = chapter.get('figures', [])
        if not figures:
            continue
        
        chapter_num += 1
        
        lines.append(f"## Chapter {chapter_num}: {chapter['title']}")
        lines.append("")
        
        for i, fig in enumerate(figures, 1):
            fig_num = f"{chapter_num}.{i}"
            label = fig.get('label', '')
            
            # Figure header with label
            lines.append(f"### Figure {fig_num}: {label}")
            lines.append("")
            
            # Add requested fields
            if 'caption' in extract_types and fig.get('caption'):
                lines.append(f"**Caption**: {fig['caption']}")
                lines.append("")
            
            if 'alt-text' in extract_types and fig.get('alt_text'):
                lines.append(f"**Alt Text**: {fig['alt_text']}")
                lines.append("")
            
            total_figures += 1
        
        lines.append("---")
        lines.append("")
    
    lines.append(f"_Total: {total_figures} figures across {chapter_num} chapters_")
    lines.append("")
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Output written to: {output_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract figure information from book chapters for MIT Press.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_figures.py --vol 1                    # All types for vol1
    python extract_figures.py --vol 2 --caption          # Captions only for vol2
    python extract_figures.py --vol 1 --alt-text         # Alt text only for vol1
    python extract_figures.py --vol 1 --caption --alt-text  # Both for vol1
        """
    )
    
    parser.add_argument(
        '--vol',
        required=True,
        help='Volume number (1, 2, etc.) - REQUIRED'
    )
    
    parser.add_argument(
        '--alt-text', 
        action='store_true',
        help='Include fig-alt (visual descriptions for accessibility)'
    )
    parser.add_argument(
        '--caption', 
        action='store_true',
        help='Include full fig-cap (explanatory caption text)'
    )
    
    args = parser.parse_args()
    
    vol = args.vol
    
    # Determine extraction types (default to all if none specified)
    # Label is always included as the figure header
    extract_types = ['label']  # Always include label as header
    if args.alt_text:
        extract_types.append('alt-text')
    if args.caption:
        extract_types.append('caption')
    
    # If no specific types requested, include both caption and alt-text
    if not args.alt_text and not args.caption:
        extract_types = ['label', 'caption', 'alt-text']
        print("No output type specified. Generating all: label, caption, alt-text")
    
    print(f"Volume: {vol}")
    print(f"Extraction types: {', '.join(extract_types)}")
    print()
    
    # Determine paths
    script_dir = Path(__file__).parent
    book_root = script_dir.parent.parent  # book/
    yaml_path = book_root / 'quarto' / 'config' / f'_quarto-pdf-vol{vol}.yml'
    
    if not yaml_path.exists():
        # Try alternate path
        yaml_path = Path(f'/Users/VJ/GitHub/mlsysbook-vols/book/quarto/config/_quarto-pdf-vol{vol}.yml')
    
    if not yaml_path.exists():
        print(f"Error: YAML config not found at {yaml_path}")
        print(f"Make sure vol{vol} configuration exists.")
        sys.exit(1)
    
    print(f"Reading YAML config from: {yaml_path}")
    
    # Read YAML content to extract all chapters (including commented ones)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_content = f.read()
    
    # Get all chapter paths from the YAML (preserving intended order)
    chapter_paths = read_all_chapters_from_yaml(yaml_content, vol)
    
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
        figures = extract_figures_all(qmd_path)
        
        if figures:
            print(f"    Found {len(figures)} figures")
            chapters_data.append({
                'path': rel_path,
                'title': title,
                'figures': figures
            })
    
    # Generate output
    output_path = yaml_path.parent.parent / f'FIGURE_LIST_VOL{vol}.md'
    generate_markdown_output(chapters_data, output_path, extract_types, vol)


if __name__ == '__main__':
    main()
