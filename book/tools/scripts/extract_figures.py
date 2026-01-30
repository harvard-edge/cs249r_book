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
    
    If no output type is specified, generates all three in separate sections.

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


def extract_figures(qmd_path: Path, extract_type: str) -> list[dict]:
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


def generate_markdown_section(chapters_data: list[dict], extract_type: str, vol: str) -> list[str]:
    """
    Generate markdown lines for a single extraction type.
    """
    type_labels = {
        'label': "Figure Labels (Titles Only)",
        'alt-text': "Figure Alt Text (Visual Descriptions)",
        'caption': "Figure Captions (Full Explanatory Text)"
    }
    type_label = type_labels.get(extract_type, extract_type)
    
    lines = [
        f"## {type_label}",
        "",
    ]
    
    chapter_num = 0
    total_figures = 0
    
    for chapter in chapters_data:
        figures = chapter.get(f'figures_{extract_type}', [])
        if not figures:
            continue
        
        chapter_num += 1
        
        lines.append(f"### Chapter {chapter_num}: {chapter['title']}")
        lines.append("")
        
        for i, fig in enumerate(figures, 1):
            lines.append(f"**Figure {chapter_num}.{i}**: {fig['text']}")
            lines.append("")
            total_figures += 1
        
    lines.append(f"_Total: {total_figures} figures_")
    lines.append("")
    
    return lines


def generate_markdown_output(chapters_data: list[dict], output_path: Path, 
                            extract_types: list[str], vol: str) -> None:
    """
    Generate the Markdown output file with chapter and figure listings.
    """
    # Get volume title from YAML if possible
    vol_titles = {
        '1': 'Volume I: Introduction',
        '2': 'Volume II: Advanced Topics',
    }
    vol_title = vol_titles.get(vol, f'Volume {vol}')
    
    lines = [
        f"# Figure List - {vol_title}",
        "",
        "_Machine Learning Systems_",
        "",
        f"**Volume**: {vol}",
        f"**Extraction Types**: {', '.join(extract_types)}",
        "",
        "---",
        "",
    ]
    
    for extract_type in extract_types:
        section_lines = generate_markdown_section(chapters_data, extract_type, vol)
        lines.extend(section_lines)
        lines.append("---")
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
    python extract_figures.py --vol 2 --label            # Labels only for vol2
    python extract_figures.py --vol 1 --alt-text         # Alt text only for vol1
    python extract_figures.py --vol 1 --caption          # Captions only for vol1
    python extract_figures.py --vol 1 --label --caption  # Labels and captions for vol1
        """
    )
    
    parser.add_argument(
        '--vol',
        required=True,
        help='Volume number (1, 2, etc.) - REQUIRED'
    )
    
    parser.add_argument(
        '--label', 
        action='store_true',
        help='Extract just the figure title (bold text from fig-cap)'
    )
    parser.add_argument(
        '--alt-text', 
        action='store_true',
        help='Extract fig-alt (visual descriptions for accessibility)'
    )
    parser.add_argument(
        '--caption', 
        action='store_true',
        help='Extract full fig-cap (explanatory caption text)'
    )
    
    args = parser.parse_args()
    
    vol = args.vol
    
    # Determine extraction types (default to all if none specified)
    extract_types = []
    if args.label:
        extract_types.append('label')
    if args.alt_text:
        extract_types.append('alt-text')
    if args.caption:
        extract_types.append('caption')
    
    # If no types specified, use all three
    if not extract_types:
        extract_types = ['label', 'caption', 'alt-text']
        print("No output type specified. Generating all three: label, caption, alt-text")
    
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
        
        chapter_info = {
            'path': rel_path,
            'title': title,
        }
        
        # Extract figures for each requested type
        has_figures = False
        for extract_type in extract_types:
            figures = extract_figures(qmd_path, extract_type)
            chapter_info[f'figures_{extract_type}'] = figures
            if figures:
                has_figures = True
                print(f"    Found {len(figures)} figures ({extract_type})")
        
        if has_figures:
            chapters_data.append(chapter_info)
    
    # Generate output
    output_path = yaml_path.parent.parent / f'FIGURE_LIST_VOL{vol}.md'
    generate_markdown_output(chapters_data, output_path, extract_types, vol)


if __name__ == '__main__':
    main()
