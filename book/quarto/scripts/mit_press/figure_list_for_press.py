#!/usr/bin/env python3
"""
Figure List Generator for MIT Press

Extracts figure metadata from QMD files and generates a clean list with:
  - Figure number (chapter.figure format)
  - Caption
  - Alt-text

Usage:
  python figure_list_for_press.py --vol 1
  python figure_list_for_press.py --vol 1 --format markdown
  python figure_list_for_press.py --vol 1 --format csv
"""

import re
import sys
from pathlib import Path
from collections import OrderedDict


def extract_figures_from_qmd(qmd_path: Path) -> list[dict]:
    """Extract figure metadata from a QMD file."""
    figures = []
    content = qmd_path.read_text(encoding='utf-8')
    
    # Pattern for markdown images: ![caption](path){#fig-label fig-alt="..."}
    md_image_pattern = re.compile(
        r'!\[((?:[^\[\]]|\[[^\]]*\])*)\]'  # ![caption] - handles [@cite]
        r'\([^)]+\)'                        # (path)
        r'\{([^}]*#fig-[^}]+)\}',           # {#fig-xxx ...attrs...}
        re.MULTILINE
    )
    
    # Pattern for fenced divs: :::: {#fig-xxx fig-cap="..." fig-alt="..."}
    div_pattern = re.compile(
        r':{3,}\s*\{([^}]*#fig-[^}]+)\}',
        re.MULTILINE
    )
    
    # Pattern for executable code block figures (```{python} with #| label: fig-xxx)
    code_block_pattern = re.compile(
        r'```\{(?:python|r|julia|ojs)\}[^\n]*\n'  # Opening fence
        r'((?:#\|[^\n]*\n)+)',                     # Cell options (one or more #| lines)
        re.MULTILINE
    )
    
    # Extract markdown images
    for match in md_image_pattern.finditer(content):
        caption = match.group(1).strip()
        attrs = match.group(2)
        
        label = re.search(r'#(fig-[\w-]+)', attrs)
        alt = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs)
        
        if label:
            figures.append({
                'label': label.group(1),
                'caption': clean_text(caption),
                'alt_text': alt.group(1) if alt else '',
                'position': match.start()
            })
    
    # Extract fenced divs
    for match in div_pattern.finditer(content):
        attrs = match.group(1)
        
        label = re.search(r'#(fig-[\w-]+)', attrs)
        cap = re.search(r'fig-cap=["\']([^"\']+)["\']', attrs)
        alt = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs)
        
        if label:
            figures.append({
                'label': label.group(1),
                'caption': cap.group(1) if cap else '',
                'alt_text': alt.group(1) if alt else '',
                'position': match.start()
            })
    
    # Extract executable code block figures (```{python} with #| label: fig-xxx)
    for match in code_block_pattern.finditer(content):
        cell_options = match.group(1)
        
        # Extract label - must be a figure label
        label_match = re.search(r'#\|\s*label:\s*(fig-[\w-]+)', cell_options)
        if not label_match:
            continue
        
        # Extract caption (can be single or double quoted)
        cap_match = re.search(r'#\|\s*fig-cap:\s*["\']([^"\']+)["\']', cell_options)
        
        # Extract alt-text
        alt_match = re.search(r'#\|\s*fig-alt:\s*["\']([^"\']+)["\']', cell_options)
        
        figures.append({
            'label': label_match.group(1),
            'caption': clean_text(cap_match.group(1)) if cap_match else '',
            'alt_text': alt_match.group(1) if alt_match else '',
            'position': match.start()
        })
    
    # Sort by position in file
    figures.sort(key=lambda x: x['position'])
    return figures


def clean_text(text: str) -> str:
    """Clean caption/alt text."""
    if not text:
        return ""
    # Remove markdown bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_chapter_files(vol: int) -> list[tuple[int, Path]]:
    """Get ordered list of chapter files for a volume."""
    script_dir = Path(__file__).parent
    quarto_dir = script_dir.parent.parent  # scripts/mit_press -> scripts -> quarto
    content_dir = quarto_dir / f'contents/vol{vol}'
    
    # Chapter order for Vol 1 (based on typical structure)
    # You may need to adjust this based on your actual book structure
    chapter_order = [
        'introduction',
        'ml_systems', 
        'workflow',
        'data_engineering',
        'dl_primer',
        'dnn_architectures',
        'frameworks',
        'training',
        'data_selection',
        'optimizations',
        'hw_acceleration',
        'benchmarking',
        'serving',
        'ops',
        'responsible_engr',
        'conclusion',
    ]
    
    chapters = []
    chapter_num = 1
    
    for chapter_name in chapter_order:
        chapter_dir = content_dir / chapter_name
        if chapter_dir.exists():
            # Find the main QMD file
            qmd_candidates = [
                chapter_dir / f'{chapter_name}.qmd',
                chapter_dir / 'index.qmd',
            ]
            for qmd in qmd_candidates:
                if qmd.exists():
                    chapters.append((chapter_num, qmd))
                    chapter_num += 1
                    break
    
    # Also check for appendices
    appendix_dir = content_dir / 'backmatter'
    if appendix_dir.exists():
        for qmd in sorted(appendix_dir.glob('appendix_*.qmd')):
            chapters.append((f'A{len(chapters) - chapter_num + 2}', qmd))
    
    return chapters


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate figure list for MIT Press')
    parser.add_argument('--vol', type=int, default=1, help='Volume number')
    parser.add_argument('--format', choices=['text', 'markdown', 'csv'], default='text')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    args = parser.parse_args()
    
    chapters = get_chapter_files(args.vol)
    
    if not chapters:
        print(f"No chapters found for Volume {args.vol}", file=sys.stderr)
        sys.exit(1)
    
    all_figures = []
    
    for chapter_num, qmd_path in chapters:
        figures = extract_figures_from_qmd(qmd_path)
        for i, fig in enumerate(figures, 1):
            fig['number'] = f"{chapter_num}.{i}"
            fig['chapter'] = qmd_path.stem
            all_figures.append(fig)
    
    # Output
    output = []
    
    if args.format == 'markdown':
        output.append(f"# Figure List - Volume {args.vol}")
        output.append(f"\nTotal: {len(all_figures)} figures\n")
        
        current_chapter = None
        for fig in all_figures:
            if fig['chapter'] != current_chapter:
                current_chapter = fig['chapter']
                output.append(f"\n## {current_chapter.replace('_', ' ').title()}\n")
            
            output.append(f"### Figure {fig['number']}: `{fig['label']}`")
            output.append(f"\n**Caption:** {fig['caption']}")
            if fig['alt_text']:
                output.append(f"\n**Alt-text:** {fig['alt_text']}")
            output.append("\n---\n")
    
    elif args.format == 'csv':
        output.append("Chapter,Figure Number,Label,Caption,Alt-Text")
        for fig in all_figures:
            chapter = fig['chapter'].replace('_', ' ').title()
            cap = fig['caption'].replace('"', '""')
            alt = fig['alt_text'].replace('"', '""')
            output.append(f'"{chapter}","{fig["number"]}","{fig["label"]}","{cap}","{alt}"')
    
    else:  # text
        output.append(f"FIGURE LIST - VOLUME {args.vol}")
        output.append(f"Total: {len(all_figures)} figures")
        output.append("=" * 80)
        output.append("")
        
        current_chapter = None
        for fig in all_figures:
            if fig['chapter'] != current_chapter:
                current_chapter = fig['chapter']
                output.append(f"\n{'=' * 80}")
                output.append(f"CHAPTER: {current_chapter.replace('_', ' ').upper()}")
                output.append(f"{'=' * 80}\n")
            
            output.append(f"Figure {fig['number']}: {fig['label']}")
            output.append(f"  Caption: {fig['caption']}")
            if fig['alt_text']:
                output.append(f"  Alt-text: {fig['alt_text']}")
            output.append("-" * 80)
            output.append("")
    
    result = '\n'.join(output)
    
    if args.output:
        Path(args.output).write_text(result)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == '__main__':
    main()
