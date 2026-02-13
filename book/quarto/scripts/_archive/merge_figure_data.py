#!/usr/bin/env python3
"""
Merge Figure Data from LaTeX and QMD Sources

Combines:
  - LaTeX output: Figure numbers and page numbers (from *_figures.txt)
  - QMD source: Labels, captions, alt-text (parsed from .qmd files)

Usage:
  python merge_figure_data.py --vol 1

Output: FIGURE_LIST_VOL1_COMPLETE.csv with all fields
"""

import re
import sys
from pathlib import Path


MANIFEST_HEADER = 'LATEX FIGURE MANIFEST'


def find_latex_manifest(quarto_dir: Path, output_dir: Path | None = None) -> Path | None:
    r"""Find the LaTeX figure manifest written by header-includes.tex.

    LaTeX writes \jobname_figures.txt into the quarto root during PDF
    compilation.  The post-render step in generate_figure_list.py moves
    it into the build output directory.  This function searches both
    locations and returns the most recently modified match, or None.
    """
    candidates: list[Path] = []
    for directory in filter(None, [output_dir, quarto_dir]):
        if directory.exists():
            candidates.extend(
                f for f in directory.glob('*_figures.txt')
                if MANIFEST_HEADER in f.read_text(encoding='utf-8')[:200]
            )
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_latex_figures(latex_file: Path) -> list[dict]:
    """Parse LaTeX-generated figure manifest."""
    figures = []
    
    if not latex_file or not latex_file.exists():
        print(f"[Merge] LaTeX file not found", file=sys.stderr)
        return figures
    
    content = latex_file.read_text()
    
    # Pattern: Figure X.Y | Page Z
    pattern = re.compile(r'Figure\s+(\d+\.\d+)\s*\|\s*Page\s*(\d+)')
    
    for match in pattern.finditer(content):
        figures.append({
            'number': match.group(1),
            'page': match.group(2)
        })
    
    return figures


def extract_qmd_figures(content_dir: Path, chapter_order: list[str]) -> list[dict]:
    """Extract figures from QMD files in order."""
    figures = []
    
    # Patterns for figure extraction
    md_image_pattern = re.compile(
        r'!\[((?:[^\[\]]|\[[^\]]*\])*)\]'
        r'\([^)]+\)'
        r'\{([^}]*#fig-[^}]+)\}',
        re.MULTILINE
    )
    
    div_pattern = re.compile(
        r':{3,}\s*\{([^}]*#fig-[^}]+)\}',
        re.MULTILINE
    )
    
    for chapter_name in chapter_order:
        chapter_dir = content_dir / chapter_name
        if not chapter_dir.exists():
            continue
        
        qmd_path = chapter_dir / f'{chapter_name}.qmd'
        if not qmd_path.exists():
            qmd_path = chapter_dir / 'index.qmd'
        if not qmd_path.exists():
            continue
        
        content = qmd_path.read_text(encoding='utf-8')
        chapter_figures = []
        
        # Extract markdown images
        for match in md_image_pattern.finditer(content):
            caption = match.group(1).strip()
            attrs = match.group(2)
            
            label = re.search(r'#(fig-[\w-]+)', attrs)
            alt = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs)
            
            if label:
                chapter_figures.append({
                    'label': label.group(1),
                    'caption': caption,
                    'alt_text': alt.group(1) if alt else '',
                    'chapter': chapter_name.replace('_', ' ').title(),
                    'position': match.start()
                })
        
        # Extract fenced divs
        for match in div_pattern.finditer(content):
            attrs = match.group(1)
            
            label = re.search(r'#(fig-[\w-]+)', attrs)
            cap = re.search(r'fig-cap=["\']([^"\']+)["\']', attrs)
            alt = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs)
            
            if label:
                chapter_figures.append({
                    'label': label.group(1),
                    'caption': cap.group(1) if cap else '',
                    'alt_text': alt.group(1) if alt else '',
                    'chapter': chapter_name.replace('_', ' ').title(),
                    'position': match.start()
                })
        
        # Sort by position within chapter
        chapter_figures.sort(key=lambda x: x['position'])
        figures.extend(chapter_figures)
    
    return figures


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge LaTeX and QMD figure data')
    parser.add_argument('--vol', type=int, default=1, help='Volume number')
    parser.add_argument('--output', '-o', help='Output file')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    quarto_dir = script_dir.parent.parent
    build_dir = quarto_dir / f'_build/pdf-vol{args.vol}'
    
    # Parse LaTeX figures (find manifest dynamically — \jobname varies by build)
    latex_file = find_latex_manifest(quarto_dir, build_dir)
    if latex_file:
        print(f"[Merge] LaTeX manifest: {latex_file.name}", file=sys.stderr)
    else:
        print(f"[Merge] WARNING: No LaTeX manifest (*_figures.txt) found", file=sys.stderr)
    latex_figures = parse_latex_figures(latex_file)
    print(f"[Merge] LaTeX: {len(latex_figures)} figures with page numbers", file=sys.stderr)
    
    # Chapter order for Vol 1
    chapter_order = [
        'introduction', 'ml_systems', 'workflow', 'data_engineering',
        'nn_computation', 'nn_architectures', 'frameworks', 'training',
        'data_selection', 'optimizations', 'hw_acceleration', 'benchmarking',
        'serving', 'ops', 'responsible_engr', 'conclusion',
    ]
    
    content_dir = quarto_dir / f'contents/vol{args.vol}'
    qmd_figures = extract_qmd_figures(content_dir, chapter_order)
    print(f"[Merge] QMD: {len(qmd_figures)} figures with metadata", file=sys.stderr)
    
    # Merge: match by position (1st LaTeX figure = 1st QMD figure, etc.)
    merged = []
    for i, qmd_fig in enumerate(qmd_figures):
        fig = {
            'chapter': qmd_fig['chapter'],
            'label': qmd_fig['label'],
            'caption': qmd_fig['caption'],
            'alt_text': qmd_fig['alt_text'],
        }
        
        if i < len(latex_figures):
            fig['number'] = latex_figures[i]['number']
            fig['page'] = latex_figures[i]['page']
        else:
            fig['number'] = f"?.{i+1}"
            fig['page'] = '?'
        
        merged.append(fig)
    
    print(f"[Merge] Merged: {len(merged)} figures", file=sys.stderr)
    
    # Output CSV
    output_lines = ["Chapter,Figure Number,Page,Label,Caption,Alt-Text"]
    for fig in merged:
        cap = fig['caption'].replace('"', '""')
        alt = fig['alt_text'].replace('"', '""')
        output_lines.append(
            f'"{fig["chapter"]}","{fig["number"]}","{fig["page"]}","{fig["label"]}","{cap}","{alt}"'
        )
    
    result = '\n'.join(output_lines)
    
    output_path = args.output or script_dir / f'FIGURE_LIST_VOL{args.vol}_COMPLETE.csv'
    Path(output_path).write_text(result)
    print(f"[Merge] Written to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
