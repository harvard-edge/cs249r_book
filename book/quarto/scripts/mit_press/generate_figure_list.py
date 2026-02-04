#!/usr/bin/env python3
"""
Figure List Generator for MIT Press

Usage:
  Pre-render:  python generate_figure_list.py --clear
  Post-render: python generate_figure_list.py

Pre-render clears stale LaTeX data from previous builds.
Post-render creates a complete figure list with:
  - Figure numbers and page numbers (from LaTeX output)
  - Labels, captions, alt-text (from QMD sources)

Output: FIGURE_LIST.txt in the same directory as the PDF
"""

import re
import os
import sys
from pathlib import Path


def clear_cache(quarto_dir: Path) -> None:
    """Clear stale figure data from previous builds."""
    latex_file = quarto_dir / 'index_figures.txt'
    
    if latex_file.exists():
        latex_file.unlink()
        print(f"[Figure List] Cleared: {latex_file.name}", file=sys.stderr)
    else:
        print(f"[Figure List] Clean (no stale data)", file=sys.stderr)


def parse_latex_figures(latex_file: Path) -> list[dict]:
    """Parse LaTeX-generated figure manifest."""
    figures = []
    
    if not latex_file.exists():
        return figures
    
    content = latex_file.read_text()
    pattern = re.compile(r'Figure\s+(\d+\.\d+)\s*\|\s*Page\s*(\d+)')
    
    for match in pattern.finditer(content):
        figures.append({
            'number': match.group(1),
            'page': match.group(2)
        })
    
    return figures


def get_chapter_order_from_config(quarto_dir: Path) -> list[Path]:
    """Get chapter QMD files in order from Quarto config."""
    import yaml
    
    # Try active config first, then vol1 config
    config_paths = [
        quarto_dir / '_quarto.yml',
        quarto_dir / 'config/_quarto-pdf-vol1.yml',
    ]
    
    for config_path in config_paths:
        if not config_path.exists():
            continue
        
        # Follow symlink if needed
        if config_path.is_symlink():
            config_path = config_path.resolve()
        
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception:
            continue
        
        if not config or 'book' not in config:
            continue
        
        qmd_files = []
        chapters = config['book'].get('chapters', [])
        
        for item in chapters:
            if isinstance(item, str) and item.endswith('.qmd'):
                qmd_path = quarto_dir / item
                if qmd_path.exists():
                    qmd_files.append(qmd_path)
            elif isinstance(item, dict) and 'chapters' in item:
                # Part with chapters
                for ch in item['chapters']:
                    if isinstance(ch, str) and ch.endswith('.qmd'):
                        qmd_path = quarto_dir / ch
                        if qmd_path.exists():
                            qmd_files.append(qmd_path)
        
        # Also get appendices
        appendices = config['book'].get('appendices', [])
        for item in appendices:
            if isinstance(item, str) and item.endswith('.qmd'):
                qmd_path = quarto_dir / item
                if qmd_path.exists():
                    qmd_files.append(qmd_path)
        
        if qmd_files:
            return qmd_files
    
    # Fallback: scan contents directory
    content_dir = quarto_dir / 'contents/vol1'
    if content_dir.exists():
        return sorted(content_dir.rglob('*.qmd'))
    
    return []


def extract_qmd_figures(quarto_dir: Path, scan_all: bool = False) -> list[dict]:
    """Extract figures from QMD files in config order (or scan all if specified)."""
    figures = []
    
    if scan_all:
        # Scan all QMD files in contents directory
        content_dir = quarto_dir / 'contents/vol1'
        qmd_files = sorted(content_dir.rglob('*.qmd')) if content_dir.exists() else []
        print(f"[Figure List] Scanning all: {len(qmd_files)} QMD files", file=sys.stderr)
    else:
        qmd_files = get_chapter_order_from_config(quarto_dir)
        print(f"[Figure List] From config: {len(qmd_files)} QMD files", file=sys.stderr)
    
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
    
    # Pattern for executable code block figures (```{python} with #| label: fig-xxx)
    # Captures the entire cell options block to extract label, fig-cap, fig-alt
    code_block_pattern = re.compile(
        r'```\{(?:python|r|julia|ojs)\}[^\n]*\n'  # Opening fence
        r'((?:#\|[^\n]*\n)+)',                     # Cell options (one or more #| lines)
        re.MULTILINE
    )
    
    for qmd_path in qmd_files:
        # Skip index, 404, and parts files
        if qmd_path.name in ['index.qmd', '404.qmd'] or 'parts' in str(qmd_path):
            continue
        
        try:
            content = qmd_path.read_text(encoding='utf-8')
        except Exception:
            continue
        
        # Get chapter name from file/directory
        chapter_name = qmd_path.stem
        if chapter_name == 'index':
            chapter_name = qmd_path.parent.name
        chapter_title = chapter_name.replace('_', ' ').title()
        
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
                    'chapter': chapter_title,
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
                    'chapter': chapter_title,
                    'position': match.start()
                })
        
        # Extract executable code block figures (```{python} with #| label: fig-xxx)
        for match in code_block_pattern.finditer(content):
            cell_options = match.group(1)
            
            # Extract label - must be a figure label
            label_match = re.search(r'#\|\s*label:\s*(fig-[\w-]+)', cell_options)
            if not label_match:
                continue
            
            # Extract caption (can be single or double quoted, may span lines)
            cap_match = re.search(r'#\|\s*fig-cap:\s*["\']([^"\']+)["\']', cell_options)
            
            # Extract alt-text
            alt_match = re.search(r'#\|\s*fig-alt:\s*["\']([^"\']+)["\']', cell_options)
            
            chapter_figures.append({
                'label': label_match.group(1),
                'caption': cap_match.group(1) if cap_match else '',
                'alt_text': alt_match.group(1) if alt_match else '',
                'chapter': chapter_title,
                'position': match.start()
            })
        
        chapter_figures.sort(key=lambda x: x['position'])
        figures.extend(chapter_figures)
    
    return figures


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate figure list for MIT Press')
    parser.add_argument('--clear', action='store_true',
                        help='Clear stale LaTeX data (run as pre-render)')
    parser.add_argument('--scan-all', action='store_true', 
                        help='Scan all QMD files instead of reading from config')
    parser.add_argument('--output', '-o', help='Output directory (default: from QUARTO_PROJECT_OUTPUT_DIR)')
    args = parser.parse_args()
    
    # Get quarto directory
    script_dir = Path(__file__).parent
    quarto_dir = script_dir.parent.parent
    
    # Pre-render mode: just clear cache and exit
    if args.clear:
        clear_cache(quarto_dir)
        return
    
    # Get output directory from Quarto environment or use default
    output_dir = args.output or os.environ.get('QUARTO_PROJECT_OUTPUT_DIR', '')
    if output_dir:
        output_dir = Path(output_dir)
    else:
        # Default to _build/pdf-vol1
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent.parent / '_build/pdf-vol1'
    
    # Find LaTeX figures file (written by header-includes.tex)
    latex_file = quarto_dir / 'index_figures.txt'
    
    latex_figures = parse_latex_figures(latex_file)
    print(f"[Figure List] LaTeX: {len(latex_figures)} figures with page numbers", file=sys.stderr)
    
    # Extract QMD figures (reads chapter order from config, or scans all)
    qmd_figures = extract_qmd_figures(quarto_dir, scan_all=args.scan_all)
    print(f"[Figure List] QMD: {len(qmd_figures)} figures with metadata", file=sys.stderr)
    
    # Merge: only include figures if counts match (both from same build)
    if len(latex_figures) != len(qmd_figures):
        print(f"[Figure List] WARNING: LaTeX ({len(latex_figures)}) and QMD ({len(qmd_figures)}) counts don't match!", file=sys.stderr)
        print(f"[Figure List] The LaTeX file may be from a previous build.", file=sys.stderr)
        print(f"[Figure List] Delete 'index_figures.txt' and rebuild to fix.", file=sys.stderr)
    
    # Merge by position - trust QMD count as source of truth
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
            fig['number'] = '?'
            fig['page'] = '?'
        
        merged.append(fig)
    
    # Generate readable text output
    lines = []
    lines.append("=" * 80)
    lines.append("FIGURE LIST FOR MIT PRESS")
    lines.append(f"Total: {len(merged)} figures")
    lines.append("=" * 80)
    lines.append("")
    
    current_chapter = None
    for fig in merged:
        # Chapter header
        if fig['chapter'] != current_chapter:
            current_chapter = fig['chapter']
            lines.append("")
            lines.append("-" * 80)
            lines.append(f"CHAPTER: {current_chapter.upper()}")
            lines.append("-" * 80)
            lines.append("")
        
        # Figure entry
        fig_num = fig['number'] if fig['number'] else '?'
        page = fig['page'] if fig['page'] else '?'
        
        lines.append(f"Figure {fig_num} (Page {page})")
        lines.append(f"Label: {fig['label']}")
        lines.append(f"Caption: {fig['caption']}")
        if fig['alt_text']:
            lines.append(f"Alt-text: {fig['alt_text']}")
        lines.append("")
    
    # Write to output directory
    output_path = output_dir / 'FIGURE_LIST.txt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines))
    
    print(f"[Figure List] Written: {output_path}", file=sys.stderr)
    print(f"[Figure List] Total: {len(merged)} figures", file=sys.stderr)


if __name__ == '__main__':
    main()
