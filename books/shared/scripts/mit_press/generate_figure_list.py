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


MANIFEST_HEADER = 'LATEX FIGURE MANIFEST'


def find_figure_aux(quarto_dir: Path, output_dir: Path) -> Path | None:
    """Find the current master AUX, including the post-render preserved copy."""
    candidates = list(output_dir.glob('*.aux')) + list(quarto_dir.glob('*.aux'))
    return max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)


def parse_aux_figures(aux_file: Path) -> dict[str, dict[str, str]]:
    """Read figure labels, numbers, and Arabic or Roman folios from LaTeX."""
    pattern = re.compile(r'\\newlabel\{(fig-[^}]+)\}\{\{([^{}]+)\}\{([^{}]+)\}')
    return {label: {'number': number, 'page': page}
            for label, number, page in pattern.findall(aux_file.read_text(encoding='utf-8'))
            if not label.endswith('@cref')}


def _is_latex_manifest(path: Path) -> bool:
    """Check whether a file is a LaTeX figure manifest."""
    try:
        return MANIFEST_HEADER in path.read_text(encoding='utf-8')[:200]
    except Exception:
        return False


def _find_manifests_in(directory: Path) -> list[Path]:
    """Return all *_figures.txt files in *directory* that are LaTeX manifests."""
    if not directory.exists():
        return []
    return [f for f in directory.glob('*_figures.txt') if _is_latex_manifest(f)]


def find_latex_manifest(quarto_dir: Path, output_dir: Path | None = None) -> Path | None:
    r"""Find the LaTeX figure manifest written by header-includes.tex.

    LaTeX writes \jobname_figures.txt into the quarto root during PDF
    compilation.  The post-render step moves it into the build output
    directory so the source tree stays clean.  This function searches
    *output_dir* first (moved copy), then *quarto_dir* (fresh from LaTeX).
    Returns the most recently modified match, or None.
    """
    candidates: list[Path] = []
    if output_dir:
        candidates.extend(_find_manifests_in(output_dir))
    candidates.extend(_find_manifests_in(quarto_dir))

    if not candidates:
        return None

    # Most recently modified wins
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def clear_cache(quarto_dir: Path, output_dir: Path | None = None) -> None:
    """Clear stale figure manifests from previous builds.

    Searches both the quarto root (where LaTeX writes) and the build
    output directory (where post-render moves them).
    """
    cleared = False
    for directory in filter(None, [quarto_dir, output_dir]):
        for f in _find_manifests_in(directory):
            f.unlink()
            print(f"[Figure List] Cleared: {f.relative_to(quarto_dir) if f.is_relative_to(quarto_dir) else f}", file=sys.stderr)
            cleared = True
    if not cleared:
        print("[Figure List] Clean (no stale data)", file=sys.stderr)


def parse_latex_figures(latex_file: Path) -> list[dict]:
    """Parse LaTeX-generated figure manifest."""
    figures = []
    
    if not latex_file.exists():
        return figures
    
    content = latex_file.read_text()
    pattern = re.compile(r'Figure\s+([A-Z\d]+\.\d+)\s*\|\s*Page\s*(\d+)')
    
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
    content_dir = quarto_dir / 'vol1'
    if content_dir.exists():
        return sorted(content_dir.rglob('*.qmd'))
    
    return []


def extract_qmd_figures(quarto_dir: Path, scan_all: bool = False) -> list[dict]:
    """Extract figures from QMD files in config order (or scan all if specified)."""
    figures = []
    
    if scan_all:
        # Scan all QMD files in contents directory
        content_dir = quarto_dir / 'vol1'
        qmd_files = sorted(content_dir.rglob('*.qmd')) if content_dir.exists() else []
        print(f"[Figure List] Scanning all: {len(qmd_files)} QMD files", file=sys.stderr)
    else:
        qmd_files = get_chapter_order_from_config(quarto_dir)
        print(f"[Figure List] From config: {len(qmd_files)} QMD files", file=sys.stderr)
    
    md_image_pattern = re.compile(
        r'!\[((?:[^\[\]]|\[[^\]]*\])*)\]'
        r'\([^)]+\)'
        r'\{(.*#fig-[^\n]*)\}\s*$',
        re.MULTILINE
    )
    
    # Use greedy .* anchored to end-of-line so interior } (from LaTeX like
    # $W_{hh}$, \mathcal{F}, \index{...}) don't truncate the match.
    div_pattern = re.compile(
        r':{3,}\s*\{(.*#fig-.*)\}\s*$',
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
        # Front matter and part openers can contain numbered PDF figures too.
        if qmd_path.name == '404.qmd' or qmd_path.name.startswith('_shelved'):
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
        body = re.sub(r'\A---\s*\n.*?\n---\s*\n', '', content, count=1, flags=re.DOTALL)
        heading = re.search(r'^#\s+(.+?)(?:\s+\{[^}]*\})?\s*$', body, re.MULTILINE)
        if heading:
            chapter_title = heading.group(1)
        
        chapter_figures = []
        
        # Extract markdown images
        for match in md_image_pattern.finditer(content):
            caption = match.group(1).strip()
            attrs = match.group(2)
            
            label = re.search(r'#(fig-[\w-]+)', attrs)
            alt = re.search(r'fig-alt="([^"]*)"', attrs)

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
            # Use double-quote-only matching so LaTeX primes like F'(x)
            # aren't treated as closing delimiters.
            cap = re.search(r'fig-cap="([^"]*)"', attrs)
            alt = re.search(r'fig-alt="([^"]*)"', attrs)
            
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
            
            # Extract caption — match double-quote delimiters only so
            # apostrophes (e.g., Moore's) don't truncate the match.
            cap_match = re.search(r'#\|\s*fig-cap:\s*"([^"]*)"', cell_options)

            # Extract alt-text
            alt_match = re.search(r'#\|\s*fig-alt:\s*"([^"]*)"', cell_options)
            
            chapter_figures.append({
                'label': label_match.group(1),
                'caption': cap_match.group(1) if cap_match else '',
                'alt_text': alt_match.group(1) if alt_match else '',
                'chapter': chapter_title,
                'position': match.start()
            })
        
        chapter_figures.sort(key=lambda x: x['position'])

        # Deduplicate: when a ::: div wraps a code block, both patterns
        # match the same figure.  Keep the entry with the richer caption.
        seen_labels = {}
        for fig in chapter_figures:
            lbl = fig['label']
            if lbl in seen_labels:
                prev = seen_labels[lbl]
                # Keep whichever has a non-empty caption (prefer first)
                if not prev['caption'] and fig['caption']:
                    seen_labels[lbl] = fig
            else:
                seen_labels[lbl] = fig
        chapter_figures = sorted(seen_labels.values(), key=lambda x: x['position'])

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
    quarto_dir = Path(__file__).resolve().parents[3]  # books/shared/scripts/mit_press -> books

    # Get output directory from Quarto environment or use default
    output_dir = Path(
        args.output
        or os.environ.get('QUARTO_PROJECT_OUTPUT_DIR', '')
        or str(quarto_dir / '_build/pdf-vol1')
    )
    
    # Pre-render mode: just clear cache and exit
    if args.clear:
        clear_cache(quarto_dir, output_dir)
        return
    
    # The AUX records stable labels and folios. Positional matching can shift
    # every caption when a front-matter figure has a Roman folio or a figure
    # is conditionally omitted from PDF.
    aux_file = find_figure_aux(quarto_dir, output_dir)
    if aux_file:
        print(f"[Figure List] LaTeX AUX: {aux_file.name}", file=sys.stderr)
    else:
        raise RuntimeError("Current PDF AUX is required to export figure numbers and pages")
    
    latex_figures = parse_aux_figures(aux_file)
    print(f"[Figure List] LaTeX: {len(latex_figures)} figures with page numbers", file=sys.stderr)
    
    # Extract QMD figures (reads chapter order from config, or scans all)
    qmd_figures = extract_qmd_figures(quarto_dir, scan_all=args.scan_all)
    print(f"[Figure List] QMD: {len(qmd_figures)} figures with metadata", file=sys.stderr)
    
    # Export only figures present in this PDF; preserve source metadata by ID.
    merged = []
    for qmd_fig in qmd_figures:
        location = latex_figures.get(qmd_fig['label'])
        if location is None:
            print(f"[Figure List] Not in this PDF: {qmd_fig['label']}", file=sys.stderr)
            continue
        fig = {
            'chapter': qmd_fig['chapter'],
            'label': qmd_fig['label'],
            'caption': qmd_fig['caption'],
            'alt_text': qmd_fig['alt_text'],
        }
        
        fig.update(location)
        
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
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    
    print(f"[Figure List] Written: {output_path}", file=sys.stderr)
    print(f"[Figure List] Total: {len(merged)} figures", file=sys.stderr)

    # --- Cleanup: move LaTeX manifest from source tree into _build/ ---
    # LaTeX can only write into the quarto root (TeX working directory).
    # Move the manifest into the build output so the source tree stays clean.
    for manifest in _find_manifests_in(quarto_dir):
        dest = output_dir / manifest.name
        try:
            import shutil
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(manifest), str(dest))
            print(f"[Figure List] Moved {manifest.name} → {output_dir.name}/", file=sys.stderr)
        except Exception as exc:
            print(f"[Figure List] WARNING: Could not move {manifest.name}: {exc}", file=sys.stderr)


if __name__ == '__main__':
    main()
