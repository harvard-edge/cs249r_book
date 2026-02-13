#!/usr/bin/env python3
"""
Figure Manifest Generator for MIT Press (Post-Render)

Parses LaTeX's List of Figures (.lof) file to extract:
  - Figure numbers (as assigned by LaTeX)
  - Page numbers
  - Captions

Then enriches with alt-text extracted from QMD source files.

Output format (figure_manifest.txt):
  FIG|{label}|{fig_number}|{page}|{alt_text}|{caption}

Usage (run as post-render script):
  python generate_figure_manifest.py

Or manually:
  python generate_figure_manifest.py --lof _build/pdf-vol1/index.lof --qmd-dir contents/vol1
"""

import re
import sys
import os
from pathlib import Path


def parse_lof_file(lof_path: Path) -> list[dict]:
    """Parse LaTeX .lof file to extract figure metadata.
    
    LOF format:
    \contentsline {figure}{\numberline {1.1}{\ignorespaces Caption text}}{5}{figure.1.1}
    
    Returns list of dicts with: label, fig_number, page, caption
    """
    figures = []
    
    if not lof_path.exists():
        print(f"[Figure Manifest] LOF file not found: {lof_path}", file=sys.stderr)
        return figures
    
    content = lof_path.read_text(encoding='utf-8', errors='replace')
    
    # Pattern to match LOF entries
    # \contentsline {figure}{\numberline {1.1}{\ignorespaces Caption}}{5}{figure.1.1}
    lof_pattern = re.compile(
        r'\\contentsline\s*\{figure\}'
        r'\{\\numberline\s*\{([^}]+)\}'  # Figure number (1.1)
        r'\{\\ignorespaces\s*(.+?)\}\}'  # Caption text
        r'\{(\d+)\}'                      # Page number
        r'\{figure\.([^}]+)\}',           # Label reference (1.1 or custom)
        re.DOTALL
    )
    
    for match in lof_pattern.finditer(content):
        fig_number = match.group(1).strip()
        caption_raw = match.group(2).strip()
        page = match.group(3).strip()
        label_ref = match.group(4).strip()
        
        # Clean caption: remove LaTeX formatting
        caption = clean_latex(caption_raw)
        
        # Try to extract original label from hyperref target
        # The label_ref might be like "1.1" or the original "fig-xxx"
        label = f"fig-{label_ref}" if not label_ref.startswith('fig-') else label_ref
        
        figures.append({
            'label': label,
            'fig_number': fig_number,
            'page': page,
            'caption': caption,
            'alt_text': ''  # Will be filled from QMD
        })
    
    # Also try alternate LOF format (Quarto sometimes uses different format)
    alt_pattern = re.compile(
        r'\\contentsline\s*\{figure\}'
        r'\{\\numberline\s*\{([^}]+)\}'  # Figure number
        r'(.+?)\}'                         # Caption (might not have \ignorespaces)
        r'\{(\d+)\}'                       # Page
        r'\{([^}]+)\}',                    # Reference
        re.DOTALL
    )
    
    if not figures:
        for match in alt_pattern.finditer(content):
            fig_number = match.group(1).strip()
            caption_raw = match.group(2).strip()
            page = match.group(3).strip()
            label_ref = match.group(4).strip()
            
            caption = clean_latex(caption_raw)
            label = label_ref if label_ref.startswith('fig-') else f"figure-{label_ref}"
            
            figures.append({
                'label': label,
                'fig_number': fig_number,
                'page': page,
                'caption': caption,
                'alt_text': ''
            })
    
    return figures


def clean_latex(text: str) -> str:
    """Remove LaTeX commands and clean text."""
    # Remove common LaTeX commands
    text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textsuperscript\{([^}]*)\}', r'^\1', text)
    text = re.sub(r'\\textsubscript\{([^}]*)\}', r'_\1', text)
    text = re.sub(r'\\ignorespaces\s*', '', text)
    text = re.sub(r'\\relax\s*', '', text)
    text = re.sub(r'\\\w+\{([^}]*)\}', r'\1', text)  # Generic command removal
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove remaining commands
    text = re.sub(r'\{|\}', '', text)  # Remove braces
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_alt_text_from_qmd(qmd_dir: Path) -> dict[str, str]:
    """Extract label -> alt-text mapping from QMD files.
    
    Returns dict mapping figure labels to alt-text.
    """
    alt_texts = {}
    
    if not qmd_dir.exists():
        return alt_texts
    
    # Pattern for markdown images with fig-alt
    md_image_pattern = re.compile(
        r'!\[(?:[^\[\]]|\[[^\]]*\])*\]'  # ![caption]
        r'\([^)]+\)'                      # (path)
        r'\{([^}]*#fig-[^}]+)\}',         # {#fig-xxx ...attributes...}
        re.MULTILINE
    )
    
    # Pattern for fenced divs with fig-alt
    div_pattern = re.compile(
        r':{3,}\s*\{([^}]*#fig-[^}]+)\}',
        re.MULTILINE
    )
    
    for qmd_path in qmd_dir.rglob('*.qmd'):
        try:
            content = qmd_path.read_text(encoding='utf-8')
        except Exception:
            continue
        
        # Extract from markdown images
        for match in md_image_pattern.finditer(content):
            attrs_str = match.group(1)
            
            label_match = re.search(r'#(fig-[\w-]+)', attrs_str)
            alt_match = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs_str)
            
            if label_match and alt_match:
                alt_texts[label_match.group(1)] = alt_match.group(1)
        
        # Extract from fenced divs
        for match in div_pattern.finditer(content):
            attrs_str = match.group(1)
            
            label_match = re.search(r'#(fig-[\w-]+)', attrs_str)
            alt_match = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs_str)
            
            if label_match and alt_match:
                alt_texts[label_match.group(1)] = alt_match.group(1)
    
    return alt_texts


def escape_field(text: str) -> str:
    """Escape text for pipe-delimited output."""
    if not text:
        return ""
    text = text.replace('|', '│')
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_lof_file(build_dir: Path) -> Path | None:
    """Find the .lof file in the build directory."""
    # Try common locations
    candidates = [
        build_dir / 'index.lof',
        build_dir / 'book.lof',
    ]
    
    # Also search recursively
    for lof in build_dir.rglob('*.lof'):
        return lof
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate figure manifest from LOF and QMD files')
    parser.add_argument('--lof', type=Path, help='Path to .lof file')
    parser.add_argument('--qmd-dir', type=Path, help='Directory containing QMD files')
    parser.add_argument('--output', '-o', default='figure_manifest.txt', help='Output file')
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    quarto_dir = script_dir.parent
    
    # Get build directory from environment or default
    build_dir = Path(os.environ.get('QUARTO_PROJECT_OUTPUT_DIR', quarto_dir / '_build/pdf-vol1'))
    
    # Find LOF file
    lof_path = args.lof
    if not lof_path:
        lof_path = find_lof_file(build_dir)
    
    if not lof_path or not lof_path.exists():
        print(f"[Figure Manifest] No .lof file found in {build_dir}", file=sys.stderr)
        print("[Figure Manifest] Falling back to QMD-only extraction", file=sys.stderr)
        
        # Fall back to QMD extraction only (no page numbers)
        qmd_dir = args.qmd_dir or quarto_dir / 'contents'
        figures = extract_figures_from_qmd_only(qmd_dir)
    else:
        print(f"[Figure Manifest] Parsing LOF: {lof_path}", file=sys.stderr)
        
        # Parse LOF for figure metadata
        figures = parse_lof_file(lof_path)
        
        # Extract alt-text from QMD files
        qmd_dir = args.qmd_dir or quarto_dir / 'contents'
        alt_texts = extract_alt_text_from_qmd(qmd_dir)
        
        # Merge alt-text into figures
        for fig in figures:
            if fig['label'] in alt_texts:
                fig['alt_text'] = alt_texts[fig['label']]
    
    print(f"[Figure Manifest] Found {len(figures)} figures", file=sys.stderr)
    
    if not figures:
        print("[Figure Manifest] No figures found", file=sys.stderr)
        return
    
    # Write manifest
    output_path = quarto_dir / args.output
    with open(output_path, 'w') as f:
        f.write("# Figure Manifest for MIT Press\n")
        f.write("# Format: FIG|label|fig_number|page|alt_text|caption\n")
        for fig in figures:
            line = "FIG|{}|{}|{}|{}|{}\n".format(
                fig['label'],
                fig.get('fig_number', ''),
                fig.get('page', ''),
                escape_field(fig.get('alt_text', '')),
                escape_field(fig.get('caption', ''))
            )
            f.write(line)
    
    print(f"[Figure Manifest] Written to {output_path}", file=sys.stderr)


def extract_figures_from_qmd_only(qmd_dir: Path) -> list[dict]:
    """Fallback: Extract figures from QMD files when no LOF available."""
    figures = []
    
    if not qmd_dir.exists():
        return figures
    
    # Combined pattern for markdown images
    md_image_pattern = re.compile(
        r'!\[((?:[^\[\]]|\[[^\]]*\])*)\]'  # ![caption]
        r'\([^)]+\)'                        # (path)
        r'\{([^}]*#fig-[^}]+)\}',           # {#fig-xxx ...attributes...}
        re.MULTILINE
    )
    
    # Pattern for fenced divs
    div_pattern = re.compile(
        r':{3,}\s*\{([^}]*#fig-[^}]+)\}',
        re.MULTILINE
    )
    
    for qmd_path in sorted(qmd_dir.rglob('*.qmd')):
        try:
            content = qmd_path.read_text(encoding='utf-8')
        except Exception:
            continue
        
        # Extract from markdown images
        for match in md_image_pattern.finditer(content):
            caption = match.group(1).strip()
            attrs_str = match.group(2)
            
            label_match = re.search(r'#(fig-[\w-]+)', attrs_str)
            alt_match = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs_str)
            
            if label_match:
                figures.append({
                    'label': label_match.group(1),
                    'fig_number': '',
                    'page': '',
                    'alt_text': alt_match.group(1) if alt_match else '',
                    'caption': caption
                })
        
        # Extract from fenced divs
        for match in div_pattern.finditer(content):
            attrs_str = match.group(1)
            
            label_match = re.search(r'#(fig-[\w-]+)', attrs_str)
            cap_match = re.search(r'fig-cap=["\']([^"\']+)["\']', attrs_str)
            alt_match = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs_str)
            
            if label_match:
                figures.append({
                    'label': label_match.group(1),
                    'fig_number': '',
                    'page': '',
                    'alt_text': alt_match.group(1) if alt_match else '',
                    'caption': cap_match.group(1) if cap_match else ''
                })
    
    return figures


if __name__ == '__main__':
    main()
