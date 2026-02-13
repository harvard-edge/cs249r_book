#!/usr/bin/env python3
"""
Figure Manifest Extractor for MIT Press

Parses QMD files directly to extract figure metadata that Quarto's crossref
system would otherwise consume before Lua filters can access it.

Output format (figure_manifest.txt):
  FIG|{label}|{source_file}|{alt_text}|{caption}

Usage:
  python extract_figure_manifest.py                    # Process active config
  python extract_figure_manifest.py --vol 1           # Process Volume 1
  python extract_figure_manifest.py --files a.qmd b.qmd  # Process specific files
"""

import re
import sys
import yaml
from pathlib import Path


def extract_figures_from_qmd(qmd_path: Path) -> list[dict]:
    """Extract figure metadata from a QMD file.
    
    Handles two figure syntaxes:
    1. Markdown images: ![caption](path){#fig-label fig-alt="..."}
    2. Fenced divs: :::: {#fig-label fig-cap="..." fig-alt="..."}
    """
    figures = []
    content = qmd_path.read_text(encoding='utf-8')
    source_file = qmd_path.name
    
    # Pattern 1: Markdown images with attributes
    # ![caption text](image/path.png){#fig-label fig-pos='t' fig-alt="alt text"}
    # Note: Caption can contain nested brackets like [@citation] so we use a more robust pattern
    md_image_pattern = re.compile(
        r'!\[((?:[^\[\]]|\[[^\]]*\])*)\]'  # ![caption] - handles nested brackets like [@cite]
        r'\(([^)]+)\)'                      # (path)
        r'\{([^}]*#fig-[^}]+)\}',           # {#fig-xxx ...attributes...}
        re.MULTILINE
    )
    
    for match in md_image_pattern.finditer(content):
        caption = match.group(1).strip()
        # group(2) is path, group(3) is attributes
        attrs_str = match.group(3)
        
        # Extract label
        label_match = re.search(r'#(fig-[\w-]+)', attrs_str)
        label = label_match.group(1) if label_match else ""
        
        # Extract fig-alt
        alt_match = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs_str)
        alt_text = alt_match.group(1) if alt_match else ""
        
        if label:
            figures.append({
                'label': label,
                'source': source_file,
                'alt_text': clean_text(alt_text),
                'caption': clean_text(caption)
            })
    
    # Pattern 2: Fenced divs with figure attributes
    # :::: {#fig-xxx fig-env="figure" fig-cap="caption" fig-alt="alt"}
    div_pattern = re.compile(
        r':{3,}\s*\{([^}]*#fig-[^}]+)\}',
        re.MULTILINE
    )
    
    for match in div_pattern.finditer(content):
        attrs_str = match.group(1)
        
        # Extract label
        label_match = re.search(r'#(fig-[\w-]+)', attrs_str)
        label = label_match.group(1) if label_match else ""
        
        # Extract fig-cap
        cap_match = re.search(r'fig-cap=["\']([^"\']+)["\']', attrs_str)
        caption = cap_match.group(1) if cap_match else ""
        
        # Extract fig-alt  
        alt_match = re.search(r'fig-alt=["\']([^"\']+)["\']', attrs_str)
        alt_text = alt_match.group(1) if alt_match else ""
        
        if label:
            figures.append({
                'label': label,
                'source': source_file,
                'alt_text': clean_text(alt_text),
                'caption': clean_text(caption)
            })
    
    return figures


def clean_text(text: str) -> str:
    """Clean text for manifest output."""
    if not text:
        return ""
    # Replace pipes with Unicode vertical bar
    text = text.replace('|', '│')
    # Replace newlines with spaces
    text = text.replace('\n', ' ').replace('\r', '')
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    return text.strip()


def get_qmd_files_from_config(config_path: Path) -> list[Path]:
    """Extract QMD file paths from a Quarto config."""
    if not config_path.exists():
        print(f"[Figure Manifest] Config not found: {config_path}", file=sys.stderr)
        return []
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    qmd_files = []
    base_dir = config_path.parent.parent  # quarto/ directory
    
    # Extract from book.chapters
    if 'book' in config and 'chapters' in config['book']:
        for item in config['book']['chapters']:
            if isinstance(item, str) and item.endswith('.qmd'):
                qmd_path = base_dir / item
                if qmd_path.exists():
                    qmd_files.append(qmd_path)
            elif isinstance(item, dict):
                # Handle part definitions with chapters list
                if 'chapters' in item:
                    for ch in item['chapters']:
                        if isinstance(ch, str) and ch.endswith('.qmd'):
                            qmd_path = base_dir / ch
                            if qmd_path.exists():
                                qmd_files.append(qmd_path)
    
    # Extract from book.appendices
    if 'book' in config and 'appendices' in config['book']:
        for item in config['book']['appendices']:
            if isinstance(item, str) and item.endswith('.qmd'):
                qmd_path = base_dir / item
                if qmd_path.exists():
                    qmd_files.append(qmd_path)
    
    return qmd_files


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Extract figure manifest from QMD files')
    parser.add_argument('--vol', type=int, help='Volume number (1 or 2)')
    parser.add_argument('--files', nargs='+', help='Specific QMD files to process')
    parser.add_argument('--output', '-o', default='figure_manifest.txt', help='Output file')
    parser.add_argument('--append', '-a', action='store_true', help='Append to existing file')
    args = parser.parse_args()
    
    # Check for Quarto environment variables (set during pre/post-render)
    quarto_input_files = os.environ.get('QUARTO_PROJECT_INPUT_FILES', '')
    quarto_output_dir = os.environ.get('QUARTO_PROJECT_OUTPUT_DIR', '')
    
    # Determine project root (script is in quarto/scripts/)
    script_dir = Path(__file__).parent
    quarto_dir = script_dir.parent
    
    qmd_files = []
    
    if args.files:
        # Process specific files from command line
        for f in args.files:
            path = Path(f)
            if not path.is_absolute():
                path = quarto_dir / f
            if path.exists():
                qmd_files.append(path)
            else:
                print(f"[Figure Manifest] File not found: {f}", file=sys.stderr)
    
    elif quarto_input_files:
        # Use Quarto environment variable (set during pre/post-render)
        print(f"[Figure Manifest] Using QUARTO_PROJECT_INPUT_FILES", file=sys.stderr)
        for f in quarto_input_files.split('\n'):
            f = f.strip()
            if f and f.endswith('.qmd'):
                path = Path(f)
                if path.exists():
                    qmd_files.append(path)
                else:
                    # Try relative to quarto_dir
                    path = quarto_dir / f
                    if path.exists():
                        qmd_files.append(path)
    
    elif args.vol:
        # Process volume config
        config_path = quarto_dir / f'config/_quarto-pdf-vol{args.vol}.yml'
        qmd_files = get_qmd_files_from_config(config_path)
    
    else:
        # Check for active config (symlink)
        active_config = quarto_dir / '_quarto.yml'
        if active_config.exists():
            qmd_files = get_qmd_files_from_config(active_config)
        
        # If no files from config, scan content directories
        if not qmd_files:
            print("[Figure Manifest] No chapters in config, scanning content directories...", file=sys.stderr)
            content_dirs = [
                quarto_dir / 'contents/vol1',
                quarto_dir / 'contents/vol2',
            ]
            for content_dir in content_dirs:
                if content_dir.exists():
                    for qmd in content_dir.rglob('*.qmd'):
                        # Skip index, parts, and other non-chapter files
                        if qmd.name not in ['index.qmd', '404.qmd'] and 'parts' not in str(qmd):
                            qmd_files.append(qmd)
        
        if not qmd_files:
            print("[Figure Manifest] No QMD files found. Use --vol or --files", file=sys.stderr)
            # Don't fail the build, just skip
            sys.exit(0)
    
    if not qmd_files:
        print("[Figure Manifest] No QMD files to process", file=sys.stderr)
        sys.exit(1)
    
    print(f"[Figure Manifest] Processing {len(qmd_files)} QMD files", file=sys.stderr)
    
    # Extract figures from all files
    all_figures = []
    for qmd_path in qmd_files:
        figures = extract_figures_from_qmd(qmd_path)
        all_figures.extend(figures)
        if figures:
            print(f"[Figure Manifest] {qmd_path.name}: {len(figures)} figures", file=sys.stderr)
    
    print(f"[Figure Manifest] Total: {len(all_figures)} figures", file=sys.stderr)
    
    # Write manifest
    output_path = quarto_dir / args.output
    mode = 'a' if args.append else 'w'
    
    with open(output_path, mode) as f:
        for fig in all_figures:
            line = f"FIG|{fig['label']}|{fig['source']}|{fig['alt_text']}|{fig['caption']}\n"
            f.write(line)
    
    print(f"[Figure Manifest] Written to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
