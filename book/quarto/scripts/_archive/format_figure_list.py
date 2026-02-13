#!/usr/bin/env python3
"""
Generate FIGURE_LIST for MIT Press.

ONLY runs when:
  1. LaTeX manifest exists (full PDF build completed)
  2. Scans only chapters listed in the volume's _quarto.yml

Output: FIGURE_LIST_VOL{N}.md in the build output directory.

Post-render script - add to _quarto.yml:
  project:
    post-render:
      - scripts/format_figure_list.py
"""

import os
import re
import yaml
from collections import defaultdict
from pathlib import Path


def get_volume_chapters(project_dir: Path, vol: str) -> list[Path]:
    """
    Read _quarto.yml to get list of chapters for this volume.
    Returns list of QMD file paths.
    """
    chapters = []
    
    # Try volume-specific config first
    config_path = project_dir / f"config/_quarto-pdf-vol{vol}.yml"
    if not config_path.exists():
        config_path = project_dir / "_quarto.yml"
    
    if not config_path.exists():
        return chapters
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except:
        return chapters
    
    # Extract chapter files from book.chapters
    book_config = config.get('book', {})
    chapter_entries = book_config.get('chapters', [])
    
    for entry in chapter_entries:
        if isinstance(entry, str):
            # Simple file path
            if entry.endswith('.qmd'):
                chapters.append(project_dir / entry)
        elif isinstance(entry, dict):
            # Part with chapters
            part_chapters = entry.get('chapters', [])
            for ch in part_chapters:
                if isinstance(ch, str) and ch.endswith('.qmd'):
                    chapters.append(project_dir / ch)
    
    return chapters


def scan_chapter_images(qmd_files: list[Path], project_dir: Path) -> dict[str, dict]:
    """
    Scan specific QMD files for images.
    Returns {label_or_path: image_data}
    """
    images = {}
    
    for qmd_file in qmd_files:
        if not qmd_file.exists():
            continue
        
        try:
            content = qmd_file.read_text(encoding='utf-8')
        except:
            continue
        
        rel_path = str(qmd_file.relative_to(project_dir))
        
        # Pattern 1: Markdown images ![...](path){...}
        for match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)(?:\{([^}]*)\})?', content):
            caption, img_path, attrs = match.groups()
            attrs = attrs or ''
            line_num = content[:match.start()].count('\n') + 1
            
            label_match = re.search(r'#(fig-[a-zA-Z0-9_-]+)', attrs)
            label = label_match.group(1) if label_match else None
            
            alt_match = re.search(r'fig-alt="([^"]*)"', attrs)
            if not alt_match:
                alt_match = re.search(r"fig-alt='([^']*)'", attrs)
            alt_text = alt_match.group(1) if alt_match else ''
            
            key = label if label else img_path
            images[key] = {
                'alt_text': alt_text,
                'caption': caption.strip(),
                'file_path': img_path,
                'source_file': rel_path,
                'line_num': line_num,
                'is_figure': bool(label),
                'label': label,
            }
        
        # Pattern 2: Div figures :::: {#fig-xxx ... fig-alt="..."}
        # Match the entire line to handle } inside quoted strings
        for match in re.finditer(r'^:{3,}\s*\{(.+)\}\s*$', content, re.MULTILINE):
            attrs = match.group(1)
            
            # Check if this is a figure
            label_match = re.search(r'#(fig-[a-zA-Z0-9_-]+)', attrs)
            if not label_match:
                continue
            
            label = label_match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            # Extract fig-alt - find the quoted value after fig-alt=
            alt_match = re.search(r'fig-alt="((?:[^"\\]|\\.)*)"', attrs)
            alt_text = alt_match.group(1) if alt_match else ''
            
            # Extract fig-cap - find the quoted value after fig-cap=
            cap_match = re.search(r'fig-cap="((?:[^"\\]|\\.)*)"', attrs)
            caption = cap_match.group(1) if cap_match else ''
            
            if label not in images:
                images[label] = {
                    'alt_text': alt_text,
                    'caption': caption,
                    'file_path': '',
                    'source_file': rel_path,
                    'line_num': line_num,
                    'is_figure': True,
                    'label': label,
                }
        
        # Pattern 3: Code cell figures
        for match in re.finditer(r'```\{[^}]*\}[^\n]*\n((?:#\|[^\n]*\n)+)', content):
            cell_meta = match.group(1)
            
            label_match = re.search(r'#\|\s*label:\s*(fig-[a-zA-Z0-9_-]+)', cell_meta)
            if not label_match:
                continue
            
            label = label_match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            # Extract fig-alt - handle quoted strings
            alt_match = re.search(r'#\|\s*fig-alt:\s*"([^"]+)"', cell_meta)
            if not alt_match:
                alt_match = re.search(r"#\|\s*fig-alt:\s*'([^']+)'", cell_meta)
            if not alt_match:
                alt_match = re.search(r'#\|\s*fig-alt:\s*(.+?)$', cell_meta, re.MULTILINE)
            alt_text = alt_match.group(1).strip() if alt_match else ''
            
            # Extract fig-cap - handle quoted strings  
            cap_match = re.search(r'#\|\s*fig-cap:\s*"([^"]+)"', cell_meta)
            if not cap_match:
                cap_match = re.search(r"#\|\s*fig-cap:\s*'([^']+)'", cell_meta)
            if not cap_match:
                cap_match = re.search(r'#\|\s*fig-cap:\s*(.+?)$', cell_meta, re.MULTILINE)
            caption = cap_match.group(1).strip() if cap_match else ''
            
            if label not in images:
                images[label] = {
                    'alt_text': alt_text,
                    'caption': caption,
                    'file_path': '',
                    'source_file': rel_path,
                    'line_num': line_num,
                    'is_figure': True,
                    'label': label,
                }
    
    return images


def parse_latex_manifest(path: Path) -> dict[str, dict]:
    """Parse LaTeX manifest. Returns {label: {number, page, caption}}"""
    figures = {}
    
    if not path.exists():
        return figures
    
    content = path.read_text(encoding='utf-8', errors='replace').strip()
    if not content:
        return figures
    
    for line in content.split('\n'):
        line = line.strip()
        if not line.startswith('FIG|'):
            continue
        
        parts = line.split('|', 4)
        if len(parts) >= 5:
            label = parts[1]
            figures[label] = {
                'number': parts[2],
                'page': parts[3],
                'caption_latex': parts[4],
            }
    
    return figures


def clean_text(text: str) -> str:
    """Remove LaTeX/Markdown formatting."""
    clean = text
    clean = re.sub(r'\\textbf\{([^}]+)\}', r'\1', clean)
    clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)
    clean = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', clean)
    clean = re.sub(r'\\[a-zA-Z]+', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def main():
    project_dir = Path(os.getenv("QUARTO_PROJECT_DIR", "."))
    output_dir = Path(os.getenv("QUARTO_PROJECT_OUTPUT_DIR", "_build"))
    
    # Determine volume
    vol = "2" if "vol2" in str(output_dir).lower() else "1"
    
    # Check for LaTeX manifest - if empty/missing, this wasn't a full PDF build
    manifest_path = project_dir / "figure_manifest.txt"
    if not manifest_path.exists():
        manifest_path = output_dir / "figure_manifest.txt"
    
    latex_figs = parse_latex_manifest(manifest_path) if manifest_path.exists() else {}
    
    # Get chapters for this volume
    qmd_files = get_volume_chapters(project_dir, vol)
    
    if not qmd_files:
        # Fallback: scan contents/vol{N}/ if config parsing failed
        vol_dir = project_dir / f"contents/vol{vol}"
        if vol_dir.exists():
            qmd_files = list(vol_dir.rglob("*.qmd"))
    
    if not qmd_files:
        print(f"[figure_list] No chapters found for volume {vol}")
        return
    
    # Scan only the volume's chapters
    all_images = scan_chapter_images(qmd_files, project_dir)
    
    if not all_images and not latex_figs:
        print("[figure_list] No images found")
        return
    
    # Check if we have LaTeX data
    has_latex = bool(latex_figs)
    
    # Build figure list
    figures = []
    decorative = []
    
    for key, img in all_images.items():
        if img['is_figure']:
            lx = latex_figs.get(img['label'], {})
            figures.append({
                'label': img['label'],
                'number': lx.get('number', ''),  # Empty if no LaTeX
                'page': lx.get('page', ''),
                'title': clean_text(img['caption'])[:60] if img['caption'] else '',
                'caption': clean_text(img['caption'] or lx.get('caption_latex', '')),
                'alt_text': img['alt_text'],
                'source_file': img['source_file'],
                'line_num': img['line_num'],
            })
        else:
            decorative.append({
                'file_path': img['file_path'],
                'alt_text': img['alt_text'],
                'source_file': img['source_file'],
                'line_num': img['line_num'],
            })
    
    # Sort by source file and line number
    figures.sort(key=lambda f: (f['source_file'], f['line_num']))
    
    # If no LaTeX data, assign provisional numbers based on source order
    if not has_latex and figures:
        # Group by source file (chapter) and assign numbers
        from collections import Counter
        file_counter = Counter()
        chapter_map = {}  # source_file -> chapter number
        
        # Assign chapter numbers based on order of appearance
        ch_num = 1
        for f in figures:
            sf = f['source_file']
            if sf not in chapter_map:
                chapter_map[sf] = ch_num
                ch_num += 1
        
        # Assign figure numbers within each chapter
        for f in figures:
            sf = f['source_file']
            ch = chapter_map[sf]
            file_counter[sf] += 1
            f['number'] = f"{ch}.{file_counter[sf]}"
            f['page'] = '—'
    
    # Sort by figure number
    def sort_key(f):
        num = f['number']
        if not num or num == '—':
            return (999, 999)
        parts = num.split('.')
        try:
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        except:
            return (999, 999)
    
    figures.sort(key=sort_key)
    
    # Stats
    total_figs = len(figures)
    total_dec = len(decorative)
    missing_alt_figs = [f for f in figures if not f['alt_text']]
    missing_alt_dec = [d for d in decorative if not d['alt_text']]
    total_missing = len(missing_alt_figs) + len(missing_alt_dec)
    
    # Group figures by chapter
    chapters = defaultdict(list)
    for fig in figures:
        ch = fig['number'].split('.')[0] if '.' in fig['number'] else '—'
        chapters[ch].append(fig)
    
    # Generate output
    lines = [
        f"# FIGURE LIST — Volume {vol}",
        "",
        "_Machine Learning Systems_",
        "",
    ]
    
    if not has_latex:
        lines.extend([
            "```",
            "ℹ️  PREVIEW BUILD — Numbers are provisional (based on source order).",
            "   Full PDF build will show final LaTeX numbers and page references.",
            "```",
            "",
        ])
    
    # Banner
    if total_missing > 0:
        lines.extend([
            "```",
            f"⚠️  {total_missing} IMAGES MISSING ALT TEXT",
            f"   • {len(missing_alt_figs)} figures",
            f"   • {len(missing_alt_dec)} decorative",
            "```",
            "",
        ])
    else:
        lines.extend(["```", "✅ ALL IMAGES HAVE ALT TEXT", "```", ""])
    
    # Summary
    lines.extend([
        "---",
        "",
        "## Summary",
        "",
        f"**Figures**: {total_figs}",
        f"**Decorative**: {total_dec}",
        "",
    ])
    
    if has_latex and chapters:
        lines.extend([
            "### By Chapter",
            "",
            "| Ch | Figs | Alt |",
            "|:--:|:----:|:---:|",
        ])
        for ch in sorted(chapters.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            figs = chapters[ch]
            n_missing = sum(1 for f in figs if not f['alt_text'])
            status = "✅" if n_missing == 0 else f"⚠️{n_missing}"
            lines.append(f"| {ch} | {len(figs)} | {status} |")
        lines.append("")
    
    # Issues
    if missing_alt_figs:
        lines.extend(["---", "", "## ⚠️ Figures Missing Alt Text", ""])
        for f in missing_alt_figs:
            lines.append(f"- `{f['label']}` — {f['source_file']}:{f['line_num']}")
        lines.append("")
    
    if missing_alt_dec:
        lines.extend(["## ⚠️ Decorative Missing Alt Text", ""])
        for d in missing_alt_dec:
            lines.append(f"- `{d['file_path']}` — {d['source_file']}:{d['line_num']}")
        lines.append("")
    
    # Full list with all details
    if figures:
        lines.extend(["---", "", "## Complete List", ""])
        for ch in sorted(chapters.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            figs = chapters[ch]
            lines.append(f"### Chapter {ch}")
            lines.append("")
            for f in figs:
                # Header line
                page_info = f"(p. {f['page']})" if f['page'] and f['page'] != '—' else ""
                lines.append(f"#### Figure {f['number']} {page_info}")
                lines.append("")
                
                # Label
                lines.append(f"**Label:** `{f['label']}`")
                
                # Source
                lines.append(f"**Source:** `{f['source_file']}:{f['line_num']}`")
                
                # Caption
                cap_status = "✅" if f['caption'] else "❌ MISSING"
                if f['caption']:
                    lines.append(f"**Caption:** {cap_status}")
                    lines.append(f"> {f['caption']}")
                else:
                    lines.append(f"**Caption:** {cap_status}")
                
                # Alt text
                alt_status = "✅" if f['alt_text'] else "❌ MISSING"
                if f['alt_text']:
                    lines.append(f"**Alt Text:** {alt_status}")
                    lines.append(f"> {f['alt_text']}")
                else:
                    lines.append(f"**Alt Text:** {alt_status}")
                
                lines.append("")
            lines.append("---")
            lines.append("")
    
    lines.append(f"_Total: {total_figs} figures, {total_dec} decorative_")
    
    # Write
    out_path = output_dir / f"FIGURE_LIST_VOL{vol}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    
    status = "✅" if total_missing == 0 else f"⚠️ {total_missing} missing alt"
    print(f"[figure_list] {status} — {out_path.name} ({total_figs} figs)")
    
    # Move manifest
    if manifest_path.exists() and manifest_path.parent != output_dir:
        try:
            (output_dir / manifest_path.name).write_text(manifest_path.read_text())
            manifest_path.unlink()
        except:
            pass


if __name__ == '__main__':
    main()
