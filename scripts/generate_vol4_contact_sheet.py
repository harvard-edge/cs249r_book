#!/usr/bin/env python3
"""
generate_vol4_contact_sheet.py
==============================
Generates a comprehensive, self-contained HTML visual contact sheet for Volume IV.
Audits and visualizes:
1. All authentic hardware photographic plates (including annotated failure mechanisms).
2. All body architectural and system SVGs.
3. All unnumbered margin visual figures.
4. Duplication check across chapters (strictly zero repeated figures).
5. Visual verification check: converts all SVGs to 1200px PNGs using rsvg-convert.
"""

import hashlib
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VOL4_DIR = REPO_ROOT / "books" / "vol4"
TMP_PNG_DIR = Path("/tmp/vol4_contact_pngs")
OUTPUT_HTML = VOL4_DIR / "vol4_visual_contact_sheet.html"


def get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:12]


def rasterize_svg(svg_path: Path, out_png: Path) -> bool:
    if out_png.exists() and out_png.stat().st_mtime >= svg_path.stat().st_mtime:
        return True
    try:
        subprocess.run(
            ["/opt/homebrew/bin/rsvg-convert", "-w", "1200", str(svg_path), "-o", str(out_png)],
            check=True,
            capture_output=True
        )
        return True
    except Exception as e:
        print(f"Warning: failed to rasterize {svg_path.name}: {e}")
        return False


def collect_chapter_assets():
    chapters = sorted([d for d in VOL4_DIR.iterdir() if d.is_dir() and (d.name.startswith("0") or d.name.startswith("1") or d.name == "backmatter")])
    
    catalog = []
    seen_hashes = {}
    duplicates = []

    TMP_PNG_DIR.mkdir(parents=True, exist_ok=True)

    for ch in chapters:
        ch_title = ch.name
        qmd_file = ch / f"{ch.name}.qmd"
        if not qmd_file.exists() and ch.name == "backmatter":
            qmd_file = ch / "appendix_spa.qmd"
        
        # Read chapter title from qmd if available
        if qmd_file.exists():
            text = qmd_file.read_text(encoding="utf-8")
            title_m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            if title_m:
                ch_title = f"{ch.name.upper()}: {title_m.group(1)}"

        images_dir = ch / "images"
        plates = []
        body_svgs = []
        margin_svgs = []

        if images_dir.exists():
            # 1. Hardware photographic plates
            jpg_dir = images_dir / "jpg"
            if jpg_dir.exists():
                for jpg in sorted(jpg_dir.glob("*.jpg")):
                    fhash = get_file_hash(jpg)
                    if fhash in seen_hashes:
                        duplicates.append((jpg, seen_hashes[fhash]))
                    else:
                        seen_hashes[fhash] = jpg

                    is_annotated = "_annotated" in jpg.name
                    plates.append({
                        "path": jpg,
                        "rel_path": str(jpg.relative_to(VOL4_DIR)),
                        "name": jpg.name,
                        "is_annotated": is_annotated,
                        "size_kb": round(jpg.stat().st_size / 1024, 1),
                        "hash": fhash
                    })

            # 2. SVGs
            svg_dir = images_dir / "svg"
            if svg_dir.exists():
                for svg in sorted(svg_dir.glob("*.svg")):
                    # Exclude shared SPA locator icons from duplicate check
                    is_locator = "spa_locator" in svg.name
                    fhash = get_file_hash(svg)
                    if not is_locator:
                        if fhash in seen_hashes:
                            duplicates.append((svg, seen_hashes[fhash]))
                        else:
                            seen_hashes[fhash] = svg

                    png_target = TMP_PNG_DIR / f"{ch.name}_{svg.stem}.png"
                    rasterize_svg(svg, png_target)

                    item = {
                        "path": svg,
                        "rel_path": str(svg.relative_to(VOL4_DIR)),
                        "png_path": str(png_target),
                        "name": svg.name,
                        "size_kb": round(svg.stat().st_size / 1024, 1),
                        "hash": fhash
                    }

                    if svg.name.startswith("margin_"):
                        margin_svgs.append(item)
                    else:
                        body_svgs.append(item)

        catalog.append({
            "chapter_id": ch.name,
            "chapter_title": ch_title,
            "plates": plates,
            "body_svgs": body_svgs,
            "margin_svgs": margin_svgs
        })

    return catalog, duplicates


def build_html(catalog, duplicates):
    total_plates = sum(len(c["plates"]) for c in catalog)
    annotated_plates = sum(sum(1 for p in c["plates"] if p["is_annotated"]) for c in catalog)
    total_body_svgs = sum(len(c["body_svgs"]) for c in catalog)
    total_margin_svgs = sum(len(c["margin_svgs"]) for c in catalog)
    total_assets = total_plates + total_body_svgs + total_margin_svgs

    dup_badge = f'<span class="badge badge-success">0 Duplicates (Zero Reuse Enforced)</span>' if not duplicates else f'<span class="badge badge-error">{len(duplicates)} DUPLICATES FOUND!</span>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLSysBook Volume IV — Visual Contact Sheet & Asset Audit</title>
<style>
  :root {{
    --bg-main: #0B0F19;
    --bg-card: #111827;
    --bg-card-hover: #1F2937;
    --border: #374151;
    --border-accent: #38BDF8;
    --text-main: #F9FAFB;
    --text-muted: #9CA3AF;
    --accent-blue: #38BDF8;
    --accent-green: #34D399;
    --accent-red: #F87171;
    --accent-amber: #FBBF24;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-main);
    color: var(--text-main);
    font-family: var(--font-sans);
    line-height: 1.5;
    padding: 2rem;
  }}
  header {{
    border-bottom: 2px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }}
  h1 {{ font-size: 1.8rem; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 0.75rem; }}
  .subtitle {{ color: var(--text-muted); font-size: 0.95rem; margin-top: 0.35rem; }}
  .stats-bar {{
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin-top: 1.25rem;
  }}
  .stat-chip {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-size: 0.85rem;
  }}
  .stat-chip strong {{ color: var(--accent-blue); font-size: 1.1rem; }}
  .badge {{
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 3px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .badge-success {{ background: #064E3B; color: #6EE7B7; border: 1px solid #059669; }}
  .badge-error {{ background: #7F1D1D; color: #FCA5A5; border: 1px solid #DC2626; }}
  .badge-plate {{ background: #1E3A8A; color: #93C5FD; border: 1px solid #2563EB; }}
  .badge-svg {{ background: #374151; color: #D1D5DB; border: 1px solid #4B5563; }}
  .badge-margin {{ background: #065F46; color: #A7F3D0; border: 1px solid #047857; }}

  .chapter-block {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 2.5rem;
    overflow: hidden;
  }}
  .chapter-header {{
    background: #1E293B;
    border-bottom: 1px solid var(--border);
    padding: 0.85rem 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .chapter-title {{ font-size: 1.15rem; font-weight: 600; color: #F8FAFC; }}
  .chapter-counts {{ font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono); }}
  .section-label {{
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--accent-blue);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 1rem 1.25rem 0.5rem;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
    gap: 1.25rem;
    padding: 1rem 1.25rem 1.5rem;
  }}
  .card {{
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 4px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }}
  .card:hover {{
    border-color: var(--accent-blue);
    transform: translateY(-2px);
  }}
  .thumb-container {{
    width: 100%;
    height: 280px;
    background: #020617;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    cursor: pointer;
    position: relative;
    border-bottom: 1px solid #334155;
  }}
  .thumb-container img {{
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }}
  .card-meta {{
    padding: 0.85rem;
    font-size: 0.8rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }}
  .card-filename {{
    font-family: var(--font-mono);
    font-weight: 600;
    color: #F1F5F9;
    word-break: break-all;
  }}
  .card-details {{
    display: flex;
    justify-content: space-between;
    color: var(--text-muted);
    font-size: 0.75rem;
  }}
  .modal {{
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0; top: 0;
    width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.9);
    align-items: center;
    justify-content: center;
    padding: 2rem;
  }}
  .modal.active {{ display: flex; }}
  .modal img {{
    max-width: 95vw;
    max-height: 95vh;
    border: 2px solid var(--border-accent);
    border-radius: 4px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
  }}
</style>
</head>
<body>

<header>
  <h1><span>📚</span> Volume IV: Physical AI Systems — Visual Contact Sheet</h1>
  <div class="subtitle">Complete catalog of architectural schematics, unnumbered margin figures, and publication hardware plates with physical breaking points.</div>
  <div class="stats-bar">
    <div class="stat-chip">Total Visual Assets: <strong>{total_assets}</strong></div>
    <div class="stat-chip">Hardware Plates: <strong>{total_plates}</strong> ({annotated_plates} publication annotated)</div>
    <div class="stat-chip">Body Architecture SVGs: <strong>{total_body_svgs}</strong></div>
    <div class="stat-chip">Margin Figures: <strong>{total_margin_svgs}</strong></div>
    <div class="stat-chip">Uniqueness Status: {dup_badge}</div>
  </div>
</header>
"""

    for ch in catalog:
        ch_id = ch["chapter_id"]
        ch_title = ch["chapter_title"]
        num_plates = len(ch["plates"])
        num_body = len(ch["body_svgs"])
        num_margin = len(ch["margin_svgs"])

        if num_plates == 0 and num_body == 0 and num_margin == 0:
            continue

        html += f"""
<div class="chapter-block" id="{ch_id}">
  <div class="chapter-header">
    <div class="chapter-title">{ch_title}</div>
    <div class="chapter-counts">{num_plates} Plates | {num_body} Body SVGs | {num_margin} Margin SVGs</div>
  </div>
"""

        # 1. Hardware plates
        if num_plates > 0:
            html += f'<div class="section-label">📷 Authentic Hardware Plates ({num_plates})</div><div class="grid">'
            for p in ch["plates"]:
                badge = '<span class="badge badge-success">Publication Annotated</span>' if p["is_annotated"] else '<span class="badge badge-plate">Hardware Photo</span>'
                html += f"""
    <div class="card">
      <div class="thumb-container" onclick="openModal('{p["rel_path"]}')">
        <img src="{p["rel_path"]}" alt="{p["name"]}" loading="lazy">
      </div>
      <div class="card-meta">
        <div class="card-filename">{p["name"]}</div>
        <div class="card-details">
          <span>{p["size_kb"]} KB | SHA: {p["hash"]}</span>
          {badge}
        </div>
      </div>
    </div>
"""
            html += '</div>'

        # 2. Body SVGs
        if num_body > 0:
            html += f'<div class="section-label">📐 Body Architecture & System Schematics ({num_body})</div><div class="grid">'
            for s in ch["body_svgs"]:
                html += f"""
    <div class="card">
      <div class="thumb-container" onclick="openModal('{s["rel_path"]}')">
        <img src="{s["rel_path"]}" alt="{s["name"]}" loading="lazy">
      </div>
      <div class="card-meta">
        <div class="card-filename">{s["name"]}</div>
        <div class="card-details">
          <span>{s["size_kb"]} KB | SHA: {s["hash"]}</span>
          <span class="badge badge-svg">Body SVG</span>
        </div>
      </div>
    </div>
"""
            html += '</div>'

        # 3. Margin SVGs
        if num_margin > 0:
            html += f'<div class="section-label">📌 Unnumbered Margin Visual Figures ({num_margin})</div><div class="grid">'
            for m in ch["margin_svgs"]:
                html += f"""
    <div class="card">
      <div class="thumb-container" onclick="openModal('{m["rel_path"]}')">
        <img src="{m["rel_path"]}" alt="{m["name"]}" loading="lazy">
      </div>
      <div class="card-meta">
        <div class="card-filename">{m["name"]}</div>
        <div class="card-details">
          <span>{m["size_kb"]} KB | SHA: {m["hash"]}</span>
          <span class="badge badge-margin">Margin Visual</span>
        </div>
      </div>
    </div>
"""
            html += '</div>'

        html += '</div>'

    html += """
<div id="modal" class="modal" onclick="closeModal()">
  <img id="modal-img" src="" alt="Enlarged view">
</div>

<script>
  function openModal(src) {
    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modal-img');
    modalImg.src = src;
    modal.classList.add('active');
  }
  function closeModal() {
    document.getElementById('modal').classList.remove('active');
  }
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') closeModal();
  });
</script>
</body>
</html>
"""
    return html


def main():
    print("Collecting Volume IV visual assets and checking uniqueness...")
    catalog, duplicates = collect_chapter_assets()

    if duplicates:
        print(f"⚠️ WARNING: {len(duplicates)} duplicate files detected across chapters:")
        for f1, f2 in duplicates:
            print(f"  - {f1.relative_to(VOL4_DIR)} duplicates {f2.relative_to(VOL4_DIR)}")
    else:
        print("✅ PASS: 0 duplicate figures detected. Every visual asset is unique across chapters.")

    html = build_html(catalog, duplicates)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ Generated standalone visual contact sheet at: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
