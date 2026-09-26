#!/usr/bin/env python3
"""
generate_vol4_individual_contact_sheets.py
==========================================
Generates individual visual contact sheets for all Volume IV chapters (Chapters 01 to 17),
plus a master index hub (`books/vol4/contact_sheets/index.html`) and updates
`books/vol4/vol4_visual_contact_sheet.html`.

Solves the "huge figure" problem:
- Balanced, scannable default thumbnail sizing (minmax(250px, 1fr)) with max-height 200px.
- Interactive Zoom controls (Compact 190px, Standard 260px, Detail 380px).
- Status filtering: All, Placed in Book, Unplaced / Drafts.
- Clean, book-proof light-neutral canvas with subtle checkerboard for transparent SVGs.
- Full metadata per figure: dimensions, aspect ratio, placement status (line #, label, caption),
  category (plate, body SVG, margin figure, SPA locator), and SVG compliance checks.
- Keyboard-navigable full-resolution modal proofing with "Open Raw" button.
"""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
VOL4_DIR = REPO_ROOT / "books" / "vol4"
OUTPUT_DIR = VOL4_DIR / "contact_sheets"
TMP_PNG_DIR = Path("/tmp/vol4_contact_pngs")

RSVG_CONVERT = "/opt/homebrew/bin/rsvg-convert" if Path("/opt/homebrew/bin/rsvg-convert").exists() else "rsvg-convert"


def get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:10]


def rasterize_svg(svg_path: Path, out_png: Path) -> bool:
    if out_png.exists() and out_png.stat().st_mtime >= svg_path.stat().st_mtime:
        return True
    try:
        subprocess.run(
            [RSVG_CONVERT, "-w", "1200", str(svg_path), "-o", str(out_png)],
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False


def parse_svg_metadata(svg_path: Path) -> dict:
    meta = {
        "viewBox": None,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "has_root_font": False,
        "has_outer_stroke": False,
        "outer_stroke_val": None,
        "has_bubbly_box": False,
    }
    try:
        content = svg_path.read_text(encoding="utf-8")
    except Exception:
        return meta

    # Root font-family
    root_m = re.search(r"<svg\b[^>]*>", content)
    if root_m:
        tag = root_m.group(0)
        meta["has_root_font"] = "font-family=" in tag

        # viewBox
        vb_m = re.search(r'viewBox=[\'"]([0-9\.\s\-]+)[\'"]', tag)
        if vb_m:
            parts = [float(p) for p in vb_m.group(1).split()]
            if len(parts) == 4:
                meta["viewBox"] = f"{int(parts[2])} × {int(parts[3])}"
                meta["width"] = parts[2]
                meta["height"] = parts[3]
                if parts[3] > 0:
                    meta["aspect_ratio"] = f"{parts[2]/parts[3]:.2f}:1"

    # Outer canvas border stroke
    for r_m in re.finditer(r"<rect\b[^>]*>", content):
        tag = r_m.group(0)
        is_full = any(w in tag for w in ('width="100%"', "width='100%'", 'width="680"', 'width="540"', 'width="800"'))
        if is_full:
            strk_m = re.search(r'stroke=[\'"]([^\'"]+)[\'"]', tag)
            if strk_m:
                val = strk_m.group(1).strip().lower()
                if val not in ("none", "transparent", ""):
                    meta["has_outer_stroke"] = True
                    meta["outer_stroke_val"] = val
                    break

    # Bubbly boxes
    for r_m in re.finditer(r"<rect\b([^>]*)>", content):
        tag = r_m.group(1)
        rx_m = re.search(r'rx=[\'"]([0-9\.]+)[\'"]', tag)
        w_m = re.search(r'width=[\'"]([0-9\.]+)[\'"]', tag)
        h_m = re.search(r'height=[\'"]([0-9\.]+)[\'"]', tag)
        if rx_m and w_m and h_m:
            rx, w, h = float(rx_m.group(1)), float(w_m.group(1)), float(h_m.group(1))
            if rx > 6 and (w > 120 or h > 60):
                meta["has_bubbly_box"] = True
                break

    return meta


def parse_chapter_qmd(qmd_path: Path) -> Dict[str, dict]:
    """Parse chapter .qmd file for image placements, labels, captions, and margin status."""
    if not qmd_path.exists():
        return {}

    lines = qmd_path.read_text(encoding="utf-8").splitlines()
    placements = {}

    current_div_label = None
    current_div_cap = None
    current_div_alt = None
    in_margin = False

    for idx, line in enumerate(lines, 1):
        if "::: {.column-margin}" in line:
            in_margin = True
        elif line.strip().startswith("::: {#fig-"):
            m_label = re.search(r"\{#(fig-[a-zA-Z0-9_\-]+)", line)
            current_div_label = m_label.group(1) if m_label else None
            m_cap = re.search(r'fig-cap=[\'"]([^\'"]+)[\'"]', line)
            current_div_cap = m_cap.group(1) if m_cap else None
            m_alt = re.search(r'fig-alt=[\'"]([^\'"]+)[\'"]', line)
            current_div_alt = m_alt.group(1) if m_alt else None
        elif line.strip() == ":::":
            in_margin = False
            current_div_label = None
            current_div_cap = None
            current_div_alt = None

        img_matches = re.finditer(r"!\[(.*?)\]\((.*?)\)(?:\{(.*?)\})?", line)
        for match in img_matches:
            cap_inline = match.group(1)
            img_rel = match.group(2)
            attrs = match.group(3) or ""

            filename = Path(img_rel).name
            label = current_div_label
            if not label:
                m_attr_label = re.search(r"#(fig-[a-zA-Z0-9_\-]+)", attrs)
                if m_attr_label:
                    label = m_attr_label.group(1)

            cap = current_div_cap or cap_inline
            alt = current_div_alt
            m_attr_alt = re.search(r'fig-alt=[\'"]([^\'"]+)[\'"]', attrs)
            if m_attr_alt:
                alt = m_attr_alt.group(1)

            is_margin_fig = in_margin or "margin_" in filename or "column-margin" in attrs

            placements[filename] = {
                "line": idx,
                "label": label,
                "caption": cap,
                "alt": alt,
                "is_margin": is_margin_fig,
            }

    return placements


def collect_chapter_data(ch_dir: Path) -> dict:
    ch_slug = ch_dir.name
    qmd_path = ch_dir / f"{ch_slug}.qmd"
    if not qmd_path.exists() and ch_slug == "backmatter":
        qmd_path = ch_dir / "appendix_spa.qmd"

    ch_title = ch_slug.replace("_", " ").title()
    if qmd_path.exists():
        text = qmd_path.read_text(encoding="utf-8")
        title_m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if title_m:
            raw_title = title_m.group(1).strip()
            ch_title = re.sub(r"\s*\{#[^}]+\}\s*", "", raw_title).strip()

    placements = parse_chapter_qmd(qmd_path)

    plates = []
    body_svgs = []
    margin_svgs = []
    spa_locators = []

    images_dir = ch_dir / "images"
    if images_dir.exists():
        # 1. Hardware Plates (JPG)
        jpg_dir = images_dir / "jpg"
        if jpg_dir.exists():
            for jpg in sorted(jpg_dir.glob("*.jpg")):
                fhash = get_file_hash(jpg)
                is_annotated = "_annotated" in jpg.name
                placement = placements.get(jpg.name)
                plates.append({
                    "name": jpg.name,
                    "rel_path": f"../{ch_slug}/images/jpg/{jpg.name}",
                    "abs_path": str(jpg),
                    "size_kb": round(jpg.stat().st_size / 1024, 1),
                    "hash": fhash,
                    "is_annotated": is_annotated,
                    "placement": placement,
                })

        # 2. SVGs
        svg_dir = images_dir / "svg"
        if svg_dir.exists():
            for svg in sorted(svg_dir.glob("*.svg")):
                fhash = get_file_hash(svg)
                meta = parse_svg_metadata(svg)
                png_target = TMP_PNG_DIR / f"{ch_slug}_{svg.stem}.png"
                rasterize_svg(svg, png_target)

                placement = placements.get(svg.name)
                item = {
                    "name": svg.name,
                    "rel_path": f"../{ch_slug}/images/svg/{svg.name}",
                    "png_path": str(png_target),
                    "abs_path": str(svg),
                    "size_kb": round(svg.stat().st_size / 1024, 1),
                    "hash": fhash,
                    "meta": meta,
                    "placement": placement,
                }

                if "spa_locator" in svg.name:
                    spa_locators.append(item)
                elif svg.name.startswith("margin_"):
                    sub = "visual"
                    if "ladder" in svg.name: sub = "ladder"
                    elif "knee" in svg.name: sub = "knee"
                    elif "sparkline" in svg.name: sub = "sparkline"
                    elif "strip" in svg.name: sub = "strip"
                    elif "envelope" in svg.name: sub = "envelope"
                    elif "taxonomy" in svg.name: sub = "taxonomy"
                    elif "chain" in svg.name: sub = "chain"
                    item["margin_subtype"] = sub
                    margin_svgs.append(item)
                else:
                    body_svgs.append(item)

    return {
        "slug": ch_slug,
        "title": ch_title,
        "qmd_path": str(qmd_path.relative_to(REPO_ROOT)) if qmd_path.exists() else None,
        "plates": plates,
        "body_svgs": body_svgs,
        "margin_svgs": margin_svgs,
        "spa_locators": spa_locators,
    }


def get_shared_css() -> str:
    return """
    :root {
      --bg-page: #F8FAFC;
      --bg-surface: #FFFFFF;
      --bg-surface-alt: #F1F5F9;
      --border-subtle: #E2E8F0;
      --border-strong: #CBD5E1;
      --text-primary: #0F172A;
      --text-secondary: #475569;
      --text-muted: #64748B;
      --primary-accent: #2563EB;
      --success: #16A34A;
      --warning: #D97706;
      --danger: #DC2626;
      --card-min-width: 260px;
      --thumb-height: 200px;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-page);
      color: var(--text-primary);
      font-family: var(--font-sans);
      line-height: 1.5;
      padding: 1.5rem 2rem 4rem;
    }

    header.top-nav {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 1.25rem 1.75rem;
      margin-bottom: 2rem;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .nav-header-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .title-area h1 {
      font-size: 1.45rem;
      font-weight: 700;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .title-area .subtitle {
      color: var(--text-muted);
      font-size: 0.9rem;
      margin-top: 0.2rem;
    }

    .controls-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
      padding-top: 0.75rem;
      border-top: 1px solid var(--border-subtle);
    }

    .btn-group {
      display: inline-flex;
      background: var(--bg-surface-alt);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 2px;
      gap: 2px;
    }

    .btn {
      padding: 0.35rem 0.75rem;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: 4px;
      border: none;
      background: transparent;
      color: var(--text-secondary);
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: background 0.15s, color 0.15s;
    }

    .btn:hover {
      background: var(--bg-surface);
      color: var(--text-primary);
    }

    .btn.active {
      background: var(--bg-surface);
      color: var(--primary-accent);
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .chapter-quick-nav {
      display: flex;
      gap: 0.35rem;
      flex-wrap: wrap;
      align-items: center;
    }

    .chip-nav {
      padding: 0.25rem 0.6rem;
      font-size: 0.78rem;
      font-family: var(--font-mono);
      font-weight: 600;
      border: 1px solid var(--border-subtle);
      background: var(--bg-surface);
      border-radius: 4px;
      color: var(--text-secondary);
      text-decoration: none;
      transition: all 0.15s ease;
    }

    .chip-nav:hover {
      border-color: var(--primary-accent);
      color: var(--primary-accent);
    }

    .chip-nav.active {
      background: var(--primary-accent);
      border-color: var(--primary-accent);
      color: #FFFFFF;
    }

    .stats-summary {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .stat-pill {
      background: var(--bg-surface-alt);
      border: 1px solid var(--border-subtle);
      border-radius: 4px;
      padding: 0.3rem 0.7rem;
      font-size: 0.8rem;
      color: var(--text-secondary);
    }

    .stat-pill strong {
      color: var(--text-primary);
    }

    .section-block {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      margin-bottom: 2rem;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    .section-header {
      background: var(--bg-surface-alt);
      border-bottom: 1px solid var(--border-subtle);
      padding: 0.75rem 1.25rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .section-title {
      font-size: 1rem;
      font-weight: 700;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .section-count {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(var(--card-min-width), 1fr));
      gap: 1.25rem;
      padding: 1.25rem;
      transition: all 0.2s ease;
    }

    /* Density modes */
    body.density-compact {
      --card-min-width: 190px;
      --thumb-height: 150px;
    }
    body.density-standard {
      --card-min-width: 260px;
      --thumb-height: 200px;
    }
    body.density-detail {
      --card-min-width: 380px;
      --thumb-height: 280px;
    }

    .card {
      background: #FFFFFF;
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
      transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .card.hidden-filter {
      display: none !important;
    }

    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
      border-color: var(--primary-accent);
    }

    .thumb-frame {
      width: 100%;
      height: var(--thumb-height);
      background: #FFFFFF;
      /* Subtle transparency checkerboard */
      background-image: linear-gradient(45deg, #F8FAFC 25%, transparent 25%),
                        linear-gradient(-45deg, #F8FAFC 25%, transparent 25%),
                        linear-gradient(45deg, transparent 75%, #F8FAFC 75%),
                        linear-gradient(-45deg, transparent 75%, #F8FAFC 75%);
      background-size: 16px 16px;
      background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 8px;
      overflow: hidden;
      cursor: pointer;
      position: relative;
      border-bottom: 1px solid var(--border-subtle);
    }

    .thumb-frame img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      filter: drop-shadow(0 1px 1px rgba(0,0,0,0.05));
    }

    .card-body {
      padding: 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      flex: 1;
      justify-content: space-between;
    }

    .card-filename {
      font-family: var(--font-mono);
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-primary);
      word-break: break-all;
      line-height: 1.3;
    }

    .card-meta-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.72rem;
      color: var(--text-muted);
      flex-wrap: wrap;
      gap: 0.3rem;
    }

    .badge {
      display: inline-block;
      padding: 0.15rem 0.45rem;
      border-radius: 3px;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .badge-placed { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
    .badge-unplaced { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .badge-plate { background: #EFF6FF; color: #1E40AF; border: 1px solid #DBEAFE; }
    .badge-plate-annotated { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
    .badge-body { background: #F1F5F9; color: #334155; border: 1px solid #E2E8F0; }
    .badge-margin { background: #F0FDF4; color: #166534; border: 1px solid #DCFCE7; }
    .badge-spa { background: #FAF5FF; color: #6B21A8; border: 1px solid #F3E8FF; }

    .badge-err { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
    .badge-ok { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }

    .card-caption-preview {
      font-size: 0.73rem;
      color: var(--text-secondary);
      font-style: italic;
      line-height: 1.25;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      margin-top: 0.2rem;
    }

    /* Modal viewer */
    .modal {
      display: none;
      position: fixed;
      z-index: 9999;
      left: 0; top: 0;
      width: 100%; height: 100%;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(4px);
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }

    .modal.active { display: flex; }

    .modal-content-wrapper {
      max-width: 90vw;
      max-height: 90vh;
      background: #FFFFFF;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
    }

    .modal-header {
      background: var(--bg-surface-alt);
      border-bottom: 1px solid var(--border-subtle);
      padding: 0.6rem 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--font-mono);
      font-size: 0.85rem;
      font-weight: 600;
      gap: 1rem;
    }

    .modal-img-container {
      padding: 1.5rem;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: auto;
      max-height: calc(90vh - 50px);
      background: #FFFFFF;
    }

    .modal-img-container img {
      max-width: 100%;
      max-height: 80vh;
      object-fit: contain;
    }

    .modal-close {
      cursor: pointer;
      font-weight: bold;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      border: 1px solid var(--border-subtle);
      background: #FFFFFF;
    }
    """


def render_card_html(item: dict, category_badge: str) -> str:
    name = item["name"]
    rel_path = item["rel_path"]
    size_kb = item["size_kb"]
    hash_str = item["hash"]
    placement = item.get("placement")

    meta = item.get("meta", {})
    vb_str = meta.get("viewBox", "")
    ar_str = meta.get("aspect_ratio", "")
    has_root_font = meta.get("has_root_font", True)
    has_outer_stroke = meta.get("has_outer_stroke", False)
    outer_stroke_val = meta.get("outer_stroke_val", "")

    is_placed = bool(placement)
    placed_attr = "true" if is_placed else "false"

    # Status badges
    if is_placed:
        p_badge = f'<span class="badge badge-placed" title="Line {placement["line"]}">Line {placement["line"]}</span>'
        if placement.get("label"):
            p_badge += f' <span class="badge badge-body" style="font-family:var(--font-mono)">{placement["label"]}</span>'
        caption_txt = placement.get("caption") or ""
    else:
        p_badge = '<span class="badge badge-unplaced">UNPLACED</span>'
        caption_txt = ""

    # Style audit indicators
    audit_flags = []
    if not has_root_font and item["name"].endswith(".svg"):
        audit_flags.append('<span class="badge badge-err" title="Missing root font-family">No Font</span>')
    if has_outer_stroke:
        audit_flags.append(f'<span class="badge badge-err" title="Outer canvas border stroke: {outer_stroke_val}">Border Box</span>')

    audit_str = " ".join(audit_flags)

    caption_html = f'<div class="card-caption-preview">{caption_txt}</div>' if caption_txt else ""
    dim_str = f"{vb_str} ({ar_str})" if vb_str else f"{size_kb} KB"

    return f"""
    <div class="card" data-placed="{placed_attr}">
      <div class="thumb-frame" onclick="openModal('{rel_path}', '{name}')">
        <img src="{rel_path}" alt="{name}" loading="lazy">
      </div>
      <div class="card-body">
        <div class="card-filename" title="{name}">{name}</div>
        <div class="card-meta-row">
          <span>{dim_str}</span>
          {category_badge}
        </div>
        <div class="card-meta-row">
          <div style="display:flex; gap:0.25rem; align-items:center;">{p_badge}</div>
          <div>{audit_str}</div>
        </div>
        {caption_html}
      </div>
    </div>
"""


def generate_chapter_sheet(ch_data: dict, all_slugs: List[str]) -> str:
    slug = ch_data["slug"]
    title = ch_data["title"]
    plates = ch_data["plates"]
    body_svgs = ch_data["body_svgs"]
    margin_svgs = ch_data["margin_svgs"]
    spa_locators = ch_data["spa_locators"]

    total_assets = len(plates) + len(body_svgs) + len(margin_svgs) + len(spa_locators)
    placed_assets = sum(1 for items in [plates, body_svgs, margin_svgs, spa_locators] for i in items if i.get("placement"))
    unplaced_assets = total_assets - placed_assets

    # Quick nav links
    curr_idx = all_slugs.index(slug)
    prev_slug = all_slugs[curr_idx - 1] if curr_idx > 0 else None
    next_slug = all_slugs[curr_idx + 1] if curr_idx < len(all_slugs) - 1 else None

    prev_btn = f'<a class="btn" href="{prev_slug}.html">← Previous ({prev_slug[:2]})</a>' if prev_slug else '<span class="btn" style="opacity:0.4;cursor:default;">← Previous</span>'
    next_btn = f'<a class="btn" href="{next_slug}.html">Next ({next_slug[:2]}) →</a>' if next_slug else '<span class="btn" style="opacity:0.4;cursor:default;">Next →</span>'

    chips_html = []
    for s in all_slugs:
        active_cls = " active" if s == slug else ""
        chips_html.append(f'<a class="chip-nav{active_cls}" href="{s}.html">{s[:2]}</a>')
    chips_html_str = "".join(chips_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chapter {slug[:2]}: {title} — Contact Sheet</title>
<style>
{get_shared_css()}
</style>
</head>
<body class="density-standard">

<header class="top-nav">
  <div class="nav-header-row">
    <div class="title-area">
      <h1><span>📖</span> Chapter {slug[:2]}: {title}</h1>
      <div class="subtitle">MLSysBook Volume IV Visual Proofing &amp; Contact Sheet</div>
    </div>
    <div style="display:flex; gap:0.5rem; align-items:center;">
      <a class="btn" href="index.html">📋 Master Index</a>
      {prev_btn}
      {next_btn}
    </div>
  </div>

  <div class="controls-row">
    <div class="chapter-quick-nav">
      <span style="font-size:0.78rem; font-weight:700; color:var(--text-muted); margin-right:0.25rem;">CHAPTERS:</span>
      {chips_html_str}
    </div>

    <div style="display:flex; gap:1rem; align-items:center; flex-wrap:wrap;">
      <div class="btn-group">
        <button class="btn active" id="filter-all" onclick="filterPlacement('all')">All ({total_assets})</button>
        <button class="btn" id="filter-placed" onclick="filterPlacement('placed')">Placed in Book ({placed_assets})</button>
        <button class="btn" id="filter-unplaced" onclick="filterPlacement('unplaced')">Unplaced / Drafts ({unplaced_assets})</button>
      </div>

      <div class="btn-group">
        <button class="btn" onclick="setDensity('compact')">Compact</button>
        <button class="btn active" onclick="setDensity('standard')">Standard</button>
        <button class="btn" onclick="setDensity('detail')">Detail</button>
      </div>
    </div>
  </div>
</header>
"""

    # Section 1: Hardware Plates
    if plates:
        html += f"""
<div class="section-block">
  <div class="section-header">
    <div class="section-title">📷 Authentic Hardware Photographic Plates</div>
    <div class="section-count">{len(plates)} assets</div>
  </div>
  <div class="grid">
"""
        for p in plates:
            badge = '<span class="badge badge-plate-annotated">Annotated</span>' if p["is_annotated"] else '<span class="badge badge-plate">Hardware</span>'
            html += render_card_html(p, badge)
        html += "</div></div>"

    # Section 2: Body Architecture SVGs
    if body_svgs:
        html += f"""
<div class="section-block">
  <div class="section-header">
    <div class="section-title">📐 Body Architecture &amp; System Schematics</div>
    <div class="section-count">{len(body_svgs)} assets</div>
  </div>
  <div class="grid">
"""
        for b in body_svgs:
            html += render_card_html(b, '<span class="badge badge-body">Body SVG</span>')
        html += "</div></div>"

    # Section 3: Margin Visuals
    if margin_svgs:
        html += f"""
<div class="section-block">
  <div class="section-header">
    <div class="section-title">📌 Unnumbered Margin Visual Figures</div>
    <div class="section-count">{len(margin_svgs)} assets</div>
  </div>
  <div class="grid">
"""
        for m in margin_svgs:
            sub = m.get("margin_subtype", "visual").upper()
            html += render_card_html(m, f'<span class="badge badge-margin">{sub}</span>')
        html += "</div></div>"

    # Section 4: S·P·A Locators
    if spa_locators:
        html += f"""
<div class="section-block">
  <div class="section-header">
    <div class="section-title">🧭 Sense·Plan·Act Locators</div>
    <div class="section-count">{len(spa_locators)} assets</div>
  </div>
  <div class="grid">
"""
        for s in spa_locators:
            html += render_card_html(s, '<span class="badge badge-spa">SPA LOCATOR</span>')
        html += "</div></div>"

    # Modal viewer & JS
    html += """
<div id="modal" class="modal" onclick="closeModal(event)">
  <div class="modal-content-wrapper" onclick="event.stopPropagation()">
    <div class="modal-header">
      <span id="modal-title">Figure Preview</span>
      <div style="display:flex; gap:0.5rem; align-items:center;">
        <button class="btn" style="background:#FFFFFF; border:1px solid var(--border-subtle);" onclick="openRawImage()">↗ Open Raw</button>
        <button class="modal-close" onclick="closeModal()">✕ Close</button>
      </div>
    </div>
    <div class="modal-img-container">
      <img id="modal-img" src="" alt="Enlarged view">
    </div>
  </div>
</div>

<script>
  let currentModalSrc = '';

  function setDensity(mode) {
    document.body.className = 'density-' + mode;
    document.querySelectorAll('.btn-group:last-child button').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
  }

  function filterPlacement(filterType) {
    document.querySelectorAll('#filter-all, #filter-placed, #filter-unplaced').forEach(b => b.classList.remove('active'));
    document.getElementById('filter-' + filterType).classList.add('active');

    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
      const isPlaced = card.getAttribute('data-placed') === 'true';
      if (filterType === 'all') {
        card.classList.remove('hidden-filter');
      } else if (filterType === 'placed') {
        if (isPlaced) card.classList.remove('hidden-filter');
        else card.classList.add('hidden-filter');
      } else if (filterType === 'unplaced') {
        if (!isPlaced) card.classList.remove('hidden-filter');
        else card.classList.add('hidden-filter');
      }
    });
  }

  function openModal(src, title) {
    currentModalSrc = src;
    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modal-img');
    const modalTitle = document.getElementById('modal-title');
    modalImg.src = src;
    modalTitle.textContent = title;
    modal.classList.add('active');
  }

  function openRawImage() {
    if (currentModalSrc) {
      window.open(currentModalSrc, '_blank');
    }
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


def generate_index_sheet(all_data: List[dict], is_root: bool = False) -> str:
    total_plates = sum(len(c["plates"]) for c in all_data)
    total_body = sum(len(c["body_svgs"]) for c in all_data)
    total_margin = sum(len(c["margin_svgs"]) for c in all_data)
    total_spa = sum(len(c["spa_locators"]) for c in all_data)
    grand_total = total_plates + total_body + total_margin + total_spa

    all_items = [item for c in all_data for cat in [c["plates"], c["body_svgs"], c["margin_svgs"], c["spa_locators"]] for item in cat]
    total_placed = sum(1 for i in all_items if i.get("placement"))
    placement_pct = round((total_placed / grand_total) * 100, 1) if grand_total else 100

    sheet_prefix = "contact_sheets/" if is_root else ""
    img_prefix_adjust = (lambda rel: rel[3:] if rel.startswith("../") else rel) if is_root else (lambda rel: rel)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MLSysBook Volume IV — Visual Contact Sheet Master Index</title>
<style>
{get_shared_css()}
.hub-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1.5rem;
}}
.hub-card {{
  background: #FFFFFF;
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  transition: all 0.2s ease;
  display: flex;
  flex-direction: column;
}}
.hub-card:hover {{
  transform: translateY(-2px);
  border-color: var(--primary-accent);
  box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}}
.hub-card-header {{
  background: var(--bg-surface-alt);
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--border-subtle);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.hub-card-header h2 {{
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text-primary);
}}
.hub-card-body {{
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  flex: 1;
  justify-content: space-between;
}}
.preview-row {{
  display: flex;
  gap: 0.5rem;
  height: 64px;
  background: #F8FAFC;
  padding: 4px;
  border-radius: 4px;
  border: 1px solid var(--border-subtle);
}}
.preview-row img {{
  flex: 1;
  max-width: 32%;
  height: 100%;
  object-fit: contain;
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 2px;
}}
.hub-stats-row {{
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  font-size: 0.75rem;
  color: var(--text-secondary);
}}
.hub-link-btn {{
  display: block;
  text-align: center;
  padding: 0.5rem;
  background: var(--bg-surface-alt);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--primary-accent);
  text-decoration: none;
  transition: all 0.15s ease;
}}
.hub-link-btn:hover {{
  background: var(--primary-accent);
  color: #FFFFFF;
  border-color: var(--primary-accent);
}}
</style>
</head>
<body class="density-standard">

<header class="top-nav">
  <div class="nav-header-row">
    <div class="title-area">
      <h1><span>📚</span> Volume IV: Physical AI Systems — Contact Sheet Hub</h1>
      <div class="subtitle">Individual chapter visual proofs, component audits, and figure style consistency review</div>
    </div>
  </div>

  <div class="controls-row">
    <div class="stats-summary">
      <div class="stat-pill">Total Figures: <strong>{grand_total}</strong></div>
      <div class="stat-pill">Placement Rate: <strong>{total_placed}/{grand_total} ({placement_pct}%)</strong></div>
      <div class="stat-pill">Hardware Plates: <strong>{total_plates}</strong></div>
      <div class="stat-pill">Body Schematics: <strong>{total_body}</strong></div>
      <div class="stat-pill">Margin Figures: <strong>{total_margin}</strong></div>
      <div class="stat-pill">S·P·A Locators: <strong>{total_spa}</strong></div>
    </div>
  </div>
</header>

<main class="hub-grid">
"""

    for ch in all_data:
        slug = ch["slug"]
        title = ch["title"]
        plates = ch["plates"]
        body_svgs = ch["body_svgs"]
        margin_svgs = ch["margin_svgs"]
        spa_locators = ch["spa_locators"]
        ch_total = len(plates) + len(body_svgs) + len(margin_svgs) + len(spa_locators)

        # Get up to 3 preview thumbnails
        preview_imgs = []
        for p in plates[:1]:
            preview_imgs.append(img_prefix_adjust(p["rel_path"]))
        for b in body_svgs[:2]:
            preview_imgs.append(img_prefix_adjust(b["rel_path"]))
        for m in margin_svgs[:1]:
            if len(preview_imgs) < 3:
                preview_imgs.append(img_prefix_adjust(m["rel_path"]))

        previews_html = "".join([f'<img src="{src}" loading="lazy">' for src in preview_imgs[:3]])

        html += f"""
  <div class="hub-card">
    <div class="hub-card-header">
      <h2>Chapter {slug[:2]}: {title}</h2>
      <span class="badge badge-body">{ch_total} Figures</span>
    </div>
    <div class="hub-card-body">
      <div class="preview-row">
        {previews_html}
      </div>
      <div class="hub-stats-row">
        <span>📷 {len(plates)} Plates</span> •
        <span>📐 {len(body_svgs)} Body</span> •
        <span>📌 {len(margin_svgs)} Margin</span> •
        <span>🧭 {len(spa_locators)} SPA</span>
      </div>
      <a class="hub-link-btn" href="{sheet_prefix}{slug}.html">Open Chapter {slug[:2]} Proof Sheet →</a>
    </div>
  </div>
"""

    html += """
</main>
</body>
</html>
"""
    return html


def main():
    print("Collecting and building individual contact sheets for Volume IV...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_PNG_DIR.mkdir(parents=True, exist_ok=True)

    chapters = sorted([d for d in VOL4_DIR.iterdir() if d.is_dir() and re.match(r"^(0[1-9]|1[0-7])_", d.name)])
    all_slugs = [d.name for d in chapters]

    all_data = []
    for ch_dir in chapters:
        print(f"  - Parsing {ch_dir.name}...")
        data = collect_chapter_data(ch_dir)
        all_data.append(data)

        # Generate individual chapter contact sheet
        html = generate_chapter_sheet(data, all_slugs)
        out_file = OUTPUT_DIR / f"{ch_dir.name}.html"
        out_file.write_text(html, encoding="utf-8")
        print(f"    ✓ Generated {out_file.name}")

    # Generate master index inside contact_sheets/
    index_html = generate_index_sheet(all_data, is_root=False)
    index_file = OUTPUT_DIR / "index.html"
    index_file.write_text(index_html, encoding="utf-8")
    print(f"✅ Generated Master Hub at {index_file}")

    # Generate root contact sheet at books/vol4/vol4_visual_contact_sheet.html
    root_index_html = generate_index_sheet(all_data, is_root=True)
    root_sheet = VOL4_DIR / "vol4_visual_contact_sheet.html"
    root_sheet.write_text(root_index_html, encoding="utf-8")
    print(f"✅ Updated {root_sheet} with properly routed links")


if __name__ == "__main__":
    main()
