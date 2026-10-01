#!/usr/bin/env python3
"""
generate_pdf_contact_sheets.py
==============================
Generates multi-page visual contact sheets and an interactive HTML audit dashboard
for TinyTorch-Book.pdf.

Features:
1. Automatically audits pages for margin bottom-overflow (y1 > 725 pt) and inter-element collisions.
2. Annotates defective pages with red bounding boxes over offending elements and red baseline markers.
3. Composites pages into multi-page contact sheets (PNG grids, e.g. 3x2 or 3x3) for rapid visual scanning via view_file.
4. Generates a self-contained interactive HTML contact sheet viewer served directly on localhost.

Usage:
  python3 tinytorch/book/tools/generate_pdf_contact_sheets.py --flagged-only
  python3 tinytorch/book/tools/generate_pdf_contact_sheets.py --all --cols 3 --rows 3
  python3 tinytorch/book/tools/generate_pdf_contact_sheets.py --pages 43,80,82,96,99
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

# Import audit logic if available
try:
    from tinytorch.book.tools.audit_margin_layout import scan_pdf
except ImportError:
    # Fallback to local import if run from repo root or book/tools
    sys.path.append(str(Path(__file__).resolve().parent))
    from audit_margin_layout import scan_pdf


def annotate_page_image(doc, page_num, findings, dpi=120):
    """
    Renders a PDF page to a PIL Image and overlays visual annotations for defects.
    page_num is 1-indexed.
    """
    page = doc[page_num - 1]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)

    # Use default font
    font = ImageFont.load_default()

    page_findings = [f for f in findings if f["page"] == page_num]

    if page_findings:
        # Draw bottom margin boundary (720 pt)
        y_boundary = 720.0 * zoom
        draw.line([(0, y_boundary), (pix.width, y_boundary)], fill=(255, 0, 0), width=2)
        draw.text((10, y_boundary - 15), "BOTTOM MARGIN BASELINE (720 pt)", fill=(255, 0, 0), font=font)

        for f in page_findings:
            b1 = f.get("bbox1")
            if b1:
                # b1 is (x0, y0, x1, y1) in 72-dpi points
                rx0, ry0, rx1, ry1 = [coord * zoom for coord in b1]
                # Draw thick red bounding box
                draw.rectangle([rx0, ry0, rx1, ry1], outline=(255, 0, 0), width=4)
                # Label
                kind = f["kind"].upper()
                depth = f.get("depth", 0)
                label = f"{kind}: {depth:.1f}pt"
                draw.rectangle([rx0, max(0, ry0 - 18), rx0 + 130, ry0], fill=(255, 0, 0))
                draw.text((rx0 + 4, max(2, ry0 - 16)), label, fill=(255, 255, 255), font=font)

            b2 = f.get("bbox2")
            if b2:
                rx0, ry0, rx1, ry1 = [coord * zoom for coord in b2]
                draw.rectangle([rx0, ry0, rx1, ry1], outline=(255, 80, 0), width=3)

    return img, len(page_findings) > 0


def create_contact_sheet(page_items, cols=3, rows=2, dpi=120, thumb_w=700):
    """
    Creates a single composite image containing a grid of page thumbnails with badges.
    page_items is a list of tuples: (page_num, chapter_name, img, is_flagged, findings_summary)
    """
    thumb_h = int(thumb_w * (792.0 / 612.0))
    header_h = 45
    cell_w = thumb_w + 30
    cell_h = thumb_h + header_h + 30

    sheet_w = cols * cell_w + 40
    sheet_h = rows * cell_h + 80

    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 246, 248))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    # Title header
    draw.rectangle([0, 0, sheet_w, 60], fill=(30, 41, 59))
    draw.text((30, 20), "TinyTorch Book Visual Contact Sheet", fill=(255, 255, 255), font=font)

    for idx, (p_num, ch_name, img, is_flagged, summary) in enumerate(page_items):
        if idx >= cols * rows:
            break
        r = idx // cols
        c = idx % cols

        x0 = 30 + c * cell_w
        y0 = 80 + r * cell_h

        # Cell card background
        card_bg = (255, 240, 240) if is_flagged else (255, 255, 255)
        border_col = (220, 38, 38) if is_flagged else (203, 213, 225)
        border_w = 3 if is_flagged else 1
        draw.rectangle([x0, y0, x0 + cell_w - 10, y0 + cell_h - 10], fill=card_bg, outline=border_col, width=border_w)

        # Header banner inside cell
        banner_fill = (220, 38, 38) if is_flagged else (71, 85, 105)
        draw.rectangle([x0, y0, x0 + cell_w - 10, y0 + 35], fill=banner_fill)
        status_txt = f"⚠️ DEFECT DETECTED" if is_flagged else "✓ CLEAN"
        header_title = f"Page {p_num} | {status_txt}"
        draw.text((x0 + 10, y0 + 10), header_title, fill=(255, 255, 255), font=font)

        if is_flagged and summary:
            draw.text((x0 + 10, y0 + 38), summary[:65], fill=(185, 28, 28), font=font)

        # Resize page image to thumbnail
        resized_img = img.resize((thumb_w - 20, thumb_h - 20), Image.Resampling.LANCZOS)
        img_x = x0 + 10
        img_y = y0 + header_h + 10
        sheet.paste(resized_img, (img_x, img_y))

        # Thin border around thumbnail
        draw.rectangle([img_x, img_y, img_x + thumb_w - 20, img_y + thumb_h - 20], outline=(148, 163, 184), width=1)

    return sheet


def generate_html_contact_sheet(doc, findings, thumbs_dir, output_html, rel_thumbs_dir):
    """
    Generates an interactive, standalone HTML contact sheet.
    """
    by_page = defaultdict(list)
    for f in findings:
        by_page[f["page"]].append(f)

    cards_html = []
    total_pages = len(doc)

    for p_num in range(1, total_pages + 1):
        is_flagged = p_num in by_page
        p_findings = by_page[p_num]
        thumb_name = f"page_{p_num:03d}.png"
        thumb_rel_path = f"{rel_thumbs_dir}/{thumb_name}"

        finding_badges = ""
        if is_flagged:
            for f in p_findings:
                k = f["kind"]
                depth = f.get("depth", 0)
                item = f.get("item1", "")[:50]
                finding_badges += f"""<div class="badge badge-error"><strong>{k.upper()}</strong>: {depth:.1f}pt<br><span>{item}</span></div>"""

        card = f"""
        <div class="page-card {'flagged' if is_flagged else 'clean'}" data-page="{p_num}" data-flagged="{'true' if is_flagged else 'false'}">
            <div class="card-header {'header-error' if is_flagged else 'header-clean'}">
                <span class="page-badge">Page {p_num}</span>
                <span class="status-badge">{'⚠️ DEFECT' if is_flagged else '✓ OK'}</span>
            </div>
            <div class="card-body">
                <a href="{thumb_rel_path}" target="_blank">
                    <img src="{thumb_rel_path}" alt="Page {p_num}" loading="lazy" class="page-thumb">
                </a>
                <div class="findings-box">
                    {finding_badges if is_flagged else '<span class="text-muted">No layout defects detected</span>'}
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TinyTorch Book Visual Layout Contact Sheet</title>
<style>
  :root {{
    --bg: #0f172a;
    --card-bg: #1e293b;
    --card-border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --danger: #ef4444;
    --danger-bg: #450a0a;
    --success: #10b981;
    --success-bg: #064e3b;
    --brand: #f97316;
  }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
  }}
  header {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: #090d16;
    border-bottom: 1px solid var(--card-border);
    padding: 16px 24px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }}
  .title-group h1 {{
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
  }}
  .title-group p {{
    margin: 4px 0 0 0;
    font-size: 0.85rem;
    color: var(--text-muted);
  }}
  .controls {{
    display: flex;
    gap: 12px;
    align-items: center;
  }}
  button.filter-btn {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    color: var(--text);
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  button.filter-btn.active {{
    background: var(--brand);
    border-color: var(--brand);
    color: #fff;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
    padding: 24px;
  }}
  .page-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
  }}
  .page-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
  }}
  .page-card.flagged {{
    border-color: var(--danger);
    box-shadow: 0 0 0 1px var(--danger);
  }}
  .card-header {{
    padding: 8px 12px;
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 700;
  }}
  .header-clean {{
    background: #1e293b;
    border-bottom: 1px solid var(--card-border);
  }}
  .header-error {{
    background: var(--danger-bg);
    border-bottom: 1px solid var(--danger);
    color: #fca5a5;
  }}
  .card-body {{
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  .page-thumb {{
    width: 100%;
    height: auto;
    border-radius: 4px;
    border: 1px solid #475569;
    display: block;
    background: #fff;
  }}
  .findings-box {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 0.75rem;
  }}
  .badge {{
    padding: 6px 8px;
    border-radius: 4px;
    line-height: 1.3;
  }}
  .badge-error {{
    background: #450a0a;
    border: 1px solid #7f1d1d;
    color: #fca5a5;
  }}
  .text-muted {{
    color: var(--text-muted);
  }}
</style>
</head>
<body>
<header>
  <div class="title-group">
    <h1>TinyTorch Book: Visual Layout Contact Sheet</h1>
    <p>Auditing 572 Pages | {len(by_page)} Pages with Detected Defects</p>
  </div>
  <div class="controls">
    <button class="filter-btn active" id="btn-flagged" onclick="filterPages('flagged')">Defects Only ({len(by_page)})</button>
    <button class="filter-btn" id="btn-all" onclick="filterPages('all')">All Pages ({total_pages})</button>
  </div>
</header>
<main class="grid" id="page-grid">
  {''.join(cards_html)}
</main>
<script>
  function filterPages(type) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(type === 'flagged' ? 'btn-flagged' : 'btn-all').classList.add('active');
    
    const cards = document.querySelectorAll('.page-card');
    cards.forEach(c => {{
      if (type === 'flagged') {{
        c.style.display = c.dataset.flagged === 'true' ? 'flex' : 'none';
      }} else {{
        c.style.display = 'flex';
      }}
    }});
  }}
  // Default to flagged only on load
  filterPages('flagged');
</script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_content, encoding="utf-8")
    print(f"Generated interactive HTML contact sheet: {output_html}")


def main():
    parser = argparse.ArgumentParser(description="Generate PDF visual contact sheets.")
    parser.add_argument("--pdf", default="tinytorch/book/_build/TinyTorch-Book.pdf", help="Path to book PDF")
    parser.add_argument("--out-dir", default="/tmp/tinytorch_contact_sheets", help="Output directory for PNG contact sheets")
    parser.add_argument("--html-out", default="tinytorch/guide/_build/contact_sheet.html", help="Output HTML contact sheet path")
    parser.add_argument("--flagged-only", action="store_true", default=True, help="Render only pages with layout defects")
    parser.add_argument("--all", action="store_true", help="Render all pages")
    parser.add_argument("--pages", help="Comma-separated list of page numbers to render")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in PNG sheet")
    parser.add_argument("--rows", type=int, default=2, help="Number of rows in PNG sheet")
    parser.add_argument("--dpi", type=int, default=120, help="DPI for page rendering")

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Auditing margin layout for {pdf_path}...")
    findings = scan_pdf(pdf_path)

    by_page = defaultdict(list)
    for f in findings:
        by_page[f["page"]].append(f)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if args.pages:
        selected_pages = [int(p.strip()) for p in args.pages.split(",") if p.strip()]
    elif args.all:
        selected_pages = list(range(1, total_pages + 1))
    else:
        # Default: flagged pages only
        selected_pages = sorted(by_page.keys())

    print(f"Rendering {len(selected_pages)} pages (total PDF pages: {total_pages}, flagged: {len(by_page)})...")

    rendered_items = []
    for p_num in selected_pages:
        img, is_flagged = annotate_page_image(doc, p_num, findings, dpi=args.dpi)
        
        # Save individual thumbnail
        thumb_path = thumbs_dir / f"page_{p_num:03d}.png"
        img.save(thumb_path, "PNG")

        # Summary text
        summary = ""
        if p_num in by_page:
            f0 = by_page[p_num][0]
            summary = f"[{f0['kind'].upper()}] depth={f0['depth']:.1f}pt | {f0.get('item1', '')[:40]}"

        rendered_items.append((p_num, f"Page {p_num}", img, is_flagged, summary))

    # Generate composite PNG contact sheets
    sheet_size = args.cols * args.rows
    num_sheets = math.ceil(len(rendered_items) / sheet_size)
    print(f"Generating {num_sheets} composite contact sheets ({args.cols}x{args.rows} grid)...")

    generated_sheets = []
    for sheet_idx in range(num_sheets):
        batch = rendered_items[sheet_idx * sheet_size : (sheet_idx + 1) * sheet_size]
        sheet_img = create_contact_sheet(batch, cols=args.cols, rows=args.rows, dpi=args.dpi)
        sheet_filename = out_dir / f"contact_sheet_{sheet_idx + 1:02d}.png"
        sheet_img.save(sheet_filename, "PNG")
        generated_sheets.append(sheet_filename)
        print(f"  Saved contact sheet: {sheet_filename}")

    # Generate HTML viewer
    html_out = Path(args.html_out)
    # Relative path from HTML to thumbs
    rel_thumbs = os.path.relpath(thumbs_dir, html_out.parent)
    generate_html_contact_sheet(doc, findings, thumbs_dir, html_out, rel_thumbs)

    print("\nVisual Contact Sheet Generation Complete!")
    print(f"• PNG Contact Sheets: {len(generated_sheets)} files in {out_dir}")
    print(f"• Interactive HTML:   {html_out}")
    print(f"• Localhost URL:      http://localhost:8000/{html_out.name}")


if __name__ == "__main__":
    main()
