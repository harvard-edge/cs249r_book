#!/usr/bin/env python3
"""
audit_vol4_figures.py
=====================
Comprehensive audit of Volume IV figures against Figure 3.7 standard and .claude rules.
"""

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VOL4_DIR = REPO_ROOT / "books" / "vol4"

def audit_all():
    svgs = sorted(list(VOL4_DIR.glob("**/images/svg/*.svg")))
    print(f"Total SVGs found: {len(svgs)}")

    missing_font = []
    outer_stroke = []
    embedded_titles = []
    bubbly_boxes = []

    for s in svgs:
        c = s.read_text(encoding="utf-8")
        rel = str(s.relative_to(VOL4_DIR))

        # 1. Root font-family
        root_m = re.search(r"<svg\b[^>]*>", c)
        if root_m:
            root_str = root_m.group(0)
            if "font-family=" not in root_str and "<text" in c:
                missing_font.append(rel)

        # 2. Outer canvas border stroke
        # Scan for full canvas rects with a visible border stroke
        for r_m in re.finditer(r"<rect\b[^>]*>", c):
            tag = r_m.group(0)
            is_full = ('width="100%"' in tag or "width='100%'" in tag or
                       'width="680"' in tag or "width='680'" in tag or
                       'width="540"' in tag or "width='540'" in tag or
                       'width="800"' in tag or "width='800'" in tag or
                       'width="640"' in tag or "width='640'" in tag)
            if is_full:
                strk_m = re.search(r'stroke=[\'"]([^\'"]+)[\'"]', tag)
                if strk_m:
                    val = strk_m.group(1).strip().lower()
                    if val not in ("none", "transparent", ""):
                        outer_stroke.append((rel, val))
                        break

        # 3. Embedded title banner candidates (text at top y <= 50 with size >= 14 or text starting with 'FIGURE')
        for t_m in re.finditer(r"<text\b([^>]*)>([^<]+)</text>", c):
            attrs = t_m.group(1)
            txt = t_m.group(2).strip()
            y_m = re.search(r'y=[\'"]([0-9\.]+)[\'"]', attrs)
            sz_m = re.search(r'font-size=[\'"]([0-9\.]+)[\'"]', attrs)
            if y_m and sz_m:
                y = float(y_m.group(1))
                sz = float(sz_m.group(1))
                if (y <= 50 and sz >= 14) or txt.upper().startswith("FIGURE ") or txt.upper().startswith("FIG. "):
                    embedded_titles.append((rel, y, sz, txt))

        # 4. Bubbly boxes (rx > 6 on large rectangular elements, not pills)
        for r_m in re.finditer(r"<rect\b([^>]*)>", c):
            tag = r_m.group(1)
            rx_m = re.search(r'rx=[\'"]([0-9\.]+)[\'"]', tag)
            w_m = re.search(r'width=[\'"]([0-9\.]+)[\'"]', tag)
            h_m = re.search(r'height=[\'"]([0-9\.]+)[\'"]', tag)
            if rx_m and w_m and h_m:
                rx = float(rx_m.group(1))
                w = float(w_m.group(1))
                h = float(h_m.group(1))
                if rx > 6 and (w > 120 or h > 60):
                    bubbly_boxes.append((rel, rx, w, h))

    print(f"\n1. SVGs with Missing Root font-family: {len(missing_font)}")
    for f in missing_font:
        print(f"   - {f}")

    print(f"\n2. SVGs with Outer Canvas Border Stroke: {len(outer_stroke)}")
    for f, val in outer_stroke:
        print(f"   - {f} (stroke={val})")

    print(f"\n3. SVGs with Potential Embedded Title Banners: {len(embedded_titles)}")
    for f, y, sz, txt in embedded_titles:
        print(f"   - {f} (y={y}, sz={sz}): '{txt}'")

    print(f"\n4. SVGs with Bubbly Containers (rx > 6): {len(bubbly_boxes)}")
    for f, rx, w, h in bubbly_boxes:
        print(f"   - {f} (rx={rx}, w={w}, h={h})")

if __name__ == "__main__":
    audit_all()
