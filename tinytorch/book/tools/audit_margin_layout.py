#!/usr/bin/env python3
"""
audit_margin_layout.py
======================
Scans TinyTorch-Book.pdf for:
1. Overlaps between distinct margin elements (e.g. footnote vs margin figure, jump link vs margin figure, footnote vs jump link).
2. Margin elements extending past the bottom margin boundary (y1 > 725 pt).
Outputs a clean, prioritized list grouped by chapter.
"""

import sys
from pathlib import Path
import re
from collections import defaultdict
import fitz

def classify_margin_block(text):
    t = text.strip()
    if re.match(r'^\d+\s+', t) or re.match(r'^\d+\n', t):
        return "footnote"
    if "Jump back:" in t or "Jump ahead:" in t:
        return "jump-link"
    if t.startswith("Core Invariant:") or t.startswith("Invariant"):
        return "invariant-callout"
    return "card-or-caption"

def scan_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    
    # Map page numbers to chapters based on running headers and titles
    page_chapters = {}
    current_chapter = "Frontmatter"
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        m = re.search(r'Chapter (\d+)[·\s:]+([^\n]+)', text)
        if m:
            current_chapter = f"Ch {int(m.group(1)):02d}: {m.group(2).strip()}"
        else:
            m2 = re.search(r'Synthesis (\d+)[·\s:]+([^\n]+)', text)
            if m2:
                current_chapter = f"Synthesis {int(m2.group(1)):02d}: {m2.group(2).strip()}"
            elif "Conclusion:" in text or "Conclusion ·" in text:
                current_chapter = "Conclusion"
            elif "Welcome" in text and page_num < 40:
                current_chapter = "Ch 00: Welcome"
                
        page_chapters[page_num + 1] = current_chapter

    findings = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        p_num = page_num + 1
        is_even = (p_num % 2 == 0)
        ch = page_chapters[p_num]

        blocks = page.get_text("blocks")
        margin_items = []

        for b in blocks:
            bbox = b[:4]
            text = b[4].strip()
            if not text:
                continue
            # Outer margin check (Letter paper: width=612, height=792)
            if (is_even and bbox[0] < 155 and bbox[2] < 165) or \
               (not is_even and bbox[0] > 440):
                margin_items.append({
                    "bbox": bbox,
                    "text": text,
                    "category": classify_margin_block(text),
                    "page": p_num,
                    "chapter": ch
                })

        # Check for overlaps between DISTINCT element categories
        # or between elements with large vertical overlap (> 8 pt)
        for i in range(len(margin_items)):
            for j in range(i + 1, len(margin_items)):
                m1 = margin_items[i]
                m2 = margin_items[j]
                b1, b2 = m1["bbox"], m2["bbox"]
                y_ov = min(b1[3], b2[3]) - max(b1[1], b2[1])
                x_ov = min(b1[2], b2[2]) - max(b1[0], b2[0])
                
                # Check if they are distinct elements
                # If one is footnote or jump-link, or if both are cards separated by > 8pt
                if y_ov > 5 and x_ov > 10:
                    is_inter_element = (m1["category"] != m2["category"]) or \
                                       (m1["category"] in ("footnote", "jump-link")) or \
                                       (m2["category"] in ("footnote", "jump-link"))
                    if is_inter_element:
                        findings.append({
                            "kind": "collision",
                            "page": p_num,
                            "chapter": ch,
                            "depth": y_ov,
                            "item1": f"[{m1['category']}] {m1['text'].replace(chr(10), ' ')[:50]}",
                            "item2": f"[{m2['category']}] {m2['text'].replace(chr(10), ' ')[:50]}",
                            "bbox1": b1,
                            "bbox2": b2,
                        })

        # Check for bottom overflow (y1 > 725 pt)
        for item in margin_items:
            y1 = item["bbox"][3]
            if y1 > 728.0:
                findings.append({
                    "kind": "bottom-overflow",
                    "page": p_num,
                    "chapter": ch,
                    "depth": y1 - 720.0,
                    "item1": f"[{item['category']}] {item['text'].replace(chr(10), ' ')[:70]}",
                    "item2": "",
                    "bbox1": item["bbox"],
                    "bbox2": None,
                })

    return findings

if __name__ == "__main__":
    pdf_path = Path("tinytorch/book/_build/TinyTorch-Book.pdf")
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    
    findings = scan_pdf(pdf_path)
    
    by_chapter = defaultdict(list)
    for f in findings:
        by_chapter[f["chapter"]].append(f)
        
    print(f"Total margin layout findings: {len(findings)}")
    print("=" * 80)
    for ch, items in sorted(by_chapter.items()):
        print(f"\n### {ch}")
        # Deduplicate
        seen = set()
        for it in items:
            key = (it["kind"], it["page"], it["item1"][:25], it["item2"][:25])
            if key in seen:
                continue
            seen.add(key)
            if it["kind"] == "collision":
                print(f"  [COLLISION] Page {it['page']} ({it['depth']:.1f}pt overlap):")
                print(f"    Item A: {it['item1']}")
                print(f"    Item B: {it['item2']}")
            else:
                print(f"  [OVERFLOW-BOTTOM] Page {it['page']} ({it['depth']:.1f}pt past baseline, y1={it['bbox1'][3]:.1f}):")
                print(f"    Item:   {it['item1']}")
