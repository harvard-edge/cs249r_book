#!/usr/bin/env python3
"""Margin-geometry oracle for the MLSysBook PDF.

Ground-truth detector for the layout problems the binder scanner is blind to:
element-on-element overlap in the outer margin (sidenote-on-sidenote,
figure-on-sidenote) and margin elements that overflow past the footer or up
into the running header.

Unlike the existing `binder layout margins` candidate-generator, this works
purely from rendered PDF geometry via PyMuPDF: it extracts the real bounding
box of every text block, image, and vector drawing, classifies each into the
margin column, clusters contiguous fragments into logical elements, and reports
true bbox intersections. No anchor heuristics, no blindness to overlap.

Page geometry is taken from book/quarto/tex/header-includes.tex (MIT Press
8x10 trim). Margin side is detected empirically (whichever band holds content)
so frontmatter parity shifts do not matter.

Usage:
  python3 margin_geometry.py <pdf> [--first N] [--last N] [--json out.json]
"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import dataclass, field, asdict

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF not available: pip install pymupdf")

IN = 72.0  # pt per inch

# --- page geometry (header-includes.tex \geometry, 8x10 MIT Press trim) -------
PAPER_W, PAPER_H = 8 * IN, 10 * IN
TOP, BOTTOM = 0.5 * IN, 0.625 * IN
INNER, OUTER = 0.875 * IN, 1.75 * IN
HEADHEIGHT, HEADSEP = 14.0, 18.0
MARGINSEP, MARGINW = 0.25 * IN, 1.25 * IN

TEXT_W = PAPER_W - INNER - OUTER            # 5.375in = 387pt
TEXT_TOP = TOP + HEADHEIGHT + HEADSEP       # ~68pt: top of body text
TEXT_BOTTOM = PAPER_H - BOTTOM              # 675pt: bottom of body text

# margin-column x-bands (a block is "in the margin" if its x-center lands here)
RIGHT_TEXT_EDGE = INNER + TEXT_W            # recto: text ends at 450pt
RIGHT_BAND = (RIGHT_TEXT_EDGE + 2, PAPER_W) # right margin column
LEFT_TEXT_EDGE = OUTER                      # verso: text starts at 126pt
LEFT_BAND = (0, LEFT_TEXT_EDGE - 2)         # left margin column

# tolerances
OVERLAP_TOL = 1.5   # pt of vertical intersection before we call it a collision
MERGE_GAP = 4.0     # blocks closer than this vertically = same logical element
EDGE_TOL = 2.0      # slack before footer overflow counts
TOP_TOL = 6.0       # an element must clear the body top by >this to count as
                    # poking into the header (filters running-head/first-line noise)


@dataclass
class Element:
    x0: float; y0: float; x1: float; y1: float
    side: str
    kind: str               # 'text' | 'image' | 'drawing' | 'mixed'
    def merge(self, o: "Element"):
        self.x0, self.y0 = min(self.x0, o.x0), min(self.y0, o.y0)
        self.x1, self.y1 = max(self.x1, o.x1), max(self.y1, o.y1)
        if self.kind != o.kind:
            self.kind = "mixed"


@dataclass
class Finding:
    page: int
    issue: str              # 'overlap' | 'overflow-bottom' | 'overflow-top'
    side: str
    detail: str
    bbox: tuple


def _band_side(x_center: float) -> str | None:
    if RIGHT_BAND[0] <= x_center <= RIGHT_BAND[1]:
        return "right"
    if LEFT_BAND[0] <= x_center <= LEFT_BAND[1]:
        return "left"
    return None


def _raw_margin_boxes(page) -> list[Element]:
    out: list[Element] = []
    d = page.get_text("dict")
    for blk in d.get("blocks", []):
        bbox = blk["bbox"]
        xc = (bbox[0] + bbox[2]) / 2
        side = _band_side(xc)
        if side is None:
            continue
        kind = "image" if blk.get("type") == 1 else "text"
        out.append(Element(*bbox, side=side, kind=kind))
    # vector drawings (TikZ margin figures, crimson sidenote rules, sparklines)
    text_h = TEXT_BOTTOM - TEXT_TOP
    for dr in page.get_drawings():
        r = dr["rect"]
        if r.is_empty or r.width < 1 or r.height < 1:
            continue
        # skip full-column background/clip rectangles (not discrete figures):
        # a real margin figure is never ~as tall as the whole text block.
        if r.height > 0.8 * text_h:
            continue
        xc = (r.x0 + r.x1) / 2
        side = _band_side(xc)
        if side is None:
            continue
        out.append(Element(r.x0, r.y0, r.x1, r.y1, side=side, kind="drawing"))
    return out


def _cluster(boxes: list[Element]) -> list[Element]:
    """Merge vertically-contiguous margin fragments into logical elements.

    A sidenote is a crimson rule + several wrapped text lines; a margin figure
    is an image/drawing + a caption block. Fragments within MERGE_GAP vertically
    (and on the same side) belong to one element. What is left as *separate*
    clusters that still overlap is a genuine collision.
    """
    elems: list[Element] = []
    for side in ("left", "right"):
        side_boxes = sorted((b for b in boxes if b.side == side), key=lambda b: b.y0)
        cur: Element | None = None
        for b in side_boxes:
            if cur is None:
                cur = Element(b.x0, b.y0, b.x1, b.y1, side, b.kind)
                continue
            # contiguous (small gap or already intersecting) -> same element
            if b.y0 <= cur.y1 + MERGE_GAP:
                cur.merge(b)
            else:
                elems.append(cur)
                cur = Element(b.x0, b.y0, b.x1, b.y1, side, b.kind)
        if cur is not None:
            elems.append(cur)
    return elems


def scan_page(page, pageno: int) -> list[Finding]:
    findings: list[Finding] = []
    elems = _cluster(_raw_margin_boxes(page))

    # (a) element-on-element overlap (per side, consecutive in y)
    for side in ("left", "right"):
        col = sorted((e for e in elems if e.side == side), key=lambda e: e.y0)
        for a, b in zip(col, col[1:]):
            inter = min(a.y1, b.y1) - max(a.y0, b.y0)
            if inter > OVERLAP_TOL:
                findings.append(Finding(
                    pageno, "overlap", side,
                    f"{a.kind}[{a.y0:.0f}-{a.y1:.0f}] ∩ {b.kind}[{b.y0:.0f}-{b.y1:.0f}] = {inter:.0f}pt",
                    (round(b.x0), round(max(a.y0, b.y0)), round(b.x1), round(min(a.y1, b.y1)))))

    # (b)/(c) overflow past footer / into header
    for e in elems:
        if e.y1 > TEXT_BOTTOM + EDGE_TOL:
            findings.append(Finding(
                pageno, "overflow-bottom", e.side,
                f"{e.kind} extends to {e.y1:.0f}pt ({(e.y1 - TEXT_BOTTOM):.0f}pt below text box; page edge {PAPER_H:.0f})",
                (round(e.x0), round(e.y0), round(e.x1), round(e.y1))))
        # genuine "pokes up into the header": starts >TOP_TOL above the body top
        # AND continues >10pt down into the body. The >TOP_TOL guard drops the
        # 2-3pt running-head / first-line cases that sit essentially at TEXT_TOP.
        if e.y0 < TEXT_TOP - TOP_TOL and e.y1 > TEXT_TOP + 10:
            findings.append(Finding(
                pageno, "overflow-top", e.side,
                f"{e.kind} starts at {e.y0:.0f}pt ({(TEXT_TOP - e.y0):.0f}pt up into header band)",
                (round(e.x0), round(e.y0), round(e.x1), round(e.y1))))
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    last = args.last or doc.page_count
    all_f: list[Finding] = []
    for i in range(args.first - 1, last):
        all_f.extend(scan_page(doc[i], i + 1))

    by_issue: dict[str, int] = {}
    for f in all_f:
        by_issue[f.issue] = by_issue.get(f.issue, 0) + 1

    print(f"Scanned pages {args.first}-{last} of {doc.page_count} | {args.pdf}")
    print(f"  overlaps:        {by_issue.get('overlap', 0)}")
    print(f"  overflow-bottom: {by_issue.get('overflow-bottom', 0)}")
    print(f"  overflow-top:    {by_issue.get('overflow-top', 0)}")
    print(f"  TOTAL findings:  {len(all_f)}\n")
    for f in all_f:
        print(f"  p{f.page:<4} {f.issue:<16} {f.side:<5} {f.detail}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([asdict(f) for f in all_f], fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
