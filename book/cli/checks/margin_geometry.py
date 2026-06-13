"""Binder-native rendered-PDF margin geometry checks.

This module is the shared implementation behind ``binder layout overlaps`` and
post-build PDF validation. It intentionally depends only on rendered PDF
geometry from PyMuPDF, not on source anchors or LaTeX log heuristics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IN = 72.0

# Production PDF geometry from book/quarto/tex/header-includes.tex.
PAPER_W, PAPER_H = 8 * IN, 10 * IN
TOP, BOTTOM = 0.5 * IN, 0.625 * IN
INNER, OUTER = 0.875 * IN, 1.75 * IN
HEADHEIGHT, HEADSEP = 14.0, 18.0
MARGINSEP, MARGINW = 0.25 * IN, 1.25 * IN

TEXT_W = PAPER_W - INNER - OUTER
TEXT_TOP = TOP + HEADHEIGHT + HEADSEP
TEXT_BOTTOM = PAPER_H - BOTTOM

RIGHT_TEXT_EDGE = INNER + TEXT_W
RIGHT_BAND = (RIGHT_TEXT_EDGE + 2, PAPER_W)
LEFT_TEXT_EDGE = OUTER
LEFT_BAND = (0, LEFT_TEXT_EDGE - 2)

OVERLAP_TOL = 1.5
MERGE_GAP = 4.0
CAPTION_GAP = 6.0
EDGE_TOL = 2.0
TOP_TOL = 6.0


@dataclass
class MarginGeometryElement:
    x0: float
    y0: float
    x1: float
    y1: float
    side: str
    kind: str
    snippet: str = ""

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def merge(self, other: "MarginGeometryElement") -> None:
        self.x0 = min(self.x0, other.x0)
        self.y0 = min(self.y0, other.y0)
        self.x1 = max(self.x1, other.x1)
        self.y1 = max(self.y1, other.y1)
        if self.kind != other.kind:
            self.kind = "mixed"
        if other.snippet and other.snippet not in self.snippet:
            self.snippet = " ".join(
                part for part in (self.snippet, other.snippet) if part
            )[:160]


@dataclass(frozen=True)
class MarginGeometryFinding:
    page: int
    issue: str
    side: str
    detail: str
    bbox: tuple[int, int, int, int]
    snippet: str = ""


@dataclass(frozen=True)
class MarginGeometrySummary:
    pdf_path: Path
    page_count: int
    pages_scanned: int
    findings: list[MarginGeometryFinding]

    @property
    def counts(self) -> Counter:
        return Counter(f.issue for f in self.findings)

    @property
    def overlaps(self) -> int:
        return self.counts.get("overlap", 0)

    @property
    def overflow_bottom(self) -> int:
        return self.counts.get("overflow-bottom", 0)

    @property
    def overflow_top(self) -> int:
        return self.counts.get("overflow-top", 0)

    @property
    def overflows(self) -> int:
        return self.overflow_bottom + self.overflow_top


def _band_side(x_center: float) -> str | None:
    if RIGHT_BAND[0] <= x_center <= RIGHT_BAND[1]:
        return "right"
    if LEFT_BAND[0] <= x_center <= LEFT_BAND[1]:
        return "left"
    return None


def _text_from_block(block: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = (span.get("text") or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts)[:160]


def _rect_tuple(rect) -> tuple[float, float, float, float]:
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


def _x_overlap(a: MarginGeometryElement, b: MarginGeometryElement) -> float:
    return min(a.x1, b.x1) - max(a.x0, b.x0)


def _y_overlap(a: MarginGeometryElement, b: MarginGeometryElement) -> float:
    return min(a.y1, b.y1) - max(a.y0, b.y0)


def _vertical_gap(a: MarginGeometryElement, b: MarginGeometryElement) -> float:
    if a.y1 < b.y0:
        return b.y0 - a.y1
    if b.y1 < a.y0:
        return a.y0 - b.y1
    return -_y_overlap(a, b)


def _contains(
    outer: MarginGeometryElement,
    inner: MarginGeometryElement,
    tol: float = 2.0,
) -> bool:
    return (
        inner.x0 >= outer.x0 - tol
        and inner.x1 <= outer.x1 + tol
        and inner.y0 >= outer.y0 - tol
        and inner.y1 <= outer.y1 + tol
    )


def _body_spanners(page) -> list[MarginGeometryElement]:
    """Large body graphics/tables whose internals may enter the margin band."""
    out: list[MarginGeometryElement] = []
    min_width = TEXT_W * 0.65
    text_h = TEXT_BOTTOM - TEXT_TOP

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1:
            continue
        x0, y0, x1, y1 = block["bbox"]
        if (x1 - x0) >= min_width:
            out.append(MarginGeometryElement(x0, y0, x1, y1, "body", "image"))

    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.is_empty:
            continue
        if rect.height > 0.9 * text_h:
            continue
        x0, y0, x1, y1 = _rect_tuple(rect)
        if (x1 - x0) >= min_width:
            out.append(MarginGeometryElement(x0, y0, x1, y1, "body", "drawing"))
    return out


def _is_body_intrusion(
    elem: MarginGeometryElement,
    body_spanners: list[MarginGeometryElement],
) -> bool:
    """Return true when a candidate is really an internal body-figure object."""
    if not body_spanners:
        return False
    for body in body_spanners:
        if _x_overlap(elem, body) <= 0:
            continue
        y_overlap = _y_overlap(elem, body)
        if y_overlap <= 0:
            continue
        if y_overlap >= min(max(elem.height * 0.5, 4.0), elem.height):
            return True
    return False


def _raw_margin_boxes(page) -> list[MarginGeometryElement]:
    out: list[MarginGeometryElement] = []
    body_spanners = _body_spanners(page)

    for block in page.get_text("dict").get("blocks", []):
        bbox = block["bbox"]
        xc = (bbox[0] + bbox[2]) / 2
        side = _band_side(xc)
        if side is None:
            continue
        kind = "image" if block.get("type") == 1 else "text"
        elem = MarginGeometryElement(*bbox, side=side, kind=kind)
        if kind == "text":
            elem.snippet = _text_from_block(block)
        if not _is_body_intrusion(elem, body_spanners):
            out.append(elem)

    text_h = TEXT_BOTTOM - TEXT_TOP
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.is_empty or rect.width < 1 or rect.height < 1:
            continue
        if rect.height > 0.8 * text_h:
            continue
        x0, y0, x1, y1 = _rect_tuple(rect)
        xc = (x0 + x1) / 2
        side = _band_side(xc)
        if side is None:
            continue
        elem = MarginGeometryElement(x0, y0, x1, y1, side=side, kind="drawing")
        if not _is_body_intrusion(elem, body_spanners):
            out.append(elem)
    return out


def _mergeable(
    a: MarginGeometryElement,
    b: MarginGeometryElement,
) -> bool:
    if a.side != b.side:
        return False

    gap = _vertical_gap(a, b)
    x_overlap = _x_overlap(a, b)

    if a.kind == "text" and b.kind == "text":
        return gap <= MERGE_GAP and x_overlap > min(a.width, b.width) * 0.2

    a_is_graphic = a.kind in {"image", "drawing", "mixed"}
    b_is_graphic = b.kind in {"image", "drawing", "mixed"}
    if a_is_graphic and b_is_graphic:
        return gap <= MERGE_GAP and x_overlap > 0

    graphic = a if a_is_graphic else b
    text = b if a_is_graphic else a
    if _contains(graphic, text):
        return True
    # Margin diagrams often use a narrow vertical side label next to stacked
    # boxes (for example the Vol. 2 chapter mini-TOC). It vertically intersects
    # the graphic by design, but it is one logical element, not a collision.
    if text.width < 14 and _y_overlap(graphic, text) > OVERLAP_TOL:
        return True
    if graphic.kind == "drawing" and (graphic.width < 4 or graphic.height < 4):
        return gap <= MERGE_GAP or _y_overlap(graphic, text) > 0
    return (
        0 <= gap <= CAPTION_GAP
        and x_overlap > min(graphic.width, text.width) * 0.25
    )


def _cluster(boxes: list[MarginGeometryElement]) -> list[MarginGeometryElement]:
    elems: list[MarginGeometryElement] = []
    for side in ("left", "right"):
        side_boxes = sorted((b for b in boxes if b.side == side), key=lambda b: b.y0)
        for box in side_boxes:
            for elem in reversed(elems):
                if elem.side != side:
                    continue
                if _mergeable(elem, box):
                    elem.merge(box)
                    break
            else:
                elems.append(MarginGeometryElement(
                    box.x0, box.y0, box.x1, box.y1, box.side, box.kind, box.snippet
                ))
    changed = True
    while changed:
        changed = False
        for i, elem in enumerate(elems):
            for j in range(i + 1, len(elems)):
                other = elems[j]
                if _mergeable(elem, other):
                    elem.merge(other)
                    del elems[j]
                    changed = True
                    break
            if changed:
                break
    return sorted(elems, key=lambda e: (e.side, e.y0))


def scan_page(page, pageno: int) -> list[MarginGeometryFinding]:
    findings: list[MarginGeometryFinding] = []
    elems = _cluster(_raw_margin_boxes(page))

    for side in ("left", "right"):
        col = sorted((e for e in elems if e.side == side), key=lambda e: e.y0)
        for a, b in zip(col, col[1:]):
            y_inter = _y_overlap(a, b)
            x_inter = _x_overlap(a, b)
            if y_inter > OVERLAP_TOL and x_inter > 1.0:
                y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
                snippet = " | ".join(part for part in (a.snippet, b.snippet) if part)
                findings.append(MarginGeometryFinding(
                    pageno,
                    "overlap",
                    side,
                    (
                        f"{a.kind}[{a.y0:.0f}-{a.y1:.0f}] ∩ "
                        f"{b.kind}[{b.y0:.0f}-{b.y1:.0f}] = {y_inter:.0f}pt"
                    ),
                    (
                        round(max(a.x0, b.x0)),
                        round(y0),
                        round(min(a.x1, b.x1)),
                        round(y1),
                    ),
                    snippet[:160],
                ))

    for elem in elems:
        if elem.y1 > TEXT_BOTTOM + EDGE_TOL:
            findings.append(MarginGeometryFinding(
                pageno,
                "overflow-bottom",
                elem.side,
                (
                    f"{elem.kind} extends to {elem.y1:.0f}pt "
                    f"({(elem.y1 - TEXT_BOTTOM):.0f}pt below text box; "
                    f"page edge {PAPER_H:.0f})"
                ),
                (round(elem.x0), round(elem.y0), round(elem.x1), round(elem.y1)),
                elem.snippet,
            ))
        if elem.y0 < TEXT_TOP - TOP_TOL and elem.y1 > TEXT_TOP + 10:
            findings.append(MarginGeometryFinding(
                pageno,
                "overflow-top",
                elem.side,
                (
                    f"{elem.kind} starts at {elem.y0:.0f}pt "
                    f"({(TEXT_TOP - elem.y0):.0f}pt up into header band)"
                ),
                (round(elem.x0), round(elem.y0), round(elem.x1), round(elem.y1)),
                elem.snippet,
            ))
    return findings


def scan_pdf(
    pdf_path: Path | str,
    *,
    limit: int = 0,
    first: int = 1,
    last: int = 0,
) -> MarginGeometrySummary:
    import fitz  # PyMuPDF

    path = Path(pdf_path)
    doc = fitz.open(str(path))
    try:
        page_count = doc.page_count
        start = max(1, first)
        stop = last or page_count
        if limit > 0:
            stop = min(stop, start + limit - 1)
        stop = min(stop, page_count)
        findings: list[MarginGeometryFinding] = []
        for i in range(start - 1, stop):
            findings.extend(scan_page(doc[i], i + 1))
        return MarginGeometrySummary(
            pdf_path=path,
            page_count=page_count,
            pages_scanned=max(0, stop - start + 1),
            findings=findings,
        )
    finally:
        doc.close()
