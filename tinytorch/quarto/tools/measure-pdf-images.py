#!/usr/bin/env python3
# ruff: noqa: E501
r"""
measure-pdf-images.py — observed-size figure sizing recommendations.

Walk the rendered TinyTorch Lab Guide PDF, read every image's actual
on-page bounding box (the geometry XeLaTeX put on the page AFTER our
global \includegraphics cap fired). For each image, match it back to
the source .qmd reference, classify the rendered size against an
aspect-appropriate target, and emit a suggested `{width=N%}` override.

A human (or AI agent) decides which suggestions to apply — this tool
only reports, never mutates .qmd sources.

Pipeline:
  1. Open the rendered PDF with pdfplumber. For each page, extract
     every image's bounding box in PDF points (1pt = 1/72 in) plus
     its native pixel dimensions (`srcsize`).
  2. Deduplicate by (name, srcsize, width-in-pts, bbox). Images that
     repeat on every page (the fire-emoji in our header is the
     canonical example) collapse into one row so the report isn't
     300x noise.
  3. Walk source images on disk under `quarto/assets/` and
     `quarto/modules/**`. For each image file, read its native
     pixel dimensions via PNG/JPEG/SVG header parsing. Build a
     `native_dims -> source_path` lookup.
  4. For each unique PDF image, resolve its source file by matching
     `srcsize` to the on-disk lookup.
  5. Classify:
       - ULTRAWIDE   aspect > 2.5 -> width=100% (span the column)
       - OVERSIZED   rendered width > 0.75 * \textwidth (4.88 in)
                     -> scale down to 0.65 * \textwidth
       - TINY        rendered width < 0.20 * \textwidth AND aspect
                     < 1.5 -> flag for review (may be intentional icon)
       - OK          otherwise
  6. Grep .qmd sources for each flagged image's basename to show
     exactly which author line the override should land on.
  7. Emit TSV to stdout.

Usage (from `tinytorch/quarto/`):
  python3 tools/measure-pdf-images.py
  python3 tools/measure-pdf-images.py --verbose    # include OK rows
  python3 tools/measure-pdf-images.py --json       # machine-readable

Requires: pdfplumber (`uv pip install pdfplumber`).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("error: pdfplumber not installed. Run `uv pip install pdfplumber` first.",
          file=sys.stderr)
    sys.exit(2)


# ------------------------------------------------------------------------
# Page geometry (letter paper, 1in margins -> ~6.5in text width).
# ------------------------------------------------------------------------

TEXTWIDTH_IN    = 6.5
OVERSIZED_FRAC  = 0.75
TARGET_FRAC     = 0.65
TINY_FRAC       = 0.20
ULTRAWIDE_ASPECT = 2.5


# ------------------------------------------------------------------------
# Native-dimension readers for source image files (no external deps).
# ------------------------------------------------------------------------

def read_png_dims(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", header[16:24])
    return w, h


def read_jpeg_dims(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            byte = f.read(1)
            while byte and byte != b"\xff":
                byte = f.read(1)
            while byte == b"\xff":
                byte = f.read(1)
            if not byte:
                return None
            marker = byte[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                f.read(3)
                h, w = struct.unpack(">HH", f.read(4))
                return w, h
            seg_len = struct.unpack(">H", f.read(2))[0]
            f.read(seg_len - 2)


def read_image_dims(path: Path) -> tuple[int, int] | None:
    ext = path.suffix.lower()
    if ext == ".png":
        return read_png_dims(path)
    if ext in (".jpg", ".jpeg"):
        return read_jpeg_dims(path)
    return None


# ------------------------------------------------------------------------
# Source index: on-disk images, keyed by native pixel dimensions.
# ------------------------------------------------------------------------

def build_source_index(site_root: Path) -> dict[tuple[int, int], list[Path]]:
    index: dict[tuple[int, int], list[Path]] = {}
    exts = {".png", ".jpg", ".jpeg"}
    for img in site_root.rglob("*"):
        if img.suffix.lower() not in exts:
            continue
        if "_build" in img.parts:
            continue
        dims = read_image_dims(img)
        if dims:
            index.setdefault(dims, []).append(img)
    return index


# ------------------------------------------------------------------------
# .qmd scanner: find the author line that referenced a given basename.
# ------------------------------------------------------------------------

IMG_MD   = re.compile(r"!\[[^\]]*\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)(?:\{(?P<attrs>[^}]*)\})?")
IMG_HTML = re.compile(r"<img[^>]*\ssrc=\"(?P<path>[^\"]+)\"(?P<attrs>[^>]*)")
WIDTH_IN_ATTRS = re.compile(r"\bwidth\s*[:=]\s*[\"']?([^\"'\s}]+)")


@dataclass
class QmdRef:
    qmd_path: Path
    line_no: int
    explicit_width: str | None


def locate_nth_mermaid_block(qmd_path: Path, n: int) -> int | None:
    """Return the 1-indexed line number where the Nth mermaid block starts."""
    try:
        lines = qmd_path.read_text().splitlines()
    except Exception:
        return None
    count = 0
    for lineno, line in enumerate(lines, start=1):
        if re.match(r"\s*```\{mermaid", line):
            count += 1
            if count == n:
                return lineno
    return None


def build_qmd_index(site_root: Path) -> dict[str, list[QmdRef]]:
    index: dict[str, list[QmdRef]] = {}
    qmd_files = [
        q for q in site_root.rglob("*.qmd")
        if "_build" not in q.parts
        and "quarto/pdf" not in str(q).replace("\\", "/")
    ]
    for qmd in qmd_files:
        try:
            lines = qmd.read_text().splitlines()
        except Exception:
            continue
        for lineno, line in enumerate(lines, start=1):
            for pattern in (IMG_MD, IMG_HTML):
                for m in pattern.finditer(line):
                    path = m.group("path")
                    if not path or path.startswith(("http:", "https:", "data:")):
                        continue
                    attrs = m.group("attrs") or ""
                    wm = WIDTH_IN_ATTRS.search(attrs)
                    basename = Path(path).name
                    index.setdefault(basename, []).append(
                        QmdRef(qmd, lineno, wm.group(1) if wm else None)
                    )
    return index


# ------------------------------------------------------------------------
# PDF walker: unique images with rendered bboxes.
# ------------------------------------------------------------------------

@dataclass
class UniquePdfImage:
    srcsize: tuple[int, int]        # native pixels from the PDF
    width_in: float                 # rendered inches
    height_in: float                # rendered inches
    first_page: int
    occurrences: int

    @property
    def aspect(self) -> float:
        return self.width_in / max(self.height_in, 0.01)


def extract_unique_images(pdf_path: Path) -> list[UniquePdfImage]:
    """
    Group by (srcsize, rounded rendered dims). Same source image
    used at the same size on many pages (e.g., header wordmark)
    collapses into one row.
    """
    bucket: dict[tuple, UniquePdfImage] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            for im in page.images:
                srcsize = tuple(im.get("srcsize", (0, 0)))
                w_in = (im["x1"] - im["x0"]) / 72.0
                h_in = (im["y1"] - im["y0"]) / 72.0
                key = (srcsize, round(w_in, 2), round(h_in, 2))
                if key in bucket:
                    bucket[key].occurrences += 1
                else:
                    bucket[key] = UniquePdfImage(
                        srcsize=srcsize,
                        width_in=w_in,
                        height_in=h_in,
                        first_page=page.page_number,
                        occurrences=1,
                    )
    return sorted(bucket.values(), key=lambda u: u.first_page)


# ------------------------------------------------------------------------
# Classifier.
# ------------------------------------------------------------------------

def classify(img: UniquePdfImage) -> tuple[str, str | None]:
    oversized_in = OVERSIZED_FRAC * TEXTWIDTH_IN
    tiny_in      = TINY_FRAC      * TEXTWIDTH_IN

    if img.aspect > ULTRAWIDE_ASPECT:
        return "ULTRAWIDE", "width=100%"
    if img.width_in > oversized_in:
        pct = round(TARGET_FRAC * 100)
        return "OVERSIZED", f"width={pct}%"
    if img.width_in < tiny_in and img.aspect < 1.5:
        return "TINY", None
    return "OK", None


# ------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------

@dataclass
class Row:
    first_page: int
    occurrences: int
    srcsize: tuple[int, int]
    rendered_w_in: float
    rendered_h_in: float
    aspect: float
    verdict: str
    suggested: str | None
    source_path: str | None
    qmd_hits: list[str]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", type=Path, default=Path.cwd())
    p.add_argument("--pdf", type=Path, default=None,
                   help="rendered PDF (default: pdf/_build/TinyTorch-Guide.pdf)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="include OK rows")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    site = args.site
    pdf_f = args.pdf or site / "pdf/_build/TinyTorch-Guide.pdf"
    if not pdf_f.exists():
        print(f"error: PDF not found at {pdf_f}. Run `quarto render pdf/` first.",
              file=sys.stderr)
        return 2

    print(f"# reading PDF: {pdf_f}", file=sys.stderr)
    pdf_images   = extract_unique_images(pdf_f)
    source_index = build_source_index(site)
    qmd_index    = build_qmd_index(site)

    rows: list[Row] = []
    for img in pdf_images:
        verdict, suggested = classify(img)
        # Resolve source by native dims
        candidates = source_index.get(img.srcsize, [])
        source_path = str(candidates[0].relative_to(site)) if len(candidates) == 1 \
                      else (f"{len(candidates)} candidates" if candidates else None)
        basename = candidates[0].name if len(candidates) == 1 else None
        qmd_hits: list[str] = []
        if basename:
            for ref in qmd_index.get(basename, []):
                tag = f" [already width={ref.explicit_width}]" if ref.explicit_width else ""
                try:
                    rel = ref.qmd_path.relative_to(site)
                except ValueError:
                    rel = ref.qmd_path
                qmd_hits.append(f"{rel}:{ref.line_no}{tag}")

        # If this is a Quarto-generated mermaid raster (source lives at
        # <stem>_files/figure-latex/mermaid-figure-N.png), map back to
        # the Nth ` ```{mermaid} ` block in <stem>.qmd. The auto-named
        # cache dirs are reliable — Quarto writes them in document
        # order during render.
        if source_path and "figure-latex/mermaid-figure-" in source_path:
            m = re.match(r"(?P<dir>.+)_files/figure-latex/mermaid-figure-(?P<n>\d+)\.png",
                         source_path)
            if m:
                stem_dir = m.group("dir")
                n = int(m.group("n"))
                stem_qmd = site / f"{stem_dir}.qmd"
                if stem_qmd.exists():
                    hit = locate_nth_mermaid_block(stem_qmd, n)
                    if hit:
                        qmd_hits.append(f"{stem_qmd.relative_to(site)}:{hit} [mermaid #{n}]")
        rows.append(Row(
            first_page=img.first_page,
            occurrences=img.occurrences,
            srcsize=img.srcsize,
            rendered_w_in=round(img.width_in, 2),
            rendered_h_in=round(img.height_in, 2),
            aspect=round(img.aspect, 2),
            verdict=verdict,
            suggested=suggested,
            source_path=source_path,
            qmd_hits=qmd_hits,
        ))

    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2, default=str))
        return 0

    cols = ["first_page", "occur", "srcsize", "rendered_in",
            "aspect", "verdict", "suggested", "source", "qmd_refs"]
    print("\t".join(cols))
    for r in rows:
        if not args.verbose and r.verdict == "OK":
            continue
        print("\t".join([
            str(r.first_page),
            f"{r.occurrences}x" if r.occurrences > 1 else "1x",
            f"{r.srcsize[0]}x{r.srcsize[1]}",
            f"{r.rendered_w_in}x{r.rendered_h_in}",
            f"{r.aspect:.2f}",
            r.verdict,
            r.suggested or "-",
            r.source_path or "(unresolved)",
            " | ".join(r.qmd_hits) if r.qmd_hits else "-",
        ]))

    flagged = sum(1 for r in rows if r.verdict != "OK")
    print(f"# {flagged} flagged / {len(rows)} unique images "
          f"(output shows non-OK by default; --verbose for full)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
