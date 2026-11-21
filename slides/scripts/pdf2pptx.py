#!/usr/bin/env python3
"""Convert PDF slide decks to PowerPoint (PPTX) using high-resolution images.

Each PDF page is rendered at 300 DPI via pdftoppm (poppler) and placed as a
full-bleed image on a 16:9 PowerPoint slide. The result is visually identical
to the PDF, for presenting in PowerPoint or Keynote with annotations and
presenter view. Slides are images, not editable text.

Speaker notes: when <deck>-notes.pdf sits beside <deck>.pdf (built with
`make notes V=<n> D=<deck>`) and has one page per slide, each page's note is
copied into that slide's notes pane, so presenter view shows the script.
Each note starts at its timing tag, "[N min]". Handout and notes PDFs passed
in a batch glob are skipped.

Requirements:
    - pdftoppm (poppler-utils): apt install poppler-utils / brew install poppler
    - python-pptx:             pip install python-pptx
    - Pillow:                  pip install Pillow

Usage:
    python3 pdf2pptx.py input.pdf                    # -> input.pptx
    python3 pdf2pptx.py input.pdf -o output.pptx     # explicit output
    python3 pdf2pptx.py _build/vol1/*.pdf             # batch convert
    python3 pdf2pptx.py _build/vol1/*.pdf --dpi 200   # lower DPI (smaller files)
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

def check_dependencies():
    if not shutil.which("pdftoppm"):
        print("ERROR: pdftoppm not found. Install poppler:", file=sys.stderr)
        print("  macOS:  brew install poppler", file=sys.stderr)
        print("  Ubuntu: sudo apt install poppler-utils", file=sys.stderr)
        sys.exit(1)
    try:
        from pptx import Presentation  # noqa: F401
    except ImportError:
        print("ERROR: python-pptx not found. Install:", file=sys.stderr)
        print("  pip install python-pptx Pillow", file=sys.stderr)
        sys.exit(1)


NOTE_START = re.compile(r"(?:\[\d+(?:\.\d+)?\s*min\]|\b\d+(?:\.\d+)?\s*min\s*[·\u00b7])")
FRAME_NUM_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
CUE_START_RE = re.compile(r"^(?:Check|Watch for|Next|Optional(?:\s*\([^)]*\))?)\s*:")


def page_count(pdf_path: str) -> int:
    out = subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 0


def _rejoin_lines(lines: list[str]) -> str:
    """Join wrapped lines inside a paragraph, fixing soft end-of-line hyphens."""
    out = ""
    for ln in lines:
        ln = " ".join(ln.split())
        if not ln:
            continue
        if not out:
            out = ln
        elif out.endswith("-") and len(out) >= 2 and out[-2].isalpha() and ln[0].islower():
            out = out[:-1] + ln
        else:
            out = out + " " + ln
    return out


def _extract_portrait_note(page) -> str:
    """Extract cleanly formatted multi-paragraph text from the bottom notes panel
    of a Portrait US Letter (top-slide / bottom-notes) page via PyMuPDF."""
    import fitz

    h = page.rect.height
    w = page.rect.width
    # Slide occupies top half (y ~ 0.04h .. 0.43h); notes occupy y ~ 0.44h .. 0.85h;
    # quiet page footer sits below 0.82h with "Study & Presenter Notes".
    clip = fitz.Rect(0, h * 0.44, w, h * 0.88)
    raw_lines = []
    for blk in page.get_text("dict", clip=clip).get("blocks", []):
        for ln in blk.get("lines", []):
            txt = "".join(sp["text"] for sp in ln.get("spans", [])).strip()
            if not txt:
                continue
            if "Study & Presenter Notes" in txt or "Machine Learning Systems" in txt:
                continue
            y0, y1 = ln["bbox"][1], ln["bbox"][3]
            raw_lines.append((y0, y1, txt))
    if not raw_lines:
        return ""
    raw_lines.sort(key=lambda t: t[0])
    # Compute typical single-line baseline step for paragraph-break detection
    gaps = [raw_lines[i][0] - raw_lines[i - 1][0] for i in range(1, len(raw_lines))]
    normal_gaps = sorted(g for g in gaps if 6.0 <= g <= 22.0)
    line_step = normal_gaps[len(normal_gaps) // 2] if normal_gaps else 13.5

    paras = []
    cur = [raw_lines[0][2]]
    for i in range(1, len(raw_lines)):
        gap = raw_lines[i][0] - raw_lines[i - 1][0]
        txt = raw_lines[i][2]
        prev_txt = raw_lines[i - 1][2]
        new_para = (
            gap > line_step * 1.32
            or bool(CUE_START_RE.match(txt))
            or bool(NOTE_START.match(prev_txt))
        )
        if new_para:
            paras.append(_rejoin_lines(cur))
            cur = [txt]
        else:
            cur.append(txt)
    if cur:
        paras.append(_rejoin_lines(cur))
    return "\n\n".join(p for p in paras if p)


def _deck_frame_numbers(pdf_path: str, total_frames: int) -> list[int]:
    """Map each 1-indexed page of the presenting PDF to its logical Beamer frame
    number (1..total_frames) using the footer's 'k / M' indicator."""
    import fitz

    mapping = []
    with fitz.open(pdf_path) as doc:
        prev = 1
        for idx, page in enumerate(doc):
            if idx == 0:
                mapping.append(1)
                prev = 1
                continue
            h, w = page.rect.height, page.rect.width
            foot = page.get_text("text", clip=fitz.Rect(w * 0.65, h * 0.88, w, h))
            m = FRAME_NUM_RE.search(foot)
            if m:
                k = int(m.group(1))
                if 1 <= k <= total_frames:
                    mapping.append(k)
                    prev = k
                    continue
            # Fallback if a frame omits the footer (e.g., plain divider)
            nxt = min(total_frames, prev + 1)
            mapping.append(nxt)
            prev = nxt
    return mapping


def read_notes(pdf_path: str, pages: int) -> list:
    """Speaker notes per slide page from <deck>-notes.pdf, or [] if absent.

    Supports both:
    - Portrait US Letter handout notes (1 page per logical frame, slide on top,
      notes on bottom): preserves paragraph breaks and maps each presenting
      overlay page to its logical frame's note (marking build steps).
    - Legacy two-screen landscape notes (1 page per presenting page): reads the
      right half while preserving paragraph breaks.
    """
    notes_pdf = str(Path(pdf_path).with_name(Path(pdf_path).stem + "-notes.pdf"))
    if not os.path.isfile(notes_pdf):
        return []
    n_notes = page_count(notes_pdf)
    if n_notes == 0:
        return []

    # Try PyMuPDF first (handles both Portrait printable notes and clean paragraphs)
    try:
        import fitz

        with fitz.open(notes_pdf) as ndoc:
            r0 = ndoc[0].rect
            is_portrait = r0.height > r0.width
            if is_portrait:
                frame_notes = [_extract_portrait_note(p) for p in ndoc]
                if n_notes == pages:
                    return frame_notes
                # Map each presenting overlay page to its logical frame
                fnums = _deck_frame_numbers(pdf_path, n_notes)
                # Count overlays per frame so multi-step builds show "(Step j of s)"
                counts = {}
                for k in fnums:
                    counts[k] = counts.get(k, 0) + 1
                seen = {}
                out = []
                for k in fnums:
                    seen[k] = seen.get(k, 0) + 1
                    txt = frame_notes[k - 1] if 1 <= k <= len(frame_notes) else ""
                    if txt and counts[k] > 1:
                        lines = txt.split("\n\n")
                        lines[0] = f"{lines[0]}  (Step {seen[k]} of {counts[k]})"
                        txt = "\n\n".join(lines)
                    out.append(txt)
                return out
    except ImportError:
        pass

    # Legacy fallback: two-screen landscape notes PDF with matching page count
    if n_notes != pages:
        return []
    box = subprocess.run(["pdfinfo", "-box", notes_pdf], capture_output=True,
                         text=True).stdout
    m = re.search(r"MediaBox:\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", box)
    w, h = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
    crop = ["-x", str(int(w / 2)), "-y", "0", "-W", str(int(w / 2) + 1),
            "-H", str(int(h) + 1)] if w else []
    notes = []
    for p in range(1, pages + 1):
        text = subprocess.run(
            ["pdftotext", *crop, "-f", str(p), "-l", str(p), notes_pdf, "-"],
            capture_output=True, text=True,
        ).stdout
        m = NOTE_START.search(text)
        body = text[m.start():] if m else ""
        # Preserve blank-line paragraph breaks instead of flattening everything
        paras = [_rejoin_lines(blk.splitlines()) for blk in re.split(r"\n\s*\n", body)]
        notes.append("\n\n".join(p for p in paras if p))
    return notes


def convert_pdf_to_pptx(pdf_path: str, pptx_path: str, dpi: int = 300) -> dict:
    """Convert a single PDF to PPTX. Returns stats dict."""
    from pptx import Presentation
    from pptx.util import Inches, Emu

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), pdf_path, os.path.join(tmp, "slide")],
            check=True,
            capture_output=True,
        )
        images = sorted(glob.glob(os.path.join(tmp, "slide-*.png")))
        if not images:
            return {"ok": False, "error": "pdftoppm produced no images"}

        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)

        notes = read_notes(pdf_path, len(images))
        for k, img_path in enumerate(images):
            slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
            slide.shapes.add_picture(
                img_path, Emu(0), Emu(0), prs.slide_width, prs.slide_height
            )
            if notes and notes[k]:
                slide.notes_slide.notes_text_frame.text = notes[k]

        os.makedirs(os.path.dirname(pptx_path) or ".", exist_ok=True)
        prs.save(pptx_path)

        size_mb = os.path.getsize(pptx_path) / (1024 * 1024)
        return {"ok": True, "slides": len(images), "size_mb": size_mb,
                "notes": sum(1 for n in notes if n)}


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF slide decks to PowerPoint (image-based)"
    )
    parser.add_argument("pdfs", nargs="+", help="PDF file(s) to convert")
    parser.add_argument("-o", "--output", help="Output path (single file only)")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    parser.add_argument("--output-dir", help="Output directory (batch mode)")
    args = parser.parse_args()

    check_dependencies()

    if args.output and len(args.pdfs) > 1:
        print("ERROR: -o/--output only works with a single input file", file=sys.stderr)
        sys.exit(1)

    total = 0
    failed = 0

    for pdf in args.pdfs:
        if not os.path.isfile(pdf):
            print(f"  SKIP {pdf} (not found)")
            continue
        if Path(pdf).stem.endswith(("-handout", "-notes")):
            continue

        if args.output:
            pptx = args.output
        elif args.output_dir:
            pptx = os.path.join(args.output_dir, Path(pdf).stem + ".pptx")
        else:
            pptx = str(Path(pdf).with_suffix(".pptx"))

        print(f"  Converting {pdf}...")
        result = convert_pdf_to_pptx(pdf, pptx, dpi=args.dpi)

        if result["ok"]:
            print(f"    -> {pptx} ({result['slides']} slides, "
                  f"{result['notes']} with notes, {result['size_mb']:.1f} MB)")
            total += 1
        else:
            print(f"    FAILED: {result['error']}")
            failed += 1

    print(f"\nDone: {total} converted, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
