#!/usr/bin/env python3
"""Report slides whose rendered text falls below a legibility floor.

Reads a deck's handout PDF with `pdftotext -bbox`, estimates each word's font
size from its box height, and lists the pages where several words in the body
of the slide (not the header, footer, or credit line) are smaller than the
floor. Text drawn inside figures counts, since that is what students must read.

Usage:
    python3 tools/legibility.py vol3/08_sandboxes [more decks]   (from slides/)
    python3 tools/legibility.py --all vol3
Options: --floor PT (default 6.5), --min-words N (default 6)

The floor is a screening threshold on the 16:9 Beamer page (about 16 cm wide):
6.5 pt there is roughly 9 to 10 pt on a projected slide, below which text is
hard to read from the back of a room. A page it lists needs a look at 100 dpi.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def words(pdf):
    """Each page's height and its text spans as (top, font size in pt, text).

    Uses PyMuPDF, which reports the real font size of every span, including
    text inside included figures (pdftotext -bbox crashes on these PDFs).
    """
    try:
        import pymupdf
    except ImportError:
        sys.exit("legibility.py needs PyMuPDF: pip install pymupdf")
    pages = []
    with pymupdf.open(str(pdf)) as doc:
        for page in doc:
            spans = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for s in line["spans"]:
                        if s["text"].strip():
                            spans.append((s["bbox"][1], s["size"], s["text"].strip()))
            pages.append({"h": page.rect.height, "w": spans})
    return pages


def kit_words():
    """Words drawn by the shared kit (stack layers, part names, lecture badges).

    The kit's stack thumbnail on divider and goal slides is a locator, not
    content, so its labels do not count against a deck.
    """
    import yaml
    ws = set()
    for f in (ROOT / "assets" / "kit").glob("vol*/*.svg"):
        ws |= set(re.findall(r">([^<>]{1,40})</", f.read_text(errors="ignore")))
    out = set()
    for w in ws:
        out |= set(w.split())
    return out


KIT = None


def check(deck, floor, min_words):
    vol, name = deck.split("/")
    pdf = ROOT / "_build" / vol / f"{name}-handout.pdf"
    if not pdf.exists():
        print(f"{deck}: no handout PDF ({pdf.name})")
        return 0
    bad = 0
    for i, p in enumerate(words(pdf), start=1):
        if i == 1:
            continue  # title slide
        h = p["h"]
        # Body only: below the header band, above the credit and footer lines.
        body = [(y, sz, t) for y, sz, t in p["w"] if 0.13 * h < y < 0.86 * h and re.search(r"\w", t)]
        global KIT
        if KIT is None:
            KIT = kit_words()
        # Axis tick numbers are read less closely than labels, so they get a
        # lower floor; any word, number or not, below 5 pt is flagged.
        small = [w for y, sz, t in body for w in t.split() if w not in KIT and (
                 sz < 5.0 or (sz < floor and not re.fullmatch(r"[-−+]?[\d.,:%]+[kKMGTx×%]?", w)))]
        if len(small) >= min_words:
            bad += 1
            sample = " ".join(small[:8])
            print(f"{deck} page {i}: {len(small)} words under {floor} pt, e.g. {sample}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decks", nargs="*")
    ap.add_argument("--all")
    ap.add_argument("--floor", type=float, default=6.5)
    ap.add_argument("--min-words", type=int, default=6)
    a = ap.parse_args()
    decks = list(a.decks)
    if a.all:
        decks += sorted(f"{a.all}/{p.name}" for p in (ROOT / a.all).iterdir()
                        if p.is_dir() and re.match(r"\d", p.name))
    total = sum(check(d, a.floor, a.min_words) for d in decks)
    print(f"legibility: {total} pages flagged")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
