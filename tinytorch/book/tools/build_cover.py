#!/usr/bin/env python3
"""Generate the TinyTorch *book* cover.

The book and the lab guide shipped byte-identical cover art, and that art is
really the guide's: its tagline ("Don't just import torch. Build it.") is the
guide's subtitle. Its two subtitle lines ("The xv6 of Machine Learning
Systems", "From Tensors to Systems Acceleration") also predate the retitle in
1770d8ea2d and match neither artifact's current title.

This gives the book its own cover, carrying the book's own title, in the
brand palette from guide/pdf/_quarto.yml. Vector, so the next retitle is an
edit here rather than a re-export from a design tool.

Usage:  python3 book/tools/build_cover.py
Writes: book/assets/images/cover.svg + cover.png (US Letter @ 300dpi)
"""
import subprocess
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "assets" / "images"

W, H = 2550, 3300                      # US Letter @ 300 dpi

NAVY      = "#1B3A5F"                  # torchnavy
NAVY_UP   = "#24496F"                  # raised band
NAVY_EDGE = "#3A6390"
ORANGE    = "#FF8246"                  # flameorange
CREAM     = "#FFFFFF"
MUTED     = "#B8CDE3"
DIM       = "#7FA3C9"

SERIF = "'TeX Gyre Pagella', Palatino, 'Palatino Linotype', Georgia, serif"
SANS  = "'TeX Gyre Heros', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO  = "'TeX Gyre Cursor', 'SF Mono', Menlo, monospace"

# Foundation 01-08, Architecture 09-13, Optimization 14-19, Capstone 20.
TIERS = [
    ("Foundation",   "01–08", 8,  False),
    ("Architecture", "09–13", 5,  False),
    ("Optimization", "14–19", 6,  False),
    ("Capstone",     "20",         1,  True),
]

L, R = 300, 2250                       # content margins
CW = R - L


def tier_band(y, h, name, rng, n, accent):
    fill, edge = ((NAVY_UP, ORANGE) if accent else (NAVY_UP, NAVY_EDGE))
    s = (f'<rect x="{L}" y="{y}" width="{CW}" height="{h}" rx="6" fill="{fill}" '
         f'stroke="{edge}" stroke-width="{3 if accent else 2}"/>\n')
    if accent:
        s += f'<rect x="{L}" y="{y}" width="10" height="{h}" fill="{ORANGE}"/>\n'
    s += (f'<text x="{L + 54}" y="{y + h/2 + 6}" font-family="{SANS}" font-size="58" '
          f'font-weight="700" fill="{CREAM}">{name}</text>\n')
    s += (f'<text x="{R - 54}" y="{y + h/2 + 4}" text-anchor="end" font-family="{MONO}" '
          f'font-size="46" fill="{ORANGE if accent else DIM}">{rng}</text>\n')
    if accent:
        return s
    # module ticks, right-aligned ahead of the range label
    tw, gap = 26, 12
    total = n * tw + (n - 1) * gap
    x0 = R - 230 - total
    for i in range(n):
        c = ORANGE if accent else DIM
        s += (f'<rect x="{x0 + i*(tw+gap)}" y="{y + h/2 - 13}" width="{tw}" height="26" '
              f'rx="2" fill="{c}" opacity="{1 if accent else 0.55}"/>\n')
    return s


def build():
    p = [f'<?xml version="1.0" encoding="UTF-8"?>',
         f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="{NAVY}"/>']

    # --- series eyebrow -----------------------------------------------------
    p.append(f'<rect x="{L}" y="300" width="150" height="7" fill="{ORANGE}"/>')
    p.append(f'<text x="{L}" y="410" font-family="{SANS}" font-size="40" font-weight="700" '
             f'letter-spacing="9" fill="{DIM}">MACHINE LEARNING SYSTEMS SERIES</text>')

    # --- brand mark ---------------------------------------------------------
    # favicon.svg is the flame logo; rsvg resolves the href relative to this file.
    p.append(f'<image href="favicon.svg" xlink:href="favicon.svg" '
             f'x="{R - 230}" y="268" width="230" height="230"/>')

    # --- title block --------------------------------------------------------
    p.append(f'<text x="{L}" y="890" font-family="{SERIF}" font-size="290" font-weight="700" '
             f'fill="{CREAM}">TinyTorch</text>')
    p.append(f'<text x="{L}" y="1030" font-family="{SERIF}" font-size="104" font-style="italic" '
             f'fill="{ORANGE}">From Tensors to Transformers</text>')
    p.append(f'<text x="{L}" y="1130" font-family="{SANS}" font-size="52" fill="{MUTED}">'
             f'Engineering Deep Learning Systems from Scratch</text>')

    # --- hero: the twenty modules you build --------------------------------
    p.append(f'<text x="{L}" y="1470" font-family="{SANS}" font-size="38" font-weight="700" '
             f'letter-spacing="7" fill="{DIM}">TWENTY MODULES, BUILT FROM NOTHING</text>')
    y, bh, gap = 1540, 190, 42
    for name, rng, n, accent in TIERS:
        p.append(tier_band(y, bh, name, rng, n, accent))
        y += bh + gap

    # --- author -------------------------------------------------------------
    p.append(f'<rect x="{L}" y="2560" width="150" height="5" fill="{ORANGE}"/>')
    p.append(f'<text x="{L}" y="2700" font-family="{SERIF}" font-size="82" font-weight="700" '
             f'fill="{CREAM}">Prof. Vijay Janapa Reddi</text>')
    p.append(f'<text x="{L}" y="2790" font-family="{SANS}" font-size="52" fill="{MUTED}">'
             f'Harvard University</text>')

    # --- footer -------------------------------------------------------------
    p.append(f'<rect x="{L}" y="3010" width="{CW}" height="2" fill="{NAVY_EDGE}"/>')
    p.append(f'<text x="{L}" y="3105" font-family="{MONO}" font-size="46" fill="{ORANGE}">'
             f'mlsysbook.ai/tinytorch</text>')
    p.append(f'<text x="{R}" y="3105" text-anchor="end" font-family="{SANS}" font-size="46" '
             f'fill="{DIM}">Open access</text>')

    p.append('</svg>')
    return "\n".join(p) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / "cover.svg"
    svg.write_text(build(), encoding="utf-8")
    subprocess.run(["rsvg-convert", "-w", str(W), "-h", str(H),
                    str(svg), "-o", str(OUT / "cover.png")], check=True)
    print(f"wrote {svg} and cover.png ({W}x{H})")
