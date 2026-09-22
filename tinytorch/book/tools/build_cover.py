#!/usr/bin/env python3
"""Generate the TinyTorch book cover.

Publication-Grade Swiss Modernist Style (ETH Zurich / Harvard Academic Palette):
- Alabaster light background with technical blueprint dot-grid band
- Integrated flame vector wordmark: Tiny🔥Torch
- Developer manifesto subtitle: Don't just import torch. Build it.
- Full 20-module runtime stack snaking circuit flow terminating in TinyGPT
- US Letter @ 300dpi (2550 x 3300)

Usage:  python3 tools/build_cover.py
Writes: assets/images/cover.svg + cover.png
"""
import subprocess
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "assets" / "images"

W, H = 2550, 3300                      # US Letter @ 300 dpi

# Swiss Modernist Palette (ETH Zurich / Harvard Academic)
BG         = "#F8F9FA"                 # Swiss alabaster light
BAND       = "#EDF1F5"                 # technical blueprint panel
DOTS       = "#C2CBD4"                 # architectural dot grid
ACCENT     = "#B7352D"                 # ETH Red / Swiss Crimson
TITLE_C    = "#111827"                 # dark charcoal
SUB_C      = "#374151"                 # medium charcoal
TEXT_C     = "#6B7280"                 # slate gray
BOX_BG     = "#FFFFFF"                 # chip white
BOX_BORDER = "#D1D5DB"                 # chip border
BOX_TEXT   = "#111827"                 # chip number
CAP_BG     = "#B7352D"                 # capstone badge
CAP_TEXT   = "#FFFFFF"                 # capstone text

SERIF = "'TeX Gyre Termes', 'Palatino Linotype', 'Book Antiqua', Palatino, serif"
SANS  = "'TeX Gyre Heros', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO  = "'JetBrains Mono', 'TeX Gyre Cursor', 'SF Mono', Menlo, monospace"

L, R = 300, 2250                       # content margins
CW = R - L                             # 1950


def _t(x, y, s, font, size, fill, anchor="start", weight="normal", style="normal", ls=0, extra=""):
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" '
            f'letter-spacing="{ls}"{extra}>{s}</text>')


def build():
    p = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs>',
        '<pattern id="dots" width="55" height="55" patternUnits="userSpaceOnUse">',
        f'<circle cx="2" cy="2" r="2" fill="{DOTS}" opacity="0.60"/></pattern>',
        f'<marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">',
        f'<path d="M0,0 L10,5 L0,10 z" fill="{ACCENT}"/></marker>',
        # Inlined Flame Gradients for 100% self-contained rendering
        '<radialGradient id="FLAME_BG" cx="68.8839" cy="124.2963" r="70.587" gradientTransform="matrix(-1 -4.343011e-03 -7.125917e-03 1.6408 131.9857 -79.3452)" gradientUnits="userSpaceOnUse">',
        '<stop offset="0.3144" stop-color="#FF9800"/>',
        '<stop offset="0.6616" stop-color="#FF6D00"/>',
        '<stop offset="0.9715" stop-color="#F44336"/>',
        '</radialGradient>',
        '<radialGradient id="FLAME_CORE" cx="64.9211" cy="54.0621" r="73.8599" gradientTransform="matrix(-0.0101 0.9999 0.7525 7.603777e-03 26.1538 -11.2668)" gradientUnits="userSpaceOnUse">',
        '<stop offset="0.2141" stop-color="#FFF176"/>',
        '<stop offset="0.3275" stop-color="#FFF27D"/>',
        '<stop offset="0.4868" stop-color="#FFF48F"/>',
        '<stop offset="0.6722" stop-color="#FFF7AD"/>',
        '<stop offset="0.7931" stop-color="#FFF9C4"/>',
        '<stop offset="0.8221" stop-color="#FFF8BD" stop-opacity="0.804"/>',
        '<stop offset="0.8627" stop-color="#FFF6AB" stop-opacity="0.529"/>',
        '<stop offset="0.9101" stop-color="#FFF38D" stop-opacity="0.2088"/>',
        '<stop offset="0.9409" stop-color="#FFF176" stop-opacity="0"/>',
        '</radialGradient>',
        '</defs>',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        f'<rect x="0" y="1250" width="{W}" height="1410" fill="{BAND}"/>',
        f'<rect x="0" y="1250" width="{W}" height="1410" fill="url(#dots)"/>',
        f'<rect x="0" y="1250" width="{W}" height="4" fill="{ACCENT}"/>'
    ]

    # --- series eyebrow -----------------------------------------------------
    p.append(f'<rect x="{L}" y="286" width="26" height="26" fill="{ACCENT}"/>')
    p.append(_t(L + 52, 311, "MACHINE LEARNING SYSTEMS SERIES", SANS, 30, TEXT_C, weight="bold", ls=8))

    # --- wordmark: Tiny🔥Torch -----------------------------------------------
    # Optically centered kerning: Tiny ends at 888, flame placed at 891 (195x195),
    # Torch starts at 1081. Exact 28px breathing room on both sides.
    flame_s = 195
    x_flame = 891
    y_flame = 880 - flame_s - 5
    x_torch = 1081

    p.append(_t(L, 880, "Tiny", SERIF, 280, TITLE_C, weight="bold"))
    p.append(
        f'<svg x="{x_flame}" y="{y_flame}" width="{flame_s}" height="{flame_s}" viewBox="0 0 128 128">'
        '<g>'
        '<path fill="url(#FLAME_BG)" d="M35.56,40.73c-0.57,6.08-0.97,16.84,2.62,21.42c0,0-1.69-11.82,13.46-26.65 '
        'c6.1-5.97,7.51-14.09,5.38-20.18c-1.21-3.45-3.42-6.3-5.34-8.29C50.56,5.86,51.42,3.93,53.05,4c9.86,0.44,25.84,3.18,32.63,20.22 '
        'c2.98,7.48,3.2,15.21,1.78,23.07c-0.9,5.02-4.1,16.18,3.2,17.55c5.21,0.98,7.73-3.16,8.86-6.14c0.47-1.24,2.1-1.55,2.98-0.56 '
        'c8.8,10.01,9.55,21.8,7.73,31.95c-3.52,19.62-23.39,33.9-43.13,33.9c-24.66,0-44.29-14.11-49.38-39.65 '
        'c-2.05-10.31-1.01-30.71,14.89-45.11C33.79,38.15,35.72,39.11,35.56,40.73z"/>'
        '<path fill="url(#FLAME_CORE)" d="M76.11,77.42c-9.09-11.7-5.02-25.05-2.79-30.37c0.3-0.7-0.5-1.36-1.13-0.93 '
        'c-3.91,2.66-11.92,8.92-15.65,17.73c-5.05,11.91-4.69,17.74-1.7,24.86c1.8,4.29-0.29,5.2-1.34,5.36 '
        'c-1.02,0.16-1.96-0.52-2.71-1.23c-2.15-2.05-3.7-4.72-4.44-7.6c-0.16-0.62-0.97-0.79-1.34-0.28c-2.8,3.87-4.25,10.08-4.32,14.47 '
        'C40.47,113,51.68,124,65.24,124c17.09,0,29.54-18.9,19.72-34.7C82.11,84.7,79.43,81.69,76.11,77.42z"/>'
        '</g>'
        '</svg>'
    )
    p.append(_t(x_torch, 880, "Torch", SERIF, 280, TITLE_C, weight="bold"))

    # --- subtitle block -----------------------------------------------------
    p.append(
        f'<text x="{L}" y="1045" xml:space="preserve">'
        f'<tspan font-family="{SERIF}" font-size="82" font-style="italic" fill="{SUB_C}">Don\'t just </tspan>'
        f'<tspan font-family="{MONO}" font-size="70" font-weight="bold" fill="{TITLE_C}">import torch</tspan>'
        f'<tspan font-family="{SERIF}" font-size="82" font-style="italic" fill="{SUB_C}">. Build it.</tspan>'
        f'</text>'
    )
    p.append(_t(L, 1135, "From Tensors to Hardware-Accelerated Transformers—and Beyond.", SANS, 42, TEXT_C, ls=1))

    # --- header divider bar -------------------------------------------------
    p.append(_t(L, 1330, "THE FULL RUNTIME STACK", MONO, 28, TEXT_C, ls=6))
    p.append(_t(R, 1330, "20 MODULES / 4 TIERS", MONO, 28, ACCENT, anchor="end", ls=6))
    p.append(f'<line x1="{L}" y1="1360" x2="{R}" y2="1360" stroke="{DOTS}" stroke-width="2"/>')

    # --- hero: 20-module continuous circuit snaking flow --------------------
    x0, pitch, n = 800, 165, 124
    lanes = [1440, 1750, 2060, 2370]
    cx = lambda c: x0 + c * pitch
    cy = lambda l: lanes[l] + n/2

    tiers = [
        ("I", "FOUNDATION", "MODULES 01–08", list(range(1, 9)), list(range(0, 8)),
         "Tensors · Activations · Layers · Losses · DataLoader · Autograd · Optimizers · Training"),
        ("II", "ARCHITECTURE", "MODULES 09–13", list(range(9, 14)), list(range(0, 5)),
         "Convolutions · Tokenization · Embeddings · Attention · Transformers"),
        ("III", "OPTIMIZATION", "MODULES 14–19", list(range(14, 20)), list(range(0, 6)),
         "Profiling · Quantization · Pruning · Acceleration · KV-Cache · MLPerf"),
    ]

    y1, y2, y3, y4 = cy(0), cy(1), cy(2), cy(3)

    p_08_x = cx(7) + n/2
    p_09_x = cx(0) + n/2
    mid_y1 = (y1 + y2) / 2 + 30
    bus_1 = (f"M {p_08_x} {y1} L {p_08_x + 60} {y1} Q {p_08_x + 90} {y1} {p_08_x + 90} {y1 + 30} "
             f"L {p_08_x + 90} {mid_y1 - 30} Q {p_08_x + 90} {mid_y1} {p_08_x + 60} {mid_y1} "
             f"L {p_09_x - 60} {mid_y1} Q {p_09_x - 90} {mid_y1} {p_09_x - 90} {mid_y1 + 30} "
             f"L {p_09_x - 90} {y2 - 30} Q {p_09_x - 90} {y2} {p_09_x - 60} {y2} L {p_09_x} {y2}")

    p_13_x = cx(4) + n/2
    p_14_x = cx(0) + n/2
    mid_y2 = (y2 + y3) / 2 + 30
    bus_2 = (f"M {p_13_x} {y2} L {p_13_x + 60} {y2} Q {p_13_x + 90} {y2} {p_13_x + 90} {y2 + 30} "
             f"L {p_13_x + 90} {mid_y2 - 30} Q {p_13_x + 90} {mid_y2} {p_13_x + 60} {mid_y2} "
             f"L {p_14_x - 60} {mid_y2} Q {p_14_x - 90} {mid_y2} {p_14_x - 90} {mid_y2 + 30} "
             f"L {p_14_x - 90} {y3 - 30} Q {p_14_x - 90} {y3} {p_14_x - 60} {y3} L {p_14_x} {y3}")

    p_19_x = cx(5) + n/2
    p_20_x = cx(0) + 225
    mid_y3 = (y3 + y4) / 2 + 30
    bus_3 = (f"M {p_19_x} {y3} L {p_19_x + 60} {y3} Q {p_19_x + 90} {y3} {p_19_x + 90} {y3 + 30} "
             f"L {p_19_x + 90} {mid_y3 - 30} Q {p_19_x + 90} {mid_y3} {p_19_x + 60} {mid_y3} "
             f"L {p_20_x} {mid_y3} L {p_20_x} {lanes[3] - 6}")

    # Draw continuous bus traces behind nodes
    p.append(f'<line x1="{cx(0) + n/2}" y1="{y1}" x2="{cx(7) + n/2}" y2="{y1}" stroke="{ACCENT}" stroke-width="4"/>')
    p.append(f'<path d="{bus_1}" fill="none" stroke="{ACCENT}" stroke-width="3.5" stroke-dasharray="8,5"/>')
    p.append(f'<line x1="{cx(0) + n/2}" y1="{y2}" x2="{cx(4) + n/2}" y2="{y2}" stroke="{ACCENT}" stroke-width="4"/>')
    p.append(f'<path d="{bus_2}" fill="none" stroke="{ACCENT}" stroke-width="3.5" stroke-dasharray="8,5"/>')
    p.append(f'<line x1="{cx(0) + n/2}" y1="{y3}" x2="{cx(5) + n/2}" y2="{y3}" stroke="{ACCENT}" stroke-width="4"/>')
    p.append(f'<path d="{bus_3}" fill="none" stroke="{ACCENT}" stroke-width="3.5" marker-end="url(#arr)"/>')

    # Draw module chip nodes and tier labels
    for (num, name, mods, ids, cols, topics), ly in zip(tiers, lanes):
        p.append(_t(L, ly + 58, num, SERIF, 72, ACCENT, weight="bold"))
        p.append(_t(L, ly + 106, name, SANS, 32, TITLE_C, weight="bold", ls=4))
        p.append(_t(L, ly + 144, mods, MONO, 24, TEXT_C, ls=2))

        for m, c in zip(ids, cols):
            x = cx(c)
            p.append(f'<rect x="{x}" y="{ly}" width="{n}" height="{n}" rx="8" fill="{BOX_BG}" stroke="{BOX_BORDER}" stroke-width="2.5"/>')
            p.append(_t(x + n/2, ly + n/2 + 15, f"{m:02d}", MONO, 42, BOX_TEXT, anchor="middle", weight="bold"))

        p.append(_t(cx(0), ly + n + 44, topics, SANS, 25, TEXT_C))

    # Tier IV: Capstone (TinyGPT)
    ly = lanes[3]
    p.append(_t(L, ly + 58, "IV", SERIF, 72, ACCENT, weight="bold"))
    p.append(_t(L, ly + 106, "CAPSTONE", SANS, 32, TITLE_C, weight="bold", ls=4))
    p.append(_t(L, ly + 144, "MODULE 20", MONO, 24, TEXT_C, ls=2))

    cap_x, cap_w = cx(0), 450
    p.append(f'<rect x="{cap_x}" y="{ly}" width="{cap_w}" height="{n}" rx="8" fill="{CAP_BG}"/>')
    p.append(
        f'<text x="{cap_x + cap_w/2}" y="{ly + n/2 + 18}" text-anchor="middle" xml:space="preserve">'
        f'<tspan font-family="{MONO}" font-size="{42}" font-weight="bold" fill="{CAP_TEXT}">20  </tspan>'
        f'<tspan font-family="{SERIF}" font-size="{56}" font-weight="bold" fill="{CAP_TEXT}">TinyGPT</tspan>'
        f'</text>'
    )
    p.append(_t(cap_x, ly + n + 44, "Full TinyGPT Architecture · MLPerf Benchmark Olympics", SANS, 25, TEXT_C))

    # Community Expansion Ports (Branching out from Capstone)
    node_out_x = cap_x + cap_w
    node_out_y = ly + n/2
    
    # Label for the branches
    p.append(_t(node_out_x + 280, node_out_y - 120, "COMMUNITY LABS &amp; EXTENSIONS", SANS, 22, TITLE_C, weight="bold", anchor="middle", ls=2))
    
    # 1. Straight right
    p.append(f'<path d="M {node_out_x} {node_out_y} L {node_out_x + 100} {node_out_y}" fill="none" stroke="{ACCENT}" stroke-width="3.5" stroke-dasharray="8,5" marker-end="url(#arr)"/>')
    p.append(_t(node_out_x + 130, node_out_y + 8, "LoRA / PEFT", MONO, 26, ACCENT, weight="bold"))

    # 2. Up and right
    p.append(f'<path d="M {node_out_x} {node_out_y} L {node_out_x + 40} {node_out_y} Q {node_out_x + 60} {node_out_y} {node_out_x + 60} {node_out_y - 30} L {node_out_x + 60} {node_out_y - 50} Q {node_out_x + 60} {node_out_y - 70} {node_out_x + 80} {node_out_y - 70} L {node_out_x + 100} {node_out_y - 70}" fill="none" stroke="{ACCENT}" stroke-width="3.5" stroke-dasharray="8,5" marker-end="url(#arr)"/>')
    p.append(_t(node_out_x + 130, node_out_y - 70 + 8, "Torch.Compile", MONO, 26, ACCENT, weight="bold"))

    # 3. Down and right
    p.append(f'<path d="M {node_out_x} {node_out_y} L {node_out_x + 40} {node_out_y} Q {node_out_x + 60} {node_out_y} {node_out_x + 60} {node_out_y + 30} L {node_out_x + 60} {node_out_y + 50} Q {node_out_x + 60} {node_out_y + 70} {node_out_x + 80} {node_out_y + 70} L {node_out_x + 100} {node_out_y + 70}" fill="none" stroke="{ACCENT}" stroke-width="3.5" stroke-dasharray="8,5" marker-end="url(#arr)"/>')
    p.append(_t(node_out_x + 130, node_out_y + 70 + 8, "Distributed", MONO, 26, ACCENT, weight="bold"))

    # --- author -------------------------------------------------------------
    p.append(f'<rect x="{L}" y="2780" width="120" height="4" fill="{ACCENT}"/>')
    p.append(_t(L, 2870, "Vijay Janapa Reddi", SERIF, 76, TITLE_C, weight="bold"))
    p.append(_t(L, 2940, "Harvard University", SANS, 38, TEXT_C))

    # --- footer -------------------------------------------------------------
    p.append(f'<line x1="{L}" y1="3040" x2="{R}" y2="3040" stroke="{DOTS}" stroke-width="2"/>')
    p.append(_t(L, 3110, "mlsysbook.ai/tinytorch", MONO, 30, TEXT_C))
    p.append(_t(R, 3110, "Open Access", SANS, 30, ACCENT, anchor="end", weight="bold", ls=2))

    p.append('</svg>')
    return "\n".join(p) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / "cover.svg"
    svg.write_text(build(), encoding="utf-8")
    subprocess.run(["rsvg-convert", "-w", str(W), "-h", str(H),
                    str(svg), "-o", str(OUT / "cover.png")], check=True)
    print(f"wrote {svg} and cover.png ({W}x{H})")
