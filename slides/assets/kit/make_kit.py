#!/usr/bin/env python3
"""Generate the shared slide figures for one volume.

The kit holds the figures that recur across a volume's lectures, so every
deck draws them the same way:

  stack_*      the ML system stack (the recurring map), whole, one layer
               highlighted, built up one layer per step, and as a roadmap
               with each lecture placed on its layer
  machine_*    one accelerator node (host, link, accelerator with HBM,
               on-chip SRAM, compute array), built up and highlighted
  dam_*        the D-A-M triangle, plain and with one vertex highlighted

Colors follow the book's color contract: memory blue, compute orange,
data green, interconnect violet, structure brown, danger red. The volume
accent is theme only and marks selection (the highlighted layer).

Usage:
    python3 make_kit.py --volume 1            # writes vol1/*.svg and *.pdf
"""

import argparse
import shutil
import subprocess
from pathlib import Path

ACCENT = {1: "#A51C30", 2: "#1F407A", 3: "#581C87", 4: "#1A4D3E"}

C = {
    "BlueLine": "#006395", "BlueL": "#D1E6F3",
    "OrangeLine": "#E67817", "OrangeL": "#FCE4CC",
    "GreenLine": "#008F45", "GreenL": "#D4EFDF",
    "VioletLine": "#7E317B", "VioletL": "#E6D4E5",
    "BrownLine": "#78492A", "BrownL": "#E3D3C8",
    "text": "#1A1A2E", "gray": "#666666", "pale": "#B8B8B8", "white": "#FFFFFF",
}

FONT = "Helvetica, Arial, sans-serif"
W, H = 900, 500

# Volume I stack, bottom to top, with the lectures that sit on each layer.
STACK = ["Hardware", "Frameworks", "Models", "Training", "Serving",
         "Operations", "Applications"]
ROADMAP = {
    "Hardware": [11, 12], "Frameworks": [7], "Models": [5, 6, 10],
    "Training": [8], "Serving": [13], "Operations": [14],
    "Applications": [15], "Data": [4, 9],
}


def svg(body, w=W, h=H, label=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{label}">\n'
            f'<title>{label}</title>\n{body}</svg>\n')


def text(x, y, s, size=22, weight=600, fill=None, anchor="start", extra=""):
    fill = fill or C["text"]
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
            f'dominant-baseline="central"{extra}>{s}</text>\n')


def rect(x, y, w, h, fill, stroke="none", sw=0, rx=8, opacity=1.0, extra=""):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" '
            f'stroke-width="{sw}"{extra}/>\n')


# ---------------------------------------------------------------- stack
def stack(accent, highlight=None, upto=None, data=True, roadmap=False):
    """The stack. highlight: layer name or 'Data'. upto: number of layers
    shown from the bottom (build steps); hidden layers are left empty so
    every step has the same geometry."""
    x0, y0, bw, gap = 60, 28, 640 if not roadmap else 560, 10
    bh = 48 if roadmap else 54
    if roadmap:
        y0 = 16
    out = []
    n = len(STACK)
    for i, name in enumerate(STACK):
        if upto is not None and i >= upto:
            continue
        y = y0 + (n - 1 - i) * (bh + gap)
        on = highlight == name
        dim = highlight is not None and not on
        fill_op = 1.0 if on else (0.10 if dim else 0.12 + 0.05 * (i % 3))
        out.append(rect(x0, y, bw, bh, accent, opacity=fill_op))
        col = C["white"] if on else (C["pale"] if dim else accent)
        out.append(text(x0 + 22, y + bh / 2, name, size=26,
                        weight=700 if on else 600, fill=col))
        if roadmap:
            out += chips(x0 + bw - 14, y + bh / 2, ROADMAP[name], accent, on)
    if data and (upto is None or upto > n):
        dx = x0 + bw + 14
        dh = n * bh + (n - 1) * gap
        on = highlight == "Data"
        dim = highlight is not None and not on
        out.append(rect(dx, y0, 56, dh, accent,
                        opacity=1.0 if on else (0.10 if dim else 0.18)))
        cx, cy = dx + 28, y0 + dh / 2
        col = C["white"] if on else (C["pale"] if dim else accent)
        out.append(text(cx, cy, "DATA", size=22, weight=700, fill=col,
                        anchor="middle",
                        extra=f' letter-spacing="3" transform="rotate(-90 {cx} {cy})"'))
        if roadmap:
            out += chips(dx + 28, y0 + dh + 26, ROADMAP["Data"], accent, on,
                         anchor="middle")
    width = x0 + bw + 14 + 56 + 40 if data else x0 + bw + 40
    return svg("".join(out), w=width, label="ML system stack")


def chips(xr, yc, lectures, accent, on, anchor="end"):
    """Lecture-number chips, right-aligned at xr (or centered)."""
    cw, ch, g = 50, 30, 6
    total = len(lectures) * cw + (len(lectures) - 1) * g
    x = xr - total if anchor == "end" else xr - total / 2
    out = []
    for k in lectures:
        out.append(rect(x, yc - ch / 2, cw, ch, C["white"],
                        stroke=C["white"] if on else accent, sw=1.5, rx=15,
                        opacity=1.0 if not on else 0.25))
        out.append(text(x + cw / 2, yc, f"L{k}", size=17, weight=700,
                        fill=C["white"] if on else accent, anchor="middle"))
        x += cw + g
    return out


# ---------------------------------------------------------------- course order
# Volume I lectures in semester order, grouped by the book's parts, each with
# the stack layer it teaches ("all" for lectures that span the stack).
COURSE = [
    ("Start", [(1, "Introduction", "all")]),
    ("Part I  Foundations", [(2, "System Paradigms", "all"), (3, "ML Lifecycle", "all"),
                              (4, "Data Engineering", "Data")]),
    ("Part II  Development", [(5, "Neural Computation", "Models"), (6, "Architectures", "Models"),
                               (7, "Frameworks", "Frameworks"), (8, "Training", "Training")]),
    ("Part III  Optimization", [(9, "Data Selection", "Data"), (10, "Compression", "Models"),
                                 (11, "Hardware Acceleration", "Hardware"), (12, "Benchmarking", "Hardware")]),
    ("Part IV  Deployment", [(13, "Serving", "Serving"), (14, "ML Operations", "Operations"),
                              (15, "Responsible Engineering", "Applications")]),
    ("Close", [(16, "Conclusion", "all")]),
]


def course_order(accent):
    """The semester as rows (one per part), each lecture a card with its
    number, title, and the stack layer it teaches."""
    out, y = [], 14
    rowh, cw, gap, x0 = 66, 222, 10, 196
    for part, lectures in COURSE:
        out.append(text(10, y + 26, part, size=17, weight=700, fill=C["gray"]))
        x = x0
        for n, title, layer in lectures:
            lit = layer != "all"
            out.append(rect(x, y, cw, 54, accent, opacity=0.14 if lit else 0.06,
                            stroke=accent, sw=1.2, rx=8))
            out.append(text(x + 10, y + 19, f"L{n}", size=15, weight=700, fill=accent))
            out.append(text(x + 48, y + 19, title,
                            size=13, weight=700, fill=C["text"]))
            out.append(text(x + 10, y + 40, "whole stack" if not lit else f"{layer} layer",
                            size=12, weight=600, fill=C["gray"]))
            x += cw + gap
        y += rowh
    return svg("".join(out), w=x0 + 4 * (cw + gap) + 4, h=y, label="Course order")


# ---------------------------------------------------------------- machine
def machine(accent, step=4, hl=None):
    """One accelerator node. step 1: compute array; 2: + on-chip SRAM;
    3: + HBM inside the accelerator package; 4: + host CPU, host DRAM and
    the host link. hl: 'compute' | 'sram' | 'hbm' | 'link' | 'host' adds
    the accent selection outline. Geometry is identical at every step."""
    out = []
    # accelerator package (structure)
    ax, ay, aw, ah = 330, 40, 540, 420
    if step >= 1:
        out.append(rect(ax, ay, aw, ah, C["BrownL"], C["BrownLine"], 2, rx=14,
                        opacity=0.35))
        out.append(text(ax + 20, ay + 26, "Accelerator", size=22, weight=700,
                        fill=C["BrownLine"]))
    # compute array: grid of orange cells
    cx, cy, cw, chh = 560, 110, 280, 210
    if step >= 1:
        out.append(rect(cx, cy, cw, chh, C["OrangeL"], C["OrangeLine"], 2,
                        rx=10))
        cols, rows = 6, 4
        pw, ph = (cw - 40) / cols, (chh - 70) / rows
        for r in range(rows):
            for c in range(cols):
                out.append(rect(cx + 20 + c * pw + 3, cy + 20 + r * ph + 3,
                                pw - 6, ph - 6, C["OrangeLine"], rx=3,
                                opacity=0.85))
        out.append(text(cx + cw / 2, cy + chh - 24, "Compute units", size=21,
                        weight=700, fill=C["OrangeLine"], anchor="middle"))
    # on-chip SRAM: thin blue band beside the compute array
    sx, sy, sw_, sh = 400, 110, 130, 210
    if step >= 2:
        out.append(rect(sx, sy, sw_, sh, C["BlueL"], C["BlueLine"], 2, rx=10))
        out.append(text(sx + sw_ / 2, sy + sh / 2 - 14, "On-chip", size=20,
                        weight=700, fill=C["BlueLine"], anchor="middle"))
        out.append(text(sx + sw_ / 2, sy + sh / 2 + 14, "SRAM", size=20,
                        weight=700, fill=C["BlueLine"], anchor="middle"))
    # HBM stacks below
    hx, hy, hw, hh = 400, 350, 440, 80
    if step >= 3:
        out.append(rect(hx, hy, hw, hh, C["BlueL"], C["BlueLine"], 2, rx=10))
        out.append(text(hx + 24, hy + hh / 2, "HBM", size=24, weight=700,
                        fill=C["BlueLine"]))
        for k in range(4):
            out.append(rect(hx + 110 + k * 80, hy + 14, 66, hh - 28,
                            C["BlueLine"], rx=4, opacity=0.25))
        # HBM to SRAM path
        out.append(f'<line x1="465" y1="350" x2="465" y2="320" '
                   f'stroke="{C["BlueLine"]}" stroke-width="5"/>\n')
    # host side
    if step >= 4:
        out.append(rect(30, 110, 200, 120, C["BrownL"], C["BrownLine"], 2,
                        rx=10, opacity=0.35))
        out.append(text(130, 170, "Host CPU", size=22, weight=700,
                        fill=C["BrownLine"], anchor="middle"))
        out.append(rect(30, 300, 200, 100, C["BlueL"], C["BlueLine"], 2,
                        rx=10))
        out.append(text(130, 350, "Host DRAM", size=22, weight=700,
                        fill=C["BlueLine"], anchor="middle"))
        out.append(f'<line x1="130" y1="230" x2="130" y2="300" '
                   f'stroke="{C["BlueLine"]}" stroke-width="5"/>\n')
        # host link (violet, a transfer between components)
        out.append(f'<line x1="230" y1="170" x2="330" y2="170" '
                   f'stroke="{C["VioletLine"]}" stroke-width="8"/>\n')
        out.append(text(280, 128, "Link", size=19, weight=700,
                        fill=C["VioletLine"], anchor="middle"))
    boxes = {"compute": (cx, cy, cw, chh), "sram": (sx, sy, sw_, sh),
             "hbm": (hx, hy, hw, hh), "link": (236, 156, 88, 28),
             "host": (30, 110, 200, 290)}
    if hl:
        x, y, w, h = boxes[hl]
        out.append(rect(x - 8, y - 8, w + 16, h + 16, "none", accent, 5,
                        rx=14))
    return svg("".join(out), label="Accelerator node")


# ---------------------------------------------------------------- D-A-M
def dam(accent, hl=None):
    pts = {"D": (450, 100), "A": (260, 400), "M": (640, 400)}
    names = {"D": "Data", "A": "Algorithm", "M": "Machine"}
    col = {"D": ("GreenL", "GreenLine"), "A": ("OrangeL", "OrangeLine"),
           "M": ("BlueL", "BlueLine")}
    out = []
    for a, b in (("D", "A"), ("A", "M"), ("M", "D")):
        (x1, y1), (x2, y2) = pts[a], pts[b]
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                   f'stroke="{C["VioletLine"]}" stroke-width="6"/>\n')
    for k, (x, y) in pts.items():
        fill, line = col[k]
        dim = hl is not None and hl != k
        out.append(f'<circle cx="{x}" cy="{y}" r="58" fill="#FFFFFF"/>\n')
        out.append(f'<circle cx="{x}" cy="{y}" r="58" fill="{C[fill]}" '
                   f'stroke="{C[line]}" stroke-width="4" '
                   f'opacity="{0.35 if dim else 1}"/>\n')
        if hl == k:
            out.append(f'<circle cx="{x}" cy="{y}" r="70" fill="none" '
                       f'stroke="{accent}" stroke-width="6"/>\n')
        out.append(text(x, y, k, size=48, weight=700, fill=C[line],
                        anchor="middle",
                        extra=f' opacity="{0.4 if dim else 1}"'))
        lx, align = {"D": (x + 82, "start"), "A": (x - 82, "end"),
                   "M": (x + 82, "start")}[k]
        out.append(text(lx, y, names[k], size=26, weight=700,
                        fill=C["text"], anchor=align,
                        extra=f' opacity="{0.4 if dim else 1}"'))
    return svg("".join(out), label="D-A-M triangle")


def to_pdf(path):
    pdf = path.with_suffix(".pdf")
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf), str(path)],
                       check=True)
    else:
        subprocess.run(["inkscape", "--export-type=pdf", str(path)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=1)
    args = ap.parse_args()
    if args.volume != 1:
        # Each volume has its own stack (the fleet stack, the agent stack, the
        # physical stack); add its layers and roadmap here before generating.
        raise SystemExit(f"Volume {args.volume} stack not defined yet")
    accent = ACCENT[args.volume]
    out = Path(__file__).parent / f"vol{args.volume}"
    out.mkdir(exist_ok=True)
    figs = {"course_order": course_order(accent),
            "stack": stack(accent),
            "stack_roadmap": stack(accent, roadmap=True)}
    for name in STACK + ["Data"]:
        figs[f"stack_{name.lower()}"] = stack(accent, highlight=name)
    for k in range(1, len(STACK) + 2):
        figs[f"stack_build{k}"] = stack(accent, upto=k)
    for k in range(1, 5):
        figs[f"machine_step{k}"] = machine(accent, step=k)
    for h in ("compute", "sram", "hbm", "link", "host"):
        figs[f"machine_hl_{h}"] = machine(accent, hl=h)
    figs["dam"] = dam(accent)
    for v in "DAM":
        figs[f"dam_{v}"] = dam(accent, hl=v)
    for name, s in figs.items():
        p = out / f"{name}.svg"
        p.write_text(s)
        to_pdf(p)
    print(f"wrote {len(figs)} figures to {out}")


if __name__ == "__main__":
    main()
