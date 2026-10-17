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

# Each volume's stack, bottom to top, with the lectures that sit on each
# layer. DATA: whether the stack carries Volume I's vertical data bar.
# PARTS: (label, first layer, last layer) brackets drawn left of the stack.
# COURSE: lectures in semester order, grouped by the book's parts, each with
# the stack layer it teaches ("all" for lectures that span the stack).
VOLUMES = {
    1: dict(
        stack=["Hardware", "Frameworks", "Models", "Training", "Serving",
               "Operations", "Applications"],
        roadmap={
            "Hardware": [11, 12], "Frameworks": [7], "Models": [5, 6, 10],
            "Training": [8], "Serving": [13], "Operations": [14],
            "Applications": [15], "Data": [4, 9],
        },
        data=True, parts=None,
        course=[
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
        ]),
    2: dict(
        stack=["Facility & Accelerators", "Fabric & Storage",
               "Distributed Execution", "Reliability & Control",
               "Runtime & Serving", "Operations", "Assurance",
               "Governance & Sustainability"],
        roadmap={
            "Facility & Accelerators": [2], "Fabric & Storage": [3, 4],
            "Distributed Execution": [5, 6], "Reliability & Control": [7, 8],
            "Runtime & Serving": [9, 10, 11], "Operations": [12],
            "Assurance": [13, 14], "Governance & Sustainability": [15, 16],
        },
        data=False,
        parts=[("Part I", 0, 1), ("Part II", 2, 3), ("Part III", 4, 5),
               ("Part IV", 6, 7)],
        course=[
            ("Start", [(1, "Introduction", "all")]),
            ("Part I  The Fleet", [(2, "Compute Infrastructure", "Facility & Accelerators"),
                                    (3, "Network Fabrics", "Fabric & Storage"),
                                    (4, "Data Storage", "Fabric & Storage")]),
            ("Part II  Distributed ML", [(5, "Collective Communication", "Distributed Execution"),
                                          (6, "Distributed Training", "Distributed Execution"),
                                          (7, "Fault Tolerance", "Reliability & Control"),
                                          (8, "Fleet Orchestration", "Reliability & Control")]),
            ("Part III  In Production", [(9, "Fleet Efficiency", "Runtime & Serving"),
                                          (10, "Serving at Scale", "Runtime & Serving"),
                                          (11, "Edge Intelligence", "Runtime & Serving"),
                                          (12, "ML Ops at Scale", "Operations")]),
            ("Part IV  Responsible", [(13, "Security & Privacy", "Assurance"),
                                       (14, "Robust AI", "Assurance"),
                                       (15, "Sustainable AI", "Governance & Sustainability"),
                                       (16, "Responsible AI", "Governance & Sustainability")]),
            ("Close", [(17, "Conclusion", "all")]),
        ]),
}

# Set from VOLUMES by main(); Volume I by default.
STACK, ROADMAP, COURSE = (VOLUMES[1]["stack"], VOLUMES[1]["roadmap"],
                          VOLUMES[1]["course"])
DATA, PARTS = True, None


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
def stack(accent, highlight=None, upto=None, data=None, roadmap=False):
    """The stack. highlight: layer name or 'Data'. upto: number of layers
    shown from the bottom (build steps); hidden layers are left empty so
    every step has the same geometry."""
    data = DATA if data is None else data
    x0, y0, bw, gap = 60, 28, 640 if not roadmap else 560, 10
    bh = 48 if roadmap else 54
    if roadmap:
        y0 = 16
    n = len(STACK)
    if n > 7:
        # Keep the Volume I height: shrink layers to fit more of them.
        total = 7 * bh + 6 * gap
        gap = 8
        bh = (total - (n - 1) * gap) / n
    out = []
    if PARTS:
        x0 = 120
        if not roadmap:
            bw = 600
        for label, lo, hi in PARTS:
            if upto is not None and lo >= upto:
                continue
            if upto is not None:
                hi = min(hi, upto - 1)
            ytop = y0 + (n - 1 - hi) * (bh + gap)
            ybot = y0 + (n - 1 - lo) * (bh + gap) + bh
            out.append(f'<path d="M {x0 - 10} {ytop} H {x0 - 26} V {ybot} '
                       f'H {x0 - 10}" fill="none" stroke="{C["gray"]}" '
                       f'stroke-width="2"/>\n')
            cx, cy = x0 - 46, (ytop + ybot) / 2
            out.append(text(cx, cy, label, size=17, weight=700, fill=C["gray"],
                            anchor="middle",
                            extra=f' transform="rotate(-90 {cx} {cy})"'))
    for i, name in enumerate(STACK):
        if upto is not None and i >= upto:
            continue
        y = y0 + (n - 1 - i) * (bh + gap)
        on = highlight == name
        dim = highlight is not None and not on
        fill_op = 1.0 if on else (0.10 if dim else 0.12 + 0.05 * (i % 3))
        out.append(rect(x0, y, bw, bh, accent, opacity=fill_op))
        col = C["white"] if on else (C["pale"] if dim else accent)
        out.append(text(x0 + 22, y + bh / 2, name.replace("&", "&amp;"),
                        size=26 if n <= 7 else 23,
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
def course_order(accent):
    """The semester as rows (one per part), each lecture a card with its
    number, title, and the stack layer it teaches."""
    # Drawn large enough to read on a slide: a 960-unit canvas fills the
    # slide width, so 17-unit text lands near 9 pt. Long titles wrap.
    out, y = [], 6
    rowh, ch, cw, gap, x0 = 92, 82, 200, 10, 120
    for part, lectures in COURSE:
        words = part.split("  ")
        for k, w in enumerate(words):
            yy = y + ch / 2 + (k - (len(words) - 1) / 2) * 22
            out.append(text(4, yy, w.replace("&", "&amp;"), size=17 if k == 0 else 15,
                            weight=700, fill=C["gray"]))
        x = x0
        for n, title, layer in lectures:
            lit = layer != "all"
            out.append(rect(x, y, cw, ch, accent, opacity=0.14 if lit else 0.06,
                            stroke=accent, sw=1.2, rx=8))
            short = layer.split(" & ")[0].split()[-1]
            where = "whole stack" if not lit else f"{short} layer"
            out.append(text(x + 10, y + 17, f"L{n}", size=16, weight=700, fill=accent))
            out.append(text(x + 52, y + 17, where, size=14, weight=600,
                            fill=C["gray"]))
            for k, line in enumerate(wrap(title, 17)):
                out.append(text(x + 10, y + 43 + k * 22, line.replace("&", "&amp;"),
                                size=17, weight=700, fill=C["text"]))
            x += cw + gap
        y += rowh
    return svg("".join(out), w=x0 + 4 * (cw + gap), h=y, label="Course order")


def wrap(s, n):
    """Split a title into at most two lines of about n characters."""
    if len(s) <= n:
        return [s]
    words, first = s.split(), ""
    while words and len(first + " " + words[0]) <= n + 1:
        first = (first + " " + words.pop(0)).strip()
    return [first or words.pop(0), " ".join(words)]


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


def layer_slug(name):
    """File-name form of a layer: 'Fabric & Storage' -> 'fabric'."""
    return name.split(" & ")[0].split()[0].lower()


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
    if args.volume not in VOLUMES:
        # Each volume has its own stack (the agent stack, the physical
        # stack); add its layers, roadmap, and course to VOLUMES first.
        raise SystemExit(f"Volume {args.volume} stack not defined yet")
    global STACK, ROADMAP, COURSE, DATA, PARTS
    v = VOLUMES[args.volume]
    STACK, ROADMAP, COURSE = v["stack"], v["roadmap"], v["course"]
    DATA, PARTS = v["data"], v["parts"]
    accent = ACCENT[args.volume]
    out = Path(__file__).parent / f"vol{args.volume}"
    out.mkdir(exist_ok=True)
    figs = {"course_order": course_order(accent),
            "stack": stack(accent),
            "stack_roadmap": stack(accent, roadmap=True)}
    for name in STACK + (["Data"] if DATA else []):
        figs[f"stack_{layer_slug(name)}"] = stack(accent, highlight=name)
    for k in range(1, len(STACK) + (2 if DATA else 1)):
        figs[f"stack_build{k}"] = stack(accent, upto=k)
    for k in range(1, 5):
        figs[f"machine_step{k}"] = machine(accent, step=k)
    for h in ("compute", "sram", "hbm", "link", "host"):
        figs[f"machine_hl_{h}"] = machine(accent, hl=h)
    if args.volume == 1:
        # D-A-M is Volume I's taxonomy; Volume II's C3 figure is not drawn yet.
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
