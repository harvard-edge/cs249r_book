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
  divstack_*   (Volume I) the stack drawn for the section-divider slot,
               legible at its placed size, one layer lit or all lit
  record_chain_*, thumb_record_chain
               (Volume IV) the sixteen records of the course, one lit per
               lecture, and a text-free thumbnail
  budget_bar_*  (Volume IV) the aisle stopping budget, one segment per step

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
        ],
        course_cols=[[0, 1], [2], [3], [4, 5]], course_slot=(15, 6.5),
        divstack=True),
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
        ],
        course_cols=[[0, 1], [2], [3], [4, 5]], course_slot=(15, 5.85)),
}

# Volumes III and IV use a few more keys, all optional:
#   bar          label of the vertical bar beside every layer (Volume I's
#                data=True is bar="Data"; Volume III's is "Trajectory")
#   tabs         a short tag drawn before each layer name (part numerals)
#   gap_below    the layer with a wider, labeled gap beneath it (Volume IV's
#                proposal boundary, between Brain and Nervous System)
#   envelope     (name, label, lectures): a dashed outline around every
#                layer (Volume IV's governance envelope); highlighting the
#                name lights the outline instead of a layer
#   slugs        file-name overrides for layer names
#   extra        extra highlight figures: {slug: [layer, layer]}
#   course_cols  draw the course order as columns, one list of COURSE
#                group indices per column, so a long semester stays legible
#   card_labels  the layer tag printed on course-order cards, per layer
#   course_slot  (width, height) in cm of the slide slot the course order
#                fills, so its text is sized for the slide
#   card_tags    False: no layer tag on course-order cards (Volume III,
#                where each part is one layer and its header names it)
VOLUMES[3] = dict(
    stack=["Model", "Memory", "Tools", "Runtime", "Learning",
           "Agents at Scale"],
    roadmap={"Model": [2, 3], "Memory": [4, 5, 6], "Tools": [7, 8, 9],
             "Runtime": [10, 11, 12, 13, 14], "Learning": [15, 16, 17],
             "Agents at Scale": [18, 19], "Trajectory": [1, 20]},
    data=False, bar="Trajectory", parts=None,
    tabs=["I", "II", "III", "IV", "V", "VI"],
    slugs={"Agents at Scale": "scale"},
    course=[
        ("Start", [(1, "Introduction", "all")]),
        ("Part I  The Model", [(2, "Foundation Model", "Model"),
                               (3, "Test-Time Compute", "Model")]),
        ("Part II  Agent Memory", [(4, "Context Engineering", "Memory"),
                                   (5, "KV Cache", "Memory"),
                                   (6, "Long-Term Memory", "Memory")]),
        ("Part III  Tool Use", [(7, "Tool Calling", "Tools"),
                                (8, "Sandboxes", "Tools"),
                                (9, "Agent Security", "Tools")]),
        ("Part IV  The Agent Runtime", [(10, "Agent Harness", "Runtime"),
                                        (11, "Durable Execution", "Runtime"),
                                        (12, "Failure Recovery", "Runtime"),
                                        (13, "Agent Evaluation", "Runtime"),
                                        (14, "Operating Agents", "Runtime")]),
        ("Part V  Learning", [(15, "Trajectory Curation", "Learning"),
                              (16, "Fine-Tuning", "Learning"),
                              (17, "RLVR", "Learning")]),
        ("Part VI  Agents at Scale", [(18, "Multi-Agent", "Agents at Scale"),
                                      (19, "Agent Economics", "Agents at Scale")]),
        ("Close", [(20, "Conclusion", "all")]),
    ],
    course_cols=[[0, 1, 2], [3, 4], [5, 6, 7]], course_slot=(15, 6.2),
    card_tags=False,
)
VOLUMES[4] = dict(
    stack=["World", "Boundary", "Body", "Nervous System", "Brain"],
    roadmap={"World": [5, 7], "Boundary": [1, 8], "Body": [2],
             "Nervous System": [4, 12, 13], "Brain": [3, 6, 8, 9, 10, 11]},
    data=False, parts=None,
    gap_below="Brain",
    envelope=("Governance", "Governance (Part IV)", [14, 15, 16, 17]),
    slugs={"Nervous System": "nervous"},
    extra={"boundary_brain": ["Boundary", "Brain"]},
    course=[
        ("Start", [(1, "The Causal Boundary", "Boundary")]),
        ("Part I  The Machine Anatomy", [(2, "The Physical Body", "Body"),
                                         (3, "The Cognitive Brain", "Brain"),
                                         (4, "The Nervous System", "Nervous System")]),
        ("Part II  Teaching the Machine", [(5, "Physical Data", "World"),
                                           (6, "Policy Training", "Brain"),
                                           (7, "Closed-Loop Evaluation", "World")]),
        ("Part III  Running the Machine", [(8, "Sensor Perception", "Boundary and Brain"),
                                           (9, "Spatial Memory", "Brain"),
                                           (10, "Grounded Intent", "Brain")]),
        ("", [(11, "Trajectory Planning", "Brain"),
              (12, "Safety Enforcement", "Nervous System"),
              (13, "Workload Placement", "Nervous System")]),
        ("Part IV  Governing the Machine", [(14, "Supervisory Intervention", "Governance"),
                                            (15, "Adversarial Verification", "Governance"),
                                            (16, "Deployment Release", "Governance"),
                                            (17, "Responsible Physical AI", "Governance")]),
        ("Close", [(18, "Engineering Physical AI", "all")]),
    ],
    course_cols=[[0, 1, 2], [3, 4], [5, 6]], course_slot=(15, 6.0),
    card_tags=False,
    card_labels={"Nervous System": "Nervous System",
                 "Boundary and Brain": "Boundary, Brain",
                 "Governance": "Governance"},
    records=True, budget=True,
)

# Volume IV's record chain (course plan, "The golden thread"; the book's
# chapter 1 spine figure, fig-01-spine): one record per lecture, Lecture 3
# none (its proposer limits enter the handoff record). Grid position is
# (row, column): row 0 is Part II, row 1 the main strand (Part I's handoff
# and Part III), row 2 Part IV; Part I's limit record sits above its handoff
# record so the whole chain fits seven columns at a legible size.
RECORDS = [  # (lecture, name lines, row, column)
    (2, ["limit", "record"], 0, 0),
    (4, ["handoff", "record"], 1, 0),
    (5, ["provenance", "record"], 0, 1),
    (6, ["policy", "manifest"], 0, 2),
    (7, ["evaluation", "record"], 0, 3),
    (8, ["observation", "contract"], 1, 1),
    (9, ["belief", "record"], 1, 2),
    (10, ["intent", "lease"], 1, 3),
    (11, ["trajectory", "record"], 1, 4),
    (12, ["enforcement", "record"], 1, 5),
    (13, ["placement", "record"], 1, 6),
    (14, ["authority", "record"], 2, 1),
    (15, ["fault", "records"], 2, 2),
    (16, ["release", "manifest"], 2, 3),
    (17, ["approval", "record"], 2, 4),
    (18, ["residual-claims", "register"], 2, 5),
]

# Set from VOLUMES by main(); Volume I by default.
STACK, ROADMAP, COURSE = (VOLUMES[1]["stack"], VOLUMES[1]["roadmap"],
                          VOLUMES[1]["course"])
DATA, PARTS = True, None
BAR, TABS, GAP_BELOW, ENVELOPE = "Data", None, None, None
SLUGS, EXTRA, COURSE_COLS, CARD_LABELS = {}, {}, None, {}
COURSE_SLOT, CARD_TAGS = (15, 6.3), True

# Volume IV palette roles (rules/volumes/vol4/palette.md) for the budget bar.
V4 = {"amber": "#D97706", "amber_d": "#B45309", "amber_l": "#F59E0B",
      "slate": "#334155", "slate_m": "#64748B", "gray": "#94A3B8",
      "green": "#059669", "navy": "#1F407A", "red": "#DC2626",
      "brown": "#92400E", "line": "#475569", "band": "#F1F5F9"}


# Helvetica advance widths (Adobe core-font metrics, 1/1000 em) for ASCII
# 32..126, so labels can be wrapped and fitted to a box before drawing.
_AFM_BOLD = [int(t) for t in (
    "278 333 474 556 556 889 722 278 333 333 389 584 278 333 278 278 556 556 "
    "556 556 556 556 556 556 556 556 333 333 584 584 584 611 975 722 722 722 "
    "722 667 611 778 722 278 556 722 611 833 722 778 667 778 722 667 611 722 "
    "667 944 667 667 611 333 278 333 584 556 278 556 611 556 611 556 333 611 "
    "611 278 278 556 278 889 611 611 611 611 389 556 333 611 556 778 556 556 "
    "500 389 280 389 584").split()]


def tw(s, size):
    """Width of s in bold Helvetica at size (canvas units)."""
    return sum(_AFM_BOLD[ord(c) - 32] if 32 <= ord(c) <= 126 else 600
               for c in s) * size / 1000


def fit_lines(s, size, width):
    """Greedy word wrap of s to lines no wider than width."""
    lines, cur = [], ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if cur and tw(t, size) > width:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


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
    hls = set(highlight) if isinstance(highlight, (list, tuple)) else {highlight}
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
    # Proposal gap: layers below gap_below sit pgap lower.
    gi = STACK.index(GAP_BELOW) if GAP_BELOW else None
    pgap = 46 if GAP_BELOW else 0
    epad, ehead = 16, 46
    if ENVELOPE:
        # Room for the envelope's outline and its header band.
        x0 += epad
        ytop_env = y0
        y0 += ehead

    def ylayer(i):
        y = y0 + (n - 1 - i) * (bh + gap)
        return y + pgap if gi is not None and i < gi else y

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
            out.append(text(cx, cy, label, size=26, weight=700, fill=C["gray"],
                            anchor="middle",
                            extra=f' transform="rotate(-90 {cx} {cy})"'))
    for i, name in enumerate(STACK):
        if upto is not None and i >= upto:
            continue
        y = ylayer(i)
        on = name in hls
        dim = highlight is not None and not on
        fill_op = 1.0 if on else (0.10 if dim else 0.12 + 0.05 * (i % 3))
        out.append(rect(x0, y, bw, bh, accent, opacity=fill_op))
        col = C["white"] if on else (C["pale"] if dim else accent)
        tx = x0 + 22
        if TABS:
            # The part numeral, in a fixed column before the layer name.
            out.append(text(tx, y + bh / 2, TABS[i], size=20, weight=700,
                            fill=col, extra=' opacity="0.75"'))
            tx += 58
        out.append(text(tx, y + bh / 2, name.replace("&", "&amp;"),
                        size=26 if n <= 7 else 23,
                        weight=700 if on else 600, fill=col))
        if roadmap:
            out += chips(x0 + bw - 14, y + bh / 2, ROADMAP[name], accent, on)
    if gi is not None and (upto is None or upto > n):
        # The proposal boundary: a dashed line across the gap, labeled.
        cy = ylayer(gi) + bh + (gap + pgap) / 2
        dim = highlight is not None and not hls & {STACK[gi], STACK[gi - 1]}
        col = C["pale"] if dim else C["gray"]
        out.append(f'<line x1="{x0}" y1="{cy}" x2="{x0 + bw}" y2="{cy}" '
                   f'stroke="{col}" stroke-width="2.5" '
                   f'stroke-dasharray="9,6"/>\n')
        # Name on the line at the left; what each side may do, right.
        out.append(rect(x0 + 10, cy - 13, 186, 26, C["white"], rx=4))
        out.append(text(x0 + 22, cy, "Proposal boundary", size=18, weight=700,
                        fill=col))
        out.append(text(x0 + bw - 14, cy - 14, "above: may only propose",
                        size=15, weight=600, fill=col, anchor="end"))
        out.append(text(x0 + bw - 14, cy + 14, "below: may command",
                        size=15, weight=600, fill=col, anchor="end"))
    if data and (upto is None or upto > n):
        dx = x0 + bw + 14
        dh = n * bh + (n - 1) * gap + pgap
        on = BAR in hls
        dim = highlight is not None and not on
        out.append(rect(dx, y0, 56, dh, accent,
                        opacity=1.0 if on else (0.10 if dim else 0.18)))
        cx, cy = dx + 28, y0 + dh / 2
        col = C["white"] if on else (C["pale"] if dim else accent)
        out.append(text(cx, cy, BAR.upper(), size=22, weight=700, fill=col,
                        anchor="middle",
                        extra=f' letter-spacing="3" transform="rotate(-90 {cx} {cy})"'))
        if roadmap:
            out += chips(dx + 28, y0 + dh + 26, ROADMAP[BAR], accent, on,
                         anchor="middle")
    if ENVELOPE and (upto is None or upto > n + (1 if gi is not None else 0)):
        # The governance envelope: an outline around every layer, with a
        # header band carrying its label (and, on the roadmap, its lectures).
        ename, elabel, elects = ENVELOPE
        on = ename in hls
        dim = highlight is not None and not on
        ex, ew = x0 - epad, bw + 2 * epad
        ebot = ylayer(0) + bh + epad
        col = C["pale"] if dim else accent
        if on:
            out.append(rect(ex, ytop_env, ew, ehead - 8, accent, rx=10))
            out.append(rect(ex, ytop_env, ew, ebot - ytop_env, "none",
                            accent, 4, rx=12))
        else:
            out.append(rect(ex, ytop_env, ew, ebot - ytop_env, "none", col,
                            2.5, rx=12, extra=' stroke-dasharray="10,7"'))
        hy = ytop_env + (ehead - 8) / 2
        out.append(text(x0 + 6, hy, elabel, size=22, weight=700,
                        fill=C["white"] if on else col))
        if roadmap:
            out += chips(x0 + bw - 14, hy, elects, accent, on)
    width = x0 + bw + 14 + 56 + 40 if data else x0 + bw + 40
    if ENVELOPE:
        width += epad
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


def course_columns(accent):
    """The course order at the largest type scale (1.0 to 1.3) whose
    layout still fits the slot's height, so a short semester fills it."""
    hmax = 1000 * COURSE_SLOT[1] / COURSE_SLOT[0]
    for f in (1.3, 1.25, 1.2, 1.15, 1.1, 1.05):
        try:
            s, h = course_columns_at(accent, f)
        except SystemExit:
            continue
        if h <= hmax:
            return s
    s, h = course_columns_at(accent, 1.0)
    if h > hmax:
        print(f"  course_order: {h:.0f} units tall, slot holds {hmax:.0f};"
              f" it will shrink to fit")
    return s


def course_columns_at(accent, f):
    """The semester as columns, read left to right: each column holds one
    or two parts, a header above each part and its lecture cards below.
    Drawn for its slide slot (COURSE_SLOT, in cm): a 1000-unit canvas fills
    the slot width, so the 22-unit titles land near 9.5 pt and the 18-unit
    tags near 8 pt (times f). Titles wrap to the measured column width and each card
    is as tall as its text, so a column of short titles stays compact. A
    part with an empty label continues the part in the previous column,
    aligned with its cards."""
    cw_all, colgap, gap, pgap = 1000, 14 if CARD_TAGS else 22, round(6 * f, 1), round(12 * f, 1)
    ncol = len(COURSE_COLS)
    cw = (cw_all - (ncol - 1) * colgap) / ncol
    pad, fl, ftag, ft, lead = (round(v * f, 1) for v in (7, 20, 18, 22, 25))
    hp = pad + 3                       # left and right text inset
    fh1, fh2, hlead = (round(v * f, 1) for v in (23, 19, 22))

    def tag_of(layer):
        if layer in CARD_LABELS:
            return CARD_LABELS[layer]
        if layer == "all":
            return "whole stack"
        return f"{layer.split(' & ')[0].split()[-1]} layer"

    def card(n, title, layer):
        """(L text, tag or None, title lines, height)."""
        ln = f"L{n}"
        tag = tag_of(layer) if CARD_TAGS else None
        if tag and tw(ln, fl) + 10 + tw(tag, ftag) > cw - 2 * hp:
            tag = tag.replace(" layer", "")
        if tag and tw(ln, fl) + 10 + tw(tag, ftag) > cw - 2 * hp:
            raise SystemExit(f"course_order: tag '{tag}' does not fit a card")
        if tag is None:
            # No tag line: the lecture number leads the first title line.
            lines = fit_lines(f"{ln} {title}", ft, cw - 2 * hp)
            return ln, None, lines, 2 * pad + (len(lines) - 1) * lead + ft
        lines = fit_lines(title, ft, cw - 2 * hp)
        return ln, tag, lines, 2 * pad + fl + 5 + (len(lines) - 1) * lead + ft

    def head(part):
        """Header lines: the part number, then its name, on one line when
        both fit the column."""
        words = part.split("  ")
        if len(words) > 1 and tw(words[0], fh1) + 12 + tw(words[1], fh2) <= cw:
            return [(words[0], words[1])], 2 + hlead + 6
        lines = [words[0]] + (fit_lines(words[1], fh2, cw) if len(words) > 1 else [])
        return lines, 2 + len(lines) * hlead + 6

    out, x, ymax, last_h = [], 0, 0, 0
    for col in COURSE_COLS:
        y = 2
        for gi in col:
            part, lectures = COURSE[gi]
            if part:
                lines, hh = head(part)
                for k, w in enumerate(lines):
                    yy = y + 2 + hlead / 2 + k * hlead
                    if isinstance(w, tuple):
                        out.append(text(x + 2, yy, w[0], size=fh1, weight=700,
                                        fill=C["gray"]))
                        w = w[1]
                        out.append(text(x + 2 + tw(lines[0][0], fh1) + 12, yy,
                                        w.replace("&", "&amp;"), size=fh2,
                                        weight=700, fill=C["gray"]))
                        continue
                    out.append(text(x + 2, yy, w.replace("&", "&amp;"),
                                    size=fh1 if k == 0 else fh2, weight=700,
                                    fill=C["gray"]))
                last_h = hh
            elif gi != col[0]:
                # The part continues in the same column: no header, no gap.
                hh = 0
                y -= pgap - gap
            else:
                hh = last_h
            y += hh
            for n, title, layer in lectures:
                ln, tag, lines, h = card(n, title, layer)
                lit = layer != "all"
                out.append(rect(x, y, cw, h, accent, opacity=0.14 if lit else 0.06,
                                stroke=accent, sw=1.4, rx=8))
                ty = y + pad
                if tag is not None:
                    out.append(text(x + hp, ty + fl / 2, ln, size=fl, weight=700,
                                    fill=accent))
                    out.append(text(x + hp + tw(ln, fl) + 10, ty + fl / 2,
                                    tag.replace("&", "&amp;"), size=ftag,
                                    weight=600, fill=C["gray"]))
                    ty += fl + 5
                for k, line in enumerate(lines):
                    s = line.replace("&", "&amp;")
                    if tag is None and k == 0:
                        s = (f'<tspan fill="{accent}">{ln}</tspan>'
                             + s[len(ln):])
                    out.append(text(x + hp, ty + ft / 2 + k * lead, s, size=ft,
                                    weight=700, fill=C["text"]))
                y += h + gap
            y += pgap - gap
        ymax = max(ymax, y - pgap)
        x += cw + colgap
    return (svg("".join(out), w=cw_all, h=round(ymax + 2), label="Course order"),
            ymax + 2)


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
    """The D-A-M triangle. Vertex positions and circle sizes are fixed:
    Lecture 1 trims the figure (30 22 45 22 bp, so x 40..840 and y 29..471
    stay visible) and draws arrows over it at fixed relative positions.
    The names sit 10 units clear of the highlight ring at 29 units, the
    largest size that keeps "Algorithm" and "Machine" inside that trim."""
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
            out.append(f'<circle cx="{x}" cy="{y}" r="67" fill="none" '
                       f'stroke="{accent}" stroke-width="5"/>\n')
        out.append(text(x, y, k, size=60, weight=700, fill=C[line],
                        anchor="middle",
                        extra=f' opacity="{0.4 if dim else 1}"'))
        # "Data" sits above right of its circle: Lecture 1 ends an arrow on
        # the circle's right side, level with its center.
        lx, ly, align = {"D": (x + 74, y - 48, "start"), "A": (x - 80, y, "end"),
                         "M": (x + 80, y, "start")}[k]
        out.append(text(lx, ly, names[k], size=29, weight=700,
                        fill=C["text"], anchor=align,
                        extra=f' opacity="{0.4 if dim else 1}"'))
    return svg("".join(out), label="D-A-M triangle")


# ---------------------------------------------------------------- divider stack
def divstack(accent, highlight=None):
    """The stack for the section-divider slot (Volume I). Drawn for its
    placed size, 5 cm tall in a right-hand column, so a 22-unit label lands
    near 9.5 pt. highlight: a layer name, BAR, or 'all' (every layer lit,
    for lectures that span the stack)."""
    n = len(STACK)
    x0, y0, bw, bh, gap = 4, 6, 214, 38, 6
    out = []
    for i, name in enumerate(STACK):
        y = y0 + (n - 1 - i) * (bh + gap)
        on = highlight == "all" or name == highlight
        out.append(rect(x0, y, bw, bh, accent, opacity=1.0 if on else 0.10, rx=6))
        out.append(text(x0 + 14, y + bh / 2, name.replace("&", "&amp;"), size=22,
                        weight=700, fill=C["white"] if on else "#7A7A7A"))
    w = x0 + bw
    if DATA:
        dx, dh = x0 + bw + 8, n * bh + (n - 1) * gap
        on = highlight in ("all", BAR)
        out.append(rect(dx, y0, 46, dh, accent, opacity=1.0 if on else 0.10, rx=6))
        cx, cy = dx + 23, y0 + dh / 2
        out.append(text(cx, cy, BAR.upper(), size=20, weight=700,
                        fill=C["white"] if on else "#7A7A7A", anchor="middle",
                        extra=f' letter-spacing="3" transform="rotate(-90 {cx} {cy})"'))
        w = dx + 46
    h = y0 + n * bh + (n - 1) * gap + 6
    return svg("".join(out), w=w + 4, h=h, label="ML system stack")


# ---------------------------------------------------------------- record chain
ARROW_DEFS = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
              'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
              '<path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>\n')


def record_chain(accent, lit=None, thumb=False, upto=None):
    """The sixteen records of Volume IV in the book's spine layout (fig-01-
    spine), redrawn for a full-width slide slot (about 14 by 5.2 cm). Each
    box names its record in one or two words ("record" is dropped: every box
    is one), and each column is as wide as its longest word, so the names
    land near 9.5 pt. lit: the lecture whose record is lit; earlier records
    stay normal and later ones are dimmed. thumb: shapes only, no text,
    heavier lines, for goal and recall thumbnails. upto: build step 1 to 4,
    showing Parts I to upto (hidden parts leave their space empty, so the
    geometry is the same at every step)."""
    fn, fl, fb = 22, 18, 19                    # name, lecture, band label
    X0, col0, gapx, bh, lead = 4, 162, 12, 76, 22
    names = {lec: [w for w in lines if w not in ("record", "records")]
             for lec, lines, _, _ in RECORDS}
    names = {k: [w.replace("-", " ") for w in v] for k, v in names.items()}
    ncol = 1 + max(c for *_, c in RECORDS)
    # Column widths: the widest one-line name in the column, plus padding.
    widths = [col0 - 42] + [0] * (ncol - 1)
    for lec, _, r, c in RECORDS:
        if c and lec != 18:
            widths[c] = max([widths[c]] + [tw(w, fn) * 0.93 + 14 for w in names[lec]])
    xs = [X0 + (col0 - widths[0]) / 2, X0 + col0 + 2]
    for c in range(1, ncol - 1):
        xs.append(xs[-1] + widths[c] + gapx)
    W = xs[-1] + widths[-1] + 14
    yrow = {0: 22, 1: 112, 2: 244}
    yr_py, yr_ry = 222, 230                    # the two routes into Part IV
    H = 346
    part = {}
    for lec, _, r, c in RECORDS:
        part[lec] = 1 if lec <= 4 else 2 if lec <= 7 else 3 if lec <= 13 else 4
    vis = {lec for lec in part if upto is None or part[lec] <= upto}

    def cx(c):
        return xs[c]

    def box(c, r, wide=False):
        w = widths[c]
        if wide:
            w = max(w, max(tw(s, fn) * 0.93 + 14 for s in names[18]))
        return cx(c), yrow[r], w, bh

    sw = 5 if thumb else 1.8
    asw = 4 if thumb else 1.8
    out = [] if thumb else [ARROW_DEFS]
    band = V4["band"]
    # Part bands, with their law, as in the book's figure.
    x2 = X0 + col0 + 2
    bands = [(1, X0, 0, col0 - 4, yrow[1] + bh + 8, "I · Irreversibility", 12),
             (2, x2, 0, cx(4) - gapx / 2 - x2, yrow[1] - 8, "II · Endogenous data", 12),
             (3, x2, yrow[1] - 6, W - 6 - x2, yr_py - 6 - (yrow[1] - 6),
              "III · Proposal is not permission", yrow[1] + bh + 14),
             (4, x2, yr_ry + 6, cx(5) + box(5, 2, True)[2] + gapx / 2 - x2,
              H - yr_ry - 6, "IV · Permission must be earned", yrow[2] + bh + 14)]
    for p, x, y, w, h, label, ly in bands:
        if upto is not None and p > upto:
            continue
        out.append(rect(x, y, w, h, band, rx=8))
        if not thumb:
            out.append(text(x + 8, ly, label, size=fb, weight=700, fill=V4["line"]))
    # Connections first, so boxes sit on top of them.
    mark = "" if thumb else ' marker-end="url(#ah)"'

    def path(ends, d, dash=False):
        if not set(ends) <= vis:
            return
        da = ' stroke-dasharray="6,5"' if dash else ""
        out.append(f'<path d="{d}" fill="none" stroke="{V4["line"]}" '
                   f'stroke-width="{asw}"{da}{mark}/>\n')

    pos = {lec: (r, c) for lec, _, r, c in RECORDS}

    def mid_right(lec):
        r, c = pos[lec]
        x, y, w, h = box(c, r)
        return x + w, y + h / 2

    def left(lec):
        return cx(pos[lec][1])

    def mid(c):
        return cx(c) + widths[c] / 2

    for row in ([5, 6, 7], [8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18]):
        for a, b in zip(row, row[1:]):
            x1, y1 = mid_right(a)
            path((a, b), f"M {x1} {y1} H {left(b) - 2}")
    lx = mid(0)
    path((2, 4), f"M {lx} {yrow[0] + bh} V {yrow[1] - 2}")             # limit -> handoff
    hx, hy = mid_right(4)
    fx = (hx + left(8)) / 2
    path((4, 8), f"M {hx} {hy} H {left(8) - 2}")                        # handoff -> observation
    path((4, 5), f"M {hx} {hy} H {fx} V {yrow[0] + bh / 2} H {left(5) - 2}")   # handoff -> provenance
    ex, ey = mid_right(7)
    path((7, 12), f"M {ex} {ey} H {mid(5)} V {yrow[1] - 2}")            # evaluation -> enforcement
    path((13, 14), f"M {mid(6)} {yrow[1] + bh} V {yr_py} H {mid(1)} V {yrow[2] - 2}")  # placement -> authority
    path((7, 16), f"M {mid(3)} {yrow[0]} V 6 H {W - 7} V {yr_ry} H {mid(3) + 10} "
         f"V {yrow[2] - 2}")                                            # evaluation -> release
    if not thumb and 4 in vis:
        # Lecture 3 has no record: its proposer limits enter the handoff record.
        path((4,), f"M {lx} {yrow[1] + bh + 44} V {yrow[1] + bh + 2}", dash=True)
        out.append(text(lx, yrow[1] + bh + 58, "L3: proposer", size=fl, weight=600,
                        fill=C["gray"], anchor="middle"))
        out.append(text(lx, yrow[1] + bh + 79, "limits", size=fl, weight=600,
                        fill=C["gray"], anchor="middle"))
    for lec, _, r, c in RECORDS:
        if lec not in vis:
            continue
        x, y, w, h = box(c, r, wide=lec == 18)
        if thumb:
            out.append(rect(x, y, w, h, accent, accent, sw, rx=6, opacity=0.16))
            continue
        on = lit == lec
        dim = lit is not None and lec > lit
        stroke = accent if on else ("#CBD5E1" if dim else V4["line"])
        out.append(rect(x, y, w, h, accent if on else C["white"], stroke, sw, rx=6))
        col = C["white"] if on else ("#A3A3A3" if dim else C["text"])
        lines = names[lec]
        y0 = y + (15 if len(lines) == 2 else 25)
        for k, s in enumerate(lines):
            out.append(text(x + w / 2, y0 + lead * k, s, size=fn, weight=500,
                            fill=col, anchor="middle"))
        out.append(text(x + w / 2, y + h - 11, f"L{lec}", size=fl, weight=700,
                        fill=C["white"] if on else ("#BDBDBD" if dim else C["gray"]),
                        anchor="middle"))
    # Six units of headroom above the band labels.
    body = f'<g transform="translate(0,6)">{"".join(out)}</g>'
    return svg(body, w=round(W), h=H + 6, label="Record chain")


# ---------------------------------------------------------------- budget bar
def budget_numbers():
    """The aisle stopping budget's terms, read from MLSysIM (Embodied.Scenario.
    WarehouseAisle and the running machine), the source the chapters print
    from. Returns the inputs and one function per segment: mm at speed v."""
    import sys
    root = Path(__file__).resolve().parents[3] / "mlsysim"
    sys.path.insert(0, str(root))
    try:
        from mlsysim import Embodied
        from mlsysim.physics import calc_c2_stop_suffix
    except ImportError as e:
        raise SystemExit(f"Volume IV budget bar reads MLSysIM, which needs numpy and pint "
                         f"({e}). Run with a Python that has them, for example "
                         f"make kit V=4 PY=/path/to/.venv/bin/python")
    A = Embodied.Scenario.WarehouseAisle
    RM = Embodied.MobileManipulator.WarehouseMobileManipulator

    def mag(q, unit):
        return q.to(unit).magnitude

    ms, m = "millisecond", "meter"
    t_age = mag(A.perception_age, ms)                              # 51.6 ms
    t_lease = mag(A.t_lease + A.t_tick + A.t_bus, ms)              # 62 ms
    t_onset = mag(A.t_brake_onset, ms)                             # 20 ms
    tau = mag(A.tau_delay, ms)                                     # 133.6 ms
    assert abs(tau - (t_age + t_lease + t_onset)) < 1e-9
    a = mag(A.a_brake, "meter / second**2")                        # 2 m/s^2
    j = mag(calc_c2_stop_suffix(RM.base.max_velocity, A.a_brake)["peak_jerk"],
            "meter / second**3")                                   # 7.11 m/s^3
    n = dict(
        v_drive=mag(RM.base.max_velocity, "meter / second"),       # 1.5 m/s
        v_aisle=mag(A.v_aisle, "meter / second"),                  # 1.3 m/s
        clear=mag(A.d_clear, m) * 1e3,                             # 1,100 mm
        tau=tau, t_age=t_age, t_lease=t_lease, t_onset=t_onset,
        t_da=mag(A.t_dispatch_to_admission, ms),                   # 63.3 ms
        t_takeover=mag(A.t_takeover_in_loop, ms),                  # 605 ms
        a=a, j=j,
        allow=mag(A.delta_loc + A.delta_margin, m) * 1e3,          # 150 mm
        track=mag(A.eps_track, m) * 1e3,                           # 40 mm
    )
    return n


# The fixed segment order, left to right, is the order the course adds the
# terms, so no segment ever moves: (key, label, lecture, mm at speed v, color).
# Lease path and observation age are carved out of the pre-brake delay
# (its 133.6 ms already contains them, Napkin Math 1.1), so they add a named
# segment without adding distance; every other step adds distance.
def budget_segments(n):
    return [
        ("delay", "delay v·τ", 1, lambda v: v * n["tau"], V4["amber"]),
        ("braking", "braking v²/2a", 1, lambda v: v * v / (2 * n["a"]) * 1e3, V4["slate"]),
        ("allow", "localization + clearance", 2, lambda v: n["allow"], V4["gray"]),
        ("lease", "lease path", 4, lambda v: v * n["t_lease"], V4["green"]),
        ("age", "observation age", 8, lambda v: v * n["t_age"], "#007A87"),
        ("jerk", "jerk ramps", 11, lambda v: v * n["a"] / (2 * n["j"]) * 1e3, V4["slate_m"]),
        ("track", "tracking", 12, lambda v: n["track"], "#B6C2D1"),
        ("da", "dispatch to admission", 13, lambda v: v * n["t_da"], V4["navy"]),
        ("tko", "takeover", 14, lambda v: v * n["t_takeover"], V4["brown"]),
    ]


def fmt_mm(x):
    return f"{x:,.1f}"


def budget_bar(accent, n, step, v, speed_label):
    """The aisle stopping budget as one horizontal bar at speed v, with the
    first `step` segments of budget_segments(). Every step has the same axis,
    and each segment's label has a fixed place, so nothing moves between
    steps. The pieces of the delay are too narrow for a leader each, so a
    bracket under the delay block groups them and a swatch legend below it
    names them in bar order; later terms are labeled below on the right,
    each on its own row with a leader."""
    segs = budget_segments(n)[:step]
    xl, xr, mmax = 20, 860, 1300.0
    s = (xr - xl) / mmax

    def X(mm):
        return xl + min(mm, mmax) * s

    fs = 20
    by, bh = 92, 48
    rows = {"B1": 168, "B2": 196, "B3": 224, "A1": 70, "A2": 44}
    out = [ARROW_DEFS.replace('fill="#475569"', f'fill="{V4["brown"]}"')]
    keys = [k for k, *_ in segs]
    mm = {k: f(v) for k, _, _, f, _ in segs}
    color = {k: c for k, _, _, _, c in segs}
    color["onset"] = V4["amber"]
    # Distance spans along the bar. Lease path and observation age are cut
    # out of the delay block in time order: age, lease path, brake onset.
    spans, x = {}, 0.0
    for k in keys:
        if k in ("lease", "age"):
            continue
        spans[k] = (x, x + mm[k])
        x += mm[k]
    total = x
    d0, d1 = spans["delay"]
    age = v * n["t_age"]
    if "age" in keys:
        spans["age"] = (d0, d0 + age)
    if "lease" in keys:
        spans["lease"] = (d0 + age, d0 + age + mm["lease"])
    if "age" in keys and "lease" in keys:
        spans["onset"] = (d0 + age + mm["lease"], d1)
    for k, (a, b) in spans.items():
        out.append(rect(X(a), by, X(b) - X(a), bh, color[k], C["white"],
                        0 if k == "delay" else 1.5, rx=0))
    if "tko" in spans and spans["tko"][1] > mmax:
        # The takeover runs far past the axis: an arrow off the right edge.
        out.append(f'<line x1="{xr - 2}" y1="{by + bh / 2}" x2="{xr + 24}" y2="{by + bh / 2}" '
                   f'stroke="{V4["brown"]}" stroke-width="6" marker-end="url(#ah)"/>\n')

    def lab(t, x, y, col, anchor="start"):
        tc = C["text"] if col in ("#B6C2D1", V4["gray"]) else col
        out.append(text(x, y, t, size=fs, weight=700, fill=tc, anchor=anchor))
        return tc


    texts = {
        "age": f"observation age {fmt_mm(v * n['t_age'])}",
        "lease": f"lease path {fmt_mm(mm.get('lease', 0))}",
        "onset": f"brake onset {fmt_mm(v * n['t_onset'])}",
        "jerk": f"jerk ramps {fmt_mm(mm.get('jerk', 0))}",
        "track": f"tracking {fmt_mm(mm.get('track', 0))}",
        "da": f"dispatch to admission {fmt_mm(mm.get('da', 0))}",
    }
    # The delay's pieces, in bar order (age, lease path, brake onset): a
    # bracket under the delay block, then one legend row per piece.
    legend = {"age": "B1", "lease": "B2", "onset": "B3"}
    if any(k in spans for k in legend):
        yb = by + bh + 7
        out.append(f'<path d="M {X(d0) + 1} {yb - 5} V {yb} H {X(d1) - 1} V {yb - 5}" '
                   f'fill="none" stroke="{C["text"]}" stroke-width="1.5"/>\n')
    for k, row in legend.items():
        if k not in spans:
            continue
        y = rows[row]
        out.append(rect(xl, y - 8, 16, 16, color[k], rx=2))
        lab(texts[k], xl + 24, y, color[k])
    place = {"jerk": ("B1", "l"), "track": ("B2", "l"), "da": ("B3", "l")}
    for k, (row, side) in place.items():
        if k not in spans:
            continue
        # A vertical leader from the segment's middle down to its row; the
        # label ends just left of it.
        a, b = spans[k]
        lx, y = (X(a) + X(b)) / 2, rows[row]
        tc = lab(texts[k], lx - 6, y, color[k], "end")
        out.append(f'<line x1="{lx}" y1="{by + bh}" x2="{lx}" y2="{y + 7}" '
                   f'stroke="{tc}" stroke-width="1.5"/>\n')
    # Above the bar: the delay (whole), the allowances, and the large terms.
    lab(f"delay v·τ ({n['tau']:.1f} ms) {fmt_mm(mm['delay'])}", xl, rows["A1"], V4["amber"])
    out.append(f'<line x1="{xl}" y1="{rows["A1"] + 12}" x2="{X(d1)}" y2="{rows["A1"] + 12}" '
               f'stroke="{V4["amber"]}" stroke-width="1.5"/>\n')
    if "braking" in spans:
        a, b = spans["braking"]
        out.append(text((X(a) + X(b)) / 2, by + bh / 2, f"braking v²/2a {fmt_mm(mm['braking'])}",
                        size=fs, weight=700, fill=C["white"], anchor="middle"))
    if "allow" in spans:
        a, b = spans["allow"]
        lab(f"allowances {fmt_mm(mm['allow'])}", (X(a) + X(b)) / 2, rows["A2"], V4["gray"],
            "middle")
        out.append(f'<line x1="{(X(a) + X(b)) / 2}" y1="{rows["A2"] + 12}" '
                   f'x2="{(X(a) + X(b)) / 2}" y2="{by}" stroke="{C["text"]}" stroke-width="1.5"/>\n')
    if "tko" in spans:
        # Above, right-aligned at the axis end; the header gives its 605 ms.
        lx = xr - 30
        lab(f"takeover {fmt_mm(mm['tko'])}", xr, rows["A2"], V4["brown"], "end")
        out.append(f'<line x1="{lx}" y1="{rows["A2"] + 12}" x2="{lx}" y2="{by}" '
                   f'stroke="{V4["brown"]}" stroke-width="1.5"/>\n')
    # The clear distance: a red line across the bar, labeled above it.
    xc = X(n["clear"])
    spare = n["clear"] - total
    if spare > 0:
        out.append(f'<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" '
                   f'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" '
                   f'stroke="{V4["red"]}" stroke-width="1.5" opacity="0.45"/></pattern>\n')
        out.append(f'<rect x="{X(total)}" y="{by}" width="{xc - X(total)}" height="{bh}" '
                   f'fill="url(#hatch)" stroke="{V4["red"]}" stroke-width="1"/>\n')
        t = f"spare {fmt_mm(spare)}"
        cxs = (X(total) + xc) / 2
        if 0.62 * fs * len(t) + 12 < xc - X(total):
            out.append(rect(cxs - 0.31 * fs * len(t) - 4, by + bh / 2 - 13,
                            0.62 * fs * len(t) + 8, 26, C["white"], rx=4))
            lab(t, cxs, by + bh / 2, V4["red"], "middle")
    out.append(f'<line x1="{xc}" y1="{rows["A1"] + 12}" x2="{xc}" y2="{by + bh + 8}" stroke="{V4["red"]}" '
               f'stroke-width="3" stroke-dasharray="7,4"/>\n')
    lab(f"clear {n['clear']:,.0f} mm", xc, rows["A1"], V4["red"], "middle")
    ya = 248
    out.append(f'<line x1="{xl}" y1="{ya}" x2="{xr}" y2="{ya}" stroke="#888888" stroke-width="1.5"/>\n')
    for t in range(0, int(mmax) + 1, 200):
        out.append(f'<line x1="{X(t)}" y1="{ya}" x2="{X(t)}" y2="{ya + 6}" stroke="#888888" '
                   f'stroke-width="1.5"/>\n')
        out.append(text(X(t), ya + 20, f"{t:,}", size=18, weight=500, fill=C["text"],
                        anchor="middle"))
    head = f"{speed_label}, {v:g} m/s"
    if "tko" not in spans:
        head += f": the stop needs {fmt_mm(total)} mm, " + (
            f"{fmt_mm(spare)} spare" if spare >= 0 else f"{fmt_mm(-spare)} over")
    else:
        head += f": an in-loop takeover adds {n['t_takeover']:.0f} ms of travel"
    out.append(text(xl, 16, head, size=fs, weight=700, fill=C["text"]))
    out.append(text((xl + xr) / 2, ya + 44, "distance from detection (mm)", size=18,
                    weight=500, fill=C["gray"], anchor="middle"))
    return svg("".join(out), w=900, h=ya + 58, label="Stopping budget")


def layer_slug(name):
    """File-name form of a layer: 'Fabric & Storage' -> 'fabric'."""
    if name in SLUGS:
        return SLUGS[name]
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
    global STACK, ROADMAP, COURSE, DATA, PARTS, BAR, TABS, GAP_BELOW
    global ENVELOPE, SLUGS, EXTRA, COURSE_COLS, CARD_LABELS, COURSE_SLOT, CARD_TAGS
    v = VOLUMES[args.volume]
    STACK, ROADMAP, COURSE = v["stack"], v["roadmap"], v["course"]
    PARTS = v["parts"]
    BAR = v.get("bar", "Data" if v["data"] else None)
    DATA = BAR is not None
    TABS, GAP_BELOW, ENVELOPE = v.get("tabs"), v.get("gap_below"), v.get("envelope")
    SLUGS, EXTRA = v.get("slugs", {}), v.get("extra", {})
    COURSE_COLS, CARD_LABELS = v.get("course_cols"), v.get("card_labels", {})
    COURSE_SLOT, CARD_TAGS = v.get("course_slot", (15, 6.3)), v.get("card_tags", True)
    accent = ACCENT[args.volume]
    out = Path(__file__).parent / f"vol{args.volume}"
    out.mkdir(exist_ok=True)
    figs = {"course_order": (course_columns(accent) if COURSE_COLS
                             else course_order(accent)),
            "stack": stack(accent),
            "stack_roadmap": stack(accent, roadmap=True)}
    marks = ([BAR] if DATA else []) + ([ENVELOPE[0]] if ENVELOPE else [])
    for name in STACK + marks:
        figs[f"stack_{layer_slug(name)}"] = stack(accent, highlight=name)
    for slug, names in EXTRA.items():
        figs[f"stack_{slug}"] = stack(accent, highlight=names)
    # One build step per layer, then one per mark (bar, proposal
    # boundary, envelope) in that order.
    steps = len(STACK) + DATA + bool(GAP_BELOW) + bool(ENVELOPE)
    for k in range(1, steps + 1):
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
    vol = VOLUMES[args.volume]
    if vol.get("divstack"):
        # The section-divider stack (theme option dividerstack): one file per
        # layer and mark, plus every layer lit for lectures spanning the stack.
        for name in STACK + ([BAR] if DATA else []):
            figs[f"divstack_{layer_slug(name)}"] = divstack(accent, highlight=name)
        figs["divstack_all"] = divstack(accent, highlight="all")
    if vol.get("records"):
        figs["record_chain"] = record_chain(accent)
        figs["thumb_record_chain"] = record_chain(accent, thumb=True)
        for lec, *_ in RECORDS:
            figs[f"record_chain_l{lec:02d}"] = record_chain(accent, lit=lec)
        # A build of the whole chain, one part per step, for the slide that
        # first shows it (Lecture 1) and the course synthesis (Lecture 18).
        for k in range(1, 5):
            figs[f"record_chain_build{k}"] = record_chain(accent, upto=k)
    if vol.get("budget"):
        n = budget_numbers()
        segs = budget_segments(n)
        # At the drive limit through Lecture 12's last term (the composite
        # budget overruns there); at the aisle speed Lecture 12 chooses, every
        # step, including Lecture 13's and Lecture 14's.
        last_drive = max(k for k, s in enumerate(segs, 1) if s[2] <= 12)
        for k in range(1, last_drive + 1):
            figs[f"budget_bar_step{k}"] = budget_bar(accent, n, k, n["v_drive"], "Drive limit")
        for k in range(1, len(segs) + 1):
            figs[f"budget_bar_aisle_step{k}"] = budget_bar(accent, n, k, n["v_aisle"],
                                                           "Aisle speed")
        print("budget bar segments (mm at drive limit / aisle speed):")
        for k, (key, label, lec, f, _) in enumerate(segs, 1):
            print(f"  step {k}  L{lec:<3d} {label:<26s} {fmt_mm(f(n['v_drive'])):>8s} "
                  f"{fmt_mm(f(n['v_aisle'])):>8s}")
    for name, s in figs.items():
        p = out / f"{name}.svg"
        p.write_text(s)
        to_pdf(p)
    print(f"wrote {len(figs)} figures to {out}")


if __name__ == "__main__":
    main()
