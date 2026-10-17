#!/usr/bin/env python3
"""Data plots for Lecture 4 (Data Engineering).

Every value is sourced from the rendered chapter (Data Engineering) unless
marked illustrative:
  - False wakes: 2,592,000 one-second windows a month; 99.9 percent rejection
    gives 2,592 false wakes, 99.999 percent gives 26 (Napkin Math 4.1 and its
    margin figure). 99.99 percent (259) is the same arithmetic.
  - Listening hours: rule of three, 2,160 h of non-keyword audio to support
    one false wake a month (Sec. 4.11); a 30-day month is 720 h.
  - Label entries: a 1920x1080 mask has 2,073,600 labels; ten boxes of four
    coordinates have 40 entries (Sec. 4.6.1).
  - Label cost: $100K labeling budget against a $20 to $40 ResNet-50 run on
    8 A100s (Sec. 4.6.2).

2026-10-17: the storage latency, storage supply, format, drift-type, and K-S
figures moved to Lecture 4b (Data Engineering, Part 2), whose own
images/make_figs.py regenerates them. Their old SVG and PDF files in this
folder are no longer used by this deck.

Diagrams (leakage split, skew, pipeline synthesis) are SVG
written by this script.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
import numpy as np  # noqa: E402

out = Path(__file__).parent
C = sp.C
plt = sp.plt


def bars_lines(path, rows, step=None, lines=(), log=True, xlim=None,
               xlabel=None, size=sp.SIZE, note=None, xticks=None):
    """Horizontal bars with direct value labels and optional vertical
    reference lines. rows: (label, value, color, value_text).
    lines: (x, text, color, text_y) reference lines, all drawn every step."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, r in enumerate(rows):
        label, v, ck, txt = r[:4]
        inside = len(r) > 4 and r[4] == "in"
        y = n - 1 - i
        shown = step is None or i < step
        ax.barh(y, v if shown else 0, color=C[ck], height=0.62,
                alpha=1.0 if shown else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=13, color=C["text"],
                fontweight="bold")
        if shown and inside:
            ax.text(v / 1.25, y, txt, va="center", ha="right", fontsize=13,
                    color="white", fontweight="bold")
        elif shown:
            ax.text(v * 1.18 if log else v + (xlim[1] if xlim else v) * 0.015,
                    y, txt, va="center", fontsize=13, color=C[ck],
                    fontweight="bold")
    for x, text, ck, ty in lines:
        ax.axvline(x, color=C[ck], lw=2, ls="--")
        ax.text(x * (1.12 if log else 1.0), ty, text, color=C[ck], fontsize=11,
                fontweight="bold", va="center",
                transform=ax.get_xaxis_transform())
    ax.set_yticks([])
    if log:
        ax.set_xscale("log")
        ax.minorticks_off()
    if xlim:
        ax.set_xlim(*xlim)
    if xticks:
        ax.set_xticks([t for t, _ in xticks])
        ax.set_xticklabels([s for _, s in xticks])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if note:
        ax.text(1.0, -0.42, note, transform=ax.transAxes, ha="right",
                fontsize=9, color=C["neutral"])
    sp.save(fig, path)


def steps(stem, rows, **kw):
    for k in range(1, len(rows) + 1):
        bars_lines(out / f"{stem}_step{k}.svg", rows, step=k, **kw)


# 1. False wakes a month at three per-window rejection rates
steps("false_wakes", [
    ("99.9% rejected", 2592, "danger", "2,592 a month"),
    ("99.99% rejected", 259, "danger", "259 a month"),
    ("99.999% rejected", 26, "danger", "26 a month"),
], lines=[(1, "Budget: 1 a month", "text", 1.04)],
    xlim=(0.3, 3e5), xlabel="Expected false wakes per month (log scale)",
    xticks=[(1, "1"), (10, "10"), (100, "100"), (1000, "1,000"), (10000, "10,000")])

# 2. Listening needed to support the budget (rule of three)
bars_lines(out / "listen_hours.svg", [
    ("One month of listening", 720, "neutral", "720 h"),
    ("Needed, zero false wakes", 2160, "data", "2,160 h  = 248 GB at 16 kHz"),
], log=False, xlim=(0, 4200), size=sp.SHORT,
    xlabel="Hours of representative non-keyword audio")

# 3. Label entries per image by granularity (log)
steps("label_entries", [
    ("Class label", 1, "data", "1"),
    ("Ten boxes", 40, "data", "40 numbers"),
    ("Pixel mask, 1920×1080", 2073600, "data", "2.1 million"),
], xlim=(0.3, 3e8), xlabel="Label entries per image (log scale)",
    xticks=[(1, "1"), (100, "100"), (1e4, "10,000"), (1e6, "1 million")])

# 4. Labeling against one training run (log)
steps("label_cost", [
    ("Label 1M images", 100000, "data", r"\$100,000   (2 to 4 weeks)"),
    ("Train ResNet-50 once", 40, "compute", r"\$20 to \$40   (1.3 hours)"),
], xlim=(5, 2e7), size=sp.SHORT, xlabel="Cost (US dollars, log scale)",
    xticks=[(10, r"\$10"), (1000, r"\$1,000"), (1e5, r"\$100,000")])

# ---------------------------------------------------------------------------
# SVG diagrams, drawn at 900 x 500 with 20 to 24 px type
# ---------------------------------------------------------------------------
FONT = 'font-family="Helvetica, Arial, sans-serif"'
HEX = {"text": "#1A1A2E", "grid": "#CCCCCC"}


def svg(path, body, w=900, h=500):
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}">\n<rect width="{w}" height="{h}" fill="white"/>\n'
        + body + "\n</svg>\n")


def box(x, y, w, h, fill, stroke, text="", size=20, bold=False, color=None,
        dash=False, rx=8):
    d = ' stroke-dasharray="7 5"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="2.5"{d}/>')
    if text:
        lines = text.split("\n")
        y0 = y + h / 2 - (len(lines) - 1) * size * 0.6
        for j, ln in enumerate(lines):
            s += txt(x + w / 2, y0 + j * size * 1.2, ln, size, bold,
                     color or HEX["text"])
    return s


def txt(x, y, s, size=20, bold=False, color="#1A1A2E", anchor="middle"):
    b = ' font-weight="bold"' if bold else ""
    return (f'<text x="{x}" y="{y}" {FONT} font-size="{size}"{b} fill="{color}" '
            f'text-anchor="{anchor}" dominant-baseline="central">{s}</text>')


def arrow(x1, y1, x2, y2, color="#666666", w=3):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 14 * math.cos(a), y2 - 14 * math.sin(a)
    p1 = (hx + 7 * math.sin(a), hy - 7 * math.cos(a))
    p2 = (hx - 7 * math.sin(a), hy + 7 * math.cos(a))
    return (f'<line x1="{x1}" y1="{y1}" x2="{hx}" y2="{hy}" stroke="{color}" '
            f'stroke-width="{w}"/><polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} '
            f'{p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')


# Leakage: split by file against split by speaker (900 x 360)
SPK = [("A", C["memory"]), ("B", C["compute"]), ("C", C["network"])]


def clips(x, y, who, size=44, gap=9):
    s = ""
    for j, (name, col) in enumerate(who):
        s += (f'<rect x="{x + j * (size + gap)}" y="{y}" width="{size}" height="{size}" '
              f'rx="6" fill="{col}"/>')
        s += txt(x + j * (size + gap) + size / 2, y + size / 2, name, 24, True, "white")
    return s


def leak(k):
    A, B, Cc = SPK
    s = txt(205, 22, "Training split", 26, True)
    s += txt(700, 22, "Test split", 26, True)
    rows = [(55, "Split by file", [A, A, B, B, Cc, A], [A, B, Cc],
             "every test voice was heard in training", C["danger"])]
    if k >= 2:
        rows.append((195, "Split by hash of speaker key", [A, A, A, A, B, B], [Cc, Cc, Cc],
                     "every test voice is new", C["data"]))
    for y, name, train_clips, test_clips, msg, col in rows:
        s += txt(20, y + 8, name, 24, True, anchor="start")
        s += box(20, y + 28, 370, 70, "#F7F7F7", "#BBBBBB", rx=10)
        s += clips(34, y + 41, train_clips)
        s += box(560, y + 28, 280, 70, "#F7F7F7", "#BBBBBB", rx=10)
        s += clips(574, y + 41, test_clips)
        s += txt(700, y + 120, msg, 22, True, col)
    s += txt(205, 358, "Squares are clips; letters are speakers.", 20, False, "#666666")
    svg(out / f"leak_step{k}.svg", s, h=375)


for k in (1, 2):
    leak(k)


# Training-serving skew: per-batch statistics against saved statistics (900 x 380)
def skew(k):
    s = txt(20, 22, "Training", 26, True, C["data"], "start")
    s += box(20, 45, 190, 85, C["data_l"], C["data"], "Clip", 24)
    s += arrow(210, 87, 290, 87)
    s += box(290, 45, 320, 85, C["data_l"], C["data"],
             "Normalize with\ntraining mean, std", 23)
    s += arrow(610, 87, 690, 87)
    s += box(690, 45, 190, 85, C["data_l"], C["data"], "Model learns", 24)
    s += txt(20, 192, "Serving", 26, True, C["memory"], "start")
    s += box(20, 215, 190, 85, C["memory_l"], C["memory"], "Same clip", 24)
    s += arrow(210, 257, 290, 257)
    if k == 1:
        s += box(290, 215, 320, 85, C["danger_l"], C["danger"],
                 "Normalize with\nthis batch's mean, std", 23)
        s += arrow(610, 257, 690, 257)
        s += box(690, 215, 190, 85, C["danger_l"], C["danger"], "Different\ninput", 24)
        s += txt(450, 350, "Same code, different state: no error, wrong numbers.",
                 25, True, C["danger"])
    else:
        s += box(290, 215, 320, 85, C["data_l"], C["data"],
                 "Normalize with saved\ntraining mean, std", 23)
        s += arrow(610, 257, 690, 257)
        s += box(690, 215, 190, 85, C["data_l"], C["data"], "Same\ninput", 24)
        s += arrow(450, 130, 450, 215, C["data"])
        s += txt(462, 172, "persisted with the model", 21, True, C["data"], "start")
        s += txt(450, 350, "Fit on the training split once; reuse everywhere.",
                 25, True, C["data"])
    svg(out / f"skew_step{k}.svg", s, h=380)


for k in (1, 2):
    skew(k)


# Synthesis: Tay's path against the KWS pipeline with its boundaries (900 x 360)
def pipeline(k):
    s = txt(10, 20, "Tay, 2016", 25, True, C["danger"], "start")
    s += box(10, 40, 220, 64, C["danger_l"], C["danger"], "Public replies", 23)
    s += arrow(230, 72, 650, 72, C["danger"])
    s += txt(440, 54, "no boundary", 21, True, C["danger"])
    s += box(650, 40, 240, 64, C["danger_l"], C["danger"], "Model behavior", 23)
    if k >= 2:
        s += txt(10, 150, "Keyword spotter pipeline", 25, True, C["data"], "start")
        stages = [("Ingest", "contracts,\nK-S"), ("Label", "rubric,\nkappa"),
                  ("Process", "frozen\nstatistics"), ("Release", "speaker-hash\nsplits"),
                  ("Deliver", "sized to the\naccelerator"), ("Monitor", "drift,\nfalse wakes")]
        for j, (name, gate) in enumerate(stages):
            x = 10 + j * 149
            later = j >= 4   # delivery and monitoring are taught in Part 2
            col = "#999999" if later else C["data"]
            s += box(x, 172, 134, 62, "#F7F7F7" if later else C["data_l"], col, name, 23,
                     True, color="#888888" if later else None, dash=later)
            if j < len(stages) - 1:
                s += arrow(x + 134, 203, x + 149, 203, col, 2.5)
            for m, ln in enumerate(gate.split("\n")):
                s += txt(x + 67, 262 + m * 26, ln, 20, False, "#999999" if later else "#333333")
        s += txt(747, 150, "Part 2", 22, True, "#888888")
        s += txt(450, 340, "Each stage turns a silent data failure into a loud one.",
                 24, True, C["text"])
    svg(out / f"pipeline_step{k}.svg", s, h=360)


for k in (1, 2):
    pipeline(k)


# Label errors in benchmark test sets (slide variant of the book figure,
# a subset of its rows; Northcutt, Athalye, and Mueller 2021)
bars_lines(out / "label_errors.svg", [
    ("QuickDraw (images)", 10.12, "memory", "10.1%"),
    ("CIFAR-100 (images)", 5.85, "memory", "5.9%"),
    ("ImageNet val (images)", 5.83, "memory", "5.8%"),
    ("IMDB (text)", 2.90, "compute", "2.9%"),
    ("AudioSet (audio)", 1.35, "network", "1.4%"),
    ("MNIST (images)", 0.15, "memory", "0.15%"),
], log=False, xlim=(0, 13), size=(5.8, 2.8),
    xlabel="Test-set labels found to be wrong (%)")


# ---------------------------------------------------------------------------
# Data cascades (chapter figure 1, after Sambasivan et al., CHI 2021), redrawn
# at slide size from the book's TikZ: the same seven stages, the same arcs
# (start stage, end stage, cause, and curvature), the same impact and restart
# markers. Stage names are horizontal and shortened so they read on a slide.
# ---------------------------------------------------------------------------
def cascades():
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch, FancyArrowPatch
    cols = {"blue": C["memory"], "red": C["danger"], "green": C["data"],
            "orange": "#CC5500", "violet": "#7030A0"}
    px = [i * 10 / 6 for i in range(7)]
    # (from, to, cause, out angle, in angle, heavy) as in the book's figure
    arcs = [(0, 6, "red", 60, 120, True), (0, 5, "red", 60, 125, True),
            (1, 6, "blue", 60, 120, True), (1, 6, "red", 50, 125, True),
            (4, 6, "blue", 60, 125, True), (3, 6, "blue", 60, 120, True),
            (1, 6, "orange", 44, 132, False), (1, 6, "green", 38, 135, False),
            (1, 5, "orange", 30, 135, False), (1, 5, "green", 36, 130, False),
            (2, 6, "orange", 40, 135, False), (2, 5, "orange", 40, 135, False)]
    fig = plt.figure(figsize=(7.0, 2.3))
    ax = fig.add_axes([0.0, 0.22, 0.64, 0.76])
    ax.plot([0, 10], [0, 0], color="#BBBBBB", lw=2.5, zorder=1)
    for a, b, ck, o, i, heavy in arcs:
        x0, x1 = px[a], px[b]
        d = 0.3915 * (x1 - x0)
        oo, ii = np.radians(o), np.radians(i)
        verts = [(x0, 0), (x0 + d * np.cos(oo), d * np.sin(oo)),
                 (x1 + d * np.cos(ii), d * np.sin(ii)), (x1, 0)]
        ax.add_patch(PathPatch(MPath(verts, [1, 4, 4, 4]), fill=False,
                               ec=cols[ck], lw=2.6 if heavy else 1.4, zorder=2))
    for k in (5, 6):  # impact markers where the cascades land
        ax.add_patch(FancyArrowPatch((px[k] - 1.15, 1.05), (px[k] - 0.12, 0.18),
                                     arrowstyle="simple,head_width=12,head_length=10,tail_width=5",
                                     color=cols["violet"], alpha=0.55, lw=0, zorder=3))
    for x0, x1 in ((px[1], 0.8), (px[6], 9.1)):  # up, then bend back to restart
        rp = MPath([(x0, 0.12), (x0, 2.0), (x0, 2.7), (x1, 2.7)], [1, 2, 3, 3])
        ax.add_patch(FancyArrowPatch(path=rp, arrowstyle="-|>,head_width=4,head_length=7",
                                     color=cols["red"], lw=1.6, ls=(0, (4, 3)), zorder=3))
    ax.scatter(px, [0] * 7, s=28, color=C["text"], zorder=4)
    names = ["Problem\nstatement", "Collect,\nlabel", "Analyze,\nclean", "Select\nmodel",
             "Train\nmodel", "Evaluate\nmodel", "Deploy\nmodel"]
    for x, nm in zip(px, names):
        ax.text(x, -0.25, nm, ha="center", va="top", fontsize=10.5, color=C["text"])
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(-1.15, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    lg = [("blue", "Interacting with physical\nworld brittleness"),
          ("red", "Inadequate application-\ndomain expertise"),
          ("green", "Conflicting reward\nsystems"),
          ("orange", "Poor cross-organizational\ndocumentation")]
    y = 0.95
    for ck, t in lg:
        fig.text(0.655, y - 0.035, "●", color=cols[ck], fontsize=13, va="center")
        fig.text(0.683, y, t, fontsize=10.5, va="top", color=C["text"], linespacing=1.1)
        y -= 0.165
    fig.add_artist(FancyArrowPatch((0.655, y - 0.035), (0.693, y - 0.035), transform=fig.transFigure,
                                   arrowstyle="simple,head_width=9,head_length=7,tail_width=4",
                                   color=cols["violet"], alpha=0.8, lw=0))
    fig.text(0.703, y - 0.035, "Impact of the cascade", fontsize=10.5, va="center", color=C["text"])
    y -= 0.11
    fig.add_artist(FancyArrowPatch((0.655, y - 0.035), (0.693, y - 0.035), transform=fig.transFigure,
                                   arrowstyle="-|>,head_width=3,head_length=5", ls=(0, (3, 2)),
                                   color=cols["red"], lw=1.6))
    fig.text(0.703, y - 0.035, "Abandon or restart", fontsize=10.5, va="center", color=C["text"])
    sp.save(fig, out / "cascades.svg")


cascades()

print("figures written")
