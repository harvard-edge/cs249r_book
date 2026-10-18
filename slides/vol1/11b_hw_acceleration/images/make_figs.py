#!/usr/bin/env python3
"""Figures for Lecture 11b (Hardware Acceleration, Part 2: mapping).

Every value is sourced. "HA" is the rendered chapter, Hardware Acceleration.
Values marked "derived" are computed here from the cited inputs and are
labeled derived on the slide.

Outside sources (read 2026-10-17):
  - Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like
    Performance: a Worklog" (siboehm.com/articles/22/CUDA-MMM, 2022),
    results table, FP32 SGEMM: naive 309.0, global-memory coalescing
    1,986.5, shared-memory caching 2,980.3, 1D blocktiling 8,474.7, 2D
    blocktiling 15,971.7, vectorized memory access 18,237.3, autotuning
    19,721.0, warptiling 21,779.3, cuBLAS 23,249.6 GFLOP/s. The article's
    text names the GPU (the author's A6000) and the size (4,092-square
    FP32 matrices). 23,249.6 / 309.0 = 75.2x (derived).
  - NVIDIA, Matrix Multiplication Background User's Guide: section 3.1 and
    figure 7 (tile quantization, A100-SXM4-80GB, cuBLAS 11.4, M = 27,648,
    K = 4,096, 256x128 tiles forced; the plotted values below were read from
    the published SVG plot for Lecture 11 Part 1); section 3.2 (wave
    quantization: 108 SMs, M = 2,304, N = 1,536 gives 108 tiles, 1,544
    gives 117, the tail uses 9/108 = 8.33 percent of the SMs, "GFLOPS
    roughly halve and duration roughly doubles").
  - Chen, Emer, Sze, Eyeriss, ISCA 2016, abstract: row stationary 1.4x to
    2.5x more energy efficient than other dataflows on AlexNet
    convolutional layers, at least 1.3x on fully connected layers for batch
    larger than 16.
  - Chen et al., TVM, OSDI 2018, Figure 4 text: fused operators 1.2x to 2x
    faster than unfused (NVIDIA Titan X).

Chapter sources:
  - 2x2 weight-stationary trace (HA Sec. 11.7.5): PE00..PE11 hold b00..b11;
    a00 enters row 0 in cycle 0; a01 enters row 1 one cycle later; C00
    leaves in cycle 1, C01 one cycle after. The second input row (a10, a11)
    follows one cycle behind, "staggered in the same way".
  - Fill and drain: an N x N weight-stationary array streaming B input rows
    takes B + 2N - 2 cycles (derived from one hop per cycle, HA 11.7.5);
    N = 128 as in TPU v4 (Jouppi et al. 2023, HA Sec. 11.4.4).
  - Listing 14 layer: weights 512 x 256, input 256 x 32, 4,194,304 MACs
    (HA Sec. 11.7.4). Traffic per loop order and the tiled count: derived.
  - H100: 228 KB shared memory per SM, 160 pJ per HBM byte (45 nm
    estimate), stage-2 map 56 x 56 x 64 (HA Lighthouse 11.2).
  - Equation 6 (HA Sec. 11.7.5): 2 T_M T_K + 2 T_K T_N + 4 T_M T_N bytes.
  - Mapping space per memory level: 4^5 = 1,024 placements, 7! = 5,040
    orders, 6!/3! = 120 parallelizations (HA Sec. 11.7.2, margin figure);
    all three at one level 1,024 x 5,040 x 120 = 619,315,200 (derived).
  - H100 registers (NVIDIA CUDA Programming Guide, compute capability 9.0,
    Table 30 and Table 31, read 2026-10-18): 64 K 32-bit registers per SM,
    at most 2,048 resident threads per SM, at most 255 registers per
    thread. Resident threads at r registers per thread: min(2,048,
    65,536 / r), rounded down to whole warps of 32 (derived; ignores
    per-block limits): 32 -> 2,048, 64 -> 1,024, 128 -> 512, and
    255 -> 65,536 / 255 = 257, rounded down to 8 warps = 256 threads.
  - Figures carried over from Lecture 11 Part 1 (same code, same sources):
    weight_reads (Example 11.4), tiling and cliff (Sec. 11.7.5), measured
    tiles (NVIDIA guide figure 7), layers (Napkin Math 11.4 to 11.10),
    recall_roof (Napkin Math 11.5 and 11.10).

Run from this folder: python3 make_figs.py
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent
FONT = 'font-family="Helvetica, Arial, sans-serif"'


# ================================================================ SVG helpers
def svg_text(x, y, s, size=20, weight="400", fill=C["text"], anchor="middle",
             opacity=1.0):
    """Text with simple subscripts: 'a_{00}' sets 00 as a subscript."""
    parts = re.split(r"_\{([^}]*)\}", s)
    body = []
    for i, piece in enumerate(parts):
        if not piece:
            continue
        if i % 2 == 0:
            body.append(piece)
        else:
            sub = round(size * 0.68)
            shift = round(size * 0.28)
            body.append(f'<tspan dy="{shift}" font-size="{sub}">{piece}</tspan>'
                        f'<tspan dy="-{shift}"></tspan>')
    return (f'<text x="{x}" y="{y}" {FONT} font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
            f'dominant-baseline="central" opacity="{opacity}">{"".join(body)}</text>')


def svg_open(width, height, colors=()):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" viewBox="0 0 {width} {height}">',
             f'<rect width="{width}" height="{height}" fill="white"/>', '<defs>']
    for col in colors:
        parts.append(f'<marker id="head_{col[1:]}" markerWidth="8" markerHeight="8" '
                     f'refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" '
                     f'fill="{col}"/></marker>')
    parts.append('</defs>')
    return parts


def arrow(x1, y1, x2, y2, color, width=3, opacity=1.0, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}" opacity="{opacity}"{extra} '
            f'marker-end="url(#head_{color[1:]})"/>')


def rect(x, y, w, h, fill, stroke, width=2.5, rx=6, opacity=1.0, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
            f'opacity="{opacity}"{extra}/>')


def write_svg(name, parts):
    parts.append('</svg>')
    (out / name).write_text("\n".join(parts))


def logbars(path, rows, ticks, xlim, xlabel, size, step=None, fontsize=13):
    """Horizontal bars on a log axis with plain tick labels (1, 10, 100)."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, (label, value, color_key, text) in enumerate(rows):
        y = n - 1 - i
        shown = step is None or i < step
        ax.barh(y, value, color=C[color_key], height=0.62, alpha=1.0 if shown else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=fontsize, fontweight="bold", color=C["text"])
        ax.text(value * 1.18, y, text, va="center", fontsize=fontsize,
                fontweight="bold", color=C[color_key], alpha=1.0 if shown else 0.0)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([lab for _, lab in ticks])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    sp.save(fig, path)


# ================================================================ hook: 75x
BOEHM = [  # kernel, GFLOP/s, colour, tag (the lecture segment that explains it)
    ("1 Naive", 309.0, "neutral", ""),
    ("2 Coalesced loads", 1986.5, "memory", "layout"),
    ("3 Shared-memory tiles", 2980.3, "memory", "tile staging"),
    ("4 1D register tiles", 8474.7, "memory", "bigger tile"),
    ("5 2D register tiles", 15971.7, "memory", "bigger tile"),
    ("6 Vector loads", 18237.3, "memory", "wider loads"),
    ("9 Autotuned", 19721.0, "memory", "search"),
    ("10 Warp tiles", 21779.3, "memory", "one more tier"),
    ("cuBLAS", 23249.6, "compute", ""),
]


def hook_fig(step):
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    rows = [("Naive kernel", 309.0, "neutral", "309 GFLOP/s"),
            ("cuBLAS", 23249.6, "compute", "23,250 GFLOP/s:  75× faster")]
    for i, (label, value, color_key, text) in enumerate(rows):
        y = 1 - i
        shown = i == 0 or step >= 2
        ax.barh(y, value if shown else 0, color=C[color_key], height=0.6)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, fontweight="bold", color=C["text"])
        ax.text(value * 1.15, y, text, va="center", fontsize=13,
                fontweight="bold", color=C[color_key], alpha=1.0 if shown else 0.0)
    ax.text(400, 0, "???", va="center", fontsize=16, fontweight="bold",
            color=C["danger"], alpha=1.0 if step == 1 else 0.0)
    ax.set_xscale("log")
    ax.set_xlim(100, 1.5e6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(["100", "1,000", "10,000", "100,000"])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("FP32 matrix multiply on one A6000 (GFLOP/s, log scale)")
    sp.save(fig, out / f"hook_step{step}.svg")


for k in (1, 2):
    hook_fig(k)

# ================================================================ synthesis ladder
fig, ax = plt.subplots(figsize=(5.8, 2.75))
n = len(BOEHM)
for i, (label, value, color_key, tag) in enumerate(BOEHM):
    y = n - 1 - i
    ax.barh(y, value, color=C[color_key], height=0.66)
    ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=11, fontweight="bold", color=C["text"])
    text = f"{value:,.0f}" + (f"   {tag}" if tag else "")
    ax.text(value * 1.12, y, text, va="center", fontsize=11, fontweight="bold",
            color=C[color_key])
ax.set_xscale("log")
ax.set_xlim(100, 2e6)
ax.set_ylim(-0.6, n - 0.4)
ax.set_yticks([])
ax.minorticks_off()
ax.set_xticks([100, 1000, 10000, 100000])
ax.set_xticklabels(["100", "1,000", "10,000", "100,000"])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("GFLOP/s, same multiply, same GPU (log scale)")
sp.save(fig, out / "ladder.svg")


# ================================================================ 2x2 trace
# Weight stationary, HA Sec. 11.7.5. Step 1 is the static picture from Part 1
# (weights stay, activations move right, partial sums move down); steps 2 to 5
# are cycles 0 to 3. Row 0 receives a_{i0} in cycle i; row 1 receives a_{i1}
# in cycle i + 1 (the skew). Row 1's queue shows "?" until cycle 1, the
# predict step.
def trace_svg(step):
    t = step - 2  # cycle shown; step 1 is "before cycle 0"
    cell, x0, y0 = 110, 250, 30
    cx = [x0, x0 + 180]
    cy = [y0, y0 + 180]
    parts = svg_open(900, 400, (C["memory"], C["data"]))
    sched = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)]  # (row, i, entry cycle)
    slot_x = {1: 188, 2: 113, 3: 38}
    for r in range(2):
        yc = cy[r] + cell / 2
        parts.append(arrow(218, yc, x0 - 4, yc, C["memory"]))
        parts.append(arrow(cx[0] + cell + 3, yc, cx[1] - 5, yc, C["memory"]))
        for c in range(2):
            parts.append(rect(cx[c], cy[r], cell, cell, C["compute_l"], C["compute"]))
            parts.append(svg_text(cx[c] + cell / 2, cy[r] + 26, f"b_{{{r}{c}}}", 26,
                                  "700", C["compute"]))
            parts.append(svg_text(cx[c] + cell - 6, cy[r] + cell - 13, f"PE{r}{c}",
                                  15, "400", C["neutral"], "end"))
    for c in range(2):
        xc = cx[c] + cell / 2
        parts.append(arrow(xc, cy[0] + cell + 3, xc, cy[1] - 5, C["data"]))
        parts.append(arrow(xc, cy[1] + cell + 3, xc, cy[1] + cell + 36, C["data"]))
    for r, i, entry in sched:
        yc = cy[r] + cell / 2
        col = t - entry
        hidden = r == 1 and t < 1
        if t >= 0 and 0 <= col <= 1:
            parts.append(svg_text(cx[col] + cell / 2, cy[r] + 70, f"a_{{{i}{r}}}",
                                  32, "700", C["memory"]))
        elif col < 0 and (entry - t) in slot_x:
            if hidden and i == 1:
                continue
            sx = slot_x[entry - t]
            parts.append(rect(sx - 30, yc - 26, 60, 52, "white", C["memory"], 2, 6))
            label = "?" if hidden else f"a_{{{i}{r}}}"
            parts.append(svg_text(sx, yc, label, 26, "700",
                                  C["danger"] if hidden else C["memory"]))
    for c in range(2):
        xc = cx[c] + cell / 2
        i = t - 1 - c
        if 0 <= i <= 1 and t >= 1:
            parts.append(svg_text(xc + 12, cy[0] + cell + 36,
                                  f"a_{{{i}0}}b_{{0{c}}}", 22, "700", C["data"], "start"))
            parts.append(svg_text(xc, cy[1] + cell + 60, f"C_{{{i}{c}}}", 30, "700",
                                  C["data"]))
    title = "Before cycle 0" if t < 0 else f"Cycle {t}"
    parts.append(svg_text(610, 45, title, 34, "700", C["text"], "start"))
    log = {1: "C_{00}", 2: "C_{01}, C_{10}", 3: "C_{11}"}
    for k in (1, 2, 3):
        if t >= k:
            parts.append(svg_text(610, 62 + 36 * k, f"cycle {k}:  {log[k]}", 24, "700",
                                  C["data"], "start"))
    if t >= 3:
        parts.append(svg_text(610, 228, "2 rows on a 2 × 2 array:", 22, "400",
                              C["text"], "start"))
        parts.append(svg_text(610, 258, "4 cycles = 2 + 2·2 − 2", 22, "700", C["text"],
                              "start"))
    legend = [("weights stay", C["compute"]), ("activations move right", C["memory"]),
              ("partial sums move down", C["data"])]
    for k, (text, col) in enumerate(legend):
        parts.append(svg_text(610, 310 + 30 * k, text, 22, "700", col, "start"))
    write_svg(f"trace2_step{step}.svg", parts)


for k in range(1, 6):
    trace_svg(k)


# ================================================================ 4x4 fill and drain
# B = 8 input rows through a 4 x 4 weight-stationary array. Cell (r, c) works
# on input row t - r - c (cycles r + c to r + c + 7). Derived from one hop
# per cycle (HA Sec. 11.7.5): 8 + 2*4 - 2 = 14 cycles.
def wave4_svg(step):
    t = {1: 2, 2: 6, 3: 11}[step]
    rows_in, n, cell, gap, x0, y0 = 8, 4, 72, 10, 40, 20
    parts = svg_open(900, 400)
    busy = 0
    for r in range(n):
        for c in range(n):
            x = x0 + c * (cell + gap)
            y = y0 + r * (cell + gap)
            k = t - r - c
            if 0 <= k < rows_in:
                busy += 1
                parts.append(rect(x, y, cell, cell, C["compute"], C["compute"]))
                parts.append(svg_text(x + cell / 2, y + cell / 2, str(k + 1), 30, "700",
                                      "white"))
            elif k >= rows_in:
                parts.append(rect(x, y, cell, cell, C["neutral_l"], C["neutral"]))
            else:
                parts.append(rect(x, y, cell, cell, "white", C["neutral"]))
    phase = {1: "filling", 2: "full", 3: "draining"}[step]
    px = 410
    parts.append(svg_text(px, 45, f"Cycle {t}", 36, "700", C["text"], "start"))
    parts.append(svg_text(px, 95, f"{busy} of 16 cells busy: {phase}", 28, "700",
                          C["compute"] if step == 2 else C["danger"], "start"))
    total = rows_in + 2 * n - 2
    for k in range(total):
        x = px + k * 32
        parts.append(rect(x, 140, 28, 36, C["text"] if k == t else "white", C["neutral"],
                          1.5, 3))
    parts.append(svg_text(px, 196, "cycle 0", 20, "400", C["neutral"], "start"))
    parts.append(svg_text(px + 13 * 32 + 28, 196, "13", 20, "400", C["neutral"], "end"))
    parts.append(svg_text(px, 250, "8 input rows: 8 + 2·4 − 2 = 14 cycles", 23, "700",
                          C["text"], "start"))
    parts.append(svg_text(px, 285, "busy on average: 8 / 14 = 57%", 23, "400",
                          C["text"], "start"))
    parts.append(svg_text(x0, 372, "Orange: busy (number = input row it works on).   "
                          "Gray: done.   White: not reached yet.", 20, "400",
                          C["neutral"], "start"))
    write_svg(f"wave4_step{step}.svg", parts)


for k in (1, 2, 3):
    wave4_svg(k)

# ================================================================ busy fraction, 128 x 128
N_ARRAY = 128
busy_rows = []
for b, color_key in ((1, "danger"), (128, "memory"), (1024, "memory"), (4096, "data")):
    frac = 100 * b / (b + 2 * N_ARRAY - 2)
    text = f"{frac:.1f}%" if frac < 1 else f"{frac:.0f}%"
    busy_rows.append((f"{b:,} input row" + ("" if b == 1 else "s"), frac, color_key, text))
sp.hbars(out / "busy.svg", busy_rows, "%", log=False, size=(5.8, 1.9),
         xlabel="Busy fraction, 128 × 128 array, one weight tile (%, derived)")


# ================================================================ three stationary grids
# The same 4 x 4 grid three times (HA Sec. 11.7.4, listing 15). The operand
# held in the cells is drawn inside them; streamed operands are arrows.
def stationary_svg(kind):
    n, cell, gap, x0, y0 = 4, 52, 24, 92, 64
    span = n * cell + (n - 1) * gap
    held = {"ws": ("w", C["compute"], C["compute_l"]),
            "os": ("z", C["data"], C["data_l"]),
            "is": ("x", C["memory"], C["memory_l"])}[kind]
    parts = svg_open(400, 425, (C["memory"], C["data"], C["compute"]))
    for r in range(n):
        for c in range(n):
            x = x0 + c * (cell + gap)
            y = y0 + r * (cell + gap)
            parts.append(rect(x, y, cell, cell, held[2], held[1]))
            parts.append(svg_text(x + cell / 2, y + cell / 2, held[0], 28, "700", held[1]))
    left = {"ws": ("X", C["memory"]), "os": ("W", C["compute"]),
            "is": ("W", C["compute"])}[kind]
    parts.append(svg_text(30, y0 + span / 2, left[0], 30, "700", left[1]))
    for r in range(n):
        y = y0 + r * (cell + gap) + cell / 2
        parts.append(arrow(48, y, x0 - 6, y, left[1]))
    if kind in ("ws", "is"):
        for c in range(n):
            x = x0 + c * (cell + gap) + cell / 2
            for r in range(n - 1):
                ya = y0 + r * (cell + gap) + cell + 2
                parts.append(arrow(x, ya, x, ya + gap - 8, C["data"], 1.8))
            parts.append(arrow(x, y0 + span + 3, x, y0 + span + 38, C["data"]))
        parts.append(svg_text(x0 + span / 2, y0 + span + 64, "partial sums Z move", 27,
                              "700", C["data"]))
    else:
        parts.append(svg_text(x0 + span / 2, 18, "X", 30, "700", C["memory"]))
        for c in range(n):
            x = x0 + c * (cell + gap) + cell / 2
            parts.append(arrow(x, 34, x, y0 - 6, C["memory"]))
        parts.append(svg_text(x0 + span / 2, y0 + span + 34, "each z written once", 27,
                              "700", C["data"]))
    write_svg(f"stationary_{kind}.svg", parts)


for kind in ("ws", "os", "is"):
    stationary_svg(kind)

# ================================================================ Eyeriss range bars
fig, ax = plt.subplots(figsize=(4.4, 1.55))
ax.barh(1, 2.5 - 1.4, left=1.4, color=C["data"], height=0.55)
ax.text(2.58, 1, "1.4–2.5×", va="center", fontsize=13, fontweight="bold",
        color=C["data"])
ax.annotate("", xy=(2.05, 0), xytext=(1.3, 0),
            arrowprops=dict(arrowstyle="-|>", color=C["data"], lw=3))
ax.plot([1.3], [0], "o", ms=9, color=C["data"])
ax.text(2.13, 0, "at least 1.3×", va="center", fontsize=13, fontweight="bold",
        color=C["data"])
for y, label in ((1, "Conv layers"), (0, "FC, batch > 16")):
    ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=12, fontweight="bold", color=C["text"])
ax.axvline(1, color=C["neutral"], lw=2)
ax.text(1.03, 1.62, "other dataflows = 1×", fontsize=10, color=C["neutral"],
        va="center")
ax.set_xlim(0.9, 3.4)
ax.set_ylim(-0.6, 1.9)
ax.set_yticks([])
ax.set_xticks([1, 1.5, 2, 2.5, 3])
ax.set_xticklabels(["1×", "1.5×", "2×", "2.5×", "3×"])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("Energy advantage of row stationary (AlexNet)")
sp.save(fig, out / "eyeriss.svg")


# ================================================================ one layer, tiled
# Listing 14 layer: W 512 x 256 (FP16), X 256 x 32 (FP16), Z 512 x 32 (FP32
# accumulators). Four 128-row weight tiles. Capacity against one H100 SM's
# 228 KB; traffic counts derived.
def layertile_mats_svg():
    """The three matrices with one 128-row weight tile (static picture)."""
    parts = svg_open(380, 292)
    parts.append(rect(20, 10, 150, 200, "white", C["compute"], 2.5, 2))
    for y in (110, 160):
        parts.append(f'<line x1="20" y1="{y}" x2="170" y2="{y}" stroke="{C["compute"]}" '
                     f'stroke-width="1.5" stroke-dasharray="6,4"/>')
    parts.append(rect(20, 60, 150, 50, C["compute_l"], C["compute"], 3, 2))
    parts.append(svg_text(95, 85, "128 rows", 26, "700", C["compute"]))
    parts.append(svg_text(192, 60, "×", 34, "700", C["text"]))
    parts.append(rect(215, 10, 40, 100, C["memory_l"], C["memory"], 2.5, 2))
    parts.append(svg_text(277, 60, "=", 34, "700", C["text"]))
    parts.append(rect(300, 10, 40, 200, "white", C["data"], 2.5, 2))
    parts.append(rect(300, 60, 40, 50, C["data_l"], C["data"], 3, 2))
    for x, y, name, dims, col in ((95, 240, "W", "512 × 256", C["compute"]),
                                  (235, 140, "X", "256 × 32", C["memory"]),
                                  (320, 240, "Z", "512 × 32", C["data"])):
        parts.append(svg_text(x, y, name, 28, "700", col))
        parts.append(svg_text(x, y + 34, dims, 24, "700", col))
    write_svg("layertile_mats.svg", parts)


def layertile_cost_svg(step):
    """Step 1 blank (keeps the slot), step 2 capacity, step 3 traffic."""
    parts = svg_open(440, 350)
    # capacity (step 2): 1.6 px per KB
    a2 = 1.0 if step >= 2 else 0.0
    x0, s = 10, 1.6
    parts.append(f'<g opacity="{a2}">')
    parts.append(svg_text(x0, 22, "One SM holds 228 KB", 22, "700", C["text"], "start"))
    x = x0
    for kb, col, light, label in ((64, C["compute"], C["compute_l"], "W tile"),
                                  (16, C["memory"], C["memory_l"], ""),
                                  (16, C["data"], C["data_l"], "")):
        parts.append(rect(x, 40, kb * s, 40, light, col, 2, 0))
        if label:
            parts.append(svg_text(x + kb * s / 2, 60, label, 20, "700", col))
        x += kb * s
    parts.append(svg_text(x0, 102, "64 + 16 + 16 = 96 KB: fits", 21, "700", C["data"],
                          "start"))
    parts.append(rect(x0, 120, 256 * s, 38, "white", C["danger"], 2.5, 0, 1.0, "7,4"))
    parts.append(svg_text(x0 + 10, 139, "all of W: 256 KB, too big", 21, "700",
                          C["danger"], "start"))
    xl = x0 + 228 * s
    parts.append(f'<line x1="{xl}" y1="32" x2="{xl}" y2="166" stroke="{C["text"]}" '
                 f'stroke-width="2.5"/>')
    parts.append(svg_text(xl, 182, "228 KB", 20, "700", C["text"]))
    parts.append('</g>')
    # traffic (step 3): log scale, 100 px per decade from 1e4
    a3 = 1.0 if step >= 3 else 0.0
    parts.append(f'<g opacity="{a3}">')
    parts.append(svg_text(x0, 214, "Element moves to HBM (log scale, derived)", 21,
                          "700", C["text"], "start"))

    def xpos(v):
        return x0 + (np.log10(v) - 4) * 100

    for y, v, col, label in ((244, 8404992, C["danger"], "untiled, output stationary"),
                             (312, 180224, C["data"], "four weight tiles")):
        parts.append(svg_text(x0, y, label, 20, "400", C["text"], "start"))
        parts.append(rect(x0, y + 13, xpos(v) - x0, 32, col, col, 1, 0))
        parts.append(svg_text(xpos(v) + 8, y + 29, f"{v:,}", 21, "700", col, "start"))
    parts.append(svg_text(x0 + 250, 335, "47× fewer", 28, "700", C["data"], "start"))
    parts.append('</g>')
    write_svg(f"layertile_cost_step{step}.svg", parts)


layertile_mats_svg()
for k in (1, 2, 3):
    layertile_cost_svg(k)

# ================================================================ tile intensity
# FP16 4,096-square multiply: T_M T_N / (T_M + T_N) FLOP per byte loaded for
# square-ish tiles (derived from HA Sec. 11.7.5).
tile_rows = [
    ("No reuse", 0.5, "danger", "0.5"),
    ("128 × 128 tiles", 128 * 128 / 256, "memory", "64"),
    ("256 × 128 tiles", 256 * 128 / 384, "memory", "85"),
    ("Load each once", 2 * 4096 ** 3 / (3 * 4096 ** 2 * 2), "compute", "1,365"),
]
logbars(out / "tileintensity.svg", tile_rows, [(1, "1"), (10, "10"), (100, "100"),
                                              (1000, "1,000")], (0.2, 40000),
        "FLOP per byte loaded (log, derived)", (3.9, 2.3), fontsize=12)


# ================================================================ equation 6: capacity
def capacity_fig(step):
    rows = [  # label, A, B, C (KB of 1,024 bytes)
        ("128³, single", 32, 32, 64),
        ("128³, double", 64, 64, 64),
        ("K-tile 256, double", 128, 128, 64),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    for i, (label, a, b, c) in enumerate(rows):
        y = len(rows) - 1 - i
        alpha = 1.0 if i < step - 1 else 0.0
        left = 0
        for value, key, tag in ((a, "compute", "A"), (b, "memory", "B"),
                                (c, "data", "acc")):
            ax.barh(y, value, left=left, color=C[key], height=0.6, alpha=alpha,
                    edgecolor="white", linewidth=1.5)
            ax.text(left + value / 2, y, tag, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", alpha=alpha)
            left += value
        total = a + b + c
        over = total > 228
        ax.text(335, y, f"{total} KB" + (": over" if over else ": fits"),
                va="center", fontsize=13, fontweight="bold",
                color=C["danger"] if over else C["data"], alpha=alpha)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=13, fontweight="bold", color=C["text"])
    ax.axvline(228, color=C["text"], lw=2, ls="--")
    ax.text(230, 2.55, "228 KB", fontsize=11, fontweight="bold", color=C["text"])
    ax.set_xlim(0, 450)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Bytes live at once (KB, derived from equation 6)")
    sp.save(fig, out / f"capacity_step{step}.svg")


for k in (1, 2, 3, 4):  # step 1: the axes and the budget only
    capacity_fig(k)


# ================================================================ wave quantization
# NVIDIA guide Sec. 3.2: A100, 108 SMs, one 256 x 128 tile per SM.
def wave_svg(step):
    cols, rows_n, box, gap = 12, 9, 24, 5
    parts = svg_open(900, 340)
    for panel, (x0, filled, title, colour) in enumerate((
            (40, 108, "Wave 1: 108 tiles on 108 SMs", C["compute"]),
            (500, 9, "Wave 2: 9 tiles, 99 SMs idle", C["danger"]))):
        alpha = 1.0 if panel == 0 or step >= 2 else 0.0
        parts.append(f'<g opacity="{alpha}">')
        parts.append(svg_text(x0, 22, title, 23, "700", colour, "start"))
        for k in range(cols * rows_n):
            r, c = divmod(k, cols)
            x = x0 + c * (box + gap)
            y = 50 + r * (box + gap)
            on = k < filled
            parts.append(rect(x, y, box, box, colour if on else "white",
                              colour if on else C["neutral"], 1.5, 3))
        parts.append('</g>')
    parts.append(svg_text(40, 330, "Each box is one SM of an A100", 18, "400",
                          C["neutral"], "start"))
    write_svg(f"wave_step{step}.svg", parts)


for k in (1, 2):
    wave_svg(k)

# ================================================================ mapping space
space_rows = [
    ("Placement", 1024, "neutral", "4⁵ = 1,024"),
    ("Loop order", 5040, "neutral", "7! = 5,040"),
    ("Parallelization", 120, "neutral", "6!/3! = 120"),
    ("All three, one level", 1024 * 5040 * 120, "danger", "≈ 620 million"),
]
for k in (1, 2):
    logbars(out / f"space_step{k}.svg", space_rows,
            [(100, "100"), (1e4, "10,000"), (1e6, "1 million"), (1e9, "1 billion")],
            (30, 1e13), "Choices for one layer (log scale)", (5.8, 2.0),
            step=3 if k == 1 else 4)


# ================================================================ compiler pipeline
# HA Sec. 11.8: the four stages on ResNet-50, lit one per step.
def compiler_svg(step):
    width, gap, x0 = 205, 18, 13
    parts = svg_open(900, 262, (C["neutral"],))
    names = [("Graph", "optimization"), ("Kernel", "selection"),
             ("Memory", "planning"), ("Scheduling", "")]
    for k in range(4):
        x = x0 + k * (width + gap)
        stage = k + 1
        alpha = 1.0 if stage <= step else 0.22
        lit = stage == step
        parts.append(f'<g opacity="{alpha}">')
        parts.append(rect(x, 6, width, 70, C["compute_l"] if lit else C["neutral_l"],
                          C["compute"] if lit else C["neutral"], 3 if lit else 2))
        a, b = names[k]
        if b:
            parts.append(svg_text(x + width / 2, 29, a, 22, "700", C["text"]))
            parts.append(svg_text(x + width / 2, 54, b, 22, "700", C["text"]))
        else:
            parts.append(svg_text(x + width / 2, 41, a, 22, "700", C["text"]))
        if k == 0:
            for j, op in enumerate(("Conv", "BN", "ReLU")):
                parts.append(rect(x + 8, 92 + j * 42, 72, 34, "white", C["neutral"], 2, 4))
                parts.append(svg_text(x + 44, 109 + j * 42, op, 18, "700", C["text"]))
            parts.append(arrow(x + 86, 151, x + 112, 151, C["neutral"], 2.5))
            parts.append(rect(x + 118, 92, 80, 118, C["compute_l"], C["compute"], 2.5, 4))
            for j, op in enumerate(("Conv", "+ BN", "+ ReLU")):
                parts.append(svg_text(x + 158, 120 + j * 32, op, 18, "700", C["compute"]))
            parts.append(svg_text(x + width / 2, 242, "3 kernels to 1", 19, "700",
                                  C["text"]))
        elif k == 1:
            for j, (op, chosen) in enumerate((("Tensor Core", True), ("CUDA cores", False),
                                              ("Fallback", False))):
                parts.append(rect(x + 20, 92 + j * 42, width - 40, 34,
                                  C["compute_l"] if chosen else "white",
                                  C["compute"] if chosen else C["neutral"],
                                  3 if chosen else 2, 4))
                parts.append(svg_text(x + width / 2, 109 + j * 42, op, 18, "700",
                                      C["compute"] if chosen else C["neutral"]))
            parts.append(svg_text(x + width / 2, 242, "one kernel per layer", 19, "700",
                                  C["text"]))
        elif k == 2:
            # five tensors, two buffers (illustrative)
            spans = [(0, 0, 2), (1, 1, 3), (0, 2, 4), (1, 3, 5), (0, 4, 6)]
            for j, (buf, start, end) in enumerate(spans):
                bx = x + 14 + start * 29
                by = 96 + buf * 50
                parts.append(rect(bx + 1, by, (end - start) * 29 - 3, 38,
                                  C["memory_l"], C["memory"], 2, 4))
                parts.append(svg_text(bx + (end - start) * 29 / 2, by + 19, f"T{j + 1}",
                                      18, "700", C["memory"]))
            parts.append(svg_text(x + width / 2, 204, "time →", 18, "400", C["neutral"]))
            parts.append(svg_text(x + width / 2, 242, "5 tensors, 2 buffers", 19, "700",
                                  C["text"]))
        else:
            for j, (lane, col, light) in enumerate((("copy", C["memory"], C["memory_l"]),
                                                    ("compute", C["compute"],
                                                     C["compute_l"]))):
                by = 96 + j * 50
                parts.append(svg_text(x + 2, by + 19, lane, 18, "700", col, "start"))
                for blk in range(3):
                    bx = x + 90 + blk * 34 + j * 15
                    parts.append(rect(bx, by, 28, 38, light, col, 2, 4))
            parts.append(svg_text(x + width / 2, 204, "time →", 18, "400", C["neutral"]))
            parts.append(svg_text(x + width / 2, 242, "copy overlaps compute", 19, "700",
                                  C["text"]))
        parts.append('</g>')
        if k < 3:
            parts.append(arrow(x + width + 1, 41, x + width + gap - 3, 41, C["neutral"], 2.5,
                               1.0 if stage < step else 0.22))
    write_svg(f"compiler_step{step}.svg", parts)


for k in (1, 2, 3, 4):
    compiler_svg(k)

# ================================================================ registers and resident threads
# H100 SM (compute capability 9.0, CUDA Programming Guide Tables 30 and 31):
# 65,536 registers, at most 2,048 threads, at most 255 registers per thread.
def resident_threads(regs):
    return min(2048, (65536 // regs) // 32 * 32)


reg_rows = []
for regs, color_key in ((32, "data"), (64, "memory"), (128, "memory"), (255, "danger")):
    threads = resident_threads(regs)
    share = 100 * threads / 2048
    reg_rows.append((f"{regs} registers", threads, color_key,
                     f"{threads:,} threads: {share:g}%"))
sp.hbars(out / "regs.svg", reg_rows, "threads", log=False, size=(5.8, 1.55),
         xlabel="Threads resident on one H100 SM (derived)")


# ================================================================ liveness (illustrative)
def liveness_fig(step):
    spans = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.0), gridspec_kw={"wspace": 0.45})
    for panel, ax in enumerate(axes):
        alpha = 1.0 if panel == 0 or step >= 2 else 0.0
        for j, (start, end) in enumerate(spans):
            row = 4 - j if panel == 0 else (1 if j % 2 == 0 else 0)
            ax.barh(row, end - start - 0.08, left=start + 0.04, height=0.62,
                    color=C["memory_l"], edgecolor=C["memory"], linewidth=1.5,
                    alpha=alpha)
            ax.text((start + end) / 2, row, f"T{j + 1}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=C["memory"], alpha=alpha)
        n_rows = 5 if panel == 0 else 2
        ax.set_ylim(-0.6, n_rows - 0.4)
        ax.set_xlim(0, 6)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"buf {n_rows - r}" for r in range(n_rows)][::-1]
                           if panel == 0 else ["buf 2", "buf 1"], fontsize=10)
        ax.set_xticks([])
        ax.set_xlabel("layers, in order →", fontsize=11)
        ax.set_title("One buffer each: 5" if panel == 0 else "Shared by liveness: 2",
                     fontsize=12, fontweight="bold",
                     color=C["danger"] if panel == 0 else C["data"])
        if panel == 1 and step < 2:
            ax.set_title("")
            ax.set_yticklabels(["", ""])
            ax.set_xlabel("")
            ax.tick_params(left=False, bottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    sp.save(fig, out / f"liveness_step{step}.svg")


for k in (1, 2):
    liveness_fig(k)


# ================================================================ Example 11.3: three checks
# Data type and shape: NVIDIA Matrix Multiplication Background guide, Table 1
# (FP16 multiples of 8, on A100 multiples of 64). The width 1,000 is
# illustrative. Layout: HA Sec. 11.7.6, the 3 x 3 x 3 RGB image.
def diag_svg(step):
    parts = svg_open(900, 284, (C["text"],))
    good, bad = (C["data"], C["data_l"]), (C["danger"], C["danger_l"])

    def head(x, y, text):
        parts.append(svg_text(x, y, text, 22, "700", C["text"], "start"))

    def pill(x, y, w, text, colors):
        parts.append(rect(x, y, w, 36, colors[1], colors[0], 2, 6))
        parts.append(svg_text(x + w / 2, y + 18, text, 19, "700", colors[0]))

    # 1: data type (top left)
    parts.append(f'<g opacity="{1.0 if step >= 2 else 0.0}">')
    head(10, 16, "1  Data type")
    pill(10, 36, 420, "FP16 or BF16 → Tensor Core kernel", good)
    pill(10, 80, 420, "FP32 → CUDA cores, unless TF32 is on", bad)
    parts.append('</g>')
    # 2: shape (top right), an illustrative width of 1,000 in blocks of 64
    parts.append(f'<g opacity="{1.0 if step >= 3 else 0.0}">')
    head(462, 16, "2  Shape (width 1,000, illustrative)")
    bw, x0 = 24, 462
    for k in range(16):
        x = x0 + k * (bw + 2)
        if k < 15:
            parts.append(rect(x, 36, bw, 34, C["compute_l"], C["compute"], 1.5, 3))
        else:
            parts.append(rect(x, 36, bw, 34, "white", C["danger"], 2, 3, 1.0, "4,3"))
            parts.append(rect(x, 36, bw * 40 / 64, 34, C["danger_l"], C["danger"], 1.5, 3))
    parts.append(svg_text(x0, 92, "1,000 = 15 × 64 + 40", 19, "700", C["danger"],
                          "start"))
    parts.append(svg_text(x0, 118, "A100: multiples of 64; pad to 1,024", 19, "700",
                          C["data"], "start"))
    parts.append('</g>')
    # 3: layout (bottom), the first nine values of the 3 x 3 RGB image
    parts.append(f'<g opacity="{1.0 if step >= 4 else 0.0}">')
    head(10, 156, "3  Layout, for convolutions: the first nine values in memory")
    chans = {"R": bad, "G": good, "B": (C["memory"], C["memory_l"])}
    seqs = [("NHWC", [c + str(p) for p in range(3) for c in "RGB"]),
            ("NCHW", ["R" + str(p) for p in range(9)])]
    for k, (name, seq) in enumerate(seqs):
        y = 176 + k * 54
        parts.append(svg_text(10, y + 20, name, 21, "700", C["text"], "start"))
        for j, lab in enumerate(seq):
            col = chans[lab[0]]
            x = 90 + j * 46
            parts.append(rect(x, y, 42, 40, col[1], col[0], 1.5, 3))
            parts.append(svg_text(x + 21, y + 20, lab, 18, "700", col[0]))
        parts.append(svg_text(90 + 9 * 46 + 6, y + 20, "…", 22, "700", C["neutral"],
                              "start"))
        note = ("pixel 0's channels: one read" if k == 0
                else "pixel 0's channels: 9 apart")
        parts.append(svg_text(550, y + 20, note, 20, "700",
                              C["data"] if k == 0 else C["danger"], "start"))
    parts.append('</g>')
    write_svg(f"diag_step{step}.svg", parts)


for k in (1, 2, 3, 4):
    diag_svg(k)

# ================================================================ HBM energy per boundary
MAP_BYTES = 56 * 56 * 64  # one image, INT8 (HA Lighthouse 11.2)
PJ_PER_BYTE = 160         # 45 nm estimate (HA Lighthouse 11.2)
int8_uj = 2 * MAP_BYTES * PJ_PER_BYTE / 1e6
fp16_uj = 2 * 2 * MAP_BYTES * PJ_PER_BYTE / 1e6
sp.hbars(out / "boundary.svg", [
    ("Unfused, FP16", fp16_uj, "danger", f"{fp16_uj:.0f} µJ"),
    ("Unfused, INT8", int8_uj, "memory", f"{int8_uj:.0f} µJ"),
    ("Fused", 0.0, "data", "about 0: the map stays on chip"),
], "µJ", log=False, size=sp.SHORT,
    xlabel="HBM energy per image at one layer boundary (µJ, derived)")


# ================================================================ carried over from Part 1
# The same code as Lecture 11 Part 1's make_figs.py, so the moved frames keep
# their figures and sources.
fig, ax = plt.subplots(figsize=(4.6, 0.95))
for y, (label, value, color_key, text) in enumerate((
        ("Filters spilled", 5184, "danger", "5,184 reads: 36×"),
        ("Filters held", 144, "data", "144 reads"))):
    ax.barh(y, value, color=C[color_key], height=0.62)
    ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=14, fontweight="bold", color=C["text"])
    ax.text(value + 80, y, text, va="center", fontsize=14, fontweight="bold",
            color=C[color_key])
ax.set_xlim(0, 8200)
ax.set_ylim(-0.5, 1.5)
ax.axis("off")
sp.save(fig, out / "weight_reads.svg")


def tiling_svg():
    side, x0, y0, tiles = 400, 40, 30, 32
    step_px = side / tiles
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="480" height="485" '
             'viewBox="0 0 480 485">', '<rect width="480" height="485" fill="white"/>']
    parts.append(f'<rect x="{x0}" y="{y0}" width="{side}" height="{side}" '
                 f'fill="{C["neutral_l"]}" fill-opacity="0.35" stroke="{C["text"]}" '
                 f'stroke-width="2"/>')
    for i in range(1, tiles):
        v = x0 + i * step_px
        h = y0 + i * step_px
        parts.append(f'<line x1="{v}" y1="{y0}" x2="{v}" y2="{y0 + side}" '
                     f'stroke="{C["neutral"]}" stroke-width="0.6"/>')
        parts.append(f'<line x1="{x0}" y1="{h}" x2="{x0 + side}" y2="{h}" '
                     f'stroke="{C["neutral"]}" stroke-width="0.6"/>')
    tx, ty = x0 + 5 * step_px, y0 + 4 * step_px
    parts.append(f'<rect x="{tx}" y="{ty}" width="{step_px}" height="{step_px}" '
                 f'fill="{C["compute"]}" stroke="{C["compute"]}" stroke-width="2"/>')
    parts.append(f'<line x1="{tx + step_px}" y1="{ty + step_px / 2}" x2="{tx + 70}" '
                 f'y2="{ty + 50}" stroke="{C["compute"]}" stroke-width="2"/>')
    parts.append(f'<rect x="{tx + 66}" y="{ty + 30}" width="270" height="42" rx="4" '
                 f'fill="white" stroke="{C["compute"]}" stroke-width="1.5"/>')
    parts.append(svg_text(tx + 201, ty + 51, "one 128 × 128 tile", 26, "700",
                          C["compute"]))
    parts.append(svg_text(x0 + side / 2, y0 + side + 30, "4,096 = 32 tiles of 128",
                          26, "700", C["text"]))
    parts.append('</svg>')
    return "\n".join(parts)


(out / "tiling.svg").write_text(tiling_svg())

widths = np.arange(64, 321)
util = widths / (128 * np.ceil(widths / 128))
fig, ax = plt.subplots(figsize=(3.6, 2.3))
ax.plot(widths, util * 100, color=C["text"], lw=2.5)
ax.axvspan(129, 256, color=C["danger_l"], zorder=0)
ax.plot([128], [100], "o", ms=10, color=C["text"], zorder=5)
ax.plot([129], [129 / 256 * 100], "o", ms=10, color=C["danger"], zorder=5)
ax.text(134, 104, "128: 100%", ha="left", fontsize=12, fontweight="bold",
        color=C["text"])
ax.text(134, 44, "129: 50%", ha="left", fontsize=12, fontweight="bold",
        color=C["danger"])
ax.set_xlim(64, 320)
ax.set_ylim(0, 118)
ax.set_xticks([64, 128, 192, 256, 320])
ax.set_yticks([0, 50, 100])
ax.set_xlabel("Layer width (128-wide array)")
ax.set_ylabel("Array use (%)")
ax.text(0.99, 0.02, "Modeled", transform=ax.transAxes, ha="right",
        va="bottom", fontsize=9, color=C["neutral"])
sp.save(fig, out / "cliff.svg")

# NVIDIA figure 7(a), A100: achieved TFLOP/s against N, read from the
# published SVG (for Lecture 11 Part 1).
nv_n = np.arange(112, 513, 8)
nv_tflops = np.array([
    167.0, 177.9, 189.7, 148.2, 158.6, 163.7, 174.8, 177.3, 190.1, 192.3,
    205.7, 205.4, 219.9, 220.0, 234.4, 232.2, 247.7, 244.7, 263.2, 192.3,
    199.7, 202.6, 210.8, 212.5, 219.8, 221.8, 231.6, 231.1, 240.7, 241.5,
    227.5, 250.6, 259.5, 259.1, 269.7, 213.7, 219.0, 221.5, 225.4, 227.9,
    233.7, 235.0, 242.0, 242.2, 248.9, 248.7, 254.2, 256.0, 263.1, 262.9,
    270.6])
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    shown = nv_n <= 128 if k == 1 else nv_n <= 512
    ax.plot(nv_n[shown], nv_tflops[shown], color=C["data"], lw=2.5,
            marker="o", ms=4)
    for edge in (128, 256, 384):
        ax.axvline(edge, color=C["neutral"], lw=1, ls=":")
    alpha = 1.0 if k == 2 else 0.0
    ax.annotate("136: 148 TFLOP/s\n6% more work,\n36% more time",
                (136, 148.2), xytext=(150, 22), textcoords="data",
                fontsize=11, fontweight="bold", color=C["danger"], alpha=alpha,
                arrowprops=dict(arrowstyle="->", color=C["danger"], alpha=alpha))
    ax.annotate("128: 190 TFLOP/s", (128, 189.7), xytext=(150, 284),
                textcoords="data", fontsize=11, fontweight="bold",
                color=C["data"], va="center",
                arrowprops=dict(arrowstyle="->", color=C["data"]))
    ax.text(300, 30, "???", fontsize=16, fontweight="bold", color=C["danger"],
            ha="center", alpha=1.0 if k == 1 else 0.0)
    ax.set_xlim(104, 520)
    ax.set_ylim(0, 300)
    ax.set_xticks([128, 256, 384, 512])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel("N, output width (tiles 128 wide)")
    ax.set_ylabel("TFLOP/s")
    sp.save(fig, out / f"measured_tiles_step{k}.svg")

# H100 roofline pieces (HA Napkin Math 11.4 to 11.10)
BW = 3.35  # TB/s
XLIM = (0.5, 5000)
YLIM = (1, 6000)
PEAK16 = 989


def roof_axes(size=(5.8, 2.9)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1,000"])
    ax.grid(color=C["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte, log scale)")
    ax.set_ylabel("TFLOP/s (log)")
    return fig, ax


def slope(ax, top):
    xs = np.logspace(np.log10(XLIM[0]), np.log10(top / BW), 50)
    ax.plot(xs, xs * BW, color=C["memory"], lw=3, zorder=3)


def fp16_roof(ax, where=(0.02, 0.95, "left")):
    slope(ax, PEAK16)
    ax.plot([PEAK16 / BW, XLIM[1]], [PEAK16, PEAK16], color=C["compute"], lw=3)
    ax.plot([PEAK16 / BW] * 2, [YLIM[0], 2.2], color=C["neutral"], lw=1.2, ls=":")
    ax.text(PEAK16 / BW / 1.1, 1.3, "ridge 295", color=C["neutral"], fontsize=10,
            ha="right")
    ax.text(where[0], where[1], "FP16 ceiling: 989 TFLOP/s", transform=ax.transAxes,
            color=C["compute"], fontsize=11, fontweight="bold", ha=where[2], va="top")


# recall: the stage-3 conv against the FP16 ridge
fig, ax = roof_axes(size=(5.2, 2.6))
fp16_roof(ax, (0.98, 0.42, "right"))
ax.plot([563.1], [PEAK16], "o", ms=11, color=C["compute"], mec="white", mew=1.5, zorder=5)
ax.annotate("ResNet-50 stage-3 conv: 563", (563.1, PEAK16), xytext=(0, 16),
            textcoords="offset points", ha="center", fontsize=11, fontweight="bold",
            color=C["compute"])
sp.save(fig, out / "recall_roof.svg")

# five layers, final state (thumbnail for the next-lecture slide)
layers = [
    ("GPT-2 XL decode, batch 1: 0.3%", 1.0, "danger", (8, -4), "left"),
    ("LayerNorm: 0.7%", 2.0, "danger", (8, -4), "left"),
    ("ResNet-50 FC head: 10.3%", 30.5, "memory", (8, -4), "left"),
    ("ResNet-50 stage-3 conv", 563.1, "compute", (4, -16), "left"),
    ("GPT-2 XL QKV prefill", 1046.7, "compute", (0, 14), "center"),
]
fig, ax = roof_axes()
fp16_roof(ax)
for label, x, color_key, offset, align in layers:
    y = min(PEAK16, BW * x)
    ax.plot([x], [y], "o", ms=10, color=C[color_key], mec="white", mew=1.5, zorder=5)
    ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha=align,
                va="center", fontsize=11, fontweight="bold", color=C[color_key])
sp.save(fig, out / "layers_step4.svg")

print("figures written")
