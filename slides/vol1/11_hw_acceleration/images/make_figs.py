#!/usr/bin/env python3
"""Figures for Lecture 11 (Hardware Acceleration).

Every value is sourced. "HA" is the rendered chapter, Hardware Acceleration.
  - TPU v1 gains: 15-30x throughput, 30-80x performance per watt against
    contemporary CPUs and GPUs on six Google inference workloads
    (HA Example 11.1; Jouppi et al. 2017).
  - INT8 ResNet-50 on the H100: batch one 0.38 -> 0.30 ms (1.3x), large
    batch 5.9x, both against FP32 (Model Compression, table 13).
  - Energy per operation, 45 nm: INT8 add 0.03 pJ, FP32 add 0.9, FP32
    multiply 3.7, 8 KB SRAM read 5, DRAM read of a 32-bit value 640 pJ
    (HA figure 4, Sec. 11.2.3; Horowitz 2014).
  - Peak FP16: P100 21.2, V100 125 TFLOP/s (HA footnote 13; NVIDIA 2017).
  - H100: 3.35 TB/s; FP32 67, FP16 989, INT8 1,979 TFLOP/s or TOPS; ridges
    about 20, 295.2, 590.7 FLOP/byte (HA Napkin Math 11.6 and 11.10).
  - Layer intensities on the H100 at FP16: GPT-2 XL decode 1, LayerNorm
    2.0, ResNet-50 FC head 30.5, stage-3 3x3 conv 563.1, GPT-2 XL QKV
    prefill 1,046.7 FLOP/byte (HA Napkin Math 11.4, 11.5, 11.7, 11.8, 11.10).
  - Batch: I = 2Bn^2 / (2Bn + 2n^2 + 2Bn) for an n x n FP16 layer, n = 2048
    (HA equation 5, Napkin Math 11.9). The crossing near batch 415 is
    derived here from that equation.
  - INT8 and the ridge: stage-3 3x3 563 / 1,126 / 569, stage-2 3x3 287 / 574,
    stage-2 1x1 51 / 102 FLOP/byte (HA Napkin Math 11.6).
  - Loop order: 144 against 5,184 weight reads (HA Example 11.4).
  - Tile fringe: utilization along one dimension w / (128 ceil(w / 128))
    (HA Sec. 11.7.5).
  - Measured tile quantization on an A100: NVIDIA, "Matrix Multiplication
    Background" user guide, figure 7 (A100-SXM4-80GB, CUDA 11.2, cuBLAS
    11.4; M = 27648, K = 4096, 256 x 128 tiles forced). The values below
    were read from the published SVG plot's coordinates.
  - Compiled ResNet-50 batch one, modeled: 3.53 ms eager (174 kernels),
    0.33 ms compiled FP16 (56 kernels) (HA Sec. 11.8).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent

# ---------------------------------------------------------------- hook gains
fig, ax = plt.subplots(figsize=(3.0, 1.45))
rows = [("Throughput", 15, 30, "compute", "15–30×"),
        ("Performance per watt", 30, 80, "data", "30–80×")]
for i, (label, low, high, color_key, text) in enumerate(rows):
    y = 1 - i
    ax.barh(y, high - low, left=low, color=C[color_key], height=0.55)
    ax.text(0.9, y, label, ha="right", va="center", fontsize=12,
            fontweight="bold", color=C["text"])
    ax.text(high * 1.12, y, text, va="center", fontsize=14,
            fontweight="bold", color=C[color_key])
ax.axvline(1, color=C["neutral"], lw=2)
ax.text(1.12, -0.7, "2015 CPUs, GPUs = 1×", color=C["neutral"],
        fontsize=10, va="center")
ax.set_xscale("log")
ax.set_xlim(0.9, 400)
ax.set_ylim(-1.0, 1.6)
ax.set_yticks([])
ax.set_xticks([1, 10, 100])
ax.set_xticklabels(["1×", "10×", "100×"])
ax.spines["left"].set_visible(False)
ax.minorticks_off()
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "hook_gains.svg")

# ---------------------------------------------------------------- recall L10
sp.hbars(out / "recall_int8.svg", [
    ("Batch one", 1.3, "neutral", "1.3×   (0.38 → 0.30 ms)"),
    ("Large batch", 5.9, "compute", "5.9×"),
], "×", log=False, xlabel="INT8 against FP32, ResNet-50, same H100",
    size=sp.SHORT)

# ---------------------------------------------------------------- energy ladder
energy_rows = [
    ("INT8 add", 0.03, "compute", "0.03 pJ"),
    ("FP32 add", 0.9, "compute", "0.9 pJ"),
    ("FP32 multiply", 3.7, "compute", "3.7 pJ"),
    ("SRAM read (8 KB)", 5, "memory", "5 pJ"),
    ("DRAM read", 640, "danger", "640 pJ = 173 multiplies"),
]
for k, shown in ((1, 3), (2, 4), (3, 5)):
    sp.hbars(out / f"energy_step{k}.svg", energy_rows, "pJ", step=shown,
             xlabel="Energy per operation or access (pJ, log scale; 45 nm)",
             size=(5.8, 2.6))

# ---------------------------------------------------------------- Tensor Cores
sp.hbars(out / "tensorcore.svg", [
    ("P100 (2016)", 21.2, "neutral", "21.2"),
    ("V100 (2017)", 125, "compute", "125: about 6×"),
], "TFLOP/s", log=False, xlabel="Peak FP16 (TFLOP/s)", size=(2.6, 1.3))

# ---------------------------------------------------------------- three units
# A diagram, written as SVG: three execution units, the operations each runs,
# and (step 2) which lighthouse leans on which unit (HA Sec. 11.3.3).
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def svg_text(x, y, s, size=20, weight="400", fill=C["text"], anchor="middle"):
    return (f'<text x="{x}" y="{y}" {FONT} font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
            f'dominant-baseline="central">{s}</text>')


def units_svg(step):
    lanes = [
        ("Vector unit", "ReLU, residual add", C["memory"], C["memory_l"],
         "ResNet-50: after every conv"),
        ("Matrix unit", "convolution, projections", C["compute"], C["compute_l"],
         "Both: most of the arithmetic"),
        ("Special-function unit", "softmax exp, GELU, rsqrt", C["network"],
         C["network_l"], "GPT-2 XL: every step"),
    ]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="440" '
             'viewBox="0 0 720 440">', '<rect width="720" height="440" fill="white"/>']
    for i, (name, ops, stroke, fill, tag) in enumerate(lanes):
        y = 8 + i * 145
        parts.append(f'<rect x="8" y="{y}" width="704" height="130" rx="10" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        parts.append(svg_text(30, y + 32, name, 30, "700", stroke, "start"))
        parts.append(svg_text(30, y + 70, ops, 27, "400", C["text"], "start"))
        opacity = "1" if step >= 2 else "0"
        parts.append(f'<g opacity="{opacity}">')
        parts.append(svg_text(30, y + 106, "→ " + tag, 26, "700", stroke, "start"))
        parts.append('</g>')
    parts.append('</svg>')
    return "\n".join(parts)


for k in (1, 2):
    (out / f"units_step{k}.svg").write_text(units_svg(k))

# ---------------------------------------------------------------- systolic
# Slide variant of HA figure 6 (systolic array dataflow), drawn at slide size:
# a 4 x 4 weight-stationary grid. Step 1 weights held in the cells; step 2
# activations enter at the left edge and move right; step 3 partial sums move
# down and leave at the bottom.


def arrow(x1, y1, x2, y2, color, width=3):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}" marker-end="url(#head_{color[1:]})"/>')


def systolic_svg(step):
    cell, gap, x0, y0, n = 74, 30, 150, 50, 4
    colors = [C["memory"], C["data"], C["compute"]]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="530" '
             'viewBox="0 0 640 530">', '<rect width="640" height="530" fill="white"/>',
             '<defs>']
    for col in colors:
        parts.append(f'<marker id="head_{col[1:]}" markerWidth="8" markerHeight="8" '
                     f'refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" '
                     f'fill="{col}"/></marker>')
    parts.append('</defs>')
    for r in range(n):
        for c in range(n):
            x = x0 + c * (cell + gap)
            y = y0 + r * (cell + gap)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="6" '
                         f'fill="{C["compute_l"]}" stroke="{C["compute"]}" stroke-width="2.5"/>')
            parts.append(svg_text(x + cell / 2, y + cell / 2, f"w{r + 1}{c + 1}", 30,
                                  "700", C["compute"]))
    show = lambda k: "1" if step >= k else "0"
    # activations: enter at the left of each row, pass right between cells
    parts.append(f'<g opacity="{show(2)}">')
    for r in range(n):
        y = y0 + r * (cell + gap) + cell / 2
        parts.append(svg_text(x0 - 84, y, f"x{r + 1}", 30, "700", C["memory"]))
        parts.append(arrow(x0 - 52, y, x0 - 6, y, C["memory"]))
        for c in range(n - 1):
            xa = x0 + c * (cell + gap) + cell + 3
            parts.append(arrow(xa, y, xa + gap - 8, y, C["memory"]))
    parts.append('</g>')
    # partial sums: move down each column and leave at the bottom
    parts.append(f'<g opacity="{show(3)}">')
    bottom = y0 + n * (cell + gap) - gap
    for c in range(n):
        x = x0 + c * (cell + gap) + cell / 2
        for r in range(n - 1):
            ya = y0 + r * (cell + gap) + cell + 3
            parts.append(arrow(x, ya, x, ya + gap - 8, C["data"]))
        parts.append(arrow(x, bottom + 3, x, bottom + 34, C["data"]))
        parts.append(svg_text(x, bottom + 58, f"y{c + 1}", 30, "700", C["data"]))
    parts.append('</g>')
    parts.append('</svg>')
    return "\n".join(parts)


for k in (1, 2, 3):
    (out / f"systolic_step{k}.svg").write_text(systolic_svg(k))

# ---------------------------------------------------------------- tiling
# A 4,096-wide output on a 128-wide array: 32 x 32 = 1,024 output tiles
# (HA Sec. 11.7.5). One tile highlighted.


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

# ---------------------------------------------------------------- H100 roofline
BW = 3.35  # TB/s
XLIM = (0.5, 5000)
YLIM = (1, 6000)


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


ceilings = [  # name, peak, ridge label, colour, line style
    ("FP32", 67, "20", "neutral", "-"),
    ("FP16", 989, "295", "compute", "-"),
    ("INT8", 1979, "591", "danger", "--"),
]
for k in (1, 2, 3):
    fig, ax = roof_axes()
    slope(ax, ceilings[k - 1][1])
    ax.text(6, 6 * BW / 3.2, "3.35 TB/s", color=C["memory"], fontsize=12,
            fontweight="bold", ha="left", va="top")
    for i, (name, peak, ridge_text, color_key, style) in enumerate(ceilings):
        alpha = 1.0 if i < k else 0.0
        ridge = peak / BW
        ax.plot([ridge, XLIM[1]], [peak, peak], color=C[color_key], lw=3,
                ls=style, alpha=alpha, zorder=3)
        ax.plot([ridge, ridge], [YLIM[0], 3.2], color=C[color_key], lw=1.2,
                ls=":", alpha=alpha)
        below = name == "FP16"
        ax.text(XLIM[1] / 1.15, peak / 1.15 if below else peak * 1.2,
                f"{name} {peak:,} " + ("TOPS" if name == "INT8" else "TFLOP/s"),
                color=C[color_key], fontsize=11,
                fontweight="bold", ha="right", va="top" if below else "bottom",
                alpha=alpha)
        left = name == "FP16"
        ax.text(ridge / 1.08 if left else ridge * 1.08, 1.25,
                f"ridge\n{ridge_text}", color=C[color_key], fontsize=10,
                fontweight="bold", ha="right" if left else "left",
                va="bottom", alpha=alpha)
    sp.save(fig, out / f"roof_h100_step{k}.svg")

# ---------------------------------------------------------------- five layers
PEAK16 = 989
layers = [  # label, intensity, colour, text offset (points), alignment
    ("GPT-2 XL decode, batch 1: 0.3%", 1.0, "danger", (8, -4), "left"),
    ("LayerNorm: 0.7%", 2.0, "danger", (8, -4), "left"),
    ("ResNet-50 FC head: 10.3%", 30.5, "memory", (8, -4), "left"),
    ("ResNet-50 stage-3 conv", 563.1, "compute", (4, -16), "left"),
    ("GPT-2 XL QKV prefill", 1046.7, "compute", (0, 14), "center"),
]
reveal = [2, 3, 4, 5]
for k, shown in enumerate(reveal, start=1):
    fig, ax = roof_axes()
    slope(ax, PEAK16)
    ax.plot([PEAK16 / BW, XLIM[1]], [PEAK16, PEAK16], color=C["compute"], lw=3)
    ax.plot([PEAK16 / BW] * 2, [YLIM[0], 2.2], color=C["neutral"], lw=1.2,
            ls=":")
    ax.text(PEAK16 / BW / 1.1, 1.3, "ridge 295", color=C["neutral"],
            fontsize=10, ha="right")
    ax.text(0.02, 0.95, "FP16 ceiling: 989 TFLOP/s", transform=ax.transAxes,
            color=C["compute"], fontsize=11, fontweight="bold", ha="left",
            va="top")
    for i, (label, x, color_key, offset, align) in enumerate(layers):
        alpha = 1.0 if i < shown else 0.0
        y = min(PEAK16, BW * x)
        ax.plot([x], [y], "o", ms=10, color=C[color_key], mec="white", mew=1.5,
                zorder=5, alpha=alpha)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                    ha=align, va="center", fontsize=11, fontweight="bold",
                    color=C[color_key], alpha=alpha)
    sp.save(fig, out / f"layers_step{k}.svg")

# ---------------------------------------------------------------- batch
WIDTH = 2048


def intensity(batch):
    return 2 * batch * WIDTH * WIDTH / (2 * batch * WIDTH + 2 * WIDTH * WIDTH
                                        + 2 * batch * WIDTH)


crossing = 295.2 * WIDTH / (WIDTH - 2 * 295.2)  # about 415, derived
batches = np.logspace(0, np.log10(4096), 200)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 5000)
    ax.set_ylim(0.5, 1500)
    ax.minorticks_off()
    ax.set_xticks([1, 8, 32, 128, 512, 2048])
    ax.set_xticklabels(["1", "8", "32", "128", "512", "2,048"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1,000"])
    ax.grid(color=C["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axhline(295.2, color=C["compute"], lw=2, ls="--")
    ax.text(0.8, 295.2 * 1.25, "H100 FP16 ridge: 295", color=C["compute"],
            fontsize=11, fontweight="bold")
    if k >= 2:
        ax.plot(batches, intensity(batches), color=C["memory"], lw=3)
    pts = [(1, "≈ 1", (12, -12)), (32, "≈ 31", (8, -10)), (512, "341", (-10, 10))]
    for j, (b, text, offset) in enumerate(pts):
        alpha = 1.0 if (k >= 2 and j < 2) or k == 3 else 0.0
        ax.plot([b], [intensity(b)], "o", ms=10, color=C["memory"], mec="white",
                mew=1.5, zorder=5, alpha=alpha)
        ax.annotate(text, (b, intensity(b)), xytext=offset,
                    textcoords="offset points", fontsize=12, fontweight="bold",
                    color=C["memory"], alpha=alpha,
                    ha="right" if offset[0] < 0 else "left")
    ax.text(512, 3, "???", fontsize=16, fontweight="bold", color=C["danger"],
            ha="center", alpha=1.0 if k == 1 else 0.0)
    alpha = 1.0 if k == 3 else 0.0
    ax.axvline(crossing, color=C["danger"], lw=1.2, ls=":", alpha=alpha)
    ax.text(crossing * 1.1, 1.0, "crosses near 415\n(derived)", color=C["danger"],
            fontsize=10, alpha=alpha, va="bottom")
    ax.set_xlabel("Batch size (log scale)")
    ax.set_ylabel("FLOP/byte (log)")
    sp.save(fig, out / f"batch_step{k}.svg")

# ---------------------------------------------------------------- INT8 ridge
# Each layer: FP16 bar, INT8 bar; stage-3 also shows INT8 weights with FP16
# activations (569).
int8_bars = [  # layer, [(value, label, style)]
    ("Stage-3 3×3 conv", [(563, "563", "fp16"), (1126, "1,126", "int8"),
                          (569, "569: INT8, activations FP16", "wide")]),
    ("Stage-2 3×3 conv", [(287, "287", "fp16"), (574, "574", "int8")]),
    ("Stage-2 1×1 expansion", [(51, "51", "fp16"), (102, "102", "int8")]),
]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    y = 0.0
    centers = []
    for name, bars in reversed(int8_bars):
        ys = []
        for value, label, style in reversed(bars):
            alpha = 1.0 if style == "fp16" or k == 3 else 0.0
            if style == "fp16":
                face, edge, ink = C["compute"], C["compute"], C["compute"]
            elif style == "int8":
                face, edge, ink = C["danger_l"], C["danger"], C["danger"]
            else:
                face, edge, ink = C["danger"], C["danger"], C["danger"]
            ax.barh(y, value, height=0.8, color=face, edgecolor=edge,
                    alpha=alpha)
            ax.text(value * 1.12, y, label, va="center", fontsize=10.5,
                    fontweight="bold", color=ink, alpha=alpha)
            ys.append(y)
            y += 1
        centers.append((name, sum(ys) / len(ys)))
        y += 0.6
    for name, center in centers:
        ax.text(-0.01, center, name, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=12, fontweight="bold",
                color=C["text"])
    top = y - 0.4
    ax.axvline(295.2, color=C["compute"], lw=2, ls="--")
    ax.text(295.2 / 1.04, top, "FP16 ridge 295", color=C["compute"],
            fontsize=10, fontweight="bold", ha="right", va="bottom")
    alpha = 1.0 if k >= 2 else 0.0
    ax.axvline(590.7, color=C["danger"], lw=2, ls="--", alpha=alpha)
    ax.text(590.7 * 1.04, top, "INT8 ridge 591", color=C["danger"],
            fontsize=10, fontweight="bold", ha="left", va="bottom", alpha=alpha)
    ax.set_xscale("log")
    ax.set_xlim(20, 8000)
    ax.set_ylim(-0.7, top + 0.9)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xticks([30, 100, 300, 1000])
    ax.set_xticklabels(["30", "100", "300", "1,000"])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Arithmetic intensity at batch 32 (FLOP/byte, log scale)")
    sp.save(fig, out / f"int8ridge_step{k}.svg")

# ---------------------------------------------------------------- loop order
sp.hbars(out / "weight_reads.svg", [
    ("Filters held", 144, "data", "144"),
    ("Filters spilled", 5184, "danger", "5,184: 36×"),
], "reads", log=False, xlabel="Weight reads", size=(2.7, 1.35))

# ---------------------------------------------------------------- tile cliff
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

# ---------------------------------------------------------------- measured
# NVIDIA figure 7(a), A100: achieved TFLOP/s against N, read from the
# published SVG.
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
    ax.annotate("136: 148 TFLOP/s\n6% more work, 36% more time",
                (136, 148.2), xytext=(160, 72), textcoords="data",
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

# ---------------------------------------------------------------- Amdahl
# Slide variant of HA figure 2: speedup = 1 / ((1 - p) + p / G) (HA
# equation 1), with Lighthouse 11.1's two scenarios marked: ResNet-50,
# p = 0.95, G = 247 -> 18.6x; GPT-2 decode, p = 0.80, G = 247 -> 4.9x.
gains = np.logspace(0, 3, 300)
fractions = np.linspace(0.78, 1.0, 300)
grid_g, grid_p = np.meshgrid(gains, fractions)
speedup = 1 / ((1 - grid_p) + grid_p / grid_g)
fig, ax = plt.subplots(figsize=(3.4, 2.7))
levels = [1, 2, 5, 10, 20, 50, 1000]
shades = ["#EEF4FA", "#D1E6F3", "#A9CFE8", "#7FB6DB", "#4C93C4", "#006395"]
ax.contourf(grid_g, grid_p, speedup, levels=levels, colors=shades)
lines = ax.contour(grid_g, grid_p, speedup, levels=[2, 5, 10, 20, 50],
                   colors="white", linewidths=1.2)
for value in (5, 10, 20, 50):
    edge = (1 - 1 / value) / (1 - 1 / 1000)
    ax.text(1150, edge, f"{value}×", va="center", ha="left", fontsize=10,
            color=C["memory"], fontweight="bold")
points = [(0.95, "ResNet-50: 18.6×", (-8, 10)), (0.80, "GPT-2 decode: 4.9×", (-8, 10))]
marks = []
for p_frac, label, offset in points:
    dot, = ax.plot([247], [p_frac], "o", ms=9, color=C["danger"], mec="white", mew=1.5)
    text = ax.annotate(label, (247, p_frac), xytext=offset, textcoords="offset points",
                       ha="right", fontsize=11, fontweight="bold", color=C["danger"],
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                 alpha=0.85))
    marks.append((dot, text))
ax.set_xscale("log")
ax.set_xticks([1, 10, 100, 1000])
ax.set_xticklabels(["1", "10", "100", "1,000"])
ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
ax.set_xlabel("Accelerator gain (log scale)")
ax.set_ylabel("Accelerated fraction")
# Step 1 shows only the ResNet-50 point; amdahl.svg (both points) is the final
# step and the thumbnail. Both files keep the same canvas.
fig.savefig(out / "amdahl.svg", format="svg", bbox_inches="tight", pad_inches=0.15,
            facecolor="white")
marks[1][0].set_alpha(0.0)
marks[1][1].set_alpha(0.0)
marks[1][1].get_bbox_patch().set_alpha(0.0)
sp.save(fig, out / "amdahl_step1.svg")

# ---------------------------------------------------------------- compiler
sp.hbars(out / "compiled.svg", [
    ("Eager (174 kernels)", 3.53, "neutral", "3.53 ms"),
    ("Compiled FP16 (56 kernels)", 0.33, "compute", "0.33 ms: 10.6× faster"),
], "ms", log=False, value_fmt="{:.2f}", xlabel="ResNet-50 batch one on an H100 (modeled)",
    size=sp.SHORT)

# ---------------------------------------------------------------- bandwidth taper
# Slide variant of HA margin figure 2 (Sec. 11.5.4): HBM 3,350, NVLink 900,
# PCIe 64, network 50 GB/s. HBM over PCIe = 52x (derived).
taper_rows = [("HBM", 3350, "memory", "3,350 GB/s"), ("NVLink", 900, "network", "900"),
              ("PCIe", 64, "danger", "64"), ("Network", 50, "neutral", "50")]
fig, ax = plt.subplots(figsize=(5.8, 2.1))
for i, (label, value, color_key, text) in enumerate(taper_rows):
    y = len(taper_rows) - 1 - i
    ax.barh(y, value, color=C[color_key], height=0.62)
    ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=13, fontweight="bold", color=C["text"])
    ax.text(value * 1.15, y, text, va="center", fontsize=13, fontweight="bold",
            color=C[color_key])
ax.set_xscale("log")
ax.set_xlim(20, 60000)
ax.set_ylim(-0.6, len(taper_rows) - 0.4)
ax.set_yticks([])
ax.minorticks_off()
ax.set_xticks([100, 1000, 10000])
ax.set_xticklabels(["100", "1,000", "10,000"])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("Bandwidth from one H100 (GB/s, log scale)")
sp.save(fig, out / "taper.svg")

# ---------------------------------------------------------------- INT8 twice
# H100 dense peaks: FP16 989 TFLOP/s, INT8 1,979 TOPS; ResNet-50 8.2 GFLOP per
# image over each peak: about 8 and 4 microseconds (HA Sec. 11.4.5, Lighthouse
# 11.1, Napkin Math 11.6).
sp.hbars(out / "int8peak.svg", [
    ("FP16", 989, "compute", "989 TFLOP/s"),
    ("INT8", 1979, "danger", "1,979 TOPS: 2×"),
], "", log=False, xlabel="H100 dense peak", size=(2.9, 1.5))
sp.hbars(out / "int8floor.svg", [
    ("FP16", 8.3, "compute", "8 µs"),
    ("INT8", 4.1, "danger", "4 µs"),
], "", log=False, xlabel="ResNet-50 arithmetic, one image", size=(2.9, 1.5))

print("figures written")
