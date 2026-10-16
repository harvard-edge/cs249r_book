#!/usr/bin/env python3
"""Figures for Lecture 10 (Model Compression).

Every value is sourced. "Chapter" means Model Compression in Volume I.
  - Memory ladder: 175B FP16 350 GB, phone RAM 8 GB, microcontroller SRAM
    512 KB, 7B FP16 14 GB (chapter Sec. 10.1). 7B INT4 3.5 GB is derived
    (7e9 parameters x 0.5 B). DS-CNN keyword spotter 800 KB FP32, 200 KB INT8
    (Lighthouse 10.2).
  - Weight streaming at 51.2 GB/s: MobileNetV2 FP32 0.27 ms, ResNet-50 FP32
    2 ms, GPT-2 XL FP16 59 ms, INT4 15 ms; 33 ms camera frame (table 1).
  - Bytes per stored nonzero: FP32 value with a 16-bit index 6 B (adds half
    again, Sec. 10.4.2.2); CSR 8 B, break-even 50 percent; COO 12 B,
    break-even 66.7 percent (chapter fallacy on sparse formats).
  - 2:4 block [0, 1.4, 0, -2.1] -> [1.4, -2.1], indices 01, 11 (Sec. 10.2.1).
    Measured INT8 2:4 GEMM speedup on A100, M = N = 10240: about 1.2x at
    K = 1,280, 1.5x at 2,560, 1.9x at 12,800 (Mishra et al. 2021, Fig. 3,
    values read from the published plot, approximate). Vendor design peak 2x
    (Mishra et al. 2021, Table 1; NVIDIA 2020).
  - Channel tiles: 256 -> 250 leaves a partly empty tile of 8; 248 fills
    every tile (Sec. 10.2.1). MobileNetV2 width 0.75: 3.5M -> 2.6M
    parameters, 600 -> 418 MFLOP (Sec. 10.2.4; Sandler et al. 2018).
  - DistilBERT against BERT: 40 percent fewer parameters, 60 percent faster,
    97 percent of the language-understanding score (Sanh et al. 2019).
  - Energy per operation, 45 nm (Horowitz 2014, table 8): add 0.90 / 0.03
    pJ, multiply 3.70 / 0.20, multiply-accumulate 4.60 / 0.30; 32-bit DRAM
    read 640 pJ (Sec. 10.3.2).
  - Napkin Math 10.1: range [-1, 3] to UINT8, s = 4/255, z = 64,
    [-1, 1, 0, 3] -> [0, 128, 64, 255].
  - Fusion: ResNet-50 layer 256 x 28 x 28, unfused 7.2 MB in 6 transfers,
    fused 4 MB in 2 (Sec. 10.4.1).
  - Roofline points: GPT-2 XL decode INT4 3.1x (Napkin Math 10.4); large-
    batch MLP INT8 about 2x (Napkin Math 10.2). Schematic axes.
  - ResNet-50 INT8 on H100 (table 13): size 4x, batch-one latency 1.3x,
    large-batch throughput 5.9x (INT8 measured by MLPerf, FP32 modeled).
  - Blur pipeline (fig-compression-waterfall, illustrative): 14 / 10.4 /
    2.6 MB; 125 / 87 / 20 ms against 33 ms.
  - Paper 8x against realized 1.5x (Sec. 10.4, illustrative); model 20
    percent of latency caps a perfect model speedup at 1.25x (Systems
    Perspective 10.4).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent
FONT = "Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------------------
# A bar helper with a show mask and optional overlay bars, so a build can
# reveal any row (not only the first k) while the axes and bounding box stay
# fixed. rows: dicts with label, value, color, text; optional over (value),
# over_color, over_text.
# ---------------------------------------------------------------------------
def bars(path, rows, shown, log=False, xlim=None, xlabel=None, size=sp.SIZE,
         vlines=(), xticks=None, note=None, label_size=13, boxed=False):
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        vis = shown[i]
        ax.barh(y, r["value"], color=C[r["color"]], height=0.62,
                alpha=1.0 if vis else 0.0)
        ax.text(-0.01, y, r["label"], transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=label_size, color=C["text"],
                fontweight="bold", alpha=1.0 if vis else 0.0)
        x_end = r["value"]
        over_vis = vis and r.get("over_on", False)
        if "over" in r:
            ax.barh(y, r["over"], color=C[r["over_color"]], height=0.34,
                    alpha=1.0 if over_vis else 0.0, zorder=3)
        tx = x_end * 1.25 if log else x_end + (xlim[1] if xlim else 1) * 0.012
        text = r["text"]
        if over_vis and "over_text" in r:
            text = r["over_text"]
        bb = dict(facecolor="white", edgecolor="none", pad=1.0, alpha=1.0 if vis else 0.0) if boxed else None
        ax.text(tx, y, text, va="center", fontsize=12, bbox=bb, zorder=4,
                color=C[r.get("over_color", r["color"]) if over_vis else r["color"]],
                fontweight="bold", alpha=1.0 if vis else 0.0)
        # reserve the widest text so every step has the same bounding box
        if "over_text" in r:
            ax.text(tx, y, r["over_text"], va="center", fontsize=12, alpha=0.0)
    for x, label, ck in vlines:
        ax.axvline(x, color=C[ck], lw=1.5, ls="--", zorder=1)
        ax.text(x, n - 0.45, " " + label, color=C[ck], fontsize=10,
                fontweight="bold", va="bottom", ha="left")
    if log:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if xticks:
        ax.set_xticks([t for t, _ in xticks])
        ax.set_xticklabels([s for _, s in xticks])
        ax.minorticks_off()
    if xlabel:
        ax.set_xlabel(xlabel)
    if note:
        fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=9,
                 color=C["neutral"])
    sp.save(fig, path)


def svg(path, w, h, body):
    Path(path).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="{FONT}">\n'
        f'<rect width="{w}" height="{h}" fill="white"/>\n{body}</svg>\n')


def text(x, y, s, size=22, color=C["text"], weight="normal", anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>\n')


def box(x, y, w, h, fill, stroke, sw=2, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n')


def arrow(x1, y1, x2, y2, color, sw=3):
    # line with a small triangular head at (x2, y2)
    ang = np.arctan2(y2 - y1, x2 - x1)
    hl, hw = 12, 7
    bx, by = x2 - hl * np.cos(ang), y2 - hl * np.sin(ang)
    px, py = -np.sin(ang) * hw, np.cos(ang) * hw
    return (f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" '
            f'stroke="{color}" stroke-width="{sw}"/>\n'
            f'<polygon points="{x2},{y2} {bx + px:.1f},{by + py:.1f} '
            f'{bx - px:.1f},{by - py:.1f}" fill="{color}"/>\n')


GB, MB, KB = 1e9, 1e6, 1e3

# ---------------------------------------------------------------------------
# 1. Hook: the memory ladder (log scale), and its return in the synthesis
# ---------------------------------------------------------------------------
ladder_rows = [
    dict(label="175B, FP16", value=350 * GB, color="memory", text="350 GB"),
    dict(label="7B, FP16", value=14 * GB, color="danger", text="14 GB",
         over=3.5 * GB, over_color="data", over_text="INT4: 3.5 GB"),
    dict(label="Phone RAM", value=8 * GB, color="neutral", text="8 GB"),
    dict(label="MCU SRAM", value=512 * KB, color="neutral", text="512 KB",
         over=200 * KB, over_color="data", over_text="spotter: 200 KB"),
]
ladder_kw = dict(log=True, xlim=(5e4, 3e14), size=(4.2, 2.9),
                 xlabel="Bytes (log scale)",
                 xticks=[(1e6, "1 MB"), (1e9, "1 GB"), (1e12, "1 TB")])
bars(out / "ladder_step1.svg", ladder_rows, [1, 0, 1, 1], **ladder_kw)
bars(out / "ladder_step2.svg", ladder_rows, [1, 1, 1, 1], **ladder_kw)
# synthesis: the same ladder with the compressed versions overlaid
syn = [dict(r) for r in ladder_rows]
syn[1]["over_on"] = True
bars(out / "synthesis_step1.svg", syn, [1, 1, 1, 1], **ladder_kw)
syn[3]["over_on"] = True
bars(out / "synthesis_step2.svg", syn, [1, 1, 1, 1], **ladder_kw)

# ---------------------------------------------------------------------------
# 2. Budget: time to stream the weights once on the phone's 51.2 GB/s bus
# ---------------------------------------------------------------------------
stream_rows = [
    dict(label="MobileNetV2, FP32", value=0.27, color="memory",
         text="0.27 ms: 1% of a frame"),
    dict(label="ResNet-50, FP32", value=2.0, color="memory", text="2 ms"),
    dict(label="GPT-2 XL, FP16", value=59, color="danger", text="59 ms per token"),
    dict(label="GPT-2 XL, INT4", value=15, color="data", text="15 ms per token"),
]
for k in range(1, 5):
    bars(out / f"stream_step{k}.svg", stream_rows, [i < k for i in range(4)],
         xlim=(0, 80), vlines=[(33.3, "33 ms camera frame", "compute")],
         xlabel="Time to read every weight once at 51.2 GB/s (ms)")

# ---------------------------------------------------------------------------
# 3. Three tiers, each moving one term (SVG, full width, 3 steps)
# ---------------------------------------------------------------------------
tiers = [
    ("What the model computes", "pruning, distillation", "FLOPs/sample, weight bytes",
     C["compute"], C["compute_l"]),
    ("How values are stored", "quantization", "bytes moved, energy/op",
     C["memory"], C["memory_l"]),
    ("How it executes", "fusion, sparse kernels", "bytes, overhead, efficiency",
     C["network"], C["network_l"]),
]
for k in range(1, 4):
    body = text(10, 34, "Software", 24, C["neutral"], "bold")
    body += arrow(55, 50, 55, 432, C["neutral"], 3)
    body += text(10, 466, "Hardware", 24, C["neutral"], "bold")
    body += text(560, 34, "Terms it moves", 24, C["neutral"], "bold")
    for i, (name, how, term, stroke, fill) in enumerate(tiers):
        y = 52 + i * 136
        op = "1" if i < k else "0"
        body += f'<g opacity="{op}">\n'
        body += box(90, y, 420, 116, fill, stroke, 3)
        body += text(300, y + 50, name, 30, C["text"], "bold", "middle")
        body += text(300, y + 92, how, 27, stroke, "normal", "middle")
        body += arrow(516, y + 58, 552, y + 58, stroke, 3)
        body += text(560, y + 50, term.split(", ", 1)[0] + ",", 27, stroke, "bold")
        body += text(560, y + 86, term.split(", ", 1)[1], 27, stroke, "bold")
        body += "</g>\n"
    svg(out / f"tiers_step{k}.svg", 900, 480, body)

# ---------------------------------------------------------------------------
# 4. Pruning: dense, unstructured, structured (3 panels, 3 steps)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(10)
W = rng.normal(0, 1, (6, 6)) * rng.choice([0.3, 1.0], size=(6, 6), p=[0.5, 0.5])
mag = np.abs(W)
thresh = np.median(mag)
keep_cols = np.sort(np.argsort(mag.sum(axis=0))[3:])  # three strongest columns


def grid(ax, x0, mat, mask=None, cols=None):
    # draw a matrix of shaded cells; mask False -> pruned (white with 0)
    nr, nc = mat.shape
    vmax = mag.max()
    for i in range(nr):
        for j in range(nc):
            keep = True if mask is None else mask[i, j]
            shade = 0.15 + 0.85 * abs(mat[i, j]) / vmax
            col = plt.cm.Oranges(shade) if keep else "white"
            ax.add_patch(plt.Rectangle((x0 + j, nr - 1 - i), 1, 1, facecolor=col,
                                       edgecolor="#999999", lw=0.8))
            if not keep:
                ax.text(x0 + j + 0.5, nr - 0.5 - i, "0", ha="center",
                        va="center", fontsize=10, color="#999999")


for k in range(1, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    grid(ax, 0, W)
    ax.text(3, -0.5, "Dense:\n36 weights", ha="center", va="top", fontsize=12,
            fontweight="bold", color=C["text"])
    if k >= 2:
        grid(ax, 11, W, mask=mag >= thresh)
        ax.text(14, -0.5, "Unstructured:\n18 zeros", ha="center",
                va="top", fontsize=12, fontweight="bold", color=C["danger"])
    if k >= 3:
        grid(ax, 22, W[:, keep_cols])
        ax.text(23.5, -0.5, "Structured:\n6×3, dense", ha="center",
                va="top", fontsize=12, fontweight="bold", color=C["data"])
    ax.set_xlim(-2, 27)
    ax.set_ylim(-2.2, 6.2)
    ax.set_aspect("equal")
    ax.axis("off")
    sp.save(fig, out / f"prune_step{k}.svg")

# ---------------------------------------------------------------------------
# 5. Sparse storage: bytes per stored nonzero
# ---------------------------------------------------------------------------
fmt_rows = [
    dict(label="Dense FP32", value=4, color="neutral", text="4 B"),
    dict(label="+ 16-bit index", value=6, color="danger", text="6 B: 2 MB → 3 MB"),
    dict(label="CSR", value=8, color="memory", text="8 B: wins above 50% zeros"),
    dict(label="COO", value=12, color="memory", text="12 B: above 66.7%"),
]
bars(out / "sparsefmt_step1.svg", fmt_rows, [1, 1, 0, 0], xlim=(0, 30),
     xlabel="Bytes stored per nonzero weight", size=(4.4, 2.6))
bars(out / "sparsefmt_step2.svg", fmt_rows, [1, 1, 1, 1], xlim=(0, 30),
     xlabel="Bytes stored per nonzero weight", size=(4.4, 2.6))

# ---------------------------------------------------------------------------
# 6. 2:4 sparsity: the block (SVG, column size) and measured GEMM speedups
# ---------------------------------------------------------------------------
cell = 78
body = text(10, 34, "Every group of 4 keeps 2", 26, C["text"], "bold")
vals = ["0", "1.4", "0", "−2.1"]
for j, v in enumerate(vals):
    keep = v != "0"
    body += box(10 + j * cell, 60, cell, cell, C["compute_l"] if keep else "white",
                C["compute"] if keep else "#999999", 2, 0)
    body += text(10 + j * cell + cell / 2, 60 + cell / 2 + 9, v, 26,
                 C["text"] if keep else "#999999", "bold" if keep else "normal", "middle")
body += arrow(170, 150, 170, 205, C["neutral"], 3)
for j, v in enumerate(["1.4", "−2.1"]):
    body += box(10 + j * cell, 215, cell, cell, C["compute_l"], C["compute"], 2, 0)
    body += text(10 + j * cell + cell / 2, 215 + cell / 2 + 9, v, 26, C["text"], "bold", "middle")
for j, v in enumerate(["01", "11"]):
    body += box(200 + j * 66, 228, 60, 52, C["neutral_l"], C["neutral"], 2, 0)
    body += text(200 + j * 66 + 30, 263, v, 24, C["text"], "normal", "middle")
body += text(10, 322, "2 values", 22, C["compute"], "bold")
body += text(200, 322, "2-bit positions", 22, C["neutral"], "bold")
body += text(10, 385, "Design peak: 2× math throughput", 24, C["text"], "bold")
body += text(10, 418, "(vendor specification)", 22, C["neutral"])
svg(out / "sparse24.svg", 420, 440, body)

gemm_rows = [
    dict(label="K = 1,280", value=1.2, color="compute", text="≈1.2×"),
    dict(label="K = 2,560", value=1.5, color="compute", text="≈1.5×"),
    dict(label="K = 12,800", value=1.9, color="compute", text="≈1.9×"),
]
fig, ax = plt.subplots(figsize=(4.6, 2.4))
for i, r in enumerate(gemm_rows):
    y = 2 - i
    ax.barh(y, r["value"], color=C["compute"], height=0.62, zorder=2)
    ax.text(r["value"] - 0.04, y, r["text"], ha="right", va="center", fontsize=13,
            color="white", fontweight="bold", zorder=3)
    ax.text(-0.02, y, r["label"], transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=13, fontweight="bold", color=C["text"])
ax.axvline(1.0, color=C["neutral"], lw=1.5, ls="--", zorder=1)
ax.axvline(2.0, color=C["danger"], lw=1.5, ls="--")
ax.text(1.0, 2.55, "dense ", color=C["neutral"], fontsize=10, fontweight="bold", ha="right")
ax.text(2.0, 2.55, " 2× design peak", color=C["danger"], fontsize=10, fontweight="bold")
ax.set_xlim(0, 2.6)
ax.set_ylim(-0.6, 2.9)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("Measured 2:4 speedup over dense\n(A100, INT8 GEMM, M = N = 10,240)")
sp.save(fig, out / "gemm24.svg")

# ---------------------------------------------------------------------------
# 7. Channel pruning on the tile grid (SVG) and MobileNetV2 width 0.75 bars
# ---------------------------------------------------------------------------
body = ""
rows = [("256 → 250 channels", 2, C["danger"], "last tile: 2 of 8 lanes busy"),
        ("256 → 248 channels", 8, C["data"], "every tile full")]
for r, (lab, last, color, note) in enumerate(rows):
    y = 10 + r * 215
    body += text(10, y + 30, lab, 30, C["text"], "bold")
    for t in range(3):
        x0 = 10 + t * 180
        fill_n = 8 if (t < 2 or r == 1) else last
        for c in range(8):
            on = c < fill_n
            body += box(x0 + c * 20, y + 50, 18, 60,
                        C["compute_l"] if on else "white",
                        C["compute"] if on else "#BBBBBB", 1.5, 0,
                        None if on else "4,3")
        num = (30 + t) if r == 0 else (29 + t)
        body += text(x0 + 80, y + 142, f"tile {num}", 24, C["neutral"], "normal", "middle")
    body += text(10, y + 186, note, 28, color, "bold")
svg(out / "tiles.svg", 560, 430, body)

fig, ax = plt.subplots(figsize=(4.4, 2.6))
groups = [("Parameters", 3.5, 2.6, "M", "74%"), ("MFLOP", 600, 418, "", "70%")]
for gi, (name, full, slim, unit, pct) in enumerate(groups):
    y = 1 - gi
    ax.barh(y + 0.2, 1.0, height=0.36, color=C["neutral"])
    ax.barh(y - 0.2, slim / full, height=0.36, color=C["compute"])
    ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=13, fontweight="bold", color=C["text"])
    ax.text(1.02, y + 0.2, f"1.0: {full:g}{unit}", va="center", fontsize=12,
            color=C["neutral"], fontweight="bold")
    ax.text(slim / full + 0.02, y - 0.2, f"0.75: {slim:g}{unit} ({pct})",
            va="center", fontsize=12, color=C["compute"], fontweight="bold")
ax.text(0.0, 1.62, "MobileNetV2 width multiplier", fontsize=12, color=C["text"])
ax.set_xlim(0, 1.75)
ax.set_ylim(-0.6, 1.8)
ax.set_yticks([])
ax.set_xticks([])
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
sp.save(fig, out / "width075.svg")

# ---------------------------------------------------------------------------
# 8. DistilBERT as a share of BERT
# ---------------------------------------------------------------------------
distil_rows = [
    dict(label="Parameters", value=60, color="data", text="60% (40% fewer)"),
    dict(label="Inference speed", value=160, color="compute", text="160% (1.6× faster)"),
    dict(label="Language score", value=97, color="memory", text="97%"),
]
bars(out / "distilbert.svg", distil_rows, [1, 1, 1], xlim=(0, 260), size=(4.8, 2.3),
     xticks=[(0, "0"), (100, "100 = BERT"), (200, "200")],
     xlabel="DistilBERT relative to BERT (%)")

# Soft targets (variant of fig-kd-targets): hard label against the teacher
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    cls = ["Cat", "Dog", "Fox"]
    hard, soft = [100, 0, 0], [85, 10, 5]
    xs = np.arange(3)
    ax.bar(xs - 0.2, hard, width=0.38, color=C["neutral"])
    for x, v in zip(xs, hard):
        ax.text(x - 0.2, v + 2, f"{v}", ha="center", fontsize=12, color=C["neutral"],
                fontweight="bold")
    a = 1.0 if k == 2 else 0.0
    ax.bar(xs + 0.2, soft, width=0.38, color=C["compute"], alpha=a)
    for x, v in zip(xs, soft):
        ax.text(x + 0.2, v + 2, f"{v}", ha="center", fontsize=12, color=C["compute"],
                fontweight="bold", alpha=a)
    ax.text(2.45, 92, "label", ha="right", fontsize=13, color=C["neutral"], fontweight="bold")
    ax.text(2.45, 78, "teacher", ha="right", fontsize=13, color=C["compute"],
            fontweight="bold", alpha=a)
    ax.set_xticks(xs)
    ax.set_xticklabels(cls, fontsize=13)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Probability (%)")
    sp.save(fig, out / f"kd_step{k}.svg")

# ---------------------------------------------------------------------------
# 9. Energy per operation, FP32 against INT8 (log scale), then a DRAM read
# ---------------------------------------------------------------------------
ops = [("Add", 0.90, 0.03, "30×"), ("Multiply", 3.70, 0.20, "18.5×"),
       ("Multiply-accumulate", 4.60, 0.30, "15×")]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ylabels = []
    for i, (name, f32, i8, ratio) in enumerate(ops):
        y = 3 - i
        ax.barh(y + 0.19, f32, height=0.34, color=C["compute"])
        ax.barh(y - 0.19, i8, height=0.34, color=C["neutral"])
        ax.text(f32 * 1.2, y + 0.19, f"FP32 {f32:.2f} pJ", va="center", fontsize=11,
                color=C["compute"], fontweight="bold")
        ax.text(i8 * 1.2, y - 0.19, f"INT8 {i8:.2f} pJ: {ratio} less", va="center",
                fontsize=11, color=C["neutral"], fontweight="bold")
        ylabels.append((y, name))
    vis = 1.0 if k == 2 else 0.0
    ax.barh(-0.05, 640, height=0.42, color=C["memory"], alpha=vis)
    ax.text(640 * 1.2, -0.05, "640 pJ", va="center", fontsize=11, color=C["memory"],
            fontweight="bold", alpha=vis)
    ylabels.append((-0.05, "32-bit DRAM read"))
    for y, name in ylabels:
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"],
                alpha=vis if name.startswith("32") else 1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 20000)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Energy (picojoules, log scale; 45 nm)")
    sp.save(fig, out / f"energy_step{k}.svg")



# ---------------------------------------------------------------------------
# 10. Napkin Math 10.1 on a number line
# ---------------------------------------------------------------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 1.7))
    ax.plot([-1.25, 3.25], [1, 1], color=C["text"], lw=1.5)
    ax.plot([-1.25, 3.25], [0, 0], color=C["neutral"], lw=1.5)
    ax.text(-1.4, 1, "real", ha="right", va="center", fontsize=15, fontweight="bold")
    ax.text(-1.4, 0, "UINT8", ha="right", va="center", fontsize=15, fontweight="bold",
            color=C["neutral"])
    for real, q in ((-1, 0), (0, 64), (1, 128), (3, 255)):
        xq = -1 + q * 4 / 255
        ax.plot([real], [1], "o", ms=9, color=C["memory"])
        ax.text(real, 1.25, f"{real:g}", ha="center", fontsize=15, fontweight="bold",
                color=C["memory"])
        a = 1.0 if k == 2 else 0.0
        ax.annotate("", (xq, 0.06), (real, 0.92), alpha=a,
                    arrowprops=dict(arrowstyle="-|>", color=C["compute"], lw=1.6, alpha=a))
        ax.plot([xq], [0], "s", ms=8, color=C["compute"], alpha=a)
        ax.text(xq, -0.5, f"{q}", ha="center", fontsize=15, fontweight="bold",
                color=C["compute"], alpha=a)
    ax.set_xlim(-2.2, 3.4)
    ax.set_ylim(-0.6, 1.5)
    ax.axis("off")
    sp.save(fig, out / f"numline_step{k}.svg")

# ---------------------------------------------------------------------------
# 11. Fusion: six transfers against two (SVG, 2 steps)
# ---------------------------------------------------------------------------
for k in (1, 2):
    body = text(10, 34, "Unfused: 6 transfers, 7.2 MB", 30, C["danger"], "bold")
    for i, name in enumerate(["Conv", "BN", "ReLU"]):
        x = 30 + i * 240
        body += box(x, 52, 190, 74, C["compute_l"], C["compute"], 2.5)
        body += text(x + 95, 100, name, 30, C["text"], "bold", "middle")
        body += arrow(x + 65, 196, x + 65, 132, C["memory"], 3.5)
        body += arrow(x + 125, 132, x + 125, 196, C["memory"], 3.5)
    body += box(10, 200, 740, 58, C["memory"], C["memory"], 0, 4)
    body += text(380, 240, "HBM (main memory)", 30, "white", "bold", "middle")
    body += f'<g opacity="{1 if k == 2 else 0}">\n'
    body += arrow(95, 262, 95, 316, C["memory"], 3.5)
    body += arrow(665, 316, 665, 262, C["memory"], 3.5)
    body += box(30, 320, 700, 74, C["compute_l"], C["compute"], 2.5)
    body += text(380, 367, "Conv + BN + ReLU, BN folded in", 30, C["text"], "bold", "middle")
    body += text(10, 440, "Fused: 2 transfers, 4 MB (44.8% less)", 30, C["data"], "bold")
    body += "</g>\n"
    svg(out / f"fusion_step{k}.svg", 760, 455, body)

# ---------------------------------------------------------------------------
# 12. One precision change, two regimes (schematic roofline)
# ---------------------------------------------------------------------------
sp.roofline_steps(out / "roofline_int", [
    ("Decode, INT4: 3.1×", 20.0, "memory"),
    ("Large-batch MLP, INT8: 2×", 3000.0, "compute"),
], note="Schematic axes")

# ---------------------------------------------------------------------------
# 13. ResNet-50 INT8 on the H100 (table 13)
# ---------------------------------------------------------------------------
r50_rows = [
    ("Model size", 4.0, "memory", "4×  (102.4 → 25.6 MB)"),
    ("Batch-one latency", 1.3, "neutral", "1.3×  (0.38 → 0.30 ms)"),
    ("Large-batch throughput", 5.9, "compute", "5.9×  (15,200 → 89,470 img/s)"),
]
sp.hbars_steps(out / "r50int8", r50_rows, "×", log=False,
               xlabel="INT8 gain over FP32 (×)", value_fmt="{:g}")

# ---------------------------------------------------------------------------
# 14. Blur pipeline waterfall (variant of fig-compression-waterfall)
# ---------------------------------------------------------------------------
stages = ["FP32\nCPU", "Pruned\nCPU", "Pruned INT8\naccelerator"]
memv = [14.0, 10.4, 2.6]
timev = [125, 87, 20]
for k in (1, 2, 3):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.3))
    for ax, vals, ck, unit, ttl in ((a1, memv, "memory", "MB", "Weight memory (MB)"),
                                    (a2, timev, "compute", "ms", "Frame time (ms)")):
        for i, v in enumerate(vals):
            a = 1.0 if i < k else 0.0
            ax.bar(i, v, color=C[ck], width=0.7, alpha=a)
            inside = ck == "compute" and v < 40
            ax.text(i, v / 2 if inside else v + max(vals) * 0.03, f"{v:g} {unit}",
                    ha="center", va="center" if inside else "bottom",
                    fontsize=10 if inside else 11,
                    fontweight="bold", color="white" if inside else C[ck], alpha=a)
        ax.set_xticks(range(3))
        ax.set_xticklabels(stages, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)
    a2.axhline(33, color=C["danger"], lw=1.5, ls="--")
    a2.text(2.45, 33, " 33 ms\n budget", color=C["danger"], fontsize=11, fontweight="bold",
            ha="left", va="center", clip_on=False)
    fig.text(0.01, -0.02, "Illustrative scenario (Example 10.1)", fontsize=9,
             color=C["neutral"])
    fig.tight_layout(w_pad=1.5)
    sp.save(fig, out / f"waterfall_step{k}.svg")

# ---------------------------------------------------------------------------
# 15. Paper against realized, then Amdahl with the model at 20 percent
# ---------------------------------------------------------------------------
sp.hbars(out / "paper_real.svg", [
    ("Paper: 2× × 4×", 8, "neutral", "8×"),
    ("Realized", 1.5, "danger", "1.5×  (18.8% of paper)"),
], "×", log=False, size=(5.8, 1.15), xlabel="Speedup (illustrative)")

for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.0))
    m = 20 if k == 1 else 0
    ax.barh(0, m, color=C["compute"], height=0.6)
    ax.barh(0, 80, left=m, color=C["neutral"], height=0.6)
    if k == 1:
        ax.text(10, 0, "Model\n20%", ha="center", va="center", fontsize=11,
                color="white", fontweight="bold")
    ax.text(m + 40, 0, "Decode, resize, display: 80%", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold")
    ax.text(102, 0, "100%" if k == 1 else "80%: 1 / 0.8 = 1.25×", va="center",
            fontsize=12, fontweight="bold",
            color=C["text"] if k == 1 else C["danger"])
    ax.text(102, 0, "80%: 1 / 0.8 = 1.25×", va="center", fontsize=12, alpha=0.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Share of the original frame latency (%)")
    ax.spines["left"].set_visible(False)
    sp.save(fig, out / f"amdahl_step{k}.svg")

print("figures written")
