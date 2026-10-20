#!/usr/bin/env python3
"""Figures for Lecture 10 (Model Compression, Part 1).

Every number comes from MLSysIM through calcs.py (imported as K): val(key) is
the string a slide prints and num(key) its value; bar lengths read the
Scenarios.* quantities. Where each figure's inputs live:
  - Memory ladder: Models (GPT-3, Llama 2 7B, DS-CNN) and Platforms (phone
    RAM, microcontroller SRAM); Scenarios.Llama2_7BTokenOnFlagshipPhone.
  - Weight streaming: Scenarios.*WeightReadPhone (flagship phone LPDDR bus);
    the camera frame from Scenarios.MobileNetV2BlurPipelinePhone.
  - Pruning grids (6 by 6, random, seed 10) and the 2:4 block [0, 1.4, 0,
    -2.1] are illustrations; their counts are read off the drawn matrices.
  - Listing mask and 4 by 4 product: Scenarios.MagnitudePruned3x3 and
    SparseMatVec4x4; sparse-format bytes and break-evens: physics.memory and
    Scenarios.WearableMobileNetV2SparseFlash.
  - 2:4 GEMM: Literature.Series.MishraInt8SparseGemm (values read from the
    published plot, approximate) and Literature.CompressionResults.
  - Channel tiles: Scenarios.Channels256To250Tile8 and Channels256To248Tile8;
    width 0.75: Models MobileNetV2 and MobileNetV2_Alpha0_75.
  - DistilBERT: Literature.CompressionResults (Sanh et al. 2019); soft
    targets: Scenarios.TeacherSoftTargetsT4.
  - Fusion: Scenarios.ResNet50Conv256FusionH100.
  - Waterfall, paper against realized, Amdahl: Scenarios.MobileNetV2Blur-
    PipelinePhone, PrunedInt8PaperVsMeasured, ModelFifthOfRequestLatency.
  - Break-even axis: CSR and COO from physics.memory; the 80 to 90 percent
    band for unstructured kernels is the chapter's (Sec. 10.4.2), unsourced.
  - MLP share: physics.transformer.calc_transformer_block_weights.
  - Smallest capable model: Scenarios.SmallestModelAtSixtyPercentMmlu
    (Epoch AI data, Literature.Series.MMLUDenseModels).
  - Frontier: Literature.Series.MobileNetV3Table3 (Howard et al. 2019).
  - Iterative pruning: Literature.Series.HanAlexNetPruning (Han et al. 2015).
  - Deep Compression cliffs: Literature.Series.DeepCompressionMethods.
  - Measured CPU bars: cpu_measure.json (measure_cpu.py, run for this deck).
  - hook_step1.svg is copied from ../10b_model_compression/images.

Run from this folder with the MLSysBook venv (it imports mlsysim):
    python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as K  # noqa: E402  (every derived number; prints its values)
from deckvals import val, num  # noqa: E402

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
         vlines=(), xticks=None, note=None, label_size=13, boxed=False,
         text_size=12, axis_size=None):
    """axis_size: tick, axis-label, and reference-line label size; raise it
    with label_size and text_size when the plot is drawn for a column slot."""
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
        ax.text(tx, y, text, va="center", fontsize=text_size, bbox=bb, zorder=4,
                color=C[r.get("over_color", r["color"]) if over_vis else r["color"]],
                fontweight="bold", alpha=1.0 if vis else 0.0)
        # reserve the widest text so every step has the same bounding box
        if "over_text" in r:
            ax.text(tx, y, r["over_text"], va="center", fontsize=text_size, alpha=0.0)
    for x, label, ck in vlines:
        ax.axvline(x, color=C[ck], lw=1.5, ls="--", zorder=1)
        ax.text(x, n - 0.45, " " + label, color=C[ck], fontsize=axis_size or 10,
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
        ax.set_xlabel(xlabel, fontsize=axis_size)
    if axis_size:
        ax.tick_params(labelsize=axis_size)
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


B = K.m  # magnitude of a quantity in a unit

# ---------------------------------------------------------------------------
# 1. Hook: the memory ladder (log scale), and its return in the synthesis
# ---------------------------------------------------------------------------
ladder_rows = [
    dict(label=f"{val('p175b_b')}B, FP16", value=B(K.mem_175b, "byte"), color="memory",
         text=f"{val('mem175b_gb')} GB"),
    dict(label=f"{val('p7b_b')}B, FP16", value=B(K.mem_7b, "byte"), color="danger", text=f"{val('mem7b_gb')} GB",
         over=B(K.mem_7b_int4, "byte"), over_color="data", over_text=f"INT4: {val('mem7b_int4_gb')} GB"),
    dict(label="Phone RAM", value=B(K.phone_ram, "byte"), color="neutral", text=f"{val('phone_ram_gb')} GB"),
    dict(label="MCU SRAM", value=B(K.tiny_ram, "byte"), color="neutral", text=f"{val('mcu_sram_kb')} KB",
         over=B(K.spotter_int8, "byte"), over_color="data", over_text=f"spotter: {val('spotter_int8_kb')} KB"),
]
ladder_kw = dict(log=True, xlim=(5e4, 3e14), size=(4.2, 2.9),
                 xlabel="Bytes (log scale)",
                 xticks=[(1e6, "1 MB"), (1e9, "1 GB"), (1e12, "1 TB")])
bars(out / "ladder_step1.svg", ladder_rows, [1, 0, 1, 1], **ladder_kw)
bars(out / "ladder_step2.svg", ladder_rows, [1, 1, 1, 1], **ladder_kw)
# ---------------------------------------------------------------------------
# 2. Budget: time to stream the weights once on the phone's 51.2 GB/s bus
# ---------------------------------------------------------------------------
stream_rows = [
    dict(label="MobileNetV2, FP32", value=B(K.MNV2_READ.read_time, "ms"), color="memory",
         text=f"{val('mnv2_read_ms')} ms: {val('mnv2_read_share_pct')}% of a frame"),
    dict(label="ResNet-50, FP32", value=B(K.R50_READ.read_time, "ms"), color="memory",
         text=f"{val('r50_read_ms')} ms"),
    dict(label="GPT-2 XL, FP16", value=B(K.GPT2_READ.read_time, "ms"), color="danger",
         text=f"{val('gpt2_read_ms')} ms per token"),
    dict(label="GPT-2 XL, INT4", value=B(K.GPT2_INT4_READ.read_time, "ms"), color="data",
         text=f"{val('gpt2_int4_read_ms')} ms per token"),
]
for k in range(1, 5):
    bars(out / f"stream_step{k}.svg", stream_rows, [i < k for i in range(4)],
         xlim=(0, 80), vlines=[(B(K.BLUR.frame_budget, "ms"), f"{val('frame_budget_ms')} ms camera frame",
                                "compute")],
         xlabel=f"Time to read every weight once at {val('phone_bw')} GB/s (ms)")

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
    ax.text(3, -0.5, f"Dense:\n{W.size} weights", ha="center", va="top", fontsize=12,
            fontweight="bold", color=C["text"])
    if k >= 2:
        grid(ax, 11, W, mask=mag >= thresh)
        ax.text(14, -0.5, f"Unstructured:\n{int((mag < thresh).sum())} zeros", ha="center",
                va="top", fontsize=12, fontweight="bold", color=C["danger"])
    if k >= 3:
        grid(ax, 22, W[:, keep_cols])
        ax.text(23.5, -0.5, f"Structured:\n{W.shape[0]}×{len(keep_cols)}, dense", ha="center",
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
    dict(label="Dense FP32", value=num("fp32_b"), color="neutral", text=f"{val('fp32_b')} B"),
    dict(label=f"+ {val('idx16_bits')}-bit index", value=num("wear_nz_b"), color="danger",
         text=f"{val('wear_nz_b')} B: {val('wear_values_mb')} MB → {val('wear_stored_mb')} MB"),
    dict(label="CSR", value=num("csr_nz_b"), color="memory",
         text=f"{val('csr_nz_b')} B: wins above {val('csr_even_pct')}% zeros"),
    dict(label="COO", value=num("coo_nz_b"), color="memory",
         text=f"{val('coo_nz_b')} B: above {val('coo_even_pct')}%"),
]
bars(out / "sparsefmt_step1.svg", fmt_rows, [1, 1, 0, 0], xlim=(0, 30),
     xlabel="Bytes stored per nonzero weight", size=(4.4, 2.3),
     label_size=14, text_size=14, axis_size=13)
bars(out / "sparsefmt_step2.svg", fmt_rows, [1, 1, 1, 1], xlim=(0, 30),
     xlabel="Bytes stored per nonzero weight", size=(4.4, 2.3),
     label_size=14, text_size=14, axis_size=13)

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
body += text(10, 385, f"Design peak: {val('mishra_peak')}× math throughput", 24, C["text"], "bold")
body += text(10, 418, "(vendor specification)", 22, C["neutral"])
svg(out / "sparse24.svg", 420, 440, body)

gemm_rows = [dict(label=f"K = {r['k']:,.0f}", value=r["speedup"], color="compute", text=f"≈{r['speedup']:.1f}×")
             for r in K.GEMM]
fig, ax = plt.subplots(figsize=(4.6, 2.4))
for i, r in enumerate(gemm_rows):
    y = len(gemm_rows) - 1 - i
    ax.barh(y, r["value"], color=C["compute"], height=0.62, zorder=2)
    ax.text(r["value"] - 0.04, y, r["text"], ha="right", va="center", fontsize=13,
            color="white", fontweight="bold", zorder=3)
    ax.text(-0.02, y, r["label"], transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=13, fontweight="bold", color=C["text"])
ax.axvline(1.0, color=C["neutral"], lw=1.5, ls="--", zorder=1)
ax.axvline(num("mishra_peak"), color=C["danger"], lw=1.5, ls="--")
ax.text(1.0, 2.55, "dense ", color=C["neutral"], fontsize=10, fontweight="bold", ha="right")
ax.text(num("mishra_peak"), 2.55, f" {val('mishra_peak')}× design peak", color=C["danger"], fontsize=10,
        fontweight="bold")
ax.set_xlim(0, 2.6)
ax.set_ylim(-0.6, 2.9)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel(f"Measured 2:4 speedup over dense\n(A100, INT8 GEMM, M = N = {val('mishra_mn')})")
sp.save(fig, out / "gemm24.svg")

# ---------------------------------------------------------------------------
# 7. Channel pruning on the tile grid (SVG) and MobileNetV2 width 0.75 bars
# ---------------------------------------------------------------------------
body = ""
T250, T248 = K.TILE250, K.TILE248
rows = [(f"{T250.channels} → {T250.pruned_channels} channels", T250, C["danger"],
         f"last tile: {T250.last_tile_lanes} of {T250.tile} lanes busy"),
        (f"{T248.channels} → {T248.pruned_channels} channels", T248, C["data"], "every tile full")]
for r, (lab, tl, color, note) in enumerate(rows):
    y = 10 + r * 215
    body += text(10, y + 30, lab, 30, C["text"], "bold")
    for t in range(3):  # the last three tiles
        x0 = 10 + t * 180
        fill_n = tl.tile if t < 2 else tl.last_tile_lanes
        for c in range(tl.tile):
            on = c < fill_n
            body += box(x0 + c * 20, y + 50, 18, 60,
                        C["compute_l"] if on else "white",
                        C["compute"] if on else "#BBBBBB", 1.5, 0,
                        None if on else "4,3")
        body += text(x0 + 80, y + 142, f"tile {tl.tiles - 2 + t}", 24, C["neutral"], "normal", "middle")
    body += text(10, y + 186, note, 28, color, "bold")
svg(out / "tiles.svg", 560, 430, body)

fig, ax = plt.subplots(figsize=(4.4, 2.6))
groups = [("Parameters", num("mnv2_params_m"), num("w075_params_m"), "M", f"{val('w075_param_pct')}%"),
          ("MFLOP", num("mnv2_mflop"), num("w075_mflop"), "", f"{val('w075_flop_pct')}%")]
for gi, (name, full, slim, unit, pct) in enumerate(groups):
    y = 1 - gi
    ax.barh(y + 0.2, 1.0, height=0.36, color=C["neutral"])
    ax.barh(y - 0.2, slim / full, height=0.36, color=C["compute"])
    ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=13, fontweight="bold", color=C["text"])
    ax.text(1.02, y + 0.2, f"{K.BLUR.full.width_multiplier:.1f}: {full:g}{unit}", va="center", fontsize=12,
            color=C["neutral"], fontweight="bold")
    ax.text(slim / full + 0.02, y - 0.2, f"{val('w075_width')}: {slim:g}{unit} ({pct})",
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
    dict(label="Parameters", value=num("distil_param_kept_pct"), color="data",
         text=f"{val('distil_param_kept_pct')}% ({val('distil_param_cut_pct')}% fewer)"),
    dict(label="Inference speed", value=num("distil_speed_rel_pct"), color="compute",
         text=f"{val('distil_speed_rel_pct')}% ({val('distil_speed_x')}× faster)"),
    dict(label="Language score", value=num("distil_kept_pct"), color="memory", text=f"{val('distil_kept_pct')}%"),
]
bars(out / "distilbert.svg", distil_rows, [1, 1, 1], xlim=(0, 260), size=(4.4, 2.0),
     xticks=[(0, "0"), (100, "100 = BERT"), (200, "200")], label_size=15,
     text_size=15, axis_size=14, xlabel="DistilBERT relative to BERT (%)")

# Soft targets (variant of fig-kd-targets): hard label, teacher at T = 1,
# then the same teacher softened at T (Scenarios.TeacherSoftTargetsT4).
soft1 = [int(val(f"kd_{c}_t1")) for c in K.SOFT.labels]
soft4 = [int(val(f"kd_{c}_t")) for c in K.SOFT.labels]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    cls = [c.capitalize() for c in K.SOFT.labels]
    hard = [100 if i == 0 else 0 for i in range(len(cls))]  # one-hot label
    xs = np.arange(3)
    wbar = 0.26
    series = [(hard, "neutral", -wbar, 1.0),
              (soft1, "compute", 0.0, 1.0 if k >= 2 else 0.0),
              (soft4, "data", wbar, 1.0 if k >= 3 else 0.0)]
    for vals, ck, dx, a in series:
        ax.bar(xs + dx, vals, width=wbar * 0.95, color=C[ck], alpha=a)
        for x, v in zip(xs, vals):
            ax.text(x + dx, v + 2, f"{v}", ha="center", fontsize=11, color=C[ck],
                    fontweight="bold", alpha=a)
    ax.text(2.45, 94, "label", ha="right", fontsize=13, color=C["neutral"], fontweight="bold")
    ax.text(2.45, 81, "teacher, T = 1", ha="right", fontsize=13, color=C["compute"],
            fontweight="bold", alpha=1.0 if k >= 2 else 0.0)
    ax.text(2.45, 68, f"teacher, T = {val('kd_temp')}", ha="right", fontsize=13, color=C["data"],
            fontweight="bold", alpha=1.0 if k >= 3 else 0.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(cls, fontsize=13)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Probability (%)")
    sp.save(fig, out / f"kd_step{k}.svg")

# ---------------------------------------------------------------------------
# 11. Fusion: six transfers against two (SVG, 2 steps)
# ---------------------------------------------------------------------------
for k in (1, 2):
    body = text(10, 34, f"Unfused: {val('fu_unfused_trips')} transfers, {val('fu_unfused_mb')} MB", 30,
                C["danger"], "bold")
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
    body += text(10, 440, f"Fused: {val('fu_fused_trips')} transfers, {val('fu_fused_mb')} MB "
                          f"({val('fu_cut_pct')}% less)", 30, C["data"], "bold")
    body += "</g>\n"
    svg(out / f"fusion_step{k}.svg", 760, 455, body)

# ---------------------------------------------------------------------------
# 14. Blur pipeline waterfall (variant of fig-compression-waterfall)
# ---------------------------------------------------------------------------
stages = ["FP32\nCPU", "Pruned\nCPU", "Pruned INT8\naccelerator"]
memv = [num("mnv2_fp32_mb"), num("w075_fp32_mb"), num("w075_int8_mb")]
timev = [num("cpu_frame_ms"), num("w075_frame_ms"), num("w075_int8_frame_ms")]  # illustrative
for k in (1, 2):  # step 3 (INT8) belongs to Part 2
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
    a2.axhline(num("frame_budget_ms"), color=C["danger"], lw=1.5, ls="--")
    a2.text(2.45, num("frame_budget_ms"), f" {val('frame_budget_ms')} ms\n budget", color=C["danger"], fontsize=11, fontweight="bold",
            ha="left", va="center", clip_on=False)
    fig.text(0.01, -0.02, "Illustrative scenario (Example 10.1)", fontsize=9,
             color=C["neutral"])
    fig.tight_layout(w_pad=1.5)
    sp.save(fig, out / f"waterfall_step{k}.svg")

# ---------------------------------------------------------------------------
# 15. Paper against realized, then Amdahl with the model at 20 percent
# ---------------------------------------------------------------------------
sp.hbars(out / "paper_real.svg", [
    (f"Paper: {val('paper_prune_x')}× × {val('paper_quant_x')}×", num("paper_x"), "neutral", f"{val('paper_x')}×"),
    ("Realized", num("real_x"), "danger", f"{val('real_x')}×  ({val('real_share_pct')}% of paper)"),
], "×", log=False, size=(5.8, 1.9), xlabel="Speedup (illustrative)")

for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.0))
    model_pct, rest_pct = num("am_model_pct"), num("am_rest_pct")
    m = model_pct if k == 1 else 0
    cap = f"{val('am_rest_pct')}%: 1 / {val('am_rest')} = {val('am_cap')}×"
    ax.barh(0, m, color=C["compute"], height=0.6)
    ax.barh(0, rest_pct, left=m, color=C["neutral"], height=0.6)
    if k == 1:
        ax.text(m / 2, 0, f"Model\n{val('am_model_pct')}%", ha="center", va="center", fontsize=11,
                color="white", fontweight="bold")
    ax.text(m + rest_pct / 2, 0, f"Decode, resize, display: {val('am_rest_pct')}%", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold")
    ax.text(102, 0, "100%" if k == 1 else cap, va="center",
            fontsize=12, fontweight="bold",
            color=C["text"] if k == 1 else C["danger"])
    ax.text(102, 0, cap, va="center", fontsize=12, alpha=0.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Share of the original frame latency (%)")
    ax.spines["left"].set_visible(False)
    sp.save(fig, out / f"amdahl_step{k}.svg")

# ---------------------------------------------------------------------------
# 16. Added 2026-10-17 with the Part 1 / Part 2 split
# ---------------------------------------------------------------------------
# 16a. Listing 1: magnitude pruning at threshold 0.5 keeps 4 of 9 (Sec. 10.2.1)
L1 = K.PRUNE3.matrix
CELL = 84
for k in (1, 2):
    body = text(10, 30, f"Dense: {val('lst_n')} weights", 26, C["text"], "bold")
    for i in range(3):
        for j in range(3):
            v = L1[i][j]
            body += box(10 + j * CELL, 50 + i * CELL, CELL, CELL, C["compute_l"],
                        C["compute"], 2, 0)
            body += text(10 + j * CELL + CELL / 2, 50 + i * CELL + CELL / 2 + 9,
                         f"{v:g}".replace("-", "−"), 26, C["text"], "bold", "middle")
    op = 1 if k == 2 else 0
    body += f'<g opacity="{op}">\n'
    body += arrow(285, 176, 375, 176, C["neutral"], 3)
    body += text(330, 160, f"|w| ≥ {val('lst_threshold')}", 22, C["neutral"], "bold", "middle")
    body += text(390, 30, f"Pruned: {val('lst_kept_n')} of {val('lst_n')} kept", 26, C["data"], "bold")
    for i in range(3):
        for j in range(3):
            v = L1[i][j]
            keep = abs(v) >= K.PRUNE3.threshold
            body += box(390 + j * CELL, 50 + i * CELL, CELL, CELL,
                        C["data_l"] if keep else "white",
                        C["data"] if keep else "#999999", 2, 0)
            body += text(390 + j * CELL + CELL / 2, 50 + i * CELL + CELL / 2 + 9,
                         f"{v:g}".replace("-", "−") if keep else "0", 26,
                         C["text"] if keep else "#999999",
                         "bold" if keep else "normal", "middle")
    body += "</g>\n"
    svg(out / f"mask_step{k}.svg", 660, 310, body)

# 16b. The chapter's 4x4 sparse matrix-vector product (Sec. 10.4.2)
M4 = K.SPMV.matrix
CELL = 74
for k in (1, 2):
    body = ""
    for i in range(4):
        for j in range(4):
            v = M4[i][j]
            live = k == 1 or v != 0
            body += box(20 + j * CELL, 20 + i * CELL, CELL, CELL,
                        C["compute_l"] if live else "white",
                        C["compute"] if live else "#BBBBBB", 2, 0)
            body += text(20 + j * CELL + CELL / 2, 20 + i * CELL + CELL / 2 + 6,
                         f"{v:g}", 26, C["text"] if live else "#AAAAAA",
                         "bold" if live else "normal", "middle")
            if k == 2 and v != 0:
                body += text(20 + j * CELL + CELL - 5, 20 + i * CELL + 22, f"c{j}", 19,
                             C["memory"], "bold", "end")
    body += text(340, 175, "×", 40, C["text"], "bold", "middle")
    for i in range(4):
        body += box(370, 20 + i * CELL, CELL, CELL, C["memory_l"], C["memory"], 2, 0)
        body += text(370 + CELL / 2, 20 + i * CELL + CELL / 2 + 8, f"x{i + 1}", 24,
                     C["text"], "bold", "middle")
    msg = (f"Dense: {val('spmv_dense')} multiply-accumulates" if k == 1 else
           f"Sparse: {val('spmv_nnz')} products + {val('spmv_nnz')} column indices")
    body += text(20, 360, msg, 26, C["compute"] if k == 1 else C["memory"], "bold")
    svg(out / f"spmv_step{k}.svg", 600, 380, body)

# 16c. Where sparsity starts to pay: break-evens on one sparsity axis
fig, ax = plt.subplots(figsize=(5.8, 2.0))
ax.axvspan(80, 90, color=C["danger_l"])
ax.text(85, 1.75, "unstructured kernel\ncan lose to dense\nuntil 80 to 90%", ha="center",
        va="center", fontsize=11, fontweight="bold", color=C["danger"])
for x, lab, ck, y in ((num("csr_even_pct"), f"CSR bytes: {val('csr_even_pct')}%", "memory", 1.05),
                      (num("coo_even_pct"), f"COO bytes: {val('coo_even_pct')}%", "memory", 0.5),
                      (50, "2:4 pattern: exactly 50%,\nhardware path", "data", -0.75)):
    ax.plot([x], [0], "o", ms=10, color=C[ck], zorder=5)
    ax.plot([x, x], [0, y - 0.2 if y > 0 else y + 0.35], color=C[ck], lw=1.2)
    ax.text(x, y, lab, ha="center", va="center", fontsize=11, fontweight="bold", color=C[ck])
ax.axhline(0, color=C["text"], lw=1.5)
ax.set_xlim(0, 100)
ax.set_ylim(-1.3, 2.5)
ax.set_yticks([])
for side in ("left", "bottom"):
    ax.spines[side].set_visible(False)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.tick_params(axis="x", length=0, pad=-6)
ax.set_xlabel("Share of weights that are zero")
sp.save(fig, out / "breakeven.svg")

# 16e. Multiply-accumulates per token in one transformer layer (Sec. 10.6.1)
mlp_rows = [
    dict(label="MLP block", value=num("mlp_d2"), color="compute", text=f"{val('mlp_d2')}d²: about two thirds"),
    dict(label="Q, K, V, O projections", value=num("proj_d2"), color="memory",
         text=f"{val('proj_d2')}d²: just under a third"),
    dict(label="LayerNorm", value=0.05, color="neutral", text="almost none"),
]
bars(out / "mlpshare.svg", mlp_rows, [1, 1, 1], xlim=(0, 14), size=(5.0, 1.9),
     xticks=[(0, "0"), (num("proj_d2"), f"{val('proj_d2')}d²"), (num("mlp_d2"), f"{val('mlp_d2')}d²")],
     xlabel="Multiply-accumulates per token, per layer")


# ---------------------------------------------------------------------------
# 17. Added 2026-10-18 (visual audit): real data and real tool output
# ---------------------------------------------------------------------------
# 17a. Smallest model at >= 60 percent MMLU, redrawn at slide size from the
# same data the book figure uses (Epoch AI Benchmarking Hub, CC BY 4.0, via
# mlsysim Literature.Series.MMLUDenseModels; chapter fig-smallest-capable-model).
import datetime as dt  # noqa: E402

from mlsysim.core.units import SEC_PER_DAY, SEC_PER_YEAR_LEAP  # noqa: E402

_rows, _front, _bar = K.SMALLEST.rows, K.SMALLEST.frontier, K.SMALLEST.threshold


def _yr(d):
    d = dt.date.fromisoformat(d)
    return d.year + (d.timetuple().tm_yday - 1) / (SEC_PER_YEAR_LEAP / SEC_PER_DAY)


fig, ax = plt.subplots(figsize=(3.7, 2.15))
for hi, ck, sz in ((False, "neutral_l", 16), (True, "memory", 22)):
    pts = [r for r in _rows if (r["mmlu"] >= _bar) == hi]
    ax.scatter([_yr(r["date"]) for r in pts], [r["parameters"] for r in pts], s=sz,
               color=C[ck], zorder=2 + hi)
fx = [_yr(r["date"]) for r in _front] + [2025.1]
fy = [r["parameters"] for r in _front] + [_front[-1]["parameters"]]
ax.step(fx, fy, where="post", color=C["danger"], lw=2.2, zorder=4)
labels = {"Gopher 280B": (6, 2, "left"), "Chinchilla 70B": (6, 4, "left"),
          "LLAMA 2 34B": (6, 4, "left"), "Mistral 7B": (6, 4, "left"),
          "Qwen2.5-Coder-3B": (-4, -16, "right")}
for _r in _front:
    if _r["model"] in labels:
        dx, dy, ha = labels[_r["model"]]
        ax.annotate(_r["model"].replace("LLAMA", "Llama").replace("Qwen2.5-Coder-3B", "Qwen2.5-Coder 3B"),
                    (_yr(_r["date"]), _r["parameters"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=11, fontweight="bold",
                    color=C["danger"], ha=ha, zorder=5,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.4))
ax.text(2021.55, 1.4e8, f"Blue: MMLU ≥ {val('mmlu_bar_pct')}%.  Gray: below.", fontsize=11, color=C["memory"])
ax.set_yscale("log")
ax.set_ylim(1e8, 1e12)
ax.set_yticks([1e9, 1e10, 1e11, 1e12])
ax.set_yticklabels(["1B", "10B", "100B", "1T"])
ax.minorticks_off()
ax.set_xticks([2022, 2023, 2024, 2025])
ax.set_xlim(2021.5, 2025.2)
ax.set_ylabel("Parameters (log)")
ax.grid(axis="y", color=C["grid"], linewidth=0.6)
sp.save(fig, out / "smallest_capable.svg")

# 17b. Accuracy-efficiency frontier: ImageNet top-1 against Pixel 1 float
# latency, one large core, TFLite benchmark tool (Howard et al. 2019, Table 3).
pts = [(r["model"], r["top1"], r["pixel1_ms"]) for r in K.MNV3]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.6, 1.95))
    for n, a, t in pts:
        on_f = n in K.FRONTIER
        ck = "memory" if on_f else "neutral"
        if n == "V2 1.0":
            ck = "danger"
        ax.plot([t], [a], "o", ms=9, color=C[ck], zorder=3)
    lab = {"V2 1.0": ("MobileNetV2 1.0: the blur model", 5, -16, "left", "danger"),
           "V3-Large 0.75": ("V3-Large 0.75", -8, 6, "right", "memory"),
           "V3-Large 1.0": ("V3-Large 1.0", -8, 6, "right", "memory"),
           "V3-Small 1.0": ("V3-Small 1.0", 8, -12, "left", "memory"),
           "MnasNet-A1": ("MnasNet-A1", 0, 8, "center", "neutral")}
    for n, a, t in pts:
        if n in lab:
            s_, dx, dy, ha, ck = lab[n]
            ax.annotate(s_, (t, a), xytext=(dx, dy), textcoords="offset points",
                        fontsize=11, fontweight="bold", color=C[ck], ha=ha)
    fr = sorted((t, a) for n, a, t in pts if n in K.FRONTIER)
    al = 1.0 if k == 2 else 0.0
    ax.plot([p[0] for p in fr], [p[1] for p in fr], "-", color=C["memory"], lw=2.0,
            alpha=al, zorder=2)
    ax.text(13, 71.0, "frontier", fontsize=12, fontweight="bold", color=C["memory"],
            alpha=al, rotation=0)
    ax.annotate("", (K.v3l75["pixel1_ms"] + 1.5, K.v3l75["top1"] - 0.2),
                (K.v2["pixel1_ms"] - 1.5, K.v2["top1"] + 0.2),
                arrowprops=dict(arrowstyle="-|>", color=C["data"], lw=2.0, alpha=al))
    ax.text(44, 69.0, f"{val('v3_speedup')}× faster,\n+{val('v3_gain_pts')} points",
            fontsize=11, fontweight="bold", color=C["data"], alpha=al, ha="left", va="center")
    ax.set_xlim(0, 80)
    ax.set_ylim(59, 77)
    ax.set_xlabel("Latency on a Pixel 1, float, one core (ms)")
    ax.set_ylabel("ImageNet top-1 (%)")
    ax.grid(color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"frontier_step{k}.svg")

# 17c. Retrain, then iterate (Han et al. 2015, Fig. 5, AlexNet top-5); points
# extracted from the published figure's vector paths, in calcs.py.
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    series = [(K.han["no_retrain"], "danger", "--", "o", "prune, no retrain", 1.0),
              (K.han["retrain"], "memory", "-", "s", "prune once, retrain", 1.0),
              (K.han["iterative"], "data", "-", "D", "prune and retrain,\nrepeated", 1.0 if k == 2 else 0.0)]
    for pts_, ck, ls, mk, name, a in series:
        xs = [x for x, _ in pts_]
        ys = [y for _, y in pts_]
        ax.plot(xs, ys, ls, marker=mk, color=C[ck], lw=2.2, ms=6, alpha=a, clip_on=True)
    ax.text(42, -1.6, "no retrain", fontsize=12, fontweight="bold", color=C["danger"])
    ax.text(70, -3.9, "prune once,\nretrain", fontsize=12, fontweight="bold", color=C["memory"],
            ha="right")
    ax.text(96.6, -3.1, "iterative", fontsize=12, fontweight="bold", color=C["data"],
            ha="left", alpha=1.0 if k == 2 else 0.0)
    ax.axhline(0, color=C["text"], lw=0.8)
    ax.set_xlim(40, 100)
    ax.set_ylim(-4.5, 0.6)
    ax.set_xticks([40, 60, 80, 90, 100])
    ax.set_xticklabels(["40%", "60%", "80%", "90%", "100%"])
    ax.set_xlabel("Weights pruned")
    ax.set_ylabel("Top-5 change (points)")
    ax.grid(color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"han_step{k}.svg")

# 17d. Measured on this CPU (measure_cpu.py, values via calcs.py)
rows_ = K.CPU["rows"]
base_ = rows_[0]
names_ = ["Width 1.0", f"Width 1.0,\n{val('cpu_zeros_pct')}% zeros", "Width 0.75"]
fig, ax = plt.subplots(figsize=(5.8, 2.4))
for i, (r, nm) in enumerate(zip(rows_, names_)):
    y = 2 - i
    fp = 100 * r["flops"] / base_["flops"]
    tm = 100 * r["median_ms"] / base_["median_ms"]
    ax.barh(y + 0.19, fp, height=0.34, color=C["neutral"])
    ax.barh(y - 0.19, tm, height=0.34, color=C["compute"])
    ax.text(fp + 1.5, y + 0.19, f"FLOPs {fp:.0f}%", va="center", fontsize=12,
            fontweight="bold", color=C["neutral"])
    ax.text(tm + 1.5, y - 0.19, f"time {tm:.0f}%  ({r['median_ms']:.1f} ms)", va="center",
            fontsize=12, fontweight="bold", color=C["compute"])
    ax.text(-0.01, y, nm, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=12, fontweight="bold", color=C["text"])
ax.set_xlim(0, 150)
ax.set_xticks([0, 50, 100])
ax.set_xticklabels(["0", "50%", "100%"])
ax.set_ylim(-0.6, 2.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel(f"Relative to width 1.0, dense (median of {val('cpu_runs')} runs)")
sp.save(fig, out / "cpu_measure.svg")

# ---------------------------------------------------------------------------
# 16. Deep Compression cliffs, redrawn at slot size (panel fix L10). Data:
# the points of the chapter's fig-compression-methods (TikZ table in
# 10_model_compression.qmd), from Han, Mao, Dally 2016 (AlexNet on ImageNet).
# Marker-less points in the book table are the lines' extensions off the axis.
# ---------------------------------------------------------------------------
_cliff_color = {"Pruning + quantization": "data", "Pruning only": "compute",
                "Quantization only": "memory", "SVD": "danger"}
CLIFF = {name: (ck, [(r["size_pct"], r["accuracy_change"]) for r in K.SERIES.DeepCompressionMethods.entries
                     if r["method"] == name])
         for name, ck in _cliff_color.items()}
fig, ax = plt.subplots(figsize=(5.8, 3.0))
label_at = {"Pruning + quantization": (2.2, 0.3, "left"), "Pruning only": (10.0, 0.3, "left"),
            "Quantization only": (15.3, 0.3, "left"), "SVD": (17.2, -3.6, "left")}
for name, (ck, pts) in CLIFF.items():
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=C[ck], lw=2.6)
    ax.plot(xs[1:], ys[1:], "o", ms=5, color=C[ck], mfc="white", mew=1.8)
    lx, ly, ha = label_at[name]
    ax.text(lx, ly, name, color=C[ck], fontsize=12, fontweight="bold", ha=ha)
ax.axhline(0, color=C["neutral"], lw=1)
ax.set_xlim(2, 20)
ax.set_ylim(-4.5, 0.9)
ax.set_xticks([2, 5, 10, 15, 20])
ax.set_xticklabels(["2%", "5%", "10%", "15%", "20%"])
ax.set_yticks([-4, -3, -2, -1, 0])
ax.set_yticklabels(["-4", "-3", "-2", "-1", "0"])
ax.set_xlabel("Model size kept (% of original)")
ax.set_ylabel("Accuracy change (points)")
ax.grid(color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "cliff_methods.svg")

print("figures written")
