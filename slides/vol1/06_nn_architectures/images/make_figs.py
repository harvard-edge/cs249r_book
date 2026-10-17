#!/usr/bin/env python3
"""Data plots for Lecture 6 (Network Architectures).

Every value is sourced from the rendered chapter (Network Architectures)
unless noted:
  - Energy per weight at batch one: 160 pJ per byte read from DRAM, so
    640 pJ per FP32 weight, against 4.6 pJ per FP32 multiply-accumulate;
    about 139x (Sec. 6.2.5, after Horowitz 2014).
  - Weight sharing: 3x3x64x64 = 36,864 weights against 224^2 x 64 x 64,
    about 205 million, about 5,575x (Definition 6.3).
  - Weight-only arithmetic intensity, FP32 weights: dense layer at batch
    one 0.5 FLOP/byte (equation 2); ResNet-50 80.1, MobileNetV2 42.8,
    KWS (DS-CNN) 25, GPT-2 XL decode 0.50 FLOP/byte (Table 8).
  - RNN state: 2 KB for a 512-dimensional FP32 state (Sec. 6.5.5).
    GPT-2 XL KV cache: 307.2 KB per token in FP16 (Sec. 6.7.4).
  - Materialized attention scores: S^2 per head at 2 bytes (FP16), 12 heads
    (the head count of Napkin Math 6.1 and the Sec. 6.6 self-check).
  - Context memory (slide redraw of the chapter's context-memory figure):
    GPT-2 XL weights 3 GB in FP16, KV cache 307.2 KB per token, one layer's
    scores for 25 heads in FP16 (GPT-2 XL shape: Radford et al. 2019), one
    accelerator of 80 GiB, so scores fill it near 41K tokens and the KV
    cache passes the weights near 10K, as the chapter states.
  - Capacity wall: 100M items x 128 x 4 bytes = 51.2 GB; two tables
    102.4 GB; 80 GB A100 (Napkin Math 6.2).
  - Added 2026-10-17 for the Part 1 / Part 2 split:
    MNIST detectors: 32 dense features 25,088 weights against 32 shared
    3x3 filters, 288 weights, 87x (Example 6.1).
    Depthwise separable count for 3x3, 64 channels in and out, per output
    position: standard 36,864 MACs; depthwise 576 + pointwise 4,096 = 4,672
    (Sec. 6.3.5 formula; ratio 7.9x derived).
    KWS DS-CNN: 200K parameters, 800 KB in FP32, 200 KB in INT8
    (Lighthouse 6.3, Table 10).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

out = Path(__file__).parent
C = sp.C

# Step builds must keep one canvas size, or the slide rescales the figure
# between steps. The shared save() crops tightly, and a value label that
# appears only in a later step widens that crop, so plots whose labels
# extend past the axes are saved here with fixed margins instead.
_tight_save = sp.save
MARGINS = {}


def _fixed_save(fig, path):
    if MARGINS:
        fig.subplots_adjust(**MARGINS)
        fig.savefig(path, format="svg", facecolor="white")
        sp.plt.close(fig)
    else:
        _tight_save(fig, path)


sp.save = _fixed_save

# --- Dense segment: moving a weight against computing with it -------------
MARGINS.update(left=0.42, right=0.78, bottom=0.30, top=0.97)
sp.hbars_steps(out / "energy_weight", [
    ("FP32 multiply-add", 4.6, "compute", "4.6 pJ"),
    ("Read FP32 weight, DRAM", 640, "memory", "640 pJ = 139×"),
], "pJ", xlabel="Energy (picojoules, log scale; 45 nm, Horowitz 2014)",
    size=sp.SHORT)

# --- Convolution segment: weight sharing ----------------------------------
MARGINS.update(left=0.26, right=0.66, bottom=0.30, top=0.97)
sp.hbars_steps(out / "params", [
    ("Dense layer", 205e6, "danger", "about 205 million"),
    ("3×3 convolution", 36864, "compute", "36,864: 5,575× fewer"),
], "weights", xlabel="Weights for 64 features from a 224×224×64 input (log scale)",
    size=sp.SHORT)

# --- Convolution segment: reuse raises intensity --------------------------
MARGINS.update(left=0.37, right=0.70, bottom=0.30, top=0.97)
sp.hbars_steps(out / "intensity_cnn", [
    ("Dense layer, batch one", 0.5, "memory", "0.5"),
    ("ResNet-50", 80.1, "compute", "80.1: 160× more"),
], "FLOP/byte", xlabel="Arithmetic intensity (FLOP per weight byte, FP32, log scale)",
    size=sp.SHORT)


# --- Recurrence segment: fixed state against a growing cache --------------
def state_plot(step):
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.3))
    s = np.logspace(1, 4, 50)
    ax.plot(s, np.full_like(s, 2e3), color=C["memory"], lw=3.5)
    ax.text(12, 2e3 * 1.9, "RNN state: 2 KB at any length", color=C["memory"],
            fontsize=12, fontweight="bold")
    if step >= 2:
        kv = 307.2e3 * s
        ax.plot(s, kv, color=C["danger"], lw=3.5)
        ax.text(10500, 3.07e9, "3.1 GB", color=C["danger"], fontsize=12,
                fontweight="bold", va="center")
        ax.text(150, 3e8, "GPT-2 XL KV cache:\n307.2 KB per token",
                color=C["danger"], fontsize=12, fontweight="bold", ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 1e4)
    ax.set_ylim(5e2, 3e10)
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.set_yticks([1e3, 1e6, 1e9])
    ax.set_yticklabels(["1 KB", "1 MB", "1 GB"])
    ax.minorticks_off()
    ax.set_xlabel("Sequence length (tokens, log scale)")
    ax.set_ylabel("State per sequence")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"state_memory_step{step}.svg")


MARGINS.update(left=0.14, right=0.86, bottom=0.21, top=0.96)
for k in (1, 2):
    state_plot(k)
MARGINS.clear()


# --- Attention segment: the S-squared parabola ----------------------------
def quad_plot(step):
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.3))
    s = np.linspace(0, 8800, 200)
    gb = s ** 2 * 12 * 2 / 1e9
    ax.plot(s, gb, color=C["compute"], lw=3.5)
    pts = [(4096, "4K tokens: 0.40 GB", C["data"]),
           (8192, "8K tokens: 1.61 GB = 4×", C["danger"])]
    for k, (x, label, col) in enumerate(pts[:max(0, step - 1)]):
        y = x ** 2 * 24 / 1e9
        ax.plot([x, x], [0, y], ls="--", color=col, lw=1.2)
        ax.plot([x], [y], "o", ms=11, color=col, mec="white", mew=1.5, zorder=5)
        ax.text(x - 450, y + 0.05, label, ha="right", va="bottom", color=col,
                fontsize=12, fontweight="bold")
    ax.set_xlim(0, 8800)
    ax.set_ylim(0, 2.0)
    ax.set_xticks([0, 2048, 4096, 6144, 8192])
    ax.set_xticklabels(["0", "2K", "4K", "6K", "8K"])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Scores, one layer (GB)")
    ax.text(0.01, 0.97, "12 heads, FP16, scores materialized", transform=ax.transAxes,
            fontsize=10, color=C["neutral"], va="top")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"quadratic_step{step}.svg")


for k in (1, 2, 3):
    quad_plot(k)


# --- Attention segment: what grows with context (GPT-2 XL) ----------------
def context_plot(step):
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.35))
    s = np.logspace(np.log10(512), np.log10(131072), 200)
    dev = 80 * 2 ** 30 / 1e9
    ax.axhline(dev, color=C["danger"], lw=1.3, ls=":")
    ax.text(600, dev / 2.2, "One 80 GB accelerator", color=C["danger"], fontsize=11)
    ax.plot(s, np.full_like(s, 3.0), color=C["memory"], lw=3.2)
    ax.text(600, 3.0 * 1.4, "Weights, 3 GB (fixed)", color=C["memory"],
            fontsize=12, fontweight="bold")
    if step >= 2:
        ax.plot(s, 307.2e3 * s / 1e9, color=C["compute"], lw=3.2)
        x = 3.0 / 307.2e-6
        ax.plot([x], [3.0], "o", ms=9, color=C["compute"], mec="white", zorder=5)
        ax.text(x * 1.25, 0.06, "KV cache passes\nthe weights near 10K",
                color=C["compute"], fontsize=11, fontweight="bold")
    if step >= 3:
        ax.plot(s, 25 * s ** 2 * 2 / 1e9, color=C["danger"], lw=3.2, ls="--")
        x = np.sqrt(80 * 2 ** 30 / 50)
        ax.plot([x], [dev], "o", ms=9, color=C["danger"], mec="white", zorder=5)
        ax.text(x / 1.9, dev * 2.2, "One layer's scores\nfill it near 41K",
                color=C["danger"], fontsize=11, fontweight="bold", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(512, 131072)
    ax.set_ylim(0.01, 2000)
    ax.set_xticks([512, 2048, 8192, 32768, 131072])
    ax.set_xticklabels(["512", "2K", "8K", "32K", "128K"])
    ax.set_yticks([0.01, 1, 100])
    ax.set_yticklabels(["10 MB", "1 GB", "100 GB"])
    ax.minorticks_off()
    ax.set_xlabel("Context length (tokens, log scale)")
    ax.set_ylabel("Memory (log scale)")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"context_memory_step{step}.svg")


for k in (1, 2, 3):
    context_plot(k)


# --- Embeddings segment: the capacity wall ---------------------------------
def capacity_plot(step):
    fig, ax = sp.plt.subplots(figsize=(5.8, 1.9))
    rows = [("Item table", 51.2, "memory", "51.2 GB: 64% of the device"),
            ("Item + user tables", 102.4, "danger", "102.4 GB: does not fit")]
    for i, (label, v, ck, txt) in enumerate(rows):
        y = 1 - i
        shown = i < step
        ax.barh(y, v if shown else 0, color=C[ck], height=0.6)
        ax.text(-1.5, y, label, ha="right", va="center", fontsize=13,
                fontweight="bold", color=C["text"])
        if shown:
            ax.text(max(v, 80) + 3, y, txt, va="center", fontsize=12,
                    fontweight="bold", color=C[ck])
    ax.axvline(80, color=C["danger"], lw=2, ls="--")
    ax.text(82, 1.62, "80 GB A100", ha="left", fontsize=11, color=C["danger"],
            fontweight="bold")
    ax.set_xlim(0, 180)
    ax.set_ylim(-0.5, 1.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Memory (GB; 128-wide FP32 rows, 100M rows per table)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"capacity_step{step}.svg")


for k in (1, 2):
    capacity_plot(k)


# --- Choosing segment: five lighthouses on one intensity scale ------------
def lighthouse_plot():
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.3))
    rows = [("ResNet-50", 80.1, "compute", "80.1"),
            ("MobileNetV2", 42.8, "compute", "42.8"),
            ("KWS (DS-CNN)", 25, "compute", "25"),
            ("GPT-2 XL, decode", 0.5, "memory", "0.50"),
            ("DLRM", None, "network", "n/a: capacity decides first")]
    n = len(rows)
    for i, (label, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        ax.text(0.17, y, label, ha="right", va="center", fontsize=13,
                fontweight="bold", color=C["text"])
        if v is None:
            ax.text(0.25, y, txt, va="center", fontsize=12, fontweight="bold",
                    color=C[ck], style="italic")
        else:
            ax.barh(y, v - 0.2, left=0.2, color=C[ck], height=0.6)
            ax.text(v * 1.15, y, txt, va="center", fontsize=12,
                    fontweight="bold", color=C[ck])
    ax.set_xscale("log")
    ax.set_xlim(0.2, 400)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("FLOP per weight byte (FP32 weight-only proxy, log scale)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "lighthouse_intensity.svg")


lighthouse_plot()

# --- Dense segment: weights per detector, dense against shared ------------
MARGINS.update(left=0.33, right=0.70, bottom=0.30, top=0.97)
sp.hbars_steps(out / "mnist_detectors", [
    ("32 dense features", 25088, "danger", "25,088"),
    ("32 shared 3×3 filters", 288, "compute", "288: 87× fewer"),
], "weights", xlabel="Weights for 32 feature detectors on a 28×28 image (log scale)",
    size=sp.SHORT)

# --- Depth and efficiency segment: factoring a 3x3 convolution ------------
MARGINS.update(left=0.36, right=0.72, bottom=0.30, top=0.97)
sp.hbars_steps(out / "depthwise", [
    ("Standard 3×3", 36864, "danger", "36,864"),
    ("Depthwise + pointwise", 4672, "compute", "576 + 4,096 = 4,672"),
], "MACs", log=False,
    xlabel="Multiply-adds per output position (3×3 filter, 64 channels in and out)",
    size=sp.SHORT)

# --- Depth and efficiency segment: KWS weight bytes ------------------------
MARGINS.update(left=0.30, right=0.80, bottom=0.30, top=0.97)
sp.hbars(out / "kws_bytes.svg", [
    ("FP32 weights", 800, "danger", "800 KB"),
    ("INT8 weights", 200, "data", "200 KB"),
], "KB", log=False, xlabel="KWS DS-CNN, 200K weights (KB)", size=sp.SHORT)
MARGINS.clear()
print("figures written")
