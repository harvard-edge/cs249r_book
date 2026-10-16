#!/usr/bin/env python3
"""Data plots for Lecture 5 (Neural Computation).

Every value is sourced from the rendered chapter (Neural Computation) unless
noted:
  - One digit by three methods (Sec. 5.2, illustrative implementations):
    about 100 comparisons (rules), about 8,000 operations (HOG + linear SVM),
    109,184 MACs (the 784-128-64-10 network); 1,100x the rule baseline.
  - Activation slopes: sigmoid's derivative peaks at 0.25; ReLU's is 1 on
    its positive branch (Sec. 5.3.2). 0.25**20 = 9.1e-13 (footnote 1).
  - Transistor tax (Sec. 5.3.2): ReLU about 50 transistor equivalents,
    exponential datapath for sigmoid or tanh about 2,500; a stylized
    teaching ratio, labeled illustrative on the slide.
  - Forward pass at batch 32 (Table 7, Napkin Math 5.1): layer MatMuls
    6,422,528 + 524,288 + 40,960 FLOPs; bias and activation 8,192 + 4,096 +
    about 640 operations.
  - Training step FLOPs at batch 32: forward 6,987,776; weight gradients
    the same; input gradients for layers 2 and 3 only, 2 x 32 x (128 x 64 +
    64 x 10) = 565,248; total about 14.5 MFLOP, 2.08x the forward work
    (Sec. 5.4.5 and Napkin Math 5.3).
  - One epoch on two machines (Napkin Math 5.3): work 46 ms (CPU) against
    0.7 ms (accelerator); fixed per-operation overhead 225 ms against
    337.5 ms; totals about 271 ms and 338 ms.
  - Training against inference memory at batch 32, FP32 (Table 9):
    parameters 437.5 KB, saved activations 126 KB (training) or 117 KB
    (inference, rolling pair), gradients 437.5 KB, Adam state 875.1 KB;
    about 1.9 MB against about 554 KB, 3.4x.
  - Number formats (footnote 20 and Sec. 5.4.5): smallest normal 1.18e-38
    (FP32) and 6.10e-5 (FP16); FP16 maximum 65,504; BF16 keeps FP32's
    exponent range. FP32 maximum 3.4e38 and the signed 16-bit integer
    maximum 32,767 are the format definitions (IEEE 754; two's complement).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

out = Path(__file__).parent
plt = sp.plt

# ---------------------------------------------------------------------------
# One digit, three methods: operations per digit on a log ladder
sp.hbars_steps(out / "ops_ladder", [
    ("Hand-written rules", 100, "neutral", "100 comparisons"),
    ("HOG features + SVM", 8000, "neutral", "8,000 operations"),
    ("Neural network", 109184, "compute", "109,184 MACs  = 1,100× rules"),
], "ops", xlabel="Work to classify one 28×28 digit (log scale)")

# ---------------------------------------------------------------------------
# Activation slopes: how much gradient each activation passes back
z = np.linspace(-6, 6, 600)
sig = 1 / (1 + np.exp(-z))
dsig = sig * (1 - sig)
drelu = (z > 0).astype(float)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.plot(z, dsig, color=sp.C["memory"], lw=3.5)
    ax.annotate("Sigmoid slope:\nat most 0.25", (0, 0.25), xytext=(-5.8, 0.5),
                fontsize=12, fontweight="bold", color=sp.C["memory"],
                arrowprops=dict(arrowstyle="-", color=sp.C["memory"], lw=1))
    if k == 2:
        ax.plot(z[z <= 0], drelu[z <= 0], color=sp.C["compute"], lw=3.5)
        ax.plot(z[z > 0], drelu[z > 0], color=sp.C["compute"], lw=3.5)
        ax.text(0.4, 0.86, "ReLU slope: 1 when z > 0", fontsize=12,
                fontweight="bold", color=sp.C["compute"])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.05, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 1.0])
    ax.set_xlabel("Input to the activation, z")
    ax.set_ylabel("Slope f′(z)")
    ax.grid(axis="y", color=sp.C["grid"], linewidth=0.6)
    sp.save(fig, out / f"slopes_step{k}.svg")

# ---------------------------------------------------------------------------
# The transistor tax (illustrative ratio from the chapter)
sp.hbars_steps(out / "transistor_tax", [
    ("ReLU: compare and select", 50, "compute", "50"),
    ("Sigmoid: exponential", 2500, "danger", "2,500  = 50× ReLU"),
], "", xlabel="Transistor equivalents per activation unit (log scale)",
    size=sp.SHORT)

# ---------------------------------------------------------------------------
# Forward pass at batch 32: where the operations go (linear axis)
fwd = [
    ("Layer 1 MatMul", 6.422528, "compute", "6,422,528"),
    ("Layer 2 MatMul", 0.524288, "compute", "524,288"),
    ("Layer 3 MatMul", 0.040960, "compute", "40,960"),
    ("Bias + activation, all layers", 0.012928, "neutral",
     "12,928  (0.2% of the total)"),
]
sp.hbars_steps(out / "fwd_ops", fwd, "", log=False,
               xlabel="Million operations for one batch of 32 digits (total about 7 million)")

# ---------------------------------------------------------------------------
# A training step: forward plus two backward products per layer


def training_flops(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    parts = [("Forward", 6.99, sp.C["compute"]),
             ("Weight gradients", 6.99, sp.C["danger"]),
             ("Input gradients", 0.57, sp.C["network"])]
    left = 0.0
    shown = parts[:step]
    for name, v, col in shown:
        ax.barh(0, v, left=left, color=col, height=0.55, edgecolor="white", lw=1.5)
        if v > 1:
            ax.text(left + v / 2, 0, f"{name}\n{v:.2f} M", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
        left += v
    if step == 3:
        ax.annotate("Input gradients\n0.57 M (layer 1 skips it)", (left - 0.28, 0.28),
                    xytext=(left + 0.3, 0.62), fontsize=11, color=sp.C["network"],
                    fontweight="bold", ha="left",
                    arrowprops=dict(arrowstyle="-", color=sp.C["network"], lw=1))
        ax.text(left + 0.3, -0.05, "14.5 M = 2.08× forward", fontsize=13,
                fontweight="bold", color=sp.C["text"], va="center")
    ax.set_xlim(0, 21.5)
    ax.set_ylim(-0.5, 0.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Million FLOPs per training step (batch 32)")
    ax.grid(axis="x", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2, 3):
    training_flops(out / f"train_flops_step{k}.svg", k)

# ---------------------------------------------------------------------------
# One epoch on two machines: arithmetic and data movement, then overhead


def epoch_plot(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    rows = [("CPU", 46.0, 225.0), ("Accelerator", 0.7, 337.5)]
    for i, (name, work, over) in enumerate(rows):
        y = 1 - i
        ax.barh(y, work, color=sp.C["compute"], height=0.55)
        ax.text(-4, y, name, ha="right", va="center", fontsize=13,
                fontweight="bold", color=sp.C["text"])
        if step == 1:
            ax.text(work + 4, y, f"{work:g} ms", va="center", fontsize=13,
                    fontweight="bold", color=sp.C["compute"])
        else:
            ax.barh(y, over, left=work, color=sp.C["neutral_l"], height=0.55,
                    edgecolor=sp.C["neutral"], lw=0.8)
            ax.text(work + over + 4, y, f"{work + over:.0f} ms", va="center",
                    fontsize=13, fontweight="bold", color=sp.C["text"])
    if step == 1:
        ax.text(80, 0.45, "Work alone: the accelerator is 67× faster",
                fontsize=12, color=sp.C["compute"], fontweight="bold")
    else:
        ax.text(170, 0.0, "fixed overhead per operation: 99.8%", fontsize=11,
                color=sp.C["text"], va="center", ha="center")
    ax.set_xlim(0, 420)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time for one epoch of MNIST at batch 32 (ms)")
    ax.grid(axis="x", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(0.0, -0.45, "Estimate (Neural Computation, Napkin Math 5.3)",
            transform=ax.transAxes, fontsize=9, color=sp.C["neutral"])
    sp.save(fig, path)


for k in (1, 2):
    epoch_plot(out / f"epoch_step{k}.svg", k)

# ---------------------------------------------------------------------------
# Training against inference memory (Table 9), built one term at a time


def memory_plot(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    train = [("Weights", 437.5, sp.C["memory"]),
             ("Activations", 126.0, sp.C["data"]),
             ("Gradients", 437.5, sp.C["danger"]),
             ("Adam state", 875.1, sp.C["network"])]
    infer = [("Weights", 437.5, sp.C["memory"]),
             ("Activations", 117.0, sp.C["data"])]
    # inference row
    left = 0
    for name, v, col in infer:
        ax.barh(0, v, left=left, color=col, height=0.55, edgecolor="white", lw=1.5)
        left += v
    ax.text(left + 25, 0, "554 KB", va="center", fontsize=13,
            fontweight="bold", color=sp.C["text"])
    # training row, revealed term by term
    left = 0
    for name, v, col in train[:step + 1]:
        if True:
            ax.barh(1, v, left=left, color=col, height=0.55, edgecolor="white", lw=1.5)
            if v > 300:
                ax.text(left + v / 2, 1, name, ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")
            left += v
    total = left
    label = "1.9 MB  = 3.4×" if step >= 3 else f"{total:,.0f} KB"
    ax.text(total + 25, 1, label, va="center", fontsize=13,
            fontweight="bold", color=sp.C["text"])
    ax.text(437.5 / 2, 0, "Weights", ha="center", va="center", fontsize=11,
            color="white", fontweight="bold")
    ax.annotate("Saved activations", (437.5 + 63, 1.28), xytext=(420, 1.75),
                fontsize=11, color=sp.C["data"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=sp.C["data"], lw=1))
    for y, name in ((1, "Training step"), (0, "Inference")):
        ax.text(-20, y, name, ha="right", va="center", fontsize=13,
                fontweight="bold", color=sp.C["text"])
    ax.set_xlim(0, 2300)
    ax.set_ylim(-0.5, 2.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Memory for the MNIST network, batch 32, FP32 (KB)")
    ax.grid(axis="x", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2, 3):
    memory_plot(out / f"memory_step{k}.svg", k)

# ---------------------------------------------------------------------------
# Number formats: the range each format can hold (log scale), with the
# gradient left after 20 saturated sigmoid layers


def ranges_plot(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    rows = [("FP32", 1.18e-38, 3.4e38, sp.C["memory"]),
            ("BF16", 1.18e-38, 3.4e38, sp.C["memory"]),
            ("FP16", 6.10e-5, 65504, sp.C["compute"]),
            ("INT16", 1, 32767, sp.C["neutral"])]
    for i, (name, lo, hi, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, np.log10(hi) - np.log10(lo), left=np.log10(lo), color=col,
                height=0.55, alpha=0.9)
        ax.text(-41.5, y, name, ha="right", va="center", fontsize=13,
                fontweight="bold", color=sp.C["text"])
    ax.text(np.log10(65504) + 1, 1, "6.1×10⁻⁵ to 65,504", va="center",
            fontsize=11, color=sp.C["compute"], fontweight="bold")
    ax.text(np.log10(32767) + 1, 0, "max 32,767 (Ariane 501)", va="center",
            fontsize=11, color=sp.C["neutral"], fontweight="bold")
    if step >= 2:
        x = np.log10(9.1e-13)
        ax.axvline(x, color=sp.C["danger"], lw=2, ls="--")
        ax.text(x - 1.2, 3.62, "20 sigmoid layers: 9.1×10⁻¹³",
                ha="right", fontsize=11, color=sp.C["danger"], fontweight="bold")
    ticks = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"10$^{{{t}}}$" for t in ticks])
    ax.set_xlim(-41, 41)
    ax.set_ylim(-0.5, 4.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Magnitudes a format holds at full precision (log scale)")
    ax.grid(axis="x", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2):
    ranges_plot(out / f"ranges_step{k}.svg", k)

print("figures written")
