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


# ===========================================================================
# Slide-size redraws of five chapter figures. The values come from running
# the chapter's own calculation cells (the same classes the book's figures
# plot: ParameterGrowth, DepthWidthCost, MnistFlopsCalc, the MNIST layer
# sizes, and the MLSysIM hardware and model registry), so the slide and the
# book cannot disagree. Only the size, type, and labeling change: direct
# labels replace legends and every label is drawn at slide scale.
# Needs the repo's mlsysim folder and the MLSysBook venv.
# ===========================================================================
def chapter_values():
    import os
    import re
    root = Path(__file__).resolve().parents[4]
    for d in (root, root / "mlsysim"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    qmd = next((root / "books" / "vol1" / "05_nn_computation").glob("*.qmd"))
    cells = re.findall(r"```\{python\}\n(.*?)```", qmd.read_text(), re.S)
    saved = dict(plt.rcParams)
    here = os.getcwd()
    os.chdir(root)
    ns = {}
    try:
        for cell in cells:
            if "plt.show(" in cell or "setup_plot(" in cell:
                continue  # figure cells; only the calculations are needed
            exec(cell, ns)
    finally:
        os.chdir(here)
        plt.close("all")
        plt.rcParams.update(saved)
    return ns


CH = chapter_values()
K = sp.C


def param_growth():
    G = CH["ParameterGrowth"]
    dom = {"Language": K["danger"], "Vision": K["memory"], "Games": K["data"]}
    d = np.array([r["domain"] if r["domain"] in dom else "Other" for r in G.rows])
    fig, ax = plt.subplots(figsize=(5.3, 2.0))
    for k, col in list(dom.items()) + [("Other", "#BBBBBB")]:
        m = d == k
        ax.scatter(G.years[m], 10 ** G.log_params[m], s=12, color=col,
                   alpha=0.75, edgecolors="none", zorder=3 if k != "Other" else 2)
    for lo, hi, sl, ic, txt, xm, lift in (
            (1950, G.split_year, G.early_slope, G.early_intercept, G.early_doubling_str, 1975, 60),
            (G.split_year, 2026.8, G.late_slope, G.late_intercept, G.late_doubling_str, 1998, 1)):
        xx = np.linspace(lo, hi, 50)
        ax.plot(xx, 10 ** (sl * xx + ic), "--", color=K["text"], lw=1.6, zorder=4)
    yr = lambda x, sl, ic: 10 ** (sl * x + ic)
    ax.text(1972, yr(1972, G.early_slope, G.early_intercept) * 150,
            f"{G.early_doubling_str}-month doubling", fontsize=11, fontweight="bold",
            ha="center", color=K["text"], zorder=6)
    ax.annotate(f"{G.late_doubling_str}-month doubling", (2014, yr(2014, G.late_slope, G.late_intercept)),
                xytext=(1990, 1e11), fontsize=11, fontweight="bold", ha="center", va="center",
                color=K["text"], zorder=6,
                arrowprops=dict(arrowstyle="-", color=K["text"], lw=1))
    offs = {"Perceptron Mark I": (4, -16, "left"), "LeNet-5": (-6, 6, "right"),
            "AlexNet": (-6, 6, "right"), "GPT-3 175B (davinci)": (-30, 2, "right"),
            "PaLM (540B)": (-8, 4, "right")}
    for n, (dx, dy, ha) in offs.items():
        for x, r in zip(G.years, G.rows):
            if r["model"] == n:
                ax.annotate(n.replace(" (davinci)", "").replace(" (540B)", ""), (x, r["parameters"]),
                            xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=10.5,
                            color=K["text"], zorder=7,
                            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.5))
                break
    mn = CH["MnistModelSizeLocal"].total_params
    ax.axhline(mn, color=K["compute"], lw=1.4, ls=":", zorder=1)
    ax.text(2027, mn, " running\n network", fontsize=10.5, color=K["compute"],
            fontweight="bold", va="center", ha="left")
    for i, (k, col) in enumerate(list(dom.items()) + [("Other", "#999999")]):
        ax.text(0.02, 0.95 - i * 0.1, k, transform=ax.transAxes, fontsize=11,
                fontweight="bold", color=col, va="top")
    ax.set_yscale("log")
    ax.set_xlim(1948, 2027)
    ax.set_yticks([1e1, 1e4, 1e7, 1e10, 1e13])
    ax.minorticks_off()
    ax.set_xlabel("Publication date")
    ax.set_ylabel("Parameters (log scale)")
    ax.grid(axis="y", color=K["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "param_growth.svg")


def depth_width():
    D = CH["DepthWidthCost"]
    fig, (aw, ad) = plt.subplots(1, 2, figsize=(5.6, 1.9), gridspec_kw={"wspace": 0.75})
    w = D.sweep_widths
    mw = D.width_sweep_macs
    guide = [mw[-1] * (x / w[-1]) ** 2 for x in w]
    aw.plot(w, guide, color=K["neutral"], lw=1.4, ls="--")
    aw.plot(w, mw, color=K["memory"], lw=2.5, marker="o", ms=5)
    aw.plot([CH["MNISTInference"].h1_dim], [D.mnist_macs], marker="D", ms=9,
            color=K["danger"], ls="none", zorder=5)
    aw.text(w[0] * 0.95, mw[3] * 4, "two hidden\nlayers", color=K["memory"], fontsize=10.5,
            fontweight="bold", ha="left", va="bottom")
    aw.text(w[4] * 1.25, guide[4] / 5, "slope two", color=K["neutral"], fontsize=10.5,
            ha="left", va="top")
    aw.annotate("MNIST\n784-128-64-10", (CH["MNISTInference"].h1_dim, D.mnist_macs),
                xytext=(300, 1.3e4), textcoords="data", fontsize=10.5,
                color=K["danger"], fontweight="bold", va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=K["danger"], lw=1))
    aw.set_xscale("log", base=2)
    aw.set_yscale("log")
    aw.set_xticks([64, 256, 1024, 4096])
    aw.set_xticklabels(["64", "256", "1,024", "4,096"])
    aw.minorticks_off()
    aw.set_xlabel("Hidden width")
    aw.set_ylabel("MACs per example")
    aw.set_title("Width: quadratic", fontsize=12, fontweight="bold")
    dp = D.sweep_depths
    md = [m / 1e6 for m in D.depth_sweep_macs]
    act = [a / 1e3 for a in D.depth_sweep_retained]
    ad.plot(dp, md, color=K["memory"], lw=2.5)
    ad.set_xlim(0, dp[-1] + 1)
    ad.set_ylim(0, None)
    ad.set_xlabel(f"Hidden layers (width {D.sweep_fixed_width})")
    ad.set_ylabel("MACs (millions)", color=K["memory"])
    ad.set_title("Depth: linear", fontsize=12, fontweight="bold")
    aa = ad.twinx()
    aa.plot(dp, act, color=K["compute"], lw=2.5, ls="--")
    aa.set_ylim(0, None)
    aa.spines["right"].set_visible(True)
    aa.set_ylabel("Saved activations\n(thousand values)", color=K["compute"])
    ad.text(2, md[-1] * 0.95, "MACs", color=K["memory"], fontsize=10.5, fontweight="bold", va="top")
    ad.text(30, md[-1] * 0.2, "saved\nactivations", color=K["compute"], fontsize=10.5,
            fontweight="bold", ha="right", va="bottom")
    sp.save(fig, out / "depth_width.svg")


def mnist_roofline():
    H = CH["Hardware"]
    TF = CH["TFLOPs"] / CH["second"]
    TBs = CH["TB"] / CH["second"]
    M = CH["MnistFlopsCalc"]
    ai = np.logspace(-1.3, 3, 400)
    fig, ax = plt.subplots(figsize=(5.5, 1.95))
    machines = [("Desktop CPU", H.Cloud.ReferenceCPU, "#8B5A2B", "--", (10, -10)),
                ("Data-center accelerator", H.Cloud.H100, K["memory"], "-", (10, -10))]
    ridges = []
    for name, hw, col, ls, off in machines:
        peak = hw.compute.precision_flops["fp32"].to(TF).magnitude
        bw = hw.memory.bandwidth.to(TBs).magnitude
        ax.plot(ai, np.minimum(peak, bw * ai), color=col, lw=2.5, ls=ls)
        ax.text(900, peak * 1.4, name, color=col, fontsize=11, fontweight="bold",
                ha="right", va="bottom")
        ridges.append(peak / bw)
        for lab, x in (("batch 1", M.single_image_arith_intensity), ("batch 32", M.batch_arith_intensity)):
            y = min(peak, bw * x)
            ax.plot(x, y, "o", color=K["danger"], ms=8, zorder=4)
            val = f"{y * 1000:,.0f} GFLOP/s" if y < 1 else f"{y:,.0f} TFLOP/s"
            up = hw is H.Cloud.H100  # up-left keeps the accelerator labels off the CPU line
            ax.annotate(f"{lab}:\n{val}" if lab == "batch 1" and up else f"{lab}: {val}", (x, y),
                        xytext=(-10, 6) if up else off, textcoords="offset points",
                        ha="right" if up else "left", va="bottom" if up else "top",
                        fontsize=10.5, color=K["text"])
    ax.text(max(ridges) * 1.15, 1.6e-3, f"ridge ≈ {np.mean(ridges):.0f} FLOP/byte on both",
            fontsize=10.5, color=K["text"], ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai[0], ai[-1])
    ax.set_ylim(1e-3, 400)
    ax.minorticks_off()
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("FP32 rate (TFLOP/s)")
    ax.grid(color=K["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / "mnist_roofline.svg")


def step_memory():
    dims = CH["_MNIST_LAYERS_DIMS"]
    bpp = int(CH["BYTES_FP32"].to(CH["byte"]).magnitude)
    nl = len(dims) - 1
    lp = [dims[i] * dims[i + 1] + dims[i + 1] for i in range(nl)]
    pkb = sum(lp) * bpp / 1000

    def timeline(batch):
        acts = [batch * dd * bpp / 1000 for dd in dims]
        labels, A, Gr = ["load"], [acts[0]], [0.0]
        live = [acts[0]]
        for i in range(nl):
            live.append(acts[i + 1]); labels.append(f"F{i + 1}"); A.append(sum(live)); Gr.append(Gr[-1])
        g = 0.0
        for i in reversed(range(nl)):
            live.pop(); g += lp[i] * bpp / 1000
            labels.append(f"B{i + 1}"); A.append(sum(live)); Gr.append(g)
        labels.append("update"); A.append(0.0); Gr.append(g)
        peak = max(acts[i] + acts[i + 1] for i in range(nl))
        return labels, np.array(A), np.array(Gr), peak

    series = [("Weights", K["memory_l"], K["memory"]), ("Adam state", K["network_l"], K["network"]),
              ("Gradients", K["compute_l"], K["compute"]), ("Saved activations", K["danger_l"], K["danger"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 1.75), gridspec_kw={"wspace": 0.3})
    B = CH["_MNIST_BATCH_SIZE"]
    for ax, batch in zip(axes, (B, 32 * B)):
        labels, A, Gr, peak = timeline(batch)
        x = np.arange(len(labels))
        bottom = np.zeros_like(A)
        for vals, (nm, fill, edge) in zip([np.full_like(A, pkb), np.full_like(A, 2 * pkb), Gr, A], series):
            ax.bar(x, vals, bottom=bottom, width=0.72, color=fill, edgecolor=edge, lw=0.8,
                   label=nm)
            bottom = bottom + vals
        ax.axhline(pkb + peak, color=K["data"], ls="--", lw=2, label="Inference peak")
        ax.axvline(nl + 0.5, color=K["neutral"], lw=1, ls=":")
        top = bottom.max() * 1.3
        ax.set_ylim(0, top)
        ax.text(nl / 2, top * 0.97, "forward", ha="center", va="top", fontsize=10.5, color=K["neutral"])
        ax.text(nl + 0.5 + (nl + 1) / 2, top * 0.97, "backward", ha="center", va="top",
                fontsize=10.5, color=K["neutral"])
        ax.set_xticks(x)
        # "update" drops a line so it clears "B1" at this width
        ax.set_xticklabels(labels[:-1] + ["\n" + labels[-1]], fontsize=9.5)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"Batch {batch:,}", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Live memory (KB)")
    h, n = axes[0].get_legend_handles_labels()
    fig.legend(h, n, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=5, frameon=False,
               fontsize=10.5, handlelength=1.4, columnspacing=1.0)
    sp.save(fig, out / "step_memory.svg")


def model_bytes():
    Mo, H = CH["Models"], CH["Hardware"]
    dims = CH["_MNIST_LAYERS_DIMS"]
    bpp = CH["BYTES_FP32"].to(CH["byte"]).magnitude
    models = [("MNIST", sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))),
              ("ResNet-50", Mo.Vision.ResNet50.parameters.to(CH["param"]).magnitude),
              ("GPT-2", Mo.Language.GPT2.parameters.to(CH["param"]).magnitude)]
    by = CH["byte"]
    tiers = [("MCU SRAM", H.Tiny.ESP32_S3.memory.sram_capacity.to(by).magnitude),
             ("H100 L2", H.Cloud.H100.memory.l2_cache.to(by).magnitude),
             ("H100 HBM", H.Cloud.H100.memory.capacity.to(by).magnitude)]
    y = np.arange(len(models))
    wts = np.array([n * bpp for _, n in models])
    fig, ax = plt.subplots(figsize=(3.6, 1.55))
    ax.barh(y + 0.19, wts * 4, height=0.36, color=K["compute_l"], edgecolor=K["compute"])
    ax.barh(y - 0.19, wts, height=0.36, color=K["memory_l"], edgecolor=K["memory"])
    ax.text(wts[1] * 1.5, 1 - 0.19, "weights", va="center", fontsize=10.5, color=K["memory"], fontweight="bold")
    ax.text(wts[1] * 6.0, 1 + 0.19, "training", va="center", fontsize=10.5, color=K["compute"], fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(1e4, 1e13)
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in models], fontsize=11)
    ax.invert_yaxis()
    for nm, cap in tiers:
        ax.axvline(cap, color=K["danger"], lw=1.5, ls="--")
        ax.text(cap, 1.03, nm, transform=ax.get_xaxis_transform(), fontsize=10.5,
                color=K["danger"], ha={"MCU SRAM": "right", "H100 L2": "left"}.get(nm, "center"),
                va="bottom", fontweight="bold")
    ax.set_xticks([1e4, 1e6, 1e8, 1e10, 1e12])
    ax.set_xticklabels(["10 KB", "1 MB", "100 MB", "10 GB", "1 TB"], fontsize=10.5)
    ax.minorticks_off()
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    sp.save(fig, out / "model_bytes.svg")


param_growth()
depth_width()
mnist_roofline()
step_memory()
model_bytes()

print("figures written")
