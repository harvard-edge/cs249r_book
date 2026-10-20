#!/usr/bin/env python3
"""Data plots for Lecture 5 (Neural Computation).

Every derived value is computed in calcs.py (imported here as cx) from the
inputs listed below; nothing derived is typed by hand. Inputs come from the
rendered chapter (Neural Computation) unless noted:
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

  - Real runs in data/ (torch_runs.py, overfit_run.py) and two published
    figures, redrawn with slide-size axes over the original pixels:
    Krizhevsky et al. 2012 Figure 1 (alexnet-fig1.png) and the PyTorch
    memory-profiler timeline (pytorch-memory-timeline.png); see SOURCES.yml.

Run from this folder with the MLSysBook venv:
  ../../../../.venv/bin/python make_figs.py
"""

import sys
from pathlib import Path

import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs as cx  # noqa: E402
from deckvals import val, num  # noqa: E402

out = Path(__file__).parent
plt = sp.plt

# ---------------------------------------------------------------------------
# One digit, three methods: operations per digit on a log ladder
PD = cx.PARADIGMS
sp.hbars_steps(out / "ops_ladder", [
    ("Hand-written rules", PD.rule_comparisons, "neutral", f"{val('rules')} comparisons"),
    ("HOG features + SVM", PD.feature_operations, "neutral", f"{val('hog_ops')} operations"),
    ("Neural network", cx.R.macs_per_example, "compute",
     f"{val('macs')} MACs  = {val('net_over_rules')}× rules"),
], "ops", xlabel=f"Work to classify one {val('img_side')}×{val('img_side')} digit (log scale)")

# ---------------------------------------------------------------------------
# Activation slopes: how much gradient each activation passes back
z = np.linspace(-6, 6, 600)
sig = 1 / (1 + np.exp(-z))
dsig = sig * (1 - sig)
drelu = (z > 0).astype(float)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    ax.plot(z, dsig, color=sp.C["memory"], lw=3.5)
    ax.annotate(f"Sigmoid slope:\nat most {val('sig_peak')}", (0, cx.CHAIN.peak_slope), xytext=(-5.8, 0.5),
                fontsize=12, fontweight="bold", color=sp.C["memory"],
                arrowprops=dict(arrowstyle="-", color=sp.C["memory"], lw=1))
    if k == 2:
        ax.plot(z[z <= 0], drelu[z <= 0], color=sp.C["compute"], lw=3.5)
        ax.plot(z[z > 0], drelu[z > 0], color=sp.C["compute"], lw=3.5)
        ax.text(0.3, 0.80, "ReLU slope: 1\nwhen z > 0", fontsize=12, va="top",
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
TT = cx.LOGIC
sp.hbars_steps(out / "transistor_tax", [
    ("ReLU: compare and select", TT.cheap_transistors, "compute", val("relu_tr")),
    ("Sigmoid: exponential", TT.costly_transistors, "danger",
     f"{val('sig_tr')}  = {val('tr_ratio')}× ReLU"),
], "", xlabel="Transistor equivalents per activation unit (illustrative ratio; log scale)",
    size=sp.SHORT)

# ---------------------------------------------------------------------------
# Forward pass at batch 32: where the operations go (linear axis)
fwd = [(f"Layer {i + 1} MatMul", v / 1e6, "compute", val(f"mm{i + 1}"))
       for i, v in enumerate(cx.R.matmul_flops_per_layer)]
fwd.append(("Bias + activation, all layers", cx.R.elementwise_flops / 1e6, "neutral",
            f"{val('other_ops')}  ({val('other_share')}% of the total)"))
sp.hbars_steps(out / "fwd_ops", fwd, "", log=False,
               xlabel=f"Million operations for one batch of {val('batch')} digits (total about {val('fwd_m')} million)")

# ---------------------------------------------------------------------------
# A training step: forward plus two backward products per layer


def training_flops(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    T = cx.R
    parts = [("Forward", T.matmul_flops / 1e6, sp.C["compute"]),
             ("Weight gradients", T.weight_gradient_flops / 1e6, sp.C["danger"]),
             ("Input gradients", T.input_gradient_flops / 1e6, sp.C["network"])]
    left = 0.0
    shown = parts[:step]
    for name, v, col in shown:
        ax.barh(0, v, left=left, color=col, height=0.55, edgecolor="white", lw=1.5)
        if v > 1:
            ax.text(left + v / 2, 0, f"{name}\n{v:.2f} M", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
        left += v
    if step == 3:
        ax.annotate(f"Input gradients\n{val('train_igrad_m')} M (layer 1 skips it)", (left - 0.28, 0.28),
                    xytext=(left + 0.3, 0.62), fontsize=11, color=sp.C["network"],
                    fontweight="bold", ha="left",
                    arrowprops=dict(arrowstyle="-", color=sp.C["network"], lw=1))
        ax.text(left + 0.3, -0.05, f"{val('train_step_m')} M = {val('train_ratio')}× forward", fontsize=13,
                fontweight="bold", color=sp.C["text"], va="center")
    ax.set_xlim(0, 21.5)
    ax.set_ylim(-0.5, 0.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Million FLOPs per training step (batch {val('batch')})")
    ax.grid(axis="x", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2, 3):
    training_flops(out / f"train_flops_step{k}.svg", k)

# ---------------------------------------------------------------------------
# One epoch on two machines: arithmetic and data movement, then overhead


def epoch_plot(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    E = cx.E
    ms = lambda q: q.to("ms").magnitude
    rows = [("CPU", ms(E.work_time["host"]), ms(E.overhead_time["host"])),
            ("Accelerator", ms(E.work_time["accelerator"]), ms(E.overhead_time["accelerator"]))]
    for i, (name, work, over) in enumerate(rows):
        y = 1.3 * (1 - i)
        ax.barh(y, work, color=sp.C["compute"], height=0.5)
        ax.text(-4, y, name, ha="right", va="center", fontsize=13,
                fontweight="bold", color=sp.C["text"])
        if step == 1:
            ax.text(work + 4, y, f"{work:.0f} ms" if work >= 10 else f"{work:.1f} ms",
                    va="center", fontsize=13,
                    fontweight="bold", color=sp.C["compute"])
        else:
            ax.barh(y, over, left=work, color=sp.C["neutral_l"], height=0.5,
                    edgecolor=sp.C["neutral"], lw=0.8)
            ax.text(work + over + 4, y, f"{work + over:.0f} ms", va="center",
                    fontsize=13, fontweight="bold", color=sp.C["text"])
    if step == 1:
        ax.text(80, 0.65, f"Work alone: the accelerator is {val('work_speedup')}× faster",
                fontsize=13, color=sp.C["compute"], fontweight="bold")
    else:
        ax.text(338, 0.65, f"fixed overhead: {val('ovh_share')}% of the accelerator's epoch",
                fontsize=13, fontweight="bold", color=sp.C["neutral"],
                ha="right", va="center")
    # two-entry legend, drawn in both steps so the axes do not move
    alpha = 1.0 if step == 2 else 0.0
    for j, (txt, fc, ec) in enumerate((
            ("arithmetic and data movement", sp.C["compute"], sp.C["compute"]),
            ("fixed per-operation overhead", sp.C["neutral_l"], sp.C["neutral"]))):
        x0, y0 = 5, 2.35 - j * 0.32
        ax.add_patch(plt.Rectangle((x0, y0 - 0.1), 14, 0.2, facecolor=fc, edgecolor=ec,
                                   lw=0.8, alpha=alpha, clip_on=False))
        ax.text(x0 + 20, y0, txt, fontsize=13, va="center", color=sp.C["text"],
                alpha=alpha)
    ax.set_xlim(0, 420)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Time for one epoch of MNIST at batch {val('batch')} (ms); estimate, Napkin Math 5.3")
    ax.grid(axis="x", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2):
    epoch_plot(out / f"epoch_step{k}.svg", k)

# ---------------------------------------------------------------------------
# Training against inference memory (Table 9), built one term at a time


def memory_plot(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    r = cx.R
    M = {"params_kb": cx.kb(r.parameter_bytes), "act_train_kb": cx.kb(r.saved_activation_bytes),
         "act_inf_kb": cx.kb(r.inference_activation_bytes), "grads_kb": cx.kb(r.gradient_bytes),
         "adam_kb": cx.kb(r.optimizer_bytes)}
    train = [("Weights", M["params_kb"], sp.C["memory"]),
             ("Activations", M["act_train_kb"], sp.C["data"]),
             ("Gradients", M["grads_kb"], sp.C["danger"]),
             ("Adam state", M["adam_kb"], sp.C["network"])]
    infer = [("Weights", M["params_kb"], sp.C["memory"]),
             ("Activations", M["act_inf_kb"], sp.C["data"])]
    # inference row
    left = 0
    for name, v, col in infer:
        ax.barh(0, v, left=left, color=col, height=0.55, edgecolor="white", lw=1.5)
        left += v
    ax.text(left + 25, 0, f"{val('mem_infer_kb')} KB", va="center", fontsize=13,
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
    label = (f"{val('mem_train_mb')} MB  = {val('mem_ratio')}×" if step >= 3
             else f"{total:,.0f} KB")
    ax.text(total + 25, 1, label, va="center", fontsize=13,
            fontweight="bold", color=sp.C["text"])
    ax.text(M["params_kb"] / 2, 0, "Weights", ha="center", va="center", fontsize=11,
            color="white", fontweight="bold")
    ax.annotate("Saved activations", (M["params_kb"] + M["act_train_kb"] / 2, 1.28), xytext=(420, 1.75),
                fontsize=11, color=sp.C["data"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=sp.C["data"], lw=1))
    for y, name in ((1, "Training step"), (0, "Inference")):
        ax.text(-20, y, name, ha="right", va="center", fontsize=13,
                fontweight="bold", color=sp.C["text"])
    ax.set_xlim(0, 2300)
    ax.set_ylim(-0.5, 2.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Memory for the MNIST network, batch {val('batch')}, FP32 (KB)")
    ax.grid(axis="x", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2, 3):
    memory_plot(out / f"memory_step{k}.svg", k)

# ---------------------------------------------------------------------------
# Number formats: the range each format can hold (log scale), with the
# gradient left after 20 saturated sigmoid layers


def usci(v):
    """2.5e-08 -> '2.5×10⁻⁸' for matplotlib labels."""
    m, e = f"{v:.1e}".split("e")
    return f"{m}×10" + str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def ranges_plot(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    F = cx.FORMATS
    (f32_max, f32_norm, _), (b16_max, b16_norm, _), (f16_max, f16_norm, f16_sub) = (
        F.range("fp32"), F.range("bf16"), F.range("fp16"))
    G = {"fp16_min_normal": f16_norm, "fp16_min_sub": f16_sub, "fp16_max": f16_max, "int16_max": F.integer_max}
    rows = [("FP32", f32_norm, f32_max, sp.C["memory"]),
            ("BF16", b16_norm, b16_max, sp.C["memory"]),
            ("FP16", f16_norm, f16_max, sp.C["compute"]),
            ("INT16", 1, F.integer_max, sp.C["neutral"])]
    for i, (name, lo, hi, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, np.log10(hi) - np.log10(lo), left=np.log10(lo), color=col,
                height=0.55, alpha=0.9)
        ax.text(-41.5, y, name, ha="right", va="center", fontsize=13,
                fontweight="bold", color=sp.C["text"])
    # FP16 subnormals: reduced precision below the smallest normal value
    lo_s, lo_n = np.log10(G["fp16_min_sub"]), np.log10(G["fp16_min_normal"])
    ax.barh(1, lo_n - lo_s, left=lo_s, color=sp.C["compute_l"], height=0.55,
            edgecolor=sp.C["compute"], lw=0.8, hatch="///")
    ax.text(np.log10(G["fp16_max"]) + 1, 1,
            f"{usci(G['fp16_min_normal'])} to {val('fp16_max')}\n"
            f"subnormals to {usci(G['fp16_min_sub'])}",
            va="center", fontsize=11, color=sp.C["compute"], fontweight="bold",
            linespacing=1.1)
    ax.text(np.log10(G["int16_max"]) + 1, 0, f"max {val('int16_max')} (Ariane 501)", va="center",
            fontsize=11, color=sp.C["neutral"], fontweight="bold")
    if step >= 2:
        g = cx.CHAIN.surviving_fraction
        x = np.log10(g)
        ax.axvline(x, color=sp.C["danger"], lw=2, ls="--")
        ax.text(x - 1.2, 3.55, f"{val('sig_depth')} sigmoid layers: {usci(g)}",
                ha="right", fontsize=13, color=sp.C["danger"], fontweight="bold")
    ticks = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"10$^{{{t}}}$" for t in ticks])
    ax.set_xlim(-41, 41)
    ax.set_ylim(-0.5, 4.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Magnitudes a format holds at full precision (log scale)")
    ax.grid(axis="x", color=sp.C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2):
    ranges_plot(out / f"ranges_step{k}.svg", k)


# ===========================================================================
# Slide-size redraws of five chapter figures. The values come from the same
# MLSysIM scenarios the chapter's figures plot (NotableModelParameterGrowth,
# DenseWidthDepthSweep, MnistDenseNetwork, MnistEpochCpuVersusH100, and the
# hardware and model registry), read through calcs.py, so the slide and the
# book cannot disagree. Only the size, type, and labeling change: direct
# labels replace legends and every label is drawn at slide scale.
# ===========================================================================
K = sp.C
ARCH = "-".join(str(w) for w in cx.NET.layer_widths)


def param_growth():
    G = cx.GROWTH
    rows = [{"model": n, "parameters": p} for n, p in zip(G.models, G.parameters)]
    years, log_params = np.array(G.years), np.array(G.log_parameters)
    dom = {"Language": K["danger"], "Vision": K["memory"], "Games": K["data"]}
    d = np.array([x if x in dom else "Other" for x in G.domains])
    fig, ax = plt.subplots(figsize=(5.3, 2.0))
    for k, col in list(dom.items()) + [("Other", "#BBBBBB")]:
        m = d == k
        ax.scatter(years[m], 10 ** log_params[m], s=12, color=col,
                   alpha=0.75, edgecolors="none", zorder=3 if k != "Other" else 2)
    for lo, hi, sl, ic, txt, xm, lift in (
            (1950, G.split_year, G.early_slope, G.early_intercept, val("growth_early_mo"), 1975, 60),
            (G.split_year, 2026.8, G.late_slope, G.late_intercept, val("growth_late_mo"), 1998, 1)):
        xx = np.linspace(lo, hi, 50)
        ax.plot(xx, 10 ** (sl * xx + ic), "--", color=K["text"], lw=1.6, zorder=4)
    yr = lambda x, sl, ic: 10 ** (sl * x + ic)
    ax.text(1972, yr(1972, G.early_slope, G.early_intercept) * 150,
            f"{val('growth_early_mo')}-month doubling", fontsize=11, fontweight="bold",
            ha="center", color=K["text"], zorder=6)
    ax.annotate(f"{val('growth_late_mo')}-month doubling", (2014, yr(2014, G.late_slope, G.late_intercept)),
                xytext=(1978, 3e11), fontsize=11, fontweight="bold", ha="center", va="center",
                color=K["text"], zorder=6,
                arrowprops=dict(arrowstyle="-", color=K["text"], lw=1))
    offs = {"Perceptron Mark I": (4, -16, "left"), "LeNet-5": (-6, 6, "right"),
            "AlexNet": (-6, 6, "right"), "GPT-3 175B (davinci)": (-30, 2, "right"),
            "PaLM (540B)": (-8, 4, "right")}
    for n, (dx, dy, ha) in offs.items():
        for x, r in zip(years, rows):
            if r["model"] == n:
                ax.annotate(n.replace(" (davinci)", "").replace(" (540B)", ""), (x, r["parameters"]),
                            xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=10.5,
                            color=K["text"], zorder=7,
                            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=0.5))
                break
    mn = cx.R.parameters
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
    D = cx.SWEEP
    h1, mnist_macs = cx.NET.layer_widths[1], cx.R.macs_per_example
    fig, (aw, ad) = plt.subplots(1, 2, figsize=(5.6, 1.9), gridspec_kw={"wspace": 0.75})
    w = D.widths
    mw = D.width_sweep_macs
    guide = [mw[-1] * (x / w[-1]) ** 2 for x in w]
    aw.plot(w, guide, color=K["neutral"], lw=1.4, ls="--")
    aw.plot(w, mw, color=K["memory"], lw=2.5, marker="o", ms=5)
    aw.plot([h1], [mnist_macs], marker="D", ms=9,
            color=K["danger"], ls="none", zorder=5)
    aw.text(w[0] * 0.95, mw[3] * 4, "two hidden\nlayers", color=K["memory"], fontsize=10.5,
            fontweight="bold", ha="left", va="bottom")
    aw.text(w[4] * 1.25, guide[4] / 5, "slope two", color=K["neutral"], fontsize=10.5,
            ha="left", va="top")
    aw.annotate(f"MNIST\n{ARCH}", (h1, mnist_macs),
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
    dp = D.depths
    md = [m / 1e6 for m in D.depth_sweep_macs]
    act = [a / 1e3 for a in D.depth_sweep_saved_activations]
    ad.plot(dp, md, color=K["memory"], lw=2.5)
    ad.set_xlim(0, dp[-1] + 1)
    ad.set_ylim(0, None)
    ad.set_xlabel(f"Hidden layers (width {val('fixed_width')})")
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
    E = cx.E
    tf = lambda q: q.to("Tflop/s").magnitude
    ai = np.logspace(-1.8, 3, 400)  # room left of batch 1 for its label
    fig, ax = plt.subplots(figsize=(5.5, 1.95))
    machines = [("Desktop CPU", "host", "#8B5A2B", "--", (12, -8)),
                ("Data-center accelerator", "accelerator", K["memory"], "-", (10, -10))]
    i1, i32 = cx.R1.intensity.to("flop/byte").magnitude, cx.R.intensity.to("flop/byte").magnitude
    for name, key, col, ls, off in machines:
        peak = tf(E.peak[key])
        bw = E.bandwidth[key].to("TB/s").magnitude
        ax.plot(ai, np.minimum(peak, bw * ai), color=col, lw=2.5, ls=ls)
        ax.text(900, peak * 1.4, name, color=col, fontsize=11, fontweight="bold",
                ha="right", va="bottom")
        for lab, x, y in (("batch 1", i1, tf(E.rate_batch1[key])), (f"batch {val('batch')}", i32, tf(E.rate[key]))):
            ax.plot(x, y, "o", color=K["danger"], ms=8, zorder=4)
            rate_lbl = (f"{y * 1000:,.0f} GFLOP/s" if y < 1 else
                   f"{y:.1f} TFLOP/s" if y < 10 else f"{y:,.0f} TFLOP/s")
            up = key == "accelerator"  # up-left keeps the accelerator labels off the CPU line
            # accelerator labels up-left; CPU labels down-right of their points
            left = up
            ax.annotate(f"{lab}:\n{rate_lbl}" if lab == "batch 1" else f"{lab}: {rate_lbl}", (x, y),
                        xytext=(-10, 6) if left else off, textcoords="offset points",
                        ha="right" if left else "left", va="bottom" if left else "top",
                        fontsize=12, color=K["text"])
    ax.text(max(r.to("flop/byte").magnitude for r in E.ridge.values()) * 2.5, 4e-4,
            f"ridge ≈ {val('ridge')} FLOP/byte\non both machines",
            fontsize=10.5, color=K["text"], ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai[0], ai[-1])
    ax.set_ylim(2e-4, 400)
    ax.minorticks_off()
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("FP32 rate (TFLOP/s)")
    ax.grid(color=K["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / "mnist_roofline.svg")


def model_bytes():
    H = cx.mlsysim.Hardware
    by = lambda q: q.to("byte").magnitude
    models = [("MNIST", by(cx.R.parameter_bytes)), ("ResNet-50", by(cx.RESNET.weights)),
              ("GPT-2", by(cx.GPT2.weights))]
    tiers = [("MCU SRAM", by(H.Tiny.ESP32_S3.memory.sram_capacity)),
             ("H100 L2", by(H.Cloud.H100.memory.l2_cache)),
             ("H100 HBM", by(H.Cloud.H100.memory.capacity))]
    y = np.arange(len(models))
    wts = np.array([b for _, b in models])
    fig, ax = plt.subplots(figsize=(3.8, 1.45))
    ax.barh(y + 0.19, wts * cx.GPT2.copies, height=0.36, color=K["compute_l"], edgecolor=K["compute"])
    ax.barh(y - 0.19, wts, height=0.36, color=K["memory_l"], edgecolor=K["memory"])
    ax.set_xscale("log")
    ax.set_xlim(1e4, 1e13)
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in models], fontsize=13)
    ax.invert_yaxis()
    # tier lines, labeled above the plot on two rows so the labels never meet
    for (nm, cap), ha in zip(tiers, ("right", "left", "left")):
        ax.axvline(cap, color=K["danger"], lw=1.5, ls="--")
        ax.text(cap, 1.03, nm, transform=ax.get_xaxis_transform(), fontsize=13,
                color=K["danger"], ha=ha, va="bottom", fontweight="bold")
    # legend in its own row above the tier labels
    for j, (txt, fc, ec) in enumerate((("weights", K["memory_l"], K["memory"]),
                                       (f"training state (×{val('train_copies')})", K["compute_l"], K["compute"]))):
        x0 = 0.0 + j * 0.38
        ax.add_patch(plt.Rectangle((x0, 1.27), 0.05, 0.1, transform=ax.transAxes,
                                   facecolor=fc, edgecolor=ec, clip_on=False))
        ax.text(x0 + 0.07, 1.32, txt, transform=ax.transAxes, fontsize=13,
                va="center", color=K["text"])
    ax.set_xticks([1e4, 1e6, 1e8, 1e10, 1e12])
    ax.set_xticklabels(["10 KB", "1 MB", "100 MB", "10 GB", "1 TB"], fontsize=12)
    ax.minorticks_off()
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    sp.save(fig, out / "model_bytes.svg")


param_growth()
depth_width()
mnist_roofline()
model_bytes()


# ---------------------------------------------------------------------------
# The running network, 784 -> 128 -> 64 -> 10, drawn as four columns of
# units with every connection between adjacent columns. Step 2 adds the
# weight count of each gap (products of adjacent widths, chapter Table 5).
# The digit is a real MNIST sample: the first image of the official test set
# (a 7), copied to data/mnist_test0.json by overfit_run.py.


def mnist_digit():
    d = json.loads((out / "data" / "mnist_test0.json").read_text())
    side = cx.Datasets.MNIST.image_width
    return np.array(d["pixels"], dtype=float).reshape(side, side) / (cx.units.COLOR_DEPTH_8BIT - 1)


def mnist_net(path, step):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    # (x position, layer width, units drawn)
    cols = list(zip((1.9, 3.25, 4.45, 5.5), cx.NET.layer_widths, (7, 6, 5, 4)))
    ys = {}
    for x, n, k in cols:
        yy = np.linspace(0.15, 1.45, k) if k > 4 else np.linspace(0.35, 1.25, k)
        ys[x] = yy
    for (x0, _, _), (x1, _, _) in zip(cols[:-1], cols[1:]):
        for a in ys[x0]:
            for b in ys[x1]:
                ax.plot([x0, x1], [a, b], color="#BBBBBB", lw=0.6, zorder=1)
    for x, n, k in cols:
        ax.scatter([x] * k, ys[x], s=70, color="white", edgecolors=K["memory"],
                   lw=1.5, zorder=3)
        ax.text(x, 1.62, f"{n:,}", ha="center", va="bottom", fontsize=14,
                fontweight="bold", color=K["text"])
    ax.imshow(mnist_digit(), cmap="Greys", extent=(0.05, 1.0, 0.3, 1.25), zorder=2,
              interpolation="nearest")
    ax.annotate("", (1.75, 0.8), xytext=(1.1, 0.8),
                arrowprops=dict(arrowstyle="-|>", color=K["text"], lw=1.4))
    ax.text(0.52, 1.62, f"{val('img_side')}×{val('img_side')} MNIST digit", ha="center", va="bottom", fontsize=12,
            color=K["neutral"])
    if step >= 2:
        for (x0, n0, _), (x1, n1, _) in zip(cols[:-1], cols[1:]):
            ax.text((x0 + x1) / 2, -0.08, f"{n0 * n1:,}", ha="center", va="top",
                    fontsize=14, fontweight="bold", color=K["compute"])
    ax.set_xlim(0, 5.8)
    ax.set_ylim(-0.42, 1.9)
    ax.set_aspect("auto")
    ax.axis("off")
    sp.save(fig, path)


for k in (1, 2):
    mnist_net(out / f"mnist_net_step{k}.svg", k)

# ---------------------------------------------------------------------------
# XOR on the unit square: no line separates the classes (step 1); the band
# 0.5 < x1 + x2 < 1.5 where y = h1 - 2 h2 exceeds 0.5 (step 2).


def xor_plot(path, step):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    if step == 2:
        xx = np.array([-0.3, 1.3])
        ax.fill_between(xx, 0.5 - xx, 1.5 - xx, color=K["compute_l"], zorder=0)
        ax.plot(xx, 0.5 - xx, color=K["compute"], lw=1.5, ls="--")
        ax.plot(xx, 1.5 - xx, color=K["compute"], lw=1.5, ls="--")
        ax.text(0.5, 0.5, "y > 0.5", rotation=-45, ha="center", va="center",
                fontsize=15, fontweight="bold", color=K["compute"])
    pts = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
    for (a, b), c in pts:
        ax.scatter(a, b, s=260, color=K["compute"] if c else "white",
                   edgecolors=K["compute"] if c else K["memory"], lw=2.5, zorder=3)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("$x_1$", fontsize=16)
    ax.set_ylabel("$x_2$", fontsize=16, rotation=0, labelpad=10)
    ax.tick_params(labelsize=15)
    ax.set_aspect("equal")
    sp.save(fig, path)


for k in (1, 2):
    xor_plot(out / f"xor_step{k}.svg", k)

# ---------------------------------------------------------------------------
# Cross-entropy for one example: loss = -log p_correct (chapter eq. 18).
# Points: p = 0.9 -> 0.11, p = 0.8 -> 0.22, uniform guess p = 0.1 -> 2.30.


def cross_entropy():
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    p = np.linspace(0.02, 1.0, 400)
    ax.plot(p, -np.log(p), color=K["memory"], lw=3)
    CE = cx.CE
    for pv, txt, dx, dy, ha in (
            (CE.misread["true"], f"p = {val('ce_p_miss')}: {val('ce_miss')}\n(uniform guess, or a\nconfident mistake)",
             0.05, 0.25, "left"),
            (CE.probabilities["correct"], f"p = {val('ce_p')}: {val('ce_loss')}", 0.0, 0.75, "center"),
            (CE.probabilities["confident"], f"p = {val('ce_p_conf')}: {val('ce_conf')}", 0.06, 0.45, "left")):
        lv = -np.log(pv)
        ax.plot(pv, lv, "o", color=K["danger"], ms=9, zorder=4)
        ax.annotate(txt, (pv, lv), xytext=(pv + dx, lv + dy), fontsize=13,
                    color=K["text"], ha=ha, va="bottom",
                    arrowprops=dict(arrowstyle="-", color=K["neutral"], lw=1))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("Probability the network gives the correct digit")
    ax.set_ylabel("Loss, −log p")
    ax.grid(color=K["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / "cross_entropy.svg")


cross_entropy()

# ---------------------------------------------------------------------------
# Risk-coverage: error among accepted digits against the fraction accepted.
# The curve's shape is schematic; the one measured point is LeCun et al.
# 1989 via chapter Table 11 (1% error after rejecting 12.1%, so 87.9%
# accepted).


def risk_coverage():
    fig, ax = plt.subplots(figsize=(3.7, 2.3))
    c = np.linspace(0.55, 1.0, 200)
    acc, err = num("lecun_acc"), num("lecun_err")
    risk = err * np.exp(9.0 * (c - acc / 100))   # schematic: through the point
    ax.plot(c * 100, risk, color=K["memory"], lw=3)
    ax.plot(acc, err, "o", color=K["danger"], ms=10, zorder=4)
    ax.annotate(f"1989 ZIP-code\nreader: {val('lecun_err')}% error,\n{val('lecun_rej')}% rejected", (acc, err),
                xytext=(57, 1.6), fontsize=12, fontweight="bold", color=K["danger"],
                arrowprops=dict(arrowstyle="-", color=K["danger"], lw=1))
    ax.axvline(acc, color=K["neutral"], ls=":", lw=1.5)
    ax.text(acc + 0.7, 3.5, "threshold", fontsize=12, color=K["neutral"], ha="left",
            va="top")
    ax.set_xlim(55, 101)
    ax.set_ylim(0, 3.6)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0%", "1%", "2%", "3%"])
    ax.set_xlabel("Digits accepted by the machine (%)")
    ax.set_ylabel("Error among\naccepted digits")
    ax.text(1.0, 1.02, "curve schematic; point measured", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color=K["neutral"])
    ax.grid(axis="y", color=K["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / "risk_coverage.svg")


risk_coverage()


# ===========================================================================
# Published figures and real runs, redrawn at slide size. The curves of the
# two published figures are the original pixels (cropped to the plot box and
# placed on the published axes); only the axis labels and direct labels are
# redrawn, at a size the back row can read.
# ===========================================================================
def load_plot_box(png, box):
    """Crop a PNG to its axes box (left, top, right, bottom pixels)."""
    from PIL import Image
    return np.asarray(Image.open(out / png).convert("RGB").crop(box))


def alexnet_relu():
    # Krizhevsky, Sutskever, Hinton, NeurIPS 2012, Figure 1 (page 3), rendered
    # at 600 dpi. Axes measured on the crop: x = 0 at px 214 and 40 at px 1413;
    # y = 0 at px 987.5 and 0.75 at px 35.
    img = load_plot_box("alexnet-fig1.png", (215, 36, 1413, 987))
    fig, ax = plt.subplots(figsize=(2.25, 1.45))
    ax.imshow(img, extent=(0, 40, 0, 0.75), aspect="auto", interpolation="lanczos")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 0.75)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_yticklabels(["0", "25%", "50%", "75%"])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Epochs (CIFAR-10)", fontsize=11)
    ax.set_ylabel("Training error", fontsize=11)
    ax.text(7.0, 0.20, "ReLU", fontsize=12, fontweight="bold", color=K["compute"], va="top")
    ax.text(24, 0.36, "tanh", fontsize=12, fontweight="bold", color=K["memory"], va="bottom")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    sp.save(fig, out / "alexnet_relu.svg")


def memory_timeline():
    # PyTorch memory-profiler timeline of ResNet-50 training (DeVito and Shi,
    # PyTorch blog, 2023, last figure). Axes measured on the PNG: x ticks
    # 0 to 250 ms at px 256 to 1294; y ticks 0.0 to 0.4 GB at px 821 to 254.
    # Cropped at px 1150 (about 215 ms): one first step and three steady steps.
    x0, x1, y_top, y_bot = 201, 1150, 116, 854
    px_ms = (1294 - 256) / 250
    px_gb = (821 - 254) / 0.4
    ext = ((x0 - 256) / px_ms, (x1 - 256) / px_ms, (821 - y_bot) / px_gb, (821 - y_top) / px_gb)
    img = load_plot_box("pytorch-memory-timeline.png", (x0, y_top, x1, y_bot))
    fig, ax = plt.subplots(figsize=(4.3, 2.45))
    ax.imshow(img, extent=ext, aspect="auto", interpolation="lanczos")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Memory (GB)")
    # direct labels in the right margin, one per band of the last step
    for y_lab, (x_pt, y_pt), txt, col in (
            (0.047, (205, 0.047), "weights", "#3B7D3B"),
            (0.145, (205, 0.145), "optimizer state (momentum)", "#A87C12"),
            (0.235, (196, 0.225), "saved activations", "#D23C3C"),
            (0.325, (210, 0.30), "gradients", "#2B2BA8"),
            (0.425, (207, 0.40), "temporary buffers", K["neutral"])):
        ax.annotate(txt, (x_pt, y_pt), xytext=(226, y_lab), annotation_clip=False,
                    fontsize=12, fontweight="bold", color=col, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.1))
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    sp.save(fig, out / "memory_timeline.svg")


def overfit_curves():
    d = json.loads((out / "data" / "overfit.json").read_text())
    small, full = d["runs"][cx.small], d["runs"][cx.full]
    ep = np.arange(1, len(small["train_acc"]) + 1)
    fig, ax = plt.subplots(figsize=(3.2, 2.35))
    ax.plot(ep, np.array(small["train_acc"]) * 100, color=K["compute"], lw=2.5)
    ax.plot(ep, np.array(small["test_acc"]) * 100, color=K["compute"], lw=2.5, ls="--")
    ax.plot(ep, np.array(full["test_acc"]) * 100, color=K["memory"], lw=2.5, ls="--")
    ax.text(40, 100.6, f"training, {val('ov_small')} images: {val('ov_train_small')}%", fontsize=11,
            color=K["compute"], fontweight="bold", ha="right", va="bottom")
    ax.text(40, num("ov_test_small") - 0.8, f"held-out: {val('ov_test_small')}%", fontsize=11,
            color=K["compute"], fontweight="bold", ha="right", va="top")
    ax.text(40, num("ov_test_full") - 0.8, f"held-out, trained on\n{val('ov_full')} images: {val('ov_test_full')}%",
            fontsize=11, color=K["memory"], fontweight="bold", ha="right", va="top",
            linespacing=1.0)
    ax.set_xlim(0, 41)
    ax.set_ylim(78, 103)
    ax.set_yticks([80, 85, 90, 95, 100])
    ax.set_yticklabels(["80%", "85%", "90%", "95%", "100%"])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Epochs", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(axis="y", color=K["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / "overfit_curves.svg")


alexnet_relu()
memory_timeline()
overfit_curves()

print("figures written")
