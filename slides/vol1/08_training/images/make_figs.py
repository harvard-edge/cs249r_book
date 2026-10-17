#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 8 (Model Training).

Every value is from the rendered chapter (Model Training) unless noted:
  - GPT-2: 3.0e9 FLOPs per forward token; 1.5e21 FLOPs per training run
    (Sec. 8.1). Ratio 5e11 is derived here.
  - Training work per token: 2P forward, 4P backward, 6P total (Lighthouse 8.1).
  - Published utilization: PaLM 46.2 percent MFU on 6,144 TPU v4 (Chowdhery
    et al. 2022, cited in Sec. 8.5); GPT models on A100 clusters 44 to 52
    percent of peak (Narayanan et al. 2021, cited in Sec. 8.2); the GPT-2 XL
    scenario assumes 50 percent (Sec. 8.2).
  - Static state: 2 + 2 + 4 + 8 = 16 bytes per parameter (Sec. 8.3.2).
  - GPT-2 XL at batch 32, 1,024 tokens, mixed precision: static 24 GB,
    activations 286.9 GB of which about 201 GB are attention scores
    (Napkin Math 8.3, Sec. 8.6.4); H100 capacity 85.9 GB (80 GiB).
  - Llama 3 8B: full fine-tune 128 GB static + 36.5 GB activations; LoRA
    16 GB frozen weights + 0.11 GB adapter state + 36.5 GB (Napkin Math 8.4).
  - Bandwidth: NVMe 7, host DRAM 50, H100 HBM 3,350 GB/s (Sec. 8.4.2).
  - ResNet-50 feeding: demand 3,200 images/s (A100) and about 10,100 (H100);
    14 host workers at 375 images/s supply 5,250 (Sec. 8.4.2).
  - Precision: FP32 518 days, TF32 70, BF16 35 on one H100 at 50 percent
    (Napkin Math 8.7).
  - FlashAttention: 4,096 x 4,096 FP32 scores 67.1 MB; 128 x 128 tile
    65.5 KB (Sec. 8.6.4).
  - Checkpointing: 8 segments of 6 layers; 0.84 GB of segment inputs plus
    10.7 GB for the live segment (Napkin Math 8.8).
  - Scoreboard: 598, 311, 109.6, 66.8 GB (Napkin Math 8.10).
  - Optimizer bytes: SGD 8, momentum 12, Adam 16 bytes per parameter in FP32
    (table 3, Sec. 8.3.2).
  - Activations against sequence length: derived from Napkin Math 8.3's line
    items (242 GB at 1,024 tokens; the library total is 287 GB).
Illustrative figures (the PaLM loss curve, the prefetch timeline, the MFU bars
of the busy-time scenarios) say so on the figure.

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
CAP = 85.9  # H100 HBM, 80 GiB in decimal GB, as the chapter compares


def label(ax, x, y, text, color, shown=True, **kw):
    """A bold label that is drawn in every step (invisible until shown), so
    every step has the same bounding box and the slide does not jump."""
    kw.setdefault("fontsize", 12)
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, text, color=color, alpha=1.0 if shown else 0.0, **kw)


# 1. PaLM loss spike (illustrative schematic) -----------------------------
rng = np.random.default_rng(3)
steps = np.arange(0, 400)
loss = 3.2 * np.exp(-steps / 260) + 1.6 + rng.normal(0, 0.025, len(steps))
spike = loss.copy()
spike[300:330] += np.concatenate([np.linspace(0, 2.2, 8), np.linspace(2.2, 0.9, 22)])
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(steps[:300], loss[:300], color=C["neutral"], lw=2.5)
    ax.plot(steps[299:330], spike[299:330], color=C["danger"], lw=2.5)
    label(ax, 318, spike[310] + 0.12, "spike", C["danger"], ha="left")
    # rewind arrow and skipped window appear in step 2
    show = k == 2
    ax.annotate("", xy=(200, loss[200] - 0.55), xytext=(300, loss[200] - 0.55),
                arrowprops=dict(arrowstyle="-|>", color=C["memory"], lw=2.2),
                alpha=1.0 if show else 0.0)
    label(ax, 250, loss[200] - 0.85, "rewind about 100 steps", C["memory"], show,
          ha="center", va="top", fontsize=11)
    ax.axvspan(300, 330, color=C["data_l"], alpha=0.9 if show else 0.0, lw=0)
    label(ax, 335, 4.2, "skip 200 to\n500 batches", C["data"], show, ha="left",
          va="top", fontsize=11)
    ax.set_xlim(0, 420)
    ax.set_ylim(1.0, 4.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Training loss")
    ax.text(0.0, -0.2, "Illustrative shape, not PaLM data", transform=ax.transAxes,
            fontsize=9, color=C["neutral"])
    sp.save(fig, out / f"palm_spike_step{k}.svg")

# 2. One training run against one generated token (log scale) ---------------
sp.hbars_steps(out / "train_vs_token", [
    ("One generated token", 3.0e9, "neutral", "3 × 10⁹ FLOPs"),
    ("One training run", 1.5e21, "compute", "1.5 × 10²¹  = 500 billion tokens"),
], "FLOPs", xlabel="Arithmetic (FLOPs, log scale), GPT-2", size=sp.SHORT)

# 3. FLOPs per token: inference 2P, training 6P -------------------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    rows = [("Inference", [(2, "forward  2P", "memory")]),
            ("Training", [(2, "forward  2P", "memory"), (4, "backward  4P", "compute")])]
    for i, (name, segs) in enumerate(rows):
        y = 1 - i
        shown = i < k
        x = 0
        for w, text, ck in segs:
            ax.barh(y, w, left=x, color=C[ck], height=0.6, edgecolor="white",
                    alpha=1.0 if shown else 0.0)
            label(ax, x + w / 2, y, text, "white", shown, ha="center", va="center")
            x += w
        ax.text(-0.1, y, name, ha="right", va="center", fontsize=13,
                fontweight="bold", color=C["text"])
        label(ax, x + 0.15, y, f"{x}P per token", C["text"], shown, va="center",
              fontsize=13)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(-0.6, 1.6)
    ax.axis("off")
    sp.save(fig, out / f"flops_per_token_step{k}.svg")

# 4. Published utilization (fraction of peak doing model arithmetic) ---------
fig, ax = plt.subplots(figsize=(5.8, 2.2))
rows = [("PaLM 540B, 6,144 TPU v4", 46.2, "compute", "46.2% (measured MFU)"),
        ("GPT models, A100 clusters", 52, "compute", "44 to 52%"),
        ("GPT-2 XL, this lecture", 50, "neutral", "50% (assumed)")]
for i, (name, v, ck, text) in enumerate(rows):
    y = len(rows) - 1 - i
    if name.startswith("GPT models"):
        ax.barh(y, 44, color=C[ck], height=0.6)
        ax.barh(y, 8, left=44, color=C["compute_l"], height=0.6)
    else:
        ax.barh(y, v, color=C[ck], height=0.6)
    ax.text(-1, y, name, ha="right", va="center", fontsize=12, fontweight="bold",
            color=C["text"])
    ax.text(v + 1.5, y, text, va="center", fontsize=12, fontweight="bold", color=C[ck])
ax.axvline(100, color=C["neutral"], lw=1.5, ls="--")
ax.text(99, 2.45, "peak", ha="right", va="bottom", fontsize=11, color=C["neutral"])
ax.set_xlim(0, 100)
ax.set_ylim(-0.6, 2.8)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Percent of peak FLOP/s spent on model arithmetic")
ax.grid(axis="x", color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "mfu_published.svg")

# 5. Four kinds of state (simplified 1:1:2 view, then activations) -----------
segs = [(1.9, "weights", "memory"), (1.9, "gradients", "network"),
        (3.8, "Adam m, v", "compute"), (3.4, "activations", "data")]
for k in range(1, 5):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    x = 0
    for j, (w, text, ck) in enumerate(segs):
        shown = j < k
        ax.barh(0, w, left=x, color=C[ck], height=0.62, edgecolor="white",
                alpha=1.0 if shown else 0.0)
        label(ax, x + w / 2, 0, text, "white", shown, ha="center", va="center")
        x += w
    label(ax, 0.95, -0.62, "inference stops here", C["memory"], k == 1,
          ha="center", va="top", fontsize=11)
    label(ax, 3.8, 0.55, "1 : 1 : 2", C["text"], 3 <= k, ha="center", va="bottom",
          fontsize=12)
    label(ax, 9.3, -0.62, "grows with batch and sequence length", C["data"], k == 4,
          ha="center", va="top", fontsize=11)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-1.15, 0.95)
    ax.axis("off")
    sp.save(fig, out / f"train_state_step{k}.svg")

# 6. Bytes per parameter: inference 2, mixed-precision Adam 16 ---------------
fig, ax = plt.subplots(figsize=(5.8, 2.0))
rows = [("Inference", [(2, "W", "memory")]),
        ("Training", [(2, "W", "memory"), (2, "G", "network"),
                      (4, "master", "neutral"), (8, "Adam m, v (FP32)", "compute")])]
for i, (name, parts) in enumerate(rows):
    y = 1 - i
    x = 0
    for w, text, ck in parts:
        ax.barh(y, w, left=x, color=C[ck], height=0.6, edgecolor="white")
        ax.text(x + w / 2, y, text, ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")
        x += w
    ax.text(-0.2, y, name, ha="right", va="center", fontsize=13, fontweight="bold",
            color=C["text"])
    ax.text(x + 0.3, y, f"{x} B", va="center", fontsize=14, fontweight="bold",
            color=C["danger"] if x == 16 else C["text"])
ax.set_xlim(0, 18.8)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([])
ax.set_xticks([0, 2, 4, 8, 16])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Bytes per parameter before activations (W, G: FP16 weights, gradients; master: FP32)", fontsize=11)
sp.save(fig, out / "static_bytes.svg")

# 7. GPT-2 XL at batch 32, mixed precision: 311 GB against 85.9 GB ------------
parts = [(24, "", "neutral"), (85.6, "activations 86", "data"),
         (201.3, "attention scores 201", "danger")]
for k in range(1, 4):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    x = 0
    for j, (w, text, ck) in enumerate(parts):
        shown = j < k
        ax.barh(0, w, left=x, color=C[ck], height=0.6, edgecolor="white",
                alpha=1.0 if shown else 0.0)
        label(ax, x + w / 2, 0, text, "white", shown, ha="center", va="center",
              fontsize=11)
        x += w
    label(ax, 0, -0.38, "static 24", C["neutral"], True, va="top", fontsize=11)
    label(ax, 312, 0, " 311 GB", C["text"], k == 3, va="center", fontsize=13)
    ax.axvline(CAP, color=C["memory"], lw=2, ls="--", zorder=0)
    ax.text(CAP + 3, 0.42, "H100: 85.9 GB (80 GiB)", color=C["memory"], fontsize=11,
            fontweight="bold", va="bottom")
    ax.set_xlim(0, 360)
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("GB for one batch-32 step of GPT-2 XL (FP16 activations)")
    sp.save(fig, out / f"gpt2_b32_step{k}.svg")

# 8. Llama 3 8B fine-tuning scope --------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 2.3))
rows = [("Full fine-tune", [(16, "W", "memory"), (112, "grads, master, Adam  112", "compute"),
                            (36.5, "act.", "data")], "about 165 GB"),
        ("LoRA adapters", [(16, "W", "memory"), (0.11, "", "compute"),
                           (36.5, "act. 36.5", "data")], "about 53 GB")]
for i, (name, parts, total) in enumerate(rows):
    y = 1 - i
    x = 0
    for w, text, ck in parts:
        ax.barh(y, w, left=x, color=C[ck], height=0.6, edgecolor="white")
        if text:
            ax.text(x + w / 2, y, text, ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold")
        x += w
    ax.text(-2, y, name, ha="right", va="center", fontsize=13, fontweight="bold",
            color=C["text"])
    ax.text(max(x, CAP) + 3, y, total, va="center", fontsize=13, fontweight="bold",
            color=C["danger"] if x > CAP else C["data"])
ax.axvline(CAP, color=C["memory"], lw=2, ls="--", zorder=0)
ax.text(CAP + 2, 1.42, "H100: 85.9 GB", color=C["memory"], fontsize=11,
        fontweight="bold", va="bottom")
ax.set_xlim(0, 215)
ax.set_ylim(-0.55, 1.75)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("GB on one accelerator, Llama 3 8B (W: 16 GB of 16-bit weights)")
sp.save(fig, out / "llama_finetune.svg")

# 9. Bandwidth ladder ------------------------------------------------------------
sp.hbars_steps(out / "bw_ladder", [
    ("NVMe storage", 7, "data", "7 GB/s"),
    ("Host DRAM", 50, "neutral", "50 GB/s"),
    ("H100 HBM", 3350, "memory", "3,350 GB/s  = 480× NVMe"),
], "GB/s", xlabel="Bandwidth (GB/s, log scale)")

# 10. ResNet-50: accelerator demand against host supply -----------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    gpus = ["A100", "H100"]
    demand = [3200, 10100]
    supply = 5250
    for i, g in enumerate(gpus):
        shown = i < k
        a = 1.0 if shown else 0.0
        ax.bar(i - 0.2, demand[i], 0.38, color=C["compute"], alpha=a)
        ax.bar(i + 0.2, supply, 0.38, color=C["data"], alpha=a)
        label(ax, i - 0.2, demand[i] + 250, f"{demand[i]:,}", C["compute"], shown,
              ha="center", fontsize=12)
        label(ax, i + 0.2, supply + 250, "5,250", C["data"], shown, ha="center",
              fontsize=12)
    label(ax, 0.0, 6300, "fed", C["data"], k >= 1, ha="center", fontsize=14)
    label(ax, 1.0, 11300, "busy 52%", C["danger"], k == 2, ha="center", fontsize=14)
    ax.text(-0.5, 14300, "accelerator demand", color=C["compute"], fontsize=12,
            fontweight="bold")
    ax.text(-0.5, 13200, "host supply (14 workers)", color=C["data"], fontsize=12,
            fontweight="bold")
    ax.set_xticks([0, 1], gpus)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 15200)
    ax.set_yticks([0, 4000, 8000, 12000], ["0", "4k", "8k", "12k"])
    ax.set_ylabel("Images per second")
    sp.save(fig, out / f"resnet_feed_step{k}.svg")

# 11. Precision sets the schedule (GPT-2 XL on one H100, 50 percent) ----------
sp.hbars_steps(out / "precision_days", [
    ("FP32 (67 TFLOP/s)", 518, "neutral", "518 days, $49,751"),
    ("TF32 (494)", 70, "memory", "70 days, $6,748"),
    ("BF16 (989)", 35, "compute", "35 days, $3,370"),
], "days", log=False, xlabel="Days for GPT-2 XL on one H100 at 50% utilization, $4/hour")

# 12. FlashAttention: the full matrix against one tile ----------------------
for k in (1, 2):
    fig = plt.figure(figsize=(5.8, 2.7))
    axm = fig.add_axes([0.0, 0.08, 0.36, 0.84])
    axb = fig.add_axes([0.52, 0.22, 0.46, 0.6])
    axm.add_patch(plt.Rectangle((0, 0), 4096, 4096, fc=C["danger_l"], ec=C["danger"], lw=2))
    axm.text(2048, 2048, "4,096 × 4,096\nscores in HBM", ha="center", va="center",
             fontsize=12, fontweight="bold", color=C["danger"])
    shown = k == 2
    axm.add_patch(plt.Rectangle((0, 4096 - 128), 128, 128, fc=C["memory"],
                                ec=C["memory"], lw=2, alpha=1.0 if shown else 0.0))
    axm.annotate("128 × 128 tile\nin SRAM", xy=(128, 4096 - 64), xytext=(700, 3150),
                 fontsize=11, fontweight="bold", color=C["memory"],
                 arrowprops=dict(arrowstyle="-|>", color=C["memory"], lw=1.6,
                                 alpha=1.0 if shown else 0.0),
                 alpha=1.0 if shown else 0.0)
    axm.set_xlim(-60, 4160)
    axm.set_ylim(-60, 4160)
    axm.set_aspect("equal")
    axm.axis("off")
    vals = [(67.1e6, "danger", "Full matrix  67.1 MB"), (65.5e3, "memory", "One tile  65.5 KB,\n1,024× smaller")]
    for i, (v, ck, text) in enumerate(vals):
        on = i == 0 or shown
        axb.barh(1 - i, v, color=C[ck], height=0.6, alpha=1.0 if on else 0.0)
        label(axb, v * 1.4, 1 - i, text, C[ck], on, va="center", fontsize=12)
    axb.set_xscale("log")
    axb.set_xlim(1e4, 3e10)
    axb.set_ylim(-0.6, 1.6)
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.set_xticks([1e4, 1e6, 1e8], ["10 KB", "1 MB", "100 MB"])
    axb.minorticks_off()
    axb.set_xlabel("Bytes per head (FP32, log scale)")
    sp.save(fig, out / f"flash_tile_step{k}.svg")

# 13. Segment checkpointing: keep 8 inputs, rebuild one segment ---------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    for s in range(8):
        x0 = s * 1.0
        ax.add_patch(plt.Rectangle((x0 + 0.02, 0.0), 0.12, 1.0, fc=C["data"], ec=C["data"]))
        rebuilt = k == 2 and s == 5
        ax.add_patch(plt.Rectangle((x0 + 0.17, 0.0), 0.78, 1.0,
                                   fc=C["compute_l"] if rebuilt else "white",
                                   ec=C["compute"] if rebuilt else C["neutral_l"],
                                   lw=2 if rebuilt else 1.2, ls="-" if rebuilt else "--"))
    label(ax, 0.0, 1.12, "kept: 8 segment inputs, 0.84 GB", C["data"], True,
          fontsize=12, va="bottom")
    label(ax, 5.56, -0.12, "rebuilt in backward, one at a time: 10.7 GB", C["compute"], k == 2,
          fontsize=12, va="top", ha="center")
    label(ax, 8.05, 1.12, "dropped", C["neutral"], True, fontsize=12, va="bottom",
          ha="right")
    label(ax, 0.56, -0.12, "6 layers", C["neutral"], True, fontsize=11, va="top",
          ha="center", fontweight="normal")
    ax.set_xlim(-0.05, 8.1)
    ax.set_ylim(-0.6, 1.5)
    ax.axis("off")
    sp.save(fig, out / f"ckpt_segments_step{k}.svg")

# 14. Running scoreboard: GPT-2 XL batch-32 step on one H100 -------------------
names = ["FP32", "Mixed\nprecision", "Flash-\nAttention", "2 micro-\nbatches"]
totals = [598, 311, 109.6, 66.8]
for k in range(1, 5):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    for i, t in enumerate(totals):
        shown = i < k
        a = 1.0 if shown else 0.0
        ax.bar(i, 24, 0.58, color=C["neutral"], alpha=a)
        ax.bar(i, t - 24, 0.58, bottom=24, color=C["data"], alpha=a)
        label(ax, i, max(t, CAP) + 18, f"{t:g} GB", C["data"] if t < CAP else C["danger"],
              shown, ha="center", fontsize=13)
    ax.axhline(CAP, color=C["memory"], lw=2, ls="--", zorder=0)
    ax.text(3.55, CAP, "H100\n85.9 GB", color=C["memory"], fontsize=11,
            fontweight="bold", ha="left", va="center")
    ax.text(3.45, 560, "gray: static 24 GB", color=C["neutral"], fontsize=11,
            fontweight="bold", ha="right")
    ax.text(3.45, 500, "green: activations", color=C["data"], fontsize=11,
            fontweight="bold", ha="right")
    ax.set_xticks(range(4), names, fontsize=11)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 660)
    ax.set_ylabel("GB per step")
    sp.save(fig, out / f"scoreboard_step{k}.svg")

# 15. The input pipeline: three stages, throughput is the minimum ------------
from matplotlib.patches import FancyBboxPatch  # noqa: E402

fig, ax = plt.subplots(figsize=(5.8, 1.7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.0)
ax.axis("off")
stages = [(0.1, "Prepare\nstorage + CPU", "data"), (3.55, "Transfer\nPCIe", "network"),
          (7.0, "Compute\naccelerator", "compute")]
for x0, text, ck in stages:
    ax.add_patch(FancyBboxPatch((x0, 0.55), 2.9, 1.25,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=C[ck + "_l"], ec=C[ck], lw=2))
    ax.text(x0 + 1.45, 1.17, text, ha="center", va="center", fontsize=13,
            color=C["text"], fontweight="bold", linespacing=1.15)
for x0 in (3.0, 6.45):
    ax.annotate("", xy=(x0 + 0.55, 1.17), xytext=(x0, 1.17),
                arrowprops=dict(arrowstyle="-|>", color=C["neutral"], lw=2,
                                mutation_scale=18))
ax.text(5.0, 0.12, "batches per second = min(prepare, transfer, compute)",
        ha="center", fontsize=13, color=C["text"])
sp.save(fig, out / "pipeline_stages.svg")

# 16. Serial against overlapped preparation (schematic units) ----------------
prep, comp, nb = 20, 15, 4
fig, axs = plt.subplots(2, 1, figsize=(5.8, 2.9), sharex=True)
for ax, mode in zip(axs, ["Serial", "Overlapped"]):
    t_prep, t_comp = [], []
    if mode == "Serial":
        t = 0
        for _ in range(nb):
            t_prep.append(t)
            t += prep
            t_comp.append(t)
            t += comp
    else:
        cend = 0
        for i in range(nb):
            t_prep.append(i * prep)
            start = max((i + 1) * prep, cend)
            t_comp.append(start)
            cend = start + comp
    for s in t_prep:
        ax.barh(1, prep, left=s, color=C["data"], edgecolor="white", height=0.6)
    for s in t_comp:
        ax.barh(0, comp, left=s, color=C["compute"], edgecolor="white", height=0.6)
    end = t_comp[-1] + comp
    ax.axvline(end, color=C["text"], lw=1.5, ls=":")
    ax.text(end + 3, 0.5, f"{mode}: {end}", fontsize=12, va="center",
            fontweight="bold", color=C["text"])
    ax.set_yticks([0, 1], ["accelerator", "host"])
    ax.set_ylim(-0.6, 1.6)
    ax.spines["left"].set_visible(False)
axs[1].set_xlim(0, 175)
axs[1].set_xlabel("Time (schematic: prepare 20, compute 15 per batch)", fontsize=11)
sp.save(fig, out / "prefetch_overlap.svg")


# 17. Optimizer state: SGD, momentum, Adam, bytes per parameter in FP32 ------
#     (table 3 and Sec. 8.3.2: weights and gradients, plus 0, 1, or 2 state
#     tensors per parameter).
opt_rows = [("SGD", [(4, "W", "memory"), (4, "G", "network")]),
            ("Momentum", [(4, "W", "memory"), (4, "G", "network"), (4, "v", "compute")]),
            ("Adam", [(4, "W", "memory"), (4, "G", "network"), (4, "m", "compute"),
                      (4, "v", "compute")])]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    for i, (name, parts) in enumerate(opt_rows):
        y = 2 - i
        shown = i < k
        x = 0
        for w, text, ck in parts:
            ax.barh(y, w, left=x, color=C[ck], height=0.6, edgecolor="white",
                    alpha=1.0 if shown else 0.0)
            label(ax, x + w / 2, y, text, "white", shown, ha="center", va="center")
            x += w
        label(ax, -0.3, y, name, C["text"], shown, ha="right", va="center", fontsize=13)
        label(ax, x + 0.4, y, f"{x} B", C["danger"] if x == 16 else C["text"], shown,
              va="center", fontsize=13)
    ax.set_xlim(0, 19.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Bytes per parameter in FP32 (W weights, G gradients, m and v optimizer state)",
                  fontsize=10.5)
    sp.save(fig, out / f"optimizer_bytes_step{k}.svg")

# 18. GPT-2 XL activations against sequence length, batch 32, FP16 ------------
#     Napkin Math 8.3 line items, scaled with sequence length S (derived):
#     linear terms 48 x (419.4 + 419.4 + 10) MB x S/1,024; scores
#     48 x 5 x 25 x S^2 x 32 bytes. At S = 1,024: about 41 + 201 = 242 GB.
seq = np.linspace(128, 2048, 200)
lin = 48 * (419.4 + 419.4 + 10) * seq / 1024 / 1000
score = 48 * 5 * 25 * seq ** 2 * 32 / 1e9
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.fill_between(seq, 0, lin, color=C["data"], lw=0)
    ax.fill_between(seq, lin, lin + score, color=C["danger"], lw=0,
                    alpha=1.0 if k == 2 else 0.0)
    ax.axhline(CAP, color=C["memory"], lw=2, ls="--")
    ax.text(140, CAP + 14, "H100: 85.9 GB", color=C["memory"], fontsize=11,
            fontweight="bold")
    ax.text(150, 250, "green: projections and\nfeed-forward, grow with S", color=C["data"],
            fontsize=11, fontweight="bold", ha="left", va="center")
    label(ax, 1350, 520, "attention scores:\ngrow with S²", C["danger"], k == 2,
          ha="right", fontsize=11.5)
    ax.plot([1024], [lin[np.abs(seq - 1024).argmin()] + score[np.abs(seq - 1024).argmin()]],
            "o", ms=9, color=C["danger"], mec="white", mew=1.5, alpha=1.0 if k == 2 else 0.0)
    label(ax, 990, 370, "1,024 tokens: 242 GB", C["danger"], k == 2, ha="right",
          fontsize=11.5)
    ax.set_xlim(128, 2048)
    ax.set_ylim(0, 900)
    ax.set_xticks([256, 512, 1024, 1536, 2048], ["256", "512", "1,024", "1,536", "2,048"])
    ax.set_xlabel("Sequence length S (batch 32; line items, derived)", fontsize=11)
    ax.set_ylabel("Activations (GB)")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"act_vs_seq_step{k}.svg")

# 19. Busy time against MFU (illustrative scenarios after the Sec. 8.5
#     self-check: a memory-stalled job 92 percent busy, a data-starved job 20
#     percent busy; MFU values are illustrative) ------------------------------
fig, ax = plt.subplots(figsize=(5.8, 2.3))
rows = [("Memory-stalled job", 92, 20), ("Data-starved job", 20, 12)]
for i, (name, busy, mfu) in enumerate(rows):
    y = 1 - i
    ax.barh(y + 0.17, busy, height=0.32, color=C["neutral"])
    ax.barh(y - 0.17, mfu, height=0.32, color=C["compute"])
    ax.text(busy + 1.5, y + 0.17, f"busy {busy}%", va="center", fontsize=12,
            fontweight="bold", color=C["neutral"])
    ax.text(mfu + 1.5, y - 0.17, "MFU (illustrative)", va="center", fontsize=12,
            fontweight="bold", color=C["compute"])
    ax.text(-1.5, y, name, ha="right", va="center", fontsize=12, fontweight="bold",
            color=C["text"])
ax.set_xlim(0, 120)
ax.set_ylim(-0.55, 1.55)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Percent of the step (busy) or of peak (MFU)", fontsize=11)
sp.save(fig, out / "busy_mfu.svg")

# ---------------------------------------------------------------------------
# The data-bound profiler trace (chapter figure, Sec. 8.5), cropped to two
# training steps and to the rows that carry the story: the GPU's step and op
# rows on top, the host's Python thread below. The crop is shown about three
# times larger than the whole screenshot could be; the slide labels the two
# halves. Pixel box of the source PNG (1000 x 398): x 285-668, y 208-398.
# ---------------------------------------------------------------------------
def trace_crop():
    from PIL import Image
    src = Image.open(out / "fig-tf-bottleneck-trace.png").convert("RGB")
    src.crop((285, 208, 668, 398)).save(out / "trace_crop.png")


trace_crop()

print("figures written")
