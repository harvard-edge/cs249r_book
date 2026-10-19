#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 8 (Model Training, Part 1).

Inputs are from the rendered chapter (Model Training) unless noted; every
derived value is computed in calcs.py and imported here.
  - GPT-2: 3.0e9 FLOPs per forward token; 1.5e21 FLOPs per run (Sec. 8.1).
  - Training work per token: 2P forward, 4P backward (Lighthouse 8.1).
  - Published MFU on one definition: PaLM Table 3 (Chowdhery et al. 2022):
    GPT-3 21.3, Gopher 32.5, Megatron-Turing NLG 30.2, PaLM 46.2 percent.
    The GPT-2 XL scenario assumes 50 percent (Sec. 8.2).
  - Static state: 2 + 2 + 4 + 8 = 16 bytes per parameter (Sec. 8.3.2).
  - GPT-2 XL at batch 32, 1,024 tokens, mixed precision: static 24 GB,
    activations 286.9 GB (library total), of which 201.3 GB are attention
    scores (Napkin Math 8.3); H100 capacity 80 GiB = 85.9 GB.
  - Llama 3 8B: Napkin Math 8.4.
  - ResNet-50 feeding: Sec. 8.4.2.
  - Optimizer bytes: table 3, Sec. 8.3.2.
Illustrative figures (the PaLM loss-spike shape, the MFU bars of the
busy-time scenarios) say so on the figure. Figures that moved to Lecture 8b
(bandwidth ladder, precision run time, FlashAttention tile, checkpoint
segments, scoreboard) and the dropped pipeline and overlap diagrams are no
longer generated here (2026-10-18).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent
CAP = round(calcs.H100_GIB * 2**30 / calcs.GB, 1)  # 80 GiB in decimal GB


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
# Tick labels at 14 pt so the powers of ten stay legible on the slide.
with sp.plt.rc_context({"xtick.labelsize": 14}):
    sp.hbars_steps(out / "train_vs_token", [
        ("One generated token", 3.0e9, "neutral", "3 × 10⁹ FLOPs"),
        ("One training run", calcs.RUN_FLOP, "compute",
         f"1.5 × 10²¹  = {calcs.RUN_FLOP / calcs.FLOP_PER_TOKEN_FWD / 1e9:.0f} billion tokens"),
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

# 4. Published MFU on one definition (PaLM Table 3), plus our assumption ----
fig, ax = plt.subplots(figsize=(5.8, 2.6))
rows = [(name, v, "compute", f"{v}%") for name, v in calcs.PALM_TABLE3]
rows.append(("GPT-2 XL, this lecture", calcs.UTIL * 100, "neutral",
             f"{calcs.UTIL * 100:.0f}% (assumed)"))
n = len(rows)
for i, (name, v, ck, text) in enumerate(rows):
    y = n - 1 - i
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(-1, y, name, ha="right", va="center", fontsize=11, fontweight="bold",
            color=C["text"])
    ax.text(v + 1.5, y, text, va="center", fontsize=12, fontweight="bold", color=C[ck])
ax.axvline(100, color=C["neutral"], lw=1.5, ls="--")
ax.text(99, n - 0.45, "peak", ha="right", va="bottom", fontsize=11, color=C["neutral"])
ax.set_xlim(0, 100)
ax.set_ylim(-0.6, n - 0.1)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Model FLOPs utilization (percent of peak)")
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
STATIC = calcs.P_GPT2XL * sum(calcs.MP_BYTES) / calcs.GB          # 24 GB
SCORES = calcs.scores_gb(calcs.SEQ)                                 # 201.3 GB
PERTOK = calcs.linear_gb(calcs.SEQ)                                 # 85.6 GB
STEP = STATIC + SCORES + PERTOK                                     # 310.9 GB
parts = [(STATIC, "", "neutral"), (PERTOK, f"activations {PERTOK:.0f}", "data"),
         (SCORES, f"attention scores {SCORES:.0f}", "danger")]
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
    label(ax, 0, -0.38, f"static {STATIC:.0f}", C["neutral"], True, va="top", fontsize=11)
    label(ax, STEP + 1, 0, f" {STEP:.0f} GB", C["text"], k == 3, va="center", fontsize=13)
    ax.axvline(CAP, color=C["memory"], lw=2, ls="--", zorder=0)
    ax.text(CAP + 3, 0.42, f"H100: {CAP} GB (80 GiB)", color=C["memory"], fontsize=11,
            fontweight="bold", va="bottom")
    ax.set_xlim(0, 360)
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("GB for one batch-32 step of GPT-2 XL (FP16 activations)")
    sp.save(fig, out / f"gpt2_b32_step{k}.svg")

# 8. Llama 3 8B fine-tuning scope --------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 2.3))
L_W = calcs.P_LLAMA8B * 2 / calcs.GB                                # 16 GB
L_REST = calcs.P_LLAMA8B * sum(calcs.MP_BYTES[1:]) / calcs.GB       # 112 GB
L_LORA = calcs.LORA_PARAMS * 16 / calcs.GB                          # 0.11 GB
L_ACT = calcs.LLAMA_ACT_GB
rows = [("Full fine-tune", [(L_W, "W", "memory"),
                            (L_REST, f"grads, master, Adam  {L_REST:.0f}", "compute"),
                            (L_ACT, "act.", "data")],
         f"about {int(L_W + L_REST + L_ACT + 0.5)} GB"),
        ("LoRA adapters", [(L_W, "W", "memory"), (L_LORA, "", "compute"),
                           (L_ACT, f"act. {L_ACT}", "data")],
         f"about {int(L_W + L_LORA + L_ACT + 0.5)} GB")]
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
ax.text(CAP + 2, 1.42, f"H100: {CAP} GB (80 GiB)", color=C["memory"], fontsize=11,
        fontweight="bold", va="bottom")
ax.set_xlim(0, 215)
ax.set_ylim(-0.55, 1.75)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("GB on one accelerator, Llama 3 8B (W: 16 GB of 16-bit weights)")
sp.save(fig, out / "llama_finetune.svg")

# 10. ResNet-50: accelerator demand against host supply -----------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    gpus = ["A100", "H100"]
    demand = [calcs.RESNET_DEMAND_A100, calcs.RESNET_DEMAND_H100]
    supply = calcs.HOST_WORKERS * calcs.PER_WORKER
    for i, g in enumerate(gpus):
        shown = i < k
        a = 1.0 if shown else 0.0
        ax.bar(i - 0.2, demand[i], 0.38, color=C["compute"], alpha=a)
        ax.bar(i + 0.2, supply, 0.38, color=C["data"], alpha=a)
        label(ax, i - 0.2, demand[i] + 250, f"{demand[i]:,}", C["compute"], shown,
              ha="center", fontsize=12)
        label(ax, i + 0.2, supply + 250, f"{supply:,}", C["data"], shown, ha="center",
              fontsize=12)
    label(ax, 0.0, 6300, "fed", C["data"], k >= 1, ha="center", fontsize=14)
    label(ax, 1.0, 11300, f"busy {100 * supply / demand[1]:.0f}%", C["danger"], k == 2, ha="center", fontsize=14)
    ax.text(-0.5, 14300, "accelerator demand", color=C["compute"], fontsize=12,
            fontweight="bold")
    ax.text(-0.5, 13200, f"host supply ({calcs.HOST_WORKERS} workers)", color=C["data"], fontsize=12,
            fontweight="bold")
    ax.set_xticks([0, 1], gpus)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 15200)
    ax.set_yticks([0, 4000, 8000, 12000], ["0", "4k", "8k", "12k"])
    ax.set_ylabel("Images per second")
    sp.save(fig, out / f"resnet_feed_step{k}.svg")

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
#     Derived in calcs.py from Napkin Math 8.3: the scores are
#     48 x 5 x 25 x S^2 x 32 bytes (201.3 GB at S = 1,024); the per-token part
#     is the library total, 286.9 GB, minus the scores (85.6 GB at 1,024),
#     scaled with S. Total at 1,024: 286.9 GB, the same as the 311 GB slide.
seq = np.linspace(128, 2048, 200)
lin = np.array([calcs.linear_gb(s) for s in seq])
score = np.array([calcs.scores_gb(s) for s in seq])
AT1024 = calcs.linear_gb(calcs.SEQ) + calcs.scores_gb(calcs.SEQ)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.fill_between(seq, 0, lin, color=C["data"], lw=0)
    ax.fill_between(seq, lin, lin + score, color=C["danger"], lw=0,
                    alpha=1.0 if k == 2 else 0.0)
    ax.axhline(CAP, color=C["memory"], lw=2, ls="--")
    ax.text(140, CAP + 14, f"H100: {CAP} GB", color=C["memory"], fontsize=11,
            fontweight="bold")
    ax.text(150, 960, "green: per-token tensors (projections, FFN,\nlayer inputs), grow with S", color=C["data"],
            fontsize=11, fontweight="bold", ha="left", va="top")
    label(ax, 1350, 620, "attention scores:\ngrow with S²", C["danger"], k == 2,
          ha="right", fontsize=11.5)
    ax.plot([1024], [AT1024],
            "o", ms=9, color=C["danger"], mec="white", mew=1.5, alpha=1.0 if k == 2 else 0.0)
    label(ax, 990, 420, f"1,024 tokens: {AT1024:.0f} GB", C["danger"], k == 2, ha="right",
          fontsize=11.5)
    ax.set_xlim(128, 2048)
    ax.set_ylim(0, 1050)
    ax.set_xticks([256, 512, 1024, 1536, 2048], ["256", "512", "1,024", "1,536", "2,048"])
    ax.set_xlabel("Sequence length S (batch 32; derived from Napkin Math 8.3)", fontsize=11)
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
