#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 8 (Model Training, Part 1).

Every number is read from MLSysIM through calcs.py (val() for a printed
label, num() for a plotted value, or the scenario itself for a curve):
  - GPT-2 FLOPs per generated token and per run (Models.Language.GPT2).
  - Training work per token: 2P forward, 4P backward (the 6P rule).
  - Published MFU on one definition: PaLM Table 3 (Literature.TrainingRuns);
    the GPT-2 XL scenario's assumed utilization (Gpt2XlRunOnOneH100).
  - Bytes per parameter of SGD, momentum, and Adam in FP32, and of
    mixed-precision Adam (TrainingStateFitScenario).
  - GPT-2 XL at batch 32, 1,024 tokens, mixed precision: static state,
    per-token activations, and attention scores (Gpt2XlBatch32Activations);
    the H100's 80 GiB in decimal GB.
  - Llama 3.1 8B full and LoRA fine-tuning (Llama3_8B*OnH100).
  - ResNet-50 feeding (ResNet50H100InputFeed).
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
from deckvals import val, num  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent
CAP = num("h100_gb")  # 80 GiB in decimal GB
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci_label(key):
    """A value recorded as key_m, key_e, printed as m × 10ᵉ for a plot label."""
    return f"{val(key + '_m')} × 10{val(key + '_e').translate(SUP)}"


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
    label(ax, 250, loss[200] - 0.85, f"rewind about {val('palm_rewind')} steps", C["memory"], show,
          ha="center", va="top", fontsize=11)
    ax.axvspan(300, 330, color=C["data_l"], alpha=0.9 if show else 0.0, lw=0)
    label(ax, 335, 4.2, f"skip {val('palm_skip_lo')} to\n{val('palm_skip_hi')} batches", C["data"], show, ha="left",
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
        ("One generated token", calcs.GPT2.inference_flops.to("flop").magnitude, "neutral",
         f"{sci_label('gpt2_fwd')} FLOPs"),
        ("One training run", calcs.GPT2.training_ops.to("flop").magnitude, "compute",
         f"{sci_label('gpt2_run')}  = {val('tokens_ratio_b')} billion tokens"),
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
PALM_TABLE3 = [("GPT-3 175B, V100", "mfu_gpt3"),
               (f"Gopher 280B, {val('gopher_chips')} TPU v3", "mfu_gopher"),
               (f"Megatron-Turing NLG 530B, {val('mtnlg_gpus')} A100", "mfu_mtnlg"),
               (f"PaLM 540B, {val('palm_chips')} TPU v4", "mfu_palm")]
rows = [(name, num(key), "compute", f"{val(key)}%") for name, key in PALM_TABLE3]
rows.append(("GPT-2 XL, this lecture", num("eta_pct"), "neutral",
             f"{val('eta_pct')}% (assumed)"))
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
W_B, MASTER_B, MOM_B, TOT_B = (round(num(k)) for k in ("mp_w_b", "mp_master_b", "mp_moments_b", "mp_total_b"))
rows = [("Inference", [(W_B, "W", "memory")]),
        ("Training", [(W_B, "W", "memory"), (W_B, "G", "network"),
                      (MASTER_B, "master", "neutral"), (MOM_B, "Adam m, v (FP32)", "compute")])]
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
            color=C["danger"] if x == TOT_B else C["text"])
ax.set_xlim(0, 18.8)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([])
ax.set_xticks([0, W_B, W_B + W_B, MASTER_B + MASTER_B, TOT_B])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Bytes per parameter before activations (W, G: FP16 weights, gradients; master: FP32)", fontsize=11)
sp.save(fig, out / "static_bytes.svg")

# 7. GPT-2 XL at batch 32, mixed precision: 311 GB against 85.9 GB ------------
STATIC = calcs.MIXED.total.to("GB").magnitude                     # 24 GB
SCORES = calcs.ACT.score_activations.to("GB").magnitude            # 201.3 GB
PERTOK = calcs.ACT.per_token_activations.to("GB").magnitude        # 85.6 GB
STEP = calcs.MIXED.training_total.to("GB").magnitude               # 310.9 GB
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
    ax.text(CAP + 3, 0.42, f"H100: {val('h100_gb')} GB ({val('h100_gib')} GiB)", color=C["memory"], fontsize=11,
            fontweight="bold", va="bottom")
    ax.set_xlim(0, 360)
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"GB for one batch-{val('batch')} step of GPT-2 XL (FP16 activations)")
    sp.save(fig, out / f"gpt2_b32_step{k}.svg")

# 8. Llama 3 8B fine-tuning scope --------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 2.3))
L_W = calcs.FULL.weights.to("GB").magnitude                       # 16 GB
L_REST = num("l8_rest_gb")                                         # 112 GB
L_LORA = calcs.LORA.adapter_state.to("GB").magnitude               # 0.11 GB
L_ACT = num("l8_act_gb")
rows = [("Full fine-tune", [(L_W, "W", "memory"),
                            (L_REST, f"grads, master, Adam  {val('l8_rest_gb')}", "compute"),
                            (L_ACT, "act.", "data")],
         f"about {val('l8_full_gb')} GB"),
        ("LoRA adapters", [(L_W, "W", "memory"), (L_LORA, "", "compute"),
                           (L_ACT, f"act. {val('l8_act_gb')}", "data")],
         f"about {val('l8_lora_gb')} GB")]
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
ax.text(CAP + 2, 1.42, f"H100: {val('h100_gb')} GB ({val('h100_gib')} GiB)", color=C["memory"], fontsize=11,
        fontweight="bold", va="bottom")
ax.set_xlim(0, 215)
ax.set_ylim(-0.55, 1.75)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel(f"GB on one accelerator, Llama 3 8B (W: {val('l8_w_gb')} GB of 16-bit weights)")
sp.save(fig, out / "llama_finetune.svg")

# 10. ResNet-50: accelerator demand against host supply -----------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    gpus = ["A100", "H100"]
    demand = [num("res_a100"), num("res_h100")]
    supply = num("res_supply")
    for i, g in enumerate(gpus):
        shown = i < k
        a = 1.0 if shown else 0.0
        ax.bar(i - 0.2, demand[i], 0.38, color=C["compute"], alpha=a)
        ax.bar(i + 0.2, supply, 0.38, color=C["data"], alpha=a)
        label(ax, i - 0.2, demand[i] + 250, val(("res_a100", "res_h100")[i]), C["compute"], shown,
              ha="center", fontsize=12)
        label(ax, i + 0.2, supply + 250, val("res_supply"), C["data"], shown, ha="center",
              fontsize=12)
    label(ax, 0.0, 6300, "fed", C["data"], k >= 1, ha="center", fontsize=14)
    label(ax, 1.0, 11300, f"busy {val('res_busy_pct')}%", C["danger"], k == 2, ha="center", fontsize=14)
    ax.text(-0.5, 14300, "accelerator demand", color=C["compute"], fontsize=12,
            fontweight="bold")
    ax.text(-0.5, 13200, f"host supply ({val('res_workers')} workers)", color=C["data"], fontsize=12,
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
F = round(num("fp32_b"))
opt_rows = [("SGD", [(F, "W", "memory"), (F, "G", "network")]),
            ("Momentum", [(F, "W", "memory"), (F, "G", "network"), (F, "v", "compute")]),
            ("Adam", [(F, "W", "memory"), (F, "G", "network"), (F, "m", "compute"),
                      (F, "v", "compute")])]
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
        label(ax, x + 0.4, y, f"{x} B", C["danger"] if x == round(num("adam_fp32_b")) else C["text"], shown,
              va="center", fontsize=13)
    ax.set_xlim(0, 19.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xticks([0, F, F + F, F + F + F, F + F + F + F])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Bytes per parameter in FP32 (W weights, G gradients, m and v optimizer state)",
                  fontsize=10.5)
    sp.save(fig, out / f"optimizer_bytes_step{k}.svg")

# 18. GPT-2 XL activations against sequence length, batch 32, FP16 ------------
#     From Scenarios.Gpt2XlBatch32Activations at each sequence length: the
#     scores grow with S^2 (201.3 GB at S = 1,024); the per-token tensors grow
#     with S (85.6 GB at 1,024). Total at 1,024: 286.9 GB, as on the 311 GB slide.
seq = np.linspace(128, 2048, 200)
at = [calcs.ACT.variant(seq_len=int(round(s))) for s in seq]
lin = np.array([a.per_token_activations.to("GB").magnitude for a in at])
score = np.array([a.score_activations.to("GB").magnitude for a in at])
AT1024 = calcs.ACT.activations("none").to("GB").magnitude
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.fill_between(seq, 0, lin, color=C["data"], lw=0)
    ax.fill_between(seq, lin, lin + score, color=C["danger"], lw=0,
                    alpha=1.0 if k == 2 else 0.0)
    ax.axhline(CAP, color=C["memory"], lw=2, ls="--")
    ax.text(140, CAP + 14, f"H100: {val('h100_gb')} GB", color=C["memory"], fontsize=11,
            fontweight="bold")
    ax.text(150, 960, "green: per-token tensors (projections, FFN,\nlayer inputs), grow with S", color=C["data"],
            fontsize=11, fontweight="bold", ha="left", va="top")
    label(ax, 1350, 620, "attention scores:\ngrow with S²", C["danger"], k == 2,
          ha="right", fontsize=11.5)
    ax.plot([calcs.ACT.sequence], [AT1024],
            "o", ms=9, color=C["danger"], mec="white", mew=1.5, alpha=1.0 if k == 2 else 0.0)
    label(ax, 990, 420, f"{val('seq')} tokens: {val('act_gb')} GB", C["danger"], k == 2, ha="right",
          fontsize=11.5)
    ax.set_xlim(128, 2048)
    ax.set_ylim(0, 1050)
    ax.set_xticks([256, 512, 1024, 1536, 2048], ["256", "512", "1,024", "1,536", "2,048"])
    ax.set_xlabel(f"Sequence length S (batch {val('batch')}; derived from Napkin Math 8.3)", fontsize=11)
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
