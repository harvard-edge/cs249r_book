#!/usr/bin/env python3
"""Figures for Lecture 15b (Responsible Engineering, Part 2: Cost, Carbon,
and Evidence).

Plots use the shared slideplot helper; diagrams are written as SVG with the
same small builder Lecture 15 uses, one file per build step, every step on the
same canvas so nothing moves. Every value is sourced:

  - Architecture-search estimate: Strubell, Ganesh, McCallum, ACL 2019,
    Table 1: 626,155 lb (284 t) for Transformer (big) with neural
    architecture search; one car over its lifetime 126,000 lb (57 t).
    Measured: Patterson et al. 2021 (arXiv 2104.10350), Table 4 and
    Sec. 4.1: 3.2 net tCO2e, "88X off in emissions for energy-efficient
    organizations like Google", 18.7X too high for the average organization.
  - Gemini median text prompt, May 2025: Elsworth et al. 2025 (arXiv
    2508.15734), Table 1 (0.14 accelerators, 0.06 host CPU and DRAM, 0.02 idle
    machines, 0.02 overhead; 0.24 Wh), Table 2 (0.023 g Scope 2 market-based,
    0.010 g Scope 1+3, 0.03 g total), Sec. 4.3 (33x energy, 44x emissions
    over 12 months; fleet PUE 1.09).
  - Four factors on one training job: Patterson et al. 2021, Table 1
    (0.1357 t for Transformer (Big) on P100 in an average US datacenter;
    0.0024 t net for Evolved Transformer (Medium) on TPU v2 in Google Iowa)
    and Figure 1 (cumulative CO2e gain read from its labels: 1.3, 7.4, 10.5,
    56.5). Only these printed values are drawn.
  - Four 175B-scale models: Luccioni, Viguier, Ligozat 2023 (arXiv
    2211.02001), Table 4, CO2e x PUE: GPT-3 552 t, Gopher 380 t, OPT 76.3 t,
    BLOOM 30 t. Entries in italics in that table were inferred by the
    authors: GPT-3's 502 t before PUE; Gopher's 1,066 MWh and 352 t; OPT's
    PUE 1.09, 231 g/kWh, 324 MWh, and 76.3 t. Those are starred here.
  - BLOOM life cycle: Luccioni et al. 2023, Table 3: embodied 11.2 t
    (22.2 percent), dynamic 24.69 t (48.9 percent), idle 14.6 t
    (28.9 percent), total 50.5 t.
  - DLRM three-year TCO (book scenario, chapter Sec. 15.4.2 and table 10):
    200 million inferences a day, 10 ms unbatched, 556 GPU-hours a day,
    23.1 GPUs, $2.50 per GPU-hour, $506,944 a year; training $38K, inference
    $1.52M, operations $510K; modeled accelerator carbon 1.5 t and 97.3 t.
    Checkpoint 15.4 and Summary: a 20 percent serving cut saves $304K and
    about 19 t; halving all training saves $19.2K and 0.75 t (derived).
  - Measured fleets: Patterson et al. 2022 (inference about 3/5 of Google's
    ML energy, 2019 to 2021); Wu et al. 2022, Fig. 3 (Meta AI power capacity
    10:20:70 for experimentation, training, inference).
  - Embodied share per GPU-hour (derived): Luccioni et al. 2023 Sec. 4.1,
    0.056 kg per server-hour and 0.003 kg per GPU-hour; 8 GPUs per server
    (assumption), so 0.003 + 0.056 / 8 = 0.010 kg per GPU-hour. Operational
    0.4 kW x 0.4 kg/kWh = 0.160 kg (Napkin Math 15.4) and 0.4 kW x 0.057 =
    0.023 kg (French grid, Luccioni et al. 2023).
  - Explanation cost (chapter Sec. 15.3.3 and 15.5.3): 1 million decisions a
    day; 47.5 percent denied (constructed credit service, 6,300 of 12,000
    approved); 1,000-sample attributor; exact Shapley 2^20 = 1,048,576.
  - Review staffing: chapter Sec. 15.5.3 (0.1 percent of 1 million = 1,000
    cases a day); 6 minutes per case (assumption); ProPublica and The Capitol
    Forum, March 25, 2023: Cigna doctors denied over 300,000 requests in two
    months, 1.2 seconds each on average.
  - Machine unlearning: Bourtoule et al. 2021 (arXiv 1912.03817), abstract:
    SISA unlearning 4.63x, 2.45x, 1.36x faster than retraining (Purchase,
    SVHN, ImageNet).
  - Differentially private accuracy: De et al. 2022 (arXiv 2204.13650v2),
    abstract: CIFAR-10 81.4 percent at (8, 1e-5)-DP from scratch, previous
    best 71.7; Sec. 3: WRN-40-4 above 95 percent without privacy; Table 6:
    ImageNet 86.7 percent at (8, 8e-7)-DP fine-tuning an NFNet-F3
    pre-trained on JFT-4B, 88.5 for the same model without privacy; Fig. 1:
    the best non-private ImageNet model 91.0 (Yu et al. 2022), 4.3 points
    above.
  - Raw audio against a model update: 16 kHz, 16-bit audio (assumption)
    = 32,000 bytes a second, 115.2 MB an hour; 200K FP32 parameters = 800 KB
    (chapter Sec. 15.5.2); ratio 144.
  - Audit volume (chapter Sec. 15.5.5 and table 12): 1 million decisions a
    day x 256 bytes (assumption) x 760 days (25 months, 12 CFR
    1002.12(b)(1)) = 195 GB. DLRM, 200 million a day for a year: 18.7 TB of
    256-byte manifests against 976 TB of 13,364-byte inputs (constructed:
    13 dense and 26 categorical features with 128-dimensional FP32 embeddings,
    as in MLPerf DLRM on Criteo Terabyte), 52x.

Moved from Lecture 15 (same generator code, same sources): tco_step*,
fleet_shares_step*, dpsgd_step1, the lineage graph (rebuilt in four steps).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as K  # noqa: E402  (every derived value on a figure comes from here)

plt = sp.plt
C = sp.C
out = Path(__file__).parent


# =============================================================================
# Plot helpers
# =============================================================================

def bar_steps(stem, rows, steps, size, xmax, xlabel, note=None, vlines=None,
              log=False, xmin=None, label_size=12):
    """Horizontal bars with per-step visibility.

    rows: (label, value, color_key, value_text, first_step). Every artist is
    drawn in every step (hidden ones transparent) so the figure never moves.
    """
    n = len(rows)
    for k in range(1, steps + 1):
        fig, ax = plt.subplots(figsize=size)
        for i, (label, v, ck, txt, first) in enumerate(rows):
            y = n - 1 - i
            on = k >= first
            ax.barh(y, v, color=C[ck], height=0.62, alpha=1.0 if on else 0.0)
            ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=label_size, color=C["text"],
                    fontweight="bold", alpha=1.0 if on else 0.0)
            x = v * 1.15 if log else v + xmax * 0.012
            ax.text(x, y, txt, va="center", fontsize=label_size, color=C[ck],
                    fontweight="bold", alpha=1.0 if on else 0.0)
        for x, label, ck, first in (vlines or []):
            on = k >= first
            ax.axvline(x, color=C[ck], lw=1.6, ls="--", zorder=1, alpha=1.0 if on else 0.0)
            ax.text(x, n - 0.45, " " + label, color=C[ck], fontsize=10,
                    fontweight="bold", va="bottom", ha="left", alpha=1.0 if on else 0.0)
        ax.set_yticks([])
        if log:
            ax.set_xscale("log")
            ax.set_xlim(xmin, xmax)
        else:
            ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        if note:
            ax.text(0.0, -0.78 / (size[1] * 0.77), note, transform=ax.transAxes,
                    fontsize=9, color=C["neutral"])
        sp.save(fig, out / f"{stem}_step{k}.svg")


def stacked(ax, y, parts, alpha=1.0, height=0.6):
    """parts: (value, color_key, inside_text). Text goes inside when given."""
    left = 0
    for v, ck, txt in parts:
        ax.barh(y, v, left=left, height=height, color=C[ck], edgecolor="white",
                alpha=alpha)
        if txt:
            dark = ck in ("compute", "memory", "data", "network", "neutral", "danger")
            ax.text(left + v / 2, y, txt, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if dark else C["text"], alpha=alpha)
        left += v
    return left


# =============================================================================
# Segment 1: joules to tonnes
# =============================================================================

# Hook: estimated against measured (log scale)
_hook_rows = [
    ("One car, whole lifetime", K.CAR_T, "neutral", f"{K.CAR_T:.0f} t"),
    ("Search, estimated in 2019", K.NAS_EST_T, "compute",
     f"{K.NAS_EST_T:.0f} t = {K.NAS_EST_T / K.CAR_T:.0f} car lifetimes"),
    # 3.2 t and 88x are the values Patterson et al. print (Table 4, App. D)
    ("Same search, measured in 2021", 3.2, "data", "3.2 t, 88× lower"),
]
bar_steps("hook_nas", [r + (3 if i == 2 else 1,) for i, r in enumerate(_hook_rows)],
          3, (5.8, 2.0), 30000, "Tonnes of CO₂e (log scale)", log=True, xmin=1)
# step 2: overwrite with the empty slot and ??? for the measured bar
for k in (2,):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    for i, (label, v, ck, txt) in enumerate(_hook_rows):
        y = 2 - i
        on = i < 2
        ax.barh(y, v, color=C[ck], height=0.62, alpha=1.0 if on else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(v * 1.15, y, txt, va="center", fontsize=12, fontweight="bold",
                color=C[ck], alpha=1.0 if on else 0.0)
    ax.text(10, 0, "???", ha="center", va="center", fontsize=16, fontweight="bold",
            color=C["danger"])
    ax.set_xscale("log")
    ax.set_xlim(1, 30000)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Tonnes of CO₂e (log scale)")
    sp.save(fig, out / "hook_nas_step2.svg")

# Gemini median prompt: energy at the full boundary, then carbon.
# Redrawn 2026-10-19 at its slot (about 13 x 4.2 cm) so labels stay above
# 6.5 pt on the page.
for k in (1, 2):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 1.75),
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 1.6})
    stacked(ax1, 0, [(0.14, "compute", "Accelerators 0.14"),
                     (0.06, "memory", "Host 0.06"),
                     (0.02, "neutral_l", ""), (0.02, "neutral", "")])
    ax1.annotate("Idle 0.02", (0.21, 0.32), xytext=(0.212, 0.85),
                 ha="right", va="bottom", fontsize=11, fontweight="bold", color=C["text"],
                 arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=1))
    ax1.annotate("Overhead 0.02 (PUE 1.09)", (0.23, 0.32), xytext=(0.228, 0.85),
                 ha="left", va="bottom", fontsize=11, fontweight="bold", color=C["text"],
                 arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=1))
    ax1.text(0.255, 0, f"= {K.GEM_WH:.2f} Wh", va="center", fontsize=13, fontweight="bold",
             color=C["text"])
    ax1.set_xlim(0, 0.36)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_xlabel("Energy per prompt (Wh)")
    a = 1.0 if k >= 2 else 0.0
    stacked(ax2, 0, [(0.023, "network", "Grid 0.023"),
                     (0.010, "neutral", "")], alpha=a)
    ax2.text(0.0335, 0, f" + 0.010 Scope 1 and 3  = {K.GEM_TOTAL_G:.2f} g", va="center",
             fontsize=12, fontweight="bold", color=C["text"], alpha=a)
    ax2.set_xlim(0, 0.0473 * 0.36 / 0.34)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_alpha(a)
    ax2.tick_params(axis="x", colors=C["text"] if k >= 2 else "white")
    ax2.set_xlabel("CO₂e per prompt (g)", color=C["text"] if k >= 2 else "white")
    sp.save(fig, out / f"gemini_step{k}.svg")

# Four factors on one training job: cumulative gain read from Patterson Fig. 1.
# Redrawn 2026-10-19 for a 0.58-width column (about 8.5 x 5 cm) beside the
# Iowa photo and the grid map.
gains = [("Base\nP100", 1.0, "neutral", "1×", 1),
         ("Model\nEvolved", K.CUM["model"], "compute", "1.3×", 1),
         ("Chip\nTPU v2", K.CUM["processor"], "compute", "7.4×", 2),
         ("PUE\n1.11", K.CUM["datacenter"], "memory", "10.5×", 3),
         ("Grid\n0.080", K.CUM["grid"], "data", "56.5×", 4)]
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(3.7, 2.05))
    for i, (lab, v, ck, txt, first) in enumerate(gains):
        on = k >= first
        ax.bar(i, v, width=0.66, color=C[ck], alpha=1.0 if on else 0.0)
        ax.text(i, v * 1.12, txt, ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=C[ck], alpha=1.0 if on else 0.0)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 200)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1×", "10×", "100×"])
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels([g[0] for g in gains], fontsize=9.5)
    ax.set_xlim(-0.6, len(gains) - 0.4)
    ax.set_ylabel("CO₂e cut, cumulative", fontsize=11)
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"factors_step{k}.svg")

# Four 175B-scale models (Luccioni et al. 2023, Table 4, CO2e x PUE)
bar_steps("llm4", [
    ("GPT-3", 552, "danger", "552 t: 1,287 MWh at 429 g/kWh", 1),
    ("Gopher", 380, "neutral", "380 t: 1,066 MWh* at 330 g/kWh", 2),
    ("OPT", 76.3, "neutral", "76.3 t*: 324 MWh* at 231 g/kWh*", 2),
    ("BLOOM", 30, "data", "30 t: 433 MWh at 57 g/kWh", 1),
], 2, (5.8, 2.3), 1150, "Training emissions × PUE (t CO₂e)", label_size=12)
# The footnote ("* inferred by Luccioni et al.") is slide text since 2026-10-19.

# BLOOM life cycle (Luccioni et al. 2023, Table 3)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.8))
    parts = [(24.69, "compute", "Dynamic\n24.7 t", 1), (14.6, "neutral", "Idle\n14.6 t", 2),
             (11.2, "network", "Embodied\n11.2 t", 3)]
    left = 0
    for v, ck, txt, first in parts:
        a = 1.0 if k >= first else 0.0
        ax.barh(0, v, left=left, height=0.7, color=C[ck], edgecolor="white", alpha=a)
        ax.text(left + v / 2, 0, txt, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", alpha=a)
        left += v
    ax.text(K.B_DYN / 2, 0.5, f"What most papers report ({K.B_DYN_SHARE:.1f}%)", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C["compute"])
    ax.text(51.2, 0, f"= {K.B_TOTAL:.1f} t", va="center", fontsize=13, fontweight="bold",
            color=C["text"], alpha=1.0 if k >= 3 else 0.0)
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.5, 0.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("BLOOM training, t CO₂e")
    sp.save(fig, out / f"bloom_life_step{k}.svg")


# =============================================================================
# Segment 2: over the system's life
# =============================================================================

# DLRM three-year TCO and modeled accelerator carbon (moved from Lecture 15)
for k in (1, 2):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 2.9),
                                   gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.9})
    parts = [("Training", K.TRAIN_3Y / 1e6, "neutral"), ("Inference", K.INF_3Y / 1e6, "compute"),
             ("Operations", K.OPS_3Y / 1e6, "neutral_l")]
    left = 0
    for name, v, ck in parts:
        ax1.barh(0, v, left=left, height=0.6, color=C[ck], edgecolor="white")
        left += v
    ax1.text(0.04, 0.55, f"Training ${K.TRAIN_3Y / 1e3:.0f}K ({K.TRAIN_SHARE:.1f}%)", fontsize=11, color=C["text"],
             va="bottom", fontweight="bold")
    ax1.text(K.TRAIN_3Y / 1e6 + K.INF_3Y / 2e6, 0, f"Inference ${K.INF_3Y / 1e6:.2f}M ({K.INF_SHARE:.1f}%)", fontsize=11,
             color="white", ha="center", va="center", fontweight="bold")
    ax1.text((K.TRAIN_3Y + K.INF_3Y + K.OPS_3Y / 2) / 1e6, 0, f"Ops ${K.OPS_3Y / 1e3:.0f}K", fontsize=11, ha="center",
             va="center", color=C["text"], fontweight="bold")
    ax1.set_xlim(0, 2.2)
    ax1.set_ylim(-0.4, 1.2)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_xlabel("Three-year cost (US$ millions)")
    a = 1.0 if k == 2 else 0.0
    for y, (name, v, ck) in zip((1, 0), (("12 training cycles", round(K.TRAIN_T, 1), "neutral"),
                                         ("3 years of inference", round(K.INF_T, 1), "compute"))):
        ax2.barh(y, v, height=0.6, color=C[ck], alpha=a)
        ax2.text(-0.01, y, name, transform=ax2.get_yaxis_transform(), ha="right",
                 va="center", fontsize=11, fontweight="bold", color=C["text"], alpha=a)
        ax2.text(v + 1.5, y, f"{v:g} t", va="center", fontsize=11, fontweight="bold",
                 color=C[ck], alpha=a)
    ax2.text(118, 0, f"{K.CARBON_RATIO_DLRM:.0f}×", va="center", fontsize=14, fontweight="bold",
             color=C["danger"], alpha=a)
    ax2.set_xlim(0, 135)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_alpha(a)
    ax2.tick_params(axis="x", colors=C["text"] if k == 2 else "white")
    ax2.set_xlabel("Modeled accelerator carbon (t CO₂e)",
                   color=C["text"] if k == 2 else "white")
    sp.save(fig, out / f"tco_step{k}.svg")

# Estimate: serving cut against halving all training (book scenario)
bar_steps("saving", [
    ("Cut serving cost 20%", K.SAVE_SERVE / 1e3, "compute",
     f"${K.SAVE_SERVE / 1e3:.0f}K, about {K.SAVE_SERVE_T:.0f} t", 1),
    ("Halve all training", K.SAVE_TRAIN / 1e3, "neutral",
     f"${K.SAVE_TRAIN / 1e3:.0f}K, {K.SAVE_TRAIN_T:.1f} t", 1),
], 1, (4.6, 1.3), 520, "Three-year saving (US$ thousands)")

# Measured fleets (moved from Lecture 15; two rows)
fleet = [("Google, ML energy, 2019 to 2021", [("Training ~2/5", 40, "neutral_l"),
                                             ("Inference ~3/5", 60, "compute")], 1),
         ("Meta, AI power capacity", [("10", 10, "neutral_l"), ("Train 20", 20, "neutral"),
                                     ("Inference 70", 70, "compute")], 2)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.9, 2.3))
    for i, (label, segs, first) in enumerate(fleet):
        y = (len(fleet) - 1 - i) * 1.6
        a = 1.0 if k >= first else 0.0
        left = 0
        for name, v, ck in segs:
            ax.barh(y, v, left=left, height=0.62, color=C[ck], edgecolor="white", alpha=a)
            dark = ck in ("compute", "neutral")
            ax.text(left + v / 2, y, name, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white" if dark else C["text"], alpha=a)
            left += v
        ax.text(0, y + 0.42, label, ha="left", va="bottom", fontsize=11,
                fontweight="bold", color=C["text"], alpha=a)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share (%)")
    sp.save(fig, out / f"fleet2_step{k}.svg")

# Embodied share of one GPU-hour on two grids (derived)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    rows = [("Grid 0.4 kg/kWh", K.OP_DIRTY_G, K.EMB_G, f"embodied {K.EMB_SHARE_DIRTY:.0f}%"),
            ("Grid 0.057 kg/kWh", K.OP_CLEAN_G, K.EMB_G, f"embodied {K.EMB_SHARE_CLEAN:.0f}%")]
    for i, (lab, op, emb, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, op, height=0.6, color=C["neutral_l"], edgecolor="white")
        a = 1.0 if k >= 2 else 0.0
        ax.barh(y, emb, left=op, height=0.6, color=C["network"], edgecolor="white", alpha=a)
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(op + emb + 3, y, f"{op:.0f} + {emb:.1f} g   {txt}" if k >= 2 else f"{op:.0f} g",
                va="center", fontsize=12, fontweight="bold",
                color=C["network"] if k >= 2 else C["text"])
    ax.set_xlim(0, 260)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("g CO₂e per GPU-hour: operational (gray) + embodied (violet)")
    sp.save(fig, out / f"embodied_step{k}.svg")


# =============================================================================
# Segment 3: duties have machine costs
# =============================================================================

bar_steps("explain", [
    ("Decisions", K.DECISIONS, "memory", "1 million", 1),
    ("Reasons, 1,000-sample attributor", K.SAMPLED, "compute",
     f"{K.SAMPLED / 1e6:.0f} million, {K.SAMPLED_X:.0f}×", 2),
    ("Reasons, exact (2²⁰ per denial)", K.EXACT, "danger", f"{K.EXACT / 1e11:.0f} × 10¹¹", 3),
], 3, (5.8, 1.7), 1e15, "Model evaluations per day (log scale)", log=True, xmin=1e5)


# Time per case against cases per reviewer-hour (2026-10-18; replaces the
# book's monitoring margin figure, book_margin_002, whose axes read "time per
# alert"). Time per case = 3,600 s / cases per hour. Points: 6 min each
# (assumption) = 10 an hour, 1,000 per 100 h; Cigna 1.2 s each (ProPublica,
# 2023) = 3,000 an hour, 300,000 per 100 h. The dashed minimum is
# illustrative and carries no number.
review_min_s = 120  # illustrative position of the minimum, not a sourced value
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    xs = [10 ** (i / 50) for i in range(0, 201)]
    ax.plot(xs, [3600 / x for x in xs], color=C["text"], lw=1.8)
    ax.axhspan(0.1, review_min_s, color=C["danger"], alpha=0.10, lw=0)
    ax.axhline(review_min_s, color=C["danger"], lw=1.5, ls="--")
    ax.text(9000, review_min_s / 1.4, "Minimum for a\nsubstantive review\n(illustrative)",
            fontsize=11, color=C["danger"], fontweight="bold", va="top", ha="right")
    ax.text(1.15, 2.0, "Rubber-stamp zone", fontsize=12, color=C["danger"],
            fontweight="bold", va="center")
    ax.plot([K.PER_H_CREDIT], [3600 / K.PER_H_CREDIT], "o", color=C["memory"], ms=8)
    ax.text(14, 520, "Credit service: 6 min each,\n1,000 per 100 reviewer-hours",
            fontsize=11, color=C["memory"], fontweight="bold", va="bottom")
    a = 1.0 if k >= 2 else 0.0
    ax.plot([K.PER_H_CIGNA], [3600 / K.PER_H_CIGNA], "o", color=C["danger"], ms=8, alpha=a)
    ax.text(2400, 0.42, "Cigna: 1.2 s each,\n300,000 per 100 h ", fontsize=11,
            color=C["danger"], fontweight="bold", ha="right", va="center", alpha=a)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 10000)
    ax.set_ylim(0.15, 20000)
    ax.set_yticks([1, 60, 3600])
    ax.set_yticklabels(["1 s", "1 min", "1 h"])
    ax.set_xlabel("Cases per reviewer-hour (log scale)")
    ax.set_ylabel("Time per case")
    ax.grid(color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"review_zone_step{k}.svg")


# =============================================================================
# Segment 4: governance as a data system
# =============================================================================

bar_steps("sisa", [
    ("Purchase", 4.63, "data", "4.63×", 1),
    ("SVHN", 2.45, "data", "2.45×", 1),
    ("ImageNet", 1.36, "data", "1.36×", 1),
], 1, (4.2, 1.7), 6.2, "Unlearning speedup over retraining")

# DP-SGD per-sample gradient memory (moved from Lecture 15)
# Redrawn 2026-10-19 for a half-width column beside the Opacus code: labels
# sit above the bars so the plot can be narrow.
fig, ax = plt.subplots(figsize=(3.0, 1.25))
ax.barh(0.9, K.KWS_BATCH_MB, height=0.45, color=C["memory"])
ax.text(36, 1.17, f"KWS: {K.KWS_BATCH_MB:.1f} MB", fontsize=11,
        fontweight="bold", color=C["memory"], va="bottom")
ax.barh(0, K.GPT2_BATCH_TB * 1e6, height=0.5, color=C["danger"])
ax.text(40, 0, f"GPT-2: {K.GPT2_BATCH_TB:.1f} TB", fontsize=11, fontweight="bold",
        color="white", va="center")
ax.axvline(8e4, color=C["neutral"], lw=1.6, ls="--")
# dashed line: one H100, 80 GB (labeled in the slide text)
ax.set_xscale("log")
ax.set_xlim(30, 1e8)
ax.set_ylim(-0.45, 1.65)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("Batch of 256, MB (log)", fontsize=11)
sp.save(fig, out / "dpsgd_step1.svg")

# Corrected 2026-10-19: the ImageNet result fine-tunes a model pre-trained on
# a large public dataset (JFT-4B), and 4.3 points is the gap to the best
# non-private model of any kind; the same model without privacy is 1.8 points
# higher (De et al. 2022, Table 6). CIFAR-10 is trained from scratch; the
# same architecture reaches above 95 percent without privacy.
bar_steps("dp_accuracy", [
    ("CIFAR-10, no privacy", K.CIFAR_NONPRIV, "neutral", "above 95%", 1),
    ("CIFAR-10, ε = 8, from scratch", K.CIFAR_DP, "data",
     f"{K.CIFAR_DP:.1f}%: over {int(K.CIFAR_GAP)} points lower", 1),
    ("ImageNet, ε = 8, public pre-training", K.IMNET_DP, "data", f"{K.IMNET_DP:.1f}%", 1),
    ("Same model, no privacy", K.IMNET_SAME, "neutral",
     f"{K.IMNET_SAME:.1f}%: {K.IMNET_GAP_SAME:.1f} points higher", 1),
    ("Best non-private model", K.IMNET_BEST, "neutral",
     f"{K.IMNET_BEST:.1f}%: {K.IMNET_GAP_BEST:.1f} points higher", 1),
], 1, (5.4, 2.2), 150, "Top-1 accuracy (%)", label_size=11)

bar_steps("kws_bytes", [
    ("One hour of raw audio", K.AUDIO_MB_H, "danger", f"{K.AUDIO_MB_H:.1f} MB", 1),
    ("One model update", K.UPDATE_MB, "data", f"{K.UPDATE_MB:.1f} MB, {K.AUDIO_RATIO:.0f}× smaller", 1),
], 1, (5.8, 1.1), 5000, "Bytes leaving the home (MB, log scale)", log=True, xmin=0.1)

bar_steps("audit", [
    ("Credit: 25 months, 256 B each", K.LOG_GB, "memory", f"{K.LOG_GB:.0f} GB", 1),
    ("DLRM: 1 year, 256 B each", K.DLRM_MANIFEST_TB * 1e3, "memory", f"{K.DLRM_MANIFEST_TB:.1f} TB", 2),
    ("DLRM: 1 year, full input", K.DLRM_INPUT_TB * 1e3, "danger",
     f"{K.DLRM_INPUT_TB:.0f} TB, {K.PAYLOAD_RATIO:.0f}×", 2),
], 2, (5.8, 1.8), 3e8, "Audit log size (GB, log scale)", log=True, xmin=10,
    note="256 B per record is an assumption; DLRM input as in MLPerf DLRM (constructed)")


# =============================================================================
# Diagrams (SVG)
# =============================================================================

FONT = 'font-family="Helvetica, Arial, sans-serif"'


class Svg:
    """A tiny SVG builder. Every element carries the first step it appears in;
    write(stem, n) emits stem_step1..n, hidden elements kept but transparent."""

    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, first, s, last=99):
        self.items.append((first, last, s))

    def rect(self, x, y, w, h, fill, stroke, first=1, sw=2, rx=8, dash=None, last=99):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(first, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>', last)

    def text(self, x, y, s, size=22, color=C["text"], weight="normal", anchor="middle",
             first=1, last=99):
        lines = s.split("\n")
        y0 = y - (len(lines) - 1) * size * 0.6
        spans = "".join(
            f'<tspan x="{x}" y="{y0 + i * size * 1.2:.1f}">{ln}</tspan>' for i, ln in enumerate(lines))
        self.add(first, f'<text {FONT} font-size="{size}" fill="{color}" font-weight="{weight}" '
                        f'text-anchor="{anchor}" dominant-baseline="middle">{spans}</text>', last)

    def line(self, x1, y1, x2, y2, color, first=1, sw=2.5, arrow=True, dash=None, last=99):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.add(first, f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                        f'stroke-width="{sw}"{d}{m}/>', last)

    def circle(self, x, y, r, fill, stroke, first=1, sw=2, last=99):
        self.add(first, f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" '
                        f'stroke-width="{sw}"/>', last)

    def write(self, stem, steps):
        colors = sorted({c for c in C.values()})
        defs = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{c}"/></marker>' for c in colors)
        for k in range(1, steps + 1):
            body = []
            for first, last, s in self.items:
                vis = first <= k <= last
                body.append(s if vis else s.replace("<", '<g opacity="0"><', 1) + "</g>")
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                   f'width="{self.w}" height="{self.h}"><defs>{defs}</defs>'
                   f'<rect width="100%" height="100%" fill="white"/>' + "".join(body) + "</svg>")
            name = f"{stem}_step{k}.svg" if steps > 1 else f"{stem}.svg"
            (out / name).write_text(svg)


# DLRM derivation strip (book scenario)
g = Svg(900, 185)
boxes = [("10M users × 20 items", f"{K.INF_DAY / 1e6:.0f}M inferences\na day", 1),
         ("× 10 ms each", f"{K.GPU_H_DAY:.0f} GPU-hours\na day", 2),
         ("÷ 24 hours", f"{K.GPUS:.1f} GPUs\nalways on", 3),
         ("× $2.50 an hour", f"${K.SERVE_YEAR / 1e3:.0f}K\na year", 4)]
for i, (op, res, first) in enumerate(boxes):
    x = 10 + i * 225
    g.text(x + 100, 26, op, 22, C["neutral"], "bold", first=first)
    g.rect(x, 52, 200, 125, C["compute_l"], C["compute"], first=first, sw=2.5)
    g.text(x + 100, 114, res, 23, C["text"], "bold", first=first)
    if i:
        g.line(x - 22, 114, x - 4, 114, C["neutral"], first=first)
g.write("dlrm_strip", 4)

# Lineage graph for an erasure request (KWS), rebuilt in four steps
g = Svg(900, 380)
W, H = 150, 80
nodes = [("Recording\nand consent", 20, 140, 1), ("Spectrogram\ncache", 200, 40, 2),
         ("Feature\nembeddings", 200, 240, 2), ("Training\nsnapshot", 380, 140, 3),
         ("Checkpoint\nv12", 560, 140, 3), ("Devices\nrunning v12", 740, 140, 4)]
edges = [(0, 1, 2), (0, 2, 2), (1, 3, 3), (2, 3, 3), (3, 4, 3), (4, 5, 4)]
for a, b, first in edges:
    xa, ya = nodes[a][1] + W, nodes[a][2] + H / 2
    xb, yb = nodes[b][1], nodes[b][2] + H / 2
    g.line(xa + 2, ya, xb - 4, yb, C["neutral"], sw=2, first=first, last=3)
    g.line(xa + 2, ya, xb - 4, yb, C["danger"], sw=4, first=4)
for lab, x, y, first in nodes:
    g.rect(x, y, W, H, C["memory_l"], C["memory"], first=first, last=3)
    g.rect(x, y, W, H, C["danger_l"], C["danger"], first=4, sw=3)
    g.text(x + W / 2, y + H / 2, lab, 20, C["text"], "bold", first=first)
g.text(635, 100, "signed manifest:\ndata hashes, seeds", 17, C["memory"], "bold", first=3)
g.text(95, 280, "Erasure\nrequest", 21, C["danger"], "bold", first=4)
g.line(95, 252, 95, 226, C["danger"], first=4)
g.text(635, 275, "Weights: retrain,\nunlearn, or other\ncontrol", 19, C["danger"], "bold", first=4)
g.text(815, 270, "Recall\nthe model", 19, C["danger"], "bold", first=4)
g.text(450, 360, "Answer within one month, extendable by two (GDPR Art. 12(3))", 19,
       C["danger"], "bold", first=4)
g.write("lineage", 4)

# DP-SGD mechanism
g = Svg(900, 330)
g.rect(10, 40, 200, 150, C["memory_l"], C["memory"])
g.text(110, 66, "Per-example\ngradients", 22, C["text"], "bold")
for j, ln in enumerate((40, 120, 70, 150)):
    yy = 115 + j * 18
    g.line(30, yy, 30 + ln, yy, C["memory"], sw=3)
g.rect(235, 40, 200, 150, C["compute_l"], C["compute"], first=2)
g.text(335, 66, "Clip each\nto norm C", 22, C["text"], "bold", first=2)
for j, ln in enumerate((40, 80, 70, 80)):
    yy = 115 + j * 18
    g.line(255, yy, 255 + ln, yy, C["compute"], sw=3, first=2)
g.add(2, f'<line x1="{255 + 82}" y1="100" x2="{255 + 82}" y2="180" stroke="{C["danger"]}" '
         f'stroke-width="2" stroke-dasharray="5 4"/>')
g.text(372, 182, "C", 24, C["danger"], "bold", first=2)
g.line(212, 115, 231, 115, C["neutral"], first=2)
g.rect(460, 40, 200, 150, C["network_l"], C["network"], first=3)
g.text(560, 85, "Sum, then add\nGaussian noise", 22, C["text"], "bold", first=3)
g.text(560, 150, "N(0, σ²C²)", 24, C["network"], "bold", first=3)
g.line(437, 115, 456, 115, C["neutral"], first=3)
g.rect(685, 40, 200, 150, C["data_l"], C["data"], first=3)
g.text(785, 115, "Update\nthe weights", 23, C["text"], "bold", first=3)
g.line(662, 115, 681, 115, C["neutral"], first=3)
g.rect(300, 235, 585, 75, "#F4F4F4", C["danger"], first=4, sw=2.5)
g.text(592, 272, "Accountant: each step spends budget ε",
       23, C["danger"], "bold", first=4)
g.line(785, 192, 785, 231, C["danger"], first=4)
g.text(12, 262, "Direct method keeps\none per example", 21, C["neutral"], anchor="start", first=1)
g.write("dpsgd_mech", 4)

# KWS: raw audio stays on the device
g = Svg(900, 190)
g.rect(5, 10, 170, 110, C["neutral_l"], C["neutral"])
g.text(90, 65, "Microphone", 25, C["text"], "bold")
g.rect(210, 10, 240, 110, C["memory_l"], C["memory"])
g.text(330, 47, "SRAM ring", 25, C["text"], "bold")
g.text(330, 87, "overwritten", 24, C["memory"], "bold")
g.rect(485, 10, 180, 110, C["compute_l"], C["compute"])
g.text(575, 65, "Wake word?", 25, C["text"], "bold")
g.line(177, 65, 206, 65, C["neutral"])
g.line(452, 65, 481, 65, C["neutral"])
g.line(667, 65, 705, 65, C["data"])
g.text(795, 47, "only then", 23, C["data"], "bold")
g.text(795, 85, "audio may leave", 23, C["data"], "bold")
g.text(450, 165, "Training clients send model updates, not audio", 25, C["text"])
g.write("kws_device", 1)

print("figures written")
