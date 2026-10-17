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
  - Differentially private accuracy: De et al. 2022 (arXiv 2204.13650),
    abstract: CIFAR-10 81.4 percent at (8, 1e-5)-DP, previous best 71.7;
    ImageNet 86.7 percent at (8, 8e-7)-DP, 4.3 points below the non-private
    best (so 91.0, derived).
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
bar_steps("hook_nas", [
    ("One car, whole lifetime", 57, "neutral", "57 t", 1),
    ("Search, estimated in 2019", 284, "compute", "284 t = 5 car lifetimes", 1),
    ("Same search, measured in 2021", 3.2, "data", "3.2 t, 88× lower", 3),
], 3, (5.8, 2.0), 30000, "Tonnes of CO₂e (log scale)", log=True, xmin=1)
# step 2: overwrite with the empty slot and ??? for the measured bar
for k in (2,):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    rows = [("One car, whole lifetime", 57, "neutral", "57 t"),
            ("Search, estimated in 2019", 284, "compute", "284 t = 5 car lifetimes"),
            ("Same search, measured in 2021", 3.2, "data", "3.2 t, 88× lower")]
    for i, (label, v, ck, txt) in enumerate(rows):
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

# Gemini median prompt: energy at the full boundary, then carbon
for k in (1, 2):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 2.6),
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 1.25})
    stacked(ax1, 0, [(0.14, "compute", "Accelerators 0.14"),
                     (0.06, "memory", "Host 0.06"),
                     (0.02, "neutral_l", ""), (0.02, "neutral", "")])
    ax1.annotate("Idle machines 0.02", (0.21, 0.32), xytext=(0.212, 0.75),
                 ha="right", va="bottom", fontsize=10, fontweight="bold", color=C["text"],
                 arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=1))
    ax1.annotate("Overhead 0.02 (PUE 1.09)", (0.23, 0.32), xytext=(0.228, 0.75),
                 ha="left", va="bottom", fontsize=10, fontweight="bold", color=C["text"],
                 arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=1))
    ax1.text(0.255, 0, "= 0.24 Wh", va="center", fontsize=13, fontweight="bold",
             color=C["text"])
    ax1.set_xlim(0, 0.36)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_xlabel("Energy per prompt (Wh)")
    a = 1.0 if k >= 2 else 0.0
    stacked(ax2, 0, [(0.023, "network", "Electricity 0.023"),
                     (0.010, "neutral", "")], alpha=a)
    ax2.text(0.0335, 0, " + 0.010 Scope 1 and 3  = 0.03 g", va="center", fontsize=12,
             fontweight="bold", color=C["text"], alpha=a)
    ax2.set_xlim(0, 0.0473 * 0.36 / 0.34)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_alpha(a)
    ax2.tick_params(axis="x", colors=C["text"] if k >= 2 else "white")
    ax2.set_xlabel("CO₂e per prompt (g)", color=C["text"] if k >= 2 else "white")
    sp.save(fig, out / f"gemini_step{k}.svg")

# Four factors on one training job: cumulative gain read from Patterson Fig. 1
gains = [("Baseline:\nP100, US avg.", 1.0, "neutral", "1×\n0.1357 t", 1),
         ("Model:\nEvolved T.", 1.3, "compute", "1.3×", 1),
         ("Processor:\nTPU v2", 7.4, "compute", "7.4×", 2),
         ("PUE:\n1.59 to 1.11", 10.5, "memory", "10.5×", 3),
         ("Grid, net:\n0.429 to 0.080", 56.5, "data", "56.5×\n0.0024 t", 4)]
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    for i, (lab, v, ck, txt, first) in enumerate(gains):
        on = k >= first
        ax.bar(i, v, width=0.62, color=C[ck], alpha=1.0 if on else 0.0)
        ax.text(i, v * 1.12, txt, ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=C[ck], alpha=1.0 if on else 0.0)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 400)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1×", "10×", "100×"])
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels([g[0] for g in gains], fontsize=10)
    ax.set_xlim(-0.6, len(gains) - 0.4)
    ax.set_ylabel("CO₂e reduction,\ncumulative")
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"factors_step{k}.svg")

# Four 175B-scale models (Luccioni et al. 2023, Table 4, CO2e x PUE)
bar_steps("llm4", [
    ("GPT-3", 552, "danger", "552 t: 1,287 MWh at 429 g/kWh", 1),
    ("Gopher", 380, "neutral", "380 t: 1,066 MWh* at 330 g/kWh", 2),
    ("OPT", 76.3, "neutral", "76.3 t*: 324 MWh* at 231 g/kWh*", 2),
    ("BLOOM", 30, "data", "30 t: 433 MWh at 57 g/kWh", 1),
], 2, (5.8, 2.3), 1150, "Training emissions × PUE (t CO₂e)",
    note="* inferred by Luccioni et al., not stated by the model's authors", label_size=12)

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
    ax.text(24.69 / 2, 0.5, "What most papers report (48.9%)", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C["compute"])
    ax.text(51.2, 0, "= 50.5 t", va="center", fontsize=13, fontweight="bold",
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
    parts = [("Training", 0.038, "neutral"), ("Inference", 1.52, "compute"),
             ("Operations", 0.51, "neutral_l")]
    left = 0
    for name, v, ck in parts:
        ax1.barh(0, v, left=left, height=0.6, color=C[ck], edgecolor="white")
        left += v
    ax1.text(0.04, 0.55, "Training $38K (1.9%)", fontsize=11, color=C["text"],
             va="bottom", fontweight="bold")
    ax1.text(0.038 + 1.52 / 2, 0, "Inference $1.52M (73.5%)", fontsize=11,
             color="white", ha="center", va="center", fontweight="bold")
    ax1.text(0.038 + 1.52 + 0.255, 0, "Ops $510K", fontsize=11, ha="center",
             va="center", color=C["text"], fontweight="bold")
    ax1.set_xlim(0, 2.2)
    ax1.set_ylim(-0.4, 1.2)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_xlabel("Three-year cost (US$ millions)")
    a = 1.0 if k == 2 else 0.0
    for y, (name, v, ck) in zip((1, 0), (("12 training cycles", 1.5, "neutral"),
                                         ("3 years of inference", 97.3, "compute"))):
        ax2.barh(y, v, height=0.6, color=C[ck], alpha=a)
        ax2.text(-0.01, y, name, transform=ax2.get_yaxis_transform(), ha="right",
                 va="center", fontsize=11, fontweight="bold", color=C["text"], alpha=a)
        ax2.text(v + 1.5, y, f"{v:g} t", va="center", fontsize=11, fontweight="bold",
                 color=C[ck], alpha=a)
    ax2.text(118, 0, "63×", va="center", fontsize=14, fontweight="bold",
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
    ("Cut serving cost 20%", 304, "compute", "$304K, about 19 t", 1),
    ("Halve all training", 19.2, "neutral", "$19K, 0.8 t", 1),
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
    rows = [("Grid 0.4 kg/kWh", 160, 10, "embodied 6%"),
            ("Grid 0.057 kg/kWh", 23, 10, "embodied 30%")]
    for i, (lab, op, emb, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, op, height=0.6, color=C["neutral_l"], edgecolor="white")
        a = 1.0 if k >= 2 else 0.0
        ax.barh(y, emb, left=op, height=0.6, color=C["network"], edgecolor="white", alpha=a)
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(op + emb + 3, y, f"{op} + {emb} g   {txt}" if k >= 2 else f"{op} g",
                va="center", fontsize=12, fontweight="bold",
                color=C["network"] if k >= 2 else C["text"])
    ax.set_xlim(0, 260)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("g CO₂e per GPU-hour: operational (gray) + embodied (violet)")
    ax.text(0.0, -0.55, "Derived: 400 W GPU; embodied 3 g per GPU-hour + 56 g per server-hour / 8 GPUs "
            "(assumed)", transform=ax.transAxes, fontsize=9, color=C["neutral"])
    sp.save(fig, out / f"embodied_step{k}.svg")


# =============================================================================
# Segment 3: duties have machine costs
# =============================================================================

bar_steps("explain", [
    ("Decisions", 1e6, "memory", "1 million", 1),
    ("Reasons, 1,000-sample attributor", 4.75e8, "compute", "475 million, 475×", 2),
    ("Reasons, exact (2²⁰ per denial)", 4.98e11, "danger", "5 × 10¹¹", 3),
], 3, (5.8, 1.9), 1e15, "Model evaluations per day (log scale)", log=True, xmin=1e5,
    note="Credit service (constructed): 47.5% of 1 million daily decisions are denials")

bar_steps("review", [
    ("Credit service, 6 min each", 1000, "memory", "1,000 reviews", 1),
    ("Cigna, 1.2 s each", 300000, "danger", "300,000 denials", 2),
], 2, (5.8, 1.45), 3e7, "Cases per 100 reviewer-hours (log scale)", log=True, xmin=100,
    note="6 minutes is an assumption; 1.2 s is the measured average (ProPublica, 2023)")


# =============================================================================
# Segment 4: governance as a data system
# =============================================================================

bar_steps("sisa", [
    ("Purchase", 4.63, "data", "4.63×", 1),
    ("SVHN", 2.45, "data", "2.45×", 1),
    ("ImageNet", 1.36, "data", "1.36×", 1),
], 1, (4.2, 1.7), 6.2, "Unlearning speedup over retraining")

# DP-SGD per-sample gradient memory (moved from Lecture 15)
bar_steps("dpsgd", [
    ("KWS, 200K parameters", 204.8, "memory", "204.8 MB", 1),
    ("GPT-2, 1.5B parameters", 1.5e6, "danger", "1.5 TB", 1),
], 1, (5.8, 1.6), 3e7, "Per-sample gradients for a batch of 256 (MB, log scale)",
    vlines=[(8e4, "One H100: 80 GB", "neutral", 1)], log=True, xmin=30)

bar_steps("dp_accuracy", [
    ("CIFAR-10, previous private best", 71.7, "neutral", "71.7%", 1),
    ("CIFAR-10, De et al., ε = 8", 81.4, "data", "81.4%", 1),
    ("ImageNet, De et al., ε = 8", 86.7, "data", "86.7%", 1),
    ("ImageNet, best non-private", 91.0, "neutral", "91.0%, 4.3 points higher", 1),
], 1, (5.8, 1.9), 135, "Top-1 accuracy (%)")

bar_steps("kws_bytes", [
    ("One hour of raw audio", 115.2, "danger", "115.2 MB", 1),
    ("One model update", 0.8, "data", "0.8 MB, 144× smaller", 1),
], 1, (5.8, 1.1), 5000, "Bytes leaving the home (MB, log scale)", log=True, xmin=0.1)

bar_steps("audit", [
    ("Credit: 25 months, 256 B each", 195, "memory", "195 GB", 1),
    ("DLRM: 1 year, 256 B each", 18700, "memory", "18.7 TB", 2),
    ("DLRM: 1 year, full input", 976000, "danger", "976 TB, 52×", 2),
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
g = Svg(900, 330)
boxes = [("10M users × 20 items", "200M inferences\na day", 1),
         ("× 10 ms each", "556 GPU-hours\na day", 2),
         ("÷ 24 hours", "23.1 GPUs\nalways on", 3),
         ("× $2.50 an hour", "$507K\na year", 4)]
for i, (op, res, first) in enumerate(boxes):
    x = 10 + i * 225
    g.text(x + 100, 30, op, 21, C["neutral"], "bold", first=first)
    g.rect(x, 55, 200, 120, C["compute_l"], C["compute"], first=first, sw=2.5)
    g.text(x + 100, 115, res, 25, C["text"], "bold", first=first)
    if i:
        g.line(x - 22, 115, x - 4, 115, C["neutral"], first=first)
g.rect(675, 215, 200, 75, "#F4F4F4", C["neutral"], first=4)
g.text(775, 252, "One training\ncycle: $3,200", 20, C["text"], "bold", first=4)
g.text(330, 255, "Book scenario: 10 ms of dedicated,\nunbatched GPU time per request",
       19, C["neutral"], first=1)
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
g.text(110, 62, "Per-example\ngradients", 20, C["text"], "bold")
for j, ln in enumerate((40, 120, 70, 150)):
    yy = 115 + j * 18
    g.line(30, yy, 30 + ln, yy, C["memory"], sw=3)
g.rect(235, 40, 200, 150, C["compute_l"], C["compute"], first=2)
g.text(335, 62, "Clip each\nto norm C", 20, C["text"], "bold", first=2)
for j, ln in enumerate((40, 80, 70, 80)):
    yy = 115 + j * 18
    g.line(255, yy, 255 + ln, yy, C["compute"], sw=3, first=2)
g.add(2, f'<line x1="{255 + 82}" y1="100" x2="{255 + 82}" y2="180" stroke="{C["danger"]}" '
         f'stroke-width="2" stroke-dasharray="5 4"/>')
g.text(375, 185, "C", 18, C["danger"], "bold", first=2)
g.line(212, 115, 231, 115, C["neutral"], first=2)
g.rect(460, 40, 200, 150, C["network_l"], C["network"], first=3)
g.text(560, 85, "Sum, then add\nGaussian noise", 20, C["text"], "bold", first=3)
g.text(560, 150, "N(0, σ²C²)", 24, C["network"], "bold", first=3)
g.line(437, 115, 456, 115, C["neutral"], first=3)
g.rect(685, 40, 200, 150, C["data_l"], C["data"], first=3)
g.text(785, 115, "Update\nthe weights", 21, C["text"], "bold", first=3)
g.line(662, 115, 681, 115, C["neutral"], first=3)
g.rect(235, 235, 650, 75, "#F4F4F4", C["danger"], first=4, sw=2.5)
g.text(560, 272, "Accountant: every step spends privacy budget ε",
       20, C["danger"], "bold", first=4)
g.line(785, 192, 785, 231, C["danger"], first=4)
g.text(110, 250, "Direct method stores\none gradient per example", 17, C["neutral"], first=1)
g.write("dpsgd_mech", 4)

# KWS: raw audio stays on the device
g = Svg(900, 230)
g.rect(10, 50, 170, 100, C["neutral_l"], C["neutral"])
g.text(95, 100, "Microphone", 22, C["text"], "bold")
g.rect(220, 50, 220, 100, C["memory_l"], C["memory"])
g.text(330, 85, "SRAM ring buffer", 21, C["text"], "bold")
g.text(330, 120, "overwritten", 19, C["memory"])
g.rect(480, 50, 170, 100, C["compute_l"], C["compute"])
g.text(565, 100, "Wake word?", 22, C["text"], "bold")
g.line(182, 100, 216, 100, C["neutral"])
g.line(442, 100, 476, 100, C["neutral"])
g.line(652, 100, 700, 100, C["data"])
g.text(790, 85, "only then", 19, C["data"], "bold")
g.text(790, 118, "audio may leave", 19, C["data"], "bold")
g.text(450, 200, "Training-capable clients send model updates, not audio", 21, C["text"])
g.write("kws_device", 1)

print("figures written")
