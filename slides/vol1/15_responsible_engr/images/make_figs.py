#!/usr/bin/env python3
"""Figures for Lecture 15 (Responsible Engineering).

Plots use the shared slideplot helper; diagrams are written as SVG with a
small builder below, one file per build step, every step on the same canvas
so nothing moves. Every value is sourced:

  - Care-management algorithm: Obermeyer, Powers, Vogeli, Mullainathan,
    Science 366:447 (2019), read from the full text. 6,079 Black and 43,539
    White patients, 2013 to 2015, one academic hospital; auto-enrollment at
    the 97th percentile of risk; Black share of auto-enrolled 17.7 percent,
    46.5 percent after removing the gap in health at a given score (Fig. 1B);
    26.3 percent more chronic illnesses at the 97th percentile (4.8 against
    3.8); $1,801 less spent per year at the same number of chronic illnesses;
    Table 2, Black share of the top 3 percent by training label: total cost
    0.141, avoidable cost 0.210, active chronic conditions 0.267.
  - Gender Shades (Buolamwini and Gebru 2018, Table 4), Face++ error:
    all 10.0, darker female 34.5, darker male 0.7, lighter female 9.8,
    lighter male 0.8 percent (chapter table 3 for the four groups).
  - Raji and Buolamwini 2019, Tables 1 and 2: darker-female error August 2018,
    Face++ 4.1, Microsoft 1.52, IBM 16.97, Amazon 31.37, Kairos 22.50; May 2017
    values from Gender Shades Table 4: Face++ 34.5, Microsoft 20.8, IBM 34.7.
  - Representation budget: Napkin Math 15.1, margin 1.96 sqrt(0.25/n).
  - COMPAS: ProPublica, "How We Analyzed the COMPAS Recidivism Algorithm"
    (2016): false positive rate 44.85 and 23.45, false negative rate 27.99 and
    47.72, PPV 0.63 and 0.59 (Black, White); base rates derived from its
    contingency tables, 1,901/3,696 and 966/2,454.
  - Credit service (constructed, chapter Sec. 15.3.2.1, tables 5 to 7) and the
    price of fairness (Napkin Math 15.2): $150 baseline, $75 after.
  - Feedback (Napkin Math 15.3, illustrative): gap 3.0 points, factor 1.10 and
    0.90 per quarter, gate 5 points.
  - Patterson et al. 2021, Table 4: net tCO2e GShard-600B 4.3, T5 46.7,
    Switch 59.1, Meena 96.4, GPT-3 552.1; net kg CO2e/kWh 0.177, 0.545, 0.330,
    0.415, 0.429 (GShard, T5, Switch, Meena, GPT-3).
  - DLRM three-year TCO (book scenario, chapter table 10): training $38K,
    inference $1.52M, operations $510K; modeled accelerator carbon 1.5 t and
    97.3 t.
  - Patterson et al. 2022: inference about 3/5 of Google's ML energy
    (2019 to 2021). Wu et al. 2022, Fig. 3 and Sec. III: Meta AI power capacity
    10:20:70 (experimentation, training, inference); embodied about 30 percent
    of the carbon of large-scale ML tasks (location-based).
  - DP-SGD per-sample gradients (chapter Sec. 15.5.2): KWS 204.8 MB per batch
    of 256, GPT-2 1.5 TB, H100 memory 80 GB.

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


# =============================================================================
# Plots
# =============================================================================

def bar_steps(stem, rows, steps, size, xmax, xlabel, note=None, vlines=None,
              log=False, xmin=None):
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
                    ha="right", va="center", fontsize=12, color=C["text"],
                    fontweight="bold", alpha=1.0 if on else 0.0)
            x = v * 1.12 if log else v + xmax * 0.012
            ax.text(x, y, txt, va="center", fontsize=12, color=C[ck],
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
            fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=9,
                     color=C["neutral"])
        sp.save(fig, out / f"{stem}_step{k}.svg")


# Hook: share of Black patients among the auto-enrolled (Obermeyer Fig. 1B)
fig_rows = [("Ranked by predicted cost", 17.7, "neutral", "17.7%", 1),
            ("Ranked by health", 46.5, "danger", "46.5%   2.6× as many", 3)]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    for i, (label, v, ck, txt, first) in enumerate(fig_rows):
        y = 1 - i
        on = k >= first
        ax.barh(y, v, color=C[ck], height=0.6, alpha=1.0 if on else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(v + 0.8, y, txt, va="center", fontsize=12, fontweight="bold",
                color=C[ck], alpha=1.0 if on else 0.0)
    # the hidden bar's slot: a dashed outline and ??? until the reveal
    show_q = k == 2
    ax.barh(0, 46.5, height=0.6, fill=False, ls="--", lw=1.2,
            edgecolor=C["neutral"], alpha=1.0 if show_q else 0.0)
    ax.text(23, 0, "???", ha="center", va="center", fontsize=16,
            fontweight="bold", color=C["danger"], alpha=1.0 if show_q else 0.0)
    ax.set_yticks([])
    ax.set_xlim(0, 75)
    ax.set_ylim(-0.6, 1.6)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Black patients among those auto-enrolled (%)")
    sp.save(fig, out / f"hook_share_step{k}.svg")

# Segment 1 measured: the training label decides who is selected (Table 2)
bar_steps("label_choice", [
    ("Label: total cost", 14.1, "neutral", "14.1%", 1),
    ("Label: avoidable cost", 21.0, "compute", "21.0%", 2),
    ("Label: active chronic illnesses", 26.7, "data", "26.7%   1.9× the cost label", 3),
], 3, (5.8, 2.2), 40, "Black patients among the top 3% of predicted risk (%)")

# Proxy and goal (illustrative, after Systems Perspective 15.2)
t = np.linspace(0, 1, 200)
knee = 0.5
proxy = 1.0 + 0.2 * t
goal = np.where(t <= knee, 1.0 + 0.2 * t, 1.1 - 0.3 * (t - knee))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    m = t <= knee if k == 1 else np.ones_like(t, dtype=bool)
    ax.plot(t[m], proxy[m], color=C["memory"], lw=3.5)
    ax.plot(t[m], goal[m], color=C["data"], lw=3.5, ls=(0, (6, 2)) if k == 1 else "-")
    ax.plot([knee], [1.1], "o", ms=11, color=C["danger"], zorder=5)
    ax.annotate("Knee", (knee, 1.1), xytext=(-8, 12), textcoords="offset points",
                ha="right", fontsize=12, fontweight="bold", color=C["danger"])
    ax.text(0.03, 1.22, "Clicks and satisfaction\nrise together", fontsize=11,
            color=C["text"], va="bottom")
    if k == 2:
        ax.fill_between(t[t >= knee], goal[t >= knee], proxy[t >= knee],
                        color=C["danger_l"], alpha=0.8)
        ax.text(1.02, 1.20, "Clicks  1.20", color=C["memory"], fontsize=12,
                fontweight="bold", va="center", transform=ax.transData)
        ax.text(1.02, 0.95, "Satisfaction  0.95", color=C["data"], fontsize=12,
                fontweight="bold", va="center")
        ax.text(0.80, 1.09, "Gap 0.25", color=C["danger"], fontsize=12,
                fontweight="bold", ha="center")
    else:
        ax.text(1.02, 1.20, "Clicks  1.20", color=C["memory"], fontsize=12,
                fontweight="bold", va="center", alpha=0.0)
        ax.text(1.02, 0.95, "Satisfaction  0.95", color=C["data"], fontsize=12,
                fontweight="bold", va="center", alpha=0.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.85, 1.32)
    ax.set_xticks([])
    ax.set_yticks([1.0])
    ax.set_yticklabels(["1.00"])
    ax.set_xlabel("Optimization effort on clicks")
    ax.set_ylabel("Normalized value")
    ax.text(0.0, -0.22, "Illustrative (Systems Perspective 15.2)", transform=ax.transAxes,
            fontsize=9, color=C["neutral"])
    sp.save(fig, out / f"proxy_goal_step{k}.svg")

# Gender Shades: one aggregate number, then the groups (Face++)
bar_steps("gender_shades", [
    ("All faces (reported)", 10.0, "neutral", "10.0%", 1),
    ("Lighter men", 0.8, "memory", "0.8%", 2),
    ("Darker men", 0.7, "memory", "0.7%", 2),
    ("Lighter women", 9.8, "memory", "9.8%", 2),
    ("Darker women", 34.5, "danger", "34.5%   43× lighter men", 2),
], 2, (5.8, 2.5), 52, "Face++ gender-classification error (%)")

# Margin of error against the number of examples in one group
n = np.logspace(2, 5, 300)
marg = 1.96 * np.sqrt(0.25 / n) * 100
fig, ax = plt.subplots(figsize=(3.6, 2.0))
ax.plot(n, marg, color=C["memory"], lw=3)
ax.axhline(1.0, color=C["danger"], lw=1.5, ls="--")
ax.text(125, 1.35, "Target: ±1 point", color=C["danger"], fontsize=10, fontweight="bold")
ax.plot([9604], [1.0], "o", ms=10, color=C["danger"], zorder=5)
ax.annotate("9,604", (9604, 1.0), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=12, fontweight="bold", color=C["danger"])
ax.set_xscale("log")
ax.set_xlim(100, 1e5)
ax.set_ylim(0, 10)
ax.set_xticks([100, 1000, 10000, 100000])
ax.set_xticklabels(["100", "1,000", "10,000", "100,000"])
ax.set_xlabel("Examples from the group (log scale)")
ax.set_ylabel("95% margin (points)")
sp.save(fig, out / "margin_curve.svg")

# Raji and Buolamwini 2019: darker-female error before and after the audit
vendors = [("Face++", 34.5, 4.1), ("IBM", 34.7, 16.97), ("Microsoft", 20.8, 1.52),
           ("Amazon", None, 31.37), ("Kairos", None, 22.50)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    nv = len(vendors)
    for i, (name, before, after) in enumerate(vendors):
        y = nv - 1 - i
        audited = before is not None
        if audited:
            ax.barh(y + 0.19, before, height=0.36, color=C["neutral_l"])
            ax.text(before + 0.6, y + 0.19, f"{before:g}%  May 2017", va="center",
                    fontsize=10, color=C["neutral"], fontweight="bold")
        on = k >= 2
        ck = "danger" if audited else "neutral"
        ax.barh(y - 0.19 if audited else y, after, height=0.36,
                color=C[ck], alpha=1.0 if on else 0.0)
        ax.text(after + 0.6, y - 0.19 if audited else y,
                f"{after:g}%  Aug 2018" + ("" if audited else " (not in first audit)"),
                va="center", fontsize=10, color=C[ck], fontweight="bold",
                alpha=1.0 if on else 0.0)
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
    ax.axhline(1.5, color=C["grid"], lw=1)
    ax.set_yticks([])
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, nv - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Error on darker-skinned women (%)")
    sp.save(fig, out / f"raji_step{k}.svg")

# COMPAS (ProPublica 2016): error rates, then precision and base rates
metrics = [("False positive rate", 44.85, 23.45, 1),
           ("False negative rate", 27.99, 47.72, 1),
           ("Precision (PPV)", 63.0, 59.0, 2),
           ("Base rate (derived)", 51.4, 39.4, 2)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    nm = len(metrics)
    for i, (name, b, w, first) in enumerate(metrics):
        y = nm - 1 - i
        on = k >= first
        a = 1.0 if on else 0.0
        ax.barh(y + 0.19, b, height=0.36, color=C["compute"], alpha=a)
        ax.barh(y - 0.19, w, height=0.36, color=C["memory"], alpha=a)
        ax.text(b + 0.8, y + 0.19, f"{b:.0f}%", va="center", fontsize=11,
                color=C["compute"], fontweight="bold", alpha=a)
        ax.text(w + 0.8, y - 0.19, f"{w:.0f}%", va="center", fontsize=11,
                color=C["memory"], fontweight="bold", alpha=a)
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"], alpha=a)
    ax.text(68, 3.2, "Black defendants", color=C["compute"], fontsize=11, fontweight="bold")
    ax.text(68, 2.75, "White defendants", color=C["memory"], fontsize=11, fontweight="bold")
    ax.axhline(1.5, color=C["grid"], lw=1, alpha=1.0 if k >= 2 else 0.0)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, nm - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Percent")
    sp.save(fig, out / f"compas_step{k}.svg")

# Price of fairness, Group B utility per applicant (constructed scenario)
bar_steps("price", [
    ("Before (TPR 60%)", 150, "memory", "$150", 1),
    ("TPR raised to 90%", 75, "danger", "$75   half the utility", 1),
], 1, (4.4, 1.3), 260, "Group B utility per applicant (US$)")

# Feedback: projected TPR gap by quarter (Napkin Math 15.3, illustrative)
q = np.arange(0, 9)
grow = 3.0 * 1.10 ** q
decay = 3.0 * 0.90 ** q
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.axhline(5.0, color=C["danger"], lw=1.6, ls="--")
    ax.text(0.1, 4.85, "Release gate: 5 points", color=C["danger"], fontsize=11,
            fontweight="bold", va="top")
    ax.plot(q, grow, "-o", color=C["compute"], lw=3, ms=6)
    ax.plot([6], [3.0 * 1.1 ** 6], "o", ms=12, color=C["danger"], zorder=5)
    ax.annotate("Quarter 6: 5.3", (6, 3.0 * 1.1 ** 6), xytext=(-10, 10),
                textcoords="offset points", ha="right", fontsize=11,
                fontweight="bold", color=C["danger"])
    ax.text(8.15, grow[-1], "factor 1.10", color=C["compute"], fontsize=11,
            fontweight="bold", va="center")
    a = 1.0 if k == 2 else 0.0
    ax.plot(q, decay, "-o", color=C["data"], lw=3, ms=6, alpha=a)
    ax.text(8.15, decay[-1], "factor 0.90\n(randomized\napprovals)", color=C["data"],
            fontsize=11, fontweight="bold", va="center", alpha=a)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7.5)
    ax.set_xlabel("Quarterly retraining cycle")
    ax.set_ylabel("TPR gap (points)")
    ax.text(0.0, -0.33, "Illustrative: factor fitted from two audits (Napkin Math 15.3)",
            transform=ax.transAxes, fontsize=9, color=C["neutral"])
    sp.save(fig, out / f"feedback_step{k}.svg")

# Patterson et al. 2021, Table 4: measured training emissions
pat = [("GShard-600B", 4.3, "0.177"), ("T5", 46.7, "0.545"),
       ("Switch", 59.1, "0.330"), ("Meena", 96.4, "0.415"), ("GPT-3", 552.1, "0.429")]
# net kg CO2e/kWh per Patterson Table 4: GShard 0.177 (North Carolina),
# T5 0.545 (Taiwan), Switch 0.330, Meena 0.415 (Georgia), GPT-3 0.429 (US average)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    for i, (name, v, ci) in enumerate(pat):
        y = len(pat) - 1 - i
        ck = "danger" if name == "GPT-3" else "neutral"
        ax.barh(y, v, height=0.62, color=C[ck])
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(v * 1.12, y, f"{v:g} t", va="center", fontsize=12,
                fontweight="bold", color=C[ck])
        ax.text(5000, y, f"{ci} kg/kWh", va="center", ha="left", fontsize=11,
                color=C["memory"], fontweight="bold", alpha=1.0 if k == 2 else 0.0)
    ax.text(5000, len(pat) - 0.35, "Grid intensity", fontsize=10, color=C["memory"],
            va="bottom", alpha=1.0 if k == 2 else 0.0)
    ax.set_xscale("log")
    ax.set_xlim(2, 25000)
    ax.set_ylim(-0.6, len(pat) - 0.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Net training emissions (tonnes CO₂e, log scale)")
    sp.save(fig, out / f"patterson_step{k}.svg")

# DLRM three-year TCO and modeled accelerator carbon (book scenario, table 10)
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

# Measured fleet shares (Patterson et al. 2022; Wu et al. 2022)
fleet = [
    ("Google ML energy,\n2019 to 2021", [("Training ~2/5", 40, "neutral_l"),
                                        ("Inference ~3/5", 60, "compute")], 1),
    ("Meta AI power capacity:\nexp., training, inference", [("10", 10, "neutral_l"), ("Training 20", 20, "neutral"),
                                ("Inference 70", 70, "compute")], 2),
    ("Meta large ML tasks,\ncarbon", [("Embodied ~30", 30, "network"),
                                     ("Operational ~70", 70, "neutral_l")], 3),
]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    for i, (label, segs, first) in enumerate(fleet):
        y = len(fleet) - 1 - i
        a = 1.0 if k >= first else 0.0
        left = 0
        for name, v, ck in segs:
            ax.barh(y, v, left=left, height=0.62, color=C[ck], edgecolor="white", alpha=a)
            dark = ck in ("compute", "network", "neutral")
            ax.text(left + v / 2, y, name, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if dark else C["text"], alpha=a)
            left += v
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=11, fontweight="bold", color=C["text"], alpha=a)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(fleet) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share (%)")
    sp.save(fig, out / f"fleet_shares_step{k}.svg")

# DP-SGD per-sample gradient memory against one accelerator (chapter Sec. 15.5.2)
bar_steps("dpsgd", [
    ("KWS, 200K parameters", 204.8, "memory", "204.8 MB", 1),
    ("GPT-2, 1.5B parameters", 1.5e6, "danger", "1.5 TB", 1),
], 1, (5.8, 1.6), 3e7, "Per-sample gradients for a batch of 256 (MB, log scale)",
    vlines=[(8e4, "One H100: 80 GB", "neutral", 1)], log=True, xmin=30)


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


# Hook mechanism: need, care, cost; the model sees only cost
g = Svg(900, 330)
for i, (lab, x) in enumerate((("Health need", 40), ("Care received", 330), ("Cost", 620))):
    g.rect(x, 40, 240, 80, C["memory_l"] if i < 2 else C["compute_l"],
           C["memory"] if i < 2 else C["compute"])
    g.text(x + 120, 80, lab, 26, weight="bold")
for x in (280, 570):
    g.line(x + 2, 80, x + 46, 80, C["neutral"])
g.text(740, 160, "The model's label", 22, C["compute"], "bold")
g.line(740, 145, 740, 126, C["compute"])
g.text(160, 160, "What the program\nshould target", 20, C["memory"], "bold")
# step 2: the access gap
g.rect(330, 205, 530, 95, C["danger_l"], C["danger"], first=2)
g.text(595, 235, "Unequal access: at the same illness,", 22, C["danger"], "bold", first=2)
g.text(595, 270, "$1,801 less is spent per Black patient per year", 22, C["danger"],
       "bold", first=2)
g.line(450, 200, 450, 126, C["danger"], first=2)
g.write("care_chain", 2)

# Verification against validation
g = Svg(900, 330)
g.text(300, 30, "Verification: built to the spec?", 24, C["text"], "bold")
for i, (a, b) in enumerate((("Latency", "within SLA"), ("Availability", "99.9%"),
                            ("Accuracy", "87% overall"))):
    x = 20 + i * 195
    g.rect(x, 70, 180, 140, C["data_l"], C["data"])
    g.text(x + 90, 115, a, 22, C["text"], "bold")
    g.text(x + 90, 160, b, 24, C["data"], "bold")
    g.circle(x + 155, 92, 9, C["data"], C["data"])
g.text(750, 30, "Validation: right spec?", 24, C["text"], "bold", first=2)
g.rect(620, 70, 260, 140, C["danger_l"], C["danger"], first=2, sw=3)
g.text(750, 115, "Fair hiring?", 22, C["text"], "bold", first=2)
g.text(750, 160, "Downgrades resumes\nwith “women's”", 20, C["danger"], "bold",
       first=2)
g.add(1, f'<line x1="600" y1="55" x2="600" y2="300" stroke="{C["grid"]}" stroke-width="2"/>')
g.text(300, 270, "Every check the team wrote is green.", 22, C["neutral"])
g.text(750, 270, "The check nobody wrote", 22, C["danger"], "bold", first=2)
g.write("verify_validate", 2)

# Fan-out: one upstream change reaches every decision below it
g = Svg(900, 400)
g.rect(10, 145, 260, 125, C["danger_l"], C["danger"], sw=3)
g.text(140, 182, "Form change:", 24, C["danger"], "bold")
g.text(140, 228, "self-employed income\nrecorded differently", 21, C["text"])
ys = [60, 125, 190, 255, 320]
for j, y in enumerate(ys):
    g.line(272, 207, 528, y, C["danger"], sw=2)
    for col in range(4):
        g.circle(560 + col * 75, y, 24, C["danger_l"], C["danger"])
g.text(670, 375, "Approved last week, declined now", 23,
       C["danger"], "bold")
# step 2: the green dashboard that saw nothing
g.rect(10, 5, 260, 125, C["data_l"], C["data"], first=2)
g.text(140, 30, "Dashboard", 22, C["data"], "bold", first=2)
g.text(140, 82, "latency OK\nconfidence high\nuptime OK", 20, C["text"], first=2)
g.write("fanout", 2)

# Release gate: point estimates with confidence intervals, three outcomes
g = Svg(900, 400)
x0, x1 = 260, 660          # axis: 0 to 10 points
def gx(v):
    return x0 + (x1 - x0) * v / 10.0
g.add(1, f'<line x1="{x0}" y1="330" x2="{x1}" y2="330" stroke="{C["neutral"]}" stroke-width="2"/>')
for v in (0, 5, 10):
    g.text(gx(v), 355, f"{v}", 22, C["neutral"])
g.text((x0 + x1) / 2, 388, "TPR gap between groups (points)", 22, C["neutral"])
g.add(1, f'<line x1="{gx(5)}" y1="30" x2="{gx(5)}" y2="330" stroke="{C["danger"]}" '
         f'stroke-width="2.5" stroke-dasharray="8 5"/>')
g.text(gx(5), 18, "Gate: 5 points", 22, C["danger"], "bold")
slices = [("Slice A", 1.8, 0.8, "PASS", "data", 1, 70),
          ("Slice B", 8.0, 1.0, "FAIL", "danger", 1, 145),
          ("Slice C: few labels", 3.5, 3.5, "INSUFFICIENT", "compute", 2, 220),
          ("Slice D", 2.6, 1.0, "PASS", "data", 3, 295)]
for name, v, h, verdict, ck, first, y in slices:
    g.text(10, y, name, 22, C["text"], "bold", "start", first=first)
    g.add(first, f'<line x1="{gx(v - h)}" y1="{y}" x2="{gx(v + h)}" y2="{y}" '
                 f'stroke="{C[ck]}" stroke-width="4"/>')
    for e in (v - h, v + h):
        g.add(first, f'<line x1="{gx(e)}" y1="{y - 10}" x2="{gx(e)}" y2="{y + 10}" '
                     f'stroke="{C[ck]}" stroke-width="3"/>')
    g.circle(gx(v), y, 9, C[ck], C[ck], first=first)
    g.text(685, y, verdict, 22, C[ck], "bold", "start", first=first)
g.write("gate", 3)

# Credit service confusion matrices (constructed; chapter tables 5 and 6)
g = Svg(900, 420)
mats = [("Group A: 10,000", 175, (4500, 500, 1000, 4000)),
        ("Group B: 2,000", 615, (600, 400, 200, 800))]
cw, ch = 135, 62
for title, gx0, (tp, fn, fp, tn) in mats:
    g.text(gx0 + cw, 22, title, 24, C["text"], "bold")
    g.text(gx0 + cw / 2, 55, "Approved", 20, C["neutral"], "bold")
    g.text(gx0 + cw * 1.5, 55, "Rejected", 20, C["neutral"], "bold")
    for r, (lab, vals) in enumerate((("Would repay", (tp, fn)), ("Would default", (fp, tn)))):
        y = 72 + r * ch
        g.text(gx0 - 10, y + ch / 2, lab, 20, C["text"], "bold", "end")
        for c, v in enumerate(vals):
            g.rect(gx0 + c * cw, y, cw - 6, ch - 6, "#F4F4F4", C["neutral_l"], rx=4)
            g.text(gx0 + c * cw + (cw - 6) / 2, y + (ch - 6) / 2, f"{v:,}", 25, C["text"], "bold")
    # highlights: step 2 approved column, step 3 repay row, step 4 default row
    g.rect(gx0 - 4, 68, cw + 2, 2 * ch + 2, "none", C["compute"], first=2, last=2, sw=4)
    g.rect(gx0 - 4, 68, 2 * cw + 2, ch + 2, "none", C["memory"], first=3, last=3, sw=4)
    g.rect(gx0 - 4, 68 + ch, 2 * cw + 2, ch + 2, "none", C["network"], first=4, last=4, sw=4)
rows = [("Approval rate (demographic parity)", "55%", "40%", "15 pts", "compute", 2, 250),
        ("Of those who would repay, share approved (TPR)", "90%", "60%", "30 pts", "memory", 3, 300),
        ("Of those who would default, share approved (FPR)", "20%", "20%", "0 pts", "network", 4, 350)]
g.text(650, 222, "A", 20, C["neutral"], "bold")
g.text(740, 222, "B", 20, C["neutral"], "bold")
g.text(845, 222, "Gap", 20, C["neutral"], "bold")
for lab, a, b, gap, ck, first, y in rows:
    g.text(10, y, lab, 21, C[ck], "bold", "start", first=first)
    g.text(650, y, a, 24, C["text"], "bold", first=first)
    g.text(740, y, b, 24, C["text"], "bold", first=first)
    g.text(845, y, gap, 24, C["danger"] if gap != "0 pts" else C["data"], "bold", first=first)
g.text(450, 400, "Constructed example (chapter credit service). Both groups repay at 50%.",
       18, C["neutral"])
g.write("matrices", 4)

# The censored retraining loop
g = Svg(520, 420)
g.rect(170, 20, 180, 70, C["compute_l"], C["compute"])
g.text(260, 55, "Model decides", 21, C["text"], "bold")
g.rect(20, 165, 200, 70, C["data_l"], C["data"])
g.text(120, 200, "Approved", 21, C["text"], "bold")
g.rect(300, 165, 200, 70, "#F4F4F4", C["neutral"], dash="6 4")
g.text(400, 200, "Denied", 21, C["neutral"], "bold")
g.rect(20, 310, 200, 80, C["data_l"], C["data"])
g.text(120, 350, "Repayment\nobserved", 20, C["text"], "bold")
g.text(400, 340, "No label:\nnever observed", 20, C["danger"], "bold")
g.line(230, 92, 140, 160, C["neutral"])
g.line(290, 92, 380, 160, C["neutral"])
g.line(120, 237, 120, 305, C["data"])
g.line(400, 237, 400, 305, C["danger"], dash="6 5")
g.line(222, 340, 262, 340, C["data"], arrow=False)
g.line(262, 340, 262, 95, C["data"])
g.text(270, 270, "retrain", 18, C["data"], "bold", "start")
g.write("censored_loop", 1)

# Lineage: an erasure request follows the graph to every derived artifact (KWS)
g = Svg(900, 360)
nodes = [("Recording\nand consent", 20, 140), ("Spectrogram\ncache", 200, 40),
         ("Feature\nembeddings", 200, 240), ("Training\nsnapshot", 380, 140),
         ("Checkpoint\nv12", 560, 140), ("Devices\nrunning v12", 740, 140)]
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
W, H = 150, 80
for a, b in edges:
    xa, ya = nodes[a][1] + W, nodes[a][2] + H / 2
    xb, yb = nodes[b][1], nodes[b][2] + H / 2
    g.line(xa + 2, ya, xb - 4, yb, C["neutral"], sw=2)
    g.line(xa + 2, ya, xb - 4, yb, C["danger"], sw=4, first=2)
for i, (lab, x, y) in enumerate(nodes):
    g.rect(x, y, W, H, C["memory_l"], C["memory"])
    g.rect(x, y, W, H, C["danger_l"], C["danger"], first=2, sw=3)
    g.text(x + W / 2, y + H / 2, lab, 20, C["text"], "bold", first=1)
g.text(95, 275, "Erasure\nrequest", 21, C["danger"], "bold", first=2)
g.line(95, 245, 95, 226, C["danger"], first=2)
g.text(635, 292, "Weights: retrain,\nunlearn, or other\ncontrol", 20, C["danger"], "bold", first=3)
g.text(815, 292, "Update the\nfleet", 20, C["danger"], "bold", first=3)
g.text(450, 345, "Illustrative lineage for the KWS pipeline (chapter Sec. 15.5.4)", 17,
       C["neutral"], first=1)
g.write("lineage", 3)

print("figures written")
