#!/usr/bin/env python3
"""Figures for Lecture 15b (Responsible Engineering, Part 2: Cost, Carbon,
and Evidence).

Plots use the shared slideplot helper; diagrams are written as SVG with the
same small builder Lecture 15 uses, one file per build step, every step on the
same canvas so nothing moves. Every value comes from MLSysIM through calcs.py
(K below; formatted labels through deckvals.val and plotted values through
deckvals.num or the scenarios themselves); calcs.py names each source. Layout
coordinates and the illustrative review minimum are the only typed numbers.

Moved from Lecture 15 (same generator code, same sources): tco_step*,
fleet_shares_step*, dpsgd_step1, the lineage graph (rebuilt in four steps).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as K  # noqa: E402  (every value on a figure comes from MLSysIM through calcs.py)
from deckvals import val, num  # noqa: E402

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
    ("One car, whole lifetime", K.CAR_T, "neutral", f"{val('car_t')} t"),
    ("Search, estimated in 2019", K.NAS_EST_T, "compute",
     f"{val('nas_est_t')} t = {val('nas_cars')} car lifetimes"),
    # the measured value and the error factor as Patterson et al. print them (Table 4, App. D)
    ("Same search, measured in 2021", num("nas_meas_t"), "data", f"{val('nas_meas_t')} t, {val('nas_err')}× lower"),
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
    _gp = [r["comprehensive_wh"] for r in K.GEM]   # accelerators, host, idle, overhead
    stacked(ax1, 0, [(_gp[0], "compute", f"Accelerators {_gp[0]:.2f}"),
                     (_gp[1], "memory", f"Host {_gp[1]:.2f}"),
                     (_gp[2], "neutral_l", ""), (_gp[3], "neutral", "")])
    ax1.annotate(f"Idle {_gp[2]:.2f}", (0.21, 0.32), xytext=(0.212, 0.85),
                 ha="right", va="bottom", fontsize=11, fontweight="bold", color=C["text"],
                 arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=1))
    ax1.annotate(f"Overhead {_gp[3]:.2f} (PUE {val('gem_pue')})", (0.23, 0.32), xytext=(0.228, 0.85),
                 ha="left", va="bottom", fontsize=11, fontweight="bold", color=C["text"],
                 arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=1))
    ax1.text(0.255, 0, f"= {val('gem_wh')} Wh", va="center", fontsize=13, fontweight="bold",
             color=C["text"])
    ax1.set_xlim(0, 0.36)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_xlabel("Energy per prompt (Wh)")
    a = 1.0 if k >= 2 else 0.0
    stacked(ax2, 0, [(K.GEM_S2, "network", f"Grid {val('gem_s2')}"),
                     (K.GEM_S13, "neutral", "")], alpha=a)
    ax2.text(0.0335, 0, f" + {val('gem_s13')} Scope 1 and 3  = {val('gem_total')} g", va="center",
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
         ("Model\nEvolved", K.CUM["model"], "compute", f"{val('ff_model')}×", 1),
         ("Chip\nTPU v2", K.CUM["processor"], "compute", f"{val('ff_processor')}×", 2),
         (f"PUE\n{val('ff_pue_b')}", K.CUM["datacenter"], "memory", f"{val('ff_datacenter')}×", 3),
         (f"Grid\n{val('ff_grid_b')}", K.CUM["grid"], "data", f"{val('ff_grid')}×", 4)]
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
def _llm(name, color, first):
    """One Table 4 bar; a star marks a value Luccioni et al. inferred."""
    d = K.LLM[name]
    star = lambda k: "*" if d[k + "_inferred"] else ""  # noqa: E731
    t = f"{d['tco2e_pue']:g} t{star('tco2e')}: {d['energy_mwh']:,.0f} MWh{star('energy')} at {d['grid_g_per_kwh']:.0f} g/kWh{star('grid')}"
    return (name, d["tco2e_pue"], color, t, first)


bar_steps("llm4", [
    _llm("GPT-3", "danger", 1),
    _llm("Gopher", "neutral", 2),
    _llm("OPT", "neutral", 2),
    _llm("BLOOM", "data", 1),
], 2, (5.8, 2.3), 1150, "Training emissions × PUE (t CO₂e)", label_size=12)
# The footnote ("* inferred by Luccioni et al.") is slide text since 2026-10-19.

# BLOOM life cycle (Luccioni et al. 2023, Table 3)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.8))
    parts = [(K.B_DYN, "compute", f"Dynamic\n{val('b_dyn')} t", 1), (K.B_IDLE, "neutral", f"Idle\n{val('b_idle')} t", 2),
             (K.B_EMB, "network", f"Embodied\n{val('b_emb')} t", 3)]
    left = 0
    for v, ck, txt, first in parts:
        a = 1.0 if k >= first else 0.0
        ax.barh(0, v, left=left, height=0.7, color=C[ck], edgecolor="white", alpha=a)
        ax.text(left + v / 2, 0, txt, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", alpha=a)
        left += v
    ax.text(K.B_DYN / 2, 0.5, f"What most papers report ({val('b_dyn_share')}%)", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C["compute"])
    ax.text(51.2, 0, f"= {val('b_total')} t", va="center", fontsize=13, fontweight="bold",
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
    _m = lambda q: K.mag(q, "dollar") / 10 ** 6  # noqa: E731
    TRAIN_M, INF_M, OPS_M = _m(K.TCO.training_cost), _m(K.TCO.inference_cost), _m(K.TCO.operations_cost)
    parts = [("Training", TRAIN_M, "neutral"), ("Inference", INF_M, "compute"),
             ("Operations", OPS_M, "neutral_l")]
    left = 0
    for name, v, ck in parts:
        ax1.barh(0, v, left=left, height=0.6, color=C[ck], edgecolor="white")
        left += v
    ax1.text(0.04, 0.55, f"Training ${val('train_3y_k')}K ({val('train_share')}%)", fontsize=11, color=C["text"],
             va="bottom", fontweight="bold")
    ax1.text(TRAIN_M + INF_M / 2, 0, f"Inference ${val('inf_3y_m')}M ({val('inf_share')}%)", fontsize=11,
             color="white", ha="center", va="center", fontweight="bold")
    ax1.text(TRAIN_M + INF_M + OPS_M / 2, 0, f"Ops ${val('ops_3y_k')}K", fontsize=11, ha="center",
             va="center", color=C["text"], fontweight="bold")
    ax1.set_xlim(0, 2.2)
    ax1.set_ylim(-0.4, 1.2)
    ax1.set_yticks([])
    ax1.spines["left"].set_visible(False)
    ax1.set_xlabel("Three-year cost (US$ millions)")
    a = 1.0 if k == 2 else 0.0
    for y, (name, v, ck) in zip((1, 0), ((f"{K.TCO.cycles} training cycles", num("train_t"), "neutral"),
                                         (f"{val('years')} years of inference", num("inf_t"), "compute"))):
        ax2.barh(y, v, height=0.6, color=C[ck], alpha=a)
        ax2.text(-0.01, y, name, transform=ax2.get_yaxis_transform(), ha="right",
                 va="center", fontsize=11, fontweight="bold", color=C["text"], alpha=a)
        ax2.text(v + 1.5, y, f"{v:g} t", va="center", fontsize=11, fontweight="bold",
                 color=C[ck], alpha=a)
    ax2.text(118, 0, f"{val('carbon_x_dlrm')}×", va="center", fontsize=14, fontweight="bold",
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
    (f"Cut serving cost {val('serve_cut_pct')}%", K.mag(K.TCO.serving_saving, "dollar") / 10 ** 3, "compute",
     f"${val('save_serve_k')}K, about {val('save_serve_t')} t", 1),
    ("Halve all training", K.mag(K.TCO.training_saving, "dollar") / 10 ** 3, "neutral",
     f"${val('save_train_k')}K, {val('save_train_t')} t", 1),
], 1, (4.6, 1.3), 520, "Three-year saving (US$ thousands)")

# Measured fleets (moved from Lecture 15; two rows)
fleet = [("Google, ML energy, 2019 to 2021", [("Training ~2/5", num("g_train_share"), "neutral_l"),
                                             ("Inference ~3/5", num("g_inf_share"), "compute")], 1),
         ("Meta, AI power capacity", [(val("m_exp"), num("m_exp"), "neutral_l"),
                                     (f"Train {val('m_train')}", num("m_train"), "neutral"),
                                     (f"Inference {val('m_inf')}", num("m_inf"), "compute")], 2)]
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
    _emb = K.mag(K.EMB.embodied_per_gpu_hour, "g")
    rows = [(f"Grid {val('fleet_grid')} kg/kWh", K.mag(K.EMB.operational_per_gpu_hour("fleet"), "g"), _emb,
             f"embodied {val('emb_dirty_pct')}%"),
            (f"Grid {val('bloom_grid')} kg/kWh", K.mag(K.EMB.operational_per_gpu_hour("France"), "g"), _emb,
             f"embodied {val('emb_clean_pct')}%")]
    for i, (lab, op, emb, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, op, height=0.6, color=C["neutral_l"], edgecolor="white")
        a = 1.0 if k >= 2 else 0.0
        ax.barh(y, emb, left=op, height=0.6, color=C["network"], edgecolor="white", alpha=a)
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(op + emb + 3, y, f"{op:.0f} + {val('emb_g')} g   {txt}" if k >= 2 else f"{op:.0f} g",
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

def _sup(digits):
    """Digits as Unicode superscripts, for an exponent in a label."""
    return digits.translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


bar_steps("explain", [
    ("Decisions", K.EXPL.decisions_per_day, "memory", f"{val('decisions_m')} million", 1),
    (f"Reasons, {val('samples')}-sample attributor", K.EXPL.sampled_evaluations, "compute",
     f"{val('sampled_m')} million, {val('sampled_x')}×", 2),
    (f"Reasons, exact (2{_sup(val('features'))} per denial)", K.EXPL.exact_evaluations, "danger",
     f"{val('exact_mant')} × 10{_sup(val('exact_exp'))}", 3),
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
    ax.plot([K.REVIEW.cases_per_hour], [3600 / K.REVIEW.cases_per_hour], "o", color=C["memory"], ms=8)
    ax.text(14, 520, f"Credit service: {val('case_min')} min each,\n{val('appeals')} per {val('review_h')} reviewer-hours",
            fontsize=11, color=C["memory"], fontweight="bold", va="bottom")
    a = 1.0 if k >= 2 else 0.0
    ax.plot([K.REVIEW.compare_cases_per_hour], [3600 / K.REVIEW.compare_cases_per_hour], "o", color=C["danger"], ms=8, alpha=a)
    ax.text(2400, 0.42, f"Cigna: {val('cigna_s')} s each,\n{val('cigna_cases')} per {val('cigna_h')} h ", fontsize=11,
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
    ("Purchase", num("sisa_purchase"), "data", f"{val('sisa_purchase')}×", 1),
    ("SVHN", num("sisa_svhn"), "data", f"{val('sisa_svhn')}×", 1),
    ("ImageNet", num("sisa_imagenet"), "data", f"{val('sisa_imagenet')}×", 1),
], 1, (4.2, 1.7), 6.2, "Unlearning speedup over retraining")

# DP-SGD per-sample gradient memory (moved from Lecture 15)
# Redrawn 2026-10-19 for a half-width column beside the Opacus code: labels
# sit above the bars so the plot can be narrow.
fig, ax = plt.subplots(figsize=(3.0, 1.25))
ax.barh(0.9, K.mag(K.GRADS.per_batch("KWS"), "MB"), height=0.45, color=C["memory"])
ax.text(36, 1.17, f"KWS: {val('kws_batch_mb')} MB", fontsize=11,
        fontweight="bold", color=C["memory"], va="bottom")
ax.barh(0, K.mag(K.GRADS.per_batch("GPT-2"), "MB"), height=0.5, color=C["danger"])
ax.text(40, 0, f"GPT-2: {val('gpt2_batch_tb')} TB", fontsize=11, fontweight="bold",
        color="white", va="center")
ax.axvline(num("h100_gb") * 10 ** 3, color=C["neutral"], lw=1.6, ls="--")
# dashed line: one H100, 80 GB (labeled in the slide text)
ax.set_xscale("log")
ax.set_xlim(30, 1e8)
ax.set_ylim(-0.45, 1.65)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel(f"Batch of {val('batch')}, MB (log)", fontsize=11)
sp.save(fig, out / "dpsgd_step1.svg")

# Corrected 2026-10-19: the ImageNet result fine-tunes a model pre-trained on
# a large public dataset (JFT-4B), and 4.3 points is the gap to the best
# non-private model of any kind; the same model without privacy is 1.8 points
# higher (De et al. 2022, Table 6). CIFAR-10 is trained from scratch; the
# same architecture reaches above 95 percent without privacy.
bar_steps("dp_accuracy", [
    ("CIFAR-10, no privacy", K.CIFAR_NP, "neutral", f"above {val('cifar_np')}%", 1),
    (f"CIFAR-10, ε = {val('eps')}, from scratch", K.CIFAR_DP, "data",
     f"{val('cifar_dp')}%: over {val('cifar_gap')} points lower", 1),
    (f"ImageNet, ε = {val('eps')}, public pre-training", K.IM_DP, "data", f"{val('im_dp')}%", 1),
    ("Same model, no privacy", K.IM_SAME, "neutral",
     f"{val('im_same')}%: {val('im_gap_same')} points higher", 1),
    ("Best non-private model", K.IM_BEST, "neutral",
     f"{val('im_best')}%: {val('im_gap_best')} points higher", 1),
], 1, (5.4, 2.2), 150, "Top-1 accuracy (%)", label_size=11)

bar_steps("kws_bytes", [
    ("One hour of raw audio", num("audio_mb_h"), "danger", f"{val('audio_mb_h')} MB", 1),
    ("One model update", num("update_mb"), "data", f"{val('update_mb')} MB, {val('audio_x')}× smaller", 1),
], 1, (5.8, 1.1), 5000, "Bytes leaving the home (MB, log scale)", log=True, xmin=0.1)

bar_steps("audit", [
    (f"Credit: {val('ret_months')} months, {val('rec_b')} B each", K.mag(K.LOG.total, "GB"), "memory", f"{val('log_gb')} GB", 1),
    (f"DLRM: 1 year, {val('rec_b')} B each", K.mag(K.DLOG.total, "GB"), "memory", f"{val('dlrm_manifest_tb')} TB", 2),
    ("DLRM: 1 year, full input", K.mag(K.DLOG.payload_total, "GB"), "danger",
     f"{val('dlrm_input_tb')} TB, {val('payload_x')}×", 2),
], 2, (5.8, 1.8), 3e8, "Audit log size (GB, log scale)", log=True, xmin=10,
    note=f"{val('rec_b')} B per record is an assumption; DLRM input as in MLPerf DLRM (constructed)")


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
boxes = [(f"{val('users_m')}M users × {val('items')} items", f"{val('inf_day_m')}M inferences\na day", 1),
         (f"× {val('service_ms')} ms each", f"{val('gpu_h_day')} GPU-hours\na day", 2),
         ("÷ 24 hours", f"{val('gpus')} GPUs\nalways on", 3),
         (f"× ${val('inf_rate')} an hour", f"${val('serve_year_k')}K\na year", 4)]
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
