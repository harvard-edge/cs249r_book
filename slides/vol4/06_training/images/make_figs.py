#!/usr/bin/env python3
"""Data plots and diagrams for Volume IV Lecture 6 (Policy Training).

Every value comes from calcs.py (which names its source) or from a paper
table typed there. Illustrative figures say so on the figure.

  - Dactyl hook and ablation: Andrychowicz et al. 2020, Tables 3, 4, 5 and
    section 6.3 (simulated hours), arXiv 1808.00177.
  - Mode averaging, drift regions: chapter section 6.2.3, 6.2.4 (schematic).
  - Decoder slack: chapter Napkin Math 6.1.
  - Discount weight: chapter section 6.2.1.
  - Offline value overestimation: chapter section 6.2.7 (schematic).
  - Coverage cells and declared ODD: chapter section 6.2.8, 6.8.
  - Reset wall: chapter Napkin Math 6.2, section 6.3.1, section 6.4.1.
  - Soft spring: chapter Napkin Math 6.3, equations 7 and 8.
  - Reality gap vector: chapter section 6.5 (illustrative linear drive).

Steps of one figure draw every element (hidden ones transparent), so the
bounding box and axes never move between steps.

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
# Volume IV diagram roles (rules/volumes/vol4/palette.md)
PROP = "#2563EB"     # unprivileged proposal
NERV = "#059669"     # deterministic, verified
VETO = "#DC2626"     # limit, hazard
SLATE = "#334155"    # physical plant


def a(shown):
    return 1.0 if shown else 0.0


# ---------------------------------------------------------------------------
# 1. Hook: Dactyl, simulation median 50 against ten physical trials
# ---------------------------------------------------------------------------
def dactyl_hook(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    ax.barh(1, calcs.DACTYL_SIM_MEDIAN, height=0.5, color=C["neutral_l"],
            edgecolor=C["neutral"])
    ax.text(calcs.DACTYL_SIM_MEDIAN + 1, 1, "median 50", va="center",
            fontsize=13, fontweight="bold", color=C["neutral"])
    trials = calcs.DACTYL_REAL["All randomizations"]
    ys = 0 + np.linspace(-0.18, 0.18, len(trials))
    ax.scatter(trials, ys, s=70, color=C["memory"], alpha=a(k >= 2), zorder=3)
    med = calcs.DACTYL_MEDIAN["All randomizations"]
    ax.plot([med, med], [-0.32, 0.32], color=C["danger"], lw=3, alpha=a(k >= 2))
    ax.text(med + 1, -0.42, "median 13", color=C["danger"], fontsize=13,
            fontweight="bold", va="top", alpha=a(k >= 2))
    ax.text(25, 0, "?", fontsize=26, fontweight="bold", color=C["neutral"],
            ha="center", va="center", alpha=a(k == 1))
    ax.set_yticks([1, 0], ["Simulation\n(100 trials)", "Physical hand\n(10 trials)"],
                  fontsize=12)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.75, 1.45)
    ax.set_xlabel("Consecutive rotations before a drop or timeout (cap 50)")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"dactyl_hook_step{k}.svg")


# ---------------------------------------------------------------------------
# 2. Dactyl ablation (Table 4) and memory (Table 5), dot plots
# ---------------------------------------------------------------------------
def dotrows(path, rows, medians, size, highlight=(), names=None):
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, (name, trials) in enumerate(rows.items()):
        y = n - 1 - i
        col = C["memory"] if name in highlight else C["neutral"]
        ys = y + np.linspace(-0.16, 0.16, len(trials))
        ax.scatter(trials, ys, s=55, color=col, zorder=3)
        m = medians[name]
        ax.plot([m, m], [y - 0.3, y + 0.3], color=C["danger"], lw=2.5, zorder=4)
        ax.text(53, y, f"{m:g}", va="center", fontsize=13, fontweight="bold",
                color=C["danger"])
    ax.text(53, n - 0.35, "median", fontsize=11, color=C["danger"], fontweight="bold")
    ax.set_yticks(range(n - 1, -1, -1), [(names or {}).get(k, k) for k in rows], fontsize=12)
    ax.set_xlim(-1, 58)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_xlabel("Rotations on the physical hand")
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


# ---------------------------------------------------------------------------
# 3. Mode averaging around a column (schematic)
# ---------------------------------------------------------------------------
def mode_avg(k):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.add_patch(plt.Circle((0, 3), 0.85, color=SLATE))
    ax.text(0, 3, "column", color="white", ha="center", va="center",
            fontsize=12, fontweight="bold")
    t = np.linspace(0, 1, 100)
    for sgn, lab in ((-1, "demo A: left"), (1, "demo B: right")):
        x = sgn * 1.75 * np.sin(np.pi * t)
        y = 6 * t
        ax.plot(x, y, color=PROP, lw=3, alpha=a(k >= 2))
        ax.text(sgn * 1.95, 3, lab, color=PROP, fontsize=14, fontweight="bold",
                ha="right" if sgn < 0 else "left", va="center", alpha=a(k >= 2))
    ax.plot([0, 0], [0, 2.05], color=VETO, lw=3.5, alpha=a(k >= 3))
    ax.annotate("", xy=(0, 2.12), xytext=(0, 1.7), alpha=a(k >= 3),
                arrowprops=dict(arrowstyle="-|>", color=VETO, lw=3,
                                alpha=a(k >= 3)))
    ax.annotate("the mean:\nno demo did this", xy=(0.08, 1.1), xytext=(2.0, 0.7),
                color=VETO, fontsize=13, fontweight="bold", va="center",
                alpha=a(k >= 3), linespacing=1.0,
                arrowprops=dict(arrowstyle="-", color=VETO, lw=1.5, alpha=a(k >= 3)))
    ax.plot(0, 0, "s", ms=16, color=SLATE)
    ax.text(0, -0.55, "robot", ha="center", va="top", fontsize=13, color=SLATE,
            fontweight="bold")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-1.1, 6.3)
    ax.set_aspect("equal")
    ax.axis("off")
    sp.save(fig, out / f"mode_avg_step{k}.svg")


# ---------------------------------------------------------------------------
# 4. Drift: three regions, and velocity against torque drift (schematic)
# ---------------------------------------------------------------------------
def drift():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.4, 2.6),
                                 gridspec_kw=dict(width_ratios=[1.15, 1]))
    x = np.linspace(0, 10, 200)
    ax.fill_between(x, -0.6, 0.6, color=C["data_l"])
    ax.fill_between(x, 0.6, 1.8, color=C["compute_l"])
    ax.fill_between(x, 1.8, 3.0, color=C["danger_l"])
    ax.plot(x, 0 * x, color=C["data"], lw=2.5)
    learner = np.where(x < 3, 0, 0.035 * np.clip(x - 3, 0, None) ** 2.1)
    ax.plot(x, np.minimum(learner, 2.9), color=SLATE, lw=2.5, ls="--")
    ax.text(0.2, -0.35, "demonstrated", color=C["data"], fontsize=12, fontweight="bold")
    ax.text(0.2, 1.05, "recoverable:\nlabels help", color=C["compute"], fontsize=11,
            fontweight="bold", va="center", linespacing=1.0)
    ax.text(0.2, 2.4, "no restoring action", color=C["danger"], fontsize=11,
            fontweight="bold", va="center")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 3.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Task progress")
    ax.set_ylabel("Deviation")
    ax.set_title("Where labels can help", fontsize=13)
    tt = np.linspace(0, 1, 100)
    bx.plot(tt, tt, color=C["memory"], lw=3)
    bx.plot(tt, tt ** 2 * 2.0, color=C["danger"], lw=3)
    bx.text(0.5, 0.08, "velocity error:\ngrows as t", color=C["memory"],
            fontsize=11, fontweight="bold", linespacing=1.0)
    bx.text(0.05, 1.3, "torque error:\ngrows as t²", color=C["danger"],
            fontsize=11, fontweight="bold", linespacing=1.0)
    bx.set_xticks([])
    bx.set_yticks([])
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 2.0)
    bx.set_xlabel("Time")
    bx.set_ylabel("Position drift")
    bx.set_title("Same persistent error", fontsize=13)
    fig.text(0.99, 0.0, "Schematic", ha="right", fontsize=9, color=C["neutral"])
    fig.tight_layout(w_pad=1.5)
    sp.save(fig, out / "drift.svg")


# ---------------------------------------------------------------------------
# 5. Decoder slack (Napkin Math 6.1)
# ---------------------------------------------------------------------------
def decoder(k):
    per, flow = calcs.per_eval, calcs.flow_ms
    rows = [("ACT, 1 pass", calcs.ACT_MS, C["memory"], 1),
            ("Flow, 4 steps", flow, C["data"], 2),
            ("DDIM, 16 steps", calcs.DDIM_MS, C["compute"], 2),
            ("DDIM + 5 ms jitter", calcs.DDIM_MS + calcs.JITTER_MS, C["compute"], 3)]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    n = len(rows)
    for i, (lab, v, col, s) in enumerate(rows):
        y = n - 1 - i
        sh = k >= s
        ax.barh(y, min(v, calcs.RENEW_MS), color=col, height=0.6, alpha=a(sh))
        if v > calcs.RENEW_MS:
            ax.barh(y, v - calcs.RENEW_MS, left=calcs.RENEW_MS, color=VETO,
                    height=0.6, alpha=a(sh))
        ax.text((calcs.RENEW_MS if 0 < calcs.RENEW_MS - v < 4 else v) + 0.8, y, f"{v:g} ms", va="center", fontsize=13,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
                fontweight="bold", color=VETO if v > calcs.RENEW_MS else col,
                alpha=a(sh))
        ax.text(-1, y, lab, ha="right", va="center", fontsize=12.5,
                fontweight="bold", color=C["text"], alpha=a(sh))
    for xv, lab, col in ((calcs.STEP_MS, "setpoint step 25 ms", VETO),
                         (calcs.RENEW_MS, "renewal 50 ms", C["neutral"])):
        ax.axvline(xv, color=col, lw=2, ls="--")
        ax.text(xv, n - 0.25, lab, color=col, fontsize=11.5, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.6, n + 0.15)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Proposal latency, ms (illustrative heads)")
    sp.save(fig, out / f"decoder_step{k}.svg")


# ---------------------------------------------------------------------------
# 6. Discount weight across one opening
# ---------------------------------------------------------------------------
def discount():
    steps, gmin = calcs.n_steps, calcs.g_half
    k = np.arange(0, int(steps) + 1)
    tsec = k * calcs.STEP_MS / 1000
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(tsec, calcs.GAMMA ** (steps - k), color=C["danger"], lw=3)
    ax.plot(tsec, gmin ** (steps - k), color=C["memory"], lw=3)
    ax.text(0.15, 0.2, "γ = 0.99: weight 0.04", color=C["danger"],
            fontsize=12.5, fontweight="bold")
    ax.text(0.15, 0.8, "γ = 0.9978: weight 0.50", color=C["memory"],
            fontsize=12.5, fontweight="bold")
    ax.axvspan(8 - 2.5, 8, color=C["danger_l"], alpha=0.6, zorder=0)
    ax.text(6.75, 1.05, "last 2.5 s\n(100 steps)", ha="center", fontsize=11,
            color=C["danger"], fontweight="bold", linespacing=1.0)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Time of the decision in an 8 s opening (s)")
    ax.set_ylabel("Weight of the\ndoor-open reward", fontsize=11)
    ax.grid(color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "discount.svg")


# ---------------------------------------------------------------------------
# 7. Offline value overestimation (schematic)
# ---------------------------------------------------------------------------
def offline(k):
    x = np.linspace(0, 10, 400)
    true_q = 2.0 * np.exp(-((x - 3.0) / 1.6) ** 2) - 0.05 * x
    fit = true_q + 2.6 * np.exp(-((x - 7.6) / 0.6) ** 2)
    logged = np.array([1.6, 2.1, 2.6, 2.9, 3.3, 3.8, 4.2])
    cql = fit - 1.2 * (1 - np.exp(-np.min(np.abs(x[:, None] - logged[None, :]), axis=1) ** 2 / 0.8))
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(x, true_q, color=C["neutral"], lw=2, ls=":")
    ax.text(0.3, 2.15, "true value", color=C["neutral"], fontsize=11, fontweight="bold")
    ax.plot(x, fit, color=C["danger"], lw=3, alpha=a(k >= 1))
    i = np.argmax(fit)
    ax.plot(x[i], fit[i], "o", ms=11, color=C["danger"], alpha=a(k >= 1))
    ax.text(x[i] - 0.3, fit[i] + 0.05, "greedy pick:\nnever tried", color=C["danger"],
            fontsize=12, fontweight="bold", ha="right", va="center", alpha=a(k >= 1),
            linespacing=1.0)
    ax.plot(x, cql, color=C["memory"], lw=3, alpha=a(k >= 2))
    j = np.argmax(cql)
    ax.plot(x[j], cql[j], "o", ms=11, color=C["memory"], alpha=a(k >= 2))
    ax.text(1.0, -2.5, "conservative fit: lowered away from logged actions",
            color=C["memory"], fontsize=12, fontweight="bold", alpha=a(k >= 2),
            linespacing=1.0)
    for lx in logged:
        ax.plot([lx, lx], [-1.9, -1.6], color=C["data"], lw=3)
    ax.text(4.5, -1.75, "logged actions", color=C["data"], fontsize=11.5,
            fontweight="bold", va="center", ha="left")
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.6, 3.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Action (for example, approach speed)")
    ax.set_ylabel("Estimated value Q")
    fig.text(0.99, 0.0, "Schematic", ha="right", fontsize=9, color=C["neutral"])
    sp.save(fig, out / f"offline_step{k}.svg")


# ---------------------------------------------------------------------------
# 8. Coverage cells: pretraining joins none
# ---------------------------------------------------------------------------
PLATES = ["nominal", "displaced +", "displaced −"]
SPEEDS = ["guarded\n0.03 m/s", "fast\n0.10 m/s"]


def grid(ax, fills, labels, title=None, fs=13):
    for r in range(3):
        for c in range(2):
            fc, ec = fills[r][c]
            ax.add_patch(plt.Rectangle((c, 2 - r), 0.94, 0.9, facecolor=fc,
                                       edgecolor=ec, lw=2))
            ax.text(c + 0.47, 2 - r + 0.45, labels[r][c], ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=ec, linespacing=1.0)
    ax.set_xlim(-0.05, 2.0)
    ax.set_ylim(-0.05, 3.05)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=C["text"])


GREEN = (C["data_l"], C["data"])
GRAY = ("#F2F2F2", C["neutral"])
RED = (C["danger_l"], C["danger"])
HELD = (C["compute_l"], C["compute"])
ASK = ("white", C["memory"])


def cells_support():
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    fills = [[GREEN, GRAY], [GRAY, GRAY], [GRAY, GRAY]]
    labels = [["supported", "absent"], ["absent", "absent"], ["absent", "absent"]]
    grid(ax, fills, labels, fs=11.5)
    for c, s in enumerate(SPEEDS):
        ax.text(c + 0.47, 3.0, s, ha="center", va="bottom", fontsize=12,
                color=C["text"], linespacing=1.0)
    for r, p in enumerate(PLATES):
        ax.text(-0.08, 2 - r + 0.45, p, ha="right", va="center", fontsize=12,
                color=C["text"])
    ax.set_xlim(-1.25, 2.0)
    ax.set_ylim(-0.1, 3.55)
    sp.save(fig, out / "cells_support.svg")


def odd(k):
    fig, axs = plt.subplots(1, 3, figsize=(5.9, 2.9))
    fast = RED
    specs = [
        ("latch-bc-01", [[GREEN, fast], [GRAY, fast], [GRAY, fast]],
         [["declared", "out"], ["absent", "out"], ["absent", "out"]], 1),
        ("latch-res-03", [[HELD, fast], [GRAY, fast], [GRAY, fast]],
         [["held", "out"], ["absent", "out"], ["absent", "out"]], 2),
        ("bc-recovery-02", [[GREEN, fast], [ASK, fast], [GRAY, fast]],
         [["declared", "out"], ["requested", "out"], ["absent", "out"]], 3),
    ]
    for ax, (title, fills, labels, s) in zip(axs, specs):
        grid(ax, fills, labels, title, fs=11.5)
        for art in ax.patches + ax.texts:
            art.set_alpha(a(k >= s))
        ax.title.set_alpha(a(k >= s))
    fig.tight_layout(w_pad=0.6)
    sp.save(fig, out / f"odd_step{k}.svg")


# ---------------------------------------------------------------------------
# 9. Reset wall (Napkin Math 6.2), log days
# ---------------------------------------------------------------------------
def reset_wall():
    r = calcs.wall
    rows = [("Manual resets", r["man"], "danger", f"{r['man']:.1f} days"),
            ("Automated resets", r["auto"], "compute", f"{r['auto']:.1f} days"),
            ("Execution only", r["exe"], "neutral", f"{r['exe']:.1f} days"),
            ("14 rigs, manual", r["farm"], "network", f"{r['farm']:.1f} days"),
            ("HIL-SERL budget", r["hil_days"], "data", "2.5 h, 750 trials"),
            ("Simulation", r["sim_days"], "memory", f"{r['sim_days'] * 1440:.1f} min")]
    for k, n in ((1, 3), (2, 4), (3, 6)):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        for i, (lab, v, ck, txt) in enumerate(rows):
            y = len(rows) - 1 - i
            sh = i < n
            ax.barh(y, v, color=C[ck], height=0.62, alpha=a(sh))
            ax.text(0.0075, y, lab, ha="right", va="center", fontsize=13,
                    fontweight="bold", color=C["text"])
            ax.text(v * 1.2, y, txt, va="center", fontsize=13, fontweight="bold",
                    color=C[ck], alpha=a(sh))
        ax.set_xscale("log")
        ax.set_xlim(0.01, 1e4)
        ticks = [0.01, 0.1, 1, 10, 100, 1000]
        ax.set_xticks(ticks, ["0.01", "0.1", "1", "10", "100", "1,000"], fontsize=12)
        ax.minorticks_off()
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Wall-clock time, days (log scale)", fontsize=12)
        sp.save(fig, out / f"reset_wall_step{k}.svg")


# ---------------------------------------------------------------------------
# 10. Soft-spring exploit (Napkin Math 6.3)
# ---------------------------------------------------------------------------
def spring(k):
    s = calcs.spring_vals
    kr = np.logspace(3, 8, 400)
    keff = 1 / (1 / calcs.K_PLATE + 1 / kr)
    f = calcs.V_FAST * np.sqrt(keff * calcs.MASS)
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    ax.axhspan(calcs.F_LIMIT, 220, color=C["danger_l"], alpha=0.5, zorder=0)
    ax.axhline(calcs.F_LIMIT, color=VETO, lw=2, ls="--")
    ax.text(1.2e3, calcs.F_LIMIT + 6, "latch limit 100 N", color=VETO, fontsize=12,
            fontweight="bold")
    ax.plot([s["k_pen"]], [s["f_sim"]], "o", ms=12, color=C["memory"])
    ax.text(1.2e3, 55, "simulator's spring:\n37.5 N",
            color=C["memory"], fontsize=12, fontweight="bold", linespacing=1.0)
    ax.axhline(s["f_rigid"], color=SLATE, lw=2, ls=":", alpha=a(k >= 2))
    ax.text(1.2e3, s["f_rigid"] + 6, "rigid arm on the real plate: 189.7 N",
            color=SLATE, fontsize=12, fontweight="bold", alpha=a(k >= 2))
    ax.plot(kr, f, color=C["compute"], lw=3, alpha=a(k >= 3))
    ax.plot([s["k_rob_star"]], [calcs.F_LIMIT], "o", ms=11, color=VETO, alpha=a(k >= 3))
    ax.text(2.2e6, 50, "arm stiffer than\n1.54×10⁵ N/m\nexceeds the limit",
            color=VETO, fontsize=12, fontweight="bold", alpha=a(k >= 3), linespacing=1.0)
    ax.set_xscale("log")
    ax.set_xlim(1e3, 1e8)
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
                  ["10\u00b3", "10\u2074", "10\u2075", "10\u2076", "10\u2077", "10\u2078"],
                  fontsize=12)
    ax.minorticks_off()
    ax.set_ylim(0, 220)
    ax.set_xlabel("Robot-side stiffness, N/m (log scale)")
    ax.set_ylabel("Peak force at 0.10 m/s, N", fontsize=11)
    ax.grid(color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"spring_step{k}.svg")


# ---------------------------------------------------------------------------
# 11. Reality gap vector, four channels (illustrative linear drive)
# ---------------------------------------------------------------------------
def gap_vector(k):
    fig, axs = plt.subplots(2, 2, figsize=(6.0, 3.1))
    sim_kw = dict(color=C["neutral"], lw=2.2, ls="--")
    hw_kw = dict(color=C["memory"], lw=2.6)
    show = [k >= 1, k >= 2, k >= 3, k >= 4]
    # (a) dynamics
    ax = axs[0, 0]
    t = np.linspace(0, 60, 300)
    ax.plot(t, np.clip(t * 0.5, 0, 10), **sim_kw)
    ax.plot(t, np.clip(t * 0.25, 0, 10), **hw_kw)
    ax.text(21, 2.6, "20 ms", color=C["neutral"], fontsize=12, fontweight="bold")
    ax.text(41, 5.0, "40 ms", color=C["memory"], fontsize=12, fontweight="bold")
    ax.set_title("Dynamics: time to 10 N", fontsize=13)
    ax.set_xlabel("ms", fontsize=12, labelpad=0)
    # (b) timing
    ax = axs[0, 1]
    t = np.linspace(0, 120, 300)
    ax.plot(t, 0.2 * t, **sim_kw)
    ax.plot(t, 0.2 * np.clip(t - 49.5, 0, None), **hw_kw)
    ax.text(3, 26, "p99 delay 49.5 ms:\n9.9 mm unobserved", color=C["memory"],
            fontsize=12, fontweight="bold", linespacing=1.0)
    ax.set_ylim(0, 40)
    ax.set_title("Timing: command to motion", fontsize=13)
    ax.set_xlabel("ms", fontsize=12, labelpad=0)
    # (c) sensing
    ax = axs[1, 0]
    t = np.linspace(0, 10, 200)
    ax.plot(t, 0 * t, **sim_kw)
    y = 3.2 + 0 * t
    y[(t > 5) & (t < 6.6)] = np.nan
    ax.plot(t, y, **hw_kw)
    ax.axvspan(5, 6.6, color=C["neutral_l"])
    ax.text(0.3, 1.0, "bias +3.2 mm", color=C["memory"], fontsize=12, fontweight="bold")
    ax.text(5.8, 4.3, "dropout", ha="center", fontsize=11.5, color=C["neutral"],
            fontweight="bold", linespacing=1.0)
    ax.set_ylim(-1, 6)
    ax.set_title("Sensing: range error", fontsize=13)
    # (d) appearance
    ax = axs[1, 1]
    ax.axvspan(-2, 2, color=C["data_l"])
    ax.plot([0], [0.5], "o", ms=10, color=C["neutral"])
    ax.plot([6], [0.5], "o", ms=10, color=VETO)
    ax.text(0, 0.8, "sim", ha="center", fontsize=12, color=C["neutral"], fontweight="bold")
    ax.text(6, 0.8, "real: 6 mm", ha="center", fontsize=12, color=VETO, fontweight="bold")
    ax.text(0, 0.15, "±2 mm tolerance", ha="center", fontsize=12, color=C["data"],
            fontweight="bold")
    ax.set_xlim(-4, 9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Appearance: contact patch", fontsize=13)
    for i, ax in enumerate(axs.flat):
        ax.set_yticks([])
        if i >= 2:
            ax.set_xticks([])
        else:
            ax.tick_params(labelsize=12)
        for art in ax.lines + ax.texts + ax.patches + ax.collections:
            art.set_alpha(a(show[i]) * (art.get_alpha() if art.get_alpha() is not None else 1))
    fig.text(0.01, 0.0, "dashed: simulator    solid: hardware    illustrative values",
             fontsize=12, color=C["neutral"])
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    sp.save(fig, out / f"gap_vector_step{k}.svg")


if __name__ == "__main__":
    calcs.n_steps, calcs.g_half = calcs.discount()
    calcs.per_eval, calcs.flow_ms = calcs.decoder()
    calcs.wall = calcs.reset_wall()
    calcs.spring_vals = calcs.spring()
    for k in (1, 2):
        dactyl_hook(k)
    dotrows(out / "dactyl_ablation.svg", calcs.DACTYL_REAL, calcs.DACTYL_MEDIAN,
            (4.0, 2.7), highlight=("All randomizations",),
            names={"All randomizations": "All", "No observation noise": "No obs. noise",
                   "No physics randomizations": "No physics",
                   "No unmodeled effects": "No added effects",
                   "No randomizations": "None"})
    dotrows(out / "dactyl_memory.svg", calcs.DACTYL_MEMORY, calcs.DACTYL_MEMORY_MEDIAN,
            (2.9, 1.9), highlight=("LSTM policy",),
            names={"LSTM policy": "LSTM", "FF policy, LSTM value": "FF, LSTM V",
                   "FF policy, FF value": "FF, FF V"})
    for k in (1, 2, 3):
        mode_avg(k)
        decoder(k)
        spring(k)
        odd(k)
    drift()
    discount()
    for k in (1, 2):
        offline(k)
    cells_support()
    reset_wall()
    for k in (1, 2, 3, 4):
        gap_vector(k)
