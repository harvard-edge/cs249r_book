#!/usr/bin/env python3
"""Figures for Volume II, Lecture 8 (Fleet Orchestration).

Every value is sourced. "FO" is the rendered chapter, Fleet Orchestration.
Values computed here are labeled "derived" on the figure or its slide;
schematic layouts are labeled "illustrative".

  - Hook pools (FO Example 8.2, table 8): A100-80 GB 400, A100-40 GB 424,
    V100-32 GB 200 GPUs; 60 percent utilization, more than 50 jobs pending.
    Synthesis (table 9): A100-80 GB 72 to 94 percent active; A100-40 GB 64 to
    88 percent allocated; V100 0 to 71 percent allocated; cluster 60 to 84.
  - Recall (Fault Tolerance, Napkin Math 7.3): detect 60 s, restart 3 min,
    load 24.5 s, warm up 2 min, about 6.4 min.
  - Kalos trace (Hu et al., NSDI 2024; AcmeTrace, CC BY 4.0), read from
    MLSysIM Literature.Series.AcmeKalosJobSizes
    (mlsysim/literature/data/series/acme_kalos_job_sizes.csv). Class shares,
    mean GPU-hours per job, the Cs lower bound, and the job-state shares are
    derived here from those rows.
  - Gang deadlock (FO figure 4, section 8.4.3): 8-GPU pool, two 8-GPU jobs.
  - Backfill (FO section 8.4.3): 900 free, gang needs 1,024; a 128-GPU,
    2-hour job backfills while 124 GPUs stay busy for at least 2 hours.
  - DRF (FO table 1): 8 GPUs, 32 cores, 256 GB, 80 Gb/s; A replica 1, 4, 32,
    10; B worker 2, 2, 16, 20; 4 replicas and 2 workers, dominant share 0.50.
  - Fragmentation grid (FO figure 3): 19 of 64 slots free, 30 percent;
    the slot layout here is illustrative.
  - M/G/1 (FO Napkin Math 8.1): W / (1/mu) = rho / (1 - rho) x (1 + Cs^2) / 2.
  - NCCL all_reduce_perf on one 8x H100 node (NVIDIA exemplar-performance,
    MLSysIM Literature.Series.NcclAllreduceH100Node): in-place bus bandwidth.
    One NDR port 50 GB/s per direction (FO section 8.4.2, spec).
  - Placement (FO Napkin Math 8.2): 1 GB at 450 GB/s = 2.2 ms, at 50 GB/s
    = 20 ms.
  - Elastic decision (FO Napkin Math 8.3): 24, 64, 39.7 epochs; GPU-hours
    derived.
  - MLPerf Training v3.1 GPT-3 175B on NVIDIA Eos (MLSysIM
    Literature.Series.MLPerfTrainingLLMScaling): minutes by GPU count.
  - Preemption (FO section 8.8.1): 15 min lost, 3 min relaunch, 24.5 s
    reload, 2 min warmup = 20.4 min; 12 per day = 4.1 h.
  - Spot (FO Systems Perspective 8.4): on-demand $344,064 over 336 h; spot
    $161,567 over 450.8 h.
  - Sharing (FO Napkin Math 8.4): 26 GB per 7B model on an 80 GB budget;
    180 against 130 GPUs.
  - Diurnal (FO Napkin Math 8.5): 1,024 GPUs; serving average 150, peak 283,
    buffer 117, static reservation 400, training base 618, headroom 6;
    reclaimed average 133. The daily curve's shape is illustrative.
  - Quota (FO section 8.11.1): Team A guarantee 500, uses 300, lends 200.
  - Tiers (FO section 8.12.1): 500, 350, 280.

Run from this folder: python3 make_figs.py
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent
SERIES = Path(__file__).resolve().parents[4] / "mlsysim" / "mlsysim" / "literature" / "data" / "series"

# Volume II C3 roles for schematics (volumes/vol2/palette.md)
V = {
    "c1": "#1F407A", "c1_l": "#EAF1F8",      # compute
    "c2": "#d97706", "c2_l": "#fef3c7",      # communication
    "c3": "#7c3aed", "c3_l": "#f3e8ff",      # coordination
    "ok": "#16a34a", "ok_l": "#dcfce7",      # healthy / storage
    "bad": "#dc2626", "bad_l": "#fee2e2",    # faults, waste
    "frame": "#94a3b8", "frame_l": "#f8fafc",
    "ink": "#0f172a", "gray": "#64748b",
}


def hide_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def box(ax, x, y, w, h, fc, ec, lw=1.6, ls="-", hatch=None, r=0.02, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, hatch=hatch, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color, lw=2.0, style="-|>", ls="-", z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw, ls=ls, zorder=z))


def load_rows(name):
    rows = list(csv.DictReader((SERIES / name).open()))
    return rows


# ====================================================================== hook
POOLS = [("A100-80 GB", 400), ("A100-40 GB", 424), ("V100-32 GB", 200)]


def hook_pools():
    fig, ax = plt.subplots(figsize=(4.2, 1.8))
    colors = [V["c1"], "#4a6fa5", V["gray"]]
    for i, ((lab, n), col) in enumerate(zip(POOLS, colors)):
        y = 2 - i
        ax.barh(y, n, color=col, height=0.62)
        ax.text(n + 10, y, f"{n} GPUs", va="center", fontsize=13, color=col, fontweight="bold")
        ax.text(-10, y, lab, ha="right", va="center", fontsize=13, fontweight="bold", color=V["ink"])
    ax.set_xlim(0, 600)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xticks([0, 200, 400])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("GPUs per pool, 1,024 in all (three procurement cycles)")
    sp.save(fig, out / "hook_pools.svg")


def hook_answer(path, k):
    """Table 9 as bars, before (gray) and after (green); k=1 before only."""
    rows = [("A100-80 GB\n(active)", 72, 94), ("A100-40 GB\n(allocated)", 64, 88),
            ("V100-32 GB\n(allocated)", 0, 71), ("Cluster", 60, 84)]
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    for i, (lab, b, a) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + 0.19, b, height=0.36, color=C["neutral"])
        ax.text(b + 1.2, y + 0.19, f"{b}%", va="center", fontsize=12, color=C["neutral"],
                fontweight="bold")
        ax.barh(y - 0.19, a if k >= 2 else 0, height=0.36, color=C["data"])
        ax.text(a + 1.2, y - 0.19, f"{a}%", va="center", fontsize=12, color=C["data"],
                fontweight="bold", alpha=1.0 if k >= 2 else 0.0)
        ax.text(-1.5, y, lab, ha="right", va="center", fontsize=12,
                fontweight="bold" if lab == "Cluster" else "normal", color=C["text"])
    ax.text(0, 3.75, "before", ha="left", fontsize=12, color=C["neutral"], fontweight="bold")
    ax.text(22, 3.75, "after new policies", ha="left", fontsize=12, color=C["data"],
            fontweight="bold", alpha=1.0 if k >= 2 else 0.0)
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.55, 4.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Utilization of the pool (percent)")
    sp.save(fig, path)


# ==================================================================== recall
def recall_budget():
    parts = [("detect", 60, C["neutral"]), ("restart", 180, V["c3"]),
             ("load", 24.5, C["data"]), ("warm up", 120, C["compute"])]
    fig, ax = plt.subplots(figsize=(5.2, 1.7))
    x = 0
    for lab, s, col in parts:
        ax.barh(0, s, left=x, color=col, height=0.55, edgecolor="white", lw=1.5)
        top = lab if s > 30 else ""
        ax.text(x + s / 2, 0.52, top, ha="center", va="bottom", fontsize=12, color=col,
                fontweight="bold")
        if s > 30:
            ax.text(x + s / 2, 0, f"{s:g} s", ha="center", va="center", fontsize=11, color="white",
                    fontweight="bold")
        else:
            ax.text(x + s / 2, -0.5, "load 24.5 s", ha="center", va="top", fontsize=11, color=col,
                    fontweight="bold")
        x += s
    ax.text(x + 8, 0, "6.4 min", va="center", fontsize=14, fontweight="bold", color=C["text"])
    ax.set_xlim(0, 450)
    ax.set_ylim(-0.95, 1.0)
    hide_axis(ax)
    sp.save(fig, out / "recall_budget.svg")


# ===================================================================== Kalos
def kalos():
    rows = load_rows("acme_kalos_job_sizes.csv")
    for r in rows:
        for k in r:
            r[k] = float(r[k])
    J = sum(r["jobs"] for r in rows)
    T = sum(r["gpu_seconds"] for r in rows)
    classes = [("1 GPU", lambda n: n == 1), ("2 to 128 GPUs", lambda n: 1 < n < 256),
               ("256 or more", lambda n: n >= 256)]
    stats = []
    for lab, f in classes:
        j = sum(r["jobs"] for r in rows if f(r["gpu_num"]))
        t = sum(r["gpu_seconds"] for r in rows if f(r["gpu_num"]))
        stats.append((lab, j / J, t / T, t / j / 3600))
    # Lower bound on the coefficient of variation of GPU-time per job, from the
    # per-gpu_num row means (within-row variance ignored).
    m1 = T / J
    m2 = sum(r["jobs"] * (r["gpu_seconds"] / r["jobs"]) ** 2 for r in rows) / J
    cs = math.sqrt(m2 - m1 * m1) / m1
    states = {s: sum(r["gpu_seconds_" + s] for r in rows) / T for s in ("completed", "failed", "canceled")}
    print("Kalos jobs", int(J), "classes", [(s[0], round(s[1], 4), round(s[2], 4), round(s[3], 2)) for s in stats])
    print("Kalos Cs lower bound", round(cs, 2), "states", {k: round(v, 3) for k, v in states.items()})

    # --- share of jobs vs share of GPU time (two steps)
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        n = len(stats)
        for i, (lab, js, ts, _) in enumerate(stats):
            y = n - 1 - i
            ax.barh(y + 0.19, js * 100, height=0.36, color=C["neutral"])
            ax.text(js * 100 + 1.2, y + 0.19, f"{js*100:.0f}% of jobs", va="center", fontsize=12,
                    color=C["neutral"], fontweight="bold")
            ax.barh(y - 0.19, ts * 100 if k >= 2 else 0, height=0.36, color=C["danger"])
            tl = f"{ts*100:.1f}% of GPU time" if ts < 0.01 else f"{ts*100:.0f}% of GPU time"
            ax.text(max(ts * 100, 0) + 1.2, y - 0.19, tl, va="center", fontsize=12,
                    color=C["danger"], fontweight="bold", alpha=1.0 if k >= 2 else 0.0)
            ax.text(-1.5, y, lab, ha="right", va="center", fontsize=13, fontweight="bold",
                    color=C["text"])
        ax.set_xlim(0, 125)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Percent, 19,907 GPU jobs, Kalos cluster, March to August 2023")
        sp.save(fig, out / f"kalos_split_step{k}.svg")

    # --- mean GPU-hours per job, log scale
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    cols = [C["neutral"], C["memory"], C["danger"]]
    for i, ((lab, _, _, h), col) in enumerate(zip(stats, cols)):
        y = 2 - i
        ax.barh(y, h, color=col, height=0.6)
        ax.text(h * 1.25, y, f"{h:,.2f} GPU-hours" if h < 1 else f"{h:,.0f} GPU-hours" if h > 100 else f"{h:,.1f} GPU-hours",
                va="center", fontsize=13, color=col, fontweight="bold")
        ax.text(-0.01, y, lab, ha="right", va="center", fontsize=13, fontweight="bold",
                color=C["text"], transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlim(0.03, 3e5)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.1", "1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Mean GPU-hours per job (derived from the Kalos trace)")
    sp.save(fig, out / "kalos_hours.svg")

    # --- GPU time by final job state
    parts = [("completed", states["completed"], C["data"]), ("failed", states["failed"], C["danger"]),
             ("canceled", states["canceled"], C["neutral"])]
    other = 1 - sum(p[1] for p in parts)
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    x = 0
    for lab, s, col in parts + [("other", other, C["neutral_l"])]:
        ax.barh(0, s * 100, left=x, color=col, height=0.6, edgecolor="white", lw=1.5)
        if s > 0.1:
            ax.text(x + s * 50, 0, f"{lab}\n{s*100:.0f}%", ha="center", va="center",
                    fontsize=12, color="white", fontweight="bold")
        x += s * 100
    ax.annotate(f"failed {states['failed']*100:.0f}%", (31.3 + 3.1, 0.3), xytext=(28, 0.62),
                fontsize=11, color=C["danger"], fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-", color=C["danger"], lw=1.2))
    ax.annotate(f"other {other*100:.0f}%", (99.2, -0.3), xytext=(88, -0.72), fontsize=11,
                color=C["neutral"], ha="left", arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=1.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of all Kalos GPU time, by the job's final state (derived)")
    sp.save(fig, out / "kalos_states.svg")
    return cs


# ============================================================ control plane
def control_plane(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    stages = [("Admission", "may it enter?", "quota, priority,\ngang complete", V["c3"], V["c3_l"]),
              ("Placement", "where does it run?", "capability, topology,\npower, I/O", V["c1"], V["c1_l"]),
              ("Reconciliation", "did it happen?", "start, retry,\nevict, fence", V["ok"], V["ok_l"])]
    xs = [0.0, 0.345, 0.69]
    w = 0.31
    for i, (name, q, items, ec, fc) in enumerate(stages):
        show = i < k
        a = 1.0 if show else 0.0
        b = box(ax, xs[i], 0.30, w, 0.55, fc, ec, lw=2)
        b.set_alpha(a)
        ax.text(xs[i] + w / 2, 0.75, name, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=ec, alpha=a)
        ax.text(xs[i] + w / 2, 0.63, q, ha="center", va="center", fontsize=11,
                style="italic", color=V["ink"], alpha=a)
        ax.text(xs[i] + w / 2, 0.44, items, ha="center", va="center", fontsize=10.5,
                color=V["ink"], alpha=a, linespacing=1.2)
        if 0 < i and show:
            arrow(ax, xs[i] - 0.033, 0.575, xs[i] - 0.002, 0.575, V["gray"], lw=2)
    if k >= 3:
        ax.add_patch(FancyArrowPatch((0.80, 0.29), (0.55, 0.29), connectionstyle="arc3,rad=-0.5",
                                     arrowstyle="-|>", mutation_scale=14, color=V["ok"], lw=1.6, ls="--"))
        ax.text(0.53, 0.12, "observed state differs:\nact again", ha="right", va="center", fontsize=10.5,
                color=V["ok"])
        box(ax, 0.69, 0.90, w, 0.11, V["bad_l"], V["bad"], lw=1.4)
        ax.text(0.69 + w / 2, 0.955, "partition: fence first", ha="center", va="center", fontsize=10.5,
                color=V["bad"], fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    hide_axis(ax)
    sp.save(fig, out / f"control_step{k}.svg")


# ====================================================================== gang
def gang(k):
    """8 GPU rows against time. k=1: J1 takes 4; k=2: J2 takes 4, both stuck;
    k=3: gang panel beside it."""
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.4), gridspec_kw={"wspace": 0.14})
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.7, 9.4)
        hide_axis(ax)
        for g in range(8):
            ax.text(-0.15, g + 0.5, f"G{7-g}", ha="right", va="center", fontsize=10, color=V["gray"])
    a = axes[0]
    a.text(5, 8.95, "Placed one by one", ha="center", fontsize=13, fontweight="bold", color=V["ink"])
    a.add_patch(Rectangle((0.3, 4.05), 9.4, 3.9, fc=V["bad_l"], ec=V["bad"], lw=1.6, hatch="//"))
    a.text(5, 6.0, "J1 holds 4 of 8", ha="center", va="center", fontsize=12, fontweight="bold",
           color=V["bad"], bbox=dict(fc="white", ec="none", pad=1.5))
    if k >= 2:
        a.add_patch(Rectangle((0.3, 0.05), 9.4, 3.9, fc=V["bad_l"], ec=V["bad"], lw=1.6, hatch="\\\\"))
        a.text(5, 2.0, "J2 holds 4 of 8", ha="center", va="center", fontsize=12, fontweight="bold",
               color=V["bad"], bbox=dict(fc="white", ec="none", pad=1.5))
        a.text(5, -1.0, "8 GPUs held, 0 computing", ha="center", va="center", fontsize=12,
               color=V["bad"], fontweight="bold")
    b = axes[1]
    b.text(5, 8.95, "Gang: all or nothing", ha="center", fontsize=13, fontweight="bold",
           color=V["ink"], alpha=1.0 if k >= 3 else 0.0)
    if k >= 3:
        b.add_patch(Rectangle((0.3, 0.05), 4.6, 7.9, fc=V["c1_l"], ec=V["c1"], lw=1.8))
        b.text(2.6, 4.0, "J1\nall 8", ha="center", va="center", fontsize=13, fontweight="bold",
               color=V["c1"])
        b.add_patch(Rectangle((5.1, 0.05), 4.6, 7.9, fc=V["c1_l"], ec=V["c1"], lw=1.8))
        b.text(7.4, 4.0, "J2\nall 8", ha="center", va="center", fontsize=13, fontweight="bold",
               color=V["c1"])
        b.text(5, -1.0, "J2 holds 0 while it waits", ha="center", va="center", fontsize=12,
               color=V["c1"], fontweight="bold")
    sp.save(fig, out / f"gang_step{k}.svg")


# =================================================================== backfill
def backfill(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    ax.add_patch(Rectangle((0, 900), 3.0, 124, fc=C["neutral_l"], ec=C["neutral"], lw=1.4))
    ax.text(1.5, 962, "running jobs hold 124", ha="center", va="center", fontsize=11, color=C["text"])
    ax.add_patch(Rectangle((3.0, 0), 3.0, 1024, fc=V["c3_l"], ec=V["c3"], lw=1.8, ls="--"))
    ax.text(5.15, 200, "reserved:\n1,024-GPU\ngang", ha="center", va="center", fontsize=12,
            color=V["c3"], fontweight="bold")
    ax.axvline(3.0, color=V["c3"], lw=2)
    ax.text(3.03, 1050, "reservation start", fontsize=11, color=V["c3"], va="bottom")
    ax.add_patch(Rectangle((0, 0), 3.0, 900, fc="#fafafa", ec=C["neutral"], lw=1, ls=":"))
    ax.text(1.5, 845, "900 free: the backfill window", ha="center", va="center", fontsize=11,
            color=C["neutral"])
    ax.add_patch(Rectangle((0.1, 60), 2.0, 280, fc=V["ok_l"], ec=V["ok"], lw=1.8))
    ax.text(1.1, 200, "128 GPUs, 2 h:\nadmitted", ha="center", va="center", fontsize=11,
            color=V["ok"], fontweight="bold")
    if k >= 2:
        ax.add_patch(Rectangle((0.1, 430), 4.2, 280, fc=V["bad_l"], ec=V["bad"], lw=1.8, ls="--"))
        ax.text(1.5, 570, "128 GPUs, 4 h: held,\nit would delay the gang", ha="center",
                va="center", fontsize=11, color=V["bad"], fontweight="bold")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1150)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticks([0, 512, 1024])
    ax.set_yticklabels(["0", "512", "1,024"])
    ax.set_xlabel("Hours from now")
    ax.set_ylabel("GPUs")
    sp.save(fig, out / f"backfill_step{k}.svg")


# ======================================================================== DRF
def drf(k):
    res = ["GPU", "CPU", "Memory", "Network"]
    A = [0.50, 0.50, 0.50, 0.50]
    B = [0.50, 0.125, 0.125, 0.50]
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    xs = np.arange(4)
    ax.bar(xs - 0.2, A, width=0.38, color=C["memory"])
    ax.bar(xs + 0.2, B if k >= 2 else [0] * 4, width=0.38, color=C["compute"])
    for x, a, b in zip(xs, A, B):
        ax.text(x - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=11, color=C["memory"], fontweight="bold")
        ax.text(x + 0.2, b + 0.02, "0.50" if b == 0.5 else f"{b:.3f}", ha="center", fontsize=11,
                color=C["compute"], fontweight="bold", alpha=1.0 if k >= 2 else 0.0)
    ax.axhline(0.5, color=C["text"], lw=1.2, ls="--")
    ax.text(4.45, 0.66, "dashed line:\ndominant share 0.50", fontsize=11, color=C["text"], va="bottom", ha="right")
    ax.text(-0.45, 0.70, "A: 4 replicas", fontsize=12, color=C["memory"], fontweight="bold")
    ax.text(1.2, 0.70, "B: 2 workers", fontsize=12, color=C["compute"], fontweight="bold",
            alpha=1.0 if k >= 2 else 0.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(res, fontsize=12)
    ax.set_xlim(-0.6, 4.5)
    ax.set_ylim(0, 0.88)
    ax.set_yticks([0, 0.25, 0.5])
    ax.set_ylabel("Share of the fleet")
    sp.save(fig, out / f"drf_step{k}.svg")


# =============================================================== fragmentation
def frag():
    # illustrative layout: free slots per node [2,3,2,3,2,3,2,2] = 19
    free = {0: [6, 7], 1: [2, 5, 7], 2: [0, 7], 3: [3, 4, 7], 4: [5, 6], 5: [1, 6, 7], 6: [3, 7], 7: [4, 5]}
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    for n in range(8):
        y = 7 - n
        for s in range(8):
            if s in free[n]:
                ax.add_patch(Rectangle((s + 0.06, y + 0.06), 0.88, 0.88, fc="white", ec=V["gray"],
                                       lw=1.2, ls="--"))
            else:
                ax.add_patch(Rectangle((s + 0.06, y + 0.06), 0.88, 0.88, fc=V["c1_l"], ec=V["c1"], lw=1.0))
        ax.text(-0.15, y + 0.5, f"node {n}", ha="right", va="center", fontsize=11, color=V["ink"])
        ax.text(8.15, y + 0.5, f"{len(free[n])} free", ha="left", va="center", fontsize=11, color=V["gray"])
    ax.add_patch(Rectangle((0.06, 8.25), 7.88, 0.8, fc=V["bad_l"], ec=V["bad"], lw=1.6, ls="--"))
    ax.text(4, 8.65, "pending: 8 GPUs, one node", ha="center", va="center", fontsize=10.5,
            color=V["bad"], fontweight="bold")
    ax.text(4, -0.45, "19 of 64 free (30%)", ha="center", va="center", fontsize=13,
            color=V["ink"], fontweight="bold")
    ax.set_xlim(-1.6, 9.4)
    ax.set_ylim(-0.8, 9.2)
    hide_axis(ax)
    sp.save(fig, out / "frag_grid.svg")


# ====================================================================== M/G/1
def mg1(k, cs_kalos, path=None, sat=False):
    rho = np.linspace(0.05, 0.985 if sat else 0.95, 400)
    load = rho / (1 - rho)
    fig, ax = plt.subplots(figsize=(5.0, 2.7) if not sat else (4.6, 2.9))
    curves = [(1.0, C["memory"], "$C_s$ = 1"), (3.0, C["compute"], "$C_s$ = 3"),
              (cs_kalos, C["danger"], f"Kalos, $C_s \\geq {cs_kalos:.1f}$\n(derived)")]
    shown = curves[:1] if sat else curves[:k]
    for cs, col, lab in shown:
        f = (1 + cs * cs) / 2
        ax.plot(rho, load * f, color=col, lw=2.6)
        if not sat:
            v = 4 * f
            ax.plot([0.8], [v], "o", ms=8, color=col, mec="white", zorder=5)
            two = "\n" in lab   # the two-line Kalos label grows upward from its curve
            ax.text(0.955, (0.95 / 0.05) * f, f" {lab}: {v:.0f}$\\times$",
                    ha="left", va="bottom" if two else "center",
                    fontsize=13, color=col, fontweight="bold")
    if sat:
        ax.plot([0.98], [49], "o", ms=9, color=C["danger"], mec="white", zorder=5)
        ax.text(0.965, 49, "98%: 49$\\times$ ", ha="right", va="center", fontsize=12,
                color=C["danger"], fontweight="bold")
        ax.plot([0.6], [1.5], "o", ms=8, color=C["memory"], mec="white", zorder=5)
        ax.text(0.6, 3.0, "60%: 1.5$\\times$", ha="center", va="bottom", fontsize=12,
                color=C["memory"], fontweight="bold")
        ax.set_ylim(0.05, 120)
        ax.set_xlim(0, 1.0)
    else:
        ax.axvline(0.8, color=C["grid"], lw=1, ls="--")
        ax.text(0.81, 0.07, "$\\rho$ = 0.8: values at the right", fontsize=12, color=C["neutral"])
        ax.set_xlim(0, 1.45)
        ax.set_ylim(0.05, 3000)
    ax.set_yscale("log")
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.minorticks_off()
    ax.set_xlabel("Offered load $\\rho$ (M/G/1 model)")
    ax.set_ylabel("Mean wait / mean job")
    sp.save(fig, path or out / f"mg1_step{k}.svg")


# ====================================================================== NCCL
def nccl():
    rows = load_rows("nccl_allreduce_h100_node.csv")
    sz = np.array([float(r["size_bytes"]) for r in rows])
    bw = np.array([float(r["ip_busbw_GBps"]) for r in rows])
    keep = sz >= 2 ** 16
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(sz[keep], bw[keep], "-o", color=C["memory"], lw=2.4, ms=4.5)
    i = int(np.where(sz == 2 ** 30)[0][0])
    ax.plot([sz[i]], [bw[i]], "o", ms=9, color=C["memory"], mec="white", zorder=5)
    ax.text(sz[i] / 1.6, bw[i] + 18, f"1 GiB: {bw[i]:.0f} GB/s", ha="right", fontsize=12,
            color=C["memory"], fontweight="bold")
    ax.axhline(50, color=C["compute"], lw=2.2, ls="--")
    ax.text(2 ** 31, 62, "one NDR InfiniBand\nport: 50 GB/s (spec)", fontsize=11.5, ha="right",
            va="bottom", color=C["compute"], fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2 ** 20, 2 ** 25, 2 ** 30])
    ax.set_xticklabels(["1 MiB", "32 MiB", "1 GiB"])
    ax.minorticks_off()
    ax.set_ylim(0, 540)
    ax.set_xlabel("All-reduce message size")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    sp.save(fig, out / "nccl_node.svg")


# ================================================================= placement
def placement(k):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    W = 0.29
    racks = [0.0, 0.33]
    spx = 0.31
    box(ax, spx - 0.08, 0.87, 0.16, 0.10, V["c2_l"], V["c2"], lw=1.6)
    ax.text(spx, 0.925, "spine", ha="center", va="center", fontsize=13, color=V["c2"], fontweight="bold")
    for r, rx in enumerate(racks):
        box(ax, rx, 0.105, W, 0.635, V["frame_l"], V["frame"], lw=1.2)
        pass  # rack outlines carry no label; the slide note names them
        box(ax, rx + W / 2 - 0.055, 0.595, 0.11, 0.08, V["c2_l"], V["c2"], lw=1.4)
        ax.text(rx + W / 2, 0.635, "leaf", ha="center", va="center", fontsize=12, color=V["c2"])
        ax.plot([rx + W / 2, spx], [0.67, 0.88], color=V["c2"], lw=1.5)
        for n in range(4):
            nx = rx + 0.010 + n * 0.069
            box(ax, nx, 0.13, 0.062, 0.40, V["c1_l"], V["c1"], lw=1.2)
            for g in range(8):
                ax.add_patch(Rectangle((nx + 0.009 + (g % 2) * 0.024, 0.155 + (g // 2) * 0.090), 0.020, 0.07,
                                       fc=V["c1"], ec="none", zorder=4))
            ax.plot([nx + 0.031, rx + W / 2], [0.53, 0.60], color=V["c2"], lw=1.0)
    if k == 1:
        ax.add_patch(Rectangle((0.004, 0.12), 0.074, 0.42, fc="none", ec=V["ok"], lw=2.6, zorder=5))
        ax.text(0.30, 0.09, "TP group inside\none node", ha="center", va="top", fontsize=12,
                color=V["ok"], fontweight="bold")
    if k >= 2:
        ax.add_patch(Rectangle((0.145, 0.14), 0.130, 0.19, fc="none", ec=V["bad"], lw=2.6, ls="--", zorder=5))
        ax.text(0.30, 0.09, "TP group split\nover two nodes", ha="center", va="top", fontsize=12,
                color=V["bad"], fontweight="bold")
    ax.text(0.83, 0.93, "1 GB exchange", ha="center", va="center", fontsize=12, color=V["ink"],
            fontweight="bold")
    bars = [("NVLink\n450 GB/s", 2.2, V["ok"], 0.745), ("InfiniBand\n50 GB/s", 20.0, V["bad"], 0.92)]
    for i, (lab, ms, col, x0) in enumerate(bars):
        a = 1.0 if (i == 0 or k >= 2) else 0.0
        h = ms / 20.0 * 0.58
        ax.add_patch(Rectangle((x0 - 0.035, 0.20), 0.07, h, fc=col, ec="none", alpha=a))
        ax.text(x0, 0.20 + h + 0.02, f"{ms:g} ms", ha="center", fontsize=12, color=col,
                fontweight="bold", alpha=a)
        ax.text(x0, 0.17, lab, ha="center", va="top", fontsize=11, color=V["ink"], linespacing=1.0,
                alpha=a)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 1)
    hide_axis(ax)
    sp.save(fig, out / f"place_step{k}.svg")


# =================================================================== elastic
def elastic(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    lanes = [("Start small", [(0, 24, 128)], "24 epochs", "3,072 GPU-h"),
             ("Wait for 512", [(4, 24, 512)], "64 epochs", "10,240 GPU-h"),
             ("Step up to 256", [(0, 4, 128), (4.17, 24, 256)], "39.7 epochs", "5,632 GPU-h")]
    scale = 0.8 / 512
    for i, (lab, segs, ep, gh) in enumerate(lanes):
        y0 = 2 - i
        ax.text(-0.4, y0 + 0.35, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=V["ink"])
        if lab == "Wait for 512":
            ax.add_patch(Rectangle((0, y0 + 0.02), 4, 0.12, fc="white", ec=V["gray"], lw=1, hatch="///"))
            ax.text(0.2, y0 + 0.22, "queued", ha="left", fontsize=11, color=V["gray"])
        for (a, b, g) in segs:
            ax.add_patch(Rectangle((a, y0 + 0.02), b - a, g * scale, fc=V["c1_l"], ec=V["c1"], lw=1.4))
            ax.text((a + b) / 2 if g == 512 else a + 0.25, y0 + 0.02 + g * scale / 2, f"{g}", va="center", fontsize=11, color=V["c1"],
                    fontweight="bold")
        if lab == "Step up to 256":
            ax.plot([4.08, 4.08], [y0, y0 + 0.45], color=V["bad"], lw=2)
            ax.text(4.25, y0 + 0.62, "10-min resize", fontsize=10.5, color=V["bad"])
        a = 1.0 if k >= 2 else 0.0
        ax.text(24.6, y0 + 0.42, ep, fontsize=12.5, color=V["c1"], fontweight="bold", va="center", alpha=a)
        ax.text(24.6, y0 + 0.12, gh, fontsize=11, color=V["gray"], va="center", alpha=a)
        if k < 2:
            ax.text(24.6, y0 + 0.3, "?", fontsize=15, color=V["bad"], fontweight="bold", va="center")
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.1, 3.0)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Hours (bar height: GPUs held)")
    sp.save(fig, out / f"elastic_step{k}.svg")


# ==================================================================== MLPerf
def mlperf():
    rows = [r for r in load_rows("mlperf_training_llm_scaling.csv")
            if r["round"] == "v3.1" and r["benchmark"] == "gpt3" and r["submitter"] == "NVIDIA"]
    pts = sorted((int(r["accelerators"]), float(r["score_minutes"])) for r in rows)
    print("MLPerf v3.1 Eos GPT-3", pts)
    g = np.array([p[0] for p in pts])
    m = np.array([p[1] for p in pts])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ideal = m[0] * g[0] / g
    ax.plot(g, ideal, ls="--", color=C["neutral"], lw=1.6)
    ax.text(g[-1] * 0.92, ideal[-1] * 0.78, "ideal", ha="right", va="top", fontsize=11, color=C["neutral"])
    ax.plot(g, m, "-o", color=C["danger"], lw=2.4, ms=7, mec="white")
    for gg, mm in zip(g, m):
        if gg in (512, 4096, 10752):
            ax.text(gg * 1.12, mm * 1.12, f"{mm:.1f} min" if mm < 10 else f"{mm:.0f} min",
                    fontsize=11.5, color=C["danger"], fontweight="bold", va="bottom")
    sp_ = m[0] / m[-1]
    ax.text(600, 1.9, f"{g[-1]/g[0]:.0f}$\\times$ the GPUs, {sp_:.1f}$\\times$ faster",
            fontsize=12, color=C["text"], fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1,024", "2,048", "4,096", "8,192"])
    ax.set_yticks([2, 5, 10, 20, 50])
    ax.set_yticklabels(["2", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_ylim(1.5, 90)
    ax.set_xlim(400, 16000)
    ax.set_xlabel("H100 GPUs (NVIDIA Eos)")
    ax.set_ylabel("Minutes to train")
    sp.save(fig, out / "mlperf_gpt3.svg")


# ================================================================ preemption
def preempt():
    parts = [("lost work", 15.0, C["danger"]), ("relaunch", 3.0, V["c3"]),
             ("reload", 24.5 / 60, C["data"]), ("warm up", 2.0, C["compute"])]
    fig, ax = plt.subplots(figsize=(4.4, 1.9))
    x = 0
    for lab, mins, col in parts:
        ax.barh(1, mins, left=x, color=col, height=0.6, edgecolor="white", lw=1.5)
        x += mins
    ax.text(7.5, 1, "15 min lost", ha="center", va="center",
            fontsize=11.5, color="white", fontweight="bold")
    ax.text(16.5, 1.42, "relaunch 3 min", ha="center", fontsize=11.5, color=V["c3"])
    ax.text(20.4, 0.55, "reload 24.5 s +\nwarm up 2 min", ha="right", va="top", fontsize=11,
            color=C["compute"])
    ax.text(x + 0.4, 1, "20.4 min\n= \\$44", va="center", fontsize=13, fontweight="bold", color=C["text"])
    ax.set_xlim(0, 25.5)
    ax.set_ylim(0.0, 1.75)
    hide_axis(ax)
    sp.save(fig, out / "preempt.svg")


def cascade():
    fig, ax = plt.subplots(figsize=(1.9, 1.8))
    root = (0.5, 0.88)
    lvl1 = [(0.25, 0.58), (0.75, 0.58)]
    lvl2 = [(0.10, 0.12), (0.36, 0.12), (0.64, 0.12), (0.90, 0.12)]
    for p in lvl1:
        ax.plot([root[0], p[0]], [root[1], p[1]], color=V["gray"], lw=1.4)
    for i, p in enumerate(lvl2):
        q = lvl1[i // 2]
        ax.plot([q[0], p[0]], [q[1], p[1]], color=V["gray"], lw=1.4)
    ax.add_patch(Rectangle((root[0] - 0.06, root[1] - 0.06), 0.12, 0.12, fc=V["bad"], ec="none", zorder=4))
    for p in lvl1:
        ax.add_patch(plt.Circle(p, 0.06, fc=V["c3_l"], ec=V["c3"], lw=1.5, zorder=4))
    for p in lvl2:
        ax.add_patch(plt.Circle(p, 0.06, fc=V["c1_l"], ec=V["c1"], lw=1.5, zorder=4))
    ax.text(0.98, 0.58, "k", fontsize=13, color=V["c3"], fontweight="bold", va="center", ha="right")
    ax.text(0.5, -0.04, "k$^2$", fontsize=13, color=V["c1"], fontweight="bold", ha="center", va="top")
    ax.plot([0.0, 1.0], [0.40, 0.40], color=V["bad"], lw=1.5, ls="--")
    ax.text(0.5, 0.40, "requeue cap", fontsize=10.5, color=V["bad"], ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.17, 1.0)
    ax.set_aspect("equal")
    hide_axis(ax)
    sp.save(fig, out / "cascade.svg")


# ====================================================================== spot
def spot(k):
    rows = [("On-demand", 344064, 336, C["neutral"]), ("Spot", 161567, 450.8, C["data"])]
    fig, ax = plt.subplots(figsize=(4.6, 1.3))
    for i, (lab, cost, hrs, col) in enumerate(rows):
        y = 1 - i
        shown = k >= 2 or i == 0
        ax.barh(y, cost if shown else 0, color=col, height=0.6)
        ax.text(cost + 6000, y, f"\\${cost:,.0f} billed, {hrs:g} h", va="center", fontsize=12,
                color=col, fontweight="bold", alpha=1.0 if shown else 0.0)
        ax.text(-6000, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 600000)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    sp.save(fig, out / f"spot_step{k}.svg")


# =================================================================== sharing
def share():
    fig, ax = plt.subplots(figsize=(4.6, 1.5))
    rows = [("One 7B model", 26, C["memory"]), ("Two 7B models", 52, C["memory"])]
    for i, (lab, used, col) in enumerate(rows):
        y = 1 - i
        ax.barh(y, used, color=col, height=0.55)
        ax.barh(y, 80 - used, left=used, color=C["neutral_l"], height=0.55)
        ax.text(used / 2, y, f"{used} GB", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
        ax.text(used + (80 - used) / 2, y, f"{(80-used)/80*100:.1f}% unused", ha="center", va="center",
                fontsize=11.5, color=C["text"])
        ax.text(-1.5, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("GB of one A100's 80 GB budget")
    sp.save(fig, out / "share_mem.svg")


# =================================================================== diurnal
def diurnal():
    t = np.linspace(0, 24, 400)
    demand = 150 + 133 * np.cos((t - 14) / 24 * 2 * np.pi)   # avg 150, peak 283 at 14:00
    buf = 117
    reclaimed = np.clip(400 - demand - buf, 0, None)
    print("diurnal mean demand", round(demand.mean(), 1), "mean reclaimed", round(reclaimed.mean(), 1),
          "peak", round(demand.max(), 1))
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.fill_between(t, 0, demand, color=C["data"], lw=0)
    ax.fill_between(t, demand, demand + buf, color=C["data_l"], lw=0, hatch="..", edgecolor=C["data"])
    ax.fill_between(t, demand + buf, demand + buf + reclaimed, color=C["memory"], alpha=0.75, lw=0)
    ax.axhline(400, color=C["text"], lw=1.4, ls="--")
    ax.text(0.3, 410, "static serving reservation: 400 GPUs", fontsize=11, color=C["text"], va="bottom")
    ax.text(14, 95, "serving demand\n(peak 283)", ha="center", va="center", fontsize=11, color="white",
            fontweight="bold")
    ax.text(14, 345, "surge buffer 117", ha="center", va="center", fontsize=11, color=C["data"])
    ax.text(3.0, 310, "lent to\ntraining", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 470)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["0:00", "6:00", "12:00", "18:00", "24:00"])
    ax.set_yticks([0, 200, 400])
    ax.set_ylabel("GPUs")
    ax.set_xlabel("Hour of day (illustrative shape; averages from Napkin Math 8.5)")
    sp.save(fig, out / "diurnal.svg")


# ===================================================================== quota
def quota():
    fig, ax = plt.subplots(figsize=(3.9, 1.9))
    ax.barh(1, 300, color=C["memory"], height=0.55)
    ax.text(150, 1, "uses 300", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.barh(1, 200, left=300, color=C["memory_l"], height=0.55, edgecolor=C["memory"], lw=1.2, ls="--")
    ax.text(400, 1, "200 idle", ha="center", va="center", fontsize=12, color=C["memory"])
    ax.text(-8, 1, "A: owns 500", ha="right", va="center", fontsize=12, fontweight="bold")
    ax.barh(0, 300, color=C["compute"], height=0.55)
    ax.text(150, 0, "owns 300", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.barh(0, 200, left=300, color=C["compute_l"], height=0.55, edgecolor=C["compute"], lw=1.2, hatch="//")
    ax.text(400, 0, "borrows 200", ha="center", va="center", fontsize=12, color=C["compute"], fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1.0))
    ax.text(-8, 0, "B: owns 300", ha="right", va="center", fontsize=12, fontweight="bold")
    ax.annotate("", (400, 0.33), (400, 0.70), arrowprops=dict(arrowstyle="-|>", color=C["text"], lw=1.4))
    ax.annotate("reclaimed when\nA needs them", (400, 0.5), xytext=(515, 0.5), va="center", fontsize=11,
                color=C["text"], ha="left",
                arrowprops=dict(arrowstyle="-", color="white", lw=0))
    ax.set_xlim(0, 760)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xticks([0, 250, 500])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("GPUs")
    sp.save(fig, out / "quota.svg")


# ===================================================================== tiers
def tiers(k):
    rows = [("Allocated", 500, C["neutral"], "held"),
            ("Active", 350, C["compute"], "busy (70%)"),
            ("Productive", 280, C["data"], "useful (56%)")]
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    for i, (lab, v, col, note) in enumerate(rows):
        y = 2 - i
        shown = i < k
        ax.barh(y, v if shown else 0, color=col, height=0.6)
        ax.text(v - 8, y, f"{v}", ha="right", va="center", fontsize=13, color="white", fontweight="bold",
                alpha=1.0 if shown else 0.0)
        ax.text(v + 8, y, note, va="center", fontsize=12, color=col, alpha=1.0 if shown else 0.0)
        ax.text(-8, y, lab, ha="right", va="center", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 680)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.set_xticks([0, 250, 500])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("GPU-equivalents, time-averaged")
    sp.save(fig, out / f"tiers_step{k}.svg")


if __name__ == "__main__":
    hook_pools()
    for k in (1, 2):
        hook_answer(out / f"answer_step{k}.svg", k)
    recall_budget()
    cs = kalos()
    for k in (1, 2, 3):
        control_plane(k)
        gang(k)
        mg1(k, cs)
        tiers(k)
    for k in (1, 2):
        drf(k)
        backfill(k)
        placement(k)
        elastic(k)
        spot(k)
    mg1(1, cs, path=out / "mg1_sat.svg", sat=True)
    frag()
    nccl()
    mlperf()
    preempt()
    cascade()
    share()
    diurnal()
    quota()
    print("figures written")
