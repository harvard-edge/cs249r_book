#!/usr/bin/env python3
"""Figures for Lecture 12b (Benchmarking, Part 2: Deciding from Measurements).

Every drawn value comes from MLSysIM through calcs.py, which this script
imports: the MLPerf Training v4.0 ResNet-50 runs and the Scenarios.* built on
them (scores, Welch intervals, run counts, the paired accuracy comparison),
the sample-size, run-order, beam, calibration, and acceptance-gate scenarios,
and the Literature entries and series of published results (AnandTech 2013;
Recht et al. 2019; Zhang et al. 2024; Sandler et al. 2018; Chu et al. 2021;
Guo et al. 2017; Koh et al. 2021). Labels print the values calcs.py records
(deckvals.val), so a figure and its slide always show the same digits. Values
the chapter gives as a scenario or an assumption, and values built for a
slide, carry an "Illustrative" or "Assumed" tag on the plot. Axis limits,
ticks, and label positions are layout, not data.

Run from this folder with the MLSysBook venv (needs matplotlib and scipy):
  python3 make_figs.py
"""

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import calcs as K  # noqa: E402
from calcs import EOS, DELL, H200, SYSTEMS, ALL_RUNS, W_DELL, W_DELL_ALL, W_H200, pct  # noqa: E402
from deckvals import val, num  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent
mag = K.m
TARGET_PCT = pct(K.TARGET)


def tag(ax, text, where=(0.0, -0.32), size=9):
    ax.text(where[0], where[1], text, transform=ax.transAxes, fontsize=size,
            color=C["neutral"], ha="left")


def interval(w):
    """A Welch comparison as plain minutes: difference, SE, low, high."""
    return dict(d=mag(w.difference, "minute"), se=mag(w.standard_error, "minute"),
                lo=mag(w.low, "minute"), hi=mag(w.high, "minute"))


# =============================================================================
# Hook: two published scores, then the ten runs behind them
# =============================================================================
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 2.25),
                                 gridspec_kw=dict(height_ratios=[1.0, 1.2], hspace=1.0))
    rows = [("Dell XE9680, 8x H100", mag(DELL.published_score, "minute"), "compute", val("dell_score")),
            ("NVIDIA Eos, 8x H100", mag(EOS.published_score, "minute"), "memory", val("eos_score"))]
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 1 - i
        a1.barh(y, v, color=C[ck], height=0.62)
        a1.text(v + 0.15, y, f"{txt} min", va="center", fontsize=12, fontweight="bold", color=C[ck])
        a1.text(-0.2, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=C["text"])
    a1.set_xlim(0, 16)
    a1.set_ylim(-0.6, 1.6)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlabel(f"Published score: minutes to {val('target_top1')}% top-1")
    al = 1.0 if k >= 2 else 0.0
    for i, (s, ck, lab) in enumerate(((DELL, "compute", "Dell runs"), (EOS, "memory", "Eos runs"))):
        y = 1 - i
        ok_sorted = sorted(s.run_minutes)
        for r in s.rows:
            x = r["minutes"]
            if r["status"] == "success":
                # Alternate neighbors up and down so close runs stay visible.
                jit = 0.2 if ok_sorted.index(x) % 2 == 0 else -0.2
                a2.plot(x, y + jit, "o", ms=8, color=C[ck], mec="white", mew=1.0, alpha=al, zorder=4)
            else:
                a2.plot(x, y, "o", ms=11, mfc="white", mec=C["danger"], mew=2, alpha=al, zorder=4)
                a2.plot(x, y, "x", ms=7, color=C["danger"], mew=2, alpha=al, zorder=5)
                a2.text(x - 0.03, y - 0.48, f"missed {val('target_top1')}%\n({val('dell_fail_top1')}% at epoch {val('epochs')})", ha="right",
                        va="center", fontsize=9.5, color=C["danger"], fontweight="bold", alpha=al)
        a2.text(13.07, y, lab, ha="right", va="center", fontsize=12, fontweight="bold",
                color=C["text"], alpha=al)
    a2.set_xlim(13.1, 14.0)
    a2.set_ylim(-0.95, 1.5)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xticks([13.2, 13.4, 13.6, 13.8, 14.0])
    a2.set_xlabel("Each run's time to train (minutes)", alpha=al)
    for s_ in a2.spines.values():
        s_.set_alpha(al)
    a2.tick_params(colors=C["text"] if k >= 2 else "white")
    sp.save(fig, out / f"hook_step{k}.svg")

# =============================================================================
# Recall thumbnail: Lecture 12's hook (illustrative, Sec. 12.7)
# =============================================================================
# Drawn for its column slot (about 2.4 by 1.2 in), with type to match.
fig, ax = plt.subplots(figsize=(2.5, 1.45))
for y, lab, v, ck, txt in ((1, f"Datasheet:\n{val('recall_tops')} TOPS at {val('recall_w')} W", num("recall_eff"), "compute",
                            f"{val('recall_eff')} TOPS/W"),
                           (0, f"Sustained:\n{val('recall_sus_tops')} TOPS at {val('recall_sus_w')} W", num("recall_sus_eff"),
                            "danger", f"{val('recall_sus_eff')} TOPS/W")):
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v + 0.6, y, txt, va="center", fontsize=14, fontweight="bold", color=C[ck])
    ax.text(-0.6, y, lab, ha="right", va="center", fontsize=14, fontweight="bold",
            linespacing=1.0)
ax.set_xlim(0, 30)
ax.set_xticks([0, 10, 20])
ax.tick_params(labelsize=14)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("TOPS per watt", fontsize=14)
# "Illustrative" is on the slide's credit line, which this slot leaves room for.
sp.save(fig, out / "recall_hook.svg")

# =============================================================================
# Noise: 20 runs, each relative to its own system's median converged run
# =============================================================================
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    for i, (s, lab, ck) in enumerate(SYSTEMS):
        y = 3 - i
        med = s.median_minutes
        ax.text(-1.25, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
        for r in s.rows:
            x = 100 * (r["minutes"] / med - 1)
            if r["status"] == "success":
                ax.plot(x, y, "o", ms=10, color=C[ck], mec="white", mew=1.2, zorder=4)
            else:
                ax.plot(1.55, y, "o", ms=11, mfc="white", mec=C["danger"], mew=2, zorder=4)
                ax.plot(1.55, y, "x", ms=7, color=C["danger"], mew=2, zorder=5)
                ax.text(1.7, y, f"+{x:.1f}%, missed\ntarget", va="center", fontsize=10.5,
                        color=C["danger"], fontweight="bold")
        ax.text(2.75, y, f"CV {pct(s.cv):.2f}%", va="center", fontsize=12, fontweight="bold",
                color=C[ck], alpha=1.0 if k >= 2 else 0.0)
    ax.axvline(0, color=C["neutral"], lw=1, ls="--", zorder=1)
    ax.axvspan(1.35, 1.75, color="white", zorder=2)
    ax.set_xlim(-1.2, 1.75)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1.0])
    ax.set_xticklabels(["−1%", "−0.5%", "median", "+0.5%", "+1%"])
    ax.set_xlabel("Run time relative to the system's median run")
    sp.save(fig, out / f"strip_step{k}.svg")

# =============================================================================
# Seeds move accuracy: top-1 at epoch 35, 20 runs
# =============================================================================
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.05))
    offsets = {s.system: 0.18 * (i - 1.5) for i, (s, _, _) in enumerate(SYSTEMS)}
    colors = {s.system: ck for s, _, ck in SYSTEMS}
    for r in ALL_RUNS:
        a = r["top1"]
        y = offsets[r["system"]]
        if a < TARGET_PCT:
            ax.plot(a, y, "o", ms=11, mfc="white", mec=C["danger"], mew=2, zorder=4)
            ax.plot(a, y, "x", ms=7, color=C["danger"], mew=2, zorder=5)
        else:
            ax.plot(a, y, "o", ms=10, color=C[colors[r["system"]]], mec="white", mew=1.2, zorder=4)
    ax.axvline(TARGET_PCT, color=C["danger"], lw=2, ls="--")
    ax.text(TARGET_PCT - 0.01, -1.15, f"target\n{val('target_top1')}%", ha="right", va="center", fontsize=11, color=C["danger"], fontweight="bold")
    for s, lab, ck in SYSTEMS:
        ax.text(76.24, offsets[s.system], lab, va="center", fontsize=10, color=C[ck], fontweight="bold")
    al = 1.0 if k >= 2 else 0.0
    lo, hi = min(r["top1"] for r in ALL_RUNS), max(r["top1"] for r in ALL_RUNS)
    ax.annotate("", xy=(lo, -0.55), xytext=(hi, -0.55), alpha=al,
                arrowprops=dict(arrowstyle="|-|", color=C["text"], lw=1.5, alpha=al))
    ax.text((lo + hi) / 2, -0.75, f"seeds: {val('acc_spread')} points", ha="center", va="top",
            fontsize=11, fontweight="bold", color=C["text"], alpha=al)
    mid, hw = pct(K.SAMPLING.rate), pct(K.SAMPLING.half_width)
    ax.annotate("", xy=(mid - hw, -1.25), xytext=(mid + hw, -1.25), alpha=al,
                arrowprops=dict(arrowstyle="|-|", color=C["memory"], lw=1.5, alpha=al))
    ax.text(mid + hw + 0.02, -1.25, f"new {val('val_images')}-image\ntest set: ±{val('binom_hw')}",
            va="center", fontsize=11, fontweight="bold", color=C["memory"], alpha=al)
    ax.set_xlim(75.65, 76.6)
    ax.set_ylim(-1.6, 0.55)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"ImageNet top-1 at the last evaluation (%), {val('n_runs')} runs")
    sp.save(fig, out / f"seedacc_step{k}.svg")

# =============================================================================
# The run rule on Dell's five runs
# =============================================================================
dell = sorted(DELL.rows, key=lambda r: r["minutes"])
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.7))
    for j, r in enumerate(dell):
        x = r["minutes"]
        y = 4 - j
        ok = r["status"] == "success"
        dropped = (j == 0 or not ok)
        col = C["neutral"] if (k >= 2 and dropped) else C["compute"]
        xp = x if ok else 13.181
        ax.text(13.1365, y, f"run {j + 1}" if ok else "run 5", ha="right", va="center",
                fontsize=11, fontweight="bold", color=C["text"])
        if ok:
            ax.plot(xp, y, "o", ms=12, color=col, mec="white", mew=1.2, zorder=4)
            left = j > 0 and x < DELL.score_minutes
            ax.text(xp - 0.0016 if left else xp + 0.0016, y + 0.05, f"{x:.3f}", va="center",
                    ha="right" if left else "left", fontsize=11, color=col, fontweight="bold")
        else:
            ax.plot(xp, y, "o", ms=12, mfc="white", mec=C["danger"], mew=2, zorder=4)
            ax.text(xp - 0.0015, y, f"{x:.3f}, missed {val('target_top1')}%, off scale", ha="right", va="center",
                    fontsize=11, color=C["danger"], fontweight="bold")
        if k >= 2 and dropped:
            ax.plot(xp, y, "x", ms=16, color=C["danger"], mew=3, zorder=6)
            if ok:
                ax.text(xp + 0.0072, y + 0.05, "fastest: dropped", va="center", fontsize=11,
                        color=C["danger"], fontweight="bold")
            else:
                ax.text(xp - 0.0015, y - 0.5, "counts as slowest: dropped", ha="right",
                        va="center", fontsize=11, color=C["danger"], fontweight="bold")
    if k >= 3:
        sc = DELL.score_minutes
        ax.plot([sc, sc], [0.6, 3.4], color=C["text"], lw=2.5, zorder=5)
        ax.text(13.1385, 0.0, f"mean of runs 2 to 4:\n{val('dell_middle')} min", ha="left",
                va="center", fontsize=11, fontweight="bold", color=C["text"])
    ax.set_xlim(13.137, 13.1835)
    ax.set_ylim(-0.8, 4.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([13.14, 13.15, 13.16, 13.17])
    ax.set_xlabel("Dell XE9680, five runs sorted (minutes; run 5 off scale)")
    sp.save(fig, out / f"olympic_step{k}.svg")

# =============================================================================
# One formula, three sample sizes (log bars)
# =============================================================================
# Drawn for its wide, short slot (about 5.0 by 0.8 in), with type to match.
fig, ax = plt.subplots(figsize=(7.2, 1.05))
rows = [("accuracy", K.SIZE_ACC.samples, "data", f"{val('n_acc')} images"),
        ("p90", K.SIZE_P90.samples, "memory", f"{val('n_p90')} queries"),
        ("p99", K.SIZE_P99.samples, "danger", f"{val('n_p99')} queries")]
for i, (lab, v, ck, txt) in enumerate(rows):
    y = 2 - i
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v * 1.15, y, txt, va="center", fontsize=16, fontweight="bold", color=C[ck])
    ax.text(700, y, lab, ha="right", va="center", fontsize=16, fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(800, 6e6)
ax.set_xticks([1e3, 1e4, 1e5, 1e6])
ax.set_xticklabels(["1,000", "10,000", "100,000", "1,000,000"])
ax.minorticks_off()
ax.tick_params(labelsize=14)
ax.set_ylim(-0.6, 2.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Samples needed (log scale)", fontsize=15)
sp.save(fig, out / "sizing.svg")

# =============================================================================
# Worked example: the runs, then the interval on the hook's difference, then H200
# Drawn for the right column of the slide. Step 1 shows the nine converged runs
# (raw times) with each mean; steps 2 to 5 build the interval below them.
# =============================================================================
for k in (1, 2, 3, 4, 5):
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(3.45, 2.45),
                                     gridspec_kw=dict(height_ratios=[1.1, 1, 1], hspace=0.95))
    # Top: the converged runs behind the two means
    for y, s, ck, lab in ((1, DELL, "compute", "Dell"), (0, EOS, "memory", "Eos")):
        xs = sorted(s.run_minutes)
        for n, x in enumerate(xs):
            jit = 0.17 if n % 2 == 0 else -0.17
            a0.plot(x, y + jit, "o", ms=7, color=C[ck], mec="white", mew=0.8, zorder=4)
        m = st.mean(xs)
        a0.plot([m, m], [y - 0.4, y + 0.4], color=C["text"], lw=2.5, zorder=5)
        a0.text(13.127, y, lab, ha="right", va="center", fontsize=11, fontweight="bold", color=C[ck])
    a0.set_xlim(13.13, 13.33)
    a0.set_ylim(-0.6, 1.6)
    a0.set_yticks([])
    a0.spines["left"].set_visible(False)
    a0.set_xticks([13.15, 13.20, 13.25, 13.30])
    a0.set_xticklabels(["13.15", "13.20", "13.25", "13.30 min"], fontsize=10)
    # Middle: Eos minus Dell
    w = interval(W_DELL)
    on = 1.0 if k >= 2 else 0.0
    a1.axvline(0, color=C["neutral"], lw=1.5, ls="--", alpha=on)
    a1.plot(w["d"], 0.5, "o", ms=10, color=C["compute"], zorder=5, alpha=on)
    if k >= 3:
        a1.annotate("", xy=(w["d"] - w["se"], 1.15), xytext=(w["d"] + w["se"], 1.15),
                    arrowprops=dict(arrowstyle="|-|", color=C["neutral"], lw=1.5))
        a1.text(w["d"] + w["se"] + 0.008, 1.15, "±1 SE", va="center",
                fontsize=10.5, color=C["neutral"], fontweight="bold")
    if k >= 4:
        a1.plot([w["lo"], w["hi"]], [0.5, 0.5], color=C["compute"], lw=6, solid_capstyle="butt", zorder=4)
        a1.text(w["hi"] + 0.006, 0.5, f"{val('lo_dell')} to {val('hi_dell')}", va="center",
                fontsize=11, fontweight="bold", color=C["compute"])
    a1.text(-0.075, 0.5, "Eos − Dell", ha="right", va="center", fontsize=11,
            fontweight="bold", color=C["compute"], alpha=on)
    a1.set_xlim(-0.07, 0.36)
    a1.set_ylim(-0.2, 1.5)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xticks([0, 0.1, 0.2, 0.3])
    a1.set_xticklabels(["0", "0.1", "0.2", "0.3"], fontsize=10)
    for s_ in a1.spines.values():
        s_.set_alpha(on)
    a1.tick_params(colors=C["text"] if k >= 2 else "white")
    # Bottom: Eos minus H200
    al = 1.0 if k >= 5 else 0.0
    h = interval(W_H200)
    a2.axvline(0, color=C["neutral"], lw=1.5, ls="--", alpha=al)
    a2.plot([h["lo"], h["hi"]], [0.5, 0.5], color=C["data"], lw=6, solid_capstyle="butt", alpha=al)
    a2.plot(h["d"], 0.5, "o", ms=10, color=C["data"], alpha=al, zorder=5)
    a2.text(h["lo"] - 0.03, 0.5, f"{val('lo_h200')} to {val('hi_h200')}", ha="right", va="center",
            fontsize=11, fontweight="bold", color=C["data"], alpha=al)
    a2.text(-0.075 * 1.45 / 0.43, 0.5, "Eos − H200", ha="right", va="center", fontsize=11,
            fontweight="bold", color=C["data"], alpha=al)
    a2.set_xlim(-0.07 * 1.45 / 0.43, 1.45)
    a2.set_ylim(-0.2, 1.5)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xticks([0, 0.5, 1.0])
    a2.set_xticklabels(["0", "0.5", "1.0"], fontsize=10)
    a2.set_xlabel("Difference in mean run time (min)", fontsize=10.5, labelpad=2, alpha=al)
    for s_ in a2.spines.values():
        s_.set_alpha(al)
    a2.tick_params(colors=C["text"] if k >= 5 else "white")
    sp.save(fig, out / f"interval_step{k}.svg")

# =============================================================================
# How many runs: 95 percent half-width of a mean against run count (Eos SD)
# =============================================================================
# Drawn for the right column of the slide (left column: runs_demo.py and its output).
ns = list(range(2, 31))
hw = [pct(K.PLAN.half_width_fraction(n)) for n in ns]
n_lo, _, n_hi = K.PLAN.run_counts
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.plot(ns, hw, color=C["memory"], lw=3)
    for y in reversed([pct(d) for d in K.PLAN.differences]):
        lab = f"{y:g}%"
        ax.axhline(y, color=C["neutral"], lw=1, ls="--")
        ax.text(30, y + 0.04, lab, va="bottom", ha="right", fontsize=11, color=C["neutral"],
                fontweight="bold")
    h5 = pct(K.PLAN.half_width_fraction(n_lo))
    ax.plot(n_lo, h5, "o", ms=10, color=C["danger"], zorder=5)
    ax.text(n_lo + 1.2, h5 + 0.1, f"{n_lo} runs: ±{val('hw_' + str(n_lo))}%", fontsize=11.5,
            fontweight="bold", color=C["danger"])
    al = 1.0 if k >= 2 else 0.0
    h20 = pct(K.PLAN.half_width_fraction(n_hi))
    ax.plot(n_hi, h20, "o", ms=10, color=C["data"], zorder=5, alpha=al)
    ax.text(n_hi, h20 + 0.17, f"{n_hi} runs: ±{val('hw_' + str(n_hi))}%", ha="center", fontsize=11.5,
            fontweight="bold", color=C["data"], alpha=al)
    ax.set_ylim(0, 1.6)
    ax.set_xlim(2, 31)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Runs averaged")
    ax.set_ylabel("95% half-width (%)")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"runs_step{k}.svg")

# =============================================================================
# Napkin Math 12.5: independent interval, then the disagreement threshold
# =============================================================================
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.7, 2.2), gridspec_kw=dict(hspace=1.3))
    a1.plot([0, 0], [-0.35, 0.35], color=C["neutral"], lw=1.5, ls="--")
    lo_pts, hi_pts = (pct(x) for x in K.VOTE.independent_interval)
    d_pts = pct(K.VOTE.accuracy_difference)
    a1.plot([lo_pts, hi_pts], [0, 0], color=C["memory"], lw=4)
    a1.plot([d_pts], [0], "o", color=C["memory"], ms=10)
    a1.text(d_pts, 0.45, f"interval about −{val('vote_lo_abs')} to {val('vote_hi')}", ha="center", fontsize=11,
            fontweight="bold", color=C["memory"])
    a1.text(-0.2, -0.7, "no change", ha="right", fontsize=10, color=C["neutral"])
    a1.set_xlim(-3, 5)
    a1.set_ylim(-1, 1)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlabel("Difference (points), if independent sets")
    al = 1.0 if k >= 2 else 0.0
    gap_n, max_n = K.VOTE.net_gap, K.VOTE.max_discordant
    a2.axvspan(gap_n, max_n, color=C["danger_l"], alpha=al)
    a2.axvline(max_n, color=C["danger"], lw=2, ls="--", alpha=al)
    a2.text((gap_n + max_n) / 2, 0.25, "regression", ha="center", fontsize=11, fontweight="bold",
            color=C["danger"], alpha=al)
    a2.text(43, 0.25, "unresolved", ha="center", fontsize=11, fontweight="bold",
            color=C["neutral"], alpha=al)
    a2.text(max_n + 0.6, -0.6, val("vote_max"), fontsize=11, fontweight="bold", color=C["danger"], alpha=al)
    a2.set_xlim(gap_n, 60)
    a2.set_ylim(-1, 1)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel("Disagreements, b + c")
    for s_ in a2.spines.values():
        s_.set_alpha(al)
    a2.tick_params(colors=C["text"] if k >= 2 else "white")
    a2.xaxis.label.set_alpha(al)
    sp.save(fig, out / f"conf_step{k}.svg")

# McNemar on two real runs: the same 50,000 images, two seeds of one recipe
gap, limit = K.SEEDS.net_gap, K.SEEDS.max_discordant
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.5))
    ax.set_xlim(0, 12000)
    ax.set_ylim(-1, 1)
    al = 1.0 if k >= 2 else 0.0
    ax.axvspan(gap, limit, color=C["memory_l"], alpha=al)
    ax.axvline(limit, color=C["memory"], lw=2, ls="--", alpha=al)
    ax.text((gap + limit) / 2, 0.2, "runs differ", ha="center", fontsize=11,
            fontweight="bold", color=C["memory"], alpha=al)
    ax.text((limit + 12000) / 2, 0.2, "seed noise", ha="center", fontsize=11,
            fontweight="bold", color=C["neutral"], alpha=al)
    ax.text(limit + 120, -0.65, val("seed_max"), fontsize=11, fontweight="bold",
            color=C["memory"], alpha=al)
    ax.plot([gap, gap], [-0.4, 0.4], color=C["text"], lw=2)
    ax.text(gap + 120, -0.65, f"at least {val('seed_gap')}", fontsize=11, fontweight="bold", color=C["text"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 3000, 6000, 9000, 12000])
    ax.set_xticklabels(["0", "3,000", "6,000", "9,000", "12,000"])
    ax.set_xlabel(f"Images on which the two runs disagree, b + c (of {val('val_images')})")
    sp.save(fig, out / f"mcn_runs_step{k}.svg")

# =============================================================================
# Run order and drift (Lee et al. 2019, Figure 8, approximate)
# =============================================================================
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    t = [i * 20 for i in range(11)]
    lat = [mag(K.ORDER.latency(K.Q_(x, "second")), "ms") for x in t]
    ax.fill_between(t, [v - 0.08 for v in lat], [v + 0.08 for v in lat],
                    color=C["neutral_l"], zorder=0)
    ax.plot(t, lat, color=C["neutral"], lw=2, zorder=1)
    ax.text(200, lat[-1] + 0.22, "device warms", ha="right", fontsize=10,
            color=C["neutral"], fontweight="bold")
    labels = K.ORDER.blocked_order if k == 1 else K.ORDER.interleaved_order
    for tq, lab in zip(K.ORDER.times, labels):
        x, y = mag(tq, "second"), mag(K.ORDER.latency(tq), "ms")
        ck = "memory" if lab == "A" else "compute"
        ax.plot(x, y, "o", ms=15, color=C[ck], zorder=4)
        ax.text(x, y, lab, ha="center", va="center", fontsize=9.5, color="white",
                fontweight="bold", zorder=5)
    msg = (f"Blocked order: B looks {val('order_blocked')}% slower" if k == 1
           else f"Interleaved: B looks {val('order_inter')}% slower")
    ax.text(5, 4.25, msg, fontsize=12, fontweight="bold",
            color=C["danger"] if k == 1 else C["data"])
    ax.text(5, 3.95, "A and B are the same code", fontsize=10, color=C["neutral"])
    ax.set_xlim(0, 200)
    ax.set_ylim(2.0, 4.5)
    ax.set_xlabel("Seconds of continuous inference")
    ax.set_ylabel("ms per inference")
    tag(ax, "Smooth curve through Lee et al. 2019, Fig. 8 endpoints (iPhone Xs GPU, approximate)", (0.0, -0.42))
    sp.save(fig, out / f"order_step{k}.svg")

# =============================================================================
# Real against useful: three intervals in percent faster than Eos
# =============================================================================
ADVICE = {"unresolved": "unresolved: measure more"}
items = [("Dell, all five runs", W_DELL_ALL, "neutral", "dellall"),
         ("Dell, converged runs", W_DELL, "compute", "dell"),
         ("H200", W_H200, "data", "h200")]
THRESH = num("min_useful")
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.15))
    ax.axvline(0, color=C["neutral"], lw=1.5, ls="--")
    ax.axvline(THRESH, color=C["danger"], lw=1.5, ls="--")
    ax.text(0.15, 2.62, "0", fontsize=10, color=C["neutral"], fontweight="bold")
    ax.text(THRESH + 0.15, 2.62, f"{val('min_useful')}%: set before the test (example)", fontsize=10, color=C["danger"],
            fontweight="bold")
    for i, (lab, w, ck, key) in enumerate(items):
        y = 2 - i
        lo, hi = pct(w.fraction_of_base(w.low)), pct(w.fraction_of_base(w.high))
        verdict = ADVICE.get(w.verdict, w.verdict)
        ax.plot([lo, hi], [y, y], color=C[ck], lw=6, solid_capstyle="butt")
        ax.plot(pct(w.fraction_of_base(w.difference)), y, "o", ms=9, color="white", mec=C[ck], mew=2, zorder=5)
        ax.text(-4.5, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
        xt = max(hi, THRESH) + 0.4
        ax.text(xt, y + 0.12, f"{val('pct_lo_' + key)} to {val('pct_hi_' + key)}%", va="center", fontsize=12,
                fontweight="bold", color=C[ck])
        ax.text(xt, y - 0.3, verdict, va="center", fontsize=11.5, fontweight="bold",
                color=C[ck], alpha=1.0 if k >= 2 else 0.0)
    ax.set_xlim(-4.2, 17)
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Percent faster than Eos (95% interval)")
    sp.save(fig, out / f"useful_step{k}.svg")

# =============================================================================
# Goodhart: wider beam (Example 12.1, illustrative)
# =============================================================================
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    lat0, lat1, slo = num("beam_lat0"), num("beam_lat1"), num("beam_slo")
    ax.bar([0], [lat0], color=C["memory"], width=0.6)
    al = 1.0 if k >= 2 else 0.0
    ax.bar([1], [lat1], color=C["danger"], width=0.6, alpha=al)
    ax.axhline(slo, color=C["data"], lw=1.5, ls="--")
    ax.text(0.5, slo + 6, f"SLO\n{val('beam_slo')} ms", fontsize=10, color=C["data"], fontweight="bold",
            va="bottom", ha="center")
    ax.text(0, lat0 + 8, f"{val('beam_lat0')} ms", ha="center", fontsize=12, fontweight="bold")
    ax.text(1, lat1 + 8, f"{val('beam_lat1')} ms", ha="center", fontsize=12, fontweight="bold", alpha=al)
    if k == 1:
        ax.text(1, 40, "???", ha="center", fontsize=16, fontweight="bold", color=C["danger"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"beam {val('beam_w0')}", f"beam {val('beam_w1')}\n(+{val('beam_gain')} BLEU)"], fontsize=11)
    ax.set_ylim(0, 245)
    ax.set_ylabel("latency (ms)")
    tag(ax, "Illustrative (Example 12.1)", (0.0, -0.42))
    sp.save(fig, out / f"beam_step{k}.svg")

# =============================================================================
# Benchmark fingerprinting: Galaxy S4, same test under two names (AnandTech)
# =============================================================================
fig, ax = plt.subplots(figsize=(6.2, 0.95))
for y, lab, v, ck, txt in ((1, f"GFXBench 2.7: {val('s4_game_mhz')} MHz", num("s4_gfx"), "memory",
                            f"{val('s4_gfx')}M triangles/s"),
                           (0, f"GLBenchmark 2.5.1: {val('s4_bench_mhz')} MHz", num("s4_glb"), "danger",
                            f"{val('s4_glb')}M  (+{val('s4_gain')}%)")):
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v + 0.6, y, txt, va="center", fontsize=12, fontweight="bold", color=C[ck])
    ax.text(-0.6, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
ax.set_xlim(0, 62)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Triangle throughput, identical test, mean of 5 runs")
sp.save(fig, out / "fingerprint.svg")

# =============================================================================
# Fresh test sets: Recht et al. 2019 table (ImageNetV2 MatchedFrequency, top-1),
# one dot per model, redrawn at slide size; GSM1k (Zhang et al. 2024, abstract)
# =============================================================================
rows = [(r["model"], r["original_top1"], r["imagenet_v2_top1"]) for r in K.SHIFT.rows
        if not str(r["model"]).startswith(K.SHIFT.baseline_prefix)]
fig, ax = plt.subplots(figsize=(3.7, 2.75))
ax.plot([55, 85], [55, 85], color=C["text"], lw=1.8, ls="--")
ax.text(66.5, 70.5, "same accuracy", rotation=31, fontsize=10.5, color=C["text"],
        fontweight="bold", ha="center")
ax.plot([o for _, o, _ in rows], [n for _, _, n in rows], "o", ms=5.5, color=C["memory"],
        mec="white", mew=0.5, zorder=4)
ax.text(84, 50.5, f"{val('recht_models')} models\n{val('recht_lo')} to {val('recht_hi')}\npoints lower;\norder kept",
        ha="right", va="top", fontsize=11, fontweight="bold", color=C["memory"])
ax.set_xlim(55, 85)
ax.set_ylim(40, 75)
ax.set_xticks([60, 70, 80])
ax.set_yticks([40, 50, 60, 70])
ax.set_xlabel("Original ImageNet top-1 (%)")
ax.set_ylabel("Rebuilt test set (%)")
ax.grid(color=C["grid"], lw=0.6)
sp.save(fig, out / "recht.svg")

fig, ax = plt.subplots(figsize=(2.2, 2.75))
ax.bar(0, num("gsm1k_drop"), color=C["danger"], width=0.5)
ax.text(0, num("gsm1k_drop") + 0.5, f"up to {val('gsm1k_drop')}%", ha="center", fontsize=12, fontweight="bold", color=C["danger"])
ax.set_xticks([0])
ax.set_xticklabels(["GSM1k\nagainst GSM8k"], fontsize=11)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(0, 15)
ax.set_ylabel("accuracy drop (%)")
sp.save(fig, out / "gsm1k.svg")

# =============================================================================
# Accuracy against latency on one phone (Sandler et al. 2018, Table 4)
# =============================================================================
pts = [(r["model"], r["latency_ms"], r["top1"], not K.dominated(r, K.SANDLER)) for r in K.SANDLER]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    for name, lat, acc, front in pts:
        col = C["memory"] if k == 1 else (C["data"] if front else C["neutral"])
        ax.plot(lat, acc, "o", ms=13, color=col, mec="white", mew=1.5, zorder=5)
        dx, dy, ha, va = (10, -2, "left", "top")
        if name == "NASNet-A":
            dx, dy, ha, va = (0, -12, "center", "top")
        if name == "MobileNetV2 (1.4)":
            dx, dy, ha, va = (-12, 4, "right", "bottom")
        lab = name if (k == 1 or front) else name + " (dominated)"
        ax.annotate(f"{lab}\n{acc:.1f}%, {lat:.0f} ms", (lat, acc), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=11,
                    fontweight="bold", color=col)
    if k == 2:
        front = sorted((lat, acc) for _, lat, acc, f in pts if f)
        ax.plot([p[0] for p in front], [p[1] for p in front], color=C["data"], lw=3, zorder=3)
    ax.set_xlim(55, 215)
    ax.set_ylim(69.0, 76.2)
    ax.set_xlabel("Latency on a Pixel 1 CPU core (ms)")
    ax.set_ylabel("ImageNet top-1 (%)")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"pareto_step{k}.svg")

# =============================================================================
# Hardware lottery: same four models, two processors of one Pixel 4 (Chu et al.)
# =============================================================================
chu = [(r["model"].split(" (")[0], r["cpu_float_ms"], r["dsp_ms"]) for r in K.CHU]
col = {"MobileNetV1": "memory", "MobileNetV2": "neutral", "MobileNetV3 Large": "data",
       "MobileNet-EdgeTPU": "network"}
fig, ax = plt.subplots(figsize=(5.8, 2.2))
cpu_rank = {n: i for i, (n, _, _) in enumerate(sorted(chu, key=lambda r: r[1]))}
dsp_rank = {n: i for i, (n, _, _) in enumerate(sorted(chu, key=lambda r: r[2]))}
for n, cpu, dsp in chu:
    y0, y1 = 3 - cpu_rank[n], 3 - dsp_rank[n]
    ax.plot([0, 1], [y0, y1], color=C[col[n]], lw=3, marker="o", ms=10)
    ax.text(-0.09, y0, f"{n}  {cpu:.1f} ms", ha="right", va="center", fontsize=11,
            fontweight="bold", color=C[col[n]])
    ax.text(1.09, y1, f"{dsp:.2f} ms  {n}", ha="left", va="center", fontsize=11,
            fontweight="bold", color=C[col[n]])
ax.text(0, 3.6, "CPU (float)", ha="center", fontsize=12, fontweight="bold")
ax.text(1, 3.6, "DSP", ha="center", fontsize=12, fontweight="bold")
ax.text(0.5, -0.75, f"fastest at top; top-1 {val('chu_top1_lo')} to {val('chu_top1_hi')}% for all four", ha="center",
        fontsize=10, color=C["neutral"])
ax.set_xlim(-1.4, 2.4)
ax.set_ylim(-0.9, 3.9)
ax.axis("off")
sp.save(fig, out / "lottery.svg")

# =============================================================================
# ECE by hand: three bins, 100 predictions (constructed for the slide)
# =============================================================================
bins = K.BINS.bins
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot([0.5, 1.0], [0.5, 1.0], color=C["data"], lw=2, ls="--")
    ax.text(0.52, 0.78, "dashed: perfect\ncalibration", fontsize=10, color=C["data"],
            fontweight="bold")
    for n, c, a in bins:
        w = 0.022 + 0.0011 * n
        ax.add_patch(Rectangle((c - w / 2, 0.5), w, a - 0.5, color=C["memory"], zorder=2))
        ax.text(c, 0.505, f"{n}", ha="center", va="bottom", fontsize=10, color="white",
                fontweight="bold", zorder=4)
        if k >= 2:
            ax.add_patch(Rectangle((c - w / 2, a), w, c - a, color=C["danger_l"], ec=C["danger"],
                                   lw=1.2, zorder=3))
            ax.text(c - w / 2 - 0.004, c + 0.004, f"{val('bin%d_gap' % (bins.index((n, c, a)) + 1))}", ha="right", va="bottom",
                    fontsize=11, fontweight="bold", color=C["danger"])
    if k >= 3:
        ax.text(0.52, 0.95, f"ECE = {val('ece')}", fontsize=14, fontweight="bold", color=C["danger"])
    ax.set_xlim(0.5, 1.03)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel("confidence in the bin")
    ax.set_ylabel("accuracy in the bin")
    tag(ax, f"Illustrative: {val('bins_total')} predictions in {len(bins)} bins", (0.0, -0.4))
    sp.save(fig, out / f"ece_step{k}.svg")

# =============================================================================
# Guo et al. 2017: more accurate, worse calibrated; temperature scaling
# =============================================================================
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.1),
                                 gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.55))
    xs = [0, 1]
    err, ece_ = ["guo_lenet_err", "guo_resnet_err"], ["guo_lenet_ece", "guo_resnet_ece"]
    a1.bar([x - 0.2 for x in xs], [num(k_) for k_ in err], width=0.38, color=C["neutral"])
    a1.bar([x + 0.2 for x in xs], [num(k_) for k_ in ece_], width=0.38, color=C["danger"])
    for x, e, c in zip(xs, err, ece_):
        a1.text(x - 0.2, num(e) + 1.2, val(e), ha="center", fontsize=10.5, fontweight="bold",
                color=C["neutral"])
        a1.text(x + 0.2, num(c) + 1.2, val(c), ha="center", fontsize=10.5, fontweight="bold",
                color=C["danger"])
    a1.set_xticks(xs)
    a1.set_xticklabels(["LeNet 5\n(1998)", "ResNet-110\n(2016)"], fontsize=10.5)
    a1.set_ylim(0, 55)
    a1.set_ylabel("percent")
    a1.text(0.62, 0.97, "error", transform=a1.transAxes, fontsize=10.5, fontweight="bold",
            color=C["neutral"], va="top")
    a1.text(0.62, 0.85, "ECE", transform=a1.transAxes, fontsize=10.5, fontweight="bold",
            color=C["danger"], va="top")
    a1.set_title("CIFAR-100", fontsize=12, fontweight="bold")
    al = 1.0 if k >= 2 else 0.0
    ts = ["guo_in_ece", "guo_in_ece_ts"]
    a2.bar([0, 1], [num(k_) for k_ in ts], width=0.55, color=[C["danger"], C["data"]], alpha=al)
    for x, k_ in zip([0, 1], ts):
        a2.text(x, num(k_) + 0.25, val(k_), ha="center", fontsize=10.5, fontweight="bold", alpha=al)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["before", "after\ntemp. scaling"], fontsize=10.5)
    a2.set_ylim(0, 7.5)
    a2.set_ylabel("ECE (%)", alpha=al)
    a2.set_title("ImageNet, ResNet-152", fontsize=12, fontweight="bold", alpha=al)
    for s_ in a2.spines.values():
        s_.set_alpha(al)
    a2.tick_params(colors=C["text"] if k >= 2 else "white")
    sp.save(fig, out / f"guo_step{k}.svg")

# =============================================================================
# WILDS iWildCam2020: in-distribution against unseen camera traps (Koh et al.)
# Drawn for the left column; the paper's camera-trap images sit on the right.
# =============================================================================
fig, ax = plt.subplots(figsize=(3.5, 2.7))
PQ = K.PQ
groups = [("Average\naccuracy", "wilds_id_acc", float(PQ.WildsIWildCamIdAccuracySd), "wilds_ood_acc",
           float(PQ.WildsIWildCamOodAccuracySd), "wilds_acc_drop"),
          ("Macro F1", "wilds_id_f1", float(PQ.WildsIWildCamIdMacroF1Sd), "wilds_ood_f1",
           float(PQ.WildsIWildCamOodMacroF1Sd), "wilds_f1_drop")]
for i, (lab, idk, ids, oodk, oods, dropk) in enumerate(groups):
    idv, oodv = num(idk), num(oodk)
    x = i * 1.6
    ax.bar(x - 0.28, idv, yerr=ids, width=0.5, color=C["memory"], capsize=4,
           error_kw=dict(lw=1.2, ecolor=C["text"]))
    ax.bar(x + 0.28, oodv, yerr=oods, width=0.5, color=C["danger"], capsize=4,
           error_kw=dict(lw=1.2, ecolor=C["text"]))
    ax.text(x - 0.28, idv + 3.5, val(idk), ha="center", fontsize=11, fontweight="bold", color=C["memory"])
    ax.text(x + 0.28, oodv + 3.5, val(oodk), ha="center", fontsize=11, fontweight="bold", color=C["danger"])
    ax.text(x, 104, f"−{val(dropk)}", ha="center", fontsize=13, fontweight="bold",
            color=C["danger"])
ax.set_xticks([0, 1.6])
ax.set_xticklabels([g[0] for g in groups], fontsize=11.5)
ax.set_xlim(-0.65, 2.25)
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75])
ax.set_ylabel("percent")
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=C["memory"]), Rectangle((0, 0), 1, 1, color=C["danger"])],
          labels=["traps seen in training", "unseen traps"], fontsize=11, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=1.0,
          columnspacing=1.0)
sp.save(fig, out / "wilds.svg")

print("figures written")
