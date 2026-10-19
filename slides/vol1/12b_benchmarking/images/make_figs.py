#!/usr/bin/env python3
"""Figures for Lecture 12b (Benchmarking, Part 2: Deciding from Measurements).

Every value is sourced. "Chapter" means the rendered Benchmarking chapter.
Values the chapter gives as a scenario or an assumption, and values built for
a slide, carry an "Illustrative" or "Assumed" tag on the plot.

Data computed here, not typed:
  - mlperf_v40_resnet_runs.csv (this folder): 20 MLPerf Training v4.0 closed
    ResNet-50 runs, one row per result_N.txt, read from the MLLOG lines
    (run_start, run_stop, last eval_accuracy, status, seed) of
    mlcommons/training_results_v4.0:
      NVIDIA/results/eos_ngc24.04_mxnet/resnet      (Eos, 8x H100-SXM5-80GB)
      Dell/results/XE9680x8H100-SXM-80GB/resnet     (Dell XE9680, 8x H100)
      NVIDIA/results/h200_ngc24.04_mxnet/resnet     (8x H200-SXM5-141GB)
      NVIDIA/results/eos_n8_ngc24.04_mxnet/resnet   (Eos, 64x H100)
    Every derived number on the slides (scores, CVs, Welch intervals, run
    counts, the binomial margins) is computed below from this file, and the
    script prints them so the notes can be checked against it.
  - Scoring rule: MLPerf Training rules (mlcommons/training_policies,
    training_rules.adoc, "Scoring" and "Score scaling"); the official
    result_summarizer in mlcommons/logging drops the fastest and slowest run
    (a non-converging run counts as the slowest), averages the rest, and
    multiplies by the scaling.json factor the review committee set for the
    benchmark folder. Every ResNet folder above carries scaling_factor
    1.0042232277526395 (normalization to the reference convergence points).

Outside values typed below, each with its source:
  - Napkin Math 12.5 (chapter): 95 against 94 percent on 1,000 images;
    McNemar threshold 26 disagreements; 1,825 images for +/-1 point.
  - MLPerf Inference rules (minimum-query table): p90 at 99 percent
    confidence, 0.5 percent margin, 23,886 rounded to 24,576; p99 at 99
    percent, 0.05 percent margin, 262,742 rounded to 270,336.
  - Example 12.1 (chapter, illustrative): beam 1 against 10, 50 to 200 ms,
    SLO 100 ms, +0.5 BLEU.
  - Klug and Lal Shimpi, "Looking at CPU/GPU Benchmark Optimizations in
    Galaxy S 4," AnandTech, July 30, 2013: GFXBench 2.7.0 at 480 MHz, 37.9M
    triangles/s (mean of 5 runs); GLBenchmark 2.5.1 at 532 MHz, 43.2M
    triangles/s; identical low-level test.
  - Recht et al. 2019 (abstract): accuracy drops of 3 to 15 points (CIFAR-10)
    and 11 to 14 points (ImageNet) on new test sets.
  - Zhang et al. 2024, GSM1k (arXiv 2405.00332 abstract): drops of up to 8
    percent against GSM8k.
  - Sandler et al. 2018, Table 4 (Pixel 1, TF-Lite): MobileNetV1 70.6% 113 ms,
    MobileNetV2 72.0% 75 ms, NASNet-A 74.0% 183 ms, MobileNetV2 (1.4) 74.7%
    143 ms.
  - Chu et al. 2021, Table 1 (Pixel 4): CPU float and DSP latency (ms) for
    MobileNetV1 (1.25) 54.7 / 3.72, MobileNetV2 (1.25) 38.8 / 4.97,
    MobileNetV3 Large 20.3 / 4.51, MobileNet-EdgeTPU 59.3 / 4.29; top-1 75.1,
    75.3, 75.3, 76.2.
  - Lee et al. 2019, Figure 8 (read from the plot by Lecture 12, approximate):
    iPhone Xs GPU, MobileNet v1, about 2.4 to 3.9 ms over 200 s.
  - ECE bins (constructed for the slide): 20 at confidence 0.65, accuracy
    0.60; 30 at 0.85, 0.80; 50 at 0.97, 0.84.
  - Guo et al. 2017 (Figure 1, Table 1, 15 bins): CIFAR-100 LeNet 5 error
    44.9, ECE 4.85%; ResNet-110 error 30.6, ECE 16.53%; ImageNet ResNet-152
    ECE 5.48% before and 1.86% after temperature scaling.
  - Koh et al. 2021, WILDS Table 4 (iWildCam2020-WILDS, ERM, ResNet-50, 3
    replicates): ID macro F1 47.0 (1.4), average accuracy 75.7 (0.3); OOD
    macro F1 31.0 (1.3), average accuracy 71.6 (2.5).

Run statistics and every derived number come from calcs.py (imported below;
it prints each value with its slide label). New data sources (2026-10-19):
  - Recht et al. 2019 appendix table (ImageNetV2 MatchedFrequency, top-1),
    recht2019_imagenetv2_top1.csv: the recht.svg scatter.
  - Zhang et al. 2024 abstract (GSM1k, up to 8 percent): gsm1k.svg.
The earlier two-panel range-bar figure is no longer drawn (replaced by these two).

Run from this folder with the MLSysBook venv (needs matplotlib and scipy):
  python3 make_figs.py
"""

import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import calcs  # noqa: E402
from calcs import (RUNS, SYSTEMS, EOS, DELL, H200, SCORE, PUB, CV,  # noqa: E402
                   W_DELL, W_DELL_ALL, W_H200, ACC, BINOM_HW, VAL_IMAGES,
                   N_ACC, N_P90, N_P99, runs, pct, runs_to_resolve, half_width_pct)

plt = sp.plt
C = sp.C
out = Path(__file__).parent


def tag(ax, text, where=(0.0, -0.32), size=9):
    ax.text(where[0], where[1], text, transform=ax.transAxes, fontsize=size,
            color=C["neutral"], ha="left")


# Every derived number is computed and printed by calcs.py (one block per slide).
calcs.report()

# =============================================================================
# Hook: two published scores, then the ten runs behind them
# =============================================================================
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 2.25),
                                 gridspec_kw=dict(height_ratios=[1.0, 1.2], hspace=1.0))
    rows = [("Dell XE9680, 8x H100", PUB[DELL], "compute"), ("NVIDIA Eos, 8x H100", PUB[EOS], "memory")]
    for i, (lab, v, ck) in enumerate(rows):
        y = 1 - i
        a1.barh(y, v, color=C[ck], height=0.62)
        a1.text(v + 0.15, y, f"{v:.2f} min", va="center", fontsize=12, fontweight="bold", color=C[ck])
        a1.text(-0.2, y, lab, ha="right", va="center", fontsize=12, fontweight="bold", color=C["text"])
    a1.set_xlim(0, 16)
    a1.set_ylim(-0.6, 1.6)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlabel("Published score: minutes to 75.9% top-1")
    al = 1.0 if k >= 2 else 0.0
    for i, (s, ck, lab) in enumerate(((DELL, "compute", "Dell runs"), (EOS, "memory", "Eos runs"))):
        y = 1 - i
        ok_sorted = sorted(float(r["minutes"]) for r in RUNS
                           if r["system"] == s and r["status"] == "success")
        for r in RUNS:
            if r["system"] != s:
                continue
            x = float(r["minutes"])
            if r["status"] == "success":
                # Alternate neighbors up and down so close runs stay visible.
                jit = 0.2 if ok_sorted.index(x) % 2 == 0 else -0.2
                a2.plot(x, y + jit, "o", ms=8, color=C[ck], mec="white", mew=1.0, alpha=al, zorder=4)
            else:
                a2.plot(x, y, "o", ms=11, mfc="white", mec=C["danger"], mew=2, alpha=al, zorder=4)
                a2.plot(x, y, "x", ms=7, color=C["danger"], mew=2, alpha=al, zorder=5)
                a2.text(x - 0.03, y - 0.48, "missed 75.9%\n(75.894% at epoch 35)", ha="right",
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
for y, lab, v, ck, txt in ((1, "Datasheet:\n10 TOPS at 0.5 W", 20, "compute", "20 TOPS/W"),
                           (0, "Sustained:\n3 TOPS at 2 W", 1.5, "danger", "1.5 TOPS/W")):
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
        med = st.median(runs(s))
        ax.text(-1.25, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
        for r in RUNS:
            if r["system"] != s:
                continue
            x = 100 * (float(r["minutes"]) / med - 1)
            if r["status"] == "success":
                ax.plot(x, y, "o", ms=10, color=C[ck], mec="white", mew=1.2, zorder=4)
            else:
                ax.plot(1.55, y, "o", ms=11, mfc="white", mec=C["danger"], mew=2, zorder=4)
                ax.plot(1.55, y, "x", ms=7, color=C["danger"], mew=2, zorder=5)
                ax.text(1.7, y, f"+{x:.1f}%, missed\ntarget", va="center", fontsize=10.5,
                        color=C["danger"], fontweight="bold")
        ax.text(2.75, y, f"CV {CV[s]:.2f}%", va="center", fontsize=12, fontweight="bold",
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
    offsets = {s: 0.18 * (i - 1.5) for i, (s, _, _) in enumerate(SYSTEMS)}
    colors = {s: ck for s, _, ck in SYSTEMS}
    for r in RUNS:
        a = float(r["top1"])
        y = offsets[r["system"]]
        if a < 75.9:
            ax.plot(a, y, "o", ms=11, mfc="white", mec=C["danger"], mew=2, zorder=4)
            ax.plot(a, y, "x", ms=7, color=C["danger"], mew=2, zorder=5)
        else:
            ax.plot(a, y, "o", ms=10, color=C[colors[r["system"]]], mec="white", mew=1.2, zorder=4)
    ax.axvline(75.9, color=C["danger"], lw=2, ls="--")
    ax.text(75.89, -1.15, "target\n75.9%", ha="right", va="center", fontsize=11, color=C["danger"], fontweight="bold")
    for s, lab, ck in SYSTEMS:
        ax.text(76.24, offsets[s], lab, va="center", fontsize=10, color=C[ck], fontweight="bold")
    al = 1.0 if k >= 2 else 0.0
    lo, hi = min(ACC), max(ACC)
    ax.annotate("", xy=(lo, -0.55), xytext=(hi, -0.55), alpha=al,
                arrowprops=dict(arrowstyle="|-|", color=C["text"], lw=1.5, alpha=al))
    ax.text((lo + hi) / 2, -0.75, f"seeds: {hi - lo:.2f} points", ha="center", va="top",
            fontsize=11, fontweight="bold", color=C["text"], alpha=al)
    m = st.mean(ACC)
    ax.annotate("", xy=(m - BINOM_HW, -1.25), xytext=(m + BINOM_HW, -1.25), alpha=al,
                arrowprops=dict(arrowstyle="|-|", color=C["memory"], lw=1.5, alpha=al))
    ax.text(m + BINOM_HW + 0.02, -1.25, f"new 50,000-image\ntest set: ±{BINOM_HW:.2f}",
            va="center", fontsize=11, fontweight="bold", color=C["memory"], alpha=al)
    ax.set_xlim(75.65, 76.6)
    ax.set_ylim(-1.6, 0.55)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ImageNet top-1 at the last evaluation (%), 20 runs")
    sp.save(fig, out / f"seedacc_step{k}.svg")

# =============================================================================
# The run rule on Dell's five runs
# =============================================================================
dell = sorted([r for r in RUNS if r["system"] == DELL], key=lambda r: float(r["minutes"]))
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.7))
    for j, r in enumerate(dell):
        x = float(r["minutes"])
        y = 4 - j
        ok = r["status"] == "success"
        dropped = (j == 0 or not ok)
        col = C["neutral"] if (k >= 2 and dropped) else C["compute"]
        xp = x if ok else 13.181
        ax.text(13.1365, y, f"run {j + 1}" if ok else "run 5", ha="right", va="center",
                fontsize=11, fontweight="bold", color=C["text"])
        if ok:
            ax.plot(xp, y, "o", ms=12, color=col, mec="white", mew=1.2, zorder=4)
            left = j > 0 and x < SCORE[DELL]
            ax.text(xp - 0.0016 if left else xp + 0.0016, y + 0.05, f"{x:.3f}", va="center",
                    ha="right" if left else "left", fontsize=11, color=col, fontweight="bold")
        else:
            ax.plot(xp, y, "o", ms=12, mfc="white", mec=C["danger"], mew=2, zorder=4)
            ax.text(xp - 0.0015, y, f"{x:.3f}, missed 75.9%, off scale", ha="right", va="center",
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
        sc = SCORE[DELL]
        ax.plot([sc, sc], [0.6, 3.4], color=C["text"], lw=2.5, zorder=5)
        ax.text(13.1385, 0.0, f"mean of runs 2 to 4:\n{sc:.3f} min", ha="left",
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
rows = [("accuracy", N_ACC, "data", f"{N_ACC:,.0f} images"),
        ("p90", N_P90, "memory", f"{N_P90:,.0f} queries"),
        ("p99", N_P99, "danger", f"{N_P99:,.0f} queries")]
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
        xs = sorted(runs(s))
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
    w = W_DELL
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
        a1.text(w["hi"] + 0.006, 0.5, f"{w['lo']:.3f} to {w['hi']:.3f}", va="center",
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
    h = W_H200
    a2.axvline(0, color=C["neutral"], lw=1.5, ls="--", alpha=al)
    a2.plot([h["lo"], h["hi"]], [0.5, 0.5], color=C["data"], lw=6, solid_capstyle="butt", alpha=al)
    a2.plot(h["d"], 0.5, "o", ms=10, color=C["data"], alpha=al, zorder=5)
    a2.text(h["lo"] - 0.03, 0.5, f"{h['lo']:.2f} to {h['hi']:.2f}", ha="right", va="center",
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
hw = [half_width_pct(n) for n in ns]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.plot(ns, hw, color=C["memory"], lw=3)
    for y, lab in ((1.0, "1%"), (0.5, "0.5%")):
        ax.axhline(y, color=C["neutral"], lw=1, ls="--")
        ax.text(30, y + 0.04, lab, va="bottom", ha="right", fontsize=11, color=C["neutral"],
                fontweight="bold")
    h5 = half_width_pct(5)
    ax.plot(5, h5, "o", ms=10, color=C["danger"], zorder=5)
    ax.text(6.2, h5 + 0.1, f"5 runs: ±{h5:.2f}%", fontsize=11.5,
            fontweight="bold", color=C["danger"])
    al = 1.0 if k >= 2 else 0.0
    h20 = half_width_pct(20)
    ax.plot(20, h20, "o", ms=10, color=C["data"], zorder=5, alpha=al)
    ax.text(20, h20 + 0.17, f"20 runs: ±{h20:.2f}%", ha="center", fontsize=11.5,
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
    a1.plot([-1, 3], [0, 0], color=C["memory"], lw=4)
    a1.plot([1], [0], "o", color=C["memory"], ms=10)
    a1.text(1, 0.45, "interval about −1 to 3", ha="center", fontsize=11,
            fontweight="bold", color=C["memory"])
    a1.text(-0.2, -0.7, "no change", ha="right", fontsize=10, color=C["neutral"])
    a1.set_xlim(-3, 5)
    a1.set_ylim(-1, 1)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlabel("Difference (points), if independent sets")
    al = 1.0 if k >= 2 else 0.0
    a2.axvspan(10, 26, color=C["danger_l"], alpha=al)
    a2.axvline(26, color=C["danger"], lw=2, ls="--", alpha=al)
    a2.text(18, 0.25, "regression", ha="center", fontsize=11, fontweight="bold",
            color=C["danger"], alpha=al)
    a2.text(43, 0.25, "unresolved", ha="center", fontsize=11, fontweight="bold",
            color=C["neutral"], alpha=al)
    a2.text(26.6, -0.6, "26", fontsize=11, fontweight="bold", color=C["danger"], alpha=al)
    a2.set_xlim(10, 60)
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
eos_runs = [r for r in RUNS if r["system"] == EOS]
best = max(eos_runs, key=lambda r: float(r["top1"]))
worst = min(eos_runs, key=lambda r: float(r["top1"]))
gap = round((float(best["top1"]) - float(worst["top1"])) / 100 * VAL_IMAGES)
limit = gap ** 2 / 3.84
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
    ax.text(limit + 120, -0.65, f"{limit:,.0f}", fontsize=11, fontweight="bold",
            color=C["memory"], alpha=al)
    ax.plot([gap, gap], [-0.4, 0.4], color=C["text"], lw=2)
    ax.text(gap + 120, -0.65, f"at least {gap}", fontsize=11, fontweight="bold", color=C["text"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 3000, 6000, 9000, 12000])
    ax.set_xticklabels(["0", "3,000", "6,000", "9,000", "12,000"])
    ax.set_xlabel("Images on which the two runs disagree, b + c (of 50,000)")
    sp.save(fig, out / f"mcn_runs_step{k}.svg")

# =============================================================================
# Run order and drift (Lee et al. 2019, Figure 8, approximate)
# =============================================================================
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    t = [i * 20 for i in range(11)]
    lat = [calcs.drift(x) for x in t]
    ax.fill_between(t, [v - 0.08 for v in lat], [v + 0.08 for v in lat],
                    color=C["neutral_l"], zorder=0)
    ax.plot(t, lat, color=C["neutral"], lw=2, zorder=1)
    ax.text(200, lat[-1] + 0.22, "device warms", ha="right", fontsize=10,
            color=C["neutral"], fontweight="bold")
    xs = calcs.ORDER_TIMES
    labels = calcs.ORDER_BLOCKED if k == 1 else calcs.ORDER_INTER
    for x, lab in zip(xs, labels):
        y = calcs.drift(x)
        ck = "memory" if lab == "A" else "compute"
        ax.plot(x, y, "o", ms=15, color=C[ck], zorder=4)
        ax.text(x, y, lab, ha="center", va="center", fontsize=9.5, color="white",
                fontweight="bold", zorder=5)
    effect = calcs.order_effects()[k - 1]
    msg = (f"Blocked order: B looks {effect:.0f}% slower" if k == 1
           else f"Interleaved: B looks {effect:.0f}% slower")
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
items = [("Dell, all five runs", W_DELL_ALL, "neutral", "unresolved: measure more"),
         ("Dell, converged runs", W_DELL, "compute", "real, below threshold"),
         ("H200", W_H200, "data", "real and useful")]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.15))
    ax.axvline(0, color=C["neutral"], lw=1.5, ls="--")
    ax.axvline(2, color=C["danger"], lw=1.5, ls="--")
    ax.text(0.15, 2.62, "0", fontsize=10, color=C["neutral"], fontweight="bold")
    ax.text(2.15, 2.62, "2%: set before the test (example)", fontsize=10, color=C["danger"],
            fontweight="bold")
    for i, (lab, w, ck, verdict) in enumerate(items):
        y = 2 - i
        lo, hi = pct(w["lo"]), pct(w["hi"])
        ax.plot([lo, hi], [y, y], color=C[ck], lw=6, solid_capstyle="butt")
        ax.plot(pct(w["d"]), y, "o", ms=9, color="white", mec=C[ck], mew=2, zorder=5)
        ax.text(-4.5, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
        xt = max(hi, 2.0) + 0.4
        ax.text(xt, y + 0.12, f"{lo:.1f} to {hi:.1f}%", va="center", fontsize=12,
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
    ax.bar([0], [50], color=C["memory"], width=0.6)
    al = 1.0 if k >= 2 else 0.0
    ax.bar([1], [200], color=C["danger"], width=0.6, alpha=al)
    ax.axhline(100, color=C["data"], lw=1.5, ls="--")
    ax.text(0.5, 106, "SLO\n100 ms", fontsize=10, color=C["data"], fontweight="bold",
            va="bottom", ha="center")
    ax.text(0, 58, "50 ms", ha="center", fontsize=12, fontweight="bold")
    ax.text(1, 208, "200 ms", ha="center", fontsize=12, fontweight="bold", alpha=al)
    if k == 1:
        ax.text(1, 40, "???", ha="center", fontsize=16, fontweight="bold", color=C["danger"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["beam 1", "beam 10\n(+0.5 BLEU)"], fontsize=11)
    ax.set_ylim(0, 245)
    ax.set_ylabel("latency (ms)")
    tag(ax, "Illustrative (Example 12.1)", (0.0, -0.42))
    sp.save(fig, out / f"beam_step{k}.svg")

# =============================================================================
# Benchmark fingerprinting: Galaxy S4, same test under two names (AnandTech)
# =============================================================================
fig, ax = plt.subplots(figsize=(6.2, 0.95))
for y, lab, v, ck, txt in ((1, "GFXBench 2.7: 480 MHz", 37.9, "memory", "37.9M triangles/s"),
                           (0, "GLBenchmark 2.5.1: 532 MHz", 43.2, "danger", "43.2M  (+14%)")):
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
rows = calcs.recht_rows()
fig, ax = plt.subplots(figsize=(3.7, 2.75))
ax.plot([55, 85], [55, 85], color=C["text"], lw=1.8, ls="--")
ax.text(66.5, 70.5, "same accuracy", rotation=31, fontsize=10.5, color=C["text"],
        fontweight="bold", ha="center")
ax.plot([o for _, o, _ in rows], [n for _, _, n in rows], "o", ms=5.5, color=C["memory"],
        mec="white", mew=0.5, zorder=4)
gaps = [o - n for _, o, n in rows]
ax.text(84, 50.5, f"{len(rows)} models\n{min(gaps):.1f} to {max(gaps):.1f}\npoints lower;\norder kept",
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
ax.bar(0, 8, color=C["danger"], width=0.5)
ax.text(0, 8.5, "up to 8%", ha="center", fontsize=12, fontweight="bold", color=C["danger"])
ax.set_xticks([0])
ax.set_xticklabels(["GSM1k\nagainst GSM8k"], fontsize=11)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(0, 15)
ax.set_ylabel("accuracy drop (%)")
sp.save(fig, out / "gsm1k.svg")

# =============================================================================
# Accuracy against latency on one phone (Sandler et al. 2018, Table 4)
# =============================================================================
pts = [("MobileNetV1", 113, 70.6, False), ("MobileNetV2", 75, 72.0, True),
       ("NASNet-A", 183, 74.0, False), ("MobileNetV2 (1.4)", 143, 74.7, True)]
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
        ax.annotate(f"{lab}\n{acc}%, {lat} ms", (lat, acc), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=11,
                    fontweight="bold", color=col)
    if k == 2:
        ax.plot([75, 143], [72.0, 74.7], color=C["data"], lw=3, zorder=3)
    ax.set_xlim(55, 215)
    ax.set_ylim(69.0, 76.2)
    ax.set_xlabel("Latency on a Pixel 1 CPU core (ms)")
    ax.set_ylabel("ImageNet top-1 (%)")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"pareto_step{k}.svg")

# =============================================================================
# Hardware lottery: same four models, two processors of one Pixel 4 (Chu et al.)
# =============================================================================
chu = [("MobileNetV1", 54.7, 3.72), ("MobileNetV2", 38.8, 4.97),
       ("MobileNetV3 Large", 20.3, 4.51), ("MobileNet-EdgeTPU", 59.3, 4.29)]
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
ax.text(0.5, -0.75, "fastest at top; top-1 75.1 to 76.2% for all four", ha="center",
        fontsize=10, color=C["neutral"])
ax.set_xlim(-1.4, 2.4)
ax.set_ylim(-0.9, 3.9)
ax.axis("off")
sp.save(fig, out / "lottery.svg")

# =============================================================================
# ECE by hand: three bins, 100 predictions (constructed for the slide)
# =============================================================================
bins = [(20, 0.65, 0.60), (30, 0.85, 0.80), (50, 0.97, 0.84)]
ece = calcs.ece(bins)
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
            ax.text(c - w / 2 - 0.004, c + 0.004, f"{c - a:.2f}", ha="right", va="bottom",
                    fontsize=11, fontweight="bold", color=C["danger"])
    if k >= 3:
        ax.text(0.52, 0.95, f"ECE = {ece:.2f}", fontsize=14, fontweight="bold", color=C["danger"])
    ax.set_xlim(0.5, 1.03)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel("confidence in the bin")
    ax.set_ylabel("accuracy in the bin")
    tag(ax, "Illustrative: 100 predictions in 3 bins", (0.0, -0.4))
    sp.save(fig, out / f"ece_step{k}.svg")

# =============================================================================
# Guo et al. 2017: more accurate, worse calibrated; temperature scaling
# =============================================================================
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.1),
                                 gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.55))
    xs = [0, 1]
    a1.bar([x - 0.2 for x in xs], [44.9, 30.6], width=0.38, color=C["neutral"])
    a1.bar([x + 0.2 for x in xs], [4.85, 16.53], width=0.38, color=C["danger"])
    for x, e, c in zip(xs, [44.9, 30.6], [4.85, 16.53]):
        a1.text(x - 0.2, e + 1.2, f"{e}", ha="center", fontsize=10.5, fontweight="bold",
                color=C["neutral"])
        a1.text(x + 0.2, c + 1.2, f"{c}", ha="center", fontsize=10.5, fontweight="bold",
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
    a2.bar([0, 1], [5.48, 1.86], width=0.55, color=[C["danger"], C["data"]], alpha=al)
    for x, v in zip([0, 1], [5.48, 1.86]):
        a2.text(x, v + 0.25, f"{v}", ha="center", fontsize=10.5, fontweight="bold", alpha=al)
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
groups = [("Average\naccuracy", 75.7, 0.3, 71.6, 2.5), ("Macro F1", 47.0, 1.4, 31.0, 1.3)]
for i, (lab, idv, ids, oodv, oods) in enumerate(groups):
    x = i * 1.6
    ax.bar(x - 0.28, idv, yerr=ids, width=0.5, color=C["memory"], capsize=4,
           error_kw=dict(lw=1.2, ecolor=C["text"]))
    ax.bar(x + 0.28, oodv, yerr=oods, width=0.5, color=C["danger"], capsize=4,
           error_kw=dict(lw=1.2, ecolor=C["text"]))
    ax.text(x - 0.28, idv + 3.5, f"{idv}", ha="center", fontsize=11, fontweight="bold", color=C["memory"])
    ax.text(x + 0.28, oodv + 3.5, f"{oodv}", ha="center", fontsize=11, fontweight="bold", color=C["danger"])
    ax.text(x, 104, f"−{idv - oodv:.1f}", ha="center", fontsize=13, fontweight="bold",
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
