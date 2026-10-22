#!/usr/bin/env python3
"""Data plots for Lecture 1 (Introduction).

Every plotted value and every printed label comes from MLSysIM through
calcs.py (val() and num() give the formatted string and the float the slides
print), or from the measured series MLSysIM holds:
  - Google Flu Trends against the CDC count: slides/vol1/01_introduction/images/data/google_flu_trends_cdc_ilinet.csv
    (Delphi Epidata API), seasons and peaks from Scenarios.GoogleFluTrendsDrift.
  - GPU-days: the registry's AlexNet, GPT-3, and GPT-4-class training runs.
  - Iron-law bars: Scenarios.ResNet50Batch64StepH100 and Gpt2XlDecodeStepH100
    (MLSysIM's roofline engine on one H100); the second bar of each pair
    applies the scenario's bandwidth multiple to the memory term only.
  - Degradation sketch: the recall points of Scenarios.PedestrianRecallWinterShift;
    the "better model" intercept, the flatter slope, and the winter position
    are illustrative layout, labeled "Illustrative" on the plot.
  - Amdahl pipeline: Scenarios.ModelSpeedupRequestPath.
  - ImageNetV2 scatter and fit: Scenarios.ImageNetV2Shift (Recht et al. 2019).
  - MLPerf: slides/vol1/01_introduction/images/data/mlperf_inference_v4_llama2_70b_offline.csv and the H100 and
    H200 registry entries.

Run from this folder with the MLSysBook venv (it imports mlsysim):
  python3 make_figs.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402  (records every value the slides print)
from deckvals import val, num  # noqa: E402

out = Path(__file__).parent
C = sp.C


# ---------------------------------------------------------------------------
# Google Flu Trends against the CDC count (real weekly data)
# ---------------------------------------------------------------------------
def epiweek_to_date(ew):
    """Approximate start date of an MMWR epiweek (good to a few days)."""
    y, w = divmod(ew, 100)
    jan4 = date(y, 1, 4)
    start = jan4 - timedelta(days=(jan4.weekday() + 1) % 7)  # Sunday on or before Jan 4
    return start + timedelta(weeks=w - 1)


gft, cdc = calcs.flu_weeks()
weeks = sorted(w for w in gft if w in cdc and 201027 <= w <= 201426)   # plotted window (layout)
FLU = calcs.GFT
peak_lo = FLU.peak_year * 100 + FLU.start_week                         # the peak season's weeks
peak_hi = (FLU.peak_year + 1) * 100 + FLU.end_week
xs = [epiweek_to_date(w) for w in weeks]


def gft_plot(path, cut=None, annotate=False, size=(5.8, 2.6), gft_label=(201027, 9.6),
             font=11):
    """gft_label: (epiweek, y) of the top-left corner of the red series label."""
    fig, ax = sp.plt.subplots(figsize=size)
    ax.plot(xs, [cdc[w] for w in weeks], color=C["neutral"], lw=3)
    ax.text(epiweek_to_date(201432), cdc[weeks[-1]] + 0.2, "CDC\n(doctor visits)",
            color=C["neutral"], fontsize=font, fontweight="bold", va="bottom")
    shown = [w for w in weeks if cut is None or w <= cut]
    ax.plot([epiweek_to_date(w) for w in shown], [gft[w] for w in shown],
            color=C["danger"], lw=3)
    ax.text(epiweek_to_date(gft_label[0]), gft_label[1], "Google Flu Trends\n(from searches)",
            color=C["danger"], fontsize=font, fontweight="bold", va="top")
    if annotate:
        pk = max((w for w in weeks if peak_lo <= w <= peak_hi), key=lambda w: gft[w])
        ck = max((w for w in weeks if peak_lo <= w <= peak_hi), key=lambda w: cdc[w])
        ax.annotate(f"{val('gft_peak')}%", (epiweek_to_date(pk), gft[pk]),
                    xytext=(8, -2), textcoords="offset points", color=C["danger"],
                    fontsize=12, fontweight="bold", va="center")
        ax.annotate(f"{val('cdc_peak')}%", (epiweek_to_date(ck), cdc[ck]),
                    xytext=(-6, 4), textcoords="offset points", ha="right",
                    color=C["neutral"], fontsize=12, fontweight="bold")
    ax.set_xlim(xs[0], epiweek_to_date(201452))
    ax.set_xticks([date(y, 1, 1) for y in range(2011, 2015)])
    ax.set_xticklabels([str(y) for y in range(2011, 2015)])
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Flu-like illness\n(% of doctor visits)")
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


gft_plot(out / "gft_step1.svg", cut=201131)
gft_plot(out / "gft_step2.svg", annotate=True)
# Column-sized copy for the synthesis slide, drawn at the size it is shown
# (about 8.7 cm wide), with the red label moved clear of the y-axis.
gft_plot(out / "gft_column.svg", annotate=True, size=(3.3, 2.05), gft_label=(201036, 10.9),
         font=10)

# Seasonal ratio of the GFT estimate to the CDC count (Scenarios.GoogleFluTrendsDrift);
# the drift seasons are drawn red and labeled with the values the notes print
seasons = list(FLU.result.season_ratios.items())
drift_labels = {calcs.season_label(FLU.first_drift_year): val("gft_ratio_first_drift"),
                FLU.result.peak_season: val("gft_ratio_peak")}
fig, ax = sp.plt.subplots(figsize=(5.8, 2.4))
for i, (lab, r) in enumerate(seasons):
    ck = "danger" if lab in drift_labels else "neutral"
    ax.bar(i, r, color=C[ck], width=0.66)
    if lab in drift_labels:
        ax.text(i, r + 0.04, drift_labels[lab], ha="center", va="bottom", color=C[ck],
                fontsize=12, fontweight="bold")
ax.axhline(1.0, color=C["text"], lw=1.2, ls="--")
ax.text(-0.45, 1.04, "estimate = CDC count", fontsize=10, color=C["text"], va="bottom")
ax.set_xticks(range(len(seasons)))
ax.set_xticklabels([s[0] for s in seasons], fontsize=10)
ax.set_ylim(0, 1.95)
ax.set_xlabel("Flu season")
ax.set_ylabel("Estimate / CDC")
sp.save(fig, out / "gft_ratio.svg")


# ---------------------------------------------------------------------------
# GPU-days to reach each capability threshold (log scale)
# ---------------------------------------------------------------------------
sp.hbars_steps(out / "gpu_days", [
    ("AlexNet (2012)", num("alexnet_gpu_days"), "neutral", f"{val('alexnet_gpu_days')} GPU-days"),
    ("GPT-3 (2020)", num("gpt3_gpu_days"), "compute", f"about {val('gpt3_gpu_days')}"),
    ("GPT-4-class (2023)", num("gpt4_gpu_days_m") * 10 ** 6, "danger",
     f"about {val('gpt4_gpu_days_m')} million (est.)"),
], "GPU-days", xlabel="Accelerator-days to train (log scale)", value_fmt="{:,.0f}")


# ---------------------------------------------------------------------------
# Iron-law bars from MLSysIM: ResNet-50 (batch 64) against GPT-2 XL decode
# (batch 1) on one H100, then the same with twice the memory bandwidth
# ---------------------------------------------------------------------------
# (key prefix in calcs, single-node scenario); terms in ms at full precision
BARS = (("rn", calcs.RESNET), ("g2", calcs.GPT2XL))
TERMS = {key: tuple(float(q.to("millisecond").magnitude) for q in
                    (sc.iron_law.memory_time, sc.iron_law.compute_time, sc.iron_law.overhead))
         for key, sc in BARS}
print("iron-law terms (ms):", TERMS)


def iron_bars(path, show_double):
    """Stacked memory, compute, and overhead bars with each segment's value
    in ms, so students can halve the memory term and add. Values sit inside
    the segments; the term names sit above the first bar."""
    # Drawn at the size it is shown (about 14 x 5.2 cm on the slide). The two
    # workloads sit in separate groups, with room above each first bar.
    fig, ax = sp.plt.subplots(figsize=(4.4, 1.75))
    ypos = [3.6, 2.75, 1.55, 0.7]
    names = [f"ResNet-50\nbatch {val('rn_batch')}", "GPT-2 XL decode\nbatch 1"]
    rows = []
    for name, (key, sc) in zip(names, BARS):
        m, c, o = TERMS[key]
        rows.append((name, key, m, c, o, False))
        rows.append((f"{val('bw_gain')}\u00d7 bandwidth", key, m / sc.memory_speedup, c, o, True))
    wide = 0.2   # ms; a segment this wide holds its value inside
    for i, (name, key, m, c, o, dbl) in enumerate(rows):
        y = ypos[i]
        vis = (not dbl) or show_double
        a = 1.0 if vis else 0.0
        left = 0
        terms = ((m, "memory", f"{key}_mem2" if dbl else f"{key}_mem"),
                 (c, "compute", f"{key}_comp"), (o, "neutral", f"{key}_over"))
        for v, ck, vkey in terms:
            col = C[ck + "_l"] if dbl else C[ck]
            ax.barh(y, v, left=left, color=col, edgecolor="white", height=0.62, alpha=a)
            if vis and v >= wide:
                ax.text(left + v / 2, y, val(vkey), ha="center", va="center",
                        color=C["text"] if dbl else "white", fontsize=11, fontweight="bold")
            left += v
        ax.text(-0.03, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=11, color=C["text"] if vis else "white",
                fontweight="bold" if not dbl else "normal", linespacing=1.0)
        lab = f"{val(key + '_total2')} ms  (-{val(key + '_cut_pct')}%)" if dbl else f"{val(key + '_total')} ms"
        # drawn transparent when hidden so both steps share one canvas
        ax.text(left + 0.03, y, lab, va="center", fontsize=12,
                color=C["text"], fontweight="bold", alpha=1 if vis else 0)
    # term names above the ResNet-50 bar; the narrow memory segment gets its value too
    rm, rc, ro = TERMS["rn"]
    top = ypos[0] + 0.36
    ax.text(0, top, f"memory {val('rn_mem')}", ha="left", va="bottom", color=C["memory"],
            fontsize=10, fontweight="bold")
    ax.text(rm + rc * 0.7, top, "compute", ha="center", va="bottom", color=C["compute"],
            fontsize=10, fontweight="bold")
    ax.text(rm + rc + ro / 2, top, "overhead", ha="center", va="bottom", color=C["neutral"],
            fontsize=10, fontweight="bold")
    # GPT-2 XL: name the memory segment, and call out the sliver of compute
    gm, gc, go = TERMS["g2"]
    top = ypos[2] + 0.36
    ax.text(gm / 2, top, "memory", ha="center", va="bottom", color=C["memory"],
            fontsize=10, fontweight="bold")
    ax.text(gm + gc / 2, top, f"compute {val('g2_comp')}", ha="left", va="bottom", color=C["compute"],
            fontsize=10, fontweight="bold")
    ax.plot([gm + gc / 2, gm + gc / 2], [ypos[2] + 0.31, top + 0.05], color=C["compute"], lw=1.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 2.25)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylim(0.3, 4.45)
    ax.set_xlabel("Time per step on one H100 (ms)")
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


iron_bars(out / "ironlaw_bars_step1.svg", False)
iron_bars(out / "ironlaw_bars_step2.svg", True)


# ---------------------------------------------------------------------------
# Degradation equation as lines: intercept Accuracy_0, slope lambda.
# Illustrative only; the 95 and 85 percent points are the chapter's
# pedestrian example (Sec. 1.6). Step 2 raises the intercept (a better
# model on the same data); step 3 flattens the slope (winter images in
# training, which the chapter says may reduce sensitivity).
# ---------------------------------------------------------------------------
def degradation(path, step):
    # drawn at the size it is shown (about 13.5 x 3.7 cm under the equation)
    fig, ax = sp.plt.subplots(figsize=(3.9, 1.2))
    winter = 0.6                                  # illustrative position of winter on the drift axis
    xs_line = [0, 1]
    r0, r1 = num("recall_before"), num("recall_after")
    base = (r0, (r0 - r1) / winter)
    lines = [(base, C["danger"], "today's model", 1),
             ((99, base[1]), C["compute"], "better model, same data: starts higher, falls as fast", 2),  # illustrative
             ((r0, 5.0), C["data"], "winter images in training: falls more slowly", 3)]               # illustrative
    for (a0, lam), col, lab, k in lines:
        ys = [a0 - lam * x for x in xs_line]
        # unshown lines are drawn transparent so every step has the same canvas
        ax.plot(xs_line, ys, color=col, lw=3, solid_capstyle="round",
                alpha=1 if k <= step else 0)
    # direct labels
    if step >= 1:
        ax.plot([0, winter], [r0, r1], "o", color=C["danger"], ms=7, zorder=4)
        ax.text(0.025, 91.8, f"{val('recall_before')}%", color=C["danger"], fontsize=11, fontweight="bold", va="top")
        ax.text(winter - 0.015, 83.5, f"{val('recall_after')}%", color=C["danger"], fontsize=11,
                fontweight="bold", va="top", ha="right")
        ax.text(1.01, r0 - base[1], "today's model", color=C["danger"], fontsize=11,
                fontweight="bold", va="center")
    ax.text(1.01, 99 - base[1], "better model, same data", color=C["compute"],
            fontsize=11, fontweight="bold", va="center", alpha=1 if step >= 2 else 0)
    ax.text(1.01, 90, "winter images in training", color=C["data"], fontsize=11,
            fontweight="bold", va="center", alpha=1 if step >= 3 else 0)
    ax.axvline(winter, color=C["neutral"], lw=1, ls="--", zorder=0)
    ax.text(winter + 0.01, 100.5, "winter", color=C["neutral"], fontsize=10, va="top")
    ax.text(0.99, 100.5, "Illustrative", color=C["neutral"], fontsize=10, va="top",
            ha="right", style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(75, 101)
    ax.set_xticks([0])
    ax.set_xticklabels(["training conditions"])
    ax.set_yticks([80, 90, 100])
    ax.set_xlabel("Distance from the training data (drift)", labelpad=2)
    ax.set_ylabel("Recall (%)")
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2, 3):
    degradation(out / f"degradation_step{k}.svg", k)


# ---------------------------------------------------------------------------
# Amdahl: a 3x faster model in a 130 ms request path
# ---------------------------------------------------------------------------
def amdahl(path, show_after):
    fig, ax = sp.plt.subplots(figsize=(5.8, 1.9))
    rows = [("Before", ("am_pre", "am_model", "am_post"), "am_total"),
            ("After", ("am_pre", "am_model2", "am_post"), "am_total2")]
    for i, (name, keys, total_key) in enumerate(rows):
        y = 1 - i
        vis = i == 0 or show_after
        left = 0
        for k, ck, lab in zip(keys, ("neutral", "compute", "neutral"),
                              (f"preprocess {val('am_pre')}", "model", f"post {val('am_post')}")):
            v = num(k)
            ax.barh(y, v, left=left, color=C[ck] if ck == "compute" else C["neutral_l"],
                    edgecolor="white", linewidth=2, height=0.6, alpha=1 if vis else 0)
            if vis:
                txt = (f"model {val(k)}" if v > 20 else val(k)) if ck == "compute" else lab
                ax.text(left + v / 2, y, txt, ha="center", va="center", fontsize=11,
                        fontweight="bold",
                        color="white" if ck == "compute" else C["text"])
            left += v
        ax.text(-2, y, name, ha="right", va="center", fontsize=13, fontweight="bold",
                color=C["text"] if vis else "white")
        if vis:
            ax.text(left + 2, y, f"{val(total_key)} ms" + (f"   (-{val('am_cut_pct')}%)" if i == 1 else ""),
                    va="center", fontsize=13, fontweight="bold",
                    color=C["danger"] if i == 1 else C["text"])
    ax.set_xlim(0, 165)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("End-to-end latency (ms)")
    sp.save(fig, path)


amdahl(out / "amdahl_step1.svg", False)
amdahl(out / "amdahl_step2.svg", True)

print("figures written")


# ---------------------------------------------------------------------------
# Algorithmic efficiency (Hernandez and Brown 2020) and training compute
# (Epoch AI), from the MLSysIM literature registry the book's figures use.
# Drawn at half-slide size so the type stays legible side by side.
# ---------------------------------------------------------------------------
def literature():
    Literature = calcs.Literature
    algo = [(e["model"], e["year"], Literature.AlgorithmicEfficiency.factor(e["model"]))
            for e in Literature.AlgorithmicEfficiency.entries]
    comp = [(e["model"], e["year"], e["flops"]) for e in Literature.ComputeTrend.entries]
    return algo, comp


ALGO, COMP = literature()
HALF = (3.5, 2.8)

fig, ax = sp.plt.subplots(figsize=HALF)
ax.scatter([a[1] for a in ALGO], [a[2] for a in ALGO], s=60, color=C["compute"],
           edgecolors="white", zorder=3)
for name, yr, f in ALGO:
    if name in ("AlexNet", "MobileNet-v2", "EfficientNet-B0"):
        lab = f"EfficientNet: {val('algo_mult')}\u00d7" if name == "EfficientNet-B0" else name
        off = {"AlexNet": (0, 10), "MobileNet-v2": (8, -6), "EfficientNet-B0": (-6, 0)}[name]
        ax.annotate(lab, (yr, f), xytext=off, textcoords="offset points",
                    ha={"EfficientNet-B0": "right", "MobileNet-v2": "left"}.get(name, "center"),
                    va="bottom" if name == "AlexNet" else "center",
                    fontsize=11, fontweight="bold",
                    color=C["compute"] if name == "EfficientNet-B0" else C["text"])
ax.set_xlim(2012, 2020)
ax.set_ylim(0, 50)
ax.set_xticks([2012, 2014, 2016, 2018, 2020])
ax.set_ylabel("Less compute than AlexNet (\u00d7)")
ax.set_xlabel("Year")
ax.grid(axis="y", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "algo_efficiency.svg")

fig, ax = sp.plt.subplots(figsize=HALF)
ax.scatter([c[1] for c in COMP], [c[2] for c in COMP], s=60, color=C["danger"],
           edgecolors="white", zorder=3)
for name, yr, fl in COMP:
    if name in ("AlexNet", "GPT-3", "GPT-4-class"):
        off = {"AlexNet": (4, 8), "GPT-3": (-6, 6), "GPT-4-class": (-6, 6)}[name]
        ax.annotate(name, (yr, fl), xytext=off, textcoords="offset points",
                    ha="left" if name == "AlexNet" else "right", va="bottom",
                    fontsize=11, fontweight="bold", color=C["text"])
ax.set_yscale("log")
ax.set_xlim(2011, 2026)
ax.set_xticks([2012, 2016, 2020, 2024])
ax.set_ylabel("Training compute (FLOPs)")
ax.set_xlabel("Year")
ax.minorticks_off()
ax.grid(axis="y", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "compute_growth.svg")
print("literature figures written")


# ---------------------------------------------------------------------------
# Measured silent degradation: ImageNet top-1 on the original validation set
# against ImageNetV2 MatchedFrequency, one point per model (Recht et al.
# 2019, appendix tables). Values and the fit come from Scenarios.ImageNetV2Shift.
# ---------------------------------------------------------------------------
REC = calcs.recht_rows()
fig, ax = sp.plt.subplots(figsize=(5.4, 1.6))
ox = [r[1] for r in REC]
ny = [r[2] for r in REC]
slope, icpt = calcs.RECHT.result.slope, calcs.RECHT.result.intercept
ax.plot([15, 90], [15, 90], color=C["neutral"], lw=1.5, ls="--", zorder=1)
ax.text(54, 57, "no loss (new = original)", color=C["neutral"], fontsize=11,
        rotation=0, ha="right", va="bottom")
ax.plot([18, 86], [icpt + slope * 18, icpt + slope * 86], color=C["danger"], lw=2.2, zorder=2)
ax.scatter(ox, ny, s=34, color=C["compute"], edgecolors="white", linewidths=0.6, zorder=3)
ax.text(55, 30, f"fit: new = {val('recht_slope')} × original − {val('recht_icpt_neg')}",
        color=C["danger"], fontsize=12, fontweight="bold", ha="left")
ax.text(19, 84, f"{val('recht_models')} models", color=C["compute"], fontsize=12, fontweight="bold",
        ha="left", va="top")
ax.set_xlim(15, 90)
ax.set_ylim(10, 90)
ax.set_xticks([20, 40, 60, 80])
ax.set_yticks([20, 40, 60, 80])
ax.set_xlabel("Top-1 on original ImageNet test set (%)")
ax.set_ylabel("ImageNetV2\ntop-1 (%)")
ax.grid(color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "recht_scatter.svg")


# ---------------------------------------------------------------------------
# MLPerf Inference v4.0, Llama 2 70B Offline: H100 against H200, throughput
# relative to the H100, with the bandwidth and compute ratios as reference
# lines. Throughput from slides/vol1/01_introduction/images/data/mlperf_inference_v4_llama2_70b_offline.csv;
# specs from the MLSysIM registry (calcs.py).
# ---------------------------------------------------------------------------
H100_RUN, H200_RUN, CTS_RUN = calcs.mlperf_rows()
bw_ratio = calcs.ratio(calcs.h200.memory.bandwidth, calcs.h100.memory.bandwidth)   # full precision for the line
rows = [(f"H100, {val('h_tdp')} W", H100_RUN, "neutral", "1.00", "mlperf_h100_tps_c"),
        (f"H200, {val('h_tdp')} W", H200_RUN, "memory", val("mlperf_h200_speedup"), "mlperf_h200_tps_c"),
        ("H200, 1000 W", CTS_RUN, "memory", val("mlperf_cts_speedup"), "mlperf_cts_tps_c")]
fig, ax = sp.plt.subplots(figsize=(5.0, 1.3))
for i, (lab, run, ck, mult, tkey) in enumerate(rows):
    y = len(rows) - 1 - i
    r = run["tokens_per_s"] / H100_RUN["tokens_per_s"]
    ax.barh(y, r, color=C[ck], height=0.6)
    ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=12, fontweight="bold", color=C["text"])
    ax.text(r - 0.03, y, f"{mult}\u00d7", ha="right", va="center", fontsize=12,
            fontweight="bold", color="white")
    ax.text(1.62, y, f"{val(tkey)} tokens/s", va="center", fontsize=11,
            fontweight="bold", color=C[ck])
ax.axvline(num("compute_ratio"), color=C["compute"], lw=1.8, ls="--", zorder=0)
ax.text(0.98, 2.45, f"compute ratio {val('compute_ratio')}", color=C["compute"], fontsize=10, ha="right",
        va="bottom", fontweight="bold")
ax.axvline(bw_ratio, color=C["memory"], lw=1.8, ls="--", zorder=0)
ax.text(bw_ratio + 0.02, 2.45, f"bandwidth ratio {val('bw_ratio')}", color=C["memory"], fontsize=10,
        ha="left", va="bottom", fontweight="bold")
ax.set_xlim(0, 2.2)
ax.set_ylim(-0.5, 2.95)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xlabel("Throughput relative to H100 (8 GPUs each)")
sp.save(fig, out / "mlperf_h200.svg")
print("measured-evidence figures written")
