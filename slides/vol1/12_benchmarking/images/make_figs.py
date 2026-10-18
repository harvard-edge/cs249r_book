#!/usr/bin/env python3
"""Figures for Lecture 12 (Benchmarking).

Every value is sourced. "Chapter" means the rendered Benchmarking chapter.
Values the chapter gives as a scenario, composite, or assumption are drawn
with an "Illustrative" tag on the plot.

  - Hook: "10 TOPS at 0.5 W" against 3 TOPS at 2 W under sustained load,
    20 against 1.5 TOPS/W, 13.3x (chapter, Sec. 12.7 opening; illustrative).
  - Fan-out tail: Dean and Barroso, "The Tail at Scale," CACM 2013, Table 1
    (measured Google service): one random leaf p50 1 ms, p99 10 ms; all leaf
    requests p50 40 ms, p99 140 ms. Model-only mean 15 ms against production
    p99 150 to 200 ms (chapter, Sec. 12.10; illustrative).
  - Roofline against MLPerf: 989 TFLOP/s / 24.6 GFLOP = 40,200 images/s;
    about 5,000 images/s per H100, 12.4 percent MFU (chapter, Napkin Math
    12.1). Published logs, MLPerf Training v4.0, NVIDIA closed division
    (mlcommons/training_results_v4.0), ResNet, images/s per GPU derived as
    35 epochs x 1,281,167 images / timed run / GPUs: Eos (8x H100-SXM5,
    one node) about 7,000; Eos_n8 (64 H100-SXM5) about 4,660.
  - Edge board waterfall: Lighthouse 12.2 and Example 12.3 (illustrative).
  - Software stack: Lee et al. 2019 (arXiv 1907.01989), MobileNet v1, mean
    of 100 runs after 10 warm-up runs: iPhone Xs TFLite GPU 2.3 ms, Metal
    MPSCNN 4.1 ms, Core ML 7.1 ms; Pixel 3 TFLite GPU 12.5 ms, TFLite CPU
    51.7 ms.
  - Scenarios: MLPerf Inference rules and chapter Sec. 12.4.4.
  - MLPerf Inference v5.0 closed, ResNet-50 (summary_results.json,
    mlcommons/inference_results_v5.0): RTX 4090 (5.0-0032) SingleStream
    0.281736 ms, Offline 44,411.1 samples/s; Cisco UCS C885A M8 8x H100
    (5.0-0010) Offline 715,761, Server 652,002; Quanta D74H-7U 8x H100
    (5.0-0065) Offline 712,859, Server 584,161.
  - Boundary: Elsworth et al. 2025 via chapter Sec. 12.7.1: 0.10 Wh against
    0.24 Wh; 58 / 25 / 17 percent.
  - Throttling: Lee et al. 2019, Figure 8, read from the plot (approximate):
    iPhone Xs TFLite CPU about 20 to 26 ms, GPU about 2.4 to 3.9 ms, Pixel 3
    GPU about 12.8 ms throughout, over 200 s. Tummalapalli et al. 2026
    (arXiv 2603.23640), iPhone 16 Pro, Qwen 2.5 1.5B: 40.49, 24.96, 23.67
    tokens/s.
  - MLPerf Tiny v1.2 closed, visual wake words (summary.csv,
    mlcommons/tiny_results_v1.2): uJ per inference.
  - Duty cycle: chapter Sec. 12.7.2 (8.3 ms at 0.4 W each 1 s, 50 mW idle).
  - Statistics: chapter Napkin Math 12.5.
  - Goodhart: chapter Example 12.1 (illustrative); Recht et al. 2019
    abstract (accuracy drops on new test sets).
  - Pareto: Sandler et al. 2018, Table 4 (Pixel 1, one large core, TF-Lite).
    (Statistics, Goodhart, and Pareto figures are no longer used by this deck;
    Lecture 12b regenerates them.)
  - Fan-out: chapter footnote 20 (Dean and Barroso 2013): 1 - 0.99^100 = 63
    percent; the curve 1 - 0.99^n is computed.
  - Time to quality: mlperf_v40_eos_result0_evals.csv (this folder), the
    eval_accuracy lines of MLPerf Training v4.0, NVIDIA eos_ngc24.04_mxnet,
    resnet/result_0.txt (minutes since run_start, top-1 percent).
  - Scaling: MLPerf Training v4.0 published scores, NVIDIA eos_ngc24.04_mxnet
    (8x H100) and eos_n8_ngc24.04_mxnet (64x H100): mean of the middle three
    runs (a non-converged run counts as slowest) times the scaling.json
    factor 1.0042232; run times from the result_N.txt logs, typed below.
  - Power boundaries: chapter Sec. 12.7.1 and figure 8 (after Tschand et al.).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent


def tag(ax, text="Illustrative", where=(0.0, -0.32)):
    ax.text(where[0], where[1], text, transform=ax.transAxes, fontsize=9,
            color=C["neutral"], ha="left")


def bars(ax, rows, k, log=False, xmax=None, value_pad=None):
    """Horizontal bars on one axis.

    rows: dicts with label, value, color, text, step (first step shown),
    and optional pending (text shown in its slot before the bar appears).
    """
    n = len(rows)
    vals = [r["value"] for r in rows]
    top = xmax or max(vals) * 1.35
    for i, r in enumerate(rows):
        y = n - 1 - i
        shown = k >= r.get("step", 1)
        ax.barh(y, r["value"], color=C[r["color"]], height=0.62,
                alpha=1.0 if shown else 0.0)
        label_on = shown or (r.get("pending") and k >= r.get("pending_from", 1))
        ax.text(-0.01, y, r["label"], transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=12, color=C["text"],
                fontweight="bold", alpha=1.0 if label_on else 0.0)
        x = r["value"] * 1.12 if log else r["value"] + (value_pad or top * 0.012)
        if shown:
            ax.text(x, y, r["text"], va="center", fontsize=12,
                    color=C[r["color"]], fontweight="bold")
        elif r.get("pending") and k >= r.get("pending_from", 1):
            ax.text(top * 0.02 if not log else min(vals) * 1.1, y,
                    r["pending"], va="center", fontsize=14,
                    color=C["danger"], fontweight="bold")
        # invisible copy keeps the bounding box the same in every step
        ax.text(x, y, r["text"], va="center", fontsize=12, alpha=0.0)
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if log:
        ax.set_xscale("log")
    else:
        ax.set_xlim(0, top)


# -----------------------------------------------------------------------------
# Hook: datasheet against sustained efficiency (illustrative, Sec. 12.7)
# -----------------------------------------------------------------------------
hook_rows = [
    dict(label="Datasheet: 10 TOPS at 0.5 W", value=20, color="compute",
         text="20 TOPS/W", step=1),
    dict(label="Sustained: 3 TOPS at 2 W", value=1.5, color="danger",
         text="1.5 TOPS/W", step=3, pending="???", pending_from=2),
]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    bars(ax, hook_rows, k, xmax=27)
    if k == 3:
        ax.annotate("", xy=(1.5, 0.5), xytext=(20, 0.5),
                    arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.5))
        ax.text(20.5, 0.5, "13.3× lower", ha="left", va="center", fontsize=13,
                fontweight="bold", color=C["text"])
    ax.set_xlabel("Efficiency (TOPS per watt)")
    tag(ax, "Illustrative (Benchmarking, §12.7)", (0.0, -0.5))
    sp.save(fig, out / f"hook_step{k}.svg")

# -----------------------------------------------------------------------------
# Fan-out tail (Dean and Barroso 2013, Table 1) and the illustrative gap
# -----------------------------------------------------------------------------
tail_rows = [
    dict(label="One leaf, median", value=1, color="memory", text="1 ms", step=1),
    dict(label="One leaf, p99", value=10, color="memory", text="10 ms", step=1),
    dict(label="All leaves, median", value=40, color="danger", text="40 ms", step=2),
    dict(label="All leaves, p99", value=140, color="danger", text="140 ms  (14× one leaf)", step=2),
]
gap_rows = [
    dict(label="Model-only mean", value=15, color="neutral", text="15 ms", step=3),
    dict(label="Production p99", value=200, color="danger",
         text="150 to 200 ms", step=3),
]
for k in (1, 2, 3):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.8, 3.3),
                                  gridspec_kw=dict(height_ratios=[2.0, 1.0], hspace=0.75))
    bars(ax, tail_rows, k, xmax=215)
    ax.set_xlabel("Latency (ms), measured Google service")
    bars(ax2, gap_rows, k, xmax=215)
    if k >= 3:
        ax2.barh(0, 50, left=150, color=C["danger_l"], height=0.62)
        ax2.plot([33, 33], [-0.31, 0.31], color="white", lw=2)
        ax2.text(36, 0, "33 ms frame deadline", color="white", fontsize=10,
                 fontweight="bold", va="center")
        tag(ax2, "Illustrative (Benchmarking, §12.10)", (0.62, 1.12))
    else:
        for side in ("bottom",):
            ax2.spines[side].set_visible(False)
        ax2.set_xticks([])
        ax2.grid(False)
    ax2.set_xlabel("Latency (ms)" if k >= 3 else "")
    sp.save(fig, out / f"tail_step{k}.svg")

# -----------------------------------------------------------------------------
# Closed-loop against open-loop load generators (mechanism, Sec. 12.2.4)
# -----------------------------------------------------------------------------
stall = (300, 800)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    ax.add_patch(Rectangle((stall[0], 0.1), stall[1] - stall[0], 2.0,
                           color=C["neutral_l"], zorder=0))
    ax.text((stall[0] + stall[1]) / 2, 2.25, "server stalls 500 ms",
            ha="center", fontsize=11, color=C["neutral"], fontweight="bold")
    # closed loop: next request only after the previous returns
    y = 1.6
    t = 0
    while t < 1000:
        if stall[0] <= t < stall[1]:
            end = stall[1] + 20
            ax.add_patch(Rectangle((t, y - 0.18), end - t, 0.36, color=C["danger"]))
            t = end + 15
            continue
        ax.add_patch(Rectangle((t, y - 0.18), 20, 0.36, color=C["memory"]))
        t += 35
    ax.text(-15, y, "Closed loop", ha="right", va="center", fontsize=12,
            fontweight="bold", color=C["text"])
    ax.text(1010, y, "1 slow request", va="center", fontsize=11,
            fontweight="bold", color=C["danger"])
    # open loop: requests on their own schedule; arrivals in the stall wait
    y = 0.6
    slow = 0
    alpha = 1.0 if k >= 2 else 0.0
    for a_ in range(0, 1000, 35):
        if stall[0] <= a_ < stall[1]:
            ax.add_patch(Rectangle((a_, y - 0.18), stall[1] + 20 - a_, 0.36,
                                   color=C["danger"], alpha=0.35 * alpha, lw=0))
            ax.plot([a_, a_], [y - 0.18, y + 0.18], color=C["danger"], lw=1.5, alpha=alpha)
            slow += 1
        else:
            ax.add_patch(Rectangle((a_, y - 0.18), 20, 0.36, color=C["memory"], alpha=alpha))
    ax.text(-15, y, "Open loop", ha="right", va="center", fontsize=12,
            fontweight="bold", color=C["text"], alpha=alpha)
    ax.text(1010, y, f"{slow} slow requests", va="center", fontsize=11,
            fontweight="bold", color=C["danger"], alpha=alpha)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0.0, 2.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xlabel("Time (ms)")
    tag(ax, "Mechanism only (Benchmarking, §12.2.4)", (0.0, -0.42))
    sp.save(fig, out / f"loop_step{k}.svg")

# Host timer against device timer (Systems Perspective 12.4, rule 5)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.6, 1.7))
    ax.add_patch(Rectangle((0, 0.8), 30, 0.4, color=C["neutral"]))
    ax.text(15, 1.35, "enqueue", ha="center", fontsize=10.5, color=C["neutral"], fontweight="bold")
    ax.add_patch(Rectangle((40, 0.8), 420, 0.4, color=C["compute"]))
    ax.text(250, 1.0, "kernel runs on the GPU", ha="center", va="center", fontsize=11,
            color="white", fontweight="bold")
    ax.annotate("", xy=(30, 0.4), xytext=(0, 0.4),
                arrowprops=dict(arrowstyle="|-|", color=C["danger"], lw=1.6))
    ax.text(45, 0.4, "host timer stops at enqueue", fontsize=11, color=C["danger"],
            fontweight="bold", va="center")
    al = 1.0 if k >= 2 else 0.0
    ax.annotate("", xy=(460, -0.15), xytext=(0, -0.15), alpha=al,
                arrowprops=dict(arrowstyle="|-|", color=C["data"], lw=1.6, alpha=al))
    ax.text(470, -0.15, "with synchronize: kernel end", fontsize=11, color=C["data"],
            fontweight="bold", va="center", alpha=al)
    ax.set_xlim(-10, 760)
    ax.set_ylim(-0.45, 1.6)
    ax.axis("off")
    tag(ax, "Mechanism only, not to scale (Benchmarking, Systems Perspective 12.4)", (0.0, -0.08))
    sp.save(fig, out / f"timer_step{k}.svg")

# -----------------------------------------------------------------------------
# Granularity: the measurement boundary widens (after the book's figure)
# -----------------------------------------------------------------------------
tiers = [
    ("ML layers", "one GEMM\nkernel", "Micro"),
    ("ML model", "ResNet-50\nforward pass", "Macro"),
    ("AI task", "+ decode,\npostprocess", "Application"),
    ("End to end", "+ network,\nqueue, storage", "End to end"),
]
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    xs = [2, 26, 50, 74]
    for i, (name, what, kind) in enumerate(tiers):
        x = xs[i]
        active = i < k
        ax.add_patch(FancyBboxPatch((x, 14), 22, 22, boxstyle="round,pad=0.3",
                                    fc=C["compute_l"] if i == 0 else C["neutral_l"],
                                    ec=C["neutral"], lw=1.0, alpha=1.0))
        ax.text(x + 11, 31, name, ha="center", va="center", fontsize=12,
                fontweight="bold", color=C["text"])
        ax.text(x + 11, 21.5, what, ha="center", va="center", fontsize=10.5,
                color=C["text"])
        ax.text(x + 11, 9, kind, ha="center", va="center", fontsize=10.5,
                color=C["danger"] if active else C["neutral"], fontweight="bold",
                alpha=1.0 if active else 0.0)
    # the measured boundary (crimson) wraps tiers 1..k
    right = xs[k - 1] + 23.2
    ax.add_patch(FancyBboxPatch((0.8, 12.5), right - 0.8, 25.5,
                                boxstyle="round,pad=0.3", fc="none",
                                ec="#A51C30", lw=2.6))
    ax.text(0.8, 43, "Timed region", color="#A51C30", fontsize=11,
            fontweight="bold")
    sp.save(fig, out / f"granularity_step{k}.svg")

# -----------------------------------------------------------------------------
# Roofline ceiling against MLPerf measurement (Napkin Math 12.1)
# -----------------------------------------------------------------------------
mfu_rows = [
    dict(label="Roofline ceiling", value=40200, color="compute",
         text="40,200 images/s", step=1),
    dict(label="MLPerf, per H100", value=5000, color="danger",
         text="about 5,000 (chapter estimate)", step=3, pending="???", pending_from=2),
]
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    bars(ax, mfu_rows, k, xmax=56000)
    if k >= 3:
        ax.annotate("", xy=(5000, 0.5), xytext=(40200, 0.5),
                    arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.4))
        ax.text(41000, 0.5, "8.0× below", ha="left", va="center", fontsize=12,
                fontweight="bold", color=C["text"])
    if k >= 4:
        ax.plot([4660, 7050], [-0.42, -0.42], color=C["text"], lw=2)
        for x in (4660, 7050):
            ax.plot([x, x], [-0.52, -0.32], color=C["text"], lw=2)
        ax.text(7600, -0.42, "MLPerf v4.0 logs: 7,000 (8 GPUs), 4,660 (64 GPUs)",
                va="center", fontsize=11, color=C["text"])
    ax.set_ylim(-0.75, 1.6)
    ax.set_xlabel("Training throughput (images/s per H100)")
    sp.save(fig, out / f"mfu_step{k}.svg")

# -----------------------------------------------------------------------------
# Edge board: isolated, end to end, energy (Lighthouse 12.2, illustrative)
# -----------------------------------------------------------------------------
for k in (1, 2, 3):
    fig, axes = plt.subplots(1, 3, figsize=(5.3, 2.15),
                             gridspec_kw=dict(width_ratios=[3, 2, 2], wspace=0.55))
    a1, a2, a3 = axes
    a1.bar([0, 1, 2], [50, 15, 2], color=[C["data"], C["data"], C["compute"]], width=0.62)
    for x, v in zip([0, 1, 2], [50, 15, 2]):
        a1.text(x, v + 1.5, f"{v} ms", ha="center", fontsize=11, fontweight="bold")
    a1.set_xticks([0, 1, 2])
    a1.set_xticklabels(["CPU\nFP32", "CPU\nINT8", "Accel.\nINT8"], fontsize=10)
    a1.set_ylim(0, 62)
    a1.set_title("Model only", fontsize=12, fontweight="bold")
    a1.text(1.5, 30, "7.5×", ha="center", fontsize=13, fontweight="bold", color=C["compute"])
    a1.set_ylabel("ms per inference")
    for ax, shown in ((a2, k >= 2), (a3, k >= 3)):
        al = 1.0 if shown else 0.0
        for sp_ in ax.spines.values():
            sp_.set_alpha(al)
        ax.tick_params(colors=C["text"] if shown else "white")
    al2 = 1.0 if k >= 2 else 0.0
    a2.bar([0, 1], [15, 2], color=[C["data"], C["compute"]], width=0.62, alpha=al2)
    a2.bar([0, 1], [3, 4], bottom=[15, 2], color=C["network_l"], hatch="///",
           edgecolor=C["network"], width=0.62, alpha=al2)
    for x, v in zip([0, 1], [18, 6]):
        a2.text(x, v + 1.5, f"{v} ms", ha="center", fontsize=11, fontweight="bold", alpha=al2)
    a2.text(0.5, 30, "3×", ha="center", fontsize=13, fontweight="bold", color=C["compute"], alpha=al2)
    a2.text(0.5, 50, "hatched: prep\n+ link", ha="center", fontsize=9.5, color=C["network"], alpha=al2)
    a2.set_ylim(0, 62)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["CPU", "Accel."], fontsize=10)
    a2.set_title("End to end", fontsize=12, fontweight="bold", alpha=al2)
    al3 = 1.0 if k >= 3 else 0.0
    a3.bar([0, 1], [1.8, 1.0], color=[C["data"], C["compute"]], width=0.62, alpha=al3)
    for x, v in zip([0, 1], ["1.8 mJ", "1 mJ"]):
        a3.text(x, [1.8, 1.0][x] + 0.06, v, ha="center", fontsize=11, fontweight="bold", alpha=al3)
    a3.set_ylim(0, 2.5)
    a3.set_xlim(-0.6, 1.6)
    a2.set_xlim(-0.6, 1.6)
    a3.set_xticks([0, 1])
    a3.set_xticklabels(["CPU", "Accel."], fontsize=10)
    a3.set_title("Active energy", fontsize=12, fontweight="bold", alpha=al3)
    a3.set_yticks([0, 1, 2])
    tag(a1, "Illustrative (Lighthouse 12.2)", (0.0, -0.45))
    sp.save(fig, out / f"waterfall_step{k}.svg")

# -----------------------------------------------------------------------------
# Same phone, same model, different software (Lee et al. 2019)
# -----------------------------------------------------------------------------
lee_rows = [
    dict(label="iPhone Xs, TFLite GPU", value=2.3, color="compute", text="2.3 ms", step=1),
    dict(label="iPhone Xs, Metal (MPSCNN)", value=4.1, color="compute", text="4.1 ms", step=1),
    dict(label="iPhone Xs, Core ML", value=7.1, color="compute", text="7.1 ms  (3.1×)", step=1),
    dict(label="Pixel 3, TFLite GPU", value=12.5, color="memory", text="12.5 ms", step=2),
    dict(label="Pixel 3, TFLite CPU", value=51.7, color="memory", text="51.7 ms  (4.1×)", step=2),
]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    bars(ax, lee_rows, k, xmax=72)
    ax.set_xlabel("MobileNet v1 latency (ms, mean of 100 runs)")
    sp.save(fig, out / f"lee_step{k}.svg")

# -----------------------------------------------------------------------------
# Four MLPerf Inference scenarios on one time axis
# -----------------------------------------------------------------------------
scen = [
    ("SingleStream", "p90 latency", "phone camera"),
    ("MultiStream", "p99 latency, 8 per query", "multi-camera car"),
    ("Server", "queries/s under a p99 bound", "web API"),
    ("Offline", "samples/s", "overnight batch"),
]
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    ax.set_xlim(-38, 132)
    ax.set_ylim(-0.6, 3.7)
    ax.axis("off")
    ax.plot([0, 80], [-0.45, -0.45], color=C["neutral"], lw=1)
    ax.text(40, -0.6, "time", ha="center", va="top", fontsize=10, color=C["neutral"])
    for i, (name, metric, use) in enumerate(scen):
        y = 3 - i
        al = 1.0 if i < k else 0.0
        ax.text(-2, y, name, ha="right", va="center", fontsize=12,
                fontweight="bold", color=C["text"], alpha=al)
        ax.text(84, y + 0.13, metric, va="center", fontsize=10, color=C["text"], alpha=al)
        ax.text(84, y - 0.22, use, va="center", fontsize=10, color=C["data"],
                fontweight="bold", alpha=al)
        if name == "SingleStream":
            for j in range(8):
                ax.add_patch(Rectangle((j * 10, y - 0.2), 9, 0.4, color=C["memory"], alpha=al))
        elif name == "MultiStream":
            for j in range(4):
                for s in range(8):
                    ax.add_patch(Rectangle((j * 20, y - 0.3 + s * 0.075), 18, 0.06,
                                           color=C["memory"], alpha=al))
        elif name == "Server":
            arr = [1, 4, 5, 13, 21, 22, 23, 30, 38, 47, 48, 55, 61, 62, 70, 76]
            for a in arr:
                ax.plot([a, a], [y - 0.25, y + 0.25], color=C["danger"], lw=2.2, alpha=al)
        else:
            ax.add_patch(Rectangle((0, y - 0.3), 80, 0.6, color=C["compute"], alpha=al))
            ax.text(40, y, "all samples available at t = 0", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold", alpha=al)
    sp.save(fig, out / f"scen_step{k}.svg")

# -----------------------------------------------------------------------------
# MLPerf Inference v5.0 closed, ResNet-50: one GPU, two scenarios; 8x H100
# -----------------------------------------------------------------------------
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.3),
                                 gridspec_kw=dict(wspace=0.9, width_ratios=[1, 1.25]))
    a1.bar([0, 1], [3550, 44411], color=[C["memory"], C["compute"]], width=0.62)
    a1.text(0, 3550 + 1500, "≈3,550\n(derived)", ha="center", fontsize=10, fontweight="bold",
            color=C["memory"])
    a1.text(1, 44411 + 1500, "44,411", ha="center", fontsize=11, fontweight="bold",
            color=C["compute"])
    a1.text(0, 30000, "12.5×", ha="center", fontsize=13, fontweight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["Single-\nStream", "Offline"], fontsize=10)
    a1.set_ylim(0, 55000)
    a1.set_yticks([0, 20000, 40000])
    a1.set_yticklabels(["0", "20k", "40k"])
    a1.set_ylabel("images/s")
    a1.set_title("1× RTX 4090", fontsize=12, fontweight="bold")
    al = 1.0 if k >= 2 else 0.0
    xs = [0, 0.78, 1.95, 2.73]
    vals = [715761, 652002, 712859, 584161]
    cols = [C["compute"], C["danger"], C["compute"], C["danger"]]
    a2.bar(xs, vals, color=cols, width=0.6, alpha=al)
    for x, v, lab in zip(xs, vals, ["716k", "652k\n−9%", "713k", "584k\n−18%"]):
        a2.text(x, v + 15000, lab, ha="center", fontsize=9.5, fontweight="bold", alpha=al)
    a2.set_xticks(xs)
    a2.set_xticklabels(["Off.", "Srv.", "Off.", "Srv."], fontsize=10)
    a2.text(0.39, -230000, "Cisco", ha="center", fontsize=10, fontweight="bold", alpha=al)
    a2.text(2.34, -230000, "Quanta", ha="center", fontsize=10, fontweight="bold", alpha=al)
    a2.set_ylim(0, 900000)
    a2.set_yticks([0, 400000, 800000])
    a2.set_yticklabels(["0", "400k", "800k"])
    a2.set_title("8× H100, two submitters", fontsize=12, fontweight="bold", alpha=al)
    for s in a2.spines.values():
        s.set_alpha(al)
    a2.tick_params(colors=C["text"] if k >= 2 else "white")
    sp.save(fig, out / f"mlperf_step{k}.svg")

# -----------------------------------------------------------------------------
# Measurement boundary: Gemini median prompt (Elsworth et al. 2025)
# -----------------------------------------------------------------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    ax.barh(1, 0.10, color=C["compute"], height=0.55)
    ax.text(0.103, 1, "0.10 Wh", va="center", fontsize=12, fontweight="bold", color=C["compute"])
    parts = [(0.24 * 0.58, C["compute"], "accelerators 58%"),
             (0.24 * 0.25, C["memory"], "host CPU, DRAM 25%"),
             (0.24 * 0.17, C["neutral"], "idle, facility 17%")]
    left = 0
    for i, (w, col, lab) in enumerate(parts):
        c = col if k >= 2 else C["danger"]
        ax.barh(0, w, left=left, color=c, height=0.55)
        if k >= 2:
            ax.text(left + w / 2, -0.52 if i != 1 else 0.48, lab, ha="center",
                    va="center", fontsize=10, color=col, fontweight="bold")
        left += w
    ax.text(0.243, 0, "0.24 Wh (2.4×)", va="center", fontsize=12, fontweight="bold",
            color=C["danger"])
    ax.text(-0.004, 1, "Active accelerators only", ha="right", va="center", fontsize=12,
            fontweight="bold")
    ax.text(-0.004, 0, "Whole serving system", ha="right", va="center", fontsize=12,
            fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-0.85, 1.5)
    ax.set_xlim(0, 0.32)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Energy per median text prompt (Wh)")
    sp.save(fig, out / f"boundary_step{k}.svg")

# -----------------------------------------------------------------------------
# Sustained load slows phones (Lee et al. 2019; Tummalapalli et al. 2026)
# -----------------------------------------------------------------------------
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.3),
                                 gridspec_kw=dict(wspace=0.55, width_ratios=[1.5, 1]))
    groups = [("iPhone\nCPU", 20, 26, "+30%"), ("iPhone\nGPU", 2.4, 3.9, "+60%"),
              ("Pixel 3\nGPU", 12.8, 12.8, "flat")]
    for i, (lab, s0, s1, d) in enumerate(groups):
        a1.bar(i - 0.17, s0, width=0.32, color=C["memory"])
        a1.bar(i + 0.17, s1, width=0.32, color=C["danger"])
        a1.text(i, max(s0, s1) + 1.2, d, ha="center", fontsize=11, fontweight="bold",
                color=C["danger"] if d != "flat" else C["neutral"])
    a1.set_xticks(range(3))
    a1.set_xticklabels([g[0] for g in groups], fontsize=10)
    a1.set_ylim(0, 32)
    a1.set_ylabel("ms per inference")
    a1.text(0.40, 0.97, "start", color=C["memory"], fontsize=10, fontweight="bold",
            transform=a1.transAxes, va="top")
    a1.text(0.40, 0.86, "after 200 s", color=C["danger"], fontsize=10, fontweight="bold",
            transform=a1.transAxes, va="top")
    a1.set_title("iPhone Xs, Pixel 3 (approx.)", fontsize=11, fontweight="bold")
    al = 1.0 if k >= 2 else 0.0
    a2.bar([0, 1, 2], [40.49, 24.96, 23.67], color=[C["memory"], C["compute"], C["danger"]],
           width=0.62, alpha=al)
    for x, v in zip([0, 1, 2], ["40.5", "25.0", "23.7\n−41.5%"]):
        a2.text(x, [40.49, 24.96, 23.67][x] + 1.2, v, ha="center", fontsize=10,
                fontweight="bold", alpha=al)
    a2.set_xticks([0, 1, 2])
    a2.set_xticklabels(["peak", "warm", "hot"], fontsize=10)
    a2.set_ylim(0, 52)
    a2.set_ylabel("tokens/s")
    a2.set_title("iPhone 16 Pro, LLM", fontsize=11, fontweight="bold", alpha=al)
    for s in a2.spines.values():
        s.set_alpha(al)
    a2.tick_params(colors=C["text"] if k >= 2 else "white")
    a2.yaxis.label.set_alpha(al)
    sp.save(fig, out / f"throttle_step{k}.svg")

# -----------------------------------------------------------------------------
# MLPerf Tiny v1.2, visual wake words: energy per inference
# -----------------------------------------------------------------------------
tiny_rows = [
    dict(label="Syntiant NDP9120, 30.7 MHz", value=71.711, color="data", text="72 µJ", step=1),
    dict(label="Syntiant NDP9120, 98.7 MHz", value=97.164, color="data",
         text="97 µJ  (3.1× faster)", step=1),
    dict(label="STM32U575", value=1354.685, color="neutral", text="1,355 µJ", step=2),
    dict(label="STM32H7A3", value=2965.461, color="neutral", text="2,965 µJ", step=2),
    dict(label="STM32L4R5", value=3894.566, color="neutral", text="3,895 µJ  (54×)", step=2),
]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    bars(ax, tiny_rows, k, log=True)
    ax.set_xlim(20, 90000)
    ax.set_xlabel("Energy per inference (µJ, log scale)")
    sp.save(fig, out / f"tiny_step{k}.svg")

# -----------------------------------------------------------------------------
# Duty cycle: average power of a keyword spotter (Sec. 12.7.2)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 0.75))
ax.barh(0, 49.6, color=C["neutral"], height=0.8)
ax.barh(0, 3.3, left=49.6, color=C["compute"], height=0.8)
ax.text(24.8, 0, "idle 49.6 mW (94%)", ha="center", va="center", color="white",
        fontsize=13, fontweight="bold")
ax.text(54, 0, "+ active 3.3 mW = 52.9 mW", va="center", fontsize=13,
        fontweight="bold", color=C["compute"])
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 0.5)
ax.axis("off")
sp.save(fig, out / "duty.svg")

# -----------------------------------------------------------------------------
# Napkin Math 12.5: interval, then the disagreement threshold
# -----------------------------------------------------------------------------
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 2.1),
                                 gridspec_kw=dict(hspace=1.25))
    a1.plot([0, 0], [-0.35, 0.35], color=C["neutral"], lw=1.5, ls="--")
    a1.plot([-1, 3], [0, 0], color=C["memory"], lw=4)
    a1.plot([1], [0], "o", color=C["memory"], ms=10)
    a1.text(1, 0.45, "1-point drop, interval about −1 to 3", ha="center", fontsize=11,
            fontweight="bold", color=C["memory"])
    a1.text(-0.2, -0.7, "no change", ha="right", fontsize=10, color=C["neutral"])
    a1.set_xlim(-3, 5)
    a1.set_ylim(-1, 1)
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlabel("Accuracy difference (points), as if independent samples")
    al = 1.0 if k >= 2 else 0.0
    a2.axvspan(10, 26, color=C["danger_l"], alpha=al)
    a2.axvline(26, color=C["danger"], lw=2, ls="--", alpha=al)
    a2.text(18, 0.25, "regression shown", ha="center", fontsize=11, fontweight="bold",
            color=C["danger"], alpha=al)
    a2.text(43, 0.25, "unresolved", ha="center", fontsize=11, fontweight="bold",
            color=C["neutral"], alpha=al)
    a2.text(26.6, -0.6, "26", fontsize=11, fontweight="bold", color=C["danger"], alpha=al)
    a2.set_xlim(10, 60)
    a2.set_ylim(-1, 1)
    a2.set_yticks([])
    a2.spines["left"].set_visible(False)
    a2.set_xlabel("Images on which the two models disagree (same 1,000 images)")
    for s in a2.spines.values():
        s.set_alpha(al)
    a2.tick_params(colors=C["text"] if k >= 2 else "white")
    a2.xaxis.label.set_alpha(al)
    sp.save(fig, out / f"conf_step{k}.svg")

# -----------------------------------------------------------------------------
# Goodhart: wider beam (illustrative) and rebuilt test sets (Recht et al.)
# -----------------------------------------------------------------------------
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.2),
                                 gridspec_kw=dict(wspace=0.6))
    a1.bar([0, 1], [50, 200], color=[C["memory"], C["danger"]], width=0.6)
    a1.axhline(100, color=C["data"], lw=1.5, ls="--")
    a1.text(-0.4, 104, "SLO 100 ms", fontsize=9.5, color=C["data"], fontweight="bold",
            va="bottom", ha="left")
    a1.text(0, 58, "50 ms", ha="center", fontsize=11, fontweight="bold")
    a1.text(1, 208, "200 ms", ha="center", fontsize=11, fontweight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["beam 1", "beam 10\n(+0.5 BLEU)"], fontsize=10)
    a1.set_ylim(0, 245)
    a1.set_ylabel("latency (ms)")
    a1.set_title("Translation (illustrative)", fontsize=12, fontweight="bold")
    al = 1.0 if k >= 2 else 0.0
    for i, (lo, hi, lab) in enumerate([(3, 15, "CIFAR-10"), (11, 14, "ImageNet")]):
        a2.bar(i, hi - lo, bottom=lo, color=C["danger"], width=0.5, alpha=al)
        a2.text(i, hi + 0.8, f"{lo} to {hi}", ha="center", fontsize=11, fontweight="bold",
                alpha=al)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["CIFAR-10", "ImageNet"], fontsize=10)
    a2.set_ylim(0, 18)
    a2.set_ylabel("accuracy drop (points)")
    a2.set_title("New test sets", fontsize=12, fontweight="bold", alpha=al)
    for s in a2.spines.values():
        s.set_alpha(al)
    a2.tick_params(colors=C["text"] if k >= 2 else "white")
    a2.yaxis.label.set_alpha(al)
    sp.save(fig, out / f"goodhart_step{k}.svg")

# -----------------------------------------------------------------------------
# Accuracy against latency on one phone (Sandler et al. 2018, Table 4)
# -----------------------------------------------------------------------------
pts = [("MobileNetV1", 113, 70.6, False), ("MobileNetV2", 75, 72.0, True),
       ("NASNet-A", 183, 74.0, False), ("MobileNetV2 (1.4)", 143, 74.7, True)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    for name, lat, acc, front in pts:
        col = C["data"] if (front or k == 1) else C["neutral"]
        if k == 1:
            col = C["memory"]
        ax.plot(lat, acc, "o", ms=11, color=col, mec="white", mew=1.5, zorder=5)
        dx, dy, ha, va = (6, -4, "left", "top")
        if name == "NASNet-A":
            dx, ha = (-6, "right")
        if name == "MobileNetV2 (1.4)":
            dx, dy, ha, va = (-6, 6, "right", "bottom")
        lab = name if (k == 1 or front) else name + " (dominated)"
        ax.annotate(f"{lab}\n{acc}%, {lat} ms", (lat, acc), xytext=(dx * 1.5, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=10,
                    fontweight="bold", color=col)
    if k == 2:
        ax.plot([75, 143], [72.0, 74.7], color=C["data"], lw=3, zorder=3)
    ax.set_xlim(55, 215)
    ax.set_ylim(69.5, 76.4)
    ax.set_xlabel("Latency on a Pixel 1 CPU core (ms)")
    ax.set_ylabel("ImageNet top-1 (%)")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"pareto_step{k}.svg")

# -----------------------------------------------------------------------------
# Recall: ResNet-50 convolutions on the H100 roofline (schematic, Lecture 11)
# -----------------------------------------------------------------------------
sp.roofline_steps(out / "recall_roof", [
    ("ResNet-50", 3000.0, "compute"),
], ridge=295.0, xticks=[1, 10, 100, 1000], ridge_label="ridge: 295",
    xlabel="FLOPs per byte (log scale)", note="Schematic", size=(4.2, 2.4))


# -----------------------------------------------------------------------------
# Fan-out: chance a request waits on at least one slow shard (footnote 20)
# -----------------------------------------------------------------------------
import math  # noqa: E402
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ns = list(range(1, 101))
    pr = [100 * (1 - 0.99 ** n) for n in ns]
    if k >= 2:
        ax.plot(ns, pr, color=C["danger"], lw=3)
        ax.plot(100, pr[-1], "o", ms=10, color=C["danger"])
        ax.text(97, pr[-1] + 4, f"100 shards: {pr[-1]:.0f}%", ha="right", fontsize=12,
                fontweight="bold", color=C["danger"])
        ax.plot(10, pr[9], "o", ms=8, color=C["text"])
        ax.text(14, 1.5, f"10 shards: {pr[9]:.1f}%", fontsize=11, color=C["text"],
                fontweight="bold")
    else:
        ax.text(100, 50, "???", ha="right", fontsize=16, fontweight="bold", color=C["danger"])
    ax.plot(1, 1, "o", ms=10, color=C["memory"])
    ax.text(2, 30, "1 shard: 1%", fontsize=11, color=C["memory"], fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 78)
    ax.set_xlabel("Shards per request (each slow 1%)")
    ax.set_ylabel("requests slowed (%)")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"fanout_step{k}.svg")

# -----------------------------------------------------------------------------
# Time to quality: one MLPerf Training v4.0 run (Eos, 8x H100, result_0)
# -----------------------------------------------------------------------------
import csv  # noqa: E402
ev = list(csv.DictReader(open(out / "mlperf_v40_eos_result0_evals.csv")))
tm = [float(r["minutes"]) for r in ev]
ac = [float(r["top1"]) for r in ev]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.15))
    ax.plot(tm, ac, "-o", color=C["memory"], lw=2.5, ms=7)
    ax.axhline(75.9, color=C["danger"], lw=1.5, ls="--")
    ax.text(0.3, 78.2, "target 75.9%", fontsize=11, color=C["danger"], fontweight="bold")
    if k >= 2:
        ax.plot(tm[-1], ac[-1], "o", ms=12, color=C["danger"], zorder=5)
        ax.annotate(f"time to quality: {tm[-1]:.2f} min\n(epoch 35, {ac[-1]:.2f}%)",
                    (tm[-1], ac[-1]), xytext=(-4, -78), textcoords="offset points",
                    ha="right", fontsize=11, fontweight="bold", color=C["danger"])
    ax.set_xlim(0, 14.2)
    ax.set_ylim(25, 84)
    ax.set_xlabel("Wall-clock minutes since run start")
    ax.set_ylabel("ImageNet top-1 (%)")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"ttq_step{k}.svg")

# -----------------------------------------------------------------------------
# Strong scaling: 8 against 64 H100s (MLPerf Training v4.0, NVIDIA Eos)
# -----------------------------------------------------------------------------
SCALE = 1.0042232277526395
eos8 = sorted([13.30620, 13.31780, 13.30162, 13.20308, 13.18598])
eos64 = sorted([2.50543, 2.50527, 2.50357, 2.50593]) + [math.inf]  # result_4 missed target
s8 = sum(eos8[1:-1]) / 3 * SCALE
s64 = sum(eos64[1:-1]) / 3 * SCALE
speed = s8 / s64
eff = speed / 8
ips8 = 35 * 1281167 / (s8 / SCALE * 60) / 8
ips64 = 35 * 1281167 / (s64 / SCALE * 60) / 64
print("scaling: %.2f -> %.2f min, %.2fx, efficiency %.0f%%, %.0f -> %.0f images/s per GPU" % (
    s8, s64, speed, 100 * eff, ips8, ips64))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.0, 1.75))
    rows = [("8 H100s", s8, "memory", f"{s8:.2f} min"),
            ("64 H100s", s64, "compute", f"{s64:.2f} min ({speed:.1f}×)")]
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 1 - i
        shown = (i == 0) or k >= 2
        ax.barh(y, v, color=C[ck], height=0.6, alpha=1.0 if shown else 0.0)
        ax.text(v + 0.2, y, txt, va="center", fontsize=12, fontweight="bold", color=C[ck],
                alpha=1.0 if shown else 0.0)
        if not shown:
            ax.text(0.3, y, "???", va="center", fontsize=15, fontweight="bold", color=C["danger"])
        ax.text(-0.25, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
    ax.plot([s8 / 8, s8 / 8], [-0.45, 0.45], color=C["text"], lw=2, ls=":",
            alpha=1.0 if k >= 2 else 0.0)
    ax.text(s8 / 8, -0.62, f"perfect 8×: {s8 / 8:.2f}", ha="center", va="top", fontsize=10,
            color=C["text"], alpha=1.0 if k >= 2 else 0.0)
    ax.set_xlim(0, 19.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim(-1.0, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Minutes to 75.9% top-1 (published)")
    sp.save(fig, out / f"scale_step{k}.svg")

# -----------------------------------------------------------------------------
# Where MLPerf Power draws the boundary (Sec. 12.7.1, figure 8)
# -----------------------------------------------------------------------------
tiers = [
    ("Tiny", ["compute cores", "on-chip SRAM", "power manager"], [], "at the DC rail"),
    ("Inference", ["host CPU, DRAM", "accelerators", "local storage", "fans"],
     ["remote storage"], "the whole node"),
    ("Training", ["compute nodes", "top-of-rack switches"],
     ["chillers", "facility PDUs", "storage cluster"], "nodes and switches"),
]
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(-7, 54)
    ax.axis("off")
    xs = [1, 34.5, 68]
    for i, (name, inside, outside, cap) in enumerate(tiers):
        x = xs[i]
        lit = (i == k - 1) or k == 4
        al = 1.0 if lit else 0.3
        ax.text(x + 15.5, 51, name, ha="center", va="center", fontsize=13,
                fontweight="bold", color=C["text"], alpha=al)
        ax.add_patch(Rectangle((x, 19), 31, 28.5, fc="none", ec="#A51C30", lw=2.6 if lit else 1.2,
                               alpha=al))
        for j, item in enumerate(inside):
            ax.add_patch(Rectangle((x + 2, 41 - j * 4.9), 27, 4.1, fc=C["compute_l"],
                                   ec=C["compute"], lw=0.8, alpha=al))
            ax.text(x + 15.5, 43.05 - j * 4.9, item, ha="center", va="center", fontsize=9.5,
                    color=C["text"], alpha=al)
        ax.text(x + 15.5, 22, cap, ha="center", va="center", fontsize=10,
                color="#A51C30", fontweight="bold", alpha=al)
        for j, item in enumerate(outside):
            ax.add_patch(Rectangle((x + 2, 12.5 - j * 5.6), 27, 4.6, fc="white",
                                   ec=C["neutral"], lw=0.8, ls="--", alpha=al))
            ax.text(x + 15.5, 14.8 - j * 5.6, item, ha="center", va="center",
                    fontsize=9.5, color=C["neutral"], alpha=al)
    ax.text(1, -5.5, "Crimson: measured boundary. Dashed: outside it.", fontsize=9.5,
            color=C["neutral"])
    sp.save(fig, out / f"powerbound_step{k}.svg")

print("figures written")
