#!/usr/bin/env python3
"""Figures for Lecture 12 (Benchmarking).

Every drawn value comes from MLSysIM through calcs.py, which this script
imports: the benchmarking Scenarios.* (datasheet gap, edge board, fan-out,
training ceiling, duty cycle, measurement boundary, MLPerf scores) and the
Literature series of published results (Dean and Barroso 2013; MLPerf
Training v4.0, Inference v5.0, and Tiny v1.2; Lee et al. 2019; Tummalapalli
et al. 2026). Labels print the values calcs.py records (deckvals.val), so a
figure and its slide always show the same digits. Values the chapter gives
as a scenario, composite, or assumption are drawn with an "Illustrative" tag.
The closed-loop, granularity, scenario, and power-boundary figures are
mechanism diagrams; their coordinates are layout, not data.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as K  # noqa: E402
from deckvals import val, num  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent
m = K.m
EFF_DS, EFF_SUS = m(K.CLAIM.claimed_efficiency, "TOPS/W"), m(K.CLAIM.sustained_efficiency, "TOPS/W")
CEIL, MEASURED = m(K.CEILING.ceiling_rate, "count/second"), m(K.CEILING.sustained_rate, "count/second")
B = K.BOARD


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
    dict(label=f"Datasheet: {val('hook_tops')} TOPS at {val('hook_w')} W", value=EFF_DS, color="compute",
         text=f"{val('hook_eff')} TOPS/W", step=1),
    dict(label=f"Sustained: {val('hook_sus_tops')} TOPS at {val('hook_sus_w')} W", value=EFF_SUS, color="danger",
         text=f"{val('hook_sus_eff')} TOPS/W", step=3, pending="???", pending_from=2),
]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    bars(ax, hook_rows, k, xmax=27)
    if k == 3:
        ax.annotate("", xy=(EFF_SUS, 0.5), xytext=(EFF_DS, 0.5),
                    arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.5))
        ax.text(EFF_DS + 0.5, 0.5, f"{val('hook_gap')}× lower", ha="left", va="center", fontsize=13,
                fontweight="bold", color=C["text"])
    ax.set_xlabel("Efficiency (TOPS per watt)")
    tag(ax, "Illustrative (Benchmarking, §12.7)", (0.0, -0.5))
    sp.save(fig, out / f"hook_step{k}.svg")

# -----------------------------------------------------------------------------
# Fan-out tail (Dean and Barroso 2013, Table 1) and the illustrative gap
# -----------------------------------------------------------------------------
tail_rows = [
    dict(label="One leaf, median", value=K.leaf_p50, color="memory", text=f"{val('leaf_p50')} ms", step=1),
    dict(label="One leaf, p99", value=K.leaf_p99, color="memory", text=f"{val('leaf_p99')} ms", step=1),
    dict(label="All leaves, median", value=K.all_p50, color="danger", text=f"{val('all_p50')} ms", step=2),
    dict(label="All leaves, p99", value=K.all_p99, color="danger",
         text=f"{val('all_p99')} ms  ({val('tail_ratio')}× one leaf)", step=2),
]
TAIL_LO, TAIL_HI = m(K.TAIL_GAP.tail_low, "ms"), m(K.TAIL_GAP.tail_high, "ms")
gap_rows = [
    dict(label="Model-only mean", value=m(K.TAIL_GAP.mean, "ms"), color="neutral",
         text=f"{val('tail_mean')} ms", step=3),
    dict(label="Production p99", value=TAIL_HI, color="danger",
         text=f"{val('tail_p99_lo')} to {val('tail_p99_hi')} ms", step=3),
]
for k in (1, 2, 3):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.8, 3.3),
                                  gridspec_kw=dict(height_ratios=[2.0, 1.0], hspace=0.75))
    bars(ax, tail_rows, k, xmax=215)
    ax.set_xlabel("Latency (ms), measured Google service")
    bars(ax2, gap_rows, k, xmax=215)
    if k >= 3:
        ax2.barh(0, TAIL_HI - TAIL_LO, left=TAIL_LO, color=C["danger_l"], height=0.62)
        deadline = num("frame_budget")
        ax2.plot([deadline, deadline], [-0.31, 0.31], color="white", lw=2)
        ax2.text(deadline + 3, 0, f"{val('frame_budget')} ms frame deadline", color="white", fontsize=10,
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
    dict(label="Roofline ceiling", value=CEIL, color="compute",
         text=f"{val('ceiling_ips')} images/s", step=1),
    dict(label="MLPerf, per H100", value=MEASURED, color="danger",
         text=f"about {val('measured_ips')} (chapter estimate)", step=3, pending="???", pending_from=2),
]
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    bars(ax, mfu_rows, k, xmax=56000)
    if k >= 3:
        ax.annotate("", xy=(MEASURED, 0.5), xytext=(CEIL, 0.5),
                    arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.4))
        ax.text(CEIL + 800, 0.5, f"{val('ceiling_gap')}× below", ha="left", va="center", fontsize=12,
                fontweight="bold", color=C["text"])
    if k >= 4:
        ax.plot([K.ips64, K.ips8], [-0.42, -0.42], color=C["text"], lw=2)
        for x in (K.ips64, K.ips8):
            ax.plot([x, x], [-0.52, -0.32], color=C["text"], lw=2)
        ax.text(K.ips8 + 550, -0.42, f"MLPerf v4.0 logs: {val('ips8')} (8 GPUs), "
                f"{val('ips64')} (64 GPUs)",
                va="center", fontsize=11, color=C["text"])
    ax.set_ylim(-0.75, 1.6)
    ax.set_xlabel("Training throughput (images/s per H100)")
    sp.save(fig, out / f"mfu_step{k}.svg")

# -----------------------------------------------------------------------------
# Edge board: isolated, end to end, energy (Lighthouse 12.2, illustrative)
# -----------------------------------------------------------------------------
for k in (1, 2, 3):
    # drawn for a 0.70-width column beside the camera-trap photos
    fig, axes = plt.subplots(1, 3, figsize=(4.3, 2.35),
                             gridspec_kw=dict(width_ratios=[3, 2, 2], wspace=0.5))
    a1, a2, a3 = axes
    iso = [m(B.host_fp32_latency, "ms"), m(B.host_int8_latency, "ms"), m(B.accelerator_latency, "ms")]
    a1.bar([0, 1, 2], iso, color=[C["data"], C["data"], C["compute"]], width=0.62)
    for x, v, key in zip([0, 1, 2], iso, ("edge_fp32", "edge_host", "edge_acc")):
        a1.text(x, v + 1.5, f"{val(key)} ms", ha="center", fontsize=10.5, fontweight="bold")
    a1.set_xticks([0, 1, 2])
    a1.set_xticklabels(["CPU\nFP32", "CPU\nINT8", "Acc.\nINT8"], fontsize=10)
    a1.set_ylim(0, 62)
    a1.set_xlim(-0.65, 2.5)
    a1.set_title("Model only", fontsize=11.5, fontweight="bold")
    a1.text(1.5, 30, f"{val('edge_iso')}×", ha="center", fontsize=13,
            fontweight="bold", color=C["compute"])
    a1.set_ylabel("ms per inference", fontsize=10)
    for ax, shown in ((a2, k >= 2), (a3, k >= 3)):
        al = 1.0 if shown else 0.0
        for sp_ in ax.spines.values():
            sp_.set_alpha(al)
        ax.tick_params(colors=C["text"] if shown else "white")
    al2 = 1.0 if k >= 2 else 0.0
    model = [m(B.host_int8_latency, "ms"), m(B.accelerator_latency, "ms")]
    e2e = [m(B.host_end_to_end, "ms"), m(B.accelerator_end_to_end, "ms")]
    a2.bar([0, 1], model, color=[C["data"], C["compute"]], width=0.62, alpha=al2)
    a2.bar([0, 1], [e2e[0] - model[0], e2e[1] - model[1]],
           bottom=model, color=C["network_l"], hatch="///",
           edgecolor=C["network"], width=0.62, alpha=al2)
    for x, v, key in zip([0, 1], e2e, ("edge_e2e_host", "edge_e2e_acc")):
        a2.text(x, v + 1.5, f"{val(key)} ms", ha="center", fontsize=10.5, fontweight="bold", alpha=al2)
    a2.text(0.5, 32, f"{val('edge_e2e')}×", ha="center", fontsize=13, fontweight="bold",
            color=C["compute"], alpha=al2)
    a2.text(0.5, 47, "hatched:\nprep + link", ha="center", fontsize=9, color=C["network"], alpha=al2)
    a2.set_ylim(0, 62)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["CPU", "Acc."], fontsize=10)
    a2.set_title("End to end", fontsize=11.5, fontweight="bold", alpha=al2)
    al3 = 1.0 if k >= 3 else 0.0
    ev_ = [m(B.host_energy, "mJ"), m(B.accelerator_energy, "mJ")]
    a3.bar([0, 1], ev_, color=[C["data"], C["compute"]], width=0.62, alpha=al3)
    for x, v in zip([0, 1], [val("edge_e_host"), val("edge_e_acc")]):
        a3.text(x, ev_[x] + 0.06, v, ha="center", fontsize=10.5, fontweight="bold", alpha=al3)
    a3.set_ylim(0, 2.5)
    a3.set_xlim(-0.6, 1.6)
    a2.set_xlim(-0.6, 1.6)
    a3.set_xticks([0, 1])
    a3.set_xticklabels(["CPU", "Acc."], fontsize=10)
    a3.set_title("Energy (mJ)", fontsize=11.5, fontweight="bold", alpha=al3)
    a3.set_yticks([0, 1, 2])
    tag(a1, "Illustrative (Lighthouse 12.2)", (0.0, -0.5))
    sp.save(fig, out / f"waterfall_step{k}.svg")

# -----------------------------------------------------------------------------
# Same phone, same model, different software (Lee et al. 2019)
# -----------------------------------------------------------------------------
lee_rows = [
    dict(label="iPhone Xs, TFLite GPU", value=K.xs_tflite, color="compute", text=f"{val('xs_tflite')} ms", step=1),
    dict(label="iPhone Xs, Metal (MPSCNN)", value=K.xs_metal, color="compute", text=f"{val('xs_metal')} ms", step=1),
    dict(label="iPhone Xs, Core ML", value=K.xs_coreml, color="compute",
         text=f"{val('xs_coreml')} ms  ({val('lee_xs')}×)", step=1),
    dict(label="Pixel 3, TFLite GPU", value=K.px_gpu, color="memory", text=f"{val('px_gpu')} ms", step=2),
    dict(label="Pixel 3, TFLite CPU", value=K.px_cpu, color="memory",
         text=f"{val('px_cpu')} ms  ({val('lee_px')}×)", step=2),
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
    a1.bar([0, 1], [K.rtx_serial, K.rtx_off], color=[C["memory"], C["compute"]], width=0.62)
    a1.text(0, K.rtx_serial + 1500, f"≈{val('rtx_serial')}\n(derived)", ha="center", fontsize=10, fontweight="bold",
            color=C["memory"])
    a1.text(1, K.rtx_off + 1500, val("rtx_off"), ha="center", fontsize=11, fontweight="bold",
            color=C["compute"])
    a1.text(0, 30000, f"{val('rtx_ratio')}×", ha="center", fontsize=13, fontweight="bold")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["Single-\nStream", "Offline"], fontsize=10)
    a1.set_ylim(0, 55000)
    a1.set_yticks([0, 20000, 40000])
    a1.set_yticklabels(["0", "20k", "40k"])
    a1.set_ylabel("images/s")
    a1.set_title("1× RTX 4090", fontsize=12, fontweight="bold")
    al = 1.0 if k >= 2 else 0.0
    xs = [0, 0.78, 1.95, 2.73]
    vals = [K.cisco_off, K.cisco_srv, K.quanta_off, K.quanta_srv]
    cols = [C["compute"], C["danger"], C["compute"], C["danger"]]
    a2.bar(xs, vals, color=cols, width=0.6, alpha=al)
    kilo = [f"{v / 1000:.0f}k" for v in vals]
    for x, v, lab in zip(xs, vals, [kilo[0], f"{kilo[1]}\n−{val('cisco_drop')}%", kilo[2],
                                          f"{kilo[3]}\n−{val('quanta_drop')}%"]):
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
    fig, ax = plt.subplots(figsize=(5.8, 1.65))
    narrow, full = m(K.GEMINI.narrow, "Wh"), m(K.GEMINI.full, "Wh")
    ax.barh(1, narrow, color=C["compute"], height=0.55)
    ax.text(narrow + 0.003, 1, f"{val('gem_narrow')} Wh", va="center", fontsize=12, fontweight="bold",
            color=C["compute"])
    parts = [(full * K.GEMINI.share("Active AI"), C["compute"], f"accelerators {val('gem_accel')}%"),
             (full * K.GEMINI.share("Active host"), C["memory"], f"host CPU, DRAM {val('gem_host')}%"),
             (full * K.GEMINI.share("Idle", "Data center"), C["neutral"], f"idle, facility {val('gem_idle')}%")]
    left = 0
    for i, (w, col, lab) in enumerate(parts):
        c = col if k >= 2 else C["danger"]
        ax.barh(0, w, left=left, color=c, height=0.55)
        if k >= 2:
            ax.text(left + w / 2, -0.52 if i != 1 else 0.48, lab, ha="center",
                    va="center", fontsize=10, color=col, fontweight="bold")
        left += w
    ax.text(full + 0.003, 0, f"{val('gem_full')} Wh ({val('gem_ratio')}×)", va="center", fontsize=12, fontweight="bold",
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
    # drawn for a 0.70-width column beside the phone photos
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.4, 2.4),
                                 gridspec_kw=dict(wspace=0.5, width_ratios=[1.5, 1]))
    groups = [("iPhone\nCPU", K.xs_cpu["start_ms"], K.xs_cpu["end_ms"], f"+{val('thr_cpu')}%"),
              ("iPhone\nGPU", K.xs_gpu["start_ms"], K.xs_gpu["end_ms"], f"+{val('thr_gpu')}%"),
              ("Pixel 3\nGPU", K.px3["start_ms"], K.px3["end_ms"], "flat")]
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
    a1.set_title("iPhone Xs, Pixel 3 (approx.)", fontsize=10.5, fontweight="bold")
    al = 1.0 if k >= 2 else 0.0
    toks = [K.tok["peak"], K.tok["warm"], K.tok["hot"]]
    a2.bar([0, 1, 2], toks, color=[C["memory"], C["compute"], C["danger"]],
           width=0.62, alpha=al)
    for x, v in zip([0, 1, 2], [val("tok_peak"), val("tok_warm"), val("tok_hot")]):
        a2.text(x, toks[x] + 1.2, v, ha="center", fontsize=10,
                fontweight="bold", alpha=al)
    a2.text(2, 33, f"−{val('tok_loss')}%", ha="center", fontsize=10,
            fontweight="bold", color=C["danger"], alpha=al)
    a2.set_xticks([0, 1, 2])
    a2.set_xticklabels(["peak", "warm", "hot"], fontsize=10)
    a2.set_ylim(0, 52)
    a2.set_ylabel("tokens/s")
    a2.set_title("iPhone 16 Pro, LLM", fontsize=10.5, fontweight="bold", alpha=al)
    for s in a2.spines.values():
        s.set_alpha(al)
    a2.tick_params(colors=C["text"] if k >= 2 else "white")
    a2.yaxis.label.set_alpha(al)
    sp.save(fig, out / f"throttle_step{k}.svg")

# -----------------------------------------------------------------------------
# MLPerf Tiny v1.2, visual wake words: energy per inference
# -----------------------------------------------------------------------------
tiny_rows = [
    dict(label=f"NDP9120, {val('syn_lo_mhz')} MHz", value=K.syn_lo["energy_uj"], color="data",
         text=f"{val('syn_lo_uj')} µJ", step=1),
    dict(label=f"NDP9120, {val('syn_hi_mhz')} MHz", value=K.syn_hi["energy_uj"], color="data",
         text=f"{val('syn_hi_uj')} µJ  ({val('tiny_speed')}× faster)", step=1),
    dict(label="STM32U575", value=K.stm["NUCLEO-U575ZI-Q"], color="neutral", text=f"{val('stm_lo')} µJ", step=2),
    dict(label="STM32H7A3", value=K.stm["NUCLEO-H7A3ZI-Q"], color="neutral", text=f"{val('stm_mid')} µJ", step=2),
    dict(label="STM32L4R5", value=K.stm["NUCLEO-L4R5ZI"], color="neutral",
         text=f"{val('stm_hi')} µJ  ({val('tiny_worst')}×)", step=2),
]
for k in (1, 2):
    # drawn for a 0.70-width column beside the Nucleo photo
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    bars(ax, tiny_rows, k, log=True)
    ax.set_xlim(20, 200000)
    ax.set_xlabel("Energy per inference (µJ, log scale)")
    sp.save(fig, out / f"tiny_step{k}.svg")

# -----------------------------------------------------------------------------
# Duty cycle: average power of a keyword spotter (Sec. 12.7.2)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 0.75))
IDLE_MW, ACTIVE_MW = m(K.KWS.idle_contribution, "mW"), m(K.KWS.active_contribution, "mW")
ax.barh(0, IDLE_MW, color=C["neutral"], height=0.8)
ax.barh(0, ACTIVE_MW, left=IDLE_MW, color=C["compute"], height=0.8)
ax.text(IDLE_MW / 2, 0, f"idle {val('kws_idle_mw')} mW ({val('kws_idle_share')}%)", ha="center", va="center", color="white",
        fontsize=13, fontweight="bold")
ax.text(54, 0, f"+ active {val('kws_active_mw')} mW = {val('kws_avg_mw')} mW", va="center", fontsize=13,
        fontweight="bold", color=C["compute"])
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 0.5)
ax.axis("off")
sp.save(fig, out / "duty.svg")

# -----------------------------------------------------------------------------
# Recall: ResNet-50 convolutions on the H100 roofline (schematic, Lecture 11)
# -----------------------------------------------------------------------------
sp.roofline_steps(out / "recall_roof", [
    ("ResNet-50", 3000.0, "compute"),
], ridge=num("h100_ridge"), xticks=[1, 10, 100, 1000], ridge_label=f"ridge: {val('h100_ridge')}",
    xlabel="FLOPs per byte (log scale)", note="Schematic", size=(4.2, 2.4))


# -----------------------------------------------------------------------------
# Fan-out: chance a request waits on at least one slow shard (footnote 20)
# -----------------------------------------------------------------------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    N, N10 = K.FANOUT.n_components, K.fan10.n_components
    ns = list(range(1, N + 1))
    pr = [100 * (1 - K.FANOUT.variant(n_components=n).all_up) for n in ns]
    if k >= 2:
        ax.plot(ns, pr, color=C["danger"], lw=3)
        ax.plot(N, pr[-1], "o", ms=10, color=C["danger"])
        ax.text(97, pr[-1] + 4, f"{val('shards')} shards: {val('any_slow_pct')}%", ha="right", fontsize=12,
                fontweight="bold", color=C["danger"])
        ax.plot(N10, pr[N10 - 1], "o", ms=8, color=C["text"])
        ax.text(14, 1.5, f"{val('shards10')} shards: {val('any_slow10_pct')}%", fontsize=11, color=C["text"],
                fontweight="bold")
    else:
        ax.text(100, 50, "???", ha="right", fontsize=16, fontweight="bold", color=C["danger"])
    ax.plot(1, pr[0], "o", ms=10, color=C["memory"])
    ax.text(2, 30, f"1 shard: {val('shard_slow_pct')}%", fontsize=11, color=C["memory"], fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 78)
    ax.set_xlabel(f"Shards per request (each slow {val('shard_slow_pct')}%)")
    ax.set_ylabel("requests slowed (%)")
    ax.grid(color=C["grid"], lw=0.6)
    sp.save(fig, out / f"fanout_step{k}.svg")

# -----------------------------------------------------------------------------
# Time to quality: one MLPerf Training v4.0 run (Eos, 8x H100, result_0)
# -----------------------------------------------------------------------------
tm = [r["minutes"] for r in K.EVALS]
ac = [r["top1"] for r in K.EVALS]
TARGET = num("target_top1")
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.15))
    ax.plot(tm, ac, "-o", color=C["memory"], lw=2.5, ms=7)
    ax.axhline(TARGET, color=C["danger"], lw=1.5, ls="--")
    ax.text(0.3, 78.2, f"target {val('target_top1')}%", fontsize=11, color=C["danger"], fontweight="bold")
    if k >= 2:
        ax.plot(tm[-1], ac[-1], "o", ms=12, color=C["danger"], zorder=5)
        ax.annotate(f"time to quality: {val('ev_last_min')} min\n(epoch {val('ev_last_epoch')}, {val('ev_last_top1')}%)",
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
# run times, middle-three means, and per-GPU rates: Scenarios.EosResNetStrongScalingMLPerfV40
s8, s64 = m(K.EOS8.published_score, "minute"), m(K.EOS64.published_score, "minute")
ideal = m(K.SCALING.ideal_time, "minute")
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.0, 1.75))
    rows = [("8 H100s", s8, "memory", f"{val('score8')} min"),
            ("64 H100s", s64, "compute", f"{val('score64')} min ({val('scale_speed')}×)")]
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 1 - i
        shown = (i == 0) or k >= 2
        ax.barh(y, v, color=C[ck], height=0.6, alpha=1.0 if shown else 0.0)
        ax.text(v + 0.2, y, txt, va="center", fontsize=12, fontweight="bold", color=C[ck],
                alpha=1.0 if shown else 0.0)
        if not shown:
            ax.text(0.3, y, "???", va="center", fontsize=15, fontweight="bold", color=C["danger"])
        ax.text(-0.25, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
    ax.plot([ideal, ideal], [-0.45, 0.45], color=C["text"], lw=2, ls=":",
            alpha=1.0 if k >= 2 else 0.0)
    ax.text(ideal, -0.62, f"perfect {val('scale_ratio')}×: {val('scale_ideal')}", ha="center", va="top", fontsize=10,
            color=C["text"], alpha=1.0 if k >= 2 else 0.0)
    ax.set_xlim(0, 19.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_ylim(-1.0, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Minutes to {val('target_top1')}% top-1 (published)")
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

# -----------------------------------------------------------------------------
# Synthesis recall: the hook's two bars, drawn at their slot (0.70 width,
# 3.1 cm tall) so every label stays at slide size.
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.4, 1.15))
ax.barh(1, EFF_DS, color=C["compute"], height=0.6)
ax.barh(0, EFF_SUS, color=C["danger"], height=0.6)
ax.text(EFF_DS + 0.4, 1, f"{val('hook_eff')} TOPS/W", va="center", fontsize=11,
        fontweight="bold", color=C["compute"])
ax.text(EFF_SUS + 0.4, 0, f"{val('hook_sus_eff')} TOPS/W", va="center", fontsize=11,
        fontweight="bold", color=C["danger"])
ax.text(12.5, 0, f"{val('hook_gap')}× lower", va="center", fontsize=11,
        fontweight="bold", color=C["text"])
ax.text(-0.4, 1, "Datasheet", ha="right", va="center", fontsize=11, fontweight="bold")
ax.text(-0.4, 0, "Sustained", ha="right", va="center", fontsize=11, fontweight="bold")
ax.set_xlim(0, 27)
ax.set_ylim(-0.5, 1.5)
ax.axis("off")
sp.save(fig, out / "hook_recall.svg")

# -----------------------------------------------------------------------------
# Text-free thumbnails for the goals, takeaways, and next-lecture slots.
# Each is a pictogram of the figure that teaches the idea, drawn at its slot
# size (2.5 by 1.05 cm; the next-lecture slot 7.4 by 4.6 cm), so no text is
# shrunk below the floor.
# -----------------------------------------------------------------------------
def thumb(name, draw, size=(1.0, 0.42)):
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95)
    draw(ax)
    ax.axis("off")
    fig.savefig(out / f"{name}.svg", format="svg", pad_inches=0.0, facecolor="white")
    plt.close(fig)


def _tail(ax):          # a short blue bar and a long crimson bar
    ax.barh([1, 0], [1.5, 8], color=[C["memory"], C["danger"]], height=0.6)
    ax.set_xlim(0, 9); ax.set_ylim(-0.6, 1.6)


def _gran(ax):          # nested timed regions
    for i, w in enumerate([2.0, 4.0, 6.0, 8.0]):
        ax.add_patch(Rectangle((0.2, 0.2 + 0.0 * i), w, 1.6, fc="none",
                               ec="#A51C30" if i == 3 else C["neutral"], lw=1.6 if i == 3 else 1.0))
    ax.add_patch(Rectangle((0.5, 0.5), 1.4, 1.0, fc=C["compute"], lw=0))
    ax.set_xlim(0, 8.5); ax.set_ylim(0, 2)


def _boundary(ax):      # a short bar against a longer stacked bar
    ax.barh(1, 4.2, color=C["compute"], height=0.6)
    for left, w, c in ((0, 5.8, "compute"), (5.8, 2.5, "memory"), (8.3, 1.7, "neutral")):
        ax.barh(0, w, left=left, color=C[c], height=0.6)
    ax.set_xlim(0, 10.2); ax.set_ylim(-0.6, 1.6)


def _ttq(ax):           # accuracy climbing to a dashed target
    ax.plot(tm, ac, "-", color=C["memory"], lw=2)
    ax.axhline(TARGET, color=C["danger"], lw=1.2, ls="--")
    ax.set_ylim(25, 82)


def _mfu(ax):           # ceiling bar, measured bar, gap arrow
    ax.barh([1, 0], [8, 1], color=[C["compute"], C["danger"]], height=0.6)
    ax.annotate("", xy=(1.1, 0.5), xytext=(8, 0.5),
                arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.0))
    ax.set_xlim(0, 9); ax.set_ylim(-0.6, 1.6)


def _scen(ax):          # one short bar, one tall bar
    ax.bar([0, 1], [1, 12.5], color=[C["memory"], C["compute"]], width=0.6)
    ax.set_xlim(-0.8, 1.8); ax.set_ylim(0, 14)


def _throttle(ax):      # paired bars, the second of each pair taller
    for i, (a, b) in enumerate(((5, 6.5), (1.5, 2.5), (4.5, 4.5))):
        ax.bar(i - 0.17, a, width=0.32, color=C["memory"])
        ax.bar(i + 0.17, b, width=0.32, color=C["danger"])
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 7.5)


thumb("thumb_tail", _tail)
thumb("thumb_gran", _gran)
thumb("thumb_boundary", _boundary)
thumb("thumb_ttq", _ttq)
thumb("thumb_mfu", _mfu)
thumb("thumb_scen", _scen)
thumb("thumb_throttle", _throttle)


def _next(ax):          # two systems, ten runs each, overlapping spreads (schematic)
    # schematic, evenly spaced: not data
    for y, c, mid in ((1, C["memory"], 5.0), (0, C["compute"], 5.15)):
        xs = [mid - 0.5 + 0.1 * i for i in range(10)]
        ax.plot([min(xs), max(xs)], [y, y], color=c, lw=10, alpha=0.3, solid_capstyle="round")
        ax.plot(xs, [y] * 10, "o", ms=13, color=c, mec="white", mew=1.5, clip_on=False)
    ax.set_xlim(4.35, 5.8); ax.set_ylim(-0.7, 1.7)


thumb("thumb_next", _next, size=(2.9, 1.8))

print("figures written")
