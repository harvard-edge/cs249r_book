#!/usr/bin/env python3
"""Figures for Lecture 13 (Model Serving, Part 1).

Every value comes from calcs.py (imported as c), which reads MLSysIM: the
ResNet-50 H100 serving profile and the model-serving Scenarios.* the Model
Serving chapter also reads, the cited results in Literature.ServingResults,
and the MLPerf Inference v5.0 rows in Literature.Series. Figure code takes
plotted values from those objects and printed labels from deckvals.val(), so
a label shows exactly the digits the slide text shows. The ResNet-50 numbers
are the chapter's modeled reference profile on the H100 (not measurements);
its host phases are stated assumptions. The AI Tax pie (Richins et al.,
arXiv 2007.10571, Fig. 6) is the paper's own figure, aitax_fig6.png.
Arrival marks in the batching-window figure are illustrative; thumbnails at
the end are text-free pictograms for the goals and takeaways slots.

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as c  # noqa: E402
from deckvals import val, num  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent
m = c.m


def tag(ax, text, x=0.0, y=-0.30):
    """A small gray provenance tag under the axes (modeled, derived ...)."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.5,
            color=C["neutral"], ha="left", va="top")


# --------------------------------------------------------------- hook cost
sp.hbars_steps(out / "hook_cost", [
    (f"Batch {c.B1}  ({val('b1_ms')} ms)", num("b1_usd"), "danger", f"\\${val('b1_usd')}"),
    (f"Batch {c.B8}  ({val('b8_ms')} ms)", num("b8_usd"), "compute", f"\\${val('b8_usd')}"),
], "", log=False, xlabel=f"Cost per million requests (dollars; modeled, \\${val('price_h100')} per GPU-hour)",
    size=sp.SHORT)

# ------------------------------------------------- recall: Lecture 12 bars
sp.hbars(out / "recall_l12.svg", [
    ("One at a time", num("rtx_ss_rate") / 1000, "neutral", f"about {val('rtx_ss_rate')}/s"),
    ("All at once", num("rtx_off") / 1000, "compute", f"{val('rtx_off')}/s"),
], "", log=False, xlabel="ResNet-50, thousand images per second, one RTX 4090",
    size=(5.0, 1.5))

# ------------------------------------------------ request budget (Amdahl)
ST = {k: m(t, "ms") for k, t in c.PATH.stages.items()}
PRE, COPY, FWD, POST = m(c.PATH.subtotal(*c.HOST), "ms"), ST["copy to GPU"], ST["forward pass"], ST["top-5"]
stages_now = [("Preprocess", PRE, "data"), ("Copy", COPY, "network"),
              ("Model", FWD, "compute"), ("Post", POST, "neutral")]
stages_fast = [("Preprocess", PRE, "data"), ("Copy", COPY, "network"),
               ("Model", m(c.PATH.accelerated_stage, "ms"), "compute"), ("Post", POST, "neutral")]


def stacked_row(ax, y, parts, show=True, labels=True, fs=11):
    left = 0.0
    for name, v, ck in parts:
        ax.barh(y, v, left=left, color=C[ck], height=0.55, edgecolor="white",
                linewidth=1.5, alpha=1.0 if show else 0.0)
        if labels and show and v >= 0.9 * (sum(p[1] for p in parts) / 9):
            ax.text(left + v / 2, y, name, ha="center", va="center",
                    color="white", fontsize=fs, fontweight="bold")
        left += v
    return left


for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    total = stacked_row(ax, 1, stages_now)
    ax.text(total + 0.12, 1, f"{val('budget_total')} ms", va="center", fontsize=13,
            fontweight="bold", color=C["text"])
    t2 = stacked_row(ax, 0, stages_fast, labels=True)
    ax.text(t2 + 0.12, 0, "???" if k == 1 else
            f"{val('fast_total')} ms: {val('fast_speedup')}× faster", va="center",
            fontsize=13, fontweight="bold", color=C["danger"])
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Today", f"Model {val('model_speedup')}× faster"], fontsize=12, fontweight="bold")
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("One ResNet-50 request (ms; modeled model, assumed host phases)",
                  fontsize=11)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    sp.save(fig, out / f"budget_step{k}.svg")

# ------------------------------------- request timeline (book variant)
# Slide-size variant of the chapter's request-timeline figure (MS figure 3,
# table 3): host phases assumed, forward pass modeled.
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    lanes = {"Host CPU": 2, "PCIe": 1, "GPU": 0}
    for name, y in lanes.items():
        ax.axhspan(y - 0.38, y + 0.38, color="#F2F2F2", lw=0, zorder=0)
    DEC, RSZ, NRM = ST["JPEG decode"], ST["resize"], ST["normalize"]
    host = [(f"decode {val('st_decode')} ms", 0, DEC), ("", DEC, RSZ), ("", DEC + RSZ, NRM)]
    for lab, x0, w in host:
        ax.barh(2, w, left=x0, color=C["data_l"], edgecolor=C["data"], lw=1.5, height=0.56)
        if lab:
            ax.text(x0 + w / 2, 2, lab, ha="center", va="center", fontsize=11)
    ax.text(DEC + RSZ / 2, 2.36, f"resize {val('st_resize')}", ha="center", va="bottom", fontsize=11)
    ax.text(DEC + RSZ + NRM / 2, 2.78, f"normalize {val('st_norm')}", ha="center", va="bottom", fontsize=11)
    if k >= 2:
        ax.barh(1, COPY, left=PRE, color=C["network_l"], edgecolor=C["network"], lw=1.5,
                height=0.56)
        ax.text(PRE + COPY + 0.1, 1, f"copy {val('st_copy')} ms", va="center", fontsize=11)
        ax.barh(0, FWD, left=PRE + COPY, color=C["compute_l"], edgecolor=C["compute"], lw=1.5,
                height=0.56)
        ax.text(PRE + COPY + FWD / 2, 0, f"forward\n{val('b1_ms')} ms", ha="center", va="center", fontsize=11)
        ax.barh(2, POST, left=PRE + COPY + FWD, color=C["data_l"], edgecolor=C["data"], lw=1.5,
                height=0.56)
        ax.text(PRE + COPY + FWD + POST / 2, 2.5, f"top-5 {val('st_post')}", ha="center", va="bottom", fontsize=11)
    if k >= 3:
        ax.barh(0, PRE + COPY, left=0, color="white", edgecolor=C["danger"], hatch="//",
                lw=1.5, height=0.56)
        ax.text((PRE + COPY) / 2, 0, f"GPU idle {val('gpu_idle')} ms", ha="center", va="center", fontsize=12,
                color=C["danger"], fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(list(lanes), fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.5, 3.25)
    ax.set_xticks(range(10))
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 10, 2))
    ax.set_xlabel(f"ms since the request arrived ({val('budget_total')} ms total)", fontsize=11)
    tag(ax, "After Model Serving, fig. 3: forward pass modeled, host assumed", y=-0.34)
    sp.save(fig, out / f"timeline_step{k}.svg")

# ------------------------------------------------------ pipelining variant
for k in (1, 2):
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 2.9), gridspec_kw={"hspace": 0.9})
    # serial: each request runs 8.6 ms end to end before the next starts
    ax = axes[0]
    TOT = PRE + COPY + FWD + POST
    for i in range(3):
        t0 = i * TOT
        ax.barh(1, PRE, left=t0, color=C["data"], height=0.6, edgecolor="white")
        ax.barh(1, COPY, left=t0 + PRE, color=C["network"], height=0.6, edgecolor="white")
        ax.barh(0, FWD, left=t0 + PRE + COPY, color=C["compute"], height=0.6, edgecolor="white")
        ax.barh(1, POST, left=t0 + PRE + COPY + FWD, color=C["neutral"], height=0.6)
    ax.text(26.2, 0, f"GPU busy {val('model_share')}%", va="center", fontsize=12, fontweight="bold",
            color=C["compute"])
    ax.set_title("Serial: one request at a time", fontsize=12, loc="left",
                 color=C["text"], fontweight="bold")
    ax2 = axes[1]
    show = k == 2
    for i in range(6):
        ax2.barh(1, PRE, left=i * PRE, color=C["data"], height=0.6,
                 edgecolor="white", alpha=1.0 if show else 0.0)
        if PRE + COPY + i * PRE + FWD <= 26.0:
            ax2.barh(0, FWD, left=PRE + COPY + i * PRE, color=C["compute"], height=0.6,
                     edgecolor="white", alpha=1.0 if show else 0.0)
    ax2.text(26.2, 0, f"GPU busy {val('pipe_duty')}%", va="center", fontsize=12, fontweight="bold",
             color=C["compute"], alpha=1.0 if show else 0.0)
    ax2.set_title("Pipelined: host prepares the next request", fontsize=12,
                  loc="left", color=C["text"], fontweight="bold",
                  alpha=1.0 if show else 0.0)
    for a in axes:
        a.set_yticks([1, 0])
        a.set_yticklabels(["Host CPU", "GPU"], fontsize=11, fontweight="bold")
        a.set_xlim(0, 34)
        a.set_ylim(-0.5, 1.5)
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", length=0)
    axes[1].set_xlabel("Time (ms; modeled, book reference request)", fontsize=10)
    if not show:
        for t in ax2.get_yticklabels():
            t.set_alpha(0.0)
    sp.save(fig, out / f"pipeline_step{k}.svg")

# ------------------------------------------------------- knee (variant)
# Five steps: the table 7 points one at a time, a predict pause before 90
# percent, then the p99 curve and the 70 percent point.
S = m(c.PLAN.service_time, "ms")  # the book's reference service time, printed as 3.5 ms
U = c.PLAN.utilization_points  # 0.5, 0.7, 0.8, 0.9, 0.95
U_EX = c.PLAN.example_utilization
KNEE = {u: num(f"knee_{100 * u:.0f}") for u in U}
points = [(u, KNEE[u]) for u in U if u != U_EX]
reach = {1: points[0][0], 2: points[1][0], 3: points[1][0], 4: 0.97, 5: 0.97}
for k in (1, 2, 3, 4, 5):
    fig, ax = plt.subplots(figsize=(5.8, 2.75))
    ax.axvspan(100 * U_EX, 100, color=C["danger_l"], alpha=0.6, lw=0)
    rho = np.linspace(0.0, reach[k], 400)
    ax.plot(rho * 100, S / (1 - rho), color=C["memory"], lw=3)
    shown = {1: 1, 2: 2, 3: 2, 4: 4, 5: 4}[k]
    for r, w in points[:shown]:
        ax.plot([r * 100], [w], "o", ms=8, color=C["memory"], mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{w:g} ms", (r * 100, w), xytext=(-8, -14 if w > 60 else 4),
                    textcoords="offset points", ha="right", fontsize=11,
                    color=C["memory"], fontweight="bold")
    if k == 3:
        u9, w9 = points[2]
        ax.plot([100 * u9], [w9], "o", ms=9, mfc="white", mec=C["danger"], mew=2, zorder=5)
        ax.annotate(f"{100 * u9:.0f}%: ???", (100 * u9, w9), xytext=(-10, 4), textcoords="offset points",
                    ha="right", fontsize=12, color=C["danger"], fontweight="bold")
    ax.text(8, 8, "Mean", color=C["memory"], fontsize=12, fontweight="bold")
    if k >= 5:
        rr = np.linspace(0.0, 0.97, 400)
        ax.plot(rr * 100, c.PLAN.tail_factor * S / (1 - rr), color=C["danger"], lw=3)
        ax.text(30, 30, "p99", color=C["danger"], fontsize=12, fontweight="bold")
        ax.plot([100 * U_EX], [num("knee_p99_ex")], "o", ms=9, color=C["danger"], mec="white", mew=1.5, zorder=6)
        ax.annotate(f"{val('util_70')}%: p99 {val('knee_p99_ex')} ms,\nmean {val('knee_70')} ms",
                    (100 * U_EX, num("knee_p99_ex")), xytext=(-10, 6),
                    textcoords="offset points", ha="right", fontsize=11,
                    color=C["danger"], fontweight="bold")
    ax.text(99, 2, "planning danger zone", fontsize=10, color=C["danger"], ha="right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Utilization (percent of time the GPU is busy)")
    ax.set_ylabel("Time in system (ms)")
    tag(ax, f"M/M/1 model, {val('b1_ms')} ms service time (Model Serving, table 7)", y=-0.27)
    sp.save(fig, out / f"knee_step{k}.svg")

# ------------------------------------------- faster-model vote thumbnail
lam = np.linspace(0, 270, 400)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    s_old, s_new = m(c.VOTE.slow_service, "ms"), m(c.VOTE.fast_service, "ms")
    w_old = s_old / (1 - lam * s_old / 1000)
    ax.plot(lam, w_old, color=C["memory"], lw=3)
    ax.text(145, 30, f"{val('b1_ms')} ms model", color=C["memory"], fontsize=11, fontweight="bold")
    lam80 = m(c.VOTE.arrival_rate, "1/s")
    ax.plot([lam80], [num("vote_old")], "o", ms=10, color=C["danger"], mec="white", mew=1.5, zorder=5)
    ax.annotate(f"{val('vote_util')}% busy: {val('vote_old')} ms", (lam80, num("vote_old")), xytext=(15, 22),
                ha="left", fontsize=11, color=C["danger"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C["danger"]))
    if k == 2:
        w_new = s_new / (1 - lam * s_new / 1000)
        ax.plot(lam, w_new, color=C["data"], lw=3)
        ax.plot([lam80], [num("vote_new")], "o", ms=10, color=C["data"], mec="white", mew=1.5, zorder=5)
        ax.text(lam80 + 8, -4.5, f"{val('fast_service_ms')} ms model, {val('vote_util_new')}% busy: {val('vote_new')} ms", ha="right",
                va="center", fontsize=11, color=C["data"], fontweight="bold")
    ax.set_xlim(0, 270)
    ax.set_ylim(-8, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Requests per second")
    ax.set_ylabel("Mean time (ms)")
    tag(ax, "M/M/1 model", y=-0.30)
    sp.save(fig, out / f"vote_fast_step{k}.svg")

# ------------------------------------------ MLPerf Server against Offline
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    rows = [("System A", c.A_OFF, c.A_SRV), ("System B", c.B_OFF, c.B_SRV)]
    for i, (name, off, srv) in enumerate(rows):
        y = 1 - i
        ax.barh(y + 0.18, off / 1000, height=0.34, color=C["neutral"])
        ax.text(off / 1000 + 8, y + 0.18, f"Offline {off:,.0f}", va="center", fontsize=11,
                color=C["neutral"], fontweight="bold")
        ax.barh(y - 0.18, srv / 1000, height=0.34, color=C["compute"],
                alpha=1.0 if k == 2 else 0.0)
        ax.text(srv / 1000 + 8, y - 0.18, f"Server {srv:,.0f}  ({srv / off:.0%} kept)",
                va="center", fontsize=11, color=C["compute"], fontweight="bold",
                alpha=1.0 if k == 2 else 0.0)
        ax.text(-10, y, name, ha="right", va="center", fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 1050)
    ax.set_ylim(-0.5, 1.5)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ResNet-50, thousands per second, 8x H100-SXM (MLPerf Inference v5.0)")
    sp.save(fig, out / f"mlperf_server_step{k}.svg")

# --------------------------------------------- fixed cost per image (table 8)
# Time per image against batch size on log axes: the fixed cost divided by the
# batch, falling toward the per-image work (the chapter's profile, calcs.py).
bb = np.unique(np.round(np.logspace(0, np.log10(c.B128), 80)).astype(int))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    PER = m(c.SVC.per_request_time, "ms")
    ax.axhline(PER, color=C["compute"], lw=2, ls="--")
    ax.text(1.05, PER * 1.25, f"work on the image itself: {val('per_img_ms')} ms",
            fontsize=11, color=C["compute"], fontweight="bold", va="bottom")
    if k == 2:
        ax.plot(bb, [m(c.SVC.time_per_request(int(b)), "ms") for b in bb], color=C["neutral"], lw=3)
    pts = (c.B1,) if k == 1 else (c.B1, c.B8, c.B32)
    offs = {c.B1: (10, -4), c.B8: (10, 2), c.B32: (10, 2)}
    for b in pts:
        v = num(f"per_image_{b}")
        ax.plot([b], [v], "o", ms=9, color=C["danger"] if b == 1 else C["text"],
                mec="white", mew=1.5, zorder=5)
        ax.annotate(f"batch {b}: {val(f'per_image_{b}')} ms ({val(f'fixed_share_{b}')}% fixed)", (b, v),
                    xytext=offs[b], textcoords="offset points", fontsize=11,
                    fontweight="bold", color=C["danger"] if b == 1 else C["text"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(0.8, 160)
    ax.set_ylim(0.02, 6)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3])
    ax.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
    ax.minorticks_off()
    ax.set_xlabel("Batch size")
    ax.set_ylabel("ms per image")
    tag(ax, f"Modeled: {val('fixed_ms')} ms fixed per pass + {val('per_img_ms')} ms per image (Model Serving, table 8)",
        y=-0.36)
    sp.save(fig, out / f"fixed_cost_step{k}.svg")

# ------------------------------------- modeled throughput and measured L4
bs = np.unique(np.round(np.logspace(0, np.log10(c.B128), 60)).astype(int))
for k in (1, 2):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.9, 2.6),
                                 gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.55})
    tput = np.array([m(c.SVC.throughput(int(b)), "1/s") for b in bs])
    ax.plot(bs, tput / 1000, color=C["compute"], lw=3)
    for b in (c.B1, c.B32, c.B128):
        lab = f"{b}: {val(f'curve_ms_{b}')} ms\n{val(f'curve_rate_{b}')}/s"
        v = m(c.SVC.throughput(b), "1/s") / 1000
        ax.plot([b], [v], "o", ms=7, color=C["compute"], mec="white", zorder=5)
        ax.annotate(lab, (b, v), xytext={c.B1: (12, 32), c.B32: (-8, 14), c.B128: (-8, 4)}[b],
                    textcoords="offset points", ha="right" if b > 1 else "left",
                    fontsize=10, color=C["text"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 128])
    ax.set_xticklabels(["1", "8", "32", "128"])
    ax.set_ylim(0, 20)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Thousand img/s")
    ax.set_title("H100, modeled", fontsize=12, loc="left", fontweight="bold")
    show = k == 2
    a = 1.0 if show else 0.0
    rows = [("1 at a time", num("l4_one") / 1000, "neutral", f"{val('l4_one')}/s"),
            (f"{val('l4_per_query')} at a time", num("l4_multi") / 1000, "compute", f"{val('l4_multi')}/s"),
            ("All at once", num("l4_each") / 1000, "compute", f"{val('l4_each')}/s")]
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 2 - i
        bx.barh(y, v, color=C[ck], height=0.6, alpha=a)
        bx.text(v + 0.3, y, txt, va="center", fontsize=10, fontweight="bold",
                color=C[ck], alpha=a)
        bx.text(-0.3, y, lab, ha="right", va="center", fontsize=10,
                fontweight="bold", alpha=a)
    bx.set_yticks([])
    bx.set_xlim(0, 20)
    bx.set_ylim(-0.5, 2.5)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("Thousand img/s", alpha=a)
    bx.set_title("L4, measured", fontsize=12, loc="left", fontweight="bold", alpha=a)
    if not show:
        bx.spines["bottom"].set_alpha(0.0)
        bx.tick_params(axis="x", colors="white")
    sp.save(fig, out / f"batch_curve_step{k}.svg")

# --------------------------------------------- batching window (one icon)
rng_arrivals = [0.8, 2.6, 3.1, 5.4, 7.9, 9.2,  # window 1 (illustrative)
                11.5, 14.8, 18.6,                # window 2
                20.9, 21.7, 23.0, 24.4, 26.1, 27.3, 28.0, 29.4]  # window 3


def poisson(m, n):
    return [math.exp(-m) * m ** i / math.factorial(i) for i in range(n)]


for k in (1, 2, 3):
    fig = plt.figure(figsize=(5.9, 3.0))
    ax = fig.add_axes([0.06, 0.62, 0.9, 0.30])
    W = num("win_ms")
    for w in (0, W, 2 * W):
        ax.axvspan(w + 0.15, w + W - 0.15, color=C["memory_l"], alpha=0.7, lw=0)
        ax.annotate("", (w + W, -0.55), xytext=(w + W, 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=C["compute"], lw=2))
    ax.text(W + 0.4, 0.75, "dispatch", ha="left", fontsize=12, color=C["compute"])
    for t in rng_arrivals:
        ax.plot([t, t], [-0.35, 0.35], color=C["text"], lw=2)
    ax.annotate("", (W, -0.95), xytext=(rng_arrivals[0], -0.95),
                arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=1.5))
    if k == 3:
        ax.annotate("cap reached: dispatch early", (26.1, 0.4), xytext=(29.8, 1.15),
                    fontsize=12, color=C["data"], ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=C["data"]))
    ax.set_xlim(0, 30)
    ax.set_ylim(-1.6, 1.6)
    ax.set_yticks([])
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel(f"ms; {val('win_ms')} ms windows, arrivals illustrative", fontsize=10,
                  color=C["neutral"], labelpad=1)
    show = k >= 2
    rates = [r for r in c.WIN.arrival_rates if r is not c.R500]
    for j, r in enumerate(rates):
        q, mean = m(r, "1/s"), c.WIN.mean_arrivals(r)
        hx = fig.add_axes([0.06 + j * 0.32, 0.08, 0.26, 0.24])
        pr = poisson(mean, 21)
        hx.bar(range(21), pr, color=C["memory"], width=0.8, alpha=1.0 if show else 0.0)
        hx.set_xlim(-0.8, 20.8)
        hx.set_ylim(0, 0.65)
        hx.set_yticks([])
        hx.spines["left"].set_visible(False)
        hx.set_xticks([0, 10, 20])
        hx.tick_params(axis="x", labelsize=11, colors=C["text"] if show else "white")
        hx.spines["bottom"].set_alpha(1.0 if show else 0.0)
        extra = f": {val('empty_50')}% empty" if r is c.R50 else ""
        hx.set_title(f"{val(f'qps_{q:.0f}')} QPS{extra}", fontsize=12, fontweight="bold",
                     color=C["text"], alpha=1.0 if show else 0.0)
    fig.text(0.06, -0.05, f"Requests per {val('win_ms')} ms window (Poisson)", fontsize=11,
             color=C["neutral"] if show else "white")
    sp.save(fig, out / f"window_step{k}.svg")

# ------------------------------------------------------------ cold start
phases = [("Weights from local SSD", num("cs_ssd"), "memory", f"{val('cs_ssd')} s"),
          ("Weights from S3", num("cs_s3_hi"), "memory", f"{val('cs_s3_lo')}–{val('cs_s3_hi')} s"),
          ("CUDA context", num("cs_cuda_hi"), "neutral", f"{val('cs_cuda_lo')}–{val('cs_cuda_hi')} s"),
          ("Warmup + runtime", num("cs_warm_rt"), "neutral", f"{val('cs_warm_rt')} s"),
          ("TensorRT engine build", num("cs_trt_hi"), "compute", f"{val('cs_trt_lo')}–{val('cs_trt_hi')} s")]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, (lab, v, ck, txt) in enumerate(phases):
        y = len(phases) - 1 - i
        a = 1.0 if (k == 2 or i < 4) else 0.0
        if lab.startswith("Weights from S3"):
            lo, hi = num("cs_s3_lo"), num("cs_s3_hi")
            ax.barh(y, hi - lo, left=lo, color=C[ck], height=0.6, alpha=a * 0.45)
            ax.barh(y, lo, color=C[ck], height=0.6, alpha=a)
        elif lab.startswith("TensorRT"):
            lo, hi = num("cs_trt_lo"), num("cs_trt_hi")
            ax.barh(y, lo, color=C[ck], height=0.6, alpha=a)
            ax.barh(y, hi - lo, left=lo, color=C[ck], height=0.6, alpha=a * 0.45)
        else:
            ax.barh(y, v, color=C[ck], height=0.6, alpha=a)
        ax.text(v * 1.15, y, txt, va="center", fontsize=13, fontweight="bold",
                color=C[ck], alpha=a)
        ax.text(0.085, y, lab, ha="right", va="center", fontsize=13,
                fontweight="bold", alpha=a)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 200)
    ax.set_ylim(-0.6, len(phases) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.set_xlabel("Seconds (log scale; representative, table 17)")
    sp.save(fig, out / f"cold_start_step{k}.svg")

# ------------------------------------------------------------- model swap
fig, ax = plt.subplots(figsize=(4.3, 2.2))
DL = num("p99_budget_ms")
ax.axvspan(DL, 360, color=C["danger_l"], alpha=0.6, lw=0)
ax.axvline(DL, color=C["danger"], lw=2.5, ls="--")
ax.text(DL + 6, 1.25, f"{val('p99_budget_ms')} ms deadline", color=C["danger"], fontsize=12, fontweight="bold")
ax.barh(0.5, num("b1_ms"), color=C["compute"], height=0.35)
ax.annotate(f"forward pass: {val('b1_ms')} ms", (4, 0.5), xytext=(62, 0.5), va="center",
            fontsize=12, color=C["compute"], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C["compute"]))
ax.barh(-0.2, num("swap_ms"), color=C["memory"], height=0.35)
ax.text(num("swap_ms") - 6, -0.2, f"load {val('swap_gb')} GB: {val('swap_ms')} ms", va="center", ha="right",
        fontsize=12, color="white", fontweight="bold")
ax.set_xlim(0, 360)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([])
ax.set_xticks([0, 100, 200, 300])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Milliseconds")
sp.save(fig, out / "model_swap.svg")

# ------------------------------------------- static against dynamic (13.1.1)
sp.hbars(out / "static_dynamic.svg", [
    ("On demand, batch 1", m(c.PHOTOS.on_demand_total, "s"), "danger", f"{val('static_b1_s')} s"),
    ("Overnight, large batch", m(c.PHOTOS.offline_total, "s"), "data",
     f"{val('static_int8_s')} s: {val('static_ratio')}× faster"),
], "", log=False, size=sp.SHORT,
    xlabel=f"Seconds to classify {val('photos')} photos on one H100")

# --------------------------------------------- second budget: DLRM (Lighthouse 13.1)
resnet = [(f"Prep {val('host_stage')}", PRE, "data"), ("", COPY, "network"),
          (f"Model {val('b1_ms')}", FWD, "compute"), ("", POST, "neutral")]
DS = {k: m(t, "ms") for k, t in c.DLRM.stages.items()}
dlrm = [("", DS["parse"], "neutral"), (f"Embeddings {val('dl_emb')}", DS["embedding lookup"], "memory"),
        ("MLP", DS["MLP forward"], "compute"), ("", DS["rank and filter"], "neutral")]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    for y, parts, show in ((1, resnet, True), (0, dlrm, k >= 2)):
        left = 0.0
        for name, v, ck in parts:
            ax.barh(y, v, left=left, color=C[ck], height=0.6, edgecolor="white",
                    lw=1.5, alpha=1.0 if show else 0.0)
            if name:
                ax.text(left + v / 2, y, name, ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold", alpha=1.0 if show else 0.0)
            left += v
        txt = (f"{val('budget_total')} ms\nmodel {val('model_share')}%" if y == 1 else
               f"{val('dl_total')} ms\nMLP {val('dl_mlp_share')}%")
        ax.text(left + 0.15, y, txt, va="center", fontsize=12, fontweight="bold",
                alpha=1.0 if show else 0.0)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["ResNet-50", "DLRM"], fontsize=12, fontweight="bold")
    if k < 2:
        ax.get_yticklabels()[1].set_alpha(0.0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("One request (ms; Model Serving, tables 3, 4)", fontsize=11)
    sp.save(fig, out / f"dlrm_step{k}.svg")

# ------------------------------------------------- fan-out tail (Sec. 13.4.5)
nn = np.arange(1, c.FAN100.n_components + 1)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.9, 2.0))
    A, N10, N100 = c.FAN100.availability, c.FAN10.n_components, c.FAN100.n_components
    ax.plot(nn, 100 * A ** nn, color=C["memory"], lw=3)
    f10, f100 = num("fan10"), num("fan100")
    ax.plot([1], [100 * A], "o", ms=8, color=C["memory"], mec="white", zorder=5)
    ax.plot([N10], [f10], "o", ms=8, color=C["memory"], mec="white", zorder=5)
    ax.annotate(f"{val('fan_n10')} servers: {val('fan10')}%", (N10, f10), xytext=(10, -4), textcoords="offset points",
                fontsize=11, fontweight="bold", color=C["memory"])
    if k == 1:
        ax.annotate(f"{val('fan_n100')}: ???", (N100, f100), xytext=(-12, -24), textcoords="offset points",
                    ha="right", fontsize=12, fontweight="bold", color=C["danger"])
    else:
        ax.plot([N100], [f100], "o", ms=9, color=C["danger"], mec="white", zorder=5)
        ax.annotate(f"{val('fan_n100')}: {val('fan100')}%", (N100, f100), xytext=(-12, -24), textcoords="offset points",
                    ha="right", fontsize=12, fontweight="bold", color=C["danger"])
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Servers each request fans out to")
    ax.set_ylabel("Requests on time (%)")
    tag(ax, f"Each server on time {val('fan_avail_pct')}%, independent", y=-0.40)
    sp.save(fig, out / f"fanout_step{k}.svg")

# ------------------------------------------------ runtime table (Example 13.6)
rt = [("PyTorch eager", m(c.SVC.service_time(c.B1), "ms"), "neutral", f"{val('b1_ms_2')} ms")]
for _n, _k, _ck in (("TorchScript", "ts", "compute"), ("ONNX Runtime", "onnx", "compute"),
                    ("TensorRT FP32", "fp32", "compute"), ("TensorRT FP16", "fp16", "data"),
                    ("TensorRT INT8", "int8", "data")):
    _x = num(f"rt_{_k}_x")
    _xs = f"{_x:.0f}" if abs(_x - round(_x)) < 0.05 else val(f"rt_{_k}_x")
    rt.append((_n, m(c.RT[_k], "ms"), _ck, f"{val(f'rt_{_k}')} ms ({_xs}×)"))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    for i, (lab, v, ck, txt) in enumerate(rt):
        y = 5 - i
        show = k == 2 or i < 4
        ax.barh(y, v, color=C[ck], height=0.62, alpha=1.0 if show else 0.0)
        ax.text(v + 0.05, y, txt, va="center", fontsize=11, fontweight="bold",
                color=C[ck], alpha=1.0 if show else 0.0)
        ax.text(-0.05, y, lab, ha="right", va="center", fontsize=11, fontweight="bold",
                alpha=1.0 if show else 0.0)
    ax.text(2.6, 3.0, f"less dispatch:\n{val('rt_fp32_x')}×", fontsize=11, color=C["compute"],
            fontweight="bold", ha="center", va="center")
    ax.text(2.6, 0.5, f"lower precision:\n{val('rt_prec_gain')}× more", fontsize=11, color=C["data"],
            fontweight="bold", ha="center", va="center", alpha=1.0 if k == 2 else 0.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel("ResNet-50, batch 1, one H100 (ms; modeled, Example 13.6)")
    sp.save(fig, out / f"runtime_step{k}.svg")

# --------------------------------------- dual-trigger trace (Example 13.4)
ARR = [m(t, "ms") for t in c.TRACE.arrivals]
TO, FREE, B4, RESP = (m(c.TRACE.timeout, "ms"), m(c.TRACE.worker_free, "ms"), m(c.TRACE.batch_time, "ms"),
                      m(c.TRACE.response_path, "ms"))
DLN, DONE = m(c.TRACE.deadline, "ms"), m(c.TRACE.oldest_latency, "ms")
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    for t in ARR:
        ax.plot([t, t], [1.65, 2.35], color=C["text"], lw=2.5)
    ax.text(ARR[-1] / 2, 2.55, "arrivals at " + ", ".join(val(f"arrive_{i}") for i in range(len(ARR))) + " ms",
            ha="center", fontsize=11, fontweight="bold")
    ax.barh(1.0, TO, color=C["memory_l"], edgecolor=C["memory"], height=0.5)
    ax.text(TO / 2, 1.0, f"timer {val('timeout_ms')} ms", ha="center", va="center", fontsize=11,
            color=C["memory"], fontweight="bold")
    a2 = 1.0 if k >= 2 else 0.0
    ax.barh(1.0, FREE - TO, left=TO, color=C["danger_l"], edgecolor=C["danger"], height=0.5,
            hatch="//", alpha=a2)
    ax.text((TO + FREE) / 2, 0.62, f"worker busy {val('backlog_ms')} ms", ha="center", va="top", fontsize=10.5,
            color=C["danger"], fontweight="bold", alpha=a2)
    ax.barh(1.0, B4, left=FREE, color=C["compute"], height=0.5, alpha=a2)
    ax.text(FREE + B4 / 2, 1.35, f"batch of {len(ARR)}: {val('batch4_ms')} ms", ha="center", va="bottom",
            fontsize=10.5, color=C["compute"], fontweight="bold", alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    ax.barh(1.0, RESP, left=FREE + B4, color=C["network"], height=0.5, alpha=a3)
    ax.text(FREE + B4 + 3.4, 0.62, f"response {val('response_ms')} ms", ha="left", va="top", fontsize=10.5,
            color=C["network"], fontweight="bold", alpha=a3)
    ax.annotate("", (DLN, -0.05), xytext=(DONE, -0.05),
                arrowprops=dict(arrowstyle="<->", color=C["data"], lw=1.8, alpha=a3))
    ax.text(DLN - 1.0, -0.2, f"done at {val('trace_done_ms')} ms: {val('trace_slack_ms')} ms of slack", ha="right",
            va="top", fontsize=11, color=C["data"], fontweight="bold", alpha=a3)
    ax.axvline(DLN, color=C["danger"], lw=2, ls="--")
    ax.text(DLN - 0.5, 2.55, f"deadline {val('deadline_ms')} ms", ha="right", fontsize=11, color=C["danger"],
            fontweight="bold")
    ax.set_xlim(-1, 52)
    ax.set_ylim(-0.9, 2.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms since the oldest request arrived (hypothetical trace)")
    sp.save(fig, out / f"dual_trigger_step{k}.svg")

# ----------------------------------- cost per image across hardware (Ex. 13.13)
inst = [("c5.xlarge (CPU)", "CPU"), ("g4dn.xlarge (T4)", "T4"), ("H100", "H100")]
T4_USD, T4_RATE = m(c.INST.prices["T4"], "dollar/hour"), m(c.INST.rates["T4"], "1/s")
H100_USD = m(c.INST.prices["H100"], "dollar/hour")
load = np.arange(50, num("b32_rate") + 1, 10)
for k in (1, 2):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.4, 2.5),
                                 gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.55})
    for i, (lab, key) in enumerate(inst):
        y = 2 - i
        cpm = m(c.INST.cost_per_million(key), "dollar")
        ck = "data" if lab == "H100" else "neutral"
        ax.barh(y, cpm, color=C[ck], height=0.6)
        ax.text(cpm + 0.03, y, f"\\${val(f'cpm_{key.lower()}')}", va="center", fontsize=12, fontweight="bold",
                color=C[ck])
        ax.text(-0.03, y, {"c5.xlarge (CPU)": "CPU (c5)", "g4dn.xlarge (T4)": "T4"}.get(lab, lab), ha="right",
                va="center", fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title("Fully busy", fontsize=12, loc="left", fontweight="bold")
    ax.set_xlabel("\\$ per million images", fontsize=11)
    a = 1.0 if k >= 2 else 0.0
    t4 = np.ceil(load / T4_RATE) * T4_USD / (load * 3600) * 1e6
    h1 = H100_USD / (load * 3600) * 1e6
    bx.plot(load / 1000, h1, color=C["data"], lw=2.5, alpha=a)
    bx.plot(load / 1000, t4, color=C["neutral"], lw=2, alpha=a)
    bx.axvline(num("t4_n_rate") / 1000 + 0.1, color=C["danger"], lw=1.5, ls="--", alpha=a)
    bx.text(3.1, 1.2, f"crossover\nnear {val('crossover')}/s", fontsize=12, color=C["danger"],
            fontweight="bold", alpha=a)
    bx.text(3.15, 1.75, "one H100", ha="left", fontsize=12, color=C["data"],
            fontweight="bold", alpha=a)
    bx.text(7.05, 0.55, "T4s sized\nto load", ha="right", fontsize=12,
            color=C["neutral"], fontweight="bold", alpha=a)
    bx.set_ylim(0, 2.0)
    bx.set_xlim(0, 7.1)
    bx.set_title("At a given load (derived)", fontsize=12, loc="left", fontweight="bold",
                 alpha=a)
    bx.set_xlabel("Load (thousand img/s)", fontsize=11, alpha=a)
    if k < 2:
        for sp_ in bx.spines.values():
            sp_.set_alpha(0.0)
        bx.tick_params(colors="white")
    sp.save(fig, out / f"cost_hw_step{k}.svg")

# ------------------------------------- next lecture: Part 2's hook, third bar hidden
HOOK = [("No deadline\n(Offline)", c.LL_OFF["tokens_per_s"], "neutral", val("llama_off")),
        (f"TTFT {val('llama_conv_ttft_s')} s,\nTPOT {val('llama_conv_tpot')} ms", c.LL_CONV["tokens_per_s"], "compute",
         val("llama_conv")),
        (f"TTFT {val('llama_int_ttft')} ms,\nTPOT {val('llama_int_tpot')} ms", c.LL_INT["tokens_per_s"], "danger", "")]
fig, ax = plt.subplots(figsize=(2.6, 2.3))
for i, (lab, v, ck, txt) in enumerate(HOOK):
    y = 2 - i
    ax.barh(y, v / 1000 if i < 2 else 0, color=C[ck], height=0.6)
    ax.text(-0.6, y, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text((v / 1000 + 0.6) if i < 2 else 0.8, y, txt if i < 2 else "???", va="center",
            fontsize=11, fontweight="bold", color=C[ck])
ax.set_yticks([])
ax.set_xlim(0, 40)
ax.set_ylim(-0.5, 2.5)
ax.spines["left"].set_visible(False)
ax.set_xticks([0, 20, 40])
ax.set_xlabel("Thousand tokens/s,\nLlama 2 70B, 8× H100", fontsize=11)
sp.save(fig, out / "next_hook.svg")

sp.hbars(out / "synth_hook.svg", [
    (f"Batch {c.B1}", num("b1_usd"), "danger", f"\\${val('b1_usd')} per million"),
    (f"Batch {c.B8}", num("b8_usd"), "compute", f"\\${val('b8_usd')}"),
], "", log=False, size=(3.4, 1.3), xlabel="Cost per million requests")

# --------------------------------------------------------------- synthesis
sp.hbars(out / "synth_gpus.svg", [
    ("Batch 1", c.PLAN.accelerators_planned, "danger", f"{val('plan_gpus')} H100s"),
    ("Batched", c.PLAN.batched_accelerators_planned, "data", f"{val('syn_batched')} H100s"),
], "", log=False, size=(3.4, 1.5),
    xlabel=f"{val('peak_rate')} req/s, {val('p99_budget_ms')} ms p99 (plan)")
sp.hbars(out / "synth_cost.svg", [
    ("Batch 1", num("syn_cost_b1"), "danger", f"\\${val('syn_cost_b1')} per million"),
    ("Batched", num("syn_cost_bt"), "data", f"\\${val('syn_cost_bt')}"),
], "", log=False, size=(3.4, 1.5),
    xlabel="Fleet cost at 5,000 req/s (derived)")

# -----------------------------------------------------------------------------
# Text-free thumbnails for the goals and takeaways slots (2.5 by 0.95 cm): a
# pictogram of the figure that teaches each idea, so no text shrinks below the
# legibility floor.
# -----------------------------------------------------------------------------
def thumb(name, draw, size=(1.0, 0.4)):
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    ax.axis("off")
    fig.savefig(out / f"{name}.svg", format="svg", pad_inches=0.0, facecolor="white")
    plt.close(fig)


def _t_timeline(ax):    # three lanes, a long host bar, a short GPU bar
    ax.barh(2, PRE, color=C["data"], height=0.6)
    ax.barh(1, COPY, left=PRE, color=C["network"], height=0.6)
    ax.barh(0, FWD, left=PRE + COPY, color=C["compute"], height=0.6)
    ax.set_xlim(0, 9); ax.set_ylim(-0.6, 2.6)


def _t_knee(ax):        # the mean and p99 curves rising toward 100 percent
    rr = np.linspace(0, 0.93, 200)
    ax.axvspan(U_EX, 1.0, color=C["danger_l"], lw=0)
    ax.plot(rr, 1 / (1 - rr), color=C["memory"], lw=1.6)
    ax.plot(rr, c.PLAN.tail_factor / (1 - rr), color=C["danger"], lw=1.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 30)


def _t_window(ax):      # arrivals in windows, a dispatch arrow at each end
    for w in (0, 10, 20):
        ax.axvspan(w + 0.3, w + 9.7, color=C["memory_l"], lw=0)
    for t in [0.8, 2.6, 3.1, 5.4, 7.9, 9.2, 11.5, 14.8, 18.6, 20.9, 23.0, 26.1, 28.0]:
        ax.plot([t, t], [0.2, 0.8], color=C["text"], lw=1)
    ax.set_xlim(0, 30); ax.set_ylim(0, 1)


def _t_cost(ax):        # a falling cost curve and a sawtooth crossing it
    ld = np.linspace(0.3, 7, 300)
    ax.plot(ld, H100_USD / ld, color=C["data"], lw=1.6)
    ax.plot(ld, np.ceil(ld / (T4_RATE / 1000)) * T4_USD / ld, color=C["neutral"], lw=1.2)
    ax.set_xlim(0, 7.1); ax.set_ylim(0, 3)


def _t_cold(ax):        # short load bars and one long compile bar
    ax.barh([3, 2, 1], [0.5, 1.0, 0.6], color=[C["memory"], C["neutral"], C["neutral"]],
            height=0.6)
    ax.barh(0, 8, color=C["compute"], height=0.6)
    ax.set_xlim(0, 8.5); ax.set_ylim(-0.6, 3.6)


thumb("thumb_timeline", _t_timeline)
thumb("thumb_knee", _t_knee)
thumb("thumb_window", _t_window)
thumb("thumb_cost", _t_cost)
thumb("thumb_cold", _t_cold)

print("figures written")
