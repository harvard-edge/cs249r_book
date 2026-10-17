#!/usr/bin/env python3
"""Figures for Lecture 13 (Model Serving).

Every value is sourced. "MS" is the rendered chapter, Model Serving.
The chapter's ResNet-50 numbers are its modeled reference profile on the
H100 (not measurements); its host phases are stated assumptions.

  - Cost of latency: batch 1 3.5 ms, 283 req/s, $3.92 per million;
    batch 8 3.8 ms, 2,129 req/s, $0.52; GPU at $4/hour (MS Example 13.1).
    The ratio 3.92 / 0.52 = 7.5 is derived here (the book rounds to 8x).
  - Request budget: JPEG decode 3, resize 1, normalize 0.5, copy 0.5,
    forward pass 3.5, softmax and top-5 0.1 ms, total 8.6 ms (MS table 3).
    10x faster model: 8.6 -> 5.5 ms, 1.6x (MS Systems Perspective 13.2).
  - Pipelining: serial duty cycle 41 percent, pipelined 78 percent
    (MS Sec. 13.3.5).
  - Measured frame latency, edge face recognition: ingestion 18.8 ms,
    detection (AI) 74.8, brokers 126.1, identification (AI) 131.5 ms
    (Richins et al., "AI Tax", arXiv 2007.10571, Fig. 6).
  - Queueing: W = S / (1 - rho), p99 ~ 4.6 S / (1 - rho) (MS eqs. 4, 5);
    table 7 values for a 3.53 ms service (printed as 3.5 ms in the book):
    50% 7.1, 70% 11.8, 80% 17.6, 90% 35.3, 95% 70.6 ms; p99 at 70%
    54.1 ms. Faster-model fallacy: 80% load, 3.5 -> 1.8 ms service,
    17.6 -> 3.0 ms (MS Sec. 13.11).
  - MLPerf Inference v5.0, closed, datacenter, ResNet-50 on 8x H100-SXM:
    Cisco UCS C885A M8 (5.0-0010) Offline 715,761 samples/s, Server
    652,002 queries/s; QuantaGrid D74H-7U (5.0-0065) Offline 712,859,
    Server 584,161 (summary_results.json, mlcommons/inference_results_v5.0).
    Server bound for ResNet-50: 15 ms at the 99th percentile (v5.0 rules).
  - Batching: affine service time 3.50 ms + 0.033 ms per image (MS
    Sec. 13.5.4); table 8 (batch 1, 8, 32: 3.53, 3.76, 4.55 ms; 280,
    2,130, 7,040 img/s; fixed share 99, 93, 77 percent); batch 128 cap
    gives about 16,600 img/s (MS Sec. 13.5.4).
  - MLPerf Inference v5.0, edge, ResNet-50 on NVIDIA L4 (HPE ProLiant
    DL145 Gen11, 3x L4, 5.0-0047): SingleStream 0.414 ms (90th percentile),
    MultiStream 0.770 ms per 8-sample query (99th percentile), Offline
    38,566 samples/s on three GPUs. Rates per GPU derived here: 1 / 0.414 ms
    = 2,415/s; 8 / 0.770 ms = 10,390/s; 38,566 / 3 = 12,855/s.
  - Batching window: E[wait] = window / 2; Poisson count per 10 ms window
    at 50, 200, 1,000 QPS (MS eqs. 8, 9; 61 percent empty at 50 QPS).
  - Cold start phases (MS table 17, representative): SSD 0.5 s, S3 3-5 s,
    CUDA context 0.3-0.5 s, TensorRT build 15-30 s, warmup 0.2 s, runtime
    0.4 s; totals about 1.5 s and about 35 s.
  - Measured loading (Fu et al., ServerlessLLM, OSDI 2024, Sec. 2.3):
    OPT-30B into 4 GPUs 34 s, Llama 2 70B into 8 GPUs 84 s with PyTorch;
    a 130 GB checkpoint at 5 GB/s takes at least 26 s.
  - Model swap: 10 GB over PCIe Gen4 x16 at 32 GB/s = 312.5 ms against a
    50 ms SLO (MS Sec. 13.8.4).
  - Splitwise (Patel et al., ISCA 2024, Table IV): Llama 2 70B, one
    DGX-H100, no batching, conversation trace, P50: TTFT 84 ms, time
    between tokens 28 ms, end to end 3,387 ms. Ticks are drawn every 28 ms.
  - GPT-2 XL decode with a 1,024-token context: weights 3.0 GB, cache
    0.31 GB per sequence; batch 1 3.3 GB in 1.0 ms (1,010 tokens/s);
    batch 32 13.1 GB in 3.9 ms (8,200 tokens/s) (MS Sec. 13.9.2).
  - vLLM (Kwon et al., SOSP 2023): share of KV memory holding token states,
    Orca (Max) 20.4, (Pow2) 26.8, (Oracle) 38.2, vLLM 96.3 percent (Fig. 2);
    average batched requests, OPT-13B, ShareGPT: 7.00, 9.81, 13.62, 30.42
    (Fig. 13a). The Orca baselines are the vLLM authors' reimplementations.
  - Joules per token, derived: a 1,000 W node (MS Napkin Math 13.5)
    divided by the token rates above: 0.99 J and 0.12 J.
  - Synthesis: 36 H100s at batch 1, 3 batched (MS Napkin Math 13.3,
    Sec. 13.10.2). MLPerf v5.0 Llama 2 70B on Dell PowerEdge XE9680, 8x
    H100-SXM (5.0-0020): Server 30,903.6 tokens/s under 2,000/200 ms
    TTFT/TPOT; 12,979.4 tokens/s under 450/40 ms (interactive).
  - Recall from Lecture 12: RTX 4090 ResNet-50 SingleStream 0.282 ms
    (about 3,550/s, derived) and Offline 44,411 samples/s (MLPerf v5.0,
    5.0-0032).

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

C = sp.C
plt = sp.plt
out = Path(__file__).parent


def tag(ax, text, x=0.0, y=-0.30):
    """A small gray provenance tag under the axes (modeled, derived ...)."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9,
            color=C["neutral"], ha="left", va="top")


# --------------------------------------------------------------- hook cost
sp.hbars_steps(out / "hook_cost", [
    ("Batch 1  (3.5 ms)", 3.92, "danger", "\\$3.92"),
    ("Batch 8  (3.8 ms)", 0.52, "compute", "\\$0.52"),
], "", log=False, xlabel="Cost per million requests (dollars; modeled, \\$4 per GPU-hour)",
    size=sp.SHORT)

# ------------------------------------------------- recall: Lecture 12 bars
sp.hbars(out / "recall_l12.svg", [
    ("One at a time", 3.55, "neutral", "about 3,550/s"),
    ("All at once", 44.411, "compute", "44,411/s"),
], "", log=False, xlabel="ResNet-50, thousand images per second, one RTX 4090",
    size=(5.0, 1.5))

# ------------------------------------------------ request budget (Amdahl)
stages_now = [("Preprocess", 4.5, "data"), ("Copy", 0.5, "network"),
              ("Model", 3.5, "compute"), ("Post", 0.1, "neutral")]
stages_fast = [("Preprocess", 4.5, "data"), ("Copy", 0.5, "network"),
               ("Model", 0.35, "compute"), ("Post", 0.1, "neutral")]
frame_ms = [("Ingest", 18.8, "neutral"), ("Detect (AI)", 74.8, "compute"),
            ("Brokers", 126.1, "network"), ("Identify (AI)", 131.5, "compute")]


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


for k in (1, 2, 3):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.8, 3.3),
                                 gridspec_kw={"height_ratios": [2, 1.1], "hspace": 0.95})
    total = stacked_row(ax, 1, stages_now)
    ax.text(total + 0.12, 1, "8.6 ms", va="center", fontsize=13,
            fontweight="bold", color=C["text"])
    t2 = stacked_row(ax, 0, stages_fast, labels=True)
    ax.text(t2 + 0.12, 0, "???" if k == 1 else "5.5 ms: 1.6× faster", va="center",
            fontsize=13, fontweight="bold", color=C["danger"])
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Today", "Model 10× faster"], fontsize=12, fontweight="bold")
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("One ResNet-50 request (ms; modeled model, assumed host phases)",
                  fontsize=10)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    show = k == 3
    ftot = stacked_row(bx, 0, frame_ms, show=show, fs=10)
    bx.text(ftot + 4, 0, "351 ms", va="center", fontsize=12, fontweight="bold",
            color=C["text"], alpha=1.0 if show else 0.0)
    bx.set_yticks([0])
    bx.set_yticklabels(["Measured frame"], fontsize=12, fontweight="bold")
    bx.set_xlim(0, 351 * 11.5 / 8.6 * 0.86)
    bx.set_ylim(-0.5, 0.5)
    bx.set_xlabel("Face recognition, one video frame (ms; AI stages 59 percent)",
                  fontsize=10)
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    for side in ("bottom",):
        bx.spines[side].set_alpha(1.0 if show else 0.0)
    if not show:
        bx.xaxis.label.set_alpha(0.0)
        for t in bx.get_xticklabels() + bx.get_yticklabels():
            t.set_alpha(0.0)
        bx.tick_params(axis="x", colors="white")
    sp.save(fig, out / f"budget_step{k}.svg")

# ------------------------------------- request timeline (book variant)
# Slide-size variant of the chapter's request-timeline figure (MS figure 3,
# table 3): host phases assumed, forward pass modeled.
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.9, 2.5))
    lanes = {"Host CPU": 2, "PCIe": 1, "GPU": 0}
    for name, y in lanes.items():
        ax.axhspan(y - 0.38, y + 0.38, color="#F2F2F2", lw=0, zorder=0)
    host = [("JPEG decode 3 ms", 0, 3.0), ("resize\n1 ms", 3.0, 1.0), ("", 4.0, 0.5)]
    for lab, x0, w in host:
        ax.barh(2, w, left=x0, color=C["data_l"], edgecolor=C["data"], lw=1.5, height=0.56)
        if lab:
            ax.text(x0 + w / 2, 2, lab, ha="center", va="center", fontsize=11)
    ax.text(4.25, 2.5, "normalize 0.5", ha="center", va="bottom", fontsize=10)
    if k >= 2:
        ax.barh(1, 0.5, left=4.5, color=C["network_l"], edgecolor=C["network"], lw=1.5,
                height=0.56)
        ax.text(5.1, 1, "copy 0.5 ms", va="center", fontsize=10)
        ax.barh(0, 3.5, left=5.0, color=C["compute_l"], edgecolor=C["compute"], lw=1.5,
                height=0.56)
        ax.text(6.75, 0, "forward pass 3.5 ms", ha="center", va="center", fontsize=11)
        ax.barh(2, 0.1, left=8.5, color=C["data_l"], edgecolor=C["data"], lw=1.5,
                height=0.56)
        ax.text(8.55, 2.5, "top-5 0.1", ha="center", va="bottom", fontsize=10)
    if k >= 3:
        ax.barh(0, 5.0, left=0, color="white", edgecolor=C["danger"], hatch="//",
                lw=1.5, height=0.56)
        ax.text(2.5, 0, "GPU idle 5 ms", ha="center", va="center", fontsize=12,
                color=C["danger"], fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(list(lanes), fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.5, 2.95)
    ax.set_xticks(range(10))
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms since the request reached the server (8.6 ms total)")
    tag(ax, "After Model Serving, figure 3: forward pass modeled, host phases assumed",
        y=-0.33)
    sp.save(fig, out / f"timeline_step{k}.svg")

# ------------------------------------------------------ pipelining variant
for k in (1, 2):
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 2.9), gridspec_kw={"hspace": 0.9})
    # serial: each request runs 8.6 ms end to end before the next starts
    ax = axes[0]
    for i in range(3):
        t0 = i * 8.6
        ax.barh(1, 4.5, left=t0, color=C["data"], height=0.6, edgecolor="white")
        ax.barh(1, 0.5, left=t0 + 4.5, color=C["network"], height=0.6, edgecolor="white")
        ax.barh(0, 3.5, left=t0 + 5.0, color=C["compute"], height=0.6, edgecolor="white")
        ax.barh(1, 0.1, left=t0 + 8.5, color=C["neutral"], height=0.6)
    ax.text(26.2, 0, "GPU busy 41%", va="center", fontsize=12, fontweight="bold",
            color=C["compute"])
    ax.set_title("Serial: one request at a time", fontsize=12, loc="left",
                 color=C["text"], fontweight="bold")
    ax2 = axes[1]
    show = k == 2
    for i in range(6):
        ax2.barh(1, 4.5, left=i * 4.5, color=C["data"], height=0.6,
                 edgecolor="white", alpha=1.0 if show else 0.0)
        if 5.0 + i * 4.5 + 3.5 <= 26.0:
            ax2.barh(0, 3.5, left=5.0 + i * 4.5, color=C["compute"], height=0.6,
                     edgecolor="white", alpha=1.0 if show else 0.0)
    ax2.text(26.2, 0, "GPU busy 78%", va="center", fontsize=12, fontweight="bold",
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
S = 3.53  # the book's reference service time, printed as 3.5 ms
points = [(0.5, 7.1), (0.8, 17.6), (0.9, 35.3), (0.95, 70.6)]
reach = {1: 0.5, 2: 0.8, 3: 0.8, 4: 0.97, 5: 0.97}
for k in (1, 2, 3, 4, 5):
    fig, ax = plt.subplots(figsize=(5.8, 2.75))
    ax.axvspan(70, 100, color=C["danger_l"], alpha=0.6, lw=0)
    rho = np.linspace(0.0, reach[k], 400)
    ax.plot(rho * 100, S / (1 - rho), color=C["memory"], lw=3)
    shown = {1: 1, 2: 2, 3: 2, 4: 4, 5: 4}[k]
    for r, w in points[:shown]:
        ax.plot([r * 100], [w], "o", ms=8, color=C["memory"], mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{w:g} ms", (r * 100, w), xytext=(-8, -14 if w > 60 else 4),
                    textcoords="offset points", ha="right", fontsize=11,
                    color=C["memory"], fontweight="bold")
    if k == 3:
        ax.plot([90], [35.3], "o", ms=9, mfc="white", mec=C["danger"], mew=2, zorder=5)
        ax.annotate("90%: ???", (90, 35.3), xytext=(-10, 4), textcoords="offset points",
                    ha="right", fontsize=12, color=C["danger"], fontweight="bold")
    ax.text(8, 8, "Mean", color=C["memory"], fontsize=12, fontweight="bold")
    if k >= 5:
        rr = np.linspace(0.0, 0.97, 400)
        ax.plot(rr * 100, 4.6 * S / (1 - rr), color=C["danger"], lw=3)
        ax.text(30, 30, "p99", color=C["danger"], fontsize=12, fontweight="bold")
        ax.plot([70], [54.1], "o", ms=9, color=C["danger"], mec="white", mew=1.5, zorder=6)
        ax.annotate("70%: p99 54.1 ms,\nmean 11.8 ms", (70, 54.1), xytext=(-10, 6),
                    textcoords="offset points", ha="right", fontsize=11,
                    color=C["danger"], fontweight="bold")
    ax.text(99, 2, "planning danger zone", fontsize=10, color=C["danger"], ha="right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Utilization (percent of time the GPU is busy)")
    ax.set_ylabel("Time in system (ms)")
    tag(ax, "M/M/1 model, 3.5 ms service time (Model Serving, table 7)", y=-0.27)
    sp.save(fig, out / f"knee_step{k}.svg")

# ------------------------------------------- faster-model vote thumbnail
lam = np.linspace(0, 270, 400)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    s_old, s_new = 3.53, 1.8
    w_old = s_old / (1 - lam * s_old / 1000)
    ax.plot(lam, w_old, color=C["memory"], lw=3)
    ax.text(145, 30, "3.5 ms model", color=C["memory"], fontsize=11, fontweight="bold")
    lam80 = 0.8 / s_old * 1000
    ax.plot([lam80], [17.6], "o", ms=10, color=C["danger"], mec="white", mew=1.5, zorder=5)
    ax.annotate("80% busy: 17.6 ms", (lam80, 17.6), xytext=(15, 22),
                ha="left", fontsize=11, color=C["danger"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C["danger"]))
    if k == 2:
        w_new = s_new / (1 - lam * s_new / 1000)
        ax.plot(lam, w_new, color=C["data"], lw=3)
        ax.plot([lam80], [3.0], "o", ms=10, color=C["data"], mec="white", mew=1.5, zorder=5)
        ax.text(lam80 + 8, -4.5, "1.8 ms model, 41% busy: 3.0 ms", ha="right",
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
    rows = [("System A", 715761, 652002), ("System B", 712859, 584161)]
    for i, (name, off, srv) in enumerate(rows):
        y = 1 - i
        ax.barh(y + 0.18, off / 1000, height=0.34, color=C["neutral"])
        ax.text(off / 1000 + 8, y + 0.18, f"Offline {off:,}", va="center", fontsize=11,
                color=C["neutral"], fontweight="bold")
        ax.barh(y - 0.18, srv / 1000, height=0.34, color=C["compute"],
                alpha=1.0 if k == 2 else 0.0)
        ax.text(srv / 1000 + 8, y - 0.18, f"Server {srv:,}  ({srv / off:.0%} kept)",
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
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    for i, b in enumerate((1, 8, 32)):
        y = 2 - i
        fixed = 3.50 / b
        work = 0.033
        show = k == 2 or b == 1
        a = 1.0 if show else 0.0
        ax.barh(y, fixed, color=C["neutral"], height=0.6, alpha=a)
        ax.barh(y, work, left=fixed, color=C["compute"], height=0.6, alpha=a)
        share = fixed / (fixed + work)
        ax.text(fixed + work + 0.06, y, f"{fixed + work:.2f} ms  ({share:.0%} fixed)",
                va="center", fontsize=12, fontweight="bold", color=C["text"], alpha=a)
        ax.text(-0.05, y, f"Batch {b}", ha="right", va="center", fontsize=12,
                fontweight="bold")
    ax.text(1.75, 2, "fixed: kernel launches + weight read", color="white", fontsize=10,
            ha="center", va="center", fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-0.5, 2.5)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time per image (ms; modeled: 3.50 ms fixed + 0.033 ms per image)")
    sp.save(fig, out / f"fixed_cost_step{k}.svg")

# ------------------------------------- modeled throughput and measured L4
bs = np.unique(np.round(np.logspace(0, np.log10(128), 60)).astype(int))
for k in (1, 2):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.9, 2.6),
                                 gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.55})
    tput = bs / (3.50 + 0.033 * bs) * 1000
    ax.plot(bs, tput / 1000, color=C["compute"], lw=3)
    for b, lab in ((1, "1: 3.5 ms\n280/s"), (32, "32: 4.55 ms\n7,040/s"),
                   (128, "128: 7.7 ms\n16,600/s")):
        v = b / (3.50 + 0.033 * b)
        ax.plot([b], [v], "o", ms=7, color=C["compute"], mec="white", zorder=5)
        ax.annotate(lab, (b, v), xytext=(-8, 6) if b > 1 else (12, 32),
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
    rows = [("1 at a time", 2.415, "neutral", "2,415/s"),
            ("8 at a time", 10.39, "compute", "10,390/s"),
            ("All at once", 12.855, "compute", "12,855/s")]
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
    for w in (0, 10, 20):
        ax.axvspan(w + 0.15, w + 9.85, color=C["memory_l"], alpha=0.7, lw=0)
        ax.annotate("", (w + 10, -0.55), xytext=(w + 10, 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=C["compute"], lw=2))
    ax.text(10.4, 0.75, "dispatch", ha="left", fontsize=12, color=C["compute"])
    for t in rng_arrivals:
        ax.plot([t, t], [-0.35, 0.35], color=C["text"], lw=2)
    ax.annotate("", (10, -0.95), xytext=(0.8, -0.95),
                arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=1.5))
    if k == 3:
        ax.annotate("cap reached: dispatch early", (26.1, 0.4), xytext=(25.5, 1.15),
                    fontsize=12, color=C["data"], ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=C["data"]))
    ax.set_xlim(0, 30)
    ax.set_ylim(-1.6, 1.6)
    ax.set_yticks([])
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("ms; 10 ms windows, arrivals illustrative", fontsize=10,
                  color=C["neutral"], labelpad=1)
    show = k >= 2
    for j, (q, m) in enumerate(((50, 0.5), (200, 2.0), (1000, 10.0))):
        hx = fig.add_axes([0.06 + j * 0.32, 0.08, 0.26, 0.24])
        pr = poisson(m, 21)
        hx.bar(range(21), pr, color=C["memory"], width=0.8, alpha=1.0 if show else 0.0)
        hx.set_xlim(-0.8, 20.8)
        hx.set_ylim(0, 0.65)
        hx.set_yticks([])
        hx.spines["left"].set_visible(False)
        hx.set_xticks([0, 10, 20])
        hx.tick_params(axis="x", labelsize=11, colors=C["text"] if show else "white")
        hx.spines["bottom"].set_alpha(1.0 if show else 0.0)
        extra = ": 61% empty" if q == 50 else ""
        hx.set_title(f"{q:,} QPS{extra}", fontsize=12, fontweight="bold",
                     color=C["text"], alpha=1.0 if show else 0.0)
    fig.text(0.06, -0.05, "Requests per 10 ms window (Poisson)", fontsize=11,
             color=C["neutral"] if show else "white")
    sp.save(fig, out / f"window_step{k}.svg")

# ------------------------------------------------------------ cold start
phases = [("Weights from local SSD", 0.5, "memory", "0.5 s"),
          ("Weights from S3", 5, "memory", "3–5 s"),
          ("CUDA context", 0.5, "neutral", "0.3–0.5 s"),
          ("Warmup + runtime", 0.6, "neutral", "0.6 s"),
          ("TensorRT engine build", 30, "compute", "15–30 s")]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, (lab, v, ck, txt) in enumerate(phases):
        y = len(phases) - 1 - i
        a = 1.0 if (k == 2 or i < 4) else 0.0
        if lab.startswith("Weights from S3"):
            ax.barh(y, 5 - 3, left=3, color=C[ck], height=0.6, alpha=a * 0.45)
            ax.barh(y, 3, color=C[ck], height=0.6, alpha=a)
        elif lab.startswith("TensorRT"):
            ax.barh(y, 15, color=C[ck], height=0.6, alpha=a)
            ax.barh(y, 15, left=15, color=C[ck], height=0.6, alpha=a * 0.45)
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
fig, ax = plt.subplots(figsize=(5.8, 1.9))
ax.axvspan(50, 360, color=C["danger_l"], alpha=0.6, lw=0)
ax.axvline(50, color=C["danger"], lw=2.5, ls="--")
ax.text(52, 1.25, "50 ms deadline", color=C["danger"], fontsize=12, fontweight="bold")
ax.barh(0.5, 3.5, color=C["compute"], height=0.35)
ax.annotate("forward pass: 3.5 ms", (4, 0.5), xytext=(62, 0.5), va="center", fontsize=11,
            color=C["compute"], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C["compute"]))
ax.barh(-0.2, 312.5, color=C["memory"], height=0.35)
ax.text(306, -0.2, "load 10 GB over PCIe: 312.5 ms", va="center", ha="right",
        fontsize=12, color="white", fontweight="bold")
ax.set_xlim(0, 360)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Milliseconds")
sp.save(fig, out / "model_swap.svg")

# ------------------------------------------- static against dynamic (13.1.1)
sp.hbars(out / "static_dynamic.svg", [
    ("On demand, batch 1", 35, "danger", "35 s"),
    ("Overnight, large batch", 0.11, "data", "0.11 s: about 320× faster"),
], "", log=False, size=sp.SHORT,
    xlabel="Seconds to classify 10,000 photos on one H100")

# --------------------------------------------- second budget: DLRM (Lighthouse 13.1)
resnet = [("Preprocess 4.5", 4.5, "data"), ("", 0.5, "network"),
          ("Model 3.5", 3.5, "compute"), ("", 0.1, "neutral")]
dlrm = [("", 0.5, "neutral"), ("Embedding lookup 6", 6.0, "memory"),
        ("MLP", 1.5, "compute"), ("", 1.0, "neutral")]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    for y, parts, show in ((1, resnet, True), (0, dlrm, k >= 2)):
        left = 0.0
        for name, v, ck in parts:
            ax.barh(y, v, left=left, color=C[ck], height=0.6, edgecolor="white",
                    lw=1.5, alpha=1.0 if show else 0.0)
            if name:
                ax.text(left + v / 2, y, name, ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold", alpha=1.0 if show else 0.0)
            left += v
        txt = "8.6 ms: model 41%" if y == 1 else "9 ms: MLP 17%"
        ax.text(left + 0.15, y, txt, va="center", fontsize=12, fontweight="bold",
                alpha=1.0 if show else 0.0)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["ResNet-50", "DLRM"], fontsize=12, fontweight="bold")
    if k < 2:
        ax.get_yticklabels()[1].set_alpha(0.0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("One request (ms; Model Serving, tables 3 and 4)")
    sp.save(fig, out / f"dlrm_step{k}.svg")

# ------------------------------------------------- fan-out tail (Sec. 13.4.5)
nn = np.arange(1, 101)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.plot(nn, 100 * 0.99 ** nn, color=C["memory"], lw=3)
    ax.plot([1], [99], "o", ms=8, color=C["memory"], mec="white", zorder=5)
    ax.plot([10], [90.4], "o", ms=8, color=C["memory"], mec="white", zorder=5)
    ax.annotate("10 servers: 90.4%", (10, 90.4), xytext=(10, -4), textcoords="offset points",
                fontsize=11, fontweight="bold", color=C["memory"])
    if k == 1:
        ax.annotate("100: ???", (100, 36.6), xytext=(-12, -24), textcoords="offset points",
                    ha="right", fontsize=12, fontweight="bold", color=C["danger"])
    else:
        ax.plot([100], [36.6], "o", ms=9, color=C["danger"], mec="white", zorder=5)
        ax.annotate("100: 36.6%", (100, 36.6), xytext=(-12, -24), textcoords="offset points",
                    ha="right", fontsize=12, fontweight="bold", color=C["danger"])
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Servers each request fans out to")
    ax.set_ylabel("Requests on time (%)")
    tag(ax, "Each server on time 99% of the time, independent (Model Serving, 13.4.5)", y=-0.30)
    sp.save(fig, out / f"fanout_step{k}.svg")

# ------------------------------------------------ runtime table (Example 13.6)
rt = [("PyTorch eager", 3.53, "neutral", "3.53 ms"),
      ("TorchScript", 1.79, "compute", "1.79 ms (2×)"),
      ("ONNX Runtime", 0.61, "compute", "0.61 ms (5.8×)"),
      ("TensorRT FP32", 0.38, "compute", "0.38 ms (9.3×)"),
      ("TensorRT FP16", 0.33, "data", "0.33 ms (10.6×)"),
      ("TensorRT INT8", 0.30, "data", "0.30 ms (11.6×)")]
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
    ax.text(2.6, 3.0, "less dispatch:\n9.3×", fontsize=11, color=C["compute"],
            fontweight="bold", ha="center", va="center")
    ax.text(2.6, 0.5, "lower precision:\n1.3× more", fontsize=11, color=C["data"],
            fontweight="bold", ha="center", va="center", alpha=1.0 if k == 2 else 0.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel("ResNet-50, batch 1, one H100 (ms; modeled, Example 13.6)")
    sp.save(fig, out / f"runtime_step{k}.svg")

# --------------------------------------- dual-trigger trace (Example 13.4)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    for t in (0, 2, 5, 9):
        ax.plot([t, t], [1.65, 2.35], color=C["text"], lw=2.5)
    ax.text(4.5, 2.55, "arrivals at 0, 2, 5, 9 ms", ha="center", fontsize=11,
            fontweight="bold")
    ax.barh(1.0, 12, color=C["memory_l"], edgecolor=C["memory"], height=0.5)
    ax.text(6, 1.0, "timer 12 ms", ha="center", va="center", fontsize=11,
            color=C["memory"], fontweight="bold")
    a2 = 1.0 if k >= 2 else 0.0
    ax.barh(1.0, 6, left=12, color=C["danger_l"], edgecolor=C["danger"], height=0.5,
            hatch="//", alpha=a2)
    ax.text(15, 0.62, "worker busy 6 ms", ha="center", va="top", fontsize=10.5,
            color=C["danger"], fontweight="bold", alpha=a2)
    ax.barh(1.0, 3.6, left=18, color=C["compute"], height=0.5, alpha=a2)
    ax.text(19.8, 1.35, "batch of 4: 3.6 ms", ha="center", va="bottom", fontsize=10.5,
            color=C["compute"], fontweight="bold", alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    ax.barh(1.0, 4, left=21.6, color=C["network"], height=0.5, alpha=a3)
    ax.text(25.0, 0.62, "response 4 ms", ha="left", va="top", fontsize=10.5,
            color=C["network"], fontweight="bold", alpha=a3)
    ax.annotate("", (50, -0.05), xytext=(25.6, -0.05),
                arrowprops=dict(arrowstyle="<->", color=C["data"], lw=1.8, alpha=a3))
    ax.text(49.0, -0.2, "done at 25.6 ms: 24.4 ms of slack", ha="right", va="top",
            fontsize=11, color=C["data"], fontweight="bold", alpha=a3)
    ax.axvline(50, color=C["danger"], lw=2, ls="--")
    ax.text(49.5, 2.55, "deadline 50 ms", ha="right", fontsize=11, color=C["danger"],
            fontweight="bold")
    ax.set_xlim(-1, 52)
    ax.set_ylim(-0.9, 2.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("ms since the oldest request arrived (hypothetical trace)")
    sp.save(fig, out / f"dual_trigger_step{k}.svg")

# ----------------------------------- cost per image across hardware (Ex. 13.13)
inst = [("c5.xlarge (CPU)", 0.17, 50), ("g4dn.xlarge (T4)", 0.53, 400), ("H100", 4.0, 7040)]
load = np.arange(50, 7041, 10)
for k in (1, 2):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.9, 2.5),
                                 gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.6})
    for i, (lab, usd, rate) in enumerate(inst):
        y = 2 - i
        cpm = usd / (rate * 3600) * 1e6
        ck = "data" if lab == "H100" else "neutral"
        ax.barh(y, cpm, color=C[ck], height=0.6)
        ax.text(cpm + 0.03, y, f"\\${cpm:.2f}", va="center", fontsize=11, fontweight="bold",
                color=C[ck])
        ax.text(-0.03, y, {"c5.xlarge (CPU)": "CPU (c5)", "g4dn.xlarge (T4)": "T4"}.get(lab, lab), ha="right",
                va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title("Fully busy", fontsize=11, loc="left", fontweight="bold")
    ax.set_xlabel("\\$ per million images", fontsize=10)
    a = 1.0 if k >= 2 else 0.0
    t4 = np.ceil(load / 400) * 0.53 / (load * 3600) * 1e6
    h1 = 4.0 / (load * 3600) * 1e6
    bx.plot(load / 1000, h1, color=C["data"], lw=2.5, alpha=a)
    bx.plot(load / 1000, t4, color=C["neutral"], lw=2, alpha=a)
    bx.axvline(2.9, color=C["danger"], lw=1.5, ls="--", alpha=a)
    bx.text(3.1, 1.15, "crossover\nnear 3,000/s", fontsize=10, color=C["danger"],
            fontweight="bold", alpha=a)
    bx.text(0.55, 1.85, "one H100", ha="left", fontsize=10, color=C["data"],
            fontweight="bold", alpha=a)
    bx.text(7.05, 0.62, "T4s sized to load", ha="right", fontsize=10,
            color=C["neutral"], fontweight="bold", alpha=a)
    bx.set_ylim(0, 2.0)
    bx.set_xlim(0, 7.1)
    bx.set_title("At a given load (derived)", fontsize=11, loc="left", fontweight="bold",
                 alpha=a)
    bx.set_xlabel("Sustained load (thousand img/s)", fontsize=10, alpha=a)
    if k < 2:
        for sp_ in bx.spines.values():
            sp_.set_alpha(0.0)
        bx.tick_params(colors="white")
    sp.save(fig, out / f"cost_hw_step{k}.svg")

# ------------------------------------- next lecture: Part 2's hook, third bar hidden
HOOK = [("No deadline\n(Offline)", 31216.8, "neutral", "31,217"),
        ("TTFT 2 s,\nTPOT 200 ms", 30903.6, "compute", "30,904"),
        ("TTFT 450 ms,\nTPOT 40 ms", 12979.4, "danger", "")]
fig, ax = plt.subplots(figsize=(3.2, 2.0))
for i, (lab, v, ck, txt) in enumerate(HOOK):
    y = 2 - i
    ax.barh(y, v / 1000 if i < 2 else 0, color=C[ck], height=0.6)
    ax.text(-0.4, y, lab, ha="right", va="center", fontsize=10, fontweight="bold")
    ax.text((v / 1000 + 0.5) if i < 2 else 0.6, y, txt if i < 2 else "???", va="center",
            fontsize=10, fontweight="bold", color=C[ck])
ax.set_yticks([])
ax.set_xlim(0, 40)
ax.set_ylim(-0.5, 2.5)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Thousand tokens/s, Llama 2 70B, 8× H100", fontsize=10)
sp.save(fig, out / "next_hook.svg")

sp.hbars(out / "synth_hook.svg", [
    ("Batch 1", 3.92, "danger", "\\$3.92 per million"),
    ("Batch 8", 0.52, "compute", "\\$0.52"),
], "", log=False, size=(3.4, 1.3), xlabel="Cost per million requests")

# --------------------------------------------------------------- synthesis
sp.hbars(out / "synth_gpus.svg", [
    ("Batch 1", 36, "danger", "36 H100s"),
    ("Batched", 3, "data", "3 H100s"),
], "", log=False, size=(3.4, 1.5),
    xlabel="5,000 req/s, 50 ms p99 (plan)")
sp.hbars(out / "synth_cost.svg", [
    ("Batch 1", 8.0, "danger", "\\$8.00 per million"),
    ("Batched", 0.67, "data", "\\$0.67"),
], "", log=False, size=(3.4, 1.5),
    xlabel="Fleet cost at 5,000 req/s (derived)")

print("figures written")
