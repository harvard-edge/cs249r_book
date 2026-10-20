#!/usr/bin/env python3
"""Plots and diagrams for Volume II, Lecture 9 (Fleet Efficiency).

Every value is from the rendered chapter (Fleet Efficiency) or MLSysIM
unless the comment says "derived" (arithmetic on chapter inputs, shown in
the speaker note) or "schematic" (shape only, labeled on the figure).

  - Llama 3 405B: effective training time > 90% (54-day snapshot), MFU 41%
    at 16,384 GPUs and 43% at 8,192 GPUs, BF16 (Dubey et al. 2024, Table 4;
    MLSysIM Literature.TrainingRuns). Split 10 / 53 / 37 (Napkin Math 9.1).
  - Utilization tiers 500 / 350 / 280 GPUs (Sec. 9.2.3, from Fleet
    Orchestration).
  - Published MFU: GPT-3 21.3%, Megatron-Turing NLG 30.2%, Gopher 32.5%,
    PaLM 46.2% (Chowdhery et al. 2022, Table 3; MLSysIM).
  - MLPerf Inference, one DGX H100, offline throughput by round (MLSysIM
    Literature.Series.MLPerfInferenceDGXH100; MLCommons 2025).
  - Trace volume: 500 MB per GPU per 5 s; 512 GB at 1,024 GPUs (Sec. 9.3.3).
  - Pool snapshot: assigned 0.80, partial gangs 0.15, input wait 0.05,
    S = 0.65, ceiling 0.80 (Napkin Math 9.2).
  - Runtime terms of the running job: startup 0.05%, input waits 7.7%,
    checkpoint 0.4%, recovery 0.03%, preemption 1.4% (Napkin Math 9.3).
  - Runtime loss against job size (derived, as the chapter's figure 5 is):
    MTBF 302 h at 128 GPUs scaling as 1/N, checkpoint write 9.8 s,
    Young-Daly interval sqrt(2 x write x MTBF), recovery 6.2 min; input
    waits, preemption, and startup held at the running job's values.
  - Node 180 ms at 65% MFU; fleet 245 ms at 48% (Napkin Math 9.4).
  - Roofline: H100 989 TFLOP/s (FP16 dense) and 3.35 TB/s HBM, ridge about
    295 FLOP/byte; LayerNorm measured 5 TFLOP/s at 2.8 TB/s (Sec. 9.6.3).
    Batch-one decode at 1 FLOP/byte (derived: 2 FLOPs per 2-byte FP16
    weight), so 3.35 TFLOP/s at the floor, 0.3% of peak (Sec. 9.6.3).
  - Waterfall: S 0.65, R 0.90, P 0.48, MPG 0.28; loss shares 35 / 6 / 31
    (Napkin Math 9.5). Lever ceilings 0.35 / 0.31 / 0.38 (Sec. 9.8.1).
  - Iso-goodput: P 0.48 to 0.65 is +35%; break-even runtime loss 26%;
    runtime headroom +11% (Sec. 9.9.1, figure 11).
  - Frontier: Llama 3 405B at 8,192 and 16,384 GPUs, MFU 43% and 41%
    (derived: throughput ratio 2 x 0.41 / 0.43 = 1.91, so time 0.52 and
    GPU-hours per unit of work 0.43 / 0.41 = 1.05).

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent

# Goodput factor colors, used the same way on every slide:
# scheduling = coordination violet, runtime = brown, program = compute orange,
# useful work = data green.
COL = {"S": C["network"], "R": "#78492A", "P": C["compute"], "U": C["data"],
       "S_l": C["network_l"], "R_l": "#E3D3C8", "P_l": C["compute_l"], "U_l": C["data_l"]}


def text(ax, x, y, s, color=C["text"], shown=True, **kw):
    """Text drawn in every step (invisible until shown) so every step has
    the same bounding box and nothing moves on the slide."""
    kw.setdefault("fontsize", 13)
    return ax.text(x, y, s, color=color, alpha=1.0 if shown else 0.0, **kw)


# ---------------------------------------------------------------------------
# 1. Llama 3 405B: two efficiency numbers and their product (Napkin Math 9.1)
# ---------------------------------------------------------------------------
def llama3_split(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 1.75))
    rows = [("Effective\ntraining time", 90, COL["R"], "> 90% of elapsed time"),
            ("MFU of\nkept steps", 41, COL["P"], "41% of peak"),
            ("Held\ncapacity", None, None, None)]
    for i, (lab, v, col, note) in enumerate(rows):
        y = 2 - i
        ax.text(-2, y, lab, ha="right", va="center", fontsize=13,
                fontweight="bold", color=C["text"])
        ax.barh(y, 100, height=0.6, color="white", edgecolor=C["grid"], lw=1)
        if v is not None:
            ax.barh(y, v, height=0.6, color=col)
            ax.text(v + 1.5, y, note, va="center", fontsize=13,
                    fontweight="bold", color=col)
    shown = step >= 2
    left = 0
    for w, col, lab, tcol in [(10, COL["R_l"], "10%", COL["R"]),
                              (53, COL["P_l"], "program 53%", COL["P"]),
                              (37, COL["U"], "progress 37%", "white")]:
        ax.barh(0, w, left=left, height=0.6, color=col, edgecolor="white",
                lw=1.5, alpha=1.0 if shown else 0.0)
        text(ax, left + w / 2, 0, lab, tcol, shown, ha="center", va="center",
             fontsize=12, fontweight="bold", linespacing=0.9)
        left += w
    text(ax, 5, -0.48, "runtime", COL["R"], shown, ha="center", va="center",
         fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, 2.4)
    ax.axis("off")
    sp.save(fig, path)


for k in (1, 2):
    llama3_split(out / f"llama3_split_step{k}.svg", k)

# ---------------------------------------------------------------------------
# 2. Utilization tiers from Fleet Orchestration (recall)
# ---------------------------------------------------------------------------
sp.hbars(out / "tiers.svg",
         [("Allocated", 500, "neutral", "500 GPUs"),
          ("Active", 350, "compute", "350 GPUs"),
          ("Productive", 280, "data", "280 GPUs")],
         "GPUs", log=False, size=(5.6, 1.3))

# ---------------------------------------------------------------------------
# 3. Goodput as a chain of three boundaries (schematic widths), 3 steps
# ---------------------------------------------------------------------------
def goodput_chain(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 2.55))
    widths = [1.0, 0.86, 0.74, 0.58]
    inside = ["offered capacity-time", "held time",
              "retained steps", "useful work"]
    fills = [C["neutral_l"], COL["S_l"], COL["R_l"], COL["U"]]
    tcols = [C["text"], C["text"], C["text"], "white"]
    keys = [None, "S", "R", "P"]
    rules = [None, "S = held / offered", "R = retained / held", "P = ideal / retained"]
    losses = [None, "unallocated, partial gangs, fragments",
              "startup, input waits, checkpoints, rework",
              "exposed comm., stragglers, stalls, dispatch"]
    for i in range(4):
        y = 3 - i
        shown = i <= step
        a = 1.0 if shown else 0.0
        ax.barh(y, widths[i], height=0.56, color=fills[i], edgecolor=C["neutral"],
                lw=1.0, alpha=a)
        text(ax, 0.025, y, inside[i], tcols[i], shown, ha="left", va="center",
             fontsize=11, fontweight="bold")
        if i > 0:
            k = keys[i]
            # the slice lost at this boundary, dashed
            ax.barh(y, widths[i - 1] - widths[i], left=widths[i], height=0.56,
                    color="white", edgecolor=COL[k], lw=1.4, ls="--", alpha=a)
            ax.plot([-0.02, 1.02], [y + 0.5, y + 0.5], color=COL[k], lw=1.2,
                    ls=":", alpha=a)
            text(ax, -0.09, y, k, "white", shown, ha="center", va="center",
                 fontsize=14, fontweight="bold",
                 bbox=dict(boxstyle="circle,pad=0.25", fc=COL[k], ec="none",
                           alpha=a))
            text(ax, 1.05, y + 0.2, rules[i], COL[k], shown, ha="left",
                 va="center", fontsize=12, fontweight="bold")
            text(ax, 1.05, y - 0.2, losses[i], C["neutral"], shown, ha="left",
                 va="center", fontsize=10.5)
    text(ax, 1.05, 3, "what the pool could run", C["neutral"], True, ha="left",
         va="center", fontsize=11.5)
    text(ax, -0.15, -0.68, "useful work = offered capacity-time × S × R × P",
         C["data"], step >= 3, ha="left", va="center", fontsize=12.5,
         fontweight="bold")
    text(ax, 2.35, 3.35, "widths schematic", C["neutral"], True, ha="right",
         va="center", fontsize=9)
    ax.set_xlim(-0.17, 2.38)
    ax.set_ylim(-0.85, 3.4)
    ax.axis("off")
    sp.save(fig, path)


for k in (1, 2, 3):
    goodput_chain(out / f"goodput_chain_step{k}.svg", k)

# ---------------------------------------------------------------------------
# 4. MLPerf Inference: one DGX H100, four rounds (MLSysIM series)
# ---------------------------------------------------------------------------
MLSYSIM = Path(__file__).resolve().parents[4] / "mlsysim"
sys.path.insert(0, str(MLSYSIM))
from mlsysim import Literature  # noqa: E402

rounds = ["v3.1", "v4.0", "v4.1", "v5.0"]
round_lab = ["v3.1\nSep 2023", "v4.0\nMar 2024", "v4.1\nAug 2024", "v5.0\nApr 2025"]
series = {}
for r in Literature.Series.MLPerfInferenceDGXH100.entries:
    if r["scenario"] == "Offline":
        series.setdefault(r["benchmark"], []).append((r["pub_month"], r["round"], r["value"]))
bench = [("gptj-99", "GPT-J", C["compute"]),
         ("llama2-70b-99.9", "Llama 2 70B", C["memory"]),
         ("stable-diffusion-xl", "SDXL", C["data"]),
         ("resnet50", "ResNet-50", C["neutral"]),
         ("bert-99", "BERT", C["neutral"])]


def mlperf(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    for key, name, col in bench:
        pts = sorted(series[key])
        base = pts[0][2]
        xs = [rounds.index(p[1]) for p in pts]
        ys = [p[2] / base for p in pts]
        shown = step >= 2 or key in ("resnet50", "bert-99")
        a = 1.0 if shown else 0.0
        ax.plot(xs, ys, "-o", color=col, lw=2.6, ms=7, alpha=a)
        if key in ("resnet50", "bert-99"):
            if key == "resnet50":
                text(ax, 0.05, 0.9, "ResNet-50, BERT 1.00×",
                     col, shown, va="top", fontsize=12, fontweight="bold")
        else:
            if key == "stable-diffusion-xl":
                text(ax, xs[-1] - 0.08, ys[-1] + 0.12, f"{name} {ys[-1]:.2f}×", col,
                     shown, ha="right", va="bottom", fontsize=12.5, fontweight="bold")
            else:
                text(ax, xs[-1] + 0.08, ys[-1], f"{name} {ys[-1]:.2f}×", col, shown,
                     va="center", fontsize=12.5, fontweight="bold")
    ax.set_xticks(range(4), round_lab, fontsize=11)
    ax.set_xlim(-0.2, 4.1)
    ax.set_ylim(0.8, 3.0)
    ax.set_ylabel("× first round")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, path)


for k in (1, 2):
    mlperf(out / f"mlperf_gains_step{k}.svg", k)

# ---------------------------------------------------------------------------
# 5. Localization ladder: three records, one rung per step
# ---------------------------------------------------------------------------
def ladder(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    rungs = [("S", "Scheduler records", "every job; free",
              "settles S", "cannot see\ninside a gang"),
             ("R", "Job timeline", "every job; bytes per step",
              "settles R", "cannot say why\nsteps are slow"),
             ("P", "Step traces", "few ranks; GB per trace",
              "settles P and\nthe term", "misses\nrestarts")]
    for i, (k, name, cover, settles, blind) in enumerate(rungs):
        y = 2 - i
        shown = i < step
        a = 1.0 if shown else 0.0
        ax.add_patch(plt.Rectangle((0, y - 0.38), 13.3, 0.76, fc="white",
                                   ec=COL[k], lw=1.6, alpha=a))
        text(ax, 0.38, y, k, "white", shown, ha="center", va="center",
             fontsize=14, fontweight="bold",
             bbox=dict(boxstyle="circle,pad=0.25", fc=COL[k], ec="none", alpha=a))
        text(ax, 0.85, y + 0.14, name, COL[k], shown, ha="left", va="center",
             fontsize=13, fontweight="bold")
        text(ax, 0.85, y - 0.19, cover, C["neutral"], shown, ha="left", va="center",
             fontsize=11)
        text(ax, 6.3, y, settles, C["data"], shown, ha="left", va="center",
             fontsize=12, fontweight="bold", linespacing=1.0)
        text(ax, 9.6, y, blind, C["danger"], shown, ha="left", va="center",
             fontsize=11, linespacing=1.0)
        if 0 < i:
            ax.annotate("", xy=(0.38, y + 0.4), xytext=(0.38, y + 0.62),
                        arrowprops=dict(arrowstyle="-|>", color=C["neutral"], lw=1.5),
                        alpha=a)
    for x, h in [(0.85, "Record, read top first"), (6.3, "Establishes"),
                 (9.6, "Blind spot")]:
        ax.text(x, 2.6, h, fontsize=11, color=C["neutral"], fontweight="bold",
                va="center")
    ax.set_xlim(-0.05, 13.4)
    ax.set_ylim(-0.45, 2.8)
    ax.axis("off")
    sp.save(fig, path)


for k in (1, 2, 3):
    ladder(out / f"ladder_step{k}.svg", k)

# ---------------------------------------------------------------------------
# 6. Trace volume: one GPU against 1,024 GPUs, 5 s windows (Sec. 9.3.3)
# ---------------------------------------------------------------------------
sp.hbars(out / "trace_volume.svg",
         [("1 GPU, 5 s", 0.5, "memory", "500 MB"),
          ("1,024 GPUs, 5 s", 512, "danger", "512 GB")],
         "GB", log=True, size=(5.6, 1.15), xlabel="Nsight Systems trace volume (GB, log)")

# ---------------------------------------------------------------------------
# 7. Pool snapshot (Napkin Math 9.2), 2 steps
# ---------------------------------------------------------------------------
def pool(path, step):
    fig, ax = plt.subplots(figsize=(4.4, 1.75))
    segs = [(0.60, C["data"], "executing\n0.60", "white"),
            (0.05, COL["R"], "", "white"),
            (0.15, COL["S_l"], "0.15", COL["S"]),
            (0.20, C["neutral_l"], "free\n0.20", C["text"])]
    left = 0
    for w, col, lab, tc in segs:
        ax.barh(0, w, left=left, height=0.55, color=col, edgecolor="white", lw=1.5,
                hatch="//" if col == COL["S_l"] else None)
        if lab:
            ax.text(left + w / 2, 0, lab, ha="center", va="center", fontsize=11.5,
                    fontweight="bold", color=tc, linespacing=0.95)
        left += w
    ax.annotate("input wait 0.05\n(complete allocation)", xy=(0.625, 0.28),
                xytext=(0.38, 0.72), fontsize=10.5, color=COL["R"],
                fontweight="bold", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=COL["R"], lw=1.2))
    ax.text(0.80, 0.62, "partial gangs", ha="center", va="bottom", fontsize=11.5,
            fontweight="bold", color=COL["S"])
    # assigned bracket above
    ax.plot([0, 0, 0.80, 0.80], [0.36, 0.42, 0.42, 0.36], color=C["text"], lw=1.3)
    ax.text(0.20, 0.5, "assigned to jobs 0.80", ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color=C["text"])
    shown = step >= 2
    a = 1.0 if shown else 0.0
    ax.plot([0, 0, 0.65, 0.65], [-0.36, -0.42, -0.42, -0.36], color=COL["S"], lw=2,
            alpha=a)
    text(ax, 0.0, -0.5, "S = 0.65, complete allocations", COL["S"], shown,
         ha="left", va="top", fontsize=12, fontweight="bold")
    text(ax, 1.0, -0.5, "ceiling 0.80", C["neutral"], shown, ha="right", va="top",
         fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.85, 1.05)
    ax.axis("off")
    sp.save(fig, path)


for k in (1, 2):
    pool(out / f"pool_snapshot_step{k}.svg", k)

# ---------------------------------------------------------------------------
# 8. Runtime terms of the running job (Napkin Math 9.3)
# ---------------------------------------------------------------------------
terms = [("Input waits", 7.7, COL["R"], "7.7%"),
         ("Preemption", 1.4, COL["S"], "1.4%"),
         ("Checkpoints", 0.4, C["neutral"], "0.4%"),
         ("Startup", 0.05, C["neutral"], "0.05%"),
         ("Recovery", 0.03, C["neutral"], "0.03%")]
fig, ax = plt.subplots(figsize=(5.4, 1.6))
for i, (lab, v, col, s) in enumerate(terms):
    y = len(terms) - 1 - i
    ax.barh(y, v, height=0.62, color=col)
    ax.text(-0.15, y, lab, ha="right", va="center", fontsize=13, fontweight="bold")
    ax.text(v + 0.15, y, s, va="center", fontsize=13, fontweight="bold", color=col)
ax.set_xlim(0, 9.5)
ax.set_ylim(-0.6, 4.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Share of held time (%)")
ax.grid(axis="x", color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "runtime_terms.svg")

# ---------------------------------------------------------------------------
# 9. Runtime loss against job size (derived, as the chapter's figure 5)
# ---------------------------------------------------------------------------
MTBF128 = 302 * 3600.0     # s, at 128 GPUs (Napkin Math 9.3)
T_WRITE = 9.8              # s, 70B resumable checkpoint at 100 GB/s
T_REC = 6.2 * 60           # s, recovery budget
INPUT = 0.05 / 0.65        # data stall share of held time
PREEMPT = 20.2 / (24 * 60)  # one forced preemption a day, 20.2 min each
STARTUP = 5.2 / (7 * 24 * 60)


def ft_terms(n):
    m = MTBF128 * 128 / n
    tau = math.sqrt(2 * T_WRITE * m)
    return T_WRITE / tau + tau / (2 * m), T_REC / m


sizes = [128 * 2 ** k for k in range(8)]
fig, ax = plt.subplots(figsize=(5.6, 2.15))
for i, n in enumerate(sizes):
    ck, rc = ft_terms(n)
    parts = [(INPUT, "#D9C3A5"), (PREEMPT, COL["S_l"]), (STARTUP, C["neutral_l"]),
             (ck, "#A0715A"), (rc, COL["R"])]
    b = 0
    for v, col in parts:
        ax.bar(i, v * 100, 0.66, bottom=b, color=col, edgecolor="white", lw=0.6)
        b += v * 100
    ax.text(i, b + 0.4, f"{b:.0f}%", ha="center", va="bottom", fontsize=11.5,
            fontweight="bold", color=C["text"])
ax.bar(8.0, 10, 0.4, color="white", edgecolor=C["text"], lw=1.6, ls="--")
ax.text(8.0, 10.4, "Llama 3\n≤ 10%", ha="center", va="bottom", fontsize=10.5,
        fontweight="bold", color=C["text"], linespacing=0.95)
cross = 6 + math.log2(12928 / 8192)
ax.axvline(cross, color=C["danger"], lw=1.4, ls=":")
ax.annotate("checkpoints + recovery\npass input waits", xy=(cross, 17.5),
            xytext=(3.0, 19.6), ha="center", va="top",
            arrowprops=dict(arrowstyle="-|>", color=C["danger"], lw=1.2), fontsize=10.5, color=C["danger"], fontweight="bold", linespacing=0.95)
ax.text(-0.4, 23.8, "brown: checkpoints, recovery\ntan: input waits; violet: preemption",
        ha="left", va="top", fontsize=10.5, color=C["neutral"], linespacing=1.0)
ax.set_xticks(list(range(8)) + [8.0],
              ["128", "256", "512", "1K", "2K", "4K", "8K", "16K", "16K\nmeasured"],
              fontsize=10.5)
ax.set_xlim(-0.5, 8.4)
ax.set_ylim(0, 24)
ax.set_ylabel("Runtime loss\n(% of held time)")
ax.set_xlabel("GPUs in the job (derived model; same inputs at every size)", fontsize=11)
sp.save(fig, out / "runtime_scaling.svg")

# ---------------------------------------------------------------------------
# 10. MFU of published training runs (PaLM Table 3; Llama 3 Table 4)
# ---------------------------------------------------------------------------
sp.hbars(out / "mfu_published.svg",
         [("GPT-3, 2020", 21.3, "compute", "21.3%"),
          ("MT-NLG, 2021", 30.2, "compute", "30.2%"),
          ("Gopher, 2021", 32.5, "compute", "32.5%"),
          ("PaLM, 2022", 46.2, "compute", "46.2%"),
          ("Llama 3, 8K GPUs", 43, "memory", "43%"),
          ("Llama 3, 16K GPUs", 41, "memory", "41%")],
         "%", log=False, size=(4.3, 2.4), xlabel="MFU (% of peak)")

# ---------------------------------------------------------------------------
# 11. One node against 128 GPUs (Napkin Math 9.4), 2 steps
# ---------------------------------------------------------------------------
def node_fleet(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 1.4))
    ax.barh(1, 180, height=0.5, color=COL["P"])
    ax.barh(0, 180, height=0.5, color=COL["P"])
    ax.barh(0, 65, left=180, height=0.5, color="white", edgecolor=C["danger"],
            hatch="//", lw=1.4)
    ax.text(-4, 1, "8-GPU node", ha="right", va="center", fontsize=13, fontweight="bold")
    ax.text(-4, 0, "128-GPU fleet", ha="right", va="center", fontsize=13,
            fontweight="bold")
    ax.text(90, 1, "180 ms", ha="center", va="center", color="white", fontsize=13,
            fontweight="bold")
    ax.text(90, 0, "180 ms of node work", ha="center", va="center", color="white",
            fontsize=12.5, fontweight="bold")
    ax.text(212.5, 0, "+65 ms", ha="center", va="center", color=C["danger"],
            fontsize=12.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    shown = step >= 2
    text(ax, 252, 1, "MFU 65%", COL["P"], shown, va="center", fontsize=13,
         fontweight="bold")
    text(ax, 252, 0, "MFU 48%", C["danger"], shown, va="center", fontsize=13,
         fontweight="bold")
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 100, 200, 245], ["0", "100", "200", "245 ms"])
    sp.save(fig, path)


for k in (1, 2):
    node_fleet(out / f"node_fleet_step{k}.svg", k)

# ---------------------------------------------------------------------------
# 12. H100 roofline with batch-one decode and a measured LayerNorm
# ---------------------------------------------------------------------------
PEAK, BW = 989.0, 3.35  # TFLOP/s (FP16 dense), TB/s
fig, ax = plt.subplots(figsize=(5.6, 2.0))
xs = np.logspace(-1, 3.6, 300)
roof = np.minimum(PEAK, xs * BW)
ridge = PEAK / BW
ax.plot(xs[xs <= ridge], roof[xs <= ridge], color=C["memory"], lw=3)
ax.plot(xs[xs >= ridge], roof[xs >= ridge], color=C["compute"], lw=3)
ax.axvline(ridge, color=C["neutral"], lw=1, ls="--")
ax.text(ridge * 1.1, 2.0, "ridge\n295 FLOP/byte", color=C["neutral"], fontsize=11,
        va="bottom", linespacing=0.95)
ax.text(3800, PEAK * 1.35, "989 TFLOP/s (FP16)", color=C["compute"], fontsize=11.5,
        fontweight="bold", ha="right")
ax.text(40, 12, "3.35 TB/s", color=C["memory"], fontsize=11.5,
        fontweight="bold", rotation=31, ha="center")
ax.plot([1.0], [BW], "o", ms=11, color=C["danger"], mec="white", mew=1.5, zorder=5)
ax.annotate("batch-1 decode: 1 FLOP/byte,\n3.35 TFLOP/s = 0.34% of peak", (1.0, BW),
            xytext=(0.12, 160), fontsize=11.5, fontweight="bold", color=C["danger"],
            arrowprops=dict(arrowstyle="-", color=C["danger"], lw=1.1), linespacing=0.95)
ax.plot([1.8], [5.0], "s", ms=9, color=C["memory"], mec="white", mew=1.5, zorder=5)
ax.annotate("LayerNorm, measured:\n1.8 FLOP/byte at 2.8 TB/s", (1.8, 5.0),
            xytext=(9, 0.25), fontsize=11, color=C["memory"], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=C["memory"], lw=1.1), linespacing=0.95)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1, 4000)
ax.set_ylim(0.1, 3000)
ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
ax.set_ylabel("TFLOP/s")
ax.minorticks_off()
sp.save(fig, out / "roofline_decode.svg")

# ---------------------------------------------------------------------------
# 13. Overlap on a shared device (schematic durations), 3 steps
# ---------------------------------------------------------------------------
def overlap(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 2.15))
    panels = [("serialized", [(0, 2), (2, 2), (4, 2)], (6, 2), None, 8, "exposed"),
              ("overlapped, idle device", [(0, 2), (2, 2), (4, 2)], (2, 2), None, 6,
               "hidden"),
              ("overlapped, shared device", [(0, 2), (2, 2.8), (4.8, 2)], (2, 2),
               (4.0, 0.8), 6.8, "hidden")]
    for i, (name, ks, coll, cont, end, clab) in enumerate(panels):
        y0 = 2 - i
        shown = i < step
        a = 1.0 if shown else 0.0
        text(ax, -0.1, y0 + 0.2, name, C["text"], shown, ha="right", va="center",
             fontsize=11.5, fontweight="bold")
        for (s, w) in ks:
            ax.barh(y0 + 0.2, w, left=s, height=0.26, color=C["memory_l"],
                    edgecolor=C["memory"], lw=1.2, alpha=a)
        ax.barh(y0 - 0.12, coll[1], left=coll[0], height=0.26, color="#fef3c7",
                edgecolor="#d97706", lw=1.2, alpha=a)
        text(ax, coll[0] + coll[1] / 2, y0 - 0.12, clab, "#b45309", shown,
             ha="center", va="center", fontsize=10.5, fontweight="bold")
        if cont:
            ax.barh(y0 + 0.2, cont[1], left=cont[0], height=0.26, color=C["danger_l"],
                    edgecolor=C["danger"], lw=1.2, alpha=a)
            text(ax, cont[0] + cont[1] + 2.15, y0 + 0.2, "contention", C["danger"],
                 shown, va="center", fontsize=10.5, fontweight="bold")
        ax.plot([end, end], [y0 - 0.3, y0 + 0.38], color=C["text"], lw=3, alpha=a)
        text(ax, end + 0.12, y0 - 0.12, f"step ends {end:g}", C["text"], shown,
             va="center", fontsize=10.5)
    text(ax, -0.1, -0.45, "blue: compute stream; amber: communication stream",
         C["neutral"], True, ha="right", va="center", fontsize=9.5)
    ax.text(10.5, -0.45, "schematic durations", ha="right", va="center", fontsize=9.5,
            color=C["neutral"])
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.65, 2.6)
    ax.axis("off")
    sp.save(fig, path)


for k in (1, 2, 3):
    overlap(out / f"overlap_step{k}.svg", k)

# ---------------------------------------------------------------------------
# 14. Waterfall of the running job's pool (Napkin Math 9.5), 3 steps, and
#     the synthesis version with Llama 3 405B beside it
# ---------------------------------------------------------------------------
def waterfall(path, step, compare=False):
    fig, ax = plt.subplots(figsize=(4.2, 2.6) if compare else (5.6, 2.1))
    levels = [100, 65, 59, 28]
    names = ["Fleet\ncapacity", "After\nscheduling", "After\nruntime", "After\nprogram"]
    losses = [None, ("-35%", COL["S"]), ("-6%", COL["R"]), ("-31%", COL["P"])]
    llama = [100, 100, 90, 37]
    xoff = -0.12 if compare else 0
    for i, v in enumerate(levels):
        shown = i <= step
        a = 1.0 if shown else 0.0
        col = C["data"] if i == 3 else C["neutral_l"]
        ax.bar(i + xoff, v, 0.5, color=col, edgecolor=C["neutral"], lw=1, alpha=a)
        text(ax, i + xoff, v / 2 if i < 3 else v / 2, f"{v}%",
             "white" if i == 3 else C["text"], shown, ha="center", va="center",
             fontsize=11.5 if compare else 13, fontweight="bold")
        if losses[i]:
            lab, lc = losses[i]
            ax.bar(i + xoff, levels[i - 1] - v, 0.5, bottom=v, color="white",
                   edgecolor=lc, lw=1.6, ls="--", alpha=a)
            text(ax, i + xoff, levels[i - 1] + 3, lab, lc, shown, ha="center",
                 va="bottom", fontsize=13, fontweight="bold")
        if compare:
            ax.bar(i + 0.27, llama[i], 0.18, color="white", edgecolor=C["text"],
                   lw=1.4)
            if llama[i] < 100:
                ax.text(i + 0.27, llama[i] + 3, f"{llama[i]}", ha="center",
                        va="bottom", fontsize=10.5, color=C["text"])
    if compare:
        ax.text(-0.45, 127, "hollow bars: Llama 3 405B on its dedicated cluster",
                ha="left", va="top", fontsize=11, color=C["text"])
    ax.set_xticks(range(4), names, fontsize=11.5)
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylim(0, 130 if compare else 112)
    ax.set_ylabel("% of fleet capacity")
    sp.save(fig, path)


for k in (1, 2, 3):
    waterfall(out / f"waterfall_step{k}.svg", k)
waterfall(out / "waterfall_compare.svg", 3, compare=True)

# ---------------------------------------------------------------------------
# 15. Each lever at its ceiling (Sec. 9.8.1), 2 steps
# ---------------------------------------------------------------------------
lever_rows = [("Today", 0.28, "neutral", "MPG 0.28"),
              ("A. Scheduling: S to 0.80", 0.35, "network", "0.35 (+23%)"),
              ("B. Input pipeline: R to 0.98", 0.31, "danger", "0.31 (+9%)"),
              ("C. Program: P to 0.65", 0.38, "compute", "0.38 (+35%)")]
COLKEY = {"network": COL["S"], "danger": COL["R"], "compute": COL["P"],
          "neutral": C["neutral"]}


def levers(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 1.6))
    for i, (lab, v, ck, s) in enumerate(lever_rows):
        y = 3 - i
        shown = i == 0 or step >= 2
        a = 1.0 if shown else 0.0
        ax.barh(y, v, height=0.62, color=COLKEY[ck], alpha=a)
        ax.text(-0.006, y, lab, ha="right", va="center", fontsize=12.5,
                fontweight="bold", color=C["text"])
        text(ax, v + 0.005, y, s, COLKEY[ck], shown, va="center", fontsize=12.5,
             fontweight="bold")
    ax.set_xlim(0, 0.47)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Pool MPG with the lever at its ceiling")
    sp.save(fig, path)


for k in (1, 2):
    levers(out / f"levers_step{k}.svg", k)

# ---------------------------------------------------------------------------
# 16. Iso-goodput curves (Sec. 9.9.1)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 2.15))
P = np.linspace(0.30, 0.75, 200)
R0, P0 = 0.90, 0.48
base = R0 * P0
ax.fill_between(P, base / P, 1.05, where=base / P < 1.05, color=C["data_l"], alpha=0.7)
for c in (0.30, 0.35, 0.50, 0.55):
    ax.plot(P, c / P, color=C["grid"], lw=1.0)
ax.plot(P, base / P, color=C["text"], lw=2.4)
ax.text(0.745, 0.88, "bold curve:\nR × P = 0.43", fontsize=11, ha="right",
        va="center", color=C["text"], fontweight="bold", linespacing=0.95)
ax.plot([P0], [R0], "o", ms=11, color=C["danger"], zorder=5)
ax.text(P0 - 0.012, R0 - 0.02, "running job", ha="right", va="top", fontsize=11.5,
        fontweight="bold", color=C["danger"])
ax.annotate("", xy=(0.65, R0), xytext=(P0 + 0.01, R0),
            arrowprops=dict(arrowstyle="-|>", color=COL["P"], lw=2.2))
ax.text(0.565, R0 - 0.02, "P to 0.65: +35%", ha="center", va="top", fontsize=11.5,
        fontweight="bold", color=COL["P"])
ax.annotate("", xy=(0.65, base / 0.65), xytext=(0.65, R0 - 0.01),
            arrowprops=dict(arrowstyle="-|>", color=COL["R"], lw=1.8, ls="--"))
ax.plot([0.65], [base / 0.65], "o", ms=9, mfc="white", mec=COL["R"], mew=2, zorder=5)
ax.text(0.66, 0.73, "break-even:\nR may fall 26%", ha="left", va="center",
        fontsize=11, fontweight="bold", color=COL["R"], linespacing=0.95)
ax.axhline(1.0, color=C["neutral"], lw=1, ls=":")
ax.annotate("", xy=(P0, 1.0), xytext=(P0, R0 + 0.005),
            arrowprops=dict(arrowstyle="-|>", color=C["neutral"], lw=1.6))
ax.text(P0 + 0.008, 0.985, "R headroom: +11% at most", ha="left", va="top",
        fontsize=10.5, color=C["neutral"], fontweight="bold")
ax.text(0.31, 0.57, "shaded: the trade raises R × P", fontsize=10, color=C["data"],
        fontweight="bold")
ax.set_xlim(0.30, 0.75)
ax.set_ylim(0.55, 1.05)
ax.set_xlabel("Program goodput P")
ax.set_ylabel("Runtime goodput R")
sp.save(fig, out / "isogoodput.svg")

# ---------------------------------------------------------------------------
# 17. Llama 3 405B at two sizes: time against GPU-hours (derived)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 2.0))
pts = [(1.0, 1.0, "8,192 GPUs, MFU 43%"), (0.5 * 0.43 / 0.41 * 1.0, 0.43 / 0.41,
                                            "16,384 GPUs, MFU 41%")]
t2 = 0.43 / (2 * 0.41)
pts[1] = (t2, 0.43 / 0.41, pts[1][2])
ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C["neutral"], lw=1.4, ls="--")
for (x, y, lab), col, off in zip(pts, [C["memory"], C["compute"]],
                                 [(0.0, -0.007), (0.02, 0.006)]):
    ax.plot([x], [y], "o", ms=13, color=col, zorder=5)
    ax.text(x + off[0], y + off[1], lab, ha="right" if off[0] <= 0 else "left",
            va="top" if off[1] < 0 else "bottom", fontsize=12, fontweight="bold",
            color=col)
ax.text(0.56, 1.006, "neither point dominates:\none is faster, one is cheaper",
        ha="center", va="center", fontsize=11, color=C["neutral"], linespacing=0.95)
ax.set_xlim(0.4, 1.1)
ax.set_ylim(0.975, 1.075)
ax.set_xlabel("Time to result (relative to 8,192 GPUs)")
ax.set_ylabel("GPU-hours per\nunit of work")
ax.set_yticks([1.00, 1.025, 1.05], ["1.00", "1.025", "1.05"])
sp.save(fig, out / "frontier_llama3.svg")

print("figures written to", out)
