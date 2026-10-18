#!/usr/bin/env python3
"""Data plots and diagrams for Volume II, Lecture 1 (Introduction).

Every value is from the rendered chapter (Volume II, Introduction) unless noted:
  - Hook and Napkin Math 1.1 (Sec. 1.5.5): 175B parameters, FP16 ring
    AllReduce about 700 GB per endpoint per step; 1.2 s of local compute
    (a chosen scenario input); 200 Gb/s InfiniBand (25 GB/s): about 28 s,
    compute-time fraction 4.1 percent; 100 Gb/s Ethernet: 56 s, 2.1 percent.
  - Growth decomposition (Sec. 1.1, figure 1b): fitted here from MLSysIM's
    copy of Epoch AI's notable-model records (146 runs, 2012 to 2024 that
    report count and duration): compute 3.9x per year = count 1.6x x
    duration 1.3x x delivered rate per accelerator 2.0x; count and duration
    give 51 percent of the exponent.
  - Scaling law (Sec. 1.3.3, theorem 1.1): the parameter term A / N^alpha of
    the Hoffmann et al. (2022) parametric fit, A = 406.4, alpha = 0.34, read
    from MLSysIM's Chinchilla registry. Halving it needs 2^(1/0.34) = 7.7x
    the parameters (derived).
  - IsoFLOP curves (Sec. 1.3.2, figure 9a): the same fit,
    L(N, D) = E + A/N^alpha + B/D^beta with E = 1.69, B = 410.7, beta = 0.28,
    D = C / (6N), for budgets 6e18 and 3e21 FLOPs (the paper's range) and one
    between.
  - Bandwidth (Sec. 1.2): NVLink 450 GB/s and InfiniBand NDR 50 GB/s per
    direction per accelerator, about 9x.
  - Ring traffic (Sec. 1.2.3 figure): FP32 gradient 700 GB; ring traffic
    about 2 times (FP32), about 1 times the FP32 size in FP16.
  - Communication intensity (Sec. 1.2.4): worker at 8.6e15 FLOPs and 700 GB,
    8.1e-5 B/FLOP; A100 with one 200 Gb/s link: 25e9 / 312e12 = 8.0e-5;
    workloads Alpha (10 MB, 1e14 FLOPs) and Beta (1 GB, 1e13 FLOPs) from the
    Sec. 1.2 self-check.
  - Llama 3 405B, Dubey et al. 2024, Table 4: 8,192 GPUs at 430 TFLOP/s per
    GPU (43 percent BF16 MFU) and 16,384 GPUs at 400 (41 percent), 16M tokens
    per batch in both.
  - Fleet Law timeline (Sec. 1.5.4, figure 15): schematic, redrawn.
  - Efficiency measures (Sec. 1.5.4): GPT-3 about 20 percent of peak, run
    average (Patterson et al. 2021); Megatron-LM 52 percent of peak on 3,072
    A100s (Narayanan et al. 2021); Llama 3 405B 41 percent MFU on 16,384 GPUs
    (Dubey et al. 2024, Table 4); Llama 3 more than 90 percent effective
    training time (Dubey et al. 2024, Sec. 3.3.4).
  - Reliability gap (Sec. 1.4.1, equation 2, figure 13): a^N; 0.999^1,000 =
    36.8 percent, 0.999^10,000 = 0.0045 percent, 0.9999^10,000 = 36.8 percent.

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent

# Volume II C3 roles (volumes/vol2/palette.md): blue compute, amber
# communication, violet coordination; red for faults and stragglers.
C1, C1L, C1T = "#1F407A", "#EAF1F8", "#14294E"
C2, C2L, C2T = "#d97706", "#fef3c7", "#b45309"
C3, C3L, C3T = "#7c3aed", "#f3e8ff", "#6d28d9"
RED, REDL = "#dc2626", "#fee2e2"
GRAY = "#64748b"


def mlsysim_root():
    return Path(__file__).resolve().parents[4] / "mlsysim"


def epoch_rows():
    sys.path.insert(0, str(mlsysim_root()))
    from mlsysim import Literature  # noqa: E402
    rows = [r for r in Literature.Series.NotableModelTrainingRuns.entries
            if r["training_compute_flop"] and "2012" <= r["date"][:4] <= "2024"]
    return [r for r in rows if r["hardware_quantity"] and r["training_time_hours"]]


def chinchilla():
    """The Hoffmann et al. 2022 parametric loss fit, L = E + A/N^alpha +
    B/D^beta (Chinchilla paper, section D.2, equation 10). Read from the
    MLSysIM registry when this checkout carries the fit; otherwise use the
    paper's published constants."""
    import yaml
    f = mlsysim_root() / "mlsysim" / "literature" / "data" / "chinchilla.yaml"
    if f.exists():
        d = yaml.safe_load(f.read_text())
        if "ParametricLossE" in d:
            return {k: float(d[k]["value"]) for k in
                    ("ParametricLossE", "ParametricLossA", "ParametricLossB",
                     "ParametricLossAlpha", "ParametricLossBeta")}
    return {"ParametricLossE": 1.69, "ParametricLossA": 406.4,
            "ParametricLossB": 410.7, "ParametricLossAlpha": 0.34,
            "ParametricLossBeta": 0.28}


# 1. Hook: one synchronized step -------------------------------------------
def hook(path, k):
    fig, ax = plt.subplots(figsize=(3.4, 1.3))
    ax.barh(0, 1.2, color=C1, height=0.55)
    ax.text(0, 0.5, "compute 1.2 s", ha="left", fontsize=11,
            fontweight="bold", color=C1T)
    if k >= 2:
        ax.barh(0, 28, left=1.2, color=C2, height=0.55)
        ax.text(15.2, 0, "AllReduce 28 s", va="center", ha="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(29.2, 0.5, "step 29.2 s", ha="right",
                fontsize=11, fontweight="bold", color=C2T)
    else:
        ax.text(29.2, 0.5, "step = ?", ha="right", fontsize=11,
                fontweight="bold", color=C2T)
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.45, 0.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("seconds per step (no overlap)", fontsize=10)
    ax.tick_params(labelsize=10)
    sp.save(fig, path)


hook(out / "hook_step1.svg", 1)
hook(out / "hook_step2.svg", 2)

# 2. Growth decomposition (Epoch AI via MLSysIM) ----------------------------
runs = epoch_rows()


def _year(s):
    import datetime as dt
    d = dt.date.fromisoformat(s)
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def _factor(value):
    xs = [_year(r["date"]) for r in runs]
    return 10 ** np.polyfit(xs, np.log10([value(r) for r in runs]), 1)[0]


g_compute = _factor(lambda r: r["training_compute_flop"])
g_count = _factor(lambda r: r["hardware_quantity"])
g_dur = _factor(lambda r: r["training_time_hours"])
g_rate = _factor(lambda r: r["training_compute_flop"]
                 / (r["hardware_quantity"] * r["training_time_hours"] * 3600))
share = (np.log(g_count) + np.log(g_dur)) / np.log(g_compute)
assert len(runs) == 146 and abs(g_compute - 3.9) < 0.05 and abs(share - 0.51) < 0.01

fig, ax = plt.subplots(figsize=(4.4, 2.1))
rows = [("Training compute", g_compute, C["neutral"]),
        ("Accelerator count", g_count, C1),
        ("Training duration", g_dur, C1),
        ("Rate per accelerator", g_rate, C["compute"])]
for i, (lab, v, col) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, np.log10(v), color=col, height=0.6)
    ax.text(-0.005, y, lab, ha="right", va="center", fontsize=13,
            fontweight="bold", color=C["text"])
    ax.text(np.log10(v) + 0.01, y, f"{v:.1f}× per year", va="center",
            fontsize=13, fontweight="bold", color=col)
ax.text(0, -0.95, f"blue bars together give {share:.0%} of the growth",
        color=C1, fontsize=12, fontweight="bold", ha="left", va="center")
ax.set_xlim(0, 0.95)
ax.set_ylim(-1.4, 3.5)
ax.set_yticks([])
ax.set_xticks([])
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
sp.save(fig, out / "growth_bars.svg")

# 3. Power law: each halving of the parameter term costs 7.7x ---------------
fit = chinchilla()
A, alpha = fit["ParametricLossA"], fit["ParametricLossAlpha"]
mult = 2 ** (1 / alpha)
fig, ax = plt.subplots(figsize=(3.7, 2.05))
ns = np.geomspace(1e8, 1e12, 200)
ax.plot(ns, A / ns ** alpha, color=C1, lw=3)
n0 = 1e8
for i in range(4):
    n1 = n0 * mult
    y0, y1 = A / n0 ** alpha, A / n1 ** alpha
    ax.plot([n0, n1], [y0, y0], color=C["compute"], lw=1.8, ls="--")
    ax.plot([n1, n1], [y0, y1], color=C["compute"], lw=1.8, ls="--")
    ax.plot([n0], [y0], "o", color=C1, ms=8)
    ax.text(np.sqrt(n0 * n1), y0 * 1.06, f"×{mult:.1f}", ha="center",
            va="bottom", fontsize=11, fontweight="bold", color=C["compute"])
    n0 = n1
ax.plot([n0], [A / n0 ** alpha], "o", color=C1, ms=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("parameters N", fontsize=11)
ax.set_ylabel("A/N$^{\\alpha}$ (reducible loss)", fontsize=11)
ax.tick_params(labelsize=10)
ax.text(1.2e8, 0.016, "each halving costs\n"
        f"{mult:.1f}× the parameters", fontsize=11, color=C1,
        fontweight="bold", va="bottom")
ax.set_ylim(0.012, 0.8)
ax.minorticks_off()
sp.save(fig, out / "powerlaw.svg")

# 4. IsoFLOP valleys -----------------------------------------------------------
E, B, beta = fit["ParametricLossE"], fit["ParametricLossB"], fit["ParametricLossBeta"]
fig, ax = plt.subplots(figsize=(5.8, 2.0))
params = np.geomspace(1e7, 1e11, 400)
for budget, shade, lab in [(6e18, C1L, "6×10$^{18}$ FLOPs"),
                           (1.3e20, "#93b0d6", "1.3×10$^{20}$"),
                           (3e21, C1, "3×10$^{21}$")]:
    loss = E + A / params ** alpha + B / (budget / (6 * params)) ** beta
    col = shade if shade != C1L else "#7aa0cc"
    ax.plot(params, loss, color=col, lw=3)
    i = loss.argmin()
    ax.plot([params[i]], [loss[i]], "o", color=C["compute"], ms=9, zorder=5)
    ax.text(params[i], loss[i] + 0.07, lab, ha="center", va="bottom", fontsize=11,
            fontweight="bold", color=C1T)
ax.set_xscale("log")
ax.set_ylim(1.8, 4.0)
ax.set_xlim(1e7, 1e11)
ax.set_xlabel("parameters N (budget fixed per curve, D = C / 6N)")
ax.set_ylabel("training loss")
ax.text(1.3e7, 1.85, "orange dot: the optimal model size for that budget",
        fontsize=11, color=C["compute"], fontweight="bold", va="bottom")
ax.minorticks_off()
sp.save(fig, out / "isoflop.svg")

# 5. Inside the node against out of it -----------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 1.5))
for y, lab, v, col in [(1, "NVLink", 450, C1), (0, "InfiniBand", 50, C2)]:
    ax.barh(y, v, color=col, height=0.55)
    ax.text(-8, y, lab, ha="right", va="center", fontsize=12, fontweight="bold",
            color=C["text"])
    ax.text(v + 8, y, f"{v} GB/s", va="center", fontsize=12, fontweight="bold",
            color=col)
ax.text(300, 0.0, "about 9× less", fontsize=12, fontweight="bold",
        color=RED, va="center")
ax.set_xlim(0, 560)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("GB/s per accelerator, each direction", fontsize=10)
ax.tick_params(labelsize=10)
sp.save(fig, out / "node_bw.svg")


# 6. Scale discontinuity: one node, then a fleet ------------------------------
def rbox(ax, x, y, w, h, fc, ec, lw=1.5, ls="-", r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def node(ax, x, y, w=2.25, h=0.85, state="ok"):
    ec = {"ok": "#94a3b8", "slow": RED, "failed": RED}[state]
    fc = {"ok": "white", "slow": "white", "failed": REDL}[state]
    rbox(ax, x, y, w, h, fc, ec, lw=2.0 if state != "ok" else 1.3,
         ls="--" if state == "slow" else "-")
    if state == "ok":
        for j in range(4):
            ax.add_patch(Rectangle((x + 0.22 + j * 0.49, y + 0.25), 0.34, 0.34, fc=C1, ec=C1))
    else:
        ax.text(x + w / 2, y + h / 2, state, ha="center", va="center", fontsize=12,
                color=RED, fontweight="bold")


def discontinuity(path, k):
    fig, ax = plt.subplots(figsize=(5.9, 2.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    # left: one node
    ax.text(1.9, 5.72, "One node", ha="center", fontsize=14, fontweight="bold", color=C["text"])
    rbox(ax, 0.2, 0.2, 3.4, 5.1, "#f8fafc", "#94a3b8")
    for j in range(4):
        ax.add_patch(Rectangle((0.5 + j * 0.78, 4.4), 0.5, 0.5, fc=C1, ec=C1))
        ax.plot([0.75 + j * 0.78] * 2, [4.4, 2.3], color=C1, lw=2.5)
        for m in range(3):
            ax.add_patch(Rectangle((0.48 + j * 0.78, 1.7 + m * 0.2), 0.54, 0.14,
                                   fc="white", ec=C1, lw=1))
    ax.text(1.9, 3.4, "memory wall", ha="center", va="center", fontsize=13, color=RED,
            fontweight="bold", bbox=dict(fc=REDL, ec=RED, ls="--", lw=1.2, boxstyle="square,pad=0.35"))
    ax.text(1.9, 0.85, "no peers to\nwait for", ha="center", va="center", fontsize=11, color=GRAY)
    if k >= 2:
        ax.plot([4.05, 4.05], [0.2, 5.8], color="#cbd5e1", lw=1, ls="--")
        ax.text(8.2, 5.72, "A fleet", ha="center", fontsize=14, fontweight="bold", color=C["text"])
        rbox(ax, 5.5, 4.55, 5.4, 0.7, C3L, C3, lw=1.6)
        ax.text(8.2, 4.9, "scheduler and recovery", ha="center", va="center",
                fontsize=12, color=C3T, fontweight="bold")
        ax.text(8.2, 2.85, "network fabric: the network wall", ha="center", va="center",
                fontsize=12, color=C2T, fontweight="bold",
                bbox=dict(fc=C2L, ec=C2, lw=1.4, boxstyle="round,pad=0.3"))
        states_top = ["ok", "ok", "failed"]
        states_bot = ["ok", "slow", "ok"]
        for j in range(3):
            x = 4.4 + j * 2.6
            node(ax, x, 3.45, state=states_top[j])
            node(ax, x, 1.4, state=states_bot[j])
            ax.plot([x + 1.12] * 2, [3.45, 3.15], color=C2, lw=1.4)
            ax.plot([x + 1.12] * 2, [2.25, 2.55], color=C2, lw=1.4)
        ax.text(8.2, 0.7, "the other nodes wait for it", ha="center",
                va="center", fontsize=12, color=RED, fontweight="bold")
    sp.save(fig, path)


discontinuity(out / "discontinuity_step1.svg", 1)
discontinuity(out / "discontinuity_step2.svg", 2)


# 7. Independent tasks against a shared step ---------------------------------
def lane(ax, y, x0, x1, fc, ec, text="", tc=C["text"], ls="-", lw=1.4):
    rbox(ax, x0, y - 0.3, x1 - x0, 0.6, fc, ec, lw=lw, ls=ls, r=0.06)
    if text:
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=12,
                color=tc, fontweight="bold")


def coupling(path, k):
    fig, ax = plt.subplots(figsize=(5.9, 2.3))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(-0.3, 5.2)
    ax.axis("off")
    ax.text(2.9, 4.85, "MapReduce", ha="center", fontsize=14, fontweight="bold")
    for i, lab in enumerate(["map 1", "map 2", "map 3"]):
        y = 3.95 - i * 0.98
        ax.text(0.9, y, lab, ha="right", va="center", fontsize=12, color=GRAY)
        if i < 2:
            lane(ax, y, 1.05, 3.3 + 0.35 * i, C1L, C1)
            ax.text(3.45 + 0.35 * i, y, "done", va="center", fontsize=12,
                    color=C["data"], fontweight="bold")
        else:
            lane(ax, y, 1.05, 2.55, REDL, RED, "failed", RED)
            lane(ax, y, 2.7, 4.3, C1L, C1, "re-run", C1T)
    ax.text(0.9, 1.05, "reduce", ha="right", va="center", fontsize=12, color=GRAY)
    lane(ax, 1.05, 3.6, 5.3, C1L, C1)
    ax.text(2.9, 0.1, "the others do not wait", ha="center", fontsize=12,
            color=C["data"], fontweight="bold")
    if k >= 2:
        ax.plot([5.75, 5.75], [-0.2, 5.1], color="#cbd5e1", lw=1, ls="--")
        ax.text(9.2, 4.85, "Synchronous training", ha="center", fontsize=14, fontweight="bold")
        for i, lab in enumerate(["GPU 0", "GPU 1", "GPU 2"]):
            y = 3.95 - i * 0.98
            ax.text(7.3, y, lab, ha="right", va="center", fontsize=12, color=GRAY)
            if i < 2:
                lane(ax, y, 7.45, 8.6 + 0.3 * i, C1L, C1)
                lane(ax, y, 8.6 + 0.3 * i, 10.7, "#e2e8f0", "#94a3b8", "wait", GRAY,
                     ls="--", lw=1)
            else:
                lane(ax, y, 7.45, 10.7, REDL, RED, "slow", RED, ls="--")
        rbox(ax, 10.8, 1.65, 0.75, 2.6, C2L, C2, lw=1.6, r=0.05)
        ax.text(11.175, 2.95, "AllReduce", rotation=90, ha="center", va="center",
                fontsize=12, color=C2T, fontweight="bold")
        ax.text(7.3, 1.05, "update", ha="right", va="center", fontsize=12, color=GRAY)
        lane(ax, 1.05, 10.8, 12.0, C1L, C1)
        ax.text(9.2, 0.1, "all wait for the slowest", ha="center", fontsize=12,
                color=RED, fontweight="bold")
    sp.save(fig, path)


coupling(out / "coupling_step1.svg", 1)
coupling(out / "coupling_step2.svg", 2)

# 8. Ring traffic ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.8, 1.45))
for y, lab, v, col, txt in [(2, "FP32 gradient", 700, C["memory"], "700 GB"),
                            (1, "ring, FP32", 1400, C["network"], "1,400 GB"),
                            (0, "ring, FP16", 700, C["network"], "700 GB")]:
    ax.barh(y, v, color=col, height=0.6)
    ax.text(-20, y, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text(v + 20, y, txt, va="center", fontsize=11, fontweight="bold", color=col)
ax.set_xlim(0, 1900)
ax.set_ylim(-0.5, 2.5)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("GB sent per worker per step", fontsize=10)
ax.tick_params(labelsize=10)
sp.save(fig, out / "ring_bars.svg")


# 9. Napkin Math 1.1: two fabrics ---------------------------------------------
def napkin(path, k):
    fig, ax = plt.subplots(figsize=(3.3, 1.25))
    data = [("InfiniBand", 28), ("Ethernet", 56)]
    for i, (lab, comm) in enumerate(data):
        y = 1 - i
        a = 1.0 if i < k else 0.0
        ax.barh(y, 1.2, color=C1, height=0.55, alpha=a)
        ax.barh(y, comm, left=1.2, color=C2, height=0.55, alpha=a)
        ax.text(-1, y, lab, ha="right", va="center", fontsize=11, fontweight="bold")
        ax.text(1.2 + comm + 1, y, f"{1.2 + comm:.1f} s", va="center",
                fontsize=11, fontweight="bold", color=C2T, alpha=a)
    ax.set_xlim(0, 72)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("seconds per step", fontsize=10)
    ax.tick_params(labelsize=10)
    sp.save(fig, path)


napkin(out / "napkin_step1.svg", 1)
napkin(out / "napkin_step2.svg", 2)

# 10. Communication intensity and the fabric balance point --------------------
BAL = 25e9 / 312e12          # A100 with one 200 Gb/s link, B/FLOP
OURS = 700e9 / (8192 * 6 * 175e9)


def ci(path, k):
    fig, ax = plt.subplots(figsize=(3.9, 2.0))
    x = np.geomspace(1e-8, 1e-2, 300)
    y = np.maximum(1.0, x / BAL)
    ax.plot(x, y, color=C["text"], lw=3)
    ax.axvline(BAL, color=RED, lw=1.6, ls="--")
    ax.text(BAL / 1.4, 150, "fabric balance\n8.0\u00d710$^{-5}$ B/FLOP\n(A100, one 200 Gb/s link)",
            color=RED, fontsize=10, fontweight="bold", va="top", ha="right")
    ax.text(3e-7, 1.45, "compute sets the step", color=C1, fontsize=10.5, fontweight="bold", ha="center")
    ax.text(1.5e-3, 2.2, "network\nsets the\nstep", color=C2T, fontsize=10.5, fontweight="bold",
            ha="left")
    if k >= 2:
        ax.plot([OURS], [max(1, OURS / BAL)], "o", ms=11, color=C2, mec="white", mew=1.5, zorder=5)
        ax.annotate("175B worker\n8.1\u00d710$^{-5}$", (OURS, 1.0), xytext=(-60, 26),
                    textcoords="offset points", fontsize=10, fontweight="bold", color=C2T,
                    ha="right", arrowprops=dict(arrowstyle="-", color=C2T, lw=1))
    if k >= 3:
        for lab, v, off in [("Alpha 10$^{-7}$", 1e-7, (0, -9)), ("Beta 10$^{-4}$", 1e-4, (10, -6))]:
            yy = max(1, v / BAL)
            ax.plot([v], [yy], "s", ms=10, color=C3, mec="white", mew=1.5, zorder=5)
            ax.annotate(lab, (v, yy), xytext=off, textcoords="offset points",
                        fontsize=10, fontweight="bold", color=C3T,
                        ha="center" if off[0] == 0 else "left", va="top" if off[0] == 0 else "center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.28, 200)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xlabel("bytes sent per local FLOP", fontsize=10)
    ax.set_ylabel("step time", fontsize=10)
    ax.set_xticks([1e-8, 1e-6, 1e-4, 1e-2])
    ax.tick_params(labelsize=10)
    sp.save(fig, path)


for k in (1, 2, 3):
    ci(out / f"ci_step{k}.svg", k)

# 11. Llama 3: same batch over twice the GPUs -------------------------------
fig, ax = plt.subplots(figsize=(3.4, 1.55))
for y, lab, v, mfu in [(1, "8,192", 430, "43% MFU"), (0, "16,384", 400, "41% MFU")]:
    ax.barh(y, v, color=C1, height=0.55)
    ax.text(-10, y, lab, ha="right", va="center", fontsize=12, fontweight="bold")
    ax.text(v + 8, y, f"{v}, {mfu}", va="center", fontsize=11, fontweight="bold", color=C1)
ax.set_xlim(0, 700)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("BF16 TFLOP/s per GPU", fontsize=10)
ax.tick_params(labelsize=10)
ax.text(-10, 1.55, "GPUs", ha="right", va="bottom", fontsize=10, color=C["neutral"])
sp.save(fig, out / "llama_mfu.svg")


# 12. C3: three axes and the evidence each leaves ------------------------------
def c3(path):
    fig, ax = plt.subplots(figsize=(5.9, 1.95))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    cols = [("Compute", "$C_1$:  T$_{compute}$/N", "kernel time, MFU,\nmemory counters", C1L, C1, C1T),
            ("Communication", "$C_2$:  T$_{comm}$(N)", "link utilization,\ncollective timing", C2L, C2, C2T),
            ("Coordination", "$C_3$:  T$_{sync}$(N)", "barrier waits,\nrecovery records", C3L, C3, C3T)]
    for i, (name, term, ev, fc, ec, tc) in enumerate(cols):
        x = 0.05 + i * 4.0
        rbox(ax, x, 0.1, 3.9, 3.8, fc, ec, lw=2.0, r=0.12)
        ax.text(x + 1.95, 3.4, name, ha="center", va="center", fontsize=12, fontweight="bold", color=tc)
        ax.text(x + 1.95, 2.7, term, ha="center", va="center", fontsize=12, color=tc)
        ax.plot([x + 0.3, x + 3.6], [2.25, 2.25], color=ec, lw=1)
        ax.text(x + 1.95, 1.85, "evidence", ha="center", va="center", fontsize=10, color=GRAY)
        ax.text(x + 1.95, 1.0, ev, ha="center", va="center", fontsize=11.5, color=tc, fontweight="bold")
    sp.save(fig, path)


c3(out / "c3.svg")


# 13. Fleet Law timeline: N devices, then 2N ------------------------------------
def fl_bar(ax, x0, x1, y, fc, ec, text, tc):
    rbox(ax, x0, y - 0.32, x1 - x0, 0.64, fc, ec, lw=1.6, r=0.05)
    ax.text((x0 + x1) / 2, y, text, ha="center", va="center", fontsize=13, color=tc, fontweight="bold")


def fleetlaw(path, k):
    fig, ax = plt.subplots(figsize=(5.9, 2.2))
    ax.set_xlim(-0.2, 13.0)
    ax.set_ylim(0.2, 6.0)
    ax.axis("off")
    ax.text(0, 5.6, "N devices", fontsize=13, fontweight="bold", va="center")
    ax.add_patch(Rectangle((4.6, 3.5), 1.4, 1.7, fc="#e2e8f0", ec="none"))
    fl_bar(ax, 0, 6.0, 4.8, C1L, C1, "T$_{compute}$/N", C1T)
    fl_bar(ax, 4.6, 8.6, 3.9, C2L, C2, "T$_{comm}$(N)", C2T)
    fl_bar(ax, 8.6, 10.0, 3.9, C3L, C3, "T$_{sync}$", C3T)
    ax.text(5.3, 5.25, "overlap", ha="center", va="bottom", fontsize=11, color=GRAY)
    ax.text(10.2, 3.9, "step = 10", fontsize=12, fontweight="bold", va="center")
    if k >= 2:
        ax.text(0, 2.75, "2N devices", fontsize=13, fontweight="bold", va="center")
        ax.add_patch(Rectangle((2.5, 0.65), 0.5, 1.7, fc="#e2e8f0", ec="none"))
        fl_bar(ax, 0, 3.0, 1.95, C1L, C1, "T$_{comp}$/2N", C1T)
        fl_bar(ax, 2.5, 6.6, 1.05, C2L, C2, "T$_{comm}$(2N)", C2T)
        fl_bar(ax, 6.6, 8.0, 1.05, C3L, C3, "T$_{sync}$", C3T)
        ax.plot([5.0, 5.0], [1.45, 2.45], color=RED, lw=1.6, ls="--")
        ax.text(5.15, 2.1, "ideal: 5", fontsize=12, color=RED, fontweight="bold", va="center")
        ax.text(8.2, 1.05, "step = 8", fontsize=12, color=RED, fontweight="bold", va="center")
    sp.save(fig, path)


fleetlaw(out / "fleetlaw_step1.svg", 1)
fleetlaw(out / "fleetlaw_step2.svg", 2)

# 14. Four efficiency numbers, four measures --------------------------------
fig, ax = plt.subplots(figsize=(5.8, 1.85))
meas = [("GPT-3 run", 20, "% of peak, run average", C["neutral"]),
        ("Megatron-LM", 52, "% of peak, 3,072 A100s", C["neutral"]),
        ("Llama 3 405B", 41, "% MFU, 16,384 H100s", C1),
        ("Llama 3 405B", 90, "%+ effective training time", C["data"])]
for i, (lab, v, m, col) in enumerate(meas):
    y = len(meas) - 1 - i
    ax.barh(y, v, color=col, height=0.6)
    ax.text(-1.5, y, lab, ha="right", va="center", fontsize=13, fontweight="bold")
    ax.text(v + 1.5, y, f"{v}{m}", va="center", fontsize=12.5, fontweight="bold", color=col)
ax.set_xlim(0, 155)
ax.set_ylim(-0.6, 3.6)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.spines["left"].set_visible(False)
ax.set_xlabel("percent")
sp.save(fig, out / "measures.svg")


# 15. Reliability gap: a^N -----------------------------------------------------
def anplot(path, k):
    fig, ax = plt.subplots(figsize=(3.9, 2.05))
    n = np.geomspace(1, 3e5, 400)
    ax.plot(n, 100 * 0.999 ** n, color=C1, lw=3)
    ax.text(4, 62, "a = 99.9%", color=C1, fontsize=11, fontweight="bold")
    ax.plot([1000], [36.8], "o", color=C1, ms=8)
    ax.text(1000 * 0.75, 30, "36.8% at 1,000", ha="right", va="top", fontsize=11,
            color=C1, fontweight="bold")
    ax.plot([10000], [0.0045], "o", color=RED, ms=8)
    ax.annotate("0.0045% at 10,000", (10000, 0.0045), xytext=(2, 10), textcoords="data",
                fontsize=11, color=RED, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    if k >= 2:
        ax.plot(n, 100 * 0.9999 ** n, color=C["compute"], lw=3)
        ax.plot([10000], [36.8], "o", color=C["compute"], ms=8)
        ax.text(1.4e4, 86, "a = 99.99%", fontsize=11, color=C["compute"], fontweight="bold")
        ax.text(1.4e4, 48, "36.8% at\n10,000", fontsize=11, color=C["compute"],
                fontweight="bold", va="center")
    ax.set_xscale("log")
    ax.set_xlim(1, 3e5)
    ax.set_ylim(-3, 105)
    ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5])
    ax.set_xlabel("components needed at once, N", fontsize=10)
    ax.set_ylabel("all up at once (%)", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.minorticks_off()
    sp.save(fig, path)


anplot(out / "an_step1.svg", 1)
anplot(out / "an_step2.svg", 2)

print("figures written to", out)
