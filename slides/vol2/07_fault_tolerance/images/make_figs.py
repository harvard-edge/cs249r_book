#!/usr/bin/env python3
"""Data plots and diagrams for Volume II, Lecture 7 (Fault Tolerance).

Every value is from the rendered chapter (Fault Tolerance) or MLSysIM unless
noted. MTBF values carry their basis, because the chapter uses three:
  - GPU-only: 50,000 h per GPU, independent, MTBF = 50,000 h / N
    (equation 4, table 1): 8 GPUs 6,250 h ... 10,000 GPUs 5 h, 25,000 GPUs 2 h.
  - Full system (worked cascade, Sec. 7.2.2): per GPU 2.0 (GPU) + 0.5 (PCIe)
    + 0.083 (one eighth of a node's NIC and cables) x 1e-5 per hour, power
    supplies redundant (0); 10,000 GPUs give 0.2583 per hour, MTBF 3.87 h.
  - Measured: Meta RSC (Kokolis et al. 2025) MTTF 1,144.8 h at 8 GPUs and
    7.9 h at 1,024 GPUs measured, 1.8 h at 16,384 and 0.23 h at 131,072
    projected; Llama 3 405B, 419 interruptions in 54 days on 16,384 H100s,
    about 3.1 h (Dubey et al. 2024). Read from MLSysIM's Literature registry.
  - Llama 3 root causes: MLSysIM Literature.Series.Llama3Interruptions
    (Dubey et al. 2024, Table 5 counts), grouped here: faulty GPU 148, HBM3 72,
    other GPU parts 42 (SRAM 19, system processor 17, thermal 6), software 56
    (bugs 54, dependency 2), network 35, host maintenance 32, host hardware 21,
    NCCL watchdog 7, silent data corruption 6; total 419.
  - Young-Daly (Napkin Math 7.2): write 24.5 s (2.45 TB at 100 GB/s), MTBF
    3.87 h = 13,932 s; tau* = 826 s = 13.8 min; waste 5.9 percent; hourly
    checkpoints 13.6 percent (equation 8).
  - Recovery budget (Napkin Math 7.3): 60 + 180 + 24.5 + 120 s = 6.4 min.
  - Straggler (Napkin Math 7.4): 1,024 GPUs, 1 s step, one GPU at half speed.
  - Elastic (figure 16): four-node ring to three, 75 percent throughput.
  - Fencing (figure 15): epochs 7 and 8.
  - Serving (Sec. 7.11.2): one replica 99 percent, 3.65 days a year; two
    52.6 min; three 31.5 s. KV cache 43 GB for a Llama-family 70B model at
    128K context in BF16 (footnote 19).
  - Degradation (Sec. 7.12.2): ViT-Large 307M, 88 percent; EfficientNet-B4
    19M, 83 percent; MobileNet-V3-Large 5.4M, 75 percent (ImageNet top-1).
  - Cascade (Example 7.1): p99 45 to 800 ms; feature-service span 12 to
    700 ms; model inference 40 ms.
  - DCP trace: measured on a laptop by dcp_trace.py in this folder (four CPU
    ranks, gloo, torch 2.11): shard files at 0.00, 0.18, 2.14, 3.28 ms and
    .metadata at 24.38 ms (file modification times relative to the first);
    each shard 12,587,643 bytes, .metadata 2,280 bytes.

Companion scripts in this folder: calcs.py prints every derived number on the
slides and notes; dcp_trace.py produced the DCP trace above (run with a torch
that has numpy); crash_trace.py produced the torchrun restart log on the
"launcher restarts the whole group" slide (torchrun --nproc-per-node=4
--max-restarts=2 crash_trace.py, torch 2.11, gloo, GLOO_SOCKET_IFNAME=lo0).

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent

# Volume II C3 architecture roles for schematics (volumes/vol2/palette.md)
V = {
    "c1": "#1F407A", "c1_l": "#EAF1F8",      # compute
    "c2": "#d97706", "c2_l": "#fef3c7",      # communication
    "c3": "#7c3aed", "c3_l": "#f3e8ff",      # coordination
    "ok": "#16a34a", "ok_l": "#dcfce7",      # storage, healthy flow
    "bad": "#dc2626", "bad_l": "#fee2e2",    # faults
    "frame": "#94a3b8", "frame_l": "#f8fafc", "text": "#0f172a",
}


def literature():
    """Measured fleet data from MLSysIM, so slide and book quote one source."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
    from mlsysim import Literature  # noqa: E402
    F = Literature.FleetOperations
    T = Literature.TrainingRuns
    rows = Literature.Series.Llama3Interruptions.entries
    kok = {
        8: float(F.Kokolis8GpuJobMttfHours),
        1024: float(F.Kokolis1024GpuJobMttfHours),
        16384: float(F.Kokolis16KGpuProjectedMttfHours),
        131072: float(F.Kokolis131KGpuProjectedMttfHours),
    }
    llama_h = float(T.Llama3_405B_SnapshotDays) * 24 / float(T.Llama3_405B_UnexpectedInterruptions)
    return kok, llama_h, rows


KOK, LLAMA_H, LLAMA_ROWS = literature()


def txt(ax, x, y, s, color=None, shown=True, **kw):
    kw.setdefault("fontsize", 12)
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, s, color=color or C["text"], alpha=1.0 if shown else 0.0, **kw)


def save_fixed(fig, path):
    """Diagrams keep their exact canvas so build steps never move."""
    fig.savefig(path, format="svg", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, fc, ec, lw=1.6, ls="-", alpha=1.0, r=0.06):
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha)
    ax.add_patch(p)
    return p


# 1. GPU-only cluster MTBF ladder (table 1) ----------------------------------
def _ladder_row(n):
    h = 50_000 / n          # GPU-only MTBF, equation 4
    s = f"{h:,.0f} h ({h / 24:.0f} days)" if h >= 24 else f"{h:g} h"
    return ("8 GPUs" if n == 8 else f"{n:,}", h, s)


LADDER = [_ladder_row(n) for n in (8, 64, 512, 1000, 4000, 10_000, 25_000)]


def ladder(path, k):
    fig, ax = plt.subplots(figsize=(5.6, 2.25))
    n = len(LADDER)
    for i, (lab, h, s) in enumerate(LADDER):
        y = n - 1 - i
        shown = i < k
        col = C["memory"] if h > 168 else C["danger"]
        ax.barh(y, h if shown else 0, color=col, height=0.62)
        txt(ax, 0.75, y, lab + "  ", C["text"], ha="right", va="center", fontsize=12)
        txt(ax, h * 1.2, y, s, col, shown, va="center", fontsize=12, zorder=4,
            bbox=dict(fc="white", ec="none", pad=0.5, alpha=1.0 if shown else 0.0))
    txt(ax, 2.5e5, 0.6, "red: a failure more\noften than once a week", C["danger"], fontsize=10.5,
        ha="right", va="center", linespacing=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3e5)
    ax.set_ylim(-0.6, n + 0.2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1 h", "10 h", "100 h", "1,000 h", "10,000 h"])
    ax.set_xlabel("Expected hours between GPU failures (GPU-only, 50,000 h each)", fontsize=11)
    ax.grid(axis="x", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for s, k in ((1, 2), (2, 4), (3, 7)):
    ladder(out / f"mtbf_ladder_step{s}.svg", k)


# 3. Measured MTTF against the GPU-only 1/N line ------------------------------
def measured(path, k):
    fig, ax = plt.subplots(figsize=(5.6, 2.35))
    ns = [4, 300000]
    ax.plot(ns, [50000 / n for n in ns], color=C["neutral"], lw=2, ls="--")
    txt(ax, 1500, 1500, "GPU-only line:\n50,000 h / N", C["neutral"], fontsize=11,
        ha="left", va="center", linespacing=1.0)
    # measured Meta RSC
    a = 1.0 if k >= 2 else 0.0
    for n in (8, 1024):
        ax.plot([n], [KOK[n]], "o", ms=10, color=C["memory"], alpha=a, zorder=5)
    txt(ax, 5, KOK[8] / 9, f"Meta RSC, measured:\n{KOK[8]:,.0f} h at 8 GPUs", C["memory"],
        k >= 2, fontsize=10.5, va="top", linespacing=1.0)
    txt(ax, 1024 / 1.6, KOK[1024] / 2.0, f"{KOK[1024]:g} h at 1,024", C["memory"], k >= 2,
        fontsize=10.5, va="top", ha="right")
    # model cascade, projections, Llama 3
    b = 1.0 if k >= 3 else 0.0
    ax.plot([10000], [3.87], "X", ms=10, color=C["compute"], alpha=b, zorder=5)
    txt(ax, 4500, 0.16, "chapter model, GPU+PCIe+NIC: 3.87 h",
        C["compute"], k >= 3, fontsize=10, ha="right", va="bottom")
    for n in (16384, 131072):
        ax.plot([n], [KOK[n]], "o", ms=10, mfc="white", mec=C["memory"], mew=2, alpha=b, zorder=5)
    ax.plot([16384], [LLAMA_H], "D", ms=10, color=C["danger"], alpha=b, zorder=6)
    txt(ax, 22000, 9, f"Llama 3 405B,\nmeasured: {LLAMA_H:.1f} h",
        C["danger"], k >= 3, fontsize=10.5, va="bottom", ha="left", linespacing=1.0)
    txt(ax, 16384, 0.8, f"projected:\n{KOK[16384]:g} h", C["memory"], k >= 3,
        fontsize=10.5, va="top", ha="center", linespacing=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4, 3e5)
    ax.set_ylim(0.1, 3e4)
    ax.set_xticks([10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000", "100,000"])
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["1 h", "10 h", "100 h", "1,000 h", "10,000 h"])
    ax.minorticks_off()
    ax.set_xlabel("GPUs in the job")
    ax.set_ylabel("Mean time to failure")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2, 3):
    measured(out / f"mttf_measured_step{k}.svg", k)


# 4. Llama 3 405B: root causes of 419 interruptions ---------------------------
def cnt(name):
    return next(r["interruption_count"] for r in LLAMA_ROWS if r["component"] == name)


cats = {}
for r in LLAMA_ROWS:
    cats[r["category"]] = cats.get(r["category"], 0) + r["interruption_count"]
other_gpu = cats["GPU"] - cnt("Faulty GPU") - cnt("GPU HBM3 Memory") - cnt("Silent Data Corruption")
CAUSES = [
    ("Faulty GPU", cnt("Faulty GPU"), "compute"),
    ("GPU HBM3 memory", cnt("GPU HBM3 Memory"), "compute"),
    ("Other GPU parts", other_gpu, "compute"),
    ("Software", cats["Dependency"], "neutral"),
    ("Network switch, cable", cats["Network"], "network"),
    ("Host maintenance", cats["Unplanned Maintenance"], "neutral"),
    ("Host hardware", cats["Host"], "neutral"),
    ("NCCL watchdog", cats["Unknown"], "neutral"),
    ("Silent data corruption", cnt("Silent Data Corruption"), "danger"),
]
assert sum(c[1] for c in CAUSES) == 419
fig, ax = plt.subplots(figsize=(4.4, 2.25))
n = len(CAUSES)
for i, (lab, v, ck) in enumerate(CAUSES):
    y = n - 1 - i
    ax.barh(y, v, color=C[ck], height=0.66)
    txt(ax, -3, y, lab, C["text"], ha="right", va="center", fontsize=11)
    txt(ax, v + 2, y, f"{v:.0f}", C[ck], va="center", fontsize=11)
ax.annotate("", xy=(176, n - 3.1), xytext=(176, n - 0.9),
            arrowprops=dict(arrowstyle="-", color=C["compute"], lw=2))
txt(ax, 186, n - 2.0, "GPU parts:\n268 of 419\n(64%)", C["compute"], va="center", fontsize=11,
    linespacing=1.0)
txt(ax, 22, 0, "detected only", C["danger"], va="center", fontsize=10, fontweight="normal")
ax.set_xlim(0, 235)
ax.set_ylim(-0.6, n - 0.4)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Unexpected interruptions, 54 days, 16,384 H100s")
ax.grid(axis="x", color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "llama3_causes.svg")


# 5. Fail-stop against Byzantine evidence (figure 3) --------------------------
def evidence(path, k):
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.75))
    for p, ax in enumerate(axes):
        shown = p == 0 or k >= 2
        a = 1.0 if shown else 0.0
        names = ["W1", "W2", "W3", "W4"] if p == 0 else ["W5", "W6", "W7", "W8"]
        for row, nm in enumerate(names):
            y = 3 - row
            txt(ax, -0.3, y, nm, C["text"], shown, ha="right", va="center", fontsize=11)
            for t in range(10):
                if p == 0:
                    if row == 1 and t >= 4:
                        continue
                    ax.plot([t, t], [y - 0.22, y + 0.22], color=V["c3"], lw=2.2, alpha=a)
                else:
                    bad = row == 1 and t >= 4
                    ax.add_patch(plt.Rectangle((t - 0.22, y - 0.22), 0.44, 0.44, alpha=a,
                                               fc=V["bad"] if bad else V["c2"], ec="none"))
            ok = not (p == 0 and row == 1)
            ax.plot([11.2], [y], marker="o" if ok else "X", ms=10, alpha=a,
                    color=V["ok"] if ok else V["bad"])
        txt(ax, 11.2, 4.0, "alive?", C["text"], shown, ha="center", fontsize=10.5)
        if p == 0:
            ax.plot([7.0], [2], marker="D", ms=8, color=V["c3"], alpha=a)
            txt(ax, 7.4, 2.0, "timeout", V["c3"], shown, ha="left", va="center", fontsize=10.5)
            txt(ax, 5, -1.0, "Fail-stop: W2 goes silent", C["text"], shown, ha="center", fontsize=11.5)
        else:
            txt(ax, 5, -1.0, "Byzantine: W6 sends wrong values", C["text"], shown,
                ha="center", fontsize=11.5)
        ax.set_xlim(-1.3, 12)
        ax.set_ylim(-1.3, 4.4)
        ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.02, wspace=0.12)
    save_fixed(fig, path)


for k in (1, 2):
    evidence(out / f"evidence_step{k}.svg", k)


# 6. Hot spare: N+1 pool, three stages (figure 10) ---------------------------
def hotspare(path, k):
    fig, ax = plt.subplots(figsize=(5.6, 1.55))
    titles = ["1. Replay implicates G2", "2. Drain G2, promote S", "3. Refill the spare"]
    for st in range(3):
        shown = st < k
        a = 1.0 if shown else 0.0
        x0 = st * 3.9
        slots = [("G0", "c1"), ("G1", "c1"), ("G2", "bad" if st == 0 else "gone"),
                 ("G3", "c1"), ("S", "spare" if st == 0 else "c1")]
        if st == 2:
            slots[2] = ("S'", "spare")
        for j, (nm, kind) in enumerate(slots):
            x = x0 + j * 0.7
            if kind == "c1":
                fc, ec, ls = V["c1_l"], V["c1"], "-"
            elif kind == "bad":
                fc, ec, ls = V["bad_l"], V["bad"], "-"
            elif kind == "gone":
                fc, ec, ls = "white", V["frame"], "--"
            else:
                fc, ec, ls = V["ok_l"], V["ok"], "--"
            box(ax, x, 0, 0.58, 0.8, fc, ec, lw=1.8, ls=ls, alpha=a)
            txt(ax, x + 0.29, 0.4, nm, ec if kind != "gone" else V["frame"], shown,
                ha="center", va="center", fontsize=11.5)
        txt(ax, x0 + 1.75, 1.05, titles[st], C["text"], shown, ha="center", va="bottom", fontsize=11)
        if st == 1:
            txt(ax, x0 + 1.75, -0.3, "G2 to diagnostics", V["bad"], shown, ha="center", va="top",
                fontsize=10.5, fontweight="normal")
    ax.set_xlim(-0.1, 11.6)
    ax.set_ylim(-0.75, 1.5)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_fixed(fig, path)


for k in (1, 2, 3):
    hotspare(out / f"hotspare_step{k}.svg", k)


# 7. Young-Daly waste curves (Napkin Math 7.2, figure 11) ---------------------
DELTA, M = 24.5, 3.87 * 3600
TAU = math.sqrt(2 * DELTA * M)
W_OPT = DELTA / TAU + TAU / (2 * M)
W_HOUR = DELTA / 3600 + 3600 / (2 * M)
assert abs(TAU - 826.2) < 1 and abs(W_OPT - 0.0593) < 0.001 and abs(W_HOUR - 0.136) < 0.001


def young_daly(path, k):
    import numpy as np
    fig, ax = plt.subplots(figsize=(4.4, 2.0))
    t = np.linspace(1.5, 90, 400) * 60
    write = DELTA / t * 100
    rework = t / (2 * M) * 100
    ax.plot(t / 60, write, color=C["memory"], lw=2.5, ls="--")
    txt(ax, 5.0, 17, "write: $\\delta/\\tau$", C["memory"], fontsize=12)
    a2 = 1.0 if k >= 2 else 0.0
    ax.plot(t / 60, rework, color=C["danger"], lw=2.5, ls="--", alpha=a2)
    txt(ax, 70, 9.6, "rework: $\\tau/2M$", C["danger"], k >= 2, fontsize=12, ha="center")
    a3 = 1.0 if k >= 3 else 0.0
    ax.plot(t / 60, write + rework, color=C["text"], lw=3, alpha=a3)
    ax.plot([TAU / 60], [W_OPT * 100], "o", ms=11, color=C["data"], alpha=a3, zorder=5)
    txt(ax, 24, 3.4, f"optimum {TAU / 60:.1f} min: {W_OPT * 100:.1f}% wasted", C["data"], k >= 3,
        fontsize=12, va="center")
    ax.plot([60], [W_HOUR * 100], "s", ms=10, color=C["compute"], alpha=a3, zorder=5)
    txt(ax, 58, 17.6, f"every hour: {W_HOUR * 100:.1f}%", C["compute"], k >= 3,
        fontsize=12, ha="center")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 22)
    ax.set_xlabel("Checkpoint interval $\\tau$ (minutes)")
    ax.set_ylabel("Share of time wasted (%)")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2, 3):
    young_daly(out / f"young_daly_step{k}.svg", k)


# 8. Consistent cut: uncoordinated against coordinated saves ------------------
def cut(path, k):
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.65))
    for p, ax in enumerate(axes):
        shown = p == 0 or k >= 2
        a = 1.0 if shown else 0.0
        for w in range(3):
            y = 2 - w
            ax.plot([0, 10], [y, y], color=V["frame"], lw=2, alpha=a)
            txt(ax, -0.3, y, f"W{w + 1}", C["text"], shown, ha="right", va="center", fontsize=11)
        if p == 0:
            saves = [(2.0, 2), (5.0, 1), (7.8, 0)]
            for x, y in saves:
                ax.add_patch(plt.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, fc=V["c1"], alpha=a))
            ax.annotate("", xy=(4.2, 1), xytext=(3.2, 2), alpha=a,
                        arrowprops=dict(arrowstyle="-|>", color=V["c2"], lw=1.8, alpha=a))
            ax.annotate("", xy=(7.0, 0), xytext=(6.0, 1),
                        arrowprops=dict(arrowstyle="-|>", color=V["c2"], lw=1.8, alpha=a))
            ax.plot([9.0], [2], "X", ms=12, color=V["bad"], alpha=a)
            ax.annotate("", xy=(2.0, 2.45), xytext=(8.8, 2.45),
                        arrowprops=dict(arrowstyle="-|>", color=V["bad"], lw=1.8, alpha=a))
            txt(ax, 5.4, 2.55, "rollback", V["bad"], shown, ha="center", va="bottom", fontsize=10.5)
            txt(ax, 5, -1.0, "Saves at different steps", C["text"], shown, ha="center", fontsize=11.5)
        else:
            ax.plot([5, 5], [-0.45, 2.45], color=V["ok"], lw=3, alpha=a)
            for y in range(3):
                ax.add_patch(plt.Rectangle((4.75, y - 0.25), 0.5, 0.5, fc=V["c1"], alpha=a))
            txt(ax, 5.3, 2.55, "all shards at step 1,000", V["ok"], shown, ha="center",
                va="bottom", fontsize=10.5)
            box(ax, 5.9, 0.27, 4.2, 0.46, V["c3_l"], V["c3"], alpha=a)
            txt(ax, 8.0, 0.5, "manifest last", V["c3"], shown, ha="center", va="center",
                fontsize=10.5)
            txt(ax, 5, -1.0, "One coordinated cut", C["text"], shown, ha="center", fontsize=11.5)
        ax.set_xlim(-1.2, 10.3)
        ax.set_ylim(-1.3, 3.1)
        ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.02, wspace=0.1)
    save_fixed(fig, path)


for k in (1, 2):
    cut(out / f"cut_step{k}.svg", k)


# 9. Measured DCP save: when each file landed (dcp_trace.py) ------------------
DCP = [("rank 0", 0.00), ("rank 1", 0.18), ("rank 2", 2.14),
       ("rank 3", 3.28), (".metadata", 24.38)]
fig, ax = plt.subplots(figsize=(2.8, 2.1))
for i, (lab, t) in enumerate(DCP):
    y = len(DCP) - 1 - i
    meta = lab == ".metadata"
    col = V["c3"] if meta else V["c1"]
    ax.plot([t], [y], "o", ms=9, color=col)
    txt(ax, -1.5, y, lab, col, ha="right", va="center", fontsize=12)
    txt(ax, t + (-1.5 if meta else 1.5), y, f"{t:.2f} ms", col,
        ha="right" if meta else "left", va="center", fontsize=11.5, fontweight="normal")
ax.set_xlim(-1, 27)
ax.set_ylim(-0.6, 4.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([0, 10, 20])
ax.set_xlabel("ms after the first shard", fontsize=11.5)
ax.tick_params(labelsize=11)
ax.grid(axis="x", color=C["grid"], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "dcp_trace.svg")


# 10. Recovery-time budget (Napkin Math 7.3) ----------------------------------
def budget(path, k):
    fig, ax = plt.subplots(figsize=(5.6, 1.25))
    parts = [("detect", 60, "c3"), ("restart", 180, "c1"), ("load", 24.5, "ok"),
             ("warm up", 120, "c2")]
    x = 0
    for lab, w, ck in parts:
        ax.barh(0, w, left=x, color=V[ck], height=0.6, edgecolor="white", lw=1.5)
        hi = lab == "load"
        txt(ax, x + w / 2, 0.8 if hi else 0.42, lab, V[ck], ha="center", va="bottom", fontsize=11.5)
        txt(ax, x + w / 2, -0.95 if hi else -0.55, f"{w:g} s", V[ck], ha="center", va="center",
            fontsize=11)
        x += w
    txt(ax, x + 8, 0, f"= {x / 60:.1f} min", C["text"], va="center", fontsize=13)
    a = 1.0 if k >= 2 else 0.0
    ax.barh(0, 200, left=x + 160, height=0.6, fc=C["danger_l"], ec=C["danger"], ls="--",
            lw=1.5, alpha=a)
    txt(ax, x + 260, 0.05, "+ rework", C["danger"], k >= 2, ha="center", va="center", fontsize=12)
    txt(ax, x + 260, -0.55, "half an interval", C["danger"], k >= 2, ha="center", va="center",
        fontsize=10.5, fontweight="normal")
    ax.set_xlim(0, 760)
    ax.set_ylim(-1.2, 1.25)
    ax.axis("off")
    sp.save(fig, path)


for k in (1, 2):
    budget(out / f"budget_step{k}.svg", k)


# 11. Epoch fencing (figure 15) ------------------------------------------------
def fencing(path, k):
    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    lanes = ["coordinator", "old worker", "replacement", "checkpoint store"]
    for i, nm in enumerate(lanes):
        y = 3 - i
        ax.plot([0, 10], [y, y], color="#e2e8f0", lw=1)
        txt(ax, -0.2, y, nm, C["text"], ha="right", va="center", fontsize=11)
    # step 1: epoch 7, old worker writes with token 7
    box(ax, 0.2, 2.8, 3.6, 0.4, V["c3_l"], V["c3"])
    txt(ax, 2.0, 3.0, "epoch 7", V["c3"], ha="center", va="center", fontsize=11)
    box(ax, 0.2, 1.8, 2.6, 0.4, V["c1_l"], V["c1"])
    txt(ax, 1.5, 2.0, "token 7", V["c1"], ha="center", va="center", fontsize=11)
    box(ax, 0.2, -0.2, 3.6, 0.4, V["ok_l"], V["ok"])
    txt(ax, 2.0, 0.0, "accepts $\\geq 7$", V["ok"], ha="center", va="center", fontsize=11)
    ax.plot([2.9, 6.6], [2.0, 2.0], color=V["frame"], lw=1.5, ls=":")
    txt(ax, 5.3, 2.08, "paused", V["frame"], ha="center", va="bottom", fontsize=10.5,
        fontweight="normal")
    # step 2: lease expires, epoch 8, fence advanced, replacement writes
    s2 = k >= 2
    a2 = 1.0 if s2 else 0.0
    ax.plot([3.95, 3.95], [-0.35, 3.35], color=V["bad"], lw=1.5, ls="--", alpha=a2)
    txt(ax, 3.95, 3.42, "lease expires", V["bad"], s2, ha="center", va="bottom", fontsize=10.5)
    box(ax, 4.1, 2.8, 5.7, 0.4, V["c3_l"], V["c3"], alpha=a2)
    txt(ax, 6.95, 3.0, "epoch 8", V["c3"], s2, ha="center", va="center", fontsize=11)
    box(ax, 4.1, -0.2, 5.7, 0.4, V["ok_l"], V["ok"], alpha=a2)
    txt(ax, 6.95, 0.0, "accepts $\\geq 8$", V["ok"], s2, ha="center", va="center", fontsize=11)
    box(ax, 4.6, 0.8, 2.2, 0.4, V["c1_l"], V["c1"], alpha=a2)
    txt(ax, 5.7, 1.0, "token 8", V["c1"], s2, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(5.7, 0.22), xytext=(5.7, 0.78),
                arrowprops=dict(arrowstyle="-|>", color=V["ok"], lw=2, alpha=a2))
    # step 3: old worker resumes with token 7, rejected
    s3 = k >= 3
    a3 = 1.0 if s3 else 0.0
    box(ax, 6.7, 1.8, 2.0, 0.4, V["c1_l"], V["c1"], alpha=a3)
    txt(ax, 7.7, 2.0, "token 7", V["c1"], s3, ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(8.3, 0.25), xytext=(8.3, 1.78),
                arrowprops=dict(arrowstyle="-|>", color=V["bad"], lw=2, alpha=a3))
    ax.plot([8.3], [0.55], "X", ms=13, color=V["bad"], alpha=a3)
    txt(ax, 8.75, 0.62, "rejected", V["bad"], s3, va="center", fontsize=11)
    ax.set_xlim(-3.2, 10.1)
    ax.set_ylim(-0.45, 3.75)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_fixed(fig, path)


for k in (1, 2, 3):
    fencing(out / f"fencing_step{k}.svg", k)


# 12. Straggler at the barrier (Napkin Math 7.4) -------------------------------
def straggler(path, k):
    fig, ax = plt.subplots(figsize=(5.4, 1.6))
    ranks = ["rank 0", "rank 1", "rank 2", "rank 517", "rank 1,023"]
    for i, nm in enumerate(ranks):
        y = len(ranks) - 1 - i
        slow = k >= 2 and nm == "rank 517"
        ax.barh(y, 2.0 if slow else 1.0, color=C["danger"] if slow else C["memory"], height=0.6)
        if k >= 2 and not slow:
            ax.barh(y, 1.0, left=1.0, color=C["neutral_l"], height=0.6, hatch="//",
                    edgecolor=C["neutral"], lw=0)
        txt(ax, -0.04, y, nm, C["text"], ha="right", va="center", fontsize=13)
    bx = 2.0 if k >= 2 else 1.0
    ax.axvline(bx, color=V["c3"], lw=2.5)
    txt(ax, bx + 0.04, len(ranks) - 0.5, "AllReduce barrier", V["c3"], va="center", fontsize=13)
    txt(ax, 1.5, 4.65, "1,023 GPUs wait", C["neutral"], k >= 2, ha="center", va="bottom",
        fontsize=13)
    txt(ax, 1.0, 1.0, "half speed: 2 s", "white", k >= 2, va="center", ha="center", fontsize=13)
    ax.set_xlim(0, 2.75)
    ax.set_ylim(-0.6, 5.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel("Seconds into the step")
    sp.save(fig, path)


for k in (1, 2):
    straggler(out / f"straggler_step{k}.svg", k)


# 13. Elastic recovery, four nodes to three (figure 16) -----------------------
def elastic(path, k):
    fig, ax = plt.subplots(figsize=(5.6, 1.5))
    titles = ["Ring of 4", "N3 fails: rendezvous,\nreload checkpoint", "Ring of 3,\n+1 shard each"]
    thr = [100, 0, 75]
    for st in range(3):
        shown = st < k
        a = 1.0 if shown else 0.0
        cx = st * 4.0 + 1.3
        alive = [0, 1, 2, 3] if st == 0 else [0, 1, 2]
        pts = {i: (cx + 0.85 * math.cos(math.pi / 2 - i * math.pi / 2),
                   1.1 + 0.85 * math.sin(math.pi / 2 - i * math.pi / 2)) for i in range(4)}
        if st == 2:
            pts = {i: (cx + 0.8 * math.cos(math.pi / 2 - i * 2 * math.pi / 3),
                       1.1 + 0.8 * math.sin(math.pi / 2 - i * 2 * math.pi / 3)) for i in range(3)}
        ring = alive if st == 1 else alive + [alive[0]]
        for i, j in zip(ring[:-1], ring[1:]):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=V["c2"], lw=2, alpha=a)
        if st == 1:
            for j in (2, 0):
                ax.plot([pts[3][0], pts[j][0]], [pts[3][1], pts[j][1]], color=V["bad"], lw=2,
                        ls="--", alpha=a)
        for i, (x, y) in pts.items():
            dead = st == 1 and i == 3
            ax.add_patch(plt.Circle((x, y), 0.34, zorder=3, fc=V["bad_l"] if dead else V["c1_l"],
                                    ec=V["bad"] if dead else V["c1"], lw=1.8, alpha=a))
            txt(ax, x, y, f"N{i}", V["bad"] if dead else V["c1"], shown, ha="center",
                va="center", fontsize=10.5, zorder=4)
        # throughput bar
        bx = cx + 1.55
        ax.add_patch(plt.Rectangle((bx, 0.25), 0.35, 1.7, fc="white", ec=V["frame"], alpha=a))
        ax.add_patch(plt.Rectangle((bx, 0.25), 0.35, 1.7 * thr[st] / 100, fc=V["ok"], alpha=a))
        txt(ax, bx + 0.17, 2.05, f"{thr[st]}%" if thr[st] else "paused", V["ok"] if thr[st] else
            V["bad"], shown, ha="center", va="bottom", fontsize=11)
        txt(ax, cx + 0.3, -0.25, titles[st], C["text"], shown, ha="center", va="top",
            fontsize=10.5, linespacing=1.0)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(-1.15, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save_fixed(fig, path)


for k in (1, 2, 3):
    elastic(out / f"elastic_step{k}.svg", k)


# 14. Serving: replicas and state (Sec. 7.11.2, footnote 19) -------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 1.8), gridspec_kw={"width_ratios": [1.15, 1]})
_down = [(1 - 0.99) ** r * 365 * 86400 for r in (1, 2, 3)]   # equation 12, 99% each
rows = [("1 replica", _down[0], f"{_down[0] / 86400:.2f} days"),
        ("2 replicas", _down[1], f"{_down[1] / 60:.1f} min"),
        ("3 replicas", _down[2], f"{_down[2]:.1f} s")]
for i, (lab, v, s) in enumerate(rows):
    y = 2 - i
    a1.barh(y, v, color=C["memory"], height=0.6)
    txt(a1, -0.03, y, lab, C["text"], ha="right", va="center", fontsize=11,
        transform=a1.get_yaxis_transform())
    txt(a1, v * 1.4, y, s, C["memory"], va="center", fontsize=11)
a1.set_xscale("log")
a1.set_xlim(3, 3e8)
a1.set_ylim(-0.6, 2.6)
a1.set_yticks([])
a1.set_xticks([10, 1e3, 1e5, 1e7])
a1.set_xticklabels(["10 s", "17 min", "28 h", "116 d"], fontsize=10)
a1.minorticks_off()
a1.spines["left"].set_visible(False)
a1.set_xlabel("Downtime per year, 99% each", fontsize=11)
for i, (lab, v, ck) in enumerate([("image retry", 0, "data"), ("LLM chat", 43, "danger")]):
    y = 1 - i
    a2.barh(y, max(v, 0.4), color=C[ck], height=0.55)
    txt(a2, -0.03, y, lab, C["text"], ha="right", va="center", fontsize=11,
        transform=a2.get_yaxis_transform())
txt(a2, 1.5, 1, "0 GB lost", C["data"], va="center", fontsize=11)
txt(a2, 42, 0, "43 GB", "white", va="center", ha="right", fontsize=12)
a2.set_xlim(0, 46)
a2.set_ylim(-0.6, 1.6)
a2.set_yticks([])
a2.spines["left"].set_visible(False)
a2.set_xlabel("State to rebuild on failover", fontsize=11)
fig.subplots_adjust(wspace=0.75)
sp.save(fig, out / "serving_state.svg")


# 15. Degradation ladder (Sec. 7.12.2) -----------------------------------------
def degrade(path, k):
    fig, ax = plt.subplots(figsize=(3.6, 1.95))
    pts = [("ViT-Large", 307, 88), ("EfficientNet-B4", 19, 83), ("MobileNet-V3-Large", 5.4, 75)]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    for i, (nm, x, y) in enumerate(pts):
        shown = i < k
        a = 1.0 if shown else 0.0
        if i > 0:
            ax.annotate("", xy=(x * 1.15, y + 0.3), xytext=(xs[i - 1] / 1.15, ys[i - 1] - 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=C["compute"], lw=2, alpha=a))
        ax.plot([x], [y], "o", ms=12, color=C["memory"], alpha=a, zorder=5)
        lx, ly, va = [(420, 88.0, "center"), (19 * 1.4, 82.4, "top"), (5.4 * 1.4, 74.6, "top")][i]
        txt(ax, lx, ly, f"{nm}\n{x:g}M, {y}%", C["memory"], shown, fontsize=11, va=va,
            ha="left", linespacing=1.0)
    txt(ax, 2.3, 92.6, "last rung: cached labels,\nor \"unavailable\"",
        C["danger"], k >= 3, fontsize=10.5, va="top", ha="left", linespacing=1.0)
    ax.set_xscale("log")
    ax.set_xlim(2, 8000)
    ax.set_ylim(70, 93)
    ax.set_xticks([3, 10, 30, 100, 300, 1000])
    ax.set_xticklabels(["3M", "10M", "30M", "100M", "300M", "1B"], fontsize=10)
    ax.minorticks_off()
    ax.set_xlabel("Parameters (fallback order: right to left)")
    ax.set_ylabel("ImageNet top-1 (%)")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, path)


for k in (1, 2, 3):
    degrade(out / f"degrade_step{k}.svg", k)


# 16. Cascade: span times before and during the incident (Example 7.1) --------
def cascade(path, k):
    fig, ax = plt.subplots(figsize=(3.6, 1.45))
    spans = [("feature service", 12, 700), ("model inference", 40, 40)]
    for i, (nm, base, inc) in enumerate(spans):
        y = 1 - i
        ax.barh(y + 0.21, base, color=C["memory"], height=0.36)
        txt(ax, base + 8, y + 0.21, f"{base} ms", C["memory"], va="center", fontsize=12)
        a = 1.0 if k >= 2 else 0.0
        ax.barh(y - 0.21, inc, color=C["danger"], height=0.36, alpha=a)
        txt(ax, inc + 8, y - 0.21, f"{inc} ms", C["danger"], k >= 2, va="center", fontsize=12)
        txt(ax, -8, y, nm, C["text"], ha="right", va="center", fontsize=12)
    txt(ax, 790, 2.25, "blue: normal", C["memory"], ha="right", fontsize=11, va="bottom")
    txt(ax, 790, 1.85, "red: 14:32 incident", C["danger"], k >= 2, ha="right", fontsize=11,
        va="bottom")
    ax.set_xlim(0, 800)
    ax.set_ylim(-0.5, 2.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Span time in the request trace (ms)")
    sp.save(fig, path)


for k in (1, 2):
    cascade(out / f"cascade_step{k}.svg", k)

print("figures written to", out)
