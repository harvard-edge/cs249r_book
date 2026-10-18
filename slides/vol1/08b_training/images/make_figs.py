#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 8b (Model Training, Part 2).

Every value is from the rendered chapter (Model Training) unless noted:
  - Scoreboard: 598, 311, 109.6, 66.8 GB; static 24 GB; H100 85.9 GB
    (80 GiB) (Napkin Math 8.6, 8.10). Checkpointing alternative 35.5 GB,
    about one-third more arithmetic, 47 days instead of 35 (Napkin Math 8.8).
  - Static state: 2 + 2 + 4 + 8 = 16 bytes per parameter (Sec. 8.3.2).
  - Peak dense throughput by format, V100 to B200 (figure 10, NVIDIA
    datasheets cited in Sec. 8.6.3); redrawn here at slide size.
  - Formats: exponent bits 8, 5, 8; mantissa 23, 10, 7; FP16 smallest normal
    6.1e-5, largest 65,504, subnormals to about 6e-8; FP32 and BF16 smallest
    normal about 1e-38 (Sec. 8.6.3, table 9). Range ends of FP32 and BF16
    (about 3.4e38) are from the IEEE 754 and bfloat16 formats.
  - Swamping: derived from the formats and checked with numpy float16:
    FP16 spacing near 0.5 is 2^-11 = 4.9e-4; 0.5 + 0.0001 rounds to 0.5;
    10,000 such updates leave FP16 at 0.5 and take FP32 to 1.5.
  - Loss-scaling histogram: illustrative shape (labeled), after Micikevicius
    et al. 2018, Figure 3; FP16 normal floor 2^-14, subnormal floor 2^-24.
  - Run time by format: 518, 70, 35 days (Napkin Math 8.7).
  - Per-layer activations at batch 32, 1,024 tokens: projections 419.4 MB,
    feed-forward 419.4 MB, scores 4,194.3 MB, LayerNorm about 10 MB
    (Napkin Math 8.3). The scores are three saved copies at 2 + 1 + 2 bytes
    per entry (softmax output, dropout mask, output after dropout; Napkin Math 8.3);
    the split of the bar follows those byte counts.
  - Tiling: 4,096^2 x 4 B = 67.1 MB; 128^2 x 4 B = 65.5 KB (Sec. 8.6.4);
    228 KB of shared memory per SM (Hardware Acceleration).
  - Stored per head at 4,096 tokens: 138.4 MB against 4.2 MB (Sec. 8.6.4).
  - Batch line: 24 GB + 2.675 GB per sequence (85.6 GB / 32, derived);
    batch 16 gives 66.8 GB, batch 32 109.6 GB (Napkin Math 8.10).
  - Checkpointing: 8 segments of 6 layers, 0.84 GB of inputs plus 10.7 GB
    (Napkin Math 8.8). Checkpoint count curve k + 48/k in layer units is
    derived from table 13.

Run from this folder: python3 make_figs.py
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent
KIT = Path(__file__).resolve().parents[3] / "assets" / "kit" / "vol1"
CAP = 85.9  # H100 HBM, 80 GiB in decimal GB, as the chapter compares


def label(ax, x, y, text, color, shown=True, **kw):
    """A bold label drawn in every step (invisible until shown), so every
    step has the same bounding box and nothing moves on the slide."""
    kw.setdefault("fontsize", 12)
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, text, color=color, alpha=1.0 if shown else 0.0, **kw)


# 1. Running scoreboard (copied from Part 1's generator) --------------------
names = ["FP32", "Mixed\nprecision", "Flash-\nAttention", "2 micro-\nbatches"]
totals = [598, 311, 109.6, 66.8]


def scoreboard(path, k, costs=None, alt=False):
    """alt adds option B of the vote, segment checkpointing at 35.5 GB, as a
    hollow bar beside the chosen option (Napkin Math 8.8)."""
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    for i, t in enumerate(totals):
        shown = i < k
        a = 1.0 if shown else 0.0
        xpos = i - 0.17 if (alt and i == 3) else i
        width = 0.32 if (alt and i == 3) else 0.58
        ax.bar(xpos, 24, width, color=C["neutral"], alpha=a)
        ax.bar(xpos, t - 24, width, bottom=24, color=C["data"], alpha=a)
        label(ax, xpos, max(t, CAP) + 18, f"{t:g} GB", C["data"] if t < CAP else C["danger"],
              shown, ha="center", fontsize=13 if not (alt and i == 3) else 11)
        if costs:
            label(ax, i, 395, costs[i], C["compute"], i > 0,
                  ha="center", va="bottom", fontsize=10.5, linespacing=1.1)
    if alt:
        ax.bar(3.19, 35.5, 0.32, color="white", edgecolor=C["compute"], lw=1.8, ls="--")
        ax.text(3.19, 150, "B:\n35.5", color=C["compute"], fontsize=13,
                fontweight="bold", ha="center", va="bottom", linespacing=1.0)
    ax.axhline(CAP, color=C["memory"], lw=2, ls="--", zorder=0)
    ax.text(3.55, CAP, "H100\n85.9 GB", color=C["memory"], fontsize=11,
            fontweight="bold", ha="left", va="center")
    if costs:
        ax.text(2.0, 520, "What each step cost", color=C["compute"], fontsize=12,
                fontweight="bold", ha="center")
    else:
        if alt:
            ax.text(3.45, 440, "dashed: B, checkpoint", color=C["compute"], fontsize=11,
                    fontweight="bold", ha="right")
        ax.text(3.45, 560, "gray: static 24 GB", color=C["neutral"], fontsize=11,
                fontweight="bold", ha="right")
        ax.text(3.45, 500, "green: activations", color=C["data"], fontsize=11,
                fontweight="bold", ha="right")
    ax.set_xticks(range(4), names, fontsize=11)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 660)
    ax.set_ylabel("GB per step")
    sp.save(fig, path)


for k in (1, 2):
    scoreboard(out / f"scoreboard_step{k}.svg", k)
scoreboard(out / "scoreboard_step4.svg", 4, alt=True)
scoreboard(out / "scoreboard_costs.svg", 4,
           costs=["", "FP32 master,\nloss scaling", "recomputed\ntiles",
                  "update every\ntwo passes"])

# 2. Bytes per parameter (copied from Part 1's generator) ---------------------
fig, ax = plt.subplots(figsize=(5.8, 1.5))
x = 0
for w, text, ck in [(2, "W", "memory"), (2, "G", "network"), (4, "master", "neutral"),
                    (8, "Adam m, v (FP32)", "compute")]:
    ax.barh(0, w, left=x, color=C[ck], height=0.6, edgecolor="white")
    ax.text(x + w / 2, 0, text, ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")
    x += w
ax.text(x + 0.3, 0, "16 B per parameter", va="center", fontsize=13, fontweight="bold",
        color=C["danger"])
ax.set_xlim(0, 24)
ax.set_ylim(-0.5, 0.5)
ax.axis("off")
sp.save(fig, out / "static_bytes.svg")

# 3. Peak dense throughput by format, four generations (book figure 10) -------
fmts = [("FP32", "#333333"), ("TF32", C["memory"]), ("FP16/BF16", C["data"]),
        ("FP8", C["compute"]), ("FP4", C["danger"])]
gens = [("V100\n(2017)", [15.7, None, 125, None, None]),
        ("A100\n(2020)", [19.5, 156, 312, None, None]),
        ("H100\n(2022)", [67, 494, 989, 1979, None]),
        ("B200\n(2024)", [75, 1125, 2250, 4500, 9000])]
show_label = {(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2), (3, 0), (3, 2), (3, 4)}
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    w = 0.17
    for g, (gname, vals) in enumerate(gens):
        shown = g == 0 or k >= 2
        dim = k == 3 and g != 2
        for j, v in enumerate(vals):
            if v is None:
                continue
            xpos = g + (j - 2) * w
            a = (0.35 if dim else 1.0) if shown else 0.0
            ax.bar(xpos, v, w * 0.92, color=fmts[j][1], alpha=a)
            if (g, j) in show_label:
                txt = f"{v:,.4g}" if v < 100 else f"{v:,.0f}"
                label(ax, xpos, v * 1.18, txt, fmts[j][1], shown and not dim,
                      ha="center", va="bottom", fontsize=10)
    for (fname, col), xk in zip(fmts, [-0.45, 0.15, 0.8, 1.9, 2.45]):
        ax.text(xk, 3.2e4, fname, color=col, fontsize=11, fontweight="bold", va="bottom")
    label(ax, 2.0, 5000, "67 against 989: 15×", C["danger"], k == 3, ha="center",
          fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(5, 6e4)
    ax.set_xlim(-0.55, 3.55)
    ax.set_xticks(range(4), [g[0] for g in gens], fontsize=11)
    ax.set_yticks([10, 100, 1000, 10000], ["10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    ax.set_ylabel("Peak dense TFLOP/s")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"precision_ladder_step{k}.svg")

# 4. Number formats: bits to scale, and the range each one covers ------------
rows = [("FP32", 8, 23, "about 7 digits", (1.2e-38, 3.4e38), None),
        ("FP16", 5, 10, "about 3 digits", (6.1e-5, 65504), (6e-8, 6.1e-5)),
        ("BF16", 8, 7, "2 to 3 digits", (1.2e-38, 3.4e38), None)]
for k in range(1, 5):
    fig = plt.figure(figsize=(5.8, 2.75))
    axb = fig.add_axes([0.13, 0.12, 0.36, 0.72])
    axr = fig.add_axes([0.55, 0.12, 0.43, 0.72])
    for i, (name, e, m, digits, (lo, hi), sub) in enumerate(rows):
        y = 2 - i
        shown = i < k
        a = 1.0 if shown else 0.0
        axb.barh(y, 1, left=0, color=C["neutral_l"], height=0.55, alpha=a, edgecolor="white")
        axb.barh(y, e, left=1, color=C["memory"], height=0.55, alpha=a, edgecolor="white")
        axb.barh(y, m, left=1 + e, color=C["compute"], height=0.55, alpha=a, edgecolor="white")
        label(axb, 1 + e / 2, y, str(e), "white", shown, ha="center", va="center", fontsize=11)
        label(axb, 1 + e + m / 2, y, str(m), "white", shown, ha="center", va="center",
              fontsize=11)
        label(axb, -0.6, y + 0.1, name, C["text"], shown, ha="right", va="center", fontsize=13)
        label(axb, -0.6, y - 0.27, digits, C["neutral"], shown, ha="right", va="center",
              fontsize=9.5, fontweight="normal")
        axr.barh(y, np.log10(hi) - np.log10(lo), left=np.log10(lo), color=C["data"],
                 height=0.45, alpha=a)
        if sub:
            axr.barh(y, np.log10(sub[1]) - np.log10(sub[0]), left=np.log10(sub[0]),
                     color=C["data_l"], height=0.45, alpha=a)
    label(axr, 3, 1.38, "6.1×10⁻⁵ to 65,504", C["data"], k >= 2, ha="left",
          va="bottom", fontsize=11)
    label(axr, 3, 2.38, "about 10⁻³⁸ to 10³⁸", C["data"], k >= 1, ha="left",
          va="bottom", fontsize=11)
    show_grad = k == 4
    axr.axvline(-6, color=C["danger"], lw=2, ls="--", alpha=1.0 if show_grad else 0.0)
    label(axr, -7.5, 0.5, "gradient 10⁻⁶", C["danger"], show_grad, ha="right",
          va="center", fontsize=11)
    axb.text(0, 2.62, "sign", color=C["neutral"], fontsize=10, fontweight="bold")
    axb.text(5.5, 2.62, "exponent", color=C["memory"], fontsize=10, fontweight="bold")
    axb.text(16, 2.62, "mantissa bits", color=C["compute"], fontsize=10, fontweight="bold")
    axb.set_xlim(0, 32.5)
    axb.set_ylim(-0.6, 2.95)
    axb.axis("off")
    axr.set_xlim(-42, 42)
    axr.set_ylim(-0.6, 2.95)
    axr.set_yticks([])
    axr.spines["left"].set_visible(False)
    axr.set_xticks([-38, -20, 0, 20, 38], ["10⁻³⁸", "10⁻²⁰", "1", "10²⁰", "10³⁸"])
    axr.set_xlabel("Normal range (log scale)", fontsize=11)
    sp.save(fig, out / f"fp_formats_step{k}.svg")

# 5. Swamping: an update below half the FP16 spacing rounds away -------------
spacing = float(np.spacing(np.float16(0.5)))  # 2^-11
w16, w32 = np.float16(0.5), np.float32(0.5)
hist16, hist32 = [float(w16)], [float(w32)]
for _ in range(10000):
    w16 = np.float16(w16 + np.float16(1e-4))
    w32 = np.float32(w32 + np.float32(1e-4))
    hist16.append(float(w16))
    hist32.append(float(w32))
assert hist16[-1] == 0.5 and abs(hist32[-1] - 1.5) < 1e-3
for k in (1, 2):
    fig = plt.figure(figsize=(5.8, 2.5))
    axg = fig.add_axes([0.02, 0.2, 0.44, 0.62])
    axl = fig.add_axes([0.6, 0.2, 0.38, 0.62])
    # left: the FP16 grid near 0.5
    axg.axhline(0, color=C["neutral"], lw=1.5)
    for j in range(-1, 3):
        xv = 0.5 + j * spacing
        axg.plot([xv, xv], [-0.18, 0.18], color=C["memory"], lw=2.5)
    axg.text(0.5, -0.3, "0.5", ha="center", va="top", fontsize=12, fontweight="bold",
             color=C["text"])
    axg.text(0.5 + spacing, -0.3, "0.50049", ha="center", va="top", fontsize=11,
             color=C["memory"])
    axg.text(0.5, -0.78, "0.5 + 0.0001 is stored as 0.5", ha="left", va="top",
             fontsize=11, color=C["danger"], fontweight="bold")
    axg.annotate("", xy=(0.5 + 1e-4, 0.42), xytext=(0.5, 0.42),
                 arrowprops=dict(arrowstyle="-|>", color=C["compute"], lw=2.2))
    axg.text(0.5 + 0.5e-4, 0.55, "+0.0001", ha="center", va="bottom", fontsize=12,
             fontweight="bold", color=C["compute"])
    axg.plot([0.5 + 1e-4], [0], "o", ms=8, mfc="white", mec=C["compute"], mew=2)
    axg.annotate("", xy=(0.5, 1.0), xytext=(0.5 + spacing, 1.0),
                 arrowprops=dict(arrowstyle="<->", color=C["memory"], lw=1.5))
    axg.text(0.5 + spacing / 2, 1.12, "FP16 spacing 4.9×10⁻⁴", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=C["memory"])
    axg.set_xlim(0.5 - 1.3 * spacing, 0.5 + 2.4 * spacing)
    axg.set_ylim(-1.2, 1.45)
    axg.axis("off")
    # right: 10,000 updates
    a = 1.0 if k == 2 else 0.0
    steps = np.arange(len(hist16))
    axl.plot(steps, hist32, color=C["memory"], lw=2.5, alpha=a)
    axl.plot(steps, hist16, color=C["danger"], lw=2.5, alpha=a)
    label(axl, 5000, 1.12, "FP32: 1.5", C["memory"], k == 2, ha="right", fontsize=12)
    label(axl, 9800, 0.58, "FP16: 0.5", C["danger"], k == 2, ha="right", fontsize=12)
    axl.set_xlim(0, 10000)
    axl.set_ylim(0.4, 1.6)
    axl.set_xticks([0, 5000, 10000], ["0", "5,000", "10,000"])
    axl.set_yticks([0.5, 1.0, 1.5])
    axl.set_xlabel("Updates of +0.0001", fontsize=11)
    axl.set_ylabel("Stored weight", fontsize=11)
    for sname in ("bottom", "left"):
        axl.spines[sname].set_alpha(a)
    axl.tick_params(colors=C["text"] if k == 2 else "white")
    axl.xaxis.label.set_alpha(a)
    axl.yaxis.label.set_alpha(a)
    sp.save(fig, out / f"swamp_step{k}.svg")

# 6. Loss scaling (illustrative distribution of gradient magnitudes) --------
rng = np.random.default_rng(7)
exps = rng.normal(-22, 4.2, 40000)
bins = np.arange(-44, 22, 1.0)
G_BEFORE = np.log2(1e-6)          # about 2^-19.9
G_AFTER = np.log2(1e-6 * 1024)    # 1.02e-3, about 2^-9.9
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    ax.axvspan(-24, -14, color=C["data_l"], alpha=0.6, lw=0)
    ax.axvspan(-14, 16, color=C["data_l"], alpha=1.0, lw=0)
    ax.axvspan(-44, -24, color=C["danger_l"], alpha=0.6, lw=0)
    h, e = np.histogram(exps, bins)
    ax.bar(e[:-1] + 0.5, h, 0.9, color=C["neutral"], alpha=1.0 if k == 1 else 0.25)
    h2, e2 = np.histogram(exps + 10, bins)
    ax.bar(e2[:-1] + 0.5, h2, 0.9, color=C["compute"], alpha=1.0 if k == 2 else 0.0)
    top = h.max() * 1.6
    ax.text(-3, top, "FP16 normal", color=C["data"], fontsize=12,
            fontweight="bold", ha="center", va="top")
    ax.text(-19, top, "sub-\nnormal", color=C["data"], fontsize=11, ha="center", va="top",
            linespacing=1.0)
    ax.text(-35, top, "becomes\nzero", color=C["danger"], fontsize=12, fontweight="bold",
            ha="center", va="top", linespacing=1.0)
    # the 65,504 ceiling, just below 2^16
    ax.axvline(16, color=C["danger"], lw=2.5)
    ax.text(15.3, top * 0.80, "overflow\n65,504", color=C["danger"], fontsize=11,
            fontweight="bold", ha="right", va="top", linespacing=1.0)
    # one gradient of 1e-6, before and after scaling
    ymark = h.max() * 1.05
    ax.plot([G_BEFORE], [ymark], "o", ms=10, color=C["text"], mec="white", mew=1.5)
    ax.text(G_BEFORE - 1.2, ymark, "10⁻⁶", color=C["text"], fontsize=12,
            fontweight="bold", ha="right", va="center")
    a2 = 1.0 if k == 2 else 0.0
    ax.annotate("", xy=(G_AFTER - 0.6, ymark), xytext=(G_BEFORE + 0.7, ymark),
                arrowprops=dict(arrowstyle="-|>", color=C["compute"], lw=2.2, alpha=a2))
    ax.plot([G_AFTER], [ymark], "o", ms=10, color=C["compute"], mec="white", mew=1.5,
            alpha=a2)
    label(ax, G_AFTER + 1.2, ymark, "1.02×10⁻³", C["compute"], k == 2,
          ha="left", va="center", fontsize=12)
    label(ax, -1, h.max() * 0.55, "× S = 2¹⁰:\nshift by 10", C["compute"], k == 2,
          ha="left", va="center", fontsize=12, linespacing=1.05)
    ax.set_xlim(-44, 21)
    ax.set_ylim(0, top * 1.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-40, -32, -24, -14, 0, 8, 16],
                  ["2⁻⁴⁰", "2⁻³²", "2⁻²⁴", "2⁻¹⁴", "1", "2⁸", "2¹⁶"], fontsize=12)
    ax.set_xlabel("Gradient magnitude (log scale). Illustrative shape.", fontsize=12)
    sp.save(fig, out / f"loss_scale_step{k}.svg")

# 7. Run time by format (after Part 1's precision_days, narrowed for a column)
sp.hbars(out / "precision_days_col.svg", [
    ("FP32", 518, "neutral", "518 days"),
    ("TF32", 70, "memory", "70 days"),
    ("BF16", 35, "compute", "35 days, $3,370"),
], "days", log=False, size=(3.9, 2.2),
    xlabel="Days on one H100, 50% utilization")

# 8. One layer's activation bytes at batch 32 (recall from Part 1), then the
# three saved copies of the scores that FlashAttention stops storing --------
parts = [(419.4, "memory"), (419.4, "compute"), (4194.3, "danger"), (10, "neutral")]
copies = [(2, "P\n2 B"), (1, "mask\n1 B"), (2, "dropped P\n2 B")]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    x = 0
    for w, ck in parts:
        ax.barh(1.45, w, left=x, color=C[ck], height=0.5, edgecolor="white")
        x += w
    ax.text(x + 150, 1.45, f"{x:,.0f} MB", va="center", fontsize=13,
            fontweight="bold", color=C["text"])
    for xt, t, ck in [(0, "proj. 419", "memory"), (2300, "scores 4,194: 83%", "danger")]:
        ax.text(xt, 1.0, t, color=C[ck], fontsize=12, fontweight="bold", va="center")
    ax.text(8100, 2.05, "× 48 layers: scores are 201 of 287 GB", color=C["danger"],
            fontsize=12, fontweight="bold", va="center", ha="right")
    shown = k == 2
    a = 1.0 if shown else 0.0
    x0 = 838.8
    for nbytes, t in copies:
        w = 4194.3 * nbytes / 5
        ax.barh(0, w, left=x0, color=C["danger_l"], height=0.62, edgecolor=C["danger"],
                lw=1.5, hatch="//", alpha=a)
        label(ax, x0 + w / 2, 0, t, C["danger"], shown, ha="center", va="center",
              fontsize=12, linespacing=1.0,
              bbox=dict(fc="white", ec="none", pad=1.5, alpha=a))
        x0 += w
    for xe in (838.8, 838.8 + 4194.3):
        ax.plot([xe, xe], [1.2, 0.31], color=C["danger"], lw=1.2, ls=":", alpha=a)
    label(ax, 5250, 0, "FlashAttention\nstores none; it keeps\na max and a sum per row", C["memory"],
          shown, va="center", fontsize=12, linespacing=1.05)
    ax.set_xlim(0, 8200)
    ax.set_ylim(-0.45, 2.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 2000, 4000], ["0", "2,000", "4,000"])
    ax.set_xlabel("MB saved per layer for backward, batch 32, FP16", fontsize=12)
    sp.save(fig, out / f"layer_bytes_step{k}.svg")

# 9. Standard attention on the kit's accelerator node -----------------------
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def kit_body(name):
    """The kit node without the host side, shifted left so the accelerator
    starts at x = 10; the right of the canvas is free for annotation."""
    keep = []
    for line in (KIT / f"{name}.svg").read_text().splitlines():
        if not line.startswith("<") or line.startswith(("<svg", "</svg", "<title")):
            continue
        xs = re.findall(r'\bx1?="([\d.]+)"', line)
        if xs and float(xs[0]) < 320:
            continue
        keep.append(line)
    return '<g transform="translate(-320,0)">\n' + "\n".join(keep) + "\n</g>"


def text(x, y, s, color, size=20, weight=700, anchor="start"):
    return (f'<text x="{x}" y="{y}" {FONT} font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}" dominant-baseline="central">{s}</text>')


for k in (1, 2, 3):
    body = kit_body("machine_hl_hbm" if k != 2 else "machine_hl_sram")
    extra = []
    blue, red, gray = C["memory"], "#A51C30", "#888888"
    extra.append(text(575, 368, "HBM", blue, 27))
    extra.append(text(575, 402, "80 GB, 3.35 TB/s", blue, 25, 400))
    if k >= 2:
        side = blue if k == 2 else gray
        extra.append(text(575, 50, "On-chip SRAM", side, 27))
        extra.append(text(575, 84, "30 to 50 MB,", side, 25, 400))
        extra.append(text(575, 116, "15 to 20 TB/s", side, 25, 400))
        extra.append(text(575, 148, "228 KB per SM", side, 25, 400))
    if k == 3:
        # SRAM bypassed: gray it out
        extra.append('<rect x="76" y="106" width="138" height="218" rx="10" fill="white" '
                     'fill-opacity="0.9" stroke="none"/>')
        extra.append(text(145, 270, "bypassed", gray, 22, 700, "middle"))
        # four numbered transfers between the compute units and HBM
        for j, (xa, down) in enumerate([(290, True), (350, False), (410, True), (470, False)]):
            y1, y2 = (318, 380) if down else (380, 318)
            head = y2 - 10 if down else y2 + 10
            extra.append(f'<line x1="{xa}" y1="{y1}" x2="{xa}" y2="{head}" stroke="{red}" '
                         'stroke-width="5"/>')
            tip = (f"{xa - 11},{head} {xa + 11},{head} {xa},{y2 + 6}" if down else
                   f"{xa - 11},{head} {xa + 11},{head} {xa},{y2 - 6}")
            extra.append(f'<polygon points="{tip}" fill="{red}"/>')
            extra.append(f'<circle cx="{xa}" cy="347" r="15" fill="white" stroke="{red}" '
                         'stroke-width="2.5"/>')
            extra.append(text(xa, 348, str(j + 1), red, 20, 700, "middle"))
        extra.append(text(575, 200, "Standard attention:", red, 25))
        extra.append(text(575, 234, "1 write S,  2 read S,", red, 25, 400))
        extra.append(text(575, 268, "3 write P,  4 read P", red, 25, 400))
        extra.append(text(10, 485, "Scores only: d_head / 2 = 64 / 2 = 32 FLOP per byte; ridge 295",
                          red, 23))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" '
           'viewBox="0 0 900 500">\n<rect width="900" height="500" fill="white"/>\n'
           + body + "\n" + "\n".join(extra) + "\n</svg>\n")
    (out / f"attention_io_step{k}.svg").write_text(svg)

# 10. Tiling: one query block streams key and value blocks -------------------
for k in (1, 2):
    fig = plt.figure(figsize=(5.8, 2.7))
    axm = fig.add_axes([0.0, 0.06, 0.38, 0.86])
    axb = fig.add_axes([0.52, 0.24, 0.46, 0.56])
    n = 32
    axm.add_patch(plt.Rectangle((0, 0), n, n, fc=C["danger_l"], ec=C["danger"], lw=2))
    for t in range(1, n):
        axm.plot([t, t], [0, n], color="white", lw=0.5)
        axm.plot([0, n], [t, t], color="white", lw=0.5)
    axm.add_patch(plt.Rectangle((0, n - 1), n, 1, fc=C["memory_l"], ec=C["memory"], lw=1.5))
    axm.add_patch(plt.Rectangle((5, n - 1), 1, 1, fc=C["memory"], ec=C["memory"], lw=1.5))
    axm.annotate("", xy=(14, n - 2.6), xytext=(6.5, n - 2.6),
                 arrowprops=dict(arrowstyle="-|>", color=C["memory"], lw=2))
    axm.text(6.5, n - 3.6, "K, V blocks stream by", color=C["memory"], fontsize=11,
             fontweight="bold", va="top")
    axm.text(n / 2, n / 2 - 3, "4,096 × 4,096 scores\n= 32 × 32 tiles of 128",
             ha="center", va="center", fontsize=11.5, fontweight="bold", color=C["danger"])
    axm.text(n / 2, n + 0.6, "one query block (128 rows)", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=C["memory"])
    axm.set_xlim(-0.5, n + 0.5)
    axm.set_ylim(-0.5, n + 3)
    axm.set_aspect("equal")
    axm.axis("off")
    shown = k == 2
    vals = [(67.1e6, "danger", "Full matrix 67.1 MB"), (65.5e3, "memory", "One tile 65.5 KB")]
    for i, (v, ck, t) in enumerate(vals):
        axb.barh(1 - i, v, color=C[ck], height=0.55, alpha=1.0 if shown else 0.0)
        label(axb, v * 1.4 if i == 0 else 3.2e5, 1 - i, t, C[ck], shown, va="center",
              fontsize=12)
    axb.axvline(228e3, color=C["data"], lw=2, ls="--", alpha=1.0 if shown else 0.0)
    label(axb, 228e3, 1.55, "228 KB shared\nmemory per SM", C["data"], shown,
          ha="center", va="bottom", fontsize=10.5)
    axb.set_xscale("log")
    axb.set_xlim(1e4, 3e10)
    axb.set_ylim(-0.6, 1.5)
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    axb.set_xticks([1e4, 1e6, 1e8], ["10 KB", "1 MB", "100 MB"])
    axb.minorticks_off()
    axb.set_xlabel("Bytes per head (FP32, log scale)", fontsize=11)
    if not shown:
        axb.tick_params(colors="white")
        axb.spines["bottom"].set_alpha(0)
        axb.xaxis.label.set_alpha(0)
    sp.save(fig, out / f"tile_grid_step{k}.svg")

# 11. What each kernel keeps for backward, per head at 4,096 tokens ----------
sp.hbars(out / "flash_store.svg", [
    ("Standard attention", 138.4, "danger", "138.4 MB"),
    ("FlashAttention", 4.2, "memory", "4.2 MB: Q, K, V, O, and m, l per row"),
], "MB", xlabel="Stored for backward, per head, 4,096 tokens (MB, log scale)",
    size=sp.SHORT)

# 12. Step memory against batch size with FlashAttention ----------------------
bs = np.arange(0, 41)
mem = 24 + 85.6 / 32 * bs
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(bs, mem, color=C["data"], lw=2.8)
    ax.axhline(CAP, color=C["memory"], lw=2, ls="--")
    ax.text(0.5, CAP + 4, "H100: 85.9 GB", color=C["memory"], fontsize=11,
            fontweight="bold")
    ax.text(39.5, 8, "24 GB + 2.68 GB per sequence", ha="right",
            va="bottom", color=C["data"], fontsize=11, fontweight="bold")
    shown = k == 2
    for b, col in [(16, C["data"]), (32, C["danger"])]:
        m = 24 + 85.6 / 32 * b
        ax.plot([b], [m], "o", ms=10, color=col, mec="white", mew=1.5,
                alpha=1.0 if shown else 0.0)
        label(ax, b - 0.8, m + 5, f"batch {b}: {m:.1f} GB", col, shown, ha="right",
              fontsize=11.5)
    ax.plot([23.1], [CAP], "o", ms=9, color=C["memory"], mec="white", mew=1.5)
    ax.annotate("about 23 fit", xy=(23.1, CAP), xytext=(26.5, 52), fontsize=11.5,
                fontweight="bold", color=C["memory"],
                arrowprops=dict(arrowstyle="-|>", color=C["memory"], lw=1.5))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 135)
    ax.set_xlabel("Sequences resident at once (1,024 tokens each)", fontsize=11)
    ax.set_ylabel("GB per step")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"batch_fit_step{k}.svg")

# 13. Segment checkpointing (copied from Part 1's generator) -----------------
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    for s in range(8):
        x0 = s * 1.0
        ax.add_patch(plt.Rectangle((x0 + 0.02, 0.0), 0.12, 1.0, fc=C["data"], ec=C["data"]))
        rebuilt = k == 2 and s == 5
        ax.add_patch(plt.Rectangle((x0 + 0.17, 0.0), 0.78, 1.0,
                                   fc=C["compute_l"] if rebuilt else "white",
                                   ec=C["compute"] if rebuilt else C["neutral_l"],
                                   lw=2 if rebuilt else 1.2, ls="-" if rebuilt else "--"))
    label(ax, 0.0, 1.12, "kept: 8 segment inputs, 0.84 GB", C["data"], True,
          fontsize=15, va="bottom")
    label(ax, 5.56, -0.12, "rebuilt in backward,\none at a time: 10.7 GB", C["compute"],
          k == 2, fontsize=15, va="top", ha="center", linespacing=1.05)
    label(ax, 8.05, 1.12, "dropped", C["neutral"], True, fontsize=15, va="bottom",
          ha="right")
    label(ax, 0.56, -0.12, "6 layers", C["neutral"], True, fontsize=15, va="top",
          ha="center")
    ax.set_xlim(-0.05, 8.1)
    ax.set_ylim(-1.05, 1.55)
    ax.axis("off")
    sp.save(fig, out / f"ckpt_segments_step{k}.svg")

# 15. Activation memory against checkpoint count, 48 layers (table 13) -------
ks = np.arange(1, 49)
units = ks + 48 / ks
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(ks, units, color=C["compute"], lw=2.8)
    ax.axhline(48, color=C["neutral"], lw=1.5, ls=":")
    ax.text(25, 51.5, "no checkpoints: 48", ha="center", color=C["neutral"], fontsize=11,
            fontweight="bold")
    for kk, txt, xt, yt, ha in [(1, "k = 1: 49", 2.5, 53, "left"),
                                (7, "k = 7: about 14", 9, 9.5, "left"),
                                (48, "k = 48: 49", 47.5, 53, "right")]:
        u = kk + 48 / kk
        ax.plot([kk], [u], "o", ms=9, color=C["compute"], mec="white", mew=1.5)
        ax.text(xt, yt, txt, ha=ha, va="center", fontsize=11.5, fontweight="bold",
                color=C["compute"])
    ax.text(40, 24, "memory ≈ k + 48/k\n(L = 48 layers)", ha="center", va="top",
            fontsize=12, fontweight="bold", color=C["compute"], linespacing=1.05)
    label(ax, 9, 3.8, "71% less memory, about 33% more arithmetic", C["data"], k == 2,
          va="center", fontsize=11.5)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 58)
    ax.set_xlabel("Checkpoints k (curve derived from table 13)", fontsize=11)
    ax.set_ylabel("Layers of activations")
    ax.grid(color=C["grid"], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"sqrt_ckpt_step{k}.svg")

print("figures written")
