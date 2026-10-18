#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 9 (Data Selection).

Every value is sourced. "Chapter" means the rendered Data Selection chapter.
  - Data wall: reported pretraining tokens and the stock of public human text
    (Villalobos et al. 2024), read from MLSysIM's literature series
    (Literature.Series.PretrainingTokens and PublicTextStock), the same data
    as the Data Engineering chapter's figure; the 2.9x per year fit is
    recomputed here exactly as the book does (least squares on log10 tokens).
  - GPT-2 on WebText: about 10 billion unique tokens, 30 billion asked for by
    the compute-optimal fit, about 167 billion processed, about 17 passes,
    4.4 days and 590 kWh on one eight-H100 machine (chapter, Sec. 9.1).
  - Collapse curve: the chapter's illustrative scenario (Sec. 9.10 and its
    margin figure), 95 to 78 percent over five recursive generations; the
    middle points are read from the book's margin figure. Mechanism:
    Shumailov et al. 2024. Labeled illustrative on the plot.
  - Tokens per parameter: GPT-3 about 1.7 (Brown et al. 2020), Llama 2 70B
    about 28.6 (Touvron et al. 2023), Chinchilla reference about 20
    (Hoffmann et al. 2022), GPT-2 on WebText 6.7 (chapter, Sec. 9.2.3).
  - Selection inequality scoreboard: Example 9.2 and Sec. 9.7.1. Baseline
    211 min; Option A saves 188.8 min after 42 s of scoring; Option B saves
    189.1 min after 18 s; rescoring every epoch costs 70 min plus 21 min.
  - Deduplication: near-duplicate shares 3.04 percent (C4) and 13.63 percent
    (RealNews), Lee et al. 2022, applied to GPT-2's 4.4-day (105.6 h),
    590 kWh run (chapter, Sec. 9.3.2).
  - Coverage floor: 1M examples, 0.1 percent rare (1,000), unstratified
    10 percent coreset keeps about 100, illustrative floor of 150 (chapter,
    Sec. 9.3.4 and 9.10).
  - Selection via proxy: Coleman et al., ICLR 2020 (arXiv:1906.11829,
    abstract): on CIFAR10, proxies remove up to 50 percent of the data
    without harming final accuracy, a 1.6x end-to-end training-time gain.
  - FixMatch on CIFAR-10: 40, 250, 4,000 labels give 88.6, 94.9, 95.7
    percent (Sohn et al. 2020; chapter table 4).
  - ResNet-50 data path on one H100: chapter, Napkin Math 9.4 and figure
    (148 GB, 21 s; 10,100 images/s demanded; 3.2e16 FLOPs, 126 s, 88 kJ;
    scoring 42 s once; host supplies 5,250 images/s, Napkin Math 9.5).
  - Data echoing: r = 10,100 / 5,250 = 1.92; utilization min(1, e / r);
    90 epochs from 6.1 h to 3.2 h at e = 2 (Napkin Math 9.5).
  - Cost breakdown: Napkin Math 9.6, an illustrative scenario, recomputed at
    the chapter's 1.28M images and 148 GB: $50,000 acquisition, $64,000
    labels, $35.52 storage, $421 compute (105.3 GPU-h at $4). ROI of a
    $55,000 deduplication pipeline saving $10,000 per run (tables 13, 14).

Run from this folder with the MLSysBook venv (it imports mlsysim):
    python3 make_figs.py
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent


def note(ax, text, x=0.0, y=-0.30):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, color=C["neutral"])


# -----------------------------------------------------------------------------
# 1. The data wall (slide variant of Data Engineering's figure, from MLSysIM)
# -----------------------------------------------------------------------------
def data_wall():
    root = Path(__file__).resolve().parents[4] / "mlsysim"
    sys.path.insert(0, str(root))
    from mlsysim import Literature
    corpora = Literature.Series.PretrainingTokens.entries
    stock = {r["bound"]: r["tokens"] for r in Literature.Series.PublicTextStock.entries}

    def year(d):
        d = dt.date.fromisoformat(d)
        return d.year + (d.timetuple().tm_yday - 1) / 365.25

    xs = np.array([year(r["date"]) for r in corpora])
    ys = np.array([r["tokens"] for r in corpora])
    slope, icept = np.polyfit(xs, np.log10(ys), 1)
    growth = 10 ** slope
    assert 2.8 <= growth <= 3.0, growth
    show = {"GPT-3": (4, 8, "left"), "Chinchilla": (-6, 6, "right"),
            "Llama 2": (6, -4, "left"), "Qwen3": (-8, 4, "right")}
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.9))
        if k == 2:
            ax.axhspan(stock["low"], stock["high"], color=C["compute_l"], zorder=0)
            ax.axhline(stock["median"], color=C["compute"], lw=1.2, ls=":")
            ax.text(2019.95, stock["median"] * 1.35, "Stock of public human text",
                    color=C["compute"], fontsize=12, fontweight="bold", va="bottom")
            ax.text(2019.95, stock["median"] * 0.75, "estimate, 90% interval",
                    color=C["compute"], fontsize=10, va="top")
        xx = np.linspace(2020, 2028.5, 50)
        ax.plot(xx, 10 ** (slope * xx + icept), color=C["memory"], lw=2, ls="--")
        ax.scatter(xs, ys, s=55, color=C["memory"], edgecolors="white", zorder=3)
        for r, x, y in zip(corpora, xs, ys):
            if r["model"] in show:
                dx, dy, ha = show[r["model"]]
                ax.annotate(r["model"], (x, y), xytext=(dx, dy), textcoords="offset points",
                            fontsize=11, ha=ha, color=C["text"])
        ax.text(2026.1, 10 ** (slope * 2026.1 + icept) / 6, f"{growth:.1f}× per year",
                color=C["memory"], fontsize=12, fontweight="bold", ha="left")
        ax.set_yscale("log")
        ax.set_ylim(1e11, 3e15)
        ax.set_xlim(2019.8, 2028.5)
        ax.set_xticks(range(2020, 2029, 2))
        ax.set_ylabel("Training tokens")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        sp.save(fig, out / f"data_wall_step{k}.svg")


# -----------------------------------------------------------------------------
# 2. GPT-2 on WebText: unique, compute-optimal, processed tokens
# -----------------------------------------------------------------------------
def gpt2_tokens():
    sp.hbars(out / "gpt2_tokens.svg", [
        ("Unique in WebText", 10, "data", "10 B"),
        ("Compute-optimal", 30, "neutral", "30 B"),
        ("Processed in training", 167, "compute", "167 B: 17 passes"),
    ], "B", log=False, xlabel="Tokens (billions)", size=(4.2, 1.7))


# -----------------------------------------------------------------------------
# 3. The collapse curve (chapter's illustrative scenario)
# -----------------------------------------------------------------------------
GENS = [1, 2, 3, 4, 5]
ACC = [95, 92, 88, 83, 78]


def collapse(size=(5.8, 2.3), stem="collapse", label_note=True, steps=(1, 2), font=1.0):
    for k in steps:
        fig, ax = plt.subplots(figsize=size)
        n = 1 if k == 1 else 5
        ax.plot(GENS[:n], ACC[:n], color=C["danger"], lw=3, marker="o", ms=9)
        ax.text(1.12, 95.6, "95%", color=C["danger"], fontsize=13 * font, fontweight="bold")
        if k == 2:
            ax.text(4.85, 78.0, "78%", color=C["danger"], fontsize=13 * font, fontweight="bold",
                    ha="right", va="center")
            if label_note:
                ax.text(3.0, 94.5, "−17 points in five generations", color=C["danger"],
                        fontsize=12, fontweight="bold", ha="left")
        ax.set_xlim(0.7, 5.3)
        ax.set_ylim(74, 99)
        ax.set_xticks(GENS)
        ax.set_xticklabels([f"Gen {g}" for g in GENS] if label_note else [str(g) for g in GENS])
        ax.tick_params(labelsize=11 * font)
        ax.set_ylabel("Accuracy (%)", fontsize=11 * font)
        if not label_note:
            ax.set_xlabel("Generation", fontsize=11 * font)
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        if label_note:
            note(ax, "Illustrative scenario from the chapter, not a measurement", y=-0.34)
        sp.save(fig, out / f"{stem}_step{k}.svg")


# -----------------------------------------------------------------------------
# 3b. Learning curves: efficient selection against random sampling
# -----------------------------------------------------------------------------
def ppd_curve():
    """Slide variant of the chapter's figure 3 (illustrative curves): the
    efficient curve plateaus near 25 percent of the data, random sampling
    climbs across the whole budget, and the gap is read at a 20 percent
    budget. The curve shapes reproduce the book figure; no value on them is
    a measurement, and none is printed."""
    x = np.linspace(0, 100, 400)
    eff = 95 * (1 - np.exp(-x / 6.5))
    rnd = 95 * (1 - np.exp(-x / 25))
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.7))
        ax.plot(x, rnd, color=C["neutral"], lw=3, ls="--")
        ax.plot(x, eff, color=C["memory"], lw=3.5)
        ax.text(62, 42, "Random sampling", color=C["neutral"], fontsize=13, fontweight="bold")
        ax.text(30, 99, "Efficient selection", color=C["memory"], fontsize=13, fontweight="bold")
        if k == 2:
            y20e, y20r = 95 * (1 - np.exp(-20 / 6.5)), 95 * (1 - np.exp(-20 / 25))
            ax.annotate("", xy=(20, y20r + 1), xytext=(20, y20e - 1),
                        arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=2))
            ax.text(21.5, 34, "same data budget:\naccuracy gap", color=C["danger"],
                    fontsize=12, fontweight="bold", va="center")
            x90r = -25 * np.log(1 - 90 / 95)
            x90e = -6.5 * np.log(1 - 90 / 95)
            ax.annotate("", xy=(x90r - 1, 90), xytext=(x90e + 1, 90),
                        arrowprops=dict(arrowstyle="<->", color=C["data"], lw=2))
            ax.text(x90r, 80, "same target:\ndata saved", color=C["data"],
                    fontsize=12, fontweight="bold", ha="center", va="top")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 50, 100])
        ax.set_xlabel("Dataset size (% of total)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        sp.save(fig, out / f"ppd_curve_step{k}.svg")


# -----------------------------------------------------------------------------
# 3c. Curriculum warmup: linear pacing serves about half the samples
# -----------------------------------------------------------------------------
def warmup():
    """Linear pacing (chapter Sec. 9.4.1): epoch j of a W-epoch warmup serves
    the easiest j/W of the D samples, so the warmup serves D(W+1)/2 in place
    of W*D. W = 10 is an illustrative choice: 5.5 D of 10 D."""
    W = 10
    j = np.arange(1, W + 1)
    frac = j / W
    assert abs(frac.sum() - (W + 1) / 2) < 1e-9
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.bar(j, np.ones(W), color="white", edgecolor=C["neutral"], lw=1.2, ls="--", width=0.75)
    ax.bar(j, frac, color=C["data"], width=0.75)
    post = np.arange(W + 1, W + 4)
    ax.bar(post, np.ones(3), color=C["neutral_l"], edgecolor=C["neutral"], lw=1, width=0.75)
    ax.text(W + 2, 1.06, "full data", ha="center", fontsize=11, color=C["neutral"])
    ax.text(1.0, 1.24, "Warmup serves 5.5 D samples in place of 10 D",
            fontsize=13, fontweight="bold", color=C["data"])
    ax.text(1.0, 1.10, "dashed: what full-data epochs would serve", fontsize=11,
            color=C["neutral"])
    ax.set_xticks(list(j) + list(post))
    ax.set_xticklabels([str(v) for v in j] + ["", "", ""])
    ax.set_xlim(0.4, W + 3.6)
    ax.set_ylim(0, 1.38)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "D/2", "D"])
    ax.set_xlabel("Epoch (warmup W = 10, illustrative)")
    ax.set_ylabel("Samples served")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / "warmup.svg")


# -----------------------------------------------------------------------------
# 4. Tokens per parameter
# -----------------------------------------------------------------------------
def tokens_per_param():
    rows = [("GPT-3 (175 B)", 1.7, "neutral", "1.7"),
            ("GPT-2 XL on WebText", 6.7, "danger", "6.7"),
            ("Llama 2 70B", 28.6, "neutral", "28.6")]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.1))
        for i, (lab, v, ck, txt) in enumerate(rows):
            y = 2 - i
            col = C[ck] if (k == 2 or ck == "neutral") else C["neutral"]
            ax.barh(y, v, color=col, height=0.6)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=13, fontweight="bold", color=C["text"])
            ax.text(v + 0.4, y, txt, va="center", fontsize=13, fontweight="bold", color=col)
        alpha = 1.0 if k == 2 else 0.0
        ax.axvline(20, color=C["data"], lw=2, ls="--", alpha=alpha)
        ax.text(20.4, 2.45, "Chinchilla reference: about 20", color=C["data"], fontsize=11,
                fontweight="bold", alpha=alpha, va="bottom")
        ax.text(20.8, 1.0, "6.7 is a third of 20", color=C["danger"], fontsize=12,
                fontweight="bold", va="center", alpha=alpha)
        ax.set_yticks([])
        ax.set_xlim(0, 36)
        ax.set_ylim(-0.5, 2.9)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Training tokens per parameter")
        sp.save(fig, out / f"tokens_per_param_step{k}.svg")


# -----------------------------------------------------------------------------
# 5. Selection inequality scoreboard (stacked: scoring + training)
# -----------------------------------------------------------------------------
def scoreboard():
    rows = [("Full data", 0.0, 211.0, "211 min"),
            ("Coreset, target scores once", 0.7, 211 - 188.8 - 0.7, "saves 89.7%"),
            ("Coreset, proxy scores once", 0.3, 211 - 189.1 - 0.3, "saves 89.9%"),
            ("Coreset, rescore every epoch", 70.0, 21.0, "70 + 21 min: saves 57%")]
    show_upto = {1: 1, 2: 3, 3: 4}
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        for i, (lab, score, train, txt) in enumerate(rows):
            y = 3 - i
            vis = i < show_upto[k]
            a = 1.0 if vis else 0.0
            ax.barh(y, train, left=score, color=C["neutral"] if i == 0 else C["data"],
                    height=0.6, alpha=a)
            if score:
                ax.barh(y, score, color=C["compute"], height=0.6, alpha=a)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=12, fontweight="bold", color=C["text"])
            ck = "danger" if i == 3 else ("neutral" if i == 0 else "data")
            ax.text(score + train + 3, y, txt, va="center", fontsize=12, fontweight="bold",
                    color=C[ck], alpha=a)
        a3 = 1.0 if k == 3 else 0.0
        ax.text(35, -0.62, "scoring", color=C["compute"], fontsize=11, fontweight="bold",
                ha="center", va="top", alpha=a3)
        ax.set_yticks([])
        ax.set_xlim(0, 300)
        ax.set_ylim(-1.0, 3.5)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Minutes on one H100 (100 epochs)")
        sp.save(fig, out / f"scoreboard_step{k}.svg")


# -----------------------------------------------------------------------------
# 6. Deduplication returns hours of GPT-2's run
# -----------------------------------------------------------------------------
def dedup_hours():
    run_h, run_kwh = 4.4 * 24, 590
    rows = [("C4: 3.04% near-duplicates", 3.04),
            ("RealNews: 13.63% near-duplicates", 13.63)]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 1.8))
        for i, (lab, pct) in enumerate(rows):
            y = 1 - i
            h = run_h * pct / 100
            e = run_kwh * pct / 100
            a = 1.0 if i < k else 0.0
            ax.barh(y, h, color=C["data"], height=0.6, alpha=a)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=12, fontweight="bold", color=C["text"])
            ax.text(h + 0.3, y, f"{h:.0f} h, {e:.0f} kWh", va="center", fontsize=13,
                    fontweight="bold", color=C["data"], alpha=a)
        ax.set_yticks([])
        ax.set_xlim(0, 22)
        ax.set_ylim(-0.5, 1.5)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Hours of GPT-2's 106-hour run saved")
        sp.save(fig, out / f"dedup_hours_step{k}.svg")


# -----------------------------------------------------------------------------
# 7. Coverage floor
# -----------------------------------------------------------------------------
def coverage():
    rows = [("Rare examples in the 1M pool", 1000, "neutral", "1,000"),
            ("Kept by an unstratified 10% coreset", 100, "danger", "about 100"),
            ("Kept with the floor reserved first", 150, "data", "150")]
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.1))
        for i, (lab, v, ck, txt) in enumerate(rows):
            y = 2 - i
            a = 1.0 if i < k else 0.0
            ax.barh(y, v, color=C[ck], height=0.6, alpha=a)
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=13, fontweight="bold", color=C["text"])
            ax.text(max(v, 150) + 25, y, txt, va="center", fontsize=13,
                    fontweight="bold", color=C[ck], alpha=a)
        ax.axvline(150, color=C["data"], lw=1.5, ls="--")
        ax.text(150, 2.45, " floor: 150 (illustrative)", color=C["data"], fontsize=11,
                fontweight="bold", va="bottom")
        ax.set_yticks([])
        ax.set_xlim(0, 1300)
        ax.set_ylim(-0.5, 2.9)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Rare-class examples")
        sp.save(fig, out / f"coverage_step{k}.svg")


# -----------------------------------------------------------------------------
# 8. Selection via proxy (Coleman et al. 2020)
# -----------------------------------------------------------------------------
def svp():
    sp.hbars(out / "svp_coreset.svg", [
        ("Target trains on all of CIFAR10", 1.0, "neutral", "1.0"),
        ("Proxy removes 50%, target trains", 1 / 1.6, "data", "1.6× faster, same final accuracy"),
    ], "", log=False, xlabel="End-to-end training time (relative)", size=(5.8, 1.6))


# -----------------------------------------------------------------------------
# 9. FixMatch on CIFAR-10
# -----------------------------------------------------------------------------
def fixmatch():
    sp.hbars(out / "fixmatch.svg", [
        ("40 labels", 88.6, "data", "88.6%"),
        ("250 labels", 94.9, "data", "94.9%"),
        ("4,000 labels", 95.7, "data", "95.7%"),
    ], "", log=False, xlabel="CIFAR-10 test accuracy (%); full labeled set: 50,000",
        size=(5.8, 2.0))


# -----------------------------------------------------------------------------
# 10. Data echoing
# -----------------------------------------------------------------------------
def echo():
    r = 10100 / 5250
    e = np.linspace(1, 3.2, 200)
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.5))
        if k >= 2:
            ax.plot(e, np.minimum(1, e / r) * 100, color=C["compute"], lw=3)
            ax.axvline(r, color=C["neutral"], lw=1, ls="--")
            ax.text(r + 0.04, 60, f"e = r = {r:.2f}", color=C["neutral"], fontsize=11)
            ax.plot([2], [100], "o", ms=11, color=C["compute"], mec="white", mew=1.5, zorder=5)
            ax.annotate("e = 2: 90 epochs, 6.1 h to 3.2 h", (2, 100), xytext=(8, -26),
                        textcoords="offset points", fontsize=12, fontweight="bold",
                        color=C["compute"])
        if k == 3:
            ax.plot(e, [100] * len(e), color=C["memory"], lw=2.5, ls=(0, (4, 2)))
            ax.text(2.35, 103, "host twice as fast (r = 0.5)", color=C["memory"],
                    fontsize=11, fontweight="bold")
        ax.plot([1], [100 / r], "o", ms=11, color=C["danger"], mec="white", mew=1.5, zorder=5)
        ax.annotate("no echo: busy 52%", (1, 100 / r), xytext=(10, -4),
                    textcoords="offset points", fontsize=12, fontweight="bold",
                    color=C["danger"])
        ax.set_xlim(0.9, 3.2)
        ax.set_ylim(30, 112)
        ax.set_xticks([1, 1.5, 2, 2.5, 3])
        ax.set_xlabel("Echo factor e (uses of each prepared image)")
        ax.set_ylabel("Accelerator busy (%)")
        ax.grid(axis="y", color=C["grid"], lw=0.6)
        sp.save(fig, out / f"echo_step{k}.svg")


# -----------------------------------------------------------------------------
# 11. Cost breakdown and amortized ROI
# -----------------------------------------------------------------------------
def cost():
    labels = 1.28e6 * 0.05
    storage = 148 * 0.02 * 12
    compute = 30 * 3.51 * 4
    total = 50000 + labels + storage + compute
    assert round(total) == 114457, total
    sp.hbars(out / "cost_bars.svg", [
        ("Acquisition (licensed images)", 50000, "data", "$50,000"),
        ("Labels (1.28M at $0.05)", labels, "data", "$64,000"),
        ("Storage (148 GB, 12 months)", storage, "neutral", "$36"),
        ("Compute (30 runs on one H100)", compute, "compute", "$421 = 0.4%"),
    ], "", log=True, xlabel="Dollars (log scale); total $114,457", size=(5.8, 2.3))


def roi():
    runs = np.array([1, 5, 10, 50])
    vals = (10000 * runs - 55000) / 55000 * 100
    assert [round(v, 1) for v in vals] == [-81.8, -9.1, 81.8, 809.1], vals
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    xs = np.linspace(1, 50, 200)
    ax.plot(xs, (10000 * xs - 55000) / 55000 * 100, color=C["data"], lw=3)
    ax.axhline(0, color=C["neutral"], lw=1)
    ax.axvline(5.5, color=C["neutral"], lw=1, ls="--")
    ax.text(5.8, 500, "break-even:\n5.5 runs", color=C["neutral"], fontsize=11)
    for n, v in zip(runs, vals):
        ck = "data" if v > 0 else "danger"
        ax.plot([n], [v], "o", ms=9, color=C[ck], zorder=5)
        off, ha = {50: ((-8, 4), "right"), 5: ((-6, -20), "right")}.get(n, ((9, -20), "left"))
        ax.annotate(f"{v:+.0f}%", (n, v), xytext=off,
                    textcoords="offset points", fontsize=12, fontweight="bold",
                    color=C[ck], ha=ha)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 50])
    ax.set_xticklabels(["1", "5", "10", "50"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 60)
    ax.set_ylim(-260, 950)
    ax.set_xlabel("Training runs that reuse the deduplicated data (log scale)")
    ax.set_ylabel("Return (%)")
    ax.grid(axis="y", color=C["grid"], lw=0.6)
    sp.save(fig, out / "roi_runs.svg")


# -----------------------------------------------------------------------------
# 12. Hand-drawn SVG diagrams (MinHash build, ResNet-50 data path)
# -----------------------------------------------------------------------------
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def svg_doc(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" {FONT}>\n'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>\n{body}</svg>\n')


def text(x, y, s, size=18, color=None, weight="normal", anchor="middle", op=1.0):
    color = color or C["text"]
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'text-anchor="{anchor}" opacity="{op}">{s}</text>\n')


def box(x, y, w, h, stroke, fill, sw=2.5, op=1.0, rx=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>\n')


def arrow(x1, y, x2, op=1.0, color="#666666"):
    return (f'<g opacity="{op}"><line x1="{x1}" y1="{y}" x2="{x2 - 10}" y2="{y}" stroke="{color}" '
            f'stroke-width="3"/><polygon points="{x2},{y} {x2 - 12},{y - 7} {x2 - 12},{y + 7}" '
            f'fill="{color}"/></g>\n')


def minhash():
    """Document, shingles, signature, bands and buckets, candidate check.
    Shingle text and signature values are illustrative; 128 hashes, 8M
    documents, and 4.1 GB come from the chapter (Sec. 9.3.2). All pairs:
    8e6 x 8e6 / 2 = 3.2e13 comparisons, computed here."""
    W, H = 900, 360
    xs = [20, 240, 470, 690]
    for k in range(1, 5):
        b = ""
        vis = [1.0 if i < k else 0.0 for i in range(4)]
        # column 1: document
        b += text(xs[0] + 95, 40, "Web page", 18, weight="bold", op=vis[0])
        b += box(xs[0], 60, 190, 150, C["neutral"], "#F4F4F4", op=vis[0])
        for i, line in enumerate(["the cat sat on", "the mat and the", "cat sat on the", "mat again ..."]):
            b += text(xs[0] + 14, 95 + 30 * i, line, 17, anchor="start", op=vis[0])
        # column 2: shingles
        b += arrow(xs[0] + 196, 135, xs[1] - 6, op=vis[1])
        b += text(xs[1] + 95, 40, "Shingles", 18, weight="bold", op=vis[1])
        for i, sh in enumerate(["the cat sat", "cat sat on", "sat on the", "on the mat"]):
            b += box(xs[1], 62 + 37 * i, 190, 30, C["memory"], C["memory_l"], sw=1.5, op=vis[1], rx=4)
            b += text(xs[1] + 95, 83 + 37 * i, sh, 16, op=vis[1])
        # column 3: signature
        b += arrow(xs[1] + 196, 135, xs[2] - 6, op=vis[2])
        b += text(xs[2] + 95, 40, "Signature", 18, weight="bold", op=vis[2])
        vals = ["17", "4", "92", "33", "8", "61", "25", "70"]
        for i, v in enumerate(vals):
            yy = 62 + 18.5 * i
            band = i // 2
            fill = [C["compute_l"], C["data_l"], C["compute_l"], C["data_l"]][band]
            b += box(xs[2] + 45, yy, 100, 18, C["neutral"], fill, sw=1, op=vis[2], rx=1)
            b += text(xs[2] + 95, yy + 15, v, 15, op=vis[2])
        b += text(xs[2] + 95, 230, "128 minimum hashes", 16, color=C["neutral"], op=vis[2])
        b += text(xs[2] + 95, 250, "(8 of them shown)", 15, color=C["neutral"], op=vis[2])
        # column 4: bands and buckets
        b += arrow(xs[2] + 196, 135, xs[3] - 6, op=vis[3])
        b += text(xs[3] + 95, 40, "Bands to buckets", 18, weight="bold", op=vis[3])
        for i in range(3):
            hl = i == 1
            b += box(xs[3] + 2 + 66 * i, 70, 60, 110, C["data"] if hl else C["neutral"],
                     C["data_l"] if hl else "#F4F4F4", sw=3 if hl else 1.5, op=vis[3], rx=4)
        b += text(xs[3] + 95, 115, "page A", 15, color=C["data"], weight="bold", op=vis[3])
        b += text(xs[3] + 95, 140, "page B", 15, color=C["data"], weight="bold", op=vis[3])
        b += text(xs[3] + 95, 210, "same bucket:", 15, color=C["data"], weight="bold", op=vis[3])
        b += text(xs[3] + 95, 230, "compare only these", 15, color=C["data"], weight="bold", op=vis[3])
        # bottom line: the all-pairs cost, crossed out in the last step
        b += text(450, 295, "All pairs of 8M pages: 3.2 × 10¹³ comparisons", 19,
                  color=C["neutral"], op=vis[3])
        b += (f'<line x1="225" y1="289" x2="675" y2="289" stroke="{C["danger"]}" '
              f'stroke-width="2.5" opacity="{vis[3]}"/>\n')
        b += text(450, 335, "Instead: one 40 GB read (6 s) and 4.1 GB of signatures", 19,
                  color=C["data"], weight="bold", op=vis[3])
        (out / f"minhash_step{k}.svg").write_text(svg_doc(W, H, b))


def data_path():
    """ResNet-50 on one H100: four stages (book figure, redrawn at slide
    size). Numbers: chapter, Napkin Math 9.4 and 9.5 and its data-path
    figure."""
    W, H = 900, 330
    stages = [("Local NVMe", "storage", C["memory"], C["memory_l"], "148 GB read", "21 s per epoch"),
              ("Host CPUs", "decode, augment", C["data"], C["data_l"], "needs 10,100", "images/s"),
              ("PCIe link", "", C["network"], C["network_l"], "193 GB", "3 s per epoch"),
              ("H100", "accelerator", C["compute"], C["compute_l"], "126 s, 88 kJ", "per epoch")]
    notes = {3: ["Coreset: 63 s, 44 kJ", "scoring: 42 s, once"],
             0: ["Halves only if the", "74 GB subset is written"],
             1: ["Supplies only 5,250/s:", "the slow stage"]}
    order = [3, 0, 1]
    xs = [10, 240, 470, 700]
    for k in range(1, 4):
        b = ""
        active = order[k - 1]
        for i, (t1, t2, st, fl, n1, n2) in enumerate(stages):
            x = xs[i]
            w = 190
            hl = i == active
            red = (i == 1 and k == 3)
            b += box(x, 140, w, 110, C["danger"] if red else st, fl, sw=6 if hl else 2.5)
            b += text(x + w / 2, 188 if t2 else 202, t1, 21, weight="bold")
            if t2:
                b += text(x + w / 2, 214, t2, 17)
            b += text(x + w / 2, 285, n1, 17, color=C["text"])
            b += text(x + w / 2, 308, n2, 17, color=C["text"])
            if i < 3:
                b += arrow(x + w + 4, 195, xs[i + 1] - 4)
            shown = i in order[:k]
            if i in notes:
                for j, line in enumerate(notes[i]):
                    col = C["danger"] if i == 1 else st
                    b += text(x + w / 2, 80 + 26 * j, line, 17, color=col, weight="bold",
                              op=1.0 if shown else 0.0)
        (out / f"data_path_step{k}.svg").write_text(svg_doc(W, H, b))


def synthetic_porch():
    """Schematic of domain-randomized synthetic porch frames for the
    discussion slide (drawn here; no openly licensed photo of synthetic
    training frames was found). No numbers."""
    W, H = 900, 335
    tiles = [("Noon, blue door", "#E9E2D0", "#3A6EA5", "#BFB8A8", 1.0, "person"),
             ("Dusk, red door", "#B7A58C", "#A23B32", "#8C8170", 0.75, "person"),
             ("Night, porch light", "#3B3F55", "#5C4A3A", "#2E3142", 0.5, "shadow")]
    b = ""
    for i, (lab, wall, door, floor, lum, who) in enumerate(tiles):
        x0, y0, w, h = 15 + 297 * i, 20, 275, 230
        b += f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="{wall}"/>\n'
        b += f'<rect x="{x0}" y="{y0 + 170}" width="{w}" height="60" fill="{floor}"/>\n'
        b += f'<rect x="{x0 + 30}" y="{y0 + 40}" width="70" height="130" fill="{door}"/>\n'
        b += f'<circle cx="{x0 + 88}" cy="{y0 + 108}" r="4" fill="#D9C27A"/>\n'
        b += f'<rect x="{x0 + 120}" y="{y0 + 55}" width="14" height="22" rx="3" fill="#F2D27A" opacity="{lum}"/>\n'
        if who == "person":
            col = "#2B2B2B" if lum > 0.8 else "#4A3A30"
            b += f'<circle cx="{x0 + 200}" cy="{y0 + 85}" r="17" fill="{col}"/>\n'
            b += f'<rect x="{x0 + 180}" y="{y0 + 104}" width="40" height="70" rx="12" fill="{col}"/>\n'
            b += f'<rect x="{x0 + 183}" y="{y0 + 170}" width="12" height="35" fill="{col}"/>\n'
            b += f'<rect x="{x0 + 205}" y="{y0 + 170}" width="12" height="35" fill="{col}"/>\n'
        else:
            b += (f'<path d="M{x0 + 160},{y0 + 205} L{x0 + 260},{y0 + 205} L{x0 + 240},{y0 + 180} '
                  f'L{x0 + 200},{y0 + 178} Z" fill="#15161F" opacity="0.8"/>\n')
        b += text(x0 + w / 2, y0 + h + 34, lab, 26, weight="bold")
    b += text(450, 322, "Rendered: every label is known exactly.", 24,
              color=C["neutral"])
    (out / "synthetic_porch.svg").write_text(svg_doc(W, H, b))


if __name__ == "__main__":
    data_wall()
    gpt2_tokens()
    collapse()
    collapse(size=(2.9, 2.2), stem="collapse_final", label_note=False, steps=(2,), font=1.1)
    ppd_curve()
    warmup()
    synthetic_porch()
    tokens_per_param()
    scoreboard()
    dedup_hours()
    coverage()
    svp()
    fixmatch()
    echo()
    cost()
    roi()
    minhash()
    data_path()
    print("figures written")
