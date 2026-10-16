#!/usr/bin/env python3
"""Data plots for Lecture 1 (Introduction).

Every value is sourced:
  - Google Flu Trends (GFT) against CDC: weekly US national series from the
    Delphi Epidata API (Carnegie Mellon), which archives the published GFT
    estimate (ILI per 100,000 doctor visits) and CDC ILINet (weighted percent
    of doctor visits for influenza-like illness). Files in data/, retrieved
    2026-10-16. The chapter's war story (Sec. 1.2) cites Lazer et al. 2014:
    roughly twice the CDC level in 2012-13, too high in 100 of 108 weeks
    from August 2011.
  - GPU-days: AlexNet 6 days on 2 GPUs; GPT-3 about 405 V100 GPU-years
    (Patterson et al. 2021); GPT-4-class about 2.5 million reference
    GPU-days, an estimate (Patel and Wong 2023). All from the chapter
    (Sec. 1.3.3 and Table 3).
  - Iron-law bars: MLSysIM SingleNodeModel on the NVIDIA H100 (FP16,
    utilization 0.5). ResNet-50 at batch 64, GPT-2 XL decoding at batch 1.
    The memory term counts weight bytes only. Doubling bandwidth halves the
    memory term and leaves the other two unchanged.
  - Amdahl pipeline: 60 ms preprocessing, 45 ms -> 15 ms model, 25 ms
    postprocessing; 130 ms -> 100 ms (Introduction, Sec. 1.11).

Run from this folder with the MLSysBook venv (it imports mlsysim):
  python3 make_figs.py
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

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


gft = {r["epiweek"]: r["num"] / 1000.0
       for r in json.load(open(out / "data" / "gft_nat.json"))["rows"]}
cdc = {r["epiweek"]: r["wili"]
       for r in json.load(open(out / "data" / "cdc_ilinet_nat.json"))["rows"]}
weeks = sorted(w for w in gft if w in cdc and 201027 <= w <= 201426)
xs = [epiweek_to_date(w) for w in weeks]


def gft_plot(path, cut=None, annotate=False, size=(5.8, 2.6)):
    fig, ax = sp.plt.subplots(figsize=size)
    ax.plot(xs, [cdc[w] for w in weeks], color=C["neutral"], lw=3)
    ax.text(epiweek_to_date(201432), cdc[weeks[-1]] + 0.2, "CDC\n(doctor visits)",
            color=C["neutral"], fontsize=11, fontweight="bold", va="bottom")
    shown = [w for w in weeks if cut is None or w <= cut]
    ax.plot([epiweek_to_date(w) for w in shown], [gft[w] for w in shown],
            color=C["danger"], lw=3)
    ax.text(epiweek_to_date(201027), 9.6, "Google Flu Trends\n(from searches)",
            color=C["danger"], fontsize=11, fontweight="bold", va="top")
    if annotate:
        pk = max((w for w in weeks if 201240 <= w <= 201320), key=lambda w: gft[w])
        ck = max((w for w in weeks if 201240 <= w <= 201320), key=lambda w: cdc[w])
        ax.annotate(f"{gft[pk]:.1f}%", (epiweek_to_date(pk), gft[pk]),
                    xytext=(8, -2), textcoords="offset points", color=C["danger"],
                    fontsize=12, fontweight="bold", va="center")
        ax.annotate(f"{cdc[ck]:.1f}%", (epiweek_to_date(ck), cdc[ck]),
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

# Seasonal ratio of the GFT estimate to the CDC count (sum over weeks 40-20)
seasons = []
for y in range(2003, 2013):
    ws = [w for w in gft if w in cdc and ((w // 100 == y and w % 100 >= 40)
                                          or (w // 100 == y + 1 and w % 100 <= 20))]
    seasons.append((f"{y % 100:02d}-{(y + 1) % 100:02d}",
                    sum(gft[w] for w in ws) / sum(cdc[w] for w in ws)))
fig, ax = sp.plt.subplots(figsize=(5.8, 2.4))
for i, (lab, r) in enumerate(seasons):
    ck = "danger" if r > 1.2 else "neutral"
    ax.bar(i, r, color=C[ck], width=0.66)
    if r > 1.2:
        ax.text(i, r + 0.04, f"{r:.2f}", ha="center", va="bottom", color=C[ck],
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
    ("AlexNet (2012)", 12, "neutral", "12 GPU-days"),
    ("GPT-3 (2020)", 405 * 365, "compute", "about 148,000"),
    ("GPT-4-class (2023)", 2.5e6, "danger", "about 2.5 million (est.)"),
], "GPU-days", xlabel="Accelerator-days to train (log scale)", value_fmt="{:,.0f}")


# ---------------------------------------------------------------------------
# Iron-law bars from MLSysIM: ResNet-50 (batch 64) against GPT-2 XL decode
# (batch 1) on one H100, then the same with twice the memory bandwidth
# ---------------------------------------------------------------------------
def mlsysim_terms():
    root = Path(__file__).resolve().parents[4] / "mlsysim"   # the repo's MLSysIM
    sys.path.insert(0, str(root))
    from mlsysim import Hardware, Models
    from mlsysim.solvers import SingleNodeModel
    h = Hardware.Cloud.H100
    sn = SingleNodeModel()
    rows = {}
    for key, m, b in (("ResNet-50, batch 64", Models.Vision.ResNet50, 64),
                      ("GPT-2 XL decode, batch 1", Models.Language.GPT2, 1)):
        p = sn.solve(m, h, batch_size=b, precision="fp16", efficiency=0.5)
        rows[key] = tuple(float(q.to("millisecond").magnitude) for q in
                          (p.latency_memory, p.latency_compute, p.latency_overhead))
    return rows


try:
    TERMS = mlsysim_terms()
except Exception as e:  # keep the last computed values if mlsysim is absent
    print("mlsysim unavailable, using recorded values:", e)
    TERMS = {"ResNet-50, batch 64": (0.113, 1.061, 0.510),
             "GPT-2 XL decode, batch 1": (0.985, 0.006, 0.490)}
print("iron-law terms (ms):", TERMS)


def iron_bars(path, show_double):
    fig, ax = sp.plt.subplots(figsize=(5.8, 2.7))
    rows = []
    for name, (m, c, o) in TERMS.items():
        rows.append((name, m, c, o, False))
        rows.append(("  with 2x bandwidth", m / 2, c, o, True))
    n = len(rows)
    for i, (name, m, c, o, dbl) in enumerate(rows):
        y = n - 1 - i
        vis = (not dbl) or show_double
        a = 1.0 if vis else 0.0
        left = 0
        for v, ck in ((m, "memory"), (c, "compute"), (o, "neutral")):
            ax.barh(y, v, left=left, color=C[ck if not dbl else ck + "_l" if ck != "neutral" else "neutral_l"],
                    edgecolor="white", height=0.62, alpha=a)
            left += v
        ax.text(-0.02, y, name.strip() if not dbl else "2x bandwidth",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=12 if not dbl else 11, color=C["text"] if vis else "white",
                fontweight="bold" if not dbl else "normal")
        if vis:
            total = m + c + o
            lab = f"{total:.2f} ms"
            if dbl:
                base = (2 * m) + c + o
                lab += f"  ({(total - base) / base * 100:+.0f}%)"
            ax.text(left + 0.03, y, lab, va="center", fontsize=12,
                    color=C["text"], fontweight="bold")
    # term labels inside the first two bars
    rm, rc, ro = TERMS["ResNet-50, batch 64"]
    ax.text(rm + rc / 2, n - 1, "compute", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    ax.text(rm + rc + ro / 2, n - 1, "overhead", ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    gm, gc, go = TERMS["GPT-2 XL decode, batch 1"]
    ax.text(gm / 2, n - 3, "memory", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    ax.text(gm + gc + go / 2, n - 3, "overhead", ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Time per step on one H100 (ms)")
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)


iron_bars(out / "ironlaw_bars_step1.svg", False)
iron_bars(out / "ironlaw_bars_step2.svg", True)


# ---------------------------------------------------------------------------
# Amdahl: a 3x faster model in a 130 ms request path
# ---------------------------------------------------------------------------
def amdahl(path, show_after):
    fig, ax = sp.plt.subplots(figsize=(5.8, 1.9))
    rows = [("Before", (60, 45, 25)), ("After", (60, 15, 25))]
    for i, (name, segs) in enumerate(rows):
        y = 1 - i
        vis = i == 0 or show_after
        left = 0
        for v, ck, lab in zip(segs, ("neutral", "compute", "neutral"),
                              ("preprocess 60", "model", "post 25")):
            ax.barh(y, v, left=left, color=C[ck] if ck == "compute" else C["neutral_l"],
                    edgecolor="white", linewidth=2, height=0.6, alpha=1 if vis else 0)
            if vis:
                txt = (f"model {v}" if v > 20 else str(v)) if ck == "compute" else lab
                ax.text(left + v / 2, y, txt, ha="center", va="center", fontsize=11,
                        fontweight="bold",
                        color="white" if ck == "compute" else C["text"])
            left += v
        ax.text(-2, y, name, ha="right", va="center", fontsize=13, fontweight="bold",
                color=C["text"] if vis else "white")
        if vis:
            ax.text(left + 2, y, f"{left} ms" + ("   (-23%)" if i == 1 else ""),
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
    from mlsysim import Literature
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
        lab = "EfficientNet: 44x" if name == "EfficientNet-B0" else name
        off = {"AlexNet": (0, 10), "MobileNet-v2": (8, -6), "EfficientNet-B0": (-6, 0)}[name]
        ax.annotate(lab, (yr, f), xytext=off, textcoords="offset points",
                    ha={"EfficientNet-B0": "right", "MobileNet-v2": "left"}.get(name, "center"),
                    va="bottom" if name == "AlexNet" else "center",
                    fontsize=11, fontweight="bold",
                    color=C["compute"] if name == "EfficientNet-B0" else C["text"])
ax.set_xlim(2012, 2020)
ax.set_ylim(0, 50)
ax.set_xticks([2012, 2014, 2016, 2018, 2020])
ax.set_ylabel("Less compute than AlexNet (x)")
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
