#!/usr/bin/env python3
"""Figures for Lecture 19 (Agent Economics), Volume III.

Every value comes from calcs.py, which this script imports, so figures and
slides agree. "AE" is the rendered chapter, Agent Economics; "NM 19.k" its
Napkin Math notebooks. The book's margin SVGs (rejected per accepted, queue
wait, break-even) and its full-page figures (critical-path waterfall,
cascade flowchart, hierarchical budget) are page-sized and fall below the
legibility floor on a slide, so each is redrawn here at the size of its slot:
  - multiplier: AE vol3_economics_rejected_per_accepted_margin.svg
  - kingman_step*: AE vol3_economics_queue_wait_margin.svg
  - breakeven: AE vol3_economics_breakeven_margin.svg
  - waterfall_step*: AE fig-vol3-critical-path-waterfall.svg
  - cascade_step*: AE fig-vol3-model-cascade-flowchart
  - budget_step*: AE fig-vol3-hierarchical-budget
Real-data plots: price_trend (Epoch AI 2025), swe_cohort (SWE-bench
trajectories), frontierswe (FrontierSWE V2),
inferencex (SemiAnalysis InferenceX). Full-width plots are drawn at about
5.8 x 2.5 in with 11 to 13 pt type; column plots at 5.4 x 3.6 in with 15 pt.

Run from this folder: python3 make_figs.py   (then make deck converts SVG)
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import slideplot as sp  # noqa: E402
import calcs  # noqa: E402

C = sp.C
plt = sp.plt
plt.rcParams["text.parse_math"] = False   # dollar signs are text
out = Path(__file__).parent

# Volume III agentic palette (volumes/vol3/palette.md): fill, stroke, text.
P = {
    "store": ("#f3e8ff", "#7c3aed", "#6d28d9"),
    "tool": ("#fef3c7", "#d97706", "#b45309"),
    "fault": ("#fee2e2", "#dc2626", "#b91c1c"),
    "ok": ("#dcfce7", "#16a34a", "#15803d"),
    "think": ("#e0f2fe", "#0284c7", "#0369a1"),
    "gray": ("#f1f5f9", "#94a3b8", "#475569"),
}
VIO, RED, GRN, AMB, SKY, GRY, INK = "#7c3aed", "#dc2626", "#16a34a", "#d97706", "#0284c7", "#64748b", "#0f172a"
FONT = "Helvetica, Arial, sans-serif"
COL = (5.4, 3.6)      # column figure
WIDE = (5.8, 2.5)     # full-width figure


def money(v, nd=2):
    return f"${v:,.{nd}f}"


def tidy(ax):
    ax.grid(axis="both", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


# =================================================================== hook
h = calcs.nm191()


def hook(step):
    """Hook reveal: cost per accepted repair, frontier against low price."""
    fig, ax = plt.subplots(figsize=COL)
    rows = [("Frontier", h["frontier"]["per_acc"], P["think"][1]),
            ("Low token\nprice", h["lowprice"]["per_acc"], RED)]
    for i, (lab, v, col) in enumerate(rows):
        shown = step >= 2
        ax.bar(i, v if shown else 0, width=0.6, color=col)
        ax.text(i, (v if shown else 0) + 0.05, money(v) if shown else "?", ha="center", va="bottom",
                fontsize=20, fontweight="bold", color=col if shown else C["neutral"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r[0] for r in rows], fontsize=16, fontweight="bold")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.75)
    ax.set_ylabel("Dollars per accepted repair", fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    if step >= 2:
        ax.text(0.5, 1.6, f"{h['ratio']:.1f}× more per accepted repair", ha="center", fontsize=16,
                color=RED, fontweight="bold")
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"hook_step{step}.svg")


for k in (1, 2):
    hook(k)


# ======================================================== attempt bill
def bill(step):
    """Per-attempt bill by term (NM 19.1): model tokens, environment, assessment."""
    fig, ax = plt.subplots(figsize=WIDE)
    names = [("frontier", "Frontier\n4 turns"), ("lowprice", "Low token price\n7 turns")]
    for i, (k, lab) in enumerate(names):
        x = h[k]
        y = 1 - i
        parts = [("model tokens", x["model"], SKY), ("environment", x["env"], AMB),
                 ("assessment", x["assess"], GRY)]
        left = 0
        for j, (pn, v, col) in enumerate(parts):
            show = step >= 2 or j == 0
            ax.barh(y, v, left=left, height=0.58, color=col, alpha=1 if show else 0)
            if show and v > 0.08:
                ax.text(left + v / 2, y, money(v, 3), ha="center", va="center", fontsize=12,
                        color="white", fontweight="bold")
            elif show and j == 0:
                ax.text(left + v / 2, y + 0.42, money(v, 3), ha="center", va="center", fontsize=12,
                        color=SKY, fontweight="bold")
            left += v
        ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=12, fontweight="bold", color=C["text"])
        if step >= 2:
            ax.text(left + 0.008, y, f"{money(x['attempt'], 3)} per attempt", va="center",
                    fontsize=12, fontweight="bold", color=C["text"])
    # legend as direct labels on the top bar
    if step >= 1:
        ax.text(0.0, 1.42, "model tokens", color=SKY, fontsize=12, fontweight="bold")
    if step >= 2:
        ax.text(0.22, 1.42, "environment", color=AMB, fontsize=12, fontweight="bold")
        ax.text(0.44, 1.42, "assessment", color=GRY, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 0.66)
    ax.set_ylim(-0.55, 1.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Dollars per attempt (NM 19.1, illustrative)", fontsize=11)
    ax.tick_params(labelsize=11)
    sp.save(fig, out / f"bill_step{step}.svg")


for k in (1, 2):
    bill(k)


# ======================================================== price trend (real)
pt = calcs.price_trend()


def price_trend():
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    end = dt.datetime(2025, 3, 1)
    style = [(("MMLU", "GPT-3"), "GPT-3-level knowledge (MMLU)", C["neutral"]),
             (("MMLU", "GPT-4-0314"), "GPT-4-level knowledge (MMLU)", C["memory"]),
             (("GPQA Diamond", "GPT-4-0314"), "GPT-4-level science (GPQA)", VIO)]
    ylab = {("MMLU", "GPT-3"): 8, ("MMLU", "GPT-4-0314"): 70, ("GPQA Diamond", "GPT-4-0314"): 24}
    short = {("MMLU", "GPT-3"): "GPT-3 level, MMLU", ("MMLU", "GPT-4-0314"): "GPT-4 level, MMLU",
             ("GPQA Diamond", "GPT-4-0314"): "GPT-4 level, GPQA"}
    for key, lab, col in style:
        s = pt["chosen"][key]
        d = [dt.datetime.fromisoformat(p["release_date"]) for p in s["pts"]]
        v = [float(p["usd_per_mtok"]) for p in s["pts"]]
        ax.step(d + [end], v + [v[-1]], where="post", color=col, lw=2.4)
        ax.scatter(d, v, color=col, s=26, zorder=4)
        ax.text(dt.datetime(2024, 6, 15), ylab[key], f"{short[key]}: {s['rate']:.0f}× per year",
                color=col, fontsize=11, fontweight="bold", va="center")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 150)
    ax.set_xlim(dt.datetime(2021, 9, 1), dt.datetime(2025, 3, 15))
    ax.set_ylim(0.03, 150)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_ylabel("Cheapest $ per M tokens", fontsize=11)
    ax.tick_params(labelsize=11)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    tidy(ax)
    sp.save(fig, out / "price_trend.svg")


price_trend()


# ======================================================== SWE-bench cohorts (real)
sw = calcs.swe_cohorts()


def swe_cohort(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    import numpy as np
    x = np.arange(len(sw))
    w = 0.4
    a = [c["mean_resolved"] for c in sw]
    b = [c["per_resolved"] for c in sw]
    ax.bar(x - w / 2, a, w, color=C["neutral_l"], edgecolor=C["neutral"], lw=1.2)
    for i, v in enumerate(a):
        ax.text(x[i] - w / 2, v + 0.08, money(v), ha="center", fontsize=12, color=C["neutral"],
                fontweight="bold")
    if step >= 2:
        ax.bar(x + w / 2, b, w, color=VIO)
        for i, v in enumerate(b):
            ax.text(x[i] + w / 2, v + 0.08, money(v), ha="center", fontsize=12, color=VIO,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c['label'].replace(' run ', ' ')}\n{c['resolved']}/{c['n']}" for c in sw], fontsize=13)
    ax.set_ylabel("Dollars", fontsize=13)
    ax.set_ylim(0, 6.2)
    ax.tick_params(axis="y", labelsize=13)
    ax.text(1.0, 5.6, "mean cost of a resolved run", ha="left", fontsize=14, color=C["neutral"],
            fontweight="bold")
    if step >= 2:
        ax.text(1.0, 4.9, "cohort cost per resolved task", ha="left", fontsize=14, color=VIO,
                fontweight="bold")
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"swe_step{step}.svg")


for k in (1, 2):
    swe_cohort(k)


# ======================================================== rejection multiplier
def multiplier():
    import numpy as np
    fig, ax = plt.subplots(figsize=COL)
    a = np.linspace(0.12, 0.98, 200)
    ax.plot(a * 100, (1 - a) / a, color=VIO, lw=3)
    for acc, col in ((0.25, RED), (0.75, GRN)):
        m = (1 - acc) / acc
        ax.plot([acc * 100], [m], "o", ms=12, color=col, mec="white", mew=1.5, zorder=5)
        ax.annotate(f"{m:.2f}" if m < 1 else f"{m:.0f}", (acc * 100, m), xytext=(10, 10),
                    textcoords="offset points", fontsize=17, fontweight="bold", color=col)
    ax.text(30, 4.3, "low price: 25 of 100 accepted", fontsize=14, color=RED, va="center")
    ax.text(58, 1.9, "frontier: 75 of 100", fontsize=14, color=GRN, va="center")
    ax.set_xlabel("Acceptance rate α (percent)", fontsize=15)
    ax.set_ylabel("Rejected per accepted", fontsize=15)
    ax.set_ylim(0, 7.5)
    ax.set_xlim(10, 100)
    ax.tick_params(labelsize=14)
    tidy(ax)
    sp.save(fig, out / "multiplier.svg")


multiplier()


# ======================================================== resend per turn
r2 = calcs.nm192()


def resend(step):
    import numpy as np
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    t = np.arange(1, 31)
    ax.bar(t - 0.2, r2["per_turn_no"], 0.4, color=C["neutral_l"], edgecolor=C["neutral"], lw=0.8)
    if step >= 2:
        ax.bar(t + 0.2, r2["per_turn_re"], 0.4, color=SKY)
    ax.text(1, 0.15, f"resend all, fresh price: {money(r2['total_no'], 3)}", fontsize=12.5,
            color=C["neutral"], fontweight="bold")
    if step >= 2:
        ax.text(1, 0.12, f"cached prefix: {money(r2['total_re'], 3)} ({r2['ratio']:.1f}× less)",
                fontsize=12.5, color=SKY, fontweight="bold")
    ax.set_xlabel("Turn of a 30-turn trajectory (NM 19.2, illustrative)", fontsize=11)
    ax.set_ylabel("Dollars per call", fontsize=11)
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 0.17)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"resend_step{step}.svg")


for k in (1, 2):
    resend(k)


# ======================================================== compaction payback
def payback():
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    for frac, col, lab in ((0.10, SKY, "cached read at 10%"),
                           (0.05, AMB, "cached read at 5%")):
        c = calcs.nm193(frac)
        n = list(range(1, 31))
        ax.plot(n, [c["diff"][i] * 100 for i in n], color=col, lw=2.8)
        ax.plot([c["nstar"]], [c["diff"][c["nstar"]] * 100], "o", ms=9, color=col, mec="white", zorder=5)
        ax.text(0.0, 10.4 if frac == 0.10 else 8.0, f"{lab}: payback {c['nstar']} calls",
                fontsize=14, color=col, fontweight="bold", clip_on=False)
    ax.axhline(0, color=INK, lw=1.2)
    ax.plot([15, 15], [-3.5, 6.5], color=RED, lw=1.6, ls="--")
    ax.text(15.3, 5.4, "15 calls remain", color=RED, fontsize=14, fontweight="bold")
    ax.set_xlabel("Calls remaining after compaction (NM 19.3, illustrative)", fontsize=13)
    ax.set_ylabel("Compact minus keep (¢)", fontsize=13)
    ax.set_xlim(0, 30)
    ax.set_ylim(-11, 6.5)
    ax.tick_params(labelsize=13)
    tidy(ax)
    sp.save(fig, out / "payback.svg")


payback()


# ======================================================== waterfall
c5 = calcs.nm195()


def waterfall(step):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    floor = c5["tests"] + c5["sandbox"] + c5["runtime"]
    rows = [("Baseline", c5["model"], f"{c5['T']:.1f} s"),
            ("Model 4× faster", c5["model"] / 4, f"{c5['t4']:.1f} s, {c5['sp4']:.2f}×"),
            ("Speculative decoding", c5["model"] / c5["s_hi"], f"{c5['t_hi']:.1f} s, {c5['sp_hi']:.2f}×"),
            ("Instant model calls", 0.0, f"{c5['t_inf']:.1f} s, {c5['sp_inf']:.2f}×")]
    for i, (lab, m, txt) in enumerate(rows):
        y = 3 - i
        show = i < step
        a = 1 if show else 0
        ax.barh(y, c5["tests"], height=0.6, color=AMB, alpha=a)
        ax.barh(y, c5["sandbox"], left=c5["tests"], height=0.6, color=P["tool"][0], edgecolor=AMB,
                hatch="///", lw=0.8, alpha=a)
        ax.barh(y, m, left=floor, height=0.6, color=SKY, alpha=a)
        ax.text(-4, y, lab, ha="right", va="center", fontsize=14, fontweight="bold", color=C["text"])
        if show:
            ax.text(floor + m + 3, y, txt, va="center", fontsize=14, fontweight="bold", color=C["text"])
    ax.text(c5["tests"] / 2, 3.47, "required tests", ha="center", fontsize=14, color=P["tool"][2],
            fontweight="bold")
    ax.text(c5["tests"] + c5["sandbox"] / 2, 3.47, "sandbox", ha="center", fontsize=14,
            color=P["tool"][2], fontweight="bold")
    ax.text(floor + 32, 3.47, "model", ha="left", fontsize=14, color=SKY, fontweight="bold")
    ax.axvline(floor, color=RED, ls="--", lw=1.8)
    ax.set_xlim(0, 430)
    ax.set_ylim(-0.5, 3.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Elapsed seconds, one serial episode (NM 19.5, illustrative)", fontsize=13)
    ax.tick_params(labelsize=11)
    sp.save(fig, out / f"waterfall_step{step}.svg")


for k in range(1, 5):
    waterfall(k)


# ======================================================== FrontierSWE (real)
fs = calcs.frontierswe()


def frontierswe():
    fig, ax = plt.subplots(figsize=(5.8, 2.75))
    for p in fs["pts"]:
        ax.plot([p["usd"], p["usd"]], [p["worst"], p["best"]], color=C["memory_l"], lw=4, solid_capstyle="butt")
        ax.plot([p["usd"]], [p["mean"]], "o", ms=7, color=C["memory"], zorder=4)
    top = fs["pts"][0]
    ax.annotate(f"top score: {top['usd']:,.0f} dollars per trial", (top["usd"], top["mean"]),
                xytext=(-12, 22), textcoords="offset points", ha="right", fontsize=11.5,
                color=C["memory"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C["memory"], lw=1))
    ax.set_xscale("log")
    ax.set_xlim(6, 2000)
    ax.set_ylim(0, 80)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_xlabel("Average cost of one trial (log scale)", fontsize=11)
    ax.set_ylabel("Score (%), worst to best", fontsize=11)
    ax.tick_params(labelsize=11)
    tidy(ax)
    sp.save(fig, out / "frontierswe.svg")


frontierswe()


# ======================================================== cascade (SVG)
c6 = calcs.nm196()


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, s, show=True):
        self.items.append(s if show else f'<g opacity="0">{s}</g>')

    def text(self, x, y, s, size=19, color=INK, bold=False, anchor="middle", show=True):
        w = ' font-weight="bold"' if bold else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" '
                 f'text-anchor="{anchor}"{w}>{s}</text>', show)

    def rect(self, x, y, w, h, fill, stroke="none", sw=2, rx=4, show=True, dash=False):
        d = ' stroke-dasharray="7,5"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>', show)

    def line(self, x1, y1, x2, y2, color=GRY, width=2.4, show=True, dash=False):
        d = ' stroke-dasharray="7,5"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>', show)

    def save(self, name):
        (out / name).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}">\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n'
            + "\n".join(self.items) + "\n</svg>\n")


def cascade(step):
    s = Svg(900, 430)
    x0, wmax = 250, 260
    tiers = [("Tier 1: small", "$0.0004 + $0.0010 gate"),
             ("Tier 2: mid", "$0.0020 + $0.0010 gate"),
             ("Tier 3: frontier", "$0.0600 + $0.0010 gate")]
    for i, (name, price) in enumerate(tiers):
        y = 14 + i * 112
        reach, acc = c6["reached"][i], c6["accepted"][i]
        s.text(8, y + 36, name, 24, INK, bold=True, anchor="start")
        s.text(8, y + 68, price, 20, "#334155", anchor="start")
        wr = wmax * reach / 100000
        wa = wmax * acc / 100000
        s.rect(x0, y + 10, wa, 66, P["ok"][0], P["ok"][1])
        s.rect(x0 + wa, y + 10, wr - wa, 66, P["fault"][0], P["fault"][1])
        if wa > 66:
            s.text(x0 + wa / 2, y + 51, f"{acc:,}", 21, P["ok"][2], bold=True)
        if wr - wa > 80:
            s.text(x0 + wa + (wr - wa) / 2, y + 51, f"{reach - acc:,}", 21, P["fault"][2], bold=True)
        s.text(x0 + wr + 10, y + 51, f"of {reach:,}", 20, "#334155", anchor="start")
        s.line(x0 - 8, y + 4, x0 - 8, y + 82, VIO, 5)
    s.text(8, 378, "green: accepted and exits;  red: rejected and escalates", 20, "#334155", anchor="start")
    if step >= 2:
        s.rect(670, 14, 222, 320, "#f8fafc", "#cbd5e1", rx=6)
        s.text(781, 52, "Per accepted", 22, INK, bold=True)
        s.text(781, 80, "result", 22, INK, bold=True)
        s.text(781, 128, "cascade", 21, VIO, bold=True)
        s.text(781, 160, money(c6["per_acc"], 5), 25, VIO, bold=True)
        s.text(781, 212, "direct frontier", 21, RED, bold=True)
        s.text(781, 244, money(c6["direct_per_acc"], 5), 25, RED, bold=True)
        s.text(781, 300, f"{c6['ratio']:.1f}× apart", 23, INK, bold=True)
    if step >= 3:
        s.text(8, 416, f"Time: mean {c6['mean_t']:.2f} s; through all tiers {c6['worst_t']:.2f} s; "
               f"direct {c6['direct_t']:.2f} s", 21, RED, bold=True, anchor="start")
    s.save(f"cascade_step{step}.svg")


for k in (1, 2, 3):
    cascade(k)


# ======================================================== InferenceX (real)
ix = calcs.inferencex()


def inferencex():
    fig, ax = plt.subplots(figsize=COL)
    xs = [p[1] for p in ix["pts"]]
    ys = [p[2] for p in ix["pts"]]
    ax.plot(xs, ys, color=SKY, lw=3, marker="o", ms=10)
    for c, u, g in ix["pts"]:
        ax.annotate(f"{c} users", (u, g), xytext=(10, -18), textcoords="offset points", fontsize=14,
                    color=P["think"][2], fontweight="bold")
    ax.set_xlabel("Tokens/s each user sees", fontsize=15)
    ax.set_ylabel("Tokens/s per GPU", fontsize=15)
    ax.set_ylim(0, 290)
    ax.set_xlim(125, 15)
    ax.tick_params(labelsize=14)
    tidy(ax)
    sp.save(fig, out / "inferencex.svg")


inferencex()


# ======================================================== Kingman
q = calcs.nm198()


def kingman(step):
    fig, ax = plt.subplots(figsize=COL)
    ax.plot([u * 100 for u, _ in q["curve"]], [w for _, w in q["curve"]], color=VIO, lw=3)
    if step >= 2:
        for u, w, col in ((0.7, q["w70"], SKY), (0.9, q["w90"], RED)):
            ax.plot([u * 100], [w], "o", ms=13, color=col, mec="white", mew=1.5, zorder=5)
            ax.annotate(f"{w:.0f} s", (u * 100, w), xytext=(-56, 4), textcoords="offset points",
                        fontsize=20, fontweight="bold", color=col)
    ax.set_xlabel("Utilization ρ (percent)", fontsize=18)
    ax.set_ylabel("Mean wait (s)", fontsize=18)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.tick_params(labelsize=17)
    tidy(ax)
    sp.save(fig, out / f"kingman_step{step}.svg")


for k in (1, 2):
    kingman(k)


# ======================================================== break-even
d9 = calcs.nm199()
d0 = calcs.nm199(0.0)


def breakeven():
    import numpy as np
    fig, ax = plt.subplots(figsize=COL)
    u = np.linspace(0, 1, 100)
    api = [d9["line"](x)[0] for x in u]
    ded = [d9["line"](x)[1] for x in u]
    ax.plot(u * 100, api, color=AMB, lw=3)
    ax.plot(u * 100, ded, color=SKY, lw=3)
    ax.text(58, 76, "per-token API", color=AMB, fontsize=15, fontweight="bold")
    ax.text(72, 24, "dedicated", color=SKY, fontsize=15, fontweight="bold")
    us = d9["ustar"] * 100
    ax.axvline(us, color=GRY, ls="--", lw=1.6)
    ax.plot([us], [d9["line"](d9["ustar"])[0]], "o", ms=12, color=VIO, mec="white", zorder=5)
    ax.text(us - 2, 76, f"u* = {us:.1f}%", color=VIO, fontsize=15, fontweight="bold", ha="right")
    ax.plot([52], [d9["api"]], "o", ms=10, color=AMB, mec="white", zorder=5)
    ax.plot([52], [d9["ded"]], "o", ms=10, color=SKY, mec="white", zorder=5)
    ax.text(50, 50, money(d9["api"]), ha="right", color=AMB, fontsize=14, fontweight="bold")
    ax.text(54, d9["ded"] - 10, money(d9["ded"]), color=SKY, fontsize=14, fontweight="bold")
    ax.set_xlabel("Delivered load (% of reference rate)", fontsize=15)
    ax.set_ylabel("Dollars per hour", fontsize=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 90)
    ax.tick_params(labelsize=14)
    tidy(ax)
    sp.save(fig, out / "breakeven.svg")


breakeven()


# ======================================================== budget conservation
b = calcs.nm1910()


def budget(step):
    """Root balances at the five boundaries of AE table 5: settled (gray),
    reserved (amber), free (green); every bar sums to the $5 budget."""
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    labs = ["1 Planning settled", "2 Children granted", "3 Trace child closed",
            "4 Fuzzing stopped", "5 Late bill settled"]
    for i, (name, s_, r_, f_) in enumerate(b["table"]):
        y = 4 - i
        show = i < step
        a = 1 if show else 0
        ax.barh(y, s_, height=0.66, color=GRY, alpha=a)
        ax.barh(y, r_, left=s_, height=0.66, color=AMB, alpha=a)
        ax.barh(y, f_, left=s_ + r_, height=0.66, color=P["ok"][0], edgecolor=GRN, lw=1.2, alpha=a)
        ax.text(-0.06, y, labs[i], ha="right", va="center", fontsize=14, fontweight="bold",
                color=C["text"])
        if show:
            if s_ > 0.5:
                ax.text(s_ / 2, y, money(s_, 3), ha="center", va="center", fontsize=13.5,
                        color="white", fontweight="bold")
            if r_ > 0.4:
                ax.text(s_ + r_ / 2, y, money(r_), ha="center", va="center", fontsize=13.5,
                        color="white", fontweight="bold")
            ax.text(s_ + r_ + f_ / 2, y, money(f_, 3), ha="center", va="center", fontsize=13.5,
                    color=P["ok"][2], fontweight="bold")
    ax.text(0.0, 4.62, "settled", color=GRY, fontsize=14, fontweight="bold")
    ax.text(1.0, 4.62, "reserved", color=AMB, fontsize=14, fontweight="bold")
    ax.text(3.2, 4.62, "free", color=GRN, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, 5.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Root budget of $5 (NM 19.10, illustrative)", fontsize=13)
    ax.tick_params(labelsize=13)
    sp.save(fig, out / f"budget_step{step}.svg")


for k in range(1, 6):
    budget(k)


# ======================================================== architectures
a11 = calcs.nm1911()


def arch(step):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.3))
    names = list(a11["d"])
    short = ["Rewriter", "Single call", "3-turn loop"]
    cols = [GRY, AMB, VIO]
    for j, (ax, key, title, nd) in enumerate((
            (axes[0], "per_auto", "Automated spend per\nautomated acceptance", 3),
            (axes[1], "per_res", "Spend per resolved ticket,\nhuman fallback included", 2))):
        show = j == 0 or step >= 2
        vals = [a11["d"][n][key] for n in names]
        for i, v in enumerate(vals):
            y = 2 - i
            ax.barh(y, v if show else 0, height=0.6, color=cols[i])
            if show:
                ax.text(v + max(vals) * 0.03, y, money(v, nd), va="center", fontsize=11.5,
                        fontweight="bold", color=cols[i])
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(short if j == 0 else ["", "", ""], fontsize=11.5)
        ax.set_xlim(0, max(vals) * 1.55)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, fontsize=11.5, color=C["text"] if show else "white", loc="left")
    sp.save(fig, out / f"arch_step{step}.svg")


for k in (1, 2):
    arch(k)


# ======================================================== recall thumbnail
def thumb_recall():
    """Text-free cue for Lecture 18's token multiples (two bars, 4 and 15)."""
    s = Svg(240, 90)
    s.rect(20, 50, 52, 30, P["think"][1])
    s.rect(100, 10, 52, 70, VIO)
    s.line(10, 80, 230, 80, GRY, 2)
    s.save("thumb_recall.svg")


thumb_recall()
print("figures written to", out)
