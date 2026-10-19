#!/usr/bin/env python3
"""Figures for Lecture 14 (ML Operations).

Every value is sourced. "Chapter" means the rendered ML Operations chapter.

  - Zillow Offers, homes bought, sold, and in inventory: Zillow Group Q2 2021
    shareholder letter (3,805 bought, 2,086 sold, 3,142 in inventory) and Q3
    2021 shareholder letter (9,680 bought, 3,032 sold, 9,790 in inventory,
    8,172 under contract), SEC Form 8-K exhibits 99.3, August and November
    2021.
  - Wild-Time: chapter figure 1 (Yao et al. 2022): training-period against
    later-period scores, Yearbook 98.0 to 79.5, MIMIC mortality AUC 90.9 to
    72.9, MIMIC readmission 73.0 to 61.3, HuffPost 79.4 to 70.4, arXiv 53.8 to
    45.9, FMoW 58.1 to 54.1 (values from the figure's alt text).
  - Operations loop and its machine costs: chapter Sec. 14.1 and table 1.
  - Google outage causes: Papasian and Underwood, "How ML Breaks", USENIX
    OpML 2020 slides, "Finding #1" bar chart; outages by how characteristic of
    ML the cause was, 0 to 4: about 61, 0, 9, 2, 24 of 96 (read from chart).
  - Correction cascade: chapter figure 4; Zillow chain from the Q3 2021
    shareholder letter.
  - Automation break-even: Napkin Math 14.1 (80 h pipeline, 4 h a week of
    manual work, 20 weeks). The 1 h a week upkeep line is derived from the
    chapter's statement that each week recovers only the difference.
  - Parity replay: chapter Sec. 14.3.1 (45 against 12 minutes, 8 percent
    accuracy loss, the swapped-value counterexample), illustrative scenario.
  - Knight Capital: SEC Release No. 34-70694 (2013), paragraphs 15 to 19.
    "About 10:15" is the 9:30 open plus the order's "approximately 45
    minutes".
  - Node-day memory map: chapter figure 8 and table 11 (100 GB of tables on
    two H100s, four H100s for six hours of retraining, one free copy slot).
  - Rollback state: chapter Sec. 14.4.3 and Checkpoint 14.2 (embedding
    dimension change, cached state restored with the weights).
  - Derived values (Wild-Time drops, break-even weeks, node-day counts, A/B
    users per variant) come from calcs.py in this folder.
  - Part 2 figures (detection time, Rabanser, silent-failure cost, retraining
    cadence, freshness, T*, slices, iceberg) moved to Lecture 14b and are
    generated there.

Plots are drawn small (about 3 to 5 in wide) because the slide shows them at
about 5 cm tall; drawn at that size their 11 to 13 pt type stays at 8 pt or
more on the slide.

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

import numpy as np

import calcs

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

plt = sp.plt
out = Path(__file__).parent
C = sp.C
FONT = "Helvetica, Arial, sans-serif"


# -----------------------------------------------------------------------------
# SVG helpers (text kept as text)
# -----------------------------------------------------------------------------
def svg_open(w=900, h=500):
    marks = "".join(
        f'<marker id="ar{k}" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        for k, col in (("n", "#555555"), ("d", C["data"]), ("r", C["danger"]),
                       ("b", C["memory"]), ("l", "#BBBBBB"),
                       ("o", C["compute"])))
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">', f"<defs>{marks}</defs>",
            f'<rect width="{w}" height="{h}" fill="white"/>']


def text(x, y, s, size=24, color=None, weight="normal", anchor="middle",
         opacity=1.0):
    color = color or C["text"]
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'opacity="{opacity}">{s}</text>')


def rect(x, y, w, h, fill, stroke, sw=2.5, rx=8, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} '
            f'opacity="{opacity}"/>')


def line(x1, y1, x2, y2, color="#555555", w=3, marker="n", dash=None,
         opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#ar{marker})"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{w}"{d}{m} opacity="{opacity}"/>')


def path(d, color="#555555", w=3, marker="n", dash=None, fill="none"):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#ar{marker})"' if marker else ""
    return (f'<path d="{d}" stroke="{color}" stroke-width="{w}" '
            f'fill="{fill}"{ds}{m}/>')


def save_svg(parts, name):
    (out / name).write_text("\n".join(parts + ["</svg>"]) + "\n")


def note(ax, s, x=0.0, y=-0.30):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, color=C["neutral"],
            ha="left")


def small_hbars(path_, rows, size, xlim, ticks, ticklabels, xlabel, log=False,
                label_x=None, value_pad=None):
    """Two or three labeled horizontal bars drawn small for a column slot.
    rows: (label, value, color_key, printed value), top to bottom."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, (lab, v, ck, s_) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=C[ck], height=0.62)
        x_text = v * 1.25 if log else v + value_pad
        ax.text(x_text, y, s_, va="center", fontsize=12, color=C[ck],
                fontweight="bold")
        ax.text(label_x, y, lab + " ", ha="right", va="center", fontsize=12,
                fontweight="bold", color=C["text"])
    if log:
        ax.set_xscale("log")
        ax.minorticks_off()
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=11)
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=11)
    sp.save(fig, out / path_)


# -----------------------------------------------------------------------------
# 1. Zillow Offers, Q2 against Q3 2021 (grouped bars, two steps)
# -----------------------------------------------------------------------------
cats = ["Bought", "Sold", "In inventory"]
q2 = [calcs.Q2_BOUGHT, calcs.Q2_SOLD, calcs.Q2_INV]
q3 = [calcs.Q3_BOUGHT, calcs.Q3_SOLD, calcs.Q3_INV]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 2.1))
    w = 0.4
    a = 1.0 if k == 2 else 0.0
    for i, v in enumerate(q2):
        ax.bar(i - w / 2, v, w, color=C["neutral"])
        ax.text(i - w / 2, v + 250, f"{v:,}", ha="center", va="bottom",
                fontsize=10, color=C["neutral"], fontweight="bold")
    for i, v in enumerate(q3):
        ax.bar(i + w / 2, v, w, color=C["danger"], alpha=a)
        ax.text(i + w / 2, v + 250, f"{v:,}", ha="center", va="bottom",
                fontsize=10, color=C["danger"], fontweight="bold", alpha=a)
    ax.text(2 + w / 2 + 0.24, 6000, f"+{calcs.Q3_CONTRACT:,}\nunder\ncontract", ha="left",
            va="center", fontsize=11, color=C["danger"], alpha=a)
    ax.text(-0.5, 12300, "Q2 2021", color=C["neutral"], fontsize=12,
            fontweight="bold", ha="left")
    ax.text(0.45, 12300, "Q3 2021", color=C["danger"], fontsize=12,
            fontweight="bold", ha="left", alpha=a)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cats, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.6, 3.05)
    ax.set_ylim(0, 13200)
    ax.set_ylabel("Homes", fontsize=11)
    ax.set_yticks([0, 5000, 10000])
    ax.set_yticklabels(["0", "5,000", "10,000"], fontsize=11)
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"zillow_step{k}.svg")


# -----------------------------------------------------------------------------
# 2. Wild-Time: the same model on later data (dumbbells, two steps)
# -----------------------------------------------------------------------------
# training and later-period scores, plotted to one decimal as in the chapter
wild = [(n, calcs.r(a_, 1), calcs.r(b_, 1)) for n, a_, b_ in calcs.WILD]
drops = [calcs.f(d, 1) for _, d in calcs.WILD_DROPS]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    n = len(wild)
    for i, (name, a_, b_) in enumerate(wild):
        y = n - 1 - i
        hi = k == 1 or i in (0, n - 1)
        alpha = 1.0 if hi else 0.3
        ax.plot([b_, a_], [y, y], color=C["neutral_l"], lw=6, zorder=1,
                alpha=alpha, solid_capstyle="round")
        ax.plot([a_], [y], "o", ms=10, color=C["memory"], zorder=3, alpha=alpha)
        ax.plot([b_], [y], "o", ms=10, color=C["danger"], zorder=3, alpha=alpha)
        ax.text(a_ + 1.6, y, f"−{drops[i]}", va="center", fontsize=12,
                fontweight="bold", color=C["danger"], alpha=alpha)
        ax.text(41, y, name, va="center", ha="right", fontsize=12,
                color=C["text"], alpha=max(alpha, 0.6))
    ax.text(wild[0][1], n - 0.3, "training\nperiod", color=C["memory"], fontsize=11,
            ha="center", va="bottom", fontweight="bold")
    ax.text(wild[0][2], n - 0.3, "later\nperiod", color=C["danger"], fontsize=11,
            ha="center", va="bottom", fontweight="bold")
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(42, 108)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.tick_params(labelsize=11)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Score on held-out test data (percent)", fontsize=11)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"wildtime_step{k}.svg")


# -----------------------------------------------------------------------------
# 3. The operations loop: five steps in a ring, costs lit in three steps
# -----------------------------------------------------------------------------
loop = [  # name, cx, cy, cost line 1, cost line 2, where the cost sits
    ("Detect", 450, 62, "telemetry bytes per second", "", "right"),
    ("Diagnose", 735, 180, "replay over", "logged requests", "below"),
    ("Decide", 645, 372, "accelerator-hours", "for a retraining run", "below"),
    ("Release", 255, 372, "a second resident copy:", "100 GB of tables", "below"),
    ("Recover", 165, 180, "a known-good", "version kept warm", "below"),
]
for k in (1, 2, 3, 4):
    p = svg_open(900, 480)
    p.append('<ellipse cx="450" cy="225" rx="285" ry="160" fill="none" '
             'stroke="#BBBBBB" stroke-width="3" stroke-dasharray="8 6"/>')
    p.append(text(450, 222, "Ranking service", 27, C["neutral"], "bold"))
    p.append(text(450, 254, "(illustrative)", 24, C["neutral"]))
    # direction arrows on the ring, between the boxes
    for (x1, y1, x2, y2) in ((572, 84, 650, 122), (722, 300, 700, 338),
                             (535, 385, 370, 385), (195, 340, 175, 300),
                             (240, 120, 320, 84)):
        p.append(line(x1, y1, x2, y2, "#888888", 3.5, "n"))
    for i, (nm, cx, cy, c1, c2, where) in enumerate(loop):
        lit = (k >= 2 and i < 2) or (k >= 3 and i >= 2)
        later = k == 4 and i < 3
        if later:
            lit = False
        p.append(rect(cx - 95, cy - 30, 190, 60,
                      C["compute_l"] if lit else "#F4F4F4",
                      C["compute"] if lit else "#999999", 3, 10))
        p.append(text(cx, cy + 10, nm, 29, "#999999" if later else C["text"],
                      "bold"))
        if later:
            if where == "right":
                p.append(text(cx + 108, cy + 9, "Part 2", 26, C["neutral"],
                              "bold", "start"))
            else:
                p.append(text(cx, cy + 66, "Part 2", 26, C["neutral"], "bold"))
            continue
        if not lit:
            continue
        if where == "right":
            p.append(text(cx + 108, cy + 9, c1, 24, C["compute"], "bold", "start"))
        else:
            p.append(rect(cx - 125, cy + 38, 250, 64, "white", "white", 0, 4))
            p.append(text(cx, cy + 62, c1, 24, C["compute"], "bold"))
            p.append(text(cx, cy + 92, c2, 24, C["compute"], "bold"))
    save_svg(p, f"oploop_step{k}.svg")


# -----------------------------------------------------------------------------
# 4. Google outages by how characteristic of ML the cause was
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.2, 2.1))
vals = calcs.OUTAGES
cols = [C["neutral"]] * 4 + [C["compute"]]
for i, v in enumerate(vals):
    ax.bar(i, v, 0.62, color=cols[i])
    ax.text(i, v + 1.5, f"{v}", ha="center", va="bottom", fontsize=12,
            fontweight="bold", color=cols[i])
ax.set_xticks(range(5))
ax.set_xticklabels(["0\nnot ML", "1", "2", "3", "4\nall ML"], fontsize=11)
ax.set_ylim(0, 72)
ax.set_yticks([0, 20, 40, 60])
ax.tick_params(axis="y", labelsize=11)
ax.set_ylabel(f"Outages (of {sum(vals)})", fontsize=11)
ax.set_xlabel("How ML-specific the cause was", fontsize=11)
ax.grid(axis="y", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "outages.svg")


# -----------------------------------------------------------------------------
# 5. A correction cannot undo a purchase: generic cascade, then Zillow
# -----------------------------------------------------------------------------
def chain(p, y, boxes, color, fill):
    xs = [12 + i * 225 for i in range(len(boxes))]
    for i, (a_, b_) in enumerate(boxes):
        p.append(rect(xs[i], y, 196, 80, fill, color, 2.5, 8))
        p.append(text(xs[i] + 98, y + 34, a_, 26, C["text"], "bold"))
        p.append(text(xs[i] + 98, y + 64, b_, 23, color))
        if i < len(boxes) - 1:
            p.append(line(xs[i] + 198, y + 40, xs[i + 1] - 4, y + 40,
                          "#555555", 3, "n"))
    return xs


for k in (1, 2):
    p = svg_open(900, 450)
    p.append(text(12, 30, "Models that correct each other", 24, C["neutral"],
                  "bold", "start"))
    xs = chain(p, 92, [("Model A", "base prediction"), ("Model B", "corrects A"),
                       ("Model C", "corrects B"), ("Model D", "corrects C")],
               C["memory"], C["memory_l"])
    p.append(path(f"M {xs[0]+98} 92 C {xs[0]+98} 52, {xs[3]+98} 52, {xs[3]+98} 88",
                  C["danger"], 2.5, "r", "7 5"))
    p.append(text(888, 30, "change A: revalidate B, C, D", 23, C["danger"],
                  anchor="end"))
    if k == 2:
        p.append(text(12, 238, "Zillow Offers, Q3 2021: the chain runs through the world",
                      24, C["neutral"], "bold", "start"))
        xs = chain(p, 256, [("Forecast", "home prices"), ("Purchase", f"{calcs.Q3_BOUGHT:,} homes"),
                            ("Inventory", f"{calcs.Q3_INV:,} + {calcs.Q3_CONTRACT:,}"),
                            ("Write-down", f"${calcs.WRITE_DOWN_M} million")],
                   C["danger"], C["danger_l"])
        p.append(path(f"M {xs[0]+98} 338 C {xs[0]+98} 410, {xs[1]+98} 410, {xs[1]+98} 342",
                      C["neutral"], 2.5, "n", "7 5"))
        p.append(text(xs[1] + 140, 420, "revised forecast: homes already bought",
                      23, C["neutral"], anchor="start"))
    save_svg(p, f"cascade_step{k}.svg")


# -----------------------------------------------------------------------------
# 6. When automation pays (Napkin Math 14.1), two steps
# -----------------------------------------------------------------------------
weeks = np.linspace(0, 40, 200)
BUILD, MANUAL, UPKEEP = 80, 4, 1          # Napkin Math 14.1; upkeep is the deck's
BE = calcs.breakeven(BUILD, MANUAL)
BE_UP = calcs.breakeven(BUILD, MANUAL, UPKEEP)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.3, 2.0))
    ax.plot(weeks, MANUAL * weeks, color=C["danger"], lw=3)
    ax.plot(weeks, [BUILD] * len(weeks), color=C["memory"], lw=3)
    ax.fill_between(weeks[weeks >= BE], BUILD, MANUAL * weeks[weeks >= BE],
                    color=C["danger_l"])
    ax.plot([BE], [BUILD], "o", ms=10, color=C["text"], zorder=5)
    ax.text(BE + 1.5, 45, f"break-even:\n{BUILD} / {MANUAL} = {calcs.f(BE, 0)} weeks", fontsize=11,
            fontweight="bold", color=C["text"])
    ax.text(40.8, 160, "Manual:\n4 h a week", color=C["danger"], fontsize=11,
            fontweight="bold", va="center")
    ax.text(40.8, 72, "Pipeline:\n80 h once", color=C["memory"], fontsize=11,
            fontweight="bold", va="center")
    a = 1.0 if k == 2 else 0.0
    ax.plot(weeks, BUILD + UPKEEP * weeks, color=C["memory"], lw=2.5, ls="--", alpha=a)
    ax.plot([BE_UP], [BUILD + UPKEEP * BE_UP], "o", ms=9, color=C["memory"], alpha=a,
            zorder=5)
    ax.text(40.8, 118, f"+{UPKEEP} h a week", color=C["memory"], fontsize=11,
            va="center", alpha=a)
    ax.text(1, 135, f"with upkeep: {BUILD} / {MANUAL - UPKEEP}\n= {calcs.f(BE_UP, 1)} weeks (derived)", fontsize=11,
            color=C["memory"], alpha=a, fontweight="bold")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 170)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Weeks", fontsize=11)
    ax.set_ylabel("Cumulative hours", fontsize=11)
    ax.grid(color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"automation_step{k}.svg")


# -----------------------------------------------------------------------------
# 7. Request-keyed parity replay, three steps
# -----------------------------------------------------------------------------
for k in (1, 2, 3):
    p = svg_open(900, 415)
    p.append(rect(10, 120, 160, 90, "#F4F4F4", "#777777", 2.5, 8))
    p.append(text(90, 158, "Recorded", 25, C["text"], "bold"))
    p.append(text(90, 188, "request", 25, C["text"], "bold"))
    p.append(rect(245, 30, 270, 86, C["data_l"], C["data"], 2.5, 8))
    p.append(text(380, 66, "Training transform", 25, C["text"], "bold"))
    p.append(text(380, 98, "offline path", 23, C["data"]))
    p.append(rect(245, 214, 270, 86, C["memory_l"], C["memory"], 2.5, 8))
    p.append(text(380, 250, "Serving transform", 25, C["text"], "bold"))
    p.append(text(380, 282, "online path", 23, C["memory"]))
    p.append(line(172, 148, 241, 90, "#555555", 3))
    p.append(line(172, 182, 241, 240, "#555555", 3))
    p.append(line(517, 73, 556, 73, "#555555", 3))
    p.append(line(517, 257, 556, 257, "#555555", 3))
    p.append(text(564, 66, "user_session_length", 23, C["data"], "bold", "start"))
    p.append(text(564, 250, "user_session_length", 23, C["memory"], "bold", "start"))
    if k >= 2:
        p.append(text(564, 96, "mean 45 min (wall clock)", 23, C["data"], anchor="start"))
        p.append(text(564, 280, "mean 12 min (active time)", 23, C["memory"], anchor="start"))
        p.append(text(564, 158, "8% accuracy lost, no change", 23, C["danger"], "bold", "start"))
        p.append(text(564, 186, "to model code or weights", 23, C["danger"], "bold", "start"))
    if k >= 3:
        y0 = 345
        for s_, x in (("Request", 70), ("Training", 215), ("Serving", 345)):
            p.append(text(x, y0, s_, 23, C["text"], "bold"))
        for j, (r, a_, b_) in enumerate((("A", "12 min", "45 min"), ("B", "45 min", "12 min"))):
            yy = y0 + 30 * (j + 1)
            p.append(text(70, yy, r, 23))
            p.append(text(215, yy, a_, 23, C["data"]))
            p.append(text(345, yy, b_, 23, C["memory"]))
        p.append(text(450, y0 + 15, "Same histogram: a K-S test passes.", 23,
                      C["neutral"], anchor="start"))
        p.append(text(450, y0 + 47, "Keyed comparison flags A and B.", 23,
                      C["danger"], "bold", "start"))
    save_svg(p, f"parity_step{k}.svg")


# -----------------------------------------------------------------------------
# 8. Knight Capital: seven of eight servers updated (SEC 34-70694, para. 15-16)
# -----------------------------------------------------------------------------
p = svg_open(900, 175)
for i in range(calcs.SERVERS):
    x = 12 + i * 111
    bad = i >= calcs.UPDATED
    col, fill = (C["danger"], C["danger_l"]) if bad else (C["data"], C["data_l"])
    p.append(rect(x + 14, 8, 72, 92, fill, col, 3, 8))
    for j in range(3):
        p.append(line(x + 26, 32 + j * 22, x + 74, 32 + j * 22, col, 3, None))
    p.append(text(x + 50, 132, f"Server {i+1}", 24, C["text"], "bold"))
    if bad:
        p.append(text(894, 170, "Power Peg", 25, col, "bold", "end"))
# one label under the seven updated servers
x7 = 12 + (calcs.UPDATED - 1) * 111
p.append(line(30, 146, x7 + 80, 146, C["data"], 2.5, None))
p.append(text((30 + x7 + 80) / 2, 170, f"new code on {calcs.UPDATED} of {calcs.SERVERS}",
              25, C["data"], "bold"))
save_svg(p, "knight_servers.svg")


# Knight timeline, two steps
def ktime(k):
    p = svg_open(900, 250)
    p.append(line(20, 130, 880, 130, "#999999", 4, None))
    marks = [(130, "8:01", "first automated e-mail:", "“Power Peg disabled”"),
             (440, "9:30", "market opens;", "97 e-mails by now"),
             (770, "about " + f"{(calcs.OPEN_MIN + calcs.KNIGHT_MIN) // 60}:"
              f"{(calcs.OPEN_MIN + calcs.KNIGHT_MIN) % 60:02d}",
              f"about {calcs.KNIGHT_MIN} minutes", "after the open")]
    for i, (x, t, a_, b_) in enumerate(marks):
        if i == 2 and k < 2:
            continue
        col = C["danger"] if i == 2 else C["text"]
        p.append(f'<circle cx="{x}" cy="130" r="11" fill="{col}"/>')
        p.append(text(x, 104, t, 27, col, "bold"))
        p.append(text(x, 172, a_, 23, C["neutral"]))
        p.append(text(x, 200, b_, 23, C["neutral"]))
    if k >= 2:
        p.append(rect(520, 8, 370, 54, C["danger_l"], C["danger"], 2, 8))
        p.append(text(705, 44, "212 orders: 4 million executions", 23,
                      C["danger"], "bold"))
        p.append(text(450, 240, "154 stocks, 397 million shares, $460 million lost",
                      25, C["danger"], "bold"))
    save_svg(p, f"knight_timeline_step{k}.svg")


ktime(1)
ktime(2)


# -----------------------------------------------------------------------------
# 9. One node over a day (redraw of chapter figure 8), two steps
# -----------------------------------------------------------------------------
def nodeday(k):
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    blocks = [  # y0, h, x0, w, fill, edge, label, step
        (0, 2, 0, 24, "memory_l", "memory", "Serving copy: 100 GB of tables", 1),
        (2, 2, 0, 24, "data_l", "data", "Hot standby: previous version", 2),
        (4, 4, 0, 6, "compute_l", "compute",
         f"Retrain\n{calcs.RETRAIN_GPUS} × {calcs.RETRAIN_H} h\n= {calcs.RETRAIN_GPUS * calcs.RETRAIN_H}\nGPU-h", 3),
        (4, 2, 6, 18, "network_l", "network", "Canary: new candidate", 3),
    ]
    for y0, h, x0, w, f, e, lab, st in blocks:
        a = 1.0 if k >= st else 0.0
        ax.add_patch(plt.Rectangle((x0, 8 - y0 - h), w, h, facecolor=C[f],
                                   edgecolor=C[e], lw=2, alpha=a))
        ax.text(x0 + w / 2, 8 - y0 - h / 2, lab, ha="center", va="center",
                fontsize=11, color=C["text"], alpha=a)
    a = 1.0 if k >= 3 else 0.0
    ax.add_patch(plt.Rectangle((6, 0.02), 18, 1.96, facecolor="white",
                               edgecolor=C["neutral"], lw=1.5, ls="--", alpha=a))
    ax.text(15, 1, "Free: one more copy", ha="center", va="center",
            fontsize=11, color=C["neutral"], alpha=a)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 8)
    ax.set_yticks([7.5 - i for i in range(8)])
    ax.set_yticklabels([f"GPU {i}" for i in range(8)], fontsize=10)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("Hours from the start of retraining", fontsize=11)
    sp.save(fig, out / f"nodeday_step{k}.svg")


nodeday(1)
nodeday(2)
nodeday(3)

# -----------------------------------------------------------------------------
# 10. Release modes: shadow, canary, staged ramp (chapter Sec. 14.4.3 and
#     table 11; the 5 percent canary is the ranking service's)
# -----------------------------------------------------------------------------
for k in (1, 2, 3):
    p = svg_open(900, 440)
    p.append(rect(10, 150, 170, 90, "#F4F4F4", "#777777", 2.5, 8))
    p.append(text(95, 190, "Router", 26, C["text"], "bold"))
    p.append(text(95, 220, "live requests", 22, C["neutral"]))
    p.append(rect(470, 40, 230, 86, C["memory_l"], C["memory"], 2.5, 8))
    p.append(text(585, 76, "v1, current", 26, C["text"], "bold"))
    p.append(text(585, 106, "100 GB copy", 22, C["memory"]))
    p.append(rect(470, 262, 230, 86, C["network_l"], C["network"], 2.5, 8))
    p.append(text(585, 298, "v2, candidate", 26, C["text"], "bold"))
    p.append(text(585, 328, "second 100 GB copy", 22, C["network"]))
    p.append(line(182, 175, 466, 90, C["memory"], 4, "b"))
    if k == 1:
        p.append(line(182, 215, 466, 300, C["network"], 3, "n", "9 6"))
        p.append(text(270, 98, "100% of users", 23, C["memory"], "bold"))
        p.append(text(300, 296, "a copy of", 23, C["network"], "bold"))
        p.append(text(300, 322, "every request", 23, C["network"], "bold"))
        p.append(text(712, 296, "answers logged,", 23, C["neutral"], anchor="start"))
        p.append(text(712, 324, "never shown", 23, C["neutral"], anchor="start"))
        p.append(text(450, 420, "Shadow: every request scored twice, no user exposed", 24,
                      C["text"], "bold"))
    else:
        p.append(line(182, 215, 466, 300, C["network"], 4, "n"))
        p.append(text(270, 98, "95% of users", 23, C["memory"], "bold"))
        p.append(text(300, 308, "5% of users", 23, C["network"], "bold"))
        p.append(text(712, 296, "compare clicks", 23, C["neutral"], anchor="start"))
        p.append(text(712, 324, "and p99 with v1", 23, C["neutral"], anchor="start"))
        if k == 2:
            p.append(text(450, 420, "Canary: a bounded share of users sees v2", 24,
                          C["text"], "bold"))
        else:
            xs = [60, 260, 460, 660]
            for i, s_ in enumerate(("5%", "25%", "50%", "100%")):
                p.append(rect(xs[i], 384, 120, 46, C["network_l"], C["network"], 2, 6))
                p.append(text(xs[i] + 60, 416, s_, 24, C["network"], "bold"))
                if i < 3:
                    p.append(line(xs[i] + 124, 407, xs[i + 1] - 6, 407, "#555555", 3))
                    p.append(text(xs[i] + 160, 396, "gate", 20, C["neutral"]))
    save_svg(p, f"release_step{k}.svg")
print("release written")


# =============================================================================
# Added 2026-10-18 for the Part 1 revision (Rebuild and Trace segment, A/B
# release decision, dual-bank update, stepped rollback). Sources:
#   - Equation 1 and footnote 4 (DVC): chapter Sec. 14.1.1.
#   - Registry lifecycle, lineage, registry bypass: Sec. 14.3.1.2 and
#     footnote 7; promotion gate's five checks: Sec. 14.7.2.
#   - Continuous training pipeline, seven stages: chapter figure 6, Sec. 14.3.2.
#   - Serving-artifact checks and the INT8 case: Sec. 14.4.2.
#   - A/B sizing: equation 4, Sec. 14.4.3 (745,644 users per variant for a
#     2 percent relative lift on 5 percent conversion; about 0.5 points
#     absolute detectable with 25,000 users per variant); table 10.
#   - Dual-bank update: Sec. 14.4.4 and table 11 (200 KB model, 4 MB flash).
#   - Stateful rollback: Sec. 14.4.3 and Checkpoint 14.2.
# =============================================================================

# Rollback in three steps: consistent, weights-only revert, restored together
for k in (1, 2, 3):
    p = svg_open(560, 420)
    wv = "v2" if k == 1 else "v1 (reverted)"
    ok = k == 1 or k == 3
    p.append(rect(4, 8, 250, 86, C["memory_l"], C["memory"], 2.5, 8))
    p.append(text(129, 44, "Model weights", 26, C["text"], "bold"))
    p.append(text(129, 78, "v1" if k == 3 else wv, 25, C["memory"], "bold"))
    cfill, cst = (C["data_l"], C["data"]) if ok else (C["danger_l"], C["danger"])
    p.append(rect(306, 8, 250, 86, cfill, cst, 2.5, 8))
    p.append(text(431, 44, "Embedding cache", 26, C["text"], "bold"))
    p.append(text(431, 78, "v1 vectors" if k == 3 else "v2 vectors", 25, cst, "bold"))
    if k == 1:
        p.append(text(280, 150, "v2 serving: weights and cache agree", 25,
                      C["data"], "bold"))
    if k >= 2:
        p.append(line(256, 51, 302, 51, C["danger"] if k == 2 else C["data"], 3,
                      "r" if k == 2 else "d"))
    if k == 2:
        p.append(text(280, 150, "v1 reads vectors from v2's space", 25,
                      C["danger"], "bold"))
        p.append(text(280, 182, "(v2 changed the embedding size)", 23, C["neutral"]))
    if k == 3:
        p.append(text(280, 150, "restored together: v1 agrees again", 25,
                      C["data"], "bold"))
        p.append(rect(4, 204, 552, 210, "#F7F7F7", C["data"], 2.5, 10))
        p.append(text(280, 244, "A tested rollback restores together:", 25,
                      C["data"], "bold"))
        for j, s_ in enumerate(("the weights, from the registry",
                                "the feature transform version",
                                "the cache (invalidate or swap)")):
            p.append(text(36, 290 + j * 40, "• " + s_, 24, C["text"],
                          anchor="start"))
    save_svg(p, f"rollback_step{k}.svg")


# Equation 1: a model is a function of four versioned inputs and the rounding
inputs = [("Code", "Git commit"), ("Data", "DVC hash"),
          ("Config", "config file"), ("Environment", "container image")]
for k in (1, 2):
    p = svg_open(900, 470)
    for i, (nm, tag) in enumerate(inputs):
        y = 20 + i * 100
        p.append(rect(10, y, 300, 82, C["data_l"], C["data"], 2.5, 8))
        p.append(text(160, y + 36, nm + " (version)", 27, C["text"], "bold"))
        p.append(text(160, y + 68, tag, 24, C["data"]))
        p.append(line(312, y + 41, 452, 210, "#555555", 3))
    p.append(rect(458, 160, 170, 100, C["memory_l"], C["memory"], 3, 10))
    p.append(text(543, 205, "training", 26, C["text"], "bold"))
    p.append(text(543, 236, "pipeline f", 26, C["text"], "bold"))
    p.append(line(630, 210, 716, 210, "#555555", 3))
    p.append(rect(722, 160, 168, 100, C["compute_l"], C["compute"], 3, 10))
    p.append(text(806, 205, "deployed", 26, C["text"], "bold"))
    p.append(text(806, 236, "model", 26, C["text"], "bold"))
    if k >= 2:
        p.append(rect(458, 330, 432, 110, "white", C["danger"], 2.5, 10, "8 6"))
        p.append(text(674, 368, "+ nondeterminism", 27, C["danger"], "bold"))
        p.append(text(674, 400, "summation order, seeds,", 24, C["danger"]))
        p.append(text(674, 428, "kernels, hardware", 24, C["danger"]))
        p.append(line(543, 328, 543, 266, C["danger"], 3, "r", "6 4"))
    save_svg(p, f"repro_step{k}.svg")


# The registry lifecycle: four states, promote and roll back (Sec. 14.3.1.2).
# Lineage is shown by the real dvc.lock on the previous frame; the promotion
# gate sits beside the MLflow screenshot on this one.
p = svg_open(900, 178)
states = [("candidate", 10), ("staging", 235), ("production", 460),
          ("retired", 685)]
for nm, x in states:
    prod = nm == "production"
    p.append(rect(x, 30, 200, 70, C["memory_l"] if prod else "#F4F4F4",
                  C["memory"] if prod else "#888888", 3 if prod else 2.5, 10))
    p.append(text(x + 100, 74, nm, 27, C["text"], "bold"))
for x in (212, 437, 662):
    p.append(line(x, 65, x + 20, 65, "#555555", 3))
p.append(text(335, 24, "promote", 22, C["neutral"], anchor="middle"))
p.append(path("M 560 102 C 560 150, 335 150, 335 106", C["danger"], 3, "r"))
p.append(text(447, 166, "roll back to the exact evaluated artifact", 24,
              C["danger"], "bold"))
save_svg(p, "registry.svg")


# The artifact that ships is not the checkpoint that passed
for k in (1, 2):
    p = svg_open(900, 380)
    boxes = [("Checkpoint", "passed evaluation", "memory"),
             ("Compile, quantize", "graph, INT8", "compute"),
             ("Serving engine", "what users get", "network")]
    for i, (a_, b_, ck) in enumerate(boxes):
        x = 10 + i * 300
        p.append(rect(x, 30, 260, 96, C[ck + "_l"], C[ck], 3, 10))
        p.append(text(x + 130, 72, a_, 26, C["text"], "bold"))
        p.append(text(x + 130, 106, b_, 23, C[ck]))
        if i < 2:
            p.append(line(x + 262, 78, x + 296, 78, "#555555", 3))
    if k == 2:
        checks = ["same outputs, slice by slice", "peak memory at load",
                  "warm-up and startup time", "runtime version pinned"]
        p.append(text(450, 186, "Check the engine, not the checkpoint:", 25,
                      C["danger"], "bold"))
        for i, c_ in enumerate(checks):
            x = 40 + (i % 2) * 440
            y = 236 + (i // 2) * 56
            p.append(rect(x, y - 34, 400, 48, "white", C["danger"], 2.5, 8))
            p.append(text(x + 200, y - 2, c_, 24, C["text"], "bold"))
    save_svg(p, f"artifact_step{k}.svg")


# A/B sizing (equation 4): users needed against users available, drawn at
# its slot (about 7.5 by 2.8 cm)
fig, ax = plt.subplots(figsize=(3.0, 1.1))
rows = (("Needed", calcs.N_NEEDED, "danger", f"{math.ceil(calcs.N_NEEDED):,}"),
        ("Have", calcs.N_AVAIL, "data", f"{calcs.N_AVAIL:,}"))
for i, (lab, v, ck, s_) in enumerate(rows):
    y = 1 - i
    ax.barh(y, v, color=C[ck], height=0.6)
    ax.text(v * 1.3, y, s_, va="center", fontsize=11, color=C[ck],
            fontweight="bold")
    ax.text(700, y, lab + " ", ha="right", va="center", fontsize=11,
            color=C["text"], fontweight="bold")
ax.set_xscale("log")
ax.minorticks_off()
ax.set_xlim(1000, 1e7)
ax.set_xticks([1e3, 1e4, 1e5, 1e6])
ax.set_xticklabels(["1K", "10K", "100K", "1M"], fontsize=10)
ax.set_yticks([])
ax.set_ylim(-0.6, 1.6)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Users per variant (log)", fontsize=10)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "ab_users.svg")


# Dual-bank update on the keyword spotter (Sec. 14.4.4)
for k in (1, 2):
    p = svg_open(900, 400)
    p.append(text(10, 32, "4 MB of flash on the device", 25, C["neutral"], "bold",
                  "start"))
    p.append(rect(10, 50, 880, 110, "#F7F7F7", "#888888", 2.5, 8))
    p.append(rect(24, 64, 420, 82, "#E8E8E8", "#999999", 2, 6))
    p.append(text(234, 113, "firmware and runtime", 24, C["neutral"], "bold"))
    p.append(rect(460, 64, 200, 82, C["data_l"], C["data"], 3, 6))
    p.append(text(560, 100, "Bank A", 25, C["data"], "bold"))
    p.append(text(560, 130, "200 KB, running", 21, C["data"]))
    p.append(rect(674, 64, 200, 82, C["network_l"], C["network"], 3, 6, "8 5"))
    p.append(text(774, 100, "Bank B", 25, C["network"], "bold"))
    p.append(text(774, 130, "200 KB, new", 21, C["network"]))
    if k == 2:
        p.append(rect(560, 220, 330, 80, "white", C["danger"], 2.5, 8))
        p.append(text(725, 254, "watchdog and self-test", 24, C["danger"], "bold"))
        p.append(text(725, 284, "on the first inferences", 22, C["danger"]))
        p.append(line(774, 162, 774, 214, C["network"], 3))
        p.append(path("M 556 262 C 480 262, 560 200, 560 166", C["danger"], 3, "r"))
        p.append(text(360, 252, "fault, hang, or late:", 24, C["danger"], "bold"))
        p.append(text(360, 284, "bootloader reverts to A", 24, C["danger"], "bold"))
        p.append(text(450, 366, "No technician needed; the cost is a second 200 KB bank.",
                      24, C["text"], "bold"))
    save_svg(p, f"dualbank_step{k}.svg")

print("Part 1 revision figures written")
