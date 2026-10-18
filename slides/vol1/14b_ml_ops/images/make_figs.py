#!/usr/bin/env python3
"""Figures for Lecture 14b (ML Operations, Part 2: Watching and Acting).

Every value is sourced. "Chapter" means the rendered ML Operations chapter.
"Derived" means computed here from sourced inputs; each slide that shows a
derived value says so.

  - Unity Audience Pinpointer, 2022: Unity Software Q1 2022 earnings call,
    10 May 2022 (transcript, The Motley Fool). J. Riccitiello: Operate
    "started the year strong in January, but then significantly slowed down
    in February and March"; "a fault in our platform that resulted in reduced
    accuracy for our Audience Pinpointer tool"; bad data "from a large
    customer"; "We are deploying monitoring, alerting and recovery systems and
    processes to promptly mitigate future complex data issues." L. Visoso:
    "We quantified this headwind at $110 million with roughly 60% impacting
    the second quarter, 30% impacting the third quarter" and 10% the fourth.
    No detection date is stated, so none is drawn.
  - Operations loop: chapter Sec. 14.1 and table 1 (as in Part 1).
  - Monitoring layers and when each can fire: chapter Sec. 14.5, 14.5.2.
  - Google data validation: Breck et al., "Data Validation for Machine
    Learning", SysML 2019, Table 1 (more than 700 pipelines, most recent 30
    days; "fired" = an anomaly fired on any day; "fixed" = none of that type
    for two days afterward) and Sec. 6.1.
  - Outcome join and label coverage: chapter Sec. 14.5.2.
  - Pending-outcome vote: illustrative values (labeled on the slide).
  - Marginal against joint statistics: chapter Sec. 14.5.2 (two binary flags).
  - Outcomes needed to confirm a drop: derived, one-sample binomial test,
    two-sided 95 percent confidence, 80 percent power, from 95 percent
    accuracy; the 1,000-outcome budget is Napkin Math 14.2.
  - Collection times: Napkin Math 14.2 (16.7 minutes; 10 days). Label delays:
    clicks "arrive within hours" (chapter Sec. 14.1); Visa card-absent fraud
    disputes, 120 calendar days from the transaction processing date (Visa
    Core Rules and Visa Product and Service Rules, 18 April 2026, table 11-29);
    Zillow resale prices "three to six months into the future" (Zillow Group
    Q3 2021 shareholder letter).
  - Detection accuracy against samples: Rabanser, Guennemann, and Lipton,
    "Failing Loudly", NeurIPS 2019, Table 2a (univariate BBSDs).
  - Cost of waiting: Napkin Math 14.3.
  - K-S critical value: 1.36 sqrt(2/n) at 95 percent (Lecture 4b, Napkin
    Math 4.5): 0.061 at 1,000 and 0.006 at 100,000; 0.0019 at 1,000,000
    derived. The 0.1 gate is the chapter's listing 2.
  - False alerts: derived from Napkin Math 14.5's 150 series (3 variants x 50
    metrics), each tested once a day. ICU alarms: Drew et al., PLOS ONE 2014
    (12,671 annotated arrhythmia alarms, 88.8 percent false).
  - PSI decomposition: chapter table 13; bands Sec. 14.5.2; severity
    deadlines chapter table 19 (illustrative policy).
  - Retraining gate: chapter listing 5 and equation 10.
  - Retraining cadences: Hazelwood et al., HPCA 2018, Table II. Freshness:
    He et al. 2014, Fig. 2 (bar heights read from the figure, approximate).
  - Sawtooth and T*: equations 8 to 12, Napkin Math 14.6, table 17. Cost per
    day at 1, 3, 7, 30 days derived from the same expression.
  - Scheduled against triggered retraining: chapter figure 9 redrawn from
    equation 8 with gamma = 0.0024/day; days to 0.88 and the delayed resets
    (0.878 after 1 day, 0.859 after 10 days) derived.
  - Slices: chapter table 20.
  - Repair sequence: chapter Sec. 14.7.1 (session-duration defect).
  - Debugging checklist: chapter Sec. 14.7.1. ML Test Score: chapter
    Sec. 14.8.1 (Breck et al. 2017); the 6, 7, 5, 1 audit is the chapter's.

Plots are drawn small because the slide shows them at about 5 cm tall;
drawn at that size their 11 to 13 pt type stays at 8 pt or more on the slide.

Run from this folder: python3 make_figs.py
"""

import math
import sys
from pathlib import Path

import numpy as np

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
         opacity=1.0, italic=False):
    color = color or C["text"]
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}" '
            f'opacity="{opacity}"{st}>{s}</text>')


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


def axis_clean(ax, grid="x"):
    ax.grid(axis=grid, color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


# -----------------------------------------------------------------------------
# 1. Unity, 2022: the slowdown, the disclosure, the impact by quarter
# -----------------------------------------------------------------------------
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]
AX_Y = 330


def mx(m):  # m: months since 1 January 2022
    return 60 + m * (800 / 12)


def unity_base(p, bars=True, bar_opacity=1.0, head=True):
    p.append(text(60, 38, "Audience Pinpointer, 2022", 28, C["text"],
                  "bold", "start"))
    # slowdown band, February and March
    p.append(rect(mx(1), 200, mx(3) - mx(1), AX_Y - 200, C["danger_l"],
                  C["danger"], 2, 4))
    p.append(text((mx(1) + mx(3)) / 2, 250, "Operate", 27, C["danger"], "bold"))
    p.append(text((mx(1) + mx(3)) / 2, 282, "slowed", 27, C["danger"], "bold"))
    p.append(line(60, AX_Y, 860, AX_Y, "#888888", 3, None))
    for i, mo in enumerate(MONTHS):
        p.append(line(mx(i), AX_Y, mx(i), AX_Y + 8, "#888888", 2, None))
        p.append(text(mx(i + 0.5), AX_Y + 31, mo, 26, C["neutral"]))
    # disclosure, 10 May
    xm = mx(4 + 9 / 31)
    p.append(f'<polygon points="{xm-10},{AX_Y+52} {xm+10},{AX_Y+52} {xm},{AX_Y+38}" '
             f'fill="{C["text"]}"/>')
    p.append(text(xm, AX_Y + 80, "10 May: first-quarter call discloses the fault",
                  27, C["text"], "bold"))
    if bars:
        for q, share in ((1, 60), (2, 30), (3, 10)):
            cx = mx(3 * q + 1.5)
            h = share * 2.2
            p.append(rect(cx - 60, AX_Y - h, 120, h, C["danger"], C["danger"],
                          0, 2, opacity=bar_opacity))
            p.append(text(cx, AX_Y - h - 12, f"Q{q+1}: {share}%", 27,
                          C["danger"], "bold", opacity=bar_opacity))
    if bars and head:
        p.append(text(860, 38, "of an estimated $110 million", 27,
                      C["danger"], "bold", "end", opacity=bar_opacity))
        p.append(text(860, 67, "2022 revenue impact (CFO)", 27, C["danger"],
                      anchor="end", opacity=bar_opacity))


for k in (1, 2):
    p = svg_open(900, 440)
    unity_base(p, bars=(k == 2))
    save_svg(p, f"unity_step{k}.svg")

# synthesis: the same timeline with the lecture's instruments drawn on it
p = svg_open(900, 440)
unity_base(p, bars=True, bar_opacity=0.25, head=False)
p.append(rect(30, 64, 420, 96, C["data_l"], C["data"], 2.5, 8))
p.append(text(240, 102, "Input check on each batch:", 28, C["data"], "bold"))
p.append(text(240, 140, "minutes after ingest, no labels", 28, C["data"]))
p.append(line(200, 160, 190, 197, C["data"], 3, "d"))
p.append(rect(470, 64, 420, 96, C["memory_l"], C["memory"], 2.5, 8))
p.append(text(680, 102, "Outcome layer: revenue,", 28, C["memory"], "bold"))
p.append(text(680, 140, "reported a quarter later", 28, C["memory"]))
save_svg(p, "unity_synth.svg")


# -----------------------------------------------------------------------------
# 2. The operations loop: Part 1 steps done, today's three steps lit
# -----------------------------------------------------------------------------
loop = [("Detect", 450, 62), ("Diagnose", 735, 180), ("Decide", 645, 372),
        ("Release", 255, 372), ("Recover", 165, 180)]
for k in (1, 2):
    p = svg_open(900, 470)
    p.append('<ellipse cx="450" cy="225" rx="285" ry="160" fill="none" '
             'stroke="#BBBBBB" stroke-width="3" stroke-dasharray="8 6"/>')
    p.append(text(450, 222, "Ranking service", 30, C["neutral"], "bold"))
    p.append(text(450, 256, "(illustrative)", 27, C["neutral"]))
    for (x1, y1, x2, y2) in ((572, 84, 650, 122), (722, 300, 700, 338),
                             (535, 385, 370, 385), (195, 340, 175, 300),
                             (240, 120, 320, 84)):
        p.append(line(x1, y1, x2, y2, "#888888", 3.5, "n"))
    for i, (nm, cx, cy) in enumerate(loop):
        done = i >= 3
        lit = k == 2 and i < 3
        if done:
            fill, stroke = C["data_l"], C["data"]
        elif lit:
            fill, stroke = C["compute_l"], C["compute"]
        else:
            fill, stroke = "#F4F4F4", "#999999"
        p.append(rect(cx - 105, cy - 34, 210, 68, fill, stroke, 3, 10))
        p.append(text(cx, cy + 12, nm, 34, C["text"], "bold"))
        if done:
            p.append(text(cx, cy + 68, "Part 1", 30, C["data"], "bold"))
        if lit:
            dy = -42 if i == 0 else 62
            dx = 160 if i == 0 else 0
            p.append(text(cx + dx, cy + 12 if i == 0 else cy + dy + 6, "today", 30,
                          C["compute"], "bold", "start" if i == 0 else "middle"))
    save_svg(p, f"loopring_step{k}.svg")


# -----------------------------------------------------------------------------
# 3. Monitoring layers, each with what it sees and when it can fire
# -----------------------------------------------------------------------------
layers = [  # name, sees, fires, color key
    ("Infrastructure telemetry", "up, fast, saturated", "per request: seconds",
     "memory"),
    ("Input checks", "schema, nulls, ranges", "per batch: before inference",
     "data"),
    ("Drift statistics", "inputs or predictions moved",
     "after hundreds of samples, no labels", "compute"),
    ("Outcomes", "accuracy fell", "after the label delay", "danger"),
]


def iceberg(p, ox, oy, s):
    def tx(x):
        return ox + x * s

    def ty(y):
        return oy + y * s
    waves = " ".join(f"L {tx(x)} {ty(122 + (8 if (x // 20) % 2 else -8))}"
                     for x in range(20, 600, 20))
    p.append(path(f"M {tx(0)} {ty(122)} {waves}", C["data"], 3, None))
    pts = " ".join(f"{tx(a)},{ty(b)}" for a, b in
                   ((236, 124), (282, 62), (300, 40), (318, 62), (364, 124)))
    p.append(f'<polygon points="{pts}" fill="#D4EFDF" stroke="{C["data"]}" '
             f'stroke-width="3"/>')
    pts = " ".join(f"{tx(a)},{ty(b)}" for a, b in
                   ((130, 132), (470, 132), (525, 250), (470, 425), (130, 425),
                    (75, 250)))
    p.append(f'<polygon points="{pts}" fill="{C["data"]}" '
             f'stroke="{C["data"]}" stroke-width="3"/>')
    p.append(text(tx(300), ty(250), "Data health", 22, "white", "bold"))
    p.append(text(tx(300), ty(330), "Model health", 22, "white", "bold"))
    p.append(text(tx(300), ty(18), "Uptime", 22, C["data"], "bold"))


for k in (1, 2, 3, 4):
    p = svg_open(900, 470)
    iceberg(p, 0, 20, 0.52)
    for i, (nm, sees, fires, ck) in enumerate(layers):
        if i >= k:
            continue
        y = 30 + i * 110
        p.append(rect(330, y, 560, 96, "white", C[ck], 3, 10))
        p.append(rect(330, y, 14, 96, C[ck], C[ck], 0, 4))
        p.append(text(362, y + 34, nm, 26, C[ck], "bold", "start"))
        p.append(text(362, y + 64, "sees: " + sees, 22, C["text"], anchor="start"))
        p.append(text(362, y + 88, "fires: " + fires, 22, C["text"], "bold",
                      "start"))
    save_svg(p, f"layers_step{k}.svg")


# -----------------------------------------------------------------------------
# 4. Google's input checks over 30 days (Breck et al. 2019, Table 1)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 2.3))
rows = [("New feature column", 10, "65%"), ("Out-of-domain category", 6, "66%"),
        ("Missing feature column", 6, "53%"), ("Data completely missing", 3, "65%"),
        ("Feature in too few examples", 3, "82%")]
for i, (lab, v, fx) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, v, color=C["data"], height=0.62)
    ax.text(v + 0.25, y, f"{v}%", va="center", fontsize=12, color=C["data"],
            fontweight="bold")
    ax.text(-0.3, y, lab, ha="right", va="center", fontsize=12, color=C["text"])
    ax.text(15.2, y, fx, ha="center", va="center", fontsize=12, color=C["neutral"])
ax.text(15.2, len(rows) - 0.35, "fixed", ha="center", va="bottom", fontsize=11,
        color=C["neutral"], fontweight="bold")
ax.set_xlim(0, 16.5)
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.set_yticks([])
ax.set_xticks([0, 5, 10])
ax.set_xticklabels(["0%", "5%", "10%"])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Share of 700+ pipelines where the check fired, 30 days", fontsize=11)
axis_clean(ax)
sp.save(fig, out / "breck_bars.svg")


# -----------------------------------------------------------------------------
# 5. Joining predictions to outcomes; pending is not negative
# -----------------------------------------------------------------------------
def logtable(p, x, y, title, cols, row, col):
    w = 380
    p.append(rect(x, y, w, 120, "white", col, 2.5, 8))
    p.append(rect(x, y, w, 40, col, col, 0, 6))
    p.append(text(x + w / 2, y + 29, title, 26, "white", "bold"))
    cw = w / len(cols)
    for j, (c_, v) in enumerate(zip(cols, row)):
        p.append(text(x + cw * (j + 0.5), y + 70, c_, 22, C["neutral"], "bold"))
        p.append(text(x + cw * (j + 0.5), y + 104, v, 24, C["text"]))


for k in (1, 2, 3):
    p = svg_open(900, 480)
    logtable(p, 10, 10, "Prediction log", ["request", "time", "score", "version"],
             ["r-17", "Mon", "0.31", "v2"], C["memory"])
    logtable(p, 510, 10, "Outcome log", ["request", "event", "observed"],
             ["r-17", "purchase", "Thu"], C["data"])
    p.append(line(392, 85, 506, 85, C["text"], 3, "n"))
    p.append(text(449, 160, "join on request ID", 24, C["text"], "bold"))
    p.append(text(449, 188, "keep the served version", 24, C["text"]))
    if k >= 2:
        y0, x0, x1 = 230, 60, 845
        p.append(line(x0, y0 + 160, 860, y0 + 160, "#888888", 3, "n"))
        p.append(line(x1, y0 - 10, x1, y0 + 166, C["text"], 2.5, None, "6 5"))
        p.append(text(x1, y0 + 190, "now", 24, C["text"], "bold"))
        p.append(text(60, y0 + 190, "prediction served", 22, C["neutral"],
                      anchor="start"))
        rws = [  # start, end of outcome window, label, color, fill, dash, dot
            (40, 330, "purchase in window: positive", "data", "data_l", None, 250),
            (150, 450, "closed, no purchase: negative", "neutral",
             "neutral_l", None, None),
            (690, 900, "window still open: pending, not negative", "danger",
             "danger_l", "8 6", None),
        ]
        for i, (a, b, lab, ck, fk, dash, dot) in enumerate(rws):
            y = y0 + i * 52
            b2 = min(b, 858)
            p.append(rect(a, y, b2 - a, 30, C[fk], C[ck], 2.5, 6, dash))
            if dot:
                p.append(f'<circle cx="{dot}" cy="{y+15}" r="9" fill="{C["data"]}"/>')
            tx = b2 + 12 if b2 < 600 else a - 12
            anc = "start" if b2 < 600 else "end"
            p.append(text(tx, y + 24, lab, 23, C[ck], "bold", anc))
    if k >= 3:
        p.append(text(450, 466, "Every metric reports label coverage: the share resolved",
                      24, C["danger"], "bold"))
    save_svg(p, f"join_step{k}.svg")


# Pending-outcome vote: illustrative values
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    vals = [("Incumbent\n30 days", 10.0, "memory", "95%"),
            ("Candidate\n2 days", 7.0, "network", "20%")]
    for i, (lab, v, ck, cov) in enumerate(vals):
        ax.bar(i, v, 0.55, color=C[ck])
        ax.text(i, v + 0.3, f"{v:.0f}%", ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=C[ck])
        a = 1.0 if k == 2 else 0.0
        ax.text(i, 0.4, f"{cov}\nresolved", ha="center", va="bottom",
                fontsize=11, color="white", fontweight="bold", alpha=a)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([v[0] for v in vals], fontsize=12, fontweight="bold")
    ax.set_ylim(0, 12.5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_yticks([0, 5, 10])
    ax.set_yticklabels(["0%", "5%", "10%"])
    ax.set_ylabel("Measured conversion", fontsize=11)
    axis_clean(ax, "y")
    ax.text(1.6, 12.2, "illustrative", ha="right", fontsize=10,
            color=C["neutral"], style="italic")
    sp.save(fig, out / f"pending_step{k}.svg")


# -----------------------------------------------------------------------------
# 6. Per-feature monitors miss a reversed relationship (chapter Sec. 14.5.2)
# -----------------------------------------------------------------------------
def marginal(p, x, y, label, col):
    p.append(text(x + 80, y, label, 24, C["text"], "bold"))
    for j, (nm, h) in enumerate((("off", 50), ("on", 50))):
        bx = x + 20 + j * 75
        p.append(rect(bx, y + 100 - h * 1.4, 55, h * 1.4, C[col], C[col], 0, 2))
        p.append(text(bx + 27, y + 126, nm, 22, C["neutral"]))
        p.append(text(bx + 27, y + 92 - h * 1.4, "50%", 22, C[col], "bold"))


def joint(p, x, y, cells, col, title):
    p.append(text(x + 120, y - 12, title, 25, C[col], "bold"))
    p.append(text(x - 10, y + 50, "A off", 22, C["neutral"], anchor="end"))
    p.append(text(x - 10, y + 110, "A on", 22, C["neutral"], anchor="end"))
    p.append(text(x + 60, y + 165, "B off", 22, C["neutral"]))
    p.append(text(x + 180, y + 165, "B on", 22, C["neutral"]))
    for r in range(2):
        for c_ in range(2):
            v = cells[r][c_]
            fill = C[col + "_l"] if v else "white"
            p.append(rect(x + c_ * 120, y + r * 60, 120, 60, fill, C[col], 2.5, 0))
            p.append(text(x + c_ * 120 + 60, y + r * 60 + 39, f"{v}%", 26,
                          C[col] if v else C["neutral"], "bold"))


for k in (1, 2, 3):
    p = svg_open(900, 450)
    p.append(text(225, 30, "Training population", 24, C["memory"], "bold"))
    p.append(text(675, 30, "Serving population", 24, C["danger"], "bold"))
    p.append(line(450, 10, 450, 440, "#CCCCCC", 2, None))
    marginal(p, 30, 62, "flag A", "memory")
    marginal(p, 250, 62, "flag B", "memory")
    marginal(p, 480, 62, "flag A", "danger")
    marginal(p, 700, 62, "flag B", "danger")
    if k == 1:
        p.append(text(450, 260, "Each flag is on half the time, before and after.",
                      26, C["text"], "bold"))
        p.append(text(450, 292, "A per-feature monitor reports no change.", 26,
                      C["text"]))
    if k >= 2:
        joint(p, 105, 250, [[50, 0], [0, 50]], "memory", "together: both off or both on")
    if k >= 3:
        joint(p, 555, 250, [[0, 50], [50, 0]], "danger", "together: reversed")
    save_svg(p, f"joint_step{k}.svg")


# -----------------------------------------------------------------------------
# 7. Outcomes needed to confirm a drop from 95 percent (derived)
# -----------------------------------------------------------------------------
ZA, ZB = 1.959964, 0.841621


def n_needed(p0, p1):
    return ((ZA * math.sqrt(p0 * (1 - p0)) + ZB * math.sqrt(p1 * (1 - p1)))
            / (p0 - p1)) ** 2


drops = [(5, 0.90), (2, 0.93), (1, 0.94)]
need = [n_needed(0.95, p1) for _, p1 in drops]  # about 185, 1,030, 3,930
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    for i, ((d, p1), n) in enumerate(zip(drops, need)):
        ax.bar(i, n, 0.55, color=C["danger"])
        ax.text(i, n * 1.15, f"{round(n, -1) if n > 1000 else round(n):,.0f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=C["danger"])
    a = 1.0 if k == 2 else 0.0
    ax.axhline(1000, color=C["memory"], lw=2, ls="--", alpha=a)
    ax.text(2.45, 760, "policy budget:\n1,000 (book)", fontsize=11,
            color=C["memory"], fontweight="bold", va="top", ha="left", alpha=a)
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_ylim(60, 12000)
    ax.set_yticks([100, 1000, 10000])
    ax.set_yticklabels(["100", "1,000", "10,000"])
    ax.set_xlim(-0.5, 3.3)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{d}-point drop\n95 to {round(p1*100)}%" for d, p1 in drops],
                       fontsize=11)
    ax.set_ylabel("Labeled outcomes", fontsize=11)
    axis_clean(ax, "y")
    sp.save(fig, out / f"evidence_step{k}.svg")


# -----------------------------------------------------------------------------
# 8. Time to confirm = label delay + collection time (log time axis)
# -----------------------------------------------------------------------------
MIN, HOUR, DAY = 1, 60, 1440
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    rws = [  # label, start, end, color, printed, step
        ("1,000 labels at 1/s", MIN, 16.7, "data", "16.7 min", 1),
        ("1,000 labels at 100/day", MIN, 10 * DAY, "danger", "10 days", 1),
        ("Ranking: clicks", HOUR, DAY, "neutral", "within hours", 2),
        ("Card fraud: disputes", MIN, 120 * DAY, "neutral", "up to 120 days", 2),
        ("Home resale prices", 90 * DAY, 180 * DAY, "neutral", "3 to 6 months", 2),
    ]
    n = len(rws)
    for i, (lab, a_, b_, ck, s_, st) in enumerate(rws):
        y = n - 1 - i
        al = 1.0 if k >= st else 0.0
        ax.barh(y, b_ - a_, left=a_, color=C[ck], height=0.6, alpha=al)
        ax.text(b_ * 1.3, y, s_, va="center", fontsize=12, fontweight="bold",
                color=C[ck], alpha=al)
        ax.text(0.8, y, lab + " ", ha="right", va="center", fontsize=12,
                color=C["text"], alpha=al, fontweight="bold" if st == 1 else "normal")
    a = 1.0 if k == 2 else 0.0
    ax.axhline(2.5, color=C["grid"], lw=1, alpha=a)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(1, 5e6)
    ax.set_xticks([MIN, HOUR, DAY, 30 * DAY, 365 * DAY])
    ax.set_xticklabels(["1 min", "1 hour", "1 day", "1 month", "1 year"])
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    axis_clean(ax)
    sp.save(fig, out / f"delay_step{k}.svg")


# -----------------------------------------------------------------------------
# 9. Detection accuracy against samples (Rabanser et al. 2019, Table 2a),
#    copied from Part 1's generator
# -----------------------------------------------------------------------------
ns = [10, 20, 50, 100, 200, 500, 1000, 10000]
acc = {
    "Large shift": ([0.32, 0.54, 0.78, 0.82, 0.83, 0.92, 0.96, 1.00], "data"),
    "Medium shift": ([0.14, 0.21, 0.39, 0.38, 0.42, 0.57, 0.66, 0.76], "compute"),
    "Small shift": ([0.00, 0.00, 0.14, 0.14, 0.18, 0.36, 0.40, 0.54], "danger"),
}
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.3, 2.1))
    for i, (lab, (v, ck)) in enumerate(acc.items()):
        a = 1.0 if (i < 2 or k == 2) else 0.0
        ax.plot(ns, v, "o-", color=C[ck], lw=3, ms=6, alpha=a)
        ax.text(13000, v[-1], lab, color=C[ck], fontsize=11, fontweight="bold",
                va="center", alpha=a)
    a = 1.0 if k == 2 else 0.0
    ax.axvline(1000, color=C["neutral"], ls="--", lw=1.2, alpha=a)
    ax.text(1150, 0.16, "at 1,000:\nlarge 0.96\nsmall 0.40", ha="left",
            fontsize=10, color=C["text"], alpha=a, fontweight="bold", va="center")
    ax.axhline(0.5, color=C["grid"], lw=1)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(8, 12000)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10\n10 s\n2.4 h", "100\n1.7 min\n1 day",
                        "1,000\n16.7 min\n10 days", "10,000\n2.8 h\n100 days"],
                       fontsize=10)
    ax.text(-0.09, -0.07, "samples\nat 1/s\nat 100/day", ha="right", va="top",
            fontsize=10, color=C["neutral"], transform=ax.transAxes)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("Detection accuracy", fontsize=11)
    ax.grid(axis="y", color=C["grid"], linewidth=0.6)
    sp.save(fig, out / f"rabanser_step{k}.svg")


# -----------------------------------------------------------------------------
# 10. Cost of a silent failure (Napkin Math 14.3), copied from Part 1
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.8, 1.15))
for i, (lab, v, ck, s_) in enumerate((("Monthly review", 191781, "danger", "$191,781"),
                                       ("Daily check", 6849, "data", "$6,849"))):
    y = 1 - i
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v + 5000, y, s_, va="center", fontsize=12, color=C[ck], fontweight="bold")
    ax.text(-4000, y, lab + " ", ha="right", va="center", fontsize=12,
            fontweight="bold", color=C["text"])
ax.set_xlim(0, 290000)
ax.set_xticks([0, 100000, 200000])
ax.set_xticklabels(["$0", "$100K", "$200K"], fontsize=11)
ax.set_yticks([])
ax.set_ylim(-0.6, 1.6)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Revenue lost per incident", fontsize=11)
axis_clean(ax)
sp.save(fig, out / "silent_cost.svg")


# -----------------------------------------------------------------------------
# 11. K-S critical value against sample size, with the 0.1 gate
# -----------------------------------------------------------------------------
nn = np.logspace(2, 7, 200)
crit = 1.36 * np.sqrt(2 / nn)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.plot(nn, crit, color=C["memory"], lw=3)
    for n_, s_, lab in ((1e3, "0.061", ""), (1e5, "0.006", ""),
                        (1e6, "0.0019", " (derived)")):
        v = 1.36 * math.sqrt(2 / n_)
        ax.plot([n_], [v], "o", ms=9, color=C["memory"], zorder=5)
        below = n_ < 1e6
        ax.text(n_ / 1.3 if below else n_ * 1.35, v * (0.5 if below else 1.25),
                s_ + lab, fontsize=12, color=C["memory"], fontweight="bold",
                ha="right" if below else "left")
    a = 1.0 if k == 2 else 0.0
    ax.axhline(0.1, color=C["danger"], lw=2.5, ls="--", alpha=a)
    ax.text(2.2e3, 0.125, "promotion gate in listing 2: 0.1", fontsize=11,
            color=C["danger"], fontweight="bold", alpha=a)
    ax.annotate("", xy=(1e6, 0.0021), xytext=(1e6, 0.092),
                arrowprops=dict(arrowstyle="<->", color=C["danger"], lw=1.8),
                alpha=a)
    ax.text(1.4e6, 0.02, "50 times\nsmaller", fontsize=11, color=C["danger"],
            fontweight="bold", alpha=a, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xlim(1e2, 2e7)
    ax.set_ylim(1e-3, 0.3)
    ax.set_xticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(["100", "1K", "10K", "100K", "1M", "10M"])
    ax.set_yticks([0.001, 0.01, 0.1])
    ax.set_yticklabels(["0.001", "0.01", "0.1"])
    ax.set_xlabel("Samples per side", fontsize=11)
    ax.set_ylabel("K-S critical value", fontsize=11)
    ax.grid(color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f"ks_step{k}.svg")


# -----------------------------------------------------------------------------
# 12. False alerts from a monitor with nothing to find (derived), and a
#     measured alarm stream (Drew et al. 2014)
# -----------------------------------------------------------------------------
SERIES = 3 * 50
fig, ax = plt.subplots(figsize=(4.6, 1.6))
for i, (alpha_, ck) in enumerate(((0.05, "danger"), (0.01, "compute"),
                                  (0.001, "data"))):
    y = 2 - i
    v = SERIES * alpha_
    ax.barh(y, v, color=C[ck], height=0.62)
    s_ = f"{v:g} a day" + ("  (about one a week)" if alpha_ == 0.001 else "")
    ax.text(v + 0.15, y, s_, va="center", fontsize=12, color=C[ck],
            fontweight="bold")
    ax.text(-0.15, y, f"false-positive rate {alpha_:g}", ha="right",
            va="center", fontsize=12, color=C["text"])
ax.set_xlim(0, 10.5)
ax.set_yticks([])
ax.set_ylim(-0.6, 2.6)
ax.set_xticks([0, 5, 10])
ax.spines["left"].set_visible(False)
ax.set_xlabel("False alerts per day, 150 series tested daily", fontsize=11)
axis_clean(ax)
sp.save(fig, out / "falsealarm.svg")

fig, ax = plt.subplots(figsize=(4.6, 0.95))
ax.barh(0, 88.8, color=C["danger"], height=0.6)
ax.barh(0, 11.2, left=88.8, color=C["data"], height=0.6)
ax.text(44.4, 0, "88.8% false", ha="center", va="center", fontsize=12,
        color="white", fontweight="bold")
ax.text(101, 0, "true", ha="left", va="center", fontsize=12, color=C["data"],
        fontweight="bold")
ax.set_xlim(0, 112)
ax.set_ylim(-0.5, 0.5)
ax.axis("off")
sp.save(fig, out / "icu.svg")


# -----------------------------------------------------------------------------
# 13. PSI: the age feature (table 13), then the escalation ladder
# -----------------------------------------------------------------------------
bins = ["18–25", "26–35", "36–45", "46–55", "56–65", "66+"]
train = [15, 25, 20, 18, 12, 10]
serve = [12, 22, 18, 20, 15, 13]
fig, ax = plt.subplots(figsize=(4.2, 2.3))
x = np.arange(len(bins))
ax.bar(x - 0.2, train, 0.4, color=C["memory"])
ax.bar(x + 0.2, serve, 0.4, color=C["compute"])
ax.text(5.6, 24.5, "training", color=C["memory"], fontsize=12, fontweight="bold",
        ha="right")
ax.text(5.6, 21.5, "serving", color=C["compute"], fontsize=12, fontweight="bold",
        ha="right")
ax.set_xticks(x)
ax.set_xticklabels(bins, fontsize=11)
ax.set_xlabel("Age bin", fontsize=11)
ax.set_ylabel("Share of users (%)", fontsize=11)
ax.set_ylim(0, 28)
ax.set_yticks([0, 10, 20])
axis_clean(ax, "y")
sp.save(fig, out / "psi_bars.svg")

for k in (1, 2):
    p = svg_open(580, 470)
    p.append(text(10, 30, "PSI band", 26, C["neutral"], "bold", "start"))
    bands = [("≥ 0.25", "notify owner", "danger", "danger_l"),
             ("0.1 to 0.25", "investigate", "compute", "compute_l"),
             ("< 0.1", "log; watch outcomes", "data", "data_l")]
    for i, (rng, act, ck, fk) in enumerate(bands):
        y = 48 + i * 116
        p.append(rect(10, y, 280, 100, C[fk], C[ck], 3, 8))
        p.append(text(28, y + 42, rng, 30, C[ck], "bold", "start"))
        p.append(text(28, y + 80, act, 25, C["text"], anchor="start"))
    p.append(text(150, 440, "age feature: 0.029", 27, C["text"], "bold"))
    p.append(line(150, 412, 150, 384, C["text"], 3, "n"))
    if k == 2:
        p.append(text(340, 30, "Severity", 26, C["neutral"], "bold", "start"))
        sev = [("P0", "15 min"), ("P1", "1 hour"), ("P2", "4 hours"),
               ("P3", "24 hours")]
        for i, (lv, t_) in enumerate(sev):
            y = 48 + i * 78
            p.append(rect(340, y, 230, 64, "white", C["network"], 2.5, 8))
            p.append(text(358, y + 42, lv, 28, C["network"], "bold", "start"))
            p.append(text(420, y + 42, t_, 27, C["text"], "bold", "start"))
        p.append(text(455, 380, "each with an", 24, C["network"]))
        p.append(text(455, 410, "owner and a", 24, C["network"]))
        p.append(text(455, 440, "deadline", 24, C["network"]))
        p.append(line(292, 98, 334, 98, C["network"], 3, "n", "6 4"))
    save_svg(p, f"psi_ladder_step{k}.svg")


# -----------------------------------------------------------------------------
# 14. The retraining gate (listing 5, equation 10)
# -----------------------------------------------------------------------------
for k in (1, 2, 3):
    p = svg_open(900, 470)
    # beam and fulcrum
    p.append(line(150, 120, 750, 120, C["text"], 5, None))
    p.append('<polygon points="450,120 425,175 475,175" fill="#888888"/>')
    p.append(line(150, 120, 150, 150, C["text"], 2, None))
    p.append(line(750, 120, 750, 150, C["text"], 2, None))
    p.append(rect(20, 150, 260, 150, C["data_l"], C["data"], 3, 10))
    p.append(text(150, 186, "Value recovered", 27, C["data"], "bold"))
    for j, s_ in enumerate(("from quality loss", "feature drift",
                            "prediction shift")):
        p.append(text(150, 224 + j * 30, s_, 25, C["text"]))
    p.append(rect(600, 150, 290, 150, C["compute_l"], C["compute"], 3, 10))
    p.append(text(745, 186, "Cost of a run", 27, C["compute"], "bold"))
    for j, s_ in enumerate(("compute + validation", "+ deployment",
                            "+ rollout risk")):
        p.append(text(745, 224 + j * 30, s_, 25, C["text"]))
    p.append(text(450, 60, "A drift alert adds evidence to the left pan.", 26,
                  C["neutral"]))
    if k >= 2:
        p.append(rect(200, 330, 500, 64, "white", C["text"], 3, 10))
        p.append(text(450, 360, "Run only if value > cost", 26, C["text"], "bold"))
        p.append(text(450, 388, "and fresh validation data exist", 24, C["text"]))
    if k >= 3:
        for j, (nm, ck) in enumerate((("scheduled", "memory"),
                                      ("triggered", "data"),
                                      ("continuous", "danger"))):
            x = 60 + j * 280
            p.append(rect(x, 414, 220, 46, C[ck + "_l"], C[ck], 2.5, 22))
            p.append(text(x + 110, 446, nm, 26, C[ck], "bold"))
    save_svg(p, f"gate_step{k}.svg")


# -----------------------------------------------------------------------------
# 15. Retraining cadence at Facebook (Hazelwood et al. 2018) and freshness
#     (He et al. 2014), copied from Part 1
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.7, 2.3))
rungs = [("Hourly", "Search"), ("Sub-daily", "Sigma"), ("Daily", "News Feed"),
         ("Weekly", "Translation,\nSpeech"), ("Multi-monthly", "Lumos")]
for i, (cad, svc) in enumerate(rungs):
    y = len(rungs) - 1 - i
    ax.plot([0], [y], "o", ms=10, color=C["memory"])
    ax.text(-0.15, y, cad, ha="right", va="center", fontsize=11,
            fontweight="bold", color=C["memory"])
    ax.text(0.15, y, svc, ha="left", va="center", fontsize=11, color=C["text"])
ax.plot([0, 0], [0, len(rungs) - 1], color=C["memory"], lw=2, zorder=0)
ax.set_xlim(-1.6, 1.4)
ax.set_ylim(-0.5, len(rungs) - 0.5)
ax.axis("off")
sp.save(fig, out / "fb_cadence.svg")

fig, ax = plt.subplots(figsize=(2.9, 1.8))
days = list(range(7))
ne = [95.65, 95.88, 95.67, 95.92, 96.02, 96.33, 96.63]
ax.plot(days, ne, "o-", color=C["compute"], lw=3, ms=7)
ax.set_xticks(days)
ax.tick_params(labelsize=10)
ax.set_xlabel("Days between training and test", fontsize=11)
ax.set_ylabel("Relative NE (%)", fontsize=11)
ax.set_ylim(95.4, 96.9)
ax.set_yticks([95.5, 96.0, 96.5])
ax.text(0.0, 96.62, "lower is better", fontsize=10, color=C["neutral"],
        va="center")
ax.grid(axis="y", color=C["grid"], linewidth=0.6)
ax.text(0.0, -0.45, "Read from the paper's figure; approximate",
        transform=ax.transAxes, fontsize=9, color=C["neutral"], ha="left")
sp.save(fig, out / "he_freshness.svg")


# -----------------------------------------------------------------------------
# 16. The sawtooth: value lost per day grows linearly, resets at each run
# -----------------------------------------------------------------------------
Q, V, A0, G, CR = 1_000_000, 0.50, 0.94, 0.0024, 5000
SLOPE = Q * V * A0 * G          # 1,128 dollars per day, per day
TSTAR = math.sqrt(2 * CR / SLOPE)  # 2.98 days
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    t = np.linspace(0, 3 * TSTAR - 1e-6, 900)
    loss = SLOPE * (t % TSTAR)
    ax.plot(t, loss, color=C["danger"], lw=3)
    ax.text(0.15, 6100, "value lost per day\n= 1,128 × days since the last run",
            fontsize=11, color=C["danger"], fontweight="bold", va="top")
    if k >= 2:
        tt = np.linspace(TSTAR, 2 * TSTAR - 1e-6, 200)
        ax.fill_between(tt, 0, SLOPE * (tt - TSTAR), color=C["danger_l"])
        ax.text(4.47, 3600, "shaded cycle:\n½ × 1,128 × T²",
                fontsize=11, color=C["danger"], fontweight="bold", ha="center",
                va="bottom")
    if k >= 3:
        for j in (1, 2, 3):
            ax.plot([j * TSTAR], [0], "s", ms=12, color=C["memory"], zorder=6,
                    clip_on=False)
        ax.plot([6.75], [5800], "s", ms=11, color=C["memory"])
        ax.text(7.0, 5800, "a run: C = $5,000", fontsize=11,
                color=C["memory"], fontweight="bold", va="center")
    ax.set_xlim(0, 9.3)
    ax.set_ylim(0, 6200)
    ax.set_xticks([0, TSTAR, 2 * TSTAR, 3 * TSTAR])
    ax.set_xticklabels(["0", "T", "2T", "3T"], fontsize=12)
    ax.tick_params(axis="x", pad=9)
    ax.set_yticks([0, 2000, 4000])
    ax.set_yticklabels(["$0", "$2,000", "$4,000"])
    ax.set_xlabel("Days (ranking service, illustrative)", fontsize=11)
    axis_clean(ax, "y")
    sp.save(fig, out / f"saw_step{k}.svg")


# Cost per day against the interval (equation 12 redrawn), from Part 1
T = np.linspace(0.6, 10, 300)
run = CR / T
stale = 0.5 * SLOPE * T
fig, ax = plt.subplots(figsize=(3.5, 2.1))
ax.plot(T, run, color=C["memory"], lw=3)
ax.plot(T, stale, color=C["danger"], lw=3)
ax.text(5.0, 1000, "run cost C/T", color=C["memory"], fontsize=11,
        fontweight="bold", va="bottom")
ax.text(10.2, 5300, "staleness", color=C["danger"], fontsize=11,
        fontweight="bold", va="center")
ax.plot(T, run + stale, color=C["text"], lw=2.5, ls="--")
ax.plot([TSTAR], [CR / TSTAR + 0.5 * SLOPE * TSTAR], "o", ms=10,
        color=C["text"], zorder=5)
ax.axvline(TSTAR, color=C["neutral"], lw=1.2, ls=":")
ax.text(TSTAR + 0.3, 7400, f"T* = {TSTAR:.2f} days", fontsize=12,
        fontweight="bold", color=C["text"])
ax.text(10.2, 6600, "total", fontsize=11, color=C["text"], fontweight="bold",
        va="center")
ax.set_xlim(0, 10)
ax.set_ylim(0, 8500)
ax.tick_params(labelsize=10)
ax.set_xlabel("Retraining interval T (days)", fontsize=11)
ax.set_ylabel("Cost per day ($)", fontsize=11)
ax.set_yticks([0, 4000, 8000])
ax.set_yticklabels(["0", "4,000", "8,000"])
ax.grid(color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / "tstar.svg")

# Cost per day at four cadences (derived from the same expression)
fig, ax = plt.subplots(figsize=(3.6, 1.9))
cad = [("Daily", 1), ("Every 3 days", 3), ("Weekly", 7), ("Monthly", 30)]
for i, (lab, T_) in enumerate(cad):
    y = len(cad) - 1 - i
    v = CR / T_ + 0.5 * SLOPE * T_
    ck = "data" if T_ == 3 else "neutral"
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v + 300, y, f"${v:,.0f}", va="center", fontsize=12, color=C[ck],
            fontweight="bold")
    ax.text(-300, y, lab, ha="right", va="center", fontsize=12, color=C["text"],
            fontweight="bold" if T_ == 3 else "normal")
ax.set_xlim(0, 22000)
ax.set_xticks([0, 10000, 20000])
ax.set_xticklabels(["$0", "$10K", "$20K"])
ax.set_yticks([])
ax.set_ylim(-0.6, len(cad) - 0.4)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Total cost per day (derived)", fontsize=11)
axis_clean(ax)
sp.save(fig, out / "tcost.svg")


# -----------------------------------------------------------------------------
# 17. Scheduled against triggered retraining, and the detection delay
# -----------------------------------------------------------------------------
FLOOR = 0.88
T_HIT = math.log(A0 / FLOOR) / G   # 27.5 days


def sawtooth(t, period):
    return A0 * np.exp(-G * (t % period))


days = np.linspace(0, 120, 4001)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    ax.axhline(FLOOR, color=C["neutral"], lw=1.8, ls="--")
    ax.text(121, FLOOR, "floor 0.88", fontsize=11, color=C["neutral"],
            va="center", fontweight="bold")
    ax.plot(days, sawtooth(days, 90), color=C["compute"], lw=2.6)
    ax.text(2, 0.748, "scheduled every 90 days: falls to 0.757", fontsize=11,
            color=C["compute"], fontweight="bold", ha="left", va="bottom")
    if k >= 2:
        ax.plot(days, sawtooth(days, T_HIT), color=C["data"], lw=2.6)
        ax.text(60, 0.947, "trigger at 0.88: every 27.5 days", fontsize=11,
                color=C["data"], fontweight="bold", ha="center")
    if k >= 3:
        per = T_HIT + 10
        ax.plot(days, sawtooth(days, per), color=C["danger"], lw=2.2, ls=(0, (5, 3)))
        ax.annotate("10-day detection delay:\nresets at 0.859", xy=(per, 0.859),
                    xytext=(92, 0.79), fontsize=11, color=C["danger"],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["danger"], lw=1.5))
    ax.set_xlim(0, 120)
    ax.set_ylim(0.74, 0.965)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.94])
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_xlabel("Days since deployment", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    axis_clean(ax, "y")
    sp.save(fig, out / f"rot_step{k}.svg")


# -----------------------------------------------------------------------------
# 18. Slices hide a failing segment (table 20), copied from Part 1
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.0, 2.2))
rows = [("Desktop (45%)", 94, "neutral", "94%"), ("iOS (30%)", 92, "neutral", "92%"),
        ("Android (20%)", 88, "neutral", "88%"), ("Tablet (5%)", 62, "danger", "62%"),
        ("Overall", 91, "memory", "91%")]
for i, (lab, v, ck, s_) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v + 1.5, y, s_, va="center", fontsize=12, color=C[ck], fontweight="bold")
    ax.text(-2, y, lab, ha="right", va="center", fontsize=12, fontweight="bold",
            color=C["text"])
ax.set_xlim(0, 112)
ax.set_xticks([0, 50, 100])
ax.tick_params(labelsize=10)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Accuracy (traffic share)", fontsize=11)
axis_clean(ax)
sp.save(fig, out / "slices.svg")


# -----------------------------------------------------------------------------
# 19. A repair restores the contract (session-duration defect, Sec. 14.7.1)
# -----------------------------------------------------------------------------
def card(p, x, y, w, h, l1, l2, ck, fk, dash=None):
    p.append(rect(x, y, w, h, C[fk], C[ck], 3, 10, dash))
    p.append(text(x + w / 2, y + h / 2 - 4, l1, 25, C["text"], "bold"))
    p.append(text(x + w / 2, y + h / 2 + 26, l2, 23, C[ck]))


for k in (1, 2, 3):
    p = svg_open(900, 470)
    W, H = 270, 92
    xs = [10, 315, 620]
    row1 = [("Alert", "click-through −12%", "danger", "danger_l"),
            ("Replay exact inputs", "45 min against 12", "memory", "memory_l"),
            ("Restore transform", "and cache namespace", "data", "data_l")]
    row2 = [("Verify on replay", "values agree again", "data", "data_l"),
            ("Bounded canary", "then full traffic", "data", "data_l"),
            ("Mature outcomes", "confirm, days later", "neutral", "neutral_l")]
    shown1 = 2 if k == 1 else 3
    for i in range(shown1):
        card(p, xs[i], 20, W, H, *row1[i])
        if i < shown1 - 1:
            p.append(line(xs[i] + W + 2, 66, xs[i + 1] - 4, 66, "#555555", 3))
    if k >= 2:
        p.append(path(f"M {xs[2]+W/2} {20+H+2} L {xs[2]+W/2} 140 L {xs[0]+W/2} 140 "
                      f"L {xs[0]+W/2} 156", "#555555", 3))
        for i in range(3):
            card(p, xs[i], 160, W, H, *row2[i])
            if i < 2:
                p.append(line(xs[i] + W + 2, 206, xs[i + 1] - 4, 206, "#555555", 3))
    if k >= 3:
        for i, (a_, b_) in enumerate((("Roll back weights only", "transform still broken"),
                                      ("Retrain now", "learns from the defect"))):
            x = 60 + i * 420
            p.append(rect(x, 320, 360, 100, "#F7F7F7", "#999999", 2.5, 10, "8 6"))
            p.append(text(x + 180, 362, a_, 25, C["neutral"], "bold"))
            p.append(text(x + 180, 398, b_, 23, C["danger"], "bold"))
            p.append(line(x + 20, 330, x + 340, 410, C["danger"], 3, None))
        p.append(text(450, 304, "Not repairs", 24, C["neutral"], "bold"))
    save_svg(p, f"repair_step{k}.svg")


# -----------------------------------------------------------------------------
# 20. Debugging checklist ending in Prevent; the ML Test Score audit
# -----------------------------------------------------------------------------
p = svg_open(900, 110)
phases = ["Reproduce", "Isolate", "Bisect", "Attribute", "Validate", "Prevent"]
for i, ph in enumerate(phases):
    x = 4 + i * 150
    last = i == len(phases) - 1
    p.append(rect(x, 20, 138, 64, C["data_l"] if last else "#F4F4F4",
                  C["data"] if last else "#999999", 3 if last else 2, 10))
    p.append(text(x + 69, 61, ph, 24, C["data"] if last else C["text"], "bold"))
    if i < len(phases) - 1:
        p.append(line(x + 139, 52, x + 151, 52, "#555555", 2.5, None))
save_svg(p, "checklist.svg")

for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    cats = [("Data", 6), ("Model", 7), ("Infra.", 5), ("Monitoring", 1)]
    for i, (nm, v) in enumerate(cats):
        hi = k == 2 and v == 1
        ck = "danger" if hi else "memory"
        ax.bar(i, v, 0.6, color=C[ck])
        ax.text(i, v + 0.2, f"{v}" if not hi else "", ha="center", va="bottom", fontsize=13,
                fontweight="bold", color=C[ck])
    a = 1.0 if k == 2 else 0.0
    ax.axhline(1, color=C["danger"], lw=2, ls="--", alpha=a)
    ax.text(3, 2.0, "score = 1", fontsize=12, color=C["danger"],
            fontweight="bold", ha="center", alpha=a)
    ax.set_xticks(range(4))
    ax.set_xticklabels([c_[0] for c_ in cats], fontsize=11)
    ax.set_ylim(0, 8.2)
    ax.set_yticks([0, 7])
    ax.set_yticklabels(["0", "7"])
    ax.set_ylabel("Tests passed (of 7)", fontsize=11)
    axis_clean(ax, "y")
    sp.save(fig, out / f"mlts_step{k}.svg")

print("figures written; T* =", round(TSTAR, 2), "days; outcomes needed:",
      [round(n_) for n_ in need], "; days to 0.88:", round(T_HIT, 1))
