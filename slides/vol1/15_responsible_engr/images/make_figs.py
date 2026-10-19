#!/usr/bin/env python3
"""Figures for Lecture 15 (Responsible Engineering).

Plots use the shared slideplot helper; diagrams are written as SVG with a
small builder below, one file per build step, every step on the same canvas
so nothing moves. Every value is sourced:

  - Care-management algorithm: Obermeyer, Powers, Vogeli, Mullainathan,
    Science 366:447 (2019), read from the full text. 6,079 Black and 43,539
    White patients, 2013 to 2015, one academic hospital; auto-enrollment at
    the 97th percentile of risk; Black share of auto-enrolled 17.7 percent,
    46.5 percent after removing the gap in health at a given score (Fig. 1B);
    26.3 percent more chronic illnesses at the 97th percentile (4.8 against
    3.8); $1,801 less spent per year at the same number of chronic illnesses;
    Table 2, Black share of the top 3 percent by training label: total cost
    0.141, avoidable cost 0.210, active chronic conditions 0.267.
  - Gender Shades (Buolamwini and Gebru 2018, Table 4), Face++ error:
    all 10.0, darker female 34.5, darker male 0.7, lighter female 9.8,
    lighter male 0.8 percent (chapter table 3 for the four groups).
  - Raji and Buolamwini 2019, Tables 1 and 2: darker-female error August 2018,
    Face++ 4.1, Microsoft 1.52, IBM 16.97, Amazon 31.37, Kairos 22.50; May 2017
    values from Gender Shades Table 4: Face++ 34.5, Microsoft 20.8, IBM 34.7.
  - Representation budget: Napkin Math 15.1, margin 1.96 sqrt(0.25/n).
  - COMPAS: ProPublica, "How We Analyzed the COMPAS Recidivism Algorithm"
    (2016): false positive rate 44.85 and 23.45, false negative rate 27.99 and
    47.72, PPV 0.63 and 0.59 (Black, White); base rates derived from its
    contingency tables, 1,901/3,696 and 966/2,454.
  - Credit service (constructed, chapter Sec. 15.3.2.1, tables 5 to 7) and the
    price of fairness (Napkin Math 15.2): $150 baseline, $75 after.
  - Feedback (Napkin Math 15.3, illustrative): gap 3.0 points, factor 1.10 and
    0.90 per quarter, gate 5 points.
  - Shared threshold (chapter figure 2 and Sec. 15.3.2.1): 75 percent overall
    accuracy at one line, 81.25 percent at the other; Subgroup B's errors
    triple (1 to 3) while Subgroup A's fall to 0.
  - Interventions (chapter Sec. 15.3.2.1): reweighting, adversarial debiasing
    with a reversed gradient, threshold adjustment (legally restricted).
  - Next-lecture teaser: Strubell et al. 2019 (284 t estimate; car lifetime
    57 t); the measured value is revealed in Lecture 15b.

Derived values come from calcs.py (imported below), never typed here.

The Patterson, DLRM TCO, fleet-share, lineage, and DP-SGD figures moved to
Lecture 15b on 2026-10-17 and are generated there; their generators were
removed from this file on 2026-10-19 (the old SVG and PDF files they wrote
remain in this folder, unused).

Run from this folder: python3 make_figs.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
import slideplot as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calcs as calc  # noqa: E402

plt = sp.plt
C = sp.C
out = Path(__file__).parent


# =============================================================================
# Plots
# =============================================================================

def bar_steps(stem, rows, steps, size, xmax, xlabel, note=None, vlines=None,
              log=False, xmin=None):
    """Horizontal bars with per-step visibility.

    rows: (label, value, color_key, value_text, first_step). Every artist is
    drawn in every step (hidden ones transparent) so the figure never moves.
    """
    n = len(rows)
    for k in range(1, steps + 1):
        fig, ax = plt.subplots(figsize=size)
        for i, (label, v, ck, txt, first) in enumerate(rows):
            y = n - 1 - i
            on = k >= first
            ax.barh(y, v, color=C[ck], height=0.62, alpha=1.0 if on else 0.0)
            ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=12, color=C["text"],
                    fontweight="bold", alpha=1.0 if on else 0.0)
            x = v * 1.12 if log else v + xmax * 0.012
            ax.text(x, y, txt, va="center", fontsize=12, color=C[ck],
                    fontweight="bold", alpha=1.0 if on else 0.0)
        for x, label, ck, first in (vlines or []):
            on = k >= first
            ax.axvline(x, color=C[ck], lw=1.6, ls="--", zorder=1, alpha=1.0 if on else 0.0)
            ax.text(x, n - 0.45, " " + label, color=C[ck], fontsize=10,
                    fontweight="bold", va="bottom", ha="left", alpha=1.0 if on else 0.0)
        ax.set_yticks([])
        if log:
            ax.set_xscale("log")
            ax.set_xlim(xmin, xmax)
        else:
            ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=C["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        if note:
            fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=9,
                     color=C["neutral"])
        sp.save(fig, out / f"{stem}_step{k}.svg")


# Hook: share of Black patients among the auto-enrolled (Obermeyer Fig. 1B)
fig_rows = [("Ranked by predicted cost", calc.HOOK_COST, "neutral", f"{calc.HOOK_COST}%", 1),
            ("Ranked by health", calc.HOOK_HEALTH, "danger",
             f"{calc.HOOK_HEALTH}%   {calc.f(calc.HOOK_MULT)}× as many", 3)]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    for i, (label, v, ck, txt, first) in enumerate(fig_rows):
        y = 1 - i
        on = k >= first
        ax.barh(y, v, color=C[ck], height=0.6, alpha=1.0 if on else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
        ax.text(v + 0.8, y, txt, va="center", fontsize=12, fontweight="bold",
                color=C[ck], alpha=1.0 if on else 0.0)
    # the hidden bar's slot: a dashed outline and ??? until the reveal
    show_q = k == 2
    ax.barh(0, calc.HOOK_HEALTH, height=0.6, fill=False, ls="--", lw=1.2,
            edgecolor=C["neutral"], alpha=1.0 if show_q else 0.0)
    ax.text(23, 0, "???", ha="center", va="center", fontsize=16,
            fontweight="bold", color=C["danger"], alpha=1.0 if show_q else 0.0)
    ax.set_yticks([])
    ax.set_xlim(0, 75)
    ax.set_ylim(-0.6, 1.6)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Black patients among those auto-enrolled (%)")
    sp.save(fig, out / f"hook_share_step{k}.svg")

# Segment 1 measured: the training label decides who is selected (Table 2)
bar_steps("label_choice", [
    ("Label: total cost", calc.LABEL_COST, "neutral", f"{calc.LABEL_COST}%", 1),
    ("Label: avoidable cost", calc.LABEL_AVOID, "compute", f"{calc.LABEL_AVOID}%", 2),
    ("Label: active chronic illnesses", calc.LABEL_CHRONIC, "data",
     f"{calc.LABEL_CHRONIC}%   {calc.f(calc.LABEL_MULT)}× the cost label", 3),
], 3, (5.8, 2.2), 40, "Black patients among the top 3% of predicted risk (%)")

# Proxy and goal (illustrative, after Systems Perspective 15.2)
t = np.linspace(0, 1, 200)
knee = 0.5
proxy = 1.0 + 0.2 * t
goal = np.where(t <= knee, 1.0 + 0.2 * t, 1.1 - 0.3 * (t - knee))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    m = t <= knee if k == 1 else np.ones_like(t, dtype=bool)
    ax.plot(t[m], proxy[m], color=C["memory"], lw=3.5)
    ax.plot(t[m], goal[m], color=C["data"], lw=3.5, ls=(0, (6, 2)) if k == 1 else "-")
    ax.plot([knee], [1.1], "o", ms=11, color=C["danger"], zorder=5)
    ax.annotate("Knee", (knee, 1.1), xytext=(-8, 12), textcoords="offset points",
                ha="right", fontsize=12, fontweight="bold", color=C["danger"])
    ax.text(0.03, 1.22, "Clicks and satisfaction\nrise together", fontsize=11,
            color=C["text"], va="bottom")
    if k == 2:
        ax.fill_between(t[t >= knee], goal[t >= knee], proxy[t >= knee],
                        color=C["danger_l"], alpha=0.8)
        ax.text(1.02, 1.20, "Clicks  1.20", color=C["memory"], fontsize=12,
                fontweight="bold", va="center", transform=ax.transData)
        ax.text(1.02, 0.95, "Satisfaction  0.95", color=C["data"], fontsize=12,
                fontweight="bold", va="center")
        ax.text(0.80, 1.09, f"Gap {calc.f(calc.PROXY_GAP, 2)}", color=C["danger"], fontsize=12,
                fontweight="bold", ha="center")
    else:
        ax.text(1.02, 1.20, "Clicks  1.20", color=C["memory"], fontsize=12,
                fontweight="bold", va="center", alpha=0.0)
        ax.text(1.02, 0.95, "Satisfaction  0.95", color=C["data"], fontsize=12,
                fontweight="bold", va="center", alpha=0.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.85, 1.32)
    ax.set_xticks([])
    ax.set_yticks([1.0])
    ax.set_yticklabels(["1.00"])
    ax.set_xlabel("Optimization effort on clicks")
    ax.set_ylabel("Normalized value")
    ax.text(0.0, -0.22, "Illustrative (Systems Perspective 15.2)", transform=ax.transAxes,
            fontsize=9, color=C["neutral"])
    sp.save(fig, out / f"proxy_goal_step{k}.svg")

# Gender Shades: one aggregate number, then the groups (Face++)
# drawn for the right column of its frame (about 8.7 by 5.2 cm)
bar_steps("gender_shades", [
    ("All faces (reported)", calc.GS_ALL, "neutral", f"{calc.GS_ALL}%", 1),
    ("Lighter men", calc.GS_LM, "memory", f"{calc.GS_LM}%", 2),
    ("Darker men", calc.GS_DM, "memory", f"{calc.GS_DM}%", 2),
    ("Lighter women", calc.GS_LF, "memory", f"{calc.GS_LF}%", 2),
    ("Darker women", calc.GS_DF, "danger", f"{calc.GS_DF}%  ({calc.f(calc.GS_MULT, 0)}×)", 2),
], 2, (3.45, 2.05), 62, "Face++ gender-classification error (%)")

# Margin of error against the number of examples in one group
n = np.logspace(2, 5, 300)
marg = 1.96 * np.sqrt(0.25 / n) * 100
fig, ax = plt.subplots(figsize=(3.6, 2.0))
ax.plot(n, marg, color=C["memory"], lw=3)
ax.axhline(1.0, color=C["danger"], lw=1.5, ls="--")
ax.text(125, 1.35, "Target: ±1 point", color=C["danger"], fontsize=10, fontweight="bold")
ax.plot([calc.N_ONE_1PT], [1.0], "o", ms=10, color=C["danger"], zorder=5)
ax.annotate(f"{calc.N_ONE_1PT:,.0f}", (calc.N_ONE_1PT, 1.0), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=12, fontweight="bold", color=C["danger"])
ax.set_xscale("log")
ax.set_xlim(100, 1e5)
ax.set_ylim(0, 10)
ax.set_xticks([100, 1000, 10000, 100000])
ax.set_xticklabels(["100", "1,000", "10,000", "100,000"])
ax.set_xlabel("Examples from the group (log scale)")
ax.set_ylabel("95% margin, one group\n(points)")
sp.save(fig, out / "margin_curve.svg")

# Raji and Buolamwini 2019: darker-female error before and after the audit
vendors = [("Face++", 34.5, 4.1), ("IBM", 34.7, 16.97), ("Microsoft", 20.8, 1.52),
           ("Amazon", None, 31.37), ("Kairos", None, 22.50)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    nv = len(vendors)
    for i, (name, before, after) in enumerate(vendors):
        y = nv - 1 - i
        audited = before is not None
        if audited:
            ax.barh(y + 0.19, before, height=0.36, color=C["neutral_l"])
            ax.text(before + 0.6, y + 0.19, f"{before:g}%  May 2017", va="center",
                    fontsize=10, color=C["neutral"], fontweight="bold")
        on = k >= 2
        ck = "danger" if audited else "neutral"
        ax.barh(y - 0.19 if audited else y, after, height=0.36,
                color=C[ck], alpha=1.0 if on else 0.0)
        ax.text(after + 0.6, y - 0.19 if audited else y,
                f"{after:g}%  Aug 2018" + ("" if audited else " (not in first audit)"),
                va="center", fontsize=10, color=C[ck], fontweight="bold",
                alpha=1.0 if on else 0.0)
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=12, fontweight="bold", color=C["text"])
    ax.axhline(1.5, color=C["grid"], lw=1)
    ax.set_yticks([])
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, nv - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Error on darker-skinned women (%)")
    sp.save(fig, out / f"raji_step{k}.svg")

# COMPAS (ProPublica 2016): error rates, then precision and base rates
metrics = [("False positive rate", 44.85, 23.45, 1),
           ("False negative rate", 27.99, 47.72, 1),
           ("Precision (PPV)", 63.0, 59.0, 2),
           ("Base rate (derived)", calc.r(calc.BASE_BLACK_PCT), calc.r(calc.BASE_WHITE_PCT), 2)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    nm = len(metrics)
    for i, (name, b, w, first) in enumerate(metrics):
        y = nm - 1 - i
        on = k >= first
        a = 1.0 if on else 0.0
        ax.barh(y + 0.19, b, height=0.36, color=C["compute"], alpha=a)
        ax.barh(y - 0.19, w, height=0.36, color=C["memory"], alpha=a)
        ax.text(b + 1.0, y + 0.19, f"{b:.0f}%", va="center", fontsize=10,
                color=C["compute"], fontweight="bold", alpha=a)
        ax.text(w + 1.0, y - 0.19, f"{w:.0f}%", va="center", fontsize=10,
                color=C["memory"], fontweight="bold", alpha=a)
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=10, fontweight="bold", color=C["text"], alpha=a)
    ax.text(72, 3.15, "Black", color=C["compute"], fontsize=10, fontweight="bold")
    ax.text(72, 2.70, "White", color=C["memory"], fontsize=10, fontweight="bold")
    ax.axhline(1.5, color=C["grid"], lw=1, alpha=1.0 if k >= 2 else 0.0)
    ax.set_yticks([])
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.6, nm - 0.4)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Percent")
    sp.save(fig, out / f"compas_step{k}.svg")

# Price of fairness, Group B utility per applicant (constructed scenario)
bar_steps("price", [
    ("Before (TPR 60%)", calc.U_BEFORE, "memory", f"${calc.U_BEFORE:,.0f}", 1),
    ("TPR raised to 90%", calc.U_AFTER, "danger", f"${calc.U_AFTER:,.0f}   half the utility", 1),
], 1, (4.4, 1.3), 260, "Group B utility per applicant (US$)")

# Feedback: projected TPR gap by quarter (Napkin Math 15.3, illustrative)
# drawn at its slot (about 13.5 by 5.2 cm); "illustrative" is in the slide credit
q = np.arange(0, 9)
grow = calc.GAP0 * calc.ALPHA ** q
decay = calc.GAP0 * calc.ALPHA_EXPL ** q
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.3, 2.05))
    ax.axhline(calc.GATE, color=C["danger"], lw=1.6, ls="--")
    ax.text(0.1, 4.85, "Release gate: 5 points", color=C["danger"], fontsize=11,
            fontweight="bold", va="top")
    ax.plot(q, grow, "-o", color=C["compute"], lw=3, ms=6)
    ax.plot([calc.K_FAIL], [calc.GAP_AT_FAIL], "o", ms=12, color=C["danger"], zorder=5)
    ax.annotate(f"Quarter {calc.K_FAIL}: {calc.f(calc.GAP_AT_FAIL)}", (calc.K_FAIL, calc.GAP_AT_FAIL), xytext=(-10, 10),
                textcoords="offset points", ha="right", fontsize=11,
                fontweight="bold", color=C["danger"])
    ax.text(8.15, grow[-1], f"factor {calc.f(calc.ALPHA, 2)}", color=C["compute"], fontsize=11,
            fontweight="bold", va="center")
    a = 1.0 if k == 2 else 0.0
    ax.plot(q, decay, "-o", color=C["data"], lw=3, ms=6, alpha=a)
    ax.text(8.15, decay[-1], f"factor {calc.f(calc.ALPHA_EXPL, 2)}\n(randomized\napprovals)", color=C["data"],
            fontsize=11, fontweight="bold", va="center", alpha=a)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7.5)
    ax.set_xlabel("Quarterly retraining cycle")
    ax.set_ylabel("TPR gap (points)")
    sp.save(fig, out / f"feedback_step{k}.svg")

# =============================================================================
# Diagrams (SVG)
# =============================================================================

FONT = 'font-family="Helvetica, Arial, sans-serif"'


class Svg:
    """A tiny SVG builder. Every element carries the first step it appears in;
    write(stem, n) emits stem_step1..n, hidden elements kept but transparent."""

    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, first, s, last=99):
        self.items.append((first, last, s))

    def rect(self, x, y, w, h, fill, stroke, first=1, sw=2, rx=8, dash=None, last=99):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(first, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>', last)

    def text(self, x, y, s, size=22, color=C["text"], weight="normal", anchor="middle",
             first=1, last=99):
        lines = s.split("\n")
        y0 = y - (len(lines) - 1) * size * 0.6
        spans = "".join(
            f'<tspan x="{x}" y="{y0 + i * size * 1.2:.1f}">{ln}</tspan>' for i, ln in enumerate(lines))
        self.add(first, f'<text {FONT} font-size="{size}" fill="{color}" font-weight="{weight}" '
                        f'text-anchor="{anchor}" dominant-baseline="middle">{spans}</text>', last)

    def line(self, x1, y1, x2, y2, color, first=1, sw=2.5, arrow=True, dash=None, last=99):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ""
        self.add(first, f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                        f'stroke-width="{sw}"{d}{m}/>', last)

    def circle(self, x, y, r, fill, stroke, first=1, sw=2, last=99):
        self.add(first, f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" '
                        f'stroke-width="{sw}"/>', last)

    def write(self, stem, steps):
        colors = sorted({c for c in C.values()})
        defs = "".join(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            f'fill="{c}"/></marker>' for c in colors)
        for k in range(1, steps + 1):
            body = []
            for first, last, s in self.items:
                vis = first <= k <= last
                body.append(s if vis else s.replace("<", '<g opacity="0"><', 1) + "</g>")
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                   f'width="{self.w}" height="{self.h}"><defs>{defs}</defs>'
                   f'<rect width="100%" height="100%" fill="white"/>' + "".join(body) + "</svg>")
            name = f"{stem}_step{k}.svg" if steps > 1 else f"{stem}.svg"
            (out / name).write_text(svg)


# Hook mechanism: need, care, cost; the model sees only cost. Drawn for the
# left column of its frame (about 8 by 5 cm), beside Obermeyer Fig. 1A.
g = Svg(520, 330)
for i, (lab, x) in enumerate((("Health\nneed", 5), ("Care\nreceived", 185), ("Cost", 365))):
    g.rect(x, 20, 150, 90, C["memory_l"] if i < 2 else C["compute_l"],
           C["memory"] if i < 2 else C["compute"])
    g.text(x + 75, 65, lab, 24, weight="bold")
for x in (155, 335):
    g.line(x + 3, 65, x + 26, 65, C["neutral"])
g.text(440, 150, "The model's\nlabel", 21, C["compute"], "bold")
g.text(80, 150, "The program's\ngoal", 21, C["memory"], "bold")
# step 2: the access gap
g.rect(5, 200, 510, 120, C["danger_l"], C["danger"], first=2)
g.text(260, 228, "Unequal access: at the same illness,", 22, C["danger"], "bold", first=2)
g.text(260, 260, "$1,801 less is spent per year", 22, C["danger"], "bold", first=2)
g.text(260, 292, "on a Black patient", 22, C["danger"], "bold", first=2)
g.line(170, 196, 170, 72, C["danger"], first=2)
g.write("care_chain", 2)

# Verification against validation
g = Svg(900, 330)
g.text(300, 30, "Verification: built to the spec?", 24, C["text"], "bold")
for i, (a, b) in enumerate((("Latency", "within SLA"), ("Availability", "99.9%"),
                            ("Accuracy", "87% overall"))):
    x = 20 + i * 195
    g.rect(x, 70, 180, 140, C["data_l"], C["data"])
    g.text(x + 90, 115, a, 22, C["text"], "bold")
    g.text(x + 90, 160, b, 24, C["data"], "bold")
    g.circle(x + 155, 92, 9, C["data"], C["data"])
g.text(750, 30, "Validation: right spec?", 24, C["text"], "bold", first=2)
g.rect(620, 70, 260, 140, C["danger_l"], C["danger"], first=2, sw=3)
g.text(750, 115, "Fair hiring?", 22, C["text"], "bold", first=2)
g.text(750, 160, "Downgrades resumes\nwith “women's”", 20, C["danger"], "bold",
       first=2)
g.add(1, f'<line x1="600" y1="55" x2="600" y2="300" stroke="{C["grid"]}" stroke-width="2"/>')
g.text(300, 270, "Every check the team wrote is green.", 22, C["neutral"])
g.text(750, 270, "The check nobody wrote", 22, C["danger"], "bold", first=2)
g.write("verify_validate", 2)

# Fan-out: one upstream change reaches every decision below it
g = Svg(900, 400)
g.rect(10, 145, 260, 125, C["danger_l"], C["danger"], sw=3)
g.text(140, 182, "Form change:", 24, C["danger"], "bold")
g.text(140, 228, "self-employed income\nrecorded differently", 21, C["text"])
ys = [60, 125, 190, 255, 320]
for j, y in enumerate(ys):
    g.line(272, 207, 528, y, C["danger"], sw=2)
    for col in range(4):
        g.circle(560 + col * 75, y, 24, C["danger_l"], C["danger"])
g.text(670, 375, "Approved last week, declined now", 23,
       C["danger"], "bold")
# step 2: the green dashboard that saw nothing
g.rect(10, 5, 260, 125, C["data_l"], C["data"], first=2)
g.text(140, 30, "Dashboard", 22, C["data"], "bold", first=2)
g.text(140, 82, "latency OK\nconfidence high\nuptime OK", 20, C["text"], first=2)
g.write("fanout", 2)

# Release gate: point estimates with confidence intervals, three outcomes
g = Svg(900, 400)
x0, x1 = 260, 660          # axis: 0 to 10 points
def gx(v):
    return x0 + (x1 - x0) * v / 10.0
g.add(1, f'<line x1="{x0}" y1="330" x2="{x1}" y2="330" stroke="{C["neutral"]}" stroke-width="2"/>')
for v in (0, 5, 10):
    g.text(gx(v), 355, f"{v}", 22, C["neutral"])
g.text((x0 + x1) / 2, 388, "TPR gap between groups (points)", 22, C["neutral"])
g.add(1, f'<line x1="{gx(5)}" y1="30" x2="{gx(5)}" y2="330" stroke="{C["danger"]}" '
         f'stroke-width="2.5" stroke-dasharray="8 5"/>')
g.text(gx(5), 18, "Gate: 5 points", 22, C["danger"], "bold")
slices = [("Slice A", 1.8, 0.8, "PASS", "data", 1, 70),
          ("Slice B", 8.0, 1.0, "FAIL", "danger", 1, 145),
          ("Slice C: few labels", 3.5, 3.5, "INSUFFICIENT", "compute", 2, 220),
          ("Slice D", 2.6, 1.0, "PASS", "data", 3, 295)]
for name, v, h, verdict, ck, first, y in slices:
    g.text(10, y, name, 22, C["text"], "bold", "start", first=first)
    g.add(first, f'<line x1="{gx(v - h)}" y1="{y}" x2="{gx(v + h)}" y2="{y}" '
                 f'stroke="{C[ck]}" stroke-width="4"/>')
    for e in (v - h, v + h):
        g.add(first, f'<line x1="{gx(e)}" y1="{y - 10}" x2="{gx(e)}" y2="{y + 10}" '
                     f'stroke="{C[ck]}" stroke-width="3"/>')
    g.circle(gx(v), y, 9, C[ck], C[ck], first=first)
    g.text(685, y, verdict, 22, C[ck], "bold", "start", first=first)
g.write("gate", 3)

# Credit service confusion matrices (constructed; chapter tables 5 and 6)
g = Svg(900, 420)
mats = [("Group A: 10,000", 175, (4500, 500, 1000, 4000)),
        ("Group B: 2,000", 615, (600, 400, 200, 800))]
cw, ch = 135, 62
for title, gx0, (tp, fn, fp, tn) in mats:
    g.text(gx0 + cw, 22, title, 24, C["text"], "bold")
    g.text(gx0 + cw / 2, 55, "Approved", 20, C["neutral"], "bold")
    g.text(gx0 + cw * 1.5, 55, "Rejected", 20, C["neutral"], "bold")
    for r, (lab, vals) in enumerate((("Would repay", (tp, fn)), ("Would default", (fp, tn)))):
        y = 72 + r * ch
        g.text(gx0 - 10, y + ch / 2, lab, 20, C["text"], "bold", "end")
        for c, v in enumerate(vals):
            g.rect(gx0 + c * cw, y, cw - 6, ch - 6, "#F4F4F4", C["neutral_l"], rx=4)
            g.text(gx0 + c * cw + (cw - 6) / 2, y + (ch - 6) / 2, f"{v:,}", 25, C["text"], "bold")
    # highlights: step 2 approved column, step 3 repay row, step 4 default row
    g.rect(gx0 - 4, 68, cw + 2, 2 * ch + 2, "none", C["compute"], first=2, last=2, sw=4)
    g.rect(gx0 - 4, 68, 2 * cw + 2, ch + 2, "none", C["memory"], first=3, last=3, sw=4)
    g.rect(gx0 - 4, 68 + ch, 2 * cw + 2, ch + 2, "none", C["network"], first=4, last=4, sw=4)
rows = [("Approval rate (demographic parity)", "55%", "40%", "15 pts", "compute", 2, 250),
        ("Of those who would repay, share approved (TPR)", "90%", "60%", "30 pts", "memory", 3, 300),
        ("Of those who would default, share approved (FPR)", "20%", "20%", "0 pts", "network", 4, 350)]
g.text(650, 222, "A", 20, C["neutral"], "bold")
g.text(740, 222, "B", 20, C["neutral"], "bold")
g.text(845, 222, "Gap", 20, C["neutral"], "bold")
for lab, a, b, gap, ck, first, y in rows:
    g.text(10, y, lab, 21, C[ck], "bold", "start", first=first)
    g.text(650, y, a, 24, C["text"], "bold", first=first)
    g.text(740, y, b, 24, C["text"], "bold", first=first)
    g.text(845, y, gap, 24, C["danger"] if gap != "0 pts" else C["data"], "bold", first=first)
g.text(450, 400, "Constructed example (chapter credit service). Both groups repay at 50%.",
       18, C["neutral"])
g.write("matrices", 4)

# The censored retraining loop
g = Svg(520, 420)
g.rect(170, 20, 180, 70, C["compute_l"], C["compute"])
g.text(260, 55, "Model decides", 21, C["text"], "bold")
g.rect(20, 165, 200, 70, C["data_l"], C["data"])
g.text(120, 200, "Approved", 21, C["text"], "bold")
g.rect(300, 165, 200, 70, "#F4F4F4", C["neutral"], dash="6 4")
g.text(400, 200, "Denied", 21, C["neutral"], "bold")
g.rect(20, 310, 200, 80, C["data_l"], C["data"])
g.text(120, 350, "Repayment\nobserved", 20, C["text"], "bold")
g.text(400, 340, "No label:\nnever observed", 20, C["danger"], "bold")
g.line(230, 92, 140, 160, C["neutral"])
g.line(290, 92, 380, 160, C["neutral"])
g.line(120, 237, 120, 305, C["data"])
g.line(400, 237, 400, 305, C["danger"], dash="6 5")
g.line(222, 340, 262, 340, C["data"], arrow=False)
g.line(262, 340, 262, 95, C["data"])
g.text(270, 270, "retrain", 18, C["data"], "bold", "start")
g.write("censored_loop", 1)

# Shared threshold on two score distributions (chapter figure 2, regenerated in
# two steps). Left of the line is approved; circles would repay, crosses would
# not. 75 percent overall at the first line (A 3 errors, B 1); 81.25 percent at
# the second (A 0, B 3).
g = Svg(900, 330)
xs = 0.643
a_pos = [(497, "o"), (615, "o"), (697, "o"), (780, "o"),
         (1010, "+"), (1085, "+"), (1170, "+"), (1255, "+")]
b_pos = [(262, "o"), (343, "o"), (425, "o"), (617, "+"), (697, "o"),
         (780, "+"), (862, "+"), (1085, "+")]
for row, pts, ck, y in (("Subgroup A", a_pos, "memory", 95), ("Subgroup B", b_pos, "danger", 200)):
    g.add(1, f'<line x1="150" y1="{y + 32}" x2="890" y2="{y + 32}" stroke="{C["neutral"]}" stroke-width="4"/>')
    g.text(10, y + 32, row, 22, C["text"], "bold", "start")
    for x, m in pts:
        cx = 40 + x * xs
        if m == "o":
            g.circle(cx, y, 16, "none", C[ck], sw=4)
        else:
            g.add(1, f'<path d="M{cx - 16},{y} H{cx + 16} M{cx},{y - 16} V{y + 16}" '
                     f'stroke="{C[ck]}" stroke-width="7"/>')
for x, lab, first, last in ((555, "75%", 1, 99), (945, "81.25%", 2, 99)):
    cx = 40 + x * xs
    g.add(first, f'<line x1="{cx}" y1="55" x2="{cx}" y2="260" stroke="{C["text"]}" '
                 f'stroke-width="3" stroke-dasharray="8 6"/>')
    g.text(cx, 38, f"overall {lab}", 20, C["text"], "bold", first=first)
g.text(450, 290, "Left of a line: approved.  Circle: would repay.  Cross: would default.",
       19, C["neutral"])
g.text(160, 322, "At 75%: A errors 3, B errors 1", 20, C["memory"], "bold", "start", first=1, last=1)
g.text(160, 322, "At 81.25%: A errors 0, B errors 3 (tripled)", 20, C["danger"], "bold", "start", first=2)
g.write("threshold", 2)

# Three places to intervene; during training, an adversary head with a reversed
# gradient (chapter Sec. 15.3.2.1)
g = Svg(560, 470)
stages = [("Before training", "Reweight the\ntraining examples", 1, "data"),
          ("During training", "", 2, "compute"),
          ("After training", "Move Group B's\nthreshold", 3, "danger")]
for i, (stage, body, first, ck) in enumerate(stages):
    y = 10 + i * 155
    g.rect(10, y, 540, 140, C[ck + "_l"], C[ck], first=first, sw=2.5)
    g.text(25, y + 24, stage, 25, C[ck], "bold", "start", first=first)
    if body:
        g.text(280, y + 80, body, 27, C["text"], "bold", first=first)
y = 165
g.rect(25, y + 50, 125, 62, "white", C["neutral"], first=2)
g.text(87, y + 81, "Encoder", 22, C["text"], "bold", first=2)
g.rect(205, y + 36, 140, 44, "white", C["data"], first=2)
g.text(275, y + 58, "Task head", 21, C["text"], "bold", first=2)
g.rect(205, y + 88, 140, 44, "white", C["danger"], first=2)
g.text(275, y + 110, "Adversary", 21, C["danger"], "bold", first=2)
g.line(152, y + 72, 201, y + 60, C["neutral"], first=2)
g.line(152, y + 92, 201, y + 106, C["neutral"], first=2)
g.text(448, y + 58, "repay?", 22, C["data"], "bold", first=2)
g.text(448, y + 104, "group?\ngradient reversed", 19, C["danger"], "bold", first=2)
g.text(280, 448, "Legally restricted in US credit", 22, C["danger"], "bold", first=3)
g.write("interventions", 3)

# Fairness-accuracy frontier, slide variant of the chapter's figure 3
# (hypothetical): A unconstrained at disparity 0.18, B 91 percent accuracy at
# 0.05, C strict equality at 0 with a large drop. Curve drawn through the
# three points as y = 0.85 + 0.1729 x^0.337.
xx = np.linspace(0, 0.18, 200)
yy = calc.frontier(xx)
fig, ax = plt.subplots(figsize=(3.9, 3.0))
ax.plot(xx, yy, color=C["neutral"], lw=2.5, ls="--")
for x, y, ck, lab, off, ha in ((calc.DISP_A, calc.frontier(calc.DISP_A), "danger", "A: unconstrained", (-6, 10), "left"),
                               (calc.DISP_B, calc.frontier(calc.DISP_B), "data", f"B: {calc.frontier(calc.DISP_B)*100:.0f}% accuracy,\ndisparity {calc.DISP_B}", (-12, -36), "right"),
                               (0.0, 0.85, "memory", "C: strict equality,\nlarge drop", (-8, 4), "right")):
    ax.plot([x], [y], "o", ms=11, color=C[ck], zorder=5)
    ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", ha=ha,
                fontsize=11, fontweight="bold", color=C[ck])
ax.set_xlim(0.2, -0.01)
ax.set_ylim(0.83, 0.97)
ax.set_xlabel("Disparity (lower is fairer)")
ax.set_ylabel("Accuracy")
ax.text(0.0, -0.3, "Hypothetical (chapter figure 3)", transform=ax.transAxes, fontsize=9,
        color=C["neutral"])
sp.save(fig, out / "frontier.svg")

# Next lecture teaser: the architecture-search estimate (Lecture 15b's hook)
fig, ax = plt.subplots(figsize=(4.6, 1.9))
for i, (label, v, ck, txt) in enumerate((("One car, lifetime", 57, "neutral", "57 t"),
                                         ("Search, estimated", 284, "compute", "284 t"),
                                         ("Search, measured", 3.2, "data", "???"))):
    y = 2 - i
    ax.barh(y, v, color=C[ck], height=0.62, alpha=0.0 if txt == "???" else 1.0)
    ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha="right",
            va="center", fontsize=12, fontweight="bold", color=C["text"])
    ax.text(v * 1.2 if txt != "???" else 8, y, txt, va="center", fontsize=13,
            fontweight="bold", color=C["danger"] if txt == "???" else C[ck])
ax.set_xscale("log")
ax.set_xlim(1, 3000)
ax.set_ylim(-0.6, 2.6)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel("Tonnes of CO₂e (log scale)")
sp.save(fig, out / "next_part2.svg")

print("figures written")
