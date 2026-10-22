#!/usr/bin/env python3
"""Figures for Lecture 15 (Responsible Engineering).

Plots use the shared slideplot helper; diagrams are written as SVG with a
small builder below, one file per build step, every step on the same canvas
so nothing moves. Every value comes from calcs.py (imported below), which
reads MLSysIM: Literature.Fairness (Obermeyer et al. 2019, Gender Shades,
ProPublica's COMPAS analysis), Literature.Series.GenderShadesAudit (Raji and
Buolamwini 2019), Literature.TrainingRuns (Strubell et al. 2019), and the
fairness Scenarios.* (credit service, its price and feedback loop, the
evaluation budget, the shared threshold, the frontier, the proxy and goal).
Figure-only values are layout: coordinates, axis limits, and the illustrative
gate slices A, B, and D (slice C's interval is the evaluation budget's).
The verification panel quotes the chapter quiz's recruiting-tool figures.

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
from deckvals import val, num  # noqa: E402

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
fig_rows = [("Ranked by predicted cost", num("hook_cost"), "neutral", f"{val('hook_cost')}%", 1),
            ("Ranked by health", num("hook_health"), "danger",
             f"{val('hook_health')}%   {val('hook_mult')}× as many", 3)]
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
    ax.barh(0, num("hook_health"), height=0.6, fill=False, ls="--", lw=1.2,
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
    ("Label: total cost", num("label_cost"), "neutral", f"{val('label_cost')}%", 1),
    ("Label: avoidable cost", num("label_avoid"), "compute", f"{val('label_avoid')}%", 2),
    ("Label: active chronic illnesses", num("label_chronic"), "data",
     f"{val('label_chronic')}%   {val('label_mult')}× the cost label", 3),
], 3, (5.8, 2.2), 40, f"Black patients among the top {val('top_pct')}% of predicted risk (%)")

# Proxy and goal (illustrative, after Systems Perspective 15.2)
t = np.linspace(0, 1, 200)
knee = 0.5                 # layout: where the curves part
P = calc.PROXY
proxy = P.start + (P.proxy_end - P.start) * t
at_knee = P.start + (P.proxy_end - P.start) * knee
goal = np.where(t <= knee, proxy, at_knee + (P.goal_end - at_knee) * (t - knee) / (1 - knee))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    m = t <= knee if k == 1 else np.ones_like(t, dtype=bool)
    ax.plot(t[m], proxy[m], color=C["memory"], lw=3.5)
    ax.plot(t[m], goal[m], color=C["data"], lw=3.5, ls=(0, (6, 2)) if k == 1 else "-")
    ax.plot([knee], [at_knee], "o", ms=11, color=C["danger"], zorder=5)
    ax.annotate("Knee", (knee, at_knee), xytext=(-8, 12), textcoords="offset points",
                ha="right", fontsize=12, fontweight="bold", color=C["danger"])
    ax.text(0.03, 1.22, "Clicks and satisfaction\nrise together", fontsize=11,
            color=C["text"], va="bottom")
    if k == 2:
        ax.fill_between(t[t >= knee], goal[t >= knee], proxy[t >= knee],
                        color=C["danger_l"], alpha=0.8)
        ax.text(1.02, P.proxy_end, f"Clicks  {val('clicks_end')}", color=C["memory"], fontsize=12,
                fontweight="bold", va="center", transform=ax.transData)
        ax.text(1.02, P.goal_end, f"Satisfaction  {val('sat_end')}", color=C["data"], fontsize=12,
                fontweight="bold", va="center")
        ax.text(0.80, 1.09, f"Gap {val('proxy_gap')}", color=C["danger"], fontsize=12,
                fontweight="bold", ha="center")
    else:
        ax.text(1.02, P.proxy_end, f"Clicks  {val('clicks_end')}", color=C["memory"], fontsize=12,
                fontweight="bold", va="center", alpha=0.0)
        ax.text(1.02, P.goal_end, f"Satisfaction  {val('sat_end')}", color=C["data"], fontsize=12,
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
    ("All faces (reported)", num("gs_all"), "neutral", f"{val('gs_all')}%", 1),
    ("Lighter men", num("gs_lm"), "memory", f"{val('gs_lm')}%", 2),
    ("Darker men", num("gs_dm"), "memory", f"{val('gs_dm')}%", 2),
    ("Lighter women", num("gs_lf"), "memory", f"{val('gs_lf')}%", 2),
    ("Darker women", num("gs_df"), "danger", f"{val('gs_df')}%  ({val('gs_mult')}×)", 2),
], 2, (3.45, 2.05), 62, "Face++ gender-classification error (%)")

# Margin of error against the number of examples in one group
n = np.logspace(2, 5, 300)
B = calc.BUDGET
marg = B.z * np.sqrt(B.variance / n) * 100
target = B.half_width * 100
fig, ax = plt.subplots(figsize=(3.6, 2.0))
ax.plot(n, marg, color=C["memory"], lw=3)
ax.axhline(target, color=C["danger"], lw=1.5, ls="--")
ax.text(125, target + 0.35, f"Target: ±{val('margin_pts')} point", color=C["danger"], fontsize=10, fontweight="bold")
ax.plot([B.samples_needed], [target], "o", ms=10, color=C["danger"], zorder=5)
ax.annotate(val("n_one"), (B.samples_needed, target), xytext=(0, 12), textcoords="offset points",
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
vendors = [("Face++", num("raji_fpp_may"), num("raji_fpp_aug")), ("IBM", num("raji_ibm_may"), num("raji_ibm_aug")),
           ("Microsoft", num("raji_msft_may"), num("raji_msft_aug")),
           ("Amazon", None, num("raji_amzn_aug")), ("Kairos", None, num("raji_kai_aug"))]
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
metrics = [("False positive rate", num("c_blk_fpr_pub"), num("c_wht_fpr_pub"), 1),
           ("False negative rate", num("c_blk_fnr_pct"), num("c_wht_fnr_pct"), 1),
           ("Precision (PPV)", num("c_blk_ppv_pct"), num("c_wht_ppv_pct"), 2),
           ("Base rate (derived)", 100 * calc.COMPAS.base_rate("Black"), 100 * calc.COMPAS.base_rate("White"), 2)]
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
    (f"Before (TPR {val('pr_tpr0_pct')}%)", num("pr_before"), "memory", f"${val('pr_before')}", 1),
    (f"TPR raised to {val('pr_tpr1_pct')}%", num("pr_after"), "danger", f"${val('pr_after')}   half the utility", 1),
], 1, (4.4, 1.3), 260, "Group B utility per applicant (US$)")

# Feedback: projected TPR gap by quarter (Napkin Math 15.3, illustrative)
# drawn at its slot (about 13.5 by 5.2 cm); "illustrative" is in the slide credit
L = calc.LOOP
q = np.arange(0, 9)
grow = np.array([L.gap_at(k) for k in q])
decay = np.array([L.gap_at(k, L.sampled_factor) for k in q])
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.3, 2.05))
    ax.axhline(L.gate, color=C["danger"], lw=1.6, ls="--")
    ax.text(0.1, L.gate - 0.15, f"Release gate: {val('gate_pts')} points", color=C["danger"], fontsize=11,
            fontweight="bold", va="top")
    ax.plot(q, grow, "-o", color=C["compute"], lw=3, ms=6)
    ax.plot([L.breach_cycle], [L.gap_at_breach], "o", ms=12, color=C["danger"], zorder=5)
    ax.annotate(f"Quarter {val('fb_fail_q')}: {val('fb_gap_fail')}", (L.breach_cycle, L.gap_at_breach), xytext=(-10, 10),
                textcoords="offset points", ha="right", fontsize=11,
                fontweight="bold", color=C["danger"])
    ax.text(8.15, grow[-1], f"factor {val('fb_alpha')}", color=C["compute"], fontsize=11,
            fontweight="bold", va="center")
    a = 1.0 if k == 2 else 0.0
    ax.plot(q, decay, "-o", color=C["data"], lw=3, ms=6, alpha=a)
    ax.text(8.15, decay[-1], f"factor {val('fb_alpha_expl')}\n(randomized\napprovals)", color=C["data"],
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
g.text(260, 260, f"${val('spend_gap')} less is spent per year", 22, C["danger"], "bold", first=2)
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
GATE, SLICE_C = calc.LOOP.gate, 100 * calc.BUDGET.gap_half_width
def gx(v):
    return x0 + (x1 - x0) * v / 10.0
g.add(1, f'<line x1="{x0}" y1="330" x2="{x1}" y2="330" stroke="{C["neutral"]}" stroke-width="2"/>')
for v in (0, GATE, 10):
    g.text(gx(v), 355, f"{v:.0f}", 22, C["neutral"])
g.text((x0 + x1) / 2, 388, "TPR gap between groups (points)", 22, C["neutral"])
g.add(1, f'<line x1="{gx(GATE)}" y1="30" x2="{gx(GATE)}" y2="330" stroke="{C["danger"]}" '
         f'stroke-width="2.5" stroke-dasharray="8 5"/>')
g.text(gx(GATE), 18, f"Gate: {val('gate_pts')} points", 22, C["danger"], "bold")
slices = [("Slice A", 1.8, 0.8, "PASS", "data", 1, 70),          # illustrative
          ("Slice B", 8.0, 1.0, "FAIL", "danger", 1, 145),       # illustrative
          ("Slice C: few labels", SLICE_C, SLICE_C, "INSUFFICIENT", "compute", 2, 220),
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
cells = lambda grp: tuple(int(calc.CREDIT.groups[grp][k]) for k in ("tp", "fn", "fp", "tn"))  # noqa: E731
mats = [(f"Group A: {val('credit_a_n')}", 175, cells("A")),
        (f"Group B: {val('credit_b_n')}", 615, cells("B"))]
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
rows = [("Approval rate (demographic parity)", f"{val('appr_a')}%", f"{val('appr_b')}%", f"{val('appr_gap')} pts", "compute", 2, 250),
        ("Of those who would repay, share approved (TPR)", f"{val('tpr_a')}%", f"{val('tpr_b')}%", f"{val('tpr_gap')} pts", "memory", 3, 300),
        ("Of those who would default, share approved (FPR)", f"{val('fpr_a')}%", f"{val('fpr_b')}%", f"{val('fpr_gap')} pts", "network", 4, 350)]
g.text(650, 222, "A", 20, C["neutral"], "bold")
g.text(740, 222, "B", 20, C["neutral"], "bold")
g.text(845, 222, "Gap", 20, C["neutral"], "bold")
for lab, a, b, gap, ck, first, y in rows:
    g.text(10, y, lab, 21, C[ck], "bold", "start", first=first)
    g.text(650, y, a, 24, C["text"], "bold", first=first)
    g.text(740, y, b, 24, C["text"], "bold", first=first)
    g.text(845, y, gap, 24, C["danger"] if not gap.startswith("0 ") else C["data"], "bold", first=first)
g.text(450, 400, f"Constructed example (chapter credit service). Both groups repay at {val('credit_base_pct')}%.",
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
# two steps) from Scenarios.SharedThresholdSixteenApplicants: each applicant's
# position on the axis is 1 minus its score, and left of a line is approved.
# Circles would repay, crosses would not.
T = calc.THRESH
x_lo, x_hi = 150, 890          # layout: the axis in SVG units
def ax_x(score):
    return x_lo + (x_hi - x_lo) * (1 - score)
g = Svg(900, 330)
for grp, ck, y in (("A", "memory", 95), ("B", "danger", 200)):
    g.add(1, f'<line x1="{x_lo}" y1="{y + 32}" x2="{x_hi}" y2="{y + 32}" stroke="{C["neutral"]}" stroke-width="4"/>')
    g.text(10, y + 32, f"Subgroup {grp}", 22, C["text"], "bold", "start")
    for score in T.scores[grp]["qualified"]:
        g.circle(ax_x(score), y, 16, "none", C[ck], sw=4)
    for score in T.scores[grp]["default"]:
        cx = ax_x(score)
        g.add(1, f'<path d="M{cx - 16},{y} H{cx + 16} M{cx},{y - 16} V{y + 16}" '
                 f'stroke="{C[ck]}" stroke-width="7"/>')
for thr, key, first in ((T.uniform_threshold, "acc_line1", 1), (T.alternative_threshold, "acc_line2", 2)):
    cx = ax_x(thr)
    g.add(first, f'<line x1="{cx}" y1="55" x2="{cx}" y2="260" stroke="{C["text"]}" '
                 f'stroke-width="3" stroke-dasharray="8 6"/>')
    g.text(cx, 38, f"overall {val(key)}%", 20, C["text"], "bold", first=first)
g.text(450, 290, "Left of a line: approved.  Circle: would repay.  Cross: would default.",
       19, C["neutral"])
g.text(160, 322, f"At {val('acc_line1')}%: A errors {val('err_a1')}, B errors {val('err_b1')}", 20, C["memory"], "bold", "start", first=1, last=1)
g.text(160, 322, f"At {val('acc_line2')}%: A errors {T.errors('A', T.alternative_threshold)}, B errors {val('err_b2')} (tripled)", 20, C["danger"], "bold", "start", first=2)
g.write("threshold", 2)

# Three places to intervene; during training, an adversary head with a reversed
# gradient (chapter Sec. 15.3.2.1)
g = Svg(560, 470)
# Before (short), during (tall: the adversary diagram needs room), after.
g.rect(10, 10, 540, 105, C["data_l"], C["data"], first=1, sw=2.5)
g.text(25, 34, "Before training", 25, C["data"], "bold", "start", first=1)
g.text(280, 80, "Reweight the training examples", 27, C["text"], "bold", first=1)
y = 125
g.rect(10, y, 540, 205, C["compute_l"], C["compute"], first=2, sw=2.5)
g.text(25, y + 24, "During training", 25, C["compute"], "bold", "start", first=2)
g.rect(25, y + 72, 135, 76, "white", C["neutral"], first=2)
g.text(92, y + 110, "Encoder", 26, C["text"], "bold", first=2)
g.rect(220, y + 52, 165, 52, "white", C["data"], first=2)
g.text(302, y + 78, "Task head", 24, C["text"], "bold", first=2)
g.rect(220, y + 120, 165, 52, "white", C["danger"], first=2)
g.text(302, y + 146, "Adversary", 24, C["danger"], "bold", first=2)
g.line(162, y + 100, 216, y + 80, C["neutral"], first=2)
g.line(162, y + 120, 216, y + 142, C["neutral"], first=2)
g.text(470, y + 78, "repay?", 25, C["data"], "bold", first=2)
g.text(470, y + 146, "group?", 25, C["danger"], "bold", first=2)
g.text(302, y + 190, "gradient reversed into the encoder", 21, C["danger"], "bold", first=2)
y = 340
g.rect(10, y, 540, 120, C["danger_l"], C["danger"], first=3, sw=2.5)
g.text(25, y + 24, "After training", 25, C["danger"], "bold", "start", first=3)
g.text(280, y + 62, "Move Group B's threshold", 27, C["text"], "bold", first=3)
g.text(280, y + 98, "Legally restricted in US credit", 22, C["danger"], "bold", first=3)
g.write("interventions", 3)

# Fairness-accuracy frontier, slide variant of the chapter's figure 3
# (hypothetical), drawn from Scenarios.FairnessAccuracyFrontierSketch: A
# unconstrained, B the compromise, C strict equality with a large drop.
FR = calc.FRONTIER
xx = np.linspace(0, FR.candidates["A"], 200)
yy = FR.accuracy(xx)
fig, ax = plt.subplots(figsize=(3.9, 3.0))
ax.plot(xx, yy, color=C["neutral"], lw=2.5, ls="--")
(xa, ya), (xb, yb), (xc, yc) = FR.point("A"), FR.point("B"), FR.point("C")
for x, y, ck, lab, off, ha in ((xa, ya, "danger", "A: unconstrained", (-6, 10), "left"),
                               (xb, yb, "data", f"B: {val('acc_b_pct')}% accuracy,\ndisparity {val('disp_b')}", (-12, -36), "right"),
                               (xc, yc, "memory", "C: strict equality,\nlarge drop", (-28, -2), "right")):
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
for i, (label, v, ck, txt) in enumerate((("One car, lifetime", num("car_t"), "neutral", f"{val('car_t')} t"),
                                         ("Search, estimated", num("nas_t"), "compute", f"{val('nas_t')} t"),
                                         ("Search, measured", 1, "data", "???"))):  # hidden until Lecture 15b
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
