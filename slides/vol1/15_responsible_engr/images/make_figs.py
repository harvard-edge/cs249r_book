#!/usr/bin/env python3
"""Figures for Lecture 15 (Responsible Engineering).

Plots use the shared slideplot helper; diagrams are written as SVG with a
small builder below, one file per build step, every step on the same canvas
so nothing moves. Every value comes from calcs.py (imported below), which
reads MLSysIM: Literature.Fairness (Obermeyer et al. 2019, Gender Shades,
ProPublica's COMPAS analysis), vol2/16_responsible_ai/data/gender_shades_audit.csv (Raji and
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
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
plt = sp.plt
C = sp.C
out = Path(__file__).parent

def bar_steps(stem, rows, steps, size, xmax, xlabel, note=None, vlines=None, log=False, xmin=None):
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
            ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, color=C['text'], fontweight='bold', alpha=1.0 if on else 0.0)
            x = v * 1.12 if log else v + xmax * 0.012
            ax.text(x, y, txt, va='center', fontsize=12, color=C[ck], fontweight='bold', alpha=1.0 if on else 0.0)
        for x, label, ck, first in vlines or []:
            on = k >= first
            ax.axvline(x, color=C[ck], lw=1.6, ls='--', zorder=1, alpha=1.0 if on else 0.0)
            ax.text(x, n - 0.45, ' ' + label, color=C[ck], fontsize=10, fontweight='bold', va='bottom', ha='left', alpha=1.0 if on else 0.0)
        ax.set_yticks([])
        if log:
            ax.set_xscale('log')
            ax.set_xlim(xmin, xmax)
        else:
            ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, n - 0.4)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=C['grid'], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        if note:
            fig.text(0.99, 0.01, note, ha='right', va='bottom', fontsize=9, color=C['neutral'])
        sp.save(fig, out / f'{stem}_step{k}.svg')
fig_rows = [('Ranked by predicted cost', 17.7, 'neutral', '17.7%', 1), ('Ranked by health', 46.5, 'danger', '46.5%   2.6× as many', 3)]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    for i, (label, v, ck, txt, first) in enumerate(fig_rows):
        y = 1 - i
        on = k >= first
        ax.barh(y, v, color=C[ck], height=0.6, alpha=1.0 if on else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
        ax.text(v + 0.8, y, txt, va='center', fontsize=12, fontweight='bold', color=C[ck], alpha=1.0 if on else 0.0)
    show_q = k == 2
    ax.barh(0, 46.5, height=0.6, fill=False, ls='--', lw=1.2, edgecolor=C['neutral'], alpha=1.0 if show_q else 0.0)
    ax.text(23, 0, '???', ha='center', va='center', fontsize=16, fontweight='bold', color=C['danger'], alpha=1.0 if show_q else 0.0)
    ax.set_yticks([])
    ax.set_xlim(0, 75)
    ax.set_ylim(-0.6, 1.6)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Black patients among those auto-enrolled (%)')
    sp.save(fig, out / f'hook_share_step{k}.svg')
bar_steps('label_choice', [('Label: total cost', 14.1, 'neutral', '14.1%', 1), ('Label: avoidable cost', 21.0, 'compute', '21.0%', 2), ('Label: active chronic illnesses', 26.7, 'data', '26.7%   1.9× the cost label', 3)], 3, (5.8, 2.2), 40, 'Black patients among the top 3% of predicted risk (%)')
t = np.linspace(0, 1, 200)
knee = 0.5
proxy = 1.0 + (1.2 - 1.0) * t
at_knee = 1.0 + (1.2 - 1.0) * knee
goal = np.where(t <= knee, proxy, at_knee + (0.95 - at_knee) * (t - knee) / (1 - knee))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    m = t <= knee if k == 1 else np.ones_like(t, dtype=bool)
    ax.plot(t[m], proxy[m], color=C['memory'], lw=3.5)
    ax.plot(t[m], goal[m], color=C['data'], lw=3.5, ls=(0, (6, 2)) if k == 1 else '-')
    ax.plot([knee], [at_knee], 'o', ms=11, color=C['danger'], zorder=5)
    ax.annotate('Knee', (knee, at_knee), xytext=(-8, 12), textcoords='offset points', ha='right', fontsize=12, fontweight='bold', color=C['danger'])
    ax.text(0.03, 1.22, 'Clicks and satisfaction\nrise together', fontsize=11, color=C['text'], va='bottom')
    if k == 2:
        ax.fill_between(t[t >= knee], goal[t >= knee], proxy[t >= knee], color=C['danger_l'], alpha=0.8)
        ax.text(1.02, 1.2, 'Clicks  1.20', color=C['memory'], fontsize=12, fontweight='bold', va='center', transform=ax.transData)
        ax.text(1.02, 0.95, 'Satisfaction  0.95', color=C['data'], fontsize=12, fontweight='bold', va='center')
        ax.text(0.8, 1.09, 'Gap 0.25', color=C['danger'], fontsize=12, fontweight='bold', ha='center')
    else:
        ax.text(1.02, 1.2, 'Clicks  1.20', color=C['memory'], fontsize=12, fontweight='bold', va='center', alpha=0.0)
        ax.text(1.02, 0.95, 'Satisfaction  0.95', color=C['data'], fontsize=12, fontweight='bold', va='center', alpha=0.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.85, 1.32)
    ax.set_xticks([])
    ax.set_yticks([1.0])
    ax.set_yticklabels(['1.00'])
    ax.set_xlabel('Optimization effort on clicks')
    ax.set_ylabel('Normalized value')
    ax.text(0.0, -0.22, 'Illustrative (Systems Perspective 15.2)', transform=ax.transAxes, fontsize=9, color=C['neutral'])
    sp.save(fig, out / f'proxy_goal_step{k}.svg')
bar_steps('gender_shades', [('All faces (reported)', 10.0, 'neutral', '10.0%', 1), ('Lighter men', 0.8, 'memory', '0.8%', 2), ('Darker men', 0.7, 'memory', '0.7%', 2), ('Lighter women', 9.8, 'memory', '9.8%', 2), ('Darker women', 34.5, 'danger', '34.5%  (43×)', 2)], 2, (3.45, 2.05), 62, 'Face++ gender-classification error (%)')
n = np.logspace(2, 5, 300)
marg = 1.959963985 * np.sqrt(0.25 / n) * 100
target = 0.01 * 100
fig, ax = plt.subplots(figsize=(3.6, 2.0))
ax.plot(n, marg, color=C['memory'], lw=3)
ax.axhline(target, color=C['danger'], lw=1.5, ls='--')
ax.text(125, target + 0.35, 'Target: ±1 point', color=C['danger'], fontsize=10, fontweight='bold')
ax.plot([9604], [target], 'o', ms=10, color=C['danger'], zorder=5)
ax.annotate('9,604', (9604, target), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=12, fontweight='bold', color=C['danger'])
ax.set_xscale('log')
ax.set_xlim(100, 100000.0)
ax.set_ylim(0, 10)
ax.set_xticks([100, 1000, 10000, 100000])
ax.set_xticklabels(['100', '1,000', '10,000', '100,000'])
ax.set_xlabel('Examples from the group (log scale)')
ax.set_ylabel('95% margin, one group\n(points)')
sp.save(fig, out / 'margin_curve.svg')
vendors = [('Face++', 34.5, 4.1), ('IBM', 34.7, 16.97), ('Microsoft', 20.8, 1.52), ('Amazon', None, 31.37), ('Kairos', None, 22.5)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    nv = len(vendors)
    for i, (name, before, after) in enumerate(vendors):
        y = nv - 1 - i
        audited = before is not None
        if audited:
            ax.barh(y + 0.19, before, height=0.36, color=C['neutral_l'])
            ax.text(before + 0.6, y + 0.19, f'{before:g}%  May 2017', va='center', fontsize=10, color=C['neutral'], fontweight='bold')
        on = k >= 2
        ck = 'danger' if audited else 'neutral'
        ax.barh(y - 0.19 if audited else y, after, height=0.36, color=C[ck], alpha=1.0 if on else 0.0)
        ax.text(after + 0.6, y - 0.19 if audited else y, f'{after:g}%  Aug 2018' + ('' if audited else ' (not in first audit)'), va='center', fontsize=10, color=C[ck], fontweight='bold', alpha=1.0 if on else 0.0)
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
    ax.axhline(1.5, color=C['grid'], lw=1)
    ax.set_yticks([])
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, nv - 0.4)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Error on darker-skinned women (%)')
    sp.save(fig, out / f'raji_step{k}.svg')
metrics = [('False positive rate', 44.85, 23.45, 1), ('False negative rate', 27.99, 47.72, 1), ('Precision (PPV)', 63.0, 59.0, 2), ('Base rate (derived)', 100 * 0.5143398268, 100 * 0.3936430318, 2)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    nm = len(metrics)
    for i, (name, b, w, first) in enumerate(metrics):
        y = nm - 1 - i
        on = k >= first
        a = 1.0 if on else 0.0
        ax.barh(y + 0.19, b, height=0.36, color=C['compute'], alpha=a)
        ax.barh(y - 0.19, w, height=0.36, color=C['memory'], alpha=a)
        ax.text(b + 1.0, y + 0.19, f'{b:.0f}%', va='center', fontsize=10, color=C['compute'], fontweight='bold', alpha=a)
        ax.text(w + 1.0, y - 0.19, f'{w:.0f}%', va='center', fontsize=10, color=C['memory'], fontweight='bold', alpha=a)
        ax.text(-0.01, y, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=10, fontweight='bold', color=C['text'], alpha=a)
    ax.text(72, 3.15, 'Black', color=C['compute'], fontsize=10, fontweight='bold')
    ax.text(72, 2.7, 'White', color=C['memory'], fontsize=10, fontweight='bold')
    ax.axhline(1.5, color=C['grid'], lw=1, alpha=1.0 if k >= 2 else 0.0)
    ax.set_yticks([])
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.6, nm - 0.4)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel('Percent')
    sp.save(fig, out / f'compas_step{k}.svg')
bar_steps('price', [('Before (TPR 60%)', 150.0, 'memory', '$150', 1), ('TPR raised to 90%', 75.0, 'danger', '$75   half the utility', 1)], 1, (4.4, 1.3), 260, 'Group B utility per applicant (US$)')
q = np.arange(0, 9)
grow = np.array([3.0, 3.3, 3.63, 3.993, 4.3923, 4.83153, 5.314683, 5.8461513, 6.43076643])
decay = np.array([3.0, 2.7, 2.43, 2.187, 1.9683, 1.77147, 1.594323, 1.4348907, 1.29140163])
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.3, 2.05))
    ax.axhline(5.0, color=C['danger'], lw=1.6, ls='--')
    ax.text(0.1, 5.0 - 0.15, 'Release gate: 5 points', color=C['danger'], fontsize=11, fontweight='bold', va='top')
    ax.plot(q, grow, '-o', color=C['compute'], lw=3, ms=6)
    ax.plot([6], [5.314683], 'o', ms=12, color=C['danger'], zorder=5)
    ax.annotate('Quarter 6: 5.3', (6, 5.314683), xytext=(-10, 10), textcoords='offset points', ha='right', fontsize=11, fontweight='bold', color=C['danger'])
    ax.text(8.15, grow[-1], 'factor 1.10', color=C['compute'], fontsize=11, fontweight='bold', va='center')
    a = 1.0 if k == 2 else 0.0
    ax.plot(q, decay, '-o', color=C['data'], lw=3, ms=6, alpha=a)
    ax.text(8.15, decay[-1], 'factor 0.90\n(randomized\napprovals)', color=C['data'], fontsize=11, fontweight='bold', va='center', alpha=a)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7.5)
    ax.set_xlabel('Quarterly retraining cycle')
    ax.set_ylabel('TPR gap (points)')
    sp.save(fig, out / f'feedback_step{k}.svg')
FONT = 'font-family="Helvetica, Arial, sans-serif"'

class Svg:
    """A tiny SVG builder. Every element carries the first step it appears in;
    write(stem, n) emits stem_step1..n, hidden elements kept but transparent."""

    def __init__(self, w, h):
        self.w, self.h, self.items = (w, h, [])

    def add(self, first, s, last=99):
        self.items.append((first, last, s))

    def rect(self, x, y, w, h, fill, stroke, first=1, sw=2, rx=8, dash=None, last=99):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(first, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>', last)

    def text(self, x, y, s, size=22, color=C['text'], weight='normal', anchor='middle', first=1, last=99):
        lines = s.split('\n')
        y0 = y - (len(lines) - 1) * size * 0.6
        spans = ''.join((f'<tspan x="{x}" y="{y0 + i * size * 1.2:.1f}">{ln}</tspan>' for i, ln in enumerate(lines)))
        self.add(first, f'<text {FONT} font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="middle">{spans}</text>', last)

    def line(self, x1, y1, x2, y2, color, first=1, sw=2.5, arrow=True, dash=None, last=99):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ''
        self.add(first, f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>', last)

    def circle(self, x, y, r, fill, stroke, first=1, sw=2, last=99):
        self.add(first, f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>', last)

    def write(self, stem, steps):
        colors = sorted({c for c in C.values()})
        defs = ''.join((f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in colors))
        for k in range(1, steps + 1):
            body = []
            for first, last, s in self.items:
                vis = first <= k <= last
                body.append(s if vis else s.replace('<', '<g opacity="0"><', 1) + '</g>')
            svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}"><defs>{defs}</defs><rect width="100%" height="100%" fill="white"/>' + ''.join(body) + '</svg>'
            name = f'{stem}_step{k}.svg' if steps > 1 else f'{stem}.svg'
            (out / name).write_text(svg)
g = Svg(520, 330)
for i, (lab, x) in enumerate((('Health\nneed', 5), ('Care\nreceived', 185), ('Cost', 365))):
    g.rect(x, 20, 150, 90, C['memory_l'] if i < 2 else C['compute_l'], C['memory'] if i < 2 else C['compute'])
    g.text(x + 75, 65, lab, 24, weight='bold')
for x in (155, 335):
    g.line(x + 3, 65, x + 26, 65, C['neutral'])
g.text(440, 150, "The model's\nlabel", 21, C['compute'], 'bold')
g.text(80, 150, "The program's\ngoal", 21, C['memory'], 'bold')
g.rect(5, 200, 510, 120, C['danger_l'], C['danger'], first=2)
g.text(260, 228, 'Unequal access: at the same illness,', 22, C['danger'], 'bold', first=2)
g.text(260, 260, '$1,801 less is spent per year', 22, C['danger'], 'bold', first=2)
g.text(260, 292, 'on a Black patient', 22, C['danger'], 'bold', first=2)
g.line(170, 196, 170, 72, C['danger'], first=2)
g.write('care_chain', 2)
g = Svg(900, 330)
g.text(300, 30, 'Verification: built to the spec?', 24, C['text'], 'bold')
for i, (a, b) in enumerate((('Latency', 'within SLA'), ('Availability', '99.9%'), ('Accuracy', '87% overall'))):
    x = 20 + i * 195
    g.rect(x, 70, 180, 140, C['data_l'], C['data'])
    g.text(x + 90, 115, a, 22, C['text'], 'bold')
    g.text(x + 90, 160, b, 24, C['data'], 'bold')
    g.circle(x + 155, 92, 9, C['data'], C['data'])
g.text(750, 30, 'Validation: right spec?', 24, C['text'], 'bold', first=2)
g.rect(620, 70, 260, 140, C['danger_l'], C['danger'], first=2, sw=3)
g.text(750, 115, 'Fair hiring?', 22, C['text'], 'bold', first=2)
g.text(750, 160, "Downgrades resumes\nwith “women's”", 20, C['danger'], 'bold', first=2)
g.add(1, f'''<line x1="600" y1="55" x2="600" y2="300" stroke="{C['grid']}" stroke-width="2"/>''')
g.text(300, 270, 'Every check the team wrote is green.', 22, C['neutral'])
g.text(750, 270, 'The check nobody wrote', 22, C['danger'], 'bold', first=2)
g.write('verify_validate', 2)
g = Svg(900, 400)
g.rect(10, 145, 260, 125, C['danger_l'], C['danger'], sw=3)
g.text(140, 182, 'Form change:', 24, C['danger'], 'bold')
g.text(140, 228, 'self-employed income\nrecorded differently', 21, C['text'])
ys = [60, 125, 190, 255, 320]
for j, y in enumerate(ys):
    g.line(272, 207, 528, y, C['danger'], sw=2)
    for col in range(4):
        g.circle(560 + col * 75, y, 24, C['danger_l'], C['danger'])
g.text(670, 375, 'Approved last week, declined now', 23, C['danger'], 'bold')
g.rect(10, 5, 260, 125, C['data_l'], C['data'], first=2)
g.text(140, 30, 'Dashboard', 22, C['data'], 'bold', first=2)
g.text(140, 82, 'latency OK\nconfidence high\nuptime OK', 20, C['text'], first=2)
g.write('fanout', 2)
g = Svg(900, 400)
x0, x1 = (260, 660)
GATE, SLICE_C = (5.0, 100 * 0.035)

def gx(v):
    return x0 + (x1 - x0) * v / 10.0
g.add(1, f'''<line x1="{x0}" y1="330" x2="{x1}" y2="330" stroke="{C['neutral']}" stroke-width="2"/>''')
for v in (0, GATE, 10):
    g.text(gx(v), 355, f'{v:.0f}', 22, C['neutral'])
g.text((x0 + x1) / 2, 388, 'TPR gap between groups (points)', 22, C['neutral'])
g.add(1, f'''<line x1="{gx(GATE)}" y1="30" x2="{gx(GATE)}" y2="330" stroke="{C['danger']}" stroke-width="2.5" stroke-dasharray="8 5"/>''')
g.text(gx(GATE), 18, 'Gate: 5 points', 22, C['danger'], 'bold')
slices = [('Slice A', 1.8, 0.8, 'PASS', 'data', 1, 70), ('Slice B', 8.0, 1.0, 'FAIL', 'danger', 1, 145), ('Slice C: few labels', SLICE_C, SLICE_C, 'INSUFFICIENT', 'compute', 2, 220), ('Slice D', 2.6, 1.0, 'PASS', 'data', 3, 295)]
for name, v, h, verdict, ck, first, y in slices:
    g.text(10, y, name, 22, C['text'], 'bold', 'start', first=first)
    g.add(first, f'<line x1="{gx(v - h)}" y1="{y}" x2="{gx(v + h)}" y2="{y}" stroke="{C[ck]}" stroke-width="4"/>')
    for e in (v - h, v + h):
        g.add(first, f'<line x1="{gx(e)}" y1="{y - 10}" x2="{gx(e)}" y2="{y + 10}" stroke="{C[ck]}" stroke-width="3"/>')
    g.circle(gx(v), y, 9, C[ck], C[ck], first=first)
    g.text(685, y, verdict, 22, C[ck], 'bold', 'start', first=first)
g.write('gate', 3)
g = Svg(900, 420)
mats = [('Group A: 10,000', 175, (4500, 500, 1000, 4000)), ('Group B: 2,000', 615, (600, 400, 200, 800))]
cw, ch = (135, 62)
for title, gx0, (tp, fn, fp, tn) in mats:
    g.text(gx0 + cw, 22, title, 24, C['text'], 'bold')
    g.text(gx0 + cw / 2, 55, 'Approved', 20, C['neutral'], 'bold')
    g.text(gx0 + cw * 1.5, 55, 'Rejected', 20, C['neutral'], 'bold')
    for r, (lab, vals) in enumerate((('Would repay', (tp, fn)), ('Would default', (fp, tn)))):
        y = 72 + r * ch
        g.text(gx0 - 10, y + ch / 2, lab, 20, C['text'], 'bold', 'end')
        for c, v in enumerate(vals):
            g.rect(gx0 + c * cw, y, cw - 6, ch - 6, '#F4F4F4', C['neutral_l'], rx=4)
            g.text(gx0 + c * cw + (cw - 6) / 2, y + (ch - 6) / 2, f'{v:,}', 25, C['text'], 'bold')
    g.rect(gx0 - 4, 68, cw + 2, 2 * ch + 2, 'none', C['compute'], first=2, last=2, sw=4)
    g.rect(gx0 - 4, 68, 2 * cw + 2, ch + 2, 'none', C['memory'], first=3, last=3, sw=4)
    g.rect(gx0 - 4, 68 + ch, 2 * cw + 2, ch + 2, 'none', C['network'], first=4, last=4, sw=4)
rows = [('Approval rate (demographic parity)', '55%', '40%', '15 pts', 'compute', 2, 250), ('Of those who would repay, share approved (TPR)', '90%', '60%', '30 pts', 'memory', 3, 300), ('Of those who would default, share approved (FPR)', '20%', '20%', '0 pts', 'network', 4, 350)]
g.text(650, 222, 'A', 20, C['neutral'], 'bold')
g.text(740, 222, 'B', 20, C['neutral'], 'bold')
g.text(845, 222, 'Gap', 20, C['neutral'], 'bold')
for lab, a, b, gap, ck, first, y in rows:
    g.text(10, y, lab, 21, C[ck], 'bold', 'start', first=first)
    g.text(650, y, a, 24, C['text'], 'bold', first=first)
    g.text(740, y, b, 24, C['text'], 'bold', first=first)
    g.text(845, y, gap, 24, C['danger'] if not gap.startswith('0 ') else C['data'], 'bold', first=first)
g.text(450, 400, 'Constructed example (chapter credit service). Both groups repay at 50%.', 18, C['neutral'])
g.write('matrices', 4)
g = Svg(520, 420)
g.rect(170, 20, 180, 70, C['compute_l'], C['compute'])
g.text(260, 55, 'Model decides', 21, C['text'], 'bold')
g.rect(20, 165, 200, 70, C['data_l'], C['data'])
g.text(120, 200, 'Approved', 21, C['text'], 'bold')
g.rect(300, 165, 200, 70, '#F4F4F4', C['neutral'], dash='6 4')
g.text(400, 200, 'Denied', 21, C['neutral'], 'bold')
g.rect(20, 310, 200, 80, C['data_l'], C['data'])
g.text(120, 350, 'Repayment\nobserved', 20, C['text'], 'bold')
g.text(400, 340, 'No label:\nnever observed', 20, C['danger'], 'bold')
g.line(230, 92, 140, 160, C['neutral'])
g.line(290, 92, 380, 160, C['neutral'])
g.line(120, 237, 120, 305, C['data'])
g.line(400, 237, 400, 305, C['danger'], dash='6 5')
g.line(222, 340, 262, 340, C['data'], arrow=False)
g.line(262, 340, 262, 95, C['data'])
g.text(270, 270, 'retrain', 18, C['data'], 'bold', 'start')
g.write('censored_loop', 1)
x_lo, x_hi = (150, 890)

def ax_x(score):
    return x_lo + (x_hi - x_lo) * (1 - score)
g = Svg(900, 330)
for grp, ck, y in (('A', 'memory', 95), ('B', 'danger', 200)):
    g.add(1, f'''<line x1="{x_lo}" y1="{y + 32}" x2="{x_hi}" y2="{y + 32}" stroke="{C['neutral']}" stroke-width="4"/>''')
    g.text(10, y + 32, f'Subgroup {grp}', 22, C['text'], 'bold', 'start')
    for score in {'memory': [0.75, 0.65, 0.58, 0.51], 'danger': [0.95, 0.88, 0.81, 0.58]}[ck]:
        g.circle(ax_x(score), y, 16, 'none', C[ck], sw=4)
    for score in {'memory': [0.32, 0.25, 0.18, 0.11], 'danger': [0.65, 0.51, 0.44, 0.25]}[ck]:
        cx = ax_x(score)
        g.add(1, f'<path d="M{cx - 16},{y} H{cx + 16} M{cx},{y - 16} V{y + 16}" stroke="{C[ck]}" stroke-width="7"/>')
for thr, key, first in ((0.7, 'acc_line1', 1), (0.37, 'acc_line2', 2)):
    cx = ax_x(thr)
    g.add(first, f'''<line x1="{cx}" y1="55" x2="{cx}" y2="260" stroke="{C['text']}" stroke-width="3" stroke-dasharray="8 6"/>''')
    g.text(cx, 38, f"overall { {0.7: '75', 0.37: '81.25'}[thr]}%", 20, C['text'], 'bold', first=first)
g.text(450, 290, 'Left of a line: approved.  Circle: would repay.  Cross: would default.', 19, C['neutral'])
g.text(160, 322, 'At 75%: A errors 3, B errors 1', 20, C['memory'], 'bold', 'start', first=1, last=1)
g.text(160, 322, f'At 81.25%: A errors {0}, B errors 3 (tripled)', 20, C['danger'], 'bold', 'start', first=2)
g.write('threshold', 2)
g = Svg(560, 470)
g.rect(10, 10, 540, 105, C['data_l'], C['data'], first=1, sw=2.5)
g.text(25, 34, 'Before training', 25, C['data'], 'bold', 'start', first=1)
g.text(280, 80, 'Reweight the training examples', 27, C['text'], 'bold', first=1)
y = 125
g.rect(10, y, 540, 205, C['compute_l'], C['compute'], first=2, sw=2.5)
g.text(25, y + 24, 'During training', 25, C['compute'], 'bold', 'start', first=2)
g.rect(25, y + 72, 135, 76, 'white', C['neutral'], first=2)
g.text(92, y + 110, 'Encoder', 26, C['text'], 'bold', first=2)
g.rect(220, y + 52, 165, 52, 'white', C['data'], first=2)
g.text(302, y + 78, 'Task head', 24, C['text'], 'bold', first=2)
g.rect(220, y + 120, 165, 52, 'white', C['danger'], first=2)
g.text(302, y + 146, 'Adversary', 24, C['danger'], 'bold', first=2)
g.line(162, y + 100, 216, y + 80, C['neutral'], first=2)
g.line(162, y + 120, 216, y + 142, C['neutral'], first=2)
g.text(470, y + 78, 'repay?', 25, C['data'], 'bold', first=2)
g.text(470, y + 146, 'group?', 25, C['danger'], 'bold', first=2)
g.text(302, y + 190, 'gradient reversed into the encoder', 21, C['danger'], 'bold', first=2)
y = 340
g.rect(10, y, 540, 120, C['danger_l'], C['danger'], first=3, sw=2.5)
g.text(25, y + 24, 'After training', 25, C['danger'], 'bold', 'start', first=3)
g.text(280, y + 62, "Move Group B's threshold", 27, C['text'], 'bold', first=3)
g.text(280, y + 98, 'Legally restricted in US credit', 22, C['danger'], 'bold', first=3)
g.write('interventions', 3)
xx = np.linspace(0, 0.18, 200)
yy = np.array([0.85, 0.8517927802, 0.8535534199, 0.8552824951, 0.8569805718, 0.8586482057, 0.8602859426, 0.8618943185, 0.8634738597, 0.8650250833, 0.8665484968, 0.8680445988, 0.8695138791, 0.8709568183, 0.8723738889, 0.8737655544, 0.8751322705, 0.8764744844, 0.8777926353, 0.8790871546, 0.8803584661, 0.8816069858, 0.8828331222, 0.8840372767, 0.8852198434, 0.8863812092, 0.8875217543, 0.8886418519, 0.8897418687, 0.8908221646, 0.8918830931, 0.8929250015, 0.8939482308, 0.8949531158, 0.8959399855, 0.8969091628, 0.8978609648, 0.8987957031, 0.8997136836, 0.9006152068, 0.9015005676, 0.9023700558, 0.903223956, 0.9040625477, 0.9048861053, 0.9056948982, 0.9064891913, 0.9072692445, 0.9080353131, 0.9087876477, 0.9095264946, 0.9102520956, 0.9109646882, 0.9116645055, 0.9123517767, 0.9130267266, 0.9136895761, 0.9143405422, 0.9149798379, 0.9156076725, 0.9162242513, 0.9168297763, 0.9174244455, 0.9180084536, 0.9185819917, 0.9191452475, 0.9196984054, 0.9202416464, 0.9207751483, 0.9212990857, 0.92181363, 0.9223189497, 0.9228152101, 0.9233025736, 0.9237811998, 0.9242512453, 0.9247128638, 0.9251662066, 0.925611422, 0.9260486556, 0.9264780505, 0.9268997474, 0.9273138842, 0.9277205964, 0.9281200171, 0.9285122771, 0.9288975048, 0.9292758261, 0.929647365, 0.930012243, 0.9303705796, 0.930722492, 0.9310680953, 0.9314075027, 0.9317408254, 0.9320681722, 0.9323896505, 0.9327053654, 0.9330154202, 0.9333199163, 0.9336189536, 0.9339126298, 0.9342010409, 0.9344842816, 0.9347624443, 0.9350356202, 0.9353038986, 0.9355673674, 0.9358261128, 0.9360802195, 0.9363297705, 0.9365748477, 0.9368155312, 0.9370518997, 0.9372840307, 0.9375120001, 0.9377358825, 0.9379557512, 0.9381716781, 0.9383837339, 0.938591988, 0.9387965086, 0.9389973626, 0.9391946157, 0.9393883325, 0.9395785764, 0.9397654096, 0.9399488933, 0.9401290875, 0.9403060513, 0.9404798425, 0.940650518, 0.9408181337, 0.9409827444, 0.9411444039, 0.9413031653, 0.9414590804, 0.9416122004, 0.9417625752, 0.9419102541, 0.9420552855, 0.9421977167, 0.9423375945, 0.9424749646, 0.942609872, 0.9427423607, 0.9428724742, 0.9430002551, 0.9431257452, 0.9432489854, 0.9433700163, 0.9434888773, 0.9436056074, 0.9437202449, 0.9438328271, 0.9439433909, 0.9440519726, 0.9441586077, 0.944263331, 0.9443661769, 0.9444671789, 0.9445663703, 0.9446637833, 0.9447594499, 0.9448534015, 0.9449456687, 0.9450362817, 0.9451252703, 0.9452126635, 0.9452984899, 0.9453827776, 0.9454655543, 0.9455468469, 0.9456266822, 0.9457050862, 0.9457820845, 0.9458577025, 0.9459319648, 0.9460048957, 0.9460765192, 0.9461468585, 0.9462159369, 0.9462837768, 0.9463504005, 0.9464158298, 0.9464800861, 0.9465431905, 0.9466051635, 0.9466660254, 0.9467257963, 0.9467844955, 0.9468421425, 0.9468987559, 0.9469543544, 0.9470089561, 0.947062579, 0.9471152405, 0.9471669579, 0.9472177481, 0.9472676278])
fig, ax = plt.subplots(figsize=(3.9, 3.0))
ax.plot(xx, yy, color=C['neutral'], lw=2.5, ls='--')
(xa, ya), (xb, yb), (xc, yc) = ((0.18, 0.9472676278), (0.05, 0.9132120559), (0.0, 0.85))
for x, y, ck, lab, off, ha in ((xa, ya, 'danger', 'A: unconstrained', (-6, 10), 'left'), (xb, yb, 'data', 'B: 91% accuracy,\ndisparity 0.05', (-12, -36), 'right'), (xc, yc, 'memory', 'C: strict equality,\nlarge drop', (-28, -2), 'right')):
    ax.plot([x], [y], 'o', ms=11, color=C[ck], zorder=5)
    ax.annotate(lab, (x, y), xytext=off, textcoords='offset points', ha=ha, fontsize=11, fontweight='bold', color=C[ck])
ax.set_xlim(0.2, -0.01)
ax.set_ylim(0.83, 0.97)
ax.set_xlabel('Disparity (lower is fairer)')
ax.set_ylabel('Accuracy')
ax.text(0.0, -0.3, 'Hypothetical (chapter figure 3)', transform=ax.transAxes, fontsize=9, color=C['neutral'])
sp.save(fig, out / 'frontier.svg')
fig, ax = plt.subplots(figsize=(4.6, 1.9))
for i, (label, v, ck, txt) in enumerate((('One car, lifetime', 57.0, 'neutral', '57 t'), ('Search, estimated', 284.0, 'compute', '284 t'), ('Search, measured', 1, 'data', '???'))):
    y = 2 - i
    ax.barh(y, v, color=C[ck], height=0.62, alpha=0.0 if txt == '???' else 1.0)
    ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
    ax.text(v * 1.2 if txt != '???' else 8, y, txt, va='center', fontsize=13, fontweight='bold', color=C['danger'] if txt == '???' else C[ck])
ax.set_xscale('log')
ax.set_xlim(1, 3000)
ax.set_ylim(-0.6, 2.6)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.grid(axis='x', color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('Tonnes of CO₂e (log scale)')
sp.save(fig, out / 'next_part2.svg')
print('figures written')
