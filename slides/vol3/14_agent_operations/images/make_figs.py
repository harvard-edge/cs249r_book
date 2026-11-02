#!/usr/bin/env python3
"""Figures for Lecture 14 (Operating Agents), Volume III.

Every value comes from calcs.py, which reads MLSysIM; this script imports it
and prints the recorded values (val, num), so figures and slides agree. The
real OpenTelemetry run (run_step*) is read from trace_run.json, the spans
images/trace_run.py recorded. "OA" is the rendered chapter, Operating Agents. The book SVGs
(tail-sampling-pipeline, canary-deployment-pipeline, SPRT corridor) are
page-sized and fall below the legibility floor on a slide, so they are
redrawn here at slide size:
  - tail_pipe_step*: OA figure 1 (tail-sampling-pipeline.svg).
  - stages_step*: OA figure 2 (vol3-canary-deployment-pipeline.svg).
  - sprt_step*: OA figure 3 (vol3_eval_margin_sprt_corridor.svg), with
    illustrative walks added.
Full-width diagrams are 900 px wide with type of 18 px or more; column
figures are drawn at 5.4 x 3.6 in (or 540 px wide). Plots use slideplot.

Run from this folder: python3 make_figs.py   (then make deck converts SVG)
"""
from types import SimpleNamespace
import math
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
import json
C = sp.C
plt = sp.plt
out = Path(__file__).parent
P = {'store': ('#f3e8ff', '#7c3aed', '#6d28d9'), 'tool': ('#fef3c7', '#d97706', '#b45309'), 'fault': ('#fee2e2', '#dc2626', '#b91c1c'), 'ok': ('#dcfce7', '#16a34a', '#15803d'), 'think': ('#e0f2fe', '#0284c7', '#0369a1'), 'plain': ('#ffffff', '#94a3b8', '#0f172a'), 'band': ('#f8fafc', '#cbd5e1', '#0f172a'), 'gray': ('#f1f5f9', '#94a3b8', '#475569')}
FONT = 'Helvetica, Arial, sans-serif'
VIO, RED, GRN, AMB, SKY, GRY = ('#7c3aed', '#dc2626', '#16a34a', '#d97706', '#0284c7', '#64748b')

class Svg:

    def __init__(self, w, h):
        self.w, self.h, self.items = (w, h, [])

    def add(self, s, show=True):
        self.items.append(s if show else f'<g opacity="0">{s}</g>')

    def _t(self, x, y, s, size, color, bold=False, anchor='middle', italic=False):
        w = ' font-weight="bold"' if bold else ''
        it = ' font-style="italic"' if italic else ''
        return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{w}{it}>{s}</text>'

    def text(self, x, y, s, size=19, color='#0f172a', show=True, **kw):
        self.add(self._t(x, y, s, size, color, **kw), show)

    def box(self, x, y, w, h, title, subs=(), role='plain', show=True, dash=False, tsize=21, ssize=18, rx=6, sw=2.4, title_color=None):
        fill, stroke, text = P[role]
        d = ' stroke-dasharray="8,5"' if dash else ''
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        n = 1 + len(subs)
        lh = [tsize] + [ssize] * len(subs)
        gap = 7
        total = sum(lh) + gap * (n - 1)
        y0 = y + (h - total) / 2
        cx = x + w / 2
        yy = y0 + tsize * 0.8
        s += self._t(cx, yy, title, tsize, title_color or text, bold=True)
        for sub in subs:
            yy += gap + ssize
            s += self._t(cx, yy, sub, ssize, '#1e293b')
        self.add(s, show)

    def arrow(self, pts, color=GRY, show=True, dash=False, width=2.6):
        d = ' '.join((f"{('M' if i == 0 else 'L')}{x},{y}" for i, (x, y) in enumerate(pts)))
        da = ' stroke-dasharray="8,5"' if dash else ''
        (x1, y1), (x2, y2) = (pts[-2], pts[-1])
        a = math.atan2(y2 - y1, x2 - x1)
        L, Wd = (14, 7)
        bx, by = (x2 - L * math.cos(a), y2 - L * math.sin(a))
        p1 = (bx + Wd * math.sin(a), by - Wd * math.cos(a))
        p2 = (bx - Wd * math.sin(a), by + Wd * math.cos(a))
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da}/><polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>', show)

    def rect(self, x, y, w, h, fill, stroke='none', show=True, rx=0, sw=2, dash=False, opacity=1.0):
        d = ' stroke-dasharray="8,5"' if dash else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>', show)

    def line(self, x1, y1, x2, y2, color=GRY, width=2, dash=False, show=True):
        d = ' stroke-dasharray="6,5"' if dash else ''
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>', show)

    def save(self, name):
        body = '\n'.join(self.items)
        (out / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')

def hook_timeline(k):
    d0 = 24.0
    d1 = 25.0
    dp = 27.0
    dr = 28.0
    mon = 2.0
    rb = 24.0 / 24
    s = Svg(900, 232 if k < 4 else 304)
    x0, day = (262, 105)

    def X(d):
        return x0 + (d - d0) * day
    s.box(10, 14, 215, 140, 'Before launch', ('offline evals:', 'looked good', 'A/B test:', 'looked good'), 'ok', tsize=19, ssize=18)
    s.line(X(d0), 180, X(d0 + 5.4), 180, '#334155', 2)
    names = {d0 + i: f'{wd} {d0 + i:.0f}' for i, wd in enumerate(('Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue'))}
    for d, nm in names.items():
        s.line(X(d), 172, X(d), 188, '#334155', 2)
        s.text(X(d), 214, nm, 18, '#334155')
    s.rect(X(d0), 14, day * (d1 - d0), 40, P['tool'][0], P['tool'][1], rx=5, sw=2.4)
    s.text(X((d0 + d1) / 2), 41, 'Rollout', 19, P['tool'][2], bold=True)
    sh2 = k >= 2
    s.rect(X(d1), 64, day * mon, 40, P['think'][0], P['think'][1], rx=5, sw=2.4, show=sh2)
    s.text(X(d1 + mon / 2), 91, 'Live; monitored', 19, P['think'][2], bold=True, show=sh2)
    sh3 = k >= 3
    late = dp + 0.85
    s.line(X(late), 10, X(late), 172, RED, 2.4, dash=True, show=sh3)
    s.text(X(late) - 8, 140, 'Prompt patch', 18, P['fault'][2], anchor='end', bold=True, show=sh3)
    s.rect(X(dr), 114, day * rb, 40, P['fault'][0], P['fault'][1], rx=5, sw=2.4, show=sh3)
    s.text(X(dr + rb / 2), 141, 'Rollback', 19, P['fault'][2], bold=True, show=sh3)
    s.text(X(dr + rb) + 8, 141, 'about 24 h', 18, P['fault'][2], anchor='start', show=sh3)
    if k >= 4:
        s.rect(10, 230, 880, 68, P['band'][0], P['band'][1], rx=6)
        for x, a, b in ((117, 'No criterion', 'for sycophancy'), (X(d1 + mon / 2), 'Detected outside', 'the declared gate'), (X(dr + rb / 2) + 20, 'Admission stop,', 'then settlement')):
            s.text(x, 258, a, 18, '#0f172a', bold=True)
            s.text(x, 284, b, 18)
    s.save(f'hook_step{k}.svg')
for k in range(1, 5):
    hook_timeline(k)

def gantt(name, rows, xmax, notes=(), size=(5.4, 3.4), crit=(), ghost=(), qmark=None, gap=None, fs=15):
    """rows: list of (label, start, end, color_key). crit: labels outlined in
    violet (the critical path). ghost: (label, start, end) drawn faint.
    gap: (start, end) hatched as unattributed time."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, (label, s0, e0, ck) in enumerate(rows):
        y = n - 1 - i
        for gl, gs, ge in ghost:
            if gl == label:
                ax.barh(y, ge - gs, left=gs, height=0.6, color='white', edgecolor=C['neutral'], hatch='///', linewidth=1.2, zorder=1)
        if e0 is not None:
            ax.barh(y, e0 - s0, left=s0, height=0.6, color=C[ck], zorder=2, edgecolor=VIO if label in crit else 'none', linewidth=3.0 if label in crit else 0)
        ax.text(-0.15, y, label, ha='right', va='center', fontsize=fs, color=C['text'], fontweight='bold')
    if gap:
        ax.axvspan(gap[0], gap[1], color=C['danger_l'], zorder=0)
    if qmark:
        y = n - 1 - [r[0] for r in rows].index(qmark[0])
        ax.text(qmark[1], y, '?', ha='center', va='center', fontsize=22, color=C['danger'], fontweight='bold')
    for x, y, txt, color, ha in notes:
        ax.text(x, y, txt, ha=ha, va='center', fontsize=fs - 1, color=color, fontweight='bold')
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Seconds after segment start', fontsize=fs - 2)
    ax.tick_params(axis='x', labelsize=fs - 2)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / name)

def run_spans():
    """Times from the real OpenTelemetry run, seconds after the root span starts."""
    recs = json.loads((out / 'trace_run.json').read_text())
    t0 = min((r['start_ns'] for r in recs))
    spans = {r['name']: ((r['start_ns'] - t0) / 10 ** 9, (r['end_ns'] - t0) / 10 ** 9) for r in recs}
    gap = spans['chat synthesize_report'][0] - spans['execute_tool run_tests'][1]
    return dict(spans=spans, gap=gap)
run = run_spans()
S = run['spans']
LBL = [('root span', 'invoke_agent repair_agent', 'neutral'), ('retrieve', 'retrieve supporting_context', 'data'), ('model: propose', 'chat propose_checks', 'memory'), ('run_tests', 'execute_tool run_tests', 'compute'), ('run_lint', 'execute_tool run_lint', 'compute'), ('model: report', 'chat synthesize_report', 'memory')]
rows = [(lbl, S[k][0], S[k][1], ck) for lbl, k, ck in LBL]
gantt('run_step1.svg', rows, 12.6)
gantt('run_step2.svg', rows, 12.6, crit=('retrieve', 'model: propose', 'run_tests', 'model: report'), gap=(S['execute_tool run_tests'][1], S['chat synthesize_report'][0]), notes=[(S['execute_tool run_lint'][1] + 0.3, 1, 'off the path', C['neutral'], 'left'), (9.9, 2, f"{run['gap']:.1f} s gap", C['danger'], 'left')])
iv = {'retrieve': (0.0, 0.2), 'propose': (0.2, 3.2), 'tests': (3.2, 9.2), 'lint': (3.2, 5.2), 'synthesize': (9.7, 11.7)}
t_ret, t_pro = (iv['retrieve'][1], iv['propose'][1])
t_chk = t_pro
cut = 1.0
was = f'was {6.0:g} s'
base = [('retrieve', iv['retrieve'][0], t_ret, 'data'), ('model: propose', t_ret, t_pro, 'memory'), ('run_tests', t_chk, t_chk + cut, 'compute'), ('run_lint', t_chk, iv['lint'][1], 'compute'), ('model: report', None, None, 'memory')]
gantt('cut_step1.svg', base, 12.6, ghost=[('run_tests', t_chk + cut, iv['tests'][1])], qmark=('model: report', 8.0), notes=[(t_chk + 6.2, 2, was, C['neutral'], 'left')])
end_l = 5.7
done = [r if r[0] != 'model: report' else ('model: report', end_l, 7.7, 'memory') for r in base]
LABEL = {'retrieve': 'retrieve', 'propose': 'model: propose', 'tests': 'run_tests', 'lint': 'run_lint', 'synthesize': 'model: report'}
gantt('cut_step2.svg', done, 12.6, ghost=[('run_tests', t_chk + cut, iv['tests'][1])], crit=tuple((LABEL[k] for k in ['retrieve', 'propose', 'lint', 'synthesize'])), notes=[(7.7 + 0.25, 0, '7.7 s', C['text'], 'left'), (t_chk + 6.2, 2, was, C['neutral'], 'left')])

def records(k):
    s = Svg(900, 400)
    s.box(300, 14, 300, 64, 'One runtime event', ('model call, tool run, or check',), 'plain', tsize=20, ssize=18)
    cards = [(30, 'Diagnostic trace', ('timing and debugging', 'may sample, cut, or drop'), 'think'), (330, 'Trajectory log', ('recovery rebuilds from it', 'never sampled'), 'store'), (630, 'Accounting record', ('every trial, verdict,', 'spend, and drop'), 'ok')]
    for i, (x, t, subs, role) in enumerate(cards):
        sh = k >= i + 1
        s.arrow([(450, 78), (x + 120, 160)], P[role][1], show=sh)
        s.box(x, 162, 240, 120, t, subs, role, show=sh, tsize=21, ssize=18)
    s.text(150, 318, 'Explains one trial', 18, P['think'][2], bold=True)
    s.text(450, 318, 'Lecture 11', 18, P['store'][2], bold=True, show=k >= 2)
    s.text(750, 318, 'Every rate comes from here', 18, P['ok'][2], bold=True, show=k >= 3)
    s.text(450, 375, "A span's status is not a task verdict.", 20, P['fault'][2], bold=True, show=k >= 3)
    s.save(f'records_step{k}.svg')
for k in range(1, 4):
    records(k)
fig, ax = plt.subplots(figsize=(5.8, 2.5))
rows_d = [('1/1', 16.3), ('1/2', 9.4), ('1/4', 6.38), ('1/8', 4.12), ('1/16', 2.12), ('1/1024', -0.2)]
err = 2.5
for i, (lab, v) in enumerate(rows_d):
    y = len(rows_d) - 1 - i
    ck = 'danger' if v > 5 else 'neutral' if v > err else 'data'
    ax.barh(y, v, height=0.62, color=C[ck])
    ax.text(-1.0, y, f"1 trace per {lab.split('/')[1]}" if lab != '1/1' else 'every request', ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
    ax.text(max(v, 0) + 0.3, y, f'{v:+.2f}%'.replace('+', '+') if v < 0 else f'+{v:g}%', va='center', fontsize=12, fontweight='bold', color=C[ck])
ax.axvspan(-err, err, color=C['neutral_l'], alpha=0.6, zorder=0)
ax.text(err + 0.2, len(rows_d) - 0.35, 'within error (±2.5%)', fontsize=11, color=C['neutral'], va='bottom')
ax.set_xlim(-3, 21)
ax.set_ylim(-0.6, len(rows_d) - 0.1)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Change in average latency, web search cluster (%)')
ax.axvline(0, color='#888888', lw=1)
sp.save(fig, out / 'dapper_bars.svg')

def head_miss(k):
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ns = list(range(1, 61))
    for idx, (sc, lab, ck) in enumerate([(SimpleNamespace(miss=0.904382075, name='A failure mode seen ten times under a 1 percent head sample', occurrences=10, retention_probability=0.01), 'p = 1%', 'danger'), (SimpleNamespace(miss=0.59049, name='A failure mode seen five times under a 10 percent head sample', occurrences=5, retention_probability=0.1), 'p = 10%', 'memory'), (SimpleNamespace(miss=0.3584859224, name='A failure mode seen twenty times under a 5 percent head sample', occurrences=20, retention_probability=0.05), 'p = 5%', 'compute')]):
        if idx == 2 and k < 2:
            continue
        ax.plot(ns, {'p = 1%': [99.0, 98.01, 97.0299, 96.059601, 95.09900499, 94.14801494, 93.20653479, 92.27446944, 91.35172475, 90.4382075, 89.53382543, 88.63848717, 87.7521023, 86.87458128, 86.00583546, 85.14577711, 84.29431934, 83.45137615, 82.61686238, 81.79069376, 80.97278682, 80.16305895, 79.36142836, 78.56781408, 77.78213594, 77.00431458, 76.23427143, 75.47192872, 74.71720943, 73.97003734, 73.23033697, 72.4980336, 71.77305326, 71.05532273, 70.3447695, 69.6413218, 68.94490859, 68.2554595, 67.57290491, 66.89717586, 66.2282041, 65.56592206, 64.91026284, 64.26116021, 63.61854861, 62.98236312, 62.35253949, 61.72901409, 61.11172395, 60.50060671, 59.89560065, 59.29664464, 58.70367819, 58.11664141, 57.535475, 56.96012025, 56.39051905, 55.82661385, 55.26834772, 54.71566424], 'p = 10%': [90.0, 81.0, 72.9, 65.61, 59.049, 53.1441, 47.82969, 43.046721, 38.7420489, 34.86784401, 31.38105961, 28.24295365, 25.41865828, 22.87679245, 20.58911321, 18.53020189, 16.6771817, 15.00946353, 13.50851718, 12.15766546, 10.94189891, 9.847709022, 8.86293812, 7.976644308, 7.178979877, 6.461081889, 5.8149737, 5.23347633, 4.710128697, 4.239115828, 3.815204245, 3.43368382, 3.090315438, 2.781283894, 2.503155505, 2.252839954, 2.027555959, 1.824800363, 1.642320327, 1.478088294, 1.330279465, 1.197251518, 1.077526366, 0.9697737298, 0.8727963568, 0.7855167211, 0.706965049, 0.6362685441, 0.5726416897, 0.5153775207, 0.4638397687, 0.4174557918, 0.3757102126, 0.3381391914, 0.3043252722, 0.273892745, 0.2465034705, 0.2218531234, 0.1996678111, 0.17970103], 'p = 5%': [95.0, 90.25, 85.7375, 81.450625, 77.37809375, 73.50918906, 69.83372961, 66.34204313, 63.02494097, 59.87369392, 56.88000923, 54.03600877, 51.33420833, 48.76749791, 46.32912302, 44.01266687, 41.81203352, 39.72143185, 37.73536025, 35.84859224, 34.05616263, 32.3533545, 30.73568677, 29.19890243, 27.73895731, 26.35200945, 25.03440897, 23.78268853, 22.5935541, 21.46387639, 20.39068257, 19.37114845, 18.40259102, 17.48246147, 16.6083384, 15.77792148, 14.9890254, 14.23957413, 13.52759543, 12.85121566, 12.20865487, 11.59822213, 11.01831102, 10.46739547, 9.944025699, 9.446824414, 8.974483193, 8.525759033, 8.099471082, 7.694497528, 7.309772651, 6.944284019, 6.597069818, 6.267216327, 5.953855511, 5.656162735, 5.373354598, 5.104686868, 4.849452525, 4.606979899]}[lab], color=C[ck], lw=3)
    ax.text(42, 100 * 0.6556592206 + 4, 'p = 1%', color=C['danger'], fontsize=15, fontweight='bold', ha='center')
    ax.text(1.5, 16, 'p = 10%', color=C['memory'], fontsize=15, fontweight='bold', ha='left')
    if k >= 2:
        None
        ax.text(42, 100 * 0.1159822213 + 5, 'p = 5%', color=C['compute'], fontsize=15, fontweight='bold', ha='left')
    for sc, key, ck, off in [(SimpleNamespace(miss=0.904382075, name='A failure mode seen ten times under a 1 percent head sample', occurrences=10, retention_probability=0.01), 'hs1_miss', 'danger', (8, 10)), (SimpleNamespace(miss=0.59049, name='A failure mode seen five times under a 10 percent head sample', occurrences=5, retention_probability=0.1), 'hs10_miss', 'memory', (-40, -22)), (SimpleNamespace(miss=0.3584859224, name='A failure mode seen twenty times under a 5 percent head sample', occurrences=20, retention_probability=0.05), 'hs5_miss', 'compute', (10, 6))]:
        n, v = ({'hs1_miss': 10, 'hs10_miss': 5, 'hs5_miss': 20}[key], 100 * {'hs1_miss': 0.904382075, 'hs10_miss': 0.59049, 'hs5_miss': 0.3584859224}[key])
        ax.plot([n], [v], 'o', ms=10, color=C[ck], mec='white', mew=1.5, zorder=5)
        ax.annotate(f"{ {'hs1_miss': '90', 'hs10_miss': '59', 'hs5_miss': '36'}[key]}%", (n, v), xytext=off, textcoords='offset points', fontsize=15, fontweight='bold', color=C[ck])
    if k < 2:
        n, v = (20, 100 * 0.3584859224)
        ax.plot([n], [v], 'o', ms=12, mfc='white', mec=C['compute'], mew=2.5)
        ax.annotate('?', (n, v), xytext=(10, 4), textcoords='offset points', fontsize=18, fontweight='bold', color=C['compute'])
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Times the failure mode occurs', fontsize=14)
    ax.set_ylabel('P(no detailed trace kept), %', fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'miss_step{k}.svg')
for k in (1, 2):
    head_miss(k)

def tail_pipe(k):
    s = Svg(900, 330)
    s.box(10, 150, 160, 80, 'Admitted', ('trials',), 'plain', tsize=21, ssize=20)
    s.box(240, 10, 360, 80, 'Accounting record', ('every trial, verdict, drop',), 'ok', tsize=21, ssize=18)
    s.arrow([(90, 150), (90, 50), (236, 50)], P['ok'][1])
    s.box(240, 140, 240, 100, 'Staging buffer', ('held until the', 'verdict window ends'), 'store', tsize=20, ssize=18)
    s.arrow([(170, 190), (236, 190)], GRY)
    sh = k >= 2
    cls = [('Failed or violation: keep', 'fault'), ('Pending: keep, flag', 'tool'), ('Slow or costly: keep', 'think'), ('Ordinary: 2% control', 'gray')]
    for i, (t_, role) in enumerate(cls):
        y = 104 + i * 56
        s.box(604, y, 286, 46, t_, (), role, show=sh, tsize=19)
        s.arrow([(480, 190), (540, 190), (540, y + 23), (600, y + 23)], P[role][1], show=sh)
    s.text(747, 98, 'evict from the bottom first', 18, '#475569', show=sh, italic=True)
    s.arrow([(360, 240), (360, 290), (200, 290), (200, 94)], RED, dash=True, show=sh)
    s.text(372, 284, 'buffer expired: the record', 18, P['fault'][2], anchor='start', show=sh)
    s.text(372, 308, 'flags the missing detail', 18, P['fault'][2], anchor='start', show=sh)
    s.save(f'tail_step{k}.svg')
for k in (1, 2):
    tail_pipe(k)
sp.hbars(str(out / 'retention.svg'), [('Keep all', 336.0, 'neutral', '336 GB'), ('Head 5%', 16.8, 'memory', '16.8 GB: 20 of 400 failures/h'), ('Tail', 23.184, 'data', '23.2 GB: all 400 failures/h')], 'GB', log=False, xlabel='Retained trace bodies per day (GB, decimal)', size=(5.8, 2.0))

def findings(k):
    s = Svg(560, 384)
    items = [('Observation adapter', 'cut the result, no notice', 'fault', 'interface', 'defect'), ('Model', 'proposed an out-of-scope edit', 'think', 'model', 'behavior'), ('Runtime gate', 'refused before dispatch', 'ok', 'gate', 'worked')]
    for i, (t_, sub, role, v1, v2) in enumerate(items):
        y = 6 + i * 128
        sh = k >= i + 1
        s.box(6, y, 404, 100, t_, (sub,), role, show=sh, tsize=24, ssize=21)
        s.text(422, y + 46, v1, 21, P[role][2], anchor='start', bold=True, show=sh)
        s.text(422, y + 72, v2, 21, P[role][2], anchor='start', bold=True, show=sh)
        if i < 2:
            s.arrow([(208, y + 100), (208, y + 126)], GRY, show=k >= i + 2)
    s.save(f'findings_step{k}.svg')
for k in (1, 2, 3):
    findings(k)
s = Svg(900, 360)
s.box(250, 10, 400, 64, 'Recorded trajectory', ('saved prefix and captured results',), 'store', tsize=20, ssize=18)
modes = [(20, 'Time-travel replay', ('shows what the', 'model received'), 'store', 'no new outcome'), (320, 'Regression replay', ('old vs. new adapter', 'on captured output'), 'tool', 'no model call'), (620, 'Counterfactual branch', ('fork the prefix,', 'change one thing'), 'think', 'fresh model calls')]
for x, t, subs, role, foot in modes:
    s.arrow([(450, 74), (x + 130, 128)], GRY)
    s.box(x, 130, 260, 132, t, subs, role, tsize=21, ssize=18)
    s.text(x + 130, 300, foot, 19, P[role][2], bold=True)
s.text(450, 345, 'Only the branch tests whether a fix changes the outcome.', 19, '#0f172a', bold=True)
s.save('replay.svg')

def branches(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    names = {'original': 'Original interface', 'warning': 'Warning only', 'paginated': 'Paginated interface'}
    rows_b = [('Original interface', 1, 19, 0), ('Warning only', 8, 0, 12), ('Paginated interface', 20, 0, 0)]
    runs = 20.0
    for i, (name, acc, forb, bud) in enumerate(rows_b):
        y = len(rows_b) - 1 - i
        ax.text(-0.4, y, name, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
        if k < 2:
            ax.barh(y, runs, height=0.6, color='white', edgecolor=C['neutral'], lw=1.2)
            ax.text(runs / 2, y, '? of 20 accepted', ha='center', va='center', fontsize=13, color=C['neutral'])
            continue
        left = 0
        for v, ck, lab in ((acc, 'data', 'accepted'), (forb, 'danger', 'forbidden edit'), (bud, 'neutral', 'budget stop')):
            if v:
                ax.barh(y, v, left=left, height=0.6, color=C[ck])
                txt = f'{v} {lab}' if v >= 6 else f'{v}'
                ax.text(left + v / 2, y, txt, ha='center', va='center', fontsize=12, color='white', fontweight='bold')
            left += v
    ax.set_xlim(0, runs)
    ax.set_xticks([runs * q / 4 for q in range(5)])
    ax.set_ylim(-0.6, len(rows_b) - 0.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('20 continuations per branch from one saved prefix (illustrative)')
    sp.save(fig, out / f'branches_step{k}.svg')
for k in (1, 2):
    branches(k)
fig, ax = plt.subplots(figsize=(5.4, 3.3))
ns = list(range(5, 121))
ax.plot(ns, [43.44824648, 39.03342879, 35.43304351, 32.44075649, 29.91450484, 27.75327999, 25.8832967, 24.24940067, 22.80953724, 21.53108027, 20.3883301, 19.36076805, 18.4318135, 17.58792236, 16.81792241, 16.11251581, 15.46390189, 14.86548746, 14.31166185, 13.79762047, 13.31922509, 12.87289219, 12.45550297, 12.06433048, 11.69697985, 11.35133932, 11.02553955, 10.71791983, 10.42699975, 10.15145542, 9.890099232, 9.641862859, 9.405782584, 9.180986823, 8.966685371, 8.76216012, 8.566757018, 8.379879087, 8.200980323, 8.029560376, 7.865159873, 7.707356309, 7.555760416, 7.410012967, 7.269781922, 7.134759913, 7.00466199, 6.879223612, 6.758198858, 6.641358809, 6.528490112, 6.419393672, 6.313883485, 6.211785579, 6.112937053, 6.017185214, 5.92438679, 5.83440721, 5.747119959, 5.662405979, 5.580153132, 5.500255701, 5.422613939, 5.347133652, 5.273725819, 5.202306241, 5.13279522, 5.06511726, 4.999200796, 4.93497794, 4.872384248, 4.811358507, 4.751842528, 4.693780971, 4.637121165, 4.581812954, 4.527808543, 4.475062369, 4.423530964, 4.373172841, 4.323948381, 4.275819729, 4.228750694, 4.182706666, 4.137654524, 4.093562556, 4.050400393, 4.008138929, 3.966750261, 3.926207629, 3.886485354, 3.847558786, 3.809404253, 3.771999012, 3.735321207, 3.699349821, 3.66406464, 3.629446215, 3.595475823, 3.562135437, 3.529407693, 3.497275857, 3.465723802, 3.434735975, 3.404297375, 3.37439353, 3.345010469, 3.316134707, 3.287753217, 3.259853416, 3.232423145, 3.20545065, 3.178924566, 3.152833902, 3.127168024, 3.101916642], color=VIO, lw=3)
for n, ck, lab in ((20, 'danger', '20 of 20: up to 16%'), (73, 'data', '73 clean runs: under 5%')):
    v = 100 * {'20 of 20: up to 16%': 0.1611251581, '73 clean runs: under 5%': 0.04999200796}[lab]
    ax.plot([n], [v], 'o', ms=10, color=C[ck], mec='white', mew=1.5, zorder=5)
    ax.annotate(lab, (n, v), xytext=(10, 8), textcoords='offset points', fontsize=15, fontweight='bold', color=C[ck])
ax.axhline(100 * 0.05, color=C['neutral'], lw=1.5, ls='--')
ax.set_xlim(0, 122)
ax.set_ylim(0, 45)
ax.set_xlabel('Consecutive accepted runs, no failures', fontsize=14)
ax.set_ylabel('Upper limit on failure rate, %', fontsize=14)
ax.tick_params(labelsize=13)
ax.grid(color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'wilson.svg')

def stages(k):
    s = Svg(560, 436)
    st = [('Offline gate', 'paired comparison', 'tool'), ('Shadow', 'copied inputs, no live effects', 'think'), ('Canary', 'fraction f of live tasks', 'tool'), ('Wider release', 'next stage, own question', 'ok')]
    for i, (t_, sub, role) in enumerate(st):
        y = 6 + i * 108
        sh = k >= i + 1
        s.box(6, y, 360, 84, t_, (sub,), role, show=sh, tsize=24, ssize=20)
        if i < 3:
            s.arrow([(186, y + 84), (186, y + 106)], GRY, show=k >= i + 2)
            s.arrow([(366, y + 42), (420, y + 42), (420, 190 if i < 2 else 262)], RED, show=sh, dash=True)
    s.box(384, 192, 170, 68, 'Admission', ('hold',), 'fault', tsize=22, ssize=21)
    s.save(f'stages_step{k}.svg')
for k in range(1, 5):
    stages(k)
fig, ax = plt.subplots(figsize=(5.4, 2.4))
for i, (name, dist) in enumerate((('Baseline', (0.7, 0.3)), ('Candidate', (0.5, 0.5)))):
    y = 1 - i
    ax.barh(y, dist[0] * 100, height=0.6, color=C['compute'])
    ax.barh(y, dist[1] * 100, left=dist[0] * 100, height=0.6, color=C['memory'])
    ax.text(dist[0] * 50, y, f'tests {dist[0] * 100:.0f}%', ha='center', va='center', color='white', fontsize=14, fontweight='bold')
    ax.text(dist[0] * 100 + dist[1] * 50, y, f'lint {dist[1] * 100:.0f}%', ha='center', va='center', color='white', fontsize=14, fontweight='bold')
    ax.text(-2, y, name, ha='right', va='center', fontsize=15, fontweight='bold', color=C['text'])
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.set_xticks([])
for sname in ('left', 'bottom'):
    ax.spines[sname].set_visible(False)
ax.set_title('JS divergence = 0.021 nats, 3% of ln 2', fontsize=15, fontweight='bold', color=C['text'])
sp.save(fig, out / 'js_bars.svg')
RED_SEED = 2

def sprt_fig(k):
    s3 = {'lower': -2.985681938, 'upper': 4.553876892, 'dn_success': -0.1823215568, 'up_fail': 0.9162907319}
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    xs = [0, 70]
    ax.fill_between(xs, s3['lower'], s3['upper'], color=C['neutral_l'], alpha=0.5)
    ax.axhline(s3['upper'], color=C['danger'], lw=3)
    ax.axhline(s3['lower'], color=C['data'], lw=3)
    ax.text(69, s3['upper'] + 0.25, 'hold: degraded (4.55)', ha='right', va='bottom', fontsize=14, fontweight='bold', color=C['danger'])
    ax.text(69, s3['lower'] - 0.25, 'select p0 = 0.90 (-2.99)', ha='right', va='top', fontsize=14, fontweight='bold', color=C['data'])
    ax.text(3, 0.5, 'keep sampling', fontsize=14, color=C['neutral'], fontweight='bold')
    if k >= 2:
        for p, ck, seed in ((0.75, 'danger', RED_SEED), (0.9, 'data', 3)):
            rng = random.Random(seed)
            x, y, xsw, ysw = (0, 0.0, [0], [0.0])
            while s3['lower'] < y < s3['upper'] and x < 70:
                ok = rng.random() < p
                y += s3['dn_success'] if ok else s3['up_fail']
                x += 1
                xsw.append(x)
                ysw.append(y)
            ax.step(xsw, ysw, where='post', color=C[ck], lw=2.4)
            ax.plot([xsw[-1]], [ysw[-1]], 'o', ms=9, color=C[ck], mec='white', mew=1.5)
        ax.text(3, 3.4, 'illustrative walks', fontsize=13, color=C['neutral'], style='italic')
    ax.set_xlim(0, 70)
    ax.set_ylim(s3['lower'] - 1.4, s3['upper'] + 1.4)
    ax.set_xlabel('Labeled canary tasks', fontsize=14)
    ax.set_ylabel('Evidence for degradation', fontsize=14)
    ax.tick_params(labelsize=13)
    sp.save(fig, out / f'sprt_step{k}.svg')
for k in (1, 2):
    sprt_fig(k)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
fs_ = [f / 100 for f in range(1, 101)]
ax.plot([100 * f for f in fs_], [75.8967724, 38.1983862, 25.63225747, 19.3491931, 15.57935448, 13.06612873, 11.27096749, 9.92459655, 8.877419156, 8.03967724, 7.354252037, 6.783064367, 6.299751723, 5.885483743, 5.526451494, 5.212298275, 4.935104259, 4.688709578, 4.468251179, 4.26983862, 4.090322495, 3.927126018, 3.778120539, 3.641532183, 3.515870896, 3.399875862, 3.292473052, 3.192741872, 3.099888704, 3.013225747, 2.932153948, 2.856149138, 2.784750679, 2.71755213, 2.654193497, 2.594354789, 2.537750605, 2.48412559, 2.433250574, 2.38491931, 2.338945668, 2.295161248, 2.253413312, 2.213563009, 2.175483831, 2.13906027, 2.104186647, 2.070766092, 2.038709641, 2.007935448, 1.978368086, 1.949937931, 1.922580611, 1.896236526, 1.870850407, 1.846370936, 1.822750393, 1.799944352, 1.777911397, 1.756612873, 1.736012662, 1.716076974, 1.696774165, 1.678074569, 1.659950345, 1.642375339, 1.625324961, 1.608776065, 1.592706846, 1.577096749, 1.561926372, 1.547177394, 1.532832499, 1.518875303, 1.505290299, 1.492062795, 1.479178862, 1.466625287, 1.454389524, 1.442459655, 1.430824351, 1.419472834, 1.408394848, 1.397580624, 1.387020852, 1.376706656, 1.366629568, 1.356781505, 1.347154746, 1.337741916, 1.32853596, 1.319530135, 1.310717983, 1.302093323, 1.293650236, 1.285383046, 1.277286313, 1.26935482, 1.26158356, 1.253967724], color=VIO, lw=3)
for f, off in ((0.05, (10, 4)), (0.1, (10, 4)), (1.0, (-14, -16))):
    v = {0.05: 15.57935448, 0.1: 8.03967724, 1.0: 1.253967724}[f]
    ax.plot([100 * f], [v], 'o', ms=10, color=C['compute'], mec='white', mew=1.5, zorder=5)
    ax.annotate(f'{100 * f:.0f}%: {v:.1f} h', (100 * f, v), xytext=off, textcoords='offset points', fontsize=15, fontweight='bold', color=C['compute'], ha='left' if f < 1 else 'right')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks([1, 5, 10, 50, 100])
ax.set_xticklabels(['1', '5', '10', '50', '100'])
ax.set_ylim(0.75, 130)
ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
ax.set_yticklabels(['1', '2', '5', '10', '20', '50', '100'])
ax.minorticks_off()
ax.set_xlabel('Canary share of eligible tasks, %', fontsize=14)
ax.set_ylabel('Hours to a decision', fontsize=14)
ax.tick_params(labelsize=13)
ax.grid(color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'calendar.svg')
fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9))
e5, e100 = (7.010709516, 11.28570952)
ax = axes[0]
ax.bar([0, 1], [e5, e100], color=[C['compute'], C['neutral']], width=0.6)
for i, (v, key) in enumerate(((e5, 'vt_canary'), (e100, 'vt_full'))):
    ax.text(i, v + 0.4, {0: '7', 1: '11'}[i], ha='center', fontsize=15, fontweight='bold')
ax.set_title('Measured regression:\nexcess failures', fontsize=13, fontweight='bold')
ax.set_ylim(0, 14)
ax2 = axes[1]
shares = (100 * 0.05, 100 * 1.0)
ax2.bar([0, 1], shares, color=[C['compute'], C['neutral']], width=0.6)
for i, v in enumerate(shares):
    ax2.text(i, v + 4, f'{v:.0f}%', ha='center', fontsize=15, fontweight='bold')
ax2.set_title('Unmeasured harm:\nexposure rate', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 125)
for a in axes:
    a.set_xticks([0, 1])
    a.set_xticklabels(['5% canary', 'full'], fontsize=13)
    a.set_yticks([])
    a.spines['left'].set_visible(False)
sp.save(fig, out / 'harm_bars.svg')
s = Svg(900, 252)
marks = [(250, 'detected', AMB, 210), (430, 'admission stopped', RED, 210), (790, 'last effect settled', GRN, 210)]
rng = random.Random(4)
for i in range(9):
    st0 = 40 + i * 42
    end = min(st0 + 120 + rng.random() * 260, 770)
    s.rect(st0, 8 + i * 18, end - st0, 13, P['tool'][0], P['tool'][1], rx=3, sw=1.6)
s.line(20, 182, 880, 182, '#334155', 2)
for x, lab, col, y in marks:
    s.line(x, 4, x, 190, col, 3, dash=lab != 'admission stopped')
    s.text(x, y, lab, 20, col, bold=True)
s.text(450, 244, 'in-flight candidate tasks keep their identities until settled', 20, '#475569', italic=True)
s.save('rollback.svg')
pi = dict(gain=100 * 0.04, lo=100 * -0.032062854, hi=100 * 0.1112112101, margin=100 * 0.02, pairs=150, need=217)
half = (pi['gain'] - pi['lo']) * math.sqrt(pi['pairs'] / pi['need'])

def paired(name, show_217=True, size=(5.4, 2.4)):
    fig, ax = plt.subplots(figsize=size)
    ax.axvline(-pi['margin'], color=C['danger'], lw=2.5, ls='--')
    ax.text(-pi['margin'] - 0.3, 1.55, 'margin\n−2 points', ha='right', va='center', fontsize=14, fontweight='bold', color=C['danger'])
    ax.plot([pi['lo'], pi['hi']], [1, 1], color=VIO, lw=5)
    ax.plot([pi['gain']], [1], 'o', ms=11, color=VIO)
    ax.text(pi['hi'] + 0.3, 1, '150 pairs', va='center', fontsize=14, fontweight='bold', color=VIO)
    ax.text(pi['lo'] - 0.3, 0.72, '-3.21', ha='right', fontsize=13, fontweight='bold', color=VIO)
    if show_217:
        ax.plot([pi['gain'] - half, pi['gain'] + half], [0.25, 0.25], color=C['neutral'], lw=5, alpha=0.6)
        ax.plot([pi['gain']], [0.25], 'o', ms=11, color=C['neutral'], alpha=0.6)
        ax.text(pi['gain'] + half + 0.3, 0.25, '217 pairs (planned)', va='center', fontsize=14, color=C['neutral'], fontweight='bold')
    ax.set_xlim(-8, 21)
    ax.set_ylim(-0.2, 2.0)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Candidate minus baseline acceptance (points)', fontsize=13)
    ax.tick_params(labelsize=12)
    sp.save(fig, out / name)
paired('paired.svg')
paired('paired_wide.svg', size=(6.0, 1.6))

def prompt_ref():
    """A span carries a reference; the payload sits in a restricted store."""
    s = Svg(900, 230)
    s.box(10, 10, 360, 210, 'Span: model call', ('trace and span ids', 'model version, latency', 'content_ref = hash'), 'think', tsize=30, ssize=26)
    s.arrow([(370, 168), (518, 168)], VIO, width=3.5)
    s.text(444, 154, 'reference', 25, P['store'][2], bold=True)
    s.box(520, 10, 370, 210, 'Restricted store', ('raw prompt and output', 'access-controlled', 'expires on schedule'), 'store', tsize=30, ssize=26)
    s.save('prompt_ref.svg')
prompt_ref()

def thumb(name, body):
    (out / name).write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 100" width="260" height="100"><rect width="260" height="100" fill="white"/>' + body + '</svg>\n')
thumb('thumb_recall.svg', f'<line x1="60" y1="10" x2="60" y2="90" stroke="{RED}" stroke-width="4" stroke-dasharray="8,6"/><line x1="40" y1="50" x2="230" y2="50" stroke="{VIO}" stroke-width="8"/><circle cx="110" cy="50" r="11" fill="{VIO}"/>')
thumb('thumb_trace.svg', f'<rect x="8" y="8" width="244" height="12" fill="#cbd5e1"/><rect x="8" y="26" width="8" height="12" fill="{GRN}"/><rect x="16" y="42" width="60" height="12" fill="{SKY}" stroke="{VIO}" stroke-width="3"/><rect x="76" y="58" width="120" height="12" fill="{AMB}" stroke="{VIO}" stroke-width="3"/><rect x="76" y="74" width="40" height="12" fill="{AMB}"/><rect x="206" y="88" width="46" height="10" fill="{SKY}" stroke="{VIO}" stroke-width="3"/>')
thumb('thumb_retain.svg', f'<rect x="8" y="35" width="50" height="30" rx="4" fill="#ffffff" stroke="#94a3b8" stroke-width="3"/><rect x="88" y="35" width="60" height="30" rx="4" fill="#f3e8ff" stroke="{VIO}" stroke-width="3"/><line x1="58" y1="50" x2="88" y2="50" stroke="{GRY}" stroke-width="3"/>' + ''.join((f'<rect x="185" y="{8 + i * 23}" width="66" height="17" rx="3" fill="{f}" stroke="{c}" stroke-width="2.5"/><line x1="148" y1="50" x2="185" y2="{16 + i * 23}" stroke="{c}" stroke-width="2"/>' for i, (f, c) in enumerate((('#fee2e2', RED), ('#fef3c7', AMB), ('#e0f2fe', SKY), ('#f1f5f9', '#94a3b8'))))))
thumb('thumb_branch.svg', f'<rect x="10" y="10" width="12" height="22" fill="{GRN}"/><rect x="22" y="10" width="228" height="22" fill="{RED}"/><rect x="10" y="39" width="96" height="22" fill="{GRN}"/><rect x="106" y="39" width="144" height="22" fill="#94a3b8"/><rect x="10" y="68" width="240" height="22" fill="{GRN}"/>')
thumb('thumb_canary.svg', ''.join((f'<rect x="{10 + i * 62}" y="34" width="52" height="32" rx="4" fill="{f}" stroke="{c}" stroke-width="3"/>' for i, (f, c) in enumerate((('#fef3c7', AMB), ('#e0f2fe', SKY), ('#fef3c7', AMB), ('#dcfce7', GRN))))) + ''.join((f'<line x1="{62 + i * 62}" y1="50" x2="{72 + i * 62}" y2="50" stroke="{GRY}" stroke-width="3"/>' for i in range(3))) + f'<rect x="100" y="80" width="60" height="16" rx="3" fill="#fee2e2" stroke="{RED}" stroke-width="2.5"/>')
thumb('thumb_gate.svg', ''.join((f'<rect x="{12 + i * 62}" y="30" width="48" height="40" rx="5" fill="{f}" stroke="{c}" stroke-width="3"/>' for i, (f, c) in enumerate((('#fee2e2', RED), ('#f1f5f9', '#94a3b8'), ('#f1f5f9', '#94a3b8'), ('#f1f5f9', '#94a3b8'))))))
nx = f'<rect x="10" y="30" width="70" height="40" rx="5" fill="#f3e8ff" stroke="{VIO}" stroke-width="3"/>'
for i, (f, c) in enumerate((('#dcfce7', GRN), ('#fef3c7', AMB), ('#e0f2fe', SKY))):
    y = 8 + i * 31
    nx += f'<line x1="80" y1="50" x2="170" y2="{y + 12}" stroke="{c}" stroke-width="2.5"/><rect x="170" y="{y}" width="80" height="24" rx="4" fill="{f}" stroke="{c}" stroke-width="3"/>'
thumb('thumb_next.svg', nx)
print('figures written')
