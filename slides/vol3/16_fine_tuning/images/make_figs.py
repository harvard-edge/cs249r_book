#!/usr/bin/env python3
"""Figures for Lecture 16 (Trajectory Fine-Tuning), Volume III.

Every value comes from calcs.py, which this script imports and which reads
MLSysIM, so figures and slides agree; labels print the recorded strings
(deckvals.val). "TF" is the rendered chapter, Trajectory Fine-Tuning. The book's
SVGs (loss masking, block-diagonal mask, exposure bias and DAgger) are
page-sized and are redrawn here at slide size, split into build steps.
Full-width diagrams are 900 px wide with type of 18 px or more; column plots
are drawn at 5.4 x 3.6 in with 14 to 16 pt type. Plots use slideplot.

Run from this folder: python3 make_figs.py   (then make deck converts SVG)
"""
import mlsysim
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from mlsysim.physics import calc_behavior_cloning_cost_bound, calc_dagger_cost_bound
C = sp.C
plt = sp.plt
out = Path(__file__).parent
P = {'store': ('#f3e8ff', '#7c3aed', '#6d28d9'), 'tool': ('#fef3c7', '#d97706', '#b45309'), 'fault': ('#fee2e2', '#dc2626', '#b91c1c'), 'ok': ('#dcfce7', '#16a34a', '#15803d'), 'think': ('#e0f2fe', '#0284c7', '#0369a1'), 'plain': ('#ffffff', '#94a3b8', '#0f172a'), 'gray': ('#f1f5f9', '#94a3b8', '#475569')}
FONT = 'Helvetica, Arial, sans-serif'
VIO, RED, GRN, AMB, SKY, GRY, INK = ('#7c3aed', '#dc2626', '#16a34a', '#d97706', '#0284c7', '#64748b', '#0f172a')

class Svg:

    def __init__(self, w, h):
        self.w, self.h, self.items = (w, h, [])

    def add(self, s, show=True):
        self.items.append(s if show else f'<g opacity="0">{s}</g>')

    def _t(self, x, y, s, size, color, bold=False, anchor='middle', italic=False):
        w = ' font-weight="bold"' if bold else ''
        it = ' font-style="italic"' if italic else ''
        return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{w}{it}>{s}</text>'

    def text(self, x, y, s, size=19, color=INK, show=True, **kw):
        self.add(self._t(x, y, s, size, color, **kw), show)

    def box(self, x, y, w, h, title, subs=(), role='plain', show=True, dash=False, tsize=20, ssize=18, rx=6, sw=2.4, fill=None):
        f, stroke, text = P[role]
        d = ' stroke-dasharray="8,5"' if dash else ''
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or f}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        n = 1 + len(subs)
        total = tsize + ssize * len(subs) + 6 * (n - 1)
        yy = y + (h - total) / 2 + tsize * 0.8
        s += self._t(x + w / 2, yy, title, tsize, text, bold=True)
        for sub in subs:
            yy += 6 + ssize
            s += self._t(x + w / 2, yy, sub, ssize, '#1e293b')
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

    def circle(self, x, y, r, fill, stroke, show=True, dash=False, sw=2.2):
        d = ' stroke-dasharray="6,4"' if dash else ''
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>', show)

    def save(self, name):
        body = '\n'.join(self.items)
        (out / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}">\n<rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')
SHORT_LABEL = {'Claude Opus 4.6': 'Opus 4.6', 'Claude Sonnet 4': 'Sonnet 4', 'GPT-5.2-Codex': 'GPT-5.2-Codex', 'Claude 3.5 Sonnet': 'Sonnet 3.5'}
SHORT_SCAF = {'mini-SWE-agent 2.x': 'mini-SWE', 'SWE-agent 1.1': 'SWE-agent 1.1', 'OpenHands': 'OpenHands', 'SWE-agent 0.x (2024)': 'SWE-agent 0.x'}

def roles(step, name='roles', size=(5.9, 2.5), fs=13, legend=True):
    rows = [{'scaffold': 'SWE-agent 0.x (2024)', 'model': 'Claude 3.5 Sonnet', 'prompt': 19.7362732, 'obs': 67.78704867, 'model_out': 12.47667813}, {'scaffold': 'SWE-agent 1.1', 'model': 'Claude Sonnet 4', 'prompt': 1.64647904, 'obs': 71.65452596, 'model_out': 26.698995}, {'scaffold': 'OpenHands', 'model': 'Claude Sonnet 4', 'prompt': 5.833470988, 'obs': 67.7212736, 'model_out': 26.44525541}, {'scaffold': 'mini-SWE-agent 2.x', 'model': 'Claude Opus 4.6', 'prompt': 13.23413415, 'obs': 56.06936616, 'model_out': 30.69649969}, {'scaffold': 'mini-SWE-agent 2.x', 'model': 'GPT-5.2-Codex', 'prompt': 9.948601554, 'obs': 72.44305053, 'model_out': 17.60834791}]
    rows.sort(key=lambda r: r['model_out'])
    fig, ax = plt.subplots(figsize=size)
    for i, r in enumerate(rows):
        left = 0
        for key, color, tc in (('prompt', C['neutral_l'], C['text']), ('obs', C['data'], 'white'), ('model_out', '#334155', 'white')):
            w = r[key]
            hide = step == 1 and key == 'model_out'
            ax.barh(i, w, left=left, color='white' if hide else color, height=0.7, edgecolor=C['neutral'] if hide else 'white', hatch='//' if hide else None, linewidth=1.0)
            if w >= 9:
                ax.text(left + w / 2, i, '?' if hide else f'{w:.0f}%', ha='center', va='center', fontsize=fs, color=C['danger'] if hide else tc, fontweight='bold')
            left += w
        ax.text(-1.5, i, f"{SHORT_LABEL[r['model']]}, {SHORT_SCAF[r['scaffold']]}", ha='right', va='center', fontsize=fs - 1, color=C['text'])
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Share of transcript characters (%)', fontsize=fs - 1)
    ax.tick_params(axis='x', labelsize=fs - 1)
    if not legend:
        ax.set_ylim(-0.6, len(rows) - 0.4)
        sp.save(fig, out / f'{name}_step{step}.svg')
        return
    ax.text(2, len(rows) - 0.2, 'prompt', fontsize=fs, color=C['neutral'], fontweight='bold')
    ax.text(40, len(rows) - 0.2, 'observations', fontsize=fs, color=C['data'], fontweight='bold')
    ax.text(100, len(rows) - 0.2, 'model output', fontsize=fs, color='#334155', fontweight='bold', ha='right')
    ax.set_ylim(-0.6, len(rows) + 0.2)
    sp.save(fig, out / f'{name}_step{step}.svg')
for k in (1, 2):
    roles(k)
roles(2, 'roles_col', size=(4.2, 2.8), fs=14, legend=False)

def homes(step):
    s = Svg(900, 400)
    cols = [('Weights', 'think', ('slow-changing procedure',), ('check diagnostics first', 'read schema after refusal')), ('Context', 'store', ('per-deployment facts',), ('tool schemas', 'permissions, observations')), ('Runtime', 'ok', ('invariants that hold',), ('validation, authorization', 'sandbox, timeouts'))]
    for i, (name, role, sub, ex) in enumerate(cols):
        x = 14 + i * 298
        show = step >= i + 1
        s.box(x, 20, 276, 116, name, sub, role, show=show, tsize=30, ssize=22)
        s.text(x + 138, 180, 'Repair agent:', 22, GRY, bold=True, show=show)
        s.text(x + 138, 214, ex[0], 22, show=show)
        s.text(x + 138, 246, ex[1], 22, show=show)
    s.rect(14, 282, 872, 104, '#faf5ff', VIO, rx=6, sw=2, show=step >= 3)
    s.text(450, 324, 'Fine-tuning changes only the first column:', 24, VIO, bold=True, show=step >= 3)
    s.text(450, 362, 'what the model proposes, not what the runtime permits', 24, INK, show=step >= 3)
    s.save(f'homes_step{step}.svg')
for k in (1, 2, 3):
    homes(k)

def triangle():
    s = Svg(540, 360)
    H = 6
    s.text(270, 30, 'One example per model call', 20, INK, bold=True)
    for t in range(1, H + 1):
        y = 50 + (t - 1) * 40
        s.text(58, y + 25, f'call {t}', 17, GRY, anchor='end')
        for j in range(t):
            fill = P['think'][0] if j < t - 1 else P['store'][0]
            st = P['think'][1] if j < t - 1 else VIO
            s.rect(70 + j * 66, y + 6, 62, 28, fill, st, rx=3, sw=1.6)
    s.text(270, 318, 'stored tokens grow as H(H+1)/2', 19, INK)
    s.text(270, 346, '30 turns: 15.5× one transcript', 19, VIO, bold=True)
    s.save('triangle.svg')

def swe_tokens():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    vals = [387195.5, 3852.5]
    texts = ['387,196', '3,852']
    labels = ['input, summed\nover calls', 'output']
    cols = [C['data'], '#334155']
    ax.bar([0, 1], vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.03, texts[i], ha='center', va='bottom', fontsize=15, fontweight='bold', color=cols[i])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Tokens per run (median)', fontsize=14)
    ax.set_ylim(0, 387195.5 * 1.25)
    ax.tick_params(axis='y', labelsize=13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v / 1000:.0f}k'))
    ax.text(0.5, 387195.5 * 0.62, '102×', ha='center', fontsize=22, fontweight='bold', color=C['danger'])
    ax.set_title('500 SWE-agent runs, median 27 calls', fontsize=14)
    sp.save(fig, out / 'swe_tokens.svg')
triangle()
swe_tokens()

def mask(step):
    s = Svg(900, 360)
    segs = [('Prompt', 0, 'supplied', 0), ('Task', 1, 'supplied', 0), ('e2', 2, 'policy_dash', 0), ('Refusal', 3, 'env', 0), ('e4', 4, 'policy', 1), ('Result', 5, 'env', 0), ('e6', 6, 'policy', 1)]
    lane_y = {'supplied': 6, 'policy': 72, 'policy_dash': 72, 'env': 138}
    s.text(4, 42, 'Supplied', 22, INK, bold=True, anchor='start')
    s.text(4, 108, 'Policy', 22, SKY, bold=True, anchor='start')
    s.text(4, 174, 'Tools', 22, AMB, bold=True, anchor='start')
    for y in (66, 132, 198):
        s.line(4, y, 896, y, '#cbd5e1', 1.5, dash=True)
    x0, wd = (130, 110)
    for a, i, lane, m in segs:
        x = x0 + i * wd
        y = lane_y[lane]
        role = {'supplied': 'gray', 'policy': 'think', 'policy_dash': 'think', 'env': 'tool'}[lane]
        s.box(x, y + 6, wd - 10, 48, a, (), role, tsize=22, dash=lane == 'policy_dash', sw=3.4 if lane == 'policy' and step >= 2 else 2.2)
        if step >= 2:
            fill = SKY if m else 'white'
            s.rect(x, 212, wd - 10, 44, fill, SKY if m else '#cbd5e1', rx=3, sw=2)
            s.text(x + (wd - 10) / 2, 242, str(m), 24, 'white' if m else GRY, bold=True)
    s.text(4, 242, 'Mask', 22, INK, bold=True, anchor='start', show=step >= 2)
    if step >= 3:
        y = 292
        s.arrow([(x0 + 6 * wd + 50, 256), (x0 + 6 * wd + 50, y), (x0 + 8, y)], GRN, width=2.8)
        s.line(x0 + 4 * wd + 50, 256, x0 + 4 * wd + 50, y, GRN, 2.8)
        for i in (0, 1, 2, 3, 5):
            s.circle(x0 + i * wd + 50, y, 8, 'white', GRN)
        s.text(4, y + 8, 'Read', 22, GRN, bold=True, anchor='start')
        s.text(480, 340, 'masked tokens are still read by later targets', 22, GRN)
    s.save(f'mask_step{step}.svg')
for k in (1, 2, 3):
    mask(k)

def inventory():
    fig, ax = plt.subplots(figsize=(5.9, 1.6))
    rows = [('rationale selected', 100 * 0.1342773438, '13.4'), ('rationale as context', 100 * 0.048828125, '4.9')]
    for i, (lab, tg, tg_text) in enumerate(rows):
        y = 1 - i
        obs = 100 * 0.771484375
        ctx = 100 - obs - tg
        ax.barh(y, obs, color=C['data'], height=0.6)
        ax.barh(y, ctx, left=obs, color=C['neutral_l'], height=0.6)
        ax.barh(y, tg, left=obs + ctx, color='#334155', height=0.6)
        ax.text(-1.5, y, lab, ha='right', va='center', fontsize=13, fontweight='bold')
        ax.text(obs / 2, y, 'observations 77.1%', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
        ax.text(101.5, y, f'targets {tg_text}%', ha='left', va='center', fontsize=12, color='#334155', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Share of 16,384 stored tokens (%), illustrative', fontsize=13)
    ax.tick_params(axis='x', labelsize=13)
    sp.save(fig, out / 'inventory.svg')
inventory()

def norm():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = [0, 1]
    tok = [100 * 0.0303030303, 100 * 0.9696969697]
    ex = [100 * 0.5, 100 * 0.5]
    ax.bar([v - 0.19 for v in x], tok, width=0.36, color=C['memory'], label='token averaging')
    ax.bar([v + 0.19 for v in x], ex, width=0.36, color=C['compute'], label='example averaging')
    for xi, v, e in zip(x, tok, ex):
        ax.text(xi - 0.19, v + 2, f'{v:.1f}%', ha='center', fontsize=14, color=C['memory'], fontweight='bold')
        ax.text(xi + 0.19, e + 2, f'{e:.0f}%', ha='center', fontsize=14, color=C['compute'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['lookup\n256 targets', 'repair trace\n8,192 targets'], fontsize=14)
    ax.set_ylabel('Weight in batch loss (%)', fontsize=14)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='y', labelsize=13)
    ax.text(-0.4, 115, 'token avg.', color=C['memory'], fontsize=14, fontweight='bold')
    ax.text(0.45, 115, 'example avg.', color=C['compute'], fontsize=14, fontweight='bold')
    sp.save(fig, out / 'norm.svg')
norm()

def occupancy():
    sp.hbars(out / 'occupancy.svg', [('padded', 100 * 0.19140625, 'neutral', '19.1%'), ('packed', 100 * 0.765625, 'data', '76.6%')], '%', log=False, xlabel='Occupied share of positions (%)', size=(3.3, 1.7))
occupancy()

def blockmask(step):
    s = Svg(640, 360)
    n, s1 = (8, 5)
    cell = 32

    def grid(ox, title, block):
        s.text(ox + n * cell / 2, 30, title, 24, INK, bold=True)
        for i in range(n):
            for j in range(n):
                x, y = (ox + j * cell, 46 + i * cell)
                causal = j <= i
                cross = i >= s1 and j < s1
                if not causal or (cross and block):
                    fill = '#e2e8f0'
                elif cross:
                    fill = '#fca5a5'
                else:
                    fill = '#bae6fd'
                s.rect(x, y, cell - 2, cell - 2, fill)
        s.line(ox + s1 * cell - 1, 42, ox + s1 * cell - 1, 48 + n * cell, INK, 2.4)
        s.line(ox - 4, 46 + s1 * cell - 1, ox + n * cell, 46 + s1 * cell - 1, INK, 2.4)
        s.text(ox + s1 * cell / 2, 46 + n * cell + 28, 'ex. 1', 22, GRY)
        s.text(ox + s1 * cell + (n - s1) * cell / 2, 46 + n * cell + 28, 'ex. 2', 22, GRY)
    grid(20, 'Causal', False)
    s.text(20 + s1 * cell / 2, 46 + 6.75 * cell, 'leak', 24, RED, bold=True)
    grid(360, 'Block-diagonal', True)
    s.text(360 + s1 * cell / 2, 46 + 6.75 * cell, '−∞', 26, GRY, bold=True)
    if step < 2:
        s.rect(340, 0, 300, 330, 'white')
    s.rect(20, 334, 20, 20, '#bae6fd')
    s.text(46, 351, 'score', 22, INK, anchor='start')
    s.rect(140, 334, 20, 20, '#e2e8f0')
    s.text(166, 351, '−∞', 22, INK, anchor='start')
    s.rect(240, 334, 20, 20, '#fca5a5')
    s.text(266, 351, 'cross-example', 22, INK, anchor='start')
    s.save(f'blockmask_step{step}.svg')
for k in (1, 2):
    blockmask(k)

def offcorpus():
    s = Svg(540, 360)
    s.rect(14, 40, 512, 96, '#f5f3ff', VIO, rx=8, sw=2, dash=True)
    s.text(24, 30, 'states in the corpus', 19, VIO, anchor='start')
    xs = [60, 170, 280, 470]
    for i, x in enumerate(xs):
        s.circle(x, 88, 30, 'white', INK)
        s.text(x, 96, ['s1', 's2', 's3', 'sH'][i], 20, INK, italic=True)
    for a, b in ((90, 140), (200, 250), (310, 360)):
        s.arrow([(a, 88), (b, 88)], GRY)
    s.text(395, 96, '…', 24, GRY)
    s.arrow([(170, 118), (170, 200), (226, 200)], RED, width=2.6)
    s.text(150, 170, 'learner', 18, RED, anchor='end')
    s.text(150, 192, 'deviates', 18, RED, anchor='end')
    for i, x in enumerate((260, 360, 460)):
        s.circle(x, 200 + i * 22, 28, '#fee2e2', RED, dash=True)
        if i:
            s.arrow([(x - 70, 200 + (i - 1) * 22 + 6), (x - 30, 200 + i * 22)], RED, dash=True)
    s.text(270, 300, 'version conflict: no recorded correction', 19, RED, bold=True)
    s.text(270, 336, 'teacher forcing never visits these states', 18, INK)
    s.save('offcorpus.svg')
offcorpus()

def bounds(step):
    H = 30
    hs = [h / 10 for h in range(0, 10 * H + 1)]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(hs, [min(h, calc_behavior_cloning_cost_bound(h, 0.02)) for h in hs], color=C['danger'], lw=3, alpha=1 if step >= 2 else 0)
    ax.plot(hs, [min(h, calc_dagger_cost_bound(h, 0.02, 1.0)) for h in hs], color=C['memory'], lw=3, alpha=1 if step >= 2 else 0)
    ax.plot(hs, hs, color=C['neutral'], lw=1.5, ls='--')
    ax.text(3, 10, 'cap: H', color=C['neutral'], fontsize=13)
    if step >= 2:
        ax.plot([H], [18.0], 'o', color=C['danger'], ms=9)
        ax.plot([H], [0.6], 'o', color=C['memory'], ms=9)
        ax.text(H - 0.6, 18.0 + 1.6, 'H²ε = 18', ha='right', color=C['danger'], fontsize=15, fontweight='bold')
        ax.text(H - 0.6, 0.6 + 1.4, 'uHε = 0.6', ha='right', color=C['memory'], fontsize=15, fontweight='bold')
        ax.text(1, 27, 'reference states', color=C['danger'], fontsize=14, fontweight='bold')
        ax.text(1, 24.4, 'learner states, u = 1', color=C['memory'], fontsize=14, fontweight='bold')
    else:
        ax.text(15, 9, '?', fontsize=40, color=C['danger'], fontweight='bold', ha='center')
    ax.set_xlim(0, H + 1)
    ax.set_ylim(0, 31)
    ax.set_xlabel('Horizon H (decision steps)', fontsize=14)
    ax.set_ylabel('Excess cost bound', fontsize=14)
    ax.tick_params(labelsize=13)
    sp.save(fig, out / f'bounds_step{step}.svg')
for k in (1, 2):
    bounds(k)

def dagger(step):
    s = Svg(900, 360)
    names = [('Run πk', ('+ reference',), 'tool'), ('Visited', ('states',), 'gray'), ('Labeler', ('proposes',), 'think'), ('Sandbox', ('verifier',), 'tool'), ('Aggregate', ('old + new',), 'store'), ('Retrain', ('πk+1',), 'think')]
    w, gap = (128, 18)
    for i, (a, subs, role) in enumerate(names):
        x = 12 + i * (w + gap)
        s.box(x, 60, w, 90, a, subs, role, tsize=20, ssize=18)
        if i:
            s.arrow([(x - gap, 105), (x - 2, 105)], GRY)
    s.arrow([(12 + 5 * (w + gap) + w / 2, 60), (12 + 5 * (w + gap) + w / 2, 30), (12 + w / 2, 30), (12 + w / 2, 58)], GRY)
    s.text(450, 24, 'next round', 18, GRY)
    vx = 12 + 3 * (w + gap) + w / 2
    s.arrow([(vx, 150), (vx, 190)], RED, dash=True)
    s.text(vx, 212, 'discard', 18, RED)
    sh = step >= 2
    s.text(14, 262, 'Fault-injected record', 19, INK, bold=True, anchor='start', show=sh)
    for i, (lab, m, role) in enumerate((('injected error', 0, 'tool'), ('diagnosis', 1, 'think'), ('corrective call', 1, 'think'))):
        x = 14 + i * 168
        s.box(x, 274, 160, 40, lab, (), role, tsize=17, show=sh)
        s.rect(x, 316, 160, 34, SKY if m else 'white', SKY if m else '#cbd5e1', sw=2, show=sh)
        s.text(x + 80, 340, f'mask {m}', 18, 'white' if m else GRY, bold=True, show=sh)
    s.arrow([(12 + (w + gap) + w / 2, 268), (12 + (w + gap) + w / 2, 152)], GRY, show=sh)
    s.box(560, 262, 330, 88, 'Learner-state bound', ('min{H, uHε}',), 'ok', tsize=19, ssize=19, show=step >= 3)
    s.save(f'dagger_step{step}.svg')
for k in (1, 2, 3):
    dagger(k)

def stk():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    cols = {'Supervised': C['neutral'], 'SMILe': C['compute'], 'DAgger': C['memory']}
    for name in cols:
        pts = sorted({'Supervised': ({'method': 'Supervised', 'iteration': 1.0, 'falls_per_lap': 3.0, 'states_per_iteration': 1000.0}, {'method': 'Supervised', 'iteration': 5.0, 'falls_per_lap': 2.9, 'states_per_iteration': 1000.0}, {'method': 'Supervised', 'iteration': 10.0, 'falls_per_lap': 3.2, 'states_per_iteration': 1000.0}, {'method': 'Supervised', 'iteration': 15.0, 'falls_per_lap': 2.9, 'states_per_iteration': 1000.0}, {'method': 'Supervised', 'iteration': 20.0, 'falls_per_lap': 4.0, 'states_per_iteration': 1000.0}), 'SMILe': ({'method': 'SMILe', 'iteration': 1.0, 'falls_per_lap': 3.0, 'states_per_iteration': 1000.0}, {'method': 'SMILe', 'iteration': 5.0, 'falls_per_lap': 2.35, 'states_per_iteration': 1000.0}, {'method': 'SMILe', 'iteration': 10.0, 'falls_per_lap': 2.1, 'states_per_iteration': 1000.0}, {'method': 'SMILe', 'iteration': 15.0, 'falls_per_lap': 1.85, 'states_per_iteration': 1000.0}, {'method': 'SMILe', 'iteration': 20.0, 'falls_per_lap': 1.95, 'states_per_iteration': 1000.0}), 'DAgger': ({'method': 'DAgger', 'iteration': 1.0, 'falls_per_lap': 3.0, 'states_per_iteration': 1000.0}, {'method': 'DAgger', 'iteration': 5.0, 'falls_per_lap': 0.3, 'states_per_iteration': 1000.0}, {'method': 'DAgger', 'iteration': 10.0, 'falls_per_lap': 0.35, 'states_per_iteration': 1000.0}, {'method': 'DAgger', 'iteration': 15.0, 'falls_per_lap': 0.0, 'states_per_iteration': 1000.0}, {'method': 'DAgger', 'iteration': 20.0, 'falls_per_lap': 0.0, 'states_per_iteration': 1000.0})}[name], key=lambda r: r['iteration'])
        ax.plot([r['iteration'] for r in pts], [r['falls_per_lap'] for r in pts], '-o', color=cols[name], lw=3, ms=7)
    ax.text(20.4, 4.0, 'supervised', color=cols['Supervised'], fontsize=14, fontweight='bold', va='center')
    ax.text(20.4, 1.95, 'SMILe', color=cols['SMILe'], fontsize=14, fontweight='bold', va='center')
    ax.text(20.4, 0.15, 'DAgger', color=cols['DAgger'], fontsize=14, fontweight='bold', va='center')
    ax.set_xlim(0, 26)
    ax.set_ylim(-0.1, 4.5)
    ax.set_xticks(sorted({1.0, 10.0, 15.0, 20.0, 5.0}))
    ax.set_xlabel('Training iterations (about 1,000 states each)', fontsize=13)
    ax.set_ylabel('Falls off the track per lap', fontsize=14)
    ax.tick_params(labelsize=13)
    sp.save(fig, out / 'stk.svg')
stk()

def prefpair():
    s = Svg(900, 250)
    s.box(6, 85, 190, 80, 'Same prefix', ('diagnosis done',), 'gray', tsize=22, ssize=20)
    s.arrow([(196, 110), (276, 60)], GRY)
    s.arrow([(196, 140), (276, 190)], GRY)
    s.box(276, 14, 430, 92, 'Targeted tests, suite once', ('leak fixed',), 'ok', tsize=23, ssize=21)
    s.box(276, 144, 430, 92, 'Full suite after every edit', ('leak fixed, slower',), 'tool', tsize=23, ssize=21)
    s.text(805, 56, 'preferred', 24, GRN, bold=True)
    s.add('<text x="805" y="88" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#16a34a" text-anchor="middle" font-style="italic">z<tspan baseline-shift="sub" font-size="21">w</tspan></text>')
    s.text(805, 186, 'dispreferred', 24, AMB, bold=True)
    s.add('<text x="805" y="218" font-family="Helvetica, Arial, sans-serif" font-size="24" fill="#d97706" text-anchor="middle" font-style="italic">z<tspan baseline-shift="sub" font-size="21">l</tspan></text>')
    s.save('prefpair.svg')
prefpair()

def distill():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    vals = [48000.0 / 1000, 4800.0 / 1000, 15000 / 1000]
    labs = ['saved,\nno caching', 'saved,\ncached prefix', 'redistill\nonce a month']
    cols = [C['data'], C['memory'], C['danger']]
    ax.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f'${v:.1f}k', ha='center', fontsize=15, fontweight='bold', color=cols[i])
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=13)
    ax.set_ylabel('Dollars per month (thousands)', fontsize=13)
    ax.set_ylim(0, 58)
    ax.tick_params(axis='y', labelsize=13)
    sp.save(fig, out / 'distill.svg')
distill()

def loramem():
    full, frozen, total = (128.0, 16.0, 16.13421773)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(0, full, color=C['compute'], width=0.55)
    ax.bar(1, frozen, color=C['neutral'], width=0.55)
    ax.bar(1, 0.134217728, bottom=frozen, color=C['compute'], width=0.55)
    ax.text(0, full + 3, '128 GB', ha='center', fontsize=15, fontweight='bold', color=C['compute'])
    ax.text(1, total + 3, '16.13 GB', ha='center', fontsize=15, fontweight='bold', color=C['text'])
    ax.text(1.35, frozen / 2, 'frozen base\n2 B/param', fontsize=13, color=C['neutral'], va='center', fontweight='bold')
    ax.text(0.35, 95, '16 B per\ntrained param', fontsize=13, color=C['compute'], fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['full\nfine-tuning', 'LoRA run\n(0.10% trained)'], fontsize=14)
    ax.set_xlim(-0.5, 2.1)
    ax.set_ylim(0, 145)
    ax.set_ylabel('Parameter state (GB)', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    sp.save(fig, out / 'loramem.svg')
loramem()

def thumb(name, draw):
    s = Svg(300, 110)
    draw(s)
    s.save(f'thumb_{name}.svg')
thumb('homes', lambda s: [s.rect(10 + i * 98, 20, 86, 70, P[r][0], P[r][1], rx=6, sw=3) for i, r in enumerate(('think', 'store', 'ok'))])

def _mask(s):
    for i, m in enumerate((0, 0, 0, 0, 1, 0, 1)):
        s.rect(8 + i * 41, 35, 37, 40, SKY if m else 'white', SKY if m else '#94a3b8', sw=3)
thumb('mask', _mask)

def _pack(s):
    s.rect(10, 20, 280, 30, '#bae6fd', INK, sw=2)
    s.rect(10, 60, 280, 30, '#e2e8f0', INK, sw=2)
    s.rect(10, 60, 170, 30, '#bae6fd', 'none')
    s.rect(180, 60, 110, 30, '#bbf7d0', 'none')
    s.line(180, 56, 180, 94, INK, 3)
thumb('pack', _pack)

def _bound(s):
    s.add('<path d="M10,100 Q150,95 290,10" fill="none" stroke="#dc2626" stroke-width="5"/>')
    s.add('<path d="M10,100 L290,92" fill="none" stroke="#006395" stroke-width="5"/>')
thumb('bound', _bound)

def _pref(s):
    s.rect(10, 40, 70, 30, '#f1f5f9', '#94a3b8', sw=2)
    s.arrow([(80, 50), (140, 25)], GRY)
    s.arrow([(80, 60), (140, 85)], GRY)
    s.rect(140, 8, 150, 30, P['ok'][0], P['ok'][1], sw=2)
    s.rect(140, 72, 150, 30, P['tool'][0], P['tool'][1], sw=2)
thumb('pref', _pref)

def _gate(s):
    s.rect(20, 20, 260, 70, '#f5f3ff', VIO, rx=8, sw=3)
    for i in range(4):
        s.circle(65 + i * 57, 55, 16, P['ok'][0] if i < 3 else P['fault'][0], P['ok'][1] if i < 3 else P['fault'][1], sw=3)
thumb('gate', _gate)

def _recall(s):
    for i in range(5):
        s.rect(10 + i * 58, 35, 50, 40, P['think'][0] if i in (1, 2, 4) else '#f1f5f9', P['think'][1] if i in (1, 2, 4) else '#94a3b8', sw=3, dash=i == 0)
thumb('recall', _recall)

def _next(s):
    for i in range(6):
        s.circle(30 + i * 48, 55, 16, P['ok'][0] if i % 2 == 0 else P['fault'][0], P['ok'][1] if i % 2 == 0 else P['fault'][1], sw=3)
    s.arrow([(10, 95), (290, 95)], VIO, width=3)
thumb('next', _next)
print('figures written')
