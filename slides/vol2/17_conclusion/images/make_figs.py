#!/usr/bin/env python3
"""Figures for Lecture 17 (Conclusion), Volume II.

Plots use the shared slideplot helper; diagrams are SVG written by the small
builder below, one file per build step on one canvas so nothing moves. Every
number comes from MLSysIM through calcs.py in this folder (imported, never
retyped): printed labels use val(key), plotted points num(key) or the same
Scenarios.* and Literature series calcs.py reads. Axis ranges and tick
positions are layout.

Diagrams follow the Volume II C3 roles: compute blue #1F407A, communication
amber #d97706, coordination violet #7c3aed, storage green, risk red.

Run from this folder: python3 make_figs.py
"""
import contextlib
import io
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
plt = sp.plt
C = sp.C
out = HERE
ETH = '#1F407A'
ETH_L = '#EAF1F8'
AMBER, AMBER_L = ('#d97706', '#fef3c7')
VIOLET, VIOLET_L = ('#7c3aed', '#f3e8ff')
GREEN, GREEN_L = ('#16a34a', '#dcfce7')
RED, RED_L = ('#dc2626', '#fee2e2')
SLATE, SLATE_L = ('#64748b', '#f1f5f9')
TEXT = '#1A1A2E'
FONT = 'font-family="Helvetica, Arial, sans-serif"'

class Svg:
    """A tiny SVG builder: every element carries the first step it appears in;
    write(stem, n) emits stem_step1..n with hidden elements kept transparent."""

    def __init__(self, w, h):
        self.w, self.h, self.items = (w, h, [])

    def add(self, first, s, last=99):
        self.items.append((first, last, s))

    def rect(self, x, y, w, h, fill, stroke, first=1, sw=2, rx=8, last=99):
        self.add(first, f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>', last)

    def text(self, x, y, s, size=22, color=TEXT, weight='normal', anchor='middle', first=1, last=99):
        lines = s.replace('&', '&amp;').split('\n')
        y0 = y - (len(lines) - 1) * size * 0.6
        spans = ''.join((f'<tspan x="{x}" y="{y0 + i * size * 1.2:.1f}">{ln}</tspan>' for i, ln in enumerate(lines)))
        self.add(first, f'<text {FONT} font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="middle">{spans}</text>', last)

    def line(self, x1, y1, x2, y2, color, first=1, sw=2.5, arrow=True, last=99):
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ''
        self.add(first, f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{m}/>', last)

    def write(self, stem, steps):
        colors = [ETH, AMBER, VIOLET, GREEN, RED, SLATE, TEXT]
        defs = ''.join((f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in colors))
        for k in range(1, steps + 1):
            body = [s if first <= k <= last else '<g opacity="0">' + s + '</g>' for first, last, s in self.items]
            svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}"><defs>{defs}</defs><rect width="100%" height="100%" fill="white"/>' + ''.join(body) + '</svg>'
            name = f'{stem}_step{k}.svg' if steps > 1 else f'{stem}.svg'
            (out / name).write_text(svg)
rows = [('Governance & Sustainability', "Same energy: Poland grid 41× Quebec's carbon", 'L15'), ('Assurance', 'DP noise: $200 error on a mean of 1,000 salaries', 'L13'), ('Operations', '864 false alarms a day from 1,000 monitors', 'L12'), ('Runtime & Serving', '12 requests of 128K tokens fill one 8-GPU node', 'L10'), ('Reliability & Control', 'Checkpoint every 13.8 min at 10,000 GPUs', 'L7'), ('Distributed Execution', '175B, pure data parallel, 1,024 GPUs: 15% efficient', 'L6'), ('Fabric & Storage', '2,450 GB checkpoint: 0.34 s to local NVMe', 'L4'), ('Facility & Accelerators', '10 GB: HBM 3.0 ms, NVLink 22.2, InfiniBand 200', 'L2')]
g = Svg(900, 372)
for i, (name, number, lec) in enumerate(rows):
    y = 4 + i * 46
    g.rect(4, y, 892, 42, ETH_L, ETH, sw=1.5, rx=6)
    g.text(16, y + 21, name, 19, ETH, 'bold', 'start')
    g.text(300, y + 21, number, 19, TEXT, 'normal', 'start')
    g.text(884, y + 21, lec, 19, ETH, 'bold', 'end')
g.write('stack_numbers', 1)
g = Svg(900, 380)
bands = [(4, 96, 'Governance', [('5  Responsibility constrains design', 'spends compute; approval gates add coordination'), ('6  Sustainability is a first-order cost', "a step's energy = average power × T_step")], GREEN, 3), (106, 60, 'Serving', [('4  Serving is bounded by its contract', 'the same terms, counted per request')], VIOLET, 2), (172, 126, 'Distribution', [('2  Communication can dominate', 'T_comm(N) and T_sync(N) need not shrink'), ('3  Failure is routine', 'recovery adds to T_sync(N); rollback adds work')], AMBER, 1), (304, 72, 'Infrastructure', [('1  Infrastructure determines capability', 'sets the rates behind T_compute / N and T_comm(N)')], ETH, 1)]
for y, h, band, prin, color, step in bands:
    g.rect(64, y, 832, h, '#ffffff', color, sw=2.5, first=step, rx=6)
    g.text(78, y + h / 2, band, 19, color, 'bold', 'start', first=step)
    n = len(prin)
    for j, (pr, term) in enumerate(prin):
        yy = y + h * (j + 0.5) / n
        g.text(250, yy - 11, pr, 19, TEXT, 'bold', 'start', first=step)
        g.text(250, yy + 12, term, 18, SLATE, 'normal', 'start', first=step)
g.line(30, 372, 30, 12, RED, sw=4, first=3)
g.add(3, f'<text {FONT} x="14" y="190" font-size="18" font-weight="700" fill="{RED}" text-anchor="middle" dominant-baseline="middle" transform="rotate(-90 14 190)">the scale moment</text>')
g.write('principles', 3)
tail_rows = [('One random leaf', 10.0, 'memory', '10 ms'), ('95% of all leaves', 70.0, 'compute', '70 ms'), ('All leaves', 140.0, 'danger', '140 ms')]
sp.hbars(out / 'tail_measured.svg', tail_rows, 'ms', log=False, size=(5.8, 1.9))
import numpy as np
mttf_h = 50000.0
fig, ax = plt.subplots(figsize=(5.8, 2.75))
ns = np.logspace(2, 4.6, 100)
ax.plot(ns, mttf_h / ns, color=C['memory'], lw=2.5)
ax.text(150, mttf_h / 150 * 1.5, 'model: 50,000 h / N', color=C['memory'], fontsize=12, fontweight='bold')
pts = [(1024.0, 48.828125, '1,024: 48.8 h'), (2048.0, 24.4140625, '2,048: 24.4 h')]
for (x, y, lab), off, ha, va in zip(pts, [(-8, -6), (8, 6)], ['right', 'left'], ['top', 'bottom']):
    ax.plot([x], [y], 'o', ms=9, color=C['memory'], mec='white', mew=1.5, zorder=5)
    ax.annotate(lab, (x, y), xytext=off, textcoords='offset points', ha=ha, va=va, fontsize=13, color=C['memory'], fontweight='bold')
x = 16384.0
llama_h = 1296.0 / 419.0
ax.plot([x], [llama_h], 'D', ms=9, color=C['danger'], zorder=6)
ax.annotate('Llama 3, measured: 3.1 h', (x, 3.1), xytext=(-14, -16), textcoords='offset points', ha='right', va='top', fontsize=12, color=C['danger'], fontweight='bold')
ax.plot([x], [1.8], 'o', ms=8, mfc='white', mec=C['danger'], mew=2, zorder=6)
ax.annotate('projected: 1.8 h', (x, 1.8), xytext=(-14, -22), textcoords='offset points', ha='right', va='top', fontsize=12, color=C['danger'])
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(100, 40000.0)
ax.set_ylim(0.25, 800)
ax.set_xticks([100, 1000, 10000])
ax.set_xticklabels(['100', '1,000', '10,000'])
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(['1 h', '10 h', '100 h'])
ax.minorticks_off()
ax.set_xlabel('GPUs the job needs at once')
ax.set_ylabel('Hours between interruptions')
ax.grid(color=C['grid'], lw=0.6)
sp.save(fig, out / 'mtbi.svg')
fig, ax = plt.subplots(figsize=(2.9, 1.35))
bars = [('run time', 1 + -0.4964585305, '−50%'), ('accel.-hours', 1 + 0.007082939005, '+0.7%')]
for i, (lab, r, txt) in enumerate(bars):
    y = 1 - i
    ax.barh(y + 0.19, 1, height=0.34, color=C['neutral_l'], edgecolor=C['neutral'])
    ax.barh(y - 0.19, r, height=0.34, color=C['memory'])
    ax.text(-0.04, y, lab, ha='right', va='center', fontsize=14, fontweight='bold', color=C['text'])
    ax.text(r + 0.03, y - 0.19, txt, va='center', fontsize=14, fontweight='bold', color=C['memory'])
ax.text(1.03, 1.19, '1,024', va='center', fontsize=13, color=C['neutral'])
ax.text(0.03, 0.81, '2,048', va='center', fontsize=13, color='white', fontweight='bold')
ax.set_xlim(0, 1.45)
ax.set_ylim(-0.5, 1.5)
ax.set_yticks([])
ax.set_xticks([])
for s in ('left', 'top', 'right', 'bottom'):
    ax.spines[s].set_visible(False)
sp.save(fig, out / 'expand_bars.svg')
fig, ax = plt.subplots(figsize=(5.8, 2.6))
ns = np.logspace(np.log10(128), np.log10(16384), 100)
ov = [0.5902918299, 0.6049357445, 0.6199429442, 0.6353224413, 0.6510834719, 0.6672355008, 0.6837882281, 0.7007515942, 0.7181357862, 0.7359512438, 0.754208666, 0.7729190169, 0.7920935328, 0.8117437286, 0.8318814049, 0.8525186552, 0.8736678728, 0.8953417586, 0.9175533285, 0.9403159213, 0.9636432067, 0.9875491937, 1.012048239, 1.037155054, 1.062884717, 1.08925268, 1.116274777, 1.143967236, 1.172346688, 1.201430175, 1.231235163, 1.261779551, 1.293081682, 1.325160353, 1.35803483, 1.391724854, 1.426250658, 1.461632975, 1.497893054, 1.53505267, 1.57313414, 1.612160331, 1.652154681, 1.693141208, 1.735144526, 1.778189859, 1.822303057, 1.867510611, 1.913839672, 1.96131806, 2.009974289, 2.059837577, 2.11093787, 2.163305855, 2.216972982, 2.271971478, 2.328334372, 2.386095514, 2.445289589, 2.505952146, 2.568119615, 2.63182933, 2.697119551, 2.764029486, 2.832599318, 2.902870226, 2.974884408, 3.048685113, 3.12431666, 3.201824469, 3.281255086, 3.362656211, 3.446076729, 3.531566737, 3.619177575, 3.708961855, 3.800973497, 3.895267757, 3.991901262, 4.090932043, 4.192419572, 4.296424796, 4.403010174, 4.512239714, 4.624179012, 4.738895291, 4.856457443, 4.976936068, 5.100403517, 5.226933938, 5.356603315, 5.489489521, 5.625672357, 5.765233607, 5.908257081, 6.054828671, 6.205036398, 6.358970466, 6.516723319, 6.678389693]
ax.plot(ns, ov, color=C['network'], lw=2.5)
for gpus, pct_key, ov_n, off, ha in [('ref_gpus', 'ov_ref_pct', 0.01669597423, (-10, 10), 'right'), ('exp_gpus', 'ov_exp_pct', 0.0236116732, (10, -18), 'left')]:
    n, v = ({'right': 1024.0, 'left': 2048.0}[ha], 100 * ov_n)
    ax.plot([n], [v], 'o', ms=9, color=C['network'], mec='white', mew=1.5, zorder=5)
    ax.annotate(f"{ {'right': '1,024', 'left': '2,048'}[ha]} GPUs: { {'right': '1.7', 'left': '2.4'}[ha]}%", (n, v), xytext=off, textcoords='offset points', ha=ha, fontsize=12, color=C['network'], fontweight='bold')
ax.text(4200, 4.6, '×1.41 per doubling\n(grows as √N)', fontsize=12, color=C['text'], ha='center')
ax.set_xscale('log')
ax.set_xlim(128, 16384)
ax.set_ylim(0, 7.5)
ax.set_xticks([128, 1024.0, 2048.0, 16384])
ax.set_xticklabels(['128', '1,024', '2,048', '16,384'])
ax.minorticks_off()
ax.set_xlabel('GPUs (independent GPU failures only)')
ax.set_ylabel('Checkpoint + rework (%)')
ax.grid(color=C['grid'], lw=0.6)
sp.save(fig, out / 'overhead.svg')
fig, ax = plt.subplots(figsize=(5.8, 2.7))
gs = [256, 512, 768, 1024]
tp = [49.0, 95.1, 136.7, 176.9]
mid = {'gpus': 512.0, 'megascale_tokens_s': 95100.0, 'megascale_mfu_pct': 63.5, 'megatron_mfu_pct': 49.9}
ideal = [mid['megascale_tokens_s'] / 1000.0 * g_ / mid['gpus'] for g_ in gs]
ax.plot(gs, ideal, ls='--', color=C['neutral'], lw=1.8)
ax.text(1090, 12, 'dashed: linear from 512', color=C['neutral'], fontsize=11, ha='right', va='bottom')
ax.plot(gs, tp, 'o-', color=C['compute'], lw=2.5, ms=9, mec='white', mew=1.5)
for g_, t in zip(gs, tp):
    ax.annotate(f'{t:.1f}k', (g_, t), xytext=(10, -14), textcoords='offset points', fontsize=11, color=C['compute'], fontweight='bold')
ax.annotate('512 to 1,024 GPUs: 1.86× (linear: 2×)', (1024, tp[-1]), xytext=(256, 175), textcoords='data', fontsize=12, color=C['text'], fontweight='bold')
ax.set_xlim(200, 1100)
ax.set_ylim(0, 215)
ax.set_xticks(gs)
ax.set_xticklabels([f'{g_:,}' for g_ in gs])
ax.set_xlabel('GPUs (batch 768, 175B model)')
ax.set_ylabel('Tokens/s (thousands)')
ax.grid(color=C['grid'], lw=0.6)
sp.save(fig, out / 'megascale.svg')
fig, ax = plt.subplots(figsize=(5.8, 2.6))
mult, step = (0.9659363289, 2.0)
xs = [1.0, 2.0, 4.0, 8.0, 16.0]
ls = [1.0, 0.9659363289, 0.9330329915, 0.9012504626, 0.8705505633]
ax.plot(xs, ls, 'o-', color=C['memory'], lw=2.5, ms=9, mec='white', mew=1.5)
for k in range(1, 4 + 1):
    ax.annotate(f"−{ {1: '0.034', 2: '0.033', 3: '0.032', 4: '0.031'}[k]}", (xs[k], ls[k]), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=11, color=C['memory'], fontweight='bold')
ax.text(xs[-1], 0.995, 'each doubling: ×0.966 (−3.4%)', ha='right', fontsize=12, color=C['text'], fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_xticks(xs)
ax.set_xticklabels([f'{x:.0f}×' for x in xs])
ax.minorticks_off()
ax.set_ylim(0.85, 1.02)
ax.set_xlabel('Training compute (accelerator-hours), multiple of today')
ax.set_ylabel('Loss (today = 1)')
ax.grid(color=C['grid'], lw=0.6)
sp.save(fig, out / 'loss_doublings.svg')
g = Svg(900, 285)
steps = [('1 Symptom', 'run falls\nbehind\nschedule', SLATE, SLATE_L, 1), ('2 Metric', 'MFU per\nstep has\ndropped', SLATE, SLATE_L, 1), ('3 C³ term', 'per-rank\ntraces: more\nexposed\ntransfer', AMBER, AMBER_L, 2), ('4 Row', 'Distributed\nExecution', AMBER, AMBER_L, 2), ('5 Cost\nmoved', 'compression\nmoves risk\nto quality', RED, RED_L, 3), ('6 Evidence', 'convergence\nand eval\nrecord', GREEN, GREEN_L, 3)]
for i, (head, body, stroke, fill, step) in enumerate(steps):
    x = 4 + i * 150
    g.rect(x, 4, 140, 58, stroke, stroke, first=step, sw=1, rx=6)
    g.text(x + 70, 33, head, 19, '#ffffff', 'bold', first=step)
    g.rect(x, 70, 140, 150, fill, stroke, first=step, sw=2, rx=6)
    g.text(x + 70, 145, body, 20, TEXT, 'normal', first=step)
    if i < 5:
        g.line(x + 141, 33, x + 149, 33, SLATE, sw=2.5, first=steps[i + 1][4])
g.text(450, 262, 'Barrier wait alone cannot tell slow transfer from late arrival.', 21, AMBER, 'bold', first=2)
g.write('diagnose', 3)
g = Svg(520, 420)
nodes = [(260, 55, 'plan'), (430, 210, 'call a tool'), (260, 365, 'observe'), (90, 210, 'act')]
for x, y, lab in nodes:
    g.rect(x - 88, y - 38, 176, 76, ETH_L, ETH, sw=2.5, rx=10)
    g.text(x, y + 11, lab, 30, ETH, 'bold')
for (x1, y1), (x2, y2) in [((350, 70), (425, 166)), ((425, 254), (350, 350)), ((170, 350), (95, 254)), ((95, 166), (170, 70))]:
    g.line(x1, y1, x2, y2, ETH, sw=3)
g.text(260, 200, 'one task,', 28, SLATE)
g.text(260, 238, 'many steps', 28, SLATE)
g.write('agent_loop', 1)
moe_rows = [('MoE, all experts resident', 1342.0, 'memory', '1,342 GB'), ('Dense 400B, read per step', 800.0, 'neutral', '800 GB'), ('MoE, active per token', 74.0, 'compute', '74 GB')]
sp.hbars(out / 'moe_bytes.svg', moe_rows, '', log=False, size=(5.8, 1.9))
print('figures written')
