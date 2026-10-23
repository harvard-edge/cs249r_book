#!/usr/bin/env python3
"""Figures for Lecture 4 (Volume III), Context Engineering.

Every number drawn here comes from calcs.py, which reads MLSysIM (the
Scenarios.* the chapter also reads, and Literature): plotted geometry uses the
scenario values and results, labels use the formatted strings the slides print
(deckvals val/num). Diagram coordinates, axis ranges, tick positions, and
label offsets are layout. "CE" is the rendered chapter. Sources:

  - Hook, bill, layout, watermark, survival, check, invalidation bars: the
    chapter's Napkin Math 4.1 to 4.4, equation 6, and Exercise 6, as
    MLSysIM Scenarios (scenario values, illustrative).
  - Diagrams redrawn at slide size from the chapter's figures:
    fig-vol3-working-set-selection, fig-vol3-context-zones,
    fig-vol3-invalidation-graph (CE Secs. 4.2, 4.4.3, 4.7.1).
  - Fiction.LiveBench via Epoch AI: Scenarios.FictionLiveBenchThrough60K over
    vol3/04_context_engineering/data/fiction_livebench_by_length.csv.
  - slides/vol3/04_context_engineering/images/data/lost_in_the_middle_gpt35_turbo_20docs.csv and ContextUse (Liu et al. 2024; Shi et
    al. 2023; the RULER bar), RulerEffectiveLength (Hsieh et al. 2024, Table 3),
    SummarizationFaithfulness (Maynez et al. 2020), PrefixCaching (Zheng et
    al. 2024, SGLang).
  - Real artifacts, text copied exactly: vLLM v0.6.0
    vllm/core/block/prefix_caching_block.py lines 834 to 855; inotify(7),
    Linux man-pages 6.19 (2026-02-14); and output of real runs recorded in
    artifacts/ (run_toy_repair.sh, tokenize_demo.py).

Run from this folder: python3 make_figs.py
"""
from mlsysim.core.series_loader import load_series
import mlsysim
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
from matplotlib.ticker import FuncFormatter
import slideplot as sp
from mlsysim.core.series_loader import load_series
from mlsysim.physics.agents import calc_unchecked_success
from mlsysim.physics.context import calc_softmax_target_weight
C = sp.C
plt = sp.plt
out = HERE
V = {'ctx_f': '#dcfce7', 'ctx_s': '#16a34a', 'ctx_t': '#15803d', 'act_f': '#fef3c7', 'act_s': '#d97706', 'act_t': '#b45309', 'sto_f': '#f3e8ff', 'sto_s': '#7c3aed', 'sto_t': '#6d28d9', 'cmp_f': '#e0f2fe', 'cmp_s': '#0284c7', 'cmp_t': '#0369a1', 'bad_f': '#fee2e2', 'bad_s': '#dc2626', 'bad_t': '#b91c1c', 'str_f': '#f8fafc', 'str_s': '#94a3b8', 'str_t': '#0f172a', 'gray': '#64748b', 'sys_f': '#e2e8f0'}
FONT = 'Helvetica, Arial, sans-serif'
MONO = 'Courier, monospace'

class Svg:

    def __init__(self, w=900, h=500):
        self.w, self.h, self.parts = (w, h, [])

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, stroke='none', sw=2, rx=6, dash=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>')

    def text(self, x, y, s, size=20, fill='#0f172a', anchor='start', weight='normal', family=FONT, italic=False):
        st = ' font-style="italic"' if italic else ''
        s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{st} xml:space="preserve">{s}</text>')

    def line(self, x1, y1, x2, y2, stroke, sw=2.5, dash=None, arrow=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#{arrow})"' if arrow else ''
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')

    def path(self, d, stroke, sw=2.5, arrow=None, dash=None):
        m = f' marker-end="url(#{arrow})"' if arrow else ''
        da = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{m}{da}/>')

    def save(self, name):
        defs = ''.join((f'<marker id="{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for k, c in [('ar', V['bad_s']), ('ag', V['gray']), ('av', V['sto_s']), ('aa', V['act_s']), ('ac', V['ctx_s'])]))
        hatch = '<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="8" height="8" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="8" stroke="#94a3b8" stroke-width="2"/></pattern>'
        body = '\n'.join(self.parts)
        (out / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}"><defs>{defs}{hatch}</defs><rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')

def terminal(name, lines, w=900, size=19, title=None, marks=None, h=None, dim=None):
    """A real artifact drawn as a terminal card. lines: exact text.
    marks: {line_index: color} highlights a line; dim: set of dimmed lines."""
    marks, dim = (marks or {}, dim or set())
    lh = size * 1.42
    top = 50 if title else 18
    hh = h or int(top + lh * len(lines) + 22)
    s = Svg(w, hh)
    s.rect(2, 2, w - 4, hh - 4, '#0f172a', rx=10)
    if title:
        s.rect(2, 2, w - 4, 38, '#1e293b', rx=10)
        s.rect(2, 26, w - 4, 14, '#1e293b', rx=0)
        s.text(20, 28, title, size=17, fill='#cbd5e1', family=FONT, weight='bold')
    for i, ln in enumerate(lines):
        y = top + lh * (i + 1) - 6
        if i in marks:
            s.rect(10, y - size + 1, w - 20, lh, marks[i], rx=3, opacity=0.35)
        fill = '#64748b' if i in dim else '#e2e8f0'
        s.text(20, y, ln, size=size, fill=fill, family=MONO)
    s.save(name)

def hook():
    fixed_k, limit_k, window_k = (39.0, 120.0, 128.0)
    root, state = (15000.0, 24000.0)
    fixed = [('Instructions\nand tools 15K', root, '#cbd5e1'), ('Task state\n24K', state, V['ctx_s'])]
    start_obs, limit = (39000.0, 120000.0)
    reserve, obs = (8000.0, 4000.0)
    overflow = int(21.0)
    mid = (start_obs + limit) / 2000
    for k in range(1, 5):
        fig, ax = plt.subplots(figsize=(6.4, 2.0))
        x = 0
        for lab, v, col in fixed:
            ax.barh(0, v / 1000, left=x / 1000, color=col, height=0.55, edgecolor='white')
            below = lab.startswith('Task')
            ax.text((x + v / 2) / 1000, -0.34 if below else 0.36, lab.replace('\n', ' ') if below else lab, ha='center', va='top' if below else 'bottom', fontsize=10.5, color=C['text'])
            x += v
        ax.barh(0, reserve / 1000, left=limit / 1000, height=0.55, color='white', edgecolor=C['neutral'], hatch='///', lw=1)
        n = {1: 0, 2: int(10.0), 3: 0, 4: overflow}[k]
        for t in range(n):
            left = start_obs + t * obs
            over = left + obs > limit
            ax.barh(0, obs / 1000, left=left / 1000, height=0.55, color=C['danger'] if over else '#fbbf24', edgecolor='white', lw=0.8)
        ax.text((limit + reserve / 2) / 1000, 0.36, 'Output\nreserve 8K', ha='center', va='bottom', fontsize=10.5, color=C['neutral'])
        ax.plot([limit / 1000] * 2, [-0.36, 0.36], color=C['danger'], lw=1.8, ls='--')
        ax.annotate('', xy=(limit / 1000, -0.66), xytext=(start_obs / 1000, -0.66), arrowprops=dict(arrowstyle='<->', color='#b45309', lw=1.6, alpha=1.0 if k == 1 else 0.0))
        texts = [(1, mid, -0.78, '81,000 tokens left for tool results; +4,000 per turn', '#b45309'), (2, (start_obs + 40000.0 / 2) / 1000 + 6, -0.72, 'Turn 10: 40,000 tokens of results', '#b45309'), (3, mid, 0.0, 'On which turn?', C['danger']), (4, mid, -0.72, 'Turn 21 needs 84,000 > 81,000 tokens', C['danger']), (4, mid, -1.08, 'Lasting 30 turns allows at most 2,700 per result', C['danger'])]
        for kk, xx, yy, tt, col in texts:
            ax.text(xx, yy, tt, ha='center', va='top' if yy < 0 else 'center', fontsize=14 if kk == 3 else 11, color=col, fontweight='bold', alpha=1.0 if kk == k else 0.0)
        ax.set_xlim(0, window_k)
        ax.set_ylim(-1.5, 1.05)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xticks([0, fixed_k, limit_k, window_k])
        ax.set_xlabel('Thousand tokens in one 128K-token request (illustrative)')
        sp.save(fig, out / f'hook_step{k}.svg')
    fig, ax = plt.subplots(figsize=(2.4, 0.9))
    x = 0
    for lab, v, col in fixed:
        ax.barh(0, v / 1000, left=x / 1000, color=col, height=0.55, edgecolor='white')
        x += v
    ax.barh(0, reserve / 1000, left=limit / 1000, height=0.55, color='white', edgecolor=C['neutral'], hatch='///', lw=1)
    for t in range(overflow):
        left = start_obs + t * obs
        over = left + obs > limit
        ax.barh(0, obs / 1000, left=left / 1000, height=0.55, color=C['danger'] if over else '#fbbf24', edgecolor='white', lw=0.6)
    ax.plot([limit / 1000] * 2, [-0.36, 0.36], color=C['danger'], lw=1.8, ls='--')
    ax.text(0, 0.36, 'Instructions\nand tools 15K', ha='left', va='bottom', fontsize=11, color=C['text'])
    ax.text(window_k, 0.36, 'Output\nreserve 8K', ha='right', va='bottom', fontsize=11, color=C['neutral'])
    ax.text(15.0, -0.34, 'Task state 24K', ha='left', va='top', fontsize=11, color=C['text'])
    ax.set_xlim(0, window_k)
    ax.set_ylim(-0.75, 0.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, fixed_k, limit_k])
    ax.set_xlabel('Thousand tokens in one\n128K-token request (illustrative)')
    sp.save(fig, out / 'hook_col.svg')

def recall_fig():
    fig, ax = plt.subplots(figsize=(4.2, 1.5))
    for y, lab, key, ck in [(1, 'Resend', 'rs_resent', 'danger'), (0, 'Reuse prefix', 'rs_fresh', 'memory')]:
        v = {'Resend': 892500.0, 'Reuse prefix': 51500.0}[lab]
        ax.barh(y, v / 1000, color=C[ck], height=0.6)
        ax.text(-15, y, lab, ha='right', va='center', fontsize=13, fontweight='bold')
        ax.text(v / 1000 + 15, y, {'Resend': '892,500', 'Reuse prefix': '51,500'}[lab], va='center', fontsize=13, color=C[ck], fontweight='bold')
    ax.set_xlim(0, 1150)
    ax.set_yticks([])
    ax.set_xticks([])
    for sname in ('left', 'bottom'):
        ax.spines[sname].set_visible(False)
    ax.set_title('Input tokens, 30 turns (Lecture 2)', fontsize=12, loc='left')
    sp.save(fig, out / 'recall_resend.svg')

def partition():
    """Canvas 900 x 340: shown full width (about 15 cm), so 17 units is about 8 pt."""

    def panel(s, x0, title, slices, state_h, state_bad, label, lab_col):
        s.text(x0, 26, title, size=21, weight='bold')
        base = 278
        for i in range(3):
            bx = x0 + 6 + i * 100
            y = base
            for h, f in [(18, '#f1f5f9'), (22, V['sys_f'])]:
                s.rect(bx, y - h, 80, h, f, V['str_s'], sw=1.5, rx=0)
                y -= h
            sh = state_h[i]
            bad = state_bad and i == 2
            s.rect(bx, y - sh, 80, sh, V['bad_f'] if bad else V['ctx_f'], V['bad_s'] if bad else V['ctx_s'], sw=2, rx=0)
            y -= sh
            for _ in range(slices[i]):
                s.rect(bx, y - 16, 80, 16, V['act_f'], V['act_s'], sw=1.3, rx=0)
                y -= 16
            s.rect(bx, 48, 80, 30, 'url(#hatch)', V['str_s'], sw=1.5, rx=0)
            s.text(bx + 40, 302, ['turn t', 'turn t+1', 'turn t+2'][i], size=18, anchor='middle', fill='#334155')
        s.line(x0, 78, x0 + 292, 78, V['bad_s'], sw=2, dash='7,5')
        s.text(x0 + 146, 330, label, size=20, anchor='middle', fill=lab_col, weight='bold')
    for k in (1, 2):
        s = Svg(900, 340)
        for y, lab, col in [(68, 'reserve', V['gray']), (174, 'results', V['act_t']), (214, 'state', V['ctx_t']), (252, 'tools', '#475569'), (276, 'system', '#475569')]:
            s.text(96, y, lab, size=18, anchor='end', fill=col, weight='bold')
        panel(s, 102, 'A. Append every result', [2, 4, 6], [100, 68, 36], True, 'task state squeezed', V['bad_t'])
        if k == 2:
            panel(s, 470, 'B. Select for the next step', [2, 2, 2], [100, 100, 100], False, 'task state keeps its share', V['ctx_t'])
            s.rect(800, 110, 92, 100, V['sto_f'], V['sto_s'], sw=2.5, rx=4)
            for j in range(4):
                s.rect(821, 120 + j * 21, 50, 16, V['act_f'], V['act_s'], sw=1.2, rx=0)
            s.text(846, 234, 'records', size=18, anchor='middle', fill=V['sto_t'], weight='bold')
            s.text(846, 256, 'behind', size=18, anchor='middle', fill=V['sto_t'])
            s.text(846, 278, 'a handle', size=18, anchor='middle', fill=V['sto_t'])
            s.line(774, 170, 796, 170, V['sto_s'], sw=2.5, arrow='av')
        s.save(f'partition_step{k}.svg')

def bill():
    t = np.arange(1, 30 + 1)
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    ax.plot(t, np.array([43000, 90000, 141000, 196000, 255000, 318000, 385000, 456000, 531000, 610000, 693000, 780000, 871000, 966000, 1065000, 1168000, 1275000, 1386000, 1501000, 1620000, 1743000, 1870000, 2001000, 2136000, 2275000, 2418000, 2565000, 2716000, 2871000, 3030000]) / 1000000.0, color=C['danger'], lw=3)
    ax.plot(t, np.array([47000, 94000, 141000, 188000, 235000, 282000, 329000, 376000, 423000, 470000, 517000, 564000, 611000, 658000, 705000, 752000, 799000, 846000, 893000, 940000, 987000, 1034000, 1081000, 1128000, 1175000, 1222000, 1269000, 1316000, 1363000, 1410000]) / 1000000.0, color=C['memory'], lw=3)
    ax.text(15, 1.3, 'Append every result', color=C['danger'], fontsize=12, fontweight='bold', ha='right')
    ax.text(21, 0.42, 'Select 47,000 per call', color=C['memory'], fontsize=12, fontweight='bold')
    ax.annotate('', xy=(30.6, 3030000 / 1000000.0), xytext=(30.6, 1410000 / 1000000.0), arrowprops=dict(arrowstyle='<->', color=C['text'], lw=1.5))
    ax.text(18.5, 2.6, '3.03M against 1.41M: 2.1×', ha='center', va='center', fontsize=12, fontweight='bold', color=C['text'])
    ax.set_xlim(0, 33)
    ax.set_ylim(0, 3.3)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Cumulative input,\nmillion tokens')
    sp.save(fig, out / 'bill.svg')

def fiction():
    lengths = ['0', '400', '1k', '2k', '4k', '8k', '16k', '32k', '60k']
    cols = ['score_' + n for n in lengths]
    rows = {e['model']: e for e in load_series('vol3/04_context_engineering/data/fiction_livebench_by_length.csv').entries}
    best, worst = (rows['gpt-5-2025-08-07_medium'], rows['nvidia-nemotron-nano-9b-v2'])
    median = [100.0, 82.2, 77.8, 72.2, 66.7, 58.3, 61.1, 55.6, 55.6]
    x = np.arange(len(lengths))
    labels = [n.upper() for n in lengths]
    for k in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.2))
        if k == 1:
            ax.plot([0], [median[0]], 'o', color=C['text'], ms=8)
            ax.text(0.25, median[0] - 2, 'Shortest version: median 100%', fontsize=11, va='top', color=C['text'], fontweight='bold')
            ax.text(8, 50, f'{labels[-1]}: ???', fontsize=15, ha='center', color=C['danger'], fontweight='bold')
        if k >= 2:
            ax.fill_between(x, [87.5, 66.7, 62.5, 52.8, 50.0, 50.0, 47.2, 44.4, 41.7], [100.0, 97.2, 94.4, 88.9, 91.7, 83.3, 75.0, 75.0, 69.4], color=C['neutral_l'], alpha=0.8, lw=0)
            ax.plot(x, median, '-o', color=C['text'], lw=2.5, ms=5)
            ax.text(8.15, median[-1], 'median\n56%', fontsize=11, va='center', color=C['text'], fontweight='bold')
            ax.text(2.2, 40, 'middle half of 53 models', fontsize=11, color=C['neutral'], alpha=1.0 if k == 2 else 0.0)
        if k == 3:
            for e, col, lab, key in [(best, C['data'], 'highest at 60K', 'fl_hi'), (worst, C['danger'], 'lowest at 60K', 'fl_lo')]:
                y = [100 * e[c] for c in cols]
                ax.plot(x, y, '-', color=col, lw=1.8)
                ax.text(8.15, y[-1], f"{lab}\n{ {'highest at 60K': '100', 'lowest at 60K': '22'}[lab]}%", fontsize=10.5, va='center', color=col, fontweight='bold')
            ax.text(0.0, 4, 'Only 3 of 53 stay at\nor above 90% at 60K', fontsize=11, color=C['danger'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.3, 9.6)
        ax.set_ylim(0, 106)
        ax.set_xlabel('Story length placed in context (tokens)')
        ax.set_ylabel('Accuracy (%)')
        sp.save(fig, out / f'fiction_step{k}.svg')

def ruler():
    rows = ({'model': 'GPT-4', 'claimed_k': 128.0, 'effective_k': 64.0, 'effective_label': '64K'}, {'model': 'Llama3.1 (70B)', 'claimed_k': 128.0, 'effective_k': 64.0, 'effective_label': '64K'}, {'model': 'Qwen2 (72B)', 'claimed_k': 128.0, 'effective_k': 32.0, 'effective_label': '32K'}, {'model': 'Llama3.1 (8B)', 'claimed_k': 128.0, 'effective_k': 32.0, 'effective_label': '32K'}, {'model': 'Mixtral-8x22B (39B/141B)', 'claimed_k': 64.0, 'effective_k': 32.0, 'effective_label': '32K'}, {'model': 'Mistral-v0.2 (7B)', 'claimed_k': 32.0, 'effective_k': 16.0, 'effective_label': '16K'}, {'model': 'DBRX (36B/132B)', 'claimed_k': 32.0, 'effective_k': 8.0, 'effective_label': '8K'}, {'model': 'Together (7B)', 'claimed_k': 32.0, 'effective_k': 4.0, 'effective_label': '4K'}, {'model': 'LongChat (7B)', 'claimed_k': 32.0, 'effective_k': None, 'effective_label': '<4K'})
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    for i, e in enumerate(rows):
        y = len(rows) - 1 - i
        claimed, eff = (e['claimed_k'], e['effective_k'])
        drawn = eff if eff is not None else 2
        ax.plot([drawn, claimed], [y, y], color=C['neutral_l'], lw=4, zorder=1)
        ax.plot([claimed], [y], 'o', color=C['neutral'], ms=8, zorder=2)
        ax.plot([drawn], [y], 'o', color=C['danger'], ms=8, zorder=3)
        ax.text(1.3, y, e['model'], ha='right', va='center', fontsize=10.5, color=C['text'])
        ax.text(190, y, f"{e['effective_label']} of {claimed:.0f}K", ha='left', va='center', fontsize=10.5, color=C['danger'], fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xlim(1.4, 180)
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(['4K', '8K', '16K', '32K', '64K', '128K'])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.text(190, len(rows) - 0.2, 'effective', ha='left', fontsize=10.5, color=C['danger'], fontweight='bold')
    ax.text(128, len(rows) - 0.2, 'claimed', ha='center', fontsize=10.5, color=C['neutral'], fontweight='bold')
    ax.set_ylim(-0.6, len(rows) + 0.4)
    ax.set_xlabel('Context length (tokens, log scale)')
    sp.save(fig, out / 'ruler.svg')

def liu():
    pos = [1, 5, 10, 15, 20]
    acc = [75.8, 57.2, 53.8, 55.4, 63.2]
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.plot(pos, acc, '-o', color=C['memory'], lw=3, ms=8)
    for i, (xx, yy) in enumerate(zip(pos, acc)):
        below = i in (2, 3)
        dx = 1.4 if i == 1 else 0
        ax.text(xx + dx, yy - 4.5 if below else yy + 2.2, f'{yy:.1f}', ha='center', fontsize=11, color=C['memory'], fontweight='bold')
    ax.axhline(56.1, color=C['danger'], ls='--', lw=1.8)
    ax.text(0.4, 46.5, 'Dashed: no documents at all, 56.1%', ha='left', fontsize=11, color=C['danger'], fontweight='bold')
    ax.set_xticks(pos)
    ax.set_xlim(0, max(pos) + 1)
    ax.set_ylim(44, 82)
    ax.set_xlabel('Position of the answer document among 20 (about 4K tokens)')
    ax.set_ylabel('Accuracy (%)')
    sp.save(fig, out / 'liu.svg')

def dilution():
    n = np.logspace(1, 5.3, 200)
    w = [100 * calc_softmax_target_weight(6.0, [(int(c) - 1, -2.0)]) for c in n]
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.plot(n, w, color=C['text'], lw=3)
    for keys, pct, kx, v, col, short in [('dil_short', 'dil_short_pct', 100, 100 * 0.9678567044, C['data'], True), ('dil_long', 'dil_long_pct', 100000, 100 * 0.02894697226, C['danger'], False)]:
        ax.plot([kx], [v], 'o', color=col, ms=10, zorder=5)
        label = f"{ {96.78567044: '100', 2.894697226: '100,000'}[v]} keys: { {96.78567044: '96.8', 2.894697226: '2.9'}[v]}%"
        if short:
            ax.text(kx * 1.4, v + 4, label, fontsize=11, color=col, fontweight='bold', ha='left')
        else:
            ax.annotate(label, (kx, v), xytext=(14, 12), fontsize=11, color=col, fontweight='bold', ha='left', arrowprops=dict(arrowstyle='->', color=col, lw=1.3))
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:,.0f}K' if v >= 1000 else f'{v:,.0f}'))
    ax.set_xlim(10, 200000.0)
    ax.set_ylim(0, 108)
    ax.set_xlabel('Keys in the request (log scale)')
    ax.set_ylabel('Weight on\nthe target (%)')
    sp.save(fig, out / 'dilution.svg')

def zones():
    """Canvas 900 x 360."""
    xr, xt, xl, xe, xe2 = (210, 370, 630, 760, 880)
    rows = [('call n', 'cold start'), ('call n+1', 'new observation'), ('call n+2', 'trunk refreshed')]
    for k in (1, 2, 3):
        s = Svg(900, 360)
        for lab, sub, xc in [('Root', 'instructions, tools', (xr + xt) / 2), ('Trunk', 'constraints, progress, source', (xt + xl) / 2), ('Leaf', 'latest result', (xl + xe) / 2)]:
            s.text(xc, 24, lab, size=21, anchor='middle', weight='bold')
            s.text(xc, 46, sub, size=17, anchor='middle', fill='#475569')
        s.text(820, 24, 'Output', size=20, anchor='middle', weight='bold', fill=V['gray'])
        s.line(xr, 60, xe - 6, 60, V['gray'], sw=2, arrow='ag')
        s.text(xr, 80, 'changes rarely', size=17, fill=V['gray'])
        s.text(xe, 80, 'changes every call', size=17, fill=V['gray'], anchor='end')
        for i in range(k):
            y = 104 + i * 76
            s.text(196, y + 20, rows[i][0], size=19, anchor='end', weight='bold')
            s.text(196, y + 40, rows[i][1], size=17, anchor='end', fill='#475569')
            reuse_to = {0: xr, 1: xl, 2: xt}[i]
            trunk_lab = 'trunk @ r18' if i == 2 else 'trunk @ r17'
            for a, bb, lab in [(xr, xt, 'root'), (xt, xl, trunk_lab), (xl, xe, f"leaf {['n', 'n+1', 'n+2'][i]}")]:
                reused = bb <= reuse_to
                s.rect(a, y, bb - a, 44, V['ctx_f'] if reused else V['cmp_f'], V['ctx_s'] if reused else V['cmp_s'], sw=2, rx=0)
                s.text((a + bb) / 2, y + 29, lab, size=19, anchor='middle')
            s.rect(xe, y, xe2 - xe, 44, 'url(#hatch)', V['str_s'], sw=1.5, rx=0)
            if i > 0:
                s.line(reuse_to, y - 6, reuse_to, y + 52, V['bad_s'], sw=3, dash='6,4')
                s.text(reuse_to + (10 if i == 2 else -10), y + 64 if i == 2 else y + 64, 'first changed token', size=17, fill=V['bad_t'], anchor='start' if i == 2 else 'end', weight='bold')
        ly = 352
        s.rect(210, ly - 16, 22, 18, V['ctx_f'], V['ctx_s'], sw=2, rx=0)
        s.text(240, ly, 'prefix reused', size=17)
        s.rect(400, ly - 16, 22, 18, V['cmp_f'], V['cmp_s'], sw=2, rx=0)
        s.text(430, ly, 'processed again', size=17)
        s.rect(610, ly - 16, 22, 18, 'url(#hatch)', V['str_s'], sw=1.5, rx=0)
        s.text(640, ly, 'output reserve', size=17)
        s.save(f'zones_step{k}.svg')

def layout_bars():
    sp.hbars(out / 'layout_bars.svg', [('Volatile field first', 4.23, 'danger', '$4.23'), ('Root, trunk, leaf', 1.3059, 'data', '$1.31')], '', log=False, size=(3.4, 1.3), xlabel='Input bill, 30 calls, $')

def sglang_bars():
    sp.hbars(out / 'sglang_hits.svg', [('LLaVA-Next-34B', 52.4, 'data', '52.4%'), ('Vicuna-33B', 74.1, 'data', '74.1%')], '', log=False, size=(3.4, 1.2), xlabel='Cache hit rate, one month (%)')

def scope():
    """Canvas 900 x 330."""
    s = Svg(900, 330)
    s.rect(8, 8, 520, 314, V['str_f'], V['str_s'], sw=2, rx=10)
    s.text(24, 38, 'Observation, as staged by the runtime', size=20, weight='bold')
    fields = [('test', 'test_waiter_notified_on_return'), ('source', 'src/pool.py @ r17'), ('tests', 'tests/test_pool.py @ r17'), ('schedule', 'waiter before return'), ('result', 'FAILED, run-0412')]
    for i, (k_, v_) in enumerate(fields):
        y = 72 + i * 31
        s.text(26, y, k_, size=18, fill=V['ctx_t'], weight='bold')
        s.text(126, y, v_, size=18, family=MONO)
    s.rect(22, 236, 492, 74, V['act_f'], V['act_s'], sw=2, rx=6, dash='7,5')
    s.text(36, 264, 'payload: raw tool output, an untrusted span', size=18, fill=V['act_t'], weight='bold')
    s.text(36, 292, 'framed as data; its text can still steer the model', size=18, fill=V['act_t'])
    s.rect(570, 8, 322, 140, V['ctx_f'], V['ctx_s'], sw=2.5, rx=8)
    s.text(731, 44, 'Supports claims about', size=19, anchor='middle', weight='bold', fill=V['ctx_t'])
    s.text(731, 76, 'r17, this test, this schedule', size=18, anchor='middle')
    s.text(731, 116, 'not about r18 or other tests', size=18, anchor='middle', fill=V['bad_t'], weight='bold')
    s.rect(570, 182, 322, 140, V['sto_f'], V['sto_s'], sw=2.5, rx=8)
    s.text(731, 216, 'Host gate, outside the prompt', size=19, anchor='middle', weight='bold', fill=V['sto_t'])
    s.text(731, 246, 'sandbox, read-only mounts,', size=18, anchor='middle')
    s.text(731, 272, 'no production deploy:', size=18, anchor='middle')
    s.text(731, 302, 'enforces what may run', size=18, anchor='middle', fill=V['sto_t'], weight='bold')
    s.line(530, 110, 564, 90, V['ctx_s'], sw=2.5, arrow='ac')
    s.line(530, 272, 564, 260, V['sto_s'], sw=2.5, arrow='av')
    s.save('scope.svg')

def maynez():
    rows = [('PTGen', 75.3, 'danger', '75.3%'), ('TConvS2S', 78.5, 'danger', '78.5%'), ('TranS2S', 79.3, 'danger', '79.3%'), ('BertS2S', 73.1, 'danger', '73.1%'), ('Gold reference', 76.9, 'neutral', '76.9%')]
    sp.hbars(out / 'maynez.svg', rows, '', log=False, size=(5.4, 2.2), xlabel='Summaries with unsupported content (%, 500 per system)')

def sawtooth():
    target, alt, trigger = (47.0, 81.0, 92.0)
    turns = 30
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.2))
        xs1, ys1 = ([0], [target])
        for i, v in enumerate([51000, 55000, 59000, 63000, 67000, 71000, 75000, 79000, 83000, 87000, 91000, 95000, 51000, 55000, 59000, 63000, 67000, 71000, 75000, 79000, 83000, 87000, 91000, 95000, 51000, 55000, 59000, 63000, 67000, 71000], start=1):
            xs1.append(i)
            ys1.append(v / 1000)
            if v >= 92000:
                ax.plot([i], [v / 1000], 'o', color=C['danger'], ms=8, zorder=5)
                xs1.append(i)
                ys1.append(target)
        ax.plot(xs1, ys1, color=C['memory'], lw=2.5)
        ax.axhline(trigger, color=C['neutral'], ls='--', lw=1.5)
        ax.axhline(target, color=C['neutral'], ls='--', lw=1.5)
        ax.axhline(120.0, color=C['danger'], lw=2)
        ax.text(turns + 0.4, 120.0, 'input limit 120K', va='center', fontsize=10.5, color=C['danger'])
        ax.text(turns + 0.4, trigger, 'trigger 92K', va='center', fontsize=10.5, color=C['neutral'])
        ax.text(turns + 0.4, target, 'target 47K', va='center', fontsize=10.5, color=C['neutral'])
        ax.text(7, 106, 'every 12 turns: 2 passes in 30', fontsize=11.5, color=C['memory'], fontweight='bold', ha='center')
        if k == 2:
            xs2, yy2 = ([0], [alt])
            for i, v in enumerate([85000, 89000, 93000, 85000, 89000, 93000, 85000, 89000, 93000, 85000, 89000, 93000, 85000, 89000, 93000, 85000, 89000, 93000, 85000, 89000, 93000, 85000, 89000, 93000, 85000, 89000, 93000, 85000, 89000, 93000], start=1):
                xs2.append(i)
                yy2.append(v / 1000)
                if v >= 92000:
                    xs2.append(i)
                    yy2.append(alt)
            ax.plot(xs2, yy2, color=C['compute'], lw=2)
            ax.axhline(alt, color=C['compute'], ls=':', lw=1.5)
            ax.text(turns + 0.4, alt - 3, 'target 81K', va='center', fontsize=10.5, color=C['compute'])
            ax.text(22, 106, 'every 3 turns: 10 passes', fontsize=11.5, color=C['compute'], fontweight='bold', ha='center')
        ax.set_xlim(0, turns)
        ax.set_ylim(30, 128)
        ax.set_xlabel('Turn')
        ax.set_ylabel('Request length,\nthousand tokens')
        sp.save(fig, out / f'sawtooth_step{k}.svg')

def survival():
    n = np.arange(0, 40 + 1)
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    ax.plot(n, [100 * calc_unchecked_success(0.95, r) for r in n], color=C['danger'], lw=3)
    for rk, pk, r, col, short in [('ret_short', 'ret_short_pct', 10, C['data'], True), ('ret_long', 'ret_long_pct', 40, C['danger'], False)]:
        v = 100 * {10: 0.5987369392, 40: 0.1285121566}[r]
        ax.plot([r], [v], 'o', color=col, ms=9, zorder=5)
        ax.text(r + 1.5 if short else r - 1, v + 6 if short else 2, f"{ {10: '10', 40: '40'}[r]} rounds: { {10: '59.9', 40: '12.9'}[r]}%", fontsize=11, color=col, fontweight='bold', ha='left' if short else 'right', va='bottom')
    ax.axhline(50.0, color=C['neutral'], ls=':', lw=1.3)
    ax.text(16, 50.0 + 4, '< 50% at round 14', fontsize=10.5, color=C['neutral'])
    ax.set_xlim(0, 40 + 1)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Compaction rounds')
    ax.set_ylabel('Constraint\nsurvives (%)')
    sp.save(fig, out / 'survival.svg')

def check_pays():
    sp.hbars(out / 'check_pays.svg', [('No check', 9.979213442, 'danger', '$9.98'), ('Host check', 3.992652475, 'data', '$3.99')], '', log=False, size=(3.2, 1.3), xlabel='Cost per accepted task, $')

def inval_graph():
    """Canvas 900 x 350."""
    for k in (1, 2, 3):
        s = Svg(900, 350)
        for x, lab in [(115, 'Workspace'), (420, 'Staged observations'), (765, 'Derived entries')]:
            s.text(x, 22, lab, size=20, anchor='middle', weight='bold')
        write = k >= 2
        prop = k >= 3
        if prop:
            s.rect(4, 34, 892, 186, '#fff1f2', V['bad_s'], sw=2, rx=6, dash='8,6')
            s.text(888, 212, 'reached by the write', size=17, anchor='end', fill=V['bad_t'], weight='bold')
        arts = [(42, 'src/pool.py', 'r17 → r18' if write else 'r17', write), (128, 'tests/test_pool.py', 'r17', False), (238, 'docs/pool.md', 'r17', False)]
        for y, name, rev, hot in arts:
            s.rect(10, y, 214, 62, 'white', V['bad_s'] if hot else V['str_s'], sw=2.5, rx=4)
            s.text(20, y + 26, name, size=17, family=MONO, weight='bold')
            s.text(20, y + 51, rev, size=17, fill=V['bad_t'] if hot else '#475569', weight='bold' if hot else 'normal')
        obs = [(42, 'o1 source excerpt', 'pool.py @ r17'), (128, 'o2 controlled test run', 'test run @ r17'), (238, 'o3 docs excerpt', 'pool.md @ r17')]
        for i, (y, name, rev) in enumerate(obs):
            hist = prop and i < 2
            s.rect(280, y, 280, 62, 'white' if hist else V['act_f'], V['act_s'], sw=2.5, rx=4)
            s.text(292, y + 26, name, size=18, weight='bold')
            s.text(292, y + 51, rev, size=17, fill='#475569')
            if hist:
                s.text(550, y + 51, 'historical', size=17, anchor='end', fill=V['act_t'], weight='bold')
            if i == 2 and prop:
                s.text(550, y + 51, 'unchanged', size=17, anchor='end', fill=V['ctx_t'], weight='bold')
        s.rect(640, 74, 248, 88, 'white' if prop else V['ctx_f'], V['bad_s'] if prop else V['ctx_s'], sw=2.5, rx=4)
        s.text(654, 102, 'Missed-notification', size=18, weight='bold')
        s.text(654, 126, 'hypothesis', size=18, weight='bold')
        if prop:
            s.text(654, 150, 'reassess', size=17, fill=V['bad_t'], weight='bold')
        s.rect(640, 238, 248, 62, V['ctx_f'], V['ctx_s'], sw=2.5, rx=4)
        s.text(654, 264, 'API usage note', size=18, weight='bold')
        if prop:
            s.text(654, 289, 'unchanged', size=17, fill=V['ctx_t'], weight='bold')
        red = V['bad_s'] if prop else V['gray']
        ar = 'ar' if prop else 'ag'
        s.path('M224,73 L272,73', red, arrow=ar)
        s.path('M224,73 L248,73 L248,150 L272,150', red, arrow=ar)
        s.path('M224,172 L272,172', V['gray'], arrow='ag')
        s.path('M560,73 L600,73 L600,105 L632,105', red, arrow=ar)
        s.path('M560,159 L600,159 L600,131 L632,131', red, arrow=ar)
        s.path('M224,269 L272,269', V['gray'], arrow='ag')
        s.path('M560,269 L632,269', V['gray'], arrow='ag')
        msg = {1: ('Each entry records what it read: artifact, kind, revision, range, content hash', '#334155'), 2: ("The agent's own patch writes src/pool.py: r17 becomes r18", V['bad_t']), 3: ('Observations become history; only conclusions built on them are reassessed', V['bad_t'])}[k]
        s.text(450, 338, msg[0], size=18, anchor='middle', fill=msg[1], weight='bold')
        s.save(f'inval_step{k}.svg')

def inval_bars():
    sp.hbars(out / 'inval_bars.svg', [('Replace in place', 31174.0, 'compute', '31,174'), ('Append a notice', 128.0, 'data', '128')], '', log=False, size=(3.1, 1.3), xlabel='Tokens prefilled again')

def shi_bars():
    rows = [('Base', 95.0, 'data', '95.0%'), ('+ 1 irrelevant', 72.4, 'danger', '72.4%'), ("+ 'ignore it'", 77.8, 'compute', '77.8%')]
    sp.hbars(out / 'shi.svg', rows, '', log=False, size=(3.3, 1.5), xlabel='Math accuracy (%)')

def artifacts():
    terminal('art_vllm.svg', ['@staticmethod', 'def hash_block_tokens(is_first_block: bool, prev_block_hash: Optional[int],', '                      cur_block_token_ids: List[int]) -> int:', '    """Computes a hash value corresponding to the contents of a block and', '    the contents of the preceding block(s). ...', '    assert (prev_block_hash is None) == is_first_block', '    return hash((is_first_block, prev_block_hash, *cur_block_token_ids))'], size=17, title='vllm/core/block/prefix_caching_block.py, v0.6.0, lines 834 to 855 (docstring cut)', marks={6: '#16a34a'}, dim={3, 4})
    tok = (HERE / 'artifacts' / 'tokenize_output.txt').read_text().splitlines()
    terminal('art_tokens.svg', tok, size=18, title='python3 tokenize_demo.py   (GPT-2 byte-pair tokenizer, real run)', marks={3: '#dc2626', 7: '#16a34a'})
    full = (HERE / 'artifacts' / 'pytest_r17.txt').read_text().splitlines()
    start = next((i for i, ln in enumerate(full) if 't.start()' in ln))
    stop = next((i for i, ln in enumerate(full) if ln.startswith('E       assert False')))
    stop = next((i for i, ln in enumerate(full) if ln.startswith('E       AssertionError')))
    py = full[start + 1:stop + 1] + [ln for ln in full if ln.startswith('FAILED')]
    terminal('art_pytest.svg', py, size=17, title='pytest -q tests/test_pool.py   at r17 (toy pool, real run, excerpt)', marks={0: '#d97706', 1: '#d97706', 2: '#d97706', len(py) - 1: '#dc2626'})
    ga = (HERE / 'artifacts' / 'git_apply_r18.txt').read_text().splitlines()
    terminal('art_gitapply.svg', ga, size=18, title='git apply --check -v stale_from_r17.patch   at r18 (real run)', marks={len(ga) - 1: '#dc2626', len(ga) - 2: '#dc2626'})
    terminal('art_inotify.svg', ['Inotify reports only events that a user-space program triggers', 'through the filesystem API.  As a result, it does not catch remote', 'events that occur on network filesystems.', '...', 'robust applications should allow for the fact that bugs in the', 'monitoring logic or races of the kind described below may leave', 'the cache inconsistent with the filesystem state.  It is probably', 'wise to do some consistency checking, and rebuild the cache when', 'inconsistencies are detected.'], size=18, title='man 7 inotify   (Linux man-pages 6.19, 2026-02-14)', marks={6: '#d97706'})
    terminal('art_audit.svg', ['def audit_record(record, inventory, events, workspace):', '    found = []', '    for f in inventory.required_fields():             # 1. contract', '        if record.value_of(f.name) != f.value: found.append(f)', '    for c in (*record.milestones, *record.current_claims):', '        e = events.get(c.event_id)                    # 2. event log', '        if e is None or c.facts() != e.facts(): found.append(c)', '    for c in record.current_claims:                   # 3. version', '        if c.version != workspace.version_of(c.artifact): found.append(c)', '    return found'], size=18, title='Listing 2 (CE §4.6.2), condensed: every compared value comes from the host', marks={2: '#16a34a', 5: '#16a34a', 7: '#16a34a'})
if __name__ == '__main__':
    hook()
    recall_fig()
    partition()
    bill()
    fiction()
    ruler()
    liu()
    dilution()
    zones()
    layout_bars()
    sglang_bars()
    scope()
    maynez()
    sawtooth()
    survival()
    check_pays()
    inval_graph()
    inval_bars()
    shi_bars()
    artifacts()
    print('figures written to', out)
