#!/usr/bin/env python3
"""Figures for Lecture 5 (KV Cache Management), Volume III.

Every number drawn here comes from calcs.py, which reads MLSysIM (the
scenarios the chapter, KV Cache Management, also reads); labels print the
values calcs.py records (deckvals.val). Outside data are MLSysIM Literature
entries read from the papers:

  - Kwon et al., "PagedAttention," SOSP 2023, Figure 2 (KV memory breakdown).
  - Zheng et al., "SGLang," 2024, Sec. 6 (production cache hit rates).
  - Abhyankar et al., "InferCept," ICML 2024, Table 1 (interception times).
  - Xiao et al., "StreamingLLM," ICLR 2024, Tables 1 and 2 (perplexity).

Schematics use the Volume III agentic-stack roles (memory and context green,
compute sky blue, tools amber, storage violet, faults red); data plots keep
the base contract of slideplot (memory blue, compute orange). Figures are
drawn at the size of their slot: full width about 5.8 x 2.4 in, column
about 4.2 x 2.8 in, with 11 to 14 pt type.

Run from this folder: python3 make_figs.py   (the Makefile converts SVG to PDF)
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
C = sp.C
plt = sp.plt
OUT = HERE
R = {'mem': '#16a34a', 'mem_l': '#dcfce7', 'mem_t': '#15803d', 'cmp': '#0284c7', 'cmp_l': '#e0f2fe', 'cmp_t': '#0369a1', 'tool': '#d97706', 'tool_l': '#fef3c7', 'tool_t': '#b45309', 'sto': '#7c3aed', 'sto_l': '#f3e8ff', 'sto_t': '#6d28d9', 'bad': '#dc2626', 'bad_l': '#fee2e2', 'bad_t': '#b91c1c', 'st': '#94a3b8', 'st_l': '#f8fafc', 'ink': '#0f172a', 'dim': '#cbd5e1'}

def save(fig, name):
    sp.save(fig, OUT / f'{name}.svg')

def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.axis('off')
    return (fig, ax)

def box(ax, x, y, w, h, text='', edge=R['st'], fill=R['st_l'], fs=12, bold=False, tc=None, lw=1.6, ls='-', alpha=1.0, round_=True):
    if round_:
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', ec=edge, fc=fill, lw=lw, ls=ls, alpha=alpha)
    else:
        p = Rectangle((x, y), w, h, ec=edge, fc=fill, lw=lw, ls=ls, alpha=alpha)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=tc or R['ink'], alpha=alpha)

def arrow(ax, x0, y0, x1, y1, col=R['ink'], lw=1.6, alpha=1.0, style='-|>', ms=14, cs='arc3'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=col, lw=lw, alpha=alpha, connectionstyle=cs))

def hb(name, rows, xlabel, size, xlim, vlines=()):
    """Log-scale horizontal bars with labels left of the axis and values on the bars."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, (label, v, ck, txt) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, v, color=C[ck], height=0.62)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
        ax.text(v * 1.15, y, txt, va='center', fontsize=12, fontweight='bold', color=C[ck])
    for x, label, ck in vlines:
        ax.axvline(x, color=C[ck], lw=1.5, ls='--', zorder=1)
        ax.text(x * 1.1, n - 0.45, label, color=C[ck], fontsize=11, fontweight='bold', va='bottom')
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4 + (0.5 if vlines else 0))
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    save(fig, name)

def hook():
    """Predict, then reveal: memory-bound turn rate, kept vs released."""
    rows = [('Keep state\nthrough the test', 0.82, C['neutral'], '0.82'), ('Release state\nfor the test', 8.2, C['memory'], '8.2')]
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(3.2, 1.55))
        for i, (lab, v, col, printed) in enumerate(rows):
            y = 1 - i
            show = i == 0 or k == 2
            ax.barh(y, v if show else 0, color=col, height=0.55)
            ax.text(-0.15, y, lab, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
            txt = f'{printed} turns/s'
            ax.text(v + 0.15, y, txt if show else '', va='center', fontsize=13, fontweight='bold', color=col)
            if i == 1 and k == 1:
                ax.text(0.15, y, '???', va='center', fontsize=15, fontweight='bold', color=C['danger'])
        ax.set_xlim(0, 11)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Turns per second that memory allows')
        save(fig, f'hook_step{k}')

def recall():
    fig, ax = plt.subplots(figsize=(3.0, 1.4))
    rows = [('Resend every turn', 892500.0, C['neutral'], '892,500'), ('Reuse the prefix', 51500.0, C['data'], '51,500')]
    for i, (lab, v, col, printed) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v / 1000.0, color=col, height=0.55)
        ax.text(-10, y, lab, ha='right', va='center', fontsize=12, fontweight='bold')
        ax.text(v / 1000.0 + 12, y, printed, va='center', fontsize=12, fontweight='bold', color=col)
    ax.set_xlim(0, 1150)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Input tokens prefilled over 30 turns (thousands)')
    save(fig, 'recall_resend')
ZONE_STYLE = {'System prompt': ('System\nprompt', 'cmp', 'z_system'), 'Tool schemas': ('Tool\nschemas', 'cmp', 'z_tools'), 'Workspace': ('Workspace', 'sto', 'z_workspace'), 'Turn history': ('Turn history', 'mem', 'z_history'), 'Latest observation': ('Obs.', 'tool', 'z_obs')}
ZONES = [('System\nprompt', 2048, 'cmp', 'z_system'), ('Tool\nschemas', 4096, 'cmp', 'z_tools'), ('Workspace', 8192, 'sto', 'z_workspace'), ('Turn history', 16384, 'mem', 'z_history'), ('Obs.', 2048, 'tool', 'z_obs')]

def zones():
    total = 32768
    W = 10.0
    for k in (1, 2):
        fig, ax = canvas(5.6, 2.45, (-0.2, 10.2), (-0.1, 5.75))
        x = 0
        for name, n, role, key in ZONES:
            w = W * n / total
            inside = w >= 2
            box(ax, x, 3.9, w, 1.0, f"{name}\n{ {'System\nprompt': '2,048', 'Tool\nschemas': '4,096', 'Workspace': '8,192', 'Turn history': '16,384', 'Obs.': '2,048'}[name]}" if inside else '', edge=R[role], fill=R[role + '_l'], fs=11, round_=False)
            x += w
        ax.text(0, 5.35, 'System prompt 2,048 + tool schemas 4,096', fontsize=10, color=R['cmp_t'], va='center')
        ax.plot([0.9, 0.9], [4.95, 5.15], color=R['cmp'], lw=1.2)
        ax.text(10.0, 5.35, 'Observation 2,048', fontsize=10, color=R['tool_t'], ha='right', va='center')
        ax.plot([9.7, 9.7], [4.95, 5.15], color=R['tool'], lw=1.2)
        pref = W * 14336 / total
        ax.annotate('', (0, 3.6), (pref, 3.6), arrowprops=dict(arrowstyle='<->', color=R['ink'], lw=1.3))
        ax.text(pref + 0.15, 3.6, 'shared invariant prefix: 14,336 tokens (43.8%)', fontsize=11, color=R['ink'], va='center')
        a2 = 1.0 if k == 2 else 0.0
        ax.text(0, 2.65, 'Every token keeps a key (K) and a value (V) at every layer:', fontsize=12, color=R['ink'], alpha=a2, va='center')
        for j, lab in enumerate(('K', 'V')):
            x0 = 0.45 + j * 2.15
            for r in range(4):
                for c in range(8):
                    ax.add_patch(Rectangle((x0 + c * 0.19, 0.85 + r * 0.33), 0.17, 0.3, fc=R['mem_l'], ec=R['mem'], lw=0.8, alpha=a2))
            ax.text(x0 - 0.12, 1.5, lab, ha='right', va='center', fontsize=14, fontweight='bold', color=R['mem_t'], alpha=a2)
        ax.text(0, 0.35, '8 KV heads × 128 values × 2 bytes, in each of 80 layers', fontsize=11, color=R['mem_t'], va='center', alpha=a2)
        ax.text(10.1, 1.75, '327,680 bytes per token', ha='right', va='center', fontsize=13, fontweight='bold', color=R['mem_t'], alpha=a2)
        ax.text(10.1, 0.95, '× 32,768 tokens = 10.7 GB', ha='right', va='center', fontsize=13, fontweight='bold', color=R['ink'], alpha=a2)
        save(fig, f'zones_step{k}')

def fit_curve():
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    s = np.logspace(np.log2(2048), np.log2(131072), 200, base=2)
    ax.plot(s, (Q_(525.9947674, 'GB') / (s * Q_(327680, 'B'))).to('').magnitude, color=C['memory'], lw=2.6)
    offsets = [(10, 4, 'left'), (10, 6, 'left'), (-4, 14, 'right')]
    for x, (dx, dy, ha) in zip([8192, 32768, 131072], offsets):
        y = {4: 195, 6: 48, 14: 12}[dy]
        ax.plot(x, y, 'o', color=C['memory'], ms=9, mec='white', mew=1.5, zorder=3)
        lab = {4: '195', 6: '48', 14: '12'}[dy] + ('  (r18)' if x == 32768 else '')
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, fontsize=14, fontweight='bold', color=C['danger'] if x == 32768 else C['text'])
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([2048, 8192, 32768, 131072], ['2K', '8K', '32K', '128K'])
    ax.set_yticks([10, 100, 1000], ['10', '100', '1,000'])
    ax.minorticks_off()
    ax.set_ylim(5, 1500)
    ax.set_xlabel('Tokens retained per sequence')
    ax.set_ylabel('Sequences that fit')
    ax.grid(color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, 'fit_curve')

def designs():
    kind_col = {'MHA': 'neutral', 'GQA': 'memory', 'MLA': 'data'}
    rows = [('GPT-3', 4718.592, 'neutral', '4,719 KB  MHA'), ('Llama-3.1-405B', 516.096, 'memory', '516 KB  GQA'), ('Llama-3.1-70B', 327.68, 'memory', '328 KB  GQA'), ('Llama-2-70B', 327.68, 'memory', '328 KB  GQA'), ('Mixtral-8x7B', 131.072, 'memory', '131 KB  GQA'), ('DeepSeek-V3', 70.272, 'data', '70.3 KB  MLA')]
    hb('designs', rows, 'KV cache per token, 16-bit (KB, log scale)', (5.4, 2.0), (30, 40000))

def kwon():
    fig, ax = plt.subplots(figsize=(5.5, 2.05))
    cols = [C['data'], C['compute'], C['danger'], C['neutral_l']]
    names = ['Token states', 'Reservation', 'Internal frag.', 'External frag. and other']
    rows = [('Orca (Max)', 20.4, 13.3, 57.3, 8.9), ('Orca (Pow2)', 26.8, 17.9, 13.6, 41.6), ('Orca (Oracle)', 38.2, 25.2, 0.0, 36.6), ('vLLM', 96.3, None, None, None)]
    for i, (lab, *vals) in enumerate(rows):
        if vals[1] is None:
            vals = [vals[0], 0, 0, 3.7]
        y = len(rows) - 1 - i
        left = 0
        for j, v in enumerate(vals):
            if v:
                ax.barh(y, v, left=left, color=cols[j], height=0.68, ec='white', lw=0.8)
                if v >= 8:
                    ax.text(left + v / 2, y, f'{v:.1f}', ha='center', va='center', fontsize=11, fontweight='bold', color='white' if j < 3 else C['text'])
            left += v
        ax.text(-2, y, lab.replace('\n', ' '), ha='right', va='center', fontsize=11, fontweight='bold')
    x = 0
    for j, n in enumerate(names):
        ax.text(x, 3.62, n, fontsize=11, color=cols[j] if j < 3 else C['neutral'], fontweight='bold', va='center')
        x += [27, 25, 27, 0][j]
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 3.9)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Share of KV cache memory (%)')
    save(fig, 'kwon_fig2')

def frag():
    for k in (1, 2):
        fig, ax = canvas(5.6, 2.2, (0, 32), (0, 8.2))
        ax.text(0, 7.6, 'Internal: a full-cap reservation, mostly unpopulated', fontsize=12, fontweight='bold', color=R['ink'], va='center')
        box(ax, 0, 5.4, 32, 1.5, edge=R['mem'], fill='white', round_=False, lw=1.6)
        box(ax, 0, 5.4, 32 * 2048 / 32768, 1.5, edge=R['mem'], fill=R['mem'], round_=False)
        ax.text(1.0, 4.85, '2,048 used', fontsize=11, color=R['mem_t'], ha='left', va='center')
        ax.text(17, 6.15, '30,720 reserved slots no other request can use', fontsize=11, ha='center', va='center', color=R['ink'])
        ax.text(32, 4.85, 'cap 32,768 (6.25% populated)', fontsize=11, ha='right', va='center', color=R['ink'])
        a = 1.0 if k == 2 else 0.0
        ax.text(0, 3.6, 'External: 768 slots free, but no gap holds 512 contiguous', fontsize=12, fontweight='bold', color=R['ink'], va='center', alpha=a)
        segs = [('live', 5), ('free', 5), ('live', 4), ('free', 5), ('live', 4), ('free', 5), ('live', 4)]
        x = 0
        for kind, w in segs:
            if kind == 'live':
                box(ax, x, 1.4, w, 1.5, '', edge=R['st'], fill=R['dim'], round_=False, alpha=a)
            else:
                box(ax, x, 1.4, w, 1.5, '256 free', edge=R['mem'], fill='white', round_=False, fs=11, alpha=a, ls='--')
            x += w
        box(ax, 20, 0.0, 12, 1.0, 'new request: 512 slots', edge=R['bad'], fill=R['bad_l'], fs=11, tc=R['bad_t'], bold=True, round_=False, alpha=a)
        ax.text(19.3, 0.5, 'fails', fontsize=12, fontweight='bold', color=R['bad_t'], ha='right', va='center', alpha=a)
        save(fig, f'frag_step{k}')

def blocktable():
    table = [3, 7, 9]
    for k in (1, 2, 3):
        fig, ax = canvas(5.6, 2.35, (-0.3, 13.3), (-0.3, 6.5))
        ax.text(0.2, 6.1, 'Logical blocks of 4 token positions', fontsize=12, fontweight='bold', color=R['ink'], va='center')
        for b in range(3):
            for s_ in range(4):
                p = b * 4 + s_
                filled = p <= 9
                hot = k == 3 and p == 9
                box(ax, 0.2 + b * 2.6 + s_ * 0.6, 4.6, 0.56, 0.9, str(p) if filled else '', edge=R['bad'] if hot else R['mem'], fill=R['mem_l'] if filled else 'white', fs=11, round_=False, lw=2.4 if hot else 1.2, bold=hot)
            ax.text(1.4 + b * 2.6, 4.25, f'block {b}', ha='center', va='center', fontsize=11, color=R['mem_t'])
        a2 = 1.0 if k >= 2 else 0.0
        arrow(ax, 7.9, 5.05, 8.85, 5.05, col=R['sto'], lw=1.6, alpha=a2)
        ax.text(10.85, 6.1, 'block table', fontsize=12, fontweight='bold', color=R['sto_t'], ha='center', va='center', alpha=a2)
        for b in range(3):
            hot = k == 3 and b == 2
            box(ax, 9.0 + b * 1.25, 4.6, 1.1, 0.9, str(table[b]), edge=R['bad'] if hot else R['sto'], fill=R['sto_l'], fs=13, bold=True, round_=False, alpha=a2, lw=2.4 if hot else 1.4)
            ax.text(9.55 + b * 1.25, 4.25, f'[{b}]', ha='center', va='center', fontsize=11, color=R['sto_t'], alpha=a2)
        others = {0, 1, 4, 5, 8, 10, 11}
        for i in range(12):
            mine = i in table
            hot = k == 3 and i == 9
            fill = R['mem_l'] if mine else R['dim'] if i in others else 'white'
            box(ax, 0.2 + i * 1.08, 1.55, 0.98, 1.0, str(i), edge=R['bad'] if hot else R['mem'] if mine else R['st'], fill=fill, fs=11, round_=False, alpha=a2, lw=2.4 if hot else 1.2, bold=mine)
        for b in range(3):
            tx = 9.55 + b * 1.25
            px = 0.2 + table[b] * 1.08 + 0.49
            hot = k == 3 and b == 2
            arrow(ax, tx, 4.0, px, 2.62, col=R['bad'] if hot else R['sto'], lw=2.2 if hot else 1.3, alpha=a2, ms=11)
        ax.text(0.2, 1.1, 'Physical blocks in HBM (gray: other requests; white: free)', fontsize=11, color=R['ink'], alpha=a2, va='center')
        a3 = 1.0 if k == 3 else 0.0
        ax.text(0.2, 0.2, 'Position 9: block 9 // 4 = 2, table entry 2 holds 9, slot 9 mod 4 = 1', fontsize=11, fontweight='bold', color=R['bad_t'], va='center', alpha=a3)
        save(fig, f'blocktable_step{k}')

def paged():
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.3, 1.45))
        rows = [('Full-cap reservation', 48, C['neutral']), ('Paged, 16-token blocks', 780, C['memory'])]
        for i, (lab, v, col) in enumerate(rows):
            y = 1 - i
            show = i == 0 or k == 2
            ax.barh(y, v if show else 0, color=col, height=0.55)
            ax.text(-12, y, lab, ha='right', va='center', fontsize=12, fontweight='bold')
            ax.text(v + 12, y, {48: '48', 780: '780'}[v] if show else '', va='center', fontsize=14, fontweight='bold', color=col)
            if i == 1 and k == 1:
                ax.text(12, y, '???', va='center', fontsize=15, fontweight='bold', color=C['danger'])
        ax.set_xlim(0, 900)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Final-length request snapshots in the 526 GB pool')
        save(fig, f'paged_step{k}')

def cow():
    for k in (1, 2):
        fig, ax = canvas(4.4, 2.05, (-0.3, 13.3), (-0.8, 5.8))
        rowsA, rowsB = ([3, 7, 9], [3, 7, 9] if k == 1 else [3, 7, 14])
        phys = {3: 7.0, 7: 8.4, 9: 9.8}
        if k == 2:
            phys[14] = 11.8
        for n, x in phys.items():
            box(ax, x, 2.6, 1.15, 1.0, f'{n}', edge=R['mem'], fill=R['mem_l'], fs=13, bold=True, round_=False)
        ax.text(13.2, 4.7, 'physical blocks', fontsize=11, color=R['mem_t'], ha='right', va='center')
        for r, (name, tab, y) in enumerate((('Candidate A', rowsA, 4.4), ('Candidate B', rowsB, 0.8))):
            ax.text(-0.2, y + 0.45, name, fontsize=11, fontweight='bold', color=R['ink'], va='center')
            for b, v in enumerate(tab):
                new = k == 2 and r == 1 and (b == 2)
                box(ax, 3.9 + b * 1.15, y, 1.0, 0.9, str(v), edge=R['bad'] if new else R['sto'], fill=R['sto_l'], fs=13, bold=True, round_=False, lw=2.2 if new else 1.4)
                tx = 4.4 + b * 1.15
                px = phys[v] + 0.57
                if r == 0:
                    arrow(ax, tx, y - 0.05, px, 3.64, col=R['sto'], lw=1.0, ms=9)
                else:
                    arrow(ax, tx, y + 0.95, px, 2.56, col=R['bad'] if v == 14 else R['sto'], lw=1.8 if v == 14 else 1.0, ms=9)
        if k == 1:
            ax.text(0.0, -0.5, 'Blocks 3, 7, 9: two owners each; nothing copied.', fontsize=11, color=R['ink'], fontweight='bold', va='center')
        else:
            arrow(ax, 10.98, 3.1, 11.78, 3.1, col=R['bad'], lw=1.8, ms=12)
            ax.text(0.0, -0.5, "B writes token 10: block 9's 2 entries copied to 14.", fontsize=11, color=R['bad_t'], fontweight='bold', va='center')
        save(fig, f'cow_step{k}')
NODES = {'root': (5.4, 4.6, None, 'shared prefix'), 'hist': (2.9, 2.7, 'root', 'repair history'), 'other': (8.4, 2.7, 'root', 'other task'), 'test': (1.4, 0.8, 'hist', 'test-log turn'), 'rej': (4.4, 0.8, 'hist', 'rejected branch'), 'final': (8.4, 0.8, 'other', 'final turn')}

def radix():
    active = {'root', 'hist', 'test'}
    for k in (1, 2, 3):
        fig, ax = canvas(5.5, 2.35, (-0.1, 10.0), (0.1, 5.5))
        evicted = {'final'} if k == 3 else set()
        eligible = {'rej', 'final'} if k == 2 else {'rej', 'other'} if k == 3 else set()
        for n, (x, y, p, lab) in NODES.items():
            if p:
                px, py = NODES[p][:2]
                on = n in active
                ax.plot([px, x], [py - 0.35, y + 0.35], color=R['mem'] if on else R['st'], lw=3.2 if on else 1.4, alpha=0.3 if n in evicted else 1.0, zorder=1)
        for n, (x, y, p, lab) in NODES.items():
            if n in active:
                edge, fill, ls, tc = (R['mem'], R['mem_l'], '-', R['mem_t'])
            elif n in evicted:
                edge, fill, ls, tc = (R['st'], 'white', ':', R['st'])
            elif n in eligible:
                edge, fill, ls, tc = (R['ink'], R['dim'], '--', R['ink'])
            else:
                edge, fill, ls, tc = (R['st'], 'white', '-', R['ink'])
            box(ax, x - 1.4, y - 0.35, 2.8, 0.7, lab, edge=edge, fill=fill, ls=ls, fs=11, tc=tc, bold=n in active, lw=1.8)
        notes = {1: "Green: the running request's path, root to leaf", 2: 'Gray, dashed: dormant leaves, the only eviction candidates', 3: 'Final turn evicted; its parent is now an eligible leaf'}
        ax.text(9.95, 5.3, notes[k], fontsize=11, ha='right', va='center', color=R['ink'], fontweight='bold')
        save(fig, f'radix_step{k}')

def hit():
    fig, ax = plt.subplots(figsize=(2.9, 1.3))
    rows = [('Miss: rebuild', 168.6174117, C['compute'], '169 ms'), ('Hit: one pass', 5.268656716, C['memory'], '5 ms')]
    for i, (lab, v, col, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=col, height=0.55)
        ax.text(-4, y, lab, ha='right', va='center', fontsize=12, fontweight='bold')
        ax.text(v + 4, y, txt, va='center', fontsize=13, fontweight='bold', color=col)
    ax.set_xlim(0, 220)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time for a 4,096-token prefix (ms)')
    save(fig, 'hit_time')
    fig, ax = plt.subplots(figsize=(2.9, 1.3))
    sglang = [('LLaVA-Next-34B', 52.4, '52.4'), ('Vicuna-33B', 74.1, '74.1')]
    for i, (lab, v, printed) in enumerate(sglang):
        y = 1 - i
        ax.barh(y, v, color=C['data'], height=0.55)
        ax.text(-2, y, lab, ha='right', va='center', fontsize=12, fontweight='bold')
        ax.text(v + 2, y, f'{printed}%', va='center', fontsize=13, fontweight='bold', color=C['data'])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Prompt tokens served from cache (%)')
    save(fig, 'sglang_hits')

def chunked():
    gm, ti, n = (167.7097308, 21.57420515, 8)
    pre = 162.441074
    dec = 5.268656716
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(5.6, 2.1))
        ax.barh(1, pre, left=0, height=0.45, color=C['compute'])
        ax.text(pre / 2, 1, 'prefill 4,096 tokens', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
        ax.barh(1, dec, left=pre, height=0.45, color=C['memory'])
        for x in (0, pre + dec):
            ax.plot([x, x], [1.28, 1.42], color=C['text'], lw=2)
        ax.annotate('', (0, 1.52), (pre + dec, 1.52), arrowprops=dict(arrowstyle='<->', color=C['danger'], lw=1.5))
        ax.text((pre + dec) / 2, 1.62, 'decodes wait 168 ms', ha='center', va='bottom', fontsize=11, fontweight='bold', color=C['danger'])
        ax.text(-3, 1, 'One unit', ha='right', va='center', fontsize=12, fontweight='bold')
        a = 1.0 if k == 2 else 0.0
        for i in range(n):
            ax.barh(0, ti - 0.6, left=i * ti, height=0.45, color=C['compute'], alpha=a)
            ax.plot([i * ti, i * ti], [0.28, 0.42], color=C['text'], lw=2, alpha=a)
        ax.plot([n * ti, n * ti], [0.28, 0.42], color=C['text'], lw=2, alpha=a)
        ax.text(ti / 2, 0, '1', ha='center', va='center', color='white', fontsize=10, fontweight='bold', alpha=a)
        ax.annotate('', (0, -0.36), (ti, -0.36), arrowprops=dict(arrowstyle='<->', color=C['text'], lw=1.3, alpha=a))
        ax.text(ti + 3, -0.36, 'gap 22 ms (512-token chunk + 32 decodes)', va='center', fontsize=11, fontweight='bold', color=C['text'], alpha=a)
        ax.text(n * ti + 3, 0, 'prompt done 173 ms', va='center', fontsize=11, alpha=a)
        ax.text(-3, 0, '8 chunks', ha='right', va='center', fontsize=12, fontweight='bold', alpha=a)
        ax.set_xlim(0, 240)
        ax.set_ylim(-0.7, 1.9)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Time (ms); ticks mark output tokens for 32 ongoing decodes')
        save(fig, f'chunk_step{k}')

def tool_waits():
    rows = [('Math (calculator)', 0.09, 'compute', None), ('Virtual env.', 90.0, 'compute', None), ('QA (Wikipedia)', 690.0, 'compute', None), ('Text to speech', 17240.0, 'network', None), ('Image generation', 20030.0, 'network', None), ('Chatbot (human)', 28600.0, 'network', None)]
    rows = [(n, ms, col, f'{ms:.2f} ms' if ms < 1 else f'{ms:.0f} ms' if ms < 1000 else f'{ms / 1000.0:.1f} s') for n, ms, col, _ in rows]
    hb('tool_waits', rows, 'Mean tool wait (ms, log scale)', (5.4, 1.95), (0.03, 300000.0), vlines=[(41.94304, 'reload round trip 42 ms', 'memory')])

def reload():
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(3.3, 1.35))
        rows = [('Reload over\n8 host links', 20.97152, C['memory'], '21 ms'), ('Recompute\nthe prefill', 1299.528592, C['compute'], '1.3 s')]
        for i, (lab, v, col, txt) in enumerate(rows):
            y = 1 - i
            ax.barh(y, v if k == 2 else 0, color=col, height=0.55)
            ax.text(-0.03, y, lab, ha='right', va='center', fontsize=12, fontweight='bold', transform=ax.get_yaxis_transform())
            ax.text(v * 1.2, y, txt if k == 2 else '', va='center', fontsize=13, fontweight='bold', color=col)
            if k == 1:
                ax.text(12, y, '???', va='center', fontsize=15, fontweight='bold', color=C['danger'])
        ax.set_xscale('log')
        ax.set_xlim(6, 20000.0)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xticks([10, 100, 1000, 10000], ['10 ms', '100 ms', '1 s', '10 s'])
        ax.minorticks_off()
        ax.set_xlabel('Time to restore 32,768 tokens (log scale)')
        save(fig, f'reload_step{k}')

def regimes():
    fig, ax = plt.subplots(figsize=(3.75, 2.05))
    ctx = np.logspace(np.log2(2048), np.log2(131072), 100, base=2)
    rt = 2 * ctx * (Q_(327680, 'B') / Q_(512.0, 'GB / s')).to('s').magnitude
    host = 60.0
    ax.fill_betweenx(ctx, 0.001, rt, color=R['mem_l'])
    ax.fill_betweenx(ctx, rt, host, color=R['sto_l'])
    ax.fill_betweenx(ctx, host, 3600, color=R['dim'])
    ax.plot(rt, ctx, color=C['memory'], lw=2.2)
    ax.axvline(host, color=C['neutral'], ls='--', lw=1.6)
    ax.text(0.0016, 60000.0, 'keep', fontsize=13, fontweight='bold', color=R['mem_t'])
    ax.text(0.15, 2600.0, 'move to host', fontsize=12, fontweight='bold', color=R['sto_t'])
    ax.text(80, 50000.0, 'park\nor\ndrop', fontsize=11, fontweight='bold', color=C['neutral'])
    r18 = (45, 32768)
    ax.plot(*r18, 'o', ms=10, color=C['danger'], mec='white', mew=1.5, zorder=4)
    ax.annotate('r18: 45 s test', r18, xytext=(-12, -16), textcoords='offset points', ha='right', fontsize=12, fontweight='bold', color=C['danger'])
    ax.set_xscale('log')
    ax.set_yscale('log', base=2)
    ax.set_xlim(0.001, 3600)
    ax.set_ylim(2048, 131072)
    ax.set_xticks([0.001, 0.1, 10, 3600], ['1 ms', '100 ms', '10 s', '1 h'])
    ax.set_yticks([2048, 8192, 32768, 131072], ['2K', '8K', '32K', '128K'])
    ax.minorticks_off()
    ax.set_xlabel('Expected tool wait')
    ax.set_ylabel('Context (tokens)')
    save(fig, 'regimes')

def burst():
    fig, ax = plt.subplots(figsize=(3.6, 1.9))
    n = np.arange(1, 32 + 1)
    one = 20.97152
    t = n * one
    ax.bar(n, t, color=C['memory'], width=0.75)
    ax.axhline(one, color=C['neutral'], ls='--', lw=1.2)
    ax.annotate('one alone: 21 ms', (1, one), xytext=(1.5, 560), fontsize=12, color=C['neutral'], fontweight='bold', arrowprops=dict(arrowstyle='-|>', color=C['neutral'], lw=1.2))
    ax.annotate('last: 0.67 s', (n[-1], t[-1]), xytext=(-6, 6), textcoords='offset points', ha='right', fontsize=13, fontweight='bold', color=C['danger'])
    ax.set_xlabel('Returning repair, in reload order')
    ax.set_ylabel('Resume delay (ms)')
    ax.set_xlim(0, 33.5)
    ax.set_ylim(0, 800)
    save(fig, 'burst')

def stream():
    fig, ax = plt.subplots(figsize=(3.05, 1.75))
    labs, vals, cols = ([], [], [])
    for e in ({'model': 'Llama-2-13B', 'cache_tokens': 1024.0, 'sink_tokens': 4.0, 'window_perplexity': 5158.07, 'sink_perplexity': 5.4, 'table': 1.0}, {'model': 'Llama-2-7B', 'cache_tokens': 4096.0, 'sink_tokens': 4.0, 'window_perplexity': 3359.95, 'sink_perplexity': 9.59, 'table': 2.0}):
        win, sink = (e['window_perplexity'], e['sink_perplexity'])
        labs += ['window', '+ first 4']
        vals += [win, sink]
        cols += [C['danger'], C['data']]
    x = np.arange(4)
    ax.bar(x, vals, color=cols, width=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.25, f'{v:,.2f}' if v < 100 else f'{v:,.0f}', ha='center', fontsize=12, fontweight='bold', color=C['text'])
    ax.set_yscale('log')
    ax.set_ylim(1, 100000.0)
    ax.set_xticks(x, labs, fontsize=10)
    for xi, name in ((0.5, 'Llama-2-13B'), (2.5, 'Llama-2-7B')):
        ax.text(xi, -0.36, name, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=11, fontweight='bold')
    ax.set_ylabel('Perplexity (log)')
    ax.minorticks_off()
    save(fig, 'stream')

def residency():
    fig, ax = plt.subplots(figsize=(5.4, 1.3))
    gen, wait = (5.0, 45.0)
    for y, w_wait, lab in ((1, wait, 'Keep'), (0, 0, 'Release')):
        ax.barh(y, gen, color=C['compute'], height=0.5)
        if y == 1:
            ax.text(0, 1.38, 'generate 5 s', ha='left', va='bottom', color=C['compute'], fontsize=11, fontweight='bold')
        if w_wait:
            ax.barh(y, wait, left=gen, color=C['neutral_l'], height=0.5)
            ax.text(gen + wait / 2, y, 'tool wait 45 s: slot still held', ha='center', va='center', fontsize=11)
        else:
            ax.barh(y, wait, left=gen, color='white', height=0.5, ec=C['neutral_l'], ls='--')
            ax.text(gen + wait / 2, y, 'tool wait 45 s: slot serves other turns', ha='center', va='center', fontsize=11, color=C['neutral'])
        ax.text(-1, y, lab, ha='right', va='center', fontsize=12, fontweight='bold')
    ax.text(gen + wait + 0.5, 1, 'W = 50 s', va='center', fontsize=12, fontweight='bold', color=C['danger'])
    ax.text(gen + wait + 0.5, 0, 'W = 5 s', va='center', fontsize=12, fontweight='bold', color=C['memory'])
    ax.set_xlim(0, 58)
    ax.set_ylim(-0.5, 1.75)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Seconds in one repair turn')
    save(fig, 'residency')

def rates():
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(3.6, 1.95))
        groups = [('41 slots, unshared', 0.82, 8.2, '0.82', '8.2'), ('138 slots, workspace shared', 2.76, 27.6, '2.76', '27.6')]
        for g, (lab, keep, rel, tk, tr) in enumerate(groups):
            show = g == 0 or k == 2
            y0 = 2.6 - g * 2.0
            for j, (v, col, nm, t) in enumerate(((keep, C['neutral'], 'keep', tk), (rel, C['memory'], 'release', tr))):
                y = y0 - j * 0.65
                ax.barh(y, v if show else 0, color=col, height=0.5)
                ax.text(-0.4, y, nm, ha='right', va='center', fontsize=11, alpha=1 if show else 0)
                ax.text(v + 0.4, y, t, va='center', fontsize=12, fontweight='bold', color=col, alpha=1 if show else 0)
            ax.text(0, y0 + 0.55, lab, ha='left', va='center', fontsize=12, fontweight='bold', alpha=1 if show else 0)
        ax.set_xlim(0, 32)
        ax.set_ylim(-0.3, 3.4)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Turns per second memory allows')
        save(fig, f'rates_step{k}')

def gain():
    fig, ax = plt.subplots(figsize=(3.05, 1.8))
    xs = np.linspace(0, 0.9, 200)
    ax.plot(xs * 100, 1 / (1 - xs), color=C['data'], lw=2.6)
    pts = [(0.1875, 1.25, 'repair: 1.25×', (28, 0.15)), (0.7058823529, 3.365853659, 'repair, shared\nworkspace: 3.4×', (3, 4.6)), (0.75, 4.0, 'support: 4.0×', (30, 8.3))]
    for f, g, lab, xy in pts:
        ax.plot(f * 100, g, 'o', ms=8, color=C['memory'], mec='white', mew=1.5, zorder=3)
        ax.annotate(lab, (f * 100, g), xytext=xy, textcoords='data', fontsize=11, fontweight='bold', color=C['text'], arrowprops=dict(arrowstyle='-', color=C['neutral'], lw=0.8))
    ax.set_xlabel('Shared fraction of each request (%)')
    ax.set_ylabel('Resident-fit gain')
    ax.set_xlim(0, 92)
    ax.set_ylim(0, 10.5)
    ax.set_yticks([0, 5, 10])
    ax.grid(color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    save(fig, 'gain_curve')
if __name__ == '__main__':
    for fn in (hook, recall, zones, fit_curve, designs, kwon, frag, blocktable, paged, cow, radix, hit, chunked, tool_waits, reload, regimes, burst, stream, residency, rates, gain):
        fn()
    print('figures written to', OUT)
