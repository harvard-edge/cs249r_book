#!/usr/bin/env python3
"""Figures for Lecture 16 (Volume II): Responsible AI.

Every number drawn comes from calcs.py (MLSysIM, via deckvals) or is a sourced
input listed here. "Ch." is the rendered chapter (Volume II, Chapter 16).

  - Ofqual, Grading gaps in summer 2020 (July 2021), Table 1 text: A level
    entries with the calculated grade equal to the CAG 59%, lower 39%,
    higher "just over 2%" (n = 457,420 entries).
  - Loan model (Ch. Napkin Math 16.1, 16.2; rates in calcs.py).
  - COMPAS: recomputed from ProPublica's compas-scores-two-years.csv with
    ProPublica's filters (counts and rates in calcs.py).
  - Dual thresholds: illustrative scores in calcs.py (after Ch. figure 5).
  - Obermeyer et al. 2019 (Ch. footnote 7): 17.7 and 46.5 percent.
  - z-test: Ch. 16.5.2 formula, computed in calcs.py.
  - Gender Shades: mlsysim literature series gender_shades_audit
    (Buolamwini and Gebru 2018, Table 4; Raji and Buolamwini 2019, Table 1).
  - Monitoring scenario: Ch. 16.5.5 (calcs.py).
  - Overheads: Ch. table 5 (scenario ranges, not measurements).
  - SHAP subsets: Ch. 16.7.2; 1 us per subset from the Ch. 16.7 self-check.
  - DP-SGD accuracy: Abadi et al., CCS 2016, Sec. 5.2 (MNIST) and Fig. 6
    (CIFAR-10), delta = 1e-5.
  - Unlearning: Ch. Napkin Math 16.6 (calcs.py).
  - Automation bias: Ch. Napkin Math 16.7 (calcs.py).
  - Waymo: mlsysim series waymo_rider_only_miles and
    waymo_vs_human_benchmark (Waymo Safety Impact Data Hub, through June 2026).

Run from this folder: python3 make_figs.py
"""
import csv
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import contextlib
import io
import matplotlib.pyplot as plt
import slideplot as sp
C = sp.C
ETH = '#1F407A'
ETH_L = '#EAF1F8'
AMBER = '#d97706'
AMBER_L = '#fef3c7'
VIOLET = '#7c3aed'
VIOLET_L = '#f3e8ff'
GREEN = '#16a34a'
GREEN_L = '#dcfce7'
RED = '#dc2626'
RED_L = '#fee2e2'
FRAME = '#94a3b8'
FRAME_L = '#f8fafc'
INK = '#0f172a'
GRAY = '#64748b'
FONT = 'Helvetica, Arial, sans-serif'

def out(name):
    return str(HERE / name)

def svg_doc(body, w=900, h=500):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">\n<defs>{arrow_defs()}</defs>\n<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>\n{body}\n</svg>\n'

def arrow_defs():
    s = ''
    for nm, col in (('blue', ETH), ('amber', AMBER), ('purple', VIOLET), ('green', GREEN), ('red', RED), ('gray', GRAY)):
        s += f'<marker id="arr-{nm}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
    return s

def box(x, y, w, h, fill, stroke, lines, size=20, color=INK, bold_first=True, rx=6, dash=None, sw=2):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n'
    n = len(lines)
    lh = size * 1.25
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for i, t in enumerate(lines):
        wt = 'bold' if i == 0 and bold_first else 'normal'
        s += f'<text x="{x + w / 2}" y="{y0 + i * lh:.1f}" text-anchor="middle" font-size="{size}" font-weight="{wt}" fill="{color}">{t}</text>\n'
    return s

def text(x, y, t, size=20, color=INK, anchor='middle', weight='normal', italic=False):
    st = ' font-style="italic"' if italic else ''
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}"{st}>{t}</text>\n'

def arrow(x1, y1, x2, y2, col='gray', w=2.5, dash=None):
    colhex = {'blue': ETH, 'amber': AMBER, 'purple': VIOLET, 'green': GREEN, 'red': RED, 'gray': GRAY}[col]
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colhex}" stroke-width="{w}" marker-end="url(#arr-{col})"{d}/>\n'

def path_arrow(dpath, col='gray', w=2.5, dash=None):
    colhex = {'blue': ETH, 'amber': AMBER, 'purple': VIOLET, 'green': GREEN, 'red': RED, 'gray': GRAY}[col]
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<path d="{dpath}" fill="none" stroke="{colhex}" stroke-width="{w}" marker-end="url(#arr-{col})"{d}/>\n'

def write(name, body, w=900, h=500):
    Path(out(name)).write_text(svg_doc(body, w, h))
plt.rcParams.update({'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'axes.labelsize': 13})

def hook():
    rows = [('Lower than teacher grade', 39.0, 'danger', '39%'), ('Same as teacher grade', 59.0, 'neutral', '59%'), ('Higher than teacher grade', 2.0, 'data', 'about 2%')]
    for k, step in ((1, 0), (2, 3)):
        sp.hbars(out(f'hook_step{k}.svg'), rows, '%', log=False, step=step, xlabel='Share of 457,420 A level entries (%)', size=(5.2, 1.8))

def ofqual_flows():
    """Ofqual Figure 1 redrawn from its Table 1: for each teacher grade (CAG),
    the share of entries whose calculated grade was higher, the same, one
    grade lower, or two or more lower. Column slot about 7 x 5.3 cm."""
    split = [(0, 61.0, 36.0, 3.0), (1.0, 56.0, 39.0, 3.0), (1.0, 59.0, 37.0, 3.0), (2.0, 59.0, 34.0, 4.0), (5.0, 59.0, 31.0, 4.0), (12.0, 60.0, 28.0, 0), (20.0, 79.0, 0, 0)]
    grades = ['A*', 'A', 'B', 'C', 'D', 'E', 'U']
    cols = [('higher', C['data']), ('same', C['neutral_l']), ('one lower', RED_L), ('two or more lower', RED)]
    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    for i, s in enumerate(split):
        y = len(split) - 1 - i
        left = 0
        for (name, col), v in zip(cols, s):
            ax.barh(y, v, left=left, height=0.7, color=col, edgecolor='white', lw=0.8)
            left += v
        lower = s[2] + s[3]
        ax.text(101.5, y, f'{lower:.0f}%', va='center', ha='left', fontsize=12, fontweight='bold', color=RED if lower else GRAY)
    ax.text(101.5, len(split) - 0.35, 'lower', va='bottom', ha='left', fontsize=12, color=RED)
    top = len(split) - 0.35
    ax.text(0, top, 'higher', ha='left', va='bottom', fontsize=12, color=C['data'])
    ax.text(36, top, 'same', ha='center', va='bottom', fontsize=12, color=GRAY)
    ax.text(80, top, 'one lower', ha='center', va='bottom', fontsize=12, color=RED, fontweight='bold')
    ax.set_yticks(range(len(split)))
    ax.set_yticklabels(list(reversed(grades)), fontsize=12)
    ax.set_ylabel('Teacher grade', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(split) + 0.3)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel('Calculated grade, share of entries (%)', fontsize=12)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    sp.save(fig, out('ofqual_flows.svg'))

def ofqual_candidates():
    """Ofqual Table 2 as bars: candidates by the number of A levels lowered."""
    n = [78470.0, 95580.0, 55730.0, 16100.0, 220.0]
    pct = [100 * v / 246110.0 for v in n]
    fig, ax = plt.subplots(figsize=(4.4, 1.8))
    for k, p in enumerate(pct):
        ax.bar(k, p, width=0.7, color=GRAY if k == 0 else RED)
        ax.text(k, p + 1.5, f'{p:.0f}%' if p >= 0.5 else '<0.5%', ha='center', va='bottom', fontsize=12, color=GRAY if k == 0 else RED, fontweight='bold')
    at_least = 68.1
    ax.plot([0.7, 4.3], [52, 52], color=RED, lw=1.5)
    for x in (0.7, 4.3):
        ax.plot([x, x], [49, 52], color=RED, lw=1.5)
    ax.text(2.5, 54, f'{at_least:.0f}% had at least one lowered', ha='center', va='bottom', fontsize=12, fontweight='bold', color=RED)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['0', '1', '2', '3', '4'], fontsize=12)
    ax.set_xlabel('A levels lowered per student', fontsize=12)
    ax.set_ylim(0, 66)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    sp.save(fig, out('ofqual_candidates.svg'))

def assurance():

    def frame(k):
        b = text(450, 30, 'One stated use, one operating envelope', 22, GRAY, weight='bold')
        ev = ['Subgroup results', 'Audit reports', 'Decision logs', 'Documentation']
        for i, t in enumerate(ev):
            b += box(10, 52 + i * 58, 200, 48, GREEN_L, GREEN, [t], size=22, bold_first=False)
        b += text(110, 300, 'Evidence', 23, GREEN, weight='bold')
        b += path_arrow('M210,170 L238,170', 'green')
        b += box(240, 120, 180, 100, FRAME_L, FRAME, ['Argument', 'why the', 'evidence holds'], size=22)
        b += path_arrow('M420,170 L448,170', 'blue')
        b += box(450, 110, 190, 120, ETH_L, ETH, ['Claim', 'loan screening', 'meets criteria'], size=22)
        if k >= 2:
            b += box(240, 262, 400, 64, 'white', GRAY, ['Assumptions:', 'population, setting, suppliers'], size=22, bold_first=True, dash='7,5')
        if k >= 3:
            b += path_arrow('M640,150 L676,110', 'purple')
            b += box(678, 52, 212, 80, VIOLET_L, VIOLET, ['Owner', 'can block launch'], size=22)
            b += path_arrow('M784,132 L784,170', 'purple')
            b += box(678, 172, 212, 100, VIOLET_L, VIOLET, ['Decision', 'launch, restrict,', 'or reject'], size=22)
        if k >= 4:
            b += box(240, 352, 200, 64, 'white', GRAY, ['Material change'], size=22)
            b += path_arrow('M440,384 L478,384', 'red')
            b += box(480, 352, 200, 64, RED_L, RED, ['Invalidation rule'], size=22)
            b += path_arrow('M680,384 C760,384 790,340 784,276', 'red', dash='8,5')
            b += text(800, 330, 'approval', 22, RED, anchor='start', weight='bold')
            b += text(800, 356, 'expires', 22, RED, anchor='start', weight='bold')
        write(f'assure_step{k}.svg', b, 900, 430)
    for k in range(1, 5):
        frame(k)

def lifecycle():
    stages = ['Context', 'Risk', 'Controls', 'Evaluate', 'Launch', 'Operate', 'Change', 'Retire']
    b = ''
    w, gap, x0, y = (100, 12, 6, 30)
    for i, s in enumerate(stages):
        x = x0 + i * (w + gap)
        fill, st = (VIOLET_L, VIOLET) if i == 4 else (RED_L, RED) if i == 6 else (GREEN_L, GREEN) if i == 7 else (ETH_L, ETH)
        b += box(x, y, w, 60, fill, st, [s], size=19, sw=3 if i in (4, 6, 7) else 2)
        if i < 7:
            b += path_arrow(f'M{x + w},{y + 30} L{x + w + gap - 1},{y + 30}', 'gray', w=2)
    xm = x0 + 6 * (w + gap) + w / 2
    xr = x0 + 1 * (w + gap) + w / 2
    b += path_arrow(f'M{xm},{y + 60} C{xm},{y + 125} {xr},{y + 125} {xr},{y + 64}', 'red', dash='8,5')
    b += text((xm + xr) / 2, y + 150, 'new model, data, corpus, region, or use: re-open the risk review', 21, RED)
    lx = x0 + 4 * (w + gap) + w / 2
    b += text(lx, y + 195, 'Launch: who may authorize, and who may stop it?', 21, VIOLET, weight='bold')
    rx = x0 + 7 * (w + gap) + w / 2
    b += text(rx - 26, y + 242, 'Retire: an appeal owner remains after shutdown', 21, GREEN, anchor='end', weight='bold')
    b += path_arrow(f'M{rx - 20},{y + 236} C{rx + 4},{y + 230} {rx},{y + 150} {rx},{y + 64}', 'green', w=2)
    write('lifecycle.svg', b, 900, 290)

def loan_metrics():
    import numpy as np
    names = ['Approval', 'TPR', 'FPR', 'PPV']
    keys = ['approval', 'tpr', 'fpr', 'ppv']
    a = [0.7, 0.8888888889, 0.5454545455, 0.5714285714]
    bb = [0.4, 0.6, 0.2, 0.75]
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    x = np.arange(4)
    ax.bar(x - 0.2, a, 0.38, color=ETH)
    ax.bar(x + 0.2, bb, 0.38, color=AMBER)
    for i in range(4):
        ax.text(x[i] - 0.2, a[i] + 0.03, f'{a[i]:.2f}', ha='center', fontsize=12.5, color=ETH, fontweight='bold')
        ax.text(x[i] + 0.2, bb[i] + 0.03, f'{bb[i]:.2f}', ha='center', fontsize=12.5, color=AMBER, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1.0])
    ax.text(-0.45, 1.12, 'Group A', color=ETH, fontsize=13, fontweight='bold')
    ax.text(0.75, 1.12, 'Group B', color=AMBER, fontsize=13, fontweight='bold')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out('loan_metrics.svg'))

def compas():
    import numpy as np
    names = ['Base rate', 'FPR', 'PPV']
    blk = [52.31496063, 42.33817701, 64.95352652]
    wht = [39.08701854, 22.01405152, 59.48275862]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(3)
    ax.bar(x - 0.2, blk, 0.38, color=ETH)
    ax.bar(x + 0.2, wht, 0.38, color=C['neutral'])
    for i in range(3):
        ax.text(x[i] - 0.2, blk[i] + 1.5, f'{blk[i]:.1f}', ha='center', fontsize=12.5, color=ETH, fontweight='bold')
        ax.text(x[i] + 0.2, wht[i] + 1.5, f'{wht[i]:.1f}', ha='center', fontsize=12.5, color=C['neutral'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylim(0, 92)
    ax.set_ylabel('Percent')
    ax.text(-0.45, 82, 'Black', color=ETH, fontsize=13, fontweight='bold')
    ax.text(0.2, 82, 'White defendants', color=C['neutral'], fontsize=13, fontweight='bold')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out('compas.svg'))

def dual_thresholds():

    def frame(k):
        b = ''
        x0, x1 = (70, 880)

        def X(s):
            return x0 + s * (x1 - x0)
        bands = (('A', 36, ETH_L, ETH), ('B', 182, AMBER_L, AMBER))
        for g, y, fill, st in bands:
            b += f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="100" fill="{fill}" stroke="{st}" stroke-width="1.5"/>\n'
            b += text(32, y + 62, g, 34, st, weight='bold')
            t = 0.75 if k == 1 else {'A': 0.75, 'B': 0.45}[g]
            b += f'<rect x="{X(t)}" y="{y}" width="{x1 - X(t)}" height="100" fill="{st}" fill-opacity="0.18"/>\n'
            b += f'<line x1="{X(t)}" y1="{y - 4}" x2="{X(t)}" y2="{y + 104}" stroke="{INK}" stroke-width="3" stroke-dasharray="8,5"/>\n'
            b += text(X(t) - 8, y - 8, f'cutoff {t:.2f}', 22, INK, anchor='end', weight='bold')
            for s in {'A': [0.93, 0.85, 0.79, 0.6], 'B': [0.8, 0.6, 0.5, 0.33]}[g]:
                b += text(X(s), y + 46, '+', 40, st, weight='bold')
            for s in {'A': [0.7, 0.55, 0.38, 0.22], 'B': [0.52, 0.4, 0.28, 0.15]}[g]:
                b += f'<circle cx="{X(s)}" cy="{y + 72}" r="10" fill="white" stroke="{st}" stroke-width="3"/>\n'
            tpr = sum(1 for s in {'A': [0.93, 0.85, 0.79, 0.6], 'B': [0.8, 0.6, 0.5, 0.33]}[g] if s >= t) / 4
            b += text(x1 - 8, y + 94, f'TPR {100 * tpr:.0f}%', 24, INK, anchor='end', weight='bold')
        b += f'<line x1="{x0}" y1="300" x2="{x1}" y2="300" stroke="{GRAY}" stroke-width="1.5"/>\n'
        for s in (0, 0.25, 0.5, 0.75, 1.0):
            b += text(X(s), 326, f'{s:g}', 22, GRAY)
        if k == 2:
            same = 0.6
            b += text(X(same), 370, 'Two applicants at 0.60: A denied, B approved', 24, RED, weight='bold')
            b += path_arrow(f'M{X(same) - 150},350 C{X(same) - 200},300 {X(same) - 40},140 {X(same) - 6},96', 'red', w=2.2)
            b += path_arrow(f'M{X(same) + 30},350 L{X(same) + 4},{182 + 46}', 'red', w=2.2)
        write(f'dual_step{k}.svg', b, 900, 385)
    frame(1)
    frame(2)

def obermeyer():
    rows = [('Before (cost label)', 17.7, 'neutral', '17.7%'), ('After (simulated fix)', 46.5, 'memory', '46.5%')]
    sp.hbars(out('obermeyer.svg'), rows, '%', log=False, xlabel='Black patients among those auto-identified (%)', size=(4.6, 1.5))

def ztest():
    import numpy as np
    ns = np.linspace(4, 130, 300)
    zs = [0.8528028654, 0.8966001118, 0.9383553663, 0.9783301215, 1.01673441, 1.053739955, 1.089489296, 1.12410229, 1.157680868, 1.190312575, 1.222073262, 1.253029166, 1.28323853, 1.312752893, 1.341618124, 1.369875259, 1.397561182, 1.424709195, 1.451349484, 1.477509512, 1.503214353, 1.528486972, 1.553348466, 1.577818269, 1.601914331, 1.62565327, 1.649050511, 1.672120397, 1.694876295, 1.717330687, 1.739495249, 1.761380923, 1.782997978, 1.804356068, 1.825464285, 1.846331197, 1.866964895, 1.88737303, 1.907562839, 1.927541185, 1.947314575, 1.96688919, 1.986270908, 2.005465321, 2.024477756, 2.043313294, 2.061976783, 2.080472851, 2.098805927, 2.116980243, 2.134999856, 2.152868648, 2.170590346, 2.188168523, 2.20560661, 2.222907904, 2.240075575, 2.257112672, 2.27402213, 2.290806775, 2.307469331, 2.324012424, 2.340438588, 2.356750266, 2.37294982, 2.389039531, 2.405021604, 2.42089817, 2.436671291, 2.452342964, 2.467915122, 2.483389635, 2.498768319, 2.514052932, 2.52924518, 2.544346716, 2.559359148, 2.574284032, 2.589122885, 2.603877176, 2.618548334, 2.63313775, 2.647646775, 2.662076723, 2.676428873, 2.69070447, 2.704904726, 2.719030822, 2.733083907, 2.747065102, 2.7609755, 2.774816164, 2.788588133, 2.80229242, 2.815930014, 2.829501878, 2.843008953, 2.85645216, 2.869832395, 2.883150535, 2.896407437, 2.909603938, 2.922740855, 2.93581899, 2.948839123, 2.96180202, 2.974708429, 2.987559083, 3.000354697, 3.013095973, 3.025783596, 3.038418241, 3.051000564, 3.06353121, 3.07601081, 3.088439985, 3.100819339, 3.113149467, 3.125430952, 3.137664366, 3.149850267, 3.161989206, 3.174081721, 3.186128341, 3.198129584, 3.21008596, 3.221997968, 3.233866097, 3.245690831, 3.25747264, 3.26921199, 3.280909336, 3.292565125, 3.304179798, 3.315753786, 3.327287514, 3.3387814, 3.350235853, 3.361651276, 3.373028066, 3.384366612, 3.395667298, 3.4069305, 3.418156589, 3.429345929, 3.440498878, 3.451615791, 3.462697012, 3.473742885, 3.484753745, 3.495729924, 3.506671746, 3.517579532, 3.528453599, 3.539294256, 3.55010181, 3.560876563, 3.571618811, 3.582328846, 3.593006958, 3.603653428, 3.614268538, 3.624852562, 3.635405772, 3.645928436, 3.656420818, 3.666883176, 3.677315769, 3.687718847, 3.698092661, 3.708437455, 3.718753473, 3.729040953, 3.73930013, 3.749531237, 3.759734502, 3.769910153, 3.780058411, 3.790179498, 3.80027363, 3.810341021, 3.820381882, 3.830396423, 3.840384849, 3.850347364, 3.860284168, 3.870195459, 3.880081433, 3.889942282, 3.899778198, 3.909589368, 3.919375978, 3.929138212, 3.938876251, 3.948590275, 3.958280459, 3.967946978, 3.977590006, 3.987209712, 3.996806265, 4.006379832, 4.015930576, 4.02545866, 4.034964245, 4.044447489, 4.053908549, 4.06334758, 4.072764735, 4.082160166, 4.091534022, 4.100886451, 4.1102176, 4.119527613, 4.128816633, 4.138084801, 4.147332257, 4.15655914, 4.165765586, 4.17495173, 4.184117707, 4.193263648, 4.202389683, 4.211495944, 4.220582557, 4.229649649, 4.238697345, 4.24772577, 4.256735046, 4.265725294, 4.274696634, 4.283649186, 4.292583066, 4.301498392, 4.310395278, 4.319273837, 4.328134184, 4.336976429, 4.345800684, 4.354607056, 4.363395656, 4.372166589, 4.380919962, 4.38965588, 4.398374448, 4.407075767, 4.41575994, 4.424427068, 4.433077251, 4.441710588, 4.450327177, 4.458927115, 4.467510498, 4.476077421, 4.484627979, 4.493162266, 4.501680373, 4.510182392, 4.518668415, 4.527138531, 4.535592829, 4.544031398, 4.552454325, 4.560861696, 4.569253598, 4.577630116, 4.585991333, 4.594337334, 4.602668202, 4.610984017, 4.619284863, 4.627570818, 4.635841963, 4.644098378, 4.65234014, 4.660567327, 4.668780016, 4.676978285, 4.685162207, 4.693331859, 4.701487315, 4.709628648, 4.717755933, 4.72586924, 4.733968642, 4.742054211, 4.750126016, 4.758184129, 4.766228618, 4.774259552, 4.782277, 4.790281029, 4.798271707, 4.806249099, 4.814213273, 4.822164293, 4.830102225, 4.838027133, 4.84593908, 4.853838131, 4.861724348]
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    ax.plot(ns, zs, color=ETH, lw=3)
    ax.axhline(round(1.959963985, 2), color=RED, lw=1.8, ls='--')
    ax.text(129, 1.45, 'z = 1.96 (5%, two-sided)', color=RED, fontsize=13, ha='right', fontweight='bold')
    for n in (10, 100):
        z = {10: 1.348399725, 100: 4.264014327}[n]
        col = AMBER if n == 10 else ETH
        ax.plot([n], [z], 'o', ms=11, color=col, mec='white', mew=1.5, zorder=5)
        if n == 10:
            ax.annotate(f'{n} per group: z = {z:.2f}', (n, z), xytext=(12, -24), textcoords='offset points', ha='left', fontsize=13, fontweight='bold', color=col)
        else:
            ax.annotate(f'{n} per group: z = {z:.2f}', (n, z), xytext=(-10, 10), textcoords='offset points', ha='right', fontsize=13, fontweight='bold', color=col)
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 5.4)
    ax.set_xlabel('Applicants per group (0.70 against 0.40 approved)')
    ax.set_ylabel('z statistic')
    ax.grid(color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out('ztest.svg'))

def gender_grid():
    acc = {'lighter male': 99.7, 'lighter female': 92.9, 'darker male': 88.0, 'darker female': 65.3}
    b = text(150, 22, 'IBM, 2017 accuracy', 22, GRAY, weight='bold')
    cells = [('lighter male', 6, 36, '#e5e7eb'), ('lighter female', 152, 36, '#d1d5db'), ('darker male', 6, 152, '#9ca3af'), ('darker female', 152, 152, RED_L)]
    for k, x, y, fill in cells:
        dk = k == 'darker female'
        b += box(x, y, 142, 110, fill, RED if dk else GRAY, [f'{acc[k]:.1f}%'] + k.split(' '), size=22, color=RED if dk else INK, sw=3 if dk else 1.5)
    write('gs_grid.svg', b, 300, 268)

def gender_dumbbell():
    rows = ({'company': 'Microsoft', 'target': 1.0, 'subgroup': 'darker female', 'error_may2017_pct': 20.8, 'error_aug2018_pct': 1.52}, {'company': 'Microsoft', 'target': 1.0, 'subgroup': 'darker male', 'error_may2017_pct': 6.0, 'error_aug2018_pct': 0.33}, {'company': 'Microsoft', 'target': 1.0, 'subgroup': 'lighter female', 'error_may2017_pct': 1.7, 'error_aug2018_pct': 0.34}, {'company': 'Microsoft', 'target': 1.0, 'subgroup': 'lighter male', 'error_may2017_pct': 0.0, 'error_aug2018_pct': 0.0}, {'company': 'Face++', 'target': 1.0, 'subgroup': 'darker female', 'error_may2017_pct': 34.5, 'error_aug2018_pct': 4.1}, {'company': 'Face++', 'target': 1.0, 'subgroup': 'darker male', 'error_may2017_pct': 0.7, 'error_aug2018_pct': 1.3}, {'company': 'Face++', 'target': 1.0, 'subgroup': 'lighter female', 'error_may2017_pct': 9.8, 'error_aug2018_pct': 1.0}, {'company': 'Face++', 'target': 1.0, 'subgroup': 'lighter male', 'error_may2017_pct': 0.8, 'error_aug2018_pct': 0.5}, {'company': 'IBM', 'target': 1.0, 'subgroup': 'darker female', 'error_may2017_pct': 34.7, 'error_aug2018_pct': 16.97}, {'company': 'IBM', 'target': 1.0, 'subgroup': 'darker male', 'error_may2017_pct': 12.0, 'error_aug2018_pct': 0.63}, {'company': 'IBM', 'target': 1.0, 'subgroup': 'lighter female', 'error_may2017_pct': 7.1, 'error_aug2018_pct': 2.37}, {'company': 'IBM', 'target': 1.0, 'subgroup': 'lighter male', 'error_may2017_pct': 0.3, 'error_aug2018_pct': 0.26}, {'company': 'Amazon', 'target': 0.0, 'subgroup': 'darker female', 'error_may2017_pct': None, 'error_aug2018_pct': 31.37}, {'company': 'Amazon', 'target': 0.0, 'subgroup': 'darker male', 'error_may2017_pct': None, 'error_aug2018_pct': 1.26}, {'company': 'Amazon', 'target': 0.0, 'subgroup': 'lighter female', 'error_may2017_pct': None, 'error_aug2018_pct': 7.12}, {'company': 'Amazon', 'target': 0.0, 'subgroup': 'lighter male', 'error_may2017_pct': None, 'error_aug2018_pct': 0.0}, {'company': 'Kairos', 'target': 0.0, 'subgroup': 'darker female', 'error_may2017_pct': None, 'error_aug2018_pct': 22.5}, {'company': 'Kairos', 'target': 0.0, 'subgroup': 'darker male', 'error_may2017_pct': None, 'error_aug2018_pct': 1.3}, {'company': 'Kairos', 'target': 0.0, 'subgroup': 'lighter female', 'error_may2017_pct': None, 'error_aug2018_pct': 6.4}, {'company': 'Kairos', 'target': 0.0, 'subgroup': 'lighter male', 'error_may2017_pct': None, 'error_aug2018_pct': 0.0})
    comps = ['IBM', 'Face++', 'Microsoft', 'Amazon', 'Kairos']
    df = {r['company']: r for r in rows if r['subgroup'] == 'darker female'}
    for k in (1, 2):
        fig, ax = plt.subplots(figsize=(4.2, 2.9))
        for i, co in enumerate(comps):
            y = len(comps) - 1 - i
            a = df[co]['error_may2017_pct']
            bb = float(df[co]['error_aug2018_pct'])
            if a:
                a = float(a)
                if k == 2:
                    ax.plot([bb, a], [y, y], color=C['grid'], lw=3, zorder=1)
                ax.plot([a], [y], 'o', ms=11, color=RED, zorder=3)
                ax.text(a + 1.4, y, f'{a:.1f}', color=RED, fontsize=13, fontweight='bold', va='center')
            if k == 2:
                ax.plot([bb], [y], 'o', ms=11, color=ETH, zorder=3)
                if a:
                    ax.text(bb, y - 0.45, f'{bb:.1f}', color=ETH, fontsize=13, fontweight='bold', ha='center', va='center')
                else:
                    ax.text(bb + 1.4, y, f'{bb:.1f}', color=ETH, fontsize=13, fontweight='bold', va='center')
            ax.text(-1.5, y, co, ha='right', va='center', fontsize=13.5, fontweight='bold')
        ax.set_ylim(-0.8, len(comps) + 0.7)
        ax.set_xlim(0, 42)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Error on darker-skinned women (%)')
        ax.text(41, 5.25, '2017 audit', color=RED, fontsize=13, fontweight='bold', ha='right')
        ax.text(41, 4.7, '2018 re-audit', color=ETH, fontsize=13, fontweight='bold', ha='right', alpha=1.0 if k == 2 else 0.0)
        ax.grid(axis='x', color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out(f'gs_step{k}.svg'))

def architecture():

    def frame(k):
        b = text(10, 28, 'Request path (inside the serving SLO)', 22, ETH, anchor='start', weight='bold')
        for x, w, t in ((10, 150, 'Request'), (190, 210, 'Access control'), (430, 160, 'Inference'), (740, 150, 'Response')):
            b += box(x, 45, w, 60, ETH_L, ETH, [t], size=24)
        b += path_arrow('M160,75 L188,75', 'blue')
        b += path_arrow('M400,75 L428,75', 'blue')
        b += path_arrow('M590,75 L738,75', 'blue')
        if k >= 2:
            b += path_arrow('M510,105 L510,143', 'green')
            b += box(360, 145, 300, 50, GREEN_L, GREEN, ['Evidence log'], size=24)
            for i in range(8):
                known = i < 5
                dash = '' if known else ' stroke-dasharray="4,3"'
                b += f'''<rect x="{366 + i * 37}" y="208" width="30" height="28" rx="3" fill="{(GREEN if known else 'white')}" fill-opacity="{(0.5 if known else 1)}" stroke="{GREEN}" stroke-width="1.5"{dash}/>\n'''
            b += text(440, 262, 'known', 19, GREEN)
            b += text(602, 262, 'pending', 19, GRAY)
        if k >= 3:
            b += text(10, 440, 'Governance tier (asynchronous)', 22, VIOLET, anchor='start', weight='bold')
            b += box(10, 320, 200, 90, VIOLET_L, VIOLET, ['Parity', 'updates now'], size=22)
            b += box(230, 320, 240, 90, VIOLET_L, VIOLET, ['Error rates', 'when labels land'], size=22)
            b += box(490, 320, 230, 90, 'white', RED, ['Thin group', 'insufficient, not 0'], size=22, color=RED, dash='6,4')
            b += box(745, 320, 150, 90, VIOLET_L, VIOLET, ['Release', 'gate: owner'], size=22)
            b += path_arrow('M365,198 L110,318', 'purple', w=2)
            b += path_arrow('M380,240 L330,318', 'purple', w=2)
            b += path_arrow('M720,365 L743,365', 'purple', w=2)
            b += path_arrow('M820,318 L820,128 L560,128 L560,108', 'purple', w=2, dash='8,5')
            b += text(812, 162, 'approved update', 19, VIOLET, anchor='end')
        write(f'arch_step{k}.svg', b, 900, 452)
    for k in (1, 2, 3):
        frame(k)

def alerts():
    n = int(150.0)
    exp = 7.5
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    cols, rows_ = (25, 6)
    red = int(exp)
    for i in range(n):
        r, c = divmod(i, cols)
        ax.add_patch(plt.Rectangle((c, rows_ - 1 - r), 0.86, 0.86, color=RED if i < red else C['neutral_l']))
    ax.add_patch(plt.Rectangle((red % cols, rows_ - 1 - red // cols), 0.43, 0.86, color=RED))
    ax.set_xlim(-0.2, cols)
    ax.set_ylim(-0.2, rows_)
    ax.set_aspect('equal')
    ax.axis('off')
    sp.save(fig, out('alerts.svg'))

def overhead():
    rows = [('Fairness-aware training', 5.0, 15.0, 'train'), ('DP-SGD', 30.0, 300.0, 'train'), ('Federated learning', 200.0, 500.0, 'train'), ('Adversarial training', 100.0, 1000.0, 'train'), ('Sampled KernelSHAP', 50.0, 200.0, 'serve'), ('Exhaustive SHAP', 5000.0, 100000.0, 'serve')]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for i, (lab, lo, hi, path) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=AMBER if path == 'train' else ETH, lw=11, solid_capstyle='butt')
        ax.text(0.85 * lo, y, lab, ha='right', va='center', fontsize=13, fontweight='bold', color=INK)
    ax.set_xscale('log')
    ax.set_xlim(1, 300000.0)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(['+10%', '+100%', '+1,000%', '+10⁴%', '+10⁵%'])
    ax.text(300000.0, 5.3, 'training path', color=AMBER, fontsize=13, fontweight='bold', ha='right')
    ax.text(300000.0, 1.3, 'serving path', color=ETH, fontsize=13, fontweight='bold', ha='right')
    ax.set_xlabel('Extra work over the unprotected path (log)')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out('overhead.svg'))

def shap_ladder():
    rows = []
    for f, ck in zip([3, 10, 20], ('data', 'compute', 'danger')):
        ms = {3: 0.008, 10: 1.024, 20: 1048.576}[f]
        sub = {3: 8, 10: 1024, 20: 1048576}[f]
        lab = f'{ms:.3f} ms' if ms < 0.1 else f'{ms:.2f} ms' if ms < 1000 else f'{ms / 1000:.2f} s'
        rows.append((f'{f} features: {sub:,.0f}', ms, ck, lab))
    sp.hbars(out('shap.svg'), rows, 'ms', log=True, size=(5.0, 1.8), xlabel='Exact SHAP time at 1 µs per subset (ms, log)', vlines=[(100.0, '100 ms budget', 'danger')])

def dp_accuracy():
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    me, ma = ([8.0, 2.0, 0.5], [97.0, 95.0, 90.0])
    ce, ca = ([8.0, 4.0, 2.0], [73.0, 70.0, 67.0])
    ax.plot(me, ma, 'o-', color=ETH, lw=3, ms=9)
    ax.plot(ce, ca, 'o-', color=AMBER, lw=3, ms=9)
    ax.axhline(98.3, color=ETH, lw=1.5, ls='--')
    ax.axhline(80.0, color=AMBER, lw=1.5, ls='--')
    ax.text(0.37, 99.3, 'MNIST, no privacy: 98.3%', color=ETH, fontsize=12.5, ha='right')
    ax.text(0.37, 81.3, 'CIFAR-10, no privacy: about 80%', color=AMBER, fontsize=12.5, ha='right')
    for e, a in zip(me, ma):
        ax.text(e, a - 5.0, f'{a:.0f}%', color=ETH, fontsize=13, ha='center', fontweight='bold')
    for e, a in zip(ce, ca):
        ax.text(e, a + 2.0, f'{a:.0f}%', color=AMBER, fontsize=13, ha='center', fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(12, 0.35)
    ax.set_xticks([8, 4, 2, 1, 0.5])
    ax.set_xticklabels(['8', '4', '2', '1', '0.5'])
    ax.minorticks_off()
    ax.set_ylim(60, 104)
    ax.set_xlabel('Privacy budget $\\varepsilon$ (smaller is stronger)')
    ax.set_ylabel('Test accuracy (%)')
    ax.grid(color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out('dp_accuracy.svg'))

def privacy_boundary():

    def frame(k):
        b = ''
        for i in range(3):
            y = 20 + i * 105
            b += box(10, y, 160, 88, ETH_L, ETH, [f'Phone {i + 1}', 'raw data'], size=22)
            b += path_arrow(f'M170,{y + 44} L226,{180 + (i - 1) * 25}', 'blue', w=2)
        b += text(10, 350, '1: data stays local', 22, ETH, anchor='start', weight='bold')
        if k >= 2:
            b += box(228, 105, 220, 150, AMBER_L, AMBER, ['Secure', 'aggregation', 'hides each', 'update'], size=22)
            b += path_arrow('M448,180 L486,180', 'amber')
            b += box(488, 105, 200, 150, VIOLET_L, VIOLET, ['Curator', 'adds DP', 'noise, counts ε'], size=22)
            b += path_arrow('M688,180 L738,180', 'purple')
            b += box(740, 130, 150, 100, GREEN_L, GREEN, ['Released', 'model'], size=22)
            b += f'<line x1="714" y1="20" x2="714" y2="290" stroke="{GRAY}" stroke-width="2" stroke-dasharray="6,5"/>\n'
            b += text(720, 40, 'public', 20, GRAY, anchor='start')
            b += text(338, 282, '2: updates hidden', 21, AMBER, weight='bold')
            b += text(588, 282, '3: model bounded', 21, VIOLET, weight='bold')
        if k >= 3:
            b += box(330, 320, 330, 80, RED_L, RED, ['Gateway log: raw prompt', 'no DP claim covers it'], size=22, color=RED)
            b += path_arrow('M815,232 C815,360 750,362 664,360', 'red', dash='7,5')
        write(f'privacy_step{k}.svg', b, 900, 410)
    for k in (1, 2, 3):
        frame(k)

def unlearn():
    rows = [('Full retraining', 4600000.0, 'danger', '\\$4.6M, 36 days'), ('SISA, 1 request', 46000.0, 'data', '\\$46,000, 8.6 h'), ('SISA, 1,000/day', 4599801.412, 'compute', 'about \\$4.6M a day')]
    for k, step in ((1, 2), (2, 3)):
        sp.hbars(out(f'unlearn_step{k}.svg'), rows, '$', log=True, step=step, size=(4.4, 1.6), xlabel='Cost to forget, 175B model ($, log)')

def genai():
    b = box(330, 120, 240, 110, ETH_L, ETH, ['Model weights', 'aligned by RLHF'], size=24)
    items = [(10, 10, 'System prompt', 'versioned per product'), (10, 250, 'Retrieval index', 'untrusted context'), (630, 10, 'Reward model', 'whose raters?'), (630, 250, 'Output filter', 'runtime policy check')]
    for x, y, t1, t2 in items:
        b += box(x, y, 260, 90, VIOLET_L, VIOLET, [t1, t2], size=23)
    b += path_arrow('M270,70 L328,140', 'purple', w=2.5)
    b += path_arrow('M270,290 L328,212', 'purple', w=2.5)
    b += path_arrow('M630,70 L572,140', 'purple', w=2.5)
    b += path_arrow('M572,212 L628,290', 'purple', w=2.5)
    write('genai.svg', b, 900, 350)

def contest():

    def frame(k):
        b = ''
        st = [('Decision provenance', 'inputs, model, policy version', ETH_L, ETH), ('Explanation', 'reasons for this audience', FRAME_L, FRAME), ('Appeal routing', 'reviewer with authority', FRAME_L, FRAME), ('Outcome tracking', 'upheld or overturned', FRAME_L, FRAME)]
        for i, (t1, t2, f, s) in enumerate(st):
            y = 10 + i * 108
            if k == 1 and i > 0:
                continue
            b += box(100, y, 390, 84, f, s, [t1, t2], size=24)
            if i < 3 and k == 2:
                b += path_arrow(f'M295,{y + 84} L295,{y + 106}', 'gray', w=2)
        if k == 2:
            b += path_arrow('M100,376 C50,376 50,52 96,52', 'green', dash='8,5')
            b += f'<text x="22" y="214" text-anchor="middle" font-size="22" font-weight="bold" fill="{GREEN}" transform="rotate(-90 22 214)">retrain</text>\n'
        write(f'contest_step{k}.svg', b, 500, 440)
    frame(1)
    frame(2)

def loop():

    def frame(k):
        b = ''
        nodes = [(315, 15, ['Biased training data'], False), (620, 165, ['Biased predictions'], False), (315, 315, ['Disparate outcomes'], True), (10, 158, ['Reinforced bias', 'in new data'], False)]
        for x, y, t, red in nodes:
            h = 84 if len(t) > 1 else 70
            b += box(x, y, 270, h, RED_L if red else ETH_L, RED if red else ETH, t, size=24, color=RED if red else INK, bold_first=True)
        b += path_arrow('M585,50 C660,55 740,90 755,163', 'blue')
        b += path_arrow('M755,235 C740,300 660,345 587,350', 'blue')
        b += path_arrow('M315,350 C220,345 150,300 145,244', 'blue')
        b += path_arrow('M145,158 C150,90 230,55 313,50', 'blue')
        b += text(450, 195, 'retrain on', 22, GRAY, italic=True)
        b += text(450, 222, 'its own outcomes', 22, GRAY, italic=True)
        if k == 2:
            for x, y in ((196, 84), (702, 86), (702, 316)):
                b += f'<circle cx="{x}" cy="{y}" r="12" fill="{GREEN}"/>\n'
            b += box(10, 10, 250, 46, GREEN_L, GREEN, ['Preprocess: reweight'], size=22, bold_first=False)
            b += box(615, 10, 280, 46, GREEN_L, GREEN, ['In-process: constraints'], size=22, bold_first=False)
            b += box(615, 340, 280, 46, GREEN_L, GREEN, ['Post-process: thresholds'], size=22, bold_first=False)
        write(f'loop_step{k}.svg', b, 900, 395)
    frame(1)
    frame(2)

def autobias():
    best = 96.5
    rows = [('Radiologist alone', 92.0, 'neutral', '92%'), ('AI alone', 95.0, 'memory', '95%'), ('Team, best case', best, 'compute', f'at most {best:.1f}%')]
    sp.hbars(out('autobias.svg'), rows, '%', log=False, size=(4.4, 1.7), xlabel='Tumors found among patients with tumors (%)')

def waymo():
    import datetime as dt
    import matplotlib.dates as mdates
    rows = ({'data_through': '2024-06', 'cumulative_ro_miles_millions': 22.199, 'method': 'sum of Ops Depot rows', 'source_file': 'm_202406.csv'}, {'data_through': '2024-07', 'cumulative_ro_miles_millions': 25.308, 'method': 'sum of Ops Depot rows', 'source_file': 'm_202407.csv'}, {'data_through': '2024-09', 'cumulative_ro_miles_millions': 33.103, 'method': 'sum of Ops Depot rows', 'source_file': 'm_202409.csv'}, {'data_through': '2024-12', 'cumulative_ro_miles_millions': 50.083, 'method': 'sum of Ops Depot rows', 'source_file': 'm_202412.csv'}, {'data_through': '2025-03', 'cumulative_ro_miles_millions': 71.62, 'method': 'sum of Ops Depot rows', 'source_file': 'm_202503.csv'}, {'data_through': '2025-06', 'cumulative_ro_miles_millions': 95.965, 'method': 'sum of Ops Depot rows', 'source_file': 'm_202506.csv'}, {'data_through': '2025-09', 'cumulative_ro_miles_millions': 127.158, 'method': 'sum of Ops Depot rows', 'source_file': 'm_202509.csv'}, {'data_through': '2025-12', 'cumulative_ro_miles_millions': 170.712, 'method': 'All Locations row', 'source_file': 'm_202512.csv'}, {'data_through': '2026-03', 'cumulative_ro_miles_millions': 220.613, 'method': 'All Locations row', 'source_file': 'm_202603.csv'}, {'data_through': '2026-06', 'cumulative_ro_miles_millions': 271.329, 'method': 'All Locations row', 'source_file': 'm_202606.csv'})
    xs = [dt.date(int(r['data_through'][:4]), int(r['data_through'][5:]), 1) for r in rows]
    ys = [float(r['cumulative_ro_miles_millions']) for r in rows]
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    ax.plot(xs, ys, 'o-', color=ETH, lw=3, ms=7)
    ax.annotate(f'{ys[0]:.0f}M', (xs[0], ys[0]), xytext=(8, 12), textcoords='offset points', ha='left', fontsize=13, fontweight='bold', color=ETH)
    ax.annotate(f'{ys[-1]:.0f}M', (xs[-1], ys[-1]), xytext=(-10, 0), textcoords='offset points', ha='right', va='center', fontsize=13, fontweight='bold', color=ETH)
    ax.text(xs[1], 200, '12.2× in two years', fontsize=13, fontweight='bold', color=INK)
    ax.set_ylabel('Rider-only miles (M)')
    ax.set_ylim(0, 300)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('Jun %Y'))
    ax.grid(color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out('waymo_miles.svg'))

def main():
    ofqual_flows()
    ofqual_candidates()
    hook()
    assurance()
    lifecycle()
    loan_metrics()
    compas()
    dual_thresholds()
    obermeyer()
    ztest()
    gender_grid()
    gender_dumbbell()
    architecture()
    alerts()
    overhead()
    shap_ladder()
    dp_accuracy()
    privacy_boundary()
    unlearn()
    genai()
    contest()
    loop()
    autobias()
    waymo()
if __name__ == '__main__':
    main()
