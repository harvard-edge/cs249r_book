#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 8 (Agent Sandboxes, Volume III).

Every number comes from calcs.py, which reads MLSysIM (the chapter's
scenarios and the cited results) and the deck's own runs in data/; labels use
val() so a figure prints the digits the slide prints. Diagrams are SVG,
plots are matplotlib SVG; the slides Makefile turns both into PDF.

Legibility rule (slides/visuals.md §2): every figure is drawn at the shape
of its slot so it is never shrunk.
  - Full-width diagrams fill a slot about 14.3 x 5 cm, so the canvas is
    900 x 300-340 units (one unit is about 0.45 pt on the slide). Text is at
    least 22 units (about 10 pt, the \\footnotesize floor) and boxes carry
    two or three words.
  - Column plots fill a slot about 7.5 x 4.5 cm, so they are drawn 5.4 in
    wide with 17-18 pt type (about 9.5-10 pt on the slide).
The volume accent is amethyst (#581C87); figures use the book's palette.

Run from this folder: python3 make_figs.py
"""
from mlsysim.core.series_loader import load_series
import mlsysim
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from mlsysim.physics import calc_poisson_tail
plt = sp.plt
C = sp.C
out = Path(__file__).parent
AME = '#581C87'
AME_L = '#E7DAF2'
INK = '#1A1A2E'
GRAY = '#BBBBBB'
FONT = 'Helvetica, Arial, sans-serif'
T = 23
TB = 25

class SVG:

    def __init__(self, w=900, h=320):
        self.w, self.h, self.p = (w, h, [])

    def rect(self, x, y, w, h, fill='none', stroke=INK, sw=2.5, rx=10):
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=3.5, arrow=None):
        m = f' marker-end="url(#{arrow})"' if arrow else ''
        self.p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{m}/>')

    def text(self, x, y, s, size=T, fill=INK, weight='normal', anchor='middle', mono=False):
        s = str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        fam = 'Courier, monospace' if mono else FONT
        self.p.append(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>')

    def save(self, name):
        marks = ''.join((f'<marker id="{k}" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L6,3 L0,6 Z" fill="{c}"/></marker>' for k, c in (('ak', INK), ('ar', C['danger']), ('ag', C['data']), ('aa', AME))))
        head = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" font-family="{FONT}"><defs>{marks}</defs><rect width="{self.w}" height="{self.h}" fill="white"/>'
        (out / f'{name}.svg').write_text(head + ''.join(self.p) + '</svg>')

def boundaries_steps():
    for step in (1, 2, 3):
        g = SVG(900, 300)
        g.rect(8, 20, 200, 260, fill='#F7F5FA', stroke=AME)
        g.text(108, 60, 'guest', TB, AME, 'bold')
        g.text(108, 128, 'run_shell', T, C['danger'], 'bold', mono=True)
        g.text(108, 228, 'submit_report', 21, C['data'], 'bold', mono=True)
        for (l1, l2), gx in ((('tool', 'admit'), 260), (('file', 'policy'), 450), (('egress', 'policy'), 640)):
            g.rect(gx, 20, 120, 260, fill=AME_L, stroke=AME, sw=2)
            g.text(gx + 60, 165, l1, T, AME, 'bold')
            g.text(gx + 60, 192, l2, T, AME, 'bold')
        g.rect(778, 160, 116, 110, fill='white', stroke=C['data'])
        g.text(836, 208, 'approved', 22, C['data'], 'bold')
        g.text(836, 236, 'service', 22, C['data'], 'bold')
        g.line(208, 120, 256, 120, C['danger'], arrow='ar')
        if step >= 2:
            g.line(380, 120, 446, 120, C['danger'], arrow='ar')
            g.text(510, 108, '✗', 46, C['danger'], 'bold')
            g.text(510, 70, 'blocked', T, C['danger'], 'bold')
        if step >= 3:
            for x1, x2 in ((208, 256), (380, 446), (570, 636)):
                g.line(x1, 225, x2, 225, C['data'], arrow='ag')
            g.line(760, 225, 774, 225, C['data'], arrow='ag')
        g.save(f'boundaries_step{step}')

def reach_steps():
    items = ['module globals', 'API_KEY', '251 classes', 'open files']
    for step in (1, 2):
        g = SVG(900, 320)
        g.rect(8, 8, 884, 304, fill='#F7F5FA', stroke=C['neutral'], sw=2)
        g.text(450, 46, 'one supervisor process', T, C['neutral'], 'bold')
        g.rect(40, 90, 270, 160, fill='white', stroke=AME)
        g.text(175, 132, 'eval scope', TB, AME, 'bold')
        g.text(175, 175, 'no builtins', T)
        g.text(175, 212, 'one helper: total', T)
        for i, s in enumerate(items):
            y = 70 + i * 60
            g.rect(600, y, 270, 48, fill='white', stroke=C['danger'], sw=2)
            g.text(735, y + 32, s, T, C['danger'], 'bold')
        if step >= 2:
            g.line(310, 130, 594, 96, C['danger'], arrow='ar')
            g.text(455, 88, '.__globals__', T, C['danger'], 'bold', mono=True)
            g.line(310, 215, 594, 213, C['danger'], arrow='ar')
            g.text(455, 262, 'subclass walk', T, C['danger'], 'bold')
        g.save(f'reach_step{step}')

def admit_steps():
    rows = [('authentic', 'protected, bound to holder'), ('op in rights', 'read, not write'), ('target in objects', 'repo/, not ~/.aws'), ('condition holds', 'this task only'), ('time < expiry', 'before 300 s')]
    for step in range(1, len(rows) + 2):
        g = SVG(900, 340)
        for i, (cond, gloss) in enumerate(rows):
            y = 8 + i * 56
            on = i < step
            col = AME if on else GRAY
            g.rect(60, y, 340, 46, fill=AME_L if on else 'white', stroke=col, sw=2)
            g.text(80, y + 31, cond, T, col, 'bold', 'start', mono=True)
            g.text(440, y + 31, gloss, T, INK if on else GRAY, anchor='start')
        if step > len(rows):
            g.rect(60, 290, 780, 44, fill='white', stroke=C['data'], sw=2.5)
            g.text(450, 320, 'default reach is empty', T, C['data'], 'bold')
        g.save(f'admit_step{step}')

def attenuate_steps():
    for step in (1, 2, 3):
        g = SVG(900, 320)
        g.rect(270, 8, 360, 112, fill=AME_L, stroke=AME)
        g.text(450, 44, 'edit grant (parent)', TB, AME, 'bold')
        g.text(450, 76, 'read, write  repo/', T, mono=True)
        g.text(450, 106, 'expires at 300 s', T, C['neutral'])
        if step >= 2:
            g.line(370, 120, 220, 182, AME, arrow='aa')
            g.rect(20, 186, 400, 128, fill='white', stroke=C['danger'])
            g.text(220, 222, 'test grant', TB, C['danger'], 'bold')
            g.text(220, 256, 'read, execute  repo/', T, mono=True)
            g.text(220, 292, 'rejected: adds execute', T, C['danger'], 'bold')
        if step >= 3:
            g.line(530, 120, 680, 182, AME, arrow='aa')
            g.rect(480, 186, 400, 128, fill='white', stroke=C['data'])
            g.text(680, 222, 'scan grant', TB, C['data'], 'bold')
            g.text(680, 256, 'read  src/pool.py', T, mono=True)
            g.text(680, 292, 'valid: narrower', T, C['data'], 'bold')
        g.save(f'attenuate_step{step}')

def revoke_steps():
    rows = [(70, 'new open for writing', 'refused: expired', C['data']), (170, 'write on an open handle', 'still succeeds', C['danger']), (270, 'close handle, stop worker', 'revocation confirmed', C['data'])]
    for step in (1, 2, 3):
        g = SVG(900, 320)
        g.rect(10, 34, 180, 272, fill=AME_L, stroke=AME)
        g.text(100, 92, 'edit grant', TB, AME, 'bold')
        g.text(100, 134, 'expires', T, INK)
        g.text(100, 162, 'at 300 s', T, INK)
        g.text(100, 222, 'read, write', T, mono=True)
        g.text(100, 252, 'repo/', T, mono=True)
        for k, (y, act, res, col) in enumerate(rows, start=1):
            if step < k:
                continue
            g.line(190, y, 206, y, AME, sw=3)
            g.rect(210, y - 36, 350, 72, fill='white', stroke=GRAY)
            g.text(385, y + 8, act, T)
            g.line(560, y, 584, y, col, arrow='ag' if col == C['data'] else 'ar')
            g.text(596, y + 8, res, T, col, 'bold', anchor='start')
        g.save(f'revoke_step{step}')

def envelopes_steps():
    cols = [(('Container', ''), ('host kernel,', 'after a filter'), C['memory'], 290), (('User-space', 'kernel'), ('emulated in', 'user space'), C['network'], 240), (('MicroVM', ''), ('guest kernel,', 'natively'), C['compute'], 240), (('Bytecode', ''), ('no syscall;', 'host imports'), C['data'], None)]
    for step in range(1, 5):
        g = SVG(900, 320)
        g.line(10, 292, 890, 292, INK, 3)
        g.text(888, 316, 'host kernel', T, C['neutral'], 'bold', 'end')
        for i, ((n1, n2), (w1, w2), col, depth) in enumerate(cols):
            x = 12 + i * 222
            on = i < step
            c = col if on else GRAY
            t = INK if on else GRAY
            g.rect(x, 8, 210, 200, fill='white', stroke=c, sw=3 if on else 2)
            g.text(x + 105, 46, n1, TB, c, 'bold')
            if n2:
                g.text(x + 105, 76, n2, TB, c, 'bold')
            g.text(x + 105, 136, w1, T, t)
            g.text(x + 105, 166, w2, T, t)
            if on and depth:
                g.line(x + 105, 208, x + 105, depth - 4, c, arrow='ak')
        g.save(f'envelopes_step{step}')

def egress_steps():
    for step in (1, 2, 3):
        g = SVG(900, 330)
        g.rect(8, 20, 170, 250, fill='#F7F5FA', stroke=AME)
        g.text(93, 60, 'guest', TB, AME, 'bold')
        g.rect(280, 20, 240, 104, fill=AME_L, stroke=AME, sw=2)
        g.text(400, 62, 'packet filter', TB, AME, 'bold')
        g.text(400, 96, 'IP and port', T)
        g.rect(280, 166, 240, 104, fill=AME_L, stroke=AME, sw=2)
        g.text(400, 208, 'mediator', TB, AME, 'bold')
        g.text(400, 242, 'operation, data', T)
        g.rect(720, 100, 172, 100, fill='white', stroke=C['data'])
        g.text(806, 142, 'approved', T, C['data'], 'bold')
        g.text(806, 172, 'registry', T, C['data'], 'bold')
        g.line(178, 72, 276, 72, C['danger'], arrow='ar')
        g.text(610, 56, 'raw socket', T, C['danger'], 'bold')
        g.text(610, 86, 'dropped', T, C['danger'], 'bold')
        if step >= 2:
            g.line(178, 218, 276, 218, C['data'], arrow='ag')
            g.line(520, 218, 716, 170, C['data'], arrow='ag')
        if step >= 3:
            g.text(450, 316, 'DNS lookups also leak, so resolve an exact allowlist', T, C['danger'], 'bold')
        g.save(f'egress_step{step}')

def reset_steps():
    doms = [('Execution, memory', 'destroy or restore'), ('Workspace, storage', 'keep outputs, drop the rest'), ('Network, broker', 'retire sessions, identities'), ('Host, devices', 'scrub queues, device memory')]
    for step in range(1, len(doms) + 2):
        g = SVG(900, 330)
        for i, (d, a) in enumerate(doms):
            y = 8 + i * 64
            on = i < step
            col = AME if on else GRAY
            g.rect(20, y, 330, 52, fill=AME_L if on else 'white', stroke=col, sw=2)
            g.text(40, y + 35, d, T, col, 'bold', 'start')
            g.text(380, y + 35, a, T, INK if on else GRAY, anchor='start')
        if step > len(doms):
            g.rect(20, 276, 860, 48, fill='white', stroke=C['danger'], sw=2.5)
            g.text(450, 308, 'one leftover fails the reset; destroy and rebuild', T, C['danger'], 'bold')
        g.save(f'reset_step{step}')
COL = {'font.size': 17, 'axes.labelsize': 17, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 15}

def cbars(path, rows, xlabel, log=False, height=2.6, xmax=None):
    """Horizontal bars for a column slot: bold labels left, values at the
    bar end, no legend. rows: (label, value, color_key, value_text)."""
    with plt.rc_context(COL):
        fig, ax = plt.subplots(figsize=(5.4, height))
        n = len(rows)
        vals = [r[1] for r in rows]
        for i, (lab, v, ck, txt) in enumerate(rows):
            y = n - 1 - i
            ax.barh(y, v, color=C[ck], height=0.6)
            ax.text(v * (1.12 if log else 1.0) + (0 if log else max(vals) * 0.02), y, txt, va='center', fontsize=17, fontweight='bold', color=C[ck])
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=17, fontweight='bold')
        ax.tick_params(axis='y', length=0)
        if log:
            ax.set_xscale('log')
            ax.set_xlim(min(vals) / 2, max(vals) * 30)
        else:
            ax.set_xlim(0, xmax or max(vals) * 1.45)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        sp.save(fig, str(path))

def syscall_plots():
    sup = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
    cbars(out / 'tax.svg', [('Container', 0.15, 'memory', '0.15 s'), ('MicroVM', 0.1, 'compute', '0.10 s'), ('User-space', 3.6, 'network', '3.6 s')], 'Added s per 10⁶ calls', height=3.0, xmax=5.0)
    cbars(out / 'gvisor.svg', [('native', 0.22, 'neutral', '0.22'), ('runc', 0.29, 'memory', '0.29'), ('Sentry', 0.63, 'network', '0.63'), ('via host', 5.96, 'network', '5.96'), ('via gofer', 45.5, 'danger', '45.5')], 'µs per call (log)', log=True, height=3.0)

def setup_plot():
    setup = 15.0
    cap = 10.0
    with plt.rc_context(COL):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        work = np.linspace(0.5, 200, 400)
        ax.plot(work, setup / (setup + work) * 100, color=AME, lw=3.5)
        ax.axhline(cap, color=C['danger'], lw=2, ls='--')
        ax.text(198, cap + 4, '10% cap', color=C['danger'], fontsize=17, fontweight='bold', ha='right')
        ax.plot([3.0], [83.3], 'o', ms=12, color=C['danger'], mec='white', mew=1.5)
        ax.text(14, 80, '3 ms work: 83.3%', fontsize=17, fontweight='bold', color=C['danger'], va='center')
        ax.plot([135.0], [cap], 'o', ms=12, color=AME, mec='white', mew=1.5)
        ax.text(135.0, 30, '135 ms', fontsize=17, fontweight='bold', color=AME, ha='center')
        ax.set_xlabel('Useful work per call (ms)')
        ax.set_ylabel('Setup share (%)')
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 100)
        ax.grid(color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, str(out / 'setup_share.svg'))

def cow_plots():
    cbars(out / 'provision.svg', [('Full copy', 76.8, 'danger', '76.8 GB'), ('Copy-on-write', 1.28, 'data', '1.28 GB')], 'GB moved, 32 workspaces (log)', log=True, height=2.4)
    cbars(out / 'clone.svg', [('Full copy', 938.0, 'danger', '938 ms'), ('Clone', 90.0, 'data', '90 ms'), ('1-byte write', 0.9, 'compute', '0.9 ms')], 'ms, 2.4 GB base (log)', log=True, height=2.4)

def pool_plots():
    lam = 0.002
    target, reserve = (0.001, 2)
    restore = 15.0

    def tail(x_ms):
        return calc_poisson_tail(lam * x_ms, reserve)
    with plt.rc_context(COL):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        r = np.linspace(2, 80, 400)
        ax.plot(r, [tail(x) for x in r], color=AME, lw=3.5)
        ax.axhline(target, color=C['danger'], lw=2, ls='--')
        ax.text(78, target * 1.6, '0.1% target', color=C['danger'], fontsize=17, fontweight='bold', ha='right')
        ax.plot([restore], [tail(restore)], 'o', ms=12, color=C['data'], mec='white')
        ax.annotate('15 ms', (restore, tail(restore)), xytext=(32, 1.2e-05), color=C['data'], fontsize=17, fontweight='bold', arrowprops=dict(arrowstyle='->', color=C['data'], lw=1.5))
        ax.axvline(22.70100888, color=C['neutral'], lw=1.5, ls=':')
        ax.text(26, 0.03, 'fails past 22 ms', color=C['neutral'], fontsize=16)
        ax.set_yscale('log')
        ax.set_xlabel('Restore time (ms)')
        ax.set_ylabel(f'Miss, {reserve} standby')
        ax.set_xlim(0, 80)
        ax.set_ylim(1e-06, 1)
        ax.grid(color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, str(out / 'pool_miss.svg'))
    cbars(out / 'pool_mem.svg', [('Private', 322.1, 'danger', '322.1 GB'), ('Shared', 39.0, 'data', '39.0 GB')], 'GB for 600 leases', height=2.3)

def cybench_plot():
    rows = [('2024-02-15', 0.075, 'Cybench leaderboard'), ('2024-02-29', 0.1, 'Cybench leaderboard'), ('2024-04-17', 0.075, 'Cybench leaderboard'), ('2024-04-18', 0.05, 'Cybench leaderboard'), ('2024-06-20', 0.175, 'Cybench leaderboard'), ('2024-07-23', 0.075, 'Cybench leaderboard'), ('2024-09-12', 0.1, 'Cybench leaderboard'), ('2024-09-12', 0.1, 'Cybench leaderboard'), ('2024-11-20', 0.125, 'Cybench leaderboard'), ('2025-01-31', 0.225, 'Cybench leaderboard'), ('2025-02-24', 0.2, 'Cybench leaderboard'), ('2025-02-27', 0.175, 'Cybench leaderboard'), ('2025-05-22', 0.35, 'Cybench leaderboard'), ('2025-05-22', 0.38, 'Cybench leaderboard'), ('2025-07-09', 0.43, 'Grok 4 Model Card'), ('2025-08-05', 0.42, 'Cybench leaderboard'), ('2025-09-19', 0.3, 'Grok 4 Fast Model Card'), ('2025-09-29', 0.55, 'Cybench leaderboard'), ('2025-09-29', 0.6, 'Opus 4.5 System Card'), ('2025-11-17', 0.39, 'Grok 4.1 Model Card'), ('2025-11-24', 0.82, 'Opus 4.5 System Card'), ('2026-02-05', 0.93, 'Opus 4.6 System Card')]
    with plt.rc_context(COL):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        yrs = [int(d[:4]) + (int(d[5:7]) - 1) / 12 + (int(d[8:10]) - 1) / 365 for d, v, s in rows]
        for yy, (d, v, s) in zip(yrs, rows):
            board = s == 'Cybench leaderboard'
            ax.scatter([yy], [v * 100], s=70, zorder=4, color=C['memory'] if board else 'none', edgecolors=C['memory'] if board else C['danger'], linewidths=2)
        best, bx, by = (0, [], [])
        for yy, v in sorted(zip(yrs, [r[1] for r in rows])):
            best = max(best, v)
            bx.append(yy)
            by.append(best * 100)
        ax.step(bx, by, where='post', color=C['neutral'], lw=2, ls='--')
        ax.text(2024.5, 27, '17.5%', fontsize=18, color=C['memory'], fontweight='bold', ha='center')
        ax.text(2025.95, 93.0, '93%', fontsize=18, color=C['danger'], fontweight='bold', va='center', ha='right')
        ax.set_ylabel('Solved unhinted (%)')
        ax.set_ylim(0, 105)
        ax.set_xlim(2023.95, 2026.25)
        ax.set_xticks([2024, 2025, 2026])
        ax.set_xticklabels(['2024', '2025', '2026'])
        ax.grid(color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, str(out / 'cybench.svg'))

def firecracker_plot():
    s = np.array([[104.0, 0.0], [104.0, 0.002], [105.0, 0.006], [106.0, 0.022], [107.0, 0.034], [108.0, 0.056], [109.0, 0.234], [110.0, 0.55], [111.0, 0.67], [112.0, 0.768], [113.0, 0.93], [114.0, 0.986], [115.0, 0.99], [116.0, 0.992], [119.0, 0.994], [120.0, 0.998], [121.0, 1.0]])
    p = np.array([[101.0, 0.0], [101.0, 0.001], [102.0, 0.002], [104.0, 0.008], [105.0, 0.018], [106.0, 0.025], [107.0, 0.041], [108.0, 0.069], [109.0, 0.099], [110.0, 0.127], [111.0, 0.156], [112.0, 0.186], [113.0, 0.223], [114.0, 0.26], [115.0, 0.301], [116.0, 0.345], [117.0, 0.391], [118.0, 0.436], [119.0, 0.474], [120.0, 0.516], [121.0, 0.568], [122.0, 0.606], [123.0, 0.643], [124.0, 0.684], [125.0, 0.719], [126.0, 0.747], [127.0, 0.761], [128.0, 0.78], [129.0, 0.8], [130.0, 0.82], [131.0, 0.831], [132.0, 0.847], [133.0, 0.865], [134.0, 0.873], [135.0, 0.884], [136.0, 0.897], [137.0, 0.906], [138.0, 0.916], [139.0, 0.923], [140.0, 0.933], [141.0, 0.941], [142.0, 0.946], [143.0, 0.951], [144.0, 0.957], [145.0, 0.965], [146.0, 0.97], [147.0, 0.974], [148.0, 0.975], [149.0, 0.98], [150.0, 0.982], [151.0, 0.983], [152.0, 0.988], [153.0, 0.992], [154.0, 0.993], [155.0, 0.995], [158.0, 0.997], [159.0, 0.998], [163.0, 0.999], [169.0, 1.0]])
    with plt.rc_context(COL):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(s[:, 0], s[:, 1] * 100, color=C['data'], lw=3.5)
        ax.plot(p[:, 0], p[:, 1] * 100, color=C['network'], lw=3.5)
        ax.annotate('one at a time', (111.5, 45), xytext=(127, 10), color=C['data'], fontsize=17, fontweight='bold', arrowprops=dict(arrowstyle='->', color=C['data'], lw=1.5))
        ax.text(126, 40, '100 at once', color=C['network'], fontsize=17, fontweight='bold')
        ax.axvline(153.0, color=C['neutral'], lw=1.5, ls='--')
        ax.text(153.0 - 1, 75, 'p99 153', color=C['neutral'], fontsize=16, ha='right')
        ax.set_xlabel('Boot time (ms)')
        ax.set_ylabel('Booted (%)')
        ax.set_xlim(100, 165)
        ax.set_ylim(0, 103)
        ax.grid(color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, str(out / 'firecracker.svg'))

def crops():
    from PIL import Image
    Image.open(out / 'tracebit_shown.png').crop((0, 8, 700, 56)).save(out / 'tb_saw.png')
    Image.open(out / 'tracebit_ran.png').crop((20, 82, 1590, 132)).save(out / 'tb_ran.png')
    Image.open(out / 'tracebit_fixed.png').crop((20, 335, 680, 388)).save(out / 'tb_fix.png')
if __name__ == '__main__':
    boundaries_steps()
    reach_steps()
    admit_steps()
    attenuate_steps()
    revoke_steps()
    envelopes_steps()
    egress_steps()
    reset_steps()
    syscall_plots()
    setup_plot()
    cow_plots()
    pool_plots()
    cybench_plot()
    firecracker_plot()
    crops()
    print('figures written to', out)
