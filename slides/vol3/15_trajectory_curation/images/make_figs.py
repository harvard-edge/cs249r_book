#!/usr/bin/env python3
"""Lecture 15 (Trajectory Curation): every figure the deck draws.

Run: python3 make_figs.py   (writes SVGs here; `make deck V=3 D=15_trajectory_curation`
converts them to PDF). Plots use the shared slideplot helper and are drawn at
the size of their slot; diagrams are SVG at 900 units wide. Every value comes
from calcs.py, which reads MLSysIM and data/corpus_stats.json; this script
imports it and prints the recorded values (val, num), so figures and slides
agree.
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
from slideplot import plt
STATS = json.loads((HERE / 'data' / 'corpus_stats.json').read_text())
BLUE, BLUE_L = ('#0284c7', '#e0f2fe')
GREEN, GREEN_L = ('#16a34a', '#dcfce7')
AMBER, AMBER_L = ('#d97706', '#fef3c7')
PURPLE, PURPLE_L = ('#7c3aed', '#f3e8ff')
RED, RED_L = ('#dc2626', '#fee2e2')
GRAY, GRAY_L = ('#64748b', '#f8fafc')
LINE = '#94a3b8'
INK = '#0f172a'
FONT = 'Helvetica, Arial, sans-serif'

def svg(path, w, h, body):
    marks = ''.join((f'<marker id="a{n}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="{c}"/></marker>' for n, c in [('g', GRAY), ('p', PURPLE), ('a', AMBER), ('r', RED), ('b', BLUE), ('n', GREEN)]))
    (HERE / path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">\n<defs>{marks}</defs>\n<rect width="{w}" height="{h}" fill="white"/>\n' + body + '\n</svg>\n')

def g(body, o=1.0):
    return f'<g opacity="{o}">{body}</g>'

def box(x, y, w, h, fill, stroke, rx=8, dash=False, sw=2.5):
    d = ' stroke-dasharray="8 6"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

def text(x, y, s, size=20, color=INK, anchor='middle', weight='normal', family=None):
    f = f' font-family="{family}"' if family else ''
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"{f}>{s}</text>'

def arrow(x1, y1, x2, y2, m='g', color=GRAY, dash=False, sw=2.5):
    d = ' stroke-dasharray="7 5"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#a{m})"/>'

def card(x, y, w, h, title, sub, fill, stroke, ts=21, ss=18, dash=False, tcolor=INK):
    out = box(x, y, w, h, fill, stroke, dash=dash)
    if sub:
        out += text(x + w / 2, y + h / 2 - 4, title, ts, tcolor, weight='bold')
        out += text(x + w / 2, y + h / 2 + 21, sub, ss, GRAY)
    else:
        out += text(x + w / 2, y + h / 2 + 7, title, ts, tcolor, weight='bold')
    return out

def hook():
    """3,000 failures a month, split by the probe that removed them."""
    W, H = (900, 300)
    x0, x1 = (40, 860)
    scale = (x1 - x0) / 3000
    parts = [('schema field missing', 2100, '2,100', AMBER, AMBER_L), ('sandbox timeout', 600, '600', PURPLE, PURPLE_L), ('residual', 300, '300', RED, RED_L)]
    for step in (1, 2, 3):
        b = text(450, 40, '3,000 failed trajectories a month (illustrative, NM 15.1)', 22, INK, weight='bold')
        if step == 1:
            b += box(x0, 70, x1 - x0, 80, '#e2e8f0', LINE, rx=6)
            b += text(450, 120, '?   what fixes them   ?', 24, GRAY, weight='bold')
        else:
            x = x0
            for label, n, shown, s, f in parts:
                w = n * scale
                b += box(x, 70, w, 80, f, s, rx=6)
                b += text(x + w / 2, 105, shown, 24, s, weight='bold')
                lab_x = x + w / 2 if n > 400 else x1 - 4
                anchor = 'middle' if n > 400 else 'end'
                b += text(lab_x, 135 if n > 400 else 185, label, 18, s, anchor=anchor)
                x += w
        if step == 3:
            sx = x0 + 2700 * scale
            b += f'<path d="M{x0},210 L{x0},225 L{sx - 6},225 L{sx - 6},210" fill="none" stroke="{GREEN}" stroke-width="3"/>'
            b += text((x0 + sx) / 2, 258, 'repair below the weights: 2,700 gone for every model', 20, GREEN, weight='bold')
            b += f'<path d="M{sx},210 L{sx},225 L{x1},225 L{x1},210" fill="none" stroke="{BLUE}" stroke-width="3"/>'
            b += text(x1, 285, 'weights: at most 300', 20, BLUE, anchor='end', weight='bold')
        svg(f'hook_step{step}.svg', W, H, b)

def ladder():
    """Drawn 900 x 420 for a full-width slot about 12 cm wide: type 18 units or more."""
    W, H = (900, 430)
    rows = [8, 88, 168, 248]
    for step in (1, 2, 3):
        o2 = 1 if step >= 2 else 0
        o3 = 1 if step >= 3 else 0
        b = text(70, 26, 'cheap,', 18, GRAY) + text(70, 46, 'reversible', 18, GRAY)
        b += f'<polygon points="62,58 78,58 90,378 50,378" fill="#e2e8f0" stroke="{LINE}"/>'
        b += text(70, 402, 'costly,', 18, GRAY) + text(70, 422, 'global', 18, GRAY)
        b += card(150, rows[0], 300, 60, 'Measured failure', 'recorded failing trace', RED_L, RED)
        b += arrow(300, 68, 300, 86, 'r', RED)
        b += card(150, rows[1], 300, 60, 'Context probe', 'supply the missing fact', GRAY_L, LINE)
        b += text(522, rows[1] + 22, 'disappears', 18, GRAY) + arrow(450, rows[1] + 30, 592, rows[1] + 30, 'n', GREEN)
        b += card(596, rows[1], 290, 60, 'Repair context', 'retrieval or assembly', GREEN_L, GREEN)
        b += g(arrow(300, 148, 300, 166, 'r', RED) + card(150, rows[2], 300, 60, 'Schema probe', 'corrected tool description', GRAY_L, LINE) + text(522, rows[2] + 22, 'disappears', 18, GRAY) + arrow(450, rows[2] + 30, 592, rows[2] + 30, 'a', AMBER) + card(596, rows[2], 290, 60, 'Repair tool or schema', 'schema, docs, adapter', AMBER_L, AMBER) + arrow(300, 228, 300, 246, 'r', RED) + card(150, rows[3], 300, 60, 'Runtime probe', 'clean sandbox, raised limits', GRAY_L, LINE) + text(522, rows[3] + 22, 'disappears', 18, GRAY) + arrow(450, rows[3] + 30, 592, rows[3] + 30, 'p', PURPLE) + card(596, rows[3], 290, 60, 'Repair runtime', 'bounds, sandbox pool', PURPLE_L, PURPLE), o2)
        b += g(f'<line x1="120" y1="320" x2="890" y2="320" stroke="{LINE}" stroke-width="2" stroke-dasharray="8 6"/>' + arrow(300, 308, 300, 350, 'r', RED) + text(312, 340, 'survives every probe', 18, GRAY, 'start') + card(150, 354, 300, 60, 'Residual failure', 'a capability gap', RED_L, RED) + text(522, 376, 'on evidence', 18, GRAY) + arrow(450, 384, 592, 384, 'b', BLUE) + card(596, 348, 290, 36, 'Supervised fine-tuning', None, BLUE_L, BLUE, ts=19) + card(596, 390, 290, 36, 'Reinforcement learning', None, BLUE_L, BLUE, ts=19), o3)
        svg(f'ladder_step{step}.svg', W, H, b)

def artifacts():
    """Drawn 640 x 380 for a column slot about 9.5 cm wide."""
    W, H = (640, 380)
    rows = [('Imitation material', 'mistake as context;', 'only the fix as target', GREEN, GREEN_L), ('Preference comparison', 'same start, tools, state;', 'a recorded reason', AMBER, AMBER_L), ('Rollout fixture', 'resettable environment', 'and a grader', PURPLE, PURPLE_L)]
    for step in (1, 2, 3):
        b = card(6, 140, 190, 96, 'One recorded run', 'trace_r19_01', GRAY_L, LINE, ts=20, ss=18)
        b += text(101, 264, 'outcome: accepted', 18, GRAY)
        for i, (t, l1, l2, s, f) in enumerate(rows):
            y = 6 + i * 126
            o = 1 if i < step else 0
            b += g(arrow(196, 188, 262, y + 56, 'g', GRAY) + box(266, y, 368, 114, f, s) + text(282, y + 34, t, 22, s, 'start', 'bold') + text(282, y + 66, l1, 20, INK, 'start') + text(282, y + 96, l2, 20, INK, 'start'), o)
        svg(f'artifacts_step{step}.svg', W, H, b)

def straddle():
    k_max = int(8.0)
    ks = list(range(1, k_max + 1))
    vals = [0.0, 32.0, 48.0, 58.88, 67.2, 73.7792, 79.0272, 83.222528]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.7))
        shown = ks if step == 2 else ks[:2]
        ax.plot(shown, vals[:len(shown)], color=PURPLE, lw=3, marker='o', ms=6)
        if step == 1:
            ax.text(k_max, 50, '8 attempts:\n?', fontsize=15, color=PURPLE, ha='center', fontweight='bold')
        else:
            ax.plot([k_max], [vals[-1]], 'o', color=RED, ms=11, zorder=5)
            ax.annotate('83.2% of tasks\non both sides', (k_max, vals[-1]), xytext=(5.0, 20), fontsize=14, color=RED, fontweight='bold', arrowprops=dict(arrowstyle='-', color=RED, lw=1.2))
        ax.set_xlim(0.6, 8.6)
        ax.set_ylim(0, 100)
        ax.set_xticks(ks)
        ax.set_xlabel('Attempts per task (random 80/20 split of traces)', fontsize=13)
        ax.set_ylabel('Tasks straddling (%)', fontsize=13)
        ax.tick_params(labelsize=12)
        ax.grid(axis='y', color=sp.C['grid'], lw=0.6)
        sp.save(fig, HERE / f'straddle_step{step}.svg')

def corpus_runs():
    """Runs per issue in the public corpus, and what each split does."""
    hist = {int(k): v for k, v in STATS['runs_per_task_hist'].items()}
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    xs = sorted(hist)
    ax.bar(xs, [hist[x] for x in xs], color=BLUE, width=0.85)
    ax.set_xlabel('Runs per issue (last bar: 40 or more)', fontsize=13)
    ax.set_ylabel('Issues', fontsize=13)
    ax.tick_params(labelsize=12)
    med = STATS['runs_per_task']['median']
    ax.axvline(med, color=INK, lw=1.5, ls='--')
    ax.text(med + 0.8, max(hist.values()) * 0.92, f'median {med}', fontsize=13, color=INK)
    ax.grid(axis='y', color=sp.C['grid'], lw=0.6)
    sp.save(fig, HERE / 'corpus_runs.svg')
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    gs = STATS['group_split']
    rows = [('random split of runs', 100 * STATS['straddle']['measured'], '89.9', RED, 'issues on both sides'), ('split by issue', 0.0, f'{0.0:.1f}', GREEN, 'issues on both sides'), ('same split, by repo', 100 * gs['test_repo_shared'] / gs['test_tasks'], '77.4', AMBER, 'test issues whose repo is in training')]
    for i, (lab, v, shown, c, what) in enumerate(rows):
        y = 2 - i
        ax.barh(y, max(v, 0.4), color=c, height=0.62)
        ax.text(-2, y, lab, ha='right', va='center', fontsize=13, fontweight='bold', color=INK)
        ax.text(max(v, 0.4) + 1.5, y, f'{shown}%  {what}', va='center', fontsize=12.5, color=c, fontweight='bold')
    ax.set_xlim(0, 175)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.tick_params(labelsize=12)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Percent of 3,591 issues (or of 743 test issues)', fontsize=12)
    sp.save(fig, HERE / 'corpus_split.svg')

def groups():
    """Drawn 660 x 400 for a column slot about 9 cm wide."""
    W, H = (660, 400)
    for step in (1, 2):
        b = text(180, 26, 'task group g = pool_repair_01', 20, PURPLE, weight='bold')
        b += box(6, 40, 350, 352, PURPLE_L, PURPLE, rx=12)
        items = [('attempts 1 to 8', 'same fixture, new seeds'), ('paraphrased prompt', 'same problem, new words'), ('fork with the same bug', 'found by similarity screen'), ('trace_r19_01_redacted_v1', 'derived record')]
        for i, (t, s) in enumerate(items):
            y = 54 + i * 84
            b += card(20, y, 322, 72, t, s, 'white', PURPLE, ts=19, ss=18)
        b += arrow(358, 216, 420, 216, 'p', PURPLE)
        b += text(389, 202, 's(g)', 20, PURPLE, weight='bold')
        b += g(card(424, 92, 230, 92, 'training', 'all of g', GREEN_L, GREEN) + card(424, 252, 230, 92, 'final test', 'none of g', '#f1f5f9', LINE, dash=True), 1 if step == 2 else 0)
        svg(f'groups_step{step}.svg', W, H, b)

def lsh():
    import numpy as np
    js = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.plot(js, [0.0, 2.000000143e-08, 3.19999952e-07, 1.619998728e-06, 5.119987303e-06, 1.249992432e-05, 2.591967458e-05, 4.801888309e-05, 8.19167495e-05, 0.00013121166, 0.0001999806262, 0.0002927784718, 0.0004146367018, 0.0005710619804, 0.0007680341345, 0.001012003597, 0.001309888191, 0.001669069152, 0.002097386283, 0.002603132113, 0.003195044956, 0.003882300721, 0.00467450335, 0.00558167375, 0.006614237065, 0.007783008146, 0.009099175076, 0.0105742806, 0.01222020128, 0.01404912434, 0.01607352181, 0.01830612219, 0.02075987913, 0.02344793723, 0.02638359482, 0.02958026349, 0.03305142442, 0.03681058139, 0.04087121034, 0.04524670559, 0.04995032259, 0.0549951173, 0.0603938822, 0.06615907909, 0.07230276866, 0.07883653721, 0.08577142051, 0.09311782521, 0.100885448, 0.1090831929, 0.1177190871, 0.1268001958, 0.1363325367, 0.1463209941, 0.156769234, 0.1676796208, 0.1790531352, 0.1908892959, 0.2031860843, 0.2159398738, 0.2291453645, 0.2427955239, 0.2568815354, 0.2713927542, 0.2863166725, 0.3016388951, 0.3173431247, 0.3334111604, 0.3498229076, 0.3665564015, 0.3835878449, 0.4008916598, 0.4184405542, 0.4362056049, 0.4541563542, 0.4722609236, 0.4904861415, 0.5087976873, 0.5271602484, 0.5455376921, 0.5638932494, 0.5821897116, 0.6003896359, 0.6184555605, 0.6363502268, 0.6540368063, 0.6714791305, 0.6886419221, 0.7054910242, 0.7219936252, 0.7381184777, 0.753836108, 0.7691190139, 0.7839418489, 0.7982815889, 0.8121176812, 0.8254321723, 0.8382098126, 0.8504381377, 0.8621075231, 0.8732112136, 0.8837453245, 0.8937088161, 0.9031034406, 0.911933663, 0.9202065557, 0.9279316696, 0.935120882, 0.9417882248, 0.9479496946, 0.9536230477, 0.9588275825, 0.9635839139, 0.9679137404, 0.9718396104, 0.9753846881, 0.978572524, 0.9814268323, 0.9839712789, 0.9862292817, 0.9882238254, 0.9899772948, 0.9915113248, 0.9928466719, 0.9940031048, 0.9949993173, 0.9958528603, 0.996580096, 0.99719617, 0.997715003, 0.9981492982, 0.9985105654, 0.9988091569, 0.999054316, 0.9992542344, 0.9994161172, 0.9995462527, 0.9996500866, 0.9997322967, 0.9997968689, 0.9998471708, 0.9998860237, 0.9999157701, 0.9999383377, 0.9999552981, 0.9999679206, 0.9999772198, 0.9999839991, 0.9999888877, 0.9999923733, 0.9999948294, 0.9999965391, 0.9999977141, 0.9999985109, 0.999999044, 0.9999993954, 0.9999996237, 0.9999997696, 0.9999998613, 0.9999999181, 0.9999999525, 0.999999973, 0.999999985, 0.9999999918, 0.9999999957, 0.9999999977, 0.9999999989, 0.9999999994, 0.9999999997, 0.9999999999, 0.9999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], color=PURPLE, lw=3)
    for jk, pk, tx, ty in (('mh_j_lo', 'mh_p_lo', 0.56, 0.62), ('mh_j_hi', 'mh_p_hi', 0.66, 0.8)):
        j = {'mh_p_lo': 0.5, 'mh_p_hi': 0.8}[pk]
        ax.plot([j], [{'mh_p_lo': 0.8732112136, 'mh_p_hi': 0.9999999525}[pk]], 'o', color=RED, ms=9, zorder=5)
        ax.text(tx, ty, f"J = { {'mh_p_lo': '0.5', 'mh_p_hi': '0.8'}[pk]}: { {'mh_p_lo': '87', 'mh_p_hi': '100'}[pk]}%", fontsize=13, color=RED, fontweight='bold')
    ax.set_xlabel('Jaccard similarity of two records', fontsize=13)
    ax.set_ylabel('Sent to review', fontsize=13)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '50%', '100%'])
    ax.tick_params(labelsize=12)
    ax.text(0.02, 0.9, '32 bands of 4 rows', fontsize=13, color=PURPLE)
    ax.grid(color=sp.C['grid'], lw=0.6)
    sp.save(fig, HERE / 'lsh.svg')

def cascade():
    """Drawn 900 x 390 for a full-width slot about 12 cm wide."""
    W, H = (900, 390)
    heads = [('R', 'record'), ('U', 'use'), ('P', 'process'), ('O', 'outcome'), ('T', 'target')]
    rows = [('record A', ['pass'] * 5, ('Admitted', 'this route', PURPLE_L, PURPLE)), ('record B', ['pass', 'pass', 'pass', 'fail', '-'], ('Rejected', 'record kept', RED_L, RED)), ('record C', ['pass', 'pending', '-', '-', '-'], ('Held', 'evidence due', '#f1f5f9', GRAY))]
    for step in (1, 2, 3):
        b = ''
        for i, (L, sub) in enumerate(heads):
            x = 160 + i * 106
            b += text(x + 48, 26, L, 24, INK, weight='bold') + text(x + 48, 50, sub, 18, GRAY)
        for r, (name, cells, end) in enumerate(rows):
            y = 66 + r * 100
            o = 1 if r < step else 0
            body = card(4, y, 140, 66, name, None, PURPLE_L, PURPLE, ts=20)
            for i, c in enumerate(cells):
                x = 160 + i * 106
                if c == '-':
                    body += box(x, y, 96, 66, 'white', '#cbd5e1', dash=True, sw=2)
                    body += text(x + 48, y + 40, 'not run', 18, GRAY)
                else:
                    fill, st = {'pass': ('white', LINE), 'fail': (RED_L, RED), 'pending': ('#f1f5f9', GRAY)}[c]
                    body += box(x, y, 96, 66, fill, st, dash=c == 'pending')
                    body += text(x + 48, y + 40, c, 19, RED if c == 'fail' else INK, weight='bold' if c != 'pass' else 'normal')
            body += arrow(690, y + 33, 708, y + 33, 'g')
            body += card(712, y, 184, 66, end[0], end[1], end[2], end[3], ts=20, ss=18)
            b += g(body, o)
        b += g(text(450, 380, 'a missing check never counts as a pass', 21, INK, weight='bold'), 1 if step == 3 else 0)
        svg(f'cascade_step{step}.svg', W, H, b)

def order():
    t_best, t_worst = ([Q_(6.2382, 's'), Q_(7.5036, 's')][i].to('second').magnitude for i in (0, 1 - 0))
    rows = [('process audit first', t_best, 'data', '6.24'), ('tests first', t_worst, 'danger', '7.50')]
    sp.hbars(str(HERE / 'order.svg'), [(a, v, c, f'{shown} s') for a, v, c, shown in rows], 's', log=False, xlabel='Slot time per attempted record (s)', size=(3.6, 1.1))

def typology():
    """Drawn 900 x 460 for a full-width slot about 13 cm wide."""
    W, H = (900, 460)
    rows = [('Direct-success demonstration', [('task and', 'context', 'c'), ('inspect the', 'failing check', 't'), ('edit the', 'pool wakeup', 't'), ('required', 'checks pass', 'c'), ('report the', 'result', 't')]), ('Recovery demonstration', [('earlier', 'context', 'c'), ('e2 malformed', 'edit, refused', 'x'), ('e3 inspect', 'the schema', 't'), ('e4 corrected', 'edit', 't'), ('e5 check', 'summary', 'c'), ('e6 report', 'and scope', 't')]), ('Hard-negative candidate', [('task and', 'context', 'c'), ('plausible', 'edit', 'x'), ('targeted', 'check passes', 'c'), ('required', 'check fails', 'c')])]
    for step in (1, 2, 3):
        b = ''
        for r, (title, cells) in enumerate(rows):
            y = 2 + r * 140
            body = text(4, y + 22, title, 24, INK, 'start', 'bold')
            for i, (l1, l2, kind) in enumerate(cells):
                x = 4 + i * 149
                fill, st, dash = {'c': (GRAY_L, LINE, False), 't': (BLUE_L, BLUE, False), 'x': (RED_L, RED, True)}[kind]
                body += box(x, y + 32, 143, 80, fill, st, dash=dash)
                body += text(x + 71.5, y + 66, l1, 21) + text(x + 71.5, y + 94, l2, 21)
                bar = BLUE if kind == 't' else '#e2e8f0'
                body += f'<rect x="{x}" y="{y + 118}" width="143" height="10" fill="{bar}"/>'
            b += g(body, 1 if r < step else 0)
        b += box(4, 430, 28, 20, BLUE_L, BLUE, rx=3) + text(40, 447, 'selected target', 21, INK, 'start') + box(230, 430, 28, 20, GRAY_L, LINE, rx=3) + text(266, 447, 'context only', 21, INK, 'start') + box(440, 430, 28, 20, RED_L, RED, rx=3, dash=True) + text(476, 447, 'never imitated', 21, INK, 'start') + f'<rect x="680" y="435" width="28" height="10" fill="{BLUE}"/>' + text(716, 447, 'in the loss', 21, INK, 'start')
        svg(f'typology_step{step}.svg', W, H, b)

def denominators():
    """NM 15.3 (illustrative) beside the same split measured in the corpus."""
    ill = [100 * 0.25, 100 * 0.4496840058, 100 * 0.390625]
    rc = STATS['recovery']
    mea = [100 * rc['recovery'] / (rc['recovery'] + rc['direct']), 100 * rc['recovery_chars'] / (rc['recovery_chars'] + rc['direct_chars']), 100 * rc['recovery_ai'] / (rc['recovery_ai'] + rc['direct_ai'])]
    for name, vals, labs, color, title in (('denom_nm', ill, ['records', 'stored tokens', 'selected targets'], PURPLE, 'NM 15.3 (illustrative)'), ('denom_corpus', mea, ['resolved runs', 'characters', 'model-written chars'], BLUE, 'Corpus: runs with a refused edit')):
        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        for i, (v, lab) in enumerate(zip(vals, labs)):
            y = (2 - i) * 1.5
            ax.barh(y, v, color=color, height=0.55)
            ax.text(v + 1.5, y, f'{v:.0f}%', va='center', fontsize=14, color=color, fontweight='bold')
            ax.text(1, y + 0.36, lab, fontsize=12.5, color=INK, va='bottom')
        ax.set_xlim(0, 62)
        ax.set_ylim(-0.5, 3.95)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50])
        ax.tick_params(labelsize=12)
        ax.spines['left'].set_visible(False)
        ax.set_title(title, fontsize=13, color=color, loc='left')
        sp.save(fig, HERE / f'{name}.svg')

def plateau():
    import numpy as np
    ps = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.plot(ps, [0.0, 0.03930695642, 0.07725530557, 0.1138854985, 0.1492369774, 0.1833481963, 0.2162566406, 0.2479988465, 0.278610421, 0.3081260603, 0.3365795687, 0.3640038772, 0.3904310615, 0.4158923597, 0.4404181903, 0.4640381693, 0.4867811269, 0.5086751248, 0.5297474724, 0.550024743, 0.56953279, 0.5882967622, 0.6063411194, 0.623689648, 0.6403654752, 0.6563910842, 0.6717883285, 0.6865784459, 0.7007820729, 0.7144192579, 0.727509475, 0.740071637, 0.7521241089, 0.7636847199, 0.7747707768, 0.7853990756, 0.7955859141, 0.8053471038, 0.8146979811, 0.8236534194, 0.83222784, 0.8404352231, 0.848289119, 0.8558026584, 0.8629885629, 0.8698591557, 0.8764263708, 0.8827017639, 0.8886965213, 0.8944214695, 0.899887085, 0.9051035028, 0.910080526, 0.9148276338, 0.9193539908, 0.9236684551, 0.9277795864, 0.9316956545, 0.9354246469, 0.9389742766, 0.94235199, 0.9455649737, 0.9486201626, 0.9515242461, 0.954283676, 0.9569046728, 0.9593932322, 0.9617551323, 0.9639959394, 0.9661210143, 0.9681355187, 0.9700444211, 0.9718525023, 0.9735643616, 0.975184422, 0.9767169356, 0.9781659894, 0.97953551, 0.9808292687, 0.9820508868, 0.98320384, 0.9842914635, 0.9853169562, 0.9862833856, 0.9871936918, 0.9880506919, 0.9888570843, 0.9896154524, 0.9903282688, 0.9909978993, 0.9916266062, 0.9922165522, 0.9927698039, 0.9932883352, 0.993774031, 0.9942286897, 0.9946540271, 0.9950516794, 0.9954232055, 0.995770091, 0.99609375, 0.9963955285, 0.9966767069, 0.9969385025, 0.997182072, 0.9974085139, 0.9976188713, 0.9978141336, 0.9979952388, 0.9981630761, 0.9983184875, 0.99846227, 0.9985951776, 0.9987179232, 0.99883118, 0.9989355839, 0.9990317348, 0.9991201983, 0.9992015075, 0.999276164, 0.99934464, 0.9994073794, 0.9994647991, 0.9995172905, 0.9995652208, 0.9996089339, 0.9996487521, 0.9996849765, 0.999717889, 0.9997477525, 0.9997748125, 0.9997992975, 0.9998214206, 0.9998413798, 0.9998593591, 0.9998755294, 0.9998900488, 0.9999030641, 0.9999147109, 0.9999251143, 0.99993439, 0.9999426444, 0.9999499754, 0.999956473, 0.9999622198, 0.9999672914, 0.999971757, 0.9999756798, 0.9999791173, 0.9999821219, 0.9999847412, 0.9999870184, 0.9999889925, 0.9999906987, 0.9999921689, 0.9999934316, 0.9999945124, 0.9999954343, 0.9999962177, 0.9999968809, 0.99999744, 0.9999979094, 0.9999983016, 0.9999986279, 0.999998898, 0.9999991204, 0.9999993024, 0.9999994506, 0.9999995705, 0.9999996668, 0.9999997437, 0.9999998046, 0.9999998524, 0.9999998897, 0.9999999184, 0.9999999404, 0.999999957, 0.9999999694, 0.9999999786, 0.9999999852, 0.99999999, 0.9999999934, 0.9999999957, 0.9999999973, 0.9999999983, 0.999999999, 0.9999999994, 0.9999999997, 0.9999999998, 0.9999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], color=BLUE, lw=2)
    ax.plot(ps, [0.0, 0.03930695642, 0.07725530557, 0.1138854985, 0.1492369774, 0.1833481963, 0.2162566406, 0.2479988465, 0.278610421, 0.3081260603, 0.3365795687, 0.3640038771, 0.3904310613, 0.4158923594, 0.4404181898, 0.4640381683, 0.4867811252, 0.5086751221, 0.5297474681, 0.5500247364, 0.56953278, 0.5882967474, 0.606341098, 0.6236896174, 0.6403654322, 0.6563910246, 0.6717882469, 0.6865783356, 0.7007819253, 0.7144190625, 0.7275092187, 0.7400713039, 0.7521236794, 0.7636841706, 0.7747700792, 0.7853981959, 0.7955848121, 0.8053457317, 0.8146962828, 0.8236513288, 0.83222528, 0.840432104, 0.8482853367, 0.8557980927, 0.8629830753, 0.8698525873, 0.8764185397, 0.8826924627, 0.8886855137, 0.8944084879, 0.8998718262, 0.9050856247, 0.9100596433, 0.9148033136, 0.9193257479, 0.9236357465, 0.9277418062, 0.9316521275, 0.9353746222, 0.938916921, 0.94228638, 0.9454900881, 0.9485348735, 0.9514273103, 0.9541737249, 0.9567802021, 0.9592525914, 0.9615965122, 0.96381736, 0.9659203118, 0.9679103312, 0.9697921736, 0.9715703913, 0.9732493382, 0.974833174, 0.9763258696, 0.9777312102, 0.9790528005, 0.9802940678, 0.9814582661, 0.98254848, 0.9835676275, 0.9845184637, 0.985403584, 0.9862254266, 0.9869862758, 0.9876882643, 0.9883333755, 0.9889234465, 0.9894601693, 0.9899450937, 0.9903796282, 0.9907650426, 0.9911024688, 0.9913929023, 0.9916372036, 0.9918360991, 0.9919901819, 0.9920999125, 0.9921656195, 0.9921875, 0.9921656195, 0.9920999125, 0.9919901819, 0.9918360991, 0.9916372036, 0.9913929023, 0.9911024688, 0.9907650426, 0.9903796282, 0.9899450937, 0.9894601693, 0.9889234465, 0.9883333755, 0.9876882643, 0.9869862758, 0.9862254266, 0.985403584, 0.9845184637, 0.9835676275, 0.98254848, 0.9814582661, 0.9802940678, 0.9790528005, 0.9777312102, 0.9763258696, 0.974833174, 0.9732493382, 0.9715703913, 0.9697921736, 0.9679103312, 0.9659203118, 0.96381736, 0.9615965122, 0.9592525914, 0.9567802021, 0.9541737249, 0.9514273103, 0.9485348735, 0.9454900881, 0.94228638, 0.938916921, 0.9353746222, 0.9316521275, 0.9277418062, 0.9236357465, 0.9193257479, 0.9148033136, 0.9100596433, 0.9050856247, 0.8998718262, 0.8944084879, 0.8886855137, 0.8826924627, 0.8764185397, 0.8698525873, 0.8629830753, 0.8557980927, 0.8482853367, 0.840432104, 0.83222528, 0.8236513288, 0.8146962828, 0.8053457317, 0.7955848121, 0.7853981959, 0.7747700792, 0.7636841706, 0.7521236794, 0.7400713039, 0.7275092187, 0.7144190625, 0.7007819253, 0.6865783356, 0.6717882469, 0.6563910246, 0.6403654322, 0.6236896174, 0.606341098, 0.5882967474, 0.56953278, 0.5500247364, 0.5297474681, 0.5086751221, 0.4867811252, 0.4640381683, 0.4404181898, 0.4158923594, 0.3904310613, 0.3640038771, 0.3365795687, 0.3081260603, 0.278610421, 0.2479988465, 0.2162566406, 0.1833481963, 0.1492369774, 0.1138854985, 0.07725530557, 0.03930695642, 0.0], color=PURPLE, lw=3.5)
    ax.text(0.3, 1.03, 'a success', fontsize=13, color=BLUE)
    ax.text(0.33, 0.62, 'mixed group', fontsize=13, color=PURPLE, fontweight='bold')
    p_lo, _, p_hi = [0.05, 0.5, 0.95]
    for p in (p_lo, p_hi):
        ax.plot([p], [{0.05: 0.3365795687, 0.95: 0.3365795687}[p]], 'o', color=RED, ms=8, zorder=5)
    ax.text(0.09, 0.22, '0.34', fontsize=13, color=RED, fontweight='bold')
    ax.text(0.73, 0.22, '0.34', fontsize=13, color=RED, fontweight='bold')
    ax.set_xlabel('Per-attempt pass rate (8 attempts)', fontsize=13)
    ax.set_ylabel('Probability', fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=12)
    ax.grid(color=sp.C['grid'], lw=0.6)
    sp.save(fig, HERE / 'plateau.svg')
    mix = STATS['outcome_mix']
    n = sum(mix.values())
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    rows = [('all runs fail', mix['all_fail'], RED), ('mixed', mix['mixed'], PURPLE), ('all runs pass', mix['all_pass'], GREEN)]
    for i, (lab, v, c) in enumerate(rows):
        y = (2 - i) * 1.5
        ax.barh(y, 100 * v / n, color=c, height=0.55)
        ax.text(100 * v / n + 2, y, f'{100 * v / n:.1f}%', va='center', fontsize=14, color=c, fontweight='bold')
        ax.text(1, y + 0.36, lab, fontsize=12.5, color=INK, va='bottom')
    ax.set_xlim(0, 115)
    ax.set_ylim(-0.5, 3.95)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.tick_params(labelsize=12)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(f'Share of {n:,} issues', fontsize=12)
    sp.save(fig, HERE / 'outcome_mix.svg')

def staging():
    store = 32.0
    t_full = 216.3461538
    wm = 0.6053333333 * store
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.8, 2.6))
        ax.plot([0, t_full], [0, store], color=RED, lw=3)
        ax.axhline(store, color=INK, lw=1.5)
        ax.text(5, store + 1.0, 'store full: 32 GB', fontsize=13, color=INK)
        ax.plot([t_full], [store], 'o', color=RED, ms=9)
        ax.text(t_full + 6, store + 1.0, '216 s', fontsize=14, color=RED, fontweight='bold', ha='left')
        ax.text(110, 6, 'verifiers stall: +148 MB/s', fontsize=13, color=RED)
        if step == 2:
            ax.axhspan(wm, store, color=AMBER_L)
            ax.axhline(wm, color=AMBER, lw=2, ls='--')
            ax.text(4, wm + 1.0, 'stop at 19.4 GB (60%)', fontsize=12.5, color=AMBER, fontweight='bold')
            ax.text(222, wm + 6.2, '8.2 GB in flight', fontsize=12, color=AMBER)
            ax.text(222, wm + 2.2, '+4.4 GB in 30 s', fontsize=12, color=AMBER)
        ax.set_xlim(0, 310)
        ax.set_ylim(0, 37)
        ax.set_xlabel('Seconds after the fixture breaks (illustrative, NM 15.4)', fontsize=12)
        ax.set_ylabel('Reserved staging (GB)', fontsize=12)
        ax.tick_params(labelsize=12)
        ax.grid(color=sp.C['grid'], lw=0.6)
        sp.save(fig, HERE / f'staging_step{step}.svg')

def lineage():
    """Drawn 900 x 400 for a full-width slot about 13 cm wide."""
    W, H = (900, 400)
    for step in (1, 2):
        b = box(4, 4, 300, 330, 'white', PURPLE, rx=10)
        b += text(154, 34, 'Source record', 24, INK, weight='bold')
        b += text(154, 62, 'trace_r19_01', 21, PURPLE, family='Courier, monospace')
        b += box(18, 76, 272, 92, GRAY_L, LINE, rx=5, sw=2)
        b += text(154, 102, 'identity, generation config,', 18, GRAY)
        b += text(154, 126, 'environment, assessment', 18, GRAY)
        b += text(154, 152, '(metadata)', 18, GRAY)
        b += box(18, 180, 272, 62, GREEN_L, GREEN, rx=5, sw=2.5)
        b += text(154, 219, 'use and partition', 23, INK, weight='bold')
        b += box(18, 254, 272, 62, PURPLE_L, PURPLE, rx=5, sw=2.5)
        b += text(154, 293, 'reverse references', 23, INK, weight='bold')
        b += card(324, 150, 176, 80, 'Redaction', 'versioned policy', GRAY_L, LINE, ts=22, ss=19)
        b += arrow(290, 190, 320, 190) + arrow(500, 190, 520, 190)
        body = box(524, 4, 372, 330, 'white', PURPLE, rx=10)
        body += text(710, 34, 'Derived record', 24, INK, weight='bold')
        body += text(710, 62, 'trace_r19_01_redacted_v1', 20, PURPLE, family='Courier, monospace')
        body += box(540, 76, 340, 56, PURPLE_L, PURPLE, rx=5, sw=2)
        body += text(710, 112, 'parent ref, changed fields', 20, INK)
        body += box(540, 144, 340, 56, GREEN_L, GREEN, rx=5, sw=2.5)
        body += text(710, 181, 'same split group', 23, INK, weight='bold')
        body += box(540, 212, 340, 56, GREEN_L, GREEN, rx=5, sw=2.5)
        body += text(710, 249, 'same use permission', 23, INK, weight='bold')
        body += box(540, 280, 340, 44, '#f1f5f9', GRAY, rx=5, sw=2, dash=True)
        body += text(710, 309, 'assessment: pending', 20, INK)
        body += arrow(540, 104, 294, 104, 'p', PURPLE)
        b += body
        b += g(f'<path d="M154,334 L154,356 L710,356 L710,338" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="7 5" marker-end="url(#ar)"/>' + text(450, 392, 'withdraw the source: every descendant is found', 23, RED, weight='bold'), 1 if step == 2 else 0)
        svg(f'lineage_step{step}.svg', W, H, b)

def dispositions():
    from matplotlib.ticker import FixedLocator, FixedFormatter
    rows = [('selected, emitted', 'co_emitted', GREEN), ('assessed, not selected', 'co_other', GRAY), ('ineligible for this use', 'co_inelig', RED), ('pending at cutoff', 'co_pending', PURPLE), ('unknown at cutoff', 'co_unknown', PURPLE)]
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    for i, (lab, key, c) in enumerate(rows):
        y = 4 - i
        v = {'selected, emitted': 15.0, 'assessed, not selected': 470.0, 'ineligible for this use': 10.0, 'pending at cutoff': 3.0, 'unknown at cutoff': 2.0}[lab]
        ax.barh(y, v, color=c, height=0.6)
        ax.text(v * 1.15, y, {'selected, emitted': '15', 'assessed, not selected': '470', 'ineligible for this use': '10', 'pending at cutoff': '3', 'unknown at cutoff': '2'}[lab], va='center', fontsize=12, color=c, fontweight='bold')
        ax.text(0.9, y, lab + ' ', va='center', ha='right', fontsize=11.5, color=INK)
    ax.set_xscale('log')
    ax.set_xlim(1, 2000)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(['1', '10', '100', '1,000']))
    ax.minorticks_off()
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_xlabel('Attempts at the cutoff (log scale)', fontsize=11)
    sp.save(fig, HERE / 'dispositions.svg')

def synthesis():
    """Drawn 900 x 200 for a full-width strip about 13 cm wide."""
    W, H = (900, 200)
    steps = [('3,000', 'failures', RED_L, RED), ('2,700', 'fixed below', GREEN_L, GREEN), ('300', 'residual', BLUE_L, BLUE), ('split', 'group fixed', PURPLE_L, PURPLE), ('R U P O T', 'all pass', PURPLE_L, PURPLE), ('e3 e4 e6', 'targets', BLUE_L, BLUE), ('redacted', 're-admitted', PURPLE_L, PURPLE)]
    b = ''
    for i, (t, s, f, st) in enumerate(steps):
        x = 4 + i * 128
        b += card(x, 40, 120, 100, t, s, f, st, ts=23, ss=20)
        if i < len(steps) - 1:
            b += arrow(x + 120, 90, x + 128, 90)
    b += text(450, 26, 'what reaches the weights, and with what evidence', 24, INK, weight='bold')
    b += text(450, 180, 'published as repair_demonstrations_v1 with its manifest', 23, PURPLE, weight='bold')
    svg('synthesis.svg', W, H, b)

def recall_thumb():
    """Text-free: one saved prefix and three continuation branches (Lecture 14's postmortem)."""
    b = f'<line x1="20" y1="110" x2="260" y2="110" stroke="{GRAY}" stroke-width="10" stroke-linecap="round"/>'
    b += f'<circle cx="260" cy="110" r="16" fill="{PURPLE}"/>'
    for y, c in ((40, GREEN), (110, GREEN), (180, RED)):
        b += f'<path d="M276,110 C330,110 340,{y} 400,{y} L560,{y}" fill="none" stroke="{c}" stroke-width="9" stroke-linecap="round"/>'
        b += f'<circle cx="580" cy="{y}" r="16" fill="{c}"/>'
    svg('recall_branches.svg', 620, 220, b)

def recipe_bars():
    """The published filter applied to the corpus: what it removes."""
    fig, ax = plt.subplots(figsize=(6.0, 1.5))
    tot, rem = (13389.0, 3971.0)
    ax.barh(1, tot, color=GREEN, height=0.55)
    ax.barh(1, rem, left=tot - rem, color=RED, height=0.55)
    ax.text(tot, 1.45, 'removed by the filter: 3,971 (29.7%)', ha='right', va='center', fontsize=12.5, color=RED, fontweight='bold')
    ax.text((tot - rem) / 2, 1, 'kept', ha='center', va='center', fontsize=12.5, color='white', fontweight='bold')
    ax.text(-tot * 0.02, 1, 'resolved runs', ha='right', va='center', fontsize=12.5)
    d, r = (27105.0, 47435.0)
    sc = tot / (r * 1.25)
    ax.barh(0.25, d * sc, color=GRAY, height=0.4)
    ax.barh(-0.3, r * sc, color=RED, height=0.4)
    ax.text(d * sc + tot * 0.01, 0.25, 'direct: 27,105 chars', va='center', fontsize=12)
    ax.text(r * sc + tot * 0.01, -0.3, 'recovery: 1.75× longer', va='center', fontsize=12, color=RED, fontweight='bold')
    ax.text(-tot * 0.02, -0.02, 'mean length', ha='right', va='center', fontsize=12.5)
    ax.set_xlim(0, tot * 1.02)
    ax.set_ylim(-0.6, 1.65)
    ax.axis('off')
    sp.save(fig, HERE / 'recipe_bars.svg')
if __name__ == '__main__':
    recall_thumb()
    hook()
    ladder()
    artifacts()
    straddle()
    corpus_runs()
    groups()
    lsh()
    cascade()
    order()
    typology()
    denominators()
    plateau()
    staging()
    lineage()
    dispositions()
    synthesis()
    recipe_bars()
