#!/usr/bin/env python3
"""Figures for Lecture 11b (Hardware Acceleration, Part 2: mapping).

Every number comes from MLSysIM through calcs.py: val(key) is the string a
slide prints and num(key) its value; bar lengths, grids, and curves read the
same registry entries, Literature results, and Scenarios.* the chapter reads.
Layout values (canvas sizes, coordinates, axis limits, ticks, label offsets)
and the illustrative liveness and compiler-stage drawings are typed here.
Outside sources are recorded with each MLSysIM entry (provenance) and in
SOURCES.yml.

Run from this folder: python3 make_figs.py
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import re
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
import contextlib, io
C = sp.C
plt = sp.plt
out = Path(__file__).parent
FONT = 'font-family="Helvetica, Arial, sans-serif"'

def svg_text(x, y, s, size=20, weight='400', fill=C['text'], anchor='middle', opacity=1.0):
    """Text with simple subscripts: 'a_{00}' sets 00 as a subscript."""
    parts = re.split('_\\{([^}]*)\\}', s)
    body = []
    for i, piece in enumerate(parts):
        if not piece:
            continue
        if i % 2 == 0:
            body.append(piece)
        else:
            sub = round(size * 0.68)
            shift = round(size * 0.28)
            body.append(f'<tspan dy="{shift}" font-size="{sub}">{piece}</tspan><tspan dy="-{shift}"></tspan>')
    return f'''<text x="{x}" y="{y}" {FONT} font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" dominant-baseline="central" opacity="{opacity}">{''.join(body)}</text>'''

def svg_open(width, height, colors=()):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', f'<rect width="{width}" height="{height}" fill="white"/>', '<defs>']
    for col in colors:
        parts.append(f'<marker id="head_{col[1:]}" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{col}"/></marker>')
    parts.append('</defs>')
    return parts

def arrow(x1, y1, x2, y2, color, width=3, opacity=1.0, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" opacity="{opacity}"{extra} marker-end="url(#head_{color[1:]})"/>'

def rect(x, y, w, h, fill, stroke, width=2.5, rx=6, opacity=1.0, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{extra}/>'

def write_svg(name, parts):
    parts.append('</svg>')
    (out / name).write_text('\n'.join(parts))

def logbars(path, rows, ticks, xlim, xlabel, size, step=None, fontsize=13):
    """Horizontal bars on a log axis with plain tick labels (1, 10, 100)."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, (label, value, color_key, text) in enumerate(rows):
        y = n - 1 - i
        shown = step is None or i < step
        ax.barh(y, value, color=C[color_key], height=0.62, alpha=1.0 if shown else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=fontsize, fontweight='bold', color=C['text'])
        ax.text(value * 1.18, y, text, va='center', fontsize=fontsize, fontweight='bold', color=C[color_key], alpha=1.0 if shown else 0.0)
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([lab for _, lab in ticks])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    sp.save(fig, path)
TAGS = {'1 Naive': ('neutral', ''), '2 Coalesced loads': ('memory', 'layout'), '3 Shared-memory tiles': ('memory', 'tile staging'), '4 1D register tiles': ('memory', 'bigger tile'), '5 2D register tiles': ('memory', 'bigger tile'), '6 Vector loads': ('memory', 'wider loads'), '9 Autotuned': ('memory', 'search'), '10 Warp tiles': ('memory', 'one more tier'), 'cuBLAS': ('compute', '')}
BOEHM = [('1 Naive', 309.0, 'neutral', ''), ('2 Coalesced loads', 1986.5, 'memory', 'layout'), ('3 Shared-memory tiles', 2980.3, 'memory', 'tile staging'), ('4 1D register tiles', 8474.7, 'memory', 'bigger tile'), ('5 2D register tiles', 15971.7, 'memory', 'bigger tile'), ('6 Vector loads', 18237.3, 'memory', 'wider loads'), ('9 Autotuned', 19721.0, 'memory', 'search'), ('10 Warp tiles', 21779.3, 'memory', 'one more tier'), ('cuBLAS', 23249.6, 'compute', '')]

def hook_fig(step):
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    rows = [('Naive kernel', 309.0, 'neutral', '309 GFLOP/s'), ('cuBLAS', 23249.6, 'compute', '23,250 GFLOP/s:  75× faster')]
    for i, (label, value, color_key, text) in enumerate(rows):
        y = 1 - i
        shown = i == 0 or step >= 2
        ax.barh(y, value if shown else 0, color=C[color_key], height=0.6)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
        ax.text(value * 1.15, y, text, va='center', fontsize=13, fontweight='bold', color=C[color_key], alpha=1.0 if shown else 0.0)
    ax.text(400, 0, '???', va='center', fontsize=16, fontweight='bold', color=C['danger'], alpha=1.0 if step == 1 else 0.0)
    ax.set_xscale('log')
    ax.set_xlim(100, 1500000.0)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.minorticks_off()
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(['100', '1,000', '10,000', '100,000'])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('FP32 matrix multiply on one A6000 (GFLOP/s, log scale)')
    sp.save(fig, out / f'hook_step{step}.svg')
for k in (1, 2):
    hook_fig(k)
fig, ax = plt.subplots(figsize=(5.8, 2.6))
n = len(BOEHM)
for i, (label, value, color_key, tag) in enumerate(BOEHM):
    y = n - 1 - i
    ax.barh(y, value, color=C[color_key], height=0.66)
    ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12.5, fontweight='bold', color=C['text'])
    text = f'{value:,.0f}' + (f'   {tag}' if tag else '')
    ax.text(value * 1.12, y, text, va='center', fontsize=12.5, fontweight='bold', color=C[color_key])
ax.set_xscale('log')
ax.set_xlim(100, 2000000.0)
ax.set_ylim(-0.6, n - 0.4)
ax.set_yticks([])
ax.minorticks_off()
ax.set_xticks([100, 1000, 10000, 100000])
ax.set_xticklabels(['100', '1,000', '10,000', '100,000'])
ax.spines['left'].set_visible(False)
ax.grid(axis='x', color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('GFLOP/s, same multiply, same GPU (log scale)')
sp.save(fig, out / 'ladder.svg')

def trace_svg(step):
    t = step - 2
    cell, x0, y0 = (110, 250, 30)
    cx = [x0, x0 + 180]
    cy = [y0, y0 + 180]
    parts = svg_open(900, 400, (C['memory'], C['data']))
    sched = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 2)]
    slot_x = {1: 188, 2: 113, 3: 38}
    for r in range(2):
        yc = cy[r] + cell / 2
        parts.append(arrow(218, yc, x0 - 4, yc, C['memory']))
        parts.append(arrow(cx[0] + cell + 3, yc, cx[1] - 5, yc, C['memory']))
        for c in range(2):
            parts.append(rect(cx[c], cy[r], cell, cell, C['compute_l'], C['compute']))
            parts.append(svg_text(cx[c] + cell / 2, cy[r] + 26, f'b_{{{r}{c}}}', 26, '700', C['compute']))
            parts.append(svg_text(cx[c] + cell - 6, cy[r] + cell - 13, f'PE{r}{c}', 15, '400', C['neutral'], 'end'))
    for c in range(2):
        xc = cx[c] + cell / 2
        parts.append(arrow(xc, cy[0] + cell + 3, xc, cy[1] - 5, C['data']))
        parts.append(arrow(xc, cy[1] + cell + 3, xc, cy[1] + cell + 36, C['data']))
    for r, i, entry in sched:
        yc = cy[r] + cell / 2
        col = t - entry
        hidden = r == 1 and t < 1
        if t >= 0 and 0 <= col <= 1:
            parts.append(svg_text(cx[col] + cell / 2, cy[r] + 70, f'a_{{{i}{r}}}', 32, '700', C['memory']))
        elif col < 0 and entry - t in slot_x:
            if hidden and i == 1:
                continue
            sx = slot_x[entry - t]
            parts.append(rect(sx - 30, yc - 26, 60, 52, 'white', C['memory'], 2, 6))
            label = '?' if hidden else f'a_{{{i}{r}}}'
            parts.append(svg_text(sx, yc, label, 26, '700', C['danger'] if hidden else C['memory']))
    for c in range(2):
        xc = cx[c] + cell / 2
        i = t - 1 - c
        if 0 <= i <= 1 and t >= 1:
            parts.append(svg_text(xc + 12, cy[0] + cell + 36, f'a_{{{i}0}}b_{{0{c}}}', 22, '700', C['data'], 'start'))
            parts.append(svg_text(xc, cy[1] + cell + 60, f'C_{{{i}{c}}}', 30, '700', C['data']))
    title = 'Before cycle 0' if t < 0 else f'Cycle {t}'
    parts.append(svg_text(610, 45, title, 34, '700', C['text'], 'start'))
    log = {1: 'C_{00}', 2: 'C_{01}, C_{10}', 3: 'C_{11}'}
    for k in (1, 2, 3):
        if t >= k:
            parts.append(svg_text(610, 62 + 36 * k, f'cycle {k}:  {log[k]}', 24, '700', C['data'], 'start'))
    if t >= 3:
        dim = 2
        parts.append(svg_text(610, 228, f'2 rows on a {dim} × {dim} array:', 22, '400', C['text'], 'start'))
        parts.append(svg_text(610, 258, f'4 cycles = 2 + 2·{dim} − 2', 22, '700', C['text'], 'start'))
    legend = [('weights stay', C['compute']), ('activations move right', C['memory']), ('partial sums move down', C['data'])]
    for k, (text, col) in enumerate(legend):
        parts.append(svg_text(610, 310 + 30 * k, text, 22, '700', col, 'start'))
    write_svg(f'trace2_step{step}.svg', parts)
for k in range(1, 6):
    trace_svg(k)

def wave4_svg(step):
    t = {1: int(2.0), 2: int(6.0), 3: int(11.0)}[step]
    rows_in, n = (8, 4)
    cell, gap, x0, y0 = (72, 10, 40, 20)
    parts = svg_open(900, 400)
    busy = 0
    for r in range(n):
        for c in range(n):
            x = x0 + c * (cell + gap)
            y = y0 + r * (cell + gap)
            k = t - r - c
            if 0 <= k < rows_in:
                busy += 1
                parts.append(rect(x, y, cell, cell, C['compute'], C['compute']))
                parts.append(svg_text(x + cell / 2, y + cell / 2, str(k + 1), 30, '700', 'white'))
            elif k >= rows_in:
                parts.append(rect(x, y, cell, cell, C['neutral_l'], C['neutral']))
            else:
                parts.append(rect(x, y, cell, cell, 'white', C['neutral']))
    phase = {1: 'filling', 2: 'full', 3: 'draining'}[step]
    px = 410
    parts.append(svg_text(px, 45, f'Cycle {t}', 36, '700', C['text'], 'start'))
    parts.append(svg_text(px, 95, f'{busy} of 16 cells busy: {phase}', 28, '700', C['compute'] if step == 2 else C['danger'], 'start'))
    total = 14
    for k in range(total):
        x = px + k * 32
        parts.append(rect(x, 140, 28, 36, C['text'] if k == t else 'white', C['neutral'], 1.5, 3))
    parts.append(svg_text(px, 196, 'cycle 0', 20, '400', C['neutral'], 'start'))
    last = total - 1
    parts.append(svg_text(px + last * 32 + 28, 196, '13', 20, '400', C['neutral'], 'end'))
    parts.append(svg_text(px, 250, '8 input rows: 8 + 2·4 − 2 = 14 cycles', 23, '700', C['text'], 'start'))
    parts.append(svg_text(px, 285, 'busy on average: 8 / 14 = 57%', 23, '400', C['text'], 'start'))
    parts.append(svg_text(x0, 372, 'Orange: busy (number = input row it works on).   Gray: done.   White: not reached yet.', 20, '400', C['neutral'], 'start'))
    write_svg(f'wave4_step{step}.svg', parts)
for k in (1, 2, 3):
    wave4_svg(k)
busy_rows = []
for s, key, color_key in zip((SimpleNamespace(array_dim=128, busy_fraction=0.003921568627, cells=16384, cycles=255, fill_drain_cycles=254, first_full_cycle=254, full_cycles=0, name='One input row through a TPU v4 MXU (128 x 128) per weight load', rows=1), SimpleNamespace(array_dim=128, busy_fraction=0.335078534, cells=16384, cycles=382, fill_drain_cycles=254, first_full_cycle=254, full_cycles=0, name='128 input rows through a TPU v4 MXU per weight load', rows=128), SimpleNamespace(array_dim=128, busy_fraction=0.8012519562, cells=16384, cycles=1278, fill_drain_cycles=254, first_full_cycle=254, full_cycles=770, name='1,024 input rows through a TPU v4 MXU per weight load', rows=1024), SimpleNamespace(array_dim=128, busy_fraction=0.9416091954, cells=16384, cycles=4350, fill_drain_cycles=254, first_full_cycle=254, full_cycles=3842, name='4,096 input rows through a TPU v4 MXU per weight load', rows=4096)), ('b1', 'b128', 'b1024', 'b4096'), ('danger', 'memory', 'memory', 'data')):
    busy_rows.append((f"{ {'b1': '1', 'b128': '128', 'b1024': '1,024', 'b4096': '4,096'}[key]} input row" + ('' if {'b1': 1, 'b128': 128, 'b1024': 1024, 'b4096': 4096}[key] == 1 else 's'), 100 * {'b1': 0.003921568627, 'b128': 0.335078534, 'b1024': 0.8012519562, 'b4096': 0.9416091954}[key], color_key, f"{ {'b1': '0.4', 'b128': '34', 'b1024': '80', 'b4096': '94'}[key]}%"))
sp.hbars(out / 'busy.svg', busy_rows, '%', log=False, size=(5.8, 1.9), xlabel='Busy fraction, 128 × 128 array, one weight tile (%, derived)')

def stationary_svg(kind):
    n, cell, gap, x0, y0 = (4, 52, 24, 92, 64)
    span = n * cell + (n - 1) * gap
    held = {'ws': ('w', C['compute'], C['compute_l']), 'os': ('z', C['data'], C['data_l']), 'is': ('x', C['memory'], C['memory_l'])}[kind]
    parts = svg_open(400, 425, (C['memory'], C['data'], C['compute']))
    for r in range(n):
        for c in range(n):
            x = x0 + c * (cell + gap)
            y = y0 + r * (cell + gap)
            parts.append(rect(x, y, cell, cell, held[2], held[1]))
            parts.append(svg_text(x + cell / 2, y + cell / 2, held[0], 28, '700', held[1]))
    left = {'ws': ('X', C['memory']), 'os': ('W', C['compute']), 'is': ('W', C['compute'])}[kind]
    parts.append(svg_text(30, y0 + span / 2, left[0], 30, '700', left[1]))
    for r in range(n):
        y = y0 + r * (cell + gap) + cell / 2
        parts.append(arrow(48, y, x0 - 6, y, left[1]))
    if kind in ('ws', 'is'):
        for c in range(n):
            x = x0 + c * (cell + gap) + cell / 2
            for r in range(n - 1):
                ya = y0 + r * (cell + gap) + cell + 2
                parts.append(arrow(x, ya, x, ya + gap - 8, C['data'], 1.8))
            parts.append(arrow(x, y0 + span + 3, x, y0 + span + 38, C['data']))
        parts.append(svg_text(x0 + span / 2, y0 + span + 64, 'partial sums Z move', 30, '700', C['data']))
    else:
        parts.append(svg_text(x0 + span / 2, 18, 'X', 30, '700', C['memory']))
        for c in range(n):
            x = x0 + c * (cell + gap) + cell / 2
            parts.append(arrow(x, 34, x, y0 - 6, C['memory']))
        parts.append(svg_text(x0 + span / 2, y0 + span + 34, 'each z written once', 30, '700', C['data']))
    write_svg(f'stationary_{kind}.svg', parts)
for kind in ('ws', 'os', 'is'):
    stationary_svg(kind)
fig, ax = plt.subplots(figsize=(4.4, 1.55))
lo, hi, fc = (1.4, 2.5, 1.3)
ax.barh(1, hi - lo, left=lo, color=C['data'], height=0.55)
ax.text(hi + 0.08, 1, '1.4–2.5×', va='center', fontsize=13, fontweight='bold', color=C['data'])
ax.annotate('', xy=(fc + 0.75, 0), xytext=(fc, 0), arrowprops=dict(arrowstyle='-|>', color=C['data'], lw=3))
ax.plot([fc], [0], 'o', ms=9, color=C['data'])
ax.text(fc + 0.83, 0, 'at least 1.3×', va='center', fontsize=13, fontweight='bold', color=C['data'])
for y, label in ((1, 'Conv layers'), (0, 'FC, batch > 16')):
    ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
ax.axvline(1, color=C['neutral'], lw=2)
ax.text(1.03, 1.62, 'other dataflows = 1×', fontsize=10, color=C['neutral'], va='center')
ax.set_xlim(0.9, 3.4)
ax.set_ylim(-0.6, 1.9)
ax.set_yticks([])
ax.set_xticks([1, 1.5, 2, 2.5, 3])
ax.set_xticklabels(['1×', '1.5×', '2×', '2.5×', '3×'])
ax.spines['left'].set_visible(False)
ax.grid(axis='x', color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('Energy advantage of row stationary (AlexNet)')
sp.save(fig, out / 'eyeriss.svg')

def layertile_mats_svg():
    """The three matrices with one 128-row weight tile (static picture)."""
    parts = svg_open(380, 292)
    parts.append(rect(20, 10, 150, 200, 'white', C['compute'], 2.5, 2))
    for y in (110, 160):
        parts.append(f'''<line x1="20" y1="{y}" x2="170" y2="{y}" stroke="{C['compute']}" stroke-width="1.5" stroke-dasharray="6,4"/>''')
    parts.append(rect(20, 60, 150, 50, C['compute_l'], C['compute'], 3, 2))
    parts.append(svg_text(95, 85, '128 rows', 26, '700', C['compute']))
    parts.append(svg_text(192, 60, '×', 34, '700', C['text']))
    parts.append(rect(215, 10, 40, 100, C['memory_l'], C['memory'], 2.5, 2))
    parts.append(svg_text(277, 60, '=', 34, '700', C['text']))
    parts.append(rect(300, 10, 40, 200, 'white', C['data'], 2.5, 2))
    parts.append(rect(300, 60, 40, 50, C['data_l'], C['data'], 3, 2))
    for x, y, name, dims, col in ((95, 240, 'W', '512 × 256', C['compute']), (235, 140, 'X', '256 × 32', C['memory']), (320, 240, 'Z', '512 × 32', C['data'])):
        parts.append(svg_text(x, y, name, 28, '700', col))
        parts.append(svg_text(x, y + 34, dims, 24, '700', col))
    write_svg('layertile_mats.svg', parts)

def layertile_cost_svg(step):
    """Step 1 blank (keeps the slot), step 2 capacity, step 3 traffic."""
    parts = svg_open(440, 350)
    a2 = 1.0 if step >= 2 else 0.0
    x0, s = (10, 1.6)
    parts.append(f'<g opacity="{a2}">')
    parts.append(svg_text(x0, 22, 'One SM holds 228 KB', 22, '700', C['text'], 'start'))
    x = x0
    for kb, col, light, label in ((64.0, C['compute'], C['compute_l'], 'W tile'), (16.0, C['memory'], C['memory_l'], ''), (16.0, C['data'], C['data_l'], '')):
        parts.append(rect(x, 40, kb * s, 40, light, col, 2, 0))
        if label:
            parts.append(svg_text(x + kb * s / 2, 60, label, 20, '700', col))
        x += kb * s
    parts.append(svg_text(x0, 102, '64 + 16 + 16 = 96 KB: fits', 21, '700', C['data'], 'start'))
    parts.append(rect(x0, 120, 256.0 * s, 38, 'white', C['danger'], 2.5, 0, 1.0, '7,4'))
    parts.append(svg_text(x0 + 10, 139, 'all of W: 256 KB, too big', 21, '700', C['danger'], 'start'))
    xl = x0 + 228.0 * s
    parts.append(f'''<line x1="{xl}" y1="32" x2="{xl}" y2="166" stroke="{C['text']}" stroke-width="2.5"/>''')
    parts.append(svg_text(xl, 182, '228 KB', 20, '700', C['text']))
    parts.append('</g>')
    a3 = 1.0 if step >= 3 else 0.0
    parts.append(f'<g opacity="{a3}">')
    parts.append(svg_text(x0, 214, 'Element moves to HBM (log scale, derived)', 21, '700', C['text'], 'start'))

    def xpos(v):
        return x0 + (np.log10(v) - 4) * 100
    untiled = int(8404992.0)
    tiled = int(180224.0)
    for y, v, col, label in ((244, untiled, C['danger'], 'untiled, output stationary'), (312, tiled, C['data'], 'four weight tiles')):
        parts.append(svg_text(x0, y, label, 20, '400', C['text'], 'start'))
        parts.append(rect(x0, y + 13, xpos(v) - x0, 32, col, col, 1, 0))
        parts.append(svg_text(xpos(v) + 8, y + 29, f'{v:,}', 21, '700', col, 'start'))
    parts.append(svg_text(x0 + 250, 335, '47× fewer', 28, '700', C['data'], 'start'))
    parts.append('</g>')
    write_svg(f'layertile_cost_step{step}.svg', parts)
layertile_mats_svg()
for k in (1, 2, 3):
    layertile_cost_svg(k)
tile_rows = [('No reuse', 0.5, 'danger', '0.5'), ('128 × 128 tiles', 64.0, 'memory', '64'), ('256 × 128 tiles', 85.33333333, 'memory', '85'), ('Load each once', 1365.333333, 'compute', '1,365')]
logbars(out / 'tileintensity.svg', tile_rows, [(1, '1'), (10, '10'), (100, '100'), (1000, '1,000')], (0.2, 40000), 'FLOP per byte loaded (log, derived)', (3.9, 2.3), fontsize=12)

def capacity_fig(step):

    def kib(q):
        return q.to('KiB').magnitude
    rows = [('128³, single', *(kib(b) for b in (Q_(32768, 'B'), Q_(32768, 'B'), Q_(65536, 'B')))), ('128³, double', *(kib(b) for b in (Q_(65536, 'B'), Q_(65536, 'B'), Q_(65536, 'B')))), ('K-tile 256, double', *(kib(b) for b in (Q_(131072, 'B'), Q_(131072, 'B'), Q_(65536, 'B'))))]
    budget = 228.0
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    for i, (label, a, b, c) in enumerate(rows):
        y = len(rows) - 1 - i
        alpha = 1.0 if i < step - 1 else 0.0
        left = 0
        for value, key, tag in ((a, 'compute', 'A'), (b, 'memory', 'B'), (c, 'data', 'acc')):
            ax.barh(y, value, left=left, color=C[key], height=0.6, alpha=alpha, edgecolor='white', linewidth=1.5)
            ax.text(left + value / 2, y, tag, ha='center', va='center', fontsize=11, fontweight='bold', color='white', alpha=alpha)
            left += value
        total = a + b + c
        over = total > budget
        ax.text(335, y, f'{total:.0f} KB' + (': over' if over else ': fits'), va='center', fontsize=13, fontweight='bold', color=C['danger'] if over else C['data'], alpha=alpha)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
    ax.axvline(budget, color=C['text'], lw=2, ls='--')
    ax.text(budget + 2, 2.55, '228 KB', fontsize=11, fontweight='bold', color=C['text'])
    ax.set_xlim(0, 450)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('Bytes live at once (KB, derived from equation 6)')
    sp.save(fig, out / f'capacity_step{step}.svg')
for k in (1, 2, 3, 4):
    capacity_fig(k)

def wave_svg(step):
    cols, rows_n = (int(12.0), int(9.0))
    box, gap = (24, 5)
    parts = svg_open(900, 340)
    for panel, (x0, filled, title, colour) in enumerate(((40, int(108.0), 'Wave 1: 108 tiles on 108 SMs', C['compute']), (500, int(9.0), 'Wave 2: 9 tiles, 99 SMs idle', C['danger']))):
        alpha = 1.0 if panel == 0 or step >= 2 else 0.0
        parts.append(f'<g opacity="{alpha}">')
        parts.append(svg_text(x0, 22, title, 23, '700', colour, 'start'))
        for k in range(cols * rows_n):
            r, c = divmod(k, cols)
            x = x0 + c * (box + gap)
            y = 50 + r * (box + gap)
            on = k < filled
            parts.append(rect(x, y, box, box, colour if on else 'white', colour if on else C['neutral'], 1.5, 3))
        parts.append('</g>')
    parts.append(svg_text(40, 330, 'Each box is one SM of an A100', 18, '400', C['neutral'], 'start'))
    write_svg(f'wave_step{step}.svg', parts)
for k in (1, 2):
    wave_svg(k)
SUP = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
space_rows = [('Placement', 1024, 'neutral', '4⁵ = 1,024'), ('Loop order', 5040, 'neutral', '7! = 5,040'), ('Parallelization', 120, 'neutral', '6!/3! = 120'), ('All three, one level', 619315200, 'danger', '≈ 620 million')]
for k in (1, 2):
    logbars(out / f'space_step{k}.svg', space_rows, [(100, '100'), (10000.0, '10,000'), (1000000.0, '1 million'), (1000000000.0, '1 billion')], (30, 10000000000000.0), 'Choices for one layer (log scale)', (5.8, 2.0), step=3 if k == 1 else 4)

def compiler_svg(step):
    width, gap, x0 = (205, 18, 13)
    parts = svg_open(900, 262, (C['neutral'],))
    names = [('Graph', 'optimization'), ('Kernel', 'selection'), ('Memory', 'planning'), ('Scheduling', '')]
    for k in range(4):
        x = x0 + k * (width + gap)
        stage = k + 1
        alpha = 1.0 if stage <= step else 0.22
        lit = stage == step
        parts.append(f'<g opacity="{alpha}">')
        parts.append(rect(x, 6, width, 70, C['compute_l'] if lit else C['neutral_l'], C['compute'] if lit else C['neutral'], 3 if lit else 2))
        a, b = names[k]
        if b:
            parts.append(svg_text(x + width / 2, 29, a, 22, '700', C['text']))
            parts.append(svg_text(x + width / 2, 54, b, 22, '700', C['text']))
        else:
            parts.append(svg_text(x + width / 2, 41, a, 22, '700', C['text']))
        if k == 0:
            for j, op in enumerate(('Conv', 'BN', 'ReLU')):
                parts.append(rect(x + 8, 92 + j * 42, 72, 34, 'white', C['neutral'], 2, 4))
                parts.append(svg_text(x + 44, 109 + j * 42, op, 18, '700', C['text']))
            parts.append(arrow(x + 86, 151, x + 112, 151, C['neutral'], 2.5))
            parts.append(rect(x + 118, 92, 80, 118, C['compute_l'], C['compute'], 2.5, 4))
            for j, op in enumerate(('Conv', '+ BN', '+ ReLU')):
                parts.append(svg_text(x + 158, 120 + j * 32, op, 18, '700', C['compute']))
            parts.append(svg_text(x + width / 2, 242, '3 kernels to 1', 19, '700', C['text']))
        elif k == 1:
            for j, (op, chosen) in enumerate((('Tensor Core', True), ('CUDA cores', False), ('Fallback', False))):
                parts.append(rect(x + 20, 92 + j * 42, width - 40, 34, C['compute_l'] if chosen else 'white', C['compute'] if chosen else C['neutral'], 3 if chosen else 2, 4))
                parts.append(svg_text(x + width / 2, 109 + j * 42, op, 18, '700', C['compute'] if chosen else C['neutral']))
            parts.append(svg_text(x + width / 2, 242, 'one kernel per layer', 19, '700', C['text']))
        elif k == 2:
            spans = [(0, 0, 2), (1, 1, 3), (0, 2, 4), (1, 3, 5), (0, 4, 6)]
            for j, (buf, start, end) in enumerate(spans):
                bx = x + 14 + start * 29
                by = 96 + buf * 50
                parts.append(rect(bx + 1, by, (end - start) * 29 - 3, 38, C['memory_l'], C['memory'], 2, 4))
                parts.append(svg_text(bx + (end - start) * 29 / 2, by + 19, f'T{j + 1}', 18, '700', C['memory']))
            parts.append(svg_text(x + width / 2, 204, 'time →', 18, '400', C['neutral']))
            parts.append(svg_text(x + width / 2, 242, '5 tensors, 2 buffers', 19, '700', C['text']))
        else:
            for j, (lane, col, light) in enumerate((('copy', C['memory'], C['memory_l']), ('compute', C['compute'], C['compute_l']))):
                by = 96 + j * 50
                parts.append(svg_text(x + 2, by + 19, lane, 18, '700', col, 'start'))
                for blk in range(3):
                    bx = x + 90 + blk * 34 + j * 15
                    parts.append(rect(bx, by, 28, 38, light, col, 2, 4))
            parts.append(svg_text(x + width / 2, 204, 'time →', 18, '400', C['neutral']))
            parts.append(svg_text(x + width / 2, 242, 'copy overlaps compute', 19, '700', C['text']))
        parts.append('</g>')
        if k < 3:
            parts.append(arrow(x + width + 1, 41, x + width + gap - 3, 41, C['neutral'], 2.5, 1.0 if stage < step else 0.22))
    write_svg(f'compiler_step{step}.svg', parts)
for k in (1, 2, 3, 4):
    compiler_svg(k)
reg_rows = []
for s, key, color_key in zip((SimpleNamespace(hardware=SimpleNamespace(accelerator_count=None, battery_capacity=None, compute=SimpleNamespace(clock_rate=None, core_count=None, fp32_core_count=16896, fp32_cores_per_sm=128, max_registers_per_thread=255, max_threads_per_sm=2048, peak_flops=Q_(989.0, 'TFLOPs / s'), precision_flops={'tf32': Q_(494.5, 'TFLOPs / s'), 'fp16_dense': Q_(989.0, 'TFLOPs / s'), 'fp16_sparse': Q_(1979.0, 'TFLOPs / s'), 'fp8': Q_(1979.0, 'TFLOPs / s'), 'int8': Q_(1979, 'TOPS'), 'tf32_sparse': Q_(989.0, 'TFLOPs / s'), 'int8_sparse': Q_(3958, 'TOPS'), 'fp32': Q_(67.0, 'TFLOPs / s')}, sm_count=132, systolic_array_dim=None, systolic_macs_per_cycle=None, tensor_core_count=528, tensor_cores_per_sm=4, warp_size=32), die_area=Q_(814, 'mm ** 2'), die_to_die_bandwidth=None, dispatch_tax=Q_(0.01, 'ms'), embodied_carbon=Q_(164.0, 'kg'), embodied_carbon_kg=164.0, interconnect=SimpleNamespace(bandwidth=Q_(64.0, 'GB / s'), bandwidth_per_direction=Q_(64.0, 'GB / s'), direction='per_direction', latency=None, name='PCIe Gen5 x16'), memory=SimpleNamespace(bandwidth=Q_(3.35, 'TB / s'), bus_width=None, capacity=Q_(80, 'GiB'), flash_bandwidth=None, flash_capacity=None, hbm_stacks=5, l2_cache=Q_(50, 'MiB'), pin_data_rate=None, register_file_per_sm=Q_(256, 'KiB'), shared_memory_per_sm=Q_(228, 'KiB'), sram_bandwidth=None, sram_capacity=None), metadata=SimpleNamespace(last_verified=None, source_label='NVIDIA H100 datasheet; NVIDIA Hopper Architecture In-Depth; Choquette (2023), IEEE Micro; NVIDIA HGX H100 product carbon footprint', version=None), mig_profiles=None, name='NVIDIA H100', nvlink=SimpleNamespace(bandwidth=Q_(900.0, 'GB / s'), bandwidth_per_direction=Q_(450.0, 'GB / s'), direction='bidirectional_total', latency=Q_(500, 'ns'), name='NVLink 4.0'), release_year=2022, storage=SimpleNamespace(bandwidth=Q_(7.0, 'GB / s'), capacity=Q_(2, 'TiB'), latency=None), tdp=Q_(700, 'W'), tdp_max=None, tdp_min=None, transistor_count=80000000000.0, unit_cost=Q_(25000, 'dollar'), unit_cost_max=Q_(30000, 'dollar')), name='An H100 kernel using 32 registers per thread', occupancy=1.0, registers_per_sm=65536, registers_per_thread=32, resident_threads=2048, threads_by_registers=2048), SimpleNamespace(hardware=SimpleNamespace(accelerator_count=None, battery_capacity=None, compute=SimpleNamespace(clock_rate=None, core_count=None, fp32_core_count=16896, fp32_cores_per_sm=128, max_registers_per_thread=255, max_threads_per_sm=2048, peak_flops=Q_(989.0, 'TFLOPs / s'), precision_flops={'tf32': Q_(494.5, 'TFLOPs / s'), 'fp16_dense': Q_(989.0, 'TFLOPs / s'), 'fp16_sparse': Q_(1979.0, 'TFLOPs / s'), 'fp8': Q_(1979.0, 'TFLOPs / s'), 'int8': Q_(1979, 'TOPS'), 'tf32_sparse': Q_(989.0, 'TFLOPs / s'), 'int8_sparse': Q_(3958, 'TOPS'), 'fp32': Q_(67.0, 'TFLOPs / s')}, sm_count=132, systolic_array_dim=None, systolic_macs_per_cycle=None, tensor_core_count=528, tensor_cores_per_sm=4, warp_size=32), die_area=Q_(814, 'mm ** 2'), die_to_die_bandwidth=None, dispatch_tax=Q_(0.01, 'ms'), embodied_carbon=Q_(164.0, 'kg'), embodied_carbon_kg=164.0, interconnect=SimpleNamespace(bandwidth=Q_(64.0, 'GB / s'), bandwidth_per_direction=Q_(64.0, 'GB / s'), direction='per_direction', latency=None, name='PCIe Gen5 x16'), memory=SimpleNamespace(bandwidth=Q_(3.35, 'TB / s'), bus_width=None, capacity=Q_(80, 'GiB'), flash_bandwidth=None, flash_capacity=None, hbm_stacks=5, l2_cache=Q_(50, 'MiB'), pin_data_rate=None, register_file_per_sm=Q_(256, 'KiB'), shared_memory_per_sm=Q_(228, 'KiB'), sram_bandwidth=None, sram_capacity=None), metadata=SimpleNamespace(last_verified=None, source_label='NVIDIA H100 datasheet; NVIDIA Hopper Architecture In-Depth; Choquette (2023), IEEE Micro; NVIDIA HGX H100 product carbon footprint', version=None), mig_profiles=None, name='NVIDIA H100', nvlink=SimpleNamespace(bandwidth=Q_(900.0, 'GB / s'), bandwidth_per_direction=Q_(450.0, 'GB / s'), direction='bidirectional_total', latency=Q_(500, 'ns'), name='NVLink 4.0'), release_year=2022, storage=SimpleNamespace(bandwidth=Q_(7.0, 'GB / s'), capacity=Q_(2, 'TiB'), latency=None), tdp=Q_(700, 'W'), tdp_max=None, tdp_min=None, transistor_count=80000000000.0, unit_cost=Q_(25000, 'dollar'), unit_cost_max=Q_(30000, 'dollar')), name='An H100 kernel using 64 registers per thread', occupancy=0.5, registers_per_sm=65536, registers_per_thread=64, resident_threads=1024, threads_by_registers=1024), SimpleNamespace(hardware=SimpleNamespace(accelerator_count=None, battery_capacity=None, compute=SimpleNamespace(clock_rate=None, core_count=None, fp32_core_count=16896, fp32_cores_per_sm=128, max_registers_per_thread=255, max_threads_per_sm=2048, peak_flops=Q_(989.0, 'TFLOPs / s'), precision_flops={'tf32': Q_(494.5, 'TFLOPs / s'), 'fp16_dense': Q_(989.0, 'TFLOPs / s'), 'fp16_sparse': Q_(1979.0, 'TFLOPs / s'), 'fp8': Q_(1979.0, 'TFLOPs / s'), 'int8': Q_(1979, 'TOPS'), 'tf32_sparse': Q_(989.0, 'TFLOPs / s'), 'int8_sparse': Q_(3958, 'TOPS'), 'fp32': Q_(67.0, 'TFLOPs / s')}, sm_count=132, systolic_array_dim=None, systolic_macs_per_cycle=None, tensor_core_count=528, tensor_cores_per_sm=4, warp_size=32), die_area=Q_(814, 'mm ** 2'), die_to_die_bandwidth=None, dispatch_tax=Q_(0.01, 'ms'), embodied_carbon=Q_(164.0, 'kg'), embodied_carbon_kg=164.0, interconnect=SimpleNamespace(bandwidth=Q_(64.0, 'GB / s'), bandwidth_per_direction=Q_(64.0, 'GB / s'), direction='per_direction', latency=None, name='PCIe Gen5 x16'), memory=SimpleNamespace(bandwidth=Q_(3.35, 'TB / s'), bus_width=None, capacity=Q_(80, 'GiB'), flash_bandwidth=None, flash_capacity=None, hbm_stacks=5, l2_cache=Q_(50, 'MiB'), pin_data_rate=None, register_file_per_sm=Q_(256, 'KiB'), shared_memory_per_sm=Q_(228, 'KiB'), sram_bandwidth=None, sram_capacity=None), metadata=SimpleNamespace(last_verified=None, source_label='NVIDIA H100 datasheet; NVIDIA Hopper Architecture In-Depth; Choquette (2023), IEEE Micro; NVIDIA HGX H100 product carbon footprint', version=None), mig_profiles=None, name='NVIDIA H100', nvlink=SimpleNamespace(bandwidth=Q_(900.0, 'GB / s'), bandwidth_per_direction=Q_(450.0, 'GB / s'), direction='bidirectional_total', latency=Q_(500, 'ns'), name='NVLink 4.0'), release_year=2022, storage=SimpleNamespace(bandwidth=Q_(7.0, 'GB / s'), capacity=Q_(2, 'TiB'), latency=None), tdp=Q_(700, 'W'), tdp_max=None, tdp_min=None, transistor_count=80000000000.0, unit_cost=Q_(25000, 'dollar'), unit_cost_max=Q_(30000, 'dollar')), name='An H100 kernel using 128 registers per thread (a fused kernel)', occupancy=0.25, registers_per_sm=65536, registers_per_thread=128, resident_threads=512, threads_by_registers=512), SimpleNamespace(hardware=SimpleNamespace(accelerator_count=None, battery_capacity=None, compute=SimpleNamespace(clock_rate=None, core_count=None, fp32_core_count=16896, fp32_cores_per_sm=128, max_registers_per_thread=255, max_threads_per_sm=2048, peak_flops=Q_(989.0, 'TFLOPs / s'), precision_flops={'tf32': Q_(494.5, 'TFLOPs / s'), 'fp16_dense': Q_(989.0, 'TFLOPs / s'), 'fp16_sparse': Q_(1979.0, 'TFLOPs / s'), 'fp8': Q_(1979.0, 'TFLOPs / s'), 'int8': Q_(1979, 'TOPS'), 'tf32_sparse': Q_(989.0, 'TFLOPs / s'), 'int8_sparse': Q_(3958, 'TOPS'), 'fp32': Q_(67.0, 'TFLOPs / s')}, sm_count=132, systolic_array_dim=None, systolic_macs_per_cycle=None, tensor_core_count=528, tensor_cores_per_sm=4, warp_size=32), die_area=Q_(814, 'mm ** 2'), die_to_die_bandwidth=None, dispatch_tax=Q_(0.01, 'ms'), embodied_carbon=Q_(164.0, 'kg'), embodied_carbon_kg=164.0, interconnect=SimpleNamespace(bandwidth=Q_(64.0, 'GB / s'), bandwidth_per_direction=Q_(64.0, 'GB / s'), direction='per_direction', latency=None, name='PCIe Gen5 x16'), memory=SimpleNamespace(bandwidth=Q_(3.35, 'TB / s'), bus_width=None, capacity=Q_(80, 'GiB'), flash_bandwidth=None, flash_capacity=None, hbm_stacks=5, l2_cache=Q_(50, 'MiB'), pin_data_rate=None, register_file_per_sm=Q_(256, 'KiB'), shared_memory_per_sm=Q_(228, 'KiB'), sram_bandwidth=None, sram_capacity=None), metadata=SimpleNamespace(last_verified=None, source_label='NVIDIA H100 datasheet; NVIDIA Hopper Architecture In-Depth; Choquette (2023), IEEE Micro; NVIDIA HGX H100 product carbon footprint', version=None), mig_profiles=None, name='NVIDIA H100', nvlink=SimpleNamespace(bandwidth=Q_(900.0, 'GB / s'), bandwidth_per_direction=Q_(450.0, 'GB / s'), direction='bidirectional_total', latency=Q_(500, 'ns'), name='NVLink 4.0'), release_year=2022, storage=SimpleNamespace(bandwidth=Q_(7.0, 'GB / s'), capacity=Q_(2, 'TiB'), latency=None), tdp=Q_(700, 'W'), tdp_max=None, tdp_min=None, transistor_count=80000000000.0, unit_cost=Q_(25000, 'dollar'), unit_cost_max=Q_(30000, 'dollar')), name='An H100 kernel at the 255-register-per-thread limit', occupancy=0.125, registers_per_sm=65536, registers_per_thread=255, resident_threads=256, threads_by_registers=257)), ('r32', 'r64', 'r128', 'rmax'), ('data', 'memory', 'memory', 'danger')):
    reg_rows.append((f"{ {'r32': '32', 'r64': '64', 'r128': '128', 'rmax': '255'}[key]} registers", {'r32': 2048, 'r64': 1024, 'r128': 512, 'rmax': 256}[key], color_key, f"{ {'r32': '2,048', 'r64': '1,024', 'r128': '512', 'rmax': '256'}[key]} threads: { {'r32': '100', 'r64': '50', 'r128': '25', 'rmax': '12.5'}[key]}%"))
sp.hbars(out / 'regs.svg', reg_rows, 'threads', log=False, size=(5.8, 1.55), xlabel='Threads resident on one H100 SM (derived)')

def liveness_fig(step):
    spans = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.0), gridspec_kw={'wspace': 0.45})
    for panel, ax in enumerate(axes):
        alpha = 1.0 if panel == 0 or step >= 2 else 0.0
        for j, (start, end) in enumerate(spans):
            row = 4 - j if panel == 0 else 1 if j % 2 == 0 else 0
            ax.barh(row, end - start - 0.08, left=start + 0.04, height=0.62, color=C['memory_l'], edgecolor=C['memory'], linewidth=1.5, alpha=alpha)
            ax.text((start + end) / 2, row, f'T{j + 1}', ha='center', va='center', fontsize=11, fontweight='bold', color=C['memory'], alpha=alpha)
        n_rows = 5 if panel == 0 else 2
        ax.set_ylim(-0.6, n_rows - 0.4)
        ax.set_xlim(0, 6)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f'buf {n_rows - r}' for r in range(n_rows)][::-1] if panel == 0 else ['buf 2', 'buf 1'], fontsize=10)
        ax.set_xticks([])
        ax.set_xlabel('layers, in order →', fontsize=11)
        ax.set_title('One buffer each: 5' if panel == 0 else 'Shared by liveness: 2', fontsize=12, fontweight='bold', color=C['danger'] if panel == 0 else C['data'])
        if panel == 1 and step < 2:
            ax.set_title('')
            ax.set_yticklabels(['', ''])
            ax.set_xlabel('')
            ax.tick_params(left=False, bottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    sp.save(fig, out / f'liveness_step{step}.svg')
for k in (1, 2):
    liveness_fig(k)

def diag_svg(step):
    parts = svg_open(900, 284, (C['text'],))
    good, bad = ((C['data'], C['data_l']), (C['danger'], C['danger_l']))

    def head(x, y, text):
        parts.append(svg_text(x, y, text, 22, '700', C['text'], 'start'))

    def pill(x, y, w, text, colors):
        parts.append(rect(x, y, w, 36, colors[1], colors[0], 2, 6))
        parts.append(svg_text(x + w / 2, y + 18, text, 19, '700', colors[0]))
    parts.append(f'<g opacity="{(1.0 if step >= 2 else 0.0)}">')
    head(10, 16, '1  Data type')
    pill(10, 36, 420, 'FP16 or BF16 → Tensor Core kernel', good)
    pill(10, 80, 420, 'FP32 → CUDA cores, unless TF32 is on', bad)
    parts.append('</g>')
    parts.append(f'<g opacity="{(1.0 if step >= 3 else 0.0)}">')
    head(462, 16, '2  Shape (width 1,000, illustrative)')
    bw, x0 = (24, 462)
    for k in range(16):
        x = x0 + k * (bw + 2)
        if k < 15:
            parts.append(rect(x, 36, bw, 34, C['compute_l'], C['compute'], 1.5, 3))
        else:
            parts.append(rect(x, 36, bw, 34, 'white', C['danger'], 2, 3, 1.0, '4,3'))
            parts.append(rect(x, 36, bw * 40 / 64, 34, C['danger_l'], C['danger'], 1.5, 3))
    parts.append(svg_text(x0, 92, '1,000 = 15 × 64 + 40', 19, '700', C['danger'], 'start'))
    parts.append(svg_text(x0, 118, 'A100: multiples of 64; pad to 1,024', 19, '700', C['data'], 'start'))
    parts.append('</g>')
    parts.append(f'<g opacity="{(1.0 if step >= 4 else 0.0)}">')
    head(10, 156, '3  Layout, for convolutions: the first nine values in memory')
    chans = {'R': bad, 'G': good, 'B': (C['memory'], C['memory_l'])}
    seqs = [('NHWC', [c + str(p) for p in range(3) for c in 'RGB']), ('NCHW', ['R' + str(p) for p in range(9)])]
    for k, (name, seq) in enumerate(seqs):
        y = 176 + k * 54
        parts.append(svg_text(10, y + 20, name, 21, '700', C['text'], 'start'))
        for j, lab in enumerate(seq):
            col = chans[lab[0]]
            x = 90 + j * 46
            parts.append(rect(x, y, 42, 40, col[1], col[0], 1.5, 3))
            parts.append(svg_text(x + 21, y + 20, lab, 18, '700', col[0]))
        parts.append(svg_text(90 + 9 * 46 + 6, y + 20, '…', 22, '700', C['neutral'], 'start'))
        note = "pixel 0's channels: one read" if k == 0 else "pixel 0's channels: 9 apart"
        parts.append(svg_text(550, y + 20, note, 20, '700', C['data'] if k == 0 else C['danger'], 'start'))
    parts.append('</g>')
    write_svg(f'diag_step{step}.svg', parts)
for k in (1, 2, 3, 4):
    diag_svg(k)

def uj(scenario):
    return scenario.energy.to('microjoule').magnitude
sp.hbars(out / 'boundary.svg', [('Unfused, FP16, 45 nm', uj(SimpleNamespace(energy=Q_(128.45056, 'µJ'), energy_per_byte=Q_(160.0, 'pJ / B'), name='The FP16 stage-2 map crossing one unfused layer boundary', tensor_bytes=Q_(401408, 'B'), trips=2)), 'danger', '128 µJ'), ('Unfused, INT8, 45 nm', uj(SimpleNamespace(energy=Q_(64.22528, 'µJ'), energy_per_byte=Q_(160.0, 'pJ / B'), name='The INT8 stage-2 map crossing one unfused layer boundary (write, then read)', tensor_bytes=Q_(200704, 'B'), trips=2)), 'memory', '64 µJ'), ('Unfused, INT8, 7 nm HBM2', uj(SimpleNamespace(energy=Q_(22.5792, 'µJ'), energy_per_byte=Q_(56.25, 'pJ / B'), name='The INT8 stage-2 map crossing one boundary at 7 nm HBM2 I/O energy, high end', tensor_bytes=Q_(200704, 'B'), trips=2)), 'neutral', '12.5 to 22.6 µJ'), ('Fused', 0.0, 'data', 'about 0: the map stays on chip')], 'µJ', log=False, size=(5.8, 2.0), xlabel='HBM energy per image at one layer boundary (µJ, derived)')
fig, ax = plt.subplots(figsize=(4.6, 0.95))
for y, (label, value, color_key, text) in enumerate((('Filters spilled', 5184.0, 'danger', '5,184 reads: 36×'), ('Filters held', 144.0, 'data', '144 reads'))):
    ax.barh(y, value, color=C[color_key], height=0.62)
    ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=14, fontweight='bold', color=C['text'])
    ax.text(value + 80, y, text, va='center', fontsize=14, fontweight='bold', color=C[color_key])
ax.set_xlim(0, 8200)
ax.set_ylim(-0.5, 1.5)
ax.axis('off')
sp.save(fig, out / 'weight_reads.svg')

def tiling_svg():
    side, x0, y0, tiles = (400, 40, 30, int(32.0))
    step_px = side / tiles
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="480" height="485" viewBox="0 0 480 485">', '<rect width="480" height="485" fill="white"/>']
    parts.append(f'''<rect x="{x0}" y="{y0}" width="{side}" height="{side}" fill="{C['neutral_l']}" fill-opacity="0.35" stroke="{C['text']}" stroke-width="2"/>''')
    for i in range(1, tiles):
        v = x0 + i * step_px
        h = y0 + i * step_px
        parts.append(f'''<line x1="{v}" y1="{y0}" x2="{v}" y2="{y0 + side}" stroke="{C['neutral']}" stroke-width="0.6"/>''')
        parts.append(f'''<line x1="{x0}" y1="{h}" x2="{x0 + side}" y2="{h}" stroke="{C['neutral']}" stroke-width="0.6"/>''')
    tx, ty = (x0 + 5 * step_px, y0 + 4 * step_px)
    parts.append(f'''<rect x="{tx}" y="{ty}" width="{step_px}" height="{step_px}" fill="{C['compute']}" stroke="{C['compute']}" stroke-width="2"/>''')
    parts.append(f'''<line x1="{tx + step_px}" y1="{ty + step_px / 2}" x2="{tx + 70}" y2="{ty + 50}" stroke="{C['compute']}" stroke-width="2"/>''')
    parts.append(f'''<rect x="{tx + 60}" y="{ty + 28}" width="310" height="46" rx="4" fill="white" stroke="{C['compute']}" stroke-width="1.5"/>''')
    parts.append(svg_text(tx + 215, ty + 51, 'one 128 × 128 tile', 29, '700', C['compute']))
    parts.append(svg_text(x0 + side / 2, y0 + side + 30, '4,096 = 32 tiles of 128', 29, '700', C['text']))
    parts.append('</svg>')
    return '\n'.join(parts)
(out / 'tiling.svg').write_text(tiling_svg())
TILE = 128
widths = np.arange(64, 321)
util = widths / (TILE * np.ceil(widths / TILE))
fig, ax = plt.subplots(figsize=(3.6, 2.3))
ax.plot(widths, util * 100, color=C['text'], lw=2.5)
W129, PAD129 = (129, 256)
ax.axvspan(W129, PAD129, color=C['danger_l'], zorder=0)
ax.plot([TILE], [100], 'o', ms=10, color=C['text'], zorder=5)
ax.plot([W129], [100 * 0.50390625], 'o', ms=10, color=C['danger'], zorder=5)
ax.text(134, 104, f'{TILE}: 100%', ha='left', fontsize=12, fontweight='bold', color=C['text'])
ax.text(134, 44, '129: 50%', ha='left', fontsize=12, fontweight='bold', color=C['danger'])
ax.set_xlim(64, 320)
ax.set_ylim(0, 118)
ax.set_xticks([64, 128, 192, 256, 320])
ax.set_yticks([0, 50, 100])
ax.set_xlabel(f'Layer width ({TILE}-wide array)')
ax.set_ylabel('Array use (%)')
ax.text(0.99, 0.02, 'Modeled', transform=ax.transAxes, ha='right', va='bottom', fontsize=11, color=C['neutral'])
sp.save(fig, out / 'cliff.svg')
nv_n = np.array([112.0, 120.0, 128.0, 136.0, 144.0, 152.0, 160.0, 168.0, 176.0, 184.0, 192.0, 200.0, 208.0, 216.0, 224.0, 232.0, 240.0, 248.0, 256.0, 264.0, 272.0, 280.0, 288.0, 296.0, 304.0, 312.0, 320.0, 328.0, 336.0, 344.0, 352.0, 360.0, 368.0, 376.0, 384.0, 392.0, 400.0, 408.0, 416.0, 424.0, 432.0, 440.0, 448.0, 456.0, 464.0, 472.0, 480.0, 488.0, 496.0, 504.0, 512.0])
nv_tflops = np.array([167.0, 177.9, 189.7, 148.2, 158.6, 163.7, 174.8, 177.3, 190.1, 192.3, 205.7, 205.4, 219.9, 220.0, 234.4, 232.2, 247.7, 244.7, 263.2, 192.3, 199.7, 202.6, 210.8, 212.5, 219.8, 221.8, 231.6, 231.1, 240.7, 241.5, 227.5, 250.6, 259.5, 259.1, 269.7, 213.7, 219.0, 221.5, 225.4, 227.9, 233.7, 235.0, 242.0, 242.2, 248.9, 248.7, 254.2, 256.0, 263.1, 262.9, 270.6])
N0, N1 = (128.0, 136.0)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    shown = nv_n <= N0 if k == 1 else nv_n <= nv_n[-1]
    ax.plot(nv_n[shown], nv_tflops[shown], color=C['data'], lw=2.5, marker='o', ms=4)
    for edge in (N0, 256.0, 384.0):
        ax.axvline(edge, color=C['neutral'], lw=1, ls=':')
    alpha = 1.0 if k == 2 else 0.0
    ax.annotate('136: 148 TFLOP/s\n6% more work,\n36% more time', (N1, 148.2), xytext=(150, 22), textcoords='data', fontsize=11, fontweight='bold', color=C['danger'], alpha=alpha, arrowprops=dict(arrowstyle='->', color=C['danger'], alpha=alpha))
    ax.annotate('128: 190 TFLOP/s', (N0, 189.7), xytext=(150, 284), textcoords='data', fontsize=11, fontweight='bold', color=C['data'], va='center', arrowprops=dict(arrowstyle='->', color=C['data']))
    ax.text(300, 30, '???', fontsize=16, fontweight='bold', color=C['danger'], ha='center', alpha=1.0 if k == 1 else 0.0)
    ax.set_xlim(104, 520)
    ax.set_ylim(0, 300)
    ax.set_xticks([128, 256, 384, 512])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel('N, output width (tiles 128 wide)')
    ax.set_ylabel('TFLOP/s')
    sp.save(fig, out / f'measured_tiles_step{k}.svg')
BW = 3.35
XLIM = (0.5, 5000)
YLIM = (1, 6000)
PEAK16 = 989.0

def roof_axes(size=(5.8, 2.9)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.minorticks_off()
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1,000'])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(['1', '10', '100', '1,000'])
    ax.grid(color=C['grid'], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel('Arithmetic intensity (FLOP/byte, log scale)')
    ax.set_ylabel('TFLOP/s (log)')
    return (fig, ax)

def slope(ax, top):
    xs = np.logspace(np.log10(XLIM[0]), np.log10(top / BW), 50)
    ax.plot(xs, xs * BW, color=C['memory'], lw=3, zorder=3)

def fp16_roof(ax, where=(0.02, 0.95, 'left'), size=None):
    slope(ax, PEAK16)
    ax.plot([PEAK16 / BW, XLIM[1]], [PEAK16, PEAK16], color=C['compute'], lw=3)
    ax.plot([PEAK16 / BW] * 2, [YLIM[0], 2.2], color=C['neutral'], lw=1.2, ls=':')
    ax.text(PEAK16 / BW / 1.1, 1.3, 'ridge 295', color=C['neutral'], fontsize=size or 10, ha='right')
    ax.text(where[0], where[1], 'FP16 ceiling: 989 TFLOP/s', transform=ax.transAxes, color=C['compute'], fontsize=size or 11, fontweight='bold', ha=where[2], va='top')
fig, ax = roof_axes(size=(4.6, 2.0))
ax.xaxis.label.set_size(13)
ax.yaxis.label.set_size(13)
ax.tick_params(labelsize=13)
fp16_roof(ax, (0.98, 0.3, 'right'), size=13)
ax.plot([563.1], [PEAK16], 'o', ms=11, color=C['compute'], mec='white', mew=1.5, zorder=5)
ax.annotate('ResNet-50 stage-3 conv: 563', (563.1, PEAK16), xytext=(0, 16), textcoords='offset points', ha='right', fontsize=13, fontweight='bold', color=C['compute'])
sp.save(fig, out / 'recall_roof.svg')
layers = [('GPT-2 XL decode, batch 1: 0.3%', 1.0, 'danger', (8, -4), 'left'), ('LayerNorm: 0.7%', 2.0, 'danger', (8, -4), 'left'), ('ResNet-50 FC head: 10.3%', 30.5, 'memory', (8, -4), 'left'), ('ResNet-50 stage-3 conv', 563.1, 'compute', (4, -16), 'left'), ('GPT-2 XL QKV prefill', 1046.7, 'compute', (0, 14), 'center')]
fig, ax = roof_axes()
fp16_roof(ax)
for label, x, color_key, offset, align in layers:
    y = min(PEAK16, BW * x)
    ax.plot([x], [y], 'o', ms=10, color=C[color_key], mec='white', mew=1.5, zorder=5)
    ax.annotate(label, (x, y), xytext=offset, textcoords='offset points', ha=align, va='center', fontsize=11, fontweight='bold', color=C[color_key])
sp.save(fig, out / 'layers_step4.svg')
print('figures written')
