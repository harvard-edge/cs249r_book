#!/usr/bin/env python3
"""Figures for Volume IV, Lecture 0 (Course overview).

Run:  python3 make_figs.py          (the figures this deck shows)
Writes SVGs here; `make deck V=4 D=00_course_overview` converts them to PDF.
Every value comes from calcs.py, which reads MLSysIM; this file types only
layout, axis ticks, and label offsets.

The stopping-budget bar and the record chain live in the Volume IV kit
(slides/assets/kit/vol4); Lecture 0 shows neither, and Lecture 1 owns both.
"""
import sys
from datetime import date
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'mlsysim'))
import matplotlib.pyplot as plt
import slideplot as sp
AMBER = '#D97706'

def out(name):
    return str(HERE / name)

def _year(ym):
    y, m = (int(s) for s in ym.split('-'))
    d = date(y, m, 28)
    return y + (d.timetuple().tm_yday - 1) / 365

def waymo_miles(step):
    rows = [('2024-06', 22.199), ('2024-07', 25.308), ('2024-09', 33.103), ('2024-12', 50.083), ('2025-03', 71.62), ('2025-06', 95.965), ('2025-09', 127.158), ('2025-12', 170.712), ('2026-03', 220.613), ('2026-06', 271.329)]
    xs = [_year(d) for d, _ in rows]
    ys = [m for _, m in rows]
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(xs, ys, 'o-', color=sp.C['data'], lw=2.5, ms=6, zorder=3)
    ax.text(xs[0], ys[0] + 22, '22.2 million\nJune 2024', color=sp.C['data'], fontsize=12, fontweight='bold', ha='left', va='bottom')
    ax.text(xs[-1] - 0.06, ys[-1], '271.3 million\nJune 2026', color=sp.C['data'], fontsize=12, fontweight='bold', ha='right', va='center')
    if step >= 2:
        d = dict(rows)
        x0, x1 = (_year('2025-06'), _year('2026-06'))
        ax.plot([x0, x0], [0, d['2025-06']], color=AMBER, lw=1.5, ls='--')
        ax.plot([x1, x1], [0, d['2026-06']], color=AMBER, lw=1.5, ls='--')
        ax.annotate('', xy=(x1, 30), xytext=(x0, 30), arrowprops=dict(arrowstyle='<->', color=AMBER, lw=1.8))
        ax.text((x0 + x1) / 2, 40, '175 million miles\nin 12 months', color=AMBER, fontsize=12, fontweight='bold', ha='center', va='bottom')
    ax.set_xlim(2024.25, 2026.6)
    ax.set_ylim(0, 300)
    ax.set_xticks([2024.5, 2025, 2025.5, 2026, 2026.5])
    ax.set_xticklabels(['mid 2024', '2025', 'mid 2025', '2026', 'mid 2026'])
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel('Driverless miles,\ncumulative (millions)')
    ax.grid(axis='y', color=sp.C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out(f'waymo_miles_step{step}.svg'))

def main():
    for k in (1, 2):
        waymo_miles(k)
if __name__ == '__main__':
    main()
