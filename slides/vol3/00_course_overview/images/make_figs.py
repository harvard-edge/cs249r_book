#!/usr/bin/env python3
"""Figures for Lecture 0 (Volume III course overview).

Every value comes from calcs.py, which reads MLSysIM:
  - SWE-bench Verified leaderboard submissions
    (Scenarios.SweBenchVerifiedLeaderboardOneAttempt over
    slides/vol3/00_course_overview/images/data/swe_bench_verified_leaderboard.csv)
  - The django__django-12050 run of mini-SWE-agent 1.0.0 with Claude 4 Sonnet
    (vol3/01_introduction/data/swe_bench_verified_mini_v1_sonnet4_django_12050_calls.csv)
This file types only dates, layout, and label offsets.

Run from this folder: python3 make_figs.py   (then make deck converts SVG to PDF)
"""
import datetime as dt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
plt = sp.plt
C = sp.C
OUT = Path(__file__).parent

def trend_plot():
    """Resolved share of SWE-bench Verified, one-attempt submissions, full width."""
    pts = [(dt.date.fromisoformat('2023-10-10'), 4.4), (dt.date.fromisoformat('2023-10-10'), 0.4), (dt.date.fromisoformat('2023-10-10'), 1.2), (dt.date.fromisoformat('2023-10-10'), 1.4), (dt.date.fromisoformat('2024-04-02'), 15.8), (dt.date.fromisoformat('2024-04-02'), 22.4), (dt.date.fromisoformat('2024-06-20'), 33.6), (dt.date.fromisoformat('2024-07-28'), 23.2), (dt.date.fromisoformat('2024-10-28'), 38.8), (dt.date.fromisoformat('2024-11-25'), 51.8), (dt.date.fromisoformat('2025-03-06'), 32.8), (dt.date.fromisoformat('2025-05-11'), 40.2), (dt.date.fromisoformat('2025-05-15'), 70.4), (dt.date.fromisoformat('2025-05-19'), 70.2), (dt.date.fromisoformat('2025-05-20'), 46.8), (dt.date.fromisoformat('2025-05-22'), 66.6), (dt.date.fromisoformat('2025-05-22'), 73.2), (dt.date.fromisoformat('2025-05-22'), 72.4), (dt.date.fromisoformat('2025-05-24'), 70.4), (dt.date.fromisoformat('2025-06-03'), 74.4), (dt.date.fromisoformat('2025-06-10'), 70.4), (dt.date.fromisoformat('2025-06-11'), 70.8), (dt.date.fromisoformat('2025-06-16'), 38.0), (dt.date.fromisoformat('2025-06-23'), 71.0), (dt.date.fromisoformat('2025-06-27'), 23.2), (dt.date.fromisoformat('2025-06-29'), 42.2), (dt.date.fromisoformat('2025-07-10'), 71.2), (dt.date.fromisoformat('2025-07-15'), 71.2), (dt.date.fromisoformat('2025-07-16'), 65.4), (dt.date.fromisoformat('2025-07-20'), 74.6), (dt.date.fromisoformat('2025-07-25'), 38.0), (dt.date.fromisoformat('2025-07-31'), 74.8), (dt.date.fromisoformat('2025-08-04'), 53.4), (dt.date.fromisoformat('2025-08-07'), 71.8), (dt.date.fromisoformat('2025-09-01'), 52.2), (dt.date.fromisoformat('2025-09-15'), 74.6), (dt.date.fromisoformat('2025-09-24'), 57.0), (dt.date.fromisoformat('2025-10-14'), 71.2), (dt.date.fromisoformat('2025-11-03'), 74.8), (dt.date.fromisoformat('2025-11-10'), 53.6), (dt.date.fromisoformat('2025-11-10'), 45.0), (dt.date.fromisoformat('2025-11-20'), 77.4), (dt.date.fromisoformat('2025-12-05'), 79.2), (dt.date.fromisoformat('2025-12-15'), 79.2), (dt.date.fromisoformat('2026-09-01'), 71.8)]
    front = [(dt.date.fromisoformat('2023-10-10'), 4.4), (dt.date.fromisoformat('2024-04-02'), 15.8), (dt.date.fromisoformat('2024-04-02'), 22.4), (dt.date.fromisoformat('2024-06-20'), 33.6), (dt.date.fromisoformat('2024-10-28'), 38.8), (dt.date.fromisoformat('2024-11-25'), 51.8), (dt.date.fromisoformat('2025-05-15'), 70.4), (dt.date.fromisoformat('2025-05-22'), 73.2), (dt.date.fromisoformat('2025-06-03'), 74.4), (dt.date.fromisoformat('2025-07-20'), 74.6), (dt.date.fromisoformat('2025-07-31'), 74.8), (dt.date.fromisoformat('2025-11-20'), 77.4), (dt.date.fromisoformat('2025-12-05'), 79.2)]
    end = dt.date(2026, 3, 1)
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=C['neutral_l'], edgecolors=C['neutral'], linewidths=0.6, zorder=3)
    fx = [p[0] for p in front] + [end]
    fy = [p[1] for p in front] + [front[-1][1]]
    ax.step(fx, fy, where='post', color=C['compute'], lw=2.6, zorder=4)
    first, best = (front[0], front[-1])
    ax.annotate('Oct 2023: 4.4%', first, xytext=(dt.date(2023, 8, 20), 88), textcoords='data', fontsize=13, fontweight='bold', color=C['compute'], arrowprops=dict(arrowstyle='-', color=C['compute'], lw=1))
    ax.annotate('Dec 2025: 79.2%', best, xytext=(-150, 14), textcoords='offset points', fontsize=13, fontweight='bold', color=C['compute'], arrowprops=dict(arrowstyle='-', color=C['compute'], lw=1))
    ax.set_ylim(0, 100)
    ax.set_xlim(dt.date(2023, 8, 1), end)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([dt.date(y, 1, 1) for y in (2024, 2025, 2026)])
    ax.set_xticklabels(['2024', '2025', '2026'])
    ax.set_ylabel('Issues resolved (%)')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / 'trend.svg')

def loop_bars():
    """Same model, retrieval baseline (no loop) vs agent loop; two steps."""
    pairs = [{'model': 'GPT-4 (1106)', 'one': 14.0, 'loop': 112.0}, {'model': 'Claude 3 Opus', 'one': 35.0, 'loop': 79.0}]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        y = [1.0, 0.0]
        for yi, p in zip(y, pairs):
            ax.barh(yi + 0.2, p['one'], height=0.36, color=C['neutral_l'], edgecolor=C['neutral'])
            ax.text(p['one'] + 2, yi + 0.2, f"{p['one']:.0f}  one call", va='center', fontsize=14, color=C['text'])
            if step == 2:
                ax.barh(yi - 0.2, p['loop'], height=0.36, color=C['compute'])
                ax.text(p['loop'] + 2, yi - 0.2, f"{p['loop']:.0f}  loop", va='center', fontsize=14, fontweight='bold', color=C['compute'])
            else:
                ax.text(4, yi - 0.2, 'in a loop: ?', va='center', fontsize=14, color=C['danger'])
        ax.set_yticks(y)
        ax.set_yticklabels([p['model'] for p in pairs], fontsize=14)
        ax.set_xlim(0, 150)
        ax.set_xlabel('Issues resolved, of 500', fontsize=13)
        ax.tick_params(axis='x', labelsize=12)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(axis='x', color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, OUT / f'loop_bars_step{step}.svg')
calls_rows = ({'call': 1.0, 'context_chars': 5499.0, 'cum_sent_chars': 5499.0, 'cum_new_chars': 5499.0}, {'call': 2.0, 'context_chars': 6621.0, 'cum_sent_chars': 12120.0, 'cum_new_chars': 6621.0}, {'call': 3.0, 'context_chars': 7159.0, 'cum_sent_chars': 19279.0, 'cum_new_chars': 7159.0}, {'call': 4.0, 'context_chars': 9106.0, 'cum_sent_chars': 28385.0, 'cum_new_chars': 9106.0}, {'call': 5.0, 'context_chars': 9692.0, 'cum_sent_chars': 38077.0, 'cum_new_chars': 9692.0}, {'call': 6.0, 'context_chars': 10236.0, 'cum_sent_chars': 48313.0, 'cum_new_chars': 10236.0}, {'call': 7.0, 'context_chars': 12094.0, 'cum_sent_chars': 60407.0, 'cum_new_chars': 12094.0}, {'call': 8.0, 'context_chars': 14809.0, 'cum_sent_chars': 75216.0, 'cum_new_chars': 14809.0}, {'call': 9.0, 'context_chars': 15435.0, 'cum_sent_chars': 90651.0, 'cum_new_chars': 15435.0}, {'call': 10.0, 'context_chars': 16136.0, 'cum_sent_chars': 106787.0, 'cum_new_chars': 16136.0}, {'call': 11.0, 'context_chars': 17117.0, 'cum_sent_chars': 123904.0, 'cum_new_chars': 17117.0}, {'call': 12.0, 'context_chars': 20451.0, 'cum_sent_chars': 144355.0, 'cum_new_chars': 20451.0}, {'call': 13.0, 'context_chars': 21093.0, 'cum_sent_chars': 165448.0, 'cum_new_chars': 21093.0}, {'call': 14.0, 'context_chars': 21488.0, 'cum_sent_chars': 186936.0, 'cum_new_chars': 21488.0}, {'call': 15.0, 'context_chars': 21952.0, 'cum_sent_chars': 208888.0, 'cum_new_chars': 21952.0}, {'call': 16.0, 'context_chars': 23109.0, 'cum_sent_chars': 231997.0, 'cum_new_chars': 23109.0}, {'call': 17.0, 'context_chars': 24327.0, 'cum_sent_chars': 256324.0, 'cum_new_chars': 24327.0}, {'call': 18.0, 'context_chars': 25886.0, 'cum_sent_chars': 282210.0, 'cum_new_chars': 25886.0}, {'call': 19.0, 'context_chars': 27813.0, 'cum_sent_chars': 310023.0, 'cum_new_chars': 27813.0}, {'call': 20.0, 'context_chars': 28204.0, 'cum_sent_chars': 338227.0, 'cum_new_chars': 28204.0}, {'call': 21.0, 'context_chars': 29176.0, 'cum_sent_chars': 367403.0, 'cum_new_chars': 29176.0}, {'call': 22.0, 'context_chars': 29594.0, 'cum_sent_chars': 396997.0, 'cum_new_chars': 29594.0})

def context_bars():
    """Characters each model call read in the real django run; column slot."""
    calls = [int(r['call']) for r in calls_rows]
    chars = [r['context_chars'] / 1000.0 for r in calls_rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.bar(calls, chars, color=C['memory'], width=0.75)
    ax.annotate('call 1: 5.5k', (calls[0], chars[0]), xytext=(-8, 26), textcoords='offset points', ha='left', fontsize=13, fontweight='bold', color=C['memory'])
    ax.annotate('call 22: 29.6k', (calls[-1], chars[-1]), xytext=(4, 6), textcoords='offset points', ha='right', fontsize=13, fontweight='bold', color=C['memory'])
    ax.set_xlabel('Model call', fontsize=13)
    ax.set_ylabel('Characters read (thousands)', fontsize=13)
    ax.set_xticks([1, 5, 10, 15, 20, calls[-1]])
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, 35)
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, OUT / 'context_bars.svg')
if __name__ == '__main__':
    trend_plot()
    loop_bars()
    context_bars()
    print('figures written')
