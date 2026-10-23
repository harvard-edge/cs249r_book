#!/usr/bin/env python3
"""Figures for Volume II, Lecture 0 (course overview).

Every value comes from calcs.py, which reads MLSysIM; this file types only
years, layout, and label offsets.

  - GPT-3: training work over one A100's peak, against the published run's
    days (Models.Language.GPT3; Patterson et al. 2021). Introduction Sec. 1.1.
  - Landmark accelerator counts: Literature.TrainingRuns and the model
    registry, as plotted in the Introduction's cluster-size figure (Sec. 1.1,
    figure 2); the fit is Scenarios.LandmarkClusterGrowth2012to2024. GPT-3's
    point is an estimate (Langston 2020).
  - The stack with each lecture on its row (kit stack_build8 canvas).

Run from this folder: python3 make_figs.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
out = Path(__file__).parent
FONT = 'Helvetica, Arial, sans-serif'
sp.hbars_steps(out / 'gpt3_days', [('One A100', 11648.3, 'neutral', '32 years'), ('10,000 V100s', 14.8, 'memory', '15 days')], 'days', xlabel='GPT-3 training time (days, log)', size=(3.8, 2.2))

def cluster_counts():
    rows = [('AlexNet', 2012, 2.0, False, (8, -3, 'left')), ('Megatron-LM', 2019, 512.0, False, (-8, 4, 'right')), ('GPT-3 (estimate)', 2020, 10000.0, True, (-10, 0, 'right')), ('PaLM', 2022, 6144.0, False, (0, 13, 'center')), ('LLaMA 1', 2023, 2048.0, False, (-4, -17, 'right')), ('DeepSeek-V3', 2024, 2048.0, False, (4, -17, 'left')), ('Llama 3', 2024, 16384.0, False, (9, 4, 'left'))]
    fig, ax = sp.plt.subplots(figsize=(5.7, 2.4))
    _, slope, intercept = (1.933702688, 0.286389701, -575.7588173)
    xs = np.linspace(2011.5, 2024.5, 100)
    ax.plot(xs, 10 ** (intercept + slope * xs), '--', color=sp.C['neutral'], lw=1.5)
    ax.text(2021.0, 40, 'fit: about 2× per year', color=sp.C['neutral'], fontsize=12, fontweight='bold', rotation=0, ha='center')
    for name, yr, n, est, (dx, dy, ha) in rows:
        ax.plot([yr], [n], 'o', ms=10, mfc='white' if est else sp.C['memory'], mec=sp.C['memory'], mew=2, zorder=5)
        ax.annotate(name, (yr, n), xytext=(dx, dy), textcoords='offset points', ha=ha, va='center', fontsize=12, color=sp.C['memory'], fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1, 100000.0)
    ax.set_xlim(2011.3, 2026.3)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(['1', '10', '100', '1,000', '10,000'])
    ax.set_xticks([2012, 2014, 2016, 2018, 2020, 2022, 2024])
    ax.set_ylabel('Accelerators')
    ax.grid(axis='y', color=sp.C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'cluster_counts.svg')
cluster_counts()
kit = Path(__file__).resolve().parents[3] / 'assets' / 'kit' / 'vol2'
stack_svg = (kit / 'stack_build8.svg').read_text()
ACCENT = '#1F407A'
ROW_Y = {'Facility': 418.25, 'Fabric': 362.5, 'Execution': 306.75, 'Control': 251.0, 'Serving': 195.25, 'Operations': 139.5, 'Assurance': 83.75, 'Governance': 28.0}
ON_ROW = {'Facility': ['L2'], 'Fabric': ['L3', 'L4'], 'Execution': ['L5', 'L6'], 'Control': ['L7', 'L8'], 'Serving': ['L9', 'L10', 'L11'], 'Operations': ['L12'], 'Assurance': ['L13', 'L14'], 'Governance': ['L15', 'L16']}

def badge(x, y, w, h, label, size=26):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="#FFFFFF" stroke="{ACCENT}" stroke-width="1.5"/>\n<text x="{x + w / 2}" y="{y + h / 2}" font-family="{FONT}" font-size="{size}" font-weight="700" fill="{ACCENT}" text-anchor="middle" dominant-baseline="central">{label}</text>\n'
badges = ''
for row, labels in ON_ROW.items():
    right = 710
    for label in reversed(labels):
        w = 66 if len(label) <= 2 else 80
        badges += badge(right - w, ROW_Y[row] + 4, w, 40, label)
        right -= w + 6
(out / 'stack_lectures.svg').write_text(stack_svg.replace('</svg>', badges + '</svg>'))
print('figures written')
