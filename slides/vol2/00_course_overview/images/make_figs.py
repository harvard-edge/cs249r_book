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
    rows = [
        ('AlexNet', 2012, 2.0, False, (8, -3, 'left'), 1),
        ('Megatron-LM', 2019, 512.0, False, (-8, 4, 'right'), 1),
        ('GPT-3 (estimate)', 2020, 10000.0, True, (-10, 0, 'right'), 1),
        ('PaLM', 2022, 6144.0, False, (0, 13, 'center'), 1),
        ('LLaMA 1', 2023, 2048.0, False, (-4, -17, 'right'), 1),
        ('DeepSeek-V3', 2024, 2048.0, False, (4, -17, 'left'), 1),
        ('Llama 3', 2024, 16384.0, False, (9, 4, 'left'), 2),
    ]
    _, slope, intercept = (1.933702688, 0.286389701, -575.7588173)
    xs = np.linspace(2011.5, 2024.5, 100)
    for step in (1, 2):
        fig, ax = sp.plt.subplots(figsize=(5.7, 2.4))
        a2 = 1.0 if step >= 2 else 0.0
        ax.plot(xs, 10 ** (intercept + slope * xs), '--', color=sp.C['neutral'], lw=1.5, alpha=a2)
        ax.text(2021.0, 40, 'fit: about 2× per year', color=sp.C['neutral'], fontsize=12, fontweight='bold', rotation=0, ha='center', alpha=a2)
        for name, yr, n, est, (dx, dy, ha), first in rows:
            a = 1.0 if step >= first else 0.0
            col = sp.C['danger'] if name == 'Llama 3' else sp.C['memory']
            ax.plot([yr], [n], 'o', ms=10, mfc='white' if est else col, mec=col, mew=2, zorder=5, alpha=a)
            ax.annotate(name, (yr, n), xytext=(dx, dy), textcoords='offset points', ha=ha, va='center', fontsize=12, color=col, fontweight='bold', alpha=a)
        ax.set_yscale('log')
        ax.set_ylim(1, 100000.0)
        ax.set_xlim(2011.3, 2026.3)
        ax.set_yticks([1, 10, 100, 1000, 10000])
        ax.set_yticklabels(['1', '10', '100', '1,000', '10,000'])
        ax.set_xticks([2012, 2014, 2016, 2018, 2020, 2022, 2024])
        ax.set_ylabel('Accelerators')
        ax.grid(axis='y', color=sp.C['grid'], linewidth=0.6)
        ax.set_axisbelow(True)
        sp.save(fig, out / f'cluster_counts_step{step}.svg')
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

# BLOOM crash / save-interval timeline (F15)
def _bloom_save_timeline():
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 220" width="900" height="220" font-family="{FONT}">',
        '<rect width="900" height="220" fill="white"/>',
        # Top lane: 8h save interval -> crash at 7:50 loses ~8h
        '<text x="16" y="32" font-size="25" font-weight="700" fill="#0f172a">Before: save every 8 h (250 steps)</text>',
        '<rect x="16" y="44" width="690" height="44" rx="6" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>',
        '<text x="361" y="73" font-size="24" font-weight="700" fill="#b91c1c" text-anchor="middle">7 h 50 m of training lost on crash</text>',
        '<line x1="706" y1="38" x2="706" y2="94" stroke="#dc2626" stroke-width="3.5"/>',
        '<rect x="720" y="44" width="164" height="44" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6 4"/>',
        '<text x="802" y="73" font-size="23" font-weight="700" fill="#dc2626" text-anchor="middle">crash at 7:50</text>',
        # Bottom lane: 4h compromise
        '<text x="16" y="138" font-size="25" font-weight="700" fill="#0f172a">After: save every 4 h (compromise)</text>',
        '<rect x="16" y="150" width="340" height="44" rx="6" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>',
        '<text x="186" y="179" font-size="24" font-weight="700" fill="#15803d" text-anchor="middle">4 h saved (durable)</text>',
        '<rect x="364" y="150" width="342" height="44" rx="6" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>',
        '<text x="535" y="179" font-size="24" font-weight="700" fill="#b45309" text-anchor="middle">at most 4 h at risk</text>',
        '<rect x="720" y="150" width="164" height="44" rx="6" fill="#EAF1F8" stroke="#1F407A" stroke-width="2"/>',
        '<text x="802" y="179" font-size="23" font-weight="700" fill="#14294E" text-anchor="middle">half the loss</text>',
        '</svg>',
    ]
    (out / 'bloom_save_timeline.svg').write_text('\n'.join(svg) + '\n')

# BLOOM 52-node reserve grid (F16): 48 active (blue) + 4 reserve (amber)
def _bloom_reserve_grid():
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 220" width="460" height="220" font-family="{FONT}">',
        '<rect width="460" height="220" fill="white"/>',
        '<text x="230" y="24" font-size="20" font-weight="700" fill="#0f172a" text-anchor="middle">52 nodes of 8 A100 GPUs (416 total)</text>',
    ]
    # 13 cols x 4 rows = 52 nodes; last 4 are reserve
    cw, ch, gap = 28, 28, 6
    x0 = (460 - (13 * cw + 12 * gap)) / 2
    y0 = 40
    for idx in range(52):
        r, c = divmod(idx, 13)
        x = x0 + c * (cw + gap)
        y = y0 + r * (ch + gap)
        is_res = idx >= 48
        fc = '#fef3c7' if is_res else '#EAF1F8'
        ec = '#d97706' if is_res else '#1F407A'
        sw = '2.5' if is_res else '1.6'
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw}" height="{ch}" rx="4" fill="{fc}" stroke="{ec}" stroke-width="{sw}"/>')
    svg.append('<rect x="28" y="186" width="18" height="18" rx="3" fill="#EAF1F8" stroke="#1F407A" stroke-width="1.6"/>')
    svg.append('<text x="54" y="201" font-size="18" font-weight="700" fill="#14294E">48 active (384 GPUs)</text>')
    svg.append('<rect x="252" y="186" width="18" height="18" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="2.5"/>')
    svg.append('<text x="278" y="201" font-size="18" font-weight="700" fill="#b45309">4 reserve (32 GPUs)</text>')
    svg.append('</svg>')
    (out / 'bloom_reserve_grid.svg').write_text('\n'.join(svg) + '\n')

# Coordination barrier visual for F24 col 3 (matches photo aspect beside H100 & switch)
def _coord_barrier():
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 220" width="360" height="220" font-family="{FONT}">',
        '<rect width="360" height="220" fill="#f8fafc" stroke="#7c3aed" stroke-width="2.5" rx="10"/>',
        '<rect x="12" y="12" width="336" height="44" rx="6" fill="#1e293b"/>',
        '<text x="180" y="41" font-family="Courier, monospace" font-size="21" font-weight="700" fill="#e2e8f0" text-anchor="middle"># testing faulty nodes</text>',
        # 4 worker lanes hitting a barrier, one red/replaced
        '<rect x="14" y="68" width="184" height="28" rx="4" fill="#EAF1F8" stroke="#1F407A" stroke-width="1.8"/>',
        '<text x="106" y="88" font-size="20" font-weight="700" fill="#14294E" text-anchor="middle">node 0: ready</text>',
        '<rect x="14" y="102" width="184" height="28" rx="4" fill="#EAF1F8" stroke="#1F407A" stroke-width="1.8"/>',
        '<text x="106" y="122" font-size="20" font-weight="700" fill="#14294E" text-anchor="middle">node 1: ready</text>',
        '<rect x="14" y="136" width="184" height="28" rx="4" fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>',
        '<text x="106" y="156" font-size="20" font-weight="700" fill="#b91c1c" text-anchor="middle">node 2: fault</text>',
        '<rect x="14" y="170" width="184" height="28" rx="4" fill="#dcfce7" stroke="#16a34a" stroke-width="1.8"/>',
        '<text x="106" y="190" font-size="20" font-weight="700" fill="#15803d" text-anchor="middle">spare: swap in</text>',
        '<line x1="212" y1="64" x2="212" y2="202" stroke="#7c3aed" stroke-width="3.5" stroke-dasharray="6 4"/>',
        '<rect x="224" y="96" width="124" height="68" rx="8" fill="#f3e8ff" stroke="#7c3aed" stroke-width="2.2"/>',
        '<text x="286" y="125" font-size="22" font-weight="700" fill="#6d28d9" text-anchor="middle">barrier</text>',
        '<text x="286" y="151" font-size="20" font-weight="700" fill="#6d28d9" text-anchor="middle">resume all</text>',
        '</svg>',
    ]
    (out / 'coord_barrier.svg').write_text('\n'.join(svg) + '\n')

# Course rhythm 3-step visual strip (F26)
def _course_rhythm():
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 175" width="900" height="175" font-family="{FONT}">',
        '<defs><marker id="ra" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>',
        '<rect width="900" height="175" fill="white"/>',
        # Step 1
        '<rect x="14" y="16" width="256" height="142" rx="10" fill="#EAF1F8" stroke="#1F407A" stroke-width="2.5"/>',
        '<circle cx="52" cy="52" r="20" fill="#1F407A"/>',
        '<text x="52" y="59" font-size="22" font-weight="700" fill="white" text-anchor="middle">1</text>',
        '<text x="84" y="60" font-size="26" font-weight="700" fill="#14294E">Read</text>',
        '<text x="32" y="102" font-size="19" fill="#0f172a">Chapter sections before</text>',
        '<text x="32" y="128" font-size="19" fill="#0f172a">class at mlsysbook.ai</text>',
        '<line x1="274" y1="87" x2="314" y2="87" stroke="#475569" stroke-width="3.5" marker-end="url(#ra)"/>',
        # Step 2
        '<rect x="322" y="16" width="256" height="142" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="2.5"/>',
        '<circle cx="360" cy="52" r="20" fill="#d97706"/>',
        '<text x="360" y="59" font-size="22" font-weight="700" fill="white" text-anchor="middle">2</text>',
        '<text x="392" y="60" font-size="26" font-weight="700" fill="#b45309">Act</text>',
        '<text x="340" y="102" font-size="19" fill="#0f172a">Predict, estimate, or</text>',
        '<text x="340" y="128" font-size="19" fill="#0f172a">discuss every 15 min</text>',
        '<line x1="582" y1="87" x2="622" y2="87" stroke="#475569" stroke-width="3.5" marker-end="url(#ra)"/>',
        # Step 3
        '<rect x="630" y="16" width="256" height="142" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2.5"/>',
        '<circle cx="668" cy="52" r="20" fill="#16a34a"/>',
        '<text x="668" y="59" font-size="22" font-weight="700" fill="white" text-anchor="middle">3</text>',
        '<text x="700" y="60" font-size="26" font-weight="700" fill="#15803d">Defend</text>',
        '<text x="648" y="102" font-size="19" fill="#0f172a">Check napkin math</text>',
        '<text x="648" y="128" font-size="19" fill="#0f172a">against real fleet data</text>',
        '</svg>',
    ]
    (out / 'course_rhythm.svg').write_text('\n'.join(svg) + '\n')

_bloom_save_timeline()
_bloom_reserve_grid()
_coord_barrier()
_course_rhythm()
print('figures written')
