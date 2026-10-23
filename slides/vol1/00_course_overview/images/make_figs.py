#!/usr/bin/env python3
"""Data plots for Lecture 0 (course overview).

Every value comes from calcs.py, which reads MLSysIM:
  - Google's measured median Gemini text prompt (local CSV series.
    GeminiPromptEnergy) against charging a phone (ReferenceStats.EnergyAnchors).
  - One day of ChatGPT prompts against training GPT-3 once
    (Scenarios.ChatGptDayAtGeminiMedianPrompt).
  - Datacenter electricity, 2024 and the 2030 projection
    (Literature.DatacenterElectricity, IEA).
  - Deployment power by tier (the Lecture 2 power ladder).
  - The stack with each lecture on its layer (kit stack_build8 canvas).

Run from this folder: python3 make_figs.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from matplotlib.ticker import FuncFormatter
out = Path(__file__).parent
_save = sp.save

def _save_with_commas(fig, path):
    for ax in fig.axes:
        if ax.get_xscale() == 'linear':
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    _save(fig, path)
sp.save = _save_with_commas
sp.hbars_steps(out / 'energy_answer', [('One prompt (Google, measured)', 864.0, 'compute', '864 J'), ('Charging a phone', 40000.0, 'neutral', '40,000 J = 46 prompts')], 'J', log=False, xlabel='Energy (joules)')
sp.hbars_steps(out / 'fleet_daily', [('Chat prompts, one day (estimate)', 600.0, 'compute', 'about 600 MWh'), ('Training GPT-3, once', 1287.0, 'neutral', '1,287 MWh')], 'MWh', log=False, xlabel='Energy (megawatt-hours)', size=sp.SHORT)
sp.hbars_steps(out / 'datacenter_electricity', [('2024', 415.0, 'neutral'), ('2030 (projected)', 945.0, 'danger')], 'TWh', log=False, xlabel='Datacenter electricity per year (TWh)', size=sp.SHORT)
TIER_LABEL = {'Cloud': '3 MW', 'Edge': '200 W', 'Mobile': '5 W', 'TinyML': '1 mW'}
with sp.plt.rc_context({'xtick.labelsize': 14}):
    sp.hbars_steps(out / 'power_tiers', [('Cloud', 3000000.0, 'neutral', '3 MW'), ('Edge', 200, 'neutral', '200 W'), ('Mobile', 5.0, 'neutral', '5 W'), ('TinyML', 0.001, 'neutral', '1 mW')], 'W', xlabel='Power (watts, log scale)', value_fmt='{:g}')
kit_vol1 = Path(__file__).resolve().parents[3] / 'assets' / 'kit' / 'vol1'
stack_svg = (kit_vol1 / 'stack_build8.svg').read_text()
ACCENT = '#A51C30'
LAYER_Y = {'Hardware': 412, 'Frameworks': 348, 'Models': 284, 'Training': 220, 'Serving': 156, 'Operations': 92, 'Applications': 28}
ON_LAYER = {'Hardware': ['L11', 'L11b', 'L12', 'L12b'], 'Frameworks': ['L7'], 'Models': ['L5', 'L6', 'L6b', 'L10', 'L10b'], 'Training': ['L8', 'L8b'], 'Serving': ['L13', 'L13b'], 'Operations': ['L14', 'L14b'], 'Applications': ['L15', 'L15b']}

def badge(x, y, w, h, label, size=28):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="#FFFFFF" stroke="{ACCENT}" stroke-width="1.5"/>\n<text x="{x + w / 2}" y="{y + h / 2}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" font-weight="700" fill="{ACCENT}" text-anchor="middle" dominant-baseline="central">{label}</text>\n'
badges = ''
for layer, labels in ON_LAYER.items():
    y0 = LAYER_Y[layer]
    right = 688
    for label in reversed(labels):
        w = 72 if len(label) <= 3 else 88
        badges += badge(right - w, y0 + 4, w, 46, label)
        right -= w + 6
for label, y in (('L4', 292), ('L4b', 348), ('L9', 404)):
    badges += badge(715, y, 54, 46, label, 26 if len(label) <= 2 else 21)
(out / 'stack_lectures.svg').write_text(stack_svg.replace('</svg>', badges + '</svg>'))
print('figures written')
