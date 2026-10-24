#!/usr/bin/env python3
"""Figures for Lecture 14 (Supervisory Intervention).

Every value comes from MLSysIM through calcs.py (imported as K); printed
numbers use the formatted strings calcs.py records (deckvals.val). "SI" is the
rendered chapter, Supervisory Intervention (Volume IV, Chapter 14). Data:
  - California DMV disengagement filings, Waymo 2015 to 2025
    (vol4/14_intervention/data/ca_dmv_disengagements.csv; SI 14.1 figure 1).
  - Zhang et al. 2019, 520 mean take-over times (slides/vol4/14_intervention/images/data/zhang2019_mean_takeover_times.csv).
  - Lufthansa 2904 timeline (cited literature; SI War Story 14.1).
Diagram colors follow the Volume IV causal-boundary palette
(rules/volumes/vol4/palette.md); red only for veto, hazard, and limits.
Figures are drawn at the size of their slot: full-width SVG 900 px wide with
text 18 px or larger; column SVG about 560 px wide with text 20 px or larger.

Run from this folder: python3 make_figs.py
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
from slideplot import plt
NAVY, BLUE, GREEN, RED, AMBER, SLATE, PETROL = ('#1F407A', '#2563EB', '#059669', '#DC2626', '#D97706', '#334155', '#0F766E')
GREEN_F, BLUE_F, RED_F, SLATE_F, AMBER_F = ('#DCFCE7', '#EFF6FF', '#FEE2E2', '#F8FAFC', '#FEF3C7')
GRAY, LGRAY = ('#94A3B8', '#E2E8F0')
TXT = '#1A1A2E'
FONT = 'Helvetica, Arial, sans-serif'

def save(fig, name):
    fig.savefig(HERE / f'{name}.svg', format='svg', bbox_inches='tight', pad_inches=0.08, facecolor='white')
    plt.close(fig)

def _marker(mid, color):
    return f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
DEFS = '<defs>' + ''.join((_marker(m, c) for m, c in [('ah', SLATE), ('ahg', GREEN), ('ahb', BLUE), ('ahr', RED), ('ahn', NAVY), ('aha', AMBER)])) + f'<pattern id="hatch" patternUnits="userSpaceOnUse" width="10" height="10" patternTransform="rotate(45)"><rect width="10" height="10" fill="{RED_F}"/><line x1="0" y1="0" x2="0" y2="10" stroke="{RED}" stroke-width="3"/></pattern></defs>'

def svg(name, w, h, body):
    (HERE / f'{name}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">\n{DEFS}\n<rect width="{w}" height="{h}" fill="white"/>\n' + '\n'.join(body) + '\n</svg>\n')

def box(x, y, w, h, fill, stroke, dash=None, rx=8, sw=2.5):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

def text(x, y, s, size=20, fill=TXT, anchor='middle', weight='normal', italic=False):
    st = ' font-style="italic"' if italic else ''
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{st}>{s}</text>'

def lines(x, y, rows, size=20, fill=TXT, anchor='middle', weight='normal', gap=None):
    gap = gap or size * 1.25
    return [text(x, y + i * gap, r, size, fill, anchor, weight) for i, r in enumerate(rows)]

def arrow(x1, y1, x2, y2, color=SLATE, sw=3, dash=None, marker='ah'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{marker})"/>'

def line(x1, y1, x2, y2, color=SLATE, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>'

def bracket(x1, x2, y, label, color=SLATE, size=19, up=True, weight='bold'):
    t = -8 if up else 8
    ty = y - 12 if up else y + 28
    return [f'<path d="M{x1},{y - t} L{x1},{y} L{x2},{y} L{x2},{y - t}" fill="none" stroke="{color}" stroke-width="2.5"/>', text((x1 + x2) / 2, ty, label, size, color, weight=weight)]

def dmv():
    rows = ({'company': 'Waymo', 'report_year': 2015.0, 'reporting_period': 'Sep 2014 - Nov 2015 (first report)', 'permit_type': 'drivered_AVT', 'filer_type': 'first_report', 'autonomous_miles': 424331.0, 'disengagements': 341.0, 'miles_per_disengagement': 1244.4, 'source_method': 'OCR of scanned PDF; stated totals (424,331 public-road CA autonomous miles; 272 failure-detection + 69 safe-operation disengagements)'}, {'company': 'Waymo', 'report_year': 2016.0, 'reporting_period': 'Dec 2015 - Nov 2016', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 635868.0, 'disengagements': 124.0, 'miles_per_disengagement': 5128.0, 'source_method': 'OCR of scanned PDF, Table 1 "Comparison of Reporting Year Data"'}, {'company': 'Waymo', 'report_year': 2017.0, 'reporting_period': 'Dec 2016 - Nov 2017', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 352544.6, 'disengagements': 63.0, 'miles_per_disengagement': 5595.9, 'source_method': 'PDF text layer, Table 1 totals'}, {'company': 'Waymo', 'report_year': 2018.0, 'reporting_period': 'Dec 2017 - Nov 2018', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 1255997.4, 'disengagements': 114.0, 'miles_per_disengagement': 11017.5, 'source_method': 'OCR of scanned PDF Section 3; sum of per-VIN ANNUAL TOTAL (111 VINs); disengagement column sum = Section 2 event rows = 114'}, {'company': 'Waymo', 'report_year': 2019.0, 'reporting_period': 'December 2018 to November 2019', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 1454137.32, 'disengagements': 110.0, 'miles_per_disengagement': 13219.4, 'source_method': 'stdlib xlsx parse: sum of Miles-sheet ANNUAL TOTAL and ANNUAL TOTAL OF DISENGAGEMENTS columns'}, {'company': 'Waymo', 'report_year': 2019.0, 'reporting_period': 'October 2018 to November 2019', 'permit_type': 'driverless_AVDT', 'filer_type': 'annual', 'autonomous_miles': 0.0, 'disengagements': 0.0, 'miles_per_disengagement': None, 'source_method': 'stdlib xlsx parse: sum of Miles-sheet ANNUAL TOTAL and ANNUAL TOTAL OF DISENGAGEMENTS columns'}, {'company': 'Waymo', 'report_year': 2020.0, 'reporting_period': 'Dec 2019 - Nov 2020', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 628838.5, 'disengagements': 21.0, 'miles_per_disengagement': 29944.7, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}, {'company': 'Waymo', 'report_year': 2021.0, 'reporting_period': 'Dec 2020 - Nov 2021', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 2325842.9, 'disengagements': 292.0, 'miles_per_disengagement': 7965.2, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}, {'company': 'Waymo', 'report_year': 2022.0, 'reporting_period': 'Dec 2021 - Nov 2022', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 2900143.8, 'disengagements': 170.0, 'miles_per_disengagement': 17059.7, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}, {'company': 'Waymo', 'report_year': 2022.0, 'reporting_period': 'Dec 2021 - Nov 2022', 'permit_type': 'driverless_AVDT', 'filer_type': 'annual', 'autonomous_miles': 51639.2, 'disengagements': 0.0, 'miles_per_disengagement': None, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}, {'company': 'Waymo', 'report_year': 2023.0, 'reporting_period': 'Dec 2022 - Nov 2023', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 3669962.4, 'disengagements': 212.0, 'miles_per_disengagement': 17311.1, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}, {'company': 'Waymo', 'report_year': 2023.0, 'reporting_period': 'Dec 2022 - Nov 2023', 'permit_type': 'driverless_AVDT', 'filer_type': 'annual', 'autonomous_miles': 1188928.4, 'disengagements': 14.0, 'miles_per_disengagement': 84923.5, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}, {'company': 'Waymo', 'report_year': 2024.0, 'reporting_period': 'Dec 2023 - Nov 2024', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 2389565.2, 'disengagements': 244.0, 'miles_per_disengagement': 9793.3, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}, {'company': 'Waymo', 'report_year': 2024.0, 'reporting_period': 'Dec 2023 - Nov 2024', 'permit_type': 'driverless_AVDT', 'filer_type': 'annual', 'autonomous_miles': 516572.1, 'disengagements': 22.0, 'miles_per_disengagement': 23480.5, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}, {'company': 'Waymo', 'report_year': 2025.0, 'reporting_period': 'Dec 2024 - Nov 2025', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 3346708.5, 'disengagements': 174.0, 'miles_per_disengagement': 19234.0, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}, {'company': 'Waymo', 'report_year': 2025.0, 'reporting_period': 'Dec 2024 - Nov 2025', 'permit_type': 'driverless_AVDT', 'filer_type': 'annual', 'autonomous_miles': 3877999.8, 'disengagements': 54.0, 'miles_per_disengagement': 71814.8, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'})
    sd = sorted(((r['report_year'], r['miles_per_disengagement']) for r in rows if r['permit_type'] == 'drivered_AVT' and r['miles_per_disengagement']))
    dl = sorted(((r['report_year'], r['miles_per_disengagement']) for r in rows if r['permit_type'] == 'driverless_AVDT' and r['miles_per_disengagement']))
    last = {'company': 'Waymo', 'report_year': 2025.0, 'reporting_period': 'Dec 2024 - Nov 2025', 'permit_type': 'drivered_AVT', 'filer_type': 'annual', 'autonomous_miles': 3346708.5, 'disengagements': 174.0, 'miles_per_disengagement': 19234.0, 'source_method': 'DMV consolidated CSV; per-VIN rows grouped by permit number; miles = sum of ANNUAL TOTAL; disengagements = sum of "Annual Total of Disengagements"'}
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.6, 2.55))
        pts = [p for p in sd if step == 2 or p[0] <= 2020]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', color=sp.C['memory'], lw=2.5, ms=6)
        ax.text(sd[0][0] + 0.25, sd[0][1] * 0.82, f'{sd[0][1]:,.0f}', color=sp.C['memory'], fontsize=12, va='top', fontweight='bold')
        y2020 = dict(sd)[2020]
        ax.text(2020, y2020 * 1.35, f'{y2020:,.0f}', color=sp.C['memory'], fontsize=12, ha='center', fontweight='bold')
        ax.text(2016.6, 22000.0, 'with a safety driver', color=sp.C['memory'], fontsize=12, fontweight='bold', ha='center')
        if step == 1:
            ax.text(2023.5, 1100.0, '2021 to 2025?', color=sp.C['neutral'], fontsize=13, ha='center', fontweight='bold')
        else:
            y2021 = dict(sd)[2021]
            ax.text(2021.15, y2021 * 0.62, f'{y2021:,.0f}', color=sp.C['memory'], fontsize=12, ha='center', va='top', fontweight='bold')
            ax.text(2025.25, sd[-1][1] * 0.9, f"{sd[-1][1]:,.0f}\n{last['disengagements']:.0f} takeovers", color=sp.C['memory'], fontsize=12, va='top', fontweight='bold')
            ax.plot([p[0] for p in dl], [p[1] for p in dl], 's-', color=sp.C['data'], lw=2.5, ms=7)
            ax.text(2025.25, dl[-1][1] * 1.05, f'{dl[-1][1]:,.0f}', color=sp.C['data'], fontsize=12, va='bottom', fontweight='bold')
            ax.text(2022.6, 125000.0, 'driverless testing', color=sp.C['data'], fontsize=12, fontweight='bold', ha='right')
        ax.set_yscale('log')
        ax.set_ylim(600, 250000.0)
        ax.set_yticks([1000.0, 10000.0, 100000.0])
        ax.set_yticklabels(['1,000', '10,000', '100,000'])
        ax.minorticks_off()
        ax.set_xlim(2014.5, 2027.7)
        ax.set_xticks([2015, 2017, 2019, 2021, 2023, 2025])
        ax.set_ylabel('Miles per disengagement')
        ax.grid(axis='y', color=sp.C['grid'], lw=0.6)
        save(fig, f'dmv_step{step}')

def propose_admit():
    b = []
    for col, (x0, title) in enumerate([(0, 'Enforcement'), (455, 'Intervention')]):
        b.append(text(x0 + 222, 34, title, 28, TXT, weight='bold'))
        if col == 0:
            b.append(box(x0 + 15, 56, 205, 80, BLUE_F, NAVY))
            b += lines(x0 + 117, 90, ['Learned', 'policy'], 22, NAVY, weight='bold', gap=26)
            b.append(box(x0 + 235, 56, 195, 80, 'white', GRAY, dash='6,4'))
            b += lines(x0 + 332, 90, ['no', 'operator'], 22, GRAY, gap=26)
            b.append(arrow(x0 + 117, 138, x0 + 117, 188, BLUE, dash='7,5', marker='ahb'))
            b.append(text(x0 + 135, 172, 'proposal', 21, BLUE, anchor='start', italic=True))
        else:
            b.append(box(x0 + 15, 56, 205, 80, 'white', GRAY, dash='6,4'))
            b += lines(x0 + 117, 90, ['Policy:', 'token revoked'], 22, GRAY, gap=26)
            b.append(box(x0 + 235, 56, 195, 80, BLUE_F, BLUE))
            b += lines(x0 + 332, 90, ['Authenticated', 'operator'], 22, BLUE, weight='bold', gap=26)
            b.append(arrow(x0 + 332, 138, x0 + 332, 188, BLUE, dash='7,5', marker='ahb'))
            b.append(text(x0 + 314, 172, 'proposal', 21, BLUE, anchor='end', italic=True))
        b.append(box(x0 + 22, 190, 400, 80, GREEN_F, GREEN))
        b += lines(x0 + 222, 224, ['Enforcer on the safety MCU', 'admits, corrects, or refuses'], 22, '#047857', weight='bold', gap=27)
        b.append(arrow(x0 + 222, 272, x0 + 222, 306, GREEN, marker='ahg'))
        b.append(box(x0 + 117, 308, 210, 52, SLATE_F, SLATE, rx=0))
        b.append(text(x0 + 222, 342, 'Drives', 22, SLATE, weight='bold'))
        cap = ['changes the proposal;', 'the proposer stays'] if col == 0 else ['changes the proposer;', 'the check stays']
        b += lines(x0 + 222, 396, cap, 23, TXT, weight='bold', gap=27)
    b.append(line(450, 20, 450, 420, LGRAY, 2))
    svg('propose_admit', 900, 432, b)

def strip():
    x0, x1, tmax = (200, 880, 650.0)
    X = lambda ms: x0 + (x1 - x0) * ms / tmax
    lanes = {'Console': 72, 'Operator': 150, 'Permission': 228}
    tv = 0.104 * 1000.0
    tp, tc, tm = (0.2 * 1000.0, 0.17 * 1000.0, 0.1 * 1000.0)
    t_act_end = tv + tp + tc + tm
    t_in = 0.605 * 1000.0
    spare_ms = 0.1310075 / 1.3 * 1000.0
    for step in (1, 2, 3):
        b = []
        for name, y in lanes.items():
            b.append(text(x0 - 12, y + 36, name, 23, TXT, anchor='end', weight='bold'))
            b.append(line(x0, y + 54, x1, y + 54, LGRAY, 1.5))
        ay = 288
        b.append(line(x0, ay, x1, ay, SLATE, 2))
        for ms in range(0, 700, 100):
            b.append(line(X(ms), ay, X(ms), ay + 8, SLATE, 2))
            b.append(text(X(ms), ay + 32, f'{ms}', 22, SLATE))
        b.append(text(x0 - 12, ay + 32, 'ms', 22, SLATE, anchor='end'))
        y = lanes['Console']
        b.append(box(X(0), y + 8, X(tv) - X(0), 44, AMBER_F, AMBER, rx=3))
        b.append(text(X(tv) + 10, y + 38, f'video {tv:.0f} ms', 23, '#92400E', anchor='start', weight='bold'))
        ty = 366
        b.append(text(x0 - 12, ty + 30, 'Travel', 23, TXT, anchor='end', weight='bold'))
        b.append(box(X(0), ty + 6, X(spare_ms) - X(0), 34, RED_F, RED, rx=2))
        if step < 3:
            b.append(text(X(spare_ms) + 10, ty + 32, f'spare {0.1310075 * 1000.0:.0f} mm, gone at {spare_ms:.0f} ms', 23, RED, anchor='start', weight='bold'))
        if step >= 2:
            y = lanes['Operator']
            for t0, dur, lab in [(tv, tp, 'perceive'), (tv + tp, tc, 'decide'), (tv + tp + tc, tm, 'act')]:
                b.append(box(X(t0), y + 8, X(t0 + dur) - X(t0), 44, BLUE_F, BLUE, rx=3))
                b.append(text((X(t0) + X(t0 + dur)) / 2, y + 38, f'{lab} {dur:.0f}' if lab != 'act' else 'act', 22, '#1D4ED8', weight='bold'))
        if step >= 3:
            y = lanes['Permission']
            b.append(box(X(t_act_end), y + 8, X(t_in) - X(t_act_end), 44, GREEN_F, GREEN, rx=3))
            b.append(text(X(t_act_end) - 10, y + 38, f'handshake {0.031 * 1000.0:.0f} ms', 23, '#047857', anchor='end', weight='bold'))
            b.append(line(X(t_in), 44, X(t_in), ay, GREEN, 2.5, dash='6,4'))
            b += bracket(X(0), X(t_in), 36, f'in-loop takeover, {t_in:.0f} ms, before any command changes', GREEN, 23)
            b.append(box(X(spare_ms), ty + 6, X(t_in) - X(spare_ms), 34, 'url(#hatch)', RED, rx=2))
            b.append(text(X(0), ty - 8, 'spare 131 mm, then 786.5 mm in all', 22, RED, anchor='start', weight='bold'))
        svg(f'strip_step{step}', 900, 412, b)

def zhang():
    vals = np.array([2.1, 2.89, 2.31, 1.86, 1.63, 3.441, 3.418, 3.775, 3.487, 3.64975, 3.99788, 3.97542, 4.31232, 3.69467, 3.73959, 2.47, 2.18, 2.1, 1.9, 3.8, 2.0, 3.8, 2.1, 2.3, 1.8, 3.0, 2.8, 4.2, 3.3, 2.92, 1.93, 2.24, 2.41, 2.71, 2.39, 2.16, 2.21, 5.86, 5.87, 2.4706, 2.1337, 1.70696, 1.70696, 1.9, 2.19, 2.14, 2.48, 2.245, 1.885, 2.22, 1.85, 2.12, 1.71, 1.8, 1.78, 1.72, 1.8, 1.74, 1.67, 1.9, 1.67, 2.22, 2.64, 2.8, 3.09, 2.53, 2.57, 2.58, 3.32, 3.52, 2.76, 3.7, 3.66, 2.41, 3.41, 3.41, 2.62, 3.25, 3.56, 4.026, 8.7, 4.3, 6.8, 2.24, 2.85, 6.91, 2.32, 2.12, 2.37, 2.21, 2.4, 2.1, 1.8, 1.77, 1.61, 1.6, 1.08, 1.47, 1.02, 1.05, 4.14, 4.73, 3.58, 3.9, 3.07, 3.33, 2.71, 2.83, 4.48, 3.83, 0.87, 1.01, 1.97, 3.02, 2.11, 1.86, 4.32, 3.36, 1.93, 1.88, 2.27, 1.1, 2.0, 2.32, 2.09, 2.33, 2.49, 2.69, 3.36, 3.48, 3.46, 3.61, 1.98, 1.82, 2.2, 1.49, 1.28, 2.36, 2.21, 2.12, 2.02, 1.74, 2.24, 1.9, 3.2, 2.86, 2.91, 3.03, 4.72, 4.77, 3.74, 5.02, 4.11, 3.3, 2.77, 5.22, 3.68, 4.32, 4.88, 5.31, 3.07, 3.22, 2.22, 2.04, 2.1, 2.15, 2.29, 1.97, 2.25, 3.2, 2.4706, 2.78504, 2.77381, 4.0428, 3.22301, 3.24547, 2.4, 3.76, 3.5, 3.98, 4.66, 5.51, 4.45, 1.0, 2.3, 2.0, 2.03, 1.8, 3.58, 3.66, 6.80538, 5.13211, 2.97, 3.94, 2.246, 3.22301, 2.25723, 2.38076, 1.83049, 2.0214, 1.03, 1.36, 2.24, 1.01, 1.0, 1.19, 1.01, 3.5, 2.82, 2.44, 2.34, 2.2, 2.09, 2.05, 2.2, 5.09, 4.63, 1.52728, 1.64, 2.56, 1.36, 1.16, 0.92, 1.08, 1.34, 2.8, 2.06, 5.64, 5.42, 2.25723, 2.62782, 2.31338, 2.39199, 2.49306, 2.61659, 2.41445, 2.48183, 2.6, 1.69, 1.54, 2.19, 1.29, 1.42, 2.06, 1.63, 1.46, 2.42, 2.48, 2.8, 8.2, 8.0, 1.51, 1.55, 1.8, 2.0, 1.84, 1.76, 2.65, 3.14, 2.47, 2.31, 2.29, 2.24, 3.56, 3.68, 2.46, 2.09, 2.12, 1.98, 3.28, 4.01, 4.17, 4.625, 4.63, 2.25, 2.8, 1.98, 2.91, 3.31, 2.83, 1.7968, 1.76311, 2.87488, 6.95137, 3.67221, 2.56, 2.73, 2.061, 2.688, 2.026, 1.96, 2.53, 2.86, 3.09, 1.92, 2.97, 1.82, 2.11, 1.51, 1.75, 2.67, 3.17, 1.09, 1.55, 0.956561086, 0.9819004525, 0.8930041152, 1.025514403, 1.13, 0.7152631579, 0.9268421053, 0.9773684211, 0.887755102, 0.806122449, 19.79, 2.086187845, 2.806629834, 3.47, 3.45, 3.9, 3.26, 1.66, 3.09, 2.33, 1.94, 4.14, 4.73, 1.984615385, 2.267307692, 2.5, 2.096153846, 2.213461538, 2.555769231, 1.89, 2.06, 2.18, 1.04, 0.69, 2.45937, 5.22195, 5.43532, 1.771362049, 1.792549476, 1.931082654, 2.154365541, 2.244004657, 2.82, 2.69, 2.62, 1.62, 2.26, 4.58, 4.94, 4.87, 4.51, 4.1, 5.46, 5.18, 5.04, 5.04, 4.99, 1.3, 1.21, 1.31, 1.39, 1.41, 1.42, 1.34, 1.46, 2.95, 2.88, 2.8, 2.06, 2.05, 2.32, 2.31, 1.54, 1.61, 1.226302306, 1.692570453, 0.7677198975, 1.067463706, 3.45, 2.26, 4.07, 3.49, 4.33, 3.61, 1.86, 2.3, 2.37, 6.27, 3.321325648, 2.701729107, 1.99, 1.75, 1.64, 1.51, 1.66, 1.89, 1.75, 1.91, 2.600760456, 3.205323194, 2.09, 1.74, 1.72, 2.833619211, 2.749571184, 2.929674099, 2.677530017, 2.497427101, 2.077186964, 2.26, 2.25, 2.11, 2.37, 2.38, 2.16, 5.26687, 1.77, 1.37, 2.36, 3.09, 1.94, 7.36, 3.8, 3.34, 5.97, 3.01, 3.275, 3.2, 2.87, 2.84, 2.79, 2.97, 2.07, 2.21, 2.16, 1.63, 2.57, 2.969207082, 3.351809084, 5.060046189, 5.027713626, 3.621247113, 4.44, 4.55, 3.96, 5.3, 6.8, 5.4, 7.0, 2.767727931, 2.585383502, 1.55, 1.36, 1.12, 2.05, 1.35, 1.14, 1.95, 1.46, 1.22, 2.661921708, 2.733096085, 2.889679715, 2.277580071, 1.722419929, 2.192170819, 1.329878049, 1.86402439, 1.441463415, 2.187804878, 2.86, 2.37, 2.1, 2.5, 2.33, 2.04, 1.98, 2.56, 2.49, 2.32, 2.15, 2.18, 2.11, 2.22, 3.45, 2.54, 2.58, 2.47, 2.48, 2.41, 2.5, 2.42, 2.18, 2.23, 1.88, 2.05, 1.97, 1.95, 2.119402985, 2.46, 2.67, 3.6, 2.5, 3.4, 3.14, 3.45, 2.92, 3.11, 2.6, 1.212903226, 1.578494624, 2.425806452, 3.802150538, 1.48, 2.88, 3.21])
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    bins = np.arange(0, 10.01, 0.25)
    ax.hist(vals[vals <= 10], bins=bins, color=sp.C['memory_l'], edgecolor=sp.C['memory'], lw=0.8)
    marks = [('median', 2.37, 'p50', 123), ('mean', 2.72, 'mean', 123), ('90th pct.', 4.33, 'p90', 80), ('99th pct.', 7.0, 'p99', 80)]
    for lab, x, key, y in marks:
        col = RED if key == 'p90' else SLATE
        ax.axvline(x, color=col, lw=2, ls='--')
        ha = 'right' if lab == 'median' else 'left'
        dx = -0.08 if ha == 'right' else 0.08
        ax.text(x + dx, y, f"{lab} {x:.2f} s\n{ {'median': 0.31924432, 'mean': 0.2853628359, '90th pct.': 0.1910752891, '99th pct.': 0.1229712742}[lab]:.2f} m/s", color=col, fontsize=11, fontweight='bold', ha=ha, va='top')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 126)
    ax.set_xlabel('Mean take-over time of one study condition (s)', fontsize=12)
    ax.set_ylabel('Conditions', fontsize=12)
    ax.text(9.95, 36, '1 condition at\n19.8 s off scale', ha='right', fontsize=10.5, color=sp.C['neutral'])
    save(fig, 'zhang_hist')

def deadline():
    v = np.linspace(0.12, 1.42, 400)
    dl = np.array([7.319829523, 7.117623696, 6.925765372, 6.743476038, 6.570053421, 6.404862371, 6.247327028, 6.096924063, 5.953176817, 5.815650211, 5.68394631, 5.55770044, 5.436577784, 5.320270382, 5.208494499, 5.10098828, 4.997509685, 4.897834648, 4.801755438, 4.709079197, 4.61962663, 4.533230836, 4.449736254, 4.368997714, 4.290879584, 4.215255, 4.142005169, 4.071018734, 4.002191203, 3.93542443, 3.870626134, 3.807709474, 3.74659265, 3.687198541, 3.629454381, 3.573291448, 3.518644792, 3.46545298, 3.413657855, 3.363204328, 3.314040174, 3.266115848, 3.219384317, 3.1738009, 3.129323128, 3.085910602, 3.043524872, 3.002129322, 2.961689058, 2.922170813, 2.883542847, 2.845774865, 2.808837935, 2.77270441, 2.73734786, 2.702743004, 2.668865649, 2.63569263, 2.603201761, 2.571371777, 2.540182292, 2.509613752, 2.479647393, 2.450265201, 2.421449876, 2.393184797, 2.365453986, 2.338242081, 2.311534307, 2.285316443, 2.259574801, 2.2342962, 2.209467942, 2.185077792, 2.161113953, 2.137565051, 2.114420115, 2.091668556, 2.069300156, 2.047305048, 2.025673702, 2.00439691, 1.983465775, 1.962871695, 1.942606353, 1.922661703, 1.903029962, 1.883703597, 1.864675316, 1.845938059, 1.827484986, 1.809309473, 1.791405099, 1.773765642, 1.756385069, 1.73925753, 1.722377349, 1.705739022, 1.689337206, 1.673166714, 1.657222514, 1.641499715, 1.62599357, 1.610699465, 1.595612918, 1.580729574, 1.566045197, 1.55155567, 1.53725699, 1.523145262, 1.509216698, 1.495467611, 1.481894413, 1.468493613, 1.455261811, 1.442195697, 1.429292045, 1.416547716, 1.40395965, 1.391524865, 1.379240456, 1.367103591, 1.355111507, 1.343261513, 1.331550984, 1.319977358, 1.308538137, 1.297230884, 1.28605322, 1.275002823, 1.264077429, 1.253274824, 1.242592848, 1.232029393, 1.221582399, 1.211249852, 1.201029747, 1.190918631, 1.180913864, 1.171013631, 1.161216161, 1.151519718, 1.141922608, 1.132423175, 1.123019797, 1.113710889, 1.104494902, 1.09537032, 1.086335658, 1.077389464, 1.06853032, 1.059756833, 1.051067643, 1.042461419, 1.033936855, 1.025492675, 1.017127628, 1.008840489, 1.000630059, 0.9924951618, 0.9844346474, 0.9764473873, 0.9685322762, 0.960688231, 0.9529141901, 0.945209113, 0.9375719796, 0.9300017899, 0.9224975635, 0.915058339, 0.9076831734, 0.9003711423, 0.8931213384, 0.8859328723, 0.8788048711, 0.8717364784, 0.8647268541, 0.8577751736, 0.8508806279, 0.8440424226, 0.8372597784, 0.8305319299, 0.8238581261, 0.8172376294, 0.8106697155, 0.8041536735, 0.7976888048, 0.7912744237, 0.7849098564, 0.7785944413, 0.7723275282, 0.7661084785, 0.7599366648, 0.7538114703, 0.7477322893, 0.7416985262, 0.735709596, 0.7297649233, 0.7238639429, 0.7180060988, 0.7121908447, 0.7064176434, 0.7006859665, 0.6949952948, 0.6893451173, 0.6837349318, 0.6781642441, 0.6726325684, 0.6671394265, 0.6616843483, 0.656266871, 0.6508865396, 0.6455429062, 0.6402355301, 0.6349639777, 0.6297278222, 0.6245266436, 0.6193600285, 0.6142275701, 0.6091288677, 0.6040635271, 0.5990311601, 0.5940313845, 0.5890638241, 0.5841281082, 0.579223872, 0.5743507563, 0.5695084071, 0.5646964759, 0.5599146194, 0.5551624995, 0.5504397832, 0.5457461422, 0.5410812534, 0.5364447982, 0.5318364629, 0.5272559382, 0.5227029194, 0.5181771064, 0.5136782031, 0.5092059181, 0.5047599638, 0.500340057, 0.4959459184, 0.4915772727, 0.4872338487, 0.4829153787, 0.478621599, 0.4743522495, 0.4701070739, 0.4658858193, 0.4616882364, 0.4575140792, 0.4533631054, 0.4492350758, 0.4451297545, 0.4410469089, 0.4369863095, 0.4329477299, 0.4289309468, 0.42493574, 0.4209618921, 0.4170091887, 0.4130774183, 0.409166372, 0.4052758438, 0.4014056306, 0.3975555318, 0.3937253493, 0.3899148878, 0.3861239545, 0.3823523591, 0.3785999137, 0.374866433, 0.3711517338, 0.3674556354, 0.3637779596, 0.3601185302, 0.3564771732, 0.3528537172, 0.3492479925, 0.3456598318, 0.3420890698, 0.3385355433, 0.3349990912, 0.3314795543, 0.3279767754, 0.3244905993, 0.3210208726, 0.3175674438, 0.3141301635, 0.3107088837, 0.3073034586, 0.3039137438, 0.300539597, 0.2971808775, 0.293837446, 0.2905091654, 0.2871958997, 0.2838975149, 0.2806138785, 0.2773448595, 0.2740903284, 0.2708501573, 0.26762422, 0.2644123913, 0.261214548, 0.2580305679, 0.2548603304, 0.2517037164, 0.248560608, 0.2454308887, 0.2423144433, 0.2392111581, 0.2361209205, 0.2330436192, 0.2299791443, 0.2269273869, 0.2238882397, 0.2208615961, 0.2178473513, 0.2148454011, 0.2118556428, 0.2088779747, 0.2059122964, 0.2029585084, 0.2000165124, 0.1970862112, 0.1941675086, 0.1912603095, 0.1883645198, 0.1854800465, 0.1826067975, 0.1797446818, 0.1768936093, 0.1740534908, 0.1712242383, 0.1684057646, 0.1655979834, 0.1628008093, 0.160014158, 0.1572379457, 0.1544720899, 0.1517165089, 0.1489711215, 0.1462358478, 0.1435106085, 0.1407953252, 0.1380899202, 0.1353943167, 0.1327084388, 0.1300322112, 0.1273655594, 0.1247084098, 0.1220606894, 0.119422326, 0.1167932482, 0.1141733852, 0.111562667, 0.1089610244, 0.1063683886, 0.1037846918, 0.1012098667, 0.09864384677, 0.09608656613, 0.09353795951, 0.09099796231, 0.08846651058, 0.08594354098, 0.08342899083, 0.08092279802, 0.07842490109, 0.07593523916, 0.07345375195, 0.07098037976, 0.06851506348, 0.06605774457, 0.06360836506, 0.06116686752, 0.05873319509, 0.05630729146, 0.05388910084, 0.05147856799, 0.04907563818, 0.04668025722, 0.04429237142, 0.0419119276, 0.03953887309, 0.03717315571, 0.03481472375, 0.03246352603, 0.03011951181, 0.02778263083, 0.02545283331, 0.02313006993, 0.02081429182, 0.01850545057, 0.01620349819, 0.01390838718, 0.01162007042]) * 1000.0
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.6))
        ax.plot(v, dl, color=GREEN, lw=2.8)
        ax.text(0.22, 42, 'longest wait before\nthe stop must begin', color='#047857', fontsize=11.5, fontweight='bold')
        ax.plot([1.3], [0.100775 * 1000.0], 'o', color=RED, ms=8)
        ax.text(1.3 - 0.02, 0.100775 * 1000.0 * 0.62, '1.3 m/s: 100.8 ms', color=RED, fontsize=11.5, fontweight='bold', ha='right', va='top')
        if step >= 2:
            ax.axhline(0.605 * 1000.0, color=AMBER, lw=2, ls='--')
            ax.text(0.22, 0.605 * 1000.0 * 0.88, 'in-loop takeover 605 ms', color='#B45309', fontsize=11, fontweight='bold', ha='left', va='top')
            ax.plot([0.75], [0.7516083333 * 1000.0], 'o', color=NAVY, ms=8)
            ax.text(0.75 + 0.03, 0.7516083333 * 1000.0 * 1.25, '0.75 m/s: 751.6 ms', color=NAVY, fontsize=11.5, fontweight='bold', va='bottom')
        if step >= 3:
            ax.axhline(4.465 * 1000.0, color=AMBER, lw=2, ls=':')
            ax.text(1.42, 4.465 * 1000.0 * 1.12, 'out-of-loop takeover 4.5 s', color='#B45309', fontsize=11, fontweight='bold', ha='right', va='bottom')
            ax.plot([0.15], [5.787829791 * 1000.0], 'o', color=NAVY, ms=8)
            ax.text(0.15 + 0.02, 5.787829791 * 1000.0 * 1.3, '0.15 m/s: 5.79 s', color=NAVY, fontsize=11.5, fontweight='bold', va='bottom')
        ax.set_yscale('log')
        ax.set_ylim(30, 20000)
        ax.set_yticks([100, 1000, 10000])
        ax.set_yticklabels(['100 ms', '1 s', '10 s'])
        ax.minorticks_off()
        ax.set_xlim(0.1, 1.45)
        ax.set_xlabel('Speed when the takeover is requested (m/s)', fontsize=12)
        ax.set_ylabel('Fallback deadline', fontsize=12)
        ax.grid(axis='y', color=sp.C['grid'], lw=0.6)
        save(fig, f'deadline_step{step}')

def leases():
    x0, x1, tmax = (20, 548, 272.0)
    X = lambda ms: x0 + (x1 - x0) * ms / tmax
    b = []
    lease, manual, period = (60.0, 100.0, 50.0)
    specs = [('Autonomous: chunk lease 60 ms', 34, [0, period, period + period], lease, NAVY, 'renewal'), ('Manual: lease 100 ms', 206, [0, 25, 50, 75, 100], manual, BLUE, 'heartbeat')]
    for title, y, ev, lease, col, lab in specs:
        b.append(text(x0, y, title, 23, col, anchor='start', weight='bold'))
        b.append(line(x0, y + 66, x1, y + 66, SLATE, 2))
        for t in ev:
            b.append(line(X(t), y + 34, X(t), y + 66, col, 4))
        last = ev[-1]
        b.append(text(X(last), y + 28, f'last {lab}', 21, col, anchor='middle'))
        b.append(box(X(last), y + 72, X(last + lease) - X(last), 22, BLUE_F, col, rx=3, sw=2))
        b.append(text((X(last) + X(last + lease)) / 2, y + 120, f'{lease:.0f} ms, MCU clock', 21, col))
        b.append(line(X(last + lease), y + 30, X(last + lease), y + 98, RED, 3))
        b.append(box(X(last + lease) + 6, y + 34, 122, 32, RED_F, RED, rx=4, sw=2))
        b.append(text(X(last + lease) + 67, y + 58, 'FALLBACK', 20, RED, weight='bold'))
    b.append(text(x0, 372, 'Silence ends authority on the MCU clock,', 21, TXT, anchor='start', weight='bold'))
    b.append(text(x0, 398, 'and the machine holds or stops.', 21, TXT, anchor='start', weight='bold'))
    svg('leases', 570, 410, b)

def fsm():
    W, H = (168, 84)
    pos = {'AUTO': (12, 74), 'PEND': (244, 74), 'BLEND': (476, 74), 'MAN': (708, 74), 'FALLBACK': (290, 250), 'PROTECTIVE': (708, 290)}
    names = {'AUTO': 'Autonomous', 'PEND': 'Handover-', 'BLEND': 'Blend', 'MAN': 'Human-', 'FALLBACK': 'Fallback', 'PROTECTIVE': 'Protective'}
    names2 = {'PEND': 'Pending', 'MAN': 'Manual'}
    token = {'AUTO': 'policy', 'PEND': 'policy', 'BLEND': 'no one', 'MAN': 'operator', 'FALLBACK': 'no one', 'PROTECTIVE': 'no one'}
    col = {'AUTO': (BLUE_F, NAVY), 'PEND': (BLUE_F, NAVY), 'BLEND': (GREEN_F, GREEN), 'MAN': (BLUE_F, BLUE), 'FALLBACK': (RED_F, RED), 'PROTECTIVE': (RED_F, RED)}

    def node(k):
        x, y = pos[k]
        f, s = col[k]
        out = [box(x, y, W, H, f, s)]
        if k in names2:
            out += lines(x + W / 2, y + 27, [names[k], names2[k]], 22, s, weight='bold', gap=24)
            out.append(text(x + W / 2, y + 76, token[k], 20, SLATE, italic=True))
        else:
            out.append(text(x + W / 2, y + 36, names[k], 22, s, weight='bold'))
            out.append(text(x + W / 2, y + 68, token[k], 20, SLATE, italic=True))
        return out
    for step in (1, 2, 3):
        b = [f'<path d="M4,22 H896 V236 H690 V392 H4 Z" fill="none" stroke="{GRAY}" stroke-width="2" stroke-dasharray="8,6"/>']
        b.append(text(14, 382, 'Italic: who holds the proposal token', 20, SLATE, anchor='start', italic=True))
        for k in ['AUTO', 'PEND', 'BLEND', 'MAN']:
            b += node(k)
        cy = 74 + H / 2
        for a, c, lab, colr, mk in [('AUTO', 'PEND', 'request', NAVY, 'ahn'), ('PEND', 'BLEND', 'commit', GREEN, 'ahg'), ('BLEND', 'MAN', 'done', BLUE, 'ahb')]:
            xa = pos[a][0] + W
            xc = pos[c][0]
            b.append(arrow(xa + 2, cy, xc - 2, cy, colr, marker=mk))
            b.append(text((xa + xc) / 2, 62, lab, 20, colr, weight='bold'))
        if step >= 2:
            b += node('FALLBACK')
            fx, fy = pos['FALLBACK']
            for k, tx in [('AUTO', fx + 20), ('PEND', fx + 60), ('BLEND', fx + 110), ('MAN', fx + 150)]:
                x, y = pos[k]
                b.append(arrow(x + W / 2, y + H, tx, fy - 2, RED, sw=2.5, marker='ahr'))
            b.append(text(fx + W + 14, fy + 36, 'lease lapse,', 20, RED, anchor='start'))
            b.append(text(fx + W + 14, fy + 60, 'timeout, bad state', 20, RED, anchor='start'))
            b.append(f'<path d="M{fx},{fy + H / 2} C{150},{fy + H / 2} {50},{fy + 10} {50},{74 + H + 4}" fill="none" stroke="{SLATE}" stroke-width="2.5" stroke-dasharray="7,5" marker-end="url(#ah)"/>')
            b.append(text(140, fy + H / 2 + 48, 'reset from rest only', 20, SLATE))
        if step >= 3:
            b += node('PROTECTIVE')
            px, py = pos['PROTECTIVE']
            b.append(arrow(px + W / 2, 236, px + W / 2, py - 2, RED, sw=3, marker='ahr'))
            b.append(text(px + W / 2 + 10, 262, 'hardwired', 20, RED, anchor='start'))
        svg(f'fsm_step{step}', 900, 400, b)

def handshake():
    x0, x1, tmax = (150, 880, 34.0)
    X = lambda ms: x0 + (x1 - x0) * ms / tmax
    lanes = {'Console': 66, 'MCU': 166, 'Drives': 266}
    b = []
    for name, y in lanes.items():
        b.append(text(x0 - 12, y + 34, name, 23, TXT, anchor='end', weight='bold'))
        b.append(line(x0, y + 52, x1, y + 52, LGRAY, 1.5))
    t = 0.0
    starts = {}
    for k, d in {'Request': 0.01, 'Acknowledge': 0.005, 'Commit': 0.001, 'Confirm': 0.015}.items():
        starts[k] = (t, t + d * 1000.0)
        t += d * 1000.0
    r0, r1 = starts['Request']
    b.append(box(X(r0), 72, X(r1) - X(r0), 40, BLUE_F, BLUE, rx=3))
    b.append(text((X(r0) + X(r1)) / 2, 100, 'Request 10', 22, '#1D4ED8', weight='bold'))
    b.append(arrow(X(r1) - 4, 114, X(r1) + 6, 172, BLUE, marker='ahb'))
    a0, a1 = starts['Acknowledge']
    b.append(box(X(a0), 174, X(a1) - X(a0), 40, GREEN_F, GREEN, rx=3))
    b.append(text((X(a0) + X(a1)) / 2, 240, 'Ack 5', 22, '#047857', weight='bold'))
    c0, c1 = starts['Commit']
    b.append(box(X(c0), 168, X(c1) - X(c0) + 2, 52, RED_F, RED, rx=2))
    b.append(line(X(c0), 44, X(c0), 326, RED, 2.5, dash='6,4'))
    b.append(text(X(c0), 36, 'Commit on one 1 ms tick changes the owner', 22, RED, weight='bold'))
    f0, f1 = starts['Confirm']
    b.append(box(X(f0) + 2, 174, X(f1) - X(f0) - 2, 40, GREEN_F, GREEN, rx=3))
    b.append(text((X(f0) + X(f1)) / 2, 202, 'Confirm 15', 22, '#047857', weight='bold'))
    b.append(arrow(X(f1) - 6, 172, X(f1) + 4, 114, GREEN, marker='ahg'))
    b.append(text(X(f1) - 8, 100, 'if lost, alarm only', 22, RED, anchor='end', italic=True))
    b.append(box(X(0), 274, X(c0) - X(0), 40, BLUE_F, NAVY, rx=3))
    b.append(text((X(0) + X(c0)) / 2, 302, 'policy setpoints', 22, NAVY))
    b.append(box(X(c1), 274, X(tmax) - X(c1), 40, GREEN_F, GREEN, rx=3))
    b.append(text((X(c1) + X(tmax)) / 2, 302, 'enforcer blend', 22, '#047857'))
    ay = 344
    b.append(line(x0, ay, x1, ay, SLATE, 2))
    for ms, lab in [(0, '0'), (10.0, '10'), (15.0, '15'), (31.0, '31 ms')]:
        b.append(line(X(ms), ay, X(ms), ay + 8, SLATE, 2))
        b.append(text(X(ms), ay + 32, lab, 22, SLATE))
    svg('handshake', 900, 385, b)

def quintic():
    W = 0.52734375
    t = np.linspace(-0.15, 0.75, 2000)
    s = np.clip(t / W, 0, 1)
    u = 2.0 * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
    du = np.where((t > 0) & (t < W), 2.0 * 30 * s ** 2 * (1 - s) ** 2 / W, 0.0)
    for step in (1, 2):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 2.9), sharex=True, gridspec_kw=dict(hspace=0.25))
        step_u = np.where(t >= 0, 2.0, 0.0)
        a1.plot(t * 1000.0, step_u, color=RED, lw=2.4)
        a1.text(20, 1.55, 'step', color=RED, fontsize=12, fontweight='bold')
        a2.annotate('', xy=(0, 9.6), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.4))
        a2.text(14, 9.0, f'step: {2.0 / 0.001:,.0f} m/s³ for one tick (off scale)', color=RED, fontsize=11.5, fontweight='bold', va='center')
        if step == 2:
            a1.plot(t * 1000.0, u, color=GREEN, lw=2.8)
            a1.text(W * 1000.0 * 0.62, 0.55, f'quintic over {W * 1000.0:.0f} ms', color='#047857', fontsize=12, fontweight='bold')
            a2.plot(t * 1000.0, du, color=GREEN, lw=2.8)
            a2.text(W * 1000.0 * 0.5 + 170, du.max() - 1.0, f'peak {du.max():.2f} m/s³', color='#047857', fontsize=11.5, fontweight='bold', ha='left')
        a1.set_ylabel('m/s²', fontsize=11)
        a1.set_ylim(-0.2, 2.4)
        a1.set_yticks([0, 1, 2])
        a2.set_ylabel('m/s³', fontsize=11)
        a2.set_ylim(0, 10.5)
        a2.set_yticks([0, 5, 10])
        a2.set_xlabel('Time from commit (ms)', fontsize=12)
        a1.set_title('Acceleration command', fontsize=12, loc='left')
        a2.set_title('Its rate (jerk)', fontsize=12, loc='left')
        a2.set_xlim(-150, 750)
        save(fig, f'quintic_step{step}')

def session():
    rows = [('Spare at 0.75 m/s', 0.56370625, 0.0), ('− takeover 605 ms', 0.10995625, 0.56370625 - 0.10995625), ('− dispatch to admission', 0.06247321, 0.10995625 - 0.06247321), ('− manual lease, 40 ms longer', 0.03247321, 0.06247321 - 0.03247321)]
    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    n = len(rows) + 1
    for i, (lab, left, spent) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, left * 1000.0, color=GREEN, height=0.6)
        if spent:
            ax.barh(y, spent * 1000.0, left=left * 1000.0, color=AMBER_F, edgecolor=AMBER, height=0.6, lw=1.2)
        ax.text(left * 1000.0 + (spent * 1000.0 if spent else 0) + 10, y, f'{left * 1000.0:.1f} mm left', va='center', fontsize=12, fontweight='bold', color='#047857')
    ax.barh(0, 0.1569703115 * 1000.0, color=RED_F, edgecolor=RED, hatch='//', height=0.6, lw=1.2)
    ax.text(0.1569703115 * 1000.0 + 10, 0, 'braking blend needs up to 157 mm', va='center', fontsize=12, fontweight='bold', color=RED)
    ax.axvline(0.03247321 * 1000.0, color=RED, lw=1.6, ls='--')
    ax.set_yticks(range(n))
    ax.set_yticklabels(['− base braking blend'] + [r[0] for r in rows][::-1], fontsize=12)
    ax.set_xlim(0, 700)
    ax.set_xlabel('Clearance left at the rack end (mm)', fontsize=12)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    save(fig, 'session')

def pallet():
    b = []
    b.append(box(195, 120, 170, 90, '#FDE68A', '#B45309', rx=4))
    b.append(text(280, 174, 'pallet', 26, '#92400E', weight='bold'))
    b.append(box(238, 322, 84, 52, SLATE_F, SLATE, rx=0))
    b.append(text(280, 356, 'robot', 24, SLATE))
    b.append(f'<path d="M280,320 C280,260 140,250 140,170 C140,120 200,80 280,56" fill="none" stroke="{BLUE}" stroke-width="4" marker-end="url(#ahb)"/>')
    b.append(f'<path d="M280,320 C280,260 420,250 420,170 C420,120 360,80 280,56" fill="none" stroke="{NAVY}" stroke-width="4" stroke-dasharray="9,6" marker-end="url(#ahn)"/>')
    b.append(arrow(280, 320, 280, 216, RED, sw=4, marker='ahr'))
    b += lines(12, 250, ['operator', 'goes left'], 25, BLUE, anchor='start', weight='bold', gap=30)
    b += lines(438, 250, ['policy', 'goes right'], 25, NAVY, anchor='start', weight='bold', gap=30)
    b += lines(280, 408, ['their average hits the pallet'], 26, RED, weight='bold')
    svg('pallet', 570, 425, b)

def flood():
    n = np.arange(0, 61)
    fin = (n + 1) * 0.0018 * 1000.0
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.plot(n, fin, color=SLATE, lw=2.6)
    for budget, lab, col, nn in [(0.100775 * 1000.0, 'deadline 100.8 ms', RED, 55), ((0.100775 - 0.06331072) * 1000.0, 'after admission 37.5 ms', AMBER, 20)]:
        ax.axhline(budget, color=col, lw=2, ls='--')
        ax.text(1 if nn > 30 else 30, budget + 3, lab, color=col, fontsize=11.5, fontweight='bold', va='bottom')
        ax.plot([nn], [(nn + 1) * 0.0018 * 1000.0], 'o', color=col, ms=8)
        ax.text(nn + 1.5, (nn + 1) * 0.0018 * 1000.0 - 4, f'{nn} forged', color=col, fontsize=11.5, fontweight='bold', va='top')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 115)
    ax.set_xlabel('Forged signed packets queued ahead of the veto', fontsize=12)
    ax.set_ylabel('Veto verified at (ms)', fontsize=12)
    ax.text(58, 8, '1.8 ms per Ed25519 check\n(illustrative)', ha='right', fontsize=10.5, color=sp.C['neutral'])
    save(fig, 'flood')

def channels():
    x0, x1, tmax = (170, 880, 610.0)
    X = lambda ms: x0 + (x1 - x0) * ms / tmax
    t_in = 0.605 * 1000.0
    for step in (1, 2):
        b = []
        b.append(text(x0, 34, f'Takeover window {t_in:.0f} ms; the spare at 1.3 m/s is {0.1310075 * 1000.0:.0f} mm', 22, SLATE, anchor='start'))
        for lab, y in [('Arm', 96), ('Base', 182), ('Console', 268)]:
            b.append(text(x0 - 12, y, lab, 24, TXT, anchor='end', weight='bold'))
        b.append(box(X(0), 66, X(t_in) - X(0), 46, BLUE_F, BLUE, rx=3))
        b.append(text((X(0) + X(t_in)) / 2, 97, 'operator aligns the gripper (MAN)', 23, '#1D4ED8', weight='bold'))
        b.append(box(X(0), 152, X(t_in) - X(0), 46, RED_F, RED, rx=3))
        b.append(text((X(0) + X(t_in)) / 2, 183, 'policy still drives at 1.3 m/s, 786.5 mm', 23, RED, weight='bold'))
        if step == 1:
            b.append(box(X(0), 238, X(t_in) - X(0), 46, SLATE_F, GRAY, rx=3))
            b.append(text((X(0) + X(t_in)) / 2, 269, 'ARM: MAN   (base not shown)', 23, SLATE, italic=True))
        else:
            b.append(box(X(0), 238, X(t_in) - X(0), 46, GREEN_F, GREEN, rx=3))
            b.append(text((X(0) + X(t_in)) / 2, 269, 'ARM: MAN   |   BASE: AUTO, MOVING', 23, '#047857', weight='bold'))
            b.append(text(20, 326, "Announce every channel's owner, or commit the untransferred", 23, '#047857', anchor='start', weight='bold'))
            b.append(text(20, 356, 'channel into a certified hold with the brake on.', 23, '#047857', anchor='start', weight='bold'))
        svg(f'channels_step{step}', 900, 370, b)

def lh2904():
    x0, x1, tmax = (270, 880, 15.0)
    X = lambda s: x0 + (x1 - x0) * s / tmax
    G, B = (9.0, 13.0)
    rows = [('Right gear', 40), ('Left gear', 102), ('Crew', 164), ('Spoilers, reversers', 226), ('Wheel brakes', 288)]
    b = []
    for name, y in rows:
        b.append(text(x0 - 12, y + 30, name, 24, TXT, anchor='end', weight='bold'))
    b.append(box(X(0), 44, X(15) - X(0), 36, GREEN_F, GREEN, rx=3))
    b.append(text(X(7.5), 71, 'on the runway', 23, '#047857'))
    b.append(box(X(0), 106, X(G) - X(0), 36, 'white', GRAY, dash='6,4', rx=3))
    b.append(text(X(4.5), 133, 'not compressed', 23, SLATE, italic=True))
    b.append(box(X(G), 106, X(15) - X(G), 36, GREEN_F, GREEN, rx=3))
    b.append(box(X(0), 168, X(15) - X(0), 36, BLUE_F, BLUE, rx=3))
    b.append(text(X(7.5), 195, 'brakes and reversers commanded', 23, '#1D4ED8', weight='bold'))
    b.append(box(X(0), 230, X(G) - X(0), 36, RED_F, RED, rx=3))
    b.append(text(X(4.5), 257, 'locked out', 23, RED, weight='bold'))
    b.append(box(X(G), 230, X(15) - X(G), 36, GREEN_F, GREEN, rx=3))
    b.append(box(X(0), 292, X(B) - X(0), 36, RED_F, RED, rx=3))
    b.append(text(X(6.5), 319, 'no braking, aquaplaning', 23, RED, weight='bold'))
    b.append(box(X(B), 292, X(15) - X(B), 36, GREEN_F, GREEN, rx=3))
    ay = 352
    b.append(line(x0, ay, x1, ay, SLATE, 2))
    for s, lab in [(0, '0'), (3, '3'), (6, '6'), (G, '9'), (B, '13'), (15, '15 s')]:
        b.append(line(X(s), ay, X(s), ay + 8, SLATE, 2))
        b.append(text(X(s), ay + 32, lab, 23, RED if s == B else SLATE, weight='bold' if s == B else 'normal'))
    b.append(line(X(G), 30, X(G), ay, RED, 2, dash='5,4'))
    b.append(line(X(B), 286, X(B), ay, RED, 2, dash='5,4'))
    b.append(text(X(G), 24, '9 s', 23, RED, weight='bold'))
    svg('lh2904', 900, 392, b)

def masks():
    cells = [('drift', 'policy drifts', '4.5 s', 'M = 0', RED_F, RED), ('blend', 'enforcer spline', '38 ms', 'M = 0', GREEN_F, GREEN), ('re-orient', 'operator hunts', 'until settled', 'M = 0', AMBER_F, AMBER), ('stable', 'operator steady', 'after a dwell', 'M = 1', BLUE_F, BLUE)]
    xs = [10, 233, 456, 679]
    w = 211
    stamps = ['divergence', 'commit', 'blend end', 'stabilized']
    for step in (1, 2):
        b = []
        b.append(text(10, 30, 'MCU timestamps split one takeover into four intervals', 23, TXT, anchor='start', weight='bold'))
        for (name, who, dur, m, f, s), x in zip(cells, xs):
            b.append(box(x, 96, w, 106, f if name == 'stable' else 'white', s, sw=3))
            b.append(text(x + w / 2, 132, name, 25, s, weight='bold'))
            b.append(text(x + w / 2, 162, who, 21, SLATE))
            b.append(text(x + w / 2, 188, dur, 21, SLATE, italic=True))
            b.append(text(x + w / 2, 236, m, 25, TXT if m == 'M = 1' else SLATE, weight='bold'))
        for x, st in zip(xs, stamps):
            xx = x if x == 10 else x - 6
            b.append(line(xx, 70, xx, 214, SLATE, 2, dash='4,4'))
            b.append(text(xx, 64, st, 21, SLATE, anchor='start' if x == 10 else 'middle'))
        b += bracket(679, 890, 262, 'trains', BLUE, 22, up=False)
        if step == 2:
            wx = 10 + w * (4.465 - 2.0) / 4.465
            b.append(box(wx, 194, 10 + w - wx, 12, AMBER, AMBER, rx=2, sw=1))
            b.append(text(10 + w - 6, 188, '2 s', 21, '#92400E', anchor='end', weight='bold'))
            b.append(text(10, 326, 'A fixed 2 s window (Sirius) masks the last 2 s of a 4.5 s drift;', 22, '#92400E', anchor='start'))
            b.append(text(10, 354, 'the first 2.5 s stay in the data as if they were good.', 22, '#92400E', anchor='start'))
        svg(f'masks_step{step}', 900, 365, b)

def main():
    dmv()
    propose_admit()
    strip()
    zhang()
    deadline()
    leases()
    fsm()
    handshake()
    quintic()
    session()
    pallet()
    flood()
    channels()
    lh2904()
    masks()
    print('\nfigures written to', HERE)
if __name__ == '__main__':
    main()
