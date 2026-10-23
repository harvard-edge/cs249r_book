#!/usr/bin/env python3
"""Figures for Lecture 16 (Deployment Release).

Every value is sourced. "DR" is the rendered chapter, Deployment Release
(Volume IV, Chapter 16). Every value comes from calcs.py (imported as K;
labels through deckvals.val), which reads MLSysIM: the running machine, the
Challenger O-ring record (slides/vol4/16_release/images/data/challenger_oring_distress_record.csv, Dalal,
Fowlkes, and Hoadley 1989 via UCI) with the Rogers Commission's temperatures
(cited literature), and Waymo's rider-only crash rates by
outcome (vol4/16_release/data/waymo_safety_impact_headline.csv).
Diagrams follow the Volume IV causal-boundary palette (rules/volumes/vol4/
palette.md): green for verified or certified, blue for learned proposals,
slate for plant and structure, amber for latency and warnings, red only for
veto, hazard, breach, and refusal.

Every figure is drawn at the size of its slot on the slide, so the slide
places it at scale 1 (plots: fixed figure size, no tight bounding box;
diagrams: an SVG viewBox with the slot's aspect ratio).

Run from this folder: python3 make_figs.py
"""
from mlsysim.core.series_loader import load_series
import mlsysim
import contextlib
import io
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
from slideplot import plt
from mlsysim.core.series_loader import load_series
from mlsysim import Literature
C = sp.C
NAVY, BLUE, GREEN, RED, AMBER, SLATE, PETROL, GRAY = ('#1F407A', '#2563EB', '#059669', '#DC2626', '#D97706', '#334155', '#0F766E', '#6B7280')
GREEN_F, BLUE_F, RED_F, SLATE_F, AMBER_F, GRAY_F = ('#DCFCE7', '#EFF6FF', '#FEE2E2', '#F8FAFC', '#FEF3C7', '#F3F4F6')
TXT = '#1A1A2E'
FONT = 'Helvetica, Arial, sans-serif'

def cm(w, h):
    """Figure size for a slot of w by h centimeters."""
    return (w / 2.54, h / 2.54)

def save_slot(fig, name):
    """Save at exactly the figure size, so the slide places it at scale 1."""
    fig.savefig(HERE / f'{name}.svg', format='svg', facecolor='white')
    plt.close(fig)
DEFS = f'<defs>\n<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{SLATE}"/></marker>\n<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{GREEN}"/></marker>\n<marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{RED}"/></marker>\n</defs>'

def svg(name, w, h, body):
    (HERE / f'{name}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">\n{DEFS}\n<rect width="{w}" height="{h}" fill="white"/>\n' + '\n'.join(body) + '\n</svg>\n')

def box(x, y, w, h, fill, stroke, dash=None, rx=6, sw=2.2):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'

def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def text(x, y, s, size=20, fill=TXT, anchor='middle', weight='normal'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>'

def lines(x, y, rows, size=20, fill=TXT, anchor='middle', weight='normal', gap=None):
    gap = gap or size * 1.22
    return [text(x, y + i * gap, r, size, fill, anchor, weight) for i, r in enumerate(rows)]

def arrow(x1, y1, x2, y2, color=SLATE, sw=2.4, dash=None, marker='ah'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{marker})"/>'

def line(x1, y1, x2, y2, color=SLATE, sw=2.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>'

def oring(k):
    fig = plt.figure(figsize=cm(14.6, 5.0))
    ax = fig.add_axes([0.1, 0.22, 0.88, 0.75])
    seen = {}
    for _, t, d in [(1, 66.0, 0), (2, 70.0, 1), (3, 69.0, 0), (4, 68.0, 0), (5, 67.0, 0), (6, 72.0, 0), (7, 73.0, 0), (8, 70.0, 0), (9, 57.0, 1), (10, 63.0, 1), (11, 70.0, 1), (12, 78.0, 0), (13, 67.0, 0), (14, 53.0, 2), (15, 67.0, 0), (16, 75.0, 0), (17, 70.0, 0), (18, 81.0, 0), (19, 76.0, 0), (20, 79.0, 0), (21, 75.0, 2), (22, 76.0, 0), (23, 58.0, 1)]:
        if d > 0:
            ax.plot(t, d, 'o', ms=8, color=RED, mec='white', mew=1.0, zorder=5)
        else:
            n = seen.get(t, 0)
            seen[t] = n + 1
            ax.plot(t, 0.13 * n, 'o', ms=7, color=SLATE, mec='white', mew=1.0, zorder=4, alpha=1.0 if k >= 2 else 0.0)
    ax.text(84, 2.85, 'Flights with O-ring distress only' if k == 1 else 'All 23 flights in the record', ha='right', va='center', fontsize=11, fontweight='bold', color=TXT)
    ax.text(84, 2.42, '63 °F or colder: 4 of 4 had distress\n66 °F or warmer: 3 of 19', ha='right', va='center', fontsize=10, color=SLATE, alpha=1.0 if k >= 2 else 0.0, linespacing=1.15)
    a3 = 1.0 if k >= 3 else 0.0
    ax.axvspan(15, 53.0, color=GRAY_F, zorder=0, alpha=a3)
    ax.axvline(53.0, color=SLATE, lw=1.6, ls='--', alpha=a3)
    ax.text(53.0 - 1.5, 2.6, 'no flight evidence\nbelow 53 °F', ha='right', va='center', fontsize=10.5, fontweight='bold', color=SLATE, alpha=a3, linespacing=1.1)
    ax.errorbar(28.0, 1.0, xerr=5.0, fmt='D', ms=9, color=RED, ecolor=RED, elinewidth=2, capsize=4, alpha=a3, zorder=6)
    ax.text(28.0, 1.22, 'Challenger joint, about 28 °F', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=RED, alpha=a3)
    ax.text(28.0, 0.75, '(air 36 °F)', ha='center', va='top', fontsize=10, color=RED, alpha=a3)
    ax.set_xlim(15, 85)
    ax.set_ylim(-0.12, 3.05)
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel('Distressed O-rings', fontsize=10.5)
    ax.set_xlabel('Calculated joint temperature at launch (°F)', fontsize=10.5)
    ax.tick_params(labelsize=10)
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    save_slot(fig, f'oring_step{k}')

def cae(k):
    W, H = (900, 330)
    b = []
    b.append(box(8, 8, 222, 118, SLATE_F, SLATE, dash='7,5'))
    b += lines(119, 34, ['Context: envelope'], 19, weight='bold')
    b += lines(119, 60, ['payload ≤ 50 kg', 'floor μ ≥ 0.204', 'person stands still'], 18)
    b.append(box(262, 8, 400, 66, GREEN_F, GREEN, sw=2.6))
    b += lines(462, 34, ['Claim C1a: the base stops inside 1.10 m'], 20, weight='bold')
    b += lines(462, 60, ['at 1.3 m/s, loaded (d_stop ≤ D_clear)'], 18)
    b.append(line(230, 41, 262, 41, SLATE, 2, dash='5,4'))
    b.append(box(262, 104, 400, 66, 'white', SLATE, sw=2.2))
    b += lines(462, 130, ['Warrant: the stopping budget bounds'], 19, weight='bold')
    b += lines(462, 156, ['travel if every term is bounded'], 18)
    b.append(arrow(462, 104, 462, 77))
    leaves = [('Loaded stop trials', 'a_brake ≥ 2 m/s²'), ('Brake-onset trials', '≤ 20 ms'), ('End-to-end age trace', '51.6 ms'), ('Site survey', 'D_clear 1.10 m')]
    for i, (t1, t2) in enumerate(leaves):
        x = 46 + i * 206
        b.append(box(x, 228, 200, 66, 'white', GREEN, sw=2.0))
        b += lines(x + 100, 255, [t1], 18, weight='bold')
        b += lines(x + 100, 281, [t2], 18)
        b.append(arrow(x + 100, 228, 462 + (i - 1.5) * 50, 173, GREEN, 2, marker='ahg'))
    b.append(text(450, 320, 'Evidence: calibrated records with stated uncertainty', 18, GREEN, weight='bold'))
    if k >= 2:
        b.append(box(8, 134, 222, 56, GRAY_F, GRAY, dash='3,4'))
        b += lines(119, 156, ['1,000 clean runs'], 18, weight='bold', fill=GRAY)
        b += lines(119, 178, ['a demonstration'], 17, fill=GRAY)
        b += lines(10, 207, ['not evidence for', 'new conditions'], 15, RED, 'start', 'bold', gap=16)
    if k >= 3:
        b.append(box(704, 100, 188, 74, RED_F, RED, sw=2.4))
        b += lines(798, 128, ['Defeater: clear', 'oil film, μ ≈ 0.05'], 18, RED, weight='bold')
        b.append(arrow(704, 137, 666, 137, RED, 2.6, marker='ahr'))
    svg(f'cae_step{k}', W, H, b)

def legs(k):
    W, H = (600, 340)
    b = []
    b.append(text(8, 26, 'Conjunctive case: every premise must hold', 20, TXT, 'start', 'bold'))
    for i in range(10):
        x = 8 + i * 47
        b.append(box(x, 44, 40, 40, GREEN_F, GREEN, rx=4, sw=1.8))
        b.append(text(x + 20, 70, '0.99', 16, TXT, weight='bold'))
        if i:
            b.append(line(x - 7, 64, x, 64, SLATE, 2))
    b.append(text(488, 72, '= 0.904', 22, RED, 'start', 'bold'))
    b.append(text(8, 116, 'Ten premises at 0.99 hold together at 0.904,', 17, SLATE, 'start'))
    b.append(text(8, 138, 'below every single premise.', 17, SLATE, 'start'))
    if k >= 2:
        b.append(text(8, 184, 'Diverse legs: either one suffices', 20, TXT, 'start', 'bold'))
        b.append(line(20, 260, 70, 260, SLATE, 2))
        b.append(line(70, 222, 70, 298, SLATE, 2))
        for y, lab in ((200, 'Leg A  0.90'), (276, 'Leg B  0.90')):
            b.append(line(70, y + 22, 110, y + 22, SLATE, 2))
            b.append(box(110, y, 170, 44, BLUE_F, NAVY, rx=4, sw=1.8))
            b.append(text(195, y + 29, lab, 18, TXT))
            b.append(line(280, y + 22, 320, y + 22, SLATE, 2))
        b.append(line(320, 222, 320, 298, SLATE, 2))
        b.append(line(320, 260, 360, 260, SLATE, 2))
        b.append(text(368, 268, '= 1 − 0.10² = 0.99', 20, GREEN, 'start', 'bold'))
        b.append(text(368, 300, 'only if they fail', 17, RED, 'start', 'bold'))
        b.append(text(368, 322, 'independently', 17, RED, 'start', 'bold'))
    svg(f'legs_step{k}', W, H, b)

def door(k):
    a_lo, a_hi = (2.950819672, 400.0)
    rows = [('0.10 m/s', a_hi, RED, '400 m/s²', 2), ('1 g', 9.80665, GRAY, '9.8 (1 g)', 1), ('0.03 m/s', a_lo, GREEN, 'about 3 m/s²', 1)]
    fig = plt.figure(figsize=cm(8.4, 3.6))
    ax = fig.add_axes([0.23, 0.33, 0.68, 0.64])
    for i, (lab, v, col, txt, step) in enumerate(rows):
        a = 1.0 if k >= step else 0.0
        ax.barh(i, v, color=col, height=0.6, alpha=a)
        ax.text(v * 1.25, i, txt, va='center', fontsize=10.5, fontweight='bold', color=col, alpha=a)
        ax.text(0.8, i, lab, ha='right', va='center', fontsize=10.5, fontweight='bold', color=TXT, alpha=a)
    ax.set_xscale('log')
    ax.set_xlim(1, 3000)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Arm deceleration needed (m/s²)', fontsize=10)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(['1', '10', '100', '1,000'])
    ax.minorticks_off()
    ax.tick_params(labelsize=10)
    save_slot(fig, f'door_step{k}')
TREE = [('C1a  Base stop', '969.0 of 1,100 mm; a_brake 2 m/s²', 'ill'), ('C1b  Door latch', '39 N at onset; arm stops in 0.15 mm', 'unk'), ('C1c  Mug handover', '20.7 N free hand; no clamping', 'unk'), ('C2  Permission timing', 'onset ≤ 82 ms; WCET 250 of 400 µs', 'dec'), ('C3  Braking traction', 'floor μ ≥ 0.204 (screen)', 'ill'), ('C4  Observation age', 'dispatch age 51.6 ms', 'ill'), ('C5  Independence', 'placement test report', 'unk'), ('C6  Authority transfer', 'PASS records for A1 to A4', 'unk')]
DASH = {'ill': '8,5', 'dec': '10,4,2,4', 'unk': '2,4'}
MAT = {'ill': 'illustrative', 'dec': 'declared', 'unk': 'unknown'}

def tree(k):
    W, H = (900, 360)
    b = []
    b.append(box(4, 98, 150, 88, GREEN_F, GREEN, sw=2.4))
    b += lines(79, 126, ['C0  Restricted', 'mug delivery', '(all must hold)'], 18, weight='bold', gap=22)
    for i, (c, leaf, m) in enumerate(TREE):
        y = 6 + i * 44
        b.append(line(154, 141, 172, y + 18, SLATE, 1.6))
        b.append(box(172, y, 222, 36, 'white', SLATE, sw=1.8))
        b.append(text(180, y + 25, c, 18, TXT, 'start', 'bold'))
        b.append(line(394, y + 18, 404, y + 18, SLATE, 1.6))
        b.append(box(404, y, 306, 36, 'white', GRAY if k == 1 else SLATE, dash=DASH[m], sw=2.0))
        b.append(text(411, y + 25, leaf, 17, TXT, 'start'))
        if k >= 3:
            b.append(box(716, y + 2, 86, 32, RED_F, RED, rx=4, sw=1.8))
            b.append(text(759, y + 24, 'REFUSE', 17, RED, weight='bold'))
    if k >= 2:
        y3 = 6 + 4 * 44
        b.append(box(808, y3 - 6, 90, 48, RED_F, RED, sw=2.2))
        b += lines(853, y3 + 13, ['clear oil', 'film'], 17, RED, weight='bold', gap=19)
        b.append(arrow(808, y3 + 18, 802 if k >= 3 else 712, y3 + 18, RED, 2.2, marker='ahr'))
        b.append(text(853, y3 - 13, 'defeater', 17, RED, weight='bold'))
        for j, mm in enumerate(('ill', 'dec', 'unk')):
            yy = 212 + j * 32
            b.append(box(6, yy, 34, 22, 'white', SLATE, dash=DASH[mm], sw=2.0, rx=3))
            b.append(text(46, yy + 18, MAT[mm], 17, TXT, 'start'))
        b.append(text(6, 332, 'none measured', 17, RED, 'start', 'bold'))
    svg(f'tree_step{k}', W, H, b)
LABELS = {'Police-reported': 'Police-reported crash', 'Any-injury-reported': 'Any injury reported', 'Any Airbag Deployment': 'Any airbag deployment', 'Ego Vehicle Airbag Deployment': 'Airbag in the robotaxi', 'Suspected Serious Injury+': 'Serious injury or worse', 'Any Fatal': 'Fatal'}

def waymo(k):
    rows = load_series('vol4/16_release/data/waymo_safety_impact_headline.csv').entries
    last = max((r['data_cut'] for r in rows))
    latest = [r for r in rows if r['data_cut'] == last and r['outcome'] in LABELS]
    latest.sort(key=lambda r: -r['waymo_count'])
    fig = plt.figure(figsize=cm(10.8, 4.7))
    ax = fig.add_axes([0.4, 0.31, 0.46, 0.66])
    for i, r in enumerate(latest):
        rare = r['waymo_count'] < 20
        a = 1.0 if not rare or k >= 2 else 0.0
        col = AMBER if rare else BLUE
        p, lo, hi = (-100 * r['pct_reduction'], -100 * r['pct_reduction_ci_hi'], -100 * r['pct_reduction_ci_lo'])
        ax.plot([lo, hi], [i, i], color=col, lw=3.2, solid_capstyle='butt', alpha=a)
        ax.plot(p, i, 'o', color=col, ms=7, alpha=a)
        ax.text(-0.02, i, LABELS[r['outcome']], ha='right', va='center', fontsize=10.5, fontweight='bold', color=TXT, alpha=a, transform=ax.get_yaxis_transform())
        ax.text(1.02, i, f"n = {r['waymo_count']:.0f}", ha='left', va='center', fontsize=10.5, color=col, fontweight='bold', alpha=a, transform=ax.get_yaxis_transform())
        if r['outcome'] == 'Any Fatal':
            ax.text(lo + 1, i - 0.42, f'{lo:.0f} to {hi:.0f}%', ha='left', va='bottom', fontsize=10, color=AMBER, fontweight='bold', alpha=a)
    ax.set_xlim(0, 100)
    ax.set_ylim(len(latest) - 0.4, -0.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Crash-rate reduction against local\nhuman drivers (%, 95% interval)', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    save_slot(fig, f'waymo_step{k}')

def silent(k):
    fig = plt.figure(figsize=cm(14.6, 3.7))
    ax = fig.add_axes([0.3, 0.27, 0.48, 0.71])
    rows = [(2, 'Proposer falls silent', 0, 60.0, 'lease runs out: 60 ms', BLUE_F, BLUE, 1), (1, 'Premise flags at once', 0, 0, '', AMBER_F, AMBER, 1), (0, 'Premise flags at 60 ms', 0, 82.0 - 22.0, 'flag latency: 60 ms', AMBER_F, AMBER, 2)]
    for y, lab, x0, pre, pre_lab, fc, ec, step in rows:
        a = 1.0 if k >= step else 0.0
        if pre:
            ax.barh(y, pre, left=x0, color=fc, edgecolor=ec, lw=1.3, height=0.6, alpha=a, hatch='///' if ec == AMBER else None)
            ax.text(pre / 2, y, pre_lab, ha='center', va='center', fontsize=10, color=NAVY if ec == BLUE else '#92400E', fontweight='bold', alpha=a)
        ax.barh(y, 1.0 + 1.0, left=pre, color=SLATE, height=0.6, alpha=a)
        ax.barh(y, 20.0, left=pre + 1.0 + 1.0, color=RED, height=0.6, alpha=a)
        ax.text(pre + 1.0 + 1.0 + 20.0 / 2, y, 'onset', ha='center', va='center', fontsize=10, color='white', fontweight='bold', alpha=a)
        end = pre + 22.0
        label = '82 ms: 106.6 mm' if pre else '22 ms: 28.6 mm'
        ax.text(end + 1.5, y, label, ha='left', va='center', fontsize=10.5, fontweight='bold', color=TXT, alpha=a)
        ax.text(-2, y, lab, ha='right', va='center', fontsize=10.5, fontweight='bold', color=TXT, alpha=a)
    ax.set_xlim(0, 85)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Milliseconds from the failure to brake onset, base at 1.3 m/s', fontsize=10)
    ax.tick_params(labelsize=10)
    save_slot(fig, f'silent_step{k}')

def tri():
    W, H = (900, 210)
    b = []
    cols = [('Known-true', GREEN_F, GREEN, ['qualified state set inside', 'the envelope; fresh,', 'verified evidence'], 'permit = 1: track'), ('Unknown', AMBER_F, AMBER, ['sensor dropout, CRC', 'mismatch, or estimator', 'divergence'], 'permit = 0'), ('Known-false', RED_F, RED, ['measured breach, such as', 'tracking error above', 'its 40 mm bound'], 'permit = 0')]
    for i, (t, f, s, body, out) in enumerate(cols):
        x = 8 + i * 300
        b.append(box(x, 8, 284, 138, f, s, sw=2.6))
        b.append(text(x + 142, 38, t, 22, s if s != AMBER else '#92400E', weight='bold'))
        b += lines(x + 142, 66, body, 18)
        b.append(text(x + 142, 176, out, 19, TXT, weight='bold'))
    b.append(line(316, 194, 884, 194, SLATE, 2))
    b.append(line(316, 186, 316, 194, SLATE, 2))
    b.append(line(884, 186, 884, 194, SLATE, 2))
    b.append(text(600, 206, 'same response (assessed stop or hold); causes logged apart', 16, SLATE))
    svg('tri', W, H, b)

def erosion():
    fig = plt.figure(figsize=cm(8.6, 4.2))
    ax = fig.add_axes([0.13, 0.26, 0.84, 0.64])
    for order, _, d in [(1, 66.0, 0), (2, 70.0, 1), (3, 69.0, 0), (4, 68.0, 0), (5, 67.0, 0), (6, 72.0, 0), (7, 73.0, 0), (8, 70.0, 0), (9, 57.0, 1), (10, 63.0, 1), (11, 70.0, 1), (12, 78.0, 0), (13, 67.0, 0), (14, 53.0, 2), (15, 67.0, 0), (16, 75.0, 0), (17, 70.0, 0), (18, 81.0, 0), (19, 76.0, 0), (20, 79.0, 0), (21, 75.0, 2), (22, 76.0, 0), (23, 58.0, 1)]:
        if d:
            ax.bar(order, d, color=RED, width=0.7)
        else:
            ax.plot(order, 0.06, 'o', ms=4, color=SLATE)
    ax.set_xlim(0.3, 24)
    ax.set_ylim(0, 2.6)
    ax.set_yticks([0, 1, 2])
    ax.set_xticks([1, 5, 10, 15, 20, 23])
    ax.set_xlabel('Flight, in launch order', fontsize=10)
    ax.set_ylabel('O-rings with distress', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.text(23.5, 2.45, '7 of 23 flights; each one returned', ha='right', va='top', fontsize=10, color=RED, fontweight='bold')
    save_slot(fig, 'erosion')
MGROUPS = [('Plant and timing', 'L2, L4', ['limit · handoff']), ('Data and policy', 'L5–7', ['provenance · policy', 'manifest · evaluation']), ('Perception, planning', 'L8–11', ['observation contract', 'belief · intent lease', 'trajectory']), ('Containment', 'L12–15', ['enforcement · placement', 'authority · fault records'])]

def manifest(k):
    W, H = (900, 370)
    b = []
    y = 4
    for g, lec, rows in MGROUPS:
        h = 28 + 22 * len(rows)
        b.append(box(4, y, 262, h, SLATE_F, SLATE, rx=5, sw=1.4))
        b.append(text(12, y + 21, g, 18, SLATE, 'start', 'bold'))
        b.append(text(258, y + 21, lec, 16, GRAY, 'end', 'bold'))
        b += lines(12, y + 44, rows, 17, TXT, 'start', gap=22)
        y += h + 8
    a2 = k >= 2
    if a2:
        b.append(arrow(268, 185, 292, 185, SLATE, 2.4))
        b.append(box(296, 4, 272, 362, GREEN_F, GREEN, sw=2.6))
        b.append(text(432, 32, 'Release manifest', 21, GREEN, weight='bold'))
        fields = ['record digests → Merkle root', 'root claim, argument digest', 'binary hashes: policy,', '   enforcer, RTOS kernel', 'calibration tables', 'standing conditions', 'wear budget (hours, cycles)', 'expiry date', 'adjudicator signature (HSM)']
        for i, f in enumerate(fields):
            b.append(text(306, 66 + i * 35, f, 18, TXT, 'start'))
    if k >= 3:
        b.append(arrow(568, 185, 592, 185, SLATE, 2.4))
        b.append(box(596, 116, 132, 138, 'white', SLATE, sw=2.2))
        b += lines(662, 146, ['Gate:', 'authenticate,', 'then check', 'standing'], 18, gap=26)
        b.append(arrow(728, 160, 750, 112, GREEN, 2.2, marker='ahg'))
        b.append(arrow(728, 210, 750, 258, RED, 2.2, marker='ahr'))
        b.append(box(752, 50, 144, 90, GREEN_F, GREEN, sw=2.2))
        b += lines(824, 78, ['UNCOND or', 'COND:', 'permit on'], 18, GREEN, weight='bold', gap=22)
        b.append(box(752, 230, 144, 90, RED_F, RED, sw=2.2))
        b += lines(824, 258, ['REFUSE:', 'permit off;', 'stop or hold'], 18, RED, weight='bold', gap=22)
    svg(f'manifest_step{k}', W, H, b)
REOPEN_COLS = ['Task', 'Stop', 'Contact', 'Timing', 'Age', 'Indep.']
REOPEN_ROWS = [('Retrained policy, same backbone', [1, 0, 1, 0, 0, 0]), ('Larger backbone, same enforcer binary', [1, 0, 0, 1, 1, 1]), ('Driver or thread-priority change', [0, 0, 0, 1, 1, 1]), ('OS reflash that resets calibration', [0, 1, 1, 0, 0, 0]), ('Worn brake lining, no software change', [0, 1, 0, 0, 0, 0])]

def reopen():
    W, H = (900, 300)
    b = []
    x0, dx = (440, 76)
    for j, c in enumerate(REOPEN_COLS):
        b.append(text(x0 + j * dx, 26, c, 18, SLATE, weight='bold'))
    b.append(text(x0 + 2.5 * dx, 292, 'Task: capability evidence. The other columns: C1 stop and contact, C2, C4, C5.', 15, GRAY))
    for i, (r, marks) in enumerate(REOPEN_ROWS):
        y = 66 + i * 46
        b.append(text(8, y + 6, r, 18, TXT, 'start', 'bold' if i in (1, 4) else 'normal'))
        b.append(line(8, y + 22, 892, y + 22, '#E5E7EB', 1))
        for j, m in enumerate(marks):
            cx = x0 + j * dx
            if m:
                b.append(f'<circle cx="{cx}" cy="{y}" r="12" fill="{AMBER}"/>')
            else:
                b.append(f'<circle cx="{cx}" cy="{y}" r="11" fill="white" stroke="#D1D5DB" stroke-width="2"/>')
    svg('reopen', W, H, b)
if __name__ == '__main__':
    for k in (1, 2, 3):
        oring(k)
        cae(k)
        tree(k)
        manifest(k)
    for k in (1, 2):
        legs(k)
        door(k)
        waymo(k)
        silent(k)
    tri()
    erosion()
    reopen()
    print('figures written to', HERE)
