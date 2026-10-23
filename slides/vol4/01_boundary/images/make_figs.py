#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 1 (The Causal Boundary), Volume IV.

Every figure is drawn at the size of its slot on the 16:9 slide (the slide
body is 15 cm, 425 pt, wide), so nothing is scaled when it is placed:
SVG diagrams use point units (viewBox in pt, width and height in pt) and
matplotlib plots use their display size in inches. Labels are 8.5 to 10 pt.

Sources: every number a figure prints or plots is read from MLSysIM through
calcs.py (deckvals val/num), which this file imports:
  - Running machine: Embodied.MobileManipulator and Embodied.Scenario, through
    Scenarios.WarehouseAisleStopAtDriveLimit (The Causal Boundary, Napkin Math 1.1).
  - Tempe: Literature.IncidentReports (NTSB HAR-19/03), through
    Scenarios.TempeActionSuppression and its ADS-cap variant.
  - Training compute: Epoch AI via vol4/01_boundary/data/epoch_models_by_domain.csv, as the
    chapter's figure 2.
  - The Tempe classification track (tempe_track) keeps the NTSB table's
    intermediate times as figure data (HAR-19/03, pp. 15-16).
Diagram coordinates, axis ranges, and tick positions are layout, not data.
  - Scope test, command chain, five levels, record chain: the chapter's
    §§1.4.1, 1.6, 1.7, 1.11 (figures 3, 6, 7, 9), redrawn at slide size.
Colors follow the Volume IV causal-boundary palette (volumes/vol4/palette.md):
navy and blue for learned proposals, green for the deterministic permission
path, slate for the plant, petrol for sensing, red for hazards and limits.

Run from this folder: python3 make_figs.py
"""
from mlsysim.core.series_loader import load_series
import mlsysim
import datetime as dt
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from mlsysim.core.series_loader import load_series
plt = sp.plt
OUT = Path(__file__).parent
NAVY, BLUE, GREEN, RED = ('#1F407A', '#2563EB', '#059669', '#DC2626')
PETROL, SLATE, AMBER, INK = ('#007A87', '#334155', '#D97706', '#1A1A2E')
GRAY, LGRAY, ORANGE = ('#666666', '#B8B8B8', '#E67817')
FILL = {'navy': '#EFF6FF', 'green': '#ECFDF5', 'red': '#FEF2F2', 'slate': '#F1F5F9', 'petrol': '#F0FDFA', 'amber': '#FFF7ED', 'gray': '#F3F4F6'}
FONT = 'Helvetica, Arial, sans-serif'
plt.rcParams.update({'font.size': 9.5, 'axes.labelsize': 9.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.edgecolor': '#888888'})

class Svg:

    def __init__(self, w, h, scale=1.0):
        self.w, self.h, self.parts, self.k = (w, h, [], scale)

    def rect(self, x, y, w, h, fill='white', stroke=SLATE, sw=1.0, rx=3, dash=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>')

    def text(self, x, y, s, size=9.5, color=INK, anchor='middle', weight='normal', italic=False):
        st = ' font-style="italic"' if italic else ''
        for i, line in enumerate(s.split('\n')):
            self.parts.append(f'<text x="{x:.1f}" y="{y + i * size * 1.18:.1f}" font-family="{FONT}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"{st}>{line}</text>')

    def line(self, x1, y1, x2, y2, color=SLATE, sw=1.2, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ''
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')

    def path(self, d, color=SLATE, sw=1.2, dash=None, arrow=False, fill='none'):
        da = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ''
        self.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="{fill}"{da}{m}/>')

    def circle(self, cx, cy, r, fill, stroke, sw=1.2, opacity=1):
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>')

    def save(self, name):
        cols = {SLATE, RED, GREEN, BLUE, NAVY, AMBER, GRAY, PETROL, INK}
        markers = ''.join((f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in cols))
        W, H = (self.w * self.k, self.h * self.k)
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.1f}pt" height="{H:.1f}pt" viewBox="0 0 {W:.1f} {H:.1f}"><defs>{markers}</defs><rect x="0" y="0" width="{W:.1f}" height="{H:.1f}" fill="white"/><g transform="scale({self.k})">' + ''.join(self.parts) + '</g></svg>'
        (OUT / f'{name}.svg').write_text(svg)

def save_plot(fig, name):
    fig.savefig(OUT / f'{name}.svg', format='svg', bbox_inches=None, facecolor='white')
    plt.close(fig)

def chain():
    boxes = [('tensor\n(a proposal)', NAVY, 'navy', True), ('permission\npath admits', GREEN, 'green', False), ('position and\ncurrent loops', GREEN, 'green', False), ('PWM register\nwrite', GREEN, 'green', False), ('current in\nthe winding', SLATE, 'slate', False), ('torque moves\nmass', SLATE, 'slate', False)]
    W, bw, y0, bh = (404, 60, 40, 40)
    gap = (W - 6 * bw) / 5
    for step in (1, 2, 3):
        s = Svg(W, 124)
        n = 4 if step == 1 else 6
        for i in range(n):
            label, stroke, fill, dashed = boxes[i]
            x = i * (bw + gap)
            s.rect(x, y0, bw, bh, fill=FILL[fill], stroke=stroke, sw=1.3, rx=0 if fill == 'slate' else 3, dash='4,2' if dashed else None)
            s.text(x + bw / 2, y0 + 16, label, size=9)
            if i < n - 1:
                s.line(x + bw + 1, y0 + bh / 2, x + bw + gap - 1, y0 + bh / 2, color=SLATE, arrow=True)
        xb = 4 * (bw + gap) - gap / 2
        s.path(f'M2,{y0 - 8} L2,{y0 - 13} L{xb - 6},{y0 - 13} L{xb - 6},{y0 - 8}', color=GREEN, sw=1.1)
        s.text((xb - 4) / 2, y0 - 18, 'rewriting a command here costs no physical work', size=9, color=GREEN)
        if step >= 2:
            s.path(f'M{xb + 6},{y0 - 8} L{xb + 6},{y0 - 13} L{W - 2},{y0 - 13} L{W - 2},{y0 - 8}', color=RED, sw=1.1)
            s.text((xb + W) / 2, y0 - 18, 'changing it here takes work', size=9, color=RED)
        if step >= 3:
            s.line(xb, y0 - 6, xb, y0 + bh + 22, color=RED, sw=1.8, dash='4,2')
            s.text(xb, y0 + bh + 34, 'causal boundary', size=10, color=RED, weight='bold')
            s.text(xb - 6, y0 + bh + 16, 'digital', size=9, color=GRAY, anchor='end')
            s.text(xb + 6, y0 + bh + 16, 'physical', size=9, color=GRAY, anchor='start')
        s.save(f'chain_step{step}')

def recovery():
    W = 262
    for step in (1, 2):
        s = Svg(W, 138)

        def row(y, title, states, colors, arrows, note, ncol):
            s.text(0, y - 6, title, size=9.5, anchor='start', weight='bold', color=INK)
            xs = [2, 104, 206]
            for i, (lab, c) in enumerate(zip(states, colors)):
                s.rect(xs[i], y, 54, 26, fill=FILL[c], stroke={'green': GREEN, 'red': RED, 'slate': SLATE}[c], rx=0, sw=1.3)
                s.text(xs[i] + 27, y + 17, lab, size=9.5)
            for i, a in enumerate(arrows):
                s.line(xs[i] + 55, y + 13, xs[i + 1] - 1, y + 13, color=SLATE, arrow=True)
                s.text((xs[i] + 55 + xs[i + 1]) / 2, y + 9, a, size=8.5, color=GRAY)
            s.text(W - 2, y + 40, note, size=9, color=ncol, anchor='end')
        row(16, 'A database transaction', ['state S0', 'state S1', 'state S0'], ['slate', 'slate', 'green'], ['write', 'abort'], 'the same state again', GREEN)
        if step >= 2:
            row(86, 'A machine that moved', ['state S0', 'state S1', 'state S2'], ['slate', 'slate', 'red'], ['torque', 'brake'], 'a new state: heat, wear', RED)
        s.save(f'recovery_step{step}')

def delay_bar(name, scanner=False):
    fig, ax = plt.subplots(figsize=(5.6, 1.15))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.7, bottom=0.3)
    age, backbone, lease = (51.6, 22.0, 60.0)
    tick, bus, onset = (1.0, 1.0, 20.0)
    segs = [(backbone, ORANGE, 'backbone\n22'), (29.6, LGRAY, 'rest of age 29.6'), (lease, SLATE, 'chunk lease 60'), (tick + bus, GREEN, ''), (onset, '#8C8C8C', 'brake\nonset 20')]
    x = 0
    for i, (w, col, lab) in enumerate(segs):
        gone = scanner and i in (0, 2)
        ax.barh(0, w, left=x, color='white' if gone else col, edgecolor=col if gone else 'white', hatch='////' if gone else None, height=0.62, lw=1.2)
        if lab:
            tc = INK if col in (LGRAY,) or gone else 'white'
            ax.text(x + w / 2, 0, lab, ha='center', va='center', color=tc, fontsize=9, fontweight='bold', linespacing=0.95, bbox=dict(fc='white', ec='none', pad=1.0) if gone else None)
        x += w
    xt = age + lease + (tick + bus) / 2
    ax.annotate('tick 1 + bus 1', xy=(xt, -0.32), xytext=(xt, -0.62), ha='center', va='top', fontsize=9, color=GREEN, arrowprops=dict(arrowstyle='-', color=GREEN, lw=1.0), annotation_clip=False)
    if scanner:
        ax.text(0, 0.42, 'with a scanner stop, backbone and lease leave: 82 ms, 123 mm at 1.5 m/s', ha='left', va='bottom', fontsize=9.5, color=RED, fontweight='bold')
    else:
        ax.text(0, 0.42, 'pre-brake delay 133.6 ms = 200.4 mm of travel at 1.5 m/s', ha='left', va='bottom', fontsize=9.5, color=INK, fontweight='bold')
    ax.set_xlim(0, 136)
    ax.set_ylim(-0.4, 0.4)
    ax.axis('off')
    save_plot(fig, name)

def crossover():
    t, a, d_clear = (0.1336, 2.0, 1100.0)
    v = np.linspace(0, 2.0, 401)
    delay, brake = (v * t * 1000, v ** 2 / (2 * a) * 1000)
    vs = 0.5344
    v_op, d_op = (1.5, 762.9)
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(3.75, 2.25))
        fig.subplots_adjust(left=0.19, right=0.8, top=0.95, bottom=0.21)
        ax.plot(v, delay, color=AMBER, lw=2.2)
        ax.text(2.03, delay[-1], 'delay\ntravel', color=AMBER, fontsize=9, fontweight='bold', va='center')
        if step >= 2:
            ax.plot(v, brake, color=SLATE, lw=2.2)
            ax.text(2.03, brake[-1], 'braking\ntravel', color=SLATE, fontsize=9, fontweight='bold', va='center')
            ax.plot([vs], [vs * t * 1000], 'o', ms=6, mfc='white', mec=INK, mew=1.5, zorder=5)
            ax.annotate('crossover 0.53 m/s', xy=(vs, vs * t * 1000), xytext=(0.08, 520), fontsize=9, fontweight='bold', color=INK, arrowprops=dict(arrowstyle='-', color=INK, lw=0.9))
        if step >= 3:
            ax.plot(v, delay + brake, color=INK, lw=2.4)
            ax.text(2.03, delay[-1] + brake[-1], 'sum', color=INK, fontsize=9, fontweight='bold', va='center', clip_on=False)
            ax.axhline(d_clear, color=RED, lw=1.6, ls='--')
            ax.text(0.04, d_clear + 30, 'clear distance 1,100 mm', color=RED, fontsize=9, fontweight='bold', va='bottom')
            ax.plot([v_op], [d_op], 'o', ms=7, color=INK, zorder=5)
            ax.text(v_op - 0.05, d_op + 40, '762.9 mm', fontsize=9, fontweight='bold', ha='right', va='bottom')
        ax.set_xlim(0, 2.0)
        ax.set_ylim(0, 1600)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        ax.set_yticks([0, 400, 800, 1200, 1600])
        ax.set_xlabel('speed when the worker appears (m/s)')
        ax.set_ylabel('distance (mm)')
        save_plot(fig, f'crossover_step{step}')

def tempe_bars():
    rows = [('traveled during the 1 s suppression', 'tempe_supp_m', RED), ("full stop at the road's 0.92 g", 'tempe_brake_road_m', SLATE), ('full stop at the ADS cap, 0.71 g', 'tempe_brake_cap_m', GRAY)]
    fig, ax = plt.subplots(figsize=(5.6, 1.55))
    fig.subplots_adjust(left=0.43, right=0.9, top=0.97, bottom=0.27)
    for i, (lab, key, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, {'traveled during the 1 s suppression': 18.7, "full stop at the road's 0.92 g": 20.7, 'full stop at the ADS cap, 0.71 g': 26.6}[lab], color=col, height=0.62)
        ax.text(-0.6, y, lab, ha='right', va='center', fontsize=9.5, color=INK, fontweight='bold')
        ax.text({'traveled during the 1 s suppression': 18.7, "full stop at the road's 0.92 g": 20.7, 'full stop at the ADS cap, 0.71 g': 26.6}[lab] + 0.4, y, f"""{ {'traveled during the 1 s suppression': '18.7', "full stop at the road's 0.92 g": '20.7', 'full stop at the ADS cap, 0.71 g': '26.6'}[lab]} m""", ha='left', va='center', fontsize=9.5, color=col, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel('meters, from 43.2 mph (19.3 m/s)')
    ax.spines['left'].set_visible(False)
    save_plot(fig, 'tempe_bars')

def epoch():
    rows = [r for r in load_series('vol4/01_boundary/data/epoch_models_by_domain.csv').entries if str(r['publication_date']) >= '2018' and r['training_flop']]
    date = lambda r: dt.datetime.fromisoformat(str(r['publication_date'])[:10])
    lang = [r for r in rows if r['domain_group'] == 'Language']
    robo = [r for r in rows if r['domain_group'] == 'Robotics' and 'Cosmos' not in r['model']]
    fig, ax = plt.subplots(figsize=(3.75, 2.25))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.95, bottom=0.15)
    ax.scatter([date(r) for r in lang], [r['training_flop'] for r in lang], s=7, color=GRAY, alpha=0.35, lw=0)
    ax.scatter([date(r) for r in robo], [r['training_flop'] for r in robo], s=26, color=GREEN, zorder=4)
    ax.text(dt.datetime(2019, 3, 1), 3e+25, 'language models', color=GRAY, fontsize=9, fontweight='bold')
    ax.text(dt.datetime(2018, 3, 1), 4e+16, 'robotics models', color=GREEN, fontsize=9, fontweight='bold')
    for name, (lab, (dx, dy)) in {'OpenVLA': ('OpenVLA', (-48, 14)), 'GR00T N1 2B': ('GR00T N1', (-12, -26))}.items():
        r = next((r for r in robo if r['model'] == name))
        ax.annotate(lab, (date(r), r['training_flop']), xytext=(dx, dy), textcoords='offset points', fontsize=9, fontweight='bold', color=GREEN, arrowprops=dict(arrowstyle='-', color=GREEN, lw=1.0))
    ax.set_yscale('log')
    ax.set_ylim(1000000000000000.0, 1e+28)
    ax.set_yticks([1e+16, 1e+19, 1e+22, 1e+25, 1e+28])
    ax.set_ylabel('training compute (FLOP)')
    ax.set_xlim(dt.datetime(2018, 1, 1), dt.datetime(2026, 6, 1))
    ax.set_xticks([dt.datetime(y, 1, 1) for y in (2018, 2020, 2022, 2024, 2026)], ['2018', '2020', '2022', '2024', '2026'])
    save_plot(fig, 'epoch_compute')

def venn():
    W, H = (256, 210)
    L, P, D, r = ((96, 82), (160, 82), (128, 136), 60)
    for step in (1, 2, 3):
        s = Svg(W, H, scale=0.85)
        s.circle(*L, r, FILL['navy'], NAVY, opacity=0.8)
        s.circle(*P, r, FILL['slate'], SLATE, opacity=0.8)
        s.circle(*D, r, FILL['green'], GREEN, opacity=0.8)
        s.text(70, 62, 'learned\nmodel', size=9.5, color=NAVY, weight='bold')
        s.text(186, 62, 'physical\nfeedback', size=9.5, color=SLATE, weight='bold')
        s.text(128, 176, 'delegated\nauthority', size=9.5, color=GREEN, weight='bold')
        if step >= 2:
            s.text(128, 50, 'advisory', size=9.5, color=INK)
            s.text(90, 120, 'digital\nautonomy', size=9.5, color=INK)
            s.text(166, 120, 'classical\ncontrol', size=9.5, color=INK)
        if step >= 3:
            s.rect(104, 82, 48, 24, fill='white', stroke=RED, sw=1.4)
            s.text(128, 92, 'physical', size=9, color=RED, weight='bold')
            s.text(128, 102.5, 'AI', size=9, color=RED, weight='bold')
        s.save(f'venn_step{step}')

def tempe_track():
    cls = [(-5.6, -5.2, 'vehicle', PETROL), (-5.2, -4.2, 'other', LGRAY), (-4.2, -3.8, 'vehicle', PETROL), (-3.8, -2.6, 'vehicle, other\nalternating', None), (-2.6, -1.5, 'bicycle', AMBER), (-1.5, -1.2, 'other', LGRAY), (-1.2, 0.0, 'bicycle', AMBER)]
    above = {0: -5.4, 2: -4.0, 5: -1.35}
    changes = [-5.2, -4.2, -2.6, -1.5, -1.2]
    path = [(-5.6, -5.2, 'none', '#E5E7EB', INK), (-5.2, -3.9, 'static', '#E5E7EB', INK), (-3.9, -2.6, 'next lane\nor static', '#E5E7EB', INK), (-2.6, -1.5, 'static, then\nnext lane', '#E5E7EB', INK), (-1.5, -1.2, '', '#FDE68A', INK), (-1.2, 0.0, "in the\nSUV's lane", '#FECACA', RED)]
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(5.75, 2.7))
        fig.subplots_adjust(left=0.13, right=0.985, top=0.89, bottom=0.23)
        for i, (a, b, lab, col) in enumerate(cls):
            if col is None:
                ax.barh(1, b - a, left=a, height=0.62, color='white', edgecolor=PETROL, hatch='//', lw=1.0)
            else:
                ax.barh(1, b - a, left=a, height=0.62, color=col, edgecolor='white')
            if i in above:
                ax.text(above[i], 1.36, lab, ha='center', va='bottom', fontsize=9.5, color=INK)
            else:
                ax.text((a + b) / 2, 1, lab, ha='center', va='center', fontsize=9.5, linespacing=0.95, color='white' if col in (PETROL, AMBER) else INK, fontweight='bold', bbox=dict(fc='white', ec='none', pad=0.6) if col is None else None)
        for d in changes:
            ax.plot([d, d], [0.68, 1.32], color=RED, lw=1.8, solid_capstyle='butt')
        if step >= 2:
            for a, b, lab, col, tc in path:
                ax.barh(0, b - a, left=a, height=0.62, color=col, edgecolor='white')
                if lab:
                    ax.text((a + b) / 2, 0, lab, ha='center', va='center', fontsize=9.5, color=tc, fontweight='bold', linespacing=0.95)
            ax.annotate('', xy=(-1.2, -0.42), xytext=(-0.2, -0.42), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.3), annotation_clip=False)
            ax.text(-1.25, -0.42, 'suppression, 1 s', ha='right', va='center', fontsize=9.5, color=RED, fontweight='bold')
        ax.set_yticks([0, 1], ['predicted\npath', 'class'] if step >= 2 else ['', 'class'])
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(-0.55, 1.62)
        ax.set_xlim(-5.6, 0.0)
        ax.set_xticks([-5, -4, -3, -2, -1, 0])
        ax.set_xlabel('seconds before impact')
        ax.spines['left'].set_visible(False)
        save_plot(fig, f'tempe_track_step{step}')

def convergence():
    W, H = (310, 150)
    threats = [('slick floor', 'the traction premise fails'), ('spoofed bus packet', 'a false measurement'), ('hallucinated free path', 'a confident wrong proposal')]
    demands = [('fresh, authenticated', 'physical state'), ('authority to refuse,', 'with a feasible fallback'), ('a synchronized record', 'of what was admitted')]
    for step in (1, 2):
        s = Svg(W, H)
        for i, (a, b) in enumerate(threats):
            y = 8 + i * 48
            s.rect(0, y, 108, 36, fill=FILL['red'], stroke=RED, rx=0)
            s.text(54, y + 15, a, size=9, color=RED, weight='bold')
            s.text(54, y + 27, b, size=8.5, color=INK)
            s.line(109, y + 18, 124, 75 + (i - 1) * 10, color=SLATE, sw=1.0, arrow=True)
        s.rect(126, 50, 58, 50, fill=FILL['green'], stroke=GREEN, sw=1.5)
        s.text(155, 71, 'permission\npath', size=9, color=GREEN, weight='bold')
        if step >= 2:
            for i, (a, b) in enumerate(demands):
                y = 8 + i * 48
                s.line(185, 75 + (i - 1) * 10, 200, y + 18, color=GREEN, sw=1.0, arrow=True)
                s.rect(202, y, 108, 36, fill=FILL['green'], stroke=GREEN)
                s.text(256, y + 15, a, size=9, color=GREEN, weight='bold')
                s.text(256, y + 27, b, size=8.5, color=INK)
        s.save(f'convergence_step{step}')

def main():
    chain()
    recovery()
    delay_bar('delay_bar_scanner', scanner=True)
    crossover()
    tempe_bars()
    epoch()
    venn()
    tempe_track()
    convergence()
    print('ok')
if __name__ == '__main__':
    main()
