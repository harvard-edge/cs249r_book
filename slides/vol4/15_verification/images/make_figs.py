#!/usr/bin/env python3
"""Figures for Lecture 15 (Adversarial Verification), drawn at the size of their slots.

Every value comes from calcs.py (imported as c), which reads MLSysIM: the
running machine and its aisle record, the stopping budgets and oracles, the
coverage and rare-event scenarios, and Literature.* for the published
measurements (DriveFI, Wang et al., VNN-COMP 2025, O'Kelly et al.). Labels
print the strings calcs.py recorded, through deckvals.val; geometry uses the
full-precision values. The STL traces and the EVT curves are illustrative
shapes drawn for the slide (AV 15.9.1, 15.10.3). Plots use the kit's
slideplot helper settings (large type, direct labels). Colors follow the
Volume IV palette: plant slate, hazard red, latency amber, sensing teal,
deterministic green, proposal blue; nothing uses the accent.

Run from this folder: python3 make_figs.py   (then `make deck V=4 D=15_verification`)
"""
import contextlib
import io
import math
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import numpy as np
import slideplot as sp
from matplotlib.patches import FancyBboxPatch, Rectangle
from slideplot import plt
SLATE = '#334155'
SLATE_L = '#CBD5E1'
RED = '#DC2626'
RED_L = '#FEE2E2'
AMBER = '#D97706'
AMBER_L = '#FDE7C4'
TEAL = '#0D9488'
GREEN = '#059669'
GREEN_L = '#DCFCE7'
BLUE = '#2563EB'
BLUE_L = '#DBEAFE'
NAVY = '#1F407A'
GRAY = '#888888'
TEXT = '#1A1A2E'

def out(name):
    return str(HERE / f'{name}.svg')

def corner():
    mus = np.linspace(0.08, 0.4, 200)
    ceil = np.array([1.055977513, 1.064449837, 1.072779917, 1.080971255, 1.089027193, 1.096950926, 1.104745508, 1.112413862, 1.119958792, 1.127382983, 1.134689014, 1.141879362, 1.148956406, 1.155922437, 1.16277966, 1.169530196, 1.176176095, 1.182719328, 1.189161804, 1.195505362, 1.201751781, 1.207902781, 1.213960026, 1.219925128, 1.225799648, 1.231585097, 1.237282943, 1.242894607, 1.248421472, 1.253864878, 1.259226129, 1.264506489, 1.269707192, 1.274829435, 1.279874384, 1.284843175, 1.289736914, 1.294556678, 1.299303518, 1.303978459, 1.3085825, 1.313116618, 1.317581764, 1.32197887, 1.326308844, 1.330572574, 1.33477093, 1.338904759, 1.342974894, 1.346982148, 1.350927316, 1.354811179, 1.358634498, 1.362398024, 1.366102487, 1.369748608, 1.373337091, 1.376868626, 1.380343893, 1.383763556, 1.387128268, 1.390438671, 1.393695395, 1.396899058, 1.400050268, 1.403149622, 1.406197708, 1.409195102, 1.412142372, 1.415040077, 1.417888765, 1.420688978, 1.423441247, 1.426146095, 1.428804039, 1.431415585, 1.433981234, 1.436501478, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177, 1.436694177])
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(3.3, 2.2))
        ax.plot(mus, ceil, color=SLATE, lw=3)
        ax.axhline(1.3, color=BLUE, lw=2.2, ls='--')
        ax.text(0.395, 1.3 + 0.012, 'aisle speed 1.3 m/s', ha='right', va='bottom', color=BLUE, fontsize=10.5, fontweight='bold')
        ax.text(0.395, 1.436694177 + 0.012, 'dry-floor ceiling 1.44 m/s', ha='right', va='bottom', color=SLATE, fontsize=10)
        ax.axvline(0.2039432426, color=GRAY, lw=1.2, ls=':')
        ax.text(0.2039432426 + 0.006, 1.595, 'traction binds below 0.204', fontsize=9.5, color=GRAY, va='top')
        if step >= 2:
            bad = ceil < 1.3
            ax.fill_between(mus, ceil, 1.3, where=bad, color=RED_L, alpha=0.9)
            ax.plot([0.12], [1.230866736], 'o', color=RED, ms=9)
            ax.annotate('', xy=(0.12 + 0.006, 1.230866736), xytext=(0.212, 1.2), arrowprops=dict(arrowstyle='-', color=RED, lw=1.2))
            ax.text(0.215, 1.22, 'inspected floor 0.12:\nceiling 1.23 m/s', color=RED, fontsize=10.5, fontweight='bold', va='top')
        ax.set_xlim(0.08, 0.4)
        ax.set_ylim(1.0, 1.6)
        ax.set_xlabel('Floor friction coefficient')
        ax.set_ylabel('Speed ceiling (m/s)')
        sp.save(fig, out(f'corner_step{step}'))

def drivefi():
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    rows = [('Every fault,\nexhaustive', 14760.0, SLATE, '98,400 faults: 615 days'), ('Bayesian\nsearch', 4.0, RED, '561 faults in < 4 h; 460 hazards')]
    for i, (lab, v, col, txt) in enumerate(rows):
        y = 1 - i
        ax.barh(y, v, color=col, height=0.5)
        ax.text(-0.15, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=TEXT)
        ax.text(v * 1.3 if i == 1 else 1.2, y + (0 if i == 1 else 0.42), txt, va='center', fontsize=12, color=col if i == 1 else TEXT, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(1, 30000.0)
    ax.set_ylim(-0.6, 1.75)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Hours of simulated injection (log)')
    sp.save(fig, out('drivefi'))

def refevents():
    cap_blind = 0.17368 * 1000.0
    stop = 0.6053125 * 1000.0
    lease = 0.1066 * 1000.0
    rows = [('Capture-\nreferenced', [(cap_blind, AMBER, '173.7'), (stop, SLATE, 'stop 605.3'), (0.15 * 1000.0, SLATE_L, '150'), (0.04 * 1000.0, GREEN_L, '')], '969.0'), ('Renewal-\nreferenced (F1)', [(lease, AMBER, '106.6'), (stop, SLATE, 'stop 605.3'), (0.04 * 1000.0, GREEN_L, '')], '751.9'), ('Appearance-\nreferenced', None, 'no value yet')]
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(5.8, 2.25))
        for i, (lab, segs, total) in enumerate(rows):
            y = 2 - i
            al = 1.0 if i < step else 0.0
            ax.text(-25, y, lab, ha='right', va='center', fontsize=11.5, fontweight='bold', color=TEXT, alpha=al)
            if segs is None:
                ax.barh(y, 1000, color='white', height=0.55, edgecolor=GRAY, hatch='//', alpha=al, lw=1.2)
                ax.text(500, y, 'needs witnessed appearance-to-capture\nand dispatch-to-admission intervals', ha='center', va='center', fontsize=11, color=TEXT, alpha=al, bbox=dict(fc='white', ec='none', alpha=0.9 * al))
                continue
            x = 0
            for w, col, txt in segs:
                ax.barh(y, w, left=x, color=col, height=0.55, alpha=al, edgecolor='white')
                if txt and w > 130:
                    ax.text(x + w / 2, y, txt, ha='center', va='center', fontsize=11, color='white' if col in (SLATE, AMBER) else TEXT, alpha=al, fontweight='bold')
                x += w
            ax.text(1125, y, f'{total} mm', va='center', fontsize=12, fontweight='bold', color=BLUE if i == 1 else TEXT, alpha=al)
        if step >= 2:
            ax.annotate('', xy=(0.7519125 * 1000.0, 1.5), xytext=(0.9689925 * 1000.0, 1.5), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.8))
            ax.text(0.7519125 * 1000.0 - 12, 1.5, '217.1 mm', ha='right', va='center', color=RED, fontsize=12, fontweight='bold')
        ax.axvline(1.1 * 1000.0, color=RED, lw=2, ls='--')
        ax.text(1.1 * 1000.0 + 15, 2.42, 'clear 1,100 mm', ha='left', va='bottom', color=RED, fontsize=11.5, fontweight='bold')
        ax.set_xlim(0, 1330)
        ax.set_ylim(-0.45, 2.75)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Travel at 1.3 m/s from the reference event (mm)')
        sp.save(fig, out(f'refevents_step{step}'))

def speed_profile(t_on, t):
    """Speed after an S-curve stop starting at t_on (s): jerk-limited ramp, then constant a."""
    v0, a, j = (1.3, 2.0, 7.111111111)
    tr = a / j
    v = np.full_like(t, v0)
    s = t - t_on
    ramp = (s > 0) & (s <= tr)
    v[ramp] = v0 - 0.5 * j * s[ramp] ** 2
    v1 = v0 - 0.5 * j * tr ** 2
    lin = s > tr
    v[lin] = np.maximum(v1 - a * (s[lin] - tr), 0)
    return v

def f1trace():
    on = 0.082
    lease_ms, cmd_ms = (0.06 * 1000.0, (0.06 + 0.001) * 1000.0)
    frame_ms = (0.06 + 0.001 + 0.001) * 1000.0
    late_ms = 0.112 * 1000.0
    for step in (1, 2):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 2.75), gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.55))
        chans = [('lease', 0, lease_ms, TEAL), ('stop cmd', cmd_ms, 200, GREEN), ('drive frame', frame_ms, 200, GREEN), ('decel. onset', on * 1000.0, 200, GREEN)]
        for i, (lab, t1, t2, col) in enumerate(chans):
            y = 3 - i
            if lab == 'lease':
                ax1.plot([0, lease_ms, lease_ms, 130], [y + 0.35, y + 0.35, y, y], color=col, lw=2.5)
            else:
                ax1.plot([0, t1, t1, 130], [y, y, y + 0.35, y + 0.35], color=col, lw=2.5)
            ax1.text(-3, y + 0.15, lab, ha='right', va='center', fontsize=11, color=TEXT)
        ax1.axvspan(frame_ms, on * 1000.0, color=GREEN_L, alpha=0.8, lw=0)
        ax1.axvline(on * 1000.0, color=GREEN, lw=1.2, ls=':')
        ax1.text(on * 1000.0 + 2, 4.0, '82 ms bound', ha='left', va='bottom', color=GREEN, fontsize=11, fontweight='bold')
        if step >= 2:
            ax1.plot([0, late_ms, late_ms, 130], [-0.05, -0.05, 0.3, 0.3], color=RED, lw=2.5, ls='--')
            ax1.annotate('', xy=(on * 1000.0, -0.5), xytext=(late_ms, -0.5), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.6))
            ax1.text(late_ms + 2, -0.5, '-30 ms', ha='left', va='center', color=RED, fontsize=11, fontweight='bold')
        ax1.set_xlim(0, 130)
        ax1.set_ylim(-0.85, 4.5)
        ax1.set_yticks([])
        ax1.spines['left'].set_visible(False)
        ax1.set_xlabel('Time after the last valid renewal (ms)', fontsize=11)
        t = np.linspace(0, 1.2, 600)
        ax2.plot(t * 1000.0, speed_profile(on, t), color=GREEN, lw=2.5)
        ax2.text(1195, 1.28, 'on time: 711.9 mm, rest at 1,013 ms', fontsize=11, color=GREEN, fontweight='bold', ha='right', va='center')
        if step >= 2:
            ax2.plot(t * 1000.0, speed_profile(0.112, t), color=RED, lw=2.5, ls='--')
            ax2.text(1195, 1.0, 'late: 750.9 mm, margin +1 mm', fontsize=11, color=RED, fontweight='bold', ha='right', va='center')
        ax2.set_xlim(0, 1200)
        ax2.set_ylim(0, 1.45)
        ax2.set_ylabel('m/s', fontsize=11)
        ax2.set_xlabel('Time after the last valid renewal (ms)', fontsize=11)
        sp.save(fig, out(f'f1trace_step{step}'))

def wang():
    fig, ax = plt.subplots(figsize=(3.1, 2.25))
    x = np.arange(4)
    ben = [84.7, 88.4, 79.2, 53.7]
    pat = [0.0, 0.0, 0.0, 0.0]
    ax.bar(x - 0.2, ben, width=0.38, color=BLUE)
    ax.bar(x + 0.2, [max(p, 1.2) for p in pat], width=0.38, color=RED)
    for xi, b in zip(x, ben):
        ax.text(xi - 0.2, b + 2, f'{b:.0f}', ha='center', fontsize=9.5, color=BLUE, fontweight='bold')
        ax.text(xi + 0.2, 4, f'{pat[xi]:.0f}', ha='center', fontsize=9.5, color=RED, fontweight='bold')
    ax.text(3.55, 118, 'no patch', color=BLUE, fontsize=10.5, fontweight='bold', ha='right')
    ax.text(3.55, 106, 'patch in view', color=RED, fontsize=10.5, fontweight='bold', ha='right')
    ax.set_xticks(x, ['Spatial', 'Object', 'Goal', 'Long'])
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel('Task success (%)')
    ax.set_xlabel('LIBERO task suite (simulated)')
    sp.save(fig, out('wang_patch'))
LADDER_COLS = ['Decision\nlogic', 'CPU\ntiming', 'Bus\ntiming', 'Supply\nsag', 'Drives,\nbrakes', 'Floor\ncontact']
LADDER_ROWS = [('SIL', 'R---MM', 'sweeps'), ('PIL', 'RRM-MM', 'target-bound'), ('HIL', 'RRRRRM', 'real time'), ('Physical rig', 'RRRRRR', 'wear-limited')]

def ladder():
    cell_w, cell_h = (1.0, 0.62)
    for step in (1, 2, 3, 4):
        fig, ax = plt.subplots(figsize=(5.8, 2.55))
        for j, lab in enumerate(LADDER_COLS):
            ax.text(j * cell_w + 0.5, 4 * cell_h + 0.08, lab, ha='center', va='bottom', fontsize=10, color=TEXT, fontweight='bold')
        ax.text(6 * cell_w + 0.15, 4 * cell_h + 0.08, 'Throughput', ha='left', va='bottom', fontsize=10.5, color=TEXT, fontweight='bold')
        for i, (name, pattern, thr) in enumerate(LADDER_ROWS):
            y = (3 - i) * cell_h
            shown = i < step
            ax.text(-0.12, y + cell_h / 2, name, ha='right', va='center', fontsize=11, color=TEXT if shown else '#EEEEEE', fontweight='bold')
            for j, ch in enumerate(pattern):
                x = j * cell_w
                if not shown:
                    ax.add_patch(Rectangle((x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1, fc='white', ec='#EEEEEE', lw=1))
                    continue
                if ch == 'R':
                    ax.add_patch(Rectangle((x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1, fc=GREEN, ec=GREEN, lw=1.5))
                elif ch == 'M':
                    ax.add_patch(Rectangle((x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1, fc='white', ec=SLATE, lw=1.6, ls='--'))
                    ax.text(x + 0.5, y + cell_h / 2, 'model', ha='center', va='center', fontsize=10, color=SLATE)
                else:
                    ax.add_patch(Rectangle((x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1, fc='white', ec='#DDDDDD', lw=1))
                    ax.text(x + 0.5, y + cell_h / 2, 'absent', ha='center', va='center', fontsize=10, color=GRAY)
            if shown:
                ax.text(6 * cell_w + 0.15, y + cell_h / 2, thr, ha='left', va='center', fontsize=10.5, color=TEXT)
        ax.set_xlim(-1.2, 7.6)
        ax.set_ylim(-0.05, 4 * cell_h + 0.75)
        ax.axis('off')
        sp.save(fig, out(f'ladder_step{step}'))

def vnn():
    fig, ax = plt.subplots(figsize=(3.5, 2.15))
    vals = sorted([68000000.0, 37000000.0, 203000.0, 680000.0, 76000.0, 1800000.0, 275.0, 100000.0, 262000.0, 138000000.0, 1700000.0, 3800000.0, 3600000.0, 7400000.0, 37000000.0, 1700000.0, 625000.0, 1500000.0, 4000.0, 67000000.0, 13000.0, 855000.0, 4000.0, 1100000.0, 35000.0])
    rng = np.random.default_rng(3)
    ys = rng.uniform(-0.25, 0.25, len(vals))
    ax.scatter(vals, ys, s=60, color=SLATE, alpha=0.8, zorder=3)
    big = max(vals)
    ax.annotate('VGGNet16, 138M', xy=(big, ys[vals.index(big)]), xytext=(3000.0, 0.72), fontsize=9.5, color=SLATE, arrowprops=dict(arrowstyle='-', color=SLATE))
    ax.axvline(80000000.0, color=BLUE, lw=2, ls='--')
    ax.text(80000000.0 / 1.4, -0.75, 'chunk policy\n80M', color=BLUE, fontsize=9.5, fontweight='bold', ha='right', va='bottom')
    ax.axvline(7000000000.0, color=RED, lw=2.5)
    ax.text(7000000000.0 * 1.6, 0.55, 'intent\nmodel\n7B', color=RED, fontsize=9.5, fontweight='bold', ha='left')
    ax.set_xscale('log')
    ax.set_xlim(100, 1000000000000.0)
    ax.set_ylim(-0.85, 1.0)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Parameters (largest network per scored benchmark)')
    sp.save(fig, out('vnn'))

def stl():
    for step in (1, 2):
        fig, ax = plt.subplots(figsize=(3.2, 2.25))
        t = np.linspace(0, 2.0, 400)
        a = 0.15 + 0.85 * (t - 1.0) ** 2 / 1.0
        b = -0.04 + 0.95 * (t - 1.15) ** 2 / 1.0
        ax.axhline(0, color=RED, lw=2)
        ax.text(0.03, -0.02, 'clearance limit', color=RED, fontsize=9.5, ha='left', va='top')
        ax.plot(t, a, color=GREEN, lw=2.5)
        ax.annotate('', xy=(1.0, 0), xytext=(1.0, 0.15), arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.6))
        ax.text(0.0, 1.13, 'solid: $\\rho = +0.15$ m, passes', color=GREEN, fontsize=10.5, ha='left', va='center', fontweight='bold', transform=ax.transAxes)
        if step >= 2:
            ax.plot(t, b, color=RED, lw=2.5, ls='--')
            ax.annotate('', xy=(1.15, 0), xytext=(1.15, -0.04), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.6))
            ax.text(0.0, 1.04, 'dashed: $\\rho = -0.04$ m, fails', color=RED, fontsize=10.5, ha='left', va='center', fontweight='bold', transform=ax.transAxes)
        ax.set_xlim(0, 2.0)
        ax.set_ylim(-0.15, 0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Clearance margin (m)')
        sp.save(fig, out(f'stl_step{step}'))

def is_ladder():
    rows = [('Plain Monte Carlo', 56650700190.0, SLATE, '5.7' + '$\\times10^{' + '10' + '}$'), ('Zero-failure campaign', 1697103349, AMBER, '1.7' + '$\\times10^{' + '9' + '}$'), ('Importance sampling', 2708.899375, GREEN, '2,709')]
    for step in (1, 2, 3):
        fig, ax = plt.subplots(figsize=(4.8, 1.8))
        for i, (lab, v, col, txt) in enumerate(rows):
            y = 2 - i
            al = 1.0 if i < step else 0.0
            ax.barh(y, v, color=col, height=0.55, alpha=al)
            ax.text(v * 1.6, y, txt, va='center', fontsize=12, fontweight='bold', color=col, alpha=al)
            ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=11.5, color=TEXT, fontweight='bold', alpha=al)
        ax.set_xscale('log')
        ax.set_xlim(1, 5000000000000.0)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Runs for 10 percent relative error (log)')
        sp.save(fig, out(f'is_ladder_step{step}'))

def okelly():
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    gammas = sorted({0.14, 0.15, 0.19, 0.2})
    x = np.arange(len(gammas))
    small, big = (min((e['runs'] for e in ({'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.14, 'estimate': 1.24e-05, 'std_error': 3.1e-06}, {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.15, 'estimate': 8.06e-05, 'std_error': 7.91e-06}, {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.19, 'estimate': 0.00133, 'std_error': 3.2e-05}, {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.2, 'estimate': 0.00186, 'std_error': 3.79e-05}, {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.14, 'estimate': 1.98e-05, 'std_error': 8.88e-06}, {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.15, 'estimate': 6.61e-05, 'std_error': 1.5e-05}, {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.19, 'estimate': 0.00108, 'std_error': 9.51e-05}, {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.2, 'estimate': 0.00164, 'std_error': 0.00014}, {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.14, 'estimate': 2e-05, 'std_error': 1.41e-05}, {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.15, 'estimate': 0.0001, 'std_error': 3.16e-05}, {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.19, 'estimate': 0.00132, 'std_error': 0.000115}, {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.2, 'estimate': 0.00185, 'std_error': 0.000136}))), max((e['runs'] for e in ({'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.14, 'estimate': 1.24e-05, 'std_error': 3.1e-06}, {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.15, 'estimate': 8.06e-05, 'std_error': 7.91e-06}, {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.19, 'estimate': 0.00133, 'std_error': 3.2e-05}, {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.2, 'estimate': 0.00186, 'std_error': 3.79e-05}, {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.14, 'estimate': 1.98e-05, 'std_error': 8.88e-06}, {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.15, 'estimate': 6.61e-05, 'std_error': 1.5e-05}, {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.19, 'estimate': 0.00108, 'std_error': 9.51e-05}, {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.2, 'estimate': 0.00164, 'std_error': 0.00014}, {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.14, 'estimate': 2e-05, 'std_error': 1.41e-05}, {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.15, 'estimate': 0.0001, 'std_error': 3.16e-05}, {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.19, 'estimate': 0.00132, 'std_error': 0.000115}, {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.2, 'estimate': 0.00185, 'std_error': 0.000136}))))
    arms = [('Naive MC', small, SLATE_L), ('Cross-entropy', small, GREEN), ('Naive MC', big, SLATE)]
    for k, (method, runs, col) in enumerate(arms):
        rows = {0: {0.14: {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.14, 'estimate': 2e-05, 'std_error': 1.41e-05}, 0.15: {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.15, 'estimate': 0.0001, 'std_error': 3.16e-05}, 0.19: {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.19, 'estimate': 0.00132, 'std_error': 0.000115}, 0.2: {'method': 'Naive MC', 'runs': 100000.0, 'gamma': 0.2, 'estimate': 0.00185, 'std_error': 0.000136}}, 1: {0.14: {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.14, 'estimate': 1.98e-05, 'std_error': 8.88e-06}, 0.15: {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.15, 'estimate': 6.61e-05, 'std_error': 1.5e-05}, 0.19: {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.19, 'estimate': 0.00108, 'std_error': 9.51e-05}, 0.2: {'method': 'Cross-entropy', 'runs': 100000.0, 'gamma': 0.2, 'estimate': 0.00164, 'std_error': 0.00014}}, 2: {0.14: {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.14, 'estimate': 1.24e-05, 'std_error': 3.1e-06}, 0.15: {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.15, 'estimate': 8.06e-05, 'std_error': 7.91e-06}, 0.19: {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.19, 'estimate': 0.00133, 'std_error': 3.2e-05}, 0.2: {'method': 'Naive MC', 'runs': 1300000.0, 'gamma': 0.2, 'estimate': 0.00186, 'std_error': 3.79e-05}}}[k]
        vals = [100 * rows[g]['std_error'] / rows[g]['estimate'] for g in gammas]
        ax.bar(x + (k - 1) * 0.27, vals, width=0.26, color=col, edgecolor=SLATE, lw=0.6)
    k_small = f'{small / 10 ** 3:.0f}K'
    ax.text(3.45, 66, f'naive, {k_small} runs', color=SLATE, fontsize=9.5, ha='right')
    ax.text(3.45, 56, f'cross-entropy, {k_small}', color=GREEN, fontsize=9.5, ha='right', fontweight='bold')
    ax.text(3.45, 46, 'naive, 1.3M runs', color=SLATE, fontsize=9.5, ha='right', fontweight='bold')
    ax.set_xticks(x, [f'{g:.2f}' for g in gammas])
    ax.set_xlabel('Danger threshold $\\gamma$ (rarer to the left)')
    ax.set_ylabel('Relative std. error (%)')
    ax.set_ylim(0, 75)
    sp.save(fig, out('okelly'))

def evt():
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    y = np.linspace(0, 60, 300)
    sig = 10.0
    for xi, col, lab in [(0.3, RED, '$\\xi>0$: heavy tail'), (0.0, SLATE, '$\\xi=0$: exponential'), (-0.3, GREEN, '$\\xi<0$: endpoint')]:
        if xi == 0:
            s = np.exp(-y / sig)
        else:
            base = 1 + xi * y / sig
            s = np.where(base > 0, np.abs(base) ** (-1 / xi), 0.0)
        ax.semilogy(y, np.maximum(s, 1e-06), color=col, lw=2.5)
    ax.text(42, 0.15, '$\\xi>0$: heavy tail', color=RED, fontsize=10.0, fontweight='bold', va='bottom')
    ax.text(37, 0.002, '$\\xi=0$: exponential', color=SLATE, fontsize=10.0, fontweight='bold', va='top')
    end = -sig / -0.3
    ax.axvline(end, color=GREEN, ls=':', lw=1.5)
    ax.text(end + 1, 1e-05, 'fitted endpoint', color=GREEN, fontsize=10.0, fontweight='bold', ha='left', va='bottom')
    ax.set_ylim(1e-06, 1.5)
    ax.set_xlim(0, 60)
    ax.set_xlabel('Overshoot past the threshold (mm, illustrative)')
    ax.set_ylabel('P(exceed)')
    sp.save(fig, out('evt'))
SMALL = {'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5}

def main():
    with plt.rc_context(SMALL):
        corner()
        wang()
        vnn()
        stl()
        okelly()
        evt()
    drivefi()
    refevents()
    f1trace()
    ladder()
    is_ladder()
if __name__ == '__main__':
    main()
