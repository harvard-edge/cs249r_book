#!/usr/bin/env python3
"""Data plots and diagrams for Volume IV Lecture 6 (Policy Training).

Every plotted value and every number in a label comes from calcs.py, which
reads it from MLSysIM (the Scenarios.* entries and Literature.RobotLearning);
coordinates and schematic shapes are layout. Illustrative figures say so on
the figure.

  - Dactyl hook and ablation: Andrychowicz et al. 2020, Tables 3, 4, 5 and
    section 6.3 (simulated hours), arXiv 1808.00177.
  - Mode averaging, drift regions: chapter section 6.2.3, 6.2.4 (schematic).
  - Decoder slack: chapter Napkin Math 6.1.
  - Discount weight: chapter section 6.2.1.
  - Offline value overestimation: chapter section 6.2.7 (schematic).
  - Coverage cells and declared ODD: chapter section 6.2.8, 6.8.
  - Reset wall: chapter Napkin Math 6.2, section 6.3.1, section 6.4.1.
  - Soft spring: chapter Napkin Math 6.3, equations 7 and 8.
  - Reality gap vector: chapter section 6.5 (illustrative linear drive).

Steps of one figure draw every element (hidden ones transparent), so the
bounding box and axes never move between steps.

Run from this folder: python3 make_figs.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
plt = sp.plt
C = sp.C
out = Path(__file__).parent
SUPERSCRIPT = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
PROP = '#2563EB'
NERV = '#059669'
VETO = '#DC2626'
SLATE = '#334155'

def a(shown):
    return 1.0 if shown else 0.0

def dactyl_hook(k):
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    sim_median = 50.0
    ax.barh(1, sim_median, height=0.5, color=C['neutral_l'], edgecolor=C['neutral'])
    ax.text(sim_median + 1, 1, 'median 50', va='center', fontsize=13, fontweight='bold', color=C['neutral'])
    trials = [50.0, 41.0, 29.0, 27.0, 14.0, 12.0, 6.0, 4.0, 4.0, 1.0]
    ys = 0 + np.linspace(-0.18, 0.18, len(trials))
    ax.scatter(trials, ys, s=70, color=C['memory'], alpha=a(k >= 2), zorder=3)
    med = 13.0
    ax.plot([med, med], [-0.32, 0.32], color=C['danger'], lw=3, alpha=a(k >= 2))
    ax.text(med + 1, -0.42, 'median 13', color=C['danger'], fontsize=13, fontweight='bold', va='top', alpha=a(k >= 2))
    ax.text(25, 0, '?', fontsize=26, fontweight='bold', color=C['neutral'], ha='center', va='center', alpha=a(k == 1))
    ax.set_yticks([1, 0], ['Simulation\n(100 trials)', 'Physical hand\n(10 trials)'], fontsize=12)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.75, 1.45)
    ax.set_xlabel('Consecutive rotations before a drop or timeout (cap 50)')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'dactyl_hook_step{k}.svg')

def dotrows(path, rows, medians, size, highlight=(), names=None):
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, (name, trials) in enumerate(rows.items()):
        y = n - 1 - i
        col = C['memory'] if name in highlight else C['neutral']
        ys = y + np.linspace(-0.16, 0.16, len(trials))
        ax.scatter(trials, ys, s=55, color=col, zorder=3)
        m = medians[name]
        ax.plot([m, m], [y - 0.3, y + 0.3], color=C['danger'], lw=2.5, zorder=4)
        ax.text(53, y, f'{m:g}', va='center', fontsize=13, fontweight='bold', color=C['danger'])
    ax.text(53, n - 0.35, 'median', fontsize=11, color=C['danger'], fontweight='bold')
    ax.set_yticks(range(n - 1, -1, -1), [(names or {}).get(k, k) for k in rows], fontsize=12)
    ax.set_xlim(-1, 58)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_xlabel('Rotations on the physical hand')
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)

def mode_avg(k):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.add_patch(plt.Circle((0, 3), 0.85, color=SLATE))
    ax.text(0, 3, 'column', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    t = np.linspace(0, 1, 100)
    for sgn, lab in ((-1, 'demo A: left'), (1, 'demo B: right')):
        x = sgn * 1.75 * np.sin(np.pi * t)
        y = 6 * t
        ax.plot(x, y, color=PROP, lw=3, alpha=a(k >= 2))
        ax.text(sgn * 1.95, 3, lab, color=PROP, fontsize=14, fontweight='bold', ha='right' if sgn < 0 else 'left', va='center', alpha=a(k >= 2))
    ax.plot([0, 0], [0, 2.05], color=VETO, lw=3.5, alpha=a(k >= 3))
    ax.annotate('', xy=(0, 2.12), xytext=(0, 1.7), alpha=a(k >= 3), arrowprops=dict(arrowstyle='-|>', color=VETO, lw=3, alpha=a(k >= 3)))
    ax.annotate('the mean:\nno demo did this', xy=(0.08, 1.1), xytext=(2.0, 0.7), color=VETO, fontsize=13, fontweight='bold', va='center', alpha=a(k >= 3), linespacing=1.0, arrowprops=dict(arrowstyle='-', color=VETO, lw=1.5, alpha=a(k >= 3)))
    ax.plot(0, 0, 's', ms=16, color=SLATE)
    ax.text(0, -0.55, 'robot', ha='center', va='top', fontsize=13, color=SLATE, fontweight='bold')
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-1.1, 6.3)
    ax.set_aspect('equal')
    ax.axis('off')
    sp.save(fig, out / f'mode_avg_step{k}.svg')

def drift():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.4, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    x = np.linspace(0, 10, 200)
    ax.fill_between(x, -0.6, 0.6, color=C['data_l'])
    ax.fill_between(x, 0.6, 1.8, color=C['compute_l'])
    ax.fill_between(x, 1.8, 3.0, color=C['danger_l'])
    ax.plot(x, 0 * x, color=C['data'], lw=2.5)
    learner = np.where(x < 3, 0, 0.035 * np.clip(x - 3, 0, None) ** 2.1)
    ax.plot(x, np.minimum(learner, 2.9), color=SLATE, lw=2.5, ls='--')
    ax.text(0.2, -0.35, 'demonstrated', color=C['data'], fontsize=12, fontweight='bold')
    ax.text(0.2, 1.05, 'recoverable:\nlabels help', color=C['compute'], fontsize=11, fontweight='bold', va='center', linespacing=1.0)
    ax.text(0.2, 2.4, 'no restoring action', color=C['danger'], fontsize=11, fontweight='bold', va='center')
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 3.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Task progress')
    ax.set_ylabel('Deviation')
    ax.set_title('Where labels can help', fontsize=13)
    tt = np.linspace(0, 1, 100)
    bx.plot(tt, tt, color=C['memory'], lw=3)
    bx.plot(tt, tt ** 2 * 2.0, color=C['danger'], lw=3)
    bx.text(0.5, 0.08, 'velocity error:\ngrows as t', color=C['memory'], fontsize=11, fontweight='bold', linespacing=1.0)
    bx.text(0.05, 1.3, 'torque error:\ngrows as t²', color=C['danger'], fontsize=11, fontweight='bold', linespacing=1.0)
    bx.set_xticks([])
    bx.set_yticks([])
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 2.0)
    bx.set_xlabel('Time')
    bx.set_ylabel('Position drift')
    bx.set_title('Same persistent error', fontsize=13)
    fig.text(0.99, 0.0, 'Schematic', ha='right', fontsize=9, color=C['neutral'])
    fig.tight_layout(w_pad=1.5)
    sp.save(fig, out / 'drift.svg')

def decoder(k):
    renew = 50.0
    rows = [('ACT, 1 pass', 40.0, C['memory'], 1), ('Flow, 4 steps', 18.0, C['data'], 2), ('DDIM, 16 steps', 48.0, C['compute'], 2), ('DDIM + 5 ms jitter', 48.0 + 5.0, C['compute'], 3)]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    n = len(rows)
    for i, (lab, v, col, s) in enumerate(rows):
        y = n - 1 - i
        sh = k >= s
        ax.barh(y, min(v, renew), color=col, height=0.6, alpha=a(sh))
        if v > renew:
            ax.barh(y, v - renew, left=renew, color=VETO, height=0.6, alpha=a(sh))
        ax.text((renew if 0 < renew - v < 4 else v) + 0.8, y, f'{v:g} ms', va='center', fontsize=13, bbox=dict(facecolor='white', edgecolor='none', pad=0.5), fontweight='bold', color=VETO if v > renew else col, alpha=a(sh))
        ax.text(-1, y, lab, ha='right', va='center', fontsize=12.5, fontweight='bold', color=C['text'], alpha=a(sh))
    for xv, lab, col in ((25.0, 'setpoint step 25 ms', VETO), (renew, 'renewal 50 ms', C['neutral'])):
        ax.axvline(xv, color=col, lw=2, ls='--')
        ax.text(xv, n - 0.25, lab, color=col, fontsize=11.5, fontweight='bold', ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.6, n + 0.15)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Proposal latency, ms (illustrative heads)')
    sp.save(fig, out / f'decoder_step{k}.svg')

def discount():
    steps, gmin = (320, 0.9978362593)
    k = np.arange(0, int(steps) + 1)
    tsec = k * 0.025
    rollout, horizon = (8.0, 2.5)
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(tsec, 0.99 ** (steps - k), color=C['danger'], lw=3)
    ax.plot(tsec, gmin ** (steps - k), color=C['memory'], lw=3)
    ax.text(0.15, 0.2, 'γ = 0.99: weight 0.04', color=C['danger'], fontsize=12.5, fontweight='bold')
    ax.text(0.15, 0.8, 'γ = 0.9978: weight 0.50', color=C['memory'], fontsize=12.5, fontweight='bold')
    ax.axvspan(rollout - horizon, rollout, color=C['danger_l'], alpha=0.6, zorder=0)
    ax.text(rollout - horizon / 2, 1.05, 'last 2.5 s\n(100 steps)', ha='center', fontsize=11, color=C['danger'], fontweight='bold', linespacing=1.0)
    ax.set_xlim(0, rollout)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel('Time of the decision in an 8 s opening (s)')
    ax.set_ylabel('Weight of the\ndoor-open reward', fontsize=11)
    ax.grid(color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / 'discount.svg')

def offline(k):
    x = np.linspace(0, 10, 400)
    true_q = 2.0 * np.exp(-((x - 3.0) / 1.6) ** 2) - 0.05 * x
    fit = true_q + 2.6 * np.exp(-((x - 7.6) / 0.6) ** 2)
    logged = np.array([1.6, 2.1, 2.6, 2.9, 3.3, 3.8, 4.2])
    cql = fit - 1.2 * (1 - np.exp(-np.min(np.abs(x[:, None] - logged[None, :]), axis=1) ** 2 / 0.8))
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.plot(x, true_q, color=C['neutral'], lw=2, ls=':')
    ax.text(0.3, 2.15, 'true value', color=C['neutral'], fontsize=11, fontweight='bold')
    ax.plot(x, fit, color=C['danger'], lw=3, alpha=a(k >= 1))
    i = np.argmax(fit)
    ax.plot(x[i], fit[i], 'o', ms=11, color=C['danger'], alpha=a(k >= 1))
    ax.text(x[i] - 0.3, fit[i] + 0.05, 'greedy pick:\nnever tried', color=C['danger'], fontsize=12, fontweight='bold', ha='right', va='center', alpha=a(k >= 1), linespacing=1.0)
    ax.plot(x, cql, color=C['memory'], lw=3, alpha=a(k >= 2))
    j = np.argmax(cql)
    ax.plot(x[j], cql[j], 'o', ms=11, color=C['memory'], alpha=a(k >= 2))
    ax.text(1.0, -2.5, 'conservative fit: lowered away from logged actions', color=C['memory'], fontsize=12, fontweight='bold', alpha=a(k >= 2), linespacing=1.0)
    for lx in logged:
        ax.plot([lx, lx], [-1.9, -1.6], color=C['data'], lw=3)
    ax.text(4.5, -1.75, 'logged actions', color=C['data'], fontsize=11.5, fontweight='bold', va='center', ha='left')
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.6, 3.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Action (for example, approach speed)')
    ax.set_ylabel('Estimated value Q')
    fig.text(0.99, 0.0, 'Schematic', ha='right', fontsize=9, color=C['neutral'])
    sp.save(fig, out / f'offline_step{k}.svg')
PLATES = ['nominal', 'displaced +', 'displaced −']
SPEEDS = ['guarded\n0.03 m/s', 'fast\n0.10 m/s']

def grid(ax, fills, labels, title=None, fs=13):
    for r in range(3):
        for c in range(2):
            fc, ec = fills[r][c]
            ax.add_patch(plt.Rectangle((c, 2 - r), 0.94, 0.9, facecolor=fc, edgecolor=ec, lw=2))
            ax.text(c + 0.47, 2 - r + 0.45, labels[r][c], ha='center', va='center', fontsize=fs, fontweight='bold', color=ec, linespacing=1.0)
    ax.set_xlim(-0.05, 2.0)
    ax.set_ylim(-0.05, 3.05)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', color=C['text'])
GREEN = (C['data_l'], C['data'])
GRAY = ('#F2F2F2', C['neutral'])
RED = (C['danger_l'], C['danger'])
HELD = (C['compute_l'], C['compute'])
ASK = ('white', C['memory'])

def cells_support():
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    fills = [[GREEN, GRAY], [GRAY, GRAY], [GRAY, GRAY]]
    labels = [['supported', 'absent'], ['absent', 'absent'], ['absent', 'absent']]
    grid(ax, fills, labels, fs=11.5)
    for c, s in enumerate(SPEEDS):
        ax.text(c + 0.47, 3.0, s, ha='center', va='bottom', fontsize=12, color=C['text'], linespacing=1.0)
    for r, p in enumerate(PLATES):
        ax.text(-0.08, 2 - r + 0.45, p, ha='right', va='center', fontsize=12, color=C['text'])
    ax.set_xlim(-1.25, 2.0)
    ax.set_ylim(-0.1, 3.55)
    sp.save(fig, out / 'cells_support.svg')

def odd(k):
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 3.0), gridspec_kw={'width_ratios': [1.6, 1, 1]})
    fast = RED
    specs = [('latch-bc-01', [[GREEN, fast], [GRAY, fast], [GRAY, fast]], [['declared', 'out'], ['absent', 'out'], ['absent', 'out']], 1), ('latch-res-03', [[HELD, fast], [GRAY, fast], [GRAY, fast]], [['held', 'out'], ['absent', 'out'], ['absent', 'out']], 2), ('bc-recovery-02', [[GREEN, fast], [ASK, fast], [GRAY, fast]], [['declared', 'out'], ['requested', 'out'], ['absent', 'out']], 3)]
    for i, (ax, (title, fills, labels, s)) in enumerate(zip(axs, specs)):
        grid(ax, fills, labels, None, fs=10.5)
        for c, sp_ in enumerate(('guarded', 'fast')):
            ax.text(c + 0.47, 3.0, sp_, ha='center', va='bottom', fontsize=11, color=C['text'])
        ax.text(0.97, 3.42, title, ha='center', va='bottom', fontsize=12, fontweight='bold', color=C['text'])
        if i == 0:
            for r, pl in enumerate(PLATES):
                ax.text(-0.08, 2 - r + 0.45, pl, ha='right', va='center', fontsize=11, color=C['text'])
            ax.set_xlim(-1.25, 2.0)
        ax.set_ylim(-0.05, 3.85)
        for art in ax.patches + ax.texts:
            if i > 0 or not hasattr(art, 'get_text') or art.get_position()[0] >= 0:
                art.set_alpha(a(k >= s))
    fig.tight_layout(w_pad=0.4)
    sp.save(fig, out / f'odd_step{k}.svg')

def reset_wall():
    rows = [('Manual resets', 439.8, 'danger', '439.8 days'), ('Automated resets', 138.9, 'compute', '138.9 days'), ('Execution only', 92.6, 'neutral', '92.6 days'), ('14 rigs, manual', 31.4, 'network', '31.4 days'), ('HIL-SERL budget', 0.1041666667, 'data', '2.5 h, 750 trials'), ('Simulation', 0.0301408179, 'memory', '43.4 min')]
    for k, n in ((1, 3), (2, 4), (3, 6)):
        fig, ax = plt.subplots(figsize=(5.8, 2.4))
        for i, (lab, v, ck, txt) in enumerate(rows):
            y = len(rows) - 1 - i
            sh = i < n
            ax.barh(y, v, color=C[ck], height=0.62, alpha=a(sh))
            ax.text(0.0075, y, lab, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
            ax.text(v * 1.2, y, txt, va='center', fontsize=13, fontweight='bold', color=C[ck], alpha=a(sh))
        ax.set_xscale('log')
        ax.set_xlim(0.01, 10000.0)
        ticks = [0.01, 0.1, 1, 10, 100, 1000]
        ax.set_xticks(ticks, ['0.01', '0.1', '1', '10', '100', '1,000'], fontsize=12)
        ax.minorticks_off()
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis='x', color=C['grid'], lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('Wall-clock time, days (log scale)', fontsize=12)
        sp.save(fig, out / f'reset_wall_step{k}.svg')

def spring(k):
    f_lim = 100.0
    kr = np.logspace(3, 8, 400)
    f = np.array([9.474996628, 9.612334515, 9.75165269, 9.892979249, 10.03634266, 10.18177179, 10.32929588, 10.47894456, 10.63074786, 10.7847362, 10.94094043, 11.09939178, 11.2601219, 11.42316286, 11.58854715, 11.75630768, 11.92647779, 12.09909125, 12.27418228, 12.45178551, 12.63193606, 12.81466945, 13.00002169, 13.18802922, 13.37872896, 13.57215827, 13.76835499, 13.96735745, 14.16920441, 14.37393515, 14.5815894, 14.79220739, 15.00582982, 15.22249791, 15.44225333, 15.66513828, 15.89119543, 16.12046798, 16.3529996, 16.58883448, 16.82801731, 17.07059329, 17.31660812, 17.56610801, 17.81913969, 18.07575038, 18.33598783, 18.59990029, 18.86753653, 19.1389458, 19.41417788, 19.69328307, 19.97631215, 20.2633164, 20.55434763, 20.84945811, 21.14870063, 21.45212846, 21.75979536, 22.07175555, 22.38806375, 22.70877513, 23.03394535, 23.36363048, 23.69788708, 24.03677211, 24.380343, 24.72865756, 25.08177403, 25.43975103, 25.80264758, 26.17052306, 26.54343721, 26.9214501, 27.30462214, 27.69301404, 28.0866868, 28.48570168, 28.89012019, 29.30000409, 29.71541531, 30.13641599, 30.5630684, 30.99543495, 31.43357816, 31.87756059, 32.32744487, 32.78329361, 33.2451694, 33.71313478, 34.18725217, 34.66758385, 35.15419193, 35.64713827, 36.14648448, 36.65229185, 37.16462129, 37.68353331, 38.20908794, 38.74134468, 39.28036246, 39.82619956, 40.37891356, 40.93856127, 41.50519866, 42.0788808, 42.65966181, 43.24759472, 43.84273146, 44.44512277, 45.05481808, 45.67186547, 46.29631157, 46.92820145, 47.56757856, 48.21448461, 48.86895952, 49.53104125, 50.20076575, 50.87816685, 51.56327613, 52.25612283, 52.95673375, 53.6651331, 54.38134242, 55.10538046, 55.83726301, 56.57700285, 57.32460959, 58.08008953, 58.84344556, 59.61467704, 60.39377961, 61.18074515, 61.97556157, 62.77821273, 63.58867827, 64.40693352, 65.23294932, 66.06669192, 66.90812285, 67.75719878, 68.61387139, 69.47808726, 70.34978771, 71.22890874, 72.11538084, 73.00912892, 73.91007221, 74.81812411, 75.73319211, 76.6551777, 77.58397627, 78.51947701, 79.46156288, 80.41011049, 81.36499003, 82.32606528, 83.29319349, 84.26622539, 85.24500514, 86.22937031, 87.21915191, 88.21417433, 89.21425544, 90.21920653, 91.22883242, 92.24293147, 93.26129569, 94.28371078, 95.30995625, 96.33980551, 97.37302602, 98.40937943, 99.44862168, 100.4905032, 101.5347693, 102.5811598, 103.6294099, 104.67925, 105.7304062, 106.7826003, 107.8355503, 108.8889703, 109.9425715, 110.9960618, 112.0491465, 113.1015285, 114.1529088, 115.2029867, 116.2514601, 117.2980261, 118.3423813, 119.384222, 120.423245, 121.4591474, 122.4916276, 123.5203853, 124.5451222, 125.565542, 126.5813512, 127.5922593, 128.5979792, 129.5982277, 130.5927258, 131.5811989, 132.5633776, 133.5389975, 134.5078, 135.4695325, 136.4239487, 137.3708086, 138.3098795, 139.2409356, 140.1637586, 141.0781378, 141.9838703, 142.8807615, 143.768625, 144.6472827, 145.5165651, 146.3763116, 147.2263704, 148.0665983, 148.8968614, 149.7170347, 150.5270024, 151.3266576, 152.1159025, 152.8946484, 153.6628156, 154.4203332, 155.1671395, 155.9031812, 156.6284138, 157.3428013, 158.0463163, 158.7389394, 159.4206593, 160.0914728, 160.7513843, 161.4004057, 162.0385564, 162.6658626, 163.2823577, 163.8880818, 164.483081, 165.0674082, 165.6411219, 166.2042863, 166.7569713, 167.2992519, 167.831208, 168.3529244, 168.8644904, 169.3659993, 169.8575487, 170.3392397, 170.8111771, 171.2734688, 171.726226, 172.1695625, 172.6035947, 173.0284415, 173.4442239, 173.8510647, 174.2490886, 174.638422, 175.0191922, 175.3915281, 175.7555593, 176.1114165, 176.4592307, 176.7991337, 177.1312574, 177.455734, 177.7726958, 178.0822748, 178.384603, 178.679812, 178.9680328, 179.2493961, 179.5240316, 179.7920686, 180.0536352, 180.3088588, 180.5578656, 180.8007808, 181.0377285, 181.2688314, 181.494211, 181.7139874, 181.9282794, 182.1372042, 182.3408777, 182.5394141, 182.7329261, 182.9215249, 183.10532, 183.2844191, 183.4589286, 183.628953, 183.7945951, 183.955956, 184.1131352, 184.2662305, 184.4153377, 184.5605512, 184.7019636, 184.8396657, 184.9737467, 185.1042939, 185.2313932, 185.3551286, 185.4755826, 185.5928357, 185.7069672, 185.8180544, 185.9261732, 186.0313978, 186.133801, 186.2334536, 186.3304255, 186.4247844, 186.516597, 186.6059283, 186.6928419, 186.7773999, 186.859663, 186.9396906, 187.0175405, 187.0932695, 187.1669328, 187.2385843, 187.3082768, 187.3760616, 187.4419891, 187.5061081, 187.5684665, 187.6291109, 187.6880868, 187.7454386, 187.8012096, 187.855442, 187.9081769, 187.9594545, 188.0093139, 188.0577931, 188.1049295, 188.1507592, 188.1953174, 188.2386387, 188.2807565, 188.3217034, 188.3615114, 188.4002113, 188.4378333, 188.4744069, 188.5099606, 188.5445224, 188.5781195, 188.6107781, 188.6425241, 188.6733826, 188.7033779, 188.7325337, 188.7608733, 188.788419, 188.8151929, 188.8412161, 188.8665095, 188.8910932, 188.914987, 188.9382098, 188.9607803, 188.9827166, 189.0040363, 189.0247566, 189.044894, 189.0644649, 189.0834849, 189.1019695, 189.1199335, 189.1373915, 189.1543576, 189.1708455, 189.1868686, 189.2024398, 189.2175719, 189.232277, 189.2465672, 189.260454, 189.2739488, 189.2870626, 189.299806, 189.3121893, 189.3242228, 189.3359163, 189.3472792, 189.3583209])
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    ax.axhspan(f_lim, 220, color=C['danger_l'], alpha=0.5, zorder=0)
    ax.axhline(f_lim, color=VETO, lw=2, ls='--')
    ax.text(1200.0, f_lim + 6, 'latch limit 100 N', color=VETO, fontsize=12, fontweight='bold')
    ax.plot([15625.0], [37.5], 'o', ms=12, color=C['memory'])
    ax.text(1200.0, 55, "simulator's spring:\n37.5 N", color=C['memory'], fontsize=12, fontweight='bold', linespacing=1.0)
    f_rigid = 189.7366596
    ax.axhline(f_rigid, color=SLATE, lw=2, ls=':', alpha=a(k >= 2))
    ax.text(1200.0, f_rigid + 6, 'rigid arm on the real plate: 189.7 N', color=SLATE, fontsize=12, fontweight='bold', alpha=a(k >= 2))
    ax.plot(kr, f, color=C['compute'], lw=3, alpha=a(k >= 3))
    ax.plot([153846.1538], [f_lim], 'o', ms=11, color=VETO, alpha=a(k >= 3))
    k_star = '1.54×10⁵'
    ax.text(2200000.0, 50, f'arm stiffer than\n{k_star} N/m\nexceeds the limit', color=VETO, fontsize=12, fontweight='bold', alpha=a(k >= 3), linespacing=1.0)
    ax.set_xscale('log')
    ax.set_xlim(1000.0, 100000000.0)
    ax.set_xticks([1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0], ['10³', '10⁴', '10⁵', '10⁶', '10⁷', '10⁸'], fontsize=12)
    ax.minorticks_off()
    ax.set_ylim(0, 220)
    ax.set_xlabel('Robot-side stiffness, N/m (log scale)')
    ax.set_ylabel('Peak force at 0.10 m/s, N', fontsize=11)
    ax.grid(color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'spring_step{k}.svg')

def gap_vector(k):
    fig, axs = plt.subplots(2, 2, figsize=(6.0, 3.1))
    sim_kw = dict(color=C['neutral'], lw=2.2, ls='--')
    hw_kw = dict(color=C['memory'], lw=2.6)
    show = [k >= 1, k >= 2, k >= 3, k >= 4]
    ax = axs[0, 0]
    t = np.linspace(0, 60, 300)
    f_target = 10.0
    ax.plot(t, np.clip(t * 0.5, 0, f_target), **sim_kw)
    ax.plot(t, np.clip(t * 0.25, 0, f_target), **hw_kw)
    ax.text(20.0 + 1, 2.6, '20 ms', color=C['neutral'], fontsize=12, fontweight='bold')
    ax.text(40.0 + 1, 5.0, '40 ms', color=C['memory'], fontsize=12, fontweight='bold')
    ax.set_title('Dynamics: time to 10 N', fontsize=13)
    ax.set_xlabel('ms', fontsize=12, labelpad=0)
    ax = axs[0, 1]
    t = np.linspace(0, 120, 300)
    mm_per_ms = 0.2
    ax.plot(t, mm_per_ms * t, **sim_kw)
    ax.plot(t, mm_per_ms * np.clip(t - 49.5, 0, None), **hw_kw)
    ax.text(3, 26, 'p99 delay 49.5 ms:\n9.9 mm unobserved', color=C['memory'], fontsize=12, fontweight='bold', linespacing=1.0)
    ax.set_ylim(0, 40)
    ax.set_title('Timing: command to motion', fontsize=13)
    ax.set_xlabel('ms', fontsize=12, labelpad=0)
    ax = axs[1, 0]
    t = np.linspace(0, 10, 200)
    ax.plot(t, 0 * t, **sim_kw)
    y = 3.2 + 0 * t
    y[(t > 5) & (t < 6.6)] = np.nan
    ax.plot(t, y, **hw_kw)
    ax.axvspan(5, 6.6, color=C['neutral_l'])
    ax.text(0.3, 1.0, 'bias +3.2 mm', color=C['memory'], fontsize=12, fontweight='bold')
    ax.text(5.8, 4.3, 'dropout', ha='center', fontsize=11.5, color=C['neutral'], fontweight='bold', linespacing=1.0)
    ax.set_ylim(-1, 6)
    ax.set_title('Sensing: range error', fontsize=13)
    ax = axs[1, 1]
    tol, shift = (2.0, 6.0)
    ax.axvspan(-tol, tol, color=C['data_l'])
    ax.plot([0], [0.5], 'o', ms=10, color=C['neutral'])
    ax.plot([shift], [0.5], 'o', ms=10, color=VETO)
    ax.text(0, 0.8, 'sim', ha='center', fontsize=12, color=C['neutral'], fontweight='bold')
    ax.text(shift, 0.8, 'real: 6 mm', ha='center', fontsize=12, color=VETO, fontweight='bold')
    ax.text(0, 0.15, '±2 mm tolerance', ha='center', fontsize=12, color=C['data'], fontweight='bold')
    ax.set_xlim(-4, 9)
    ax.set_ylim(0, 1.1)
    ax.set_title('Appearance: contact patch', fontsize=13)
    for i, ax in enumerate(axs.flat):
        ax.set_yticks([])
        if i >= 2:
            ax.set_xticks([])
        else:
            ax.tick_params(labelsize=12)
        for art in ax.lines + ax.texts + ax.patches + ax.collections:
            art.set_alpha(a(show[i]) * (art.get_alpha() if art.get_alpha() is not None else 1))
    fig.text(0.01, 0.0, 'dashed: simulator    solid: hardware    illustrative values', fontsize=12, color=C['neutral'])
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    sp.save(fig, out / f'gap_vector_step{k}.svg')
if __name__ == '__main__':
    for k in (1, 2):
        dactyl_hook(k)
    ablation = {'All randomizations': [50.0, 41.0, 29.0, 27.0, 14.0, 12.0, 6.0, 4.0, 4.0, 1.0], 'No observation noise': [45.0, 35.0, 23.0, 11.0, 9.0, 8.0, 7.0, 6.0, 6.0, 1.0], 'No physics randomizations': [7.0, 7.0, 7.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0], 'No unmodeled effects': [16.0, 7.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0], 'No randomizations': [6.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    memory = {'LSTM policy': [50.0, 41.0, 29.0, 27.0, 14.0, 12.0, 6.0, 4.0, 4.0, 1.0], 'FF policy, LSTM value': [15.0, 7.0, 6.0, 5.0, 4.0, 3.0, 3.0, 2.0, 2.0, 0.0], 'FF policy, FF value': [15.0, 8.0, 6.0, 5.0, 3.0, 3.0, 2.0, 2.0, 2.0, 0.0]}
    dotrows(out / 'dactyl_ablation.svg', ablation, {'All randomizations': 13.0, 'No observation noise': 8.5, 'No physics randomizations': 2.0, 'No unmodeled effects': 2.0, 'No randomizations': 0.0}, (4.0, 2.7), highlight=('All randomizations',), names={'All randomizations': 'All', 'No observation noise': 'No obs. noise', 'No physics randomizations': 'No physics', 'No unmodeled effects': 'No added effects', 'No randomizations': 'None'})
    dotrows(out / 'dactyl_memory.svg', memory, {'LSTM policy': 13.0, 'FF policy, LSTM value': 3.5, 'FF policy, FF value': 3.0}, (2.9, 1.9), highlight=('LSTM policy',), names={'LSTM policy': 'LSTM', 'FF policy, LSTM value': 'FF, LSTM V', 'FF policy, FF value': 'FF, FF V'})
    for k in (1, 2, 3):
        mode_avg(k)
        decoder(k)
        spring(k)
        odd(k)
    drift()
    discount()
    for k in (1, 2):
        offline(k)
    cells_support()
    reset_wall()
    for k in (1, 2, 3, 4):
        gap_vector(k)
