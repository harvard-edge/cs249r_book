#!/usr/bin/env python3
"""Figures for Lecture 3 (Volume IV, The Cognitive Brain).

Every number drawn here comes from calcs.py, which reads MLSysIM (the
Embodied.* machine and site, Literature.*, and the Scenarios.* the chapter also
reads): plotted geometry uses the scenario values (namespace K below), and
labels use the formatted strings the slides print (deckvals val). Axis ranges,
tick positions, and label coordinates are layout. Figures are drawn at the size of their slot on the 16 x 9 cm
page and saved at that size (save_slot), so the slide places them at scale 1
and every label keeps its drawn size (8 to 9.5 pt).

  - MLPerf Inference edge, single stream, p90, Jetson AGX Orin 64G (TensorRT),
    latest round per model: Scenarios.OrinDevKitSingleStream, as the chapter's
    figure 9 reads it.
  - Memory wall, roofline, evaluation bars, decoder ladder:
    Scenarios.IntentModelEvaluation*, IntentModelDecodeRoofline,
    IntentBackboneDecoderFamilies, WarehouseProposerTraffic.
  - Published two-speed rates: Literature.VlaSystems (Figure AI 2025; Bjorck et al. 2025).
  - Evidence horizon and timing strip: Scenarios.ConveyorMugEvidenceHorizon.
  - KV ladder: Scenarios.ThreeViewKvHistory. Confidence and success:
    Scenarios.ProposerConfidenceRegimes (stipulated, illustrative).
  - Volume IV schematic colors (palette.md): cognition navy #1F407A, proposal
    blue #2563EB (dashed), permission green #059669, veto red #DC2626, plant
    slate #334155. Resource plots keep the base contract (memory blue,
    compute orange).

Run from this folder: ../../../../.venv/bin/python make_figs.py
"""
import contextlib
import io
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle
m = lambda q, u: q.to(u).magnitude
C = sp.C
plt = sp.plt
out = Path(__file__).parent
NAVY, PROP, GREEN, RED, SLATE, AMBER = ('#1F407A', '#2563EB', '#059669', '#DC2626', '#334155', '#D97706')
FS, TFS, SFS = (9, 8.5, 8)

def cm(w, h):
    return (w / 2.54, h / 2.54)

def save_slot(fig, path):
    """Save at exactly the figure size, so the slide places it at scale 1."""
    fig.savefig(path, format='svg', facecolor='white')
    plt.close(fig)

def tag(ax, text, x=0.0, y=-0.3, fs=SFS):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, color=C['neutral'], ha='left', va='top')

def hide(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def ms_label(v):
    if v < 10:
        return f'{v:.2f} ms'
    if v < 100:
        return f'{v:.1f} ms'
    if v < 1000:
        return f'{v:.0f} ms'
    return f'{v / 1000:.1f} s'

def mlperf_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.9))
    fig.subplots_adjust(left=0.36, right=0.98, top=0.9, bottom=0.2)
    rows = [('Image classification', 'ResNet-50', 0.640259, 'v3.1'), ('Question answering', 'BERT', 5.712265, 'v3.1'), ('Object detection', 'RetinaNet', 11.674034, 'v3.1'), ('Speech recognition', 'RNN-T', 94.006947, 'v3.1'), ('3D segmentation', '3D U-Net', 4371.455994, 'v3.1'), ('Text summarization', 'GPT-J 6B', 4176.62608, 'v4.1'), ('Text to image', 'SDXL', 9966.781359, 'v4.1')]
    for i, (task, model, v, rnd) in enumerate(rows):
        col = C['memory'] if v <= 50.0 else C['compute'] if v <= 500.0 else C['neutral']
        shown = k >= 2
        ax.barh(i, v, color=col, height=0.62, alpha=1.0 if shown else 0.0)
        ax.text(v * 1.18, i, ms_label(v), va='center', fontsize=FS, color=col, zorder=6, fontweight='bold', alpha=1.0 if shown else 0.0, bbox=dict(fc='white', ec='none', pad=0.4, alpha=1.0 if shown else 0.0))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{r[0]} ({r[1]})' for r in rows], fontsize=FS)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlim(0.1, 300000.0)
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(['0.1', '1', '10', '100', '1,000', '10,000', '100,000'])
    ax.minorticks_off()
    ax.tick_params(axis='x', labelsize=TFS, pad=1)
    ax.tick_params(axis='y', length=0)
    for t, lab, ha, dx in ((50.0, 'chunk period 50 ms', 'right', 0.9), (500.0, 'intent period 500 ms', 'left', 1.1)):
        ax.axvline(t, color=C['danger'], ls='--', lw=1.5, zorder=1)
        ax.text(t * dx, -0.85, lab, fontsize=FS, color=C['danger'], ha=ha, va='center', fontweight='bold')
    ax.set_ylim(len(rows) - 0.5, -1.25)
    ax.set_xlabel('90th-percentile latency of one query (ms, log scale)', fontsize=FS, labelpad=1)
    save_slot(fig, out / f'mlperf_step{k}.svg')
for k in (1, 2):
    mlperf_fig(k)

def cadence_fig():
    fig, ax = plt.subplots(figsize=cm(14.2, 4.8))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.21)
    n = np.logspace(7, 11, 200)
    for bpp, lab, ls in ((2, 'FP16 ceiling', '-'), (0.5, 'INT4 ceiling', '--')):
        ax.plot(n, 142800000000.0 / (n * bpp), color=C['memory'], lw=2, ls=ls)
    ax.text(30000000.0, 142800000000.0 / (30000000.0 * 2) * 0.55, 'FP16 ceiling', color=C['memory'], fontsize=FS, fontweight='bold', rotation=-17, ha='left', va='top')
    ax.text(4000000000.0, 142800000000.0 / (4000000000.0 * 0.5) * 1.35, 'INT4 ceiling', color=C['memory'], fontsize=FS, fontweight='bold', rotation=-17, ha='left', va='bottom')
    for r, lab, x, ha in ((1000.0, 'permission tick 1 kHz', 110000000000.0, 'right'), (20.0, 'chunk renewal 20 Hz', 11000000.0, 'left'), (2.0, 'intent revision 2 Hz', 11000000.0, 'left')):
        ax.axhline(r, color=C['neutral'], lw=1, ls=':')
        ax.text(x, r * 1.12, lab, fontsize=SFS, color=C['neutral'], ha=ha, va='bottom')
    pts = [(80000000.0, 20.0, 'chunk policy 80M', (6, 4), 'left'), (7000000000.0, 2.0, 'intent model 7B', (-6, 6), 'right')]
    for x, y, lab, off, ha in pts:
        ax.plot([x], [y], 'o', ms=8, color=NAVY, mec='white', mew=1.2, zorder=5)
        ax.annotate(lab, (x, y), xytext=off, textcoords='offset points', fontsize=FS, fontweight='bold', color=NAVY, ha=ha, va='bottom')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10000000.0, 120000000000.0)
    ax.set_ylim(0.5, 30000.0)
    ax.set_xticks([10000000.0, 100000000.0, 1000000000.0, 10000000000.0, 100000000000.0])
    ax.set_xticklabels(['10M', '100M', '1B', '10B', '100B'])
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_yticklabels(['1', '10', '100', '1,000', '10,000'])
    ax.minorticks_off()
    ax.tick_params(labelsize=TFS, pad=1)
    ax.set_xlabel('Parameters', fontsize=FS, labelpad=1)
    ax.set_ylabel('Fresh evaluations per s', fontsize=FS, labelpad=1)
    save_slot(fig, out / 'cadence.svg')
cadence_fig()

def roofline_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.8))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.95, bottom=0.21)
    peak = 43000000000000.0 / 1000000000000.0
    bw = 204000000000.0 / 1000000000000.0
    xs = np.logspace(-1, 4, 300)
    roof = np.minimum(peak, bw * xs)
    m = xs <= 210.7843137
    ax.plot(xs[m], roof[m], color=C['memory'], lw=2.5)
    ax.plot(xs[~m], roof[~m], color=C['compute'], lw=2.5)
    ax.fill_between(xs[m], 0.001, roof[m], color=C['memory_l'], alpha=0.6)
    ax.fill_between(xs[~m], 0.001, roof[~m], color=C['compute_l'], alpha=0.6)
    ax.axvline(210.7843137, color=C['neutral'], lw=1, ls='--')
    ax.text(210.7843137 * 1.1, 0.012, 'ridge 210.8 FLOP/byte', fontsize=SFS, color=C['neutral'])
    ax.text(0.15, peak * 1.25, 'dense FP16 peak 43 TFLOP/s', fontsize=SFS, color=C['compute'])
    ax.text(0.9, 0.012, 'slope: 204 GB/s', fontsize=SFS, color=C['memory'])
    if k >= 2:
        x, y = (1.0, bw * 1.0)
        ax.plot([x], [y], 'o', ms=9, color=C['memory'], mec='white', mew=1.4, zorder=5)
        ax.annotate('decode, one token: 1 FLOP/byte\n0.204 TFLOP/s, 0.47% of peak', (x, y), xytext=(0.13, 2.2), textcoords='data', ha='left', va='bottom', fontsize=FS, color=C['memory'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=C['memory'], lw=1))
    if k >= 3:
        x, y = (288.0, peak)
        ax.plot([x], [y], 'o', ms=9, color=C['compute'], mec='white', mew=1.4, zorder=5)
        ax.annotate('prefill, 288 tokens:\n288 FLOP/byte', (x, y), xytext=(10, -26), textcoords='offset points', ha='left', fontsize=FS, color=C['compute'], fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.1, 10000.0)
    ax.set_ylim(0.008, 150)
    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000])
    ax.set_xticklabels(['0.1', '1', '10', '100', '1,000', '10,000'])
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
    ax.minorticks_off()
    ax.tick_params(labelsize=TFS, pad=1)
    ax.set_xlabel('Arithmetic intensity (FLOP per byte from off-chip memory)', fontsize=FS, labelpad=1)
    ax.set_ylabel('TFLOP/s', fontsize=FS, labelpad=1)
    save_slot(fig, out / f'roofline_step{k}.svg')
for k in (1, 2, 3):
    roofline_fig(k)

def eval_fig(k, size=(14.2, 2.6), name='eval'):
    fig, ax = plt.subplots(figsize=cm(*size))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.36)
    rows = [('FP16', 588.2352941, 776.7814928, '588', '777'), ('INT4', 147.0588235, 335.6050222, '147', '336')]
    for i, (lab, dec, tot, dec_txt, tot_txt) in enumerate(rows):
        shown = i < k
        a = 1.0 if shown else 0.0
        ax.barh(i, 188.5461987, color=C['compute'], height=0.6, alpha=a)
        ax.barh(i, dec, left=188.5461987, color=C['memory'], height=0.6, alpha=a)
        ax.text(188.5461987 / 2, i, 'prefill 189', ha='center', va='center', color='white', fontsize=FS, fontweight='bold', alpha=a)
        ax.text(188.5461987 + dec / 2, i, f'decode {dec_txt}', ha='center', va='center', color='white', fontsize=FS, fontweight='bold', alpha=a)
        ax.text(tot + 12, i, f'{tot_txt} ms', va='center', fontsize=FS, fontweight='bold', color=C['text'], alpha=a)
    ax.axvline(500.0, color=C['danger'], ls='--', lw=1.5)
    ax.text(500.0 + 8, 1.55, '500 ms period (2 Hz)', color=C['danger'], fontsize=SFS, fontweight='bold', va='center')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['FP16', 'INT4'], fontsize=FS, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()
    ax.set_ylim(1.8, -0.5)
    ax.set_xlim(0, 900)
    ax.tick_params(axis='x', labelsize=TFS, pad=1)
    ax.set_xlabel('ms for one fresh evaluation (computed, Napkin Math 3.1)', fontsize=FS, labelpad=1)
    save_slot(fig, out / f'{name}_step{k}.svg')
for k in (1, 2):
    eval_fig(k)

def supply_fig(k):
    fig, ax = plt.subplots(figsize=cm(8.4, 3.7))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.82, bottom=0.3)
    rows = [('one setpoint\nper pass', 560000000000.0 / 1000000000.0, C['danger'], '560'), ('chunk of 16', 35000000000.0 / 1000000000.0, C['data'], '35')]
    for i, (lab, v, col, txt) in enumerate(rows):
        a = 1.0 if i < k else 0.0
        ax.barh(i, v, color=col, height=0.6, alpha=a)
        ax.text(v + 8, i, f'{txt} GB/s', va='center', fontsize=FS, fontweight='bold', color=col, alpha=a)
    ax.axvline(142800000000.0 / 1000000000.0, color=C['memory'], ls='--', lw=1.5)
    ax.text(142800000000.0 / 1000000000.0, -0.62, 'sustained 142.8 GB/s', color=C['memory'], fontsize=SFS, fontweight='bold', ha='center', va='bottom')
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS)
    ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()
    ax.set_ylim(1.5, -0.65)
    ax.set_xlim(0, 700)
    ax.tick_params(axis='x', labelsize=TFS, pad=1)
    ax.set_xlabel('FP16 weight traffic at 40 setpoints/s', fontsize=FS, labelpad=1)
    save_slot(fig, out / f'supply_step{k}.svg')
for k in (1, 2):
    supply_fig(k)

def decoder_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.4))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.22)
    rows = [('Binned tokens', 2909.134434, '112 sequential', '2.9 s'), ('FAST tokens', 899.3305124, '30 sequential', '899 ms'), ('Parallel head', 264.3667944, 'one longer prefill', '264 ms'), ('Flow expert', 241.136881, '10 expert steps', '241 ms')]
    for i, (lab, v, why, txt) in enumerate(rows):
        a = 1.0 if k >= 2 else 0.0
        ax.barh(i, v, color=NAVY, height=0.6, alpha=a)
        ax.text(v * 1.08, i, f'{txt}   ({why})', va='center', fontsize=FS, fontweight='bold', color=NAVY, alpha=a)
    ax.axvline(188.5461987, color=C['compute'], ls=':', lw=1.8)
    ax.text(188.5461987 * 1.05, -0.55, 'shared prefill 189 ms', color=C['compute'], fontsize=SFS, fontweight='bold', ha='left', va='center')
    ax.axvline(50.0, color=C['danger'], ls='--', lw=1.8)
    ax.text(50.0 * 1.05, -1.0, 'renewal 50 ms', color=C['danger'], fontsize=SFS, fontweight='bold', ha='left', va='center')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(len(rows) - 0.5, -1.3)
    ax.set_xscale('log')
    ax.set_xlim(20, 30000.0)
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 3000, 10000])
    ax.set_xticklabels(['20', '50', '100', '200', '500', '1,000', '3,000', '10,000'])
    ax.minorticks_off()
    ax.tick_params(axis='x', labelsize=TFS, pad=1)
    ax.set_xlabel('ms for one fresh 16-step chunk, INT4 7B backbone on the Orin (computed)', fontsize=FS, labelpad=1)
    save_slot(fig, out / f'decoder_step{k}.svg')
for k in (1, 2):
    decoder_fig(k)

def dual_fig():
    fig, ax = plt.subplots(figsize=cm(9.0, 4.4))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.86, bottom=0.25)
    for i, (name, lo, hi, fast) in enumerate([('Helix', 7.0, 9.0, 200.0), ('GR00T N1', 10.0, 10.0, 120.0), ('This machine', 2.0, 2.0, 20.0)]):
        y = i
        ax.plot([hi, fast], [y, y], color=C['neutral_l'], lw=4, zorder=1)
        if hi > lo:
            ax.plot([lo, hi], [y, y], color=NAVY, lw=8, solid_capstyle='round', zorder=3)
        else:
            ax.plot([lo], [y], 'o', ms=9, color=NAVY, zorder=3)
        ax.plot([fast], [y], 'o', ms=9, color=PROP, zorder=3)
        slow_txt = f'{lo:.0f} to {hi:.0f} Hz' if hi > lo else f'{lo:.0f} Hz'
        ax.text(lo / 1.25, y - 0.32, slow_txt, ha='right', va='center', fontsize=SFS, color=NAVY, fontweight='bold')
        ax.text(fast * 1.25, y - 0.32, f'{fast:.0f} Hz', ha='left', va='center', fontsize=SFS, color=PROP, fontweight='bold')
    ax.text(1.1, -0.95, 'slow model', color=NAVY, fontsize=FS, fontweight='bold', ha='left')
    ax.text(900, -0.95, 'fast policy', color=PROP, fontsize=FS, fontweight='bold', ha='right')
    ax.set_yticks(range(len([('Helix', 7.0, 9.0, 200.0), ('GR00T N1', 10.0, 10.0, 120.0), ('This machine', 2.0, 2.0, 20.0)])))
    ax.set_yticklabels(['Helix', 'GR00T N1', 'This machine'], fontsize=FS, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(len([('Helix', 7.0, 9.0, 200.0), ('GR00T N1', 10.0, 10.0, 120.0), ('This machine', 2.0, 2.0, 20.0)]) - 0.5, -1.3)
    ax.set_xscale('log')
    ax.set_xlim(0.8, 900)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(['1', '10', '100'])
    ax.minorticks_off()
    ax.tick_params(axis='x', labelsize=TFS, pad=1)
    ax.set_xlabel('Rate (Hz, log scale)', fontsize=FS, labelpad=1)
    save_slot(fig, out / 'dual_rates.svg')
dual_fig()

def box(ax, x, y, w, h, txt, ec, fc, ls='-', fs=FS, bold=True, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.12', ec=ec, fc=fc, lw=1.6, ls=ls))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color=tc or ec)

def arrow(ax, p, q, col, ls='-', lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, color=col, lw=lw, ls=ls))

def twospeed_fig():
    fig, ax = plt.subplots(figsize=cm(6.8, 5.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 6.8)
    ax.set_ylim(0, 5.0)
    hide(ax)
    box(ax, 0.1, 3.75, 3.6, 1.0, 'Intent model, 7B INT4\n2 Hz, 336 ms', NAVY, '#EFF6FF')
    box(ax, 0.1, 2.15, 3.6, 1.0, 'Chunk policy, 80M\n20 Hz, 40 ms', NAVY, '#EFF6FF')
    box(ax, 0.1, 0.25, 3.6, 1.15, '16 setpoints,\n25 ms apart', PROP, '#F8FAFC', ls='--')
    arrow(ax, (1.9, 3.75), (1.9, 3.15), NAVY)
    ax.text(2.05, 3.45, 'target +\ncapture time', fontsize=SFS, color=NAVY, va='center')
    arrow(ax, (1.9, 2.15), (1.9, 1.4), PROP, ls='--')
    ax.plot([4.25, 4.25], [0.1, 4.9], color='#475569', ls=(0, (4, 3)), lw=1.3)
    ax.text(4.35, 4.75, 'proposal\nboundary', fontsize=SFS, color='#475569', va='top')
    box(ax, 4.55, 1.85, 2.15, 1.3, 'Permission\npath\n1 kHz', GREEN, '#F0FDF4')
    box(ax, 4.55, 0.25, 2.15, 1.0, 'Drives', SLATE, '#FFFFFF')
    arrow(ax, (3.7, 0.85), (4.55, 2.2), PROP, ls='--')
    arrow(ax, (5.62, 1.85), (5.62, 1.25), GREEN)
    save_slot(fig, out / 'twospeed.svg')
twospeed_fig()

def drift_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.7))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.22)
    t = np.linspace(0, 420, 400)
    ax.axhline(12.0, color=C['danger'], ls='--', lw=1.6)
    ax.text(415, 12.0 - 1.0, 'grasp margin 12 mm', color=C['danger'], fontsize=SFS, fontweight='bold', ha='right', va='top')
    belt = 0.2 * t
    ax.plot(t[t <= 100], belt[t <= 100], color=C['neutral'], lw=2.2)
    ax.plot([60.0], [12.0], 'o', ms=7, color=C['neutral'], zorder=5)
    ax.text(6, 44, 'untracked, belt drift:\nmargin gone at 60 ms', ha='left', va='top', fontsize=FS, color=C['neutral'], fontweight='bold')
    if k >= 2:
        slip = 0.5 * 0.5 * (t / 1000) ** 2 * 1000
        sel = t <= (400.0 if k >= 3 else 219.089023 + 15)
        ax.plot(t[sel], slip[sel], color=C['data'], lw=2.4)
        ax.plot([219.089023], [12.0], 'o', ms=8, color=C['data'], zorder=5)
        ax.annotate('tracked, slip only:\nevidence horizon 219 ms', (219.089023, 12.0), xytext=(125, 30), textcoords='data', ha='left', va='center', fontsize=FS, color=C['data'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=C['data'], lw=1))
        ax.plot([90.0], [2.025], 'o', ms=7, color=NAVY, zorder=5)
        ax.annotate('replaced at 90 ms: 2.0 mm', (90.0, 2.025), xytext=(135, 0.8), textcoords='data', ha='left', va='bottom', fontsize=SFS, color=NAVY, arrowprops=dict(arrowstyle='-', color=NAVY, lw=1), fontweight='bold')
    if k >= 3:
        ax.plot([400.0], [40.0], 'o', ms=8, color=C['danger'], zorder=5)
        ax.annotate('chunk end 400 ms:\n40 mm', (400.0, 40.0), xytext=(-8, 4), textcoords='offset points', ha='right', va='bottom', fontsize=FS, color=C['danger'], fontweight='bold')
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 52)
    ax.set_xticks([0, 60.0, 90.0, 150, 219.089023, 300, 400.0])
    ax.set_xticklabels(['0', '60', '90', '150', '219', '300', '400'])
    ax.tick_params(labelsize=TFS, pad=1)
    ax.set_xlabel('Evidence age since capture (ms)', fontsize=FS, labelpad=1)
    ax.set_ylabel('Mug drift (mm)', fontsize=FS, labelpad=1)
    save_slot(fig, out / f'drift_step{k}.svg')
for k in (1, 2, 3):
    drift_fig(k)

def strip_fig(k):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.6))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.2)
    lanes = {'capture': 4, 'inference': 3, 'in force': 2, 'chunk 0': 1, 'setpoints': 0}
    T = 50.0
    for cap in np.arange(0, 400.0, T):
        ax.plot([cap, cap], [lanes['capture'] - 0.3, lanes['capture'] + 0.3], color=SLATE, lw=2)
        ax.add_patch(Rectangle((cap, lanes['inference'] - 0.28), 40.0, 0.56, fc=NAVY, ec='none', alpha=0.85))
    for j, cap in enumerate(np.arange(40.0, 400.0, T)):
        col = GREEN if j == 0 else '#A7F3D0'
        ax.add_patch(Rectangle((cap, lanes['in force'] - 0.28), T, 0.56, fc=col, ec='white', lw=1))
        if cap + T <= 400.0:
            ax.text(cap + T / 2, lanes['in force'], f'{j}', ha='center', va='center', fontsize=SFS, color='white' if j == 0 else SLATE, fontweight='bold')
    for s in np.arange(0, 400.0 + 1, 25.0):
        ax.plot([s, s], [lanes['setpoints'] - 0.18, lanes['setpoints'] + 0.18], color=PROP, lw=1.3)
    y = lanes['chunk 0']
    ax.add_patch(Rectangle((0, y - 0.3), 40.0, 0.6, fc=C['neutral_l'], ec='none'))
    ax.add_patch(Rectangle((40.0, y - 0.3), 90.0 - 40.0, 0.6, fc=GREEN, ec='none'))
    ax.text(40.0 + (90.0 - 40.0) / 2, y, 'runs', ha='center', va='center', fontsize=SFS, color='white', fontweight='bold')
    if k >= 2:
        ax.add_patch(Rectangle((90.0, y - 0.3), 219.089023 - 90.0, 0.6, fc='#D1FAE5', ec='none'))
        ax.text((90.0 + 219.089023) / 2, y, 'evidence still valid', ha='center', va='center', fontsize=SFS, color=SLATE, fontweight='bold')
        ax.plot([219.089023, 219.089023], [-0.5, 4.5], color=C['danger'], ls='--', lw=1.6)
        ax.text(219.089023 + 3, 4.55, 'evidence horizon 219 ms', color=C['danger'], fontsize=SFS, fontweight='bold', va='center')
    if k >= 3:
        ax.add_patch(Rectangle((219.089023, y - 0.3), 400.0 - 219.089023, 0.6, fc='white', ec=C['danger'], hatch='///', lw=1.0))
        ax.text((219.089023 + 400.0) / 2, y, 'buffer past the evidence (181 ms)', ha='center', va='center', fontsize=SFS, color=C['danger'], fontweight='bold', bbox=dict(fc='white', ec='none', pad=0.6))
    ax.set_yticks(list(lanes.values()))
    ax.set_yticklabels(list(lanes.keys()), fontsize=FS)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.6, 4.8)
    ax.set_xlim(0, 405)
    ax.set_xticks([0, 40.0, 90.0, 150, 219.089023, 300, 400.0])
    ax.set_xticklabels(['0', '40', '90', '150', '219', '300', '400'])
    ax.tick_params(axis='x', labelsize=TFS, pad=1)
    for s in ('left', 'right', 'top'):
        ax.spines[s].set_visible(False)
    ax.set_xlabel('ms since the capture behind chunk 0', fontsize=FS, labelpad=1)
    save_slot(fig, out / f'strip_step{k}.svg')
for k in (1, 2, 3):
    strip_fig(k)

def blend_fig():
    fig, ax = plt.subplots(figsize=cm(7.6, 2.9))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 2.9)
    hide(ax)
    ax.add_patch(Circle((3.8, 1.45), 0.42, fc=C['danger_l'], ec=RED, lw=1.6))
    s_ = np.linspace(0, 1, 60)
    x = 0.3 + 7.0 * s_
    left = 1.45 + 0.95 * np.sin(np.pi * s_)
    right = 1.45 - 0.95 * np.sin(np.pi * s_)
    ax.plot(x, left, color=NAVY, lw=2)
    ax.plot(x, right, color=PROP, lw=2)
    ax.plot(x, (left + right) / 2, color=RED, lw=2.2, ls='--')
    ax.text(3.8, 2.55, 'older chunk: left', ha='center', va='center', fontsize=SFS, color=NAVY, fontweight='bold')
    ax.text(3.8, 0.33, 'newer chunk: right', ha='center', va='center', fontsize=SFS, color=PROP, fontweight='bold')
    ax.text(5.6, 1.55, 'average', ha='left', va='bottom', fontsize=SFS, color=RED, fontweight='bold')
    ax.text(3.8, 0.95, 'obstacle', ha='center', va='top', fontsize=SFS, color=RED, fontweight='bold')
    save_slot(fig, out / 'blend.svg')
blend_fig()

def kv_fig():
    rows = [('Module memory', 68.71947674, C['neutral'], '68.7 GB'), ('5 s of history', 20.97152, C['danger'], '21.0 GB'), ('FP16 weights', 14.0, C['memory'], '14 GB'), ('One step, 800 tokens', 419.4304 / 1000, C['data'], '419 MB'), ('One token', 0.000524288, C['data'], '0.52 MB')]
    fig, ax = plt.subplots(figsize=cm(12.4, 4.0))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.97, bottom=0.25)
    for i, (lab, v, col, txt) in enumerate(rows):
        ax.barh(i, v, color=col, height=0.6)
        ax.text(v * 1.2, i, txt, va='center', fontsize=FS, fontweight='bold', color=col)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS)
    ax.tick_params(axis='y', length=0)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlim(0.0001, 1000.0)
    ax.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(['0.001', '0.01', '0.1', '1', '10', '100'])
    ax.minorticks_off()
    ax.tick_params(axis='x', labelsize=TFS, pad=1)
    ax.set_xlabel('GB (log scale), full-width FP16 keys and values, 32 layers', fontsize=FS, labelpad=1)
    save_slot(fig, out / 'kv_ladder.svg')
kv_fig()

def depth_fig():
    fig, axs = plt.subplots(1, 2, figsize=cm(7.4, 4.0), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.83, bottom=0.27, wspace=0.12)
    x = np.linspace(0, 100, 200)
    real = 30 * np.exp(-((x - 50) / 14) ** 2)
    for ax, y, title, col in ((axs[0], real, 'mug handle', GREEN), (axs[1], 0 * x, 'printed photo', RED)):
        ax.plot(x, y, color=col, lw=2.4)
        ax.set_title(title, fontsize=FS, color=col, fontweight='bold', pad=3)
        ax.set_xticks([])
        ax.set_ylim(-4, 36)
        ax.tick_params(labelsize=TFS, pad=1)
        ax.axvspan(40, 60, color=C['neutral_l'], alpha=0.6)
    axs[0].set_ylabel('Relief (mm)', fontsize=FS, labelpad=1)
    axs[0].set_xlabel('along the grasp', fontsize=SFS, labelpad=1)
    axs[1].set_xlabel('along the grasp', fontsize=SFS, labelpad=1)
    axs[1].text(50, 8, 'flat: refuse', ha='center', fontsize=FS, color=RED, fontweight='bold')
    save_slot(fig, out / 'depth_check.svg')
depth_fig()

def conf_fig():
    fig, ax = plt.subplots(figsize=cm(8.6, 4.3))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.27)
    names = ['In distribution', 'Near OOD', 'Sensor degraded', 'Far OOD']
    xs = np.arange(len(names))
    conf = [0.91, 0.89, 0.84, 0.78]
    acc = [0.89, 0.52, 0.31, 0.08]
    ax.bar(xs - 0.2, conf, 0.38, color=NAVY)
    ax.bar(xs + 0.2, acc, 0.38, color=C['data'])
    for x, cf, a in zip(xs, conf, acc):
        ax.text(x - 0.2, cf + 0.02, f'{cf:.2f}', ha='center', fontsize=SFS, color=NAVY, fontweight='bold')
        ax.text(x + 0.2, a + 0.02, f'{a:.2f}', ha='center', fontsize=SFS, color=C['data'], fontweight='bold')
    ax.text(-0.45, 1.1, 'confidence', color=NAVY, fontsize=FS, fontweight='bold', ha='left')
    ax.text(1.35, 1.1, 'task success', color=C['data'], fontsize=FS, fontweight='bold', ha='left')
    ax.set_xticks(xs)
    ax.set_xticklabels(['In\ndistribution', 'Near\nOOD', 'Sensor\ndegraded', 'Far\nOOD'], fontsize=SFS)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis='y', labelsize=TFS, pad=1)
    save_slot(fig, out / 'conf_success.svg')
conf_fig()

def synth_fig():
    fig, ax = plt.subplots(figsize=cm(14.2, 3.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    hide(ax)
    ax.set_xlim(-95, 412)
    ax.set_ylim(-0.25, 3.45)
    ax.text(-8, 2.55, 'intent model', ha='right', va='center', fontsize=FS, color=C['text'], fontweight='bold')
    ax.text(-8, 1.4, 'chunk policy', ha='right', va='center', fontsize=FS, color=NAVY, fontweight='bold')
    ax.add_patch(Rectangle((0, 2.2), 188.5461987, 0.7, fc=C['compute'], ec='white'))
    ax.add_patch(Rectangle((188.5461987, 2.2), 335.6050222 - 188.5461987, 0.7, fc=C['memory'], ec='white'))
    ax.text(188.5461987 / 2, 2.55, 'prefill 189 ms (arithmetic)', ha='center', va='center', fontsize=FS, color='white', fontweight='bold')
    ax.text(335.6050222 - 4, 2.55, 'decode 147 ms', ha='right', va='center', fontsize=FS, color='white', fontweight='bold')
    for cap in np.arange(0, 400.0, 50.0):
        ax.add_patch(Rectangle((cap, 1.15), 40.0, 0.5, fc=NAVY, ec='none', alpha=0.85))
    ax.plot([219.089023, 219.089023], [0.9, 3.05], color=C['danger'], ls='--', lw=1.6)
    ax.text(219.089023 + 3, 3.08, 'evidence horizon 219 ms', color=C['danger'], fontsize=SFS, fontweight='bold', va='bottom')
    ax.plot([0, 400], [0.35, 0.35], color=SLATE, lw=1.2)
    for t in (0, 100, 200, 300, 400):
        ax.plot([t, t], [0.25, 0.45], color=SLATE, lw=1.2)
        ax.text(t, 0.05, f'{t} ms', ha='center', va='top', fontsize=TFS, color=SLATE)
    save_slot(fig, out / 'synthesis.svg')
synth_fig()
print('figures written')
