#!/usr/bin/env python3
"""Data plots for Lecture 4 (Data Engineering).

Every plotted value and every number in a label comes from calcs.py, which
reads MLSysIM (the keyword-spotter and labeling Scenarios.*, Literature,
Datasets); this file types no data. The scenarios behind each figure:
  - Listening hours: Scenarios.KeywordSpotterFalseWakeEvidence (Sec. 4.11).
  - Label entries: Scenarios.FullHdMaskVersusBoxes (Sec. 4.6.1).
  - Label cost: Scenarios.ImageNetLabelingVersusResNetRun (Sec. 4.6.2).
  - Picovoice miss rates: Literature.WakeWordBenchmarks.
  - Label errors: vol1/04_data_engineering/data/test_set_label_errors.csv (Northcutt et al. 2021).
  - Forced alignment: Datasets.MSWC.

2026-10-17: the storage latency, storage supply, format, drift-type, and K-S
figures moved to Lecture 4b (Data Engineering, Part 2), whose own
images/make_figs.py regenerates them. Their old SVG and PDF files in this
folder are no longer used by this deck.

Diagrams (leakage split by file, skew, pipeline synthesis) are SVG
written by this script.

Run from this folder: python3 make_figs.py
"""
_NUMS = {'tay_hours': 16.0, 'kws_detect_pct': 98.0, 'kws_false_wakes': 1.0, 'kws_model_kb': 256.0, 'kws_prep_kb': 16.0, 'kws_budget_k': 150.0, 'kws_budget_usd': 150000.0, 'kws_months': 6.0, 'kws_base_pct': 90.0, 'rej_a': 99.9, 'rej_b': 99.99, 'rej_c': 99.999, 'fw_a': 2592.0, 'fw_b': 259.0, 'fw_c': 26.0, 'rej_a_miss': 0.1, 'fw_a_round': 2600.0, 'fw_a_day': 86.0, 'fw_c_over': 26.0, 'windows_month': 2592000.0, 'listen_h_day': 24.0, 'days_month': 30.0, 'rej_needed': 99.99996, 'fpr_mant': 3.9, 'fpr_exp': -7.0, 'pv_pocketsphinx': 52.0, 'pv_snowboy': 31.9, 'pv_porcupine': 2.7, 'pv_hours': 10.0, 'pv_month': 72.0, 'pv_over': 72.0, 'month_hours': 720.0, 'pv_recordings': 300.0, 'pv_speakers': 50.0, 'pv_snr': 10.0, 'pv_envs': 18.0, 'share_label': 60.0, 'share_label_frac': 0.6, 'label_price': 0.1, 'review_pct': 20.0, 'review_mult': 1.2, 'labeling_k': 90.0, 'labeling_usd': 90000.0, 'label_cost': 0.12, 'labels_k': 750.0, 'labels': 750000.0, 'storage_k': 37.5, 'governance_k': 22.5, 'mfcc_lo': 13.0, 'mfcc_hi': 40.0, 'conf_pct': 95.0, 'rule3': 3.0, 'eval_hours': 2160.0, 'eval_gb': 248.0, 'eval_khz': 16.0, 'eval_bits': 16.0, 'clip_s': 1.0, 'mask_w': 1920.0, 'mask_h': 1080.0, 'mask_m': 2.1, 'box_entries': 40.0, 'mask_ratio': 51840.0, 'box_s': 35.0, 'mask_hours': 1.5, 'mask_total_mh': 1.5, 'mask_person_years': 750.0, 'work_year_h': 2000.0, 'lab_budget_k': 100.0, 'lab_budget_usd': 100000.0, 'lab_weeks_lo': 2.0, 'lab_weeks_hi': 4.0, 'gpus': 8.0, 'epochs': 90.0, 'train_h': 1.3, 'train_usd_lo': 20.0, 'train_usd_hi': 40.0, 'lab_ratio_lo': 2500.0, 'lab_ratio_hi': 5000.0, 'img_per_s': 3200.0, 'err_quickdraw': 10.12, 'err_cifar100': 5.85, 'err_imagenetval': 5.83, 'err_imdb': 2.9, 'err_audioset': 1.35, 'err_mnist': 0.15, 'err_max': 10.0, 'not_spam_pct': 90.0, 'not_spam': 0.9, 'spam': 0.1, 'p_e': 0.82, 'p_o_pct': 85.0, 'kappa_base': 0.85, 'kappa_later': 0.72, 'al_pool_m': 10.0, 'al_budget_k': 50.0, 'al_budget_usd': 50000.0, 'al_label': 0.5, 'al_score': 0.01, 'al_base_pct': 85.0, 'al_target_pct': 95.0, 'al_random_m': 1.0, 'al_random_k': 500.0, 'al_random_usd': 500000.0, 'al_active_lo': 100000.0, 'al_active_hi': 200000.0, 'al_active_usd_lo': 50000.0, 'al_active_usd_hi': 100000.0, 'al_scoreall_k': 100.0, 'al_scoreall_usd': 100000.0, 'al_cand_k': 500.0, 'al_cand': 500000.0, 'al_compute_k': 5.0, 'al_compute_usd': 5000.0, 'al_rest_usd': 45000.0, 'al_labels_k': 90.0, 'al_labels': 90000.0, 'mswc_m': 23.4, 'mswc_kw': 340000.0, 'mswc_kw_k': 340.0, 'mswc_lang': 50.0, 'mswc_s': 10.0, 'mswc_hours': 65000.0, 'mswc_years': 32.5, 'mswc_clip_s': 1.0, 'sc_val_pct': 10.0, 'sc_test_pct': 10.0, 'sc_train_pct': 80.0, 'sc_train': 84848.0, 'sc_valid': 9982.0, 'sc_test': 4890.0, 'sc_test_share': 4.9, 'ex_mult': 10.0, 'ex_gain_lo': 5.0, 'ex_gain_hi': 8.0, 'corpus_growth': 2.9, 'text_stock_t': 300.0, 'stock_year': 2028.0}
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
import numpy as np
out = Path(__file__).parent
C = sp.C
plt = sp.plt

def bars_lines(path, rows, step=None, lines=(), log=True, xlim=None, xlabel=None, size=sp.SIZE, note=None, xticks=None):
    """Horizontal bars with direct value labels and optional vertical
    reference lines. rows: (label, value, color, value_text).
    lines: (x, text, color, text_y) reference lines, all drawn every step."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, r in enumerate(rows):
        label, v, ck, txt = r[:4]
        inside = len(r) > 4 and r[4] == 'in'
        y = n - 1 - i
        shown = step is None or i < step
        ax.barh(y, v if shown else 0, color=C[ck], height=0.62, alpha=1.0 if shown else 0.0)
        ax.text(-0.01, y, label, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, color=C['text'], fontweight='bold')
        if shown and inside:
            ax.text(v / 1.25, y, txt, va='center', ha='right', fontsize=13, color='white', fontweight='bold')
        elif shown:
            ax.text(v * 1.18 if log else v + (xlim[1] if xlim else v) * 0.015, y, txt, va='center', fontsize=13, color=C[ck], fontweight='bold')
    for x, text, ck, ty in lines:
        ax.axvline(x, color=C[ck], lw=2, ls='--')
        ax.text(x * (1.12 if log else 1.0), ty, text, color=C[ck], fontsize=11, fontweight='bold', va='center', transform=ax.get_xaxis_transform())
    ax.set_yticks([])
    if log:
        ax.set_xscale('log')
        ax.minorticks_off()
    if xlim:
        ax.set_xlim(*xlim)
    if xticks:
        ax.set_xticks([t for t, _ in xticks])
        ax.set_xticklabels([s for _, s in xticks])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if note:
        ax.text(1.0, -0.42, note, transform=ax.transAxes, ha='right', fontsize=9, color=C['neutral'])
    sp.save(fig, path)

def steps(stem, rows, **kw):
    for k in range(1, len(rows) + 1):
        bars_lines(out / f'{stem}_step{k}.svg', rows, step=k, **kw)

def picovoice():
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    names = ['PocketSphinx', 'Snowboy', 'Porcupine']
    vals = [_NUMS[f'pv_{n.lower()}'] for n in names]
    cols = [C['neutral'], C['neutral'], C['memory']]
    ax.bar(range(3), vals, width=0.5, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{ {52.0: '52.0', 31.9: '31.9', 2.7: '2.7'}[v]}%", ha='center', va='bottom', fontsize=14, fontweight='bold', color=cols[i])
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylim(0, 65)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel('Wake words missed (%)')
    ax.grid(axis='y', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(2.45, 58, 'All three held to 1 false alarm per 10 h\n= 72 a month; budget: 1 a month', ha='right', va='top', fontsize=12, color=C['danger'], fontweight='bold')
    sp.save(fig, out / 'pv_missrate.svg')
picovoice()
bars_lines(out / 'listen_hours.svg', [('One month of listening', 720.0, 'neutral', '720 h'), ('Needed, zero false wakes', 2160.0, 'data', '2,160 h  = 248 GB at 16 kHz')], log=False, xlim=(0, 4200), size=sp.SHORT, xlabel='Hours of representative non-keyword audio')
steps('label_entries', [('Class label', 1, 'data', '1'), ('Ten boxes', 40.0, 'data', '40'), ('Mask, 1920×1080', 2073600, 'data', '2.1 M', 'in')], xlim=(0.3, 10000000.0), xlabel='Label entries per image (log)', size=(3.6, 2.5), xticks=[(1, '1'), (100, '100'), (10000.0, '10$^4$'), (1000000.0, '10$^6$')])
steps('label_cost', [('Label 1M images', 100000.0, 'data', '\\$100,000   (2 to 4 weeks)'), ('Train ResNet-50 once', 40.03646875, 'compute', '\\$20 to \\$40   (1.3 hours)')], xlim=(5, 20000000.0), size=sp.SHORT, xlabel='Cost (US dollars, log scale)', xticks=[(10, '\\$10'), (1000, '\\$1,000'), (100000.0, '\\$100,000')])
FONT = 'font-family="Helvetica, Arial, sans-serif"'
HEX = {'text': '#1A1A2E', 'grid': '#CCCCCC'}

def svg(path, body, w=900, h=500):
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n<rect width="{w}" height="{h}" fill="white"/>\n' + body + '\n</svg>\n')

def box(x, y, w, h, fill, stroke, text='', size=20, bold=False, color=None, dash=False, rx=8):
    d = ' stroke-dasharray="7 5"' if dash else ''
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2.5"{d}/>'
    if text:
        lines = text.split('\n')
        y0 = y + h / 2 - (len(lines) - 1) * size * 0.6
        for j, ln in enumerate(lines):
            s += txt(x + w / 2, y0 + j * size * 1.2, ln, size, bold, color or HEX['text'])
    return s

def txt(x, y, s, size=20, bold=False, color='#1A1A2E', anchor='middle'):
    b = ' font-weight="bold"' if bold else ''
    return f'<text x="{x}" y="{y}" {FONT} font-size="{size}"{b} fill="{color}" text-anchor="{anchor}" dominant-baseline="central">{s}</text>'

def arrow(x1, y1, x2, y2, color='#666666', w=3):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = (x2 - 14 * math.cos(a), y2 - 14 * math.sin(a))
    p1 = (hx + 7 * math.sin(a), hy - 7 * math.cos(a))
    p2 = (hx - 7 * math.sin(a), hy + 7 * math.cos(a))
    return f'<line x1="{x1}" y1="{y1}" x2="{hx}" y2="{hy}" stroke="{color}" stroke-width="{w}"/><polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>'
SPK = [('A', C['memory']), ('B', C['compute']), ('C', C['network'])]

def clips(x, y, who, size=44, gap=9):
    s = ''
    for j, (name, col) in enumerate(who):
        s += f'<rect x="{x + j * (size + gap)}" y="{y}" width="{size}" height="{size}" rx="6" fill="{col}"/>'
        s += txt(x + j * (size + gap) + size / 2, y + size / 2, name, 24, True, 'white')
    return s

def leak_file():
    A, B, Cc = SPK
    s = txt(20, 18, 'Split by file', 24, True, anchor='start')
    s += txt(205, 52, 'Training split', 22, True)
    s += txt(700, 52, 'Test split', 22, True)
    s += box(20, 72, 370, 70, '#F7F7F7', '#BBBBBB', rx=10)
    s += clips(34, 85, [A, A, B, B, Cc, A])
    s += box(560, 72, 280, 70, '#F7F7F7', '#BBBBBB', rx=10)
    s += clips(574, 85, [A, B, Cc])
    s += txt(475, 95, 'every test', 20, True, C['danger'])
    s += txt(475, 120, 'voice heard', 20, True, C['danger'])
    svg(out / 'leak_file.svg', s, h=155)
leak_file()

def skew(k):
    s = txt(20, 22, 'Training', 26, True, C['data'], 'start')
    s += box(20, 45, 190, 85, C['data_l'], C['data'], 'Clip', 24)
    s += arrow(210, 87, 290, 87)
    s += box(290, 45, 320, 85, C['data_l'], C['data'], 'Normalize with\ntraining mean, std', 23)
    s += arrow(610, 87, 690, 87)
    s += box(690, 45, 190, 85, C['data_l'], C['data'], 'Model learns', 24)
    s += txt(20, 192, 'Serving', 26, True, C['memory'], 'start')
    s += box(20, 215, 190, 85, C['memory_l'], C['memory'], 'Same clip', 24)
    s += arrow(210, 257, 290, 257)
    if k == 1:
        s += box(290, 215, 320, 85, C['danger_l'], C['danger'], "Normalize with\nthis batch's mean, std", 23)
        s += arrow(610, 257, 690, 257)
        s += box(690, 215, 190, 85, C['danger_l'], C['danger'], 'Different\ninput', 24)
        s += txt(450, 350, 'Same code, different state: no error, wrong numbers.', 25, True, C['danger'])
    else:
        s += box(290, 215, 320, 85, C['data_l'], C['data'], 'Normalize with saved\ntraining mean, std', 23)
        s += arrow(610, 257, 690, 257)
        s += box(690, 215, 190, 85, C['data_l'], C['data'], 'Same\ninput', 24)
        s += arrow(450, 130, 450, 215, C['data'])
        s += txt(462, 172, 'persisted with the model', 21, True, C['data'], 'start')
        s += txt(450, 350, 'Fit on the training split once; reuse everywhere.', 25, True, C['data'])
    svg(out / f'skew_step{k}.svg', s, h=380)
for k in (1, 2):
    skew(k)

def pipeline(k):
    s = txt(10, 20, 'Tay, 2016', 25, True, C['danger'], 'start')
    s += box(10, 40, 220, 64, C['danger_l'], C['danger'], 'Public replies', 23)
    s += arrow(230, 72, 650, 72, C['danger'])
    s += txt(440, 54, 'no boundary', 21, True, C['danger'])
    s += box(650, 40, 240, 64, C['danger_l'], C['danger'], 'Model behavior', 23)
    if k >= 2:
        s += txt(10, 150, 'Keyword spotter pipeline', 25, True, C['data'], 'start')
        stages = [('Ingest', 'contracts,\nK-S'), ('Label', 'rubric,\nkappa'), ('Process', 'frozen\nstatistics'), ('Release', 'speaker-hash\nsplits'), ('Deliver', 'sized to the\naccelerator'), ('Monitor', 'drift,\nfalse wakes')]
        for j, (name, gate) in enumerate(stages):
            x = 10 + j * 149
            later = j >= 4
            col = '#999999' if later else C['data']
            s += box(x, 172, 134, 62, '#F7F7F7' if later else C['data_l'], col, name, 23, True, color='#888888' if later else None, dash=later)
            if j < len(stages) - 1:
                s += arrow(x + 134, 203, x + 149, 203, col, 2.5)
            for m, ln in enumerate(gate.split('\n')):
                s += txt(x + 67, 262 + m * 26, ln, 20, False, '#999999' if later else '#333333')
        s += txt(747, 150, 'Part 2', 22, True, '#888888')
        s += txt(450, 340, 'Each stage turns a silent data failure into a loud one.', 24, True, C['text'])
    svg(out / f'pipeline_step{k}.svg', s, h=360)
for k in (1, 2):
    pipeline(k)
_ERR_COLOR = {'image': 'memory', 'text': 'compute', 'audio': 'network'}

def _err_key(name):
    return 'err_' + ''.join((c for c in name.lower() if c.isalnum()))
bars_lines(out / 'label_errors.svg', [('QuickDraw (images)', 10.12, 'memory', '10.12%'), ('CIFAR-100 (images)', 5.85, 'memory', '5.85%'), ('ImageNet val (images)', 5.83, 'memory', '5.83%'), ('IMDB (text)', 2.9, 'compute', '2.90%'), ('AudioSet (audio)', 1.35, 'network', '1.35%'), ('MNIST (images)', 0.15, 'memory', '0.15%')], log=False, xlim=(0, 13), size=(5.8, 2.8), xlabel='Test-set labels found to be wrong (%)')

def cascades():
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch, FancyArrowPatch
    cols = {'blue': C['memory'], 'red': C['danger'], 'green': C['data'], 'orange': '#CC5500', 'violet': '#7030A0'}
    px = [i * 10 / 6 for i in range(7)]
    arcs = [(0, 6, 'red', 60, 120, True), (0, 5, 'red', 60, 125, True), (1, 6, 'blue', 60, 120, True), (1, 6, 'red', 50, 125, True), (4, 6, 'blue', 60, 125, True), (3, 6, 'blue', 60, 120, True), (1, 6, 'orange', 44, 132, False), (1, 6, 'green', 38, 135, False), (1, 5, 'orange', 30, 135, False), (1, 5, 'green', 36, 130, False), (2, 6, 'orange', 40, 135, False), (2, 5, 'orange', 40, 135, False)]
    fig = plt.figure(figsize=(7.0, 2.3))
    ax = fig.add_axes([0.0, 0.22, 0.64, 0.76])
    ax.plot([0, 10], [0, 0], color='#BBBBBB', lw=2.5, zorder=1)
    for a, b, ck, o, i, heavy in arcs:
        x0, x1 = (px[a], px[b])
        d = 0.3915 * (x1 - x0)
        oo, ii = (np.radians(o), np.radians(i))
        verts = [(x0, 0), (x0 + d * np.cos(oo), d * np.sin(oo)), (x1 + d * np.cos(ii), d * np.sin(ii)), (x1, 0)]
        ax.add_patch(PathPatch(MPath(verts, [1, 4, 4, 4]), fill=False, ec=cols[ck], lw=2.6 if heavy else 1.4, zorder=2))
    for k in (5, 6):
        ax.add_patch(FancyArrowPatch((px[k] - 1.15, 1.05), (px[k] - 0.12, 0.18), arrowstyle='simple,head_width=12,head_length=10,tail_width=5', color=cols['violet'], alpha=0.55, lw=0, zorder=3))
    for x0, x1 in ((px[1], 0.8), (px[6], 9.1)):
        rp = MPath([(x0, 0.12), (x0, 2.0), (x0, 2.7), (x1, 2.7)], [1, 2, 3, 3])
        ax.add_patch(FancyArrowPatch(path=rp, arrowstyle='-|>,head_width=4,head_length=7', color=cols['red'], lw=1.6, ls=(0, (4, 3)), zorder=3))
    ax.scatter(px, [0] * 7, s=28, color=C['text'], zorder=4)
    names = ['Problem\nstatement', 'Collect,\nlabel', 'Analyze,\nclean', 'Select\nmodel', 'Train\nmodel', 'Evaluate\nmodel', 'Deploy\nmodel']
    for x, nm in zip(px, names):
        ax.text(x, -0.25, nm, ha='center', va='top', fontsize=10.5, color=C['text'])
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(-1.15, 3.0)
    ax.set_aspect('equal')
    ax.axis('off')
    lg = [('blue', 'Interacting with physical\nworld brittleness'), ('red', 'Inadequate application-\ndomain expertise'), ('green', 'Conflicting reward\nsystems'), ('orange', 'Poor cross-organizational\ndocumentation')]
    y = 0.95
    for ck, t in lg:
        fig.text(0.655, y - 0.035, '●', color=cols[ck], fontsize=13, va='center')
        fig.text(0.683, y, t, fontsize=10.5, va='top', color=C['text'], linespacing=1.1)
        y -= 0.165
    fig.add_artist(FancyArrowPatch((0.655, y - 0.035), (0.693, y - 0.035), transform=fig.transFigure, arrowstyle='simple,head_width=9,head_length=7,tail_width=4', color=cols['violet'], alpha=0.8, lw=0))
    fig.text(0.703, y - 0.035, 'Impact of the cascade', fontsize=10.5, va='center', color=C['text'])
    y -= 0.11
    fig.add_artist(FancyArrowPatch((0.655, y - 0.035), (0.693, y - 0.035), transform=fig.transFigure, arrowstyle='-|>,head_width=3,head_length=5', ls=(0, (3, 2)), color=cols['red'], lw=1.6))
    fig.text(0.703, y - 0.035, 'Abandon or restart', fontsize=10.5, va='center', color=C['text'])
    sp.save(fig, out / 'cascades.svg')
cascades()
print('figures written')

def synthetic_audio():
    import warnings
    from scipy.io import wavfile
    from scipy.signal import spectrogram, fftconvolve
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sr, x = wavfile.read(out / 'artifacts' / 'tts_alexa.wav')
    x = x.astype(float) / 32768.0
    clip = np.zeros(sr)
    start = int(0.15 * sr)
    clip[start:start + len(x)] = x[:sr - start]
    rng = np.random.default_rng(4)
    rt60 = 0.5
    t = np.arange(int(rt60 * sr)) / sr
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    wet = fftconvolve(clip, rir)[:sr]
    wet *= np.max(np.abs(clip)) / np.max(np.abs(wet))

    def spec(sig):
        f, tt, S = spectrogram(sig, fs=sr, nperseg=400, noverlap=240)
        return (f / 1000, tt, 10 * np.log10(S + 1e-10))
    f, tt, S0 = spec(clip)
    _, _, S1 = spec(wet)
    S2 = S1.copy()
    fmask = (f >= 2.0) & (f < 3.2)
    tmask = (tt >= 0.45) & (tt < 0.58)
    floor = S1.max() - 80
    S2[fmask, :] = floor
    S2[:, tmask] = floor
    panels = [('1  Text-to-speech', S0), ('2  + room echo', S1), ('3  + masks', S2)]
    vmax = max(S0.max(), S1.max())
    for k in (1, 2, 3):
        fig, axes = plt.subplots(1, 3, figsize=(6.2, 1.5), sharey=True)
        for i, (ax, (title, S)) in enumerate(zip(axes, panels)):
            ax.set_title(title, fontsize=11, fontweight='bold', color=C['text'], loc='left', alpha=1.0 if i < k else 0.0)
            if i < k:
                ax.pcolormesh(tt, f, S, shading='auto', cmap='magma', vmin=vmax - 80, vmax=vmax, rasterized=True)
                if i == 2:
                    ax.text(0.97, 2.6, 'band', color='white', fontsize=10, ha='right', va='center', fontweight='bold')
                    ax.text(0.515, 6.0, 'span', color='white', fontsize=10, ha='center', va='center', fontweight='bold', rotation=90)
            else:
                ax.set_facecolor('white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 8)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0', '0.5', '1'])
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        axes[0].set_ylabel('kHz')
        axes[1].set_xlabel('seconds', fontsize=11)
        axes[0].set_yticks([0, 4, 8])
        fig.subplots_adjust(wspace=0.16)
        sp.save(fig, out / f'synthetic_audio_step{k}.svg')
synthetic_audio()

def forced_alignment():
    G, GL, O, OL = (C['data'], C['data_l'], C['compute'], C['compute_l'])
    heads = ['Sentence audio\n+ transcript', 'Forced alignment\nfinds word edges', 'Cut 1 s clips\nfor each keyword', 'Train the\nkeyword model']
    s = ''
    xs = [10, 237, 464, 691]
    for j, (x, h) in enumerate(zip(xs, heads)):
        s += box(x, 8, 200, 74, GL, G, h, 21, True)
        if j < 3:
            s += arrow(x + 200, 45, x + 227, 45, '#666666', 3)
    s += box(10, 112, 200, 150, '#F7F7F7', '#999999', '', rx=8)
    s += txt(110, 150, '“he gazed up', 22, False, C['text'])
    s += txt(110, 182, 'the steep bank”', 22, False, C['text'])
    s += txt(110, 228, 'words, no times', 20, True, '#666666')
    s += box(237, 112, 200, 150, '#F7F7F7', '#999999', '', rx=8)
    rows = [['he', 'gazed', 'up'], ['the', 'steep', 'bank']]
    for r, words in enumerate(rows):
        x = 246
        for w in words:
            wd = 12 + 11 * len(w)
            hl = w == 'bank'
            s += box(x, 126 + r * 46, wd, 36, OL if hl else 'white', O if hl else '#999999', w, 19, hl, rx=4)
            x += wd + 5
    s += txt(337, 228, 'edges found', 20, True, '#666666')
    s += box(464, 112, 200, 150, '#F7F7F7', '#999999', '', rx=8)
    s += box(504, 132, 120, 50, OL, O, 'bank', 24, True, rx=4)
    s += txt(564, 204, 'one 1 s clip', 20, True, '#666666')
    s += txt(564, 232, 'labeled “bank”', 20, True, '#666666')
    s += box(691, 112, 200, 150, GL, G, '', rx=8)
    for m, ln in enumerate(['23.4M clips', '340K keywords', '50 languages']):
        s += txt(791, 146 + m * 40, ln, 23, True, C['text'])
    svg(out / 'forced_alignment.svg', s, w=900, h=270)
forced_alignment()
