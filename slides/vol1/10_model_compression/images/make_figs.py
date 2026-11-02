#!/usr/bin/env python3
"""Figures for Lecture 10 (Model Compression, Part 1).

Every number comes from MLSysIM through calcs.py (imported as K): val(key) is
the string a slide prints and num(key) its value; bar lengths read the
Scenarios.* quantities. Where each figure's inputs live:
  - Memory ladder: Models (GPT-3, Llama 2 7B, DS-CNN) and Platforms (phone
    RAM, microcontroller SRAM); Scenarios.Llama2_7BTokenOnFlagshipPhone.
  - Weight streaming: Scenarios.*WeightReadPhone (flagship phone LPDDR bus);
    the camera frame from Scenarios.MobileNetV2BlurPipelinePhone.
  - Pruning grids (6 by 6, random, seed 10) and the 2:4 block [0, 1.4, 0,
    -2.1] are illustrations; their counts are read off the drawn matrices.
  - Listing mask and 4 by 4 product: Scenarios.MagnitudePruned3x3 and
    SparseMatVec4x4; sparse-format bytes and break-evens: physics.memory and
    Scenarios.WearableMobileNetV2SparseFlash.
  - 2:4 GEMM: slides/vol1/10_model_compression/images/data/mishra2021_int8_sparse_gemm.csv (values read from the
    published plot, approximate) and Literature.CompressionResults.
  - Channel tiles: Scenarios.Channels256To250Tile8 and Channels256To248Tile8;
    width 0.75: Models MobileNetV2 and MobileNetV2_Alpha0_75.
  - DistilBERT: Literature.CompressionResults (Sanh et al. 2019); soft
    targets: Scenarios.TeacherSoftTargetsT4.
  - Fusion: Scenarios.ResNet50Conv256FusionH100.
  - Waterfall, paper against realized, Amdahl: Scenarios.MobileNetV2Blur-
    PipelinePhone, PrunedInt8PaperVsMeasured, ModelFifthOfRequestLatency.
  - Break-even axis: CSR and COO from physics.memory; the 80 to 90 percent
    band for unstructured kernels is the chapter's (Sec. 10.4.2), unsourced.
  - MLP share: physics.transformer.calc_transformer_block_weights.
  - Smallest capable model: Epoch AI data
    (vol1/10_model_compression/data/mmlu_dense_models.csv).
  - Frontier: slides/vol1/10_model_compression/images/data/mobilenetv3_table3_pixel1.csv (Howard et al. 2019).
  - Iterative pruning: slides/vol1/10_model_compression/images/data/han2015_alexnet_pruning.csv (Han et al. 2015).
  - Deep Compression cliffs: slides/vol1/10_model_compression/images/data/han2016_compression_methods.csv.
  - Measured CPU bars: cpu_measure.json (measure_cpu.py, run for this deck).
  - hook_step1.svg is copied from ../10b_model_compression/images.

Run from this folder with the MLSysBook venv (it imports mlsysim):
    python3 make_figs.py
"""
from types import SimpleNamespace
from mlsysim.core.series_loader import load_series
import mlsysim
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from mlsysim.core.series_loader import load_series
plt = sp.plt
C = sp.C
out = Path(__file__).parent
FONT = 'Helvetica, Arial, sans-serif'

def bars(path, rows, shown, log=False, xlim=None, xlabel=None, size=sp.SIZE, vlines=(), xticks=None, note=None, label_size=13, boxed=False, text_size=12, axis_size=None):
    """axis_size: tick, axis-label, and reference-line label size; raise it
    with label_size and text_size when the plot is drawn for a column slot."""
    fig, ax = plt.subplots(figsize=size)
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        vis = shown[i]
        ax.barh(y, r['value'], color=C[r['color']], height=0.62, alpha=1.0 if vis else 0.0)
        ax.text(-0.01, y, r['label'], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=label_size, color=C['text'], fontweight='bold', alpha=1.0 if vis else 0.0)
        x_end = r['value']
        over_vis = vis and r.get('over_on', False)
        if 'over' in r:
            ax.barh(y, r['over'], color=C[r['over_color']], height=0.34, alpha=1.0 if over_vis else 0.0, zorder=3)
        tx = x_end * 1.25 if log else x_end + (xlim[1] if xlim else 1) * 0.012
        text = r['text']
        if over_vis and 'over_text' in r:
            text = r['over_text']
        bb = dict(facecolor='white', edgecolor='none', pad=1.0, alpha=1.0 if vis else 0.0) if boxed else None
        ax.text(tx, y, text, va='center', fontsize=text_size, bbox=bb, zorder=4, color=C[r.get('over_color', r['color']) if over_vis else r['color']], fontweight='bold', alpha=1.0 if vis else 0.0)
        if 'over_text' in r:
            ax.text(tx, y, r['over_text'], va='center', fontsize=text_size, alpha=0.0)
    for x, label, ck in vlines:
        ax.axvline(x, color=C[ck], lw=1.5, ls='--', zorder=1)
        ax.text(x, n - 0.45, ' ' + label, color=C[ck], fontsize=axis_size or 10, fontweight='bold', va='bottom', ha='left')
    if log:
        ax.set_xscale('log')
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    if xticks:
        ax.set_xticks([t for t, _ in xticks])
        ax.set_xticklabels([s for _, s in xticks])
        ax.minorticks_off()
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=axis_size)
    if axis_size:
        ax.tick_params(labelsize=axis_size)
    if note:
        fig.text(0.99, 0.01, note, ha='right', va='bottom', fontsize=9, color=C['neutral'])
    sp.save(fig, path)

def svg(path, w, h, body):
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">\n<rect width="{w}" height="{h}" fill="white"/>\n{body}</svg>\n')

def text(x, y, s, size=22, color=C['text'], weight='normal', anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>\n'

def box(x, y, w, h, fill, stroke, sw=2, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>\n'

def arrow(x1, y1, x2, y2, color, sw=3):
    ang = np.arctan2(y2 - y1, x2 - x1)
    hl, hw = (12, 7)
    bx, by = (x2 - hl * np.cos(ang), y2 - hl * np.sin(ang))
    px, py = (-np.sin(ang) * hw, np.cos(ang) * hw)
    return f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{sw}"/>\n<polygon points="{x2},{y2} {bx + px:.1f},{by + py:.1f} {bx - px:.1f},{by - py:.1f}" fill="{color}"/>\n'
ladder_rows = [dict(label='175B, FP16', value=350000000000.0, color='memory', text='350 GB'), dict(label='7B, FP16', value=14000000000.0, color='danger', text='14 GB', over=3500000000.0, over_color='data', over_text='INT4: 3.5 GB'), dict(label='Phone RAM', value=8589934592, color='neutral', text='8 GB'), dict(label='MCU SRAM', value=524288, color='neutral', text='512 KB', over=200000.0, over_color='data', over_text='spotter: 200 KB')]
ladder_kw = dict(log=True, xlim=(50000.0, 300000000000000.0), size=(4.2, 2.9), xlabel='Bytes (log scale)', xticks=[(1000000.0, '1 MB'), (1000000000.0, '1 GB'), (1000000000000.0, '1 TB')])
bars(out / 'ladder_step1.svg', ladder_rows, [1, 0, 1, 1], **ladder_kw)
bars(out / 'ladder_step2.svg', ladder_rows, [1, 1, 1, 1], **ladder_kw)
stream_rows = [dict(label='MobileNetV2, FP32', value=0.273818125, color='memory', text='0.27 ms: 1% of a frame'), dict(label='ResNet-50, FP32', value=1.996643125, color='memory', text='2 ms'), dict(label='GPT-2 XL, FP16', value=58.59375, color='danger', text='59 ms per token'), dict(label='GPT-2 XL, INT4', value=14.6484375, color='data', text='15 ms per token')]
for k in range(1, 5):
    bars(out / f'stream_step{k}.svg', stream_rows, [i < k for i in range(4)], xlim=(0, 80), vlines=[(33.33333333, '33 ms camera frame', 'compute')], xlabel='Time to read every weight once at 51.2 GB/s (ms)')
tiers = [('What the model computes', 'pruning, distillation', 'FLOPs/sample, weight bytes', C['compute'], C['compute_l']), ('How values are stored', 'quantization', 'bytes moved, energy/op', C['memory'], C['memory_l']), ('How it executes', 'fusion, sparse kernels', 'bytes, overhead, efficiency', C['network'], C['network_l'])]
for k in range(1, 4):
    body = text(10, 34, 'Software', 24, C['neutral'], 'bold')
    body += arrow(55, 50, 55, 432, C['neutral'], 3)
    body += text(10, 466, 'Hardware', 24, C['neutral'], 'bold')
    body += text(560, 34, 'Terms it moves', 24, C['neutral'], 'bold')
    for i, (name, how, term, stroke, fill) in enumerate(tiers):
        y = 52 + i * 136
        op = '1' if i < k else '0'
        body += f'<g opacity="{op}">\n'
        body += box(90, y, 420, 116, fill, stroke, 3)
        body += text(300, y + 50, name, 30, C['text'], 'bold', 'middle')
        body += text(300, y + 92, how, 27, stroke, 'normal', 'middle')
        body += arrow(516, y + 58, 552, y + 58, stroke, 3)
        body += text(560, y + 50, term.split(', ', 1)[0] + ',', 27, stroke, 'bold')
        body += text(560, y + 86, term.split(', ', 1)[1], 27, stroke, 'bold')
        body += '</g>\n'
    svg(out / f'tiers_step{k}.svg', 900, 480, body)
rng = np.random.default_rng(10)
W = rng.normal(0, 1, (6, 6)) * rng.choice([0.3, 1.0], size=(6, 6), p=[0.5, 0.5])
mag = np.abs(W)
thresh = np.median(mag)
keep_cols = np.sort(np.argsort(mag.sum(axis=0))[3:])

def grid(ax, x0, mat, mask=None, cols=None):
    nr, nc = mat.shape
    vmax = mag.max()
    for i in range(nr):
        for j in range(nc):
            keep = True if mask is None else mask[i, j]
            shade = 0.15 + 0.85 * abs(mat[i, j]) / vmax
            col = plt.cm.Oranges(shade) if keep else 'white'
            ax.add_patch(plt.Rectangle((x0 + j, nr - 1 - i), 1, 1, facecolor=col, edgecolor='#999999', lw=0.8))
            if not keep:
                ax.text(x0 + j + 0.5, nr - 0.5 - i, '0', ha='center', va='center', fontsize=10, color='#999999')
for k in range(1, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    grid(ax, 0, W)
    ax.text(3, -0.5, f'Dense:\n{W.size} weights', ha='center', va='top', fontsize=12, fontweight='bold', color=C['text'])
    if k >= 2:
        grid(ax, 11, W, mask=mag >= thresh)
        ax.text(14, -0.5, f'Unstructured:\n{int((mag < thresh).sum())} zeros', ha='center', va='top', fontsize=12, fontweight='bold', color=C['danger'])
    if k >= 3:
        grid(ax, 22, W[:, keep_cols])
        ax.text(23.5, -0.5, f'Structured:\n{W.shape[0]}×{len(keep_cols)}, dense', ha='center', va='top', fontsize=12, fontweight='bold', color=C['data'])
    ax.set_xlim(-2, 27)
    ax.set_ylim(-2.2, 6.2)
    ax.set_aspect('equal')
    ax.axis('off')
    sp.save(fig, out / f'prune_step{k}.svg')
fmt_rows = [dict(label='Dense FP32', value=4.0, color='neutral', text='4 B'), dict(label='+ 16-bit index', value=6.0, color='danger', text='6 B: 2 MB → 3 MB'), dict(label='CSR', value=8.0, color='memory', text='8 B: wins above 50% zeros'), dict(label='COO', value=12.0, color='memory', text='12 B: above 66.7%')]
bars(out / 'sparsefmt_step1.svg', fmt_rows, [1, 1, 0, 0], xlim=(0, 30), xlabel='Bytes stored per nonzero weight', size=(4.4, 2.3), label_size=14, text_size=14, axis_size=13)
bars(out / 'sparsefmt_step2.svg', fmt_rows, [1, 1, 1, 1], xlim=(0, 30), xlabel='Bytes stored per nonzero weight', size=(4.4, 2.3), label_size=14, text_size=14, axis_size=13)
cell = 78
body = text(10, 34, 'Every group of 4 keeps 2', 26, C['text'], 'bold')
vals = ['0', '1.4', '0', '−2.1']
for j, v in enumerate(vals):
    keep = v != '0'
    body += box(10 + j * cell, 60, cell, cell, C['compute_l'] if keep else 'white', C['compute'] if keep else '#999999', 2, 0)
    body += text(10 + j * cell + cell / 2, 60 + cell / 2 + 9, v, 26, C['text'] if keep else '#999999', 'bold' if keep else 'normal', 'middle')
body += arrow(170, 150, 170, 205, C['neutral'], 3)
for j, v in enumerate(['1.4', '−2.1']):
    body += box(10 + j * cell, 215, cell, cell, C['compute_l'], C['compute'], 2, 0)
    body += text(10 + j * cell + cell / 2, 215 + cell / 2 + 9, v, 26, C['text'], 'bold', 'middle')
for j, v in enumerate(['01', '11']):
    body += box(200 + j * 66, 228, 60, 52, C['neutral_l'], C['neutral'], 2, 0)
    body += text(200 + j * 66 + 30, 263, v, 24, C['text'], 'normal', 'middle')
body += text(10, 322, '2 values', 22, C['compute'], 'bold')
body += text(200, 322, '2-bit positions', 22, C['neutral'], 'bold')
body += text(10, 385, 'Design peak: 2× math throughput', 24, C['text'], 'bold')
body += text(10, 418, '(vendor specification)', 22, C['neutral'])
svg(out / 'sparse24.svg', 420, 440, body)
gemm_rows = [{'label': 'K = 1,280', 'value': 1.2, 'color': 'compute', 'text': '≈1.2×'}, {'label': 'K = 2,560', 'value': 1.5, 'color': 'compute', 'text': '≈1.5×'}, {'label': 'K = 12,800', 'value': 1.9, 'color': 'compute', 'text': '≈1.9×'}]
fig, ax = plt.subplots(figsize=(4.6, 2.4))
for i, r in enumerate(gemm_rows):
    y = len(gemm_rows) - 1 - i
    ax.barh(y, r['value'], color=C['compute'], height=0.62, zorder=2)
    ax.text(r['value'] - 0.04, y, r['text'], ha='right', va='center', fontsize=13, color='white', fontweight='bold', zorder=3)
    ax.text(-0.02, y, r['label'], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
ax.axvline(1.0, color=C['neutral'], lw=1.5, ls='--', zorder=1)
ax.axvline(2.0, color=C['danger'], lw=1.5, ls='--')
ax.text(1.0, 2.55, 'dense ', color=C['neutral'], fontsize=10, fontweight='bold', ha='right')
ax.text(2.0, 2.55, ' 2× design peak', color=C['danger'], fontsize=10, fontweight='bold')
ax.set_xlim(0, 2.6)
ax.set_ylim(-0.6, 2.9)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.grid(axis='x', color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('Measured 2:4 speedup over dense\n(A100, INT8 GEMM, M = N = 10,240)')
sp.save(fig, out / 'gemm24.svg')
body = ''
rows = [(f'{256} → {250} channels', SimpleNamespace(channels=256, full_tiles=31, last_tile_lanes=2, name='A 256-channel layer pruned to 250 on a matrix unit with tiles of 8', padded_channels=256, pruned_channels=250, remainder=2, tile=8, tiles=32, utilization=0.9765625, work_removed=0.0234375), C['danger'], f'last tile: {2} of {8} lanes busy'), (f'{256} → {248} channels', SimpleNamespace(channels=256, full_tiles=31, last_tile_lanes=8, name='A 256-channel layer pruned to 248 on a matrix unit with tiles of 8', padded_channels=248, pruned_channels=248, remainder=0, tile=8, tiles=31, utilization=1.0, work_removed=0.03125), C['data'], 'every tile full')]
for r, (lab, tl, color, note) in enumerate([('256 → 250 channels', SimpleNamespace(channels=256, full_tiles=31, last_tile_lanes=2, name='A 256-channel layer pruned to 250 on a matrix unit with tiles of 8', padded_channels=256, pruned_channels=250, remainder=2, tile=8, tiles=32, utilization=0.9765625, work_removed=0.0234375), '#CB202D', 'last tile: 2 of 8 lanes busy'), ('256 → 248 channels', SimpleNamespace(channels=256, full_tiles=31, last_tile_lanes=8, name='A 256-channel layer pruned to 248 on a matrix unit with tiles of 8', padded_channels=248, pruned_channels=248, remainder=0, tile=8, tiles=31, utilization=1.0, work_removed=0.03125), '#008F45', 'every tile full')]):
    y = 10 + r * 215
    body += text(10, y + 30, lab, 30, C['text'], 'bold')
    for t in range(3):
        x0 = 10 + t * 180
        fill_n = {'last tile: 2 of 8 lanes busy': 8, 'every tile full': 8}[note] if t < 2 else {'last tile: 2 of 8 lanes busy': 2, 'every tile full': 8}[note]
        for c in range({'last tile: 2 of 8 lanes busy': 8, 'every tile full': 8}[note]):
            on = c < fill_n
            body += box(x0 + c * 20, y + 50, 18, 60, C['compute_l'] if on else 'white', C['compute'] if on else '#BBBBBB', 1.5, 0, None if on else '4,3')
        body += text(x0 + 80, y + 142, f"tile { {'last tile: 2 of 8 lanes busy': 32, 'every tile full': 31}[note] - 2 + t}", 24, C['neutral'], 'normal', 'middle')
    body += text(10, y + 186, note, 28, color, 'bold')
svg(out / 'tiles.svg', 560, 430, body)
fig, ax = plt.subplots(figsize=(4.4, 2.6))
groups = [('Parameters', 3.5, 2.6, 'M', '74%'), ('MFLOP', 600.0, 418.0, '', '70%')]
for gi, (name, full, slim, unit, pct) in enumerate(groups):
    y = 1 - gi
    ax.barh(y + 0.2, 1.0, height=0.36, color=C['neutral'])
    ax.barh(y - 0.2, slim / full, height=0.36, color=C['compute'])
    ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
    ax.text(1.02, y + 0.2, f'{1.0:.1f}: {full:g}{unit}', va='center', fontsize=12, color=C['neutral'], fontweight='bold')
    ax.text(slim / full + 0.02, y - 0.2, f'0.75: {slim:g}{unit} ({pct})', va='center', fontsize=12, color=C['compute'], fontweight='bold')
ax.text(0.0, 1.62, 'MobileNetV2 width multiplier', fontsize=12, color=C['text'])
ax.set_xlim(0, 1.75)
ax.set_ylim(-0.6, 1.8)
ax.set_yticks([])
ax.set_xticks([])
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
sp.save(fig, out / 'width075.svg')
distil_rows = [dict(label='Parameters', value=60.0, color='data', text='60% (40% fewer)'), dict(label='Inference speed', value=160.0, color='compute', text='160% (1.6× faster)'), dict(label='Language score', value=97.0, color='memory', text='97%')]
bars(out / 'distilbert.svg', distil_rows, [1, 1, 1], xlim=(0, 260), size=(4.4, 2.0), xticks=[(0, '0'), (100, '100 = BERT'), (200, '200')], label_size=15, text_size=15, axis_size=14, xlabel='DistilBERT relative to BERT (%)')
soft1 = [85, 10, 5]
soft4 = [48, 28, 24]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    cls = ['Cat', 'Dog', 'Fox']
    hard = [100 if i == 0 else 0 for i in range(len(cls))]
    xs = np.arange(3)
    wbar = 0.26
    series = [(hard, 'neutral', -wbar, 1.0), (soft1, 'compute', 0.0, 1.0 if k >= 2 else 0.0), (soft4, 'data', wbar, 1.0 if k >= 3 else 0.0)]
    for vals, ck, dx, a in series:
        ax.bar(xs + dx, vals, width=wbar * 0.95, color=C[ck], alpha=a)
        for x, v in zip(xs, vals):
            ax.text(x + dx, v + 2, f'{v}', ha='center', fontsize=11, color=C[ck], fontweight='bold', alpha=a)
    ax.text(2.45, 94, 'label', ha='right', fontsize=13, color=C['neutral'], fontweight='bold')
    ax.text(2.45, 81, 'teacher, T = 1', ha='right', fontsize=13, color=C['compute'], fontweight='bold', alpha=1.0 if k >= 2 else 0.0)
    ax.text(2.45, 68, 'teacher, T = 4', ha='right', fontsize=13, color=C['data'], fontweight='bold', alpha=1.0 if k >= 3 else 0.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(cls, fontsize=13)
    ax.set_ylim(0, 112)
    ax.set_ylabel('Probability (%)')
    sp.save(fig, out / f'kd_step{k}.svg')
for k in (1, 2):
    body = text(10, 34, 'Unfused: 6 transfers, 7.2 MB', 30, C['danger'], 'bold')
    for i, name in enumerate(['Conv', 'BN', 'ReLU']):
        x = 30 + i * 240
        body += box(x, 52, 190, 74, C['compute_l'], C['compute'], 2.5)
        body += text(x + 95, 100, name, 30, C['text'], 'bold', 'middle')
        body += arrow(x + 65, 196, x + 65, 132, C['memory'], 3.5)
        body += arrow(x + 125, 132, x + 125, 196, C['memory'], 3.5)
    body += box(10, 200, 740, 58, C['memory'], C['memory'], 0, 4)
    body += text(380, 240, 'HBM (main memory)', 30, 'white', 'bold', 'middle')
    body += f'<g opacity="{(1 if k == 2 else 0)}">\n'
    body += arrow(95, 262, 95, 316, C['memory'], 3.5)
    body += arrow(665, 316, 665, 262, C['memory'], 3.5)
    body += box(30, 320, 700, 74, C['compute_l'], C['compute'], 2.5)
    body += text(380, 367, 'Conv + BN + ReLU, BN folded in', 30, C['text'], 'bold', 'middle')
    body += text(10, 440, 'Fused: 2 transfers, 4 MB (44.8% less)', 30, C['data'], 'bold')
    body += '</g>\n'
    svg(out / f'fusion_step{k}.svg', 760, 455, body)
stages = ['FP32\nCPU', 'Pruned\nCPU', 'Pruned INT8\naccelerator']
memv = [14.0, 10.4, 2.6]
timev = [125.0, 87.0, 20.0]
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.3))
    for ax, vals, ck, unit, ttl in ((a1, memv, 'memory', 'MB', 'Weight memory (MB)'), (a2, timev, 'compute', 'ms', 'Frame time (ms)')):
        for i, v in enumerate(vals):
            a = 1.0 if i < k else 0.0
            ax.bar(i, v, color=C[ck], width=0.7, alpha=a)
            inside = ck == 'compute' and v < 40
            ax.text(i, v / 2 if inside else v + max(vals) * 0.03, f'{v:g} {unit}', ha='center', va='center' if inside else 'bottom', fontsize=10 if inside else 11, fontweight='bold', color='white' if inside else C[ck], alpha=a)
        ax.set_xticks(range(3))
        ax.set_xticklabels(stages, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(ttl, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)
    a2.axhline(33.0, color=C['danger'], lw=1.5, ls='--')
    a2.text(2.45, 33.0, ' 33 ms\n budget', color=C['danger'], fontsize=11, fontweight='bold', ha='left', va='center', clip_on=False)
    fig.text(0.01, -0.02, 'Illustrative scenario (Example 10.1)', fontsize=9, color=C['neutral'])
    fig.tight_layout(w_pad=1.5)
    sp.save(fig, out / f'waterfall_step{k}.svg')
sp.hbars(out / 'paper_real.svg', [('Paper: 2× × 4×', 8.0, 'neutral', '8×'), ('Realized', 1.5, 'danger', '1.5×  (18.8% of paper)')], '×', log=False, size=(5.8, 1.9), xlabel='Speedup (illustrative)')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.0))
    model_pct, rest_pct = (20.0, 80.0)
    m = model_pct if k == 1 else 0
    cap = '80%: 1 / 0.8 = 1.25×'
    ax.barh(0, m, color=C['compute'], height=0.6)
    ax.barh(0, rest_pct, left=m, color=C['neutral'], height=0.6)
    if k == 1:
        ax.text(m / 2, 0, 'Model\n20%', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax.text(m + rest_pct / 2, 0, 'Decode, resize, display: 80%', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    ax.text(102, 0, '100%' if k == 1 else cap, va='center', fontsize=12, fontweight='bold', color=C['text'] if k == 1 else C['danger'])
    ax.text(102, 0, cap, va='center', fontsize=12, alpha=0.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Share of the original frame latency (%)')
    ax.spines['left'].set_visible(False)
    sp.save(fig, out / f'amdahl_step{k}.svg')
L1 = [[0.8, 0.1, -0.7], [0.05, -0.9, 0.03], [-0.6, 0.02, 0.4]]
CELL = 84
for k in (1, 2):
    body = text(10, 30, 'Dense: 9 weights', 26, C['text'], 'bold')
    for i in range(3):
        for j in range(3):
            v = L1[i][j]
            body += box(10 + j * CELL, 50 + i * CELL, CELL, CELL, C['compute_l'], C['compute'], 2, 0)
            body += text(10 + j * CELL + CELL / 2, 50 + i * CELL + CELL / 2 + 9, f'{v:g}'.replace('-', '−'), 26, C['text'], 'bold', 'middle')
    op = 1 if k == 2 else 0
    body += f'<g opacity="{op}">\n'
    body += arrow(285, 176, 375, 176, C['neutral'], 3)
    body += text(330, 160, '|w| ≥ 0.5', 22, C['neutral'], 'bold', 'middle')
    body += text(390, 30, 'Pruned: 4 of 9 kept', 26, C['data'], 'bold')
    for i in range(3):
        for j in range(3):
            v = L1[i][j]
            keep = abs(v) >= 0.5
            body += box(390 + j * CELL, 50 + i * CELL, CELL, CELL, C['data_l'] if keep else 'white', C['data'] if keep else '#999999', 2, 0)
            body += text(390 + j * CELL + CELL / 2, 50 + i * CELL + CELL / 2 + 9, f'{v:g}'.replace('-', '−') if keep else '0', 26, C['text'] if keep else '#999999', 'bold' if keep else 'normal', 'middle')
    body += '</g>\n'
    svg(out / f'mask_step{k}.svg', 660, 310, body)
M4 = [[2.0, 0.0, 0.0, 1.0], [0.0, 3.0, 0.0, 0.0], [4.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 6.0]]
CELL = 74
for k in (1, 2):
    body = ''
    for i in range(4):
        for j in range(4):
            v = M4[i][j]
            live = k == 1 or v != 0
            body += box(20 + j * CELL, 20 + i * CELL, CELL, CELL, C['compute_l'] if live else 'white', C['compute'] if live else '#BBBBBB', 2, 0)
            body += text(20 + j * CELL + CELL / 2, 20 + i * CELL + CELL / 2 + 6, f'{v:g}', 26, C['text'] if live else '#AAAAAA', 'bold' if live else 'normal', 'middle')
            if k == 2 and v != 0:
                body += text(20 + j * CELL + CELL - 5, 20 + i * CELL + 22, f'c{j}', 19, C['memory'], 'bold', 'end')
    body += text(340, 175, '×', 40, C['text'], 'bold', 'middle')
    for i in range(4):
        body += box(370, 20 + i * CELL, CELL, CELL, C['memory_l'], C['memory'], 2, 0)
        body += text(370 + CELL / 2, 20 + i * CELL + CELL / 2 + 8, f'x{i + 1}', 24, C['text'], 'bold', 'middle')
    msg = 'Dense: 16 multiply-accumulates' if k == 1 else 'Sparse: 6 products + 6 column indices'
    body += text(20, 360, msg, 26, C['compute'] if k == 1 else C['memory'], 'bold')
    svg(out / f'spmv_step{k}.svg', 600, 380, body)
fig, ax = plt.subplots(figsize=(5.8, 2.6))
ax.axvspan(80, 90, color=C['danger_l'])
ax.text(85, 1.85, 'unstructured kernel\ncan lose to dense\nuntil 80 to 90%', ha='center', va='center', fontsize=12, fontweight='bold', color=C['danger'])
for x, lab, ck, y in ((50.0, 'CSR bytes: 50%', 'memory', 1.05), (66.7, 'COO bytes: 66.7%', 'memory', 0.5), (50, '2:4 pattern: exactly 50%,\nhardware path', 'data', -1.05)):
    ax.plot([x], [0], 'o', ms=10, color=C[ck], zorder=5)
    ax.plot([x, x], [0, y - 0.2 if y > 0 else y + 0.35], color=C[ck], lw=1.2)
    ax.text(x, y, lab, ha='center', va='center', fontsize=12, fontweight='bold', color=C[ck])
ax.axhline(0, color=C['text'], lw=1.5)
ax.set_xlim(0, 100)
ax.set_ylim(-1.6, 2.6)
ax.set_yticks([])
for side in ('left', 'bottom'):
    ax.spines[side].set_visible(False)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.tick_params(axis='x', length=0, pad=-6)
ax.set_xlabel('Share of weights that are zero')
sp.save(fig, out / 'breakeven.svg')
mlp_rows = [dict(label='MLP block', value=8.0, color='compute', text='8d²: about two thirds'), dict(label='Q, K, V, O projections', value=4.0, color='memory', text='4d²: just under a third'), dict(label='LayerNorm', value=0.05, color='neutral', text='almost none')]
bars(out / 'mlpshare.svg', mlp_rows, [1, 1, 1], xlim=(0, 14), size=(5.0, 1.9), xticks=[(0, '0'), (4.0, '4d²'), (8.0, '8d²')], xlabel='Multiply-accumulates per token, per layer')
import datetime as dt
from mlsysim.core.units import SEC_PER_DAY, SEC_PER_YEAR_LEAP
_rows, _front, _bar = ([{'model': 'GPT-J-6B', 'date': '2021-08-05', 'parameters': 6000000000.0, 'mmlu': 0.251, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'GPT-J-6B', 'date': '2021-08-05', 'parameters': 6000000000.0, 'mmlu': 0.257, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Gopher 280B', 'date': '2021-12-08', 'parameters': 280000000000.0, 'mmlu': 0.6, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'GPT-3 175B', 'date': '2022-01-27', 'parameters': 175000000000.0, 'mmlu': 0.439, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'Chinchilla 70B', 'date': '2022-03-29', 'parameters': 70000000000.0, 'mmlu': 0.675, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'PaLM 62B', 'date': '2022-04-04', 'parameters': 62000000000.0, 'mmlu': 0.537, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'PaLM 540B', 'date': '2022-04-04', 'parameters': 540000000000.0, 'mmlu': 0.693, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'OPT-13B', 'date': '2022-05-11', 'parameters': 13000000000.0, 'mmlu': 0.251, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'OPT-13B', 'date': '2022-05-11', 'parameters': 13000000000.0, 'mmlu': 0.244, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'LLAMA 1 7B', 'date': '2023-02-24', 'parameters': 7000000000.0, 'mmlu': 0.351, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'LLAMA 1 13B', 'date': '2023-02-24', 'parameters': 13000000000.0, 'mmlu': 0.469, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'LLAMA 1 33B', 'date': '2023-02-24', 'parameters': 33000000000.0, 'mmlu': 0.578, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'LLaMA 1 33B    ', 'date': '2023-02-24', 'parameters': 33000000000.0, 'mmlu': 0.568, 'shots': 5.0, 'source': 'Mixtral of Experts'}, {'model': 'LLAMA 1 65B', 'date': '2023-02-24', 'parameters': 65000000000.0, 'mmlu': 0.634, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'LLaMA 7B', 'date': '2023-02-24', 'parameters': 7000000000.0, 'mmlu': 0.356, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'LLaMA 7B', 'date': '2023-02-24', 'parameters': 7000000000.0, 'mmlu': 0.351, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'LLaMA 13B', 'date': '2023-02-24', 'parameters': 13000000000.0, 'mmlu': 0.477, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'LLaMA 13B', 'date': '2023-02-24', 'parameters': 13000000000.0, 'mmlu': 0.469, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'LLaMA 33B', 'date': '2023-02-24', 'parameters': 33000000000.0, 'mmlu': 0.587, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'LLaMA 33B', 'date': '2023-02-24', 'parameters': 33000000000.0, 'mmlu': 0.578, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'LLaMA 65B', 'date': '2023-02-24', 'parameters': 65000000000.0, 'mmlu': 0.634, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'LLaMA-7B', 'date': '2023-02-24', 'parameters': 7000000000.0, 'mmlu': 0.351, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'LLaMA-7B', 'date': '2023-02-24', 'parameters': 7000000000.0, 'mmlu': 0.351, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'LLaMA-7B', 'date': '2023-02-24', 'parameters': 7000000000.0, 'mmlu': 0.32, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Llama-7B', 'date': '2023-02-24', 'parameters': 7000000000.0, 'mmlu': 0.352, 'shots': 2.0, 'source': 'Textbooks Are All You Need II: phi-1.5 technical report'}, {'model': 'LLaMA-13B', 'date': '2023-02-24', 'parameters': 13000000000.0, 'mmlu': 0.463, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'Falcon 40B', 'date': '2023-03-15', 'parameters': 40000000000.0, 'mmlu': 0.569, 'shots': 5.0, 'source': 'Falcon2-11B Technical ReportFalcon2-11B Technical Report'}, {'model': 'Falcon 40B', 'date': '2023-03-15', 'parameters': 40000000000.0, 'mmlu': 0.554, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'Falcon 40B', 'date': '2023-03-15', 'parameters': 40000000000.0, 'mmlu': 0.554, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'Cerebras-GPT-13B', 'date': '2023-03-20', 'parameters': 13000000000.0, 'mmlu': 0.262, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Cerebras-GPT-13B', 'date': '2023-03-20', 'parameters': 13000000000.0, 'mmlu': 0.246, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Dolly-v2-12B', 'date': '2023-04-11', 'parameters': 12000000000.0, 'mmlu': 0.262, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Dolly-v2-12B', 'date': '2023-04-11', 'parameters': 12000000000.0, 'mmlu': 0.254, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Falcon 7B', 'date': '2023-04-24', 'parameters': 7000000000.0, 'mmlu': 0.35, 'shots': 5.0, 'source': 'Falcon2-11B Technical Report'}, {'model': 'Falcon 7B', 'date': '2023-04-24', 'parameters': 7000000000.0, 'mmlu': 0.262, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'Falcon 7B', 'date': '2023-04-24', 'parameters': 7000000000.0, 'mmlu': 0.262, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'Falcon-7B', 'date': '2023-04-24', 'parameters': 7000000000.0, 'mmlu': 0.2603, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'Falcon-7B', 'date': '2023-04-24', 'parameters': 7000000000.0, 'mmlu': 0.269, 'shots': 2.0, 'source': 'Textbooks Are All You Need II: phi-1.5 technical report'}, {'model': 'Falcon-7B', 'date': '2023-04-24', 'parameters': 7000000000.0, 'mmlu': 0.269, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Falcon-7B', 'date': '2023-04-24', 'parameters': 7000000000.0, 'mmlu': 0.239, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Redpajama-7B', 'date': '2023-05-04', 'parameters': 7000000000.0, 'mmlu': 0.263, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Redpajama-7B', 'date': '2023-05-04', 'parameters': 7000000000.0, 'mmlu': 0.258, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'MPT 7B', 'date': '2023-05-05', 'parameters': 7000000000.0, 'mmlu': 0.308, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'MPT 7B', 'date': '2023-05-05', 'parameters': 7000000000.0, 'mmlu': 0.268, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'MPT-7B', 'date': '2023-05-05', 'parameters': 7000000000.0, 'mmlu': 0.2793, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'MPT-7B', 'date': '2023-05-05', 'parameters': 7000000000.0, 'mmlu': 0.268, 'shots': 2.0, 'source': 'Textbooks Are All You Need II: phi-1.5 technical report'}, {'model': 'MPT-7B', 'date': '2023-05-05', 'parameters': 7000000000.0, 'mmlu': 0.267, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'MPT-7B', 'date': '2023-05-05', 'parameters': 7000000000.0, 'mmlu': 0.274, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'Baichuan 1-7B', 'date': '2023-06-01', 'parameters': 7000000000.0, 'mmlu': 0.423, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'OpenLLaMA-7B', 'date': '2023-06-07', 'parameters': 7000000000.0, 'mmlu': 0.299, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'OpenLLaMA-7B', 'date': '2023-06-07', 'parameters': 7000000000.0, 'mmlu': 0.286, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'MPT 30B', 'date': '2023-06-22', 'parameters': 30000000000.0, 'mmlu': 0.479, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'MPT 30B', 'date': '2023-06-22', 'parameters': 30000000000.0, 'mmlu': 0.469, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'ChatGLM2 6B', 'date': '2023-06-24', 'parameters': 6000000000.0, 'mmlu': 0.479, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'XGen-7B', 'date': '2023-06-27', 'parameters': 7000000000.0, 'mmlu': 0.363, 'shots': 5.0, 'source': 'XGen-7B Technical Report'}, {'model': 'XGen-7B', 'date': '2023-06-27', 'parameters': 7000000000.0, 'mmlu': 0.321, 'shots': 0.0, 'source': 'XGen-7B Technical Report'}, {'model': 'InternLM 7B', 'date': '2023-07-05', 'parameters': 7000000000.0, 'mmlu': 0.51, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'Baichuan 1-13B-Base', 'date': '2023-07-11', 'parameters': 13000000000.0, 'mmlu': 0.516, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'Llama 2 (7B)', 'date': '2023-07-18', 'parameters': 7000000000.0, 'mmlu': 0.458, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama 2 (13B)', 'date': '2023-07-18', 'parameters': 13000000000.0, 'mmlu': 0.554, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama 2 (70B)', 'date': '2023-07-18', 'parameters': 70000000000.0, 'mmlu': 0.695, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'LLAMA 2 7B', 'date': '2023-07-18', 'parameters': 7000000000.0, 'mmlu': 0.453, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'LLaMA 2 7B     ', 'date': '2023-07-18', 'parameters': 7000000000.0, 'mmlu': 0.444, 'shots': 5.0, 'source': 'Mixtral of Experts'}, {'model': 'LLAMA 2 13B', 'date': '2023-07-18', 'parameters': 13000000000.0, 'mmlu': 0.548, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'LLaMA 2 13B    ', 'date': '2023-07-18', 'parameters': 13000000000.0, 'mmlu': 0.556, 'shots': 5.0, 'source': 'Mixtral of Experts'}, {'model': 'LLAMA 2 34B', 'date': '2023-07-18', 'parameters': 34000000000.0, 'mmlu': 0.626, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'LLaMA 2 34B', 'date': '2023-07-18', 'parameters': 34000000000.0, 'mmlu': 0.626, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'LLAMA 2 70B', 'date': '2023-07-18', 'parameters': 70000000000.0, 'mmlu': 0.689, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'LLaMA 2 70B', 'date': '2023-07-18', 'parameters': 70000000000.0, 'mmlu': 0.698, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'LLaMA 2 70B    ', 'date': '2023-07-18', 'parameters': 70000000000.0, 'mmlu': 0.699, 'shots': 5.0, 'source': 'Mixtral of Experts'}, {'model': 'LLaMA 2-7B', 'date': '2023-07-18', 'parameters': 7000000000.0, 'mmlu': 0.4573, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'LLaMA 2-13B', 'date': '2023-07-18', 'parameters': 13000000000.0, 'mmlu': 0.5509, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'LLaMA-2 7B', 'date': '2023-07-18', 'parameters': 7000000000.0, 'mmlu': 0.453, 'shots': 5.0, 'source': 'Gemma: Open Models Based on Gemini Research and Technology'}, {'model': 'LLaMA-2 13B', 'date': '2023-07-18', 'parameters': 13000000000.0, 'mmlu': 0.548, 'shots': 5.0, 'source': 'Gemma: Open Models Based on Gemini Research and Technology'}, {'model': 'LLaMA-2 13B', 'date': '2023-07-18', 'parameters': 13000000000.0, 'mmlu': 0.548, 'shots': 5.0, 'source': 'Nemotron-4 15B Technical Report'}, {'model': 'LLaMA-2 34B', 'date': '2023-07-18', 'parameters': 34000000000.0, 'mmlu': 0.626, 'shots': 5.0, 'source': 'Nemotron-4 15B Technical Report'}, {'model': 'Llama2-7B', 'date': '2023-07-18', 'parameters': 7000000000.0, 'mmlu': 0.453, 'shots': 2.0, 'source': 'Textbooks Are All You Need II: phi-1.5 technical report'}, {'model': 'LLaMA2-Chat 13B', 'date': '2023-07-18', 'parameters': 13000000000.0, 'mmlu': 0.509, 'shots': 0.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'LLaMA2-Chat 13B', 'date': '2023-07-18', 'parameters': 13000000000.0, 'mmlu': 0.473, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'LLaMA2-Chat 70B', 'date': '2023-07-18', 'parameters': 70000000000.0, 'mmlu': 0.594, 'shots': 0.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'LLaMA2-Chat 70B', 'date': '2023-07-18', 'parameters': 70000000000.0, 'mmlu': 0.599, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'StableBeluga2 70B', 'date': '2023-07-20', 'parameters': 70000000000.0, 'mmlu': 0.686, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'Baichuan 2-13B-Base', 'date': '2023-09-06', 'parameters': 13000000000.0, 'mmlu': 0.5917, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'Baichuan-2 13B', 'date': '2023-09-06', 'parameters': 13000000000.0, 'mmlu': 0.592, 'shots': 5.0, 'source': 'Nemotron-4 15B Technical Report'}, {'model': 'Baichuan2-Chat 13B', 'date': '2023-09-06', 'parameters': 13000000000.0, 'mmlu': 0.551, 'shots': 0.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Baichuan2-Chat 13B', 'date': '2023-09-06', 'parameters': 13000000000.0, 'mmlu': 0.501, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Falcon-180B', 'date': '2023-09-06', 'parameters': 180000000000.0, 'mmlu': 0.706, 'shots': 5.0, 'source': 'The Falcon Series of Open Language Models'}, {'model': 'InternLM-Chat 20B', 'date': '2023-09-17', 'parameters': 20000000000.0, 'mmlu': 0.556, 'shots': 0.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'InternLM-Chat 20B', 'date': '2023-09-17', 'parameters': 20000000000.0, 'mmlu': 0.574, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Baichuan 2-7B-Base', 'date': '2023-09-20', 'parameters': 7000000000.0, 'mmlu': 0.5416, 'shots': 5.0, 'source': 'Baichuan 2: Open Large-scale Language Models'}, {'model': 'Qwen-Chat 14B', 'date': '2023-09-24', 'parameters': 14000000000.0, 'mmlu': 0.64, 'shots': 0.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Qwen-Chat 14B', 'date': '2023-09-24', 'parameters': 14000000000.0, 'mmlu': 0.65, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Mistral 7B', 'date': '2023-09-27', 'parameters': 7000000000.0, 'mmlu': 0.625, 'shots': 5.0, 'source': 'Mixtral of Experts'}, {'model': 'Mistral 7B', 'date': '2023-09-27', 'parameters': 7000000000.0, 'mmlu': 0.601, 'shots': 5.0, 'source': 'Nemotron-4 15B Technical Report'}, {'model': 'Mistral 7b             ', 'date': '2023-09-27', 'parameters': 7000000000.0, 'mmlu': 0.617, 'shots': 5.0, 'source': 'Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone'}, {'model': 'Mistral v0.1 (7B)', 'date': '2023-09-27', 'parameters': 7000000000.0, 'mmlu': 0.566, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Qwen 7B', 'date': '2023-09-28', 'parameters': 7000000000.0, 'mmlu': 0.45, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'QWEN 14B', 'date': '2023-09-28', 'parameters': 14000000000.0, 'mmlu': 0.663, 'shots': 5.0, 'source': 'Nemotron-4 15B Technical Report'}, {'model': 'Qwen 14B', 'date': '2023-09-28', 'parameters': 14000000000.0, 'mmlu': 0.534, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'DS-Coder-1.3B-Base', 'date': '2023-11-02', 'parameters': 1300000000.0, 'mmlu': 0.258, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'DS-Coder-6.7B-Base', 'date': '2023-11-02', 'parameters': 6700000000.0, 'mmlu': 0.364, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'DS-Coder-33B-Base', 'date': '2023-11-02', 'parameters': 33000000000.0, 'mmlu': 0.394, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Yi (6B)', 'date': '2023-11-02', 'parameters': 6000000000.0, 'mmlu': 0.64, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Yi (34B)', 'date': '2023-11-02', 'parameters': 34000000000.0, 'mmlu': 0.762, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Yi 6B', 'date': '2023-11-02', 'parameters': 6000000000.0, 'mmlu': 0.632, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Yi 34B', 'date': '2023-11-02', 'parameters': 34000000000.0, 'mmlu': 0.763, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Yi-Chat 6B', 'date': '2023-11-22', 'parameters': 6000000000.0, 'mmlu': 0.582, 'shots': 0.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Yi-Chat 6B', 'date': '2023-11-22', 'parameters': 6000000000.0, 'mmlu': 0.61, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Yi-Chat 34B', 'date': '2023-11-22', 'parameters': 34000000000.0, 'mmlu': 0.676, 'shots': 0.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Yi-Chat 34B', 'date': '2023-11-22', 'parameters': 34000000000.0, 'mmlu': 0.735, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Qwen 1.8B', 'date': '2023-11-30', 'parameters': 1800000000.0, 'mmlu': 0.282, 'shots': 5.0, 'source': 'Qwen Technical Report'}, {'model': 'Mistral 7B', 'date': '2023-12-11', 'parameters': 7000000000.0, 'mmlu': 0.625, 'shots': 5.0, 'source': 'Gemma: Open Models Based on Gemini Research and Technology'}, {'model': 'Qwen1.5 (7B)', 'date': '2024-02-04', 'parameters': 7000000000.0, 'mmlu': 0.626, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Qwen1.5 (14B)', 'date': '2024-02-04', 'parameters': 14000000000.0, 'mmlu': 0.686, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Qwen1.5 (32B)', 'date': '2024-02-04', 'parameters': 32000000000.0, 'mmlu': 0.744, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'StarCoder2-7B', 'date': '2024-02-20', 'parameters': 7000000000.0, 'mmlu': 0.388, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'StarCoder2-15B', 'date': '2024-02-20', 'parameters': 15000000000.0, 'mmlu': 0.641, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Gemma (7B)', 'date': '2024-02-21', 'parameters': 7000000000.0, 'mmlu': 0.661, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Gemma 2B', 'date': '2024-02-21', 'parameters': 2000000000.0, 'mmlu': 0.423, 'shots': 5.0, 'source': 'Gemma: Open Models Based on Gemini Research and Technology'}, {'model': 'Gemma 7B', 'date': '2024-02-21', 'parameters': 7000000000.0, 'mmlu': 0.643, 'shots': 5.0, 'source': 'Gemma: Open Models Based on Gemini Research and Technology'}, {'model': 'Gemma 7B', 'date': '2024-02-21', 'parameters': 7000000000.0, 'mmlu': 0.643, 'shots': 5.0, 'source': 'Nemotron-4 15B Technical Report'}, {'model': 'Gemma 7b               ', 'date': '2024-02-21', 'parameters': 7000000000.0, 'mmlu': 0.636, 'shots': 5.0, 'source': 'Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone'}, {'model': 'StarCoder2-3B', 'date': '2024-02-22', 'parameters': 3000000000.0, 'mmlu': 0.366, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Nemotron-4 15B', 'date': '2024-02-26', 'parameters': 15000000000.0, 'mmlu': 0.587, 'shots': 5.0, 'source': 'Nemotron-4 15B Technical Report'}, {'model': 'Yi-9B', 'date': '2024-03-01', 'parameters': 9000000000.0, 'mmlu': 0.684, 'shots': 5.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'CodeQwen1.5-7B', 'date': '2024-04-15', 'parameters': 7000000000.0, 'mmlu': 0.405, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Llama 3 (8B)', 'date': '2024-04-18', 'parameters': 8000000000.0, 'mmlu': 0.688, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama 3 (70B)', 'date': '2024-04-18', 'parameters': 70000000000.0, 'mmlu': 0.793, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama-3-In 8b          ', 'date': '2024-04-18', 'parameters': 8000000000.0, 'mmlu': 0.665, 'shots': 5.0, 'source': 'Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone'}, {'model': 'Phi-3 (14B)', 'date': '2024-04-23', 'parameters': 14000000000.0, 'mmlu': 0.775, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Phi-3 14b', 'date': '2024-04-23', 'parameters': 14000000000.0, 'mmlu': 0.779, 'shots': 5.0, 'source': 'Phi-4 Technical Report'}, {'model': 'Falcon2-11B', 'date': '2024-05-09', 'parameters': 11000000000.0, 'mmlu': 0.584, 'shots': 5.0, 'source': 'Falcon2-11B Technical Report'}, {'model': 'Mistral Instruct v0.3 (7B)', 'date': '2024-05-27', 'parameters': 7000000000.0, 'mmlu': 0.599, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Qwen2 Instruct (72B)', 'date': '2024-06-07', 'parameters': 72000000000.0, 'mmlu': 0.824, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Gemma 2 Instruct (9B)', 'date': '2024-06-24', 'parameters': 9000000000.0, 'mmlu': 0.721, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Gemma 2 Instruct (27B)', 'date': '2024-06-24', 'parameters': 27000000000.0, 'mmlu': 0.757, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama 3.1 Instruct Turbo (8B)', 'date': '2024-07-23', 'parameters': 8000000000.0, 'mmlu': 0.561, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama 3.1 Instruct Turbo (70B)', 'date': '2024-07-23', 'parameters': 70000000000.0, 'mmlu': 0.801, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama 3.1 Instruct Turbo (405B)', 'date': '2024-07-23', 'parameters': 405000000000.0, 'mmlu': 0.845, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'LLaMA-3.1 405B Base', 'date': '2024-07-23', 'parameters': 405000000000.0, 'mmlu': 0.844, 'shots': 5.0, 'source': 'DeepSeek-V3 Technical Report'}, {'model': 'Qwen2.5-Coder-0.5B', 'date': '2024-09-18', 'parameters': 500000000.0, 'mmlu': 0.42, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Qwen2.5-Coder-1.5B', 'date': '2024-09-18', 'parameters': 1500000000.0, 'mmlu': 0.536, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Qwen2.5-Coder-3B', 'date': '2024-09-18', 'parameters': 3000000000.0, 'mmlu': 0.612, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Qwen2.5-Coder-7B', 'date': '2024-09-18', 'parameters': 7000000000.0, 'mmlu': 0.68, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Qwen2.5-Coder-14B', 'date': '2024-09-18', 'parameters': 14000000000.0, 'mmlu': 0.752, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Qwen2.5-Coder-32B', 'date': '2024-09-18', 'parameters': 32000000000.0, 'mmlu': 0.791, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}, {'model': 'Qwen2.5 14b instruct', 'date': '2024-09-19', 'parameters': 14000000000.0, 'mmlu': 0.799, 'shots': 5.0, 'source': 'Phi-4 Technical Report'}, {'model': 'Qwen2.5 72B Base', 'date': '2024-09-19', 'parameters': 72000000000.0, 'mmlu': 0.85, 'shots': 5.0, 'source': 'DeepSeek-V3 Technical Report'}, {'model': 'Qwen2.5 72b instruct', 'date': '2024-09-19', 'parameters': 72000000000.0, 'mmlu': 0.853, 'shots': 5.0, 'source': 'Phi-4 Technical Report'}, {'model': 'Qwen2.5 Instruct Turbo (7B)', 'date': '2024-09-19', 'parameters': 7000000000.0, 'mmlu': 0.729, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Qwen2.5 Instruct Turbo (72B)', 'date': '2024-09-19', 'parameters': 72000000000.0, 'mmlu': 0.834, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama 3.2 Vision Instruct Turbo (11B)', 'date': '2024-09-24', 'parameters': 11000000000.0, 'mmlu': 0.565, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama 3.2 Vision Instruct Turbo (90B)', 'date': '2024-09-24', 'parameters': 90000000000.0, 'mmlu': 0.803, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama 3.3 Instruct Turbo (70B)', 'date': '2024-12-06', 'parameters': 70000000000.0, 'mmlu': 0.791, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Llama3.3 70b instruct', 'date': '2024-12-06', 'parameters': 70000000000.0, 'mmlu': 0.863, 'shots': 5.0, 'source': 'Phi-4 Technical Report'}, {'model': 'Phi-4 14b', 'date': '2024-12-12', 'parameters': 14000000000.0, 'mmlu': 0.848, 'shots': 5.0, 'source': 'Phi-4 Technical Report'}], [{'model': 'Gopher 280B', 'date': '2021-12-08', 'parameters': 280000000000.0, 'mmlu': 0.6, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'Chinchilla 70B', 'date': '2022-03-29', 'parameters': 70000000000.0, 'mmlu': 0.675, 'shots': 5.0, 'source': 'LLaMA: Open and Efficient Foundation Language Models'}, {'model': 'LLAMA 1 65B', 'date': '2023-02-24', 'parameters': 65000000000.0, 'mmlu': 0.634, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'LLAMA 2 34B', 'date': '2023-07-18', 'parameters': 34000000000.0, 'mmlu': 0.626, 'shots': 5.0, 'source': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'}, {'model': 'Qwen-Chat 14B', 'date': '2023-09-24', 'parameters': 14000000000.0, 'mmlu': 0.64, 'shots': 0.0, 'source': 'Yi: Open Foundation Models by 01.AI'}, {'model': 'Mistral 7B', 'date': '2023-09-27', 'parameters': 7000000000.0, 'mmlu': 0.625, 'shots': 5.0, 'source': 'Mixtral of Experts'}, {'model': 'Yi (6B)', 'date': '2023-11-02', 'parameters': 6000000000.0, 'mmlu': 0.64, 'shots': 5.0, 'source': 'Stanford CRFM Leaderboard'}, {'model': 'Qwen2.5-Coder-3B', 'date': '2024-09-18', 'parameters': 3000000000.0, 'mmlu': 0.612, 'shots': None, 'source': 'Qwen2.5-Coder Technical Report'}], 0.6)

def _yr(d):
    d = dt.date.fromisoformat(d)
    return d.year + (d.timetuple().tm_yday - 1) / (SEC_PER_YEAR_LEAP / SEC_PER_DAY)
fig, ax = plt.subplots(figsize=(4.0, 2.6))
for hi, ck, sz in ((False, 'neutral_l', 16), (True, 'memory', 22)):
    pts = [r for r in _rows if (r['mmlu'] >= _bar) == hi]
    ax.scatter([_yr(r['date']) for r in pts], [r['parameters'] for r in pts], s=sz, color=C[ck], zorder=2 + hi)
fx = [_yr(r['date']) for r in _front] + [2025.1]
fy = [r['parameters'] for r in _front] + [_front[-1]['parameters']]
ax.step(fx, fy, where='post', color=C['danger'], lw=2.2, zorder=4)
labels = {'Gopher 280B': (8, 4, 'left'), 'Chinchilla 70B': (8, 6, 'left'), 'LLAMA 2 34B': (8, 8, 'left'), 'Mistral 7B': (8, 6, 'left'), 'Qwen2.5-Coder-3B': (-4, -16, 'right')}
for _r in _front:
    if _r['model'] in labels:
        dx, dy, ha = labels[_r['model']]
        ax.annotate(_r['model'].replace('LLAMA', 'Llama').replace('Qwen2.5-Coder-3B', 'Qwen2.5-Coder 3B'), (_yr(_r['date']), _r['parameters']), xytext=(dx, dy), textcoords='offset points', fontsize=11, fontweight='bold', color=C['danger'], ha=ha, zorder=5, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=0.4))
ax.text(2021.55, 140000000.0, 'Blue: MMLU ≥ 60%.  Gray: below.', fontsize=11, color=C['memory'])
ax.set_yscale('log')
ax.set_ylim(100000000.0, 1000000000000.0)
ax.set_yticks([1000000000.0, 10000000000.0, 100000000000.0, 1000000000000.0])
ax.set_yticklabels(['1B', '10B', '100B', '1T'])
ax.minorticks_off()
ax.set_xticks([2022, 2023, 2024, 2025])
ax.set_xlim(2021.5, 2025.2)
ax.set_ylabel('Parameters (log)')
ax.grid(axis='y', color=C['grid'], linewidth=0.6)
sp.save(fig, out / 'smallest_capable.svg')
pts = [('V3-Large 1.0', 75.2, 51.0), ('V3-Large 0.75', 73.3, 39.0), ('MnasNet-A1', 75.2, 71.0), ('Proxyless', 74.6, 72.0), ('V2 1.0', 72.0, 64.0), ('V3-Small 1.0', 67.4, 15.8), ('V3-Small 0.75', 65.4, 12.8), ('Mnas-small', 64.9, 20.3), ('V2 0.35', 60.8, 16.6)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.6, 1.95))
    for n, a, t in pts:
        on_f = n in ['V3-Large 1.0', 'V3-Large 0.75', 'V3-Small 1.0', 'V3-Small 0.75']
        ck = 'memory' if on_f else 'neutral'
        if n == 'V2 1.0':
            ck = 'danger'
        ax.plot([t], [a], 'o', ms=9, color=C[ck], zorder=3)
    lab = {'V2 1.0': ('MobileNetV2 1.0: the blur model', 5, -16, 'left', 'danger'), 'V3-Large 0.75': ('V3-Large 0.75', -8, 6, 'right', 'memory'), 'V3-Large 1.0': ('V3-Large 1.0', -8, 6, 'right', 'memory'), 'V3-Small 1.0': ('V3-Small 1.0', 8, -12, 'left', 'memory'), 'MnasNet-A1': ('MnasNet-A1', 0, 8, 'center', 'neutral')}
    for n, a, t in pts:
        if n in lab:
            s_, dx, dy, ha, ck = lab[n]
            ax.annotate(s_, (t, a), xytext=(dx, dy), textcoords='offset points', fontsize=11, fontweight='bold', color=C[ck], ha=ha)
    fr = sorted(((t, a) for n, a, t in pts if n in ['V3-Large 1.0', 'V3-Large 0.75', 'V3-Small 1.0', 'V3-Small 0.75']))
    al = 1.0 if k == 2 else 0.0
    ax.plot([p[0] for p in fr], [p[1] for p in fr], '-', color=C['memory'], lw=2.0, alpha=al, zorder=2)
    ax.text(13, 71.0, 'frontier', fontsize=12, fontweight='bold', color=C['memory'], alpha=al, rotation=0)
    ax.annotate('', (39.0 + 1.5, 73.3 - 0.2), (64.0 - 1.5, 72.0 + 0.2), arrowprops=dict(arrowstyle='-|>', color=C['data'], lw=2.0, alpha=al))
    ax.text(44, 69.0, '1.64× faster,\n+1.3 points', fontsize=11, fontweight='bold', color=C['data'], alpha=al, ha='left', va='center')
    ax.set_xlim(0, 80)
    ax.set_ylim(59, 77)
    ax.set_xlabel('Latency on a Pixel 1, float, one core (ms)')
    ax.set_ylabel('ImageNet top-1 (%)')
    ax.grid(color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'frontier_step{k}.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    series = [([(0.0, 0.0), (33.5, -0.03), (49.9, -0.12), (66.6, -0.93), (74.9, -2.16), (80.0, -4.07)], 'danger', '--', 'o', 'prune, no retrain', 1.0), ([(0.0, 0.0), (33.5, -0.02), (50.0, 0.02), (66.6, 0.17), (75.0, 0.15), (80.0, 0.02), (83.3, -0.18), (85.7, -0.47), (87.5, -0.82), (89.9, -1.59), (93.0, -3.88)], 'memory', '-', 's', 'prune once, retrain', 1.0), ([(87.5, 0.05), (88.9, 0.02), (90.0, -0.08), (91.7, -0.35), (92.8, -0.62), (93.8, -1.05), (94.7, -2.0), (95.8, -4.19)], 'data', '-', 'D', 'prune and retrain,\nrepeated', 1.0 if k == 2 else 0.0)]
    for pts_, ck, ls, mk, name, a in series:
        xs = [x for x, _ in pts_]
        ys = [y for _, y in pts_]
        ax.plot(xs, ys, ls, marker=mk, color=C[ck], lw=2.2, ms=6, alpha=a, clip_on=True)
    ax.text(42, -1.6, 'no retrain', fontsize=12, fontweight='bold', color=C['danger'])
    ax.text(70, -3.9, 'prune once,\nretrain', fontsize=12, fontweight='bold', color=C['memory'], ha='right')
    ax.text(96.6, -3.1, 'iterative', fontsize=12, fontweight='bold', color=C['data'], ha='left', alpha=1.0 if k == 2 else 0.0)
    ax.axhline(0, color=C['text'], lw=0.8)
    ax.set_xlim(40, 100)
    ax.set_ylim(-4.5, 0.6)
    ax.set_xticks([40, 60, 80, 90, 100])
    ax.set_xticklabels(['40%', '60%', '80%', '90%', '100%'])
    ax.set_xlabel('Weights pruned')
    ax.set_ylabel('Top-5 change (points)')
    ax.grid(color=C['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'han_step{k}.svg')
rows_ = [{'name': 'width 1.0, dense', 'width': 1.0, 'target_sparsity': 0.0, 'zeros_share': 0.0, 'params': 3504872, 'flops': 601548544, 'median_ms': 14.12856253, 'p10_ms': 11.33871684, 'p90_ms': 16.18063373, 'runs': 1000}, {'name': 'width 1.0, 80% zeros', 'width': 1.0, 'target_sparsity': 0.8, 'zeros_share': 0.8, 'params': 3504872, 'flops': 601548544, 'median_ms': 14.20027099, 'p10_ms': 11.41669219, 'p90_ms': 16.23724636, 'runs': 1000}, {'name': 'width 0.75, dense', 'width': 0.75, 'target_sparsity': 0.0, 'zeros_share': 0.0, 'params': 2636424, 'flops': 418139584, 'median_ms': 12.50358298, 'p10_ms': 9.58135, 'p90_ms': 14.33517917, 'runs': 1000}]
base_ = rows_[0]
names_ = ['Width 1.0', 'Width 1.0,\n80% zeros', 'Width 0.75']
fig, ax = plt.subplots(figsize=(5.8, 2.4))
for i, (r, nm) in enumerate(zip(rows_, names_)):
    y = 2 - i
    fp = 100 * r['flops'] / base_['flops']
    tm = 100 * r['median_ms'] / base_['median_ms']
    ax.barh(y + 0.19, fp, height=0.34, color=C['neutral'])
    ax.barh(y - 0.19, tm, height=0.34, color=C['compute'])
    ax.text(fp + 1.5, y + 0.19, f'FLOPs {fp:.0f}%', va='center', fontsize=12, fontweight='bold', color=C['neutral'])
    ax.text(tm + 1.5, y - 0.19, f"time {tm:.0f}%  ({r['median_ms']:.1f} ms)", va='center', fontsize=12, fontweight='bold', color=C['compute'])
    ax.text(-0.01, y, nm, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
ax.set_xlim(0, 150)
ax.set_xticks([0, 50, 100])
ax.set_xticklabels(['0', '50%', '100%'])
ax.set_ylim(-0.6, 2.6)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.grid(axis='x', color=C['grid'], linewidth=0.6)
ax.set_axisbelow(True)
ax.set_xlabel('Relative to width 1.0, dense (median of 1,000 runs)')
sp.save(fig, out / 'cpu_measure.svg')
_cliff_color = {'Pruning + quantization': 'data', 'Pruning only': 'compute', 'Quantization only': 'memory', 'SVD': 'danger'}
CLIFF = {name: (ck, [(r['size_pct'], r['accuracy_change']) for r in load_series('slides/vol1/10_model_compression/images/data/han2016_compression_methods.csv').entries if r['method'] == name]) for name, ck in _cliff_color.items()}
fig, ax = plt.subplots(figsize=(5.8, 2.7))
label_at = {'Pruning + quantization': (2.2, 0.3, 'left'), 'Pruning only': (10.0, 0.3, 'left'), 'Quantization only': (15.3, 0.3, 'left'), 'SVD': (17.2, -3.6, 'left')}
for name, (ck, pts) in CLIFF.items():
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=C[ck], lw=2.6)
    ax.plot(xs[1:], ys[1:], 'o', ms=5, color=C[ck], mfc='white', mew=1.8)
    lx, ly, ha = label_at[name]
    ax.text(lx, ly, name, color=C[ck], fontsize=12, fontweight='bold', ha=ha)
ax.axhline(0, color=C['neutral'], lw=1)
ax.set_xlim(2, 20)
ax.set_ylim(-4.5, 0.9)
ax.set_xticks([2, 5, 10, 15, 20])
ax.set_xticklabels(['2%', '5%', '10%', '15%', '20%'])
ax.set_yticks([-4, -3, -2, -1, 0])
ax.set_yticklabels(['-4', '-3', '-2', '-1', '0'])
ax.set_xlabel('Model size kept (% of original)')
ax.set_ylabel('Accuracy change (points)')
ax.grid(color=C['grid'], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'cliff_methods.svg')
print('figures written')
