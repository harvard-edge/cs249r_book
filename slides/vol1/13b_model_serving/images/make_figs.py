#!/usr/bin/env python3
"""Figures for Lecture 13b (Model Serving, Part 2: Serving a Stream of Tokens).

Every number comes from calcs.py (imported as K), which reads MLSysIM:
the token-serving Scenarios.* and the cited results in Literature (MLPerf
Inference v5.0 Llama 2 70B; Splitwise; DistServe; PagedAttention; the API
price series behind the chapter's figure 9). Labels print the formatted
strings calcs.py records (deckvals.val), so a figure and its slide show the
same digits. Values computed here are labeled "derived" on the figure or its
slide. Illustrative geometry (the fragmentation strip, the paging example,
slot placement in the schedules) is layout and stays in this file.

Run from this folder: python3 make_figs.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from matplotlib.patches import FancyArrowPatch, Rectangle
import contextlib
import io
C = sp.C
plt = sp.plt
out = Path(__file__).parent

def cm(w, h):
    """Figure size for a slot of w by h centimeters on the 16 x 9 cm page."""
    return (w / 2.54, h / 2.54)

def save_slot(fig, path):
    """Save at exactly the figure size, so the slide places it at scale 1."""
    fig.savefig(path, format='svg', facecolor='white')
    plt.close(fig)

def tag(ax, text, x=0.0, y=-0.3):
    """A small gray provenance tag under the axes (modeled, derived ...)."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, color=C['neutral'], ha='left', va='top')

def hide_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
HOOK = [('No deadline\n(Offline)', 31216.8, 'neutral', '31,217'), ('TTFT 2 s,\nTPOT 200 ms', 30903.6, 'compute', '30,904  (99%)'), ('TTFT 450 ms,\nTPOT 40 ms', 12979.4, 'danger', '12,979  (42%)')]

def hook_fig(path, k, size=(5.8, 2.6), note=True, fs=12, xlabel=True, rows=None):
    fig, ax = plt.subplots(figsize=size)
    for i, (lab, v, ck, txt) in enumerate(rows or HOOK):
        y = 2 - i
        shown = i < k if i < 2 else k >= 4
        ax.barh(y, v / 1000, color=C[ck], height=0.6, alpha=1.0 if shown else 0.0)
        ax.text(-0.4, y, lab, ha='right', va='center', fontsize=fs, fontweight='bold', alpha=1.0 if i < k or (i == 2 and k >= 3) else 0.0)
        if i == 2 and k == 3:
            ax.text(0.6, y, '???', va='center', fontsize=fs + 2, fontweight='bold', color=C['danger'])
        ax.text(v / 1000 + 0.5, y, txt, va='center', fontsize=fs, fontweight='bold', color=C[ck], alpha=1.0 if shown else 0.0)
    if note:
        ax.text(39, -0.75, 'The same server keeps 2.4 times fewer tokens per second when TPOT tightens 5 times.', ha='right', va='center', fontsize=11, color=C['danger'], fontweight='bold', alpha=1.0 if k >= 4 else 0.0)
    ax.set_yticks([])
    ax.set_xlim(0, 40)
    ax.set_ylim(-1.1 if note else -0.5, 2.5)
    ax.spines['left'].set_visible(False)
    if xlabel:
        ax.set_xlabel('Thousand tokens/s, Llama 2 70B on one 8× H100 server')
    else:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
    sp.save(fig, path)
for k in (1, 2, 3, 4):
    hook_fig(out / f'hook_step{k}.svg', k)
THUMB = [('Offline', 31216.8, 'neutral', '31,217'), ('200 ms', 30903.6, 'compute', '30,904'), ('40 ms', 12979.4, 'danger', '12,979')]
hook_fig(out / 'hook_thumb.svg', 4, size=(3.6, 1.6), note=False, fs=16, xlabel=False, rows=THUMB)
sp.hbars(out / 'recall_cost.svg', [('Batch 1', 3.92, 'danger', '\\$3.92'), ('Batch 8', 0.52, 'compute', '\\$0.52')], '', log=False, size=(4.0, 1.4), xlabel='ResNet-50, \\$ per million requests (modeled)')
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.9, 2.6))
    rows = [('DGX-H100', 1.3, 84.0, 28.0, 3387.0), ('DGX-A100', 0.0, 155.0, 40.0, 4957.0)]
    for name, y, ttft, tbt, e2e in rows:
        a = 1.0 if y > 0.5 or k >= 3 else 0.0
        ax.barh(y, ttft, color=C['compute'], height=0.42, alpha=a)
        for t in np.arange(ttft + tbt, e2e, tbt):
            ax.plot([t, t], [y - 0.2, y + 0.2], color=C['memory'], lw=1.1, alpha=a)
        ax.text(-60, y, name, ha='right', va='center', fontsize=12, fontweight='bold', alpha=a)
        ax.text(e2e + 50, y, f'{e2e:,.0f} ms', va='center', fontsize=11, fontweight='bold', alpha=a)
        ax.text(600, y + 0.3, f'first token {ttft:.0f} ms, then one every {tbt:.0f} ms', fontsize=10.5, color=C['memory'], fontweight='bold', alpha=a)
    if k >= 2:
        ax.text(600, 0.82, 'decode: 98% of the request (derived)', ha='left', va='center', fontsize=10.5, color=C['memory'], fontweight='bold')
    if k >= 4:
        ax.axvline(500.0, color=C['danger'], lw=2, ls='--')
        ax.text(600, -0.62, 'book target: TTFT < 500 ms;  TPOT < 50 ms (20 tokens/s)', fontsize=10.5, color=C['danger'], fontweight='bold', va='center')
    ax.set_xlim(0, 5600)
    ax.set_ylim(-0.85, 1.85)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Milliseconds, one chat request (Llama 2 70B, median, no batching)')
    sp.save(fig, out / f'ttft_step{k}.svg')
PHASE_PRE = [('GPT-2 XL,\n512 tokens', 3.882709808, '3.9'), ('Llama 3 8B,\n1,000 tokens', 41.93186047, '41.9')]
PHASE_DEC = [('GPT-2 XL', 0.8955223881, 'memory', '0.9 ms'), ('Llama 3 8B,\nINT4', 1.567649681, 'memory', '1.2 floor, 1.57'), ('GPT-2 XL,\neager', 8.64, 'neutral', '8.6 ms')]
FS = 9
for k in (1, 2, 3, 4):
    fig = plt.figure(figsize=cm(10.4, 5.0))
    ax = fig.add_axes([0.21, 0.46, 0.25, 0.4])
    bx = fig.add_axes([0.71, 0.46, 0.27, 0.4])
    cx = fig.add_axes([0.21, 0.1, 0.77, 0.12])
    for i, (lab, v, vtxt) in enumerate(PHASE_PRE):
        y = 1 - i
        ax.barh(y, v, color=C['compute'], height=0.6)
        ax.text(v + 1.5 if v < 30 else v - 1.5, y, f'{vtxt} ms', va='center', ha='left' if v < 30 else 'right', fontsize=FS, fontweight='bold', color=C['compute'] if v < 30 else 'white')
        ax.text(-2, y, lab, ha='right', va='center', fontsize=FS - 0.5, fontweight='bold')
    ax.set_xlim(0, 50)
    ax.set_title('Prefill: FLOP ÷ (40% of peak)', fontsize=FS, loc='left', fontweight='bold', color=C['compute'])
    ax.set_xlabel('ms (sets TTFT)', fontsize=FS - 0.5, labelpad=1)
    a2 = 1.0 if k >= 2 else 0.0
    for i, (lab, v, ck, txt) in enumerate(PHASE_DEC):
        y = 2 - i
        a = a2 if i < 2 else 1.0 if k >= 3 else 0.0
        bx.barh(y, v, color=C[ck], height=0.6, alpha=a)
        bx.text(v + 0.3 if v < 5 else v - 0.3, y, txt, va='center', ha='left' if v < 5 else 'right', fontsize=FS - 0.5, fontweight='bold', color=C[ck] if v < 5 else 'white', alpha=a)
        bx.text(-0.4, y, lab, ha='right', va='center', fontsize=FS - 0.5, fontweight='bold', alpha=a)
    bx.set_xlim(0, 10)
    bx.set_title('Decode: GB ÷ 3.35 TB/s', fontsize=FS, loc='right', fontweight='bold', color=C['memory'], alpha=a2)
    bx.set_xlabel('ms (sets TPOT)', fontsize=FS - 0.5, alpha=a2, labelpad=1)
    if k < 2:
        bx.spines['bottom'].set_alpha(0.0)
        bx.tick_params(axis='x', colors='white')
    for a_ in (ax, bx):
        a_.set_yticks([])
        a_.spines['left'].set_visible(False)
        a_.set_ylim(-0.5, 2.5 if a_ is bx else 1.5)
        a_.tick_params(axis='x', labelsize=8, pad=1)
    a4 = 1.0 if k >= 4 else 0.0
    tp = 1.567649681
    cx.barh(0, tp, color=C['memory'], height=0.8, alpha=a4)
    cx.barh(0, 20 - tp, left=tp, color=C['data_l'], edgecolor=C['data'], hatch='//', height=0.8, lw=0, alpha=a4)
    cx.text(10.8, 0, '18.4 ms: room to share this read (derived)', ha='center', va='center', fontsize=FS - 0.5, fontweight='bold', color=C['data'], alpha=a4, bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=a4))
    cx.text(-0.3, 0, 'Llama 3 8B,\n20 ms TPOT', ha='right', va='center', fontsize=FS - 0.5, fontweight='bold', alpha=a4)
    cx.set_xlim(0, 20)
    cx.set_ylim(-0.6, 0.6)
    cx.set_yticks([])
    cx.set_xticks([0, 5, 10, 15, 20])
    cx.spines['left'].set_visible(False)
    cx.tick_params(axis='x', labelsize=8, pad=1, colors=C['text'] if k >= 4 else 'white')
    cx.spines['bottom'].set_alpha(a4)
    save_slot(fig, out / f'phases_step{k}.svg')
for k in (1, 2):
    fig = plt.figure(figsize=cm(14.2, 2.9))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    s = 1.5
    ax.add_patch(Rectangle((0, 1.0), 0.296119403 * s, 0.5, color=C['data']))
    ax.text(0.296119403 / 2 * s, 1.25, 'draft', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.text(0.0, 1.62, 'Draft proposes 4 tokens (0.296 ms); the target checks all 5 positions in one pass.', ha='left', va='bottom', fontsize=9.5, color=C['text'], fontweight='bold')
    ax.add_patch(Rectangle((0.296119403 * s, 1.0), 0.8955223881 * s, 0.5, color=C['memory']))
    ax.text((0.296119403 + 0.8955223881 / 2) * s, 1.25, 'target: 0.896 ms', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.text(1.191641791 * s + 0.05, 1.25, '1.192 ms per round', va='center', fontsize=9.5, fontweight='bold')
    a = 1.0 if k >= 2 else 0.0
    marks = [('t1', 'data', 'kept'), ('t2', 'data', 'kept'), ('t3', 'danger', 'rejected'), ('t4', 'neutral', 'dropped'), ('new', 'memory', "target's")]
    for j, (lab, ck, word) in enumerate(marks):
        x = j * 0.42
        ax.add_patch(Rectangle((x, 0.3), 0.36, 0.45, facecolor=C[ck + '_l'], edgecolor=C[ck], lw=1.5, alpha=a))
        ax.text(x + 0.18, 0.525, lab, ha='center', va='center', fontsize=9, fontweight='bold', color=C[ck], alpha=a)
        ax.text(x + 0.18, 0.1, word, ha='center', va='center', fontsize=8.5, color=C[ck], alpha=a)
    ax.text(2.2, 0.525, 'one example round: 3 tokens', va='center', fontsize=9.5, fontweight='bold', alpha=a)
    ax.set_xlim(0, 3.6)
    ax.set_ylim(-0.05, 1.95)
    hide_axis(ax)
    save_slot(fig, out / f'spec_step{k}.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    for i, b in enumerate((1, 32)):
        cache = b * 0.3145728
        ms = f"{ {1: '1.0', 32: '3.9'}[b]} ms"
        rate = f"{ {1: '1,011', 32: '8,204'}[b]} tokens/s"
        y = 1 - i
        a = 1.0 if k == 2 or b == 1 else 0.0
        ax.barh(y, 3.0, color=C['memory'], height=0.55, alpha=a)
        ax.barh(y, cache, left=3.0, color=C['data'], height=0.55, alpha=a)
        ax.text(3.0 + cache + 0.2, y, f"{ {1: '3.3', 32: '13.1'}[b]} GB: {ms}, {rate}", va='center', fontsize=11, fontweight='bold', color=C['text'], alpha=a)
        ax.text(-0.15, y, f'Batch {b}', ha='right', va='center', fontsize=12, fontweight='bold', alpha=a)
    ax.text(1.5, 1.45, 'weights 3 GB', ha='center', color=C['memory'], fontsize=11, fontweight='bold')
    ax.text(8.0, -0.55, 'KV cache, 0.31 GB per sequence', ha='center', va='top', color=C['data'], fontsize=11, fontweight='bold', alpha=1.0 if k == 2 else 0.0)
    ax.set_xlim(0, 22)
    ax.set_ylim(-0.9, 1.7)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Bytes read per decode step (GB; GPT-2 XL, 1,024-token context, modeled)')
    sp.save(fig, out / f'decode_bytes_step{k}.svg')
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=cm(14.2, 4.9))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.3)
    top = {1: 32, 2: 203, 3: 263}[k]
    bs = np.arange(1, top + 1)
    ax.plot(bs / {1: np.array([0.9894247164, 1.083327045, 1.177229373, 1.271131701, 1.36503403, 1.458936358, 1.552838687, 1.646741015, 1.740643343, 1.834545672, 1.928448, 2.022350328, 2.116252657, 2.210154985, 2.304057313, 2.397959642, 2.49186197, 2.585764299, 2.679666627, 2.773568955, 2.867471284, 2.961373612, 3.05527594, 3.149178269, 3.243080597, 3.336982925, 3.430885254, 3.524787582, 3.61868991, 3.712592239, 3.806494567, 3.900396896]), 2: np.array([0.9894247164, 1.083327045, 1.177229373, 1.271131701, 1.36503403, 1.458936358, 1.552838687, 1.646741015, 1.740643343, 1.834545672, 1.928448, 2.022350328, 2.116252657, 2.210154985, 2.304057313, 2.397959642, 2.49186197, 2.585764299, 2.679666627, 2.773568955, 2.867471284, 2.961373612, 3.05527594, 3.149178269, 3.243080597, 3.336982925, 3.430885254, 3.524787582, 3.61868991, 3.712592239, 3.806494567, 3.900396896, 3.994299224, 4.088201552, 4.182103881, 4.276006209, 4.369908537, 4.463810866, 4.557713194, 4.651615522, 4.745517851, 4.839420179, 4.933322507, 5.027224836, 5.121127164, 5.215029493, 5.308931821, 5.402834149, 5.496736478, 5.590638806, 5.684541134, 5.778443463, 5.872345791, 5.966248119, 6.060150448, 6.154052776, 6.247955104, 6.341857433, 6.435759761, 6.52966209, 6.623564418, 6.717466746, 6.811369075, 6.905271403, 6.999173731, 7.09307606, 7.186978388, 7.280880716, 7.374783045, 7.468685373, 7.562587701, 7.65649003, 7.750392358, 7.844294687, 7.938197015, 8.032099343, 8.126001672, 8.219904, 8.313806328, 8.407708657, 8.501610985, 8.595513313, 8.689415642, 8.78331797, 8.877220299, 8.971122627, 9.065024955, 9.158927284, 9.252829612, 9.34673194, 9.440634269, 9.534536597, 9.628438925, 9.722341254, 9.816243582, 9.91014591, 10.00404824, 10.09795057, 10.1918529, 10.28575522, 10.37965755, 10.47355988, 10.56746221, 10.66136454, 10.75526687, 10.84916919, 10.94307152, 11.03697385, 11.13087618, 11.22477851, 11.31868084, 11.41258316, 11.50648549, 11.60038782, 11.69429015, 11.78819248, 11.88209481, 11.97599713, 12.06989946, 12.16380179, 12.25770412, 12.35160645, 12.44550878, 12.5394111, 12.63331343, 12.72721576, 12.82111809, 12.91502042, 13.00892275, 13.10282507, 13.1967274, 13.29062973, 13.38453206, 13.47843439, 13.57233672, 13.66623904, 13.76014137, 13.8540437, 13.94794603, 14.04184836, 14.13575069, 14.22965301, 14.32355534, 14.41745767, 14.51136, 14.60526233, 14.69916466, 14.79306699, 14.88696931, 14.98087164, 15.07477397, 15.1686763, 15.26257863, 15.35648096, 15.45038328, 15.54428561, 15.63818794, 15.73209027, 15.8259926, 15.91989493, 16.01379725, 16.10769958, 16.20160191, 16.29550424, 16.38940657, 16.4833089, 16.57721122, 16.67111355, 16.76501588, 16.85891821, 16.95282054, 17.04672287, 17.14062519, 17.23452752, 17.32842985, 17.42233218, 17.51623451, 17.61013684, 17.70403916, 17.79794149, 17.89184382, 17.98574615, 18.07964848, 18.17355081, 18.26745313, 18.36135546, 18.45525779, 18.54916012, 18.64306245, 18.73696478, 18.8308671, 18.92476943, 19.01867176, 19.11257409, 19.20647642, 19.30037875, 19.39428107, 19.4881834, 19.58208573, 19.67598806, 19.76989039, 19.86379272, 19.95769504]), 3: np.array([0.9894247164, 1.083327045, 1.177229373, 1.271131701, 1.36503403, 1.458936358, 1.552838687, 1.646741015, 1.740643343, 1.834545672, 1.928448, 2.022350328, 2.116252657, 2.210154985, 2.304057313, 2.397959642, 2.49186197, 2.585764299, 2.679666627, 2.773568955, 2.867471284, 2.961373612, 3.05527594, 3.149178269, 3.243080597, 3.336982925, 3.430885254, 3.524787582, 3.61868991, 3.712592239, 3.806494567, 3.900396896, 3.994299224, 4.088201552, 4.182103881, 4.276006209, 4.369908537, 4.463810866, 4.557713194, 4.651615522, 4.745517851, 4.839420179, 4.933322507, 5.027224836, 5.121127164, 5.215029493, 5.308931821, 5.402834149, 5.496736478, 5.590638806, 5.684541134, 5.778443463, 5.872345791, 5.966248119, 6.060150448, 6.154052776, 6.247955104, 6.341857433, 6.435759761, 6.52966209, 6.623564418, 6.717466746, 6.811369075, 6.905271403, 6.999173731, 7.09307606, 7.186978388, 7.280880716, 7.374783045, 7.468685373, 7.562587701, 7.65649003, 7.750392358, 7.844294687, 7.938197015, 8.032099343, 8.126001672, 8.219904, 8.313806328, 8.407708657, 8.501610985, 8.595513313, 8.689415642, 8.78331797, 8.877220299, 8.971122627, 9.065024955, 9.158927284, 9.252829612, 9.34673194, 9.440634269, 9.534536597, 9.628438925, 9.722341254, 9.816243582, 9.91014591, 10.00404824, 10.09795057, 10.1918529, 10.28575522, 10.37965755, 10.47355988, 10.56746221, 10.66136454, 10.75526687, 10.84916919, 10.94307152, 11.03697385, 11.13087618, 11.22477851, 11.31868084, 11.41258316, 11.50648549, 11.60038782, 11.69429015, 11.78819248, 11.88209481, 11.97599713, 12.06989946, 12.16380179, 12.25770412, 12.35160645, 12.44550878, 12.5394111, 12.63331343, 12.72721576, 12.82111809, 12.91502042, 13.00892275, 13.10282507, 13.1967274, 13.29062973, 13.38453206, 13.47843439, 13.57233672, 13.66623904, 13.76014137, 13.8540437, 13.94794603, 14.04184836, 14.13575069, 14.22965301, 14.32355534, 14.41745767, 14.51136, 14.60526233, 14.69916466, 14.79306699, 14.88696931, 14.98087164, 15.07477397, 15.1686763, 15.26257863, 15.35648096, 15.45038328, 15.54428561, 15.63818794, 15.73209027, 15.8259926, 15.91989493, 16.01379725, 16.10769958, 16.20160191, 16.29550424, 16.38940657, 16.4833089, 16.57721122, 16.67111355, 16.76501588, 16.85891821, 16.95282054, 17.04672287, 17.14062519, 17.23452752, 17.32842985, 17.42233218, 17.51623451, 17.61013684, 17.70403916, 17.79794149, 17.89184382, 17.98574615, 18.07964848, 18.17355081, 18.26745313, 18.36135546, 18.45525779, 18.54916012, 18.64306245, 18.73696478, 18.8308671, 18.92476943, 19.01867176, 19.11257409, 19.20647642, 19.30037875, 19.39428107, 19.4881834, 19.58208573, 19.67598806, 19.76989039, 19.86379272, 19.95769504, 20.05159737, 20.1454997, 20.23940203, 20.33330436, 20.42720669, 20.52110901, 20.61501134, 20.70891367, 20.802816, 20.89671833, 20.99062066, 21.08452299, 21.17842531, 21.27232764, 21.36622997, 21.4601323, 21.55403463, 21.64793696, 21.74183928, 21.83574161, 21.92964394, 22.02354627, 22.1174486, 22.21135093, 22.30525325, 22.39915558, 22.49305791, 22.58696024, 22.68086257, 22.7747649, 22.86866722, 22.96256955, 23.05647188, 23.15037421, 23.24427654, 23.33817887, 23.43208119, 23.52598352, 23.61988585, 23.71378818, 23.80769051, 23.90159284, 23.99549516, 24.08939749, 24.18329982, 24.27720215, 24.37110448, 24.46500681, 24.55890913, 24.65281146, 24.74671379, 24.84061612, 24.93451845, 25.02842078, 25.1223231, 25.21622543, 25.31012776, 25.40403009, 25.49793242, 25.59183475])}[k] * 1000, {1: np.array([0.9894247164, 1.083327045, 1.177229373, 1.271131701, 1.36503403, 1.458936358, 1.552838687, 1.646741015, 1.740643343, 1.834545672, 1.928448, 2.022350328, 2.116252657, 2.210154985, 2.304057313, 2.397959642, 2.49186197, 2.585764299, 2.679666627, 2.773568955, 2.867471284, 2.961373612, 3.05527594, 3.149178269, 3.243080597, 3.336982925, 3.430885254, 3.524787582, 3.61868991, 3.712592239, 3.806494567, 3.900396896]), 2: np.array([0.9894247164, 1.083327045, 1.177229373, 1.271131701, 1.36503403, 1.458936358, 1.552838687, 1.646741015, 1.740643343, 1.834545672, 1.928448, 2.022350328, 2.116252657, 2.210154985, 2.304057313, 2.397959642, 2.49186197, 2.585764299, 2.679666627, 2.773568955, 2.867471284, 2.961373612, 3.05527594, 3.149178269, 3.243080597, 3.336982925, 3.430885254, 3.524787582, 3.61868991, 3.712592239, 3.806494567, 3.900396896, 3.994299224, 4.088201552, 4.182103881, 4.276006209, 4.369908537, 4.463810866, 4.557713194, 4.651615522, 4.745517851, 4.839420179, 4.933322507, 5.027224836, 5.121127164, 5.215029493, 5.308931821, 5.402834149, 5.496736478, 5.590638806, 5.684541134, 5.778443463, 5.872345791, 5.966248119, 6.060150448, 6.154052776, 6.247955104, 6.341857433, 6.435759761, 6.52966209, 6.623564418, 6.717466746, 6.811369075, 6.905271403, 6.999173731, 7.09307606, 7.186978388, 7.280880716, 7.374783045, 7.468685373, 7.562587701, 7.65649003, 7.750392358, 7.844294687, 7.938197015, 8.032099343, 8.126001672, 8.219904, 8.313806328, 8.407708657, 8.501610985, 8.595513313, 8.689415642, 8.78331797, 8.877220299, 8.971122627, 9.065024955, 9.158927284, 9.252829612, 9.34673194, 9.440634269, 9.534536597, 9.628438925, 9.722341254, 9.816243582, 9.91014591, 10.00404824, 10.09795057, 10.1918529, 10.28575522, 10.37965755, 10.47355988, 10.56746221, 10.66136454, 10.75526687, 10.84916919, 10.94307152, 11.03697385, 11.13087618, 11.22477851, 11.31868084, 11.41258316, 11.50648549, 11.60038782, 11.69429015, 11.78819248, 11.88209481, 11.97599713, 12.06989946, 12.16380179, 12.25770412, 12.35160645, 12.44550878, 12.5394111, 12.63331343, 12.72721576, 12.82111809, 12.91502042, 13.00892275, 13.10282507, 13.1967274, 13.29062973, 13.38453206, 13.47843439, 13.57233672, 13.66623904, 13.76014137, 13.8540437, 13.94794603, 14.04184836, 14.13575069, 14.22965301, 14.32355534, 14.41745767, 14.51136, 14.60526233, 14.69916466, 14.79306699, 14.88696931, 14.98087164, 15.07477397, 15.1686763, 15.26257863, 15.35648096, 15.45038328, 15.54428561, 15.63818794, 15.73209027, 15.8259926, 15.91989493, 16.01379725, 16.10769958, 16.20160191, 16.29550424, 16.38940657, 16.4833089, 16.57721122, 16.67111355, 16.76501588, 16.85891821, 16.95282054, 17.04672287, 17.14062519, 17.23452752, 17.32842985, 17.42233218, 17.51623451, 17.61013684, 17.70403916, 17.79794149, 17.89184382, 17.98574615, 18.07964848, 18.17355081, 18.26745313, 18.36135546, 18.45525779, 18.54916012, 18.64306245, 18.73696478, 18.8308671, 18.92476943, 19.01867176, 19.11257409, 19.20647642, 19.30037875, 19.39428107, 19.4881834, 19.58208573, 19.67598806, 19.76989039, 19.86379272, 19.95769504]), 3: np.array([0.9894247164, 1.083327045, 1.177229373, 1.271131701, 1.36503403, 1.458936358, 1.552838687, 1.646741015, 1.740643343, 1.834545672, 1.928448, 2.022350328, 2.116252657, 2.210154985, 2.304057313, 2.397959642, 2.49186197, 2.585764299, 2.679666627, 2.773568955, 2.867471284, 2.961373612, 3.05527594, 3.149178269, 3.243080597, 3.336982925, 3.430885254, 3.524787582, 3.61868991, 3.712592239, 3.806494567, 3.900396896, 3.994299224, 4.088201552, 4.182103881, 4.276006209, 4.369908537, 4.463810866, 4.557713194, 4.651615522, 4.745517851, 4.839420179, 4.933322507, 5.027224836, 5.121127164, 5.215029493, 5.308931821, 5.402834149, 5.496736478, 5.590638806, 5.684541134, 5.778443463, 5.872345791, 5.966248119, 6.060150448, 6.154052776, 6.247955104, 6.341857433, 6.435759761, 6.52966209, 6.623564418, 6.717466746, 6.811369075, 6.905271403, 6.999173731, 7.09307606, 7.186978388, 7.280880716, 7.374783045, 7.468685373, 7.562587701, 7.65649003, 7.750392358, 7.844294687, 7.938197015, 8.032099343, 8.126001672, 8.219904, 8.313806328, 8.407708657, 8.501610985, 8.595513313, 8.689415642, 8.78331797, 8.877220299, 8.971122627, 9.065024955, 9.158927284, 9.252829612, 9.34673194, 9.440634269, 9.534536597, 9.628438925, 9.722341254, 9.816243582, 9.91014591, 10.00404824, 10.09795057, 10.1918529, 10.28575522, 10.37965755, 10.47355988, 10.56746221, 10.66136454, 10.75526687, 10.84916919, 10.94307152, 11.03697385, 11.13087618, 11.22477851, 11.31868084, 11.41258316, 11.50648549, 11.60038782, 11.69429015, 11.78819248, 11.88209481, 11.97599713, 12.06989946, 12.16380179, 12.25770412, 12.35160645, 12.44550878, 12.5394111, 12.63331343, 12.72721576, 12.82111809, 12.91502042, 13.00892275, 13.10282507, 13.1967274, 13.29062973, 13.38453206, 13.47843439, 13.57233672, 13.66623904, 13.76014137, 13.8540437, 13.94794603, 14.04184836, 14.13575069, 14.22965301, 14.32355534, 14.41745767, 14.51136, 14.60526233, 14.69916466, 14.79306699, 14.88696931, 14.98087164, 15.07477397, 15.1686763, 15.26257863, 15.35648096, 15.45038328, 15.54428561, 15.63818794, 15.73209027, 15.8259926, 15.91989493, 16.01379725, 16.10769958, 16.20160191, 16.29550424, 16.38940657, 16.4833089, 16.57721122, 16.67111355, 16.76501588, 16.85891821, 16.95282054, 17.04672287, 17.14062519, 17.23452752, 17.32842985, 17.42233218, 17.51623451, 17.61013684, 17.70403916, 17.79794149, 17.89184382, 17.98574615, 18.07964848, 18.17355081, 18.26745313, 18.36135546, 18.45525779, 18.54916012, 18.64306245, 18.73696478, 18.8308671, 18.92476943, 19.01867176, 19.11257409, 19.20647642, 19.30037875, 19.39428107, 19.4881834, 19.58208573, 19.67598806, 19.76989039, 19.86379272, 19.95769504, 20.05159737, 20.1454997, 20.23940203, 20.33330436, 20.42720669, 20.52110901, 20.61501134, 20.70891367, 20.802816, 20.89671833, 20.99062066, 21.08452299, 21.17842531, 21.27232764, 21.36622997, 21.4601323, 21.55403463, 21.64793696, 21.74183928, 21.83574161, 21.92964394, 22.02354627, 22.1174486, 22.21135093, 22.30525325, 22.39915558, 22.49305791, 22.58696024, 22.68086257, 22.7747649, 22.86866722, 22.96256955, 23.05647188, 23.15037421, 23.24427654, 23.33817887, 23.43208119, 23.52598352, 23.61988585, 23.71378818, 23.80769051, 23.90159284, 23.99549516, 24.08939749, 24.18329982, 24.27720215, 24.37110448, 24.46500681, 24.55890913, 24.65281146, 24.74671379, 24.84061612, 24.93451845, 25.02842078, 25.1223231, 25.21622543, 25.31012776, 25.40403009, 25.49793242, 25.59183475])}[k], color=C['memory'], lw=2.5)
    pts = [(1, f'batch {1}'), (8, f'{8}'), (32, f'{32}'), (64, f'{64}'), (128, f'{128}'), (203, f'{203}'), (263, '')]
    for b, lab in pts:
        if b > top:
            continue
        x, y = (b / {'batch 1': 0.9894247164, '8': 1.646741015, '32': 3.900396896, '64': 6.905271403, '128': 12.91502042, '203': 19.95769504, '': 25.59183475}[lab] * 1000, {'batch 1': 0.9894247164, '8': 1.646741015, '32': 3.900396896, '64': 6.905271403, '128': 12.91502042, '203': 19.95769504, '': 25.59183475}[lab])
        derived = b > 32
        ax.plot([x], [y], 'o', ms=6, color=C['neutral'] if derived else C['memory'], mec='white', zorder=5)
        ax.annotate(lab, (x, y), xytext=(-7, 3), textcoords='offset points', ha='right', fontsize=9, fontweight='bold', color=C['neutral'] if derived else C['memory'])
    if k >= 2:
        ax.axhline(20, color=C['danger'], lw=2, ls='--')
        ax.text(200, 21, f'TPOT target 20 ms: at most batch {203}', fontsize=9, color=C['danger'], fontweight='bold')
        ax.axvline(10649.3632, color=C['neutral'], lw=1.5, ls=':')
        ax.text(10649.3632 + 150, 8, 'ceiling:\n10,649 tokens/s', ha='left', va='bottom', fontsize=9, color=C['neutral'], fontweight='bold')
    if k >= 3:
        ax.text(10649.3632 + 150, 25.0, f'{263}: memory full', ha='left', fontsize=9, color=C['danger'], fontweight='bold')
    ax.set_xlim(0, 14200)
    ax.set_ylim(0, 30)
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
    ax.set_xticklabels(['0', '2K', '4K', '6K', '8K', '10K', '12K'])
    ax.tick_params(labelsize=8.5, pad=1)
    ax.set_xlabel('Aggregate tokens/s', fontsize=9, labelpad=1)
    ax.set_ylabel('TPOT (ms)', fontsize=9, labelpad=1)
    tag(ax, f'GPT-2 XL, one H100, full 1,024-token contexts. Book: batch 1 and {32}; gray: derived (bandwidth floor).', y=-0.3)
    for t in ax.texts[-1:]:
        t.set_fontsize(8)
    save_slot(fig, out / f'frontier_step{k}.svg')
REQ1 = [('A', 10), ('B', 100), ('C', 40), ('D', 60)]
REQ2 = {'E': 30, 'F': 50, 'G': 20, 'H': 40}
LONGEST = max((L for _, L in REQ1))
END = 150
WAITING = 'waiting:\n' + ', '.join((f'{n} {L}' for n, L in list(REQ2.items())[:2])) + ',\n' + ', '.join((f'{n} {L}' for n, L in list(REQ2.items())[2:]))

def grid(ax, title):
    for s in range(4):
        ax.axhspan(3 - s - 0.42, 3 - s + 0.42, xmin=0, xmax=1, color='#F4F4F4', lw=0, zorder=0)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(['slot 1', 'slot 2', 'slot 3', 'slot 4'], fontsize=9)
    ax.tick_params(axis='y', length=0, pad=2)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, END)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xticks(sorted({0, END, *(L for _, L in REQ1)}))
    ax.tick_params(axis='x', labelsize=8.5, pad=1)
    ax.set_xlabel('Decode step (lengths illustrative; one token per step, prefill omitted)', fontsize=8.5, labelpad=1)
    ax.set_title(title, fontsize=9.5, loc='left', fontweight='bold', pad=3)

def bar(ax, slot, start, length, name, ck):
    ax.barh(3 - slot, length, left=start, height=0.7, color=C[ck], edgecolor='white', lw=1.5)
    ax.text(start + length / 2, 3 - slot, f'{name} {length}' if length >= 18 else name, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

def idle(ax, slot, start, length):
    ax.barh(3 - slot, length, left=start, height=0.7, color=C['danger_l'], edgecolor=C['danger'], hatch='//', lw=0)

def summary(ax, text, color):
    ax.text(1.0, -0.3, text, ha='right', va='top', fontsize=9, fontweight='bold', color=color, transform=ax.transAxes)
used1 = 210
for k in range(1, 8):
    h = 6.0 if k <= 3 else 4.9
    fig, ax = plt.subplots(figsize=cm(14.2, h))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.9, bottom=0.27 if k <= 3 else 0.31)
    if k <= 3:
        grid(ax, 'Request-level: the batch is fixed until its longest member ends')
        for s, (n, L) in enumerate(REQ1):
            bar(ax, s, 0, L, n, 'memory')
        if k == 1:
            idle(ax, 0, 10, LONGEST - 10)
        if k >= 2:
            for s, (n, L) in enumerate(REQ1):
                if L < LONGEST:
                    idle(ax, s, L, LONGEST - L)
            ax.text(55, 3, f'{used1} of {400} slot-steps used', ha='center', va='center', fontsize=9, color=C['danger'], fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
        if k == 3:
            for s, n in enumerate('EFGH'):
                bar(ax, s, {0: 100, 1: 100, 2: 100, 3: 100}[s], REQ2[n], n, 'data')
            summary(ax, 'done at step 150; 350 of 600 slot-steps used (58%)', C['danger'])
        else:
            ax.text(125, 1.5, WAITING, ha='center', va='center', fontsize=9, color=C['data'], fontweight='bold')
    else:
        grid(ax, 'Continuous: a freed slot takes the next request at the next step')
        for s, (n, L) in enumerate(REQ1):
            bar(ax, s, 0, L, n, 'memory')
        if k == 4:
            ax.text(125, 1.5, WAITING, ha='center', va='center', fontsize=9, color=C['data'], fontweight='bold')
        cb = {'E': (10, 30), 'F': (40, 50), 'G': (40, 20), 'H': (60, 40)}
        if k >= 5:
            bar(ax, 0, *cb['E'], 'E', 'data')
        if k >= 6:
            bar(ax, 2, *cb['F'], 'F', 'data')
            bar(ax, 0, *cb['G'], 'G', 'data')
            bar(ax, 0, *cb['H'], 'H', 'data')
            idle(ax, 2, 90, 100 - 90)
            idle(ax, 3, 60, 100 - 60)
        if k >= 7:
            summary(ax, 'done at step 100; 350 of 400 slot-steps used (87.5%)', C['data'])
    save_slot(fig, out / f'cb_step{k}.svg')
DEC = 2.771058455
PRE = 41.93186047
for k in (1, 2, 3):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=cm(9.9, 4.4), sharey=True, gridspec_kw={'wspace': 0.1})
    fig.subplots_adjust(left=0.2, right=0.99, top=0.88, bottom=0.12)
    left = [DEC] * 4 + [DEC + PRE] + [DEC] * 4
    for i, v in enumerate(left):
        tall = i == 4
        a = 1.0 if not tall or k >= 2 else 0.0
        ax.bar(i, v, color=C['compute'] if tall else C['memory'], width=0.75, alpha=a)
    ax.text(4, DEC + PRE + 1.5, '44.7 ms', ha='center', fontsize=9, fontweight='bold', color=C['compute'], alpha=1.0 if k >= 2 else 0.0)
    ax.set_title('Whole prompt in one step', fontsize=8.5, loc='left', fontweight='bold', pad=3)
    right = [DEC] * 2 + [DEC + PRE / 4] * 4 + [DEC] * 3
    a3 = 1.0 if k >= 3 else 0.0
    for i, v in enumerate(right):
        chunk = 2 <= i <= 5
        bx.bar(i, v, color=C['compute'] if chunk else C['memory'], width=0.75, alpha=a3)
    bx.text(4.0, 27, f'{4} chunks of 250:\n13.3 ms each', ha='center', fontsize=8.5, fontweight='bold', color=C['compute'], alpha=a3)
    bx.set_title('Four chunks', fontsize=8.5, loc='left', fontweight='bold', alpha=a3, pad=3)
    for a_, show in ((ax, True), (bx, k >= 3)):
        a_.axhline(20, color=C['danger'], lw=1.8, ls='--', alpha=1.0 if show else 0.0)
        a_.set_xticks([])
        a_.set_xlabel('decode steps', fontsize=8.5, alpha=1.0 if show else 0.0, labelpad=2)
        a_.set_ylim(0, 52)
        if not show:
            a_.spines['bottom'].set_alpha(0.0)
    ax.tick_params(axis='y', labelsize=8, pad=1)
    ax.set_yticks([0, 20, 2 * 20])
    ax.set_yticklabels(['0', 'TPOT 20', f'{2 * 20:.0f}'])
    ax.get_yticklabels()[1].set_color(C['danger'])
    ax.get_yticklabels()[1].set_fontweight('bold')
    ax.text(0, DEC + 1.5, '2.8 ms', ha='left', fontsize=8.5, color=C['memory'], fontweight='bold')
    ax.set_ylabel('Step time (ms)', fontsize=8.5, labelpad=1)
    save_slot(fig, out / f'stall_step{k}.svg')
ctx8 = np.linspace(0, 16384, 200)
fig, ax = plt.subplots(figsize=cm(8.7, 4.6))
fig.subplots_adjust(left=0.13, right=0.8, top=0.88, bottom=0.21)
FREE_GB = 69.83934592
ax.axhspan(FREE_GB, 125, color=C['danger_l'], alpha=0.6, lw=0)
ax.axhline(FREE_GB, color=C['danger'], lw=2, ls='--')
ax.text(16200, FREE_GB + 2, 'budget: 69.8 GB', ha='right', va='bottom', fontsize=9, color=C['danger'], fontweight='bold')
N8 = 65
for b, ck in ((16, 'data'), (32, 'compute'), (N8, 'network')):
    kv = b * ctx8 * 131072 / 1000000000.0
    ax.plot(ctx8, kv, color=C[ck], lw=2.5)
    if b < N8:
        ax.text(16600, b * 16384 * 131072 / 1000000000.0, f'{b} chats', ha='left', va='center', fontsize=9, color=C[ck], fontweight='bold')
ax.plot([8192], [N8 * 1.073741824], 'o', ms=7, color=C['network'], mec='white', mew=1.5, zorder=5)
ax.text(300, 97, f'{N8} chats:\nfull at 8,192 tokens', ha='left', va='top', fontsize=9, color=C['network'], fontweight='bold')
ax.set_xlim(0, 16384)
ax.set_ylim(0, 100)
ax.set_xticks([0, 4096, 8192, 12288, 16384])
ax.set_xticklabels(['0', '4K', '8K', '12K', '16K'])
ax.tick_params(labelsize=8, pad=1)
ax.set_xlabel('Context length (tokens)', fontsize=8.5, labelpad=1)
ax.set_ylabel('KV cache (GB, FP16)', fontsize=8.5, labelpad=1)
ax.set_title('Llama 3 8B, 16-bit weights, one H100', fontsize=9, fontweight='bold', loc='left', pad=3)
save_slot(fig, out / 'kv_budget_8b.svg')
ctx = np.linspace(0, 32768, 200)
fig, ax = plt.subplots(figsize=cm(8.7, 4.6))
fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.21)
B70 = 50.59934592
ax.axhspan(B70, 125, color=C['danger_l'], alpha=0.6, lw=0)
ax.axhline(B70, color=C['danger'], lw=2, ls='--')
ax.text(32000, B70 + 2.5, 'budget: 50.6 GB', ha='right', va='bottom', fontsize=8.5, color=C['danger'], fontweight='bold')
for b, ck in ((1, 'memory'), (4, 'data'), (16, 'compute'), (32, 'network')):
    kv = b * ctx * 327680 / 1000000000.0
    ax.plot(ctx, kv, color=C[ck], lw=2.5)
    if b > 4:
        xe = 102000000000.0 / (b * 327680)
        lx, ly = (xe * 0.96, 102)
    else:
        lx = 24000 if b == 4 else 32000
        ly = b * lx * 327680 / 1000000000.0 + 3
    ax.text(lx, ly, f'batch {b}', ha='right', fontsize=8.5, color=C[ck], fontweight='bold')
x32 = 4825.529663
ax.plot([x32], [B70], 'o', ms=7, color=C['network'], mec='white', mew=1.5, zorder=5)
ax.annotate('32 chats full near\n4,800 tokens', (x32, B70), xytext=(19500, 72), fontsize=8.5, color=C['network'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=C['network']))
ax.set_xlim(0, 32768)
ax.set_ylim(0, 120)
ax.set_xticks([0, 8192, 16384, 24576, 32768])
ax.set_xticklabels(['0', '8K', '16K', '24K', '32K'])
ax.tick_params(labelsize=8, pad=1)
ax.set_xlabel('Context length (tokens)', fontsize=8.5, labelpad=1)
ax.set_ylabel('KV cache (GB, FP16)', fontsize=8.5, labelpad=1)
ax.set_title('Larger model: 70B class, 4-bit weights, one H100', fontsize=9, fontweight='bold', loc='left', pad=3)
save_slot(fig, out / 'kv_budget.svg')
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.9, 1.9))
    y0, h = (0.0, 0.8)

    def seg(x, w, fc, ec, hatch=None, label=None, tc='white', a=1.0):
        ax.add_patch(Rectangle((x, y0), w, h, facecolor=fc, edgecolor=ec, hatch=hatch, lw=1.2, alpha=a))
        if label:
            ax.text(x + w / 2, y0 + h / 2, label, ha='center', va='center', fontsize=10, fontweight='bold', color=tc, alpha=a)
    ax.add_patch(Rectangle((0, y0), 600, h, facecolor='white', edgecolor=C['neutral'], lw=1.2))
    seg(0, 70, C['data'], 'white', label='A: 70')
    seg(70, 130, C['compute_l'], C['compute'], hatch='//')
    ax.text(135, -0.25, "reserved for A's maximum", ha='center', va='center', fontsize=10, color=C['compute'], fontweight='bold')
    a2 = 1.0 if k >= 2 else 0.0
    seg(300, 120, C['data'], 'white', label='B: 120', a=a2)
    seg(420, 80, C['danger_l'], C['danger'], hatch='xx', a=a2)
    ax.text(460, -0.25, 'internal: B stops at 120', ha='center', va='center', fontsize=10, color=C['danger'], fontweight='bold', alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    ax.add_patch(Rectangle((200, y0), 100, h, facecolor=C['neutral_l'], edgecolor=C['neutral'], lw=1.2, alpha=a3))
    ax.text(250, y0 + h / 2, 'free 100', ha='center', va='center', fontsize=10, fontweight='bold', color=C['text'], alpha=a3)
    ax.add_patch(Rectangle((175, 1.3), 150, 0.5, facecolor=C['memory_l'], edgecolor=C['memory'], lw=1.2, alpha=a3))
    ax.text(250, 1.55, 'new request: 150', ha='center', va='center', fontsize=10, fontweight='bold', color=C['memory'], alpha=a3)
    ax.text(335, 1.55, 'external: the hole is too small', va='center', fontsize=10, color=C['danger'], fontweight='bold', alpha=a3)
    ax.set_xlim(-5, 605)
    ax.set_ylim(-0.5, 2.0)
    hide_axis(ax)
    sp.save(fig, out / f'frag_step{k}.svg')
names = list({'Orca (Max)': 20.4, 'Orca (Pow2)': 26.8, 'Orca (Oracle)': 38.2, 'vLLM': 96.3})
used = [20.4, 26.8, 38.2, 96.3]
fig, ax = plt.subplots(figsize=cm(12.0, 2.7))
fig.subplots_adjust(left=0.17, right=0.97, top=0.98, bottom=0.3)
for i, n in enumerate(names):
    y = 3 - i
    ck = 'data' if n == 'vLLM' else 'neutral'
    ax.barh(y, used[i], color=C[ck], height=0.66)
    ax.text(used[i] + 1.5, y, f'{used[i]:g}%', va='center', fontsize=8.5, fontweight='bold', color=C[ck])
    ax.text(-1.5, y, n, ha='right', va='center', fontsize=8.5, fontweight='bold')
ax.set_xlim(0, 110)
ax.set_yticks([])
ax.set_ylim(-0.5, 3.5)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=8, pad=1)
ax.set_xlabel('Measured share of KV memory holding token states (%)', fontsize=8.5, labelpad=1)
save_slot(fig, out / 'vllm_used.svg')
A_TOK = ['Four', 'score', 'and', 'seven', 'years', 'ago', 'our', 'fathers', 'brought']
B_TOK = ['It', 'was', 'the', 'best', 'of', 'times']
A_PHYS = [7, 1, 3]
B_PHYS = [5, 2]
CW, CH = (0.6, 0.32)
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.9, 3.0))
    na = {1: 7, 2: 8, 3: 9, 4: 9}[k]
    new_tok = {2: 7, 3: 8}.get(k, -1)
    ax.text(0.05, 3.55, 'Request A (logical)', fontsize=10.5, fontweight='bold', color=C['memory'])
    for lb in range(3):
        y = 3.0 - lb * 0.5
        a = 1.0 if lb * 4 < na else 0.0
        ax.text(-0.02, y + CH / 2, f'{lb}', ha='right', va='center', fontsize=10, alpha=a)
        for j in range(4):
            t = lb * 4 + j
            ax.add_patch(Rectangle((0.05 + j * CW, y), CW, CH, facecolor='white', edgecolor=C['neutral'], lw=1, alpha=a))
            if t < na:
                ax.text(0.05 + j * CW + CW / 2, y + CH / 2, A_TOK[t], ha='center', va='center', fontsize=8, color=C['danger'] if t == new_tok else C['text'], fontweight='bold' if t == new_tok else 'normal')
    ax.text(2.6, 3.55, 'Block table', fontsize=10.5, fontweight='bold')
    for lb in range(3):
        y = 3.0 - lb * 0.5
        a = 1.0 if lb * 4 < na else 0.0
        filled = max(0, min(4, na - lb * 4))
        changed = k == 2 and lb == 1 or (k == 3 and lb == 2)
        ax.add_patch(Rectangle((2.6, y), 1.45, CH, facecolor=C['memory_l'], edgecolor=C['memory'], lw=1, alpha=a))
        ax.text(3.325, y + CH / 2, f'{lb} → {A_PHYS[lb]}  ({filled} filled)', ha='center', va='center', fontsize=9, alpha=a, color=C['danger'] if changed else C['text'], fontweight='bold' if changed else 'normal')
    ax.text(4.45, 3.55, 'Physical blocks', fontsize=10.5, fontweight='bold')
    for pb in range(8):
        y = 3.05 - pb * 0.37
        ax.text(4.4, y + 0.14, f'{pb}', ha='right', va='center', fontsize=9.5)
        owner = None
        if pb in A_PHYS:
            owner = ('A', A_PHYS.index(pb))
        elif k >= 4 and pb in B_PHYS:
            owner = ('B', B_PHYS.index(pb))
        for j in range(4):
            fc, ec, word = ('white', C['neutral'], None)
            if owner:
                toks = A_TOK[:na] if owner[0] == 'A' else B_TOK
                t = owner[1] * 4 + j
                if t < len(toks):
                    word = toks[t]
                    fc = C['memory_l'] if owner[0] == 'A' else C['data_l']
                    ec = C['memory'] if owner[0] == 'A' else C['data']
            ax.add_patch(Rectangle((4.45 + j * CW, y), CW, 0.28, facecolor=fc, edgecolor=ec, lw=0.8))
            if word:
                ax.text(4.45 + j * CW + CW / 2, y + 0.14, word, ha='center', va='center', fontsize=8)
    captions = {1: '7-token prompt: 2 blocks, 1 slot free', 2: "Step 1: 'fathers' fills the free slot", 3: 'Step 2: a new block, anywhere free', 4: 'Request B shares the same pool'}
    ax.text(0.05, 1.3, captions[k], fontsize=10.5, fontweight='bold')
    ax.text(0.05, 0.9, 'Request B: logical 0 → 5, 1 → 2', fontsize=10.5, color=C['data'], fontweight='bold', alpha=1.0 if k >= 4 else 0.0)
    ax.text(0.05, 0.5, 'Block size 4 for legibility (vLLM default: 16)', fontsize=9, color=C['neutral'])
    ax.set_xlim(-0.2, 6.95)
    ax.set_ylim(0.3, 3.8)
    hide_axis(ax)
    sp.save(fig, out / f'paging_step{k}.svg')
COPIES = 26.2144
ONE = 0.262144
for k in (1, 2):
    fig = plt.figure(figsize=cm(7.6, 3.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    for j in range(5):
        ax.add_patch(Rectangle((0.15 + j * 0.62, 2.15), 0.55, 0.55, facecolor=C['network_l'], edgecolor=C['network'], lw=1.2))
    ax.text(0.15, 2.95, 'system prompt: 2,000 tokens, 262 MB, stored once', ha='left', fontsize=8.5, color=C['network'], fontweight='bold')
    for i in range(3):
        x = 0.15 + i * 1.05
        ax.add_patch(Rectangle((x, 0.55), 0.9, 0.55, facecolor=C['memory_l'], edgecolor=C['memory'], lw=1.2))
        ax.text(x + 0.45, 0.825, f'chat {i + 1}' if i < 2 else 'chat N', ha='center', va='center', fontsize=8.5)
        ax.add_patch(FancyArrowPatch((x + 0.45, 1.12), (1.65 + (x - 1.2) * 0.4, 2.12), arrowstyle='-|>', mutation_scale=9, color=C['network'], lw=1.2))
    ax.text(0.15, 0.12, 'each block table points at the shared blocks', ha='left', fontsize=8, color=C['neutral'])
    a = 1.0 if k >= 2 else 0.0
    bars = [('100 copies', COPIES, 'danger', '26.2 GB'), ('Shared', ONE, 'data', '0.26 GB')]
    for i, (lab, v, ck, txt) in enumerate(bars):
        y = 1.55 - i * 0.65
        ax.add_patch(Rectangle((4.75, y), v / COPIES * 1.4, 0.45, color=C[ck], alpha=a))
        ax.text(4.68, y + 0.22, lab, ha='right', va='center', fontsize=8.5, fontweight='bold', alpha=a)
        ax.text(4.75 + v / COPIES * 1.4 + 0.07, y + 0.22, txt, va='center', fontsize=8.5, fontweight='bold', color=C[ck], alpha=a)
    ax.text(7.45, 0.55, '100 conversations\n(derived)', ha='right', fontsize=8, color=C['neutral'], alpha=a)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(0, 3.4)
    hide_axis(ax)
    save_slot(fig, out / f'share_step{k}.svg')
fig, ax = plt.subplots(figsize=cm(9.3, 3.4))
fig.subplots_adjust(left=0.24, right=0.97, top=0.97, bottom=0.3)
DS = [('Both phases', 1.6, 'neutral'), ('Prefill only', 5.6, 'compute'), ('Decode only', 10.0, 'memory')]
for i, (lab, v, ck) in enumerate(DS):
    y = 2 - i
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v + 0.15, y, f'{v:g}', va='center', fontsize=9.5, fontweight='bold', color=C[ck])
    ax.text(-0.2, y, lab, ha='right', va='center', fontsize=9, fontweight='bold')
ax.set_xlim(0, 11.5)
ax.set_yticks([])
ax.set_ylim(-0.5, 2.5)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', labelsize=8, pad=1)
ax.set_xlabel('Requests/s per A100 meeting both deadlines\n(13B model, measured)', fontsize=8.5, labelpad=1)
save_slot(fig, out / 'distserve_bars.svg')
fig, ax = plt.subplots(figsize=(4.4, 2.3))
for i, (lab, ck) in enumerate((('prefill', 'compute'), ('prefill', 'compute'))):
    ax.add_patch(Rectangle((0.1, 1.3 - i * 1.0), 1.2, 0.7, facecolor=C[ck + '_l'], edgecolor=C[ck], lw=1.8))
    ax.text(0.7, 1.65 - i * 1.0, f'GPU {i + 1}\n{lab}: {5.6:g}/s', ha='center', va='center', fontsize=10.5, fontweight='bold', color=C[ck])
ax.add_patch(Rectangle((2.6, 0.8), 1.2, 0.7, facecolor=C['memory_l'], edgecolor=C['memory'], lw=1.8))
ax.text(3.2, 1.15, f'GPU 3\ndecode: {10.0:g}/s', ha='center', va='center', fontsize=10.5, fontweight='bold', color=C['memory'])
for yy in (1.65, 0.65):
    ax.add_patch(FancyArrowPatch((1.32, yy), (2.58, 1.15), arrowstyle='-|>', mutation_scale=12, color=C['network'], lw=1.6))
ax.text(1.95, 1.85, 'KV cache', ha='center', fontsize=10, color=C['network'], fontweight='bold')
ax.text(2.0, -0.05, '10 requests/s on 3 GPUs: 3.3 each, 2.1× colocated', ha='center', fontsize=10.5, fontweight='bold', color=C['data'])
ax.set_xlim(0, 4.0)
ax.set_ylim(-0.3, 2.1)
hide_axis(ax)
sp.save(fig, out / 'distserve_pool.svg')
sp.hbars(out / 'split_link.svg', [('Cache traffic at 10 req/s', 11.3, 'network', '11.3 GB/s'), ('NVLink, A100', 600.0, 'memory', '600 GB/s')], '', log=False, size=(5.0, 1.5), xlabel='GB/s (OPT-66B, 512-token prompts; DistServe)')
from datetime import date
rows = ({'model': 'Legacy GPT-3.5', 'date': '2022-11-30', 'usd_per_mtok_input': 20.0, 'source_key': 'openai2023chatgptwhisper', 'check': 'matches Epoch AI lowest-price series (GPT-3.5, 2022-11-30, $20); the cited post prices gpt-3.5-turbo at one tenth of existing GPT-3.5 models'}, {'model': 'GPT-3.5 Turbo', 'date': '2023-03-01', 'usd_per_mtok_input': 2.0, 'source_key': 'openai2023chatgptwhisper', 'check': 'matches Epoch AI lowest-price series ($2); openai.com refuses automated fetches'}, {'model': 'GPT-4 (Original)', 'date': '2023-03-14', 'usd_per_mtok_input': 30.0, 'source_key': 'openai2023gpt4web', 'check': '8K-context prompt price; checked 2026-10-13 against OpenAI API pricing (developers.openai.com/api/docs/pricing): gpt-4-0613 $30.00 input'}, {'model': 'Claude 3 Opus', 'date': '2024-03-04', 'usd_per_mtok_input': 15.0, 'source_key': 'anthropic2024claude3family', 'check': 'fetched 2026-10-13: $15 input per million tokens, post dated 2024-03-04'}, {'model': 'Claude 3 Haiku', 'date': '2024-03-04', 'usd_per_mtok_input': 0.25, 'source_key': 'anthropic2024claude3family', 'check': 'fetched 2026-10-13: $0.25 input per million tokens, post dated 2024-03-04'}, {'model': 'GPT-4o', 'date': '2024-05-13', 'usd_per_mtok_input': 5.0, 'source_key': 'openaiCommunity2024gpt4oApi', 'check': 'fetched 2026-10-13: $5 per million input tokens, post dated 2024-05-13'}, {'model': 'GPT-4o-mini', 'date': '2024-07-18', 'usd_per_mtok_input': 0.15, 'source_key': 'openai2024gpt4oMini', 'check': 'checked 2026-10-13 against OpenAI API pricing (developers.openai.com/api/docs/pricing): gpt-4o-mini $0.15 input'}, {'model': 'Gemini 1.5 Flash', 'date': '2024-08-12', 'usd_per_mtok_input': 0.075, 'source_key': 'google2024gemini15flashPricing', 'check': 'fetched 2026-10-13: $0.075 input for prompts up to 128K, effective 2024-08-12'}, {'model': 'DeepSeek-V3', 'date': '2025-02-08', 'usd_per_mtok_input': 0.27, 'source_key': 'deepseek2024v3', 'check': 'fetched 2026-10-13: $0.27 per million input tokens (cache miss) from Feb 8 2025, post dated 2024-12-26'})
OFFSET = {'Legacy GPT-3.5': (4, -11, 'left'), 'GPT-3.5 Turbo': (5, -10, 'left'), 'GPT-4 (Original)': (5, 2, 'left'), 'Claude 3 Opus': (5, 3, 'left'), 'Claude 3 Haiku': (-5, -2, 'right'), 'GPT-4o': (5, 2, 'left'), 'GPT-4o-mini': (5, -3, 'left'), 'Gemini 1.5 Flash': (-5, 3, 'right'), 'DeepSeek-V3': (-5, 5, 'right')}
for k in (1, 2):
    fig, ax = plt.subplots(figsize=cm(8.7, 5.1))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.12)
    for r in rows:
        d = date.fromisoformat(r['date'])
        x = d.year + (d.timetuple().tm_yday - 1) / 365.0
        y = float(r['usd_per_mtok_input'])
        ax.plot([x], [y], 'o', ms=5.5, color=C['memory'], mec='white', zorder=5)
        dx, dy, ha = OFFSET[r['model']]
        ax.annotate(r['model'], (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, fontsize=7.5, color=C['text'])
    if k >= 2:
        ax.axhline(0.06634819533, color=C['data'], lw=2.2)
        ax.text(2022.75, 0.06634819533 * 0.86, '\\$0.066: Llama 3 8B\ncase-study bound (derived)', ha='left', va='top', fontsize=8, color=C['data'], fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(0.02, 80)
    ax.set_xlim(2022.7, 2025.4)
    ax.set_xticks([2023, 2024, 2025])
    ax.set_xticklabels(['2023', '2024', '2025'])
    ax.set_yticks([0.1, 1, 10])
    ax.set_yticklabels(['\\$0.10', '\\$1', '\\$10'])
    ax.tick_params(labelsize=8, pad=1)
    ax.set_ylabel('\\$ per M input tokens', fontsize=8.5, labelpad=1)
    save_slot(fig, out / f'api_price_step{k}.svg')
sp.hbars(out / 'joules_token.svg', [('Batch 1, 1 kW', 0.9894247164, 'danger', '0.99 J per token'), ('Batch 1, 600 W', 0.5936548299, 'neutral', '0.59 J per token'), ('Batch 32, 1 kW', 0.121887403, 'data', '0.12 J per token')], '', log=False, value_fmt='{:.2f}', size=sp.SHORT, xlabel="GPT-2 XL decode, joules per token (derived; gray: the chapter's low-use power)")
print('figures written')
