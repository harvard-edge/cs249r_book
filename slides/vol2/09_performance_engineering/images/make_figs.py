#!/usr/bin/env python3
"""Plots and diagrams for Volume II, Lecture 9 (Fleet Efficiency).

Every plotted value and every number in a label comes from calcs.py, which
reads MLSysIM (val() gives the printed string, num() the value at its printed
precision); the runtime-scaling sweep and the MLPerf series are read from
their MLSysIM scenarios directly. Literals left here are layout (axis limits,
label positions, bar widths) or "schematic" shapes labeled as such on the
figure.

Run from this folder: python3 make_figs.py
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
from mlsysim.core.series_loader import load_series
_VALS = {'llama_gpus': '16,384', 'llama_days': '54', 'llama_interrupts': '419', 'llama_r_pct': '90', 'llama_r': '0.90', 'llama_p_pct': '41', 'llama_p': '0.41', 'llama_p_gap': '0.59', 'llama_r_loss_pct': '10', 'llama_p_loss_pct': '53', 'llama_rp_pct': '37', 'llama_rp': '0.37', 'llama_r_loss_frac_words': 'ten', 'llama_pool': '0.24', 'job_gpus': '128', 'job_nodes': '16', 'job_nodes_half': '8', 'node_gpus': '8', 'node_gpus_words': 'eight', 'tiers_alloc': '500', 'tiers_active': '350', 'tiers_prod': '280', 'tiers_prod_pct': '56', 'mtbf_h': '302', 'mtbf_days': '12.6', 'gain_gptj': '2.69', 'gain_gptj_1': '2.7', 'gain_llama2': '1.44', 'gain_llama2_1': '1.4', 'gain_sdxl': '1.25', 'gain_sdxl_1': '1.2', 'gain_flat': '1.00', 'gain_flat_1': '1.0', 'gain_months': '19', 'trace_gpus': '1,024', 'trace_s': '5', 'trace_mb': '500', 'trace_gb': '512', 'megascale_slow_pct': '0.5', 'cf_cpu_pct': '100', 'cf_min': '27', 'cf_drop_pct': '82', 'snap_assigned': '0.80', 'snap_partial': '0.15', 'snap_input': '0.05', 'snap_exec': '0.60', 'snap_free': '0.20', 'sched': '0.65', 'sched_ceiling': '0.80', 'google_s_pct': '95', 'google_s': '0.95', 'rt_loss_pct': '10', 'runtime': '0.90', 'runtime_gap': '0.10', 'tau_min': '77', 'tau_int_min': '21', 'preempt_per_day_words': 'one', 'preempt_min': '20.2', 'preempt_lost_min': '15', 'write_s': '9.8', 'term_input_pct': '7.7', 'term_preempt_pct': '1.4', 'term_ckpt_pct': '0.4', 'term_startup_pct': '0.05', 'term_recovery_pct': '0.03', 'startup_min': '5.2', 'startup_hour_pct': '9', 'cross_gpus': '13,000', 'rt_loss_16k_pct': '18', 'mfu_gpt3': '21.3', 'mfu_mtnlg': '30.2', 'mfu_gopher': '32.5', 'mfu_palm': '46.2', 'llama_p8_pct': '43', 'llama_p8': '0.43', 'llama_gpus_8k': '8,192', 'palm_chips': '6,144', 'llama_mfu_drop': '2', 'node_mfu_pct': '65', 'node_ms': '180', 'fleet_ms': '245', 'fleet_s': '0.245', 'extra_ms': '65', 'kept': '0.735', 'kept_pct': '74', 'fleet_mfu_est': '0.478', 'program': '0.48', 'program_pct': '48', 'useful_pflop': '14.9', 'peak_pflop': '31', 'h100_peak': '989', 'h100_bw': '3.35', 'ridge': '295', 'decode_flops_per_param': '2', 'fp16_bytes': '2', 'decode_ai': '1', 'decode_rate': '3.35', 'decode_peak_pct': '0.34', 'decode_peak_pct_1': '0.3', 'decode_ms': '5.2', 'ln_tflops': '5', 'ln_tbs': '2.8', 'ln_ai': '1.8', 'ln_bw_pct': '83.6', 'tp_ms': '45', 'dec_ar_kb': '32.8', 'dec_ar_x': '70', 'mpg': '0.28', 'mpg_3': '0.282', 'mpg_pct': '28', 'after_r': '0.59', 'job_rp': '0.43', 'program_gap': '0.52', 's_loss_pct': '35', 'r_loss_pct': '6', 'p_loss_pct': '31', 'lvl_s': '65', 'lvl_r': '59', 'runtime_ceiling': '0.98', 'program_ceiling': '0.65', 'lever_s_pct': '23', 'lever_s_mpg': '0.35', 'point_s': '0.43', 'lever_r_pct': '9', 'lever_r_mpg': '0.31', 'point_r': '0.31', 'lever_p_pct': '35', 'lever_p_mpg': '0.38', 'point_p': '0.59', 'price': '2', 'useful_price': '7.09', 'cost_x': '3.55', 'cost_x_1': '3.5', 'job_cost_x': '2.30', 'p_cut_pct': '26', 'r_headroom_pct': '11', 'trade_p': '1.10', 'trade_r_den': '12', 'trade_r_num': '11', 'trade_x': '1.008', 'cmp_rate': '120', 'cmp_step': '8.33', 'cmp_mm_pct': '50', 'cmp_ew_cut_pct': '70', 'cmp_ln_cut_pct': '80', 'cmp_mm': '0.50', 'cmp_ew': '0.35', 'cmp_ew_keep': '0.30', 'cmp_ln': '0.15', 'cmp_ln_keep': '0.20', 'cmp_ew_pct': '35', 'cmp_ln_pct': '15', 'cmp_frac': '0.635', 'cmp_new_ms': '5.3', 'cmp_saved_ms': '3.0', 'cmp_build_min': '15', 'cmp_tokens': '296,000', 'cmp_tokens_ch': '300,000', 'cmp_repay_min': '26', 'cmp_replicas': '128', 'cmp_shared_s': '12', 'cmp_rollout_h': '32', 'fr_dev_ratio': '2', 'fr_thr': '1.91', 'fr_time': '0.52', 'fr_cost': '1.049', 'fr_cost_pct': '4.9'}
_NUMS = {'llama_gpus': 16384.0, 'llama_days': 54.0, 'llama_interrupts': 419.0, 'llama_r_pct': 90.0, 'llama_r': 0.9, 'llama_p_pct': 41.0, 'llama_p': 0.41, 'llama_p_gap': 0.59, 'llama_r_loss_pct': 10.0, 'llama_p_loss_pct': 53.0, 'llama_rp_pct': 37.0, 'llama_rp': 0.37, 'llama_pool': 0.24, 'job_gpus': 128.0, 'job_nodes': 16.0, 'job_nodes_half': 8.0, 'node_gpus': 8.0, 'tiers_alloc': 500.0, 'tiers_active': 350.0, 'tiers_prod': 280.0, 'tiers_prod_pct': 56.0, 'mtbf_h': 302.0, 'mtbf_days': 12.6, 'gain_gptj': 2.69, 'gain_gptj_1': 2.7, 'gain_llama2': 1.44, 'gain_llama2_1': 1.4, 'gain_sdxl': 1.25, 'gain_sdxl_1': 1.2, 'gain_flat': 1.0, 'gain_flat_1': 1.0, 'gain_months': 19.0, 'trace_gpus': 1024.0, 'trace_s': 5.0, 'trace_mb': 500.0, 'trace_gb': 512.0, 'megascale_slow_pct': 0.5, 'cf_cpu_pct': 100.0, 'cf_min': 27.0, 'cf_drop_pct': 82.0, 'snap_assigned': 0.8, 'snap_partial': 0.15, 'snap_input': 0.05, 'snap_exec': 0.6, 'snap_free': 0.2, 'sched': 0.65, 'sched_ceiling': 0.8, 'google_s_pct': 95.0, 'google_s': 0.95, 'rt_loss_pct': 10.0, 'runtime': 0.9, 'runtime_gap': 0.1, 'tau_min': 77.0, 'tau_int_min': 21.0, 'preempt_min': 20.2, 'preempt_lost_min': 15.0, 'write_s': 9.8, 'term_input_pct': 7.7, 'term_preempt_pct': 1.4, 'term_ckpt_pct': 0.4, 'term_startup_pct': 0.05, 'term_recovery_pct': 0.03, 'startup_min': 5.2, 'startup_hour_pct': 9.0, 'cross_gpus': 13000.0, 'rt_loss_16k_pct': 18.0, 'mfu_gpt3': 21.3, 'mfu_mtnlg': 30.2, 'mfu_gopher': 32.5, 'mfu_palm': 46.2, 'llama_p8_pct': 43.0, 'llama_p8': 0.43, 'llama_gpus_8k': 8192.0, 'palm_chips': 6144.0, 'llama_mfu_drop': 2.0, 'node_mfu_pct': 65.0, 'node_ms': 180.0, 'fleet_ms': 245.0, 'fleet_s': 0.245, 'extra_ms': 65.0, 'kept': 0.735, 'kept_pct': 74.0, 'fleet_mfu_est': 0.478, 'program': 0.48, 'program_pct': 48.0, 'useful_pflop': 14.9, 'peak_pflop': 31.0, 'h100_peak': 989.0, 'h100_bw': 3.35, 'ridge': 295.0, 'decode_flops_per_param': 2.0, 'fp16_bytes': 2.0, 'decode_ai': 1.0, 'decode_rate': 3.35, 'decode_peak_pct': 0.34, 'decode_peak_pct_1': 0.3, 'decode_ms': 5.2, 'ln_tflops': 5.0, 'ln_tbs': 2.8, 'ln_ai': 1.8, 'ln_bw_pct': 83.6, 'tp_ms': 45.0, 'dec_ar_kb': 32.8, 'dec_ar_x': 70.0, 'mpg': 0.28, 'mpg_3': 0.282, 'mpg_pct': 28.0, 'after_r': 0.59, 'job_rp': 0.43, 'program_gap': 0.52, 's_loss_pct': 35.0, 'r_loss_pct': 6.0, 'p_loss_pct': 31.0, 'lvl_s': 65.0, 'lvl_r': 59.0, 'runtime_ceiling': 0.98, 'program_ceiling': 0.65, 'lever_s_pct': 23.0, 'lever_s_mpg': 0.35, 'point_s': 0.43, 'lever_r_pct': 9.0, 'lever_r_mpg': 0.31, 'point_r': 0.31, 'lever_p_pct': 35.0, 'lever_p_mpg': 0.38, 'point_p': 0.59, 'price': 2.0, 'useful_price': 7.09, 'cost_x': 3.55, 'cost_x_1': 3.5, 'job_cost_x': 2.3, 'p_cut_pct': 26.0, 'r_headroom_pct': 11.0, 'trade_p': 1.1, 'trade_r_den': 12.0, 'trade_r_num': 11.0, 'trade_x': 1.008, 'cmp_rate': 120.0, 'cmp_step': 8.33, 'cmp_mm_pct': 50.0, 'cmp_ew_cut_pct': 70.0, 'cmp_ln_cut_pct': 80.0, 'cmp_mm': 0.5, 'cmp_ew': 0.35, 'cmp_ew_keep': 0.3, 'cmp_ln': 0.15, 'cmp_ln_keep': 0.2, 'cmp_ew_pct': 35.0, 'cmp_ln_pct': 15.0, 'cmp_frac': 0.635, 'cmp_new_ms': 5.3, 'cmp_saved_ms': 3.0, 'cmp_build_min': 15.0, 'cmp_tokens': 296000.0, 'cmp_tokens_ch': 300000.0, 'cmp_repay_min': 26.0, 'cmp_replicas': 128.0, 'cmp_shared_s': 12.0, 'cmp_rollout_h': 32.0, 'fr_dev_ratio': 2.0, 'fr_thr': 1.91, 'fr_time': 0.52, 'fr_cost': 1.049, 'fr_cost_pct': 4.9}
import math
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
plt = sp.plt
C = sp.C
out = Path(__file__).parent
COL = {'S': C['network'], 'R': '#78492A', 'P': C['compute'], 'U': C['data'], 'S_l': C['network_l'], 'R_l': '#E3D3C8', 'P_l': C['compute_l'], 'U_l': C['data_l']}

def text(ax, x, y, s, color=C['text'], shown=True, **kw):
    """Text drawn in every step (invisible until shown) so every step has
    the same bounding box and nothing moves on the slide."""
    kw.setdefault('fontsize', 13)
    return ax.text(x, y, s, color=color, alpha=1.0 if shown else 0.0, **kw)

def llama3_split(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 1.75))
    rows = [('Effective\ntraining time', 90.0, COL['R'], '> 90% of elapsed time'), ('MFU of\nkept steps', 41.0, COL['P'], '41% of peak'), ('Held\ncapacity', None, None, None)]
    for i, (lab, v, col, note) in enumerate(rows):
        y = 2 - i
        ax.text(-2, y, lab, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
        ax.barh(y, 100, height=0.6, color='white', edgecolor=C['grid'], lw=1)
        if v is not None:
            ax.barh(y, v, height=0.6, color=col)
            ax.text(v + 1.5, y, note, va='center', fontsize=13, fontweight='bold', color=col)
    shown = step >= 2
    left = 0
    for w, col, lab, tcol in [(10.0, COL['R_l'], '10%', COL['R']), (53.0, COL['P_l'], 'program 53%', COL['P']), (37.0, COL['U'], 'progress 37%', 'white')]:
        ax.barh(0, w, left=left, height=0.6, color=col, edgecolor='white', lw=1.5, alpha=1.0 if shown else 0.0)
        text(ax, left + w / 2, 0, lab, tcol, shown, ha='center', va='center', fontsize=12, fontweight='bold', linespacing=0.9)
        left += w
    text(ax, 10.0 / 2, -0.48, 'runtime', COL['R'], shown, ha='center', va='center', fontsize=11.5, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, 2.4)
    ax.axis('off')
    sp.save(fig, path)
for k in (1, 2):
    llama3_split(out / f'llama3_split_step{k}.svg', k)
sp.hbars(out / 'tiers.svg', [('Allocated', 500.0, 'neutral', '500 GPUs'), ('Active', 350.0, 'compute', '350 GPUs'), ('Productive', 280.0, 'data', '280 GPUs')], 'GPUs', log=False, size=(5.6, 1.3))

def goodput_chain(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 2.55))
    widths = [1.0, 0.86, 0.74, 0.58]
    inside = ['offered capacity-time', 'held time', 'retained steps', 'useful work']
    fills = [C['neutral_l'], COL['S_l'], COL['R_l'], COL['U']]
    tcols = [C['text'], C['text'], C['text'], 'white']
    keys = [None, 'S', 'R', 'P']
    rules = [None, 'S = held / offered', 'R = retained / held', 'P = ideal / retained']
    losses = [None, 'unallocated, partial gangs, fragments', 'startup, input waits, checkpoints, rework', 'exposed comm., stragglers, stalls, dispatch']
    for i in range(4):
        y = 3 - i
        shown = i <= step
        a = 1.0 if shown else 0.0
        ax.barh(y, widths[i], height=0.56, color=fills[i], edgecolor=C['neutral'], lw=1.0, alpha=a)
        text(ax, 0.025, y, inside[i], tcols[i], shown, ha='left', va='center', fontsize=11, fontweight='bold')
        if i > 0:
            k = keys[i]
            ax.barh(y, widths[i - 1] - widths[i], left=widths[i], height=0.56, color='white', edgecolor=COL[k], lw=1.4, ls='--', alpha=a)
            ax.plot([-0.02, 1.02], [y + 0.5, y + 0.5], color=COL[k], lw=1.2, ls=':', alpha=a)
            text(ax, -0.09, y, k, 'white', shown, ha='center', va='center', fontsize=14, fontweight='bold', bbox=dict(boxstyle='circle,pad=0.25', fc=COL[k], ec='none', alpha=a))
            text(ax, 1.05, y + 0.2, rules[i], COL[k], shown, ha='left', va='center', fontsize=12, fontweight='bold')
            text(ax, 1.05, y - 0.2, losses[i], C['neutral'], shown, ha='left', va='center', fontsize=10.5)
    text(ax, 1.05, 3, 'what the pool could run', C['neutral'], True, ha='left', va='center', fontsize=11.5)
    text(ax, -0.15, -0.68, 'useful work = offered capacity-time × S × R × P', C['data'], step >= 3, ha='left', va='center', fontsize=12.5, fontweight='bold')
    ax.set_xlim(-0.17, 2.38)
    ax.set_ylim(-0.85, 3.4)
    ax.axis('off')
    sp.save(fig, path)
for k in (1, 2, 3):
    goodput_chain(out / f'goodput_chain_step{k}.svg', k)
rounds = ['v3.1', 'v4.0', 'v4.1', 'v5.0']
round_lab = ['v3.1\nSep 2023', 'v4.0\nMar 2024', 'v4.1\nAug 2024', 'v5.0\nApr 2025']
series = {'gptj-99': [('2023-09', 'v3.1', 106.315), ('2024-03', 'v4.0', 239.895), ('2024-08', 'v4.1', 286.084)], 'llama2-70b-99.9': [('2024-03', 'v4.0', 21805.8), ('2024-08', 'v4.1', 24524.9), ('2025-04', 'v5.0', 31306.8)], 'stable-diffusion-xl': [('2024-03', 'v4.0', 13.1263), ('2024-08', 'v4.1', 16.3484)], 'bert-99': [('2023-09', 'v3.1', 70169.7), ('2024-03', 'v4.0', 70116.2)], 'resnet50': [('2023-09', 'v3.1', 704412.0), ('2024-03', 'v4.0', 703770.0)]}
bench = [('gptj-99', 'GPT-J', C['compute']), ('llama2-70b-99.9', 'Llama 2 70B', C['memory']), ('stable-diffusion-xl', 'SDXL', C['data']), ('resnet50', 'ResNet-50', C['neutral']), ('bert-99', 'BERT', C['neutral'])]

def mlperf(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    for key, name, col in bench:
        pts = sorted(series[key])
        base = pts[0][2]
        xs = [rounds.index(p[1]) for p in pts]
        ys = [p[2] / base for p in pts]
        shown = step >= 2 or key in ('resnet50', 'bert-99')
        a = 1.0 if shown else 0.0
        ax.plot(xs, ys, '-o', color=col, lw=2.6, ms=7, alpha=a)
        if key in ('resnet50', 'bert-99'):
            if key == 'resnet50':
                text(ax, 0.05, 0.9, 'ResNet-50, BERT 1.00×', col, shown, va='top', fontsize=12, fontweight='bold')
        elif key == 'stable-diffusion-xl':
            text(ax, xs[-1] - 0.08, ys[-1] + 0.12, f'{name} {ys[-1]:.2f}×', col, shown, ha='right', va='bottom', fontsize=12.5, fontweight='bold')
        else:
            text(ax, xs[-1] + 0.08, ys[-1], f'{name} {ys[-1]:.2f}×', col, shown, va='center', fontsize=12.5, fontweight='bold')
    ax.set_xticks(range(4), round_lab, fontsize=11)
    ax.set_xlim(-0.2, 4.1)
    ax.set_ylim(0.8, 3.0)
    ax.set_ylabel('× first round')
    ax.grid(axis='y', color=C['grid'], lw=0.6)
    sp.save(fig, path)
for k in (1, 2):
    mlperf(out / f'mlperf_gains_step{k}.svg', k)

def ladder(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    rungs = [('S', 'Scheduler records', 'every job; free', 'settles S', 'cannot see\ninside a gang'), ('R', 'Job timeline', 'every job; bytes per step', 'settles R', 'cannot say why\nsteps are slow'), ('P', 'Step traces', 'few ranks; GB per trace', 'settles P and\nthe term', 'misses\nrestarts')]
    for i, (k, name, cover, settles, blind) in enumerate(rungs):
        y = 2 - i
        shown = i < step
        a = 1.0 if shown else 0.0
        ax.add_patch(plt.Rectangle((0, y - 0.42), 13.3, 0.84, fc='white', ec=COL[k], lw=1.6, alpha=a))
        text(ax, 0.38, y, k, 'white', shown, ha='center', va='center', fontsize=14, fontweight='bold', bbox=dict(boxstyle='circle,pad=0.25', fc=COL[k], ec='none', alpha=a))
        text(ax, 0.85, y + 0.14, name, COL[k], shown, ha='left', va='center', fontsize=13, fontweight='bold')
        text(ax, 0.85, y - 0.19, cover, C['neutral'], shown, ha='left', va='center', fontsize=11)
        text(ax, 6.3, y, settles, C['data'], shown, ha='left', va='center', fontsize=12, fontweight='bold', linespacing=1.0)
        text(ax, 10.0, y, blind, C['danger'], shown, ha='left', va='center', fontsize=11, linespacing=1.0)
        if 0 < i:
            ax.annotate('', xy=(0.38, y + 0.43), xytext=(0.38, y + 0.58), arrowprops=dict(arrowstyle='-|>', color=C['neutral'], lw=1.5), alpha=a)
    for x, h in [(0.85, 'Record, read top first'), (6.3, 'Establishes'), (10.0, 'Blind spot')]:
        ax.text(x, 2.6, h, fontsize=11, color=C['neutral'], fontweight='bold', va='center')
    ax.set_xlim(-0.05, 13.4)
    ax.set_ylim(-0.45, 2.8)
    ax.axis('off')
    sp.save(fig, path)
for k in (1, 2, 3):
    ladder(out / f'ladder_step{k}.svg', k)
sp.hbars(out / 'trace_volume.svg', [('1 GPU, 5 s', 500.0 / 10 ** 3, 'memory', '500 MB'), ('1,024 GPUs, 5 s', 512.0, 'danger', '512 GB')], 'GB', log=True, size=(5.6, 1.15), xlabel='Nsight Systems trace volume (GB, log)')

def pool(path, step):
    fig, ax = plt.subplots(figsize=(4.4, 1.75))
    segs = [(0.6, C['data'], 'executing\n0.60', 'white'), (0.05, COL['R'], '', 'white'), (0.15, COL['S_l'], '0.15', COL['S']), (0.2, C['neutral_l'], 'free\n0.20', C['text'])]
    left = 0
    for w, col, lab, tc in segs:
        ax.barh(0, w, left=left, height=0.55, color=col, edgecolor='white', lw=1.5, hatch='//' if col == COL['S_l'] else None)
        if lab:
            ax.text(left + w / 2, 0, lab, ha='center', va='center', fontsize=11.5, fontweight='bold', color=tc, linespacing=0.95)
        left += w
    ax.annotate('input wait 0.05\n(complete allocation)', xy=(0.6 + 0.05 / 2, 0.28), xytext=(0.38, 0.72), fontsize=10.5, color=COL['R'], fontweight='bold', ha='center', va='bottom', arrowprops=dict(arrowstyle='-', color=COL['R'], lw=1.2))
    ax.text(0.8, 0.62, 'partial gangs', ha='center', va='bottom', fontsize=11.5, fontweight='bold', color=COL['S'])
    a_s = 0.8
    ax.plot([0, 0, a_s, a_s], [0.36, 0.42, 0.42, 0.36], color=C['text'], lw=1.3)
    ax.text(0.2, 0.5, 'assigned to jobs 0.80', ha='center', va='bottom', fontsize=11.5, fontweight='bold', color=C['text'])
    shown = step >= 2
    a = 1.0 if shown else 0.0
    s_v = 0.65
    ax.plot([0, 0, s_v, s_v], [-0.36, -0.42, -0.42, -0.36], color=COL['S'], lw=2, alpha=a)
    text(ax, 0.0, -0.5, 'S = 0.65, complete allocations', COL['S'], shown, ha='left', va='top', fontsize=12, fontweight='bold')
    text(ax, 1.0, -0.5, 'ceiling 0.80', C['neutral'], shown, ha='right', va='top', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.85, 1.05)
    ax.axis('off')
    sp.save(fig, path)
for k in (1, 2):
    pool(out / f'pool_snapshot_step{k}.svg', k)
terms = [(lab, _NUMS[key], col, f'{_VALS[key]}%') for lab, key, col in [('Input waits', 'term_input_pct', COL['R']), ('Preemption', 'term_preempt_pct', COL['S']), ('Checkpoints', 'term_ckpt_pct', C['neutral']), ('Startup', 'term_startup_pct', C['neutral']), ('Recovery', 'term_recovery_pct', C['neutral'])]]
fig, ax = plt.subplots(figsize=(5.4, 1.6))
for i, (lab, v, col, s) in enumerate(terms):
    y = len(terms) - 1 - i
    ax.barh(y, v, height=0.62, color=col)
    ax.text(-0.15, y, lab, ha='right', va='center', fontsize=13, fontweight='bold')
    ax.text(v + 0.15, y, s, va='center', fontsize=13, fontweight='bold', color=col)
ax.set_xlim(0, 9.5)
ax.set_ylim(-0.6, 4.6)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Share of held time (%)')
ax.grid(axis='x', color=C['grid'], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'runtime_terms.svg')
sizes = [int(128.0) * 2 ** k for k in range(8)]
fig, ax = plt.subplots(figsize=(5.6, 2.15))
for i, n in enumerate(sizes):
    t = {128: SimpleNamespace(checkpoint_share=0.004242989861, failure_side=0.004582658009, input_wait_share=0.07692307692, interruption_checkpoint_interval=Q_(20.87615423, 'min'), interruption_interval=Q_(22.23539875, 'h'), interval=Q_(76.98973539, 'min'), mtbf=Q_(302.4193548, 'h'), preemption_cost=Q_(20.16333333, 'min'), preemption_share=0.01400231481, recovery_share=0.0003396681481, recovery_time=Q_(6.163333333, 'min'), rework_share=0.002121494931, runtime_goodput=0.9039797148, runtime_loss=0.0960202852, startup_share=0.0005122354497, startup_time=Q_(5.163333333, 'min'), write_share=0.002121494931, write_time=Q_(9.8, 's')), 256: SimpleNamespace(checkpoint_share=0.006000493807, failure_side=0.006679830103, input_wait_share=0.07692307692, interruption_checkpoint_interval=Q_(20.14857819, 'min'), interruption_interval=Q_(20.71251036, 'h'), interval=Q_(54.43996397, 'min'), mtbf=Q_(151.2096774, 'h'), preemption_cost=Q_(20.16333333, 'min'), preemption_share=0.01400231481, recovery_share=0.0006793362963, recovery_time=Q_(6.163333333, 'min'), rework_share=0.003000246903, runtime_goodput=0.9018825427, runtime_loss=0.09811745729, startup_share=0.0005122354497, startup_time=Q_(5.163333333, 'min'), write_share=0.003000246903, write_time=Q_(9.8, 's')), 512: SimpleNamespace(checkpoint_share=0.008485979723, failure_side=0.009844652315, input_wait_share=0.07692307692, interruption_checkpoint_interval=Q_(18.89592837, 'min'), interruption_interval=Q_(18.21714841, 'h'), interval=Q_(38.49486769, 'min'), mtbf=Q_(75.60483871, 'h'), preemption_cost=Q_(20.16333333, 'min'), preemption_share=0.01400231481, recovery_share=0.001358672593, recovery_time=Q_(6.163333333, 'min'), rework_share=0.004242989861, runtime_goodput=0.8987177205, runtime_loss=0.1012822795, startup_share=0.0005122354497, startup_time=Q_(5.163333333, 'min'), write_share=0.004242989861, write_time=Q_(9.8, 's')), 1024: SimpleNamespace(checkpoint_share=0.01200098761, failure_side=0.0147183328, input_wait_share=0.07692307692, interruption_checkpoint_interval=Q_(16.96253341, 'min'), interruption_interval=Q_(14.67997651, 'h'), interval=Q_(27.21998199, 'min'), mtbf=Q_(37.80241935, 'h'), preemption_cost=Q_(20.16333333, 'min'), preemption_share=0.01400231481, recovery_share=0.002717345185, recovery_time=Q_(6.163333333, 'min'), rework_share=0.006000493807, runtime_goodput=0.89384404, runtime_loss=0.10615596, startup_share=0.0005122354497, startup_time=Q_(5.163333333, 'min'), write_share=0.006000493807, write_time=Q_(9.8, 's')), 2048: SimpleNamespace(checkpoint_share=0.01697195945, failure_side=0.02240664982, input_wait_share=0.07692307692, interruption_checkpoint_interval=Q_(14.39606131, 'min'), interruption_interval=Q_(10.57380516, 'h'), interval=Q_(19.24743385, 'min'), mtbf=Q_(18.90120968, 'h'), preemption_cost=Q_(20.16333333, 'min'), preemption_share=0.01400231481, recovery_share=0.00543469037, recovery_time=Q_(6.163333333, 'min'), rework_share=0.008485979723, runtime_goodput=0.886155723, runtime_loss=0.113844277, startup_share=0.0005122354497, startup_time=Q_(5.163333333, 'min'), write_share=0.008485979723, write_time=Q_(9.8, 's')), 4096: SimpleNamespace(checkpoint_share=0.02400197523, failure_side=0.03487135597, input_wait_share=0.07692307692, interruption_checkpoint_interval=Q_(11.52819917, 'min'), interruption_interval=Q_(6.780580418, 'h'), interval=Q_(13.60999099, 'min'), mtbf=Q_(9.450604839, 'h'), preemption_cost=Q_(20.16333333, 'min'), preemption_share=0.01400231481, recovery_share=0.01086938074, recovery_time=Q_(6.163333333, 'min'), rework_share=0.01200098761, runtime_goodput=0.8736910168, runtime_loss=0.1263089832, startup_share=0.0005122354497, startup_time=Q_(5.163333333, 'min'), write_share=0.01200098761, write_time=Q_(9.8, 's')), 8192: SimpleNamespace(checkpoint_share=0.03394391889, failure_side=0.05568268037, input_wait_share=0.07692307692, interruption_checkpoint_interval=Q_(8.796626572, 'min'), interruption_interval=Q_(3.947991788, 'h'), interval=Q_(9.623716923, 'min'), mtbf=Q_(4.725302419, 'h'), preemption_cost=Q_(20.16333333, 'min'), preemption_share=0.01400231481, recovery_share=0.02173876148, recovery_time=Q_(6.163333333, 'min'), rework_share=0.01697195945, runtime_goodput=0.8528796924, runtime_loss=0.1471203076, startup_share=0.0005122354497, startup_time=Q_(5.163333333, 'min'), write_share=0.01697195945, write_time=Q_(9.8, 's')), 16384: SimpleNamespace(checkpoint_share=0.04800395045, failure_side=0.09148147342, input_wait_share=0.07692307692, interruption_checkpoint_interval=Q_(6.492903094, 'min'), interruption_interval=Q_(2.150907683, 'h'), interval=Q_(6.804995497, 'min'), mtbf=Q_(2.36265121, 'h'), preemption_cost=Q_(20.16333333, 'min'), preemption_share=0.01400231481, recovery_share=0.04347752296, recovery_time=Q_(6.163333333, 'min'), rework_share=0.02400197523, runtime_goodput=0.8170808994, runtime_loss=0.1829191006, startup_share=0.0005122354497, startup_time=Q_(5.163333333, 'min'), write_share=0.02400197523, write_time=Q_(9.8, 's'))}[n]
    parts = [(t.input_wait_share, '#D9C3A5'), (t.preemption_share, COL['S_l']), (t.startup_share, C['neutral_l']), (t.checkpoint_share, '#A0715A'), (t.recovery_share, COL['R'])]
    b = 0
    for v, col in parts:
        ax.bar(i, v * 100, 0.66, bottom=b, color=col, edgecolor='white', lw=0.6)
        b += v * 100
    ax.text(i, b + 0.4, f'{b:.0f}%', ha='center', va='bottom', fontsize=11.5, fontweight='bold', color=C['text'])
ax.bar(8.0, 10.0, 0.4, color='white', edgecolor=C['text'], lw=1.6, ls='--')
ax.text(8.0, 10.0 + 0.4, 'Llama 3\n≤ 10%', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=C['text'], linespacing=0.95)
cross = math.log2(12928 / sizes[0])
ax.axvline(cross, color=C['danger'], lw=1.4, ls=':')
ax.annotate('checkpoints + recovery\npass input waits', xy=(cross, 17.5), xytext=(3.0, 19.6), ha='center', va='top', arrowprops=dict(arrowstyle='-|>', color=C['danger'], lw=1.2), fontsize=10.5, color=C['danger'], fontweight='bold', linespacing=0.95)
ax.text(-0.4, 23.8, 'brown: checkpoints, recovery\ntan: input waits; violet: preemption', ha='left', va='top', fontsize=10.5, color=C['neutral'], linespacing=1.0)
ax.set_xticks(list(range(8)) + [8.0], ['128', '256', '512', '1K', '2K', '4K', '8K', '16K', '16K\nmeasured'], fontsize=10.5)
ax.set_xlim(-0.5, 8.4)
ax.set_ylim(0, 24)
ax.set_ylabel('Runtime loss\n(% of held time)')
ax.set_xlabel('GPUs in the job (derived model; same inputs at every size)', fontsize=11)
sp.save(fig, out / 'runtime_scaling.svg')
sp.hbars(out / 'mfu_published.svg', [(lab, _NUMS[key], col, f'{_VALS[key]}%') for lab, key, col in [('GPT-3, 2020', 'mfu_gpt3', 'compute'), ('MT-NLG, 2021', 'mfu_mtnlg', 'compute'), ('Gopher, 2021', 'mfu_gopher', 'compute'), ('PaLM, 2022', 'mfu_palm', 'compute'), ('Llama 3, 8K GPUs', 'llama_p8_pct', 'memory'), ('Llama 3, 16K GPUs', 'llama_p_pct', 'memory')]], '%', log=False, size=(4.3, 2.4), xlabel='MFU (% of peak)')

def node_fleet(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 1.4))
    t_node, t_extra, t_fleet = (180.0, 65.0, 245.0)
    ax.barh(1, t_node, height=0.5, color=COL['P'])
    ax.barh(0, t_node, height=0.5, color=COL['P'])
    ax.barh(0, t_extra, left=t_node, height=0.5, color='white', edgecolor=C['danger'], hatch='//', lw=1.4)
    ax.text(-4, 1, '8-GPU node', ha='right', va='center', fontsize=13, fontweight='bold')
    ax.text(-4, 0, '128-GPU fleet', ha='right', va='center', fontsize=13, fontweight='bold')
    ax.text(t_node / 2, 1, '180 ms', ha='center', va='center', color='white', fontsize=13, fontweight='bold')
    ax.text(t_node / 2, 0, '180 ms of node work', ha='center', va='center', color='white', fontsize=12.5, fontweight='bold')
    ax.text(t_node + t_extra / 2, 0, '+65 ms', ha='center', va='center', color=C['danger'], fontsize=12.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))
    shown = step >= 2
    text(ax, 252, 1, 'MFU 65%', COL['P'], shown, va='center', fontsize=13, fontweight='bold')
    text(ax, 252, 0, 'MFU 48%', C['danger'], shown, va='center', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 100, 200, t_fleet], ['0', '100', '200', '245 ms'])
    sp.save(fig, path)
for k in (1, 2):
    node_fleet(out / f'node_fleet_step{k}.svg', k)
PEAK, BW = (989.0, 3.35)
fig, ax = plt.subplots(figsize=(5.6, 2.0))
xs = np.logspace(-1, 3.6, 300)
roof = np.minimum(PEAK, xs * BW)
ridge = PEAK / BW
ax.plot(xs[xs <= ridge], roof[xs <= ridge], color=C['memory'], lw=3)
ax.plot(xs[xs >= ridge], roof[xs >= ridge], color=C['compute'], lw=3)
ax.axvline(ridge, color=C['neutral'], lw=1, ls='--')
ax.text(ridge * 1.1, 2.0, 'ridge\n295 FLOP/byte', color=C['neutral'], fontsize=11, va='bottom', linespacing=0.95)
ax.text(3800, PEAK * 1.35, '989 TFLOP/s (FP16)', color=C['compute'], fontsize=11.5, fontweight='bold', ha='right')
ax.text(40, 12, '3.35 TB/s', color=C['memory'], fontsize=11.5, fontweight='bold', rotation=31, ha='center')
DEC_AI, DEC_RATE = (1.0, 3.35)
ax.plot([DEC_AI], [DEC_RATE], 'o', ms=11, color=C['danger'], mec='white', mew=1.5, zorder=5)
ax.annotate('batch-1 decode: 1 FLOP/byte,\n3.35 TFLOP/s = 0.34% of peak', (DEC_AI, DEC_RATE), xytext=(0.12, 160), fontsize=11.5, fontweight='bold', color=C['danger'], arrowprops=dict(arrowstyle='-', color=C['danger'], lw=1.1), linespacing=0.95)
LN_AI, LN_RATE = (1.8, 5.0)
ax.plot([LN_AI], [LN_RATE], 's', ms=9, color=C['memory'], mec='white', mew=1.5, zorder=5)
ax.annotate('LayerNorm, measured:\n1.8 FLOP/byte at 2.8 TB/s', (LN_AI, LN_RATE), xytext=(9, 0.25), fontsize=11, color=C['memory'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=C['memory'], lw=1.1), linespacing=0.95)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 4000)
ax.set_ylim(0.1, 3000)
ax.set_xlabel('Arithmetic intensity (FLOP/byte)')
ax.set_ylabel('TFLOP/s')
ax.minorticks_off()
sp.save(fig, out / 'roofline_decode.svg')

def overlap(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 2.15))
    panels = [('serialized', [(0, 2), (2, 2), (4, 2)], (6, 2), None, 8, 'exposed'), ('overlapped, idle device', [(0, 2), (2, 2), (4, 2)], (2, 2), None, 6, 'hidden'), ('overlapped, shared device', [(0, 2), (2, 2.8), (4.8, 2)], (2, 2), (4.0, 0.8), 6.8, 'hidden')]
    for i, (name, ks, coll, cont, end, clab) in enumerate(panels):
        y0 = 2 - i
        shown = i < step
        a = 1.0 if shown else 0.0
        text(ax, -0.1, y0 + 0.2, name, C['text'], shown, ha='right', va='center', fontsize=11.5, fontweight='bold')
        for s, w in ks:
            ax.barh(y0 + 0.2, w, left=s, height=0.26, color=C['memory_l'], edgecolor=C['memory'], lw=1.2, alpha=a)
        ax.barh(y0 - 0.12, coll[1], left=coll[0], height=0.26, color='#fef3c7', edgecolor='#d97706', lw=1.2, alpha=a)
        text(ax, coll[0] + coll[1] / 2, y0 - 0.12, clab, '#b45309', shown, ha='center', va='center', fontsize=10.5, fontweight='bold')
        if cont:
            ax.barh(y0 + 0.2, cont[1], left=cont[0], height=0.26, color=C['danger_l'], edgecolor=C['danger'], lw=1.2, alpha=a)
            text(ax, cont[0] + cont[1] + 2.15, y0 + 0.2, 'contention', C['danger'], shown, va='center', fontsize=10.5, fontweight='bold')
        ax.plot([end, end], [y0 - 0.3, y0 + 0.38], color=C['text'], lw=3, alpha=a)
        text(ax, end + 0.12, y0 - 0.12, f'step ends {end:g}', C['text'], shown, va='center', fontsize=10.5)
    text(ax, -0.1, -0.45, 'blue: compute stream; amber: communication stream', C['neutral'], True, ha='right', va='center', fontsize=9.5)
    ax.text(10.5, -0.45, 'schematic durations', ha='right', va='center', fontsize=9.5, color=C['neutral'])
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.65, 2.6)
    ax.axis('off')
    sp.save(fig, path)
for k in (1, 2, 3):
    overlap(out / f'overlap_step{k}.svg', k)

def waterfall(path, step, compare=False):
    fig, ax = plt.subplots(figsize=(4.2, 2.6) if compare else (5.6, 2.1))
    levels = [100, int(65.0), int(59.0), int(28.0)]
    names = ['Fleet\ncapacity', 'After\nscheduling', 'After\nruntime', 'After\nprogram']
    losses = [None, ('-35%', COL['S']), ('-6%', COL['R']), ('-31%', COL['P'])]
    llama = [100, 100, int(90.0), int(37.0)]
    xoff = -0.12 if compare else 0
    for i, v in enumerate(levels):
        shown = i <= step
        a = 1.0 if shown else 0.0
        col = C['data'] if i == 3 else C['neutral_l']
        ax.bar(i + xoff, v, 0.5, color=col, edgecolor=C['neutral'], lw=1, alpha=a)
        text(ax, i + xoff, v / 2 if i < 3 else v / 2, f'{v}%', 'white' if i == 3 else C['text'], shown, ha='center', va='center', fontsize=11.5 if compare else 13, fontweight='bold')
        if losses[i]:
            lab, lc = losses[i]
            ax.bar(i + xoff, levels[i - 1] - v, 0.5, bottom=v, color='white', edgecolor=lc, lw=1.6, ls='--', alpha=a)
            text(ax, i + xoff, levels[i - 1] + 3, lab, lc, shown, ha='center', va='bottom', fontsize=13, fontweight='bold')
        if compare:
            ax.bar(i + 0.27, llama[i], 0.18, color='white', edgecolor=C['text'], lw=1.4)
            if llama[i] < 100:
                ax.text(i + 0.27, llama[i] + 3, f'{llama[i]}', ha='center', va='bottom', fontsize=10.5, color=C['text'])
    if compare:
        ax.text(-0.45, 127, 'hollow bars: Llama 3 405B on its dedicated cluster', ha='left', va='top', fontsize=11, color=C['text'])
    ax.set_xticks(range(4), names, fontsize=11.5)
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylim(0, 130 if compare else 112)
    ax.set_ylabel('% of fleet capacity')
    sp.save(fig, path)
for k in (1, 2, 3):
    waterfall(out / f'waterfall_step{k}.svg', k)
waterfall(out / 'waterfall_compare.svg', 3, compare=True)
lever_rows = [('Today', 0.28, 'neutral', 'MPG 0.28')] + [(f'{lab}: {f} to {_VALS[ceil]}', _NUMS[f'lever_{k}_mpg'], ck, f"{_VALS[f'lever_{k}_mpg']} (+{_VALS[f'lever_{k}_pct']}%)") for lab, f, ceil, k, ck in [('A. Scheduling', 'S', 'sched_ceiling', 's', 'network'), ('B. Input pipeline', 'R', 'runtime_ceiling', 'r', 'danger'), ('C. Program', 'P', 'program_ceiling', 'p', 'compute')]]
COLKEY = {'network': COL['S'], 'danger': COL['R'], 'compute': COL['P'], 'neutral': C['neutral']}

def levers(path, step):
    fig, ax = plt.subplots(figsize=(5.6, 1.6))
    for i, (lab, v, ck, s) in enumerate(lever_rows):
        y = 3 - i
        shown = i == 0 or step >= 2
        a = 1.0 if shown else 0.0
        ax.barh(y, v, height=0.62, color=COLKEY[ck], alpha=a)
        ax.text(-0.006, y, lab, ha='right', va='center', fontsize=12.5, fontweight='bold', color=C['text'])
        text(ax, v + 0.005, y, s, COLKEY[ck], shown, va='center', fontsize=12.5, fontweight='bold')
    ax.set_xlim(0, 0.47)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Pool MPG with the lever at its ceiling')
    sp.save(fig, path)
for k in (1, 2):
    levers(out / f'levers_step{k}.svg', k)
fig, ax = plt.subplots(figsize=(5.8, 2.5))
P = np.linspace(0.3, 0.75, 200)
R0, P0, P1 = (0.9, 0.48, 0.65)
base = R0 * P0
ax.fill_between(P, base / P, 1.05, where=base / P < 1.05, color=C['data_l'], alpha=0.7)
for c in (0.3, 0.35, 0.5, 0.55):
    ax.plot(P, c / P, color=C['grid'], lw=1.0)
ax.plot(P, base / P, color=C['text'], lw=2.4)
ax.text(0.745, 0.9, 'bold curve:\nR × P = 0.43', fontsize=12, ha='right', va='center', color=C['text'], fontweight='bold', linespacing=0.95)
ax.plot([P0], [R0], 'o', ms=11, color=C['danger'], zorder=5)
ax.text(P0 - 0.012, R0 - 0.02, 'running job', ha='right', va='top', fontsize=11.5, fontweight='bold', color=C['danger'])
ax.annotate('', xy=(P1, R0), xytext=(P0 + 0.01, R0), arrowprops=dict(arrowstyle='-|>', color=COL['P'], lw=2.2))
ax.text((P0 + P1) / 2 + 0.005, R0 - 0.025, 'P to 0.65: +35%', ha='center', va='top', fontsize=11.5, fontweight='bold', color=COL['P'])
ax.annotate('', xy=(P1, base / P1), xytext=(P1, R0 - 0.01), arrowprops=dict(arrowstyle='-|>', color=COL['R'], lw=1.8, ls='--'))
ax.plot([P1], [base / P1], 'o', ms=9, mfc='white', mec=COL['R'], mew=2, zorder=5)
ax.text(P1 + 0.01, 0.73, 'break-even:\nR may fall 26%', ha='left', va='center', fontsize=11, fontweight='bold', color=COL['R'], linespacing=0.95)
ax.axhline(1.0, color=C['neutral'], lw=1, ls=':')
ax.annotate('', xy=(P0, 1.0), xytext=(P0, R0 + 0.005), arrowprops=dict(arrowstyle='-|>', color=C['neutral'], lw=1.6))
ax.text(P0 - 0.045, 0.955, 'R headroom:\n+11% at most', ha='right', va='center', fontsize=11, color=C['neutral'], fontweight='bold', linespacing=0.95)
ax.text(0.745, 1.02, 'shaded: the trade raises R × P', fontsize=11, color=C['data'], fontweight='bold', ha='right', va='center')
ax.set_xlim(0.3, 0.75)
ax.set_ylim(0.55, 1.05)
ax.set_xlabel('Program goodput P')
ax.set_ylabel('Runtime goodput R')
sp.save(fig, out / 'isogoodput.svg')
fig, ax = plt.subplots(figsize=(5.6, 2.0))
pts = [(1.0, 1.0, '8,192 GPUs, MFU 43%'), (1 / 1.906976744, 0.43 / 0.41, '16,384 GPUs, MFU 41%')]
ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C['neutral'], lw=1.4, ls='--')
for (x, y, lab), col, off in zip(pts, [C['memory'], C['compute']], [(0.0, -0.007), (0.02, 0.006)]):
    ax.plot([x], [y], 'o', ms=13, color=col, zorder=5)
    ax.text(x + off[0], y + off[1], lab, ha='right' if off[0] <= 0 else 'left', va='top' if off[1] < 0 else 'bottom', fontsize=12, fontweight='bold', color=col)
ax.text(0.56, 1.006, 'neither point dominates:\none is faster, one is cheaper', ha='center', va='center', fontsize=11, color=C['neutral'], linespacing=0.95)
ax.set_xlim(0.4, 1.1)
ax.set_ylim(0.975, 1.075)
ax.set_xlabel('Time to result (relative to 8,192 GPUs)')
ax.set_ylabel('GPU-hours per\nunit of work')
ax.set_yticks([1.0, 1.025, 1.05], ['1.00', '1.025', '1.05'])
sp.save(fig, out / 'frontier_llama3.svg')
print('figures written to', out)
