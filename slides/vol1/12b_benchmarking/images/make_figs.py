#!/usr/bin/env python3
"""Figures for Lecture 12b (Benchmarking, Part 2: Deciding from Measurements).

Every drawn value comes from MLSysIM through calcs.py, which this script
imports: the MLPerf Training v4.0 ResNet-50 runs and the Scenarios.* built on
them (scores, Welch intervals, run counts, the paired accuracy comparison),
the sample-size, run-order, beam, calibration, and acceptance-gate scenarios,
and the Literature entries and series of published results (AnandTech 2013;
Recht et al. 2019; Zhang et al. 2024; Sandler et al. 2018; Chu et al. 2021;
Guo et al. 2017; Koh et al. 2021). Labels print the values calcs.py records
(deckvals.val), so a figure and its slide always show the same digits. Values
the chapter gives as a scenario or an assumption, and values built for a
slide, carry an "Illustrative" or "Assumed" tag on the plot. Axis limits,
ticks, and label positions are layout, not data.

Run from this folder with the MLSysBook venv (needs matplotlib and scipy):
  python3 make_figs.py
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
_NUMS = {'eos_score': 13.33, 'dell_score': 13.22, 'hook_gap_pct': 0.8, 'dell_spread': 0.02, 'eos_spread': 0.13, 'target_top1': 75.9, 'epochs': 35.0, 'scale_factor': 1.0042, 'dell_fail_top1': 75.894, 'dell_fail_frac': 0.75894, 'n_runs': 20.0, 'n_systems': 4.0, 'recall_tops': 10.0, 'recall_w': 0.5, 'recall_sus_tops': 3.0, 'recall_sus_w': 2.0, 'recall_eff': 20.0, 'recall_sus_eff': 1.5, 'log_seed': 5675.0, 'global_batch': 3200.0, 'ev_first_frac': 0.29274, 'ev_first_epoch': 3.0, 'ev_between': 7.0, 'ev_every': 4.0, 'ev_last_epoch': 35.0, 'log_eos_frac': 0.75936, 'log_eos_top1': 75.936, 'log_eos_ms': 798372.0, 'log_eos_min': 13.3062, 'cv_eos': 0.48, 'cv_dell': 0.06, 'cv_h200': 0.07, 'cv_n8': 0.04, 'eos_span': 0.99, 'eos_span_int': 1.0, 'others_span': 0.2, 'dell_fail_pct': 6.0, 'n8_fail_pct': 5.4, 'cv_ratio': 7.5, 'rtx_cv': 2.2, 'acc_lo': 75.88, 'acc_hi': 76.21, 'acc_spread': 0.32, 'acc_spread_eos': 0.31, 'acc_mean_int': 76.0, 'val_images': 50000.0, 'binom_hw': 0.37, 'dell_fail_min': 13.94, 'dell_middle': 13.163, 'z95': 1.96, 'z99': 2.576, 'acc_p': 0.95, 'acc_p_pct': 95.0, 'acc_q': 0.05, 'acc_m': 0.01, 'acc_m_pts': 1.0, 'acc_conf': 95.0, 'p90_p': 0.9, 'p90_m': 0.005, 'p90_m_pts': 0.5, 'p99_p': 0.99, 'p99_m': 0.0005, 'p99_m_pts': 0.05, 'q_conf': 99.0, 'n_acc': 1825.0, 'n_p90': 23886.0, 'n_p99': 262742.0, 'min_p90': 24576.0, 'min_p99': 270336.0, 'edge_share_pct': 5.0, 'eos_n': 5.0, 'dell_n': 4.0, 'eos_mean': 13.263, 'eos_mean_short': 13.26, 'eos_sd': 0.063, 'dell_mean': 13.159, 'dell_sd': 0.0084, 'd_dell': 0.104, 'se_dell': 0.0285, 't_dell': 2.73, 'dof_dell': 4.2, 'lo_dell': 0.026, 'hi_dell': 0.182, 'lo_dell_short': 0.03, 'hi_dell_short': 0.18, 'd_h200': 1.235, 'se_h200': 0.0285, 't_h200': 2.74, 'lo_h200': 1.16, 'hi_h200': 1.31, 'eq_hw': 0.1, 'pct_dell': 0.8, 'pct_lo_dell': 0.2, 'pct_hi_dell': 1.4, 'pct_dellall': -0.4, 'pct_lo_dellall': -3.7, 'pct_hi_dellall': 2.9, 'pct_h200': 9.3, 'pct_lo_h200': 8.7, 'pct_hi_h200': 9.9, 'pct_lo_dellall_abs': 3.7, 'pct_h200_int': 9.0, 'min_useful': 2.0, 'hw_5': 0.59, 'hw_10': 0.34, 'hw_20': 0.22, 'res_diff': 0.5, 'res_runs': 15.0, 'res_runs_one': 4.0, 'z_power': 0.84, 'vote_base': 95.0, 'vote_comp': 94.0, 'vote_n': 1000.0, 'vote_err_base': 50.0, 'vote_err_comp': 60.0, 'vote_lo_abs': 1.0, 'vote_hi': 3.0, 'vote_gap': 10.0, 'chi2': 3.84, 'vote_max': 26.0, 'best_run': 2.0, 'worst_run': 4.0, 'best_top1': 76.206, 'worst_top1': 75.9, 'seed_gap': 153.0, 'seed_max': 6093.0, 'seed_max_pct': 12.0, 'sys_dell_diff': 30.0, 'sys_dell_all': 44.0, 'sys_h200_diff': 7.0, 'sys_h200_all': 44.0, 'h200_bw': 4.8, 'h100_bw': 3.35, 'h200_bw_more': 43.0, 'h200_sparse': 1979.0, 'lee_start': 2.4, 'lee_end': 3.9, 'order_blocked': 20.0, 'order_inter': 4.0, 'beam_w0': 1.0, 'beam_w1': 10.0, 'beam_gain': 0.5, 'beam_lat0': 50.0, 'beam_lat1': 200.0, 'beam_slo': 100.0, 's4_gfx': 37.9, 's4_glb': 43.2, 's4_gain': 14.0, 's4_game_mhz': 480.0, 's4_bench_mhz': 532.0, 's4_max_mhz': 533.0, 'recht_models': 64.0, 'recht_lo': 10.3, 'recht_hi': 13.6, 'recht_rho': 0.997, 'gsm1k_drop': 8.0, 'sd_mobilenetv1_top1': 70.6, 'sd_mobilenetv1_ms': 113.0, 'sd_mobilenetv2_top1': 72.0, 'sd_mobilenetv2_ms': 75.0, 'sd_nasneta_top1': 74.0, 'sd_nasneta_ms': 183.0, 'sd_mobilenetv214_top1': 74.7, 'sd_mobilenetv214_ms': 143.0, 'frame_budget': 33.0, 'chu_top1_lo': 75.1, 'chu_top1_hi': 76.2, 'chu_v3_cpu': 20.3, 'chu_v1_cpu': 54.7, 'chu_v1_dsp': 3.72, 'chu_v3_dsp': 4.51, 'bin1_n': 20.0, 'bin1_conf': 0.65, 'bin1_acc_pct': 60.0, 'bin1_gap': 0.05, 'bin1_w': 0.2, 'bin2_n': 30.0, 'bin2_conf': 0.85, 'bin2_acc_pct': 80.0, 'bin2_gap': 0.05, 'bin2_w': 0.3, 'bin3_n': 50.0, 'bin3_conf': 0.97, 'bin3_acc_pct': 84.0, 'bin3_gap': 0.13, 'bin3_w': 0.5, 'bins_total': 100.0, 'ece': 0.09, 'alert_conf': 85.0, 'guo_lenet_err': 44.9, 'guo_resnet_err': 30.6, 'guo_lenet_ece': 4.85, 'guo_resnet_ece': 16.53, 'guo_in_ece': 5.48, 'guo_in_ece_ts': 1.86, 'guo_ratio': 3.4, 'wilds_id_acc': 75.7, 'wilds_ood_acc': 71.6, 'wilds_id_f1': 47.0, 'wilds_ood_f1': 31.0, 'wilds_acc_drop': 4.1, 'wilds_f1_drop': 16.0, 'wilds_acc_drop_int': 4.0, 'wilds_f1_drop_int': 16.0, 'wilds_reps': 3.0, 'camelyon_id': 93.2, 'camelyon_ood': 70.3, 'g_top1_fp32': 71.8, 'g_top1_int8': 70.9, 'g_top5_fp32': 91.0, 'g_top5_int8': 90.4, 'g_ece_fp32': 0.031, 'g_ece_int8': 0.089, 'g_edge_fp32': 68.2, 'g_edge_int8': 61.4, 'g_edge_fp32_int': 68.0, 'g_limit': 1.0, 'g_top1_loss': 0.9, 'g_edge_drop': 6.8, 'g_test': 1000.0, 'g_edge_n': 50.0, 'g_edge_hw': 13.0, 'img_w': 224.0, 'img_h': 224.0, 'img_c': 3.0, 'img_bits': 8.0, 'img_levels': 256.0, 'img_values': 150528.0, 'space_exp': 362508.0, 'test_n': 50000.0, 'test_exp': 4.7, 'frac_exp': 362503.0}
import statistics as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
from matplotlib.patches import Rectangle
plt = sp.plt
C = sp.C
out = Path(__file__).parent
TARGET_PCT = 75.9

def tag(ax, text, where=(0.0, -0.32), size=9):
    ax.text(where[0], where[1], text, transform=ax.transAxes, fontsize=size, color=C['neutral'], ha='left')
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 2.25), gridspec_kw=dict(height_ratios=[1.0, 1.2], hspace=1.0))
    rows = [('Dell XE9680, 8x H100', 13.21836607, 'compute', '13.22'), ('NVIDIA Eos, 8x H100', 13.3263435, 'memory', '13.33')]
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 1 - i
        a1.barh(y, v, color=C[ck], height=0.62)
        a1.text(v + 0.15, y, f'{txt} min', va='center', fontsize=12, fontweight='bold', color=C[ck])
        a1.text(-0.2, y, lab, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
    a1.set_xlim(0, 16)
    a1.set_ylim(-0.6, 1.6)
    a1.set_yticks([])
    a1.spines['left'].set_visible(False)
    a1.set_xlabel('Published score: minutes to 75.9% top-1')
    al = 1.0 if k >= 2 else 0.0
    for i, (s, ck, lab) in enumerate(((SimpleNamespace(accelerators=8, all_run_minutes=[13.15873, 13.16148, 13.14803, 13.16812, 13.94403], cv=0.0006353589481, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[13.94403], global_batch=3200, mean_minutes=13.15909, median_minutes=13.160105, name='Dell XE9680, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7097.16722, 'count / s'), published_score=Q_(13.21836607, 'min'), rows=({'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}), run_minutes=[13.15873, 13.16148, 13.14803, 13.16812], scaling_factor=1.004223228, score_minutes=13.16277667, sd_minutes=0.008360745581, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='XE9680x8H100-SXM-80GB', top1=[76.09, 76.144, 75.96, 76.018, 75.894]), 'compute', 'Dell runs'), (SimpleNamespace(accelerators=8, all_run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], cv=0.004751164049, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=13.262936, median_minutes=13.30162, name='NVIDIA Eos, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7039.662034, 'count / s'), published_score=Q_(13.3263435, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}), run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], scaling_factor=1.004223228, score_minutes=13.2703, sd_minutes=0.06301438471, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='eos_ngc24.04_mxnet', top1=[75.936, 75.958, 76.206, 76.056, 75.9]), 'memory', 'Eos runs'))):
        y = 1 - i
        ok_sorted = sorted({'Dell runs': [13.15873, 13.16148, 13.14803, 13.16812], 'Eos runs': [13.3062, 13.3178, 13.30162, 13.20308, 13.18598]}[lab])
        for r in {'Dell runs': ({'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}), 'Eos runs': ({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0})}[lab]:
            x = r['minutes']
            if r['status'] == 'success':
                jit = 0.2 if ok_sorted.index(x) % 2 == 0 else -0.2
                a2.plot(x, y + jit, 'o', ms=8, color=C[ck], mec='white', mew=1.0, alpha=al, zorder=4)
            else:
                a2.plot(x, y, 'o', ms=11, mfc='white', mec=C['danger'], mew=2, alpha=al, zorder=4)
                a2.plot(x, y, 'x', ms=7, color=C['danger'], mew=2, alpha=al, zorder=5)
                a2.text(x - 0.03, y - 0.48, 'missed 75.9%\n(75.894% at epoch 35)', ha='right', va='center', fontsize=9.5, color=C['danger'], fontweight='bold', alpha=al)
        a2.text(13.07, y, lab, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'], alpha=al)
    a2.set_xlim(13.1, 14.0)
    a2.set_ylim(-0.95, 1.5)
    a2.set_yticks([])
    a2.spines['left'].set_visible(False)
    a2.set_xticks([13.2, 13.4, 13.6, 13.8, 14.0])
    a2.set_xlabel("Each run's time to train (minutes)", alpha=al)
    for s_ in a2.spines.values():
        s_.set_alpha(al)
    a2.tick_params(colors=C['text'] if k >= 2 else 'white')
    sp.save(fig, out / f'hook_step{k}.svg')
fig, ax = plt.subplots(figsize=(2.5, 1.45))
for y, lab, v, ck, txt in ((1, 'Datasheet:\n10 TOPS at 0.5 W', 20.0, 'compute', '20 TOPS/W'), (0, 'Sustained:\n3 TOPS at 2 W', 1.5, 'danger', '1.5 TOPS/W')):
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v + 0.6, y, txt, va='center', fontsize=14, fontweight='bold', color=C[ck])
    ax.text(-0.6, y, lab, ha='right', va='center', fontsize=14, fontweight='bold', linespacing=1.0)
ax.set_xlim(0, 30)
ax.set_xticks([0, 10, 20])
ax.tick_params(labelsize=14)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('TOPS per watt', fontsize=14)
sp.save(fig, out / 'recall_hook.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    for i, (s, lab, ck) in enumerate([(SimpleNamespace(accelerators=8, all_run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], cv=0.004751164049, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=13.262936, median_minutes=13.30162, name='NVIDIA Eos, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7039.662034, 'count / s'), published_score=Q_(13.3263435, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}), run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], scaling_factor=1.004223228, score_minutes=13.2703, sd_minutes=0.06301438471, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='eos_ngc24.04_mxnet', top1=[75.936, 75.958, 76.206, 76.056, 75.9]), 'Eos, 8x H100', 'memory'), (SimpleNamespace(accelerators=8, all_run_minutes=[13.15873, 13.16148, 13.14803, 13.16812, 13.94403], cv=0.0006353589481, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[13.94403], global_batch=3200, mean_minutes=13.15909, median_minutes=13.160105, name='Dell XE9680, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7097.16722, 'count / s'), published_score=Q_(13.21836607, 'min'), rows=({'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}), run_minutes=[13.15873, 13.16148, 13.14803, 13.16812], scaling_factor=1.004223228, score_minutes=13.16277667, sd_minutes=0.008360745581, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='XE9680x8H100-SXM-80GB', top1=[76.09, 76.144, 75.96, 76.018, 75.894]), 'Dell, 8x H100', 'compute'), (SimpleNamespace(accelerators=8, all_run_minutes=[12.01915, 12.02395, 12.03392, 12.04, 12.02242], cv=0.0007258054484, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=12.027888, median_minutes=12.02395, name='NVIDIA, 8x H200, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7767.54514, 'count / s'), published_score=Q_(12.07755509, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 0.0, 'minutes': 12.01915, 'top1': 76.072, 'epochs': 35.0, 'status': 'success', 'seed': 16768.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 1.0, 'minutes': 12.02395, 'top1': 76.118, 'epochs': 35.0, 'status': 'success', 'seed': 26019.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 2.0, 'minutes': 12.03392, 'top1': 75.998, 'epochs': 35.0, 'status': 'success', 'seed': 10990.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 3.0, 'minutes': 12.04, 'top1': 75.932, 'epochs': 35.0, 'status': 'success', 'seed': 8496.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 4.0, 'minutes': 12.02242, 'top1': 75.992, 'epochs': 35.0, 'status': 'success', 'seed': 12050.0}), run_minutes=[12.01915, 12.02395, 12.03392, 12.04, 12.02242], scaling_factor=1.004223228, score_minutes=12.02676333, sd_minutes=0.008729906643, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='h200_ngc24.04_mxnet', top1=[76.072, 76.118, 75.998, 75.932, 75.992]), 'NVIDIA, 8x H200', 'data'), (SimpleNamespace(accelerators=64, all_run_minutes=[2.50543, 2.50527, 2.50357, 2.50593, 2.64088], cv=0.0004095449081, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[2.64088], global_batch=3200, mean_minutes=2.50505, median_minutes=2.50535, name='NVIDIA Eos_n8, 64x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=50.0, per_accelerator_rate=Q_(4660.587279, 'count / s'), published_score=Q_(2.516124813, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 0.0, 'minutes': 2.50543, 'top1': 76.074, 'epochs': 35.0, 'status': 'success', 'seed': 19424.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 1.0, 'minutes': 2.50527, 'top1': 76.146, 'epochs': 35.0, 'status': 'success', 'seed': 16963.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 2.0, 'minutes': 2.50357, 'top1': 75.964, 'epochs': 35.0, 'status': 'success', 'seed': 2223.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 3.0, 'minutes': 2.50593, 'top1': 76.02, 'epochs': 35.0, 'status': 'success', 'seed': 7618.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 4.0, 'minutes': 2.64088, 'top1': 75.882, 'epochs': 35.0, 'status': 'aborted', 'seed': 23246.0}), run_minutes=[2.50543, 2.50527, 2.50357, 2.50593], scaling_factor=1.004223228, score_minutes=2.505543333, sd_minutes=0.001025930472, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='eos_n8_ngc24.04_mxnet', top1=[76.074, 76.146, 75.964, 76.02, 75.882]), 'Eos, 64x H100', 'network')]):
        y = 3 - i
        med = {'Eos, 8x H100': 13.30162, 'Dell, 8x H100': 13.160105, 'NVIDIA, 8x H200': 12.02395, 'Eos, 64x H100': 2.50535}[lab]
        ax.text(-1.25, y, lab, ha='right', va='center', fontsize=12, fontweight='bold')
        for r in {'Eos, 8x H100': ({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}), 'Dell, 8x H100': ({'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}), 'NVIDIA, 8x H200': ({'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 0.0, 'minutes': 12.01915, 'top1': 76.072, 'epochs': 35.0, 'status': 'success', 'seed': 16768.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 1.0, 'minutes': 12.02395, 'top1': 76.118, 'epochs': 35.0, 'status': 'success', 'seed': 26019.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 2.0, 'minutes': 12.03392, 'top1': 75.998, 'epochs': 35.0, 'status': 'success', 'seed': 10990.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 3.0, 'minutes': 12.04, 'top1': 75.932, 'epochs': 35.0, 'status': 'success', 'seed': 8496.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 4.0, 'minutes': 12.02242, 'top1': 75.992, 'epochs': 35.0, 'status': 'success', 'seed': 12050.0}), 'Eos, 64x H100': ({'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 0.0, 'minutes': 2.50543, 'top1': 76.074, 'epochs': 35.0, 'status': 'success', 'seed': 19424.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 1.0, 'minutes': 2.50527, 'top1': 76.146, 'epochs': 35.0, 'status': 'success', 'seed': 16963.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 2.0, 'minutes': 2.50357, 'top1': 75.964, 'epochs': 35.0, 'status': 'success', 'seed': 2223.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 3.0, 'minutes': 2.50593, 'top1': 76.02, 'epochs': 35.0, 'status': 'success', 'seed': 7618.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 4.0, 'minutes': 2.64088, 'top1': 75.882, 'epochs': 35.0, 'status': 'aborted', 'seed': 23246.0})}[lab]:
            x = 100 * (r['minutes'] / med - 1)
            if r['status'] == 'success':
                ax.plot(x, y, 'o', ms=10, color=C[ck], mec='white', mew=1.2, zorder=4)
            else:
                ax.plot(1.55, y, 'o', ms=11, mfc='white', mec=C['danger'], mew=2, zorder=4)
                ax.plot(1.55, y, 'x', ms=7, color=C['danger'], mew=2, zorder=5)
                ax.text(1.7, y, f'+{x:.1f}%, missed\ntarget', va='center', fontsize=10.5, color=C['danger'], fontweight='bold')
        ax.text(2.75, y, f"CV { {'Eos, 8x H100': 0.4751164049, 'Dell, 8x H100': 0.06353589481, 'NVIDIA, 8x H200': 0.07258054484, 'Eos, 64x H100': 0.04095449081}[lab]:.2f}%", va='center', fontsize=12, fontweight='bold', color=C[ck], alpha=1.0 if k >= 2 else 0.0)
    ax.axvline(0, color=C['neutral'], lw=1, ls='--', zorder=1)
    ax.axvspan(1.35, 1.75, color='white', zorder=2)
    ax.set_xlim(-1.2, 1.75)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1.0])
    ax.set_xticklabels(['−1%', '−0.5%', 'median', '+0.5%', '+1%'])
    ax.set_xlabel("Run time relative to the system's median run")
    sp.save(fig, out / f'strip_step{k}.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.05))
    offsets = {'eos_ngc24.04_mxnet': -0.27, 'XE9680x8H100-SXM-80GB': -0.09, 'h200_ngc24.04_mxnet': 0.09, 'eos_n8_ngc24.04_mxnet': 0.27}
    colors = {'eos_ngc24.04_mxnet': 'memory', 'XE9680x8H100-SXM-80GB': 'compute', 'h200_ngc24.04_mxnet': 'data', 'eos_n8_ngc24.04_mxnet': 'network'}
    for r in ({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 0.0, 'minutes': 12.01915, 'top1': 76.072, 'epochs': 35.0, 'status': 'success', 'seed': 16768.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 1.0, 'minutes': 12.02395, 'top1': 76.118, 'epochs': 35.0, 'status': 'success', 'seed': 26019.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 2.0, 'minutes': 12.03392, 'top1': 75.998, 'epochs': 35.0, 'status': 'success', 'seed': 10990.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 3.0, 'minutes': 12.04, 'top1': 75.932, 'epochs': 35.0, 'status': 'success', 'seed': 8496.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 4.0, 'minutes': 12.02242, 'top1': 75.992, 'epochs': 35.0, 'status': 'success', 'seed': 12050.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 0.0, 'minutes': 2.50543, 'top1': 76.074, 'epochs': 35.0, 'status': 'success', 'seed': 19424.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 1.0, 'minutes': 2.50527, 'top1': 76.146, 'epochs': 35.0, 'status': 'success', 'seed': 16963.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 2.0, 'minutes': 2.50357, 'top1': 75.964, 'epochs': 35.0, 'status': 'success', 'seed': 2223.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 3.0, 'minutes': 2.50593, 'top1': 76.02, 'epochs': 35.0, 'status': 'success', 'seed': 7618.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 4.0, 'minutes': 2.64088, 'top1': 75.882, 'epochs': 35.0, 'status': 'aborted', 'seed': 23246.0}):
        a = r['top1']
        y = offsets[r['system']]
        if a < TARGET_PCT:
            ax.plot(a, y, 'o', ms=11, mfc='white', mec=C['danger'], mew=2, zorder=4)
            ax.plot(a, y, 'x', ms=7, color=C['danger'], mew=2, zorder=5)
        else:
            ax.plot(a, y, 'o', ms=10, color=C[colors[r['system']]], mec='white', mew=1.2, zorder=4)
    ax.axvline(TARGET_PCT, color=C['danger'], lw=2, ls='--')
    ax.text(TARGET_PCT - 0.01, -1.15, 'target\n75.9%', ha='right', va='center', fontsize=11, color=C['danger'], fontweight='bold')
    for s, lab, ck in [(SimpleNamespace(accelerators=8, all_run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], cv=0.004751164049, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=13.262936, median_minutes=13.30162, name='NVIDIA Eos, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7039.662034, 'count / s'), published_score=Q_(13.3263435, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}), run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], scaling_factor=1.004223228, score_minutes=13.2703, sd_minutes=0.06301438471, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='eos_ngc24.04_mxnet', top1=[75.936, 75.958, 76.206, 76.056, 75.9]), 'Eos, 8x H100', 'memory'), (SimpleNamespace(accelerators=8, all_run_minutes=[13.15873, 13.16148, 13.14803, 13.16812, 13.94403], cv=0.0006353589481, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[13.94403], global_batch=3200, mean_minutes=13.15909, median_minutes=13.160105, name='Dell XE9680, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7097.16722, 'count / s'), published_score=Q_(13.21836607, 'min'), rows=({'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}), run_minutes=[13.15873, 13.16148, 13.14803, 13.16812], scaling_factor=1.004223228, score_minutes=13.16277667, sd_minutes=0.008360745581, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='XE9680x8H100-SXM-80GB', top1=[76.09, 76.144, 75.96, 76.018, 75.894]), 'Dell, 8x H100', 'compute'), (SimpleNamespace(accelerators=8, all_run_minutes=[12.01915, 12.02395, 12.03392, 12.04, 12.02242], cv=0.0007258054484, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=12.027888, median_minutes=12.02395, name='NVIDIA, 8x H200, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7767.54514, 'count / s'), published_score=Q_(12.07755509, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 0.0, 'minutes': 12.01915, 'top1': 76.072, 'epochs': 35.0, 'status': 'success', 'seed': 16768.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 1.0, 'minutes': 12.02395, 'top1': 76.118, 'epochs': 35.0, 'status': 'success', 'seed': 26019.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 2.0, 'minutes': 12.03392, 'top1': 75.998, 'epochs': 35.0, 'status': 'success', 'seed': 10990.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 3.0, 'minutes': 12.04, 'top1': 75.932, 'epochs': 35.0, 'status': 'success', 'seed': 8496.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 4.0, 'minutes': 12.02242, 'top1': 75.992, 'epochs': 35.0, 'status': 'success', 'seed': 12050.0}), run_minutes=[12.01915, 12.02395, 12.03392, 12.04, 12.02242], scaling_factor=1.004223228, score_minutes=12.02676333, sd_minutes=0.008729906643, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='h200_ngc24.04_mxnet', top1=[76.072, 76.118, 75.998, 75.932, 75.992]), 'NVIDIA, 8x H200', 'data'), (SimpleNamespace(accelerators=64, all_run_minutes=[2.50543, 2.50527, 2.50357, 2.50593, 2.64088], cv=0.0004095449081, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[2.64088], global_batch=3200, mean_minutes=2.50505, median_minutes=2.50535, name='NVIDIA Eos_n8, 64x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=50.0, per_accelerator_rate=Q_(4660.587279, 'count / s'), published_score=Q_(2.516124813, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 0.0, 'minutes': 2.50543, 'top1': 76.074, 'epochs': 35.0, 'status': 'success', 'seed': 19424.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 1.0, 'minutes': 2.50527, 'top1': 76.146, 'epochs': 35.0, 'status': 'success', 'seed': 16963.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 2.0, 'minutes': 2.50357, 'top1': 75.964, 'epochs': 35.0, 'status': 'success', 'seed': 2223.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 3.0, 'minutes': 2.50593, 'top1': 76.02, 'epochs': 35.0, 'status': 'success', 'seed': 7618.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 4.0, 'minutes': 2.64088, 'top1': 75.882, 'epochs': 35.0, 'status': 'aborted', 'seed': 23246.0}), run_minutes=[2.50543, 2.50527, 2.50357, 2.50593], scaling_factor=1.004223228, score_minutes=2.505543333, sd_minutes=0.001025930472, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='eos_n8_ngc24.04_mxnet', top1=[76.074, 76.146, 75.964, 76.02, 75.882]), 'Eos, 64x H100', 'network')]:
        ax.text(76.24, offsets[{'Eos, 8x H100': 'eos_ngc24.04_mxnet', 'Dell, 8x H100': 'XE9680x8H100-SXM-80GB', 'NVIDIA, 8x H200': 'h200_ngc24.04_mxnet', 'Eos, 64x H100': 'eos_n8_ngc24.04_mxnet'}[lab]], lab, va='center', fontsize=10, color=C[ck], fontweight='bold')
    al = 1.0 if k >= 2 else 0.0
    lo, hi = (min((r['top1'] for r in ({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 0.0, 'minutes': 12.01915, 'top1': 76.072, 'epochs': 35.0, 'status': 'success', 'seed': 16768.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 1.0, 'minutes': 12.02395, 'top1': 76.118, 'epochs': 35.0, 'status': 'success', 'seed': 26019.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 2.0, 'minutes': 12.03392, 'top1': 75.998, 'epochs': 35.0, 'status': 'success', 'seed': 10990.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 3.0, 'minutes': 12.04, 'top1': 75.932, 'epochs': 35.0, 'status': 'success', 'seed': 8496.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 4.0, 'minutes': 12.02242, 'top1': 75.992, 'epochs': 35.0, 'status': 'success', 'seed': 12050.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 0.0, 'minutes': 2.50543, 'top1': 76.074, 'epochs': 35.0, 'status': 'success', 'seed': 19424.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 1.0, 'minutes': 2.50527, 'top1': 76.146, 'epochs': 35.0, 'status': 'success', 'seed': 16963.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 2.0, 'minutes': 2.50357, 'top1': 75.964, 'epochs': 35.0, 'status': 'success', 'seed': 2223.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 3.0, 'minutes': 2.50593, 'top1': 76.02, 'epochs': 35.0, 'status': 'success', 'seed': 7618.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 4.0, 'minutes': 2.64088, 'top1': 75.882, 'epochs': 35.0, 'status': 'aborted', 'seed': 23246.0}))), max((r['top1'] for r in ({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 0.0, 'minutes': 12.01915, 'top1': 76.072, 'epochs': 35.0, 'status': 'success', 'seed': 16768.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 1.0, 'minutes': 12.02395, 'top1': 76.118, 'epochs': 35.0, 'status': 'success', 'seed': 26019.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 2.0, 'minutes': 12.03392, 'top1': 75.998, 'epochs': 35.0, 'status': 'success', 'seed': 10990.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 3.0, 'minutes': 12.04, 'top1': 75.932, 'epochs': 35.0, 'status': 'success', 'seed': 8496.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 4.0, 'minutes': 12.02242, 'top1': 75.992, 'epochs': 35.0, 'status': 'success', 'seed': 12050.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 0.0, 'minutes': 2.50543, 'top1': 76.074, 'epochs': 35.0, 'status': 'success', 'seed': 19424.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 1.0, 'minutes': 2.50527, 'top1': 76.146, 'epochs': 35.0, 'status': 'success', 'seed': 16963.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 2.0, 'minutes': 2.50357, 'top1': 75.964, 'epochs': 35.0, 'status': 'success', 'seed': 2223.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 3.0, 'minutes': 2.50593, 'top1': 76.02, 'epochs': 35.0, 'status': 'success', 'seed': 7618.0}, {'submitter': 'NVIDIA', 'system': 'eos_n8_ngc24.04_mxnet', 'label': 'Eos 64x H100', 'run': 4.0, 'minutes': 2.64088, 'top1': 75.882, 'epochs': 35.0, 'status': 'aborted', 'seed': 23246.0}))))
    ax.annotate('', xy=(lo, -0.55), xytext=(hi, -0.55), alpha=al, arrowprops=dict(arrowstyle='|-|', color=C['text'], lw=1.5, alpha=al))
    ax.text((lo + hi) / 2, -0.75, 'seeds: 0.32 points', ha='center', va='top', fontsize=11, fontweight='bold', color=C['text'], alpha=al)
    mid, hw = (76.018, 0.3742518948)
    ax.annotate('', xy=(mid - hw, -1.25), xytext=(mid + hw, -1.25), alpha=al, arrowprops=dict(arrowstyle='|-|', color=C['memory'], lw=1.5, alpha=al))
    ax.text(mid + hw + 0.02, -1.25, 'new 50,000-image\ntest set: ±0.37', va='center', fontsize=11, fontweight='bold', color=C['memory'], alpha=al)
    ax.set_xlim(75.65, 76.6)
    ax.set_ylim(-1.6, 0.55)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('ImageNet top-1 at the last evaluation (%), 20 runs')
    sp.save(fig, out / f'seedacc_step{k}.svg')
dell = sorted(({'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}), key=lambda r: r['minutes'])
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 1.7))
    for j, r in enumerate(dell):
        x = r['minutes']
        y = 4 - j
        ok = r['status'] == 'success'
        dropped = j == 0 or not ok
        col = C['neutral'] if k >= 2 and dropped else C['compute']
        xp = x if ok else 13.181
        ax.text(13.1365, y, f'run {j + 1}' if ok else 'run 5', ha='right', va='center', fontsize=11, fontweight='bold', color=C['text'])
        if ok:
            ax.plot(xp, y, 'o', ms=12, color=col, mec='white', mew=1.2, zorder=4)
            left = j > 0 and x < 13.16277667
            ax.text(xp - 0.0016 if left else xp + 0.0016, y + 0.05, f'{x:.3f}', va='center', ha='right' if left else 'left', fontsize=11, color=col, fontweight='bold')
        else:
            ax.plot(xp, y, 'o', ms=12, mfc='white', mec=C['danger'], mew=2, zorder=4)
            ax.text(xp - 0.0015, y, f'{x:.3f}, missed 75.9%, off scale', ha='right', va='center', fontsize=11, color=C['danger'], fontweight='bold')
        if k >= 2 and dropped:
            ax.plot(xp, y, 'x', ms=16, color=C['danger'], mew=3, zorder=6)
            if ok:
                ax.text(xp + 0.0072, y + 0.05, 'fastest: dropped', va='center', fontsize=11, color=C['danger'], fontweight='bold')
            else:
                ax.text(xp - 0.0015, y - 0.5, 'counts as slowest: dropped', ha='right', va='center', fontsize=11, color=C['danger'], fontweight='bold')
    if k >= 3:
        sc = 13.16277667
        ax.plot([sc, sc], [0.6, 3.4], color=C['text'], lw=2.5, zorder=5)
        ax.text(13.1385, 0.0, 'mean of runs 2 to 4:\n13.163 min', ha='left', va='center', fontsize=11, fontweight='bold', color=C['text'])
    ax.set_xlim(13.137, 13.1835)
    ax.set_ylim(-0.8, 4.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([13.14, 13.15, 13.16, 13.17])
    ax.set_xlabel('Dell XE9680, five runs sorted (minutes; run 5 off scale)')
    sp.save(fig, out / f'olympic_step{k}.svg')
fig, ax = plt.subplots(figsize=(7.2, 1.05))
rows = [('accuracy', 1824.69294, 'data', '1,825 images'), ('p90', 23885.62776, 'memory', '23,886 queries'), ('p99', 262741.9054, 'danger', '262,742 queries')]
for i, (lab, v, ck, txt) in enumerate(rows):
    y = 2 - i
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v * 1.15, y, txt, va='center', fontsize=16, fontweight='bold', color=C[ck])
    ax.text(700, y, lab, ha='right', va='center', fontsize=16, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim(800, 6000000.0)
ax.set_xticks([1000.0, 10000.0, 100000.0, 1000000.0])
ax.set_xticklabels(['1,000', '10,000', '100,000', '1,000,000'])
ax.minorticks_off()
ax.tick_params(labelsize=14)
ax.set_ylim(-0.6, 2.6)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Samples needed (log scale)', fontsize=15)
sp.save(fig, out / 'sizing.svg')
for k in (1, 2, 3, 4, 5):
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(3.45, 2.45), gridspec_kw=dict(height_ratios=[1.1, 1, 1], hspace=0.95))
    for y, s, ck, lab in ((1, SimpleNamespace(accelerators=8, all_run_minutes=[13.15873, 13.16148, 13.14803, 13.16812, 13.94403], cv=0.0006353589481, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[13.94403], global_batch=3200, mean_minutes=13.15909, median_minutes=13.160105, name='Dell XE9680, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7097.16722, 'count / s'), published_score=Q_(13.21836607, 'min'), rows=({'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}), run_minutes=[13.15873, 13.16148, 13.14803, 13.16812], scaling_factor=1.004223228, score_minutes=13.16277667, sd_minutes=0.008360745581, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='XE9680x8H100-SXM-80GB', top1=[76.09, 76.144, 75.96, 76.018, 75.894]), 'compute', 'Dell'), (0, SimpleNamespace(accelerators=8, all_run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], cv=0.004751164049, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=13.262936, median_minutes=13.30162, name='NVIDIA Eos, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7039.662034, 'count / s'), published_score=Q_(13.3263435, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}), run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], scaling_factor=1.004223228, score_minutes=13.2703, sd_minutes=0.06301438471, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='eos_ngc24.04_mxnet', top1=[75.936, 75.958, 76.206, 76.056, 75.9]), 'memory', 'Eos')):
        xs = sorted({'Dell': [13.15873, 13.16148, 13.14803, 13.16812], 'Eos': [13.3062, 13.3178, 13.30162, 13.20308, 13.18598]}[lab])
        for n, x in enumerate(xs):
            jit = 0.17 if n % 2 == 0 else -0.17
            a0.plot(x, y + jit, 'o', ms=7, color=C[ck], mec='white', mew=0.8, zorder=4)
        m = st.mean(xs)
        a0.plot([m, m], [y - 0.4, y + 0.4], color=C['text'], lw=2.5, zorder=5)
        a0.text(13.127, y, lab, ha='right', va='center', fontsize=11, fontweight='bold', color=C[ck])
    a0.set_xlim(13.13, 13.33)
    a0.set_ylim(-0.6, 1.6)
    a0.set_yticks([])
    a0.spines['left'].set_visible(False)
    a0.set_xticks([13.15, 13.2, 13.25, 13.3])
    a0.set_xticklabels(['13.15', '13.20', '13.25', '13.30 min'], fontsize=10)
    w = {'d': 0.103846, 'se': 0.02848926206, 'lo': 0.02603966027, 'hi': 0.1816523397}
    on = 1.0 if k >= 2 else 0.0
    a1.axvline(0, color=C['neutral'], lw=1.5, ls='--', alpha=on)
    a1.plot(w['d'], 0.5, 'o', ms=10, color=C['compute'], zorder=5, alpha=on)
    if k >= 3:
        a1.annotate('', xy=(w['d'] - w['se'], 1.15), xytext=(w['d'] + w['se'], 1.15), arrowprops=dict(arrowstyle='|-|', color=C['neutral'], lw=1.5))
        a1.text(w['d'] + w['se'] + 0.008, 1.15, '±1 SE', va='center', fontsize=10.5, color=C['neutral'], fontweight='bold')
    if k >= 4:
        a1.plot([w['lo'], w['hi']], [0.5, 0.5], color=C['compute'], lw=6, solid_capstyle='butt', zorder=4)
        a1.text(w['hi'] + 0.006, 0.5, '0.026 to 0.182', va='center', fontsize=11, fontweight='bold', color=C['compute'])
    a1.text(-0.075, 0.5, 'Eos − Dell', ha='right', va='center', fontsize=11, fontweight='bold', color=C['compute'], alpha=on)
    a1.set_xlim(-0.07, 0.36)
    a1.set_ylim(-0.2, 1.5)
    a1.set_yticks([])
    a1.spines['left'].set_visible(False)
    a1.set_xticks([0, 0.1, 0.2, 0.3])
    a1.set_xticklabels(['0', '0.1', '0.2', '0.3'], fontsize=10)
    for s_ in a1.spines.values():
        s_.set_alpha(on)
    a1.tick_params(colors=C['text'] if k >= 2 else 'white')
    al = 1.0 if k >= 5 else 0.0
    h = {'d': 1.235048, 'se': 0.02845004025, 'lo': 1.157195997, 'hi': 1.312900003}
    a2.axvline(0, color=C['neutral'], lw=1.5, ls='--', alpha=al)
    a2.plot([h['lo'], h['hi']], [0.5, 0.5], color=C['data'], lw=6, solid_capstyle='butt', alpha=al)
    a2.plot(h['d'], 0.5, 'o', ms=10, color=C['data'], alpha=al, zorder=5)
    a2.text(h['lo'] - 0.03, 0.5, '1.16 to 1.31', ha='right', va='center', fontsize=11, fontweight='bold', color=C['data'], alpha=al)
    a2.text(-0.075 * 1.45 / 0.43, 0.5, 'Eos − H200', ha='right', va='center', fontsize=11, fontweight='bold', color=C['data'], alpha=al)
    a2.set_xlim(-0.07 * 1.45 / 0.43, 1.45)
    a2.set_ylim(-0.2, 1.5)
    a2.set_yticks([])
    a2.spines['left'].set_visible(False)
    a2.set_xticks([0, 0.5, 1.0])
    a2.set_xticklabels(['0', '0.5', '1.0'], fontsize=10)
    a2.set_xlabel('Difference in mean run time (min)', fontsize=10.5, labelpad=2, alpha=al)
    for s_ in a2.spines.values():
        s_.set_alpha(al)
    a2.tick_params(colors=C['text'] if k >= 5 else 'white')
    sp.save(fig, out / f'interval_step{k}.svg')
ns = list(range(2, 31))
hw = [4.268751534, 1.180254579, 0.7560162237, 0.5899349349, 0.4986040884, 0.439409387, 0.3972072547, 0.3652067981, 0.3398778014, 0.3191875439, 0.3018745616, 0.28710997, 0.274324045, 0.2631106951, 0.2531716613, 0.2442823143, 0.2362698296, 0.2289987769, 0.2223613222, 0.2162703946, 0.2106548127, 0.2054557404, 0.2006240626, 0.1961184129, 0.191903669, 0.1879497901, 0.1842309091, 0.1807246151, 0.1774113813]
n_lo, _, n_hi = [5, 10, 20]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.plot(ns, hw, color=C['memory'], lw=3)
    for y in reversed([0.5, 1.0]):
        lab = f'{y:g}%'
        ax.axhline(y, color=C['neutral'], lw=1, ls='--')
        ax.text(30, y + 0.04, lab, va='bottom', ha='right', fontsize=11, color=C['neutral'], fontweight='bold')
    h5 = 0.5899349349
    ax.plot(n_lo, h5, 'o', ms=10, color=C['danger'], zorder=5)
    ax.text(n_lo + 1.2, h5 + 0.1, f'{n_lo} runs: ±0.59%', fontsize=11.5, fontweight='bold', color=C['danger'])
    al = 1.0 if k >= 2 else 0.0
    h20 = 0.2223613222
    ax.plot(n_hi, h20, 'o', ms=10, color=C['data'], zorder=5, alpha=al)
    ax.text(n_hi, h20 + 0.17, f'{n_hi} runs: ±0.22%', ha='center', fontsize=11.5, fontweight='bold', color=C['data'], alpha=al)
    ax.set_ylim(0, 1.6)
    ax.set_xlim(2, 31)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_xlabel('Runs averaged')
    ax.set_ylabel('95% half-width (%)')
    ax.grid(color=C['grid'], lw=0.6)
    sp.save(fig, out / f'runs_step{k}.svg')
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.7, 2.2), gridspec_kw=dict(hspace=1.3))
    a1.plot([0, 0], [-0.35, 0.35], color=C['neutral'], lw=1.5, ls='--')
    lo_pts, hi_pts = ((lambda x: 100 * x)(x) for x in (-0.009978177381, 0.02997817738))
    d_pts = 1.0
    a1.plot([lo_pts, hi_pts], [0, 0], color=C['memory'], lw=4)
    a1.plot([d_pts], [0], 'o', color=C['memory'], ms=10)
    a1.text(d_pts, 0.45, 'interval about −1 to 3', ha='center', fontsize=11, fontweight='bold', color=C['memory'])
    a1.text(-0.2, -0.7, 'no change', ha='right', fontsize=10, color=C['neutral'])
    a1.set_xlim(-3, 5)
    a1.set_ylim(-1, 1)
    a1.set_yticks([])
    a1.spines['left'].set_visible(False)
    a1.set_xlabel('Difference (points), if independent sets')
    al = 1.0 if k >= 2 else 0.0
    gap_n, max_n = (10, 26)
    a2.axvspan(gap_n, max_n, color=C['danger_l'], alpha=al)
    a2.axvline(max_n, color=C['danger'], lw=2, ls='--', alpha=al)
    a2.text((gap_n + max_n) / 2, 0.25, 'regression', ha='center', fontsize=11, fontweight='bold', color=C['danger'], alpha=al)
    a2.text(43, 0.25, 'unresolved', ha='center', fontsize=11, fontweight='bold', color=C['neutral'], alpha=al)
    a2.text(max_n + 0.6, -0.6, '26', fontsize=11, fontweight='bold', color=C['danger'], alpha=al)
    a2.set_xlim(gap_n, 60)
    a2.set_ylim(-1, 1)
    a2.set_yticks([])
    a2.spines['left'].set_visible(False)
    a2.set_xlabel('Disagreements, b + c')
    for s_ in a2.spines.values():
        s_.set_alpha(al)
    a2.tick_params(colors=C['text'] if k >= 2 else 'white')
    a2.xaxis.label.set_alpha(al)
    sp.save(fig, out / f'conf_step{k}.svg')
gap, limit = (153, 6093)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.5))
    ax.set_xlim(0, 12000)
    ax.set_ylim(-1, 1)
    al = 1.0 if k >= 2 else 0.0
    ax.axvspan(gap, limit, color=C['memory_l'], alpha=al)
    ax.axvline(limit, color=C['memory'], lw=2, ls='--', alpha=al)
    ax.text((gap + limit) / 2, 0.2, 'runs differ', ha='center', fontsize=11, fontweight='bold', color=C['memory'], alpha=al)
    ax.text((limit + 12000) / 2, 0.2, 'seed noise', ha='center', fontsize=11, fontweight='bold', color=C['neutral'], alpha=al)
    ax.text(limit + 120, -0.65, '6,093', fontsize=11, fontweight='bold', color=C['memory'], alpha=al)
    ax.plot([gap, gap], [-0.4, 0.4], color=C['text'], lw=2)
    ax.text(gap + 120, -0.65, 'at least 153', fontsize=11, fontweight='bold', color=C['text'])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 3000, 6000, 9000, 12000])
    ax.set_xticklabels(['0', '3,000', '6,000', '9,000', '12,000'])
    ax.set_xlabel('Images on which the two runs disagree, b + c (of 50,000)')
    sp.save(fig, out / f'mcn_runs_step{k}.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    t = [i * 20 for i in range(11)]
    lat = [2.4, 2.77278406, 3.052922817, 3.263440731, 3.421640164, 3.540523445, 3.629861532, 3.696997075, 3.747447912, 3.78536057, 3.813851071]
    ax.fill_between(t, [v - 0.08 for v in lat], [v + 0.08 for v in lat], color=C['neutral_l'], zorder=0)
    ax.plot(t, lat, color=C['neutral'], lw=2, zorder=1)
    ax.text(200, lat[-1] + 0.22, 'device warms', ha='right', fontsize=10, color=C['neutral'], fontweight='bold')
    labels = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'] if k == 1 else ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    for tq, lab in zip([Q_(10, 's'), Q_(28, 's'), Q_(46, 's'), Q_(64, 's'), Q_(82, 's'), Q_(100, 's'), Q_(118, 's'), Q_(136, 's'), Q_(154, 's'), Q_(172, 's')], labels):
        x, y = ({10.0: 10, 28.0: 28, 46.0: 46, 64.0: 64, 82.0: 82, 100.0: 100, 118.0: 118, 136.0: 136, 154.0: 154, 172.0: 172}[tq.magnitude], {10.0: 2.59968315, 28.0: 2.894519931, 46.0: 3.122504754, 64.0: 3.298795783, 82.0: 3.435114185, 100.0: 3.540523445, 118.0: 3.622031972, 136.0: 3.68505907, 154.0: 3.733795262, 172.0: 3.7714809}[tq.magnitude])
        ck = 'memory' if lab == 'A' else 'compute'
        ax.plot(x, y, 'o', ms=15, color=C[ck], zorder=4)
        ax.text(x, y, lab, ha='center', va='center', fontsize=9.5, color='white', fontweight='bold', zorder=5)
    msg = 'Blocked order: B looks 20% slower' if k == 1 else 'Interleaved: B looks 4% slower'
    ax.text(5, 4.25, msg, fontsize=12, fontweight='bold', color=C['danger'] if k == 1 else C['data'])
    ax.text(5, 3.95, 'A and B are the same code', fontsize=10, color=C['neutral'])
    ax.set_xlim(0, 200)
    ax.set_ylim(2.0, 4.5)
    ax.set_xlabel('Seconds of continuous inference')
    ax.set_ylabel('ms per inference')
    tag(ax, 'Smooth curve through Lee et al. 2019, Fig. 8 endpoints (iPhone Xs GPU, approximate)', (0.0, -0.42))
    sp.save(fig, out / f'order_step{k}.svg')
ADVICE = {'unresolved': 'unresolved: measure more'}
THRESH = 2.0
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.15))
    ax.axvline(0, color=C['neutral'], lw=1.5, ls='--')
    ax.axvline(THRESH, color=C['danger'], lw=1.5, ls='--')
    ax.text(0.15, 2.62, '0', fontsize=10, color=C['neutral'], fontweight='bold')
    ax.text(THRESH + 0.15, 2.62, '2%: set before the test (example)', fontsize=10, color=C['danger'], fontweight='bold')
    for i, (lab, w, ck, key) in enumerate([('Dell, all five runs', SimpleNamespace(base=SimpleNamespace(accelerators=8, all_run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], cv=0.004751164049, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=13.262936, median_minutes=13.30162, name='NVIDIA Eos, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7039.662034, 'count / s'), published_score=Q_(13.3263435, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}), run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], scaling_factor=1.004223228, score_minutes=13.2703, sd_minutes=0.06301438471, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='eos_ngc24.04_mxnet', top1=[75.936, 75.958, 76.206, 76.056, 75.9]), confidence=0.95, difference=Q_(-0.053142, 'min'), dof=4.257413545, high=Q_(0.3794204312, 'min'), include_failed=True, low=Q_(-0.4857044312, 'min'), minimum_useful=0.02, name='Eos against Dell, every run timed as a run (MLPerf Training v4.0 ResNet-50)', other=SimpleNamespace(accelerators=8, all_run_minutes=[13.15873, 13.16148, 13.14803, 13.16812, 13.94403], cv=0.0006353589481, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[13.94403], global_batch=3200, mean_minutes=13.15909, median_minutes=13.160105, name='Dell XE9680, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7097.16722, 'count / s'), published_score=Q_(13.21836607, 'min'), rows=({'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}), run_minutes=[13.15873, 13.16148, 13.14803, 13.16812], scaling_factor=1.004223228, score_minutes=13.16277667, sd_minutes=0.008360745581, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='XE9680x8H100-SXM-80GB', top1=[76.09, 76.144, 75.96, 76.018, 75.894]), standard_error=Q_(0.1595301852, 'min'), t_critical=2.71147702, verdict='unresolved'), 'neutral', 'dellall'), ('Dell, converged runs', SimpleNamespace(base=SimpleNamespace(accelerators=8, all_run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], cv=0.004751164049, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=13.262936, median_minutes=13.30162, name='NVIDIA Eos, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7039.662034, 'count / s'), published_score=Q_(13.3263435, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}), run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], scaling_factor=1.004223228, score_minutes=13.2703, sd_minutes=0.06301438471, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='eos_ngc24.04_mxnet', top1=[75.936, 75.958, 76.206, 76.056, 75.9]), confidence=0.95, difference=Q_(0.103846, 'min'), dof=4.175280908, high=Q_(0.1816523397, 'min'), include_failed=False, low=Q_(0.02603966027, 'min'), minimum_useful=0.02, name='Eos against Dell, 8x H100 each, converged runs (MLPerf Training v4.0 ResNet-50)', other=SimpleNamespace(accelerators=8, all_run_minutes=[13.15873, 13.16148, 13.14803, 13.16812, 13.94403], cv=0.0006353589481, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[13.94403], global_batch=3200, mean_minutes=13.15909, median_minutes=13.160105, name='Dell XE9680, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7097.16722, 'count / s'), published_score=Q_(13.21836607, 'min'), rows=({'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 0.0, 'minutes': 13.15873, 'top1': 76.09, 'epochs': 35.0, 'status': 'success', 'seed': 1304.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 1.0, 'minutes': 13.16148, 'top1': 76.144, 'epochs': 35.0, 'status': 'success', 'seed': 19251.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 2.0, 'minutes': 13.14803, 'top1': 75.96, 'epochs': 35.0, 'status': 'success', 'seed': 4927.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 3.0, 'minutes': 13.16812, 'top1': 76.018, 'epochs': 35.0, 'status': 'success', 'seed': 27344.0}, {'submitter': 'Dell', 'system': 'XE9680x8H100-SXM-80GB', 'label': 'Dell XE9680 8x H100', 'run': 4.0, 'minutes': 13.94403, 'top1': 75.894, 'epochs': 35.0, 'status': 'aborted', 'seed': 8443.0}), run_minutes=[13.15873, 13.16148, 13.14803, 13.16812], scaling_factor=1.004223228, score_minutes=13.16277667, sd_minutes=0.008360745581, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='XE9680x8H100-SXM-80GB', top1=[76.09, 76.144, 75.96, 76.018, 75.894]), standard_error=Q_(0.02848926206, 'min'), t_critical=2.731075995, verdict='real, below threshold'), 'compute', 'dell'), ('H200', SimpleNamespace(base=SimpleNamespace(accelerators=8, all_run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], cv=0.004751164049, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=13.262936, median_minutes=13.30162, name='NVIDIA Eos, 8x H100, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7039.662034, 'count / s'), published_score=Q_(13.3263435, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 0.0, 'minutes': 13.3062, 'top1': 75.936, 'epochs': 35.0, 'status': 'success', 'seed': 5675.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 1.0, 'minutes': 13.3178, 'top1': 75.958, 'epochs': 35.0, 'status': 'success', 'seed': 21142.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 2.0, 'minutes': 13.30162, 'top1': 76.206, 'epochs': 35.0, 'status': 'success', 'seed': 13541.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 3.0, 'minutes': 13.20308, 'top1': 76.056, 'epochs': 35.0, 'status': 'success', 'seed': 11999.0}, {'submitter': 'NVIDIA', 'system': 'eos_ngc24.04_mxnet', 'label': 'Eos 8x H100', 'run': 4.0, 'minutes': 13.18598, 'top1': 75.9, 'epochs': 35.0, 'status': 'success', 'seed': 16155.0}), run_minutes=[13.3062, 13.3178, 13.30162, 13.20308, 13.18598], scaling_factor=1.004223228, score_minutes=13.2703, sd_minutes=0.06301438471, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='eos_ngc24.04_mxnet', top1=[75.936, 75.958, 76.206, 76.056, 75.9]), confidence=0.95, difference=Q_(1.235048, 'min'), dof=4.153486377, high=Q_(1.312900003, 'min'), include_failed=False, low=Q_(1.157195997, 'min'), minimum_useful=0.02, name='Eos (8x H100) against 8x H200, same host and software (MLPerf Training v4.0 ResNet-50)', other=SimpleNamespace(accelerators=8, all_run_minutes=[12.01915, 12.02395, 12.03392, 12.04, 12.02242], cv=0.0007258054484, dataset=SimpleNamespace(categorical_features=None, dense_features=None, embodiments=None, episodes=None, full_examples=Q_(14197122, 'count'), image_channels=3, image_height=224, image_width=224, keywords=None, languages=None, mean_compressed_image_size=Q_(115657.3434, 'B'), metadata=SimpleNamespace(last_verified=None, source_label='Russakovsky et al. (2015), ImageNet Large Scale Visual Recognition Challenge', version=None), name='ImageNet-1k', num_classes=1000, sample_duration=None, sample_rate=None, sample_width=None, tasks=None, test_examples=Q_(100000, 'count'), total_duration=None, training_archive_size=Q_(138, 'GiB'), training_examples=Q_(1281167, 'count'), validation_examples=Q_(50000, 'count')), epochs=35.0, failed_run_minutes=[], global_batch=3200, mean_minutes=12.027888, median_minutes=12.02395, name='NVIDIA, 8x H200, ResNet-50, MLPerf Training v4.0', per_accelerator_batch=400.0, per_accelerator_rate=Q_(7767.54514, 'count / s'), published_score=Q_(12.07755509, 'min'), rows=({'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 0.0, 'minutes': 12.01915, 'top1': 76.072, 'epochs': 35.0, 'status': 'success', 'seed': 16768.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 1.0, 'minutes': 12.02395, 'top1': 76.118, 'epochs': 35.0, 'status': 'success', 'seed': 26019.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 2.0, 'minutes': 12.03392, 'top1': 75.998, 'epochs': 35.0, 'status': 'success', 'seed': 10990.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 3.0, 'minutes': 12.04, 'top1': 75.932, 'epochs': 35.0, 'status': 'success', 'seed': 8496.0}, {'submitter': 'NVIDIA', 'system': 'h200_ngc24.04_mxnet', 'label': 'NVIDIA 8x H200', 'run': 4.0, 'minutes': 12.02242, 'top1': 75.992, 'epochs': 35.0, 'status': 'success', 'seed': 12050.0}), run_minutes=[12.01915, 12.02395, 12.03392, 12.04, 12.02242], scaling_factor=1.004223228, score_minutes=12.02676333, sd_minutes=0.008729906643, series_name='slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv', system='h200_ngc24.04_mxnet', top1=[76.072, 76.118, 75.998, 75.932, 75.992]), standard_error=Q_(0.02845004025, 'min'), t_critical=2.736446162, verdict='real and useful'), 'data', 'h200')]):
        y = 2 - i
        lo, hi = ({'Dell, all five runs': -3.662118487, 'Dell, converged runs': 0.1963340566, 'H200': 8.725036421}[lab], {'Dell, all five runs': 2.860757461, 'Dell, converged runs': 1.369623888, 'H200': 9.899014844}[lab])
        verdict = ADVICE.get({'Dell, all five runs': 'unresolved', 'Dell, converged runs': 'real, below threshold', 'H200': 'real and useful'}[lab], {'Dell, all five runs': 'unresolved', 'Dell, converged runs': 'real, below threshold', 'H200': 'real and useful'}[lab])
        ax.plot([lo, hi], [y, y], color=C[ck], lw=6, solid_capstyle='butt')
        ax.plot({'Dell, all five runs': -0.400680513, 'Dell, converged runs': 0.7829789724, 'H200': 9.312025633}[lab], y, 'o', ms=9, color='white', mec=C[ck], mew=2, zorder=5)
        ax.text(-4.5, y, lab, ha='right', va='center', fontsize=12, fontweight='bold')
        xt = max(hi, THRESH) + 0.4
        ax.text(xt, y + 0.12, f"{ {'Dell, all five runs': '-3.7', 'Dell, converged runs': '0.2', 'H200': '8.7'}[lab]} to { {'Dell, all five runs': '2.9', 'Dell, converged runs': '1.4', 'H200': '9.9'}[lab]}%", va='center', fontsize=12, fontweight='bold', color=C[ck])
        ax.text(xt, y - 0.3, verdict, va='center', fontsize=11.5, fontweight='bold', color=C[ck], alpha=1.0 if k >= 2 else 0.0)
    ax.set_xlim(-4.2, 17)
    ax.set_ylim(-0.6, 2.9)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Percent faster than Eos (95% interval)')
    sp.save(fig, out / f'useful_step{k}.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    lat0, lat1, slo = (50.0, 200.0, 100.0)
    ax.bar([0], [lat0], color=C['memory'], width=0.6)
    al = 1.0 if k >= 2 else 0.0
    ax.bar([1], [lat1], color=C['danger'], width=0.6, alpha=al)
    ax.axhline(slo, color=C['data'], lw=1.5, ls='--')
    ax.text(0.5, slo + 6, 'SLO\n100 ms', fontsize=10, color=C['data'], fontweight='bold', va='bottom', ha='center')
    ax.text(0, lat0 + 8, '50 ms', ha='center', fontsize=12, fontweight='bold')
    ax.text(1, lat1 + 8, '200 ms', ha='center', fontsize=12, fontweight='bold', alpha=al)
    if k == 1:
        ax.text(1, 40, '???', ha='center', fontsize=16, fontweight='bold', color=C['danger'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['beam 1', 'beam 10\n(+0.5 BLEU)'], fontsize=11)
    ax.set_ylim(0, 245)
    ax.set_ylabel('latency (ms)')
    tag(ax, 'Illustrative (Example 12.1)', (0.0, -0.42))
    sp.save(fig, out / f'beam_step{k}.svg')
fig, ax = plt.subplots(figsize=(6.2, 0.95))
for y, lab, v, ck, txt in ((1, 'GFXBench 2.7: 480 MHz', 37.9, 'memory', '37.9M triangles/s'), (0, 'GLBenchmark 2.5.1: 532 MHz', 43.2, 'danger', '43.2M  (+14%)')):
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(v + 0.6, y, txt, va='center', fontsize=12, fontweight='bold', color=C[ck])
    ax.text(-0.6, y, lab, ha='right', va='center', fontsize=12, fontweight='bold')
ax.set_xlim(0, 62)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Triangle throughput, identical test, mean of 5 runs')
sp.save(fig, out / 'fingerprint.svg')
rows = [('pnasnet_large_tf', 82.9, 72.2), ('pnasnet5large', 82.7, 72.1), ('nasnet_large_tf', 82.7, 72.2), ('nasnetalarge', 82.5, 72.2), ('senet154', 81.3, 70.5), ('polynet', 80.9, 70.3), ('inception_resnet_v2_tf', 80.4, 69.7), ('inceptionresnetv2', 80.3, 69.6), ('se_resnext101_32x4d', 80.2, 69.3), ('inception_v4_tf', 80.2, 68.8), ('inceptionv4', 80.1, 69.1), ('dpn107', 79.7, 68.1), ('dpn131', 79.4, 67.9), ('dpn92', 79.4, 67.3), ('dpn98', 79.2, 67.8), ('se_resnext50_32x4d', 79.1, 67.9), ('resnext101_64x4d', 79.0, 67.1), ('xception', 78.8, 67.2), ('se_resnet152', 78.7, 67.5), ('se_resnet101', 78.4, 67.2), ('resnet152', 78.3, 67.0), ('resnext101_32x4d', 78.2, 66.2), ('inception_v3_tf', 78.0, 66.1), ('resnet_v2_152_tf', 77.8, 66.1), ('se_resnet50', 77.6, 66.2), ('fbresnet152', 77.4, 65.8), ('resnet101', 77.4, 65.7), ('inceptionv3', 77.3, 65.7), ('inception_v3', 77.2, 65.4), ('densenet161', 77.1, 65.3), ('dpn68b', 77.0, 64.7), ('resnet_v2_101_tf', 77.0, 64.6), ('densenet201', 76.9, 64.7), ('resnet_v1_152_tf', 76.8, 64.6), ('resnet_v1_101_tf', 76.4, 64.5), ('cafferesnet101', 76.2, 64.3), ('resnet50', 76.1, 63.3), ('dpn68', 75.9, 63.4), ('densenet169', 75.6, 63.9), ('resnet_v2_50_tf', 75.6, 62.7), ('resnet_v1_50_tf', 75.2, 62.6), ('densenet121', 74.4, 62.2), ('vgg19_bn', 74.2, 61.9), ('pnasnet_mobile_tf', 74.1, 60.9), ('nasnetamobile', 74.1, 61.6), ('inception_v2_tf', 74.0, 61.2), ('nasnet_mobile_tf', 74.0, 60.8), ('bninception', 73.5, 62.1), ('vgg16_bn', 73.4, 60.8), ('resnet34', 73.3, 61.2), ('vgg19', 72.4, 59.7), ('vgg16', 71.6, 58.8), ('vgg13_bn', 71.6, 59.0), ('mobilenet_v1_tf', 71.0, 57.4), ('vgg_19_tf', 71.0, 58.6), ('vgg_16_tf', 70.9, 58.4), ('vgg11_bn', 70.4, 57.4), ('vgg13', 69.9, 57.1), ('inception_v1_tf', 69.8, 56.6), ('resnet18', 69.8, 57.2), ('vgg11', 69.0, 55.8), ('squeezenet1_1', 58.2, 45.3), ('squeezenet1_0', 58.1, 45.0), ('alexnet', 56.5, 44.0)]
fig, ax = plt.subplots(figsize=(3.7, 2.75))
ax.plot([55, 85], [55, 85], color=C['text'], lw=1.8, ls='--')
ax.text(66.5, 70.5, 'same accuracy', rotation=31, fontsize=10.5, color=C['text'], fontweight='bold', ha='center')
ax.plot([o for _, o, _ in rows], [n for _, _, n in rows], 'o', ms=5.5, color=C['memory'], mec='white', mew=0.5, zorder=4)
ax.text(84, 50.5, '64 models\n10.3 to 13.6\npoints lower;\norder kept', ha='right', va='top', fontsize=11, fontweight='bold', color=C['memory'])
ax.set_xlim(55, 85)
ax.set_ylim(40, 75)
ax.set_xticks([60, 70, 80])
ax.set_yticks([40, 50, 60, 70])
ax.set_xlabel('Original ImageNet top-1 (%)')
ax.set_ylabel('Rebuilt test set (%)')
ax.grid(color=C['grid'], lw=0.6)
sp.save(fig, out / 'recht.svg')
fig, ax = plt.subplots(figsize=(2.2, 2.75))
ax.bar(0, 8.0, color=C['danger'], width=0.5)
ax.text(0, 8.0 + 0.5, 'up to 8%', ha='center', fontsize=12, fontweight='bold', color=C['danger'])
ax.set_xticks([0])
ax.set_xticklabels(['GSM1k\nagainst GSM8k'], fontsize=11)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(0, 15)
ax.set_ylabel('accuracy drop (%)')
sp.save(fig, out / 'gsm1k.svg')
pts = [('MobileNetV1', 113.0, 70.6, False), ('MobileNetV2', 75.0, 72.0, True), ('NASNet-A', 183.0, 74.0, False), ('MobileNetV2 (1.4)', 143.0, 74.7, True)]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    for name, lat, acc, front in pts:
        col = C['memory'] if k == 1 else C['data'] if front else C['neutral']
        ax.plot(lat, acc, 'o', ms=13, color=col, mec='white', mew=1.5, zorder=5)
        dx, dy, ha, va = (10, -2, 'left', 'top')
        if name == 'NASNet-A':
            dx, dy, ha, va = (0, -12, 'center', 'top')
        if name == 'MobileNetV2 (1.4)':
            dx, dy, ha, va = (-12, 4, 'right', 'bottom')
        lab = name if k == 1 or front else name + ' (dominated)'
        ax.annotate(f'{lab}\n{acc:.1f}%, {lat:.0f} ms', (lat, acc), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=11, fontweight='bold', color=col)
    if k == 2:
        front = sorted(((lat, acc) for _, lat, acc, f in pts if f))
        ax.plot([p[0] for p in front], [p[1] for p in front], color=C['data'], lw=3, zorder=3)
    ax.set_xlim(55, 215)
    ax.set_ylim(69.0, 76.2)
    ax.set_xlabel('Latency on a Pixel 1 CPU core (ms)')
    ax.set_ylabel('ImageNet top-1 (%)')
    ax.grid(color=C['grid'], lw=0.6)
    sp.save(fig, out / f'pareto_step{k}.svg')
chu = [('MobileNetV1', 54.7, 3.72), ('MobileNetV2', 38.8, 4.97), ('MobileNetV3 Large', 20.3, 4.51), ('MobileNet-EdgeTPU', 59.3, 4.29)]
col = {'MobileNetV1': 'memory', 'MobileNetV2': 'neutral', 'MobileNetV3 Large': 'data', 'MobileNet-EdgeTPU': 'network'}
fig, ax = plt.subplots(figsize=(5.8, 2.2))
cpu_rank = {n: i for i, (n, _, _) in enumerate(sorted(chu, key=lambda r: r[1]))}
dsp_rank = {n: i for i, (n, _, _) in enumerate(sorted(chu, key=lambda r: r[2]))}
for n, cpu, dsp in chu:
    y0, y1 = (3 - cpu_rank[n], 3 - dsp_rank[n])
    ax.plot([0, 1], [y0, y1], color=C[col[n]], lw=3, marker='o', ms=10)
    ax.text(-0.09, y0, f'{n}  {cpu:.1f} ms', ha='right', va='center', fontsize=11, fontweight='bold', color=C[col[n]])
    ax.text(1.09, y1, f'{dsp:.2f} ms  {n}', ha='left', va='center', fontsize=11, fontweight='bold', color=C[col[n]])
ax.text(0, 3.6, 'CPU (float)', ha='center', fontsize=12, fontweight='bold')
ax.text(1, 3.6, 'DSP', ha='center', fontsize=12, fontweight='bold')
ax.text(0.5, -0.75, 'fastest at top; top-1 75.1 to 76.2% for all four', ha='center', fontsize=10, color=C['neutral'])
ax.set_xlim(-1.4, 2.4)
ax.set_ylim(-0.9, 3.9)
ax.axis('off')
sp.save(fig, out / 'lottery.svg')
bins = [(20, 0.65, 0.6), (30, 0.85, 0.8), (50, 0.97, 0.84)]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot([0.5, 1.0], [0.5, 1.0], color=C['data'], lw=2, ls='--')
    ax.text(0.52, 0.78, 'dashed: perfect\ncalibration', fontsize=10, color=C['data'], fontweight='bold')
    for n, c, a in bins:
        w = 0.022 + 0.0011 * n
        ax.add_patch(Rectangle((c - w / 2, 0.5), w, a - 0.5, color=C['memory'], zorder=2))
        ax.text(c, 0.505, f'{n}', ha='center', va='bottom', fontsize=10, color='white', fontweight='bold', zorder=4)
        if k >= 2:
            ax.add_patch(Rectangle((c - w / 2, a), w, c - a, color=C['danger_l'], ec=C['danger'], lw=1.2, zorder=3))
            ax.text(c - w / 2 - 0.004, c + 0.004, f"{ {20: '0.05', 30: '0.05', 50: '0.13'}[n]}", ha='right', va='bottom', fontsize=11, fontweight='bold', color=C['danger'])
    if k >= 3:
        ax.text(0.52, 0.95, 'ECE = 0.09', fontsize=14, fontweight='bold', color=C['danger'])
    ax.set_xlim(0.5, 1.03)
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel('confidence in the bin')
    ax.set_ylabel('accuracy in the bin')
    tag(ax, f'Illustrative: 100 predictions in {len(bins)} bins', (0.0, -0.4))
    sp.save(fig, out / f'ece_step{k}.svg')
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.1), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.55))
    xs = [0, 1]
    err, ece_ = (['guo_lenet_err', 'guo_resnet_err'], ['guo_lenet_ece', 'guo_resnet_ece'])
    a1.bar([x - 0.2 for x in xs], [_NUMS[k_] for k_ in err], width=0.38, color=C['neutral'])
    a1.bar([x + 0.2 for x in xs], [_NUMS[k_] for k_ in ece_], width=0.38, color=C['danger'])
    for x, e, c in zip(xs, err, ece_):
        a1.text(x - 0.2, {'guo_lenet_ece': 44.9, 'guo_resnet_ece': 30.6}[c] + 1.2, {'guo_lenet_ece': '44.9', 'guo_resnet_ece': '30.6'}[c], ha='center', fontsize=10.5, fontweight='bold', color=C['neutral'])
        a1.text(x + 0.2, {'guo_lenet_ece': 4.85, 'guo_resnet_ece': 16.53}[c] + 1.2, {'guo_lenet_ece': '4.85', 'guo_resnet_ece': '16.53'}[c], ha='center', fontsize=10.5, fontweight='bold', color=C['danger'])
    a1.set_xticks(xs)
    a1.set_xticklabels(['LeNet 5\n(1998)', 'ResNet-110\n(2016)'], fontsize=10.5)
    a1.set_ylim(0, 55)
    a1.set_ylabel('percent')
    a1.text(0.62, 0.97, 'error', transform=a1.transAxes, fontsize=10.5, fontweight='bold', color=C['neutral'], va='top')
    a1.text(0.62, 0.85, 'ECE', transform=a1.transAxes, fontsize=10.5, fontweight='bold', color=C['danger'], va='top')
    a1.set_title('CIFAR-100', fontsize=12, fontweight='bold')
    al = 1.0 if k >= 2 else 0.0
    ts = ['guo_in_ece', 'guo_in_ece_ts']
    a2.bar([0, 1], [_NUMS[k_] for k_ in ts], width=0.55, color=[C['danger'], C['data']], alpha=al)
    for x, k_ in zip([0, 1], ts):
        a2.text(x, {'guo_in_ece': 5.48, 'guo_in_ece_ts': 1.86}[k_] + 0.25, {'guo_in_ece': '5.48', 'guo_in_ece_ts': '1.86'}[k_], ha='center', fontsize=10.5, fontweight='bold', alpha=al)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(['before', 'after\ntemp. scaling'], fontsize=10.5)
    a2.set_ylim(0, 7.5)
    a2.set_ylabel('ECE (%)', alpha=al)
    a2.set_title('ImageNet, ResNet-152', fontsize=12, fontweight='bold', alpha=al)
    for s_ in a2.spines.values():
        s_.set_alpha(al)
    a2.tick_params(colors=C['text'] if k >= 2 else 'white')
    sp.save(fig, out / f'guo_step{k}.svg')
fig, ax = plt.subplots(figsize=(3.5, 2.7))
groups = [('Average\naccuracy', 'wilds_id_acc', float(0.3), 'wilds_ood_acc', float(2.5), 'wilds_acc_drop'), ('Macro F1', 'wilds_id_f1', float(1.4), 'wilds_ood_f1', float(1.3), 'wilds_f1_drop')]
for i, (lab, idk, ids, oodk, oods, dropk) in enumerate(groups):
    idv, oodv = ({'wilds_ood_acc': 75.7, 'wilds_ood_f1': 47.0}[oodk], {'wilds_ood_acc': 71.6, 'wilds_ood_f1': 31.0}[oodk])
    x = i * 1.6
    ax.bar(x - 0.28, idv, yerr=ids, width=0.5, color=C['memory'], capsize=4, error_kw=dict(lw=1.2, ecolor=C['text']))
    ax.bar(x + 0.28, oodv, yerr=oods, width=0.5, color=C['danger'], capsize=4, error_kw=dict(lw=1.2, ecolor=C['text']))
    ax.text(x - 0.28, idv + 3.5, {'wilds_ood_acc': '75.7', 'wilds_ood_f1': '47.0'}[oodk], ha='center', fontsize=11, fontweight='bold', color=C['memory'])
    ax.text(x + 0.28, oodv + 3.5, {'wilds_ood_acc': '71.6', 'wilds_ood_f1': '31.0'}[oodk], ha='center', fontsize=11, fontweight='bold', color=C['danger'])
    ax.text(x, 104, f"−{ {'wilds_ood_acc': '4.1', 'wilds_ood_f1': '16.0'}[oodk]}", ha='center', fontsize=13, fontweight='bold', color=C['danger'])
ax.set_xticks([0, 1.6])
ax.set_xticklabels([g[0] for g in groups], fontsize=11.5)
ax.set_xlim(-0.65, 2.25)
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75])
ax.set_ylabel('percent')
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=C['memory']), Rectangle((0, 0), 1, 1, color=C['danger'])], labels=['traps seen in training', 'unseen traps'], fontsize=11, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=1.0, columnspacing=1.0)
sp.save(fig, out / 'wilds.svg')
print('figures written')
