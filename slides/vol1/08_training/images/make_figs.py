#!/usr/bin/env python3
"""Data plots and diagrams for Lecture 8 (Model Training, Part 1).

Every number is read from MLSysIM through calcs.py (val() for a printed
label, num() for a plotted value, or the scenario itself for a curve):
  - GPT-2 FLOPs per generated token and per run (Models.Language.GPT2).
  - Training work per token: 2P forward, 4P backward (the 6P rule).
  - Published MFU on one definition: PaLM Table 3 (Literature.TrainingRuns);
    the GPT-2 XL scenario's assumed utilization (Gpt2XlRunOnOneH100).
  - Bytes per parameter of SGD, momentum, and Adam in FP32, and of
    mixed-precision Adam (TrainingStateFitScenario).
  - GPT-2 XL at batch 32, 1,024 tokens, mixed precision: static state,
    per-token activations, and attention scores (Gpt2XlBatch32Activations);
    the H100's 80 GiB in decimal GB.
  - Llama 3.1 8B full and LoRA fine-tuning (Llama3_8B*OnH100).
  - ResNet-50 feeding (ResNet50H100InputFeed).
Illustrative figures (the PaLM loss-spike shape, the MFU bars of the
busy-time scenarios) say so on the figure. Figures that moved to Lecture 8b
(bandwidth ladder, precision run time, FlashAttention tile, checkpoint
segments, scoreboard) and the dropped pipeline and overlap diagrams are no
longer generated here (2026-10-18).

Run from this folder: python3 make_figs.py
"""
_VALS = {'h100_peak': '989', 'h100_bw': '3.35', 'h100_tdp': '700', 'h100_gib': '80', 'h100_gb': '85.9', 'gpt2_params_m': '1.5', 'gpt2_params_e': '9', 'gpt2_params_b': '1.5', 'gpt2_layers': '48', 'gpt2_hidden': '1600', 'gpt2_hidden_c': '1,600', 'gpt2_heads': '25', 'gpt2_ctx': '1024', 'gpt2_fwd_m': '3', 'gpt2_fwd_e': '9', 'gpt2_run_m': '1.5', 'gpt2_run_e': '21', 'tokens_ratio_m': '5', 'tokens_ratio_e': '11', 'tokens_ratio_b': '500', 'tokens_ratio_orders_word': 'twelve', 'batch': '32', 'seq': '1,024', 'tokens_step': '32,768', 'step_tflop': '295', 'step_flop_m': '2.95', 'step_flop_e': '14', 'layer_step_tflop': '320.8', 'attn_extra_pct': '9', 'eta': '0.50', 'eta_pct': '50', 'step_s': '0.60', 'steps_m': '5.1', 'run_tokens_m': '1.7', 'run_tokens_e': '11', 'eta_half_pct': '25', 'step_s_half': '1.2', 'run_s_m': '3.0', 'run_s_e': '6', 'run_days': '35', 'run_kwh': '590', 'price_hr': '4', 'run_usd': '3,370', 'point_hours': '16.5', 'point_pct': '2', 'node_gpus_word': 'eight', 'node_days': '4.4', 'card_8b_mh': '1.46', 'card_70b_mh': '7.0', 'card_405b_mh': '30.84', 'card_total_mh': '39.3', 'card_w': '700', 'l405_params_b': '405', 'l405_tokens_t': '15.6', 'l405_6pd_m': '3.79', 'l405_6pd_e': '25', 'l405_paper_m': '3.8', 'l405_paper_e': '25', 'sec_per_hr': '3,600', 'card_rate': '341', 'card_util_pct': '34.5', 'card_gwh': '21.6', 'card_over_run': '36,600', 'l405_mfu_lo': '38', 'l405_mfu_hi': '43', 'l405_interruptions': '466', 'l405_snapshot_days': '54', 'card_tco2': '8,930', 'fp32_b': '4', 'sgd_b': '8', 'momentum_b': '12', 'adam_fp32_b': '16', 'mp_w_b': '2', 'mp_master_b': '4', 'mp_moments_b': '8', 'mp_total_b': '16', 'mp_total_b_word': 'sixteen', 'mp_zero_k': '12', 'mp_copies_word': 'eight', 'mp_w_gb': '3', 'mp_g_gb': '3', 'mp_master_gb': '6', 'mp_moments_gb': '12', 'mp_state_gb': '24', 'fp32_w_gb': '6', 'fp32_g_gb': '6', 'fp32_state_gb': '24', 'state_share_pct': '28', 'zero_act_gb': '60', 'b7_params_b': '7', 'b7_w_gb': '14', 'b7_g_gb': '14', 'b7_master_gb': '28', 'b7_moments_gb': '56', 'b7_state_gb': '112', 'upd_read_b': '16', 'upd_write_b': '12', 'upd_b': '28', 'upd_gb': '42', 'upd_ms': '12.5', 'upd_share_pct': '2', 'act_tok_gb': '86', 'act_scores_gb': '201', 'act_gb': '287', 'act_gb_1': '286.9', 'seq2': '2,048', 'act2k_tok_gb': '171', 'act2k_scores_gb': '805', 'act_items_gb': '242', 'act_extra_gb': '45', 'layer_scores_mb': '4,194', 'layer_items_mb': '5,043', 'layer_scores_pct': '83', 'step_gb': '311', 'step_x': '3.6', 'fp32_step_gb': '598', 'fp32_step_h100s_word': 'seven', 'l8_params_b': '8.0', 'l8_params_b0': '8', 'l8_act_gb': '36.5', 'l8_batch': '4', 'l8_seq': '2,048', 'l8_lora_m': '6.8', 'l8_lora_pct': '0.08', 'l8_w_gb': '16', 'l8_g_gb': '16', 'l8_master_gb': '32', 'l8_moments_gb': '64', 'l8_state_gb': '128', 'l8_full_gb': '165', 'l8_adapter_gb': '0.11', 'l8_lora_gb': '53', 'l8_headroom_gb': '33', 'l8_rest_gb': '112', 'lora_gpt3_full_tb': '1.2', 'lora_gpt3_gb': '350', 'mfu_gpt3': '21.3', 'mfu_gopher': '32.5', 'gopher_chips': '4,096', 'mtnlg_gpus': '2,240', 'mfu_mtnlg': '30.2', 'mfu_palm': '46.2', 'hfu_palm': '57.8', 'megatron_lo': '44', 'megatron_hi': '52', 'res_gflop': '24.6', 'res_a100': '3,200', 'res_h100': '10,100', 'res_workers': '14', 'res_worker_rate': '375', 'res_supply': '5,250', 'res_busy_pct': '52', 'gil_prep_ms': '250', 'gil_gpu_ms': '12', 'gil_busy_pct': '4.6', 'gil_step_ms': '262', 'gil_fast_step_ms': '256', 'ckpt_b': '14', 'ckpt_gb': '21', 'nvme_gbs': '7', 'ckpt_s': '3', 'ckpt_often': '100', 'ckpt_rarely': '10,000', 'ckpt_work_s': '60', 'ckpt_stall_pct': '5', 'ckpt_rare_s': '5,960', 'ckpt_replay_min': '99', 'palm_params_b': '540', 'palm_chips': '6,144', 'palm_pods_word': 'two', 'palm_pod_chips': '3,072', 'palm_spikes': '20', 'palm_rewind': '100', 'palm_skip_lo': '200', 'palm_skip_hi': '500', 'palm_state_tb': '8.6', 'palm_h100s': '100', 'palm_ckpt_tb': '7.6', 'opt_gpus': '992', 'opt_manual': '35', 'opt_auto': '70', 'opt_runs': '22', 'opt_nov28_word': 'seven'}
_NUMS = {'h100_peak': 989.0, 'h100_bw': 3.35, 'h100_tdp': 700.0, 'h100_gib': 80.0, 'h100_gb': 85.9, 'gpt2_params_m': 1.5, 'gpt2_params_e': 9.0, 'gpt2_params_b': 1.5, 'gpt2_layers': 48.0, 'gpt2_hidden': 1600.0, 'gpt2_hidden_c': 1600.0, 'gpt2_heads': 25.0, 'gpt2_ctx': 1024.0, 'gpt2_fwd_m': 3.0, 'gpt2_fwd_e': 9.0, 'gpt2_run_m': 1.5, 'gpt2_run_e': 21.0, 'tokens_ratio_m': 5.0, 'tokens_ratio_e': 11.0, 'tokens_ratio_b': 500.0, 'batch': 32.0, 'seq': 1024.0, 'tokens_step': 32768.0, 'step_tflop': 295.0, 'step_flop_m': 2.95, 'step_flop_e': 14.0, 'layer_step_tflop': 320.8, 'attn_extra_pct': 9.0, 'eta': 0.5, 'eta_pct': 50.0, 'step_s': 0.6, 'steps_m': 5.1, 'run_tokens_m': 1.7, 'run_tokens_e': 11.0, 'eta_half_pct': 25.0, 'step_s_half': 1.2, 'run_s_m': 3.0, 'run_s_e': 6.0, 'run_days': 35.0, 'run_kwh': 590.0, 'price_hr': 4.0, 'run_usd': 3370.0, 'point_hours': 16.5, 'point_pct': 2.0, 'node_days': 4.4, 'card_8b_mh': 1.46, 'card_70b_mh': 7.0, 'card_405b_mh': 30.84, 'card_total_mh': 39.3, 'card_w': 700.0, 'l405_params_b': 405.0, 'l405_tokens_t': 15.6, 'l405_6pd_m': 3.79, 'l405_6pd_e': 25.0, 'l405_paper_m': 3.8, 'l405_paper_e': 25.0, 'sec_per_hr': 3600.0, 'card_rate': 341.0, 'card_util_pct': 34.5, 'card_gwh': 21.6, 'card_over_run': 36600.0, 'l405_mfu_lo': 38.0, 'l405_mfu_hi': 43.0, 'l405_interruptions': 466.0, 'l405_snapshot_days': 54.0, 'card_tco2': 8930.0, 'fp32_b': 4.0, 'sgd_b': 8.0, 'momentum_b': 12.0, 'adam_fp32_b': 16.0, 'mp_w_b': 2.0, 'mp_master_b': 4.0, 'mp_moments_b': 8.0, 'mp_total_b': 16.0, 'mp_zero_k': 12.0, 'mp_w_gb': 3.0, 'mp_g_gb': 3.0, 'mp_master_gb': 6.0, 'mp_moments_gb': 12.0, 'mp_state_gb': 24.0, 'fp32_w_gb': 6.0, 'fp32_g_gb': 6.0, 'fp32_state_gb': 24.0, 'state_share_pct': 28.0, 'zero_act_gb': 60.0, 'b7_params_b': 7.0, 'b7_w_gb': 14.0, 'b7_g_gb': 14.0, 'b7_master_gb': 28.0, 'b7_moments_gb': 56.0, 'b7_state_gb': 112.0, 'upd_read_b': 16.0, 'upd_write_b': 12.0, 'upd_b': 28.0, 'upd_gb': 42.0, 'upd_ms': 12.5, 'upd_share_pct': 2.0, 'act_tok_gb': 86.0, 'act_scores_gb': 201.0, 'act_gb': 287.0, 'act_gb_1': 286.9, 'seq2': 2048.0, 'act2k_tok_gb': 171.0, 'act2k_scores_gb': 805.0, 'act_items_gb': 242.0, 'act_extra_gb': 45.0, 'layer_scores_mb': 4194.0, 'layer_items_mb': 5043.0, 'layer_scores_pct': 83.0, 'step_gb': 311.0, 'step_x': 3.6, 'fp32_step_gb': 598.0, 'l8_params_b': 8.0, 'l8_params_b0': 8.0, 'l8_act_gb': 36.5, 'l8_batch': 4.0, 'l8_seq': 2048.0, 'l8_lora_m': 6.8, 'l8_lora_pct': 0.08, 'l8_w_gb': 16.0, 'l8_g_gb': 16.0, 'l8_master_gb': 32.0, 'l8_moments_gb': 64.0, 'l8_state_gb': 128.0, 'l8_full_gb': 165.0, 'l8_adapter_gb': 0.11, 'l8_lora_gb': 53.0, 'l8_headroom_gb': 33.0, 'l8_rest_gb': 112.0, 'lora_gpt3_full_tb': 1.2, 'lora_gpt3_gb': 350.0, 'mfu_gpt3': 21.3, 'mfu_gopher': 32.5, 'gopher_chips': 4096.0, 'mtnlg_gpus': 2240.0, 'mfu_mtnlg': 30.2, 'mfu_palm': 46.2, 'hfu_palm': 57.8, 'megatron_lo': 44.0, 'megatron_hi': 52.0, 'res_gflop': 24.6, 'res_a100': 3200.0, 'res_h100': 10100.0, 'res_workers': 14.0, 'res_worker_rate': 375.0, 'res_supply': 5250.0, 'res_busy_pct': 52.0, 'gil_prep_ms': 250.0, 'gil_gpu_ms': 12.0, 'gil_busy_pct': 4.6, 'gil_step_ms': 262.0, 'gil_fast_step_ms': 256.0, 'ckpt_b': 14.0, 'ckpt_gb': 21.0, 'nvme_gbs': 7.0, 'ckpt_s': 3.0, 'ckpt_often': 100.0, 'ckpt_rarely': 10000.0, 'ckpt_work_s': 60.0, 'ckpt_stall_pct': 5.0, 'ckpt_rare_s': 5960.0, 'ckpt_replay_min': 99.0, 'palm_params_b': 540.0, 'palm_chips': 6144.0, 'palm_pod_chips': 3072.0, 'palm_spikes': 20.0, 'palm_rewind': 100.0, 'palm_skip_lo': 200.0, 'palm_skip_hi': 500.0, 'palm_state_tb': 8.6, 'palm_h100s': 100.0, 'palm_ckpt_tb': 7.6, 'opt_gpus': 992.0, 'opt_manual': 35.0, 'opt_auto': 70.0, 'opt_runs': 22.0}
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
plt = sp.plt
C = sp.C
out = Path(__file__).parent
CAP = 85.9
SUP = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')

def label(ax, x, y, text, color, shown=True, **kw):
    """A bold label that is drawn in every step (invisible until shown), so
    every step has the same bounding box and the slide does not jump."""
    kw.setdefault('fontsize', 12)
    kw.setdefault('fontweight', 'bold')
    return ax.text(x, y, text, color=color, alpha=1.0 if shown else 0.0, **kw)
rng = np.random.default_rng(3)
steps = np.arange(0, 400)
loss = 3.2 * np.exp(-steps / 260) + 1.6 + rng.normal(0, 0.025, len(steps))
spike = loss.copy()
spike[300:330] += np.concatenate([np.linspace(0, 2.2, 8), np.linspace(2.2, 0.9, 22)])
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(steps[:300], loss[:300], color=C['neutral'], lw=2.5)
    ax.plot(steps[299:330], spike[299:330], color=C['danger'], lw=2.5)
    label(ax, 318, spike[310] + 0.12, 'spike', C['danger'], ha='left')
    show = k == 2
    ax.annotate('', xy=(200, loss[200] - 0.55), xytext=(300, loss[200] - 0.55), arrowprops=dict(arrowstyle='-|>', color=C['memory'], lw=2.2), alpha=1.0 if show else 0.0)
    label(ax, 250, loss[200] - 0.85, 'rewind about 100 steps', C['memory'], show, ha='center', va='top', fontsize=11)
    ax.axvspan(300, 330, color=C['data_l'], alpha=0.9 if show else 0.0, lw=0)
    label(ax, 335, 4.2, 'skip 200 to\n500 batches', C['data'], show, ha='left', va='top', fontsize=11)
    ax.set_xlim(0, 420)
    ax.set_ylim(1.0, 4.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Training loss')
    ax.text(0.0, -0.2, 'Illustrative shape, not PaLM data', transform=ax.transAxes, fontsize=9, color=C['neutral'])
    sp.save(fig, out / f'palm_spike_step{k}.svg')
with sp.plt.rc_context({'xtick.labelsize': 14}):
    sp.hbars_steps(out / 'train_vs_token', [('One generated token', 3000000000.0, 'neutral', '3 × 10⁹ FLOPs'), ('One training run', 1.5e+21, 'compute', '1.5 × 10²¹  = 500 billion tokens')], 'FLOPs', xlabel='Arithmetic (FLOPs, log scale), GPT-2', size=sp.SHORT)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    rows = [('Inference', [(2, 'forward  2P', 'memory')]), ('Training', [(2, 'forward  2P', 'memory'), (4, 'backward  4P', 'compute')])]
    for i, (name, segs) in enumerate(rows):
        y = 1 - i
        shown = i < k
        x = 0
        for w, text, ck in segs:
            ax.barh(y, w, left=x, color=C[ck], height=0.6, edgecolor='white', alpha=1.0 if shown else 0.0)
            label(ax, x + w / 2, y, text, 'white', shown, ha='center', va='center')
            x += w
        ax.text(-0.1, y, name, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
        label(ax, x + 0.15, y, f'{x}P per token', C['text'], shown, va='center', fontsize=13)
    ax.set_xlim(0, 7.6)
    ax.set_ylim(-0.6, 1.6)
    ax.axis('off')
    sp.save(fig, out / f'flops_per_token_step{k}.svg')
fig, ax = plt.subplots(figsize=(5.8, 2.6))
PALM_TABLE3 = [('GPT-3 175B, V100', 'mfu_gpt3'), ('Gopher 280B, 4,096 TPU v3', 'mfu_gopher'), ('Megatron-Turing NLG 530B, 2,240 A100', 'mfu_mtnlg'), ('PaLM 540B, 6,144 TPU v4', 'mfu_palm')]
rows = [(name, _NUMS[key], 'compute', f'{_VALS[key]}%') for name, key in PALM_TABLE3]
rows.append(('GPT-2 XL, this lecture', 50.0, 'neutral', '50% (assumed)'))
n = len(rows)
for i, (name, v, ck, text) in enumerate(rows):
    y = n - 1 - i
    ax.barh(y, v, color=C[ck], height=0.62)
    ax.text(-1, y, name, ha='right', va='center', fontsize=11, fontweight='bold', color=C['text'])
    ax.text(v + 1.5, y, text, va='center', fontsize=12, fontweight='bold', color=C[ck])
ax.axvline(100, color=C['neutral'], lw=1.5, ls='--')
ax.text(99, n - 0.45, 'peak', ha='right', va='bottom', fontsize=11, color=C['neutral'])
ax.set_xlim(0, 100)
ax.set_ylim(-0.6, n - 0.1)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Model FLOPs utilization (percent of peak)')
ax.grid(axis='x', color=C['grid'], lw=0.6)
ax.set_axisbelow(True)
sp.save(fig, out / 'mfu_published.svg')
segs = [(1.9, 'weights', 'memory'), (1.9, 'gradients', 'network'), (3.8, 'Adam m, v', 'compute'), (3.4, 'activations', 'data')]
for k in range(1, 5):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    x = 0
    for j, (w, text, ck) in enumerate(segs):
        shown = j < k
        ax.barh(0, w, left=x, color=C[ck], height=0.62, edgecolor='white', alpha=1.0 if shown else 0.0)
        label(ax, x + w / 2, 0, text, 'white', shown, ha='center', va='center')
        x += w
    label(ax, 0.95, -0.62, 'inference stops here', C['memory'], k == 1, ha='center', va='top', fontsize=11)
    label(ax, 3.8, 0.55, '1 : 1 : 2', C['text'], 3 <= k, ha='center', va='bottom', fontsize=12)
    label(ax, 9.3, -0.62, 'grows with batch and sequence length', C['data'], k == 4, ha='center', va='top', fontsize=11)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-1.15, 0.95)
    ax.axis('off')
    sp.save(fig, out / f'train_state_step{k}.svg')
fig, ax = plt.subplots(figsize=(5.8, 2.0))
W_B, MASTER_B, MOM_B, TOT_B = (round(_NUMS[k]) for k in ('mp_w_b', 'mp_master_b', 'mp_moments_b', 'mp_total_b'))
rows = [('Inference', [(W_B, 'W', 'memory')]), ('Training', [(W_B, 'W', 'memory'), (W_B, 'G', 'network'), (MASTER_B, 'master', 'neutral'), (MOM_B, 'Adam m, v (FP32)', 'compute')])]
for i, (name, parts) in enumerate(rows):
    y = 1 - i
    x = 0
    for w, text, ck in parts:
        ax.barh(y, w, left=x, color=C[ck], height=0.6, edgecolor='white')
        ax.text(x + w / 2, y, text, ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        x += w
    ax.text(-0.2, y, name, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
    ax.text(x + 0.3, y, f'{x} B', va='center', fontsize=14, fontweight='bold', color=C['danger'] if x == TOT_B else C['text'])
ax.set_xlim(0, 18.8)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([])
ax.set_xticks([0, W_B, W_B + W_B, MASTER_B + MASTER_B, TOT_B])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Bytes per parameter before activations (W, G: FP16 weights, gradients; master: FP32)', fontsize=11)
sp.save(fig, out / 'static_bytes.svg')
STATIC = 24.0
SCORES = 201.326592
PERTOK = 85.5638016
STEP = 310.8903936
parts = [(STATIC, '', 'neutral'), (PERTOK, f'activations {PERTOK:.0f}', 'data'), (SCORES, f'attention scores {SCORES:.0f}', 'danger')]
for k in range(1, 4):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    x = 0
    for j, (w, text, ck) in enumerate(parts):
        shown = j < k
        ax.barh(0, w, left=x, color=C[ck], height=0.6, edgecolor='white', alpha=1.0 if shown else 0.0)
        label(ax, x + w / 2, 0, text, 'white', shown, ha='center', va='center', fontsize=11)
        x += w
    label(ax, 0, -0.38, f'static {STATIC:.0f}', C['neutral'], True, va='top', fontsize=11)
    label(ax, STEP + 1, 0, f' {STEP:.0f} GB', C['text'], k == 3, va='center', fontsize=13)
    ax.axvline(CAP, color=C['memory'], lw=2, ls='--', zorder=0)
    ax.text(CAP + 3, 0.42, 'H100: 85.9 GB (80 GiB)', color=C['memory'], fontsize=11, fontweight='bold', va='bottom')
    ax.set_xlim(0, 360)
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('GB for one batch-32 step of GPT-2 XL (FP16 activations)')
    sp.save(fig, out / f'gpt2_b32_step{k}.svg')
fig, ax = plt.subplots(figsize=(5.8, 2.3))
L_W = 16.06
L_REST = 112.0
L_LORA = 0.109051904
L_ACT = 36.5
rows = [('Full fine-tune', [(L_W, 'W', 'memory'), (L_REST, 'grads, master, Adam  112', 'compute'), (L_ACT, 'act.', 'data')], 'about 165 GB'), ('LoRA adapters', [(L_W, 'W', 'memory'), (L_LORA, '', 'compute'), (L_ACT, 'act. 36.5', 'data')], 'about 53 GB')]
for i, (name, parts, total) in enumerate(rows):
    y = 1 - i
    x = 0
    for w, text, ck in parts:
        ax.barh(y, w, left=x, color=C[ck], height=0.6, edgecolor='white')
        if text:
            ax.text(x + w / 2, y, text, ha='center', va='center', color='white', fontsize=11, fontweight='bold')
        x += w
    ax.text(-2, y, name, ha='right', va='center', fontsize=13, fontweight='bold', color=C['text'])
    ax.text(max(x, CAP) + 3, y, total, va='center', fontsize=13, fontweight='bold', color=C['danger'] if x > CAP else C['data'])
ax.axvline(CAP, color=C['memory'], lw=2, ls='--', zorder=0)
ax.text(CAP + 2, 1.42, 'H100: 85.9 GB (80 GiB)', color=C['memory'], fontsize=11, fontweight='bold', va='bottom')
ax.set_xlim(0, 215)
ax.set_ylim(-0.55, 1.75)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('GB on one accelerator, Llama 3 8B (W: 16 GB of 16-bit weights)')
sp.save(fig, out / 'llama_finetune.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    gpus = ['A100', 'H100']
    demand = [3200.0, 10100.0]
    supply = 5250.0
    for i, g in enumerate(gpus):
        shown = i < k
        a = 1.0 if shown else 0.0
        ax.bar(i - 0.2, demand[i], 0.38, color=C['compute'], alpha=a)
        ax.bar(i + 0.2, supply, 0.38, color=C['data'], alpha=a)
        label(ax, i - 0.2, demand[i] + 250, {'A100': '3,200', 'H100': '10,100'}[g], C['compute'], shown, ha='center', fontsize=12)
        label(ax, i + 0.2, supply + 250, '5,250', C['data'], shown, ha='center', fontsize=12)
    label(ax, 0.0, 6300, 'fed', C['data'], k >= 1, ha='center', fontsize=14)
    label(ax, 1.0, 11300, 'busy 52%', C['danger'], k == 2, ha='center', fontsize=14)
    ax.text(-0.5, 14300, 'accelerator demand', color=C['compute'], fontsize=12, fontweight='bold')
    ax.text(-0.5, 13200, 'host supply (14 workers)', color=C['data'], fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1], gpus)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 15200)
    ax.set_yticks([0, 4000, 8000, 12000], ['0', '4k', '8k', '12k'])
    ax.set_ylabel('Images per second')
    sp.save(fig, out / f'resnet_feed_step{k}.svg')
F = round(4.0)
opt_rows = [('SGD', [(F, 'W', 'memory'), (F, 'G', 'network')]), ('Momentum', [(F, 'W', 'memory'), (F, 'G', 'network'), (F, 'v', 'compute')]), ('Adam', [(F, 'W', 'memory'), (F, 'G', 'network'), (F, 'm', 'compute'), (F, 'v', 'compute')])]
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.1))
    for i, (name, parts) in enumerate(opt_rows):
        y = 2 - i
        shown = i < k
        x = 0
        for w, text, ck in parts:
            ax.barh(y, w, left=x, color=C[ck], height=0.6, edgecolor='white', alpha=1.0 if shown else 0.0)
            label(ax, x + w / 2, y, text, 'white', shown, ha='center', va='center')
            x += w
        label(ax, -0.3, y, name, C['text'], shown, ha='right', va='center', fontsize=13)
        label(ax, x + 0.4, y, f'{x} B', C['danger'] if x == round(16.0) else C['text'], shown, va='center', fontsize=13)
    ax.set_xlim(0, 19.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xticks([0, F, F + F, F + F + F, F + F + F + F])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Bytes per parameter in FP32 (W weights, G gradients, m and v optimizer state)', fontsize=10.5)
    sp.save(fig, out / f'optimizer_bytes_step{k}.svg')
seq = np.linspace(128, 2048, 200)
lin = np.array([10.6954752, 11.5310592, 12.2830848, 13.1186688, 13.9542528, 14.7062784, 15.5418624, 16.3774464, 17.129472, 17.965056, 18.7170816, 19.5526656, 20.3882496, 21.1402752, 21.9758592, 22.8114432, 23.5634688, 24.3990528, 25.2346368, 25.9866624, 26.8222464, 27.6578304, 28.409856, 29.24544, 30.081024, 30.8330496, 31.6686336, 32.5042176, 33.2562432, 34.0918272, 34.8438528, 35.6794368, 36.5150208, 37.2670464, 38.1026304, 38.9382144, 39.69024, 40.525824, 41.361408, 42.1134336, 42.9490176, 43.7846016, 44.5366272, 45.3722112, 46.2077952, 46.9598208, 47.7954048, 48.5474304, 49.3830144, 50.2185984, 50.970624, 51.806208, 52.641792, 53.3938176, 54.2294016, 55.0649856, 55.8170112, 56.6525952, 57.4881792, 58.2402048, 59.0757888, 59.9113728, 60.6633984, 61.4989824, 62.251008, 63.086592, 63.922176, 64.6742016, 65.5097856, 66.3453696, 67.0973952, 67.9329792, 68.7685632, 69.5205888, 70.3561728, 71.1917568, 71.9437824, 72.7793664, 73.6149504, 74.366976, 75.20256, 76.038144, 76.7901696, 77.6257536, 78.3777792, 79.2133632, 80.0489472, 80.8009728, 81.6365568, 82.4721408, 83.2241664, 84.0597504, 84.8953344, 85.64736, 86.482944, 87.318528, 88.0705536, 88.9061376, 89.7417216, 90.4937472, 91.3293312, 92.0813568, 92.9169408, 93.7525248, 94.5045504, 95.3401344, 96.1757184, 96.927744, 97.763328, 98.598912, 99.3509376, 100.1865216, 101.0221056, 101.7741312, 102.6097152, 103.4452992, 104.1973248, 105.0329088, 105.7849344, 106.6205184, 107.4561024, 108.208128, 109.043712, 109.879296, 110.6313216, 111.4669056, 112.3024896, 113.0545152, 113.8900992, 114.7256832, 115.4777088, 116.3132928, 117.1488768, 117.9009024, 118.7364864, 119.5720704, 120.324096, 121.15968, 121.9117056, 122.7472896, 123.5828736, 124.3348992, 125.1704832, 126.0060672, 126.7580928, 127.5936768, 128.4292608, 129.1812864, 130.0168704, 130.8524544, 131.60448, 132.440064, 133.275648, 134.0276736, 134.8632576, 135.6152832, 136.4508672, 137.2864512, 138.0384768, 138.8740608, 139.7096448, 140.4616704, 141.2972544, 142.1328384, 142.884864, 143.720448, 144.556032, 145.3080576, 146.1436416, 146.9792256, 147.7312512, 148.5668352, 149.3188608, 150.1544448, 150.9900288, 151.7420544, 152.5776384, 153.4132224, 154.165248, 155.000832, 155.836416, 156.5884416, 157.4240256, 158.2596096, 159.0116352, 159.8472192, 160.6828032, 161.4348288, 162.2704128, 163.1059968, 163.8580224, 164.6936064, 165.445632, 166.281216, 167.1168, 167.8688256, 168.7044096, 169.5399936, 170.2920192, 171.1276032])
score = np.array([3.145728, 3.656448, 4.148928, 4.732608, 5.354688, 5.947392, 6.642432, 7.375872, 8.0688, 8.8752, 9.633792, 10.513152, 11.430912, 12.289728, 13.280448, 14.309568, 15.268608, 16.370688, 17.511168, 18.570432, 19.783872, 21.035712, 22.1952, 23.52, 24.8832, 26.142912, 27.579072, 29.053632, 30.413568, 31.961088, 33.386688, 35.007168, 36.666048, 38.191872, 39.923712, 41.693952, 43.32, 45.1632, 47.0448, 48.771072, 50.725632, 52.718592, 54.545088, 56.611008, 58.715328, 60.642048, 62.819328, 64.811712, 67.061952, 69.350592, 71.4432, 73.8048, 76.2048, 78.397632, 80.870592, 83.381952, 85.675008, 88.259328, 90.882048, 93.275328, 95.971008, 98.705088, 101.198592, 104.005632, 106.5648, 109.4448, 112.3632, 115.022592, 118.013952, 121.043712, 123.803328, 126.906048, 130.047168, 132.907008, 136.121088, 139.373568, 142.333632, 145.659072, 149.022912, 152.0832, 155.52, 158.9952, 162.155712, 165.703872, 168.930048, 172.551168, 176.210688, 179.537088, 183.269568, 187.040448, 190.467072, 194.310912, 198.193152, 201.72, 205.6752, 209.6688, 213.295872, 217.362432, 221.467392, 225.194688, 229.372608, 233.165568, 237.416448, 241.705728, 245.598912, 249.961152, 254.361792, 258.3552, 262.8288, 267.3408, 271.434432, 276.019392, 280.642752, 284.836608, 289.532928, 294.267648, 298.561728, 303.369408, 307.729152, 312.609792, 317.528832, 321.9888, 326.9808, 332.0112, 336.571392, 341.674752, 346.816512, 351.476928, 356.691648, 361.944768, 366.705408, 372.031488, 377.395968, 382.256832, 387.694272, 393.170112, 398.1312, 403.68, 408.706752, 414.328512, 419.988672, 425.115648, 430.848768, 436.620288, 441.847488, 447.691968, 453.574848, 458.902272, 464.858112, 470.852352, 476.28, 482.3472, 488.4528, 493.980672, 500.159232, 505.752768, 512.004288, 518.294208, 523.987968, 530.350848, 536.752128, 542.546112, 549.020352, 555.532992, 561.4272, 568.0128, 574.6368, 580.631232, 587.328192, 594.063552, 600.158208, 606.966528, 613.126848, 620.008128, 626.927808, 633.188352, 640.180992, 647.212032, 653.5728, 660.6768, 667.8192, 674.280192, 681.495552, 688.749312, 695.310528, 702.637248, 710.002368, 716.663808, 724.101888, 731.578368, 738.340032, 745.889472, 752.7168, 760.3392, 768.0, 774.927552, 782.661312, 790.433472, 797.461248, 805.306368])
AT1024 = 286.8903936
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    ax.fill_between(seq, 0, lin, color=C['data'], lw=0)
    ax.fill_between(seq, lin, lin + score, color=C['danger'], lw=0, alpha=1.0 if k == 2 else 0.0)
    ax.axhline(CAP, color=C['memory'], lw=2, ls='--')
    ax.text(140, CAP + 14, 'H100: 85.9 GB', color=C['memory'], fontsize=11, fontweight='bold')
    ax.text(150, 960, 'green: per-token tensors (projections, FFN,\nlayer inputs), grow with S', color=C['data'], fontsize=11, fontweight='bold', ha='left', va='top')
    label(ax, 1350, 620, 'attention scores:\ngrow with S²', C['danger'], k == 2, ha='right', fontsize=11.5)
    ax.plot([1024], [AT1024], 'o', ms=9, color=C['danger'], mec='white', mew=1.5, alpha=1.0 if k == 2 else 0.0)
    label(ax, 990, 420, '1,024 tokens: 287 GB', C['danger'], k == 2, ha='right', fontsize=11.5)
    ax.set_xlim(128, 2048)
    ax.set_ylim(0, 1050)
    ax.set_xticks([256, 512, 1024, 1536, 2048], ['256', '512', '1,024', '1,536', '2,048'])
    ax.set_xlabel('Sequence length S (batch 32; derived from Napkin Math 8.3)', fontsize=11)
    ax.set_ylabel('Activations (GB)')
    ax.grid(color=C['grid'], lw=0.5)
    ax.set_axisbelow(True)
    sp.save(fig, out / f'act_vs_seq_step{k}.svg')
fig, ax = plt.subplots(figsize=(5.8, 2.3))
rows = [('Memory-stalled job', 92, 20), ('Data-starved job', 20, 12)]
for i, (name, busy, mfu) in enumerate(rows):
    y = 1 - i
    ax.barh(y + 0.17, busy, height=0.32, color=C['neutral'])
    ax.barh(y - 0.17, mfu, height=0.32, color=C['compute'])
    ax.text(busy + 1.5, y + 0.17, f'busy {busy}%', va='center', fontsize=12, fontweight='bold', color=C['neutral'])
    ax.text(mfu + 1.5, y - 0.17, 'MFU (illustrative)', va='center', fontsize=12, fontweight='bold', color=C['compute'])
    ax.text(-1.5, y, name, ha='right', va='center', fontsize=12, fontweight='bold', color=C['text'])
ax.set_xlim(0, 120)
ax.set_ylim(-0.55, 1.55)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Percent of the step (busy) or of peak (MFU)', fontsize=11)
sp.save(fig, out / 'busy_mfu.svg')

def trace_crop():
    from PIL import Image
    src = Image.open(out / 'fig-tf-bottleneck-trace.png').convert('RGB')
    src.crop((285, 208, 668, 398)).save(out / 'trace_crop.png')
trace_crop()
print('figures written')
