#!/usr/bin/env python3
"""Figures for Lecture 13 (Model Serving, Part 1).

Every value comes from calcs.py (imported as c), which reads MLSysIM: the
ResNet-50 H100 serving profile and the model-serving Scenarios.* the Model
Serving chapter also reads, the cited results in cited literature,
and the MLPerf Inference v5.0 rows in local CSV series. Figure code takes
plotted values from those objects and printed labels from deckvals.val(), so
a label shows exactly the digits the slide text shows. The ResNet-50 numbers
are the chapter's modeled reference profile on the H100 (not measurements);
its host phases are stated assumptions. The AI Tax pie (Richins et al.,
arXiv 2007.10571, Fig. 6) is the paper's own figure, aitax_fig6.png.
Arrival marks in the batching-window figure are illustrative; thumbnails at
the end are text-free pictograms for the goals and takeaways slots.

Run from this folder: python3 make_figs.py
"""
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
_VALS = {'price_h100': '4', 'b1_ms': '3.5', 'b1_ms_2': '3.53', 'b1_ms_3': '3.528', 'b8_ms': '3.8', 'b1_rate': '283', 'b1_rate_1': '283.4', 'b8_rate': '2,129', 'b1_usd': '3.92', 'b8_usd': '0.52', 'hook_ratio': '7.5', 'hook_added_ms': '0.23', 'hook_added_ms_1': '0.2', 'hook_label_gap': '0.3', 'hook_added_pct': '6', 'eager_kernels': '174', 'batch_8': '8', 'rtx_ss_ms': '0.282', 'rtx_ss_ms_2': '0.28', 'rtx_ss_rate': '3,550', 'rtx_off': '44,411', 'rtx_off_over_ss': '12.5', 'b32_rate': '7,040', 'batch_32': '32', 'plan_rho_pct': '67.5', 'plan_rho_pct_0': '68', 'plan_rho': '0.6754', 'p99_budget_ms': '50', 'photos': '10,000', 'moderation_ms': '100', 'int8_offline': '89,500', 'static_b1_s': '35', 'static_int8_s': '0.11', 'static_ratio': '316', 'static_ratio_book': '320', 'static_ratio_floor': '300', 'img_side': '224', 'st_decode': '3', 'st_resize': '1', 'st_norm': '0.5', 'st_copy': '0.5', 'st_post': '0.1', 'budget_total': '8.6', 'gpu_idle': '5', 'model_share': '41', 'pre_share': '52', 'dl_parse': '0.5', 'dl_emb': '6', 'dl_mlp': '1.5', 'dl_rank': '1', 'dl_total': '9', 'dl_mlp_share': '17', 'dlrm_gb': '100', 'host_stage': '4.5', 'pipe_duty': '78', 'model_speedup': '10', 'fast_model_ms': '0.35', 'fast_total': '5.5', 'fast_speedup': '1.6', 'naive_ms': '0.86', 'rich_ingest': '18.8', 'rich_detect': '74.8', 'rich_brokers': '126.1', 'rich_ident': '131.5', 'rich_total': '351', 'rich_ai': '59', 'little_rate': '1,000', 'little_w_s': '0.05', 'little_w_ms': '50', 'little_n': '50', 'gpu_batch': '32', 'little_wait': '18', 'little_w2_ms': '10', 'little_rate2': '5,000', 'util_50': '50', 'knee_50': '7.1', 'util_70': '70', 'knee_70': '11.8', 'util_80': '80', 'knee_80': '17.6', 'util_90': '90', 'knee_90': '35.3', 'util_95': '95', 'knee_95': '70.6', 'knee_p99_ex': '54.1', 'knee_mean_ex_0': '12', 'p99_factor': '4.6', 'ln100': '4.6', 'peak_rate': '5,000', 'required_rate': '7,403', 'gpus_exact': '26.1', 'gpus': '27', 'headroom_pct': '30', 'plan_gpus': '36', 'plan_gpus_less1': '35', 'fail_util': '50.4', 'naive_gpus_exact': '17.6', 'naive_gpus': '18', 'rho_at_printed': '0.678', 'mlperf_bound_ms': '15', 'a_off_k': '716,000', 'b_off_k': '713,000', 'a_kept': '91', 'b_kept': '82', 'a_lost': '9', 'b_lost': '18', 'kept_gap': '9', 'vote_util': '80', 'vote_service_gain': '2', 'fast_service_ms': '1.8', 'vote_old': '17.6', 'vote_wait_old': '14.1', 'vote_util_new': '41', 'vote_new': '3.0', 'vote_wait_new': '1.2', 'vote_gain': '6', 'fan_n10': '10', 'fan_n100': '100', 'fan_avail': '0.99', 'fan_avail_pct': '99', 'fan10': '90.4', 'fan100': '36.6', 'db_leaf': '10', 'db_all': '140', 'db_ratio': '14', 'pa_client': '708', 'pa_server': '382', 'pa_infer': '257', 'pa_infer_share': '36', 'pa_outside': '326', 'pa_count': '25,348', 'pa_exec': '25,348', 'pa_tput': '1,407.84', 'pa_little': '1.00', 'fixed_ms': '3.50', 'fixed_ms_3': '3.495', 'per_img_ms': '0.033', 'per_img_ms_4': '0.0329', 'host_per_kernel_us': '15', 'issue_ms': '2.6', 'weights_mb': '102', 'per_image_1': '3.53', 'fixed_share_1': '99', 'energy_1': '39.2', 'move_share_1': '52', 'per_image_8': '0.47', 'fixed_share_8': '93', 'energy_8': '24.9', 'move_share_8': '24', 'per_image_32': '0.14', 'fixed_share_32': '77', 'energy_32': '23.4', 'move_share_32': '19', 'b32_ms': '4.55', 'tput_gain': '25', 'rt_ts': '1.79', 'rt_ts_x': '2.0', 'rt_onnx': '0.61', 'rt_onnx_x': '5.8', 'rt_fp32': '0.38', 'rt_fp32_x': '9.3', 'rt_fp16': '0.33', 'rt_fp16_x': '10.6', 'rt_int8': '0.30', 'rt_int8_x': '11.6', 'fused_kernels': '56', 'rt_prec_gain': '1.3', 'curve_ms_1': '3.53', 'curve_rate_1': '283', 'curve_ms_32': '4.55', 'curve_rate_32': '7,040', 'curve_ms_128': '7.7', 'curve_rate_128': '16,600', 'batch_128': '128', 'l4_ss_ms': '0.414', 'l4_ms_ms': '0.770', 'l4_per_query': '8', 'l4_count': '3', 'l4_count_word': 'three', 'l4_one': '2,415', 'l4_multi': '10,390', 'l4_each': '12,855', 'l4_lat_ratio': '1.9', 'l4_tput_ratio': '4.3', 'win_ms': '10', 'win20_ms': '20', 'win20_wait': '10', 'win20_budget': '50', 'win20_share': '20', 'qps_50': '50', 'win_mean_50': '0.5', 'qps_200': '200', 'win_mean_200': '2', 'qps_500': '500', 'win_mean_500': '5', 'qps_1000': '1,000', 'win_mean_1000': '10', 'empty_50': '61', 'seen_mean': '6', 'seen_p99': '12', 'triton_delay_us': '100', 'triton_pref_small': '4', 'triton_pref_large': '8', 'triton_max_batch': '8', 'arrive_0': '0', 'arrive_1': '2', 'arrive_2': '5', 'arrive_3': '9', 'trace_n_word': 'four', 'timeout_ms': '12', 'worker_free_ms': '18', 'batch4_ms': '3.6', 'response_ms': '4', 'deadline_ms': '50', 'backlog_ms': '6', 'batch_end_ms': '21.6', 'trace_done_ms': '25.6', 'trace_slack_ms': '24.4', 'trace_wrong_ms': '15.6', 'cap_offer': '500', 'cap_small': '16', 'cap_large': '32', 'cap_small_rate': '3,980', 'cap_large_rate': '7,040', 'cap_small_ms': '4.0', 'cap_large_ms': '4.5', 'cap_gain': '77', 'fill_small': '15', 'fill_large': '31', 'cap_over_load': '8', 'cs_ssd': '0.5', 'cs_s3_lo': '3', 'cs_s3_hi': '5', 'cs_cuda_lo': '0.3', 'cs_cuda_hi': '0.5', 'cs_trt_lo': '15', 'cs_trt_hi': '30', 'cs_warm_rt': '0.6', 'cs_warm': '1.5', 'cs_cold': '35', 'ckpt_gb': '5', 'pickle_s': '15', 'safetensors_s': '1.5', 'opt_s': '34', 'opt_gpus': '4', 'llama_s': '84', 'llama_gpus': '8', 'sllm_gb': '130', 'sllm_gbps': '5', 'sllm_dl': '26', 'log_ready1': '12.4', 'log_ready2': '2.7', 'log_comp1': '10.7', 'log_comp_share': '86', 'log_gain': '4.6', 'log_pickle_ms': '18', 'log_st_ms': '2', 'swap_gb': '10', 'swap_bw': '32', 'swap_ms': '312.5', 'swap_over_word': 'six', 'swap_passes': '90', 'surge_normal': '1,000', 'surge_peak': '10,000', 'surge_word': 'ten', 'surge_latency_s': '10', 'cpu_price': '0.17', 't4_price': '0.53', 'cpu_rate': '50', 't4_rate': '400', 'cpm_cpu': '0.94', 'cpm_t4': '0.37', 'cpm_h100': '0.16', 'cpu_per_hour': '180,000', 't4_n_word': 'seven', 't4_n_usd': '3.71', 't4_n_rate': '2,800', 'crossover': '3,000', 'syn_batched': '3', 'syn_cost_b1': '8.00', 'syn_cost_bt': '0.67', 'syn_ratio': '12', 'llama_off': '31,217', 'llama_conv': '30,904', 'llama_conv_k': '31,000', 'llama_conv_tpot': '200', 'llama_conv_ttft_s': '2', 'llama_int_tpot': '40', 'llama_int_ttft': '450', 'llama_int': '12,979'}
_NUMS = {'price_h100': 4.0, 'b1_ms': 3.5, 'b1_ms_2': 3.53, 'b1_ms_3': 3.528, 'b8_ms': 3.8, 'b1_rate': 283.0, 'b1_rate_1': 283.4, 'b8_rate': 2129.0, 'b1_usd': 3.92, 'b8_usd': 0.52, 'hook_ratio': 7.5, 'hook_added_ms': 0.23, 'hook_added_ms_1': 0.2, 'hook_label_gap': 0.3, 'hook_added_pct': 6.0, 'eager_kernels': 174.0, 'batch_8': 8.0, 'rtx_ss_ms': 0.282, 'rtx_ss_ms_2': 0.28, 'rtx_ss_rate': 3550.0, 'rtx_off': 44411.0, 'rtx_off_over_ss': 12.5, 'b32_rate': 7040.0, 'batch_32': 32.0, 'plan_rho_pct': 67.5, 'plan_rho_pct_0': 68.0, 'plan_rho': 0.6754, 'p99_budget_ms': 50.0, 'photos': 10000.0, 'moderation_ms': 100.0, 'int8_offline': 89500.0, 'static_b1_s': 35.0, 'static_int8_s': 0.11, 'static_ratio': 316.0, 'static_ratio_book': 320.0, 'static_ratio_floor': 300.0, 'img_side': 224.0, 'st_decode': 3.0, 'st_resize': 1.0, 'st_norm': 0.5, 'st_copy': 0.5, 'st_post': 0.1, 'budget_total': 8.6, 'gpu_idle': 5.0, 'model_share': 41.0, 'pre_share': 52.0, 'dl_parse': 0.5, 'dl_emb': 6.0, 'dl_mlp': 1.5, 'dl_rank': 1.0, 'dl_total': 9.0, 'dl_mlp_share': 17.0, 'dlrm_gb': 100.0, 'host_stage': 4.5, 'pipe_duty': 78.0, 'model_speedup': 10.0, 'fast_model_ms': 0.35, 'fast_total': 5.5, 'fast_speedup': 1.6, 'naive_ms': 0.86, 'rich_ingest': 18.8, 'rich_detect': 74.8, 'rich_brokers': 126.1, 'rich_ident': 131.5, 'rich_total': 351.0, 'rich_ai': 59.0, 'little_rate': 1000.0, 'little_w_s': 0.05, 'little_w_ms': 50.0, 'little_n': 50.0, 'gpu_batch': 32.0, 'little_wait': 18.0, 'little_w2_ms': 10.0, 'little_rate2': 5000.0, 'util_50': 50.0, 'knee_50': 7.1, 'util_70': 70.0, 'knee_70': 11.8, 'util_80': 80.0, 'knee_80': 17.6, 'util_90': 90.0, 'knee_90': 35.3, 'util_95': 95.0, 'knee_95': 70.6, 'knee_p99_ex': 54.1, 'knee_mean_ex_0': 12.0, 'p99_factor': 4.6, 'ln100': 4.6, 'peak_rate': 5000.0, 'required_rate': 7403.0, 'gpus_exact': 26.1, 'gpus': 27.0, 'headroom_pct': 30.0, 'plan_gpus': 36.0, 'plan_gpus_less1': 35.0, 'fail_util': 50.4, 'naive_gpus_exact': 17.6, 'naive_gpus': 18.0, 'rho_at_printed': 0.678, 'mlperf_bound_ms': 15.0, 'a_off_k': 716000.0, 'b_off_k': 713000.0, 'a_kept': 91.0, 'b_kept': 82.0, 'a_lost': 9.0, 'b_lost': 18.0, 'kept_gap': 9.0, 'vote_util': 80.0, 'vote_service_gain': 2.0, 'fast_service_ms': 1.8, 'vote_old': 17.6, 'vote_wait_old': 14.1, 'vote_util_new': 41.0, 'vote_new': 3.0, 'vote_wait_new': 1.2, 'vote_gain': 6.0, 'fan_n10': 10.0, 'fan_n100': 100.0, 'fan_avail': 0.99, 'fan_avail_pct': 99.0, 'fan10': 90.4, 'fan100': 36.6, 'db_leaf': 10.0, 'db_all': 140.0, 'db_ratio': 14.0, 'pa_client': 708.0, 'pa_server': 382.0, 'pa_infer': 257.0, 'pa_infer_share': 36.0, 'pa_outside': 326.0, 'pa_count': 25348.0, 'pa_exec': 25348.0, 'pa_tput': 1407.84, 'pa_little': 1.0, 'fixed_ms': 3.5, 'fixed_ms_3': 3.495, 'per_img_ms': 0.033, 'per_img_ms_4': 0.0329, 'host_per_kernel_us': 15.0, 'issue_ms': 2.6, 'weights_mb': 102.0, 'per_image_1': 3.53, 'fixed_share_1': 99.0, 'energy_1': 39.2, 'move_share_1': 52.0, 'per_image_8': 0.47, 'fixed_share_8': 93.0, 'energy_8': 24.9, 'move_share_8': 24.0, 'per_image_32': 0.14, 'fixed_share_32': 77.0, 'energy_32': 23.4, 'move_share_32': 19.0, 'b32_ms': 4.55, 'tput_gain': 25.0, 'rt_ts': 1.79, 'rt_ts_x': 2.0, 'rt_onnx': 0.61, 'rt_onnx_x': 5.8, 'rt_fp32': 0.38, 'rt_fp32_x': 9.3, 'rt_fp16': 0.33, 'rt_fp16_x': 10.6, 'rt_int8': 0.3, 'rt_int8_x': 11.6, 'fused_kernels': 56.0, 'rt_prec_gain': 1.3, 'curve_ms_1': 3.53, 'curve_rate_1': 283.0, 'curve_ms_32': 4.55, 'curve_rate_32': 7040.0, 'curve_ms_128': 7.7, 'curve_rate_128': 16600.0, 'batch_128': 128.0, 'l4_ss_ms': 0.414, 'l4_ms_ms': 0.77, 'l4_per_query': 8.0, 'l4_count': 3.0, 'l4_one': 2415.0, 'l4_multi': 10390.0, 'l4_each': 12855.0, 'l4_lat_ratio': 1.9, 'l4_tput_ratio': 4.3, 'win_ms': 10.0, 'win20_ms': 20.0, 'win20_wait': 10.0, 'win20_budget': 50.0, 'win20_share': 20.0, 'qps_50': 50.0, 'win_mean_50': 0.5, 'qps_200': 200.0, 'win_mean_200': 2.0, 'qps_500': 500.0, 'win_mean_500': 5.0, 'qps_1000': 1000.0, 'win_mean_1000': 10.0, 'empty_50': 61.0, 'seen_mean': 6.0, 'seen_p99': 12.0, 'triton_delay_us': 100.0, 'triton_pref_small': 4.0, 'triton_pref_large': 8.0, 'triton_max_batch': 8.0, 'arrive_0': 0.0, 'arrive_1': 2.0, 'arrive_2': 5.0, 'arrive_3': 9.0, 'timeout_ms': 12.0, 'worker_free_ms': 18.0, 'batch4_ms': 3.6, 'response_ms': 4.0, 'deadline_ms': 50.0, 'backlog_ms': 6.0, 'batch_end_ms': 21.6, 'trace_done_ms': 25.6, 'trace_slack_ms': 24.4, 'trace_wrong_ms': 15.6, 'cap_offer': 500.0, 'cap_small': 16.0, 'cap_large': 32.0, 'cap_small_rate': 3980.0, 'cap_large_rate': 7040.0, 'cap_small_ms': 4.0, 'cap_large_ms': 4.5, 'cap_gain': 77.0, 'fill_small': 15.0, 'fill_large': 31.0, 'cap_over_load': 8.0, 'cs_ssd': 0.5, 'cs_s3_lo': 3.0, 'cs_s3_hi': 5.0, 'cs_cuda_lo': 0.3, 'cs_cuda_hi': 0.5, 'cs_trt_lo': 15.0, 'cs_trt_hi': 30.0, 'cs_warm_rt': 0.6, 'cs_warm': 1.5, 'cs_cold': 35.0, 'ckpt_gb': 5.0, 'pickle_s': 15.0, 'safetensors_s': 1.5, 'opt_s': 34.0, 'opt_gpus': 4.0, 'llama_s': 84.0, 'llama_gpus': 8.0, 'sllm_gb': 130.0, 'sllm_gbps': 5.0, 'sllm_dl': 26.0, 'log_ready1': 12.4, 'log_ready2': 2.7, 'log_comp1': 10.7, 'log_comp_share': 86.0, 'log_gain': 4.6, 'log_pickle_ms': 18.0, 'log_st_ms': 2.0, 'swap_gb': 10.0, 'swap_bw': 32.0, 'swap_ms': 312.5, 'swap_passes': 90.0, 'surge_normal': 1000.0, 'surge_peak': 10000.0, 'surge_latency_s': 10.0, 'cpu_price': 0.17, 't4_price': 0.53, 'cpu_rate': 50.0, 't4_rate': 400.0, 'cpm_cpu': 0.94, 'cpm_t4': 0.37, 'cpm_h100': 0.16, 'cpu_per_hour': 180000.0, 't4_n_usd': 3.71, 't4_n_rate': 2800.0, 'crossover': 3000.0, 'syn_batched': 3.0, 'syn_cost_b1': 8.0, 'syn_cost_bt': 0.67, 'syn_ratio': 12.0, 'llama_off': 31217.0, 'llama_conv': 30904.0, 'llama_conv_k': 31000.0, 'llama_conv_tpot': 200.0, 'llama_conv_ttft_s': 2.0, 'llama_int_tpot': 40.0, 'llama_int_ttft': 450.0, 'llama_int': 12979.0}
import math
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
import slideplot as sp
C = sp.C
plt = sp.plt
out = Path(__file__).parent
m = lambda q, u: q.to(u).magnitude

def tag(ax, text, x=0.0, y=-0.3):
    """A small gray provenance tag under the axes (modeled, derived ...)."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.5, color=C['neutral'], ha='left', va='top')
sp.hbars_steps(out / 'hook_cost', [(f'Batch {1}  (3.5 ms)', 3.92, 'danger', '\\$3.92'), (f'Batch {8}  (3.8 ms)', 0.52, 'compute', '\\$0.52')], '', log=False, xlabel='Cost per million requests (dollars; modeled, \\$4 per GPU-hour)', size=sp.SHORT)
sp.hbars(out / 'recall_l12.svg', [('One at a time', 3550.0 / 1000, 'neutral', 'about 3,550/s'), ('All at once', 44411.0 / 1000, 'compute', '44,411/s')], '', log=False, xlabel='ResNet-50, thousand images per second, one RTX 4090', size=(5.0, 1.5))
ST = {'JPEG decode': 3.0, 'resize': 1.0, 'normalize': 0.5, 'copy to GPU': 0.5, 'forward pass': 3.528119406, 'top-5': 0.1}
PRE, COPY, FWD, POST = (4.5, ST['copy to GPU'], ST['forward pass'], ST['top-5'])
stages_now = [('Preprocess', PRE, 'data'), ('Copy', COPY, 'network'), ('Model', FWD, 'compute'), ('Post', POST, 'neutral')]
stages_fast = [('Preprocess', PRE, 'data'), ('Copy', COPY, 'network'), ('Model', 0.3528119406, 'compute'), ('Post', POST, 'neutral')]

def stacked_row(ax, y, parts, show=True, labels=True, fs=11):
    left = 0.0
    for name, v, ck in parts:
        ax.barh(y, v, left=left, color=C[ck], height=0.55, edgecolor='white', linewidth=1.5, alpha=1.0 if show else 0.0)
        if labels and show and (v >= 0.9 * (sum((p[1] for p in parts)) / 9)):
            ax.text(left + v / 2, y, name, ha='center', va='center', color='white', fontsize=fs, fontweight='bold')
        left += v
    return left
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.9))
    total = stacked_row(ax, 1, stages_now)
    ax.text(total + 0.12, 1, '8.6 ms', va='center', fontsize=13, fontweight='bold', color=C['text'])
    t2 = stacked_row(ax, 0, stages_fast, labels=True)
    ax.text(t2 + 0.12, 0, '???' if k == 1 else '5.5 ms: 1.6× faster', va='center', fontsize=13, fontweight='bold', color=C['danger'])
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['Today', 'Model 10× faster'], fontsize=12, fontweight='bold')
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('One ResNet-50 request (ms; modeled model, assumed host phases)', fontsize=11)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    sp.save(fig, out / f'budget_step{k}.svg')
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    lanes = {'Host CPU': 2, 'PCIe': 1, 'GPU': 0}
    for name, y in lanes.items():
        ax.axhspan(y - 0.38, y + 0.38, color='#F2F2F2', lw=0, zorder=0)
    DEC, RSZ, NRM = (ST['JPEG decode'], ST['resize'], ST['normalize'])
    host = [('decode 3 ms', 0, DEC), ('', DEC, RSZ), ('', DEC + RSZ, NRM)]
    for lab, x0, w in host:
        ax.barh(2, w, left=x0, color=C['data_l'], edgecolor=C['data'], lw=1.5, height=0.56)
        if lab:
            ax.text(x0 + w / 2, 2, lab, ha='center', va='center', fontsize=11)
    ax.text(DEC + RSZ / 2, 2.36, 'resize 1', ha='center', va='bottom', fontsize=11)
    ax.text(DEC + RSZ + NRM / 2, 2.78, 'normalize 0.5', ha='center', va='bottom', fontsize=11)
    if k >= 2:
        ax.barh(1, COPY, left=PRE, color=C['network_l'], edgecolor=C['network'], lw=1.5, height=0.56)
        ax.text(PRE + COPY + 0.1, 1, 'copy 0.5 ms', va='center', fontsize=11)
        ax.barh(0, FWD, left=PRE + COPY, color=C['compute_l'], edgecolor=C['compute'], lw=1.5, height=0.56)
        ax.text(PRE + COPY + FWD / 2, 0, 'forward\n3.5 ms', ha='center', va='center', fontsize=11)
        ax.barh(2, POST, left=PRE + COPY + FWD, color=C['data_l'], edgecolor=C['data'], lw=1.5, height=0.56)
        ax.text(PRE + COPY + FWD + POST / 2, 2.5, 'top-5 0.1', ha='center', va='bottom', fontsize=11)
    if k >= 3:
        ax.barh(0, PRE + COPY, left=0, color='white', edgecolor=C['danger'], hatch='//', lw=1.5, height=0.56)
        ax.text((PRE + COPY) / 2, 0, 'GPU idle 5 ms', ha='center', va='center', fontsize=12, color=C['danger'], fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1))
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(list(lanes), fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.5, 3.25)
    ax.set_xticks(range(10))
    ax.spines['left'].set_visible(False)
    ax.set_xticks(range(0, 10, 2))
    ax.set_xlabel('ms since the request arrived (8.6 ms total)', fontsize=11)
    tag(ax, 'After Model Serving, fig. 3: forward pass modeled, host assumed', y=-0.34)
    sp.save(fig, out / f'timeline_step{k}.svg')
for k in (1, 2):
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 2.9), gridspec_kw={'hspace': 0.9})
    ax = axes[0]
    TOT = PRE + COPY + FWD + POST
    for i in range(3):
        t0 = i * TOT
        ax.barh(1, PRE, left=t0, color=C['data'], height=0.6, edgecolor='white')
        ax.barh(1, COPY, left=t0 + PRE, color=C['network'], height=0.6, edgecolor='white')
        ax.barh(0, FWD, left=t0 + PRE + COPY, color=C['compute'], height=0.6, edgecolor='white')
        ax.barh(1, POST, left=t0 + PRE + COPY + FWD, color=C['neutral'], height=0.6)
    ax.text(26.2, 0, 'GPU busy 41%', va='center', fontsize=12, fontweight='bold', color=C['compute'])
    ax.set_title('Serial: one request at a time', fontsize=12, loc='left', color=C['text'], fontweight='bold')
    ax2 = axes[1]
    show = k == 2
    for i in range(6):
        ax2.barh(1, PRE, left=i * PRE, color=C['data'], height=0.6, edgecolor='white', alpha=1.0 if show else 0.0)
        if PRE + COPY + i * PRE + FWD <= 26.0:
            ax2.barh(0, FWD, left=PRE + COPY + i * PRE, color=C['compute'], height=0.6, edgecolor='white', alpha=1.0 if show else 0.0)
    ax2.text(26.2, 0, 'GPU busy 78%', va='center', fontsize=12, fontweight='bold', color=C['compute'], alpha=1.0 if show else 0.0)
    ax2.set_title('Pipelined: host prepares the next request', fontsize=12, loc='left', color=C['text'], fontweight='bold', alpha=1.0 if show else 0.0)
    for a in axes:
        a.set_yticks([1, 0])
        a.set_yticklabels(['Host CPU', 'GPU'], fontsize=11, fontweight='bold')
        a.set_xlim(0, 34)
        a.set_ylim(-0.5, 1.5)
        a.spines['left'].set_visible(False)
        a.tick_params(axis='y', length=0)
    axes[1].set_xlabel('Time (ms; modeled, book reference request)', fontsize=10)
    if not show:
        for t in ax2.get_yticklabels():
            t.set_alpha(0.0)
    sp.save(fig, out / f'pipeline_step{k}.svg')
S = 3.528119406
U = [0.5, 0.7, 0.8, 0.9, 0.95]
U_EX = 0.7
KNEE = {u: _NUMS[f'knee_{100 * u:.0f}'] for u in U}
points = [(u, KNEE[u]) for u in U if u != U_EX]
reach = {1: points[0][0], 2: points[1][0], 3: points[1][0], 4: 0.97, 5: 0.97}
for k in (1, 2, 3, 4, 5):
    fig, ax = plt.subplots(figsize=(5.8, 2.75))
    ax.axvspan(100 * U_EX, 100, color=C['danger_l'], alpha=0.6, lw=0)
    rho = np.linspace(0.0, reach[k], 400)
    ax.plot(rho * 100, S / (1 - rho), color=C['memory'], lw=3)
    shown = {1: 1, 2: 2, 3: 2, 4: 4, 5: 4}[k]
    for r, w in points[:shown]:
        ax.plot([r * 100], [w], 'o', ms=8, color=C['memory'], mec='white', mew=1.5, zorder=5)
        ax.annotate(f'{w:g} ms', (r * 100, w), xytext=(-8, -14 if w > 60 else 4), textcoords='offset points', ha='right', fontsize=11, color=C['memory'], fontweight='bold')
    if k == 3:
        u9, w9 = points[2]
        ax.plot([100 * u9], [w9], 'o', ms=9, mfc='white', mec=C['danger'], mew=2, zorder=5)
        ax.annotate(f'{100 * u9:.0f}%: ???', (100 * u9, w9), xytext=(-10, 4), textcoords='offset points', ha='right', fontsize=12, color=C['danger'], fontweight='bold')
    ax.text(8, 8, 'Mean', color=C['memory'], fontsize=12, fontweight='bold')
    if k >= 5:
        rr = np.linspace(0.0, 0.97, 400)
        ax.plot(rr * 100, 4.6 * S / (1 - rr), color=C['danger'], lw=3)
        ax.text(30, 30, 'p99', color=C['danger'], fontsize=12, fontweight='bold')
        ax.plot([100 * U_EX], [54.1], 'o', ms=9, color=C['danger'], mec='white', mew=1.5, zorder=6)
        ax.annotate('70%: p99 54.1 ms,\nmean 11.8 ms', (100 * U_EX, 54.1), xytext=(-10, 6), textcoords='offset points', ha='right', fontsize=11, color=C['danger'], fontweight='bold')
    ax.text(99, 2, 'planning danger zone', fontsize=10, color=C['danger'], ha='right')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Utilization (percent of time the GPU is busy)')
    ax.set_ylabel('Time in system (ms)')
    tag(ax, 'M/M/1 model, 3.5 ms service time (Model Serving, table 7)', y=-0.27)
    sp.save(fig, out / f'knee_step{k}.svg')
lam = np.linspace(0, 270, 400)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    s_old, s_new = (3.528119406, 1.788119406)
    w_old = s_old / (1 - lam * s_old / 1000)
    ax.plot(lam, w_old, color=C['memory'], lw=3)
    ax.text(145, 30, '3.5 ms model', color=C['memory'], fontsize=11, fontweight='bold')
    lam80 = 226.7496952
    ax.plot([lam80], [17.6], 'o', ms=10, color=C['danger'], mec='white', mew=1.5, zorder=5)
    ax.annotate('80% busy: 17.6 ms', (lam80, 17.6), xytext=(15, 22), ha='left', fontsize=11, color=C['danger'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=C['danger']))
    if k == 2:
        w_new = s_new / (1 - lam * s_new / 1000)
        ax.plot(lam, w_new, color=C['data'], lw=3)
        ax.plot([lam80], [3.0], 'o', ms=10, color=C['data'], mec='white', mew=1.5, zorder=5)
        ax.text(lam80 + 8, -4.5, '1.8 ms model, 41% busy: 3.0 ms', ha='right', va='center', fontsize=11, color=C['data'], fontweight='bold')
    ax.set_xlim(0, 270)
    ax.set_ylim(-8, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlabel('Requests per second')
    ax.set_ylabel('Mean time (ms)')
    tag(ax, 'M/M/1 model', y=-0.3)
    sp.save(fig, out / f'vote_fast_step{k}.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.3))
    rows = [('System A', 715761.0, 652002.0), ('System B', 712859.0, 584161.0)]
    for i, (name, off, srv) in enumerate(rows):
        y = 1 - i
        ax.barh(y + 0.18, off / 1000, height=0.34, color=C['neutral'])
        ax.text(off / 1000 + 8, y + 0.18, f'Offline {off:,.0f}', va='center', fontsize=11, color=C['neutral'], fontweight='bold')
        ax.barh(y - 0.18, srv / 1000, height=0.34, color=C['compute'], alpha=1.0 if k == 2 else 0.0)
        ax.text(srv / 1000 + 8, y - 0.18, f'Server {srv:,.0f}  ({srv / off:.0%} kept)', va='center', fontsize=11, color=C['compute'], fontweight='bold', alpha=1.0 if k == 2 else 0.0)
        ax.text(-10, y, name, ha='right', va='center', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 1050)
    ax.set_ylim(-0.5, 1.5)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('ResNet-50, thousands per second, 8x H100-SXM (MLPerf Inference v5.0)')
    sp.save(fig, out / f'mlperf_server_step{k}.svg')
bb = np.unique(np.round(np.logspace(0, np.log10(128), 80)).astype(int))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    PER = 0.03286147624
    ax.axhline(PER, color=C['compute'], lw=2, ls='--')
    ax.text(1.05, PER * 1.25, 'work on the image itself: 0.033 ms', fontsize=11, color=C['compute'], fontweight='bold', va='bottom')
    if k == 2:
        ax.plot(bb, [3.528119406, 1.780490441, 1.197947453, 0.9066759586, 0.7319130621, 0.6154044645, 0.5321840376, 0.4697687174, 0.4212234684, 0.3823872692, 0.3506121971, 0.3241329704, 0.3017274708, 0.2825227569, 0.2658786715, 0.2513150968, 0.2384648839, 0.2270424723, 0.2168224199, 0.2076243727, 0.1917368367, 0.1848292123, 0.1784972233, 0.1672944735, 0.1576921166, 0.1533876117, 0.145611732, 0.1387783832, 0.1327259885, 0.1273279068, 0.1202429245, 0.1160819031, 0.1105338747, 0.1056793498, 0.1013959454, 0.09758847493, 0.09312454399, 0.09016078656, 0.08663467515, 0.08351738826, 0.0800946915, 0.07767247534, 0.07497301756, 0.07213403724, 0.07004507123, 0.06781405553, 0.06583560765, 0.06379296234, 0.06198862565, 0.06016817881], color=C['neutral'], lw=3)
    pts = (1,) if k == 1 else (1, 8, 32)
    offs = {1: (10, -4), 8: (10, 2), 32: (10, 2)}
    for b in pts:
        v = _NUMS[f'per_image_{b}']
        ax.plot([b], [v], 'o', ms=9, color=C['danger'] if b == 1 else C['text'], mec='white', mew=1.5, zorder=5)
        ax.annotate(f"batch {b}: {_VALS[f'per_image_{b}']} ms ({_VALS[f'fixed_share_{b}']}% fixed)", (b, v), xytext=offs[b], textcoords='offset points', fontsize=11, fontweight='bold', color=C['danger'] if b == 1 else C['text'])
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlim(0.8, 160)
    ax.set_ylim(0.02, 6)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
    ax.set_yticks([0.03, 0.1, 0.3, 1, 3])
    ax.set_yticklabels(['0.03', '0.1', '0.3', '1', '3'])
    ax.minorticks_off()
    ax.set_xlabel('Batch size')
    ax.set_ylabel('ms per image')
    tag(ax, 'Modeled: 3.50 ms fixed per pass + 0.033 ms per image (Model Serving, table 8)', y=-0.36)
    sp.save(fig, out / f'fixed_cost_step{k}.svg')
bs = np.unique(np.round(np.logspace(0, np.log10(128), 60)).astype(int))
for k in (1, 2):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.9, 2.6), gridspec_kw={'width_ratios': [1.1, 1], 'wspace': 0.55})
    tput = np.array([283.437119, 561.6430041, 834.7611556, 1102.929873, 1366.282489, 1624.947588, 1879.049219, 2128.707091, 2374.036764, 2615.149825, 2852.154056, 3085.153599, 3314.249105, 3539.53788, 3761.114023, 3979.068558, 4404.462257, 4612.069178, 5017.502805, 5410.400161, 5791.333837, 6160.842079, 6519.431319, 7037.877818, 7371.19681, 7853.737844, 8316.49766, 8904.78639, 9462.586609, 9992.208587, 10495.73493, 11091.29632, 11652.30505, 12283.99738, 12874.57359, 13516.77256, 14113.6467, 14746.20552, 15401.66082, 16004.53091, 16620.08091])
    ax.plot(bs, tput / 1000, color=C['compute'], lw=3)
    for b in (1, 32, 128):
        lab = f"{b}: { {1: '3.53', 32: '4.55', 128: '7.7'}[b]} ms\n{ {1: '283', 32: '7,040', 128: '16,600'}[b]}/s"
        v = {1: 283.437119, 32: 7037.877818, 128: 16620.08091}[b] / 1000
        ax.plot([b], [v], 'o', ms=7, color=C['compute'], mec='white', zorder=5)
        ax.annotate(lab, (b, v), xytext={1: (12, 32), 32: (-8, 14), 128: (-8, 4)}[b], textcoords='offset points', ha='right' if b > 1 else 'left', fontsize=10, color=C['text'])
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 8, 32, 128])
    ax.set_xticklabels(['1', '8', '32', '128'])
    ax.set_ylim(0, 20)
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Thousand img/s')
    ax.set_title('H100, modeled', fontsize=12, loc='left', fontweight='bold')
    show = k == 2
    a = 1.0 if show else 0.0
    rows = [('1 at a time', 2415.0 / 1000, 'neutral', '2,415/s'), ('8 at a time', 10390.0 / 1000, 'compute', '10,390/s'), ('All at once', 12855.0 / 1000, 'compute', '12,855/s')]
    for i, (lab, v, ck, txt) in enumerate(rows):
        y = 2 - i
        bx.barh(y, v, color=C[ck], height=0.6, alpha=a)
        bx.text(v + 0.3, y, txt, va='center', fontsize=10, fontweight='bold', color=C[ck], alpha=a)
        bx.text(-0.3, y, lab, ha='right', va='center', fontsize=10, fontweight='bold', alpha=a)
    bx.set_yticks([])
    bx.set_xlim(0, 20)
    bx.set_ylim(-0.5, 2.5)
    bx.spines['left'].set_visible(False)
    bx.set_xlabel('Thousand img/s', alpha=a)
    bx.set_title('L4, measured', fontsize=12, loc='left', fontweight='bold', alpha=a)
    if not show:
        bx.spines['bottom'].set_alpha(0.0)
        bx.tick_params(axis='x', colors='white')
    sp.save(fig, out / f'batch_curve_step{k}.svg')
rng_arrivals = [0.8, 2.6, 3.1, 5.4, 7.9, 9.2, 11.5, 14.8, 18.6, 20.9, 21.7, 23.0, 24.4, 26.1, 27.3, 28.0, 29.4]
for k in (1, 2, 3):
    fig = plt.figure(figsize=(5.9, 3.0))
    ax = fig.add_axes([0.06, 0.62, 0.9, 0.3])
    W = 10.0
    for w in (0, W, 2 * W):
        ax.axvspan(w + 0.15, w + W - 0.15, color=C['memory_l'], alpha=0.7, lw=0)
        ax.annotate('', (w + W, -0.55), xytext=(w + W, 0.55), arrowprops=dict(arrowstyle='-|>', color=C['compute'], lw=2))
    ax.text(W + 0.4, 0.75, 'dispatch', ha='left', fontsize=12, color=C['compute'])
    for t in rng_arrivals:
        ax.plot([t, t], [-0.35, 0.35], color=C['text'], lw=2)
    ax.annotate('', (W, -0.95), xytext=(rng_arrivals[0], -0.95), arrowprops=dict(arrowstyle='<->', color=C['danger'], lw=1.5))
    if k == 3:
        ax.annotate('cap reached: dispatch early', (26.1, 0.4), xytext=(29.8, 1.15), fontsize=12, color=C['data'], ha='right', va='bottom', arrowprops=dict(arrowstyle='-', color=C['data']))
    ax.set_xlim(0, 30)
    ax.set_ylim(-1.6, 1.6)
    ax.set_yticks([])
    for s in ('left',):
        ax.spines[s].set_visible(False)
    ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xlabel('ms; 10 ms windows, arrivals illustrative', fontsize=10, color=C['neutral'], labelpad=1)
    show = k >= 2
    rates = [Q_(50, '1 / s'), Q_(200, '1 / s'), Q_(1000, '1 / s')]
    for j, r in enumerate(rates):
        q, mean = ({50.0: 50, 200.0: 200, 1000.0: 1000}[r.magnitude], {50.0: 0.5, 200.0: 2.0, 1000.0: 10.0}[r.magnitude])
        hx = fig.add_axes([0.06 + j * 0.32, 0.08, 0.26, 0.24])
        pr = {50.0: [0.6065306597, 0.3032653299, 0.07581633246, 0.01263605541, 0.001579506926, 0.0001579506926, 1.316255772e-05, 9.401826942e-07, 5.876141839e-08, 3.264523244e-09, 1.632261622e-10, 7.419371009e-12, 3.091404587e-13, 1.189001764e-14, 4.246434872e-16, 1.415478291e-17, 4.423369659e-19, 1.300991076e-20, 3.6138641e-22, 9.510168685e-24, 2.377542171e-25], 200.0: [0.1353352832, 0.2706705665, 0.2706705665, 0.1804470443, 0.09022352216, 0.03608940886, 0.01202980295, 0.003437086558, 0.0008592716396, 0.0001909492532, 3.818985065e-05, 6.943609209e-06, 1.157268201e-06, 1.780412618e-07, 2.543446597e-08, 3.391262129e-09, 4.239077661e-10, 4.98715019e-11, 5.541277988e-12, 5.832924198e-13, 5.832924198e-14], 1000.0: [4.539992976e-05, 0.0004539992976, 0.002269996488, 0.00756665496, 0.0189166374, 0.0378332748, 0.063055458, 0.09007922572, 0.1125990321, 0.1251100357, 0.1251100357, 0.1137363961, 0.09478033009, 0.07290794622, 0.05207710445, 0.03471806963, 0.02169879352, 0.01276399619, 0.007091108993, 0.003732162628, 0.001866081314]}[r.magnitude]
        hx.bar(range(21), pr, color=C['memory'], width=0.8, alpha=1.0 if show else 0.0)
        hx.set_xlim(-0.8, 20.8)
        hx.set_ylim(0, 0.65)
        hx.set_yticks([])
        hx.spines['left'].set_visible(False)
        hx.set_xticks([0, 10, 20])
        hx.tick_params(axis='x', labelsize=11, colors=C['text'] if show else 'white')
        hx.spines['bottom'].set_alpha(1.0 if show else 0.0)
        extra = ': 61% empty' if r is Q_(50, '1 / s') else ''
        hx.set_title(f"{ {50.0: '50', 200.0: '200', 1000.0: '1,000'}[r.magnitude]} QPS{extra}", fontsize=12, fontweight='bold', color=C['text'], alpha=1.0 if show else 0.0)
    fig.text(0.06, -0.05, 'Requests per 10 ms window (Poisson)', fontsize=11, color=C['neutral'] if show else 'white')
    sp.save(fig, out / f'window_step{k}.svg')
phases = [('Weights from local SSD', 0.5, 'memory', '0.5 s'), ('Weights from S3', 5.0, 'memory', '3–5 s'), ('CUDA context', 0.5, 'neutral', '0.3–0.5 s'), ('Warmup + runtime', 0.6, 'neutral', '0.6 s'), ('TensorRT engine build', 30.0, 'compute', '15–30 s')]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, (lab, v, ck, txt) in enumerate(phases):
        y = len(phases) - 1 - i
        a = 1.0 if k == 2 or i < 4 else 0.0
        if lab.startswith('Weights from S3'):
            lo, hi = (3.0, 5.0)
            ax.barh(y, hi - lo, left=lo, color=C[ck], height=0.6, alpha=a * 0.45)
            ax.barh(y, lo, color=C[ck], height=0.6, alpha=a)
        elif lab.startswith('TensorRT'):
            lo, hi = (15.0, 30.0)
            ax.barh(y, lo, color=C[ck], height=0.6, alpha=a)
            ax.barh(y, hi - lo, left=lo, color=C[ck], height=0.6, alpha=a * 0.45)
        else:
            ax.barh(y, v, color=C[ck], height=0.6, alpha=a)
        ax.text(v * 1.15, y, txt, va='center', fontsize=13, fontweight='bold', color=C[ck], alpha=a)
        ax.text(0.085, y, lab, ha='right', va='center', fontsize=13, fontweight='bold', alpha=a)
    ax.set_xscale('log')
    ax.set_xlim(0.1, 200)
    ax.set_ylim(-0.6, len(phases) - 0.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(['0.1', '1', '10', '100'])
    ax.set_xlabel('Seconds (log scale; representative, table 17)')
    sp.save(fig, out / f'cold_start_step{k}.svg')
fig, ax = plt.subplots(figsize=(4.3, 2.2))
DL = 50.0
ax.axvspan(DL, 360, color=C['danger_l'], alpha=0.6, lw=0)
ax.axvline(DL, color=C['danger'], lw=2.5, ls='--')
ax.text(DL + 6, 1.25, '50 ms deadline', color=C['danger'], fontsize=12, fontweight='bold')
ax.barh(0.5, 3.5, color=C['compute'], height=0.35)
ax.annotate('forward pass: 3.5 ms', (4, 0.5), xytext=(62, 0.5), va='center', fontsize=12, color=C['compute'], fontweight='bold', arrowprops=dict(arrowstyle='-', color=C['compute']))
ax.barh(-0.2, 312.5, color=C['memory'], height=0.35)
ax.text(312.5 - 6, -0.2, 'load 10 GB: 312.5 ms', va='center', ha='right', fontsize=12, color='white', fontweight='bold')
ax.set_xlim(0, 360)
ax.set_ylim(-0.6, 1.6)
ax.set_yticks([])
ax.set_xticks([0, 100, 200, 300])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Milliseconds')
sp.save(fig, out / 'model_swap.svg')
sp.hbars(out / 'static_dynamic.svg', [('On demand, batch 1', 35.28119406, 'danger', '35 s'), ('Overnight, large batch', 0.111769152, 'data', '0.11 s: 316× faster')], '', log=False, size=sp.SHORT, xlabel='Seconds to classify 10,000 photos on one H100')
resnet = [('Prep 4.5', PRE, 'data'), ('', COPY, 'network'), ('Model 3.5', FWD, 'compute'), ('', POST, 'neutral')]
DS = {'parse': 0.5, 'embedding lookup': 6.0, 'MLP forward': 1.5, 'rank and filter': 1.0}
dlrm = [('', DS['parse'], 'neutral'), ('Embeddings 6', DS['embedding lookup'], 'memory'), ('MLP', DS['MLP forward'], 'compute'), ('', DS['rank and filter'], 'neutral')]
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    for y, parts, show in ((1, resnet, True), (0, dlrm, k >= 2)):
        left = 0.0
        for name, v, ck in parts:
            ax.barh(y, v, left=left, color=C[ck], height=0.6, edgecolor='white', lw=1.5, alpha=1.0 if show else 0.0)
            if name:
                ax.text(left + v / 2, y, name, ha='center', va='center', color='white', fontsize=11, fontweight='bold', alpha=1.0 if show else 0.0)
            left += v
        txt = '8.6 ms\nmodel 41%' if y == 1 else '9 ms\nMLP 17%'
        ax.text(left + 0.15, y, txt, va='center', fontsize=12, fontweight='bold', alpha=1.0 if show else 0.0)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['ResNet-50', 'DLRM'], fontsize=12, fontweight='bold')
    if k < 2:
        ax.get_yticklabels()[1].set_alpha(0.0)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('One request (ms; Model Serving, tables 3, 4)', fontsize=11)
    sp.save(fig, out / f'dlrm_step{k}.svg')
nn = np.arange(1, 100 + 1)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(3.9, 2.0))
    A, N10, N100 = (0.99, 10, 100)
    ax.plot(nn, 100 * A ** nn, color=C['memory'], lw=3)
    f10, f100 = (90.4, 36.6)
    ax.plot([1], [100 * A], 'o', ms=8, color=C['memory'], mec='white', zorder=5)
    ax.plot([N10], [f10], 'o', ms=8, color=C['memory'], mec='white', zorder=5)
    ax.annotate('10 servers: 90.4%', (N10, f10), xytext=(10, -4), textcoords='offset points', fontsize=11, fontweight='bold', color=C['memory'])
    if k == 1:
        ax.annotate('100: ???', (N100, f100), xytext=(-12, -24), textcoords='offset points', ha='right', fontsize=12, fontweight='bold', color=C['danger'])
    else:
        ax.plot([N100], [f100], 'o', ms=9, color=C['danger'], mec='white', zorder=5)
        ax.annotate('100: 36.6%', (N100, f100), xytext=(-12, -24), textcoords='offset points', ha='right', fontsize=12, fontweight='bold', color=C['danger'])
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Servers each request fans out to')
    ax.set_ylabel('Requests on time (%)')
    tag(ax, 'Each server on time 99%, independent', y=-0.4)
    sp.save(fig, out / f'fanout_step{k}.svg')
rt = [('PyTorch eager', 3.528119406, 'neutral', '3.53 ms')]
for _n, _k, _ck in (('TorchScript', 'ts', 'compute'), ('ONNX Runtime', 'onnx', 'compute'), ('TensorRT FP32', 'fp32', 'compute'), ('TensorRT FP16', 'fp16', 'data'), ('TensorRT INT8', 'int8', 'data')):
    _x = {'ts': 2.0, 'onnx': 5.8, 'fp32': 9.3, 'fp16': 10.6, 'int8': 11.6}[_k]
    _xs = f'{_x:.0f}' if abs(_x - round(_x)) < 0.05 else {'ts': '2.0', 'onnx': '5.8', 'fp32': '9.3', 'fp16': '10.6', 'int8': '11.6'}[_k]
    rt.append((_n, {'ts': 1.788119406, 'onnx': 0.6081194058, 'fp32': 0.3812388116, 'fp16': 0.3331194058, 'int8': 0.30380588}[_k], _ck, f"{ {'ts': '1.79', 'onnx': '0.61', 'fp32': '0.38', 'fp16': '0.33', 'int8': '0.30'}[_k]} ms ({_xs}×)"))
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    for i, (lab, v, ck, txt) in enumerate(rt):
        y = 5 - i
        show = k == 2 or i < 4
        ax.barh(y, v, color=C[ck], height=0.62, alpha=1.0 if show else 0.0)
        ax.text(v + 0.05, y, txt, va='center', fontsize=11, fontweight='bold', color=C[ck], alpha=1.0 if show else 0.0)
        ax.text(-0.05, y, lab, ha='right', va='center', fontsize=11, fontweight='bold', alpha=1.0 if show else 0.0)
    ax.text(2.6, 3.0, 'less dispatch:\n9.3×', fontsize=11, color=C['compute'], fontweight='bold', ha='center', va='center')
    ax.text(2.6, 0.5, 'lower precision:\n1.3× more', fontsize=11, color=C['data'], fontweight='bold', ha='center', va='center', alpha=1.0 if k == 2 else 0.0)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel('ResNet-50, batch 1, one H100 (ms; modeled, Example 13.6)')
    sp.save(fig, out / f'runtime_step{k}.svg')
ARR = [0, 2, 5, 9]
TO, FREE, B4, RESP = (12, 18, 3.626703835, 4)
DLN, DONE = (50, 25.62670383)
for k in (1, 2, 3):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    for t in ARR:
        ax.plot([t, t], [1.65, 2.35], color=C['text'], lw=2.5)
    ax.text(ARR[-1] / 2, 2.55, 'arrivals at ' + ', '.join((_VALS[f'arrive_{i}'] for i in range(len(ARR)))) + ' ms', ha='center', fontsize=11, fontweight='bold')
    ax.barh(1.0, TO, color=C['memory_l'], edgecolor=C['memory'], height=0.5)
    ax.text(TO / 2, 1.0, 'timer 12 ms', ha='center', va='center', fontsize=11, color=C['memory'], fontweight='bold')
    a2 = 1.0 if k >= 2 else 0.0
    ax.barh(1.0, FREE - TO, left=TO, color=C['danger_l'], edgecolor=C['danger'], height=0.5, hatch='//', alpha=a2)
    ax.text((TO + FREE) / 2, 0.62, 'worker busy 6 ms', ha='center', va='top', fontsize=10.5, color=C['danger'], fontweight='bold', alpha=a2)
    ax.barh(1.0, B4, left=FREE, color=C['compute'], height=0.5, alpha=a2)
    ax.text(FREE + B4 / 2, 1.35, f'batch of {len(ARR)}: 3.6 ms', ha='center', va='bottom', fontsize=10.5, color=C['compute'], fontweight='bold', alpha=a2)
    a3 = 1.0 if k >= 3 else 0.0
    ax.barh(1.0, RESP, left=FREE + B4, color=C['network'], height=0.5, alpha=a3)
    ax.text(FREE + B4 + 3.4, 0.62, 'response 4 ms', ha='left', va='top', fontsize=10.5, color=C['network'], fontweight='bold', alpha=a3)
    ax.annotate('', (DLN, -0.05), xytext=(DONE, -0.05), arrowprops=dict(arrowstyle='<->', color=C['data'], lw=1.8, alpha=a3))
    ax.text(DLN - 1.0, -0.2, 'done at 25.6 ms: 24.4 ms of slack', ha='right', va='top', fontsize=11, color=C['data'], fontweight='bold', alpha=a3)
    ax.axvline(DLN, color=C['danger'], lw=2, ls='--')
    ax.text(DLN - 0.5, 2.55, 'deadline 50 ms', ha='right', fontsize=11, color=C['danger'], fontweight='bold')
    ax.set_xlim(-1, 52)
    ax.set_ylim(-0.9, 2.9)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('ms since the oldest request arrived (hypothetical trace)')
    sp.save(fig, out / f'dual_trigger_step{k}.svg')
inst = [('c5.xlarge (CPU)', 'CPU'), ('g4dn.xlarge (T4)', 'T4'), ('H100', 'H100')]
T4_USD, T4_RATE = (0.53, 400)
H100_USD = 4.0
load = np.arange(50, 7040.0 + 1, 10)
for k in (1, 2):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.4, 2.5), gridspec_kw={'width_ratios': [1, 1.3], 'wspace': 0.55})
    for i, (lab, key) in enumerate(inst):
        y = 2 - i
        cpm = {'c5.xlarge (CPU)': 0.9444444444, 'g4dn.xlarge (T4)': 0.3680555556, 'H100': 0.1578758739}[lab]
        ck = 'data' if lab == 'H100' else 'neutral'
        ax.barh(y, cpm, color=C[ck], height=0.6)
        ax.text(cpm + 0.03, y, f"\\${ {'c5.xlarge (CPU)': '0.94', 'g4dn.xlarge (T4)': '0.37', 'H100': '0.16'}[lab]}", va='center', fontsize=12, fontweight='bold', color=C[ck])
        ax.text(-0.03, y, {'c5.xlarge (CPU)': 'CPU (c5)', 'g4dn.xlarge (T4)': 'T4'}.get(lab, lab), ha='right', va='center', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 1.3)
    ax.set_ylim(-0.5, 2.5)
    ax.set_title('Fully busy', fontsize=12, loc='left', fontweight='bold')
    ax.set_xlabel('\\$ per million images', fontsize=11)
    a = 1.0 if k >= 2 else 0.0
    t4 = np.ceil(load / T4_RATE) * T4_USD / (load * 3600) * 1000000.0
    h1 = H100_USD / (load * 3600) * 1000000.0
    bx.plot(load / 1000, h1, color=C['data'], lw=2.5, alpha=a)
    bx.plot(load / 1000, t4, color=C['neutral'], lw=2, alpha=a)
    bx.axvline(2800.0 / 1000 + 0.1, color=C['danger'], lw=1.5, ls='--', alpha=a)
    bx.text(3.1, 1.2, 'crossover\nnear 3,000/s', fontsize=12, color=C['danger'], fontweight='bold', alpha=a)
    bx.text(3.15, 1.75, 'one H100', ha='left', fontsize=12, color=C['data'], fontweight='bold', alpha=a)
    bx.text(7.05, 0.55, 'T4s sized\nto load', ha='right', fontsize=12, color=C['neutral'], fontweight='bold', alpha=a)
    bx.set_ylim(0, 2.0)
    bx.set_xlim(0, 7.1)
    bx.set_title('At a given load (derived)', fontsize=12, loc='left', fontweight='bold', alpha=a)
    bx.set_xlabel('Load (thousand img/s)', fontsize=11, alpha=a)
    if k < 2:
        for sp_ in bx.spines.values():
            sp_.set_alpha(0.0)
        bx.tick_params(colors='white')
    sp.save(fig, out / f'cost_hw_step{k}.svg')
HOOK = [('No deadline\n(Offline)', 31216.8, 'neutral', '31,217'), ('TTFT 2 s,\nTPOT 200 ms', 30903.6, 'compute', '30,904'), ('TTFT 450 ms,\nTPOT 40 ms', 12979.4, 'danger', '')]
fig, ax = plt.subplots(figsize=(2.6, 2.3))
for i, (lab, v, ck, txt) in enumerate(HOOK):
    y = 2 - i
    ax.barh(y, v / 1000 if i < 2 else 0, color=C[ck], height=0.6)
    ax.text(-0.6, y, lab, ha='right', va='center', fontsize=11, fontweight='bold')
    ax.text(v / 1000 + 0.6 if i < 2 else 0.8, y, txt if i < 2 else '???', va='center', fontsize=11, fontweight='bold', color=C[ck])
ax.set_yticks([])
ax.set_xlim(0, 40)
ax.set_ylim(-0.5, 2.5)
ax.spines['left'].set_visible(False)
ax.set_xticks([0, 20, 40])
ax.set_xlabel('Thousand tokens/s,\nLlama 2 70B, 8× H100', fontsize=11)
sp.save(fig, out / 'next_hook.svg')
sp.hbars(out / 'synth_hook.svg', [(f'Batch {1}', 3.92, 'danger', '\\$3.92 per million'), (f'Batch {8}', 0.52, 'compute', '\\$0.52')], '', log=False, size=(3.4, 1.3), xlabel='Cost per million requests')
sp.hbars(out / 'synth_gpus.svg', [('Batch 1', 36, 'danger', '36 H100s'), ('Batched', 3, 'data', '3 H100s')], '', log=False, size=(3.4, 1.5), xlabel='5,000 req/s, 50 ms p99 (plan)')
sp.hbars(out / 'synth_cost.svg', [('Batch 1', 8.0, 'danger', '\\$8.00 per million'), ('Batched', 0.67, 'data', '\\$0.67')], '', log=False, size=(3.4, 1.5), xlabel='Fleet cost at 5,000 req/s (derived)')

def thumb(name, draw, size=(1.0, 0.4)):
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    ax.axis('off')
    fig.savefig(out / f'{name}.svg', format='svg', pad_inches=0.0, facecolor='white')
    plt.close(fig)

def _t_timeline(ax):
    ax.barh(2, PRE, color=C['data'], height=0.6)
    ax.barh(1, COPY, left=PRE, color=C['network'], height=0.6)
    ax.barh(0, FWD, left=PRE + COPY, color=C['compute'], height=0.6)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.6, 2.6)

def _t_knee(ax):
    rr = np.linspace(0, 0.93, 200)
    ax.axvspan(U_EX, 1.0, color=C['danger_l'], lw=0)
    ax.plot(rr, 1 / (1 - rr), color=C['memory'], lw=1.6)
    ax.plot(rr, 4.6 / (1 - rr), color=C['danger'], lw=1.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 30)

def _t_window(ax):
    for w in (0, 10, 20):
        ax.axvspan(w + 0.3, w + 9.7, color=C['memory_l'], lw=0)
    for t in [0.8, 2.6, 3.1, 5.4, 7.9, 9.2, 11.5, 14.8, 18.6, 20.9, 23.0, 26.1, 28.0]:
        ax.plot([t, t], [0.2, 0.8], color=C['text'], lw=1)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)

def _t_cost(ax):
    ld = np.linspace(0.3, 7, 300)
    ax.plot(ld, H100_USD / ld, color=C['data'], lw=1.6)
    ax.plot(ld, np.ceil(ld / (T4_RATE / 1000)) * T4_USD / ld, color=C['neutral'], lw=1.2)
    ax.set_xlim(0, 7.1)
    ax.set_ylim(0, 3)

def _t_cold(ax):
    ax.barh([3, 2, 1], [0.5, 1.0, 0.6], color=[C['memory'], C['neutral'], C['neutral']], height=0.6)
    ax.barh(0, 8, color=C['compute'], height=0.6)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(-0.6, 3.6)
thumb('thumb_timeline', _t_timeline)
thumb('thumb_knee', _t_knee)
thumb('thumb_window', _t_window)
thumb('thumb_cost', _t_cost)
thumb('thumb_cold', _t_cold)
print('figures written')
