#!/usr/bin/env python3
"""Plots and diagrams for Volume II, Lecture 6 (Distributed Training).

Every value is read from MLSysIM through calcs.py (the same Scenarios.* the
chapter reads) or from a cited Literature series:

  - Hook and batch sweep: slides/vol2/06_distributed_training/images/data/goyal_resnet50_large_minibatch.csv
    (Goyal et al. 2017, Table 1 and Figure 3).
  - 175B data-parallel step and the hybrid scoreboard: Scenarios.Gpt3DataParallel1K,
    Gpt3TensorDataParallel1K, Gpt3ThreeDParallel1K.
  - GPT-2 inside and across nodes: Scenarios.Gpt2DataParallelOneNode and
    Gpt2DataParallelFourNodes10GbE.
  - MLPerf Training: slides/vol2/06_distributed_training/images/data/mlperf_training_llm_scaling.csv.
  - Straggler: Scenarios.StragglerFreeStep1K (a^N, calc_all_up_availability).
  - Critical batch: gradient-noise model of McCandlish et al. 2018 as the
    chapter plots it, steps 1 + Bcrit/B and samples 1 + B/Bcrit (dimensionless).
  - Memory wall: slides/vol2/06_distributed_training/images/data/epoch_notable_models.csv.
  - ZeRO: Scenarios.Llama2_7B_ZeroStages64.
  - Pipeline: Scenarios.NaiveFourStagePipeline, GPipeFourStageWalkthrough,
    Pipeline8Stage32Micro, Pipeline16Stage32Micro (calc_pipeline_bubble).
  - Tensor parallelism: Scenarios.Gpt3TensorParallelLayerTp8.
  - Megatron-LM weak scaling: slides/vol2/06_distributed_training/images/data/megatron_lm_weak_scaling.csv.
  - The top half of the scaling-wall figure and the timelines are schematic.

Run from this folder: python make_figs.py
"""
from types import SimpleNamespace
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "mlsysim"))
from mlsysim.core.units import Q_
import mlsysim
_VALS = {'goyal_gpus': '256', 'goyal_hours_word': 'one', 'goyal_hours': '1', 'goyal_base_hours': '29', 'goyal_base_gpus': '8', 'goyal_base_batch': '256', 'goyal_batch': '8,192', 'goyal_batch_k': '8', 'goyal_servers': '32', 'goyal_per_gpu': '32', 'goyal_batch_x': '32', 'goyal_gpu_x': '32', 'goyal_eff_pct': '90', 'goyal_eth_gbps': '50', 'goyal_epochs': '90', 'goyal_warmup_epochs': '5', 'goyal_err_base': '23.60', 'goyal_err_nowarm': '24.84', 'goyal_err_warm': '23.74', 'goyal_spread': '0.3', 'goyal_up1_k': '16', 'goyal_up1_err': '24.79', 'goyal_up2_k': '32', 'goyal_up2_err': '27.55', 'goyal_up3_k': '64', 'goyal_up3_err': '33.96', 'params_b': '175', 'state_bpp': '16', 'opt_bpp': '12', 'fp16_bytes': '2', 'state_gb': '2,800', 'state_tb': '2.8', 'hbm_gb': '80', 'state_gpus': '35', 'ref_gpus': '1,024', 'ref_nodes': '128', 'gpus_per_node': '8', 'nvlink_gbs': '450', 'ndr_gbs': '50', 'nvlink_ndr_x': '9', 'grad_gb': '350', 'ring_gb': '699', 'ar_a_s': '14', 'batch_tokens_m': '2', 'flops_per_pt': '6', 'step_eflops': '2.1', 'local_eff_pct': '50', 'fp8_tflops': '1,979', 't_compute_s': '2.1', 'exposed_a_s': '11.9', 'step_a_s': '14.0', 'eff_a_pct': '15', 'resid_pct': '1', 'resid_gpus': '512', 'resid_share_pct': '84', 'gpt2_step_s': '1.8', 'gpt2_step_ms': '1,800', 'gpt2_gpus_1': '8', 'gpt2_sync1_gb': '5.2', 'gpt2_gpus_4': '32', 'gpt2_sync4_gb': '5.8', 'eth10_gbs': '1.25', 'eth10_gbps': '10', 'gpt2_comm1_ms': '11.7', 'gpt2_eff1_pct': '99.4', 'gpt2_inter_ms': '4,650', 'gpt2_comm4_ms': '4,661.7', 'gpt2_comm4_round_ms': '4,662', 'gpt2_step4_ms': '6,461.7', 'gpt2_step1_ms': '1,812', 'gpt2_step4_round_ms': '6,462', 'gpt2_eff4_pct': '27.9', 'gpt2_speedup4': '8.9', 'gpt2_accum': '4', 'gpt2_batch': '512', 'gpt2_acc_hours': '3.1', 'gpt2_acc_usd': '100', 'gpt2_4_usd': '359', 'gpt2_4_hours': '2.8', 'node_hour_usd': '32', 'eos_n0': '512', 'eos_n1': '10,752', 'eos_t0': '58', 'eos_t1': '3.9', 'eos_n_x': '21', 'eos_t_x': '14.9', 'eos_eff_pct': '71', 'mlperf_min_pct': '71', 'mlperf_max_pct': '84', 'strag_pct': '1', 'strag_on': '0.99', 'strag_n': '1,000', 'strag_n_plain': '1000', 'strag_expected': '10', 'strag_node_pct': '92', 'strag_100_pct': '37', 'strag_1k_pct': '0.004', 'cb_trace_m': '1.6', 'cb_norm': '100', 'cb_devices': '64', 'cb_local': '4', 'cb_accum': '32', 'cb_global': '8,192', 'cb_bcrit': '16,000', 'cb_per_gpu': '128', 'cb_crossover': '125', 'cb_params_b': '7', 'ssp_base_iters': '100,000', 'bsp8_workers': '8', 'bsp8_iters': '12,800', 'bsp8_comm_pct': '15', 'bsp8_speedup': '6.8', 'bsp8_usd': '26', 'bsp64_workers': '64', 'bsp64_iters': '1,720', 'bsp64_comm_pct': '45', 'bsp64_speedup': '40.1', 'bsp64_usd': '35', 'ssp64_workers': '64', 'ssp64_iters': '2,100', 'ssp64_comm_pct': '25', 'ssp64_speedup': '38.1', 'ssp64_usd': '37', 'ssp_s': '4', 'ssp_batch': '32', 'gpu_hour_usd': '2', 'wall_growth': '15', 'wall_orders_word': 'four', 'wall_gap': '35', 'z7_params_b': '7', 'z7_gpus': '64', 'a100_gb': '80', 'z7_ddp_gb': '112', 'z7_opt_gb': '84', 'z7_z1_gb': '29.3', 'z7_z2_gb': '15.5', 'z7_z3_gb': '1.75', 'z7_weights_gb': '14', 'z3_traffic_x': '1.5', 'z175_gpus': '64', 'z175_state_gb': '43.75', 'z175_act_gb': '4.8', 'z175_total_gb': '48.6', 'z175_seq': '2,048', 'z175_gather_tb': '1.1', 'z175_gather_x': '10', 'z175_gather_x_word': 'ten', 'naive_busy': '2', 'naive_slots': '8', 'naive_idle': '6', 'pipe_p': '4', 'pipe_busy': '8', 'pipe_slots': '14', 'pipe_idle': '6', 'pipe_bubble_pct': '43', 'pipe_fill': '3', 'pipe_phase': '7', 'p8m32_p': '8', 'p8m32_m': '32', 'p8m32_fill': '7', 'p8m32_phase': '39', 'p16m32_p': '16', 'p16m32_m': '32', 'p16m32_fill': '15', 'p16m32_phase': '47', 'p8m8_p': '8', 'p8m8_m': '8', 'p8m8_fill': '7', 'p8m8_phase': '15', 'p8m32_bubble_pct': '18', 'p16m32_bubble_pct': '31.9', 'p16m32_bubble_round_pct': '32', 'p8m8_bubble_pct': '47', 'bubble_target_pct': '10', 'bubble_m_coef': '9', 'p16_m_needed': '135', 'depth_word': 'four', 'tp_degree': '8', 'tp_tokens': '2,048', 'tp_compute_ms': '1.9', 'tp_nvlink_ms': '0.4', 'tp_ndr_ms': '3.5', 'gpt3_layers': '96', 'meg_2_pct': '95', 'meg_4_pct': '82', 'meg_8_pct': '77', 'meg_top_pct': '74', 'meg_gpus': '512', 'meg_servers': '64', 'meg_base_b': '1.2', 'meg_top_b': '8.3', 'nvswitch_gbs': '300', 'ring_nvlink_tokens': '1,100', 'ring_ndr_tokens': '9,900', 'dlrm_tb': '10', 'tp_b': '8', 'dp_a': '1024', 'dp_b': '128', 'pp_c': '4', 'dp_c': '32', 'dp_c_others': '31', 'm_c': '32', 'shard_b_gb': '43.75', 'shard_c_gb': '10.9', 'ar_b_s': '1.75', 'ar_c_s': '0.44', 'eff_b_raw_pct': '54.5', 'eff_b_pct': '83', 'eff_c_pct': '87', 'bubble_c_pct': '8.6', 'bubble_c_fill': '3', 'bubble_c_phase': '35', 'state_b_gb': '350', 'state_c_gb': '87.5', 'zero1_c_gb': '24', 'dp_byte_cut_x': '8', 'l16k_gpus': '16,384', 'l16k_pp': '16', 'l16k_state_gb': '21.9', 'l16k_bubble_pct': '31.9', 'll3_gpus_8k': '8,192', 'll3_gpus_16k': '16,384', 'll3_tp': '8', 'll3_pp': '16', 'll3_cp_long': '16', 'll3_dp_8k': '64', 'll3_dp_16k': '128', 'll3_dp_long': '8', 'll3_seq': '8,192', 'll3_seq_long': '131,072', 'll3_tokens_m': '16', 'll3_tflops_8k': '430', 'll3_tflops_16k': '400', 'll3_tflops_long': '380', 'll3_mfu_8k': '43', 'll3_mfu_16k': '41', 'll3_mfu_long': '38', 'recomp_base': '3', 'recomp_full': '4', 'recomp_pct': '33', 'act_none_tb': '1.1', 'act_full_gb': '19.3', 'act_mb': '4', 'rlhf_policy_b': '70', 'rlhf_critic_b': '13', 'rlhf_policy_gb': '1,120', 'rlhf_ref_gb': '140', 'rlhf_reward_gb': '26', 'rlhf_value_gb': '208', 'rlhf_ppo_gb': '1,494', 'rlhf_dpo_gb': '1,260', 'rlhf_dpo_cut_pct': '15.7', 'rlhf_kv_gb': '85.9', 'rlhf_rollouts': '256', 'rlhf_rollout_tokens': '1,024', 'rlhf_tp': '8', 'rlhf_pp': '4', 'rlhf_dp': '8', 'rlhf_gpus': '256', 'rlhf_policy_per_gpu_gb': '35', 'rlhf_train_infer_word': 'eight', 'fleet_mfu_pct': '41', 'double_mfu_pct': '34', 'gpu_mttf_h': '50,000', 'fail_gpus': '1,024', 'fail_interval_h': '48.8', 'fail_interval_days_word': 'two'}
_NUMS = {'goyal_gpus': 256.0, 'goyal_hours': 1.0, 'goyal_base_hours': 29.0, 'goyal_base_gpus': 8.0, 'goyal_base_batch': 256.0, 'goyal_batch': 8192.0, 'goyal_batch_k': 8.0, 'goyal_servers': 32.0, 'goyal_per_gpu': 32.0, 'goyal_batch_x': 32.0, 'goyal_gpu_x': 32.0, 'goyal_eff_pct': 90.0, 'goyal_eth_gbps': 50.0, 'goyal_epochs': 90.0, 'goyal_warmup_epochs': 5.0, 'goyal_err_base': 23.6, 'goyal_err_nowarm': 24.84, 'goyal_err_warm': 23.74, 'goyal_spread': 0.3, 'goyal_up1_k': 16.0, 'goyal_up1_err': 24.79, 'goyal_up2_k': 32.0, 'goyal_up2_err': 27.55, 'goyal_up3_k': 64.0, 'goyal_up3_err': 33.96, 'params_b': 175.0, 'state_bpp': 16.0, 'opt_bpp': 12.0, 'fp16_bytes': 2.0, 'state_gb': 2800.0, 'state_tb': 2.8, 'hbm_gb': 80.0, 'state_gpus': 35.0, 'ref_gpus': 1024.0, 'ref_nodes': 128.0, 'gpus_per_node': 8.0, 'nvlink_gbs': 450.0, 'ndr_gbs': 50.0, 'nvlink_ndr_x': 9.0, 'grad_gb': 350.0, 'ring_gb': 699.0, 'ar_a_s': 14.0, 'batch_tokens_m': 2.0, 'flops_per_pt': 6.0, 'step_eflops': 2.1, 'local_eff_pct': 50.0, 'fp8_tflops': 1979.0, 't_compute_s': 2.1, 'exposed_a_s': 11.9, 'step_a_s': 14.0, 'eff_a_pct': 15.0, 'resid_pct': 1.0, 'resid_gpus': 512.0, 'resid_share_pct': 84.0, 'gpt2_step_s': 1.8, 'gpt2_step_ms': 1800.0, 'gpt2_gpus_1': 8.0, 'gpt2_sync1_gb': 5.2, 'gpt2_gpus_4': 32.0, 'gpt2_sync4_gb': 5.8, 'eth10_gbs': 1.25, 'eth10_gbps': 10.0, 'gpt2_comm1_ms': 11.7, 'gpt2_eff1_pct': 99.4, 'gpt2_inter_ms': 4650.0, 'gpt2_comm4_ms': 4661.7, 'gpt2_comm4_round_ms': 4662.0, 'gpt2_step4_ms': 6461.7, 'gpt2_step1_ms': 1812.0, 'gpt2_step4_round_ms': 6462.0, 'gpt2_eff4_pct': 27.9, 'gpt2_speedup4': 8.9, 'gpt2_accum': 4.0, 'gpt2_batch': 512.0, 'gpt2_acc_hours': 3.1, 'gpt2_acc_usd': 100.0, 'gpt2_4_usd': 359.0, 'gpt2_4_hours': 2.8, 'node_hour_usd': 32.0, 'eos_n0': 512.0, 'eos_n1': 10752.0, 'eos_t0': 58.0, 'eos_t1': 3.9, 'eos_n_x': 21.0, 'eos_t_x': 14.9, 'eos_eff_pct': 71.0, 'mlperf_min_pct': 71.0, 'mlperf_max_pct': 84.0, 'strag_pct': 1.0, 'strag_on': 0.99, 'strag_n': 1000.0, 'strag_n_plain': 1000.0, 'strag_expected': 10.0, 'strag_node_pct': 92.0, 'strag_100_pct': 37.0, 'strag_1k_pct': 0.004, 'cb_trace_m': 1.6, 'cb_norm': 100.0, 'cb_devices': 64.0, 'cb_local': 4.0, 'cb_accum': 32.0, 'cb_global': 8192.0, 'cb_bcrit': 16000.0, 'cb_per_gpu': 128.0, 'cb_crossover': 125.0, 'cb_params_b': 7.0, 'ssp_base_iters': 100000.0, 'bsp8_workers': 8.0, 'bsp8_iters': 12800.0, 'bsp8_comm_pct': 15.0, 'bsp8_speedup': 6.8, 'bsp8_usd': 26.0, 'bsp64_workers': 64.0, 'bsp64_iters': 1720.0, 'bsp64_comm_pct': 45.0, 'bsp64_speedup': 40.1, 'bsp64_usd': 35.0, 'ssp64_workers': 64.0, 'ssp64_iters': 2100.0, 'ssp64_comm_pct': 25.0, 'ssp64_speedup': 38.1, 'ssp64_usd': 37.0, 'ssp_s': 4.0, 'ssp_batch': 32.0, 'gpu_hour_usd': 2.0, 'wall_growth': 15.0, 'wall_gap': 35.0, 'z7_params_b': 7.0, 'z7_gpus': 64.0, 'a100_gb': 80.0, 'z7_ddp_gb': 112.0, 'z7_opt_gb': 84.0, 'z7_z1_gb': 29.3, 'z7_z2_gb': 15.5, 'z7_z3_gb': 1.75, 'z7_weights_gb': 14.0, 'z3_traffic_x': 1.5, 'z175_gpus': 64.0, 'z175_state_gb': 43.75, 'z175_act_gb': 4.8, 'z175_total_gb': 48.6, 'z175_seq': 2048.0, 'z175_gather_tb': 1.1, 'z175_gather_x': 10.0, 'naive_busy': 2.0, 'naive_slots': 8.0, 'naive_idle': 6.0, 'pipe_p': 4.0, 'pipe_busy': 8.0, 'pipe_slots': 14.0, 'pipe_idle': 6.0, 'pipe_bubble_pct': 43.0, 'pipe_fill': 3.0, 'pipe_phase': 7.0, 'p8m32_p': 8.0, 'p8m32_m': 32.0, 'p8m32_fill': 7.0, 'p8m32_phase': 39.0, 'p16m32_p': 16.0, 'p16m32_m': 32.0, 'p16m32_fill': 15.0, 'p16m32_phase': 47.0, 'p8m8_p': 8.0, 'p8m8_m': 8.0, 'p8m8_fill': 7.0, 'p8m8_phase': 15.0, 'p8m32_bubble_pct': 18.0, 'p16m32_bubble_pct': 31.9, 'p16m32_bubble_round_pct': 32.0, 'p8m8_bubble_pct': 47.0, 'bubble_target_pct': 10.0, 'bubble_m_coef': 9.0, 'p16_m_needed': 135.0, 'tp_degree': 8.0, 'tp_tokens': 2048.0, 'tp_compute_ms': 1.9, 'tp_nvlink_ms': 0.4, 'tp_ndr_ms': 3.5, 'gpt3_layers': 96.0, 'meg_2_pct': 95.0, 'meg_4_pct': 82.0, 'meg_8_pct': 77.0, 'meg_top_pct': 74.0, 'meg_gpus': 512.0, 'meg_servers': 64.0, 'meg_base_b': 1.2, 'meg_top_b': 8.3, 'nvswitch_gbs': 300.0, 'ring_nvlink_tokens': 1100.0, 'ring_ndr_tokens': 9900.0, 'dlrm_tb': 10.0, 'tp_b': 8.0, 'dp_a': 1024.0, 'dp_b': 128.0, 'pp_c': 4.0, 'dp_c': 32.0, 'dp_c_others': 31.0, 'm_c': 32.0, 'shard_b_gb': 43.75, 'shard_c_gb': 10.9, 'ar_b_s': 1.75, 'ar_c_s': 0.44, 'eff_b_raw_pct': 54.5, 'eff_b_pct': 83.0, 'eff_c_pct': 87.0, 'bubble_c_pct': 8.6, 'bubble_c_fill': 3.0, 'bubble_c_phase': 35.0, 'state_b_gb': 350.0, 'state_c_gb': 87.5, 'zero1_c_gb': 24.0, 'dp_byte_cut_x': 8.0, 'l16k_gpus': 16384.0, 'l16k_pp': 16.0, 'l16k_state_gb': 21.9, 'l16k_bubble_pct': 31.9, 'll3_gpus_8k': 8192.0, 'll3_gpus_16k': 16384.0, 'll3_tp': 8.0, 'll3_pp': 16.0, 'll3_cp_long': 16.0, 'll3_dp_8k': 64.0, 'll3_dp_16k': 128.0, 'll3_dp_long': 8.0, 'll3_seq': 8192.0, 'll3_seq_long': 131072.0, 'll3_tokens_m': 16.0, 'll3_tflops_8k': 430.0, 'll3_tflops_16k': 400.0, 'll3_tflops_long': 380.0, 'll3_mfu_8k': 43.0, 'll3_mfu_16k': 41.0, 'll3_mfu_long': 38.0, 'recomp_base': 3.0, 'recomp_full': 4.0, 'recomp_pct': 33.0, 'act_none_tb': 1.1, 'act_full_gb': 19.3, 'act_mb': 4.0, 'rlhf_policy_b': 70.0, 'rlhf_critic_b': 13.0, 'rlhf_policy_gb': 1120.0, 'rlhf_ref_gb': 140.0, 'rlhf_reward_gb': 26.0, 'rlhf_value_gb': 208.0, 'rlhf_ppo_gb': 1494.0, 'rlhf_dpo_gb': 1260.0, 'rlhf_dpo_cut_pct': 15.7, 'rlhf_kv_gb': 85.9, 'rlhf_rollouts': 256.0, 'rlhf_rollout_tokens': 1024.0, 'rlhf_tp': 8.0, 'rlhf_pp': 4.0, 'rlhf_dp': 8.0, 'rlhf_gpus': 256.0, 'rlhf_policy_per_gpu_gb': 35.0, 'fleet_mfu_pct': 41.0, 'double_mfu_pct': 34.0, 'gpu_mttf_h': 50000.0, 'fail_gpus': 1024.0, 'fail_interval_h': 48.8}
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'assets' / 'kit'))
sys.path.insert(0, str(HERE))
import slideplot as sp
from mlsysim.physics import calc_all_up_availability, calc_pipeline_bubble
plt = sp.plt
C = sp.C
out = HERE
V = {'c1': '#1F407A', 'c1_l': '#EAF1F8', 'c2': '#d97706', 'c2_l': '#fef3c7', 'c3': '#7c3aed', 'c3_l': '#f3e8ff', 'ok': '#16a34a', 'ok_l': '#dcfce7', 'bad': '#dc2626', 'bad_l': '#fee2e2', 'frame': '#94a3b8', 'frame_l': '#f8fafc', 'text': '#0f172a', 'gray': '#64748b'}

def label(ax, x, y, text, color, shown=True, **kw):
    """Text drawn in every step, invisible until shown, so steps align."""
    kw.setdefault('fontsize', 12)
    kw.setdefault('fontweight', 'bold')
    return ax.text(x, y, text, color=color, alpha=1.0 if shown else 0.0, **kw)

class SVG:

    def __init__(self, w, h):
        self.w, self.h, self.items = (w, h, [])

    def rect(self, x, y, w, h, fill, stroke='none', sw=2, rx=4, op=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>')

    def text(self, x, y, s, size=22, color=V['text'], weight='bold', anchor='middle', op=1.0):
        s = s.replace('&', '&amp;').replace('<', '&lt;')
        self.items.append(f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" dominant-baseline="middle" opacity="{op}">{s}</text>')

    def line(self, x1, y1, x2, y2, color, sw=3, arrow=False, dash=None, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        m = f' marker-end="url(#a{color[1:]})"' if arrow else ''
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" opacity="{op}"{d}{m}/>')

    def save(self, path):
        cols = {V[k] for k in V} | {'#000000'}
        defs = ''.join((f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in cols))
        body = '\n'.join(self.items)
        Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}"><defs>{defs}</defs><rect width="{self.w}" height="{self.h}" fill="white"/>\n{body}\n</svg>\n')
k8 = '8k'
rows = [('8 GPUs, batch 256', 'goyal_err_base', 'neutral'), (f'256 GPUs, batch {k8}, no warmup', 'goyal_err_nowarm', 'danger'), (f'256 GPUs, batch {k8}, gradual warmup', 'goyal_err_warm', 'data')]

def hook_warmup(path, k, size, fs, names):
    fig, ax = plt.subplots(figsize=size)
    for i, (name, key, ck) in enumerate(rows):
        y = 2 - i
        shown = i < k
        v = {'8 GPUs, batch 256': 23.6, '256 GPUs, batch 8k, no warmup': 24.84, '256 GPUs, batch 8k, gradual warmup': 23.74}[name]
        ax.barh(y, v - 23.0, left=23.0, color=C[ck], height=0.6, alpha=1 if shown else 0)
        ax.text(22.97, y, names[i], ha='right', va='center', fontsize=fs, fontweight='bold', color=C['text'])
        label(ax, v + 0.04, y, f"{ {'8 GPUs, batch 256': '23.60', '256 GPUs, batch 8k, no warmup': '24.84', '256 GPUs, batch 8k, gradual warmup': '23.74'}[name]}%", C[ck], shown, va='center', fontsize=fs + 0.5)
        if not shown:
            ax.text(23.1, y, '?', va='center', fontsize=fs + 3, fontweight='bold', color=C['neutral'])
    ax.set_xlim(23.0, 25.6)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([23, 24, 25])
    ax.tick_params(labelsize=fs - 1)
    ax.set_xlabel('top-1 validation error (%)', fontsize=fs - 0.5)
    ax.grid(axis='x', color=C['grid'], lw=0.6)
    ax.set_axisbelow(True)
    sp.save(fig, path)
for k in (1, 3):
    hook_warmup(out / f'hook_warmup_step{k}.svg', k, (5.8, 1.6), 12.5, [r[0] for r in rows])
hook_warmup(out / 'hook_warmup_col.svg', 3, (5.4, 3.0), 15, ['batch 256', f'{k8}, no warmup', f'{k8}, warmup'])
for k in (1, 2, 3):
    s = SVG(900, 330)
    xs = [40, 255, 470, 685]
    for i, x in enumerate(xs):
        s.rect(x, 20, 175, 70, V['frame_l'], V['frame'], rx=6)
        s.text(x + 87, 42, f'GPU {i}', 24, V['text'])
        s.text(x + 87, 70, 'full model copy', 21, V['gray'], weight='normal')
        s.rect(x + 10, 108, 155, 42, V['ok_l'], V['ok'], rx=4)
        s.text(x + 87, 129, f'batch slice {i}', 21, V['ok'])
        op = 1 if k >= 2 else 0
        s.rect(x + 10, 172, 155, 42, V['c1_l'], V['c1'], rx=4, op=op)
        s.text(x + 87, 193, f'gradient g{i}', 21, V['c1'], op=op)
        s.line(x + 87, 150, x + 87, 168, V['c1'], 3, arrow=True, op=op)
        op3 = 1 if k >= 3 else 0
        s.line(x + 87, 214, x + 87, 240, V['c2'], 3, arrow=True, op=op3)
    op3 = 1 if k >= 3 else 0
    s.rect(40, 244, 820, 40, V['c2_l'], V['c2'], rx=6, op=op3)
    s.text(450, 264, 'AllReduce: every GPU receives g = (g0 + g1 + g2 + g3) / 4', 24, '#b45309', op=op3)
    s.text(450, 311, 'Each GPU applies the same update to its own copy', 24, V['text'], op=op3)
    s.save(out / f'dp_flow_step{k}.svg')
for k in (1, 2):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 2.3), gridspec_kw={'height_ratios': [2, 1.25]})
    for y, comp, name in [(1, 4.0, 'Few GPUs'), (0, 1.0, '4× the GPUs')]:
        a1.barh(y, comp, color=V['c1'], height=0.6)
        a1.barh(y, 1.6, left=comp, color=V['c2'], height=0.6)
        a1.text(-0.08, y, name, ha='right', va='center', fontsize=13.5, fontweight='bold')
    a1.text(2.0, 1, 'compute', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    a1.text(4.8, 1, 'comm.', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    a1.text(1.8, 0, 'comm.', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    a1.text(2.8, 0, 'same comm., now longer than compute', va='center', fontsize=12.5, fontweight='bold', color=V['c2'])
    a1.text(8.3, 1.55, 'illustrative', ha='right', fontsize=10, color=C['neutral'], style='italic')
    a1.set_xlim(0, 8.3)
    a1.set_ylim(-0.5, 1.7)
    a1.axis('off')
    sh = k >= 2
    tc, te = (2.1, 11.9)
    a2.barh(0, tc, color=V['c1'], height=0.6, alpha=1 if sh else 0)
    a2.barh(0, te, left=tc, color=V['c2'], height=0.6, alpha=1 if sh else 0)
    a2.text(-0.15, 0, '175B, 1,024 GPUs', ha='right', va='center', fontsize=13.5, fontweight='bold', alpha=1 if sh else 0)
    label(a2, tc / 2, 0, '2.1 s', 'white', sh, ha='center', va='center', fontsize=11)
    label(a2, 8.0, 0, '11.9 s exposed AllReduce', 'white', sh, ha='center', va='center', fontsize=13)
    label(a2, tc + te + 0.2, 0, '15%', C['danger'], sh, va='center', fontsize=15)
    a2.set_xlim(0, 16.5)
    a2.set_ylim(-0.5, 0.5)
    a2.set_yticks([])
    a2.spines['left'].set_visible(False)
    a2.set_xlabel('seconds per step', alpha=1 if sh else 0)
    for t in a2.get_xticklabels():
        t.set_alpha(1 if sh else 0)
    a2.tick_params(axis='x', colors=C['text'] if sh else 'white')
    a2.spines['bottom'].set_alpha(1 if sh else 0)
    fig.subplots_adjust(hspace=0.35)
    sp.save(fig, out / f'eta_bars_step{k}.svg')
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    data = [('8 GPUs, one node\n(NVLink)', 1800.0, 11.7, '99.4%', '1,812'), ('32 GPUs, four nodes\n(10 Gb Ethernet)', 1800.0, 4661.7, '27.9%', '6,462')]
    for i, (name, comp, comm, eff, total) in enumerate(data):
        y = 1 - i
        shown = i == 0 or k >= 2
        a = 1 if shown else 0
        ax.barh(y, comp, color=V['c1'], height=0.6, alpha=a)
        ax.barh(y, comm, left=comp, color=V['c2'], height=0.6, alpha=a)
        ax.text(-60, y, name, ha='right', va='center', fontsize=12, fontweight='bold')
        label(ax, comp + comm + 80, y, f'{total} ms   {eff}', C['data'] if i == 0 else C['danger'], shown, va='center', fontsize=12.5)
    label(ax, 900, 1, 'compute', 'white', True, ha='center', va='center', fontsize=11)
    label(ax, 900, 0, 'compute', 'white', k >= 2, ha='center', va='center', fontsize=11)
    label(ax, 4100, 0, 'communication 4,662 ms', 'white', k >= 2, ha='center', va='center', fontsize=11)
    ax.set_xlim(0, 8500)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('milliseconds per step')
    sp.save(fig, out / f'gpt2_step{k}.svg')
names_ml = [('GPT-3, H100 (NVIDIA Eos)', C['memory']), ('GPT-3, TPU v5p', C['data']), ('GPT-3, TPU Trillium', C['compute']), ('Llama 3.1 405B, H100', C['network']), ('Llama 3.1 405B, GB200', C['danger'])]
ml = SimpleNamespace(efficiencies=[0.7078628494, 0.838517763, 0.7835967169, 0.7336626043, 0.7991552784], labels=['GPT-3, NVIDIA H100 (v3.1)', 'GPT-3, Google TPU v5p (v4.1)', 'GPT-3, Google Trillium (v4.1)', 'Llama 3.1 405B, NVIDIA H100 (v5.0)', 'Llama 3.1 405B, GB200 (v5.0)'], series=[[{'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos_n64_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 512.0, 'runs': 3.0, 'min_minutes': 58.247, 'mean_minutes': 58.301, 'score_minutes': 58.301, 'global_batch_size': 2048.0, 'trained_samples_median': 1207959552.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos_n96_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 768.0, 'runs': 3.0, 'min_minutes': 40.596, 'mean_minutes': 40.616, 'score_minutes': 40.616, 'global_batch_size': 3072.0, 'trained_samples_median': 1258291200.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n512_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 4096.0, 'runs': 3.0, 'min_minutes': 8.566, 'mean_minutes': 8.576, 'score_minutes': 8.576, 'global_batch_size': 2048.0, 'trained_samples_median': 1207959552.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n768_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 6144.0, 'runs': 3.0, 'min_minutes': 6.017, 'mean_minutes': 6.025, 'score_minutes': 6.025, 'global_batch_size': 3072.0, 'trained_samples_median': 1258291200.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n1024_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 8192.0, 'runs': 3.0, 'min_minutes': 4.864, 'mean_minutes': 4.927, 'score_minutes': 4.927, 'global_batch_size': 4096.0, 'trained_samples_median': 1308622848.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n1280_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 10240.0, 'runs': 3.0, 'min_minutes': 4.07, 'mean_minutes': 4.071, 'score_minutes': 4.071, 'global_batch_size': 2560.0, 'trained_samples_median': 1205862400.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n1344_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 10752.0, 'runs': 3.0, 'min_minutes': 3.916, 'mean_minutes': 3.922, 'score_minutes': 3.922, 'global_batch_size': 2688.0, 'trained_samples_median': 1211105280.0}], [{'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-v5p-4096-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-v5p', 'accelerators': 2048.0, 'runs': 3.0, 'min_minutes': 29.503, 'mean_minutes': 29.598, 'score_minutes': 29.598, 'global_batch_size': 4096.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-v5p-8192-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-v5p', 'accelerators': 4096.0, 'runs': 3.0, 'min_minutes': 16.243, 'mean_minutes': 16.26, 'score_minutes': 16.26, 'global_batch_size': 4096.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-v5p-12288-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-v5p', 'accelerators': 6144.0, 'runs': 3.0, 'min_minutes': 11.753, 'mean_minutes': 11.766, 'score_minutes': 11.766, 'global_batch_size': 6144.0, 'trained_samples_median': None}], [{'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-trillium-512-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-trillium', 'accelerators': 512.0, 'runs': 3.0, 'min_minutes': 101.945, 'mean_minutes': 102.344, 'score_minutes': 102.344, 'global_batch_size': 1536.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-trillium-1024-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-trillium', 'accelerators': 1024.0, 'runs': 3.0, 'min_minutes': 49.745, 'mean_minutes': 49.853, 'score_minutes': 49.853, 'global_batch_size': 2048.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-trillium-2048-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-trillium', 'accelerators': 2048.0, 'runs': 3.0, 'min_minutes': 27.281, 'mean_minutes': 27.317, 'score_minutes': 27.317, 'global_batch_size': 4096.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-trillium-3072-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-trillium', 'accelerators': 3072.0, 'runs': 3.0, 'min_minutes': 20.612, 'mean_minutes': 21.768, 'score_minutes': 21.768, 'global_batch_size': 6144.0, 'trained_samples_median': None}], [{'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n64_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 512.0, 'runs': 3.0, 'min_minutes': 230.617, 'mean_minutes': 230.687, 'score_minutes': 230.687, 'global_batch_size': 1152.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n256_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 2048.0, 'runs': 3.0, 'min_minutes': 59.906, 'mean_minutes': 63.345, 'score_minutes': 63.345, 'global_batch_size': 1152.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n304_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 2432.0, 'runs': 3.0, 'min_minutes': 59.523, 'mean_minutes': 60.472, 'score_minutes': 60.472, 'global_batch_size': 1216.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n320_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 2560.0, 'runs': 3.0, 'min_minutes': 49.697, 'mean_minutes': 50.343, 'score_minutes': 50.343, 'global_batch_size': 1120.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n1024_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 8192.0, 'runs': 3.0, 'min_minutes': 19.636, 'mean_minutes': 19.652, 'score_minutes': 19.652, 'global_batch_size': 1536.0, 'trained_samples_median': None}], [{'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu512_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 512.0, 'runs': 3.0, 'min_minutes': 104.739, 'mean_minutes': 105.107, 'score_minutes': 105.107, 'global_batch_size': 896.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu1024_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 1024.0, 'runs': 3.0, 'min_minutes': 53.181, 'mean_minutes': 53.194, 'score_minutes': 53.194, 'global_batch_size': 768.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu1536_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 1536.0, 'runs': 3.0, 'min_minutes': 35.801, 'mean_minutes': 37.987, 'score_minutes': 37.987, 'global_batch_size': 1152.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu2048_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 2048.0, 'runs': 3.0, 'min_minutes': 31.962, 'mean_minutes': 32.013, 'score_minutes': 32.013, 'global_batch_size': 1280.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu2496_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 2496.0, 'runs': 3.0, 'min_minutes': 26.926, 'mean_minutes': 26.979, 'score_minutes': 26.979, 'global_batch_size': 1248.0, 'trained_samples_median': None}]])
fig, ax = plt.subplots(figsize=(5.8, 2.4))
for (name, col), pts, eff, (_, (_, _, _, acc)) in zip(names_ml, [[{'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos_n64_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 512.0, 'runs': 3.0, 'min_minutes': 58.247, 'mean_minutes': 58.301, 'score_minutes': 58.301, 'global_batch_size': 2048.0, 'trained_samples_median': 1207959552.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos_n96_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 768.0, 'runs': 3.0, 'min_minutes': 40.596, 'mean_minutes': 40.616, 'score_minutes': 40.616, 'global_batch_size': 3072.0, 'trained_samples_median': 1258291200.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n512_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 4096.0, 'runs': 3.0, 'min_minutes': 8.566, 'mean_minutes': 8.576, 'score_minutes': 8.576, 'global_batch_size': 2048.0, 'trained_samples_median': 1207959552.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n768_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 6144.0, 'runs': 3.0, 'min_minutes': 6.017, 'mean_minutes': 6.025, 'score_minutes': 6.025, 'global_batch_size': 3072.0, 'trained_samples_median': 1258291200.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n1024_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 8192.0, 'runs': 3.0, 'min_minutes': 4.864, 'mean_minutes': 4.927, 'score_minutes': 4.927, 'global_batch_size': 4096.0, 'trained_samples_median': 1308622848.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n1280_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 10240.0, 'runs': 3.0, 'min_minutes': 4.07, 'mean_minutes': 4.071, 'score_minutes': 4.071, 'global_batch_size': 2560.0, 'trained_samples_median': 1205862400.0}, {'round': 'v3.1', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n1344_ngc23.09_nemo', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 10752.0, 'runs': 3.0, 'min_minutes': 3.916, 'mean_minutes': 3.922, 'score_minutes': 3.922, 'global_batch_size': 2688.0, 'trained_samples_median': 1211105280.0}], [{'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-v5p-4096-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-v5p', 'accelerators': 2048.0, 'runs': 3.0, 'min_minutes': 29.503, 'mean_minutes': 29.598, 'score_minutes': 29.598, 'global_batch_size': 4096.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-v5p-8192-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-v5p', 'accelerators': 4096.0, 'runs': 3.0, 'min_minutes': 16.243, 'mean_minutes': 16.26, 'score_minutes': 16.26, 'global_batch_size': 4096.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-v5p-12288-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-v5p', 'accelerators': 6144.0, 'runs': 3.0, 'min_minutes': 11.753, 'mean_minutes': 11.766, 'score_minutes': 11.766, 'global_batch_size': 6144.0, 'trained_samples_median': None}], [{'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-trillium-512-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-trillium', 'accelerators': 512.0, 'runs': 3.0, 'min_minutes': 101.945, 'mean_minutes': 102.344, 'score_minutes': 102.344, 'global_batch_size': 1536.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-trillium-1024-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-trillium', 'accelerators': 1024.0, 'runs': 3.0, 'min_minutes': 49.745, 'mean_minutes': 49.853, 'score_minutes': 49.853, 'global_batch_size': 2048.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-trillium-2048-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-trillium', 'accelerators': 2048.0, 'runs': 3.0, 'min_minutes': 27.281, 'mean_minutes': 27.317, 'score_minutes': 27.317, 'global_batch_size': 4096.0, 'trained_samples_median': None}, {'round': 'v4.1', 'submitter': 'Google', 'system': 'cloud-tpu-trillium-3072-maxtext', 'division': 'closed', 'benchmark': 'gpt3', 'accelerator': 'TPU-trillium', 'accelerators': 3072.0, 'runs': 3.0, 'min_minutes': 20.612, 'mean_minutes': 21.768, 'score_minutes': 21.768, 'global_batch_size': 6144.0, 'trained_samples_median': None}], [{'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n64_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 512.0, 'runs': 3.0, 'min_minutes': 230.617, 'mean_minutes': 230.687, 'score_minutes': 230.687, 'global_batch_size': 1152.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n256_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 2048.0, 'runs': 3.0, 'min_minutes': 59.906, 'mean_minutes': 63.345, 'score_minutes': 63.345, 'global_batch_size': 1152.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n304_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 2432.0, 'runs': 3.0, 'min_minutes': 59.523, 'mean_minutes': 60.472, 'score_minutes': 60.472, 'global_batch_size': 1216.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n320_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 2560.0, 'runs': 3.0, 'min_minutes': 49.697, 'mean_minutes': 50.343, 'score_minutes': 50.343, 'global_batch_size': 1120.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'NVIDIA', 'system': 'eos-dfw_n1024_ngc25.01_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA H100-SXM5-80GB', 'accelerators': 8192.0, 'runs': 3.0, 'min_minutes': 19.636, 'mean_minutes': 19.652, 'score_minutes': 19.652, 'global_batch_size': 1536.0, 'trained_samples_median': None}], [{'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu512_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 512.0, 'runs': 3.0, 'min_minutes': 104.739, 'mean_minutes': 105.107, 'score_minutes': 105.107, 'global_batch_size': 896.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu1024_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 1024.0, 'runs': 3.0, 'min_minutes': 53.181, 'mean_minutes': 53.194, 'score_minutes': 53.194, 'global_batch_size': 768.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu1536_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 1536.0, 'runs': 3.0, 'min_minutes': 35.801, 'mean_minutes': 37.987, 'score_minutes': 37.987, 'global_batch_size': 1152.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu2048_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 2048.0, 'runs': 3.0, 'min_minutes': 31.962, 'mean_minutes': 32.013, 'score_minutes': 32.013, 'global_batch_size': 1280.0, 'trained_samples_median': None}, {'round': 'v5.0', 'submitter': 'IBM+CoreWeave+NVIDIA', 'system': 'carina_ngpu2496_ngc25.04_nemo', 'division': 'closed', 'benchmark': 'llama31_405b', 'accelerator': 'NVIDIA Blackwell GPU (GB200)', 'accelerators': 2496.0, 'runs': 3.0, 'min_minutes': 26.926, 'mean_minutes': 26.979, 'score_minutes': 26.979, 'global_batch_size': 1248.0, 'trained_samples_median': None}]], [0.7078628494, 0.838517763, 0.7835967169, 0.7336626043, 0.7991552784], [('GPT-3, NVIDIA H100 (v3.1)', ('v3.1', 'NVIDIA', 'gpt3', 'H100-SXM5')), ('GPT-3, Google TPU v5p (v4.1)', ('v4.1', 'Google', 'gpt3', 'TPU-v5p')), ('GPT-3, Google Trillium (v4.1)', ('v4.1', 'Google', 'gpt3', 'TPU-trillium')), ('Llama 3.1 405B, NVIDIA H100 (v5.0)', ('v5.0', 'NVIDIA', 'llama31_405b', 'H100-SXM5')), ('Llama 3.1 405B, GB200 (v5.0)', ('v5.0', 'IBM+CoreWeave', 'llama31_405b', 'Blackwell'))]):
    n = [r['accelerators'] for r in pts]
    t = [r['score_minutes'] for r in pts]
    ax.plot(n, t, '-o', ms=5, lw=2.2, color=col, zorder=3)
    ax.plot([n[0], n[-1]], [t[0], t[0] * n[0] / n[-1]], '--', lw=1.2, color=col, alpha=0.7)
    dy = -11 if acc == 'TPU-trillium' else 5 if acc == 'Blackwell' else 0
    ax.annotate(f'{eff:.0%}', (n[-1], t[-1]), xytext=(6, dy), textcoords='offset points', fontsize=11, fontweight='bold', color=col, va='center')
    ax.plot([], [], '-o', color=col, lw=2.2, ms=5, label=name)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='lower left', fontsize=9.5, frameon=False, handlelength=1.6)
ax.set_xlim(150, 16000)
ax.set_ylim(1.2, 300)
ax.set_xticks([512, 2048, 8192])
ax.set_xticklabels(['512', '2,048', '8,192'])
ax.set_yticks([5, 10, 30, 100])
ax.set_yticklabels(['5', '10', '30', '100'])
ax.minorticks_off()
ax.set_xlabel('accelerators (log)')
ax.set_ylabel('minutes to train (log)')
ax.text(15500, 250, "dashed: ideal from each series' first point;\n% = efficiency of the largest run", ha='right', va='top', fontsize=9.5, color=C['neutral'])
sp.save(fig, out / 'mlperf_scaling.svg')
fig, ax = plt.subplots(figsize=(5.4, 3.4))
N = np.logspace(0, 4, 200)
ax.plot(N, np.array([99.0, 98.95287797, 98.90354777, 98.85190713, 98.79784913, 98.74126204, 98.68202913, 98.6200284, 98.55513241, 98.48720801, 98.41611617, 98.34171167, 98.26384287, 98.18235146, 98.09707219, 98.00783258, 97.91445266, 97.81674463, 97.71451262, 97.60755234, 97.4956508, 97.37858592, 97.25612627, 97.12803072, 96.99404805, 96.85391666, 96.70736419, 96.55410714, 96.39385057, 96.22628768, 96.05109947, 95.8679544, 95.67650799, 95.47640249, 95.26726651, 95.04871473, 94.82034751, 94.58175058, 94.3324948, 94.0721358, 93.80021379, 93.5162533, 93.21976298, 92.91023545, 92.58714721, 92.24995853, 91.89811347, 91.53103992, 91.14814972, 90.74883885, 90.33248774, 89.89846164, 89.44611107, 88.97477247, 88.48376893, 87.972411, 87.4399978, 86.88581815, 86.30915193, 85.7092717, 85.08544445, 84.43693357, 83.76300115, 83.06291045, 82.33592869, 81.58133013, 80.79839946, 79.98643549, 79.14475526, 78.27269839, 77.36963192, 76.43495543, 75.46810669, 74.46856752, 73.43587025, 72.36960447, 71.26942421, 70.13505549, 68.96630436, 67.7630651, 66.52532895, 65.25319297, 63.94686922, 62.60669411, 61.23313786, 59.82681393, 58.38848845, 56.91908941, 55.41971558, 53.89164496, 52.33634257, 50.75546753, 49.1508791, 47.52464158, 45.87902776, 44.21652082, 42.5398143, 40.85181002, 39.15561358, 37.4545274, 35.75204089, 34.05181756, 32.3576791, 30.67358598, 29.00361472, 27.35193161, 25.7227631, 24.1203627, 22.54897482, 21.01279567, 19.51593165, 18.06235575, 16.65586252, 15.30002228, 13.99813553, 12.75318831, 11.56780957, 10.44423164, 9.384254861, 8.38921741, 7.459971496, 6.596866805, 5.799742112, 5.067925731, 4.400245296, 3.795047093, 3.250224883, 2.763257853, 2.33125696, 1.951018669, 1.619084718, 1.331806301, 1.085410839, 0.876069348, 0.6999623588, 0.5533423865, 0.432591055, 0.3342692332, 0.2551588456, 0.1922954107, 0.1429907975, 0.1048461477, 0.07575536124, 0.05389995255, 0.03773642806, 0.02597759097, 0.01756933013, 0.0116644935, 0.007595384965, 0.004846271436, 0.003027064153, 0.00184907089, 0.001103430291, 0.0006425623518, 0.0003647216641, 0.0002015379314, 0.0001082802036, 5.648853297e-05, 2.857492143e-05, 1.399552454e-05, 6.62687125e-06, 3.028638732e-06, 1.333759154e-06, 5.649828421e-07, 2.297850464e-07, 8.955705784e-08, 3.338048645e-08, 1.187358865e-08, 4.021666336e-09, 1.294068944e-09, 3.946217886e-10, 1.137553726e-10, 3.091523711e-11, 7.898976534e-12, 1.89189482e-12, 4.23468661e-13, 8.829800534e-14, 1.709334542e-14, 3.061413047e-15, 5.053999814e-16, 7.661095755e-17, 1.062021926e-17, 1.340677708e-18, 1.534397561e-19, 1.584736598e-20, 1.469838028e-21, 1.218042752e-22, 8.970546295e-24, 5.838636412e-25, 3.338858753e-26, 1.667309459e-27, 7.223992402e-29, 2.697494045e-30, 8.619989674e-32, 2.339979268e-33, 5.354519265e-35, 1.024512838e-36, 1.625254964e-38, 2.118737355e-40, 2.24877485e-42]), color=V['bad'], lw=3)
for n_, lab, dy in [(8.0, '8 GPUs: 92%', 12), (100, '100: 37%', 12), (1000.0, '1,000: 0.004%', 14)]:
    v = {'8 GPUs: 92%': 92.27446944, '100: 37%': 36.60323413, '1,000: 0.004%': 0.004317124741}[lab]
    ax.plot([n_], [max(v, 0.4)], 'o', ms=9, color=V['bad'], mec='white', mew=1.5)
    ax.annotate(lab, (n_, max(v, 0.4)), xytext=(8, dy), textcoords='offset points', fontsize=15, fontweight='bold', color=V['text'])
ax.set_xscale('log')
ax.set_xlim(1, 10000)
ax.set_ylim(0, 110)
ax.set_xlabel('GPUs in the synchronous group', fontsize=15)
ax.set_ylabel('% of steps with\nno late GPU', fontsize=15)
ax.tick_params(labelsize=14)
sp.save(fig, out / 'straggler.svg')

def sync_panel(s, x0, title, kind, op):
    unit = 24
    s.text(x0 + 135, 22, title, 24, V['text'], op=op)
    for w in range(4):
        y = 50 + w * 52
        s.text(x0 - 2, y + 18, f'W{w}', 20, V['gray'], anchor='end', op=op)
        slow = w == 3
        t = 0.0
        it = 0
        while t < 10.0 and it < 6:
            dur = 2.6 if slow else 1.6
            if t + dur > 10.0:
                break
            s.rect(x0 + 8 + t * unit, y, dur * unit - 3, 36, V['bad_l'] if slow else V['c1_l'], V['bad'] if slow else V['c1'], 2, 3, op)
            t += dur
            it += 1
            if kind == 'bsp' and (not slow):
                wait = 1.0
                s.rect(x0 + 8 + t * unit, y + 12, wait * unit - 3, 12, '#e2e8f0', 'none', 0, 2, op)
                t += wait
            elif kind == 'ssp' and (not slow) and (it % 2 == 0):
                s.rect(x0 + 8 + t * unit, y + 12, 1.0 * unit - 3, 12, '#e2e8f0', 'none', 0, 2, op)
                t += 1.0
    if kind == 'bsp':
        for b in (2.6, 5.2, 7.8):
            s.line(x0 + 8 + b * unit, 44, x0 + 8 + b * unit, 252, V['c3'], 2.5, dash='6,4', op=op)
    s.text(x0 + 135, 278, {'bsp': 'barrier every step', 'ssp': 'lead of at most s steps', 'asp': 'no barrier; stale updates'}[kind], 20, V['gray'], weight='normal', op=op)
for k in (1, 2, 3):
    s = SVG(900, 295)
    for i, (x0, title, kind) in enumerate([(40, 'BSP', 'bsp'), (340, 'SSP (s = 1)', 'ssp'), (640, 'ASP', 'asp')]):
        sync_panel(s, x0, title, kind, 1.0 if i < k else 0.0)
    s.save(out / f'sync_step{k}.svg')
fig, ax = plt.subplots(figsize=(5.4, 3.4))
r = np.logspace(-2, 2, 300)
ax.plot(r, 1 + 1 / r, color=V['c1'], lw=3)
ax.plot(r, 1 + r, color=V['bad'], lw=3)
ax.text(12, 1.4, 'optimizer steps', color=V['c1'], fontsize=15, fontweight='bold')
ax.text(4, 25, 'samples processed', color=V['bad'], fontsize=15, fontweight='bold', ha='right')
ax.axvline(1, color=V['gray'], lw=1.5, ls='--')
ax.plot([1], [2], 'o', ms=10, color=V['text'])
ax.annotate('at $B = B_{\\mathrm{crit}}$, both are' + '\n2× their minimum', (1, 2), xytext=(5, 2.3), fontsize=13, fontweight='bold', color=V['text'], arrowprops=dict(arrowstyle='->', color=V['text'], lw=1.2))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.01, 100)
ax.set_ylim(0.8, 120)
ax.set_xticks([0.01, 0.1, 1, 10, 100])
ax.set_xticklabels(['0.01', '0.1', '1', '10', '100'])
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(['1', '10', '100'])
ax.minorticks_off()
ax.set_xlabel('batch $B$ / critical batch $B_{\\mathrm{crit}}$', fontsize=15)
ax.set_ylabel('cost / minimum cost', fontsize=15)
ax.tick_params(labelsize=14)
sp.save(fig, out / 'critical_batch.svg')
sweep = sorted(((r['global_batch'], r['top1_error_pct']) for r in ({'source': 'table1', 'global_batch': 256.0, 'warmup': 'none', 'top1_error_pct': 23.6}, {'source': 'table1', 'global_batch': 8192.0, 'warmup': 'none', 'top1_error_pct': 24.84}, {'source': 'table1', 'global_batch': 8192.0, 'warmup': 'constant', 'top1_error_pct': 25.88}, {'source': 'table1', 'global_batch': 8192.0, 'warmup': 'gradual', 'top1_error_pct': 23.74}, {'source': 'figure3', 'global_batch': 128.0, 'warmup': 'gradual', 'top1_error_pct': 23.49}, {'source': 'figure3', 'global_batch': 256.0, 'warmup': 'gradual', 'top1_error_pct': 23.6}, {'source': 'figure3', 'global_batch': 512.0, 'warmup': 'gradual', 'top1_error_pct': 23.48}, {'source': 'figure3', 'global_batch': 1024.0, 'warmup': 'gradual', 'top1_error_pct': 23.53}, {'source': 'figure3', 'global_batch': 2048.0, 'warmup': 'gradual', 'top1_error_pct': 23.49}, {'source': 'figure3', 'global_batch': 4096.0, 'warmup': 'gradual', 'top1_error_pct': 23.56}, {'source': 'figure3', 'global_batch': 8192.0, 'warmup': 'gradual', 'top1_error_pct': 23.74}, {'source': 'figure3', 'global_batch': 16384.0, 'warmup': 'gradual', 'top1_error_pct': 24.79}, {'source': 'figure3', 'global_batch': 32768.0, 'warmup': 'gradual', 'top1_error_pct': 27.55}, {'source': 'figure3', 'global_batch': 65536.0, 'warmup': 'gradual', 'top1_error_pct': 33.96}) if r['source'] == 'figure3'))
bs = [b for b, _ in sweep]
err = [e for _, e in sweep]
fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.axvspan(11000, 90000, color=C['danger_l'], zorder=0)
ax.plot(bs, err, '-o', color=C['memory'], lw=3, ms=8, zorder=3)
ax.annotate(f'{k8}: 23.74%', (8192.0, 23.74), xytext=(-110, 22), textcoords='offset points', fontsize=15, fontweight='bold', color=C['memory'])
ax.annotate('64k: 33.96%', (bs[-1], err[-1]), xytext=(-125, -6), textcoords='offset points', fontsize=15, fontweight='bold', color=C['danger'])
ax.text(14000, 31, f'past\n{k8}', fontsize=15, fontweight='bold', color=C['danger'])
ax.set_xscale('log', base=2)
ax.set_xticks([128, 512, 2048, 8192, 32768])
ax.set_xticklabels(['128', '512', '2k', '8k', '32k'])
ax.minorticks_off()
ax.set_ylim(22, 36)
ax.set_xlabel('global batch (images)', fontsize=15)
ax.set_ylabel('top-1 val. error (%)', fontsize=15)
ax.tick_params(labelsize=14)
sp.save(fig, out / 'goyal_batch.svg')
deck_label = {'AlexNet': 'AlexNet', 'VGG-16': '', 'BERT-Large': '', 'GPT-2': 'GPT-2', 'GPT-3': 'GPT-3', 'PaLM': 'PaLM', 'Llama 3 (405B)': 'Llama 3 405B'}
wall_rows = [{'Year': 2012, 'Date': '2012-09-30', 'Name': 'AlexNet', 'Type': 'Model', 'Memory_GB': 0.96}, {'Year': 2014, 'Date': '2014-09-04', 'Name': 'VGG-16', 'Type': 'Model', 'Memory_GB': 2.208}, {'Year': 2015, 'Date': '2015-12-10', 'Name': 'ResNet-152', 'Type': 'Model', 'Memory_GB': 0.9632}, {'Year': 2018, 'Date': '2018-10-11', 'Name': 'BERT-Large', 'Type': 'Model', 'Memory_GB': 5.44}, {'Year': 2019, 'Date': '2019-02-14', 'Name': 'GPT-2', 'Type': 'Model', 'Memory_GB': 24.0}, {'Year': 2020, 'Date': '2020-05-28', 'Name': 'GPT-3', 'Type': 'Model', 'Memory_GB': 2793.6}, {'Year': 2022, 'Date': '2022-04-04', 'Name': 'PaLM', 'Type': 'Model', 'Memory_GB': 8645.6}, {'Year': 2024, 'Date': '2024-07-23', 'Name': 'Llama 3 (405B)', 'Type': 'Model', 'Memory_GB': 6480.0}, {'Year': 2014, 'Date': None, 'Name': 'K80', 'Type': 'GPU', 'Memory_GB': 12}, {'Year': 2017, 'Date': None, 'Name': 'V100', 'Type': 'GPU', 'Memory_GB': 32}, {'Year': 2020, 'Date': None, 'Name': 'A100', 'Type': 'GPU', 'Memory_GB': 80}, {'Year': 2022, 'Date': None, 'Name': 'H100', 'Type': 'GPU', 'Memory_GB': 80}, {'Year': 2024, 'Date': None, 'Name': 'B200', 'Type': 'GPU', 'Memory_GB': 180}]
mx, my, ml = ([], [], [])
for r_ in wall_rows:
    if r_['Type'] == 'Model' and r_['Name'] in deck_label:
        d = r_['Date']
        mx.append(int(d[:4]) + (int(d[5:7]) - 1) / 12)
        my.append(r_['Memory_GB'])
        ml.append(deck_label[r_['Name']])
gx, gy, gl = ([], [], [])
for r_ in wall_rows:
    if r_['Type'] == 'GPU':
        gx.append(r_['Year'])
        gy.append(r_['Memory_GB'])
        gl.append(r_['Name'])
gpt3 = next((r_ for r_ in wall_rows if r_['Name'] == 'GPT-3'))
fig, ax = plt.subplots(figsize=(5.8, 2.35))
ax.plot(gx, gy, '--s', color=C['memory'], lw=2.5, ms=7)
ax.plot(mx, my, '-o', color=C['danger'], lw=2.5, ms=7)
for x, y, lab in zip(gx, gy, gl):
    ax.annotate(lab, (x, y), xytext=(0, -16), textcoords='offset points', ha='center', fontsize=10.5, fontweight='bold', color=C['memory'])
offs = {'AlexNet': (4, 8), '': (0, 0), 'GPT-2': (8, -12), 'GPT-3': (-10, 8), 'PaLM': (0, 9), 'Llama 3 405B': (0, -18)}
for x, y, lab in zip(mx, my, ml):
    ax.annotate(lab, (x, y), xytext=offs[lab], textcoords='offset points', ha='right' if lab == 'GPT-3' else 'left' if lab == 'GPT-2' else 'center', fontsize=10.5, fontweight='bold', color=C['danger'])
ax.annotate('GPT-3: 2.8 TB of state,\n35× one 80 GB GPU', (2020.4, gpt3['Memory_GB']), xytext=(2013.6, 1500), fontsize=11, fontweight='bold', color=C['text'], arrowprops=dict(arrowstyle='->', color=C['text'], lw=1.2))
ax.text(2012.3, 70, "one accelerator's memory", fontsize=11, fontweight='bold', color=C['memory'])
ax.text(2015.6, 0.35, 'training state, 16 B per parameter', fontsize=11, fontweight='bold', color=C['danger'])
ax.set_yscale('log')
ax.set_ylim(0.2, 30000)
ax.set_xlim(2011.8, 2025.2)
ax.set_ylabel('GB (log)')
ax.set_yticks([1, 10, 100, 1000, 10000])
ax.set_yticklabels(['1', '10', '100', '1,000', '10,000'])
ax.minorticks_off()
sp.save(fig, out / 'memory_wall.svg')
Ng = 64
W, G, O = (q.to('GB').magnitude for q in (Q_(14.0, 'GB'), Q_(14.0, 'GB'), Q_(84.0, 'GB')))
stage_key = {'DDP': 'z7_ddp_gb', 'ZeRO-1': 'z7_z1_gb', 'ZeRO-2': 'z7_z2_gb', 'ZeRO-3': 'z7_z3_gb'}
stages = [('DDP', W, G, O), ('ZeRO-1', W, G, O / Ng), ('ZeRO-2', W, G / Ng, O / Ng), ('ZeRO-3', W / Ng, G / Ng, O / Ng)]
cap = 80.0
for k in (1, 2, 3, 4):
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    for i, (name, wv, gv, ov) in enumerate(stages):
        a = 1 if i < k else 0
        traffic = f"{ {'DDP': 1.0, 'ZeRO-1': 1.0, 'ZeRO-2': 1.0, 'ZeRO-3': 1.5}[name]:g}×"
        ax.bar(i, wv, 0.58, color=C['memory'], alpha=a)
        ax.bar(i, gv, 0.58, bottom=wv, color=C['network'], alpha=a)
        ax.bar(i, ov, 0.58, bottom=wv + gv, color=C['compute'], alpha=a)
        tot = wv + gv + ov
        label(ax, i, tot + 4, f"{ {'DDP': '112', 'ZeRO-1': '29.3', 'ZeRO-2': '15.5', 'ZeRO-3': '1.75'}[name]} GB", C['danger'] if tot > cap else C['data'], i < k, ha='center', fontsize=13)
        label(ax, i, -26, f'traffic {traffic}', C['neutral'], i < k, ha='center', fontsize=11)
    ax.axhline(cap, color=C['danger'], lw=2, ls='--')
    ax.text(3.45, cap + 2, '80 GB GPU', color=C['danger'], fontsize=11, fontweight='bold', ha='right', va='bottom')
    ax.text(1.55, 120, 'weights', color=C['memory'], fontsize=11, fontweight='bold', ha='left')
    ax.text(1.55, 108, 'gradients', color=C['network'], fontsize=11, fontweight='bold', ha='left')
    ax.text(1.55, 96, 'optimizer (FP32)', color=C['compute'], fontsize=11, fontweight='bold', ha='left')
    ax.set_xticks(range(4), [s_[0] for s_ in stages], fontsize=12)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 128)
    ax.set_ylabel('GB per GPU')
    ax.tick_params(axis='x', length=0, pad=4)
    sp.save(fig, out / f'zero_step{k}.svg')

def pipe_grid(s, y0, slots, title, op, cw=46):
    s.text(20, y0 - 18, title, 23, V['text'], anchor='start', op=op)
    for st in range(4):
        y = y0 + st * 34
        s.text(62, y + 16, f'GPU {st}', 20, V['gray'], anchor='end', op=op)
        for t in range(len(slots[st])):
            x = 70 + t * cw
            cell = slots[st][t]
            if cell is None:
                s.rect(x, y, cw - 4, 30, V['bad_l'], 'none', 0, 3, op)
            else:
                fwd = cell[0] == 'F'
                s.rect(x, y, cw - 4, 30, V['c1_l'] if fwd else V['c2_l'], V['c1'] if fwd else V['c2'], 2, 3, op)
                s.text(x + (cw - 4) / 2, y + 16, cell, 20, V['c1'] if fwd else '#b45309', op=op)

def naive(p=4):
    T = 8
    g = [[None] * T for _ in range(p)]
    for st in range(p):
        g[st][st] = 'F'
        g[st][T - 1 - st] = 'B'
    return g

def gpipe(p=4, m=4):
    T = 2 * (m + p - 1)
    g = [[None] * T for _ in range(p)]
    for st in range(p):
        for i in range(m):
            g[st][st + i] = f'F{i + 1}'
            g[st][m + p - 1 + (p - 1 - st) + i] = f'B{m - i}'
    return g
for k in (1, 2):
    s = SVG(900, 345)
    pipe_grid(s, 30, naive(), 'One batch through 4 stages: each GPU idles 6 of 8 slots', 1.0)
    pipe_grid(s, 210, gpipe(), 'Four microbatches (GPipe): each GPU idles 6 of 14 slots, 43%', 1.0 if k >= 2 else 0.0, cw=55)
    s.save(out / f'pipe_step{k}.svg')
m = np.arange(1, 257)
for k in (1, 2):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    curves = [(4, C['memory'], True, 90), (8, C['compute'], True, 79), (16, C['network'], k >= 2, 68)]
    for p, col, show, ly in curves:
        f = np.array([calc_pipeline_bubble(p, int(mm)) for mm in m])
        ax.plot(m, 100 * f, color=col, lw=2.6, alpha=1 if show else 0)
        label(ax, 240, ly, f'p = {p} stages', col, show, fontsize=15, ha='right')
    pts = [(SimpleNamespace(bubble=0.4285714286, fill_slots=3, idle_slots=6, microbatches=4, name='GPipe, four stages and four microbatches', phase_slots=7, slots=14, stages=4), C['memory'], 'p=4, m=4: 43%', None, True), (SimpleNamespace(bubble=0.1794871795, fill_slots=7, idle_slots=14, microbatches=32, name='Eight pipeline stages at 32 microbatches', phase_slots=39, slots=78, stages=8), C['compute'], 'p=8, m=32: 18%', (-10, -22), True), (SimpleNamespace(bubble=0.3191489362, fill_slots=15, idle_slots=30, microbatches=32, name='Sixteen pipeline stages at 32 microbatches', phase_slots=47, slots=94, stages=16), C['network'], 'm=32: 32%', (8, 8), k >= 2)]
    for sc, col, lab, off, show in pts:
        p, mm, v = (sc.stages, sc.microbatches, 100 * sc.bubble)
        ax.plot([mm], [v], 'o', ms=9, color=col, mec='white', mew=1.5, alpha=1 if show else 0)
        if p in (4, 8):
            ax.annotate(lab, (mm, v), xytext=(1.15, 19 if p == 4 else 3), fontsize=15, fontweight='bold', color=col, arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
            continue
        ax.annotate(lab, (mm, v), xytext=off, textcoords='offset points', fontsize=15, fontweight='bold', color=col, alpha=1 if show else 0)
    ax.axhline(10.0, color=C['neutral'], lw=1.4, ls='--')
    ax.text(1.05, 10.0 + 2, '10%', color=C['neutral'], fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 4, 16, 64, 256])
    ax.set_xticklabels(['1', '4', '16', '64', '256'])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_xlim(1, 256)
    ax.set_xlabel('microbatches per step, m', fontsize=15)
    ax.set_ylabel('idle fraction (%)', fontsize=15)
    ax.tick_params(labelsize=14)
    sp.save(fig, out / f'bubble_step{k}.svg')
for k in (1, 2, 3):
    s = SVG(900, 300)
    for r_, gpu in enumerate((0, 1)):
        y = 40 + r_ * 120
        s.text(40, y + 30, f'GPU {gpu}', 23, V['gray'], anchor='start')
        s.rect(130, y + 8, 110, 46, V['ok_l'], V['ok'])
        s.text(185, y + 31, 'X (copy)', 22, V['ok'])
        s.line(240, y + 31, 278, y + 31, V['gray'], 3, arrow=True)
        s.rect(280, y + 8, 110, 46, V['c1_l'], V['c1'])
        s.text(335, y + 31, f'A{gpu + 1}: cols', 22, V['c1'])
        s.line(390, y + 31, 428, y + 31, V['gray'], 3, arrow=True)
        s.rect(430, y + 8, 90, 46, V['c1_l'], V['c1'])
        s.text(475, y + 31, f'Y{gpu + 1}', 23, V['c1'])
        op2 = 1 if k >= 2 else 0
        s.line(520, y + 31, 558, y + 31, V['gray'], 3, arrow=True, op=op2)
        s.rect(560, y + 8, 110, 46, V['c1_l'], V['c1'], op=op2)
        s.text(615, y + 31, f'B{gpu + 1}: rows', 22, V['c1'], op=op2)
        s.line(670, y + 31, 698, y + 31, V['gray'], 3, arrow=True, op=op2)
        s.rect(700, y + 8, 70, 46, '#f1f5f9', V['gray'], op=op2)
        s.text(735, y + 31, f'Z{gpu + 1}', 23, V['text'], op=op2)
    s.text(475, 132, 'no exchange: Y stays local', 21, V['c1'], weight='normal')
    op2 = 1 if k >= 2 else 0
    s.text(700, 132, 'partial sums', 21, V['gray'], weight='normal', op=op2)
    op3 = 1 if k >= 3 else 0
    s.rect(790, 40, 90, 206, V['c2_l'], V['c2'], op=op3)
    s.text(835, 128, 'All-', 23, '#b45309', op=op3)
    s.text(835, 154, 'Reduce', 23, '#b45309', op=op3)
    s.line(770, 71, 788, 71, V['c2'], 3, arrow=True, op=op3)
    s.line(770, 191, 788, 191, V['c2'], 3, arrow=True, op=op3)
    s.text(835, 274, 'Z = Z1 + Z2', 23, V['text'], op=op3)
    s.text(30, 274, 'Megatron: QKV, then attention out; FFN-1, then FFN-2', 21, V['gray'], weight='normal', anchor='start', op=op3)
    s.save(out / f'tp_step{k}.svg')
fig, ax = plt.subplots(figsize=(5.4, 3.4))
bars = [('compute', 'tp_compute_ms', V['c1']), ('TP over NVLink\n450 GB/s', 'tp_nvlink_ms', V['ok']), ('TP over one NDR\nport, 50 GB/s', 'tp_ndr_ms', V['bad'])]
for i, (n_, key, col) in enumerate(bars):
    y = 2 - i
    v = {'compute': 1.9, 'TP over NVLink\n450 GB/s': 0.4, 'TP over one NDR\nport, 50 GB/s': 3.5}[n_]
    ax.barh(y, v, color=col, height=0.6)
    ax.text(-0.08, y, n_, ha='right', va='center', fontsize=15, fontweight='bold')
    ax.text(v + 0.06, y, f"{ {'compute': '1.9', 'TP over NVLink\n450 GB/s': '0.4', 'TP over one NDR\nport, 50 GB/s': '3.5'}[n_]} ms", va='center', fontsize=16, fontweight='bold', color=col)
ax.axvline(1.9, color=V['c1'], lw=1.5, ls='--')
ax.set_xlim(0, 4.6)
ax.set_ylim(-0.6, 2.6)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('ms per GPT-3 layer, forward', fontsize=15)
ax.tick_params(labelsize=14)
sp.save(fig, out / 'tp_tiers.svg')
fig, ax = plt.subplots(figsize=(5.8, 1.95))
meg = sorted(({'mode': 'model', 'gpus': 1.0, 'efficiency_pct': 100.0}, {'mode': 'model', 'gpus': 2.0, 'efficiency_pct': 95.0}, {'mode': 'model', 'gpus': 4.0, 'efficiency_pct': 82.0}, {'mode': 'model', 'gpus': 8.0, 'efficiency_pct': 77.0}, {'mode': 'model+data', 'gpus': 64.0, 'efficiency_pct': 96.0}, {'mode': 'model+data', 'gpus': 128.0, 'efficiency_pct': 83.0}, {'mode': 'model+data', 'gpus': 256.0, 'efficiency_pct': 79.0}, {'mode': 'model+data', 'gpus': 512.0, 'efficiency_pct': 74.0}), key=lambda r: (r['mode'] != 'model', r['gpus']))
xs_ = [f"{r['gpus']:.0f}" for r in meg]
vals = [r['efficiency_pct'] for r in meg]
cols = [C['memory'] if r['mode'] == 'model' else C['compute'] for r in meg]
for i, (x_, v, c_) in enumerate(zip(xs_, vals, cols)):
    xi = i + (0.6 if i >= 4 else 0)
    ax.bar(xi, v, 0.62, color=c_)
    ax.text(xi, v + 3, f'{v:.0f}%', ha='center', fontsize=12, fontweight='bold', color=c_)
ax.set_xticks([i + (0.6 if i >= 4 else 0) for i in range(8)], xs_)
ax.text(1.5, 122, 'tensor parallel, one server', ha='center', fontsize=11.5, fontweight='bold', color=C['memory'])
ax.text(6.1, 122, f"{max((r['gpus'] for r in meg if r['mode'] == 'model')):.0f}-way TP × data parallel", ha='center', fontsize=11.5, fontweight='bold', color=C['compute'])
ax.set_ylim(0, 130)
ax.set_yticks([0, 50, 100])
ax.set_xlabel('GPUs (V100, DGX-2H)')
ax.set_ylabel('weak-scaling\nefficiency (%)')
sp.save(fig, out / 'megatron_scaling.svg')
cfg = [('A: DP-1024', 'eff_a_pct', 'state_gb', None), ('B: TP8 × DP128', 'eff_b_pct', 'state_b_gb', None), ('C: TP8 × PP4 × DP32', 'eff_c_pct', 'state_c_gb', 'zero1_c_gb')]
cap = 80.0
for k in (1, 2, 3):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.9, 2.15), gridspec_kw={'width_ratios': [1, 1]})
    for i, (name, eff_key, mem_key, z1_key) in enumerate(cfg):
        y = 2 - i
        sh = i < k
        eff, mem = ({'A: DP-1024': 15.0, 'B: TP8 × DP128': 83.0, 'C: TP8 × PP4 × DP32': 87.0}[name], {'A: DP-1024': 2800.0, 'B: TP8 × DP128': 350.0, 'C: TP8 × PP4 × DP32': 87.5}[name])
        z1 = _NUMS[z1_key] if z1_key else None
        a1.barh(y, eff, color=C['data'] if eff > 50 else C['danger'], height=0.6, alpha=1 if sh else 0)
        a1.text(-3, y, name, ha='right', va='center', fontsize=11.5, fontweight='bold')
        label(a1, eff + 2, y, f"{ {'A: DP-1024': '15', 'B: TP8 × DP128': '83', 'C: TP8 × PP4 × DP32': '87'}[name]}%", C['data'] if eff > 50 else C['danger'], sh, va='center', fontsize=12.5)
        a2.barh(y, mem, color=C['danger'] if mem > cap else C['data'], height=0.6, alpha=1 if sh else 0)
        label(a2, mem * 1.15, y, f"{ {'A: DP-1024': '2,800', 'B: TP8 × DP128': '350', 'C: TP8 × PP4 × DP32': '87.5'}[name]} GB", C['danger'], sh, va='center', fontsize=12.5)
        if z1:
            a2.barh(y - 0.36, z1, color=C['data'], height=0.28, alpha=1 if sh else 0)
            label(a2, 160, y - 0.36, f'with ZeRO-1: {_VALS[z1_key]} GB', C['data'], sh, va='center', fontsize=11)
    a1.set_xlim(0, 110)
    a1.set_ylim(-0.7, 2.6)
    a1.set_yticks([])
    a1.spines['left'].set_visible(False)
    a1.set_xlabel('scaling efficiency (%)')
    a2.set_xscale('log')
    a2.set_xlim(10, 20000)
    a2.set_ylim(-0.7, 2.6)
    a2.axvline(cap, color=C['danger'], lw=1.6, ls='--')
    a2.text(cap, 2.55, ' 80 GB', color=C['danger'], fontsize=10.5, fontweight='bold', va='bottom')
    a2.set_yticks([])
    a2.spines['left'].set_visible(False)
    a2.set_xticks([10, 100, 1000, 10000])
    a2.set_xticklabels(['10', '100', '1k', '10k'])
    a2.minorticks_off()
    a2.set_xlabel('training state per GPU (GB, log)')
    fig.subplots_adjust(wspace=0.08)
    sp.save(fig, out / f'hybrid_step{k}.svg')
s = SVG(900, 330)
for rrow, (y, lab) in enumerate([(30, 'replica 1'), (130, 'replica 2')]):
    s.text(18, y + 40, lab, 21, V['gray'], anchor='start')
    for n_ in range(4):
        x = 120 + n_ * 190
        s.rect(x, y, 160, 80, V['c1_l'], V['c1'], 2.5, 8)
        for gi in range(8):
            gx_ = x + 10 + gi * 18.4
            s.rect(gx_, y + 10, 14, 26, 'white', V['c1'], 1.5, 2)
        s.text(x + 80, y + 60, f'stage {n_ + 1}', 23, V['c1'])
        if n_ < 4 - 1:
            s.line(x + 162, y + 40, x + 186, y + 40, V['c3'], 3.5, arrow=True)
s.text(450, 228, '... 32 replicas in all', 22, V['gray'])
for n_ in range(4):
    x = 120 + n_ * 190 + 80
    s.line(x, 112, x, 128, V['c2'], 3.5, arrow=True)
s.rect(20, 250, 280, 66, V['c1_l'], V['c1'], 2, 6)
s.text(160, 270, 'TP = 8 inside a node', 23, V['c1'])
s.text(160, 296, 'NVLink, 450 GB/s', 21, V['c1'], weight='normal')
s.rect(310, 250, 280, 66, V['c3_l'], V['c3'], 2, 6)
s.text(450, 270, 'PP = 4 across nodes', 23, '#6d28d9')
s.text(450, 296, 'activations, point to point', 21, '#6d28d9', weight='normal')
s.rect(600, 250, 280, 66, V['c2_l'], V['c2'], 2, 6)
s.text(740, 270, 'DP = 32 across replicas', 23, '#b45309')
s.text(740, 296, '10.9 GB gradient shard', 21, '#b45309', weight='normal')
s.save(out / 'grid3d.svg')
print('figures written to', out)
_pol, _ref = (1120.0, 140.0)
_rew, _valg = (26.0, 208.0)
fig, ax = plt.subplots(figsize=(5.8, 1.1))
for y, parts, tot in ((1, [(_pol, C['memory'], 'policy'), (_ref, C['neutral'], 'ref.'), (_rew, C['data'], ''), (_valg, C['compute'], 'value')], 'PPO 1,494 GB'), (0, [(_pol, C['memory'], 'policy'), (_ref, C['neutral'], 'ref.')], 'DPO 1,260 GB')):
    left = 0
    for w, col, lab in parts:
        ax.barh(y, w, 0.62, left=left, color=col, edgecolor='white', lw=1)
        if lab:
            ax.text(left + w / 2, y, lab, ha='center', va='center', fontsize=11.5, fontweight='bold', color='white')
        left += w
    ax.text(left + 15, y, tot, va='center', fontsize=12.5, fontweight='bold', color=C['text'])
_x = _pol + _ref + _rew / 2
ax.annotate('reward', (_x, 1.31), xytext=(_x - 25, 1.62), ha='right', va='center', fontsize=11, fontweight='bold', color=C['data'], arrowprops=dict(arrowstyle='-', color=C['data'], lw=1.2))
ax.set_xlim(0, (_pol + _ref + _rew + _valg) * 1.32)
ax.set_ylim(-0.4, 1.8)
ax.axis('off')
sp.save(fig, out / 'rlhf_state.svg')
