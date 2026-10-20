#!/usr/bin/env python3
"""Every number on the Volume II Lecture 1 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature) and the calculations
(Scenarios.*, each backed by a solver); the chapter reads the same scenarios.
This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Models, Systems = mlsysim.Hardware, mlsysim.Models, mlsysim.Systems
Literature, Scenarios, ReferenceStats = mlsysim.Literature, mlsysim.Scenarios, mlsysim.ReferenceStats
Q_ = mlsysim.core.units.Q_
GPT3, LLAMA405 = Models.Language.GPT3, Models.Language.Llama3_405B
RUNS, CHIN = Literature.TrainingRuns, Literature.Chinchilla
HDR, NDR, ETH = (Systems.Fabrics.InfiniBand_HDR.bandwidth, Systems.Fabrics.InfiniBand_NDR.bandwidth,
                 Systems.Fabrics.Ethernet_100G.bandwidth)
REF = Systems.Clusters.Training_1K

SYNC, FAST = Scenarios.Gpt3GradientSync, Scenarios.Gpt3GradientSyncFasterAccelerator
CI = Scenarios.Gpt3DataParallelIntensity
OPT = Scenarios.Gpt3ComputeOptimal
PROFILE, DOUBLE = Scenarios.FleetLawProfileExample, Scenarios.FleetLawDoublingExample
PLAN = Scenarios.CoordinationOverheadExample
A1K, A10K, A4N = Scenarios.NodeAvailability1K, Scenarios.NodeAvailability10K, Scenarios.FourNinesAvailability10K
GROW = Scenarios.NotableRunGrowth2012to2024
LFAIL = Scenarios.GpuFailuresLlama3Run


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


print("== Hook and Napkin Math 1.1 (gradient-sync scenario)")
s, f = SYNC.result, FAST.result
v("gpt3_params_b", m(GPT3.parameters, "Gparam"), "{:.0f}", "running model, billions of parameters")
v("sync_gb", m(s.sync_bytes, "GB"), "{:.0f}", "16-bit ring traffic per worker per step, GB")
v("ib_gbps", m(HDR, "Gbps"), "{:.0f}", "InfiniBand HDR, Gb/s")
v("ib_gbs", m(HDR, "GB/s"), "{:.0f}", "InfiniBand HDR, GB/s")
v("eth_gbps", m(ETH, "Gbps"), "{:.0f}", "Ethernet, Gb/s")
v("eth_gbs", m(ETH, "GB/s"), "{:.1f}", "Ethernet, GB/s")
v("compute_s", m(SYNC.compute_time, "s"), "{:.1f}", "stated compute per step, s")
v("t_ib", m(s.transfer_times["InfiniBand"], "s"), "{:.0f}", "transfer over InfiniBand, s")
v("step_ib", m(s.step_times["InfiniBand"], "s"), "{:.1f}", "step over InfiniBand, s")
v("step_ib_round", m(s.step_times["InfiniBand"], "s"), "{:.0f}", "the same, whole seconds (vote answer)")
v("frac_ib", 100 * s.compute_fractions["InfiniBand"], "{:.1f}", "compute fraction over InfiniBand, %")
v("exposed_ib", 100 * (1 - s.compute_fractions["InfiniBand"]), "{:.0f}", "exposed communication over InfiniBand, %")
v("t_eth", m(s.transfer_times["Ethernet"], "s"), "{:.0f}", "transfer over Ethernet, s")
v("step_eth", m(s.step_times["Ethernet"], "s"), "{:.1f}", "step over Ethernet, s")
v("frac_eth", 100 * s.compute_fractions["Ethernet"], "{:.1f}", "compute fraction over Ethernet, %")
v("pj_bit", m(SYNC.energy_per_bit, "pJ/bit"), "{:.0f}", "InfiniBand energy, pJ/bit")
v("energy_j", m(s.energy_per_step, "J"), "{:.0f}", "network energy per worker per step, J")
v("energy_mj", m(s.energy_total, "MJ"), "{:.0f}", "the same over the run, MJ")
v("steps_m", SYNC.steps / 10 ** 6, "{:.0f}", "steps in the run, millions")
v("fast_compute_s", m(f.compute_time, "s"), "{:.1f}", "compute on 2x accelerators, s")
v("fast_step_s", m(f.step_times["InfiniBand"], "s"), "{:.1f}", "step on 2x accelerators, s")
v("fast_speedup", FAST.compute_speedup, "{:.0f}", "accelerator speedup")
v("fast_reduction", 100 * (1 - ratio(f.step_times["InfiniBand"], s.step_times["InfiniBand"])), "{:.1f}",
  "step reduction from 2x accelerators, %")

print("== Recall: failure clock (Llama 3 405B size)")
lf = LFAIL.result
v("gpu_mttf_h", m(LFAIL.component_mtbf, "hour"), "{:,.0f}", "per-GPU MTTF, h")
v("llama_gpus", LFAIL.n_components, "{:,.0f}", "Llama 3 405B GPUs")
v("llama_interval_h", m(lf.interval, "hour"), "{:.2f}", "system MTBF at 16,384 GPUs, h")
v("llama_per_day", lf.failures_per_day, "{:.1f}", "interruptions per day")

print("== Goals and running case")
state = (ReferenceStats.MixedPrecisionAdam.TrainingStateBytesPerParameter * GPT3.parameters).to("TB")
v("ref_gpus", REF.total_accelerators, "{:,.0f}", "reference cluster GPUs")
v("ref_nodes", REF.count, "{:.0f}", "reference cluster nodes")
v("state_bytes_param", m(ReferenceStats.MixedPrecisionAdam.TrainingStateBytesPerParameter, "byte/param"), "{:.0f}",
  "training state, bytes per parameter")
v("state_tb", m(state, "TB"), "{:.1f}", "training state, TB")
v("opt_gpus", RUNS.Opt175BGpus, "{:.0f}", "OPT-175B GPUs")

print("== Growth decomposition (Epoch AI via MLSysIM)")
g = GROW.result
v("growth_runs", g.run_count, "{:.0f}", "runs reporting count and duration")
v("growth_compute", g.compute_factor, "{:.1f}", "compute growth per year")
v("growth_count", g.count_factor, "{:.1f}", "accelerator count growth per year")
v("growth_duration", g.duration_factor, "{:.1f}", "duration growth per year")
v("growth_rate", g.rate_factor, "{:.1f}", "delivered rate growth per year")
v("growth_share", 100 * g.fleet_share, "{:.0f}", "share from count and duration, %")
v("alexnet_gpus", RUNS.AlexNetGpus, "{:.0f}", "AlexNet GPUs")

print("== Scaling laws (Chinchilla fit)")
v("chin_alpha", CHIN.ParametricLossAlpha, "{:.2f}", "parameter exponent")
v("chin_a", CHIN.ParametricLossA, "{:.1f}", "parameter coefficient A")
v("chin_e", CHIN.ParametricLossE, "{:.2f}", "irreducible loss E")
v("chin_b", CHIN.ParametricLossB, "{:.1f}", "token coefficient B")
v("chin_beta", CHIN.ParametricLossBeta, "{:.2f}", "token exponent")
v("halving_mult", Scenarios.ChinchillaParameterPowerLaw.halving_multiple, "{:.1f}", "2^(1/alpha) at the fit's alpha")
v("shallow_alpha", Scenarios.ShallowPowerLawExample.exponent, "{:.1f}", "illustrative shallow exponent")
v("shallow_inverse", 1 / Scenarios.ShallowPowerLawExample.exponent, "{:.0f}", "1 / that exponent")
v("shallow_mult", Scenarios.ShallowPowerLawExample.halving_multiple, "{:,.0f}", "its halving multiple")
v("gopher_params_b", CHIN.GopherBudgetOptimalParams / 10 ** 9, "{:.0f}", "Gopher-budget optimal parameters, B")
v("gopher_tokens_t", CHIN.GopherBudgetOptimalTokens / 10 ** 12, "{:.1f}", "Gopher-budget optimal tokens, T")
v("gopher_tpp", CHIN.GopherBudgetOptimalTokens / CHIN.GopherBudgetOptimalParams, "{:.0f}", "tokens per parameter there")
v("chin_tpp", CHIN.TokensPerParam, "{:.0f}", "Chinchilla tokens per parameter")
v("chin_k", CHIN.ComputeConstant, "{:.0f}", "FLOPs per parameter per token")
v("chin_kr", CHIN.ComputeConstant * CHIN.TokensPerParam, "{:.0f}", "the product, C = k r N^2")

print("== GPT-3 allocation (compute-optimal scenario)")
n_opt, d_opt = OPT.optimal
v("gpt3_params_1d", m(GPT3.parameters, "Gparam"), "{:.1f}", "GPT-3 parameters, B (model card)")
v("gpt3_layers", GPT3.layers, "{:.0f}", "GPT-3 layers")
v("gpt3_hidden", GPT3.hidden_dim, "{:.0f}", "GPT-3 width (model card, no comma)")
v("gpt3_hidden_c", GPT3.hidden_dim, "{:,.0f}", "GPT-3 width")
v("gpt3_batch_m", RUNS.Gpt3BatchTokens / 10 ** 6, "{:.1f}", "GPT-3 batch, M tokens")
v("gpt3_tokens_b", m(GPT3.training_tokens, "count") / 10 ** 9, "{:.0f}", "GPT-3 training tokens, B")
v("gpt3_budget_mant", OPT.budget_flops / 10 ** 23, "{:.2f}", "GPT-3 budget, x 10^23 FLOPs")
v("gpt3_tpp", OPT.actual_tokens_per_param, "{:.1f}", "GPT-3 tokens per parameter")
v("gpt3_tpp_gap", OPT.tokens_per_param / OPT.actual_tokens_per_param, "{:.0f}", "how far below Chinchilla")
v("gpt3_tpp_gap_word", round(OPT.tokens_per_param / OPT.actual_tokens_per_param), "words", "the same, in prose")
v("opt_params_b", n_opt / 10 ** 9, "{:.0f}", "optimal N at the same budget, B")
v("opt_tokens_t", d_opt / 10 ** 12, "{:.1f}", "optimal D at the same budget, T")

print("== Bandwidth step")
nvlink = Hardware.Cloud.H100.nvlink.bandwidth_per_direction
v("nvlink_gbs", m(nvlink, "GB/s"), "{:.0f}", "NVLink per direction, GB/s")
v("ndr_gbps", m(NDR, "Gbps"), "{:.0f}", "InfiniBand NDR, Gb/s")
v("ndr_gbs", m(NDR, "GB/s"), "{:.0f}", "InfiniBand NDR, GB/s")
v("nvlink_ndr", ratio(nvlink, NDR), "{:.0f}", "NVLink over NDR")
v("nvlink_hdr", ratio(nvlink, HDR), "{:.0f}", "NVLink over the hook's HDR")

print("== Ring traffic")
ring = float(Literature.Communication.RingAllreduceFactor)
grad32 = GPT3.size_in_bytes("fp32")
v("grad32_gb", m(grad32, "GB"), "{:.0f}", "FP32 gradient, GB")
v("ring32_gb", m(ring * grad32, "GB"), "{:,.0f}", "FP32 ring traffic, large N, GB")
v("ring16_gb", m(ring * GPT3.size_in_bytes("fp16"), "GB"), "{:.0f}", "FP16 ring traffic, large N, GB")
v("ring_n_small", Systems.Clusters.Training_512_H100.total_accelerators, "{:.0f}", "smaller worker count")
v("ring_n_large", REF.total_accelerators, "{:,.0f}", "larger worker count")

print("== Communication intensity (data-parallel intensity scenario)")
c = CI.result
a100 = Hardware.Cloud.A100
v("ci_tokens", c.local_tokens, "{:,.0f}", "tokens per worker per step")
v("ci_flops_mant", m(c.flops_per_step, "flop") / 10 ** 15, "{:.1f}", "FLOPs per worker per step, x 10^15")
v("ci_mant", c.intensity * 10 ** 5, "{:.1f}", "175B worker CI, x 10^-5 B/FLOP")
v("ci_balance_mant", c.balance * 10 ** 5, "{:.1f}", "fabric balance, x 10^-5 B/FLOP")
v("a100_tflops", m(a100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "A100 16-bit peak, TFLOP/s")
v("ci_peak_s", m(c.compute_time_at_peak, "s"), "{:.1f}", "compute time at peak, s")
alpha_w, beta_w = Scenarios.WorkloadAlphaIntensity, Scenarios.WorkloadBetaIntensity
v("alpha_mb", m(alpha_w.bytes_per_step, "MB"), "{:.0f}", "Alpha bytes sent, MB")
v("alpha_flops_exp", math.log10(m(alpha_w.flops_per_step, "flop")), "{:.0f}", "Alpha FLOPs, power of ten")
v("beta_gb", m(beta_w.bytes_per_step, "GB"), "{:.0f}", "Beta bytes sent, GB")
v("beta_flops_exp", math.log10(m(beta_w.flops_per_step, "flop")), "{:.0f}", "Beta FLOPs, power of ten")
v("alpha_ci_exp", math.log10(alpha_w.intensity), "{:.0f}", "Alpha CI, power of ten")
v("beta_ci_exp", math.log10(beta_w.intensity), "{:.0f}", "Beta CI, power of ten")
v("alpha_ci_neg", -math.log10(alpha_w.intensity), "{:.0f}", "Alpha CI, negative exponent (spoken)")
v("beta_ci_neg", -math.log10(beta_w.intensity), "{:.0f}", "Beta CI, negative exponent (spoken)")
v("beta_over_alpha", beta_w.intensity / alpha_w.intensity, "{:,.0f}", "Beta over Alpha")

print("== Llama 3 405B (Dubey et al. 2024, Table 4)")
t8, t16 = RUNS.Llama3_405B_TflopsPerGpu8K, RUNS.Llama3_405B_TflopsPerGpu16K
v("llama_t8k", t8, "{:.0f}", "BF16 TFLOP/s per GPU at 8,192")
v("llama_t16k", t16, "{:.0f}", "BF16 TFLOP/s per GPU at 16,384")
v("llama_mfu8k", 100 * RUNS.Llama3_405B_Mfu8K, "{:.0f}", "MFU at 8,192, %")
v("llama_mfu16k", 100 * RUNS.Llama3_405B_Mfu16K, "{:.0f}", "MFU at 16,384, %")
v("llama_gpus8k", RUNS.Llama3_405B_Gpus8K, "{:,.0f}", "first-stage GPUs")
v("llama_dp8k", RUNS.Llama3_405B_DataParallel8K, "{:.0f}", "DP at 8,192")
v("llama_dp16k", RUNS.Llama3_405B_DataParallel16K, "{:.0f}", "DP at 16,384")
v("llama_batch_m", RUNS.Llama3_405B_BatchTokens / 10 ** 6, "{:.0f}", "global batch, M tokens")
v("llama_drop", 100 * (1 - t16 / t8), "{:.0f}", "per-GPU throughput drop, %")
v("llama_gpu_ratio", LFAIL.n_components / RUNS.Llama3_405B_Gpus8K, "{:.0f}", "GPU ratio")

print("== Fleet Law (illustrative scenarios)")
v("fl_step_n", DOUBLE.step_at_n, "{:.0f}", "step at N, units")
v("fl_ideal_2n", DOUBLE.ideal_step, "{:.0f}", "ideal step at 2N, units")
v("fl_step_2n", DOUBLE.step_at_scaled, "{:.0f}", "step at 2N, units")
v("fl_speedup", DOUBLE.speedup, "{:.2f}", "speedup N to 2N")
v("fl_eff", DOUBLE.efficiency, "{:.2f}", "efficiency of the doubling")
v("dg_compute", m(PROFILE.compute_time, "s"), "{:.1f}", "profile compute, s")
v("dg_comm", m(PROFILE.communication_time, "s"), "{:.1f}", "profile communication, s")
v("dg_sync", m(PROFILE.synchronization_time, "s"), "{:.1f}", "profile sync, s")
v("dg_overlap", m(PROFILE.overlap_time, "s"), "{:.0f}", "profile overlap, s")
v("dg_step", m(PROFILE.step_time, "s"), "{:.1f}", "profile step, s")
v("dg_f", PROFILE.compute_fraction, "{:.2f}", "compute fraction f")
v("dg_f_pct", 100 * PROFILE.compute_fraction, "{:.0f}", "the same, percent")
fast_profile = PROFILE.variant(compute_speedup=FAST.compute_speedup)
v("dg_fast_step", m(fast_profile.step_time, "s"), "{:.1f}", "step with 2x compute, s")
v("dg_saving", m(PROFILE.step_time - fast_profile.step_time, "s"), "{:.1f}", "most option A saves, s")
v("pl_planned", PLAN.planned_speedup, "{:.0f}", "planned speedup")
v("pl_overhead_pct", 100 * PLAN.overhead_fraction, "{:.0f}", "overhead share, %")
v("pl_keep", 1 - PLAN.overhead_fraction, "{:.1f}", "share kept")
v("pl_realized", PLAN.realized_speedup, "{:.1f}", "realized speedup")

print("== Efficiency measures")
v("gpt3_peak_pct", 100 * Scenarios.Gpt3PublishedRun.peak_fraction, "{:.0f}", "GPT-3 run-average share of V100 peak, %")
v("megatron_pct", 100 * RUNS.MegatronLm2021PeakFraction, "{:.0f}", "Megatron-LM percent of peak")
v("megatron_gpus", RUNS.MegatronLm2021Gpus, "{:,.0f}", "Megatron-LM A100s")
v("llama_effective_pct", 100 * RUNS.Llama3_405B_EffectiveTrainingTimeFraction, "{:.0f}", "Llama 3 effective training time, %")

print("== Reliability gap (availability scenarios)")
v("avail_node_pct", 100 * A1K.availability, "{:.1f}", "per-node availability, %")
v("avail_node", A1K.availability, "{:.3f}", "the same, fraction")
v("avail_n_small", A1K.n_components, "{:,.0f}", "smaller job, nodes")
v("avail_n_large", A10K.n_components, "{:,.0f}", "larger job, nodes")
v("avail_1k", A1K.all_up, "{:.3f}", "all-up availability at 1,000")
v("avail_1k_pct", 100 * A1K.all_up, "{:.1f}", "the same, percent")
v("avail_10k_pct", 100 * A10K.all_up, "{:.4f}", "all-up availability at 10,000, %")
v("avail_4n_node_pct", 100 * A4N.availability, "{:.2f}", "four-nines per node, %")
v("avail_4n_pct", 100 * A4N.all_up, "{:.1f}", "all-up at 10,000 with four nines, %")
v("downtime_min", m(A4N.downtime_per_year, "minute"), "{:.1f}", "four nines, minutes down per year")

print("== OPT-175B and Llama 3 cadence")
opt_restarts = RUNS.Opt175BManualRestarts + RUNS.Opt175BAutomaticRestarts
v("opt_manual", RUNS.Opt175BManualRestarts, "{:.0f}", "manual restarts (lower bound)")
v("opt_auto", RUNS.Opt175BAutomaticRestarts, "{:.0f}", "automatic restarts (estimate)")
v("opt_restarts", opt_restarts, "{:.0f}", "restarts (lower bound)")
v("opt_days", RUNS.Opt175BTrainingDays, "{:.0f}", "training days (about two months)")
v("opt_hosts", RUNS.Opt175BHostsCycled, "{:.0f}", "hosts cycled out")
v("opt_hours", m(Q_(RUNS.Opt175BTrainingDays, "day"), "hour") / opt_restarts, "{:.0f}", "hours between restarts (bound)")
v("llama_interruptions", RUNS.Llama3_405B_UnexpectedInterruptions, "{:.0f}", "Llama 3 unexpected interruptions")
v("llama_days", RUNS.Llama3_405B_SnapshotDays, "{:.0f}", "Llama 3 snapshot days")
v("llama_cadence_h", m(Q_(RUNS.Llama3_405B_SnapshotDays, "day"), "hour") / RUNS.Llama3_405B_UnexpectedInterruptions,
  "{:.1f}", "hours between Llama 3 interruptions")

print("== Archetypes")
v("rec_table_tb", m(ReferenceStats.RecommenderAtScaleProfile.EmbeddingTableSize, "TB"), "{:.0f}", "Archetype B tables, TB")
v("phone_low_w", m(ReferenceStats.MobilePower.MobileMlSustainedLow, "W"), "{:.0f}", "Archetype C sustained low, W")
v("phone_high_w", m(ReferenceStats.MobilePower.MobileMlBurstHigh, "W"), "{:.0f}", "Archetype C burst high, W")
v("campus_mw", m(ReferenceStats.FacilityScaleAnchors.HyperscaleCampusPower, "MW"), "{:.0f}", "facility that can bind, MW")

if __name__ == "__main__":
    write(__file__)
