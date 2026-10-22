#!/usr/bin/env python3
"""Every number on the Volume II Lecture 0 slides and notes, read from MLSysIM.

Lecture 0 previews the course, so each number it shares with a later lecture is
read from the same registry entry or scenario that lecture's deck reads (GPT-3
and the A100 as in Lecture 1, the reference cluster and archetypes as in
Lecture 1, the H100 ridge point as in Lecture 2). This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Models, Systems = mlsysim.Hardware, mlsysim.Models, mlsysim.Systems
Literature, Scenarios, ReferenceStats = mlsysim.Literature, mlsysim.Scenarios, mlsysim.ReferenceStats
GPT3, LLAMA405, PALM = Models.Language.GPT3, Models.Language.Llama3_405B, Models.Language.PaLM_540B
A100, H100 = Hardware.Cloud.A100, Hardware.Cloud.H100
RUNS, FLEET, SERVE = Literature.TrainingRuns, Literature.FleetOperations, Literature.ServingResults
BLOOM = Scenarios.BloomTrainingAllocation
GROW = Scenarios.LandmarkClusterGrowth2012to2024
REF = Systems.Clusters.Training_1K


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


print("== Hook: GPT-3 on one A100 against the published run")
work, a100_peak = GPT3.training_ops, A100.compute.peak_flops
one_a100 = (work / a100_peak).to("s")
fleet_days = GPT3.training_days_ref
ops_exp = math.floor(math.log10(m(work, "flop")))
s_exp = math.floor(math.log10(m(one_a100, "s")))
v("gpt3_ops_mant", m(work, "flop") / 10 ** ops_exp, "{:.2f}", "GPT-3 training work, mantissa")
v("gpt3_ops_exp", ops_exp, "{:.0f}", "GPT-3 training work, power of ten")
v("a100_tflops", m(a100_peak, "TFLOP/s"), "{:.0f}", "A100 16-bit peak, TFLOP/s")
v("one_a100_s_mant", m(one_a100, "s") / 10 ** s_exp, "{:.1f}", "one A100 at peak, seconds, mantissa")
v("one_a100_s_exp", s_exp, "{:.0f}", "the same, power of ten")
v("one_a100_years", m(one_a100, "year"), "{:.0f}", "one A100 at peak, years")
v("one_a100_days", m(one_a100, "day"), "{:.1f}", "one A100 at peak, days (figure bar)")
v("gpt3_gpus", m(GPT3.training_accelerators_ref, "count"), "{:,.0f}", "GPT-3 run, V100 GPUs")
v("gpt3_days", m(fleet_days, "day"), "{:.0f}", "GPT-3 run, days")
v("gpt3_days_1d", m(fleet_days, "day"), "{:.1f}", "GPT-3 run, days (figure bar)")
v("fleet_ratio", ratio(one_a100, fleet_days), "{:.0f}", "one-A100 time over the run's time")

print("== Landmark accelerator counts (Introduction cluster-size figure)")
alexnet, llama = float(RUNS.AlexNetGpus), m(LLAMA405.training_accelerators_ref, "count")
v("alexnet_gpus", alexnet, "{:.0f}", "AlexNet GPUs")
v("megatron_gpus", float(RUNS.MegatronLm2019Gpus), "{:.0f}", "Megatron-LM (2019) GPUs")
v("msft_gpus", float(RUNS.MicrosoftOpenAI2020Gpus), "{:,.0f}", "Microsoft-OpenAI 2020 machine, GPUs (GPT-3 estimate)")
v("palm_chips", m(PALM.training_accelerators_ref, "count"), "{:.0f}", "PaLM TPU v4 chips")
v("llama1_gpus", float(RUNS.Llama1_65B_Gpus), "{:.0f}", "LLaMA 65B GPUs")
v("deepseek_gpus", float(RUNS.DeepSeekV3Gpus), "{:.0f}", "DeepSeek-V3 GPUs")
v("llama3_gpus", llama, "{:,.0f}", "Llama 3 405B H100s")
v("cluster_ratio", llama / alexnet, "{:,.0f}", "Llama 3 over AlexNet")
v("cluster_growth", GROW.growth, "{:.2f}", "fitted growth in accelerator count per year")
v("cluster_growth_round", GROW.growth, "{:.0f}", "the same, whole number (figure, slide)")

print("== Alps (ETH Zurich press release, 14 September 2024)")
alps = float(FLEET.AlpsGraceHopperSuperchips)
v("alps_chips", alps, "{:,.0f}", "Alps Grace Hopper superchips")
v("alps_vs_gpt3", alps / m(GPT3.training_accelerators_ref, "count"), "{:.2f}", "Alps over GPT-3's V100 count")

print("== Serving (Altman, OpenAI DevDay, 6 October 2025)")
tpm = float(SERVE.OpenAiApiTokensPerMinute2025)
v("chatgpt_users_m", float(SERVE.ChatGptWeeklyUsers2025) / 10 ** 6, "{:.0f}", "ChatGPT weekly users, millions")
v("api_tpm_b", tpm / 10 ** 9, "{:.0f}", "API tokens per minute, billions")
v("api_tps_m", m(mlsysim.core.units.Q_(tpm, "1/minute"), "1/s") / 10 ** 6, "{:.0f}", "API tokens per second, millions")

print("== BLOOM-176B as launched (BigScience tr11-176B-ml README and Slurm script)")
gpu_hours = BLOOM.accelerator_hours
v("bloom_gpus", BLOOM.accelerators_used, "{:.0f}", "GPUs the job used")
v("bloom_nodes", BLOOM.nodes_used, "{:.0f}", "nodes the job used")
v("bloom_alloc_gpus", BLOOM.accelerators_allocated, "{:.0f}", "GPUs allocated")
v("bloom_alloc_nodes", BLOOM.nodes_allocated, "{:.0f}", "nodes allocated")
v("bloom_per_node", BLOOM.accelerators_per_node, "{:.0f}", "GPUs per node")
v("bloom_nodes_doubled", BLOOM.nodes_used + BLOOM.nodes_used, "{:.0f}", "twice the nodes (prediction 1)")
v("bloom_reserve", BLOOM.reserve_accelerators, "{:.0f}", "GPUs in reserve")
v("bloom_reserve_nodes", BLOOM.reserve_nodes, "{:.0f}", "nodes in reserve")
v("bloom_reserve_pct", 100 * BLOOM.reserve_fraction, "{:.1f}", "reserve share of the allocation, %")
v("bloom_days", m(BLOOM.duration, "day"), "{:.0f}", "days from start to the end of epoch one")
v("hours_per_day", m(mlsysim.core.units.Q_(1, "day"), "hour"), "{:.0f}", "hours in a day")
v("bloom_gpu_hours", m(gpu_hours, "hour"), "{:,.0f}", "GPU-hours, every GPU every hour")
v("bloom_gpu_hours_m", m(gpu_hours, "hour") / 10 ** 6, "{:.1f}", "the same, millions")
v("bloom_one_gpu_years", m(gpu_hours, "year"), "{:.0f}", "one GPU alone, years")
v("bloom_time_limit_h", m(BLOOM.launch_time_limit, "hour"), "{:.0f}", "scheduler limit per launch, h")
v("bloom_launches", BLOOM.min_launches, "{:.0f}", "launches the limit forces, at least")
v("bloom_tp", BLOOM.tensor_parallel, "{:.0f}", "tensor-parallel degree")
v("bloom_pp", BLOOM.pipeline_parallel, "{:.0f}", "pipeline-parallel degree")
v("bloom_replica_gpus", BLOOM.accelerators_per_replica, "{:.0f}", "GPUs per model copy")
v("bloom_save_steps", float(RUNS.BloomSaveIntervalSteps), "{:.0f}", "launch script save interval, steps")
step = mlsysim.core.units.Q_(float(RUNS.BloomStepMinutes), "minute")
early = float(RUNS.BloomEarlySaveIntervalSteps)
v("bloom_step_min", m(step, "minute"), "{:.0f}", "time per step at full batch, min")
v("bloom_early_save_steps", early, "{:.0f}", "save interval before the crash, steps")
v("bloom_early_save_h", m(early * step, "hour"), "{:.0f}", "the same, hours")
v("bloom_chosen_save_h", float(RUNS.BloomChosenSaveIntervalHours), "{:.0f}", "interval the team settled on, h")

print("== Self-check and running cases")
state_bpp = ReferenceStats.MixedPrecisionAdam.TrainingStateBytesPerParameter
v("state_bytes_param", m(state_bpp, "byte/param"), "{:.0f}", "training state, bytes per parameter")
v("h100_tflops", m(H100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "H100 dense 16-bit peak, TFLOP/s")
v("h100_tbs", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM bandwidth, TB/s")
v("h100_ridge", m(H100.ridge_point(), "flop/byte"), "{:.0f}", "H100 ridge point, FLOP/byte")
v("gpt3_params_b", m(GPT3.parameters, "Gparam"), "{:.0f}", "Archetype A, billions of parameters")
v("state_tb", m(state_bpp * GPT3.parameters, "TB"), "{:.1f}", "Archetype A training state, TB")
v("ref_gpus", REF.total_accelerators, "{:,.0f}", "reference cluster GPUs")
v("rec_table_tb", m(ReferenceStats.RecommenderAtScaleProfile.EmbeddingTableSize, "TB"), "{:.0f}",
  "Archetype B tables, TB")
v("rec_qps_m", m(ReferenceStats.RecommenderAtScaleProfile.PlatformQps, "1/s") / 10 ** 6, "{:.0f}",
  "Archetype B queries per second, millions")
v("phone_low_w", m(ReferenceStats.MobilePower.MobileMlSustainedLow, "W"), "{:.0f}", "Archetype C sustained low, W")
v("phone_high_w", m(ReferenceStats.MobilePower.MobileMlBurstHigh, "W"), "{:.0f}", "Archetype C burst high, W")

if __name__ == "__main__":
    write(__file__)
