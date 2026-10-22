#!/usr/bin/env python3
"""Every number on the Volume II Lecture 17 (Conclusion) slides and notes, read from MLSysIM.

The conclusion recalls numbers from Lectures 1 to 16, so it reads the same
Scenarios.* those decks and their chapters read, plus the expansion decision
the Conclusion chapter works (Scenarios.Gpt3ClusterDoubling1K and the
Kaplan compute-law scenarios). MLSysIM holds the inputs (registry, Literature)
and the calculations; this file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.

Typed in the deck and not from MLSysIM: the settings of the concept question
and the storage diagnosis (taken from the chapter's self-check), the wrong
answer choices of the votes, the MegaScale configuration (batch 768, sequence
2,048), and the 4,096-GPU reading exercise.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Q_ = mlsysim.core.units.Q_
Models, Systems, Literature, Scenarios = mlsysim.Models, mlsysim.Systems, mlsysim.Literature, mlsysim.Scenarios
RUNS, FLEET = Literature.TrainingRuns, Literature.FleetOperations
LLAMA3 = Literature.Series.Llama3Interruptions.entries
MEGA = Literature.Series.MegaScaleStrongScaling.entries
TAIL = Literature.Series.TailAtScaleLeafLatency
P50, P99 = sorted(set(TAIL.column("percentile")))  # the two percentiles the table rows carry
GPT3, DEEPSEEK = Models.Language.GPT3, Models.Language.DeepSeek_V3
NODE = Systems.Nodes.DGX_H100

STAIR = Scenarios.BufferTransfer10GB                  # Lecture 2, Napkin Math 2.3
STAGED = Scenarios.Gpt3TieredCheckpoint2K             # Lecture 4, Napkin Math 4.7
DP = Scenarios.Gpt3DataParallel1K                     # Lecture 6, configuration A
YD = Scenarios.Gpt3CheckpointInterval10K              # Lecture 7, Napkin Math 7.2
FULL = Scenarios.FullSystemFailures10K
LFAIL = Scenarios.GpuFailuresLlama3Run                # GPU-only model at 16,384 GPUs
LONG = Scenarios.Llama3_70BLongContextAdmissionDgxH100  # Lecture 10, Napkin Math 10.5
REASON = Scenarios.ReasoningTokenLatency              # Lecture 10, Napkin Math 10.4
IX, COST = Scenarios.InferenceXServingFrontier, Scenarios.H100OutputTokenCost
FANOUT = Scenarios.ServingFanout100Leaves             # Lecture 10 fan-out arithmetic
ALERTS = Scenarios.ThreeSigmaModelFleetAlerts         # Lecture 12, Napkin Math 12.7
SALARY = Scenarios.SalaryMeanLaplaceRelease           # Lecture 13, Napkin Math 13.3
GEO = Scenarios.QuebecVersusPolandPlacement           # Lecture 15, Napkin Math 15.2
EXP = Scenarios.Gpt3ClusterDoubling1K                 # Conclusion, the expansion example
SLOW = Scenarios.SlowWorker8K
RARE = Scenarios.OneInAMillionAtMillionDaily
K2, K16 = Scenarios.KaplanComputeDoubling, Scenarios.KaplanSixteenfoldCompute
MOE = Scenarios.DeepSeekV3WeightTraffic
LINK, LINK2 = Scenarios.OpticalLinkEnergyShare, Scenarios.OpticalLinkEnergyDoubleShare


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def pct(x):
    return 100 * x


def tail(leaves, percentile):
    return next(r["latency_ms"] for r in TAIL.entries
                if r["leaves_awaited"] == leaves and r["percentile"] == percentile)


def mega(gpus):
    return next(r for r in MEGA if r["gpus"] == gpus)


print("== Retrieval: one number per row of the fleet stack")
t = STAIR.result.times
v("stair_gb", m(STAIR.payload, "GB"), "{:.0f}", "buffer moved, GB")
v("stair_hbm_ms", m(t["HBM"], "ms"), "{:.1f}", "10 GB over HBM, ms")
v("stair_nvlink_ms", m(t["NVLink"], "ms"), "{:.1f}", "10 GB over NVLink, ms")
v("stair_ib_ms", m(t["InfiniBand"], "ms"), "{:.0f}", "10 GB over InfiniBand, ms")
v("ckpt_gb", m(STAGED.workload.parameters * STAGED.bytes_per_param, "GB"), "{:,.0f}", "checkpoint, GB")
v("pause_s", m(STAGED.result.local_write_time, "s"), "{:.2f}", "checkpoint to local NVMe, s")
v("dp_eff_pct", pct(DP.result.efficiency), "{:.0f}", "pure data parallelism on 1,024 GPUs, efficiency %")
v("yd_gpus", FULL.n_devices, "{:,.0f}", "Lecture 7's large cluster, GPUs")
v("yd_tau_min", m(YD.result.interval, "minute"), "{:.1f}", "Young-Daly interval there, min")
lr = LONG.result
v("kv_long_n", lr.max_requests, "{:.0f}", "128K-token requests that fit one node")
v("kv_long_n_words", lr.max_requests, "words", "the same, in prose")
v("kv_context_k", LONG.context_tokens >> 10, "{:.0f}", "context, K tokens (K = 1,024)")
v("node_gpus", LONG.accelerators, "{:.0f}", "GPUs per node")
v("al_monitors", ALERTS.monitors, "{:,.0f}", "monitors")
v("al_daily", ALERTS.daily_false_alerts, "{:,.0f}", "3-sigma false alarms per day")
v("sal_err", m(SALARY.mean_error(), "USD"), "{:.0f}", "DP error of the mean, $")
v("sal_n", SALARY.group_size, "{:,.0f}", "salaries in the mean")
g = GEO.result
v("geo_ratio", ratio(g.emissions["Poland"], g.emissions["Quebec"]), "{:.0f}", "Poland / Quebec carbon, same energy")

print("== Running case: the reference job and the proposed doubling")
v("params_b", m(GPT3.parameters, "Gparam"), "{:.0f}", "dense transformer, billions of parameters")
v("ckpt_tb", m(EXP.checkpoint.checkpoint_bytes, "TB"), "{:.2f}", "checkpoint, TB")
v("ref_gpus", EXP.n_devices, "{:,.0f}", "reference cluster, GPUs")
v("ref_nodes", Systems.Clusters.Training_1K.count, "{:.0f}", "reference cluster, nodes")
v("gpus_per_node", NODE.accelerators_per_node, "{:.0f}", "GPUs per node")
v("exp_gpus", EXP.expanded_devices, "{:,.0f}", "proposed cluster, GPUs")
v("exp_factor", EXP.factor, "{:.0f}", "expansion factor (ideal speedup)")

print("== Failure clock (GPU-only model, MTTF / N) against Llama 3")
v("gpu_mttf_h", m(EXP.component_mtbf, "hour"), "{:,.0f}", "per-GPU MTTF, h")
v("mtbi_ref_h", m(EXP.mtbf, "hour"), "{:.1f}", "1,024 GPUs, h between interruptions")
v("mtbi_exp_h", m(EXP.expanded_mtbf, "hour"), "{:.1f}", "2,048 GPUs, h")
v("llama_gpus", LFAIL.n_components, "{:,.0f}", "Llama 3 405B GPUs")
v("mtbi_llama_h", m(LFAIL.result.interval, "hour"), "{:.2f}", "16,384 GPUs, model, h")
llama_n = float(RUNS.Llama3_405B_UnexpectedInterruptions)
llama_days = float(RUNS.Llama3_405B_SnapshotDays)
v("llama_interrupts", llama_n, "{:.0f}", "Llama 3 unexpected interruptions")
v("llama_days", llama_days, "{:.0f}", "snapshot, days")
v("llama_mttf_h", m(Q_(llama_days, "day"), "hour") / llama_n, "{:.1f}", "measured h between interruptions")
v("kok_16k_h", float(FLEET.Kokolis16KGpuProjectedMttfHours), "{:.1f}", "projected job MTTF at 16,384 GPUs, h")
gpu_parts = sum(r["interruption_count"] for r in LLAMA3 if r["category"] == "GPU")
v("causes_gpu", gpu_parts, "{:.0f}", "GPU-part interruptions")
v("causes_non_gpu", llama_n - gpu_parts, "{:.0f}", "interruptions the GPU-only model misses")
v("causes_non_gpu_pct", pct((llama_n - gpu_parts) / llama_n), "{:.0f}", "their share, %")

print("== Tail at scale")
v("fanout_leaves", FANOUT.n_components, "{:.0f}", "servers per request")
v("fanout_leaf_slow_pct", pct(1 - FANOUT.availability), "{:.0f}", "each slow, % of the time")
v("fanout_slow_pct", pct(1 - FANOUT.all_up), "{:.0f}", "slow requests, %")
v("fanout_slow_pct1", pct(1 - FANOUT.all_up), "{:.1f}", "the same, one decimal")
v("db_one_p99", tail("one random leaf", P99), "{:.0f}", "one random leaf, p99 ms")
v("db_95_p99", tail("95% of all leaves", P99), "{:.0f}", "95% of the leaves, p99 ms")
v("db_all_p99", tail("all leaves", P99), "{:.0f}", "all leaves, p99 ms")
v("db_tail_ratio", tail("all leaves", P99) / tail("one random leaf", P99), "{:.0f}", "all leaves / one leaf, p99")
v("db_one_p50", tail("one random leaf", P50), "{:.0f}", "one random leaf, median ms")
v("db_all_p50", tail("all leaves", P50), "{:.0f}", "all leaves, median ms")

print("== Rare events and the straggler")
day = Q_(1, "day")
per_day = RARE.events_per_second * m(day, "s")
v("rare_req_exp", round(math.log10(m(RARE.request_rate, "1/day"))), "{:.0f}", "requests a day, power of ten")
v("rare_p_exp", round(math.log10(RARE.event_probability)), "{:.0f}", "failure probability, power of ten")
v("rare_per_day", per_day, "{:.0f}", "expected failures a day")
v("rare_req10_m", 10 * m(RARE.request_rate, "1/day") / 10 ** 6, "{:.0f}", "the grown service, millions a day")
v("rare_per_day10", 10 * per_day, "{:.0f}", "expected failures a day there")
v("strag_n", SLOW.n_devices, "{:,.0f}", "workers")
v("strag_speed_pct", pct(SLOW.slow_speed), "{:.0f}", "slow worker's speed, %")
v("strag_speed", SLOW.slow_speed, "{:.1f}", "the same, fraction")
v("strag_mult", ratio(SLOW.slow_step, SLOW.step_time), "{:.2f}", "step-time multiplier")
v("strag_loss_pct", pct(1 - SLOW.throughput), "{:.0f}", "throughput loss, %")

print("== The expansion: recovery")
v("t_write_s", m(EXP.checkpoint.t_write, "s"), "{:.1f}", "checkpoint write, s")
v("ov_ref_pct", pct(EXP.overhead), "{:.1f}", "Young-Daly overhead at 1,024, %")
v("ov_exp_pct", pct(EXP.expanded_overhead), "{:.1f}", "Young-Daly overhead at 2,048, %")
v("ov_ref", EXP.overhead, "{:.3f}", "the same at 1,024, fraction")
v("ov_exp", EXP.expanded_overhead, "{:.3f}", "the same at 2,048, fraction")
v("ov_growth", EXP.overhead_growth, "{:.2f}", "overhead growth per doubling (sqrt 2)")
v("tau_ref_min", m(EXP.interval, "minute"), "{:.0f}", "Young-Daly interval at 1,024, min")
v("tau_exp_min", m(EXP.expanded_interval, "minute"), "{:.0f}", "Young-Daly interval at 2,048, min")
v("retained_ref", EXP.retained, "{:.3f}", "retained fraction at 1,024")
v("retained_exp", EXP.expanded_retained, "{:.3f}", "retained fraction at 2,048")
v("breakeven", EXP.breakeven_speedup, "{:.3f}", "break-even step speedup")
v("breakeven2", EXP.breakeven_speedup, "{:.2f}", "the same as the chapter rounds it")
v("ideal_hours_pct", pct(EXP.accelerator_hours_change(EXP.factor)), "{:.1f}", "accelerator-hours at ideal 2x, %")
v("ideal_sooner_pct", -pct(EXP.run_time_change(EXP.factor)), "{:.0f}", "run shorter at an ideal 2x, %")
v("superlinear", EXP.superlinear_threshold, "{:.3f}", "step speedup that would save accelerator-hours")

print("== The expansion: communication (MegaScale, Jiang et al. 2024, Table 2)")
lo, mid, hi = MEGA[0], mega(EXP.n_devices / EXP.factor), mega(EXP.n_devices)
ms_speedup = hi["megascale_tokens_s"] / mid["megascale_tokens_s"]
v("ms_lo_gpus", lo["gpus"], "{:,.0f}", "smallest MegaScale run, GPUs")
v("ms_mid_gpus", mid["gpus"], "{:,.0f}", "half the reference cluster, GPUs")
v("ms_hi_gpus", hi["gpus"], "{:,.0f}", "largest MegaScale run, GPUs")
v("ms_lo_k", lo["megascale_tokens_s"] / 10 ** 3, "{:.1f}", "throughput at 256, k tokens/s")
v("ms_hi_k", hi["megascale_tokens_s"] / 10 ** 3, "{:.1f}", "throughput at 1,024, k tokens/s")
v("ms_speedup", ms_speedup, "{:.2f}", "measured speedup 512 -> 1,024")
v("ms_mfu_mid", mid["megascale_mfu_pct"], "{:.1f}", "MFU at 512, %")
v("ms_mfu_hi", hi["megascale_mfu_pct"], "{:.1f}", "MFU at 1,024, %")
v("ms_sooner_pct", -pct(EXP.run_time_change(ms_speedup)), "{:.0f}", "run shorter at the measured speedup, %")
v("ms_hours_pct", pct(EXP.accelerator_hours_change(ms_speedup)), "{:.1f}", "accelerator-hours at the measured speedup, %")

print("== The expansion: what doubled compute buys (Kaplan et al. 2020)")
v("kaplan_alpha", K2.exponent, "{:.3f}", "compute exponent")
v("loss_factor", K2.multiple_factor, "{:.3f}", "loss multiplier per doubling")
v("loss_cut_pct", pct(K2.multiple_reduction), "{:.1f}", "loss reduction per doubling, %")
doublings = round(math.log(K16.resource_multiple, K2.resource_multiple))
loss = 1
for k in range(1, doublings + 1):
    v(f"loss_drop{k}", loss * K2.multiple_reduction, "{:.3f}", f"doubling {k}: absolute drop, loss 1 at start")
    loss *= K2.multiple_factor
v("k16", K16.resource_multiple, "{:.0f}", "compute multiple for four doublings")
v("k16_words", K16.resource_multiple, "words", "the same, in prose")
v("k16_cut_pct", pct(K16.multiple_reduction), "{:.1f}", "loss reduction at sixteen times the compute, %")

print("== Where next: reasoning, MoE, links")
v("reason_tokens", REASON.reasoning_tokens, "{:.0f}", "reasoning tokens")
v("reason_tok_ms", m(REASON.per_token_time, "ms"), "{:.0f}", "time per token, ms")
v("reason_time_s", m(REASON.reasoning_time, "s"), "{:.1f}", "reasoning time, s")
b200 = IX.solve("b200", COST.targets[2])
v("ix_span", b200.peak_throughput / b200.best_at, "{:.0f}", "B200 per-GPU output, slowest / fastest stream per user")
v("moe_total_b", m(DEEPSEEK.parameters, "Gparam"), "{:.0f}", "DeepSeek-V3 parameters, billions")
v("moe_active_b", m(DEEPSEEK.active_parameters, "Gparam"), "{:.0f}", "active per token, billions")
v("moe_resident_gb", m(MOE.resident_bytes, "GB"), "{:,.0f}", "all experts resident, FP16, GB")
v("moe_active_gb", m(MOE.active_bytes, "GB"), "{:.0f}", "read per token, FP16, GB")
v("moe_ratio", MOE.resident_to_active, "{:.1f}", "resident / active bytes")
v("moe_ratio0", MOE.resident_to_active, "{:.0f}", "the same, whole number")
v("dense_b", m(MOE.dense_parameters, "Gparam"), "{:.0f}", "dense baseline, billions")
v("dense_gb", m(MOE.dense_bytes, "GB"), "{:.0f}", "dense baseline read per step, FP16, GB")
v("link_gain", 1 / LINK.energy_ratio, "{:.0f}", "link energy-per-bit improvement, x")
v("link_share", LINK.share, "{:.2f}", "links' share of fleet energy (illustrative)")
v("link_share_pct", pct(LINK.share), "{:.0f}", "the same, %")
v("link_ratio", LINK.energy_ratio, "{:.1f}", "new energy per bit / old")
v("link_saving_pct", pct(LINK.saving), "{:.0f}", "fleet saving, %")
v("link_ceiling_pct", pct(LINK.ceiling), "{:.0f}", "ceiling for a free link, %")
v("link2_share_pct", pct(LINK2.share), "{:.0f}", "twice the share, %")
v("link2_saving_pct", pct(LINK2.saving), "{:.0f}", "fleet saving at twice the share, %")

if __name__ == "__main__":
    write(__file__)
