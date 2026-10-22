#!/usr/bin/env python3
"""Every number on the Lecture 16 (Conclusion) slides and notes, read from MLSysIM.

The conclusion recalls earlier lectures, so each recalled number is read from
the same MLSysIM object the earlier deck reads (named in each block), not
retyped: the hardware and model registries, the Scenarios.* the earlier
chapters compute with, and the cited results in Literature. The scenarios the
Conclusion chapter computes with (Llama2_70B_DecodeTwoH100,
ChatEndpointTtftMeanVersusP99, HostPreprocessingRequestPath,
ComposedAssistantRequestPath, Conv3x3At256Channels, GpuFailuresTraining1K) are
the ones its LEGO cells read. This file only selects values, converts units,
and formats them; it types no numbers. Each value is recorded with v() under
the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports this module, so figures and slides
share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

from mlsysim.core.units import BYTES_FP16, BYTES_FP32, BYTES_INT8  # noqa: E402

Hardware, Models, Systems, Platforms = mlsysim.Hardware, mlsysim.Models, mlsysim.Systems, mlsysim.Platforms
Scenarios, Literature = mlsysim.Scenarios, mlsysim.Literature
TR, IR = Literature.TrainingRuns, Literature.IncidentReports
DM, TECH = Literature.DeviceMeasurements, Hardware.Tech
H100 = Hardware.Cloud.H100

# Recalled scenarios, each as the earlier deck reads it
CLAIM = Scenarios.DatasheetVersusSustainedEdgeChip          # Lecture 12
HOST = Scenarios.Gpt2XlEagerDecodeDispatch                  # Lecture 7
TOKEN = Scenarios.Llama2_7BTokenOnFlagshipPhone             # Lecture 10
MIXED = Scenarios.Gpt2MixedPrecisionAdamOnH100              # Lecture 8
FP32_STEP = Scenarios.Gpt2Fp32AdamBatch32OnH100             # Lecture 8
PALM = Scenarios.Palm540BMixedPrecisionAdam                 # Lecture 8
SVC = Scenarios.ResNet50EagerFp16BatchingH100               # Lecture 13
COST = Scenarios.ResNet50BatchOneVersusEightCostH100        # Lecture 13
FAN100 = Scenarios.ServingFanout100Leaves                   # Lecture 13
WEBTEXT = Scenarios.Gpt2XlWebTextCorpus                     # Lecture 9
SEP64 = Scenarios.ImageNet64ChannelConv                     # Lecture 6
QKV = Scenarios.Gpt2XlQkvProjectionPrefill                  # Lecture 11
S3_MIXED = Scenarios.ResNet50Stage3ConvBatch32Int8Weights   # Lecture 11
ROOF_DECODE = Scenarios.Gpt2XlDecodeH100Roofline            # Lecture 11
ARM = Scenarios.RoboticArmRegionalCloud                     # Lecture 2
GFT = Scenarios.GoogleFluTrendsDrift                        # Lecture 1
IRON_RN = Scenarios.ResNet50Batch64StepH100                 # Lecture 1
YIELD = Scenarios.ThaiClinicRetinalPhotoYield               # Lecture 3
POD, SPARK = Systems.Pods.TPUv4_4096, Hardware.Workstation.DGX_Spark   # Lecture 2
# The Conclusion chapter's own scenarios
SEP256 = Scenarios.Conv3x3At256Channels
L70 = Scenarios.Llama2_70B_DecodeTwoH100
TTFT = Scenarios.ChatEndpointTtftMeanVersusP99
AMDAHL = Scenarios.HostPreprocessingRequestPath
DAG = Scenarios.ComposedAssistantRequestPath
FAIL = Scenarios.GpuFailuresTraining1KThreeWeeks
FAIL16K = Scenarios.GpuFailuresLlama3Run
GPT2_DECODE = Scenarios.Gpt2XlFp16DecodeOnH100


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def pct(x):
    """A fraction as percent."""
    return 100 * x


def row(series, **match):
    """The one row of a Literature series whose fields equal ``match``."""
    rows = [r for r in series.entries if all(r[k] == x for k, x in match.items())]
    assert len(rows) == 1, match
    return rows[0]


print("== The stack, one number per layer (each from the lecture that taught it)")
v("tpu_ppw_lo", lit(DM.TpuV1PerfPerWattGainLow), "{:.0f}", "TPU v1 performance per watt, low (L11)")
v("tpu_ppw_hi", lit(DM.TpuV1PerfPerWattGainHigh), "{:.0f}", "TPU v1 performance per watt, high (L11)")
v("hook_eff", m(CLAIM.claimed_efficiency, "TOPS/W"), "{:.0f}", "datasheet TOPS/W (L12)")
v("hook_sus_eff", m(CLAIM.sustained_efficiency, "TOPS/W"), "{:.1f}", "sustained TOPS/W (L12)")
v("gpt2_host_ms", m(HOST.host_time, "ms"), "{:.1f}", "GPT-2 XL token, eager host time, ms (L7)")
v("gpt2_read_ms", m(HOST.device_time, "ms"), "{:.1f}", "GPT-2 XL token, weight read, ms (L7)")
v("p7b_b", m(TOKEN.workload.parameters, "param") / 10 ** 9, "{:.0f}", "7B model parameters, billions (L10)")
v("mem7b_gb", m(TOKEN.weight_bytes, "GB"), "{:.0f}", "7B in FP16, GB (L10)")
v("phone_ram_gb", m(Platforms.Mobile.ram, "GiB"), "{:.0f}", "phone RAM, GB as labeled (L10)")
v("mp_total_b", m(MIXED.bytes_per_trained_parameter, "byte"), "{:.0f}", "training bytes per parameter (L8)")
b_one, b_eight = COST.batch_sizes
v("b1_usd", m(COST.cost_per_million(b_one), "dollar"), "{:.2f}", "batch 1, $ per million requests (L13)")
v("b8_usd", m(COST.cost_per_million(b_eight), "dollar"), "{:.2f}", "batch 8, $ per million requests (L13)")
v("batch_eight", b_eight, "{:.0f}", "the batched size")
v("hook_ratio", COST.cost_ratio, "{:.1f}", "batch 1 cost over batch 8 cost (L13)")
v("z_writedown_m", lit(IR.ZillowQ3WriteDownUsd) / 10 ** 6, "{:.0f}", "Zillow Q3 write-down, $ million (L14)")
v("ober_before", lit(Literature.Fairness.ObermeyerBlackShareBeforePct), "{:.1f}", "Black share, cost label, %")
v("ober_after", lit(Literature.Fairness.ObermeyerBlackShareAfterPct), "{:.1f}", "Black share, need label, %")
v("tay_hours", IR.TayHoursToSuspension, "{:.0f}", "Tay launch to suspension, hours (L4)")
v("gpt2_passes", WEBTEXT.passes, "{:.0f}", "GPT-2 XL passes over WebText (L9)")

print("== Depthwise separable (Scenarios.Conv3x3At256Channels; Lecture 6's ImageNet64ChannelConv)")
v("dw_ratio", SEP256.separable_ratio, "{:.1f}", "standard over separable, 3x3, 256 channels")
v("dw_kernel", SEP256.kernel, "{:.0f}", "kernel width")
v("dw_channels", SEP256.out_channels, "{:.0f}", "output channels")
v("dw_ratio_l6", SEP64.separable_ratio, "{:.1f}", "the same at 64 channels (L6)")
v("dw_channels_l6", SEP64.out_channels, "{:.0f}", "Lecture 6's channels")
v("int8_of_fp32_pct", pct(ratio(BYTES_INT8, BYTES_FP32)), "{:.0f}", "INT8 bytes as a share of FP32, %")

print("== Roofline on one H100 (Hardware.Cloud.H100; Lecture 11's scenarios)")
bw, p16, p8 = H100.memory.bandwidth, H100.compute.peak_flops, H100.compute.precision_flops["int8"]
ridge16, ridge8 = ratio(p16, bw * mlsysim.core.units.Q_(1, "flop/byte")), ratio(p8, bw * mlsysim.core.units.Q_(1, "flop/byte"))
v("h100_bw", m(bw, "TB/s"), "{:.2f}", "H100 HBM bandwidth, TB/s")
v("h100_fp16", m(p16, "TFLOP/s"), "{:,.0f}", "H100 dense FP16, TFLOP/s")
v("h100_int8", m(p8, "TFLOP/s"), "{:,.0f}", "H100 dense INT8, TOPS")
v("ridge16", ridge16, "{:.0f}", "FP16 ridge, FLOP/byte")
v("ridge8", ridge8, "{:.0f}", "INT8 ridge, OP/byte")
v("dec_ai", m(ROOF_DECODE.intensity, "flop/byte"), "{:.0f}", "GPT-2 XL decode intensity, FLOP/byte")
v("dec_ai_word", m(ROOF_DECODE.intensity, "flop/byte"), "words", "the same, in prose")
v("qkv_ai", QKV.traffic_intensity, "{:,.0f}", "GPT-2 XL QKV at prefill, FLOP/byte")
v("s3_mixed", S3_MIXED.traffic_intensity, "{:.0f}", "ResNet-50 stage-3 conv, INT8 weights, FP16 activations")

print("== Llama 2 70B at batch 1 on two H100s (Scenarios.Llama2_70B_DecodeTwoH100)")
d70 = L70.result
shards = L70.tp_size
v("l70_gb", m(d70.weight_bytes * shards, "GB"), "{:.0f}", "FP16 weights, GB")
v("l70_gflop", m(d70.flops_per_step * shards, "GFLOP"), "{:.0f}", "work per token, GFLOP")
v("two_bw", m(L70.hardware.memory.bandwidth * shards, "TB/s"), "{:.2f}", "two H100s, TB/s")
v("two_peak", m(L70.hardware.compute.peak_flops * shards, "TFLOP/s"), "{:,.0f}", "two H100s, TFLOP/s")
v("l70_mem_ms", m(d70.memory_time, "ms"), "{:.1f}", "memory floor, ms")
v("l70_comp_ms", m(d70.compute_time, "ms"), "{:.2f}", "compute floor, ms")
v("l70_ratio", d70.bound_ratio, "{:.1f}", "memory floor over compute floor")
v("l70_ratio_round", d70.bound_ratio, "{:.0f}", "the same, whole (title)")
MLP = load_series("slides/vol1/13_model_serving/images/data/mlperf_inference_v5_0_llama2_70b.csv")
conv = row(MLP, benchmark="llama2-70b-99", scenario="Server")
inter = row(MLP, benchmark="llama2-70b-interactive-99", scenario="Server")
v("mlp_gpus", conv["accelerators"], "{:.0f}", "H100s in the MLPerf system")
v("mlp_conv_tps", conv["tokens_per_s"], "{:,.0f}", "Server tokens/s, conversational bounds")
v("mlp_conv_ttft", conv["ttft_bound_ms"], "{:,.0f}", "conversational TTFT bound, ms")
v("mlp_conv_tpot", conv["tpot_bound_ms"], "{:.0f}", "conversational TPOT bound, ms")
v("mlp_int_tps", inter["tokens_per_s"], "{:,.0f}", "Server tokens/s, interactive bounds")
v("mlp_int_ttft", inter["ttft_bound_ms"], "{:.0f}", "interactive TTFT bound, ms")
v("mlp_int_tpot", inter["tpot_bound_ms"], "{:.0f}", "interactive TPOT bound, ms")

print("== Part I: data (Lecture 1's Google Flu Trends, Lecture 2's robot arm)")
g = GFT.result
v("gft_peak", g.peak_proxy, "{:.1f}", "2012-13 peak of the estimate, % of visits")
v("cdc_peak", g.peak_reference, "{:.1f}", "2012-13 peak of the CDC count, % of visits")
v("gft_miss_weeks", Literature.PredictionQuality.GoogleFluTrendsOverestimateWeeks, "{:.0f}", "weeks too high")
v("gft_window_weeks", Literature.PredictionQuality.GoogleFluTrendsOverestimateWindowWeeks, "{:.0f}", "weeks compared")
v("arm_km", m(ARM.distance, "km"), "{:,.0f}", "arm to cloud region, km")
v("arm_rtt_ms", m(ARM.round_trip, "ms"), "{:.0f}", "fiber round trip, ms")
v("arm_budget_ms", m(ARM.budget, "ms"), "{:.0f}", "stop deadline, ms")

print("== Part II: the iron law and training memory (Lectures 1 and 8)")
v("iron_eta", IRON_RN.efficiency, "{:.1f}", "utilization of Lecture 1's bars")
v("iron_rn_batch", IRON_RN.batch_size, "{:.0f}", "ResNet-50 batch in the bars")
v("palm_state_tb", m(PALM.total, "TB"), "{:.1f}", "PaLM mixed-precision Adam state, TB")
v("palm_h100s", round(PALM.capacity_share, -1), "{:.0f}", "the same in H100s of memory, to the ten")
v("palm_chips", TR.Palm540B_Phase1Chips, "{:,.0f}", "PaLM's TPU v4 chips")
v("fp32_step_h100s_word", FP32_STEP.training_capacity_multiple, "words", "FP32 GPT-2 XL step in H100s")

print("== Part III: optimization (Hardware.Tech energies; Scenarios.HostPreprocessingRequestPath)")
e_dram = m(TECH.Memory.DRAM.energy_per_access, "pJ")
v("dram_per_mul", e_dram / m(TECH.Op.FlopFp32.energy, "pJ/flop"), "{:.0f}", "DRAM read over FP32 multiply")
v("dram_per_fp16", e_dram / m(TECH.Op.FlopFp16.energy, "pJ/flop"), "{:.0f}", "DRAM read over FP16 operation")
a = AMDAHL.result
host_share = 1 - a.stage_fraction
v("amdahl_speedup", AMDAHL.speedup, "{:.0f}", "inference made faster, times")
v("amdahl_avg", (1 + AMDAHL.speedup) / (1 + 1), "{:.1f}", "wrong answer B: the two speedups averaged")
v("amdahl_host", host_share, "{:.2f}", "host preprocessing, fraction")
v("amdahl_inf", a.stage_fraction, "{:.2f}", "inference, fraction")
v("amdahl_host_pct", pct(host_share), "{:.0f}", "host preprocessing, percent")
v("amdahl_inf_pct", pct(a.stage_fraction), "{:.0f}", "inference, percent")
v("amdahl_new", m(a.upgraded_total_time, "percent") / 100, "{:.2f}", "new request time, fraction of the old")
v("amdahl_new_pct", m(a.upgraded_total_time, "percent"), "{:.0f}", "the same, percent")
v("amdahl_gain", a.overall_speedup, "{:.1f}", "whole-request speedup")

print("== Part IV: deployment (Lecture 3's Thai clinics)")
v("beede_ungradable", YIELD.rejected, "{:,.0f}", "photos rejected as ungradable")
v("beede_submitted", YIELD.attempted, "{:,.0f}", "photos submitted")
v("ungradable_pct", pct(YIELD.rejected_share), "{:.0f}", "ungradable share, %")

print("== Dual mandate: energy per prompt (Elsworth et al. 2025, as in Lecture 15b)")
full_wh = sum(r["comprehensive_wh"] for r in load_series("vol2/15_sustainable_ai/data/gemini_prompt_energy.csv").entries if r["comprehensive_counted"] == 1)
v("prompt_wh", full_wh, "{:.2f}", "median Gemini prompt, full boundary, Wh")
v("prompt_gco2", row(load_series("vol2/15_sustainable_ai/data/google_serving_factors.csv"), parameter="prompt_emissions_total")["value"], "{:.2f}",
  "the same prompt, g CO2e")
v("prompt_cut", row(load_series("vol2/15_sustainable_ai/data/google_serving_factors.csv"), parameter="energy_reduction_12_months")["value"], "{:.0f}",
  "energy cut over twelve months, times")

print("== Tails (Scenarios.ChatEndpointTtftMeanVersusP99; Dean and Barroso 2013 via Lecture 12; Lecture 13)")
v("ttft_mean", m(TTFT.mean, "ms"), "{:.0f}", "illustrative mean TTFT, ms")
v("ttft_p99", m(TTFT.tail_high, "ms"), "{:,.0f}", "illustrative p99 TTFT, ms")
v("ttft_gap", TTFT.gap_high, "{:.0f}", "p99 over mean")
TAS = load_series("slides/vol1/12_benchmarking/images/data/tail_at_scale_leaf_latency.csv")


def median_and_p99(leaves):
    """The median (lowest percentile) and p99 (highest) rows of one fan-out case."""
    rows = sorted((r for r in TAS.entries if r["leaves_awaited"] == leaves), key=lambda r: r["percentile"])
    return rows[0]["latency_ms"], rows[-1]["latency_ms"]


leaf_p50, leaf_p99 = median_and_p99("one random leaf")
all_p50, all_p99 = median_and_p99("all leaves")
v("leaf_p50", leaf_p50, "{:.0f}", "one leaf, median, ms")
v("leaf_p99", leaf_p99, "{:.0f}", "one leaf, p99, ms")
v("all_p50", all_p50, "{:.0f}", "all leaves, median, ms")
v("all_p99", all_p99, "{:.0f}", "all leaves, p99, ms")
v("tail_ratio", all_p99 / leaf_p99, "{:.0f}", "all-leaves p99 over one-leaf p99")
v("fan_n100", FAN100.n_components, "{:.0f}", "servers (L13)")
v("fan_avail_pct", pct(FAN100.availability), "{:.0f}", "each server on time, % (L13)")
v("fan100", pct(FAN100.all_up), "{:.1f}", "all 100 on time, % (L13)")

print("== MTBF of a 1,024-GPU job (Scenarios.GpuFailuresTraining1KThreeWeeks)")
f = FAIL.result
mttf_h = m(FAIL.component_mtbf, "hour")
v("mttf_years", m(FAIL.component_mtbf, "year"), "{:.1f}", "one GPU's MTTF, years")
v("mttf_h", mttf_h, "{:,.0f}", "one GPU's MTTF, hours")
v("n_gpu", FAIL.n_components, "{:,.0f}", "GPUs in the job")
v("mtbf_h", m(f.interval, "hour"), "{:.1f}", "job MTBF, hours")
v("mtbf_days_word", m(f.interval, "day"), "words", "the same, whole days, in prose")
v("run_h", m(FAIL.run_duration, "hour"), "{:.0f}", "three-week run, hours")
v("run_fails_word", round(f.expected_failures), "words", "failures expected in the run, in prose")

print("== Power ladder (Lecture 2's tiers) and the phone against an H100 (Conclusion)")
cloud_w, edge_w = m(POD.power, "W"), m(SPARK.tdp, "W")
mobile_w = float(Platforms.Mobile.tdp_range_w.split("-")[1])
tiny_w = m(Platforms.Tiny.power_threshold, "W")
v("cloud_mw", cloud_w / 10 ** 6, "{:.0f}", "cloud rung, MW")
v("edge_w", edge_w, "{:.0f}", "edge rung, W")
v("mobile_w", mobile_w, "{:.0f}", "mobile rung, W")
v("tiny_mw", tiny_w * 10 ** 3, "{:.0f}", "TinyML rung, mW")
v("power_orders_word", math.floor(math.log10(cloud_w / tiny_w)), "words", "whole orders of magnitude, cloud to TinyML")
phone = Hardware.Mobile.iPhone15Pro
v("phone_mem_ratio", ratio(H100.memory.capacity, Platforms.Mobile.ram), "{:.0f}", "H100 memory over a phone's")
v("phone_power_ratio", ratio(H100.tdp, phone.tdp), "{:.0f}", "H100 power over a phone's")

print("== Composed assistant (Scenarios.ComposedAssistantRequestPath)")
ret, tool_a, tool_b, gen, ver = (m(DAG.durations[k], "ms") for k in sorted(DAG.durations))
v("dag_ret", ret, "{:.0f}", "retrieval, ms")
v("dag_tool_a", tool_a, "{:.0f}", "tool A, ms")
v("dag_tool_b", tool_b, "{:.0f}", "tool B, ms")
v("dag_gen", gen, "{:.0f}", "generation, ms")
v("dag_ver", ver, "{:.0f}", "verifier, ms")
v("dag_crit", m(DAG.makespan, "ms"), "{:.0f}", "critical path, ms")
v("dag_sum", m(DAG.work, "ms"), "{:.0f}", "sum of every box, ms")

print("== Llama 3 405B interruptions (Literature.TrainingRuns, Series.Llama3Interruptions)")
QUOTED = ("Faulty GPU", "GPU HBM3 Memory", "Software Bug", "Network Switch/Cable", "Host Maintenance")
T5 = load_series("vol2/07_fault_tolerance/data/llama3_interruptions.csv").entries
unexpected, planned = TR.Llama3_405B_UnexpectedInterruptions, TR.Llama3_405B_PlannedInterruptions
days, gpus = TR.Llama3_405B_SnapshotDays, TR.Llama3_405B_Gpus16K
assert sum(r["interruption_count"] for r in T5) == unexpected
for key, name in zip(("faulty_gpu", "hbm3", "sw_bug", "network", "host_maint"), QUOTED):
    v(f"t5_{key}", row(load_series("vol2/07_fault_tolerance/data/llama3_interruptions.csv"), component=name)["interruption_count"], "{:.0f}", name)
rest = [r for r in T5 if r["component"] not in QUOTED]
v("t5_rest_rows", len(rest), "{:.0f}", "rows not quoted")
v("t5_rest_sum", sum(r["interruption_count"] for r in rest), "{:.0f}", "their counts summed")
v("l3_unexpected", unexpected, "{:.0f}", "unexpected interruptions")
v("l3_planned", planned, "{:.0f}", "planned interruptions")
v("l3_total", unexpected + planned, "{:.0f}", "all interruptions")
v("l3_days", days, "{:.0f}", "snapshot, days")
v("l3_gpus", gpus, "{:,.0f}", "H100s, up to")
v("l3_gap_h", days * m(mlsysim.core.units.Q_(1, "day"), "hour") / unexpected, "{:.1f}",
  "hours between unexpected interruptions")
v("l3_hw_pct", pct(TR.Llama3_405B_HardwareInterruptionFraction), "{:.0f}", "hardware share of unexpected, %")
v("l3_effective_pct", pct(TR.Llama3_405B_EffectiveTrainingTimeFraction), "{:.0f}", "effective training time, %")
v("l3_manual_word", TR.Llama3_405B_ManualInterventions, "words", "manual interventions, in prose")
v("model_16k_h", m(FAIL16K.result.interval, "hour"), "{:.2f}", "MTTF / 16,384, hours")

print("== Voice request (the Conclusion's registry trace; Scenarios.Gpt2XlFp16DecodeOnH100)")
kws, mcu, mnv2, dlrm = Models.Tiny.DS_CNN, Hardware.Tiny.STM32F4, Models.Vision.MobileNetV2, Models.Recommendation.DLRM
dec = GPT2_DECODE.result
v("kws_kb", m(kws.size_in_bytes(BYTES_INT8), "KB"), "{:.0f}", "KWS INT8 weights, KB")
v("mcu_flash_mb", m(mcu.memory.flash_capacity, "MiB"), "{:.0f}", "microcontroller flash, labeled MB")
v("kws_mflop", m(kws.inference_flops, "MFLOP"), "{:.0f}", "KWS work per inference, MFLOP")
v("mnv2_mflop", m(mnv2.inference_flops, "MFLOP"), "{:.0f}", "MobileNetV2 work, MFLOP")
v("mnv2_int8_mb", m(mnv2.size_in_bytes(BYTES_INT8), "MB"), "{:.1f}", "MobileNetV2 INT8 weights, MB")
v("phone_w", m(phone.tdp, "W"), "{:.0f}", "phone power envelope, W")
v("dlrm_gb", m(dlrm.model_size, "GB"), "{:.0f}", "DLRM embedding tables, GB")
v("h100_gib", m(H100.memory.capacity, "GiB"), "{:.0f}", "one H100, labeled GB")
v("gpt2_gb", m(dec.weight_bytes, "GB"), "{:.0f}", "GPT-2 XL FP16 weights per token, GB")
v("gpt2_mem_ms", m(dec.memory_time, "ms"), "{:.1f}", "memory floor, ms")
v("gpt2_ratio", dec.bound_ratio, "{:.0f}", "memory floor over compute floor")

print("== Lecture 0's questions (IEA via Literature.DatacenterElectricity; Lecture 10's phone token)")
DCE = Literature.DatacenterElectricity
v("dc_twh_2024", lit(DCE.Global2024Twh), "{:.0f}", "datacenters, 2024, TWh")
v("dc_twh_2030", lit(DCE.Global2030Twh), "{:.0f}", "datacenters, 2030 projection, TWh")
v("dc_growth", lit(DCE.Global2030Twh) / lit(DCE.Global2024Twh), "{:.1f}", "2030 over 2024")
v("token_ms", round(m(TOKEN.read_time, "ms"), -1), "{:.0f}", "7B token on a phone, ms, to the ten (L10)")
v("token_w", round(m(TOKEN.read_power, "W")), "{:.0f}", "7B DRAM power while decoding, W (L10)")

# L15 recall: the same object Lecture 15's feedback slide reads (Napkin Math 15.3)
LOOP15 = Scenarios.CreditServiceBiasFeedback
v("fb_gap0", LOOP15.launch_gap, "{:.1f}", "L15: gap at launch, points")
v("fb_alpha", LOOP15.factor, "{:.2f}", "L15: fitted feedback factor")
v("fb_gate", LOOP15.gate, "{:.0f}", "L15: release gate, points")
v("fb_fail_q", LOOP15.breach_cycle, "{:.0f}", "L15: quarter the gate fails")

if __name__ == "__main__":
    write(__file__)
