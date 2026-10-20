#!/usr/bin/env python3
"""Every number on the Volume I Lecture 1 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature, and the measured series behind
the data plots) and the calculations (Scenarios.*, each backed by a solver);
the chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex.

One exception is measured here, not in MLSysIM: the GPT-2 decode profile in
data/gpt2_profile.txt is this deck's own torch.profiler run
(images/profile_gpt2.py), read as the profiler printed it.

Run from this folder: python3 calcs.py
"""

import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Models, Literature = mlsysim.Hardware, mlsysim.Models, mlsysim.Literature
Scenarios, Datasets = mlsysim.Scenarios, mlsysim.Datasets
units = mlsysim.core.units
from mlsysim.engine import calibration  # noqa: E402
from mlsysim.engine.solvers.quality import season_label  # noqa: E402
from mlsysim.physics import calc_decoder_matmul_weights, memory_from_params  # noqa: E402

HERE = Path(__file__).resolve().parent
GPT3, GPT4, ALEXNET = Models.Language.GPT3, Models.Language.GPT4, Models.Vision.AlexNet
RUNS, QUALITY, GAMES = Literature.TrainingRuns, Literature.PredictionQuality, Literature.GamePlaying
SERIES = Literature.Series
RUN, RUN_HI = Scenarios.Gpt3TrainingTime1KA100, Scenarios.Gpt3TrainingTime1KA100HigherUtilization
SPACE = Scenarios.ImageNetInputSpace
PED, SKEW = Scenarios.PedestrianRecallWinterShift, Scenarios.AccountAgeUnitSkew
RECHT, GFT = Scenarios.ImageNetV2Shift, Scenarios.GoogleFluTrendsDrift
NAPKIN, BUDGET = Scenarios.IronLawMemoryBoundExample, Scenarios.IronLawComputeHeavyExample
RESNET, GPT2XL = Scenarios.ResNet50Batch64StepH100, Scenarios.Gpt2XlDecodeStepH100
AMDAHL = Scenarios.ModelSpeedupRequestPath


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def mantissa_exponent(x):
    """(mantissa, exponent) with 1 <= mantissa < 10."""
    e = math.floor(math.log10(x))
    return x / 10 ** e, e


# ---------------------------------------------------------------------------
# Measured series, shared with make_figs.py
# ---------------------------------------------------------------------------
def flu_weeks():
    """Weekly Google Flu Trends estimate and CDC count, percent of doctor visits."""
    rows = SERIES.GoogleFluTrendsVsCdc.entries
    gft = {int(r["epiweek"]): r[GFT.proxy_key] for r in rows}
    cdc = {int(r["epiweek"]): r[GFT.reference_key] for r in rows}
    return gft, cdc


def recht_rows():
    """(model, original top-1, ImageNetV2 top-1) per model, Recht et al. 2019."""
    return [(r["model"], r[RECHT.original_key], r[RECHT.new_key]) for r in RECHT.rows]


def mlperf_rows():
    """MLPerf Inference v4.0 Llama 2 70B Offline rows, H100 system first."""
    return SERIES.MLPerfInferenceV4Llama2Offline.entries


print("== Recall and efficiency: GPT-3 on 1,024 A100s (cluster training-time scenarios)")
ops_mant, ops_exp = mantissa_exponent(m(GPT3.training_ops, "flop"))
v("gpt3_ops_mant", ops_mant, "{:.2f}", "GPT-3 training work, x 10^exp FLOPs")
v("gpt3_ops_exp", ops_exp, "{:.0f}", "its power of ten")
v("n_gpu", RUN.n_devices, "{:,.0f}", "cluster accelerators")
v("a100_tflops", m(RUN.hardware.compute.peak_flops, "TFLOP/s"), "{:.0f}", "A100 16-bit peak, TFLOP/s")
v("eta_base", RUN.efficiency, "{:.2f}", "utilization")
v("eta_base_pct", 100 * RUN.efficiency, "{:.0f}", "the same, percent")
v("eta_opt_pct", 100 * RUN_HI.efficiency, "{:.0f}", "higher utilization, percent")
v("days_base", m(RUN.training_time, "day"), "{:.0f}", "days at the base utilization")
v("days_opt", m(RUN_HI.training_time, "day"), "{:.0f}", "days at the higher utilization")

print("== Hook and drift: Google Flu Trends against the CDC (proxy-drift scenario)")
g = GFT.result
v("gft_peak", g.peak_proxy, "{:.1f}", "2012-13 peak of the estimate, % of visits")
v("cdc_peak", g.peak_reference, "{:.1f}", "2012-13 peak of the CDC count, % of visits")
v("gft_peak_ratio", g.peak_ratio, "{:.1f}", "peak ratio")
v("gft_week_ratio", g.max_week_ratio, "{:.1f}", "largest single-week ratio")
v("gft_miss_weeks", QUALITY.GoogleFluTrendsOverestimateWeeks, "{:.0f}", "weeks too high (Lazer et al.)")
v("gft_window_weeks", QUALITY.GoogleFluTrendsOverestimateWindowWeeks, "{:.0f}", "of weeks from August 2011")
v("gft_stable_pct", 100 * g.stable_max_deviation, "{:.0f}", "largest seasonal deviation before drift, %")
v("gft_ratio_first_drift", g.season_ratios[season_label(GFT.first_drift_year)], "{:.2f}", "2011-12 ratio")
v("gft_ratio_peak", g.season_ratios[g.peak_season], "{:.2f}", "2012-13 ratio")

print("== No test set covers the inputs (input-space scenario)")
v("img_w", SPACE.dataset.image_width, "{:.0f}", "ImageNet image width, pixels")
v("img_h", SPACE.dataset.image_height, "{:.0f}", "image height, pixels")
v("img_channels_word", SPACE.dataset.image_channels, "words", "color channels, in prose")
v("pixel_levels", SPACE.levels, "{:.0f}", "values per 8-bit channel element")
v("input_elements", SPACE.elements, "{:,.0f}", "bytes per image")
v("input_digits", SPACE.digits, "{:,.0f}", "digits in the count of possible images")
v("imagenet_val", SPACE.test_examples, "{:,.0f}", "ImageNet validation images")

print("== AlexNet (registry and Literature)")
alex_weights = memory_from_params(ALEXNET.parameters, units.BYTES_FP32 / units.param)
v("alexnet_params_m", m(ALEXNET.parameters, "Mparam"), "{:.0f}", "AlexNet parameters, millions")
v("fp32_bytes", m(units.BYTES_FP32, "byte"), "{:.0f}", "bytes per 32-bit weight")
v("alexnet_weights_mb", m(alex_weights, "MB"), "{:.0f}", "AlexNet weights, MB")
v("gtx580_gb", m(Hardware.Workstation.GTX580_3GB.memory.capacity, "GiB"), "{:.0f}", "GTX 580 memory, GB (branded)")
v("alexnet_top5", QUALITY.AlexNetIlsvrc2012Top5ErrorPct, "{:.1f}", "AlexNet top-5 error, %")
v("runner_up_top5", QUALITY.Ilsvrc2012RunnerUpTop5ErrorPct, "{:.1f}", "second-best top-5 error, %")
v("imagenet_train_m", m(Datasets.ImageNet.training_examples, "count") / 10 ** 6, "{:.1f}",
  "ImageNet training images, millions")

print("== Bitter lesson (Literature)")
v("deep_blue_chips", GAMES.DeepBlueChessChips, "{:.0f}", "Deep Blue chess chips")
v("deep_blue_low_m", GAMES.DeepBlueSearchRateLowMillions, "{:.0f}", "positions per second, low, millions")
v("deep_blue_high_m", GAMES.DeepBlueSearchRateHighMillions, "{:.0f}", "positions per second, high, millions")
v("alphago_days_word", GAMES.AlphaGoZeroTrainingDaysToBeatLee, "words", "days of self-play, in prose")
v("alphago_wins", GAMES.AlphaGoZeroWinsVsLee, "{:.0f}", "games won against AlphaGo Lee")
v("alphago_losses", GAMES.AlphaGoZeroLossesVsLee, "{:.0f}", "games lost")

print("== GPU-days (registry)")
alex_gpu_days = RUNS.AlexNetGpus * RUNS.AlexNetTrainingDaysHigh
gpt3_gpu_days = m(GPT3.training_accelerators_ref, "count") * m(GPT3.training_days_ref, "day")
v("alexnet_gpus_word", RUNS.AlexNetGpus, "words", "AlexNet GPUs, in prose")
v("alexnet_days_word", RUNS.AlexNetTrainingDaysHigh, "words", "AlexNet training days, in prose")
v("alexnet_gpu_days", alex_gpu_days, "{:.0f}", "AlexNet GPU-days")
v("gpt3_gpu_years", gpt3_gpu_days / m(units.Q_(1, "year"), "day"), "{:.0f}", "GPT-3 V100 GPU-years")
v("gpt3_gpu_days", gpt3_gpu_days, "{:,.0f}", "GPT-3 GPU-days")
v("gpt4_gpu_days_m", GPT4.training_gpu_days / 10 ** 6, "{:.1f}", "GPT-4-class GPU-days, millions")
v("gpt4_gpus", m(GPT4.training_accelerators_ref, "count"), "{:,.0f}", "GPT-4-class GPUs")
v("gpt4_days", m(GPT4.training_days_ref, "day"), "{:.0f}", "GPT-4-class days")
opt = Scenarios.Gpt3ComputeOptimal
budget_mant, budget_exp = mantissa_exponent(opt.budget_flops)
v("chin_k", opt.flops_per_param_token, "{:.0f}", "FLOPs per parameter per token")
v("gpt3_params_b", m(GPT3.parameters, "Gparam"), "{:.0f}", "GPT-3 parameters, billions")
v("gpt3_tokens_b", m(GPT3.training_tokens, "count") / 10 ** 9, "{:.0f}", "GPT-3 training tokens, billions")
v("gpt3_6nd_mant", budget_mant, "{:.2f}", "6ND, x 10^exp")
v("gpt3_6nd_exp", budget_exp, "{:.0f}", "its power of ten")

print("== Degradation: pedestrian recall (recall-shift scenario)")
v("recall_before", 100 * PED.recall_before, "{:.0f}", "recall before, %")
v("recall_after", 100 * PED.recall_after, "{:.0f}", "recall after, %")

print("== ImageNetV2 (shifted-test-set scenario, Recht et al. 2019)")
r = RECHT.result
v("recht_models", r.model_count, "{:.0f}", "models")
v("recht_networks", r.model_count - r.baseline_count, "{:.0f}", "neural networks")
v("recht_baselines_word", r.baseline_count, "words", "Fisher-vector baselines, in prose")
v("recht_net_gap_min", r.network_gap_min, "{:.1f}", "smallest network drop, points")
v("recht_net_gap_max", r.network_gap_max, "{:.1f}", "largest network drop, points")
v("recht_gap_min_round", r.gap_min, "{:.0f}", "smallest drop of any model, whole points")
v("recht_gap_max_round", r.gap_max, "{:.0f}", "largest drop of any model, whole points")
v("recht_slope", r.slope, "{:.2f}", "least-squares slope")
v("recht_icpt_neg", -r.intercept, "{:.1f}", "minus the intercept, points")

print("== Skew: account age in days and seconds (unit-skew scenario)")
v("skew_days", m(SKEW.value, "day"), "{:.0f}", "account age, days")
v("skew_seconds", m(SKEW.serving_value, "s"), "{:,.0f}", "the same, seconds")
v("skew_seconds_m", m(SKEW.serving_value, "s") / 10 ** 6, "{:.1f}", "the same, millions of seconds")
v("skew_ratio", SKEW.skew_ratio, "{:,.0f}", "seconds per day")

print("== Napkin Math 1.1 (iron-law scenarios)")
n = NAPKIN.result
v("nap_gb", m(NAPKIN.data_volume, "GB"), "{:.0f}", "bytes moved, GB")
v("nap_bw", m(NAPKIN.bandwidth, "GB/s"), "{:.0f}", "bandwidth, GB/s")
v("nap_gflop", m(NAPKIN.operations, "GFLOP"), "{:.0f}", "work, GFLOP")
v("nap_rate", m(NAPKIN.effective_rate, "GFLOP/s"), "{:.0f}", "effective rate, GFLOP/s")
v("nap_over", m(NAPKIN.overhead, "ms"), "{:.0f}", "overhead, ms")
v("nap_bw_gain", NAPKIN.memory_speedup, "{:.0f}", "bandwidth multiple")
v("nap_mem", m(n.memory_time, "ms"), "{:.0f}", "memory term, ms")
v("nap_comp", m(n.compute_time, "ms"), "{:.0f}", "compute term, ms")
v("nap_total", m(n.sequential_time, "ms"), "{:.0f}", "sequential total, ms")
v("nap_mem2", m(n.upgraded_memory_time, "ms"), "{:.0f}", "memory term with more bandwidth, ms")
v("nap_total2", m(n.upgraded_sequential_time, "ms"), "{:.0f}", "sequential total with more bandwidth, ms")
v("nap_naive", m(n.naive_upgraded_time, "ms"), "{:.1f}", "naive halving, ms")
v("nap_cut_pct", 100 * n.sequential_reduction, "{:.0f}", "cut from more bandwidth, %")
v("nap_naive_cut_pct", 100 * (1 - ratio(n.naive_upgraded_time, n.sequential_time)), "{:.0f}",
  "the cut students expect, %")
v("nap_overlap", m(n.overlapped_time, "ms"), "{:.0f}", "overlapped bound, ms")
v("nap_overlap2", m(n.upgraded_overlapped_time, "ms"), "{:.0f}", "overlapped bound after, ms")
b = BUDGET.result
v("bud_mem", m(b.memory_time, "s"), "{:.0f}", "chapter example: movement, s")
v("bud_comp", m(b.compute_time, "s"), "{:.0f}", "computation, s")
v("bud_over", m(b.overhead, "s"), "{:.0f}", "overhead, s")
v("bud_total2", m(b.upgraded_sequential_time, "s"), "{:.0f}", "total with twice the compute rate, s")
v("bud_naive", m(b.naive_upgraded_time, "s"), "{:.1f}", "naive halving, s")

print("== Iron-law bars (single-node scenarios on one H100)")
for key, sc in (("rn", RESNET), ("g2", GPT2XL)):
    p, il = sc.result, sc.iron_law
    v(f"{key}_mem", m(p.latency_memory, "ms"), "{:.2f}", f"{sc.workload.name}: memory term, ms")
    v(f"{key}_comp", m(p.latency_compute, "ms"), "{:.2f}" if key == "rn" else "{:.3f}", "compute term, ms")
    v(f"{key}_over", m(p.latency_overhead, "ms"), "{:.2f}", "overhead, ms")
    v(f"{key}_mem2", m(il.upgraded_memory_time, "ms"), "{:.2f}", "memory term with twice the bandwidth, ms")
    v(f"{key}_total", m(il.sequential_time, "ms"), "{:.2f}", "sum of the three, ms")
    v(f"{key}_total2", m(il.upgraded_sequential_time, "ms"), "{:.2f}", "sum with twice the bandwidth, ms")
    v(f"{key}_cut_pct", 100 * il.sequential_reduction, "{:.0f}", "cut from twice the bandwidth, %")
    v(f"{key}_ai", p.arithmetic_intensity.magnitude, "{:,.1f}" if key == "g2" else "{:,.0f}",
      "arithmetic intensity, FLOP/byte")
v("rn_traffic_mb", m(RESNET.memory_traffic, "MB"), "{:,.0f}", "ResNet-50 bytes moved, MB")
v("rn_batch", RESNET.batch_size, "{:.0f}", "ResNet-50 batch")
v("rn_overlapped", m(RESNET.result.latency, "ms"), "{:.2f}", "ResNet-50 overlapped total, ms")
v("g2_params_b", m(GPT2XL.workload.parameters, "Gparam"), "{:.1f}", "GPT-2 XL parameters, billions")
v("g2_weights_gb", m(GPT2XL.workload.size_in_bytes(GPT2XL.precision), "GB"), "{:.0f}", "GPT-2 XL 16-bit weights, GB")
v("sn_eta", RESNET.efficiency, "{:.1f}", "utilization")
v("bw_gain", RESNET.memory_speedup, "{:.0f}", "bandwidth multiple")
v("launch_us", m(units.Q_(calibration.FRAMEWORK_LAYER_TAX_MS, "ms"), "us"), "{:.0f}",
  "per-layer launch overhead, microseconds")

print("== MLPerf Inference v4.0, Llama 2 70B Offline (Literature series and registry)")
h100_run, h200_run, cts_run = mlperf_rows()
h100, h200 = Hardware.Cloud.H100, Hardware.Cloud.H200
v("mlperf_h100_tps", h100_run["tokens_per_s"], "{:.1f}", "H100 system tokens/s, as logged")
v("mlperf_h200_tps", h200_run["tokens_per_s"], "{:.1f}", "H200 system tokens/s, as logged")
for key, run in (("h100", h100_run), ("h200", h200_run), ("cts", cts_run)):
    v(f"mlperf_{key}_tps_c", run["tokens_per_s"], "{:,.0f}", f"{run['system']} tokens/s, whole")
v("mlperf_h200_speedup", h200_run["tokens_per_s"] / h100_run["tokens_per_s"], "{:.2f}", "H200 over H100")
v("mlperf_cts_speedup", cts_run["tokens_per_s"] / h100_run["tokens_per_s"], "{:.2f}", "H200 CTS over H100")
v("mlperf_gpus_word", h100_run["accelerators"], "words", "GPUs per system, in prose")
v("h_tflops", m(h100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "H100 and H200 dense FP16, TFLOP/s")
v("compute_ratio", ratio(h200.compute.peak_flops, h100.compute.peak_flops), "{:.2f}", "dense FP16 ratio")
v("h200_bw", m(h200.memory.bandwidth, "TB/s"), "{:.1f}", "H200 bandwidth, TB/s")
v("h100_bw", m(h100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 bandwidth, TB/s")
v("bw_ratio", ratio(h200.memory.bandwidth, h100.memory.bandwidth), "{:.2f}", "bandwidth ratio")
v("h200_gb", m(h200.memory.capacity, "GiB"), "{:.0f}", "H200 memory, GB (branded)")
v("h100_gb", m(h100.memory.capacity, "GiB"), "{:.0f}", "H100 memory, GB (branded)")
v("cap_ratio", ratio(h200.memory.capacity, h100.memory.capacity), "{:.2f}", "capacity ratio")
v("h_tdp", m(h100.tdp, "W"), "{:.0f}", "H100 and H200 SXM power, W")

print("== GPT-2 decode profile (this deck's measurement, data/gpt2_profile.txt)")
prof = (HERE / "data" / "gpt2_profile.txt").read_text()


def self_ms(op):
    """Self CPU time of one profiler row, in ms."""
    value, unit = re.search(rf"^\s*{re.escape(op)}\s+[\d.]+%\s+([\d.]+)(ms|us)", prof, re.M).groups()
    return float(value) / (1 if unit == "ms" else 10 ** 3)


def calls(op):
    """Call count of one profiler row (the last column)."""
    return int(re.search(rf"^\s*{re.escape(op)}\s.*?(\d+)\s*$", prof, re.M).group(1))


gpt2 = Models.Language.GPT2_Small
w = calc_decoder_matmul_weights(gpt2.layers, gpt2.hidden_dim, gpt2.vocab_size)
fp32 = m(units.BYTES_FP32, "byte")
t_addmm, t_mm, t_step = self_ms("aten::addmm"), self_ms("aten::mm"), self_ms("decode_step")
t_total = float(re.search(r"Self CPU time total:\s*([\d.]+)ms", prof).group(1))
v("gpt2_params_m", m(gpt2.parameters, "Mparam"), "{:.0f}", "GPT-2 small parameters, millions")
v("gpt2_layers", gpt2.layers, "{:.0f}", "GPT-2 small blocks")
v("addmm_calls", calls("aten::addmm"), "{:.0f}", "addmm calls (profiler)")
v("mm_calls", calls("aten::mm"), "{:.0f}", "mm calls (profiler)")
v("block_mb", w["block"] * fp32 / 10 ** 6, "{:.0f}", "block weight bytes, MB")
v("head_mb", w["output"] * fp32 / 10 ** 6, "{:.0f}", "output-layer weight bytes, MB")
v("addmm_ms", t_addmm, "{:.2f}", "addmm self time, ms")
v("mm_ms", t_mm, "{:.2f}", "mm self time, ms")
v("addmm_gbs", w["block"] * fp32 / (t_addmm / 10 ** 3) / 10 ** 9, "{:.0f}", "addmm weight bytes per second, GB/s")
v("mm_gbs", w["output"] * fp32 / (t_mm / 10 ** 3) / 10 ** 9, "{:.0f}", "mm weight bytes per second, GB/s")
v("matmul_share", 100 * (t_addmm + t_mm) / t_total, "{:.0f}", "matrix multiplies share of the step, %")
v("step_self_share", 100 * t_step / t_total, "{:.0f}", "decode_step self time share, %")
check_matrices = w["block_matrices"] == calls("aten::addmm")
if not check_matrices:
    raise SystemExit("profile addmm count does not match the decoder's block matrices")

print("== Amdahl (request-path scenario)")
a = AMDAHL.result
v("am_pre", m(AMDAHL.stages["preprocess"], "ms"), "{:.0f}", "preprocessing, ms")
v("am_model", m(AMDAHL.stages["model"], "ms"), "{:.0f}", "model, ms")
v("am_post", m(AMDAHL.stages["postprocess"], "ms"), "{:.0f}", "postprocessing, ms")
v("am_total", m(a.total_time, "ms"), "{:.0f}", "total before, ms")
v("am_model2", m(a.upgraded_stage_time, "ms"), "{:.0f}", "model after, ms")
v("am_total2", m(a.upgraded_total_time, "ms"), "{:.0f}", "total after, ms")
v("am_untouched", m(a.untouched_time, "ms"), "{:.0f}", "untouched time, ms")
v("am_speedup", AMDAHL.speedup, "{:.0f}", "model speedup")
v("am_speedup_word", AMDAHL.speedup, "words", "the same, in prose")
v("am_cut_pct", 100 * a.end_to_end_reduction, "{:.0f}", "end-to-end cut, %")
v("am_model_cut_pct", 100 * a.stage_reduction, "{:.0f}", "model-stage cut, %")
v("am_share_pct", 100 * a.stage_fraction, "{:.0f}", "model share of the request, %")
v("am_overall", a.overall_speedup, "{:.1f}", "overall speedup")
v("am_bound", a.speedup_bound, "{:.1f}", "bound with an infinitely fast model")

print("== Efficiency (Literature)")
efficientnet = Literature.AlgorithmicEfficiency.factor("EfficientNet-B0")
trend = {e["model"]: e["flops"] for e in Literature.ComputeTrend.entries}
v("algo_mult", efficientnet, "{:.0f}", "EfficientNet compute reduction at AlexNet accuracy")
v("growth_exp", round(math.log10(trend["GPT-4-class"] / trend["AlexNet"])), "{:.0f}",
  "orders of magnitude of training-compute growth")
v("growth_exp_word", round(math.log10(trend["GPT-4-class"] / trend["AlexNet"])), "words", "the same, in prose")
_fit = np.polyfit([e["year"] for e in Literature.ComputeTrend.entries],
                  np.log2([e["flops"] for e in Literature.ComputeTrend.entries]), 1)[0]
v("growth_doubling_mo", mlsysim.core.units.Q_(1 / _fit, "year").to("month").magnitude, "{:.0f}", "least-squares doubling time of the plotted runs, months (the chapter's caption fit)")

if __name__ == "__main__":
    write(__file__)
