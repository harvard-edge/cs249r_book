#!/usr/bin/env python3
"""Every number on the Lecture 12 (Benchmarking, Part 1) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Hardware, Models, Datasets, Literature and its
Series of published results) and the calculations the Benchmarking chapter
also makes (the benchmarking Scenarios.*). Three files in this folder are
measurements run here and quoted on the slides (timer_demo_output.txt,
profile_step_output.txt) or a published log shown verbatim
(mlperf_v50_cisco_resnet50_server_summary.txt); this file reads them as
printed. It types no numbers: it selects values, converts units, and formats
them. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
this module, so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""

import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Literature, Scenarios = mlsysim.Hardware, mlsysim.Literature, mlsysim.Scenarios
Q_ = mlsysim.core.units.Q_
SERIES, BENCH = Literature.Series, Literature.Benchmarks
HERE = Path(__file__).resolve().parent
H100 = Hardware.Cloud.H100

CLAIM = Scenarios.DatasheetVersusSustainedEdgeChip
FANOUT = Scenarios.ServingFanout100Leaves
TAIL_GAP = Scenarios.ModelOnlyMeanVersusProductionP99
CEILING = Scenarios.ResNet50H100TrainingCeiling
BOARD = Scenarios.MobileNetV2CameraTrapBoard
AMDAHL = Scenarios.PreprocessBoundVisionPipeline
INSET = Scenarios.ModelStageInFiftyMsRequest
TTQ = Scenarios.FasterStepsMoreSteps
EOS8, EOS64 = Scenarios.EosEightH100ResNetMLPerfV40, Scenarios.EosSixtyFourH100ResNetMLPerfV40
DELL8 = EOS8.variant(system="XE9680x8H100-SXM-80GB")
SCALING = Scenarios.EosResNetStrongScalingMLPerfV40
ILLUSTRATIVE_SCALING = Scenarios.ImageClassifierEightGpuScaling
GEMINI = Scenarios.GeminiMedianPromptEnergyBoundary
KWS = Scenarios.KeywordSpottingOnEsp32S3


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    """A fraction as percent."""
    return 100 * x


def row(series, **match):
    """The one row of a Literature series whose fields equal ``match``."""
    rows = [r for r in series.entries if all(r[k] == val for k, val in match.items())]
    assert len(rows) == 1, (match, rows)
    return rows[0]


print("== Hook: a datasheet line against sustained load (Scenarios.DatasheetVersusSustainedEdgeChip)")
v("hook_tops", m(CLAIM.claimed_rate, "TOPS"), "{:.0f}", "datasheet TOPS")
v("hook_w", m(CLAIM.claimed_power, "W"), "{:.1f}", "datasheet watts")
v("hook_sus_tops", m(CLAIM.sustained_rate, "TOPS"), "{:.0f}", "sustained TOPS")
v("hook_sus_w", m(CLAIM.sustained_power, "W"), "{:.0f}", "sustained watts")
v("hook_eff", m(CLAIM.claimed_efficiency, "TOPS/W"), "{:.0f}", "datasheet TOPS/W")
v("hook_sus_eff", m(CLAIM.sustained_efficiency, "TOPS/W"), "{:.1f}", "sustained TOPS/W")
v("hook_gap", CLAIM.gap, "{:.1f}", "efficiency gap")
v("hook_gap_int", CLAIM.gap, "{:.0f}", "efficiency gap, whole (answer choice)")
v("hook_rate_fall", CLAIM.rate_fall, "{:.1f}", "throughput fell")
v("hook_power_rise", CLAIM.power_rise, "{:.0f}", "power rose")

print("== Recall: the H100 ridge (Hardware.Cloud.H100)")
v("h100_ridge", m(H100.ridge_point(), "flop/byte"), "{:.0f}", "H100 ridge, FLOP/byte")

print("== Tail: Dean and Barroso 2013, Table 1 (Literature.Series.TailAtScaleLeafLatency)")
TAS = SERIES.TailAtScaleLeafLatency


def median_and_p99(leaves):
    """The median (lowest percentile) and p99 (highest) rows of one fan-out case."""
    rows = sorted((r for r in TAS.entries if r["leaves_awaited"] == leaves), key=lambda r: r["percentile"])
    return rows[0], rows[-1]


(_l50, leaf_p99_row), (_a50, _a99) = median_and_p99("one random leaf"), median_and_p99("all leaves")
leaf_p50, leaf_p99, all_p50, all_p99 = (r["latency_ms"] for r in (_l50, leaf_p99_row, _a50, _a99))
v("leaf_p50", leaf_p50, "{:.0f}", "one leaf, median, ms")
v("leaf_p99", leaf_p99, "{:.0f}", "one leaf, p99, ms")
v("all_p50", all_p50, "{:.0f}", "all leaves, median, ms")
v("all_p99", all_p99, "{:.0f}", "all leaves, p99, ms")
v("tail_ratio", all_p99 / leaf_p99, "{:.0f}", "all-leaves p99 over one-leaf p99")
v("tail_mean", m(TAIL_GAP.mean, "ms"), "{:.0f}", "model-only mean, ms (illustrative)")
v("tail_p99_lo", m(TAIL_GAP.tail_low, "ms"), "{:.0f}", "production p99, low, ms")
v("tail_p99_hi", m(TAIL_GAP.tail_high, "ms"), "{:.0f}", "production p99, high, ms")
v("frame_budget", m(BOARD.frame_budget, "ms"), "{:.0f}", "camera trap frame deadline, ms")
million = 10 ** 6
v("p99_users", million * (1 - leaf_p99_row["percentile"] / 100), "{:,.0f}", "users past p99 per million requests")

print("== Fan-out estimate (Scenarios.ServingFanout100Leaves)")
slow = 1 - FANOUT.availability
fan10 = FANOUT.variant(n_components=10)
v("shard_fast", FANOUT.availability, "{:.2f}", "chance one shard is fast")
v("shards", FANOUT.n_components, "{:.0f}", "shards per request")
v("shard_fast_pct", pct(FANOUT.availability), "{:.0f}", "chance one shard is fast, percent")
v("shard_slow_pct", pct(slow), "{:.0f}", "chance one shard is slow, percent")
v("all_fast", FANOUT.all_up, "{:.3f}", "chance all shards are fast")
v("all_fast_pct", pct(FANOUT.all_up), "{:.1f}", "the same, percent")
v("any_slow_pct", pct(1 - FANOUT.all_up), "{:.0f}", "chance a request meets a slow shard, percent")
v("shards10", fan10.n_components, "{:.0f}", "the smaller fan-out")
v("any_slow10_pct", pct(1 - fan10.all_up), "{:.1f}", "chance at 10 shards, percent")

print("== Host timer (timer_demo.py, run here)")
_tout = (HERE / "timer_demo_output.txt").read_text()
timer_host = float(re.search(r"host timer:\s+([\d.]+) ms", _tout).group(1))
timer_sync = float(re.search(r"with synchronize:\s+([\d.]+) ms", _tout).group(1))
v("timer_host", timer_host, "{:.3f}", "host timer, ms")
v("timer_sync", timer_sync, "{:.3f}", "with synchronize, ms")
v("timer_ratio", timer_sync / timer_host, "{:.0f}", "kernel time over the host timer")

print("== Granularity inset (Scenarios.ModelStageInFiftyMsRequest)")
v("inset_speedup", INSET.speedup, "{:.0f}", "model-stage speedup")
v("inset_stage", m(INSET.stages["model"], "ms"), "{:.0f}", "model stage, ms")
v("inset_request", m(INSET.total, "ms"), "{:.0f}", "request, ms")
v("inset_saved", m(INSET.total - INSET.accelerated_total, "ms"), "{:.0f}", "time saved, ms")
v("inset_e2e", INSET.overall_speedup, "{:.1f}", "end-to-end speedup")

print("== Roofline against MLPerf (Scenarios.ResNet50H100TrainingCeiling; MLPerf Training v4.0 runs)")
v("fwd_gflop", m(CEILING.workload.inference_flops, "GFLOP"), "{:.1f}", "ResNet-50 forward, GFLOP")
v("train_factor", CEILING.training_flops_factor, "{:.0f}", "training work over the forward pass")
v("train_gflop", m(CEILING.training_flops_per_sample, "GFLOP"), "{:.1f}", "training GFLOP per image")
v("h100_peak", m(H100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "H100 dense FP16/BF16 peak, TFLOP/s")
v("ceiling_ips", round(m(CEILING.ceiling_rate, "count/second"), -2), "{:,.0f}", "ceiling, images/s (to the hundred)")
v("measured_ips", m(CEILING.sustained_rate, "count/second"), "{:,.0f}", "MLPerf v3.1 rate per H100, images/s")
v("achieved_tflops", m(CEILING.achieved_flops, "TFLOP/s"), "{:.0f}", "achieved model arithmetic, TFLOP/s")
v("mfu_pct", pct(CEILING.sustained_fraction), "{:.1f}", "MFU, percent")
v("ceiling_gap", CEILING.ceiling_gap, "{:.1f}", "ceiling over the measured rate")
v("ceiling_gap_int", CEILING.ceiling_gap, "{:.0f}", "the same, whole")
ips8, ips64 = m(EOS8.per_accelerator_rate, "count/second"), m(EOS64.per_accelerator_rate, "count/second")
v("ips8", round(ips8, -1), "{:,.0f}", "Eos 8x H100, images/s per GPU (to the ten)")
v("ips64", round(ips64, -1), "{:,.0f}", "Eos 64x H100, images/s per GPU (to the ten)")
ceiling = m(CEILING.ceiling_rate, "count/second")
v("gap_ips8", ceiling / ips8, "{:.1f}", "ceiling over the 8-GPU rate")
v("gap_ips64", ceiling / ips64, "{:.1f}", "ceiling over the 64-GPU rate")

print("== Profiler (profile_step.py, run here): Self CPU % per operator")
_prof = (HERE / "profile_step_output.txt").read_text()


def _share(name):
    return float(re.search(re.escape(name) + r"\s+([\d.]+)%", _prof).group(1))


prof_conv = _share("aten::_nnpack_spatial_convolution") + _share("aten::convolution_backward")
prof_bn = _share("aten::native_batch_norm") + _share("aten::native_batch_norm_backward")
v("prof_conv", prof_conv, "{:.1f}", "convolutions, forward and backward, percent")
v("prof_bn", prof_bn, "{:.1f}", "batch norm, forward and backward, percent")
v("prof_rest", 100 - prof_conv - prof_bn, "{:.1f}", "everything else, percent")

print("== Edge board (Scenarios.MobileNetV2CameraTrapBoard; illustrative composite)")
v("edge_fp32", m(BOARD.host_fp32_latency, "ms"), "{:.0f}", "host FP32, ms")
v("edge_host", m(BOARD.host_int8_latency, "ms"), "{:.0f}", "host INT8, ms")
v("edge_acc", m(BOARD.accelerator_latency, "ms"), "{:.0f}", "accelerator INT8, ms")
v("edge_iso", BOARD.isolated_speedup, "{:.1f}", "model-only speedup")
v("edge_e2e_host", m(BOARD.host_end_to_end, "ms"), "{:.0f}", "host end to end, ms")
v("edge_e2e_acc", m(BOARD.accelerator_end_to_end, "ms"), "{:.0f}", "accelerator end to end, ms")
v("edge_e2e", BOARD.end_to_end_speedup, "{:.0f}", "end-to-end speedup")
v("edge_e_host", m(BOARD.host_energy, "mJ"), "{:.1f}", "host active energy, mJ")
v("edge_e_acc", m(BOARD.accelerator_energy, "mJ"), "{:.0f}", "accelerator active energy, mJ")
v("edge_p_host", m(BOARD.host_power, "mW"), "{:.0f}", "host active power, mW")
v("edge_p_acc", m(BOARD.accelerator_power, "mW"), "{:.0f}", "accelerator active power, mW")

print("== Notebook: Amdahl on a vision pipeline (Scenarios.PreprocessBoundVisionPipeline)")
v("nb_prep", m(AMDAHL.stages["preprocess"], "ms"), "{:.0f}", "preprocessing, ms")
v("nb_infer", m(AMDAHL.stages["inference"], "ms"), "{:.0f}", "inference, ms")
v("nb_speedup", AMDAHL.speedup, "{:.0f}", "inference speedup")
v("nb_infer_after", m(AMDAHL.accelerated_stage, "ms"), "{:.0f}", "inference after, ms")
v("nb_after", m(AMDAHL.accelerated_total, "ms"), "{:.0f}", "total after, ms")
v("nb_total", m(AMDAHL.total, "ms"), "{:.0f}", "total before, ms")
v("nb_gain", AMDAHL.overall_speedup, "{:.1f}", "end-to-end speedup")
v("nb_limit", AMDAHL.limit, "{:.2f}", "Amdahl ceiling")

print("== Software stack: Lee et al. 2019, Tables 4 and 5 (Literature.Series.LeeMobileGpuLatency)")
LEE = SERIES.LeeMobileGpuLatency
xs_tflite = row(LEE, device="iPhone XS", framework="TFLite GPU")["latency_ms"]
xs_metal = row(LEE, device="iPhone XS", framework="Metal (MPSCNN)")["latency_ms"]
xs_coreml = row(LEE, device="iPhone XS", framework="Core ML")["latency_ms"]
px_gpu = row(LEE, device="Pixel 3", framework="TFLite GPU")["latency_ms"]
px_cpu = row(LEE, device="Pixel 3", framework="TFLite CPU")["latency_ms"]
v("xs_tflite", xs_tflite, "{:.1f}", "iPhone XS TFLite GPU, ms")
v("xs_metal", xs_metal, "{:.1f}", "iPhone XS Metal, ms")
v("xs_coreml", xs_coreml, "{:.1f}", "iPhone XS Core ML, ms")
v("px_gpu", px_gpu, "{:.1f}", "Pixel 3 TFLite GPU, ms")
v("px_cpu", px_cpu, "{:.1f}", "Pixel 3 TFLite CPU, ms")
v("lee_xs", xs_coreml / xs_tflite, "{:.1f}", "iPhone XS spread")
v("lee_xs_int", xs_coreml / xs_tflite, "{:.0f}", "the same, whole (title)")
v("lee_px", px_cpu / px_gpu, "{:.1f}", "Pixel 3 spread")

print("== Time to quality: MLPerf Training v4.0, Eos run 0 (Literature.Series.MLPerfTrainingV40EosResNetEvals)")
EVALS = SERIES.MLPerfTrainingV40EosResNetEvals.entries
target = float(BENCH.MLPerfResNet50QualityTargetTop1)
v("target_top1", pct(target), "{:.1f}", "MLPerf ResNet-50 quality target, percent")
ev_first, ev_mid = EVALS[0], EVALS[statistics.median_low(range(len(EVALS)))]
ev_late = [e for e in EVALS if e["top1"] < pct(target)][-1]      # the last evaluation below the target
v("ev_first_min", ev_first["minutes"], "{:.1f}", "first evaluation, minutes")
v("ev_mid_min", ev_mid["minutes"], "{:.1f}", "middle evaluation, minutes")
v("ev_late_min", ev_late["minutes"], "{:.1f}", "last evaluation below the target, minutes")
v("ev_first_top1", ev_first["top1"], "{:.0f}", "first evaluation, top-1 percent")
v("ev_mid_top1", ev_mid["top1"], "{:.0f}", "middle evaluation, top-1 percent")
v("ev_late_top1", ev_late["top1"], "{:.1f}", "last evaluation below the target, top-1 percent")
v("ev_last_epoch", EVALS[-1]["epoch"], "{:.0f}", "epoch of the crossing")
v("ev_last_min", EVALS[-1]["minutes"], "{:.2f}", "time to quality of run 0, minutes")
v("ev_last_top1", EVALS[-1]["top1"], "{:.2f}", "top-1 at the crossing, percent")
v("ev_every", EVALS[1]["epoch"] - EVALS[0]["epoch"], "{:.0f}", "epochs between evaluations")
v("ttq_step", TTQ.step_speedup, "{:.1f}", "self-check: step speedup")
v("ttq_steps", TTQ.steps_multiplier, "{:.1f}", "self-check: steps multiplier")
v("ttq_slow", TTQ.time_ratio, "{:.2f}", "self-check: slower by")

print("== Strong scaling: Eos 8 against 64 H100s (Scenarios.EosResNetStrongScalingMLPerfV40)")
v("score8", m(EOS8.published_score, "minute"), "{:.2f}", "8-GPU published score, min")
v("score64", m(EOS64.published_score, "minute"), "{:.2f}", "64-GPU published score, min")
v("scale_ratio", SCALING.device_ratio, "{:.0f}", "accelerator ratio")
v("scale_speed", SCALING.speedup, "{:.1f}", "speedup")
v("scale_ideal", m(SCALING.ideal_time, "minute"), "{:.2f}", "ideal 64-GPU time, min")
v("scale_ideal_short", m(SCALING.ideal_time, "minute"), "{:.1f}", "the same, one decimal (answer choice)")
v("score64_short", m(EOS64.published_score, "minute"), "{:.1f}", "64-GPU score, one decimal (answer choice)")
v("scale_eff", pct(SCALING.efficiency), "{:.0f}", "strong-scaling efficiency, percent")
v("global_batch", EOS8.global_batch, "{:,.0f}", "global batch, images")
v("epochs", EOS8.epochs, "{:.0f}", "epochs")
v("imagenet_train", m(EOS8.dataset.training_examples, "count"), "{:,.0f}", "ImageNet training images")
v("batch8", EOS8.per_accelerator_batch, "{:.0f}", "images per GPU per step, 8 GPUs")
v("batch64", EOS64.per_accelerator_batch, "{:.0f}", "images per GPU per step, 64 GPUs")
v("scale_factor", EOS8.scaling_factor, "{:.4f}", "MLPerf scaling factor")
v("illus_eff", pct(ILLUSTRATIVE_SCALING.efficiency), "{:.0f}", "chapter's illustrative efficiency, percent")
v("next_gap", pct(m(EOS8.published_score, "minute") / m(DELL8.published_score, "minute") - 1), "{:.1f}",
  "Eos score over Dell score, percent (Lecture 12b)")

print("== MLPerf Inference v5.0, ResNet-50 (Literature.Series.MLPerfInferenceV50ResNet50)")
INF = SERIES.MLPerfInferenceV50ResNet50
rtx_p90 = row(INF, result_id="5.0-0032", scenario="SingleStream")["value"]
rtx_off = row(INF, result_id="5.0-0032", scenario="Offline")["value"]
cisco_off = row(INF, result_id="5.0-0010", scenario="Offline")["value"]
cisco_srv = row(INF, result_id="5.0-0010", scenario="Server")["value"]
quanta_off = row(INF, result_id="5.0-0065", scenario="Offline")["value"]
quanta_srv = row(INF, result_id="5.0-0065", scenario="Server")["value"]
rtx_serial = m(1 / Q_(rtx_p90, "ms"), "1/second")
v("rtx_p90", rtx_p90, "{:.2f}", "RTX 4090 SingleStream p90, ms")
v("rtx_p90_long", rtx_p90, "{:.4f}", "the same, four decimals")
v("rtx_off", rtx_off, "{:,.0f}", "RTX 4090 Offline, samples/s")
v("rtx_serial", round(rtx_serial, -1), "{:,.0f}", "one-at-a-time rate, /s (to the ten)")
v("rtx_ratio", rtx_off / rtx_serial, "{:.1f}", "Offline over one at a time")
v("cisco_off", cisco_off, "{:,.0f}", "Cisco Offline")
v("cisco_srv", cisco_srv, "{:,.0f}", "Cisco Server")
v("quanta_off", quanta_off, "{:,.0f}", "Quanta Offline")
v("quanta_srv", quanta_srv, "{:,.0f}", "Quanta Server")
v("cisco_drop", pct(1 - cisco_srv / cisco_off), "{:.0f}", "Cisco Server below Offline, percent")
v("quanta_drop", pct(1 - quanta_srv / quanta_off), "{:.0f}", "Quanta Server below Offline, percent")
v("srv_spread", pct(cisco_srv / quanta_srv - 1), "{:.1f}", "Cisco over Quanta under Server, percent")

print("== The Server log (mlperf_log_summary.txt of 5.0-0010, shown verbatim)")
_log = (HERE / "mlperf_v50_cisco_resnet50_server_summary.txt").read_text()


def _logq(key, unit):
    return Q_(float(re.search(re.escape(key) + r"\s*:\s*([\d.]+)", _log).group(1)), unit)


log_p99 = _logq("99.00 percentile latency (ns)", "ns")
log_bound = _logq("target_latency (ns)", "ns")
log_min = _logq("min_duration (ms)", "ms")
v("log_p50_ms", m(_logq("50.00 percentile latency (ns)", "ns"), "ms"), "{:.2f}", "median latency, ms")
v("log_p99_ns", m(log_p99, "ns"), "{:,.0f}", "p99 latency, ns")
v("log_p99_ms", m(log_p99, "ms"), "{:.2f}", "p99 latency, ms")
v("log_bound_ms", m(log_bound, "ms"), "{:.0f}", "latency bound, ms")
v("log_p99_share", pct((log_p99 / log_bound).to("").magnitude), "{:.0f}", "p99 share of the bound, percent")
v("log_min_ms", m(log_min, "ms"), "{:,.0f}", "minimum run, ms")
v("log_min_s", m(log_min, "s"), "{:.0f}", "minimum run, s")

print("== Measurement boundary: Gemini median prompt (Scenarios.GeminiMedianPromptEnergyBoundary)")
v("gem_narrow", m(GEMINI.narrow, "Wh"), "{:.2f}", "active accelerators only, Wh")
v("gem_full", m(GEMINI.full, "Wh"), "{:.2f}", "whole serving system, Wh")
v("gem_ratio", GEMINI.ratio, "{:.1f}", "full over narrow")
v("gem_accel", pct(GEMINI.share("Active AI")), "{:.0f}", "accelerators, percent of full")
v("gem_host", pct(GEMINI.share("Active host")), "{:.0f}", "host CPU and DRAM, percent")
v("gem_idle", pct(GEMINI.share("Idle", "Data center")), "{:.0f}", "idle machines and facility, percent")

print("== Throttling: Lee et al. 2019, Figure 8; Tummalapalli et al. 2026 (Literature.Series)")
SUS = SERIES.LeeSustainedLatency
xs_cpu = row(SUS, device="iPhone XS", processor="CPU")
xs_gpu = row(SUS, device="iPhone XS", processor="GPU")
px3 = row(SUS, device="Pixel 3", processor="GPU")
v("thr_cpu0", xs_cpu["start_ms"], "{:.0f}", "iPhone XS CPU at start, ms")
v("thr_cpu1", xs_cpu["end_ms"], "{:.0f}", "iPhone XS CPU after 200 s, ms")
v("thr_gpu0", xs_gpu["start_ms"], "{:.1f}", "iPhone XS GPU at start, ms")
v("thr_gpu1", xs_gpu["end_ms"], "{:.1f}", "iPhone XS GPU after 200 s, ms")
v("thr_px", px3["start_ms"], "{:.1f}", "Pixel 3 GPU, ms")
v("thr_cpu", pct(xs_cpu["end_ms"] / xs_cpu["start_ms"] - 1), "{:.0f}", "iPhone CPU slowdown, percent")
v("thr_gpu", pct(xs_gpu["end_ms"] / xs_gpu["start_ms"] - 1), "{:.1f}", "iPhone GPU slowdown, percent (exactly 62.5)")
TOK = SERIES.TummalapalliIphoneThermal
tok = {r["state"]: r["tokens_per_s"] for r in TOK.entries}
v("tok_peak", tok["peak"], "{:.1f}", "tokens/s at peak")
v("tok_warm", tok["warm"], "{:.1f}", "tokens/s once warm")
v("tok_hot", tok["hot"], "{:.1f}", "tokens/s once hot")
v("tok_loss", pct(1 - tok["hot"] / tok["peak"]), "{:.1f}", "loss when hot, percent")

print("== MLPerf Tiny v1.2, visual wake words (Literature.Series.MLPerfTinyV12VwwEnergy)")
VWW = SERIES.MLPerfTinyV12VwwEnergy
syn_lo = row(VWW, system="syntiant_9120_0v9_30mhz")
syn_hi = row(VWW, system="syntiant_9120_1v1_98mhz")
stm = {r["board"]: r["energy_uj"] for r in VWW.entries if r["organization"] == "STMicroelectronics"}
v("syn_lo_mhz", syn_lo["clock_mhz"], "{:.1f}", "Syntiant low clock, MHz")
v("syn_lo_v", syn_lo["voltage_v"], "{:.1f}", "Syntiant low voltage, V")
v("syn_lo_uj", syn_lo["energy_uj"], "{:.0f}", "Syntiant low energy, uJ")
v("syn_hi_mhz", syn_hi["clock_mhz"], "{:.1f}", "Syntiant high clock, MHz")
v("syn_hi_v", syn_hi["voltage_v"], "{:.1f}", "Syntiant high voltage, V")
v("syn_hi_uj", syn_hi["energy_uj"], "{:.0f}", "Syntiant high energy, uJ")
v("tiny_speed", syn_hi["throughput_ips"] / syn_lo["throughput_ips"], "{:.1f}", "Syntiant faster at the high clock")
v("tiny_energy", syn_hi["energy_uj"] / syn_lo["energy_uj"], "{:.2f}", "Syntiant energy ratio")
v("stm_lo", min(stm.values()), "{:,.0f}", "lowest STM32 energy, uJ")
v("stm_mid", sorted(stm.values())[1], "{:,.0f}", "middle STM32 energy, uJ")
v("stm_hi", max(stm.values()), "{:,.0f}", "highest STM32 energy, uJ")
v("tiny_worst", max(stm.values()) / syn_lo["energy_uj"], "{:.0f}", "highest STM32 over the Syntiant low point")

print("== Duty cycle: keyword spotting (Scenarios.KeywordSpottingOnEsp32S3)")
v("kws_t_ms", m(KWS.inference_time, "ms"), "{:.1f}", "inference time, ms")
v("kws_p_active_w", m(KWS.active_power, "W"), "{:.1f}", "active power, W")
v("kws_p_active_mw", m(KWS.active_power, "mW"), "{:.0f}", "active power, mW")
v("kws_p_idle_mw", m(KWS.idle_power, "mW"), "{:.0f}", "idle power, mW")
v("kws_duty", KWS.duty, "{:.4f}", "duty cycle")
v("kws_idle_frac", 1 - KWS.duty, "{:.4f}", "idle fraction")
v("kws_duty_pct", pct(KWS.duty), "{:.1f}", "duty cycle, percent")
v("kws_active_mw", m(KWS.active_contribution, "mW"), "{:.1f}", "active share of the average, mW")
v("kws_idle_mw", m(KWS.idle_contribution, "mW"), "{:.1f}", "idle share of the average, mW")
v("kws_avg_mw", m(KWS.average_power, "mW"), "{:.1f}", "average power, mW")
v("kws_idle_share", pct(KWS.idle_share), "{:.0f}", "idle share of the drain, percent")
v("kws_cross", pct(KWS.crossover_duty), "{:.0f}", "break-even duty cycle, percent")
v("kws_sum_mw", m(KWS.active_power + KWS.idle_power, "mW"), "{:.0f}", "active plus idle power, mW")
v("kws_cut_mw", m(KWS.cut_saving, "mW"), "{:.2f}", "saving if inference energy halves, mW")
v("kws_cut_gain", pct(KWS.cut_gain), "{:.0f}", "battery gain, percent")

if __name__ == "__main__":
    write(__file__)
