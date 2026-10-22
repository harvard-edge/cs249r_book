#!/usr/bin/env python3
"""Every number on the Lecture 12b (Benchmarking, Part 2) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Hardware, Datasets, Literature and its Series of
published results) and the calculations the Benchmarking chapter also makes
(the benchmarking Scenarios.*: MLPerf run scores, Welch intervals, run and
sample counts, the paired accuracy comparison, run order on a warming device,
the wider beam, binned calibration, and the INT8 acceptance gates). Two
sources in this folder are read as published: the MLPerf system description
files (mlperf_v40_systems/, compared by sysdiff.py). This file types no
numbers: it selects values, converts units, and formats them. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex. make_figs.py imports this module, so figures
and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402
import sysdiff  # noqa: E402

Hardware, Literature, Scenarios, Datasets = mlsysim.Hardware, mlsysim.Literature, mlsysim.Scenarios, mlsysim.Datasets
Q_ = mlsysim.core.units.Q_
BENCH = Literature.Benchmarks
ROB, PQ = Literature.RobustnessStudies, Literature.PredictionQuality

EOS = Scenarios.EosEightH100ResNetMLPerfV40
DELL = Scenarios.DellEightH100ResNetMLPerfV40
H200 = Scenarios.NvidiaEightH200ResNetMLPerfV40
N8 = Scenarios.EosSixtyFourH100ResNetMLPerfV40
W_DELL = Scenarios.EosAgainstDellResNetMLPerfV40
W_DELL_ALL = Scenarios.EosAgainstDellAllRunsResNetMLPerfV40
W_H200 = Scenarios.EosAgainstH200ResNetMLPerfV40
PLAN = Scenarios.EosResNetRunCountPlanning
SAMPLING = Scenarios.ImageNetValidationSamplingNoise
SIZE_ACC, SIZE_P90, SIZE_P99 = (Scenarios.AccuracyWithinOnePointSampleSize, Scenarios.MLPerfP90QuerySampleSize,
                                Scenarios.MLPerfP99QuerySampleSize)
VOTE = Scenarios.NinetyFiveAgainstNinetyFourOnSameImages
SEEDS = Scenarios.EosResNetTwoSeedsSameValidationSet
ORDER = Scenarios.WarmingPhoneRunOrder
BEAM = Scenarios.WiderBeamSearchTranslation
BINS = Scenarios.ThreeBinReliabilityDiagram
GATES = Scenarios.MobileNetV2Int8CameraTrapGates
SPACE = Scenarios.ImageNetInputSpace
SHIFT = Scenarios.ImageNetV2Shift
CLAIM = Scenarios.DatasheetVersusSustainedEdgeChip
BOARD = Scenarios.MobileNetV2CameraTrapBoard

# One row per submission on the noise and seed figures: scenario, label, color.
SYSTEMS = [(EOS, "Eos, 8x H100", "memory"), (DELL, "Dell, 8x H100", "compute"),
           (H200, "NVIDIA, 8x H200", "data"), (N8, "Eos, 64x H100", "network")]
ALL_RUNS = load_series("slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_resnet_runs.csv").entries
TARGET = float(BENCH.MLPerfResNet50QualityTargetTop1)


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    """A fraction as percent."""
    return 100 * x


def span_pct(s):
    """Spread of a submission's converged runs, percent of its median."""
    return pct((max(s.run_minutes) - min(s.run_minutes)) / s.median_minutes)


print("== Hook: published scores (Scenarios.*ResNetMLPerfV40; mean of the middle three x scaling factor)")
eos_score, dell_score = m(EOS.published_score, "minute"), m(DELL.published_score, "minute")
v("eos_score", eos_score, "{:.2f}", "Eos published score, min")
v("dell_score", dell_score, "{:.2f}", "Dell published score, min")
v("hook_gap_pct", pct(eos_score / dell_score - 1), "{:.1f}", "Eos score over Dell score, percent")
v("dell_spread", max(DELL.run_minutes) - min(DELL.run_minutes), "{:.2f}", "Dell converged runs, spread, min")
v("eos_spread", max(EOS.run_minutes) - min(EOS.run_minutes), "{:.2f}", "Eos runs, spread, min")
v("target_top1", pct(TARGET), "{:.1f}", "MLPerf ResNet-50 quality target, percent")
v("epochs", EOS.epochs, "{:.0f}", "epochs every run trained")
v("scale_factor", EOS.scaling_factor, "{:.4f}", "MLPerf scaling factor")
dell_fail = next(r for r in DELL.rows if r["status"] != "success")
v("dell_fail_top1", dell_fail["top1"], "{:.3f}", "Dell's missed run, last top-1, percent")
v("dell_fail_frac", dell_fail["top1"] / 100, "{:.5f}", "the same as logged (fraction)")
v("n_runs", len(ALL_RUNS), "{:.0f}", "runs in the four submissions")
v("n_systems", len(SYSTEMS), "{:.0f}", "submissions")

print("== Recall: Lecture 12's hook (Scenarios.DatasheetVersusSustainedEdgeChip)")
v("recall_tops", m(CLAIM.claimed_rate, "TOPS"), "{:.0f}", "datasheet TOPS")
v("recall_w", m(CLAIM.claimed_power, "W"), "{:.1f}", "datasheet watts")
v("recall_sus_tops", m(CLAIM.sustained_rate, "TOPS"), "{:.0f}", "sustained TOPS")
v("recall_sus_w", m(CLAIM.sustained_power, "W"), "{:.0f}", "sustained watts")
v("recall_eff", m(CLAIM.claimed_efficiency, "TOPS/W"), "{:.0f}", "datasheet TOPS/W")
v("recall_sus_eff", m(CLAIM.sustained_efficiency, "TOPS/W"), "{:.1f}", "sustained TOPS/W")

print("== The log: Eos result_0 and Dell result_4 (local CSV series MLPerf Training v4.0)")
EVALS = load_series("slides/vol1/12_benchmarking/images/data/mlperf_training_v4_0_eos_resnet_result0_evals.csv").entries
eos0 = next(r for r in EOS.rows if r["run"] == 0)
v("log_seed", eos0["seed"], "{:.0f}", "Eos run 0 seed")
v("global_batch", EOS.global_batch, "{:.0f}", "global batch, as logged")
v("ev_first_frac", EVALS[0]["top1"] / 100, "{:.5f}", "first evaluation, as logged (fraction)")
v("ev_first_epoch", EVALS[0]["epoch"], "{:.0f}", "first evaluation epoch")
v("ev_between", len(EVALS[1:-1]), "{:.0f}", "evaluations between the first and the last")
v("ev_every", EVALS[1]["epoch"] - EVALS[0]["epoch"], "{:.0f}", "epochs between evaluations")
v("ev_last_epoch", EVALS[-1]["epoch"], "{:.0f}", "last evaluation epoch")
v("log_eos_frac", eos0["top1"] / 100, "{:.5f}", "Eos run 0 last top-1, as logged (fraction)")
v("log_eos_top1", eos0["top1"], "{:.3f}", "Eos run 0 last top-1, percent")
v("log_eos_ms", m(Q_(eos0["minutes"], "minute"), "ms"), "{:,.0f}", "run_stop - run_start, ms")
v("log_eos_min", eos0["minutes"], "{:.4f}", "the same, min")

print("== Noise: 20 runs relative to each system's median (converged runs)")
for s, key in ((EOS, "eos"), (DELL, "dell"), (H200, "h200"), (N8, "n8")):
    v(f"cv_{key}", pct(s.cv), "{:.2f}", f"{key}: CV of converged runs, percent")
v("eos_span", span_pct(EOS), "{:.2f}", "Eos span, percent of median")
v("eos_span_int", span_pct(EOS), "{:.0f}", "the same, whole (title)")
v("others_span", max(span_pct(s) for s in (DELL, H200, N8)), "{:.1f}", "largest span of the other three, percent")
v("dell_fail_pct", pct(DELL.failed_run_minutes[0] / DELL.median_minutes - 1), "{:.1f}", "Dell missed run over median, percent")
v("n8_fail_pct", pct(N8.failed_run_minutes[0] / N8.median_minutes - 1), "{:.1f}", "64-GPU missed run over median, percent")
v("cv_ratio", EOS.cv / DELL.cv, "{:.1f}", "Eos CV over Dell CV")
v("rtx_cv", pct(float(BENCH.TummalapalliRtx4050DecodeSdTokS) / float(BENCH.TummalapalliRtx4050DecodeMeanTokS)),
  "{:.1f}", "RTX 4050 decode CV, percent (Tummalapalli et al.)")

print("== Seeds: top-1 at the last evaluation, 20 runs (Scenarios.ImageNetValidationSamplingNoise)")
acc = [r["top1"] for r in ALL_RUNS]
v("acc_lo", min(acc), "{:.2f}", "lowest top-1, percent")
v("acc_hi", max(acc), "{:.2f}", "highest top-1, percent")
v("acc_spread", max(acc) - min(acc), "{:.2f}", "spread, all runs, points")
v("acc_spread_eos", max(EOS.top1) - min(EOS.top1), "{:.2f}", "spread, Eos's five, points")
v("acc_mean_int", pct(SAMPLING.rate), "{:.0f}", "mean top-1, whole percent")
v("val_images", SAMPLING.n_samples, "{:,.0f}", "ImageNet validation images")
v("binom_hw", pct(SAMPLING.half_width), "{:.2f}", "binomial 95% half-width, points")

print("== Run rule on Dell's five runs")
v("dell_fail_min", DELL.failed_run_minutes[0], "{:.2f}", "Dell's missed run, min")
v("dell_middle", DELL.score_minutes, "{:.3f}", "mean of the middle three, min")

print("== Samples a number needs (Scenarios.*SampleSize; MLPerf Inference rules)")
v("z95", SIZE_ACC.z, "{:.2f}", "z at 95 percent")
v("z99", SIZE_P90.z, "{:.3f}", "z at 99 percent")
v("acc_p", SIZE_ACC.rate, "{:.2f}", "accuracy p")
v("acc_p_pct", pct(SIZE_ACC.rate), "{:.0f}", "accuracy p, percent")
v("acc_q", 1 - SIZE_ACC.rate, "{:.2f}", "accuracy 1 - p")
v("acc_m", SIZE_ACC.half_width, "{:.2f}", "accuracy margin")
v("acc_m_pts", pct(SIZE_ACC.half_width), "{:.0f}", "accuracy margin, points")
v("acc_conf", pct(SIZE_ACC.confidence), "{:.0f}", "accuracy confidence, percent")
v("p90_p", SIZE_P90.rate, "{:.2f}", "p90 p")
v("p90_m", SIZE_P90.half_width, "{:.3f}", "p90 margin")
v("p90_m_pts", pct(SIZE_P90.half_width), "{:.1f}", "p90 margin, points")
v("p99_p", SIZE_P99.rate, "{:.2f}", "p99 p")
v("p99_m", SIZE_P99.half_width, "{:.4f}", "p99 margin")
v("p99_m_pts", pct(SIZE_P99.half_width), "{:.2f}", "p99 margin, points")
v("q_conf", pct(SIZE_P90.confidence), "{:.0f}", "query confidence, percent")
v("n_acc", SIZE_ACC.samples_needed, "{:,.0f}", "images for accuracy +/- 1 point")
v("n_p90", SIZE_P90.samples_needed, "{:,.0f}", "queries for p90")
v("n_p99", SIZE_P99.samples_needed, "{:,.0f}", "queries for p99")
v("min_p90", SIZE_P90.published_minimum, "{:,.0f}", "MLPerf minimum, p90")
v("min_p99", SIZE_P99.published_minimum, "{:,.0f}", "MLPerf minimum, p99")
v("edge_share_pct", pct(GATES.edge_share), "{:.0f}", "edge cases, percent of inputs")

print("== Interval on the hook's difference (Scenarios.EosAgainst*ResNetMLPerfV40, Welch 95 percent)")
v("eos_n", len(EOS.run_minutes), "{:.0f}", "Eos converged runs")
v("dell_n", len(DELL.run_minutes), "{:.0f}", "Dell converged runs")
v("n_conv_words", len(EOS.run_minutes) + len(DELL.run_minutes), "words", "converged runs of the hook")
v("eos_mean", EOS.mean_minutes, "{:.3f}", "Eos mean, min")
v("eos_mean_short", EOS.mean_minutes, "{:.2f}", "Eos mean, two decimals")
v("eos_sd", EOS.sd_minutes, "{:.3f}", "Eos SD, min")
v("dell_mean", DELL.mean_minutes, "{:.3f}", "Dell mean, min")
v("dell_sd", DELL.sd_minutes, "{:.4f}", "Dell SD, min")
v("d_dell", m(W_DELL.difference, "minute"), "{:.3f}", "Eos - Dell, min")
v("se_dell", m(W_DELL.standard_error, "minute"), "{:.4f}", "SE, min")
v("t_dell", W_DELL.t_critical, "{:.2f}", "t multiplier")
v("dof_dell", W_DELL.dof, "{:.1f}", "Welch degrees of freedom")
v("lo_dell", m(W_DELL.low, "minute"), "{:.3f}", "interval low, min")
v("hi_dell", m(W_DELL.high, "minute"), "{:.3f}", "interval high, min")
v("lo_dell_short", m(W_DELL.low, "minute"), "{:.2f}", "interval low, two decimals (title)")
v("hi_dell_short", m(W_DELL.high, "minute"), "{:.2f}", "interval high, two decimals (title)")
v("d_h200", m(W_H200.difference, "minute"), "{:.3f}", "Eos - H200, min")
v("se_h200", m(W_H200.standard_error, "minute"), "{:.4f}", "SE, H200, min")
v("t_h200", W_H200.t_critical, "{:.2f}", "t multiplier, H200")
v("lo_h200", m(W_H200.low, "minute"), "{:.2f}", "H200 interval low, min")
v("hi_h200", m(W_H200.high, "minute"), "{:.2f}", "H200 interval high, min")
v("eq_hw", m(W_DELL.half_width_with_spread(Q_(EOS.sd_minutes, "minute")), "minute"), "{:.2f}",
  "half-width if Dell were as noisy as Eos, min")

print("== Real against useful: percent faster than Eos")
for w, key in ((W_DELL, "dell"), (W_DELL_ALL, "dellall"), (W_H200, "h200")):
    v(f"pct_{key}", pct(w.fraction_of_base(w.difference)), "{:.1f}", f"{key}: percent faster than Eos")
    v(f"pct_lo_{key}", pct(w.fraction_of_base(w.low)), "{:.1f}", f"{key}: interval low, percent")
    v(f"pct_hi_{key}", pct(w.fraction_of_base(w.high)), "{:.1f}", f"{key}: interval high, percent")
v("pct_lo_dellall_abs", -pct(W_DELL_ALL.fraction_of_base(W_DELL_ALL.low)), "{:.1f}", "the same low, as 'minus'")
v("pct_h200_int", pct(W_H200.fraction_of_base(W_H200.difference)), "{:.0f}", "H200 faster, whole percent")
v("min_useful", pct(W_DELL.minimum_useful), "{:.0f}", "minimum useful speedup, percent (illustrative)")

print("== Runs against interval width (Scenarios.EosResNetRunCountPlanning)")
for n in PLAN.run_counts:
    v(f"hw_{n}", pct(PLAN.half_width_fraction(n)), "{:.2f}", f"half-width of a mean of {n} runs, percent")
d_half, d_one = PLAN.differences
v("res_diff", pct(d_half), "{:.1f}", "difference to resolve, percent")
v("res_runs", math.ceil(PLAN.runs_per_system(d_half)), "{:.0f}", "runs per system for it, rounded up")
v("res_runs_one", math.ceil(PLAN.runs_per_system(d_one)), "{:.0f}", "runs per system for one percent, rounded up")
v("z_power", PLAN.z_beta, "{:.2f}", "z at 80 percent power")

print("== Vote: 95 against 94 percent on the same 1,000 images (Scenarios.NinetyFiveAgainstNinetyFourOnSameImages)")
v("vote_base", pct(VOTE.baseline_accuracy), "{:.0f}", "baseline accuracy, percent")
v("vote_comp", pct(VOTE.candidate_accuracy), "{:.0f}", "compressed accuracy, percent")
v("vote_n", VOTE.n_samples, "{:,.0f}", "test images")
v("vote_err_base", VOTE.baseline_errors, "{:.0f}", "baseline errors")
v("vote_err_comp", VOTE.candidate_errors, "{:.0f}", "compressed errors")
lo, hi = VOTE.independent_interval
v("vote_lo_abs", -pct(lo), "{:.0f}", "independent-set interval low, as 'minus', points")
v("vote_hi", pct(hi), "{:.0f}", "independent-set interval high, points")
v("vote_gap", VOTE.net_gap, "{:.0f}", "b - c")
v("chi2", VOTE.chi2_critical, "{:.2f}", "chi-square critical value at 95 percent")
v("vote_max", VOTE.max_discordant, "{:.0f}", "largest b + c that flags the drop")

print("== McNemar on two Eos seeds (Scenarios.EosResNetTwoSeedsSameValidationSet)")
best = max(EOS.rows, key=lambda r: r["top1"])
worst = min(EOS.rows, key=lambda r: r["top1"])
v("best_run", best["run"], "{:.0f}", "most accurate Eos run")
v("worst_run", worst["run"], "{:.0f}", "least accurate Eos run")
v("best_top1", pct(SEEDS.baseline_accuracy), "{:.3f}", "its top-1, percent")
v("worst_top1", pct(SEEDS.candidate_accuracy), "{:.3f}", "its top-1, percent")
v("seed_gap", SEEDS.net_gap, "{:.0f}", "net gap, images")
v("seed_max", SEEDS.max_discordant, "{:,.0f}", "largest b + c that flags it")
v("seed_max_pct", pct(SEEDS.max_discordant_share), "{:.0f}", "the same, percent of the set")

print("== System descriptions (mlperf_v40_systems/, sysdiff.py; Hardware.Cloud)")
dell_diff, dell_keys = sysdiff.differ(EOS.system, DELL.system)
h200_diff, h200_keys = sysdiff.differ(EOS.system, H200.system)
v("sys_dell_diff", len(dell_diff), "{:.0f}", "fields that differ, Eos and Dell")
v("sys_dell_all", len(dell_keys), "{:.0f}", "fields compared")
v("sys_h200_diff", len(h200_diff), "{:.0f}", "fields that differ, Eos and H200")
v("sys_h200_all", len(h200_keys), "{:.0f}", "fields compared")


def ngc_release(system):
    """Year and month of the first NGC container release (YY.MM) named in a system's framework field."""
    yy, mm = re.search(r"(\d+)\.(\d+)", sysdiff.flat(system)["framework"]).groups()
    return int(yy), int(mm)


(y0, m0), (y1, m1) = ngc_release(DELL.system), ngc_release(EOS.system)
v("container_gap", (y1 - y0) * m(Q_(1, "year"), "month") + (m1 - m0), "words", "container age gap, months")
H100, H200_HW = Hardware.Cloud.H100, Hardware.Cloud.H200
v("h200_bw", m(H200_HW.memory.bandwidth, "TB/s"), "{:.1f}", "H200 HBM bandwidth, TB/s")
v("h100_bw", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM bandwidth, TB/s")
v("h200_bw_more", pct(m(H200_HW.memory.bandwidth / H100.memory.bandwidth, "")) - 100, "{:.0f}",
  "H200 bandwidth over H100, percent more")
v("h200_sparse", m(H200_HW.compute.precision_flops["fp16_sparse"], "TFLOP/s"), "{:,.0f}", "H200 FP16 with sparsity, TFLOP/s")

print("== Run order on a warming device (Scenarios.WarmingPhoneRunOrder; Lee et al. 2019, Figure 8)")
v("lee_start", m(ORDER.start, "ms"), "{:.1f}", "iPhone XS GPU at start, ms")
v("lee_end", m(ORDER.end, "ms"), "{:.1f}", "after 200 s, ms")
v("order_blocked", pct(ORDER.blocked_bias), "{:.0f}", "blocked: B looks slower by, percent")
v("order_inter", pct(ORDER.interleaved_bias), "{:.0f}", "interleaved: B looks slower by, percent")

print("== Beam search (Scenarios.WiderBeamSearchTranslation; Example 12.1, illustrative)")
v("beam_w0", BEAM.original_width, "{:.0f}", "beam before")
v("beam_w1", BEAM.optimized_width, "{:.0f}", "beam after")
v("beam_gain", BEAM.metric_gain, "{:.1f}", "BLEU gain")
v("beam_lat0", m(BEAM.original_latency, "ms"), "{:.0f}", "latency before, ms")
v("beam_lat1", m(BEAM.optimized_latency, "ms"), "{:.0f}", "latency after, ms")
v("beam_slo", m(BEAM.latency_budget, "ms"), "{:.0f}", "latency objective, ms")
v("beam_ratio_words", BEAM.width_ratio, "words", "candidate evaluations ratio")
v("beam_slow_words", BEAM.slowdown, "words", "slowdown")

print("== Galaxy S 4 (Literature.Benchmarks; AnandTech 2013)")
gfx, glb = float(BENCH.GalaxyS4GfxBench27TrianglesM), float(BENCH.GalaxyS4GlBench251TrianglesM)
v("s4_gfx", gfx, "{:.1f}", "GFXBench 2.7, million triangles/s")
v("s4_glb", glb, "{:.1f}", "GLBenchmark 2.5.1, million triangles/s")
v("s4_gain", pct(glb / gfx - 1), "{:.0f}", "GLBenchmark over GFXBench, percent")
v("s4_game_mhz", float(BENCH.GalaxyS4GameGpuClockMhz), "{:.0f}", "game GPU clock, MHz")
v("s4_bench_mhz", float(BENCH.GalaxyS4BenchmarkGpuClockMhz), "{:.0f}", "benchmark GPU clock, MHz")
v("s4_max_mhz", float(BENCH.GalaxyS4SamsungMaxGpuClockMhz), "{:.0f}", "Samsung's stated maximum, MHz")

print("== Fresh test sets (Scenarios.ImageNetV2Shift; Literature.Benchmarks.Gsm1kMaxAccuracyDropPct)")
shift = SHIFT.result
v("recht_models", shift.model_count - shift.baseline_count, "{:.0f}", "networks plotted")
v("recht_lo", shift.network_gap_min, "{:.1f}", "smallest drop, points")
v("recht_hi", shift.network_gap_max, "{:.1f}", "largest drop, points")
v("recht_rho", shift.network_rank_correlation, "{:.3f}", "Spearman rank correlation")
v("gsm1k_drop", float(BENCH.Gsm1kMaxAccuracyDropPct), "{:.0f}", "GSM1k drop, up to, percent")

print("== Pareto on a Pixel 1 (slides/vol1/12b_benchmarking/images/data/sandler2018_mobilenetv2_pixel1.csv)")
SANDLER = load_series("slides/vol1/12b_benchmarking/images/data/sandler2018_mobilenetv2_pixel1.csv").entries


def dominated(row, rows):
    """Another row is at least as accurate and at least as fast, and better on one."""
    return any(o is not row and o["top1"] >= row["top1"] and o["latency_ms"] <= row["latency_ms"]
               and (o["top1"] > row["top1"] or o["latency_ms"] < row["latency_ms"]) for o in rows)


for r in SANDLER:
    key = re.sub(r"[^a-z0-9]", "", r["model"].lower())
    v(f"sd_{key}_top1", r["top1"], "{:.1f}", f"{r['model']}: top-1, percent")
    v(f"sd_{key}_ms", r["latency_ms"], "{:.0f}", f"{r['model']}: latency, ms" + (", dominated" if dominated(r, SANDLER) else ""))
v("frame_budget", m(BOARD.frame_budget, "ms"), "{:.0f}", "camera trap frame deadline, ms")

print("== Hardware lottery on a Pixel 4 (slides/vol1/12b_benchmarking/images/data/chu2021_pixel4_cpu_dsp.csv)")
CHU = load_series("slides/vol1/12b_benchmarking/images/data/chu2021_pixel4_cpu_dsp.csv").entries
v("chu_top1_lo", min(r["top1"] for r in CHU), "{:.1f}", "lowest top-1, percent")
v("chu_top1_hi", max(r["top1"] for r in CHU), "{:.1f}", "highest top-1, percent")
chu = {r["model"]: r for r in CHU}
v("chu_v3_cpu", chu["MobileNetV3 Large"]["cpu_float_ms"], "{:.1f}", "MobileNetV3 Large, CPU, ms")
v("chu_v1_cpu", chu["MobileNetV1 (1.25)"]["cpu_float_ms"], "{:.1f}", "MobileNetV1, CPU, ms")
v("chu_v1_dsp", chu["MobileNetV1 (1.25)"]["dsp_ms"], "{:.2f}", "MobileNetV1, DSP, ms")
v("chu_v3_dsp", chu["MobileNetV3 Large"]["dsp_ms"], "{:.2f}", "MobileNetV3 Large, DSP, ms")

print("== ECE by hand (Scenarios.ThreeBinReliabilityDiagram, constructed bins)")
for i, ((n, c, a), gap, w) in enumerate(zip(BINS.bins, BINS.gaps, BINS.weights), start=1):
    v(f"bin{i}_n", n, "{:.0f}", f"bin {i}: predictions")
    v(f"bin{i}_conf", c, "{:.2f}", f"bin {i}: confidence")
    v(f"bin{i}_acc_pct", pct(a), "{:.0f}", f"bin {i}: accuracy, percent")
    v(f"bin{i}_gap", gap, "{:.2f}", f"bin {i}: gap")
    v(f"bin{i}_w", w, "{:.1f}", f"bin {i}: weight")
v("bins_total", BINS.total, "{:.0f}", "predictions")
v("ece", BINS.ece, "{:.2f}", "ECE")
v("alert_conf", pct(GATES.confidence_threshold), "{:.0f}", "camera trap alerts above, percent confidence")

print("== Guo et al. 2017 (Literature.RobustnessStudies)")
v("guo_lenet_err", float(ROB.Cifar100LeNet5ErrorPct), "{:.1f}", "LeNet 5 error, percent")
v("guo_resnet_err", float(ROB.Cifar100ResNet110ErrorPct), "{:.1f}", "ResNet-110 error, percent")
v("guo_lenet_ece", float(ROB.Cifar100LeNet5EcePct), "{:.2f}", "LeNet 5 ECE, percent")
v("guo_resnet_ece", float(ROB.Cifar100ResNet110EcePct), "{:.2f}", "ResNet-110 ECE, percent")
v("guo_in_ece", float(ROB.ImageNetResNet152EcePct), "{:.2f}", "ResNet-152 ECE, percent")
v("guo_in_ece_ts", float(ROB.ImageNetResNet152EceTempScaledPct), "{:.2f}", "after temperature scaling, percent")
v("guo_ratio", float(ROB.Cifar100ResNet110EcePct) / float(ROB.Cifar100LeNet5EcePct), "{:.1f}", "ECE ratio")

print("== WILDS (Literature.PredictionQuality; Koh et al. 2021, Tables 1 and 4)")
id_acc, ood_acc = float(PQ.WildsIWildCamIdAccuracy), float(PQ.WildsIWildCamOodAccuracy)
id_f1, ood_f1 = float(PQ.WildsIWildCamIdMacroF1), float(PQ.WildsIWildCamOodMacroF1)
v("wilds_id_acc", id_acc, "{:.1f}", "ID average accuracy, percent")
v("wilds_ood_acc", ood_acc, "{:.1f}", "OOD average accuracy, percent")
v("wilds_id_f1", id_f1, "{:.1f}", "ID macro F1, percent")
v("wilds_ood_f1", ood_f1, "{:.1f}", "OOD macro F1, percent")
v("wilds_acc_drop", id_acc - ood_acc, "{:.1f}", "accuracy drop, points")
v("wilds_f1_drop", id_f1 - ood_f1, "{:.1f}", "macro F1 drop, points")
v("wilds_acc_drop_int", id_acc - ood_acc, "{:.0f}", "the same, whole (title)")
v("wilds_f1_drop_int", id_f1 - ood_f1, "{:.0f}", "the same, whole (title)")
v("wilds_reps", float(PQ.WildsIWildCamReplicates), "{:.0f}", "replicates")
v("camelyon_id", float(PQ.WildsCamelyon17IdAccuracy), "{:.1f}", "Camelyon17 ID accuracy, percent")
v("camelyon_ood", float(PQ.WildsCamelyon17OodAccuracy), "{:.1f}", "Camelyon17 OOD accuracy, percent")

print("== Diagnose: acceptance gates (Scenarios.MobileNetV2Int8CameraTrapGates; Table 18, assumed values)")
v("g_top1_fp32", GATES.float_top1, "{:.1f}", "FP32 top-1, percent")
v("g_top1_int8", GATES.int_top1, "{:.1f}", "INT8 top-1, percent")
v("g_top5_fp32", GATES.float_top5, "{:.0f}", "FP32 top-5, percent")
v("g_top5_int8", GATES.int_top5, "{:.1f}", "INT8 top-5, percent")
v("g_ece_fp32", GATES.float_ece, "{:.3f}", "FP32 ECE")
v("g_ece_int8", GATES.int_ece, "{:.3f}", "INT8 ECE")
v("g_edge_fp32", GATES.float_edge, "{:.1f}", "FP32 edge-case accuracy, percent")
v("g_edge_int8", GATES.int_edge, "{:.1f}", "INT8 edge-case accuracy, percent")
v("g_edge_fp32_int", GATES.float_edge, "{:.0f}", "the same, whole")
v("g_limit", GATES.accuracy_loss_limit, "{:.0f}", "accuracy-loss gate, points")
v("g_limit_words", GATES.accuracy_loss_limit, "words", "the same, in words")
v("g_top1_loss", GATES.top1_loss, "{:.1f}", "top-1 loss, points")
v("g_edge_drop", GATES.edge_drop, "{:.1f}", "edge-case drop, points")
v("g_test", GATES.test_images, "{:,.0f}", "lab test images")
v("g_edge_n", GATES.edge_images, "{:.0f}", "edge cases among them")
v("g_edge_hw", GATES.edge_half_width, "{:.0f}", "binomial 95% half-width on them, points")

print("== Verification gap (Scenarios.ImageNetInputSpace)")
v("img_w", SPACE.dataset.image_width, "{:.0f}", "image width")
v("img_h", SPACE.dataset.image_height, "{:.0f}", "image height")
v("img_c", SPACE.dataset.image_channels, "{:.0f}", "channels")
v("img_bits", math.log2(SPACE.levels), "{:.0f}", "bits per channel value")
v("img_levels", SPACE.levels, "{:.0f}", "levels per value")
v("img_values", SPACE.elements, "{:,.0f}", "values per image")
v("space_exp", SPACE.log10_inputs, "{:,.0f}", "log10 of the possible images")
v("test_n", SPACE.test_examples, "{:,.0f}", "validation images")
v("test_exp", math.log10(SPACE.test_examples), "{:.1f}", "log10 of the validation images")
v("frac_exp", SPACE.log10_inputs - math.log10(SPACE.test_examples), "{:,.0f}", "fraction covered = 10 to the minus this")

if __name__ == "__main__":
    write(__file__)
