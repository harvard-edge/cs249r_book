#!/usr/bin/env python3
"""Every number on the Lecture 13 (Model Serving, Part 1) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the ResNet-50 H100 serving
profile (engine.serving_profile) and the model-serving Scenarios.* built on it
(the request path, the M/M/1 fleet plan, the batching window, the dual-trigger
trace, the batch cap, replica startup, the cost of each instance), the cited
results in Literature.ServingResults, and the MLPerf Inference v5.0 rows in
local CSV series. The Model Serving chapter reads the same scenarios. This
file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
this module, so figures and slides share one computation.

Two printed tool outputs are quoted on the slides as the tools printed them:
the Triton perf_analyzer quick-start output (Literature.ServingResults holds
its values for the text around it) and coldstart_log.txt, produced for this
course by coldstart_log.py; its values are parsed from the log below.

Run from this folder: python3 calcs.py
"""

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, val, num, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

from mlsysim.core.units import Q_, BYTES_FP32  # noqa: E402
from mlsysim.engine.serving_profile import resnet50_h100_profile  # noqa: E402
from mlsysim.physics import calc_mm1_max_utilization, calc_littles_law_rate  # noqa: E402

HERE = Path(__file__).resolve().parent
Scenarios, Literature, Models = mlsysim.Scenarios, mlsysim.Literature, mlsysim.Models
Datasets, ReferenceStats = mlsysim.Datasets, mlsysim.ReferenceStats
SR = Literature.ServingResults

PROFILE = resnet50_h100_profile()
SVC = Scenarios.ResNet50EagerFp16BatchingH100
COST = Scenarios.ResNet50BatchOneVersusEightCostH100
PHOTOS = Scenarios.PhotoLibraryPrecomputeH100
PATH = Scenarios.ResNet50RequestPathH100
DLRM = Scenarios.DlrmRequestPath
LITTLE, LITTLE10 = Scenarios.ImageServiceLittlesLaw, Scenarios.ImageServiceLittlesLaw10ms
PLAN = Scenarios.ResNet50FleetPlanH100
VOTE = Scenarios.ResNet50FasterModelAt80PctH100
FAN10, FAN100 = Scenarios.ServingFanout10Leaves, Scenarios.ServingFanout100Leaves
WIN, WIN20 = Scenarios.BatchWindow10ms, Scenarios.BatchWindow20msIn50msBudget
TRACE = Scenarios.DualTriggerBatchFourRequests
CAP = Scenarios.BatchCap16Or32At500Qps
STARTUP = Scenarios.ResNet50ReplicaStartup
SWAP = Scenarios.ModelSwapPcieGen4
SURGE = Scenarios.RecommenderTrafficSurge
ENERGY = Scenarios.ResNet50BatchEnergyPerImage
INST = Scenarios.ResNet50CloudInstanceCost
LOADING = ReferenceStats.ModelLoading

B1, B8, B32, B128 = SVC.batch_sizes
HOST = ("JPEG decode", "resize", "normalize")
BEFORE_MODEL = HOST + ("copy to GPU",)


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def row(series, **match):
    """The one row of a Literature series whose fields equal ``match``."""
    rows = [r for r in series.entries if all(r[k] == x for k, x in match.items())]
    assert len(rows) == 1, match
    return rows[0]


def pct(x):
    """A fraction as percent."""
    return 100 * x


print("== Hook (Scenarios.ResNet50BatchOneVersusEightCostH100)")
v("price_h100", m(COST.price, "dollar/hour"), "{:.0f}", "H100 rental, $ per hour")
v("b1_ms", m(SVC.service_time(B1), "ms"), "{:.1f}", "batch 1 forward pass, ms")
v("b1_ms_2", m(SVC.service_time(B1), "ms"), "{:.2f}", "the same, ms (tables)")
v("b1_ms_3", m(SVC.service_time(B1), "ms"), "{:.3f}", "the same, ms (estimate)")
v("b8_ms", m(SVC.service_time(B8), "ms"), "{:.1f}", "batch 8 forward pass, ms")
v("b1_rate", m(SVC.throughput(B1), "1/s"), "{:,.0f}", "batch 1 requests per second")
v("b1_rate_1", m(SVC.throughput(B1), "1/s"), "{:.1f}", "the same, one decimal")
v("b8_rate", m(SVC.throughput(B8), "1/s"), "{:,.0f}", "batch 8 requests per second")
v("b1_usd", m(COST.cost_per_million(B1), "dollar"), "{:.2f}", "batch 1, $ per million requests")
v("b8_usd", m(COST.cost_per_million(B8), "dollar"), "{:.2f}", "batch 8, $ per million requests")
v("hook_ratio", COST.cost_ratio, "{:.1f}", "batch 1 cost over batch 8 cost")
v("hook_added_ms", m(COST.added_time, "ms"), "{:.2f}", "batch 8 time beyond batch 1, ms")
v("hook_added_ms_1", m(COST.added_time, "ms"), "{:.1f}", "the same, one decimal")
v("hook_label_gap", num("b8_ms") - num("b1_ms"), "{:.1f}", "gap between the rounded labels, ms")
v("hook_added_pct", pct(m(COST.added_time / SVC.service_time(B8), "")), "{:.0f}", "added time / batch 8 time, %")
v("eager_kernels", PROFILE.eager_kernels, "{:.0f}", "kernel launches per eager pass")
v("batch_8", B8, "{:.0f}", "the batched size of the hook")

print("== Recall Lecture 12 (slides/vol1/12_benchmarking/images/data/mlperf_inference_v5_0_resnet50.csv, 5.0-0032)")
RES = load_series("slides/vol1/12_benchmarking/images/data/mlperf_inference_v5_0_resnet50.csv")
RTX_SS = row(RES, result_id="5.0-0032", scenario="SingleStream")["value"]
RTX_OFF = row(RES, result_id="5.0-0032", scenario="Offline")["value"]
rtx_rate = 10 ** 3 / RTX_SS
v("rtx_ss_ms", RTX_SS, "{:.3f}", "RTX 4090 SingleStream p90, ms")
v("rtx_ss_ms_2", RTX_SS, "{:.2f}", "the same, two decimals")
v("rtx_ss_rate", round(rtx_rate, -1), "{:,.0f}", "one at a time, per second (derived), to the ten")
v("rtx_off", RTX_OFF, "{:,.0f}", "RTX 4090 Offline, samples/s")
v("rtx_off_over_ss", RTX_OFF / rtx_rate, "{:.1f}", "Offline over one at a time")

print("== Training against serving (batching sweep, fleet plan)")
v("b32_rate", round(m(SVC.throughput(B32), "1/s"), -1), "{:,.0f}", "batch 32 images/s, to the ten (book)")
v("batch_32", B32, "{:.0f}", "batch 32")
v("plan_rho_pct", pct(PLAN.max_utilization), "{:.1f}", "highest safe utilization, %")
v("plan_rho_pct_0", pct(PLAN.max_utilization), "{:.0f}", "the same, as the chapter prints it")
v("plan_rho", PLAN.max_utilization, "{:.4f}", "highest safe utilization")
v("p99_budget_ms", m(PLAN.tail_budget, "ms"), "{:.0f}", "queue-plus-GPU p99 budget, ms")

print("== Static against dynamic (Scenarios.PhotoLibraryPrecomputeH100)")
v("photos", PHOTOS.items, "{:,.0f}", "photos in the library")
v("moderation_ms", m(PHOTOS.latency_budget, "ms"), "{:.0f}", "moderation deadline, ms")
v("int8_offline", round(m(PHOTOS.offline_rate, "1/s"), -2), "{:,.0f}", "MLPerf INT8 Offline per H100, to the hundred")
v("static_b1_s", m(PHOTOS.on_demand_total, "s"), "{:.0f}", "library one at a time, s")
v("static_int8_s", m(PHOTOS.offline_total, "s"), "{:.2f}", "library at the Offline rate, s")
v("static_ratio", PHOTOS.ratio, "{:.0f}", "ratio")
v("static_ratio_book", round(PHOTOS.ratio, -1), "{:.0f}", "the ratio as the chapter rounds it")
v("static_ratio_floor", math.floor(PHOTOS.ratio / 100) * 100, "{:.0f}", "the ratio, whole hundreds (title)")

print("== Request budget (Scenarios.ResNet50RequestPathH100)")
v("img_side", Datasets.ImageNet.image_height, "{:.0f}", "model input side, pixels")
v("st_decode", m(PATH.stages["JPEG decode"], "ms"), "{:.0f}", "JPEG decode, ms")
v("st_resize", m(PATH.stages["resize"], "ms"), "{:.0f}", "resize, ms")
v("st_norm", m(PATH.stages["normalize"], "ms"), "{:.1f}", "normalize, ms")
v("st_copy", m(PATH.stages["copy to GPU"], "ms"), "{:.1f}", "copy over PCIe, ms")
v("st_post", m(PATH.stages["top-5"], "ms"), "{:.1f}", "top-5 on the host, ms")
v("budget_total", m(PATH.total, "ms"), "{:.1f}", "one request, ms")
v("gpu_idle", m(PATH.subtotal(*BEFORE_MODEL), "ms"), "{:.0f}", "time before the GPU has work, ms")
v("model_share", pct(PATH.share("forward pass")), "{:.0f}", "model share, %")
v("pre_share", pct(PATH.share(*HOST)), "{:.0f}", "preprocessing share, %")

print("== DLRM (Scenarios.DlrmRequestPath)")
v("dl_parse", m(DLRM.stages["parse"], "ms"), "{:.1f}", "parse, ms")
v("dl_emb", m(DLRM.stages["embedding lookup"], "ms"), "{:.0f}", "embedding lookup, ms")
v("dl_mlp", m(DLRM.stages["MLP forward"], "ms"), "{:.1f}", "MLP, ms")
v("dl_rank", m(DLRM.stages["rank and filter"], "ms"), "{:.0f}", "rank and filter, ms")
v("dl_total", m(DLRM.total, "ms"), "{:.0f}", "one request, ms")
v("dl_mlp_share", pct(DLRM.share("MLP forward")), "{:.0f}", "MLP share, %")
v("dlrm_gb", m(Models.Recommendation.DLRM.model_size, "GB"), "{:.0f}", "DLRM reference model, GB")

print("== Pipelining (overlap of host and GPU)")
v("host_stage", m(PATH.subtotal(*HOST), "ms"), "{:.1f}", "host stage, ms")
v("pipe_duty", pct(PATH.overlapped_share("forward pass", *HOST)), "{:.0f}", "pipelined GPU busy, %")

print("== Predict: a 10x faster model (Amdahl on the request; Literature.ServingResults, Richins 2020)")
v("model_speedup", PATH.speedup, "{:.0f}", "model speedup")
v("fast_model_ms", m(PATH.accelerated_stage, "ms"), "{:.2f}", "the faster forward pass, ms")
v("fast_total", m(PATH.accelerated_total, "ms"), "{:.1f}", "new request time, ms")
v("fast_speedup", PATH.overall_speedup, "{:.1f}", "end-to-end speedup")
v("naive_ms", m(PATH.total / PATH.speedup, "ms"), "{:.2f}", "the whole request divided by 10, ms")
RICH = [lit(SR.RichinsAiTaxIngestionMs), lit(SR.RichinsAiTaxDetectionMs),
        lit(SR.RichinsAiTaxBrokersMs), lit(SR.RichinsAiTaxIdentificationMs)]
rich_total = sum(RICH)
v("rich_ingest", RICH[0], "{:.1f}", "ingestion, ms")
v("rich_detect", RICH[1], "{:.1f}", "face detection, ms")
v("rich_brokers", RICH[2], "{:.1f}", "message brokers, ms")
v("rich_ident", RICH[3], "{:.1f}", "identification, ms")
v("rich_total", rich_total, "{:.0f}", "one frame, ms")
v("rich_ai", pct((RICH[1] + RICH[3]) / rich_total), "{:.0f}", "AI stages, %")

print("== Little's law (Scenarios.ImageServiceLittlesLaw)")
v("little_rate", m(LITTLE.arrival_rate, "1/s"), "{:,.0f}", "arrivals per second")
v("little_w_s", m(LITTLE.time_in_system, "s"), "{:.2f}", "time inside, s")
v("little_w_ms", m(LITTLE.time_in_system, "ms"), "{:.0f}", "time inside, ms")
v("little_n", LITTLE.in_flight, "{:.0f}", "requests inside")
v("gpu_batch", LITTLE.batch_size, "{:.0f}", "largest batch on the GPU")
v("little_wait", LITTLE.in_flight - LITTLE.batch_size, "{:.0f}", "requests waiting in host memory")
v("little_w2_ms", m(LITTLE10.time_in_system, "ms"), "{:.0f}", "time inside in the check, ms")
v("little_rate2", m(calc_littles_law_rate(LITTLE.in_flight, LITTLE10.time_in_system), "1/s"), "{:,.0f}",
  "rate those requests sustain, per second")

print("== Knee (Scenarios.ResNet50FleetPlanH100, M/M/1)")
for u in PLAN.utilization_points:
    key = f"{pct(u):.0f}"
    v(f"util_{key}", pct(u), "{:.0f}", "utilization, %")
    v(f"knee_{key}", m(PLAN.time_in_system(u), "ms"), "{:.1f}", "mean time in system, ms")
U_EX = PLAN.example_utilization
v("knee_p99_ex", m(PLAN.tail_time(U_EX), "ms"), "{:.1f}", "p99 at the example load, ms")
v("knee_mean_ex_0", m(PLAN.time_in_system(U_EX), "ms"), "{:.0f}", "mean at the example load, whole ms")
v("p99_factor", PLAN.tail_factor, "{:.1f}", "p99 over the mean, M/M/1")
v("ln100", PLAN.exact_tail_factor, "{:.1f}", "ln 100")

print("== Estimate: H100s for the peak (Scenarios.ResNet50FleetPlanH100)")
v("peak_rate", m(PLAN.peak_rate, "1/s"), "{:,.0f}", "peak requests per second")
v("required_rate", m(PLAN.required_rate, "1/s"), "{:,.0f}", "rate the fleet must serve")
v("gpus_exact", PLAN.accelerators_exact, "{:.1f}", "GPUs, exact")
v("gpus", PLAN.accelerators, "{:.0f}", "GPUs, rounded up")
v("headroom_pct", pct(PLAN.headroom - 1), "{:.0f}", "headroom, %")
v("plan_gpus", PLAN.accelerators_planned, "{:.0f}", "GPUs with headroom")
v("plan_gpus_less1", PLAN.accelerators_planned - 1, "{:.0f}", "GPUs after one fails")
v("fail_util", pct(PLAN.utilization_after_failure), "{:.1f}", "utilization after one failure, %")
v("naive_gpus_exact", PLAN.saturated_accelerators_exact, "{:.1f}", "GPUs at 100 percent busy, exact")
v("naive_gpus", math.ceil(PLAN.saturated_accelerators_exact), "{:.0f}", "the same, rounded up")
v("rho_at_printed", calc_mm1_max_utilization(Q_(num("b1_ms"), "ms"), PLAN.tail_budget, PLAN.tail_factor),
  "{:.3f}", "safe utilization with the printed 3.5 ms")

print("== MLPerf Server against Offline (slides/vol1/12_benchmarking/images/data/mlperf_inference_v5_0_resnet50.csv)")
A_OFF = row(RES, result_id="5.0-0010", scenario="Offline")["value"]
A_SRV = row(RES, result_id="5.0-0010", scenario="Server")["value"]
B_OFF = row(RES, result_id="5.0-0065", scenario="Offline")["value"]
B_SRV = row(RES, result_id="5.0-0065", scenario="Server")["value"]
v("mlperf_bound_ms", lit(SR.MLPerfInferenceResNet50ServerP99BoundMs), "{:.0f}", "Server p99 bound, ms")
v("a_off_k", round(A_OFF, -3), "{:,.0f}", "system A Offline, to the thousand")
v("b_off_k", round(B_OFF, -3), "{:,.0f}", "system B Offline, to the thousand")
v("a_kept", pct(A_SRV / A_OFF), "{:.0f}", "system A keeps, %")
v("b_kept", pct(B_SRV / B_OFF), "{:.0f}", "system B keeps, %")
v("a_lost", 100 - pct(A_SRV / A_OFF), "{:.0f}", "system A loses, %")
v("b_lost", 100 - pct(B_SRV / B_OFF), "{:.0f}", "system B loses, %")
v("kept_gap", pct(A_SRV / A_OFF - B_SRV / B_OFF), "{:.0f}", "gap, points")

print("== Vote: a faster model under load (Scenarios.ResNet50FasterModelAt80PctH100)")
v("vote_util", pct(VOTE.utilization), "{:.0f}", "load, %")
v("vote_service_gain", VOTE.service_gain, "{:.0f}", "service-time gain")
v("fast_service_ms", m(VOTE.fast_service, "ms"), "{:.1f}", "faster service time, ms")
v("vote_old", m(VOTE.slow_time, "ms"), "{:.1f}", "mean before, ms")
v("vote_wait_old", m(VOTE.slow_wait, "ms"), "{:.1f}", "wait before, ms")
v("vote_util_new", pct(VOTE.fast_utilization), "{:.0f}", "load after, %")
v("vote_new", m(VOTE.fast_time, "ms"), "{:.1f}", "mean after, ms")
v("vote_wait_new", m(VOTE.fast_wait, "ms"), "{:.1f}", "wait after, ms")
v("vote_gain", VOTE.time_gain, "{:.0f}", "mean-time gain")

print("== Fan-out (Scenarios.ServingFanout*Leaves; slides/vol1/12_benchmarking/images/data/tail_at_scale_leaf_latency.csv)")
TAIL = load_series("slides/vol1/12_benchmarking/images/data/tail_at_scale_leaf_latency.csv")
TAIL_Q = round(pct(FAN100.availability))  # the leaves' own percentile, p99
leaf = row(TAIL, leaves_awaited="one random leaf", percentile=TAIL_Q)["latency_ms"]
leaves = row(TAIL, leaves_awaited="all leaves", percentile=TAIL_Q)["latency_ms"]
v("fan_n10", FAN10.n_components, "{:.0f}", "servers")
v("fan_n100", FAN100.n_components, "{:.0f}", "servers")
v("fan_avail", FAN100.availability, "{:.2f}", "each server on time")
v("fan_avail_pct", pct(FAN100.availability), "{:.0f}", "the same, %")
v("fan10", pct(FAN10.all_up), "{:.1f}", "10 servers all on time, %")
v("fan100", pct(FAN100.all_up), "{:.1f}", "100 servers all on time, %")
v("db_leaf", leaf, "{:.0f}", "one leaf p99, ms")
v("db_all", leaves, "{:.0f}", "all leaves p99, ms")
v("db_ratio", leaves / leaf, "{:.0f}", "all over one")

print("== perf_analyzer output (Literature.ServingResults, Triton quick start)")
PA_CLIENT, PA_SERVER = lit(SR.TritonPerfAnalyzerClientLatencyUs), lit(SR.TritonPerfAnalyzerServerLatencyUs)
PA_INFER = lit(SR.TritonPerfAnalyzerComputeInferUs)
v("pa_client", PA_CLIENT, "{:.0f}", "client latency, us")
v("pa_server", PA_SERVER, "{:.0f}", "server latency, us")
v("pa_infer", PA_INFER, "{:.0f}", "compute infer, us")
v("pa_infer_share", pct(PA_INFER / PA_CLIENT), "{:.0f}", "compute infer share of client latency, %")
v("pa_outside", PA_CLIENT - PA_SERVER, "{:.0f}", "client minus server, us")
v("pa_count", lit(SR.TritonPerfAnalyzerInferenceCount), "{:,.0f}", "inferences")
v("pa_exec", lit(SR.TritonPerfAnalyzerExecutionCount), "{:,.0f}", "executions")
v("pa_tput", lit(SR.TritonPerfAnalyzerThroughputInferPerS), "{:,.2f}", "throughput, per second")
v("pa_little", lit(SR.TritonPerfAnalyzerThroughputInferPerS) * PA_CLIENT / 10 ** 6, "{:.2f}",
  "throughput x latency: requests in flight")

print("== Fixed cost per image (Scenarios.ResNet50EagerFp16BatchingH100, ResNet50BatchEnergyPerImage)")
v("fixed_ms", m(SVC.fixed_time, "ms"), "{:.2f}", "fixed cost per pass, ms")
v("fixed_ms_3", m(SVC.fixed_time, "ms"), "{:.3f}", "the same, three decimals")
v("per_img_ms", m(SVC.per_request_time, "ms"), "{:.3f}", "work per image, ms")
v("per_img_ms_4", m(SVC.per_request_time, "ms"), "{:.4f}", "the same, four decimals")
v("host_per_kernel_us", m(PROFILE.python_dispatch + PROFILE.kernel_launch, "microsecond"), "{:.0f}",
  "host time per kernel, us")
v("issue_ms", m(PROFILE.eager_kernels * (PROFILE.python_dispatch + PROFILE.kernel_launch), "ms"), "{:.1f}",
  "kernel issue per eager pass, ms")
v("weights_mb", m(ENERGY.weights, "MB"), "{:.0f}", "FP32 weights, MB")
for b in (B1, B8, B32):
    v(f"per_image_{b}", m(SVC.time_per_request(b), "ms"), "{:.2f}", f"batch {b}: ms per image")
    v(f"fixed_share_{b}", pct(SVC.fixed_share(b)), "{:.0f}", f"batch {b}: fixed share, %")
    v(f"energy_{b}", m(ENERGY.energy(b), "mJ"), "{:.1f}", f"batch {b}: energy per image, mJ")
    v(f"move_share_{b}", pct(ENERGY.movement_share(b)), "{:.0f}", f"batch {b}: data movement share of energy, %")
v("b32_ms", m(SVC.service_time(B32), "ms"), "{:.2f}", "batch 32 pass, ms")
v("tput_gain", round(m(SVC.throughput(B32) / SVC.throughput(B1), "")), "{:.0f}", "batch 32 rate over batch 1")

print("== Runtimes at batch 1 (engine.serving_profile)")
RT = {"ts": PROFILE.torchscript_fp16(1), "onnx": PROFILE.onnx_fp16(1), "fp32": PROFILE.tensorrt("fp32", 1),
      "fp16": PROFILE.tensorrt("fp16", 1), "int8": PROFILE.tensorrt("int8", 1)}
for k, t in RT.items():
    v(f"rt_{k}", m(t, "ms"), "{:.2f}", f"{k}, batch 1, ms")
    v(f"rt_{k}_x", m(SVC.service_time(B1) / t, ""), "{:.1f}", f"{k} over eager")
v("fused_kernels", PROFILE.fused_kernels, "{:.0f}", "kernels after fusion")
v("rt_prec_gain", m(RT["fp32"] / RT["int8"], ""), "{:.1f}", "INT8 over TensorRT FP32")

print("== Modeled curve and a measured L4 (slides/vol1/13_model_serving/images/data/mlperf_inference_v5_0_resnet50_l4.csv)")
for b in (B1, B32, B128):
    v(f"curve_ms_{b}", m(SVC.service_time(b), "ms"), "{:.2f}" if b < B128 else "{:.1f}", f"batch {b} pass, ms")
    v(f"curve_rate_{b}", float(f"{m(SVC.throughput(b), '1/s'):.3g}"), "{:,.0f}", f"batch {b}, per second, 3 figures")
v("batch_128", B128, "{:.0f}", "batch 128")
L4 = load_series("slides/vol1/13_model_serving/images/data/mlperf_inference_v5_0_resnet50_l4.csv")
L4_SS, L4_MS = row(L4, scenario="SingleStream"), row(L4, scenario="MultiStream")
L4_OFF = row(L4, scenario="Offline")
l4_one = L4_SS["samples_per_query"] / L4_SS["value"] * 10 ** 3
l4_multi = L4_MS["samples_per_query"] / L4_MS["value"] * 10 ** 3
l4_each = L4_OFF["value"] / L4_OFF["accelerators"]
v("l4_ss_ms", L4_SS["value"], "{:.3f}", "SingleStream p90, ms")
v("l4_ms_ms", L4_MS["value"], "{:.3f}", "MultiStream p99, ms")
v("l4_per_query", L4_MS["samples_per_query"], "{:.0f}", "images per MultiStream query")
v("l4_count", L4_OFF["accelerators"], "{:.0f}", "L4s in the system")
v("l4_count_word", L4_OFF["accelerators"], "words", "the same, in prose")
v("l4_one", l4_one, "{:,.0f}", "one at a time, per second")
v("l4_multi", float(f"{l4_multi:.4g}"), "{:,.0f}", "eight at a time, per second, 4 figures")
v("l4_each", l4_each, "{:,.0f}", "Offline per GPU, per second")
v("l4_lat_ratio", L4_MS["value"] / L4_SS["value"], "{:.1f}", "latency ratio")
v("l4_tput_ratio", l4_multi / l4_one, "{:.1f}", "throughput ratio")

print("== Batching window (Scenarios.BatchWindow10ms, BatchWindow20msIn50msBudget)")
R50, R200, R500, R1000 = WIN.arrival_rates
v("win_ms", m(WIN.window, "ms"), "{:.0f}", "window, ms")
v("win20_ms", m(WIN20.window, "ms"), "{:.0f}", "the wider window, ms")
v("win20_wait", m(WIN20.mean_wait, "ms"), "{:.0f}", "its mean wait, ms")
v("win20_budget", m(WIN20.latency_budget, "ms"), "{:.0f}", "latency budget, ms")
v("win20_share", pct(WIN20.budget_share), "{:.0f}", "share of the budget, %")
for r in WIN.arrival_rates:
    q = m(r, "1/s")
    v(f"qps_{q:.0f}", q, "{:,.0f}", "requests per second")
    v(f"win_mean_{q:.0f}", WIN.mean_arrivals(r), "{:g}", "arrivals per window")
v("empty_50", pct(WIN.empty_share(R50)), "{:.0f}", "empty windows at the lowest rate, %")
v("seen_mean", WIN.seen_batch_mean(R500), "{:.0f}", "batch a request sees, mean")
v("seen_p99", WIN.seen_batch_tail(R500), "{:.0f}", "batch a request sees, p99")

print("== Triton dynamic batcher (Literature.ServingResults)")
v("triton_delay_us", lit(SR.TritonBatcherExampleMaxQueueDelayUs), "{:.0f}", "example timeout, us")
v("triton_pref_small", lit(SR.TritonBatcherExamplePreferredSmall), "{:.0f}", "preferred size")
v("triton_pref_large", lit(SR.TritonBatcherExamplePreferredLarge), "{:.0f}", "preferred size")
v("triton_max_batch", lit(SR.TritonExampleModelMaxBatchSize), "{:.0f}", "example model max_batch_size")

print("== Dual trigger (Scenarios.DualTriggerBatchFourRequests)")
for i, t in enumerate(TRACE.arrivals):
    v(f"arrive_{i}", m(t, "ms"), "{:.0f}", "arrival, ms")
v("trace_n_word", len(TRACE.arrivals), "words", "requests in the batch, in prose")
v("timeout_ms", m(TRACE.timeout, "ms"), "{:.0f}", "batch timeout, ms")
v("worker_free_ms", m(TRACE.worker_free, "ms"), "{:.0f}", "worker busy until, ms")
v("batch4_ms", m(TRACE.batch_time, "ms"), "{:.1f}", "batch of four, ms")
v("response_ms", m(TRACE.response_path, "ms"), "{:.0f}", "response path, ms")
v("deadline_ms", m(TRACE.deadline, "ms"), "{:.0f}", "deadline, ms")
v("backlog_ms", m(TRACE.backlog_wait, "ms"), "{:.0f}", "wait for the worker, ms")
v("batch_end_ms", m(TRACE.dispatch + TRACE.batch_time, "ms"), "{:.1f}", "batch ends, ms")
v("trace_done_ms", m(TRACE.oldest_latency, "ms"), "{:.1f}", "oldest request done, ms")
v("trace_slack_ms", m(TRACE.slack, "ms"), "{:.1f}", "slack, ms")
v("trace_wrong_ms", m(TRACE.timer_only_latency, "ms"), "{:.1f}", "timer-only answer, ms")

print("== Batch cap 16 or 32 (Scenarios.BatchCap16Or32At500Qps)")
v("cap_offer", m(CAP.arrival_rate, "1/s"), "{:.0f}", "offered requests per second")
v("cap_small", CAP.small_cap, "{:.0f}", "smaller cap")
v("cap_large", CAP.large_cap, "{:.0f}", "larger cap")
v("cap_small_rate", round(m(CAP.service.throughput(CAP.small_cap), "1/s"), -1), "{:,.0f}", "cap 16 rate, to the ten")
v("cap_large_rate", round(m(CAP.service.throughput(CAP.large_cap), "1/s"), -1), "{:,.0f}", "cap 32 rate, to the ten")
v("cap_small_ms", m(CAP.service.service_time(CAP.small_cap), "ms"), "{:.1f}", "cap 16 pass, ms")
v("cap_large_ms", m(CAP.service.service_time(CAP.large_cap), "ms"), "{:.1f}", "cap 32 pass, ms")
v("cap_gain", pct(CAP.throughput_gain), "{:.0f}", "throughput gain, %")
v("fill_small", m(CAP.formation_wait(CAP.small_cap), "ms"), "{:.0f}", "mean wait to fill 16, ms")
v("fill_large", m(CAP.formation_wait(CAP.large_cap), "ms"), "{:.0f}", "mean wait to fill 32, ms")
v("cap_over_load", CAP.capacity_over_load, "{:.0f}", "capacity at 16 over load")

print("== Cold start (Scenarios.ResNet50ReplicaStartup; ReferenceStats.ModelLoading; ServerlessLLM)")
PH, RG = STARTUP.phases, STARTUP.ranges
v("cs_ssd", m(PH["weights from local SSD"], "s"), "{:.1f}", "weights from SSD, s")
v("cs_s3_lo", m(RG["weights from object storage"][0], "s"), "{:.0f}", "from object storage, low, s")
v("cs_s3_hi", m(RG["weights from object storage"][1], "s"), "{:.0f}", "from object storage, high, s")
v("cs_cuda_lo", m(RG["CUDA context"][0], "s"), "{:.1f}", "CUDA context, low, s")
v("cs_cuda_hi", m(RG["CUDA context"][1], "s"), "{:.1f}", "CUDA context, high, s")
v("cs_trt_lo", m(RG["TensorRT engine build"][0], "s"), "{:.0f}", "engine build, low, s")
v("cs_trt_hi", m(RG["TensorRT engine build"][1], "s"), "{:.0f}", "engine build, high, s")
v("cs_warm_rt", m(PH["warmup"] + PH["runtime"], "s"), "{:.1f}", "warmup plus runtime, s")
v("cs_warm", m(STARTUP.warm_total, "s"), "{:.1f}", "precompiled start, s")
v("cs_cold", m(STARTUP.cold_total, "s"), "{:.0f}", "first deploy, s")
v("ckpt_gb", m(LOADING.StableDiffusionV15CheckpointSize, "GB"), "{:.0f}", "checkpoint, GB")
v("pickle_s", m(LOADING.StableDiffusionV15PickleLoadTime, "s"), "{:.0f}", "pickle load, s")
v("safetensors_s", m(LOADING.StableDiffusionV15SafetensorsLoadTime, "s"), "{:.1f}", "safetensors load, s")
v("opt_s", lit(SR.FuServerlessLlmOpt30bLoadS), "{:.0f}", "OPT-30B load, s")
v("opt_gpus", lit(SR.FuServerlessLlmOpt30bGpus), "{:.0f}", "its GPUs")
v("llama_s", lit(SR.FuServerlessLlmLlama2_70bLoadS), "{:.0f}", "Llama 2 70B load, s")
v("llama_gpus", lit(SR.FuServerlessLlmLlama2_70bGpus), "{:.0f}", "its GPUs")
v("sllm_gb", lit(SR.FuServerlessLlmCheckpointGB), "{:.0f}", "checkpoint, GB")
v("sllm_gbps", lit(SR.FuServerlessLlmNetworkGBps), "{:.0f}", "network, GB/s")
v("sllm_dl", lit(SR.FuServerlessLlmCheckpointGB) / lit(SR.FuServerlessLlmNetworkGBps), "{:.0f}", "download, s")

print("== Cold-start log (coldstart_log.txt, from coldstart_log.py)")
_log = (HERE / "coldstart_log.txt").read_text().split("# run 2")
_ready = [float(re.findall(r"\[\s*([\d.]+) s\] ready", p)[0]) for p in _log]
_comp = [float(re.findall(r"compiles\): ([\d.]+) s", p)[0]) for p in _log]
_pkl = [float(re.findall(r"\(pickle\): ([\d.]+) s", p)[0]) for p in _log]
_st = [float(re.findall(r"safetensors: ([\d.]+) s", p)[0]) for p in _log]
v("log_ready1", _ready[0], "{:.1f}", "run 1 ready, s")
v("log_ready2", _ready[1], "{:.1f}", "run 2 ready, s")
v("log_comp1", _comp[0], "{:.1f}", "run 1 compile, s")
v("log_comp_share", pct(_comp[0] / _ready[0]), "{:.0f}", "compile share of run 1, %")
v("log_gain", _ready[0] / _ready[1], "{:.1f}", "run 2 ready sooner, times")
v("log_pickle_ms", _pkl[0] * 10 ** 3, "{:.0f}", "pickle load in run 1, ms")
v("log_st_ms", _st[0] * 10 ** 3, "{:.0f}", "safetensors load in run 1, ms")

print("== Model swap (Scenarios.ModelSwapPcieGen4)")
LINK = next(iter(SWAP.links))
swap = SWAP.transfer_time(LINK)
v("swap_gb", m(SWAP.payload, "GB"), "{:.0f}", "model, GB")
v("swap_bw", m(SWAP.links[LINK], "GB/s"), "{:.0f}", "PCIe Gen4 x16, GB/s")
v("swap_ms", m(swap, "ms"), "{:.1f}", "swap time, ms")
v("swap_over_word", math.floor(m(swap / PLAN.tail_budget, "")), "words", "whole deadlines in the swap, in prose")
v("swap_passes", round(m(swap / SVC.service_time(B1), ""), -1), "{:.0f}", "forward passes in the swap, to the ten")

print("== Traffic spike (Scenarios.RecommenderTrafficSurge)")
v("surge_normal", m(SURGE.normal_rate, "1/s"), "{:,.0f}", "normal requests per second")
v("surge_peak", m(SURGE.surge_rate, "1/s"), "{:,.0f}", "surge requests per second")
v("surge_word", SURGE.surge_factor, "words", "surge factor, in prose")
v("surge_latency_s", m(SURGE.surge_latency, "s"), "{:.0f}", "latency in the surge, s")

print("== Cost per image (Scenarios.ResNet50CloudInstanceCost)")
P, RATES = INST.prices, INST.rates
v("cpu_price", m(P["CPU"], "dollar/hour"), "{:.2f}", "CPU instance, $ per hour")
v("t4_price", m(P["T4"], "dollar/hour"), "{:.2f}", "T4 instance, $ per hour")
v("cpu_rate", m(RATES["CPU"], "1/s"), "{:.0f}", "CPU images/s")
v("t4_rate", m(RATES["T4"], "1/s"), "{:.0f}", "T4 images/s")
for k in ("CPU", "T4", "H100"):
    v(f"cpm_{k.lower()}", m(INST.cost_per_million(k), "dollar"), "{:.2f}", f"{k}, $ per million images")
v("cpu_per_hour", INST.requests_per_hour("CPU"), "{:,.0f}", "CPU images per hour")
v("t4_n_word", INST.small_within_large_price, "words", "T4s cheaper than one H100, in prose")
v("t4_n_usd", INST.small_within_large_price * m(P["T4"], "dollar/hour"), "{:.2f}", "their hourly cost")
v("t4_n_rate", m(INST.small_fleet_rate, "1/s"), "{:,.0f}", "their images/s")
v("crossover", round(m(INST.breakeven_rate, "1/s"), -2), "{:,.0f}", "crossover load, to the hundred")

print("== Synthesis (Scenarios.ResNet50FleetPlanH100)")
v("syn_batched", PLAN.batched_accelerators_planned, "{:.0f}", "batched plan, H100s")
v("syn_cost_b1", m(PLAN.cost_per_million(PLAN.accelerators_planned), "dollar"), "{:.2f}", "batch-1 fleet, $ per million")
v("syn_cost_bt", m(PLAN.cost_per_million(PLAN.batched_accelerators_planned), "dollar"), "{:.2f}",
  "batched fleet, $ per million")
v("syn_ratio", PLAN.accelerators_planned / PLAN.batched_accelerators_planned, "{:.0f}", "fleet cost ratio")

print("== Next lecture (slides/vol1/13_model_serving/images/data/mlperf_inference_v5_0_llama2_70b.csv)")
LL = load_series("slides/vol1/13_model_serving/images/data/mlperf_inference_v5_0_llama2_70b.csv")
LL_OFF = row(LL, benchmark="llama2-70b-99", scenario="Offline")
LL_CONV = row(LL, benchmark="llama2-70b-99", scenario="Server")
LL_INT = row(LL, benchmark="llama2-70b-interactive-99", scenario="Server")
v("llama_off", LL_OFF["tokens_per_s"], "{:,.0f}", "Offline tokens/s")
v("llama_conv", LL_CONV["tokens_per_s"], "{:,.0f}", "conversational Server tokens/s")
v("llama_conv_k", round(LL_CONV["tokens_per_s"], -3), "{:,.0f}", "the same, to the thousand")
v("llama_conv_tpot", LL_CONV["tpot_bound_ms"], "{:.0f}", "conversational TPOT bound, ms")
v("llama_conv_ttft_s", LL_CONV["ttft_bound_ms"]  / 10 ** 3, "{:.0f}", "conversational TTFT bound, s")
v("llama_int_tpot", LL_INT["tpot_bound_ms"], "{:.0f}", "interactive TPOT bound, ms")
v("llama_int_ttft", LL_INT["ttft_bound_ms"], "{:.0f}", "interactive TTFT bound, ms")
v("llama_int", LL_INT["tokens_per_s"], "{:,.0f}", "interactive Server tokens/s")

if __name__ == "__main__":
    write(__file__)
