#!/usr/bin/env python3
"""Every number on the Volume I Lecture 7 (ML Frameworks) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the GPT-2 XL decode step and
its eager host work (Scenarios.Gpt2XlEagerDecodeDispatch), the dispatch share
of a small and a large operation, the small-MLP and GPT-3-block dispatch
ratios (HostDispatchScenario), kernel fusion (KernelFusionScenario), the
compile breakeven (CompileBreakevenScenario), the Jacobian of one wide layer,
the ResNet-50 training state with saved activations (TrainingStateFitScenario),
the tensor and batch copies across an H100's links, the data loader, the
784-128-10 training step (MlpTrainingStepScenario), the embedded deployment
paths, and the cited figures in Literature (operator registries, PyTorch 2
speedups, FlashAttention traffic, the Mars Climate Orbiter report). The
chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports this module, so figures and slides
share one computation.

Measured inputs are read from data/, which profile_frameworks.py writes by
running PyTorch on this machine (ResNet-50 memory over two training steps,
the batch-128 peak, and the MNIST step times).

Run from this folder: python3 calcs.py
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Scenarios, Literature = mlsysim.Scenarios, mlsysim.Literature
units = mlsysim.core.units
Q_ = units.Q_

DECODE = Scenarios.Gpt2XlFp16DecodeOnH100
GPT2 = Scenarios.Gpt2XlEagerDecodeDispatch
SMALL_OP = Scenarios.SmallKernelDispatch
LARGE_OP = Scenarios.LargeKernelDispatch
MLP = Scenarios.SmallMlpDispatch
GPT3 = Scenarios.Gpt3BlockDispatch
FUSE = Scenarios.LayerNormDropoutReluFusion
COMPILE = Scenarios.ResNet50CompileBreakeven
JAC = Scenarios.Dense100kLayerJacobianFp32
RES_LO = Scenarios.ResNet50Fp32AdamBatch64
RES_HI = Scenarios.ResNet50Fp32AdamBatch64HighActivations
RES_BIG = Scenarios.ResNet50Fp32AdamBatch512
TENSOR = Scenarios.Fp32MillionElementTensorAcrossH100Links
FEED = Scenarios.ImageNetBatch64HostFeed
STEP = Scenarios.MnistTwoLayerTrainingStepOnH100
PATHS = Scenarios.EmbeddedLinuxFrameworkPaths
H100 = DECODE.hardware
CR = Literature.CompilerResults
MCO = Literature.SpaceflightAnomalies


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


# ---------------------------------------------------------------------------
# Measured data (data/, written by profile_frameworks.py)
# ---------------------------------------------------------------------------
def load_resnet_memory():
    """Rows (event, step, phase, module, bytes) of the ResNet-50 batch-64 run,
    and the parameter count from the file header."""
    rows, n_params = [], None
    for line in open(DATA / "resnet50_memory.csv"):
        if line.startswith("#"):
            found = re.search(r"(\d+) parameters", line)
            if found:
                n_params = int(found.group(1))
            continue
        if line.startswith("event"):
            continue
        e, s, phase, mod, b = line.rstrip("\n").split(",")
        rows.append((int(e), int(s), phase, mod, int(b)))
    return rows, n_params


def load_resnet_b128():
    t = (DATA / "resnet50_peak_b128.txt").read_text()
    start = int(re.search(r"start bytes: (\d+)", t).group(1))
    peak = int(re.search(r"peak bytes: (\d+)", t).group(1))
    return start, peak


def load_mnist_times():
    """Median eager step time (us) and trial count per batch size, Apple GPU (MPS)."""
    t = (DATA / "mnist_step.txt").read_text()
    out = {}
    for b, us, trials in re.findall(r"batch (\d+): median step ([\d.]+) us \(trials ([^)]*)\)", t):
        out[int(b)] = (float(us), len(trials.split(",")))
    return out


print("== Mars Climate Orbiter (MIB Phase I report; Literature.SpaceflightAnomalies)")
v("mco_factor", m(Q_(1, "lbf * s"), "N * s"), "{:.2f}", "newton-seconds per pound-force second")
v("mco_planned_km", MCO.MarsClimateOrbiterPlannedAltitudeKm, "{:.0f}", "planned (estimated) altitude, km")
v("mco_actual_km", MCO.MarsClimateOrbiterActualAltitudeKm, "{:.0f}", "actual altitude, km")

print("== Fusion (Scenarios.LayerNormDropoutReluFusion; Literature.CompilerResults)")
v("fuse_trips", FUSE.unfused_trips, "{:.0f}", "HBM trips, three kernels")
v("fuse_launches", FUSE.unfused_launches, "{:.0f}", "launches, unfused")
v("fuse_launches_word", FUSE.unfused_launches, "words", "the same, in prose")
v("fuse_trips_fused", FUSE.fused_trips, "{:.0f}", "HBM trips, fused")
v("fuse_launch_fused", FUSE.fused_launches, "{:.0f}", "launches, fused")
v("fuse_ratio", FUSE.traffic_ratio, "{:.0f}", "traffic, unfused over fused")
v("flash_hbm_ratio", CR.FlashAttentionGpt2MediumStandardHbmTrafficGB / CR.FlashAttentionGpt2MediumFusedHbmTrafficGB,
  "{:.0f}", "FlashAttention: standard over fused HBM traffic (Dao 2022, Fig. 2)")

print("== Operator surfaces (Literature.Series, 2025 registries)")
pt = Literature.Series.PyTorchOperatorSchemas.entries[-1]
tf = Literature.Series.TensorFlowOperators.entries[-1]
core = [r for r in Literature.Series.PyTorchOperatorSchemas.entries if r["core_aten"]][-1]["core_aten"]
xla = Literature.Series.XLAHLOOpcodes.entries[-1]["opcodes"]
onnx = Literature.Series.ONNXOpsetOperators.entries[-1]["operators"]
jax = Literature.Series.JAXLaxPrimitives.entries[-1]["primitives"]
OPS = [("PyTorch ATen (eager)", "ops_pt", pt["aten_schemas"]), ("TensorFlow (eager)", "ops_tf", tf["ops"]),
       ("ONNX standard", "ops_onnx", onnx), ("Core ATen IR", "ops_core", core),
       ("JAX lax primitives", "ops_jax", jax), ("XLA HLO", "ops_xla", xla)]
for _, key, n in OPS:
    v(key, n, "{:,.0f}", f"{key} operator count")
v("pt_release", pt["release"].lstrip("v").rsplit(".0", 1)[0], "{}", "PyTorch release")
v("tf_release", tf["release"].lstrip("v").rsplit(".0", 1)[0], "{}", "TensorFlow release")
v("ops_year", int(pt["date"].split("-")[0]), "{:.0f}", "registry year")
gaps = (pt["aten_schemas"] / core, tf["ops"] / xla)
v("ops_gap_lo", round(min(gaps)), "{:.0f}", "eager over core, smaller gap")
v("ops_gap_hi", round(max(gaps)), "{:.0f}", "eager over core, larger gap")

print("== GPT-2 XL decode (Scenarios.Gpt2XlEagerDecodeDispatch)")
v("gpt2_weights_gb", m(DECODE.result.weight_bytes, "GB"), "{:.0f}", "FP16 weights, GB")
v("hbm_tbps", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM3, TB/s")
v("gpt2_read_ms", m(GPT2.device_time, "ms"), "{:.1f}", "weight read per token, ms")
v("gpt2_blocks", GPT2.blocks, "{:.0f}", "transformer blocks")
v("gpt2_kernels_per_block", GPT2.ops_per_block, "{:.0f}", "eager kernels per block")
v("gpt2_kernels", GPT2.launches, "{:.0f}", "kernels per step")
v("dispatch_us", m(GPT2.dispatch, "microsecond"), "{:.0f}", "Python dispatch per operation, us")
v("launch_us", m(GPT2.launch, "microsecond"), "{:.0f}", "kernel launch, us")
v("per_op_us", m(GPT2.per_op_cost, "microsecond"), "{:.0f}", "host cost per operation, us")
v("gpt2_launch_ms", m(GPT2.launch_time, "ms"), "{:.1f}", "launches only, ms")
v("gpt2_host_ms", m(GPT2.host_time, "ms"), "{:.1f}", "dispatch plus launch, ms")
v("gpt2_host_ms_round", m(GPT2.host_time, "ms"), "{:.0f}", "the same, to the millisecond (vote answer)")
v("gpt2_launch_mult_word", round(GPT2.launch_ratio), "words", "launches over the read, in prose")
v("gpt2_host_mult_word", round(GPT2.overhead_ratio), "words", "host over the read, in prose")

print("== Dispatch share (Scenarios.SmallKernelDispatch, LargeKernelDispatch)")
v("small_kernel_us", m(SMALL_OP.device_time, "microsecond"), "{:.0f}", "small kernel, us")
v("small_step_us", m(SMALL_OP.host_time + SMALL_OP.device_time, "microsecond"), "{:.0f}", "small op step, us")
v("large_kernel_us", m(LARGE_OP.device_time, "microsecond"), "{:.0f}", "large kernel, us")
v("small_share_pct", 100 * SMALL_OP.host_share, "{:.0f}", "small op dispatch share, %")
v("large_share_pct", 100 * LARGE_OP.host_share, "{:.0f}", "large op dispatch share, %")

print("== Dispatch overhead law (Scenarios.SmallMlpDispatch, Gpt3BlockDispatch)")
v("mlp_ops", MLP.operations, "{:.0f}", "small MLP operations")
v("mlp_device_us", m(MLP.device_time, "microsecond"), "{:.1f}", "small MLP device time, us")
v("mlp_host_us", m(MLP.host_time, "microsecond"), "{:.0f}", "small MLP dispatch, us")
v("mlp_ratio", MLP.overhead_ratio, "{:.0f}", "small MLP overhead ratio")
v("mlp_host_pct", 100 * MLP.host_share, "{:.0f}", "small MLP host share, %")
v("gpt3_kernels", GPT3.operations, "{:.0f}", "one block's kernels")
v("gpt3_host_us", m(GPT3.host_time, "microsecond"), "{:.0f}", "GPT-3 block dispatch, us")
v("gpt3_host_ms", m(GPT3.host_time, "ms"), "{:.2f}", "the same, ms")
v("gpt3_device_ms", m(GPT3.device_time, "ms"), "{:.0f}", "GPT-3 layer device time, ms")
v("gpt3_ratio", GPT3.overhead_ratio, "{:.3f}", "GPT-3 layer overhead ratio")

print("== Compile breakeven (Scenarios.ResNet50CompileBreakeven, hypothetical)")
v("eager_ips", m(COMPILE.eager_rate, "1/second"), "{:,.0f}", "eager images/s")
v("compiled_ips", m(COMPILE.compiled_rate, "1/second"), "{:,.0f}", "compiled images/s")
v("compile_s", m(COMPILE.compile_time, "second"), "{:.0f}", "compile per code change, s")
v("eager_img_ms", m(COMPILE.eager_item_time, "ms"), "{:.3f}", "eager per image, ms")
v("compiled_img_ms", m(COMPILE.compiled_item_time, "ms"), "{:.3f}", "compiled per image, ms")
v("saving_img_ms", m(COMPILE.saving_per_item, "ms"), "{:.3f}", "saving per image, ms")
v("breakeven_images", round(COMPILE.breakeven_items, -3), "{:,.0f}", "breakeven images, to the thousand")
v("proto_images", COMPILE.experiment_items, "{:,.0f}", "prototyping images")
v("proto_changes", COMPILE.recompiles, "{:.0f}", "code changes")
v("proto_changes_word", COMPILE.recompiles, "words", "the same, in prose")
v("proto_eager_s", m(COMPILE.experiment_eager, "second"), "{:.1f}", "prototype eager, s")
v("proto_compiles_s", m(COMPILE.recompiles * COMPILE.compile_time, "second"), "{:.0f}", "compiles alone, s")
v("proto_run_s", m(COMPILE.experiment_compiled_running, "second"), "{:.1f}", "compiled running, s")
v("proto_compiled_s", m(COMPILE.experiment_compiled_total, "second"), "{:.0f}", "prototype compiled total, s")
v("proto_slowdown", COMPILE.experiment_slowdown, "{:.0f}", "prototype slowdown")

print("== PyTorch 2 measured speedups (Ansel et al. 2024; Literature.CompilerResults)")
PT2 = {"TorchBench": (CR.TorchInductorA100Fp32TorchBenchInferenceSpeedup, CR.TorchInductorA100Fp32TorchBenchTrainingSpeedup),
       "Hugging Face": (CR.TorchInductorA100Fp32HuggingFaceInferenceSpeedup, CR.TorchInductorA100Fp32HuggingFaceTrainingSpeedup),
       "TIMM": (CR.TorchInductorA100Fp32TimmInferenceSpeedup, CR.TorchInductorA100Fp32TimmTrainingSpeedup)}
for suite, (inf, trn) in PT2.items():
    key = suite.lower().replace(" ", "")
    v(f"pt2_{key}_inf", inf, "{:.2f}", f"{suite} inference speedup")
    v(f"pt2_{key}_trn", trn, "{:.2f}", f"{suite} training speedup")
v("pt2_models", CR.TorchInductorA100Fp32TorchBenchInferenceModels + CR.TorchInductorA100Fp32HuggingFaceInferenceModels
  + CR.TorchInductorA100Fp32TimmInferenceModels, "{:.0f}", "models across the three suites (inference)")
v("pt2_all", CR.TorchInductorA100Fp16HuggingFaceAllOptimizationsSpeedup, "{:.2f}", "HF fp16, all optimizations")
v("pt2_no_fusion", CR.TorchInductorA100Fp16HuggingFaceWithoutFusionAndInliningSpeedup, "{:.2f}",
  "HF fp16, without fusion and inlining")

print("== Jacobian (Scenarios.Dense100kLayerJacobianFp32)")
v("jac_width", JAC.input_width, "{:,.0f}", "layer width")
v("jac_gb", m(JAC.size, "GB"), "{:.0f}", "full Jacobian, FP32, GB")

print("== ResNet-50 training memory (Scenarios.ResNet50Fp32AdamBatch64, illustrative)")
fp32_b = m(units.BYTES_FP32, "byte")
v("res_params_m", m(RES_LO.workload.parameters, "Mparam"), "{:.1f}", "parameters, millions")
v("fp32_bytes", fp32_b, "{:.0f}", "bytes per FP32 value")
v("res_weights_mb", m(RES_LO.weights, "MB"), "{:.1f}", "FP32 weights, MB")
v("res_adam_mb", m(RES_LO.optimizer_state, "MB"), "{:.1f}", "Adam moments, MB")
v("res_batch", RES_LO.batch_size, "{:.0f}", "batch size")
v("res_act_lo_gb", m(RES_LO.activations, "GB"), "{:.0f}", "saved activations, low, GB")
v("res_act_hi_gb", m(RES_HI.activations, "GB"), "{:.0f}", "saved activations, high, GB")
v("res_total_lo_gb", m(RES_LO.training_total, "GB"), "{:.1f}", "training total, low, GB")
v("res_total_hi_gb", m(RES_HI.training_total, "GB"), "{:.1f}", "training total, high, GB")
v("res_copies_lo", RES_LO.training_copies, "{:.0f}", "total over weights, low")
v("res_copies_hi", RES_HI.training_copies, "{:.0f}", "total over weights, high")
v("res_big_batch", RES_BIG.batch_size, "{:.0f}", "large batch")
v("res_big_act_lo_gb", m(RES_LO.activations_at(RES_BIG.batch_size), "GB"), "{:.0f}", "activations at the large batch, low, GB")
v("res_big_act_hi_gb", m(RES_HI.activations_at(RES_BIG.batch_size), "GB"), "{:.0f}", "activations at the large batch, high, GB")
v("h100_gb", m(H100.memory.capacity, "GiB"), "{:.0f}", "H100 memory (GB, as labeled)")
v("img_side", mlsysim.Datasets.ImageNet.image_width, "{:.0f}", "image side, pixels")
v("img_channels", mlsysim.Datasets.ImageNet.image_channels, "{:.0f}", "color channels")

print("== ResNet-50 training memory, measured (Apple GPU, MPS; data/resnet50_memory.csv)")
rows, n_params = load_resnet_memory()
w_meas = n_params * units.BYTES_FP32
s1 = [r for r in rows if r[1] == 1]
s2 = [r for r in rows if r[1] != 1]
start1, start2 = s1[0][4], s2[0][4]
peak1, peak2 = max(r[4] for r in s1), max(r[4] for r in s2)
after_bwd1 = next(r[4] for r in s1 if r[2] == "after_backward")
after_step1 = next(r[4] for r in s1 if r[2] == "after_step")
st128, pk128 = load_resnet_b128()
rise64, rise128 = peak2 - start2, pk128 - st128
B = units.byte
v("meas_weights_mb", m(w_meas, "MB"), "{:.0f}", "measured model FP32 weights, MB")
v("meas_start1_gb", m(start1 * B, "GB"), "{:.2f}", "step 1 start, GB")
v("meas_peak1_gb", m(peak1 * B, "GB"), "{:.1f}", "step 1 peak, GB")
v("meas_start2_gb", m(start2 * B, "GB"), "{:.2f}", "step 2 start, GB")
v("meas_peak2_gb", m(peak2 * B, "GB"), "{:.1f}", "step 2 peak, GB")
v("meas_peak_mult", (peak2 * B / w_meas).to("").magnitude, "{:.0f}", "peak over weights")
v("meas_adam_mb", m((after_step1 - after_bwd1) * B, "MB"), "{:.0f}", "Adam state added at the first step, MB")
v("meas_rise_ratio", rise128 / rise64, "{:.2f}", "rise, batch 128 over batch 64")
v("meas_big_act_gb", m(rise64 * B * RES_BIG.batch_size / RES_LO.batch_size, "GB"), "{:.0f}",
  "batch-64 rise scaled to the large batch, GB")

print("== Host link (Scenarios.Fp32MillionElementTensorAcrossH100Links)")
links = TENSOR.links
v("hbm_gbps", m(links["HBM"], "GB/s"), "{:,.0f}", "HBM3, GB/s")
v("nvlink_gbps", m(links["NVLink"], "GB/s"), "{:.0f}", "NVLink, one direction, GB/s")
v("pcie_gbps", m(links["PCIe"], "GB/s"), "{:.0f}", "PCIe Gen5 x16, GB/s")
hbm_over_pcie = (links["HBM"] / links["PCIe"]).to("").magnitude
v("hbm_over_pcie", hbm_over_pcie, "{:.0f}", "HBM over PCIe")
v("tensor_mb", m(TENSOR.total_bytes, "MB"), "{:.0f}", "tensor, MB")
v("tensor_hbm_us", m(TENSOR.transfer_time("HBM"), "microsecond"), "{:.1f}", "tensor over HBM, us")
v("tensor_pcie_us", m(TENSOR.transfer_time("PCIe"), "microsecond"), "{:.1f}", "tensor over PCIe, us")

print("== Copy estimate and DataLoader (Scenarios.ImageNetBatch64HostFeed)")
v("feed_batch", FEED.batch_size, "{:.0f}", "batch size")
v("batch_mb", m(FEED.batch_bytes, "MB"), "{:.1f}", "batch, MB")
v("copy_ms", m(FEED.copy_time, "ms"), "{:.1f}", "over PCIe, ms")
v("copy_hbm_us", m(FEED.device_copy_time, "microsecond"), "{:.1f}", "inside HBM, us")
v("step_big_ms", m(FEED.step_times["large model"], "ms"), "{:.0f}", "large model step, ms")
v("step_small_ms", m(FEED.step_times["small model"], "ms"), "{:.1f}", "small model step, ms")
v("copy_share_pct", 100 * FEED.copy_share("large model"), "{:.0f}", "serial copy over the large step, %")
v("dl_workers", FEED.workers, "{:.0f}", "loader workers")
v("dl_workers_word", FEED.workers, "words", "the same, in prose")
v("dl_prefetch", FEED.prefetch_factor, "{:.0f}", "prefetch factor")
v("dl_prefetch_word", FEED.prefetch_factor, "words", "the same, in prose")
v("dl_in_flight", FEED.batches_in_flight, "{:.0f}", "batches in flight")
v("dl_in_flight_word", FEED.batches_in_flight, "words", "the same, in prose")
v("feed_ips", m(FEED.consume_rate, "1/second"), "{:,.0f}", "images per second")
v("feed_mbps", m(FEED.raw_feed, "MB/s"), "{:.1f}", "raw uint8 feed, MB/s")

print("== One training step (Scenarios.MnistTwoLayerTrainingStepOnH100)")
v("step_batch", STEP.batch_size, "{:.0f}", "batch")
v("step_in", STEP.input_width, "{:.0f}", "inputs")
v("step_hidden", STEP.hidden_width, "{:.0f}", "hidden units")
v("step_out", STEP.output_width, "{:.0f}", "outputs")
v("adam_bytes_per_param", m(STEP.optimizer_values_moved * STEP.bytes_per_value, "byte"), "{:.0f}",
  "Adam bytes moved per parameter")
v("step_params", STEP.parameters, "{:,.0f}", "parameters")
v("adam_step_mb", m(STEP.optimizer_step_bytes, "MB"), "{:.2f}", "Adam bytes moved, MB")
v("step_mflop", m(STEP.total_flops, "MFLOP"), "{:.1f}", "step FLOPs, millions")
v("step_mb", m(STEP.total_bytes, "MB"), "{:.1f}", "step traffic, MB")
v("fp32_tflops", m(STEP.peak_rate, "TFLOP/s"), "{:.0f}", "H100 FP32 peak, TFLOP/s")
v("step_compute_us", m(STEP.compute_time, "microsecond"), "{:.1f}", "arithmetic, us")
v("step_memory_us", m(STEP.memory_time, "microsecond"), "{:.1f}", "memory, us")
v("step_entries", STEP.host.operations, "{:.0f}", "Python entries")
v("step_entries_word", STEP.host.operations, "words", "the same, in prose")
v("step_launches", STEP.host.launches, "{:.0f}", "kernel launches")
v("step_launches_word", STEP.host.launches, "words", "the same, in prose")
v("step_host_us", m(STEP.host.host_time, "microsecond"), "{:.0f}", "host, us")
mt = load_mnist_times()
small_b, big_b = STEP.batch_size, max(mt)
v("meas_small_batch_ms", m(Q_(mt[small_b][0], "microsecond"), "ms"), "{:.2f}", "measured eager step, batch 32, ms")
v("meas_big_batch", big_b, "{:,.0f}", "large measured batch")
v("meas_big_batch_ms", m(Q_(mt[big_b][0], "microsecond"), "ms"), "{:.2f}", "measured eager step, large batch, ms")
v("meas_trials", mt[small_b][1], "{:.0f}", "trials per batch")
v("meas_trials_word", mt[small_b][1], "words", "the same, in prose")
v("meas_batch_mult_word", big_b / small_b, "words", "images per step, large over small, in prose")

print("== Framework paths (Scenarios.EmbeddedLinuxFrameworkPaths, hypothetical)")
onnx_path, tfl_path = (PATHS.paths[k] for k in PATHS.paths)
v("path_lat_budget_ms", m(PATHS.latency_budget, "ms"), "{:.0f}", "p99 budget, ms")
v("path_mem_budget_mb", m(PATHS.memory_budget, "MB"), "{:.0f}", "memory budget, MB")
v("path_a_ms", m(onnx_path["latency"], "ms"), "{:.0f}", "path A p99, ms")
v("path_a_mb", m(onnx_path["memory"], "MB"), "{:.0f}", "path A peak memory, MB")
v("path_b_ms", m(tfl_path["latency"], "ms"), "{:.0f}", "path B p99, ms")
v("path_b_mb", m(tfl_path["memory"], "MB"), "{:.0f}", "path B peak memory, MB")


def figure_values():
    """The unrounded values make_figs.py draws."""
    return {
        "decode_read_ms": m(GPT2.device_time, "ms"),
        "decode_launch_ms": m(GPT2.launch_time, "ms"),
        "decode_host_ms": m(GPT2.host_time, "ms"),
        "dispatch_rows": [(SMALL_OP, "small_kernel_us"), (LARGE_OP, "large_kernel_us")],
        "pt2": PT2,
        "ops": OPS,
        "resnet_weights_mb": m(RES_LO.weights, "MB"),
        "resnet_total_gb": (m(RES_LO.training_total, "GB"), m(RES_HI.training_total, "GB")),
        "resnet_act_big_gb": (m(RES_LO.activations_at(RES_BIG.batch_size), "GB"),
                              m(RES_HI.activations_at(RES_BIG.batch_size), "GB")),
        "h100_gb": m(H100.memory.capacity, "GiB"),
        "bandwidth_gbps": {k: m(b, "GB/s") for k, b in links.items()},
        "hbm_over_pcie": hbm_over_pcie,
        "mnist_compute_us": m(STEP.compute_time, "microsecond"),
        "mnist_memory_us": m(STEP.memory_time, "microsecond"),
        "mnist_host_us": m(STEP.host.host_time, "microsecond"),
        "fp32_bytes": fp32_b,
    }


if __name__ == "__main__":
    write(__file__)
