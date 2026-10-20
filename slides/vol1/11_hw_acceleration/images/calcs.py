#!/usr/bin/env python3
"""Every number on the Lecture 11 (Hardware Acceleration, Part 1) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Hardware, Hardware.Tech, Models, Literature) and the
calculations (the hardware-acceleration Scenarios.*); the Hardware
Acceleration chapter reads the same scenarios. This file only selects values,
converts units, and formats them; it types no numbers. Each value is recorded
with v() under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports this module, so figures and slides
share one computation.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.engine.serving_profile import resnet50_h100_profile  # noqa: E402

Hardware, Models, Literature, Scenarios = mlsysim.Hardware, mlsysim.Models, mlsysim.Literature, mlsysim.Scenarios
TECH, DM = Hardware.Tech, Literature.DeviceMeasurements
H100, A100, V100, B200, T4 = (Hardware.Cloud.H100, Hardware.Cloud.A100, Hardware.Cloud.V100,
                              Hardware.Cloud.B200, Hardware.Cloud.T4)
P100, TPU1, TPU4, H100_PCIE = (Hardware.Cloud.P100, Hardware.Cloud.TPUv1, Hardware.Cloud.TPUv4,
                               Hardware.Cloud.H100_PCIe)
GPT2, RESNET = Models.Language.GPT2, Models.Vision.ResNet50
FP16, INT8 = mlsysim.BYTES_FP16, mlsysim.BYTES_INT8

PROFILE = resnet50_h100_profile()          # Model Compression's ResNet-50 INT8 profile
CPU_GAP = Scenarios.H100OverReferenceCpu
REUSE = Scenarios.SystolicArrayReuseEnergy
MMA = Scenarios.TensorCoreMmaM16n8k16
WALL = Scenarios.DenseParametersVsBandwidthSince2014
HBM_CYCLES = Scenarios.HbmAccessAt2Ghz
GENS = Scenarios.NvidiaGenerationsV100ToB200
ENVELOPE = Scenarios.AcceleratorEnvelopeSinceTpuV2
K_A100, K_H100 = Scenarios.KernelAt200FlopPerByteA100, Scenarios.KernelAt200FlopPerByteH100
QKV, SOFTMAX = Scenarios.Gpt2XlQkvProjectionPrefill, Scenarios.Gpt2XlSoftmaxPrefill
LN, HEAD = Scenarios.Gpt2XlLayerNormPrefill, Scenarios.ResNet50ClassifierHeadBatch32
S3, S3_INT8, S3_MIXED = (Scenarios.ResNet50Stage3ConvBatch32, Scenarios.ResNet50Stage3ConvBatch32Int8,
                         Scenarios.ResNet50Stage3ConvBatch32Int8Weights)
S2, S2_INT8 = Scenarios.ResNet50Stage2ConvBatch32, Scenarios.ResNet50Stage2ConvBatch32Int8
P2, P2_INT8 = Scenarios.ResNet50Stage2ExpansionBatch32, Scenarios.ResNet50Stage2ExpansionBatch32Int8
ROOF_DECODE, ROOF_LN = Scenarios.Gpt2XlDecodeH100Roofline, Scenarios.Gpt2XlLayerNormH100Roofline
ROOF_HEAD, ROOF_CONV, ROOF_QKV = (Scenarios.ResNet50ClassifierHeadH100Roofline,
                                  Scenarios.ResNet50Stage3ConvH100Roofline, Scenarios.Gpt2XlQkvH100Roofline)
BATCHES = (Scenarios.Dense2048Fp16Batch1, Scenarios.Dense2048Fp16Batch32, Scenarios.Dense2048Fp16Batch512)
FLOOR8, FLOOR16 = Scenarios.ResNet50Batch1H100Int8PeakRate, Scenarios.ResNet50Batch1H100Fp16PeakRate
AMDAHL_RN = Scenarios.ResNet50InferenceH100OverServerCpu
AMDAHL_GPT, AMDAHL_BW = Scenarios.Gpt2DecodeH100OverServerCpu, Scenarios.Gpt2DecodeH100OverServerCpuBandwidth


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


print("== Hook: the first TPU (Jouppi et al. 2017; Hardware.Cloud.TPUv1)")
v("voice_minutes_words", lit(DM.TpuV1VoiceSearchMinutesPerDay), "words", "voice-search minutes a day, in prose")
v("tpu_mhz", m(TPU1.compute.clock_rate, "MHz"), "{:.0f}", "TPU v1 clock, MHz")
v("tpu_dim", TPU1.compute.systolic_array_dim, "{:.0f}", "TPU v1 array width")
v("tpu_macs", TPU1.compute.systolic_macs_per_cycle, "{:,.0f}", "TPU v1 MACs per cycle")
v("tpu_buffer_mb", m(TPU1.memory.sram_capacity, "MiB"), "{:.0f}", "TPU v1 Unified Buffer, labeled MB (24 MiB)")
v("tpu_thr_lo", lit(DM.TpuV1ThroughputGainLow), "{:.0f}", "throughput gain, low")
v("tpu_thr_hi", lit(DM.TpuV1ThroughputGainHigh), "{:.0f}", "throughput gain, high")
v("tpu_ppw_lo", lit(DM.TpuV1PerfPerWattGainLow), "{:.0f}", "performance-per-watt gain, low")
v("tpu_ppw_hi", lit(DM.TpuV1PerfPerWattGainHigh), "{:.0f}", "performance-per-watt gain, high")
v("tpu_workloads_words", lit(DM.TpuV1EvaluatedWorkloads), "words", "evaluated workloads, in prose")

print("== Recall: ResNet-50 INT8 on the H100 (Model Compression's profile, engine.serving_profile)")
lat32, lat8 = PROFILE.tensorrt("fp32", batch=1), PROFILE.tensorrt("int8", batch=1)
v("lat_fp32_ms", m(lat32, "ms"), "{:.2f}", "batch-one FP32 latency, ms")
v("lat_int8_ms", m(lat8, "ms"), "{:.2f}", "batch-one INT8 latency, ms")
v("lat_int8_ms_short", m(lat8, "ms"), "{:.1f}", "the same, one decimal")
v("int8_b1_gain", ratio(lat32, lat8), "{:.1f}", "batch-one INT8 speedup")
v("int8_big_gain", PROFILE.large_batch_throughput("int8") / PROFILE.large_batch_throughput("fp32"), "{:.1f}",
  "large-batch INT8 throughput gain")

print("== The chip: H100 against a reference CPU; the TPU v1 floor plan")
v("h100_fp16", m(H100.compute.peak_flops, "TFLOP/s"), "{:,.0f}", "H100 dense FP16, TFLOP/s")
v("cpu_lo", m(CPU_GAP.baseline_low, "TFLOP/s"), "{:.0f}", "reference CPU, low, TFLOP/s")
v("cpu_hi", m(CPU_GAP.baseline_high, "TFLOP/s"), "{:.0f}", "reference CPU, high, TFLOP/s")
v("cpu_gap_lo", round(CPU_GAP.ratio_low, -1), "{:.0f}", "H100 over the high CPU, to the ten (as the chapter)")
v("cpu_gap_hi", round(CPU_GAP.ratio_high, -1), "{:.0f}", "H100 over the low CPU, to the ten")
v("h100_transistors_b", H100.transistor_count / 10 ** 9, "{:.0f}", "H100 transistors, billions")
v("die_buffers", lit(DM.TpuV1DieBuffersPct), "{:.0f}", "TPU v1 die in buffers, %")
v("die_compute", lit(DM.TpuV1DieComputePct), "{:.0f}", "TPU v1 die in compute, %")
v("die_io", lit(DM.TpuV1DieIoPct), "{:.0f}", "TPU v1 die in I/O, %")
v("die_control", lit(DM.TpuV1DieControlPct), "{:.0f}", "TPU v1 die in control, %")
v("die_ub", lit(DM.TpuV1DieUnifiedBufferPct), "{:.0f}", "TPU v1 Unified Buffer, %")
v("die_mxu", lit(DM.TpuV1DieMatrixUnitPct), "{:.0f}", "TPU v1 matrix unit, %")
v("h100_cuda_cores", H100.compute.fp32_core_count, "{:,.0f}", "H100 FP32 (CUDA) cores")
v("h100_tensor_cores", H100.compute.tensor_core_count, "{:,.0f}", "H100 Tensor Cores")

print("== Energy ladder (Hardware.Tech, Horowitz 2014 at 45 nm; Jouppi et al. 2021 at 7 nm)")
e_dram = m(TECH.Memory.DRAM.energy_per_access, "pJ")
e_mul = m(TECH.Op.FlopFp32.energy, "pJ/flop")
e_add = m(TECH.Op.AddFp32.energy, "pJ/flop")
e_sram = m(TECH.Memory.SRAM_8KB.energy_per_access, "pJ")
v("e_int8_add", m(TECH.Op.AddInt8.energy, "pJ/flop"), "{:.2f}", "INT8 add, pJ")
v("e_fp32_add", e_add, "{:.1f}", "FP32 add, pJ")
v("e_fp32_mul", e_mul, "{:.1f}", "FP32 multiply, pJ")
v("e_sram8k", e_sram, "{:.0f}", "8 KB SRAM read, pJ")
v("e_dram", e_dram, "{:.0f}", "32-bit DRAM read, pJ")
v("dram_per_mul", e_dram / e_mul, "{:.0f}", "DRAM read over FP32 multiply")
v("dram_per_sram", e_dram / e_sram, "{:.0f}", "DRAM read over 8 KB SRAM read")
v("dram_per_fma", e_dram / (e_mul + e_add), "{:.0f}", "DRAM read over FP32 multiply-add (Lecture 6)")
v("e_fp32_mul_7nm", m(TECH.Op.FlopFp32_7nm.energy, "pJ/flop"), "{:.2f}", "FP32 multiply at 7 nm, pJ")
v("mul_gain_7nm", e_mul / m(TECH.Op.FlopFp32_7nm.energy, "pJ/flop"), "{:.1f}", "45 nm over 7 nm multiply")
v("e_ddr64_45", m(TECH.Memory.DdrDram_Access64b_45nm.energy_per_access, "pJ"), "{:,.0f}",
  "DDR 64-bit access at 45 nm, pJ")
v("e_ddr64_7", m(TECH.Memory.DdrDram_Access64b_7nm.energy_per_access, "pJ"), "{:,.0f}",
  "DDR 64-bit access at 7 nm, pJ")

print("== Matrix units (Hardware.Cloud.P100, V100; Scenarios.TensorCoreMmaM16n8k16)")
v("p100_fp16", m(P100.compute.peak_flops, "TFLOP/s"), "{:.1f}", "P100 FP16, TFLOP/s")
v("v100_fp16", m(V100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "V100 FP16 Tensor Core, TFLOP/s")
v("tc_gain", ratio(V100.compute.peak_flops, P100.compute.peak_flops), "{:.0f}", "V100 over P100 FP16 peak")
v("mma_macs", MMA.macs, "{:,.0f}", "MACs in one m16n8k16 instruction")
v("mma_per_thread", MMA.macs_per_thread, "{:.0f}", "scalar FMAs per thread it replaces")
v("resnet_classes", HEAD.out_features, "{:,.0f}", "ResNet-50 classes (softmax width)")

print("== Systolic array (Hardware.Cloud.TPUv4; Scenarios.SystolicArrayReuseEnergy)")
v("tpu4_dim", TPU4.compute.systolic_array_dim, "{:.0f}", "TPU v4 array width")
v("tpu4_macs", TPU4.compute.systolic_macs_per_cycle, "{:,.0f}", "TPU v4 MACs per cycle")
v("sys_dim", REUSE.array_dim, "{:.0f}", "array width in the energy estimate")
v("sys_edge_loads", REUSE.edge_loads, "{:.0f}", "operand loads per array pass")
v("sys_edge_loads_words", REUSE.edge_loads, "words", "the same, in prose")
v("naive_accesses", REUSE.naive_accesses, "{:.0f}", "DRAM accesses per MAC, naive")
v("naive_accesses_words", REUSE.naive_accesses, "words", "the same, in prose")
v("mac_pj", m(REUSE.mac_energy, "pJ"), "{:.0f}", "one MAC, pJ")
v("naive_pj", m(REUSE.naive_energy, "pJ"), "{:,.0f}", "naive energy per MAC, pJ")
v("sys_access", REUSE.systolic_accesses, "{:.3f}", "systolic DRAM accesses per MAC")
v("sys_pj", m(REUSE.systolic_energy, "pJ"), "{:.0f}", "systolic energy per MAC, pJ")
v("sys_ratio", REUSE.ratio, "{:.1f}", "naive over systolic")
v("sys_ratio_int", REUSE.ratio, "{:.0f}", "the same, whole")

print("== Memory wall (Scenarios.DenseParametersVsBandwidthSince2014)")
v("wall_params", round(WALL.parameter_growth, -2), "{:,.0f}", "dense parameter growth, to the hundred")
v("wall_bw", WALL.bandwidth_growth, "{:.0f}", "per-device bandwidth growth")
v("gpt2_gb", m(GPT2.size_in_bytes(FP16), "GB"), "{:.0f}", "GPT-2 XL FP16 weights, GB")

print("== Memory tiers (Hardware.Cloud.H100, Hardware.Tech.Memory; Scenarios.HbmAccessAt2Ghz)")
v("e_reg", m(TECH.Memory.Register.energy_per_access, "pJ"), "{:.1f}", "register access, pJ")
v("smem_kb", m(H100.memory.shared_memory_per_sm, "KiB"), "{:.0f}", "shared memory per SM, labeled KB (228 KiB)")
v("smem_ns", m(TECH.Memory.HopperSharedMemory.latency, "ns"), "{:.0f}", "Hopper shared-memory latency, ns")
v("l2_mb", m(H100.memory.l2_cache, "MiB"), "{:.0f}", "H100 L2, labeled MB (50 MiB)")
v("l2_ns", m(TECH.Memory.HopperL2.latency, "ns"), "{:.0f}", "Hopper L2 latency, ns")
v("hbm_gb", m(H100.memory.capacity, "GiB"), "{:.0f}", "H100 HBM, labeled GB (80 GiB)")
v("h100_bw", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM bandwidth, TB/s")
v("hbm_ns", m(HBM_CYCLES.latency, "ns"), "{:.0f}", "Hopper HBM latency, ns (Luo et al. 2024)")
v("e_dram_byte", m(TECH.Memory.DRAM.energy_per_byte, "pJ/B"), "{:.0f}", "DRAM energy per byte, pJ")
v("clock_ghz", m(HBM_CYCLES.clock, "GHz"), "{:.0f}", "illustrative clock, GHz")
v("cycle_ps", m(HBM_CYCLES.cycle_time, "ps"), "{:.0f}", "one cycle at that clock, ps")
v("hbm_cycles", HBM_CYCLES.cycles, "{:.0f}", "HBM access in cycles")
cnc = {"l1": lit(DM.H100PcieL1LatencyNs), "l2": lit(DM.H100PcieL2LatencyNs),
       "l2far": lit(DM.H100PcieFarL2LatencyNs), "hbm": lit(DM.H100PcieHbmLatencyNs)}
v("cnc_l1", cnc["l1"], "{:.1f}", "measured L1, ns")
v("cnc_l2", cnc["l2"], "{:.1f}", "measured L2, ns")
v("cnc_l2far", cnc["l2far"], "{:.0f}", "measured far L2, ns")
v("cnc_hbm", cnc["hbm"], "{:.1f}", "measured HBM, ns")
v("cnc_hbm_l1", cnc["hbm"] / cnc["l1"], "{:.0f}", "measured HBM over L1")
v("pcie_sms", H100_PCIE.compute.sm_count, "{:.0f}", "H100 PCIe SMs")
v("pcie_bw", m(H100_PCIE.memory.bandwidth, "TB/s"), "{:.0f}", "H100 PCIe HBM, TB/s")

print("== From HBM alone (H100; intensity of one FLOP per byte)")
need = H100.compute.peak_flops / mlsysim.core.units.Q_(1, "flop/byte")
v("need_tbs", m(need, "TB/s"), "{:,.0f}", "bandwidth needed at 1 FLOP/byte, TB/s")
v("need_short", ratio(need, H100.memory.bandwidth), "{:.0f}", "shortfall against HBM")

print("== Bandwidth taper (Hardware.Cloud.H100, Systems.Fabrics.InfiniBand_NDR)")
net = mlsysim.Systems.Fabrics.InfiniBand_NDR.bandwidth
v("hbm_gbs", m(H100.memory.bandwidth, "GB/s"), "{:,.0f}", "HBM, GB/s")
v("nvlink_gbs", m(H100.nvlink.bandwidth, "GB/s"), "{:.0f}", "NVLink 4, GB/s")
v("pcie_gbs", m(H100.interconnect.bandwidth, "GB/s"), "{:.0f}", "PCIe Gen5 x16, GB/s")
v("net_gbs", m(net, "GB/s"), "{:.0f}", "InfiniBand NDR, GB/s")
v("hbm_pcie", ratio(H100.memory.bandwidth, H100.interconnect.bandwidth), "{:.0f}", "HBM over PCIe")

print("== Roofline: A100 recall and the three H100 ridges")
v("a100_fp16", m(A100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "A100 dense FP16, TFLOP/s")
v("a100_bw", m(A100.memory.bandwidth, "TB/s"), "{:.2f}", "A100 HBM, TB/s")
v("a100_ridge", m(A100.ridge_point(), "flop/byte"), "{:.0f}", "A100 FP16 ridge")
v("h100_fp32", m(H100.compute.precision_flops["fp32"], "TFLOP/s"), "{:.0f}", "H100 FP32, TFLOP/s")
v("h100_int8", m(H100.compute.precision_flops["int8"], "TOPS"), "{:,.0f}", "H100 INT8, TOPS")
r32 = m(H100.ridge_point("fp32"), "flop/byte")
r16 = m(H100.ridge_point(), "flop/byte")
r8 = m(H100.ridge_point("int8"), "flop/byte")
v("ridge32", r32, "{:.0f}", "FP32 ridge")
v("ridge16", r16, "{:.1f}", "FP16 ridge")
v("ridge16_int", r16, "{:.0f}", "FP16 ridge, whole")
v("ridge8", r8, "{:.1f}", "INT8 ridge")
v("ridge8_int", r8, "{:.0f}", "INT8 ridge, whole")
v("int8_over_fp16", ratio(H100.compute.precision_flops["int8"], H100.compute.peak_flops), "{:.0f}",
  "H100 INT8 over FP16 peak")
v("fp16_over_fp32", ratio(H100.compute.peak_flops, H100.compute.precision_flops["fp32"]), "{:.0f}",
  "H100 FP16 over FP32 peak")

print("== The ridge across generations (Scenarios.KernelAt200FlopPerByte*, NvidiaGenerationsV100ToB200)")
v("k_ai", m(K_A100.point_intensity, "flop/byte"), "{:.0f}", "kernel intensity, FLOP/byte")
v("k_compute_ratio", ratio(H100.compute.peak_flops, A100.compute.peak_flops), "{:.1f}", "H100 over A100 FP16 peak")
v("k_bw_ratio", ratio(H100.memory.bandwidth, A100.memory.bandwidth), "{:.1f}", "H100 over A100 bandwidth")
v("k_a100", m(K_A100.attainable, "TFLOP/s"), "{:.0f}", "A100 attainable, TFLOP/s")
v("k_h100", m(K_H100.attainable, "TFLOP/s"), "{:.0f}", "H100 attainable, TFLOP/s")
v("k_gain", ratio(K_H100.attainable, K_A100.attainable), "{:.1f}", "A100 to H100 gain")
v("k_gain_2", ratio(K_H100.attainable, K_A100.attainable), "{:.2f}", "the same, two decimals")
RIDGE_PLOT = [h for h in GENS.accelerators if h is not Hardware.Cloud.H200]  # the H200 keeps the H100's compute
for h in RIDGE_PLOT:
    key = h.name.split()[-1].lower()
    v(f"ridge_{key}", m(h.ridge_point(), "flop/byte"), "{:.0f}", f"{h.name} FP16 ridge")
v("gen_first_year", GENS.accelerators[0].release_year, "{:.0f}", "V100 year")
v("gen_last_year", GENS.accelerators[-1].release_year, "{:.0f}", "B200 year")
v("gen_compute", GENS.compute_growth, "{:.0f}", "dense FP16 growth, V100 to B200")
v("gen_bw", GENS.bandwidth_growth, "{:.1f}", "bandwidth growth, V100 to B200")
v("ridge_years_words", H100.release_year - V100.release_year, "words", "years from V100 to H100, in prose")

print("== Across vendors (Scenarios.AcceleratorEnvelopeSinceTpuV2)")
v("env_devices", ENVELOPE.device_count, "{:.0f}", "accelerators in the table")
v("env_compute", ENVELOPE.compute_growth, "{:.0f}", "best dense FP16 growth since TPU v2")
v("env_bw", ENVELOPE.bandwidth_growth, "{:.0f}", "best bandwidth growth since TPU v2")
v("env_best_bw", ENVELOPE.bandwidth_envelope[-1][1] / 10 ** 12, "{:.0f}", "best bandwidth, TB/s")
v("env_since_year", int(ENVELOPE.best_bandwidth_since.split("-")[0]), "{:.0f}", "year the best bandwidth was reached")

print("== Five layers on the H100 FP16 roofline (layer and roofline Scenarios.*)")
v("qkv_batch", QKV.batch_size // GPT2.context_length, "{:.0f}", "QKV prefill batch, sequences")
v("conv_batch", S3.batch_size, "{:.0f}", "ResNet-50 batch")
v("qkv_gflop", m(QKV.flops, "GFLOP"), "{:.1f}", "QKV FLOPs, G")
v("qkv_mb", m(QKV.traffic_bytes, "MB"), "{:.1f}", "QKV bytes, MB")
v("qkv_ai", QKV.traffic_intensity, "{:,.1f}", "QKV intensity")
v("conv_gflop", m(S3.flops, "GFLOP"), "{:.1f}", "stage-3 conv FLOPs, G")
v("conv_mb", m(S3.traffic_bytes, "MB"), "{:.1f}", "stage-3 conv bytes, MB")
v("conv_ai", S3.traffic_intensity, "{:.1f}", "stage-3 conv intensity")
v("head_mflop", m(HEAD.flops, "MFLOP"), "{:.1f}", "FC head FLOPs, M")
v("head_mb", m(HEAD.traffic_bytes, "MB"), "{:.1f}", "FC head bytes, MB")
v("head_ai", HEAD.traffic_intensity, "{:.1f}", "FC head intensity")
v("ln_mflop", m(LN.flops, "MFLOP"), "{:.1f}", "LayerNorm FLOPs, M")
v("ln_mb", m(LN.traffic_bytes, "MB"), "{:.1f}", "LayerNorm bytes, MB")
v("ln_ai", LN.intensity, "{:.1f}", "LayerNorm intensity")
v("ln_ops_words", LN.flops_per_element, "words", "LayerNorm operations per element, in prose")
v("dec_gflop", m(GPT2.inference_flops, "GFLOP"), "{:.0f}", "decode FLOPs per token, G")
v("dec_ai", m(ROOF_DECODE.point_intensity, "flop/byte"), "{:.0f}", "decode intensity")
v("head_tflops", m(ROOF_HEAD.attainable, "TFLOP/s"), "{:.0f}", "FC head ceiling, TFLOP/s")
v("head_pct", pct(ROOF_HEAD.peak_fraction), "{:.1f}", "FC head ceiling, % of peak")
v("ln_tflops", m(ROOF_LN.attainable, "TFLOP/s"), "{:.1f}", "LayerNorm ceiling, TFLOP/s")
v("ln_pct", pct(ROOF_LN.peak_fraction), "{:.1f}", "LayerNorm ceiling, % of peak")
v("dec_tflops", m(ROOF_DECODE.attainable, "TFLOP/s"), "{:.1f}", "decode ceiling, TFLOP/s")
v("dec_pct", pct(ROOF_DECODE.peak_fraction), "{:.1f}", "decode ceiling, % of peak")
v("top_pct", pct(ROOF_CONV.peak_fraction), "{:.0f}", "compute-bound layers, % of peak")
v("softmax_ai", SOFTMAX.intensity, "{:.2f}", "softmax intensity")
v("nsight_dram_pct", lit(DM.NsightComputeRooflineExampleDramThroughputPct), "{:.2f}",
  "Nsight guide example, DRAM throughput, % of peak")

print("== Batch and intensity (Scenarios.Dense2048Fp16Batch*)")
b1, b32, b512 = BATCHES
v("dense_width", b1.in_features, "{:.0f}", "dense layer width")
v("batch_mid", b32.batch_size, "{:.0f}", "middle batch")
v("batch_big", b512.batch_size, "{:.0f}", "large batch")
v("ai_b1", b1.traffic_intensity, "{:.0f}", "intensity at batch 1")
v("ai_b32", b32.traffic_intensity, "{:.0f}", "intensity at batch 32")
v("ai_b512", b512.traffic_intensity, "{:.0f}", "intensity at batch 512")
v("ai_b512_1", b512.traffic_intensity, "{:.1f}", "the same, one decimal")
v("crossing", b1.batch_for_traffic_intensity(r16), "{:.0f}", "batch where intensity reaches the FP16 ridge")

print("== INT8 (H100 peaks; Scenarios.ResNet50Batch1H100*PeakRate and the ResNet-50 layer scenarios)")
v("resnet_gflop", m(RESNET.inference_flops, "GFLOP"), "{:.1f}", "ResNet-50 GFLOP per image")
v("floor16_us", m(FLOOR16.result.latency_compute, "us"), "{:.0f}", "arithmetic at the FP16 peak, us")
v("floor8_us", m(FLOOR8.result.latency_compute, "us"), "{:.0f}", "arithmetic at the INT8 peak, us")
v("s3_fp16", S3.traffic_intensity, "{:.0f}", "stage-3 3x3 at FP16")
v("s3_int8", S3_INT8.traffic_intensity, "{:,.0f}", "stage-3 3x3 at INT8")
v("s3_mixed", S3_MIXED.traffic_intensity, "{:.0f}", "stage-3 3x3, INT8 weights, FP16 activations")
v("s2_fp16", S2.traffic_intensity, "{:.0f}", "stage-2 3x3 at FP16")
v("s2_int8", S2_INT8.traffic_intensity, "{:.0f}", "stage-2 3x3 at INT8")
v("p2_fp16", P2.traffic_intensity, "{:.0f}", "stage-2 1x1 at FP16")
v("p2_int8", P2_INT8.traffic_intensity, "{:.0f}", "stage-2 1x1 at INT8")

print("== Amdahl (Scenarios.*OverServerCpu*)")
v("p_resnet", AMDAHL_RN.fraction, "{:.2f}", "ResNet-50 accelerated fraction")
v("p_resnet_pct", pct(AMDAHL_RN.fraction), "{:.0f}", "the same, %")
v("serial_resnet", 1 - AMDAHL_RN.fraction, "{:.2f}", "ResNet-50 unaccelerated fraction")
v("serial_resnet_pct", pct(1 - AMDAHL_RN.fraction), "{:.0f}", "the same, %")
v("p_gpt", AMDAHL_GPT.fraction, "{:.2f}", "decode accelerated fraction")
v("p_gpt_pct", pct(AMDAHL_GPT.fraction), "{:.0f}", "the same, %")
v("cpu_tops", m(AMDAHL_RN.baseline_rate, "TOPS"), "{:.0f}", "assumed server-CPU INT8 rate, TOPS")
v("gain", AMDAHL_RN.gain, "{:.0f}", "matrix-path gain")
v("amdahl_rn", AMDAHL_RN.speedup, "{:.1f}", "ResNet-50 speedup")
v("amdahl_gpt", AMDAHL_GPT.speedup, "{:.1f}", "decode speedup")
v("bw_gain", AMDAHL_BW.gain, "{:.0f}", "bandwidth gain")
v("amdahl_bw", AMDAHL_BW.speedup, "{:.1f}", "decode speedup at the bandwidth gain")
v("amdahl_ceiling", AMDAHL_GPT.ceiling, "{:.0f}", "decode ceiling")

print("== Lighthouses (Models.*, Hardware.Cloud.H100)")
LH = {"resnet": RESNET, "mobilenet": Models.Vision.MobileNetV2, "gpt2": GPT2, "kws": Models.Tiny.DS_CNN}
v("lh_resnet16", m(RESNET.size_in_bytes(FP16), "MB"), "{:.1f}", "ResNet-50 FP16, MB")
v("lh_resnet8", m(RESNET.size_in_bytes(INT8), "MB"), "{:.1f}", "ResNet-50 INT8, MB")
v("lh_mobilenet16", m(LH["mobilenet"].size_in_bytes(FP16), "MB"), "{:.1f}", "MobileNetV2 FP16, MB")
v("lh_mobilenet8", m(LH["mobilenet"].size_in_bytes(INT8), "MB"), "{:.1f}", "MobileNetV2 INT8, MB")
v("lh_mobilenet_gflop", m(LH["mobilenet"].inference_flops, "GFLOP"), "{:.1f}", "MobileNetV2 GFLOP")
v("lh_gpt8", m(GPT2.size_in_bytes(INT8), "GB"), "{:.1f}", "GPT-2 XL INT8, GB")
v("lh_dlrm_gb", m(Models.Recommendation.DLRM.model_size, "GB"), "{:.0f}", "DLRM tables, GB")
v("lh_kws16", m(LH["kws"].size_in_bytes(FP16), "KB"), "{:.0f}", "KWS FP16, KB")
v("lh_kws8", m(LH["kws"].size_in_bytes(INT8), "KB"), "{:.0f}", "KWS INT8, KB")
v("lh_kws_mflop", m(LH["kws"].inference_flops, "MFLOP"), "{:.0f}", "KWS MFLOP")

print("== True or false (Hardware.Cloud.A100, T4, H100 datasheet rows)")
v("a100_t4_peak", ratio(A100.compute.peak_flops, T4.compute.peak_flops), "{:.1f}", "A100 over T4 peak")
v("a100_t4_bw", ratio(A100.memory.bandwidth, T4.memory.bandwidth), "{:.1f}", "A100 over T4 bandwidth")
v("ds_tf32", m(H100.compute.precision_flops["tf32_sparse"], "TFLOP/s"), "{:,.0f}", "datasheet TF32*, TFLOPS")
v("ds_fp16", m(H100.compute.precision_flops["fp16_sparse"], "TFLOP/s"), "{:,.0f}", "datasheet FP16*, TFLOPS")
v("ds_int8", m(H100.compute.precision_flops["int8_sparse"], "TOPS"), "{:,.0f}", "datasheet INT8*, TOPS")

print("== Next lecture (Boehm 2022, Literature.CompilerResults)")
v("mapping_gain", lit(Literature.CompilerResults.BoehmSgemmCublasGflops)
  / lit(Literature.CompilerResults.BoehmSgemmNaiveGflops), "{:.0f}", "cuBLAS over the naive SGEMM kernel")

if __name__ == "__main__":
    write(__file__)
