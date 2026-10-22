#!/usr/bin/env python3
"""Every number on the Lecture 10b (Model Compression, Part 2) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (format layouts, Hardware.Tech energies, Models,
Platforms, the quantization and compression Scenarios.*, and the cited
results in Literature.CompressionResults and Literature.Series) and the
calculations (physics.numerics for grids, scales, and block formats); the
Model Compression chapter reads the same scenarios. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex. make_figs.py imports this module, so figures
and slides share one computation.

Two printed tool outputs are quoted on the slides as the tools printed them
and are not regenerated here: the PyTorch observer run (torch_qparams.py) and
the llama-bench excerpt; the values they print are recorded below from the
same sources for the slide text around them.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

from mlsysim.core.units import (  # noqa: E402
    BYTES_FP16, BYTES_FP32, BYTES_INT4, BYTES_INT8, FLOAT_FORMAT_BITS, FP8_FORMAT_BITS, FP32_BITS, INT4_BITS,
    INT8_BITS, SIMD_REGISTER_BITS)
from mlsysim.physics import (  # noqa: E402
    calc_affine_scale, calc_block_bits_per_weight, calc_float_range, calc_float_values, calc_int_levels,
    calc_int_max)

Models, Platforms, Datasets = mlsysim.Models, mlsysim.Platforms, mlsysim.Datasets
Scenarios, Literature = mlsysim.Scenarios, mlsysim.Literature
CR, SERIES = Literature.CompressionResults, Literature.Series

KQ = SERIES.KrishnamoorthiQuantization.entries
WU = SERIES.WuInt8Calibration.entries
DET = SERIES.DettmersInt8Perplexity.entries
GGUF = SERIES.Llama2_7BGgufFiles.entries
BENCH = SERIES.LlamaBenchAppleSilicon.entries

BLUR = Scenarios.MobileNetV2BlurPipelinePhone
GPT2_READ = Scenarios.Gpt2XlFp16WeightReadPhone
DEC16, DEC4 = Scenarios.Gpt2XlFp16DecodeFullContextPhone, Scenarios.Gpt2XlInt4DecodeFullContextPhone
ENERGY = Scenarios.Int8VsFp32OperationEnergy45nm
TOY = Scenarios.ThreeBitGridMinusOneToTwoAndHalf
NAPKIN = Scenarios.Uint8RangeMinusOneToThree
LISTING = Scenarios.SymmetricInt8FourWeights
SYM_UNIT, ASYM = Scenarios.SymmetricInt8UnitRange, Scenarios.AsymmetricInt8RangeMinusHalfToOneAndHalf
RELU_SYM, RELU_ASYM = Scenarios.Relu6SymmetricInt8, Scenarios.Relu6AsymmetricUint8
CLIP_FULL, CLIP_CUT = Scenarios.ActivationsCoveredToTwoAndHalfUint8, Scenarios.ActivationsClippedAtOnePointOneUint8
DOT = Scenarios.Int8DotProductThreeTerms
PROFILE = Scenarios.ResNet50Int8ProfileH100
H100 = mlsysim.Hardware.Cloud.H100


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def pct(x):
    """A fraction as percent."""
    return 100 * x


def kq(model, method, weight_bits, granularity="", scheme=""):
    """Krishnamoorthi 2018 top-1 (percent) for one row: float, or ptq/qat at ``weight_bits``."""
    rows = [r for r in KQ if r["model"] == model and r["method"] == method and r["weight_bits"] == weight_bits
            and r["granularity"] == granularity and r["scheme"] == scheme]
    assert len(rows) == 1, (model, method, weight_bits, granularity, scheme)
    return pct(rows[0]["top1"])


MBV2, R50, MBV1 = "MobileNetV2 1.0 224", "ResNet-v1-50", "MobileNet-v1 1.0 224"
MBV2_FLOAT, R50_FLOAT, MBV1_FLOAT = (kq(x, "float", FP32_BITS) for x in (MBV2, R50, MBV1))
MBV2_TENSOR = kq(MBV2, "ptq", INT8_BITS, "per_tensor", "asymmetric")
R50_TENSOR = kq(R50, "ptq", INT8_BITS, "per_tensor", "asymmetric")
MBV2_CHANNEL = kq(MBV2, "ptq", INT8_BITS, "per_channel", "symmetric")
R50_CHANNEL = kq(R50, "ptq", INT8_BITS, "per_channel", "symmetric")
MBV1_SYM = kq(MBV1, "ptq", INT8_BITS, "per_channel", "symmetric")
MBV1_ASYM = kq(MBV1, "ptq", INT8_BITS, "per_channel", "asymmetric")
MBV2_QAT = kq(MBV2, "qat", INT8_BITS, "per_channel", "symmetric")
R50_QAT = kq(R50, "qat", INT8_BITS, "per_channel", "symmetric")
R50_W4 = {g: kq(R50, "ptq", INT4_BITS, g) for g in ("per_tensor", "per_channel")}
MBV2_W4 = {g: kq(MBV2, "ptq", INT4_BITS, g) for g in ("per_tensor", "per_channel")}
R50_W4_QAT, MBV2_W4_QAT = kq(R50, "qat", INT4_BITS, "per_channel"), kq(MBV2, "qat", INT4_BITS, "per_channel")
PX2_FP32, PX2_CPU, PX2_DSP = (lit(CR.KrishnamoorthiPixel2Fp32CpuMs), lit(CR.KrishnamoorthiPixel2Int8CpuMs),
                              lit(CR.KrishnamoorthiPixel2Int8DspMs))

print("== Hook (Literature.Series.KrishnamoorthiQuantization, Table 3; Nagel et al. 2019)")
classes = Datasets.ImageNet.num_classes
v("r50_float", R50_FLOAT, "{:.1f}", "ResNet-50 float top-1, %")
v("r50_tensor", R50_TENSOR, "{:.1f}", "ResNet-50 INT8 per-tensor top-1, %")
v("r50_tensor_loss", R50_FLOAT - R50_TENSOR, "{:.1f}", "ResNet-50 INT8 per-tensor loss, points")
v("mbv2_float", MBV2_FLOAT, "{:.1f}", "MobileNetV2 float top-1, %")
v("mbv2_tensor", MBV2_TENSOR, "{:.1f}", "MobileNetV2 INT8 per-tensor top-1, %")
v("chance_pct", 100 / classes, "{:.1f}", "chance on ImageNet, %")
v("imagenet_classes", classes, "{:,.0f}", "ImageNet classes")
v("nagel_float", lit(CR.NagelMobileNetV2FloatTop1Pct), "{:.2f}", "Nagel MobileNetV2 FP32 top-1, %")
v("nagel_tensor", lit(CR.NagelMobileNetV2Int8PerTensorTop1Pct), "{:.2f}", "Nagel MobileNetV2 per-layer INT8 top-1, %")
v("r50_chapter_top1", PROFILE.float_top1_pct, "{:.1f}", "the chapter's ResNet-50 FP32 top-1, %")

print("== Recall (Scenarios.Gpt2XlFp16WeightReadPhone, MobileNetV2BlurPipelinePhone)")
v("gpt2_fp16_gb", m(GPT2_READ.weight_bytes, "GB"), "{:.0f}", "GPT-2 XL FP16 weights, GB")
v("gpt2_read_ms", m(GPT2_READ.read_time, "ms"), "{:.0f}", "read once at the phone bus, ms")
v("phone_bw", m(GPT2_READ.hardware.memory.bandwidth, "GB/s"), "{:.1f}", "phone memory bandwidth, GB/s")
v("blur_cpu_ms", m(BLUR.cpu_frame, "ms"), "{:.0f}", "blur FP32 frame on the CPU, ms")
v("blur_pruned_ms", m(BLUR.pruned_cpu_frame, "ms"), "{:.0f}", "blur pruned frame on the CPU, ms")
v("blur_int8_ms", m(BLUR.pruned_accelerator_frame, "ms"), "{:.0f}", "blur pruned INT8 frame on the accelerator, ms")
v("frame_ms", m(BLUR.frame_budget, "ms"), "{:.0f}", "frame budget at the camera rate, ms")
MNV2_READ, R50_READ = Scenarios.MobileNetV2Fp32WeightReadPhone, Scenarios.ResNet50Fp32WeightReadPhone
v("mnv2_read_ms", m(MNV2_READ.read_time, "ms"), "{:.2f}", "MobileNetV2 FP32 weight read, ms")
v("mnv2_read_share_pct", pct(m(MNV2_READ.read_time / BLUR.frame_budget, "")), "{:.0f}", "of the frame, %")
v("r50_read_ms", m(R50_READ.read_time, "ms"), "{:.0f}", "ResNet-50 FP32 weight read, ms")
v("blur_fp32_mb", m(BLUR.full_bytes, "MB"), "{:.1f}", "blur model in FP32, MB")
v("blur_pruned_mb", m(BLUR.pruned_bytes, "MB"), "{:.1f}", "pruned model in FP32, MB")
v("blur_int8_mb", m(BLUR.pruned_int_bytes, "MB"), "{:.1f}", "pruned model in INT8, MB")

print("== Formats (core.units FLOAT_FORMAT_BITS, FP8_FORMAT_BITS; physics.numerics)")
fp32_e, fp32_f = FLOAT_FORMAT_BITS["fp32"]
bf16_e, _ = FLOAT_FORMAT_BITS["bf16"]
e4m3_e, e4m3_f = FP8_FORMAT_BITS["e4m3"]
e5m2_e, e5m2_f = FP8_FORMAT_BITS["e5m2"]
E4M3 = calc_float_values(e4m3_e, e4m3_f, ieee=False)
E4M3_POS = [x for x in E4M3 if x > 0]
v("fp32_exp_bits", fp32_e, "{:.0f}", "FP32 exponent bits")
v("fp32_frac_bits", fp32_f, "{:.0f}", "FP32 mantissa bits")
v("bf16_exp_bits", bf16_e, "{:.0f}", "BF16 exponent bits")
v("fp16_max", calc_float_range(*FLOAT_FORMAT_BITS["fp16"])[0], "{:,.0f}", "FP16 largest finite value")
v("e4m3_exp_bits", e4m3_e, "{:.0f}", "E4M3 exponent bits")
v("e4m3_frac_bits", e4m3_f, "{:.0f}", "E4M3 mantissa bits")
v("e4m3_max", max(E4M3), "{:.0f}", "E4M3 largest value")
v("e5m2_max", max(calc_float_values(e5m2_e, e5m2_f)), "{:,.0f}", "E5M2 largest value")
v("int8_levels", calc_int_levels(INT8_BITS), "{:.0f}", "INT8 levels")
v("int4_levels", calc_int_levels(INT4_BITS), "{:.0f}", "INT4 levels")
v("int8_smaller", FP32_BITS / INT8_BITS, "{:.0f}", "INT8 smaller than FP32, times")
v("int4_smaller", FP32_BITS / INT4_BITS, "{:.0f}", "INT4 smaller than FP32, times")

print("== A real 4-bit file (Literature.Series.Llama2_7BGgufFiles; CompressionResults llama.cpp blocks)")
LLAMA_PARAMS = lit(CR.LlamaBenchLlama2_7BParamsB) * 10 ** 9
QK = lit(CR.LlamaCppQ4_0BlockWeights)
SCALE_BITS = lit(CR.LlamaCppBlockScaleBits)
Q4_0_BPW = calc_block_bits_per_weight(QK, INT4_BITS, SCALE_BITS)
Q8_0_BPW = calc_block_bits_per_weight(QK, INT8_BITS, SCALE_BITS)


def gguf(quant):
    return next(r for r in GGUF if r["quant"] == quant)


def bits_per_weight(nbytes):
    return m(mlsysim.core.units.Q_(nbytes, "byte"), "bit") / LLAMA_PARAMS


for q in ("Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q6_K", "Q8_0"):
    key = q.lower()
    r = gguf(q)
    v(f"gguf_{key}_bits", r["nominal_bits"], "{:.0f}", f"{q} nominal bits")
    v(f"gguf_{key}_gb", r["bytes"] / 10 ** 9, "{:.2f}", f"{q} file, GB")
    v(f"gguf_{key}_bpw", bits_per_weight(r["bytes"]), "{:.2f}", f"{q} bits per weight")
v("llama_params_b", LLAMA_PARAMS / 10 ** 9, "{:.2f}", "Llama 2 7B weights, billions (llama-bench)")
v("q4_block", QK, "{:.0f}", "weights per Q4_0 block")
v("q4_scale_bytes", m(mlsysim.core.units.Q_(SCALE_BITS, "bit"), "byte"), "{:.0f}", "Q4_0 scale, bytes")
v("q4_scale_bits", SCALE_BITS, "{:.0f}", "Q4_0 scale, bits")
v("q4_code_bytes", m(QK * BYTES_INT4, "byte"), "{:.0f}", "Q4_0 codes per block, bytes")
v("q4_block_bytes", m(mlsysim.core.units.Q_(Q4_0_BPW * QK, "bit"), "byte"), "{:.0f}", "Q4_0 block, bytes")
v("q4_bpw", Q4_0_BPW, "{:.1f}", "Q4_0 block bits per weight")
v("q4_overhead_pct", pct((Q4_0_BPW - INT4_BITS) / INT4_BITS), "{:.1f}", "scale overhead over 4 bits, %")
v("q2k_block_bpw", lit(CR.GgufQ2KBlockBitsPerWeight), "{:.4f}", "Q2_K block bits per weight (model card)")

print("== FP8 grid against INT8 grid (physics.numerics.calc_float_values)")
int8_pos = calc_int_max(INT8_BITS)
INT8_448 = Scenarios.SymmetricInt8UnitRange.variant(alpha=-max(E4M3), beta=max(E4M3))  # INT8 asked to reach 448
int8_step = INT8_448.scale
top_binade = calc_int_levels(math.floor(math.log2(max(E4M3))))
v("e4m3_pos", len(E4M3_POS), "{:.0f}", "E4M3 positive values")
v("e4m3_below1", sum(1 for x in E4M3_POS if x < 1), "{:.0f}", "of them below 1")
v("e4m3_top", sum(1 for x in E4M3_POS if x >= top_binade), "{:.0f}", "of them in the top binade")
v("e4m3_top_binade", top_binade, "{:.0f}", "the top binade starts at")
v("e4m3_min", min(E4M3_POS), "{:.3f}", "E4M3 smallest positive value")
v("e4m3_min_exp", math.log2(min(E4M3_POS)), "{:.0f}", "its power of two")
v("e4m3_decades_word", math.floor(math.log10(max(E4M3) / min(E4M3_POS))), "words", "decades spanned, in prose")
v("int8_pos", int8_pos, "{:.0f}", "INT8 positive levels")
v("int8_step448", int8_step, "{:.2f}", "INT8 step to cover 448")
v("int8_zero_below", INT8_448.max_rounding_error, "{:.2f}", "everything below half a step rounds to 0")

print("== Energy (Scenarios.Int8VsFp32OperationEnergy45nm; Models.Tiny.DS_CNN; Platforms.Tiny)")
dscnn = Models.Tiny.DS_CNN
v("e_add_fp32", m(ENERGY.wide_add, "pJ/flop"), "{:.2f}", "FP32 add, pJ")
v("e_add_int8", m(ENERGY.narrow_add, "pJ/flop"), "{:.2f}", "INT8 add, pJ")
v("e_add_ratio", ENERGY.add_ratio, "{:.0f}", "add FP32/INT8")
v("e_mul_ratio", ENERGY.mul_ratio, "{:.1f}", "multiply FP32/INT8")
v("e_mac_ratio", ENERGY.mac_ratio, "{:.1f}", "MAC FP32/INT8")
v("e_dram", m(ENERGY.memory_read, "pJ"), "{:.0f}", "32-bit DRAM read, pJ")
v("e_dram_mac", round(ENERGY.read_over_narrow_mac, -2), "{:,.0f}", "DRAM read over INT8 MAC (to the hundred, as the chapter)")
v("spotter_fp32_kb", m(dscnn.size_in_bytes(BYTES_FP32), "KB"), "{:.0f}", "keyword spotter FP32, KB")
v("spotter_int8_kb", m(dscnn.size_in_bytes(BYTES_INT8), "KB"), "{:.0f}", "keyword spotter INT8, KB")
v("mcu_sram_kb", m(Platforms.Tiny.ram, "KiB"), "{:.0f}", "microcontroller SRAM, KB (as labeled)")

print("== The 3-bit toy grid (Scenarios.ThreeBitGridMinusOneToTwoAndHalf)")
v("toy_alpha", TOY.lo, "{:.0f}", "alpha")
v("toy_neg_alpha", -TOY.lo, "{:.0f}", "minus alpha")
v("toy_beta", TOY.hi, "{:.1f}", "beta")
v("toy_bits", TOY.bits, "{:.0f}", "bits")
v("toy_bits_word", TOY.bits, "words", "bits, in prose")
v("toy_outside_word", sum(1 for x in TOY.values if not TOY.lo <= x <= TOY.hi), "words", "values outside the range, in prose")
v("toy_levels", calc_int_levels(TOY.bits), "{:.0f}", "codes")
v("toy_steps", TOY.steps, "{:.0f}", "steps (also the top code)")
v("toy_range", TOY.hi - TOY.lo, "{:.1f}", "range")
v("toy_s", TOY.scale, "{:.1f}", "scale")
v("toy_z", TOY.zero_point, "{:.0f}", "zero-point")
v("toy_half", TOY.max_rounding_error, "{:.2f}", "half a step")
for i, (x, q, xr) in enumerate(zip(TOY.values, TOY.codes, TOY.readback)):
    v(f"toy_x{i}", x, "{:.1f}", f"value {i}")
    v(f"toy_q{i}", q, "{:.0f}", f"its code")
    v(f"toy_r{i}", xr, "{:.1f}" if xr != TOY.lo else "{:.0f}", f"it reads back as")
v("toy_err5", TOY.errors[-1], "{:.1f}", "error of the last value")

print("== Napkin Math 10.1 (Scenarios.Uint8RangeMinusOneToThree)")
v("nm_alpha", NAPKIN.lo, "{:.0f}", "alpha")
v("nm_neg_alpha", -NAPKIN.lo, "{:.0f}", "minus alpha")
v("nm_beta", NAPKIN.hi, "{:.0f}", "beta")
v("nm_bits", NAPKIN.bits, "{:.0f}", "bits")
v("nm_range", NAPKIN.hi - NAPKIN.lo, "{:.0f}", "range")
v("nm_steps", NAPKIN.steps, "{:.0f}", "steps (also the top code)")
v("nm_s", NAPKIN.scale, "{:.4f}", "scale")
v("nm_z_exact", NAPKIN.zero_point_exact, "{:.2f}", "minus alpha over s")
v("nm_z", NAPKIN.zero_point, "{:.0f}", "zero-point")
v("nm_nudge", NAPKIN.zero_point - NAPKIN.zero_point_exact, "{:.2f}", "the rounding nudge, in steps")
for i, (x, q) in enumerate(zip(NAPKIN.values, NAPKIN.codes)):
    v(f"nm_x{i}", x, "{:.0f}", f"matrix entry {i}")
    v(f"nm_q{i}", q, "{:.0f}", "its code")
v("nm_r0", NAPKIN.readback[0], "{:.3f}", "alpha reads back as")
v("nm_r0_torch", NAPKIN.readback[0], "{:.4f}", "alpha reads back as, as PyTorch prints it")

print("== Listing 2 (Scenarios.SymmetricInt8FourWeights)")
v("l2_max", max(abs(x) for x in LISTING.values), "{:.3f}", "largest magnitude")
v("l2_int_max", LISTING.qmax, "{:.0f}", "INT8 largest code")
v("l2_s", LISTING.scale, "{:.5f}", "scale")
v("l2_half", LISTING.max_rounding_error, "{:.5f}", "half a step")
l2_max = max(abs(x) for x in LISTING.values)
v("l2_outlier", l2_max + l2_max, "{:.3f}", "an outlier at twice the range")
for i, (x, q, e) in enumerate(zip(LISTING.values, LISTING.codes, LISTING.errors)):
    v(f"l2_w{i}", x, "{:.3f}", f"weight {i}")
    v(f"l2_q{i}", q, "{:.0f}", "its code")
    v(f"l2_e{i}", e, "{:.5f}" if e > LISTING.scale / 100 else "{:.0f}", "its error")

print("== Zero-point (Scenarios.SymmetricInt8UnitRange, AsymmetricInt8..., Relu6*; Krishnamoorthi Table 3)")
v("zp_sym_lo", SYM_UNIT.lo, "{:.0f}", "symmetric range low")
v("zp_sym_hi", SYM_UNIT.hi, "{:.0f}", "symmetric range high")
v("zp_sym_qmin", SYM_UNIT.qmin, "{:.0f}", "symmetric lowest code")
v("zp_sym_qmax", SYM_UNIT.qmax, "{:.0f}", "symmetric highest code")
v("zp_asym_lo", ASYM.lo, "{:.1f}", "asymmetric range low")
v("zp_asym_hi", ASYM.hi, "{:.1f}", "asymmetric range high")
v("zp_asym_qmin", ASYM.qmin, "{:.0f}", "asymmetric lowest code")
v("zp_asym_qmax", ASYM.qmax, "{:.0f}", "asymmetric highest code")
v("zp_asym_range", ASYM.hi - ASYM.lo, "{:.0f}", "asymmetric range width")
v("zp_asym_steps", ASYM.steps, "{:.0f}", "asymmetric steps")
v("zp_asym_z", ASYM.zero_point, "{:.0f}", "asymmetric zero-point")
v("int8_shift", -ASYM.qmin, "{:.0f}", "signed INT8 codes sit this far below unsigned ones")
v("relu_hi", RELU_SYM.hi, "{:.0f}", "ReLU6 top")
v("relu_sym_div", RELU_SYM.qmax, "{:.0f}", "symmetric divisor")
v("relu_asym_div", RELU_ASYM.steps, "{:.0f}", "asymmetric divisor")
v("relu_sym_s", RELU_SYM.scale, "{:.3f}", "symmetric step")
v("relu_asym_s", RELU_ASYM.scale, "{:.3f}", "asymmetric step")
v("mbv1_float", MBV1_FLOAT, "{:.1f}", "MobileNet-v1 float, %")
v("mbv1_sym", MBV1_SYM, "{:.1f}", "MobileNet-v1 symmetric per-channel, %")
v("mbv1_asym", MBV1_ASYM, "{:.1f}", "MobileNet-v1 asymmetric per-channel, %")
v("mbv1_sym_cost", MBV1_ASYM - MBV1_SYM, "{:.0f}", "symmetric cost, points")

print("== One INT8 output (Scenarios.Int8DotProductThreeTerms)")
v("il_w_max", max(abs(x) for x in DOT.weights), "{:.2f}", "largest weight")
v("il_x_max", max(abs(x) for x in DOT.inputs), "{:.0f}", "largest input")
v("il_int_max", calc_int_max(DOT.bits), "{:.0f}", "INT8 largest code")
for i, (w, x, wq, xq, p) in enumerate(zip(DOT.weights, DOT.inputs, DOT.weight_codes, DOT.input_codes, DOT.products)):
    v(f"il_w{i}", w, "{:.2f}" if i else "{:.1f}", f"weight {i}")
    v(f"il_x{i}", x, "{:.1f}", f"input {i}")
    v(f"il_wq{i}", wq, "{:.0f}", "weight code")
    v(f"il_xq{i}", xq, "{:.0f}", "input code")
    v(f"il_p{i}", p, "{:,.0f}", "product")
v("il_w0_ratio", DOT.weights[0] / DOT.weight_scale, "{:.1f}", "first weight over its scale")
v("il_x0_ratio", DOT.inputs[0] / DOT.input_scale, "{:.1f}", "first input over its scale")
v("il_sum", DOT.accumulator, "{:,.0f}", "INT32 sum")
v("il_readback", DOT.readback, "{:.3f}", "sum times both scales")
v("il_exact", DOT.exact, "{:.3f}", "exact")
v("il_m", DOT.requant_scale, "{:.4f}", "requantization scale M")
v("il_y", DOT.output_code, "{:.0f}", "output code")
v("il_sy_div", round(1 / DOT.output_scale), "{:.0f}", "output scale is one over this")
v("il_headroom", round(DOT.accumulator_headroom, -3), "{:,.0f}", "worst-case products in INT32, about")
v("il_acc_exp", DOT.accumulator_bits - 1, "{:.0f}", "accumulator power of two")

print("== Clipping (Scenarios.ActivationsCoveredToTwoAndHalfUint8, ActivationsClippedAtOnePointOneUint8)")
v("clip_full", CLIP_FULL.hi, "{:.1f}", "full range")
v("clip_cut", CLIP_CUT.hi, "{:.1f}", "clip")
v("clip_steps", CLIP_FULL.steps, "{:.0f}", "steps (also the top code)")
v("clip_full_s", CLIP_FULL.scale, "{:.4f}", "step, full range")
v("clip_cut_s", CLIP_CUT.scale, "{:.4f}", "step, clipped")
v("clip_finer", CLIP_FULL.scale / CLIP_CUT.scale, "{:.1f}", "finer, times")

print("== Measured calibration (Literature.Series.WuInt8Calibration)")
WU_COLS = ("fp32", "max", "entropy", "pct_99_99")
WU_ONE_DECIMAL = {("EfficientNet b0", "max")}  # printed to one decimal in the paper
for r in WU:
    key = r["model"].split()[0].lower()
    for col in WU_COLS:
        fmt = "{:.1f}" if (r["model"], col) in WU_ONE_DECIMAL else "{:.2f}"
        v(f"wu_{key}_{col}", r[col], fmt, f"{r['model']} {col}")
bert = next(r for r in WU if r["model"].startswith("BERT"))
resnet = next(r for r in WU if r["model"].startswith("ResNet"))
v("wu_bert_best", bert["pct_99_999"], "{:.2f}", "BERT Large at the 99.999th percentile")

print("== Per-channel scales (Krishnamoorthi Tables 3, 4; Nagel et al. 2019)")
nagel_wide, nagel_narrow = lit(CR.NagelExampleWideChannelMax), lit(CR.NagelExampleNarrowChannelMax)
v("mbv2_channel", MBV2_CHANNEL, "{:.1f}", "MobileNetV2 per-channel PTQ, %")
v("mbv2_qat", MBV2_QAT, "{:.1f}", "MobileNetV2 per-channel QAT, %")
v("mbv2_qat_gain", MBV2_QAT - MBV2_CHANNEL, "{:.1f}", "QAT gain, points")
v("r50_channel", R50_CHANNEL, "{:.1f}", "ResNet-50 per-channel PTQ, %")
v("r50_qat", R50_QAT, "{:.1f}", "ResNet-50 QAT, %")
v("nagel_wide", nagel_wide, "{:.0f}", "wide channel half-width")
v("nagel_narrow", nagel_narrow, "{:.1f}", "narrow channel half-width")
v("nagel_step", calc_affine_scale(-nagel_wide, nagel_wide, INT8_BITS), "{:.0f}", "one 8-bit step over the wide channel")
v("nagel_channels", lit(CR.NagelFirstDepthwiseChannels), "{:.0f}", "channels in the figure")
v("awq_group", lit(CR.LinAwqGroupSize), "{:.0f}", "AWQ group size")

print("== Outliers (Literature.Series.DettmersInt8Perplexity; CompressionResults Dettmers)")


def det(params_b):
    return next(r for r in DET if r["params_b"] == params_b)


sizes = [r["params_b"] for r in DET]
v("det_small_m", sizes[0] * 10 ** 3, "{:.0f}", "smallest model, millions")
v("det_mid_b", sizes[1], "{:.1f}", "second model, billions")
v("det_large_b", sizes[-1], "{:.0f}", "largest model, billions")
v("det_fourth_b", sizes[:-1][-1], "{:.1f}", "second-largest model, billions")
v("det_absmax_small", DET[0]["absmax"], "{:.2f}", "125M under absmax")
v("det_absmax_large", DET[-1]["absmax"], "{:.2f}", "13B under absmax")
v("det_fp32_large", DET[-1]["fp32"], "{:.2f}", "13B in float")
v("out_b", lit(CR.DettmersOutlierEmergenceParamsB), "{:.1f}", "outliers in every layer from, billions")
v("out_floor_b", math.floor(lit(CR.DettmersOutlierEmergenceParamsB)), "{:.0f}", "below about this many billions they come and go")
v("out_tokens_pct", lit(CR.DettmersOutlierTokenSharePct), "{:.0f}", "positions affected, %")
v("out_mult", lit(CR.DettmersOutlierMagnitudeMult), "{:.0f}", "outlier magnitude, times the rest")
v("out_dims", lit(CR.DettmersOutlierFeatureDims), "{:.0f}", "hidden dimensions holding them")
v("out_threshold", lit(CR.DettmersOutlierThreshold), "{:.0f}", "outlier magnitude threshold")

print("== The cliff (Krishnamoorthi Tables 3 and 5)")
v("cliff_8bit_gap", max(MBV2_FLOAT - MBV2_CHANNEL, R50_FLOAT - R50_CHANNEL), "{:.0f}", "8-bit per-channel within, points")
v("int4_bits", INT4_BITS, "{:.0f}", "INT4 bits")
v("int8_bits", INT8_BITS, "{:.0f}", "INT8 bits")
v("r50_w4_tensor", R50_W4["per_tensor"], "{:.1f}", "ResNet-50 4-bit per-tensor PTQ, %")
v("mbv2_w4_tensor", MBV2_W4["per_tensor"], "{:.1f}", "MobileNetV2 4-bit per-tensor PTQ, %")
v("r50_w4_channel", R50_W4["per_channel"], "{:.1f}", "ResNet-50 4-bit per-channel PTQ, %")
v("mbv2_w4_channel", MBV2_W4["per_channel"], "{:.1f}", "MobileNetV2 4-bit per-channel PTQ, %")
v("r50_w4_qat", R50_W4_QAT, "{:.1f}", "ResNet-50 4-bit QAT, %")
v("mbv2_w4_qat", MBV2_W4_QAT, "{:.1f}", "MobileNetV2 4-bit QAT, %")
v("r50_w4_qat_gap", R50_FLOAT - R50_W4_QAT, "{:.0f}", "ResNet-50 4-bit QAT gap, points")
v("mbv2_w4_qat_gap", MBV2_FLOAT - MBV2_W4_QAT, "{:.0f}", "MobileNetV2 4-bit QAT gap, points")

print("== GPTQ and AWQ (Literature.CompressionResults: Frantar et al. 2023; Lin et al. 2024)")
v("gptq_fp16", lit(CR.FrantarOpt175BFp16Ppl), "{:.2f}", "OPT-175B FP16 perplexity")
v("gptq_rtn4", lit(CR.FrantarOpt175BRtn4BitPpl), "{:.2f}", "4-bit RTN")
v("gptq_4", lit(CR.FrantarOpt175BGptq4BitPpl), "{:.2f}", "4-bit GPTQ")
v("gptq_rtn3_k", lit(CR.FrantarOpt175BRtn3BitPpl) / 10 ** 3, "{:.1f}", "3-bit RTN, thousands")
v("gptq_3", lit(CR.FrantarOpt175BGptq3BitPpl), "{:.2f}", "3-bit GPTQ")
v("awq_fp16", lit(CR.LinOpt6_7BFp16Ppl), "{:.2f}", "OPT-6.7B FP16 perplexity")
v("awq_rtn", lit(CR.LinOpt6_7BRtnInt3Ppl), "{:.2f}", "INT3 RTN")
v("awq_by_weight", lit(CR.LinOpt6_7BKeepByWeightPpl), "{:.2f}", "keep 1% by weight")
v("awq_by_act", lit(CR.LinOpt6_7BKeepByActivationPpl), "{:.2f}", "keep 1% by activation")
v("awq_kept_pct", lit(CR.LinAwqKeptPct), "{:.0f}", "share kept in FP16, %")

print("== Roofline (Scenarios.Gpt2Xl*DecodeFullContextPhone; Hardware.Cloud.H100; Frantar Table 6)")
rates = H100.compute.precision_flops
v("dec_w16_gb", m(DEC16.weight_bytes, "GB"), "{:.0f}", "FP16 weights, GB")
v("dec_w4_gb", m(DEC4.weight_bytes, "GB"), "{:.2f}", "INT4 weights, GB")
v("dec_kv_gb", m(DEC16.kv_cache_bytes, "GB"), "{:.2f}", "KV cache, GB")
v("dec_t16_gb", m(DEC16.token_bytes, "GB"), "{:.2f}", "FP16 bytes per token, GB")
v("dec_t4_gb", m(DEC4.token_bytes, "GB"), "{:.2f}", "INT4 bytes per token, GB")
v("dec_t16_ms", m(DEC16.token_time, "ms"), "{:.0f}", "FP16 ms per token")
v("dec_t4_ms", m(DEC4.token_time, "ms"), "{:.0f}", "INT4 ms per token")
v("dec_speedup", (DEC16.token_time / DEC4.token_time).to("").magnitude, "{:.1f}", "speedup, times")
v("dec_byte_ratio", (BYTES_FP16 / BYTES_INT4).to("").magnitude, "{:.0f}", "FP16 over INT4 bytes per weight")
v("h100_int8", m(rates["int8"], "TOPS"), "{:,.0f}", "H100 INT8 dense, tera-ops/s")
v("h100_fp16", m(rates["fp16_dense"], "TFLOPs/s"), "{:,.0f}", "H100 FP16 dense, tera-ops/s")
v("h100_int8_x", m(rates["int8"], "TOPS") / m(rates["fp16_dense"], "TFLOPs/s"), "{:.0f}", "INT8 over FP16, times")
v("h100_int8_x_word", m(rates["int8"], "TOPS") / m(rates["fp16_dense"], "TFLOPs/s"), "words", "the same, in prose")
gptq_x = lit(CR.FrantarOpt175BFp16TokenMs) / lit(CR.FrantarOpt175B3BitTokenMs)
v("gptq_fp16_ms", lit(CR.FrantarOpt175BFp16TokenMs), "{:.0f}", "OPT-175B FP16, ms per token")
v("gptq_3bit_ms", lit(CR.FrantarOpt175B3BitTokenMs), "{:.0f}", "OPT-175B 3-bit, ms per token")
v("gptq_fp16_gpus_word", lit(CR.FrantarOpt175BFp16Gpus), "words", "A100s for FP16, in prose")
v("gptq_3bit_gpus_word", lit(CR.FrantarOpt175B3BitGpus), "words", "A100s for 3-bit, in prose")
v("gptq_x", gptq_x, "{:.2f}", "measured speedup, times")

print("== llama-bench (Literature.Series.LlamaBenchAppleSilicon)")


def bench(chip, quant):
    return next(r for r in BENCH if r["chip"] == chip and r["quant"] == quant)


ultra16, ultra8, ultra4 = (bench("M2 Ultra", q) for q in ("F16", "Q8_0", "Q4_0"))
v("lb_ultra_bw", ultra16["bandwidth_gbs"], "{:.0f}", "M2 Ultra bandwidth, GB/s")
v("lb_tg4_x", ultra4["tg128_tps"] / ultra16["tg128_tps"], "{:.2f}", "generation Q4_0 over F16")
v("lb_tg8_x", ultra8["tg128_tps"] / ultra16["tg128_tps"], "{:.2f}", "generation Q8_0 over F16")
v("lb_pp4_x", ultra4["pp512_tps"] / ultra16["pp512_tps"], "{:.2f}", "prompt Q4_0 over F16")
v("lb_file_x", ultra16["size_gib"] / ultra4["size_gib"], "{:.2f}", "F16 file over Q4_0 file")
for q, r in (("16", ultra16), ("8", ultra8), ("4", ultra4)):
    v(f"lb_tg{q}", r["tg128_tps"], "{:.0f}", f"M2 Ultra generation (tg 128), {r['quant']}, tokens/s")
    v(f"lb_pp{q}", r["pp512_tps"], "{:,.0f}", f"M2 Ultra prompt (pp 512), {r['quant']}, tokens/s")
others = [c for c in dict.fromkeys(r["chip"] for r in BENCH) if c != "M2 Ultra"]
tg = [bench(c, "Q4_0")["tg128_tps"] / bench(c, "F16")["tg128_tps"] for c in others]
pp = [bench(c, "Q4_0")["pp512_tps"] / bench(c, "F16")["pp512_tps"] for c in others]
v("lb_other_tg_lo", min(tg), "{:.2f}", "other chips, generation, low")
v("lb_other_tg_hi", max(tg), "{:.2f}", "other chips, generation, high")
v("lb_other_pp_lo", min(pp), "{:.2f}", "other chips, prompt, low")
v("lb_other_pp_hi", max(pp), "{:.2f}", "other chips, prompt, high")
GIB = mlsysim.core.units.Q_(1, "GiB")
v("lb_stream16", ultra16["tg128_tps"] * m(ultra16["size_gib"] * GIB, "GB"), "{:.0f}", "F16 weights streamed, GB/s")
v("lb_stream4", ultra4["tg128_tps"] * m(ultra4["size_gib"] * GIB, "GB"), "{:.0f}", "Q4_0 weights streamed, GB/s")

print("== Separate dequantize pass (Scenarios.Gpt2XlInt4DecodeFullContextPhone)")
v("dq_sep_gb", m(DEC4.separate_unpack_bytes, "GB"), "{:.2f}", "separate pass, GB per token")
v("dq_sep_ms", m(DEC4.separate_unpack_time, "ms"), "{:.0f}", "separate pass, ms per token")
v("dq_sep_x", (DEC4.separate_unpack_time / DEC16.token_time).to("").magnitude, "{:.2f}", "against FP16, times")

print("== Pixel 2 (Literature.CompressionResults: Krishnamoorthi Table 7; core.units SIMD)")
v("px_fp32", PX2_FP32, "{:.0f}", "FP32 on the CPU, ms")
v("px_cpu", PX2_CPU, "{:.0f}", "INT8 on the CPU, ms")
v("px_dsp", PX2_DSP, "{:.1f}", "INT8 on the DSP, ms")
v("px_cpu_x", PX2_FP32 / PX2_CPU, "{:.1f}", "CPU INT8 speedup")
v("px_dsp_x", PX2_FP32 / PX2_DSP, "{:.1f}", "DSP INT8 speedup")
v("simd_bits", SIMD_REGISTER_BITS, "{:.0f}", "register width, bits")
v("simd_fp32", SIMD_REGISTER_BITS // FP32_BITS, "{:.0f}", "FP32 lanes")
v("simd_int8", SIMD_REGISTER_BITS // INT8_BITS, "{:.0f}", "INT8 lanes")

print("== The INT8 profile (Scenarios.ResNet50Int8ProfileH100)")
v("pf_fp32_top1", PROFILE.float_top1_pct, "{:.1f}", "FP32 top-1, %")
v("pf_int8_top1", PROFILE.int_top1_pct, "{:.1f}", "INT8 top-1, %")
v("pf_top1_limit", PROFILE.top1_limit_pct, "{:.1f}", "top-1 limit, %")
v("pf_top1_drop", PROFILE.top1_drop, "{:.1f}", "top-1 drop, points")
v("pf_fp32_cal", PROFILE.float_calibration_error_pct, "{:.1f}", "FP32 calibration error, %")
v("pf_int8_cal", PROFILE.int_calibration_error_pct, "{:.1f}", "INT8 calibration error, %")
v("pf_cal_limit", PROFILE.calibration_error_limit_pct, "{:.1f}", "calibration error limit, %")
v("pf_latency_x", PROFILE.latency_speedup, "{:.1f}", "batch-one speedup")
v("pf_throughput_x", PROFILE.throughput_gain, "{:.1f}", "large-batch throughput gain")
v("pf_energy_x", PROFILE.energy_improvement, "{:.1f}", "energy per image gain")
v("pf_tdp_w", m(PROFILE.hardware.tdp, "W"), "{:.0f}", "H100 board power, W")

print("== Synthesis (Models.Language.Llama2_7B; Platforms.Mobile)")
llama = Models.Language.Llama2_7B
v("p7b_b", m(llama.parameters, "param") / 10 ** 9, "{:.0f}", "7B model parameters, billions")
v("p7b_fp16_gb", m(llama.size_in_bytes(BYTES_FP16), "GB"), "{:.0f}", "7B in FP16, GB")
v("p7b_int4_gb", m(llama.size_in_bytes(BYTES_INT4), "GB"), "{:.1f}", "7B in 4-bit codes only, GB")
v("p7b_q4_gb", m(llama.parameters, "param") * Q4_0_BPW / m(mlsysim.core.units.Q_(1, "GB"), "bit"), "{:.1f}",
  "7B at Q4_0's bits per weight, GB")
v("int4_bytes", m(BYTES_INT4, "byte"), "{:.1f}", "bytes per INT4 weight")
v("phone_ram_gb", m(Platforms.Mobile.ram, "GiB"), "{:.0f}", "phone RAM, GB (as labeled)")

if __name__ == "__main__":
    write(__file__)
