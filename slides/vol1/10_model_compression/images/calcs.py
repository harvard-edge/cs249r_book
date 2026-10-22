#!/usr/bin/env python3
"""Every number on the Lecture 10 (Model Compression, Part 1) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Models, Hardware, Platforms, the compression
Scenarios.*, Literature.CompressionResults, and the literature series behind
the measured plots) and the calculations; the Model Compression chapter reads
the same scenarios. This file only selects values, converts units, and formats
them; it types no numbers. Each value is recorded with v() under the key the
deck prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports this module, so figures and slides share one computation.

One input is not in MLSysIM: cpu_measure.json, the timing run made for this
deck by measure_cpu.py (MobileNetV2 on one laptop CPU thread).

Run from this folder: python3 calcs.py
"""

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

from mlsysim.core.units import BYTES_FP16, BYTES_FP32, BYTES_INT4, BYTES_INT8, BYTES_INT16, BYTES_INT32  # noqa: E402
from mlsysim.physics import (  # noqa: E402
    calc_pareto_frontier, calc_sparse_breakeven_zero_share, calc_sparse_bytes_per_nonzero,
    calc_transformer_block_weights)

HERE = Path(__file__).parent
Models, Hardware, Platforms = mlsysim.Models, mlsysim.Hardware, mlsysim.Platforms
Scenarios, Literature = mlsysim.Scenarios, mlsysim.Literature
CR, SERIES = Literature.CompressionResults, Literature.Series

TOKEN = Scenarios.Llama2_7BTokenOnFlagshipPhone
MNV2_READ, R50_READ = Scenarios.MobileNetV2Fp32WeightReadPhone, Scenarios.ResNet50Fp32WeightReadPhone
GPT2_READ, GPT2_INT4_READ = Scenarios.Gpt2XlFp16WeightReadPhone, Scenarios.Gpt2XlInt4WeightReadPhone
BLUR = Scenarios.MobileNetV2BlurPipelinePhone
SMALLEST = Scenarios.SmallestModelAtSixtyPercentMmlu
PRUNE3 = Scenarios.MagnitudePruned3x3
SPMV = Scenarios.SparseMatVec4x4
WEARABLE = Scenarios.WearableMobileNetV2SparseFlash
TILE250, TILE248 = Scenarios.Channels256To250Tile8, Scenarios.Channels256To248Tile8
LOWRANK = Scenarios.Rank128Factorization4096
SOFT = Scenarios.TeacherSoftTargetsT4
FUSION = Scenarios.ResNet50Conv256FusionH100
PAPER = Scenarios.PrunedInt8PaperVsMeasured
TRACE = Scenarios.IrregularSparsityLatencyTrace
BERT = Scenarios.BertBaseMobileCompression
AMDAHL20 = Scenarios.ModelFifthOfRequestLatency.result
AMDAHL60 = Scenarios.ModelThreeFifthsFourTimesFaster


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


def pct(x):
    """A fraction as percent."""
    return 100 * x


def fewer(pruned_pct):
    """Times fewer weights after pruning ``pruned_pct`` percent of them."""
    return 1 / (1 - pruned_pct / 100)


print("== Hook: a 7B model on a phone (Scenarios.Llama2_7BTokenOnFlagshipPhone)")
gpt3, tiny_ram, phone_ram = Models.Language.GPT3, Platforms.Tiny.ram, Platforms.Mobile.ram
dscnn = Models.Tiny.DS_CNN
mem_175b = gpt3.size_in_bytes(BYTES_FP16)
mem_7b = TOKEN.weight_bytes
mem_7b_int4 = TOKEN.workload.size_in_bytes(BYTES_INT4)
spotter_fp32, spotter_int8 = dscnn.size_in_bytes(BYTES_FP32), dscnn.size_in_bytes(BYTES_INT8)
tdp_low, tdp_high = (float(x) for x in Platforms.Mobile.tdp_range_w.split("-"))
v("p175b_b", m(gpt3.parameters, "param") / 10 ** 9, "{:.0f}", "frontier model parameters, billions")
v("p7b_b", m(TOKEN.workload.parameters, "param") / 10 ** 9, "{:.0f}", "7B model parameters, billions")
v("mem175b_gb", m(mem_175b, "GB"), "{:.0f}", "175B in FP16, GB")
v("mem7b_gb", m(mem_7b, "GB"), "{:.0f}", "7B in FP16, GB")
v("mem7b_int4_gb", m(mem_7b_int4, "GB"), "{:.1f}", "7B in INT4, GB")
v("fp16_bytes_word", m(BYTES_FP16, "byte"), "words", "bytes per FP16 weight, in prose")
v("phone_ram_gb", m(phone_ram, "GiB"), "{:.0f}", "phone RAM, GB (as labeled)")
v("mcu_sram_kb", m(tiny_ram, "KiB"), "{:.0f}", "microcontroller SRAM, KB (as labeled)")
v("ladder_orders_word", round(math.log10(m(mem_175b, "byte") / m(tiny_ram, "byte"))), "words",
  "orders of magnitude from 175B FP16 to MCU SRAM, in prose")
v("phone_bw", m(TOKEN.hardware.memory.bandwidth, "GB/s"), "{:.1f}", "flagship phone LPDDR bus, GB/s")
v("token_ms", round(m(TOKEN.read_time, "ms"), -1), "{:.0f}", "7B token weight read, ms (to the ten, as the chapter)")
v("token_j", round(m(TOKEN.read_energy, "J")), "{:.0f}", "7B DRAM energy per token, J (as the chapter rounds)")
v("token_w", round(m(TOKEN.read_power, "W")), "{:.0f}", "7B DRAM power while decoding, W (as the chapter rounds)")
v("phone_tdp_lo", tdp_low, "{:.0f}", "phone sustained power, low, W")
v("phone_tdp_hi", tdp_high, "{:.0f}", "phone sustained power, high, W")
v("spotter_fp32_kb", m(spotter_fp32, "KB"), "{:.0f}", "DS-CNN keyword spotter in FP32, KB")
v("spotter_int8_kb", m(spotter_int8, "KB"), "{:.0f}", "DS-CNN keyword spotter in INT8, KB")

print("== Smallest model at 60 percent MMLU (Scenarios.SmallestModelAtSixtyPercentMmlu)")
v("mmlu_bar_pct", pct(SMALLEST.threshold), "{:.0f}", "MMLU bar, %")
v("gopher_b", SMALLEST.first["parameters"] / 10 ** 9, "{:.0f}", "Gopher parameters, billions")
v("gopher_year", dt.date.fromisoformat(SMALLEST.first["date"]).year, "{:.0f}", "Gopher year")
v("qwen_b", SMALLEST.last["parameters"] / 10 ** 9, "{:.0f}", "Qwen2.5-Coder parameters, billions")
v("qwen_year", dt.date.fromisoformat(SMALLEST.last["date"]).year, "{:.0f}", "Qwen2.5-Coder year")
v("smallest_shrink", round(SMALLEST.shrink), "{:.0f}", "shrink, times")
v("smallest_years_word", math.ceil(SMALLEST.span_years), "words", "years the shrink took, at most, in prose")

print("== The running case (Scenarios.MobileNetV2BlurPipelinePhone)")
mnv2 = BLUR.full
v("cpu_fps", m(BLUR.cpu_rate, "1/s"), "{:.0f}", "FP32 frame rate on the CPU, FPS")
v("cam_fps", m(BLUR.frame_rate, "1/s"), "{:.0f}", "camera frame rate, FPS")
v("cpu_frame_ms", m(BLUR.cpu_frame, "ms"), "{:.0f}", "FP32 frame time, ms")
v("frame_budget_ms", m(BLUR.frame_budget, "ms"), "{:.0f}", "frame budget, ms")
v("mnv2_params_m", m(mnv2.parameters, "param") / 10 ** 6, "{:.1f}", "MobileNetV2 parameters, millions")
v("mnv2_fp32_mb", m(BLUR.full_bytes, "MB"), "{:.0f}", "MobileNetV2 in FP32, MB")
v("mnv2_mflop", m(mnv2.inference_flops, "MFLOP"), "{:.0f}", "MobileNetV2 per frame, MFLOP")

print("== Reading the weights once on the phone (Scenarios.*WeightReadPhone)")
v("mnv2_read_ms", m(MNV2_READ.read_time, "ms"), "{:.2f}", "MobileNetV2 FP32 weight read, ms")
v("mnv2_read_share_pct", pct(m(MNV2_READ.read_time / BLUR.frame_budget, "")), "{:.0f}", "of the frame, %")
v("r50_fp32_mb", m(R50_READ.weight_bytes, "MB"), "{:.1f}", "ResNet-50 in FP32, MB")
v("r50_read_ms", m(R50_READ.read_time, "ms"), "{:.0f}", "ResNet-50 FP32 weight read, ms")
v("gpt2_fp16_gb", m(GPT2_READ.weight_bytes, "GB"), "{:.0f}", "GPT-2 XL in FP16, GB")
v("gpt2_read_ms", m(GPT2_READ.read_time, "ms"), "{:.0f}", "GPT-2 XL FP16 weight read per token, ms")
v("gpt2_int4_read_ms", m(GPT2_INT4_READ.read_time, "ms"), "{:.0f}", "GPT-2 XL INT4 weight read per token, ms")

print("== The accuracy-efficiency frontier (Literature.Series.MobileNetV3Table3)")
MNV3 = SERIES.MobileNetV3Table3.entries
FRONTIER = [MNV3[i]["model"] for i in calc_pareto_frontier([(r["pixel1_ms"], r["top1"]) for r in MNV3])]
v2 = next(r for r in MNV3 if r["model"] == "V2 1.0")
v3l75 = next(r for r in MNV3 if r["model"] == "V3-Large 0.75")
v("v2_top1", v2["top1"], "{:.1f}", "MobileNetV2 1.0 top-1, %")
v("v2_ms", v2["pixel1_ms"], "{:.0f}", "MobileNetV2 1.0 on a Pixel 1, ms")
v("v3l75_top1", v3l75["top1"], "{:.1f}", "V3-Large 0.75 top-1, %")
v("v3l75_ms", v3l75["pixel1_ms"], "{:.0f}", "V3-Large 0.75 on a Pixel 1, ms")
v("v3_speedup", v2["pixel1_ms"] / v3l75["pixel1_ms"], "{:.2f}", "V3-Large 0.75 faster than V2 1.0, times")
v("v3_gain_pts", v3l75["top1"] - v2["top1"], "{:.1f}", "V3-Large 0.75 more accurate, points")

print("== Pruning (Literature.CompressionResults: Gale et al. 2019)")
v("gale_pruned_pct", lit(CR.GaleResNet50PrunedPct), "{:.0f}", "ResNet-50 weights removed, %")
v("gale_loss_word", lit(CR.GaleResNet50Top1LossPts), "words", "top-1 points lost, in prose")

print("== Magnitude pruning, the listing (Scenarios.MagnitudePruned3x3)")
kept, removed = PRUNE3.survivors, PRUNE3.removed
v("lst_threshold", PRUNE3.threshold, "{:.1f}", "magnitude threshold")
for i, w in enumerate(kept, 1):
    v(f"lst_kept{i}", w, "{:.1f}", f"survivor {i}")
v("lst_largest_pruned", max(removed, key=abs), "{:.1f}", "largest weight the threshold removes")
v("lst_kept_n", PRUNE3.nonzeros, "{:.0f}", "survivors")
v("lst_kept_word", PRUNE3.nonzeros, "words", "survivors, in prose")
v("lst_zero_n", PRUNE3.zeros, "{:.0f}", "zeros after pruning")
v("lst_zero_word", PRUNE3.zeros, "words", "zeros after pruning, in prose")
v("lst_n", PRUNE3.entries, "{:.0f}", "weights in the listing")
v("lst_n_word", PRUNE3.entries, "words", "weights in the listing, in prose")
v("lst_sparsity_pct", pct(PRUNE3.zero_share), "{:.0f}", "sparsity, %")

print("== PyTorch CSR of the listing (Scenarios.MagnitudePruned3x3, 64-bit indices)")
v("torch_dense_b", m(PRUNE3.dense_bytes, "byte"), "{:.0f}", "dense FP32 bytes")
v("torch_csr_b", m(PRUNE3.csr_bytes, "byte"), "{:.0f}", "CSR bytes")
v("torch_values_b", PRUNE3.nonzeros * m(PRUNE3.value_bytes, "byte"), "{:.0f}", "CSR values, bytes")
v("torch_index_b", m(PRUNE3.csr_bytes, "byte") - PRUNE3.nonzeros * m(PRUNE3.value_bytes, "byte"), "{:.0f}",
  "CSR column indices and row pointers, bytes")
v("torch_index_bits", m(PRUNE3.index_bytes, "bit"), "{:.0f}", "PyTorch CSR index width, bits")
v("torch_rowptr_n", PRUNE3.row_pointers, "{:.0f}", "CSR row pointers")

print("== The 4 by 4 sparse product (Scenarios.SparseMatVec4x4)")
v("spmv_dense", SPMV.entries, "{:.0f}", "dense multiply-accumulates")
v("spmv_nnz", SPMV.nonzeros, "{:.0f}", "nonzero products")
v("spmv_nnz_word", SPMV.nonzeros, "words", "nonzero products, in prose")
v("spmv_skipped", SPMV.zeros, "{:.0f}", "skipped multiplies")
v("spmv_dim", SPMV.rows, "{:.0f}", "matrix dimension")
v("spmv_zero_pct", pct(SPMV.zero_share), "{:.1f}", "zeros, %")
v("fp32_b", m(BYTES_FP32, "byte"), "{:.0f}", "FP32 value, bytes")
v("spmv_dense_b", m(SPMV.dense_bytes, "byte"), "{:.0f}", "dense FP32 bytes")
v("spmv_pair_b", m(SPMV.csr_value_index_bytes, "byte"), "{:.0f}", "CSR values and columns, bytes")
v("spmv_rowptr_n", SPMV.row_pointers, "{:.0f}", "row pointers")
v("spmv_rowptr_b", m(SPMV.row_pointer_bytes, "byte"), "{:.0f}", "row pointer bytes")
v("spmv_csr_b", m(SPMV.csr_bytes, "byte"), "{:.0f}", "CSR total, bytes")
v("big_layer_dim", LOWRANK.in_dim, "{:,.0f}", "a large layer's rows")
v("big_layer_rowptr", SPMV.variant(matrix=[[0]] * LOWRANK.in_dim).row_pointers, "{:,.0f}", "its row pointers")

print("== Sparse formats (physics.memory, Scenarios.WearableMobileNetV2SparseFlash)")
csr_nz = calc_sparse_bytes_per_nonzero(BYTES_FP32, BYTES_INT32)
coo_nz = calc_sparse_bytes_per_nonzero(BYTES_FP32, BYTES_INT32, indices=np.ndim(SPMV.matrix))  # COO: one index per axis
v("idx16_bits", m(BYTES_INT16, "bit"), "{:.0f}", "short index, bits")
v("idx32_bits", m(BYTES_INT32, "bit"), "{:.0f}", "long index, bits")
v("idx16_b", m(BYTES_INT16, "byte"), "{:.0f}", "short index, bytes")
v("wear_flash_mb", m(WEARABLE.flash_budget, "MB"), "{:.0f}", "wearable flash, MB")
v("wear_values_mb", m(WEARABLE.surviving_bytes, "MB"), "{:.0f}", "surviving FP32 values, MB")
v("wear_nz_b", m(WEARABLE.bytes_per_nonzero, "byte"), "{:.0f}", "FP32 value plus 16-bit index, bytes")
v("wear_stored_mb", m(WEARABLE.stored_bytes, "MB"), "{:.0f}", "stored with indices, MB")
v("csr_nz_b", m(csr_nz, "byte"), "{:.0f}", "CSR bytes per nonzero, 32-bit index")
v("coo_nz_b", m(coo_nz, "byte"), "{:.0f}", "COO bytes per nonzero, two 32-bit indices")
v("csr_even_pct", pct(calc_sparse_breakeven_zero_share(BYTES_FP32, csr_nz)), "{:.0f}", "CSR break-even zeros, %")
v("coo_even_pct", pct(calc_sparse_breakeven_zero_share(BYTES_FP32, coo_nz)), "{:.1f}", "COO break-even zeros, %")

print("== Sparse kernels (Literature.CompressionResults: Gale et al. 2020; Mishra et al. 2021)")
GEMM = SERIES.MishraInt8SparseGemm.entries
v("gale20_lo", lit(CR.GaleSparseKernelSpeedupLow), "{:.1f}", "sparse-kernel speedup, low")
v("gale20_hi", lit(CR.GaleSparseKernelSpeedupHigh), "{:.1f}", "sparse-kernel speedup, high")
v("gale20_peak_pct", lit(CR.GaleSparseKernelPeakPct), "{:.0f}", "share of FP32 peak, %")
v("mishra_peak", lit(CR.MishraDesignPeakSpeedup), "{:.0f}", "2:4 design peak, times")
v("mishra_mn", lit(CR.MishraGemmMN), "{:,.0f}", "M and N of the measured GEMMs")
v("mishra_k_lo", GEMM[0]["k"], "{:,.0f}", "smallest K measured")
v("mishra_x_lo", GEMM[0]["speedup"], "{:.1f}", "speedup at that K")
v("mishra_k_hi", GEMM[-1]["k"], "{:,.0f}", "largest K measured")
v("mishra_x_hi", GEMM[-1]["speedup"], "{:.1f}", "speedup at that K")
v("mishra_dense_top1", lit(CR.MishraResNet50DenseTop1Pct), "{:.1f}", "ResNet-50 dense top-1, %")
v("mishra_sparse_top1", lit(CR.MishraResNet50SparseTop1Pct), "{:.1f}", "ResNet-50 2:4 top-1, %")

print("== Channel tiles and width 0.75 (Scenarios.Channels256To*Tile8, MobileNetV2BlurPipelinePhone)")
narrow = BLUR.pruned
v("tile", TILE250.tile, "{:.0f}", "channels per tile")
v("ch_full", TILE250.channels, "{:.0f}", "channels before pruning")
v("ch_250", TILE250.pruned_channels, "{:.0f}", "pruned to")
v("ch_248", TILE248.pruned_channels, "{:.0f}", "pruned to a multiple of the tile")
v("ch_250_lanes", TILE250.last_tile_lanes, "{:.0f}", "busy lanes in the last tile")
v("ch_250_removed_pct", pct(TILE250.work_removed), "{:.0f}", "work removed, %")
v("w075_params_m", m(narrow.parameters, "param") / 10 ** 6, "{:.1f}", "width 0.75 parameters, millions")
v("w075_mflop", m(narrow.inference_flops, "MFLOP"), "{:.0f}", "width 0.75 per frame, MFLOP")
v("w075_flop_pct", pct(BLUR.flop_fraction), "{:.0f}", "arithmetic kept, %")
v("w075_param_pct", pct(BLUR.param_fraction), "{:.0f}", "parameters kept, %")
v("w075_width", narrow.width_multiplier, "{:.2f}", "width multiplier")
v("w075_square_pct", pct((narrow.width_multiplier / BLUR.full.width_multiplier) ** 2), "{:.0f}",
  "arithmetic kept if every layer scaled with the square of the width, %")

print("== Iterative pruning (Literature.Series.HanAlexNetPruning)")
HAN = SERIES.HanAlexNetPruning.entries
han = {c: [(r["pruned_pct"], r["top5_change"]) for r in HAN if r["curve"] == c]
       for c in ("no_retrain", "retrain", "iterative")}
two_thirds = han["no_retrain"][3]  # the 66.6 percent point
retrain_hold = max(x for x, y in han["retrain"] if y >= 0)
iter_hold = max(x for x, y in han["iterative"] if y >= 0)
v("han_two_thirds_loss", -two_thirds[1], "{:.1f}", "no retraining, points lost at two thirds pruned")
v("han_no_retrain_end_loss", -han["no_retrain"][-1][1], "{:.0f}", "no retraining, points lost at the last point")
v("han_no_retrain_end_pct", han["no_retrain"][-1][0], "{:.0f}", "the last point, % pruned")
v("han_retrain_pct", retrain_hold, "{:.0f}", "one retraining holds accuracy to, % pruned")
v("han_retrain_x", fewer(retrain_hold), "{:.0f}", "the same, times fewer weights")
v("han_iter_pct", iter_hold, "{:.1f}", "iterative holds accuracy to, % pruned")
v("han_iter_x", fewer(iter_hold), "{:.0f}", "the same, times fewer weights")
v("han_iter_drop_x", fewer(min(x for x, y in han["iterative"] if y < 0)), "{:.0f}",
  "iterative falls only past, times fewer weights")

print("== Low rank (Scenarios.Rank128Factorization4096)")
v("lr_dim", LOWRANK.in_dim, "{:.0f}", "layer dimension")
v("lr_dim_c", LOWRANK.in_dim, "{:,.0f}", "layer dimension, with comma")
v("lr_rank", LOWRANK.rank, "{:.0f}", "rank")
v("lr_dense_mb", m(LOWRANK.full_bytes, "MB"), "{:.1f}", "dense FP32, MB")
v("lr_fact_mb", m(LOWRANK.update_bytes, "MB"), "{:.1f}", "two factors, MB")
v("lr_x", LOWRANK.compression, "{:.0f}", "fewer values, times")

print("== Distillation (Scenarios.TeacherSoftTargetsT4; Literature.CompressionResults: Sanh et al. 2019)")
for lab, p_, s_ in zip(SOFT.labels, SOFT.probabilities, SOFT.softened):
    v(f"kd_{lab}_t1", pct(p_), "{:.0f}", f"teacher {lab} at T = 1, %")
    v(f"kd_{lab}_t", pct(s_), "{:.0f}", f"teacher {lab} at T, %")
v("kd_temp", SOFT.temperature, "{:.0f}", "temperature")
v("kd_uniform_word", 1 / SOFT.uniform, "words", "classes (the limit is one over this), in prose")
v("distil_param_cut_pct", lit(CR.SanhDistilBertParamReductionPct), "{:.0f}", "DistilBERT fewer parameters, %")
v("distil_param_kept_pct", 100 - lit(CR.SanhDistilBertParamReductionPct), "{:.0f}", "DistilBERT parameters, % of BERT")
v("distil_speed_pct", lit(CR.SanhDistilBertSpeedupPct), "{:.0f}", "DistilBERT faster, %")
v("distil_speed_rel_pct", 100 + lit(CR.SanhDistilBertSpeedupPct), "{:.0f}", "DistilBERT speed, % of BERT")
v("distil_speed_x", 1 + lit(CR.SanhDistilBertSpeedupPct) / 100, "{:.1f}", "DistilBERT faster, times")
v("distil_kept_pct", lit(CR.SanhDistilBertRetainedPct), "{:.0f}", "DistilBERT score kept, %")

print("== Folding batch norm (Scenarios.ResNet50Conv256FusionH100)")
v("fu_ch", FUSION.channels, "{:.0f}", "channels")
v("fu_hw", FUSION.height, "{:.0f}", "feature map side")
v("fu_fmap_kb", m(FUSION.feature_map, "KB"), "{:.1f}", "feature map, KB")
v("fu_w_mb", m(FUSION.conv_weights, "MB"), "{:.1f}", "convolution weights, MB")
v("fu_conv_mb", m(FUSION.conv_traffic, "MB"), "{:.0f}", "convolution traffic, MB")
v("fu_bn_mb", m(FUSION.norm_traffic, "MB"), "{:.1f}", "batch-norm traffic, MB")
v("fu_relu_mb", m(FUSION.relu_traffic, "MB"), "{:.1f}", "ReLU traffic, MB")
v("fu_unfused_mb", m(FUSION.unfused_traffic, "MB"), "{:.1f}", "unfused traffic, MB")
v("fu_fused_mb", m(FUSION.fused_traffic, "MB"), "{:.0f}", "fused traffic, MB")
v("fu_unfused_trips_word", FUSION.unfused_trips, "words", "unfused transfers, in prose")
v("fu_fused_trips_word", FUSION.fused_trips, "words", "fused transfers, in prose")
v("fu_unfused_trips", FUSION.unfused_trips, "{:.0f}", "unfused transfers")
v("fu_fused_trips", FUSION.fused_trips, "{:.0f}", "fused transfers")
v("fu_cut_pct", pct(FUSION.traffic_cut), "{:.1f}", "traffic cut, %")
v("fu_cut_round_pct", pct(FUSION.traffic_cut), "{:.0f}", "traffic cut, whole percent")
v("h100_bw", m(FUSION.hardware.memory.bandwidth, "GB/s"), "{:,.0f}", "H100 HBM bandwidth, GB/s")
v("fu_unfused_us", m(FUSION.unfused_time, "us"), "{:.1f}", "unfused traffic time, us")
v("fu_fused_us", m(FUSION.fused_time, "us"), "{:.1f}", "fused traffic time, us")
v("fu_ratio", FUSION.time_ratio, "{:.2f}", "unfused over fused, times")

print("== MLP share (physics.transformer.calc_transformer_block_weights)")
att, mlp, block = calc_transformer_block_weights(1)
v("mlp_d2", mlp, "{:.0f}", "MLP multiply-accumulates per token, in d^2")
v("proj_d2", att, "{:.0f}", "projection multiply-accumulates per token, in d^2")
v("block_d2", block, "{:.0f}", "block, in d^2")

print("== Deep Compression cliffs (Literature.CompressionResults: Han et al. 2016)")
v("cliff_svd_pct", lit(CR.HanDeepCompressionSvdCliffPct), "{:.0f}", "SVD cliff, % of size")
v("cliff_prune_pct", lit(CR.HanDeepCompressionPruningCliffPct), "{:.0f}", "pruning cliff, % of size")
v("cliff_both_pct", lit(CR.HanDeepCompressionCombinedCliffPct), "{:.0f}", "pruning plus quantization, % of size")

print("== The blur waterfall (Scenarios.MobileNetV2BlurPipelinePhone; BertBaseMobileCompression)")
v("w075_fp32_mb", m(BLUR.pruned_bytes, "MB"), "{:.1f}", "width 0.75 in FP32, MB")
v("w075_int8_mb", m(BLUR.pruned_int_bytes, "MB"), "{:.1f}", "width 0.75 in INT8, MB")
v("w075_frame_ms", m(BLUR.pruned_cpu_frame, "ms"), "{:.0f}", "width 0.75 frame if time follows FLOPs, ms")
v("w075_int8_frame_ms", m(BLUR.pruned_accelerator_frame, "ms"), "{:.0f}", "pruned INT8 frame on the accelerator, ms")
v("bert_staged_loss_pct", pct(BERT.final_loss), "{:.1f}", "staged accuracy loss, %")
v("bert_quant_first_pct", pct(BERT.quantize_first_loss), "{:.1f}", "quantize-first accuracy loss, %")

print("== Paper against device (Scenarios.PrunedInt8PaperVsMeasured)")
v("paper_prune_x", PAPER.factors["pruning"], "{:.0f}", "pruning on paper, times")
v("paper_quant_x", PAPER.factors["quantization"], "{:.0f}", "INT8 on paper, times")
v("paper_x", PAPER.paper, "{:.0f}", "combined on paper, times")
v("real_x", PAPER.measured, "{:.1f}", "realized, times")
v("real_share_pct", pct(PAPER.realized_share), "{:.1f}", "realized share of paper, %")

print("== Amdahl (Scenarios.ModelFifthOfRequestLatency, ModelThreeFifthsFourTimesFaster)")
a60 = AMDAHL60.result
v("am_model_pct", pct(AMDAHL20.stage_fraction), "{:.0f}", "model share of latency, %")
v("am_rest_pct", pct(1 - AMDAHL20.stage_fraction), "{:.0f}", "rest of the path, %")
v("am_rest", 1 - AMDAHL20.stage_fraction, "{:.1f}", "rest of the path, fraction")
v("am_frac", AMDAHL20.stage_fraction, "{:.1f}", "model share of latency, fraction")
v("am_cap", AMDAHL20.speedup_bound, "{:.2f}", "cap with an infinitely fast model, times")
v("am_wrong", 1 / AMDAHL20.stage_fraction, "{:.0f}", "the wrong answer 100 / 20, times")
v("am_p", a60.stage_fraction, "{:.1f}", "second case: model share")
v("am_p_pct", pct(a60.stage_fraction), "{:.0f}", "second case: model share, %")
v("am_s", AMDAHL60.speedup, "{:.0f}", "second case: model speedup, times")
v("am_one_minus_p", 1 - a60.stage_fraction, "{:.1f}", "second case: 1 - p")
v("am_p_over_s", a60.stage_fraction / AMDAHL60.speedup, "{:.2f}", "second case: p / s")
v("am_overall", a60.overall_speedup, "{:.2f}", "second case: overall speedup, times")

print("== Measured on this CPU (cpu_measure.json, made by measure_cpu.py)")
CPU = json.loads((HERE / "cpu_measure.json").read_text())
dense_, zeros_, slim_ = CPU["rows"]
v("cpu_runs", dense_["runs"], "{:,.0f}", "timed runs per model")
v("cpu_dense_ms", dense_["median_ms"], "{:.1f}", "width 1.0 median, ms")
v("cpu_slim_ms", slim_["median_ms"], "{:.1f}", "width 0.75 median, ms")
v("cpu_zeros_pct", pct(zeros_["zeros_share"]), "{:.0f}", "weights zeroed, %")
v("cpu_zeros_time_pct", pct(zeros_["median_ms"] / dense_["median_ms"]), "{:.0f}", "80% zeros: time, % of dense")
v("cpu_slim_time_pct", pct(slim_["median_ms"] / dense_["median_ms"]), "{:.0f}", "width 0.75: time, % of dense")
v("cpu_slim_flop_pct", pct(slim_["flops"] / dense_["flops"]), "{:.0f}", "width 0.75: FLOPs, % of dense")
v("cpu_slim_saving_pct", 100 - pct(slim_["median_ms"] / dense_["median_ms"]), "{:.0f}", "width 0.75 saves, %")

print("== True or false (Scenarios.IrregularSparsityLatencyTrace)")
v("trace_params_pct", pct(TRACE.work_reduction), "{:.0f}", "parameters removed, %")
v("trace_latency_pct", pct(TRACE.latency_reduction), "{:.0f}", "latency removed, %")

print("== Next lecture (Literature.CompressionResults: Krishnamoorthi 2018)")
v("kq_float_top1", lit(CR.KrishnamoorthiResNetV1_50FloatTop1Pct), "{:.1f}", "ResNet-v1-50 float top-1, %")
v("kq_int8_top1", lit(CR.KrishnamoorthiResNetV1_50Int8Top1Pct), "{:.1f}", "ResNet-v1-50 INT8 top-1, %")

if __name__ == "__main__":
    write(__file__)
