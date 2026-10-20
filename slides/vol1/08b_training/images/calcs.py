#!/usr/bin/env python3
"""Every number on the Volume I Lecture 8b (Model Training, Part 2) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: GPT-2 XL's batch-32 step on
one H100 in FP32 (Scenarios.Gpt2Fp32AdamBatch32OnH100), in mixed precision
(Gpt2MixedPrecisionAdamOnH100), without stored attention scores
(Gpt2MixedPrecisionFlashAttentionOnH100), as two micro-batches of 16
(Gpt2MixedPrecisionTwoMicroBatchesOnH100), and with segment checkpointing
(Gpt2MixedPrecisionSegmentCheckpointedOnH100, Gpt2XlBatch32SegmentCheckpointed);
the run on the FP32, TF32, and 16-bit paths (Gpt2XlRunOnOneH100Fp32, ...Tf32,
Gpt2XlRunOnOneH100); the number formats and the swamped update
(Fp16WeightSmallUpdates); loss scaling on the SSD gradient histogram
(SsdGradientLossScaling); the online-softmax toy (OnlineSoftmaxFourScores);
the attention tiles and stored tensors (AttentionScores4096TokensFp32); and
the single-core tokenizer (Gpt2XlSingleCoreTokenizerFeed). Cited results
(FlashAttention, Korthikanti et al., Chen et al., Micikevicius et al., the
Hugging Face guide) are in Literature. The chapter reads the same scenarios.

Two measured runs of our own are read from data/: the GradScaler trace
(scaler_run.py writes data/scaler_trace.csv) and the SDPA profile
(sdpa_profile.py writes data/sdpa_profile.txt).

This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
this module, so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Scenarios, Literature, Hardware, Models = mlsysim.Scenarios, mlsysim.Literature, mlsysim.Hardware, mlsysim.Models
units = mlsysim.core.units
Q_ = units.Q_
BITS = units.FLOAT_FORMAT_BITS
TR, TM, CR = Literature.TrainingRuns, Literature.TrainingMemory, Literature.CompilerResults

RUN = Scenarios.Gpt2XlRunOnOneH100
RUN_FP32 = Scenarios.Gpt2XlRunOnOneH100Fp32
RUN_TF32 = Scenarios.Gpt2XlRunOnOneH100Tf32
ACT = Scenarios.Gpt2XlBatch32Activations
ACT16 = Scenarios.Gpt2XlMicroBatch16Activations
SEG = Scenarios.Gpt2XlBatch32SegmentCheckpointed
FP32 = Scenarios.Gpt2Fp32AdamBatch32OnH100
MIXED = Scenarios.Gpt2MixedPrecisionAdamOnH100
FLASH = Scenarios.Gpt2MixedPrecisionFlashAttentionOnH100
ACCUM = Scenarios.Gpt2MixedPrecisionTwoMicroBatchesOnH100
CKPT = Scenarios.Gpt2MixedPrecisionSegmentCheckpointedOnH100
SWAMP = Scenarios.Fp16WeightSmallUpdates
LS = Scenarios.SsdGradientLossScaling
SM = Scenarios.OnlineSoftmaxFourScores
TILE = Scenarios.AttentionScores4096TokensFp32
FEED = Scenarios.Gpt2XlSingleCoreTokenizerFeed
H100 = MIXED.hardware
GPT2 = MIXED.workload
GTX580 = Hardware.Workstation.GTX580_3GB
SRAM = Hardware.Tech.Memory.AcceleratorOnChipSram
LADDER = [Hardware.Cloud.V100, Hardware.Cloud.A100, Hardware.Cloud.H100, Hardware.Cloud.B200]
DATA = HERE / "data"


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def r(a, b):
    """A dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def sci(key, x, fmt, label):
    """Record x as a mantissa (key_m) and a power of ten (key_e), for $m\\times10^{e}$."""
    e = math.floor(math.log10(x))
    mant = x / 10 ** e
    if float(fmt.format(mant)) >= 10:   # a mantissa that rounds up to 10
        e, mant = e + 1, mant / 10
    v(f"{key}_m", mant, fmt, f"{label}, mantissa")
    v(f"{key}_e", e, "{:.0f}", f"{label}, power of ten")


def digits(fmt):
    """Decimal digits a binary float keeps."""
    return mlsysim.physics.calc_float_decimal_digits(BITS[fmt][1])


def scaler_trace():
    """Rows (step, scale after the step, skipped) of our GradScaler run."""
    with (DATA / "scaler_trace.csv").open() as f:
        next(f)
        return [(int(row["step"]), float(row["scale_after_step"]), int(row["skipped"]))
                for row in csv.DictReader(f)]


def sdpa_profile():
    """The profiler rows of our SDPA run: section, operator -> (self MB, total ms); and the max difference."""
    rows, section, diff = {}, None, None
    for line in (DATA / "sdpa_profile.txt").read_text().splitlines():
        if line.startswith("=="):
            section = line.strip("= ").strip()
        elif line.startswith("aten::"):
            name, mb, ms = line.split()
            rows[(section, name)] = (float(mb), float(ms))
        elif line.startswith("max |difference|"):
            diff = float(line.split()[-1])
    return rows, diff


fp32_run_d = m(RUN_FP32.training_time, "day")
run_d = m(RUN.training_time, "day")
ckpt_run_d = run_d * (1 + ACT.recompute_overhead)

print("== The machines (Hardware.Cloud.H100; Hardware.Workstation.GTX580_3GB)")
v("h100_gib", m(H100.memory.capacity, "GiB"), "{:.0f}", "H100 HBM, GiB (sold as GB)")
v("h100_gb", m(H100.memory.capacity, "GB"), "{:.1f}", "H100 HBM, decimal GB")
v("h100_bw", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM bandwidth, TB/s")
v("h100_smem_kb", m(H100.memory.shared_memory_per_sm, "KiB"), "{:.0f}", "shared memory per SM, KB (KiB)")
v("gtx580_gb", m(GTX580.memory.capacity, "GiB"), "{:.0f}", "GTX 580 memory, GB (3 GiB board)")
v("gtx580_growth", r(H100.memory.capacity, GTX580.memory.capacity), "{:.0f}", "H100 memory over a GTX 580's")
v("alexnet_params_m", m(Models.Vision.AlexNet.parameters, "count") / 10 ** 6, "{:.0f}", "AlexNet parameters, millions")
v("alexnet_gpus_word", TR.AlexNetGpus, "words", "GTX 580s AlexNet trained on")
v("sram_lo_mb", m(SRAM.capacity_low, "MB"), "{:.0f}", "on-chip SRAM, low, MB")
v("sram_hi_mb", m(SRAM.capacity_high, "MB"), "{:.0f}", "on-chip SRAM, high, MB")
v("sram_bw_lo", m(SRAM.bandwidth_low, "TB/s"), "{:.0f}", "on-chip SRAM bandwidth, low, TB/s")
v("sram_bw_hi", m(SRAM.bandwidth_high, "TB/s"), "{:.0f}", "on-chip SRAM bandwidth, high, TB/s")

print("== GPT-2 XL and the step (Models.Language.GPT2; Scenarios.Gpt2XlBatch32Activations)")
v("gpt2_params_b", m(GPT2.parameters, "count") / 10 ** 9, "{:.1f}", "parameters, billions")
v("gpt2_layers", GPT2.layers, "{:.0f}", "layers")
v("gpt2_hidden", GPT2.hidden_dim, "{:,.0f}", "width")
v("gpt2_heads", GPT2.heads, "{:.0f}", "attention heads")
v("dhead", GPT2.hidden_dim // GPT2.heads, "{:.0f}", "head width")
v("batch", MIXED.batch_size, "{:.0f}", "sequences per step")
v("seq", ACT.sequence, "{:,.0f}", "tokens per sequence")

print("== Static state (Scenarios.Gpt2MixedPrecisionAdamOnH100)")
v("mp_w_b", m(MIXED.bytes_per_value, "byte"), "{:.0f}", "16-bit weight, B")
v("mp_master_b", m(MIXED.master_bytes_per_parameter, "byte"), "{:.0f}", "master copy, B")
v("mp_moments_b", m(MIXED.optimizer_bytes_per_parameter, "byte"), "{:.0f}", "two FP32 moments, B")
v("mp_total_b", m(MIXED.bytes_per_trained_parameter, "byte"), "{:.0f}", "B per parameter")
v("static_gb", m(MIXED.total, "GB"), "{:.0f}", "static state, GB")

print("== The scoreboard (Scenarios.Gpt2Fp32AdamBatch32OnH100 ... Gpt2MixedPrecisionSegmentCheckpointedOnH100)")
v("fp32_step_gb", m(FP32.training_total, "GB"), "{:.0f}", "FP32 step, GB")
v("fp32_act_gb", m(FP32.activations, "GB"), "{:.0f}", "FP32 activations, GB")
v("fp32_h100s_word", round(FP32.training_capacity_multiple), "words", "FP32 step in H100s, rounded")
v("act_gb", m(MIXED.activations, "GB"), "{:.0f}", "16-bit activations, GB")
v("scores_gb", m(ACT.score_activations, "GB"), "{:.0f}", "attention scores, GB")
v("tok_act_gb", m(ACT.per_token_activations, "GB"), "{:.0f}", "activations without the scores, GB")
v("tok_act_gb_1", m(ACT.per_token_activations, "GB"), "{:.1f}", "the same, one decimal")
v("mp_step_gb", m(MIXED.training_total, "GB"), "{:.0f}", "mixed-precision step, GB")
v("mp_step_x", MIXED.training_capacity_multiple, "{:.1f}", "mixed-precision step in H100s")
v("flash_step_gb", m(FLASH.training_total, "GB"), "{:.1f}", "FlashAttention step, GB")
v("flash_over_gb", m(FLASH.training_total - H100.memory.capacity, "GB"), "{:.0f}", "over one H100, GB")
v("accum_step_gb", m(ACCUM.training_total, "GB"), "{:.1f}", "two micro-batches of 16, GB")
v("ckpt_step_gb", m(CKPT.training_total, "GB"), "{:.1f}", "segment checkpointing, GB")
v("fit_shrink_word", round(r(FP32.training_total, ACCUM.training_total)), "words", "FP32 step over the fitted step")
v("fit_shrink_x", r(FP32.training_total, ACCUM.training_total), "{:.0f}", "the same, as a number")
v("gtx580_step_x", round(r(FP32.training_total, GTX580.memory.capacity), -2), "{:.0f}",
  "FP32 step over one GTX 580, to the hundred")

print("== Precision ladder (Hardware.Cloud V100, A100, H100, B200: precision_flops)")


def rate(hw, key):
    pf = hw.compute.precision_flops
    if key == "fp16_dense" and key not in pf:
        return m(hw.compute.peak_flops, "TFLOP/s")
    return m(pf[key], "TFLOP/s") if key in pf else None


LADDER_FORMATS = ["fp32", "tf32", "fp16_dense", "fp8", "fp4"]
LADDER_RATES = [[rate(hw, f) for f in LADDER_FORMATS] for hw in LADDER]
v("v100_fp32", rate(LADDER[0], "fp32"), "{:.1f}", "V100 FP32, TFLOP/s")
v("v100_fp16", rate(LADDER[0], "fp16_dense"), "{:.0f}", "V100 16-bit tensor, TFLOP/s")
v("ladder_fp32_x", rate(LADDER[3], "fp32") / rate(LADDER[0], "fp32"), "{:.1f}", "B200 over V100, FP32")
v("ladder_fp16_x", rate(LADDER[3], "fp16_dense") / rate(LADDER[0], "fp16_dense"), "{:.0f}",
  "B200 over V100, 16-bit tensor")
v("h100_fp32", rate(H100, "fp32"), "{:.0f}", "H100 FP32, TFLOP/s")
v("h100_tf32", rate(H100, "tf32"), "{:.0f}", "H100 TF32 dense, TFLOP/s")
v("h100_peak", m(H100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "H100 16-bit dense, TFLOP/s")
v("h100_peak_sparse", rate(H100, "fp16_sparse"), "{:,.0f}", "H100 16-bit with sparsity, TFLOP/s")
v("h100_tf32_sparse", rate(H100, "tf32_sparse"), "{:.0f}", "H100 TF32 with sparsity, TFLOP/s")
fp16_over_fp32 = r(H100.compute.peak_flops, H100.compute.precision_flops["fp32"])
v("fp16_over_fp32", fp16_over_fp32, "{:.0f}", "16-bit over FP32 peak, H100")
v("fp16_over_fp32_word", round(fp16_over_fp32), "words", "the same, in prose")
v("fp32_unused", round(fp16_over_fp32) - 1, "{:.0f}", "FP32 leaves this many of every 15 unused")

print("== Number formats (core.units.FLOAT_FORMAT_BITS; Scenarios.Fp16WeightSmallUpdates)")
_, fp32_lo, _ = mlsysim.physics.calc_float_range(*BITS["fp32"])
v("fp32_e", BITS["fp32"][0], "{:.0f}", "FP32 exponent bits")
v("fp32_f", BITS["fp32"][1], "{:.0f}", "FP32 mantissa bits")
v("fp16_e", BITS["fp16"][0], "{:.0f}", "FP16 exponent bits")
v("fp16_f", BITS["fp16"][1], "{:.0f}", "FP16 mantissa bits")
v("bf16_f", BITS["bf16"][1], "{:.0f}", "BF16 mantissa bits")
v("fp32_digits_word", round(digits("fp32")), "words", "FP32 decimal digits, rounded")
v("fp16_digits_word", round(digits("fp16")), "words", "FP16 decimal digits, rounded")
v("bf16_digits_lo_word", math.floor(digits("bf16")), "words", "BF16 decimal digits, low")
v("bf16_digits_hi_word", math.ceil(digits("bf16")), "words", "BF16 decimal digits, high")
sci("fp16_min", LS.smallest_normal, "{:.1f}", "FP16 smallest normal")
v("fp16_max", LS.largest, "{:,.0f}", "FP16 largest")
sci("fp16_sub", LS.smallest_subnormal, "{:.0f}", "FP16 smallest subnormal, about")
v("fp16_sub_log2", LS.subnormal_floor_log2, "{:.0f}", "log2 of the FP16 smallest subnormal")
v("fp16_min_log2", LS.normal_floor_log2, "{:.0f}", "log2 of the FP16 smallest normal")
v("fp32_min_e", math.floor(math.log10(fp32_lo)), "{:.0f}", "FP32 and BF16 smallest normal, power of ten")
v("grad_e", math.log10(LS.probe_gradient), "{:.0f}", "the small gradient, power of ten")

print("== Swamping (Scenarios.Fp16WeightSmallUpdates)")
v("sw_weight", SWAMP.weight, "{:.1f}", "weight")
v("sw_update", SWAMP.update, "{:.4f}", "update")
v("sw_true", SWAMP.weight + SWAMP.update, "{:.4f}", "true sum")
v("sw_repeats", SWAMP.repeats, "{:,.0f}", "updates")
v("sw_fp32", SWAMP.final("fp32"), "{:.1f}", "FP32 after the updates")
v("sw_fp16", SWAMP.final("fp16"), "{:.1f}", "FP16 after the updates")
v("sw_next", SWAMP.next_value("fp16"), "{:.5f}", "next FP16 value above the weight")
v("sw_fp16_log2", math.log2(SWAMP.spacing("fp16")), "{:.0f}", "FP16 spacing near 0.5, log2")
sci("sw_fp16_sp", SWAMP.spacing("fp16"), "{:.1f}", "FP16 spacing near 0.5")
v("sw_bf16_log2", math.log2(SWAMP.spacing("bf16")), "{:.0f}", "BF16 spacing near 0.5, log2")
sci("sw_bf16_sp", SWAMP.spacing("bf16"), "{:.1f}", "BF16 spacing near 0.5")
v("sw_ratio", SWAMP.vanishing_ratio("fp16"), "{:,.0f}", "weight over the largest update that rounds away")

print("== Loss scaling (Scenarios.SsdGradientLossScaling; Micikevicius et al. Figure 3)")
v("ls_zero_pct", 100 * LS.zero_fraction, "{:.0f}", "exact zeros, percent")
v("ls_below_pct", 100 * LS.below_share_of_nonzero, "{:.0f}", "nonzero values below the subnormal floor, percent")
v("ls_top_log2", LS.top_edge_log2, "{:.0f}", "largest occupied bin edge, log2")
v("ls_scale", LS.scale, "{:.0f}", "the paper's loss scale")
v("ls_shift", LS.shift, "{:.0f}", "doublings it adds")
v("ls_headroom", LS.headroom_doublings, "{:.0f}", "doublings of headroom")
v("ls_dyn_scale", LS.dynamic_scale, "{:,.0f}", "a dynamic scaler's setting")
sci("ls_scaled", LS.scaled_probe, "{:.2f}", "the small gradient after scaling")
v("ls_overflow", LS.dynamic_overflow_threshold, "{:.0f}", "gradient that overflows at that scale")

print("== Our GradScaler run (data/scaler_trace.csv)")
TRACE = scaler_trace()
trace_init = TRACE[0][1]
first_double = next(s for s, sc, _ in TRACE if sc > trace_init)
skips = [s for s, _, k in TRACE if k]
top = max(sc for _, sc, _ in TRACE)
v("gs_steps", len(TRACE), "{:,.0f}", "steps run")
v("gs_init_log2", math.log2(trace_init), "{:.0f}", "initial scale, log2")
v("gs_interval", first_double, "{:,.0f}", "clean steps per doubling")
v("gs_doublings_word", math.log2(top / trace_init), "words", "doublings before the first skip")
v("gs_top_log2", math.log2(top), "{:.0f}", "ceiling, log2")
v("gs_first_skip", skips[0], "{:,.0f}", "first skipped step")
v("gs_cycle", skips[1] - skips[0], "{:,.0f}", "steps between skips")
v("gs_skips", len(skips), "{:.0f}", "skipped steps")

print("== Run time by format (Scenarios.Gpt2XlRunOnOneH100Fp32, ...Tf32, Gpt2XlRunOnOneH100)")
v("eta_pct", 100 * RUN.efficiency, "{:.0f}", "utilization held fixed, percent")
v("price_hr", m(RUN.hourly_price, "USD/hour"), "{:.0f}", "USD per H100-hour")
v("fp32_days", fp32_run_d, "{:.0f}", "FP32 run, days")
v("fp32_usd", round(m(RUN_FP32.rental_cost, "USD")), "{:,.0f}", "FP32 run, USD")
v("tf32_days", m(RUN_TF32.training_time, "day"), "{:.0f}", "TF32 run, days")
v("run_days", run_d, "{:.0f}", "16-bit run, days")
v("run_usd", round(m(RUN.rental_cost, "USD")), "{:,.0f}", "16-bit run, USD")

print("== One layer's bytes (Scenarios.Gpt2XlBatch32Activations line items)")
v("layer_scores_mb", m(ACT.score_activations_per_layer, "MB"), "{:,.0f}", "scores per layer, MB")
v("layer_items_mb", m(ACT.line_items_per_layer, "MB"), "{:,.0f}", "line items per layer, MB")
v("layer_scores_pct", 100 * r(ACT.score_activations_per_layer, ACT.line_items_per_layer), "{:.0f}",
  "scores' share of a layer, percent")
v("layer_proj_mb", m(ACT.projection_activations_per_layer, "MB"), "{:.0f}", "projections per layer, MB")
v("act_items_gb", m(ACT.line_items_total, "GB"), "{:.0f}", "line items over all layers, GB")
mem = mlsysim.physics.memory
v("score_p_b", mem.SCORE_SOFTMAX_BYTES, "{:.0f}", "softmax probabilities, B per entry")
v("score_mask_b", mem.SCORE_DROPOUT_MASK_BYTES, "{:.0f}", "dropout mask, B per entry")
v("score_entry_b", mem.KORTHIKANTI_SCORE_BYTES, "{:.0f}", "bytes per score entry")

print("== Attention I/O (Hardware.Cloud.H100; head width of GPT-2 XL)")
v("attn_intensity", m(mlsysim.physics.calc_score_matrix_intensity(GPT2.hidden_dim // GPT2.heads, units.BYTES_FP16),
                       "flop/byte"), "{:.0f}", "score intensity d_head/2, FLOP/B")
v("h100_ridge", m(H100.compute.peak_flops / H100.memory.bandwidth, "flop/byte"), "{:.0f}", "H100 ridge, FLOP/B")

print("== FlashAttention, Figures 1 and 2 (Literature.CompilerResults)")
v("fa_sram_tbs", CR.FlashAttentionA100SramBandwidthTBs, "{:.0f}", "A100 SRAM, TB/s")
v("fa_sram_mb", CR.FlashAttentionA100SramCapacityMB, "{:.0f}", "A100 SRAM, MB")
v("fa_hbm_tbs", CR.FlashAttentionA100HbmBandwidthTBs, "{:.1f}", "A100 HBM, TB/s")
v("fa_hbm_gb", CR.FlashAttentionA100HbmCapacityGB, "{:.0f}", "A100 HBM, GB")
v("fa_bw_x", CR.FlashAttentionA100SramBandwidthTBs / CR.FlashAttentionA100HbmBandwidthTBs, "{:.1f}",
  "SRAM over HBM bandwidth")
v("fa_speedup", CR.FlashAttentionGpt2AttentionSpeedup, "{:.1f}", "GPT-2 attention speedup")
v("fa_gflop_std", CR.FlashAttentionGpt2MediumStandardGflops, "{:.1f}", "standard, GFLOPs")
v("fa_gflop_fa", CR.FlashAttentionGpt2MediumFusedGflops, "{:.1f}", "FlashAttention, GFLOPs")
v("fa_hbm_std", CR.FlashAttentionGpt2MediumStandardHbmTrafficGB, "{:.1f}", "standard, HBM GB")
v("fa_hbm_fa", CR.FlashAttentionGpt2MediumFusedHbmTrafficGB, "{:.1f}", "FlashAttention, HBM GB")
v("fa_ms_std", CR.FlashAttentionGpt2MediumStandardRuntimeMs, "{:.1f}", "standard, ms")
v("fa_ms_fa", CR.FlashAttentionGpt2MediumFusedRuntimeMs, "{:.1f}", "FlashAttention, ms")
v("fa_more_flops_pct", 100 * (CR.FlashAttentionGpt2MediumFusedGflops / CR.FlashAttentionGpt2MediumStandardGflops - 1),
  "{:.0f}", "more arithmetic, percent")
v("fa_fewer_bytes_x", CR.FlashAttentionGpt2MediumStandardHbmTrafficGB / CR.FlashAttentionGpt2MediumFusedHbmTrafficGB,
  "{:.0f}", "fewer HBM bytes, times, about")
v("fa_faster", CR.FlashAttentionGpt2MediumStandardRuntimeMs / CR.FlashAttentionGpt2MediumFusedRuntimeMs, "{:.1f}",
  "faster, times")

print("== Tiles and stored tensors (Scenarios.AttentionScores4096TokensFp32)")
v("tile_seq", TILE.seq_len, "{:,.0f}", "tokens")
v("tile_side", TILE.tile, "{:.0f}", "tile side")
v("tile_kb", m(TILE.tile_bytes, "KB"), "{:.1f}", "one tile, KB")
v("tile_full_mb", m(TILE.score_matrix, "MB"), "{:.1f}", "full matrix, MB")
v("tiles_per_side", TILE.tiles_per_side, "{:.0f}", "tiles per side")
v("tile_count", TILE.tile_count, "{:,.0f}", "tiles in all")
v("tile_heads", TILE.heads, "{:.0f}", "heads")
v("tile_heads_gb", m(TILE.all_heads_scores, "GB"), "{:.1f}", "all heads' scores, GB")
v("store_std_mb", m(TILE.stored_standard, "MB"), "{:.1f}", "standard attention stores, MB")
v("store_fa_mb", m(TILE.stored_flash, "MB"), "{:.1f}", "FlashAttention stores, MB")
v("store_x", TILE.stored_ratio, "{:.1f}", "ratio")

print("== Online softmax (Scenarios.OnlineSoftmaxFourScores)")
b1, b2 = SM.blocks
whole = SM.whole_row
for i, (x, val_) in enumerate(zip(SM.scores, SM.values), start=1):
    v(f"sm_x{i}", x, "{:.0f}", f"score {i}")
    v(f"sm_v{i}", val_, "{:.0f}", f"value {i}")
    v(f"sm_d{i}", x - whole["max"], "{:.0f}", f"score {i} minus the row maximum")
v("sm_d2b1", SM.scores[0] - b1["max"], "{:.0f}", "score 1 minus block 1's maximum")
v("sm_m", whole["max"], "{:.0f}", "row maximum")
v("sm_m1", b1["max"], "{:.0f}", "block 1 maximum")
v("sm_l", whole["sum"], "{:.3f}", "whole-row sum")
v("sm_l1", b1["sum"], "{:.3f}", "block 1 sum")
v("sm_f", b2["rescale"], "{:.3f}", "rescale factor")
v("sm_l1f", b1["sum"] * b2["rescale"], "{:.3f}", "old sum rescaled")
v("sm_t3", SM.block_terms(1)[0], "{:.3f}", "block 2's first term")
v("sm_o", whole["output"], "{:.2f}", "whole-row output")
v("sm_o1", b1["output"], "{:.2f}", "block 1 output")
v("sm_o1f", b1["output"] * b2["rescale"], "{:.2f}", "old output rescaled")
v("sm_vt3", SM.values[2] * SM.block_terms(1)[0], "{:.2f}", "block 2's first weighted value")
v("sm_avg", SM.weighted_average, "{:.1f}", "O over l")

print("== Our SDPA profile (data/sdpa_profile.txt; GPT-2 XL attention shape)")
PROF, PROF_DIFF = sdpa_profile()
t_flash = PROF[("default dispatch", "aten::_scaled_dot_product_flash_attention_for_cpu")][1]
t_math = PROF[("forced math backend", "aten::_scaled_dot_product_attention_math")][1]
v("sdpa_flash_ms", t_flash, "{:.1f}", "fused kernel, ms")
v("sdpa_math_ms", t_math, "{:.1f}", "math backend, ms")
v("sdpa_x", t_math / t_flash, "{:.1f}", "math over fused")
v("sdpa_bmm_mb", PROF[("forced math backend", "aten::bmm")][0], "{:.1f}", "batched matmul allocation, MB")
v("sdpa_out_mb", PROF[("default dispatch", "aten::empty_strided")][0], "{:.1f}", "output allocation, MB")
score_mb = m(GPT2.heads * ACT.sequence ** 2 * units.BYTES_FP32, "MB")
v("sdpa_scores_mb", score_mb, "{:.1f}", "one sequence's FP32 score matrix, MB")
v("sdpa_diff_e", math.floor(math.log10(PROF_DIFF)), "{:.0f}", "outputs agree to this power of ten")

print("== Batch fit and accumulation (Scenarios.Gpt2MixedPrecisionFlashAttentionOnH100, ...TwoMicroBatches)")
v("per_seq_gb", m(ACT.per_sequence_activations, "GB"), "{:.2f}", "activations per sequence, GB")
v("fit_batch", FLASH.resident_batch_limit, "{:.0f}", "sequences that fit, about")
v("micro", ACCUM.batch_size, "{:.0f}", "micro-batch")
v("accum_k", MIXED.batch_size // ACCUM.batch_size, "{:.0f}", "micro-batches per update")
v("accum_k_word", MIXED.batch_size // ACCUM.batch_size, "words", "the same, in prose")

print("== Segment checkpointing (Scenarios.Gpt2XlBatch32SegmentCheckpointed)")
v("segments", SEG.checkpoint_segments, "{:.0f}", "segments")
v("segments_word", SEG.checkpoint_segments, "words", "the same, in prose")
v("segment_layers", SEG.segment_layers, "{:.0f}", "layers per segment")
v("seg_inputs_gb", m(SEG.segment_inputs, "GB"), "{:.2f}", "kept segment inputs, GB")
v("live_seg_gb", m(SEG.live_segment, "GB"), "{:.1f}", "one rebuilt segment, GB")
v("ckpt_act_gb", m(SEG.checkpointed_activations, "GB"), "{:.1f}", "checkpointed activations, GB")
v("ckpt_x", r(SEG.per_token_activations, SEG.checkpointed_activations), "{:.1f}", "less than without, times")
v("fwd_passes_word", ACT.forward_equivalents, "words", "a step in forward passes")
v("fwd_passes_ckpt_word", ACT.recompute_forward_equivalents, "words", "the same with checkpointing")
v("recompute_pct", 100 * ACT.recompute_overhead, "{:.0f}", "extra step arithmetic, percent")
v("ckpt_days", ckpt_run_d, "{:.0f}", "run with checkpointing, days")
v("ckpt_extra_days", ckpt_run_d - run_d, "{:.0f}", "days checkpointing adds")

print("== Square-root checkpointing (Scenarios.Gpt2XlBatch32SegmentCheckpointed.checkpoint_memory_units)")
L_ = GPT2.layers
units_k = {k: SEG.checkpoint_memory_units(k) for k in range(1, L_ + 1)}
k_best = min(units_k, key=units_k.get)
v("sqrt_l", math.sqrt(L_), "{:.0f}", "square root of the layer count, about")
v("k_one_units", units_k[1], "{:.0f}", "k = 1, layers")
v("k_best", k_best, "{:.0f}", "best k")
v("k_best_units", units_k[k_best], "{:.0f}", "best k, layers, about")
v("k_seg_units", units_k[SEG.checkpoint_segments], "{:.0f}", "k = 8, layers")
v("sqrt_saving_pct", 100 * (1 - units_k[k_best] / L_), "{:.0f}", "less activation memory, percent")
v("chen_layers", TM.ChenResidualNetLayers, "{:,.0f}", "Chen et al.: layers")
v("chen_before_gb", TM.ChenResidualNetMemoryBeforeGB, "{:.0f}", "Chen et al.: GB before")
v("chen_after_gb", TM.ChenResidualNetMemoryAfterGB, "{:.0f}", "Chen et al.: GB after")
v("chen_time_pct", 100 * TM.ChenResidualNetExtraTimeFraction, "{:.0f}", "Chen et al.: more time, percent")

print("== Selective recomputation (Literature.TrainingMemory; Scenarios.Gpt2XlBatch32Activations)")
v("k_tok_coef", ACT.per_token_bytes_coefficient, "{:.0f}", "34 of sbh(34 + 5as/h)")
v("k_score_coef", ACT.score_bytes_coefficient, "{:.0f}", "5as/h for GPT-2 XL")
v("k_score_pct", 100 * ACT.score_share, "{:.1f}", "scores' share, percent")
k_none, k_full = TM.Korthikanti22BLayerNoRecomputeMs, TM.Korthikanti22BLayerFullRecomputeMs
k_sel = TM.Korthikanti22BLayerSelectiveRecomputeMs
v("k_none_ms", k_none, "{:.1f}", "Table 4, no recomputation, ms")
v("k_full_ms", k_full, "{:.1f}", "Table 4, full recomputation, ms")
v("k_sel_ms", k_sel, "{:.1f}", "Table 4, selective recomputation, ms")
v("k_full_pct", 100 * (k_full / k_none - 1), "{:.0f}", "full recomputation, percent more")
v("k_sel_pct", 100 * (k_sel / k_none - 1), "{:.0f}", "selective recomputation, percent more")
v("k_model_b", TM.KorthikantiTableModelParamsB, "{:.0f}", "the timed model, billions of parameters")
v("k_530b", TM.Korthikanti530BParamsB, "{:.0f}", "largest model, billions")
v("k_gpus", TM.Korthikanti530BA100Gpus, "{:,.0f}", "A100s")
v("k_mfu", 100 * TM.Korthikanti530BMfu, "{:.1f}", "MFU, percent")
v("k_mfu_full", 100 * TM.Korthikanti530BMfuFullRecompute, "{:.1f}", "MFU with full recomputation, percent")
v("k_mfu_gain_pct", 100 * (TM.Korthikanti530BMfu / TM.Korthikanti530BMfuFullRecompute - 1), "{:.0f}",
  "faster, percent")

print("== Hugging Face guide (Literature.TrainingMemory)")
v("hf_slow_pct", 100 * TM.HuggingFaceCheckpointingSlowdownFraction, "{:.0f}", "checkpointing slowdown, percent")
v("hf_batch", TM.HuggingFaceAccumulationExampleBatch, "{:.0f}", "example per-device batch")
v("hf_steps", TM.HuggingFaceAccumulationExampleSteps, "{:.0f}", "example accumulation steps")
v("hf_effective", TM.HuggingFaceAccumulationExampleBatch * TM.HuggingFaceAccumulationExampleSteps, "{:.0f}",
  "example effective batch")

print("== The fitted run's input (Scenarios.Gpt2XlSingleCoreTokenizerFeed)")
v("busy_pct", 100 * round(FEED.busy_fraction, 2), "{:.0f}", "accelerator busy with one tokenizer core, percent")

if __name__ == "__main__":
    write(__file__)
