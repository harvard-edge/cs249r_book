#!/usr/bin/env python3
"""Every number on the Volume I Lecture 8 (Model Training, Part 1) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: GPT-2 XL's run on one H100
(Scenarios.Gpt2XlRunOnOneH100: time, steps, energy, rent), its batch-32 step
counted layer by layer (Gpt2XlBatch32Block), its training state in FP32 and
in mixed precision with the Adam update traffic and the checkpoint
(TrainingStateFitScenario), its saved activations with the per-layer line
items (Gpt2XlBatch32Activations), the 7B and PaLM 540B states, the Llama 3.1
8B full and LoRA fine-tunes, the ResNet-50 input feed on an H100
(ResNet50H100InputFeed), the GIL-bound loader (GilThreadedLoaderStall), and
the Llama 3.1 405B model-card run (Llama31_405BModelCardRun). Cited figures
(PaLM Table 3, the PaLM loss spikes, the Llama 3.1 model card, the OPT-175B
logbook, ZeRO, LoRA) are in Literature. The chapter reads the same scenarios.
This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
this module, so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Scenarios, Literature, Hardware, Systems = mlsysim.Scenarios, mlsysim.Literature, mlsysim.Hardware, mlsysim.Systems
units = mlsysim.core.units
Q_ = units.Q_
TR = Literature.TrainingRuns

RUN = Scenarios.Gpt2XlRunOnOneH100
RUN_HALF = Scenarios.Gpt2XlRunOnOneH100HalfUtilization
RUN_NODE = Scenarios.Gpt2XlRunOnOneDgxH100
BLOCK = Scenarios.Gpt2XlBatch32Block
ACT = Scenarios.Gpt2XlBatch32Activations
ACT2K = Scenarios.Gpt2XlBatch32ActivationsAt2048
FP32 = Scenarios.Gpt2Fp32AdamOnH100
FP32_STEP = Scenarios.Gpt2Fp32AdamBatch32OnH100
MIXED = Scenarios.Gpt2MixedPrecisionAdamOnH100
SEVEN_B = Scenarios.Llama2_7BMixedPrecisionAdam
PALM = Scenarios.Palm540BMixedPrecisionAdam
FULL = Scenarios.Llama3_8BFullFineTuneOnH100
LORA = Scenarios.Llama3_8BLoraOnH100
FEED = Scenarios.ResNet50H100InputFeed
FEED_A100 = Scenarios.ResNet50A100InputFeed
GIL = Scenarios.GilThreadedLoaderStall
GIL_FAST = Scenarios.GilThreadedLoaderFasterAccelerator
CARD = Scenarios.Llama31_405BModelCardRun
H100 = RUN.hardware
GPT2 = RUN.workload
NVME = Hardware.Tech.Storage.NvmeGen4.bandwidth


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def sci(key, x, fmt, label):
    """Record x as a mantissa (key_m) and a power of ten (key_e), for $m\\times10^{e}$."""
    e = math.floor(math.log10(x))
    mant = x / 10 ** e
    if float(fmt.format(mant)) >= 10:   # a mantissa that rounds up to 10
        e, mant = e + 1, mant / 10
    v(f"{key}_m", mant, fmt, f"{label}, mantissa")
    v(f"{key}_e", e, "{:.0f}", f"{label}, power of ten")


GB_PER_GIB = m(Q_(1, "GiB"), "GB")

print("== Reference machine (Hardware.Cloud.H100)")
v("h100_peak", m(H100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "dense 16-bit peak, TFLOP/s")
v("h100_bw", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "HBM bandwidth, TB/s")
v("h100_tdp", m(H100.tdp, "W"), "{:.0f}", "TDP, W")
v("h100_gib", m(H100.memory.capacity, "GiB"), "{:.0f}", "HBM, GiB (sold as GB)")
v("h100_gb", m(H100.memory.capacity, "GB"), "{:.1f}", "HBM, decimal GB")

print("== GPT-2 XL (Models.Language.GPT2)")
sci("gpt2_params", m(GPT2.parameters, "count"), "{:.1f}", "parameters")
v("gpt2_params_b", m(GPT2.parameters, "count") / 10 ** 9, "{:.1f}", "parameters, billions")
v("gpt2_layers", GPT2.layers, "{:.0f}", "layers")
v("gpt2_hidden", GPT2.hidden_dim, "{:.0f}", "width (config n_embd)")
v("gpt2_hidden_c", GPT2.hidden_dim, "{:,.0f}", "width, with a comma")
v("gpt2_heads", GPT2.heads, "{:.0f}", "attention heads")
v("gpt2_ctx", GPT2.context_length, "{:.0f}", "context (config n_ctx)")
sci("gpt2_fwd", m(GPT2.inference_flops, "flop"), "{:.0f}", "FLOPs per generated token")
sci("gpt2_run", m(GPT2.training_ops, "flop"), "{:.1f}", "FLOPs per training run")
ratio = (GPT2.training_ops / GPT2.inference_flops).to("").magnitude
sci("tokens_ratio", ratio, "{:.0f}", "run over one token")
v("tokens_ratio_b", ratio / 10 ** 9, "{:.0f}", "the same, billions")
v("tokens_ratio_orders_word", round(math.log10(ratio)), "words", "orders of magnitude apart, rounded")

print("== One step (Scenarios.Gpt2XlBatch32Block, Gpt2XlRunOnOneH100)")
v("batch", BLOCK.batch_size, "{:.0f}", "sequences per step")
v("seq", BLOCK.context_tokens, "{:,.0f}", "tokens per sequence")
v("tokens_step", RUN.tokens_per_step, "{:,.0f}", "tokens per step")
v("step_tflop", m(RUN.step_operations, "TFLOP"), "{:.0f}", "6P step, TFLOP")
sci("step_flop", m(RUN.step_operations, "flop"), "{:.2f}", "6P step, FLOPs")
v("layer_step_tflop", m(BLOCK.training_step_flops, "TFLOP"), "{:.1f}", "per-layer step count, TFLOP")
v("attn_extra_pct", 100 * (BLOCK.training_step_flops / BLOCK.six_p_step_flops - 1).to("").magnitude,
  "{:.0f}", "attention products over the 6P count, percent")
v("eta", RUN.efficiency, "{:.2f}", "utilization")
v("eta_pct", 100 * RUN.efficiency, "{:.0f}", "utilization, percent")
v("step_s", m(RUN.step_time, "s"), "{:.2f}", "step time, s")
v("steps_m", RUN.steps / 10 ** 6, "{:.1f}", "steps, millions")
sci("run_tokens", m(RUN.training_tokens, "count"), "{:.1f}", "tokens the run covers")
v("eta_half_pct", 100 * RUN_HALF.efficiency, "{:.0f}", "half the utilization, percent")
v("step_s_half", m(RUN_HALF.step_time, "s"), "{:.1f}", "step time at half the utilization, s")

print("== The run (Scenarios.Gpt2XlRunOnOneH100)")
sci("run_s", m(RUN.training_time, "s"), "{:.1f}", "run time, s")
v("run_days", m(RUN.training_time, "day"), "{:.0f}", "run time, days")
v("run_kwh", m(RUN.energy, "kWh"), "{:.0f}", "accelerator energy, kWh")
v("price_hr", m(RUN.hourly_price, "USD/hour"), "{:.0f}", "rental price, USD per H100-hour")
v("run_usd", round(m(RUN.rental_cost, "USD")), "{:,.0f}", "rental cost, USD")
plus_one = RUN.variant(efficiency=RUN.efficiency + 1 / 100)
v("point_hours", m(RUN.training_time - plus_one.training_time, "hour"), "{:.1f}", "one utilization point, hours")
v("point_pct", 100 * (1 - plus_one.training_time / RUN.training_time).to("").magnitude, "{:.0f}",
  "one utilization point, percent of the run")
v("node_gpus_word", RUN_NODE.n_devices, "words", "H100s in one DGX H100")
v("node_days", m(RUN_NODE.training_time, "day"), "{:.1f}", "run on one node, perfect scaling, days")

print("== Llama 3.1 model card (Literature.TrainingRuns; Scenarios.Llama31_405BModelCardRun)")
v("card_8b_mh", TR.Llama31_8B_GpuHours / 10 ** 6, "{:.2f}", "8B GPU hours, millions")
v("card_70b_mh", TR.Llama31_70B_GpuHours / 10 ** 6, "{:.1f}", "70B GPU hours, millions")
v("card_405b_mh", TR.Llama31_405B_GpuHours / 10 ** 6, "{:.2f}", "405B GPU hours, millions")
v("card_total_mh", TR.Llama31_TotalGpuHours / 10 ** 6, "{:.1f}", "total GPU hours, millions")
v("card_w", TR.Llama31_CardPowerW, "{:.0f}", "power per GPU, W")
v("l405_params_b", m(CARD.workload.parameters, "count") / 10 ** 9, "{:.0f}", "405B parameters, billions")
v("l405_tokens_t", m(CARD.workload.training_tokens, "count") / 10 ** 12, "{:.1f}", "405B tokens, trillions")
sci("l405_6pd", m(CARD.training_operations, "flop"), "{:.2f}", "6PD, FLOPs")
sci("l405_paper", float(TR.Llama3_405B_TrainingFlops), "{:.1f}", "paper's stated budget, FLOPs")
v("sec_per_hr", m(Q_(1, "hour"), "s"), "{:,.0f}", "seconds per hour")
v("card_rate", m(CARD.achieved_rate, "TFLOP/s"), "{:.0f}", "delivered per GPU over all hours, TFLOP/s")
v("card_util_pct", 100 * (CARD.achieved_rate / H100.compute.peak_flops).to("").magnitude, "{:.1f}",
  "implied utilization, percent")
v("card_gwh", m(CARD.energy, "GWh"), "{:.1f}", "energy at the card's power, GWh")
v("card_over_run", round((CARD.energy / RUN.energy).to("").magnitude, -2), "{:,.0f}",
  "card energy over the GPT-2 XL run, rounded to hundreds")
v("l405_mfu_lo", 100 * TR.Llama3_405B_MfuLong, "{:.0f}", "Table 4 MFU, low, percent")
v("l405_mfu_hi", 100 * TR.Llama3_405B_Mfu8K, "{:.0f}", "Table 4 MFU, high, percent")
v("l405_interruptions", TR.Llama3_405B_UnexpectedInterruptions + TR.Llama3_405B_PlannedInterruptions,
  "{:.0f}", "job interruptions in the snapshot")
v("l405_snapshot_days", TR.Llama3_405B_SnapshotDays, "{:.0f}", "snapshot, days")
v("card_tco2", TR.Llama31_405B_LocationTco2eq, "{:,.0f}", "405B location-based emissions, t CO2eq")

print("== Optimizer state, FP32 (Scenarios.Gpt2Fp32AdamOnH100, ResNet50Fp32SgdMomentum)")
fp32_b = m(FP32.bytes_per_value, "byte")
v("fp32_b", fp32_b, "{:.0f}", "bytes per FP32 value")
v("sgd_b", m(FP32.bytes_per_trained_parameter - FP32.optimizer_bytes_per_parameter, "byte"), "{:.0f}",
  "SGD: weights and gradients, B per parameter")
v("momentum_b", m(Scenarios.ResNet50Fp32SgdMomentum.bytes_per_trained_parameter, "byte"), "{:.0f}",
  "momentum, B per parameter")
v("adam_fp32_b", m(FP32.bytes_per_trained_parameter, "byte"), "{:.0f}", "Adam in FP32, B per parameter")

print("== Mixed-precision Adam (Scenarios.Gpt2MixedPrecisionAdamOnH100)")
p = m(GPT2.parameters, "count")
v("mp_w_b", m(MIXED.bytes_per_value, "byte"), "{:.0f}", "16-bit weight, B")
v("mp_master_b", m(MIXED.master_bytes_per_parameter, "byte"), "{:.0f}", "master copy, B")
v("mp_moments_b", m(MIXED.optimizer_bytes_per_parameter, "byte"), "{:.0f}", "two FP32 moments, B")
v("mp_total_b", m(MIXED.bytes_per_trained_parameter, "byte"), "{:.0f}", "B per parameter")
v("mp_total_b_word", m(MIXED.bytes_per_trained_parameter, "byte"), "words", "the same, in prose")
v("mp_zero_k", m(MIXED.master_bytes_per_parameter + MIXED.optimizer_bytes_per_parameter, "byte"), "{:.0f}",
  "ZeRO's K: master copy and moments, B")
v("mp_copies_word", (MIXED.bytes_per_trained_parameter / MIXED.bytes_per_value).to("").magnitude, "words",
  "training state over the 16-bit weights")
v("mp_w_gb", m(MIXED.weights, "GB"), "{:.0f}", "GPT-2 XL 16-bit weights, GB")
v("mp_g_gb", m(MIXED.gradients, "GB"), "{:.0f}", "16-bit gradients, GB")
v("mp_master_gb", m(MIXED.master_weights, "GB"), "{:.0f}", "master copy, GB")
v("mp_moments_gb", m(MIXED.optimizer_state, "GB"), "{:.0f}", "moments, GB")
v("mp_state_gb", m(MIXED.total, "GB"), "{:.0f}", "static state, GB")
v("fp32_w_gb", m(FP32.weights, "GB"), "{:.0f}", "FP32 weights, GB")
v("fp32_g_gb", m(FP32.gradients, "GB"), "{:.0f}", "FP32 gradients, GB")
v("fp32_state_gb", m(FP32.total, "GB"), "{:.0f}", "FP32 static state, GB")
v("state_share_pct", 100 * MIXED.capacity_share, "{:.0f}", "static state, percent of an H100")
v("zero_act_gb", TR.ZeroGpt2ActivationEstimateGB, "{:.0f}", "ZeRO's activation estimate, GB")

print("== A 7B model (Scenarios.Llama2_7BMixedPrecisionAdam)")
v("b7_params_b", m(SEVEN_B.workload.parameters, "count") / 10 ** 9, "{:.0f}", "parameters, billions")
v("b7_w_gb", m(SEVEN_B.weights, "GB"), "{:.0f}", "16-bit weights (serving), GB")
v("b7_g_gb", m(SEVEN_B.gradients, "GB"), "{:.0f}", "gradients, GB")
v("b7_master_gb", m(SEVEN_B.master_weights, "GB"), "{:.0f}", "master copy, GB")
v("b7_moments_gb", m(SEVEN_B.optimizer_state, "GB"), "{:.0f}", "moments, GB")
v("b7_state_gb", m(SEVEN_B.total, "GB"), "{:.0f}", "training state, GB")

print("== Adam update traffic (Scenarios.Gpt2Fp32AdamOnH100)")
read_b = m(FP32.gradients + FP32.weights + FP32.optimizer_state, "byte") / p
v("upd_read_b", read_b, "{:.0f}", "bytes read per parameter")
v("upd_write_b", m(FP32.update_bytes_per_parameter, "byte") - read_b, "{:.0f}", "bytes written per parameter")
v("upd_b", m(FP32.update_bytes_per_parameter, "byte"), "{:.0f}", "bytes per parameter")
v("upd_gb", m(FP32.update_traffic, "GB"), "{:.0f}", "traffic per update, GB")
v("upd_ms", m(FP32.update_time, "ms"), "{:.1f}", "update at HBM bandwidth, ms")
v("upd_share_pct", 100 * (FP32.update_time / RUN.step_time).to("").magnitude, "{:.0f}", "share of a step, percent")

print("== Activations (Scenarios.Gpt2XlBatch32Activations)")
v("act_tok_gb", m(ACT.per_token_activations, "GB"), "{:.0f}", "per-token tensors at 1,024, GB")
v("act_scores_gb", m(ACT.score_activations, "GB"), "{:.0f}", "attention scores at 1,024, GB")
v("act_gb", m(ACT.activations("none"), "GB"), "{:.0f}", "activations at 1,024, GB")
v("act_gb_1", m(ACT.activations("none"), "GB"), "{:.1f}", "the same, one decimal")
v("seq2", ACT2K.sequence, "{:,.0f}", "doubled sequence")
v("act2k_tok_gb", m(ACT2K.per_token_activations, "GB"), "{:.0f}", "per-token tensors at 2,048, GB")
v("act2k_scores_gb", m(ACT2K.score_activations, "GB"), "{:.0f}", "attention scores at 2,048, GB")
v("act_items_gb", m(ACT.line_items_total, "GB"), "{:.0f}", "the chapter's line items, all layers, GB")
v("act_extra_gb", m(ACT.activations("none") - ACT.line_items_total, "GB"), "{:.0f}",
  "library total over the line items, GB")
v("layer_scores_mb", m(ACT.score_activations_per_layer, "MB"), "{:,.0f}", "scores per layer, MB")
v("layer_items_mb", m(ACT.line_items_per_layer, "MB"), "{:,.0f}", "line items per layer, MB")
v("layer_scores_pct", 100 * (ACT.score_activations_per_layer / ACT.line_items_per_layer).to("").magnitude,
  "{:.0f}", "scores' share of a layer, percent")
v("step_gb", m(MIXED.training_total, "GB"), "{:.0f}", "mixed-precision step, GB")
v("step_x", MIXED.training_capacity_multiple, "{:.1f}", "step over one H100")
v("fp32_step_gb", m(FP32_STEP.training_total, "GB"), "{:.0f}", "FP32 step, GB")
v("fp32_step_h100s_word", FP32_STEP.training_capacity_multiple, "words", "FP32 step in H100s")

print("== Llama 3.1 8B fine-tuning (Scenarios.Llama3_8BFullFineTuneOnH100, Llama3_8BLoraOnH100)")
LA = Scenarios.Llama3_8BBatch4Activations
v("l8_params_b", m(FULL.workload.parameters, "count") / 10 ** 9, "{:.1f}", "parameters, billions")
v("l8_params_b0", m(FULL.workload.parameters, "count") / 10 ** 9, "{:.0f}", "the same, whole billions")
v("l8_act_gb", m(FULL.activations, "GB"), "{:.1f}", "activations, GB")
v("l8_batch", LA.micro_batch, "{:.0f}", "sequences per step")
v("l8_seq", LA.sequence, "{:,.0f}", "tokens per sequence")
v("l8_lora_m", LORA.adapter_parameters / 10 ** 6, "{:.1f}", "adapter parameters, millions")
v("l8_lora_pct", 100 * LORA.adapter_fraction, "{:.2f}", "adapters, percent of the model")
v("l8_w_gb", m(FULL.weights, "GB"), "{:.0f}", "16-bit weights, GB")
v("l8_g_gb", m(FULL.gradients, "GB"), "{:.0f}", "gradients, GB")
v("l8_master_gb", m(FULL.master_weights, "GB"), "{:.0f}", "master copy, GB")
v("l8_moments_gb", m(FULL.optimizer_state, "GB"), "{:.0f}", "moments, GB")
v("l8_state_gb", m(FULL.total, "GB"), "{:.0f}", "full fine-tuning state, GB")
v("l8_full_gb", m(FULL.training_total, "GB"), "{:.0f}", "full fine-tuning with activations, GB")
v("l8_adapter_gb", m(LORA.adapter_state, "GB"), "{:.2f}", "adapter state, GB")
v("l8_lora_gb", m(LORA.training_total, "GB"), "{:.0f}", "LoRA total, GB")
v("l8_headroom_gb", m(H100.memory.capacity - LORA.training_total, "GB"), "{:.0f}", "LoRA headroom, GB")
v("l8_rest_gb", m(FULL.gradients + FULL.master_weights + FULL.optimizer_state, "GB"), "{:.0f}",
  "gradients, master, and moments, GB (figure)")
v("lora_gpt3_full_tb", m(Q_(TR.LoraGpt3TrainingMemoryFullGB, "GB"), "TB"), "{:.1f}", "GPT-3 full, TB")
v("lora_gpt3_gb", TR.LoraGpt3TrainingMemoryLoraGB, "{:.0f}", "GPT-3 with LoRA, GB")

print("== Published MFU (Literature.TrainingRuns, PaLM Table 3)")
v("mfu_gpt3", 100 * TR.Gpt3MfuPalmTable, "{:.1f}", "GPT-3, percent")
v("mfu_gopher", 100 * TR.GopherMfu, "{:.1f}", "Gopher, percent")
v("gopher_chips", TR.GopherTpuV3Chips, "{:,.0f}", "Gopher's TPU v3 chips")
v("mtnlg_gpus", TR.MtNlgA100Gpus, "{:,.0f}", "Megatron-Turing NLG's A100s")
v("mfu_mtnlg", 100 * TR.MtNlgMfu, "{:.1f}", "Megatron-Turing NLG, percent")
v("mfu_palm", 100 * TR.PalmMfu, "{:.1f}", "PaLM, percent")
v("hfu_palm", 100 * TR.PalmHfu, "{:.1f}", "PaLM HFU, percent")
v("megatron_lo", 100 * TR.MegatronLm2021PeakFractionLow, "{:.0f}", "Megatron-LM, low, percent of peak")
v("megatron_hi", 100 * TR.MegatronLm2021PeakFraction, "{:.0f}", "Megatron-LM, high, percent of peak")

print("== ResNet-50 input feed (Scenarios.ResNet50H100InputFeed)")
v("res_gflop", m(FEED.training_flops_per_sample, "GFLOP"), "{:.1f}", "training FLOPs per image, GFLOP")
v("res_a100", m(FEED_A100.sustained_rate, "count/s"), "{:,.0f}", "A100 demand, images/s")
v("res_h100", round(m(FEED.sustained_rate, "count/s"), -2), "{:,.0f}", "H100 demand, images/s")
v("res_workers", FEED.host_workers, "{:.0f}", "loader workers per accelerator")
v("res_worker_rate", m(FEED.worker_rate, "count/s"), "{:.0f}", "images/s per worker")
v("res_supply", m(FEED.host_supply, "count/s"), "{:,.0f}", "host supply, images/s")
v("res_busy_pct", 100 * FEED.busy_fraction, "{:.0f}", "H100 busy, percent")

print("== The GIL-bound loader (Scenarios.GilThreadedLoaderStall)")
v("gil_prep_ms", m(GIL.io_time, "ms"), "{:.0f}", "host preparation, ms")
v("gil_gpu_ms", m(GIL.compute_time, "ms"), "{:.0f}", "accelerator compute, ms")
v("gil_busy_pct", 100 * (1 - GIL.result.sequential_stall), "{:.1f}", "busy, percent")
v("gil_step_ms", m(GIL.result.sequential_step, "ms"), "{:.0f}", "step, ms")
v("gil_fast_step_ms", m(GIL_FAST.result.sequential_step, "ms"), "{:.0f}", "step with a twice-as-fast GPU, ms")

print("== Checkpoints (Scenarios.Gpt2MixedPrecisionAdamOnH100.checkpoint; Hardware.Tech.Storage.NvmeGen4)")
save = MIXED.checkpoint_write_time(NVME)
v("ckpt_b", m(MIXED.checkpoint, "byte") / p, "{:.0f}", "checkpoint, B per parameter")
v("ckpt_gb", m(MIXED.checkpoint, "GB"), "{:.0f}", "GPT-2 XL checkpoint, GB")
v("nvme_gbs", m(NVME, "GB/s"), "{:.0f}", "local NVMe, GB/s")
v("ckpt_s", m(save, "s"), "{:.0f}", "save time, s")
often, rarely = 100, 10 ** 4          # the two save intervals the exercise compares, in steps
v("ckpt_often", often, "{:,.0f}", "frequent interval, steps")
v("ckpt_rarely", rarely, "{:,.0f}", "rare interval, steps")
v("ckpt_work_s", m(often * RUN.step_time, "s"), "{:.0f}", "work between frequent saves, s")
v("ckpt_stall_pct", 100 * (save / (often * RUN.step_time)).to("").magnitude, "{:.0f}", "stall added, percent")
v("ckpt_rare_s", round(m(rarely * RUN.step_time, "s"), -1), "{:,.0f}", "work between rare saves, s")
v("ckpt_replay_min", m(rarely * RUN.step_time, "minute"), "{:.0f}", "replay after a failure, minutes")

print("== PaLM (Literature.TrainingRuns; Scenarios.Palm540BMixedPrecisionAdam)")
v("palm_params_b", m(PALM.workload.parameters, "count") / 10 ** 9, "{:.0f}", "parameters, billions")
v("palm_chips", TR.Palm540B_Phase1Chips, "{:,.0f}", "TPU v4 chips")
v("palm_pods_word", TR.Palm540B_Pods, "words", "pods")
v("palm_pod_chips", TR.Palm540B_Phase1Chips / TR.Palm540B_Pods, "{:,.0f}", "chips per pod")
v("palm_spikes", TR.PalmLossSpikes, "{:.0f}", "loss spikes")
v("palm_rewind", TR.PalmSpikeRewindSteps, "{:.0f}", "rewind, steps")
v("palm_skip_lo", TR.PalmSpikeSkippedBatchesLow, "{:.0f}", "batches skipped, low")
v("palm_skip_hi", TR.PalmSpikeSkippedBatchesHigh, "{:.0f}", "batches skipped, high")
v("palm_state_tb", m(PALM.total, "TB"), "{:.1f}", "mixed-precision Adam state, TB")
v("palm_h100s", round(PALM.capacity_share, -1), "{:.0f}", "state in H100s of memory, to the ten")
v("palm_ckpt_tb", m(PALM.checkpoint, "TB"), "{:.1f}", "checkpoint, TB")

print("== OPT-175B (Literature.TrainingRuns)")
v("opt_gpus", TR.Opt175BGpus, "{:.0f}", "A100s")
v("opt_manual", TR.Opt175BManualRestarts, "{:.0f}", "manual restarts, at least")
v("opt_auto", TR.Opt175BAutomaticRestarts, "{:.0f}", "automatic restarts, estimated")
v("opt_runs", TR.Opt175BLogbookLastRun - TR.Opt175BLogbookFirstRun + 1, "{:.0f}", "runs in the logbook screenshot")
v("opt_nov28_word", TR.Opt175BLogbookNov28LastRun - TR.Opt175BLogbookNov28FirstRun + 1, "words",
  "runs launched on 2021-11-28")

if __name__ == "__main__":
    write(__file__)
