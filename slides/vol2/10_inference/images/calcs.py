#!/usr/bin/env python3
"""Every number on the Lecture 10 (Serving at Scale) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature series) and the calculations
(Scenarios.*, each backed by a solver); the chapter reads the same scenarios.
This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Models, Scenarios = mlsysim.Hardware, mlsysim.Models, mlsysim.Scenarios
H100, GB200 = Hardware.Cloud.H100, Hardware.Cloud.GB200_NVL72
LLAMA3, LLAMA2, GPT3 = Models.Language.Llama3_70B, Models.Language.Llama2_70B, Models.Language.GPT3
BYTES_FP16, BYTES_INT4 = mlsysim.core.units.BYTES_FP16, mlsysim.core.units.BYTES_INT4

HOOK = Scenarios.LlmServingFirstYear
VOTE = Scenarios.SloGoodputTwoConfigurations
TAX = Scenarios.InteractiveRequestServingTax
KNEE_VIS, KNEE_LLM = Scenarios.VisionBatchingKnee, Scenarios.LlmDecodeBatchingKnee
DECODE = Scenarios.Llama2_70B_DecodeH100  # the reference replica (Llama-2 70B)
LITTLE, CHAT_LITTLE = Scenarios.VisionServingLittlesLaw, Scenarios.LlmChatLittlesLaw
CHAT, CODE = Scenarios.LlmChatTrace2023, Scenarios.LlmCodeTrace2023
WEEK = Scenarios.LlmChatWeekTraffic2024
BATCH = Scenarios.FourRequestDecodeBatch
REASON = Scenarios.ReasoningTokenLatency
LONG, CHATKV = Scenarios.Llama3_70BLongContextAdmissionDgxH100, Scenarios.Llama3_70BChatAdmissionDgxH100
INFLIGHT = Scenarios.LlmChatKvInFlight
PREFIX = Scenarios.ChatbotSharedSystemPrompt
PREC = Scenarios.Llama2_70BPrecisionDividendGqaTp8
CHUNK = Scenarios.LongPromptChunkedPrefill
HANDOFF_CHAT, HANDOFF_LONG = Scenarios.MedianChatKvHandoff, Scenarios.LongContextKvHandoff
TP = Scenarios.Llama2_70BPrefillTensorParallel8
BUDGET, SPEC = Scenarios.Llama3_70B_ReplicaDecodeBudget, Scenarios.Llama3_70B_ReplicaSpeculation
P2C = Scenarios.RandomVsTwoChoicePlacement
HETERO = Scenarios.HeterogeneousH100A100Routing
FANOUT, TAIL = Scenarios.ServingFanout100Leaves, Scenarios.RecommenderTailEvents
COLD = Scenarios.Llama70BReplicaColdStart
NOISY = Scenarios.SharedPoolNoisyNeighbor
IX, COST = Scenarios.InferenceXServingFrontier, Scenarios.H100OutputTokenCost


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


print("== Hook: training once against a year of serving")
v("hook_users_m", HOOK.daily_users / 10 ** 6, "{:.0f}", "daily users, millions")
v("hook_users_words", HOOK.daily_users / 10 ** 6, "words", "daily users, millions, spelled")
v("hook_req_per_user", HOOK.requests_per_user_per_day, "{:.0f}", "requests per user per day")
v("hook_cost_req", HOOK.cost_per_request, "{:.3f}", "cost per request, USD")
v("hook_train_m", HOOK.training_cost / 10 ** 6, "{:.0f}", "training cost, $M")
v("hook_train_m1", HOOK.training_cost / 10 ** 6, "{:.1f}", "training cost, $M, one decimal")
v("hook_daily_m", HOOK.daily_requests / 10 ** 6, "{:.0f}", "requests per day, millions")
v("hook_annual_b", HOOK.annual_requests / 10 ** 9, "{:.2f}", "requests per year, billions")
v("hook_serve_m", HOOK.annual_serving_cost / 10 ** 6, "{:.2f}", "first-year serving, $M")
v("hook_serve_m0", HOOK.annual_serving_cost / 10 ** 6, "{:.0f}", "first-year serving, $M, rounded")
v("hook_ratio", HOOK.serving_to_training, "{:.1f}", "serving / training")
v("hook_gain_pct", HOOK.efficiency_gain * 100, "{:.0f}", "cut in cost per request, percent")
v("hook_savings_m", HOOK.efficiency_savings / 10 ** 6, "{:.1f}", "first-year savings of that cut, $M")

print("== The reference replica and the measured traffic")
LR, CR = LONG.result, CHATKV.result
chat, code = CHAT.result, CODE.result
v("node_gpus", LONG.accelerators, "{:.0f}", "GPUs per replica")
v("node_gpus_words", LONG.accelerators, "words", "GPUs per replica, spelled")
v("h100_gib", m(H100.memory.capacity, "GiB"), "{:.0f}", "H100 HBM, GiB")
v("h100_gb", m(H100.memory.capacity, "GB"), "{:.1f}", "H100 HBM, decimal GB")
v("h100_bw", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM bandwidth, TB/s")
v("h100_fp16", m(H100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "H100 dense FP16, TFLOP/s")
v("llama3_layers", LLAMA3.layers, "{:.0f}", "layers")
v("llama3_heads", LLAMA3.heads, "{:.0f}", "query heads")
v("llama3_kv_heads", LLAMA3.kv_heads, "{:.0f}", "key-value heads (GQA)")
v("llama3_hidden", LLAMA3.hidden_dim, "{:.0f}", "hidden size")
v("llama3_head_dim", LR.head_dim, "{:.0f}", "head width")
v("llama3_context", LLAMA3.context_length, "{:,.0f}", "context window, tokens")
v("gqa_ratio", LLAMA3.heads / LLAMA3.kv_heads, "{:.0f}", "query heads per KV head")
v("replica_weights_gb", m(LLAMA2.size_in_bytes(BYTES_FP16), "GB"), "{:.0f}", "reference replica (Llama-2 70B) FP16 weights, GB")
v("lc_weights_gb", m(LR.weights, "GB"), "{:.1f}", "128K-context 70B model (Llama-3.1 70B) FP16 weights, GB")
v("replica_weights_per_gpu", m(LLAMA2.size_in_bytes(BYTES_FP16), "GB") / LONG.accelerators, "{:.1f}", "reference replica FP16 weights per GPU, GB")
v("hbm_total_gb", m(LR.hbm_total, "GB"), "{:.1f}", "replica HBM, decimal GB")
v("hbm_trap_gb", LONG.accelerators * m(H100.memory.capacity, "GiB"), "{:.0f}", "8 x 80 read as decimal, GB")
v("min_shards", math.ceil(ratio(LR.weights, H100.memory.capacity)), "{:.0f}", "GPUs the FP16 weights need")
v("chat_requests", chat.requests, "{:,.0f}", "chat trace requests")
v("chat_rate", m(chat.arrival_rate, "1/s"), "{:.1f}", "chat trace arrivals per second")
v("chat_med_prompt", chat.median_prompt, "{:,.0f}", "chat median prompt, tokens")
v("chat_med_output", chat.median_output, "{:.0f}", "chat median answer, tokens")
v("chat_mean_output", chat.mean_output, "{:.0f}", "chat mean answer, tokens")
v("chat_max_output", chat.max_output, "{:,.0f}", "chat longest answer, tokens")
v("chat_row0_prompt", CHAT.prompt_tokens[0], "{:.0f}", "first trace row: prompt tokens")
v("chat_row0_output", CHAT.output_tokens[0], "{:.0f}", "first trace row: generated tokens")

print("== Contract: SLO goodput and the serving tax")
v("vote_x_rate", m(VOTE.completed["X"], "1/s"), "{:,.0f}", "X completed per second")
v("vote_y_rate", m(VOTE.completed["Y"], "1/s"), "{:,.0f}", "Y completed per second")
v("vote_x_pct", VOTE.met_share["X"] * 100, "{:.0f}", "X share meeting targets, percent")
v("vote_y_pct", VOTE.met_share["Y"] * 100, "{:.0f}", "Y share meeting targets, percent")
v("vote_x_share", VOTE.met_share["X"], "{:.2f}", "X share meeting targets")
v("vote_y_share", VOTE.met_share["Y"], "{:.2f}", "Y share meeting targets")
v("vote_x_goodput", m(VOTE.goodput("X"), "1/s"), "{:,.0f}", "X SLO goodput")
v("vote_y_goodput", m(VOTE.goodput("Y"), "1/s"), "{:,.0f}", "Y SLO goodput")
v("tax_target_ms", m(TAX.latency_target, "ms"), "{:.0f}", "request latency target, ms")
v("tax_model_ms", m(TAX.model_time, "ms"), "{:.0f}", "model execution, ms")
v("tax_outside_ms", m(TAX.outside_model, "ms"), "{:.0f}", "budget outside the model, ms")
v("tax_exchanges", TAX.exchanges, "{:.0f}", "sequential exchanges")
v("tax_exchanges_words", TAX.exchanges, "words", "sequential exchanges, spelled")
v("tax_exch_low_us", m(TAX.exchange_low, "microsecond"), "{:.0f}", "one exchange, low, us")
v("tax_exch_high_us", m(TAX.exchange_high, "microsecond"), "{:.0f}", "one exchange, high, us")
v("tax_exchanges_low_ms", m(TAX.exchanges_low, "ms"), "{:.1f}", "all exchanges, low, ms")
v("tax_exchanges_high_ms", m(TAX.exchanges_high, "ms"), "{:.0f}", "all exchanges, high, ms")

print("== InferenceX: the measured frontier")
b200 = IX.solve("b200", COST.targets[2])
sweep = IX.sweep("b200", framework="sglang")
lo, hi = sweep[0], sweep[-1]
v("ix_isl", lo["isl"], "{:,.0f}", "prompt and answer tokens")
v("ix_isl_raw", hi["isl"], "{:.0f}", "raw row: prompt tokens")
v("ix_osl_raw", hi["osl"], "{:.0f}", "raw row: answer tokens")
v("b200_peak", b200.peak_throughput, "{:,.0f}", "B200 top tok/s per GPU")
v("b200_peak_user", b200.peak_interactivity, "{:.0f}", "at tok/s per user")
v("b200_at_high", b200.best_at, "{:,.0f}", "B200 best at >= 100 tok/s per user")
v("b200_span", b200.peak_throughput / b200.best_at, "{:.0f}", "B200 frontier span")
v("tps_low", COST.targets[0], "{:.0f}", "per-user target, low")
v("tps_mid", COST.targets[1], "{:.0f}", "per-user target, middle")
v("tps_high", COST.targets[2], "{:.0f}", "per-user target, high")
h100_50, gb200_50 = IX.best_at("h100", COST.targets[1]), IX.best_at("gb200", COST.targets[1])
v("h100_at_mid", h100_50, "{:.0f}", "H100 best at 50 tok/s per user")
v("gb200_at_mid", gb200_50, "{:,.0f}", "GB200 NVL72 best at 50 tok/s per user")
v("gb200_over_h100", gb200_50 / h100_50, "{:.0f}", "GB200 / H100 at 50")
v("gb200_peak_fp8_ratio",
  ratio(GB200.compute.precision_flops["fp8"] / GB200.accelerator_count, H100.compute.precision_flops["fp8"]),
  "{:.1f}", "GB200 / H100 peak FP8 per GPU")
v("conc_lo", lo["conc"], "{:.0f}", "lowest concurrency")
v("conc_hi", hi["conc"], "{:.0f}", "highest concurrency")
v("conc_hi_user", hi["interactivity_tok_s_user"], "{:.2f}", "raw row: tok/s per user at the highest concurrency")
v("conc_hi_gpu_raw", hi["output_tok_s_gpu"], "{:.2f}", "raw row: tok/s per GPU at the highest concurrency")
v("conc_hi_gpu_raw_c", hi["output_tok_s_gpu"], "{:,.2f}", "same, with a thousands comma")
v("conc_lo_gpu", lo["output_tok_s_gpu"], "{:.0f}", "tok/s per GPU at concurrency 1")
v("conc_hi_gpu", hi["output_tok_s_gpu"], "{:,.0f}", "tok/s per GPU at the highest concurrency")
v("conc_gpu_gain", hi["output_tok_s_gpu"] / lo["output_tok_s_gpu"], "{:.0f}", "per-GPU gain")
v("conc_lo_user", lo["interactivity_tok_s_user"], "{:.0f}", "tok/s per user at concurrency 1")
v("conc_hi_user0", hi["interactivity_tok_s_user"], "{:.0f}", "tok/s per user at the highest concurrency")
v("conc_user_drop", lo["interactivity_tok_s_user"] / hi["interactivity_tok_s_user"], "{:.1f}", "per-user slowdown")
v("conc_max_at_mid", IX.max_concurrency_at("b200", COST.targets[1], framework="sglang"), "{:.0f}",
  "highest concurrency still >= 50 tok/s per user")

print("== Batching: knee, Little's law, fixed batches, reasoning")
v("knee_vis_t0", m(KNEE_VIS.fixed_time, "ms"), "{:.0f}", "vision fixed time, ms")
v("knee_vis_treq", m(KNEE_VIS.per_request_time, "ms"), "{:.0f}", "vision per-image time, ms")
v("knee_vis", KNEE_VIS.knee, "{:.0f}", "vision knee")
v("knee_llm_t0", m(KNEE_LLM.fixed_time, "ms"), "{:.0f}", "decode fixed time, ms")
v("knee_llm_treq", m(KNEE_LLM.per_request_time, "ms"), "{:.1f}", "decode per-sequence time, ms")
v("knee_llm", KNEE_LLM.knee, "{:.0f}", "decode knee")
v("decode_weight_ms", m(DECODE.result.memory_time, "ms"), "{:.1f}", "replica FP16 weights through one H100, ms")
v("little_rate", m(LITTLE.arrival_rate, "1/s"), "{:,.0f}", "arrivals per second")
v("little_latency_ms", m(LITTLE.time_in_system, "ms"), "{:.0f}", "vision mean latency, ms")
v("little_latency_s", m(LITTLE.time_in_system, "s"), "{:.1f}", "vision mean latency, s")
v("little_batch", LITTLE.batch_size, "{:.0f}", "batch")
v("little_service_ms", m(LITTLE.batch_service_time, "ms"), "{:.0f}", "batch service time, ms")
v("little_service_s", m(LITTLE.batch_service_time, "s"), "{:.2f}", "batch service time, s")
v("little_util_pct", LITTLE.utilization_target * 100, "{:.0f}", "utilization target, percent")
v("little_util", LITTLE.utilization_target, "{:.1f}", "utilization target")
v("little_inflight", LITTLE.in_flight, "{:.0f}", "vision requests in flight")
v("little_replica_rate", m(LITTLE.replica_throughput, "1/s"), "{:.0f}", "one replica, requests per second")
v("little_replicas", LITTLE.replicas_needed, "{:.0f}", "replicas at the utilization target")
v("little_floor", LITTLE.throughput_floor, "{:.0f}", "replicas at full utilization")
v("tpot_ms", m(BUDGET.tpot_target, "ms"), "{:.0f}", "TPOT target, ms")
v("chat_time_s", m(CHAT_LITTLE.time_in_system, "s"), "{:.2f}", "chat time in system, s")
v("chat_inflight", CHAT_LITTLE.in_flight, "{:,.0f}", "chat requests in flight")
v("fixed_batch", chat.batch_size, "{:.0f}", "fixed batch of consecutive requests")
v("fixed_batch_words", chat.batch_size, "words", "fixed batch, spelled")
v("chat_idle_pct", chat.fixed_batch_idle_share * 100, "{:.0f}", "chat idle share, percent")
v("code_idle_pct", code.fixed_batch_idle_share * 100, "{:.0f}", "code idle share, percent")
v("cb_idle_pct", BATCH.idle_share * 100, "{:.1f}", "four-request idle share, percent")
for i, n in enumerate(BATCH.output_lengths):
    v(f"cb_out{i + 1}", n, "{:.0f}", f"request {i + 1} answer, tokens")
v("cb_iter_ms", m(BATCH.iteration_time, "ms"), "{:.0f}", "decode iteration, ms")
v("cb_max", BATCH.max_output, "{:.0f}", "longest answer, iterations")
v("cb_idle_iters", BATCH.slot_iterations - BATCH.useful_iterations, "{:.0f}", "idle slot-iterations")
v("cb_slot_iters", BATCH.slot_iterations, "{:.0f}", "slot-iterations")
v("cb_n_words", len(BATCH.output_lengths), "words", "requests in the batch, spelled")
v("cb_advanced", BATCH.requests_advanced, "{:.0f}", "requests advanced with refills")
v("cb_advanced_words", BATCH.requests_advanced, "words", "requests advanced with refills, spelled")
v("reason_tokens", REASON.reasoning_tokens, "{:.0f}", "serial reasoning tokens")
v("reason_tok_ms", m(REASON.per_token_time, "ms"), "{:.0f}", "time per token, ms")
v("reason_time_s", m(REASON.reasoning_time, "s"), "{:.1f}", "reasoning time, s")

print("== KV memory: admission, sharing, precision")
v("kv_bytes_elem", m(LONG.kv_bytes_per_elem, "byte"), "{:.0f}", "bytes per cached element")
v("kv_bytes_token", m(LR.kv_per_token, "byte"), "{:,.0f}", "bytes per token")
v("kv_mb_token", m(LR.kv_per_token, "MB"), "{:.2f}", "MB per token")
v("kv_reserve_gb", m(LONG.reserve, "GB"), "{:.0f}", "system reserve, GB")
v("kv_budget_gb", m(LR.kv_budget, "GB"), "{:.1f}", "free for KV, GB")
v("kv_budget_gb0", m(LR.kv_budget, "GB"), "{:.0f}", "free for KV, GB, rounded")
v("kv_long_gb", m(LR.kv_per_request, "GB"), "{:.1f}", "one 128K request, GB")
v("kv_long_n", LR.max_requests, "{:.0f}", "128K requests that fit")
v("chat_tokens", CHATKV.context_tokens, "{:,.0f}", "median chat request, tokens")
v("kv_chat_mb", m(CR.kv_per_request, "MB"), "{:.0f}", "median chat KV, MB")
v("kv_chat_n", CR.max_requests, "{:,.0f}", "median chats that fit")
v("kv_chat_over_long", CR.max_requests / LR.max_requests, "{:.0f}", "chats per long request")
v("kv_inflight_tb", m(INFLIGHT.kv_in_flight, "TB"), "{:.1f}", "chat KV in flight, TB")
v("kv_inflight_replicas", INFLIGHT.replicas_by_kv, "{:.1f}", "replicas of KV memory")
v("prefix_tokens", PREFIX.prefix_tokens, "{:,.0f}", "shared system prompt, tokens")
v("prefix_users", PREFIX.users, "{:,.0f}", "users")
v("prefix_heads", PREFIX.workload.heads, "{:.0f}", "attention heads cached (MHA)")
v("prefix_before_tb", m(PREFIX.kv_unshared, "TB"), "{:.1f}", "KV without sharing, TB")
v("prefix_after_tb", m(PREFIX.kv_shared, "TB"), "{:.1f}", "KV with sharing, TB")
v("prefix_reduction_pct", PREFIX.reduction * 100, "{:.1f}", "reduction, percent")
PR = PREC.result
v("w70_params_b", m(LLAMA2.parameters, "Gparam"), "{:.0f}", "70B parameters, billions")
v("w70_fp16_gb", m(LLAMA2.size_in_bytes(BYTES_FP16), "GB"), "{:.0f}", "70B FP16 weights, GB")
v("w70_int4_gb", m(LLAMA2.size_in_bytes(BYTES_INT4), "GB"), "{:.0f}", "70B INT4 weights, GB")
v("w175_int4_gb", m(GPT3.size_in_bytes(BYTES_INT4), "GB"), "{:.1f}", "175B INT4 weights, GB")
v("prec_hbm_gb", m(PREC.hbm_per_accelerator, "GB"), "{:.0f}", "decimal per-GPU budget, GB")
v("prec_tp", PREC.tp_size, "{:.0f}", "GPUs sharing the weights and KV")
v("prec_batch", PR.max_batch, "{:.0f}", "max batch, FP16")
v("prec_batch_low", PR.max_batch_low, "{:.0f}", "max batch, INT4 weights + INT8 KV")
v("prec_dividend", PR.dividend, "{:.1f}", "precision dividend")
v("prec_budget_gain", PR.budget_gain, "{:.2f}", "KV budget gain from INT4 weights")

print("== Phases and shards")
v("ridge", m(H100.ridge_point(), "flop/byte"), "{:.0f}", "H100 FP16 ridge, FLOP/byte")
v("decode_intensity", m(DECODE.result.intensity, "flop/byte"), "{:.0f}", "batch-one decode, FLOP per weight byte")
v("chunk_prompt", CHUNK.prompt_tokens, "{:,.0f}", "long prompt, tokens")
v("chunk_size", CHUNK.chunk_tokens, "{:.0f}", "chunk, tokens")
v("chunk_iters", CHUNK.iterations, "{:.0f}", "iterations, at least")
v("nvlink_gbs", m(HANDOFF_CHAT.links["NVLink"], "GB/s"), "{:.0f}", "NVLink per direction, GB/s")
v("ib_gbs", m(HANDOFF_CHAT.links["InfiniBand"], "GB/s"), "{:.0f}", "InfiniBand NDR, GB/s")
v("handoff_chat_nvl_ms", m(HANDOFF_CHAT.result.times["NVLink"], "ms"), "{:.1f}", "median chat KV over NVLink, ms")
v("handoff_chat_ib_ms", m(HANDOFF_CHAT.result.times["InfiniBand"], "ms"), "{:.1f}", "median chat KV over IB, ms")
v("handoff_long_nvl_ms", m(HANDOFF_LONG.result.times["NVLink"], "ms"), "{:.0f}", "128K KV over NVLink, ms")
gains = [IX.split_gain(h, t) for h in ("b200", "h200") for t in (IX.targets[0], IX.targets[1])]
v("disagg_gain_lo", min(gains), "{:.1f}", "split over colocated at 25-50 tok/s per user, low")
v("disagg_gain_hi", max(gains), "{:.1f}", "split over colocated at 25-50 tok/s per user, high")
v("disagg_t1", IX.targets[0], "{:.0f}", "per-user rate compared, low")
v("disagg_t2", IX.targets[1], "{:.0f}", "per-user rate compared, middle")
v("disagg_t3", IX.targets[2], "{:.0f}", "per-user rate compared, high")
for h in ("b200", "h200"):
    v(f"{h}_split_high", IX.best_at(h, IX.targets[2], disagg=1.0), "{:.0f}", f"{h} split at 100")
    v(f"{h}_coloc_high", IX.best_at(h, IX.targets[2], disagg=0.0), "{:.0f}", f"{h} colocated at 100")
v("tp_size", TP.tp_size, "{:.0f}", "tensor-parallel GPUs")
v("tp_layers", TP.layers, "{:.0f}", "layers")
v("tp_layer_ms", m(TP.serial_layer_time, "ms"), "{:.0f}", "serial layer, ms")
v("tp_attn_ms", m(TP.serial_layer_times["attention"], "ms"), "{:.0f}", "attention, ms")
v("tp_ffn_ms", m(TP.serial_layer_times["feed-forward"], "ms"), "{:.0f}", "feed-forward, ms")
v("tp_attn_shard_ms", m(TP.sharded("attention"), "ms"), "{:.1f}", "attention per GPU, ms")
v("tp_ffn_shard_ms", m(TP.sharded("feed-forward"), "ms"), "{:.2f}", "feed-forward per GPU, ms")
v("tp_compute_ms", m(TP.sharded_compute_time, "ms"), "{:.2f}", "compute per layer on 8, ms")
v("tp_ar_ms", m(TP.allreduce_time, "ms"), "{:.1f}", "one AllReduce, ms")
v("tp_layer_par_ms", m(TP.parallel_layer_time, "ms"), "{:.2f}", "layer with collectives, ms")
v("tp_serial_total", m(TP.serial_total, "ms"), "{:,.0f}", "80 layers on 1 GPU, ms")
v("tp_parallel_total", m(TP.parallel_total, "ms"), "{:.0f}", "80 layers on 8 GPUs, ms")
v("tp_speedup", TP.speedup, "{:.1f}", "speedup")
base, prec_col, fused = BUDGET.baseline, BUDGET.after_precision, BUDGET.after_fusion
v("budget_base", BUDGET.totals[0], "{:.0f}", "baseline step, ms")
v("budget_base1", BUDGET.totals[0], "{:.1f}", "baseline step, ms, one decimal")
v("budget_prec", BUDGET.totals[1], "{:.1f}", "after INT4 weights, ms")
v("budget_fused", BUDGET.totals[2], "{:.1f}", "after fusion and replay, ms")
v("budget_floor", m(BUDGET.floor, "ms"), "{:.1f}", "bandwidth floor, ms")
v("budget_over_floor", BUDGET.totals[0] / m(BUDGET.floor, "ms"), "{:.0f}", "baseline over floor")
v("gemm_cut_pct", BUDGET.gemm_cut * 100, "{:.0f}", "GEMM cut from INT4, percent")
v("budget_nongemm", sum(base[1:]), "{:.0f}", "non-GEMM intervals, ms")
v("budget_allreduce", base[3], "{:.1f}", "AllReduce interval, ms")
v("allreduces_step", BUDGET.allreduces_per_step, "{:.0f}", "AllReduces per decode step")
v("ar_per_collective_us", m(BUDGET.allreduce_per_collective, "microsecond"), "{:.0f}", "per collective, us")
v("spec_k", SPEC.draft_length, "{:.0f}", "draft tokens")
v("spec_draft_ms", m(SPEC.draft_time, "ms"), "{:.0f}", "draft time, ms")
v("spec_acc", SPEC.acceptance, "{:.2f}", "acceptance")
v("spec_tokens", SPEC.expected_tokens, "{:.2f}", "tokens per round")
v("spec_round", m(SPEC.round_time, "ms"), "{:.1f}", "round, ms")
v("spec_per_token", m(SPEC.per_token_time, "ms"), "{:.1f}", "per token, ms")
v("spec_gain", SPEC.tpot_gain, "{:.2f}", "TPOT gain")

print("== Fleet: routing, tail, traffic, tenants")
p2c = P2C.result
v("p2c_n", p2c.bins[-1], "{:,.0f}", "replicas in the largest run")
v("p2c_random", p2c.max_load[1][-1], "{:.1f}", "busiest replica, random")
v("p2c_two", p2c.max_load[2][-1], "{:.1f}", "busiest replica, two probes")
v("hetero_n_h100", HETERO.servers["H100"], "{:.0f}", "H100 servers")
v("hetero_n_a100", HETERO.servers["A100"], "{:.0f}", "A100 servers")
v("hetero_h100_cap", HETERO.capacity["H100"], "{:,.0f}", "H100 server capacity, req/s")
v("hetero_a100_cap", HETERO.capacity["A100"], "{:.0f}", "A100 server capacity, req/s")
v("hetero_offered", HETERO.offered, "{:,.0f}", "offered req/s")
v("hetero_util_pct", HETERO.utilization("H100") * 100, "{:.1f}", "balanced utilization, percent")
v("fanout_leaves", FANOUT.n_components, "{:.0f}", "leaves per request")
v("fanout_avail", FANOUT.availability, "{:.2f}", "leaf fast share")
v("fanout_slow_pct", (1 - FANOUT.all_up) * 100, "{:.0f}", "slow requests, percent")
v("fanout_leaf_budget_pct", (1 - FANOUT.required_availability(FANOUT.availability)) * 100, "{:.2f}",
  "leaf slow budget, percent")
v("fanout_strict_slow_pct",
  (1 - FANOUT.variant(availability=FANOUT.required_availability(FANOUT.availability)).all_up) * 100,
  "{:.0f}", "slow requests with every leaf inside that budget, percent")
v("tail_qps_exp", math.log10(m(TAIL.request_rate, "1/s")), "{:.0f}", "platform QPS, power of ten")
v("tail_qps_m", m(TAIL.request_rate, "1/s") / 10 ** 6, "{:.0f}", "platform QPS, millions")
v("tail_prob_exp", math.log10(TAIL.event_probability), "{:.0f}", "event probability, power of ten")
v("tail_one_in", 1 / TAIL.event_probability, "{:,.0f}", "one request in")
v("tail_events", TAIL.events_per_second, "{:,.0f}", "slow events per second")
wk = WEEK.result
v("week_peak", m(wk.peak_rate, "1/s"), "{:.0f}", "week peak, req/s")
v("week_trough_lo", m(wk.weekday_trough_low, "1/s"), "{:.0f}", "weekday trough, low, req/s")
v("week_trough_hi", m(wk.weekday_trough_high, "1/s"), "{:.0f}", "weekday trough, high, req/s")
v("week_peak_lo", m(wk.weekday_peak_low, "1/s"), "{:.0f}", "weekday peak, low, req/s")
v("week_peak_hi", m(wk.weekday_peak_high, "1/s"), "{:.0f}", "weekday peak, high, req/s")
v("week_swing", wk.weekday_peak_to_trough, "{:.1f}", "weekday peak / trough")
v("week_swing0", wk.weekday_peak_to_trough, "{:.0f}", "weekday peak / trough, rounded")
v("code_per_s", code.per_second_mean, "{:.1f}", "code trace, mean req per second")
v("code_peak_s", code.per_second_peak, "{:.0f}", "code trace, busiest second")
v("cs_provision_min", m(COLD.provisioning_time, "minute"), "{:.0f}", "provision, min")
v("cs_artifact_gb", m(COLD.artifact, "GB"), "{:.0f}", "FP16 weight artifact, GB")
v("cs_fetch_gbs", m(COLD.fetch_bandwidth, "GB/s"), "{:.1f}", "fetch bandwidth, GB/s")
v("cs_fetch_min", m(COLD.fetch_time, "minute"), "{:.1f}", "fetch, min")
v("cs_warmup_min", m(COLD.load_warmup_time, "minute"), "{:.0f}", "load and warmup, min")
v("cs_total_min", m(COLD.cold_start_time, "minute"), "{:.1f}", "cold start, min")
v("cs_lead_min", m(COLD.prewarm_lead, "minute"), "{:.0f}", "pre-warm lead, min")
v("cs_reactive", COLD.reactive_peak_replicas, "{:.0f}", "peak replicas, reactive")
v("cs_predictive", COLD.predictive_peak_replicas, "{:.0f}", "peak replicas, predictive")
v("nn_base_qps", NOISY.base_rate, "{:.0f}", "tenant QPS")
v("nn_burst_qps", NOISY.burst_rate, "{:.0f}", "bursting tenant QPS")
v("nn_burst_mult", NOISY.burst_multiple, "{:.0f}", "burst multiple")
v("nn_slo_ms", m(NOISY.slo, "ms"), "{:.0f}", "p99 SLO, ms")
v("nn_p99_lo", m(NOISY.unprotected_p99_low, "ms"), "{:.0f}", "unprotected p99, low, ms")
v("nn_p99_hi", m(NOISY.unprotected_p99_high, "ms"), "{:.0f}", "unprotected p99, high, ms")
v("nn_quota", NOISY.quota_rate, "{:.0f}", "quota, QPS")
v("nn_protected_ms", m(NOISY.protected_p99, "ms"), "{:.0f}", "protected p99, ms")
v("nn_pool_gb", m(NOISY.kv_pool, "GB"), "{:.0f}", "shared KV pool, GB")
v("nn_swollen_gb", m(NOISY.swollen_share, "GB"), "{:.0f}", "swollen tenant share, GB")
v("nn_base_seqs", NOISY.base_sequences, "{:.0f}", "sequences per tenant, equal shares")
v("nn_other_seqs", NOISY.other_sequences, "{:.0f}", "sequences per other tenant, swollen")

print("== Synthesis: what one request costs")
v("syn_price", m(COST.gpu_hour_price, "USD/hour"), "{:.2f}", "hypothetical H100 price, $/hour")
for i, t in enumerate(COST.targets):
    v(f"syn_tok{i + 1}", COST.throughput(t), "{:.0f}", f"H100 best tok/s per GPU at {t}")
    v(f"syn_cost{i + 1}", m(COST.cost_per_million(t), "USD"), "{:.2f}", f"$ per million output tokens at {t}")
v("syn_spread", COST.spread, "{:.0f}", "cost spread, 100 over 13")
v("syn_answer", m(COST.answer_cost(round(chat.mean_output), COST.targets[1]), "USD"), "{:.3f}",
  "mean chat answer at 50 tok/s per user, $")

if __name__ == "__main__":
    write(__file__)
