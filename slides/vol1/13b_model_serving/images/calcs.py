#!/usr/bin/env python3
"""Every number on the Lecture 13b (Model Serving, Part 2) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the token-serving Scenarios.*
(GPT-2 XL decode batches with cache reads, speculative decoding, the Llama 3
8B case-study service, the prefill stall, the KV budgets, the request-level
and continuous schedules, prefix sharing, vLLM's KV pool, DistServe's pools,
energy per token, and Little's law on the MLPerf server), the cited results in
Literature.ServingResults, and the MLPerf Inference v5.0 Llama 2 70B rows,
the vLLM benchmark runs, the PagedAttention memory breakdown, and the API
price series in local CSV series. The Model Serving chapter reads the same
scenarios. This file only selects values, converts units, and formats them;
it types no numbers. Each value is recorded with v() under the key the deck
prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports this module, so figures and slides share one computation.

The two vLLM printouts (the serving benchmark and the startup log) are quoted
on the slides as vLLM printed them; Literature holds their values for the
text around them.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, val, num, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

from mlsysim.core.units import Q_, BYTES_FP16, BYTES_INT4  # noqa: E402
from mlsysim.physics import calc_littles_law_rate  # noqa: E402

Scenarios, Literature, Models, Hardware = mlsysim.Scenarios, mlsysim.Literature, mlsysim.Models, mlsysim.Hardware
SR = Literature.ServingResults

H100 = Hardware.Cloud.H100
CHAT = Scenarios.Gpt2XlChatDecodeBatchesH100
SPEC = Scenarios.Gpt2XlSpeculativeGpt2SmallH100
EAGER = Scenarios.Gpt2XlEagerDecodeDispatch
DECODE1 = Scenarios.Gpt2XlFp16DecodeOnH100
CASE = Scenarios.Llama3_8BInt4ChatServiceH100
TARGETS = Scenarios.InteractiveChatTargetsLlama3_8B
STALL = Scenarios.Llama3_8BInt4PrefillStallH100
CONV8 = Scenarios.Llama3_8BFp16ConversationsH100
KV70 = Scenarios.Llama3_70BInt4KvBudgetH100
SCHED = Scenarios.RequestLevelVersusContinuousBatching
PREFIX = Scenarios.Llama3_8BSharedSystemPrompt
RELOAD = Scenarios.Llama3_8BConversationReloadPcieGen4
POOL = Scenarios.VllmKvPoolLlama3_8BH100
DS = Scenarios.DistServe13BPoolsA100
ENERGY = Scenarios.H100ChatEnergyPerToken
STREAMS200 = Scenarios.Llama2_70BConversationalDecodeStreams
STREAMS40 = Scenarios.Llama2_70BInteractiveDecodeStreams
SYN16 = Scenarios.Llama2_70BFp16ConversationsEightH100
SYN8 = Scenarios.Llama2_70BInt8ConversationsEightH100
COST = Scenarios.ResNet50BatchOneVersusEightCostH100     # Lecture 13's hook
PLAN = Scenarios.ResNet50FleetPlanH100                   # Lecture 13's fleet plan

B1, B8, B32, B64, B128 = CHAT.batch_sizes
GB = Q_(1, "GB")


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


print("== Hook (slides/vol1/13_model_serving/images/data/mlperf_inference_v5_0_llama2_70b.csv, 5.0-0020)")
LL = load_series("slides/vol1/13_model_serving/images/data/mlperf_inference_v5_0_llama2_70b.csv")
LL_OFF = row(LL, benchmark="llama2-70b-99", scenario="Offline")
LL_CONV = row(LL, benchmark="llama2-70b-99", scenario="Server")
LL_INT = row(LL, benchmark="llama2-70b-interactive-99", scenario="Server")
LL_INT_OFF = row(LL, benchmark="llama2-70b-interactive-99", scenario="Offline")
MLP_OFFLINE, MLP_SERVER, MLP_INTER = LL_OFF["tokens_per_s"], LL_CONV["tokens_per_s"], LL_INT["tokens_per_s"]
v("llama_off", MLP_OFFLINE, "{:,.0f}", "Offline tokens/s")
v("llama_conv", MLP_SERVER, "{:,.0f}", "conversational Server tokens/s")
v("llama_int", MLP_INTER, "{:,.0f}", "interactive Server tokens/s")
v("llama_int_off", LL_INT_OFF["tokens_per_s"], "{:,.0f}", "interactive entry's own Offline tokens/s")
v("llama_conv_ttft_s", LL_CONV["ttft_bound_ms"] / 10 ** 3, "{:.0f}", "conversational TTFT bound, s")
v("llama_conv_tpot", LL_CONV["tpot_bound_ms"], "{:.0f}", "conversational TPOT bound, ms")
v("llama_int_ttft", LL_INT["ttft_bound_ms"], "{:.0f}", "interactive TTFT bound, ms")
v("llama_int_tpot", LL_INT["tpot_bound_ms"], "{:.0f}", "interactive TPOT bound, ms")
v("llama_conv_pct", pct(MLP_SERVER / MLP_OFFLINE), "{:.0f}", "conversational kept of Offline, %")
v("llama_int_pct", pct(MLP_INTER / MLP_SERVER), "{:.0f}", "interactive over conversational, %")
v("hook_answer", round(num("llama_int_pct"), -1), "{:.0f}", "the answer choice, to the ten, %")
v("tpot_tighten", LL_CONV["tpot_bound_ms"] / LL_INT["tpot_bound_ms"], "{:g}", "TPOT tightening, times")
v("hook_fewer", MLP_SERVER / MLP_INTER, "{:.1f}", "times fewer tokens/s under 40 ms")
v("mlperf_gpus", LL_CONV["accelerators"], "{:.0f}", "H100s in the server")

print("== Recall (Lecture 13: Scenarios.ResNet50BatchOneVersusEightCostH100, ResNet50FleetPlanH100)")
v("recall_b1_usd", m(COST.cost_per_million(1), "dollar"), "{:.2f}", "batch 1, $ per million requests")
v("recall_b8_usd", m(COST.cost_per_million(COST.batch_sizes[1]), "dollar"), "{:.2f}", "batch 8, $ per million requests")
v("recall_ratio", COST.cost_ratio, "{:.1f}", "batch 1 over batch 8 cost")
v("recall_price", m(COST.price, "dollar/hour"), "{:.0f}", "GPU-hour price of Example 13.1, $")
v("recall_rho_pct", pct(PLAN.max_utilization), "{:.0f}", "Part 1's highest safe utilization, %")
v("recall_budget_ms", m(PLAN.tail_budget, "ms"), "{:.0f}", "Part 1's p99 budget, ms")
v("l3_kv_tok", m(CONV8.result.kv_per_token, "byte"), "{:,.0f}", "Llama 3 8B cache per token, bytes")
v("l3_kv_kb", m(CONV8.result.kv_per_token, "kB"), "{:.0f}", "the same, KB")

print("== Running models (Models.Language.GPT2)")
v("gpt2_params_b", m(CHAT.workload.parameters, "count") / 10 ** 9, "{:.1f}", "GPT-2 XL parameters, billions")
v("gpt2_w_gb", m(CHAT.weights, "GB"), "{:.0f}", "GPT-2 XL FP16 weights, GB")
v("gpt2_w_gb1", m(CHAT.weights, "GB"), "{:.1f}", "the same, one decimal")

print("== Two deadlines (Literature.ServingResults, Splitwise Table IV; InteractiveChatTargetsLlama3_8B)")
H_TTFT, H_TBT, H_E2E = (lit(SR.SplitwiseLlama70bH100TtftMs), lit(SR.SplitwiseLlama70bH100TbtMs),
                        lit(SR.SplitwiseLlama70bH100E2eMs))
A_TTFT, A_TBT, A_E2E = (lit(SR.SplitwiseLlama70bA100TtftMs), lit(SR.SplitwiseLlama70bA100TbtMs),
                        lit(SR.SplitwiseLlama70bA100E2eMs))
DECODE_SHARE = (H_E2E - H_TTFT) / H_E2E
v("sw_h_ttft", H_TTFT, "{:.0f}", "DGX-H100 TTFT, ms")
v("sw_h_tbt", H_TBT, "{:.0f}", "DGX-H100 TBT, ms")
v("sw_h_e2e", H_E2E, "{:,.0f}", "DGX-H100 E2E, ms")
v("sw_a_ttft", A_TTFT, "{:.0f}", "DGX-A100 TTFT, ms")
v("sw_a_tbt", A_TBT, "{:.0f}", "DGX-A100 TBT, ms")
v("sw_a_e2e", A_E2E, "{:,.0f}", "DGX-A100 E2E, ms")
v("sw_decode_pct", pct(DECODE_SHARE), "{:.0f}", "decode share of the request, %")
v("sw_tokens", round((H_E2E - H_TTFT) / H_TBT + 1, -1), "{:.0f}", "answer tokens, about, to the ten")
v("target_ttft", m(TARGETS.ttft_target, "ms"), "{:.0f}", "illustrative TTFT target, ms")
v("target_tpot", m(TARGETS.tpot_target, "ms"), "{:.0f}", "illustrative TPOT target, ms")
v("target_rate", m(1 / TARGETS.tpot_target, "1/s"), "{:.0f}", "the TPOT target as tokens/s")

print("== Prefill and decode (Gpt2XlChatDecodeBatchesH100, Llama3_8BInt4ChatServiceH100)")
GPT2_PREFILL_MS = m(CHAT.prefill_time, "ms")
PRE_MS = m(CASE.ttft, "ms")
GPT2_FLOOR_MS = m(DECODE1.result.memory_time, "ms")
L3_FLOOR_MS = m(CASE.weight_floor, "ms")
L3_TPOT_MS = m(CASE.tpot, "ms")
TPOT_TARGET_MS = m(CASE.tpot_target, "ms")
v("h100_tflops", m(CHAT.peak_rate, "TFLOPs/s"), "{:.0f}", "H100 dense FP16 peak, TFLOP/s")
v("h100_bw", m(CHAT.bandwidth, "TB/s"), "{:.2f}", "H100 HBM bandwidth, TB/s")
v("prefill_eff", pct(CHAT.prefill_efficiency), "{:.0f}", "prefill share of peak, %")
v("gpt2_prompt", CHAT.prompt_tokens, "{:,.0f}", "GPT-2 XL prompt, tokens")
v("gpt2_prefill_tflop", m(CHAT.prefill_ops, "TFLOPs"), "{:.1f}", "GPT-2 XL prefill, TFLOP")
v("gpt2_prefill_ms", GPT2_PREFILL_MS, "{:.1f}", "GPT-2 XL prefill, ms")
v("cs_prompt", CASE.prompt_tokens, "{:,.0f}", "case-study prompt, tokens")
v("cs_ttft", PRE_MS, "{:.1f}", "Llama 3 8B 1,000-token prefill, ms")
v("gpt2_floor_ms", GPT2_FLOOR_MS, "{:.1f}", "GPT-2 XL decode floor, ms")
v("cs_w_gb", m(CASE.weights, "GB"), "{:.0f}", "Llama 3 8B INT4 weights, GB")
v("cs_floor_ms", L3_FLOOR_MS, "{:.1f}", "Llama 3 8B INT4 floor, ms")
v("cs_tpot_ms", L3_TPOT_MS, "{:.2f}", "modeled INT4 TPOT, ms")
v("cs_tpot_target", TPOT_TARGET_MS, "{:.0f}", "case-study TPOT target, ms")
ROOM_MS = TPOT_TARGET_MS - L3_TPOT_MS
v("cs_room_ms", ROOM_MS, "{:.1f}", "room in the TPOT target, ms")
v("sw_prompt", lit(SR.SplitwiseBloomPromptTokens), "{:,.0f}", "Splitwise prompt tokens")
v("sw_equiv", lit(SR.SplitwiseBloomEquivalentOutputTokens), "{:.0f}", "as long as this many output tokens")
EAGER_MS = m(EAGER.host_time, "ms")
v("eager_ms", EAGER_MS, "{:.1f}", "eager host submission per GPT-2 XL step, ms")
v("eager_kernels", EAGER.launches, "{:.0f}", "kernels per eager step")
v("decode_intensity", m(DECODE1.result.intensity, "flop/byte"), "{:.0f}", "batch-one decode intensity, FLOP/B")
v("h100_ridge", m(CHAT.ridge, "flop/byte"), "{:.0f}", "H100 ridge, FLOP/B")

print("== Speculative decoding (Scenarios.Gpt2XlSpeculativeGpt2SmallH100; Leviathan 2023)")
DRAFT, VERIFY, ROUND = m(SPEC.draft_cost, "ms"), m(SPEC.verify_time, "ms"), m(SPEC.round_time, "ms")
TPR = SPEC.expected_tokens
v("spec_k", SPEC.draft_length, "{:.0f}", "draft tokens per round")
v("spec_k1", SPEC.draft_length + 1, "{:.0f}", "positions the target checks")
v("spec_alpha", SPEC.acceptance, "{:.1f}", "acceptance")
v("spec_one_minus", 1 - SPEC.acceptance, "{:.1f}", "1 - acceptance")
v("spec_draft_ms", DRAFT, "{:.3f}", "draft cost per round, ms")
v("spec_verify_ms", VERIFY, "{:.3f}", "verification pass, ms")
v("spec_round_ms", ROUND, "{:.3f}", "one round, ms")
v("spec_tokens", TPR, "{:.1f}", "tokens per round")
v("spec_per_token", m(SPEC.per_token_time, "ms"), "{:.2f}", "ms per token")
v("spec_gain", SPEC.tpot_gain, "{:.1f}", "speedup over batch-one decode")
v("spec_wrong", SPEC.acceptance * SPEC.draft_length + 1, "{:.1f}", "wrong answer: alpha k + 1")
v("spec_draft_mb", m(Models.Language.GPT2_Small.size_in_bytes(BYTES_FP16), "MB"), "{:.0f}", "GPT-2 Small FP16 weights, MB")
v("lev_low", lit(SR.LeviathanSpeedupLow), "{:.0f}", "measured speedup, low")
v("lev_high", lit(SR.LeviathanSpeedupHigh), "{:.0f}", "measured speedup, high")

print("== Decode bytes and the batch frontier (Scenarios.Gpt2XlChatDecodeBatchesH100)")
KV_SEQ_GB = m(CHAT.kv_per_sequence, "GB")
GPT2_W_GB = m(CHAT.weights, "GB")
v("gpt2_kv_kb", m(CHAT.kv_per_token, "kB"), "{:.1f}", "GPT-2 XL cache per token, KB")
v("gpt2_kv_seq_gb", KV_SEQ_GB, "{:.2f}", "cache per full context, GB")
v("gpt2_kv_seq_gb4", KV_SEQ_GB, "{:.4f}", "the same, four decimals")
v("gpt2_ctx", CHAT.context_tokens, "{:,.0f}", "GPT-2 XL context, tokens")
for b in (B1, B32):
    v(f"step_gb_{b}", m(CHAT.step_bytes(b), "GB"), "{:.1f}", f"batch {b}: bytes per step, GB")
    v(f"step_ms_{b}", m(CHAT.step_time(b), "ms"), "{:.1f}", f"batch {b}: step, ms")
    v(f"rate_{b}", m(CHAT.token_rate(b), "1/s"), "{:,.0f}", f"batch {b}: tokens/s")
v("batch_8", B8, "{:.0f}", "batch 8")
v("batch_32", B32, "{:.0f}", "batch 32")
v("batch_128", B128, "{:.0f}", "batch 128")
v("cache_share_32", pct(CHAT.cache_share(B32)), "{:.0f}", "batch 32 cache share of bytes, %")
v("tpot_growth", m(CHAT.step_time(B32) / CHAT.step_time(B1), ""), "{:.1f}", "TPOT growth batch 1 to 32")
v("tpot_growth_0", m(CHAT.step_time(B32) / CHAT.step_time(B1), ""), "{:.0f}", "the same, whole (the answer)")
v("rate_growth", m(CHAT.token_rate(B32) / CHAT.token_rate(B1), ""), "{:.1f}", "throughput growth batch 1 to 32")
v("step_ms_128", m(CHAT.step_time(B128), "ms"), "{:.1f}", "batch 128 step, ms")
CEILING = m(CHAT.rate_ceiling, "1/s")
v("ceiling", CEILING, "{:,.0f}", "throughput ceiling, tokens/s")
B_TPOT = CHAT.max_batch_within(CASE.tpot_target)
B_TPOT40 = CHAT.max_batch_within(Q_(LL_INT["tpot_bound_ms"], "ms"))   # the interactive bound
B_MEM = CHAT.max_batch_in_memory
v("b_tpot", B_TPOT, "{:.0f}", "batch cap under 20 ms")
v("b_mem", B_MEM, "{:.0f}", "batch cap by memory")
assert B_TPOT40 > B_MEM  # under 40 ms memory binds
v("h100_mem_gb", m(H100.memory.capacity, "GB"), "{:.1f}", "H100 memory, decimal GB")
v("h100_mem_gib", m(H100.memory.capacity, "GiB"), "{:.0f}", "H100 memory, GiB")
v("sw_batch", lit(SR.SplitwiseTbtBatch), "{:.0f}", "Splitwise batch size")

print("== vLLM serving benchmark (slides/vol1/13b_model_serving/images/data/vllm_serving_benchmark_h100.csv, issue 24684)")
VB = load_series("slides/vol1/13b_model_serving/images/data/vllm_serving_benchmark_h100.csv")
SHORT, LONG = row(VB, prompts="sharegpt"), row(VB, prompts="random")
for name, r in (("short", SHORT), ("long", LONG)):
    v(f"vb_{name}_in", r["input_tokens"] / r["requests"], "{:,.0f}", f"{name}: input tokens per request")
    v(f"vb_{name}_out", r["output_tokens"] / r["requests"], "{:.0f}", f"{name}: output tokens per request")
v("vb_conc", SHORT["max_concurrency"], "{:.0f}", "concurrent requests")
v("vb_ttft_med", SHORT["median_ttft_ms"], "{:.0f}", "short: median TTFT, ms")
v("vb_tpot", SHORT["mean_tpot_ms"], "{:.2f}", "short: mean TPOT, ms")
v("vb_tpot_1", SHORT["mean_tpot_ms"], "{:.1f}", "the same, one decimal")
LITTLE = m(calc_littles_law_rate(SHORT["max_concurrency"], Q_(SHORT["mean_tpot_ms"], "ms")), "1/s")
v("vb_little", LITTLE, "{:,.0f}", "Little's law bound, tokens/s")
v("vb_rate", SHORT["output_tok_s"], "{:,.0f}", "measured output tokens/s")
v("vb_share", pct(SHORT["output_tok_s"] / LITTLE), "{:.0f}", "measured over the bound, %")
v("vb_itl_p99_short", SHORT["p99_itl_ms"], "{:.2f}", "short: p99 ITL, ms")
v("vb_itl_med_long", LONG["median_itl_ms"], "{:.2f}", "long: median ITL, ms")
v("vb_itl_p99_long", LONG["p99_itl_ms"], "{:.2f}", "long: p99 ITL, ms")
v("vb_itl_ratio", LONG["p99_itl_ms"] / LONG["median_itl_ms"], "{:.1f}", "long: p99 over median ITL")

print("== Batching schedules (Scenarios.RequestLevelVersusContinuousBatching)")
NAMES = "ABCDEFGH"
REQ_SPAN = dict(zip(NAMES, SCHED.request_level_spans()))
CB_SPAN = dict(zip(NAMES, SCHED.continuous_spans()))
LENGTHS = dict(zip(NAMES, SCHED.all_lengths))
SLOTS = len(SCHED.output_lengths)
for n in "ABCDE":
    v(f"len_{n.lower()}", LENGTHS[n], "{:.0f}", f"request {n}, tokens")
for n in "EFGH":
    v(f"req_end_{n.lower()}", REQ_SPAN[n][1], "{:.0f}", f"request-level: {n} ends, step")
for n in "DEG":
    v(f"cb_end_{n.lower()}", CB_SPAN[n][1], "{:.0f}", f"continuous: {n} ends, step")
first_used = SCHED.useful_iterations
v("slots", SLOTS, "words", "decode slots, in prose")
v("a_idle", SCHED.max_output - LENGTHS["A"], "{:.0f}", "steps A's slot idles")
v("a_idle_pct", pct((SCHED.max_output - LENGTHS["A"]) / SCHED.max_output), "{:.0f}", "A's slot wasted, %")
v("first_used", first_used, "{:.0f}", "first batch: slot-steps used")
v("first_total", SCHED.slot_iterations, "{:.0f}", "first batch: slot-steps")
v("all_used", SCHED.all_useful_iterations, "{:.0f}", "all slot-steps used")
v("req_total", SLOTS * SCHED.request_level_iterations, "{:.0f}", "request-level: slot-steps")
v("req_steps", SCHED.request_level_iterations, "{:.0f}", "request-level: done at step")
v("req_util", pct(SCHED.request_level_utilization), "{:.0f}", "request-level: used, %")
v("cb_total", SLOTS * SCHED.continuous_iterations, "{:.0f}", "continuous: slot-steps")
v("cb_steps", SCHED.continuous_iterations, "{:.0f}", "continuous: done at step")
v("cb_util", pct(SCHED.continuous_utilization), "{:.1f}", "continuous: used, %")
v("e_wait", REQ_SPAN["E"][0], "{:.0f}", "request-level: E's wait for its first token, steps")
v("orca_gain", lit(SR.OrcaThroughputOverFasterTransformer), "{:.1f}", "Orca over FasterTransformer")
v("vllm_rate_low", lit(SR.VllmRateOverOrcaOracleLow), "{:.1f}", "vLLM over Orca (Oracle), low")
v("vllm_rate_high", lit(SR.VllmRateOverOrcaOracleHigh), "{:.1f}", "vLLM over Orca (Oracle), high")
v("vllm_rate_ft", lit(SR.VllmRateOverFasterTransformer), "{:.0f}", "vLLM over FasterTransformer, up to")

print("== Generation stall and chunked prefill (Scenarios.Llama3_8BInt4PrefillStallH100; Sarathi-Serve)")
DEC_MS = m(STALL.decode_step, "ms")
STALL_MS = m(STALL.stalled_step, "ms")
CHUNK_MS = m(STALL.chunked_step, "ms")
CHUNKS = STALL.chunks
v("st_decodes", STALL.decodes, "{:.0f}", "streams decoding")
v("st_ctx", STALL.context_tokens, "{:,.0f}", "tokens per stream")
v("st_bytes", m(STALL.decode_bytes, "GB"), "{:.1f}", "decode step bytes, GB")
v("st_dec_ms", DEC_MS, "{:.1f}", "decode step, ms")
v("st_stall_ms", STALL_MS, "{:.1f}", "step with the whole prefill, ms")
v("st_chunks", CHUNKS, "words", "chunks, in prose")
v("st_chunk_tok", STALL.chunk_tokens, "{:.0f}", "tokens per chunk")
v("st_chunk_pre", m(STALL.chunk_prefill, "ms"), "{:.1f}", "prefill of one chunk, ms")
v("st_chunk_ms", CHUNK_MS, "{:.1f}", "step with one chunk, ms")
v("st_first", m(STALL.chunked_first_token, "ms"), "{:.0f}", "first token after the chunked steps, ms")
v("st_safe", STALL.largest_safe_chunk, "{:.0f}", "largest safe chunk, tokens")
v("sarathi_low", lit(SR.SarathiMistral7bCapacityGain), "{:.1f}", "Sarathi-Serve over vLLM, Mistral-7B")
v("sarathi_high", lit(SR.SarathiYi34bCapacityGain), "{:.1f}", "Sarathi-Serve over vLLM, Yi-34B")

print("== KV budget (Scenarios.Llama3_8BFp16ConversationsH100, Llama3_70BInt4KvBudgetH100)")
R8, R70 = CONV8.result, KV70.result
FREE_GB = m(R8.kv_budget, "GB")
PER_CONV_GB = m(R8.kv_per_request, "GB")
N8K = R8.max_requests
L3_KV_TOK = m(R8.kv_per_token, "byte")
v("l3_w16_gb", m(R8.weights, "GB"), "{:.1f}", "Llama 3 8B FP16 weights, GB")
v("conv_tokens", CONV8.context_tokens, "{:,.0f}", "conversation, tokens")
v("conv_gb", PER_CONV_GB, "{:.2f}", "per conversation, GB")
v("free_gb", FREE_GB, "{:.1f}", "free for the cache, GB")
v("n_conv", N8K, "{:.0f}", "conversations that fit")
v("n_conv_noweights", m(H100.memory.capacity / R8.kv_per_request, ""), "{:.0f}", "the common error: without the weights")
v("n_conv_gb", N8K * PER_CONV_GB, "{:.1f}", "65 conversations, GB")
BUDGET70_GB = m(R70.kv_budget, "GB")
KV70_TOK = m(R70.kv_per_token, "byte")
v("kv70_budget", BUDGET70_GB, "{:.1f}", "70B-class cache budget, GB")
v("kv70_tok", KV70_TOK, "{:,.0f}", "70B-class cache per token, bytes")
v("kv70_ratio", KV70_TOK / m(R8.kv_per_token, "byte"), "{:.1f}", "70B over 8B per token")
v("kv70_batch", KV70.requests, "{:.0f}", "conversations in the crossing")
X32 = KV70.context_fit()
v("kv70_cross", round(X32, -2), "{:,.0f}", "32 conversations fill the budget near, tokens")
v("kv70_layers", KV70.workload.layers, "{:.0f}", "70B-class layers")
v("kv70_heads", KV70.workload.kv_heads, "{:.0f}", "70B-class key-value heads")
v("kv70_hdim", R70.head_dim, "{:.0f}", "70B-class head width")
v("kwon_param_pct", lit(Literature.ServingMemoryLayout.ParameterSharePct), "{:.0f}", "13B on A100 40 GB: weights, %")
v("kwon_kv_pct", lit(Literature.ServingMemoryLayout.KvCacheSharePct), "{:.0f}", "13B on A100 40 GB: cache, %")

print("== Reservation and paging (slides/vol1/13b_model_serving/images/data/pagedattention_kv_memory_breakdown.csv; ServingResults)")
PA = load_series("slides/vol1/13b_model_serving/images/data/pagedattention_kv_memory_breakdown.csv")
USED = {r["system"]: r["token_states"] for r in PA.entries}
orca_used = [USED["Orca (Max)"], USED["Orca (Pow2)"], USED["Orca (Oracle)"]]
v("orca_used_low", min(orca_used), "{:.1f}", "Orca: token states, lowest %")
v("orca_used_high", max(orca_used), "{:.1f}", "Orca: token states, highest %")
v("vllm_used", USED["vLLM"], "{:.1f}", "vLLM: token states, %")
v("vllm_batch_orca", lit(SR.VllmBatchedRequestsOrcaOracle), "{:.1f}", "Orca (Oracle) batch")
v("vllm_batch_vllm", lit(SR.VllmBatchedRequestsVllm), "{:.1f}", "vLLM batch")
v("vllm_batch_gain", lit(SR.VllmBatchedRequestsVllm) / lit(SR.VllmBatchedRequestsOrcaOracle), "{:.1f}",
  "batch growth")
v("block_default", POOL.block_tokens, "{:.0f}", "vLLM default block, tokens")

print("== vLLM's startup log (Scenarios.VllmKvPoolLlama3_8BH100, issue 21266)")
v("log_weights_gib", m(POOL.weights, "GiB"), "{:.2f}", "weights, GiB")
v("log_weights_gb", m(POOL.weights, "GB"), "{:.1f}", "weights, GB")
v("log_tokens", POOL.kv_pool_tokens, "{:,.0f}", "pool, tokens")
v("log_kv_gib", m(POOL.pool_bytes, "GiB"), "{:.2f}", "pool tokens x cache per token, GiB")
v("log_conc", POOL.concurrency, "{:.2f}", "pool tokens over 8,192")
v("log_conc_0", math.floor(POOL.concurrency), "{:.0f}", "whole conversations")
v("log_util", POOL.memory_utilization, "{:.1f}", "share of the GPU vLLM claims")
v("log_util_pct", pct(POOL.memory_utilization), "{:.0f}", "the same, %")
v("log_claimed_gib", m(POOL.claimed, "GiB"), "{:.0f}", "claimed, GiB")
v("log_workspace_gib", m(POOL.workspace, "GiB"), "{:.2f}", "left for activations and workspace, GiB")
v("log_high_pct", pct(N8K / POOL.concurrency - 1), "{:.0f}", "estimate high by, %")
v("log_req_tokens", POOL.request_tokens, "{:,.0f}", "paged conversation, tokens")
v("log_blocks", POOL.blocks_per_request, "{:.0f}", "blocks per conversation")
v("log_paged_est", POOL.paged_requests(R8.kv_budget), "{:.0f}", "paged estimate, conversations")
v("log_paged_pool", POOL.pool_paged_requests, "{:.0f}", "vLLM's pool, conversations")
v("cs_resident", CASE.resident_requests, "{:.0f}", "the chapter's bound for 4-bit weights")
v("block_waste_mb", m(POOL.last_block_waste, "MB"), "{:.0f}", "most one sequence wastes, MB")

print("== Prefix sharing (Scenarios.Llama3_8BSharedSystemPrompt, Llama3_8BConversationReloadPcieGen4)")
v("prefix_tokens", PREFIX.prefix_tokens, "{:,.0f}", "shared prefix, tokens")
v("prefix_mb", m(PREFIX.kv_shared, "MB"), "{:.0f}", "the prefix's cache, MB")
v("prefix_users", PREFIX.users, "{:.0f}", "conversations")
v("prefix_saved_gb", m(PREFIX.kv_unshared - PREFIX.kv_shared, "GB"), "{:.0f}", "saved by sharing, GB")
v("prefix_copies_gb", m(PREFIX.kv_unshared, "GB"), "{:.1f}", "100 copies, GB")
v("prefix_shared_gb", m(PREFIX.kv_shared, "GB"), "{:.2f}", "shared copy, GB")
v("prefix_users_less1", PREFIX.users - 1, "{:.0f}", "conversations that skip the prefill")
LINK = next(iter(RELOAD.links))
v("reload_ms", m(RELOAD.transfer_time(LINK), "ms"), "{:.0f}", "reload one 8,192-token cache, ms")
v("reload_gbps", m(RELOAD.links[LINK], "GB/s"), "{:.0f}", "PCIe Gen4 x16, GB/s")
v("share_ps_low", lit(SR.VllmParallelSamplingSavingLowPct), "{:.1f}", "parallel sampling saving, low %")
v("share_ps_high", lit(SR.VllmParallelSamplingSavingHighPct), "{:.1f}", "parallel sampling saving, high %")
v("share_bs_low", lit(SR.VllmBeamSearchSavingLowPct), "{:.1f}", "beam search saving, low %")
v("share_bs_high", lit(SR.VllmBeamSearchSavingHighPct), "{:.1f}", "beam search saving, high %")

print("== DistServe (Scenarios.DistServe13BPoolsA100)")
COLO, PRE_ONLY, DEC_ONLY = m(DS.colocated_rate, "1/s"), m(DS.prefill_rate, "1/s"), m(DS.decode_rate, "1/s")
v("ds_colo", COLO, "{:.1f}", "both phases, requests/s")
v("ds_pre", PRE_ONLY, "{:.1f}", "prefill only, requests/s")
v("ds_dec", DEC_ONLY, "{:.0f}", "decode only, requests/s")
v("ds_params", lit(SR.DistServeModelParamsB), "{:.0f}", "model, billions of parameters")
v("ds_in", lit(SR.DistServeInputTokens), "{:.0f}", "tokens in")
v("ds_out", lit(SR.DistServeOutputTokens), "{:.0f}", "tokens out")
v("ds_slo_pct", lit(SR.DistServeSloAttainmentPct), "{:.0f}", "SLO attainment, %")
v("ds_pre_exact", DS.prefill_accelerators_exact, "{:.1f}", "prefill GPUs, exact")
v("ds_pre_n", DS.prefill_accelerators, "{:.0f}", "prefill GPUs")
v("ds_gpus", DS.accelerators, "{:.0f}", "GPUs in the pools")
v("ds_gpus_word", DS.accelerators, "words", "the same, in prose")
v("ds_per_gpu", m(DS.per_accelerator_rate, "1/s"), "{:.1f}", "per GPU, requests/s")
v("ds_gain", DS.gain, "{:.1f}", "gain over both phases on one GPU")
v("ds_more", lit(SR.DistServeMoreRequests), "{:.1f}", "more requests, up to")
v("ds_tighter", lit(SR.DistServeTighterSlo), "{:.1f}", "tighter SLO")
v("ds_kv_gb", m(DS.kv_per_request, "GB"), "{:.2f}", "OPT-66B cache per 512-token request, GB")
v("ds_kv_rate", m(DS.kv_arrival_rate, "1/s"), "{:.0f}", "requests/s in the transfer estimate")
v("ds_traffic", m(DS.kv_traffic, "GB/s"), "{:.1f}", "cache traffic, GB/s")
v("ds_traffic_gbps", m(DS.kv_traffic, "Gbit/s"), "{:.0f}", "cache traffic, Gbps")
v("ds_link", m(DS.link_bandwidth, "GB/s"), "{:.0f}", "A100 link, GB/s")
v("ds_xfer_pct", lit(SR.DistServeKvTransferShareMaxPct), "{:.1f}", "DistServe transfer share, under %")
v("sw_xfer_pct", lit(SR.SplitwiseKvTransferShareOfE2ePct), "{:.1f}", "Splitwise transfer share, %")
v("sw_xfer_ms", lit(SR.SplitwiseKvTransferH100Ms), "{:.0f}", "Splitwise transfer on H100, ms")
v("sw_cap_from", lit(SR.SplitwisePowerCapFromW), "{:.0f}", "uncapped power, W")
v("sw_cap_to", lit(SR.SplitwisePowerCapToW), "{:.0f}", "capped power, W")
v("sw_gain", lit(SR.SplitwiseThroughputGain), "{:.1f}", "split clusters' throughput gain")
v("sw_cost_pct", lit(SR.SplitwiseCostReductionPct), "{:.0f}", "split clusters' cost reduction, %")

print("== Joules per token (Scenarios.H100ChatEnergyPerToken over Gpt2XlChatDecodeBatchesH100)")
P_NODE, P_LOW = m(ENERGY.node_power, "W"), m(ENERGY.low_use_power, "W")
J1 = m(ENERGY.energy_per_token(CHAT.token_rate(B1)), "J")
J32 = m(ENERGY.energy_per_token(CHAT.token_rate(B32)), "J")
J1_LOW = m(ENERGY.energy_per_token(CHAT.token_rate(B1), low_use=True), "J")
v("p_node", P_NODE, "{:,.0f}", "node power, W")
v("p_node_kw", m(ENERGY.node_power, "kW"), "{:.0f}", "the same, kW")
v("p_low", P_LOW, "{:.0f}", "low-use node power, W")
v("p_low_gpu", m(ENERGY.low_use_accelerator_power, "W"), "{:.0f}", "low-use GPU power, W")
v("p_host", m(ENERGY.host_power, "W"), "{:.0f}", "host share, W")
v("j1", J1, "{:.2f}", "batch 1, 1 kW, J per token")
v("j32", J32, "{:.2f}", "batch 32, 1 kW, J per token")
v("j1_low", J1_LOW, "{:.2f}", "batch 1, 600 W, J per token")
v("j_low_ratio", J1_LOW / J32, "{:.1f}", "600 W batch 1 over batch 32")
v("p_ratio", P_NODE / P_LOW, "{:.1f}", "power ratio")
v("cc_joules", m(ENERGY.joules_per_token, "J"), "{:.1f}", "the chapter's J per token")
v("cc_low_joules", m(ENERGY.low_use_joules_per_token, "J"), "{:.1f}", "the chapter's low-use J per token")
v("g_reduction", lit(SR.GooglePromptEnergyReduction), "{:.0f}", "Google: energy per prompt fell, times")
v("g_model", lit(SR.GooglePromptEnergyModelReduction), "{:.0f}", "of it from models, times")
v("g_util", lit(SR.GooglePromptEnergyUtilizationReduction), "{:.1f}", "of it from utilization, times")
GEMINI = load_series("vol2/15_sustainable_ai/data/gemini_prompt_energy.csv").entries
v("g_wh", sum(r["comprehensive_wh"] for r in GEMINI if r["comprehensive_counted"] == 1), "{:.2f}",
  "median prompt, Wh (full stack)")

print("== Cost per million tokens (Scenarios.Llama3_8BInt4ChatServiceH100; vol1/13_model_serving/data/llm_api_input_prices.csv)")
COST_M = m(CASE.cost_per_million_tokens, "dollar")
v("cs_price", m(CASE.price, "dollar/hour"), "{:.0f}", "H100, $ per hour")
v("cs_prefill_rate", m(CASE.prefill_rate, "1/s"), "{:,.0f}", "prefill, prompt tokens/s")
v("cs_req_rate", m(CASE.request_rate, "1/s"), "{:.0f}", "requests per second")
v("cs_decode", CASE.decode_tokens, "{:.0f}", "generated tokens")
v("cs_tokens", CASE.total_tokens, "{:,.0f}", "tokens per request")
v("cs_mtok_h", CASE.tokens_per_hour / 10 ** 6, "{:.1f}", "million tokens per hour")
v("cs_usd_m", COST_M, "{:.3f}", "$ per million tokens")
PRICES = {r["model"]: r["usd_per_mtok_input"] for r in load_series("vol1/13_model_serving/data/llm_api_input_prices.csv").entries}
v("price_gpt4", PRICES["GPT-4 (Original)"], "{:.0f}", "GPT-4 (original), $ per million input tokens")
v("price_flash", PRICES["Gemini 1.5 Flash"], "{:.3f}", "Gemini 1.5 Flash, $ per million input tokens")

print("== Synthesis (Scenarios.Llama2_70B*DecodeStreams, Llama2_70B*ConversationsEightH100)")
S200, S40 = STREAMS200.in_flight, STREAMS40.in_flight
R16, R8b = SYN16.result, SYN8.result
v("llama_conv_tpot_s", m(STREAMS200.time_in_system, "s"), "{:.1f}", "200 ms, s")
v("llama_int_tpot_s", m(STREAMS40.time_in_system, "s"), "{:.2f}", "40 ms, s")
v("streams_200", S200, "{:,.0f}", "streams at most under 200 ms")
v("streams_40", S40, "{:,.0f}", "streams at most under 40 ms")
v("syn_mem_gb", m(R16.hbm_total, "GB"), "{:.0f}", "eight H100s, GB")
v("syn_w16_gb", m(R16.weights, "GB"), "{:.0f}", "16-bit 70B weights, GB")
v("syn_w4_gb", m(R70.weights, "GB"), "{:.1f}", "the chapter's 4-bit 70B weights, GB")
v("syn_w_ratio", m(BYTES_FP16 / BYTES_INT4, ""), "{:.0f}", "16-bit over 4-bit")
v("syn_free_gb", m(R16.kv_budget, "GB"), "{:.0f}", "free, GB")
v("syn_conv_mb", m(R16.kv_per_request, "MB"), "{:.0f}", "one conversation, MB")
v("syn_streams", R16.max_requests, "{:,.0f}", "memory streams")
v("syn_step_ms", m(R16.max_requests / STREAMS200.arrival_rate, "ms"), "{:.0f}", "step at the memory limit, ms")
v("syn8_free_gb", m(R8b.kv_budget, "GB"), "{:.0f}", "8-bit: free, GB")
v("syn8_conv_mb", m(R8b.kv_per_request, "MB"), "{:.0f}", "8-bit: conversation, MB")
v("syn8_streams", round(R8b.max_requests, -3), "{:,.0f}", "8-bit: memory streams, about, to the thousand")
v("syn_servers", MLP_SERVER / MLP_INTER, "{:.1f}", "servers for 30,904 tokens/s under 40 ms")

if __name__ == "__main__":
    write(__file__)
