#!/usr/bin/env python3
"""Every number on the Lecture 6b (Network Architectures, Part 2) slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the attention and transformer
scenarios (Scenarios.FourTokenToyAttentionHead, Gpt2XlBlockFullWindow,
TwelveHeadScoresAt100kTokens, the KV footprints, Gpt2XlContextMemoryOnH100),
the decode scenarios (Llama2_70B_DecodeTwoH100, H100DecodeRoofline), the
recommender scenarios (CriteoShapedRequestOnH100, HundredMillionItemTableOnA100),
the model registry (GPT-2 XL and Small, BERT-Base, GPT-3, Llama 2 and 3.1,
PaLM, OPT-13B, DLRM, the vision lighthouses), the H100 and A100, and the cited
results in Literature (context windows, FlashAttention's Tables 20 and 21,
Kwon et al.'s memory layout, Gupta et al., Mudigere et al., Fiction.LiveBench).
The chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports this module, so figures and slides
share one computation.

The attention weights of the opening slide are a measurement run here
(attention_probe.py writes data/attention_probe.txt); this file reads them
from that file.

Kept typed in the deck, by rule: quoted code (nanoGPT, DLRM) and the counting
rules written as formulas (4d^2, 8d^2, 12d^2, 6d, 2Ld), wrong answer choices,
years, layer and head indices of the probe, TikZ layout (arrow widths, the
drawn query-head strip, the six drawn tables), and lecture and section numbers.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Hardware, Models, Literature, Scenarios = mlsysim.Hardware, mlsysim.Models, mlsysim.Literature, mlsysim.Scenarios
units = mlsysim.core.units
FP32, FP16 = units.BYTES_FP32, units.BYTES_FP16
HERE = Path(__file__).resolve().parent

LM = Models.Language
XL, SMALL, BERT, GPT3 = LM.GPT2, LM.GPT2_Small, LM.BERT_Base, LM.GPT3
L7, L8, L70, PALM, OPT = LM.Llama2_7B, LM.Llama3_8B, LM.Llama2_70B, LM.PaLM_540B, LM.OPT_13B
DLRM = Models.Recommendation.DLRM
H100, A100 = Hardware.Cloud.H100, Hardware.Cloud.A100
WINDOWS, FLASH = Literature.ContextWindows, load_series("slides/vol1/06b_nn_architectures/images/data/flashattention_a100_benchmark.csv")
LAYOUT, RECSYS = Literature.ServingMemoryLayout, Literature.RecommendationSystems

TOY = Scenarios.FourTokenToyAttentionHead
QKV = Scenarios.Gpt2SmallSixTokenQkvProjection
BLOCK, BLOCK_LONG = Scenarios.Gpt2XlBlockFullWindow, Scenarios.Gpt2XlBlockAt100kTokens
SCORES, SCORES_4K, SCORES_8K = (Scenarios.TwelveHeadScoresAt100kTokens, Scenarios.TwelveHeadScoresAt4096Tokens,
                                Scenarios.TwelveHeadScoresAt8192Tokens)
HOOK, BERT_4K = Scenarios.Gpt3ToGemini15PreviewWindow, Scenarios.BertWindowToFourK
KV_XL, KV_XL_10K = Scenarios.Gpt2XlKvFullWindow, Scenarios.Gpt2XlKvAt10kTokens
KV_L7, KV_L8, KV_L8_1M = Scenarios.Llama2_7BKvAt2048Tokens, Scenarios.Llama3_8BKvFullWindow, Scenarios.Llama3_8BKvAtMillionTokens
VLLM, VLLM_HALF = Scenarios.Opt13BKvOnA100_40GB, Scenarios.Opt13BKvOnA100_40GBHalfWidth
CTX = Scenarios.Gpt2XlContextMemoryOnH100
DECODE2, ROOF = Scenarios.Llama2_70B_DecodeTwoH100, Scenarios.H100DecodeRoofline
MILLION, SCORES_1M = Scenarios.Llama3_8BMillionTokenPrompt, Scenarios.OneHeadScoresAtMillionTokens
GATHER = Scenarios.CriteoShapedRequestOnH100
TABLE, TABLES2, USERS = (Scenarios.HundredMillionItemTableOnA100, Scenarios.TwoHundredMillionRowTablesOnA100,
                         Scenarios.BillionUserTable)
STATE = {"resnet": Scenarios.ResNet50Fp32Adam, "mobilenet": Scenarios.MobileNetV2Fp32Adam,
         "kws": Scenarios.KwsDsCnnFp32Adam, "gpt2": Scenarios.Gpt2Fp32AdamOnH100}


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def labeled_gb(hw):
    """A device's capacity as its label prints it (80 GiB reads "80 GB")."""
    return m(hw.memory.capacity, "GiB")


def intensity(workload):
    """FLOPs per FP32 weight byte (the chapter's weight-only proxy, Table 8)."""
    return m(workload.inference_flops / workload.size_in_bytes(FP32), "flop/byte")


def mantissa_exponent(x):
    e = math.floor(math.log10(x))
    return x / 10 ** e, e


def kv_kb(workload):
    return m(workload.get_kv_cache_size(1, 1, FP16), "KB")


def load_probe():
    """Attention rows measured by attention_probe.py (bert-base-uncased):
    {(sentence, pronoun): {token: weight}}."""
    rows, cur = {}, None
    for line in (HERE / "data" / "attention_probe.txt").read_text().splitlines():
        if line.startswith("The "):
            cur = line.split("  (from")[0]
            rows[cur] = {}
        elif cur and line.startswith("  "):
            t, w = line.split()
            rows[cur][t] = float(w)
    return rows


def toy_fmt(x):
    """The toy's cells print a zero as 0 and anything else to two places."""
    return "{:.2f}" if round(x, 2) else "{:.0f}"


print("== Hook: context windows (Models; Literature.ContextWindows)")
v("bert_window", BERT.context_length, "{:,.0f}", "BERT positions")
v("gpt2_window", XL.context_length, "{:,.0f}", "GPT-2 window")
v("gpt3_window", GPT3.context_length, "{:,.0f}", "GPT-3 window")
v("gpt4t_window", WINDOWS.Gpt4TurboTokens, "{:,.0f}", "GPT-4 Turbo window")
v("claude21_window", WINDOWS.Claude21Tokens, "{:,.0f}", "Claude 2.1 window")
v("gemini_std_window", WINDOWS.Gemini15ProStandardTokens, "{:,.0f}", "Gemini 1.5 Pro standard window")
growth_round = round(HOOK.ratio, -2)
v("window_growth_round", growth_round, "{:.0f}", "the same, as the slide states it")
v("score_growth_round", growth_round ** 2, "{:,.0f}", "score work at that window, times")

print("== Recall: recurrence (Scenarios.ElmanRnnStep, RecurrentState512Fp32; Gpt2XlKv*)")
v("rnn_steps", Scenarios.ElmanRnnStep.steps, "{:,.0f}", "sequence length, steps")
v("rnn_state_kb", m(Scenarios.RecurrentState512Fp32.state_bytes, "KiB"), "{:.0f}", "512-wide FP32 state, KB (binary)")
v("rnn_width", Scenarios.RecurrentState512Fp32.hidden_dim, "{:.0f}", "state width")
v("kv_xl_kb", m(KV_XL.per_token, "KB"), "{:.1f}", "GPT-2 XL KV per token, KB")
v("kv_xl_kb_round", m(KV_XL.per_token, "KB"), "{:.0f}", "the same, to the unit")
v("kv_10k_tokens", KV_XL_10K.tokens, "{:,.0f}", "tokens")
v("kv_10k_gb", m(KV_XL_10K.total, "GB"), "{:.1f}", "GPT-2 XL KV at 10,000 tokens, GB")

print("== Running example: GPT-2 XL (Models.Language.GPT2)")
v("xl_layers", XL.layers, "{:.0f}", "blocks")
v("xl_d", XL.hidden_dim, "{:,.0f}", "width")
v("xl_d_raw", XL.hidden_dim, "{:.0f}", "width, as config.json prints it")
v("xl_heads", XL.heads, "{:.0f}", "heads")
v("xl_vocab_raw", XL.vocab_size, "{:.0f}", "vocabulary, as config.json prints it")
v("xl_vocab", XL.vocab_size, "{:,.0f}", "vocabulary")
v("xl_params_b", m(XL.parameters, "Bparam"), "{:.1f}", "published parameters, billions")
v("xl_gflop", m(XL.inference_flops, "GFLOP"), "{:.0f}", "GFLOP per token (the chapter's figure)")
v("xl_head_width", XL.head_width, "{:.0f}", "head width")

print("== Measured attention (data/attention_probe.txt; BERT-base)")
probe = load_probe()
they = probe["The student did not finish all the homework because they were tired."]
v("probe_student", they["student"], "{:.2f}", "weight from 'they' on 'student'")
v("probe_homework", they["homework"], "{:.2f}", "weight from 'they' on 'homework'")
v("probe_trophy_big", probe["The trophy did not fit in the suitcase because it was too big."]["trophy"], "{:.2f}",
  "weight from 'it' on 'trophy', with 'big'")
v("probe_trophy_small", probe["The trophy did not fit in the suitcase because it was too small."]["trophy"], "{:.2f}",
  "weight from 'it' on 'trophy', with 'small'")
v("bert_heads_total", BERT.layers * BERT.heads, "{:.0f}", "attention heads in BERT-base")

print("== Toy head (Scenarios.FourTokenToyAttentionHead)")
v("toy_s", TOY.tokens, "{:.0f}", "tokens")
v("toy_d", TOY.width, "{:.0f}", "width")
v("toy_macs", TOY.macs_per_product, "{:.0f}", "multiply-adds per product, S^2 d")
for name, rows in (("tq", TOY.queries), ("tk", TOY.keys), ("tv", TOY.values)):
    for r, row in enumerate(rows, start=1):
        for c, x in enumerate(row, start=1):
            v(f"{name}{r}{c}", x, "{:.0f}", f"{name[1].upper()} row {r}, column {c}")
for c, k in enumerate(TOY.keys, start=1):
    v(f"td1{c}", sum(a * b for a, b in zip(TOY.queries[0], k)), "{:.0f}", f"q1 . k{c}, before scaling")
for r, row in enumerate(TOY.scores, start=1):
    for c, x in enumerate(row, start=1):
        v(f"ts{r}{c}", x, toy_fmt(x), f"score, token {r} on key {c}")
for r, row in enumerate(TOY.weights, start=1):
    for c, x in enumerate(row, start=1):
        v(f"tw{r}{c}", x, "{:.2f}", f"softmax weight, token {r} on key {c}")
for r, row in enumerate(TOY.outputs, start=1):
    for c, x in enumerate(row, start=1):
        v(f"to{r}{c}", x, "{:.2f}", f"output, token {r}, column {c}")

print("== Heads: GPT-2 Small's packed projection (Scenarios.Gpt2SmallSixTokenQkvProjection)")
v("s_tokens", QKV.tokens, "{:.0f}", "tokens")
v("s_tokens_word", QKV.tokens, "words", "the same, in prose")
v("s_d", QKV.width, "{:.0f}", "width")
v("s_qkv", QKV.packed_width, "{:,.0f}", "packed Q, K, V columns")
v("s_heads", SMALL.heads, "{:.0f}", "heads")
v("s_heads_word", SMALL.heads, "words", "the same, in prose")
v("s_head_width", SMALL.head_width, "{:.0f}", "head width")

print("== GPT-2 XL's weights from the block rule (Scenarios.Gpt2XlBlockFullWindow)")
v("block_coef", BLOCK.block_coefficient, "{:.0f}", "weights per block, units of d^2")
v("block_m", BLOCK.weights_per_block / 10 ** 6, "{:.1f}", "weights per block, millions")
v("blocks_b", BLOCK.block_weights / 10 ** 9, "{:.2f}", "all blocks, billions")
v("table_m", BLOCK.token_table / 10 ** 6, "{:.0f}", "token table, millions")
v("table_b", BLOCK.token_table / 10 ** 9, "{:.2f}", "token table, billions")
v("rule_b", BLOCK.block_rule_parameters / 10 ** 9, "{:.2f}", "blocks plus token table, billions")
v("attn_proj_b", BLOCK.attention_weights / 10 ** 9, "{:.2f}", "attention projections, billions")
v("mlp_b", BLOCK.mlp_weights / 10 ** 9, "{:.2f}", "MLPs, billions")
v("exact_m", BLOCK.exact_parameters / 10 ** 6, "{:,.0f}", "exact count with biases, norms, positions, millions")
v("xl_gflop_rule", m(BLOCK.flops_per_token, "GFLOP"), "{:.2f}", "FLOPs per token from the rule, GFLOP")

print("== Predict: attention's share at 1,024 tokens (Scenarios.Gpt2XlBlockFullWindow)")
v("dense_bmac", BLOCK.dense_macs / 10 ** 9, "{:.1f}", "12 S d^2, billions of MACs")
v("attn_bmac", BLOCK.attention_macs / 10 ** 9, "{:.2f}", "2 S^2 d, billions of MACs")
v("block_bmac", BLOCK.block_macs / 10 ** 9, "{:.1f}", "block total, billions of MACs")
v("attn_share_pct", 100 * BLOCK.attention_share, "{:.0f}", "attention share, %")
v("prompt_dense_tflop", m(BLOCK.prompt_dense_flops, "TFLOP"), "{:.2f}", "prompt weight work, TFLOP")
v("prompt_attn_tflop", m(BLOCK.prompt_attention_flops, "TFLOP"), "{:.2f}", "prompt attention, TFLOP")
v("prompt_tflop", m(BLOCK.prompt_flops, "TFLOP"), "{:.2f}", "prompt total, TFLOP")

print("== Crossover (Scenarios.Gpt2XlBlockFullWindow, Gpt2XlBlockAt100kTokens)")
v("crossover", BLOCK.crossover, "{:,.0f}", "crossover, tokens (6d)")
v("crossover_mult_word", BLOCK.crossover_multiple, "words", "crossover over the width, in prose")
v("long_tokens", BLOCK_LONG.context_tokens, "{:,.0f}", "long context, tokens")
v("long_tokens_k", BLOCK_LONG.context_tokens / 10 ** 3, "{:.0f}", "the same, thousands")
v("long_ratio", BLOCK_LONG.attention_to_dense, "{:.0f}", "attention over the rest at 100K, times")
v("long_ratio_word", round(BLOCK_LONG.attention_to_dense), "words", "the same, in prose")

print("== Materialized scores (Scenarios.TwelveHeadScoresAt100kTokens and variants)")
v("sc_tokens", SCORES.seq_len, "{:,.0f}", "tokens")
v("sc_heads", SCORES.heads, "{:.0f}", "heads")
v("sc_layers", SCORES.layers, "{:.0f}", "layers")
v("sc_sq_exp", math.log10(SCORES.seq_len ** 2), "{:.0f}", "tokens squared, power of ten")
mant, exp = mantissa_exponent(SCORES.scores_per_layer)
v("sc_count_mant", mant, "{:.1f}", "scores per layer, mantissa")
v("sc_count_exp", exp, "{:.0f}", "scores per layer, power of ten")
v("sc_bytes", m(SCORES.bytes_per_score, "byte"), "{:.0f}", "bytes per FP16 score")
v("sc_bytes_word", m(SCORES.bytes_per_score, "byte"), "words", "the same, in prose")
v("sc_layer_gb", m(SCORES.layer_bytes, "GB"), "{:.0f}", "one layer, GB")
v("sc_total_gb", m(SCORES.total_bytes, "GB"), "{:,.0f}", "all layers, GB")
v("sc_device_word", SCORES.layer_device_multiple, "words", "one layer over an 80 GB device, in prose")
v("sc_wrong_gb", m(SCORES.layer_bytes, "GB") / 100, "{:.1f}", "the answer with a factor of 100 dropped, GB")
v("sc4k_tokens", SCORES_4K.seq_len, "{:,.0f}", "tokens")
v("sc8k_tokens", SCORES_8K.seq_len, "{:,.0f}", "tokens")
v("sc4k_gb", m(SCORES_4K.layer_bytes, "GB"), "{:.2f}", "one layer at 4,096 tokens, GB")
v("sc8k_gb", m(SCORES_8K.layer_bytes, "GB"), "{:.2f}", "one layer at 8,192 tokens, GB")
v("sc_double_word", m(SCORES_8K.layer_bytes / SCORES_4K.layer_bytes, ""), "words", "growth for twice the tokens, in prose")
v("dev_gb", labeled_gb(H100), "{:.0f}", "one accelerator's labeled memory, GB")

print("== True or false (Scenarios.BertWindowToFourK; slides/vol1/06b_nn_architectures/images/data/flashattention_a100_benchmark.csv)")
v("bert_long", BERT_4K.tokens, "{:,.0f}", "longer input, tokens")
v("bert_long_ratio", BERT_4K.score_ratio, "{:.0f}", "score memory growth, times")
fa = {int(r["seq_len"]): r for r in FLASH.entries}
seqs = sorted(fa)
s4k, s1k, s64k = BERT_4K.tokens, XL.context_length, seqs[-1]
v("fa_mem_4k_gb", m(units.Q_(fa[s4k]["flashattention_memory_mb"], "MB"), "GB"), "{:.2f}", "FlashAttention memory at 4K, GB")
v("fa_mem_64k_gb", m(units.Q_(fa[s64k]["flashattention_memory_mb"], "MB"), "GB"), "{:.1f}", "FlashAttention memory at 64K, GB")
v("fa_mem_saving", fa[s4k]["pytorch_memory_mb"] / fa[s4k]["flashattention_memory_mb"], "{:.0f}",
  "PyTorch over FlashAttention memory at 4K, times")
v("fa_time_growth", fa[s64k]["flashattention_runtime_ms"] / fa[s1k]["flashattention_runtime_ms"], "{:,.0f}",
  "FlashAttention runtime, 1K to 64K, times")
v("fa_square", (s64k / s1k) ** 2, "{:,.0f}", "a pure square over the same range")
v("fa_doubling_word", fa[s64k]["flashattention_runtime_ms"] / fa[max(x for x in seqs if x < s64k)]["flashattention_runtime_ms"], "words",
  "runtime growth per doubling, in prose")

print("== Decode (Scenarios.Llama2_70B_DecodeTwoH100, H100DecodeRoofline)")
r2 = DECODE2.result
v("h100_tflops", m(H100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "H100 dense FP16 peak, TFLOP/s")
v("h100_tbs", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM bandwidth, TB/s")
v("ridge", m(ROOF.ridge, "flop/byte"), "{:.0f}", "H100 ridge, FLOP per byte")
v("decode_i", m(ROOF.point_intensity, "flop/byte"), "{:.0f}", "decode intensity, FLOP per byte")
v("decode_i_one", m(ROOF.point_intensity, "flop/byte"), "{:.1f}", "the same, one decimal")
v("fp16_bytes", m(FP16, "byte"), "{:.0f}", "bytes per FP16 value")
v("fp32_bytes", m(FP32, "byte"), "{:.0f}", "bytes per FP32 value")
v("l70_params_b", m(L70.parameters, "Bparam"), "{:.0f}", "parameters, billions")
v("l70_gb", m(L70.size_in_bytes(FP16), "GB"), "{:.0f}", "FP16 weights, GB")
v("tp_word", DECODE2.tp_size, "words", "the same, in prose")
v("l70_gb_each", m(r2.weight_bytes, "GB"), "{:.0f}", "weights per H100, GB")
v("decode_ms", m(r2.memory_time, "ms"), "{:.0f}", "weight read per token, ms")
v("decode_tps", 1 / m(r2.memory_time, "s"), "{:.0f}", "tokens per second, at most")

print("== KV cache per token (Scenarios.Gpt2XlKvFullWindow; Models.Language.Llama3_8B)")
v("kv_xl_window_mb", m(KV_XL.total, "MB"), "{:.1f}", "GPT-2 XL KV at its 1,024-token window, MB")
v("l8_dim_raw", L8.hidden_dim, "{:.0f}", "Llama 3.1 8B width, as the source prints it")
v("l8_dim", L8.hidden_dim, "{:,.0f}", "Llama 3.1 8B width")
v("l8_heads", L8.heads, "{:.0f}", "query heads")
v("l8_kv_heads", L8.kv_heads, "{:.0f}", "key-value heads")
v("l8_head_width", L8.head_width, "{:.0f}", "head width")
v("l8_kv_width", L8.kv_width, "{:,.0f}", "key-value width")
v("l8_group_word", L8.queries_per_kv_head, "words", "query heads per key-value head, in prose")

print("== Context memory (Scenarios.Gpt2XlContextMemoryOnH100)")
v("ctx_weights_gb", m(CTX.weights, "GB"), "{:.0f}", "GPT-2 XL FP16 weights, GB")
v("ctx_kv_cross", round(CTX.kv_crossing, -4), "{:,.0f}", "KV passes the weights near, tokens")
v("ctx_kv_cross_k", round(CTX.kv_crossing, -4) / 10 ** 3, "{:.0f}", "the same, thousands")
v("ctx_score_cross", round(CTX.score_crossing, -4), "{:,.0f}", "one layer's scores fill the device near, tokens")
v("ctx_score_cross_k", round(CTX.score_crossing, -4) / 10 ** 3, "{:.0f}", "the same, thousands")
v("ctx_lines_cross", round(CTX.kv_score_crossing, -2), "{:,.0f}", "KV and score lines meet near, tokens")

print("== vLLM's memory layout (Literature.ServingMemoryLayout; Scenarios.Opt13BKvOnA100_40GB)")
v("opt_params_b", m(OPT.parameters, "Bparam"), "{:.0f}", "parameters, billions")
v("a100s_gb", labeled_gb(VLLM.hardware), "{:.0f}", "A100 labeled memory, GB")
v("opt_layers", OPT.layers, "{:.0f}", "layers")
v("opt_d", OPT.kv_width, "{:,.0f}", "key-value width")
v("opt_kv_mb", m(VLLM.per_token, "MB"), "{:.2f}", "KV per token, MB")
v("opt_kv_kib", m(VLLM.per_token, "KiB"), "{:.0f}", "the same, KB (binary)")
v("opt_window", VLLM.tokens, "{:,.0f}", "request length, tokens")
v("opt_req_gb", m(VLLM.total, "GB"), "{:.2f}", "one full-length request, GB")
v("vllm_weights_gb", LAYOUT.ParameterMemoryGb, "{:.0f}", "parameter memory, GB (figure)")
v("vllm_weights_pct", LAYOUT.ParameterSharePct, "{:.0f}", "parameter share, % (figure)")
v("vllm_kv_pct", LAYOUT.KvCacheSharePct, "{:.0f}", "KV share, more than, % (figure)")
v("vllm_requests", VLLM.requests, "{:.0f}", "full-length requests in that share")
v("vllm_requests_word", round(VLLM.requests), "words", "the same, in prose")
v("vllm_half_word", round(VLLM_HALF.requests), "words", "with half the key-value width, in prose")

print("== Sharing keys across heads (Models.Language; Scenarios.Llama3_8BKvFullWindow, Llama2_7BKvAt2048Tokens)")
unshared = L70.model_copy(update={"kv_heads": L70.heads})
v("kv_l7_kb", kv_kb(L7), "{:.0f}", "Llama 2 7B KV per token, KB")
v("kv_l8_kb", kv_kb(L8), "{:.0f}", "Llama 3.1 8B KV per token, KB")
v("kv_l70_kb", kv_kb(L70), "{:.0f}", "Llama 2 70B KV per token, KB")
v("kv_l70_unshared_mb", m(unshared.get_kv_cache_size(1, 1, FP16), "MB"), "{:.1f}", "without sharing, MB")
v("kv_palm_kb", kv_kb(PALM), "{:.0f}", "PaLM 540B KV per token, KB")
v("l7_heads", L7.heads, "{:.0f}", "Llama 2 7B heads")
v("l7_head_width", L7.head_width, "{:.0f}", "Llama 2 7B head width")
v("l70_kv_heads", L70.kv_heads, "{:.0f}", "Llama 2 70B key-value heads")
v("l70_heads", L70.heads, "{:.0f}", "Llama 2 70B query heads")
v("palm_head_width", PALM.head_width, "{:.0f}", "PaLM head width")
v("gqa_ratio", kv_kb(L7) / kv_kb(L8), "{:.0f}", "Llama 2 7B over Llama 3.1 8B, times")
v("gqa_ratio_word", kv_kb(L7) / kv_kb(L8), "words", "Llama 2 7B over Llama 3.1 8B, in prose")
v("l70_ratio_word", L70.queries_per_kv_head, "words", "Llama 2 70B unshared over shared, in prose")
v("l8_window_k", L8.context_length / m(units.Q_(1, "KiB"), "byte"), "{:.0f}", "Llama 3.1 8B window, K tokens (binary thousands)")
v("l8_window_cache_gb", m(KV_L8.total, "GB"), "{:.1f}", "cache at the full window, GB")
v("l8_weights_gb", m(KV_L8.weights, "GB"), "{:.1f}", "16-bit weights, GB")
v("l7_weights_gb", m(KV_L7.weights, "GB"), "{:.0f}", "Llama 2 7B FP16 weights, GB")
v("l7_req_tokens", KV_L7.tokens, "{:,.0f}", "request length, tokens")
v("l7_req_gb", m(KV_L7.total, "GB"), "{:.0f}", "cache per request, GB")

print("== Recommender (Literature.RecommendationSystems; Scenarios.CriteoShapedRequestOnH100)")
v("rec_cycles_pct", RECSYS.FacebookInferenceCyclesPct, "{:.0f}", "share of Facebook AI inference cycles, %")
v("rec_features", GATHER.features, "{:.0f}", "categorical features")
v("rec_width", GATHER.width, "{:.0f}", "row width")
v("rec_row_b", m(GATHER.row_bytes, "byte"), "{:.0f}", "bytes per row")
v("rec_req_kb", m(GATHER.request_bytes, "KB"), "{:.1f}", "request bytes, KB")
v("rec_vectors", GATHER.vectors, "{:.0f}", "vectors in the interaction")
v("rec_pairs", GATHER.pairs, "{:.0f}", "pairs")
v("rec_macs", GATHER.interaction_macs, "{:,.0f}", "interaction MACs")
v("rec_i", m(GATHER.intensity, "flop/byte"), "{:.0f}", "FLOP per gathered byte")
v("rec_hbm_m", m(GATHER.device_rate, "1/s") / 10 ** 6, "{:.0f}", "HBM bound, millions of requests/s")
v("rec_pcie_m", m(GATHER.host_rate, "1/s") / 10 ** 6, "{:.1f}", "PCIe bound, millions of requests/s")
v("rec_ratio", GATHER.bandwidth_ratio, "{:.0f}", "HBM over PCIe, times")
v("pcie_gbs", m(H100.interconnect.bandwidth, "GB/s"), "{:.0f}", "PCIe Gen5 x16, GB/s")

print("== Embedding tables (Scenarios.HundredMillionItemTableOnA100 and variants; Literature)")
v("tbl_rows_m", TABLE.rows / 10 ** 6, "{:.0f}", "rows, millions")
v("tbl_rows_exp", math.log10(TABLE.rows), "{:.0f}", "rows, power of ten")
v("a100_gb", labeled_gb(TABLE.hardware), "{:.0f}", "A100 labeled memory, GB")
v("tbl_gb", m(TABLE.table_bytes, "GB"), "{:.1f}", "one table, GB")
v("tbl_pct", 100 * TABLE.labeled_share, "{:.0f}", "share of the device, %")
v("tbl2_gb", m(TABLES2.total_bytes, "GB"), "{:.1f}", "two tables, GB")
v("users_gb", m(USERS.table_bytes, "GB"), "{:.0f}", "a billion-user table, GB")
dlrm_max = RECSYS.LargestTrainedDlrmParameters
v("dlrm_max_t", dlrm_max / 10 ** 12, "{:.0f}", "largest trained DLRM, trillions of parameters")
v("dlrm_max_tb", m(dlrm_max * FP32, "TB"), "{:.0f}", "its FP32 weights, TB")

print("== Lighthouse table (Models; Scenarios.*Fp32Adam)")
v("lh_resnet_i", intensity(Models.Vision.ResNet50), "{:.1f}", "ResNet-50 FLOP per FP32 weight byte")
v("lh_mobilenet_i", intensity(Models.Vision.MobileNetV2), "{:.1f}", "MobileNetV2")
v("lh_kws_i", intensity(Models.Tiny.DS_CNN), "{:.0f}", "keyword spotter")
v("lh_gpt2_i", intensity(XL), "{:.2f}", "GPT-2 XL per token")
v("lh_resnet_state_mb", m(STATE["resnet"].total, "MB"), "{:.1f}", "ResNet-50 training state, MB")
v("lh_mobilenet_state_mb", m(STATE["mobilenet"].total, "MB"), "{:.1f}", "MobileNetV2, MB")
v("lh_kws_state_mb", m(STATE["kws"].total, "MB"), "{:.1f}", "keyword spotter, MB")
v("lh_gpt2_state_gb", m(STATE["gpt2"].total, "GB"), "{:.0f}", "GPT-2 XL, GB")
v("lh_dlrm_gb", m(DLRM.model_size, "GB"), "{:.0f}", "DLRM tables, GB")
v("lh_dlrm_state_gb", m(DLRM.model_size * STATE["gpt2"].copies, "GB"), "{:.0f}", "DLRM training state, GB")
v("state_bytes", m(STATE["gpt2"].total / STATE["gpt2"].workload.parameters, "byte/param"), "{:.0f}",
  "training state, bytes per parameter")
v("xl_read_ms", m(Scenarios.Gpt2XlDecodeH100.result.memory_time, "ms"), "{:.1f}", "GPT-2 XL weight read per decode step, ms")

print("== One million-token request (Scenarios.Llama3_8BMillionTokenPrompt and variants)")
v("mil_weights_gb", m(KV_L8_1M.weights, "GB"), "{:.1f}", "16-bit weights, GB")
v("mil_kv_b", m(KV_L8_1M.per_token, "byte"), "{:,.0f}", "KV per token, bytes")
v("mil_kv_gb", m(KV_L8_1M.total, "GB"), "{:.0f}", "KV cache, GB")
v("mil_scores_tb", m(SCORES_1M.layer_bytes, "TB"), "{:.0f}", "scores per head per layer, TB")
mant, exp = mantissa_exponent(m(MILLION.prompt_attention_flops, "flop"))
v("mil_flop_mant", mant, "{:.0f}", "attention FLOP per prompt, mantissa")
v("mil_flop_exp", exp, "{:.0f}", "attention FLOP per prompt, power of ten")
v("mil_minutes", m(MILLION.attention_time_at_peak, "minute"), "{:.0f}", "minutes at H100 peak")
v("mil_layers", L8.layers, "{:.0f}", "Llama 3.1 8B layers")

print("== Open questions (vol3/04_context_engineering/data/fiction_livebench_by_length.csv)")
row = load_series("vol3/04_context_engineering/data/fiction_livebench_by_length.csv").where(model="chutes/DeepSeek-R1-0528")[0]
short_col, long_col = "score_400", "score_60k"


def col_tokens(col):
    return float(col.split("_")[1].replace("k", "000"))


v("fl_short_tokens", col_tokens(short_col), "{:,.0f}", "short story, tokens")
v("fl_long_tokens", col_tokens(long_col), "{:,.0f}", "long story, tokens")
v("fl_short", row[short_col], "{:.3f}", "score with the short story")
v("fl_long", row[long_col], "{:.3f}", "score with the long story")


def note_minutes():
    """Title-slide note: total note minutes and the total without [OPTIONAL]
    notes, summed from the deck's own [N min] tags (a check; not recorded)."""
    import re
    tex = (HERE.parent / "06b_nn_architectures.tex").read_text()
    notes = re.findall(r"\\(?:note|mlsysnote)\{\[(\d+(?:\.\d+)?) min\](.*?)\n", tex)
    total = sum(float(n) for n, _ in notes)
    core = sum(float(n) for n, t in notes if "[OPTIONAL" not in t)
    print(f"\n[Title note: lecture length] notes total {total:g} min; without [OPTIONAL] notes {core:g} min")


if __name__ == "__main__":
    note_minutes()
    write(__file__)
