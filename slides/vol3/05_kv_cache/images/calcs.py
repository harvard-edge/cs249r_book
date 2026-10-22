#!/usr/bin/env python3
"""Every number on the Volume III Lecture 5 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (the Llama 3 70B geometry, the DGX H100 node, the
repair request's zones, the scenario sizes, the support profile, and the
Literature results of Kwon et al., Zheng et al., Abhyankar et al., and Xiao
et al.) and the calculations (Scenarios.*, each backed by a solver in
engine/solvers/kv_cache.py, and the Lecture 2 resend scenario the recall
slide repeats); the chapter reads the same scenarios. This file only selects
values, converts units, and formats them; it types no numbers except the
note question's 100 contexts. Each value is recorded with v() under the key
the deck prints as \\val{key}; running this file writes images/numbers.tex.
make_figs.py imports this module for the values and the tables it plots.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios

ZONES = Scenarios.RepairRequestZones
ADMIT = {sc.context_tokens: sc for sc in (Scenarios.Llama3_70BShortContextAdmissionDgxH100,
                                          Scenarios.Llama3_70BAgentContextAdmissionDgxH100,
                                          Scenarios.Llama3_70BLongContextAdmissionDgxH100)}
POOL = Scenarios.Llama3_70BAgentContextAdmissionDgxH100
DESIGN = Scenarios.KvBytesPerTokenByAttentionDesign
PAGED = Scenarios.CappedRequestsPagedAllocation
FORK = Scenarios.ForkedCandidatesPrefixCopy
HIT = Scenarios.InvariantPrefixRebuildDgxH100
CHUNK = Scenarios.ChunkedPrefillAmongDecodes
PAUSE = Scenarios.RepairToolWaitPausedState
SHARE, SHARE90 = Scenarios.RepairBlockSharingResidency, Scenarios.RepairBlockSharingResidencyLongTest
REPAIR, SUPPORT = Scenarios.RepairAgentResidency, Scenarios.SupportAgentResidency
RESEND = Scenarios.CodingAgentResend30Turns
MODEL, NODE = POOL.workload, HIT.node
KWON = load_series("slides/vol1/13b_model_serving/images/data/pagedattention_kv_memory_breakdown.csv").entries
WAITS = load_series("slides/vol3/05_kv_cache/images/data/infercept_interception_times.csv").entries
SINKS = load_series("slides/vol3/05_kv_cache/images/data/streamingllm_attention_sink_perplexity.csv").entries
SGLANG = Literature.PrefixCaching
OPERATOR_TARGET = 100  # note question: resident repair contexts an operator wants

p, pg, h, c, pz = POOL.result, PAGED.result, HIT.result, CHUNK.result, PAUSE.result
sh, rp, sp_ = SHARE.result, REPAIR.result, SUPPORT.result
CONTEXTS = sorted(ADMIT)
FIT = {s: ADMIT[s].result.max_requests for s in CONTEXTS}


def mag(q, unit):
    return q.to(unit).magnitude


print("== Hook, residency, and turn rates (block-sharing scenario)")
v("fit_unshared", sh.fit_unshared, "{:.0f}", "unshared repair slots")
v("fit_shared", sh.fit_shared, "{:.0f}", "slots with the workspace prefix shared")
v("t_gen", mag(SHARE.generation, "s"), "{:.0f}", "generation per turn, s")
v("t_wait", mag(SHARE.wait, "s"), "{:.0f}", "integration-test wait, s")
v("t_cycle", mag(sh.retained_time, "s"), "{:.0f}", "kept residence per turn, s")
v("rate_keep", mag(sh.retained_rate, "1/s"), "{:.2f}", "kept, turns/s")
v("rate_release", mag(sh.released_rate, "1/s"), "{:.1f}", "released, turns/s")
v("shared_rate_keep", mag(sh.shared_retained_rate, "1/s"), "{:.2f}", "shared, kept, turns/s")
v("shared_rate_rel", mag(sh.shared_released_rate, "1/s"), "{:.1f}", "shared, released, turns/s")
v("tax", sh.residency_factor, "{:.0f}", "residency tax 1 + wait / generation")
v("tax_words", sh.residency_factor, "words", "the same, spelled")
v("t_wait_long", mag(SHARE90.wait, "s"), "{:.0f}", "note question: a longer test, s")
v("tax_long", SHARE90.result.residency_factor, "{:.0f}", "its residency tax")
v("keep_to_best", mag(sh.shared_released_rate / sh.retained_rate, ""), "{:.0f}",
  "shared and released / unshared and kept")

print("== Recall (Lecture 2 resend scenario)")
r30 = RESEND.result
v("rs_turns", RESEND.turns, "{:.0f}", "turns")
v("rs_first", RESEND.first_prompt, "{:,.0f}", "first prompt, tokens")
v("rs_delta", RESEND.appended, "{:,.0f}", "appended per turn, tokens")
v("rs_resent", r30.resent_tokens, "{:,.0f}", "resent input, tokens")
v("rs_fresh", r30.reused_prefix_tokens, "{:,.0f}", "fresh prefill with reuse, tokens")
v("rs_reduction", r30.reuse_reduction, "{:.1f}", "resent / fresh")

print("== Context zones (repair request scenario)")
for key, zone in (("z_system", "System prompt"), ("z_tools", "Tool schemas"), ("z_workspace", "Workspace"),
                  ("z_history", "Turn history"), ("z_obs", "Latest observation")):
    v(key, ZONES.zones[zone], "{:,.0f}", f"{zone}, tokens")
v("z_total", ZONES.total_tokens, "{:,.0f}", "request, tokens")
v("z_shared", ZONES.shared_tokens, "{:,.0f}", "invariant shared prefix, tokens")
v("z_shared_pct", 100 * ZONES.shared_share, "{:.1f}", "shared prefix share, %")

print("== One token's KV state (pool scenario, Llama 3 70B registry geometry)")
v("hidden", MODEL.hidden_dim, "{:,.0f}", "hidden size")
v("q_heads", MODEL.heads, "{:.0f}", "query heads")
v("layers", MODEL.layers, "{:.0f}", "layers")
v("kv_heads", MODEL.kv_heads, "{:.0f}", "key-value heads")
v("head_dim", p.head_dim, "{:.0f}", "head dimension")
v("elem_bytes", mag(POOL.kv_bytes_per_elem, "byte"), "{:.0f}", "bytes per 16-bit element")
v("bytes_per_token", mag(p.kv_per_token, "byte"), "{:,.0f}", "KV bytes per token")
v("q_per_kv", MODEL.heads // MODEL.kv_heads, "{:.0f}", "query heads per stored KV head")
mha_bpt = mlsysim.physics.agents.calc_kv_cache_bytes_per_token(MODEL, n_kv_heads=MODEL.heads)
v("mha_bpt_mb", mag(mha_bpt, "MB"), "{:.1f}", "note error: counting 64 heads, MB per token")
v("max_context", MODEL.context_length, "{:,.0f}", "Llama 3.1 context window, tokens")

print("== Resident fit (pool scenario)")
v("n_acc", POOL.accelerators, "{:.0f}", "accelerators per node")
v("acc_gib", mag(POOL.hbm_per_accelerator, "GiB"), "{:.0f}", "HBM per accelerator, GiB")
v("acc_gb", mag(POOL.hbm_per_accelerator, "GB"), "{:.1f}", "HBM per accelerator, GB")
v("hbm_gb", mag(p.hbm_total, "GB"), "{:.1f}", "node HBM, GB")
v("params_b", mag(MODEL.parameters, "count") / 10 ** 9, "{:.1f}", "parameters, billions")
v("weights_gb", mag(p.weights, "GB"), "{:.1f}", "16-bit weights, GB")
v("reserve_gb", mag(POOL.reserve, "GB"), "{:.0f}", "runtime reserve, GB")
v("pool_gb", mag(p.kv_budget, "GB"), "{:.1f}", "KV pool, GB")
v("pool_gb_whole", mag(p.kv_budget, "GB"), "{:.0f}", "KV pool, GB (whole)")
for s in CONTEXTS:
    v(f"ctx_{s}", s, "{:,.0f}", "retained tokens")
    v(f"gb_{s}", mag(ADMIT[s].result.kv_per_request, "GB"), "{:.1f}", "one context, GB")
    v(f"fit_{s}", FIT[s], "{:.0f}", "contexts that fit")
v("ctx_for_target", mag(p.kv_budget / OPERATOR_TARGET / p.kv_per_token, ""), "{:,.0f}",
  f"note question: longest context for {OPERATOR_TARGET} residents, tokens")
v("operator_target", OPERATOR_TARGET, "{:.0f}", "note question: resident contexts")
v("pool_fit_wrong_weights", math.floor(mag(p.hbm_total / p.kv_per_request, "")), "{:.0f}",
  "note error: forgetting the weights")
decimal_hbm = POOL.accelerators * mag(POOL.hbm_per_accelerator, "GiB") * mlsysim.core.units.ureg.GB
v("pool_fit_decimal", math.floor(mag((decimal_hbm - p.weights - POOL.reserve) / p.kv_per_request, "")),
  "{:.0f}", "note error: reading 80 GiB as 80 GB")

print("== Attention design (design scenario)")
d = DESIGN.result
DESIGNS = [(m.name, mag(d.per_token[m.name], "kB"), d.design[m.name]) for m in DESIGN.models]
v("kb_mha", mag(d.per_token[DESIGN.mha.name], "kB"), "{:,.0f}", "GPT-3 KB per token")
v("kb_mla", mag(d.per_token[DESIGN.mla.name], "kB"), "{:.1f}", "DeepSeek-V3 KB per token")
v("mha_heads", DESIGN.mha.heads, "{:.0f}", "GPT-3 query heads")
v("mha_over_mla", d.mha_over_mla, "{:.0f}", "GPT-3 / DeepSeek-V3")
v("gqa_over_mla", d.gqa_over_mla, "{:.1f}", "Llama 70B / DeepSeek-V3")
v("gqa_session_gb", mag(d.gqa_session, "GB"), "{:.1f}", "Llama 70B, one long session, GB")
v("mla_session_gb", mag(d.mla_session, "GB"), "{:.1f}", "DeepSeek-V3, one long session, GB")

print("== Kwon et al. 2023, Figure 2 (slides/vol1/13b_model_serving/images/data/pagedattention_kv_memory_breakdown.csv)")
kw = {e["system"]: e for e in KWON}
v("kwon_max", kw["Orca (Max)"]["token_states"], "{:.1f}", "Orca Max token states, %")
v("kwon_pow2", kw["Orca (Pow2)"]["token_states"], "{:.1f}", "Orca Pow2 token states, %")
v("kwon_oracle", kw["Orca (Oracle)"]["token_states"], "{:.1f}", "Orca Oracle token states, %")
v("kwon_best", kw["Orca (Oracle)"]["token_states"], "{:.0f}", "best pre-paging share, %")
v("kwon_vllm", kw["vLLM"]["token_states"], "{:.1f}", "vLLM token states, %")
v("kwon_oracle_waste", 100 - kw["Orca (Oracle)"]["token_states"], "{:.0f}", "Oracle: not token states, %")
VLLM_OTHER = 100 - kw["vLLM"]["token_states"]  # the bar's remainder, derived

print("== Static reservation, blocks, paging (paged-allocation scenario)")
v("cap", PAGED.cap, "{:,.0f}", "admission cap, tokens")
v("mean_final", PAGED.mean_final, "{:,.0f}", "mean final length, tokens")
v("populated_pct", 100 * pg.final_length_efficiency, "{:.2f}", "reservation populated at completion, %")
v("reserved_unused", PAGED.cap - PAGED.mean_final, "{:,.0f}", "slots a mean request never fills")
v("reservations", pg.reservations, "{:.0f}", "full-cap reservations in the pool")
v("block_small", PAGED.small_block, "{:.0f}", "small block, tokens")
v("block_large", PAGED.large_block, "{:.0f}", "large block, tokens")
v("tail_small", pg.small_tail_tokens, "{:.0f}", "worst tail, small blocks, slots")
v("tail_small_mb", mag(pg.small_tail, "MB"), "{:.2f}", "worst tail, small blocks, MB")
v("tail_large_mb", mag(pg.large_tail, "MB"), "{:.2f}", "worst tail, large blocks, MB")
v("tail_share_pct", 100 * pg.large_tail_share, "{:.1f}", "worst large tail / cap, %")
v("frag_small_pct", 100 * pg.small_waste, "{:.1f}", "expected waste, small blocks, %")
v("frag_large_pct", 100 * pg.large_waste, "{:.1f}", "expected waste, large blocks, %")
v("fit_paged", pg.paged_fit, "{:.0f}", "final-length snapshots, small blocks")
v("paged_gain", pg.paged_gain, "{:.0f}", "paged / reserved")
v("fork_prefix", FORK.prefix_tokens, "{:,.0f}", "forked prefix, tokens")
v("fork_branches_words", FORK.branches, "words", "candidates")
v("fork_gb", mag(FORK.copied, "GB"), "{:.1f}", "copied without sharing, GB")

print("== Prefix hit (rebuild scenario) and SGLang (Literature.PrefixCaching)")
v("prefix_tokens", HIT.prefix_tokens, "{:,.0f}", "invariant prefix, tokens")
v("suffix_tokens", HIT.suffix_tokens, "{:.0f}", "suffix after a hit, tokens")
v("prefix_gb", mag(h.state, "GB"), "{:.2f}", "prefix KV state, GB")
v("util_pct", 100 * HIT.utilization, "{:.0f}", "sustained share of 16-bit peak, %")
v("rate_tflops", mag(h.sustained_rate, "TFLOPs/s"), "{:,.0f}", "sustained rate, trillion ops/s")
v("linear_t", mag(h.linear_ops, "flop") / 10 ** 12, "{:.0f}", "linear ops, trillions")
v("attn_t", mag(h.attention_ops, "flop") / 10 ** 12, "{:.0f}", "attention ops, trillions")
v("t_rebuild", mag(h.rebuild_time, "ms"), "{:.0f}", "rebuild on a miss, ms")
v("t_floor", mag(h.weight_read_time, "ms"), "{:.0f}", "weight-read floor, ms")
v("sg_llava", SGLANG.SglangLlavaNext34BHitRate, "{:.1f}", "LLaVA-Next-34B hit rate, %")
v("sg_vicuna", SGLANG.SglangVicuna33BHitRate, "{:.1f}", "Vicuna-33B hit rate, %")
v("sg_ttft", SGLANG.SglangVicuna33BFirstTokenSpeedup, "{:.1f}", "first-token latency reduction")

print("== Chunked prefill (chunked scenario)")
v("decodes", CHUNK.decodes, "{:.0f}", "ongoing decodes")
v("prompt_tokens", CHUNK.prompt_tokens, "{:,.0f}", "new prompt, tokens")
v("chunk_tokens", CHUNK.chunk_tokens, "{:.0f}", "chunk, tokens")
v("iter_tokens", CHUNK.chunk_tokens + CHUNK.decodes, "{:.0f}", "positions per chunked iteration")
v("n_chunks", c.chunks, "{:.0f}", "chunks")
v("t_decode", mag(c.decode_step, "ms"), "{:.0f}", "decode-only step, ms")
v("mono_prefill", mag(c.monolithic_prefill, "ms"), "{:.0f}", "monolithic prefill, ms")
v("gap_mono", mag(c.monolithic_gap, "ms"), "{:.0f}", "decode gap, monolithic, ms")
v("t_iter", mag(c.chunked_iteration, "ms"), "{:.0f}", "chunked iteration, ms")
v("chunk_done", mag(c.chunked_done, "ms"), "{:.0f}", "prompt done, chunked, ms")

print("== Tool waits (slides/vol3/05_kv_cache/images/data/infercept_interception_times.csv, ToolInterception)")
wt = {e["workload"]: e["mean_interception_s"] for e in WAITS}
v("w_calc_ms", 10 ** 3 * wt["Math (calculator)"], "{:.2f}", "calculator, ms")
v("w_venv_ms", 10 ** 3 * wt["Virtual env."], "{:.0f}", "virtual environment, ms")
v("w_qa_ms", 10 ** 3 * wt["QA (Wikipedia)"], "{:.0f}", "Wikipedia QA, ms")
v("w_tts_s", wt["Text to speech"], "{:.0f}", "text to speech, s")
v("w_image_s", wt["Image generation"], "{:.0f}", "image generation, s")
v("w_chat_s", wt["Chatbot (human)"], "{:.1f}", "human chat reply, s")
span = max(wt.values()) / min(wt.values())
v("wait_orders_words", math.floor(math.log10(span)), "words", "orders of magnitude spanned")
v("recompute_low", Literature.ToolInterception.RecomputeShareLow, "{:.0f}", "recompute share, low, %")
v("recompute_high", Literature.ToolInterception.RecomputeShareHigh, "{:.0f}", "recompute share, high, %")

print("== Paused state (paused-state scenario)")
v("host_tib", mag(NODE.host_memory, "TiB"), "{:.0f}", "host memory, TiB (sold as TB)")
v("state_gb", mag(pz.state, "GB"), "{:.2f}", "paused state, GB")
v("link_per_acc", mag(NODE.accelerator.interconnect.bandwidth, "GB/s"), "{:.0f}", "host link per accelerator, GB/s")
v("link_gbs", mag(pz.link, "GB/s"), "{:.0f}", "aggregate host links, GB/s")
v("t_reload", mag(pz.copy_in, "ms"), "{:.0f}", "reload, ms")
v("round_trip", mag(pz.round_trip, "ms"), "{:.0f}", "copy out and back, ms")
v("t_recomp", mag(pz.recompute, "s"), "{:.1f}", "recompute, s")
v("reload_ratio", pz.recompute_over_reload, "{:.0f}", "recompute / reload")
v("reload_ratio_q", pz.recompute_over_reload_derated, "{:.0f}", "at a quarter of link bandwidth")
v("free_window", mag(pz.released_on_demand, "s"), "{:.2f}", "memory freed during the wait, s")
v("held_gbs", mag(pz.held_occupancy, "GB*s"), "{:.0f}", "memory time held if kept, GB-s")
v("returns", PAUSE.returns, "{:.0f}", "correlated returns")
v("burst_gb", mag(pz.burst, "GB"), "{:.0f}", "returning state, GB")
v("burst_s", mag(pz.burst_time, "s"), "{:.2f}", "last reload done, s")
v("burst_ratio", pz.burst_slowdown, "{:.0f}", "last reload / one reload")

print("== StreamingLLM (slides/vol3/05_kv_cache/images/data/streamingllm_attention_sink_perplexity.csv)")
s13, s7 = ({e["model"]: e for e in SINKS}[k] for k in ("Llama-2-13B", "Llama-2-7B"))
v("sink_tokens", s13["sink_tokens"], "{:.0f}", "initial tokens kept")
v("s13_cache", s13["cache_tokens"], "{:,.0f}", "Llama-2-13B cache, tokens")
v("s13_window", s13["window_perplexity"], "{:,.0f}", "Llama-2-13B, window only")
v("s13_sink", s13["sink_perplexity"], "{:.2f}", "Llama-2-13B, with sinks")
v("s7_window", s7["window_perplexity"], "{:,.0f}", "Llama-2-7B, window only")
v("s7_sink", s7["sink_perplexity"], "{:.2f}", "Llama-2-7B, with sinks")

print("== Block sharing (block-sharing scenario)")
v("output_tokens", SHARE.output, "{:,.0f}", "output allowance, tokens")
v("common_tokens", SHARE.common, "{:,.0f}", "shared workspace prefix, tokens")
v("block_tokens", SHARE.block_tokens, "{:.0f}", "block, tokens")
v("headroom_pct", 100 * SHARE.headroom, "{:.0f}", "headroom, %")
v("block_mb", mag(sh.block_bytes, "MB"), "{:.2f}", "block, MB")
v("usable_blocks", sh.usable_blocks, "{:,.0f}", "usable blocks")
v("unshared_blocks", sh.unshared_blocks, "{:,.0f}", "blocks per unshared request")
v("common_blocks", sh.common_blocks, "{:,.0f}", "shared prefix blocks")
v("private_blocks", sh.private_blocks, "{:,.0f}", "private blocks per request")
v("shared_frac_pct", 100 * sh.shared_fraction, "{:.1f}", "shared fraction, %")
v("share_gain", sh.share_gain, "{:.1f}", "shared fit / unshared fit")
v("share_bound", sh.share_bound, "{:.1f}", "1 / (1 - f)")

print("== Repair and support archetypes (archetype scenarios)")
v("arch_prefix", REPAIR.shared_prefix, "{:,.0f}", "system prompt and tool schemas, tokens")
v("repair_frac_pct", 100 * rp.shared_fraction, "{:.1f}", "repair shared fraction, %")
v("support_frac_pct", 100 * sp_.shared_fraction, "{:.0f}", "support shared fraction, %")
v("repair_unsh", rp.fit_unshared, "{:.0f}", "repair unshared fit")
v("repair_sh", rp.fit_shared, "{:.0f}", "repair shared fit")
v("repair_gain", rp.share_gain, "{:.2f}", "repair sharing gain")
v("support_ctx", SUPPORT.context_tokens, "{:,.0f}", "support session, tokens")
v("support_unsh", sp_.fit_unshared, "{:.0f}", "support unshared fit")
v("support_sh", sp_.fit_shared, "{:.0f}", "support shared fit")
v("support_gain_words", sp_.share_gain, "words", "support sharing gain")
v("support_gain", sp_.share_gain, "{:.1f}", "support sharing gain (figure)")
v("support_gen", mag(SUPPORT.generation, "s"), "{:.0f}", "support generation, s")
v("support_wait", mag(SUPPORT.wait, "s"), "{:.0f}", "customer reply, s")
v("support_tax", sp_.residency_factor, "{:.0f}", "support residency tax")
v("repair_reload", mag(rp.reload, "ms"), "{:.0f}", "repair reload, ms")
v("support_reload", mag(sp_.reload, "ms"), "{:.0f}", "support reload, ms")
v("repair_recomp", mag(rp.recompute, "s"), "{:.1f}", "repair recompute, s")
v("support_recomp", mag(sp_.recompute, "s"), "{:.2f}", "support recompute, s")
v("repair_recomp_pct", 100 * rp.recompute_share, "{:.1f}", "repair recompute / turn cycle, %")

if __name__ == "__main__":
    write(__file__)
