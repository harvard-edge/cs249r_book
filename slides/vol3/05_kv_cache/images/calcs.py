#!/usr/bin/env python3
"""Every derived number in Lecture 5 (KV Cache), computed once.

Inputs come from MLSysIM (the registry the chapter's LEGO cells use) or are
typed with their source: "KV" is the rendered chapter, KV Cache Management
(Volume III, dev text read 2026-10-18), with its section, notebook (NM), or
table. Outside measurements name the paper and its table or figure, read
from the paper itself. Each block prints the value with the label the slide
uses. Run from anywhere:

    python3 calcs.py

make_figs.py imports the values (from calcs import V).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))

from mlsysim import Hardware, Literature, Models, Systems  # noqa: E402
from mlsysim.core.units import BYTES_FP16  # noqa: E402
from mlsysim.physics.memory import calc_expected_paged_kv_tokens  # noqa: E402

V = {}


def out(key, value, label):
    V[key] = value
    if __name__ == "__main__":
        print(f"{key:28s} {label}")


GB = 1e9
MODEL = Models.Language.Llama3_70B          # KV §5.2 geometry (registry name Llama-3.1-70B)
NODE = Systems.Nodes.DGX_H100               # KV §5.2 reference node
N_PARAMS = MODEL.parameters.to("count").magnitude

# -- Slide: one token's KV state (KV eq. 1, NM 5.1) --------------------------
# Inputs: 80 layers, 8 KV heads, head dim 128 (8,192 / 64), 2 bytes (FP16);
# the same values appear in the model's public config.json (shown on slide).
L, H_KV, D_HEAD, E = 80, 8, 8192 // 64, 2
bpt = 2 * L * H_KV * D_HEAD * E
reg_bpt = MODEL.get_kv_cache_size(seq_len=1, batch_size=1, precision=BYTES_FP16).to("byte").magnitude
assert bpt == reg_bpt == 327_680
out("bytes_per_token", bpt, f"2 x 80 x 8 x 128 x 2 = {bpt:,} bytes per token")

# -- Slide: resident fit (NM 5.1, table 1) -----------------------------------
hbm = NODE.accelerators_per_node * NODE.accelerator.memory.capacity.to("byte").magnitude / GB
weights = MODEL.size_in_bytes(BYTES_FP16).to("byte").magnitude / GB
reserve = 20.0                                   # KV NM 5.1: stipulated runtime reserve, GB
pool = hbm - weights - reserve
out("hbm_gb", hbm, f"8 x 80 GiB = {hbm:.1f} GB")
out("weights_gb", weights, f"weights (70.6B x 2 B) = {weights:.1f} GB")
out("pool_gb", pool, f"KV pool = {hbm:.1f} - {weights:.1f} - 20 = {pool:.1f} GB")
for s in (8_192, 32_768, 131_072):
    gb = s * bpt / GB
    out(f"fit_{s}", math.floor(pool / gb), f"{s:,} tokens: {gb:.1f} GB, fit {math.floor(pool / gb)}")
    out(f"gb_{s}", gb, "")
fit_curve_tokens = [2**k for k in range(11, 18)]
out("fit_curve", [(s, pool / (s * bpt / GB)) for s in fit_curve_tokens], "fit curve (pool / footprint)")

# -- Slide: attention design (KV figure 2, MLSysIM registry geometries) ------
LM = Models.Language
designs = []
for m in [LM.GPT3, LM.Llama3_405B, LM.Llama3_70B, LM.Llama2_70B, LM.Mixtral_8x7B, LM.DeepSeek_V3]:
    kb = m.get_kv_cache_size(seq_len=1, batch_size=1, precision=BYTES_FP16).to("kilobyte").magnitude
    kind = "MLA" if m.kv_lora_rank else ("GQA" if m.kv_heads and m.kv_heads < m.heads else "MHA")
    designs.append((m.name, kb, kind))
    out(f"kb_{m.name}", kb, f"{m.name} ({kind}): {kb:,.1f} KB per token")
out("designs", designs, "")
gpt3_kb = designs[0][1]
ds_kb = designs[-1][1]
out("mha_over_mla", gpt3_kb / ds_kb, f"GPT-3 / DeepSeek-V3 = {gpt3_kb / ds_kb:.0f}x")
out("gqa_over_mla", 327.68 / ds_kb, f"Llama 70B / DeepSeek-V3 = {327.68 / ds_kb:.1f}x")
out("ds_131k_gb", ds_kb * 1e3 * 131_072 / GB, f"DeepSeek-V3 at 131,072 tokens: {ds_kb * 1e3 * 131_072 / GB:.1f} GB")

# -- Slide: static reservation (KV §5.3.1) -----------------------------------
CAP, MEAN = 32_768, 2_048                       # KV §5.3.1 scenario cap and mean final length
out("populated", MEAN / CAP, f"populated at completion = 2,048 / 32,768 = {100 * MEAN / CAP:.2f}%")

# -- Slide: Kwon et al. 2023, Figure 2 (read from the paper) -----------------
KWON = [("Orca\n(Max)", 20.4, 13.3, 57.3, 8.9), ("Orca\n(Pow2)", 26.8, 17.9, 13.6, 41.6),
        ("Orca\n(Oracle)", 38.2, 25.2, 0.0, 36.6), ("vLLM", 96.3, None, None, None)]
out("kwon", KWON, "token states / reservation / internal / external+others, percent")
out("vllm_other", 100 - 96.3, f"vLLM non-token share (derived) = {100 - 96.3:.1f}%")

# -- Slide: block size and tail waste (NM 5.2) -------------------------------
for b in (16, 64):
    out(f"block_mb_{b}", b * bpt / 1e6, f"{b}-token block = {b * bpt / 1e6:.2f} MB")
    out(f"tail_mb_{b}", (b - 1) * bpt / 1e6, f"max tail {b - 1} slots = {(b - 1) * bpt / 1e6:.2f} MB")
out("tail_share_64", 63 / 32_768, f"64-token tail / 32,768 tokens = {100 * 63 / 32768:.1f}%")

# -- Slide: paged snapshots (KV §5.4.2) --------------------------------------
alloc16, frag16 = calc_expected_paged_kv_tokens(MEAN, CAP, 16)
alloc64, frag64 = calc_expected_paged_kv_tokens(MEAN, CAP, 64)
fit_paged = int(pool * GB / (alloc16 * bpt))
out("frag16", frag16, f"16-token blocks waste {100 * frag16:.1f}%")
out("frag64", frag64, f"64-token blocks waste {100 * frag64:.1f}%")
out("fit_paged", fit_paged, f"paged final-length snapshots = {fit_paged}")
out("paged_gain", fit_paged / V["fit_32768"], f"paged / static = {fit_paged / V['fit_32768']:.1f}x")

# -- Slide: block table lookup (KV §5.4.1) -----------------------------------
pos, bs, table = 9, 4, [3, 7, 9]
out("lookup", (pos // bs, table[pos // bs], pos % bs), f"position 9: logical {pos // bs}, physical {table[pos // bs]}, slot {pos % bs}")

# -- Slide: fork without sharing (KV §5.4.3) ---------------------------------
fork_gb = 4 * 8_192 * bpt / GB
out("fork_gb", fork_gb, f"4 copies x 8,192 tokens = {fork_gb:.1f} GB copied")

# -- Slide: what a prefix hit avoids (KV §5.5.4) -----------------------------
rate = NODE.accelerators_per_node * NODE.accelerator.compute.peak_flops.to("flop/s").magnitude \
    * Literature.Training.MfuHigh * 0.9          # KV §5.5.4 sustained rate
util = Literature.Training.MfuHigh * 0.9
prefix = 4_096
linear = 2 * N_PARAMS * prefix
attn = 2 * MODEL.layers * MODEL.hidden_dim * prefix**2  # KV, appendix prefill-long term
t_rebuild = (linear + attn) / rate * 1e3
node_bw = NODE.accelerators_per_node * NODE.accelerator.memory.bandwidth.to("byte/s").magnitude
t_floor = weights * GB / node_bw * 1e3
out("rate_tflops", rate / 1e12, f"sustained rate {100 * util:.0f}% of peak = {rate / 1e12:,.0f} TFLOP/s")
out("prefix_gb", prefix * bpt / GB, f"4,096-token prefix state = {prefix * bpt / GB:.2f} GB")
out("linear_t", linear / 1e12, f"linear ops = {linear / 1e12:.0f} trillion")
out("attn_t", attn / 1e12, f"attention ops = {attn / 1e12:.0f} trillion")
out("t_rebuild", t_rebuild, f"rebuild on a miss = {t_rebuild:.0f} ms")
out("t_floor", t_floor, f"weight-read floor (hit, 32-token suffix) = {t_floor:.1f} ms")
out("hit_ratio", t_rebuild / t_floor, f"miss / hit = {t_rebuild / t_floor:.0f}x")

# -- Slide: hit rate example (KV §5.5.4, eq. 2) ------------------------------
out("hit_full_block", (10 // 4) * 4, "10 matched tokens, 4-token full blocks only: reuse 8, prefill 2")

# -- Slide: SGLang production hit rates (Zheng et al. 2024, §6, read) --------
out("sglang", [("LLaVA-Next-34B", 52.4), ("Vicuna-33B", 74.1)], "one month in Chatbot Arena; TTFT 1.7x lower (Vicuna)")

# -- Slide: chunked prefill (KV §5.6.2) --------------------------------------
def step_ms(n):
    return max(t_floor, 2 * N_PARAMS * n / rate * 1e3)


decodes, prompt, chunk = 32, 4_096, 512
mono_prefill = 2 * N_PARAMS * prompt / rate * 1e3
gap_mono = mono_prefill + step_ms(decodes)
t_iter = step_ms(chunk + decodes)
n_iter = prompt // chunk
out("mono_prefill", mono_prefill, f"monolithic prefill arithmetic = {mono_prefill:.0f} ms")
out("gap_mono", gap_mono, f"decode gap, monolithic = {gap_mono:.0f} ms")
out("t_iter", t_iter, f"chunked iteration (512 + 32 positions) = {t_iter:.1f} ms")
out("chunk_done", n_iter * t_iter, f"prompt done after {n_iter} chunks = {n_iter * t_iter:.0f} ms")
out("gap_ratio", gap_mono / t_iter, f"gap shrinks {gap_mono / t_iter:.1f}x")

# -- Slide: reload or recompute (KV §5.7.3, eq. 4) ---------------------------
state = 32_768 * bpt / GB
link = NODE.accelerators_per_node * NODE.accelerator.interconnect.bandwidth.to("byte/s").magnitude / GB
t_reload = state / link * 1e3
t_recomp = 2 * N_PARAMS * 32_768 / rate
out("state_gb", state, f"32,768-token state = {state:.2f} GB")
out("link_gbs", link, f"8 x 64 GB/s host links = {link:.0f} GB/s")
out("t_reload", t_reload, f"reload = {t_reload:.0f} ms")
out("t_recomp", t_recomp, f"recompute = {t_recomp:.1f} s")
out("reload_ratio", t_recomp * 1e3 / t_reload, f"recompute / reload = {t_recomp * 1e3 / t_reload:.0f}x")
out("reload_ratio_q", t_recomp * 1e3 / t_reload / 4, f"at a quarter of link bandwidth = {t_recomp * 1e3 / t_reload / 4:.1f}x")
out("round_trip", 2 * t_reload, f"round trip = {2 * t_reload:.0f} ms")
out("held_gbs", state * 45, f"kept through 45 s: {state * 45:.2f} GB.s")
out("free_window", 45 - t_reload / 1e3, f"freed for {45 - t_reload / 1e3:.2f} s")
out("drop_share", t_recomp / 50, f"drop adds {100 * t_recomp / 50:.1f}% to a 50 s turn")

# -- Slide: return burst (KV §5.7.4) -----------------------------------------
burst_gb = 32 * state
out("burst_gb", burst_gb, f"32 returns x {state:.2f} GB = {burst_gb:.0f} GB")
out("burst_s", burst_gb / link, f"reload all at full link rate = {burst_gb / link:.2f} s")
out("burst_ratio", (burst_gb / link) / (t_reload / 1e3), f"last reload waits {(burst_gb / link) / (t_reload / 1e3):.0f}x the isolated reload")

# -- Slide: InferCept Table 1 (Abhyankar et al. 2024, read) ------------------
INFERCEPT = [("Math (calculator)", 9e-5), ("Virtual env.", 0.09), ("QA (Wikipedia)", 0.69),
             ("Text to speech", 17.24), ("Image generation", 20.03), ("Chatbot (human)", 28.6)]
out("infercept", INFERCEPT, "mean interception time, s (Table 1)")
out("infercept_span", 28.6 / 9e-5, f"chatbot / calculator = {28.6 / 9e-5:,.0f}x")

# -- Slide: StreamingLLM (Xiao et al. 2024, Tables 1 and 2, read) ------------
STREAM = [("Llama-2-13B", 5158.07, 5.40, "0+1024 vs 4+1020"), ("Llama-2-7B", 3359.95, 9.59, "0+4096 vs 4+4092")]
out("stream", STREAM, "perplexity, window vs window plus 4 initial tokens")
out("stream_ratio", 5158.07 / 5.40, f"Llama-2-13B perplexity ratio = {5158.07 / 5.40:,.0f}x")

# -- Slide: Little's law and the residency tax (NM 5.3, NM 5.4) --------------
head = 0.10                                     # NM 5.3: withheld headroom
blk = 16
usable = math.floor((1 - head) * pool * GB / (blk * bpt))
req_blocks = math.ceil((32_768 + 2_048) / blk)
fit_unshared = usable // req_blocks
pre_blocks = 24_576 // blk
priv_blocks = math.ceil((32_768 + 2_048 - 24_576) / blk)
fit_shared = (usable - pre_blocks) // priv_blocks
f = pre_blocks / req_blocks
out("usable_blocks", usable, f"usable blocks = {usable:,}")
out("req_blocks", req_blocks, f"unshared request = {req_blocks:,} blocks ({req_blocks * blk * bpt / GB:.2f} GB)")
out("fit_unshared", fit_unshared, f"unshared fit = {fit_unshared}")
out("pre_blocks", pre_blocks, f"shared prefix = {pre_blocks:,} blocks ({pre_blocks * blk * bpt / GB:.2f} GB)")
out("priv_blocks", priv_blocks, f"private per request = {priv_blocks} blocks ({priv_blocks * blk * bpt / GB:.2f} GB)")
out("fit_shared", fit_shared, f"shared fit = {fit_shared}")
out("shared_frac", f, f"shared fraction = {100 * f:.1f}%")
out("share_gain", fit_shared / fit_unshared, f"gain = {fit_shared / fit_unshared:.1f}x; limit 1/(1-f) = {1 / (1 - f):.1f}x")
t_model, t_tool = 5.0, 45.0                      # NM 5.4
out("rate_keep", fit_unshared / (t_model + t_tool), f"keep: 41 / 50 s = {fit_unshared / (t_model + t_tool):.2f} turns/s")
out("rate_release", fit_unshared / t_model, f"release: 41 / 5 s = {fit_unshared / t_model:.1f} turns/s")
out("tax", 1 + t_tool / t_model, f"residency tax = 1 + 45/5 = {1 + t_tool / t_model:.0f}x")
out("gain_curve", [(x / 100, 1 / (1 - x / 100)) for x in range(0, 91)], "1/(1-f) curve")

# -- Slide: archetypes (Lighthouse 5.1, table 5) -----------------------------
shared = 6_144
for name, s, tm, tt in (("repair", 32_768, 5.0, 45.0), ("support", 8_192, 2.0, 20.0)):
    fs = shared / s
    unsh = math.floor(pool / (s * bpt / GB))
    sh = math.floor((pool - shared * bpt / GB) / ((s - shared) * bpt / GB))
    rl = s * bpt / GB / link * 1e3
    rc = 2 * N_PARAMS * s / rate
    out(f"{name}_frac", fs, f"{name}: shared fraction {100 * fs:.1f}%")
    out(f"{name}_unsh", unsh, f"{name}: unshared {unsh}")
    out(f"{name}_sh", sh, f"{name}: shared {sh} ({sh / unsh:.2f}x)")
    out(f"{name}_tax", 1 + tt / tm, f"{name}: tax {1 + tt / tm:.0f}x")
    out(f"{name}_reload", rl, f"{name}: reload {rl:.0f} ms, recompute {rc:.2f} s")

# -- Slide: why the cache matters across turns (Lecture 2, NM 2.2) -----------
resent = sum(8_000 + 1_500 * t for t in range(30))
fresh = 8_000 + 29 * 1_500
out("resent", resent, f"30 turns resent = {resent:,} input tokens")
out("fresh", fresh, f"with prefix reuse = {fresh:,} fresh tokens ({resent / fresh:.1f}x)")

# -- Slide: synthesis, support agent estimate --------------------------------
out("support_rate_keep", V["support_sh"] / 22, f"support, shared, keep: {V['support_sh']} / 22 s = {V['support_sh'] / 22:.0f} turns/s")
out("support_rate_rel", V["support_sh"] / 2, f"support, shared, release: {V['support_sh']} / 2 s = {V['support_sh'] / 2:.0f} turns/s")
out("shared_rate_keep", fit_shared / 50, f"repair, workspace shared, keep: {fit_shared} / 50 s = {fit_shared / 50:.2f} turns/s")
out("shared_rate_rel", fit_shared / 5, f"repair, workspace shared, release: {fit_shared} / 5 s = {fit_shared / 5:.1f} turns/s")

# -- Numbers quoted in speaker notes (derived) -------------------------------
out("keep_to_best", V["shared_rate_rel"] / V["rate_keep"], f"synthesis: 27.6 / 0.82 = {V['shared_rate_rel'] / V['rate_keep']:.0f}x")
out("ctx_for_100", V["pool_gb"] * GB / 100 / bpt, f"fit note: 100 contexts allow {V['pool_gb'] * GB / 100 / bpt:,.0f} tokens each")
out("mha_bpt", 2 * 80 * 64 * 128 * 2 / 1e6, f"config note: with 64 KV heads, {2 * 80 * 64 * 128 * 2 / 1e6:.1f} MB per token")
out("q_per_kv", 64 // 8, "config note: 64 query heads / 8 KV heads = 8 per group")
out("tax_90", 1 + 90 / 5, f"Little note: 90 s test, tax = {1 + 90 / 5:.0f}")
out("oracle_waste", 100 - 38.2, f"Kwon note: Oracle wastes {100 - 38.2:.1f}%")
out("pool_fit_wrong_weights", math.floor(hbm / V["gb_32768"]), f"estimate note error: forgetting weights gives {math.floor(hbm / V['gb_32768'])}")
out("pool_fit_decimal_80", math.floor((640 - V["weights_gb"] - 20) / V["gb_32768"]), f"estimate note error: 80 GB decimal gives {math.floor((640 - V['weights_gb'] - 20) / V['gb_32768'])}")
