#!/usr/bin/env python3
"""Every derived number in Lecture 2 (The Foundation Model), computed from sourced inputs.

Run from this folder: python3 calcs.py
Each line prints a slide label and the value the deck shows. Inputs carry their source.
"Ch." is the rendered chapter, The Foundation Model (Volume III, Chapter 2).
"run" is data/model_runs.json, written by run_model.py (Qwen2.5-Coder-0.5B-Instruct on a
laptop CPU). "Azure" is data/azure_llm_code_2023.csv (Azure LLM inference trace 2023, code).
"""

import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).parent
RUN = json.loads((HERE / "data" / "model_runs.json").read_text())


def show(label, value):
    print(f"{label:<62} {value}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ------------------------------------------------------------- hook (Ch. War Story 2.1)
# Inputs typed from Anthropic 2025 postmortem and Ch. WS 2.1: 0.8% initial, 16% peak, ~30% of
# coding-agent users. No derived numbers on the hook slide.

# ------------------------------------------------------------- recall (Ch. 1 NM 1.1, NM 1.2)
eps, H = 0.01, 30                 # Lighthouse 1.1: 30-step budget; 99% per step (NM 1.1)
show("recall: (1-eps)^30 at 99% per step", pct((1 - eps) ** H))
show("recall: (1-eps)^10", pct((1 - eps) ** 10))
show("recall: (1-eps)^100", pct((1 - eps) ** 100))
attempt, p_acc = 2.10, 0.40       # NM 1.2
show("recall: C_eff = 2.10 / 0.40 ($)", f"{attempt / p_acc:.2f}")

# ------------------------------------------------------------- wire request (run)
vis = RUN["visible_tokens"]["system"] + RUN["visible_tokens"]["user"]
wire = RUN["wire_tokens"]
show("wire: visible message tokens", vis)
show("wire: serialized request tokens", wire)
show("wire: framing tokens (wire - visible)", wire - vis)
show("wire: wire / visible", f"{wire / vis:.1f}x")
show("wire: vocab size / context positions", f"{RUN['vocab_size']:,} / {RUN['max_position_embeddings']:,}")

# ------------------------------------------------------------- tokenizer (run, cl100k_base)
show("tok: flat line tokens", len(RUN["cl100k_flat"]["ids"]))
show("tok: indented line tokens", len(RUN["cl100k_indented"]["ids"]))
shared = len(set(RUN["cl100k_flat"]["ids"]) & set(RUN["cl100k_indented"]["ids"]))
show("tok: ids shared by both lines", shared)
show("tok: src/pool.py tokens", len(RUN["cl100k_path"]["ids"]))

# ------------------------------------------------------------- admission (Ch. 2.3.2, 2.3.3)
S_max, K_max, S = 8192, 1024, 6000   # Ch. 2.3.2 (scenario values)
img = 2048                            # Ch. 2.3.3
show("admit: input ceiling S_max - K_max", f"{S_max - K_max:,}")
show("admit: room left at S = 6,000", f"{S_max - K_max - S:,}")
show("admit: demand with image", f"{S + img + K_max:,}")
show("admit: excess with image", f"{S + img + K_max - S_max:,}")
show("admit: excess with image, K_max halved", f"{S + img + K_max // 2 - S_max:,}")

# ------------------------------------------------------------- real next-token distribution (run)
pl = RUN["path_logits"]
for t, p1, p5 in zip(pl["tokens"], pl["p_T1"], pl["p_T05"]):
    show(f"logits: '{t}' at T=1 / T=0.5", f"{pct(p1)} / {pct(p5)}")
show("logits: top-8 mass at T=1", pct(sum(pl["p_T1"])))
show("logits: nucleus 0.90 size (tokens)", pl["nucleus_090_size"])
i_src = pl["tokens"].index("src")
show("logits: rank of 'src' (the file the prompt names)", i_src + 1)

# ------------------------------------------------------------- NM 2.1 (Ch. 2.4.2)
z = [12, 10.5, 9, 6, 3]


def softmax(v, T=1.0):
    e = [math.exp(x / T) for x in v]
    s = sum(e)
    return [x / s for x in e]


p1, p05 = softmax(z), softmax(z, 0.5)
show("NM2.1: partition sum at T=1", f"{sum(math.exp(x) for x in z):.1f}")
show("NM2.1: p at T=1", [round(x, 4) for x in p1])
show("NM2.1: top token at T=0.5", pct(p05[0]))
show("NM2.1: tail at T=1 / T=0.5", f"{pct(1 - p1[0])} / {pct(1 - p05[0])}")
show("NM2.1: tail reduction", f"{(1 - p1[0]) / (1 - p05[0]):.1f}x")
show("NM2.1: top-2 cumulative", f"{p1[0] + p1[1]:.4f}")
show("NM2.1: renormalized v1 / v2", f"{p1[0] / (p1[0] + p1[1]):.4f} / {p1[1] / (p1[0] + p1[1]):.4f}")

# ------------------------------------------------------------- batch logits (run)
bl = RUN["batch_logits"]
show("batch: other request tokens / pad tokens added",
     f"{RUN['batch_other_tokens']} / {bl['float32']['pad_tokens_added']}")
show("batch: fp32 logits changed / vocab", f"{bl['float32']['n_logits_changed']:,} / {RUN['vocab_size']:,}")
show("batch: fp32 share changed", pct(bl['float32']['n_logits_changed'] / RUN['vocab_size']))
show("batch: fp32 max abs diff", f"{bl['float32']['max_abs_diff']:.1e}")
show("batch: bf16 max abs diff", bl['bfloat16']['max_abs_diff'])

# ------------------------------------------------------------- truncated call (run)
show("trunc: completed call tokens / stop", f"{RUN['completed_call']['new_tokens']} / {RUN['completed_call']['stop']}")
show("trunc: truncated call tokens / stop", f"{RUN['truncated_call']['new_tokens']} / {RUN['truncated_call']['stop']}")

# ------------------------------------------------------------- streaming (Ch. 2.5.3)
K_stream, step_rej = 4096, 96
tau_d = 0.021075                     # Ch. 2.8.2 reference decode floor (s/token)
show("stream: skipped decode steps", f"{K_stream - step_rej:,}")
show("stream: decode work avoided (s)", f"{(K_stream - step_rej) * tau_d:.1f}")

# ------------------------------------------------------------- self-review (Huang et al. 2024, Table 3)
huang = {"GSM8K": (95.5, 89.0, 97.5), "CommonSenseQA": (82.0, 80.0, 85.5), "HotpotQA": (49.0, 43.0, 59.0)}
for k, (std, sc2, orc) in huang.items():
    show(f"huang GPT-4 {k}: self-correct r2 - standard (points)", f"{sc2 - std:+.1f}")
    show(f"huang GPT-4 {k}: oracle-stopped - standard (points)", f"{orc - std:+.1f}")

# ------------------------------------------------------------- Bayes (Ch. Systems Perspective 2.1)
d, fa, pv = 0.18, 0.65, 0.85


def resid(d, fa, pv):
    return fa * d / (fa * d + pv * (1 - d))


show("bayes: residual defect among passes", pct(resid(d, fa, pv), 2))
show("bayes: residual with FA halved (0.325)", pct(resid(d, fa / 2, pv), 1))
show("bayes: share of candidates that pass", pct(fa * d + pv * (1 - d), 1))

# ------------------------------------------------------------- per-token floors (Ch. 2.8.2)
P, b, R, BW = 70.6e9, 1, 1979e12, 3.35e12     # Ch. 2.8.2 reference deployment
tp, td = 2 * P / R, P * b / BW
show("floors: GFLOP per token", f"{2 * P / 1e9:.1f}")
show("floors: prefill floor (us/token)", f"{tp * 1e6:.2f}")
show("floors: decode floor (ms/token)", f"{td * 1e3:.3f}")
show("floors: decode / prefill", f"{td / tp:.0f}")
S0, K0 = 8000, 200                            # Ch. 2.8.2 first repair turn
pre, dec = S0 * tp, K0 * td
show("turn1: prefill (s) / decode (s) / total", f"{pre:.3f} / {dec:.3f} / {pre + dec:.3f}")
show("turn1: output share of tokens", pct(K0 / (S0 + K0)))
show("turn1: output share of time", pct(dec / (pre + dec)))

# ------------------------------------------------------------- measured on a laptop (run)
tm = RUN["timing"]
pre_tok = tm["prefill_s_median"] / tm["prompt_tokens"]
show("laptop: prompt tokens / prefill median (s)", f"{tm['prompt_tokens']} / {tm['prefill_s_median']:.2f}")
show("laptop: prefill per token (ms)", f"{pre_tok * 1e3:.2f}")
show("laptop: decode per token (ms)", f"{tm['decode_s_per_token_median'] * 1e3:.1f}")
show("laptop: decode / prefill per token", f"{tm['decode_s_per_token_median'] / pre_tok:.0f}")
show("laptop: decode / prefill, min and max over runs",
     f"{min(tm['decode_s_per_token_runs']) / (max(tm['prefill_s_runs']) / tm['prompt_tokens']):.0f}"
     f" to {max(tm['decode_s_per_token_runs']) / (min(tm['prefill_s_runs']) / tm['prompt_tokens']):.0f}")

# ------------------------------------------------------------- Azure code trace (2023)
rows = list(csv.DictReader(open(HERE / "data" / "azure_llm_code_2023.csv")))
ctx = [int(r["ContextTokens"]) for r in rows]
gen = [int(r["GeneratedTokens"]) for r in rows]
show("azure: requests", f"{len(rows):,}")
show("azure: median prompt / median output", f"{statistics.median(ctx):,.0f} / {statistics.median(gen):,.0f}")
show("azure: input share of all tokens", pct(sum(ctx) / (sum(ctx) + sum(gen))))
show("azure: median prompt / median output ratio", f"{statistics.median(ctx) / statistics.median(gen):.0f}")
# time share if each request ran at the reference floors (derived, illustrative)
t_in = sum(ctx) * tp
t_out = sum(gen) * td
show("azure: output share of floor time (reference floors)", pct(t_out / (t_in + t_out)))

# ------------------------------------------------------------- NM 2.2 (Ch. 2.8.3)
H, S0r, D, K = 30, 8000, 1500, 200


def resent(H):
    return H * S0r + D * H * (H - 1) // 2


def fresh(H):
    return S0r + (H - 1) * D


show("NM2.2: re-sent input tokens", f"{resent(H):,}")
show("NM2.2: fresh prefill with reuse", f"{fresh(H):,}")
show("NM2.2: reduction", f"{resent(H) / fresh(H):.1f}x")
show("NM2.2: output tokens", f"{H * K:,}")
show("NM2.2: input share of tokens", pct(resent(H) / (resent(H) + H * K)))
show("NM2.2: prefill work no reuse / reuse (s)", f"{resent(H) * tp:.2f} / {fresh(H) * tp:.2f}")
show("NM2.2: decode work (s)", f"{H * K * td:.2f}")
show("NM2.2: 60 turns: re-sent growth / fresh growth",
     f"{resent(60) / resent(30):.1f}x / {fresh(60) / fresh(30):.1f}x")
show("NM2.2: 90 turns: re-sent growth / fresh growth",
     f"{resent(90) / resent(30):.1f}x / {fresh(90) / fresh(30):.1f}x")

# ------------------------------------------------------------- NM 2.3 (Ch. 2.9.3)
whole = (8000 + 350) * tp + 1850 * td
local = (8000 + 120) * tp + 80 * td
show("NM2.3: whole file (s)", f"{whole:.2f}")
show("NM2.3: local edit (s)", f"{local:.2f}")
show("NM2.3: ratio", f"{whole / local:.1f}x")
show("NM2.3: decode share of whole-file time", pct(1850 * td / whole))
show("NM2.3: local edit margin under 15 s", f"{15 / local:.1f}x")
show("NM2.3: yield 0.95 x 0.60", f"{0.95 * 0.60:.2f}")
show("NM2.3: whole file over deadline by (s)", f"{whole - 15:.1f}")

# ------------------------------------------------------------- Lighthouse 2.1 support agent (Ch.)
show("LH2.1: prefill 8,192 tokens (s)", f"{8192 * tp:.2f}")
show("LH2.1: 150-token reply (s)", f"{150 * td:.2f}")
show("LH2.1: tokens per second", f"{1 / td:.0f}")
show("LH2.1: fresh prefill 2,048 tokens (s)", f"{2048 * tp:.2f}")
show("bayes: joint defective-and-pass (common wrong answer)", pct(0.65 * 0.18, 1))
