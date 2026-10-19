#!/usr/bin/env python3
"""Every derived number in Lecture 6b (Network Architectures, Part 2),
computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports the inputs below so figures and slides share
one computation.

Run from this folder: python3 calcs.py
"""

import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
# GPT-2 XL: config.json of openai-community/gpt2-xl on Hugging Face
# (n_layer 48, n_embd 1600, n_head 25, vocab_size 50257, n_positions 1024);
# the same shape is MLSysIM GPT2 (layers 48, hidden_dim 1600, heads 25).
L_XL, D_XL, H_XL, VOCAB, NPOS = 48, 1600, 25, 50257, 1024
P_XL_PUB = 1.5e9                 # published 1.5B (Radford et al. 2019; MLSysIM GPT2)
FP16 = 2                         # bytes per FP16/BF16 value
FP32 = 4                         # bytes per FP32 value
# H100 SXM (MLSysIM H100): 989 TFLOP/s dense FP16, 3.35 TB/s HBM3, PCIe Gen5 x16 64 GB/s
H100_FLOPS, H100_BW, PCIE_BW = 989e12, 3.35e12, 64e9
DEV_GB = 80                      # one 80 GB accelerator (A100 or H100 80 GB)
# Llama 3.1 8B: meta-llama/llama-models, models/sku_list.py,
# llama3_1_base_models() arch_args: dim 4096, n_layers 32, n_heads 32,
# n_kv_heads 8 (also MLSysIM Llama3_8B: 8.03B parameters, kv_heads 8)
L3_DIM, L3_LAYERS, L3_HEADS, L3_KV_HEADS, L3_PARAMS = 4096, 32, 32, 8, 8.03e9
L3_WINDOW = 131072               # 128K, Llama 3.1 model card (Hugging Face)


def kv_bytes_per_token(layers, kv_width, nbytes=FP16):
    """Chapter 6.7.4 and Table 7: 2 (K and V) x layers x key-value width x bytes."""
    return 2 * layers * kv_width * nbytes


KV_XL = kv_bytes_per_token(L_XL, D_XL)
L3_HEAD_DIM = L3_DIM // L3_HEADS
KV_L3 = kv_bytes_per_token(L3_LAYERS, L3_KV_HEADS * L3_HEAD_DIM)

# FlashAttention measurements: Dao et al., NeurIPS 2022, arXiv 2205.14135v2,
# Table 20 (forward + backward runtime, ms, no dropout or masking) and
# Table 21 (memory, MB); one A100 40 GB, batch 16, 8 heads of 64, FP16.
FA_SEQ = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
FA_TIME_PT = [0.67, 0.70, 1.18, 3.67, 13.22, 50.44]          # PyTorch, ms (OOM after 4K)
FA_TIME_FA = [0.31, 0.31, 0.73, 2.29, 7.64, 30.09, 118.50, 470.51, 1876.08, 7492.85]
FA_MEM_PT = [36, 104, 336, 1184, 4416, 17024]                # MB
FA_MEM_FA = [22, 44, 104, 209, 418, 836, 1672, 3344, 6688, 13376]

# Toy attention head (illustrative values chosen for easy arithmetic)
TOY_Q = np.array([[1, 0], [0, 1], [1, 1], [1, 0]], float)
TOY_K = np.array([[2, 0], [0, 2], [1, 1], [0, 0]], float)
TOY_V = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], float)


def load_probe():
    """Attention rows measured by attention_probe.py (bert-base-uncased,
    data/attention_probe.txt): {sentence: [(token, weight), ...]}."""
    rows, cur = {}, None
    for line in (HERE / "data" / "attention_probe.txt").read_text().splitlines():
        if line.startswith("The "):
            cur = line.split("  (from")[0]
            rows[cur] = []
        elif cur and line.startswith("  "):
            t, v = line.split()
            rows[cur].append((t, float(v)))
    return rows


ARROW_PT_PER_WEIGHT = 6.8        # slide arrow width in pt per unit of attention weight


def main():
    head("Attention from a pronoun, measured (BERT-base, layer 4, head 11)")
    for sent, row in load_probe().items():
        top = sorted(row, key=lambda r: -r[1])[:4]
        show(sent[:40], ", ".join(f"{t} {v:.2f}" for t, v in top))
    they = dict(load_probe()["The student did not finish all the homework because they were tired."])
    for t in ("student", "homework", "tired"):
        show(f"arrow width to {t} (pt)", f"{ARROW_PT_PER_WEIGHT * they[t]:.1f}")

    head("Hook: context windows")
    # GPT-3 2,048 (Brown et al. 2020); Gemini 1.5 Pro 1M in preview (Google, 15 Feb 2024)
    growth = 1e6 / 2048
    show("GPT-3 to Gemini 1.5 window growth (x)", f"{growth:.0f}  (slide: about 500)")
    show("premise 500x window: score work (x)", f"{500 ** 2:,}")
    show("premise 500x window: memory per request (x)", 500)
    show("exact 488x window: score work (x)", f"{growth ** 2:,.0f}")

    head("Recall: RNN state against the KV cache")
    rnn_state = 512 * FP32           # 512-wide FP32 state (Sec. 6.5.5)
    show("RNN state (KB)", f"{rnn_state / 1e3:.1f}  (slide: 2 KB)")
    show("GPT-2 XL KV per token (KB)", f"{KV_XL / 1e3:.1f}")
    show("KV at 10,000 tokens (GB)", f"{KV_XL * 1e4 / 1e9:.2f}")

    head("Toy head (four tokens, d = 2)")
    scores = TOY_Q @ TOY_K.T / math.sqrt(2)
    w = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    out = w @ TOY_V
    print("  scores\n", scores.round(2))
    print("  weights\n", w.round(2))
    print("  outputs\n", out.round(2))
    show("token 3 weight on tokens 1 to 3", w[2, :3].round(2))
    show("MACs per product, S^2 d", 4 ** 2 * 2)

    head("Heads: GPT-2 Small packed projection")
    d_small, h_small = 768, 12       # MLSysIM GPT2_Small
    show("packed QKV columns", f"{3 * d_small:,}")
    show("head width", d_small // h_small)

    head("nanoGPT attention code: weights in c_attn and c_proj")
    show("c_attn weights / d^2", 3)
    show("c_proj weights / d^2", 1)
    show("attention weights / d^2", 3 + 1)
    show("MLP c_fc + c_proj / d^2", 4 + 4)
    show("block / d^2", 3 + 1 + 4 + 4)
    show("GPT-2 XL c_attn weights (M)", f"{D_XL * 3 * D_XL / 1e6:.2f}")

    head("GPT-2 XL's weights from the block rule")
    per_block = 12 * D_XL ** 2
    blocks = per_block * L_XL
    table = VOCAB * D_XL
    show("per block (M)", f"{per_block / 1e6:.1f}")
    show("48 blocks (B)", f"{blocks / 1e9:.2f}")
    show("token table (M)", f"{table / 1e6:.0f}")
    show("total (B)", f"{(blocks + table) / 1e9:.2f}")
    show("attention projections 4d^2 x 48 (B)", f"{4 * D_XL ** 2 * L_XL / 1e9:.2f}")
    show("MLPs 8d^2 x 48 (B)", f"{8 * D_XL ** 2 * L_XL / 1e9:.2f}")
    show("MLP share of block weights", f"{8 / 12:.2f}")
    flop_tok = 2 * blocks
    show("FLOP per token, 2 x blocks (GFLOP)", f"{flop_tok / 1e9:.2f}")
    show("output layer, 2 x vocab x d (GFLOP)", f"{2 * VOCAB * D_XL / 1e9:.2f}")
    # Exact count with biases, LayerNorms, and position table (nanoGPT model.py
    # with bias=True, as GPT-2): per block c_attn d x 3d + 3d, c_proj d^2 + d,
    # c_fc d x 4d + 4d, c_proj 4d x d + d, two LayerNorms 2 x 2d.
    d = D_XL
    exact_block = (d * 3 * d + 3 * d) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d) + 2 * 2 * d
    exact = L_XL * exact_block + VOCAB * d + NPOS * d + 2 * d
    show("exact parameters with biases, norms, positions", f"{exact:,}  (nanoGPT comment: 1558M)")
    show("exact (M)", f"{exact / 1e6:.0f}")

    head("Predict: attention's share of a block at 1,024 tokens")
    S = 1024
    dense = 12 * S * D_XL ** 2
    attn = 2 * S ** 2 * D_XL
    show("12 S d^2 (B MACs)", f"{dense / 1e9:.1f}")
    show("2 S^2 d (B MACs)", f"{attn / 1e9:.2f}")
    show("block total (B MACs)", f"{(dense + attn) / 1e9:.1f}")
    show("attention share (%)", f"{100 * attn / (dense + attn):.1f}")
    show("dense : attention", f"{dense / attn:.1f}")
    prompt_dense = S * flop_tok
    prompt_attn = attn * 2 * L_XL
    show("prompt, 1,024 x 2.95 GFLOP (TFLOP)", f"{prompt_dense / 1e12:.2f}")
    show("prompt attention, 48 blocks, 2 FLOP/MAC (TFLOP)", f"{prompt_attn / 1e12:.2f}")
    show("prompt total (TFLOP)", f"{(prompt_dense + prompt_attn) / 1e12:.2f}")

    head("Crossover")
    show("6d (tokens)", f"{6 * D_XL:,}")
    show("attention / rest at 100K tokens", f"{1e5 / (6 * D_XL):.1f}")
    show("crossover for a model twice as wide", f"{6 * 2 * D_XL:,}")

    head("Materialized scores (Napkin Math 6.1: 12 heads, 32 layers)")
    n = 1e5 ** 2 * 12
    show("scores", f"{n:.2e}")
    show("bytes per layer (GB)", f"{n * FP16 / 1e9:.0f}")
    show("32 layers (GB)", f"{n * FP16 * 32 / 1e9:,.0f}")
    show("times an 80 GB device", f"{n * FP16 / 1e9 / DEV_GB:.0f}")
    for s in (4096, 8192):
        show(f"one layer at {s} (GB)", f"{s ** 2 * 12 * FP16 / 1e9:.2f}")

    head("True or false")
    show("(4,096 / 512)^2", f"{(4096 / 512) ** 2:.0f}")
    show("(2,048 / 512)^2", f"{(2048 / 512) ** 2:.0f}")
    i1k, i64k = FA_SEQ.index(1024), FA_SEQ.index(65536)
    show("FlashAttention time, 1K -> 64K (x)", f"{FA_TIME_FA[i64k] / FA_TIME_FA[i1k]:,.0f}")
    show("FlashAttention memory, 1K -> 64K (x)", f"{FA_MEM_FA[i64k] / FA_MEM_FA[i1k]:.0f}")
    show("(64K / 1K)^2 and 64K / 1K", f"{(65536 / 1024) ** 2:,.0f} and {65536 / 1024:.0f}")
    i4k = FA_SEQ.index(4096)
    show("PyTorch memory at 4K (GB)", f"{FA_MEM_PT[i4k] / 1e3:.1f}")
    show("FlashAttention memory at 4K (GB)", f"{FA_MEM_FA[i4k] / 1e3:.2f}")
    show("memory saving at 4K (x)", f"{FA_MEM_PT[i4k] / FA_MEM_FA[i4k]:.0f}")
    show("FlashAttention memory at 64K (GB)", f"{FA_MEM_FA[i64k] / 1e3:.1f}")
    show("time per doubling, 32K -> 64K (x)", f"{FA_TIME_FA[i64k] / FA_TIME_FA[i64k - 1]:.2f}")
    show("memory per doubling, 32K -> 64K (x)", f"{FA_MEM_FA[i64k] / FA_MEM_FA[i64k - 1]:.2f}")

    head("Decode: a 70B model at batch one on an H100")
    ridge = H100_FLOPS / H100_BW
    w70 = 70e9 * FP16
    t = w70 / H100_BW
    show("H100 ridge (FLOP/byte)", f"{ridge:.0f}")
    show("decode intensity, 2 FLOP per 2-byte weight", 2 / FP16)
    show("70B FP16 weights (GB)", f"{w70 / 1e9:.0f}")
    show("time per token (ms)", f"{t * 1e3:.1f}")
    show("tokens per second", f"{1 / t:.1f}")
    show("arithmetic time per token (ms)", f"{2 * 70e9 / H100_FLOPS * 1e3:.2f}")
    # 140 GB does not fit one 80 GB H100; split across two H100s (tensor
    # parallel, as Conclusion S16.4 does), each reads its 70 GB half per token
    # at the same time, so the floor halves (communication not counted).
    t2 = (w70 / 2) / H100_BW
    show("two H100s: weights per GPU (GB)", f"{w70 / 2 / 1e9:.0f}")
    show("two H100s: time per token (ms)", f"{t2 * 1e3:.1f}")
    show("two H100s: tokens per second", f"{1 / t2:.1f}")

    head("KV cache per token")
    show("GPT-2 XL (B)", f"{KV_XL:,}")
    show("GPT-2 XL (KB)", f"{KV_XL / 1e3:.1f}")
    show("GPT-2 XL at 1,024 tokens (MB)", f"{KV_XL * 1024 / 1e6:.1f}")
    show("GPT-2 XL key-value width 25 x 64", H_XL * 64)
    show("Llama 3.1 8B head width dim / n_heads", L3_HEAD_DIM)
    show("Llama 3.1 8B key-value width 8 x 128", L3_KV_HEADS * L3_HEAD_DIM)
    show("Llama 3.1 8B query heads per key-value head", L3_HEADS // L3_KV_HEADS)
    show("Llama 3.1 8B KV per token (B)", f"{KV_L3:,}")

    head("Context memory (GPT-2 XL, chapter figure 13)")
    w_xl = P_XL_PUB * FP16
    show("weights FP16 (GB)", f"{w_xl / 1e9:.1f}")
    show("KV passes weights at (tokens)", f"{w_xl / KV_XL:,.0f}")
    dev_gib = DEV_GB * 2 ** 30   # the plot draws the device as 80 GiB (Part 1's code)
    show("one layer's scores (25 heads) fill 80 GiB at (tokens)", f"{math.sqrt(dev_gib / (H_XL * FP16)):,.0f}")
    show("... fill 80 GB (decimal) at (tokens)", f"{math.sqrt(DEV_GB * 1e9 / (H_XL * FP16)):,.0f}")
    show("KV line meets scores line at (tokens)", f"{KV_XL / (H_XL * FP16):,.0f}")

    head("vLLM memory layout (Kwon et al. 2023, Fig. 1)")
    # Inputs: 13B-parameter model on an A100 40 GB; parameters 26 GB, 65 percent;
    # KV cache over 30 percent (figure labels).
    show("13B in FP16 (GB)", f"{13e9 * FP16 / 1e9:.0f}")
    show("26 GB of 40 GB (%)", f"{100 * 26 / 40:.0f}")
    # Its KV per token, OPT-13B (40 layers, width 5,120), as the paper computes
    # it in Sec. 3: 2 x 5,120 x 40 x 2 B ("800 KB", binary units).
    kv13 = kv_bytes_per_token(40, 5120)
    show("13B KV per token (MB)", f"{kv13 / 1e6:.2f}")
    show("tokens in 30% of 40 GB", f"{0.30 * 40e9 / kv13:,.0f}")
    req13 = kv13 * 2048              # OPT's 2,048-token maximum (paper, Sec. 3)
    show("one 2,048-token request (GB)", f"{req13 / 1e9:.2f}")
    show("such requests in 30% of 40 GB", f"{0.30 * 40e9 / req13:.1f}")
    show("... with half the key-value width", f"{0.30 * 40e9 / (req13 / 2):.1f}")

    head("Sharing keys across heads")
    rows = [("GPT-2 XL", 48, 1600), ("Llama 2 7B", 32, 4096),
            ("Llama 3.1 8B", 32, 1024), ("Llama 2 70B", 80, 1024),
            ("Llama 2 70B unshared", 80, 8192), ("PaLM 540B", 118, 256)]
    for name, layers, width in rows:   # MLSysIM shapes; PaLM: one KV head of 256 (Chowdhery et al. 2022)
        show(f"{name} KV per token (KB)", f"{kv_bytes_per_token(layers, width) / 1e3:.1f}")
    show("Llama 2 7B / Llama 3.1 8B", kv_bytes_per_token(32, 4096) / KV_L3)
    show("Llama 2 70B unshared / shared", kv_bytes_per_token(80, 8192) / kv_bytes_per_token(80, 1024))
    show("Llama 3.1 8B cache at 128K (GB)", f"{KV_L3 * L3_WINDOW / 1e9:.1f}")
    show("Llama 3.1 8B weights BF16 (GB)", f"{L3_PARAMS * FP16 / 1e9:.1f}")
    show("Llama 2 7B cache at 2,048 tokens (GB)", f"{kv_bytes_per_token(32, 4096) * 2048 / 1e9:.2f}")

    head("Recommender request")
    feats, width = 26, 128         # Criteo 26 categorical features; 128-wide FP32 rows (Sec. 6.8.3)
    row = width * FP32
    req = feats * row
    vecs = feats + 1
    pairs = vecs * (vecs - 1) // 2
    macs = pairs * width
    show("bytes per row", row)
    show("request bytes (KB)", f"{req / 1e3:.1f}")
    show("pairs among 27 vectors", pairs)
    show("interaction MACs", f"{macs:,}")
    show("FLOP per gathered byte", f"{2 * macs / req:.2f}")
    show("HBM bound (M requests/s)", f"{H100_BW / req / 1e6:.0f}")
    show("PCIe bound (M requests/s)", f"{PCIE_BW / req / 1e6:.1f}")
    show("HBM / PCIe", f"{H100_BW / PCIE_BW:.0f}")

    head("Does one table fit? (Napkin Math 6.2)")
    t1 = 1e8 * 128 * FP32
    show("one table (GB)", f"{t1 / 1e9:.1f}")
    show("share of 80 GB (%)", f"{100 * t1 / 1e9 / DEV_GB:.0f}")
    show("two tables (GB)", f"{2 * t1 / 1e9:.1f}")
    show("a billion users (GB)", f"{1e9 * 128 * FP32 / 1e9:.0f}")
    # Mudigere et al. 2022: "up to 12 Trillion parameters"
    show("12 trillion FP32 parameters (TB)", f"{12e12 * FP32 / 1e12:.0f}")

    head("Lighthouse table (chapter Tables 8 and 10)")
    for name, params in [("ResNet-50", 25.6e6), ("MobileNetV2", 3.5e6), ("KWS", 200e3),
                         ("GPT-2 XL", 1.5e9), ("DLRM", 25e9)]:
        show(f"{name} training state, 16 B/param (MB)", f"{params * 16 / 1e6:,.1f}")
    show("GPT-2 XL weights read per decode step (ms)", f"{w_xl / H100_BW * 1e3:.2f}")

    head("One 1M-token request (Llama 3.1 8B shape)")
    S = 1e6
    show("weights BF16 (GB)", f"{L3_PARAMS * FP16 / 1e9:.1f}")
    show("KV cache (GB)", f"{KV_L3 * S / 1e9:.0f}")
    show("scores per head per layer (TB)", f"{FP16 * S ** 2 / 1e12:.0f}")
    attn_flop = 2 * 2 * S ** 2 * L3_DIM * L3_LAYERS     # two products, 2 FLOP per MAC
    show("attention FLOP per prompt", f"{attn_flop:.2e}")
    show("H100 peak time (min)", f"{attn_flop / H100_FLOPS / 60:.1f}")
    show("window 1M / 2K", f"{S / 2048:.0f}")


if __name__ == "__main__":
    main()
