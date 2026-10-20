#!/usr/bin/env python3
"""Every derived number in Lecture 13b (Model Serving, Part 2), one block per slide.

Inputs carry their source. "MS" is the rendered chapter, Model Serving
(Volume I, Chapter 13). Every printed value is what the slide or its note
shows, with the label used there. make_figs.py imports from this file.

Run from this folder: python3 calcs.py
"""

import math

GB = 1e9
GIB = 2 ** 30


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


# =============================================================================
# Shared inputs
# =============================================================================
H100_FP16_FLOPS = 989e12        # H100 SXM dense FP16 peak (MS Sec. 13.9.2)
H100_BW = 3.35e12               # H100 SXM HBM3 bandwidth, B/s (MS Sec. 13.9.2)
H100_MEM = 80 * GIB             # 80 GiB datasheet capacity (MS Sec. 13.9.5)
PREFILL_EFF = 0.40              # prefill at 40% of peak (MS Sec. 13.9.2)

GPT2XL_PARAMS = 1.5e9           # GPT-2 XL parameters (MS Sec. 13.9.2)
GPT2XL_W = 3.0 * GB             # GPT-2 XL FP16 weights (MS Sec. 13.9.2)
GPT2XL_KV_TOK = 307.2e3         # B per token, GPT-2 XL (MS Sec. 13.9.2)
GPT2XL_CTX = 1024               # full context (MS Sec. 13.9.2)

L3_KV_TOK = 131072              # B per token, Llama 3 8B, FP16 (MS Sec. 13.9.5)
L3_W16 = 16.1 * GB              # Llama 3 8B FP16 weights (MS Sec. 13.9.5)
L3_W4 = 4.0 * GB                # Llama 3 8B INT4 weights (MS Sec. 13.10.3)
L3_PREFILL_1K = 41.9e-3         # s per 1,000-token prefill (MS Sec. 13.10.3)
L3_TPOT_MODEL = 1.57e-3         # modeled INT4 TPOT (MS Sec. 13.10.3)
TPOT_TARGET = 20e-3             # case-study TPOT target (MS Sec. 13.10.3)
PROMPT, GEN = 1000, 256         # case-study request (MS Sec. 13.10.3)
CONV = PROMPT + GEN

KV70_TOK = 2 * 80 * 8 * 128 * 2  # 70B class: K and V, 80 layers, 8 KV heads of 128, FP16 (assumed)

# MLPerf Inference v5.0, 5.0-0020, 8x H100-SXM-80GB, Llama 2 70B (summary_results.json)
MLP_OFFLINE = 31216.8
MLP_SERVER = 30903.6            # TTFT 2 s, TPOT 200 ms
MLP_INTER = 12979.4             # TTFT 450 ms, TPOT 40 ms


# =============================================================================
# Slide: Predict, what does a tighter token deadline cost? (hook)
# =============================================================================
head("Hook (MLPerf v5.0, 5.0-0020)")
show("Offline tokens/s", f"{MLP_OFFLINE:,.0f}")
show("Server tokens/s", f"{MLP_SERVER:,.0f}")
show("Interactive tokens/s", f"{MLP_INTER:,.0f}")
show("Server / Offline", f"{MLP_SERVER / MLP_OFFLINE:.0%}")
show("Interactive / Server", f"{MLP_INTER / MLP_SERVER:.0%}")
show("Server / Interactive (times fewer)", f"{MLP_SERVER / MLP_INTER:.1f}")
show("TPOT tightening 200/40", f"{200 / 40:g}")

# =============================================================================
# Slide: Recall (Part 1 thumbnail; $3.92 and $0.52 from Lecture 13, Example 13.1)
# =============================================================================
head("Recall")
show("batch 1 / batch 8 cost", f"{3.92 / 0.52:.1f}")
show("Llama 3 8B cache per token (KB)", f"{L3_KV_TOK / 1e3:.0f}")

# =============================================================================
# Slide: A chat answer has two deadlines (Splitwise Table IV, Llama 2 70B, P50)
# =============================================================================
head("Two deadlines (Splitwise Table IV)")
H_TTFT, H_TBT, H_E2E = 84, 28, 3387     # DGX-H100, ms
A_TTFT, A_TBT, A_E2E = 155, 40, 4957    # DGX-A100, ms
DECODE_SHARE = (H_E2E - H_TTFT) / H_E2E
show("decode share of the request", f"{DECODE_SHARE:.0%}")
show("answer tokens, about", f"{(H_E2E - H_TTFT) / H_TBT + 1:.0f}")
show("50 ms TPOT in tokens/s", f"{1 / 0.050:.0f}")

# =============================================================================
# Slide: Prefill spends arithmetic; decode spends a weight read
# =============================================================================
head("Prefill and decode")
gpt2_prefill_flop = 2 * GPT2XL_PARAMS * 512
GPT2_PREFILL_MS = gpt2_prefill_flop / (PREFILL_EFF * H100_FP16_FLOPS) * 1e3
show("GPT-2 XL 512-token prefill (TFLOP)", f"{gpt2_prefill_flop / 1e12:.1f}")
show("GPT-2 XL prefill (ms)", f"{GPT2_PREFILL_MS:.1f}")
GPT2_FLOOR_MS = GPT2XL_W / H100_BW * 1e3
L3_FLOOR_MS = L3_W4 / H100_BW * 1e3
show("GPT-2 XL decode floor (ms)", f"{GPT2_FLOOR_MS:.1f}")
show("Llama 3 8B INT4 floor (ms)", f"{L3_FLOOR_MS:.1f}")
ROOM_MS = (TPOT_TARGET - L3_TPOT_MODEL) * 1e3
show("room in a 20 ms TPOT (ms)", f"{ROOM_MS:.1f}")
show("H100 ridge (FLOP/B)", f"{H100_FP16_FLOPS / H100_BW:.0f}")

# =============================================================================
# Slide: Estimate, tokens per speculative round (MS equation 15)
# =============================================================================
head("Speculative decoding")
DRAFT, VERIFY, K, ALPHA = 0.30, 0.90, 4, 0.7
ROUND = DRAFT + VERIFY
TPR = (1 - ALPHA ** (K + 1)) / (1 - ALPHA)
show("round (ms)", f"{ROUND:.2f}")
show("tokens per round", f"{TPR:.1f}")
show("ms per token", f"{ROUND / TPR:.2f}")
show("speedup over 0.90 ms", f"{VERIFY / (ROUND / TPR):.1f}")
show("wrong answer 0.7 x 4 + 1", f"{ALPHA * K + 1:.1f}")
show("GPT-2 Small weights (MB), 124M x 2 B", f"{124e6 * 2 / 1e6:.0f}")

# =============================================================================
# Slides: Batching decode; A TPOT target caps the batch (GPT-2 XL)
# =============================================================================
head("Decode bytes and the batch frontier (GPT-2 XL)")
KV_SEQ = GPT2XL_KV_TOK * GPT2XL_CTX


def step_s(b):
    """Bandwidth floor of one decode step at batch b, full contexts."""
    return (GPT2XL_W + b * KV_SEQ) / H100_BW


def rate(b):
    return b / step_s(b)


show("cache per full context (GB)", f"{KV_SEQ / GB:.2f}")
for b in (1, 32):
    show(f"batch {b}: bytes (GB)", f"{(GPT2XL_W + b * KV_SEQ) / GB:.1f}")
    show(f"batch {b}: step (ms)", f"{step_s(b) * 1e3:.1f}")
    show(f"batch {b}: tokens/s", f"{rate(b):,.0f}")
show("batch 32 cache share of bytes", f"{32 * KV_SEQ / (GPT2XL_W + 32 * KV_SEQ):.0%}")
show("TPOT growth batch 1 to 32", f"{step_s(32) / step_s(1):.1f}")
show("throughput growth batch 1 to 32", f"{rate(32) / rate(1):.1f}")
show("batch 128 step (ms)", f"{step_s(128) * 1e3:.1f}")
CEILING = H100_BW / KV_SEQ
show("throughput ceiling (tokens/s)", f"{CEILING:,.0f}")
B_TPOT = math.floor((TPOT_TARGET * H100_BW - GPT2XL_W) / KV_SEQ)
B_MEM = math.floor((H100_MEM - GPT2XL_W) / KV_SEQ)
B_TPOT40 = math.floor((0.040 * H100_BW - GPT2XL_W) / KV_SEQ)
show("H100 memory (GB)", f"{H100_MEM / GB:.1f}")
show("TPOT 20 ms caps batch at", B_TPOT)
show("TPOT 40 ms caps batch at", B_TPOT40)
show("memory caps batch at", B_MEM)

# =============================================================================
# Slide: A real server reports both deadlines (vLLM benchmark, issue #24684)
# Two runs, one H100 80GB HBM3, Llama-3.1-8B-Instruct, vLLM 0.10.1.1,
# --max-concurrency 64, 640 prompts (github.com/vllm-project/vllm/issues/24684)
# =============================================================================
head("vLLM serving benchmark (issue 24684)")
SHORT = dict(inp=136352, out=327680, tpot=9.50, itl_med=9.27, itl_p99=12.55,
             ttft_med=275.99, out_rate=6411.31)
LONG = dict(inp=654052, out=81920, tpot=18.41, itl_med=11.13, itl_p99=198.59,
            ttft_med=582.78, out_rate=2802.42)
N_REQ, CONC = 640, 64
for name, r in (("short prompts", SHORT), ("long prompts", LONG)):
    show(f"{name}: input tokens per request", f"{r['inp'] / N_REQ:,.0f}")
    show(f"{name}: output tokens per request", f"{r['out'] / N_REQ:.0f}")
    show(f"{name}: p99 ITL / median ITL", f"{r['itl_p99'] / r['itl_med']:.1f}")
LITTLE = CONC / (SHORT["tpot"] / 1e3)
show("Little's law, 64 streams / 9.50 ms (tokens/s)", f"{LITTLE:,.0f}")
show("measured / Little bound", f"{SHORT['out_rate'] / LITTLE:.0%}")

# =============================================================================
# Slides: Request-level and continuous batching (illustrative lengths)
# =============================================================================
head("Batching schedules")
REQ1 = [("A", 10), ("B", 100), ("C", 40), ("D", 60)]
REQ2 = {"E": 30, "F": 50, "G": 20, "H": 40}
work1 = sum(L for _, L in REQ1)
work = work1 + sum(REQ2.values())
show("request-level, first batch used", f"{work1} of {4 * 100}")
show("request-level, all used", f"{work} of {4 * 150} ({work / 600:.0%})")
show("continuous, all used", f"{work} of {4 * 100} ({work / 400:.1%})")
show("A's slot idle share", f"{(100 - 10) / 100:.0%}")

# =============================================================================
# Slide: Diagnose, why do all streams freeze at once? (Llama 3 8B INT4)
# =============================================================================
head("Generation stall and chunked prefill")
N_DEC = 32
STALL_BYTES = L3_W4 + N_DEC * CONV * L3_KV_TOK
DEC_MS = STALL_BYTES / H100_BW * 1e3
PRE_MS = L3_PREFILL_1K * 1e3
STALL_MS = DEC_MS + PRE_MS
CHUNKS = 4
CHUNK_MS = DEC_MS + PRE_MS / CHUNKS
show("decode step bytes (GB)", f"{STALL_BYTES / GB:.1f}")
show("decode step (ms)", f"{DEC_MS:.1f}")
show("step with whole prefill (ms)", f"{STALL_MS:.1f}")
show("chunk of 250 tokens prefill (ms)", f"{PRE_MS / CHUNKS:.1f}")
show("step with one chunk (ms)", f"{CHUNK_MS:.1f}")
show("first token after 4 chunked steps (ms)", f"{CHUNKS * CHUNK_MS:.0f}")
show("largest safe chunk (tokens)", f"{(TPOT_TARGET * 1e3 - DEC_MS) / (PRE_MS / 1000):.0f}")

# =============================================================================
# Slide: Estimate, how many 8K-token chats fit on one H100?
# =============================================================================
head("KV budget, Llama 3 8B FP16")
FREE = H100_MEM - L3_W16
PER_CONV8K = L3_KV_TOK * 8192
N8K = FREE / PER_CONV8K
show("free (GB)", f"{FREE / GB:.1f}")
show("per 8,192-token conversation (GB)", f"{PER_CONV8K / GB:.2f}")
show("conversations", f"{N8K:.0f}")
show("65 x per conversation (GB)", f"{65 * PER_CONV8K / GB:.1f}")
BUDGET70 = 50.6 * GB            # MS figure 8: budget after 4-bit 70B weights
show("70B class bytes per token", f"{KV70_TOK:,}")
show("70B / 8B per token", f"{KV70_TOK / L3_KV_TOK:.1f}")
show("70B: 32 conversations full at (tokens)", f"{BUDGET70 / (32 * KV70_TOK):,.0f}")

# =============================================================================
# Slide: Reserving wastes; paging (vLLM Fig. 13a, OPT-13B, ShareGPT)
# =============================================================================
head("vLLM measured batch")
show("30.42 / 13.62", f"{30.42 / 13.62:.1f}")

# =============================================================================
# Slide: vLLM's startup log checks the estimate (issue #21266)
# Meta-Llama-3-8B, BF16, one H100 (SM90), vLLM V1 engine v0.9.1, defaults
# (gpu_memory_utilization 0.9, block size 16), max_seq_len 8,192.
# github.com/vllm-project/vllm/issues/21266
# =============================================================================
head("vLLM startup log (issue 21266)")
LOG_WEIGHTS_GIB = 14.9596
LOG_KV_GIB = 51.39
LOG_TOKENS = 420976
LOG_CONC = 51.39
BLOCK = 16
UTIL = 0.9
show("log tokens x 131,072 B (GiB)", f"{LOG_TOKENS * L3_KV_TOK / GIB:.2f}")
show("log tokens x 131,072 B (GB)", f"{LOG_TOKENS * L3_KV_TOK / GB:.1f}")
show("log tokens / 8,192", f"{LOG_TOKENS / 8192:.2f}")
show("weights (GB)", f"{LOG_WEIGHTS_GIB * GIB / GB:.1f}")
budget_gib = UTIL * H100_MEM / GIB
other_gib = budget_gib - LOG_WEIGHTS_GIB - LOG_KV_GIB
show("0.9 x 80 GiB (GiB)", f"{budget_gib:.0f}")
show("0.9 x 80 GiB (GB)", f"{UTIL * H100_MEM / GB:.1f}")
show("left for activations and workspace (GiB)", f"{other_gib:.2f}")
show("left for activations and workspace (GB)", f"{other_gib * GIB / GB:.1f}")
show("held back by the 0.9 cap (GB)", f"{(1 - UTIL) * H100_MEM / GB:.1f}")
show("estimate 65 vs log", f"{LOG_CONC / N8K:.0%}")
show("estimate high by", f"{N8K / LOG_CONC - 1:.0%}")
PAGED_BLOCKS = math.ceil(CONV / BLOCK)
PAGED_BYTES = PAGED_BLOCKS * BLOCK * L3_KV_TOK
N_PAGED = FREE / PAGED_BYTES
LOG_PAGED = (LOG_TOKENS // BLOCK) // PAGED_BLOCKS
show("blocks per 1,256-token conversation", PAGED_BLOCKS)
show("bytes per paged conversation (MB)", f"{PAGED_BYTES / 1e6:.1f}")
show("paged estimate (conversations)", f"{N_PAGED:.0f}")
show("paged / reserved", f"{N_PAGED / N8K:.1f}")
show("log blocks / 79 (conversations)", LOG_PAGED)
show("max waste per conversation (MB), 15 slots", f"{(BLOCK - 1) * L3_KV_TOK / 1e6:.1f}")

# =============================================================================
# Slide: Requests with a common prefix can share its cache blocks
# =============================================================================
head("Prefix sharing")
PREFIX = 2000 * L3_KV_TOK
show("2,000-token prefix (MB)", f"{PREFIX / 1e6:.0f}")
show("100 copies (GB)", f"{100 * PREFIX / GB:.1f}")
show("shared copy (GB)", f"{PREFIX / GB:.2f}")
show("saved by sharing across 100 (GB)", f"{99 * PREFIX / GB:.0f}")
show("prefill tokens skipped", f"{99 * 2000:,}")
PCIE = 32e9                     # PCIe Gen4 x16, B/s (MS Sec. 13.8.4)
show("reload one 8,192-token cache (ms)", f"{PER_CONV8K / PCIE * 1e3:.0f}")

# =============================================================================
# Slides: DistServe (Fig. 1 and Sec. 1, 13B model, one 80 GB A100)
# =============================================================================
head("DistServe pools")
COLO, PRE_ONLY, DEC_ONLY = 1.6, 5.6, 10
n_pre = math.ceil(DEC_ONLY / PRE_ONLY)
per_gpu = DEC_ONLY / (n_pre + 1)
show("prefill GPUs needed (exact)", f"{DEC_ONLY / PRE_ONLY:.1f}")
show("prefill GPUs", n_pre)
show("per GPU (requests/s)", f"{per_gpu:.1f}")
show("gain over colocated", f"{per_gpu / COLO:.1f}")
XFER = 1.13 * GB                # OPT-66B, 512-token prompt (DistServe Sec. 3.3)
show("cache traffic at 10 req/s (GB/s)", f"{XFER * 10 / GB:.1f}")
show("cache traffic (Gbps)", f"{XFER * 10 * 8 / 1e9:.0f}")

# =============================================================================
# Slide: Joules per token (MS Napkin Math 13.5 power; GPT-2 XL rates above)
# =============================================================================
head("Joules per token")
P_NODE, P_LOW = 1000.0, 600.0
J1 = P_NODE / rate(1)
J32 = P_NODE / rate(32)
J1_LOW = P_LOW / rate(1)
show("batch 1, 1 kW (J)", f"{J1:.2f}")
show("batch 32, 1 kW (J)", f"{J32:.2f}")
show("batch 1, 600 W (J)", f"{J1_LOW:.2f}")
show("600 W batch 1 / batch 32", f"{J1_LOW / J32:.1f}")
show("power ratio", f"{P_NODE / P_LOW:.1f}")

# =============================================================================
# Slide: Estimate, what does a million tokens cost on one H100?
# =============================================================================
head("Cost per million tokens")
REQ_RATE, PRICE_H = 10, 3.0     # MS Sec. 13.10.3 (assumed prefill rate, $3/hour)
TOK_H = REQ_RATE * 3600 * CONV
COST_M = PRICE_H / (TOK_H / 1e6)
show("tokens per hour (M)", f"{TOK_H / 1e6:.1f}")
show("$ per million tokens", f"{COST_M:.3f}")

# =============================================================================
# Slide: Why 40 ms keeps fewer tokens than 200 ms (Little's law, MLPerf hook)
# =============================================================================
head("Synthesis")
S200 = MLP_SERVER * 0.200
S40 = MLP_INTER * 0.040
show("streams at most, 200 ms", f"{S200:,.0f}")
show("streams at most, 40 ms", f"{S40:,.0f}")
MEM8 = 8 * H100_MEM
W70_16 = 4 * 35.3 * GB          # 4x the chapter's 35.3 GB at 4 bits
FREE8 = MEM8 - W70_16
CONV70 = CONV * KV70_TOK
N_MEM = FREE8 / CONV70
show("8 x H100 (GB)", f"{MEM8 / GB:.0f}")
show("16-bit 70B weights (GB)", f"{W70_16 / GB:.0f}")
show("free (GB)", f"{FREE8 / GB:.0f}")
show("1,256-token conversation (MB)", f"{CONV70 / 1e6:.0f}")
show("memory streams", f"{N_MEM:,.0f}")
show("step at the memory limit (ms)", f"{N_MEM / MLP_SERVER * 1e3:.0f}")
FREE8_8 = MEM8 - W70_16 / 2
CONV70_8 = CONV70 / 2
show("8-bit: free (GB)", f"{FREE8_8 / GB:.0f}")
show("8-bit: conversation (MB)", f"{CONV70_8 / 1e6:.0f}")
show("8-bit: memory streams", f"{FREE8_8 / CONV70_8:,.0f}")
show("servers for 30,904 under 40 ms", f"{MLP_SERVER / MLP_INTER:.1f}")
