#!/usr/bin/env python3
"""Every derived number in Lecture 8b (Model Training, Part 2).

One block per slide, in deck order. Inputs carry their source as a comment
("ch" is the rendered chapter, Model Training); each printed line is a value
the slide or its note shows, under the label used there. make_figs.py
imports from this file so figures and slides share one computation.

Run from this folder: python3 calcs.py
"""
import csv
import math
from pathlib import Path

HERE = Path(__file__).parent


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


# ---------------------------------------------------------------------------
# Inputs (sources in comments)
# ---------------------------------------------------------------------------
GB = 1e9
DAY = 86400
P = 1.5e9                    # ch Lighthouse 8.1: GPT-2 XL parameters
RUN_FLOP = 1.5e21            # ch Sec. 8.1: GPT-2 XL training run
BATCH, SEQ = 32, 1024        # ch Sec. 8.6.6
LAYERS, HIDDEN, HEADS, DHEAD = 48, 1600, 25, 64   # ch Napkin Math 8.3
H100_GIB = 80                # H100 capacity, GiB (sold as "80 GB")
GTX580_GB = 3                # ch Sec. 8.7.1: GTX 580 memory
UTIL = 0.50                  # ch Napkin Math 8.7
PRICE = 4.0                  # ch Napkin Math 8.7: dollars per H100-hour
BUSY = 48                    # ch Sec. 8.6.6: percent busy, single-core tokenizer
MP_BYTES = (2, 2, 4, 8)      # ch Sec. 8.3.2: FP16 W, FP16 G, FP32 master, Adam m+v
FP32_BYTES = (4, 4, 0, 8)    # ch Napkin Math 8.6: FP32 W, FP32 G, no master, Adam

# Peak dense TFLOP/s by format (ch figure 10; NVIDIA datasheets)
PEAK = {"V100": {"FP32": 15.7, "FP16": 125},
        "A100": {"FP32": 19.5, "TF32": 156, "FP16": 312},
        "H100": {"FP32": 67, "TF32": 494, "FP16": 989, "FP8": 1979},
        "B200": {"FP32": 75, "TF32": 1125, "FP16": 2250, "FP8": 4500, "FP4": 9000}}
H100_BW = 3.35e12            # ch Sec. 8.3: HBM bandwidth, B/s
# NVIDIA H100 product page spec table (H100 SXM, retrieved 2026-10-18),
# rates marked "* With sparsity": TF32 989, BF16/FP16 1,979 TFLOP/s
H100_SPARSE = {"TF32": 989, "FP16": 1979}

# ch Napkin Math 8.3: activations per layer, FP16, batch 32, 1,024 tokens
ATTN_MB, FFN_MB, LN_MB = 419.4, 419.4, 10
LIB_ACT_GB = 286.9           # ch: library estimate of saved activations
SCORE_BYTES = (2, 1, 2)      # ch: softmax P, dropout mask, P after dropout

# ch Sec. 8.6.4 tiling example: 4,096 tokens, one head, FP32 scores
N_TILE_EX, TILE = 4096, 128
SMEM_KB = 228                # Hardware Acceleration: shared memory per SM

# Dao et al. 2022, Figure 2 (left): GPT-2 medium attention fwd+bwd, A100
FA_GFLOP = (66.6, 75.2)
FA_HBM_GB = (40.3, 4.4)
FA_MS = (41.7, 7.3)
FA_FIG1_SPEEDUP = 7.6        # Dao et al. 2022, Figure 1 caption (GPT-2, attention)
FA_FIG1_BW = (19, 1.5)       # Dao et al. 2022, Figure 1 left: SRAM, HBM TB/s (A100)

# Napkin Math 8.8 (ch): 8 segments of 6 layers
SEGMENTS = 8
LIVE_SEG_GB = 10.7           # ch: live reconstructed segment

# Chen et al. 2016, abstract: 1,000-layer residual net
CHEN_GB = (48, 7)

# Korthikanti et al. 2023 (arXiv 2205.05198v1). Table 2: activation bytes per
# layer = s*b*h*(34 + 5*a*s/h) storing everything; s*b*h*34 with selective
# recomputation (attention core only); s*b*h*2 with full recomputation.
# Table 4: one layer of the 22B model, forward + backward (ms).
K_TABLE4 = {"No recompute": 19.6, "Full recompute": 27.2, "Selective recompute": 20.9}
K_MFU = (42.1, 54.2)         # abstract: 530B on 2,240 A100s, full vs present work

# Micikevicius et al. 2018, Figure 3 (SSD activation gradients), read from the
# figure image (arXiv 1710.03740v3, page 4, 898 x 575 px): gridline y pixel of
# 64 percent = 35, of 1/512 percent = 504; bar top pixels per bin [lo, hi) in
# log2(magnitude). The zero bin uses the caption's 67 percent.
FIG3_Y64, FIG3_Y512 = 35, 504
FIG3_BINS = [(-75, -60, None), (-60, -45, 436), (-45, -40, 330), (-40, -38, 312),
             (-38, -36, 274), (-36, -34, 234), (-34, -32, 195), (-32, -30, 162),
             (-30, -28, 140), (-28, -26, 134), (-26, -24, 143), (-24, -22, 169),
             (-22, -20, 206), (-20, -18, 242), (-18, -16, 272), (-16, -14, 310),
             (-14, -12, 358), (-12, -10, 439)]
FIG3_ZERO_PCT = 67           # caption
SSD_SCALE = 8                # Sec. 3.2: scaling by 8 matched FP32

# torch.amp.GradScaler defaults (PyTorch docs)
SCALER_INIT, SCALER_INTERVAL = 2.0 ** 16, 2000

# Hugging Face docs (perf_train_gpu_one.md, v5.0.0): checkpointing "~20%" slower
HF_CKPT_SLOWER = 20


def h100_gb():
    return H100_GIB * 2 ** 30 / GB


def static_gb():
    return P * sum(MP_BYTES) / GB


def scores_layer_bytes(b=BATCH, s=SEQ):
    return sum(SCORE_BYTES) * HEADS * s * s * b


def fp16_act_gb():
    return LIB_ACT_GB


def fp32_act_gb():
    return 2 * LIB_ACT_GB


def run_days(rate_tflops):
    return RUN_FLOP / (rate_tflops * 1e12 * UTIL) / DAY


def k_bytes_per_layer(coeff):
    """Korthikanti Table 2 with the deck's s, b, h (bytes per layer)."""
    return SEQ * BATCH * HIDDEN * coeff


def k_score_coeff():
    return 5 * HEADS * SEQ / HIDDEN


def flash_act_gb():
    return LIB_ACT_GB * 34 / (34 + k_score_coeff())


def per_seq_gb():
    return flash_act_gb() / BATCH


def seg_inputs_gb():
    return SEGMENTS * BATCH * SEQ * HIDDEN * 2 / GB


def fig3_pct():
    """Digitized percentages per bin (None for an empty bin)."""
    per_oct = (FIG3_Y512 - FIG3_Y64) / 15       # 64 = 2^6 down to 1/512 = 2^-9
    out = []
    for lo, hi, y in FIG3_BINS:
        out.append((lo, hi, None if y is None else 2 ** (6 - (y - FIG3_Y64) / per_oct)))
    return out


def scaler_trace():
    path = HERE / "data" / "scaler_trace.csv"
    rows = []
    with path.open() as f:
        next(f)
        for r in csv.DictReader(f):
            rows.append((int(r["step"]), float(r["scale_after_step"]), int(r["skipped"])))
    return rows


# ---------------------------------------------------------------------------
def hook():
    head("Slide 2: AlexNet's GTX 580 and the H100")
    show("H100 80 GiB in GB", f"{h100_gb():.1f}")
    show("memory growth GTX 580 to H100", f"{H100_GIB / GTX580_GB:.0f}x (80 GB / 3 GB, about 27)")
    fp32 = P * sum(FP32_BYTES) / GB + fp32_act_gb()
    show("FP32 step", f"{fp32:.0f} GB (static {P * sum(FP32_BYTES) / GB:.0f} + activations {fp32_act_gb():.1f})")


def predict():
    head("Slide 3: Predict")
    fp32 = P * sum(FP32_BYTES) / GB + fp32_act_gb()
    show("static", f"{static_gb():.0f} GB")
    show("FP32 activations", f"{fp32 - static_gb():.0f} GB")
    show("H100s of memory", f"{fp32 / h100_gb():.2f}")
    show("BF16 run", f"{run_days(PEAK['H100']['FP16']):.1f} days")


def recall():
    head("Slide 4: Recall")
    show("bytes per parameter", f"{sum(MP_BYTES)} B")
    show("static", f"{static_gb():.0f} GB")
    scores = LAYERS * scores_layer_bytes() / GB
    show("scores over 48 layers", f"{scores:.1f} GB")
    show("other activations", f"{fp16_act_gb() - scores:.1f} GB")
    show("mixed-precision step", f"{static_gb() + fp16_act_gb():.1f} GB")


def ladder():
    head("Slide 6: precision ladder")
    show("V100 to B200 FP32 growth", f"{PEAK['B200']['FP32'] / PEAK['V100']['FP32']:.1f}x")
    show("V100 to B200 FP16 tensor growth", f"{PEAK['B200']['FP16'] / PEAK['V100']['FP16']:.0f}x")
    r = PEAK["H100"]["FP16"] / PEAK["H100"]["FP32"]
    show("H100 FP16 over FP32", f"{r:.1f}x (about 15; 14 of every 15 unused)")
    for f, v in H100_SPARSE.items():
        show(f"H100 {f} sparse / dense", f"{v} / {PEAK['H100'][f]} = {v / PEAK['H100'][f]:.2f}")


def formats():
    head("Slide 7: formats")
    show("FP16 smallest normal 2^-14", f"{2 ** -14:.2e}")
    show("FP16 largest", f"{(2 - 2 ** -10) * 2 ** 15:,.0f}")
    show("FP16 smallest subnormal 2^-24", f"{2 ** -24:.1e}")
    show("FP32 smallest normal 2^-126", f"{2 ** -126:.1e}")
    show("FP32 largest", f"{(2 - 2 ** -23) * 2 ** 127:.2e}")


def swamp():
    head("Slide 8: swamping")
    show("FP16 spacing near 0.5 (2^-11)", f"{2 ** -11:.2e}")
    show("next FP16 value above 0.5", f"{0.5 + 2 ** -11:.5f}")
    show("BF16 spacing near 0.5 (2^-8)", f"{2 ** -8:.2e}")
    show("update/weight ratio that rounds away (0.5 / half spacing)", f"{0.5 / 2 ** -12:,.0f}")
    show("FP32 after 10,000 x 0.0001", f"{0.5 + 10000 * 1e-4:.1f}")


def loss_scale():
    head("Slide 9: loss scaling (Micikevicius et al. Figure 3)")
    bins = fig3_pct()
    below = sum(p for lo, hi, p in bins if p and hi <= -24)
    sub = sum(p for lo, hi, p in bins if p and lo >= -24 and hi <= -14)
    normal = sum(p for lo, hi, p in bins if p and lo >= -14)
    nonzero = below + sub + normal
    show("digitized [-34,-32) (caption: 2%)", f"{dict(((lo, hi), p) for lo, hi, p in bins)[(-34, -32)]:.1f}%")
    show("nonzero values below 2^-24", f"{below:.1f}% of all, {100 * below / nonzero:.0f}% of nonzero")
    show("in the subnormal band", f"{sub:.1f}%")
    show("at or above 2^-14", f"{normal:.2f}%")
    show("nonzero total read from bars (+ 67% zeros)", f"{nonzero:.1f}% (sum {nonzero + FIG3_ZERO_PCT:.0f}%)")
    shift = math.log2(SSD_SCALE)
    show("x8 shift in exponent", f"{shift:.0f}")
    rescued = sum(p for lo, hi, p in bins if p and hi <= -24 and lo >= -24 - shift)
    show("nonzero values moved above 2^-24 by x8 (bins fully inside)", f"{rescued:.1f}%")
    lo_after = max(lo for lo, hi, p in bins if p) + shift
    show("largest bin edge after x8", f"2^{lo_after + 2:.0f} (far below 2^16)")
    top_edge = max(hi for lo, hi, p in bins if p)
    show("headroom: largest bin edge to 65,504", f"{math.log2(65504) - top_edge:.0f} doublings")
    show("1e-6 in log2", f"{math.log2(1e-6):.1f}")
    show("S = 1,024: 1e-6 x S", f"{1e-6 * 1024:.2e}")
    show("S = 1,024: overflow above", f"{65504 / 1024:.0f}")


def diagnose_trace():
    head("Slide 11: real GradScaler trace")
    rows = scaler_trace()
    skips = [s for s, _, k in rows if k]
    top = max(sc for _, sc, _ in rows)
    show("steps run", f"{len(rows):,}")
    show("first skipped step", f"{skips[0]:,}")
    show("doublings before first skip", f"{math.log2(top / SCALER_INIT):.0f} (2^16 to 2^{math.log2(top):.0f})")
    show("skipped steps", f"{len(skips)}")
    gaps = sorted({b - a for a, b in zip(skips, skips[1:])})
    show("steps between skips", f"{gaps}")
    show("scale at the ceiling", f"2^{math.log2(top):.0f} = {top:,.0f}; after halving 2^{math.log2(top) - 1:.0f}")
    show("share of steps skipped after the climb",
         f"{100 * len(skips) / (len(rows) - skips[0] + 1):.2f}%")


def days():
    head("Slide 12: 518 days to 35, 598 GB to 311")
    for f in ("FP32", "TF32", "FP16"):
        d = run_days(PEAK["H100"][f])
        show(f"{f} run", f"{d:.1f} days, ${d * 24 * PRICE:,.0f}")
    show("mixed-precision activations", f"{fp32_act_gb():.1f} -> {fp16_act_gb():.1f} GB")
    show("step", f"{static_gb() + fp16_act_gb():.1f} GB = {(static_gb() + fp16_act_gb()) / h100_gb():.2f} H100s")


def layer_bytes():
    head("Slide 14: one layer's bytes")
    sc = scores_layer_bytes() / 1e6
    total = ATTN_MB + FFN_MB + sc + LN_MB
    show("scores per layer", f"{sc:,.1f} MB")
    show("layer total", f"{total:,.1f} MB")
    show("score share", f"{100 * sc / total:.0f}%")
    show("x48 scores", f"{LAYERS * sc / 1e3:.1f} GB")
    show("line items x48", f"{LAYERS * total / 1e3:.1f} GB")
    show("bytes per score entry", f"{sum(SCORE_BYTES)}")


def attention_io():
    head("Slide 15: standard attention")
    show("intensity d_head/2", f"{DHEAD / 2:.0f} FLOP/B")
    show("ridge 989 / 3.35", f"{PEAK['H100']['FP16'] * 1e12 / H100_BW:.0f} FLOP/B")


def fa_figure1():
    head("Slide 16: FlashAttention Figure 1 (A100)")
    show("speedup (caption, input)", f"{FA_FIG1_SPEEDUP}x")
    show("SRAM / HBM bandwidth, A100 (19 / 1.5 TB/s)", f"{FA_FIG1_BW[0] / FA_FIG1_BW[1]:.1f}x")


def tiles():
    head("Slide 17: tiles")
    full = N_TILE_EX ** 2 * 4
    tile = TILE ** 2 * 4
    show("full matrix", f"{full / 1e6:.1f} MB")
    show("tile", f"{tile / 1e3:.1f} KB")
    show("tiles per side, total", f"{N_TILE_EX // TILE}, {(N_TILE_EX // TILE) ** 2:,}")
    show("16 heads", f"{16 * full / 1e9:.2f} GB")


def online_softmax():
    head("Slide 18: online softmax toy")
    x, v = [1, 3, 2, 5], [10, 20, 30, 40]
    m = max(x)
    ell = sum(math.exp(xi - m) for xi in x)
    o = sum(math.exp(xi - m) * vi for xi, vi in zip(x, v))
    m1 = max(x[:2])
    l1 = sum(math.exp(xi - m1) for xi in x[:2])
    o1 = sum(math.exp(xi - m1) * vi for xi, vi in zip(x[:2], v[:2]))
    f = math.exp(m1 - m)
    show("whole-row l", f"{ell:.3f}")
    show("block 1 l", f"{l1:.3f}")
    show("rescale factor e^(3-5)", f"{f:.3f}")
    show("old sum rescaled", f"{l1 * f:.3f}")
    show("e^-3", f"{math.exp(-3):.3f}")
    show("block 2 l", f"{l1 * f + math.exp(2 - 5) + 1:.3f}")
    show("whole-row O", f"{o:.2f}")
    show("block 1 O", f"{o1:.2f}")
    show("old O rescaled", f"{o1 * f:.2f}")
    show("30e^-3", f"{30 * math.exp(-3):.2f}")
    show("O/l", f"{o / ell:.1f}")


def flash_store():
    head("Slide 19: stored per head, 4,096 tokens, FP32")
    n, d = N_TILE_EX, DHEAD
    qkvo = 4 * n * d * 4
    std = qkvo + 2 * n * n * 4
    fa = qkvo + 2 * n * 4
    show("standard (Q, K, V, O, S, P)", f"{std / 1e6:.1f} MB")
    show("FlashAttention (Q, K, V, O, m, l)", f"{fa / 1e6:.1f} MB")
    show("ratio", f"{std / fa:.1f}x")


def fa_table():
    head("Slide 20: FlashAttention table")
    show("more FLOPs", f"{100 * (FA_GFLOP[1] / FA_GFLOP[0] - 1):.0f}%")
    show("fewer HBM bytes", f"{FA_HBM_GB[0] / FA_HBM_GB[1]:.1f}x")
    show("faster", f"{FA_MS[0] / FA_MS[1]:.1f}x")
    step = static_gb() + flash_act_gb()
    show("GPT-2 XL step with FlashAttention", f"{step:.1f} GB, {step - h100_gb():.1f} GB over")


def sdpa():
    head("Slide 21: SDPA profile (data/sdpa_profile.txt)")
    txt = (HERE / "data" / "sdpa_profile.txt").read_text().splitlines()
    tot = [float(l.split()[-1]) for l in txt if l.startswith("aten::scaled_dot_product_attention ")]
    show("default vs math total ms", f"{tot[0]:.2f} vs {tot[1]:.2f}: {tot[1] / tot[0]:.1f}x")
    show("one head-batch score matrix, FP32", f"{HEADS * SEQ * SEQ * 4 / 1e6:.1f} MB (25 x 1,024^2 x 4 B)")
    show("output tensor", f"{HEADS * SEQ * DHEAD * 4 / 1e6:.1f} MB")


def batch_fit():
    head("Slide 23: batch fit")
    show("activations after FlashAttention", f"{flash_act_gb():.1f} GB")
    show("per sequence", f"{per_seq_gb():.3f} GB")
    show("sequences that fit", f"{(h100_gb() - static_gb()) / per_seq_gb():.1f}")
    for b in (16, 32):
        show(f"batch {b}", f"{static_gb() + b * per_seq_gb():.1f} GB")


def accumulate():
    head("Slide 24: accumulation")
    show("2 x 16", f"{static_gb() + 16 * per_seq_gb():.1f} GB")


def checkpoint():
    head("Slide 27: segment checkpointing")
    show("one segment input", f"{BATCH * SEQ * HIDDEN * 2 / 1e6:.1f} MB")
    show("eight inputs", f"{seg_inputs_gb():.2f} GB")
    act = seg_inputs_gb() + LIVE_SEG_GB
    show("activations", f"{act:.1f} GB, {flash_act_gb() / act:.1f}x less")
    show("step", f"{static_gb() + act:.1f} GB")
    show("arithmetic factor (fwd+bwd = 3 fwd, +1)", f"{4 / 3:.3f}")
    d = run_days(PEAK["H100"]["FP16"])
    show("run with checkpointing", f"{d * 4 / 3:.1f} days ({d * 4 / 3 - d:.1f} more)")


def sqrt_ckpt():
    head("Slide 28: sqrt(L)")
    units = {k: k + LAYERS / k for k in range(1, LAYERS + 1)}
    kmin = min(units, key=units.get)
    show("sqrt(48)", f"{math.sqrt(LAYERS):.2f}")
    show("k=1, k=48", f"{units[1]:.0f}, {units[48]:.0f}")
    show("minimum", f"k = {kmin}: {units[kmin]:.2f}; k = 8: {units[8]:.0f}")
    show("less than 48 units", f"{100 * (1 - units[kmin] / LAYERS):.0f}%")
    show("Chen: 48 -> 7 GB", f"{CHEN_GB[0] / CHEN_GB[1]:.1f}x less")


def korthikanti():
    head("Slide 29: selective recomputation (Korthikanti et al.)")
    c = k_score_coeff()
    show("5as/h for GPT-2 XL", f"{c:.0f}")
    full = k_bytes_per_layer(34 + c)
    show("per layer, everything stored", f"{full / GB:.2f} GB; x48 = {LAYERS * full / GB:.1f} GB")
    show("score share 5as/h / (34 + 5as/h)", f"{100 * c / (34 + c):.1f}%")
    show("selective: 34 sbh per layer x48", f"{LAYERS * k_bytes_per_layer(34) / GB:.1f} GB")
    show("full recompute: 2 sbh per layer x48", f"{LAYERS * k_bytes_per_layer(2) / GB:.2f} GB")
    show("5as equals the scores: 5 a s^2 b", f"{scores_layer_bytes() / GB:.3f} GB vs {k_bytes_per_layer(c) / GB:.3f} GB")
    base = K_TABLE4["No recompute"]
    for name, ms in K_TABLE4.items():
        show(f"Table 4 {name}", f"{ms} ms, +{100 * (ms / base - 1):.0f}%")
    show("530B MFU gain", f"{K_MFU[0]} -> {K_MFU[1]}: {100 * (K_MFU[1] / K_MFU[0] - 1):.0f}% faster")


def vote():
    head("Slide 30: vote")
    fp32 = P * sum(FP32_BYTES) / GB + fp32_act_gb()
    final = static_gb() + 16 * per_seq_gb()
    show("598 -> 66.8", f"{fp32 / final:.1f}x smaller")


def hf_flags():
    head("Slide 31: Hugging Face arguments")
    show("checkpointing, our estimate vs docs", f"+{100 * (4 / 3 - 1):.0f}% arithmetic vs ~{HF_CKPT_SLOWER}% slower (docs)")
    show("docs example effective batch", f"{4 * 16}")


def alexnet_back():
    head("Slide 34: back to AlexNet")
    fp32 = P * sum(FP32_BYTES) / GB + fp32_act_gb()
    show("FP32 step over one GTX 580", f"{fp32 / GTX580_GB:.0f}x")
    d = run_days(PEAK["H100"]["FP16"])
    show("checkpointing extra days", f"{d * 4 / 3 - d:.1f}")
    show("busy", f"{BUSY}%")


if __name__ == "__main__":
    for fn in (hook, predict, recall, ladder, formats, swamp, loss_scale, diagnose_trace,
               days, layer_bytes, attention_io, fa_figure1, tiles, online_softmax,
               flash_store, fa_table, sdpa, batch_fit, accumulate, checkpoint, sqrt_ckpt,
               korthikanti, vote, hf_flags, alexnet_back):
        fn()
