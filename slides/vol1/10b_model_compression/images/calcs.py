#!/usr/bin/env python3
"""Every derived number in Lecture 10b (Model Compression, Part 2), computed
from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports from this file so figures and slides share
one computation. "Chapter" means Model Compression in Volume I.

Run from this folder: python3 calcs.py
"""

import math

import numpy as np


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


def r(x, nd=2):
    """Round half away from zero (Python's round() is half to even)."""
    q = 10 ** nd
    return math.floor(abs(x) * q + 0.5) / q * (1 if x >= 0 else -1)


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
# Krishnamoorthi 2018 (arXiv 1806.08342), Table 3 (post-training INT8), Table 4
# (QAT), Table 5 (4-bit weights), Table 7 (Pixel 2, ms).
MBV2_FLOAT, MBV2_PER_TENSOR, MBV2_PER_CHANNEL, MBV2_QAT = 71.9, 0.1, 69.8, 71.1
RN50_FLOAT, RN50_PER_TENSOR, RN50_PER_CHANNEL = 75.2, 75.0, 75.1
RN50_W4_QAT, MBV2_W4_QAT = 73.2, 62.0
MBV1_SYM, MBV1_ASYM, MBV1_FLOAT = 59.1, 70.3, 70.9
PX2_FP32_CPU, PX2_INT8_CPU, PX2_INT8_DSP = 105.0, 63.0, 15.5
IMAGENET_CLASSES = 1000

# GPT-2 XL decoding on the chapter's phone (Napkin Math 10.4, table 1):
# 1.5B parameters; 48 layers, 25 heads of 64; 1,024-token context; 51.2 GB/s.
GPT2_PARAMS = 1.5e9
GPT2_LAYERS, GPT2_HEADS, GPT2_HEAD_DIM, CTX = 48, 25, 64, 1024
PHONE_BW = 51.2e9            # bytes/s
FP16_B, INT4_B, FP32_B, INT8_B = 2.0, 0.5, 4.0, 1.0   # bytes per value
GB = 1e9

# Llama 2 7B as llama-bench reports it (github.com/ggml-org/llama.cpp,
# discussion 4167, "params 6.74 B"; sizes in GiB).
LLAMA2_PARAMS = 6.74e9
GIB = 2 ** 30


# ---------------------------------------------------------------------------
# Slide 2: hook
# ---------------------------------------------------------------------------
def hook():
    head("Slide 2: the same INT8 recipe keeps ResNet-50, not MobileNetV2")
    show("chance on 1,000 classes (%)", 100 / IMAGENET_CLASSES)
    show("ResNet-50 loss under per-tensor INT8 (points)", r(RN50_FLOAT - RN50_PER_TENSOR, 1))


# ---------------------------------------------------------------------------
# Slide 3: recall (Part 1 budget terms)
# ---------------------------------------------------------------------------
def recall():
    head("Slide 3: recall")
    w = GPT2_PARAMS * FP16_B
    show("GPT-2 XL FP16 weights (GB)", w / GB)
    show("stream once at 51.2 GB/s (ms)", r(w / PHONE_BW * 1e3, 1))


# ---------------------------------------------------------------------------
# Slide 6: formats (bit fields)
# ---------------------------------------------------------------------------
def formats():
    head("Slide 6: formats")
    for name, bits in (("BF16/FP16", 16), ("FP8", 8), ("INT8", 8), ("INT4", 4)):
        show(f"{name} smaller than FP32 (x)", 32 // bits)
    show("INT8 levels", 2 ** 8)
    show("INT4 levels", 2 ** 4)


# ---------------------------------------------------------------------------
# Slide 7: real 4-bit files (GGUF)
# Inputs: TheBloke/Llama-2-7B-GGUF on Hugging Face (commit b4e04e1), README
# "Provided files" table (Name, Bits, Size) and the byte sizes the Hub API
# lists for each file (api/models/TheBloke/Llama-2-7B-GGUF/tree/main), read
# 2026-10-18. F16 file: ggml-org/Llama-2-7B-GGUF, Llama-2-7B-F16.gguf.
# Block layout: llama.cpp v0.7.0, ggml/src/ggml-common.h, block_q4_0 and
# block_q8_0 (one ggml_half scale d plus QK = 32 quants).
# ---------------------------------------------------------------------------
GGUF_FILES = [  # name, nominal bits (README), bytes (Hub API)
    ("llama-2-7b.Q2_K.gguf", 2, 2825940672),
    ("llama-2-7b.Q3_K_M.gguf", 3, 3298004672),
    ("llama-2-7b.Q4_0.gguf", 4, 3825807040),
    ("llama-2-7b.Q4_K_M.gguf", 4, 4081004224),
    ("llama-2-7b.Q6_K.gguf", 6, 5529194176),
    ("llama-2-7b.Q8_0.gguf", 8, 7161089728),
]
F16_FILE_BYTES = 13478105600
QK = 32                     # weights per block (QK4_0, QK8_0)
HALF_BYTES = 2              # sizeof(ggml_half)


def q4_0_bits():
    block = HALF_BYTES + QK // 2
    return block, block * 8 / QK


def q8_0_bits():
    block = HALF_BYTES + QK
    return block, block * 8 / QK


def gguf():
    head("Slide 7: a real 4-bit file stores more than 4 bits per weight")
    for name, bits, nbytes in GGUF_FILES:
        show(f"{name} size (GB, 1e9)", r(nbytes / GB, 2))
        show(f"{name} bits per weight (6.74B params)", r(nbytes * 8 / LLAMA2_PARAMS, 2))
    show("F16 file (GB)", r(F16_FILE_BYTES / GB, 2))
    show("F16 file bits per weight", r(F16_FILE_BYTES * 8 / LLAMA2_PARAMS, 2))
    b, bpw = q4_0_bits()
    show("block_q4_0 bytes (2 + 32/2)", b)
    show("block_q4_0 bits per weight", bpw)
    b8, bpw8 = q8_0_bits()
    show("block_q8_0 bytes (2 + 32)", b8)
    show("block_q8_0 bits per weight", bpw8)
    show("Q4_0 file smaller than F16 (x)", r(F16_FILE_BYTES / GGUF_FILES[2][2], 2))
    show("Q4_0 scale overhead over 4 bits (%)", r((bpw - 4) / 4 * 100, 1))


# ---------------------------------------------------------------------------
# Slide 8: float grid against integer grid (FP8 E4M3 definition,
# Micikevicius et al. 2022, Table 1: max 448, min subnormal 2^-9)
# ---------------------------------------------------------------------------
def e4m3_values():
    vals = []
    for ef in range(16):
        for mf in range(8):
            if ef == 15 and mf == 7:
                continue  # the single NaN pattern
            vals.append((mf / 8) * 2.0 ** -6 if ef == 0 else (1 + mf / 8) * 2.0 ** (ef - 7))
    return np.array(sorted(vals))


def grids():
    head("Slide 8: FP8 grid against INT8 grid")
    v = e4m3_values()
    pos = v[v > 0]
    show("E4M3 positive values", len(pos))
    show("E4M3 positive values below 1", int((pos < 1).sum()))
    show("E4M3 values from 256 to 448", int((pos >= 256).sum()))
    show("smallest positive", pos.min())
    show("INT8 step to cover 448 (448/127)", r(448 / 127, 2))
    show("everything below half a step rounds to 0", r(448 / 127 / 2, 2))


# ---------------------------------------------------------------------------
# Slide 9: energy (Horowitz 2014 via chapter table 8, pJ at 45 nm)
# ---------------------------------------------------------------------------
ENERGY = [("Add", 0.90, 0.03), ("Multiply", 3.70, 0.20), ("Multiply-accumulate", 4.60, 0.30)]
DRAM_32 = 640.0
KWS_FP32_KB, SRAM_KB = 800, 512   # Lighthouse 10.2


def energy_ratio(f32, i8):
    return f32 / i8


def energy():
    head("Slide 9: energy")
    for name, f32, i8 in ENERGY:
        show(f"{name} FP32/INT8", r(energy_ratio(f32, i8), 1))
    show("DRAM read / INT8 MAC", r(DRAM_32 / ENERGY[2][2], 0))
    show("keyword spotter INT8 (KB)", KWS_FP32_KB * INT8_B / FP32_B)


# ---------------------------------------------------------------------------
# Slide 12: 3-bit toy grid (alpha, beta, b chosen for the slide)
# ---------------------------------------------------------------------------
TOY_ALPHA, TOY_BETA, TOY_B = -1.0, 2.5, 3
TOY_S = (TOY_BETA - TOY_ALPHA) / (2 ** TOY_B - 1)
TOY_Z = round(-TOY_ALPHA / TOY_S)
TOY_SAMPLES = [-1.4, -0.3, 0.8, 1.6, 2.9]


def quant_toy(x):
    q = int(np.clip(math.floor(x / TOY_S + TOY_Z + 0.5), 0, 2 ** TOY_B - 1))
    return q, TOY_S * (q - TOY_Z)


def affgrid():
    head("Slide 12: 3-bit toy grid")
    show("s", TOY_S)
    show("z", TOY_Z)
    for x in TOY_SAMPLES + [2.2]:
        q, xr = quant_toy(x)
        show(f"x = {x}: code, reads back, error", (q, xr, r(abs(xr - x), 2)))
    show("max rounding error s/2", TOY_S / 2)


# ---------------------------------------------------------------------------
# Slide 13: Napkin Math 10.1 (range [-1, 3], UINT8)
# ---------------------------------------------------------------------------
NM_ALPHA, NM_BETA = -1.0, 3.0
NM_S = (NM_BETA - NM_ALPHA) / 255
NM_Z_EXACT = -NM_ALPHA / NM_S
NM_Z = math.floor(NM_Z_EXACT + 0.5)
NM_X = [-1.0, 1.0, 0.0, 3.0]


def nm_codes():
    return [int(np.clip(math.floor(x / NM_S + NM_Z + 0.5), 0, 255)) for x in NM_X]


def napkin():
    head("Slide 13: real zero lands on 64")
    show("s = 4/255", r(NM_S, 4))
    show("-alpha/s", r(NM_Z_EXACT, 2))
    show("z", NM_Z)
    show("codes for -1, 1, 0, 3", nm_codes())
    show("-1 reads back as", r(NM_S * (0 - NM_Z), 3))


# ---------------------------------------------------------------------------
# Slide 14: PyTorch reproduces Napkin Math 10.1. The values on the slide are
# printed by the tool itself (torch 2.13.0, CPU); this block only checks them.
# Command: python -W ignore images/torch_qparams.py
# ---------------------------------------------------------------------------
def torch_check():
    head("Slide 14: PyTorch check (expected from slide 13)")
    show("scale (float32 of 4/255)", float(np.float32(NM_S)))
    show("dequantized codes", [r(NM_S * (q - NM_Z), 4) for q in nm_codes()])


# ---------------------------------------------------------------------------
# Slide 15: listing 2, symmetric INT8 (chapter section 10.3.3)
# ---------------------------------------------------------------------------
L2_W = [0.127, -0.084, 0.392, -0.203]


def listing2():
    head("Slide 15: listing 2 round trip")
    s = max(abs(w) for w in L2_W) / 127
    show("s = 0.392/127", r(s, 5))
    for w in L2_W:
        q = int(math.copysign(math.floor(abs(w) / s + 0.5), w))
        show(f"w = {w}: code, error", (q, r(abs(q * s - w), 5)))
    show("s/2", r(s / 2, 5))
    show("toy grid s/2", TOY_S / 2)


# ---------------------------------------------------------------------------
# Slide 16: symmetric against asymmetric (chapter figure 22; ReLU6 [0, 6])
# ---------------------------------------------------------------------------
ASYM_LO, ASYM_HI = -0.5, 1.5


def zeropoint():
    head("Slide 16: zero-point")
    s = (ASYM_HI - ASYM_LO) / 255
    show("asymmetric s = 2/255", r(s, 4))
    show("real zero lands at code", -128 + math.floor(-ASYM_LO / s + 0.5))
    show("ReLU6 symmetric step 6/127", r(6 / 127, 3))
    show("ReLU6 asymmetric step 6/255", r(6 / 255, 3))
    show("MobileNet-v1 symmetric cost (points)", r(MBV1_ASYM - MBV1_SYM, 1))


ZP_ASYM = -128 + math.floor(-ASYM_LO / ((ASYM_HI - ASYM_LO) / 255) + 0.5)


# ---------------------------------------------------------------------------
# Slide 17: one INT8 output by hand (scales: largest magnitude / 127)
# ---------------------------------------------------------------------------
IL_W = [0.5, -0.25, 0.75]
IL_X = [1.0, 2.0, 0.5]
IL_SY = 1 / 127


def intlayer():
    head("Slide 17: integer layer")
    sw, sx = max(map(abs, IL_W)) / 127, max(map(abs, IL_X)) / 127
    rq = lambda v, s: int(math.copysign(math.floor(abs(v) / s + 0.5), v))  # noqa: E731
    wq = [rq(w, sw) for w in IL_W]
    xq = [rq(x, sx) for x in IL_X]
    prods = [a * b for a, b in zip(wq, xq)]
    acc = sum(prods)
    show("0.5/(0.75/127)", r(0.5 / sw, 1))
    show("1.0/(2/127)", r(1.0 / sx, 1))
    show("w_q", wq)
    show("x_q", xq)
    show("products", prods)
    show("INT32 sum", acc)
    show("sum x s_w s_x", r(acc * sw * sx, 3))
    show("exact", sum(a * b for a, b in zip(IL_W, IL_X)))
    m = sw * sx / IL_SY
    show("M = s_w s_x / s_y", r(m, 4))
    show("y_q", math.floor(acc * m + 0.5))
    show("worst-case products in INT32 (2^31/127^2)", 2 ** 31 // 127 ** 2)


# ---------------------------------------------------------------------------
# Slide 19: clipping (chapter figure 21, illustrative range 2.5, clip 1.1)
# ---------------------------------------------------------------------------
def clipping():
    head("Slide 19: clipping")
    a, b = 2.5 / 255, 1.1 / 255
    show("step over [0, 2.5]", r(a, 4))
    show("step over [0, 1.1]", r(b, 4))
    show("finer (x)", r(a / b, 1))


# ---------------------------------------------------------------------------
# Slide 20: measured calibration (Wu et al. 2020, arXiv 2004.09602, Table 5:
# weights per-channel max; activations calibrated as listed; top-1 or F1)
# ---------------------------------------------------------------------------
WU_T5 = {  # model: (fp32, max, entropy, 99.9%, 99.99%, 99.999%)
    "ResNet50 v1.5": (76.16, 75.82, 76.05, 75.68, 75.98, 75.97),
    "MobileNet v2": (71.88, 69.41, 70.28, 70.68, 71.14, 70.72),
    "Inception v4": (79.71, 0.12, 79.60, 78.16, 79.63, 79.12),
    "EfficientNet b0": (76.85, 22.3, 72.06, 70.87, 68.33, 51.88),
    "BERT Large": (91.01, 85.92, 37.40, 26.18, 89.59, 90.20),
}


def wu():
    head("Slide 20: measured calibration (Wu et al. 2020, Table 5)")
    cols = ["max", "entropy", "99.9%", "99.99%", "99.999%"]
    for m, v in WU_T5.items():
        best = int(np.argmax(v[1:]))
        show(f"{m}: best method, loss from fp32", (cols[best], r(v[0] - v[1 + best], 2)))


# ---------------------------------------------------------------------------
# Slide 21: per-channel scales (Nagel et al. 2019 section 3.1 example;
# Krishnamoorthi Tables 3, 4)
# ---------------------------------------------------------------------------
def perchannel():
    head("Slide 21: per-channel scales")
    show("step for [-128, 128] at 8 bits (256/255)", r(256 / 255, 2))
    show("QAT gain over per-channel PTQ (points)", r(MBV2_QAT - MBV2_PER_CHANNEL, 1))


# ---------------------------------------------------------------------------
# Slide 26: the cliff (Krishnamoorthi Tables 3 and 5)
# ---------------------------------------------------------------------------
def cliff():
    head("Slide 26: the cliff")
    show("MobileNetV2 8-bit per-channel gap (points)", r(MBV2_FLOAT - MBV2_PER_CHANNEL, 1))
    show("ResNet-50 4-bit QAT gap (points)", r(RN50_FLOAT - RN50_W4_QAT, 1))
    show("MobileNetV2 4-bit QAT gap (points)", r(MBV2_FLOAT - MBV2_W4_QAT, 1))


# ---------------------------------------------------------------------------
# Slide 31: two mechanisms on the roofline (Napkin Math 10.2 and 10.4;
# Frantar et al. 2023, Table 6)
# ---------------------------------------------------------------------------
H100_INT8, H100_FP16 = 1979, 989
GPTQ_FP16_MS, GPTQ_3BIT_MS = 230, 71


def kv_bytes():
    return 2 * GPT2_LAYERS * GPT2_HEADS * GPT2_HEAD_DIM * CTX * FP16_B


def decode():
    w16 = GPT2_PARAMS * FP16_B
    w4 = GPT2_PARAMS * INT4_B
    kv = kv_bytes()
    t16, t4 = w16 + kv, w4 + kv
    sep = w4 + w16 + w16 + kv     # read packed, write FP16, read FP16, read cache
    return dict(w16=w16, w4=w4, kv=kv, t16=t16, t4=t4, sep=sep)


def roofline():
    head("Slide 31: roofline")
    d = decode()
    show("FP16 weights (GB)", d["w16"] / GB)
    show("KV cache (GB)", r(d["kv"] / GB, 2))
    show("FP16 bytes per token (GB)", r(d["t16"] / GB, 2))
    show("FP16 ms per token", r(d["t16"] / PHONE_BW * 1e3, 0))
    show("FP16 tokens/s", r(PHONE_BW / d["t16"], 0))
    show("INT4 weights (GB)", d["w4"] / GB)
    show("INT4 bytes per token (GB)", r(d["t4"] / GB, 2))
    show("INT4 ms per token", r(d["t4"] / PHONE_BW * 1e3, 0))
    show("INT4 tokens/s", r(PHONE_BW / d["t4"], 0))
    show("speedup (x)", r(d["t16"] / d["t4"], 1))
    show("H100 INT8/FP16 (x)", r(H100_INT8 / H100_FP16, 2))
    show("GPTQ measured (x)", r(GPTQ_FP16_MS / GPTQ_3BIT_MS, 2))


# ---------------------------------------------------------------------------
# Slide 32: llama-bench on Apple Silicon (github.com/ggml-org/llama.cpp,
# discussion 4167, summary table and the M2 Ultra post; build 8e672ef; Llama 2
# 7B; PP = prompt processing, 512 tokens; TG = text generation, batch one;
# tokens/s). Read 2026-10-18 via the GitHub API.
# ---------------------------------------------------------------------------
LLAMA_BENCH = {  # chip (BW GB/s, GPU cores): F16 PP, F16 TG, Q8_0 PP, Q8_0 TG, Q4_0 PP, Q4_0 TG
    "M1 Pro (200, 16)": (302.14, 12.75, 270.37, 22.34, 266.25, 36.41),
    "M2 Ultra (800, 76)": (1401.85, 41.02, 1248.59, 66.64, 1238.48, 94.27),
    "M3 Max (400, 40)": (779.17, 25.09, 757.64, 42.75, 759.70, 66.31),
    "M4 Max (546, 40)": (922.83, 31.64, 891.94, 54.05, 885.68, 83.06),
}
LB_SIZE_GIB = {"F16": 12.55, "Q8_0": 6.67, "Q4_0": 3.56}   # llama-bench "size" column


def llama_ratios(chip):
    f16pp, f16tg, q8pp, q8tg, q4pp, q4tg = LLAMA_BENCH[chip]
    return dict(tg8=q8tg / f16tg, tg4=q4tg / f16tg, pp8=q8pp / f16pp, pp4=q4pp / f16pp)


def llama():
    head("Slide 32: llama-bench, bits help generation, not prompt processing")
    for chip in LLAMA_BENCH:
        rt = llama_ratios(chip)
        show(f"{chip} TG Q8_0/F16, Q4_0/F16; PP Q8_0/F16, Q4_0/F16",
             (r(rt["tg8"], 2), r(rt["tg4"], 2), r(rt["pp8"], 2), r(rt["pp4"], 2)))
    show("file F16/Q8_0 (x)", r(LB_SIZE_GIB["F16"] / LB_SIZE_GIB["Q8_0"], 2))
    show("file F16/Q4_0 (x)", r(LB_SIZE_GIB["F16"] / LB_SIZE_GIB["Q4_0"], 2))
    for q in LB_SIZE_GIB:
        show(f"{q} bits per weight", r(LB_SIZE_GIB[q] * GIB * 8 / LLAMA2_PARAMS, 2))
    f16tg, q4tg = LLAMA_BENCH["M2 Ultra (800, 76)"][1], LLAMA_BENCH["M2 Ultra (800, 76)"][5]
    show("M2 Ultra bytes streamed per s, F16 (GB/s)", r(f16tg * LB_SIZE_GIB["F16"] * GIB / GB, 0))
    show("M2 Ultra bytes streamed per s, Q4_0 (GB/s)", r(q4tg * LB_SIZE_GIB["Q4_0"] * GIB / GB, 0))


# ---------------------------------------------------------------------------
# Slide 33: separate dequantize pass (Napkin Math 10.4; section 10.4.1)
# ---------------------------------------------------------------------------
def dequant():
    head("Slide 33: separate dequantize pass")
    d = decode()
    show("separate pass (GB)", r(d["sep"] / GB, 2))
    show("separate pass (ms)", r(d["sep"] / PHONE_BW * 1e3, 0))
    show("against FP16 (x slower)", r(d["sep"] / d["t16"], 2))


# ---------------------------------------------------------------------------
# Slide 34: Pixel 2 (Krishnamoorthi Table 7; Napkin Math 10.3)
# ---------------------------------------------------------------------------
FRAME_MS = 1000 / 30


def pixel2():
    head("Slide 34: Pixel 2")
    show("INT8 CPU speedup (x)", r(PX2_FP32_CPU / PX2_INT8_CPU, 1))
    show("INT8 DSP speedup (x)", r(PX2_FP32_CPU / PX2_INT8_DSP, 1))
    show("frame budget at 30 fps (ms)", r(FRAME_MS, 1))
    show("512-bit register: FP32 lanes, INT8 lanes", (512 // 32, 512 // 8))


# ---------------------------------------------------------------------------
# Slide 37: back to MobileNetV2; Part 1's 7B model (7 billion parameters)
# ---------------------------------------------------------------------------
P7B = 7e9


def close():
    head("Slide 37: synthesis")
    show("7B FP16 (GB)", P7B * FP16_B / GB)
    show("7B INT4, 4 bits only (GB)", P7B * INT4_B / GB)
    show("7B INT4 at Q4_0's 4.5 bits (GB)", r(P7B * q4_0_bits()[1] / 8 / GB, 1))


if __name__ == "__main__":
    hook()
    recall()
    formats()
    gguf()
    grids()
    energy()
    affgrid()
    napkin()
    torch_check()
    listing2()
    zeropoint()
    intlayer()
    clipping()
    wu()
    perchannel()
    cliff()
    roofline()
    llama()
    dequant()
    pixel2()
    close()
