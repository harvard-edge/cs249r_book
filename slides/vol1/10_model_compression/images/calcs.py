#!/usr/bin/env python3
"""Every derived number in Lecture 10 (Model Compression, Part 1).

One block per slide, in deck order. Inputs carry their source; every printed
value is what the slide and its note show. "Chapter" means Model Compression
in Volume I (section numbers as rendered). make_figs.py imports from here.

Run from this folder: python3 calcs.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
GB, MB, KB = 1e9, 1e6, 1e3
OUT = []


def show(label, value):
    OUT.append((label, value))
    print(f"{label}: {value}")


# --- Frame: A 7B model needs 14 GB; the phone has 8 GB (chapter Sec. 10.1) --
P_175B, P_7B = 175e9, 7e9            # parameters (chapter Sec. 10.1)
BYTES_FP16, BYTES_INT4 = 2, 0.5       # bytes per weight
PHONE_BW = 51.2 * GB                  # flagship phone LPDDR bus (chapter Sec. 10.1)
DRAM_PJ_PER_32BIT = 640               # 32-bit DRAM read, 45 nm (Horowitz 2014; chapter Sec. 10.3.2)
mem_175b = P_175B * BYTES_FP16
mem_7b = P_7B * BYTES_FP16
mem_7b_int4 = P_7B * BYTES_INT4
t_token_7b = mem_7b / PHONE_BW
e_token_7b = mem_7b * 8 / 32 * DRAM_PJ_PER_32BIT * 1e-12
show("ladder 175B FP16 (GB)", f"{mem_175b / GB:.0f}")
show("ladder 7B FP16 (GB)", f"{mem_7b / GB:.0f}")
show("ladder 7B INT4 (GB)", f"{mem_7b_int4 / GB:.1f}")
show("7B token time at 51.2 GB/s (ms)", f"{t_token_7b * 1e3:.0f}  (slide: about 270)")
show("7B DRAM energy per token (J)", f"{e_token_7b:.2f}  (slide: about 2)")
show("7B DRAM power while decoding (W)", f"{e_token_7b / t_token_7b:.1f}  (slide: about 8)")

# --- Frame: smallest model to pass 60 percent MMLU (Epoch AI via mlsysim) ----
GOPHER, QWEN_CODER_3B = 280e9, 3e9    # parameters (chapter Sec. 10.1)
show("MMLU>=60 shrink Gopher to Qwen2.5-Coder-3B (x)", f"{GOPHER / QWEN_CODER_3B:.0f}")

# --- Frame: blur model, 8 FPS against 30 (Example 10.1, composite) ----------
FPS_CPU, FPS_TARGET = 8, 30
show("blur frame time at 8 FPS (ms)", f"{1e3 / FPS_CPU:.0f}")
show("blur budget at 30 FPS (ms)", f"{1e3 / FPS_TARGET:.0f}")

# --- Frame: weight streaming at 51.2 GB/s (chapter table 1) -----------------
MNV2_PARAMS = 3.5e6                   # MobileNetV2 (chapter table 1)
MNV2_FP32 = MNV2_PARAMS * 4           # 14 MB
R50_FP32 = 102.4 * MB                 # ResNet-50 FP32 (table 1)
GPT2XL_FP16 = 3 * GB                  # GPT-2 XL FP16 (table 1)
GPT2XL_INT4 = GPT2XL_FP16 / 4         # 4 of 16 bits
FRAME = 1 / FPS_TARGET
stream = {
    "MobileNetV2 FP32": MNV2_FP32 / PHONE_BW,
    "ResNet-50 FP32": R50_FP32 / PHONE_BW,
    "GPT-2 XL FP16": GPT2XL_FP16 / PHONE_BW,
    "GPT-2 XL INT4": GPT2XL_INT4 / PHONE_BW,
}
show("MobileNetV2 FP32 weights (MB)", f"{MNV2_FP32 / MB:.0f}")
for k, v in stream.items():
    show(f"stream {k} (ms)", f"{v * 1e3:.2f}")
show("MobileNetV2 stream share of 33 ms frame (%)",
     f"{100 * stream['MobileNetV2 FP32'] / FRAME:.1f}  (slide: about 1%)")

# --- Frame: accuracy-efficiency frontier (Howard et al. 2019, Table 3) ------
# Published: ImageNet top-1 (%), Pixel 1 float latency (ms), one large core,
# TFLite benchmark tool. (name, top1, P-1 ms, MAdds M, params M)
MNV3_TABLE3 = [
    ("V3-Large 1.0", 75.2, 51, 219, 5.4),
    ("V3-Large 0.75", 73.3, 39, 155, 4.0),
    ("MnasNet-A1", 75.2, 71, 315, 3.9),
    ("Proxyless", 74.6, 72, 320, 4.0),
    ("V2 1.0", 72.0, 64, 300, 3.4),
    ("V3-Small 1.0", 67.4, 15.8, 56, 2.5),
    ("V3-Small 0.75", 65.4, 12.8, 44, 2.0),
    ("Mnas-small", 64.9, 20.3, 65.1, 1.9),
    ("V2 0.35", 60.8, 16.6, 59.2, 1.6),
]


def frontier(rows):
    """Names of rows no other row beats on both latency and accuracy."""
    keep = []
    for n, a, t, *_ in rows:
        if not any((t2 <= t and a2 >= a) and (t2 < t or a2 > a) for _, a2, t2, *_ in rows):
            keep.append(n)
    return keep


FRONTIER = frontier(MNV3_TABLE3)
show("frontier models (Pixel 1, float)", ", ".join(FRONTIER))
v2 = next(r for r in MNV3_TABLE3 if r[0] == "V2 1.0")
v3l75 = next(r for r in MNV3_TABLE3 if r[0] == "V3-Large 0.75")
show("V3-Large 0.75 vs V2 1.0: latency ratio (x)", f"{v2[2] / v3l75[2]:.2f}")
show("V3-Large 0.75 vs V2 1.0: top-1 gain (points)", f"{v3l75[1] - v2[1]:.1f}")
show("V3-Large 0.75 vs V2 1.0: ms saved", f"{v2[2] - v3l75[2]:.0f}")

# --- Frame: pruning matrices (make_figs: 6x6, half zeroed) ------------------
show("6x6 unstructured zeros", f"{36 // 2}")

# --- Frame: magnitude pruning keeps four of nine (chapter listing 1) --------
L1 = [0.8, 0.1, -0.7, 0.05, -0.9, 0.03, -0.6, 0.02, 0.4]
kept = [w for w in L1 if abs(w) >= 0.5]
show("listing 1 kept", f"{len(kept)} of {len(L1)}: {kept}")
show("listing 1 sparsity (%)", f"{100 * (len(L1) - len(kept)) / len(L1):.0f}")

# --- Frame: PyTorch prune output (torch_demos.py) ---------------------------
# Dense FP32 3x3 = 36 B; CSR with int64 indices: 4 values x 4 B + 4 column
# indices x 8 B + 4 row pointers x 8 B, matching the printed 36 and 80.
show("torch dense bytes", f"{9 * 4}")
show("torch CSR bytes (int64 indices)", f"{4 * 4 + 4 * 8 + (3 + 1) * 8}")

# --- Frame: sparse product, 4x4 (chapter Sec. 10.4.2, Sec. 10.7) ------------
M4 = [[2, 0, 0, 1], [0, 3, 0, 0], [4, 0, 5, 0], [0, 0, 0, 6]]
nnz = sum(v != 0 for r in M4 for v in r)
show("4x4 dense multiplies", f"{16}")
show("4x4 nonzero products", f"{nnz}")
show("4x4 skipped multiplies", f"{16 - nnz}")
show("4x4 zero share (%)", f"{100 * (16 - nnz) / 16:.1f}")
show("4x4 dense bytes FP32", f"{16 * 4}")
show("4x4 CSR values+cols (B)", f"{nnz * (4 + 4)}")
show("4x4 CSR row pointers (B)", f"{(4 + 1) * 4}")
show("4x4 CSR total (B)", f"{nnz * 8 + 5 * 4}")
show("4096-row layer row pointers", f"{4096 + 1:,}")

# --- Frame: sparse formats (chapter Sec. 10.4.2.2 scenario, Sec. 10.7) ------
show("FP32 + 16-bit index (B per nonzero)", f"{4 + 2}")
show("wearable survivors with indices (MB)", f"{2 * (4 + 2) / 4:.0f}")
show("CSR bytes per nonzero, 32-bit index", f"{4 + 4}")
show("COO bytes per nonzero, 2 x 32-bit index", f"{4 + 4 + 4}")
show("CSR break-even zeros (%)", f"{100 * (1 - 4 / 8):.0f}")
show("COO break-even zeros (%)", f"{100 * (1 - 4 / 12):.1f}")

# --- Frame: channel tiles and width 0.75 (chapter Sec. 10.2.1, 10.2.4) ------
TILE = 8
show("250 channels: last tile lanes busy", f"{250 % TILE} of {TILE}")
show("250 channels: tiles", f"{-(-250 // TILE)}")
show("248 channels: tiles, remainder", f"{248 // TILE}, {248 % TILE}")
show("256 -> 250 work removed (%)", f"{100 * 6 / 256:.1f}  (slide: about 2)")
show("width 0.75 params kept (%)", f"{100 * 2.6 / 3.5:.0f}")
show("width 0.75 MFLOP kept (%)", f"{100 * 418 / 600:.0f}")
show("width 0.75 square law (%)", f"{100 * 0.75 ** 2:.0f}")

# --- Frame: iterative pruning (Han et al. 2015, Fig. 5, AlexNet) ------------
# Points extracted from the vector paths of the published figure
# (arXiv:1506.02626v3, p. 7); x = share of parameters pruned (%), y = top-5
# accuracy change (percentage points). Inputs, not derived.
HAN_NO_RETRAIN = [(0, 0.0), (33.5, -0.03), (49.9, -0.12), (66.6, -0.93), (74.9, -2.16),
                  (80.0, -4.07)]
HAN_RETRAIN = [(0, 0.0), (33.5, -0.02), (50.0, 0.02), (66.6, 0.17), (75.0, 0.15),
               (80.0, 0.02), (83.3, -0.18), (85.7, -0.47), (87.5, -0.82), (89.9, -1.59),
               (93.0, -3.88)]
HAN_ITERATIVE = [(87.5, 0.05), (88.9, 0.02), (90.0, -0.08), (91.7, -0.35), (92.8, -0.62),
                 (93.8, -1.05), (94.7, -2.0), (95.8, -4.19)]


def fewer(p):
    return 1 / (1 - p / 100)


show("Han no retrain: loss at 2/3 pruned (points)", f"{HAN_NO_RETRAIN[3][1]:.1f}")
show("Han one prune + retrain: no loss up to (x fewer)", f"{fewer(80.0):.0f}")
show("Han iterative: no loss up to (x fewer)", f"{fewer(88.9):.0f}")
show("Han iterative at 90% (x fewer)", f"{fewer(90.0):.0f}")

# --- Frame: low rank (chapter Sec. 10.2.2) ----------------------------------
N, K = 4096, 128
dense_lr = N * N * 4
fact_lr = 2 * N * K * 4
show("low rank dense (MB)", f"{dense_lr / MB:.1f}")
show("low rank factored (MB)", f"{fact_lr / MB:.1f}")
show("low rank ratio (x)", f"{dense_lr / fact_lr:.0f}")

# --- Frame: distillation (chapter Sec. 10.2.3; Sanh et al. 2019) ------------
P_T1 = [85, 10, 5]                    # teacher probabilities at T = 1 (chapter figure)
T = 4
r = [p ** (1 / T) for p in P_T1]      # softmax(z / T) = p^(1/T), renormalized
SOFT4 = [round(100 * x / sum(r)) for x in r]
show("teacher at T = 4 (%)", ", ".join(map(str, SOFT4)))
show("DistilBERT params share (%)", f"{100 - 40}")
show("DistilBERT speed (x)", f"{1 + 0.60:.1f}")

# --- Frame: folding batch norm (chapter Sec. 10.4.1) -------------------------
C_, H_, W_ = 256, 28, 28
act = C_ * H_ * W_ * 4
wts = C_ * C_ * 3 * 3 * 4
bn_params = C_ * 2 * 4
conv = act + wts + act
bn = act + bn_params + act
relu = act + act
unfused = conv + bn + relu
fused = act + wts + act
HBM = 3350 * GB                       # reference H100 (chapter Sec. 10.4.1)
show("feature map (KB)", f"{act / KB:.1f}")
show("conv weights (MB)", f"{wts / MB:.1f}")
show("conv traffic (MB)", f"{conv / MB:.1f}")
show("BN traffic (MB)", f"{bn / MB:.1f}")
show("ReLU traffic (MB)", f"{relu / MB:.1f}")
show("unfused total (MB)", f"{unfused / MB:.1f}")
show("fused total (MB)", f"{fused / MB:.1f}")
show("fusion traffic cut (%)", f"{100 * (1 - fused / unfused):.1f}")
show("unfused time at 3,350 GB/s (us)", f"{unfused / HBM * 1e6:.1f}")
show("fused time at 3,350 GB/s (us)", f"{fused / HBM * 1e6:.1f}")
show("fusion traffic-time ratio (x)", f"{unfused / fused:.2f}")

# --- Frame: MLP share (chapter Sec. 10.6.1) ----------------------------------
show("MLP share of 12 d^2 (%)", f"{100 * 8 / 12:.0f}")
show("projections share of 12 d^2 (%)", f"{100 * 4 / 12:.0f}")

# --- Frame: blur waterfall (fig-compression-waterfall, illustrative) --------
W075_PARAMS = 2.6e6
show("width 0.75 FP32 weights (MB)", f"{W075_PARAMS * 4 / MB:.1f}")
show("width 0.75 frame if time ~ FLOPs (ms)", f"{125 * 418 / 600:.0f}")
show("width 0.75 INT8 weights (MB)", f"{W075_PARAMS * 1 / MB:.1f}")

# --- Frame: Amdahl (chapter Sec. 10.4, Systems Perspective 10.4) ------------
show("paper speedup 2 x 4 (x)", f"{2 * 4}")
show("realized 1.5x share of paper (%)", f"{100 * 1.5 / 8:.1f}")
show("Amdahl cap, model 20% (x)", f"{1 / (1 - 0.2):.2f}")
show("Amdahl p = 0.6, s = 4 (x)", f"{1 / ((1 - 0.6) + 0.6 / 4):.2f}")
show("Amdahl p = 0.6, s = 4 denominators", f"{1 - 0.6:.1f} + {0.6 / 4:.2f}")

# --- Frame: measured on this CPU (measure_cpu.py -> cpu_measure.json) -------
CPU = json.loads((HERE / "cpu_measure.json").read_text())
base = CPU["rows"][0]
for row in CPU["rows"]:
    show(f"cpu {row['name']}: median (ms)", f"{row['median_ms']:.1f}")
    show(f"cpu {row['name']}: GFLOP", f"{row['flops'] / 1e9:.2f}")
    show(f"cpu {row['name']}: time vs width 1.0 (%)",
         f"{100 * row['median_ms'] / base['median_ms']:.0f}")
    show(f"cpu {row['name']}: FLOPs vs width 1.0 (%)", f"{100 * row['flops'] / base['flops']:.0f}")
    show(f"cpu {row['name']}: zero share (%)", f"{100 * row['zeros_share']:.0f}")

# --- Frame: next lecture (Krishnamoorthi 2018, Table 3) ---------------------
show("ResNet-v1-50 float -> INT8 top-1 change (points)", f"{75.0 - 75.2:.1f}")
