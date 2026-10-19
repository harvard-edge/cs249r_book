#!/usr/bin/env python3
"""Every derived number in Lecture 11b (Hardware Acceleration, Part 2).

One block per slide, in deck order. Inputs carry their source; every printed
value is what the slide or its note shows. "HA" is the rendered chapter,
Hardware Acceleration (Volume I). make_figs.py imports from here.

Run from this folder: python3 calcs.py
"""

import math

T, G = 1e12, 1e9
OUT = {}


def show(label, value, key=None):
    if key:
        OUT[key] = value
    print(f"{label}: {value}")


def head(title):
    print(f"\n--- {title}")


# --- Hook: Boehm's ladder ------------------------------------------------------
# Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a
# Worklog," 2022, results table (GFLOP/s, FP32 SGEMM, the author's A6000).
BOEHM = [("1 Naive", 309.0), ("2 Coalesced loads", 1986.5),
         ("3 Shared-memory tiles", 2980.3), ("4 1D register tiles", 8474.7),
         ("5 2D register tiles", 15971.7), ("6 Vector loads", 18237.3),
         ("9 Autotuned", 19721.0), ("10 Warp tiles", 21779.3),
         ("cuBLAS", 23249.6)]
head("Hook and ladder (Boehm 2022)")
show("cuBLAS / naive", f"{BOEHM[-1][1] / BOEHM[0][1]:.1f}", "boehm_total")
show("cuBLAS GFLOP/s, rounded to tens", f"{round(BOEHM[-1][1], -1):,.0f}")
for (a, ga), (b, gb) in zip(BOEHM, BOEHM[1:]):
    show(f"  step {a} -> {b}", f"{gb / ga:.2f}")

# --- Recall: H100 ridges (HA Napkin Math 11.5 and 11.10) -----------------------
H100_FP16 = 989 * T     # FLOP/s dense (HA Napkin Math 11.10)
H100_INT8 = 1979 * T    # op/s dense (HA Sec. 11.4.5)
H100_BW = 3.35 * T      # bytes/s (HA Napkin Math 11.10)
head("Recall")
show("H100 FP16 ridge (FLOP/byte)", f"{H100_FP16 / H100_BW:.1f}", "ridge16")
show("H100 INT8 ridge (op/byte)", f"{H100_INT8 / H100_BW:.1f}", "ridge8")
# Stage-3 conv intensities 563.1 (FP16), 1,126 (INT8), 569 (INT8 weights, FP16
# activations) are HA Napkin Math 11.6 and 11.10 inputs, not derived here.


# --- Systolic timing (HA Sec. 11.7.5: one hop per cycle) -----------------------
def cycles(b, n):
    return b + 2 * n - 2


def busy_cells(t, b, n):
    return sum(1 for r in range(n) for c in range(n) if 0 <= t - r - c < b)


head("2x2 trace and 4x4 fill and drain")
show("2x2 array, 2 input rows: cycles", cycles(2, 2))
show("4x4 array, 8 input rows: cycles", cycles(8, 4), "c4")
show("4x4 average busy share (%)", f"{100 * 8 / cycles(8, 4):.0f}", "busy4")
for t in (2, 6, 7, 11):
    show(f"  cycle {t}: busy cells of 16", busy_cells(t, 8, 4))

head("128x128 busy fraction (TPU v4 MXU size, Jouppi et al. 2023)")
N_ARRAY = 128
for b in (1, 128, 1024, 4096):
    show(f"  B = {b:,}: busy %", f"{100 * b / cycles(b, N_ARRAY):.2f}")

# --- TPUv4i floorplan (Jouppi et al., ISCA 2021, Figure 6, Tables 1 and 7) -----
# Inputs: 4 MXUs of 128x128 per core; MXUs 11% of the die; CMEM 28% of the
# area; average FLOPS utilization of eight production apps 33% (TPUv4i),
# 20% (TPUv1), 51% (TPUv2), 38% (TPUv3).
head("TPUv4i")
show("MACs per cycle, 4 MXUs of 128x128", f"{4 * 128 * 128:,}")

# --- Loop orders and traffic (HA listing 14: W 512x256, X 256x32) --------------
M_, K_, N_ = 512, 256, 32
MACS = M_ * K_ * N_
W_EL, X_EL, Z_EL = M_ * K_, K_ * N_, M_ * N_
head("Loop orders (element moves)")
show("MACs", f"{MACS:,}")
show("weights / inputs / outputs", f"{W_EL:,} / {X_EL:,} / {Z_EL:,}")
show("weight-stationary partial sums (read + write per MAC)", f"{2 * MACS:,}")
ws = W_EL + MACS + 2 * MACS
os_ = MACS + MACS + Z_EL
is_ = MACS + X_EL + 2 * MACS
show("WS total", f"{ws:,}")
show("OS total", f"{os_:,}", "untiled")
show("IS total", f"{is_:,}")
show("WS, row of 128: input reads", f"{MACS // 128:,}")
show("reuse per held weight (batch columns)", N_)
show("reuse per held input (output rows)", M_)
# Example 11.4 (modeled): 16 filters of 3x3 on an 8x8 map, 36 output positions.
head("Example 11.4")
show("filter weights held", 16 * 3 * 3)
show("spilled reads 36 x 144", f"{36 * 144:,}")
show("ratio", 36 * 144 // 144)

# --- Four weight tiles (H100 SM, 228 KB shared memory, HA Lighthouse 11.2) -----
KB = 1024
head("Four weight tiles")
show("all W at FP16 (KB)", W_EL * 2 // KB)
show("one 128-row W tile (KB)", 128 * K_ * 2 // KB)
show("X at FP16 (KB)", X_EL * 2 // KB)
show("128x32 FP32 output tile (KB)", 128 * N_ * 4 // KB)
show("live total (KB)", (128 * K_ * 2 + X_EL * 2 + 128 * N_ * 4) // KB)
tiled = W_EL + 4 * X_EL + Z_EL
show("tiled moves", f"{tiled:,}", "tiled")
show("input reads, four tiles", f"{4 * X_EL:,}")
show("untiled / tiled", f"{os_ / tiled:.1f}", "tile_gain")

# --- Boehm kernels 1 and 3 (github.com/siboehm/SGEMM_CUDA @ 5a7dcc5) ----------
# Kernel 1 inner loop: 2 global loads (A, B) per multiply-add.
# Kernel 3: per BLOCKSIZE = 32 step, each thread loads 1 A and 1 B element
# into shared memory, then does 32 multiply-adds from shared memory.
BLOCK = 32
head("Boehm kernels 1 and 3")
show("kernel 1: global loads per multiply-add", 2)
show("kernel 3: global loads per multiply-add", f"{2 / BLOCK:.4f}")
show("kernel 3 fewer global loads per multiply-add", f"{2 / (2 / BLOCK):.0f}")
show("kernel 3 / kernel 1 GFLOP/s", f"{BOEHM[2][1] / BOEHM[0][1]:.1f}")
show("shared memory per block, As + Bs (KB, FP32)", 2 * BLOCK * BLOCK * 4 // KB)
show("kernel 5 / kernel 3 GFLOP/s (register tiles)", f"{BOEHM[4][1] / BOEHM[2][1]:.1f}")
# BLOCKSIZE = 32: src/runner.cu, run_sgemm_shared_mem_block (sgemm_shared_mem_block<32>).

# --- Tile intensity (HA Sec. 11.7.5; FP16, 4,096-square multiply) -------------
head("Tile intensity")


def intensity(tm, tn):
    return tm * tn / (tm + tn)


show("no reuse: 2 FLOP / 4 bytes", 2 / 4)
show("output tiles", f"{(4096 // 128) ** 2:,}")
show("tile products per output tile", 4096 // 128)
show("bytes per tile product, two 128x128 FP16 tiles", f"{2 * 128 * 128 * 2:,}")
show("FLOP per tile product 2 x 128^3", f"{2 * 128 ** 3:,}")
show("128x128 intensity", f"{intensity(128, 128):.0f}", "i128")
show("256x128 intensity", f"{intensity(256, 128):.1f}", "i256")
show("load each matrix once", f"{2 * 4096 ** 3 / (3 * 4096 ** 2 * 2):.1f}", "ionce")

# --- Equation 6 (HA Sec. 11.7.5): 2 TM TK + 2 TK TN + 4 TM TN bytes ------------
head("Equation 6 capacity (KB of 1,024 bytes)")


def eq6(tm, tn, tk, double=False):
    a, b = 2 * tm * tk, 2 * tk * tn
    if double:
        a, b = 2 * a, 2 * b
    return a // KB, b // KB, 4 * tm * tn // KB


for lab, args in (("128^3 single", (128, 128, 128)), ("128^3 double", (128, 128, 128, True)),
                  ("TK = 256 double", (128, 128, 256, True))):
    a, b, c = eq6(*args)
    show(f"  {lab}: A, B, acc, total", f"{a}, {b}, {c}, {a + b + c}")
show("128^3 single, bytes", f"{2 * 128 * 128 + 2 * 128 * 128 + 4 * 128 * 128:,}")

# --- Width-129 cliff (HA Sec. 11.7.5) ------------------------------------------
head("Cliff")
show("tile products, 1,024 tiles x 32", f"{1024 * 32:,}")
show("129 of 256 (%)", f"{100 * 129 / 256:.1f}")
show("(129/256)^2 (%)", f"{100 * (129 / 256) ** 2:.1f}")

# --- Measured tile quantization (NVIDIA guide, figure 7, A100) ----------------
# Inputs read from the published plot: 189.7 TFLOP/s at N = 128, 148.2 at 136.
head("Measured tile quantization")
show("work growth 136 / 128", f"{136 / 128:.4f}")
show("useful share of the extra tile, 8 / 128 (%)", f"{100 * 8 / 128:.2f}")
show("time growth 1.0625 x 189.7 / 148.2", f"{136 / 128 * 189.7 / 148.2:.2f}", "time36")

# --- Wave quantization (NVIDIA guide Sec. 3.2: 108 SMs, 256x128 tiles) ---------
head("Wave quantization")
show("tiles at N = 1,536", f"{(2304 // 256) * (1536 // 128)}")
show("tiles at N = 1,544", f"{(2304 // 256) * math.ceil(1544 / 128)}")
show("tail share of SMs (%)", f"{100 * 9 / 108:.2f}")
show("extra work 1,544 / 1,536 - 1 (%)", f"{100 * (1544 / 1536 - 1):.2f}")

# --- Mapping space (HA Sec. 11.7.2 margin figure) ------------------------------
head("Mapping space")
show("4^5", 4 ** 5)
show("7!", math.factorial(7))
show("6!/3!", math.factorial(6) // math.factorial(3))
space = 4 ** 5 * math.factorial(7) * (math.factorial(6) // math.factorial(3))
show("product", f"{space:,}", "space")
show("product, millions", f"{space / 1e6:.0f}")

# --- Compiler pipeline (HA Sec. 11.8, 11.4.5) ----------------------------------
head("Compiler")
show("ResNet-50 8.2 GFLOP at 989 TFLOP/s (us)", f"{8.2 * G / H100_FP16 * 1e6:.1f}")

# --- Inductor output (fused_kernel.py, torch 2.13.0, Apple M5 Max CPU) ---------
# One Conv-BN-ReLU on a 1 x 64 x 56 x 56 FP32 map (ResNet-50 stage 2).
EL = 64 * 56 * 56
head("Inductor fused BN + ReLU")
show("elements per map", f"{EL:,}")
show("FP32 map bytes", f"{EL * 4:,}")
show("unfused BN then ReLU: map passes", 4)
show("fused: map passes", 2)
show("bytes saved by fusion (MB)", f"{2 * EL * 4 / 1e6:.2f}")

# --- Registers (CUDA Programming Guide, cc 9.0, Tables 30 and 31) --------------
head("Registers")


def resident_threads(regs):
    return min(2048, (65536 // regs) // 32 * 32)


for r in (32, 64, 128, 255):
    th = resident_threads(r)
    show(f"  {r} registers: threads, share %", f"{th:,}, {100 * th / 2048:g}")
show("65,536 / 255", 65536 // 255)

# --- Diagnose (illustrative width 1,000; NVIDIA guide Table 1: multiples of 64)
head("Diagnose")
show("1,000 = 64 x q + r", f"{1000 // 64} x 64 + {1000 % 64}")
show("1,024 / 64", 1024 // 64)

# --- PyTorch TF32 note (docs/source/notes/cuda.rst, v2.9.0; GA100) -------------
# Inputs: 10,240-square matmul, 0.016 s with TF32, 0.11 s without.
head("TF32 on A100 (PyTorch docs)")
FLOP = 2 * 10240 ** 3
show("time ratio 0.11 / 0.016", f"{0.11 / 0.016:.1f}", "tf32_ratio")
show("TF32 rate (TFLOP/s)", f"{FLOP / 0.016 / T:.0f}")
show("FP32 rate (TFLOP/s)", f"{FLOP / 0.11 / T:.1f}")
# NVIDIA A100 datasheet: FP32 19.5 TFLOP/s, TF32 156 TFLOP/s dense (inputs).

# --- One activation (HA Lighthouse 11.2; 160 pJ per byte, 45 nm estimate) -------
MAP = 56 * 56 * 64          # bytes per image at INT8
PJ = 160                    # pJ per byte (HA Lighthouse 11.2; Horowitz 640 pJ / 4 B)
head("Activation trip")
show("Horowitz 640 pJ per 32-bit read / 4 bytes", 640 / 4)
show("INT8 map (KB, decimal)", f"{MAP / 1e3:.1f}")
show("FP16 map (KB, decimal)", f"{2 * MAP / 1e3:.1f}")
show("INT8 map (KiB)", f"{MAP / KB:.0f}")
show("FP16 map (KiB)", f"{2 * MAP / KB:.0f}")
show("read once at INT8 (uJ)", f"{MAP * PJ / 1e6:.1f}")
show("batch 32 at INT8 (MB)", f"{32 * MAP / 1e6:.1f}")

head("Layer boundary (write + read)")
show("INT8 boundary at 160 pJ/B (uJ)", f"{2 * MAP * PJ / 1e6:.1f}", "b_int8")
show("FP16 boundary at 160 pJ/B (uJ)", f"{2 * 2 * MAP * PJ / 1e6:.1f}", "b_fp16")
# 7 nm basis: Jouppi et al., ISCA 2021, Table 2: HBM2 250-450 pJ per 64-bit
# access, I/O only; DDR3/4 1,300 pJ at both 45 and 7 nm.
HBM_LO, HBM_HI = 250 / 8, 450 / 8
show("DDR3/4 1,300 pJ per 64 bits, per byte", f"{1300 / 8:.1f}")
show("HBM2 pJ per byte, 7 nm I/O", f"{HBM_LO:.1f} to {HBM_HI:.1f}")
show("INT8 boundary at HBM2 (uJ)", f"{2 * MAP * HBM_LO / 1e6:.1f} to {2 * MAP * HBM_HI / 1e6:.1f}",
     "b_hbm")
show("160 pJ/B over HBM2 range", f"{PJ / HBM_HI:.1f} to {PJ / HBM_LO:.1f}")

if __name__ == "__main__":
    pass
