#!/usr/bin/env python3
"""Every derived number in Lecture 11 (Hardware Acceleration, Part 1).

One block per slide, in deck order. Inputs carry their source; every printed
value is what the slide and its note show. "HA" is the rendered chapter,
Hardware Acceleration (Volume I). make_figs.py imports from here.

Run from this folder: python3 calcs.py
"""

import math

T, G, M, K = 1e12, 1e9, 1e6, 1e3
OUT = {}


def show(label, value, key=None):
    if key:
        OUT[key] = value
    print(f"{label}: {value}")


# --- Shared inputs ---------------------------------------------------------
H100_FP32 = 67 * T        # FLOP/s, CUDA cores (HA Napkin Math 11.10)
H100_FP16 = 989 * T       # FLOP/s, dense Tensor Core (HA Napkin Math 11.10)
H100_INT8 = 1979 * T      # op/s, dense (HA Sec. 11.4.5; Choquette 2023)
H100_BW = 3.35 * T        # bytes/s, HBM3 (HA Napkin Math 11.10)
A100_FP16 = 312 * T       # FLOP/s (HA Sec. 11.6 and algorithm-machine co-design)
A100_RIDGE = 153          # FLOP/byte, printed by HA for the A100 tensor path
A100_BW = 2.039 * T       # bytes/s (MLSysIM Hardware.Cloud.A100; HA batch-1 ceiling 2.04 TFLOP/s)

# --- Frame: hook (TPU v1 gains, Jouppi et al. 2017; HA Example 11.1) ---------
# Inputs only (15-30x, 30-80x, 700 MHz, 256x256 array); nothing derived.
show("TPU v1 array MACs per cycle (256 x 256)", f"{256 * 256:,}")

# --- Frame: recall (Model Compression table 13) ------------------------------
show("INT8 batch-one speedup 0.38 / 0.30 ms", f"{0.38 / 0.30:.2f}  (slide: 1.3)")

# --- Frame: H100 and the TPU v1 floorplan ------------------------------------
CPU_LO, CPU_HI = 1 * T, 2 * T  # reference CPU, FLOP/s (HA Definition 11.1)
show("H100 FP16 over 2 TFLOP/s CPU", f"{H100_FP16 / CPU_HI:.1f}  (slide/note: about 490)")
show("H100 FP16 over 1 TFLOP/s CPU", f"{H100_FP16 / CPU_LO:.0f}  (note: 990)")
# TPU v1 die shares, Jouppi et al. 2017, Figure 2 caption (inputs):
# buffers 37%, compute 30%, I/O 10%, control 2%. Block labels in the figure:
# Unified Buffer 29%, Matrix Multiply Unit 24%, Accumulators 6%,
# Activation Pipeline 6%.
show("TPU v1 buffers over compute area", f"{37 / 30:.2f}")
show("TPU v1 buffers over control area", f"{37 / 2:.1f}")

# --- Frame: energy ladder (Horowitz 2014; HA figure 4) -----------------------
E_DRAM32, E_FMUL32, E_SRAM8K, E_FADD32, E_IADD8 = 640, 3.7, 5, 0.9, 0.03  # pJ
show("DRAM read / FP32 multiply", f"{E_DRAM32 / E_FMUL32:.0f}", "dram_per_mul")
show("DRAM read / 8 KB SRAM read", f"{E_DRAM32 / E_SRAM8K:.0f}", "dram_per_sram")
# Lecture 0 and Lecture 6 recalls (same Horowitz table):
# Lecture 0 input: DRAM 160 pJ/byte (640 / 4); FP16 multiply 1.1 pJ (Horowitz 2014)
show("L0: 160 pJ/byte over FP16 multiply 1.1 pJ", f"{160 / 1.1:.0f}  (note: about 145)")
show("L6: 640 / 4.6 pJ FP32 multiply-add", f"{640 / 4.6:.0f}  (note: about 139)")
# 7 nm update: Jouppi et al., "Ten Lessons From Three Generations Shaped
# Google's TPUv4i," ISCA 2021, Table 2 (pJ; memory per 64-bit access, DRAM
# is DDR3/4 I/O only): FP32 multiply 3.7 -> 1.31; 8 KB SRAM 10 -> 7.5;
# DDR3/4 1300 -> 1300.
MUL45, MUL7, SR45, SR7, DR45, DR7 = 3.7, 1.31, 10, 7.5, 1300, 1300
show("7 nm: FP32 multiply 45/7 gain", f"{MUL45 / MUL7:.1f}", "mul_gain")
show("7 nm: 8 KB SRAM 45/7 gain", f"{SR45 / SR7:.1f}", "sram_gain")
show("7 nm: DRAM I/O 45/7 gain", f"{DR45 / DR7:.1f}", "dram_gain")
show("DRAM 64-bit I/O over FP32 multiply, 45 nm", f"{DR45 / MUL45:.0f}", "dram_mul45")
show("DRAM 64-bit I/O over FP32 multiply, 7 nm", f"{DR7 / MUL7:.0f}", "dram_mul7")

# --- Frame: matrix units (HA footnote 13; NVIDIA 2017) -----------------------
show("V100 / P100 FP16 peak", f"{125 / 21.2:.1f}  (slide: about 6)")
show("Tensor Core MACs per instruction / 32 threads", f"{2048 / 32:.0f} FMAs per thread")

# --- Frame: systolic array (HA Sec. 11.4.4) -----------------------------------
show("128 x 128 MACs per cycle", f"{128 * 128:,}")

# --- Frame: systolic energy (HA Napkin Math 11.1) ------------------------------
E_MAC = 1.0                   # pJ, "about 1 pJ" (HA Napkin Math 11.1)
naive = 4 * E_DRAM32 + E_MAC  # 3 loads + 1 write per MAC
acc_per_mac = 2 / 128
systolic = acc_per_mac * E_DRAM32 + E_MAC
show("naive energy per MAC (pJ)", f"{naive:,.0f}")
show("systolic accesses per MAC", f"{acc_per_mac:.4f}  (slide: 0.016)")
show("systolic energy per MAC (pJ)", f"{systolic:.1f}")
show("naive / systolic", f"{naive / systolic:.1f}")

# --- Frame: memory wall (HA figure 7) ----------------------------------------
# 3,900x and 33x are read from the book figure (inputs, not derived here).
show("GPT-2 XL FP16 weights (GB): 1.5e9 x 2 bytes", f"{1.5e9 * 2 / G:.0f}")

# --- Frame: tiers and the measured latency curve ------------------------------
CLOCK = 2e9                   # Hz (HA Napkin Math 11.2)
show("cycle at 2 GHz (ps)", f"{1 / CLOCK * 1e12:.0f}")
show("300 ns HBM access at 2 GHz (cycles)", f"{300e-9 * CLOCK:.0f}", "hbm_cycles")
# Chips and Cheese, "Nvidia's H100: Funny L2, and Tons of Bandwidth" (2023),
# H100 PCIe pointer-chasing data labels (inputs): L1 22.4 ns, L2 159.7 ns,
# far L2 294.0 ns, HBM 377.9 ns.
CNC = {"L1": 22.4, "L2": 159.7, "L2 far": 294.0, "HBM": 377.9}
show("measured HBM / L1 latency", f"{CNC['HBM'] / CNC['L1']:.0f}", "cnc_hbm_l1")
show("measured HBM / L2 latency", f"{CNC['HBM'] / CNC['L2']:.1f}")
show("measured HBM latency at 2 GHz (cycles)", f"{CNC['HBM'] * 1e-9 * CLOCK:.0f}")

# --- Frame: from HBM alone, 295x short (HA equation 3) -------------------------
need = H100_FP16 / 1.0        # bytes/s at 1 FLOP/byte
show("required bandwidth at 1 FLOP/byte (TB/s)", f"{need / T:.0f}")
show("shortfall vs 3.35 TB/s", f"{need / H100_BW:.1f}")

# --- Frame: bandwidth taper (HA Sec. 11.5.4 margin figure) --------------------
TAPER = {"HBM": 3350, "NVLink": 900, "PCIe": 64, "Network": 50}  # GB/s
show("HBM / PCIe", f"{TAPER['HBM'] / TAPER['PCIe']:.1f}", "hbm_pcie")
show("HBM / NVLink", f"{TAPER['HBM'] / TAPER['NVLink']:.1f}")

# --- Frame: three ridges (HA Napkin Math 11.6 and 11.10) -----------------------
r32, r16, r8 = H100_FP32 / H100_BW, H100_FP16 / H100_BW, H100_INT8 / H100_BW
show("FP32 ridge", f"{r32:.1f}", "ridge32")
show("FP16 ridge", f"{r16:.1f}", "ridge16")
show("INT8 ridge", f"{r8:.1f}", "ridge8")
show("FP16 / FP32 peak", f"{H100_FP16 / H100_FP32:.1f}")

# --- Frame: the ridge doubled; A100 -> H100 at 200 FLOP/byte (HA Napkin 11.3) --
I_K = 200
show("A100 ridge from 312 / 2.039", f"{A100_FP16 / A100_BW:.1f}  (HA prints 153)")
a100 = min(A100_FP16, A100_BW * I_K)
h100 = min(H100_FP16, H100_BW * I_K)
show("A100 attainable at 200 (TFLOP/s)", f"{a100 / T:.0f}  ({'compute' if a100 == A100_FP16 else 'bandwidth'} bound)")
show("H100 attainable at 200 (TFLOP/s)", f"{h100 / T:.0f}  ({'compute' if h100 == H100_FP16 else 'bandwidth'} bound)")
show("speedup A100 -> H100 at 200 FLOP/byte", f"{h100 / a100:.3f}  (slide: about 2.1)", "speedup200")
show("bandwidth ratio H100 / A100", f"{H100_BW / A100_BW:.2f}")
show("compute ratio H100 / A100", f"{H100_FP16 / A100_FP16:.2f}")
# Rising ridge, V100 2017 to B200 2024 (MLSysIM Hardware.Cloud, as HA figure 9)
V100_FP16, V100_BW, B200_FP16, B200_BW = 125 * T, 0.9 * T, 2250 * T, 8 * T
RIDGE_CHIPS = [("V100", 2017, V100_FP16, V100_BW), ("A100", 2020, A100_FP16, A100_BW),
               ("H100", 2022, H100_FP16, H100_BW), ("B200", 2024, B200_FP16, B200_BW)]
OUT["ridges"] = [(n, y, f / bw) for n, y, f, bw in RIDGE_CHIPS]
for n, y, f, bw in RIDGE_CHIPS:
    show(f"{n} ({y}) ridge", f"{f / bw:.0f}")
show("V100 ridge", f"{V100_FP16 / V100_BW:.0f}")
show("B200 ridge", f"{B200_FP16 / B200_BW:.0f}")
show("compute growth V100 -> B200", f"{B200_FP16 / V100_FP16:.0f}")
show("bandwidth growth V100 -> B200", f"{B200_BW / V100_BW:.1f}")

# --- Frame: five layers (HA Napkin Math 11.4, 11.5, 11.7, 11.8, 11.10) --------
B2 = 2  # bytes per FP16 value
d, b, s = 1600, 8, 1024
qkv_f = 2 * 3 * b * s * d * d
qkv_b = (b * s * d + 3 * d * d + b * s * d * 3) * B2
conv_out = 32 * 128 * 28 * 28
conv_f = conv_out * 128 * 3 * 3 * 2
conv_in_b, conv_w_b, conv_o_b = conv_out * B2, 128 * 128 * 9 * B2, conv_out * B2
conv_b = conv_in_b + conv_w_b + conv_o_b
fc_f = 2 * 32 * 2048 * 1000
fc_b = (32 * 2048 + 2048 * 1000 + 32 * 1000) * B2
ln_el = b * s * d
ln_f = ln_el * 8
ln_b = ln_el * B2 * 2 + d * 2 * B2
dec_f, dec_b = 2 * 1.5e9, 1.5e9 * B2
layers = {"QKV": (qkv_f, qkv_b), "conv3": (conv_f, conv_b), "FC": (fc_f, fc_b),
          "LayerNorm": (ln_f, ln_b), "decode": (dec_f, dec_b)}
INTENSITY = {}
for name, (f, by) in layers.items():
    i = f / by
    INTENSITY[name] = i
    att = min(H100_FP16, H100_BW * i)
    show(f"{name}: FLOPs {f / G:.3f} G, bytes {by / M:.1f} MB, I", f"{i:.1f}  "
         f"ceiling {att / T:.1f} TFLOP/s = {100 * att / H100_FP16:.1f}% of peak")
OUT["intensity"] = INTENSITY
show("softmax intensity 629.1 MFLOP / 838.9 MB",
     f"{8 * 25 * 1024 * 1024 * 3 / (8 * 25 * 1024 * 1024 * 2 * 2):.2f}")

# --- Frame: batch and intensity (HA equation 5, Napkin Math 11.9) --------------
WIDTH = 2048


def batch_intensity(batch, n=WIDTH):
    return 2 * batch * n * n / (2 * batch * n + 2 * n * n + 2 * batch * n)


for bb in (1, 32, 512):
    show(f"I at batch {bb}", f"{batch_intensity(bb):.1f}")
crossing = r16 * WIDTH / (WIDTH - 2 * r16)
show("batch where I reaches the FP16 ridge", f"{crossing:.0f}", "crossing")

# --- Frame: INT8 helps twice (HA Sec. 11.4.5, Lighthouse 11.1) -----------------
R50_FLOP = 8.2 * G
show("INT8 / FP16 peak", f"{H100_INT8 / H100_FP16:.2f}")
show("ResNet-50 arithmetic at FP16 peak (us)", f"{R50_FLOP / H100_FP16 * 1e6:.1f}", "floor16")
show("ResNet-50 arithmetic at INT8 peak (us)", f"{R50_FLOP / H100_INT8 * 1e6:.1f}", "floor8")

# --- Frame: INT8 lifts the ceiling (HA Napkin Math 11.6) ----------------------
show("stage-3 conv FP16 I", f"{conv_f / conv_b:.0f}")
show("stage-3 conv INT8 I", f"{conv_f / (conv_b / 2):.0f}")
show("stage-3 conv INT8 weights, FP16 activations",
     f"{conv_f / (conv_in_b + conv_w_b / 2 + conv_o_b):.0f}")
s2_pos = 32 * 56 * 56
s2_f = s2_pos * 64 * 64 * 9 * 2
s2_b = s2_pos * 64 * B2 + 64 * 64 * 9 * B2 + s2_pos * 64 * B2  # in + weights + out
show("stage-2 3x3 FP16 / INT8 I", f"{s2_f / s2_b:.0f} / {s2_f / (s2_b / 2):.0f}")
e_f = s2_pos * 64 * 256 * 2
e_b = (s2_pos * 64 + 64 * 256 + s2_pos * 256) * B2
show("stage-2 1x1 expansion FP16 / INT8 I", f"{e_f / e_b:.0f} / {e_f / (e_b / 2):.0f}")

# --- Frame: Amdahl (HA equation 1, Lighthouse 11.1) ---------------------------


def amdahl(p, gain):
    return 1 / ((1 - p) + p / gain)


G_MAT = H100_INT8 / (8 * T)   # 1,979 TOPS over an assumed ~8 TOPS CPU
show("matrix-path gain 1979 / 8", f"{G_MAT:.0f}")
show("ResNet-50 p=0.95", f"{amdahl(0.95, G_MAT):.1f}")
show("GPT-2 decode p=0.80, G=247", f"{amdahl(0.80, G_MAT):.1f}")
show("GPT-2 decode p=0.80, G=67", f"{amdahl(0.80, 67):.1f}")
show("ceiling p=0.80", f"{1 / 0.2:.0f}")

# --- Frame: lighthouses (HA table 17) -----------------------------------------
for name, fp16_bytes in (("ResNet-50", 51.2 * M), ("MobileNetV2", 7.0 * M),
                         ("GPT-2 XL", 3 * G), ("Keyword spotting", 400 * K)):
    show(f"{name} INT8 bytes (half of FP16)", f"{fp16_bytes / 2 / M:.1f} MB")

# --- Frame: true or false (HA Sec. 11.12) -------------------------------------
T4_FP16, T4_BW = 65 * T, 0.32 * T  # HA: T4 65 TFLOP/s; batch-1 ceiling 0.32 TFLOP/s
show("A100 / T4 peak", f"{A100_FP16 / T4_FP16:.1f}")
show("A100 / T4 bandwidth (batch-one ceilings)", f"{2.04 * T / T4_BW:.1f}  (HA's 2.04 and 0.32 TFLOP/s ceilings)")
# NVIDIA H100 SXM product page spec table (retrieved 2026-10-19): FP16 Tensor
# Core 1,979 teraFLOPS and INT8 3,958 TOPS, both marked "* With sparsity".
show("datasheet FP16 with sparsity / 2 (dense)", f"{1979 / 2:.1f}  (HA: 989)")
show("datasheet INT8 with sparsity / 2 (dense)", f"{3958 / 2:.0f}")

# --- Frame: close (TPU v1 reuse) ----------------------------------------------
show("TPU v1 reuse across a 256-wide row", "256")
show("ResNet-50 INT8 arithmetic share of a 0.30 ms pass",
     f"{R50_FLOP / H100_INT8 / 0.30e-3 * 100:.1f}%")
