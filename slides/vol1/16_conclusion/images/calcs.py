#!/usr/bin/env python3
"""Every derived number in Lecture 16 (Conclusion), computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports from this file so figures and slides share
one computation. "Chapter" means the rendered Conclusion chapter of Volume I.

Run from this folder: python3 calcs.py
"""

import math


def r(x, nd=1):
    """Round half away from zero (Python's round() is half to even)."""
    q = 10 ** nd
    return math.floor(abs(x) * q + 0.5 + 1e-9) / q * (1 if x >= 0 else -1)


def f(x, nd=1, commas=True):
    v = r(x, nd)
    return f"{v:,.{nd}f}" if commas else f"{v:.{nd}f}"


def head(s):
    print(f"\n[{s}]")


def show(label, value):
    print(f"  {label}: {value}")


# ---------------------------------------------------------------------------
# Shared hardware inputs (chapter, NVIDIA H100 reference machine; as in L11)
# ---------------------------------------------------------------------------
H100_BW = 3.35e12          # B/s, HBM3
H100_FP16 = 989e12         # FLOP/s, dense FP16
H100_INT8 = 1979e12        # OP/s, dense INT8
H100_GIB = 80              # GiB of HBM (Lecture 8 uses 80 GiB)
RIDGE16 = H100_FP16 / H100_BW
RIDGE8 = H100_INT8 / H100_BW

# Slide 2: the stack, one number per layer -----------------------------------
# L13 (Example 13.1): batch 1 $3.92 and batch 8 $0.52 per million requests.
B1_COST, B8_COST = 3.92, 0.52
BATCH_COST_RATIO = B1_COST / B8_COST
# L10 (Model Compression Sec. 10.1): 7B parameters at 2 bytes (FP16).
P7B, FP16_BYTES = 7e9, 2
MODEL_7B_GB = P7B * FP16_BYTES / 1e9
# L8 (derived there): PaLM 540B parameters (Chowdhery et al. 2022) at the
# chapter's 16 bytes per parameter of training state.
P_PALM, TRAIN_BYTES = 540e9, 16
PALM_STATE_TB = P_PALM * TRAIN_BYTES / 1e12
PALM_H100S = P_PALM * TRAIN_BYTES / (H100_GIB * 2**30)
# L9 (Data Selection Sec. 9.1): GPT-2 XL 1.5B parameters, 1.5e21 FLOP run,
# 6 FLOP per parameter per token, WebText about 10 billion unique tokens.
GPT2_RUN_FLOP, GPT2_P, WEBTEXT_TOKENS = 1.5e21, 1.5e9, 10e9
WEBTEXT_PASSES = GPT2_RUN_FLOP / (6 * GPT2_P) / WEBTEXT_TOKENS
# L8 (Chowdhery et al. 2022): PaLM trained on 6,144 TPU v4 chips (input).
PALM_CHIPS = 6144

# Slide 3 and journey: depthwise separable 3x3 conv, 256 output channels
# (chapter table 2). Cost ratio standard / separable = 1 / (1/N + 1/K^2).
K, N_OUT = 3, 256
DW_RATIO = 1 / (1 / N_OUT + 1 / K**2)
INT8_OF_FP32 = 8 / 32

# Slide 9: roofline ridges on one H100 ----------------------------------------
# Intensities 1 (decode), 1,046.7 (prefill), 569 (ResNet-50 conv, INT8 weights)
# are read from Hardware Acceleration Napkin Math 11.4 to 11.10 (inputs).
PREFILL_I, RESNET_INT8_I = 1046.7, 569

# Slide 10: Llama 2 70B at batch 1 on two H100s (chapter Sec. 16.4) ----------
P70 = 70e9
L70_BYTES = P70 * FP16_BYTES
L70_FLOP = 2 * P70
TWO_BW, TWO_PEAK = 2 * H100_BW, 2 * H100_FP16
T_MEM_MS = L70_BYTES / TWO_BW * 1e3
T_COMP_MS = L70_FLOP / TWO_PEAK * 1e3
MEM_OVER_COMP = T_MEM_MS / T_COMP_MS

# Slide 13 (Part I): a stop command to a cloud 1,500 km away (Lecture 2).
# Light in fiber about 200,000 km/s (two thirds of c).
DIST_KM, FIBER_KMS = 1500, 200_000
RTT_MS = 2 * DIST_KM / FIBER_KMS * 1e3

# Slide 15 (Part III): energy of one 32-bit DRAM read against one FP32
# multiply, 45 nm (Horowitz 2014, via Lecture 11): 640 pJ and 3.7 pJ.
E_DRAM32, E_FMUL32 = 640, 3.7
DRAM_OVER_MUL = E_DRAM32 / E_FMUL32

# Slide 16: Amdahl, 90 percent host preprocessing, inference 10x faster
# (chapter Sec. 16.7).
SERIAL, SPEEDUP = 0.90, 10
NEW_TIME = SERIAL + (1 - SERIAL) / SPEEDUP
AMDAHL = 1 / NEW_TIME
AMDAHL_AVG = (1 + SPEEDUP) / 2   # the wrong answer B: averaging the speedups

# Slide 17 (Part IV): Thai clinics, 393 of 1,838 images ungradable (Lecture 3,
# Beede et al. 2020). Fairness gap 3.0 points grows 1.10 a quarter against a
# 5-point gate (Lecture 15, Napkin Math 15.3).
UNGRADABLE, IMAGES = 393, 1838
GAP0, ALPHA, GATE = 3.0, 1.10, 5.0
K_CROSS = math.log(GATE / GAP0) / math.log(ALPHA)

# Slide 22: tails. Illustrative TTFT 50 ms mean, 2,000 ms p99 (chapter Sec.
# 16.4). Dean and Barroso 2013, Table 1: one leaf p99 10 ms, all leaves p99
# 140 ms. Lecture 13: 100 servers each meeting a target 99 percent of the time.
TTFT_MEAN, TTFT_P99 = 50, 2000
LEAF_P99, ALL_P99 = 10, 140
FANOUT, P_OK = 100, 0.99

# Slide 23: MTBF of a 1,024-GPU job (chapter Sec. 16.5) ----------------------
MTTF_YEARS, HOURS_PER_YEAR = 5.7, 8766       # 365.25 days
MTTF_H_EXACT = MTTF_YEARS * HOURS_PER_YEAR
MTTF_H = 50_000                               # the chapter's rounded value
N_GPU = 1024
MTBF_H = MTTF_H / N_GPU
RUN_H = 21 * 24                               # a three-week run
FAILS_IN_RUN = RUN_H / MTBF_H

# Slide 25: power ladder (System Paradigms Sec. 2.1, as in Lecture 0) ---------
P_CLOUD, P_TINY = 3e6, 1e-3                   # W
POWER_DECADES = math.log10(P_CLOUD / P_TINY)

# Slide 26: composed assistant (chapter self-check 16.5, question 1) ---------
RET, TOOL_A, TOOL_B, GEN, VER = 50, 110, 70, 180, 30
CRIT = RET + max(TOOL_A, TOOL_B) + GEN + VER
ALL_SUM = RET + TOOL_A + TOOL_B + GEN + VER

# Slides 27 and 28: Llama 3 405B (Llama Team 2024, arXiv 2407.21783v3,
# Sec. 3.3.4 and Table 5): 466 interruptions in a 54-day snapshot, 47 planned;
# Table 5 counts by component; up to 16,384 H100s.
INTERRUPTS, PLANNED, DAYS, LLAMA_GPUS = 466, 47, 54, 16384
TABLE5 = [("Faulty GPU", "GPU", 148, 30.1), ("GPU HBM3 Memory", "GPU", 72, 17.2),
          ("Software Bug", "Dependency", 54, 12.9),
          ("Network Switch/Cable", "Network", 35, 8.4),
          ("Host Maintenance", "Unplanned Maintenance", 32, 7.6),
          ("GPU SRAM Memory", "GPU", 19, 4.5), ("GPU System Processor", "GPU", 17, 4.1),
          ("NIC", "Host", 7, 1.7), ("NCCL Watchdog Timeouts", "Unknown", 7, 1.7),
          ("Silent Data Corruption", "GPU", 6, 1.4),
          ("GPU Thermal Interface + Sensor", "GPU", 6, 1.4), ("SSD", "Host", 3, 0.7),
          ("Power Supply", "Host", 3, 0.7), ("Server Chassis", "Host", 2, 0.5),
          ("IO Expansion Board", "Host", 2, 0.5), ("Dependency", "Dependency", 2, 0.5),
          ("CPU", "Host", 2, 0.5), ("System Memory", "Host", 2, 0.5)]
UNEXPECTED = INTERRUPTS - PLANNED
TOP_ROWS = 5                                  # rows the slide quotes
TOP_SUM = sum(c for _, _, c, _ in TABLE5[:TOP_ROWS])
REST_SUM = sum(c for _, _, c, _ in TABLE5[TOP_ROWS:])
REST_ROWS = len(TABLE5) - TOP_ROWS
LLAMA_GAP_H = DAYS * 24 / UNEXPECTED
MODEL_16K_H = MTTF_H / LLAMA_GPUS

# Slide 30 (voice request, chapter Sec. 16.8): GPT-2 XL 1.5B parameters in
# FP16 (3 GB) read per token on one H100.
P_GPT2XL = 1.5e9
GPT2_BYTES = P_GPT2XL * FP16_BYTES
GPT2_MEM_MS = GPT2_BYTES / H100_BW * 1e3
GPT2_COMP_MS = 2 * P_GPT2XL / H100_FP16 * 1e3


if __name__ == "__main__":
    head("Slide 2: the stack")
    show("batch 1 against batch 8 cost", f"{BATCH_COST_RATIO:.1f}x")
    show("7B model in FP16", f"{MODEL_7B_GB:.0f} GB")
    show("PaLM training state", f"{PALM_STATE_TB:.2f} TB (slide: 8.6 TB)")
    show("H100s of memory (80 GiB)", f"{PALM_H100S:.0f} (slide: about 100)")
    show("GPT-2 passes over each WebText token", f"{WEBTEXT_PASSES:.1f} (slide: about 17)")

    head("Slides 3 and 7: MobileNetV2")
    show("depthwise separable FLOP ratio, 3x3, 256 out", f"{DW_RATIO:.1f}x")
    show("INT8 bytes as a fraction of FP32", f"{INT8_OF_FP32:.2f}")

    head("Slide 9: roofline")
    show("FP16 ridge", f"{RIDGE16:.1f} FLOP/byte (slide: 295)")
    show("INT8 ridge", f"{RIDGE8:.1f} OP/byte (slide: 591)")
    show("ResNet-50 INT8-weight conv below the INT8 ridge", f"{RESNET_INT8_I} < {RIDGE8:.0f}")
    show("prefill above the FP16 ridge", f"{PREFILL_I:,} > {RIDGE16:.0f}")

    head("Slide 10: Llama 2 70B, batch 1, two H100s")
    show("weights", f"{L70_BYTES / 1e9:.0f} GB")
    show("work per token", f"{L70_FLOP / 1e9:.0f} GFLOP")
    show("two H100s bandwidth", f"{TWO_BW / 1e12:.2f} TB/s")
    show("two H100s peak", f"{TWO_PEAK / 1e12:,.0f} TFLOP/s")
    show("memory floor", f"{T_MEM_MS:.1f} ms")
    show("compute floor", f"{T_COMP_MS:.2f} ms")
    show("ratio", f"{MEM_OVER_COMP:.1f}x")

    head("Slide 13: Part I")
    show("round trip to 1,500 km in fiber", f"{RTT_MS:.0f} ms")

    head("Slide 14: Part II")
    show("PaLM state", f"{PALM_STATE_TB:.1f} TB")
    show("H100s of memory", f"{PALM_H100S:.0f}")

    head("Slide 15: Part III")
    show("DRAM read in FP32 multiplies", f"{DRAM_OVER_MUL:.0f}x")
    show("decode intensity against the ridge", f"1 against {RIDGE16:.0f}")

    head("Slide 16: Amdahl")
    show("new request time", f"{NEW_TIME * 100:.0f}% of the old")
    show("speedup", f"{AMDAHL:.2f}x (slide: 1.1x)")
    show("wrong answer B, averaging", f"{AMDAHL_AVG:.1f}x")

    head("Slide 17: Part IV")
    show("ungradable images", f"{UNGRADABLE / IMAGES * 100:.1f}% (slide: 21%)")
    show("quarters for 3.0 to pass 5.0 at 1.10", f"{K_CROSS:.2f} (first whole quarter {math.ceil(K_CROSS)})")
    show("gap in quarter 6", f"{GAP0 * ALPHA ** 6:.2f} points")

    head("Slide 22: tails")
    show("illustrative p99 over mean", f"{TTFT_P99 / TTFT_MEAN:.0f}x")
    show("measured all-leaves p99 over one-leaf p99", f"{ALL_P99 / LEAF_P99:.0f}x")
    show("fan-out of 100 all meeting target", f"{P_OK ** FANOUT * 100:.1f}%")

    head("Slide 23: MTBF")
    show("5.7 years in hours", f"{MTTF_H_EXACT:,.0f} h (chapter: about 50,000)")
    show("MTBF of 1,024 GPUs", f"{MTBF_H:.1f} h")
    show("in days", f"{MTBF_H / 24:.1f} days")
    show("three-week run", f"{RUN_H} h")
    show("failures expected in the run", f"{FAILS_IN_RUN:.1f}")

    head("Slide 25: power ladder")
    show("3 MW over 1 mW, decades", f"{POWER_DECADES:.1f} (note: nine orders of magnitude)")

    head("Slide 26: composed assistant")
    show("critical path", f"{CRIT} ms")
    show("sum of every box", f"{ALL_SUM} ms")

    head("Slides 27 and 28: Llama 3 405B")
    show("unexpected interruptions", f"{UNEXPECTED}")
    show("hours between unexpected interruptions", f"{LLAMA_GAP_H:.1f} h")
    show("model at 16,384 GPUs", f"{MODEL_16K_H:.2f} h")
    tot = sum(c for _, _, c, _ in TABLE5)
    gpu = sum(c for _, cat, c, _ in TABLE5 if cat == "GPU")
    show("Table 5 counts sum", f"{tot} (paper: 419)")
    show("top five rows", f"{TOP_SUM}")
    show(f"other {REST_ROWS} rows", f"{REST_SUM}")
    show("top two rows (faulty GPU + HBM3)", f"{148 + 72}")
    show("top two rows, share of 419", f"{(148 + 72) / UNEXPECTED * 100:.1f}%")
    show("faulty GPU share of 419 (count)", f"{148 / UNEXPECTED * 100:.1f}% (Table 5 prints 30.1%)")
    show("GPU-category share of 419 (counts)", f"{gpu / UNEXPECTED * 100:.1f}% (text: 58.7%)")

    head("Slide 30: voice request, GPT-2 XL reply")
    show("weights per token", f"{GPT2_BYTES / 1e9:.0f} GB")
    show("memory floor", f"{GPT2_MEM_MS:.1f} ms")
    show("compute floor", f"{GPT2_COMP_MS * 1e3:.2f} us")
    show("ratio", f"{GPT2_MEM_MS / GPT2_COMP_MS:.0f}x")
