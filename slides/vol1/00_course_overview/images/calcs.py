#!/usr/bin/env python3
"""Every derived number Lecture 0 shows, computed from sourced inputs.

Run from this folder: python3 calcs.py
Each line prints the value and the slide (or note) that shows it. Copy the
printed values onto the slides; never compute a slide number by hand.
"""

# --- One chat answer (Model Serving, Napkin Math 13.5) ----------------------
GPU_W = 700            # H100 thermal design power (board limit at full load)
HOST_W = 300           # host share: CPUs, memory, fans (Napkin Math 13.5)
USERS = 114            # concurrent requests on one GPU
TOK_PER_USER = 8       # tokens/s each user receives
ANSWER_TOK = 500       # tokens in one answer
WORDS_PER_TOKEN = 0.75  # OpenAI tokenizer guide: one token ~ 3/4 word
GOOGLE_WH = 0.24       # Elsworth et al. 2025: median Gemini Apps text prompt
PHONE_J = 40_000       # charging a smartphone (Napkin Math 13.5)

node_w = GPU_W + HOST_W
rate = USERS * TOK_PER_USER
j_per_tok = node_w / rate
j_answer = j_per_tok * ANSWER_TOK
google_j = GOOGLE_WH * 3600

# --- Utilization (Napkin Math 13.5, same box) ------------------------------
LOW_GPU_W = 300        # GPU draw at low load ("still draws ~300 W")
LOW_FRACTION = 0.10    # 10 percent of the batched token rate
low_rate = rate * LOW_FRACTION
low_j = (LOW_GPU_W + HOST_W) / low_rate

# --- Google's measured breakdown (Elsworth et al. 2025, figure 2 and text) --
G_ACCEL_WH, G_HOST_WH, G_IDLE_WH, G_PUE_WH = 0.14, 0.06, 0.02, 0.02
G_MODEL_X, G_UTIL_X, G_ENERGY_X = 23, 1.4, 33   # one-year reductions, May 2024 to May 2025

# --- Fleet scale -------------------------------------------------------------
PROMPTS_PER_DAY = 2.5e9    # ChatGPT, OpenAI to Axios, July 2025
GPT3_MWH = 1287            # Patterson et al. 2021
fleet_mwh = PROMPTS_PER_DAY * GOOGLE_WH / 1e6

# --- GPT-3 on 1,024 GPUs (Introduction, Napkin Math 1.2) --------------------
WORK = 3.14e23
N_GPU = 1024
PEAK = 312e12
UTIL, UTIL_HI = 0.45, 0.60
sec_per_flop = 1 / (N_GPU * PEAK * UTIL)
t = WORK * sec_per_flop
t_hi = WORK / (N_GPU * PEAK * UTIL_HI)

# --- Energy per operation (Introduction 1.7, after Horowitz 2014) ------------
DRAM_PJ, FP16_PJ, INT8_PJ = 160, 1.1, 0.2

# --- Roofline (Hardware Acceleration 11.6) -----------------------------------
H100_FP16_TFLOPS, H100_TBPS = 989, 3.35
FLOPS_PER_WEIGHT, BYTES_PER_WEIGHT = 2, 2   # decode, FP16 weights, batch 1

# --- Power tiers (System Paradigms 2.1) and silent failure ------------------
CLOUD_W, TINY_W = 3e6, 1e-3
RECALL_0, RECALL_12 = 0.95, 0.85

# --- Drift report (Evidently 0.7.23 run, UCI Bike Sharing, Jan vs Jun 2011) --
COLS, DRIFTED = 6, 5

import math  # noqa: E402

rows = [
    ("Predict: 500 tokens in words", f"{ANSWER_TOK * WORDS_PER_TOKEN:.0f} words"),
    ("Worked example: node power", f"{node_w:,} W"),
    ("Worked example: token rate", f"{rate} tokens/s"),
    ("Worked example: J per token", f"{j_per_tok:.2f} J (slide: 1.1 J)"),
    ("Worked example: J per answer", f"{j_answer:.1f} J (slide: 548 J)"),
    ("Worked example: GPU share of node power", f"{GPU_W / node_w:.0%}"),
    ("Google median prompt in joules", f"{google_j:.0f} J"),
    ("Google: accelerator share (text)", f"{G_ACCEL_WH / GOOGLE_WH:.0%}"),
    ("Google: host CPU and DRAM share (text; figure shows 24%)", f"{G_HOST_WH / GOOGLE_WH:.0%}"),
    ("Google: model x utilization reductions", f"{G_MODEL_X * G_UTIL_X:.1f}x (reported {G_ENERGY_X}x, rounded factors)"),
    ("Phone charge in answers", f"{PHONE_J / j_answer:.0f} answers"),
    ("Fleet: prompts x 0.24 Wh", f"{fleet_mwh:.0f} MWh per day"),
    ("Fleet: GPT-3 runs per day", f"{fleet_mwh / GPT3_MWH:.2f} (one run every {GPT3_MWH / fleet_mwh:.1f} days)"),
    ("GPT-3: seconds per FLOP", f"{sec_per_flop:.2e} s"),
    ("GPT-3: time at 45%", f"{t:.2e} s = {t / 86400:.1f} days"),
    ("GPT-3: time at 60%", f"{t_hi / 86400:.1f} days, saving {(t - t_hi) / 86400:.1f} days"),
    ("DRAM byte vs FP16 multiply", f"{DRAM_PJ / FP16_PJ:.0f}x"),
    ("H100 ridge point", f"{H100_FP16_TFLOPS / H100_TBPS:.0f} FLOP/byte"),
    ("Decode intensity, batch 1", f"{FLOPS_PER_WEIGHT / BYTES_PER_WEIGHT:.0f} FLOP/byte"),
    ("Utilization: low token rate", f"{low_rate:.0f} tokens/s"),
    ("Utilization: J per token at 10%", f"{low_j:.1f} J"),
    ("Utilization: ratio to batched", f"{low_j / j_per_tok:.1f}x"),
    ("Power tiers: cloud over TinyML", f"10^{math.log10(CLOUD_W / TINY_W):.1f}"),
    ("Silent failure: miss rate before, after", f"{1 - RECALL_0:.0%} -> {1 - RECALL_12:.0%}, x{(1 - RECALL_12) / (1 - RECALL_0):.0f}"),
    ("Drift report: share of drifted columns", f"{DRIFTED}/{COLS} = {DRIFTED / COLS:.3f}"),
]

for label, value in rows:
    print(f"{label:58s} {value}")
