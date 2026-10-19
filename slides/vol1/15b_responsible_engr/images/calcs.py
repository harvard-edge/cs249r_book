#!/usr/bin/env python3
"""Every derived number in Lecture 15b (Responsible Engineering, Part 2),
computed from sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports from this file so figures and slides share
one computation. "RE" means the chapter Responsible Engineering in Volume I.

Run from this folder: python3 calcs.py
"""

import math


def r(x, nd=1):
    """Round half away from zero (Python's round() is half to even)."""
    q = 10 ** nd
    return math.floor(abs(x) * q + 0.5 + 1e-9) / q * (1 if x >= 0 else -1)


def f(x, nd=1, commas=False):
    v = r(x, nd)
    return f"{v:,.{nd}f}" if commas else f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# Hook: how much did one architecture search emit?
# Strubell, Ganesh, McCallum, ACL 2019, Table 1: 626,155 lb CO2e for
# Transformer (big) with NAS; one car over its lifetime 126,000 lb.
# Patterson et al. 2021 (arXiv 2104.10350v3), Appendix D, p. 22: the actual
# search ran 6.8 days on 200 TPU v2 chips at a measured 208 W, PUE 1.10,
# 0.431 kg CO2e/kWh; their own line is
# "(6.8 * 24 * 200 * 208 * 1.10 / 1000) * 0.431 / 1000 = 3.2 tCO2e".
# Appendix C: proxy task 18.7x. Appendix D: 117,780 P100 hours at 189 W
# against 32,633 TPU v2 hours at 208 W; US PUE 1.58 and grid 0.423.
# ---------------------------------------------------------------------------
LB_PER_KG = 2.20462               # 1 kg = 2.20462 lb
NAS_EST_LB, CAR_LB = 626_155, 126_000
NAS_EST_T = NAS_EST_LB / LB_PER_KG / 1000
CAR_T = CAR_LB / LB_PER_KG / 1000
NAS_DAYS, NAS_CHIPS, NAS_W, NAS_PUE, NAS_GRID = 6.8, 200, 208, 1.10, 0.431
NAS_KWH = NAS_DAYS * 24 * NAS_CHIPS * NAS_W * NAS_PUE / 1000
NAS_MEAS_T = NAS_KWH * NAS_GRID / 1000
NAS_RATIO = NAS_EST_T / 3.2       # against the paper's printed 3.2 t
PROXY = 18.7                      # Appendix C
TPU_PERF_W = (117_780 / 32_633) * (189 / 208)
PUE_RATIO = 1.58 / 1.10
GRID_RATIO = 0.423 / 0.431
FACTORS = PROXY * TPU_PERF_W * PUE_RATIO
FACTORS_WITH_GRID = FACTORS * GRID_RATIO
MEENA_SAVED_T = 48.5              # Patterson et al. 2021, Table 4 note
PAYBACK = MEENA_SAVED_T / 3.2

# ---------------------------------------------------------------------------
# Recall (Part 1): a 3-point gap grows by 1.10 a quarter; when does it pass 5?
# Inputs from Lecture 15 (RE Napkin Math 15.3).
# ---------------------------------------------------------------------------
GAP0, ALPHA, GATE = 3.0, 1.10, 5.0
K_CROSS = math.log(GATE / GAP0) / math.log(ALPHA)
Q_CROSS = math.ceil(K_CROSS)
GAP_AT_Q = GAP0 * ALPHA ** Q_CROSS

# ---------------------------------------------------------------------------
# One Gemini prompt. Elsworth et al. 2025 (arXiv 2508.15734), Table 1
# (Wh: 0.14 accelerators, 0.06 host, 0.02 idle, 0.02 overhead), Table 2
# (0.023 g Scope 2 market-based, 0.010 g Scope 1 and 3), Sec. 4.3 (33x
# energy, 44x carbon; 23x from models, 1.4x from utilization).
# ---------------------------------------------------------------------------
GEM_PARTS = (0.14, 0.06, 0.02, 0.02)
GEM_WH = sum(GEM_PARTS)
GEM_G_S2, GEM_G_S13 = 0.023, 0.010
GEM_INTENSITY = GEM_G_S2 / GEM_WH          # g/Wh = kg/kWh
GEM_TOTAL_G = GEM_G_S2 + GEM_G_S13
GEM_MODEL_X_UTIL = 23 * 1.4

# ---------------------------------------------------------------------------
# Four choices on one training job. Patterson et al. 2021, Table 1
# (0.1357 t and 0.0024 t net; 316 and 30 kWh) and Figure 1 (cumulative gains
# printed: 1.3, 7.4, 10.5, 56.5). Sec. 2.3 PUE 1.59 to 1.11; Sec. 2.4 grid
# 0.429 to 0.080 net. Per-step factors are derived from the printed values.
# ---------------------------------------------------------------------------
CUM = {"model": 1.3, "processor": 7.4, "datacenter": 10.5, "grid": 56.5}
STEP_PROC = CUM["processor"] / CUM["model"]
STEP_DC = CUM["datacenter"] / CUM["processor"]
STEP_GRID = CUM["grid"] / CUM["datacenter"]
PUE_STEP = 1.59 / 1.11
GRID_STEP = 0.429 / 0.080
TABLE1_RATIO = 0.1357 / 0.0024

# ---------------------------------------------------------------------------
# Estimate: GPT-3 against BLOOM. RE Napkin Math 15.5 (GPT-3 1,287 MWh with
# PUE 1.1 included; 0.429 kg/kWh; car-year 4.6 t). Luccioni et al. 2023,
# Sec. 4.2 and Table 4: BLOOM 433,195 kWh at the GPUs, PUE 1.2, 0.057 kg/kWh;
# Table 4 rounds BLOOM with PUE to 30 t.
# ---------------------------------------------------------------------------
GPT3_KWH, GPT3_GRID, CAR_YEAR_T = 1_287_000, 0.429, 4.6
BLOOM_GPU_KWH, BLOOM_PUE, BLOOM_GRID = 433_195, 1.2, 0.057
GPT3_T = GPT3_KWH * GPT3_GRID / 1000
GPT3_CAR_YEARS = GPT3_T / CAR_YEAR_T
BLOOM_KWH = BLOOM_GPU_KWH * BLOOM_PUE
BLOOM_T = BLOOM_KWH * BLOOM_GRID / 1000
ENERGY_RATIO = GPT3_KWH / BLOOM_KWH
GRID_RATIO_GB = GPT3_GRID / BLOOM_GRID
CARBON_RATIO = GPT3_T / BLOOM_T
NAIVE_ENERGY = GPT3_KWH / BLOOM_GPU_KWH
NAIVE_CARBON = GPT3_T / (BLOOM_GPU_KWH * BLOOM_GRID / 1000)
LLM4 = {"GPT-3": 552, "Gopher": 380, "OPT": 76.3, "BLOOM": 30}   # Table 4
LLM4_SPAN = LLM4["GPT-3"] / LLM4["BLOOM"]

# ---------------------------------------------------------------------------
# BLOOM life cycle. Luccioni et al. 2023, Table 3 (t): embodied 11.2,
# dynamic 24.69, idle 14.6; Sec. 4.1: servers 7.57 t, GPUs 3.64 t; a server
# 2,500 kg and a GPU 150 kg to make, 6-year life, 85 percent use; Sec. 4.3
# and Table 2: 27, 64, 109 kWh per interval (off, idle, training).
# ---------------------------------------------------------------------------
B_DYN, B_IDLE, B_EMB = 24.69, 14.6, 11.2
B_TOTAL = B_DYN + B_IDLE + B_EMB
B_DYN_SHARE = 100 * B_DYN / B_TOTAL
B_TIMES_DYN = B_TOTAL / B_DYN
B_CODE_SHARE = 100 * (109 - 64) / 109
LIFE_H = 6 * 365 * 24 * 0.85
SERVER_KG_H = 2500 / LIFE_H
GPU_KG_H = 150 / LIFE_H

# ---------------------------------------------------------------------------
# BLOOM model card (Hugging Face, bigscience/bloom @ 7f10a99, README.md
# metadata): co2_eq_emissions: emissions: 24700000 (grams).
# ---------------------------------------------------------------------------
CARD_G = 24_700_000
CARD_T = CARD_G / 1e6
CARD_SHARE = 100 * CARD_T / B_TOTAL

# ---------------------------------------------------------------------------
# DLRM ranking service (book scenario, RE Sec. 15.4.2 and table 10):
# 10 million users x 20 items; 10 ms GPU time each; $2.50 per GPU-hour
# reserved; training cycle $400 + $2,000 + $800 at $4 an hour on demand;
# operations $170K a year; 400 W per GPU on 0.4 kg/kWh (Napkin Math 15.4).
# ---------------------------------------------------------------------------
USERS, ITEMS, GPU_S, PRICE = 10e6, 20, 0.010, 2.50
INF_DAY = USERS * ITEMS
GPU_H_DAY = INF_DAY * GPU_S / 3600
GPUS = GPU_H_DAY / 24
SERVE_YEAR = GPU_H_DAY * 365 * PRICE
SERVERS_8 = GPUS / 8
CYCLE = 400 + 2000 + 800
CYCLE_H = CYCLE / 4.0
TRAIN_3Y = 12 * CYCLE
INF_3Y = GPU_H_DAY * 1095 * PRICE
OPS_3Y = 170_000 * 3
TCO = TRAIN_3Y + INF_3Y + OPS_3Y
TRAIN_SHARE = 100 * TRAIN_3Y / TCO
INF_SHARE = 100 * INF_3Y / TCO
COST_RATIO = INF_3Y / TRAIN_3Y
KG_PER_GPU_H = 0.4 * 0.4
INF_GPU_H_3Y = GPU_H_DAY * 1095
INF_T = INF_GPU_H_3Y * KG_PER_GPU_H / 1000
TRAIN_T = 12 * CYCLE_H * KG_PER_GPU_H / 1000
CARBON_RATIO_DLRM = INF_T / TRAIN_T

# Estimate: cut serving 20 percent or halve all training (Checkpoint 15.4)
SAVE_SERVE = 0.2 * INF_3Y
SAVE_SERVE_T = 0.2 * INF_T
SAVE_TRAIN = 0.5 * TRAIN_3Y
SAVE_TRAIN_T = 0.5 * TRAIN_T
SAVE_SERVE_PCT = 100 * SAVE_SERVE / TCO
SAVE_TRAIN_PCT = 100 * SAVE_TRAIN / TCO
HALVINGS = SAVE_SERVE / SAVE_TRAIN
FINAL_RUN_HALF = 800 / 2

# Real fleets: BLOOM API (Luccioni et al. 2023, Sec. 4.4): 914 kWh,
# 230,768 requests.
API_WH = 914 * 1000 / 230_768

# Embodied share of one GPU-hour (derived; Luccioni Sec. 4.1 factors,
# 8 GPUs per server assumed, 400 W GPU).
OP_DIRTY_G = 0.4 * 0.4 * 1000
OP_CLEAN_G = 0.4 * 0.057 * 1000
EMB_G = (GPU_KG_H + SERVER_KG_H / 8) * 1000
EMB_SHARE_DIRTY = 100 * EMB_G / (OP_DIRTY_G + EMB_G)
EMB_SHARE_CLEAN = 100 * EMB_G / (OP_CLEAN_G + EMB_G)
DLRM_EMB_T = INF_GPU_H_3Y * EMB_G / 1e6

# ---------------------------------------------------------------------------
# Duties. DPC press release, 4 January 2023: EUR 210M + EUR 180M.
# Credit service (constructed; Part 1 worked example: 6,300 of 12,000
# approved); 1 million decisions a day; 1,000-sample attributor; 20 features.
# Review: 0.1 percent appealed (RE Sec. 15.5.3); 6 minutes each (assumed);
# Cigna: 300,000 denials at 1.2 s each (ProPublica, 2023).
# ---------------------------------------------------------------------------
DPC_TOTAL_M = 210 + 180
DENY = 1 - 6300 / 12000
DECISIONS = 1e6
NOTICES = DECISIONS * DENY
SAMPLED = NOTICES * 1000
SAMPLED_X = SAMPLED / DECISIONS
EXACT_PER = 2 ** 20
EXACT = NOTICES * EXACT_PER
APPEALS = 0.001 * DECISIONS
REVIEW_H = APPEALS * 6 / 60
REVIEWERS = REVIEW_H / 8
CIGNA_H = 300_000 * 1.2 / 3600
PER_H_CREDIT = 60 / 6
PER_H_CIGNA = 3600 / 1.2

# ---------------------------------------------------------------------------
# Governance. DP definition: e^epsilon at epsilon 8.
# Per-sample gradients (RE Sec. 15.5.2): KWS 200,000 FP32 parameters;
# GPT-2 1.5 billion; batch of 256.
# ---------------------------------------------------------------------------
E8 = math.exp(8)
KWS_GRAD_KB = 200_000 * 4 / 1000
KWS_BATCH_MB = 200_000 * 4 * 256 / 1e6
GPT2_GRAD_GB = 1.5e9 * 4 / 1e9
GPT2_BATCH_TB = 1.5e9 * 4 * 256 / 1e12

# DP accuracy. De et al. 2022 (arXiv 2204.13650v2): CIFAR-10 81.4 at
# epsilon 8, previous best 71.7, trained from scratch (abstract); WRN-40-4
# reaches >95 without privacy (Sec. 3, citing Zagoruyko and Komodakis 2016);
# ImageNet 86.7 at epsilon 8 fine-tuning an NFNet-F3 pre-trained on JFT-4B
# (Table 6); the same model fine-tuned without privacy 88.5 (Table 6, Sec. 4);
# best non-private ImageNet model of any kind 91.0 (Yu et al. 2022, Fig. 1).
CIFAR_DP, CIFAR_PREV, CIFAR_NONPRIV = 81.4, 71.7, 95.0
IMNET_DP, IMNET_SAME, IMNET_BEST = 86.7, 88.5, 91.0
CIFAR_GAIN = CIFAR_DP - CIFAR_PREV
CIFAR_GAP = CIFAR_NONPRIV - CIFAR_DP
IMNET_GAP_SAME = IMNET_SAME - IMNET_DP
IMNET_GAP_BEST = IMNET_BEST - IMNET_DP

# KWS bytes: 16 kHz, 16-bit audio (assumed); update 200K FP32.
AUDIO_BPS = 16_000 * 2
AUDIO_MB_H = AUDIO_BPS * 3600 / 1e6
UPDATE_MB = 200_000 * 4 / 1e6
AUDIO_RATIO = AUDIO_MB_H / UPDATE_MB

# Audit log (RE Sec. 15.5.5, table 12): 256 B per record (assumed);
# 25 months (12 CFR 1002.12(b)(1)), taken as 760 days.
DAYS_25M = 25 * 365.25 / 12
LOG_MB_DAY = DECISIONS * 256 / 1e6
LOG_GB = DECISIONS * 256 * 760 / 1e9
DLRM_MANIFEST_TB = INF_DAY * 256 * 365 / 1e12
INPUT_B = 13 * 4 + 26 * 128 * 4       # 13 dense FP32 + 26 x 128 FP32
DLRM_INPUT_TB = INF_DAY * INPUT_B * 365 / 1e12
PAYLOAD_RATIO = INPUT_B / 256

# CodeCarbon 3.3.1, run here on 19 Oct 2026 (OfflineEmissionsTracker,
# country USA, 30 s of 2048 x 2048 float32 matmuls on an Apple M5 Max);
# emissions.csv fields: energy_consumed 0.00018553642733992439 kWh,
# cpu_energy 0.00013543562574311828, emissions 6.855070041856389e-05 kg.
CC_KWH, CC_CPU_KWH, CC_KG = 0.00018553642733992439, 0.00013543562574311828, 6.855070041856389e-05
CC_WH = CC_KWH * 1000
CC_INTENSITY = CC_KG / CC_KWH
CC_CPU_SHARE = 100 * CC_CPU_KWH / CC_KWH
CC_MG = CC_KG * 1e6

# Ledger
INF_T_YEAR = INF_T / 3
EMB_T_YEAR = DLRM_EMB_T / 3


def main():
    def head(t):
        print(f"\n[{t}]")

    def show(label, v):
        print(f"  {label}: {v}")

    head("Hook: architecture search")
    show("estimate, t", f(NAS_EST_T, 0))
    show("car lifetime, t", f(CAR_T, 0))
    show("estimate / car", f(NAS_EST_T / CAR_T, 1))
    show("Appendix D energy, kWh", f(NAS_KWH, 0, True))
    show("Appendix D carbon, t", f(NAS_MEAS_T, 2))
    show("estimate / 3.2 t", f(NAS_RATIO, 1))
    show("TPU v2 perf/W factor", f(TPU_PERF_W, 2))
    show("PUE factor 1.58/1.10", f(PUE_RATIO, 2))
    show("proxy x hw x PUE", f(FACTORS, 1))
    show("... x grid 0.423/0.431", f(FACTORS_WITH_GRID, 1))
    show("Meena payback", f(PAYBACK, 1))

    head("Recall")
    show("k at crossing", f(K_CROSS, 2))
    show("quarter", Q_CROSS)
    show("gap at that quarter", f(GAP_AT_Q, 1))

    head("Gemini prompt")
    show("energy, Wh", f(GEM_WH, 2))
    show("intensity, kg/kWh", f(GEM_INTENSITY, 3))
    show("total, g", f(GEM_TOTAL_G, 3))
    show("23 x 1.4", f(GEM_MODEL_X_UTIL, 1))

    head("Four factors")
    show("processor step", f(STEP_PROC, 1))
    show("datacenter step", f(STEP_DC, 2))
    show("PUE 1.59/1.11", f(PUE_STEP, 2))
    show("grid step", f(STEP_GRID, 2))
    show("grid 0.429/0.080", f(GRID_STEP, 2))
    show("Table 1 ratio", f(TABLE1_RATIO, 1))

    head("Estimate: GPT-3 against BLOOM")
    show("GPT-3, t", f(GPT3_T, 1))
    show("GPT-3 car-years", f(GPT3_CAR_YEARS, 0))
    show("BLOOM energy with PUE, kWh", f(BLOOM_KWH, 0, True))
    show("BLOOM, t", f(BLOOM_T, 1))
    show("energy ratio", f(ENERGY_RATIO, 2))
    show("grid ratio", f(GRID_RATIO_GB, 2))
    show("carbon ratio", f(CARBON_RATIO, 1))
    show("naive energy ratio", f(NAIVE_ENERGY, 1))
    show("naive carbon ratio", f(NAIVE_CARBON, 1))
    show("Table 4 span 552/30", f(LLM4_SPAN, 1))

    head("BLOOM life cycle")
    show("total, t", f(B_TOTAL, 1))
    show("dynamic share, %", f(B_DYN_SHARE, 1))
    show("total / dynamic", f(B_TIMES_DYN, 2))
    show("code share of training power, %", f(B_CODE_SHARE, 0))
    show("server kg per hour", f(SERVER_KG_H, 3))
    show("GPU kg per hour", f(GPU_KG_H, 4))

    head("BLOOM model card")
    show("card, t", f(CARD_T, 1))
    show("card share of 50.5 t, %", f(CARD_SHARE, 0))

    head("DLRM")
    show("inferences a day", f(INF_DAY, 0, True))
    show("GPU-hours a day", f(GPU_H_DAY, 1))
    show("GPUs", f(GPUS, 1))
    show("8-GPU servers", f(SERVERS_8, 1))
    show("serving a year, $", f(SERVE_YEAR, 0, True))
    show("cycle, $", CYCLE)
    show("train 3y, $", f(TRAIN_3Y, 0, True))
    show("inference 3y, $", f(INF_3Y, 0, True))
    show("ops 3y, $", f(OPS_3Y, 0, True))
    show("TCO, $", f(TCO, 0, True))
    show("train share, %", f(TRAIN_SHARE, 1))
    show("inference share, %", f(INF_SHARE, 1))
    show("cost ratio", f(COST_RATIO, 1))
    show("kg per GPU-hour", f(KG_PER_GPU_H, 2))
    show("inference GPU-hours 3y", f(INF_GPU_H_3Y, 0, True))
    show("inference t", f(INF_T, 1))
    show("training t", f(TRAIN_T, 2))
    show("carbon ratio", f(CARBON_RATIO_DLRM, 1))

    head("Estimate: serving cut against training halving")
    show("serving cut, $", f(SAVE_SERVE, 0, True))
    show("serving cut, t", f(SAVE_SERVE_T, 1))
    show("halve training, $", f(SAVE_TRAIN, 0, True))
    show("halve training, t", f(SAVE_TRAIN_T, 2))
    show("serving cut share of TCO, %", f(SAVE_SERVE_PCT, 1))
    show("training halving share of TCO, %", f(SAVE_TRAIN_PCT, 1))
    show("halvings", f(HALVINGS, 1))
    show("half the final run, $", f(FINAL_RUN_HALF, 0))

    head("Fleets and embodied")
    show("BLOOM API Wh per request", f(API_WH, 2))
    show("operational, dirty grid, g", f(OP_DIRTY_G, 0))
    show("operational, French grid, g", f(OP_CLEAN_G, 1))
    show("embodied per GPU-hour, g", f(EMB_G, 1))
    show("embodied share, dirty, %", f(EMB_SHARE_DIRTY, 1))
    show("embodied share, clean, %", f(EMB_SHARE_CLEAN, 1))
    show("DLRM embodied 3y, t", f(DLRM_EMB_T, 1))

    head("Duties")
    show("DPC fine, EUR M", DPC_TOTAL_M)
    show("denial share, %", f(100 * DENY, 1))
    show("notices a day", f(NOTICES, 0, True))
    show("sampled evaluations", f(SAMPLED, 0, True))
    show("sampled / decisions", f(SAMPLED_X, 0))
    show("2^20", f(EXACT_PER, 0, True))
    show("exact evaluations", f"{EXACT:.3e}")
    show("appeals a day", f(APPEALS, 0, True))
    show("reviewer-hours", f(REVIEW_H, 0))
    show("reviewers on 8 h", f(REVIEWERS, 1))
    show("Cigna hours", f(CIGNA_H, 0))
    show("cases per hour, credit / Cigna", f"{PER_H_CREDIT:.0f} / {PER_H_CIGNA:,.0f}")

    head("Governance")
    show("e^8", f(E8, 0, True))
    show("KWS gradient, KB", f(KWS_GRAD_KB, 0))
    show("KWS batch, MB", f(KWS_BATCH_MB, 1))
    show("GPT-2 gradient, GB", f(GPT2_GRAD_GB, 1))
    show("GPT-2 batch, TB", f(GPT2_BATCH_TB, 3))
    show("CIFAR gain over previous, points", f(CIFAR_GAIN, 1))
    show("CIFAR gap to non-private (>95), points", f(CIFAR_GAP, 1))
    show("ImageNet gap, same model", f(IMNET_GAP_SAME, 1))
    show("ImageNet gap, best model", f(IMNET_GAP_BEST, 1))
    show("audio bytes/s", f(AUDIO_BPS, 0, True))
    show("audio MB/h", f(AUDIO_MB_H, 1))
    show("update MB", f(UPDATE_MB, 1))
    show("audio / update", f(AUDIO_RATIO, 0))
    show("25 months in days", f(DAYS_25M, 1))
    show("log MB a day", f(LOG_MB_DAY, 0))
    show("log GB", f(LOG_GB, 1))
    show("DLRM manifests TB", f(DLRM_MANIFEST_TB, 2))
    show("input bytes", f(INPUT_B, 0, True))
    show("DLRM inputs TB", f(DLRM_INPUT_TB, 1))
    show("payload ratio", f(PAYLOAD_RATIO, 1))

    head("CodeCarbon run")
    show("energy, Wh", f(CC_WH, 3))
    show("implied intensity, kg/kWh", f(CC_INTENSITY, 3))
    show("CPU share, %", f(CC_CPU_SHARE, 0))
    show("emissions, mg", f(CC_MG, 1))

    head("Ledger")
    show("inference t a year", f(INF_T_YEAR, 1))
    show("embodied t a year", f(EMB_T_YEAR, 1))


if __name__ == "__main__":
    main()
