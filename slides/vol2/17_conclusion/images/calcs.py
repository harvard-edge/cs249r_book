#!/usr/bin/env python3
"""Every derived number in Lecture 17 (Conclusion), computed once.

Inputs are typed with their source; every printed value is copied onto the
slide and into its note with the label printed here. make_figs.py imports
this module, so the figures use the same values. "Chapter 17" is the
rendered Conclusion chapter; other chapters are named by number.

Run from this folder: python3 calcs.py
"""

import math

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
# Running case: Archetype A reference job (Chapter 17 Sec. 17.3)
GPU_MTTF_H = 50_000          # registry GPU MTTF, hours (Chapter 17 Sec. 17.3)
N_REF = 1_024                # reference cluster, GPUs (Chapter 17 Sec. 17.3)
N_BIG = 2_048                # proposed expansion (Chapter 17 Sec. 17.3)
CKPT_WRITE_S = 24.5          # 2.45 TB checkpoint write time, s (Chapter 17 Sec. 17.3)
CKPT_TB = 2.45               # checkpoint size, TB (Chapter 17 Sec. 17.3)
KAPLAN_EXP = -0.050          # loss ~ compute^-0.050 (Chapter 17 Sec. 17.3; Kaplan et al. 2020)

# Llama 3 405B (Dubey et al. 2024, Sec. 3.3.4 and Table 5, read from arXiv 2407.21783 v3)
LLAMA_GPUS = 16_384
LLAMA_DAYS = 54
LLAMA_UNEXPECTED = 419
LLAMA_TABLE5 = {             # component: (category, count)
    "Faulty GPU": ("GPU", 148), "GPU HBM3 memory": ("GPU", 72),
    "Software bug": ("Dependency", 54), "Network switch/cable": ("Network", 35),
    "Host maintenance": ("Unplanned maintenance", 32), "GPU SRAM memory": ("GPU", 19),
    "GPU system processor": ("GPU", 17), "NIC": ("Host", 7),
    "NCCL watchdog timeouts": ("Unknown", 7), "Silent data corruption": ("GPU", 6),
    "GPU thermal interface and sensor": ("GPU", 6), "SSD": ("Host", 3),
    "Power supply": ("Host", 3), "Server chassis": ("Host", 2),
    "IO expansion board": ("Host", 2), "Dependency": ("Dependency", 2),
    "CPU": ("Host", 2), "System memory": ("Host", 2),
}
KOKOLIS_16K_MTTF_H = 1.8     # projected MTTF, 16,384-GPU job (Chapter 7 Sec. 7.2; Kokolis et al. 2025)

# Tail at scale (Chapter 17 Sec. 17.2 and footnote 2; Dean and Barroso 2013, Table 1, read from the paper)
P_SLOW = 0.01                # one server beyond its p99
FANOUT = 100
DB_ONE_LEAF_P99_MS = 10      # one random leaf finishes, p99
DB_95_LEAVES_P99_MS = 70     # 95% of all leaf requests finish, p99
DB_ALL_LEAVES_P99_MS = 140   # 100% of all leaf requests finish, p99
REQ_PER_DAY = 1_000_000      # Chapter 17 Sec. 17.2
P_RARE = 1e-6

# Straggler fallacy (Chapter 17 Sec. 17.6)
STRAGGLER_SPEED = 0.80

# MegaScale strong scaling, 175B, batch 768, NVIDIA Ampere GPUs
# (Jiang et al. 2024, NSDI, Table 2, read from arXiv 2402.15627 v1): tokens/s
MEGASCALE = {256: 49.0e3, 512: 95.1e3, 768: 136.7e3, 1024: 176.9e3}
MEGASCALE_MFU = {256: 65.3, 512: 63.5, 768: 61.3, 1024: 59.0}
MEGATRON_MFU = {256: 53.0, 512: 49.9, 768: 46.7, 1024: 44.7}

# Carbon (Chapter 17 Sec. 17.2; Chapter 15 Napkin Math 15.2: rounded annual grid averages)
GPT3_MWH = 1_287
POLAND_G_PER_KWH = 820
QUEBEC_G_PER_KWH = 20

# Where the constraint moves next (Chapter 10)
TT_TOKENS = 128              # Napkin Math 10.4
TT_MS_PER_TOKEN = 100        # Napkin Math 10.4 (round figure)
IX_LOW_INTERACT = 6_300      # InferenceX, DeepSeek-R1 FP8, best B200, tok/s/GPU at ~13 tok/s/user (Sec. 10.6)
IX_HIGH_INTERACT = 210       # same, at >= 100 tok/s/user
MOE_RESIDENT_GB = 1_342      # DeepSeek-V3 671B in FP16 (Sec. 10.10.4)
MOE_ACTIVE_GB = 74           # 37B active in FP16
DENSE_400B_GB = 800          # dense 400B in FP16
LINK_SHARE_ILLUS = 0.10      # illustrative share of fleet power drawn by links
LINK_ENERGY_RATIO = 0.5      # optical link halves energy per bit (Chapter 17 self-check 17.5)

# Stack row numbers (one per row, from the owning chapter, inputs only)
STAIR_GB = 10                # Napkin Math 2.3
HBM_GBS, NVLINK_GBS, IB_GBS = 3_350, 450, 50

# ---------------------------------------------------------------------------
results = {}


def show(label, value, fmt="{:.3g}"):
    results[label] = value
    print(f"{label:58s} {fmt.format(value)}")


def mtbi(n):
    return GPU_MTTF_H / n


def yd_overhead(n):
    """Young-Daly checkpoint-and-rework overhead, sqrt(2 delta / M)."""
    return math.sqrt(2 * CKPT_WRITE_S / (mtbi(n) * 3600))


def main():
    print("== Stack: the staircase (Napkin Math 2.3)")
    show("10 GB over HBM, ms", STAIR_GB / HBM_GBS * 1e3, "{:.1f}")
    show("10 GB over NVLink, ms", STAIR_GB / NVLINK_GBS * 1e3, "{:.1f}")
    show("10 GB over InfiniBand, ms", STAIR_GB / IB_GBS * 1e3, "{:.0f}")
    show("NVLink / InfiniBand bandwidth ratio", NVLINK_GBS / IB_GBS, "{:.0f}")

    print("== Failure clock")
    show("MTBI 1,024 GPUs, h", mtbi(N_REF), "{:.1f}")
    show("MTBI 2,048 GPUs, h", mtbi(N_BIG), "{:.1f}")
    show("MTBI 16,384 GPUs (model), h", mtbi(LLAMA_GPUS), "{:.2f}")
    show("Llama 3 measured hours per unexpected interruption",
         LLAMA_DAYS * 24 / LLAMA_UNEXPECTED, "{:.1f}")
    cats = {}
    for _, (cat, k) in LLAMA_TABLE5.items():
        cats[cat] = cats.get(cat, 0) + k
    for cat, k in sorted(cats.items(), key=lambda t: -t[1]):
        show(f"Llama 3 interruptions, {cat}", k, "{:.0f}")
    show("Llama 3 interruptions, total", sum(cats.values()), "{:.0f}")
    show("Llama 3 GPU share of unexpected, percent", 100 * cats["GPU"] / LLAMA_UNEXPECTED, "{:.0f}")
    show("Llama 3 non-GPU unexpected interruptions", LLAMA_UNEXPECTED - cats["GPU"], "{:.0f}")
    show("Llama 3 non-GPU share of unexpected, percent", 100 * (LLAMA_UNEXPECTED - cats["GPU"]) / LLAMA_UNEXPECTED, "{:.0f}")
    show("Llama 3 interruptions per day", LLAMA_UNEXPECTED / LLAMA_DAYS, "{:.1f}")

    print("== Tail at scale")
    show("P(all 100 fast), percent", 100 * (1 - P_SLOW) ** FANOUT, "{:.1f}")
    show("P(at least one slow of 100), percent", 100 * (1 - (1 - P_SLOW) ** FANOUT), "{:.1f}")
    show("Dean-Barroso all-leaves p99 / one-leaf p99", DB_ALL_LEAVES_P99_MS / DB_ONE_LEAF_P99_MS, "{:.0f}")
    show("Rare failures per day at 1M requests", REQ_PER_DAY * P_RARE, "{:.0f}")
    show("Rare failures per day at 10M requests", 10 * REQ_PER_DAY * P_RARE, "{:.0f}")
    show("Straggler step-time multiplier", 1 / STRAGGLER_SPEED, "{:.2f}")
    show("Straggler throughput loss, percent", 100 * (1 - STRAGGLER_SPEED), "{:.0f}")

    print("== The expansion, 1,024 -> 2,048 GPUs")
    o1, o2 = yd_overhead(N_REF), yd_overhead(N_BIG)
    show("Young-Daly overhead 1,024, percent", 100 * o1, "{:.1f}")
    show("Young-Daly overhead 2,048, percent", 100 * o2, "{:.1f}")
    show("Overhead ratio (sqrt 2)", o2 / o1, "{:.2f}")
    show("Young-Daly interval 1,024, min",
         math.sqrt(2 * CKPT_WRITE_S * mtbi(N_REF) * 3600) / 60, "{:.0f}")
    show("Young-Daly interval 2,048, min",
         math.sqrt(2 * CKPT_WRITE_S * mtbi(N_BIG) * 3600) / 60, "{:.0f}")
    retained = (1 - o1) / (1 - o2)
    show("Retained fraction 1,024", 1 - o1, "{:.3f}")
    show("Retained fraction 2,048", 1 - o2, "{:.3f}")
    show("Break-even step speedup", retained, "{:.3f}")
    show("Accelerator-hours change at ideal 2x, percent", 100 * (2 * retained / 2 - 1), "{:.1f}")
    show("Step speedup above which doubling saves accelerator-hours", 2 * retained, "{:.3f}")
    ms_speedup = MEGASCALE[1024] / MEGASCALE[512]
    show("MegaScale measured speedup 512 -> 1,024", ms_speedup, "{:.2f}")
    show("Accelerator-hours change at measured speedup, percent",
         100 * (2 * retained / ms_speedup - 1), "{:.1f}")
    show("Calendar time change at measured speedup, percent",
         100 * (retained / ms_speedup - 1), "{:.0f}")
    show("MegaScale scaling efficiency 256 -> 1,024, percent",
         100 * MEGASCALE[1024] / (4 * MEGASCALE[256]), "{:.0f}")
    for n, t in MEGASCALE.items():
        show(f"MegaScale tokens/s per GPU at {n}", t / n, "{:.0f}")
    mult = 2 ** KAPLAN_EXP
    show("Loss multiplier per compute doubling", mult, "{:.3f}")
    show("Loss reduction per doubling, percent", 100 * (1 - mult), "{:.1f}")
    loss = 1.0
    for k in range(1, 5):
        new = loss * mult
        show(f"Doubling {k}: absolute drop (normalized loss 1.0 at start)", loss - new, "{:.4f}")
        loss = new
    show("Compute multiplier for 4 doublings", 2 ** 4, "{:.0f}")
    show("Loss after 4 doublings, normalized", loss, "{:.3f}")
    show("Loss reduction after 16x compute, percent", 100 * (1 - loss), "{:.1f}")

    print("== Carbon")
    show("GPT-3 at Poland grid, t CO2", GPT3_MWH * 1e3 * POLAND_G_PER_KWH / 1e6, "{:,.0f}")
    show("GPT-3 at Quebec grid, t CO2", GPT3_MWH * 1e3 * QUEBEC_G_PER_KWH / 1e6, "{:.1f}")
    show("Poland / Quebec", POLAND_G_PER_KWH / QUEBEC_G_PER_KWH, "{:.0f}")

    print("== Where next")
    show("Test-time latency, s", TT_TOKENS * TT_MS_PER_TOKEN / 1e3, "{:.1f}")
    show("InferenceX per-GPU output ratio", IX_LOW_INTERACT / IX_HIGH_INTERACT, "{:.0f}")
    show("MoE resident / active bytes", MOE_RESIDENT_GB / MOE_ACTIVE_GB, "{:.1f}")
    show("Dense 400B / MoE active bytes", DENSE_400B_GB / MOE_ACTIVE_GB, "{:.1f}")
    show("Fleet energy saving, illustrative, percent",
         100 * LINK_SHARE_ILLUS * (1 - LINK_ENERGY_RATIO), "{:.0f}")
    show("Fleet saving ceiling at link share, percent", 100 * LINK_SHARE_ILLUS, "{:.0f}")
    show("Fleet energy saving at twice the link share, percent",
         100 * 2 * LINK_SHARE_ILLUS * (1 - LINK_ENERGY_RATIO), "{:.0f}")


main()
