#!/usr/bin/env python3
"""Every derived number in Lecture 7 (Fault Tolerance), computed from sourced inputs.

Run from this folder: python3 calcs.py
Each line prints the value and the slide (or note) that shows it. Inputs carry
their source as a comment; nothing on a slide is computed by hand.
"""

import math

# --- Inputs -------------------------------------------------------------------
GPU_MTTF_H = 50_000          # Fault Tolerance Sec. 7.2, planning constant
PCIE_MTTF_H = 200_000        # Sec. 7.2.2 cascade
NIC_MTTF_H = 150_000         # Sec. 7.2.2, per 8-GPU node
GPUS_PER_NODE = 8            # Sec. 7.2.2
CLUSTER = 10_000             # Sec. 7.2.2
REF_CLUSTER = 1_024          # Napkin Math 7.3, 7.4
GPU_AVAIL = 0.9999           # Sec. 7.2.1
LLAMA_DAYS, LLAMA_INTERRUPTS, LLAMA_GPUS = 54, 419, 16_384   # Dubey et al. 2024 (War Story 7.2)
LLAMA_GPU_PARTS = 268        # Dubey Table 5 via MLSysIM (148+72+19+17+6+6)
KOK = {8: 1144.8, 1024: 7.9}  # Kokolis et al. 2025 via MLSysIM, measured job MTTF (h)
PARAMS = 175e9               # reference job (Sec. 7.7.1)
CKPT_BYTES_PER_PARAM = 14    # 2 BF16 + 4 FP32 master + 8 Adam (Sec. 7.8.1)
OPT_BYTES = 12               # FP32 master + Adam moments (footnote 13, figure)
WRITE_BW = 100e9             # B/s, Sec. 7.8.2
DETECT, RESTART, WARMUP = 60, 180, 120   # s, Napkin Math 7.3
GPU_PRICE = 2.0              # $/GPU-hour, Sec. 7.7.1
REPLICA_A = 0.99             # Sec. 7.11.2
DCP_SHARD_BYTES = 12_587_643  # measured, dcp_trace.py
DCP_TIMES_MS = [0.00, 0.18, 2.14, 3.28, 24.38]  # measured mtimes, dcp_trace.py

out = []


def show(label, value, fmt):
    s = format(value, fmt)
    out.append((label, s))
    print(f"{label:<62} {s}")
    return value


# --- Failure clock ------------------------------------------------------------
show("hook: Llama 3 hours between interruptions", LLAMA_DAYS * 24 / LLAMA_INTERRUPTS, ".1f")
show("predict: GPU MTTF in years", GPU_MTTF_H / 8760, ".1f")
show("predict: GPU-only MTBF at 10,000 GPUs (h)", GPU_MTTF_H / CLUSTER, ".0f")
for n in (8, 64, 512, 1000, 4000, 10_000, 25_000):
    h = GPU_MTTF_H / n
    show(f"ladder: {n:,} GPUs MTBF (h)", h, ".1f")
    if h > 24:
        show(f"ladder: {n:,} GPUs MTBF (days)", h / 24, ".0f")
show("ladder note: all 10,000 up at 99.99% each (%)", 100 * GPU_AVAIL ** CLUSTER, ".1f")
lam_gpu = 1 / GPU_MTTF_H
lam_pcie = 1 / PCIE_MTTF_H
lam_nic = 1 / NIC_MTTF_H / GPUS_PER_NODE
show("estimate: GPU rate (1e-5 per h)", lam_gpu * 1e5, ".1f")
show("estimate: PCIe rate (1e-5 per h)", lam_pcie * 1e5, ".1f")
show("estimate: NIC share rate (1e-5 per h)", lam_nic * 1e5, ".3f")
lam_tot = lam_gpu + lam_pcie + lam_nic
show("estimate: per-GPU total (1e-5 per h)", lam_tot * 1e5, ".3f")
show("estimate: GPU-only cluster rate (per h)", lam_gpu * CLUSTER, ".2f")
show("estimate: cluster rate (per h)", lam_tot * CLUSTER, ".3f")
M_FULL_H = 1 / (lam_tot * CLUSTER)
show("estimate: full-system MTBF (h)", M_FULL_H, ".2f")
show("estimate: failures per day", 24 / M_FULL_H, ".1f")
show("measured: 8-GPU GPU-only over measured (x)", GPU_MTTF_H / 8 / KOK[8], ".1f")
show("measured: 1,024-GPU GPU-only line (h)", GPU_MTTF_H / 1024, ".1f")
show("measured: 1,024-GPU GPU-only over measured (x)", GPU_MTTF_H / 1024 / KOK[1024], ".1f")

# --- Fault types --------------------------------------------------------------
show("causes: GPU parts share (%)", 100 * LLAMA_GPU_PARTS / LLAMA_INTERRUPTS, ".0f")

# --- Checkpoints ---------------------------------------------------------------
ckpt = PARAMS * CKPT_BYTES_PER_PARAM
show("young-daly: checkpoint size (TB)", ckpt / 1e12, ".2f")
show("young-daly: optimizer share of checkpoint (%)", 100 * OPT_BYTES / CKPT_BYTES_PER_PARAM, ".0f")
delta = ckpt / WRITE_BW
show("young-daly: write time delta (s)", delta, ".1f")
M = 3.87 * 3600   # chapter rounds MTBF to 3.87 h before converting (Napkin Math 7.2)
show("your turn: M (s)", M, ".0f")
tau = math.sqrt(2 * delta * M)
show("your turn: tau* (s)", tau, ".0f")
show("your turn: tau* (min)", tau / 60, ".1f")
show("your turn: each term at optimum (%)", 100 * delta / tau, ".2f")
show("your turn: waste at optimum (%)", 100 * (delta / tau + tau / (2 * M)), ".1f")
show("your turn: hourly write term (%)", 100 * delta / 3600, ".1f")
show("your turn: hourly rework term (%)", 100 * 3600 / (2 * M), ".1f")
show("your turn: hourly waste (%)", 100 * (delta / 3600 + 3600 / (2 * M)), ".1f")
show("your turn: 2M (s)", 2 * M, ".0f")
show("your turn: sqrt(2)", math.sqrt(2), ".1f")
M100k = 1 / (lam_tot * 100_000) * 3600
show("open q: M at 100,000 GPUs (h)", M100k / 3600, ".2f")
show("open q: M at 100,000 GPUs (s)", M100k, ".0f")
show("open q: optimized waste at 100,000 GPUs (%)", 100 * math.sqrt(2 * delta / M100k), ".0f")
show("T/F: 1,000 GPUs, 5-min write, 50 h MTBF, tau* (min)", math.sqrt(2 * 5 * 3000), ".0f")
show("dcp: shard size (MB)", DCP_SHARD_BYTES / 1e6, ".1f")
show("dcp: last shard (ms)", DCP_TIMES_MS[3], ".1f")
show("dcp: metadata (ms)", DCP_TIMES_MS[4], ".1f")

# --- Recovery ------------------------------------------------------------------
load = ckpt / WRITE_BW
rec = DETECT + RESTART + load + WARMUP
show("budget: total recovery (min)", rec / 60, ".1f")
for n in (REF_CLUSTER, CLUSTER):
    per_day = 24 / (GPU_MTTF_H / n)
    mins = per_day * rec / 60
    show(f"budget: {n:,} GPUs GPU-only failures per day", per_day, ".2f")
    show(f"budget: {n:,} GPUs recovery minutes per day", mins, ".0f")
    show(f"budget: {n:,} GPUs share of day (%)", 100 * mins / 1440, ".1f")
    if n == CLUSTER:
        show("budget: 10,000 GPUs GPU-hours lost per day", mins / 60 * n, ",.0f")
spare_day = GPUS_PER_NODE * GPU_PRICE * 24
show("budget note: hot spare cost ($ per day)", spare_day, ",.0f")
per_min = (24 / M_FULL_H) * CLUSTER * GPU_PRICE / 60
show("budget note: value of 1 min per recovery ($ per day)", per_min, ",.0f")
show("budget note: break-even cut per recovery (s)", 60 * spare_day / per_min, ".0f")
show("straggler: throughput (%)", 100 * 1 / 2, ".0f")
show("straggler: idle cost, 1,024 GPUs at half throughput ($ per hour)",
     REF_CLUSTER * GPU_PRICE / 2, ",.0f")
show("elastic: devices kept after one node (%)", 100 * (REF_CLUSTER - 8) / REF_CLUSTER, ".1f")
show("elastic: ring of 3 of 4 (%)", 100 * 3 / 4, ".0f")

# --- Serving -------------------------------------------------------------------
for r in (1, 2, 3):
    down = (1 - REPLICA_A) ** r * 365 * 86400
    show(f"serving: {r} replica(s) availability (%)", 100 * (1 - (1 - REPLICA_A) ** r), ".4f")
    show(f"serving: {r} replica(s) downtime (s per year)", down, ",.1f")
show("serving: 1 replica downtime (days)", 0.01 * 365, ".2f")
show("serving: 2 replicas downtime (min)", 1e-4 * 365 * 1440, ".1f")
