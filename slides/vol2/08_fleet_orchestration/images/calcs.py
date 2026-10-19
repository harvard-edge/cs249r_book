#!/usr/bin/env python3
"""Every derived number in Lecture 8 (Fleet Orchestration), computed from sourced inputs.

Run from this folder: python3 calcs.py
Each printed line is "<slide label>: <value>". The slides and notes copy these values.
"FO" is the rendered chapter, Fleet Orchestration (Volume II, Chapter 8).
"""

import csv
import math
from pathlib import Path

SERIES = Path(__file__).resolve().parents[4] / "mlsysim" / "mlsysim" / "literature" / "data" / "series"
RATE = 2.0  # $ per GPU-hour, FO scenario rate (§8.1)


def show(label, value, fmt="{:,.2f}"):
    print(f"{label}: {fmt.format(value)}")


def rows(name):
    out = list(csv.DictReader((SERIES / name).open()))
    for r in out:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return out


print("== Hook and synthesis (FO Example 8.2, tables 8 and 9)")
pools = [400, 424, 200]  # A100-80, A100-40, V100-32 (table 8)
show("cluster GPUs", sum(pools), "{:,.0f}")
gain = 0.84 - 0.60  # cluster row, table 9
show("GPU-equivalents recovered (0.24 x 1,024)", gain * 1024, "{:,.0f}")
raw = round(gain * 1024) / 0.60
show("raw GPUs replaced (246 / 0.60)", raw, "{:,.0f}")
show("annual value of 410 GPUs at $2 ($M)", round(raw) * 8760 * RATE / 1e6, "{:.1f}")
show("table 9 pool rows, allocated GPUs (400 x 0.95 + 424 x 0.64)", 400 * 0.95 + 424 * 0.64, "{:,.0f}")
show("  as share of 1,024 (percent)", (400 * 0.95 + 424 * 0.64) / 1024 * 100, "{:.0f}")

print("== Recall (Fault Tolerance, Napkin Math 7.3): detect 60 s, restart 180 s, load 24.5 s, warm 120 s")
show("recovery minutes", (60 + 180 + 24.5 + 120) / 60, "{:.1f}")

print("== Kalos trace (Hu et al. 2024, AcmeTrace via MLSysIM)")
k = rows("acme_kalos_job_sizes.csv")
J = sum(r["jobs"] for r in k)
T = sum(r["gpu_seconds"] for r in k)
show("jobs", J, "{:,.0f}")
for lab, f in [("1 GPU", lambda n: n == 1), ("2 to 128 GPUs", lambda n: 1 < n < 256), ("256 or more", lambda n: n >= 256)]:
    j = sum(r["jobs"] for r in k if f(r["gpu_num"]))
    t = sum(r["gpu_seconds"] for r in k if f(r["gpu_num"]))
    show(f"{lab}: share of jobs (percent)", j / J * 100, "{:.1f}")
    show(f"{lab}: share of GPU time (percent)", t / T * 100, "{:.2f}")
    show(f"{lab}: mean GPU-hours per job", t / j / 3600, "{:,.2f}")
m1 = T / J
m2 = sum(r["jobs"] * (r["gpu_seconds"] / r["jobs"]) ** 2 for r in k) / J
cs = math.sqrt(m2 - m1 * m1) / m1
show("Cs lower bound (between gpu_num rows)", cs, "{:.1f}")
for st in ("completed", "failed", "canceled"):
    show(f"GPU time in {st} jobs (percent)", sum(r["gpu_seconds_" + st] for r in k) / T * 100, "{:.0f}")
show("GPU time in other states (percent)",
     (1 - sum(r["gpu_seconds_" + s] for r in k for s in ("completed", "failed", "canceled")) / T) * 100, "{:.0f}")

print("== M/G/1 (FO Napkin Math 8.1): wait / mean job = rho/(1-rho) x (1+Cs^2)/2")
mg1 = lambda rho, c: rho / (1 - rho) * (1 + c * c) / 2
show("rho 0.8, Cs 1", mg1(0.8, 1), "{:.0f}")
show("rho 0.8, Cs 3", mg1(0.8, 3), "{:.0f}")
show("variability factor at Kalos bound", (1 + cs * cs) / 2, "{:.0f}")
show("rho 0.8, Kalos Cs bound", mg1(0.8, cs), "{:.0f}")
show("rho 0.98, Cs 1", mg1(0.98, 1), "{:.0f}")
show("rho 0.60, Cs 1", mg1(0.60, 1), "{:.1f}")

print("== Gang deadlock cost (FO §8.4.3)")
show("10,000 GPUs idle, $ per hour", 10000 * RATE, "{:,.0f}")
show("900 GPUs held by one stuck gang, $ per hour", 900 * RATE, "{:,.0f}")

print("== DRF (FO table 1): cluster 8 GPU, 32 cores, 256 GB, 80 Gb/s")
cap = [8, 32, 256, 80]
A, B = [1, 4, 32, 10], [2, 2, 16, 20]
show("A dominant share, 4 replicas", max(4 * a / c for a, c in zip(A, cap)))
show("B dominant share, 2 workers", max(2 * b / c for b, c in zip(B, cap)))
show("B CPU share", 2 * 2 / 32, "{:.3f}")
show("B memory share", 2 * 16 / 256, "{:.3f}")

print("== Fragmentation (FO §8.4.2)")
show("free share, 19 of 64 (percent)", 19 / 64 * 100, "{:.0f}")
show("6-GPU jobs on 8-GPU nodes (percent usable)", 6 / 8 * 100, "{:.0f}")

print("== Bandwidth and placement (FO §8.4.2, Napkin Math 8.2; NCCL via MLSysIM)")
n = rows("nccl_allreduce_h100_node.csv")
show("NCCL in-place bus bandwidth at 1 GiB (GB/s)", [r for r in n if r["size_bytes"] == 2 ** 30][0]["ip_busbw_GBps"], "{:.0f}")
show("NVLink / NDR port ratio (450 / 50)", 450 / 50, "{:.0f}")
show("1 GB over NVLink (ms)", 1 / 450 * 1000, "{:.1f}")
show("1 GB over InfiniBand (ms)", 1 / 50 * 1000, "{:.0f}")
show("256 GPUs = TP x PP x DP", 8 * 4 * 8, "{:.0f}")

print("== Elastic decision (FO Napkin Math 8.3)")
e1, e2, e3 = 24 * 1.0, 20 * 3.2, 4 * 1.0 + (24 - 4 - 10 / 60) * 1.8
show("start small epochs", e1, "{:.0f}")
show("wait for 512 epochs", e2, "{:.0f}")
show("step up epochs", e3, "{:.1f}")
g1, g2, g3 = 128 * 24, 512 * 20, 128 * 4 + 256 * 20
show("start small GPU-hours", g1, "{:,.0f}")
show("wait GPU-hours", g2, "{:,.0f}")
show("step up GPU-hours (resize pause held at 256)", g3, "{:,.0f}")
show("epochs per 1,000 GPU-h, start small", e1 / g1 * 1000, "{:.1f}")
show("epochs per 1,000 GPU-h, wait", e2 / g2 * 1000, "{:.2f}")
show("epochs per 1,000 GPU-h, step up", e3 / g3 * 1000, "{:.1f}")
show("175B replica GPUs (TP 8 x 8 stages)", 8 * 8, "{:.0f}")
show("torchrun 4:16 nodes x 8 = workers min", 4 * 8, "{:.0f}")
show("torchrun max workers", 16 * 8, "{:.0f}")

print("== MLPerf Training v3.1 GPT-3 175B, NVIDIA Eos (via MLSysIM)")
mp = sorted((r["accelerators"], r["score_minutes"]) for r in rows("mlperf_training_llm_scaling.csv")
            if r["round"] == "v3.1" and r["benchmark"] == "gpt3" and r["submitter"] == "NVIDIA")
(g0, t0), (gn, tn) = mp[0], mp[-1]
show("GPU ratio", gn / g0, "{:.0f}")
show("speedup", t0 / tn, "{:.1f}")
show("share of ideal (percent)", t0 / tn / (gn / g0) * 100, "{:.0f}")

print("== Preemption (FO §8.8.1)")
over = 15 + 3 + 24.5 / 60 + 2
show("overhead minutes", over, "{:.1f}")
show("$ per event, 64 GPUs", 64 * over / 60 * RATE, "{:.0f}")
show("hours lost at 12 per day", 12 * over / 60, "{:.1f}")
show("GPU-hours lost at 12 per day", 12 * over / 60 * 64, "{:.0f}")

print("== Spot (FO §8.13 fallacy; Systems Perspective 8.4)")
eff = 0.35 * (2 + 0.75) / 2
show("effective cost / on-demand", eff, "{:.2f}")
show("savings (percent)", (1 - eff) * 100, "{:.0f}")
od = 512 * 336 * 2.0
hrs = 336 + 0.05 * 336 + 196 * 0.5
spot = 512 * hrs * 0.70
show("on-demand $", od, "{:,.0f}")
show("spot hours", hrs, "{:.1f}")
show("spot $", spot, "{:,.0f}")
show("spot savings (percent)", (1 - spot / od) * 100, "{:.1f}")
show("spot longer (percent)", (hrs / 336 - 1) * 100, "{:.1f}")

print("== Sharing (FO Napkin Math 8.4)")
show("one 7B model unused (percent)", (80 - 26) / 80 * 100, "{:.1f}")
show("two 7B models unused (percent)", (80 - 52) / 80 * 100, "{:.1f}")
ex, sh = 100 + 10 * 8, 50 + 10 * 8
show("GPUs exclusive", ex, "{:.0f}")
show("GPUs shared", sh, "{:.0f}")
show("fewer GPUs (percent)", (ex - sh) / ex * 100, "{:.1f}")
show("released value $ per year", (ex - sh) * 8760 * RATE, "{:,.0f}")

print("== Diurnal shift (FO Napkin Math 8.5)")
static_active = 150 + 618
dyn_active = 150 + 618 + 133
show("static active GPUs", static_active, "{:.0f}")
show("static utilization (percent)", static_active / 1024 * 100, "{:.0f}")
show("dynamic active GPUs", dyn_active, "{:.0f}")
show("dynamic utilization (percent)", dyn_active / 1024 * 100, "{:.0f}")
show("reclaimed average (283 - 150)", 400 - 117 - 150, "{:.0f}")
show("value $M per year", 133 * 8760 * RATE / 1e6, "{:.1f}")

print("== Quota (FO §8.11, Example 8.1)")
show("hoarding $ (500 GPUs x 3 weeks)", 500 * 21 * 24 * RATE, "{:,.0f}")
show("idle guarantee lent (500 x 0.4)", 500 * (1 - 0.6), "{:.0f}")

print("== Tiers (FO §8.12.1)")
show("active / allocated (percent)", 350 / 500 * 100, "{:.0f}")
show("productive / allocated (percent)", 280 / 500 * 100, "{:.0f}")
