#!/usr/bin/env python3
"""Every derived number in Volume II, Lecture 1 (Introduction), computed from
sourced inputs. Run from this folder: python3 calcs.py

Each printed line is "<slide label>: <value>", and the slide or note carries
the printed value. Inputs name their source in a comment.
"""

import math

rows = []


def show(label, value):
    rows.append((label, value))
    print(f"{label}: {value}")


# --- Hook and Napkin Math 1.1 (Introduction, Sec. 1.5.5) -------------------
ring_gb = 700.0          # GB sent per endpoint per step, FP16 ring (Napkin Math 1.1)
ib_gbps = 200.0          # Gb/s InfiniBand link (Napkin Math 1.1)
eth_gbps = 100.0         # Gb/s Ethernet link (Napkin Math 1.1)
compute_s = 1.2          # s local compute per step, scenario input (Napkin Math 1.1)
pj_per_bit = 35.0        # pJ per bit on the InfiniBand fabric (Napkin Math 1.1)
steps = 1_000_000        # steps in the run (Napkin Math 1.1)

ib_gbs = ib_gbps / 8
eth_gbs = eth_gbps / 8
t_ib = ring_gb / ib_gbs
t_eth = ring_gb / eth_gbs
show("hook: InfiniBand GB/s", f"{ib_gbs:g}")
show("hook: transfer over InfiniBand (s)", f"{t_ib:.0f}")
show("hook: step over InfiniBand (s)", f"{compute_s + t_ib:.1f}")
show("hook: compute fraction over InfiniBand", f"{compute_s / (compute_s + t_ib):.1%}")
show("napkin: Ethernet GB/s", f"{eth_gbs:g}")
show("napkin: transfer over Ethernet (s)", f"{t_eth:.0f}")
show("napkin: step over Ethernet (s)", f"{compute_s + t_eth:.1f}")
show("napkin: compute fraction over Ethernet", f"{compute_s / (compute_s + t_eth):.1%}")
show("napkin: exposed share over InfiniBand", f"{t_ib / (compute_s + t_ib):.0%}")
energy_j = ring_gb * 1e9 * 8 * pj_per_bit * 1e-12
show("napkin: energy per worker per step (J)", f"{energy_j:.0f}")
show("napkin: energy per worker over the run (MJ)", f"{energy_j * steps / 1e6:.0f}")

# --- Failure clock (Introduction, Sec. 1.1; Lecture 0's failure slide) ------
mttf_h = 50_000          # assumed per-GPU MTTF, hours (Sec. 1.1)
gpus = 16_384            # H100s in the Llama 3 405B run (Dubey et al. 2024), the
                         # job Lecture 0's failure-clock vote used
mtbf = mttf_h / gpus
show("recall: system MTBF at 16,384 GPUs (h)", f"{mtbf:.2f}")
show("recall: interruptions per day", f"{24 / mtbf:.1f}")

# --- Scaling law (Sec. 1.3.3; Hoffmann et al. 2022 fit via MLSysIM) ----------
alpha = 0.34             # parameter exponent of the Chinchilla parametric fit
show("powerlaw: 2^(1/alpha) at alpha = 0.34", f"{2 ** (1 / alpha):.1f}")
show("powerlaw note: 2^(1/0.1)", f"{2 ** (1 / 0.1):,.0f}")
gopher_n, gopher_d = 63e9, 1.4e12   # Sec. 1.3.2 projection
show("isoflop note: tokens per parameter at the Gopher budget", f"{gopher_d / gopher_n:.0f}")
show("isoflop note: 6ND at the Gopher budget (FLOPs)", f"{6 * gopher_n * gopher_d:.1e}")

# --- GPT-3 allocation (Sec. 1.3.2; Brown et al. 2020, Table 2.1) -------------
n_gpt3, d_gpt3 = 175e9, 300e9
tpp = 20.0               # compute-optimal tokens per parameter (Sec. 1.3.2)
c_gpt3 = 6 * n_gpt3 * d_gpt3
show("estimate: GPT-3 budget C = 6ND (FLOPs)", f"{c_gpt3:.2e}")
show("estimate: GPT-3 tokens per parameter", f"{d_gpt3 / n_gpt3:.1f}")
show("estimate note: how far below 20 (times)", f"{tpp / (d_gpt3 / n_gpt3):.0f}")
n_opt = math.sqrt(c_gpt3 / (6 * tpp))
show("estimate: optimal N at 20 tokens per parameter (B)", f"{n_opt / 1e9:.0f}")
show("estimate: optimal D at 20 tokens per parameter (T)", f"{tpp * n_opt / 1e12:.1f}")

# --- Bandwidth step (Sec. 1.2) ------------------------------------------------
nvlink, ndr = 450.0, 50.0   # GB/s per accelerator per direction (Sec. 1.2)
show("node: NVLink over NDR", f"{nvlink / ndr:.0f}x")
show("node: NDR 400 Gb/s in GB/s", f"{400 / 8:.0f}")

nvlink_over_hdr = nvlink / ib_gbs   # hook link: 200 Gb/s InfiniBand HDR (25 GB/s)
show("synthesis: NVLink over the hook's HDR link", f"{nvlink_over_hdr:.0f}x")

# --- Ring AllReduce traffic (Sec. 1.2.3) -------------------------------------
params = 175e9
grad_fp32_gb = params * 4 / 1e9
show("ring: FP32 gradient (GB)", f"{grad_fp32_gb:.0f}")
show("ring: FP32 ring traffic, large N (GB)", f"{2 * grad_fp32_gb:,.0f}")
show("ring: FP16 ring traffic, large N (GB)", f"{2 * params * 2 / 1e9:,.0f}")
for n in (512, 1024):
    show(f"ring note: 2(N-1)/N at N = {n}", f"{2 * (n - 1) / n:.3f}")

# --- Communication intensity (Sec. 1.2.4) ------------------------------------
tokens = 8192
flops_worker = 6 * params * tokens
ci = ring_gb * 1e9 / flops_worker
a100_peak = 312e12       # A100 16-bit peak FLOP/s (Sec. 1.1, footnote 6)
balance = ib_gbs * 1e9 / a100_peak
show("ci: FLOPs per worker per step", f"{flops_worker:.1e}")
show("ci: 175B worker CI (B/FLOP)", f"{ci:.1e}")
show("ci: FLOPs per byte", f"{1 / ci:,.0f}")
show("ci: fabric balance A100 + 200 Gb/s (B/FLOP)", f"{balance:.1e}")
show("ci note: compute time at peak (s)", f"{flops_worker / a100_peak:.1f}")
alpha_ci = 10e6 / 1e14   # Workload Alpha, Sec. 1.2 self-check 6
beta_ci = 1e9 / 1e13     # Workload Beta
show("estimate: Alpha CI (B/FLOP)", f"{alpha_ci:.0e}")
show("estimate: Beta CI (B/FLOP)", f"{beta_ci:.0e}")
show("estimate: Beta over Alpha", f"{beta_ci / alpha_ci:,.0f}x")

# --- Llama 3 405B (Dubey et al. 2024, Table 4) -------------------------------
t8k, t16k = 430.0, 400.0  # BF16 TFLOP/s per GPU at 8,192 and 16,384 GPUs
show("llama: per-GPU throughput drop", f"{1 - t16k / t8k:.0%}")
show("llama: GPUs ratio", f"{16384 / 8192:.0f}x")
show("llama: tokens per GPU per step at 8,192 (K)", f"{16e6 / 8192 / 1e3:.1f}")
show("llama: tokens per GPU per step at 16,384 (K)", f"{16e6 / 16384 / 1e3:.2f}")

# --- Fleet Law timeline (illustrative units, figure) -------------------------
t_n, t_2n = 10.0, 8.0
show("fleetlaw: ideal step at 2N (units)", f"{t_n / 2:g}")
show("fleetlaw: speedup N to 2N", f"{t_n / t_2n:.2f}")
show("fleetlaw: efficiency of the doubling", f"{(t_n / t_2n) / 2:.2f}")

# --- Diagnose (Sec. 1.5 self-check 3) -----------------------------------------
tc, tm, ts = 0.8, 1.0, 0.2
step = tc + tm + ts
show("diagnose: step (s)", f"{step:.1f}")
show("diagnose: compute fraction f", f"{tc / step:.2f}")
show("diagnose: step with 2x compute (s)", f"{tc / 2 + tm + ts:.1f}")
show("diagnose: most option A saves (s)", f"{tc / 2:.1f}")

# --- Fallacies (Sec. 1.8) -----------------------------------------------------
fast = compute_s / 2 + t_ib
show("fallacy: step with 2x accelerators (s)", f"{fast:.1f}")
show("fallacy: step reduction", f"{1 - fast / (compute_s + t_ib):.1%}")
show("pitfall: 3x times (1 - 0.40)", f"{3 * (1 - 0.40):.1f}x")

# --- Reliability gap (Sec. 1.4.1) ---------------------------------------------
show("aN: 0.999^1,000", f"{0.999 ** 1000:.1%}")
show("aN: 0.999^10,000", f"{0.999 ** 10000:.4%}")
show("aN: 0.9999^10,000", f"{0.9999 ** 10000:.1%}")
show("aN note: minutes down per year at 99.99%", f"{(1 - 0.9999) * 365 * 24 * 60:.1f}")

# --- OPT-175B and Llama 3 cadence ---------------------------------------------
opt_restarts = 35 + 70   # lower bounds, Zhang et al. 2022, Sec. 2.5
opt_days = 60            # "2 months", taken as 60 days
show("opt: restarts (lower bound)", f"{opt_restarts}")
show("opt: hours between restarts (upper bound)", f"{opt_days * 24 / opt_restarts:.0f}")
show("opt: hosts in 992 GPUs at 8 per host", f"{992 / 8:.0f}")
show("llama: hours between interruptions", f"{54 * 24 / 419:.1f}")  # Dubey et al. 2024
