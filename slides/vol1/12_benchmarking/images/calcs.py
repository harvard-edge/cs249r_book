#!/usr/bin/env python3
"""Every derived number in Lecture 12 (Benchmarking, Part 1), computed from
sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports from this file so figures and slides share
one computation. "Chapter" means Benchmarking in Volume I.

Run from this folder: python3 calcs.py
"""

import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def show(label, value):
    print(f"  {label}: {value}")


def head(title):
    print(f"\n[{title}]")


def r(x, nd=1):
    """Round half away from zero (Python's round() is half to even)."""
    q = 10 ** nd
    return math.floor(abs(x) * q + 0.5) / q * (1 if x >= 0 else -1)


def f(x, nd=1, commas=False):
    v = r(x, nd)
    return f"{v:,.{nd}f}" if commas else f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# Hook: datasheet against sustained efficiency (chapter Sec. 12.7 opening,
# illustrative): 10 TOPS at 0.5 W; 3 TOPS at 2 W under sustained load.
# ---------------------------------------------------------------------------
HOOK_TOPS_DS, HOOK_W_DS = 10.0, 0.5
HOOK_TOPS_SUS, HOOK_W_SUS = 3.0, 2.0
hook_eff_ds = HOOK_TOPS_DS / HOOK_W_DS
hook_eff_sus = HOOK_TOPS_SUS / HOOK_W_SUS
hook_gap = hook_eff_ds / hook_eff_sus
hook_tput_fall = HOOK_TOPS_DS / HOOK_TOPS_SUS
hook_power_rise = HOOK_W_SUS / HOOK_W_DS

# ---------------------------------------------------------------------------
# Tail: Dean and Barroso 2013, Table 1 (measured): one leaf p99 10 ms, all
# leaves p99 140 ms. Chapter Sec. 12.10 (illustrative): 15 ms mean against
# 150 to 200 ms p99. Users at p99 out of a million requests.
# ---------------------------------------------------------------------------
LEAF_P99, ALL_P99 = 10.0, 140.0
MEAN_ILL, P99_LO, P99_HI = 15.0, 150.0, 200.0
tail_ratio = ALL_P99 / LEAF_P99
gap_lo, gap_hi = P99_LO / MEAN_ILL, P99_HI / MEAN_ILL
REQUESTS = 1_000_000
p99_users = REQUESTS * (1 - 0.99)

# ---------------------------------------------------------------------------
# Fan-out estimate: chapter footnote 20; each shard slow 1 percent.
# ---------------------------------------------------------------------------
P_FAST = 0.99
all_fast_100 = P_FAST ** 100
slow_100 = 1 - all_fast_100
slow_10 = 1 - P_FAST ** 10

# ---------------------------------------------------------------------------
# Claim frame: MLPerf Inference rules, ResNet-50 row: "99% of FP32 (76.46%)"
# (mlcommons/inference_policies, inference_rules.adoc). Input only.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Host timer: timer_demo.py, run here (timer_demo_output.txt).
# ---------------------------------------------------------------------------
_tout = (HERE / "timer_demo_output.txt").read_text()
timer_host_ms = float(re.search(r"host timer:\s+([\d.]+) ms", _tout).group(1))
timer_sync_ms = float(re.search(r"with synchronize:\s+([\d.]+) ms", _tout).group(1))
timer_ratio = timer_sync_ms / timer_host_ms
MM_N = 8192
mm_flop = 2 * MM_N ** 3
mm_tflops = mm_flop / (timer_sync_ms * 1e-3) / 1e12

# ---------------------------------------------------------------------------
# Granularity inset: chapter Sec. 12.3 margin figure (illustrative): a 3x
# speedup of a 10 ms model stage inside a 50 ms request.
# ---------------------------------------------------------------------------
REQ_MS, STAGE_MS, STAGE_SPEEDUP = 50.0, 10.0, 3.0
gran_after = REQ_MS - STAGE_MS + STAGE_MS / STAGE_SPEEDUP
gran_saved = REQ_MS - gran_after
gran_speedup = REQ_MS / gran_after

# ---------------------------------------------------------------------------
# Roofline against MLPerf: Napkin Math 12.1. 8.2 GFLOP forward per image,
# training = 3x forward; H100 dense BF16 peak 989 TFLOP/s; the chapter's
# estimate of about 5,000 images/s per H100.
# MLPerf Training v4.0 closed, NVIDIA eos_ngc24.04_mxnet (8x H100) and
# eos_n8_ngc24.04_mxnet (64x H100), ResNet run times in minutes from the
# result_N.txt logs; published score = mean of the middle three runs times
# the scaling.json factor (a run that missed the target counts as slowest).
# ---------------------------------------------------------------------------
FWD_GFLOP, TRAIN_MULT, H100_PEAK_TFLOPS, CHAPTER_IPS = 8.2, 3, 989.0, 5000.0
train_gflop = FWD_GFLOP * TRAIN_MULT
ceiling_ips = H100_PEAK_TFLOPS * 1e12 / (train_gflop * 1e9)
achieved_tflops = CHAPTER_IPS * train_gflop * 1e9 / 1e12
mfu = achieved_tflops / H100_PEAK_TFLOPS
mfu_gap = ceiling_ips / CHAPTER_IPS

SCALE = 1.0042232277526395
EOS8 = [13.30620, 13.31780, 13.30162, 13.20308, 13.18598]
EOS64 = [2.50543, 2.50527, 2.50357, 2.50593, math.inf]   # result_4 missed target
IMAGENET_TRAIN, EPOCHS, GLOBAL_BATCH = 1_281_167, 35, 3200
_m8 = sorted(EOS8)[1:-1]
_m64 = sorted(EOS64)[1:-1]
score8 = sum(_m8) / 3 * SCALE
score64 = sum(_m64) / 3 * SCALE
ips8 = EPOCHS * IMAGENET_TRAIN / (score8 / SCALE * 60) / 8
ips64 = EPOCHS * IMAGENET_TRAIN / (score64 / SCALE * 60) / 64
gap_vs_ips8 = ceiling_ips / ips8
gap_vs_ips64 = ceiling_ips / ips64

# ---------------------------------------------------------------------------
# Profiler: profile_step.py, run here (profile_step_output.txt); Self CPU %
# per operator for one ResNet-50 training step on a CPU.
# ---------------------------------------------------------------------------
_prof = (HERE / "profile_step_output.txt").read_text()


def _pct(name):
    m = re.search(re.escape(name) + r"\s+([\d.]+)%", _prof)
    return float(m.group(1))


prof_conv = _pct("aten::_nnpack_spatial_convolution") + _pct("aten::convolution_backward")
prof_bn = _pct("aten::native_batch_norm") + _pct("aten::native_batch_norm_backward")
prof_rest = 100 - prof_conv - prof_bn
prof_total_s = float(re.search(r"Self CPU time total:\s+([\d.]+)s", _prof).group(1))

# ---------------------------------------------------------------------------
# Edge board (Lighthouse 12.2, Example 12.3; illustrative composite):
# host INT8 15 ms at 120 mW, accelerator INT8 2 ms at 500 mW; end to end
# 18 ms against 6 ms. Notebook case: 8 ms preprocessing + 10 ms inference,
# a 5x faster model.
# ---------------------------------------------------------------------------
HOST_FP32, HOST_INT8, ACC_INT8 = 50.0, 15.0, 2.0
E2E_HOST, E2E_ACC = 18.0, 6.0
P_HOST_MW, P_ACC_MW = 120.0, 500.0
edge_isolated = HOST_INT8 / ACC_INT8
edge_e2e = E2E_HOST / E2E_ACC
edge_e_host = P_HOST_MW * HOST_INT8 / 1000      # mJ
edge_e_acc = P_ACC_MW * ACC_INT8 / 1000          # mJ
PREP, INFER, MODEL_SPEEDUP = 8.0, 10.0, 5.0
nb_after = PREP + INFER / MODEL_SPEEDUP
nb_speedup = (PREP + INFER) / nb_after
nb_limit = (PREP + INFER) / PREP

# ---------------------------------------------------------------------------
# Software stack: Lee et al. 2019, Tables 4 and 5 (ms, mean of 100 runs).
# ---------------------------------------------------------------------------
XS_TFLITE_GPU, XS_METAL, XS_COREML = 2.3, 4.1, 7.1
PX3_GPU, PX3_CPU = 12.5, 51.7
lee_xs = XS_COREML / XS_TFLITE_GPU
lee_px3 = PX3_CPU / PX3_GPU

# ---------------------------------------------------------------------------
# Time to quality self-check (chapter): 1.8x faster steps, 2.5x the steps.
# ---------------------------------------------------------------------------
STEP_SPEEDUP, STEP_MULT = 1.8, 2.5
ttq_slowdown = STEP_MULT / STEP_SPEEDUP

# ---------------------------------------------------------------------------
# Strong scaling (same MLPerf logs as above).
# ---------------------------------------------------------------------------
scale_speed = score8 / score64
scale_ideal = score8 / 8
scale_eff = score8 / (8 * score64)
per_gpu_batch8 = GLOBAL_BATCH / 8
per_gpu_batch64 = GLOBAL_BATCH / 64

# ---------------------------------------------------------------------------
# MLPerf Inference v5.0 closed, ResNet-50 (summary_results.json):
# RTX 4090 (5.0-0032) SingleStream p90 0.281736 ms, Offline 44,411.1/s;
# Cisco (5.0-0010) Offline 715,761, Server 652,002; Quanta (5.0-0065)
# Offline 712,859, Server 584,161.
# ---------------------------------------------------------------------------
RTX_SS_MS, RTX_OFF = 0.281736, 44411.1
CISCO_OFF, CISCO_SRV, QUANTA_OFF, QUANTA_SRV = 715761, 652002, 712859, 584161
rtx_serial = 1000 / RTX_SS_MS
rtx_ratio = RTX_OFF / rtx_serial
cisco_drop = 1 - CISCO_SRV / CISCO_OFF
quanta_drop = 1 - QUANTA_SRV / QUANTA_OFF
srv_spread = CISCO_SRV / QUANTA_SRV - 1

# ---------------------------------------------------------------------------
# The Server log: mlperf_log_summary.txt of 5.0-0010 (Cisco C885A, 8x H100
# SXM), closed/Cisco/results/C885A_H100_SXMx8_TRT/resnet50/Server/
# performance/run_1 (mlcommons/inference_results_v5.0); a copy is in this
# folder.
# ---------------------------------------------------------------------------
_log = (HERE / "mlperf_v50_cisco_resnet50_server_summary.txt").read_text()


def _logv(key):
    return float(re.search(re.escape(key) + r"\s*:\s*([\d.]+)", _log).group(1))


log_rate = _logv("Completed samples per second")
log_p50_ms = _logv("50.00 percentile latency (ns)") / 1e6
log_p99_ms = _logv("99.00 percentile latency (ns)") / 1e6
log_p999_ms = _logv("99.90 percentile latency (ns)") / 1e6
log_bound_ms = _logv("target_latency (ns)") / 1e6
log_min_s = _logv("min_duration (ms)") / 1e3
log_p99_share = log_p99_ms / log_bound_ms

# ---------------------------------------------------------------------------
# Boundary: Elsworth et al. 2025 via chapter Sec. 12.7.1: 0.10 Wh active
# accelerators against 0.24 Wh whole serving system; split 58/25/17 percent.
# PUE range (chapter Sec. 12.7.1, Barroso et al. 2019): about 1.1 to over 2.0.
# ---------------------------------------------------------------------------
GEM_ACTIVE, GEM_FULL = 0.10, 0.24
gem_ratio = GEM_FULL / GEM_ACTIVE

# ---------------------------------------------------------------------------
# Throttling: Lee et al. 2019, Figure 8, read from the plot (approximate):
# iPhone Xs CPU 20 -> 26 ms, GPU 2.4 -> 3.9 ms, Pixel 3 GPU 12.8 ms flat.
# Tummalapalli et al. 2026 (arXiv 2603.23640): 40.49, 24.96, 23.67 tokens/s.
# ---------------------------------------------------------------------------
XS_CPU0, XS_CPU1, XS_GPU0, XS_GPU1 = 20.0, 26.0, 2.4, 3.9
TOK_PEAK, TOK_WARM, TOK_HOT = 40.49, 24.96, 23.67
thr_cpu = XS_CPU1 / XS_CPU0 - 1
thr_gpu = XS_GPU1 / XS_GPU0 - 1
thr_llm = 1 - TOK_HOT / TOK_PEAK

# ---------------------------------------------------------------------------
# MLPerf Tiny v1.2 closed, visual wake words (summary.csv): Syntiant
# NDP9120 at 30.7 MHz 78.6 inf/s, 71.711 uJ; at 98.7 MHz 243.883 inf/s,
# 97.164 uJ; NUCLEO-U575ZI-Q 1354.685, NUCLEO-H7A3ZI-Q 2965.461,
# NUCLEO-L4R5ZI 3894.566 uJ.
# ---------------------------------------------------------------------------
SYN_LO_IPS, SYN_LO_UJ, SYN_HI_IPS, SYN_HI_UJ = 78.6, 71.711, 243.883, 97.164
U575, H7A3, L4R5 = 1354.685, 2965.461, 3894.566
tiny_speed = SYN_HI_IPS / SYN_LO_IPS
tiny_energy = SYN_HI_UJ / SYN_LO_UJ
tiny_worst = L4R5 / SYN_LO_UJ

# ---------------------------------------------------------------------------
# Duty cycle: chapter Sec. 12.7.2: one inference per second, 8.3 ms at
# 0.4 W, idle 50 mW.
# ---------------------------------------------------------------------------
T_ACTIVE_MS, PERIOD_MS, P_ACTIVE_MW, P_IDLE_MW = 8.3, 1000.0, 400.0, 50.0
duty = T_ACTIVE_MS / PERIOD_MS
duty_active = duty * P_ACTIVE_MW
duty_idle = (1 - duty) * P_IDLE_MW
duty_avg = duty_active + duty_idle
duty_idle_share = duty_idle / duty_avg
duty_breakeven = P_IDLE_MW / (P_ACTIVE_MW + P_IDLE_MW)
duty_halved_saving = duty_active / 2
duty_halved_gain = duty_halved_saving / duty_avg


def main():
    head("Hook (slide 2)")
    show("datasheet TOPS/W", f(hook_eff_ds, 0))
    show("sustained TOPS/W", f(hook_eff_sus, 1))
    show("gap (x)", f(hook_gap, 1))
    show("throughput fell (x)", f(hook_tput_fall, 1))
    show("power rose (x)", f(hook_power_rise, 0))

    head("Tail (One server's p99 ...)")
    show("all-leaves p99 / one-leaf p99 (x)", f(tail_ratio, 0))
    show("illustrative p99 / mean (x)", f"{f(gap_lo, 0)} to {f(gap_hi, 1)}")
    show("users at p99 per million requests", f"{p99_users:,.0f}")

    head("Fan-out estimate")
    show("0.99^100", f(all_fast_100, 3))
    show("slow at 100 shards (%)", f(100 * slow_100, 0))
    show("slow at 10 shards (%)", f(100 * slow_10, 1))
    show("all fast at 100 shards (%)", f(100 * all_fast_100, 1))

    head("Host timer (run here)")
    show("host timer (ms)", f(timer_host_ms, 3))
    show("with synchronize (ms)", f(timer_sync_ms, 3))
    show("ratio (x)", f(timer_ratio, 0))
    show("8192 matmul TFLOP/s achieved", f(mm_tflops, 1))

    head("Granularity inset")
    show("after (ms)", f(gran_after, 1))
    show("saved (ms)", f(gran_saved, 1))
    show("speedup (x)", f(gran_speedup, 2))

    head("Roofline against MLPerf")
    show("GFLOP per training image", f(train_gflop, 1))
    show("ceiling images/s", f"{r(ceiling_ips, -2):,.0f}")
    show("achieved TFLOP/s at 5,000", f(achieved_tflops, 0))
    show("MFU (%)", f(100 * mfu, 1))
    show("gap (x)", f(mfu_gap, 1))
    show("MLPerf per-H100 images/s, 8 GPUs", f"{r(ips8, -1):,.0f}")
    show("MLPerf per-H100 images/s, 64 GPUs", f"{r(ips64, -1):,.0f}")
    show("gap against the logs (x)", f"{f(gap_vs_ips8, 1)} to {f(gap_vs_ips64, 1)}")

    head("Profiler (run here)")
    show("convolution fwd+bwd self CPU (%)", f(prof_conv, 1))
    show("batch norm fwd+bwd self CPU (%)", f(prof_bn, 1))
    show("everything else (%)", f(prof_rest, 1))
    show("step self CPU total (s)", f(prof_total_s, 2))

    head("Edge board")
    show("isolated speedup (x)", f(edge_isolated, 1))
    show("end-to-end speedup (x)", f(edge_e2e, 0))
    show("host energy (mJ)", f(edge_e_host, 1))
    show("accelerator energy (mJ)", f(edge_e_acc, 0))
    show("notebook: after (ms)", f(nb_after, 0))
    show("notebook: speedup (x)", f(nb_speedup, 1))
    show("notebook: limit (x)", f(nb_limit, 2))

    head("Software stack")
    show("iPhone Xs spread (x)", f(lee_xs, 1))
    show("Pixel 3 spread (x)", f(lee_px3, 1))

    head("Time to quality")
    show("self-check slowdown (x)", f(ttq_slowdown, 2))

    head("Scaling")
    show("8-GPU score (min)", f(score8, 2))
    show("64-GPU score (min)", f(score64, 2))
    show("speedup (x)", f(scale_speed, 1))
    show("ideal 64-GPU time (min)", f(scale_ideal, 2))
    show("efficiency (%)", f(100 * scale_eff, 0))
    show("per-GPU images/s, 8 -> 64", f"{r(ips8, -1):,.0f} -> {r(ips64, -1):,.0f}")
    show("images per GPU per step, 8 -> 64", f"{per_gpu_batch8:.0f} -> {per_gpu_batch64:.0f}")

    head("RTX 4090 scenarios")
    show("one-at-a-time rate (/s)", f"{r(rtx_serial, -1):,.0f}")
    show("Offline / serial (x)", f(rtx_ratio, 1))
    show("Cisco Server below Offline (%)", f(100 * cisco_drop, 0))
    show("Quanta Server below Offline (%)", f(100 * quanta_drop, 0))
    show("Cisco over Quanta, Server (%)", f(100 * srv_spread, 1))

    head("Server log (Cisco 5.0-0010)")
    show("completed samples/s", f(log_rate, 2, commas=True))
    show("p50 (ms)", f(log_p50_ms, 2))
    show("p99 (ms)", f(log_p99_ms, 2))
    show("p99.9 (ms)", f(log_p999_ms, 2))
    show("bound (ms)", f(log_bound_ms, 0))
    show("p99 share of bound (%)", f(100 * log_p99_share, 0))
    show("minimum duration (s)", f(log_min_s, 0))

    head("Boundary")
    show("whole system / active (x)", f(gem_ratio, 1))

    head("Throttling")
    show("iPhone CPU rise (%)", f(100 * thr_cpu, 0))
    show("iPhone GPU rise (%)", f(100 * thr_gpu, 0))
    show("LLM loss hot (%)", f(100 * thr_llm, 1))

    head("MLPerf Tiny")
    show("Syntiant faster at 98.7 MHz (x)", f(tiny_speed, 1))
    show("Syntiant energy ratio (x)", f(tiny_energy, 2))
    show("STM32L4R5 / Syntiant low (x)", f(tiny_worst, 0))

    head("Duty cycle")
    show("duty (%)", f(100 * duty, 2))
    show("active share (mW)", f(duty_active, 1))
    show("idle share (mW)", f(duty_idle, 1))
    show("average (mW)", f(duty_avg, 1))
    show("idle share of drain (%)", f(100 * duty_idle_share, 0))
    show("break-even duty (%)", f(100 * duty_breakeven, 0))
    show("halving inference energy saves (mW)", f(duty_halved_saving, 2))
    show("battery gain (%)", f(100 * duty_halved_gain, 0))


if __name__ == "__main__":
    main()
