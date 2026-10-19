#!/usr/bin/env python3
"""Every derived number in Lecture 13 (Model Serving, Part 1), computed from
sourced inputs.

One block per slide, in deck order. Inputs carry their source as a comment;
each printed line is a value the slide or its note shows, under the label
used there. make_figs.py imports from this file so figures and slides share
one computation. "MS" means the chapter Model Serving in Volume I.

The chapter's ResNet-50 numbers are its modeled reference profile on the
H100 (mlsysim.engine.serving_profile.resnet50_h100_profile, the LEGO source
of MS tables 8 and 14): fixed cost 3.4952836 ms and 0.0328615 ms per image
for eager FP16, so batch 1 takes 3.5281451 ms. The chapter prints this as
3.5 ms in prose and 3.53 ms in its tables.

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
    return math.floor(abs(x) * q + 0.5 + 1e-9) / q * (1 if x >= 0 else -1)


def f(x, nd=1, commas=False):
    v = r(x, nd)
    return f"{v:,.{nd}f}" if commas else f"{v:.{nd}f}"


def sig(x, n=2):
    """Round to n significant figures."""
    if x == 0:
        return 0
    return r(x, n - 1 - int(math.floor(math.log10(abs(x)))))


# ---------------------------------------------------------------------------
# The chapter's reference profile (mlsysim serving_profile, eager FP16).
# ---------------------------------------------------------------------------
FIXED_MS = 3.4952835820895523      # eager_fp16(0)
PER_IMG_MS = 0.03286147623862488   # per_image_fp16
S_MS = FIXED_MS + PER_IMG_MS       # eager_fp16(1) = 3.528 ms
S_PRINT = 3.5                      # MS prose value (Example 13.1, table 3)
AFF_FIXED, AFF_PER = 3.50, 0.033   # MS Sec. 13.5.4: affine model as printed


def batch_ms(b):
    return FIXED_MS + b * PER_IMG_MS


# ---------------------------------------------------------------------------
# Hook: MS Example 13.1. $3.92 and $0.52 per million requests.
# ---------------------------------------------------------------------------
HOOK_B1_USD, HOOK_B8_USD = 3.92, 0.52
HOOK_B1_MS, HOOK_B8_MS = 3.5, 3.8
hook_ratio = HOOK_B1_USD / HOOK_B8_USD
# Panel fix L7: the extra time of a batch of 8 comes from the example's own
# rates (283 and 2,129 requests/s), not from the rounded 3.5 and 3.8 ms labels.
HOOK_B1_RATE, HOOK_B8_RATE = 283, 2129   # requests/s, MS Example 13.1
hook_b1_ms = 1 / HOOK_B1_RATE * 1000
hook_b8_ms = 8 / HOOK_B8_RATE * 1000
hook_extra_ms = hook_b8_ms - hook_b1_ms
hook_extra_pct = hook_extra_ms / hook_b8_ms * 100

# ---------------------------------------------------------------------------
# Recall from Lecture 12: MLPerf Inference v5.0, 5.0-0032, RTX 4090,
# SingleStream p90 0.2817 ms, Offline 44,411 samples/s.
# ---------------------------------------------------------------------------
SS_MS, OFF_4090 = 0.2817, 44411
ss_rate = 1000 / SS_MS
off_over_ss = OFF_4090 / ss_rate

# ---------------------------------------------------------------------------
# Static against dynamic: MS Sec. 13.1.1. 10,000 photos, 3.5 ms each at
# batch 1; 89,500 images/s for a compiled INT8 engine (MLPerf, as cited).
# ---------------------------------------------------------------------------
PHOTOS, INT8_IPS = 10_000, 89_500
static_b1_s = PHOTOS * S_PRINT / 1000
static_int8_s = PHOTOS / INT8_IPS
static_ratio = static_b1_s / static_int8_s

# ---------------------------------------------------------------------------
# Request budget: MS table 3 (host phases assumed, forward pass modeled).
# ---------------------------------------------------------------------------
DECODE, RESIZE, NORM, COPY, FWD, POST = 3.0, 1.0, 0.5, 0.5, 3.5, 0.1
budget_total = DECODE + RESIZE + NORM + COPY + FWD + POST
pre_ms = DECODE + RESIZE + NORM
gpu_idle_ms = pre_ms + COPY
model_share = FWD / budget_total * 100
pre_share = pre_ms / budget_total * 100

# ---------------------------------------------------------------------------
# DLRM: MS Lighthouse 13.1, table 4 (scenario values).
# ---------------------------------------------------------------------------
DL_PARSE, DL_EMB, DL_MLP, DL_RANK = 0.5, 6.0, 1.5, 1.0
dlrm_total = DL_PARSE + DL_EMB + DL_MLP + DL_RANK
dlrm_mlp_share = DL_MLP / dlrm_total * 100

# ---------------------------------------------------------------------------
# Pipelining: MS Sec. 13.3.5. The host stage is preprocessing (4.5 ms); the
# copy runs on the PCIe lane, so a request leaves every max(4.5, 3.5) ms.
# ---------------------------------------------------------------------------
host_stage = pre_ms
pipe_duty = FWD / max(host_stage, FWD) * 100
serial_duty = model_share

# ---------------------------------------------------------------------------
# Predict 10x: MS Systems Perspective 13.2. Richins et al., arXiv 2007.10571,
# Fig. 6: ingestion 18.8, detection 74.8, brokers 126.1, identification
# 131.5 ms (paper's percentages 5.3, 21.3, 35.9, 37.4).
# ---------------------------------------------------------------------------
fast_total = budget_total - FWD + FWD / 10
fast_speedup = budget_total / fast_total
naive_ms = budget_total / 10
R_ING, R_DET, R_BRK, R_ID = 18.8, 74.8, 126.1, 131.5
rich_total = R_ING + R_DET + R_BRK + R_ID
rich_ai = (R_DET + R_ID) / rich_total * 100
rich_brk = R_BRK / rich_total * 100

# ---------------------------------------------------------------------------
# perf_analyzer quick start output (Triton perf_analyzer docs, quick_start.md):
# client avg 708 usec; server avg request 382 usec = overhead 41 + queue 41 +
# compute input 26 + compute infer 257 + compute output 16; 1407.84 infer/s.
# ---------------------------------------------------------------------------
PA_CLIENT, PA_SERVER = 708, 382
PA_PARTS = {"overhead": 41, "queue": 41, "compute input": 26,
            "compute infer": 257, "compute output": 16}
PA_TPUT = 1407.84
pa_outside = PA_CLIENT - PA_SERVER
pa_infer_share = PA_PARTS["compute infer"] / PA_CLIENT * 100
pa_parts_sum = sum(PA_PARTS.values())
pa_little = PA_TPUT * PA_CLIENT * 1e-6

# ---------------------------------------------------------------------------
# Little's law: MS Napkin Math 13.2.
# ---------------------------------------------------------------------------
LAM, W_S, GPU_BATCH = 1000, 0.05, 32
little_n = LAM * W_S
little_wait = little_n - GPU_BATCH
little_conv = 50 / 0.010

# ---------------------------------------------------------------------------
# Knee: MS eqs. 4, 5; table 7 uses the profile's service time.
# ---------------------------------------------------------------------------
P99_K = 4.6               # MS eq. 5 (ln 100 rounded)
ln100 = math.log(100)


def mean_w(rho, s=S_MS):
    return s / (1 - rho)


def p99_w(rho, s=S_MS):
    return P99_K * s / (1 - rho)


KNEE_RHOS = (0.5, 0.7, 0.8, 0.9, 0.95)
knee = {rho: r(mean_w(rho), 1) for rho in KNEE_RHOS}
knee_p99_70 = r(p99_w(0.7), 1)

# ---------------------------------------------------------------------------
# Estimate: MS Napkin Math 13.3 with the profile's 3.528 ms service time.
# Scenario: 5,000 req/s peak, 50 ms queue + GPU p99, 30 percent headroom.
# ---------------------------------------------------------------------------
PEAK, P99_BUDGET, HEADROOM = 5000, 50.0, 1.3
est_mu1 = 1000 / S_MS
est_rho = 1 - P99_K * S_MS / P99_BUDGET
est_rate = PEAK / est_rho
est_raw = est_rate / est_mu1
est_ceil = math.ceil(est_raw)
est_final = math.ceil(est_ceil * HEADROOM)
est_fail_util = PEAK / (est_mu1 * (est_final - 1)) * 100
est_naive = PEAK / est_mu1
# The slide before the fix: 5,000 / 0.68
old_rate = PEAK / 0.68
old_rate_353 = PEAK / (1 - P99_K * 3.53 / P99_BUDGET)
rho_at_35 = 1 - P99_K * S_PRINT / P99_BUDGET

# ---------------------------------------------------------------------------
# MLPerf Inference v5.0 Server against Offline (summary_results.json).
# ---------------------------------------------------------------------------
A_OFF, A_SRV, B_OFF, B_SRV = 715761, 652002, 712859, 584161
a_kept = A_SRV / A_OFF * 100
b_kept = B_SRV / B_OFF * 100

# ---------------------------------------------------------------------------
# Faster-model vote: MS Sec. 13.11. 80 percent load; 1.8 ms service.
# ---------------------------------------------------------------------------
S_NEW = 1.8
vote_lam = 0.8 / S_MS * 1000
vote_rho_new = vote_lam * S_NEW / 1000
vote_old = mean_w(0.8)
vote_new = S_NEW / (1 - vote_rho_new)
vote_wait_old = vote_old - S_MS
vote_wait_new = vote_new - S_NEW
vote_gain = vote_old / vote_new

# ---------------------------------------------------------------------------
# Fan-out: MS Sec. 13.4.5. Dean and Barroso, CACM 2013, Table 1: one leaf
# p99 10 ms, all leaves p99 140 ms.
# ---------------------------------------------------------------------------
fan10 = 0.99 ** 10 * 100
fan100 = 0.99 ** 100 * 100
DB_LEAF, DB_ALL = 10, 140
db_ratio = DB_ALL / DB_LEAF

# ---------------------------------------------------------------------------
# Fixed cost per image: MS Sec. 13.5.4 affine model, tables 8 and 9.
# 174 kernels at 10 us dispatch + 5 us launch (MS Sec. 13.3.2).
# ---------------------------------------------------------------------------
KERNELS, ISSUE_US = 174, 15
launch_ms = KERNELS * ISSUE_US / 1000
per_image = {b: batch_ms(b) / b for b in (1, 8, 32)}
fixed_share = {b: FIXED_MS / batch_ms(b) * 100 for b in (1, 8, 32)}
b1_ips = 1000 / S_MS
T8_IPS, T1_IPS, T32_IPS = 2130, 280, 7040     # MS table 8 as printed
tput_gain = T32_IPS / T1_IPS

# ---------------------------------------------------------------------------
# Runtimes at batch 1: MS table 14 from the mlsysim profile (unrounded).
# ---------------------------------------------------------------------------
RT = {"TorchScript": 1.7881, "ONNX Runtime": 0.6081, "TensorRT FP32": 0.3814,
      "TensorRT FP16": 0.3331, "TensorRT INT8": 0.3038}
rt_speedup = {k: S_MS / v for k, v in RT.items()}
rt_prec_gain = RT["TensorRT FP32"] / RT["TensorRT INT8"]
rt_rounded_fp16 = 3.53 / 0.33     # from the table's printed latencies
rt_rounded_int8 = 3.53 / 0.30

# ---------------------------------------------------------------------------
# Modeled curve and measured L4: MS Sec. 13.5.4; MLPerf v5.0, 5.0-0047:
# SingleStream p90 0.414 ms, MultiStream (8 per query) p99 0.770 ms,
# Offline 38,566 samples/s on three L4s.
# ---------------------------------------------------------------------------
aff = {b: batch_ms(b) for b in (1, 32, 128)}
aff_tput = {b: b / aff[b] * 1000 for b in (1, 32, 128)}
L4_SS, L4_MS8, L4_OFF3 = 0.414, 0.770, 38566
l4_one = 1000 / L4_SS
l4_eight = 8 / L4_MS8 * 1000
l4_all = L4_OFF3 / 3
l4_lat_ratio = L4_MS8 / L4_SS
l4_tput_ratio = l4_eight / l4_one

# ---------------------------------------------------------------------------
# Batching window: MS eqs. 8, 9, 12. Poisson counts per 10 ms window.
# ---------------------------------------------------------------------------
WIN = 10.0
win_wait_20 = 20 / 2
win_share_20 = win_wait_20 / P99_BUDGET * 100
empty_50 = math.exp(-50 * WIN / 1000) * 100
seen_mean = 1 + 500 * WIN / 1000            # size-biased: 1 + lambda W


def poisson_ppf(m, q):
    k, c = 0, math.exp(-m)
    while c < q:
        k += 1
        c += math.exp(-m) * m ** k / math.factorial(k)
    return k


seen_p99 = 1 + poisson_ppf(500 * WIN / 1000, 0.99)

# ---------------------------------------------------------------------------
# Triton dynamic batcher (docs/user_guide/batcher.md): delay 100 us against
# the 12 ms timeout of Example 13.4.
# ---------------------------------------------------------------------------
TRITON_DELAY_US = 100
triton_vs_example = 12_000 / TRITON_DELAY_US
triton_arrivals_500 = 500 * TRITON_DELAY_US * 1e-6   # expected arrivals per delay

# ---------------------------------------------------------------------------
# Dual trigger: MS Example 13.4. Arrivals 0, 2, 5, 9 ms; timeout 12 ms;
# worker busy until 18 ms; batch of four 3.6 ms; response 4 ms.
# ---------------------------------------------------------------------------
TIMEOUT, FREE_AT, B4_MS, RESP = 12, 18, 3.6, 4
dt_start = max(TIMEOUT, FREE_AT)
dt_wait_worker = FREE_AT - TIMEOUT
dt_end_batch = dt_start + B4_MS
dt_done = dt_end_batch + RESP
dt_slack = P99_BUDGET - dt_done
dt_wrong = TIMEOUT + B4_MS
b4_affine = AFF_FIXED + 4 * AFF_PER

# ---------------------------------------------------------------------------
# Batch cap 16 or 32: MS Sec. 13.11, eq. 6. 500 req/s offered.
# ---------------------------------------------------------------------------
OFFER, CAP16_IPS, CAP32_IPS = 500, 3980, 7040
fill16 = (16 - 1) / (2 * OFFER) * 1000
fill32 = (32 - 1) / (2 * OFFER) * 1000
cap_gain = (CAP32_IPS / CAP16_IPS - 1) * 100
cap_over_load = CAP16_IPS / OFFER

# ---------------------------------------------------------------------------
# Cold start: Fu et al., ServerlessLLM, Sec. 2.3: 130 GB at 5 GB/s.
# ---------------------------------------------------------------------------
sllm_dl = 130 / 5

# ---------------------------------------------------------------------------
# Cold-start log: coldstart_log.py, run here (coldstart_log.txt).
# ---------------------------------------------------------------------------
_log = (HERE / "coldstart_log.txt").read_text().split("# run 2")
_ready = [float(re.findall(r"\[\s*([\d.]+) s\] ready", p)[0]) for p in _log]
_comp = [float(re.findall(r"compiles\): ([\d.]+) s", p)[0]) for p in _log]
_pkl = [float(re.findall(r"\(pickle\): ([\d.]+) s", p)[0]) for p in _log]
_st = [float(re.findall(r"safetensors: ([\d.]+) s", p)[0]) for p in _log]
cs_ready1, cs_ready2 = _ready
cs_comp1, cs_comp2 = _comp
cs_comp_share = cs_comp1 / cs_ready1 * 100
cs_ready_gain = cs_ready1 / cs_ready2
cs_pkl1, cs_st1 = _pkl[0], _st[0]
RESNET_PARAMS_M = float(re.findall(r"ResNet-50 \(([\d.]+) M parameters", _log[0])[0])
cs_weights_mb = RESNET_PARAMS_M * 4          # FP32, 4 bytes per parameter

# ---------------------------------------------------------------------------
# Model swap: MS Sec. 13.8.4. 10 GB over PCIe Gen4 x16 at 32 GB/s.
# ---------------------------------------------------------------------------
swap_ms = 10 / 32 * 1000
swap_over_slo = swap_ms / P99_BUDGET
swap_passes = swap_ms / S_PRINT

# ---------------------------------------------------------------------------
# Cost per image: MS Example 13.13, table 18 (illustrative prices).
# ---------------------------------------------------------------------------
INST = {"CPU": (0.17, 50), "T4": (0.53, 400), "H100": (4.0, 7040)}
cpm = {k: usd / (rate * 3600) * 1e6 for k, (usd, rate) in INST.items()}
cpu_per_hour = INST["CPU"][1] * 3600
t4_7_usd = 7 * INST["T4"][0]
t4_7_ips = 7 * INST["T4"][1]
t4_8_usd = 8 * INST["T4"][0]
t4_cross = max(n for n in range(1, 50) if n * INST["T4"][0] < INST["H100"][0]) * INST["T4"][1]

# ---------------------------------------------------------------------------
# Synthesis: MS Napkin Math 13.3, Sec. 13.10.2; $4 per H100-hour.
# Batched plan: the same utilization factor applied to 7,040 images/s.
# ---------------------------------------------------------------------------
syn_batched = math.ceil(math.ceil(est_rate / T32_IPS) * HEADROOM)
syn_cost_b1 = est_final * 4.0 / (PEAK * 3600) * 1e6
syn_cost_bt = syn_batched * 4.0 / (PEAK * 3600) * 1e6
syn_ratio = syn_cost_b1 / syn_cost_bt
syn_cards_photo = est_final / 4              # the photo shows four H100 cards


def main():
    head("Hook (Example 13.1)")
    show("cost ratio", f"{f(hook_ratio)}x")
    show("batch 1 / batch 8 time from rates", f"{hook_b1_ms:.3f} / {hook_b8_ms:.3f} ms")
    show("batch 8 adds", f"{hook_extra_ms:.2f} ms ({hook_extra_pct:.0f} percent of the batch-8 time)")
    head("Recall Lecture 12")
    show("one at a time", f"about {sig(ss_rate, 3):,.0f}/s")
    show("Offline over one at a time", f"{f(off_over_ss)}x")
    head("Static against dynamic")
    show("library at batch 1", f"{f(static_b1_s, 0)} s")
    show("library at 89,500/s", f"{f(static_int8_s, 2)} s")
    show("ratio", f"{f(static_ratio, 0)}x (the chapter: about 320x)")
    head("Request budget (table 3)")
    show("total", f"{f(budget_total)} ms")
    show("model share", f"{f(model_share, 0)} percent")
    show("preprocessing share", f"{f(pre_share, 0)} percent")
    show("GPU idle before work", f"{f(gpu_idle_ms)} ms")
    head("DLRM")
    show("total", f"{f(dlrm_total, 0)} ms")
    show("MLP share", f"{f(dlrm_mlp_share, 0)} percent")
    head("Pipelining")
    show("serial duty", f"{f(serial_duty, 0)} percent")
    show("pipelined: one request every", f"{f(host_stage)} ms")
    show("pipelined duty", f"{f(pipe_duty, 0)} percent")
    head("Predict: 10x faster model")
    show("new total", f"{f(fast_total)} ms")
    show("speedup", f"{f(fast_speedup)}x")
    show("naive answer", f"{f(naive_ms, 2)} ms")
    show("Richins frame total", f"{f(rich_total, 0)} ms")
    show("AI stages", f"{f(rich_ai, 0)} percent")
    show("brokers", f"{f(rich_brk, 1)} percent")
    head("perf_analyzer output (Triton docs)")
    show("client minus server", f"{pa_outside} usec")
    show("compute infer share of client latency", f"{f(pa_infer_share, 0)} percent")
    show("server parts sum", f"{pa_parts_sum} usec (reported 382)")
    show("Little's law: throughput x latency", f"{f(pa_little, 2)} requests in flight")
    head("Little's law")
    show("N", f"{f(little_n, 0)}")
    show("waiting in host memory", f"{f(little_wait, 0)}")
    show("50 at 10 ms sustains", f"{little_conv:,.0f} per second")
    head("Knee (M/M/1, S = 3.528 ms)")
    for rho in KNEE_RHOS:
        show(f"mean at {rho:.0%}", f"{knee[rho]:.1f} ms")
    show("p99 at 70%", f"{knee_p99_70:.1f} ms")
    show("ln 100", f"{f(ln100, 2)}")
    head("Estimate (Napkin Math 13.3)")
    show("service time", f"{f(S_MS, 3)} ms, {f(est_mu1, 1)}/s per GPU")
    show("rho limit", f"{f(est_rho, 4)}")
    show("rate needed", f"{f(est_rate, 0, True)} per second")
    show("GPUs", f"{f(est_raw)} -> {est_ceil}")
    show("with 30% headroom", f"{est_final}")
    show("one fails", f"{est_final - 1} GPUs at {f(est_fail_util)} percent busy")
    show("naive", f"{f(est_naive, 1)} -> {math.ceil(est_naive)} GPUs")
    show("old slide 5,000 / 0.68", f"{f(old_rate, 0, True)} (slide had 7,403)")
    show("with S = 3.53 ms", f"{f(old_rate_353, 0, True)}")
    show("rho limit with exactly 3.5 ms", f"{f(rho_at_35, 3)}")
    head("MLPerf Server against Offline")
    show("A kept", f"{f(a_kept, 0)} percent")
    show("B kept", f"{f(b_kept, 0)} percent")
    show("gap", f"{f(a_kept - b_kept, 0)} points")
    head("Vote: 2x faster model at 80 percent")
    show("old mean", f"{f(vote_old)} ms (wait {f(vote_wait_old)})")
    show("new utilization", f"{f(vote_rho_new * 100, 0)} percent")
    show("new mean", f"{f(vote_new)} ms (wait {f(vote_wait_new)})")
    show("gain", f"{f(vote_gain)}x")
    head("Fan-out")
    show("10 servers", f"{f(fan10)} percent")
    show("100 servers", f"{f(fan100)} percent")
    show("Dean and Barroso all/one leaf", f"{f(db_ratio, 0)}x")
    head("Fixed cost per image")
    show("launch time at batch 1", f"{f(launch_ms)} ms")
    for b in (1, 8, 32):
        show(f"batch {b}", f"{f(per_image[b], 2)} ms per image, {f(fixed_share[b], 0)} percent fixed")
    show("batch 1 rate", f"{f(b1_ips, 0)}/s")
    show("throughput gain 1 -> 32 (table 8: 280 -> 7,040)", f"{f(tput_gain, 0)}x")
    head("Runtimes (table 14)")
    for k, v in rt_speedup.items():
        show(k, f"{f(v)}x")
    show("INT8 over TensorRT FP32", f"{f(rt_prec_gain)}x")
    show("from rounded table latencies: FP16, INT8",
         f"{f(rt_rounded_fp16)}x, {f(rt_rounded_int8)}x (table prints 10.6x, 11.6x)")
    head("Modeled curve and L4")
    for b in (1, 32, 128):
        show(f"batch {b}", f"{f(aff[b], 2)} ms, {sig(aff_tput[b], 3):,.0f}/s")
    show("L4 one at a time", f"{f(l4_one, 0, True)}/s")
    show("L4 eight at a time", f"{sig(l4_eight, 4):,.0f}/s")
    show("L4 all at once per GPU", f"{f(l4_all, 0, True)}/s")
    show("latency ratio", f"{f(l4_lat_ratio)}x")
    show("throughput ratio", f"{f(l4_tput_ratio)}x")
    head("Batching window")
    show("20 ms window mean wait", f"{f(win_wait_20, 0)} ms, {f(win_share_20, 0)} percent of 50 ms")
    show("empty windows at 50 QPS", f"{f(empty_50, 0)} percent")
    show("batch a request sees at 500 QPS", f"mean {f(seen_mean, 0)}, p99 {seen_p99}")
    head("Triton dynamic batcher")
    show("12 ms timeout over 100 us delay", f"{f(triton_vs_example, 0)}x")
    show("arrivals in 100 us at 500 req/s", f"{f(triton_arrivals_500, 2)}")
    head("Dual trigger (Example 13.4)")
    show("batch runs", f"{f(dt_start, 0)} to {f(dt_end_batch)} ms (waits {dt_wait_worker} ms for the worker)")
    show("done", f"{f(dt_done)} ms, slack {f(dt_slack)} ms")
    show("wrong answer", f"{f(dt_wrong)} ms")
    show("affine model, batch of 4", f"{f(b4_affine)} ms")
    head("Batch cap 16 or 32")
    show("mean wait to fill", f"{f(fill16, 0)} -> {f(fill32, 0)} ms")
    show("throughput gain", f"{f(cap_gain, 0)} percent")
    show("capacity over load at 16", f"{f(cap_over_load, 0)}x")
    head("Cold start")
    show("ServerlessLLM download", f"{f(sllm_dl, 0)} s")
    head("Cold-start log (coldstart_log.txt)")
    show("run 1 ready", f"{f(cs_ready1)} s; compile {f(cs_comp1)} s ({f(cs_comp_share, 0)} percent)")
    show("run 2 ready", f"{f(cs_ready2)} s; compile {f(cs_comp2)} s")
    show("ready sooner", f"{f(cs_ready_gain)}x")
    show("FP32 weights", f"{f(cs_weights_mb, 0)} MB")
    show("pickle against safetensors, run 1", f"{f(cs_pkl1 * 1000, 0)} ms against {f(cs_st1 * 1000, 0)} ms")
    head("Model swap")
    show("10 GB at 32 GB/s", f"{f(swap_ms)} ms")
    show("over a 50 ms deadline", f"{f(swap_over_slo, 2)}x")
    show("forward passes", f"{f(swap_passes, 0)}")
    head("Cost per image (Example 13.13)")
    for k, v in cpm.items():
        show(f"{k} per million", f"${f(v, 2)}")
    show("CPU images per hour", f"{cpu_per_hour:,.0f}")
    show("7 T4s", f"${f(t4_7_usd, 2)}/h, {t4_7_ips:,}/s")
    show("8 T4s", f"${f(t4_8_usd, 2)}/h")
    show("T4s cheaper than one H100 up to", f"{t4_cross:,}/s")
    head("Synthesis")
    show("batched GPUs", f"{syn_batched}")
    show("cost batch 1", f"${f(syn_cost_b1, 2)} per million")
    show("cost batched", f"${f(syn_cost_bt, 2)} per million")
    show("ratio", f"{f(syn_ratio, 0)}x")
    show("batch-1 fleet over the four cards in the photo", f"{f(syn_cards_photo, 0)}x")


if __name__ == "__main__":
    main()
