#!/usr/bin/env python3
"""Every derived number in Lecture 3 (The Cognitive Brain), one block per slide.

Inputs carry their source. "Ch3" is the rendered chapter, The Cognitive Brain
(Volume IV, Chapter 3); "NM 3.x" is its Napkin Math; "RM" is the running
machine's MLSysIM registry entry (Embodied.MobileManipulator), which the
chapter prints from. Every printed value is what the slide or its note shows,
with the label used there. make_figs.py imports from this file.

Run from this folder: python3 calcs.py   (needs pint: use the MLSysBook venv)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Embodied, Literature  # noqa: E402
from mlsysim.physics.transformer import calc_vla_inference_latency  # noqa: E402

GB = 1e9
MB = 1e6

RM = Embodied.MobileManipulator.WarehouseMobileManipulator


def head(title):
    print(f"\n--- {title}")


def show(label, value):
    print(f"  {label}: {value}")


def q(x, unit):
    """Magnitude of a pint quantity in the given unit."""
    return x.to(unit).magnitude


# =============================================================================
# Shared inputs (the running machine; Ch3 Napkin Math 3.1 to 3.3)
# =============================================================================
V_DRIVE = q(RM.base.max_velocity, "m/s")      # 1.5 m/s base drive limit (RM; Ch1 machine card)
N_PARAMS = q(RM.intent_model.parameters, "param")  # 7e9, OpenVLA-sized intent model (NM 3.1, datasheet)
N_LAYERS = RM.intent_model.layers             # 32 (NM 3.1, datasheet)
D_MODEL = RM.intent_model.hidden_dim          # 4096 (NM 3.1, datasheet)
D_ACTION = RM.intent_model.action_dim         # 7 tokens, one per output dimension (NM 3.1)
S_IN = RM.intent_input_tokens                 # 288 = 256 patches + 32 prompt (NM 3.1)
PEAK_FP16 = q(RM.brain_soc.compute.precision_flops["fp16"], "flop/s")  # 43 TFLOP/s dense FP16 (Ch3 Sec. 3.4.1)
BW_PEAK = q(RM.brain_soc.memory.bandwidth, "byte/s")  # 204 GB/s (Ch3 Sec. 3.4.1)
SOC_MEM = q(RM.brain_soc.memory.capacity, "byte")     # 64 GiB module = 68.7 GB (Ch3 Sec. 3.5.2)
ETA_BUS = RM.dram_efficiency                  # 0.70 sustained fraction (NM 3.1, illustrative)
U_PREFILL = RM.intent_compute_utilization     # 0.50 achieved fraction of peak (NM 3.1, illustrative)
INTENT_RATE = q(RM.intent_rate, "Hz")         # 2 Hz (chosen, NM 3.1)
CHUNK_RATE = q(RM.chunk_rate, "Hz")           # 20 Hz (chosen, NM 3.2)
CHUNK_INFER = q(RM.chunk_inference_latency, "ms")  # 40 ms p99 target incl. vision (NM 3.3, illustrative)
CHUNK_STEP = q(RM.chunk_step, "ms")           # 25 ms setpoint step (NM 3.3, chosen)
H = RM.chunk_horizon                          # 16 setpoints per chunk (chosen)
CHUNK_PARAMS = q(RM.chunk_model.parameters, "param")  # 80e6, ACT's size (NM 3.2, datasheet)
TICK_RATE = q(RM.permission_rate, "Hz")       # 1 kHz permission tick (Ch3 Sec. 3.1)
MEASURED_INT4 = q(RM.intent_reference_latency, "ms")  # 336 ms, Jetson AI Lab (Franklin 2024)

BW_SUS = BW_PEAK * ETA_BUS
W_FP16 = N_PARAMS * 2.0
W_INT8 = N_PARAMS * 1.0
W_INT4 = N_PARAMS * 0.5


# =============================================================================
# Slide: hook, how far does the base move while the model thinks?
# =============================================================================
head("Hook (Jetson AI Lab measurement; RM drive limit)")
LOG_INT4 = 336.2      # ms, the per-frame log line shown (Jetson AI Lab, step 355)
HOOK_MM = MEASURED_INT4 / 1000 * V_DRIVE * 1000
show("base drive limit (m/s)", V_DRIVE)
show("travel during 336 ms at 1.5 m/s (mm, derived)", f"{HOOK_MM:.0f}")
show("same, logged frame 336.2 ms (mm)", f"{LOG_INT4 / 1000 * V_DRIVE * 1000:.1f}")
show("evaluations per second at 336 ms", f"{1000 / MEASURED_INT4:.2f}")

# =============================================================================
# Slide: a VLA reads tokens (OpenVLA config.json: 224 px, patch 14)
# =============================================================================
head("Tokens (Ch3 Sec. 3.3)")
PATCHES = (224 // 14) ** 2
show("patches per 224x224 view at patch 14", PATCHES)
show("three views (tokens)", 3 * PATCHES)
show("three views + 32 prompt + 16 state", 3 * PATCHES + 32 + 16)
show("intent model: one view + 32 prompt (tokens)", PATCHES + 32)  # NM 3.1: S = 288
EMBED_MB = 800 * D_MODEL * 2 / MB
show("embeddings for S = 800, FP16 (MB)", f"{EMBED_MB:.2f}")
KV_TOKEN = 2 * N_LAYERS * D_MODEL * 2
show("KV cache per token, FP16, 32 layers x 4096 (bytes)", f"{KV_TOKEN:,}")
show("KV cache per token (MB)", f"{KV_TOKEN / MB:.3f}")
show("FP16 weights (GB)", f"{W_FP16 / GB:.0f}")
show("weights / one step of embeddings", f"{W_FP16 / (EMBED_MB * MB):,.0f}x")

# =============================================================================
# Slide: measured on the Orin (MLPerf Inference edge, single stream, p90)
# =============================================================================
head("MLPerf Inference on Jetson AGX Orin 64G (Ch3 figure 9 data, via MLSysIM)")
SYSTEM = "NVIDIA Jetson AGX Orin Developer Kit 64G (TensorRT)"
MLPERF_TASKS = [
    ("resnet50", "Image classification", "ResNet-50"),
    ("bert-99", "Question answering", "BERT"),
    ("retinanet", "Object detection", "RetinaNet"),
    ("rnnt", "Speech recognition", "RNN-T"),
    ("3d-unet-99", "3D segmentation", "3D U-Net"),
    ("gptj-99", "Text summarization", "GPT-J 6B"),
    ("stable-diffusion-xl", "Text to image", "SDXL"),
]


def mlperf_latest(bench):
    hits = [r for r in Literature.EdgeInferenceOrin.entries
            if r["system"] == SYSTEM and r["benchmark"] == bench
            and r["scenario"] == "SingleStream" and r["metric"].startswith("p90")]
    best = max(hits, key=lambda r: r["round"])
    return best["value"], best["round"], sorted(hits, key=lambda r: r["round"])


MLPERF = []
for bench, task, model in MLPERF_TASKS:
    v, rnd, allr = mlperf_latest(bench)
    MLPERF.append((task, model, v, rnd))
    show(f"{task} ({model}) p90 ms, {rnd}", f"{v:,.2f}")
T_CHUNK = 1000 / CHUNK_RATE
T_INTENT = 1000 / INTENT_RATE
n_in_chunk = sum(1 for m in MLPERF if m[2] <= T_CHUNK)
show("models inside the 50 ms chunk period", n_in_chunk)
GPTJ = [m for m in MLPERF if m[1] == "GPT-J 6B"][0][2]
GPTJ_EARLY = mlperf_latest("gptj-99")[2][0]["value"]
show("GPT-J / intent period", f"{GPTJ / T_INTENT:.1f}x")
show("GPT-J one round earlier (s)", f"{GPTJ_EARLY / 1000:.1f}")
SPAN = max(m[2] for m in MLPERF) / min(m[2] for m in MLPERF)
show("slowest / fastest", f"{SPAN:,.0f}x")

# =============================================================================
# Slide: the memory wall (focus) and the cadence plot
# =============================================================================
head("Memory wall (Ch3 Definition 3.1, NM 3.1)")
T_SWEEP16 = W_FP16 / BW_SUS * 1000
T_SWEEP8 = W_INT8 / BW_SUS * 1000
T_SWEEP4 = W_INT4 / BW_SUS * 1000
show("sustained bandwidth 0.7 x 204 (GB/s)", f"{BW_SUS / GB:.1f}")
show("FP16 sweep 14 GB / 142.8 GB/s (ms)", f"{T_SWEEP16:.1f}")
show("FP16 ceiling (Hz)", f"{1000 / T_SWEEP16:.1f}")
show("INT4 sweep 3.5 GB (ms)", f"{T_SWEEP4:.1f}")
show("INT4 ceiling (Hz)", f"{1000 / T_SWEEP4:.1f}")
show("FP16 ceiling below the 1 kHz tick", f"{TICK_RATE / (1000 / T_SWEEP16):.0f}x")
CHUNK_W16 = CHUNK_PARAMS * 2
show("chunk policy FP16 weights (GB)", f"{CHUNK_W16 / GB:.2f}")
show("chunk policy FP16 ceiling (Hz)", f"{BW_SUS / CHUNK_W16:.0f}")
show("chunk policy sweep (ms)", f"{CHUNK_W16 / BW_SUS * 1000:.2f}")

# =============================================================================
# Slide: roofline (decode at 1 FLOP/byte, prefill past the ridge)
# =============================================================================
head("Roofline (Ch3 Sec. 3.4.1)")
RIDGE = PEAK_FP16 / BW_PEAK
show("ridge 43 TFLOP/s / 204 GB/s (FLOP/byte)", f"{RIDGE:.1f}")
DECODE_I = 2 / 2
show("decode intensity FP16 (FLOP/byte)", DECODE_I)
show("decode attainable (TFLOP/s)", f"{BW_PEAK * DECODE_I / 1e12:.3f}")
show("decode share of FP16 peak (%)", f"{BW_PEAK * DECODE_I / PEAK_FP16 * 100:.2f}")
PREFILL_I = 2 * N_PARAMS * S_IN / W_FP16
show("prefill intensity, S = 288 (FLOP/byte)", f"{PREFILL_I:.0f}")

# =============================================================================
# Slide: worked example 1, pricing one evaluation (NM 3.1)
# =============================================================================
head("Pricing one evaluation (NM 3.1, MLSysIM calc_vla_inference_latency)")
PRE_LIN = 2 * N_PARAMS * S_IN
PRE_ATT = 2 * N_LAYERS * D_MODEL * S_IN ** 2
show("prefill linear 2NS (FLOP)", f"{PRE_LIN:.3e}")
show("prefill attention 2 L d S^2 (FLOP)", f"{PRE_ATT:.3e}")
show("prefill total (FLOP)", f"{PRE_LIN + PRE_ATT:.3e}")
T_PREFILL = (PRE_LIN + PRE_ATT) / (U_PREFILL * PEAK_FP16) * 1000
show("prefill at 50% of 43 TFLOP/s (ms)", f"{T_PREFILL:.1f}")
show("achieved prefill rate (TFLOP/s)", f"{U_PREFILL * PEAK_FP16 / 1e12:.1f}")
T_FP16 = T_PREFILL + (D_ACTION - 1) * T_SWEEP16
T_INT8 = T_PREFILL + (D_ACTION - 1) * T_SWEEP8
T_INT4 = T_PREFILL + (D_ACTION - 1) * T_SWEEP4
show("FP16 decode 6 sweeps (ms)", f"{(D_ACTION - 1) * T_SWEEP16:.0f}")
show("FP16 total (ms)", f"{T_FP16:.0f}")
show("FP16 rate (Hz)", f"{1000 / T_FP16:.1f}")
show("INT4 decode 6 sweeps (ms)", f"{(D_ACTION - 1) * T_SWEEP4:.0f}")
show("INT4 total (ms)", f"{T_INT4:.0f}")
show("INT4 rate ceiling (Hz)", f"{1000 / T_INT4:.2f}")
show("8-bit total (ms)", f"{T_INT8:.0f}")
show("prefill share at INT4 (%)", f"{T_PREFILL / T_INT4 * 100:.0f}")
show("weight-only ceiling 1 / prefill (Hz)", f"{1000 / T_PREFILL:.1f}")
show("3 Hz period (ms)", f"{1000 / 3:.0f}")
# cross-check against MLSysIM's own function (the chapter's source of these values)
R16 = calc_vla_inference_latency(RM.intent_model.parameters, RM.intent_weight_memory * 4, S_IN, D_ACTION,
                                 RM.brain_soc.compute.precision_flops["fp16"], RM.brain_soc.memory.bandwidth,
                                 bus_efficiency=ETA_BUS, compute_utilization=U_PREFILL,
                                 n_layers=N_LAYERS, hidden_dim=D_MODEL)
show("MLSysIM check, FP16 total (ms)", f"{q(R16['t_total'], 'ms'):.1f}")
show("MLSysIM check, INT4 total (ms)", f"{q(RM.intent_inference_latency, 'ms'):.1f}")

# =============================================================================
# Slide: measured against computed (Jetson AI Lab table; Ch3 table 1)
# =============================================================================
head("Quantization, computed vs measured (Ch3 table 1; Franklin 2024)")
MEAS = {"FP16": 840.0, "8-bit": 471.0, "INT4": 336.0}           # ms, Jetson AI Lab table
SCORE = {"FP16": 95.3, "8-bit": 95.2, "INT4": 90.1}             # action score %, same table
COMP = {"FP16": T_FP16, "8-bit": T_INT8, "INT4": T_INT4}
for k in MEAS:
    show(f"{k}: computed {COMP[k]:.0f} ms vs measured {MEAS[k]:.0f} ms, gap (%)",
         f"{(COMP[k] - MEAS[k]) / MEAS[k] * 100:+.1f}")
show("INT4 speedup over FP16, measured", f"{MEAS['FP16'] / MEAS['INT4']:.1f}x")
show("action score drop FP16 to INT4 (points)", f"{SCORE['FP16'] - SCORE['INT4']:.1f}")
OVLA = {"bfloat16": 71.3, "INT8": 58.1, "INT4": 71.9}            # % success, Kim et al. 2024 Table 2
show("OpenVLA real-robot success (Kim et al. 2024)", OVLA)
show("INT8 control rate vs training data (Hz)", "1.2 vs 5")
show("AWQ scale overhead, 16-bit scale per 128 four-bit weights (%)", f"{16 / (128 * 4) * 100:.1f}")

# =============================================================================
# Activity: would halving the input tokens fit a 5 Hz period? (Checkpoint 3.1)
# =============================================================================
head("Estimate: half the input tokens (Checkpoint 3.1, prefill proportional to S)")
T_HALF = T_PREFILL / 2 + (D_ACTION - 1) * T_SWEEP4
show("prefill halved (ms)", f"{T_PREFILL / 2:.0f}")
show("total with 6 INT4 sweeps (ms)", f"{T_HALF:.0f}")
show("5 Hz period (ms)", f"{1000 / 5:.0f}")
show("over the period by (ms)", f"{T_HALF - 200:.0f}")
show("prefill alone that would fit (ms)", f"{200 - (D_ACTION - 1) * T_SWEEP4:.0f}")
show("token fraction that would fit", f"{(200 - (D_ACTION - 1) * T_SWEEP4) / T_PREFILL:.2f}")
show("tokens that would fit, of 288", f"{(200 - (D_ACTION - 1) * T_SWEEP4) / T_PREFILL * S_IN:.0f}")
T_3CAM = q(calc_vla_inference_latency(RM.intent_model.parameters, RM.intent_weight_memory, 800, D_ACTION,
                                      RM.brain_soc.compute.precision_flops["fp16"], RM.brain_soc.memory.bandwidth,
                                      bus_efficiency=ETA_BUS, compute_utilization=U_PREFILL,
                                      n_layers=N_LAYERS, hidden_dim=D_MODEL)["t_total"], "ms")
show("three views, S = 800, INT4 total (ms)", f"{T_3CAM:.0f}")

# =============================================================================
# Slide: chunking divides weight traffic by H (Ch3 Sec. 3.5.1)
# =============================================================================
head("Chunking supply (Ch3 Sec. 3.5.1)")
SETPOINT_RATE = 1000 / CHUNK_STEP
DEMAND_SINGLE = SETPOINT_RATE * W_FP16
show("setpoint rate (Hz)", f"{SETPOINT_RATE:.0f}")
show("single-step demand 40 x 14 GB (GB/s)", f"{DEMAND_SINGLE / GB:.0f}")
show("demand / sustained", f"{DEMAND_SINGLE / BW_SUS:.1f}x")
show("chunked demand, H = 16 (GB/s)", f"{DEMAND_SINGLE / H / GB:.0f}")
show("chunked share of sustained (%)", f"{DEMAND_SINGLE / H / BW_SUS * 100:.1f}")
REACT = 1000 / CHUNK_RATE + CHUNK_INFER
show("reaction latency 50 + 40 (ms)", f"{REACT:.0f}")

# =============================================================================
# Slide: decoder families on the intent-sized backbone (Ch3 Sec. 3.5.1)
# =============================================================================
head("Decoder ladder (Ch3 Sec. 3.5.1, INT4 backbone, one 16-step chunk of 7 values)")


def vla_ms(tokens, parallel=False):
    r = calc_vla_inference_latency(RM.intent_model.parameters, RM.intent_weight_memory, S_IN, tokens,
                                   RM.brain_soc.compute.precision_flops["fp16"], RM.brain_soc.memory.bandwidth,
                                   bus_efficiency=ETA_BUS, compute_utilization=U_PREFILL, parallel_decode=parallel,
                                   n_layers=N_LAYERS, hidden_dim=D_MODEL)
    return q(r["t_total"], "ms")


N_BINNED = H * 7
T_BINNED = vla_ms(N_BINNED)
T_FAST = vla_ms(30)                       # FAST: 30 tokens per arm per 1 s chunk (Pertsch et al. 2025)
T_PAR = vla_ms(N_BINNED, parallel=True)
T_PRE_ONLY = vla_ms(1)
EXPERT_MB = 300e6 * 2 / MB                # pi0 action expert, 300M params, 16-bit (Black et al. 2024)
PREFIX_KV_MB = KV_TOKEN * S_IN / MB
T_EXPERT_STEP = (EXPERT_MB + PREFIX_KV_MB) * MB / BW_SUS * 1000
T_EXPERT = T_PRE_ONLY + 10 * T_EXPERT_STEP
show("binned tokens", N_BINNED)
show("binned autoregressive (s)", f"{T_BINNED / 1000:.1f}")
show("FAST, 30 tokens (ms)", f"{T_FAST:.0f}")
show("parallel head (ms)", f"{T_PAR:.0f}")
show("prefill only (ms)", f"{T_PRE_ONLY:.0f}")
show("expert weights (MB)", f"{EXPERT_MB:.0f}")
show("cached prefix KV (MB)", f"{PREFIX_KV_MB:.0f}")
show("expert step on bandwidth (ms)", f"{T_EXPERT_STEP:.1f}")
show("flow expert, 10 steps (ms)", f"{T_EXPERT:.0f}")
show("spread binned / expert", f"{T_BINNED / T_EXPERT:.0f}x")
show("prefill / 50 ms period", f"{T_PRE_ONLY / T_CHUNK:.1f}x")

# =============================================================================
# Slide: published dual systems (Ch3 table 2)
# =============================================================================
head("Published two-speed designs (Ch3 table 2)")
DUAL = [  # (system, slow lo, slow hi, fast Hz), each from Ch3 table 2
    ("Helix", 7.0, 9.0, 200.0),       # Figure AI 2025: 7B at 7-9 Hz, 80M at 200 Hz
    ("GR00T N1", 10.0, 10.0, 120.0),  # Bjorck et al. 2025
    ("This machine", INTENT_RATE, INTENT_RATE, CHUNK_RATE),
]
for name, lo, hi, fast in DUAL:
    show(f"{name}: fast / slow", f"{fast / hi:.0f}x to {fast / lo:.0f}x" if lo != hi else f"{fast / lo:.0f}x")

# =============================================================================
# Slide: worked example 2, the renewal gap (NM 3.2)
# =============================================================================
head("Renewal gap (NM 3.2)")
T_AMORT = T_SWEEP16 / H
show("amortized FP16 sweep per setpoint 98.0 / 16 (ms)", f"{T_AMORT:.1f}")
show("INT4 evaluation / chunk period", f"{q(RM.intent_inference_latency, 'ms') / T_CHUNK:.1f}")
show("chunk policy sweep 0.16 GB / 142.8 GB/s (ms)", f"{CHUNK_W16 / BW_SUS * 1000:.2f}")
show("chunk sweep share of its 50 ms period (%)", f"{CHUNK_W16 / BW_SUS * 1000 / T_CHUNK * 100:.1f}")
BUS_INTENT = D_ACTION * W_INT4 * INTENT_RATE
BUS_CHUNK = CHUNK_W16 * CHUNK_RATE
show("intent traffic 7 x 3.5 GB x 2 Hz (GB/s)", f"{BUS_INTENT / GB:.0f}")
show("chunk traffic 0.16 GB x 20 Hz (GB/s)", f"{BUS_CHUNK / GB:.1f}")
show("total (GB/s)", f"{(BUS_INTENT + BUS_CHUNK) / GB:.1f}")
show("share of sustained (%)", f"{(BUS_INTENT + BUS_CHUNK) / BW_SUS * 100:.0f}")
show("intent decode window per 500 ms (ms)", f"{(D_ACTION - 1) * T_SWEEP4:.0f}")

# =============================================================================
# Slides: evidence horizon and the timing strip (NM 3.3)
# =============================================================================
head("Evidence horizon (NM 3.3)")
V_CONV = 0.20          # m/s conveyor speed (NM 3.3, illustrative)
A_SLIP = 0.50          # m/s^2 residual slip (NM 3.3, illustrative)
R_TOL = 15.0           # mm grasp tolerance (NM 3.3, chosen)
E0 = 3.0               # mm initial handle error (NM 3.3, illustrative)
MARGIN = R_TOL - E0
T_UNTRACKED = MARGIN / V_CONV          # mm / (m/s) = ms
TAU_EV = math.sqrt(2 * MARGIN / 1000 / A_SLIP) * 1000
DRIFT_90 = 0.5 * A_SLIP * (REACT / 1000) ** 2 * 1000
SPAN = H * CHUNK_STEP
DRIFT_SPAN = 0.5 * A_SLIP * (SPAN / 1000) ** 2 * 1000
H_MIN = math.ceil(REACT / CHUNK_STEP)
show("margin 15 - 3 (mm)", MARGIN)
show("untracked deadline 12 mm / 0.20 m/s (ms)", f"{T_UNTRACKED:.0f}")
show("tracked horizon sqrt(2 x 12 mm / 0.5) (ms)", f"{TAU_EV:.0f}")
show("drift at the 90 ms age (mm)", f"{DRIFT_90:.1f}")
show("minimum setpoints ceil(90 / 25)", H_MIN)
show("chunk span 16 x 25 (ms)", f"{SPAN:.0f}")
show("drift over the whole chunk (mm)", f"{DRIFT_SPAN:.0f}")
show("error at chunk end 3 + 40 (mm)", f"{E0 + DRIFT_SPAN:.0f}")
show("chunk outlasts evidence by (ms)", f"{SPAN - TAU_EV:.0f}")
show("setpoints inside the evidence horizon", f"{TAU_EV / CHUNK_STEP:.1f}")
REACT_10HZ = 100 + CHUNK_INFER
show("10 Hz renewal: age at replacement (ms)", f"{REACT_10HZ:.0f}")
show("10 Hz renewal: minimum setpoints", math.ceil(REACT_10HZ / CHUNK_STEP))

head("Estimate: residual slip quadruples (Checkpoint 3.2)")
TAU_4X = math.sqrt(2 * MARGIN / 1000 / (4 * A_SLIP)) * 1000
show("horizon factor 1/sqrt(4)", 0.5)
show("new horizon (ms)", f"{TAU_4X:.0f}")
show("spare after the 90 ms age (ms)", f"{TAU_4X - REACT:.0f}")
show("drift at 90 ms with 4x slip (mm)", f"{0.5 * 4 * A_SLIP * (REACT / 1000) ** 2 * 1000:.1f}")

# =============================================================================
# Slide: temporal ensembling (ACT code, k = 0.01)
# =============================================================================
head("ACT temporal ensembling weights (imitate_episodes.py, k = 0.01, chunk 100)")
K_ACT = 0.01
w = [math.exp(-K_ACT * i) for i in range(100)]
show("weight of oldest / newest of 100 overlapping predictions", f"{w[0] / w[-1]:.2f}")
show("newest prediction's share of the blend (%)", f"{w[-1] / sum(w) * 100:.2f}")
show("machine: overlapping chunks at one step, 16 / 2", H // 2)

# =============================================================================
# Slide: retained history (KV ladder, Ch3 Sec. 3.5.2)
# =============================================================================
head("KV history (Ch3 Sec. 3.5.2)")
KV_768 = KV_TOKEN * 768 / MB
KV_STEP = KV_TOKEN * 800 / MB
KV_5S = KV_STEP * 5 * 10 / 1000
show("one 3-view frame set, 768 tokens (MB)", f"{KV_768:.1f}")
show("one step, 800 tokens (MB)", f"{KV_STEP:.1f}")
show("5 s at 10 Hz (GB)", f"{KV_5S:.1f}")
show("history + FP16 weights (GB)", f"{KV_5S + W_FP16 / GB:.1f}")
show("module memory 64 GiB (GB)", f"{SOC_MEM / GB:.1f}")
show("share of module (%)", f"{(KV_5S + W_FP16 / GB) / (SOC_MEM / GB) * 100:.0f}")

# =============================================================================
# Slide: softmax illusion (Ch3 table 4, illustrative values)
# =============================================================================
head("Confidence vs success (Ch3 table 4, illustrative)")
REGIMES = [("In distribution", 0.91, 0.89), ("Near OOD", 0.89, 0.52),
           ("Sensor degraded", 0.84, 0.31), ("Far OOD", 0.78, 0.08)]
for name, c, a in REGIMES:
    show(f"{name}: gap conf - success", f"{c - a:+.2f}")
show("near-OOD attempts failing (%)", f"{(1 - 0.52) * 100:.0f}")

# =============================================================================
# Synthesis: the hook answered
# =============================================================================
head("Synthesis")
show("336 ms = prefill + decode (ms)", f"{T_PREFILL:.0f} + {(D_ACTION - 1) * T_SWEEP4:.0f}")
show("intent evaluation vs tracked evidence horizon (ms)", f"{q(RM.intent_inference_latency, 'ms'):.0f} vs {TAU_EV:.0f}")
show("chunk reaction vs evidence horizon (ms)", f"{REACT:.0f} vs {TAU_EV:.0f}")
show("base travel in the 90 ms chunk reaction at 1.5 m/s (mm)", f"{REACT / 1000 * V_DRIVE * 1000:.0f}")
