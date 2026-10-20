#!/usr/bin/env python3
"""Every derived number in Lecture 9 (Fleet Efficiency), computed from sourced
inputs and printed with the slide that shows it. Copy printed values onto the
slides and notes; rerun after any change and compare.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Literature, Models  # noqa: E402

runs = Literature.TrainingRuns
# --- Inputs -----------------------------------------------------------------
R_LLAMA = float(runs.Llama3_405B_EffectiveTrainingTimeFraction)  # 0.90, lower bound (Dubey 2024)
P_LLAMA16 = float(runs.Llama3_405B_Mfu16K)    # 0.41, Table 4, BF16, 16,384 GPUs
P_LLAMA8 = float(runs.Llama3_405B_Mfu8K)      # 0.43, Table 4, BF16, 8,192 GPUs
PALM_CHIPS = int(Models.Language.PaLM_540B.training_accelerators_ref.magnitude)  # 6,144
ASSIGNED, PARTIAL, INPUT_WAIT = 0.80, 0.15, 0.05  # pool snapshot, Napkin Math 9.2
R_JOB = 1 - 0.0961        # chapter's unrounded runtime goodput (Napkin Math 9.3, 9.5)
P_NODE, T_NODE, T_FLEET = 0.65, 180.0, 245.0  # Napkin Math 9.4 (ms)
USEFUL_PFLOP = 14.9       # useful work per step, Napkin Math 9.4
H100_PEAK = 989.0         # TFLOP/s, dense FP16/BF16 (chapter)
H100_BW = 3.35            # TB/s HBM (chapter)
LN_FLOPS, LN_BW = 5.0, 2.8  # measured LayerNorm, TFLOP/s and TB/s (Sec. 9.6.3)
MTBF128_H = 302.0         # hours at 128 GPUs (Napkin Math 9.3)
T_WRITE_S, T_REC_MIN, T_START_MIN, PREEMPT_MIN = 9.8, 6.2, 5.2, 20.2  # Napkin Math 9.3
PRICE = 2.0               # $ per in-service GPU-hour, internal fleet rate (Sec. 9.8.2)
S_CEIL, R_CEIL, P_CEIL = 0.80, 0.98, 0.65  # lever ceilings (Sec. 9.8.1)
TRACE_MB_PER_GPU = 500    # 5 s Nsight Systems trace per GPU (Sec. 9.3.3)
COMPILE = dict(step_ms=8.33, mm=0.50, ew=0.35, launch=0.15, ew_cut=0.70,
               launch_cut=0.80, build_min=15, replicas=128)  # Napkin Math 9.6
TIERS = (500, 350, 280)   # allocated, active, productive GPUs (Sec. 9.2.3)


def show(slide, label, value):
    print(f"[{slide}] {label}: {value}")


# --- Segment 1 --------------------------------------------------------------
show("Predict: held capacity", "runtime loss", f"{1 - R_LLAMA:.2f}")
show("Predict: held capacity", "program loss", f"{R_LLAMA * (1 - P_LLAMA16):.2f}")
show("Predict: held capacity", "progress", f"{R_LLAMA * P_LLAMA16:.2f}")
show("Recall tiers", "productive / allocated", f"{TIERS[2] / TIERS[0]:.0%}")

# --- Segment 2 --------------------------------------------------------------
show("Contract", "days between faults at 128 GPUs", f"{MTBF128_H / 24:.1f}")
show("Trace volume", "1,024 GPUs x 5 s", f"{TRACE_MB_PER_GPU * 1024 / 1000:.0f} GB")
for key, name in [("gptj-99", "GPT-J"), ("llama2-70b-99.9", "Llama 2 70B"),
                  ("stable-diffusion-xl", "SDXL"), ("resnet50", "ResNet-50"),
                  ("bert-99", "BERT")]:
    pts = sorted((r["pub_month"], r["value"]) for r in
                 Literature.Series.MLPerfInferenceDGXH100.entries
                 if r["benchmark"] == key and r["scenario"] == "Offline")
    show("MLPerf predict", f"{name} gain", f"{pts[-1][1] / pts[0][1]:.2f}x")
show("Diagnose", "fleet step over node step", f"{T_FLEET / T_NODE:.2f}")

# --- Segment 3 --------------------------------------------------------------
S = ASSIGNED - PARTIAL
show("Snapshot", "S", f"{S:.2f}")
show("Snapshot", "ceiling with partial holds refilled", f"{S + PARTIAL:.2f}")
show("Runtime terms", "input waits / held", f"{INPUT_WAIT / S:.3f}")
show("Runtime terms", "startup / week", f"{T_START_MIN / (7 * 24 * 60):.4f}")
show("Runtime terms", "preemption, one a day", f"{PREEMPT_MIN / (24 * 60):.4f}")
show("Runtime terms", "recovery / MTBF", f"{T_REC_MIN / (MTBF128_H * 60):.4f}")
tau = math.sqrt(2 * T_WRITE_S * MTBF128_H * 3600)
show("Runtime terms", "Young-Daly interval (min)", f"{tau / 60:.0f}")
show("Runtime terms", "checkpoint waste",
     f"{T_WRITE_S / tau + tau / (2 * MTBF128_H * 3600):.4f}")
show("Runtime scaling", "startup share of a 1-hour job", f"{T_START_MIN / 60:.0%}")

# --- Segment 4 --------------------------------------------------------------
show("MFU bars", "PaLM chips", f"{PALM_CHIPS:,}")
show("MFU bars", "Llama 3 MFU drop, points", f"{100 * (P_LLAMA8 - P_LLAMA16):.0f}")
kept = T_NODE / T_FLEET
show("Node vs fleet", "node efficiency kept", f"{kept:.3f}")
show("Node vs fleet", "fleet MFU = 0.65 x kept", f"{P_NODE * kept:.3f}")
peak_work = 128 * H100_PEAK * 1e12 * T_FLEET / 1000 / 1e15
show("Node vs fleet", "peak work per step (PFLOP)", f"{peak_work:.1f}")
show("Node vs fleet", "fleet MFU direct", f"{USEFUL_PFLOP / peak_work:.3f}")
show("Node vs fleet", "time beyond node (ms)", f"{T_FLEET - T_NODE:.0f}")
show("Decode", "ridge (FLOP/byte)", f"{H100_PEAK / H100_BW:.0f}")
show("Decode", "batch-1 intensity (2 FLOPs / 2-byte weight)", f"{2 / 2:.1f}")
show("Decode", "rate at floor (TFLOP/s)", f"{H100_BW * 1.0:.2f}")
show("Decode", "share of peak", f"{H100_BW / H100_PEAK:.2%}")
show("Decode", "LayerNorm intensity", f"{LN_FLOPS / LN_BW:.1f}")
show("Decode", "LayerNorm share of HBM peak", f"{LN_BW / H100_BW:.1%}")

# --- Segment 5 --------------------------------------------------------------
P_JOB = 0.48             # fleet MFU of retained steps (Napkin Math 9.4)
mpg = S * R_JOB * P_JOB
show("Waterfall", "MPG", f"{mpg:.3f}")
show("Waterfall", "after runtime", f"{S * R_JOB:.3f}")
show("Waterfall", "loss shares S / R / P",
     f"{1 - S:.2f} / {S * (1 - R_JOB):.3f} / {S * R_JOB * (1 - 0.48):.3f}")
show("Waterfall", "job product R x P", f"{R_JOB * 0.48:.3f}")
for name, ratio in [("A scheduling", S_CEIL / S), ("B input pipeline", R_CEIL / 0.90),
                    ("C program", P_CEIL / 0.48)]:
    show("Levers", f"{name} gain and MPG", f"+{ratio - 1:.0%}, {mpg * ratio:.2f}")
show("Levers", "point worth S / R / P",
     f"{0.90 * 0.48:.2f} / {S * 0.48:.2f} / {S * 0.90:.2f}")
show("Cost", "useful GPU-hour", f"${PRICE / mpg:.2f}")
show("Cost", "multiple of raw price", f"{1 / mpg:.2f}")
show("Cost", "held-hours chargeback multiple", f"{1 / (R_JOB * 0.48):.2f}")
show("Cost", "cut at P = 0.65", f"{1 - 0.48 / P_CEIL:.0%}")
show("Iso-goodput", "P lever gain", f"{P_CEIL / 0.48 - 1:.0%}")
show("Iso-goodput", "break-even runtime loss", f"{1 - 0.48 / P_CEIL:.0%}")
show("Iso-goodput", "runtime headroom", f"{1 / 0.90 - 1:.0%}")
show("Iso-goodput", "trade 1.10 x 11/12", f"{1.10 * 11 / 12:.3f}")
c = COMPILE
frac = c["mm"] + c["ew"] * (1 - c["ew_cut"]) + c["launch"] * (1 - c["launch_cut"])
saved = c["step_ms"] * (1 - frac)
tokens = c["build_min"] * 60 / (saved / 1000)
gen_min = tokens * c["step_ms"] * frac / 1000 / 60  # at the compiled rate
show("Compile", "step fraction", f"{frac:.3f}")
show("Compile", "compiled step (ms)", f"{c['step_ms'] * frac:.1f}")
show("Compile", "saved per token (ms)", f"{saved:.1f}")
show("Compile", "tokens to repay", f"{tokens:,.0f}")
show("Compile", "minutes of compiled generation", f"{gen_min:.0f}")
show("Compile", "shared across replicas (s)", f"{gen_min * 60 / c['replicas']:.0f}")
show("Compile", "per-replica builds (GPU-hours)", f"{c['replicas'] * c['build_min'] / 60:.0f}")
thr = 2 * P_LLAMA16 / P_LLAMA8
show("Frontier", "throughput ratio 16K / 8K", f"{thr:.2f}")
show("Frontier", "time ratio", f"{1 / thr:.2f}")
show("Frontier", "GPU-hours per unit ratio", f"{P_LLAMA8 / P_LLAMA16:.3f}")

# --- Wrap-up ----------------------------------------------------------------
show("Synthesis", "Llama 3 on the shared pool", f"{S * R_LLAMA * P_LLAMA16:.2f}")
