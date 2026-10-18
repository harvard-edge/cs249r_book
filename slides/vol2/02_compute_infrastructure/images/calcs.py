#!/usr/bin/env python3
"""Every derived number on the Lecture 2 slides and notes, computed from
sourced inputs. Run from this folder: python3 calcs.py

Inputs are quoted from the rendered chapter (Compute Infrastructure, dev
build of 18 October 2026) unless noted. Each printed line carries the label
the slide or note uses; the deck copies these values.
"""

# --- Inputs (chapter section in the comment) ---------------------------------
H100_FP16 = 989e12        # FLOP/s dense FP16, Sec. 2.2.5
H100_FP8 = 1979e12        # FLOP/s dense FP8, table 1
H100_HBM_BW = 3.35e12     # B/s, Sec. 2.3.1
H100_HBM_GB = 80          # GB nameplate, Sec. 2.3.1
H200_HBM_BW = 4.8e12      # B/s, Napkin Math 2.2
A100_HBM_BW = 2.04e12     # B/s, Sec. 2.3.1
B200_HBM_BW = 8e12        # B/s, Sec. 2.3.1
DDR5_BW = 200e9           # B/s, table 5
LLAMA70_PARAMS = 70e9     # Napkin Math 2.1
LLAMA70_WEIGHTS = 141.2e9 # B, FP16, Napkin Math 2.1
HBM_PJ_BIT, DRAM_PJ_BIT = 4, 20   # table 5
HBM2_PJ_BIT, MAC_PJ = 3.97, 1.5  # Sec. 2.3.1 (O'Connor et al. 2017; Horowitz 2014)
TDP = {"V100": 300, "A100": 400, "H100": 700, "B200": 1000}        # W, table 2
FP16 = {"V100": 125, "A100": 312, "H100": 989, "B200": 2250}       # TFLOP/s, table 2
NVLINK, IB, PCIE = 450e9, 50e9, 64e9  # B/s per direction, table 9
P175 = 175e9              # parameters, running model
HIDDEN, MICRO, SEQ = 12288, 4, 2048   # Sec. 2.5.2
N_ALLREDUCE, COMPUTE_MS = 192, 200    # Sec. 2.5.2 (compute is a scenario input)
TOKENS = 300e9            # Napkin Math 2.4
N_POD, EFF_LOCAL, EFF_SCALE = 8192, 0.45, 0.56   # Napkin Math 2.4
OVERHEADS = (1.03, 1.10, 1.05)                   # Napkin Math 2.4
MTTF_H, N_WSC = 50000, 10000                      # Sec. 2.7.1


def p(label, value, fmt="{:,.3g}"):
    print(f"{label:58s} {fmt.format(value)}")


print("== Hook (Napkin Math 2.1)")
flops_tok = 2 * LLAMA70_PARAMS
t_arith = flops_tok / H100_FP16 * 1e3
t_mem = LLAMA70_WEIGHTS / H100_HBM_BW * 1e3
p("FLOPs per token", flops_tok, "{:.2e}")
p("arithmetic time, ms", t_arith, "{:.2f}")
p("weight transfer time, ms", t_mem, "{:.1f}")
p("memory share of the sum, %", 100 * t_mem / (t_mem + t_arith), "{:.1f}")
p("memory / arithmetic ratio", t_mem / t_arith, "{:.0f}")

print("== Walls")
p("NVLink / InfiniBand per direction", NVLINK / IB, "{:.0f}")
p("hours between GPU failures at 10,000 GPUs", MTTF_H / N_WSC, "{:.0f}")

print("== HBM")
p("HBM3 / DDR5 bandwidth", H100_HBM_BW / DDR5_BW, "{:.1f}")
p("interface width ratio 1024 / 64", 1024 / 64, "{:.0f}")
p("energy per bit ratio", DRAM_PJ_BIT / HBM_PJ_BIT, "{:.0f}")
p("one 16-bit operand from HBM2, pJ", 16 * HBM2_PJ_BIT, "{:.0f}")
p("HBM operand / FP16 MAC energy", 16 * HBM2_PJ_BIT / MAC_PJ, "{:.0f}")

print("== Two tests (70B FP16)")
p("H100s needed for weights (ceil)", -(-LLAMA70_WEIGHTS // (H100_HBM_GB * 1e9)), "{:.0f}")
for name, bw in (("A100", A100_HBM_BW), ("H100", H100_HBM_BW), ("B200", B200_HBM_BW)):
    p(f"decode floor {name}, ms", LLAMA70_WEIGHTS / bw * 1e3, "{:.1f}")
p("H200 / H100 bandwidth", H200_HBM_BW / H100_HBM_BW, "{:.2f}")

print("== Roofline (H100 FP16)")
ridge = H100_FP16 / H100_HBM_BW
p("ridge, FLOP/byte", ridge, "{:.1f}")
p("batch 1 bound, % of peak", 100 * 1 / ridge, "{:.2f}")
p("batch 1 bound, TFLOP/s", H100_HBM_BW * 1 / 1e12, "{:.2f}")
p("batch 64 bound, % of peak", 100 * 64 / ridge, "{:.1f}")

print("== Per watt (table 2)")
for k in TDP:
    p(f"{k} TFLOP/s per W", FP16[k] / TDP[k], "{:.2f}")
p("V100 to B200 FP16 peak", FP16["B200"] / FP16["V100"], "{:.0f}")
p("V100 to B200 TDP", TDP["B200"] / TDP["V100"], "{:.1f}")
p("V100 to B200 per watt", (FP16["B200"] / TDP["B200"]) / (FP16["V100"] / TDP["V100"]), "{:.1f}")

print("== Node memory (table 11, Example 2.4)")
p("persistent state, TB (16 B/param)", 16 * P175 / 1e12, "{:.1f}")
w, g, master, adam = 2 * P175 / 1e9, 2 * P175 / 1e9, 4 * P175 / 1e9, 8 * P175 / 1e9
p("Adam moments per GPU, TP-8, GB", adam / 8, "{:.1f}")
p("master per GPU, TP-8, GB", master / 8, "{:.1f}")
p("weights + grads per GPU, TP-8, GB", (w + g) / 8, "{:.1f}")
p("activations + workspace = 375 - rest, GB", 375 - (adam + master + w + g) / 8, "{:.1f}")
p("after offloading moments and master, GB", 375 - (adam + master) / 8, "{:.1f}")
p("weights per GPU, TP-8, GB", w / 8, "{:.2f}")
p("Adam per GPU over 1,024, GB", adam / 1024, "{:.2f}")
p("Adam per 8-GPU group, GB", adam / 1024 * 8, "{:.1f}")
p("weights + Adam subtotal, GB", w / 8 + adam / 1024, "{:.1f}")
p("GPU-only H100 weight total check, GB", 32 * 700 / 1000, "{:.1f}")

print("== Staircase (Napkin Math 2.3)")
for name, bw in (("HBM", H100_HBM_BW), ("NVLink", NVLINK), ("InfiniBand", IB)):
    p(f"10 GB over {name}, ms", 10e9 / bw * 1e3, "{:.1f}")

print("== Tensor-parallel AllReduce (Sec. 2.5.2, Example 2.3)")
payload = MICRO * SEQ * HIDDEN * 2
ring = payload * 2 * 7 / 8
p("payload, MB", payload / 1e6, "{:.0f}")
p("ring traffic per GPU, MB", ring / 1e6, "{:.0f}")
t_nv, t_pc = ring / NVLINK * 1e3, ring / PCIE * 1e3
p("one AllReduce on NVLink, ms", t_nv, "{:.3f}")
p("192 on NVLink, ms", N_ALLREDUCE * t_nv, "{:.1f}")
p("compute share, NVLink, %", 100 * COMPUTE_MS / (COMPUTE_MS + N_ALLREDUCE * t_nv), "{:.1f}")
p("step on NVLink, ms", COMPUTE_MS + N_ALLREDUCE * t_nv, "{:.0f}")
p("one AllReduce on PCIe, ms", t_pc, "{:.2f}")
p("192 on PCIe, ms", N_ALLREDUCE * t_pc, "{:.0f}")
p("communication share, PCIe, %", 100 * N_ALLREDUCE * t_pc / (COMPUTE_MS + N_ALLREDUCE * t_pc), "{:.1f}")
p("step on PCIe, ms", COMPUTE_MS + N_ALLREDUCE * t_pc, "{:,.0f}")

print("== Rack and power")
p("32 GPUs x 700 W, kW", 32 * 700 / 1e3, "{:.1f}")
p("32 racks x 33.5 kW, MW", 32 * 33.5 / 1e3, "{:.2f}")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "mlsysim"))
from mlsysim import Literature  # noqa: E402
_sys = [r for r in Literature.Series.AISupercomputers.entries if r["power_mw"]]
_last = max((r for r in _sys if r["first_operational_date"] <= "2025-12-31"), key=lambda r: r["power_mw"])
_front = next(r for r in _sys if "Frontier" in r["name"])
p("Frontier, MW (Epoch AI via MLSysIM)", _front["power_mw"], "{:.0f}")
p("largest system by 2025, MW", _last["power_mw"], "{:.0f}")
p("largest 2025 / Frontier", _last["power_mw"] / _front["power_mw"], "{:.0f}")
p("package 1,400 / 700 W", 1400 / 700, "{:.0f}")

print("== Transient (Sec. 2.6.1, scenario inputs)")
amps = (700 - 400) / 0.8
p("current step at 0.8 V, A", amps, "{:.0f}")
for rise in (1e-6, 100e-9):
    dv = 50e-12 * amps / rise
    p(f"droop over {rise*1e9:.0f} ns, mV", dv * 1e3, "{:.0f}")
    p(f"droop over {rise*1e9:.0f} ns, % of 0.8 V", 100 * dv / 0.8, "{:.1f}")
step_kw = 1024 * 300 / 1e3
p("1,024-GPU step, kW", step_kw, "{:.0f}")
p("ramp over 100 us, GW/s", step_kw * 1e3 / 100e-6 / 1e9, "{:.2f}")

print("== Cooling (Sec. 2.6.2, equation 3)")
air_kg = 100e3 / (1005 * 15)
air_m3 = air_kg / 1.2
water_kg = 100e3 / (4180 * 15)
p("air mass flow, kg/s", air_kg, "{:.1f}")
p("air volume flow, m3/s", air_m3, "{:.1f}")
p("air volume flow, CFM", air_m3 * 2118.88, "{:,.0f}")
p("water flow at 15 K, kg/s (about L/s)", water_kg, "{:.1f}")
p("air / water volume ratio", air_m3 * 1000 / water_kg, "{:,.0f}")
p("GB200 NVL72 120 kW at 10 K, L/min", 120e3 / (4180 * 10) * 60, "{:.0f}")

print("== Failure domains (Sec. 2.6.4, $2 per GPU-hour)")
p("256 GPUs x 4 h x $2", 256 * 4 * 2, "{:,.0f}")
p("10,000 GPUs x 4 h x $2", 10000 * 4 * 2, "{:,.0f}")

print("== Pod")
p("10,000 x 989 TFLOP/s, EFLOP/s", N_WSC * H100_FP16 / 1e18, "{:.2f}")
p("10,000 x 50 GB/s, TB/s", N_WSC * IB / 1e12, "{:.0f}")
p("10,000 x 700 W, MW", N_WSC * 700 / 1e6, "{:.0f}")
p("GPU failures in 336 h at 5 h", 336 / (MTTF_H / N_WSC), "{:.0f}")
p("node failures, 128 nodes, 1,000 h MTBF, 336 h", 128 * 336 / 1000, "{:.0f}")
p("node failures per day", 128 * 24 / 1000, "{:.1f}")
p("1,000 GPU-h at eta 0.4", 1000 / 0.4, "{:,.0f}")

print("== Pod chain (Napkin Math 2.4)")
flops = 6 * P175 * TOKENS
peak = N_POD * H100_FP8
h_peak = flops / peak / 3600
h = h_peak / EFF_LOCAL
p("training FLOPs", flops, "{:.3g}")
p("hours at 100% of FP8 peak", h_peak, "{:.1f}")
p("compute-only baseline, h", h, "{:.1f}")
h /= EFF_SCALE
p("after scaling efficiency, h", h, "{:.1f}")
for name, f in zip(("checkpoint", "failures", "maintenance"), OVERHEADS):
    h *= f
    p(f"after {name}, h", h, "{:.1f}")
p("share of FP8 peak delivered, %", 100 * h_peak / h, "{:.0f}")

print("== Fallacies (Sec. 2.10)")
p("speed from 1,024 to 8,192 GPUs", 8 * 0.56 / 0.83, "{:.1f}")
p("H100 / V100 FP16 peak", 989 / 125, "{:.1f}")
p("10,000 V100 GPU-h in H100 GPU-h", 10000 * 125 / 989, "{:,.0f}")
p("71 GB INT8 at 3.35 TB/s, ms", 71e9 / H100_HBM_BW * 1e3, "{:.0f}")
p("141.2 GB FP16 at 2.04 TB/s, ms", LLAMA70_WEIGHTS / A100_HBM_BW * 1e3, "{:.1f}")
p("embedding 10 TB / 687 GB per node (ceil)", -(-10000 // 687), "{:.0f}")

print("== Rack title")
p("DGX H100 rack / server rack", 33.5 / 12, "{:.1f}")
p("GB200 NVL72 rack / server rack", 120 / 12, "{:.0f}")
p("communication share, NVLink, %", 100 * N_ALLREDUCE * t_nv / (COMPUTE_MS + N_ALLREDUCE * t_nv), "{:.1f}")
