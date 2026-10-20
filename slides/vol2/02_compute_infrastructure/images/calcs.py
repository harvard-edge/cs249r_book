#!/usr/bin/env python3
"""Every derived number on the Lecture 2 slides and notes.

Inputs MLSysIM holds come from MLSysIM; the rest are quoted from the rendered
chapter (Compute Infrastructure, dev build of 18 October 2026) with the
section in the comment. Each value is recorded with v() under the key the
deck uses as \\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Models, Systems, Literature = (
    mlsysim.Hardware, mlsysim.Models, mlsysim.Systems, mlsysim.Literature)

# --- Inputs from MLSysIM -----------------------------------------------------
_h100 = Hardware.Cloud.H100
H100_FP16 = _h100.compute.precision_flops["fp16_dense"].to("flop/s").magnitude
H100_FP8 = _h100.compute.precision_flops["fp8"].to("flop/s").magnitude
H100_HBM_BW = _h100.memory.bandwidth.to("B/s").magnitude
H100_HBM_GB = _h100.memory.capacity.magnitude      # nameplate figure, 80
PCIE = _h100.interconnect.bandwidth.to("B/s").magnitude
NVLINK = _h100.nvlink.bandwidth.to("B/s").magnitude / 2   # datasheet sums both directions
H200_HBM_BW = Hardware.Cloud.H200.memory.bandwidth.to("B/s").magnitude
A100_HBM_BW = Hardware.Cloud.A100.memory.bandwidth.to("B/s").magnitude
B200_HBM_BW = Hardware.Cloud.B200.memory.bandwidth.to("B/s").magnitude
IB = Systems.Fabrics.InfiniBand_NDR.bandwidth.to("B/s").magnitude
_gpus = {k: getattr(Hardware.Cloud, k) for k in ("V100", "A100", "H100", "B200")}
TDP = {k: g.tdp.to("W").magnitude for k, g in _gpus.items()}
FP16 = {k: g.compute.peak_flops.to("TFLOP/s").magnitude for k, g in _gpus.items()}
_llama = Models.Language.Llama3_70B
LLAMA70_PARAMS = _llama.parameters.magnitude
LLAMA70_WEIGHTS = 2 * LLAMA70_PARAMS                  # bytes, FP16
P175 = Models.Language.GPT3.parameters.magnitude
MTTF_H = Systems.Reliability.Gpu.mttf_hours
_tech = Hardware.Tech
HBM2_PJ_BIT = _tech.Memory.HBM2_AccessEnergy2017.energy_per_byte.to("pJ/B").magnitude / 8
DRAM_PJ_BIT = _tech.Memory.DRAM.energy_per_byte.to("pJ/B").magnitude / 8
MAC_PJ = (_tech.Op.FlopFp16.energy + _tech.Op.AddFp16.energy).to("pJ/flop").magnitude

# --- Inputs from the chapter (no MLSysIM entry) --------------------------------
DDR5_BW = 200e9           # B/s, table 5
HBM_PJ_BIT = 4            # pJ/bit, table 5
HIDDEN, MICRO, SEQ = 12288, 4, 2048   # Sec. 2.5.2
N_ALLREDUCE, COMPUTE_MS = 192, 200    # Sec. 2.5.2 (compute is a scenario input)
TOKENS = 300e9            # Napkin Math 2.4
N_POD, EFF_LOCAL, EFF_SCALE = 8192, 0.45, 0.56   # Napkin Math 2.4
OVERHEADS = (1.03, 1.10, 1.05)                   # Napkin Math 2.4
N_WSC = 10000             # Sec. 2.7.1

print("== Inputs shown on the slides")
v("h100_fp16", H100_FP16 / 1e12, "{:,.0f}", "H100 dense FP16, TFLOP/s")
v("h100_fp8", H100_FP8 / 1e12, "{:,.0f}", "H100 FP8, TFLOP/s")
v("h100_bw", H100_HBM_BW / 1e12, "{:.2f}", "H100 HBM, TB/s")
v("h100_bw_gbs", H100_HBM_BW / 1e9, "{:,.0f}", "H100 HBM, GB/s")
v("h100_hbm_gb", H100_HBM_GB, "{:.0f}", "H100 HBM capacity, GB (nameplate)")
v("h200_bw", H200_HBM_BW / 1e12, "{:.1f}", "H200 HBM, TB/s")
v("a100_bw", A100_HBM_BW / 1e12, "{:.2f}", "A100 HBM, TB/s")
v("b200_bw", B200_HBM_BW / 1e12, "{:.0f}", "B200 HBM, TB/s")
v("nvlink_gbs", NVLINK / 1e9, "{:.0f}", "NVLink 4 per direction, GB/s")
v("ib_gbs", IB / 1e9, "{:.0f}", "InfiniBand NDR per GPU, GB/s")
v("pcie_gbs", PCIE / 1e9, "{:.0f}", "PCIe Gen5 x16, GB/s")
v("llama70_params_b", LLAMA70_PARAMS / 1e9, "{:.1f}", "Llama-3 70B parameters, billions")
v("llama70_weights_gb", LLAMA70_WEIGHTS / 1e9, "{:.1f}", "Llama-3 70B FP16 weights, GB")
v("gpt3_params_b", P175 / 1e9, "{:.0f}", "running model, billions of parameters")
v("gpu_mttf_h", MTTF_H, "{:,.0f}", "per-GPU MTTF, h")
v("tdp_v100", TDP["V100"], "{:,.0f}", "V100 TDP, W")
v("tdp_h100", TDP["H100"], "{:,.0f}", "H100 TDP, W")
v("tdp_b200", TDP["B200"], "{:,.0f}", "B200 TDP, W")

print("== Hook (Napkin Math 2.1)")
flops_tok = 2 * LLAMA70_PARAMS
t_arith = flops_tok / H100_FP16 * 1e3
t_mem = LLAMA70_WEIGHTS / H100_HBM_BW * 1e3
v("hook_flops_mant", flops_tok / 1e11, "{:.2f}", "FLOPs per token, x 10^11")
v("hook_t_arith_ms", t_arith, "{:.2f}", "arithmetic time, ms")
v("hook_t_mem_ms", t_mem, "{:.1f}", "weight transfer time, ms")
v("hook_mem_share", 100 * t_mem / (t_mem + t_arith), "{:.1f}", "memory share of the sum, %")
v("hook_ratio", t_mem / t_arith, "{:.0f}", "memory / arithmetic ratio")

print("== Walls")
v("nvlink_ib_ratio", NVLINK / IB, "{:.0f}", "NVLink / InfiniBand per direction")
v("nvlink_ib_ratio_word", NVLINK / IB, "words", "the same, in prose")
v("pod_fail_h", MTTF_H / N_WSC, "{:.0f}", "hours between GPU failures at 10,000 GPUs")

print("== HBM")
v("hbm_ddr5_ratio", H100_HBM_BW / DDR5_BW, "{:.0f}", "HBM3 / DDR5 bandwidth")
v("width_ratio", 1024 / 64, "{:.0f}", "interface width ratio 1024 / 64")
v("dram_pj_bit", DRAM_PJ_BIT, "{:.0f}", "DRAM energy, pJ/bit")
v("energy_bit_ratio", DRAM_PJ_BIT / HBM_PJ_BIT, "{:.0f}", "energy per bit ratio")
v("hbm_operand_pj", 16 * HBM2_PJ_BIT, "{:.0f}", "one 16-bit operand from HBM2, pJ")
v("mac_pj", MAC_PJ, "{:.1f}", "FP16 multiply-add, pJ")
v("operand_mac_ratio", 16 * HBM2_PJ_BIT / MAC_PJ, "{:.0f}", "HBM operand / FP16 MAC energy")

print("== Two tests (70B FP16)")
v("weights_gpus", -(-LLAMA70_WEIGHTS // (H100_HBM_GB * 1e9)), "words", "H100s needed for weights (ceil)")
v("floor_a100_ms", LLAMA70_WEIGHTS / A100_HBM_BW * 1e3, "{:.1f}", "decode floor A100, ms")
v("floor_b200_ms", LLAMA70_WEIGHTS / B200_HBM_BW * 1e3, "{:.1f}", "decode floor B200, ms")
v("h200_bw_ratio", H200_HBM_BW / H100_HBM_BW, "{:.1f}", "H200 / H100 bandwidth")

print("== Roofline (H100 FP16)")
ridge = H100_FP16 / H100_HBM_BW
v("ridge", ridge, "{:.0f}", "ridge, FLOP/byte")
v("b1_pct", 100 * 1 / ridge, "{:.2f}", "batch 1 bound, % of peak")
v("b1_tflops", H100_HBM_BW * 1 / 1e12, "{:.2f}", "batch 1 bound, TFLOP/s")
v("b64_pct", 100 * 64 / ridge, "{:.0f}", "batch 64 bound, % of peak")

print("== Per watt (table 2)")
for k in TDP:
    v(f"perwatt_{k.lower()}", FP16[k] / TDP[k], "{:.2f}", f"{k} TFLOP/s per W")
v("v100_b200_peak", FP16["B200"] / FP16["V100"], "{:.0f}", "V100 to B200 FP16 peak")
v("v100_b200_tdp", TDP["B200"] / TDP["V100"], "{:.1f}", "V100 to B200 TDP")
v("v100_b200_perwatt", (FP16["B200"] / TDP["B200"]) / (FP16["V100"] / TDP["V100"]), "{:.1f}", "V100 to B200 per watt")
v("h100_fp8_perwatt_ratio", (H100_FP8 / 1e12 / TDP["H100"]) / (FP16["V100"] / TDP["V100"]), "{:.1f}",
  "H100 FP8 per watt / V100 FP16 per watt")

print("== Node memory (table 11, Example 2.4)")
v("node_hbm_gb", 8 * H100_HBM_GB, "{:.0f}", "DGX H100 HBM, GB (8 x nameplate)")
v("state_tb", 16 * P175 / 1e12, "{:.1f}", "persistent state, TB (16 B/param)")
w, g, master, adam = 2 * P175 / 1e9, 2 * P175 / 1e9, 4 * P175 / 1e9, 8 * P175 / 1e9
other = 375 - (adam + master + w + g) / 8   # 375 GB per GPU is the chapter's total
v("weights_gb", w, "{:.0f}", "FP16 weights, GB")
v("master_gb", master, "{:.0f}", "FP32 master weights, GB")
v("adam_moment_gb", adam / 2, "{:.0f}", "one Adam moment, GB")
v("adam_gb", adam, "{:,.0f}", "both Adam moments, GB")
v("tp8_adam_gb", adam / 8, "{:.0f}", "Adam moments per GPU, TP-8, GB")
v("tp8_master_gb", master / 8, "{:.1f}", "master per GPU, TP-8, GB")
v("tp8_wg_gb", (w + g) / 8, "{:.1f}", "weights + grads per GPU, TP-8, GB")
v("tp8_other_gb", other, "{:.0f}", "activations + workspace = 375 - rest, GB")
v("tp8_offload_gb", 375 - (adam + master) / 8, "{:.1f}", "after offloading moments and master, GB")
v("tp8_weights_gb", w / 8, "{:.2f}", "weights per GPU, TP-8, GB")
v("zero_adam_gb", adam / 1024, "{:.1f}", "Adam per GPU over 1,024, GB")
v("zero_adam_group_gb", adam / 1024 * 8, "{:.1f}", "Adam per 8-GPU group, GB")
v("zero_subtotal_gb", w / 8 + adam / 1024, "{:.1f}", "weights + Adam subtotal, GB")

print("== Staircase (Napkin Math 2.3)")
v("stair_hbm_ms", 10e9 / H100_HBM_BW * 1e3, "{:.1f}", "10 GB over HBM, ms")
v("stair_nvlink_ms", 10e9 / NVLINK * 1e3, "{:.1f}", "10 GB over NVLink, ms")
v("stair_ib_ms", 10e9 / IB * 1e3, "{:.0f}", "10 GB over InfiniBand, ms")

print("== Tensor-parallel AllReduce (Sec. 2.5.2, Example 2.3)")
payload = MICRO * SEQ * HIDDEN * 2
ring = payload * 2 * 7 / 8
t_nv, t_pc = ring / NVLINK * 1e3, ring / PCIE * 1e3
v("tp_payload_mb", payload / 1e6, "{:.0f}", "payload, MB")
v("tp_ring_mb", ring / 1e6, "{:.0f}", "ring traffic per GPU, MB")
v("tp_nvlink_ms", t_nv, "{:.3f}", "one AllReduce on NVLink, ms")
v("tp_nvlink_total_ms", N_ALLREDUCE * t_nv, "{:.1f}", "192 on NVLink, ms")
v("tp_nvlink_compute_pct", 100 * COMPUTE_MS / (COMPUTE_MS + N_ALLREDUCE * t_nv), "{:.0f}", "compute share, NVLink, %")
v("tp_nvlink_step_ms", COMPUTE_MS + N_ALLREDUCE * t_nv, "{:.0f}", "step on NVLink, ms")
v("tp_pcie_ms", t_pc, "{:.1f}", "one AllReduce on PCIe, ms")
v("tp_pcie_total_ms", N_ALLREDUCE * t_pc, "{:,.0f}", "192 on PCIe, ms")
v("tp_pcie_compute_pct", 100 * COMPUTE_MS / (COMPUTE_MS + N_ALLREDUCE * t_pc), "{:.0f}", "compute share, PCIe, %")
v("tp_pcie_step_ms", COMPUTE_MS + N_ALLREDUCE * t_pc, "{:,.0f}", "step on PCIe, ms")
v("tp_pcie_comm_pct", 100 * N_ALLREDUCE * t_pc / (COMPUTE_MS + N_ALLREDUCE * t_pc), "{:.1f}", "communication share, PCIe, %")

print("== Rack and power")
v("rack_gpu_kw", 32 * TDP["H100"] / 1e3, "{:.1f}", "32 GPUs x 700 W, kW")
v("cluster_mw", 32 * 33.5 / 1e3, "{:.1f}", "32 racks x 33.5 kW, MW")
_sys = [r for r in Literature.Series.AISupercomputers.entries if r["power_mw"]]
_last = max((r for r in _sys if r["first_operational_date"] <= "2025-12-31"), key=lambda r: r["power_mw"])
_front = next(r for r in _sys if "Frontier" in r["name"])
v("frontier_mw", _front["power_mw"], "{:.0f}", "Frontier, MW (Epoch AI via MLSysIM)")
v("largest_2025_mw", _last["power_mw"], "{:.0f}", "largest system by 2025, MW")
v("largest_over_frontier", _last["power_mw"] / _front["power_mw"], "{:.0f}", "largest 2025 / Frontier")
v("largest_over_frontier_word", round(_last["power_mw"] / _front["power_mw"]), "words", "the same, in prose")

print("== Transient (Sec. 2.6.1, scenario inputs)")
amps = (TDP["H100"] - 400) / 0.8
v("swing_w", TDP["H100"] - 400, "{:.0f}", "per-GPU power step, W")
v("swing_a", amps, "{:.0f}", "current step at 0.8 V, A")
for key, rise in (("1us", 1e-6), ("100ns", 100e-9)):
    dv = 50e-12 * amps / rise
    v(f"droop_{key}_mv", dv * 1e3, "{:.0f}", f"droop over {rise*1e9:.0f} ns, mV")
    v(f"droop_{key}_pct", 100 * dv / 0.8, "{:.1f}" if key == "1us" else "{:.0f}", f"droop over {rise*1e9:.0f} ns, % of 0.8 V")
step_kw = 1024 * (TDP["H100"] - 400) / 1e3
v("step_kw", step_kw, "{:.0f}", "1,024-GPU step, kW")
v("ramp_gws", step_kw * 1e3 / 100e-6 / 1e9, "{:.2f}", "ramp over 100 us, GW/s")

print("== Cooling (Sec. 2.6.2, equation 3)")
air_kg = 100e3 / (1005 * 15)
air_m3 = air_kg / 1.2
water_kg = 100e3 / (4180 * 15)
v("air_kgs", air_kg, "{:.1f}", "air mass flow, kg/s")
v("air_m3s", air_m3, "{:.1f}", "air volume flow, m3/s")
v("air_cfm", air_m3 * 2118.88, "{:,.0f}", "air volume flow, CFM")
v("water_kgs", water_kg, "{:.1f}", "water flow at 15 K, kg/s (about L/s)")
v("air_water_ratio", round(air_m3 * 1000 / water_kg, -2), "{:,.0f}", "air / water volume ratio, to the hundred")
v("nvl72_lpm", 120e3 / (4180 * 10) * 60, "{:.0f}", "GB200 NVL72 120 kW at 10 K, L/min")

print("== Failure domains (Sec. 2.6.4, $2 per GPU-hour)")
v("cdu_cost", 256 * 4 * 2, "{:,.0f}", "256 GPUs x 4 h x $2")
v("job_cost", N_WSC * 4 * 2, "{:,.0f}", "10,000 GPUs x 4 h x $2")

print("== Pod")
v("pod_eflops", N_WSC * H100_FP16 / 1e18, "{:.2f}", "10,000 x 989 TFLOP/s, EFLOP/s")
v("pod_tbs", N_WSC * IB / 1e12, "{:.0f}", "10,000 x 50 GB/s, TB/s")
v("pod_mw", N_WSC * TDP["H100"] / 1e6, "{:.0f}", "10,000 x 700 W, MW")
v("run_failures", 336 / (MTTF_H / N_WSC), "{:.0f}", "GPU failures in 336 h at 5 h")
v("node_fail_day", 128 * 24 / 1000, "{:.0f}", "node failures per day")
v("eta_gpuh", 1000 / 0.4, "{:,.0f}", "1,000 GPU-h at eta 0.4")

print("== Pod chain (Napkin Math 2.4)")
flops = 6 * P175 * TOKENS
peak = N_POD * H100_FP8
h_peak = flops / peak / 3600
h = h_peak / EFF_LOCAL
v("chain_flops_mant", flops / 1e23, "{:.2f}", "training FLOPs, x 10^23")
v("chain_peak_h", h_peak, "{:.1f}", "hours at 100% of FP8 peak")
v("chain_compute_h", h, "{:.1f}", "compute-only baseline, h")
h /= EFF_SCALE
v("chain_scaled_h", h, "{:.1f}", "after scaling efficiency, h")
for name, f in zip(("ckpt", "fail"), OVERHEADS[:2]):
    h *= f
    v(f"chain_{name}_h", h, "{:.1f}", f"after {name}, h")
h *= OVERHEADS[2]
v("chain_total_h", h, "{:.1f}", "after checkpoint, failures, maintenance, h")
v("chain_share_pct", 100 * h_peak / h, "{:.0f}", "share of FP8 peak delivered, %")

print("== Fallacies (Sec. 2.10)")
v("fallacy_speed", 8 * 0.56 / 0.83, "{:.1f}", "speed from 1,024 to 8,192 GPUs")
v("fallacy_lost_pct", 100 * (1 - 0.56), "{:.0f}", "share lost to communication and bubbles, %")
v("h100_v100_peak", FP16["H100"] / FP16["V100"], "{:.1f}", "H100 / V100 FP16 peak")
v("v100_in_h100_gpuh", 10000 * FP16["V100"] / FP16["H100"], "{:,.0f}", "10,000 V100 GPU-h in H100 GPU-h")
v("int8_floor_ms", 71e9 / H100_HBM_BW * 1e3, "{:.0f}", "71 GB INT8 at 3.35 TB/s, ms")

print("== Rack title")
v("dgx_rack_ratio", 33.5 / 12, "{:.0f}", "DGX H100 rack / server rack")
v("nvl72_rack_ratio", 120 / 12, "{:.0f}", "GB200 NVL72 rack / server rack")

if __name__ == "__main__":
    write(__file__)
