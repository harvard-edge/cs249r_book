#!/usr/bin/env python3
"""Every number on the Lecture 2 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry) and the calculations (Scenarios.*, each
backed by a solver); the chapter reads the same scenarios. This file only
selects values, converts units, and formats them; it types no numbers. Each
value is recorded with v() under the key the deck prints as \\val{key};
running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Models, Systems = mlsysim.Hardware, mlsysim.Models, mlsysim.Systems
Infrastructure, Literature, Scenarios = mlsysim.Infrastructure, mlsysim.Literature, mlsysim.Scenarios
SCALE = mlsysim.ReferenceStats.TrainingScaleProfiles
TECH = Hardware.Tech
H100, A100, B200, H200, V100 = (Hardware.Cloud.H100, Hardware.Cloud.A100, Hardware.Cloud.B200,
                                Hardware.Cloud.H200, Hardware.Cloud.V100)
NVL72 = Hardware.Cloud.GB200_NVL72
DGX, RACK = Systems.Nodes.DGX_H100, Systems.Racks.DGX_H100_4Node
REF_CLUSTER, POD = Systems.Clusters.Training_1K, Systems.Clusters.Training_10K

DECODE = Scenarios.Llama3_70B_DecodeH100
LADDER = Scenarios.BufferTransfer10GB
TP = Scenarios.Gpt3TensorParallelStepDgxH100
RUN = Scenarios.Gpt3PretrainingFrontier8K
STATE = Scenarios.Gpt3TrainingStateTp8
STEP = Scenarios.H100BarrierLoadStep
RACK_AIR = Scenarios.AiRackAirCooling
CDU = Scenarios.CduOutageTraining10K
GPU_FAIL = Scenarios.GpuFailuresTraining10K
NODE_FAIL = Scenarios.NodeFailuresTraining1K
BUDGET = Scenarios.PeakPlannedTrainingBudget


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


print("== Inputs shown on the slides")
v("h100_fp16", m(H100.compute.peak_flops, "TFLOP/s"), "{:,.0f}", "H100 dense FP16, TFLOP/s")
v("h100_fp8", m(H100.compute.precision_flops["fp8"], "TFLOP/s"), "{:,.0f}", "H100 FP8, TFLOP/s")
v("h100_bw", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM, TB/s")
v("h100_bw_gbs", m(H100.memory.bandwidth, "GB/s"), "{:,.0f}", "H100 HBM, GB/s")
v("h100_hbm_gb", H100.memory.capacity.magnitude, "{:.0f}", "H100 HBM, labeled GB (80 GiB)")
v("h200_bw", m(H200.memory.bandwidth, "TB/s"), "{:.1f}", "H200 HBM, TB/s")
v("a100_bw", m(A100.memory.bandwidth, "TB/s"), "{:.2f}", "A100 HBM, TB/s")
v("b200_bw", m(B200.memory.bandwidth, "TB/s"), "{:.0f}", "B200 HBM, TB/s")
v("nvlink_gbs", m(LADDER.links["NVLink"], "GB/s"), "{:.0f}", "NVLink 4 per direction, GB/s")
v("ib_gbs", m(LADDER.links["InfiniBand"], "GB/s"), "{:.0f}", "InfiniBand NDR per GPU, GB/s")
v("pcie_gbs", m(LADDER.links["PCIe"], "GB/s"), "{:.0f}", "PCIe Gen5 x16, GB/s")
v("llama70_params_b", m(DECODE.workload.parameters, "Gparam"), "{:.1f}", "Llama-3 70B parameters, billions")
v("llama70_weights_gb", m(DECODE.result.weight_bytes, "GB"), "{:.1f}", "Llama-3 70B FP16 weights, GB")
v("gpt3_params_b", m(Models.Language.GPT3.parameters, "Gparam"), "{:.0f}", "running model, billions of parameters")
v("gpu_mttf_h", m(GPU_FAIL.component_mtbf, "hour"), "{:,.0f}", "per-GPU MTTF, h")
v("tdp_v100", m(V100.tdp, "W"), "{:,.0f}", "V100 TDP, W")
v("tdp_h100", m(H100.tdp, "W"), "{:,.0f}", "H100 TDP, W")
v("tdp_b200", m(B200.tdp, "W"), "{:,.0f}", "B200 TDP, W")

print("== Scenario inputs shown on the slides")
v("buffer_gb", m(LADDER.payload, "GB"), "{:.0f}", "buffer moved across the tiers, GB")
v("tp_compute_ms", m(TP.compute_time, "ms"), "{:.0f}", "stated compute per TP step, ms")
v("tp_collectives", TP.result.collectives_per_step, "{:.0f}", "AllReduces per TP step")
v("tp_microbatch", TP.microbatch, "{:.0f}", "sequences per microbatch")
v("tp_seq", TP.workload.context_length, "{:,.0f}", "tokens per sequence")
v("tp_hidden", TP.workload.hidden_dim, "{:,.0f}", "hidden dimension")
v("tp_ring_factor", TP.result.ring_factor, "{:.2f}", "ring AllReduce factor 2(N-1)/N")
v("step_low_w", m(STEP.low_power, "W"), "{:.0f}", "communication-phase power, W")
v("rail_v", m(STEP.rail_voltage, "V"), "{:.1f}", "core-rail voltage, V")
v("loop_ph", m(STEP.loop_inductance, "pH"), "{:.0f}", "board loop inductance, pH")
v("rise_slow_us", m(STEP.rise_times["slow"], "microsecond"), "{:.0f}", "slow edge, us")
v("rise_fast_ns", m(STEP.rise_times["fast"], "nanosecond"), "{:.0f}", "fast edge, ns")
v("ramp_us", m(STEP.ramp_window, "microsecond"), "{:.0f}", "cluster ramp window, us")
v("ref_gpus", REF_CLUSTER.total_accelerators, "{:,.0f}", "reference cluster GPUs")
v("ref_nodes", REF_CLUSTER.count, "{:.0f}", "reference cluster nodes")
v("cool_rise_k", m(RACK_AIR.temperature_rise, "K"), "{:.0f}", "air temperature rise, K")
v("air_cp", m(mlsysim.physics.constants.AIR_SPECIFIC_HEAT, "J/(kg*K)"), "{:,.0f}", "air specific heat, J/(kg K)")
v("air_rho", m(mlsysim.physics.constants.AIR_DENSITY_SEA_LEVEL, "kg/m**3"), "{:.1f}", "air density, kg/m3")
v("water_cp", m(mlsysim.physics.constants.WATER_SPECIFIC_HEAT, "J/(kg*K)"), "{:,.0f}", "water specific heat, J/(kg K)")
v("rack_heat_kw", m(RACK_AIR.heat_load, "kW"), "{:.0f}", "high-density rack heat load, kW")
v("rack_heat_w", m(RACK_AIR.heat_load, "W"), "{:,.0f}", "the same, W")
v("nvl72_kw", m(NVL72.tdp, "kW"), "{:.0f}", "GB200 NVL72 rack power, kW")
v("nvl72_rise_k", m(Scenarios.Nvl72WaterCooling.temperature_rise, "K"), "{:.0f}", "NVL72 coolant rise, K")
v("trad_rack_kw", Infrastructure.Racks.Traditional.power_kw, "{:.0f}", "enterprise rack, kW")
v("dgx_rack_kw", m(RACK.total_power, "kW"), "{:.1f}", "four-node DGX H100 rack, kW")
v("cdu_gpus", CDU.failed_devices, "{:.0f}", "GPUs behind the failed CDU")
v("cdu_repair_h", m(CDU.repair_time, "hour"), "{:.0f}", "CDU repair window, h")
v("gpu_hour_usd", m(CDU.device_hour_rate, "dollar/hour"), "{:.0f}", "fleet GPU-hour, $")
v("pod_gpus", POD.total_accelerators, "{:,.0f}", "pod GPUs")
v("run_h", m(GPU_FAIL.run_duration, "hour"), "{:.0f}", "two-week run, h")
v("node_mtbf_h", m(NODE_FAIL.component_mtbf, "hour"), "{:,.0f}", "composite node MTBF, h")
v("budget_gpuh", BUDGET.planned_device_hours, "{:,.0f}", "GPU-hours planned from peak")
v("sustained", BUDGET.sustained_fraction, "{:.1f}", "sustained share of peak")
v("sustained_pct", 100 * BUDGET.sustained_fraction, "{:.0f}", "the same, percent")
v("stretch", BUDGET.stretch, "{:.1f}", "1 / sustained share")
v("chain_tokens_b", m(RUN.workload.training_tokens, "count") / 10 ** 9, "{:.0f}", "training tokens, billions")
v("pod8k_gpus", RUN.fleet.total_accelerators, "{:,.0f}", "8,192-GPU pod")
v("local_eff", RUN.local_efficiency, "{:.2f}", "local compute efficiency")
v("local_eff_pct", 100 * RUN.local_efficiency, "{:.0f}", "the same, percent")
v("scale_eff", RUN.scaling_efficiency, "{:.2f}", "scaling efficiency at 8,192 GPUs")
v("scale_eff_1k", SCALE.Eff1024Gpu, "{:.2f}", "scaling efficiency at 1,024 GPUs")
v("ovh_ckpt", 1 + RUN.overheads["checkpoint"], "{:.2f}", "checkpoint factor")
v("ovh_fail", 1 + RUN.overheads["failures"], "{:.2f}", "failure factor")
v("ovh_maint", 1 + RUN.overheads["maintenance"], "{:.2f}", "maintenance factor")
v("ovh_ckpt_pct", 100 * RUN.overheads["checkpoint"], "{:.0f}", "checkpoint allowance, %")
v("ovh_fail_pct", 100 * RUN.overheads["failures"], "{:.0f}", "failure allowance, %")
v("ovh_maint_pct", 100 * RUN.overheads["maintenance"], "{:.0f}", "maintenance allowance, %")
v("v100_job_gpuh", Scenarios.V100ToH100.baseline_device_hours, "{:,.0f}", "job size on V100s, GPU-h")
tpu = Hardware.Cloud.TPUv1
v("tpu_v1_tops", m(tpu.compute.precision_flops["int8"], "TOPS"), "{:.0f}", "TPU v1 INT8, TOPS")
v("tpu_v1_w", m(tpu.tdp, "W"), "{:.0f}", "TPU v1 TDP, W")
mix = Models.Language.Mixtral_8x7B
v("mixtral_active_b", m(mix.active_parameters, "Gparam"), "{:.1f}", "Mixtral-8x7B active parameters, B")
v("mixtral_total_b", m(mix.parameters, "Gparam"), "{:.1f}", "Mixtral-8x7B total parameters, B")

print("== Hook (decode scenario)")
d = DECODE.result
v("hook_flops_mant", m(d.flops_per_step, "flop") / 10 ** 11, "{:.2f}", "FLOPs per token, x 10^11")
v("hook_t_arith_ms", m(d.compute_time, "ms"), "{:.2f}", "arithmetic time, ms")
v("hook_t_mem_ms", m(d.memory_time, "ms"), "{:.1f}", "weight transfer time, ms")
v("hook_mem_share", 100 * d.memory_share, "{:.1f}", "memory share of the sum, %")
v("hook_ratio", d.bound_ratio, "{:.0f}", "memory / arithmetic ratio")

print("== Walls")
nv_ib = ratio(LADDER.links["NVLink"], LADDER.links["InfiniBand"])
v("nvlink_ib_ratio", nv_ib, "{:.0f}", "NVLink / InfiniBand per direction")
v("nvlink_ib_ratio_word", nv_ib, "words", "the same, in prose")
v("pod_fail_h", m(GPU_FAIL.result.interval, "hour"), "{:.0f}", "hours between GPU failures at 10,000 GPUs")

print("== HBM")
hbm_e, dram_e = TECH.Memory.HBM2_AccessEnergy2017.energy_per_byte, TECH.Memory.DRAM.energy_per_byte
operand = (mlsysim.BYTES_FP16 * hbm_e).to("pJ")
mac = ((TECH.Op.FlopFp16.energy + TECH.Op.AddFp16.energy) * mlsysim.flop).to("pJ")
v("hbm_ddr5_ratio", ratio(H100.memory.bandwidth, TECH.Storage.HostDram.bandwidth), "{:.0f}", "HBM3 / DDR5 bandwidth")
v("width_ratio", ratio(TECH.Memory.HBM3.interface_width, TECH.Memory.DDR5.interface_width), "{:.0f}",
  "HBM3 stack / DDR5 channel interface width")
v("dram_pj_bit", m(dram_e, "pJ/bit"), "{:.0f}", "DRAM energy, pJ/bit")
v("energy_bit_ratio", ratio(dram_e, hbm_e), "{:.0f}", "DRAM / HBM energy per bit")
v("hbm_operand_pj", m(operand, "pJ"), "{:.0f}", "one 16-bit operand from HBM2, pJ")
v("mac_pj", m(mac, "pJ"), "{:.1f}", "FP16 multiply-add, pJ")
v("operand_mac_ratio", ratio(operand, mac), "{:.0f}", "HBM operand / FP16 MAC energy")

print("== Two tests (decode scenario on other accelerators)")
v("weights_gpus", math.ceil(ratio(d.weight_bytes, H100.memory.capacity)), "words", "H100s needed for weights (ceil)")
v("floor_a100_ms", m(DECODE.variant(hardware=A100).result.memory_time, "ms"), "{:.1f}", "decode floor A100, ms")
v("floor_b200_ms", m(DECODE.variant(hardware=B200).result.memory_time, "ms"), "{:.1f}", "decode floor B200, ms")
srv = Scenarios.Llama3_70B_ServingH100
v("h200_cap_ratio", ratio(H200.memory.capacity, H100.memory.capacity), "{:.1f}", "H200 / H100 labeled HBM capacity")
v("h200_speedup", ratio(srv.result.itl, srv.variant(hardware=H200).result.itl), "{:.1f}", "H200 / H100 serving-model decode speed")
v("h200_bw_ratio", ratio(H200.memory.bandwidth, H100.memory.bandwidth), "{:.1f}", "H200 / H100 bandwidth")

print("== Roofline (decode scenario at batch 1 and 64)")
v("ridge", m(H100.ridge_point(), "flop/byte"), "{:.0f}", "ridge, FLOP/byte")
v("b1_pct", 100 * d.peak_fraction, "{:.2f}", "batch 1 bound, % of peak")
v("b1_tflops", m(d.attainable_flops, "TFLOP/s"), "{:.2f}", "batch 1 bound, TFLOP/s")
v("b64_pct", 100 * Scenarios.Llama3_70B_DecodeH100Batch64.result.peak_fraction, "{:.0f}", "batch 64 bound, % of peak")

print("== Per watt")
for name, hw in (("v100", V100), ("a100", A100), ("h100", H100), ("b200", B200)):
    v(f"perwatt_{name}", m(hw.compute.peak_flops / hw.tdp, "TFLOP/s/W"), "{:.2f}", f"{name.upper()} TFLOP/s per W")
gen = Scenarios.V100ToB200
v("v100_b200_peak", gen.peak_ratio, "{:.0f}", "V100 to B200 FP16 peak")
v("v100_b200_tdp", gen.tdp_ratio, "{:.1f}", "V100 to B200 TDP")
v("v100_b200_perwatt", gen.efficiency_ratio, "{:.1f}", "V100 to B200 per watt")
v("h100_fp8_perwatt_ratio",
  ratio(H100.compute.precision_flops["fp8"] / H100.tdp, V100.compute.peak_flops / V100.tdp),
  "{:.1f}", "H100 FP8 per watt / V100 FP16 per watt")

print("== Node memory (training-state scenario)")
s = STATE.result
v("node_hbm_gb", DGX.accelerators_per_node * H100.memory.capacity.magnitude, "{:.0f}", "DGX H100 HBM, labeled GB")
table = mlsysim.ReferenceStats.RecommenderAtScaleProfile.EmbeddingTableSize
node_hbm = DGX.accelerators_per_node * H100.memory.capacity
v("rec_table_tb", m(table, "TB"), "{:.0f}", "recommender embedding table, TB")
v("node_hbm_decimal_gb", m(node_hbm, "GB"), "{:.0f}", "DGX H100 HBM in decimal bytes, GB")
v("rec_min_nodes", math.ceil(ratio(table, node_hbm)), "{:.0f}", "nodes to hold the table in HBM")
v("state_tb", m(s.persistent_state, "TB"), "{:.1f}", "persistent state, TB")
v("weights_gb", m(s.weights, "GB"), "{:.0f}", "FP16 weights, GB")
v("master_gb", m(s.master_weights, "GB"), "{:.0f}", "FP32 master weights, GB")
v("adam_moment_gb", m(s.adam_moment, "GB"), "{:.0f}", "one Adam moment, GB")
v("adam_gb", m(s.optimizer_moments, "GB"), "{:,.0f}", "both Adam moments, GB")
v("tp8_adam_gb", m(s.tp_moments_per_device, "GB"), "{:.0f}", "Adam moments per GPU, TP-8, GB")
v("tp8_master_gb", m(s.tp_master_per_device, "GB"), "{:.1f}", "master per GPU, TP-8, GB")
v("tp8_wg_gb", m(s.tp_weights_per_device + s.tp_gradients_per_device, "GB"), "{:.1f}",
  "weights + grads per GPU, TP-8, GB")
v("tp8_other_gb", m(s.tp_activations_per_device, "GB"), "{:.0f}", "activations per GPU, TP-8, GB")
v("tp8_total_gb", m(s.tp_total_per_device, "GB"), "{:.0f}", "everything per GPU, TP-8, GB")
v("tp8_offload_gb", m(s.tp_offloaded_per_device, "GB"), "{:.1f}", "after offloading moments and master, GB")
v("tp8_weights_gb", m(s.tp_weights_per_device, "GB"), "{:.2f}", "weights per GPU, TP-8, GB")
v("zero_adam_gb", m(s.zero_moments_per_device, "GB"), "{:.1f}", "Adam per GPU over TP x DP, GB")
v("zero_adam_group_gb", m(s.zero_moments_per_device * STATE.tp_size, "GB"), "{:.1f}", "Adam per 8-GPU group, GB")
v("zero_subtotal_gb", m(s.zero_weights_and_moments_per_device, "GB"), "{:.1f}", "weights + Adam subtotal, GB")

print("== Staircase (transfer scenario)")
t = LADDER.result.times
v("stair_hbm_ms", m(t["HBM"], "ms"), "{:.1f}", "10 GB over HBM, ms")
v("stair_nvlink_ms", m(t["NVLink"], "ms"), "{:.1f}", "10 GB over NVLink, ms")
v("stair_ib_ms", m(t["InfiniBand"], "ms"), "{:.0f}", "10 GB over InfiniBand, ms")

print("== Tensor-parallel AllReduce (TP scenario on NVLink and on PCIe)")
nv = TP.result
pc = TP.variant(link_bandwidth=LADDER.links["PCIe"]).result
v("tp_payload_mb", m(nv.payload, "MB"), "{:.0f}", "payload, MB")
v("tp_ring_mb", m(nv.traffic_per_device, "MB"), "{:.0f}", "ring traffic per GPU, MB")
v("tp_nvlink_ms", m(nv.per_collective_time, "ms"), "{:.3f}", "one AllReduce on NVLink, ms")
v("tp_nvlink_total_ms", m(nv.communication_time, "ms"), "{:.1f}", "all AllReduces on NVLink, ms")
v("tp_nvlink_compute_pct", 100 * nv.compute_fraction, "{:.0f}", "compute share, NVLink, %")
v("tp_nvlink_step_ms", m(nv.step_time, "ms"), "{:.0f}", "step on NVLink, ms")
v("tp_pcie_ms", m(pc.per_collective_time, "ms"), "{:.1f}", "one AllReduce on PCIe, ms")
v("tp_pcie_compute_pct", 100 * pc.compute_fraction, "{:.0f}", "compute share, PCIe, %")
v("tp_pcie_step_ms", m(pc.step_time, "ms"), "{:,.0f}", "step on PCIe, ms")
v("tp_pcie_total_ms", m(pc.communication_time, "ms"), "{:,.0f}", "all AllReduces on PCIe, ms")
v("tp_pcie_comm_pct", 100 * pc.communication_fraction, "{:.1f}", "communication share, PCIe, %")

print("== Rack and power")
v("rack_gpu_kw", m(RACK.accelerator_count * H100.tdp, "kW"), "{:.1f}", "GPUs in one DGX rack x TDP, kW")
v("cluster_mw", m(REF_CLUSTER.total_accelerators / RACK.accelerator_count * RACK.total_power, "MW"),
  "{:.1f}", "reference cluster in DGX racks, MW")
series = Literature.Series.AISupercomputers
front = series.find("name", "Frontier")
last = series.largest("power_mw", "first_operational_date", "2025-12-31")
v("frontier_mw", front["power_mw"], "{:.0f}", "Frontier, MW (Epoch AI via MLSysIM)")
v("largest_2025_mw", last["power_mw"], "{:.0f}", "largest system by 2025, MW")
v("largest_over_frontier", last["power_mw"] / front["power_mw"], "{:.0f}", "largest 2025 / Frontier")
chips = Literature.Series.AcceleratorPowerRatings
v("chip_growth", chips.largest("power_w")["power_w"] / m(H100.tdp, "W"), "{:.0f}",
  "highest package rating / H100 TDP")
v("largest_over_frontier_word", round(last["power_mw"] / front["power_mw"]), "words", "the same, in prose")

print("== Transient (load-step scenario)")
p = STEP.result
v("swing_w", m(p.power_step, "W"), "{:.0f}", "per-GPU power step, W")
v("swing_a", m(p.current_step, "A"), "{:.0f}", "current step, A")
v("droop_1us_mv", m(p.droop["slow"], "mV"), "{:.0f}", "droop over the slow edge, mV")
v("droop_1us_pct", 100 * p.droop_share["slow"], "{:.1f}", "droop over the slow edge, % of rail")
v("droop_100ns_mv", m(p.droop["fast"], "mV"), "{:.0f}", "droop over the fast edge, mV")
v("droop_100ns_pct", 100 * p.droop_share["fast"], "{:.0f}", "droop over the fast edge, % of rail")
v("step_kw", m(p.cluster_power_step, "kW"), "{:.0f}", "cluster step, kW")
v("ramp_gws", m(p.ramp_rate, "GW/s"), "{:.2f}", "ramp over the window, GW/s")

print("== Cooling (cooling scenarios)")
air = RACK_AIR.result
water = RACK_AIR.variant(coolant="water").result
v("air_kgs", m(air.mass_flow, "kg/s"), "{:.1f}", "air mass flow, kg/s")
v("air_m3s", m(air.volume_flow, "m**3/s"), "{:.1f}", "air volume flow, m3/s")
v("air_cfm", m(air.volume_flow, "ft**3/min"), "{:,.0f}", "air volume flow, CFM")
v("water_kgs", m(water.mass_flow, "kg/s"), "{:.1f}", "water mass flow, kg/s (about L/s)")
v("air_water_ratio", round(ratio(air.volume_flow, water.volume_flow), -2), "{:,.0f}",
  "air / water volume ratio, to the hundred")
v("nvl72_lpm", m(Scenarios.Nvl72WaterCooling.result.volume_flow, "L/min"), "{:.0f}", "GB200 NVL72 water flow, L/min")

print("== Failure domains (outage scenario)")
v("cdu_cost", CDU.result.direct_cost, "{:,.0f}", "failed rack's GPUs idle, $")
v("job_cost", CDU.result.job_cost, "{:,.0f}", "whole synchronous job idle, $")

print("== Pod")
v("pod_eflops", m(POD.total_accelerators * H100.compute.peak_flops, "EFLOP/s"), "{:.2f}", "pod FP16 peak, EFLOP/s")
v("pod_tbs", m(POD.total_accelerators * LADDER.links["InfiniBand"], "TB/s"), "{:.0f}", "pod injection, TB/s")
v("pod_mw", m(POD.total_accelerators * H100.tdp, "MW"), "{:.0f}", "pod GPU power, MW")
v("run_failures", GPU_FAIL.result.expected_failures, "{:.0f}", "GPU failures over the run")
v("node_fail_day", NODE_FAIL.result.failures_per_day, "{:.0f}", "node failures per day")
v("node_fail_day_word", round(NODE_FAIL.result.failures_per_day), "words", "the same, in prose")
v("eta_gpuh", BUDGET.required_device_hours, "{:,.0f}", "GPU-hours at the sustained share")

print("== Pod chain (training-run scenario)")
r = RUN.result
v("chain_flops_mant", m(r.total_flops, "flop") / 10 ** 23, "{:.2f}", "training FLOPs, x 10^23")
v("chain_peak_h", m(r.peak_time, "hour"), "{:.1f}", "hours at 100% of FP8 peak")
v("chain_compute_h", m(r.compute_time, "hour"), "{:.1f}", "compute-only baseline, h")
v("chain_scaled_h", m(r.scaled_time, "hour"), "{:.1f}", "after scaling efficiency, h")
v("chain_ckpt_h", m(r.stage_times["checkpoint"], "hour"), "{:.1f}", "after checkpoints, h")
v("chain_fail_h", m(r.stage_times["failures"], "hour"), "{:.1f}", "after failures, h")
v("chain_total_h", m(r.total_time, "hour"), "{:.1f}", "after maintenance, h")
v("chain_share_pct", 100 * r.delivered_fraction, "{:.0f}", "share of FP8 peak delivered, %")

print("== Fallacies")
hw_ratio = RUN.fleet.total_accelerators / REF_CLUSTER.total_accelerators
v("fallacy_speed", hw_ratio * SCALE.Eff8192Gpu / SCALE.Eff1024Gpu, "{:.1f}", "speed from 1,024 to 8,192 GPUs")
v("fallacy_lost_pct", 100 * (1 - SCALE.Eff8192Gpu), "{:.0f}", "share lost to communication and bubbles, %")
v("h100_v100_peak", Scenarios.V100ToH100.peak_ratio, "{:.1f}", "H100 / V100 FP16 peak")
v("v100_in_h100_gpuh", Scenarios.V100ToH100.candidate_device_hours, "{:,.0f}", "V100 job in H100 GPU-hours")
v("int8_floor_ms", m(DECODE.variant(precision="int8").result.memory_time, "ms"), "{:.0f}", "70B INT8 floor on H100, ms")

print("== Rack title")
trad = Infrastructure.Racks.Traditional.power_kw
v("dgx_rack_ratio", m(RACK.total_power, "kW") / trad, "{:.0f}", "DGX H100 rack / server rack")
v("nvl72_rack_ratio", m(NVL72.tdp, "kW") / trad, "{:.0f}", "GB200 NVL72 rack / server rack")

if __name__ == "__main__":
    write(__file__)
