#!/usr/bin/env python3
"""Every number on the Lecture 7 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature) and the calculations
(Scenarios.*, each backed by a solver, a scenario type, or reliability
physics); the chapter reads the same scenarios. This file only selects values,
converts units, and formats them; it types no numbers. Each value is recorded
with v() under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex.

Not from MLSysIM, and typed where they appear: the PyTorch DCP and torchrun
traces (measured by dcp_trace.py and crash_trace.py in this folder), the
torchvision weights metadata on the degradation card, Example 7.1's incident
timings, and the self-check and vote settings.

Run from this folder: python3 calcs.py
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
sys.path.insert(0, str(HERE))
from deckvals import v, write, mlsysim  # noqa: E402
from digitize_sdc import BOX  # noqa: E402  (the keynote slide's highlighted window)

Q_ = mlsysim.core.units.Q_
Systems, Models, Literature, Scenarios = (mlsysim.Systems, mlsysim.Models, mlsysim.Literature,
                                          mlsysim.Scenarios)
RUNS, FLEET = Literature.TrainingRuns, Literature.FleetOperations
LLAMA3 = Literature.Series.Llama3Interruptions.entries
NODE = Systems.Nodes.DGX_H100
LADDER = Scenarios.GpuFailureLadder                   # GPU-only, MTTF / N
FULL, FULL100K = Scenarios.FullSystemFailures10K, Scenarios.FullSystemFailures100K
YD, YD100K = Scenarios.Gpt3CheckpointInterval10K, Scenarios.Gpt3CheckpointInterval100K
FIVE = Scenarios.FiveMinuteCheckpoint1K
REC1K, REC10K = Scenarios.Gpt3RecoveryBudget1K, Scenarios.Gpt3RecoveryBudget10K
SPARE = Scenarios.HotSpareNode10K
STRAG = Scenarios.HalfSpeedStraggler1K
NINES = Scenarios.GpuAvailability10K
REPLICAS = (Scenarios.ServingReplica99, Scenarios.ServingReplicas99x2, Scenarios.ServingReplicas99x3)
PROBE = Scenarios.InferenceReadinessProbe
SDC = Scenarios.SdcExposure100K


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    return 100 * x


def component(name):
    return next(r["interruption_count"] for r in LLAMA3 if r["component"] == name)


def category(name):
    return sum(r["interruption_count"] for r in LLAMA3 if r["category"] == name)


print("== Hook: Llama 3 405B (Dubey et al. 2024)")
llama_gpus = m(Models.Language.Llama3_405B.training_accelerators_ref, "count")
llama_days = float(RUNS.Llama3_405B_SnapshotDays)
llama_n = float(RUNS.Llama3_405B_UnexpectedInterruptions)
llama_mttf = m(Q_(llama_days, "day"), "hour") / llama_n
v("llama_gpus", llama_gpus, "{:,.0f}", "H100s in the run")
v("llama_gpus_k_words", round(llama_gpus / 10 ** 3), "words", "the same, thousands, in prose")
v("llama_days", llama_days, "{:.0f}", "snapshot, days")
v("llama_interrupts", llama_n, "{:.0f}", "unexpected interruptions")
v("llama_mttf_h", llama_mttf, "{:.1f}", "hours between interruptions")
v("hours_per_day", m(Q_(1, "day"), "hour"), "{:.0f}", "hours in a day")
v("llama_eff_pct", pct(float(RUNS.Llama3_405B_EffectiveTrainingTimeFraction)), "{:.0f}",
  "effective training time, over this %")
v("llama_manual", float(RUNS.Llama3_405B_ManualInterventions), "{:.0f}", "manual interventions")
v("llama_manual_words", float(RUNS.Llama3_405B_ManualInterventions), "words", "the same, in prose")

print("== Running case and clusters")
ref_gpus = Systems.Clusters.Training_1K.total_accelerators
v("ref_gpus", ref_gpus, "{:,.0f}", "reference cluster GPUs")
v("ref_gpus_less1", ref_gpus - 1, "{:,.0f}", "the other GPUs at the barrier")
v("ref_nodes", Systems.Clusters.Training_1K.count, "{:.0f}", "reference cluster nodes")
v("ref_nodes_less1", Systems.Clusters.Training_1K.count - 1, "{:.0f}", "nodes after one is lost")
v("big_gpus", FULL.n_devices, "{:,.0f}", "the chapter's large cluster")
v("gpus_per_node", NODE.accelerators_per_node, "{:.0f}", "GPUs per node")
v("params_b", m(Models.Language.GPT3.parameters, "Gparam"), "{:.0f}", "running model, billions")
v("ckpt_tb", m(YD.checkpoint_bytes, "TB"), "{:.2f}", "resumable checkpoint, TB")
mpa = mlsysim.ReferenceStats.MixedPrecisionAdam
ckpt_bpp = m(mpa.CheckpointBytesPerParameter, "byte/param")
opt_bpp = m(mpa.OptimizerStateBytesPerParameter, "byte/param")
fp32 = m(mlsysim.BYTES_FP32, "byte")
v("ckpt_bpp", ckpt_bpp, "{:.0f}", "checkpoint, B per parameter")
v("bf16_bytes", m(mlsysim.BYTES_FP16, "byte"), "{:.0f}", "BF16 weights, B per parameter")
v("fp32_bytes", fp32, "{:.0f}", "FP32 master weights, B per parameter")
v("adam_bytes", opt_bpp - fp32, "{:.0f}", "Adam moments, B per parameter")
v("opt_share_pct", pct(opt_bpp / ckpt_bpp), "{:.0f}", "optimizer state share of the checkpoint, %")

print("== Failure clock: GPU-only (MTTF / N)")
mttf_h = m(LADDER.component_mtbf, "hour")
v("gpu_mttf_h", mttf_h, "{:,.0f}", "GPU MTTF, hours")
v("gpu_mttf_yr", m(LADDER.component_mtbf, "year"), "{:.1f}", "the same, years")
day_h = m(Q_(1, "day"), "hour")
for n in LADDER.sizes:
    h = m(LADDER.intervals[n], "hour")
    v(f"ladder_{n}_h", h, "{:,.0f}" if h >= day_h else "{:g}", f"{n} GPUs, hours")
    if h >= day_h:
        v(f"ladder_{n}_d", m(LADDER.intervals[n], "day"), "{:.0f}", f"{n} GPUs, days")
v("nines_pct", pct(NINES.availability), "{:.2f}", "per-GPU availability, %")
v("all_up_pct", pct(NINES.all_up), "{:.1f}", "all 10,000 up at once, %")

print("== Failure clock: full system (series cascade)")
fr = FULL.result
x5 = 10 ** 5
v("psus_per_node", NODE.psus_per_node, "{:.0f}", "power supplies per node")
v("psus_needed", NODE.psus_per_node - NODE.psu_redundancy, "{:.0f}", "supplies the node needs")
v("psu_redundancy", NODE.psu_redundancy, "{:.0f}", "redundant supplies")
v("pcie_mttf_h", m(FULL.component_mttf["pcie"], "hour"), "{:,.0f}", "PCIe link MTTF, hours")
v("nic_mttf_h", m(FULL.component_mttf["network"], "hour"), "{:,.0f}", "node network path MTTF, hours")
v("psu_mttf_h", m(FULL.component_mttf["power"], "hour"), "{:,.0f}", "PSU MTTF, hours")
v("rate_gpu", fr.component_rates["gpu"] * x5, "{:.1f}", "GPU, x 1e-5 per hour")
v("rate_pcie", fr.component_rates["pcie"] * x5, "{:.1f}", "PCIe, x 1e-5 per hour")
v("rate_nic", fr.component_rates["network"] * x5, "{:.3f}", "NIC share, x 1e-5 per hour")
v("rate_power", fr.component_rates["power"] * x5, "{:.0f}", "power, x 1e-5 per hour")
v("rate_total", fr.device_rate * x5, "{:.3f}", "per GPU, x 1e-5 per hour")
v("rate_system", fr.system_rate, "{:.3f}", "cluster failures per hour")
v("full_mtbf_h", m(fr.mtbf, "hour"), "{:.2f}", "full-system MTBF, hours")
v("full_per_day", fr.failures_per_day, "{:.1f}", "failures per day")

print("== Failure clock: measured (Kokolis et al. 2025)")
kok8, kok1k = float(FLEET.Kokolis8GpuJobMttfHours), float(FLEET.Kokolis1024GpuJobMttfHours)
kok16k = float(FLEET.Kokolis16KGpuProjectedMttfHours)
line8 = m(LADDER.intervals[NODE.accelerators_per_node], "hour")
line1k = m(mlsysim.physics.calc_mtbf_cluster(LADDER.component_mtbf, ref_gpus), "hour")
v("kok_8_h", kok8, "{:,.0f}", "measured job MTTF at 8 GPUs, hours")
v("kok_1k_h", kok1k, "{:.1f}", "measured job MTTF at 1,024 GPUs, hours")
v("kok_16k_h", kok16k, "{:.1f}", "projected job MTTF at 16,384 GPUs, hours")
v("line_1k_h", line1k, "{:.1f}", "GPU-only line at 1,024 GPUs, hours")
v("kok_8_x", line8 / kok8, "{:.1f}", "GPU-only line over measured, 8 GPUs")
v("kok_1k_x", line1k / kok1k, "{:.1f}", "GPU-only line over measured, 1,024 GPUs")

print("== Fault types: Llama 3 root causes (Table 5 counts)")
gpu_parts = category("GPU")
v("causes_gpu", gpu_parts, "{:.0f}", "GPU-part interruptions")
v("causes_gpu_pct", pct(gpu_parts / llama_n), "{:.0f}", "their share, %")
v("causes_faulty", component("Faulty GPU"), "{:.0f}", "faulty GPU")
v("causes_hbm", component("GPU HBM3 Memory"), "{:.0f}", "HBM3")
v("causes_sram", component("GPU SRAM Memory"), "{:.0f}", "SRAM")
v("causes_sysproc", component("GPU System Processor"), "{:.0f}", "system processor")
v("causes_thermal", component("GPU Thermal Interface + Sensor"), "{:.0f}", "thermal")
v("causes_software", category("Dependency"), "{:.0f}", "software, all")
v("causes_bugs", component("Software Bug"), "{:.0f}", "software bugs")
v("causes_dependency", component("Dependency"), "{:.0f}", "dependency failures")
v("causes_network", category("Network"), "{:.0f}", "network switches and cables")
v("causes_maint", category("Unplanned Maintenance"), "{:.0f}", "unplanned host maintenance")
v("causes_sdc", component("Silent Data Corruption"), "{:.0f}", "detected silent data corruption")
v("causes_sdc_words", component("Silent Data Corruption"), "words", "the same, in prose")
v("tpu_pod_chips", Systems.Pods.TPUv4_4096.chips, "{:,.0f}", "TPU v4 pod chips")

print("== Silent data corruption")
with open(HERE / "sdc_trace.csv") as f:
    rows = [(float(a), float(b)) for a, b in list(csv.reader(f))[1:]]
boxed = [g for t, g in rows if BOX[0] <= t <= BOX[1]]
v("sdc_spike_words", round(max(boxed)), "words", "tallest boxed point over the normal band (photo reading)")
v("sdc_rate_per_m_words", SDC.event_rate * 10 ** 6, "words", "illustrative events per million GPU-hours")
v("sdc_fleet_gpus", SDC.n_devices, "{:,.0f}", "fleet GPUs")
v("sdc_steps", SDC.steps_between, "{:,.0f}", "steps between events")
v("sdc_step_words", m(SDC.step_time, "second"), "words", "step time, seconds, in prose")
v("sdc_hours", m(SDC.interval, "hour"), "{:.0f}", "hours between events")

print("== Checkpoints: Young-Daly on the full-system MTBF")
yd = YD.result
hour = Q_(1, "hour")
v("write_bw_gbs", m(YD.write_bandwidth, "GB/s"), "{:.0f}", "checkpoint write bandwidth, GB/s")
v("delta_s", m(yd.write_time, "second"), "{:.1f}", "write time, s")
v("m_h", m(yd.mtbf, "hour"), "{:.2f}", "system MTBF, hours")
v("m_s", m(yd.mtbf, "second"), "{:,.0f}", "system MTBF, s")
v("two_m_s", m(yd.interval, "second") / yd.rework_share, "{:,.0f}", "2M, s")
v("tau_s", m(yd.interval, "second"), "{:.0f}", "Young-Daly interval, s")
v("tau_min", m(yd.interval, "minute"), "{:.1f}", "Young-Daly interval, min")
v("term_pct", pct(yd.write_share), "{:.2f}", "each term at the optimum, %")
v("waste_pct", pct(yd.waste), "{:.1f}", "waste at the optimum, %")
v("sec_per_hour", m(hour, "second"), "{:,.0f}", "seconds in an hour")
v("hourly_write_pct", pct((yd.write_time / hour).to("").magnitude), "{:.1f}", "hourly saves, write term, %")
v("hourly_rework_pct", pct(yd.rework_share * (hour / yd.interval).to("").magnitude), "{:.1f}",
  "hourly saves, rework term, %")
v("hourly_waste_pct", pct(YD.waste_at(hour)), "{:.1f}", "hourly saves, waste, %")
doubled = FULL.variant(n_devices=FULL.n_devices + FULL.n_devices)   # twice the GPUs, half the MTBF
yd_doubled = YD.variant(mtbf=doubled.result.mtbf).result
v("sqrt2", yd_doubled.waste / YD.variant(mtbf=FULL.result.mtbf).result.waste, "{:.1f}",
  "interval and waste change when the cluster doubles (MTBF halves)")
yd100k = YD100K.result
v("big_open_gpus", FULL100K.n_devices, "{:,.0f}", "open question, GPUs")
v("m100k_h", m(yd100k.mtbf, "hour"), "{:.2f}", "full-system MTBF at 100,000 GPUs, hours")
v("m100k_s", m(yd100k.mtbf, "second"), "{:,.0f}", "the same, s")
v("waste100k", yd100k.waste, "{:.2f}", "optimized waste at 100,000 GPUs, fraction")
v("waste100k_pct", pct(yd100k.waste), "{:.0f}", "the same, %")
five = FIVE.result
v("tf_write_min", m(FIVE.write_time, "minute"), "{:.0f}", "T/F: write, minutes")
v("tf_mtbf_h", m(FIVE.mtbf, "hour"), "{:.0f}", "T/F: GPU-only MTBF at 1,000 GPUs, hours")
v("tf_gpus", LADDER.sizes[3], "{:,.0f}", "T/F: GPUs")
v("tf_tau_min", m(five.interval, "minute"), "{:.0f}", "T/F: Young-Daly interval, minutes")

print("== Recovery: cold-restart budget (175B, GPU-only failure rate)")
r1k, r10k = REC1K.result, REC10K.result
v("t_detect_s", m(r1k.detect, "second"), "{:.0f}", "detect, s")
v("t_restart_s", m(r1k.restart, "second"), "{:.0f}", "restart, s")
v("t_restart_min", m(r1k.restart, "minute"), "{:.0f}", "restart, min")
v("t_load_s", m(r1k.reload, "second"), "{:.1f}", "load, s")
v("t_warmup_s", m(r1k.warmup, "second"), "{:.0f}", "warmup, s")
v("t_warmup_min", m(r1k.warmup, "minute"), "{:.0f}", "warmup, min")
v("t_total_min", m(r1k.total, "minute"), "{:.1f}", "one cold restart, min")
v("fpd_1k", r1k.failures_per_day, "{:.2f}", "1,024 GPUs: failures per day")
v("loss_1k_min", m(r1k.lost_per_day, "minute"), "{:.0f}", "1,024 GPUs: recovery per day, min")
v("share_1k_pct", pct(r1k.day_share), "{:.1f}", "1,024 GPUs: share of the day, %")
v("fpd_10k", r10k.failures_per_day, "{:.1f}", "10,000 GPUs: failures per day")
v("loss_10k_min", m(r10k.lost_per_day, "minute"), "{:.0f}", "10,000 GPUs: recovery per day, min")
v("share_10k_pct", pct(r10k.day_share), "{:.1f}", "10,000 GPUs: share of the day, %")
v("lost_gpuh_10k", r10k.gpu_hours_per_day, "{:,.0f}", "10,000 GPUs: GPU-hours lost per day")
v("gpu_price", m(SPARE.device_hour_rate, "USD/hour"), "{:.0f}", "price, $ per GPU-hour")
v("spare_day_usd", m(SPARE.spare_cost_per_day, "USD"), "{:,.0f}", "hot-spare node, $ per day")
v("spare_value_usd", m(SPARE.value_per_minute, "USD"), "{:,.0f}", "one minute off every recovery, $ per day")
v("spare_breakeven_s", m(SPARE.breakeven, "second"), "{:.0f}", "break-even cut per recovery, s")

print("== Recovery: straggler and elastic")
v("strag_step_s", m(STRAG.step_time, "second"), "{:.0f}", "healthy step, s")
v("strag_slow_s", m(STRAG.slow_step, "second"), "{:.0f}", "step with the straggler, s")
v("strag_tput_pct", pct(STRAG.throughput), "{:.0f}", "job throughput, %")
v("strag_idle_usd", m(STRAG.idle_cost, "USD"), "{:,.0f}", "idle compute, $ per hour")
v("strag_hw_pct", pct(1 / STRAG.n_devices), "{:.1f}", "one GPU's share of the hardware, %")
v("elastic_kept_pct", pct((ref_gpus - NODE.accelerators_per_node) / ref_gpus), "{:.1f}",
  "devices kept after losing one node, %")

print("== Serving: replicas, KV cache, probes")
v("replica_pct", pct(REPLICAS[0].availability), "{:.0f}", "one replica's availability, %")
v("down1_days", m(REPLICAS[0].replicated_downtime_per_year, "day"), "{:.2f}", "one replica, days a year")
v("down2_min", m(REPLICAS[1].replicated_downtime_per_year, "minute"), "{:.1f}", "two replicas, min a year")
v("down3_s", m(REPLICAS[2].replicated_downtime_per_year, "second"), "{:.1f}", "three replicas, s a year")
llama70 = Models.Language.Llama3_70B
kv = llama70.get_kv_cache_size(seq_len=llama70.context_length, batch_size=1, precision=mlsysim.BYTES_FP16)
v("kv_gb", m(kv, "GB"), "{:.0f}", "KV cache of one full-context conversation, GB")
v("kv_context_k", llama70.context_length >> 10, "{:.0f}", "context, K tokens (K = 1,024)")
early, late = PROBE.detection_window
v("probe_period_s", m(PROBE.interval, "second"), "{:.0f}", "probe period, s")
v("probe_timeout_s", m(PROBE.timeout, "second"), "{:.0f}", "probe timeout, s")
v("probe_fail", PROBE.failure_threshold, "{:.0f}", "failure threshold")
v("probe_fail_words", PROBE.failure_threshold, "words", "the same, in prose")
v("probe_success", PROBE.success_threshold, "{:.0f}", "success threshold")
v("probe_early_s", m(early, "second"), "{:.0f}", "earliest removal, s")
v("probe_late_s", m(late, "second"), "{:.0f}", "latest removal, s")
v("vit_params_m", m(Models.Vision.ViT_Large.parameters, "Mparam"), "{:.0f}", "ViT-Large, registry, millions")
v("effnet_params_m", m(Models.Vision.EfficientNetB4.parameters, "Mparam"), "{:.0f}", "EfficientNet-B4, millions")
v("mobilenet_params_m", m(Models.Vision.MobileNetV3_Large.parameters, "Mparam"), "{:.1f}",
  "MobileNetV3-Large, millions")

if __name__ == "__main__":
    write(__file__)
