#!/usr/bin/env python3
"""Every number on the Lecture 8 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature, ReferenceStats) and the
calculations (Scenarios.*, each backed by a solver or a scenario type); the
chapter, Fleet Orchestration, reads the same scenarios. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Systems, Literature, Scenarios, Ops = (mlsysim.Hardware, mlsysim.Systems, mlsysim.Literature,
                                                 mlsysim.Scenarios, mlsysim.Ops)
FLEET_OPS = Literature.FleetOperations
RATE = mlsysim.Infrastructure.Pricing.Fleet.GpuHourRef.rate
SPOT_RATE = mlsysim.Infrastructure.Pricing.Fleet.SpotGpuHourRef.rate
FIX = Scenarios.MixedFleetPolicyFix1K             # hook and synthesis: the three-pool 1,024-GPU cluster
MIX = Scenarios.KalosJobSizeMix                   # Kalos trace by job size
GANG = Scenarios.PartialGangHold                  # 900 of 1,024 GPUs held while the gang assembles
IDLE10K = Scenarios.Training10KIdleHour
DRF = Scenarios.DrfTwoTenantExample
PACK, SNAP = Scenarios.SixGpuJobsOnDgxH100Nodes, Scenarios.FragmentedEightNodeSnapshot
Q1, QML, Q98 = (Scenarios.Mg1ExponentialJobQueueAt80, Scenarios.Mg1MixedJobQueueAt80,
                Scenarios.Mg1ExponentialJobQueueAt98)
TPX = Scenarios.TensorParallelExchange1GB
LAYOUT = Scenarios.Gpt3ThreeDParallel256
ELASTIC = Scenarios.ElasticResizeDecision
REPLICA = Scenarios.Gpt3SelfContainedReplicaDgxH100
PREEMPT = Scenarios.ForcedPreemption175BFineTune64
SPOT, SPOTRUN = Scenarios.SpotInterruptionEffectiveCost, Scenarios.Llama2_70BSpotRun512
SHARE = Scenarios.ServingConsolidationA100
DIURNAL = Scenarios.DiurnalServingLending1K
HOARD, QUOTA = Scenarios.ReservedGpusIdleThreeWeeks, Scenarios.TeamQuotaLending500
TIERS = mlsysim.ReferenceStats.UtilizationTierProfile
RECOVERY = Systems.Reliability.Recovery
GIB_BYTES = mlsysim.core.units.Q_(1, "GiB").to("byte").magnitude   # the NCCL message the plot marks


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities (or two numbers)."""
    r = a / b
    return r.to("").magnitude if hasattr(r, "to") else r


def pct(x):
    return 100 * x


print("== Hook and synthesis: the three-pool 1,024-GPU cluster (Example 8.2)")
pools = FIX.pools
v("cluster_gpus", FIX.fleet_gpus, "{:,.0f}", "cluster GPUs")
v("pool_a80", pools["A100-80GB"], "{:.0f}", "A100-80 GB pool, GPUs")
v("pool_a40", pools["A100-40GB"], "{:.0f}", "A100-40 GB pool, GPUs")
v("pool_v100", pools["V100-32GB"], "{:.0f}", "V100-32 GB pool, GPUs")
v("a100_gb", m(Hardware.Cloud.A100.memory.capacity, "GiB"), "{:.0f}", "A100 memory, published label, GB")
v("a100_40_gb", m(Hardware.Cloud.A100_40GB.memory.capacity, "GiB"), "{:.0f}", "A100 40 GB memory, GB")
v("v100_gb", m(Hardware.Cloud.V100.memory.capacity, "GiB"), "{:.0f}", "V100 memory, GB")
v("pending_jobs", FIX.pending_jobs, "{:.0f}", "jobs pending before the fix")
v("util_before_pct", pct(FIX.utilization_before), "{:.0f}", "cluster utilization before, %")
v("util_after_pct", pct(FIX.utilization_after), "{:.0f}", "cluster utilization after, %")
v("util_before", FIX.utilization_before, "{:.1f}", "the same, as a fraction (0.6)")
v("util_before_2", FIX.utilization_before, "{:.2f}", "the same, two decimals (0.60)")
v("util_gain", FIX.improvement, "{:.2f}", "utilization gain, fraction")
v("util_gain_pts", pct(FIX.improvement), "{:.0f}", "utilization gain, points")
v("fix_gain_gpus", FIX.productive_gain, "{:.0f}", "GPU-equivalents recovered")
v("fix_raw_gpus", FIX.raw_gpu_equivalent, "{:.0f}", "raw GPUs those replace")
v("fix_value_musd", FIX.annual_value / 10 ** 6, "{:.1f}", "annual value, $M")
v("fix_value_musd_round", FIX.annual_value / 10 ** 6, "{:.0f}", "the same, whole $M (the chapter's rounding)")
v("job_need_gb", m(FIX.job_memory_need, "GB"), "{:.0f}", "memory the template-bound jobs needed, GB")
for pool, key in (("A100-80GB", "a80"), ("A100-40GB", "a40"), ("V100-32GB", "v100")):
    pm = FIX.pool_metrics[pool]
    tier = "active" if "active_before" in pm else "allocated"
    v(f"{key}_before_pct", pct(pm[f"{tier}_before"]), "{:.0f}", f"{pool} {tier} before, %")
    v(f"{key}_after_pct", pct(pm[f"{tier}_after"]), "{:.0f}", f"{pool} {tier} after, %")
v("philly_pct", pct(float(FLEET_OPS.PhillyMeanGpuUtilization)), "{:.0f}", "Philly mean utilization, %")

print("== Recall: one recovery of the 175B job (Fault Tolerance)")
v("params_b", m(REPLICA.workload.parameters, "Gparam"), "{:.0f}", "running model, billions of parameters")
ckpt = PREEMPT.checkpoint_bytes
v("ckpt_tb", m(ckpt, "TB"), "{:.2f}", "175B resumable checkpoint, TB")
v("rec_detect_s", float(RECOVERY.detection_time_s), "{:.0f}", "detect, s")
v("rec_restart_s", float(RECOVERY.restart_time_s), "{:.0f}", "restart, s")
v("rec_restart_min", m(PREEMPT.result.restart, "minute"), "{:.0f}", "restart, min")
v("rec_load_s", m(PREEMPT.result.reload, "s"), "{:.1f}", "checkpoint load, s")
v("rec_warm_s", float(RECOVERY.warmup_time_s), "{:.0f}", "warm up, s")
v("rec_warm_min", m(PREEMPT.result.warmup, "minute"), "{:.0f}", "warm up, min")
v("rec_total_min", m(RECOVERY.cold_restart_time(ckpt), "minute"), "{:.1f}", "four-term recovery, min")
v("ft_gpus", PREEMPT.n_devices, "{:.0f}", "fine-tuning run, GPUs")
v("fleet10k_gpus", IDLE10K.gpus, "{:,.0f}", "large fleet, GPUs")

print("== Kalos trace by job size (Hu et al. 2024)")
km = MIX.result
cl = km.classes
v("kalos_jobs", km.job_count, "{:,.0f}", "GPU jobs")
v("kalos_large_gpus", MIX.large_threshold, "{:.0f}", "large-job threshold, GPUs")
v("kalos_single_job_pct", pct(cl["single"]["job_share"]), "{:.0f}", "one-GPU jobs, % of jobs")
v("kalos_single_time_pct", pct(cl["single"]["time_share"]), "{:.1f}", "one-GPU jobs, % of GPU time")
v("kalos_medium_job_pct", pct(cl["medium"]["job_share"]), "{:.0f}", "mid-size jobs, % of jobs")
v("kalos_medium_time_pct", pct(cl["medium"]["time_share"]), "{:.0f}", "mid-size jobs, % of GPU time")
v("kalos_large_job_pct", pct(cl["large"]["job_share"]), "{:.0f}", "large jobs, % of jobs")
v("kalos_large_time_pct", pct(cl["large"]["time_share"]), "{:.0f}", "large jobs, % of GPU time")
v("kalos_single_h", cl["single"]["mean_gpu_hours"], "{:.2f}", "one-GPU jobs, mean GPU-hours")
v("kalos_medium_h", cl["medium"]["mean_gpu_hours"], "{:.1f}", "mid-size jobs, mean GPU-hours")
v("kalos_large_h", cl["large"]["mean_gpu_hours"], "{:,.0f}", "large jobs, mean GPU-hours")
v("kalos_orders_word", math.floor(math.log10(cl["large"]["mean_gpu_hours"] / cl["single"]["mean_gpu_hours"])),
  "words", "orders of magnitude between the two means")
v("kalos_cs", km.service_cv_lower_bound, "{:.1f}", "C_s lower bound")
QK = Q1.variant(service_cv=km.service_cv_lower_bound)
v("kalos_var", QK.variability_factor, "{:.0f}", "variability factor at the Kalos bound")
v("mg1_kalos", QK.wait_ratio, "{:.0f}", "wait over mean job at 80% load, Kalos bound")
for state in ("completed", "failed", "canceled", "other"):
    v(f"kalos_{state}_pct", pct(km.state_time_shares[state]), "{:.0f}", f"GPU time in {state} jobs, %")

print("== Gang deadlock and backfill")
v("gpu_hour_usd", m(RATE, "USD/hour"), "{:.0f}", "GPU-hour rate, $")
v("deadlock_usd_h", IDLE10K.cost, "{:,.0f}", "10,000 idle GPUs, $ per hour")
v("openai_nodes", float(FLEET_OPS.OpenAiKubernetesNodes), "{:,.0f}", "OpenAI Kubernetes nodes")
v("gang_gpus", GANG.gang_gpus, "{:,.0f}", "gang size, GPUs")
v("gang_held", GANG.gpus, "{:.0f}", "free GPUs while the gang waits")
v("gang_missing", GANG.missing_gpus, "{:.0f}", "GPUs still held by running jobs")

print("== Dominant-resource fairness (table 1)")
dr = DRF.result
cap, ba, bb = DRF.capacity, DRF.bundles["A"], DRF.bundles["B"]
for key, res in (("gpu", "accelerators"), ("cpu", "cpu_cores"), ("mem", "memory_gb"), ("net", "network_gbps")):
    v(f"drf_cap_{key}", cap[res], "{:.0f}", f"cluster {res}")
    v(f"drf_a_{key}", ba[res], "{:.0f}", f"A replica {res}")
    v(f"drf_b_{key}", bb[res], "{:.0f}", f"B worker {res}")
v("drf_a_count_word", dr["bundle_counts"]["A"], "words", "A replicas granted")
v("drf_b_count_word", dr["bundle_counts"]["B"], "words", "B workers granted")
v("drf_b_gang_gpus", dr["tenants"]["B"]["allocation"]["accelerators"], "{:.0f}", "B's gang, GPUs")
v("drf_dom", dr["tenants"]["A"]["dominant_share"], "{:.2f}", "dominant share, both tenants")
v("drf_b_cpu_share", dr["tenants"]["B"]["shares"]["cpu_cores"], "{:.3f}", "B CPU and memory share")

print("== Fragmentation (figure 3) and packing")
v("node_gpus", SNAP.gpus_per_node, "{:.0f}", "GPUs per node")
v("frag_free", SNAP.free_slots, "{:.0f}", "free slots in the snapshot")
v("frag_total", SNAP.total_gpus, "{:.0f}", "slots in the snapshot")
v("frag_free_pct", pct(SNAP.free_share), "{:.0f}", "free share, %")
v("pack_job_gpus", PACK.job_gpus, "{:.0f}", "GPUs per job")
v("pack_stranded", PACK.stranded_per_node, "{:.0f}", "stranded GPUs per node")
v("pack_usable_pct", pct(PACK.usable_share), "{:.0f}", "usable capacity, %")

print("== M/G/1 (Napkin Math 8.1) and the 98 percent pitfall")
v("mg1_rho_pct", pct(QML.utilization), "{:.0f}", "offered load, %")
v("mg1_rho", QML.utilization, "{:.1f}", "offered load")
v("mg1_one_minus", 1 - QML.utilization, "{:.1f}", "1 - rho")
v("mg1_load", QML.load_factor, "{:.0f}", "load factor at 0.8")
v("cs_uniform", Q1.service_cv, "{:.0f}", "C_s, exponential")
v("cs_ml", QML.service_cv, "{:.0f}", "C_s, mixed ML queue")
v("cs_ml_sq", QML.service_cv * QML.service_cv, "{:.0f}", "C_s squared")
v("mg1_var_uniform", Q1.variability_factor, "{:.0f}", "variability factor, C_s = 1")
v("mg1_var_ml", QML.variability_factor, "{:.0f}", "variability factor, C_s = 3")
v("mg1_uniform", Q1.wait_ratio, "{:.0f}", "wait over mean job, C_s = 1")
v("mg1_ml", QML.wait_ratio, "{:.0f}", "wait over mean job, C_s = 3")
v("mg1_wrong", QML.load_factor * QML.service_cv, "{:.0f}", "the answer that scales with C_s, not its square")
v("sat_rho_pct", pct(Q98.utilization), "{:.0f}", "pitfall utilization, %")
v("sat_rho", Q98.utilization, "{:.2f}", "the same, fraction")
v("sat_one_minus", 1 - Q98.utilization, "{:.2f}", "1 - rho")
v("sat_wait", Q98.wait_ratio, "{:.0f}", "wait over mean job at 98%")
Q60 = Q1.variant(utilization=FIX.utilization_before)
v("q60_wait", Q60.wait_ratio, "{:.1f}", "wait over mean job at 60%")
v("pollux_lo", pct(float(FLEET_OPS.PolluxJctReductionLow)), "{:.0f}", "Pollux JCT cut, low, %")
v("pollux_hi", pct(float(FLEET_OPS.PolluxJctReductionHigh)), "{:.0f}", "Pollux JCT cut, high, %")

print("== Bandwidth and placement (Napkin Math 8.2; NCCL)")
nccl = next(r for r in Literature.Series.NCCLAllReduceH100Node.entries if r["size_bytes"] == GIB_BYTES)
v("nccl_busbw", nccl["ip_busbw_GBps"], "{:.0f}", "in-place bus bandwidth at 1 GiB, GB/s")
links = TPX.links
v("nvlink_gbs", m(links["NVLink"], "GB/s"), "{:.0f}", "H100 NVLink per direction, GB/s")
v("ndr_gbs", m(links["InfiniBand NDR"], "GB/s"), "{:.0f}", "one NDR port, GB/s")
v("nvlink_ndr_x", ratio(links["NVLink"], links["InfiniBand NDR"]), "{:.0f}", "NVLink over NDR")
times = TPX.result.times
v("tp_payload_gb", m(TPX.payload, "GB"), "{:.0f}", "exchange size, GB")
v("tp_ms_nvlink", m(times["NVLink"], "ms"), "{:.1f}", "1 GB over NVLink, ms")
v("tp_ms_ib", m(times["InfiniBand NDR"], "ms"), "{:.0f}", "1 GB over NDR, ms")
v("tp_slow_x", ratio(times["InfiniBand NDR"], times["NVLink"]), "{:.0f}", "slowdown, x")
v("lay_gpus", LAYOUT.n_devices, "{:.0f}", "placement example, GPUs")
v("lay_tp", LAYOUT.tp_size, "{:.0f}", "TP")
v("lay_pp", LAYOUT.pp_size, "{:.0f}", "PP")
v("lay_dp", LAYOUT.dp_size, "{:.0f}", "DP")

print("== Elastic decision (Napkin Math 8.3)")
E = ELASTIC
v("es_window_h", E.window_h, "{:.0f}", "window, hours")
v("es_wait_h", E.wait_h, "{:.0f}", "queue wait for the full size, hours")
v("es_rescale_min", m(E.rescale_overhead, "minute"), "{:.0f}", "resize pause, minutes")
v("es_small_gpus", E.small_gpus, "{:.0f}", "start small, GPUs")
v("es_step_gpus", E.step_up_gpus, "{:.0f}", "step-up, GPUs")
v("es_full_gpus", E.full_gpus, "{:.0f}", "full allocation, GPUs")
v("es_left_gpus", E.full_gpus - E.step_up_gpus, "{:.0f}", "GPUs the step-up leaves to queued jobs")
v("es_small_tp", E.small_throughput, "{:.0f}", "epochs/h at the small size")
v("es_step_tp", E.step_up_throughput, "{:.1f}", "epochs/h at the step-up size")
v("es_full_tp", E.full_throughput, "{:.1f}", "epochs/h at the full size")
v("es_small_epochs", E.start_small_epochs, "{:.0f}", "start small, epochs")
v("es_wait_run_h", E.wait_run_h, "{:.0f}", "wait: hours running")
v("es_wait_epochs", E.wait_epochs, "{:.0f}", "wait, epochs")
v("es_step_p1", E.step_up_phase1_epochs, "{:.0f}", "step up, phase 1 epochs")
v("es_step_p2_h", E.step_up_phase2_h, "{:.2f}", "step up, phase 2 hours")
v("es_step_epochs", E.step_up_epochs, "{:.1f}", "step up, epochs")
v("es_resume_h", E.wait_h + E.rescale_h, "{:.2f}", "hour the step-up resumes")
gh, eg = E.gpu_hours, E.epochs_per_gpu_hour
for key in ("start_small", "wait", "step_up"):
    v(f"es_{key}_gpuh", gh[key], "{:,.0f}", f"{key}, GPU-hours")
v("es_small_per_kgh", eg["start_small"] * 10 ** 3, "{:.1f}", "start small, epochs per 1,000 GPU-h")
v("es_wait_per_kgh", eg["wait"] * 10 ** 3, "{:.2f}", "wait, epochs per 1,000 GPU-h")
v("es_step_per_kgh", eg["step_up"] * 10 ** 3, "{:.1f}", "step up, epochs per 1,000 GPU-h")
v("es_step_eff_pct", pct(E.step_up_efficiency), "{:.0f}", "step-up scaling efficiency, %")
v("es_full_eff_pct", pct(E.full_efficiency), "{:.0f}", "full scaling efficiency, %")
v("replica_gpus", REPLICA.replica_gpus, "{:.0f}", "175B self-contained replica, GPUs")
v("replica_tp", REPLICA.tensor_parallel, "{:.0f}", "its TP")
v("replica_pp", REPLICA.pipeline_depth, "{:.0f}", "its pipeline stages")
ml = Scenarios.MLPerfLlmScalingSeries.result
eos = ml.series[0]
v("eos_n_x", ratio(eos[-1]["accelerators"], eos[0]["accelerators"]), "{:.0f}", "Eos GPUs, x (Lecture 6)")
v("eos_t_x", ratio(eos[0]["score_minutes"], eos[-1]["score_minutes"]), "{:.1f}", "Eos time cut, x (Lecture 6)")

print("== Preemption (8.8.1)")
pr = PREEMPT.result
v("pre_lost_min", m(pr.lost_work, "minute"), "{:.0f}", "work lost, minutes")
v("pre_restart_min", m(pr.restart, "minute"), "{:.0f}", "relaunch, minutes")
v("pre_reload_s", m(pr.reload, "s"), "{:.1f}", "reload, s")
v("pre_warm_min", m(pr.warmup, "minute"), "{:.0f}", "warm up, minutes")
v("pre_min", m(pr.per_event, "minute"), "{:.1f}", "per event, minutes")
v("pre_usd", pr.cost_per_event, "{:.0f}", "per event, $")
v("pre_per_day", pr.events_per_day, "{:.0f}", "forced preemptions a day")
v("pre_day_h", m(pr.lost_per_day, "hour"), "{:.1f}", "lost a day, hours")
v("pre_day_gpuh", pr.gpu_hours_per_day, "{:.0f}", "lost a day, GPU-hours")

print("== Spot (8.13 fallacy; Systems Perspective 8.4)")
v("spot_usd", m(SPOT_RATE, "USD/hour"), "{:.2f}", "spot GPU-hour, $")
v("aws_notice_min", m(Ops.Preemption.AwsSpotNotice, "minute"), "{:.0f}", "AWS spot notice, minutes")
v("gcp_notice_s", m(Ops.Preemption.GcpPreemptibleNotice, "s"), "{:.0f}", "GCP preemptible notice, s")
v("spot_disc_pct", pct(SPOT.nominal_discount), "{:.0f}", "nominal discount, %")
v("spot_price_frac", 1 - SPOT.nominal_discount, "{:.2f}", "spot over on-demand price")
v("spot_ckpt_min", m(SPOT.checkpoint_overhead, "minute"), "{:.0f}", "checkpoint per interruption, min")
v("spot_restart_min", m(SPOT.restart_overhead, "minute"), "{:.0f}", "restart per interruption, min")
v("spot_lost_min", m(SPOT.lost_per_interruption, "minute"), "{:.0f}", "billed overhead per interruption, min")
v("spot_interval_h", m(SPOT.retained_interval, "hour"), "{:.0f}", "retained progress between interruptions, h")
v("spot_billed_h", SPOT.billed_over_retained * m(SPOT.retained_interval, "hour"), "{:.2f}", "billed hours per interval")
v("spot_eff_ratio", SPOT.effective_cost_ratio, "{:.2f}", "effective over on-demand price")
v("spot_eff_save_pct", pct(SPOT.effective_savings), "{:.0f}", "effective savings, %")
R = SPOTRUN
v("spotrun_params_b", m(R.workload.parameters, "Gparam"), "{:.0f}", "model, billions of parameters")
v("spotrun_gpus", R.gpus, "{:.0f}", "GPUs")
v("spotrun_days", R.run_days, "{:.0f}", "run, days")
v("spotrun_od_usd", R.on_demand_cost, "{:,.0f}", "on-demand bill, $")
v("spotrun_spot_usd", R.spot_total_cost, "{:,.0f}", "spot bill, $")
v("spotrun_od_h", R.run_hours, "{:.0f}", "on-demand hours")
v("spotrun_spot_h", R.spot_total_hours, "{:.1f}", "spot hours")
v("spotrun_save_pct", pct(R.savings_share), "{:.1f}", "spot cheaper, %")
v("spotrun_longer_pct", pct(R.longer_share), "{:.1f}", "spot longer, %")

print("== Sharing (Napkin Math 8.4)")
S = SHARE
v("share_tenants_word", S.tenants, "words", "small models per shared GPU")
v("share_small_b", m(S.small_model.parameters, "Gparam"), "{:.0f}", "small model, billions")
v("share_large_b", m(S.large_model.parameters, "Gparam"), "{:.0f}", "large model, billions")
v("share_large_word", S.gpus_per_large, "words", "GPUs per large replica, in prose")
v("share_large_gpus", S.gpus_per_large, "{:.0f}", "GPUs per large replica")
v("share_large_gb", m(S.large_weights, "GB"), "{:.0f}", "175B FP16 weights, GB")
v("share_dev_gb", m(S.device_memory, "GB"), "{:.0f}", "device budget, GB")
v("share_small_gb", m(S.small_footprint, "GB"), "{:.0f}", "7B footprint, GB")
v("share_pair_gb", m(S.shared_footprint, "GB"), "{:.0f}", "two 7B footprints, GB")
v("share_excl_unused_pct", pct(S.exclusive_unused), "{:.1f}", "one 7B, unused, %")
v("share_pair_unused_pct", pct(S.shared_unused), "{:.1f}", "two 7B, unused, %")
v("share_n_small", S.n_small, "{:.0f}", "small models")
v("share_n_large", S.n_large, "{:.0f}", "large models")
v("share_excl_gpus", S.exclusive_gpus, "{:.0f}", "GPUs, exclusive")
v("share_pair_gpus", S.shared_gpus, "{:.0f}", "GPUs, pairs shared")
v("share_saved_pct", pct(S.saved_share), "{:.1f}", "fewer GPUs, %")
v("share_value_usd", S.annual_value, "{:,.0f}", "released value, $ a year")
v("perlmutter_mem_pct", pct(float(FLEET_OPS.PerlmutterJobMemoryThresholdFraction)), "{:.0f}",
  "Perlmutter memory threshold, % (half the jobs used at most this much)")
for p in Hardware.Cloud.A100.mig_profiles:
    key = p.name.split(".")[0]
    v(f"mig_{key}_gb", m(p.gpu_memory, "GiB"), "{:.0f}", f"MIG {p.name} memory, GB")
    v(f"mig_{key}_sm", p.sm_count, "{:.0f}", f"MIG {p.name} SMs")

print("== Diurnal shift (Napkin Math 8.5)")
D = DIURNAL
v("di_fleet", D.fleet_gpus, "{:,.0f}", "fleet GPUs")
v("di_reserve", D.serving_reservation, "{:.0f}", "static serving reservation")
v("di_peak", D.peak_demand, "{:.0f}", "forecast serving peak")
v("di_buffer", D.surge_buffer, "{:.0f}", "surge buffer")
v("di_avg", D.serving_average, "{:.0f}", "average serving demand")
v("di_base", D.training_base, "{:.0f}", "training base")
v("di_headroom", D.headroom, "{:.0f}", "maintenance headroom")
v("di_static", D.static_active, "{:.0f}", "active, static split")
v("di_static_pct", pct(D.static_utilization), "{:.0f}", "utilization, static, %")
v("di_lent", D.lent, "{:.0f}", "lent to training on average")
v("di_dynamic", D.dynamic_active, "{:.0f}", "active, lending")
v("di_dynamic_pct", pct(D.dynamic_utilization), "{:.0f}", "utilization, lending, %")
v("di_value_musd", D.annual_value / 10 ** 6, "{:.1f}", "value, $M a year")

print("== Quota (8.11)")
v("hoard_gpus", HOARD.gpus, "{:.0f}", "reserved GPUs held idle")
v("hoard_weeks", m(HOARD.duration, "week"), "{:.0f}", "weeks idle")
v("hoard_weeks_word", m(HOARD.duration, "week"), "words", "the same, in prose")
v("hoard_hours", m(HOARD.duration, "hour"), "{:.0f}", "hours idle")
v("hoard_usd", HOARD.cost, "{:,.0f}", "idle cost, $")
v("quota_guarantee", QUOTA.guarantee, "{:.0f}", "team A guarantee")
v("quota_used", QUOTA.used, "{:.0f}", "team A in use")
v("quota_lent", QUOTA.lendable, "{:.0f}", "team A idle, lent")
v("quota_b_owns", QUOTA.borrower_guarantee, "{:.0f}", "team B guarantee")

print("== Utilization tiers (8.12.1)")
alloc, active, prod = (int(TIERS.AllocatedGpus), int(TIERS.ActiveEquivalents), int(TIERS.ProductiveEquivalents))
v("tier_alloc", alloc, "{:.0f}", "allocated GPUs")
v("tier_active", active, "{:.0f}", "active equivalents")
v("tier_prod", prod, "{:.0f}", "productive equivalents")
v("tier_active_pct", pct(active / alloc), "{:.0f}", "active share, %")
v("tier_prod_pct", pct(prod / alloc), "{:.0f}", "productive share, %")

if __name__ == "__main__":
    write(__file__)
