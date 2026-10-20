#!/usr/bin/env python3
"""Every number on the Lecture 9 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature, ReferenceStats) and the
calculations (Scenarios.*, each backed by a solver or a scenario type); the
chapter, Fleet Efficiency, reads the same scenarios. This file only selects
values, converts units, and formats them; it types no numbers. Each value is
recorded with v() under the key the deck prints as \\val{key}; running this
file writes images/numbers.tex.

Not from MLSysIM, and typed where they appear: the MLPerf system-file software
versions and the Cloudflare regular expression (quoted records), the 13B model
label of the compilation example, and vote and question settings that feed no
calculation (a gang holding half its nodes, a compiler upgrade adding minutes,
fifteen nodes waiting for one).

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Q_ = mlsysim.core.units.Q_
Models, Literature, Scenarios = mlsysim.Models, mlsysim.Literature, mlsysim.Scenarios
RUNS, FLEET, INCIDENT = Literature.TrainingRuns, Literature.FleetOperations, Literature.IncidentReports
NODE = mlsysim.Systems.Nodes.DGX_H100
H100 = mlsysim.Hardware.Cloud.H100
TIERS = mlsysim.ReferenceStats.UtilizationTierProfile
LLAMA3, LLAMA3_POOL = Scenarios.Llama3_405BGoodput16K, Scenarios.Llama3_405BOnSharedPool
SNAP = Scenarios.SharedPoolSchedulingSnapshot
JOB = Scenarios.Llama2_70BRuntimeGoodput128           # five runtime terms, one week held
MFU = Scenarios.Llama2_70BFleetMfu128                 # node 180 ms against fleet 245 ms
POOL = Scenarios.Llama2_70BSharedPoolGoodput128       # S x R x P, ceilings, price
TRADE = Scenarios.Llama2_70BTenthTwelfthTrade
DECODE = Scenarios.Llama2_70BDecodeFloorTp8
LN = Scenarios.LayerNormMeasuredH100
TPX = Scenarios.Llama2_70BNodeTensorParallelStep
DEC_AR = Scenarios.Llama2_70BDecodeAllReduceTp8
COMPILE = Scenarios.Decode13BCompilationPayback
TRACE = Scenarios.NsightTraceVolume1K
GAINS = Scenarios.MlperfInferenceSoftwareGainsDgxH100


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    return 100 * x


print("== Hook: Llama 3 405B (Dubey et al. 2024)")
l3 = LLAMA3.result
v("llama_gpus", m(Models.Language.Llama3_405B.training_accelerators_ref, "count"), "{:,.0f}", "H100s")
v("llama_days", float(RUNS.Llama3_405B_SnapshotDays), "{:.0f}", "snapshot, days")
v("llama_interrupts", float(RUNS.Llama3_405B_UnexpectedInterruptions), "{:.0f}", "unexpected interruptions")
v("llama_r_pct", pct(LLAMA3.runtime), "{:.0f}", "effective training time, %")
v("llama_r", LLAMA3.runtime, "{:.2f}", "runtime goodput R")
v("llama_p_pct", pct(LLAMA3.program), "{:.0f}", "MFU at 16,384 GPUs, %")
v("llama_p", LLAMA3.program, "{:.2f}", "program goodput P")
v("llama_p_gap", 1 - LLAMA3.program, "{:.2f}", "1 - P")
v("llama_r_loss_pct", pct(l3.runtime_share), "{:.0f}", "held capacity lost at runtime, %")
v("llama_p_loss_pct", pct(l3.program_share), "{:.0f}", "held capacity lost in kept steps, %")
v("llama_rp_pct", pct(l3.mpg), "{:.0f}", "held capacity that became progress, %")
v("llama_rp", l3.mpg, "{:.2f}", "R x P")
v("llama_r_loss_frac_words", round(1 / l3.runtime_share), "words", "a tenth: 1 / (1 - R)")
v("llama_pool", LLAMA3_POOL.result.mpg, "{:.2f}", "Llama 3 on the shared pool")

print("== Running case")
gpus = JOB.failures.n_devices
v("job_gpus", gpus, "{:.0f}", "running job GPUs")
v("job_nodes", gpus / NODE.accelerators_per_node, "{:.0f}", "nodes")
v("job_nodes_half", gpus / NODE.accelerators_per_node / (1 + 1), "{:.0f}", "half the gang's nodes")
v("node_gpus", NODE.accelerators_per_node, "{:.0f}", "GPUs per node")
v("node_gpus_words", NODE.accelerators_per_node, "words", "the same, in prose")
v("tiers_alloc", float(TIERS.AllocatedGpus), "{:.0f}", "allocated GPUs")
v("tiers_active", float(TIERS.ActiveEquivalents), "{:.0f}", "active GPUs")
v("tiers_prod", float(TIERS.ProductiveEquivalents), "{:.0f}", "productive GPUs")
v("tiers_prod_pct", pct(float(TIERS.ProductiveEquivalents) / float(TIERS.AllocatedGpus)), "{:.0f}",
  "productive / allocated, %")

print("== Measurement: window, MLPerf gains, traces")
r = JOB.result
v("mtbf_h", m(r.mtbf, "hour"), "{:.0f}", "running job MTBF, hours")
v("mtbf_days", m(r.mtbf, "day"), "{:.1f}", "the same, days")
for key, bench in [("gptj", "gptj-99"), ("llama2", "llama2-70b-99.9"), ("sdxl", "stable-diffusion-xl")]:
    g = GAINS.gain(bench)
    v(f"gain_{key}", g, "{:.2f}", f"{bench} gain, x")
    v(f"gain_{key}_1", g, "{:.1f}", "the same, as the chapter rounds it")
v("gain_flat", GAINS.gain("resnet50"), "{:.2f}", "ResNet-50 (and BERT) gain")
v("gain_flat_1", GAINS.gain("resnet50"), "{:.1f}", "the same, one decimal")
v("gain_months", GAINS.months(), "{:.0f}", "months from first to last round")
v("trace_gpus", TRACE.devices, "{:,.0f}", "traced GPUs")
v("trace_s", m(TRACE.window, "second"), "{:.0f}", "trace window, s")
v("trace_mb", m(TRACE.per_device, "MB"), "{:.0f}", "one GPU, MB")
v("trace_gb", m(TRACE.total, "GB"), "{:.0f}", "every GPU, GB")
v("megascale_slow_pct", pct(float(FLEET.MegaScaleSlowMachineFraction)), "{:.1f}", "MegaScale slow machines, %")
v("cf_cpu_pct", pct(float(INCIDENT.CloudflareOutage2019CpuUtilization)), "{:.0f}", "Cloudflare edge CPU, %")
v("cf_min", float(INCIDENT.CloudflareOutage2019Minutes), "{:.0f}", "Cloudflare outage, minutes")
v("cf_drop_pct", pct(float(INCIDENT.CloudflareOutage2019TrafficDropFraction)), "{:.0f}", "traffic lost, %")

print("== Scheduling goodput of the pool")
v("snap_assigned", SNAP.assigned_share, "{:.2f}", "assigned share")
v("snap_partial", SNAP.incomplete_gang_share, "{:.2f}", "incomplete gangs")
v("snap_input", SNAP.input_wait_share, "{:.2f}", "input waits in complete allocations")
v("snap_exec", SNAP.executing_share, "{:.2f}", "executing")
v("snap_free", SNAP.unassigned_share, "{:.2f}", "unassigned")
v("sched", SNAP.scheduling_goodput, "{:.2f}", "S")
v("sched_ceiling", SNAP.partial_hold_ceiling, "{:.2f}", "S with partial holds refilled")
v("google_s_pct", pct(float(FLEET.GoogleTpuSchedulingGoodputFloor)), "{:.0f}", "Google TPU S floor, %")
v("google_s", float(FLEET.GoogleTpuSchedulingGoodputFloor), "{:.2f}", "the same, as a fraction")

print("== Runtime goodput, five terms")
v("rt_loss_pct", pct(r.runtime_loss), "{:.0f}", "runtime loss, %")
v("runtime", r.runtime_goodput, "{:.2f}", "R")
v("runtime_gap", 1 - r.runtime_goodput, "{:.2f}", "1 - R")
v("tau_min", m(r.interval, "minute"), "{:.0f}", "Young-Daly interval, minutes")
v("tau_int_min", m(r.interruption_checkpoint_interval, "minute"), "{:.0f}", "with preemptions counted")
v("preempt_per_day_words", m(JOB.preemption_rate, "1/day"), "words", "budgeted preemptions a day")
v("preempt_min", m(r.preemption_cost, "minute"), "{:.1f}", "per preemption, minutes")
v("preempt_lost_min", m(JOB.lost_work, "minute"), "{:.0f}", "work lost per preemption, minutes")
v("write_s", m(r.write_time, "second"), "{:.1f}", "checkpoint write, s")
v("term_input_pct", pct(r.input_wait_share), "{:.1f}", "input waits, %")
v("term_preempt_pct", pct(r.preemption_share), "{:.1f}", "preemption, %")
v("term_ckpt_pct", pct(r.checkpoint_share), "{:.1f}", "checkpoints, %")
v("term_startup_pct", pct(r.startup_share), "{:.2f}", "startup, %")
v("term_recovery_pct", pct(r.recovery_share), "{:.2f}", "recovery, %")
v("startup_min", m(r.startup_time, "minute"), "{:.1f}", "startup, minutes")
short = JOB.variant(window=Q_(1, "hour"))
v("startup_hour_pct", pct(short.result.startup_share), "{:.0f}", "startup share of a one-hour job, %")
cross = JOB.crossover(100)
v("cross_gpus", round(cross, -3), "{:,.0f}", "failure terms pass input waits, GPUs")
big = JOB.at(int(m(Models.Language.Llama3_405B.training_accelerators_ref, "count"))).result
v("rt_loss_16k_pct", pct(big.runtime_loss), "{:.0f}", "runtime loss at 16,384 GPUs, %")

print("== Program goodput: MFU, node to fleet, rooflines")
# One decimal, as the PaLM table and the figure print them.
v("mfu_gpt3", pct(float(RUNS.Gpt3MfuPalmTable)), "{:.1f}", "GPT-3 MFU, %")
v("mfu_mtnlg", pct(float(RUNS.MtNlgMfu)), "{:.1f}", "MT-NLG MFU, %")
v("mfu_gopher", pct(float(RUNS.GopherMfu)), "{:.1f}", "Gopher MFU, %")
v("mfu_palm", pct(float(RUNS.PalmMfu)), "{:.1f}", "PaLM MFU, %")
v("llama_p8_pct", pct(float(RUNS.Llama3_405B_Mfu8K)), "{:.0f}", "Llama 3 MFU at 8,192 GPUs, %")
v("llama_p8", float(RUNS.Llama3_405B_Mfu8K), "{:.2f}", "the same, as a fraction")
v("llama_gpus_8k", float(RUNS.Llama3_405B_Gpus8K), "{:,.0f}", "Llama 3 first-stage GPUs")
v("palm_chips", m(Models.Language.PaLM_540B.training_accelerators_ref, "count"), "{:,.0f}", "PaLM TPU v4 chips")
v("llama_mfu_drop", pct(float(RUNS.Llama3_405B_Mfu8K) - float(RUNS.Llama3_405B_Mfu16K)), "{:.0f}",
  "MFU points lost doubling GPUs")
v("node_mfu_pct", pct(MFU.node_mfu), "{:.0f}", "node MFU, %")
v("node_ms", m(MFU.node_step_time, "ms"), "{:.0f}", "node step, ms")
v("fleet_ms", m(MFU.fleet_step_time, "ms"), "{:.0f}", "fleet step, ms")
v("fleet_s", m(MFU.fleet_step_time, "s"), "{:.3f}", "fleet step, s")
v("extra_ms", m(MFU.extra_step_time, "ms"), "{:.0f}", "time beyond the node, ms")
kept = (MFU.node_step_time / MFU.fleet_step_time).to("").magnitude
v("kept", kept, "{:.3f}", "node step / fleet step")
v("kept_pct", pct(MFU.scaling_efficiency), "{:.0f}", "scaling efficiency, as the chapter rounds it")
v("fleet_mfu_est", MFU.node_mfu * kept, "{:.3f}", "node MFU x kept")
v("program", MFU.fleet_mfu, "{:.2f}", "P = fleet MFU")
v("program_pct", pct(MFU.fleet_mfu), "{:.0f}", "the same, %")
v("useful_pflop", m(MFU.fleet_useful_work, "PFLOP"), "{:.1f}", "useful work per step, PFLOP")
v("peak_pflop", m(MFU.fleet_peak_work, "PFLOP"), "{:.0f}", "peak work per step, PFLOP")
v("h100_peak", m(H100.compute.peak_flops, "TFLOP/s"), "{:.0f}", "H100 dense FP16, TFLOP/s")
v("h100_bw", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM, TB/s")
d = DECODE.result
v("ridge", m(LN.ridge, "flop/byte"), "{:.0f}", "H100 ridge, FLOP/byte")
v("decode_flops_per_param", float(Literature.Chinchilla.DecodeConstant), "{:.0f}", "decode FLOPs per weight")
v("fp16_bytes", m(mlsysim.core.units.BYTES_FP16, "byte"), "{:.0f}", "bytes per FP16 weight")
v("decode_ai", m(d.intensity, "flop/byte"), "{:.0f}", "decode intensity, FLOP/byte")
v("decode_rate", m(d.attainable_flops, "TFLOP/s"), "{:.2f}", "decode rate at the floor, TFLOP/s")
v("decode_peak_pct", pct(d.peak_fraction), "{:.2f}", "share of peak, %")
v("decode_peak_pct_1", pct(d.peak_fraction), "{:.1f}", "the same, as the chapter rounds it")
v("decode_ms", m(d.memory_time, "ms"), "{:.1f}", "weight-streaming floor, ms per token")
v("ln_tflops", m(LN.achieved_rate, "TFLOP/s"), "{:.0f}", "LayerNorm achieved, TFLOP/s")
v("ln_tbs", m(LN.achieved_bandwidth, "TB/s"), "{:.1f}", "LayerNorm achieved, TB/s")
v("ln_ai", m(LN.point_intensity, "flop/byte"), "{:.1f}", "LayerNorm intensity, FLOP/byte")
v("ln_bw_pct", pct(LN.bandwidth_utilization), "{:.1f}", "LayerNorm share of HBM peak, %")
v("tp_ms", m(TPX.result.communication_time, "ms"), "{:.0f}", "TP exchanges per node step, ms")
v("dec_ar_kb", m(DEC_AR.message, "KB"), "{:.1f}", "decode AllReduce, KB")
v("dec_ar_x", round(DEC_AR.startup_to_transfer, -1), "{:.0f}", "launch over wire time")

print("== The whole pool: MPG, levers, price")
g = POOL.result
v("mpg", g.mpg, "{:.2f}", "MPG")
v("mpg_3", g.mpg, "{:.3f}", "MPG, three places")
v("mpg_pct", pct(g.mpg), "{:.0f}", "MPG, %")
v("after_r", g.after_runtime, "{:.2f}", "S x R")
v("job_rp", g.job_product, "{:.2f}", "R x P")
v("program_gap", 1 - POOL.program, "{:.2f}", "1 - P")
v("s_loss_pct", pct(g.scheduling_share), "{:.0f}", "lost before any job holds it, %")
v("r_loss_pct", pct(g.runtime_share), "{:.0f}", "lost at runtime, % of fleet")
v("p_loss_pct", pct(g.program_share), "{:.0f}", "lost in retained steps, % of fleet")
v("lvl_s", pct(g.after_scheduling), "{:.0f}", "waterfall level after S, %")
v("lvl_r", pct(g.after_runtime), "{:.0f}", "waterfall level after R, %")
v("runtime_ceiling", POOL.ceilings["R"], "{:.2f}", "R with input waits removed")
v("program_ceiling", POOL.ceilings["P"], "{:.2f}", "P at the node baseline")
for k, name in (("S", "s"), ("R", "r"), ("P", "p")):
    v(f"lever_{name}_pct", pct(g.ceiling_ratio[k] - 1), "{:.0f}", f"{k} lever gain, %")
    v(f"lever_{name}_mpg", g.mpg_at_ceiling[k], "{:.2f}", f"MPG with {k} at its ceiling")
    v(f"point_{name}", g.point_value[k], "{:.2f}", f"one point of {k}")
v("price", m(POOL.device_hour_rate, "USD/hour"), "{:.0f}", "fleet rate, $ per GPU-hour")
v("useful_price", m(g.useful_rate, "USD/hour"), "{:.2f}", "useful GPU-hour, $")
v("cost_x", g.cost_multiple, "{:.2f}", "multiple of the raw price")
v("cost_x_1", g.cost_multiple, "{:.1f}", "the same, as the chapter rounds it")
v("job_cost_x", g.job_cost_multiple, "{:.2f}", "held-hours chargeback multiple")
v("p_cut_pct", pct(1 - 1 / g.ceiling_ratio["P"]), "{:.0f}", "cost cut at the P ceiling, %")
v("r_headroom_pct", pct(g.runtime_headroom), "{:.0f}", "R headroom, %")
v("trade_p", TRADE.program / POOL.program, "{:.2f}", "P ratio of the trade")
r_den = 1 / (1 - TRADE.runtime / POOL.runtime)
v("trade_r_den", r_den, "{:.0f}", "R falls by one part in this many")
v("trade_r_num", r_den - 1, "{:.0f}", "parts of R kept")
v("trade_x", POOL.trade_ratio(TRADE), "{:.3f}", "product of the ratios")

print("== Compilation payback")
v("cmp_rate", m(COMPILE.token_rate, "1/s"), "{:.0f}", "eager tokens/s")
v("cmp_step", m(COMPILE.step_time, "ms"), "{:.2f}", "eager step, ms")
sh, rm = COMPILE.shares, COMPILE.removed
v("cmp_mm_pct", pct(sh["matmul"]), "{:.0f}", "matmul share kept, %")
v("cmp_ew_cut_pct", pct(rm["elementwise"]), "{:.0f}", "element-wise removed, %")
v("cmp_ln_cut_pct", pct(rm["launch"]), "{:.0f}", "launch removed, %")
v("cmp_mm", sh["matmul"], "{:.2f}", "matmul share")
v("cmp_ew", sh["elementwise"], "{:.2f}", "element-wise share")
v("cmp_ew_keep", 1 - rm["elementwise"], "{:.2f}", "element-wise kept")
v("cmp_ln", sh["launch"], "{:.2f}", "launch share")
v("cmp_ln_keep", 1 - rm["launch"], "{:.2f}", "launch kept")
v("cmp_ew_pct", pct(sh["elementwise"]), "{:.0f}", "element-wise share, %")
v("cmp_ln_pct", pct(sh["launch"]), "{:.0f}", "launch share, %")
v("cmp_frac", COMPILE.step_fraction, "{:.3f}", "compiled step fraction")
v("cmp_new_ms", m(COMPILE.compiled_step, "ms"), "{:.1f}", "compiled step, ms")
v("cmp_saved_ms", m(COMPILE.saved_per_token, "ms"), "{:.1f}", "saved per token, ms")
v("cmp_build_min", m(COMPILE.builds["autotune"], "minute"), "{:.0f}", "autotuning build, min")
v("cmp_tokens", round(COMPILE.tokens_to_repay("autotune"), -3), "{:,.0f}", "tokens to repay")
v("cmp_tokens_ch", round(COMPILE.tokens_to_repay("autotune"), -4), "{:,.0f}", "the same, as the chapter rounds it")
v("cmp_repay_min", m(COMPILE.time_to_repay("autotune"), "minute"), "{:.0f}", "minutes of compiled generation")
v("cmp_replicas", COMPILE.replicas, "{:.0f}", "replicas")
v("cmp_shared_s", m(COMPILE.shared_repay("autotune"), "s"), "{:.0f}", "shared artifact, s per replica")
v("cmp_rollout_h", m(COMPILE.rollout_cost("autotune"), "hour"), "{:.0f}", "per-replica builds, GPU-hours")

print("== Frontier: Llama 3 at 8,192 and 16,384 GPUs (equal R assumed)")
p8, p16 = float(RUNS.Llama3_405B_Mfu8K), float(RUNS.Llama3_405B_Mfu16K)
dev_ratio = m(Models.Language.Llama3_405B.training_accelerators_ref, "count") / float(RUNS.Llama3_405B_Gpus8K)
thr = dev_ratio * p16 / p8
v("fr_dev_ratio", dev_ratio, "{:.0f}", "device ratio")
v("fr_thr", thr, "{:.2f}", "throughput ratio")
v("fr_time", 1 / thr, "{:.2f}", "time ratio")
v("fr_cost", p8 / p16, "{:.3f}", "GPU-hours per unit of work ratio")
v("fr_cost_pct", pct(p8 / p16 - 1), "{:.1f}", "more GPU-hours, %")

if __name__ == "__main__":
    write(__file__)
