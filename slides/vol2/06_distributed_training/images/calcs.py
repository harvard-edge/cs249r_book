#!/usr/bin/env python3
"""Every number on the Lecture 6 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature) and the calculations
(Scenarios.*, each backed by a solver or a scenario type); the chapter reads
the same scenarios. This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Systems, Literature, Scenarios = (mlsysim.Hardware, mlsysim.Systems,
                                            mlsysim.Literature, mlsysim.Scenarios)
H100 = Hardware.Cloud.H100
RUNS = Literature.TrainingRuns
A = Scenarios.Gpt3DataParallel1K                  # pure data parallelism, 1,024 H100s
B = Scenarios.Gpt3TensorDataParallel1K            # TP 8 x DP 128
C = Scenarios.Gpt3ThreeDParallel1K                # TP 8 x PP 4 x DP 32
L16K = Scenarios.Gpt3ThreeDParallel16K            # TP 8 x PP 16 x DP 128
GPT2_1, GPT2_4, GPT2_ACC = (Scenarios.Gpt2DataParallelOneNode, Scenarios.Gpt2DataParallelFourNodes10GbE,
                            Scenarios.Gpt2GradientAccumulationOneNode)
MLPERF = Scenarios.MLPerfLlmScalingSeries
STRAG = Scenarios.StragglerFreeStep1K
CB = Scenarios.CriticalBatchPilotTrace7B
BSP8, BSP64, SSP64 = (Scenarios.TransformerBsp8Workers, Scenarios.TransformerBsp64Workers,
                      Scenarios.TransformerSsp64Workers)
WALL = Scenarios.NotableModelMemoryWall
ZERO7 = Scenarios.Llama2_7B_ZeroStages64
ZERO175 = Scenarios.Gpt3Zero3On64
NAIVE, GPIPE = Scenarios.NaiveFourStagePipeline, Scenarios.GPipeFourStageWalkthrough
P8M32, P16M32, P8M8 = (Scenarios.Pipeline8Stage32Micro, Scenarios.Pipeline16Stage32Micro,
                       Scenarios.Pipeline8Stage8Micro)
TPL = Scenarios.Gpt3TensorParallelLayerTp8
RING = Scenarios.H100RingAttentionChunk
ACT = Scenarios.Gpt3ActivationsMicrobatch4
RLHF = Scenarios.Llama2_70B_PpoAlignment
FAIL = Scenarios.GpuFailuresTraining1K
RESID = Scenarios.ResidualOnePercent512
GOYAL = Literature.Series.GoyalLargeMinibatchResNet50
MEGATRON = Literature.Series.MegatronLmWeakScaling


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities (or two numbers)."""
    r = a / b
    return r.to("").magnitude if hasattr(r, "to") else r


def pct(x):
    return 100 * x


print("== Hook: the one-hour ImageNet run (Goyal et al. 2017)")
g_gpus, g_batch = float(RUNS.Goyal2017Gpus), float(RUNS.Goyal2017Minibatch)
g_base_gpus, g_base_batch = float(RUNS.Goyal2017BaselineGpus), float(RUNS.Goyal2017BaselineMinibatch)
v("goyal_gpus", g_gpus, "{:.0f}", "GPUs in the one-hour run")
v("goyal_hours_word", float(RUNS.Goyal2017TrainingHours), "words", "run time, hours, in prose")
v("goyal_hours", float(RUNS.Goyal2017TrainingHours), "{:.0f}", "run time, hours")
v("goyal_base_hours", float(RUNS.Goyal2017BaselineHours), "{:.0f}", "8-GPU baseline, hours")
v("goyal_base_gpus", g_base_gpus, "{:.0f}", "baseline GPUs")
v("goyal_base_batch", g_base_batch, "{:.0f}", "baseline global batch")
v("goyal_batch", g_batch, "{:,.0f}", "one-hour run global batch")
v("goyal_batch_k", g_batch / 10 ** 3, "{:.0f}", "the same, in thousands (the paper's 8k)")
v("goyal_servers", ratio(g_gpus, float(RUNS.Goyal2017GpusPerServer)), "{:.0f}", "Big Basin servers")
v("goyal_per_gpu", ratio(g_batch, g_gpus), "{:.0f}", "images per GPU")
v("goyal_batch_x", ratio(g_batch, g_base_batch), "{:.0f}", "batch growth, x")
v("goyal_gpu_x", ratio(g_gpus, g_base_gpus), "{:.0f}", "GPU growth, x")
v("goyal_eff_pct", pct(float(RUNS.Goyal2017ScalingEfficiency)), "{:.0f}", "scaling efficiency, %")
v("goyal_eth_gbps", float(RUNS.Goyal2017EthernetGbps), "{:.0f}", "Ethernet, Gb/s")
v("goyal_epochs", float(RUNS.Goyal2017Epochs), "{:.0f}", "epochs")
v("goyal_warmup_epochs", float(RUNS.Goyal2017WarmupEpochs), "{:.0f}", "warmup epochs")
t1 = {(r["global_batch"], r["warmup"]): r["top1_error_pct"] for r in GOYAL.entries if r["source"] == "table1"}
fig3 = {r["global_batch"]: r["top1_error_pct"] for r in GOYAL.entries if r["source"] == "figure3"}
v("goyal_err_base", t1[(g_base_batch, "none")], "{:.2f}", "batch 256 baseline error, %")
v("goyal_err_nowarm", t1[(g_batch, "none")], "{:.2f}", "8k, no warmup, %")
v("goyal_err_warm", t1[(g_batch, "gradual")], "{:.2f}", "8k, gradual warmup, %")
upto = [e for b, e in fig3.items() if b <= g_batch]
num_k = round(g_batch / 10 ** 3)
v("goyal_spread", max(upto) - min(upto), "{:.1f}", "error spread up to 8k, points")
for i, b in enumerate(sorted(b for b in fig3 if b > g_batch), start=1):
    v(f"goyal_up{i}_k", ratio(b, g_batch) * num_k, "{:.0f}", "larger batch, in the paper's k")
    v(f"goyal_up{i}_err", fig3[b], "{:.2f}", "its error, %")

print("== Running case: 175B on the 1,024-GPU reference cluster")
ra = A.result
cluster = Systems.Clusters.Training_1K
v("params_b", m(A.workload.parameters, "Gparam"), "{:.0f}", "running model, billions of parameters")
v("state_bpp", m(mlsysim.ReferenceStats.MixedPrecisionAdam.TrainingStateBytesPerParameter, "byte/param"),
  "{:.0f}", "mixed-precision Adam, B per parameter")
v("opt_bpp", m(mlsysim.ReferenceStats.MixedPrecisionAdam.OptimizerStateBytesPerParameter, "byte/param"),
  "{:.0f}", "optimizer state, B per parameter")
v("fp16_bytes", m(mlsysim.BYTES_FP16, "byte"), "{:.0f}", "bytes per 16-bit value")
v("state_gb", m(ra.state_total, "GB"), "{:,.0f}", "175B training state, GB")
v("state_tb", m(ra.state_total, "TB"), "{:.1f}", "the same, TB")
v("hbm_gb", m(A.device_memory, "GB"), "{:.0f}", "H100 memory, published label, GB")
v("state_gpus", ra.min_devices_for_state, "{:.0f}", "fewest H100s that hold the state")
v("ref_gpus", A.n_devices, "{:,.0f}", "reference cluster GPUs")
v("ref_nodes", cluster.count, "{:.0f}", "reference cluster nodes")
v("gpus_per_node", cluster.node.accelerators_per_node, "{:.0f}", "GPUs per node")
v("nvlink_gbs", m(H100.nvlink.bandwidth_per_direction, "GB/s"), "{:.0f}", "NVLink per direction, GB/s")
v("ndr_gbs", m(A.link_bandwidth, "GB/s"), "{:.0f}", "NDR port, GB/s")
v("nvlink_ndr_x", ratio(H100.nvlink.bandwidth_per_direction, A.link_bandwidth), "{:.0f}", "NVLink / NDR")
v("grad_gb", m(ra.gradient_bytes, "GB"), "{:.0f}", "FP16 gradients, GB")
v("ring_gb", m(ra.ring_bytes, "GB"), "{:.0f}", "ring AllReduce bytes per GPU at N = 1,024, GB")
v("ar_a_s", m(ra.allreduce_time, "s"), "{:.0f}", "full-gradient AllReduce at 50 GB/s, s")

print("== Replicas: the scaling wall (configuration A)")
v("batch_tokens_m", A.batch_tokens / 10 ** 6, "{:.0f}", "tokens per update, millions")
v("flops_per_pt", A.flops_per_param_token, "{:.0f}", "FLOPs per parameter per token")
v("step_eflops", m(ra.step_flops, "EFLOP"), "{:.1f}", "step FLOPs, x 10^18")
v("local_eff_pct", pct(A.local_efficiency), "{:.0f}", "local share of peak, %")
v("fp8_tflops", m(H100.compute.precision_flops["fp8"], "TFLOP/s"), "{:,.0f}", "H100 FP8 peak, TFLOP/s")
v("t_compute_s", m(ra.compute_time, "s"), "{:.1f}", "compute window per step, s")
v("exposed_a_s", m(ra.exposed_time, "s"), "{:.1f}", "exposed AllReduce, s")
v("step_a_s", m(ra.step_time, "s"), "{:.1f}", "step with exposed AllReduce, s")
v("eff_a_pct", pct(ra.efficiency), "{:.0f}", "configuration A efficiency, %")
v("resid_pct", pct(RESID.residual), "{:.0f}", "residual of the one-device step, %")
v("resid_gpus", RESID.devices, "{:.0f}", "GPUs")
v("resid_share_pct", pct(RESID.residual_share), "{:.0f}", "residual's share of the step, %")

print("== Replicas: GPT-2 inside and across nodes")
r1, r4, racc = GPT2_1.result, GPT2_4.result, GPT2_ACC.result
v("gpt2_step_s", m(GPT2_1.step_time, "s"), "{:.1f}", "compute per step, s")
v("gpt2_step_ms", m(GPT2_1.step_time, "ms"), "{:,.0f}", "compute per step, ms")
v("gpt2_gpus_1", r1.n_devices, "{:.0f}", "one node, GPUs")
v("gpt2_sync1_gb", m(r1.intra_bytes, "GB"), "{:.1f}", "in-node sync bytes, GB")
v("gpt2_gpus_4", r4.n_devices, "{:.0f}", "four nodes, GPUs")
v("gpt2_sync4_gb", m(r4.inter_bytes, "GB"), "{:.1f}", "cross-node sync bytes, GB")
v("eth10_gbs", m(GPT2_4.inter_bandwidth, "GB/s"), "{:.2f}", "10 GbE, GB/s")
v("eth10_gbps", m(GPT2_4.inter_bandwidth, "Gbps"), "{:.0f}", "10 GbE, Gb/s")
v("gpt2_comm1_ms", m(r1.communication_time, "ms"), "{:.1f}", "in-node AllReduce, ms")
v("gpt2_eff1_pct", pct(r1.efficiency), "{:.1f}", "one node efficiency, %")
v("gpt2_inter_ms", m(r4.inter_time, "ms"), "{:,.0f}", "cross-node AllReduce, ms")
v("gpt2_comm4_ms", m(r4.communication_time, "ms"), "{:,.1f}", "four-node communication, ms")
v("gpt2_comm4_round_ms", m(r4.communication_time, "ms"), "{:,.0f}", "the same, whole ms")
v("gpt2_step4_ms", m(r4.update_time, "ms"), "{:,.1f}", "four-node step, ms")
v("gpt2_step1_ms", m(r1.update_time, "ms"), "{:,.0f}", "one-node step, whole ms")
v("gpt2_step4_round_ms", m(r4.update_time, "ms"), "{:,.0f}", "four-node step, whole ms")
v("gpt2_eff4_pct", pct(r4.efficiency), "{:.1f}", "four nodes efficiency, %")
v("gpt2_speedup4", r4.speedup, "{:.1f}", "four nodes speedup, x")
v("gpt2_accum", GPT2_ACC.accumulation_steps, "{:.0f}", "accumulation steps")
v("gpt2_batch", racc.global_batch, "{:.0f}", "global batch, both layouts")
v("gpt2_acc_hours", m(racc.training_time, "hour"), "{:.1f}", "accumulation run, hours")
v("gpt2_acc_usd", racc.cost, "{:.0f}", "accumulation run, $")
v("gpt2_4_usd", r4.cost, "{:.0f}", "four-node run, $")
v("gpt2_4_hours", m(r4.training_time, "hour"), "{:.1f}", "four-node run, hours")
v("node_hour_usd", m(GPT2_1.node_hour_rate, "USD/hour"), "{:.0f}", "node-hour price, $")

print("== Replicas: MLPerf Training series")
ml = MLPERF.result
eos = ml.series[0]
v("eos_n0", eos[0]["accelerators"], "{:.0f}", "Eos smallest run, H100s")
v("eos_n1", eos[-1]["accelerators"], "{:,.0f}", "Eos largest run, H100s")
v("eos_t0", eos[0]["score_minutes"], "{:.0f}", "Eos at 512, minutes")
v("eos_t1", eos[-1]["score_minutes"], "{:.1f}", "Eos at 10,752, minutes")
v("eos_n_x", ratio(eos[-1]["accelerators"], eos[0]["accelerators"]), "{:.0f}", "accelerators, x")
v("eos_t_x", ratio(eos[0]["score_minutes"], eos[-1]["score_minutes"]), "{:.1f}", "time cut, x")
v("eos_eff_pct", pct(ml.efficiencies[0]), "{:.0f}", "Eos efficiency, %")
v("mlperf_min_pct", pct(min(ml.efficiencies)), "{:.0f}", "lowest series efficiency, %")
v("mlperf_max_pct", pct(max(ml.efficiencies)), "{:.0f}", "highest series efficiency, %")

print("== Batches: stragglers, the critical batch, SSP")
v("strag_pct", pct(1 - STRAG.availability), "{:.0f}", "per-GPU late steps, %")
v("strag_on", STRAG.availability, "{:.2f}", "per-GPU on-time probability")
v("strag_n", STRAG.n_components, "{:,.0f}", "workers")
v("strag_n_plain", STRAG.n_components, "{:.0f}", "workers, no comma (exponent)")
v("strag_expected", STRAG.n_components * (1 - STRAG.availability), "{:.0f}", "expected late GPUs per step")
for key, n in (("node", cluster.node.accelerators_per_node), ("100", 100)):
    v(f"strag_{key}_pct", pct(STRAG.variant(n_components=n).all_up), "{:.0f}", f"on-time steps at {n} GPUs, %")
v("strag_1k_pct", pct(STRAG.all_up), "{:.3f}", "on-time steps at 1,000 GPUs, %")
v("cb_trace_m", CB.gradient_noise_trace / 10 ** 6, "{:.1f}", "tr(Sigma), millions")
v("cb_norm", CB.gradient_norm_sq, "{:.0f}", "squared gradient norm")
v("cb_devices", CB.devices, "{:.0f}", "A100s")
v("cb_local", CB.local_batch, "{:.0f}", "sequences per GPU")
v("cb_accum", CB.accumulation_steps, "{:.0f}", "accumulation steps")
v("cb_global", CB.global_batch, "{:,.0f}", "proposed global batch")
v("cb_bcrit", CB.critical_batch, "{:,.0f}", "critical batch")
v("cb_per_gpu", CB.per_device_samples, "{:.0f}", "samples per GPU per update")
v("cb_crossover", CB.crossover_devices, "{:.0f}", "GPUs at the crossover")
v("cb_params_b", m(Scenarios.Llama2_7B_ZeroStages64.workload.parameters, "Gparam"), "{:.0f}",
  "pilot model, billions of parameters")
v("ssp_base_iters", BSP8.base_iterations, "{:,.0f}", "one-GPU iterations")
for key, s in (("bsp8", BSP8), ("bsp64", BSP64), ("ssp64", SSP64)):
    v(f"{key}_workers", s.workers, "{:.0f}", "workers")
    v(f"{key}_iters", s.iterations, "{:,.0f}", "iterations to target")
    v(f"{key}_comm_pct", pct(s.communication_overhead), "{:.0f}", "communication overhead, %")
    v(f"{key}_speedup", s.speedup, "{:.1f}", "speedup, x")
    v(f"{key}_usd", s.cost, "{:.0f}", "allocated cost, $")
v("ssp_s", SSP64.staleness, "{:.0f}", "SSP staleness bound")
v("ssp_batch", BSP8.per_worker_batch, "{:.0f}", "examples per worker")
v("gpu_hour_usd", m(BSP8.device_hour_rate, "USD/hour"), "{:.0f}", "GPU-hour price, $")

print("== Sharding: the memory wall and ZeRO")
wall = WALL.result
v("wall_growth", wall.capacity_growth, "{:.0f}", "accelerator memory, K80 to B200, x")
v("wall_orders_word", round(wall.demand_orders), "words", "demand span, orders of magnitude, in prose")
v("wall_gap", wall.crossover_gap, "{:.0f}", "GPT-3 state over 2020's largest GPU, x")
z7 = ZERO7.result
v("z7_params_b", m(ZERO7.workload.parameters, "Gparam"), "{:.0f}", "ZeRO example, billions of parameters")
v("z7_gpus", ZERO7.n_devices, "{:.0f}", "ZeRO group, GPUs")
v("a100_gb", m(Hardware.Cloud.A100.memory.capacity, "GiB"), "{:.0f}", "A100 memory, published label, GB")
v("z7_ddp_gb", m(z7.stage_state["DDP"], "GB"), "{:.0f}", "DDP per GPU, GB")
v("z7_opt_gb", m(z7.optimizer_state, "GB"), "{:.0f}", "optimizer state, GB")
v("z7_z1_gb", m(z7.stage_state["ZeRO-1"], "GB"), "{:.1f}", "ZeRO-1 per GPU, GB")
v("z7_z2_gb", m(z7.stage_state["ZeRO-2"], "GB"), "{:.1f}", "ZeRO-2 per GPU, GB")
v("z7_z3_gb", m(z7.stage_state["ZeRO-3"], "GB"), "{:.2f}", "ZeRO-3 per GPU, GB")
v("z7_weights_gb", m(z7.weights, "GB"), "{:.0f}", "7B FP16 weights (and gradient buffer), GB")
v("z3_traffic_x", z7.traffic_multiple["ZeRO-3"], "{:.1f}", "ZeRO-3 bytes over DDP's, x")
z175 = ZERO175.result
v("z175_gpus", ZERO175.n_devices, "{:.0f}", "175B ZeRO-3 group, GPUs")
v("z175_state_gb", m(z175.stage_state["ZeRO-3"], "GB"), "{:.2f}", "175B state per GPU, GB")
v("z175_act_gb", m(z175.activations, "GB"), "{:.1f}", "checkpointed activations, GB")
v("z175_total_gb", m(z175.zero3_total_per_device, "GB"), "{:.1f}", "state plus activations, GB")
v("z175_seq", ZERO175.activations.workload.context_length, "{:,.0f}", "sequence length, tokens")
v("z175_gather_tb", m(z175.gather_traffic, "TB"), "{:.1f}", "regather traffic per GPU per step, TB")
v("z175_gather_x", ratio(z175.gather_time, ra.compute_time), "{:.0f}", "regather time over compute, x")
v("z175_gather_x_word", ratio(z175.gather_time, ra.compute_time), "words", "the same, in prose")

print("== Splitting: pipelines, tensor parallelism, more axes")
v("naive_busy", NAIVE.slots - NAIVE.idle_slots, "{:.0f}", "naive split, busy slots per GPU")
v("naive_slots", NAIVE.slots, "{:.0f}", "naive split, slots per step")
v("naive_idle", NAIVE.idle_slots, "{:.0f}", "naive split, idle slots")
v("pipe_p", GPIPE.stages, "{:.0f}", "GPipe example, stages")
v("pipe_busy", GPIPE.slots - GPIPE.idle_slots, "{:.0f}", "GPipe, busy slots per GPU")
v("pipe_slots", GPIPE.slots, "{:.0f}", "GPipe, slots per step")
v("pipe_idle", GPIPE.idle_slots, "{:.0f}", "GPipe, idle slots")
v("pipe_bubble_pct", pct(GPIPE.bubble), "{:.0f}", "GPipe bubble, %")
v("pipe_fill", GPIPE.fill_slots, "{:.0f}", "p - 1")
v("pipe_phase", GPIPE.phase_slots, "{:.0f}", "m + p - 1")
for key, s in (("p8m32", P8M32), ("p16m32", P16M32), ("p8m8", P8M8)):
    v(f"{key}_p", s.stages, "{:.0f}", "stages")
    v(f"{key}_m", s.microbatches, "{:.0f}", "microbatches")
    v(f"{key}_fill", s.fill_slots, "{:.0f}", "p - 1")
    v(f"{key}_phase", s.phase_slots, "{:.0f}", "m + p - 1")
v("p8m32_bubble_pct", pct(P8M32.bubble), "{:.0f}", "p = 8, m = 32 bubble, %")
v("p16m32_bubble_pct", pct(P16M32.bubble), "{:.1f}", "p = 16, m = 32 bubble, %")
v("p16m32_bubble_round_pct", pct(P16M32.bubble), "{:.0f}", "the same, whole %")
v("p8m8_bubble_pct", pct(P8M8.bubble), "{:.0f}", "p = 8, m = 8 bubble, %")
target = 1 / 10  # the 10 percent bubble target the slide states
v("bubble_target_pct", pct(target), "{:.0f}", "bubble target, %")
v("bubble_m_coef", ratio(1 - target, target), "{:.0f}", "m > coef x (p - 1) for the target")
v("p16_m_needed", P16M32.microbatches_for_bubble(target), "{:.0f}", "p = 16: microbatches above which the bubble is below 10%")
v("depth_word", ratio(L16K.pp_size, C.pp_size), "words", "PP 16 over PP 4, in prose")
tl = TPL.result
v("tp_degree", TPL.tp_size, "{:.0f}", "tensor-parallel degree")
v("tp_tokens", TPL.tokens, "{:,.0f}", "tokens per sequence")
v("tp_compute_ms", m(tl.compute_time, "ms"), "{:.1f}", "one layer's compute per GPU, ms")
v("tp_nvlink_ms", m(tl.transfer_times["NVLink"], "ms"), "{:.1f}", "TP exchange over NVLink, ms")
v("tp_ndr_ms", m(tl.transfer_times["InfiniBand NDR"], "ms"), "{:.1f}", "TP exchange over one NDR port, ms")
v("gpt3_layers", TPL.workload.layers, "{:.0f}", "GPT-3 layers")
meg = {(r["mode"], r["gpus"]): r["efficiency_pct"] for r in MEGATRON.entries}
for mode, n in sorted(meg):
    if mode == "model" and n > 1:
        v(f"meg_{n:.0f}_pct", meg[(mode, n)], "{:.0f}", f"Megatron {n:.0f}-way TP efficiency, %")
v("meg_top_pct", meg[("model+data", max(n for mode, n in meg))], "{:.0f}", "Megatron largest run, %")
v("meg_gpus", float(RUNS.MegatronLm2019Gpus), "{:.0f}", "Megatron largest run, GPUs")
v("meg_servers", ratio(float(RUNS.MegatronLm2019Gpus), max(n for mode, n in meg if mode == "model")),
  "{:.0f}", "DGX-2H servers")
v("meg_base_b", float(RUNS.MegatronLm2019BaselineParams) / 10 ** 9, "{:.1f}", "one-GPU model, billions")
v("meg_top_b", float(RUNS.MegatronLm2019Params) / 10 ** 9, "{:.1f}", "8-GPU model, billions")
v("nvswitch_gbs", m(Hardware.Cloud.V100.nvlink.bandwidth, "GB/s"), "{:.0f}", "V100 NVSwitch, GB/s")
chunk = RING.min_chunk_tokens
v("ring_nvlink_tokens", round(chunk["NVLink"], -2), "{:,.0f}", "ring-attention chunk on NVLink, tokens")
v("ring_ndr_tokens", round(chunk["InfiniBand NDR"], -2), "{:,.0f}", "ring-attention chunk on NDR, tokens")
v("dlrm_tb", m(mlsysim.ReferenceStats.RecommenderAtScaleProfile.EmbeddingTableSize, "TB"), "{:.0f}",
  "embedding tables, TB")

print("== Placement: the scoreboard for the 175B job")
rb, rc, r16 = B.result, C.result, L16K.result
v("tp_b", B.tp_size, "{:.0f}", "TP degree")
v("dp_a", A.dp_size, "{:.0f}", "A: DP degree")
v("dp_b", B.dp_size, "{:.0f}", "B: DP degree")
v("pp_c", C.pp_size, "{:.0f}", "C: PP degree")
v("dp_c", C.dp_size, "{:.0f}", "C: DP degree")
v("dp_c_others", C.dp_size - 1, "{:.0f}", "other replicas each shard syncs with")
v("m_c", C.microbatches, "{:.0f}", "C: microbatches")
v("shard_b_gb", m(rb.dp_shard, "GB"), "{:.2f}", "B: DP shard, GB")
v("shard_c_gb", m(rc.dp_shard, "GB"), "{:.1f}", "C: DP shard, GB")
v("ar_b_s", m(rb.allreduce_time, "s"), "{:.2f}", "B: AllReduce, s")
v("ar_c_s", m(rc.allreduce_time, "s"), "{:.2f}", "C: AllReduce, s")
v("eff_b_raw_pct", pct(rb.efficiency_no_overlap), "{:.1f}", "B: efficiency, no overlap, %")
v("eff_b_pct", pct(rb.efficiency), "{:.0f}", "B: efficiency, %")
v("eff_c_pct", pct(rc.efficiency), "{:.0f}", "C: efficiency, %")
v("bubble_c_pct", pct(rc.bubble), "{:.1f}", "C: bubble, %")
v("bubble_c_fill", C.pp_size - 1, "{:.0f}", "C: p - 1")
v("bubble_c_phase", C.microbatches + C.pp_size - 1, "{:.0f}", "C: m + p - 1")
v("state_b_gb", m(rb.state_per_device, "GB"), "{:.0f}", "B: state per GPU, GB")
v("state_c_gb", m(rc.state_per_device, "GB"), "{:.1f}", "C: state per GPU, GB")
v("zero1_c_gb", m(rc.zero1_state_per_device, "GB"), "{:.0f}", "C plus ZeRO-1: state per GPU, GB")
v("dp_byte_cut_x", ratio(ra.dp_shard, rb.dp_shard), "{:.0f}", "TP 8 cuts the DP bytes, x")
v("l16k_gpus", L16K.n_devices, "{:,.0f}", "16K layout, GPUs")
v("l16k_pp", L16K.pp_size, "{:.0f}", "16K layout, PP")
v("l16k_state_gb", m(r16.state_per_device, "GB"), "{:.1f}", "16K layout state per GPU, GB")
v("l16k_bubble_pct", pct(r16.bubble), "{:.1f}", "16K layout bubble, %")
v("ll3_gpus_8k", float(RUNS.Llama3_405B_Gpus8K), "{:,.0f}", "Llama 3 first stage, GPUs")
v("ll3_gpus_16k", float(RUNS.Llama3_405B_Gpus16K), "{:,.0f}", "Llama 3 later stages, GPUs")
v("ll3_tp", float(RUNS.Llama3_405B_TensorParallel), "{:.0f}", "TP")
v("ll3_pp", float(RUNS.Llama3_405B_PipelineParallel), "{:.0f}", "PP")
v("ll3_cp_long", float(RUNS.Llama3_405B_ContextParallelLong), "{:.0f}", "CP, long context")
v("ll3_dp_8k", float(RUNS.Llama3_405B_DataParallel8K), "{:.0f}", "DP on 8K GPUs")
v("ll3_dp_16k", float(RUNS.Llama3_405B_DataParallel16K), "{:.0f}", "DP on 16K GPUs")
v("ll3_dp_long", float(RUNS.Llama3_405B_DataParallelLong), "{:.0f}", "DP, long context")
v("ll3_seq", float(RUNS.Llama3_405B_SeqLen8K), "{:,.0f}", "sequence length")
v("ll3_seq_long", float(RUNS.Llama3_405B_SeqLenLong), "{:,.0f}", "long-context sequence length")
v("ll3_tokens_m", float(RUNS.Llama3_405B_BatchTokens) / 10 ** 6, "{:.0f}", "tokens per batch, millions")
v("ll3_tflops_8k", float(RUNS.Llama3_405B_TflopsPerGpu8K), "{:.0f}", "TFLOP/s per GPU, 8K")
v("ll3_tflops_16k", float(RUNS.Llama3_405B_TflopsPerGpu16K), "{:.0f}", "TFLOP/s per GPU, 16K")
v("ll3_tflops_long", float(RUNS.Llama3_405B_TflopsPerGpuLong), "{:.0f}", "TFLOP/s per GPU, long context")
v("ll3_mfu_8k", pct(float(RUNS.Llama3_405B_Mfu8K)), "{:.0f}", "BF16 MFU, 8K, %")
v("ll3_mfu_16k", pct(float(RUNS.Llama3_405B_Mfu16K)), "{:.0f}", "BF16 MFU, 16K, %")
v("ll3_mfu_long", pct(float(RUNS.Llama3_405B_MfuLong)), "{:.0f}", "BF16 MFU, long context, %")
v("recomp_base", ACT.forward_equivalents, "{:.0f}", "step in forward-pass equivalents")
v("recomp_full", ACT.recompute_forward_equivalents, "{:.0f}", "with full recomputation")
v("recomp_pct", pct(ACT.recompute_overhead), "{:.0f}", "extra arithmetic, %")
v("act_none_tb", m(ACT.activations("none"), "TB"), "{:.1f}", "175B activations, no recomputation, TB")
v("act_full_gb", m(ACT.activations("full"), "GB"), "{:.1f}", "with full recomputation, GB")
v("act_mb", ACT.micro_batch, "{:.0f}", "sequences per microbatch")
rl = RLHF.result
v("rlhf_policy_b", m(RLHF.policy.parameters, "Gparam"), "{:.0f}", "policy, billions")
v("rlhf_critic_b", m(RLHF.critic.parameters, "Gparam"), "{:.0f}", "reward and value, billions")
v("rlhf_policy_gb", m(rl.policy_state, "GB"), "{:,.0f}", "policy state, GB")
v("rlhf_ref_gb", m(rl.reference_weights, "GB"), "{:.0f}", "reference weights, GB")
v("rlhf_reward_gb", m(rl.reward_weights, "GB"), "{:.0f}", "reward weights, GB")
v("rlhf_value_gb", m(rl.value_state, "GB"), "{:.0f}", "value state, GB")
v("rlhf_ppo_gb", m(rl.ppo_total, "GB"), "{:,.0f}", "PPO total, GB")
v("rlhf_dpo_gb", m(rl.dpo_total, "GB"), "{:,.0f}", "DPO total, GB")
v("rlhf_dpo_cut_pct", pct(rl.dpo_reduction), "{:.1f}", "DPO reduction, %")
v("rlhf_kv_gb", m(rl.kv_cache, "GB"), "{:.1f}", "rollout KV cache, GB")
v("rlhf_rollouts", RLHF.rollout_sequences, "{:.0f}", "rollout sequences")
v("rlhf_rollout_tokens", RLHF.rollout_tokens, "{:,.0f}", "tokens per rollout")
v("rlhf_tp", RLHF.tp_size, "{:.0f}", "TP")
v("rlhf_pp", RLHF.pp_size, "{:.0f}", "PP")
v("rlhf_dp", RLHF.dp_size, "{:.0f}", "DP")
v("rlhf_gpus", RLHF.tp_size * RLHF.pp_size * RLHF.dp_size, "{:.0f}", "policy pool, H100s")
v("rlhf_policy_per_gpu_gb", m(rl.policy_per_device, "GB"), "{:.0f}", "policy state per GPU, GB")
v("rlhf_train_infer_word", ratio(rl.policy_state, rl.reference_weights), "words", "policy over reference state, x")

print("== Wrap-up")
v("fleet_mfu_pct", pct(rb.fleet_mfu), "{:.0f}", "local 50% x B's efficiency, %")
v("double_mfu_pct", pct(rb.double_counted_mfu), "{:.0f}", "the same with efficiency applied twice, %")
v("gpu_mttf_h", m(FAIL.component_mtbf, "hour"), "{:,.0f}", "per-GPU MTTF, hours")
v("fail_gpus", FAIL.n_components, "{:,.0f}", "GPUs")
v("fail_interval_h", m(FAIL.result.interval, "hour"), "{:.1f}", "a GPU fails every, hours")
v("fail_interval_days_word", round(m(FAIL.result.interval, "day")), "words", "about this many days, in prose")

if __name__ == "__main__":
    write(__file__)
