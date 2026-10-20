#!/usr/bin/env python3
"""Every number on the Lecture 5 slides and notes, read from MLSysIM.

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

Hardware, Systems, Models, Scenarios, Literature = (mlsysim.Hardware, mlsysim.Systems, mlsysim.Models,
                                                    mlsysim.Scenarios, mlsysim.Literature)
LIT = Literature.Communication
H100, NDR = Hardware.Cloud.H100, Systems.Fabrics.InfiniBand_NDR
NSTAR = Scenarios.CriticalSizeByInterconnect           # n* = alpha beta per interconnect
FUSE = Scenarios.RoutingMessageFusionNdr               # 100 x 4 KB, separately and fused
BUCKET = Scenarios.DdpBucketRingNdr                    # 25 MB bucket on 64 and 1,024 ranks
BUF = Scenarios.GradientBufferAllReduceNdr             # 1 GB buffer at 8, 64, 1,024 ranks
EST = Scenarios.Llama2_70BGradientRing64Ndr            # 140 GB gradient on 64 GPUs
RVT = Scenarios.RingVsTreeSmallBuffer                  # 1 MB, 64 ranks, 10 us, 10 GB/s
HIER = Scenarios.HierarchicalAllReduce8x8DgxH100       # 1 GB on 8 nodes of 8
HREF = Scenarios.HierarchicalAllReduceReferenceCluster  # 175B gradient on 1,024 GPUs
MOE = Scenarios.MoeDispatch8Gpu64Experts
TRACE = Scenarios.RingTraceFourGpu
EF, EFC = Scenarios.ErrorFeedbackFiveSteps, Scenarios.ErrorFeedbackConstantGradient
PAY = Scenarios.CompressionPayback8x
RUN = Scenarios.Int8CompressionRun
ENC = Scenarios.GradientEncodingRatios
FSDP = Scenarios.ShardedDataParallelCalls7B
OV, OVI = Scenarios.OverlapBudget7B64Gpu, Scenarios.OverlapBudget7B64GpuIdealBuckets
REF = Scenarios.DenseLLMPretrainingReference


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


print("== Hook: Horovod at Uber")
hvd = {r["model"]: r for r in Literature.Series.HorovodScaling128Gpu.entries}
v("hvd_gpus", LIT.HorovodBenchmarkGpus, "{:.0f}", "Pascal GPUs in the benchmark")
v("hvd_gbe", LIT.HorovodEthernetGbps, "{:.0f}", "Ethernet, Gb/s")
v("hvd_eff_pct", LIT.HorovodScalingEfficiencyPct, "{:.0f}", "Horovod efficiency, %")
v("hvd_base_eff_pct", 100 - LIT.HorovodTensorflowResourceLossPct, "{:.0f}", "TensorFlow efficiency, about, %")
for key, model in (("incv3", "Inception V3"), ("resnet", "ResNet-101")):
    r = hvd[model]
    v(f"hvd_ideal_{key}", r["ideal_images_per_s"], "{:,.0f}", f"{model} ideal, images/s")
    v(f"hvd_tf_{key}", r["tensorflow_images_per_s"], "{:.0f}", f"{model} TensorFlow bar, images/s")
    v(f"hvd_hv_{key}", r["horovod_images_per_s"], "{:.0f}", f"{model} Horovod bar, images/s")
    v(f"hvd_tf_{key}_pct", 100 * r["tensorflow_images_per_s"] / r["ideal_images_per_s"], "{:.0f}",
      f"{model} TensorFlow share of ideal, %")
    v(f"hvd_hv_{key}_pct", 100 * r["horovod_images_per_s"] / r["ideal_images_per_s"], "{:.0f}",
      f"{model} Horovod share of ideal, %")
v("hvd_fusion_pct", LIT.HorovodTensorFusionGainPct, "{:.0f}", "Tensor Fusion gain, up to, %")
v("hvd_fusion_mb", LIT.HorovodFusionBufferMB, "{:.0f}", "fusion buffer, MB")
v("hvd_rdma_lo_pct", LIT.HorovodRdmaGainLowPct, "{:.0f}", "RDMA gain, low, %")
v("hvd_rdma_hi_pct", LIT.HorovodRdmaGainHighPct, "{:.0f}", "RDMA gain, high, %")
v("hvd_rdma_vgg_pct", LIT.HorovodRdmaGainVgg16Pct, "{:.0f}", "RDMA gain, VGG-16, %")
v("hvd_rdma_ratio_word", LIT.HorovodRdmaGainVgg16Pct / LIT.HorovodRdmaGainLowPct, "words",
  "VGG-16 gain over the others, in prose")

print("== Horovod Timeline (redrawn trace)")
rows = Literature.Series.HorovodTimelineTrace.entries
ph = {(r["tensors"], r["phase"]): (r["start_ms"], r["end_ms"]) for r in rows}
t0 = LIT.HorovodTimelineOriginMs
ar_start = LIT.HorovodTimelineAllreduceStartMs - t0
ar_end = ar_start + LIT.HorovodTimelineAllreduceSpanMs
slow_span = ar_end - ph[("2-4", "negotiate")][0]
slow_nccl = ph[("2-4", "NCCL")][1] - ph[("2-4", "NCCL")][0]
v("hvd_t0_ms", t0, "{:,.0f}", "first tick of the screenshot, ms")
v("hvd_ar_start", ar_start, "{:.1f}", "ALLREDUCE start after the tick, ms")
v("hvd_ar_end", ar_end, "{:.1f}", "ALLREDUCE end after the tick, ms")
v("hvd_ar_ms", LIT.HorovodTimelineAllreduceSpanMs, "{:.0f}", "ALLREDUCE span, ms")
v("hvd_negotiate_ms", ph[("2-4", "negotiate")][1] - ph[("2-4", "negotiate")][0], "{:.0f}",
  "tensors 2-4 negotiate, ms")
v("hvd_span_ms", slow_span, "{:.0f}", "tensors 2-4 first request to finish, ms")
v("hvd_nccl_ms", slow_nccl, "{:.0f}", "tensors 2-4 NCCL, ms")
v("hvd_nccl_pct", 100 * slow_nccl / slow_span, "{:.0f}", "NCCL share of that span, %")
v("hvd_fast_ms", ph[("0-1", "NCCL")][1] - ph[("0-1", "negotiate")][0], "{:.0f}",
  "tensors 0-1 first request to finish, ms")

print("== Running case and link rates")
alpha_ndr = NSTAR.alphas["NDR"]
v("ib_gbs", m(NDR.bandwidth, "GB/s"), "{:.0f}", "NDR port, GB/s per direction")
v("nvlink_gbs", m(H100.nvlink.bandwidth_per_direction, "GB/s"), "{:.0f}", "NVLink per direction, GB/s")
v("nvlink_bidir_gbs", m(H100.nvlink.bandwidth, "GB/s"), "{:.0f}", "NVLink both directions, GB/s")
v("nvlink_ib_ratio", ratio(H100.nvlink.bandwidth_per_direction, NDR.bandwidth), "{:.0f}", "NVLink / NDR port")
v("nvlink_ib_word", ratio(H100.nvlink.bandwidth_per_direction, NDR.bandwidth), "words", "the same, in prose")
v("alpha_us", m(alpha_ndr, "microsecond"), "{:.1f}", "NDR alpha, us")
v("gpus_per_node", HIER.gpus_per_node, "{:.0f}", "GPUs per DGX H100 node")
v("gpus_per_node_word", HIER.gpus_per_node, "words", "the same, in prose")
v("buf_gb", m(BUF.message, "GB"), "{:.0f}", "reference buffer, GB")
v("grp_gpus", BUF.group_sizes[1], "{:.0f}", "scoreboard GPUs")
v("ref_n", BUF.group_sizes[-1], "{:,.0f}", "reference cluster GPUs")
v("run_steps", EST.run_steps, "{:,.0f}", "steps in the run")
grad70 = Models.Language.Llama2_70B
v("grad70_params_b", m(grad70.parameters, "Gparam"), "{:.0f}", "model, billions of parameters")
v("grad70_gb", m(EST.message, "GB"), "{:.0f}", "BF16 gradient, GB")
v("bf16_bytes", m(mlsysim.BYTES_FP16, "byte"), "{:.0f}", "bytes per BF16 value")

print("== FSDP calls")
v("fsdp_layers", FSDP.workload.layers, "{:.0f}", "layers")
v("fsdp_calls", FSDP.calls("FSDP, reshard after forward"), "{:.0f}", "calls per step, resharding")

print("== Critical size n* = alpha beta")
x = NSTAR.result.crossovers
v("nstar_nvlink_kb", round(m(x["NVLink"], "KB"), -2), "{:.0f}", "NVLink n*, to the hundred KB")
v("nstar_nvlink_mb", m(x["NVLink"], "MB"), "{:.1f}", "NVLink n*, MB")
v("nstar_pcie_kb", m(x["PCIe Gen5"], "KB"), "{:.0f}", "PCIe Gen5 n*, KB")
v("nstar_roce_kb", m(x["RoCE"], "KB"), "{:.1f}", "RoCE n*, KB")
v("nstar_hdr_kb", m(x["HDR"], "KB"), "{:.1f}", "HDR n*, KB")
v("nstar_ndr_kb", m(x["NDR"], "KB"), "{:.0f}", "NDR n*, KB")

print("== One 25 MB bucket on 64 and 1,024 ranks")
b = BUCKET.result
ns, nl = BUCKET.group_sizes
v("bucket_mb", m(BUCKET.message, "MB"), "{:.0f}", "bucket, MB")
v("bkt_n_small", ns, "{:.0f}", "small ring, ranks")
v("bkt_n_large", nl, "{:,.0f}", "large ring, ranks")
for key, n in (("small", ns), ("large", nl)):
    v(f"bkt_chunk_{key}_kb", m(b.chunk[n], "KB"), "{:.1f}", f"chunk at {n}, KB")
    v(f"bkt_rounds_{key}", b.ring_rounds[n], "{:,.0f}", f"rounds at {n}")
    v(f"bkt_bytes_{key}_ms", m(b.ring_byte_time[n], "ms"), "{:.1f}", f"byte term at {n}, ms")
    v(f"bkt_total_{key}_ms", m(b.ring_time[n], "ms"), "{:.1f}", f"ring time at {n}, ms")
v("bkt_chunk_large_kb0", m(b.chunk[nl], "KB"), "{:.0f}", "chunk at 1,024, whole KB")
v("bkt_round_small_us", m(b.ring_time[ns] / b.ring_rounds[ns], "microsecond"), "{:.1f}", "one round at 64, us")
v("bkt_ratio", ratio(b.ring_time[nl], b.ring_time[ns]), "{:.1f}", "1,024 / 64 ring time")
v("bkt_knee_ratio", ratio(BUCKET.message, x["NDR"]), "{:.0f}", "bucket / NDR n*")

print("== Fusion")
v("fuse_count", FUSE.count, "{:.0f}", "messages fused")
v("fuse_msg_kb", m(FUSE.message, "KB"), "{:.0f}", "message, KB")
v("fuse_total_kb", m(FUSE.fused_message, "KB"), "{:.0f}", "fused message, KB")
v("fuse_sep_us", m(FUSE.separate_time, "microsecond"), "{:.0f}", "separately, us")
v("fuse_fused_us", m(FUSE.fused_time, "microsecond"), "{:.1f}", "fused, us")
v("fuse_speedup", FUSE.speedup, "{:.0f}", "separate / fused")

print("== Scoreboard: 1 GB on 64 GPUs at 50 GB/s")
g = BUF.result
n64 = BUF.group_sizes[1]
v("ps_gb", m(n64 * BUF.message, "GB"), "{:.0f}", "server receives (and sends), GB")
v("ps_ms", m(g.star_byte_time[n64], "ms"), "{:,.0f}", "parameter server, ms each way")
v("ring_bytes_ms", m(g.ring_byte_time[n64], "ms"), "{:.1f}", "ring byte term, ms")
v("tree_bytes_ms", m(g.tree_byte_time[n64], "ms"), "{:.0f}", "simple tree byte term, ms")
v("ps_ring_ratio", ratio(g.star_byte_time[n64], g.ring_byte_time[n64]), "{:.1f}", "server / ring")
v("ps_half_n", g.star_ring_limit[n64], "{:.0f}", "N/2")
v("tree_transfers_grp", g.tree_rounds[n64], "{:.0f}", "full-buffer transfers in the tree")

print("== Ring trace (four GPUs)")
t = TRACE.result
for i, vec in enumerate(TRACE.vectors):
    for j, val in enumerate(vec):
        v(f"tv{i}{j}", val, "{:.0f}", f"GPU {i} element {j}")
for j, val in enumerate(t.totals):
    v(f"tsum{j}", val, "{:.0f}", f"sum, element {j}")
for s_, row in enumerate(t.scatter_reduce, 1):
    for gpu, (_, val) in enumerate(row):
        v(f"rs{s_}_g{gpu}", val, "{:.0f}", f"ReduceScatter {s_}, GPU {gpu}")
for s_, row in enumerate(t.all_gather, 1):
    for gpu, (_, val) in enumerate(row):
        v(f"ag{s_}_g{gpu}", val, "{:.0f}", f"AllGather {s_}, GPU {gpu}")
v("trace_steps", t.steps, "{:.0f}", "steps, 2(N-1)")
v("trace_steps_word", t.steps, "words", "the same, in prose")

print("== Estimate: 140 GB on 64 GPUs")
e = EST.result
ne = EST.group_sizes[0]
v("est_n", ne, "{:.0f}", "GPUs in the ring")
v("est_n_minus1", ne - 1, "{:.0f}", "N - 1")
v("est_factor", e.ring_factor[ne], "{:.5f}", "2(N-1)/N")
v("est_bytes_ms", m(e.ring_byte_time[ne], "ms"), "{:,.1f}", "byte term, ms")
v("est_rounds", e.ring_rounds[ne], "{:.0f}", "rounds")
v("est_startup_ms", m(e.ring_startup_time[ne], "ms"), "{:.1f}", "startup term, ms")
v("est_total_s", m(e.ring_time[ne], "second"), "{:.1f}", "total, s")
v("est_startup_pct", 100 * ratio(e.ring_startup_time[ne], e.ring_time[ne]), "{:.4f}", "startup share, %")
v("est_saved_s", m(EST.saved_time(ne), "second"), "{:.2f}", "halving saves per step, s")
v("est_saved_gpuh", round(EST.saved_device_hours(ne), -2), "{:,.0f}", "GPU-hours over the run, to the hundred")

print("== Rounds at 1,024")
nr = BUF.group_sizes[-1]
v("ring_rounds_large", g.ring_rounds[nr], "{:,.0f}", "ring rounds")
v("tree_rounds_large", g.tree_rounds[nr], "{:.0f}", "tree rounds")
v("round_ratio", round(g.ring_rounds[nr] / g.tree_rounds[nr], -1), "{:.0f}", "ring / tree, to the ten")
v("ref_cross_mb", m(g.crossover_estimate[nr], "MB"), "{:.1f}", "crossover on 1,024 NDR GPUs, MB")

print("== Ring or tree: 1 MB on 64 ranks")
r = RVT.result
nv = RVT.group_sizes[0]
v("rt_n", nv, "{:.0f}", "ranks")
v("rt_m_mb", m(RVT.message, "MB"), "{:.0f}", "buffer, MB")
v("rt_alpha_us", m(RVT.alpha, "microsecond"), "{:.0f}", "alpha, us")
v("rt_beta_gbs", m(RVT.bandwidth, "GB/s"), "{:.0f}", "beta, GB/s")
v("rt_ring_rounds", r.ring_rounds[nv], "{:.0f}", "ring rounds")
v("rt_tree_rounds", r.tree_rounds[nv], "{:.0f}", "tree rounds")
v("rt_ring_lat_us", m(r.ring_startup_time[nv], "microsecond"), "{:,.0f}", "ring startup, us")
v("rt_tree_lat_us", m(r.tree_startup_time[nv], "microsecond"), "{:,.0f}", "tree startup, us")
v("rt_ring_bw_us", m(r.ring_byte_time[nv], "microsecond"), "{:.1f}", "ring bytes, us")
v("rt_tree_bw_us", m(r.tree_byte_time[nv], "microsecond"), "{:,.0f}", "tree bytes, us")
v("rt_ring_total_us", m(r.ring_time[nv], "microsecond"), "{:,.1f}", "ring total, us")
v("rt_ring_total_us0", m(r.ring_time[nv], "microsecond"), "{:,.0f}", "ring total, whole us")
v("rt_tree_total_us", m(r.tree_time[nv], "microsecond"), "{:,.0f}", "tree total, us")
v("rt_margin_pct", 100 * (1 - ratio(r.tree_time[nv], r.ring_time[nv])), "{:.0f}", "tree faster by, %")
v("rt_cross_mb", m(r.crossover_estimate[nv], "MB"), "{:.1f}", "crossover estimate, MB")
v("rt_cross_exact_mb", m(r.crossover_exact[nv], "MB"), "{:.2f}", "exact crossover, MB (plot)")

print("== Hierarchical AllReduce")
h = HIER.result
v("hier_nodes", HIER.n_nodes, "{:.0f}", "nodes")
v("hier_nodes_word", HIER.n_nodes, "words", "the same, in prose")
v("hier_gpus", HIER.n_devices, "{:.0f}", "GPUs")
v("hier_intra_mb", m(h.intra_bytes, "MB"), "{:.0f}", "ReduceScatter bytes per GPU, MB")
v("hier_rs_ms", m(h.reduce_scatter_time, "ms"), "{:.2f}", "ReduceScatter (and AllGather), ms")
v("hier_shard_mb", m(h.shard, "MB"), "{:.0f}", "shard, MB")
v("hier_inter_ms", m(h.inter_time, "ms"), "{:.2f}", "cross-node AllReduce, ms")
v("hier_total_ms", m(h.total_time, "ms"), "{:.1f}", "total, ms")
v("hier_flat_ms", m(h.flat_bound_time, "ms"), "{:.0f}", "flat ring 2M/beta, ms")
v("hier_speedup", h.speedup_vs_flat_bound, "{:.1f}", "flat / hierarchical")
hr = HREF.result
v("ref_params_b", m(REF.workload.parameters, "Gparam"), "{:.0f}", "running model, billions of parameters")
v("ref_grad_gb", m(HREF.message, "GB"), "{:.0f}", "its BF16 gradient, GB")
v("ref_flat_s", m(hr.flat_ring_time, "second"), "{:.1f}", "flat ring on 1,024, s")
v("ref_hier_s", m(hr.total_time, "second"), "{:.1f}", "hierarchical on 1,024, s")

print("== MoE dispatch")
d = MOE.result
v("moe_gpus", MOE.n_devices, "{:.0f}", "GPUs")
v("moe_experts", MOE.n_experts, "{:.0f}", "experts")
v("moe_experts_per_gpu", d.experts_per_device, "{:.0f}", "experts per GPU")
v("moe_tokens_per_gpu", d.tokens_per_device, "{:.0f}", "tokens per GPU")
v("moe_pair_tokens", d.tokens_per_pair, "{:.0f}", "tokens to each peer")
v("moe_peers", d.remote_peers, "{:.0f}", "remote peers")
v("moe_peers_word", d.remote_peers, "words", "the same, in prose")
v("moe_hidden", MOE.hidden_dim, "{:,.0f}", "values per token")
v("moe_token_kb", m(d.bytes_per_token, "KB"), "{:.3f}", "bytes per token, KB")
v("moe_sent_mb", m(d.sent_bytes, "MB"), "{:.2f}", "sent per GPU, MB")
v("moe_rx_us", m(d.uniform_receive_bound, "microsecond"), "{:.1f}", "uniform receive bound, us")
v("moe_naive_gbs", d.remote_peers * m(MOE.bandwidth, "GB/s"), "{:.0f}", "seven ports' worth, GB/s")
v("moe_skew", MOE.receive_skew, "{:.1f}", "hot-receiver skew")
v("moe_uniform_tokens", d.uniform_remote_tokens, "{:.0f}", "uniform remote tokens")
v("moe_hot_tokens", d.hot_remote_tokens, "{:.0f}", "hot remote tokens")
v("moe_hot_mb", m(d.hot_receive_bytes, "MB"), "{:.2f}", "hot receive bytes, MB")
v("moe_hot_us", m(d.hot_receive_bound, "microsecond"), "{:.1f}", "hot receive bound, us")

print("== Compression payback")
v("cp_comm_ms", m(PAY.communication_time, "ms"), "{:.0f}", "uncompressed, ms")
v("cp_ratio", PAY.compression_ratio, "{:.0f}", "effective ratio")
v("cp_xfer_ms", m(PAY.compressed_time, "ms"), "{:.0f}", "compressed transfer, ms")
v("cp_codec_ms", m(PAY.codec_time, "ms"), "{:.0f}", "codec, ms")
v("cp_total_ms", m(PAY.total_time, "ms"), "{:.0f}", "total, ms")
v("cp_speedup", PAY.speedup, "{:.1f}", "speedup")
v("cp_fast_ms", m(PAY.fast_communication_time, "ms"), "{:.0f}", "same reduction on NVLink, ms")
v("enc_fp16", ENC.ratio("FP16", "FP32"), "{:.0f}", "FP16 vs FP32")
v("enc_int8", ENC.ratio("INT8", "FP32"), "{:.0f}", "INT8 vs FP32")
v("enc_topk", ENC.ratio("Top-k", "FP32"), "{:.0f}", "top-1% with indices vs FP32")
v("enc_topk_pct", 100 * ENC.kept_fractions["Top-k"], "{:.0f}", "top-k kept, %")
v("enc_1bit", ENC.ratio("1-bit", "FP32"), "{:.0f}", "1-bit vs FP32")

print("== Run level")
c = RUN.result
v("cr_steps", RUN.steps, "{:,.0f}", "steps")
v("cr_steps_c", c.steps_compressed, "{:,.0f}", "steps with compression")
v("cr_compute_ms", m(RUN.compute_time, "ms"), "{:.0f}", "compute per step, ms")
v("cr_ratio", RUN.compression_ratio, "{:.0f}", "INT8 vs FP32")
v("cr_penalty_pct", 100 * RUN.step_penalty, "{:.0f}", "extra steps, %")
for k in ("slow", "fast"):
    comm = RUN.communication_times[k]
    v(f"cr_{k}_comm_ms", m(comm, "ms"), "{:.0f}", f"{k}: communication, ms")
    v(f"cr_{k}_step_ms", m(c.step_times[k], "ms"), "{:.0f}", f"{k}: step, ms")
    v(f"cr_{k}_base_s", m(c.totals[k], "second"), "{:,.0f}", f"{k}: baseline, s")
    v(f"cr_{k}_comp_s", m(c.compressed_totals[k], "second"), "{:,.0f}", f"{k}: compressed, s")
v("cr_slow_comm_c_ms", m(RUN.communication_times["slow"] / RUN.compression_ratio, "ms"), "{:.0f}",
  "slow: compressed communication, ms")
v("cr_slow_step_c_ms", m(c.compressed_step_times["slow"], "ms"), "{:.0f}", "slow: compressed step, ms")
v("cr_fast_saved_ms", m(c.step_times["fast"] - c.compressed_step_times["fast"], "ms"), "{:.1f}",
  "fast: saved per step, ms")
v("cr_fast_step_c_ms", m(c.compressed_step_times["fast"], "ms"), "{:.1f}", "fast: compressed step, ms")
v("cr_slow_gain_pct", -100 * c.changes["slow"], "{:.0f}", "slow: faster by, %")
v("cr_fast_loss_pct", 100 * c.changes["fast"], "{:.1f}", "fast: slower by, %")

print("== Error feedback")
f = EF.result
for i, gv in enumerate(EF.gradients, 1):
    v(f"ef_g{i}", gv, "{:.1f}", f"gradient, step {i}")
    v(f"ef_s{i}", f.corrected[i - 1], "{:.1f}", f"g + e, step {i}")
    v(f"ef_v{i}", f.sent[i - 1], "{:.0f}", f"sent, step {i}")
    v(f"ef_n{i}", f.errors[i], "{:.1f}", f"new residual, step {i}")
    if i > 1:
        v(f"ef_e{i}", f.errors[i - 1], "{:.1f}", f"residual in, step {i}")
v("ef_e1", f.errors[0], "{:.0f}", "residual in, step 1 (zero)")
v("ef_naive", f.naive_total_sent, "{:.0f}", "sent without a residual")
v("ef_true", f.true_total, "{:.1f}", "true total")
v("ef_sent", f.total_sent, "{:.0f}", "sent with error feedback")
v("ef_final", f.final_error, "{:.1f}", "final residual")
v("efc_g", EFC.gradients[0], "{:.1f}", "constant gradient")
v("efc_steps_word", len(EFC.gradients), "words", "steps, in prose")
v("efc_sent_word", EFC.result.total_sent, "words", "units sent, in prose")

print("== 1-bit Adam")
v("oba_volume", LIT.OneBitAdamVolumeReduction, "{:.0f}", "communication volume, up to x")
v("oba_bert", LIT.OneBitAdamBertLargeSpeedup, "{:.1f}", "BERT-Large throughput, up to x")
v("oba_squad", LIT.OneBitAdamSquadSpeedup, "{:.1f}", "SQuAD throughput, up to x")
v("oba_gpus", LIT.OneBitAdamMaxGpus, "{:.0f}", "GPUs, up to")

print("== Overlap budget")
o, oi = OV.result, OVI.result
v("ov_params_b", m(OV.workload.parameters, "Gparam"), "{:.0f}", "model, billions of parameters")
v("ov_layers", o.layers, "{:.0f}", "layers")
v("ov_layers_minus1", o.layers - 1, "{:.0f}", "layers after the first")
v("ov_gpus", OV.n_devices, "{:.0f}", "GPUs")
v("ov_bucket_mb", m(OV.bucket_size, "MB"), "{:.0f}", "bucket, MB")
v("ov_bucket_ms", m(OV.time_per_bucket, "ms"), "{:.0f}", "assumed time per bucket, ms")
v("ov_bwd_ms", m(OV.backward_per_layer, "ms"), "{:.0f}", "backward per layer, ms")
v("ov_comm_ms", m(o.communication_per_layer, "ms"), "{:.0f}", "reduction per layer, ms")
v("ov_total_bwd_ms", m(o.total_backward, "ms"), "{:.0f}", "backward pass, ms")
v("ov_seq_ms", m(o.sequential_time, "ms"), "{:,.0f}", "sequential, ms")
v("ov_exposed_ms", m(o.exposed_per_layer, "ms"), "{:.0f}", "backlog per layer, ms")
v("ov_overlap_ms", m(o.overlapped_time, "ms"), "{:.0f}", "overlapped, ms")
v("ov_savings_pct", 100 * o.savings_fraction, "{:.1f}", "reduction, %")
v("ov_ideal_bucket_ms", m(OVI.time_per_bucket, "ms"), "{:.2f}", "ideal time per bucket, ms")
v("ov_ideal_layer_ms", m(oi.communication_per_layer, "ms"), "{:.1f}", "ideal reduction per layer, ms")
v("ov_fast_ms", m(oi.overlapped_time, "ms"), "{:.1f}", "overlapped at the ideal rate, ms")

if __name__ == "__main__":
    write(__file__)
