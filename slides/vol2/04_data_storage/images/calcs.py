#!/usr/bin/env python3
"""Every number on the Lecture 4 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature) and the calculations
(Scenarios.*, each backed by a solver in engine/solvers/storage.py); the
chapter reads the same scenarios. This file only selects values, converts
units, and formats them; it types no numbers. Each value is recorded with v()
under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Hardware, Models, Systems = mlsysim.Hardware, mlsysim.Models, mlsysim.Systems
Infrastructure, Literature, Scenarios = mlsysim.Infrastructure, mlsysim.Literature, mlsysim.Scenarios
units = mlsysim.core.units
CORPUS = mlsysim.ReferenceStats.StorageTrainingCorpus
TECH = Hardware.Tech.Storage
H100 = Hardware.Cloud.H100
STORE = Systems.Storage
RUN_CLUSTER, REF_CLUSTER = Systems.Clusters.Production_2K, Systems.Clusters.Training_1K
NODE = Systems.Nodes.DGX_H100
PRICE = Infrastructure.Pricing

FOOT = Scenarios.Gpt3StorageFootprint
STAGED = Scenarios.Gpt3TieredCheckpoint2K
STORM = Scenarios.Llama2_70BCheckpointStorm1K
TEXT, IMAGE, VIDEO = Scenarios.TextInputDemand2K, Scenarios.ImageInputDemand2K, Scenarios.VideoInputDemand2K
STALL, WALL = Scenarios.OverlappedInputStall, Scenarios.QuarterDeliveryIoWall
PREFETCH = Scenarios.PrefetchDepthByTier
HOT, MULTI = Scenarios.HotTierCacheExample, Scenarios.MultimodalCorpusNodeCache
GDS = Scenarios.GpuDirectStorageNodeCpu
TAIL, MELTDOWN, PFS = Scenarios.StorageTailFanout100, Scenarios.ImageFileMetadataMeltdown, Scenarios.ParallelFileSystem1TBps
BILL, PLAN, IDLE = Scenarios.ObjectStorage100TB, Scenarios.VisionCorpusStreamVsStage, Scenarios.InputStallIdleCost1K
SYN, BENCH = Scenarios.SyntheticDataProvenance, Scenarios.MLPerfStorageV1Demand
LOAD, ROLLOUT = Scenarios.Gpt3WeightsColdStart, Scenarios.Gpt3RolloutFanout100
CODECS = Scenarios.NodeStripeCodecs
FS3 = Literature.StorageSystems


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def ratio(a, b):
    """Dimensionless ratio of two quantities."""
    return (a / b).to("").magnitude


def bpp(q):
    """Bytes per parameter (or per item) as a plain number."""
    return q.to("byte / param").magnitude if q.dimensionality != units.byte.dimensionality else q.to("byte").magnitude


print("== Running case")
f = FOOT.result
v("params_b", m(Models.Language.GPT3.parameters, "Gparam"), "{:.0f}", "running model, billions of parameters")
v("run_gpus", RUN_CLUSTER.total_accelerators, "{:,.0f}", "running cluster GPUs")
v("run_nodes", RUN_CLUSTER.count, "{:.0f}", "running cluster nodes")
v("gpus_per_node_word", NODE.accelerators_per_node, "words", "GPUs per node, in prose")
v("run_days", m(FOOT.run_duration, "day"), "{:.0f}", "run length, days")
v("tokens_t", m(CORPUS.TrainingTokens, "count") / 10 ** 12, "{:.1f}", "training tokens, trillions")
v("token_bytes", m(CORPUS.TokenIdBytes, "byte"), "{:.0f}", "bytes per token ID")
v("save_min", m(FOOT.save_interval, "minute"), "{:.0f}", "minutes between saves")
v("save_s", m(FOOT.save_interval, "second"), "{:.0f}", "seconds between saves")
v("saves_per_hour_word", ratio(units.Q_(1, "hour"), FOOT.save_interval), "words", "saves per hour, in prose")
v("nvme_drives_word", STORE.LocalNvmeGen4_7T68x4.devices_per_node, "words", "local NVMe drives per node, in prose")
v("nvme_drives", STORE.LocalNvmeGen4_7T68x4.devices_per_node, "{:.0f}", "local NVMe drives per node")
v("corpus_tb", m(FOOT.corpus_bytes, "TB"), "{:.0f}", "tokenized corpus, TB")
v("ckpt_gb", m(f.checkpoint_bytes, "GB"), "{:,.0f}", "checkpoint, GB")
v("ckpt_tb", m(f.checkpoint_bytes, "TB"), "{:.2f}", "checkpoint, TB")
v("saves", f.saves, "{:,.0f}", "saves in the run")
v("written_pb", m(f.checkpoint_bytes_written, "PB"), "{:.1f}", "checkpoint bytes written, PB")
v("written_tb", m(f.checkpoint_bytes_written, "TB"), "{:,.0f}", "checkpoint bytes written, TB")
v("write_read", f.write_read_ratio, "{:,.0f}", "written / read")
v("ckpt_bpp", bpp(FOOT.bytes_per_param), "{:.0f}", "checkpoint bytes per parameter")
v("bpp_fp16", bpp(units.BYTES_FP16), "{:.0f}", "FP16 bytes")
v("bpp_fp32", bpp(units.BYTES_FP32), "{:.0f}", "FP32 bytes")
v("bpp_adam", bpp(units.BYTES_ADAM_STATE), "{:.0f}", "two FP32 Adam moments, bytes")
state_bpp = mlsysim.ReferenceStats.MixedPrecisionAdam.TrainingStateBytesPerParameter
v("bpp_train", bpp(state_bpp), "{:.0f}", "training state with gradients, bytes per parameter")
v("bpp_grad", bpp(state_bpp - FOOT.bytes_per_param), "{:.0f}", "gradient bytes per parameter")
weights = LOAD.payload
v("weights_gb", m(weights, "GB"), "{:.0f}", "FP16 weights, GB")
v("master_gb", m(Models.Language.GPT3.parameters * units.BYTES_FP32 / units.param, "GB"), "{:.0f}", "FP32 master, GB")
v("adam_gb", m(Models.Language.GPT3.parameters * units.BYTES_ADAM_STATE / units.param, "GB"), "{:,.0f}", "Adam moments, GB")
v("avg_read_mbs", m(f.average_read_rate, "MB/s"), "{:.1f}", "corpus over the run, MB/s")
v("shards", f.shard_count, "{:,.0f}", "corpus shards")
v("shard_gb", m(FOOT.shard_bytes, "GB"), "{:.0f}", "shard size, GB")
v("archive_tb", m(f.archive_bytes, "TB"), "{:.1f}", "archived lineage, TB")
v("source_tb", m(FOOT.source_bytes, "TB"), "{:.0f}", "compressed source, TB")
v("archive_rate", m(PRICE.Storage.GlacierStandardPerTbMonth.rate, "USD/GB/month"), "{:.3f}", "archive, $/GB-month")

print("== Tiers")
v("hbm_tbs", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM, TB/s")
v("hbm_gbs", m(H100.memory.bandwidth, "GB/s"), "{:,.0f}", "H100 HBM, GB/s")
v("hbm_cap_gb", H100.memory.capacity.magnitude, "{:.0f}", "H100 HBM, labeled GB")
v("dram_gbs", m(TECH.HostDram.bandwidth, "GB/s"), "{:.0f}", "host DRAM, GB/s")
v("gen4_gbs", m(TECH.NvmeGen4.bandwidth, "GB/s"), "{:.0f}", "Gen4 NVMe drive, GB/s")
v("gen5_gbs", m(TECH.NvmeGen5.bandwidth, "GB/s"), "{:.0f}", "Gen5 NVMe drive, GB/s")
v("nvme_lat_us", m(TECH.NvmeGen4.latency, "microsecond"), "{:.0f}", "NVMe latency, us")
raid = STORE.LocalNvmeGen4_7T68x4.aggregate_bandwidth
v("raid_gbs", m(raid, "GB/s"), "{:.0f}", "four-drive stripe, GB/s")
v("pfs_tbs", m(STORE.PfsOneTbPerSecond.bandwidth, "TB/s"), "{:.0f}", "parallel file system, TB/s")
v("obj_agg_gbs", m(ROLLOUT.links["object store"], "GB/s"), "{:.0f}", "object store aggregate, GB/s")
share = STAGED.result.durable_share_per_node
v("pfs_share_gbs", m(share, "GB/s"), "{:.1f}", "file-system share per node, GB/s")
lad = [ratio(H100.memory.bandwidth, TECH.HostDram.bandwidth), ratio(TECH.HostDram.bandwidth, raid), ratio(raid, share)]
v("r_hbm_dram", lad[0], "{:.1f}", "HBM / DRAM")
v("r_dram_nvme", lad[1], "{:.1f}", "DRAM / stripe")
v("r_nvme_pfs", lad[2], "{:.1f}", "stripe / file-system share")
v("r_min", min(lad), "{:.0f}", "smallest step")
v("r_max", max(lad), "{:.0f}", "largest step")

print("== Cache")
h = HOT.result
v("hot_hit_pct", 100 * h.hit_rate, "{:.0f}", "illustrative hit rate, %")
v("hot_miss_pct", 100 * (1 - h.hit_rate), "{:.0f}", "illustrative miss rate, %")
v("hot_hit", h.hit_rate, "{:.2f}", "illustrative hit rate")
v("hot_miss", 1 - h.hit_rate, "{:.2f}", "illustrative miss rate")
v("hot_penalty", h.miss_penalty, "{:.0f}", "miss penalty, x")
v("hot_mean", h.mean_service, "{:.2f}", "E[T], hot service times")
v("hot_frac_pct", 100 * h.delivered_fraction, "{:.0f}", "delivered share of the hot rate, %")
v("hot_lost_pct", 100 * (1 - h.delivered_fraction), "{:.0f}", "rate lost to misses, %")
v("hot_miss_over_hit_word", round((1 - h.hit_rate) * h.miss_penalty / h.hit_rate), "words",
  "miss time over hit time, in prose")
mm = MULTI.result
v("mm_corpus_tb", m(MULTI.corpus_bytes, "TB"), "{:.0f}", "multimodal corpus, TB")
v("mm_local_tb", m(MULTI.hot_capacity, "TB"), "{:.1f}", "local NVMe capacity, TB")
v("mm_hit_pct", 100 * mm.hit_rate, "{:.0f}", "hit rate, %")
v("mm_penalty", mm.miss_penalty, "{:.0f}", "miss penalty, x")
v("mm_cold_gbs", m(MULTI.miss_bandwidth, "GB/s"), "{:.0f}", "file-system share, GB/s")
v("mm_rate_gbs", m(mm.delivered_rate, "GB/s"), "{:.0f}", "delivered, GB/s")
v("mm_frac_pct", 100 * mm.delivered_fraction, "{:.0f}", "delivered share of local, %")

print("== I/O wall")
v("wall_plateau", WALL.plateau, "{:.0f}", "throughput plateau, x")

print("== Input demand")
tr, ir = TEXT.result, IMAGE.result
v("step_ms", m(TEXT.step_time, "ms"), "{:.0f}", "compute per step, ms")
v("step_s", m(TEXT.step_time, "s"), "{:.1f}", "compute per step, s")
v("seq_tokens", TEXT.samples_per_step, "{:,.0f}", "tokens per GPU per step")
v("img_per_gpu", IMAGE.samples_per_step, "{:.0f}", "images per GPU per step")
v("img_kb", m(IMAGE.bytes_per_sample, "kB"), "{:.0f}", "mean ImageNet JPEG, KB")
v("img_kb1", m(IMAGE.bytes_per_sample, "kB"), "{:.1f}", "mean ImageNet JPEG, KB, one decimal")
v("text_mbs", m(tr.bandwidth, "MB/s"), "{:.0f}", "text demand, MB/s")
v("text_mbs1", m(tr.bandwidth, "MB/s"), "{:.1f}", "text demand, MB/s, one decimal")
v("img_gbs", m(ir.bandwidth, "GB/s"), "{:.0f}", "image demand, GB/s")
v("img_text", round(ratio(ir.bandwidth, tr.bandwidth), -2), "{:,.0f}", "image / text, to the hundred")
v("video_mb", m(VIDEO.bytes_per_sample, "MB"), "{:.0f}", "video clip, MB")
v("video_img", ratio(VIDEO.bytes_per_sample, IMAGE.bytes_per_sample), "{:.1f}", "clip / JPEG")

print("== MLPerf Storage v1.0")
b = BENCH.result
v("mlperf_results", b.result_count, "{:.0f}", "passing closed results")
for w in BENCH.workloads:
    for a in BENCH.accelerators:
        v(f"mlp_{w}_{a.lower()}", b.medians[w][a], "{:.2f}", f"median {w} {a}, GB/s per accelerator")
v("mlp_gen_min", min(b.generation_ratios.values()), "{:.1f}", "H100 / A100, smallest")
v("mlp_gen_max", max(b.generation_ratios.values()), "{:.0f}", "H100 / A100, largest")
row = Literature.Series.MLPerfStorageV1Training.find(
    "system", "AI400X2T_24x14TiB_nvme_8xNDR200_16clients/cosmoflow-h100")
v("ex_accels", row["num_accelerators"], "{:.0f}", "example: emulated H100s")
v("ex_hosts", row["num_hosts"], "{:.0f}", "example: hosts")
v("ex_gbs", row["io_GBps"], "{:.2f}", "example: GB/s")
v("ex_busy", row["au_mean_pct"], "{:.2f}", "example: busy, %")

print("== Stall and prefetch")
s = STALL.result
v("io_ms", m(STALL.io_time, "ms"), "{:.0f}", "delivery per batch, ms")
v("seq_ms", m(s.sequential_step, "ms"), "{:.0f}", "sequential step, ms")
v("seq_stall_pct", 100 * s.sequential_stall, "{:.1f}", "sequential stall, %")
v("ovl_ms", m(s.overlapped_step, "ms"), "{:.0f}", "overlapped step, ms")
v("ovl_wait_ms", m(s.overlapped_step - STALL.compute_time, "ms"), "{:.0f}", "overlapped wait, ms")
v("ovl_stall_pct", 100 * s.overlapped_stall, "{:.0f}", "overlapped stall, %")
p = PREFETCH.result
v("p99_obj_ms", m(PREFETCH.tail_latencies["object"], "ms"), "{:.0f}", "object p99, ms")
v("p99_nvme_ms", m(PREFETCH.tail_latencies["nvme"], "ms"), "{:.0f}", "local NVMe p99, ms")
v("span_obj", p.spans["object"], "{:.1f}", "object p99 / compute")
v("depth_obj", p.depths["object"], "{:.0f}", "ready batches for object p99")
v("depth_safety", PREFETCH.safety_depth, "{:.0f}", "safety depth")
v("depth_safety_word", PREFETCH.safety_depth, "words", "safety depth, in prose")
v("depth_nvme_word", p.depths["nvme"], "words", "ready batches for local p99, in prose")
v("batch_mb", m(PREFETCH.batch_bytes, "MB"), "{:.0f}", "decoded batch, MB")
v("prefetch_node_gb", m(p.buffer_per_node, "GB"), "{:.1f}", "host DRAM at the safety depth, GB")

print("== GPUDirect Storage")
g = GDS.result
v("gds_rate", g.request_rate, "{:,.0f}", "images per second per node")
v("gds_host_us", m(GDS.service_times["host"], "microsecond"), "{:.0f}", "host path CPU, us")
v("gds_gds_us", m(GDS.service_times["gds"], "microsecond"), "{:.0f}", "GDS path CPU, us")
v("gds_host_cpu", g.cpu_seconds_per_second["host"], "{:.2f}", "host path, CPU-s per s")
v("gds_gds_cpu", g.cpu_seconds_per_second["gds"], "{:.2f}", "GDS path, CPU-s per s")
v("gds_host_cores", g.cpu_seconds_per_second["host"], "{:.0f}", "host path, cores")
v("gds_gds_cores", g.cpu_seconds_per_second["gds"], "{:.0f}", "GDS path, cores")
v("gds_freed", round(g.cpu_seconds_per_second["host"]) - round(g.cpu_seconds_per_second["gds"]), "{:.0f}",
  "cores freed")

print("== Shards and metadata")
v("files_m", MELTDOWN.files / 10 ** 6, "{:.0f}", "image files, millions")
v("tar_shards", MELTDOWN.shards, "{:,.0f}", "tar shards")
v("files_per_shard", MELTDOWN.files_per_shard, "{:,.0f}", "files per shard")
v("fsr", m(MELTDOWN.rated_bandwidth, "GB/s"), "{:.0f}", "rated file system, GB/s")
v("fs1", m(MELTDOWN.rated_bandwidth, "GB/s") / 100, "{:.0f}", "1 percent of the rating, GB/s")
v("oss", PFS.servers, "{:.0f}", "object storage servers")
v("oss_gbs", m(PFS.server_bandwidth, "GB/s"), "{:.0f}", "one OSS, GB/s")
v("tail_servers", TAIL.n_components, "{:.0f}", "servers per step")
v("tail_slow_pct", 100 * (1 - TAIL.availability), "{:.0f}", "slow share per server, %")
v("tail_fast", TAIL.availability, "{:.2f}", "fast share per server")
v("tail_all_fast", TAIL.all_up, "{:.3f}", "all fast")
v("tail_some_slow_pct", 100 * (1 - TAIL.all_up), "{:.0f}", "steps meeting a slow server, %")

print("== 3FS (DeepSeek README)")
v("fs3_tibs", float(FS3.Fs3ReadThroughputTiBps), "{:.1f}", "aggregate read, TiB/s")
v("fs3_nodes", float(FS3.Fs3StorageNodes), "{:.0f}", "storage nodes")
v("fs3_ssds", float(FS3.Fs3SsdsPerNode), "{:.0f}", "SSDs per node")
v("fs3_ssds_word", float(FS3.Fs3SsdsPerNode), "words", "SSDs per node, in prose")
v("fs3_ssd_tib", float(FS3.Fs3SsdCapacityTiB), "{:.0f}", "SSD capacity, TiB")
v("fs3_nics", float(FS3.Fs3NicsPerNode), "{:.0f}", "NICs per node")
v("fs3_nics_word", float(FS3.Fs3NicsPerNode), "words", "NICs per node, in prose")
v("fs3_nic_gbps", float(FS3.Fs3NicGbps), "{:.0f}", "NIC rate, Gb/s")
v("fs3_clients", float(FS3.Fs3ClientNodes), "{:.0f}", "client nodes (lower bound)")
v("fs3_per_node_gib", m(units.Q_(float(FS3.Fs3ReadThroughputTiBps), "TiB/s") / float(FS3.Fs3StorageNodes), "GiB/s"), "{:.1f}",
  "per storage node, GiB/s")
v("tib_tb", ratio(units.Q_(1, "TiB"), units.Q_(1, "TB")), "{:.1f}", "TiB / TB")

print("== Checkpoints")
st = STAGED.result
v("shard_node_gb", m(st.per_node_shard, "GB"), "{:.1f}", "per-node shard, GB")
v("pause_s", m(st.local_write_time, "s"), "{:.2f}", "local write (pause), s")
v("drain_s", m(st.drain_time, "s"), "{:.2f}", "drain to the file system, s")
v("both_s", m(st.local_write_time + st.drain_time, "s"), "{:.1f}", "pause plus drain, s")
v("pause_pct", 100 * st.pause_fraction, "{:.2f}", "pause share of the interval, %")
sm = STORM.result
v("storm_params_b", m(STORM.workload.parameters, "Gparam"), "{:.0f}", "storm model, billions")
v("storm_state_gb", m(sm.state_bytes, "GB"), "{:.0f}", "resumable state, GB")
v("storm_replicas", STORM.replicas, "{:,.0f}", "replicas")
v("storm_ingest_gbs", m(STORM.ingest_bandwidth, "GB/s"), "{:.0f}", "storage ingest, GB/s")
v("storm_tb", m(sm.storm_bytes, "TB"), "{:,.0f}", "every replica's copy, TB")
v("storm_s", m(sm.storm_time, "s"), "{:,.0f}", "every replica, s")
v("storm_min", m(sm.storm_time, "minute"), "{:.0f}", "every replica, min")
v("once_s", m(sm.single_copy_time, "s"), "{:.1f}", "one copy, s")
v("storm_ratio", sm.slowdown, "{:,.0f}", "storm / one copy")
storm_step_s, storm_every = 10, 100  # the note's question: 10-second steps, a save every 100 steps
v("storm_train_s", storm_step_s * storm_every, "{:,.0f}", "training between saves in the question, s")

print("== Versions")
y = SYN.result
v("syn_tb", m(SYN.payload_bytes, "TB"), "{:.0f}", "generated payload, TB")
v("syn_rest_tb", m(y.stored_bytes, "TB"), "{:.1f}", "at rest, TB")
v("syn_moved_tb", m(y.pipeline_bytes, "TB"), "{:.1f}", "moved, TB")
v("syn_passes_word", SYN.verification_passes, "words", "verifier passes, in prose")

print("== Cost")
bl = BILL.result
v("ds_tb", m(BILL.dataset_bytes, "TB"), "{:.0f}", "dataset, TB")
v("obj_rate", m(BILL.storage_rate, "USD/GB/month"), "{:.2f}", "object storage, $/GB-month")
v("egress_rate", m(BILL.egress_rate, "USD/GB"), "{:.2f}", "egress, $/GB")
v("nvme_rate", m(PLAN.local_rate, "USD/GB/month"), "{:.2f}", "local NVMe, $/GB-month")
v("store_year", bl.storage_cost, "{:,.0f}", "a year of object storage, $")
v("egress_one", bl.egress_per_read, "{:,.0f}", "one read across regions, $")
v("egress_reads", BILL.reads, "{:.0f}", "reads")
v("egress_all", bl.egress_cost, "{:,.0f}", "all reads, $")
v("files_get_m", BILL.files / 10 ** 6, "{:.0f}", "files read, millions")
v("get_fees", bl.request_cost, "{:,.0f}", "GET fees, $")
v("restore", bl.archive_restore_cost, "{:,.0f}", "archive restore with egress, $")
u = IDLE.result
v("idle_gpus", IDLE.fleet.total_accelerators, "{:,.0f}", "reference cluster GPUs")
v("gpu_hour", m(IDLE.device_hour_rate, "USD/hour"), "{:.0f}", "GPU-hour, $")
v("fleet_day", u.fleet_cost_per_day, "{:,.0f}", "fleet per day, $")
v("util_high", 100 * IDLE.healthy_utilization, "{:.0f}", "healthy busy share, %")
v("util_low", 100 * IDLE.degraded_utilization, "{:.0f}", "stalled busy share, %")
v("util_drop", 100 * (IDLE.healthy_utilization - IDLE.degraded_utilization), "{:.0f}", "busy-share drop, points")
v("idle_day", u.loss_per_day, "{:,.0f}", "idle loss per day, $")
pl = PLAN.result
v("ss_tb", m(PLAN.dataset_bytes, "TB"), "{:.0f}", "vision corpus, TB")
v("ss_epochs", PLAN.epochs, "{:.0f}", "epochs per run")
v("ss_runs", PLAN.runs_per_year, "{:.0f}", "runs a year")
v("ss_runs_word", PLAN.runs_per_year, "words", "runs a year, in prose")
v("ss_source", pl.source_storage_cost, "{:,.0f}", "object storage a year, $")
v("ss_per_run", pl.stream_egress_per_run, "{:,.0f}", "streaming egress per run, $")
v("ss_stream_egress", pl.stream_egress_cost, "{:,.0f}", "streaming egress a year, $")
v("ss_stream", pl.stream_cost, "{:,.0f}", "stream total, $")
v("ss_local", pl.local_storage_cost, "{:,.0f}", "local NVMe a year, $")
v("ss_stage_egress", pl.stage_egress_cost, "{:,.0f}", "staging egress a year, $")
v("ss_stage", pl.stage_cost, "{:,.0f}", "stage total, $")
v("ss_saving", pl.savings, "{:,.0f}", "staging saves, $")
v("ss_per_epoch", PLAN.runs_per_year * m(PLAN.dataset_bytes * PLAN.egress_rate, "USD"), "{:,.0f}",
  "streaming egress per epoch-per-run, $")
v("ss_break_even", pl.break_even_epochs, "{:.1f}", "break-even, epochs per run")

print("== Serving")
t = LOAD.result.times
v("load_obj_gbs", m(LOAD.links["object store"], "GB/s"), "{:.0f}", "single object-store client, GB/s")
v("load_pfs_gbs", m(LOAD.links["file system share"], "GB/s"), "{:.0f}", "file-system share, GB/s")
v("load_obj_s", m(t["object store"], "s"), "{:.0f}", "from object store, s")
v("load_pfs_s", m(t["file system share"], "s"), "{:.1f}", "from the file-system share, s")
v("load_gen5_s", m(t["one Gen5 NVMe"], "s"), "{:.0f}", "from one Gen5 drive, s")
v("load_stripe_s", m(t["eight Gen4 NVMe"], "s"), "{:.1f}", "from eight Gen4 drives, s")
v("load_dram_s", m(t["host DRAM"], "s"), "{:.1f}", "from host DRAM, s")
v("load_drives_word", NODE.accelerators_per_node, "words", "drives striped, in prose")
v("rollout_replicas", ROLLOUT.copies, "{:.0f}", "rollout replicas")
v("rollout_tb", m(ROLLOUT.total_bytes, "TB"), "{:.0f}", "rollout bytes, TB")
v("rollout_pfs_s", m(ROLLOUT.result.times["file system"], "s"), "{:.0f}", "rollout from the file system, s")

print("== Fallacies")
v("codec_dense", CODECS.compression_ratios["dense"], "{:.0f}", "denser codec ratio")
v("codec_light", CODECS.compression_ratios["light"], "{:.0f}", "light codec ratio")
v("codec_dense_gbs", m(CODECS.decode_rates["dense"], "GB/s"), "{:.1f}", "denser decode per core, GB/s")
v("codec_light_gbs", m(CODECS.decode_rates["light"], "GB/s"), "{:.1f}", "light decode per core, GB/s")

if __name__ == "__main__":
    write(__file__)
