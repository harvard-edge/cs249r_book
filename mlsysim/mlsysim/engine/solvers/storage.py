"""Storage solvers: what a training fleet reads, saves, and pays to move.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
Each model is a bandwidth or rate bound: it counts bytes and divides by a
stated rate, and leaves queueing, contention, and protocol overheads out.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, Optional, Sequence

from ..results import (
    CheckpointStormResult,
    DataStallResult,
    HostIoCpuResult,
    InputDemandResult,
    ObjectStorageBillResult,
    PrefetchDepthResult,
    StorageBenchmarkResult,
    StorageFootprintResult,
    StreamVsStageResult,
    SyntheticFootprintResult,
    TieredCacheResult,
    TieredCheckpointResult,
    UtilizationLossResult,
)
from ...core.types import Quantity
from ...core.units import HOURS_PER_DAY, Q_
from ...physics.quantities import memory_from_params
from .base import ForwardModel


def _ratio(a, b) -> float:
    return (a / b).to("").magnitude


class StorageFootprintModel(ForwardModel):
    """
    The bytes a training run reads once and the bytes it saves again and again.

    A resumable checkpoint holds ``bytes_per_param`` for every parameter and is
    written once per save interval, so a run of duration ``D`` saves
    ``D / interval`` times. Against a corpus read once, the written bytes can
    exceed the read bytes by orders of magnitude. The corpus is packaged into
    shards of ``shard_bytes``; the archive keeps every ``archive_every``-th
    checkpoint (and the final one when the count is not a multiple) plus the
    source and the tokenized corpus.
    """
    requires = ("workload",)
    produces = StorageFootprintResult

    def solve(self, parameters: Quantity, bytes_per_param: Quantity, corpus_bytes: Quantity,
              source_bytes: Quantity, run_duration: Quantity, save_interval: Quantity,
              shard_bytes: Quantity, archive_every: int) -> StorageFootprintResult:
        ckpt = memory_from_params(parameters, bytes_per_param)
        saves = int(_ratio(run_duration, save_interval))
        written = ckpt * saves
        archived = saves // archive_every + int(saves % archive_every != 0)
        return StorageFootprintResult(
            checkpoint_bytes=ckpt,
            saves=saves,
            checkpoint_bytes_written=written,
            write_read_ratio=_ratio(written, corpus_bytes),
            average_read_rate=(corpus_bytes / run_duration).to("MB/second"),
            shard_count=round(_ratio(corpus_bytes, shard_bytes)),
            archived_checkpoints=archived,
            archive_bytes=archived * ckpt + source_bytes + corpus_bytes,
        )


class TieredCheckpointModel(ForwardModel):
    """
    A checkpoint staged on each node's local storage, then drained to a durable tier.

    Each node writes its shard (``checkpoint / nodes``) to local storage and
    training pauses only for that write. The durable copy proceeds in the
    background at the shared tier's aggregate bandwidth, which is the same
    time as each node's shard at its per-node share of that bandwidth.
    """
    requires = ("workload", "fleet")
    produces = TieredCheckpointResult

    def solve(self, checkpoint_bytes: Quantity, nodes: int, local_bandwidth: Quantity,
              durable_bandwidth: Quantity, save_interval: Quantity) -> TieredCheckpointResult:
        shard = checkpoint_bytes / nodes
        pause = shard / local_bandwidth
        return TieredCheckpointResult(
            checkpoint_bytes=checkpoint_bytes,
            per_node_shard=shard,
            local_write_time=pause.to("second"),
            drain_time=(checkpoint_bytes / durable_bandwidth).to("second"),
            durable_share_per_node=durable_bandwidth / nodes,
            pause_fraction=_ratio(pause, save_interval),
        )


class CheckpointStormModel(ForwardModel):
    """
    Synchronized saves when every replica writes its own copy of the state.

    ``replicas`` copies of the full resumable state cross one shared ingest
    bandwidth; writing one logical copy, split across the ranks that own it,
    moves the state once. The ratio is the replica count.
    """
    requires = ("workload",)
    produces = CheckpointStormResult

    def solve(self, state_bytes: Quantity, replicas: int, ingest_bandwidth: Quantity) -> CheckpointStormResult:
        storm = state_bytes * replicas
        storm_t = storm / ingest_bandwidth
        once_t = state_bytes / ingest_bandwidth
        return CheckpointStormResult(
            state_bytes=state_bytes, storm_bytes=storm,
            storm_time=storm_t.to("second"), single_copy_time=once_t.to("second"),
            slowdown=_ratio(storm_t, once_t),
        )


class InputDemandModel(ForwardModel):
    """
    The input bandwidth a fleet draws: consumers x busy fraction x batch bytes / compute time.

    Each consumer reads a distinct batch of ``samples_per_step`` samples of
    ``bytes_per_sample`` every ``step_time`` of compute; ``busy_fraction`` is
    the share of wall-clock time spent computing.
    """
    requires = ("workload",)
    produces = InputDemandResult

    def solve(self, consumers: int, samples_per_step: int, bytes_per_sample: Quantity,
              step_time: Quantity, busy_fraction: float = 1.0) -> InputDemandResult:
        batch = samples_per_step * bytes_per_sample
        bw = consumers * batch / step_time
        if busy_fraction != 1.0:
            bw = bw * busy_fraction
        return InputDemandResult(batch_bytes=batch, bandwidth=bw)


class DataStallModel(ForwardModel):
    """
    Input wait per step when delivery and compute run in sequence or overlap.

    In sequence a step costs delivery plus compute; overlapped, the loader
    fetches the next batch during compute, so a step costs the longer of the
    two. The stall is the share of the step not spent computing.
    """
    requires = ("workload",)
    produces = DataStallResult

    def solve(self, compute_time: Quantity, io_time: Quantity) -> DataStallResult:
        seq = io_time + compute_time
        ovl = max(compute_time, io_time)
        return DataStallResult(
            sequential_step=seq,
            sequential_stall=_ratio(io_time, seq),
            overlapped_step=ovl,
            overlapped_stall=_ratio(ovl - compute_time, ovl),
        )


class PrefetchDepthModel(ForwardModel):
    """
    Ready batches a prefetch queue needs to cover a tier's tail read latency.

    One read at a tier's tail latency spans ``latency / compute_time`` compute
    windows, so the queue needs the ceiling of that many ready batches. A
    deeper safety depth costs ``depth x batch`` of host memory per consumer.
    """
    requires = ("workload",)
    produces = PrefetchDepthResult

    def solve(self, compute_time: Quantity, tail_latencies: Dict[str, Quantity], batch_bytes: Quantity,
              safety_depth: int, consumers_per_node: int) -> PrefetchDepthResult:
        spans = {k: _ratio(t, compute_time) for k, t in tail_latencies.items()}
        per = safety_depth * batch_bytes
        return PrefetchDepthResult(
            spans=spans, depths={k: math.ceil(s) for k, s in spans.items()},
            buffer_per_consumer=per, buffer_per_node=per * consumers_per_node,
        )


class TieredCacheModel(ForwardModel):
    """
    Mean service of a two-tier cache: E[T] = h T_hit + (1 - h) T_miss.

    With service times in units of one hit, ``E[T] = h + (1 - h) x penalty``,
    and the delivered rate is ``1 / E[T]`` of the all-hit rate. Given tier
    bandwidths, the penalty is their ratio and the delivered rate is the hot
    bandwidth times that fraction (time per byte averaged over the mix).
    """
    requires = ()
    produces = TieredCacheResult

    def solve(self, hit_rate: float, miss_penalty: Optional[float] = None,
              hot_bandwidth: Optional[Quantity] = None, miss_bandwidth: Optional[Quantity] = None) -> TieredCacheResult:
        rate = None
        if hot_bandwidth is not None and miss_bandwidth is not None:
            t_hot = (1 / hot_bandwidth).to("second/GB")
            t_miss = (1 / miss_bandwidth).to("second/GB")
            miss_penalty = _ratio(t_miss, t_hot)
            rate = (1 / (hit_rate * t_hot + (1 - hit_rate) * t_miss)).to("GB/second")
        mean = hit_rate * 1 + (1 - hit_rate) * miss_penalty
        frac = 1 / mean if rate is None else _ratio(rate, hot_bandwidth)
        return TieredCacheResult(hit_rate=hit_rate, miss_penalty=miss_penalty, mean_service=mean,
                                 delivered_fraction=frac, delivered_rate=rate)


class HostIoCpuModel(ForwardModel):
    """
    Host CPU time spent issuing and completing I/O requests.

    Each request costs a stated CPU service time on the host; at a request
    rate ``r`` the host spends ``r x service`` CPU-seconds per second, the
    number of cores kept busy by data movement alone.
    """
    requires = ()
    produces = HostIoCpuResult

    def solve(self, request_rate: float, service_times: Dict[str, Quantity]) -> HostIoCpuResult:
        return HostIoCpuResult(
            request_rate=request_rate,
            cpu_seconds_per_second={k: request_rate * t.to("second").magnitude for k, t in service_times.items()},
        )


class ObjectStorageBillModel(ForwardModel):
    """
    The object-storage bill beyond capacity: egress per read, GET requests, archive retrieval.

    Capacity is billed per byte-month; every read that leaves the region or
    provider pays egress per byte; each file read pays a GET fee; an archive
    restore pays a retrieval fee and then egress for one read.
    """
    requires = ()
    produces = ObjectStorageBillResult

    def solve(self, dataset_bytes: Quantity, storage_rate: Quantity, billing_period: Quantity,
              egress_rate: Quantity, reads: int, files: int, request_fee: Quantity, requests_per_fee: int,
              retrieval_rate: Quantity) -> ObjectStorageBillResult:
        egress_one = (dataset_bytes * egress_rate).to("USD").magnitude
        retrieval = (dataset_bytes * retrieval_rate).to("USD").magnitude
        return ObjectStorageBillResult(
            storage_cost=(dataset_bytes * storage_rate * billing_period).to("USD").magnitude,
            egress_per_read=egress_one,
            egress_cost=(dataset_bytes * reads * egress_rate).to("USD").magnitude,
            request_cost=files * (request_fee.to("USD").magnitude / requests_per_fee),
            archive_retrieval_cost=retrieval,
            archive_restore_cost=retrieval + egress_one,
        )


class StreamVsStageModel(ForwardModel):
    """
    Streaming every epoch from remote object storage against staging once per run.

    Both options keep the dataset in object storage as the source of truth.
    Streaming pays egress for every epoch of every run; staging adds a year of
    local capacity and pays egress once per run. They break even where the
    streaming egress per run equals staging's extra cost per run.
    """
    requires = ()
    produces = StreamVsStageResult

    def solve(self, dataset_bytes: Quantity, epochs: int, runs_per_year: int, object_rate: Quantity,
              local_rate: Quantity, egress_rate: Quantity, year: Quantity) -> StreamVsStageResult:
        source = (dataset_bytes * object_rate * year).to("USD").magnitude
        per_run = (dataset_bytes * epochs * egress_rate).to("USD").magnitude
        stream_egress = runs_per_year * per_run
        local = (dataset_bytes * local_rate * year).to("USD").magnitude
        stage_egress = (dataset_bytes * runs_per_year * egress_rate).to("USD").magnitude
        stream = source + stream_egress
        stage = source + local + stage_egress
        epoch_egress = runs_per_year * (dataset_bytes * egress_rate).to("USD").magnitude
        return StreamVsStageResult(
            source_storage_cost=source, stream_egress_per_run=per_run, stream_egress_cost=stream_egress,
            stream_cost=stream, local_storage_cost=local, stage_egress_cost=stage_egress, stage_cost=stage,
            savings=stream - stage, break_even_epochs=(local + stage_egress) / epoch_egress,
        )


class UtilizationLossModel(ForwardModel):
    """
    Dollars of accelerator time idled when utilization falls from one level to another.

    A fleet of ``n_devices`` billed at ``device_hour_rate`` costs its full rate
    every day whether it computes or waits; a drop in the busy fraction wastes
    that share of the daily cost.
    """
    requires = ("fleet",)
    produces = UtilizationLossResult

    def solve(self, n_devices: int, device_hour_rate: Quantity, healthy_utilization: float,
              degraded_utilization: float, duration: Quantity) -> UtilizationLossResult:
        daily = (n_devices * device_hour_rate * Q_(HOURS_PER_DAY, "hour")).to("USD").magnitude
        loss = daily * (healthy_utilization - degraded_utilization)
        return UtilizationLossResult(fleet_cost_per_day=daily, loss_per_day=loss,
                                     loss_total=loss * duration.to("day").magnitude)


class SyntheticFootprintModel(ForwardModel):
    """
    Generated training data at rest and in motion.

    Each sample carries provenance metadata (a fraction of its payload) at
    rest; the pipeline writes payload and metadata once and reads the payload
    once per verification pass.
    """
    requires = ()
    produces = SyntheticFootprintResult

    def solve(self, payload_bytes: Quantity, provenance_overhead: float,
              verification_passes: int) -> SyntheticFootprintResult:
        stored = payload_bytes * (1 + provenance_overhead)
        verify = payload_bytes * verification_passes
        traffic = payload_bytes + verify + payload_bytes * provenance_overhead
        return SyntheticFootprintResult(
            stored_bytes=stored, verification_bytes=verify, pipeline_bytes=traffic,
            capacity_amplification=_ratio(stored, payload_bytes),
            traffic_amplification=_ratio(traffic, payload_bytes),
        )


class StorageBenchmarkDemandModel(ForwardModel):
    """
    Measured storage demand per accelerator, from benchmark submissions.

    Keeps the submissions that pass a division and check, then takes, per
    workload and emulated accelerator, the median of sustained read throughput
    divided by accelerator count. The generation ratio divides the newer
    accelerator's median by the older one's.
    """
    requires = ("series",)
    produces = StorageBenchmarkResult

    def solve(self, rows: Iterable[dict], division: str, workloads: Sequence[str],
              accelerators: Sequence[str]) -> StorageBenchmarkResult:
        kept = [r for r in rows if r["division"] == division and r["au_check"] == "pass"]
        med = {w: {a: statistics.median(r["io_GBps"] / r["num_accelerators"] for r in kept
                                        if r["workload"] == w and r["accelerator"] == a)
                   for a in accelerators} for w in workloads}
        old, new = accelerators[0], accelerators[-1]
        return StorageBenchmarkResult(result_count=len(kept), medians=med,
                                      generation_ratios={w: med[w][new] / med[w][old] for w in workloads})
