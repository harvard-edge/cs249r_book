"""Storage solvers: what a training fleet reads, saves, and pays to move.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
Each model is a bandwidth or rate bound: it counts bytes and divides by a
stated rate, and leaves queueing, contention, and protocol overheads out.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from ..results import (
    CheckpointStormResult,
    DataStallResult,
    HostIoCpuResult,
    InputDemandResult,
    ObjectStorageBillResult,
    PrefetchDepthResult,
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
        """Account for checkpoint writes, corpus reads, and retained archive bytes.

        Parameters
        ----------
        parameters : Quantity
            Nonnegative parameter count, in param.
        bytes_per_param : Quantity
            Checkpoint bytes per parameter, including the desired optimizer state.
        corpus_bytes : Quantity
            Positive tokenized corpus size, in bytes; read once over the run.
        source_bytes : Quantity
            Nonnegative original source size, in bytes.
        run_duration : Quantity
            Positive elapsed run duration, convertible to seconds.
        save_interval : Quantity
            Positive time between saves.
        shard_bytes : Quantity
            Positive nominal corpus shard size, in bytes.
        archive_every : int
            Positive archive stride in checkpoints.

        Returns
        -------
        StorageFootprintResult
            Checkpoint and archive bytes, save and shard counts, write/read ratio,
            and average corpus read rate in MB/s.

        See Also
        --------
        mlsysim.physics.quantities.memory_from_params :
            Convert a parameter count and byte width to a storage quantity.

        Notes
        -----
        Equation: Checkpoint C = parameters * bytes_per_param; saves N = floor(D/I);
        written bytes = ``N*C``; retained checkpoints = ceil(N/archive_every).
        Archive bytes add retained checkpoints, source, and corpus.

        Assumptions: The corpus is read once; every save has identical size. The shard
        count uses round(corpus_bytes/shard_bytes), not a capacity ceiling.

        Excludes: Replication, compression, partial final checkpoints, rereads, and
        storage-service overhead.

        References
        ----------
        Derivation: Multiply fixed checkpoint size by completed save intervals. Archive
        retention counts every requested stride and one final retained save;
        source and tokenized corpus occupy additional storage.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import StorageFootprintModel
        >>> r = StorageFootprintModel().solve(Q_(10, "param"), Q_(2, "byte/param"),
        ...     Q_(100, "byte"), Q_(20, "byte"), Q_(4, "s"), Q_(1, "s"),
        ...     Q_(10, "byte"), 2)
        >>> assert r.checkpoint_bytes_written == r.checkpoint_bytes * r.saves
        """
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
        """Separate local checkpoint pause from background durable drain time.

        Parameters
        ----------
        checkpoint_bytes : Quantity
            Positive full checkpoint size, in bytes.
        nodes : int
            Positive number of equally loaded writer nodes.
        local_bandwidth : Quantity
            Positive local write bandwidth per node, in bytes/s.
        durable_bandwidth : Quantity
            Positive aggregate durable-tier bandwidth, in bytes/s.
        save_interval : Quantity
            Positive time between checkpoints.

        Returns
        -------
        TieredCheckpointResult
            Shard size, local write and durable drain times in seconds,
            per-node durable bandwidth, and local pause fraction.

        Notes
        -----
        Equation: Per-node shard = C/N; pause = ``C/(N*local_bandwidth)``;
        drain = C/durable_bandwidth; pause fraction = pause/save_interval.

        Assumptions: All nodes write equal shards concurrently. Background drain does
        not reduce the stated local bandwidth.

        Excludes: Backpressure when drain outlasts the save interval, replication,
        queueing, and durable-tier failure recovery.

        References
        ----------
        Derivation: Equal sharding divides local write volume by node count. The durable
        tier still receives the entire checkpoint, so its drain uses aggregate
        bandwidth without another division by nodes.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import TieredCheckpointModel
        >>> r = TieredCheckpointModel().solve(Q_(100, "GB"), 2,
        ...     Q_(10, "GB/s"), Q_(5, "GB/s"), Q_(1, "minute"))
        >>> assert r.per_node_shard * 2 == r.checkpoint_bytes
        """
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
        """Compare replicated checkpoint traffic with one logical state copy.

        Parameters
        ----------
        state_bytes : Quantity
            Positive size of one resumable state, in bytes.
        replicas : int
            Positive number of identical copies written concurrently.
        ingest_bandwidth : Quantity
            Positive shared aggregate ingest bandwidth, in bytes/s.

        Returns
        -------
        CheckpointStormResult
            Logical and replicated bytes, transfer times in seconds,
            and the dimensionless replicated/single-copy slowdown.

        Notes
        -----
        Equation: Storm bytes = replicas * state_bytes; each transfer time is
        its bytes divided by the same ingest bandwidth.

        Assumptions: Every replica writes a complete identical state to one shared
        bandwidth-limited sink.

        Excludes: Deduplication, per-client bandwidth limits, queueing, and metadata
        operations.

        References
        ----------
        Derivation: Identical complete state copies add their bytes at the shared sink.
        Dividing both byte totals by the same bandwidth makes the time ratio
        equal the copy count.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import CheckpointStormModel
        >>> r = CheckpointStormModel().solve(Q_(10, "GB"), 2, Q_(1, "GB/s"))
        >>> assert r.storm_time == r.single_copy_time * 2
        """
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
        """Compute aggregate input bandwidth demanded by active consumers.

        Parameters
        ----------
        consumers : int
            Nonnegative independent consumer count.
        samples_per_step : int
            Nonnegative samples in each consumer batch.
        bytes_per_sample : Quantity
            Nonnegative stored bytes per sample.
        step_time : Quantity
            Positive compute time per batch.
        busy_fraction : float, optional
            Dimensionless active fraction in [0, 1]; default 1 uses continuous compute.

        Returns
        -------
        InputDemandResult
            Per-consumer batch bytes and aggregate average bandwidth
            with byte/time dimensions.

        Notes
        -----
        Equation: Batch bytes = samples_per_step * bytes_per_sample;
        bandwidth = consumers * batch_bytes * busy_fraction / step_time.

        Assumptions: Consumers read distinct batches at the same fixed rate; the busy
        fraction scales average demand rather than peak bursts.

        Excludes: Caching, repeated samples, preprocessing expansion, and correlated
        arrival bursts.

        References
        ----------
        Derivation: Samples per batch times bytes per sample gives one consumer's
        transfer volume. Dividing by compute time gives its active rate;
        consumer count and busy fraction scale aggregate average demand.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import InputDemandModel
        >>> r = InputDemandModel().solve(2, 4, Q_(1, "MB"), Q_(1, "s"))
        >>> assert r.bandwidth * Q_(1, "s") == r.batch_bytes * 2
        """
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
        """Compare sequential delivery with ideally overlapped input and compute.

        Parameters
        ----------
        compute_time : Quantity
            Positive time to compute one batch.
        io_time : Quantity
            Nonnegative time to deliver one batch; compatible with compute_time.

        Returns
        -------
        DataStallResult
            Sequential and overlapped step durations, with dimensionless
            fractions of each step spent waiting for input.

        Notes
        -----
        Equation: Sequential step = compute + I/O; overlapped step = max(compute, I/O).
        Stall fraction = (step - compute)/step in each schedule.

        Assumptions: The overlapped pipeline is in steady state and independent stages
        can run simultaneously.

        Excludes: Pipeline startup, scheduling overhead, shared-resource contention,
        and variation across batches.

        References
        ----------
        Derivation: Sequential stages add durations. Ideal overlap completes when the
        slower stage finishes. Subtracting useful compute from either step
        gives its input wait, which is normalized by that step duration.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import DataStallModel
        >>> r = DataStallModel().solve(Q_(2, "s"), Q_(1, "s"))
        >>> assert r.overlapped_step <= r.sequential_step
        """
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
        """Size ready-batch buffers from specified tail-latency scenarios.

        Parameters
        ----------
        compute_time : Quantity
            Positive time to consume one batch.
        tail_latencies : dict of str to Quantity
            Named nonnegative read-latency scenarios, with time units.
        batch_bytes : Quantity
            Nonnegative bytes retained per ready batch.
        safety_depth : int
            Nonnegative chosen allocation depth; independent of computed depths.
        consumers_per_node : int
            Nonnegative consumers sharing one host.

        Returns
        -------
        PrefetchDepthResult
            Latency spans and ceiling depths per scenario; allocated
            buffer bytes per consumer and node.

        Notes
        -----
        Equation: Required depth for scenario k = ceil(latency[k]/compute_time).
        Allocated buffer = safety_depth * batch_bytes per consumer.

        Assumptions: Ready batches cover deterministic compute windows; safety_depth
        is supplied, not automatically set to the largest required depth.

        Excludes: Read throughput limits, batch-size variation, and correlated
        latency tails across consumers.

        References
        ----------
        Derivation: A read spanning multiple compute windows needs one ready batch per
        window, rounded upward. Buffer capacity instead uses the separately
        chosen safety depth and sums identical consumer allocations.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import PrefetchDepthModel
        >>> r = PrefetchDepthModel().solve(Q_(1, "s"), {"tail": Q_(2, "s")},
        ...     Q_(10, "MB"), 3, 2)
        >>> assert r.buffer_per_node == r.buffer_per_consumer * 2
        """
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
        """Average hit and miss service time for a two-tier read mixture.

        Parameters
        ----------
        hit_rate : float
            Dimensionless byte-weighted hit fraction in [0, 1].
        miss_penalty : float, optional
            Positive miss/hit service-time ratio, required unless both bandwidths are supplied.
        hot_bandwidth : Quantity, optional
            Positive hot-tier bandwidth in bytes/s; use together with miss_bandwidth.
        miss_bandwidth : Quantity, optional
            Positive miss-tier bandwidth in bytes/s; overrides miss_penalty when both are supplied.

        Returns
        -------
        TieredCacheResult
            Mean service relative to a hit, delivered fraction of hot-tier
            rate, and delivered GB/s when bandwidths are supplied.

        Notes
        -----
        Equation: Mean normalized service = h + ``(1-h)*miss_penalty``; delivered
        fraction = 1/mean. With bandwidths, average seconds/byte then invert.

        Assumptions: The same byte volume determines hit and miss shares. Each miss
        uses miss-tier service alone; the hot and miss costs are alternatives.

        Excludes: Queueing, additional lookup costs, concurrent independent tier
        service, and eviction policy.

        References
        ----------
        Derivation: Average time per byte over mutually exclusive hit and miss outcomes,
        then invert to obtain bytes per time. Averaging bandwidths directly
        would not conserve the service time of the byte mixture.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import TieredCacheModel
        >>> r = TieredCacheModel().solve(1.0, miss_penalty=4.0)
        >>> assert r.delivered_fraction == 1.0
        """
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
        """Convert request rates and host service times into occupied CPU cores.

        Parameters
        ----------
        request_rate : float
            Nonnegative requests per second, supplied as a scalar.
        service_times : dict of str to Quantity
            Named nonnegative CPU service times per request, convertible to seconds.

        Returns
        -------
        HostIoCpuResult
            Request rate and CPU-seconds per second for each service-time
            scenario; the latter equals fully occupied core equivalents.

        Notes
        -----
        Equation: CPU-seconds per second = request_rate * service_time_in_seconds.

        Assumptions: Each request consumes the stated CPU service independently; time
        is CPU time, not wall-clock I/O latency.

        Excludes: CPU saturation, scheduling overhead, contention, and parallel
        execution within a request.

        References
        ----------
        Derivation: Requests per second times CPU-seconds per request cancels the
        request unit and yields CPU-seconds per second. One such unit is one
        continuously occupied core.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import HostIoCpuModel
        >>> r = HostIoCpuModel().solve(100, {"path": Q_(1, "ms")})
        >>> assert r.cpu_seconds_per_second["path"] > 0
        """
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
        """Price capacity, transfer, requests, and an archive restore separately.

        Parameters
        ----------
        dataset_bytes : Quantity
            Nonnegative stored dataset size, in bytes.
        storage_rate : Quantity
            Nonnegative capacity price in currency/(byte*time).
        billing_period : Quantity
            Nonnegative capacity billing duration.
        egress_rate : Quantity
            Nonnegative currency per byte transferred out.
        reads : int
            Nonnegative complete dataset reads charged egress.
        files : int
            Nonnegative total GET requests to bill; not multiplied by reads.
        request_fee : Quantity
            Nonnegative currency fee for requests_per_fee requests.
        requests_per_fee : int
            Positive request-count billing unit.
        retrieval_rate : Quantity
            Nonnegative archive retrieval charge in currency/byte.

        Returns
        -------
        ObjectStorageBillResult
            Separate costs as USD scalars: storage, egress, requests,
            archive retrieval, and archive restore including one egress read.

        Notes
        -----
        Equation: Storage = bytes * rate * period; egress = bytes * reads * egress_rate;
        requests = files * fee/requests_per_fee; restore = retrieval + one egress.

        Assumptions: Prices are linear and supplied by the caller. The files input
        already counts the requests that should be billed.

        Excludes: Provider discounts, minimum storage periods, taxes, tiered pricing,
        and request retries.

        References
        ----------
        Derivation: Capacity charges multiply stored bytes by residence time and unit
        price. Transfer and retrieval charges multiply moved bytes by their
        prices; request charges prorate the quoted fee by its request unit.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import ObjectStorageBillModel
        >>> r = ObjectStorageBillModel().solve(Q_(1, "GB"), Q_(1, "USD/GB/day"),
        ...     Q_(1, "day"), Q_(1, "USD/GB"), 2, 10, Q_(1, "USD"),
        ...     10, Q_(1, "USD/GB"))
        >>> assert r.egress_cost == r.egress_per_read * 2
        """
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
        """Compare repeated remote reads with one local staging transfer per run.

        Parameters
        ----------
        dataset_bytes : Quantity
            Positive dataset size, in bytes.
        epochs : int
            Nonnegative complete dataset passes per run.
        runs_per_year : int
            Positive runs in the accounting period.
        object_rate : Quantity
            Nonnegative source storage price in currency/(byte*time).
        local_rate : Quantity
            Nonnegative local storage price in currency/(byte*time).
        egress_rate : Quantity
            Positive remote transfer price in currency/byte.
        year : Quantity
            Positive accounting period duration; normally one year.

        Returns
        -------
        StreamVsStageResult
            USD cost components, stream-minus-stage savings, and the
            dimensionless epoch count at equal modeled cost.

        Notes
        -----
        Equation: Stream = source_storage + runs * epochs * bytes * egress_rate;
        stage = source_storage + local_storage + runs * bytes * egress_rate.

        Assumptions: Both choices keep remote source storage. Local capacity is billed
        for the full period; each staged run transfers the dataset once.

        Excludes: Local I/O costs, staging delay, cache persistence across runs,
        and bandwidth-dependent performance differences.

        References
        ----------
        Derivation: Remote source capacity is common to both options and cancels at
        break-even. Staging adds local capacity and one transfer per run;
        streaming instead pays for a transfer on every epoch.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import StreamVsStageModel
        >>> r = StreamVsStageModel().solve(Q_(1, "GB"), 2, 1,
        ...     Q_(1, "USD/GB/day"), Q_(1, "USD/GB/day"), Q_(1, "USD/GB"),
        ...     Q_(1, "day"))
        >>> assert r.savings == r.stream_cost - r.stage_cost
        """
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
        """Price the billed accelerator time lost to a utilization decrease.

        Parameters
        ----------
        n_devices : int
            Nonnegative billed accelerator count.
        device_hour_rate : Quantity
            Nonnegative rental price per device, in currency/hour.
        healthy_utilization : float
            Dimensionless reference busy fraction in [0, 1].
        degraded_utilization : float
            Dimensionless comparison busy fraction in [0, 1]; higher values yield negative loss.
        duration : Quantity
            Nonnegative accounting duration, convertible to days.

        Returns
        -------
        UtilizationLossResult
            Full fleet cost per day, lost cost per day, and total lost
            cost, all as USD scalars.

        Notes
        -----
        Equation: Daily fleet cost = devices * hourly_rate * hours_per_day;
        loss = daily_cost * (healthy - degraded) * duration_in_days.

        Assumptions: The fleet incurs its full rental bill whether busy or idle.
        Utilization differences are constant over the accounting period.

        Excludes: Actual billing reductions, energy savings, and causal attribution
        of the utilization change.

        References
        ----------
        Derivation: Multiply the full device rental rate by elapsed time to obtain billed
        cost. Multiplying by the lost busy fraction prices the corresponding
        unused capacity, without asserting that the invoice itself decreases.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import UtilizationLossModel
        >>> r = UtilizationLossModel().solve(2, Q_(1, "USD/hour"), 0.8, 0.8, Q_(1, "day"))
        >>> assert r.loss_total == 0.0
        """
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
        """Count generated payload, provenance storage, and verification reads.

        Parameters
        ----------
        payload_bytes : Quantity
            Positive total generated payload size, in bytes.
        provenance_overhead : float
            Nonnegative metadata bytes as a dimensionless fraction of payload.
        verification_passes : int
            Nonnegative full-payload verification read count.

        Returns
        -------
        SyntheticFootprintResult
            Stored bytes, verification-read bytes, total pipeline bytes,
            and dimensionless capacity and traffic amplification.

        Notes
        -----
        Equation: Stored = ``payload*(1+overhead)``; verification = ``payload*passes``;
        pipeline = stored + verification.

        Assumptions: Payload and metadata are written once; each verification pass
        reads payload only.

        Excludes: Intermediate generations, failed attempts, metadata rereads,
        compression, and replication.

        References
        ----------
        Derivation: Stored volume is payload plus metadata. The pipeline writes that
        volume once and additionally reads the payload on each verification
        pass; those disjoint transfers add to total byte traffic.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import SyntheticFootprintModel
        >>> r = SyntheticFootprintModel().solve(Q_(1, "GB"), 0.1, 2)
        >>> assert r.pipeline_bytes == r.stored_bytes + r.verification_bytes
        """
        stored = payload_bytes * (1 + provenance_overhead)
        verify = payload_bytes * verification_passes
        traffic = payload_bytes + verify + payload_bytes * provenance_overhead
        return SyntheticFootprintResult(
            stored_bytes=stored, verification_bytes=verify, pipeline_bytes=traffic,
            capacity_amplification=_ratio(stored, payload_bytes),
            traffic_amplification=_ratio(traffic, payload_bytes),
        )


