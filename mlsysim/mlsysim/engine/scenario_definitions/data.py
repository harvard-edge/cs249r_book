"""Data preparation, selection, storage, and delivery.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from ...core.types import Quantity
from ...core.units import Q_
from ...hardware.types import HardwareNode
from ...models.types import Workload
from ...systems.types import Fleet
from .base import BaseScenario


class StorageFootprintScenario(BaseScenario):
    """What a training run of ``workload`` reads once, saves every interval, and archives.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.StorageFootprintModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.StorageFootprintModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.StorageFootprintModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.StorageFootprintModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3StorageFootprint
    >>> type(case.result).__name__
    'StorageFootprintResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    bytes_per_param: Quantity  # resumable checkpoint contract
    corpus_bytes: Quantity  # bytes read per epoch
    source_bytes: Quantity  # the compressed source kept for lineage
    run_duration: Quantity
    save_interval: Quantity
    shard_bytes: Quantity
    archive_every: int = 100  # keep every Nth checkpoint (and the final one)

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.StorageFootprintModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import StorageFootprintModel

        return StorageFootprintModel().solve(
            self.workload.parameters,
            self.bytes_per_param,
            self.corpus_bytes,
            self.source_bytes,
            self.run_duration,
            self.save_interval,
            self.shard_bytes,
            self.archive_every,
        )


class TieredCheckpointScenario(BaseScenario):
    """One save of ``workload``'s state on ``fleet``: local write per node, then a durable
    drain.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.quantities.memory_from_params``,
    ``mlsysim.solvers.TieredCheckpointModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.TieredCheckpointModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.TieredCheckpointModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.quantities.memory_from_params``,
    ``mlsysim.solvers.TieredCheckpointModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3TieredCheckpoint2K
    >>> type(case.result).__name__
    'TieredCheckpointResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    bytes_per_param: Quantity
    fleet: Fleet
    local_bandwidth: Quantity  # per node
    durable_bandwidth: Quantity  # aggregate, shared by every node
    save_interval: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.TieredCheckpointModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ...physics.quantities import memory_from_params
        from ..solvers import TieredCheckpointModel

        return TieredCheckpointModel().solve(
            memory_from_params(self.workload.parameters, self.bytes_per_param),
            self.fleet.count,
            self.local_bandwidth,
            self.durable_bandwidth,
            self.save_interval,
        )


class CheckpointStormScenario(BaseScenario):
    """Every one of ``replicas`` writes ``workload``'s full resumable state through one ingest
    path.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.quantities.memory_from_params``,
    ``mlsysim.solvers.CheckpointStormModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.CheckpointStormModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.CheckpointStormModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.quantities.memory_from_params``,
    ``mlsysim.solvers.CheckpointStormModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama2_70BCheckpointStorm1K
    >>> type(case.result).__name__
    'CheckpointStormResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    bytes_per_param: Quantity
    replicas: int
    ingest_bandwidth: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.CheckpointStormModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...physics.quantities import memory_from_params
        from ..solvers import CheckpointStormModel

        return CheckpointStormModel().solve(
            memory_from_params(self.workload.parameters, self.bytes_per_param), self.replicas, self.ingest_bandwidth
        )


class InputDemandScenario(BaseScenario):
    """``consumers`` each read ``samples_per_step`` samples per ``step_time`` of compute.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.InputDemandModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.InputDemandModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.InputDemandModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.InputDemandModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TextInputDemand2K
    >>> type(case.result).__name__
    'InputDemandResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    consumers: int
    samples_per_step: int
    bytes_per_sample: Quantity
    step_time: Quantity
    busy_fraction: float = 1.0

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.InputDemandModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import InputDemandModel

        return InputDemandModel().solve(
            self.consumers, self.samples_per_step, self.bytes_per_sample, self.step_time, self.busy_fraction
        )


class DataStallScenario(BaseScenario):
    """A batch delivered in ``io_time`` against ``compute_time`` of compute, in sequence or
    overlapped.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.DataStallModel``.

    Assumptions: Compute and I/O durations describe one batch; throughput comparisons use
    ideal overlap and fixed I/O time.

    Excludes: Queue transients, buffering limits, and partial overlap are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.DataStallModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.OverlappedInputStall
    >>> type(case.result).__name__
    'DataStallResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    compute_time: Quantity
    io_time: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.DataStallModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import DataStallModel

        return DataStallModel().solve(self.compute_time, self.io_time)

    def relative_throughput(self, compute_speedup: float) -> float:
        """Throughput change after speeding only compute in a perfectly overlapped pipeline.
        Equation: old ``max(compute_time, io_time)`` divided by the new maximum. Units:
        dimensionless; batch work and I/O time are fixed.

        Overlapped batches per second when compute runs ``compute_speedup`` times faster,
        delivery fixed.
        """
        return (
            (max(self.compute_time, self.io_time) / max(self.compute_time / compute_speedup, self.io_time))
            .to("")
            .magnitude
        )

    @property
    def plateau(self) -> float:
        """Maximum compute-only speedup once I/O becomes the bottleneck. Equation:
        ``max(compute_time, io_time) / io_time``. Units: dimensionless; requires positive I/O
        time and ideal overlap.
        """
        return (max(self.compute_time, self.io_time) / self.io_time).to("").magnitude


class PrefetchDepthScenario(BaseScenario):
    """Ready batches that cover each tier's tail read, and host memory at a safety depth.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.PrefetchDepthModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.PrefetchDepthModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.PrefetchDepthModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.PrefetchDepthModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PrefetchDepthByTier
    >>> type(case.result).__name__
    'PrefetchDepthResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    compute_time: Quantity
    tail_latencies: Dict[str, Quantity]
    batch_bytes: Quantity
    safety_depth: int
    consumers_per_node: int

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.PrefetchDepthModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import PrefetchDepthModel

        return PrefetchDepthModel().solve(
            self.compute_time, self.tail_latencies, self.batch_bytes, self.safety_depth, self.consumers_per_node
        )


class TieredCacheScenario(BaseScenario):
    """A fast tier in front of a slow one: a stated hit rate, or capacity over a uniformly
    sampled corpus.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.TieredCacheModel``.

    Assumptions: A stated hit fraction overrides capacity inference; otherwise accesses are
    uniform over nonoverlapping corpus parts.

    Excludes: Capacity-derived hit fractions are not clipped here, and eviction policy or
    locality is not simulated.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.TieredCacheModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.HotTierCacheExample
    >>> round(case.hit, 4)
    0.95
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    hit_rate: Optional[float] = None
    hot_capacity: Optional[Quantity] = None
    corpus_parts: Optional[Dict[str, Quantity]] = None  # a corpus sampled uniformly across its parts
    miss_penalty: Optional[float] = None  # miss / hit service time, when no bandwidths are given
    hot_bandwidth: Optional[Quantity] = None
    miss_bandwidth: Optional[Quantity] = None

    @property
    def hit(self) -> float:
        """Supplied hit fraction, or uniform-corpus capacity fraction when absent. Equation:
        ``hit_rate`` else ``hot_capacity / corpus_bytes``. Units: dimensionless; the capacity
        estimate assumes uniform access and is not clipped here.
        """
        if self.hit_rate is not None:
            return self.hit_rate
        return (self.hot_capacity / self.corpus_bytes).to("").magnitude

    @property
    def corpus_bytes(self) -> Quantity:
        """Total stored size of the named corpus parts. Equation: sum of ``corpus_parts.values()``.
        Units: bytes; the parts must be nonoverlapping.
        """
        return sum(self.corpus_parts.values())

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.TieredCacheModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import TieredCacheModel

        return TieredCacheModel().solve(self.hit, self.miss_penalty, self.hot_bandwidth, self.miss_bandwidth)


class HostIoCpuScenario(BaseScenario):
    """``devices`` each consuming ``requests_per_device`` reads per second, at each path's CPU
    service time.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.HostIoCpuModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.HostIoCpuModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.HostIoCpuModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.HostIoCpuModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.GpuDirectStorageNodeCpu
    >>> type(case.result).__name__
    'HostIoCpuResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    requests_per_device: float  # per second
    devices: int
    service_times: Dict[str, Quantity]

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.HostIoCpuModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import HostIoCpuModel

        return HostIoCpuModel().solve(self.devices * self.requests_per_device, self.service_times)


class ObjectStorageBillScenario(BaseScenario):
    """A dataset kept in object storage for a billing period and read ``reads`` times across a
    boundary.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ObjectStorageBillModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.ObjectStorageBillModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.ObjectStorageBillModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ObjectStorageBillModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ObjectStorage100TB
    >>> type(case.result).__name__
    'ObjectStorageBillResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    dataset_bytes: Quantity
    storage_rate: Quantity
    billing_period: Quantity
    egress_rate: Quantity
    reads: int
    files: int
    request_fee: Quantity
    requests_per_fee: int
    retrieval_rate: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ObjectStorageBillModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import ObjectStorageBillModel

        return ObjectStorageBillModel().solve(
            self.dataset_bytes,
            self.storage_rate,
            self.billing_period,
            self.egress_rate,
            self.reads,
            self.files,
            self.request_fee,
            self.requests_per_fee,
            self.retrieval_rate,
        )


class StreamVsStageScenario(BaseScenario):
    """Stream every epoch from remote object storage, or stage the dataset locally once per
    run.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.StreamVsStageModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.StreamVsStageModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.StreamVsStageModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.StreamVsStageModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.VisionCorpusStreamVsStage
    >>> type(case.result).__name__
    'StreamVsStageResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    dataset_bytes: Quantity
    epochs: int
    runs_per_year: int
    object_rate: Quantity
    local_rate: Quantity
    egress_rate: Quantity
    year: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.StreamVsStageModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import StreamVsStageModel

        return StreamVsStageModel().solve(
            self.dataset_bytes,
            self.epochs,
            self.runs_per_year,
            self.object_rate,
            self.local_rate,
            self.egress_rate,
            self.year,
        )


class UtilizationLossScenario(BaseScenario):
    """``fleet``'s accelerators billed at ``device_hour_rate`` while utilization falls.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.UtilizationLossModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.UtilizationLossModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.UtilizationLossModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.UtilizationLossModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.InputStallIdleCost1K
    >>> type(case.result).__name__
    'UtilizationLossResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet: Fleet
    device_hour_rate: Quantity
    healthy_utilization: float
    degraded_utilization: float
    duration: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.UtilizationLossModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import UtilizationLossModel

        return UtilizationLossModel().solve(
            self.fleet.total_accelerators,
            self.device_hour_rate,
            self.healthy_utilization,
            self.degraded_utilization,
            self.duration,
        )


class SyntheticFootprintScenario(BaseScenario):
    """Generated samples with provenance metadata, read once per verification pass.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.SyntheticFootprintModel``.

    Assumptions: The stated fields are inputs to
    ``mlsysim.solvers.SyntheticFootprintModel``; its documented validity regime, units, and
    modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.SyntheticFootprintModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.SyntheticFootprintModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SyntheticDataProvenance
    >>> type(case.result).__name__
    'SyntheticFootprintResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    payload_bytes: Quantity
    provenance_overhead: float
    verification_passes: int

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.SyntheticFootprintModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import SyntheticFootprintModel

        return SyntheticFootprintModel().solve(self.payload_bytes, self.provenance_overhead, self.verification_passes)


class ShardPackagingScenario(BaseScenario):
    """``files`` samples packaged into ``shards`` large files on a file system rated at
    ``rated_bandwidth``.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Every sample belongs to one shard; files-per-shard is an average, not an
    integer packing guarantee.

    Excludes: Metadata overhead and file-system throughput scaling are not calculated by
    this ratio.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ImageFileMetadataMeltdown
    >>> round(case.files_per_shard, 4)
    4000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    files: int
    shards: int
    rated_bandwidth: Quantity

    @property
    def files_per_shard(self) -> float:
        """Average small-file count represented by one shard. Equation: ``files / shards``. Units:
        files per shard; uneven shard sizes are summarized by an average.
        """
        return self.files / self.shards


class CompressedReadScenario(BaseScenario):
    """Compressed shards read at ``read_bandwidth``, decoded on CPU cores at each codec's
    output rate.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Compression ratios are uncompressed/compressed sizes; decoder rates are
    per-core uncompressed output rates and scale linearly across cores.

    Excludes: Codec startup, load imbalance, memory contention, and quality changes are
    absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.NodeStripeCodecs
    >>> type(case.output_bandwidth(next(iter(case.compression_ratios)))).__name__
    'Quantity'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    read_bandwidth: Quantity
    compression_ratios: Dict[str, float]  # decoded bytes per stored byte
    decode_rates: Dict[str, Quantity]  # decoded output per core

    def output_bandwidth(self, codec: str) -> Quantity:
        """Uncompressed byte rate generated by reading a named codec's compressed data. Equation:
        ``read_bandwidth * compression_ratios[codec]``. Units: bandwidth; the ratio is
        uncompressed/compressed size.
        """
        return self.read_bandwidth * self.compression_ratios[codec]

    def decode_cores(self, codec: str) -> float:
        """Ideal CPU-equivalent core demand to keep up with the decompressed stream. Equation:
        ``output_bandwidth(codec) / decode_rates[codec]``. Units: dimensionless core count;
        linear core scaling and identical cores are assumed.
        """
        return (self.output_bandwidth(codec) / self.decode_rates[codec]).to("").magnitude


class AlwaysOnDetectorTargetsScenario(BaseScenario):
    """An always-on detector's product contract: what it must catch, how rarely it may wake falsely,
    what it may occupy on the device, and the data budget and schedule that must deliver it.

    Notes
    -----
    Equation: this record declares detection, false-alarm, memory, time, and cost
    constraints without computing achieved performance.

    Assumptions: Targets are supplied product requirements; memory and time budgets carry
    units.

    Excludes: A solver proving these targets are jointly achievable or an empirical
    validation of detection quality.

    References
    ----------
    Derivation: this record declares detection, false-alarm,
    memory, time, and cost constraints without computing achieved performance. The
    individual properties state units and intermediate accounting; this class introduces
    no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative requirements; this reads its declared target.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.KeywordSpotterTargets
    >>> round(float(case.detection_target), 3)
    0.98"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    detection_target: float  # fraction of spoken wake words caught
    false_wake_budget: float  # expected false wakes per false_wake_horizon
    false_wake_horizon: Quantity
    response_limit: Quantity
    model_memory: Quantity  # always-on model-size budget
    preprocessing_memory: Quantity  # separate preprocessing-state budget
    data_budget: Quantity  # USD
    timeline: Quantity
    base_detection_rate: float  # an illustrative starting detection rate


class DataBudgetScenario(BaseScenario):
    """A data budget split into ``shares`` by category; the labeling share buys labels at
    ``label_price`` plus ``review_overhead`` for quality review.

    Notes
    -----
    Equation: category allocation = total * share; labels = labeling allocation / (label
    price * (1 + review overhead)).

    Assumptions: Shares and review overhead are supplied budget assumptions and the
    effective label price is positive.

    Excludes: Measured label quality, economies of scale, and integer rounding of
    purchased labels.

    References
    ----------
    Derivation: category allocation = total * share; labels
    = labeling allocation / (label price * (1 + review overhead)). The individual
    properties state units and intermediate accounting; this class introduces no fitted
    coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.KeywordSpotterDataBudget
    >>> round(float(case.effective_label_cost.to('USD').magnitude), 3)
    0.12"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    total: Quantity  # USD
    shares: Dict[str, float]
    label_price: Quantity  # USD per label
    review_overhead: float
    labeling_category: str = "labeling"

    def allocation(self, category: str) -> Quantity:
        """Currency allocated to a named category, total budget times its fractional share.

        Equation: ``return total * shares[category]``."""
        return self.total * self.shares[category]

    @property
    def effective_label_cost(self) -> Quantity:
        """Currency per label including proportional review overhead.

        Equation: ``return label_price * (1 + review_overhead)``."""
        return self.label_price * (1 + self.review_overhead)

    @property
    def labels(self) -> float:
        """Affordable label count, labeling allocation divided by effective label cost;
        fractional labels are retained.

        Equation: ``return (allocation(labeling_category) /
        effective_label_cost).to('').magnitude``."""
        return (self.allocation(self.labeling_category) / self.effective_label_cost).to("").magnitude


class LabelingVersusTrainingScenario(BaseScenario):
    """A labeling budget against one training run: ``epochs`` over ``dataset_examples`` at
    ``throughput_per_accelerator`` on ``accelerators``, priced at a low and a high hourly rate.

    Notes
    -----
    Equation: training hours = epochs * examples / aggregate throughput; bill = occupied
    device-hours * price.

    Assumptions: Accelerators sustain the same fixed per-device sample rate and training
    volume is constant.

    Excludes: Scaling loss, retries, evaluation overhead, and changes in labeling quality.

    References
    ----------
    Derivation: training hours = epochs * examples /
    aggregate throughput; bill = occupied device-hours * price. The individual properties
    state units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ImageNetLabelingVersusResNetRun
    >>> round(float(case.train_time.to('h').magnitude), 3)
    1.251"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    labeling_budget: Quantity  # USD
    dataset_examples: Quantity  # count
    epochs: int
    throughput_per_accelerator: Quantity  # count/second
    accelerators: int
    price_low: Quantity  # USD per accelerator-hour
    price_high: Quantity
    labeling_time_low: Optional[Quantity] = None  # calendar time the labeling takes, low
    labeling_time_high: Optional[Quantity] = None  # and high

    @property
    def train_time(self) -> Quantity:
        """Ideal training hours from epochs times examples divided by aggregate constant
        throughput.

        Equation: ``return (dataset_examples * epochs / (accelerators *
        throughput_per_accelerator)).to('hour')``."""
        return (self.dataset_examples * self.epochs / (self.accelerators * self.throughput_per_accelerator)).to("hour")

    @property
    def train_cost_low(self) -> Quantity:
        """Low-price training bill in USD for all occupied device-hours.

        Equation: ``return (accelerators * train_time * price_low).to('USD')``."""
        return (self.accelerators * self.train_time * self.price_low).to("USD")

    @property
    def train_cost_high(self) -> Quantity:
        """High-price training bill in USD for all occupied device-hours.

        Equation: ``return (accelerators * train_time * price_high).to('USD')``."""
        return (self.accelerators * self.train_time * self.price_high).to("USD")

    @property
    def ratio_low(self) -> float:
        """Labeling budget divided by the high-price training bill, a dimensionless lower
        cost ratio.

        Equation: ``return (labeling_budget / train_cost_high).to('').magnitude``."""
        return (self.labeling_budget / self.train_cost_high).to("").magnitude

    @property
    def ratio_high(self) -> float:
        """Labeling budget divided by the low-price training bill, a dimensionless upper cost
        ratio.

        Equation: ``return (labeling_budget / train_cost_low).to('').magnitude``."""
        return (self.labeling_budget / self.train_cost_low).to("").magnitude


class LabelGranularityScenario(BaseScenario):
    """Label entries for one image: a per-pixel mask against ``boxes`` boxes of ``coords_per_box``.

    Notes
    -----
    Equation: mask entries = width * height; box entries = boxes * coordinates per box.

    Assumptions: Images are rectangular and each pixel or coordinate contributes one label
    entry.

    Excludes: Different bit widths, compressed encodings, object-class metadata, and
    annotation time.

    References
    ----------
    Derivation: mask entries = width * height; box entries =
    boxes * coordinates per box. The individual properties state units and intermediate
    accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FullHdMaskVersusBoxes
    >>> round(float(case.mask_entries), 3)
    2073600.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    image_width: int
    image_height: int
    boxes: int
    coords_per_box: int

    @property
    def mask_entries(self) -> int:
        """Per-pixel label-entry count, image width times height; no encoding compression is
        assumed.

        Equation: ``return image_width * image_height``."""
        return self.image_width * self.image_height

    @property
    def box_entries(self) -> int:
        """Coordinate-entry count for all boxes, boxes times coordinates per box.

        Equation: ``return boxes * coords_per_box``."""
        return self.boxes * self.coords_per_box

    @property
    def mask_to_box_ratio(self) -> float:
        """Per-pixel entry count divided by box-coordinate entry count, not a measured labor
        ratio.

        Equation: ``return mask_entries / box_entries``."""
        return self.mask_entries / self.box_entries


class LaborEffortScenario(BaseScenario):
    """``items`` labeled by hand at ``time_per_item`` each: total hours and full-time person-years.

    Notes
    -----
    Equation: labor hours = item count * time per item; worker-years = hours / work-year
    hours.

    Assumptions: Per-item labeling time and productive hours per person-year are constant.

    Excludes: Learning curves, disagreement review, staffing queues, and calendar
    scheduling.

    References
    ----------
    Derivation: labor hours = item count * time per item;
    worker-years = hours / work-year hours. The individual properties state units and
    intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CityscapesMasksForMillionFrames
    >>> round(float(case.total_time.to('h').magnitude), 3)
    1500000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    items: float
    time_per_item: Quantity
    work_year: Quantity  # working hours in one full-time person-year

    @property
    def total_time(self) -> Quantity:
        """Total labeling labor in hours, item count times constant time per item.

        Equation: ``return (items * time_per_item).to('hour')``."""
        return (self.items * self.time_per_item).to("hour")

    @property
    def person_years(self) -> float:
        """Full-time worker-years of labeling effort, total hours divided by the declared
        work-year hours.

        Equation: ``return (total_time / work_year).to('').magnitude``."""
        return (self.total_time / self.work_year).to("").magnitude


class ActiveLearningBudgetScenario(BaseScenario):
    """Active learning against a fixed label budget when scoring the pool is charged to the same budget.

    Counts are examples; prices are USD. ``compute_share`` of the budget pays for
    scoring a narrowed candidate pool, and the rest buys labels.

    Notes
    -----
    Equation: budget = scoring spend + label spend; affordable counts divide each
    allocation by its unit price.

    Assumptions: Required label counts are stipulated for the target quality and unit
    prices remain constant.

    Excludes: Predicting an acquisition strategy's accuracy or discovering the required
    label count empirically.

    References
    ----------
    Derivation: budget = scoring spend + label spend;
    affordable counts divide each allocation by its unit price. The individual properties
    state units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ActiveLearningTenMillionPool
    >>> round(float(case.random_label_cost.to('USD').magnitude), 3)
    500000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pool: float
    label_budget: Quantity
    label_price: Quantity
    scoring_price: Quantity
    random_labels_needed: float
    active_labels_low: float
    active_labels_high: float
    compute_share: float
    baseline_accuracy: float
    target_accuracy: float

    @property
    def random_label_cost(self) -> Quantity:
        """Currency needed for the stipulated random-sampling label count.

        Equation: ``return random_labels_needed * label_price``."""
        return self.random_labels_needed * self.label_price

    @property
    def active_label_cost_low(self) -> Quantity:
        """Currency needed for the lower stipulated active-learning label count.

        Equation: ``return active_labels_low * label_price``."""
        return self.active_labels_low * self.label_price

    @property
    def active_label_cost_high(self) -> Quantity:
        """Currency needed for the upper stipulated active-learning label count.

        Equation: ``return active_labels_high * label_price``."""
        return self.active_labels_high * self.label_price

    @property
    def full_pool_scoring_cost(self) -> Quantity:
        """Currency charged to score every candidate in the full pool once.

        Equation: ``return pool * scoring_price``."""
        return self.pool * self.scoring_price

    @property
    def active_total_low(self) -> Quantity:
        """Currency for full-pool scoring plus the lower active-label requirement.

        Equation: ``return full_pool_scoring_cost + active_label_cost_low``."""
        return self.full_pool_scoring_cost + self.active_label_cost_low

    @property
    def active_total_high(self) -> Quantity:
        """Currency for full-pool scoring plus the upper active-label requirement.

        Equation: ``return full_pool_scoring_cost + active_label_cost_high``."""
        return self.full_pool_scoring_cost + self.active_label_cost_high

    @property
    def compute_budget(self) -> Quantity:
        """Currency allocated to candidate scoring by the declared compute share.

        Equation: ``return label_budget * compute_share``."""
        return self.label_budget * self.compute_share

    @property
    def candidate_pool(self) -> float:
        """Candidates that the scoring budget can process at the fixed per-candidate price.

        Equation: ``return (compute_budget / scoring_price).to('').magnitude``."""
        return (self.compute_budget / self.scoring_price).to("").magnitude

    @property
    def affordable_labels(self) -> float:
        """Labels purchasable after reserving the scoring budget; fractional counts are
        retained.

        Equation: ``return ((label_budget - compute_budget) /
        label_price).to('').magnitude``."""
        return ((self.label_budget - self.compute_budget) / self.label_price).to("").magnitude


class BatchVersusStreamingIngestScenario(BaseScenario):
    """One event stream ingested by always-on streaming cores or by a batch job each hour.

    Streaming runs ``stream_cores`` (``redundancy_factor`` cores per primary) for
    ``stream_hours_per_day``; batch runs ``batch_cores`` for ``batch_window`` of each
    clock hour. Both pay ``core_hour_price``.

    Notes
    -----
    Equation: daily bill = occupied core-hours * price; per-core throughput = required
    aggregate bandwidth / productive cores.

    Assumptions: A batch drains one hour of arrivals inside each batch window; streaming
    redundancy is already included in reserved core count.

    Excludes: Traffic bursts, queue latency, storage I/O bottlenecks, and scheduling
    overhead.

    References
    ----------
    Derivation: daily bill = occupied core-hours * price;
    per-core throughput = required aggregate bandwidth / productive cores. The individual
    properties state units and intermediate accounting; this class introduces no fitted
    coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MillionEventBatchVersusStreaming
    >>> round(float(case.throughput.to('GB/s').magnitude), 3)
    1.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    event_rate: Quantity  # events per second
    event_size: Quantity
    stream_cores: int
    stream_hours_per_day: Quantity
    batch_cores: int
    batch_window: Quantity
    core_hour_price: Quantity  # USD per core-hour
    redundancy_factor: int = 1

    @property
    def throughput(self) -> Quantity:
        """Incoming data rate in GB/s, event rate times bytes per event.

        Equation: ``return (event_rate * event_size).to('GB/s')``."""
        return (self.event_rate * self.event_size).to("GB/s")

    @property
    def hourly_data(self) -> Quantity:
        """Data arriving in one clock hour, expressed in TB.

        Equation: ``return (throughput * Q_(1, 'hour')).to('TB')``."""
        return (self.throughput * Q_(1, "hour")).to("TB")

    @property
    def stream_core_hours_per_day(self) -> float:
        """Reserved streaming core-hours per day, including redundant cores already in the
        count.

        Equation: ``return stream_cores * stream_hours_per_day.to('hour').magnitude``."""
        return self.stream_cores * self.stream_hours_per_day.to("hour").magnitude

    @property
    def stream_cost_per_day(self) -> Quantity:
        """Daily streaming bill in USD from reserved core-hours and price.

        Equation: ``return (stream_core_hours_per_day * Q_(1, 'hour') *
        core_hour_price).to('USD')``."""
        return (self.stream_core_hours_per_day * Q_(1, "hour") * self.core_hour_price).to("USD")

    @property
    def batch_core_hours_per_day(self) -> float:
        """Daily batch core-hours when one batch runs in every clock hour.

        Equation: ``hours_per_day = Q_(1, 'day').to('hour').magnitude; return batch_cores
        * batch_window.to('hour').magnitude * hours_per_day``."""
        hours_per_day = Q_(1, "day").to("hour").magnitude
        return self.batch_cores * self.batch_window.to("hour").magnitude * hours_per_day

    @property
    def batch_cost_per_day(self) -> Quantity:
        """Daily batch bill in USD from its occupied core-hours.

        Equation: ``return (batch_core_hours_per_day * Q_(1, 'hour') *
        core_hour_price).to('USD')``."""
        return (self.batch_core_hours_per_day * Q_(1, "hour") * self.core_hour_price).to("USD")

    @property
    def cost_ratio(self) -> float:
        """Streaming bill divided by batch bill, a dimensionless cost ratio.

        Equation: ``return (stream_cost_per_day / batch_cost_per_day).to('').magnitude``."""
        return (self.stream_cost_per_day / self.batch_cost_per_day).to("").magnitude

    @property
    def stream_primary_cores(self) -> int:
        """Primary streaming cores after integer division by the redundancy factor.

        Equation: ``return stream_cores // redundancy_factor``."""
        return self.stream_cores // self.redundancy_factor

    @property
    def stream_per_core(self) -> Quantity:
        """Incoming MB/s per primary streaming core, assuming equal traffic partitioning.

        Equation: ``return (throughput / stream_primary_cores).to('MB/s')``."""
        return (self.throughput / self.stream_primary_cores).to("MB/s")

    @property
    def batch_aggregate(self) -> Quantity:
        """GB/s needed to drain one hour's arrivals inside the batch window.

        Equation: ``return (hourly_data / batch_window).to('GB/s')``."""
        return (self.hourly_data / self.batch_window).to("GB/s")

    @property
    def batch_per_core(self) -> Quantity:
        """Batch decode rate in MB/s per core under equal work partitioning.

        Equation: ``return (batch_aggregate / batch_cores).to('MB/s')``."""
        return (self.batch_aggregate / self.batch_cores).to("MB/s")


class DesignTradeoffScenario(BaseScenario):
    """One design-space move of a knob from ``baseline`` to ``alternative``, with an illustrative
    quality gain from ``gain_low`` to ``gain_high`` (fractions); costs scale by ``ratio``.

    Notes
    -----
    Equation: cost multiplier = alternative knob value / baseline knob value.

    Assumptions: The scenario stipulates proportional cost scaling and a nonzero baseline;
    quality-gain ranges are supplied.

    Excludes: A fitted cost curve or a calculation establishing the quality gains.

    References
    ----------
    Derivation: cost multiplier = alternative knob value /
    baseline knob value. The individual properties state units and intermediate
    accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.KeywordSpotterSampleRateChoice
    >>> round(float(case.ratio), 3)
    2.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline: float
    alternative: float
    gain_low: float
    gain_high: float

    @property
    def ratio(self) -> float:
        """Alternative knob value divided by baseline, the stipulated cost-scaling factor.

        Equation: ``return alternative / baseline``."""
        return self.alternative / self.baseline


class AcceleratorInputFeedScenario(BaseScenario):
    """``accelerators`` copies of ``hardware`` training ``workload`` at ``sustained_rate`` samples a second each,
    fed from ``supply`` with ``encoded_sample`` bytes stored per sample and ``decoded_sample`` bytes after decode.

    The arithmetic ceiling is peak FLOP/s over the training FLOPs of one sample
    (``training_flops_factor`` times the forward pass). Demand is rate times bytes
    per sample; the feeding loss is the share of the training rate a slower
    supply removes (``DataModel``). ``worker_rate`` is what one host loader
    worker decodes and augments a second; ``host_decode_cap`` is where adding
    workers stops helping because the host's cores run out.

    Notes
    -----
    Equation: sample ceiling = peak arithmetic / training FLOPs per sample; bandwidth
    demand = samples per second * bytes per sample.

    Assumptions: Forward work and training multiplier define the arithmetic estimate;
    rates and encoded/decoded sample sizes refer to the same workload.

    Excludes: Host-contention scaling, asynchronous pipeline transients, and automatic
    enforcement of the optional host decode cap.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.DataModel.solve``. The solver contract records the equations and
    their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50A100InputFeed
    >>> round(float(case.training_flops_per_sample.to('flop').magnitude), 3)
    24600000000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    sustained_rate: Quantity  # samples per second per accelerator
    # Optional (added 2026-10-20): None for a compute-ceiling comparison with no input path.
    encoded_sample: Optional[Quantity] = None  # bytes per sample as stored
    decoded_sample: Optional[Quantity] = None  # bytes per sample after decode
    supply: Any = None  # StorageSubsystem that feeds the accelerators
    training_flops_factor: float = 3.0
    accelerators: int = 1
    worker_rate: Optional[Quantity] = None  # samples per second per loader worker
    host_decode_cap: Optional[Quantity] = None  # samples per second the whole host can decode
    # Optional (added 2026-10-20): loader workers the host gives each accelerator.
    host_workers: Optional[int] = None

    @property
    def host_supply(self) -> Quantity:
        """Host-decoded samples per second from worker count times per-worker throughput.

        Equation: ``return (host_workers * worker_rate).to('count/second')``."""
        return (self.host_workers * self.worker_rate).to("count/second")

    @property
    def busy_fraction(self) -> float:
        """Accelerator busy fraction limited by host supply, capped at one.

        Equation: ``return min(1.0, (host_supply / sustained_rate).to('').magnitude)``."""
        return min(1.0, (self.host_supply / self.sustained_rate).to("").magnitude)

    @property
    def training_flops_per_sample(self) -> Quantity:
        """Training work in FLOPs per sample, forward work times the declared training
        multiplier.

        Equation: ``return (workload.inference_flops *
        training_flops_factor).to('flop')``."""
        return (self.workload.inference_flops * self.training_flops_factor).to("flop")

    @property
    def ceiling_rate(self) -> Quantity:
        """Samples per second at peak arithmetic throughput for one accelerator.

        Equation: ``return (hardware.compute.peak_flops /
        training_flops_per_sample).to('count/second')``."""
        return (self.hardware.compute.peak_flops / self.training_flops_per_sample).to("count/second")

    @property
    def sustained_fraction(self) -> float:
        """Measured or assumed sample throughput divided by the modeled arithmetic ceiling.

        Equation: ``return (sustained_rate / ceiling_rate).to('').magnitude``."""
        return (self.sustained_rate / self.ceiling_rate).to("").magnitude

    @property
    def achieved_flops(self) -> Quantity:
        """Useful training arithmetic rate in TFLOP/s at the supplied sample throughput.

        Equation: ``return (sustained_rate * training_flops_per_sample / Q_(1,
        'count')).to('TFLOP/s')``."""
        return (self.sustained_rate * self.training_flops_per_sample / Q_(1, "count")).to("TFLOP/s")

    @property
    def ceiling_gap(self) -> float:
        """Arithmetic ceiling divided by sustained throughput, a dimensionless gap factor.

        Equation: ``return 1 / sustained_fraction``."""
        return 1 / self.sustained_fraction

    @property
    def total_rate(self) -> Quantity:
        """Aggregate sustained samples per second for the declared accelerator count.

        Equation: ``return (accelerators * sustained_rate).to('count/second')``."""
        return (self.accelerators * self.sustained_rate).to("count/second")

    def demand(self, decoded: bool = False, rate: Optional[Quantity] = None) -> Quantity:
        """Required MB/s for all accelerators, using encoded or decoded bytes and the
        selected per-device rate.

        Equation: ``per = sustained_rate if rate is None else rate; size = decoded_sample
        if decoded else encoded_sample; return (accelerators * per * size / Q_(1,
        'count')).to('MB/s')``."""
        per = self.sustained_rate if rate is None else rate
        size = self.decoded_sample if decoded else self.encoded_sample
        return (self.accelerators * per * size / Q_(1, "count")).to("MB/s")

    @property
    def ceiling_demand(self) -> Quantity:
        """Required encoded bandwidth in MB/s if every accelerator reached the arithmetic
        ceiling.

        Equation: ``return demand(rate=ceiling_rate)``."""
        return self.demand(rate=self.ceiling_rate)

    @property
    def decode_expansion(self) -> float:
        """Decoded sample bytes divided by encoded sample bytes, a dimensionless expansion
        factor.

        Equation: ``return (decoded_sample / encoded_sample).to('').magnitude``."""
        return (self.decoded_sample / self.encoded_sample).to("").magnitude

    def feeding_loss(self, decoded: bool = False) -> float:
        """Lost throughput fraction from ``DataModel.solve`` for the selected storage demand.

        Equation: Delegates the declared fields to ``mlsysim.solvers.DataModel.solve``;
        returned quantities and feasibility follow that contract."""
        from ..solvers import DataModel

        node = self.hardware.model_copy(update={"storage": self.supply})
        r = DataModel().solve(self.demand(decoded), node)
        return 1.0 - 1.0 / max(r.utilization, 1.0)

    @property
    def supplied_fraction(self) -> float:
        """Fraction of encoded sustained demand delivered by storage, capped at one.

        Equation: ``return min((supply.bandwidth / demand()).to('').magnitude, 1.0)``."""
        return min((self.supply.bandwidth / self.demand()).to("").magnitude, 1.0)

    def samples_supplied(self, bandwidth: Quantity) -> Quantity:
        """Encoded samples per second supported by the supplied storage bandwidth.

        Equation: ``return (bandwidth / encoded_sample * Q_(1,
        'count')).to('count/second')``."""
        return (bandwidth / self.encoded_sample * Q_(1, "count")).to("count/second")

    def streams_needed(self, bandwidth: Quantity, rate: Optional[Quantity] = None) -> float:
        """Fractional count of equal-bandwidth streams needed to meet the selected demand.

        Equation: ``target = ceiling_demand if rate is None else demand(rate=rate); return
        (target / bandwidth).to('').magnitude``."""
        target = self.ceiling_demand if rate is None else self.demand(rate=rate)
        return (target / bandwidth).to("").magnitude

    @property
    def worker_crossing(self) -> float:
        """Fractional worker count whose aggregate decode rate matches one accelerator's
        sustained demand.

        Equation: ``return (sustained_rate / worker_rate).to('').magnitude``."""
        return (self.sustained_rate / self.worker_rate).to("").magnitude

    @property
    def workers_per_accelerator(self) -> int:
        """Whole loader-worker count per accelerator, rounding the crossing upward with
        floating-point tolerance.

        Equation: ``return math.ceil(worker_crossing - 1e-09)``."""
        import math

        return math.ceil(self.worker_crossing - 1e-9)

    @property
    def workers_pooled(self) -> int:
        """Whole loader-worker count for aggregate demand in one shared pool, rounded upward.

        Equation: ``return math.ceil((total_rate / worker_rate).to('').magnitude -
        1e-09)``."""
        import math

        return math.ceil((self.total_rate / self.worker_rate).to("").magnitude - 1e-9)


class FeatureLookupScenario(BaseScenario):
    """An online service at ``request_rate`` that reads ``reads_per_request`` stored features per request.

    Optional (added 2026-10-20): ``request_budget`` is the end-to-end latency
    budget of one request and ``lookup_budget_low``/``lookup_budget_high`` the
    share of it the feature store may take.

    Notes
    -----
    Equation: read rate = request rate * reads per request; size reduction = raw bytes /
    feature bytes.

    Assumptions: Every request reads the same number of features and optional byte sizes
    are supplied when comparing representation sizes.

    Excludes: Cache hit rates, batching, lookup latency distributions, and storage
    amplification.

    References
    ----------
    Derivation: read rate = request rate * reads per
    request; size reduction = raw bytes / feature bytes. The individual properties state
    units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RecommendationFeatureLookups
    >>> round(float(case.reads_per_day), 3)
    8640000000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    request_rate: Quantity  # requests per second
    reads_per_request: int
    feature_bytes: Optional[Quantity] = None
    raw_sample_bytes: Optional[Quantity] = None  # the raw record a feature vector summarizes
    request_budget: Optional[Quantity] = None
    lookup_budget_low: Optional[Quantity] = None
    lookup_budget_high: Optional[Quantity] = None

    @property
    def reads_per_day(self) -> float:
        """Expected feature reads per day, request rate times reads per request.

        Equation: ``return (request_rate * reads_per_request).to('1/day').magnitude``."""
        return (self.request_rate * self.reads_per_request).to("1/day").magnitude

    @property
    def reads_per_second(self) -> float:
        """Expected feature reads per second at the stated request rate.

        Equation: ``return (request_rate * reads_per_request).to('1/second').magnitude``."""
        return (self.request_rate * self.reads_per_request).to("1/second").magnitude

    @property
    def storage_reduction(self) -> float:
        """Raw-record bytes divided by feature-vector bytes, a dimensionless representation-
        size ratio.

        Equation: ``return (raw_sample_bytes / feature_bytes).to('').magnitude``."""
        return (self.raw_sample_bytes / self.feature_bytes).to("").magnitude


class CorpusTierScenario(BaseScenario):
    """An audio corpus (``dataset``) kept on storage tiers at ``storage_rates`` and read at ``read_bandwidths``.

    Notes
    -----
    Equation: audio bytes = examples * duration * sample rate * sample width; monthly bill
    = bytes * storage price.

    Assumptions: Audio is stored at the declared raw sample width and each tier sustains
    its stated sequential bandwidth.

    Excludes: Compression, metadata, access-operation charges, replication, and egress.

    References
    ----------
    Derivation: audio bytes = examples * duration * sample
    rate * sample width; monthly bill = bytes * storage price. The individual properties
    state units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MswcCorpusStorageTiers
    >>> round(float(case.corpus_bytes.to('GB').magnitude), 3)
    748.8"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    dataset: Any  # DatasetProfile with count, duration, sample rate and width
    storage_rates: Dict[str, Quantity]  # USD per byte-month, by tier
    read_bandwidths: Dict[str, Quantity]  # sequential read, by tier

    @property
    def corpus_bytes(self) -> Quantity:
        """Raw audio corpus size in GB: examples times duration, sample rate, and bytes per
        sample.

        Equation: ``d = dataset; return (d.training_examples.to('count').magnitude *
        d.sample_duration * d.sample_rate * d.sample_width).to('GB')``."""
        d = self.dataset
        return (d.training_examples.to("count").magnitude * d.sample_duration * d.sample_rate * d.sample_width).to("GB")

    def monthly_cost(self, tier: str) -> Quantity:
        """Storage bill in dollars per month for the named tier at the full raw corpus size.

        Equation: ``return (corpus_bytes * storage_rates[tier]).to('dollar/month')``."""
        return (self.corpus_bytes * self.storage_rates[tier]).to("dollar/month")

    def load_time(self, tier: str) -> Quantity:
        """Ideal sequential load duration in seconds from corpus size divided by tier read
        bandwidth.

        Equation: ``return (corpus_bytes / read_bandwidths[tier]).to('second')``."""
        return (self.corpus_bytes / self.read_bandwidths[tier]).to("second")


class ColumnProjectionScenario(BaseScenario):
    """A model that uses ``columns_used`` of ``columns_total`` equal-size columns, read from a row or a columnar file.

    Notes
    -----
    Equation: selected fraction = used columns / total columns; ideal speedup = 1 /
    selected fraction.

    Assumptions: Columns have equal byte sizes and unused columns can be skipped
    completely.

    Excludes: Predicate selectivity, decompression overhead, row-group metadata, and
    measured end-to-end speedup.

    References
    ----------
    Derivation: selected fraction = used columns / total
    columns; ideal speedup = 1 / selected fraction. The individual properties state units
    and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FraudModelColumnProjection
    >>> round(float(case.read_fraction), 3)
    0.2"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    columns_used: int
    columns_total: int

    @property
    def read_fraction(self) -> float:
        """Fraction of equal-size columns that a projection needs to fetch.

        Equation: ``return columns_used / columns_total``."""
        return self.columns_used / self.columns_total

    @property
    def wasted_fraction(self) -> float:
        """Fraction of row-scan bytes discarded because their columns are unused.

        Equation: ``return 1 - read_fraction``."""
        return 1 - self.read_fraction

    @property
    def speedup(self) -> float:
        """Ideal I/O speedup from projection, reciprocal of the selected-column fraction.

        Equation: ``return 1 / read_fraction``."""
        return 1 / self.read_fraction


class CodecEpochScenario(BaseScenario):
    """``epoch_bytes`` decompressed on the host each epoch for ``epochs`` epochs, at each codec's ``decode_rates``.

    Notes
    -----
    Equation: decode duration = epoch bytes / codec rate; run saving = epochs * per-epoch
    saving.

    Assumptions: Each epoch decodes the full corpus at a fixed supplied codec rate.

    Excludes: Storage overlap, compression-ratio differences, and codec warm-up or
    parallel scaling.

    References
    ----------
    Derivation: decode duration = epoch bytes / codec rate;
    run saving = epochs * per-epoch saving. The individual properties state units and
    intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.GzipVersusSnappyEpochs
    >>> round(float(case.decode_time('gzip').to('min').magnitude), 3)
    13.889"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    epoch_bytes: Quantity
    epochs: int
    decode_rates: Dict[str, Quantity]

    def decode_time(self, codec: str) -> Quantity:
        """CPU decompression time for one epoch in minutes at the selected codec's fixed
        rate.

        Equation: ``return (epoch_bytes / decode_rates[codec]).to('minute')``."""
        return (self.epoch_bytes / self.decode_rates[codec]).to("minute")

    def time_saved(self, slow: str, fast: str) -> Quantity:
        """Per-epoch minutes saved by the fast codec relative to the slow one.

        Equation: ``return decode_time(slow) - decode_time(fast)``."""
        return self.decode_time(slow) - self.decode_time(fast)

    def run_time_saved(self, slow: str, fast: str) -> Quantity:
        """Total decompression hours saved across all epochs.

        Equation: ``return (epochs * time_saved(slow, fast)).to('hour')``."""
        return (self.epochs * self.time_saved(slow, fast)).to("hour")

    def rate_ratio(self, fast: str, slow: str) -> float:
        """Fast codec throughput divided by slow codec throughput, a dimensionless ratio.

        Equation: ``return (decode_rates[fast] / decode_rates[slow]).to('').magnitude``."""
        return (self.decode_rates[fast] / self.decode_rates[slow]).to("").magnitude


class CorpusRepetitionScenario(BaseScenario):
    """A training ``run`` (``ClusterTrainingTimeScenario``) set against the unique tokens of its
    workload's own corpus.

    The corpus holds ``training_dataset_size / bytes_per_token`` unique tokens. The
    run processes ``training_ops / (flops_per_param_token x parameters)`` tokens,
    so it passes over the corpus that many times. A compute-optimal fit asks for
    ``tokens_per_param`` training tokens per parameter.

    Notes
    -----
    Equation: ``passes`` is the worked quantity; each property below states its equation and units.

    Assumptions: The run uses fixed per-parameter token FLOPs, corpus bytes per token, and board power.

    Excludes: Data-quality effects, changing utilization, and measured energy or compute-optimal quality.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_compute_optimal_allocation``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt2XlWebTextCorpus
    >>> value = case.passes
    >>> round(value, 3)
    16.667"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    run: Any  # ClusterTrainingTimeScenario
    bytes_per_token: Quantity  # bytes per token
    tokens_per_param: float
    flops_per_param_token: float

    @property
    def workload(self) -> Workload:
        """Workload specification carried by the training run.

        Equation:
            return self.run.workload"""
        return self.run.workload

    @property
    def hardware(self) -> HardwareNode:
        """Accelerator specification carried by the training run.

        Equation:
            return self.run.hardware"""
        return self.run.hardware

    @property
    def accelerators(self) -> int:
        """Accelerator count used by the training run.

        Equation:
            return self.run.n_devices"""
        return self.run.n_devices

    @property
    def utilization(self) -> float:
        """Declared compute-utilization fraction of the training run.

        Equation:
            return self.run.efficiency"""
        return self.run.efficiency

    @property
    def parameters(self) -> float:
        """Total model parameters as a scalar count.

        Equation:
            return self.workload.parameters.to('param').magnitude"""
        return self.workload.parameters.to("param").magnitude

    @property
    def corpus_bytes(self) -> Quantity:
        """Registered unique-corpus storage size, retaining its byte units.

        Equation:
            return self.workload.training_dataset_size"""
        return self.workload.training_dataset_size

    @property
    def training_flops(self) -> Quantity:
        """Registered training work, retaining its FLOP units.

        Equation:
            return self.workload.training_ops"""
        return self.workload.training_ops

    @property
    def unique_tokens(self) -> Quantity:
        """Unique token count inferred by dividing corpus bytes by bytes per token.

        Equation:
            return (self.corpus_bytes / self.bytes_per_token).to('count')"""
        return (self.corpus_bytes / self.bytes_per_token).to("count")

    @property
    def optimal_tokens(self) -> Quantity:
        """Token count requested by the supplied compute-optimal tokens-per-parameter ratio.

        Training tokens the compute-optimal ratio asks for at this parameter count.

        Equation:
            return Q_(self.tokens_per_param * self.parameters, 'count')"""
        return Q_(self.tokens_per_param * self.parameters, "count")

    @property
    def shortfall(self) -> float:
        """Dimensionless ratio of desired compute-optimal tokens to unique corpus tokens.

        Compute-optimal tokens over the corpus's unique tokens.

        Equation:
            return (self.optimal_tokens / self.unique_tokens).to('').magnitude"""
        return (self.optimal_tokens / self.unique_tokens).to("").magnitude

    @property
    def unique_tokens_per_param(self) -> float:
        """Unique corpus tokens per parameter, as a dimensionless count ratio.

        Equation:
            return self.unique_tokens.magnitude / self.parameters"""
        return self.unique_tokens.magnitude / self.parameters

    @property
    def tokens_processed(self) -> Quantity:
        """Processed training tokens inferred from total FLOPs and per-parameter token work.

        Equation:
            return Q_(self.training_flops.to('flop').magnitude / (self.flops_per_param_token * self.parameters), 'count')"""
        return Q_(self.training_flops.to("flop").magnitude / (self.flops_per_param_token * self.parameters), "count")

    @property
    def trained_tokens_per_param(self) -> float:
        """Processed training tokens per parameter, as a dimensionless count ratio.

        Equation:
            return self.tokens_processed.magnitude / self.parameters"""
        return self.tokens_processed.magnitude / self.parameters

    @property
    def passes(self) -> float:
        """Effective complete corpus passes, processed tokens divided by unique tokens.

        Times the run read its corpus.

        Equation:
            return (self.tokens_processed / self.unique_tokens).to('').magnitude"""
        return (self.tokens_processed / self.unique_tokens).to("").magnitude

    @property
    def run_time(self) -> Quantity:
        """Modeled training-run duration in seconds.

        Equation:
            return self.run.training_time.to('second')"""
        return self.run.training_time.to("second")

    @property
    def run_energy(self) -> Quantity:
        """Whole-fleet board-power energy in joules, accelerator count times TDP times duration.

        Accelerator energy at board power for the whole run.

        Equation:
            return (self.accelerators * self.hardware.tdp * self.run_time).to('joule')"""
        return (self.accelerators * self.hardware.tdp * self.run_time).to("joule")

    @property
    def compute_optimal(self):
        """Compute-optimal parameter/token allocation from the shared scaling helper.

        (parameters, tokens) that spend the same compute at ``tokens_per_param``.

        Equation:
            return calc_compute_optimal_allocation(self.training_flops.to('flop').magnitude, self.tokens_per_param, self.flops_per_param_token)"""
        from ...physics.scaling import calc_compute_optimal_allocation

        return calc_compute_optimal_allocation(
            self.training_flops.to("flop").magnitude, self.tokens_per_param, self.flops_per_param_token
        )


class AllocationGrowthScenario(BaseScenario):
    """A compute-optimal allocation ``X_opt ~ C^exponent`` when compute grows by ``compute_multiple``.

    Notes
    -----
    Equation: ``growth`` is the worked quantity; each property below states its equation and units.

    Assumptions: A positive compute multiple scales an allocation by the supplied power-law exponent.

    Excludes: Fitting the exponent or predicting training quality.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_allocation_growth``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ChinchillaComputeDoubling
    >>> value = case.growth
    >>> round(value, 3)
    0.414"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    compute_multiple: float
    exponent: float

    @property
    def growth(self) -> float:
        """Dimensionless allocation growth from the shared power-law helper.

        Fractional growth of the allocation.

        Equation:
            return calc_allocation_growth(self.compute_multiple, self.exponent)"""
        from ...physics.scaling import calc_allocation_growth

        return calc_allocation_growth(self.compute_multiple, self.exponent)


class DuplicateRemovalScenario(BaseScenario):
    """Near-duplicate shares removed from ``run``'s corpus: the training time and energy each
    share returns, against one read of the corpus and the MinHash signatures of its documents.

    Notes
    -----
    Equation: ``read_time`` is the worked quantity; each property below states its equation and units.

    Assumptions: Removed data proportionally reduces training work and signatures use fixed hashes per document.

    Excludes: Deduplication quality, selection computation, index overhead, and nonlinear training effects.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt2WebTextNearDuplicates
    >>> value = case.read_time
    >>> print(f"{value:.3f~P}")
    5.714 s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    run: CorpusRepetitionScenario
    duplicate_shares: Dict[str, float]  # corpus name -> fraction of documents removed
    documents: float
    signature_hashes: int
    hash_bytes: Quantity
    read_bandwidth: Quantity

    def time_saved(self, corpus: str) -> Quantity:
        """Hours avoided for the named corpus, removed share times full-run duration.

        Equation:
            return (self.duplicate_shares[corpus] * self.run.run_time).to('hour')"""
        return (self.duplicate_shares[corpus] * self.run.run_time).to("hour")

    def energy_saved(self, corpus: str) -> Quantity:
        """Kilowatt-hours avoided for the named corpus at the run's board-power estimate.

        Equation:
            return (self.duplicate_shares[corpus] * self.run.run_energy).to('kWh')"""
        return (self.duplicate_shares[corpus] * self.run.run_energy).to("kWh")

    @property
    def signature_bytes(self) -> Quantity:
        """Raw MinHash signature storage in GB, documents times hashes times bytes per hash.

        Equation:
            return (self.documents * self.signature_hashes * self.hash_bytes).to('GB')"""
        return (self.documents * self.signature_hashes * self.hash_bytes).to("GB")

    @property
    def read_time(self) -> Quantity:
        """One full corpus read in seconds at the specified sustained read bandwidth.

        Equation:
            return (self.run.corpus_bytes / self.read_bandwidth).to('second')"""
        return (self.run.corpus_bytes / self.read_bandwidth).to("second")

    @property
    def all_pairs(self) -> float:
        """Unordered document-pair count for exhaustive comparison.

        Pairwise comparisons an exhaustive search would make.

        Equation:
            return self.documents * (self.documents - 1) / 2"""
        return self.documents * (self.documents - 1) / 2


class MinHashBandingScenario(BaseScenario):
    """A MinHash signature cut into ``bands`` bands of ``rows`` hashes over ``shingle_words``-word shingles.

    Notes
    -----
    Equation: ``threshold`` is the worked quantity; each property below states its equation and units.

    Assumptions: Hash matches and bands follow the independence approximation of the shared LSH helpers.

    Excludes: Correlated hashes, exact index overhead, and empirical recall calibration.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_lsh_candidate_probability``, ``calc_lsh_threshold``, ``calc_minhash_standard_error``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TrajectoryMinHashScreen
    >>> value = case.threshold
    >>> round(value, 3)
    0.42"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    bands: int
    rows: int
    shingle_words: Optional[int] = None  # None when the features are not word shingles (added 2026-10-20)
    probe_similarities: List[float] = []  # Jaccard similarities at which to read the candidate chance
    # Optional (added 2026-10-20): stored width of one hash value and the records
    # indexed, for the raw signature matrix.
    hash_bytes: Optional[Quantity] = None
    records: Optional[int] = None

    @property
    def max_standard_error(self) -> float:
        """Largest Jaccard standard error, evaluated at probability one half with the full hash count.

        Largest standard error of the Jaccard estimate (at J = 0.5) from the full signature.

        Equation:
            return calc_minhash_standard_error(0.5, self.hashes)"""
        from ...physics.retrieval import calc_minhash_standard_error

        return calc_minhash_standard_error(0.5, self.hashes)

    @property
    def signature_bytes(self) -> Quantity:
        """One record's signature storage in bytes; requires hash_bytes.

        Equation:
            return (self.hashes * self.hash_bytes).to('byte')"""
        return (self.hashes * self.hash_bytes).to("byte")

    @property
    def index_bytes(self) -> Quantity:
        """Raw signature-index storage in bytes; requires records and hash_bytes.

        Equation:
            return (self.records * self.signature_bytes).to('byte')"""
        return (self.records * self.signature_bytes).to("byte")

    @property
    def hashes(self) -> int:
        """Hash count in the full signature, bands times rows.

        Equation:
            return self.bands * self.rows"""
        return self.bands * self.rows

    @property
    def threshold(self) -> float:
        """Dimensionless approximate LSH transition threshold from the shared retrieval helper.

        Equation:
            return calc_lsh_threshold(self.bands, self.rows)"""
        from ...physics.retrieval import calc_lsh_threshold

        return calc_lsh_threshold(self.bands, self.rows)

    def candidate_probability(self, similarity: float) -> float:
        """Candidate probability for the supplied Jaccard similarity under independent bands.

        Equation:
            return calc_lsh_candidate_probability(similarity, self.bands, self.rows)"""
        from ...physics.retrieval import calc_lsh_candidate_probability

        return calc_lsh_candidate_probability(similarity, self.bands, self.rows)


class InformationComputeRatioScenario(BaseScenario):
    """Accuracy gained per training FLOP for one epoch over all of ``dataset`` and over a
    ``keep_fraction`` subset, with ``gain_full`` and ``gain_subset`` points of gain.

    Notes
    -----
    Equation: ``icr_ratio`` is the worked quantity; each property below states its equation and units.

    Assumptions: Accuracy gains are supplied percentage-point observations and work scales with retained sample count.

    Excludes: Predicting accuracy gains, convergence, or the subset-selection computation.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50HalfImageNetIcr
    >>> value = case.icr_ratio
    >>> round(value, 3)
    1.8"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    dataset: Any  # DatasetProfile
    gain_full: float  # percentage points
    gain_subset: float  # percentage points
    keep_fraction: float
    training_flops_factor: float = 3.0

    @property
    def examples(self) -> float:
        """Dataset training-example count as a scalar.

        Equation:
            return self.dataset.training_examples.to('count').magnitude"""
        return self.dataset.training_examples.to("count").magnitude

    @property
    def training_flops_per_sample(self) -> Quantity:
        """Assumed training work per sample in GFLOP, forward work times training factor.

        Equation:
            return (self.workload.inference_flops * self.training_flops_factor).to('GFLOP')"""
        return (self.workload.inference_flops * self.training_flops_factor).to("GFLOP")

    @property
    def epoch_flops(self) -> Quantity:
        """FLOPs for one full training epoch.

        Equation:
            return (self.examples * self.training_flops_per_sample).to('flop')"""
        return (self.examples * self.training_flops_per_sample).to("flop")

    @property
    def subset_examples(self) -> float:
        """Expected retained-example count, with no integer rounding.

        Equation:
            return self.examples * self.keep_fraction"""
        return self.examples * self.keep_fraction

    @property
    def subset_flops(self) -> Quantity:
        """FLOPs for one epoch over the retained subset.

        Equation:
            return (self.subset_examples * self.training_flops_per_sample).to('flop')"""
        return (self.subset_examples * self.training_flops_per_sample).to("flop")

    @property
    def icr_full(self) -> float:
        """Supplied full-data accuracy gain in percentage points per training FLOP.

        Equation:
            return self.gain_full / self.epoch_flops.magnitude"""
        return self.gain_full / self.epoch_flops.magnitude

    @property
    def icr_subset(self) -> float:
        """Supplied subset accuracy gain in percentage points per training FLOP.

        Equation:
            return self.gain_subset / self.subset_flops.magnitude"""
        return self.gain_subset / self.subset_flops.magnitude

    @property
    def icr_ratio(self) -> float:
        """Dimensionless subset-to-full information/compute ratio.

        Equation:
            return self.icr_subset / self.icr_full"""
        return self.icr_subset / self.icr_full


class CoresetRunScenario(BaseScenario):
    """``workload`` trained for ``epochs`` over ``dataset`` on one ``hardware`` at ``utilization``,
    against a ``keep_fraction`` coreset scored by one forward pass of ``workload`` or ``proxy``.

    The selection inequality compares scoring plus subset training with full
    training; rescoring repeats the scoring pass before every epoch. The data-path
    fields follow one epoch through storage (the archive at the node's storage
    bandwidth), the host (``host_rate`` decoded samples a second), and the
    host-to-accelerator link (``decoded_sample`` bytes each).

    Notes
    -----
    Equation: ``epoch_time`` is the worked quantity; each property below states its equation and units.

    Assumptions: Training and scoring use fixed FLOPs per sample and a fixed sustained device rate.

    Excludes: Quality preservation, memory/communication limits, overlap, and measured power; TDP prices time.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50TenPercentCoresetH100
    >>> value = case.epoch_time
    >>> print(f"{value:.3f~P}")
    126.303 s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    dataset: Any  # DatasetProfile
    hardware: HardwareNode
    utilization: float
    keep_fraction: float
    epochs: int
    training_flops_factor: float = 3.0
    proxy: Optional[Workload] = None
    proxy_epochs: Optional[int] = None  # epochs a proxy trained from scratch runs before scoring
    host_rate: Optional[Quantity] = None  # samples per second the host prepares
    decoded_sample: Optional[Quantity] = None  # bytes per sample sent to the accelerator

    @property
    def examples(self) -> float:
        """Dataset training-example count as a scalar.

        Equation:
            return self.dataset.training_examples.to('count').magnitude"""
        return self.dataset.training_examples.to("count").magnitude

    @property
    def sustained_rate(self) -> Quantity:
        """Sustained compute rate, accelerator peak FLOP rate times declared utilization.

        Equation:
            return self.hardware.compute.peak_flops * self.utilization"""
        return self.hardware.compute.peak_flops * self.utilization

    @property
    def training_flops_per_sample(self) -> Quantity:
        """Assumed training FLOPs per sample, forward work times training factor.

        Equation:
            return self.training_flops_factor * self.workload.inference_flops"""
        return self.training_flops_factor * self.workload.inference_flops

    @property
    def epoch_flops(self) -> Quantity:
        """FLOPs for one full-data training epoch.

        Equation:
            return (self.examples * self.training_flops_per_sample).to('flop')"""
        return (self.examples * self.training_flops_per_sample).to("flop")

    @property
    def epoch_time(self) -> Quantity:
        """Compute-only full-epoch duration in seconds.

        Equation:
            return (self.epoch_flops / self.sustained_rate).to('second')"""
        return (self.epoch_flops / self.sustained_rate).to("second")

    @property
    def epoch_energy(self) -> Quantity:
        """Full-epoch board-power energy in kJ, TDP times modeled compute duration.

        Equation:
            return (self.hardware.tdp * self.epoch_time).to('kJ')"""
        return (self.hardware.tdp * self.epoch_time).to("kJ")

    @property
    def sample_rate(self) -> float:
        """Samples per second consumed by the modeled accelerator compute path.

        Samples a second the accelerator consumes at the sustained rate.

        Equation:
            return self.examples / self.epoch_time.magnitude"""
        return self.examples / self.epoch_time.magnitude

    @property
    def run_time(self) -> Quantity:
        """Full-training duration in hours over the specified epoch count.

        ``epochs`` full epochs.

        Equation:
            return (self.epochs * self.epoch_time).to('hour')"""
        return (self.epochs * self.epoch_time).to("hour")

    # Selection inequality
    @property
    def full_time(self) -> Quantity:
        """Compute-only full-training duration in seconds.

        Equation:
            return (self.examples * self.epochs * self.training_flops_factor * self.workload.inference_flops / self.sustained_rate).to('second')"""
        return (
            self.examples
            * self.epochs
            * self.training_flops_factor
            * self.workload.inference_flops
            / self.sustained_rate
        ).to("second")

    @property
    def subset_time(self) -> Quantity:
        """Subset training duration, retained share times full-training duration.

        Equation:
            return self.keep_fraction * self.full_time"""
        return self.keep_fraction * self.full_time

    def score_time(self, scorer: Optional[Workload] = None) -> Quantity:
        """Seconds for one full-pool forward scoring pass; scorer defaults to the workload.

        One forward pass of ``scorer`` (default: the workload) over the whole pool.

        Equation:
            model = scorer or self.workload
            return (self.examples * model.inference_flops / self.sustained_rate).to('second')"""
        model = scorer or self.workload
        return (self.examples * model.inference_flops / self.sustained_rate).to("second")

    def score_energy(self, scorer: Optional[Workload] = None) -> Quantity:
        """Board-power energy in kJ for one full-pool scoring pass.

        Equation:
            return (self.hardware.tdp * self.score_time(scorer)).to('kJ')"""
        return (self.hardware.tdp * self.score_time(scorer)).to("kJ")

    def selection_time(self, scorer: Optional[Workload] = None) -> Quantity:
        """Seconds for one scoring pass followed by subset training.

        Scoring once plus training on the subset.

        Equation:
            return self.score_time(scorer) + self.subset_time"""
        return self.score_time(scorer) + self.subset_time

    def saving(self, scorer: Optional[Workload] = None) -> Quantity:
        """Seconds saved after charging scoring and subset training; negative means a loss.

        Equation:
            return self.full_time - self.selection_time(scorer)"""
        return self.full_time - self.selection_time(scorer)

    def saving_fraction(self, scorer: Optional[Workload] = None) -> float:
        """Dimensionless saving relative to full-data training, including scoring cost.

        Equation:
            return (self.saving(scorer) / self.full_time).to('').magnitude"""
        return (self.saving(scorer) / self.full_time).to("").magnitude

    @property
    def rescoring_time(self) -> Quantity:
        """Minutes spent rescoring the whole pool before every epoch.

        The workload rescoring the pool before every epoch.

        Equation:
            return (self.epochs * self.score_time()).to('minute')"""
        return (self.epochs * self.score_time()).to("minute")

    @property
    def rescoring_saving_fraction(self) -> float:
        """Saving fraction after charging every-epoch rescoring and subset training.

        Equation:
            return ((self.full_time - (self.rescoring_time + self.subset_time)) / self.full_time).to('').magnitude"""
        return ((self.full_time - (self.rescoring_time + self.subset_time)) / self.full_time).to("").magnitude

    @property
    def rescoring_over_training(self) -> float:
        """Dimensionless repeated-scoring duration divided by subset-training duration.

        Equation:
            return (self.rescoring_time / self.subset_time).to('').magnitude"""
        return (self.rescoring_time / self.subset_time).to("").magnitude

    # One pruned epoch through the machine
    @property
    def pruned_epoch_time(self) -> Quantity:
        """Compute duration of one subset epoch, linear in retained share.

        Equation:
            return self.keep_fraction * self.epoch_time"""
        return self.keep_fraction * self.epoch_time

    @property
    def pruned_epoch_energy(self) -> Quantity:
        """Board-power energy of one subset epoch, linear in retained share.

        Equation:
            return self.keep_fraction * self.epoch_energy"""
        return self.keep_fraction * self.epoch_energy

    @property
    def run_saving_time(self) -> Quantity:
        """Hours avoided across the run before charging selection cost.

        Equation:
            return (self.epochs * (self.epoch_time - self.pruned_epoch_time)).to('hour')"""
        return (self.epochs * (self.epoch_time - self.pruned_epoch_time)).to("hour")

    @property
    def run_saving_energy(self) -> Quantity:
        """MJ avoided across the run before charging selection cost.

        Equation:
            return (self.epochs * (self.epoch_energy - self.pruned_epoch_energy)).to('MJ')"""
        return (self.epochs * (self.epoch_energy - self.pruned_epoch_energy)).to("MJ")

    @property
    def saving_over_score_energy(self) -> float:
        """Dimensionless run energy avoided divided by one scoring pass energy.

        Equation:
            return (self.run_saving_energy / self.score_energy()).to('').magnitude"""
        return (self.run_saving_energy / self.score_energy()).to("").magnitude

    @property
    def archive_bytes(self) -> Quantity:
        """Registered full training archive size in GB.

        Equation:
            return self.dataset.training_archive_size.to('GB')"""
        return self.dataset.training_archive_size.to("GB")

    @property
    def storage_time(self) -> Quantity:
        """Seconds for one full archive read at registered storage bandwidth.

        Equation:
            return (self.archive_bytes / self.hardware.storage.bandwidth).to('second')"""
        return (self.archive_bytes / self.hardware.storage.bandwidth).to("second")

    @property
    def materialized_bytes(self) -> Quantity:
        """GB of archive retained if the coreset is materialized in its own shards.

        The coreset written out as its own shards.

        Equation:
            return (self.keep_fraction * self.archive_bytes).to('GB')"""
        return (self.keep_fraction * self.archive_bytes).to("GB")

    @property
    def link_bytes(self) -> Quantity:
        """GB sent over the host-device link for one full epoch of decoded samples.

        Equation:
            return (self.examples * self.decoded_sample).to('GB')"""
        return (self.examples * self.decoded_sample).to("GB")

    @property
    def link_time(self) -> Quantity:
        """Seconds for the full decoded epoch to traverse the host-device link.

        Equation:
            return (self.link_bytes / self.hardware.interconnect.bandwidth).to('second')"""
        return (self.link_bytes / self.hardware.interconnect.bandwidth).to("second")

    @property
    def host_epoch_time(self) -> Quantity:
        """Seconds for the host to prepare one full epoch at its supplied sample rate.

        Equation:
            return Q_(self.examples / self.host_rate.to('count/second').magnitude, 'second')"""
        return Q_(self.examples / self.host_rate.to("count/second").magnitude, "second")


class DataEchoingScenario(BaseScenario):
    """A host that prepares ``host_rate`` samples a second feeding an accelerator that consumes
    ``accelerator_rate``, over ``epochs`` of ``examples``, with each prepared sample used
    ``echo_factor`` times.

    Notes
    -----
    Equation: ``echoed_run_time`` is the worked quantity; each property below states its equation and units.

    Assumptions: The pipeline is host-limited and an echoed sample needs no repeated host preparation.

    Excludes: Convergence impact, prefetch queues, and utilization clamping when host rate exceeds device rate.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50EchoingH100
    >>> value = case.echoed_run_time
    >>> print(f"{value:.3f~P}")
    11416.340 s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    accelerator_rate: Quantity
    host_rate: Quantity
    examples: Quantity  # count
    epochs: int
    echo_factor: float

    @property
    def ratio(self) -> float:
        """Dimensionless accelerator-to-host sample-rate ratio, the ideal useful echo limit.

        Accelerator rate over host rate; echoing helps up to this factor.

        Equation:
            return (self.accelerator_rate / self.host_rate).to('').magnitude"""
        return (self.accelerator_rate / self.host_rate).to("").magnitude

    @property
    def utilization(self) -> float:
        """Unclamped host-to-accelerator rate ratio; interpret as busy share only when host-limited.

        Accelerator busy share without echoing.

        Equation:
            return (self.host_rate / self.accelerator_rate).to('').magnitude"""
        return (self.host_rate / self.accelerator_rate).to("").magnitude

    @property
    def idle_fraction(self) -> float:
        """Unclamped idle fraction in the host-limited regime.

        Equation:
            return 1 - (self.host_rate / self.accelerator_rate).to('').magnitude"""
        return 1 - (self.host_rate / self.accelerator_rate).to("").magnitude

    @property
    def run_time(self) -> Quantity:
        """Time for all epoch samples at the host preparation rate.

        Equation:
            return self.epochs * self.examples / self.host_rate"""
        return self.epochs * self.examples / self.host_rate

    @property
    def echoed_rate(self) -> Quantity:
        """Effective echoed sample rate, capped by accelerator consumption capacity.

        Equation:
            return min(self.host_rate * self.echo_factor, self.accelerator_rate)"""
        return min(self.host_rate * self.echo_factor, self.accelerator_rate)

    @property
    def echoed_run_time(self) -> Quantity:
        """Time for all epoch samples at the capped echoed rate.

        Equation:
            return self.epochs * self.examples / self.echoed_rate"""
        return self.epochs * self.examples / self.echoed_rate


class SelectionBudgetScenario(BaseScenario):
    """Selection that takes ``selection_time`` and leaves a subset that trains in ``subset_time``,
    against full training in ``full_time`` (the selection inequality in hours).

    Notes
    -----
    Equation: ``total_time`` is the worked quantity; each property below states its equation and units.

    Assumptions: Selection and subset training run sequentially, with all compared durations supplied.

    Excludes: Accuracy equivalence, overlapping stages, and any price per unit time.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ExpensiveSelectionPitfall
    >>> value = case.total_time
    >>> print(f"{value:.3f~P}")
    12.000 h"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    selection_time: Quantity
    subset_time: Quantity
    full_time: Quantity
    pool: Optional[float] = None
    kept: Optional[float] = None

    @property
    def total_time(self) -> Quantity:
        """Combined selection and subset-training duration.

        Equation:
            return self.selection_time + self.subset_time"""
        return self.selection_time + self.subset_time

    @property
    def net_saving(self) -> Quantity:
        """Time saved against full training; negative means selection costs more.

        Equation:
            return self.full_time - self.total_time"""
        return self.full_time - self.total_time

    @property
    def pays(self) -> bool:
        """Whether selection plus subset training is faster than full training.

        Equation:
            return self.total_time < self.full_time"""
        return self.total_time < self.full_time

    @property
    def breakeven_selection(self) -> Quantity:
        """Selection duration at equality with full-training time; strict savings require less.

        Longest selection that still saves time.

        Equation:
            return self.full_time - self.subset_time"""
        return self.full_time - self.subset_time

    @property
    def overhead_ratio(self) -> float:
        """Dimensionless selection duration divided by subset-training duration.

        Selection time over subset training time.

        Equation:
            return (self.selection_time / self.subset_time).to('').magnitude"""
        return (self.selection_time / self.subset_time).to("").magnitude

    @property
    def selection_share(self) -> float:
        """Dimensionless selection duration divided by full-training duration.

        Selection time over full training time.

        Equation:
            return (self.selection_time / self.full_time).to('').magnitude"""
        return (self.selection_time / self.full_time).to("").magnitude


class CoverageFloorScenario(BaseScenario):
    """A ``keep_fraction`` coreset drawn without regard to class from ``pool`` examples, of which
    ``rare_share`` belong to a rare class that needs at least ``floor`` examples.

    Notes
    -----
    Equation: ``rare`` is the worked quantity; each property below states its equation and units.

    Assumptions: Selection retains each class in expectation at the same keep fraction.

    Excludes: Sampling variance, a probability guarantee for minimum coverage, and class-aware selection.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RareClassTenPercentCoreset
    >>> value = case.rare
    >>> round(value, 3)
    1000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pool: float
    rare_share: float
    keep_fraction: float
    floor: float

    @property
    def rare(self) -> float:
        """Expected rare-class examples in the full pool.

        Equation:
            return self.pool * self.rare_share"""
        return self.pool * self.rare_share

    @property
    def expected_kept(self) -> float:
        """Expected rare-class examples retained by class-agnostic selection.

        Equation:
            return self.rare * self.keep_fraction"""
        return self.rare * self.keep_fraction

    @property
    def shortfall(self) -> float:
        """Signed rare-class shortfall relative to the requested floor; negative means surplus.

        Equation:
            return self.floor - self.expected_kept"""
        return self.floor - self.expected_kept


class CurriculumScheduleScenario(BaseScenario):
    """A curriculum that reaches target in ``curriculum_epochs`` against ``baseline_epochs``
    shuffled, with an optional linear warmup of ``warmup_epochs`` (epoch j serves j/W of the data).

    Notes
    -----
    Equation: ``epoch_reduction`` is the worked quantity; each property below states its equation and units.

    Assumptions: Epoch counts are supplied and warmup epoch j uses j/W of a full dataset.

    Excludes: Predicting curriculum convergence, fractional-epoch scheduling, and data-selection overhead.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Cifar10LinearCurriculum
    >>> value = case.epoch_reduction
    >>> round(value, 3)
    0.233"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline_epochs: int
    curriculum_epochs: int
    warmup_epochs: Optional[int] = None

    @property
    def epoch_reduction(self) -> float:
        """Fractional epoch reduction relative to the shuffled baseline.

        Equation:
            return (self.baseline_epochs - self.curriculum_epochs) / self.baseline_epochs"""
        return (self.baseline_epochs - self.curriculum_epochs) / self.baseline_epochs

    @property
    def warmup_samples(self) -> float:
        """Equivalent full-data passes during a discrete linear warmup; requires warmup_epochs.

        Dataset passes the linear warmup serves: (W + 1) / 2.

        Equation:
            return (self.warmup_epochs + 1) / 2"""
        return (self.warmup_epochs + 1) / 2

    @property
    def warmup_share(self) -> float:
        """Dimensionless fraction of full-data warmup work actually served.

        Warmup samples over W full passes.

        Equation:
            return self.warmup_samples / self.warmup_epochs"""
        return self.warmup_samples / self.warmup_epochs


class ActiveLearningRoundsScenario(BaseScenario):
    """Active learning from ``initial_labels`` adding ``labels_per_round`` for ``rounds`` rounds,
    each round paying ``round_scoring_cost`` to score the pool, against random labeling that
    needs ``random_labels`` at the same ``label_price``.

    Notes
    -----
    Equation: ``active_labels`` is the worked quantity; each property below states its equation and units.

    Assumptions: Each round buys fixed labels and pays a fixed scoring bill; baseline label demand is supplied.

    Excludes: Quality prediction, turnaround time in the returned costs, and changing unit prices.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ActiveLearningTenRounds
    >>> value = case.active_labels
    >>> value
    2000"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    label_price: Quantity  # USD per label
    initial_labels: int
    labels_per_round: int
    rounds: int
    round_scoring_cost: Quantity  # USD per round
    random_labels: int
    label_turnaround: Optional[Quantity] = None  # time each round waits for its labels

    @property
    def active_labels(self) -> int:
        """Total labels acquired across the initial pool and subsequent rounds.

        Equation:
            return self.initial_labels + self.rounds * self.labels_per_round"""
        return self.initial_labels + self.rounds * self.labels_per_round

    @property
    def random_cost(self) -> Quantity:
        """Random-baseline labeling cost in USD.

        Equation:
            return (self.random_labels * self.label_price).to('USD')"""
        return (self.random_labels * self.label_price).to("USD")

    @property
    def active_label_cost(self) -> Quantity:
        """Active-learning labeling cost in USD, excluding scoring.

        Equation:
            return (self.active_labels * self.label_price).to('USD')"""
        return (self.active_labels * self.label_price).to("USD")

    @property
    def scoring_cost(self) -> Quantity:
        """Total pool-scoring cost in USD across all rounds.

        Equation:
            return (self.rounds * self.round_scoring_cost).to('USD')"""
        return (self.rounds * self.round_scoring_cost).to("USD")

    @property
    def active_cost(self) -> Quantity:
        """Combined active-learning labeling and scoring cost in USD.

        Equation:
            return self.active_label_cost + self.scoring_cost"""
        return self.active_label_cost + self.scoring_cost

    @property
    def roi(self) -> float:
        """Dimensionless net saving divided by active-learning cost.

        Saving over the active-learning cost.

        Equation:
            return ((self.random_cost - self.active_cost) / self.active_cost).to('').magnitude"""
        return ((self.random_cost - self.active_cost) / self.active_cost).to("").magnitude


class BudgetMatchedLabelingScenario(BaseScenario):
    """A ``pool`` to label at ``label_price`` under ``budget``: random labeling spends the budget,
    active learning reaches the same quality with ``active_labels``.

    Notes
    -----
    Equation: ``full_cost`` is the worked quantity; each property below states its equation and units.

    Assumptions: Labels have constant unit price and the active-label requirement is supplied.

    Excludes: Predicting matched quality, pool-size clamping, and deadline feasibility.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MedicalScanActiveLearning
    >>> value = case.full_cost
    >>> print(f"{value:.3f~P}")
    5000000.000 USD"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pool: Quantity  # count
    label_price: Quantity  # USD per count
    budget: Quantity  # USD
    active_labels: Quantity  # count
    deadline: Optional[Quantity] = None

    @property
    def full_cost(self) -> Quantity:
        """Cost of labeling the entire pool in the supplied currency units.

        Equation:
            return self.pool * self.label_price"""
        return self.pool * self.label_price

    @property
    def random_labels(self) -> Quantity:
        """Label count affordable at the supplied budget and unit price.

        Equation:
            return (self.budget / self.label_price).to('count')"""
        return (self.budget / self.label_price).to("count")

    @property
    def active_cost(self) -> Quantity:
        """Cost of the supplied active-learning label count.

        Equation:
            return self.active_labels * self.label_price"""
        return self.active_labels * self.label_price

    @property
    def unspent_share(self):
        """Dimensionless quantity for the unspent budget share; negative means overspend.

        Equation:
            return (self.budget - self.active_cost) / self.budget"""
        return (self.budget - self.active_cost) / self.budget

    @property
    def label_ratio(self) -> float:
        """Dimensionless ratio of affordable random labels to active labels.

        Equation:
            return (self.random_labels / self.active_labels).to('').magnitude"""
        return (self.random_labels / self.active_labels).to("").magnitude

    @property
    def saving(self) -> Quantity:
        """Budget minus active-labeling cost in currency units.

        Equation:
            return self.budget - self.active_cost"""
        return self.budget - self.active_cost

    @property
    def over_budget_ratio(self) -> float:
        """Dimensionless full-pool labeling cost divided by budget.

        Equation:
            return (self.full_cost / self.budget).to('').magnitude"""
        return (self.full_cost / self.budget).to("").magnitude


class SemiSupervisedCostScenario(BaseScenario):
    """A supervised baseline with ``supervised_labels`` and ``supervised_compute`` against a
    semi-supervised run with ``ssl_labels`` and ``ssl_compute``, labels at ``label_price``.

    Notes
    -----
    Equation: ``supervised_label_cost`` is the worked quantity; each property below states its equation and units.

    Assumptions: Both methods have supplied label counts and compute bills at a common label price.

    Excludes: Predicting equal quality, extra data acquisition, and campaign variance.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FixMatchCifar10Cost
    >>> value = case.supervised_label_cost
    >>> print(f"{value:.3f~P}")
    4000.000 USD"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    label_price: Quantity  # USD per label
    supervised_labels: int
    supervised_compute: Quantity  # USD
    ssl_labels: int
    ssl_compute: Quantity  # USD

    @property
    def supervised_label_cost(self) -> Quantity:
        """Supervised labeling cost in USD.

        Equation:
            return (self.supervised_labels * self.label_price).to('USD')"""
        return (self.supervised_labels * self.label_price).to("USD")

    @property
    def supervised_total(self) -> Quantity:
        """Supervised labeling plus compute cost in currency units.

        Equation:
            return self.supervised_label_cost + self.supervised_compute"""
        return self.supervised_label_cost + self.supervised_compute

    @property
    def ssl_label_cost(self) -> Quantity:
        """Semi-supervised labeling cost in USD.

        Equation:
            return (self.ssl_labels * self.label_price).to('USD')"""
        return (self.ssl_labels * self.label_price).to("USD")

    @property
    def ssl_total(self) -> Quantity:
        """Semi-supervised labeling plus compute cost in currency units.

        Equation:
            return self.ssl_label_cost + self.ssl_compute"""
        return self.ssl_label_cost + self.ssl_compute

    @property
    def compute_multiple(self) -> float:
        """Dimensionless semi-supervised-to-supervised compute-cost ratio.

        Equation:
            return (self.ssl_compute / self.supervised_compute).to('').magnitude"""
        return (self.ssl_compute / self.supervised_compute).to("").magnitude

    @property
    def cost_reduction(self) -> float:
        """Dimensionless supervised-to-semi-supervised total-cost ratio, not a fractional cut.

        Equation:
            return (self.supervised_total / self.ssl_total).to('').magnitude"""
        return (self.supervised_total / self.ssl_total).to("").magnitude

    @property
    def label_reduction(self) -> float:
        """Dimensionless supervised-to-semi-supervised label-count ratio.

        Equation:
            return self.supervised_labels / self.ssl_labels"""
        return self.supervised_labels / self.ssl_labels


class PretrainingAmortizationScenario(BaseScenario):
    """``tasks`` tasks trained from scratch at ``scratch_compute`` each against one pretraining run
    of ``pretrain_compute`` plus ``finetune_compute`` per task, with the labels each path needs.

    Notes
    -----
    Equation: ``scratch_total`` is the worked quantity; each property below states its equation and units.

    Assumptions: Pretraining is paid once, per-task costs are constant, and scratch compute exceeds fine-tuning compute for crossover.

    Excludes: Quality differences, task heterogeneity, and a monetary conversion of accelerator time.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FoundationModelTenTasks
    >>> value = case.scratch_total
    >>> print(f"{value:.3f~P}")
    10000.000 h"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pretrain_compute: Quantity  # accelerator-hours
    scratch_compute: Quantity  # accelerator-hours per task
    finetune_compute: Quantity  # accelerator-hours per task
    tasks: int
    scratch_labels: Quantity  # count per task
    finetune_labels: Quantity  # count per task
    label_price: Quantity  # USD per count

    @property
    def scratch_total(self) -> Quantity:
        """Total accelerator time when every task trains from scratch.

        Equation:
            return self.scratch_compute * self.tasks"""
        return self.scratch_compute * self.tasks

    @property
    def foundation_total(self) -> Quantity:
        """One pretraining run plus per-task fine-tuning accelerator time.

        Equation:
            return self.pretrain_compute + self.finetune_compute * self.tasks"""
        return self.pretrain_compute + (self.finetune_compute * self.tasks)

    @property
    def scratch_label_cost(self) -> Quantity:
        """Label cost for scratch training across every task.

        Equation:
            return self.scratch_labels * self.label_price * self.tasks"""
        return self.scratch_labels * self.label_price * self.tasks

    @property
    def finetune_label_cost(self) -> Quantity:
        """Label cost for fine-tuning across every task.

        Equation:
            return self.finetune_labels * self.label_price * self.tasks"""
        return self.finetune_labels * self.label_price * self.tasks

    @property
    def label_cost_reduction(self):
        """Dimensionless quantity comparing scratch-label cost with fine-tuning-label cost.

        Equation:
            return self.scratch_label_cost / self.finetune_label_cost"""
        return self.scratch_label_cost / self.finetune_label_cost

    @property
    def marginal_compute_reduction(self) -> float:
        """Dimensionless per-task scratch-to-fine-tuning compute ratio.

        Equation:
            return (self.scratch_compute / self.finetune_compute).to('').magnitude"""
        return (self.scratch_compute / self.finetune_compute).to("").magnitude

    @property
    def crossover_tasks(self) -> float:
        """Continuous task count at compute break-even; requires scratch cost above fine-tuning cost.

        Tasks after which pretraining has paid for itself in compute.

        Equation:
            return (self.pretrain_compute / (self.scratch_compute - self.finetune_compute)).to('').magnitude"""
        return (self.pretrain_compute / (self.scratch_compute - self.finetune_compute)).to("").magnitude


class DataCostBreakdownScenario(BaseScenario):
    """A supervised dataset's bill: ``acquisition``, ``labels`` at ``label_price``,
    ``storage_bytes`` kept for ``storage_duration`` at ``storage_rate``, and ``runs`` training
    runs of ``run_time`` each on ``accelerators`` at ``accelerator_rate``.

    Notes
    -----
    Equation: ``label_cost`` is the worked quantity; each property below states its equation and units.

    Assumptions: Unit prices, storage occupancy, and run durations remain constant throughout the campaign.

    Excludes: Discounts, idle capacity, finance charges, and overlapping costs not listed in the scenario.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ImageNetDataCostBreakdown
    >>> value = case.label_cost
    >>> print(f"{value:.3f~P}")
    64058.350 dollar"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    acquisition: Quantity  # USD
    labels: Quantity  # count
    label_price: Quantity  # USD per count
    storage_bytes: Quantity
    storage_rate: Quantity  # USD per GB per month
    storage_duration: Quantity
    run_time: Quantity
    runs: int
    accelerators: int
    accelerator_rate: Quantity  # USD per accelerator-hour

    @property
    def label_cost(self) -> Quantity:
        """Label-acquisition cost, label count times unit label price.

        Equation:
            return self.labels * self.label_price"""
        return self.labels * self.label_price

    @property
    def storage_cost(self) -> Quantity:
        """Storage cost in USD, stored bytes times duration times unit storage price.

        Equation:
            return (self.storage_bytes * self.storage_rate * self.storage_duration).to('USD')"""
        return (self.storage_bytes * self.storage_rate * self.storage_duration).to("USD")

    @property
    def accelerator_hours_per_run(self) -> Quantity:
        """Accelerator-hours consumed by one run, device count times duration.

        Equation:
            return self.accelerators * self.run_time"""
        return self.accelerators * self.run_time

    @property
    def accelerator_hours(self) -> Quantity:
        """Accelerator-hours consumed by the full run campaign.

        Equation:
            return self.accelerator_hours_per_run * self.runs"""
        return self.accelerator_hours_per_run * self.runs

    @property
    def compute_cost(self) -> Quantity:
        """Campaign compute cost in USD at the supplied accelerator-hour price.

        Equation:
            return (self.accelerator_hours * self.accelerator_rate).to('USD')"""
        return (self.accelerator_hours * self.accelerator_rate).to("USD")

    @property
    def total(self) -> Quantity:
        """Total acquisition, labeling, storage, and compute cost.

        Equation:
            return self.acquisition + self.label_cost + self.storage_cost + self.compute_cost"""
        return self.acquisition + self.label_cost + self.storage_cost + self.compute_cost

    @property
    def data_share(self):
        """Dimensionless quantity for the data-related share of total cost.

        Equation:
            return (self.acquisition + self.label_cost + self.storage_cost) / self.total"""
        return (self.acquisition + self.label_cost + self.storage_cost) / self.total

    @property
    def compute_share(self):
        """Dimensionless quantity for the compute share of total cost.

        Equation:
            return self.compute_cost / self.total"""
        return self.compute_cost / self.total


class RunAmortizationScenario(BaseScenario):
    """A one-time investment of ``build_cost`` plus ``operating_cost`` that saves ``saving_per_run``
    on every training run that reuses its output.

    Notes
    -----
    Equation: ``investment`` is the worked quantity; each property below states its equation and units.

    Assumptions: Investment is paid once and each reuse yields the same positive saving.

    Excludes: Discounting, depreciation, and changes in savings across runs.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MinHashPipelineAmortization
    >>> value = case.investment
    >>> print(f"{value:.3f~P}")
    55000.000 USD"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    build_cost: Quantity  # USD
    operating_cost: Quantity  # USD, one-time
    saving_per_run: Quantity  # USD per run
    run_counts: List[int] = []  # campaign sizes at which to read the return

    @property
    def investment(self) -> Quantity:
        """Initial investment in currency units, build plus one-time operating cost.

        Equation:
            return self.build_cost + self.operating_cost"""
        return self.build_cost + self.operating_cost

    def net_return(self, runs: int) -> float:
        """Net return fraction after the supplied run count; savings minus investment over investment.

        (runs x saving - investment) / investment.

        Equation:
            return ((runs * self.saving_per_run - self.investment) / self.investment).to('').magnitude"""
        return ((runs * self.saving_per_run - self.investment) / self.investment).to("").magnitude

    def savings_multiple(self, runs: int) -> float:
        """Dimensionless gross savings divided by initial investment.

        Total savings over the investment.

        Equation:
            return (runs * self.saving_per_run / self.investment).to('').magnitude"""
        return (runs * self.saving_per_run / self.investment).to("").magnitude

    @property
    def breakeven_runs(self) -> float:
        """Continuous number of runs needed to recover the investment.

        Equation:
            return (self.investment / self.saving_per_run).to('').magnitude"""
        return (self.investment / self.saving_per_run).to("").magnitude


class DataScalingReturnScenario(BaseScenario):
    """Growing a dataset from ``base_examples`` to ``scaled_examples`` for ``accuracy_gain`` points.

    Notes
    -----
    Equation: ``scale`` is the worked quantity; each property below states its equation and units.

    Assumptions: Dataset counts and accuracy gain are supplied observations with a positive baseline count.

    Excludes: A law predicting accuracy from dataset size.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TenfoldDataFourPoints
    >>> value = case.scale
    >>> round(value, 3)
    10.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    base_examples: Quantity  # count
    scaled_examples: Quantity  # count
    accuracy_gain: float  # percentage points

    @property
    def scale(self) -> float:
        """Dimensionless dataset-size multiple; supplied accuracy gain is descriptive evidence.

        Equation:
            return (self.scaled_examples / self.base_examples).to('').magnitude"""
        return (self.scaled_examples / self.base_examples).to("").magnitude


class FailureTriageScenario(BaseScenario):
    """``runs`` trajectories a month failing at ``failure_rate``, each probe's attributed share, and a
    fine-tuning run on every failure of ``workload`` on ``devices`` x ``hardware``.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.FailureTriageModel.solve``.

    Assumptions: Inputs are routed to ``FailureTriageModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to FailureTriageModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairAgentFailureTriage
    >>> value = case.result.cost
    >>> print(f"{value:.3g~}")
    58.5 dollar"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    runs: int
    failure_rate: float
    probe_shares: Dict[str, float]  # failures each counterfactual probe attributes to its rung
    workload: Workload
    hardware: HardwareNode
    devices: int
    utilization: float  # model FLOPs utilization
    hourly_price: Quantity  # USD per device-hour
    sequence_tokens: int
    epochs: int

    @property
    def result(self):
        """Evaluate the stated failure triage case through ``FailureTriageModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.FailureTriageModel.solve``."""
        from ..solvers import FailureTriageModel

        return FailureTriageModel().solve(
            self.runs,
            self.failure_rate,
            self.probe_shares,
            self.workload.parameters,
            self.sequence_tokens,
            self.epochs,
            self.devices,
            self.hardware.compute.peak_flops,
            self.utilization,
            self.hourly_price,
        )


class RunSplitLeakageScenario(BaseScenario):
    """``attempts`` runs per task, each assigned to training with probability ``train_fraction``.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.RunSplitLeakageModel.solve``.

    Assumptions: Inputs are routed to ``RunSplitLeakageModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to RunSplitLeakageModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.EightAttemptRunSplit
    >>> value = case.result.straddle
    >>> print(f"{value:.3g}")
    0.832"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    attempts: int
    train_fraction: float
    tasks: Optional[int] = None

    @property
    def result(self):
        """Evaluate the stated run split leakage case through ``RunSplitLeakageModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.RunSplitLeakageModel.solve``."""
        from ..solvers import RunSplitLeakageModel

        return RunSplitLeakageModel().solve(self.attempts, self.train_fraction, self.tasks)


class AdmissionOrderScenario(BaseScenario):
    """Independent admission checks (pass probability, slot time) run in each of ``orders``.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.AdmissionOrderModel.solve``.

    Assumptions: Inputs are routed to ``AdmissionOrderModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to AdmissionOrderModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DemonstrationAdmissionOrder
    >>> value = case.result.saving_per_attempted
    >>> print(f"{value:.3g~}")
    1.27 s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    checks: Dict[str, Tuple[float, Quantity]]
    orders: List[List[str]]
    candidates: Optional[int] = None

    @property
    def result(self):
        """Evaluate the stated admission order case through ``AdmissionOrderModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.AdmissionOrderModel.solve``."""
        from ..solvers import AdmissionOrderModel

        return AdmissionOrderModel().solve(self.checks, self.orders, self.candidates)


class CorpusDenominatorsScenario(BaseScenario):
    """A retained pool of record classes (records, stored tokens per record, selected target tokens per
    record); ``focus`` is the class whose shares are read, ``admitted`` the classes a route admits.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.CorpusDenominatorsModel.solve``.

    Assumptions: Inputs are routed to ``CorpusDenominatorsModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to CorpusDenominatorsModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RecoveryPoolDenominators
    >>> value = case.result.target_share
    >>> print(f"{value:.3g}")
    0.391"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    classes: Dict[str, Tuple[int, Quantity, Quantity]]
    focus: str
    admitted: List[str]

    @property
    def result(self):
        """Evaluate the stated corpus denominators case through ``CorpusDenominatorsModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.CorpusDenominatorsModel.solve``."""
        from ..solvers import CorpusDenominatorsModel

        return CorpusDenominatorsModel().solve(self.classes, self.focus, self.admitted)


class AttemptGroupContrastScenario(BaseScenario):
    """``attempts`` independent attempts per fixture at each of ``pass_rates``.

    Notes
    -----
    Equation: The calculation members below define ``success_yield``, ``mixed_fraction``, ``attempts_per_success``.

    Assumptions: Attempts have identical independent pass probability; each fixture consumes the full attempt count.

    Excludes: Correlated attempts, early stopping, and unequal attempt costs.

    References
    ----------
    Derivation: ``attempts`` independent attempts per fixture at each of ``pass_rates``. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.EightAttemptGroupContrast
    >>> value = case.success_yield(case.pass_rates[0])
    >>> print(f"{value:.3g}")
    0.337"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    attempts: int
    pass_rates: List[float]

    def success_yield(self, pass_rate: float) -> float:
        """Dimensionless probability of at least one pass, via ``calc_independent_success``.

        Chance of at least one success.

        Calculation: ``mlsysim.physics.agents.calc_independent_success``."""
        from ...physics.agents import calc_independent_success

        return calc_independent_success(pass_rate, self.attempts)

    def mixed_fraction(self, pass_rate: float) -> float:
        """Dimensionless probability of both pass and failure, via ``calc_mixed_outcome_fraction``.

        Chance of at least one success and one failure.

        Calculation: ``mlsysim.physics.agents.calc_mixed_outcome_fraction``."""
        from ...physics.agents import calc_mixed_outcome_fraction

        return calc_mixed_outcome_fraction(pass_rate, self.attempts)

    def attempts_per_success(self, pass_rate: float) -> float:
        """Expected attempts spent per fixture yielding any success; requires positive success yield.

        Attempts spent per fixture that yields a success.

        Equation: ``attempts / success_yield(pass_rate)``."""
        return self.attempts / self.success_yield(pass_rate)


class StagingBurstScenario(BaseScenario):
    """A collector of ``rollouts`` concurrent rollouts feeding ``slots`` verifiers, whose checks
    stall at ``timeout``, with records charged at ``record_ceiling`` against ``store``.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.StagingBurstModel.solve``.

    Assumptions: Inputs are routed to ``StagingBurstModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to StagingBurstModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.VerifierTimeoutStagingBurst
    >>> value = case.result.arrival_rate
    >>> print(f"{value:.3g~}")
    22.8 / s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rollouts: int
    rollout_time: Quantity
    slots: int
    mean_service: Quantity
    timeout: Quantity
    record_ceiling: Quantity
    store: Quantity
    reaction: Quantity

    @property
    def result(self):
        """Evaluate the stated staging burst case through ``StagingBurstModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.StagingBurstModel.solve``."""
        from ..solvers import StagingBurstModel

        return StagingBurstModel().solve(
            self.rollouts,
            self.rollout_time,
            self.slots,
            self.mean_service,
            self.timeout,
            self.record_ceiling,
            self.store,
            self.reaction,
        )


class CohortDispositionScenario(BaseScenario):
    """A finite cohort of attempts by disposition at a cutoff, its total cost, and the disposition
    that emits a learning artifact.

    Notes
    -----
    Equation: The calculation members below define ``attempts``, ``emission_yield``, ``cost_per_attempt``, ``cost_per_artifact``.

    Assumptions: Dispositions partition a nonempty cohort; artifact yield is positive for finite artifact cost.

    Excludes: Overlapping categories, time-varying costs, and causal claims about a yield intervention.

    References
    ----------
    Derivation: A finite cohort of attempts by disposition at a cutoff, its total cost, and the disposition that emits a learning artifact. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CurationCohortDispositions
    >>> value = case.cost_per_artifact()
    >>> print(f"{value:.3g~}")
    20 USD"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    dispositions: Dict[str, int]
    emitted: str
    cohort_cost: Quantity
    yield_multiple: Optional[float] = None  # a raised emission yield to compare, as a multiple

    @property
    def attempts(self) -> int:
        """Total attempt count across all disposition categories.

        Equation: ``sum(dispositions.values())``."""
        return sum(self.dispositions.values())

    @property
    def emission_yield(self) -> float:
        """Dimensionless fraction of attempts in the artifact-emitting disposition.

        Equation: ``dispositions[emitted] / attempts``."""
        return self.dispositions[self.emitted] / self.attempts

    @property
    def cost_per_attempt(self) -> Quantity:
        """Cohort cost divided by total attempts, in USD per attempt.

        Equation: ``(cohort_cost / attempts).to('USD')``."""
        return (self.cohort_cost / self.attempts).to("USD")

    def cost_per_artifact(self, emission_yield: Optional[float] = None) -> Quantity:
        """USD per emitted artifact from ``calc_cost_per_accepted``; an override changes yield only.

        Attempt cost over emission yield; another yield prices the same attempts at that yield.

        Calculation: ``mlsysim.physics.agents.calc_cost_per_accepted``."""
        from ...physics.agents import calc_cost_per_accepted

        y = self.emission_yield if emission_yield is None else emission_yield
        return calc_cost_per_accepted(self.cost_per_attempt, y).to("USD")
