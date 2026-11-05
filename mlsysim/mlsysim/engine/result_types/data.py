"""Typed result models for data calculations."""

from __future__ import annotations

from typing import Dict, Optional, List
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "DataResult",
    "TransformationResult",
    "StorageFootprintResult",
    "TieredCheckpointResult",
    "CheckpointStormResult",
    "InputDemandResult",
    "DataStallResult",
    "PrefetchDepthResult",
    "TieredCacheResult",
    "HostIoCpuResult",
    "ObjectStorageBillResult",
    "StreamVsStageResult",
    "UtilizationLossResult",
    "SyntheticFootprintResult",
    "FailureTriageResult",
    "RunSplitLeakageResult",
    "AdmissionOrderResult",
    "CorpusDenominatorsResult",
    "StagingBurstResult",
]


class DataResult(SolverResult):
    """Result from DataModel: data pipeline feasibility."""

    __module__ = "mlsysim.engine.results"

    is_stalled: bool
    utilization: float
    demand_bw: Quantity
    supply_bw: Quantity
    bottleneck: str
    margin: Quantity


class TransformationResult(SolverResult):
    """Result from TransformationModel: CPU preprocessing bottleneck."""

    __module__ = "mlsysim.engine.results"

    transform_time: Quantity
    accelerator_step_time: Quantity
    is_bottleneck: bool
    accelerator_utilization: float
    slowdown_factor: float


class StorageFootprintResult(SolverResult):
    """Result from StorageFootprintModel: what a training run reads, saves, and keeps."""

    __module__ = "mlsysim.engine.results"

    checkpoint_bytes: Quantity
    saves: int
    checkpoint_bytes_written: Quantity
    write_read_ratio: float  # checkpoint bytes written / corpus bytes read
    average_read_rate: Quantity  # corpus bytes / run duration
    shard_count: int
    archived_checkpoints: int
    archive_bytes: Quantity  # retained checkpoints + source + corpus


class TieredCheckpointResult(SolverResult):
    """Result from TieredCheckpointModel: a save staged on local storage, then drained."""

    __module__ = "mlsysim.engine.results"

    checkpoint_bytes: Quantity
    per_node_shard: Quantity
    local_write_time: Quantity  # the training pause
    drain_time: Quantity  # background copy to the durable tier
    durable_share_per_node: Quantity
    pause_fraction: float  # pause / save interval


class CheckpointStormResult(SolverResult):
    """Result from CheckpointStormModel: every replica writes against one logical copy."""

    __module__ = "mlsysim.engine.results"

    state_bytes: Quantity
    storm_bytes: Quantity
    storm_time: Quantity
    single_copy_time: Quantity
    slowdown: float


class InputDemandResult(SolverResult):
    """Result from InputDemandModel: bandwidth a fleet of consumers draws from storage."""

    __module__ = "mlsysim.engine.results"

    batch_bytes: Quantity  # one consumer's batch
    bandwidth: Quantity


class DataStallResult(SolverResult):
    """Result from DataStallModel: input wait with and without overlap."""

    __module__ = "mlsysim.engine.results"

    sequential_step: Quantity
    sequential_stall: float
    overlapped_step: Quantity
    overlapped_stall: float


class PrefetchDepthResult(SolverResult):
    """Result from PrefetchDepthModel: ready batches needed to cover each tier's tail."""

    __module__ = "mlsysim.engine.results"

    spans: Dict[str, float]  # tail latency / compute window
    depths: Dict[str, int]  # ceil(span)
    buffer_per_consumer: Quantity  # at the stated safety depth
    buffer_per_node: Quantity


class TieredCacheResult(SolverResult):
    """Result from TieredCacheModel: mean service of a two-tier cache."""

    __module__ = "mlsysim.engine.results"

    hit_rate: float
    miss_penalty: float  # miss service time / hit service time
    mean_service: float  # in hit service times
    delivered_fraction: float  # of the all-hit rate
    delivered_rate: Optional[Quantity] = None


class HostIoCpuResult(SolverResult):
    """Result from HostIoCpuModel: CPU time per second spent issuing I/O."""

    __module__ = "mlsysim.engine.results"

    request_rate: float  # requests per second
    cpu_seconds_per_second: Dict[str, float]


class ObjectStorageBillResult(SolverResult):
    """Result from ObjectStorageBillModel: storing and reading a dataset in object storage."""

    __module__ = "mlsysim.engine.results"

    storage_cost: float  # dollars over the billing period
    egress_per_read: float  # dollars
    egress_cost: float  # dollars, all reads
    request_cost: float  # dollars, one GET per file
    archive_retrieval_cost: float  # dollars, retrieval fee only
    archive_restore_cost: float  # dollars, retrieval plus one egress


class StreamVsStageResult(SolverResult):
    """Result from StreamVsStageModel: reading every epoch remotely against staging per run."""

    __module__ = "mlsysim.engine.results"

    source_storage_cost: float  # dollars a year, both options
    stream_egress_per_run: float
    stream_egress_cost: float
    stream_cost: float
    local_storage_cost: float
    stage_egress_cost: float
    stage_cost: float
    savings: float
    break_even_epochs: float


class UtilizationLossResult(SolverResult):
    """Result from UtilizationLossModel: the cost of devices idled by a utilization drop."""

    __module__ = "mlsysim.engine.results"

    fleet_cost_per_day: float  # dollars
    loss_per_day: float
    loss_total: float


class SyntheticFootprintResult(SolverResult):
    """Result from SyntheticFootprintModel: generated data at rest and in motion."""

    __module__ = "mlsysim.engine.results"

    stored_bytes: Quantity
    verification_bytes: Quantity
    pipeline_bytes: Quantity
    capacity_amplification: float
    traffic_amplification: float


class FailureTriageResult(SolverResult):
    """Result from FailureTriageModel: failures repaired below the weights and the price of training on them."""

    __module__ = "mlsysim.engine.results"

    failures: int
    removed: Dict[str, int]  # failures each probe attributes to its rung
    system_failures: int  # removed by every probe together
    residual: int  # survive every probe
    system_share: float
    training_tokens: Quantity  # every failure, all epochs
    residual_tokens: Quantity
    training_flops: Quantity
    residual_flops: Quantity
    sustained_rate: Quantity  # devices x peak x utilization
    wall_time: Quantity  # training on every failure
    device_hours: Quantity
    cost: Quantity  # training on every failure
    residual_cost: Quantity  # training on the residual alone


class RunSplitLeakageResult(SolverResult):
    """Result from RunSplitLeakageModel: tasks a random split of runs places on both sides."""

    __module__ = "mlsysim.engine.results"

    whole: float  # f^k + (1 - f)^k
    straddle: float
    straddling_tasks: Optional[float]  # straddle x tasks, when tasks are given
    sibling_in_training: float  # a test run has a sibling run in training: 1 - (1 - f)^(k-1)
    majority_attempts: int  # smallest attempt count at which most tasks straddle
    majority_straddle: float  # straddle share at that count
    curve: Dict[int, float]  # attempts -> straddle share, 1 to max_attempts


class AdmissionOrderResult(SolverResult):
    """Result from AdmissionOrderModel: one admission route's checks run in several orders."""

    __module__ = "mlsysim.engine.results"

    cost_per_rejection: Dict[str, Quantity]  # t / (1 - p) per check
    time_per_attempted: List[Quantity]  # expected slot time per attempted record, per order
    time_per_admitted: List[Quantity]
    route_yield: float  # the same for every order
    admitted: Optional[float]  # route yield x candidates, when candidates are given
    best_order: int  # index of the cheapest order
    saving: float  # 1 - cheapest / dearest time per attempted record
    saving_per_attempted: Quantity  # dearest - cheapest


class CorpusDenominatorsResult(SolverResult):
    """Result from CorpusDenominatorsModel: one record class's share of a pool under three denominators."""

    __module__ = "mlsysim.engine.results"

    records: int
    stored_tokens: Quantity
    admitted_stored_tokens: Quantity  # stored tokens of the admitted classes
    target_tokens: Quantity
    record_share: float
    stored_share: float
    admitted_stored_share: float
    target_share: float
    context_per_target: Dict[str, float]  # stored tokens per target token, per class with targets


class StagingBurstResult(SolverResult):
    """Result from StagingBurstModel: how fast a verifier timeout burst fills reserved staging."""

    __module__ = "mlsysim.engine.results"

    arrival_rate: Quantity  # records per second from the rollouts
    capacity: Quantity  # nominal verifier service rate
    utilization: float
    burst_capacity: Quantity  # slots / timeout
    net_growth: Quantity  # unassessed records added per second
    growth: Quantity  # bytes reserved per second at the ceiling
    time_to_full: Quantity
    in_flight: Quantity  # rollouts already running, at the ceiling
    reaction_growth: Quantity  # growth during the controller's delay
    high_watermark: float  # largest share of the store at which dispatch may stop
