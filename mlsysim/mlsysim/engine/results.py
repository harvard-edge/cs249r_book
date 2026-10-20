"""Typed result models for all mlsysim models and solvers.

Layer A of the composition architecture: every resolver returns a typed
Pydantic model instead of Dict[str, Any].  This gives students
autocomplete, documentation, and type safety when composing analytical 
models and analysis solvers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field
from ..core.types import Quantity

# Explicit public surface: result classes only. Without this, the
# `from .results import *` in engine/__init__.py would also leak the imported
# helper names (Any, Dict, BaseModel, Quantity, ...) into mlsysim.engine.
__all__ = [
    "SolverResult",
    # Supply layer
    "DistributedResult",
    "ReliabilityResult",
    "CheckpointResult",
    "SustainabilityResult",
    "ServingResult",
    "TrainingMemoryResult",
    "ServingCapacityResult",
    "MoERoutingResult",
    "ContinuousBatchingResult",
    "WeightStreamingResult",
    "TailLatencyResult",
    "EconomicsResult",
    "DataResult",
    "OffloadResult",
    "TopologyResult",
    "EfficiencyResult",
    "TransformationResult",
    # Demand layer
    "ScalingResult",
    "CompressionResult",
    "CompressionCandidate",
    "CompressionSweepResult",
    "SynthesisResult",
    # Consequence layer
    "OrchestrationResult",
    "InferenceScalingResult",
    "SensitivityResult",
    "ResponsibleEngineeringResult",
    # Search & optimization
    "OptimizerResult",
    "ParallelismOptimizerResult",
    "BatchingOptimizerResult",
    "PlacementOptimizerResult",
    # Situation models (2026-10-20): one solver per recurring calculation that
    # chapters and lecture slides both quote, so the arithmetic has one home.
    "DecodeBoundResult",
    "TransferLadderResult",
    "TensorParallelCommResult",
    "TrainingRunResult",
    "TrainingStateResult",
    "PowerTransientResult",
    "CoolantFlowResult",
    "OutageCostResult",
    "FailureIntervalResult",
    "GradientSyncResult",
    "CommunicationIntensityResult",
    "TrainingGrowthResult",
    "AlphaBetaResult",
    "BisectionResult",
    "IncastResult",
    "CollectiveFitResult",
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
    "StorageBenchmarkResult",
    # Agent trajectories (added 2026-10-20)
    "HorizonCompoundingResult",
    "AcceptedTaskCostResult",
    "AppendOnlyContextResult",
    "TaskHorizonTrendResult",
    "AgentRunSummaryResult",
    # Model calls and agent trajectories (added 2026-10-20)
    "ContextAdmissionResult",
    "SamplingResult",
    "CheckerPosteriorResult",
    "ModelCallResult",
    "TrajectoryResendResult",
    "TokenMixResult",
    # Distributed training (added 2026-10-20)
    "ParallelLayoutResult",
    "ZeroShardingResult",
    "DataParallelScaleOutResult",
    "TensorParallelLayerResult",
    "AlignmentMemoryResult",
    "ScalingSeriesResult",
    "MemoryWallResult",
    # Single-machine iron law, Amdahl, and silent-degradation results (added 2026-10-20)
    "IronLawResult",
    "AmdahlResult",
    "ShiftedTestSetResult",
    "ProxyDriftResult",
    # physical-plant budgets (added 2026-10-20)
    "StoppingBudgetResult",
    "SpeedStepResult",
    "FreshnessProfileResult",
    "BrakingCeilingsResult",
    "HubDriveElectricalResult",
    "GearedRotorResult",
    "GearRatioCostResult",
    "ThermalBurstResult",
    "StallDeratingResult",
    "ContinuousDeratingResult",
    "DriveBusResult",
    "ZeroFailureResult",
    "ThrottledStageResult",
]


class SolverResult(BaseModel):
    """Base class for all model and solver results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    constraint_trace: Optional[List[str]] = Field(
        default=None,
        description="A trace of physical constraints evaluated during the solve. If feasible=False, this explains why."
    )


# ── Supply Layer Results ──────────────────────────────────────────

class DistributedResult(SolverResult):
    """Result from DistributedModel: fleet-wide training performance."""
    node_profile: Any  # PerformanceProfile (avoid circular import)
    dp_communication_latency: Quantity
    tp_communication_latency: Quantity
    ep_communication_latency: Quantity
    communication_latency: Quantity  # total across all parallelism dimensions
    pipeline_bubble_latency: Quantity
    bubble_fraction: float
    step_latency_total: Quantity
    scaling_efficiency: float
    effective_throughput: Quantity
    parallelism: Dict[str, int]


class ReliabilityResult(SolverResult):
    """Result from ReliabilityModel: failure modeling and checkpointing."""
    fleet_mtbf: Quantity
    failure_probability: float
    optimal_checkpoint_interval: Quantity
    expected_failures: float
    goodput_ratio: float = 1.0


class CheckpointResult(SolverResult):
    """Result from CheckpointModel: IOPS and MFU penalty modeling."""
    checkpoint_size: Quantity
    write_time_seconds: Quantity
    max_bandwidth_required: Quantity
    storage_bottleneck: bool
    mfu_penalty_pct: float

    @property
    def write_time(self) -> Quantity:
        """Checkpoint write duration as a Pint Quantity."""
        return self.write_time_seconds


class SustainabilityResult(SolverResult):
    """Result from SustainabilityModel: energy, carbon, and water footprint."""
    it_energy_kwh: Quantity
    total_energy_kwh: Quantity
    carbon_footprint_kg: float
    water_usage_liters: float
    pue: float
    region_name: str
    embodied_carbon_kg: float = 0.0

    @property
    def it_energy(self) -> Quantity:
        """IT energy consumption as a Pint Quantity."""
        return self.it_energy_kwh

    @property
    def total_energy(self) -> Quantity:
        """Total facility energy consumption (post-PUE) as a Pint Quantity."""
        return self.total_energy_kwh

    @property
    def carbon_footprint(self) -> Quantity:
        """Total carbon footprint (operational + embodied) as a Pint Quantity in kg."""
        from ..core.units import Q_
        return Q_(self.carbon_footprint_kg, "kg")

    @property
    def water_usage(self) -> Quantity:
        """Total cooling water consumption as a Pint Quantity in liters."""
        from ..core.units import Q_
        return Q_(self.water_usage_liters, "liter")

    @property
    def embodied_carbon(self) -> Quantity:
        """Embodied manufacturing carbon as a Pint Quantity in kg."""
        from ..core.units import Q_
        return Q_(self.embodied_carbon_kg, "kg")


class ServingResult(SolverResult):
    """Result from ServingModel: LLM two-phase inference performance."""
    feasible: bool
    ttft: Quantity
    itl: Quantity
    kv_cache_size: Quantity
    model_weights_size: Quantity
    total_memory_required: Quantity
    memory_utilization: float
    prompt_cache_hit_ratio: float = 0.0
    prefill_chunks: int = 1
    prefill_chunk_time: Optional[Quantity] = None
    decode_stall_bound: Optional[Quantity] = None


class TrainingMemoryResult(SolverResult):
    """Result from TrainingMemoryModel: per-accelerator training memory accounting."""
    feasible: bool
    total_memory: Quantity
    available_memory: Quantity
    memory_utilization: float
    weights: Quantity
    gradients: Quantity
    optimizer_state: Quantity
    activations: Quantity
    communication_buffers: Quantity
    precision: str
    optimizer: str
    activation_checkpointing: str
    parallelism: Dict[str, int]


class ServingCapacityResult(SolverResult):
    """Result from ServingCapacityModel: replica count for an inference SLA."""
    feasible: bool
    required_replicas: int
    qps_target: float
    qps_capacity: float
    per_replica_qps_capacity: float
    utilization: float
    target_p99_latency: Quantity
    estimated_p99_latency: Quantity
    base_request_latency: Quantity
    queue_wait_p99: Quantity
    active_batch_size: int
    ttft: Quantity
    itl: Quantity
    bottleneck: str


class MoERoutingResult(SolverResult):
    """Result from MoERoutingModel: MoE routing imbalance and all-to-all cost."""
    effective_active_experts: float
    effective_active_parameters: Quantity
    active_parameter_multiplier: float
    routing_imbalance_factor: float
    token_dispatch_bytes: Quantity
    all_to_all_latency: Optional[Quantity] = None
    ep_size: int = 1


class ContinuousBatchingResult(SolverResult):
    """Result from ContinuousBatchingModel: static max-length KV reservation vs PagedAttention.

    Unprefixed throughput, request, TTFT, and ITL fields describe the paged
    allocator; ``static_*`` fields describe max-length reservation on the same
    KV budget. Fragmentation fields are fractions in [0, 1] of allocated KV
    tokens left unused inside each request's allocation.
    """
    feasible: bool
    throughput_tokens_per_sec: float
    max_active_requests: int
    static_throughput_tokens_per_sec: float
    static_max_active_requests: int
    mean_request_tokens: float
    paged_internal_fragmentation: float
    static_internal_fragmentation: float
    paged_kv_cache_size: Quantity
    static_kv_cache_size: Quantity
    ttft: Quantity
    itl: Quantity
    speedup_vs_static: float


class WeightStreamingResult(SolverResult):
    """Result from WeightStreamingModel: Wafer-scale SRAM batch processing."""
    feasible: bool
    throughput_tokens_per_sec: float
    bottleneck: str
    layer_compute_time: Quantity
    layer_injection_time: Quantity
    optimal_batch_size: int
    wafer_memory_utilization: float


class TailLatencyResult(SolverResult):
    """Result from TailLatencyModel: queuing theory limits (M/M/c)."""
    p50_latency: Quantity
    p99_latency: Quantity
    queue_utilization: float
    is_stable: bool
    slo_headroom_ratio: float


class EconomicsResult(SolverResult):
    """Result from EconomicsModel: TCO breakdown.

    Embeds sustainability metrics since EconomicsModel delegates to
    SustainabilityModel internally.
    """
    capex_usd: float
    opex_energy_usd: float
    opex_maintenance_usd: float
    total_opex_usd: float
    tco_usd: float
    # Embedded from SustainabilityResult
    it_energy_kwh: Quantity
    total_energy_kwh: Quantity
    carbon_footprint_kg: float
    water_usage_liters: float
    pue: float
    region_name: str

    @property
    def capex(self) -> Quantity:
        """Amortized hardware CapEx as a Pint Quantity in USD."""
        from ..core.units import Q_
        return Q_(self.capex_usd, "USD")

    @property
    def opex_energy(self) -> Quantity:
        """Electricity OpEx as a Pint Quantity in USD."""
        from ..core.units import Q_
        return Q_(self.opex_energy_usd, "USD")

    @property
    def opex_maintenance(self) -> Quantity:
        """Maintenance OpEx as a Pint Quantity in USD."""
        from ..core.units import Q_
        return Q_(self.opex_maintenance_usd, "USD")

    @property
    def total_opex(self) -> Quantity:
        """Total OpEx as a Pint Quantity in USD."""
        from ..core.units import Q_
        return Q_(self.total_opex_usd, "USD")

    @property
    def tco(self) -> Quantity:
        """Total Cost of Ownership (CapEx + OpEx) as a Pint Quantity in USD."""
        from ..core.units import Q_
        return Q_(self.tco_usd, "USD")

    @property
    def it_energy(self) -> Quantity:
        """IT energy consumption as a Pint Quantity."""
        return self.it_energy_kwh

    @property
    def total_energy(self) -> Quantity:
        """Total facility energy consumption (post-PUE) as a Pint Quantity."""
        return self.total_energy_kwh

    @property
    def carbon_footprint(self) -> Quantity:
        """Carbon footprint as a Pint Quantity in kg."""
        from ..core.units import Q_
        return Q_(self.carbon_footprint_kg, "kg")

    @property
    def water_usage(self) -> Quantity:
        """Cooling water consumption as a Pint Quantity in liters."""
        from ..core.units import Q_
        return Q_(self.water_usage_liters, "liter")


class DataResult(SolverResult):
    """Result from DataModel: data pipeline feasibility."""
    is_stalled: bool
    utilization: float
    demand_bw: Quantity
    supply_bw: Quantity
    bottleneck: str
    margin: Quantity


class OffloadResult(SolverResult):
    """Legacy result type. Offload analysis is now folded into SingleNodeModel/PerformanceProfile."""
    feasible: bool
    transfer_time: Quantity
    compute_time: Quantity
    bottleneck: str
    effective_bandwidth: Quantity
    memory_spill_bytes: Quantity


class TopologyResult(SolverResult):
    """Result from TopologyModel: network bisection bandwidth."""
    effective_bw: Quantity
    bisection_bw: Quantity
    bisection_bw_fraction: float
    hops_avg: float
    topology: str
    num_nodes: int


class EfficiencyResult(SolverResult):
    """Result from EfficiencyModel: achievable MFU breakdown."""
    mfu: float
    achievable_flops: Quantity
    peak_flops: Quantity
    workload_type: str
    use_flash_attention: bool
    overhead_breakdown: Dict[str, float]


class TransformationResult(SolverResult):
    """Result from TransformationModel: CPU preprocessing bottleneck."""
    transform_time: Quantity
    accelerator_step_time: Quantity
    is_bottleneck: bool
    accelerator_utilization: float
    slowdown_factor: float


# ── Demand Layer Results ──────────────────────────────────────────

class ScalingResult(SolverResult):
    """Result from ScalingModel: Chinchilla-optimal training parameters."""
    optimal_parameters: Quantity
    optimal_tokens: Quantity
    compute_budget_flops: Quantity
    tokens_per_parameter: float


class CompressionResult(SolverResult):
    """Result from CompressionModel: compression trade-offs.

    ``original_size_gb``, ``compression_ratio``, ``memory_savings_pct`` and
    ``inference_speedup`` are all relative to ``baseline_precision``.
    """
    baseline_precision: str
    original_size_gb: Quantity
    compressed_size_gb: Quantity
    compression_ratio: float
    estimated_accuracy_delta: float
    memory_savings_pct: float
    inference_speedup: float = 1.0

    @property
    def original_size(self) -> Quantity:
        """Baseline model weight size as a Pint Quantity."""
        return self.original_size_gb

    @property
    def compressed_size(self) -> Quantity:
        """Compressed model weight size as a Pint Quantity."""
        return self.compressed_size_gb


class CompressionCandidate(SolverResult):
    """One evaluated compression configuration with lab-ready feasibility metadata."""
    label: str
    method: str
    target_bitwidth: Optional[int] = None
    sparsity: float = 0.0
    sparsity_type: str = "unstructured"
    baseline_precision: str
    original_size_gb: Quantity
    compressed_size_gb: Quantity
    compression_ratio: float
    estimated_accuracy_delta: float
    memory_savings_pct: float
    inference_speedup: float = 1.0
    hardware_supported: bool = True
    feasible: bool = True
    binding_constraint: str = "none"
    guardrail_violations: List[str] = Field(default_factory=list)
    pareto_status: str = "unknown"
    source_trace: List[str] = Field(default_factory=list)

    @property
    def original_size(self) -> Quantity:
        """Baseline model weight size as a Pint Quantity."""
        return self.original_size_gb

    @property
    def compressed_size(self) -> Quantity:
        """Compressed model weight size as a Pint Quantity."""
        return self.compressed_size_gb


class CompressionSweepResult(SolverResult):
    """A compression design-space sweep with candidate and Pareto metadata."""
    candidates: List[CompressionCandidate]
    frontier_labels: List[str] = Field(default_factory=list)
    dominated_labels: List[str] = Field(default_factory=list)
    best_candidate_label: Optional[str] = None
    objective: str = "min_size_max_speed_preserve_quality"


class SynthesisResult(SolverResult):
    """Result from SynthesisSolver: inverse-Roofline hardware requirements."""
    required_bw: Quantity
    required_flops: Quantity
    required_memory: Quantity
    compute_memory_ratio: Quantity
    target_latency: Quantity
    model_size: Quantity
    total_ops: Quantity


# ── Consequence Layer Results ─────────────────────────────────────

class OrchestrationResult(SolverResult):
    """Result from OrchestrationModel: queueing and wait times."""
    cluster_utilization: float
    avg_wait_time_days: Quantity
    avg_queue_length: float
    is_stable: bool

    @property
    def avg_wait_time(self) -> Quantity:
        """Expected queue wait time as a Pint Quantity."""
        return self.avg_wait_time_days


class InferenceScalingResult(SolverResult):
    """Result from InferenceScalingModel: CoT reasoning cost."""
    total_reasoning_time: Quantity
    ttft: Quantity
    itl: Quantity
    tokens_generated: int
    reasoning_steps: int
    time_per_step: Quantity
    energy_per_query: Quantity
    feasible: bool
    serving_detail: Any  # ServingResult (avoids circular validation)


class SensitivityResult(SolverResult):
    """Result from SensitivitySolver: binding constraint identification.

    ``feasible`` mirrors the baseline solve. When it is False the model does
    not fit in memory, ``binding_constraint`` is ``"memory_capacity"``, and
    ``constraint_trace`` explains why the perturbation sensitivities are not
    a ranking of hardware levers.
    """
    feasible: bool
    sensitivities: Dict[str, float]
    binding_constraint: str
    baseline_latency: Quantity
    perturbation_pct: float


class ResponsibleEngineeringResult(SolverResult):
    """Result from ResponsibleEngineeringModel: ethics tax quantification."""
    dp_slowdown_factor: float
    effective_training_time: Quantity
    additional_data_requirement: float
    epsilon: float
    delta: float
    min_subgroup_prevalence: float
    privacy_cost_ratio: float
    fairness_data_ratio: float


# ── Search & Optimization Results ─────────────────────────────────

class OptimizerResult(BaseModel):
    """Base class for all search-based results."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    objective_value: float
    best_config: Dict[str, Any]
    total_searched: int


class ParallelismOptimizerResult(OptimizerResult):
    """Result from ParallelismOptimizer: optimal TP/PP/DP split."""
    best_mfu: float
    best_throughput: Quantity
    best_step_time: Quantity
    search_space_size: int
    top_candidates: List[Dict[str, Any]]


class BatchingOptimizerResult(OptimizerResult):
    """Result from BatchingOptimizer: optimal batch size for latency SLA."""
    best_batch_size: int
    max_throughput: float
    p99_latency: Quantity
    slo_headroom_ratio: float
    is_feasible: bool


class PlacementOptimizerResult(OptimizerResult):
    """Result from PlacementOptimizer: optimal datacenter region."""
    best_region: str
    lowest_tco: float
    carbon_footprint: float
    pue: float
    top_candidates: List[Dict[str, Any]]


# ── Situation results (2026-10-20) ────────────────────────────────

class DecodeBoundResult(SolverResult):
    """Result from DecodeBoundModel: weight-streaming bound on one decode step."""
    weight_bytes: Quantity
    flops_per_step: Quantity
    compute_time: Quantity
    memory_time: Quantity
    memory_share: float          # memory_time / (compute_time + memory_time)
    bound_ratio: float           # memory_time / compute_time
    intensity: Quantity          # FLOP per weight byte
    attainable_flops: Quantity   # min(peak, bandwidth x intensity)
    peak_fraction: float         # attainable / peak
    bottleneck: str


class TransferLadderResult(SolverResult):
    """Result from TransferLadderModel: one payload moved over each link tier."""
    payload: Quantity
    bandwidths: Dict[str, Quantity]
    times: Dict[str, Quantity]


class TensorParallelCommResult(SolverResult):
    """Result from TensorParallelCommModel: activation AllReduces in one TP step."""
    payload: Quantity            # one AllReduce's logical tensor
    ring_factor: float           # 2(N-1)/N
    traffic_per_device: Quantity
    collectives_per_step: int
    per_collective_time: Quantity
    communication_time: Quantity
    compute_time: Quantity
    step_time: Quantity          # compute + communication, no overlap
    compute_fraction: float
    communication_fraction: float


class TrainingRunResult(SolverResult):
    """Result from TrainingRunModel: wall-clock chain from peak to delivered time."""
    total_flops: Quantity
    peak_time: Quantity          # every device at its peak
    compute_time: Quantity       # at the local compute efficiency
    scaled_time: Quantity        # after the scaling efficiency
    stage_times: Dict[str, Quantity]   # after each overhead, in order
    total_time: Quantity
    delivered_fraction: float    # peak_time / total_time


class TrainingStateResult(SolverResult):
    """Result from TrainingStateModel: persistent training state and its shards."""
    weights: Quantity
    gradients: Quantity
    master_weights: Quantity
    adam_moment: Quantity        # one Adam moment (m or v)
    optimizer_moments: Quantity  # both moments
    persistent_state: Quantity
    activations: Quantity
    tp_weights_per_device: Quantity
    tp_gradients_per_device: Quantity
    tp_master_per_device: Quantity
    tp_moments_per_device: Quantity
    tp_activations_per_device: Quantity
    tp_total_per_device: Quantity
    tp_offloaded_per_device: Quantity  # after moving master and moments to host
    zero_moments_per_device: Quantity  # moments sharded over TP x DP
    zero_moments_per_tp_group: Quantity
    zero_weights_and_moments_per_device: Quantity


class PowerTransientResult(SolverResult):
    """Result from PowerTransientModel: one device's load step and a cluster's ramp."""
    power_step: Quantity
    current_step: Quantity
    droop: Dict[str, Quantity]         # keyed by rise time label
    droop_share: Dict[str, float]
    cluster_power_step: Quantity
    ramp_rate: Quantity


class CoolantFlowResult(SolverResult):
    """Result from CoolantFlowModel: coolant flow that carries away a heat load."""
    mass_flow: Quantity
    volume_flow: Quantity


class OutageCostResult(SolverResult):
    """Result from OutageCostModel: idle cost of a failure domain going down."""
    direct_cost: float           # dollars, failed devices only
    job_cost: float              # dollars, a synchronous job idles every device
    job_cost_with_rollback: Dict[str, float]


class FailureIntervalResult(SolverResult):
    """Result from FailureIntervalModel: series failures across identical components."""
    interval: Quantity
    expected_failures: float
    failures_per_day: float


class GradientSyncResult(SolverResult):
    """Result from GradientSyncModel: one data-parallel gradient exchange per link."""
    sync_bytes: Quantity
    compute_time: Quantity
    transfer_times: Dict[str, Quantity]
    step_times: Dict[str, Quantity]
    compute_fractions: Dict[str, float]
    energy_per_step: Quantity      # on the energy link, per worker
    energy_total: Quantity         # over all steps, per worker


class CommunicationIntensityResult(SolverResult):
    """Result from CommunicationIntensityModel: bytes sent per local FLOP vs. fabric balance."""
    local_tokens: int
    bytes_per_step: Quantity
    flops_per_step: Quantity
    intensity: float               # bytes per FLOP
    balance: float                 # link bytes/s per peak FLOP/s
    compute_time_at_peak: Quantity


class TrainingGrowthResult(SolverResult):
    """Result from TrainingGrowthModel: annual growth of training compute and its factors."""
    run_count: int
    compute_factor: float
    count_factor: float
    duration_factor: float
    rate_factor: float
    fleet_share: float             # share of the compute exponent from count and duration
    span_orders: float             # orders of magnitude from AlexNet to the largest run


class AlphaBetaResult(SolverResult):
    """Result from AlphaBetaModel: alpha + n/beta for each message on each path."""
    alphas: Dict[str, Quantity]
    bandwidths: Dict[str, Quantity]
    messages: Dict[str, Quantity]
    times: Dict[str, Dict[str, Quantity]]               # times[path][message]
    transmission_times: Dict[str, Dict[str, Quantity]]  # the n/beta term alone
    crossovers: Dict[str, Quantity]                     # n* = alpha * beta, per path


class BisectionResult(SolverResult):
    """Result from BisectionModel: a ring AllReduce across a full and an oversubscribed cut."""
    devices_per_side: int
    bisection_full: Quantity
    bisection_oversubscribed: Quantity
    ring_factor: float
    crossing_bytes: Quantity
    time_full: Quantity
    time_oversubscribed: Quantity
    slowdown: float
    relative_throughput: float
    throughput_loss: float
    idle_cost: Quantity


class IncastResult(SolverResult):
    """Result from IncastModel: many senders into one egress port with a finite buffer."""
    offered_load: Quantity
    oversubscription: float
    net_fill_rate: Quantity
    overflow_time: Quantity          # buffer fills from empty
    drain_time: Quantity             # a full buffer drained by the egress port alone
    bandwidth_delay_product: Quantity


class CollectiveFitResult(SolverResult):
    """Result from CollectiveFitModel: a floor-plus-bandwidth fit to a measured collective sweep."""
    row_count: int
    floor_time: Quantity
    bandwidth: Quantity
    knee: Quantity
    worst_mid_ratio: float
    peak_bus_bandwidth: Quantity
    mid_bus_bandwidth_low: Quantity
    mid_bus_bandwidth_high: Quantity

    def model_time(self, size: Quantity) -> Quantity:
        """Modeled time for a message of ``size``: floor + size / bandwidth."""
        return (self.floor_time + size / self.bandwidth).to("microsecond")
# ── Storage calculation results (added 2026-10-20) ────────────────

class StorageFootprintResult(SolverResult):
    """Result from StorageFootprintModel: what a training run reads, saves, and keeps."""
    checkpoint_bytes: Quantity
    saves: int
    checkpoint_bytes_written: Quantity
    write_read_ratio: float          # checkpoint bytes written / corpus bytes read
    average_read_rate: Quantity      # corpus bytes / run duration
    shard_count: int
    archived_checkpoints: int
    archive_bytes: Quantity          # retained checkpoints + source + corpus


class TieredCheckpointResult(SolverResult):
    """Result from TieredCheckpointModel: a save staged on local storage, then drained."""
    checkpoint_bytes: Quantity
    per_node_shard: Quantity
    local_write_time: Quantity       # the training pause
    drain_time: Quantity             # background copy to the durable tier
    durable_share_per_node: Quantity
    pause_fraction: float            # pause / save interval


class CheckpointStormResult(SolverResult):
    """Result from CheckpointStormModel: every replica writes against one logical copy."""
    state_bytes: Quantity
    storm_bytes: Quantity
    storm_time: Quantity
    single_copy_time: Quantity
    slowdown: float


class InputDemandResult(SolverResult):
    """Result from InputDemandModel: bandwidth a fleet of consumers draws from storage."""
    batch_bytes: Quantity            # one consumer's batch
    bandwidth: Quantity


class DataStallResult(SolverResult):
    """Result from DataStallModel: input wait with and without overlap."""
    sequential_step: Quantity
    sequential_stall: float
    overlapped_step: Quantity
    overlapped_stall: float


class PrefetchDepthResult(SolverResult):
    """Result from PrefetchDepthModel: ready batches needed to cover each tier's tail."""
    spans: Dict[str, float]          # tail latency / compute window
    depths: Dict[str, int]           # ceil(span)
    buffer_per_consumer: Quantity    # at the stated safety depth
    buffer_per_node: Quantity


class TieredCacheResult(SolverResult):
    """Result from TieredCacheModel: mean service of a two-tier cache."""
    hit_rate: float
    miss_penalty: float              # miss service time / hit service time
    mean_service: float              # in hit service times
    delivered_fraction: float        # of the all-hit rate
    delivered_rate: Optional[Quantity] = None


class HostIoCpuResult(SolverResult):
    """Result from HostIoCpuModel: CPU time per second spent issuing I/O."""
    request_rate: float              # requests per second
    cpu_seconds_per_second: Dict[str, float]


class ObjectStorageBillResult(SolverResult):
    """Result from ObjectStorageBillModel: storing and reading a dataset in object storage."""
    storage_cost: float              # dollars over the billing period
    egress_per_read: float           # dollars
    egress_cost: float               # dollars, all reads
    request_cost: float              # dollars, one GET per file
    archive_retrieval_cost: float    # dollars, retrieval fee only
    archive_restore_cost: float      # dollars, retrieval plus one egress


class StreamVsStageResult(SolverResult):
    """Result from StreamVsStageModel: reading every epoch remotely against staging per run."""
    source_storage_cost: float       # dollars a year, both options
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
    fleet_cost_per_day: float        # dollars
    loss_per_day: float
    loss_total: float


class SyntheticFootprintResult(SolverResult):
    """Result from SyntheticFootprintModel: generated data at rest and in motion."""
    stored_bytes: Quantity
    verification_bytes: Quantity
    pipeline_bytes: Quantity
    capacity_amplification: float
    traffic_amplification: float


class StorageBenchmarkResult(SolverResult):
    """Result from StorageBenchmarkDemandModel: measured storage demand per accelerator."""
    result_count: int
    medians: Dict[str, Dict[str, float]]   # workload -> accelerator -> GB/s per accelerator
    generation_ratios: Dict[str, float]    # workload -> newer / older accelerator


# --- Embodied stopping, traction, and domain compute (added 2026-10-20) -----

class StoppingBudgetResult(SolverResult):
    """Result from StoppingBudgetModel: the elementary stop against a clear distance."""
    speed: Quantity                          # speed when the hazard appears; braking starts from it
    delay: Quantity
    deceleration: Quantity
    delay_travel: Quantity
    braking_travel: Quantity
    stop_distance: Quantity                  # delay travel + braking travel: travel to rest
    overhead: Quantity                       # localization bound + protective margin (zero if unstated)
    required_distance: Quantity              # stop_distance + overhead: what the clear distance must hold
    braking_time: Quantity
    crossover_speed: Quantity                # 2 a tau: delay travel equals braking travel there
    braking_over_delay: Optional[float] = None
    delay_share_of_braking: Optional[float] = None
    clear_distance: Optional[Quantity] = None
    spare: Optional[Quantity] = None
    fits: Optional[bool] = None
    clear_over_stop: Optional[float] = None
    speed_ceiling: Optional[Quantity] = None  # fastest speed whose stop, with the overhead, fits
    added_delay_travel: Optional[Quantity] = None  # travel at ``speed`` during a stated delay step

    def travel_over(self, duration: Quantity) -> Quantity:
        """Distance covered at ``speed`` over ``duration`` (the cost of one term of the delay)."""
        return (self.speed * duration).to("meter")


class TractionLossResult(SolverResult):
    """Result from TractionLossModel: what a drop in floor friction does to braking."""
    braking_distance_ratio: float            # braking distance at the traction limit, after / before
    traction_ceiling: Quantity               # deceleration the lower friction can support
    required_friction: float                 # friction the stated deceleration needs
    premise_holds: bool                      # the lower friction still supports it


class DomainComputeGapResult(SolverResult):
    """Result from DomainComputeGapModel: one model against the largest of another domain that year."""
    model: str
    year: str
    model_flop: float
    reference_model: str
    reference_flop: float
    gap: float                               # reference_flop / model_flop
    domain_models_with_compute: int          # rows of the model's domain that carry a compute estimate


# --- Agent trajectories: reliability, cost, context, measured runs (added 2026-10-20) ---

class HorizonCompoundingResult(SolverResult):
    """Result from HorizonCompoundingModel: unchecked task success across horizons."""
    step_error: float                    # 1 - per-step success
    success: Dict[int, float]            # horizon (turns) -> unchecked task success
    required_step_error: float           # per-step error that holds the target at the long horizon
    error_ratio: float                   # step_error / required_step_error
    required_error_horizon: float        # required_step_error x long horizon
    small_error_limit: float             # -ln(target): the same product as eps -> 0


class AcceptedTaskCostResult(SolverResult):
    """Result from AcceptedTaskCostModel: cost per accepted task and two levers on it (dollars)."""
    attempt_cost: float
    token_share: float                   # model-token share of the attempt cost
    attempts_per_acceptance: float       # 1 / p_acc
    cost_per_accepted: float
    accepted_only_understatement: float  # cost_per_accepted / attempt_cost
    cheaper_attempt_cost: Optional[float] = None
    cost_per_accepted_cheaper_tokens: Optional[float] = None
    cost_per_accepted_improved: Optional[float] = None
    break_even_acceptance: Optional[float] = None
    cost_per_accepted_cheaper_other: Optional[float] = None
    cost_per_accepted_acceptance_gain: Optional[float] = None
    independent_success: Optional[float] = None


class AppendOnlyContextResult(SolverResult):
    """Result from AppendOnlyContextModel: what an append-only trajectory reads and resends."""
    last_call_tokens: Quantity
    window_fraction: float               # last call / context window
    cumulative_input_tokens: Quantity
    reread_ratio: float                  # cumulative input / last call


class TaskHorizonTrendResult(SolverResult):
    """Result from TaskHorizonTrendModel: growth of the measured 50% task horizon."""
    first_frontier_minutes: float
    last_frontier_minutes: float
    first_date: str
    last_date: str
    growth: float                        # last / first frontier horizon
    doubling_all: Quantity               # whole-period fit
    doubling_recent: Quantity            # fit from the recent start date
    recent_slope_per_day: float          # log2(minutes) per day, recent fit
    recent_intercept: float              # log2(minutes) at date ordinal 0, recent fit


class AgentRunSummaryResult(SolverResult):
    """Result from AgentRunSummaryModel: one recorded agent run over a benchmark, one attempt per task."""
    tasks: int
    accepted: int
    rejected: int
    acceptance: float
    submitted: int                       # tasks whose loop ended by the agent's own submission
    total_cost: float                    # dollars
    mean_attempt_cost: float
    cost_per_accepted: float
    accepted_mean_cost: float            # mean over accepted attempts only
    accepted_only_understatement: float  # cost_per_accepted / accepted_mean_cost
    calls_median: float
    calls_min: int
    calls_max: int
    commands: Dict[str, int]             # command class -> count
    command_total: int
    observations: int
    nonzero_exits: int
    truncated_outputs: int
    syntax_error_tasks: int
    syntax_error_accepted: int
    call_bins: List[Dict[str, Any]]      # {"low", "high", "tasks", "accepted"} per bin
# ── Model calls and agent trajectories (added 2026-10-20) ─────────

class ContextAdmissionResult(SolverResult):
    """Result from ContextAdmissionModel: token-measured admission, S <= S_max - K_max."""
    input_ceiling: int               # window - reserved output
    headroom: int                    # input ceiling - text prompt (negative: no room)
    total_demand: int                # text + image + reserved output
    excess: int                      # total demand - window (positive: rejected)
    admitted: bool


class SamplingResult(SolverResult):
    """Result from SamplingModel: one logit vector at two temperatures, then top-p."""
    partition_sum: float             # sum of exp(z / T) at the base temperature
    base_probabilities: List[float]
    sharp_probabilities: List[float]
    base_tail: float                 # 1 - p(top token) at the base temperature
    sharp_tail: float
    tail_ratio: float                # base tail / sharp tail
    nucleus_size: int
    nucleus_mass: float
    nucleus_probabilities: List[float]


class CheckerPosteriorResult(SolverResult):
    """Result from CheckerPosteriorModel: defects among candidates a fallible check passes."""
    defective_pass: float            # joint P(Def and PASS)
    pass_share: float                # P(PASS)
    posterior: float                 # P(Def | PASS)
    catch_rate: float                # P(FAIL | Def)
    valid_reject_rate: float         # P(FAIL | Valid)


class ModelCallResult(SolverResult):
    """Result from ModelCallModel: prefill and decode work of one call at stated floors."""
    prompt_tokens: int
    output_tokens: int
    prefill_time: Quantity
    decode_time: Quantity
    total_time: Quantity
    output_token_share: float
    output_time_share: float
    fresh_prefill_time: Quantity     # prefill of the tokens past a reused shared prefix
    decode_rate: Quantity            # output tokens per second at the decode floor
    deadline_margin: Optional[float] = None   # deadline / total time
    acceptance: Optional[float] = None        # product of the stated stage success rates


class TrajectoryResendResult(SolverResult):
    """Result from TrajectoryResendModel: input a trajectory resends, with and without prefix reuse."""
    turns: int
    first_prompt_term: int           # H * S0
    appended_term: int               # D * H (H - 1) / 2
    resent_tokens: int               # their sum: input sent without reuse
    reused_prefix_tokens: int        # fresh prefill with prefix reuse
    reuse_reduction: float           # resent / reused
    output_tokens: int
    input_share: float               # resent / (resent + output)
    prefill_time: Quantity           # resent tokens at the prefill floor
    reused_prefill_time: Quantity
    decode_time: Quantity


class TokenMixResult(SolverResult):
    """Result from TokenMixModel: prompt and output tokens of a request trace."""
    requests: int
    median_prompt: float
    median_output: float
    median_ratio: float
    input_share: float               # prompt tokens / all tokens
    output_time_share: float         # decode time / all time at the stated floors
# --- Collective communication (added 2026-10-20) ---------------------------

class AllReduceAlgorithmResult(SolverResult):
    """Result from AllReduceAlgorithmModel: ring, simple tree, and star schedules per group size.

    Every dict is keyed by the group size N. Byte times exclude startup; startup
    times are the round count times alpha, kept in alpha's own unit.
    """
    message: Quantity
    alpha: Quantity
    bandwidth: Quantity
    chunk: Dict[int, Quantity]                  # M / N, the bytes one ring round moves
    ring_factor: Dict[int, float]               # 2(N-1)/N
    ring_rounds: Dict[int, int]                 # 2(N-1)
    tree_rounds: Dict[int, int]                 # 2 ceil(log2 N)
    ring_byte_time: Dict[int, Quantity]
    ring_startup_time: Dict[int, Quantity]
    ring_time: Dict[int, Quantity]
    tree_byte_time: Dict[int, Quantity]
    tree_startup_time: Dict[int, Quantity]
    tree_time: Dict[int, Quantity]
    star_byte_time: Dict[int, Quantity]         # one server port carries N x M each way
    star_ring_limit: Dict[int, float]           # N/2, the large-N limit of star over ring byte time
    crossover_estimate: Dict[int, Quantity]     # N alpha beta / log2 N
    crossover_exact: Dict[int, Quantity]        # where the two models are equal


class HierarchicalAllReduceResult(SolverResult):
    """Result from HierarchicalAllReduceModel: three phases against one flat ring."""
    intra_bytes: Quantity          # bytes each GPU sends in the in-node ReduceScatter
    reduce_scatter_time: Quantity
    shard: Quantity                # M / G, the payload of each cross-node ring
    inter_byte_time: Quantity
    inter_startup_time: Quantity
    inter_time: Quantity
    all_gather_time: Quantity
    total_time: Quantity
    flat_ring_time: Quantity       # one ring over every GPU at the inter-node rate, with startup
    flat_bound_time: Quantity      # the large-N byte bound 2M / beta of that ring
    speedup_vs_flat_ring: float
    speedup_vs_flat_bound: float


class AllToAllDispatchResult(SolverResult):
    """Result from AllToAllDispatchModel: one expert-parallel dispatch under uniform routing."""
    tokens_per_device: int
    experts_per_device: int
    tokens_per_expert: int
    tokens_per_pair: int           # tokens one device sends one peer
    remote_peers: int
    bytes_per_token: Quantity
    pair_bytes: Quantity
    sent_bytes: Quantity           # per device, to all remote peers
    injection_bound: Quantity      # alpha + sent / beta, the peers sharing one port
    uniform_remote_tokens: int
    hot_remote_tokens: int
    uniform_receive_bytes: Quantity
    hot_receive_bytes: Quantity
    uniform_receive_bound: Quantity
    hot_receive_bound: Quantity


class RingTraceResult(SolverResult):
    """Result from RingTraceModel: every cell of a ring AllReduce on small integer vectors.

    ``scatter_reduce[s][g]`` and ``all_gather[s][g]`` are (chunk index, value)
    received by GPU g in step s.
    """
    scatter_reduce: List[List[Tuple[int, int]]]
    owned: List[Tuple[int, int]]     # the fully reduced chunk each GPU holds after ReduceScatter
    totals: List[int]
    all_gather: List[List[Tuple[int, int]]]
    held: List[Set[int]]             # chunks each GPU holds after AllGather
    steps: int


class ErrorFeedbackResult(SolverResult):
    """Result from ErrorFeedbackModel: a rounding compressor with and without a carried residual."""
    errors: List[float]              # e_1 .. e_{T+1}; errors[t] enters step t + 1
    corrected: List[float]           # g_t + e_t
    sent: List[int]                  # v_t
    cumulative_true: List[float]
    cumulative_sent: List[int]
    total_sent: int
    final_error: float
    true_total: float
    naive_total_sent: int            # every value rounded alone, no residual


class CompressionRunResult(SolverResult):
    """Result from CompressionRunModel: time to a quality target with and without compression."""
    steps_compressed: int
    step_times: Dict[str, Quantity]
    compressed_step_times: Dict[str, Quantity]
    totals: Dict[str, Quantity]
    compressed_totals: Dict[str, Quantity]
    changes: Dict[str, float]        # compressed / baseline - 1 (negative is faster)


class OverlapBudgetResult(SolverResult):
    """Result from OverlapBudgetModel: layer-by-layer AllReduce, sequential against overlapped."""
    layers: int
    gradient_per_layer: Quantity
    buckets_per_layer: int
    communication_per_layer: Quantity
    total_backward: Quantity
    sequential_time: Quantity
    exposed_per_layer: Quantity
    overlapped_time: Quantity
    savings_fraction: float
# --- Distributed training (added 2026-10-20) ---------------------------------

class ParallelLayoutResult(SolverResult):
    """Result from ParallelLayoutModel: one TP x PP x DP layout's step and state."""
    n_devices: int
    step_flops: Quantity
    compute_time: Quantity           # per device per step (T_compute / N)
    gradient_bytes: Quantity         # full gradient at the communication precision
    dp_shard: Quantity               # gradient bytes each data-parallel rank reduces
    ring_bytes: Quantity             # 2(DP-1)/DP x shard, exact ring volume per device
    allreduce_time: Quantity         # 2 x shard / link bandwidth (large-N ring bound)
    hidden_time: Quantity
    exposed_time: Quantity
    bubble: float                    # pipeline idle fraction
    efficiency: float                # with overlap and bubble
    efficiency_no_overlap: float
    overlap_share: float             # hidden / allreduce
    step_time: Quantity              # compute / (1 - bubble) + exposed
    state_total: Quantity            # persistent mixed-precision Adam state
    state_per_device: Quantity       # split by TP x PP
    zero1_state_per_device: Quantity # plus optimizer state sharded across DP
    fleet_mfu: float                 # local efficiency x scaling efficiency
    double_counted_mfu: float        # the pitfall: scaling efficiency applied twice
    min_devices_for_state: Optional[int] = None


class ZeroShardingResult(SolverResult):
    """Result from ZeroShardingModel: per-device state at each ZeRO stage."""
    weights: Quantity
    gradients: Quantity
    optimizer_state: Quantity
    stage_state: Dict[str, Quantity]       # "DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3" per device
    traffic_multiple: Dict[str, float]     # per-device bytes relative to DDP's AllReduce
    activations: Quantity
    zero3_total_per_device: Quantity       # ZeRO-3 state plus activations
    gather_traffic: Quantity               # ZeRO-3 per-device bytes per step
    gather_time: Optional[Quantity] = None


class DataParallelScaleOutResult(SolverResult):
    """Result from DataParallelScaleOutModel: a data-parallel job inside and across nodes."""
    n_devices: int
    gradient_bytes: Quantity
    intra_bytes: Quantity
    inter_bytes: Quantity
    intra_time: Quantity
    inter_time: Quantity
    communication_time: Quantity
    compute_time: Quantity           # per optimizer update (accumulation x step)
    update_time: Quantity
    communication_overhead: float    # communication / compute per update
    compute_share: float
    communication_share: float
    speedup: float
    efficiency: float
    global_batch: int
    training_time: Quantity
    cost: Optional[float] = None     # dollars


class TensorParallelLayerResult(SolverResult):
    """Result from TensorParallelLayerModel: one layer's compute against its TP exchange."""
    layer_flops: Quantity
    compute_time: Quantity
    layer_bytes: Quantity
    transfer_times: Dict[str, Quantity]


class AlignmentMemoryResult(SolverResult):
    """Result from AlignmentMemoryModel: PPO and DPO model state for RLHF."""
    policy_state: Quantity
    reference_weights: Quantity
    reward_weights: Quantity
    value_state: Quantity
    ppo_total: Quantity
    dpo_total: Quantity
    dpo_reduction: float
    model_parallel_devices: int
    policy_per_device: Quantity
    reference_per_device: Quantity
    kv_cache: Quantity
    kv_per_device: Quantity
    min_devices_for_ppo: Optional[int] = None


class ScalingSeriesResult(SolverResult):
    """Result from ScalingSeriesModel: same-round benchmark series and their efficiencies."""
    labels: List[str]
    series: List[List[Dict[str, Any]]]   # each series' rows, smallest configuration first
    efficiencies: List[float]            # largest configuration relative to the smallest


class MemoryWallResult(SolverResult):
    """Result from MemoryWallModel: demand and capacity rows and the crossover."""
    rows: List[Dict[str, Any]]           # Year, Date, Name, Type ("Model" or "GPU"), Memory_GB
    capacity_growth: float               # largest over smallest accelerator memory
    demand_orders: float                 # log10 of largest over smallest demand
    crossover_name: str
    crossover_year: int
    crossover_gap: float                 # crossover demand over that year's largest accelerator
# ── Single-machine iron law, Amdahl, and silent-degradation results (added 2026-10-20) ──

class IronLawResult(SolverResult):
    """Result from IronLawModel: the three iron-law terms, summed and overlapped, before and after an upgrade."""
    memory_time: Quantity
    compute_time: Quantity
    overhead: Quantity
    sequential_time: Quantity          # memory + compute + overhead
    overlapped_time: Quantity          # max(memory, compute) + overhead
    upgraded_memory_time: Quantity
    upgraded_compute_time: Quantity
    upgraded_sequential_time: Quantity
    upgraded_overlapped_time: Quantity
    naive_upgraded_time: Quantity      # the sequential time divided by the whole upgrade
    sequential_reduction: float        # fraction of the sequential time the upgrade removes


class AmdahlResult(SolverResult):
    """Result from AmdahlModel: one stage of a sequential pipeline made faster."""
    total_time: Quantity
    stage_time: Quantity
    upgraded_stage_time: Quantity
    upgraded_total_time: Quantity
    untouched_time: Quantity
    stage_fraction: float              # share of the original total spent in the stage
    overall_speedup: float
    end_to_end_reduction: float        # fraction of the total removed
    stage_reduction: float             # fraction of the stage removed
    speedup_bound: float               # overall speedup with an infinitely fast stage


class ShiftedTestSetResult(SolverResult):
    """Result from ShiftedTestSetModel: accuracy on an original and a re-collected test set."""
    model_count: int
    baseline_count: int                # models named with the baseline prefix (non-neural)
    gap_min: float                     # percentage points, all models
    gap_max: float
    network_gap_min: float             # percentage points, models without the baseline prefix
    network_gap_max: float
    mean_gap: float
    slope: float                       # least-squares fit: new = slope * original + intercept
    intercept: float


class ProxyDriftResult(SolverResult):
    """Result from ProxyDriftModel: a proxy estimate against its reference, season by season."""
    season_ratios: Dict[str, float]    # season label -> summed proxy / summed reference
    stable_max_deviation: float        # largest |ratio - 1| before the first drift season
    peak_season: str
    peak_proxy: float                  # the season's highest weekly proxy value (reference units)
    peak_reference: float
    peak_ratio: float
    max_week_ratio: float              # largest single-week proxy / reference in the peak season
# ── Physical-plant budget results: stopping, actuation, heat, power (added 2026-10-20) ──



class SpeedStepResult(SolverResult):
    """Result from SpeedStepModel: what raising a speed setpoint costs in braking travel."""
    braking_growth: float            # raised braking / base braking - 1
    braking_added: Quantity
    equivalent_delay: Quantity       # added braking expressed as delay at the base speed


class FreshnessProfileResult(SolverResult):
    """Result from FreshnessProfileModel: a target's freshness deadline against an age profile."""
    allowed_error: Quantity
    deadline: Quantity
    stage_sum_p50: Quantity
    stage_sum_p99: Quantity
    stage_sum_max: Quantity
    focus_stage: Quantity            # the stage a benchmark reports, at the median
    rest_of_chain: Quantity          # median chain minus that stage
    focus_share: float               # of the median chain
    focus_rate: Quantity             # loop rate the benchmark alone suggests
    travel_p50: Quantity
    travel_tail: Quantity
    slack_p50: Quantity
    overrun_tail: Quantity


class BrakingCeilingsResult(SolverResult):
    """Result from BrakingCeilingsModel: traction, tip-over, drive-torque, and back-EMF checks."""
    wheel_torque: Quantity
    force: Quantity
    rotating_mass: Quantity
    rotating_share: float
    m_eff: Quantity
    a_torque: Quantity
    torque_margin: float
    a_torque_rated: Quantity
    mu_required: float               # all weight on the driven wheels
    drive_load_share: float          # upper bound from the load rating
    mu_required_rated: float         # at that load share
    min_load_share_dry: float
    a_traction_dry: Quantity
    a_traction_inspected: Quantity
    cg_ratio_max: float
    a_tip: Optional[Quantity] = None
    passes: bool
    wheel_speed: Quantity
    back_emf: Quantity
    v_brake: Quantity
    v_drive: Quantity
    headroom: Quantity


class HubDriveElectricalResult(SolverResult):
    """Result from HubDriveElectricalModel: current rise and phase-short braking of a hub drive."""
    time_constant: Quantity
    ramp_time: Quantity
    ramp_share: float                # of the brake onset
    full_step_current: Quantity
    step_over_peak: float
    short_peak_electrical: Quantity  # rad/s
    short_peak_wheel_speed: Quantity # rpm


class GearedRotorResult(SolverResult):
    """Result from GearedRotorModel: reflected rotor inertia and the inertia match."""
    reflected_inertia: Quantity
    flywheel_mass: Quantity
    radius_of_gyration: Quantity
    matched_ratio: float
    acceleration_share: float        # at the stated ratio, of the matched-ratio acceleration


class GearRatioCostResult(SolverResult):
    """Result from GearRatioCostModel: holding heat and reflected inertia per gear ratio."""
    motor_constant: Quantity
    motor_torque: Dict[int, Quantity]
    copper_loss: Dict[int, Quantity]
    reflected_inertia: Dict[int, Quantity]
    loss_inertia_product: Dict[int, Quantity]
    over_peak: Dict[int, float]      # motor torque / motor peak torque
    peak_over_rated: float


class ThermalBurstResult(SolverResult):
    """Result from ThermalBurstModel: burst horizons, duty screen, and periodic peak."""
    time_constant: Quantity
    rated_loss: Quantity
    rated_equilibrium: Quantity
    burst_current: Quantity
    burst_loss: Quantity
    asymptote: Quantity
    horizon_cold: Quantity
    horizon_hot: Quantity
    hot_cut: float
    average_loss_max: Quantity
    duty: float
    off_time: Quantity
    off_time_scheduled: Quantity
    periodic_peak: Quantity


class StallDeratingResult(SolverResult):
    """Result from StallDeratingModel: stall equilibrium with copper feedback, and the sustainable hold."""
    motor_torque: Quantity
    stall_current: Quantity
    cold_loss: Quantity
    feedback_constant: Quantity
    feedback_gain: float
    steady_rise: Quantity
    winding_temperature: Quantity
    sustainable_loss: Quantity
    hot_resistance: Quantity
    sustainable_current: Quantity
    sustainable_hold: Quantity
    derating: float


class ContinuousDeratingResult(SolverResult):
    """Result from ContinuousDeratingModel: installed continuous torque against the catalog rating."""
    ambient_share: float
    combined_share: float


class DriveBusResult(SolverResult):
    """Result from DriveBusModel: sustained sag, regenerative surge, and chopper sizing."""
    compute_current: Quantity
    drive_current: Quantity
    sag: Quantity
    sustained_full: Quantity
    sustained_depleted: Quantity
    margin_depleted: Quantity
    headroom_depleted: Quantity
    crossing_current: Quantity
    kinetic_energy: Quantity
    regenerated: Quantity
    capacitor_only_endpoint: Quantity
    capacitor_to_clamp: Quantity
    stop_time: Quantity
    peak_power: Quantity
    resistor_max: Quantity


class ZeroFailureResult(SolverResult):
    """Result from ZeroFailureModel: the failure rate a clean campaign supports."""
    coefficient: float               # -ln(1 - C)
    rate_upper: float
    one_in: float


class ThrottledStageResult(SolverResult):
    """Result from ThrottledStageModel: a compute-bound stage slowed by a lower clock."""
    throttled: Quantity
    added_age: Quantity
    added_travel: Quantity


# ── Test-time deliberation: allocation, pools, checks, search (added 2026-10-20) ──

class ProbeVsAnswerResult(SolverResult):
    """Result from ProbeVsAnswerModel: a generated explanation against a prepared test."""
    token_time: Quantity             # decode floor per output token
    answer_time: Quantity            # answer_tokens * token_time
    answer_cost: Quantity            # USD of output tokens
    time_ratio: float                # answer_time / probe_time


class AllocationAxesResult(SolverResult):
    """Result from AllocationAxesModel: one output ceiling spent as depth, breadth, or feedback."""
    token_time: Quantity
    depth_input: int
    depth_output: int
    depth_time: Quantity
    depth_cost: Quantity
    breadth_input: int
    breadth_output: int
    breadth_time: Quantity           # concurrent streams: one candidate's generation time
    breadth_cost: Quantity
    feedback_input: int              # append-only: every round resends earlier candidates and reports
    feedback_output: int
    feedback_time: Quantity          # generation plus every check
    feedback_cost: Quantity
    path_ratio: float                # depth_time / breadth_time
    bill_ratio: float                # breadth_cost / depth_cost
    breadth_premium: int             # (N - 1) S re-read prompt tokens
    serial_breadth_time: Quantity    # breadth when one stream runs at a time
    grown_premium: Optional[int] = None          # premium at a grown prompt
    grown_premium_cost: Optional[Quantity] = None


class BestOfNPoolResult(SolverResult):
    """Result from BestOfNPoolModel: pass@N, parallel pool against sequential retry, mixed workload."""
    pass_at: Dict[int, float]        # pool size -> 1 - (1 - p)^N
    last_gain: float                 # what the largest pool's last candidate adds
    parallel_per_success: float      # candidates per success, parallel pool of the largest size
    sequential_candidates: float     # expected candidates, stop at the first pass, same cap
    sequential_per_success: float    # reduces to 1 / p
    cost_ratio: float                # parallel / sequential candidates per success
    mixed_average_rate: float        # the workload's average single-sample rate
    mixed_plug_in: float             # pass@N at the averaged rate (wrong)
    mixed_true: float                # pass@N per task, then averaged
    easy_pass_at_max: float          # the easy half's pass@N


class PassAtKEstimateResult(SolverResult):
    """Result from PassAtKEstimateModel: unbiased subset estimate against the plug-in."""
    subset_estimate: float
    plug_in: float


class CoverageVsVoteResult(SolverResult):
    """Result from CoverageVsVoteModel: pool coverage against majority-vote accuracy."""
    coverage_small: float            # at the largest pool
    coverage_large: float
    coverage_small_first: float      # at one sample
    coverage_large_first: float
    vote_small: float
    vote_large: float
    vote_gain_small: float           # vote gain from the knee to the largest pool, small model
    vote_gain_after_knee: float      # the larger of the two models' vote gains
    coverage_gain_small: float       # coverage gain over the same range
    repair_first: float              # repository-repair coverage at one sample
    repair_max: float                # at the repair pool size


class VerifiedPoolResult(SolverResult):
    """Result from VerifiedPoolModel: first-passing selection from a fixed pool under a fallible check."""
    valid_pass: float                # v (1 - eps_FR)
    invalid_pass: float              # (1 - v) eps_FA
    pass_probability: float          # q, a candidate passes
    precision: float                 # passing-set precision
    defect_given_selection: float
    any_selection: float             # 1 - (1 - q)^n
    no_selection: float              # abstention
    wrong_selection: float           # defective selection per attempt
    any_false_pass: float            # the batch holds at least one false pass
    expected_valid_passes: float
    expected_invalid_passes: float
    expected_passes: float
    conditional_loss: Quantity       # loss per selection
    task_loss: Quantity              # loss per attempt
    output_bill: Quantity
    loss_to_bill: float


class ArgmaxNoiseResult(SolverResult):
    """Result from ArgmaxNoiseModel: the winner's scoring error and a loophole's presence in a pool."""
    expected_max: Dict[int, float]   # pool size -> E[max of N standard normals]
    asymptote: Dict[int, float]      # pool size (N > 1) -> sqrt(2 ln N)
    exploit_presence: float          # 1 - (1 - rate)^N at the large pool


class ScopedRevisionResult(SolverResult):
    """Result from ScopedRevisionModel: output of a full against a scoped regeneration."""
    full_output: int
    scoped_output: int
    output_share: float
    avoided_share: float


class PrunedSearchTreeResult(SolverResult):
    """Result from PrunedSearchTreeModel: tree size with and without a step check."""
    effective_branching: float       # b (1 - r)
    critical_rejection: float        # 1 - 1/b: the tree dies out above it
    unpruned_nodes: int
    pruned_nodes: float
    unpruned_leaves: int
    pruned_leaves: float
    node_saving: float
    flat_cut_nodes: float            # the unpruned count cut by r once (a reference)
    wide_leaves: Optional[int] = None


class SearchAccountsResult(SolverResult):
    """Result from SearchAccountsModel: flat best-of-N against beam search, four accounts."""
    token_time: Quantity
    frontier_counts: List[int]       # extensions generated per beam stage
    beam_segments: int
    retained: int
    flat_output_tokens: int
    beam_output_tokens: int
    flat_prompt_tokens: int
    beam_prompt_tokens: int
    beam_common_prompt_tokens: int   # the task prompt alone, once per extension
    scorer_tokens: int
    flat_waves: int
    beam_waves: int
    flat_test_waves: int
    beam_test_waves: int
    beam_scoring_time: Quantity
    flat_elapsed: Quantity
    beam_elapsed: Quantity
    flat_token_cost: Quantity
    beam_token_cost: Quantity
    output_saving: float
    output_saving_cost: Quantity
    scorer_cost: Quantity
    input_premium_cost: Quantity
    breakeven_input_share: float     # input price fraction at which the bills match


class DeadlinePolicyResult(SolverResult):
    """Result from DeadlinePolicyModel: one candidate, a parallel pool, and feedback under one deadline."""
    token_time: Quantity
    round_time: Quantity
    rounds: int                      # feedback rounds that fit the deadline (capped)
    requested_time: Quantity         # the requested rounds, deadline ignored
    revision_input: int
    a_pass: float
    b_pass: float
    c_pass: float
    c_rounds: float                  # expected rounds of the feedback policy
    a_input: int
    b_input: int
    c_input: float
    a_output: int
    b_output: int
    c_output: float
    a_cost: Quantity
    b_cost: Quantity
    c_cost: Quantity
    c_time: Quantity
    max_c_time: Quantity
    a_cost_per_pass: Quantity
    b_cost_per_pass: Quantity
    c_cost_per_pass: Quantity
    a_output_per_pass: float
    b_output_per_pass: float
    c_output_per_pass: float
    b_over_a: float


class RecordedRunCostResult(SolverResult):
    """Result from RecordedRunCostModel: one public agent run's recorded spend per resolved task."""
    tasks: int
    resolved: int
    resolve_rate: float
    cost_per_attempt: float          # dollars
    cost_per_resolved: float
    unresolved_share: float          # share of spend on attempts that failed
    input_output_ratio: float
    median_calls: float
