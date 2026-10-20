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
    "SeriesFailureResult",
    "CheckpointIntervalResult",
    "RecoveryBudgetResult",
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
    # Fleet orchestration (added 2026-10-20)
    "JobSizeMixResult",
    "PreemptionTaxResult",
    # Context engineering (added 2026-10-20)
    "ContextBudgetResult",
    "AttentionDilutionResult",
    "ComprehensionByLengthResult",
    "PrefixLayoutBillResult",
    "CompactionCadenceResult",
    "ContextInvalidationResult",
    "ProtectedFieldCheckResult",
    # learned proposers (added 2026-10-20)
    "VlaEvaluationResult",
    "DecoderFamiliesResult",
    "ProposerTrafficResult",
    "EvidenceHorizonResult",
    # Fleet goodput (added 2026-10-20)
    "GoodputResult",
    "RuntimeGoodputResult",
    # Retrieval from long-term memory (added 2026-10-20)
    "RetrievalPayloadResult",
    "Bm25PairResult",
    "ExactScanFloorResult",
    "AnnFootprintResult",
    "GraphNeighborhoodResult",
    "IndexMaintenanceResult",
    "StorageAmplificationResult",
    "MaintainedPolicyCostResult",
    "WordOverlapResult",
    # Tool calling (added 2026-10-20)
    "BlindRetryResult",
    "ObservationCarryResult",
    "ModelPollingResult",
    "ToolCatalogSelectionResult",
    "ToolkitGranularityResult",
    "IntegrationAdaptersResult",
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


class SeriesFailureResult(SolverResult):
    """Result from SeriesFailureModel: a job that stops when any one component fails.

    Rates are per hour; ``component_rates`` is each component's share per device.
    """
    component_rates: Dict[str, float]
    device_rate: float
    system_rate: float
    mtbf: Quantity
    failures_per_day: float


class CheckpointIntervalResult(SolverResult):
    """Result from CheckpointIntervalModel: the Young-Daly interval and its waste."""
    write_time: Quantity
    mtbf: Quantity
    interval: Quantity
    write_share: float
    rework_share: float
    waste: float


class RecoveryBudgetResult(SolverResult):
    """Result from RecoveryBudgetModel: one cold restart, and its daily cost at a failure rate."""
    detect: Quantity
    restart: Quantity
    reload: Quantity
    warmup: Quantity
    total: Quantity
    failure_interval: Quantity
    failures_per_day: float
    lost_per_day: Quantity
    day_share: float
    gpu_hours_per_day: float


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
    delay_ceiling: Optional[Quantity] = None  # longest constant-speed delay whose stop, with the overhead, fits
    delay_spare: Optional[Quantity] = None    # delay_ceiling - delay

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
    first_horizon_below_target: int      # first horizon whose success falls below the target


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
    truncated_tasks: int = 0             # tasks with at least one truncated output (added 2026-10-20)
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
    gap_median: Optional[float] = None  # percentage points, all models (added 2026-10-20)
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
    pole_frequency: Optional[Quantity] = None  # 1 / (2 pi tau_e), Hz: the winding's current pole


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
    # Exact binomial bound and the campaign a target needs (added 2026-10-20).
    rate_upper_exact: Optional[float] = None         # 1 - (1 - C)^(1/n)
    trials_for_target: Optional[int] = None          # ceil(ln(1 - C) / ln(1 - target))
    trials_for_target_approx: Optional[float] = None  # -ln(1 - C) / target (rule of three)


class ThrottledStageResult(SolverResult):
    """Result from ThrottledStageModel: a compute-bound stage slowed by a lower clock."""
    throttled: Quantity
    added_age: Quantity
    added_travel: Quantity


# --- Fleet orchestration (added 2026-10-20) ---

class JobSizeMixResult(SolverResult):
    """Result from JobSizeMixModel: a cluster trace's jobs and GPU time by job-size class.

    Each class dict holds ``jobs``, ``gpu_seconds``, ``job_share``,
    ``time_share``, and ``mean_gpu_hours`` (GPU-hours per job).
    """
    job_count: float
    gpu_seconds: float
    classes: Dict[str, Dict[str, float]]
    service_cv_lower_bound: float    # spread of GPU time per job, between size rows only
    state_time_shares: Dict[str, float]


class PreemptionTaxResult(SolverResult):
    """Result from PreemptionTaxModel: what one forced preemption, and a day of them, costs."""
    lost_work: Quantity
    restart: Quantity
    reload: Quantity
    warmup: Quantity
    relaunch: Quantity               # restart + reload + warmup, no detection
    per_event: Quantity              # lost work + relaunch
    cost_per_event: float            # USD
    events_per_day: float
    lost_per_day: Quantity
    gpu_hours_per_day: float
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
    # Optional (added 2026-10-20): runs that ended above a recorded cost limit.
    cost_limit: Optional[float] = None       # dollars
    over_limit_runs: Optional[int] = None
    mean_overshoot: Optional[float] = None   # dollars above the limit, over those runs
    max_overshoot: Optional[float] = None
# --- Context engineering (added 2026-10-20) ----------------------------------
# What one agent request carries: a budget by role, the softmax cost of
# admitted keys, measured comprehension by length, a zoned layout's input bill,
# compaction watermarks, invalidation edits, and a protected-field check.

class ContextBudgetResult(SolverResult):
    """Result from ContextBudgetModel: a request budgeted by role, appended against selected."""
    input_limit: int                     # window - output reserve
    fixed_tokens: int                    # system + tools + task state
    evidence_tokens: int                 # input limit - system - tools: state plus observations
    evidence_share: float                # evidence tokens / window
    observation_budget: int              # input limit - fixed tokens
    turns_fit: int                       # appended turns that fit the observation budget
    overflow_turn: int                   # first turn that does not fit
    overflow_observations: int           # observations the overflow turn needs
    digest_turns: int                    # appended turns that fit at the digest size
    max_observation_per_turn: int        # largest per-turn observation that lasts every turn
    selected_request: int                # fixed tokens + selected observations
    append_requests: List[int]           # request of each turn when every result is appended
    append_cumulative: List[int]         # running input sent by appending
    select_cumulative: List[int]         # running input sent by selecting
    append_bill: int                     # input tokens over the turns, appending
    select_bill: int                     # input tokens over the turns, selecting
    bill_ratio: float                    # append bill / select bill
    final_append_request: int            # the appending policy's last request
    final_append_with_reserve: int       # the same plus the output reserve


class AttentionDilutionResult(SolverResult):
    """Result from AttentionDilutionModel: one head's weight on a target among fixed-logit keys."""
    short_weight: float                  # target weight among the short context's keys
    long_weight: float                   # target weight among the long context's keys
    weight_without_lookalikes: Optional[float] = None    # unrelated keys only
    weight_with_lookalikes: Optional[float] = None       # unrelated keys plus look-alike keys
    lookalike_equivalent_keys: Optional[float] = None    # unrelated keys that weigh as much as the look-alikes
    weight_half_unrelated: Optional[float] = None        # half the unrelated keys dropped, look-alikes kept
    weight_no_unrelated: Optional[float] = None          # every unrelated key dropped, look-alikes kept


class ComprehensionByLengthResult(SolverResult):
    """Result from ComprehensionByLengthModel: models scored at every length of a sweep."""
    lengths: List[str]                   # length labels, shortest first
    scored_models: int                   # models with a score at every length
    median: Dict[str, float]             # length -> median score (fraction)
    q25: Dict[str, float]                # length -> 25th percentile
    q75: Dict[str, float]                # length -> 75th percentile
    held_count: int                      # models at or above the bar at the longest length
    lowest_last: float                   # lowest score at the longest length
    highest_last: float                  # highest score at the longest length
    lowest_model: str
    highest_model: str


class PrefixLayoutBillResult(SolverResult):
    """Result from PrefixLayoutBillModel: input bill of a trajectory, volatile first against zoned."""
    request_tokens: int                  # root + trunk + leaf, every call
    total_input: int                     # request tokens x calls
    warm_calls: int                      # calls that change only the leaf
    uncached_tokens: int                 # zoned layout's prefill
    cached_tokens: int                   # zoned layout's reused prefix tokens
    hit_rate: float                      # cached / total input
    unstable_cost: Quantity              # USD, no prefix reused
    stable_cost: Quantity                # USD, zoned layout
    cost_ratio: float                    # unstable / stable
    prefill_ratio: float                 # total input / uncached tokens
    unstable_cost_per_call: Quantity
    stable_cost_per_call: Quantity
    rewrite_tokens: int                  # trunk + leaf: the prefill of one trunk rewrite
    rewrite_in_leaf_calls: float         # rewrite tokens / leaf tokens


class CompactionCadenceResult(SolverResult):
    """Result from CompactionCadenceModel: passes between two watermarks over a turn budget."""
    input_limit: int                     # window - output reserve
    cycle_turns: int                     # turns between passes
    trigger_length: int                  # request length when a pass fires
    overshoot: int                       # trigger length - trigger watermark
    length_with_reserve: int             # trigger length + output reserve
    fits_window: bool
    passes: int                          # passes within the turn budget
    lengths: List[int]                   # request length after each turn's append, before any pass
    alternate_cycle_turns: Optional[int] = None    # the same at the alternate target
    alternate_passes: Optional[int] = None
    alternate_lengths: Optional[List[int]] = None
    frequency_ratio: Optional[float] = None        # cycle / alternate cycle


class ContextInvalidationResult(SolverResult):
    """Result from ContextInvalidationModel: remove, replace, or append a notice for stale text."""
    removed_total: int                   # resident tokens after removal
    removed_suffix: int                  # tokens after the reused prefix
    replaced_total: int                  # resident tokens after an in-place marker
    replaced_suffix: int
    appended_total: int                  # resident tokens after an appended notice
    appended_suffix: int
    suffix_ratio: float                  # replaced suffix / appended suffix


class ProtectedFieldCheckResult(SolverResult):
    """Result from ProtectedFieldCheckModel: cost per accepted task with and without a check."""
    exposures: int                       # constraints x rounds
    survival_unchecked: float            # every constraint survives every round
    acceptance_unchecked: float
    cost_per_accepted_unchecked: float   # USD
    undetected_loss: float               # per-round loss the check misses
    survival_checked: float
    acceptance_checked: float
    attempt_cost_checked: float          # USD: attempt plus one check per round
    cost_per_accepted_checked: float     # USD
    checked_rounds_below_floor: int      # first round count at which checked survival falls below the floor
    max_loss_for_target: Optional[float] = None    # per-round loss that holds the target survival
# --- Learned proposers: VLA evaluation, decoders, bus traffic, evidence (added 2026-10-20) ---

class VlaEvaluationResult(SolverResult):
    """Result from VlaEvaluationModel: one fresh VLA evaluation and its roofline coordinates."""
    weight_memory: Quantity                  # stored weights at the evaluated precision
    sustained_bandwidth: Quantity            # peak bandwidth x bus efficiency
    achieved_rate: Quantity                  # 16-bit peak x compute utilization
    ridge_intensity: Quantity                # 16-bit peak / peak bandwidth
    prefill_tokens: int
    decode_steps: int
    prefill_linear_flops: Quantity           # 2 N S
    prefill_attention_flops: Quantity        # the S^2 attention term
    prefill_flops: Quantity
    prefill_intensity: Quantity              # linear FLOPs per weight byte
    decode_intensity: Quantity               # 2 N FLOP per weight sweep, per weight byte
    t_prefill_compute: Quantity
    t_weight_sweep: Quantity
    weight_sweep_rate: Quantity              # 1 / t_weight_sweep: the weight-streaming ceiling
    t_prefill: Quantity
    prefill_bound: str                       # "compute" or "memory"
    t_decode_step: Quantity
    t_decode: Quantity
    t_total: Quantity
    f_max: Quantity
    prefill_share: float                     # t_prefill / t_total
    prefill_ceiling: Quantity                # 1 / t_prefill: no weight-only format passes it


class DecoderFamiliesResult(SolverResult):
    """Result from DecoderFamiliesModel: one fresh chunk per decoder family on one backbone."""
    binned_tokens: int
    t_prefill: Quantity                      # prefill only, shared by every family
    t_binned: Quantity
    t_fast: Quantity
    t_parallel: Quantity
    expert_weights: Quantity
    kv_per_token: Quantity                   # full-width keys and values, all layers
    prefix_kv: Quantity                      # cached KV of the input tokens
    t_expert_step_bandwidth: Quantity
    t_expert_step_compute: Quantity
    t_expert_step: Quantity
    t_expert: Quantity
    binned_over_expert: float
    prefill_over_period: float               # shared prefill / renewal period


class ProposerTrafficResult(SolverResult):
    """Result from ProposerTrafficModel: two proposers' weight traffic on one memory bus."""
    sustained_bandwidth: Quantity
    intent_weights_fp16: Quantity
    intent_weights_deployed: Quantity
    chunk_weights: Quantity                  # chunk policy, 16-bit
    intent_sweep_fp16: Quantity
    intent_sweep_rate_fp16: Quantity         # the 16-bit weight-streaming ceiling
    intent_sweep_deployed: Quantity
    intent_sweep_rate_deployed: Quantity
    amortized_sweep: Quantity                # 16-bit sweep / H
    chunk_sweep: Quantity
    chunk_sweep_rate: Quantity
    chunk_sweep_share: float                 # chunk sweep x renewal rate
    tick_over_sweep_rate: float              # permission tick / 16-bit sweep rate
    setpoint_rate: Quantity
    single_step_demand: Quantity             # one 16-bit sweep per setpoint
    single_step_overload: float              # that demand / sustained bandwidth
    chunked_demand: Quantity                 # one sweep per chunk of H
    chunked_share: float
    renewal_gap: float                       # intent latency in chunk renewal periods
    sweeps_per_evaluation: int
    intent_traffic: Quantity
    chunk_traffic: Quantity
    proposer_traffic: Quantity
    proposer_share: float
    decode_duty: float                       # intent decode time x intent rate


class EvidenceHorizonResult(SolverResult):
    """Result from EvidenceHorizonModel: a tracked target's evidence horizon against the chunk schedule."""
    margin: Quantity                         # tolerance - initial error
    untracked_deadline: Quantity             # margin / target speed
    horizon: Quantity                        # sqrt(2 margin / a_slip)
    renewal_period: Quantity
    age_at_replacement: Quantity             # renewal period + inference
    drift_at_replacement: Quantity
    horizon_spare: Quantity                  # horizon - age at replacement
    fits: bool
    min_supply_setpoints: int
    chunk_span: Quantity
    drift_at_chunk_end: Quantity
    error_at_chunk_end: Quantity             # initial error + drift at chunk end
    outlast: Quantity                        # chunk span - horizon
    setpoints_within_horizon: float


# ── Fleet goodput (added 2026-10-20) ──────────────────────────────────────────

class GoodputResult(SolverResult):
    """Result from GoodputModel: ML productivity goodput S x R x P and where the rest went.

    Shares are of offered capacity-time; each lost interval is charged at the
    first boundary it fails (scheduling, then runtime, then program).
    """
    mpg: float                               # S x R x P
    job_product: float                       # R x P, the job's own share of held time
    after_scheduling: float                  # S
    after_runtime: float                     # S x R
    scheduling_share: float                  # 1 - S
    runtime_share: float                     # S (1 - R)
    program_share: float                     # S R (1 - P)
    point_value: Dict[str, float]            # one point of a factor = product of the other two
    ceiling_ratio: Dict[str, float]          # ceiling / current, per factor with a ceiling
    ceiling_gain: Dict[str, float]           # MPG gained at each factor's ceiling, the others held
    mpg_at_ceiling: Dict[str, float]
    full_scheduling_gain: float              # MPG gained by a perfect S
    cost_multiple: float                     # 1 / MPG, price of a useful unit over the raw price
    job_cost_multiple: float                 # 1 / (R x P), a chargeback on held hours only
    runtime_headroom: float                  # 1 / R - 1, the largest relative gain left in R
    useful_rate: Optional[Quantity] = None   # device-hour rate / MPG


class RuntimeGoodputResult(SolverResult):
    """Result from RuntimeGoodputModel: held time a job keeps as retained steps, term by term.

    Every ``*_share`` is a share of held time; the five add to ``runtime_loss``
    (first-order: each term small, so overlaps are neglected).
    """
    mtbf: Quantity                           # system failure interval
    write_time: Quantity                     # one checkpoint write
    interval: Quantity                       # Young-Daly checkpoint interval
    write_share: float
    rework_share: float
    checkpoint_share: float                  # write + rework at the interval
    recovery_time: Quantity                  # detect + restart + reload + warmup
    recovery_share: float                    # recovery_time / MTBF (rework counted in checkpoints)
    startup_time: Quantity                   # restart + reload + warmup, no detection
    startup_share: float                     # startup_time / held window
    input_wait_share: float
    preemption_cost: Quantity                # lost work + startup, per forced preemption
    preemption_share: float                  # rate x cost
    failure_side: float                      # checkpoints + recovery
    runtime_loss: float
    runtime_goodput: float
    interruption_interval: Quantity          # MTBF with forced preemptions counted as interruptions
    interruption_checkpoint_interval: Quantity
# --- Serving at scale: traces, frontiers, admission (added 2026-10-20) --------

class RequestTraceResult(SolverResult):
    """Result from RequestTraceModel: arrivals and token lengths of a request trace."""
    requests: int
    duration: Quantity                       # first to last arrival
    arrival_rate: Quantity                   # requests / duration
    median_prompt: float
    median_output: float
    mean_prompt: float
    mean_output: float
    max_output: float
    batch_size: int
    fixed_batch_idle_share: float            # mean of 1 - mean/max over consecutive batches
    per_second_mean: float                   # arrivals per one-second bin, empty seconds included
    per_second_peak: int                     # busiest one-second bin


class TrafficProfileResult(SolverResult):
    """Result from TrafficProfileModel: request rate over a binned week."""
    peak_rate: Quantity
    min_rate: Quantity
    weekday_trough_low: Quantity             # lowest and highest nightly minimum, weekdays
    weekday_trough_high: Quantity
    weekday_peak_low: Quantity               # lowest and highest daily maximum, weekdays
    weekday_peak_high: Quantity
    weekday_peak_to_trough: float            # mean over weekdays of max / min


class ServingFrontierResult(SolverResult):
    """Result from ServingFrontierModel: one accelerator's measured frontier and its best rate at a target."""
    frontier: List[Tuple[float, float]]      # (interactivity, throughput), no point beaten on both
    peak_interactivity: float                # interactivity of the highest-throughput point
    peak_throughput: float
    best_at: float                           # highest throughput among rows meeting the target


# --- KV cache management: footprint, paging, prefix reuse, pauses, residency (added 2026-10-20) ---
# KvAdmissionResult: copied verbatim from the v2-10 inference branch (same name and fields).
class KvAdmissionResult(SolverResult):
    """Result from KvAdmissionModel: requests of one context length that fit a replica's KV budget."""
    head_dim: int
    weights: Quantity
    hbm_total: Quantity
    kv_budget: Quantity                      # HBM - weights - reserve
    kv_per_token: Quantity
    kv_per_request: Quantity
    max_requests: int                        # floor(kv_budget / kv_per_request)


class PrecisionDividendResult(SolverResult):
    """Result from PrecisionDividendModel: maximum batch before and after lower precision."""
    weights_per_device: Quantity
    weights_per_device_low: Quantity
    kv_budget_per_device: Quantity
    kv_budget_per_device_low: Quantity
    kv_per_request: Quantity                 # whole request at the base KV precision
    kv_per_request_low: Quantity
    kv_per_request_per_device: Quantity
    kv_per_request_per_device_low: Quantity
    max_batch: int
    max_batch_low: int
    dividend: float                          # max_batch_low / max_batch
    budget_gain: float                       # KV budget low / KV budget base


class BallsIntoBinsResult(SolverResult):
    """Result from BallsIntoBinsModel: simulated busiest-bin load, n balls into n bins."""
    bins: List[int]
    max_load: Dict[int, List[float]]         # choices -> mean busiest-bin load per bin count


__all__ += [
    "RequestTraceResult",
    "TrafficProfileResult",
    "ServingFrontierResult",
    "KvAdmissionResult",
    "PrecisionDividendResult",
    "BallsIntoBinsResult",
]
# --- Retrieval from long-term memory (added 2026-10-20) ------------------------
# Staging whole records in a token allowance, BM25 ranking, the exact-scan
# bandwidth floor, approximate-index bytes, typed traversal depth, keeping an
# index current, auxiliary bytes per store, and word overlap.

class RetrievalPayloadResult(SolverResult):
    """Result from RetrievalPayloadModel: whole serialized records in a retrieval allowance."""
    allowance: int                       # window - output reserve - other input
    record: int                          # body + metadata, tokens per serialized record
    records: int                         # whole records that fit after the envelope
    used: int                            # envelope + records x record
    remaining: int                       # allowance - used
    body_only: int                       # records a body-only count admits
    body_only_used: int                  # those records serialized with metadata and envelope
    overshoot: int                       # body_only_used - allowance
    overcount: int                       # body_only - records
    scaled_metadata: Optional[int] = None          # metadata after the stated scale
    scaled_metadata_records: Optional[int] = None  # whole records at that metadata size


class Bm25PairResult(SolverResult):
    """Result from Bm25PairModel: one query term scored in a short and a long document."""
    idf: float
    short_normalizer: float              # k1 (1 - b + b |D| / avgdl), short document
    long_normalizer: float
    short_score: float
    long_score: float


class ExactScanFloorResult(SolverResult):
    """Result from ExactScanFloorModel: the bandwidth floor of an exact vector scan."""
    footprint: Quantity                  # bytes streamed per scan
    bytes_per_vector: Quantity
    flops: Quantity                      # 2 N d
    intensity: Quantity                  # FLOP per byte
    ridge: Quantity                      # host peak / bandwidth, FLOP per byte
    stream_time: Quantity                # footprint / bandwidth
    math_time: Quantity                  # flops / peak
    trajectory_time: Optional[Quantity] = None     # searches x stream time
    filtered_time: Optional[Quantity] = None       # stream time of the filtered set
    budget_vectors: Optional[float] = None         # vectors the budget can stream


class AnnFootprintResult(SolverResult):
    """Result from AnnFootprintModel: resident bytes of a graph index and of PQ codes."""
    raw: Quantity                        # full-precision vectors
    graph: Quantity                      # graph links
    graph_total: Quantity                # vectors + links
    codes: Quantity                      # PQ codes
    compression: float                   # raw / codes
    lookup_table: Quantity               # subvectors x centroids x entry
    sub_dim: int                         # components per subvector
    bytes_per_vector: Quantity           # full precision
    code_bytes_per_vector: Quantity


class GraphNeighborhoodResult(SolverResult):
    """Result from GraphNeighborhoodModel: an exact tree's neighborhood against an allowance."""
    allowance: int
    nodes_by_depth: List[int]            # cumulative nodes through depth 0, 1, ..., depth
    tokens_by_depth: List[int]           # serialized tokens through each depth
    nodes: int                           # through the stated depth
    required: int                        # tokens through the stated depth
    fit: int                             # most whole nodes the allowance holds
    packed: int                          # tokens of that packing
    shallow_nodes: int                   # one level less
    shallow_required: int


class IndexMaintenanceResult(SolverResult):
    """Result from IndexMaintenanceModel: re-indexing an edit against excluding stale entries."""
    chunks: int
    batches: int
    local_time: Quantity                 # parse + postings + local embedding batches
    remote_batch_time: Quantity          # parse + postings + one remote request per batch
    remote_per_file_time: Quantity       # parse + postings + one remote request per file
    exclusion_write_time: Quantity       # one exclusion entry per changed file
    exclusion_check_time: Quantity       # one lookup per candidate
    write_path_ratio: float              # local re-index / exclusion write


class StorageAmplificationResult(SolverResult):
    """Result from StorageAmplificationModel: auxiliary bytes and re-index time of two stores."""
    raw: Quantity                        # source bytes
    postings: Quantity
    graph: Quantity                      # structural edges
    chunks: int
    vectors: Quantity
    vector_links: Quantity               # graph-index links over the vectors
    aux_lexical_graph: Quantity          # postings + edges
    aux_lexical_dense: Quantity          # postings + vectors + links
    ratio_lexical_graph: float           # auxiliary / source
    ratio_lexical_dense: float
    edited_chunks: int
    reindex_lexical_graph: Quantity
    reindex_lexical_dense: Quantity
    bytes_ratio: float                   # dense store / graph store, auxiliary bytes
    time_ratio: float                    # dense store / graph store, re-index time


class MaintainedPolicyCostResult(SolverResult):
    """Result from MaintainedPolicyCostModel: cost per accepted task with and without upkeep."""
    baseline_cost: float                 # cost per accepted task, units of the baseline attempt
    maintained_cost: float
    change: float                        # maintained / baseline - 1
    break_even_acceptance: float         # acceptance at which the two match


class WordOverlapResult(SolverResult):
    """Result from WordOverlapModel: distinct words two texts share."""
    shared: int
    union: int
    first_words: int
    second_words: int
    jaccard: float                       # shared / union

class KvAttentionDesignResult(SolverResult):
    """Result from KvAttentionDesignModel: bytes per token by attention design."""
    per_token: Dict[str, Quantity]           # model name -> KV bytes per token
    design: Dict[str, str]                   # model name -> MHA, GQA, or MLA
    mha_over_mla: float
    gqa_over_mla: float
    gqa_session: Quantity                    # one session of session_tokens, GQA model
    mla_session: Quantity


class PagedAllocationResult(SolverResult):
    """Result from PagedAllocationModel: full-cap reservation against paged blocks."""
    reservation: Quantity                    # cap x bytes per token
    reservations: int                        # full-cap reservations in the pool
    final_length_efficiency: float           # mean final length / cap
    small_block: Quantity                    # bytes of one small block
    large_block: Quantity
    small_tail_tokens: int                   # worst-case unused slots, small blocks
    large_tail_tokens: int
    small_tail: Quantity
    large_tail: Quantity
    large_tail_share: float                  # worst large tail / cap
    small_allocated_tokens: float            # expected tokens held per request
    small_waste: float                       # expected internal fragmentation
    large_allocated_tokens: float
    large_waste: float
    paged_fit: int                           # final-length snapshots, small blocks
    paged_gain: float                        # paged fit / reservations


class PrefixRebuildResult(SolverResult):
    """Result from PrefixRebuildModel: the prefill a prefix hit avoids, against one suffix pass."""
    n_params: float
    state: Quantity                          # KV bytes of the prefix
    linear_ops: Quantity                     # 2 P S
    attention_ops: Quantity                  # 2 L d S^2 (causal)
    sustained_rate: Quantity                 # accelerators x peak x utilization
    rebuild_time: Quantity                   # (linear + attention) / rate
    weight_read_time: Quantity               # weights / node bandwidth
    suffix_ops_time: Quantity                # 2 P S_suffix / rate
    rebuild_over_hit: float                  # rebuild time / max(weight read, suffix ops)


class ChunkedPrefillResult(SolverResult):
    """Result from ChunkedPrefillModel: decode gap with a monolithic prefill against chunks."""
    decode_step: Quantity                    # decode-only iteration
    monolithic_prefill: Quantity             # the whole prompt in one iteration
    monolithic_gap: Quantity                 # decode step + monolithic prefill
    chunks: int
    chunked_iteration: Quantity              # one chunk beside the decodes
    chunked_done: Quantity                   # chunks x iteration
    gap_ratio: float                         # monolithic gap / chunked iteration


class PausedKvStateResult(SolverResult):
    """Result from PausedKvStateModel: keep, move, or drop a paused request's KV state."""
    blocks: int
    state: Quantity                          # uniquely reclaimable KV bytes
    link: Quantity                           # accelerators x host link
    copy_out: Quantity
    copy_in: Quantity
    round_trip: Quantity                     # copy out + copy in
    recompute: Quantity                      # 2 P S / rate, weight work only
    recompute_over_reload: float
    recompute_over_reload_derated: float     # at the derated link fraction
    held_occupancy: Quantity                 # state x wait (bytes x time)
    released_on_demand: Quantity             # wait - copy out
    released_prefetch: Quantity              # wait - copy out - copy in
    burst: Quantity                          # returns x state
    burst_time: Quantity                     # burst / link
    burst_slowdown: float                    # burst time / one reload


class KvBlockSharingResult(SolverResult):
    """Result from KvBlockSharingModel: block-rounded resident fit, shared prefix, and turn rate."""
    block_bytes: Quantity
    usable_blocks: int                       # floor((1 - headroom) pool / block)
    usable: Quantity
    unshared_blocks: int                     # ceil((context + output) / block)
    common_blocks: int
    private_blocks: int
    unshared_bytes: Quantity
    common_bytes: Quantity
    private_bytes: Quantity
    fit_unshared: int
    fit_shared: int
    retained_time: Quantity                  # generation + wait
    retained_rate: Quantity                  # fit unshared / retained time
    released_rate: Quantity                  # fit unshared / generation
    shared_retained_rate: Quantity           # fit shared / retained time
    shared_released_rate: Quantity           # fit shared / generation
    residency_factor: float                  # 1 + wait / generation
    shared_fraction: float                   # common blocks / unshared blocks
    share_gain: float                        # fit shared / fit unshared
    share_bound: float                       # 1 / (1 - f)


class KvResidencyArchetypeResult(SolverResult):
    """Result from KvResidencyArchetypeModel: sharing, residency tax, and resume cost for one workload."""
    shared_fraction: float                   # shared prefix / context
    fit_unshared: int
    fit_shared: int
    share_gain: float
    residency_factor: float
    turn_cycle: Quantity                     # generation + wait
    recompute: Quantity
    reload: Quantity
    recompute_share: float                   # recompute / turn cycle
    shared_retained_rate: Quantity           # fit shared / turn cycle
    shared_released_rate: Quantity           # fit shared / generation
# --- The real-time permission path of an embodied machine (added 2026-10-20) ---

class PhaseLagBudgetResult(SolverResult):
    """Result from PhaseLagBudgetModel: delay terms of a sampled loop and the phase they spend at crossover."""
    hold_delay: Quantity                     # zero-order hold, half a sample period
    total_delay: Quantity                    # hold + compute + transport
    phase_hold: Quantity                     # degrees at crossover
    phase_compute: Quantity
    phase_transport: Quantity
    phase_total: Quantity
    phase_per_delay: Quantity                # degrees per millisecond of delay at crossover
    allowed_delay: Optional[Quantity] = None  # largest total delay the allowance absorbs
    delay_margin: Optional[Quantity] = None   # allowed_delay - total_delay (negative when it fails)
    crossover_ceiling: Optional[Quantity] = None  # highest crossover the total delay supports
    over_allowance: Optional[float] = None   # phase_total / phase_allowance
    fits: Optional[bool] = None


class ResponseTimeResult(SolverResult):
    """Result from ResponseTimeModel: R = C + B + I of one periodic task against its deadline."""
    wcet: Quantity
    blocking: Quantity
    interference: Quantity
    deadline: Quantity
    response: Quantity
    margin: Quantity                         # deadline - response (negative when it misses)
    meets_deadline: bool
    blocking_limit: Quantity                 # largest blocking that still meets the deadline
    wcet_claim: Optional[Quantity] = None    # declared WCET above the observed unloaded run


class CanFdBusResult(SolverResult):
    """Result from CanFdBusModel: serialization occupancy of a cyclic CAN FD axis exchange."""
    frame_time: Quantity                     # one frame, both phases
    nominal_time: Quantity                   # arbitration-phase share of one frame
    data_time: Quantity                      # data-phase share of one frame
    bits_nominal: int
    bits_data: int
    stuff_bits: int
    frames: int                              # frames per cycle
    occupancy: Quantity                      # frames x frame_time
    utilization: float                       # occupancy / cycle
    unallocated: Quantity                    # cycle - occupancy (negative when it overruns)
    fits: bool
    axes_capacity: int                       # most axes whose frames fit one cycle


class EtherCatCycleResult(SolverResult):
    """Result from EtherCatCycleModel: one summation frame through a chain of drives."""
    frame_time: Quantity                     # serialization of the one frame
    forward_time: Quantity                   # per-node forwarding, all nodes
    occupancy: Quantity                      # frame_time + forward_time
    utilization: float
    unallocated: Quantity
    fits: bool


class LeaseBudgetResult(SolverResult):
    """Result from LeaseBudgetModel: a chunk lease priced in the stopping budget and bounded by the lease rule."""
    lease: Quantity
    lease_path: Quantity                     # lease + tick + bus: last admission to the stop command
    lease_path_travel: Quantity              # travel at speed along the lease path
    admission_to_brake_onset: Quantity       # lease path + brake onset
    admission_to_brake_onset_travel: Quantity
    body_terms_distance: Quantity            # braking + overhead + onset travel
    running_total: Quantity                  # body terms + lease path travel
    spare: Quantity                          # clear distance - running total
    delay_ceiling: Quantity                  # longest pre-brake delay whose stop fits
    delay_spare: Quantity                    # delay ceiling - admission to brake onset
    lease_floor: Quantity                    # renewal period + renewal jitter
    lease_ceiling: Quantity                  # delay ceiling - tick - bus - brake onset
    lease_ceiling_after_age: Quantity        # ... - observation age at dispatch
    lease_ceiling_at_admission: Quantity     # ... - evidence age at admission
    fits: bool                               # floor < lease <= ceiling at admission
    stall_last_admission: Quantity           # second renewal completes, then the proposer stalls
    stall_lease_expiry: Quantity
    stall_brake_onset: Quantity
    renewal_lateness_limit: Optional[Quantity] = None  # lease - renewal period: a renewal this late lapses the lease (added 2026-10-20)


class LatencyTailResult(SolverResult):
    """Result from LatencyTailModel: the tail of a measured latency sample."""
    count: int
    p50: Quantity
    p99: Quantity
    p999: Quantity
    maximum: Quantity
    over_threshold: Optional[int] = None     # samples longer than the stated threshold


# --- Tool calling (added 2026-10-20) ------------------------------------------

class BlindRetryResult(SolverResult):
    """Result from BlindRetryModel: timeouts and duplicate effects under one blind retry."""
    timeout_probability: float           # 1 - (1 - q)(1 - r)
    duplicate_probability: float         # (1 - q) r (1 - q)
    calls: int                           # calls in the period
    timeouts: float                      # expected timeouts in the period
    duplicates: float                    # expected duplicate effects in the period


class ObservationCarryResult(SolverResult):
    """Result from ObservationCarryModel: a raw log against the shaped view later calls carry."""
    raw_tokens: Quantity                 # raw bytes / bytes per token
    raw_windows: float                   # raw tokens / window
    shaped_share: float                  # shaped tokens / window
    carried_tokens: Quantity             # shaped tokens x later calls
    carried_cost_uncached: Quantity      # USD
    carried_cost_cached: Quantity        # USD, at the cached share of the price


class ModelPollingResult(SolverResult):
    """Result from ModelPollingModel: status calls a model makes while a job is pending."""
    polls: int                           # ceil(T / dt) - 1
    input_tokens: int
    output_tokens: int
    cached_price: Quantity               # USD per cached input token
    blended_input_price: Quantity        # USD per input token at the cache hit share
    input_cost: Quantity                 # USD
    output_cost: Quantity                # USD
    cost: Quantity                       # USD, input + output
    worst_delay: Quantity                # one poll interval


class ToolCatalogSelectionResult(SolverResult):
    """Result from ToolCatalogSelectionModel: descriptor tokens staged per call, full against selected."""
    full_tokens: int                     # registered x per descriptor
    selected_descriptor_tokens: int      # selected x per descriptor
    selected_tokens: int                 # selected descriptors + overhead
    released_tokens: int                 # full - selected
    unused_descriptor_tokens: int        # descriptors selection leaves out
    break_even_unused: float             # overhead / per descriptor


class ToolkitGranularityResult(SolverResult):
    """Result from ToolkitGranularityModel: structured tools against one shell tool, per accepted task."""
    structured_descriptor_tokens: int    # tools x tokens, per call
    saved_tokens: int                    # per trajectory
    saved_cost: Quantity                 # USD per trajectory
    shell_attempt_cost: Quantity         # structured attempt - savings
    structured_cost_per_accepted: Quantity
    shell_cost_per_accepted: Quantity
    structured_cost_per_accepted_with_loss: Quantity
    shell_cost_per_accepted_with_loss: Quantity
    break_even_change_probability: float  # shell change probability matching structured


class IntegrationAdaptersResult(SolverResult):
    """Result from IntegrationAdaptersModel: pairwise adapters against one shared protocol."""
    pairwise: int                        # hosts x services
    shared: int                          # hosts + services


# --- Agent sandboxes (added 2026-10-20) ---
class SyscallTaxResult(SolverResult):
    """Result from SyscallTaxModel: per-call mediation cost and added time by envelope."""
    container_call: Quantity                 # trap + in-kernel filter
    microvm_call: Quantity                   # native trap in the guest
    userspace_call: Quantity                 # trap + emulation + switch
    container_total: Quantity                # over all ordinary calls
    microvm_total: Quantity
    userspace_total: Quantity
    userspace_vs_microvm: float              # per-call ratio


class EnvelopeAmortizationResult(SolverResult):
    """Result from EnvelopeAmortizationModel: setup paid per call or once per trajectory."""
    useful_work: Quantity                    # calls x work per call
    per_call_setup: Quantity                 # calls x setup
    per_call_total: Quantity                 # fresh envelope per call
    per_trajectory_total: Quantity           # one envelope retained
    setup_saved: Quantity
    reuse_speedup: float                     # per-call total / per-trajectory total
    setup_share: float                       # setup / (setup + work) of a fresh call
    work_for_cap: Quantity                   # work per call that holds setup to the cap


class WorkspaceProvisioningResult(SolverResult):
    """Result from WorkspaceProvisioningModel: full copy against copy-on-write, and first-write granularity."""
    full_copy: Quantity                      # workspaces x base image
    cow_written: Quantity                    # workspaces x bytes written
    full_copy_time: Quantity                 # lower bound at the drive's sequential rate
    cow_time: Quantity
    copy_vs_cow: float                       # base image / bytes written
    file_copy: Quantity                      # workspaces x fixture
    block_copy: Quantity                     # workspaces x one block
    logical_changes: Quantity                # workspaces x bytes modified
    file_vs_block: float                     # fixture / block


class SandboxPoolReserveResult(SolverResult):
    """Result from SandboxPoolReserveModel: active leases, standby reserve, and resident memory."""
    active: float                            # Little's law, arrival rate x hold time
    restore_arrivals: float                  # arrivals during one restore (Poisson mean)
    initial_miss: float                      # miss probability with the initial reserve
    reserve: int                             # smallest reserve meeting the target
    reserve_miss: float
    private_memory: Quantity                 # every active lease with a private image
    pool_memory: Quantity                    # copy-on-write: one image + dirty pages + standby
    memory_ratio: float
    max_restore: Quantity                    # longest restore the reserve tolerates
    slow_arrivals: float                     # arrivals during a slower restore
    slow_reserve: int


class BootTimeCdfResult(SolverResult):
    """Result from BootTimeCdfModel: the median and a tail percentile of a released boot-time CDF."""
    booting: int                             # instances booting at once
    median: Quantity
    tail: Quantity


# --- Neural computation: a dense network's counts, and one training epoch (added 2026-10-20) ---

class DenseNetworkResult(SolverResult):
    """Result from DenseNetworkModel: what one dense network costs at one batch size.

    Counts are plain integers; byte totals and intensity carry units. Per-layer
    lists run from the first layer to the output layer.
    """
    layer_widths: List[int]
    batch_size: int
    layers: int
    weights_per_layer: List[int]
    biases_per_layer: List[int]
    params_per_layer: List[int]
    weights: int
    biases: int
    parameters: int
    macs_per_example: int                    # one MAC per weight
    flops_per_example: int                   # 2 FLOPs per MAC
    first_layer_share: float                 # first layer's weights / all weights (= its share of the MACs)
    matmul_flops_per_layer: List[int]        # 2 B n_in n_out, at the batch
    elementwise_flops_per_layer: List[int]   # bias add + activation, 2 B n_out
    matmul_flops: int
    elementwise_flops: int
    forward_flops: int                       # matmul + elementwise, at the batch
    forward_flops_per_example: float
    matmul_share: float
    elementwise_share: float
    parameter_bytes: Quantity
    traffic_values: int                      # activation values crossing memory, at the batch
    traffic_bytes: Quantity                  # weights once + activation traffic
    intensity: Quantity                      # forward FLOPs / traffic bytes
    weight_gradient_flops: int
    input_gradient_flops: int
    training_step_flops: int                 # forward matmul + both gradient products
    training_flops_ratio: float              # training step / forward matmul
    activation_values_per_layer: List[int]   # B n_l for the input and every layer output
    activation_values_per_example: int       # sum of widths
    saved_activation_values: int             # training keeps all of them, at the batch
    saved_activation_bytes: Quantity
    inference_activation_values: int         # largest adjacent pair, at the batch
    inference_activation_bytes: Quantity
    gradient_bytes: Quantity
    optimizer_bytes: Quantity
    training_memory: Quantity                # weights + saved activations + gradients + optimizer
    inference_memory: Quantity               # weights + the rolling activation pair
    training_memory_ratio: float
    steps_per_epoch: Optional[int] = None    # training examples // batch, when a dataset is given


class EpochOverheadResult(SolverResult):
    """Result from EpochOverheadModel: one training epoch's work and fixed overhead on a host and an accelerator.

    Dictionaries are keyed ``"host"`` and ``"accelerator"``.
    """
    steps: int
    step_flops: Quantity
    ops_per_step: int
    peak: Dict[str, Quantity]
    bandwidth: Dict[str, Quantity]
    ridge: Dict[str, Quantity]
    rate_batch1: Dict[str, Quantity]         # attainable rate at the batch-1 intensity
    rate: Dict[str, Quantity]                # attainable rate at the network's batch
    batching_speedup: Dict[str, float]       # rate / rate_batch1
    work_time: Dict[str, Quantity]
    overhead_time: Dict[str, Quantity]
    total_time: Dict[str, Quantity]
    work_speedup: float                      # host work / accelerator work
    overhead_share_accelerator: float
    movement_to_arithmetic_energy: float     # DRAM energy per FLOP's bytes / FLOP energy, at the batch
    movement_to_arithmetic_energy_batch1: float


__all__ += [
    "DenseNetworkResult",
    "EpochOverheadResult",
]


class ParameterGrowthResult(SolverResult):
    """Result from ParameterGrowthModel: parameter-count fits before and after a split year (added 2026-10-20)."""
    models: List[str]
    domains: List[str]
    parameters: List[float]
    years: List[float]                 # decimal publication years
    log_parameters: List[float]
    split_year: int
    early_slope: float                 # decades of parameters per year, before the split
    early_intercept: float
    late_slope: float
    late_intercept: float
    early_doubling_months: float
    late_doubling_months: float
    largest_parameters: float
    model_count: int


__all__ += ["ParameterGrowthResult"]


# --- Agent security: injection exposure and defense evaluation (added 2026-10-20) ---
class InjectionExposureResult(SolverResult):
    """Result from InjectionExposureModel: per-task payload exposure by read volume."""
    exposure: Dict[str, float]        # 1 - (1 - q)^n per workload
    linear: Dict[str, float]          # q n, the small-exposure estimate
    screened_rate: float              # q m, payload rate behind the screen
    screened: Dict[str, float]        # 1 - (1 - q m)^n per workload
    screened_per_cohort: Dict[str, float]  # screened exposure x cohort tasks


class TurnExposureResult(SolverResult):
    """Result from TurnExposureModel: exposure counted in turns, and the same reads counted as items."""
    per_turn: float                   # 1 - (1 - q)^r
    trajectory: float                 # 1 - (1 - p_turn)^H
    reads: int                        # r H
    same_reads: float                 # 1 - (1 - q)^(r H); equals trajectory


class PolicyComparisonResult(SolverResult):
    """Result from PolicyComparisonModel: attack success and benign acceptance for two policies."""
    baseline_lower: float             # confirmed harm / attacked
    candidate_lower: float
    baseline_upper: float             # (confirmed + unresolved) / attacked
    candidate_upper: float
    reduction_observed: float         # 1 - candidate_lower / baseline_lower
    reduction_upper: float            # both arms at their upper rates
    reduction_least_favorable: float  # candidate upper against baseline lower
    baseline_acceptance: float        # accepted benign / benign
    candidate_acceptance: float
    lost_benign: int
    extra_false_blocks: int
    cost_rise: float                  # p_acc(baseline) / p_acc(candidate) - 1
    baseline_cost_per_accepted: Optional[float] = None   # attempt cost / p_acc, when priced
    candidate_cost_per_accepted: Optional[float] = None


__all__ += [
    "InjectionExposureResult",
    "TurnExposureResult",
    "PolicyComparisonResult",
]

# --- The agent harness (added 2026-10-20) -------------------------------------
# Approval escrow by Little's law, weighted deficit round-robin, bounded-liability
# admission, a trajectory metered against its budget vector, and call or step
# ceilings read from public runs.

class ApprovalEscrowResult(SolverResult):
    """Result from ApprovalEscrowModel: what approval escrow holds for two workloads."""
    held: float                              # repair trajectories in escrow, Little's law
    retained_memory: Quantity                # held x workspace lease, GiB
    remediation_held: float                  # remediation trajectories in escrow
    time_to_mitigate: Quantity               # diagnosis + decision + apply and verify
    approval_share: float                    # decision / time to mitigate


class DeficitRoundRobinResult(SolverResult):
    """Result from DeficitRoundRobinModel: charge shares against call shares, per class."""
    credit_per_round: Dict[str, float]       # weight x quantum
    charge_per_call: Dict[str, float]        # input + output allowance
    calls_per_round: Dict[str, float]        # long-run, backlogged
    rounds_per_call: Dict[str, float]
    charge_share: Dict[str, float]           # weight / total weight
    call_share: Dict[str, float]
    call_ratio: float                        # most calls per round / fewest
    simulated_rounds: int
    simulated_call_share: Dict[str, float]
    simulated_charge_share: Dict[str, float]


class BoundedLiabilityAdmissionResult(SolverResult):
    """Result from BoundedLiabilityAdmissionModel: a spend check against the reservation test (dollars)."""
    input_liability: float
    output_liability: float
    liability: float                         # most the call can bill
    spend_headroom: float                    # allowance - settled
    headroom: float                          # allowance - settled - reserved - finalization reserve
    spend_check_admits: bool
    admitted: bool
    finalization_shortfall: float            # reserve left unfunded if the call were admitted
    input_share: float                       # input liability / liability
    max_input_tokens: float                  # largest input that fits beside the output allowance
    tokens_per_probe: float                  # input tokens one probe amount of unsettled spend costs
    reservation_share: float                 # outstanding reservation / allowance


class TrajectoryBudgetMeterResult(SolverResult):
    """Result from TrajectoryBudgetMeterModel: one trajectory's used fractions, with and without prefix reuse."""
    input_tokens: int
    cached_tokens: int
    uncached_tokens: int
    output_tokens: int
    total_tokens: int
    cost_with_reuse: float                   # dollars
    cost_without_reuse: float
    cost_ratio: float                        # without / with
    used_with_reuse: Dict[str, float]        # turns, tokens, time, dollars
    used_without_reuse: Dict[str, float]
    leading_with_reuse: str                  # dimension with the largest used fraction
    leading_without_reuse: str


class CallCeilingResult(SolverResult):
    """Result from CallCeilingModel: a call ceiling at the accepted runs' quantile, per configuration."""
    quantile: float
    runs: int
    configurations: List[Dict[str, Any]]     # one row per submission, sorted by submission id
    failed_halted_low: float
    failed_halted_high: float
    accepted_halted_high: float
    cap_low: float
    cap_high: float
    median_calls_low: float
    median_calls_high: float
    longer_when_failed: bool                 # failed median above accepted median everywhere


class StepCeilingReturnResult(SolverResult):
    """Result from StepCeilingReturnModel: success of the same model at several enforced step ceilings."""
    steps: List[int]
    scores: Dict[str, Dict[int, float]]      # model -> step ceiling -> percent completed
    gain: Dict[str, float]                   # points from the lowest to the highest ceiling
    flat_after_middle: int                   # models no better at the highest than the middle ceiling
    best_gain: float
    best_model: str


__all__ += [
    "ApprovalEscrowResult",
    "DeficitRoundRobinResult",
    "BoundedLiabilityAdmissionResult",
    "TrajectoryBudgetMeterResult",
    "CallCeilingResult",
    "StepCeilingReturnResult",
]
# --- Sensor perception: capture timing, age, and error scales (added 2026-10-20) ---

class ObservationAgeResult(SolverResult):
    """Result from ObservationAgeModel: a camera observation's age from exposure midpoint to dispatch."""
    stages: List[Tuple[str, Quantity, Quantity]]  # (stage, duration, age after the stage)
    age: Quantity                    # exposure midpoint to dispatch
    age_travel: Quantity
    arrival_offset: Quantity         # exposure midpoint to DMA completion (an arrival stamp's error)
    arrival_travel: Quantity
    trigger_offset: Quantity         # exposure start to midpoint (a trigger stamp's error)
    trigger_travel: Quantity
    exposure_travel: Quantity        # travel during the exposure itself (blur scale)
    capture_before_compute: Quantity  # half the exposure plus the readout: top-row sensor age
    sensor_age_middle_row: Quantity  # half the exposure plus half the readout
    top_row_excess: Quantity         # top-row sensor age minus one frame period
    capture_share: float             # capture_before_compute / age
    backbone_share: float            # backbone stage / age
    frame_period: Quantity
    frame_travel: Quantity           # travel during one frame period (one dropped frame)
    batch_frames: int
    batch_waits: List[Quantity]      # wait of each batched frame but the newest, oldest first
    batch_wait: Quantity             # the oldest frame's wait, (n - 1) frame periods
    batch_wait_travel: Quantity
    batched_age: Quantity            # age + batch_wait
    batched_age_ratio: float


class RollingShutterSkewResult(SolverResult):
    """Result from RollingShutterSkewModel: row timing and the skew a moving camera records."""
    row_interval: Quantity
    midpoint_lag: Quantity           # half the exposure
    skew: Quantity                   # along-track travel between the first and last rows
    skew_over_allowance: float
    lateral_skew: Quantity           # sideways shift of a target at range while turning
    lateral_share: float             # of the side clearance
    exposure_travel: Quantity        # blur: travel during one row's exposure


class TimingAttitudeErrorResult(SolverResult):
    """Result from TimingAttitudeErrorModel: position errors from a timing error and an angle error."""
    timing_travel: Quantity          # v dt
    misorientation: Quantity         # omega dt
    timing_lateral: Quantity         # d omega dt
    angle_lateral: Quantity          # d theta (small angle)
    angle_lateral_share: Optional[float] = None  # of the stated reference distance


class LidarSweepAgeResult(SolverResult):
    """Result from LidarSweepAgeModel: one sweep's motion and a sector's age, streamed or whole."""
    sweep_period: Quantity
    sweep_travel: Quantity
    sweep_rotation: Quantity
    sector_angle: Quantity
    sector_time: Quantity
    age_streamed: Quantity           # sector time + data latency + sync bound
    age_full: Quantity               # sweep period + data latency + sync bound
    travel_streamed: Quantity
    travel_full: Quantity
    slack_streamed: Quantity         # reference age - streamed age


class StereoDepthErrorResult(SolverResult):
    """Result from StereoDepthErrorModel: one-sigma depth error against range and the clearances it spends."""
    sigma_near: Quantity
    sigma_far: Quantity
    sigma_target: Quantity
    range_ratio: float               # far / near
    sigma_growth: float              # sigma_far / sigma_near = range_ratio ** 2
    inset_near: Quantity             # ray clearance + one sigma (the shortcut inset)
    inset_far: Quantity
    range_at_loc_allowance: Quantity
    range_at_side_clearance: Quantity
    range_at_side_clearance_half_baseline: Quantity
    baseline_bias: Quantity


class DelayPhaseMarginResult(SolverResult):
    """Result from DelayPhaseMarginModel: the delay a loop's phase margin can absorb."""
    phase_margin: Quantity           # radians
    crossover_frequency: Quantity    # Hz
    delay_ceiling: Quantity
    delay: Quantity
    share_used: float


class ConformalInsetResult(SolverResult):
    """Result from ConformalInsetModel: the calibration rank behind a split-conformal bound."""
    alpha: float
    calibration_count: int
    min_count: int
    rank: int
    rank_from_top: int


class ClockConversionResult(SolverResult):
    """Result from ClockConversionModel: drift between syncs and the dispatch limit a conversion bound sets."""
    drift_per_interval: Quantity
    dispatch_age_limit: Quantity
    bound_travel: Quantity


class VisibilityRangeResult(SolverResult):
    """Result from VisibilityRangeModel: the range at which fog erases contrast."""
    visibility_range: Quantity


class ThermalExpansionResult(SolverResult):
    """Result from ThermalExpansionModel: a heated member's length change."""
    expansion: Quantity
