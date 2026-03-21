"""Typed result models for all mlsysim models and solvers.

Layer A of the composition architecture: every resolver returns a typed
Pydantic model instead of Dict[str, Any].  This gives students
autocomplete, documentation, and type safety when composing analytical 
models and analysis solvers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, ConfigDict, Field
from .types import Quantity


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


class CheckpointResult(SolverResult):
    """Result from CheckpointModel: IOPS and MFU penalty modeling."""
    checkpoint_size: Quantity
    write_time_seconds: Quantity
    max_bandwidth_required: Quantity
    storage_bottleneck: bool
    mfu_penalty_pct: float


class SustainabilityResult(SolverResult):
    """Result from SustainabilityModel: energy, carbon, and water footprint."""
    it_energy_kwh: Quantity
    total_energy_kwh: Quantity
    carbon_footprint_kg: float
    water_usage_liters: float
    pue: float
    region_name: str


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


class ContinuousBatchingResult(SolverResult):
    """Result from ContinuousBatchingModel: production LLM serving with PagedAttention."""
    feasible: bool
    throughput_tokens_per_sec: float
    max_active_requests: int
    memory_fragmentation_pct: float
    paged_kv_cache_size: Quantity
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
    slo_violation_probability: float


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
    """Result from CompressionModel: compression trade-offs."""
    original_size_gb: Quantity
    compressed_size_gb: Quantity
    compression_ratio: float
    estimated_accuracy_delta: float
    memory_savings_pct: float


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
    """Result from SensitivitySolver: binding constraint identification."""
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
    slo_violation_probability: float
    is_feasible: bool


class PlacementOptimizerResult(OptimizerResult):
    """Result from PlacementOptimizer: optimal datacenter region."""
    best_region: str
    lowest_tco: float
    carbon_footprint: float
    pue: float
    top_candidates: List[Dict[str, Any]]
