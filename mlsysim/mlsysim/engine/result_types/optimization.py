"""Typed result models for optimization calculations."""

from __future__ import annotations

from typing import Any, Dict, Optional, List
from pydantic import Field
from ...core.types import Quantity
from .base import OptimizerResult, SolverResult

__all__ = [
    "EfficiencyResult",
    "CompressionResult",
    "CompressionCandidate",
    "CompressionSweepResult",
    "SynthesisResult",
    "OrchestrationResult",
    "SensitivityResult",
    "ParallelismOptimizerResult",
    "BatchingOptimizerResult",
    "PlacementOptimizerResult",
    "DecodeBoundResult",
    "IronLawResult",
    "AmdahlResult",
    "PreemptionTaxResult",
]


class EfficiencyResult(SolverResult):
    """Result from EfficiencyModel: achievable MFU breakdown."""

    __module__ = "mlsysim.engine.results"

    mfu: float
    achievable_flops: Quantity
    peak_flops: Quantity
    workload_type: str
    use_flash_attention: bool
    overhead_breakdown: Dict[str, float]


class CompressionResult(SolverResult):
    """Result from CompressionModel: compression trade-offs.

    ``original_size_gb``, ``compression_ratio``, ``memory_savings_pct`` and
    ``inference_speedup`` are all relative to ``baseline_precision``.
    """

    __module__ = "mlsysim.engine.results"

    baseline_precision: str
    original_size_gb: Quantity
    compressed_size_gb: Quantity
    compression_ratio: float
    estimated_accuracy_delta: float
    memory_savings_pct: float
    inference_speedup: float = 1.0

    @property
    def original_size(self) -> Quantity:
        """Return weight storage at the baseline precision with units.

        Returns
        -------
        Quantity
            Weight storage at the baseline precision as a Pint quantity, conventionally in GB.

        Notes
        -----
        Equation: original_size = original_size_gb, with no arithmetic or unit conversion.

        Assumptions: The producing solver supplied original_size_gb as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing original_size_gb field
        directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import CompressionResult
        >>> r = CompressionResult(baseline_precision="fp16", original_size_gb=Q_(2, "GB"),
        ...     compressed_size_gb=Q_(1, "GB"), compression_ratio=2,
        ...     estimated_accuracy_delta=0, memory_savings_pct=50)
        >>> r.original_size is r.original_size_gb
        True
        """
        return self.original_size_gb

    @property
    def compressed_size(self) -> Quantity:
        """Return weight storage after compression with units.

        Returns
        -------
        Quantity
            Weight storage after compression as a Pint quantity, conventionally in GB.

        Notes
        -----
        Equation: compressed_size = compressed_size_gb, with no arithmetic or unit
        conversion.

        Assumptions: The producing solver supplied compressed_size_gb as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing compressed_size_gb
        field directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import CompressionResult
        >>> r = CompressionResult(baseline_precision="fp16", original_size_gb=Q_(2, "GB"),
        ...     compressed_size_gb=Q_(1, "GB"), compression_ratio=2,
        ...     estimated_accuracy_delta=0, memory_savings_pct=50)
        >>> r.compressed_size is r.compressed_size_gb
        True
        """
        return self.compressed_size_gb


class CompressionCandidate(SolverResult):
    """One evaluated compression configuration with lab-ready feasibility metadata."""

    __module__ = "mlsysim.engine.results"

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
        """Return weight storage at the baseline precision with units.

        Returns
        -------
        Quantity
            Weight storage at the baseline precision as a Pint quantity, conventionally in GB.

        Notes
        -----
        Equation: original_size = original_size_gb, with no arithmetic or unit conversion.

        Assumptions: The producing solver supplied original_size_gb as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing original_size_gb field
        directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import CompressionCandidate
        >>> r = CompressionCandidate(label="example", method="quantization",
        ...     baseline_precision="fp16", original_size_gb=Q_(2, "GB"),
        ...     compressed_size_gb=Q_(1, "GB"), compression_ratio=2,
        ...     estimated_accuracy_delta=0, memory_savings_pct=50)
        >>> r.original_size is r.original_size_gb
        True
        """
        return self.original_size_gb

    @property
    def compressed_size(self) -> Quantity:
        """Return weight storage after compression with units.

        Returns
        -------
        Quantity
            Weight storage after compression as a Pint quantity, conventionally in GB.

        Notes
        -----
        Equation: compressed_size = compressed_size_gb, with no arithmetic or unit
        conversion.

        Assumptions: The producing solver supplied compressed_size_gb as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing compressed_size_gb
        field directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import CompressionCandidate
        >>> r = CompressionCandidate(label="example", method="quantization",
        ...     baseline_precision="fp16", original_size_gb=Q_(2, "GB"),
        ...     compressed_size_gb=Q_(1, "GB"), compression_ratio=2,
        ...     estimated_accuracy_delta=0, memory_savings_pct=50)
        >>> r.compressed_size is r.compressed_size_gb
        True
        """
        return self.compressed_size_gb


class CompressionSweepResult(SolverResult):
    """A compression design-space sweep with candidate and Pareto metadata."""

    __module__ = "mlsysim.engine.results"

    candidates: List[CompressionCandidate]
    frontier_labels: List[str] = Field(default_factory=list)
    dominated_labels: List[str] = Field(default_factory=list)
    best_candidate_label: Optional[str] = None
    objective: str = "min_size_max_speed_preserve_quality"


class SynthesisResult(SolverResult):
    """Result from SynthesisSolver: inverse-Roofline hardware requirements."""

    __module__ = "mlsysim.engine.results"

    required_bw: Quantity
    required_flops: Quantity
    required_memory: Quantity
    compute_memory_ratio: Quantity
    target_latency: Quantity
    model_size: Quantity
    total_ops: Quantity


class OrchestrationResult(SolverResult):
    """Result from OrchestrationModel: queueing and wait times."""

    __module__ = "mlsysim.engine.results"

    cluster_utilization: float
    avg_wait_time_days: Quantity
    avg_queue_length: float
    is_stable: bool

    @property
    def avg_wait_time(self) -> Quantity:
        """Return expected queue waiting time with units.

        Returns
        -------
        Quantity
            Expected queue waiting time as a Pint quantity, conventionally in days. An unstable
            queue may carry an infinite wait.

        Notes
        -----
        Equation: avg_wait_time = avg_wait_time_days, with no arithmetic or unit conversion.

        Assumptions: The producing solver supplied avg_wait_time_days as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing avg_wait_time_days
        field directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import OrchestrationResult
        >>> r = OrchestrationResult(cluster_utilization=0.5,
        ...     avg_wait_time_days=Q_(0.5, "day"), avg_queue_length=0.25, is_stable=True)
        >>> r.avg_wait_time is r.avg_wait_time_days
        True
        """
        return self.avg_wait_time_days


class SensitivityResult(SolverResult):
    """Result from SensitivitySolver: binding constraint identification.

    ``feasible`` mirrors the baseline solve. When it is False the model does
    not fit in memory, ``binding_constraint`` is ``"memory_capacity"``, and
    ``constraint_trace`` explains why the perturbation sensitivities are not
    a ranking of hardware levers.
    """

    __module__ = "mlsysim.engine.results"

    feasible: bool
    sensitivities: Dict[str, float]
    binding_constraint: str
    baseline_latency: Quantity
    perturbation_pct: float


class ParallelismOptimizerResult(OptimizerResult):
    """Result from ParallelismOptimizer: optimal TP/PP/DP split."""

    __module__ = "mlsysim.engine.results"

    best_mfu: float
    best_throughput: Quantity
    best_step_time: Quantity
    search_space_size: int
    top_candidates: List[Dict[str, Any]]


class BatchingOptimizerResult(OptimizerResult):
    """Result from BatchingOptimizer: optimal batch size for latency SLA."""

    __module__ = "mlsysim.engine.results"

    best_batch_size: int
    max_throughput: float
    p99_latency: Quantity
    slo_headroom_ratio: float
    is_feasible: bool


class PlacementOptimizerResult(OptimizerResult):
    """Result from PlacementOptimizer: optimal datacenter region."""

    __module__ = "mlsysim.engine.results"

    best_region: str
    lowest_tco: float
    carbon_footprint: float
    pue: float
    top_candidates: List[Dict[str, Any]]


class DecodeBoundResult(SolverResult):
    """Result from DecodeBoundModel: weight-streaming bound on one decode step."""

    __module__ = "mlsysim.engine.results"

    weight_bytes: Quantity
    flops_per_step: Quantity
    compute_time: Quantity
    memory_time: Quantity
    memory_share: float  # memory_time / (compute_time + memory_time)
    bound_ratio: float  # memory_time / compute_time
    intensity: Quantity  # FLOP per weight byte
    attainable_flops: Quantity  # min(peak, bandwidth x intensity)
    peak_fraction: float  # attainable / peak
    bottleneck: str


class IronLawResult(SolverResult):
    """Result from IronLawModel: the three iron-law terms, summed and overlapped, before and after an upgrade."""

    __module__ = "mlsysim.engine.results"

    memory_time: Quantity
    compute_time: Quantity
    overhead: Quantity
    sequential_time: Quantity  # memory + compute + overhead
    overlapped_time: Quantity  # max(memory, compute) + overhead
    upgraded_memory_time: Quantity
    upgraded_compute_time: Quantity
    upgraded_sequential_time: Quantity
    upgraded_overlapped_time: Quantity
    naive_upgraded_time: Quantity  # the sequential time divided by the whole upgrade
    sequential_reduction: float  # fraction of the sequential time the upgrade removes


class AmdahlResult(SolverResult):
    """Result from AmdahlModel: one stage of a sequential pipeline made faster."""

    __module__ = "mlsysim.engine.results"

    total_time: Quantity
    stage_time: Quantity
    upgraded_stage_time: Quantity
    upgraded_total_time: Quantity
    untouched_time: Quantity
    stage_fraction: float  # share of the original total spent in the stage
    overall_speedup: float
    end_to_end_reduction: float  # fraction of the total removed
    stage_reduction: float  # fraction of the stage removed
    speedup_bound: float  # overall speedup with an infinitely fast stage


class PreemptionTaxResult(SolverResult):
    """Result from PreemptionTaxModel: what one forced preemption, and a day of them, costs."""

    __module__ = "mlsysim.engine.results"

    lost_work: Quantity
    restart: Quantity
    reload: Quantity
    warmup: Quantity
    relaunch: Quantity  # restart + reload + warmup, no detection
    per_event: Quantity  # lost work + relaunch
    cost_per_event: float  # USD
    events_per_day: float
    lost_per_day: Quantity
    gpu_hours_per_day: float
