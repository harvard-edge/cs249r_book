"""Chapter 5 distributed-training experiments.

This module composes the existing distributed performance and training-memory
models into the five comparisons used by the Volume II lab.  Scenario fixtures
are deliberately small and explicit: they are teaching assumptions, not live
measurements or claims about a particular production training run.

The public package exports are intentionally unchanged.  The staged lab imports
this module directly until the notebook design has passed its audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ..core.types import Quantity
from ..core.units import resolve_precision, ureg
from ..models.registry import Models
from ..models.types import TransformerWorkload, Workload
from ..physics import calc_activation_memory, calc_pipeline_bubble
from ..systems.registry import Systems
from ..systems.types import Fleet
from .solvers.distributed import DistributedModel
from .solvers.training import TrainingMemoryModel


ScalingMode = Literal["strong", "weak"]


@dataclass(frozen=True)
class TrackScenario:
    """An upstream training job for one deployment track."""

    key: str
    deployment_target: str
    model: Workload
    fleet: Fleet
    worker_counts: tuple[int, ...]
    local_batch: int
    global_batch: int
    model_parallel_applicable: bool
    applicability_note: str


@dataclass(frozen=True)
class ScalingPoint:
    workers: int
    local_batch: int
    global_batch: int
    step_time: Quantity
    useful_throughput: Quantity
    speedup: float
    parallel_efficiency: float
    communication_time: Quantity


@dataclass(frozen=True)
class MemoryPlan:
    feasible: bool
    weights: Quantity
    gradients: Quantity
    optimizer_state: Quantity
    activations: Quantity
    communication_buffers: Quantity
    total: Quantity
    available: Quantity
    retained_microbatches: int
    microbatch_size: int
    parallelism: dict[str, int]


@dataclass(frozen=True)
class PipelineSchedule:
    stages: int
    microbatch_count: int
    microbatch_size: int
    retained_microbatches: int
    bubble_fraction: float
    retained_activation_memory: Quantity
    total_memory: Quantity
    feasible: bool


@dataclass(frozen=True)
class ParallelLayout:
    devices: int
    dp_size: int
    tp_size: int
    pp_size: int
    tensor_parallel_tier: str
    step_time: Quantity
    dp_communication_time: Quantity
    tp_communication_time: Quantity
    pipeline_bubble_time: Quantity
    memory: MemoryPlan


@dataclass(frozen=True)
class ConvergenceFixture:
    """Illustrative gradient-noise-scale curve for a matched quality target.

    ``minimum_steps`` is the asymptotic number of optimizer steps.  The model
    follows S(B) = S_min * (1 + B_crit / B), so larger global batches reduce
    optimizer steps but encounter diminishing returns around ``critical_batch``.
    ``work_multiplier`` can encode supplied evidence for a policy whose stale or
    dropped gradients require additional optimization work.
    """

    name: str
    minimum_steps: float
    critical_batch: int


@dataclass(frozen=True)
class QualityPolicyResult:
    name: str
    global_batch: int
    step_time: Quantity
    optimizer_steps: int
    time_to_quality: Quantity
    work_multiplier: float


@dataclass(frozen=True)
class StragglerPolicyFixture:
    """Explicit illustrative behavior fixture for matched-quality comparison."""

    name: str
    step_time_multiplier: float
    work_multiplier: float


@dataclass(frozen=True)
class QualityComparison:
    baseline: QualityPolicyResult
    intervention: QualityPolicyResult
    fixture: ConvergenceFixture
    policy: StragglerPolicyFixture


_TRACKS = {
    "tinyml": TrackScenario(
        key="tinyml",
        deployment_target="battery sensor fleet",
        model=Models.Tiny.DS_CNN,
        fleet=Systems.Clusters.Lab_64_H100,
        worker_counts=(8, 16, 32, 64),
        local_batch=32,
        global_batch=256,
        model_parallel_applicable=False,
        applicability_note=(
            "The compact model trains upstream on an accelerator cluster; the "
            "deployed microcontrollers do not execute tensor or pipeline parallelism."
        ),
    ),
    "mobile": TrackScenario(
        key="mobile",
        deployment_target="mobile application fleet",
        model=Models.Vision.MobileNetV2,
        fleet=Systems.Clusters.Lab_64_H100,
        worker_counts=(8, 16, 32, 64),
        local_batch=32,
        global_batch=256,
        model_parallel_applicable=False,
        applicability_note=(
            "The deployment model trains upstream; its state fits on each training "
            "accelerator, so model parallelism is unnecessary for this fixture."
        ),
    ),
    "edge": TrackScenario(
        key="edge",
        deployment_target="edge perception fleet",
        model=Models.Vision.ResNet50,
        fleet=Systems.Clusters.Lab_64_H100,
        worker_counts=(8, 16, 32, 64),
        local_batch=16,
        global_batch=128,
        model_parallel_applicable=False,
        applicability_note=(
            "The perception model trains upstream with data parallelism; tensor and "
            "pipeline parallelism are not needed to make this model fit."
        ),
    ),
    "cloud": TrackScenario(
        key="cloud",
        deployment_target="cloud language service",
        model=Models.Language.Llama3_8B,
        fleet=Systems.Clusters.Research_256,
        worker_counts=(8, 32, 128, 256),
        local_batch=4,
        global_batch=256,
        model_parallel_applicable=True,
        applicability_note=(
            "The upstream language-model job can use data, tensor, pipeline, and "
            "state-sharding dimensions on the accelerator fleet."
        ),
    ),
}


_QUALITY_FIXTURES = {
    "tinyml": (
        ConvergenceFixture("illustrative compact-model convergence", 1_200, 256),
        StragglerPolicyFixture("drop slow workers", 0.82, 1.35),
    ),
    "mobile": (
        ConvergenceFixture("illustrative mobile-model convergence", 1_600, 512),
        StragglerPolicyFixture("drop slow workers", 0.80, 1.40),
    ),
    "edge": (
        ConvergenceFixture("illustrative perception convergence", 2_000, 1_024),
        StragglerPolicyFixture("drop slow workers", 0.78, 1.45),
    ),
    "cloud": (
        ConvergenceFixture("illustrative language-model convergence", 2_500, 2_048),
        StragglerPolicyFixture("drop slow workers", 0.75, 1.55),
    ),
}


def get_track_scenario(track: str) -> TrackScenario:
    """Return an immutable upstream-training fixture for a teaching track."""

    key = track.lower()
    if key not in _TRACKS:
        raise ValueError(f"unknown track {track!r}; choose from {', '.join(_TRACKS)}")
    return _TRACKS[key]


def get_quality_fixtures(
    track: str,
) -> tuple[ConvergenceFixture, StragglerPolicyFixture]:
    """Return the labeled illustrative convergence and straggler assumptions."""

    get_track_scenario(track)  # shared validation and error message
    return _QUALITY_FIXTURES[track.lower()]


def _subfleet(fleet: Fleet, workers: int) -> Fleet:
    """Select whole nodes from a fleet without inventing partial-node topology."""

    per_node = fleet.node.accelerators_per_node
    if workers < per_node or workers % per_node:
        raise ValueError(
            f"workers must be a positive multiple of {per_node} accelerators per node"
        )
    if workers > fleet.total_accelerators:
        raise ValueError(
            f"workers ({workers}) exceed fixture capacity ({fleet.total_accelerators})"
        )
    return fleet.model_copy(
        update={"name": f"{workers}-accelerator slice of {fleet.name}", "count": workers // per_node}
    )


def scaling_sweep(
    scenario: TrackScenario,
    worker_counts: Sequence[int],
    *,
    mode: ScalingMode,
    global_batch: int | None = None,
    local_batch: int | None = None,
    precision: str = "fp16",
    efficiency: float = 0.5,
) -> tuple[ScalingPoint, ...]:
    """Compare fixed-work (strong) or work-per-worker (weak) scaling.

    Strong scaling holds ``global_batch`` fixed.  Weak scaling holds
    ``local_batch`` fixed, making global batch grow with the number of
    data-parallel workers.  These inputs are intentionally separate so a UI
    cannot silently turn one experiment into the other.
    """

    if mode not in {"strong", "weak"}:
        raise ValueError("mode must be 'strong' or 'weak'")
    if not worker_counts:
        raise ValueError("worker_counts must not be empty")
    fixed_global = scenario.global_batch if global_batch is None else global_batch
    fixed_local = scenario.local_batch if local_batch is None else local_batch
    if fixed_global < 1 or fixed_local < 1:
        raise ValueError("batch sizes must be at least one")

    raw: list[tuple[int, int, int, object]] = []
    solver = DistributedModel()
    for workers in worker_counts:
        fleet = _subfleet(scenario.fleet, workers)
        batch = fixed_global if mode == "strong" else fixed_local * workers
        result = solver.solve(
            scenario.model,
            fleet,
            batch_size=batch,
            precision=precision,
            efficiency=efficiency,
            tp_size=1,
            pp_size=1,
            microbatch_count=1,
        )
        raw.append((workers, math.ceil(batch / workers), batch, result))

    baseline_workers, _, baseline_batch, baseline = raw[0]
    baseline_throughput = baseline_batch / baseline.step_latency_total.to("second").magnitude
    points: list[ScalingPoint] = []
    for workers, per_worker, batch, result in raw:
        throughput = (batch / result.step_latency_total.to("second").magnitude) / ureg.second
        speedup = throughput.magnitude / baseline_throughput
        points.append(
            ScalingPoint(
                workers=workers,
                local_batch=per_worker,
                global_batch=batch,
                step_time=result.step_latency_total.to("ms"),
                useful_throughput=throughput,
                speedup=float(speedup),
                parallel_efficiency=float(speedup * baseline_workers / workers),
                communication_time=result.communication_latency.to("ms"),
            )
        )
    return tuple(points)


def parallel_layout(
    model: TransformerWorkload,
    fleet: Fleet,
    *,
    global_batch: int,
    seq_len: int,
    dp_size: int,
    tp_size: int,
    pp_size: int,
    microbatch_count: int,
    zero_stage: int = 0,
    precision: str = "fp16",
    efficiency: float = 0.5,
    activation_checkpointing: str = "selective",
) -> ParallelLayout:
    """Evaluate one equal-budget DP × TP × PP layout on explicit link tiers."""

    devices = dp_size * tp_size * pp_size
    if devices != fleet.total_accelerators:
        raise ValueError(
            "dp_size * tp_size * pp_size must equal the fleet accelerator budget"
        )
    performance = DistributedModel().solve(
        model,
        fleet,
        batch_size=global_batch,
        seq_len=seq_len,
        precision=precision,
        efficiency=efficiency,
        tp_size=tp_size,
        pp_size=pp_size,
        microbatch_count=microbatch_count,
        zero_stage=zero_stage,
    )
    memory = training_memory_plan(
        model,
        fleet,
        global_batch=global_batch,
        seq_len=seq_len,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        microbatch_count=microbatch_count,
        zero_stage=zero_stage,
        precision=precision,
        activation_checkpointing=activation_checkpointing,
    )
    tier = (
        "intra-node"
        if tp_size <= fleet.node.accelerators_per_node
        else "inter-node"
    )
    return ParallelLayout(
        devices=devices,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size,
        tensor_parallel_tier=tier,
        step_time=performance.step_latency_total.to("ms"),
        dp_communication_time=performance.dp_communication_latency.to("ms"),
        tp_communication_time=performance.tp_communication_latency.to("ms"),
        pipeline_bubble_time=performance.pipeline_bubble_latency.to("ms"),
        memory=memory,
    )


def get_parallel_layout_specs(fleet: Fleet) -> dict[str, dict[str, int]]:
    """Return candidate equal-budget DP x TP x PP layout configurations for a fleet."""

    devices = fleet.total_accelerators
    per_node = fleet.node.accelerators_per_node
    if devices < per_node * 2:
        raise ValueError(
            f"fleet must have at least {per_node * 2} accelerators for multi-node layout"
        )
    return {
        "within": {"dp_size": devices // per_node, "tp_size": per_node, "pp_size": 1},
        "cross": {"dp_size": devices // (per_node * 2), "tp_size": per_node * 2, "pp_size": 1},
        "pipeline": {"dp_size": devices // (per_node * 2), "tp_size": per_node, "pp_size": 2},
    }


def get_capacity_dimensions(fleet: Fleet) -> dict[str, int]:
    """Return default batch and parallelism dimensions for the 70B capacity case."""

    per_node = fleet.node.accelerators_per_node
    dp_size = max(1, fleet.total_accelerators // per_node)
    global_batch = max(64, dp_size * 8)
    return {
        "global_batch": global_batch,
        "dp_size": dp_size,
        "tp_size": per_node,
        "pp_size": 1,
    }


def get_pipeline_dimensions(fleet: Fleet) -> dict[str, int]:
    """Return default parallelism dimensions for the 8-stage pipeline schedule."""

    per_node = fleet.node.accelerators_per_node
    dp_size = max(1, fleet.total_accelerators // (8 * per_node))
    return {
        "dp_size": dp_size,
        "tp_size": per_node,
        "pp_size": 8,
    }


def training_memory_plan(
    model: TransformerWorkload,
    fleet: Fleet,
    *,
    global_batch: int,
    seq_len: int,
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
    zero_stage: int = 0,
    microbatch_count: int = 1,
    gradient_accumulation_steps: int = 1,
    precision: str = "fp16",
    activation_checkpointing: str = "selective",
) -> MemoryPlan:
    """Account explicit model state plus worst-stage retained activations."""

    for name, value in {
        "global_batch": global_batch,
        "seq_len": seq_len,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "dp_size": dp_size,
        "microbatch_count": microbatch_count,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be at least one")
    if tp_size * pp_size * dp_size > fleet.total_accelerators:
        raise ValueError("parallelism plan exceeds the selected fleet")

    # The existing solver gets one pipeline microbatch in residence.  Passing
    # the full subdivision as accumulation makes its activation size exactly
    # the microbatch used below while preserving its state-sharding equations.
    subdivision = gradient_accumulation_steps * microbatch_count
    state = TrainingMemoryModel().solve(
        model,
        fleet.node.accelerator,
        batch_size=global_batch,
        seq_len=seq_len,
        precision=precision,
        activation_checkpointing=activation_checkpointing,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        zero_stage=zero_stage,
        gradient_accumulation_steps=subdivision,
    )
    microbatch_size = max(1, math.ceil(global_batch / (dp_size * subdivision)))
    retained = min(microbatch_count, pp_size) if pp_size > 1 else 1
    retained_activations = state.activations * retained
    total = (
        state.weights
        + state.gradients
        + state.optimizer_state
        + retained_activations
        + state.communication_buffers
    ).to("GB")
    return MemoryPlan(
        feasible=bool(total <= state.available_memory),
        weights=state.weights,
        gradients=state.gradients,
        optimizer_state=state.optimizer_state,
        activations=retained_activations.to("GB"),
        communication_buffers=state.communication_buffers,
        total=total,
        available=state.available_memory,
        retained_microbatches=retained,
        microbatch_size=microbatch_size,
        parallelism={"dp": dp_size, "tp": tp_size, "pp": pp_size},
    )


def pipeline_schedule(
    model: TransformerWorkload,
    fleet: Fleet,
    *,
    microbatch_size: int,
    seq_len: int,
    pp_size: int,
    microbatch_count: int,
    dp_size: int = 1,
    tp_size: int = 1,
    zero_stage: int = 0,
    precision: str = "fp16",
    activation_checkpointing: str = "selective",
) -> PipelineSchedule:
    """Return pipeline bubble and retained-activation capacity together.

    This experiment holds ``microbatch_size`` fixed.  Increasing
    ``microbatch_count`` therefore injects more work into the schedule and may
    increase the number of activation sets simultaneously retained.  Keeping
    this input separate avoids the common error of treating microbatch count
    and microbatch size as the same quantity.
    """

    if microbatch_size < 1:
        raise ValueError("microbatch_size must be at least one")
    global_batch = microbatch_size * microbatch_count * dp_size

    plan = training_memory_plan(
        model,
        fleet,
        global_batch=global_batch,
        seq_len=seq_len,
        pp_size=pp_size,
        microbatch_count=microbatch_count,
        dp_size=dp_size,
        tp_size=tp_size,
        zero_stage=zero_stage,
        precision=precision,
        activation_checkpointing=activation_checkpointing,
    )
    return PipelineSchedule(
        stages=pp_size,
        microbatch_count=microbatch_count,
        microbatch_size=microbatch_size,
        retained_microbatches=plan.retained_microbatches,
        bubble_fraction=calc_pipeline_bubble(pp_size, microbatch_count),
        retained_activation_memory=plan.activations,
        total_memory=plan.total,
        feasible=plan.feasible,
    )


def time_to_quality(
    *,
    name: str,
    step_time: Quantity,
    global_batch: int,
    fixture: ConvergenceFixture,
    work_multiplier: float = 1.0,
) -> QualityPolicyResult:
    """Convert step latency to time for the same fixture-defined quality target."""

    if global_batch < 1:
        raise ValueError("global_batch must be at least one")
    if fixture.minimum_steps <= 0 or fixture.critical_batch < 1:
        raise ValueError("convergence fixture values must be positive")
    if work_multiplier < 1.0:
        raise ValueError("work_multiplier must be at least one")
    if step_time.to("second").magnitude <= 0:
        raise ValueError("step_time must be positive")

    steps = math.ceil(
        fixture.minimum_steps
        * (1.0 + fixture.critical_batch / global_batch)
        * work_multiplier
    )
    return QualityPolicyResult(
        name=name,
        global_batch=global_batch,
        step_time=step_time.to("ms"),
        optimizer_steps=steps,
        time_to_quality=(step_time * steps).to("hour"),
        work_multiplier=work_multiplier,
    )


def compare_straggler_policy(
    *,
    step_time: Quantity,
    global_batch: int,
    fixture: ConvergenceFixture,
    policy: StragglerPolicyFixture,
) -> QualityComparison:
    """Compare strict synchronization with one supplied straggler policy."""

    if not 0 < policy.step_time_multiplier <= 1:
        raise ValueError("policy step_time_multiplier must be in (0, 1]")
    baseline = time_to_quality(
        name="strict synchronization",
        step_time=step_time,
        global_batch=global_batch,
        fixture=fixture,
    )
    intervention = time_to_quality(
        name=policy.name,
        step_time=step_time * policy.step_time_multiplier,
        global_batch=global_batch,
        fixture=fixture,
        work_multiplier=policy.work_multiplier,
    )
    return QualityComparison(baseline, intervention, fixture, policy)


def serialize_quantity(value: Quantity, unit: str) -> dict[str, float | str]:
    """Serialize a Pint quantity into a replayable value-and-unit record."""

    converted = value.to(unit)
    return {"value": float(converted.magnitude), "unit": unit}


def scaling_point_record(
    point: ScalingPoint, *, inputs: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "inputs": dict(inputs),
        "workers": point.workers,
        "local_batch": point.local_batch,
        "global_batch": point.global_batch,
        "step_time": serialize_quantity(point.step_time, "ms"),
        "useful_throughput": serialize_quantity(point.useful_throughput, "1/s"),
        "speedup": point.speedup,
        "parallel_efficiency": point.parallel_efficiency,
        "communication_time": serialize_quantity(point.communication_time, "ms"),
    }


def memory_plan_record(
    plan: MemoryPlan, *, inputs: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "inputs": dict(inputs),
        "feasible": plan.feasible,
        "weights": serialize_quantity(plan.weights, "GB"),
        "gradients": serialize_quantity(plan.gradients, "GB"),
        "optimizer_state": serialize_quantity(plan.optimizer_state, "GB"),
        "activations": serialize_quantity(plan.activations, "GB"),
        "communication_buffers": serialize_quantity(plan.communication_buffers, "GB"),
        "total": serialize_quantity(plan.total, "GB"),
        "available": serialize_quantity(plan.available, "GB"),
        "retained_microbatches": plan.retained_microbatches,
        "microbatch_size": plan.microbatch_size,
        "parallelism": dict(plan.parallelism),
    }


def parallel_layout_record(
    layout: ParallelLayout, *, inputs: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "inputs": dict(inputs),
        "devices": layout.devices,
        "dp_size": layout.dp_size,
        "tp_size": layout.tp_size,
        "pp_size": layout.pp_size,
        "tensor_parallel_tier": layout.tensor_parallel_tier,
        "step_time": serialize_quantity(layout.step_time, "ms"),
        "dp_communication_time": serialize_quantity(layout.dp_communication_time, "ms"),
        "tp_communication_time": serialize_quantity(layout.tp_communication_time, "ms"),
        "pipeline_bubble_time": serialize_quantity(layout.pipeline_bubble_time, "ms"),
        "memory": memory_plan_record(layout.memory, inputs=inputs),
    }


def pipeline_schedule_record(
    schedule: PipelineSchedule, *, inputs: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "inputs": dict(inputs),
        "stages": schedule.stages,
        "microbatch_count": schedule.microbatch_count,
        "microbatch_size": schedule.microbatch_size,
        "retained_microbatches": schedule.retained_microbatches,
        "bubble_fraction": schedule.bubble_fraction,
        "retained_activation_memory": serialize_quantity(
            schedule.retained_activation_memory, "GB"
        ),
        "total_memory": serialize_quantity(schedule.total_memory, "GB"),
        "feasible": schedule.feasible,
    }


def quality_policy_record(
    result: QualityPolicyResult, *, inputs: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "inputs": dict(inputs),
        "name": result.name,
        "global_batch": result.global_batch,
        "step_time": serialize_quantity(result.step_time, "ms"),
        "optimizer_steps": result.optimizer_steps,
        "time_to_quality": serialize_quantity(result.time_to_quality, "hour"),
        "work_multiplier": result.work_multiplier,
    }


def activation_bytes_for_microbatch(
    model: TransformerWorkload,
    *,
    seq_len: int,
    microbatch_size: int,
    pp_size: int,
    precision: str = "fp16",
    activation_checkpointing: str = "selective",
) -> Quantity:
    """Expose the canonical per-stage activation calculation for audit panels."""

    _, precision_bytes = resolve_precision(precision)
    return calc_activation_memory(
        n_layers=math.ceil(model.layers / pp_size),
        seq_len=seq_len,
        batch_size=microbatch_size,
        hidden_dim=model.hidden_dim or 4096,
        n_heads=model.heads,
        precision_bytes=precision_bytes.to("byte").magnitude,
        strategy=activation_checkpointing,
    ).to("GB")
