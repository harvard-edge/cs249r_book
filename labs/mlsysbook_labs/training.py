"""Training-feasibility helpers for track-aware training labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class TrainingStrategyOption:
    strategy_id: str
    label: str
    training_location: str
    trainable_fraction: float
    precision_bytes: int
    optimizer: str
    optimizer_state_bytes: float
    checkpointing: str
    activation_factor: float
    batch_factor: float
    memory_budget_mb: float | None
    throughput_factor: float
    convergence_risk: str
    validation_location: str
    hidden_cost: str
    residual_risk: str


@dataclass(frozen=True)
class TrainingTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    model_params_m: float
    memory_capacity_mb: float
    memory_bandwidth_gbs: float
    tdp_w: float
    stakeholder: str
    training_story: str
    workload_label: str
    default_batch_size: int
    batch_min: int
    batch_max: int
    batch_step: int
    sample_mb: float
    activation_mb_per_mparam: float
    training_memory_budget_mb: float
    throughput_budget_samples_s: float
    quality_floor_pct: float
    strategy_options: tuple[TrainingStrategyOption, ...]
    validation_tests: tuple[str, ...]
    deployment_handoff: str
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class TrainingMemoryStackResult:
    strategy_id: str
    strategy_label: str
    batch_size: int
    weights_mb: float
    gradients_mb: float
    optimizer_mb: float
    activations_mb: float
    data_batch_mb: float
    total_mb: float
    budget_mb: float
    memory_utilization_pct: float
    throughput_samples_s: float
    dominant_component: str
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class TrainingFrontierPoint:
    batch_size: int
    total_mb: float
    throughput_samples_s: float
    feasible: bool


@dataclass(frozen=True)
class TrainingFrontierResult:
    strategy_id: str
    strategy_label: str
    points: tuple[TrainingFrontierPoint, ...]
    max_feasible_batch: int | None
    first_infeasible_batch: int | None


@dataclass(frozen=True)
class TrainingPlanResult:
    selected_id: str
    selected_label: str
    training_location: str
    validation_location: str
    feasible: bool
    dominant_component: str
    total_memory_mb: float
    max_feasible_batch: int | None
    convergence_risk: str
    hidden_cost: str
    deployment_handoff: str
    residual_risk: str
    rejected_alternatives: tuple[str, ...]
    memo_summary: str


def _quantity_to_float(value: Any, unit: str, default: float) -> float:
    if value is None:
        return default
    if hasattr(value, "m_as"):
        try:
            return float(value.m_as(unit))
        except Exception:
            return default
    if hasattr(value, "to"):
        try:
            return float(value.to(unit).magnitude)
        except Exception:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _model_params_m(model: Any, default: float) -> float:
    params = getattr(model, "parameters", None)
    value = _quantity_to_float(params, "count", 0.0)
    if value > 0:
        return value / 1_000_000.0
    return default


def _tuple_str(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if value:
        return (str(value),)
    return ()


def _strategy_options(defaults: Mapping[str, Any]) -> tuple[TrainingStrategyOption, ...]:
    raw_options = defaults.get("strategy_options", {})
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    options: list[TrainingStrategyOption] = []
    for strategy_id, raw in raw_options.items():
        details = raw if isinstance(raw, Mapping) else {}
        options.append(
            TrainingStrategyOption(
                strategy_id=str(strategy_id),
                label=str(details.get("label", strategy_id)),
                training_location=str(details.get("training_location", "unspecified")),
                trainable_fraction=float(details.get("trainable_fraction", 1.0)),
                precision_bytes=int(details.get("precision_bytes", 2)),
                optimizer=str(details.get("optimizer", "adam")),
                optimizer_state_bytes=float(details.get("optimizer_state_bytes", 8.0)),
                checkpointing=str(details.get("checkpointing", "none")),
                activation_factor=float(details.get("activation_factor", 1.0)),
                batch_factor=float(details.get("batch_factor", 1.0)),
                memory_budget_mb=(
                    float(details["memory_budget_mb"])
                    if "memory_budget_mb" in details and details["memory_budget_mb"] is not None
                    else None
                ),
                throughput_factor=float(details.get("throughput_factor", 1.0)),
                convergence_risk=str(details.get("convergence_risk", "convergence risk not specified")),
                validation_location=str(details.get("validation_location", "validation location not specified")),
                hidden_cost=str(details.get("hidden_cost", "hidden cost not specified")),
                residual_risk=str(details.get("residual_risk", "residual training risk not specified")),
            )
        )
    if options:
        return tuple(options)
    return (
        TrainingStrategyOption(
            "baseline",
            "Baseline training",
            "unspecified",
            1.0,
            2,
            "adam",
            8.0,
            "none",
            1.0,
            1.0,
            None,
            1.0,
            "baseline convergence risk",
            "baseline validation",
            "baseline hidden cost",
            "no mitigation selected",
        ),
    )


def training_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> TrainingTrackProfile:
    """Build a source-traced training profile from variant defaults."""
    defaults = variant.defaults
    memory = getattr(hardware, "memory", None)
    default_params_m = float(defaults.get("model_params_m", 10.0))
    return TrainingTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        model_params_m=float(defaults.get("model_params_m", _model_params_m(model, default_params_m))),
        memory_capacity_mb=_quantity_to_float(getattr(memory, "capacity", None), "MB", 0.0),
        memory_bandwidth_gbs=_quantity_to_float(getattr(memory, "bandwidth", None), "GB/s", 0.0),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0),
        stakeholder=variant.stakeholder,
        training_story=str(defaults.get("training_story", variant.workload_summary)),
        workload_label=str(defaults.get("workload_label", variant.workload_summary)),
        default_batch_size=int(defaults.get("default_batch_size", 4)),
        batch_min=int(defaults.get("batch_min", 1)),
        batch_max=int(defaults.get("batch_max", 64)),
        batch_step=max(1, int(defaults.get("batch_step", 1))),
        sample_mb=float(defaults.get("sample_mb", 1.0)),
        activation_mb_per_mparam=float(defaults.get("activation_mb_per_mparam", 1.0)),
        training_memory_budget_mb=float(defaults.get("training_memory_budget_mb", 1024.0)),
        throughput_budget_samples_s=float(defaults.get("throughput_budget_samples_s", 1.0)),
        quality_floor_pct=float(defaults.get("quality_floor_pct", 80.0)),
        strategy_options=_strategy_options(defaults),
        validation_tests=_tuple_str(defaults.get("validation_tests")),
        deployment_handoff=str(defaults.get("deployment_handoff", "deployment handoff not specified")),
        report_artifact=str(variant.assumptions.get("report_artifact", "training feasibility plan")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=tuple(ref for ref in (variant.hardware_ref, variant.model_ref, variant.system_ref) if ref),
    )


def _strategy(profile: TrainingTrackProfile, strategy_id: str) -> TrainingStrategyOption:
    return next(
        (option for option in profile.strategy_options if option.strategy_id == strategy_id),
        profile.strategy_options[0],
    )


def training_memory_stack(
    profile: TrainingTrackProfile,
    *,
    strategy_id: str,
    batch_size: int | None = None,
) -> TrainingMemoryStackResult:
    """Compute the training memory stack for a strategy and batch size."""
    strategy = _strategy(profile, strategy_id)
    batch = max(profile.batch_min, min(profile.batch_max, int(batch_size or profile.default_batch_size)))
    params = profile.model_params_m * 1_000_000.0
    trainable = params * max(0.0, min(1.0, strategy.trainable_fraction))
    precision = strategy.precision_bytes
    weights_mb = params * precision / 1_000_000.0
    gradients_mb = trainable * precision / 1_000_000.0
    optimizer_mb = trainable * strategy.optimizer_state_bytes / 1_000_000.0
    trainable_activation_scale = max(strategy.trainable_fraction, 0.02 if strategy.trainable_fraction > 0 else 0.0)
    activations_mb = (
        profile.model_params_m
        * batch
        * profile.activation_mb_per_mparam
        * strategy.activation_factor
        * trainable_activation_scale
    )
    data_batch_mb = batch * profile.sample_mb * strategy.batch_factor
    total = weights_mb + gradients_mb + optimizer_mb + activations_mb + data_batch_mb
    budget_mb = strategy.memory_budget_mb or profile.training_memory_budget_mb
    throughput = (
        batch
        * strategy.throughput_factor
        * max(0.05, 1.0 - 0.12 * max(0.0, strategy.activation_factor - 1.0))
    )
    components = {
        "weights": weights_mb,
        "gradients": gradients_mb,
        "optimizer state": optimizer_mb,
        "activations": activations_mb,
        "data batch": data_batch_mb,
    }
    dominant = max(components, key=components.get)
    violations = []
    if total > budget_mb:
        violations.append(f"training memory {total:.2f} MB > {budget_mb:.2f} MB")
    if throughput < profile.throughput_budget_samples_s:
        violations.append(
            f"throughput {throughput:.2f} samples/s < {profile.throughput_budget_samples_s:.2f} samples/s"
        )
    return TrainingMemoryStackResult(
        strategy_id=strategy.strategy_id,
        strategy_label=strategy.label,
        batch_size=batch,
        weights_mb=weights_mb,
        gradients_mb=gradients_mb,
        optimizer_mb=optimizer_mb,
        activations_mb=activations_mb,
        data_batch_mb=data_batch_mb,
        total_mb=total,
        budget_mb=budget_mb,
        memory_utilization_pct=100.0 * total / max(budget_mb, 1e-9),
        throughput_samples_s=throughput,
        dominant_component=dominant,
        feasible=not violations,
        violations=tuple(violations),
    )


def training_frontier(
    profile: TrainingTrackProfile,
    *,
    strategy_id: str,
) -> TrainingFrontierResult:
    """Sweep batch size and record the memory/throughput frontier."""
    values = tuple(range(profile.batch_min, profile.batch_max + 1, profile.batch_step))
    points = tuple(
        TrainingFrontierPoint(
            batch_size=result.batch_size,
            total_mb=result.total_mb,
            throughput_samples_s=result.throughput_samples_s,
            feasible=result.feasible,
        )
        for result in (
            training_memory_stack(profile, strategy_id=strategy_id, batch_size=batch)
            for batch in values
        )
    )
    feasible_batches = [point.batch_size for point in points if point.feasible]
    max_feasible = max(feasible_batches) if feasible_batches else None
    if max_feasible is None:
        first_after_feasible = min((point.batch_size for point in points if not point.feasible), default=None)
    else:
        first_after_feasible = min(
            (point.batch_size for point in points if not point.feasible and point.batch_size > max_feasible),
            default=None,
        )
    return TrainingFrontierResult(
        strategy_id=strategy_id,
        strategy_label=_strategy(profile, strategy_id).label,
        points=points,
        max_feasible_batch=max_feasible,
        first_infeasible_batch=first_after_feasible,
    )


def training_plan(
    profile: TrainingTrackProfile,
    *,
    strategy_id: str,
    batch_size: int | None = None,
) -> TrainingPlanResult:
    """Return the recommendation memo fields for the selected training strategy."""
    strategy = _strategy(profile, strategy_id)
    stack = training_memory_stack(profile, strategy_id=strategy.strategy_id, batch_size=batch_size)
    frontier = training_frontier(profile, strategy_id=strategy.strategy_id)
    rejected_items = []
    for option in profile.strategy_options:
        if option.strategy_id == strategy.strategy_id:
            continue
        result = training_memory_stack(profile, strategy_id=option.strategy_id, batch_size=batch_size)
        rejected_items.append(
            f"{option.label}: {result.dominant_component}; {'feasible' if result.feasible else 'not feasible'}"
        )
    summary = (
        f"Use {strategy.label} for {profile.label}; training happens at "
        f"{strategy.training_location}, validation at {strategy.validation_location}, "
        f"and dominant memory is {stack.dominant_component}."
    )
    return TrainingPlanResult(
        selected_id=strategy.strategy_id,
        selected_label=strategy.label,
        training_location=strategy.training_location,
        validation_location=strategy.validation_location,
        feasible=stack.feasible,
        dominant_component=stack.dominant_component,
        total_memory_mb=stack.total_mb,
        max_feasible_batch=frontier.max_feasible_batch,
        convergence_risk=strategy.convergence_risk,
        hidden_cost=strategy.hidden_cost,
        deployment_handoff=profile.deployment_handoff,
        residual_risk=strategy.residual_risk,
        rejected_alternatives=tuple(rejected_items),
        memo_summary=summary,
    )


__all__ = [
    "TrainingFrontierPoint",
    "TrainingFrontierResult",
    "TrainingMemoryStackResult",
    "TrainingPlanResult",
    "TrainingStrategyOption",
    "TrainingTrackProfile",
    "training_frontier",
    "training_memory_stack",
    "training_plan",
    "training_track_profile",
]
