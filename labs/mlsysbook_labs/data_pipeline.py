"""Data-pipeline helpers for track-aware data gravity labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class DataMovementStrategy:
    strategy_id: str
    label: str
    data_reduction_factor: float
    latency_factor: float
    cost_factor: float
    quality_factor: float
    privacy_risk: str
    residual_risk: str


@dataclass(frozen=True)
class DataPipelineTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    stakeholder: str
    data_source: str
    data_rate_mb_s: float
    burst_multiplier: float
    ingest_capacity_mb_s: float
    preprocess_capacity_mb_s: float
    storage_capacity_mb_s: float
    upload_capacity_mb_s: float
    retention_days: float
    local_storage_mb: float
    egress_cost_per_gb: float
    privacy_stance: str
    default_sample_multiplier: float
    sample_min: float
    sample_max: float
    sample_step: float
    strategies: tuple[DataMovementStrategy, ...]
    retention_options: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class PipelineStageResult:
    stage: str
    demand_mb_s: float
    capacity_mb_s: float
    utilization_pct: float
    feasible: bool


@dataclass(frozen=True)
class PipelineResult:
    sample_multiplier: float
    effective_rate_mb_s: float
    daily_raw_gb: float
    retained_gb: float
    local_storage_days: float
    bottleneck_stage: str
    compute_starvation_pct: float
    feasible: bool
    stages: tuple[PipelineStageResult, ...]


@dataclass(frozen=True)
class MovementFrontierResult:
    strategy_id: str
    strategy_label: str
    data_moved_gb: float
    transfer_hours: float
    egress_cost: float
    effective_latency_s: float
    quality_retained_pct: float
    privacy_risk: str
    residual_risk: str


@dataclass(frozen=True)
class PipelineArchitectureResult:
    strategy_id: str
    strategy_label: str
    retention_policy: str
    bottleneck_stage: str
    retained_gb: float
    quality_retained_pct: float
    accepted_data_risk: str
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


def _strategy_options(defaults: Mapping[str, Any]) -> tuple[DataMovementStrategy, ...]:
    raw_strategies = defaults.get("movement_strategies", {})
    if not isinstance(raw_strategies, Mapping):
        raw_strategies = {}
    strategies: list[DataMovementStrategy] = []
    for strategy_id, raw in raw_strategies.items():
        details = raw if isinstance(raw, Mapping) else {}
        strategies.append(
            DataMovementStrategy(
                strategy_id=str(strategy_id),
                label=str(details.get("label", strategy_id)),
                data_reduction_factor=float(details.get("data_reduction_factor", 1.0)),
                latency_factor=float(details.get("latency_factor", 1.0)),
                cost_factor=float(details.get("cost_factor", 1.0)),
                quality_factor=float(details.get("quality_factor", 1.0)),
                privacy_risk=str(details.get("privacy_risk", "unclassified privacy risk")),
                residual_risk=str(details.get("residual_risk", "unmodeled data risk")),
            )
        )
    if strategies:
        return tuple(strategies)
    return (
        DataMovementStrategy(
            "raw_upload",
            "Upload raw data",
            1.0,
            1.0,
            1.0,
            1.0,
            "raw data leaves the source device",
            "high movement cost and privacy exposure",
        ),
        DataMovementStrategy(
            "local_summary",
            "Local summary features",
            0.08,
            0.7,
            0.2,
            0.93,
            "raw data stays local but summaries may leak attributes",
            "rare events may be discarded by summarization",
        ),
    )


def data_pipeline_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> DataPipelineTrackProfile:
    """Build a source-traced data-pipeline profile from variant defaults."""
    defaults = variant.defaults
    storage = getattr(hardware, "storage", None)
    memory = getattr(hardware, "memory", None)
    storage_mb = _quantity_to_float(getattr(storage, "capacity", None), "MB", 0.0)
    if storage_mb <= 0:
        storage_mb = _quantity_to_float(getattr(memory, "flash_capacity", None), "MB", 0.0)
    if storage_mb <= 0:
        storage_mb = _quantity_to_float(getattr(memory, "capacity", None), "MB", 1024.0)
    return DataPipelineTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        stakeholder=variant.stakeholder,
        data_source=str(defaults.get("data_source", "application events")),
        data_rate_mb_s=float(defaults.get("data_rate_mb_s", 1.0)),
        burst_multiplier=float(defaults.get("burst_multiplier", 1.0)),
        ingest_capacity_mb_s=float(defaults.get("ingest_capacity_mb_s", 10.0)),
        preprocess_capacity_mb_s=float(defaults.get("preprocess_capacity_mb_s", 10.0)),
        storage_capacity_mb_s=float(defaults.get("storage_capacity_mb_s", 10.0)),
        upload_capacity_mb_s=float(defaults.get("upload_capacity_mb_s", 10.0)),
        retention_days=float(defaults.get("retention_days", 7.0)),
        local_storage_mb=float(defaults.get("local_storage_mb", storage_mb)),
        egress_cost_per_gb=float(defaults.get("egress_cost_per_gb", 0.08)),
        privacy_stance=str(defaults.get("privacy_stance", "data minimization")),
        default_sample_multiplier=float(defaults.get("default_sample_multiplier", 1.0)),
        sample_min=float(defaults.get("sample_min", 0.1)),
        sample_max=float(defaults.get("sample_max", 5.0)),
        sample_step=float(defaults.get("sample_step", 0.1)),
        strategies=_strategy_options(defaults),
        retention_options=tuple(str(item) for item in defaults.get("retention_options", ())),
        report_artifact=str(variant.assumptions.get("report_artifact", "data pipeline architecture memo")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=tuple(ref for ref in (variant.hardware_ref, variant.model_ref, variant.system_ref) if ref),
    )


def evaluate_pipeline(
    profile: DataPipelineTrackProfile,
    *,
    sample_multiplier: float,
) -> PipelineResult:
    """Evaluate pipeline stage utilization and retention pressure."""
    multiplier = max(profile.sample_min, min(profile.sample_max, float(sample_multiplier)))
    effective_rate = profile.data_rate_mb_s * multiplier * profile.burst_multiplier
    daily_raw_gb = effective_rate * 86_400 / 1024
    retained_gb = daily_raw_gb * profile.retention_days
    local_storage_days = profile.local_storage_mb / max(effective_rate * 86_400, 1e-9)
    stage_specs = (
        ("ingest", effective_rate, profile.ingest_capacity_mb_s),
        ("preprocess", effective_rate * 0.85, profile.preprocess_capacity_mb_s),
        ("storage write", effective_rate * 0.65, profile.storage_capacity_mb_s),
        ("upload/movement", effective_rate, profile.upload_capacity_mb_s),
    )
    stages = tuple(
        PipelineStageResult(
            stage=name,
            demand_mb_s=demand,
            capacity_mb_s=capacity,
            utilization_pct=demand / capacity * 100 if capacity > 0 else 100.0,
            feasible=demand <= capacity,
        )
        for name, demand, capacity in stage_specs
    )
    bottleneck = max(stages, key=lambda stage: stage.utilization_pct)
    starvation = max(0.0, 100.0 - min(100.0, 100.0 / max(1.0, bottleneck.utilization_pct)))
    storage_feasible = retained_gb * 1024 <= profile.local_storage_mb or profile.local_storage_mb <= 0
    return PipelineResult(
        sample_multiplier=multiplier,
        effective_rate_mb_s=effective_rate,
        daily_raw_gb=daily_raw_gb,
        retained_gb=retained_gb,
        local_storage_days=local_storage_days,
        bottleneck_stage=bottleneck.stage if storage_feasible else "retention storage",
        compute_starvation_pct=starvation,
        feasible=all(stage.feasible for stage in stages) and storage_feasible,
        stages=stages,
    )


def movement_frontier(
    profile: DataPipelineTrackProfile,
    *,
    strategy_id: str,
    dataset_gb: float,
    network_gbps: float,
) -> MovementFrontierResult:
    """Compare data movement under the selected strategy."""
    strategy = next((candidate for candidate in profile.strategies if candidate.strategy_id == strategy_id), profile.strategies[0])
    moved_gb = max(0.0, float(dataset_gb)) * strategy.data_reduction_factor
    network = max(0.001, float(network_gbps))
    transfer_hours = moved_gb * 8 / (network * 3600)
    egress_cost = moved_gb * profile.egress_cost_per_gb * strategy.cost_factor
    effective_latency = transfer_hours * 3600 * strategy.latency_factor
    quality = max(0.0, min(100.0, 100.0 * strategy.quality_factor))
    return MovementFrontierResult(
        strategy_id=strategy.strategy_id,
        strategy_label=strategy.label,
        data_moved_gb=moved_gb,
        transfer_hours=transfer_hours,
        egress_cost=egress_cost,
        effective_latency_s=effective_latency,
        quality_retained_pct=quality,
        privacy_risk=strategy.privacy_risk,
        residual_risk=strategy.residual_risk,
    )


def pipeline_architecture(
    profile: DataPipelineTrackProfile,
    pipeline: PipelineResult,
    movement: MovementFrontierResult,
    *,
    retention_policy: str,
) -> PipelineArchitectureResult:
    """Package the final data architecture memo decision."""
    policy = retention_policy or (profile.retention_options[0] if profile.retention_options else "retain enough evidence for audits")
    summary = (
        f"Use {movement.strategy_label}; bottleneck is {pipeline.bottleneck_stage}; "
        f"retain {pipeline.retained_gb:.1f} GB under policy '{policy}'."
    )
    return PipelineArchitectureResult(
        strategy_id=movement.strategy_id,
        strategy_label=movement.strategy_label,
        retention_policy=policy,
        bottleneck_stage=pipeline.bottleneck_stage,
        retained_gb=pipeline.retained_gb,
        quality_retained_pct=movement.quality_retained_pct,
        accepted_data_risk=movement.residual_risk,
        memo_summary=summary,
    )


__all__ = [
    "DataMovementStrategy",
    "DataPipelineTrackProfile",
    "MovementFrontierResult",
    "PipelineArchitectureResult",
    "PipelineResult",
    "PipelineStageResult",
    "data_pipeline_profile",
    "evaluate_pipeline",
    "movement_frontier",
    "pipeline_architecture",
]
