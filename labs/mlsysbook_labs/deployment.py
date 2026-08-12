"""Deployment-envelope helpers for track-aware physics labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class PlacementOption:
    placement_id: str
    label: str
    network_latency_ms: float
    memory_factor: float
    energy_factor: float
    bandwidth_factor: float
    cost_factor: float
    risk: str


@dataclass(frozen=True)
class DeploymentTrackProfile:
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    peak_tflops: float
    memory_capacity_mb: float
    sram_capacity_kib: float
    flash_capacity_mb: float
    memory_bandwidth_gbs: float
    tdp_w: float
    battery_wh: float
    model_parameters_m: float
    model_size_mb: float
    model_flops_g: float
    stakeholder: str
    envelope_story: str
    workload_knob: str
    workload_unit: str
    knob_min: float
    knob_max: float
    knob_step: float
    default_knob: float
    memory_budget_mb: float
    flash_budget_mb: float
    latency_budget_ms: float
    energy_budget_mj: float
    power_budget_w: float
    bandwidth_budget_gbs: float
    cost_budget_per_1k: float
    runtime_overhead_mb: float
    activation_mb_at_default: float
    latency_ms_at_default: float
    latency_exponent: float
    energy_mj_at_default: float
    power_w_at_default: float
    bandwidth_gbs_at_default: float
    cost_per_1k_at_default: float
    placement_options: tuple[PlacementOption, ...]
    mitigation_options: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    value: float
    limit: float
    unit: str
    headroom_pct: float
    feasible: bool


@dataclass(frozen=True)
class DeploymentEnvelopeResult:
    workload_value: float
    workload_multiplier: float
    placement_id: str
    placement_label: str
    memory_required_mb: float
    flash_required_mb: float
    latency_ms: float
    energy_mj: float
    power_w: float
    bandwidth_gbs: float
    cost_per_1k: float
    first_wall: str
    feasible: bool
    violations: tuple[str, ...]
    checks: tuple[ConstraintCheck, ...]


@dataclass(frozen=True)
class DeploymentSweepResult:
    knob_values: tuple[float, ...]
    worst_headroom_pct: tuple[float, ...]
    first_walls: tuple[str, ...]
    feasible: tuple[bool, ...]
    threshold_crossing: float | None
    threshold_wall: str


@dataclass(frozen=True)
class MitigationResult:
    placement_id: str
    placement_label: str
    binding_constraint: str
    mitigation: str
    avoided_wall: str
    new_risk: str
    feasible_after_mitigation: bool


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


def _peak_tflops(hardware: Any, precision: str) -> float:
    peak = None
    precision_flops = getattr(getattr(hardware, "compute", None), "precision_flops", {}) or {}
    if precision:
        peak = precision_flops.get(precision)
    if peak is None:
        peak = getattr(getattr(hardware, "compute", None), "peak_flops", None)
    value = _quantity_to_float(peak, "TFLOPs/s", 0.0)
    if value == 0.0:
        value = _quantity_to_float(peak, "TOPS", 0.0)
    return value


def _memory_capacity_mb(hardware: Any) -> float:
    memory = getattr(hardware, "memory", None)
    return _quantity_to_float(getattr(memory, "capacity", None), "MB", 0.0)


def _model_parameters(model: Any) -> float:
    return _quantity_to_float(getattr(model, "parameters", None), "count", 0.0)


def _model_flops(model: Any) -> float:
    return _quantity_to_float(getattr(model, "inference_flops", None), "flop", 0.0)


def _placement_options(defaults: Mapping[str, Any]) -> tuple[PlacementOption, ...]:
    raw_options = defaults.get("placement_options", {})
    if not isinstance(raw_options, Mapping):
        raw_options = {}
    options: list[PlacementOption] = []
    for placement_id, raw in raw_options.items():
        details = raw if isinstance(raw, Mapping) else {}
        options.append(
            PlacementOption(
                placement_id=str(placement_id),
                label=str(details.get("label", placement_id)),
                network_latency_ms=float(details.get("network_latency_ms", 0.0)),
                memory_factor=float(details.get("memory_factor", 1.0)),
                energy_factor=float(details.get("energy_factor", 1.0)),
                bandwidth_factor=float(details.get("bandwidth_factor", 1.0)),
                cost_factor=float(details.get("cost_factor", 1.0)),
                risk=str(details.get("risk", "new operational risk")),
            )
        )
    if options:
        return tuple(options)
    return (
        PlacementOption("local", "Local deployment", 0.0, 1.0, 1.0, 1.0, 1.0, "local resource pressure"),
        PlacementOption("cloud", "Cloud offload", 25.0, 0.5, 2.0, 0.2, 1.2, "network and privacy exposure"),
    )


def deployment_track_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
) -> DeploymentTrackProfile:
    """Build a deployment-envelope profile from MLSysIM refs and variant defaults."""
    defaults = variant.defaults
    precision_bytes = int(defaults.get("precision_bytes", 2))
    precision = str(defaults.get("precision", ""))
    parameters = _model_parameters(model)
    flops = _model_flops(model)
    model_size_mb = parameters * precision_bytes / 1_000_000
    memory = getattr(hardware, "memory", None)
    return DeploymentTrackProfile(
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        peak_tflops=_peak_tflops(hardware, precision),
        memory_capacity_mb=_memory_capacity_mb(hardware),
        sram_capacity_kib=_quantity_to_float(getattr(memory, "sram_capacity", None), "KiB", 0.0),
        flash_capacity_mb=_quantity_to_float(getattr(memory, "flash_capacity", None), "MB", 0.0),
        memory_bandwidth_gbs=_quantity_to_float(getattr(memory, "bandwidth", None), "GB/s", 0.0),
        tdp_w=_quantity_to_float(getattr(hardware, "tdp", None), "W", 0.0),
        battery_wh=_quantity_to_float(getattr(hardware, "battery_capacity", None), "Wh", 0.0),
        model_parameters_m=parameters / 1_000_000,
        model_size_mb=model_size_mb,
        model_flops_g=flops / 1_000_000_000,
        stakeholder=variant.stakeholder,
        envelope_story=str(defaults.get("envelope_story", variant.workload_summary)),
        workload_knob=str(defaults.get("workload_knob", "workload intensity")),
        workload_unit=str(defaults.get("workload_unit", "units")),
        knob_min=float(defaults.get("knob_min", 1.0)),
        knob_max=float(defaults.get("knob_max", 100.0)),
        knob_step=float(defaults.get("knob_step", 1.0)),
        default_knob=float(defaults.get("default_knob", 10.0)),
        memory_budget_mb=float(defaults.get("memory_budget_mb", _memory_capacity_mb(hardware) * 0.25)),
        flash_budget_mb=float(defaults.get("flash_budget_mb", 1_000_000.0)),
        latency_budget_ms=float(defaults.get("latency_budget_ms", 100.0)),
        energy_budget_mj=float(defaults.get("energy_budget_mj", 1_000_000.0)),
        power_budget_w=float(defaults.get("power_budget_w", max(1.0, _quantity_to_float(getattr(hardware, "tdp", None), "W", 1.0)))),
        bandwidth_budget_gbs=float(defaults.get("bandwidth_budget_gbs", max(0.001, _quantity_to_float(getattr(memory, "bandwidth", None), "GB/s", 1.0) * 0.8))),
        cost_budget_per_1k=float(defaults.get("cost_budget_per_1k", 1_000_000.0)),
        runtime_overhead_mb=float(defaults.get("runtime_overhead_mb", 0.0)),
        activation_mb_at_default=float(defaults.get("activation_mb_at_default", model_size_mb)),
        latency_ms_at_default=float(defaults.get("latency_ms_at_default", 10.0)),
        latency_exponent=float(defaults.get("latency_exponent", 1.0)),
        energy_mj_at_default=float(defaults.get("energy_mj_at_default", 1.0)),
        power_w_at_default=float(defaults.get("power_w_at_default", max(0.01, _quantity_to_float(getattr(hardware, "tdp", None), "W", 1.0) * 0.5))),
        bandwidth_gbs_at_default=float(defaults.get("bandwidth_gbs_at_default", 0.1)),
        cost_per_1k_at_default=float(defaults.get("cost_per_1k_at_default", 0.0)),
        placement_options=_placement_options(defaults),
        mitigation_options=tuple(str(item) for item in defaults.get("mitigation_options", ())),
        report_artifact=str(variant.assumptions.get("report_artifact", "physics-of-deployment memo")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=tuple(ref for ref in (variant.hardware_ref, variant.model_ref, variant.system_ref) if ref),
    )


def _placement(profile: DeploymentTrackProfile, placement_id: str | None) -> PlacementOption:
    if placement_id:
        for option in profile.placement_options:
            if option.placement_id == placement_id:
                return option
    return profile.placement_options[0]


def evaluate_deployment_envelope(
    profile: DeploymentTrackProfile,
    *,
    workload_value: float,
    placement_id: str | None = None,
) -> DeploymentEnvelopeResult:
    """Evaluate memory, latency, energy, power, bandwidth, and cost limits."""
    workload = max(profile.knob_min, min(profile.knob_max, float(workload_value)))
    baseline = max(profile.default_knob, 1e-9)
    multiplier = max(0.001, workload / baseline)
    placement = _placement(profile, placement_id)

    memory_required = (
        profile.model_size_mb
        + profile.runtime_overhead_mb
        + profile.activation_mb_at_default * multiplier
    ) * placement.memory_factor
    flash_required = profile.model_size_mb + profile.runtime_overhead_mb * 0.35
    latency = profile.latency_ms_at_default * (multiplier**profile.latency_exponent) + placement.network_latency_ms
    energy = profile.energy_mj_at_default * multiplier * placement.energy_factor
    power = profile.power_w_at_default * (multiplier**1.05)
    bandwidth = profile.bandwidth_gbs_at_default * multiplier * placement.bandwidth_factor
    cost = profile.cost_per_1k_at_default * multiplier * placement.cost_factor

    raw_checks = (
        ("memory", memory_required, profile.memory_budget_mb, "MB"),
        ("flash/OTA", flash_required, profile.flash_budget_mb, "MB"),
        ("latency", latency, profile.latency_budget_ms, "ms"),
        ("energy", energy, profile.energy_budget_mj, "mJ"),
        ("power", power, profile.power_budget_w, "W"),
        ("bandwidth", bandwidth, profile.bandwidth_budget_gbs, "GB/s"),
        ("cost", cost, profile.cost_budget_per_1k, "$/1K"),
    )
    checks = tuple(
        ConstraintCheck(
            name=name,
            value=value,
            limit=limit,
            unit=unit,
            headroom_pct=(limit - value) / limit * 100 if limit > 0 else 0.0,
            feasible=value <= limit,
        )
        for name, value, limit, unit in raw_checks
    )
    worst = min(checks, key=lambda check: check.headroom_pct)
    violations = tuple(
        f"{check.name}: {check.value:.3g} {check.unit} > {check.limit:.3g} {check.unit}"
        for check in checks
        if not check.feasible
    )
    return DeploymentEnvelopeResult(
        workload_value=workload,
        workload_multiplier=multiplier,
        placement_id=placement.placement_id,
        placement_label=placement.label,
        memory_required_mb=memory_required,
        flash_required_mb=flash_required,
        latency_ms=latency,
        energy_mj=energy,
        power_w=power,
        bandwidth_gbs=bandwidth,
        cost_per_1k=cost,
        first_wall=worst.name,
        feasible=not violations,
        violations=violations,
        checks=checks,
    )


def sweep_deployment_knob(
    profile: DeploymentTrackProfile,
    *,
    placement_id: str | None = None,
    samples: int = 36,
) -> DeploymentSweepResult:
    """Sweep the track's workload knob and return the first feasibility crossing."""
    count = max(2, int(samples))
    span = profile.knob_max - profile.knob_min
    values = tuple(profile.knob_min + span * idx / (count - 1) for idx in range(count))
    results = tuple(
        evaluate_deployment_envelope(profile, workload_value=value, placement_id=placement_id)
        for value in values
    )
    crossing = next((result for result in results if not result.feasible), None)
    return DeploymentSweepResult(
        knob_values=tuple(result.workload_value for result in results),
        worst_headroom_pct=tuple(min(check.headroom_pct for check in result.checks) for result in results),
        first_walls=tuple(result.first_wall for result in results),
        feasible=tuple(result.feasible for result in results),
        threshold_crossing=crossing.workload_value if crossing else None,
        threshold_wall=crossing.first_wall if crossing else "none within sweep",
    )


def deployment_mitigation(
    profile: DeploymentTrackProfile,
    result: DeploymentEnvelopeResult,
    *,
    placement_id: str | None = None,
) -> MitigationResult:
    """Return the mitigation memo for the active wall and placement choice."""
    placement = _placement(profile, placement_id or result.placement_id)
    options = profile.mitigation_options
    match = next((item for item in options if result.first_wall.lower() in item.lower()), "")
    mitigation = match or (options[0] if options else f"Reduce {result.first_wall} pressure and rerun the envelope.")
    mitigated = result.feasible or any(token in mitigation.lower() for token in ("reduce", "simplify", "cache", "batch", "local", "shard"))
    return MitigationResult(
        placement_id=placement.placement_id,
        placement_label=placement.label,
        binding_constraint=result.first_wall,
        mitigation=mitigation,
        avoided_wall=result.first_wall,
        new_risk=placement.risk,
        feasible_after_mitigation=mitigated,
    )


__all__ = [
    "ConstraintCheck",
    "DeploymentEnvelopeResult",
    "DeploymentSweepResult",
    "DeploymentTrackProfile",
    "MitigationResult",
    "PlacementOption",
    "deployment_mitigation",
    "deployment_track_profile",
    "evaluate_deployment_envelope",
    "sweep_deployment_knob",
]
