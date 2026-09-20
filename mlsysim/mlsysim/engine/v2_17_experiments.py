"""Cross-layer fleet synthesis for the Volume II capstone lab.

This module composes the validated Chapter 1, 7, 10, and 15 experiment
backends.  It does not replace their physical models.  The built-in track
configurations are explicitly illustrative instructor scenarios, not measured
production fleets or branded-hardware specifications.

For TinyML and Mobile tracks, C3 scaling and recovery reflect the backend
training and checkpoint cluster supporting the fleet, deployment serving
models the gateway/device inference pool, and facility checks enforce the
backend/system energy boundary rather than an impossible MCU collective.

Evidence compatibility and physical feasibility are deliberately independent.
A missing ledger entry means that a deployment claim lacks student evidence;
it does not make a simulated fleet fail a physical constraint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any, Literal

from mlsysim.core.units import Q_, ureg
from mlsysim.engine.v2_01_experiments import evaluate_c3_scaling
from mlsysim.engine.v2_01_experiments import restore_quantities as restore_v2_01_quantities
from mlsysim.engine.v2_07_experiments import checkpoint_tradeoff
from mlsysim.engine.v2_10_experiments import (
    Request,
    ServiceModel,
    illustrative_track_scenario,
    replay_requests,
    deserialize_quantity as deserialize_v2_10_quantity,
    serialize_quantity as serialize_v2_10_quantity,
)
from mlsysim.engine.v2_15_experiments import (
    calculate_energy,
    check_facility,
    deserialize_experiment_value as deserialize_v2_15_value,
    evaluate_mitigation,
)


TrackId = Literal["tinyml", "mobile", "edge", "cloud"]
Decision = Literal["release", "restrict", "defer"]
EvidenceStatus = Literal["ledger", "instructor_scenario", "missing", "incompatible"]

REQUIRED_MODELS: dict[int, str] = {
    1: "v2_01_experiments",
    7: "v2_07_experiments",
    10: "v2_10_experiments",
    15: "v2_15_experiments",
}
REPLAY_TARGETS = {
    "v2_01_experiments": ("B", "v2_01_experiments.evaluate_c3_scaling"),
    "v2_07_experiments": ("E", "v2_07_experiments.checkpoint_tradeoff"),
    "v2_10_experiments": ("E", "v2_10_experiments.replay_requests"),
    "v2_15_experiments": ("D", "v2_15_experiments.evaluate_mitigation"),
}


def _quantity(value: Any, unit: str, name: str, *, positive: bool = False):
    if not isinstance(value, ureg.Quantity):
        raise TypeError(f"{name} must be a Pint Quantity")
    try:
        result = value.to(unit)
    except Exception as exc:
        raise ValueError(f"{name} must be compatible with {unit}") from exc
    magnitude = float(result.magnitude)
    if not math.isfinite(magnitude) or magnitude < 0 or (positive and magnitude == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


@dataclass(frozen=True)
class C3Configuration:
    workers: int
    single_worker_compute_time: Any
    communication_payload: Any
    effective_bandwidth: Any
    communication_startup: Any
    coordination_base: Any
    coordination_per_added_worker: Any
    useful_work_per_step: Any
    overlap_fraction: float


@dataclass(frozen=True)
class RecoveryConfiguration:
    interval_min: float
    state_gb: float
    write_bandwidth_gbs: float
    checkpoint_pause_s: float | None = None
    detection_s: float | None = None
    restart_s: float | None = None
    load_s: float | None = None
    warmup_s: float | None = None


@dataclass(frozen=True)
class ServingConfiguration:
    requests: tuple[Request, ...]
    service: ServiceModel
    replicas: int
    policy: Literal["immediate", "windowed_batch"]
    slo: Any
    max_batch: int = 1
    batch_window: Any = Q_(0, "millisecond")
    warmup: Any = Q_(0, "millisecond")
    lost_replicas: int = 0
    service_scale: float = 1.0
    handoff_time: Any = Q_(0, "millisecond")
    incremental_replica_energy: Any = Q_(0, "joule")
    incremental_replica_power: Any = Q_(0, "watt")


@dataclass(frozen=True)
class EnergyConfiguration:
    operations: Any
    moved_bytes: Any
    energy_per_operation: Any
    energy_per_byte: Any
    pue: float
    other_it_energy: Any
    it_power: Any
    electrical_capacity: Any
    cooling_capacity: Any


@dataclass(frozen=True)
class FleetLimits:
    c3_step_time: Any
    recovery_time: Any
    recovery_waste_fraction: float
    serving_p99: Any
    facility_energy: Any


@dataclass(frozen=True)
class FleetConfiguration:
    """Causal inputs for one reproducible cross-layer experiment."""

    track_id: TrackId
    scenario_label: str
    scenario_origin: Literal["instructor_scenario", "ledger_replay"]
    c3: C3Configuration
    recovery: RecoveryConfiguration
    serving: ServingConfiguration
    energy: EnergyConfiguration
    limits: FleetLimits


@dataclass(frozen=True)
class ConstraintCheck:
    constraint: str
    observed: Any
    limit: Any
    passed: bool | None
    calculation: str


@dataclass(frozen=True)
class FleetEvaluation:
    configuration: FleetConfiguration
    c3_result: Any
    recovery_result: Mapping[str, Any]
    serving_result: Any
    energy_result: Any
    facility_result: Any
    checks: tuple[ConstraintCheck, ...]

    @property
    def physical_feasible(self) -> bool | None:
        if any(check.passed is None for check in self.checks):
            return None
        return all(check.passed is True for check in self.checks)

    @property
    def failed_constraints(self) -> tuple[str, ...]:
        return tuple(check.constraint for check in self.checks if check.passed is False)

    @property
    def unevaluated_constraints(self) -> tuple[str, ...]:
        return tuple(check.constraint for check in self.checks if check.passed is None)


@dataclass(frozen=True)
class EvidenceItem:
    chapter: int
    expected_model_id: str
    status: EvidenceStatus
    source_label: str
    entry: Mapping[str, Any] | None
    issue: str | None = None


@dataclass(frozen=True)
class EvidenceAudit:
    items: tuple[EvidenceItem, ...]

    @property
    def ready(self) -> bool:
        return all(item.status in ("ledger", "instructor_scenario") for item in self.items)

    @property
    def missing_chapters(self) -> tuple[int, ...]:
        return tuple(item.chapter for item in self.items if item.status == "missing")

    @property
    def incompatible_chapters(self) -> tuple[int, ...]:
        return tuple(item.chapter for item in self.items if item.status == "incompatible")


@dataclass(frozen=True)
class ControlPatch:
    """Multipliers and deltas applied independently to one frozen baseline."""

    bandwidth_multiplier: float = 1.0
    coordination_multiplier: float = 1.0
    recovery_time_multiplier: float = 1.0
    extra_replicas: int = 0
    lost_replicas_delta: int = 0
    serving_time_multiplier: float = 1.0
    energy_work_multiplier: float = 1.0
    energy_efficiency_multiplier: float = 1.0
    other_it_energy_delta: Any = Q_(0, "joule")
    it_power_multiplier: float = 1.0


@dataclass(frozen=True)
class RepairPlan:
    repair_id: str
    label: str
    cost: Any
    patch: ControlPatch
    consequence: str


@dataclass(frozen=True)
class RepairComparison:
    failing_evaluation: FleetEvaluation
    stress: ControlPatch
    repairs: tuple[tuple[RepairPlan, FleetEvaluation], ...]
    common_budget: Any


@dataclass(frozen=True)
class DeploymentRecord:
    decision: Decision
    selected_repair: str
    rejected_alternative: str
    condition: str
    residual_uncertainty: str


_TRACK_ASSUMPTIONS = {
    # Values are compact browser-runnable teaching fixtures, not measurements.
    "tinyml": dict(
        workers=8,
        compute_s=8.0,
        payload_gb=0.20,
        bandwidth_gbs=2.0,
        startup_ms=2.0,
        coord_ms=1.0,
        coord_worker_ms=0.30,
        replicas=2,
        operations=1e15,
        moved_bytes=1e12,
        other_kj=0.5,
        it_power_kw=4.0,
        electrical_kw=6.0,
        cooling_kw=5.0,
        replica_energy_kj=0.15,
        replica_power_kw=0.35,
        c3_budget_s=1.2,
        energy_budget_kwh=0.003,
    ),
    "mobile": dict(
        workers=16,
        compute_s=16.0,
        payload_gb=0.50,
        bandwidth_gbs=4.0,
        startup_ms=2.0,
        coord_ms=1.5,
        coord_worker_ms=0.35,
        replicas=2,
        operations=5e15,
        moved_bytes=4e12,
        other_kj=1.0,
        it_power_kw=12.0,
        electrical_kw=18.0,
        cooling_kw=15.0,
        replica_energy_kj=0.75,
        replica_power_kw=1.0,
        c3_budget_s=1.25,
        energy_budget_kwh=0.013,
    ),
    "edge": dict(
        workers=32,
        compute_s=28.0,
        payload_gb=1.5,
        bandwidth_gbs=8.0,
        startup_ms=1.5,
        coord_ms=2.0,
        coord_worker_ms=0.40,
        replicas=3,
        operations=3e16,
        moved_bytes=2e13,
        other_kj=3.0,
        it_power_kw=40.0,
        electrical_kw=60.0,
        cooling_kw=48.0,
        replica_energy_kj=3.5,
        replica_power_kw=4.0,
        c3_budget_s=1.2,
        energy_budget_kwh=0.070,
    ),
    "cloud": dict(
        workers=64,
        compute_s=52.0,
        payload_gb=4.0,
        bandwidth_gbs=16.0,
        startup_ms=1.0,
        coord_ms=2.5,
        coord_worker_ms=0.45,
        replicas=4,
        operations=5e17,
        moved_bytes=2e14,
        other_kj=15.0,
        it_power_kw=250.0,
        electrical_kw=380.0,
        cooling_kw=300.0,
        replica_energy_kj=40.0,
        replica_power_kw=22.0,
        c3_budget_s=1.15,
        energy_budget_kwh=1.2,
    ),
}


def instructor_scenario(track_id: TrackId) -> FleetConfiguration:
    """Build one explicitly labeled scenario by composing earlier modules' inputs."""
    if track_id not in _TRACK_ASSUMPTIONS:
        raise ValueError(f"unknown track_id: {track_id}")
    values = _TRACK_ASSUMPTIONS[track_id]
    serving = illustrative_track_scenario(track_id)
    recovery_state_gb = {
        "tinyml": 12.0,
        "mobile": 80.0,
        "edge": 480.0,
        "cloud": 1400.0,
    }[track_id]
    recovery_bandwidth_gbs = {
        "tinyml": 0.2,
        "mobile": 1.0,
        "edge": 8.0,
        "cloud": 25.0,
    }[track_id]
    resolved_recovery = checkpoint_tradeoff(
        track_id,
        interval_min=30.0,
        state_gb=recovery_state_gb,
        write_bandwidth_gbs=recovery_bandwidth_gbs,
    )
    labels = {
        "tinyml": (
            "TinyML fleet synthesis: backend training/recovery, gateway serving"
            " pool, facility/system energy boundary (illustrative instructor"
            " scenario)"
        ),
        "mobile": (
            "Mobile fleet synthesis: backend training/recovery, deployment"
            " serving pool, facility/system energy boundary (illustrative"
            " instructor scenario)"
        ),
        "edge": (
            "Edge fleet synthesis: regional backend training/recovery,"
            " deployment serving pool, facility energy boundary (illustrative"
            " instructor scenario)"
        ),
        "cloud": (
            "Cloud fleet synthesis: data center training/recovery, deployment"
            " serving pool, facility energy boundary (illustrative instructor"
            " scenario)"
        ),
    }
    return FleetConfiguration(
        track_id=track_id,
        scenario_label=labels[track_id],
        scenario_origin="instructor_scenario",
        c3=C3Configuration(
            workers=values["workers"],
            single_worker_compute_time=Q_(values["compute_s"], "second"),
            communication_payload=Q_(values["payload_gb"], "gigabyte"),
            effective_bandwidth=Q_(values["bandwidth_gbs"], "gigabyte/second"),
            communication_startup=Q_(values["startup_ms"], "millisecond"),
            coordination_base=Q_(values["coord_ms"], "millisecond"),
            coordination_per_added_worker=Q_(values["coord_worker_ms"], "millisecond"),
            useful_work_per_step=Q_(1024, "count"),
            overlap_fraction=0.25,
        ),
        recovery=RecoveryConfiguration(
            interval_min=30.0,
            state_gb=recovery_state_gb,
            write_bandwidth_gbs=recovery_bandwidth_gbs,
            checkpoint_pause_s=resolved_recovery["checkpoint_pause_s"],
            detection_s=resolved_recovery["detection_s"],
            restart_s=resolved_recovery["restart_s"],
            load_s=resolved_recovery["load_s"],
            warmup_s=resolved_recovery["warmup_s"],
        ),
        serving=ServingConfiguration(
            requests=serving.requests,
            service=serving.service,
            replicas=values["replicas"],
            policy="immediate",
            slo=serving.slo,
            incremental_replica_energy=Q_(values["replica_energy_kj"], "kilojoule"),
            incremental_replica_power=Q_(values["replica_power_kw"], "kilowatt"),
        ),
        energy=EnergyConfiguration(
            operations=Q_(values["operations"], "flop"),
            moved_bytes=Q_(values["moved_bytes"], "byte"),
            energy_per_operation=Q_(6.0, "picojoule/flop"),
            energy_per_byte=Q_(45.0, "picojoule/byte"),
            pue=1.15,
            other_it_energy=Q_(values["other_kj"], "kilojoule"),
            it_power=Q_(values["it_power_kw"], "kilowatt"),
            electrical_capacity=Q_(values["electrical_kw"], "kilowatt"),
            cooling_capacity=Q_(values["cooling_kw"], "kilowatt"),
        ),
        limits=FleetLimits(
            c3_step_time=Q_(values["c3_budget_s"], "second"),
            recovery_time=Q_(15, "minute"),
            recovery_waste_fraction=0.25,
            serving_p99=serving.slo,
            facility_energy=Q_(values["energy_budget_kwh"], "kilowatt_hour"),
        ),
    )


def instructor_stress(severity: Literal["moderate", "severe"] = "severe") -> ControlPatch:
    """Return a disclosed cross-layer stress used by the capstone scenario."""
    stresses = {
        "moderate": ControlPatch(
            bandwidth_multiplier=0.45,
            coordination_multiplier=2.0,
            energy_work_multiplier=1.45,
        ),
        "severe": ControlPatch(
            bandwidth_multiplier=0.20,
            coordination_multiplier=4.0,
            energy_work_multiplier=2.0,
        ),
    }
    try:
        return stresses[severity]
    except KeyError as exc:
        raise ValueError("severity must be moderate or severe") from exc


def instructor_repair_plans(track_id: TrackId) -> tuple[RepairPlan, ...]:
    """Return three equal-cost repair assumptions with explicit side effects."""
    if track_id not in _TRACK_ASSUMPTIONS:
        raise ValueError(f"unknown track_id: {track_id}")
    budget = {
        "tinyml": Q_(25_000, "dollar"),
        "mobile": Q_(60_000, "dollar"),
        "edge": Q_(120_000, "dollar"),
        "cloud": Q_(300_000, "dollar"),
    }[track_id]
    fabric_energy = {
        "tinyml": Q_(0.2, "kilojoule"),
        "mobile": Q_(1.0, "kilojoule"),
        "edge": Q_(5.0, "kilojoule"),
        "cloud": Q_(60.0, "kilojoule"),
    }[track_id]
    return (
        RepairPlan(
            "fabric",
            "Increase fabric margin",
            budget,
            ControlPatch(
                bandwidth_multiplier=5.0,
                coordination_multiplier=0.25,
                other_it_energy_delta=fabric_energy,
            ),
            "The C3 path improves while fabric energy increases.",
        ),
        RepairPlan(
            "resilience",
            "Add a serving replica and shorten recovery",
            budget,
            ControlPatch(recovery_time_multiplier=0.65, extra_replicas=1),
            "Recovery and serving headroom improve while power and energy increase.",
        ),
        RepairPlan(
            "efficiency",
            "Reduce execution and energy per useful unit",
            budget,
            ControlPatch(
                serving_time_multiplier=0.80,
                energy_efficiency_multiplier=0.55,
            ),
            "Serving and energy improve while the stressed C3 path remains unchanged.",
        ),
    )


def adverse_condition(
    condition: Literal["fabric", "recovery", "replica_loss", "demand"]
) -> ControlPatch:
    """Return one held-out operating condition for a repaired configuration."""
    conditions = {
        "fabric": ControlPatch(
            bandwidth_multiplier=0.50, coordination_multiplier=1.5
        ),
        "recovery": ControlPatch(recovery_time_multiplier=1.5),
        "replica_loss": ControlPatch(lost_replicas_delta=1),
        "demand": ControlPatch(
            energy_work_multiplier=1.25, it_power_multiplier=1.10
        ),
    }
    try:
        return conditions[condition]
    except KeyError as exc:
        raise ValueError(
            "condition must be fabric, recovery, replica_loss, or demand"
        ) from exc


def obligation_stress(
    serving_level: Literal["busy", "surge", "overload"],
    energy_level: Literal["steady", "growth", "double"],
) -> ControlPatch:
    """Combine explicit serving-time and useful-work stress assumptions."""
    serving_multipliers = {"busy": 5.0, "surge": 20.0, "overload": 100.0}
    energy_multipliers = {"steady": 1.0, "growth": 1.5, "double": 2.0}
    try:
        return ControlPatch(
            serving_time_multiplier=serving_multipliers[serving_level],
            energy_work_multiplier=energy_multipliers[energy_level],
        )
    except KeyError as exc:
        raise ValueError("unknown serving or energy stress level") from exc


def evaluate_configuration(configuration: FleetConfiguration) -> FleetEvaluation:
    """Rerun all four physical models and compare their native-unit outputs."""
    c3 = configuration.c3
    c3_result = evaluate_c3_scaling(
        workers=c3.workers,
        single_worker_compute_time=c3.single_worker_compute_time,
        communication_payload=c3.communication_payload,
        effective_bandwidth=c3.effective_bandwidth,
        communication_startup=c3.communication_startup,
        coordination_base=c3.coordination_base,
        coordination_per_added_worker=c3.coordination_per_added_worker,
        useful_work_per_step=c3.useful_work_per_step,
        overlap_fraction=c3.overlap_fraction,
    )
    recovery = configuration.recovery
    recovery_result = checkpoint_tradeoff(
        configuration.track_id,
        interval_min=recovery.interval_min,
        state_gb=recovery.state_gb,
        write_bandwidth_gbs=recovery.write_bandwidth_gbs,
        checkpoint_pause_s=recovery.checkpoint_pause_s,
        detection_s=recovery.detection_s,
        restart_s=recovery.restart_s,
        load_s=recovery.load_s,
        warmup_s=recovery.warmup_s,
    )
    serving = configuration.serving
    serving_result = replay_requests(
        serving.requests,
        service=serving.service,
        replicas=serving.replicas,
        policy=serving.policy,
        slo=serving.slo,
        max_batch=serving.max_batch,
        batch_window=serving.batch_window,
        warmup=serving.warmup,
        lost_replicas=serving.lost_replicas,
        service_scale=serving.service_scale,
        handoff_time=serving.handoff_time,
    )
    energy = configuration.energy
    energy_result = calculate_energy(
        operations=energy.operations,
        moved_bytes=energy.moved_bytes,
        energy_per_operation=energy.energy_per_operation,
        energy_per_byte=energy.energy_per_byte,
        pue=energy.pue,
        other_it_energy=energy.other_it_energy,
    )
    facility_result = check_facility(
        it_power=energy.it_power,
        pue=energy.pue,
        electrical_capacity=energy.electrical_capacity,
        cooling_capacity=energy.cooling_capacity,
    )
    limits = configuration.limits
    recovery_time = Q_(recovery_result["recovery_s"], "second")
    recovery_waste = recovery_result["total_waste_fraction"]
    recovery_valid = recovery_result["loss_approximation_valid"]
    checks = (
        ConstraintCheck(
            "C3 step time",
            c3_result.step_time,
            limits.c3_step_time,
            c3_result.step_time <= limits.c3_step_time,
            "Chapter 1 fixed-work C3 decomposition",
        ),
        ConstraintCheck(
            "recovery time",
            recovery_time,
            limits.recovery_time,
            recovery_time <= limits.recovery_time if recovery_valid else None,
            "Chapter 7 checkpoint and recovery replay",
        ),
        ConstraintCheck(
            "recovery waste",
            recovery_waste,
            limits.recovery_waste_fraction,
            recovery_waste <= limits.recovery_waste_fraction if recovery_valid else None,
            "Chapter 7 steady-state wall-clock loss fraction",
        ),
        ConstraintCheck(
            "serving p99",
            serving_result.p99_duration,
            limits.serving_p99,
            (serving_result.p99_duration <= limits.serving_p99) if serving_result.completed_count else None,
            "Chapter 10 empirical nearest-rank request replay",
        ),
        ConstraintCheck(
            "serving completion",
            Q_(serving_result.completed_count, "count"),
            Q_(serving_result.request_count, "count"),
            serving_result.completed_count == serving_result.request_count,
            "Chapter 10 completed requests in the explicit trace",
        ),
        ConstraintCheck(
            "absolute facility energy",
            energy_result.facility_energy,
            limits.facility_energy,
            energy_result.facility_energy <= limits.facility_energy,
            "Chapter 15 operations-plus-movement energy",
        ),
        ConstraintCheck(
            "electrical capacity",
            facility_result.facility_power,
            facility_result.electrical_capacity,
            facility_result.electrical_feasible,
            "Chapter 15 facility electrical boundary",
        ),
        ConstraintCheck(
            "cooling capacity",
            facility_result.heat_load,
            facility_result.cooling_capacity,
            facility_result.cooling_feasible,
            "Chapter 15 heat-rejection boundary",
        ),
    )
    return FleetEvaluation(
        configuration,
        c3_result,
        recovery_result,
        serving_result,
        energy_result,
        facility_result,
        checks,
    )


def apply_patch(configuration: FleetConfiguration, patch: ControlPatch) -> FleetConfiguration:
    """Apply a causal stress or repair without mutating the saved configuration."""
    multipliers = {
        "bandwidth_multiplier": patch.bandwidth_multiplier,
        "coordination_multiplier": patch.coordination_multiplier,
        "recovery_time_multiplier": patch.recovery_time_multiplier,
        "serving_time_multiplier": patch.serving_time_multiplier,
        "energy_work_multiplier": patch.energy_work_multiplier,
        "energy_efficiency_multiplier": patch.energy_efficiency_multiplier,
        "it_power_multiplier": patch.it_power_multiplier,
    }
    for name, value in multipliers.items():
        _positive(value, name)
    if isinstance(patch.extra_replicas, bool) or not isinstance(patch.extra_replicas, int):
        raise ValueError("extra_replicas must be an integer")
    if isinstance(patch.lost_replicas_delta, bool) or not isinstance(patch.lost_replicas_delta, int):
        raise ValueError("lost_replicas_delta must be an integer")
    other_delta = _quantity(patch.other_it_energy_delta, "joule", "other_it_energy_delta")

    c3 = configuration.c3
    recovery = configuration.recovery
    serving = configuration.serving
    energy = configuration.energy
    recovery_fields = {}
    for field in ("detection_s", "restart_s", "load_s", "warmup_s"):
        value = getattr(recovery, field)
        if value is not None:
            recovery_fields[field] = value * patch.recovery_time_multiplier
    replicas = serving.replicas + patch.extra_replicas
    lost = serving.lost_replicas + patch.lost_replicas_delta
    if replicas < 1 or lost < 0 or lost >= replicas:
        raise ValueError("patch must leave at least one serving replica available")
    replica_energy = (
        _quantity(serving.incremental_replica_energy, "joule", "incremental_replica_energy") * patch.extra_replicas
    )
    replica_power = (
        _quantity(serving.incremental_replica_power, "watt", "incremental_replica_power") * patch.extra_replicas
    )
    updated_other_energy = (energy.other_it_energy * patch.energy_work_multiplier + other_delta + replica_energy).to(
        "joule"
    )
    updated_it_power = (energy.it_power * patch.it_power_multiplier + replica_power).to("watt")
    if updated_other_energy.magnitude < 0 or updated_it_power.magnitude < 0:
        raise ValueError("replica removal cannot make energy or power negative")
    return replace(
        configuration,
        c3=replace(
            c3,
            effective_bandwidth=c3.effective_bandwidth * patch.bandwidth_multiplier,
            coordination_base=c3.coordination_base * patch.coordination_multiplier,
            coordination_per_added_worker=(c3.coordination_per_added_worker * patch.coordination_multiplier),
        ),
        recovery=replace(recovery, **recovery_fields),
        serving=replace(
            serving,
            replicas=replicas,
            lost_replicas=lost,
            service_scale=serving.service_scale * patch.serving_time_multiplier,
        ),
        energy=replace(
            energy,
            operations=energy.operations * patch.energy_work_multiplier,
            moved_bytes=energy.moved_bytes * patch.energy_work_multiplier,
            energy_per_operation=(energy.energy_per_operation * patch.energy_efficiency_multiplier),
            energy_per_byte=energy.energy_per_byte * patch.energy_efficiency_multiplier,
            other_it_energy=(updated_other_energy),
            it_power=updated_it_power,
        ),
    )


def compare_equal_budget_repairs(
    configuration: FleetConfiguration,
    *,
    stress: ControlPatch,
    repairs: Sequence[RepairPlan],
) -> RepairComparison:
    """Stress once, then rerun each equal-cost repair from that frozen failure."""
    plans = tuple(repairs)
    if len(plans) < 2:
        raise ValueError("at least two repairs are required")
    costs = [_quantity(plan.cost, "dollar", "repair cost", positive=True) for plan in plans]
    common = costs[0]
    if any(not math.isclose(cost.magnitude, common.magnitude, rel_tol=1e-9) for cost in costs[1:]):
        raise ValueError("repair plans must use the same resource budget")
    stressed_configuration = apply_patch(configuration, stress)
    failing = evaluate_configuration(stressed_configuration)
    if failing.physical_feasible is True:
        raise ValueError("stress case must violate at least one physical constraint")
    if failing.physical_feasible is None:
        raise ValueError("stress case must be evaluable before repairs are compared")
    evaluated = tuple((plan, evaluate_configuration(apply_patch(stressed_configuration, plan.patch))) for plan in plans)
    return RepairComparison(failing, stress, evaluated, common)


def _wire_quantity(value: Any) -> dict[str, Any]:
    if not isinstance(value, ureg.Quantity):
        raise TypeError("serialized value must be a Pint Quantity")
    magnitude = float(value.magnitude)
    if not math.isfinite(magnitude):
        raise ValueError("configuration quantities must be finite")
    return {"magnitude": magnitude, "unit": str(value.units)}


def serialize_configuration(configuration: FleetConfiguration) -> dict[str, Any]:
    """Serialize exact evaluator arguments without discarding physical units."""
    c3 = configuration.c3
    recovery = configuration.recovery
    serving = configuration.serving
    energy = configuration.energy
    limits = configuration.limits
    return {
        "track_id": configuration.track_id,
        "scenario_label": configuration.scenario_label,
        "scenario_origin": configuration.scenario_origin,
        "c3": {
            "workers": c3.workers,
            "single_worker_compute_time": _wire_quantity(c3.single_worker_compute_time),
            "communication_payload": _wire_quantity(c3.communication_payload),
            "effective_bandwidth": _wire_quantity(c3.effective_bandwidth),
            "communication_startup": _wire_quantity(c3.communication_startup),
            "coordination_base": _wire_quantity(c3.coordination_base),
            "coordination_per_added_worker": _wire_quantity(
                c3.coordination_per_added_worker
            ),
            "useful_work_per_step": _wire_quantity(c3.useful_work_per_step),
            "overlap_fraction": c3.overlap_fraction,
        },
        "recovery": {
            "interval_min": recovery.interval_min,
            "state_gb": recovery.state_gb,
            "write_bandwidth_gbs": recovery.write_bandwidth_gbs,
            "checkpoint_pause_s": recovery.checkpoint_pause_s,
            "detection_s": recovery.detection_s,
            "restart_s": recovery.restart_s,
            "load_s": recovery.load_s,
            "warmup_s": recovery.warmup_s,
        },
        "serving": {
            "requests": [request.to_evaluator_args() for request in serving.requests],
            "service": serving.service.to_evaluator_args(),
            "replicas": serving.replicas,
            "policy": serving.policy,
            "slo": serialize_v2_10_quantity(serving.slo),
            "max_batch": serving.max_batch,
            "batch_window": serialize_v2_10_quantity(serving.batch_window),
            "warmup": serialize_v2_10_quantity(serving.warmup),
            "lost_replicas": serving.lost_replicas,
            "service_scale": serving.service_scale,
            "handoff_time": serialize_v2_10_quantity(serving.handoff_time),
            "incremental_replica_energy": _wire_quantity(
                serving.incremental_replica_energy
            ),
            "incremental_replica_power": _wire_quantity(
                serving.incremental_replica_power
            ),
        },
        "energy": {
            "operations": _wire_quantity(energy.operations),
            "moved_bytes": _wire_quantity(energy.moved_bytes),
            "energy_per_operation": _wire_quantity(energy.energy_per_operation),
            "energy_per_byte": _wire_quantity(energy.energy_per_byte),
            "pue": energy.pue,
            "other_it_energy": _wire_quantity(energy.other_it_energy),
            "it_power": _wire_quantity(energy.it_power),
            "electrical_capacity": _wire_quantity(energy.electrical_capacity),
            "cooling_capacity": _wire_quantity(energy.cooling_capacity),
        },
        "limits": {
            "c3_step_time": _wire_quantity(limits.c3_step_time),
            "recovery_time": _wire_quantity(limits.recovery_time),
            "recovery_waste_fraction": limits.recovery_waste_fraction,
            "serving_p99": _wire_quantity(limits.serving_p99),
            "facility_energy": _wire_quantity(limits.facility_energy),
        },
    }


def deserialize_configuration(data: Mapping[str, Any]) -> FleetConfiguration:
    """Restore a configuration produced by :func:`serialize_configuration`."""
    c3 = dict(data["c3"])
    for key in (
        "single_worker_compute_time",
        "communication_payload",
        "effective_bandwidth",
        "communication_startup",
        "coordination_base",
        "coordination_per_added_worker",
        "useful_work_per_step",
    ):
        c3[key] = deserialize_v2_10_quantity(c3[key])
    serving = _restore_v2_10_inputs(dict(data["serving"]))
    for key in ("incremental_replica_energy", "incremental_replica_power"):
        serving[key] = deserialize_v2_10_quantity(serving[key])
    energy = {
        key: deserialize_v2_10_quantity(value) if isinstance(value, Mapping) else value
        for key, value in data["energy"].items()
    }
    limits = {
        key: deserialize_v2_10_quantity(value) if isinstance(value, Mapping) else value
        for key, value in data["limits"].items()
    }
    return FleetConfiguration(
        track_id=data["track_id"],
        scenario_label=str(data["scenario_label"]),
        scenario_origin=data["scenario_origin"],
        c3=C3Configuration(**c3),
        recovery=RecoveryConfiguration(**data["recovery"]),
        serving=ServingConfiguration(**serving),
        energy=EnergyConfiguration(**energy),
        limits=FleetLimits(**limits),
    )


def _safe_output(value: Any) -> Any:
    """Serialize results, representing undefined or infinite amounts as null."""
    if isinstance(value, ureg.Quantity):
        magnitude = float(value.magnitude)
        return {
            "magnitude": magnitude if math.isfinite(magnitude) else None,
            "unit": str(value.units),
            "finite": math.isfinite(magnitude),
        }
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe_output(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_output(item) for item in value]
    return value


def evaluation_record(
    configuration: FleetConfiguration, evaluation: FleetEvaluation
) -> dict[str, Any]:
    """Create a strict-JSON evidence record with replayable evaluator inputs."""
    status = (
        "FEASIBLE"
        if evaluation.physical_feasible is True
        else "INFEASIBLE"
        if evaluation.physical_feasible is False
        else "UNEVALUABLE"
    )
    return {
        "inputs": {"configuration": serialize_configuration(configuration)},
        "outputs": {
            "status": status,
            "failed_constraints": list(evaluation.failed_constraints),
            "unevaluated_constraints": list(evaluation.unevaluated_constraints),
            "c3": {
                "compute_time": _safe_output(evaluation.c3_result.compute_time),
                "communication_time": _safe_output(
                    evaluation.c3_result.communication_time
                ),
                "coordination_time": _safe_output(
                    evaluation.c3_result.coordination_time
                ),
                "step_time": _safe_output(evaluation.c3_result.step_time),
                "dominant_term": evaluation.c3_result.dominant_term,
            },
            "recovery": {
                "recovery_time": _safe_output(
                    Q_(evaluation.recovery_result["recovery_s"], "second")
                ),
                "waste_fraction": _safe_output(
                    evaluation.recovery_result["total_waste_fraction"]
                ),
                "valid": evaluation.recovery_result["loss_approximation_valid"],
            },
            "serving": {
                "p99_duration": _safe_output(
                    evaluation.serving_result.p99_duration
                ),
                "completed_count": evaluation.serving_result.completed_count,
                "request_count": evaluation.serving_result.request_count,
                "p99_method": evaluation.serving_result.p99_method,
            },
            "energy": {
                "facility_energy": _safe_output(
                    evaluation.energy_result.facility_energy
                ),
                "facility_power": _safe_output(
                    evaluation.facility_result.facility_power
                ),
                "heat_load": _safe_output(evaluation.facility_result.heat_load),
            },
        },
    }


def audit_ledger_evidence(
    history: Mapping[Any, Any],
    *,
    track_id: TrackId,
    instructor_fallbacks: Mapping[int, Mapping[str, Any]] | None = None,
) -> EvidenceAudit:
    """Audit stable ledger records without converting coverage into a score."""
    fallbacks = instructor_fallbacks or {}
    items = []
    for chapter, model_id in REQUIRED_MODELS.items():
        entry = history.get(chapter, history.get(str(chapter))) if isinstance(history, Mapping) else None
        if entry is None:
            fallback = fallbacks.get(chapter)
            if fallback is not None:
                items.append(
                    EvidenceItem(
                        chapter, model_id, "instructor_scenario", "explicitly labeled instructor scenario", fallback
                    )
                )
            else:
                items.append(
                    EvidenceItem(chapter, model_id, "missing", "no evidence", None, "required ledger chapter is absent")
                )
            continue
        issue = _ledger_issue(
            entry,
            chapter=chapter,
            track_id=track_id,
            expected_model_id=model_id,
        )
        if issue:
            items.append(
                EvidenceItem(
                    chapter,
                    model_id,
                    "incompatible",
                    "incompatible ledger entry",
                    entry if isinstance(entry, Mapping) else None,
                    issue,
                )
            )
        else:
            items.append(EvidenceItem(chapter, model_id, "ledger", "student ledger", entry))
    return EvidenceAudit(tuple(items))


def _ledger_issue(entry: Any, *, chapter: int, track_id: str, expected_model_id: str) -> str | None:
    if not isinstance(entry, Mapping):
        return "entry is not a mapping"
    if entry.get("schema_version") != 1:
        return "schema_version is not 1"
    if entry.get("lab_id") != f"v2_{chapter:02d}":
        return "lab_id does not match the required chapter"
    if entry.get("track_id") != track_id:
        return "track_id does not match the active track"
    if entry.get("model_id") != expected_model_id:
        return "model_id is not the recognized chapter backend"
    evidence = entry.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        return "evidence snapshots are missing"
    target_part, target_key = REPLAY_TARGETS[expected_model_id]
    snapshot = evidence.get(target_part)
    if not isinstance(snapshot, Mapping):
        return f"required evidence part {target_part!r} is missing"
    if snapshot.get("track") != track_id:
        return f"evidence snapshot {target_part!r} has a different track"
    if snapshot.get("model_key") != target_key:
        return f"evidence snapshot {target_part!r} names an unrecognized evaluator"
    records = [snapshot.get("baseline"), snapshot.get("result")]
    if snapshot.get("chosen_result") is not None:
        records.append(snapshot.get("chosen_result"))
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("inputs"), Mapping):
            return f"evidence snapshot {target_part!r} lacks exact evaluator inputs"
        try:
            _dispatch_replay(expected_model_id, target_key, dict(record["inputs"]))
        except Exception as exc:
            return f"evidence snapshot {target_part!r} cannot replay: {exc}"
    return None


def evidence_inputs(
    item: EvidenceItem,
    *,
    part: str,
    condition: Literal["baseline", "result"] = "result",
) -> dict[str, Any]:
    """Return a copy of exact evaluator inputs from compatible student evidence."""
    if item.status != "ledger" or item.entry is None:
        raise ValueError("only compatible student ledger evidence can be replayed")
    snapshot = item.entry["evidence"].get(part)
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"part {part!r} is absent from the ledger evidence")
    record = snapshot.get(condition)
    if condition == "result" and isinstance(snapshot.get("chosen_result"), Mapping):
        record = snapshot["chosen_result"]
    if not isinstance(record, Mapping) or not isinstance(record.get("inputs"), Mapping):
        raise ValueError(f"part {part!r} lacks {condition} evaluator inputs")
    return dict(record["inputs"])


def rehydrate_evidence_inputs(
    item: EvidenceItem,
    *,
    part: str,
    condition: Literal["baseline", "result"] = "result",
) -> dict[str, Any]:
    """Restore exact evaluator kwargs using the owning chapter's wire format."""
    inputs = evidence_inputs(item, part=part, condition=condition)
    snapshot = item.entry["evidence"][part]
    model_key = snapshot.get("model_key")
    if model_key != REPLAY_TARGETS.get(item.expected_model_id, (None, None))[1]:
        raise ValueError("evidence names an unrecognized evaluator")
    return _restore_inputs(item.expected_model_id, inputs)


def _restore_inputs(model_id: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    if model_id == "v2_01_experiments":
        restored = restore_v2_01_quantities(inputs)
    elif model_id == "v2_07_experiments":
        restored = inputs
    elif model_id == "v2_10_experiments":
        restored = _restore_v2_10_inputs(inputs)
    elif model_id == "v2_15_experiments":
        restored = deserialize_v2_15_value(inputs)
    else:  # guarded by audit, retained as a defensive boundary
        raise ValueError("unrecognized evidence model")
    if not isinstance(restored, Mapping):
        raise ValueError("restored evaluator inputs must be a mapping")
    return dict(restored)


def replay_evidence(
    item: EvidenceItem,
    *,
    part: str,
    condition: Literal["baseline", "result"] = "result",
):
    """Re-run one audited snapshot through its recognized owning evaluator."""
    target_part, _target_key = REPLAY_TARGETS[item.expected_model_id]
    if part != target_part:
        raise ValueError(f"capstone replay requires evidence part {target_part}")
    inputs = evidence_inputs(item, part=part, condition=condition)
    model_key = item.entry["evidence"][part]["model_key"]
    return _dispatch_replay(item.expected_model_id, model_key, inputs)


def _dispatch_replay(model_id: str, model_key: str, inputs: Mapping[str, Any]):
    if REPLAY_TARGETS.get(model_id, (None, None))[1] != model_key:
        raise ValueError("unrecognized model_id and evaluator pair")
    restored = _restore_inputs(model_id, inputs)
    evaluators = {
        "v2_01_experiments": evaluate_c3_scaling,
        "v2_07_experiments": checkpoint_tradeoff,
        "v2_10_experiments": replay_requests,
        "v2_15_experiments": evaluate_mitigation,
    }
    return evaluators[model_id](**restored)


def _restore_v2_10_inputs(value: Any):
    if isinstance(value, Mapping):
        if set(value) == {"magnitude", "unit"}:
            return deserialize_v2_10_quantity(dict(value))
        restored = {key: _restore_v2_10_inputs(item) for key, item in value.items()}
        if set(restored) == {"request_id", "arrival", "input_units", "output_units", "state_bytes"}:
            return Request(**restored)
        service_fields = {
            "fixed_time",
            "input_time_per_unit",
            "output_time_per_unit",
            "device_memory",
            "weight_memory",
            "reserved_memory",
            "fragmentation_fraction",
        }
        if set(restored) == service_fields:
            return ServiceModel(**restored)
        return restored
    if isinstance(value, list):
        return tuple(_restore_v2_10_inputs(item) for item in value)
    return value


def record_deployment_decision(
    *,
    decision: Decision,
    selected_repair: str,
    rejected_alternative: str,
    condition: str,
    residual_uncertainty: str,
) -> DeploymentRecord:
    """Preserve the learner's judgment without calculating board approval."""
    if decision not in ("release", "restrict", "defer"):
        raise ValueError("decision must be release, restrict, or defer")
    fields = {
        "selected_repair": selected_repair,
        "rejected_alternative": rejected_alternative,
        "condition": condition,
        "residual_uncertainty": residual_uncertainty,
    }
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        raise ValueError("deployment record fields must be nonempty")
    if selected_repair == rejected_alternative:
        raise ValueError("rejected alternative must differ from the selected repair")
    return DeploymentRecord(
        decision,
        selected_repair.strip(),
        rejected_alternative.strip(),
        condition.strip(),
        residual_uncertainty.strip(),
    )


__all__ = [
    "C3Configuration",
    "ConstraintCheck",
    "ControlPatch",
    "DeploymentRecord",
    "EnergyConfiguration",
    "EvidenceAudit",
    "EvidenceItem",
    "FleetConfiguration",
    "FleetEvaluation",
    "FleetLimits",
    "REQUIRED_MODELS",
    "RecoveryConfiguration",
    "REPLAY_TARGETS",
    "RepairComparison",
    "RepairPlan",
    "ServingConfiguration",
    "adverse_condition",
    "apply_patch",
    "audit_ledger_evidence",
    "compare_equal_budget_repairs",
    "evidence_inputs",
    "evaluation_record",
    "evaluate_configuration",
    "deserialize_configuration",
    "instructor_repair_plans",
    "instructor_scenario",
    "instructor_stress",
    "obligation_stress",
    "record_deployment_decision",
    "replay_evidence",
    "rehydrate_evidence_inputs",
    "serialize_configuration",
]
