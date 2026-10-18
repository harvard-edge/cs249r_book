"""Physical sustainability experiments for the Volume II fleet lab.

The models in this module keep four boundaries explicit:

* useful work is fixed as operations and moved bytes;
* facility power and heat rejection are feasibility constraints;
* carbon is calculated only after energy and feasibility are known; and
* replacement accounting includes only additional manufacturing caused by the
  replacement decision.  Embodied emissions of hardware already in service are
  sunk under that comparison boundary.

Track profiles are illustrative fleet scenarios, not measurements of branded
systems.  Water use is reported only when a caller supplies a water usage
effectiveness (WUE) quantity.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any, Iterable

from mlsysim.core.units import Q_, ureg
from mlsysim.infrastructure.registry import Infrastructure
from mlsysim.physics.quantities import carbon_from_energy


def _quantity(value: Any, unit: str, name: str, *, positive: bool = False):
    """Return ``value`` in ``unit`` and reject non-finite/negative inputs."""
    if not isinstance(value, ureg.Quantity):
        raise TypeError(f"{name} must be a Pint Quantity")
    try:
        normalized = value.to(unit)
    except Exception as exc:
        raise ValueError(f"{name} must be compatible with {unit}") from exc
    magnitude = normalized.magnitude
    if isinstance(magnitude, bool) or not math.isfinite(float(magnitude)):
        raise ValueError(f"{name} must be finite")
    if magnitude < 0 or (positive and magnitude == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def _scalar(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


def _count(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


@dataclass(frozen=True)
class EnergyBreakdown:
    """Energy for one fixed useful workload and one facility boundary."""

    operation_energy: Any
    movement_energy: Any
    other_it_energy: Any
    it_energy: Any
    facility_overhead_energy: Any
    facility_energy: Any
    pue: float


def calculate_energy(
    *,
    operations,
    moved_bytes,
    energy_per_operation,
    energy_per_byte,
    pue: float,
    other_it_energy=Q_(0, "joule"),
) -> EnergyBreakdown:
    """Decompose energy for fixed work into operations, movement, and overhead."""
    operations = _quantity(operations, "flop", "operations")
    moved_bytes = _quantity(moved_bytes, "byte", "moved_bytes")
    energy_per_operation = _quantity(energy_per_operation, "joule/flop", "energy_per_operation")
    energy_per_byte = _quantity(energy_per_byte, "joule/byte", "energy_per_byte")
    other_it_energy = _quantity(other_it_energy, "joule", "other_it_energy")
    pue = _scalar(pue, "pue", minimum=1.0)

    operation_energy = (operations * energy_per_operation).to("joule")
    movement_energy = (moved_bytes * energy_per_byte).to("joule")
    it_energy = (operation_energy + movement_energy + other_it_energy).to("joule")
    facility_energy = (it_energy * pue).to("joule")
    return EnergyBreakdown(
        operation_energy=operation_energy,
        movement_energy=movement_energy,
        other_it_energy=other_it_energy,
        it_energy=it_energy,
        facility_overhead_energy=(facility_energy - it_energy).to("joule"),
        facility_energy=facility_energy,
        pue=pue,
    )


@dataclass(frozen=True)
class FacilityFeasibility:
    """Independent electrical and heat-rejection checks for a fleet load."""

    it_power: Any
    facility_power: Any
    heat_load: Any
    electrical_capacity: Any
    cooling_capacity: Any
    electrical_headroom: Any
    cooling_headroom: Any
    electrical_feasible: bool
    cooling_feasible: bool
    feasible: bool


def check_facility(*, it_power, pue: float, electrical_capacity, cooling_capacity) -> FacilityFeasibility:
    """Check power delivery and cooling without using carbon intensity.

    ``cooling_capacity`` is the heat-removal capacity available to IT equipment.
    The electrical boundary sees IT power multiplied by PUE; the cooling
    boundary sees the IT heat load itself.
    """
    it_power = _quantity(it_power, "watt", "it_power")
    electrical_capacity = _quantity(electrical_capacity, "watt", "electrical_capacity", positive=True)
    cooling_capacity = _quantity(cooling_capacity, "watt", "cooling_capacity", positive=True)
    pue = _scalar(pue, "pue", minimum=1.0)
    facility_power = (it_power * pue).to("watt")
    heat_load = it_power.to("watt")
    electrical_headroom = (electrical_capacity - facility_power).to("watt")
    cooling_headroom = (cooling_capacity - heat_load).to("watt")
    electrical_feasible = electrical_headroom.magnitude >= 0
    cooling_feasible = cooling_headroom.magnitude >= 0
    return FacilityFeasibility(
        it_power=it_power,
        facility_power=facility_power,
        heat_load=heat_load,
        electrical_capacity=electrical_capacity,
        cooling_capacity=cooling_capacity,
        electrical_headroom=electrical_headroom,
        cooling_headroom=cooling_headroom,
        electrical_feasible=electrical_feasible,
        cooling_feasible=cooling_feasible,
        feasible=electrical_feasible and cooling_feasible,
    )


@dataclass(frozen=True)
class LifecycleImpact:
    """Training, inference, and incremental embodied emissions over a horizon."""

    horizon: Any
    inference_requests: int
    training_facility_energy: Any
    inference_facility_energy: Any
    total_facility_energy: Any
    operational_emissions: Any
    additional_manufacturing_emissions: Any
    total_emissions: Any
    inference_dominates_training: bool


def calculate_lifecycle(
    *,
    horizon,
    training_it_energy,
    inference_it_energy_per_request,
    requests_per_time,
    pue: float,
    carbon_intensity,
    additional_manufacturing_emissions=Q_(0, "kilogram"),
) -> LifecycleImpact:
    """Account lifecycle impact for an explicit, common operating horizon.

    ``additional_manufacturing_emissions`` is zero when continuing to use
    existing hardware.  For a replacement design it contains only emissions
    caused by manufacturing the replacement equipment.
    """
    horizon = _quantity(horizon, "second", "horizon", positive=True)
    training_it_energy = _quantity(training_it_energy, "joule", "training_it_energy")
    inference_it_energy_per_request = _quantity(
        inference_it_energy_per_request,
        "joule/count",
        "inference_it_energy_per_request",
    )
    requests_per_time = _quantity(requests_per_time, "count/second", "requests_per_time")
    carbon_intensity = _quantity(carbon_intensity, "kilogram/kilowatt_hour", "carbon_intensity")
    additional_manufacturing_emissions = _quantity(
        additional_manufacturing_emissions,
        "kilogram",
        "additional_manufacturing_emissions",
    )
    pue = _scalar(pue, "pue", minimum=1.0)

    request_quantity = (requests_per_time * horizon).to("count")
    request_count = int(round(float(request_quantity.magnitude)))
    inference_it_energy = (inference_it_energy_per_request * request_quantity).to("joule")
    training_facility_energy = (training_it_energy * pue).to("joule")
    inference_facility_energy = (inference_it_energy * pue).to("joule")
    total_facility_energy = (training_facility_energy + inference_facility_energy).to("kilowatt_hour")
    operational_emissions = carbon_from_energy(total_facility_energy, carbon_intensity).to("kilogram")
    total_emissions = (operational_emissions + additional_manufacturing_emissions).to("kilogram")
    return LifecycleImpact(
        horizon=horizon,
        inference_requests=request_count,
        training_facility_energy=training_facility_energy.to("kilowatt_hour"),
        inference_facility_energy=inference_facility_energy.to("kilowatt_hour"),
        total_facility_energy=total_facility_energy,
        operational_emissions=operational_emissions,
        additional_manufacturing_emissions=additional_manufacturing_emissions,
        total_emissions=total_emissions,
        inference_dominates_training=inference_facility_energy > training_facility_energy,
    )


@dataclass(frozen=True)
class ReplacementComparison:
    """Like-for-like lifecycle comparison against keeping installed hardware."""

    keep: LifecycleImpact
    replace: LifecycleImpact
    emissions_saved: Any
    replacement_break_even: bool
    operational_savings: Any
    additional_manufacturing_emissions: Any


def compare_replacement(
    *,
    horizon,
    training_it_energy,
    requests_per_time,
    current_inference_it_energy_per_request,
    replacement_inference_it_energy_per_request,
    pue: float,
    carbon_intensity,
    replacement_manufacturing_emissions,
) -> ReplacementComparison:
    """Compare keeping installed hardware with manufacturing a replacement.

    Both alternatives deliver the same request rate over the same horizon.
    Prior embodied emissions are excluded from both alternatives because they
    are sunk and therefore cannot change the replacement decision.
    """
    replacement_manufacturing_emissions = _quantity(
        replacement_manufacturing_emissions,
        "kilogram",
        "replacement_manufacturing_emissions",
    )
    common = dict(
        horizon=horizon,
        training_it_energy=training_it_energy,
        requests_per_time=requests_per_time,
        pue=pue,
        carbon_intensity=carbon_intensity,
    )
    keep = calculate_lifecycle(
        **common,
        inference_it_energy_per_request=current_inference_it_energy_per_request,
    )
    replace = calculate_lifecycle(
        **common,
        inference_it_energy_per_request=replacement_inference_it_energy_per_request,
        additional_manufacturing_emissions=replacement_manufacturing_emissions,
    )
    operational_savings = (keep.operational_emissions - replace.operational_emissions).to("kilogram")
    emissions_saved = (keep.total_emissions - replace.total_emissions).to("kilogram")
    return ReplacementComparison(
        keep=keep,
        replace=replace,
        emissions_saved=emissions_saved,
        replacement_break_even=emissions_saved.magnitude >= 0,
        operational_savings=operational_savings,
        additional_manufacturing_emissions=replacement_manufacturing_emissions,
    )


@dataclass(frozen=True)
class WorkloadPlacement:
    """A fixed job that may or may not be movable between fleet locations."""

    name: str
    it_energy: Any
    it_power: Any
    accelerator_demand: int
    deadline: Any
    movable: bool
    home_site: str
    allowed_sites: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlacementSite:
    """Capacity, execution time, and accounting inputs for one location."""

    name: str
    available_accelerators: int
    execution_time: Any
    pue: float
    carbon_intensity: Any
    electrical_capacity: Any
    cooling_capacity: Any
    wue: Any | None = None


@dataclass(frozen=True)
class PlacementEvaluation:
    site: str
    feasible: bool
    reasons: tuple[str, ...]
    facility: FacilityFeasibility
    facility_energy: Any
    operational_emissions: Any
    water_use: Any | None


def evaluate_placements(workload: WorkloadPlacement, sites: Iterable[PlacementSite]) -> tuple[PlacementEvaluation, ...]:
    """Evaluate placement constraints and impacts without silently migrating work."""
    it_energy = _quantity(workload.it_energy, "joule", "workload.it_energy")
    it_power = _quantity(workload.it_power, "watt", "workload.it_power")
    deadline = _quantity(workload.deadline, "second", "workload.deadline", positive=True)
    demand = _count(workload.accelerator_demand, "workload.accelerator_demand", positive=True)
    allowed = set(workload.allowed_sites)
    rows = []
    for site in sites:
        execution_time = _quantity(site.execution_time, "second", f"{site.name}.execution_time", positive=True)
        available = _count(site.available_accelerators, f"{site.name}.available_accelerators")
        carbon_intensity = _quantity(
            site.carbon_intensity,
            "kilogram/kilowatt_hour",
            f"{site.name}.carbon_intensity",
        )
        facility = check_facility(
            it_power=it_power,
            pue=site.pue,
            electrical_capacity=site.electrical_capacity,
            cooling_capacity=site.cooling_capacity,
        )
        reasons: list[str] = []
        if not workload.movable and site.name != workload.home_site:
            reasons.append("immutable workload cannot leave its home site")
        if allowed and site.name not in allowed:
            reasons.append("site violates locality policy")
        if available < demand:
            reasons.append("insufficient accelerator capacity")
        if execution_time > deadline:
            reasons.append("execution misses deadline")
        if not facility.electrical_feasible:
            reasons.append("electrical capacity exceeded")
        if not facility.cooling_feasible:
            reasons.append("cooling capacity exceeded")

        facility_energy = (it_energy * site.pue).to("kilowatt_hour")
        emissions = carbon_from_energy(facility_energy, carbon_intensity).to("kilogram")
        water_use = None
        if site.wue is not None:
            wue = _quantity(site.wue, "liter/kilowatt_hour", f"{site.name}.wue")
            water_use = (facility_energy * wue).to("liter")
        rows.append(
            PlacementEvaluation(
                site=site.name,
                feasible=not reasons,
                reasons=tuple(reasons),
                facility=facility,
                facility_energy=facility_energy,
                operational_emissions=emissions,
                water_use=water_use,
            )
        )
    return tuple(rows)


def select_lowest_carbon_placement(
    workload: WorkloadPlacement, sites: Iterable[PlacementSite]
) -> PlacementEvaluation | None:
    """Return the lowest-emission feasible site, or ``None`` when none qualify."""
    feasible = [row for row in evaluate_placements(workload, sites) if row.feasible]
    if not feasible:
        return None
    return min(feasible, key=lambda row: row.operational_emissions.magnitude)


@dataclass(frozen=True)
class MitigationEvaluation:
    energy: EnergyBreakdown
    latency: Any
    quality: float
    meets_latency: bool
    meets_quality: bool
    acceptable: bool


def evaluate_mitigation(
    *,
    operations,
    moved_bytes,
    energy_per_operation,
    energy_per_byte,
    pue: float,
    latency,
    latency_deadline,
    quality: float,
    minimum_quality: float,
    other_it_energy=Q_(0, "joule"),
) -> MitigationEvaluation:
    """Evaluate energy while treating supplied quality and latency as constraints."""
    energy = calculate_energy(
        operations=operations,
        moved_bytes=moved_bytes,
        energy_per_operation=energy_per_operation,
        energy_per_byte=energy_per_byte,
        pue=pue,
        other_it_energy=other_it_energy,
    )
    latency = _quantity(latency, "second", "latency")
    deadline = _quantity(latency_deadline, "second", "latency_deadline", positive=True)
    quality = _scalar(quality, "quality")
    minimum_quality = _scalar(minimum_quality, "minimum_quality")
    meets_latency = latency <= deadline
    meets_quality = quality >= minimum_quality
    return MitigationEvaluation(
        energy=energy,
        latency=latency,
        quality=quality,
        meets_latency=meets_latency,
        meets_quality=meets_quality,
        acceptable=meets_latency and meets_quality,
    )


@dataclass(frozen=True)
class ReboundImpact:
    horizon: Any
    baseline_requests: int
    optimized_requests: int
    demand_multiplier: float
    energy_reduction_per_request: float
    break_even_demand_multiplier: float
    baseline_emissions: Any
    optimized_emissions: Any
    emissions_change: Any
    rebound_increases_emissions: bool


def calculate_rebound(
    *,
    horizon,
    baseline_requests_per_time,
    baseline_it_energy_per_request,
    optimized_it_energy_per_request,
    demand_multiplier: float,
    pue: float,
    carbon_intensity,
) -> ReboundImpact:
    """Calculate total emissions after an explicit efficiency-driven demand change."""
    horizon = _quantity(horizon, "second", "horizon", positive=True)
    rate = _quantity(
        baseline_requests_per_time,
        "count/second",
        "baseline_requests_per_time",
    )
    baseline_energy = _quantity(
        baseline_it_energy_per_request,
        "joule/count",
        "baseline_it_energy_per_request",
        positive=True,
    )
    optimized_energy = _quantity(
        optimized_it_energy_per_request,
        "joule/count",
        "optimized_it_energy_per_request",
        positive=True,
    )
    demand_multiplier = _scalar(demand_multiplier, "demand_multiplier")
    pue = _scalar(pue, "pue", minimum=1.0)
    carbon_intensity = _quantity(carbon_intensity, "kilogram/kilowatt_hour", "carbon_intensity")

    baseline_request_quantity = (rate * horizon).to("count")
    optimized_request_quantity = baseline_request_quantity * demand_multiplier
    baseline_facility_energy = (baseline_energy * baseline_request_quantity * pue).to("kilowatt_hour")
    optimized_facility_energy = (optimized_energy * optimized_request_quantity * pue).to("kilowatt_hour")
    baseline_emissions = carbon_from_energy(baseline_facility_energy, carbon_intensity).to("kilogram")
    optimized_emissions = carbon_from_energy(optimized_facility_energy, carbon_intensity).to("kilogram")
    reduction = 1.0 - (optimized_energy / baseline_energy).to("").magnitude
    break_even = (baseline_energy / optimized_energy).to("").magnitude
    change = (optimized_emissions - baseline_emissions).to("kilogram")
    return ReboundImpact(
        horizon=horizon,
        baseline_requests=int(round(float(baseline_request_quantity.magnitude))),
        optimized_requests=int(round(float(optimized_request_quantity.magnitude))),
        demand_multiplier=demand_multiplier,
        energy_reduction_per_request=float(reduction),
        break_even_demand_multiplier=float(break_even),
        baseline_emissions=baseline_emissions,
        optimized_emissions=optimized_emissions,
        emissions_change=change,
        rebound_increases_emissions=change.magnitude > 0,
    )


# Illustrative fleet assumptions. TinyML and mobile inference remains on the
# deployed device fleet; their backend work may be scheduled separately.
TRACK_PROFILES: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML",
        "fleet_shape": "sensor fleet with a regional training backend",
        "fleet_size": 100_000,
        "inference_movable": False,
        "home_site": "device fleet",
    },
    "mobile": {
        "display": "Mobile",
        "fleet_shape": "phone fleet with regional service backends",
        "fleet_size": 50_000,
        "inference_movable": False,
        "home_site": "phone fleet",
    },
    "edge": {
        "display": "Edge",
        "fleet_shape": "site gateway fleet with locality-bound inference",
        "fleet_size": 2_000,
        "inference_movable": False,
        "home_site": "site gateways",
    },
    "cloud": {
        "display": "Cloud",
        "fleet_shape": "regional accelerator pools",
        "fleet_size": 512,
        "inference_movable": True,
        "home_site": "primary region",
    },
}


def track_profile(track_id: str) -> dict[str, Any]:
    """Return a copy of one illustrative fleet profile."""
    if track_id not in TRACK_PROFILES:
        raise ValueError(f"track_id must be one of {', '.join(TRACK_PROFILES)}")
    return dict(TRACK_PROFILES[track_id])


# Complete, explicitly illustrative evaluator arguments for the four lab tracks.
# These are scenario assumptions rather than branded hardware measurements.  The
# unit of useful work is a fleet service interval; device tracks therefore
# aggregate the work of many independent deployed devices without pretending
# that their memory or compute pools into one machine.
TRACK_SCENARIOS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "energy_args": {
            "operations": Q_(80, "teraflop"),
            "moved_bytes": Q_(1.2, "terabyte"),
            "energy_per_operation": Q_(8, "picojoule/flop"),
            "energy_per_byte": Q_(45, "picojoule/byte"),
            "other_it_energy": Q_(0.08, "kilowatt_hour"),
            "pue": 1.15,
        },
        "facility_args": {
            "it_power": Q_(48, "kilowatt"),
            "pue": 1.15,
            "electrical_capacity": Q_(60, "kilowatt"),
            "cooling_capacity": Q_(52, "kilowatt"),
        },
        "lifecycle_args": {
            "horizon": Q_(2, "year"),
            "training_it_energy": Q_(12, "kilowatt_hour"),
            "inference_it_energy_per_request": Q_(5, "millijoule/count"),
            "requests_per_time": Q_(100_000, "count/second"),
            "pue": 1.0,
            "carbon_intensity": Q_(0.32, "kilogram/kilowatt_hour"),
        },
        "replacement_args": {
            "horizon": Q_(2, "year"),
            "training_it_energy": Q_(12, "kilowatt_hour"),
            "requests_per_time": Q_(100_000, "count/second"),
            "current_inference_it_energy_per_request": Q_(5, "millijoule/count"),
            "replacement_inference_it_energy_per_request": Q_(3, "millijoule/count"),
            "pue": 1.0,
            "carbon_intensity": Q_(0.32, "kilogram/kilowatt_hour"),
            "replacement_manufacturing_emissions": Q_(40_000, "kilogram"),
        },
        "placement_workload": WorkloadPlacement(
            name="deployed sensor inference",
            it_energy=Q_(10, "kilowatt_hour"),
            it_power=Q_(20, "kilowatt"),
            accelerator_demand=1,
            deadline=Q_(20, "millisecond"),
            movable=False,
            home_site="device fleet",
            allowed_sites=("device fleet",),
        ),
        "rebound_args": {
            "horizon": Q_(2, "year"),
            "baseline_requests_per_time": Q_(100_000, "count/second"),
            "baseline_it_energy_per_request": Q_(5, "millijoule/count"),
            "optimized_it_energy_per_request": Q_(3, "millijoule/count"),
            "demand_multiplier": 1.0,
            "pue": 1.0,
            "carbon_intensity": Q_(0.32, "kilogram/kilowatt_hour"),
        },
    },
    "mobile": {
        "energy_args": {
            "operations": Q_(600, "teraflop"),
            "moved_bytes": Q_(8, "terabyte"),
            "energy_per_operation": Q_(12, "picojoule/flop"),
            "energy_per_byte": Q_(60, "picojoule/byte"),
            "other_it_energy": Q_(0.4, "kilowatt_hour"),
            "pue": 1.12,
        },
        "facility_args": {
            "it_power": Q_(180, "kilowatt"),
            "pue": 1.12,
            "electrical_capacity": Q_(220, "kilowatt"),
            "cooling_capacity": Q_(190, "kilowatt"),
        },
        "lifecycle_args": {
            "horizon": Q_(2, "year"),
            "training_it_energy": Q_(90, "kilowatt_hour"),
            "inference_it_energy_per_request": Q_(0.5, "joule/count"),
            "requests_per_time": Q_(2_500, "count/second"),
            "pue": 1.0,
            "carbon_intensity": Q_(0.28, "kilogram/kilowatt_hour"),
        },
        "replacement_args": {
            "horizon": Q_(2, "year"),
            "training_it_energy": Q_(90, "kilowatt_hour"),
            "requests_per_time": Q_(2_500, "count/second"),
            "current_inference_it_energy_per_request": Q_(0.5, "joule/count"),
            "replacement_inference_it_energy_per_request": Q_(0.3, "joule/count"),
            "pue": 1.0,
            "carbon_intensity": Q_(0.28, "kilogram/kilowatt_hour"),
            "replacement_manufacturing_emissions": Q_(22_000, "kilogram"),
        },
        "placement_workload": WorkloadPlacement(
            name="on-device interactive inference",
            it_energy=Q_(20, "kilowatt_hour"),
            it_power=Q_(80, "kilowatt"),
            accelerator_demand=1,
            deadline=Q_(50, "millisecond"),
            movable=False,
            home_site="phone fleet",
            allowed_sites=("phone fleet",),
        ),
        "rebound_args": {
            "horizon": Q_(2, "year"),
            "baseline_requests_per_time": Q_(2_500, "count/second"),
            "baseline_it_energy_per_request": Q_(0.5, "joule/count"),
            "optimized_it_energy_per_request": Q_(0.3, "joule/count"),
            "demand_multiplier": 1.0,
            "pue": 1.0,
            "carbon_intensity": Q_(0.28, "kilogram/kilowatt_hour"),
        },
    },
    "edge": {
        "energy_args": {
            "operations": Q_(5, "petaflop"),
            "moved_bytes": Q_(60, "terabyte"),
            "energy_per_operation": Q_(18, "picojoule/flop"),
            "energy_per_byte": Q_(80, "picojoule/byte"),
            "other_it_energy": Q_(2, "kilowatt_hour"),
            "pue": 1.18,
        },
        "facility_args": {
            "it_power": Q_(900, "kilowatt"),
            "pue": 1.18,
            "electrical_capacity": Q_(1.1, "megawatt"),
            "cooling_capacity": Q_(950, "kilowatt"),
        },
        "lifecycle_args": {
            "horizon": Q_(2, "year"),
            "training_it_energy": Q_(800, "kilowatt_hour"),
            "inference_it_energy_per_request": Q_(4, "joule/count"),
            "requests_per_time": Q_(10_000, "count/second"),
            "pue": 1.12,
            "carbon_intensity": Q_(0.36, "kilogram/kilowatt_hour"),
        },
        "replacement_args": {
            "horizon": Q_(2, "year"),
            "training_it_energy": Q_(800, "kilowatt_hour"),
            "requests_per_time": Q_(10_000, "count/second"),
            "current_inference_it_energy_per_request": Q_(4, "joule/count"),
            "replacement_inference_it_energy_per_request": Q_(2.4, "joule/count"),
            "pue": 1.12,
            "carbon_intensity": Q_(0.36, "kilogram/kilowatt_hour"),
            "replacement_manufacturing_emissions": Q_(180_000, "kilogram"),
        },
        "placement_workload": WorkloadPlacement(
            name="local video inference",
            it_energy=Q_(120, "kilowatt_hour"),
            it_power=Q_(600, "kilowatt"),
            accelerator_demand=64,
            deadline=Q_(25, "millisecond"),
            movable=False,
            home_site="site gateways",
            allowed_sites=("site gateways",),
        ),
        "rebound_args": {
            "horizon": Q_(2, "year"),
            "baseline_requests_per_time": Q_(10_000, "count/second"),
            "baseline_it_energy_per_request": Q_(4, "joule/count"),
            "optimized_it_energy_per_request": Q_(2.4, "joule/count"),
            "demand_multiplier": 1.0,
            "pue": 1.12,
            "carbon_intensity": Q_(0.36, "kilogram/kilowatt_hour"),
        },
    },
    "cloud": {
        "energy_args": {
            "operations": Q_(80, "petaflop"),
            "moved_bytes": Q_(900, "terabyte"),
            "energy_per_operation": Q_(22, "picojoule/flop"),
            "energy_per_byte": Q_(95, "picojoule/byte"),
            "other_it_energy": Q_(18, "kilowatt_hour"),
            "pue": 1.12,
        },
        "facility_args": {
            "it_power": Q_(5.2, "megawatt"),
            "pue": 1.12,
            "electrical_capacity": Q_(6, "megawatt"),
            "cooling_capacity": Q_(5.5, "megawatt"),
        },
        "lifecycle_args": {
            "horizon": Q_(2, "year"),
            "training_it_energy": Q_(500, "megawatt_hour"),
            "inference_it_energy_per_request": Q_(80, "joule/count"),
            "requests_per_time": Q_(100, "count/second"),
            "pue": 1.12,
            "carbon_intensity": Q_(0.4, "kilogram/kilowatt_hour"),
        },
        "replacement_args": {
            "horizon": Q_(2, "year"),
            "training_it_energy": Q_(500, "megawatt_hour"),
            "requests_per_time": Q_(100, "count/second"),
            "current_inference_it_energy_per_request": Q_(80, "joule/count"),
            "replacement_inference_it_energy_per_request": Q_(48, "joule/count"),
            "pue": 1.12,
            "carbon_intensity": Q_(0.4, "kilogram/kilowatt_hour"),
            "replacement_manufacturing_emissions": Q_(50_000, "kilogram"),
        },
        "placement_workload": WorkloadPlacement(
            name="movable batch training",
            it_energy=Q_(500, "megawatt_hour"),
            it_power=Q_(4, "megawatt"),
            accelerator_demand=512,
            deadline=Q_(7, "day"),
            movable=True,
            home_site="primary region",
            allowed_sites=("primary region", "clean region"),
        ),
        "rebound_args": {
            "horizon": Q_(2, "year"),
            "baseline_requests_per_time": Q_(100, "count/second"),
            "baseline_it_energy_per_request": Q_(80, "joule/count"),
            "optimized_it_energy_per_request": Q_(48, "joule/count"),
            "demand_multiplier": 1.0,
            "pue": 1.12,
            "carbon_intensity": Q_(0.4, "kilogram/kilowatt_hour"),
        },
    },
}


_AIR_COOLED_WUE = Q_(Infrastructure.FacilityCooling.BestAir.wue, "liter/kilowatt_hour")
_LIQUID_COOLED_WUE = Q_(Infrastructure.FacilityCooling.LiquidCooled.wue, "liter/kilowatt_hour")


PLACEMENT_SITES: dict[str, tuple[PlacementSite, ...]] = {
    "tinyml": (
        PlacementSite(
            "device fleet",
            1,
            Q_(8, "millisecond"),
            1.0,
            Q_(0.32, "kilogram/kilowatt_hour"),
            Q_(25, "kilowatt"),
            Q_(22, "kilowatt"),
            None,
        ),
        PlacementSite(
            "clean region",
            64,
            Q_(80, "millisecond"),
            1.08,
            Q_(0.04, "kilogram/kilowatt_hour"),
            Q_(1, "megawatt"),
            Q_(900, "kilowatt"),
            _LIQUID_COOLED_WUE,
        ),
    ),
    "mobile": (
        PlacementSite(
            "phone fleet",
            1,
            Q_(35, "millisecond"),
            1.0,
            Q_(0.28, "kilogram/kilowatt_hour"),
            Q_(100, "kilowatt"),
            Q_(90, "kilowatt"),
            None,
        ),
        PlacementSite(
            "clean region",
            128,
            Q_(95, "millisecond"),
            1.08,
            Q_(0.04, "kilogram/kilowatt_hour"),
            Q_(2, "megawatt"),
            Q_(1.8, "megawatt"),
            _LIQUID_COOLED_WUE,
        ),
    ),
    "edge": (
        PlacementSite(
            "site gateways",
            96,
            Q_(18, "millisecond"),
            1.12,
            Q_(0.36, "kilogram/kilowatt_hour"),
            Q_(800, "kilowatt"),
            Q_(700, "kilowatt"),
            _AIR_COOLED_WUE,
        ),
        PlacementSite(
            "clean region",
            512,
            Q_(70, "millisecond"),
            1.08,
            Q_(0.04, "kilogram/kilowatt_hour"),
            Q_(8, "megawatt"),
            Q_(7, "megawatt"),
            _LIQUID_COOLED_WUE,
        ),
    ),
    "cloud": (
        PlacementSite(
            "primary region",
            640,
            Q_(5, "day"),
            1.12,
            Q_(0.4, "kilogram/kilowatt_hour"),
            Q_(6, "megawatt"),
            Q_(5, "megawatt"),
            _AIR_COOLED_WUE,
        ),
        PlacementSite(
            "clean region",
            560,
            Q_(6, "day"),
            1.08,
            Q_(0.04, "kilogram/kilowatt_hour"),
            Q_(5, "megawatt"),
            Q_(4.5, "megawatt"),
            _LIQUID_COOLED_WUE,
        ),
    ),
}


_MITIGATION_TIMING_AND_QUALITY = {
    "tinyml": (Q_(14, "millisecond"), Q_(20, "millisecond"), 0.91, 0.90),
    "mobile": (Q_(42, "millisecond"), Q_(50, "millisecond"), 0.92, 0.90),
    "edge": (Q_(21, "millisecond"), Q_(25, "millisecond"), 0.94, 0.92),
    "cloud": (Q_(180, "millisecond"), Q_(200, "millisecond"), 0.95, 0.93),
}


def track_scenario(track_id: str) -> dict[str, Any]:
    """Return independent evaluator defaults for one teaching track."""
    if track_id not in TRACK_SCENARIOS:
        raise ValueError(f"track_id must be one of {', '.join(TRACK_SCENARIOS)}")
    scenario = copy.deepcopy(TRACK_SCENARIOS[track_id])
    scenario["placement_sites"] = copy.deepcopy(PLACEMENT_SITES[track_id])
    latency, deadline, quality, minimum_quality = _MITIGATION_TIMING_AND_QUALITY[track_id]
    scenario["mitigation_args"] = {
        **copy.deepcopy(scenario["energy_args"]),
        "latency": copy.deepcopy(latency),
        "latency_deadline": copy.deepcopy(deadline),
        "quality": quality,
        "minimum_quality": minimum_quality,
    }
    return scenario


def facility_experiment_args(track_id: str, *, load_scale: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return baseline and scaled-load kwargs for ``check_facility``."""
    load_scale = _scalar(load_scale, "load_scale", minimum=0.01)
    baseline = track_scenario(track_id)["facility_args"]
    changed = copy.deepcopy(baseline)
    changed["it_power"] = baseline["it_power"] * load_scale
    return baseline, changed


def lifecycle_experiment_args(track_id: str, *, horizon_years: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return keep and replacement lifecycle kwargs for a common horizon."""
    horizon_years = _scalar(horizon_years, "horizon_years", minimum=0.01)
    scenario = track_scenario(track_id)
    replacement = scenario["replacement_args"]
    common = {
        "horizon": Q_(horizon_years, "year"),
        "training_it_energy": replacement["training_it_energy"],
        "requests_per_time": replacement["requests_per_time"],
        "pue": replacement["pue"],
        "carbon_intensity": replacement["carbon_intensity"],
    }
    keep = {
        **common,
        "inference_it_energy_per_request": replacement["current_inference_it_energy_per_request"],
        "additional_manufacturing_emissions": Q_(0, "kilogram"),
    }
    replace = {
        **common,
        "inference_it_energy_per_request": replacement["replacement_inference_it_energy_per_request"],
        "additional_manufacturing_emissions": replacement["replacement_manufacturing_emissions"],
    }
    return keep, replace


def lower_lifecycle_option(keep: LifecycleImpact, replace: LifecycleImpact) -> str:
    """Return the lower-emission lifecycle option for a supplied pair."""
    if not isinstance(keep, LifecycleImpact) or not isinstance(replace, LifecycleImpact):
        raise TypeError("keep and replace must be LifecycleImpact results")
    return "replace" if replace.total_emissions < keep.total_emissions else "keep"


def placement_experiment_args(
    track_id: str, *, deadline_scale: float, clean_capacity_scale: float = 1.0
) -> dict[str, Any]:
    """Return exact placement evaluator arguments after two causal controls."""
    deadline_scale = _scalar(deadline_scale, "deadline_scale", minimum=0.01)
    clean_capacity_scale = _scalar(clean_capacity_scale, "clean_capacity_scale", minimum=0.0)
    scenario = track_scenario(track_id)
    workload = scenario["placement_workload"]
    workload = WorkloadPlacement(**{**asdict(workload), "deadline": workload.deadline * deadline_scale})
    sites = list(scenario["placement_sites"])
    clean = sites[1]
    sites[1] = PlacementSite(
        **{
            **asdict(clean),
            "available_accelerators": int(math.floor(clean.available_accelerators * clean_capacity_scale)),
        }
    )
    return {"workload": workload, "sites": tuple(sites)}


def mitigation_experiment_args(track_id: str, *, intervention: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return common-baseline kwargs for one supplied mitigation fixture."""
    factors = {
        "compute": (0.65, 1.0, 0.80, 0.0),
        "movement": (1.0, 0.50, 0.90, 0.0),
        "balanced": (0.80, 0.70, 0.85, -0.01),
    }
    if intervention not in factors:
        raise ValueError(f"intervention must be one of {', '.join(factors)}")
    baseline = track_scenario(track_id)["mitigation_args"]
    operation_factor, movement_factor, latency_factor, quality_delta = factors[intervention]
    changed = copy.deepcopy(baseline)
    changed["operations"] = baseline["operations"] * operation_factor
    changed["moved_bytes"] = baseline["moved_bytes"] * movement_factor
    changed["latency"] = baseline["latency"] * latency_factor
    changed["quality"] = baseline["quality"] + quality_delta
    return baseline, changed


def rebound_experiment_args(track_id: str, *, demand_multiplier: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return baseline and explicit-demand kwargs for ``calculate_rebound``."""
    demand_multiplier = _scalar(demand_multiplier, "demand_multiplier", minimum=0.0)
    baseline = track_scenario(track_id)["rebound_args"]
    baseline["demand_multiplier"] = 1.0
    changed = copy.deepcopy(baseline)
    changed["demand_multiplier"] = demand_multiplier
    return baseline, changed


_SERIALIZABLE_DATACLASSES = {
    "EnergyBreakdown": EnergyBreakdown,
    "FacilityFeasibility": FacilityFeasibility,
    "LifecycleImpact": LifecycleImpact,
    "ReplacementComparison": ReplacementComparison,
    "WorkloadPlacement": WorkloadPlacement,
    "PlacementSite": PlacementSite,
    "PlacementEvaluation": PlacementEvaluation,
    "MitigationEvaluation": MitigationEvaluation,
    "ReboundImpact": ReboundImpact,
}


def serialize_experiment_value(value: Any) -> Any:
    """Convert evaluator arguments to immutable JSON-compatible values.

    Pint quantities retain both magnitude and unit.  The representation is
    intentionally explicit so a capstone can replay calculations without
    guessing units from field names.
    """
    if isinstance(value, ureg.Quantity):
        magnitude = value.magnitude
        if hasattr(magnitude, "tolist"):
            magnitude = magnitude.tolist()
        return {
            "__type__": "quantity",
            "magnitude": magnitude,
            "unit": str(value.units),
        }
    if is_dataclass(value):
        return {
            "__type__": "dataclass",
            "class": type(value).__name__,
            "fields": serialize_experiment_value(asdict(value)),
        }
    if isinstance(value, dict):
        return {str(key): serialize_experiment_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {
            "__type__": "tuple",
            "items": [serialize_experiment_value(item) for item in value],
        }
    if isinstance(value, list):
        return [serialize_experiment_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize experiment value of type {type(value).__name__}")


def deserialize_experiment_value(value: Any) -> Any:
    """Restore quantities and supported evaluator dataclasses from a snapshot."""
    if isinstance(value, list):
        return [deserialize_experiment_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    value_type = value.get("__type__")
    if value_type == "quantity":
        return Q_(value["magnitude"], value["unit"])
    if value_type == "tuple":
        return tuple(deserialize_experiment_value(item) for item in value["items"])
    if value_type == "dataclass":
        class_name = value.get("class")
        if class_name not in _SERIALIZABLE_DATACLASSES:
            raise ValueError(f"unsupported experiment dataclass: {class_name}")
        fields = deserialize_experiment_value(value["fields"])
        return _SERIALIZABLE_DATACLASSES[class_name](**fields)
    return {key: deserialize_experiment_value(item) for key, item in value.items()}


def serialize_evaluator_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Serialize the exact keyword arguments recorded with an experiment result."""
    if not isinstance(arguments, dict):
        raise TypeError("arguments must be a dictionary of evaluator keyword arguments")
    return serialize_experiment_value(copy.deepcopy(arguments))
