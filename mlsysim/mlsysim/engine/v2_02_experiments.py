"""Chapter 2 compute-infrastructure experiments.

This module composes registry hardware with explicit teaching-scenario inputs.
It deliberately keeps five mechanisms separate: Roofline throughput, memory
capacity and bandwidth, transfer tiers, facility limits, and lifetime cost.
The analytical results are estimates, not measurements.

TinyML and mobile tracks describe the development hosts that train models for
their deployment fleets.  They do not pool memory or compute across end-user
devices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pint

from mlsysim import Hardware, Infrastructure, Systems
from mlsysim.core.units import Q_, ureg
from mlsysim.physics import transfer_time


Quantity = pint.Quantity
MODEL_ID = "v2_02_experiments"

_SCENARIO_NOTE = (
    "Workload sizes, overheads, capacities, duty cycles, and efficiency are "
    "illustrative scenario assumptions; accelerator and tier specifications are "
    "registry-backed."
)


@dataclass(frozen=True)
class TrackConfig:
    """Bounded scenario assumptions for one teaching track."""

    track_id: str
    label: str
    infrastructure_role: str
    roofline_devices: tuple[str, str]
    memory_device: str
    procurement_devices: tuple[str, str]
    working_set: Quantity
    bytes_moved: Quantity
    operations: Quantity
    transfer_payload: Quantity
    accelerator_count: int
    accelerators_per_node: int
    nodes_per_rack: int
    node_overhead: Quantity
    rack_overhead: Quantity
    electrical_capacity: Quantity
    cooling_capacity: Quantity
    network_tier_name: str
    network_bandwidth: Quantity


_TRACKS = {
    "tinyml": TrackConfig(
        track_id="tinyml",
        label="TinyML",
        infrastructure_role="development host training models for a wearable fleet",
        roofline_devices=("workstation:MacBookM3Max", "workstation:DGX_Spark"),
        memory_device="workstation:DGX_Spark",
        procurement_devices=("cloud:T4", "workstation:DGX_Spark"),
        working_set=Q_("12 GiB"),
        bytes_moved=Q_("48 GiB"),
        operations=Q_("3.84 TFLOP"),
        transfer_payload=Q_("4 GiB"),
        accelerator_count=1,
        accelerators_per_node=1,
        nodes_per_rack=4,
        node_overhead=Q_("120 W"),
        rack_overhead=Q_("250 W"),
        electrical_capacity=Q_("2 kW"),
        cooling_capacity=Q_("1.5 kW"),
        network_tier_name="configured host network (10G Ethernet)",
        network_bandwidth=Systems.Fabrics.Ethernet_10G.bandwidth,
    ),
    "mobile": TrackConfig(
        track_id="mobile",
        label="Mobile",
        infrastructure_role="development host training models for a phone fleet",
        roofline_devices=("workstation:MacBookM3Max", "workstation:DGX_Spark"),
        memory_device="workstation:DGX_Spark",
        procurement_devices=("cloud:T4", "workstation:DGX_Spark"),
        working_set=Q_("40 GiB"),
        bytes_moved=Q_("160 GiB"),
        operations=Q_("24 TFLOP"),
        transfer_payload=Q_("16 GiB"),
        accelerator_count=2,
        accelerators_per_node=1,
        nodes_per_rack=4,
        node_overhead=Q_("150 W"),
        rack_overhead=Q_("300 W"),
        electrical_capacity=Q_("3 kW"),
        cooling_capacity=Q_("2.5 kW"),
        network_tier_name="configured backend network (100G Ethernet)",
        network_bandwidth=Systems.Fabrics.Ethernet_100G.bandwidth,
    ),
    "edge": TrackConfig(
        track_id="edge",
        label="Edge",
        infrastructure_role="regional training pool for vehicle and gateway workloads",
        roofline_devices=("cloud:Gaudi3", "cloud:MI300X"),
        memory_device="cloud:MI300X",
        procurement_devices=("cloud:A100", "cloud:MI300X"),
        working_set=Q_("100 GiB"),
        bytes_moved=Q_("400 GiB"),
        operations=Q_("120 TFLOP"),
        transfer_payload=Q_("64 GiB"),
        accelerator_count=8,
        accelerators_per_node=8,
        nodes_per_rack=4,
        node_overhead=Q_("1.8 kW"),
        rack_overhead=Q_("2.5 kW"),
        electrical_capacity=Q_("16 kW"),
        cooling_capacity=Q_("14 kW"),
        network_tier_name="configured backend network (100G Ethernet)",
        network_bandwidth=Systems.Fabrics.Ethernet_100G.bandwidth,
    ),
    "cloud": TrackConfig(
        track_id="cloud",
        label="Cloud",
        infrastructure_role="coupled accelerator training fleet",
        roofline_devices=("cloud:Gaudi3", "cloud:MI300X"),
        memory_device="cloud:B200",
        procurement_devices=("cloud:H100", "cloud:B200"),
        working_set=Q_("150 GiB"),
        bytes_moved=Q_("600 GiB"),
        operations=Q_("600 TFLOP"),
        transfer_payload=Q_("160 GiB"),
        accelerator_count=8,
        accelerators_per_node=8,
        nodes_per_rack=4,
        node_overhead=Q_("2.4 kW"),
        rack_overhead=Q_("3.0 kW"),
        electrical_capacity=Q_("24 kW"),
        cooling_capacity=Q_("21 kW"),
        network_tier_name="configured cluster fabric (InfiniBand NDR)",
        network_bandwidth=Systems.Fabrics.InfiniBand_NDR.bandwidth,
    ),
}

_ALIASES = {
    "oura_ring": "tinyml",
    "iphone": "mobile",
    "robotaxi": "edge",
    "cloud_fleet": "cloud",
}


def serialize_quantity(value: Quantity) -> dict[str, float | str]:
    """Serialize a scalar Pint quantity without dropping its unit."""

    if not isinstance(value, ureg.Quantity):
        raise TypeError("value must be a Pint Quantity")
    magnitude = float(value.magnitude)
    if not math.isfinite(magnitude):
        raise ValueError("quantity magnitude must be finite")
    return {"magnitude": magnitude, "unit": str(value.units)}


def deserialize_quantity(payload: dict[str, Any]) -> Quantity:
    """Rebuild a scalar Pint quantity produced by :func:`serialize_quantity`."""

    if set(payload) != {"magnitude", "unit"}:
        raise ValueError("quantity payload must contain only magnitude and unit")
    try:
        value = Q_(float(payload["magnitude"]), str(payload["unit"]))
    except (TypeError, ValueError, pint.UndefinedUnitError) as exc:
        raise ValueError("invalid serialized quantity") from exc
    if not math.isfinite(float(value.magnitude)):
        raise ValueError("quantity magnitude must be finite")
    return value


def _positive_quantity(value: Quantity, unit: str, name: str) -> Quantity:
    if not isinstance(value, ureg.Quantity):
        raise TypeError(f"{name} must be a Pint Quantity")
    try:
        normalized = value.to(unit)
    except pint.DimensionalityError as exc:
        raise ValueError(f"{name} must have units compatible with {unit}") from exc
    if not math.isfinite(float(normalized.magnitude)) or normalized.magnitude <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return normalized


def _nonnegative_quantity(value: Quantity, unit: str, name: str) -> Quantity:
    if not isinstance(value, ureg.Quantity):
        raise TypeError(f"{name} must be a Pint Quantity")
    try:
        normalized = value.to(unit)
    except pint.DimensionalityError as exc:
        raise ValueError(f"{name} must have units compatible with {unit}") from exc
    if not math.isfinite(float(normalized.magnitude)) or normalized.magnitude < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return normalized


def _fraction(value: float, name: str, *, allow_zero: bool = True) -> float:
    value = float(value)
    lower_ok = value >= 0 if allow_zero else value > 0
    if not math.isfinite(value) or not lower_ok or value > 1:
        boundary = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be finite and in {boundary}")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be an integer greater than zero")
    return int(value)


def _device(device_key: str):
    try:
        tier, key = device_key.split(":", 1)
        registry = {
            "cloud": Hardware.Cloud,
            "workstation": Hardware.Workstation,
        }[tier]
        return getattr(registry, key)
    except (ValueError, KeyError, AttributeError) as exc:
        raise ValueError(f"unknown accelerator key: {device_key}") from exc


def get_track_config(track_id: str) -> TrackConfig:
    """Return the immutable scenario for a canonical track or notebook alias."""

    canonical = _ALIASES.get(track_id, track_id)
    try:
        return _TRACKS[canonical]
    except KeyError as exc:
        raise ValueError(f"unknown track_id: {track_id}") from exc


def _roofline_row(device_key: str, intensity: Quantity) -> dict[str, Any]:
    device = _device(device_key)
    peak = device.compute.peak_flops.to("TFLOP/s")
    bandwidth_ceiling = (device.memory.bandwidth * intensity).to("TFLOP/s")
    useful = min(peak, bandwidth_ceiling)
    bottleneck = "memory bandwidth" if bandwidth_ceiling < peak else "compute"
    return {
        "device_key": device_key,
        "device_name": device.name,
        "peak_tflops": float(peak.magnitude),
        "memory_bandwidth_gb_s": float(device.memory.bandwidth.to("GB/s").magnitude),
        "ridge_flop_per_byte": float(device.ridge_point().to("flop/byte").magnitude),
        "useful_tflops": float(useful.magnitude),
        "bottleneck": bottleneck,
        "result_kind": "analytical",
    }


def evaluate_roofline(track_id: str, arithmetic_intensity: Quantity) -> dict[str, Any]:
    """Compare two registry accelerators at one workload arithmetic intensity."""

    config = get_track_config(track_id)
    intensity = _positive_quantity(
        arithmetic_intensity, "flop/byte", "arithmetic_intensity"
    )
    rows = [_roofline_row(key, intensity) for key in config.roofline_devices]
    winner = max(rows, key=lambda row: row["useful_tflops"])
    return {
        "track_id": config.track_id,
        "model_id": MODEL_ID,
        "model_key": "evaluate_roofline",
        "infrastructure_role": config.infrastructure_role,
        "arithmetic_intensity_flop_per_byte": float(intensity.magnitude),
        "rows": rows,
        "winner_device_key": winner["device_key"],
        "inputs": {
            "track_id": config.track_id,
            "arithmetic_intensity": serialize_quantity(intensity),
        },
        "scenario_note": _SCENARIO_NOTE,
        "result_kind": "analytical",
    }


def evaluate_memory_floor(
    track_id: str,
    *,
    device_key: str | None = None,
    working_set: Quantity | None = None,
    bytes_moved: Quantity | None = None,
    operations: Quantity | None = None,
) -> dict[str, Any]:
    """Separate memory capacity feasibility from the bandwidth latency floor."""

    config = get_track_config(track_id)
    key = device_key or config.memory_device
    device = _device(key)
    footprint = _positive_quantity(
        config.working_set if working_set is None else working_set,
        "GiB",
        "working_set",
    )
    traffic = _positive_quantity(
        config.bytes_moved if bytes_moved is None else bytes_moved,
        "GiB",
        "bytes_moved",
    )
    work = _positive_quantity(
        config.operations if operations is None else operations,
        "TFLOP",
        "operations",
    )
    capacity = device.memory.capacity.to("GiB")
    fits = footprint <= capacity
    bandwidth_time = transfer_time(traffic, device.memory.bandwidth).to("ms")
    compute_time = (work / device.compute.peak_flops).to("ms")
    latency_floor = max(bandwidth_time, compute_time) if fits else None
    bottleneck = None
    if fits:
        bottleneck = "memory bandwidth" if bandwidth_time > compute_time else "compute"
    return {
        "track_id": config.track_id,
        "model_id": MODEL_ID,
        "model_key": "evaluate_memory_floor",
        "device_key": key,
        "device_name": device.name,
        "working_set_gib": float(footprint.magnitude),
        "capacity_gib": float(capacity.magnitude),
        "headroom_gib": float((capacity - footprint).magnitude),
        "fits": bool(fits),
        "bytes_moved_gib": float(traffic.magnitude),
        "bandwidth_floor_ms": float(bandwidth_time.magnitude),
        "compute_floor_ms": float(compute_time.magnitude),
        "latency_floor_ms": None if latency_floor is None else float(latency_floor.magnitude),
        "bottleneck": bottleneck,
        "inputs": {
            "track_id": config.track_id,
            "device_key": key,
            "working_set": serialize_quantity(footprint),
            "bytes_moved": serialize_quantity(traffic),
            "operations": serialize_quantity(work),
        },
        "scenario_note": _SCENARIO_NOTE,
        "result_kind": "analytical",
    }


def evaluate_memory_scale(
    track_id: str,
    *,
    scale: float,
    device_key: str | None = None,
) -> dict[str, Any]:
    """Scale one track workload's footprint and traffic from a fixed baseline."""

    config = get_track_config(track_id)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and greater than zero")
    result = evaluate_memory_floor(
        track_id,
        device_key=device_key,
        working_set=config.working_set * scale,
        bytes_moved=config.bytes_moved * scale,
        operations=config.operations,
    )
    result["model_key"] = "evaluate_memory_scale"
    result["scale"] = scale
    result["inputs"] = {
        "track_id": config.track_id,
        "scale": scale,
        "device_key": result["device_key"],
    }
    return result


def compare_transfer_tiers(
    track_id: str,
    *,
    payload: Quantity | None = None,
    device_key: str | None = None,
) -> dict[str, Any]:
    """Move one identical payload over registry-backed bandwidth tiers.

    NVLink is reported at its one-way deliverable rate, rather than NVIDIA's
    bidirectional aggregate. Missing registry links are reported as omitted/unmodeled
    rather than fabricated. Latency and contention are outside this simple
    streaming lower bound and are stated in the result.
    """

    config = get_track_config(track_id)
    moved = _positive_quantity(
        config.transfer_payload if payload is None else payload, "GiB", "payload"
    )
    key = device_key or config.memory_device
    device = _device(key)

    tiers: list[tuple[str, Quantity]] = [
        ("accelerator memory", device.memory.bandwidth),
    ]
    omitted_links: list[str] = []

    if device.nvlink is not None:
        tiers.append(
            ("within-node accelerator link", device.nvlink.bandwidth_per_direction)
        )
    else:
        omitted_links.append(
            f"within-node accelerator link (peer link not modeled in registry for {device.name})"
        )

    if device.interconnect is not None:
        tiers.append(
            ("host-to-accelerator link", device.interconnect.bandwidth_per_direction)
        )
    else:
        omitted_links.append(
            f"host-to-accelerator link (host I/O not modeled in registry for {device.name})"
        )

    tiers.append((config.network_tier_name, config.network_bandwidth))
    tiers.append(
        ("durable object store", Systems.Storage.ObjectStoreSingleStream.bandwidth),
    )
    rows = []
    for name, bandwidth in tiers:
        duration = transfer_time(moved, bandwidth)
        rows.append(
            {
                "tier": name,
                "bandwidth_gb_s": float(bandwidth.to("GB/s").magnitude),
                "transfer_ms": float(duration.to("ms").magnitude),
            }
        )
    return {
        "track_id": config.track_id,
        "model_id": MODEL_ID,
        "model_key": "compare_transfer_tiers",
        "device_key": key,
        "device_name": device.name,
        "payload_gib": float(moved.magnitude),
        "baseline_tier": "accelerator memory",
        "default_destination_tier": config.network_tier_name,
        "omitted_links": omitted_links,
        "rows": rows,
        "assumption": (
            "streaming lower bound; excludes startup latency and contention; "
            "network bandwidth is an illustrative configuration, not inherent device specification"
        ),
        "inputs": {
            "track_id": config.track_id,
            "payload": serialize_quantity(moved),
            "device_key": key,
        },
        "scenario_note": _SCENARIO_NOTE,
        "result_kind": "analytical",
    }


def evaluate_transfer_tier(
    track_id: str,
    *,
    tier: str,
    payload: Quantity | None = None,
    device_key: str | None = None,
    baseline_tier: str = "accelerator memory",
) -> dict[str, Any]:
    """Evaluate one named transfer tier for an immutable placement contrast."""

    comparison = compare_transfer_tiers(
        track_id, payload=payload, device_key=device_key
    )
    selected = next(
        (row for row in comparison["rows"] if row["tier"] == tier),
        None,
    )
    if selected is None:
        valid = ", ".join(row["tier"] for row in comparison["rows"])
        raise ValueError(f"unknown transfer tier: {tier}; choose one of {valid}")

    base = next(
        (row for row in comparison["rows"] if row["tier"] == baseline_tier),
        None,
    )
    if base is None:
        valid = ", ".join(row["tier"] for row in comparison["rows"])
        raise ValueError(f"unknown baseline tier: {baseline_tier}; choose one of {valid}")
    slowdown_vs_baseline = selected["transfer_ms"] / base["transfer_ms"]
    result: dict[str, Any] = {
        "track_id": comparison["track_id"],
        "model_id": MODEL_ID,
        "model_key": "evaluate_transfer_tier",
        "device_key": comparison["device_key"],
        "tier": selected["tier"],
        "baseline_tier": base["tier"],
        "payload_gib": comparison["payload_gib"],
        "bandwidth_gb_s": selected["bandwidth_gb_s"],
        "transfer_ms": selected["transfer_ms"],
        "slowdown_vs_baseline": slowdown_vs_baseline,
        "omitted_links": comparison["omitted_links"],
        "assumption": comparison["assumption"],
        "inputs": {
            "track_id": comparison["track_id"],
            "tier": selected["tier"],
            "baseline_tier": base["tier"],
            "payload": comparison["inputs"]["payload"],
            "device_key": comparison["device_key"],
        },
        "scenario_note": _SCENARIO_NOTE,
        "result_kind": "analytical",
    }
    within_node = next(
        (row for row in comparison["rows"] if row["tier"] == "within-node accelerator link"),
        None,
    )
    if within_node is not None:
        result["slowdown_vs_within_node"] = (
            selected["transfer_ms"] / within_node["transfer_ms"]
        )
    return result


def get_transfer_tier_options(
    track_id: str,
    device_key: str | None = None,
) -> dict[str, str]:
    """Return selectable destination transfer tiers excluding the baseline."""

    comparison = compare_transfer_tiers(track_id, device_key=device_key)
    base = comparison["baseline_tier"]
    return {
        row["tier"]: row["tier"]
        for row in comparison["rows"]
        if row["tier"] != base
    }


def get_facility_count_policy(track_id: str) -> dict[str, int]:
    """Return backend policy for facility accelerator count controls."""

    config = get_track_config(track_id)
    base = config.accelerator_count
    min_count = min(1, base)
    max_count = max(16, base * 2)
    default_count = max(base + 1, min(base * 2, max_count)) if base < max_count else max(2, base)
    return {
        "baseline_count": base,
        "min_count": min_count,
        "max_count": max_count,
        "default_count": default_count,
        "step": 1,
    }


def evaluate_facility(
    track_id: str,
    *,
    device_key: str | None = None,
    accelerator_count: int | None = None,
    accelerators_per_node: int | None = None,
    nodes_per_rack: int | None = None,
    node_overhead: Quantity | None = None,
    rack_overhead: Quantity | None = None,
    electrical_capacity: Quantity | None = None,
    cooling_capacity: Quantity | None = None,
    pue: float | None = None,
) -> dict[str, Any]:
    """Check IT heat and PUE-loaded electrical draw against separate limits."""

    config = get_track_config(track_id)
    key = device_key or config.memory_device
    device = _device(key)
    if device.tdp is None:
        raise ValueError(f"{device.name} has no registry TDP")
    count = _positive_int(
        config.accelerator_count if accelerator_count is None else accelerator_count,
        "accelerator_count",
    )
    per_node = _positive_int(
        config.accelerators_per_node
        if accelerators_per_node is None
        else accelerators_per_node,
        "accelerators_per_node",
    )
    per_rack = _positive_int(
        config.nodes_per_rack if nodes_per_rack is None else nodes_per_rack,
        "nodes_per_rack",
    )
    node_count = math.ceil(count / per_node)
    rack_count = math.ceil(node_count / per_rack)
    node_extra = _positive_quantity(
        config.node_overhead if node_overhead is None else node_overhead,
        "W",
        "node_overhead",
    )
    rack_extra = _positive_quantity(
        config.rack_overhead if rack_overhead is None else rack_overhead,
        "W",
        "rack_overhead",
    )
    electrical_limit = _positive_quantity(
        config.electrical_capacity
        if electrical_capacity is None
        else electrical_capacity,
        "W",
        "electrical_capacity",
    )
    cooling_limit = _positive_quantity(
        config.cooling_capacity if cooling_capacity is None else cooling_capacity,
        "W",
        "cooling_capacity",
    )
    facility_pue = float(
        Infrastructure.FacilityCooling.LiquidCooled.pue if pue is None else pue
    )
    if not math.isfinite(facility_pue) or facility_pue < 1:
        raise ValueError("pue must be finite and at least 1")
    accelerator_power = count * device.tdp
    it_power = (accelerator_power + node_count * node_extra + rack_count * rack_extra).to("kW")
    facility_power = (it_power * facility_pue).to("kW")
    electrical_ok = facility_power <= electrical_limit
    cooling_ok = it_power <= cooling_limit
    return {
        "track_id": config.track_id,
        "model_id": MODEL_ID,
        "model_key": "evaluate_facility",
        "device_key": key,
        "accelerator_count": count,
        "node_count": node_count,
        "rack_count": rack_count,
        "accelerator_power_kw": float(accelerator_power.to("kW").magnitude),
        "it_power_kw": float(it_power.magnitude),
        "facility_power_kw": float(facility_power.magnitude),
        "electrical_capacity_kw": float(electrical_limit.to("kW").magnitude),
        "cooling_capacity_kw": float(cooling_limit.to("kW").magnitude),
        "electrical_ok": bool(electrical_ok),
        "cooling_ok": bool(cooling_ok),
        "feasible": bool(electrical_ok and cooling_ok),
        "pue": facility_pue,
        "inputs": {
            "track_id": config.track_id,
            "device_key": key,
            "accelerator_count": count,
            "accelerators_per_node": per_node,
            "nodes_per_rack": per_rack,
            "node_overhead": serialize_quantity(node_extra),
            "rack_overhead": serialize_quantity(rack_extra),
            "electrical_capacity": serialize_quantity(electrical_limit),
            "cooling_capacity": serialize_quantity(cooling_limit),
            "pue": facility_pue,
        },
        "scenario_note": _SCENARIO_NOTE,
        "result_kind": "analytical",
    }


def compare_systems(
    track_id: str,
    *,
    arithmetic_intensity: Quantity,
    horizon: Quantity = Q_("3 year"),
    duty_cycle: float = 0.60,
    sustained_efficiency: float = 0.55,
    idle_power_fraction: float = 0.25,
    electricity_price: Quantity | None = None,
    accelerator_count: int | None = None,
    non_accelerator_capex: Quantity = Q_("0 USD"),
    maintenance_fraction_per_year: float | None = None,
) -> dict[str, Any]:
    """Compare equal-count systems over one common horizon and duty cycle.

    Capital cost covers the registry accelerator prices plus an explicit
    non-accelerator scenario input.  The model reports this boundary rather
    than implying that it is full datacenter TCO.
    """

    config = get_track_config(track_id)
    intensity = _positive_quantity(
        arithmetic_intensity, "flop/byte", "arithmetic_intensity"
    )
    duration = _positive_quantity(horizon, "hour", "horizon")
    duty = _fraction(duty_cycle, "duty_cycle", allow_zero=False)
    efficiency = _fraction(
        sustained_efficiency, "sustained_efficiency", allow_zero=False
    )
    idle_fraction = _fraction(idle_power_fraction, "idle_power_fraction")
    count = _positive_int(
        config.accelerator_count if accelerator_count is None else accelerator_count,
        "accelerator_count",
    )
    price = (
        Infrastructure.Pricing.OnPremises.ElectricityPerKwh.rate
        if electricity_price is None
        else electricity_price
    )
    price = _positive_quantity(price, "USD/kWh", "electricity_price")
    extra_capex = _nonnegative_quantity(
        non_accelerator_capex, "USD", "non_accelerator_capex"
    )
    maintenance = (
        float(Infrastructure.Pricing.Capital.AnnualMaintenanceRatio.rate)
        if maintenance_fraction_per_year is None
        else _fraction(maintenance_fraction_per_year, "maintenance_fraction_per_year")
    )
    years = duration.to("year").magnitude
    rows = []
    for key in config.procurement_devices:
        device = _device(key)
        if device.unit_cost is None or device.tdp is None:
            raise ValueError(f"{device.name} lacks registry cost or TDP")
        roofline = _roofline_row(key, intensity)
        memory = evaluate_memory_floor(track_id, device_key=key)
        facility = evaluate_facility(
            track_id,
            device_key=key,
            accelerator_count=count,
        )
        active_it_power = Q_(facility["it_power_kw"], "kW")
        average_it_power = active_it_power * (duty + (1 - duty) * idle_fraction)
        facility_energy = (average_it_power * duration * facility["pue"]).to("kWh")
        capex = (count * device.unit_cost + extra_capex).to("USD")
        maintenance_cost = capex * maintenance * years
        electricity_cost = (facility_energy * price).to("USD")
        ownership_cost = capex + maintenance_cost + electricity_cost
        active_rate = (
            count * Q_(roofline["useful_tflops"], "TFLOP/s") * efficiency
        )
        # A PFLOP-hour is a rate of PFLOP/s sustained for one hour.
        useful_pflop_hours = (
            active_rate.to("PFLOP/s").magnitude * duty * duration.to("hour").magnitude
        )
        rows.append(
            {
                "device_key": key,
                "device_name": device.name,
                "memory_fits": memory["fits"],
                "facility_fits": facility["feasible"],
                "feasible": memory["fits"] and facility["feasible"],
                "active_sustained_tflops": float(active_rate.to("TFLOP/s").magnitude),
                "average_sustained_tflops": float(
                    (active_rate * duty).to("TFLOP/s").magnitude
                ),
                "useful_pflop_hours": float(useful_pflop_hours),
                "capex_usd": float(capex.magnitude),
                "maintenance_usd": float(maintenance_cost.magnitude),
                "electricity_usd": float(electricity_cost.magnitude),
                "ownership_cost_usd": float(ownership_cost.magnitude),
                "usd_per_useful_pflop_hour": float(ownership_cost.magnitude / useful_pflop_hours),
                "average_it_power_kw": float(average_it_power.to("kW").magnitude),
                "facility_energy_kwh": float(facility_energy.magnitude),
                "bottleneck": roofline["bottleneck"],
            }
        )
    feasible_rows = [row for row in rows if row["feasible"]]
    selected = (
        min(feasible_rows, key=lambda row: row["usd_per_useful_pflop_hour"])["device_key"]
        if feasible_rows
        else None
    )
    return {
        "track_id": config.track_id,
        "model_id": MODEL_ID,
        "model_key": "compare_systems",
        "horizon_hours": float(duration.magnitude),
        "duty_cycle": duty,
        "sustained_efficiency": efficiency,
        "idle_power_fraction": idle_fraction,
        "accelerator_count": count,
        "rows": rows,
        "lowest_cost_per_work_feasible_device_key": selected,
        "cost_boundary": "accelerator capital, explicit added capital, maintenance, and electricity",
        "inputs": {
            "track_id": config.track_id,
            "arithmetic_intensity": serialize_quantity(intensity),
            "horizon": serialize_quantity(duration),
            "duty_cycle": duty,
            "sustained_efficiency": efficiency,
            "idle_power_fraction": idle_fraction,
            "electricity_price": serialize_quantity(price),
            "accelerator_count": count,
            "non_accelerator_capex": serialize_quantity(extra_capex),
            "maintenance_fraction_per_year": maintenance,
        },
        "scenario_note": _SCENARIO_NOTE,
        "result_kind": "analytical",
    }


def replay(model_key: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Replay a recognized evaluator from its serialized ``inputs`` payload."""

    evaluators = {
        "evaluate_roofline": evaluate_roofline,
        "evaluate_memory_floor": evaluate_memory_floor,
        "evaluate_memory_scale": evaluate_memory_scale,
        "compare_transfer_tiers": compare_transfer_tiers,
        "evaluate_transfer_tier": evaluate_transfer_tier,
        "evaluate_facility": evaluate_facility,
        "compare_systems": compare_systems,
    }
    try:
        evaluator = evaluators[model_key]
    except KeyError as exc:
        raise ValueError(f"unknown model_key for {MODEL_ID}: {model_key}") from exc
    if not isinstance(inputs, dict):
        raise TypeError("inputs must be a dictionary")
    decoded = {
        key: deserialize_quantity(value)
        if isinstance(value, dict) and set(value) == {"magnitude", "unit"}
        else value
        for key, value in inputs.items()
    }
    return evaluator(**decoded)


__all__ = [
    "MODEL_ID",
    "TrackConfig",
    "compare_systems",
    "compare_transfer_tiers",
    "deserialize_quantity",
    "evaluate_facility",
    "evaluate_memory_floor",
    "evaluate_memory_scale",
    "evaluate_roofline",
    "evaluate_transfer_tier",
    "get_facility_count_policy",
    "get_track_config",
    "get_transfer_tier_options",
    "replay",
    "serialize_quantity",
]
