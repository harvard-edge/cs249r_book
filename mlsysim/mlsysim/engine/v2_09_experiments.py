"""Physical performance-engineering experiments for Volume II, Chapter 9.

The model evaluates a repeated workload path phase by phase.  Each phase has
useful operations, memory traffic, and launch overhead.  Fusion changes
intermediate traffic and launch count; tiling changes off-chip reuse and
workspace; batching amortizes fixed tensor traffic while increasing live
state; precision changes bytes and selects a supported execution rate; and
compilation trades one-time setup (performed ahead of time on an engineering
host for device fleets, or during server graph capture in cloud) for lower
repeated dispatch overhead.

TinyML, mobile, and edge fleet units are independent deployed devices.  Their
memory does not pool.  The cloud unit is one four-H100 sharded serving replica.
Fleet size therefore scales aggregate useful throughput only; feasibility and
per-run latency remain properties of one execution unit.

Hardware ceilings come from the MLSysIM registry.  Workload shapes, attainable
rate fractions, and quality observations are bounded teaching fixtures rather
than measurements of the named products or universal quality laws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mlsysim.core.types import Quantity
from mlsysim.core.units import Q_, resolve_precision
from mlsysim.hardware.registry import Hardware

MODEL_ID = "v2_09_experiments"


@dataclass(frozen=True)
class EvaluationInputs:
    """Fully resolved evaluator arguments retained with every result."""

    track_id: str
    batch_size: int
    fleet_units: int
    precision: str
    fusion: bool
    tiling: bool
    compiled: bool
    reuse_count: int


@dataclass(frozen=True)
class PhaseSpec:
    """Useful work and FP16-equivalent tensor traffic for one workload phase."""

    name: str
    operations_per_item: Quantity
    fixed_elements: int
    stream_elements_per_item: int
    intermediate_elements_per_item: int
    launches: int


@dataclass(frozen=True)
class PrecisionEvidence:
    """Supported rate and fixed illustrative task outcome for one precision."""

    precision: str
    peak_rate: Quantity
    attainable_fraction: float
    quality_pct: float
    evidence_label: str


@dataclass(frozen=True)
class TrackScenario:
    """One bounded fleet scenario and its per-unit physical supply."""

    track_id: str
    display_name: str
    workload: str
    fleet_semantics: str
    hardware_name: str
    accelerators_per_unit: int
    memory_capacity: Quantity
    memory_bandwidth: Quantity
    dispatch_latency: Quantity
    resident_elements: int
    live_elements_per_item: int
    fusion_workspace: Quantity
    tiling_workspace: Quantity
    compile_setup: Quantity
    default_batch_size: int
    default_fleet_units: int
    default_precision: str
    phases: tuple[PhaseSpec, ...]
    precision_evidence: tuple[PrecisionEvidence, ...]
    provenance: str


@dataclass(frozen=True)
class PhaseResult:
    """Analytical timing decomposition for one phase."""

    name: str
    operations: Quantity
    traffic: Quantity
    compute_time: Quantity
    movement_time: Quantity
    exposed_time: Quantity
    launch_time: Quantity
    launches: int


@dataclass(frozen=True)
class ExperimentResult:
    """A complete simulated/analytical result for one execution unit and fleet."""

    track_id: str
    batch_size: int
    fleet_units: int
    precision: str
    fusion: bool
    tiling: bool
    compiled: bool
    reuse_count: int
    phases: tuple[PhaseResult, ...]
    total_operations: Quantity
    total_traffic: Quantity
    compute_time: Quantity
    movement_time: Quantity
    exposed_kernel_time: Quantity
    launch_time: Quantity
    compile_time_per_run: Quantity
    latency: Quantity
    memory_required: Quantity
    memory_capacity: Quantity
    memory_feasible: bool
    bottleneck: str
    arithmetic_intensity: Quantity
    per_unit_throughput: Quantity
    fleet_throughput: Quantity
    attainable_per_unit_throughput: Quantity
    attainable_fleet_throughput: Quantity
    potential_per_unit_throughput: Quantity
    potential_fleet_throughput: Quantity
    quality_pct: float
    quality_evidence: str
    result_label: str
    inputs: EvaluationInputs


def _phase(
    name: str,
    operations: str,
    fixed_elements: int,
    stream_elements: int,
    intermediate_elements: int,
    launches: int,
) -> PhaseSpec:
    return PhaseSpec(
        name=name,
        operations_per_item=Q_(operations),
        fixed_elements=fixed_elements,
        stream_elements_per_item=stream_elements,
        intermediate_elements_per_item=intermediate_elements,
        launches=launches,
    )


def _precision(
    precision: str,
    peak_rate: Quantity,
    attainable_fraction: float,
    quality_pct: float,
) -> PrecisionEvidence:
    return PrecisionEvidence(
        precision=precision,
        peak_rate=peak_rate,
        attainable_fraction=attainable_fraction,
        quality_pct=quality_pct,
        evidence_label=(
            "Illustrative fixed task outcome supplied for this scenario; "
            "it is not a hardware-derived or measured quality claim."
        ),
    )


_TINY = Hardware.Tiny.nRF52840
_MOBILE = Hardware.Mobile.AppleM2
_EDGE = Hardware.Edge.JetsonAGXOrin
_CLOUD = Hardware.Cloud.H100


TRACKS: dict[str, TrackScenario] = {
    "tinyml": TrackScenario(
        track_id="tinyml",
        display_name="TinyML",
        workload="per-device sensor-window inference across a wearable fleet",
        fleet_semantics="independent devices; memory and latency are per device",
        hardware_name=_TINY.name,
        accelerators_per_unit=1,
        memory_capacity=_TINY.memory.capacity,
        memory_bandwidth=_TINY.memory.bandwidth,
        dispatch_latency=_TINY.dispatch_tax,
        resident_elements=170_000,
        live_elements_per_item=25_000,
        fusion_workspace=Q_("32 KiB"),
        tiling_workspace=Q_("96 KiB"),
        compile_setup=Q_("18 ms"),
        default_batch_size=1,
        default_fleet_units=10_000,
        default_precision="int8",
        phases=(
            _phase("feature extraction", "0.06 MFLOP", 20_000, 18_000, 35_000, 3),
            _phase("classifier", "0.10 MFLOP", 150_000, 5_000, 20_000, 2),
        ),
        precision_evidence=(
            _precision("fp32", _TINY.compute.peak_flops, 0.55, 92.0),
            _precision("int8", _TINY.compute.precision_flops["int8"], 0.62, 91.7),
        ),
        provenance="Registry hardware ceilings plus an illustrative wearable-workload fixture.",
    ),
    "mobile": TrackScenario(
        track_id="mobile",
        display_name="Mobile",
        workload="per-phone assistant turn across a regional enrolled-device fleet",
        fleet_semantics="independent phones; unified memory is local to each phone",
        hardware_name=_MOBILE.name,
        accelerators_per_unit=1,
        memory_capacity=_MOBILE.memory.capacity,
        memory_bandwidth=_MOBILE.memory.bandwidth,
        dispatch_latency=_MOBILE.dispatch_tax,
        resident_elements=420_000_000,
        live_elements_per_item=25_000_000,
        fusion_workspace=Q_("96 MiB"),
        tiling_workspace=Q_("320 MiB"),
        compile_setup=Q_("420 ms"),
        default_batch_size=1,
        default_fleet_units=25_000,
        default_precision="int8",
        phases=(
            _phase("prompt encoding", "0.9 TFLOP", 160_000_000, 18_000_000, 65_000_000, 7),
            _phase("token step", "0.55 TFLOP", 260_000_000, 7_000_000, 40_000_000, 6),
        ),
        precision_evidence=(
            _precision("fp16", _MOBILE.compute.peak_flops, 0.34, 89.4),
            _precision("int8", _MOBILE.compute.peak_flops, 0.58, 88.8),
        ),
        provenance="Registry hardware ceilings plus an illustrative on-device assistant fixture.",
    ),
    "edge": TrackScenario(
        track_id="edge",
        display_name="Edge",
        workload="per-vehicle perception frame across an autonomous fleet",
        fleet_semantics="independent vehicles; each frame must fit and finish locally",
        hardware_name=_EDGE.name,
        accelerators_per_unit=1,
        memory_capacity=_EDGE.memory.capacity,
        memory_bandwidth=_EDGE.memory.bandwidth,
        dispatch_latency=_EDGE.dispatch_tax,
        resident_elements=1_300_000_000,
        live_elements_per_item=180_000_000,
        fusion_workspace=Q_("480 MiB"),
        tiling_workspace=Q_("2.5 GiB"),
        compile_setup=Q_("760 ms"),
        default_batch_size=1,
        default_fleet_units=2_000,
        default_precision="int8",
        phases=(
            _phase("sensor backbone", "4.8 TFLOP", 900_000_000, 130_000_000, 360_000_000, 10),
            _phase("detection heads", "2.6 TFLOP", 400_000_000, 50_000_000, 210_000_000, 8),
        ),
        precision_evidence=(
            _precision("fp16", _EDGE.compute.peak_flops, 0.30, 94.6),
            _precision("int8", _EDGE.compute.precision_flops["int8"], 0.52, 93.9),
        ),
        provenance="Registry hardware ceilings plus an illustrative perception fixture.",
    ),
    "cloud": TrackScenario(
        track_id="cloud",
        display_name="Cloud",
        workload="one sharded language-model serving path in a regional replica pool",
        fleet_semantics="each unit is a four-H100 replica; fleet units add independent replicas",
        hardware_name=f"4 x {_CLOUD.name}",
        accelerators_per_unit=4,
        memory_capacity=_CLOUD.memory.capacity * 4,
        memory_bandwidth=_CLOUD.memory.bandwidth * 4,
        dispatch_latency=_CLOUD.dispatch_tax,
        resident_elements=135_000_000_000,
        live_elements_per_item=1_100_000_000,
        fusion_workspace=Q_("8 GiB"),
        tiling_workspace=Q_("28 GiB"),
        compile_setup=Q_("2.4 s"),
        default_batch_size=2,
        default_fleet_units=24,
        default_precision="fp16",
        phases=(
            _phase("prefill", "34 TFLOP", 65_000_000_000, 800_000_000, 8_000_000_000, 14),
            _phase("decode", "18 TFLOP", 70_000_000_000, 300_000_000, 5_000_000_000, 12),
        ),
        precision_evidence=(
            _precision("fp16", _CLOUD.compute.peak_flops * 4, 0.48, 86.7),
            _precision("fp8", _CLOUD.compute.precision_flops["fp8"] * 4, 0.54, 86.2),
            _precision("int8", _CLOUD.compute.precision_flops["int8"] * 4, 0.50, 85.8),
        ),
        provenance="Registry H100 ceilings plus an illustrative four-way sharded serving fixture.",
    ),
}


def get_scenario(track_id: str) -> TrackScenario:
    """Return a track scenario or reject an unknown track explicitly."""

    try:
        return TRACKS[track_id]
    except KeyError as exc:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}") from exc


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _precision_evidence(scenario: TrackScenario, precision: str) -> PrecisionEvidence:
    canonical, _ = resolve_precision(precision)
    for evidence in scenario.precision_evidence:
        if evidence.precision == canonical:
            return evidence
    supported = ", ".join(item.precision for item in scenario.precision_evidence)
    raise ValueError(f"{canonical} is not supported for {scenario.track_id}; choose {supported}")


def evaluate(
    track_id: str,
    *,
    batch_size: int | None = None,
    fleet_units: int | None = None,
    precision: str | None = None,
    fusion: bool = False,
    tiling: bool = False,
    compiled: bool = False,
    reuse_count: int = 1,
) -> ExperimentResult:
    """Evaluate one phase path from physical work and hardware ceilings.

    Kernel compute and movement may overlap within a phase, so the larger term
    is exposed.  Phases remain sequential and launch/compile costs are added.
    Combined optimizations alter the shared traffic, launch, workspace, and
    execution-rate inputs before time is recomputed.
    """

    scenario = get_scenario(track_id)
    batch_size = _positive_int(
        scenario.default_batch_size if batch_size is None else batch_size,
        "batch_size",
    )
    fleet_units = _positive_int(
        scenario.default_fleet_units if fleet_units is None else fleet_units,
        "fleet_units",
    )
    reuse_count = _positive_int(reuse_count, "reuse_count")
    evidence = _precision_evidence(scenario, precision or scenario.default_precision)
    _, bytes_per_element = resolve_precision(evidence.precision)
    effective_rate = evidence.peak_rate * evidence.attainable_fraction

    phase_results: list[PhaseResult] = []
    total_operations = Q_("0 flop")
    total_traffic = Q_("0 byte")
    total_compute = Q_("0 ms")
    total_movement = Q_("0 ms")
    total_exposed = Q_("0 ms")
    total_launch = Q_("0 ms")

    for phase in scenario.phases:
        operations = phase.operations_per_item * batch_size
        fixed_elements = phase.fixed_elements * (0.62 if tiling else 1.0)
        intermediate_fraction = (0.22 if fusion else 1.0) * (0.78 if tiling else 1.0)
        traffic_elements = (
            fixed_elements
            + batch_size * phase.stream_elements_per_item
            + batch_size * phase.intermediate_elements_per_item * intermediate_fraction
        )
        traffic = bytes_per_element * traffic_elements
        launches = max(1, phase.launches - (math.ceil(phase.launches * 0.55) if fusion else 0))
        dispatch_factor = 0.28 if compiled else 1.0
        launch_time = scenario.dispatch_latency * launches * dispatch_factor
        compute_time = (operations / effective_rate).to("ms")
        movement_time = (traffic / scenario.memory_bandwidth).to("ms")
        exposed_time = max(compute_time, movement_time)
        phase_results.append(
            PhaseResult(
                name=phase.name,
                operations=operations.to("flop"),
                traffic=traffic.to("byte"),
                compute_time=compute_time,
                movement_time=movement_time,
                exposed_time=exposed_time,
                launch_time=launch_time.to("ms"),
                launches=launches,
            )
        )
        total_operations += operations
        total_traffic += traffic
        total_compute += compute_time
        total_movement += movement_time
        total_exposed += exposed_time
        total_launch += launch_time

    workspace = Q_("0 byte")
    if fusion:
        workspace += scenario.fusion_workspace
    if tiling:
        workspace += scenario.tiling_workspace
    memory_required = (
        scenario.resident_elements * bytes_per_element
        + scenario.live_elements_per_item * batch_size * bytes_per_element
        + workspace
    ).to("byte")
    memory_feasible = memory_required <= scenario.memory_capacity
    compile_time = (scenario.compile_setup / reuse_count if compiled else Q_("0 ms")).to("ms")
    latency = (total_exposed + total_launch + compile_time).to("ms")
    potential_per_unit_throughput = (batch_size / latency.to("s")).to("1/s")
    potential_fleet_throughput = potential_per_unit_throughput * fleet_units
    attainable_per_unit_throughput = (
        potential_per_unit_throughput if memory_feasible else Q_("0 1/s")
    )
    attainable_fleet_throughput = (
        potential_fleet_throughput if memory_feasible else Q_("0 1/s")
    )
    per_unit_throughput = attainable_per_unit_throughput
    fleet_throughput = attainable_fleet_throughput

    if total_launch + compile_time > max(total_compute, total_movement):
        bottleneck = "overhead"
    elif total_compute >= total_movement:
        bottleneck = "compute"
    else:
        bottleneck = "memory"

    return ExperimentResult(
        track_id=track_id,
        batch_size=batch_size,
        fleet_units=fleet_units,
        precision=evidence.precision,
        fusion=fusion,
        tiling=tiling,
        compiled=compiled,
        reuse_count=reuse_count,
        phases=tuple(phase_results),
        total_operations=total_operations.to("flop"),
        total_traffic=total_traffic.to("byte"),
        compute_time=total_compute.to("ms"),
        movement_time=total_movement.to("ms"),
        exposed_kernel_time=total_exposed.to("ms"),
        launch_time=total_launch.to("ms"),
        compile_time_per_run=compile_time,
        latency=latency,
        memory_required=memory_required,
        memory_capacity=scenario.memory_capacity.to("byte"),
        memory_feasible=memory_feasible,
        bottleneck=bottleneck,
        arithmetic_intensity=(total_operations / total_traffic).to("flop/byte"),
        per_unit_throughput=per_unit_throughput,
        fleet_throughput=fleet_throughput,
        attainable_per_unit_throughput=attainable_per_unit_throughput,
        attainable_fleet_throughput=attainable_fleet_throughput,
        potential_per_unit_throughput=potential_per_unit_throughput,
        potential_fleet_throughput=potential_fleet_throughput,
        quality_pct=evidence.quality_pct,
        quality_evidence=evidence.evidence_label,
        result_label="simulated analytical bound",
        inputs=EvaluationInputs(
            track_id=track_id,
            batch_size=batch_size,
            fleet_units=fleet_units,
            precision=evidence.precision,
            fusion=fusion,
            tiling=tiling,
            compiled=compiled,
            reuse_count=reuse_count,
        ),
    )


def compare(track_id: str, **candidate_options: object) -> dict[str, object]:
    """Compare an intervention with an identical unoptimized baseline."""

    scenario = get_scenario(track_id)
    shared = {
        "batch_size": candidate_options.get("batch_size", scenario.default_batch_size),
        "fleet_units": candidate_options.get("fleet_units", scenario.default_fleet_units),
        "precision": scenario.default_precision,
        "reuse_count": candidate_options.get("reuse_count", 1),
    }
    return compare_configs(track_id, baseline_options=shared, result_options=candidate_options)


def compare_configs(
    track_id: str,
    *,
    baseline_options: dict[str, object],
    result_options: dict[str, object],
) -> dict[str, object]:
    """Compare two explicit configurations and retain both exact input sets."""

    baseline = evaluate(track_id, **baseline_options)
    candidate = evaluate(track_id, **result_options)
    speedup = (baseline.latency / candidate.latency).to_base_units().magnitude
    return {
        "baseline": baseline,
        "result": candidate,
        "speedup": speedup,
        "latency_improved": candidate.latency < baseline.latency,
        "latency_delta": candidate.latency - baseline.latency,
        "traffic_delta": sum((item.traffic for item in candidate.phases), Q_("0 byte"))
        - sum((item.traffic for item in baseline.phases), Q_("0 byte")),
        "memory_delta": candidate.memory_required - baseline.memory_required,
        "quality_delta_pct": candidate.quality_pct - baseline.quality_pct,
    }


def quantity_to_dict(quantity: Quantity) -> dict[str, object]:
    """Serialize a physical quantity as a JSON-safe magnitude and unit pair."""

    magnitude = quantity.magnitude
    if hasattr(magnitude, "item"):
        magnitude = magnitude.item()
    return {"magnitude": magnitude, "unit": str(quantity.units)}


def quantity_from_dict(payload: dict[str, object]) -> Quantity:
    """Restore a quantity produced by :func:`quantity_to_dict`."""

    if set(payload) != {"magnitude", "unit"}:
        raise ValueError("quantity payload must contain exactly magnitude and unit")
    if not isinstance(payload["unit"], str):
        raise ValueError("quantity unit must be a string")
    return Q_(payload["magnitude"], payload["unit"])


def result_to_dict(result: ExperimentResult) -> dict[str, object]:
    """Serialize a result with exact replay inputs and unit-bearing outputs."""

    quantity_fields = (
        "compute_time",
        "movement_time",
        "exposed_kernel_time",
        "launch_time",
        "compile_time_per_run",
        "latency",
        "memory_required",
        "memory_capacity",
        "arithmetic_intensity",
        "per_unit_throughput",
        "fleet_throughput",
        "attainable_per_unit_throughput",
        "attainable_fleet_throughput",
        "potential_per_unit_throughput",
        "potential_fleet_throughput",
        "total_operations",
        "total_traffic",
    )
    payload: dict[str, object] = {
        "model_id": MODEL_ID,
        "inputs": {
            "track_id": result.inputs.track_id,
            "batch_size": result.inputs.batch_size,
            "fleet_units": result.inputs.fleet_units,
            "precision": result.inputs.precision,
            "fusion": result.inputs.fusion,
            "tiling": result.inputs.tiling,
            "compiled": result.inputs.compiled,
            "reuse_count": result.inputs.reuse_count,
        },
        "memory_feasible": result.memory_feasible,
        "bottleneck": result.bottleneck,
        "quality_pct": result.quality_pct,
        "quality_evidence": result.quality_evidence,
        "result_label": result.result_label,
        "phases": [
            {
                "name": phase.name,
                "operations": quantity_to_dict(phase.operations),
                "traffic": quantity_to_dict(phase.traffic),
                "compute_time": quantity_to_dict(phase.compute_time),
                "movement_time": quantity_to_dict(phase.movement_time),
                "exposed_time": quantity_to_dict(phase.exposed_time),
                "launch_time": quantity_to_dict(phase.launch_time),
                "launches": phase.launches,
            }
            for phase in result.phases
        ],
    }
    payload.update(
        {field: quantity_to_dict(getattr(result, field)) for field in quantity_fields}
    )
    return payload


def comparison_to_dict(comparison: dict[str, object]) -> dict[str, object]:
    """Serialize a comparison using the ledger contract's baseline/result keys."""

    baseline = comparison.get("baseline")
    result = comparison.get("result")
    if not isinstance(baseline, ExperimentResult) or not isinstance(result, ExperimentResult):
        raise ValueError("comparison must contain ExperimentResult baseline and result values")
    return {
        "model_id": MODEL_ID,
        "baseline": result_to_dict(baseline),
        "result": result_to_dict(result),
        "speedup": comparison["speedup"],
        "latency_improved": comparison["latency_improved"],
        "latency_delta": quantity_to_dict(comparison["latency_delta"]),
        "traffic_delta": quantity_to_dict(comparison["traffic_delta"]),
        "memory_delta": quantity_to_dict(comparison["memory_delta"]),
        "quality_delta_pct": comparison["quality_delta_pct"],
    }


__all__ = [
    "TRACKS",
    "MODEL_ID",
    "EvaluationInputs",
    "ExperimentResult",
    "PhaseResult",
    "PhaseSpec",
    "PrecisionEvidence",
    "TrackScenario",
    "compare",
    "compare_configs",
    "comparison_to_dict",
    "evaluate",
    "get_scenario",
    "quantity_from_dict",
    "quantity_to_dict",
    "result_to_dict",
]
