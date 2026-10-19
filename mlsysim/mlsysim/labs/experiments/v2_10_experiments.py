"""Deterministic inference-fleet experiments for Volume II, Chapter 10.

The models in this module are deliberately small enough for browser execution.
They operate on explicit request records and report analytical or simulated
results.  They are not measurements of a production service.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Literal, Sequence

from mlsysim.core.units import ureg


TrackId = Literal["tinyml", "mobile", "edge", "cloud"]
Policy = Literal["immediate", "windowed_batch"]
MODEL_ID = "v2_10_experiments"


def _quantity(value, unit, name: str):
    if not isinstance(value, ureg.Quantity):
        raise TypeError(f"{name} must be a Pint Quantity")
    try:
        return value.to(unit)
    except Exception as exc:  # pragma: no cover - Pint supplies the detail
        raise ValueError(f"{name} must have units compatible with {unit}") from exc


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def serialize_quantity(value) -> dict[str, float | str]:
    """Serialize a finite scalar Pint quantity without discarding its unit."""

    if not isinstance(value, ureg.Quantity):
        raise TypeError("value must be a Pint Quantity")
    magnitude = float(value.magnitude)
    if not math.isfinite(magnitude):
        raise ValueError("serialized quantity magnitude must be finite")
    return {"magnitude": magnitude, "unit": str(value.units)}


def deserialize_quantity(data: dict[str, float | str]):
    """Restore a quantity produced by :func:`serialize_quantity`."""

    if isinstance(data, ureg.Quantity):
        return data
    if not isinstance(data, dict):
        raise TypeError("quantity data must be a dictionary or Pint Quantity")
    if set(data) != {"magnitude", "unit"}:
        raise ValueError("quantity data must contain exactly magnitude and unit")
    return float(data["magnitude"]) * ureg.Unit(str(data["unit"]))


@dataclass(frozen=True)
class Request:
    """One request in an explicit arrival trace.

    ``input_units`` and ``output_units`` are tokens for language services and
    compatible work units (frames, windows, or decisions) for other tracks.
    ``state_bytes`` is KV state only when the workload actually uses a KV cache.
    """

    request_id: str
    arrival: object
    input_units: int
    output_units: int
    state_bytes: object

    def __post_init__(self) -> None:
        arrival = _quantity(self.arrival, ureg.millisecond, "arrival")
        state = _quantity(self.state_bytes, ureg.byte, "state_bytes")
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.input_units < 0 or self.output_units < 0:
            raise ValueError("request work units must be nonnegative")
        if arrival.magnitude < 0 or state.magnitude < 0:
            raise ValueError("arrival and state_bytes must be nonnegative")

    def to_evaluator_args(self) -> dict[str, object]:
        """Return replayable arguments for immutable evidence capture."""

        return {
            "request_id": self.request_id,
            "arrival": serialize_quantity(self.arrival),
            "input_units": self.input_units,
            "output_units": self.output_units,
            "state_bytes": serialize_quantity(self.state_bytes),
        }


@dataclass(frozen=True)
class ServiceModel:
    """Explicit per-batch execution and memory assumptions."""

    fixed_time: object
    input_time_per_unit: object
    output_time_per_unit: object
    device_memory: object
    weight_memory: object
    reserved_memory: object
    fragmentation_fraction: float = 0.0

    def __post_init__(self) -> None:
        quantities = (
            _quantity(self.fixed_time, ureg.millisecond, "fixed_time"),
            _quantity(self.input_time_per_unit, ureg.millisecond, "input_time_per_unit"),
            _quantity(self.output_time_per_unit, ureg.millisecond, "output_time_per_unit"),
            _quantity(self.device_memory, ureg.byte, "device_memory"),
            _quantity(self.weight_memory, ureg.byte, "weight_memory"),
            _quantity(self.reserved_memory, ureg.byte, "reserved_memory"),
        )
        if any(quantity.magnitude < 0 for quantity in quantities):
            raise ValueError("service times and memory quantities must be nonnegative")
        if not 0 <= self.fragmentation_fraction < 1:
            raise ValueError("fragmentation_fraction must be in [0, 1)")

    def to_evaluator_args(self) -> dict[str, object]:
        """Return replayable arguments for immutable evidence capture."""

        return {
            "fixed_time": serialize_quantity(self.fixed_time),
            "input_time_per_unit": serialize_quantity(self.input_time_per_unit),
            "output_time_per_unit": serialize_quantity(self.output_time_per_unit),
            "device_memory": serialize_quantity(self.device_memory),
            "weight_memory": serialize_quantity(self.weight_memory),
            "reserved_memory": serialize_quantity(self.reserved_memory),
            "fragmentation_fraction": self.fragmentation_fraction,
        }


@dataclass(frozen=True)
class CostCrossover:
    horizon: object
    training_cost: object
    serving_cost: object
    serving_to_training_ratio: float
    crossover_time: object


def lifetime_cost_crossover(
    *,
    training_cost,
    request_rate,
    cost_per_request,
    horizon,
) -> CostCrossover:
    """Compare one-time training and recurring serving over one horizon."""

    train = _quantity(training_cost, ureg.dollar, "training_cost")
    rate = _quantity(request_rate, 1 / ureg.second, "request_rate")
    unit_cost = _quantity(cost_per_request, ureg.dollar, "cost_per_request")
    span = _quantity(horizon, ureg.second, "horizon")
    if train.magnitude < 0 or rate.magnitude < 0 or unit_cost.magnitude < 0 or span.magnitude < 0:
        raise ValueError("cost, rate, and horizon inputs must be nonnegative")
    serving = (rate * span * unit_cost).to(ureg.dollar)
    ratio = serving.magnitude / train.magnitude if train.magnitude else math.inf
    if (rate * unit_cost).magnitude == 0:
        crossover = math.inf * ureg.day
    else:
        crossover = (train / (rate * unit_cost)).to(ureg.day)
    return CostCrossover(span, train, serving, ratio, crossover)


@dataclass(frozen=True)
class AdmissionResult:
    physical_free_memory: object
    usable_state_memory: object
    state_per_request: object
    max_concurrent_requests: int
    admitted_request_ids: tuple[str, ...]
    rejected_request_ids: tuple[str, ...]


def state_admission(
    requests: Sequence[Request],
    *,
    device_memory,
    weight_memory,
    reserved_memory=0 * ureg.byte,
    fragmentation_fraction: float = 0.0,
) -> AdmissionResult:
    """Admit trace records in order under weight, reserve, and fragmentation."""

    capacity = _quantity(device_memory, ureg.byte, "device_memory")
    weights = _quantity(weight_memory, ureg.byte, "weight_memory")
    reserve = _quantity(reserved_memory, ureg.byte, "reserved_memory")
    if capacity.magnitude < 0 or weights.magnitude < 0 or reserve.magnitude < 0:
        raise ValueError("memory quantities must be nonnegative")
    if not 0 <= fragmentation_fraction < 1:
        raise ValueError("fragmentation_fraction must be in [0, 1)")
    physical_free = max(0.0, (capacity - weights - reserve).m_as(ureg.byte)) * ureg.byte
    usable = physical_free * (1.0 - fragmentation_fraction)
    admitted: list[str] = []
    rejected: list[str] = []
    used = 0.0
    positive_states: list[float] = []
    for request in requests:
        state = _quantity(request.state_bytes, ureg.byte, "state_bytes").magnitude
        positive_states.append(state)
        if used + state <= usable.m_as(ureg.byte) + 1e-9:
            admitted.append(request.request_id)
            used += state
        else:
            rejected.append(request.request_id)
    representative = max(positive_states, default=0.0) * ureg.byte
    max_concurrent = (
        math.floor(usable.m_as(ureg.byte) / representative.m_as(ureg.byte))
        if representative.magnitude > 0
        else len(requests)
    )
    return AdmissionResult(
        physical_free.to(ureg.byte),
        usable.to(ureg.byte),
        representative.to(ureg.byte),
        max_concurrent,
        tuple(admitted),
        tuple(rejected),
    )


@dataclass(frozen=True)
class RequestOutcome:
    request_id: str
    replica: int | None
    batch_id: int | None
    arrival: object
    start: object | None
    finish: object | None
    duration: object | None
    on_time: bool
    admitted: bool


@dataclass(frozen=True)
class ReplayResult:
    policy: Policy
    outcomes: tuple[RequestOutcome, ...]
    completed_durations: tuple[object, ...]
    p99_duration: object
    p99_method: str
    request_count: int
    deadline_compliant_count: int
    rejected_count: int
    completed_count: int
    makespan: object

    @property
    def on_time_count(self) -> int:
        """Backward-compatible plain-language alias for deadline compliance."""

        return self.deadline_compliant_count


def _nearest_rank_p99(durations_ms: Sequence[float]) -> float:
    """Return the empirical p99 of complete request durations."""

    if not durations_ms:
        return math.inf
    ordered = sorted(durations_ms)
    rank = max(1, math.ceil(0.99 * len(ordered)))
    return ordered[rank - 1]


def replay_requests(
    requests: Iterable[Request],
    *,
    service: ServiceModel,
    replicas: int,
    policy: Policy,
    slo,
    max_batch: int = 1,
    batch_window=0 * ureg.millisecond,
    warmup=0 * ureg.millisecond,
    lost_replicas: int = 0,
    service_scale: float = 1.0,
    handoff_time=0 * ureg.millisecond,
) -> ReplayResult:
    """Replay identical arrivals through a deterministic replica timeline.

    A windowed batch starts after its explicit collection window.  Its work is
    the longest input and output in that batch, which models parallel padded
    execution without a policy multiplier.  State must fit concurrently on the
    selected replica.  Warmup delays every surviving replica; lost replicas are
    absent for the entire replay.
    """

    if replicas < 1 or lost_replicas < 0 or lost_replicas >= replicas:
        raise ValueError("at least one replica must survive")
    if max_batch < 1:
        raise ValueError("max_batch must be positive")
    if policy not in ("immediate", "windowed_batch"):
        raise ValueError(f"unknown policy: {policy}")
    scale = _nonnegative(service_scale, "service_scale")
    deadline_ms = _quantity(slo, ureg.millisecond, "slo").magnitude
    window_ms = _quantity(batch_window, ureg.millisecond, "batch_window").magnitude
    warmup_ms = _quantity(warmup, ureg.millisecond, "warmup").magnitude
    handoff_ms = _quantity(handoff_time, ureg.millisecond, "handoff_time").magnitude
    fixed_ms = service.fixed_time.to(ureg.millisecond).magnitude
    input_ms = service.input_time_per_unit.to(ureg.millisecond).magnitude
    output_ms = service.output_time_per_unit.to(ureg.millisecond).magnitude
    surviving = replicas - lost_replicas
    available = [warmup_ms for _ in range(surviving)]
    ordered = sorted(tuple(requests), key=lambda r: (r.arrival.to(ureg.millisecond).magnitude, r.request_id))
    outcomes: list[RequestOutcome] = []
    cursor = 0
    batch_id = 0
    while cursor < len(ordered):
        replica = min(range(surviving), key=lambda idx: (available[idx], idx))
        first = ordered[cursor]
        first_arrival = first.arrival.to(ureg.millisecond).magnitude
        collection_end = max(available[replica], first_arrival)
        if policy == "windowed_batch":
            collection_end += window_ms
        batch = [first]
        cursor += 1
        if policy == "windowed_batch":
            while cursor < len(ordered) and len(batch) < max_batch:
                arrival = ordered[cursor].arrival.to(ureg.millisecond).magnitude
                if arrival > collection_end:
                    break
                batch.append(ordered[cursor])
                cursor += 1

        admission = state_admission(
            batch,
            device_memory=service.device_memory,
            weight_memory=service.weight_memory,
            reserved_memory=service.reserved_memory,
            fragmentation_fraction=service.fragmentation_fraction,
        )
        admitted_ids = set(admission.admitted_request_ids)
        admitted = [request for request in batch if request.request_id in admitted_ids]
        if admitted:
            duration_ms = scale * (
                fixed_ms
                + max(request.input_units for request in admitted) * input_ms
                + max(request.output_units for request in admitted) * output_ms
            ) + handoff_ms
            finish_ms = collection_end + duration_ms
            available[replica] = finish_ms
        else:
            finish_ms = collection_end
        for request in batch:
            arrival_ms = request.arrival.to(ureg.millisecond).magnitude
            if request.request_id in admitted_ids:
                complete_ms = finish_ms - arrival_ms
                outcomes.append(RequestOutcome(
                    request.request_id, replica, batch_id,
                    arrival_ms * ureg.millisecond,
                    collection_end * ureg.millisecond,
                    finish_ms * ureg.millisecond,
                    complete_ms * ureg.millisecond,
                    complete_ms <= deadline_ms,
                    True,
                ))
            else:
                outcomes.append(RequestOutcome(
                    request.request_id, None, batch_id,
                    arrival_ms * ureg.millisecond, None, None, None, False, False,
                ))
        batch_id += 1

    completed = tuple(outcome.duration for outcome in outcomes if outcome.duration is not None)
    durations_ms = [duration.to(ureg.millisecond).magnitude for duration in completed]
    p99 = _nearest_rank_p99(durations_ms) * ureg.millisecond
    arrivals = [request.arrival.to(ureg.millisecond).magnitude for request in ordered]
    makespan_ms = max(available, default=0.0) - min(arrivals, default=0.0)
    return ReplayResult(
        policy,
        tuple(outcomes),
        completed,
        p99,
        "empirical nearest-rank over complete request durations",
        len(outcomes),
        sum(outcome.on_time for outcome in outcomes),
        sum(not outcome.admitted for outcome in outcomes),
        len(completed),
        max(0.0, makespan_ms) * ureg.millisecond,
    )


@dataclass(frozen=True)
class PlacementResult:
    placement: Literal["replicate", "shard", "split_phases"]
    device_budget: int
    replay: ReplayResult | None
    transfer_time_per_request: object | None
    available: bool = True
    unsupported_reason: str | None = None


def _replay_split_phases(
    requests: Sequence[Request],
    *,
    service: ServiceModel,
    replicas_per_phase: int,
    slo,
    handoff_time,
) -> ReplayResult:
    """Replay requests through distinct input and output phase queues."""

    deadline_ms = _quantity(slo, ureg.millisecond, "slo").magnitude
    transfer_ms = _quantity(handoff_time, ureg.millisecond, "handoff_time").magnitude
    fixed_ms = service.fixed_time.to(ureg.millisecond).magnitude
    input_ms = service.input_time_per_unit.to(ureg.millisecond).magnitude
    output_ms = service.output_time_per_unit.to(ureg.millisecond).magnitude
    stage_one_available = [0.0] * replicas_per_phase
    stage_two_available = [0.0] * replicas_per_phase
    outcomes: list[RequestOutcome] = []
    phase_service = replace(
        service,
        weight_memory=service.weight_memory / 2,
        reserved_memory=service.reserved_memory / 2,
    )
    ordered = sorted(requests, key=lambda r: (r.arrival.to(ureg.millisecond).magnitude, r.request_id))
    for batch_id, request in enumerate(ordered):
        admission = state_admission(
            (request,),
            device_memory=phase_service.device_memory,
            weight_memory=phase_service.weight_memory,
            reserved_memory=phase_service.reserved_memory,
            fragmentation_fraction=phase_service.fragmentation_fraction,
        )
        arrival_ms = request.arrival.to(ureg.millisecond).magnitude
        if admission.rejected_request_ids:
            outcomes.append(RequestOutcome(
                request.request_id, None, batch_id, arrival_ms * ureg.millisecond,
                None, None, None, False, False,
            ))
            continue
        stage_one_replica = min(
            range(replicas_per_phase), key=lambda idx: (stage_one_available[idx], idx)
        )
        stage_one_start = max(arrival_ms, stage_one_available[stage_one_replica])
        stage_one_finish = stage_one_start + fixed_ms / 2 + request.input_units * input_ms
        stage_one_available[stage_one_replica] = stage_one_finish
        ready_for_stage_two = stage_one_finish + transfer_ms
        stage_two_replica = min(
            range(replicas_per_phase), key=lambda idx: (stage_two_available[idx], idx)
        )
        stage_two_start = max(ready_for_stage_two, stage_two_available[stage_two_replica])
        finish_ms = stage_two_start + fixed_ms / 2 + request.output_units * output_ms
        stage_two_available[stage_two_replica] = finish_ms
        duration_ms = finish_ms - arrival_ms
        outcomes.append(RequestOutcome(
            request.request_id,
            stage_one_replica,
            batch_id,
            arrival_ms * ureg.millisecond,
            stage_one_start * ureg.millisecond,
            finish_ms * ureg.millisecond,
            duration_ms * ureg.millisecond,
            duration_ms <= deadline_ms,
            True,
        ))
    completed = tuple(outcome.duration for outcome in outcomes if outcome.duration is not None)
    durations_ms = [duration.to(ureg.millisecond).magnitude for duration in completed]
    first_arrival = min(
        (request.arrival.to(ureg.millisecond).magnitude for request in ordered), default=0.0
    )
    finish = max(stage_two_available, default=first_arrival)
    return ReplayResult(
        "immediate",
        tuple(outcomes),
        completed,
        _nearest_rank_p99(durations_ms) * ureg.millisecond,
        "empirical nearest-rank over complete request durations",
        len(outcomes),
        sum(outcome.on_time for outcome in outcomes),
        sum(not outcome.admitted for outcome in outcomes),
        len(completed),
        max(0.0, finish - first_arrival) * ureg.millisecond,
    )


def compare_equal_budget_placements(
    requests: Iterable[Request],
    *,
    service: ServiceModel,
    device_budget: int,
    slo,
    handoff_bytes,
    link_bandwidth,
    link_latency,
    track_id: TrackId = "cloud",
) -> tuple[PlacementResult, ...]:
    """Compare replication, sharding, and phase split with equal devices.

    The shard case divides compute across all devices and performs one explicit
    transfer per shard boundary.  The phase-split case assigns half the devices
    to each phase and includes one activation handoff.  This compact model does
    not claim overlap between phase pipelines.  Unavailable architectures for
    specific deployment tracks are reported categorically with None metrics.
    """

    if device_budget < 2:
        raise ValueError("device_budget must be at least two")
    payload = _quantity(handoff_bytes, ureg.byte, "handoff_bytes")
    bandwidth = _quantity(link_bandwidth, ureg.byte / ureg.second, "link_bandwidth")
    latency = _quantity(link_latency, ureg.millisecond, "link_latency")
    if bandwidth.magnitude <= 0:
        raise ValueError("link_bandwidth must be positive")
    one_transfer = (latency + (payload / bandwidth).to(ureg.millisecond)).to(ureg.millisecond)
    if track_id not in ("tinyml", "mobile", "edge", "cloud"):
        raise ValueError(f"unknown track_id: {track_id}")
    trace = tuple(requests)
    replicated = replay_requests(
        trace, service=service, replicas=device_budget, policy="immediate", slo=slo,
    )
    results: list[PlacementResult] = [
        PlacementResult("replicate", device_budget, replicated, 0 * ureg.millisecond, available=True, unsupported_reason=None)
    ]

    if track_id in ("tinyml", "mobile"):
        results.append(
            PlacementResult(
                "shard",
                device_budget,
                None,
                None,
                available=False,
                unsupported_reason="collective sharding is unsupported on endpoints",
            )
        )
    else:
        shard_service = replace(
            service,
            device_memory=service.device_memory * device_budget,
            reserved_memory=service.reserved_memory * device_budget,
        )
        sharded = replay_requests(
            trace,
            service=shard_service,
            replicas=1,
            policy="immediate",
            slo=slo,
            service_scale=1.0 / device_budget,
            handoff_time=(device_budget - 1) * one_transfer,
        )
        results.append(
            PlacementResult(
                "shard",
                device_budget,
                sharded,
                (device_budget - 1) * one_transfer,
                available=True,
                unsupported_reason=None,
            )
        )

    if track_id in ("tinyml", "mobile"):
        results.append(
            PlacementResult(
                "split_phases",
                device_budget,
                None,
                None,
                available=False,
                unsupported_reason="prefill/decode phase splitting is unsupported on endpoints",
            )
        )
    elif track_id == "edge":
        results.append(
            PlacementResult(
                "split_phases",
                device_budget,
                None,
                None,
                available=False,
                unsupported_reason="phase splitting is unsupported for frame perception without language phases",
            )
        )
    else:
        phase_replicas = max(1, device_budget // 2)
        phase_split = _replay_split_phases(
            trace,
            service=service,
            replicas_per_phase=phase_replicas,
            slo=slo,
            handoff_time=one_transfer,
        )
        results.append(
            PlacementResult(
                "split_phases",
                device_budget,
                phase_split,
                one_transfer,
                available=True,
                unsupported_reason=None,
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class QualifiedCost:
    """Resource cost normalized by successful, on-time requests."""

    total_cost: object
    request_count: int
    completed_count: int
    rejected_count: int
    deadline_compliant_count: int
    cost_per_deadline_compliant_request: object


def qualified_cost(*, total_cost, replay: ReplayResult) -> QualifiedCost:
    """Normalize cost without dropping rejected or late requests from evidence."""

    if not isinstance(total_cost, ureg.Quantity):
        raise TypeError("total_cost must be a Pint Quantity")
    if replay.deadline_compliant_count:
        unit_cost = total_cost / replay.deadline_compliant_count
    else:
        unit_cost = math.inf * total_cost.units
    return QualifiedCost(
        total_cost,
        replay.request_count,
        replay.completed_count,
        replay.rejected_count,
        replay.deadline_compliant_count,
        unit_cost,
    )


@dataclass(frozen=True)
class FormatEvidence:
    """Supplied outcome observation for one format, task, and population."""

    format_id: str
    task_id: str
    population_id: str
    successful_outcome_rate: float
    evidence_label: str
    illustrative: bool = True

    def __post_init__(self) -> None:
        if not self.format_id or not self.task_id or not self.population_id:
            raise ValueError("format, task, and population identifiers are required")
        if not 0 <= self.successful_outcome_rate <= 1:
            raise ValueError("successful_outcome_rate must be in [0, 1]")
        if not self.evidence_label:
            raise ValueError("evidence_label is required")


@dataclass(frozen=True)
class FormatAssessment:
    format_id: str
    accepted: bool
    observed_outcome_rate: float
    required_outcome_rate: float
    evidence_label: str
    illustrative: bool


def assess_quality_qualified_format(
    evidence: FormatEvidence,
    *,
    task_id: str,
    population_id: str,
    required_outcome_rate: float,
) -> FormatAssessment:
    """Apply a threshold only to matched supplied outcome evidence.

    This acceptance check does not alter latency, memory, or the underlying
    observed outcome.  It deliberately avoids deriving quality from a hardware
    format or compression ratio.
    """

    threshold = float(required_outcome_rate)
    if not 0 <= threshold <= 1:
        raise ValueError("required_outcome_rate must be in [0, 1]")
    if evidence.task_id != task_id or evidence.population_id != population_id:
        raise ValueError("quality evidence must match both task and population")
    return FormatAssessment(
        evidence.format_id,
        evidence.successful_outcome_rate >= threshold,
        evidence.successful_outcome_rate,
        threshold,
        evidence.evidence_label,
        evidence.illustrative,
    )


@dataclass(frozen=True)
class TrackScenario:
    """Illustrative, non-measured fixture for one compatible serving regime."""

    track_id: TrackId
    workload_kind: str
    state_kind: str
    backend_role: str
    startup_kind: str
    supported_placements: tuple[Literal["replicate", "shard", "split_phases"], ...]
    requests: tuple[Request, ...]
    service: ServiceModel
    slo: object
    training_cost: object
    request_rate_options: tuple[tuple[str, object], ...]
    cost_per_request: object
    horizon: object
    fragmentation_options: tuple[float, ...]
    batch_window_options: tuple[object, ...]
    device_budget: int
    handoff_bytes: object
    link_bandwidth: object
    link_latency: object
    baseline_replicas: int
    warmup_options: tuple[object, ...]


def illustrative_track_scenario(track_id: TrackId) -> TrackScenario:
    """Return small deterministic fixtures used to exercise all four tracks."""

    definitions = {
        "tinyml": ("sensor windows", "sensor window buffers", 250, 1.5, 0.010, 0.0, 0.001, 0.2, 0.12, 0.18, 40_000, 0.000002, 2),
        "mobile": ("camera frames", "activation buffers", 50, 2.0, 0.08, 0.0, 8, 14, 1_500, 96, 250_000, 0.00002, 4),
        "edge": ("perception frames", "sensor-frame buffers", 30, 1.0, 0.35, 0.0, 32, 17, 6_000, 1_024, 2_000_000, 0.00008, 4),
        "cloud": ("language requests", "KV cache", 200, 3.0, 0.020, 0.12, 640, 140_000, 90_000, 20_480, 2_000_000, 0.01, 8),
    }
    roles = {
        "tinyml": ("microcontroller sensor endpoint fleet", "device wake / sensor activation", ("replicate",)),
        "mobile": ("mobile on-device client fleet", "device wake / app activation", ("replicate",)),
        "edge": ("local edge gateway appliance pool", "service provisioning / container warmup", ("replicate", "shard")),
        "cloud": ("regional cloud accelerator pool", "service provisioning / cold start", ("replicate", "shard", "split_phases")),
    }
    if track_id not in definitions:
        raise ValueError(f"unknown track_id: {track_id}")
    backend_role, startup_kind, supported_placements = roles[track_id]
    (
        workload, state, slo_ms, fixed, input_ms, output_ms, memory_gb,
        weight_mb, state_mb, reserve_mb, training_usd, request_cost_usd,
        device_budget,
    ) = definitions[track_id]
    work_trace = (
        ((0, 256, 32), (3, 768, 96), (8, 128, 24), (14, 1_024, 128), (25, 384, 48))
        if track_id == "cloud"
        else ((0, 12, 2), (3, 20, 4), (8, 8, 1), (14, 30, 6), (25, 10, 2))
    )
    requests = tuple(
        Request(
            f"{track_id}-{index}",
            arrival * ureg.millisecond,
            input_units,
            output_units,
            state_mb * ureg.megabyte,
        )
        for index, (arrival, input_units, output_units) in enumerate(work_trace)
    )
    service = ServiceModel(
        fixed * ureg.millisecond,
        input_ms * ureg.millisecond,
        output_ms * ureg.millisecond,
        memory_gb * ureg.gigabyte,
        weight_mb * ureg.megabyte,
        reserve_mb * ureg.megabyte,
        0.10,
    )
    transfer_mb = {"tinyml": 1, "mobile": 8, "edge": 32, "cloud": 128}[track_id]
    bandwidth_gbs = {"tinyml": 0.05, "mobile": 1, "edge": 10, "cloud": 50}[track_id]
    link_latency_ms = {"tinyml": 8, "mobile": 3, "edge": 0.5, "cloud": 0.1}[track_id]
    rate_hz = {"tinyml": 0.02, "mobile": 2, "edge": 30, "cloud": 100}[track_id]
    return TrackScenario(
        track_id,
        workload,
        state,
        backend_role,
        startup_kind,
        supported_placements,
        requests,
        service,
        slo_ms * ureg.millisecond,
        training_usd * ureg.dollar,
        (("normal", rate_hz / ureg.second), ("growth", 5 * rate_hz / ureg.second)),
        request_cost_usd * ureg.dollar,
        26 * ureg.week,
        (0.0, 0.15, 0.30),
        (0 * ureg.millisecond, 10 * ureg.millisecond, 25 * ureg.millisecond),
        device_budget,
        transfer_mb * ureg.megabyte,
        bandwidth_gbs * ureg.gigabyte / ureg.second,
        link_latency_ms * ureg.millisecond,
        4,
        (0 * ureg.millisecond, (slo_ms / 2) * ureg.millisecond, slo_ms * ureg.millisecond),
    )


def unavailable_snapshot(
    *,
    inputs: dict[str, object],
    reason: str = "unsupported architecture for track",
) -> dict[str, object]:
    """Return an explicit unavailable structured record with None metrics for JSON."""

    return {
        "inputs": inputs,
        "available": False,
        "status": f"unavailable: {reason}",
        "unsupported_reason": reason,
        "request_count": None,
        "completed_count": None,
        "rejected_count": None,
        "deadline_compliant_count": None,
        "p99_duration": None,
        "p99_status": f"unavailable: {reason}",
        "p99_method": "unavailable",
        "makespan": None,
        "completed_durations": [],
    }


def replay_snapshot(
    result: ReplayResult | None,
    *,
    inputs: dict[str, object],
    available: bool = True,
    unsupported_reason: str | None = None,
) -> dict[str, object]:
    """Return a JSON-ready replay result with exact evaluator arguments."""

    if not available or result is None:
        return unavailable_snapshot(inputs=inputs, reason=unsupported_reason or "unsupported architecture for track")
    p99_available = math.isfinite(result.p99_duration.magnitude)
    return {
        "inputs": inputs,
        "available": True,
        "status": "available",
        "unsupported_reason": None,
        "request_count": result.request_count,
        "completed_count": result.completed_count,
        "rejected_count": result.rejected_count,
        "deadline_compliant_count": result.deadline_compliant_count,
        "p99_duration": serialize_quantity(result.p99_duration) if p99_available else None,
        "p99_status": "available" if p99_available else "unavailable: no completed requests",
        "p99_method": result.p99_method,
        "makespan": serialize_quantity(result.makespan),
        "completed_durations": [serialize_quantity(value) for value in result.completed_durations],
    }


def admission_snapshot(result: AdmissionResult, *, inputs: dict[str, object]) -> dict[str, object]:
    """Return a JSON-ready state-admission result."""

    return {
        "inputs": inputs,
        "physical_free_memory": serialize_quantity(result.physical_free_memory),
        "usable_state_memory": serialize_quantity(result.usable_state_memory),
        "state_per_request": serialize_quantity(result.state_per_request),
        "max_concurrent_requests": result.max_concurrent_requests,
        "admitted_request_ids": list(result.admitted_request_ids),
        "rejected_request_ids": list(result.rejected_request_ids),
    }


def cost_snapshot(result: CostCrossover, *, inputs: dict[str, object]) -> dict[str, object]:
    """Return a JSON-ready lifetime cost comparison."""

    crossover_available = math.isfinite(result.crossover_time.magnitude)
    return {
        "inputs": inputs,
        "horizon": serialize_quantity(result.horizon),
        "training_cost": serialize_quantity(result.training_cost),
        "serving_cost": serialize_quantity(result.serving_cost),
        "serving_to_training_ratio": result.serving_to_training_ratio,
        "crossover_time": serialize_quantity(result.crossover_time) if crossover_available else None,
        "crossover_status": "available" if crossover_available else "unavailable: zero serving cost rate",
    }


__all__ = [
    "AdmissionResult",
    "CostCrossover",
    "FormatAssessment",
    "FormatEvidence",
    "MODEL_ID",
    "PlacementResult",
    "QualifiedCost",
    "ReplayResult",
    "Request",
    "RequestOutcome",
    "ServiceModel",
    "TrackScenario",
    "compare_equal_budget_placements",
    "admission_snapshot",
    "assess_quality_qualified_format",
    "deserialize_quantity",
    "cost_snapshot",
    "illustrative_track_scenario",
    "lifetime_cost_crossover",
    "qualified_cost",
    "replay_requests",
    "replay_snapshot",
    "serialize_quantity",
    "state_admission",
    "unavailable_snapshot",
]
