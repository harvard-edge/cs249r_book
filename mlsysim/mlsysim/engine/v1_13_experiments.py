"""Deterministic request-level experiments for single-system model serving.

The analytical serving solvers answer steady-state capacity questions.  This
module complements them with a small discrete-event replay for teaching
transients, batching timeouts, admission loss, and complete request latency.
It deliberately models one bounded serving system; it does not model fleet
routing or distributed autoscaling.

Every scenario in this module is illustrative.  The values are explicit
assumptions for controlled comparisons, not measurements of named products.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

from pint import Quantity

from mlsysim.core.units import Q_


Track = Literal["tinyml", "mobile", "edge", "cloud"]
ArrivalPattern = Literal["steady", "bursty", "shift"]
PolicyName = Literal["baseline", "reserve", "activate", "admit"]
BatchingPolicyName = Literal["single", "pair-short", "quad-patient"]
StatePressure = Literal["baseline", "elevated", "overflow"]
MODEL_ID = "v1_13_experiments"
REPLAY_METHOD = "simulate_serving"


@dataclass(frozen=True)
class PreprocessingPath:
    """A synchronous preprocessing path and its output contract."""

    name: str
    latency: Quantity
    output_signature: str


@dataclass(frozen=True)
class ServingScenario:
    """Illustrative single-system workload and resource assumptions."""

    track: Track
    label: str
    workload: str
    deadline: Quantity
    ingress_latency: Quantity
    execution_latency: Quantity
    postprocess_latency: Quantity
    batch_marginal_cost: float
    device_memory: Quantity
    resident_model_memory: Quantity
    request_working_state: Quantity
    request_kv_state: Quantity
    batch_workspace: Quantity
    reference_preprocessing_signature: str
    preprocessing_paths: tuple[PreprocessingPath, ...]
    assumption_note: str
    serving_role: str = "single-system serving"


@dataclass(frozen=True)
class ArrivalPhase:
    """A constant-rate segment in a deterministic offered-load trace."""

    duration: Quantity
    rate_qps: float


@dataclass(frozen=True)
class ServingPolicy:
    """Batching, capacity activation, and admission controls."""

    name: str
    batch_limit: int
    batch_timeout: Quantity
    reserved_replicas: int
    on_demand_replicas: int = 0
    activation_start: Quantity = Q_(0, "ms")
    activation_delay: Quantity = Q_(0, "ms")
    queue_capacity: int = 128


@dataclass(frozen=True)
class StateCapacity:
    """Per-replica live-state ceiling under the selected batch workspace."""

    feasible: bool
    fixed_memory: Quantity
    state_per_request: Quantity
    kv_state_per_request: Quantity
    max_requests_per_replica: int


@dataclass(frozen=True)
class RequestRecord:
    """Captured lifecycle of one offered request."""

    request_id: int
    arrival: Quantity
    ready: Quantity
    dispatch: Quantity | None
    completion: Quantity | None
    formation_wait: Quantity | None
    resource_queue_wait: Quantity | None
    total_latency: Quantity | None
    admitted: bool
    deadline_met: bool
    useful: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class BatchRecord:
    """One batch dispatched to one bounded serving replica."""

    replica_id: int
    request_ids: tuple[int, ...]
    eligible_at: Quantity
    dispatch: Quantity
    execution_latency: Quantity
    completion: Quantity


@dataclass(frozen=True)
class ServingReplayInputs:
    """Exact evaluation arguments retained for evidence and deterministic replay."""

    scenario: ServingScenario
    arrivals: tuple[Quantity, ...]
    policy: ServingPolicy
    preprocessing: PreprocessingPath


@dataclass(frozen=True)
class ServingSimulationResult:
    """Observable outcomes from one deterministic request replay."""

    inputs: ServingReplayInputs
    offered_count: int
    admitted_count: int
    rejected_count: int
    completed_count: int
    deadline_met_count: int
    deadline_missed_count: int
    useful_completion_count: int
    useful_completion_fraction: float
    preprocessing_parity: bool
    p50_latency: Quantity | None
    p99_latency: Quantity | None
    mean_latency: Quantity | None
    max_queue_depth: int
    busy_replica_time: Quantity
    provisioned_replica_time: Quantity
    records: tuple[RequestRecord, ...]
    batches: tuple[BatchRecord, ...]


_SCENARIOS: dict[Track, ServingScenario] = {
    "tinyml": ServingScenario(
        track="tinyml",
        label="TinyML",
        workload="local keyword event detection",
        deadline=Q_(40, "ms"),
        ingress_latency=Q_(0.4, "ms"),
        execution_latency=Q_(7.5, "ms"),
        postprocess_latency=Q_(0.6, "ms"),
        batch_marginal_cost=0.72,
        device_memory=Q_(512, "KiB"),
        resident_model_memory=Q_(310, "KiB"),
        request_working_state=Q_(18, "KiB"),
        request_kv_state=Q_(0, "byte"),
        batch_workspace=Q_(48, "KiB"),
        reference_preprocessing_signature="pcm16-16khz-logmel-v1",
        preprocessing_paths=(
            PreprocessingPath("Matched audio features", Q_(2.2, "ms"), "pcm16-16khz-logmel-v1"),
            PreprocessingPath("Mismatched audio features", Q_(1.4, "ms"), "pcm16-8khz-logmel-v1"),
        ),
        assumption_note="Illustrative local-device workload; working state is an activation/audio buffer, not KV state.",
        serving_role="Dedicated on-device microcontroller",
    ),
    "mobile": ServingScenario(
        track="mobile",
        label="Mobile",
        workload="on-device image classification",
        deadline=Q_(80, "ms"),
        ingress_latency=Q_(1.5, "ms"),
        execution_latency=Q_(18, "ms"),
        postprocess_latency=Q_(1.0, "ms"),
        batch_marginal_cost=0.68,
        device_memory=Q_(768, "MiB"),
        resident_model_memory=Q_(220, "MiB"),
        request_working_state=Q_(42, "MiB"),
        request_kv_state=Q_(0, "byte"),
        batch_workspace=Q_(96, "MiB"),
        reference_preprocessing_signature="rgb-224-bilinear-v1",
        preprocessing_paths=(
            PreprocessingPath("Matched image transform", Q_(7, "ms"), "rgb-224-bilinear-v1"),
            PreprocessingPath("Different resize transform", Q_(4, "ms"), "rgb-224-nearest-v1"),
        ),
        assumption_note="Illustrative on-device vision workload; live state is image/activation workspace, not KV state.",
        serving_role="Dedicated on-device application",
    ),
    "edge": ServingScenario(
        track="edge",
        label="Edge",
        workload="shared camera-event inference",
        deadline=Q_(120, "ms"),
        ingress_latency=Q_(4, "ms"),
        execution_latency=Q_(22, "ms"),
        postprocess_latency=Q_(3, "ms"),
        batch_marginal_cost=0.48,
        device_memory=Q_(4, "GiB"),
        resident_model_memory=Q_(1.1, "GiB"),
        request_working_state=Q_(160, "MiB"),
        request_kv_state=Q_(0, "byte"),
        batch_workspace=Q_(420, "MiB"),
        reference_preprocessing_signature="yuv-rgb-letterbox-v1",
        preprocessing_paths=(
            PreprocessingPath("Matched camera transform", Q_(10, "ms"), "yuv-rgb-letterbox-v1"),
            PreprocessingPath("Stale crop transform", Q_(7, "ms"), "yuv-rgb-center-crop-v0"),
        ),
        assumption_note="Illustrative shared edge-vision workload; live state is frame/activation workspace, not KV state.",
        serving_role="Shared edge gateway",
    ),
    "cloud": ServingScenario(
        track="cloud",
        label="Cloud",
        workload="interactive language generation step",
        deadline=Q_(250, "ms"),
        ingress_latency=Q_(8, "ms"),
        execution_latency=Q_(32, "ms"),
        postprocess_latency=Q_(4, "ms"),
        batch_marginal_cost=0.32,
        device_memory=Q_(24, "GiB"),
        resident_model_memory=Q_(13.5, "GiB"),
        request_working_state=Q_(180, "MiB"),
        request_kv_state=Q_(620, "MiB"),
        batch_workspace=Q_(1.2, "GiB"),
        reference_preprocessing_signature="tokenizer-v3-bos-on",
        preprocessing_paths=(
            PreprocessingPath("Matched tokenizer", Q_(6, "ms"), "tokenizer-v3-bos-on"),
            PreprocessingPath("Legacy tokenizer", Q_(4, "ms"), "tokenizer-v2-bos-off"),
        ),
        assumption_note="Illustrative language workload; per-request live state includes an explicit KV-cache allowance.",
        serving_role="Multi-tenant cloud service",
    ),
}


_ARRIVAL_PHASES: dict[tuple[Track, ArrivalPattern], tuple[ArrivalPhase, ...]] = {
    ("tinyml", "steady"): (ArrivalPhase(Q_(4, "s"), 18),),
    ("tinyml", "bursty"): (ArrivalPhase(Q_(1, "s"), 12), ArrivalPhase(Q_(0.8, "s"), 75), ArrivalPhase(Q_(1, "s"), 12)),
    ("tinyml", "shift"): (ArrivalPhase(Q_(1.5, "s"), 15), ArrivalPhase(Q_(1.5, "s"), 42)),
    ("mobile", "steady"): (ArrivalPhase(Q_(4, "s"), 10),),
    ("mobile", "bursty"): (ArrivalPhase(Q_(1, "s"), 6), ArrivalPhase(Q_(0.8, "s"), 38), ArrivalPhase(Q_(1, "s"), 6)),
    ("mobile", "shift"): (ArrivalPhase(Q_(1.5, "s"), 8), ArrivalPhase(Q_(1.5, "s"), 24)),
    ("edge", "steady"): (ArrivalPhase(Q_(4, "s"), 28),),
    ("edge", "bursty"): (ArrivalPhase(Q_(1, "s"), 20), ArrivalPhase(Q_(0.8, "s"), 105), ArrivalPhase(Q_(1, "s"), 20)),
    ("edge", "shift"): (ArrivalPhase(Q_(1.5, "s"), 24), ArrivalPhase(Q_(1.5, "s"), 68)),
    ("cloud", "steady"): (ArrivalPhase(Q_(4, "s"), 45),),
    ("cloud", "bursty"): (ArrivalPhase(Q_(1, "s"), 32), ArrivalPhase(Q_(0.8, "s"), 145), ArrivalPhase(Q_(1, "s"), 32)),
    ("cloud", "shift"): (ArrivalPhase(Q_(1.5, "s"), 38), ArrivalPhase(Q_(1.5, "s"), 95)),
}


def get_serving_scenario(track: Track) -> ServingScenario:
    """Return the immutable illustrative scenario for a teaching track."""

    try:
        return _SCENARIOS[track]
    except KeyError as exc:
        raise ValueError(f"unknown serving track: {track!r}") from exc


def preprocessing_path(scenario: ServingScenario, name: str) -> PreprocessingPath:
    """Select one of a scenario's explicit preprocessing alternatives."""

    for path in scenario.preprocessing_paths:
        if path.name == name:
            return path
    raise ValueError(f"unknown preprocessing path for {scenario.track}: {name!r}")


def make_arrival_trace(
    phases: Sequence[ArrivalPhase], *, start: Quantity = Q_(0, "ms")
) -> tuple[Quantity, ...]:
    """Build an explicit deterministic arrival trace from piecewise rates.

    Each phase starts at its boundary.  Arrivals are evenly spaced within a
    phase, which isolates the causal effect of the specified rate changes.
    """

    now_ms = _milliseconds(start, "start")
    if now_ms < 0:
        raise ValueError("start must be nonnegative")
    arrivals: list[Quantity] = []
    for phase in phases:
        duration_ms = _milliseconds(phase.duration, "phase duration")
        if duration_ms <= 0:
            raise ValueError("phase duration must be positive")
        if not math.isfinite(phase.rate_qps) or phase.rate_qps < 0:
            raise ValueError("phase rate_qps must be finite and nonnegative")
        if phase.rate_qps > 0:
            interval_ms = 1000.0 / phase.rate_qps
            offset_ms = 0.0
            while offset_ms < duration_ms - 1e-9:
                arrivals.append(Q_(now_ms + offset_ms, "ms"))
                offset_ms += interval_ms
        now_ms += duration_ms
    return tuple(arrivals)


def arrival_trace_for_track(
    track: Track, pattern: ArrivalPattern, *, rate_scale: float = 1.0
) -> tuple[Quantity, ...]:
    """Return a controlled traffic trace, optionally scaling every phase rate."""

    get_serving_scenario(track)
    if not math.isfinite(rate_scale) or rate_scale <= 0:
        raise ValueError("rate_scale must be finite and positive")
    try:
        phases = _ARRIVAL_PHASES[(track, pattern)]
    except KeyError as exc:
        raise ValueError(f"unknown arrival pattern: {pattern!r}") from exc
    scaled_phases = tuple(
        ArrivalPhase(duration=phase.duration, rate_qps=phase.rate_qps * rate_scale)
        for phase in phases
    )
    return make_arrival_trace(scaled_phases)


def policy_for_track(track: Track, name: PolicyName) -> ServingPolicy:
    """Return comparable policy choices scaled to a track's latency regime."""

    scenario = get_serving_scenario(track)
    timeout = scenario.deadline * 0.08
    if name == "baseline":
        return ServingPolicy(name="baseline", batch_limit=1, batch_timeout=Q_(0, "ms"), reserved_replicas=1)
    if name == "reserve":
        return ServingPolicy(name="reserve", batch_limit=2, batch_timeout=timeout, reserved_replicas=2)
    if name == "activate":
        return ServingPolicy(
            name="activate",
            batch_limit=2,
            batch_timeout=timeout,
            reserved_replicas=1,
            on_demand_replicas=1,
            activation_start=Q_(900, "ms"),
            activation_delay=scenario.deadline * 1.5,
        )
    if name == "admit":
        return ServingPolicy(
            name="admit",
            batch_limit=2,
            batch_timeout=timeout,
            reserved_replicas=1,
            queue_capacity=4,
        )
    raise ValueError(f"unknown serving policy: {name!r}")


def batching_policy_for_track(track: Track, name: BatchingPolicyName) -> ServingPolicy:
    """Return controlled batch-size/timeout alternatives for one replica."""

    scenario = get_serving_scenario(track)
    if name == "single":
        return ServingPolicy(name="single", batch_limit=1, batch_timeout=Q_(0, "ms"), reserved_replicas=1)
    if name == "pair-short":
        return ServingPolicy(
            name="pair-short",
            batch_limit=2,
            batch_timeout=scenario.deadline * 0.03,
            reserved_replicas=1,
        )
    if name == "quad-patient":
        return ServingPolicy(
            name="quad-patient",
            batch_limit=4,
            batch_timeout=scenario.deadline * 0.08,
            reserved_replicas=1,
        )
    raise ValueError(f"unknown batching policy: {name!r}")


def state_pressure_scenario(track: Track, pressure: StatePressure) -> ServingScenario:
    """Return a controlled live-state alternative for concurrency experiments."""

    scenario = get_serving_scenario(track)
    if pressure == "baseline":
        return scenario
    if pressure == "elevated":
        if track == "cloud":
            return replace(
                scenario,
                request_kv_state=scenario.request_kv_state * 2,
                assumption_note=scenario.assumption_note + " Elevated case doubles per-request KV state.",
            )
        return replace(
            scenario,
            request_working_state=scenario.request_working_state * 2,
            assumption_note=scenario.assumption_note + " Elevated case doubles per-request working state.",
        )
    if pressure == "overflow":
        if track == "cloud":
            return replace(
                scenario,
                request_kv_state=scenario.device_memory,
                assumption_note=scenario.assumption_note + " Overflow case makes one request's KV state equal device memory.",
            )
        return replace(
            scenario,
            request_working_state=scenario.device_memory,
            assumption_note=scenario.assumption_note + " Overflow case makes one request's working state equal device memory.",
        )
    raise ValueError(f"unknown state pressure: {pressure!r}")


def state_capacity(scenario: ServingScenario) -> StateCapacity:
    """Calculate the per-replica request-state ceiling with Pint quantities."""

    fixed = (scenario.resident_model_memory + scenario.batch_workspace).to("byte")
    per_request = (scenario.request_working_state + scenario.request_kv_state).to("byte")
    available = (scenario.device_memory.to("byte") - fixed).to("byte")
    feasible = available.magnitude >= 0 and per_request.magnitude > 0
    maximum = math.floor(available.magnitude / per_request.magnitude) if feasible else 0
    return StateCapacity(
        feasible=feasible and maximum >= 1,
        fixed_memory=fixed,
        state_per_request=per_request,
        kv_state_per_request=scenario.request_kv_state.to("byte"),
        max_requests_per_replica=max(0, maximum),
    )


def simulate_serving(
    scenario: ServingScenario,
    arrivals: Sequence[Quantity],
    policy: ServingPolicy,
    preprocessing: PreprocessingPath | None = None,
) -> ServingSimulationResult:
    """Replay arrivals through timeout-aware batching and a bounded pool.

    Percentiles use the nearest-rank empirical definition over complete
    durations of completed requests.  Rejected requests are reported
    separately, and all offered requests remain in the denominator of useful
    completion fraction.
    """

    _validate_policy(policy)
    path = preprocessing or scenario.preprocessing_paths[0]
    parity = path.output_signature == scenario.reference_preprocessing_signature
    arrival_ms = [_milliseconds(value, "arrival") for value in arrivals]
    if any(value < 0 for value in arrival_ms):
        raise ValueError("arrivals must be nonnegative")
    if any(left > right for left, right in zip(arrival_ms, arrival_ms[1:])):
        raise ValueError("arrivals must be sorted")

    ingress_ms = _milliseconds(scenario.ingress_latency, "ingress latency")
    preprocess_ms = _milliseconds(path.latency, "preprocessing latency")
    execution_ms = _milliseconds(scenario.execution_latency, "execution latency")
    postprocess_ms = _milliseconds(scenario.postprocess_latency, "postprocess latency")
    timeout_ms = _milliseconds(policy.batch_timeout, "batch timeout")
    deadline_ms = _milliseconds(scenario.deadline, "deadline")
    activation_ms = _milliseconds(policy.activation_start, "activation start")
    activation_delay_ms = _milliseconds(policy.activation_delay, "activation delay")
    if min(ingress_ms, preprocess_ms, execution_ms, postprocess_ms, timeout_ms, deadline_ms) < 0:
        raise ValueError("latencies and deadlines must be nonnegative")
    if deadline_ms <= 0 or execution_ms <= 0:
        raise ValueError("deadline and execution latency must be positive")
    if not 0 < scenario.batch_marginal_cost <= 1:
        raise ValueError("batch_marginal_cost must be in (0, 1]")

    capacity = state_capacity(scenario)
    effective_batch_limit = min(policy.batch_limit, capacity.max_requests_per_replica)
    server_available = [0.0] * policy.reserved_replicas
    server_available.extend(
        [activation_ms + activation_delay_ms] * policy.on_demand_replicas
    )

    ready_ms = [value + ingress_ms + preprocess_ms for value in arrival_ms]
    pending: list[int] = []
    cursor = 0
    records: list[RequestRecord | None] = [None] * len(arrival_ms)
    batches: list[BatchRecord] = []
    max_queue_depth = 0
    busy_replica_ms = 0.0

    if effective_batch_limit == 0:
        for request_id, (arrival, ready) in enumerate(zip(arrival_ms, ready_ms)):
            records[request_id] = _rejected_record(request_id, arrival, ready, "state capacity")
    else:
        while cursor < len(arrival_ms) or pending:
            if not pending:
                next_ready = ready_ms[cursor]
                cursor = _admit_ready_requests(
                    next_ready, ready_ms, arrival_ms, cursor, pending, records, policy.queue_capacity
                )
                max_queue_depth = max(max_queue_depth, len(pending))

            if len(pending) >= effective_batch_limit:
                eligible_ms = min(
                    ready_ms[pending[effective_batch_limit - 1]],
                    ready_ms[pending[0]] + timeout_ms,
                )
            else:
                eligible_ms = ready_ms[pending[0]] + timeout_ms
            replica_id = min(range(len(server_available)), key=server_available.__getitem__)
            candidate_dispatch_ms = max(eligible_ms, server_available[replica_id])

            if cursor < len(arrival_ms) and ready_ms[cursor] <= candidate_dispatch_ms + 1e-9:
                cursor = _admit_ready_requests(
                    ready_ms[cursor], ready_ms, arrival_ms, cursor, pending, records, policy.queue_capacity
                )
                max_queue_depth = max(max_queue_depth, len(pending))
                continue

            request_ids = tuple(pending[:effective_batch_limit])
            del pending[: len(request_ids)]
            batch_execution_ms = execution_ms * (
                1.0 + scenario.batch_marginal_cost * (len(request_ids) - 1)
            )
            batch_completion_ms = candidate_dispatch_ms + batch_execution_ms
            server_available[replica_id] = batch_completion_ms
            busy_replica_ms += batch_execution_ms
            batches.append(
                BatchRecord(
                    replica_id=replica_id,
                    request_ids=request_ids,
                    eligible_at=Q_(eligible_ms, "ms"),
                    dispatch=Q_(candidate_dispatch_ms, "ms"),
                    execution_latency=Q_(batch_execution_ms, "ms"),
                    completion=Q_(batch_completion_ms, "ms"),
                )
            )
            for request_id in request_ids:
                formation_wait_ms = max(0.0, eligible_ms - ready_ms[request_id])
                resource_wait_ms = max(0.0, candidate_dispatch_ms - max(eligible_ms, ready_ms[request_id]))
                completion_ms = batch_completion_ms + postprocess_ms
                total_ms = completion_ms - arrival_ms[request_id]
                deadline_met = total_ms <= deadline_ms + 1e-9
                records[request_id] = RequestRecord(
                    request_id=request_id,
                    arrival=Q_(arrival_ms[request_id], "ms"),
                    ready=Q_(ready_ms[request_id], "ms"),
                    dispatch=Q_(candidate_dispatch_ms, "ms"),
                    completion=Q_(completion_ms, "ms"),
                    formation_wait=Q_(formation_wait_ms, "ms"),
                    resource_queue_wait=Q_(resource_wait_ms, "ms"),
                    total_latency=Q_(total_ms, "ms"),
                    admitted=True,
                    deadline_met=deadline_met,
                    useful=deadline_met and parity,
                )

    final_records = tuple(record for record in records if record is not None)
    if len(final_records) != len(arrival_ms):
        raise RuntimeError("request replay did not produce a record for every offered request")
    completed = [record for record in final_records if record.completion is not None]
    latencies_ms = [record.total_latency.to("ms").magnitude for record in completed if record.total_latency is not None]
    rejected_count = sum(not record.admitted for record in final_records)
    deadline_met_count = sum(record.deadline_met for record in completed)
    useful_count = sum(record.useful for record in final_records)
    horizon_ms = max(
        [0.0, *arrival_ms, *[record.completion.to("ms").magnitude for record in completed if record.completion]],
    )
    activation_times = [0.0] * policy.reserved_replicas + [
        activation_ms + activation_delay_ms
    ] * policy.on_demand_replicas
    provisioned_ms = sum(max(0.0, horizon_ms - active_at) for active_at in activation_times)

    return ServingSimulationResult(
        inputs=ServingReplayInputs(
            scenario=scenario,
            arrivals=tuple(Q_(value, "ms") for value in arrival_ms),
            policy=policy,
            preprocessing=path,
        ),
        offered_count=len(arrival_ms),
        admitted_count=len(completed),
        rejected_count=rejected_count,
        completed_count=len(completed),
        deadline_met_count=deadline_met_count,
        deadline_missed_count=len(completed) - deadline_met_count,
        useful_completion_count=useful_count,
        useful_completion_fraction=(useful_count / len(arrival_ms)) if arrival_ms else 0.0,
        preprocessing_parity=parity,
        p50_latency=_quantile_quantity(latencies_ms, 0.50),
        p99_latency=_quantile_quantity(latencies_ms, 0.99),
        mean_latency=Q_(sum(latencies_ms) / len(latencies_ms), "ms") if latencies_ms else None,
        max_queue_depth=max_queue_depth,
        busy_replica_time=Q_(busy_replica_ms, "ms"),
        provisioned_replica_time=Q_(provisioned_ms, "ms"),
        records=final_records,
        batches=tuple(batches),
    )


def replay_inputs_to_snapshot(inputs: ServingReplayInputs) -> dict[str, object]:
    """Convert exact replay arguments to a JSON-compatible evidence payload."""

    scenario = inputs.scenario
    return {
        REPLAY_METHOD: {
            "scenario": {
                "track": scenario.track,
                "label": scenario.label,
                "workload": scenario.workload,
                "serving_role": scenario.serving_role,
                "deadline_ms": _milliseconds(scenario.deadline, "deadline"),
                "ingress_latency_ms": _milliseconds(scenario.ingress_latency, "ingress latency"),
                "execution_latency_ms": _milliseconds(scenario.execution_latency, "execution latency"),
                "postprocess_latency_ms": _milliseconds(scenario.postprocess_latency, "postprocess latency"),
                "batch_marginal_cost": scenario.batch_marginal_cost,
                "device_memory_bytes": _bytes(scenario.device_memory, "device memory"),
                "resident_model_memory_bytes": _bytes(
                    scenario.resident_model_memory, "resident model memory"
                ),
                "request_working_state_bytes": _bytes(
                    scenario.request_working_state, "request working state"
                ),
                "request_kv_state_bytes": _bytes(scenario.request_kv_state, "request KV state"),
                "batch_workspace_bytes": _bytes(scenario.batch_workspace, "batch workspace"),
                "reference_preprocessing_signature": scenario.reference_preprocessing_signature,
                "preprocessing_paths": [
                    {
                        "name": path.name,
                        "latency_ms": _milliseconds(path.latency, "preprocessing latency"),
                        "output_signature": path.output_signature,
                    }
                    for path in scenario.preprocessing_paths
                ],
                "assumption_note": scenario.assumption_note,
            },
            "arrivals_ms": [_milliseconds(value, "arrival") for value in inputs.arrivals],
            "policy": {
                "name": inputs.policy.name,
                "batch_limit": inputs.policy.batch_limit,
                "batch_timeout_ms": _milliseconds(inputs.policy.batch_timeout, "batch timeout"),
                "reserved_replicas": inputs.policy.reserved_replicas,
                "on_demand_replicas": inputs.policy.on_demand_replicas,
                "activation_start_ms": _milliseconds(inputs.policy.activation_start, "activation start"),
                "activation_delay_ms": _milliseconds(inputs.policy.activation_delay, "activation delay"),
                "queue_capacity": inputs.policy.queue_capacity,
            },
            "preprocessing": {
                "name": inputs.preprocessing.name,
                "latency_ms": _milliseconds(inputs.preprocessing.latency, "preprocessing latency"),
                "output_signature": inputs.preprocessing.output_signature,
            },
        }
    }


def latency_stage_summary(result: ServingSimulationResult) -> dict[str, float]:
    """Return mean complete-path stage contributions in milliseconds."""

    completed = [record for record in result.records if record.completion is not None]
    if not completed:
        return {
            "ingress_ms": 0.0,
            "preprocessing_ms": 0.0,
            "formation_wait_ms": 0.0,
            "resource_queue_wait_ms": 0.0,
            "execution_ms": 0.0,
            "postprocess_ms": 0.0,
        }
    batch_execution = {
        request_id: batch.execution_latency.to("ms").magnitude
        for batch in result.batches
        for request_id in batch.request_ids
    }
    count = len(completed)
    return {
        "ingress_ms": _milliseconds(result.inputs.scenario.ingress_latency, "ingress latency"),
        "preprocessing_ms": _milliseconds(result.inputs.preprocessing.latency, "preprocessing latency"),
        "formation_wait_ms": sum(record.formation_wait.to("ms").magnitude for record in completed) / count,
        "resource_queue_wait_ms": sum(record.resource_queue_wait.to("ms").magnitude for record in completed) / count,
        "execution_ms": sum(batch_execution[record.request_id] for record in completed) / count,
        "postprocess_ms": _milliseconds(result.inputs.scenario.postprocess_latency, "postprocess latency"),
    }


def simulation_result_to_snapshot(result: ServingSimulationResult) -> dict[str, object]:
    """Convert replay inputs and observable outputs to immutable JSON data."""

    def optional_ms(value: Quantity | None) -> float | None:
        return None if value is None else _milliseconds(value, "result latency")

    return {
        "inputs": replay_inputs_to_snapshot(result.inputs),
        "offered_count": result.offered_count,
        "admitted_count": result.admitted_count,
        "rejected_count": result.rejected_count,
        "completed_count": result.completed_count,
        "deadline_met_count": result.deadline_met_count,
        "deadline_missed_count": result.deadline_missed_count,
        "useful_completion_count": result.useful_completion_count,
        "useful_completion_fraction": result.useful_completion_fraction,
        "preprocessing_parity": result.preprocessing_parity,
        "latency_status": "available" if result.completed_count else "unavailable_no_completed_requests",
        "p50_latency_ms": optional_ms(result.p50_latency),
        "p99_latency_ms": optional_ms(result.p99_latency),
        "mean_latency_ms": optional_ms(result.mean_latency),
        "max_queue_depth": result.max_queue_depth,
        "batch_count": len(result.batches),
        "mean_batch_size": (
            sum(len(batch.request_ids) for batch in result.batches) / len(result.batches)
            if result.batches
            else 0.0
        ),
        "busy_replica_time_ms": _milliseconds(result.busy_replica_time, "busy replica time"),
        "provisioned_replica_time_ms": _milliseconds(
            result.provisioned_replica_time, "provisioned replica time"
        ),
        "mean_stage_latency_ms": latency_stage_summary(result),
    }


def _format_memory(quantity: Quantity) -> str:
    bytes_val = float(quantity.to("byte").magnitude)
    if bytes_val == 0:
        return "0 B"
    if bytes_val < 1024 * 1024:
        return f"{quantity.to('KiB').magnitude:.1f} KiB"
    if bytes_val >= 1024 * 1024 * 1024:
        return f"{quantity.to('GiB').magnitude:.2f} GiB"
    return f"{quantity.to('MiB').magnitude:.1f} MiB"


def state_capacity_to_snapshot(capacity: StateCapacity, *, pressure: StatePressure) -> dict[str, object]:
    """Convert a state-capacity result to JSON evidence data."""

    kv_bytes = _bytes(capacity.kv_state_per_request, "KV state per request")
    return {
        "inputs": {"state_capacity": {"pressure": pressure}},
        "feasible": capacity.feasible,
        "fixed_memory_bytes": _bytes(capacity.fixed_memory, "fixed memory"),
        "fixed_memory_mib": float(capacity.fixed_memory.to("MiB").magnitude),
        "fixed_memory_display": _format_memory(capacity.fixed_memory),
        "state_per_request_bytes": _bytes(capacity.state_per_request, "state per request"),
        "state_per_request_mib": float(capacity.state_per_request.to("MiB").magnitude),
        "state_per_request_display": _format_memory(capacity.state_per_request),
        "kv_state_per_request_bytes": kv_bytes,
        "kv_state_per_request_mib": float(capacity.kv_state_per_request.to("MiB").magnitude),
        "kv_state_per_request_display": _format_memory(capacity.kv_state_per_request) if kv_bytes > 0 else "N/A",
        "max_requests_per_replica": capacity.max_requests_per_replica,
    }


def replay_inputs_from_snapshot(snapshot: Mapping[str, object]) -> ServingReplayInputs:
    """Restore typed replay arguments from a JSON-decoded evidence payload."""

    payload = _mapping(snapshot.get(REPLAY_METHOD), REPLAY_METHOD)
    scenario_data = _mapping(payload.get("scenario"), "scenario")
    paths_data = _sequence(scenario_data.get("preprocessing_paths"), "preprocessing_paths")
    paths = tuple(_preprocessing_from_mapping(_mapping(item, "preprocessing path")) for item in paths_data)
    scenario = ServingScenario(
        track=_track(scenario_data.get("track")),
        label=_string(scenario_data.get("label"), "scenario label"),
        workload=_string(scenario_data.get("workload"), "scenario workload"),
        deadline=Q_(_number(scenario_data.get("deadline_ms"), "deadline_ms"), "ms"),
        ingress_latency=Q_(_number(scenario_data.get("ingress_latency_ms"), "ingress_latency_ms"), "ms"),
        execution_latency=Q_(
            _number(scenario_data.get("execution_latency_ms"), "execution_latency_ms"), "ms"
        ),
        postprocess_latency=Q_(
            _number(scenario_data.get("postprocess_latency_ms"), "postprocess_latency_ms"), "ms"
        ),
        batch_marginal_cost=_number(scenario_data.get("batch_marginal_cost"), "batch_marginal_cost"),
        device_memory=Q_(_number(scenario_data.get("device_memory_bytes"), "device_memory_bytes"), "byte"),
        resident_model_memory=Q_(
            _number(scenario_data.get("resident_model_memory_bytes"), "resident_model_memory_bytes"), "byte"
        ),
        request_working_state=Q_(
            _number(scenario_data.get("request_working_state_bytes"), "request_working_state_bytes"), "byte"
        ),
        request_kv_state=Q_(
            _number(scenario_data.get("request_kv_state_bytes"), "request_kv_state_bytes"), "byte"
        ),
        batch_workspace=Q_(
            _number(scenario_data.get("batch_workspace_bytes"), "batch_workspace_bytes"), "byte"
        ),
        reference_preprocessing_signature=_string(
            scenario_data.get("reference_preprocessing_signature"), "reference preprocessing signature"
        ),
        preprocessing_paths=paths,
        assumption_note=_string(scenario_data.get("assumption_note"), "assumption note"),
        serving_role=_string(scenario_data.get("serving_role", "single-system serving"), "serving_role"),
    )
    policy_data = _mapping(payload.get("policy"), "policy")
    policy = ServingPolicy(
        name=_string(policy_data.get("name"), "policy name"),
        batch_limit=_integer(policy_data.get("batch_limit"), "batch_limit"),
        batch_timeout=Q_(_number(policy_data.get("batch_timeout_ms"), "batch_timeout_ms"), "ms"),
        reserved_replicas=_integer(policy_data.get("reserved_replicas"), "reserved_replicas"),
        on_demand_replicas=_integer(policy_data.get("on_demand_replicas"), "on_demand_replicas"),
        activation_start=Q_(_number(policy_data.get("activation_start_ms"), "activation_start_ms"), "ms"),
        activation_delay=Q_(_number(policy_data.get("activation_delay_ms"), "activation_delay_ms"), "ms"),
        queue_capacity=_integer(policy_data.get("queue_capacity"), "queue_capacity"),
    )
    preprocessing = _preprocessing_from_mapping(_mapping(payload.get("preprocessing"), "preprocessing"))
    arrivals_data = _sequence(payload.get("arrivals_ms"), "arrivals_ms")
    arrivals = tuple(Q_(_number(value, "arrival"), "ms") for value in arrivals_data)
    return ServingReplayInputs(
        scenario=scenario,
        arrivals=arrivals,
        policy=policy,
        preprocessing=preprocessing,
    )


def replay_serving_snapshot(snapshot: Mapping[str, object]) -> ServingSimulationResult:
    """Restore and evaluate a captured ``simulate_serving`` input payload."""

    inputs = replay_inputs_from_snapshot(snapshot)
    return simulate_serving(inputs.scenario, inputs.arrivals, inputs.policy, inputs.preprocessing)


def _admit_ready_requests(
    through_ms: float,
    ready_ms: Sequence[float],
    arrival_ms: Sequence[float],
    cursor: int,
    pending: list[int],
    records: list[RequestRecord | None],
    queue_capacity: int,
) -> int:
    while cursor < len(ready_ms) and ready_ms[cursor] <= through_ms + 1e-9:
        if len(pending) < queue_capacity:
            pending.append(cursor)
        else:
            records[cursor] = _rejected_record(cursor, arrival_ms[cursor], ready_ms[cursor], "admission queue full")
        cursor += 1
    return cursor


def _rejected_record(request_id: int, arrival_ms: float, ready_ms: float, reason: str) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        arrival=Q_(arrival_ms, "ms"),
        ready=Q_(ready_ms, "ms"),
        dispatch=None,
        completion=None,
        formation_wait=None,
        resource_queue_wait=None,
        total_latency=None,
        admitted=False,
        deadline_met=False,
        useful=False,
        rejection_reason=reason,
    )


def _quantile_quantity(values_ms: Sequence[float], quantile: float) -> Quantity | None:
    if not values_ms:
        return None
    ordered = sorted(values_ms)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return Q_(ordered[rank - 1], "ms")


def _milliseconds(value: Quantity, name: str) -> float:
    try:
        magnitude = float(value.to("ms").magnitude)
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"{name} must be a Pint time quantity") from exc
    if not math.isfinite(magnitude):
        raise ValueError(f"{name} must be finite")
    return magnitude


def _bytes(value: Quantity, name: str) -> float:
    try:
        magnitude = float(value.to("byte").magnitude)
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"{name} must be a Pint memory quantity") from exc
    if not math.isfinite(magnitude) or magnitude < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return magnitude


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: object, name: str) -> int:
    number = _number(value, name)
    integer = int(number)
    if integer != number:
        raise ValueError(f"{name} must be an integer")
    return integer


def _track(value: object) -> Track:
    track = _string(value, "track")
    if track not in _SCENARIOS:
        raise ValueError(f"unknown serving track: {track!r}")
    return track  # type: ignore[return-value]


def _preprocessing_from_mapping(data: Mapping[str, object]) -> PreprocessingPath:
    return PreprocessingPath(
        name=_string(data.get("name"), "preprocessing name"),
        latency=Q_(_number(data.get("latency_ms"), "preprocessing latency_ms"), "ms"),
        output_signature=_string(data.get("output_signature"), "preprocessing output_signature"),
    )


def _validate_policy(policy: ServingPolicy) -> None:
    if policy.batch_limit < 1:
        raise ValueError("batch_limit must be at least 1")
    if policy.reserved_replicas < 1:
        raise ValueError("reserved_replicas must be at least 1")
    if policy.on_demand_replicas < 0:
        raise ValueError("on_demand_replicas must be nonnegative")
    if policy.queue_capacity < 1:
        raise ValueError("queue_capacity must be at least 1")


__all__ = [
    "ArrivalPhase",
    "BatchRecord",
    "MODEL_ID",
    "REPLAY_METHOD",
    "PreprocessingPath",
    "RequestRecord",
    "ServingPolicy",
    "ServingReplayInputs",
    "ServingScenario",
    "ServingSimulationResult",
    "StateCapacity",
    "arrival_trace_for_track",
    "batching_policy_for_track",
    "get_serving_scenario",
    "make_arrival_trace",
    "policy_for_track",
    "preprocessing_path",
    "replay_inputs_from_snapshot",
    "replay_inputs_to_snapshot",
    "replay_serving_snapshot",
    "simulation_result_to_snapshot",
    "simulate_serving",
    "state_capacity",
    "state_capacity_to_snapshot",
    "state_pressure_scenario",
]
