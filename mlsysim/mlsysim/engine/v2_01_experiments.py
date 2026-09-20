"""Experiments for introducing fleet-scale machine-learning systems.

The models in this module deliberately separate three systems that are easy to
conflate in an introductory fleet lab:

* one machine that must fit and sustain its own workload;
* a coupled job whose progress depends on every participating worker; and
* independently operating deployed devices whose memory and failures do not
  combine into one virtual machine or one interrupted job.

Scenario inputs are illustrative. Results are analytical predictions, not
measurements of a production fleet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Literal, Mapping

from ..core.units import Q_, ureg
from ..physics.reliability import calc_failure_probability, calc_mtbf_cluster


DeploymentCoupling = Literal["independent_devices", "coupled_job"]
PartitionPolicy = Literal["wait_for_fresh", "serve_last_confirmed"]


@dataclass(frozen=True)
class TrackProfile:
    """Semantic roles used by one selectable teaching track."""

    track_id: str
    label: str
    deployed_system: str
    deployed_coupling: DeploymentCoupling
    scaling_system: str
    scaling_coupling: Literal["coupled_job"] = "coupled_job"


@dataclass(frozen=True)
class TrackScenario:
    """Illustrative physical inputs for all four introductory experiments."""

    profile: TrackProfile
    model_state: object
    compact_model_state: object
    memory_per_machine: object
    memory_display_unit: str
    required_rate: object
    rate_per_machine: object
    machine_count_options: tuple[int, ...]
    worker_options: tuple[int, ...]
    single_worker_compute_time: object
    communication_payload: object
    bandwidth_options: Mapping[str, object]
    communication_startup: object
    coordination_base: object
    coordination_options: Mapping[str, object]
    useful_work_per_step: object
    overlap_fraction: float
    fleet_size_options: tuple[int, ...]
    component_mtbf: object
    observation_horizon: object
    recovery_options: Mapping[str, object]
    request_rate: object
    partition_duration_options: Mapping[str, object]
    partitioned_fraction_options: Mapping[str, float]
    last_confirmed_model_age: object


TRACK_PROFILES = {
    "tinyml": TrackProfile(
        track_id="tinyml",
        label="TinyML",
        deployed_system="independent sensor devices",
        deployed_coupling="independent_devices",
        scaling_system="shared backend training job",
    ),
    "mobile": TrackProfile(
        track_id="mobile",
        label="Mobile",
        deployed_system="independent phones in a rollout",
        deployed_coupling="independent_devices",
        scaling_system="regional backend training job",
    ),
    "edge": TrackProfile(
        track_id="edge",
        label="Edge",
        deployed_system="independent device-gateway service cells",
        deployed_coupling="independent_devices",
        scaling_system="regional backend training job",
    ),
    "cloud": TrackProfile(
        track_id="cloud",
        label="Cloud",
        deployed_system="synchronized training workers",
        deployed_coupling="coupled_job",
        scaling_system="synchronized training job",
    ),
}


TRACK_SCENARIOS = {
    "tinyml": TrackScenario(
        profile=TRACK_PROFILES["tinyml"],
        model_state=Q_(768, "KiB"), compact_model_state=Q_(256, "KiB"),
        memory_per_machine=Q_(512, "KiB"), memory_display_unit="KiB",
        required_rate=Q_(10, "count/second"),
        rate_per_machine=Q_(20, "count/second"), machine_count_options=(1, 4, 16),
        worker_options=(1, 8, 64, 512), single_worker_compute_time=Q_(48, "second"),
        communication_payload=Q_(600, "MB"),
        bandwidth_options={"constrained": Q_(1, "GB/second"), "planned": Q_(4, "GB/second"), "upgraded": Q_(12, "GB/second")},
        communication_startup=Q_(3, "millisecond"), coordination_base=Q_(20, "millisecond"),
        coordination_options={"tight": Q_(1, "millisecond"), "planned": Q_(8, "millisecond"), "stressed": Q_(30, "millisecond")},
        useful_work_per_step=Q_(4096, "count"), overlap_fraction=0.25,
        fleet_size_options=(1_000, 100_000, 1_000_000), component_mtbf=Q_(800_000, "hour"),
        observation_horizon=Q_(24, "hour"),
        recovery_options={"automated": Q_(5, "minute"), "planned": Q_(45, "minute"), "manual": Q_(8, "hour")},
        request_rate=Q_(20, "count/second"),
        partition_duration_options={"brief": Q_(1, "minute"), "extended": Q_(30, "minute"), "overnight": Q_(8, "hour")},
        partitioned_fraction_options={"small cohort": 0.05, "regional cohort": 0.25, "broad cohort": 0.60},
        last_confirmed_model_age=Q_(2, "hour"),
    ),
    "mobile": TrackScenario(
        profile=TRACK_PROFILES["mobile"],
        model_state=Q_(8, "GiB"), compact_model_state=Q_(4, "GiB"),
        memory_per_machine=Q_(6, "GiB"), memory_display_unit="GiB",
        required_rate=Q_(12, "count/second"),
        rate_per_machine=Q_(20, "count/second"), machine_count_options=(1, 2, 8),
        worker_options=(1, 16, 128, 1024), single_worker_compute_time=Q_(96, "second"),
        communication_payload=Q_(4, "GB"),
        bandwidth_options={"constrained": Q_(2, "GB/second"), "planned": Q_(10, "GB/second"), "upgraded": Q_(25, "GB/second")},
        communication_startup=Q_(2, "millisecond"), coordination_base=Q_(15, "millisecond"),
        coordination_options={"tight": Q_(1, "millisecond"), "planned": Q_(6, "millisecond"), "stressed": Q_(20, "millisecond")},
        useful_work_per_step=Q_(8192, "count"), overlap_fraction=0.30,
        fleet_size_options=(10_000, 1_000_000, 5_000_000), component_mtbf=Q_(1_500_000, "hour"),
        observation_horizon=Q_(24, "hour"),
        recovery_options={"automated": Q_(5, "minute"), "planned": Q_(30, "minute"), "manual": Q_(4, "hour")},
        request_rate=Q_(200, "count/second"),
        partition_duration_options={"brief": Q_(30, "second"), "extended": Q_(10, "minute"), "commute": Q_(1, "hour")},
        partitioned_fraction_options={"small cohort": 0.02, "regional cohort": 0.15, "broad cohort": 0.50},
        last_confirmed_model_age=Q_(15, "minute"),
    ),
    "edge": TrackScenario(
        profile=TRACK_PROFILES["edge"],
        model_state=Q_(20, "GiB"), compact_model_state=Q_(12, "GiB"),
        memory_per_machine=Q_(16, "GiB"), memory_display_unit="GiB",
        required_rate=Q_(30, "count/second"),
        rate_per_machine=Q_(18, "count/second"), machine_count_options=(1, 2, 4),
        worker_options=(1, 8, 64, 512), single_worker_compute_time=Q_(72, "second"),
        communication_payload=Q_(8, "GB"),
        bandwidth_options={"constrained": Q_(1, "GB/second"), "planned": Q_(8, "GB/second"), "upgraded": Q_(20, "GB/second")},
        communication_startup=Q_(4, "millisecond"), coordination_base=Q_(25, "millisecond"),
        coordination_options={"tight": Q_(2, "millisecond"), "planned": Q_(10, "millisecond"), "stressed": Q_(35, "millisecond")},
        useful_work_per_step=Q_(4096, "count"), overlap_fraction=0.20,
        fleet_size_options=(100, 10_000, 100_000), component_mtbf=Q_(100_000, "hour"),
        observation_horizon=Q_(24, "hour"),
        recovery_options={"automated": Q_(2, "minute"), "planned": Q_(20, "minute"), "manual": Q_(3, "hour")},
        request_rate=Q_(80, "count/second"),
        partition_duration_options={"brief": Q_(10, "second"), "extended": Q_(5, "minute"), "site outage": Q_(2, "hour")},
        partitioned_fraction_options={"one cell": 0.02, "regional cohort": 0.20, "broad cohort": 0.70},
        last_confirmed_model_age=Q_(5, "minute"),
    ),
    "cloud": TrackScenario(
        profile=TRACK_PROFILES["cloud"],
        model_state=Q_(180, "GiB"), compact_model_state=Q_(70, "GiB"),
        memory_per_machine=Q_(80, "GiB"), memory_display_unit="GiB",
        required_rate=Q_(120, "count/second"),
        rate_per_machine=Q_(50, "count/second"), machine_count_options=(1, 2, 3, 8),
        worker_options=(1, 64, 1024, 8192), single_worker_compute_time=Q_(240, "second"),
        communication_payload=Q_(350, "GB"),
        bandwidth_options={"constrained": Q_(12.5, "GB/second"), "planned": Q_(50, "GB/second"), "upgraded": Q_(100, "GB/second")},
        communication_startup=Q_(1, "millisecond"), coordination_base=Q_(10, "millisecond"),
        coordination_options={"tight": Q_(0.2, "millisecond"), "planned": Q_(2, "millisecond"), "stressed": Q_(10, "millisecond")},
        useful_work_per_step=Q_(16384, "count"), overlap_fraction=0.45,
        fleet_size_options=(64, 8192, 32768), component_mtbf=Q_(50_000, "hour"),
        observation_horizon=Q_(8, "hour"),
        recovery_options={"automated": Q_(2, "minute"), "planned": Q_(15, "minute"), "manual": Q_(2, "hour")},
        request_rate=Q_(10_000, "count/second"),
        partition_duration_options={"brief": Q_(5, "second"), "extended": Q_(2, "minute"), "regional": Q_(30, "minute")},
        partitioned_fraction_options={"one zone": 0.10, "regional cohort": 0.35, "broad cohort": 0.70},
        last_confirmed_model_age=Q_(1, "minute"),
    ),
}


@dataclass(frozen=True)
class MachineBoundaryResult:
    """Capacity result for one machine before fleet composition."""

    coupling: DeploymentCoupling
    machine_count: int
    model_state: object
    memory_per_machine: object
    memory_margin: object
    required_rate: object
    rate_per_machine: object
    local_memory_fits: bool
    local_rate_meets_demand: bool
    machines_for_memory: int | None
    machines_for_rate: int | None
    minimum_coupled_machines: int | None
    distribution_required: bool
    feasible: bool
    remedy: str

    @property
    def effective_state_per_machine(self) -> object:
        if self.coupling == "coupled_job":
            return self.model_state / self.machine_count
        return self.model_state

    @property
    def effective_required_rate_per_machine(self) -> object:
        if self.coupling == "coupled_job":
            return self.required_rate / self.machine_count
        return self.required_rate

    @property
    def effective_memory_margin(self) -> object:
        return self.memory_per_machine - self.effective_state_per_machine

@dataclass(frozen=True)
class C3ScalingResult:
    """Compute, communication, and coordination terms for a coupled job."""

    workers: int
    compute_time: object
    communication_time: object
    coordination_time: object
    overlapped_time: object
    step_time: object
    useful_work_per_step: object
    useful_throughput: object
    ideal_throughput: object
    scaling_efficiency: float
    compute_fraction: float
    communication_fraction: float
    coordination_fraction: float
    dominant_term: str


@dataclass(frozen=True)
class ReliabilityComparisonResult:
    """Failure consequences over a horizon plus repair-time availability."""

    nodes: int
    observation_horizon: object
    component_mtbf: object
    recovery_time: object
    per_node_event_probability: float
    probability_any_node_event: float
    expected_node_failure_events: float
    coupled_job_interruption_probability: float
    coupled_expected_interruptions: float
    coupled_expected_recovery_time: object
    independent_expected_affected_devices: float
    independent_expected_unavailable_device_time: object
    independent_mean_unavailable_devices: float
    independent_available_capacity_fraction: float
    independent_unavailable_capacity_fraction: float


@dataclass(frozen=True)
class PartitionResult:
    """Request consequences of a partition policy over a stated horizon."""

    policy: PartitionPolicy
    observation_horizon: object
    partition_duration: object
    total_requests: object
    affected_requests: object
    fresh_requests: object
    stale_requests: object
    unavailable_requests: object
    maximum_model_age: object


def get_track_profile(track_id: str) -> TrackProfile:
    """Return one of the four semantically distinct teaching tracks."""

    try:
        return TRACK_PROFILES[track_id]
    except KeyError as exc:
        choices = ", ".join(TRACK_PROFILES)
        raise ValueError(f"unknown track_id {track_id!r}; choose one of: {choices}") from exc


def get_track_scenario(track_id: str) -> TrackScenario:
    """Return the illustrative physical inputs for one teaching track."""

    get_track_profile(track_id)
    return TRACK_SCENARIOS[track_id]


def serialize_value(value: object):
    """Convert experiment data to JSON-safe values without discarding units."""

    if isinstance(value, ureg.Quantity):
        magnitude = value.magnitude
        if not isinstance(magnitude, (int, float)):
            raise TypeError("only scalar Pint quantities can be serialized")
        if not math.isfinite(float(magnitude)):
            raise ValueError("quantity magnitude must be finite")
        return {
            "__type__": "quantity",
            "magnitude": magnitude,
            "unit": f"{value.units:~}",
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: serialize_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize experiment value of type {type(value).__name__}")


def restore_quantities(value: object):
    """Restore quantities inside data produced by :func:`serialize_value`."""

    if isinstance(value, Mapping):
        if value.get("__type__") == "quantity":
            if set(value) != {"__type__", "magnitude", "unit"}:
                raise ValueError("serialized quantity has unexpected fields")
            return Q_(value["magnitude"], value["unit"])
        return {str(key): restore_quantities(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_quantities(item) for item in value]
    return value


def serialize_evaluation(*, inputs: Mapping[str, object], result: object) -> dict[str, object]:
    """Package exact evaluator kwargs and outputs for immutable evidence capture."""

    if not inputs:
        raise ValueError("inputs must contain the exact evaluator arguments")
    return {
        "inputs": serialize_value(inputs),
        "outputs": serialize_value(result),
    }


def _quantity(value: object, unit: object, name: str, *, allow_zero: bool = False):
    if not isinstance(value, ureg.Quantity):
        raise TypeError(f"{name} must be a Pint Quantity")
    try:
        converted = value.to(unit)
    except Exception as exc:
        raise ValueError(f"{name} must have units compatible with {unit}") from exc
    magnitude = converted.magnitude
    if not math.isfinite(float(magnitude)):
        raise ValueError(f"{name} must be finite")
    if magnitude < 0 or (magnitude == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return converted


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1")
    return value


def evaluate_machine_boundary(
    *,
    model_state: object,
    memory_per_machine: object,
    required_rate: object,
    rate_per_machine: object,
    coupling: DeploymentCoupling,
    machine_count: int = 1,
) -> MachineBoundaryResult:
    """Find the one-machine memory or demand boundary.

    Coupled jobs may shard state and aggregate rate across machines. Independent
    deployed devices must each fit and serve their local workload; increasing
    ``machine_count`` therefore cannot repair a per-device memory or rate miss.
    """

    state = _quantity(model_state, ureg.byte, "model_state")
    memory = _quantity(memory_per_machine, ureg.byte, "memory_per_machine")
    demand = _quantity(required_rate, ureg.count / ureg.second, "required_rate")
    rate = _quantity(rate_per_machine, ureg.count / ureg.second, "rate_per_machine")
    count = _positive_int(machine_count, "machine_count")
    if coupling not in ("independent_devices", "coupled_job"):
        raise ValueError("coupling must be 'independent_devices' or 'coupled_job'")

    local_memory_fits = state <= memory
    local_rate_meets_demand = rate >= demand
    memory_ratio = (state / memory).to_base_units().magnitude
    rate_ratio = (demand / rate).to_base_units().magnitude
    machines_for_memory = max(1, math.ceil(memory_ratio))
    machines_for_rate = max(1, math.ceil(rate_ratio))

    if coupling == "coupled_job":
        minimum = max(machines_for_memory, machines_for_rate)
        feasible = count >= minimum
        distribution_required = minimum > 1
        remedy = "add coupled machines" if not feasible else "within coupled-machine capacity"
        reported_memory_machines: int | None = machines_for_memory
        reported_rate_machines: int | None = machines_for_rate
    else:
        minimum = None
        feasible = local_memory_fits and local_rate_meets_demand
        distribution_required = False
        remedy = (
            "reduce per-device state or demand"
            if not feasible
            else "within each independent device's capacity"
        )
        reported_memory_machines = None
        reported_rate_machines = None

    return MachineBoundaryResult(
        coupling=coupling,
        machine_count=count,
        model_state=state,
        memory_per_machine=memory,
        memory_margin=(memory - state).to(ureg.byte),
        required_rate=demand,
        rate_per_machine=rate,
        local_memory_fits=local_memory_fits,
        local_rate_meets_demand=local_rate_meets_demand,
        machines_for_memory=reported_memory_machines,
        machines_for_rate=reported_rate_machines,
        minimum_coupled_machines=minimum,
        distribution_required=distribution_required,
        feasible=feasible,
        remedy=remedy,
    )


def evaluate_c3_scaling(
    *,
    workers: int,
    single_worker_compute_time: object,
    communication_payload: object,
    effective_bandwidth: object,
    communication_startup: object,
    coordination_base: object,
    coordination_per_added_worker: object,
    useful_work_per_step: object,
    overlap_fraction: float = 0.0,
) -> C3ScalingResult:
    """Decompose one fixed-work coupled step into explicit C³ terms.

    This is a deliberately small strong-scaling model. Communication is one
    disclosed payload transfer per step; coordination grows with the number of
    additional workers. It does not select or optimize a collective algorithm.
    """

    n = _positive_int(workers, "workers")
    compute_one = _quantity(single_worker_compute_time, ureg.second, "single_worker_compute_time")
    payload = _quantity(communication_payload, ureg.byte, "communication_payload", allow_zero=True)
    bandwidth = _quantity(effective_bandwidth, ureg.byte / ureg.second, "effective_bandwidth")
    startup = _quantity(communication_startup, ureg.second, "communication_startup", allow_zero=True)
    coord_base = _quantity(coordination_base, ureg.second, "coordination_base", allow_zero=True)
    coord_worker = _quantity(
        coordination_per_added_worker,
        ureg.second,
        "coordination_per_added_worker",
        allow_zero=True,
    )
    work = _quantity(useful_work_per_step, ureg.count, "useful_work_per_step")
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be between 0 and 1")

    compute = (compute_one / n).to(ureg.second)
    if n == 1:
        communication = Q_(0, "second")
        coordination = Q_(0, "second")
    else:
        communication = (startup + payload / bandwidth).to(ureg.second)
        coordination = (coord_base + (n - 1) * coord_worker).to(ureg.second)
    overlapped = (min(compute, communication) * overlap_fraction).to(ureg.second)
    step = (compute + communication + coordination - overlapped).to(ureg.second)
    throughput = (work / step).to(ureg.count / ureg.second)
    ideal = (n * work / compute_one).to(ureg.count / ureg.second)
    efficiency = (throughput / ideal).to_base_units().magnitude

    visible_compute = compute - overlapped
    shares = {
        "compute": (visible_compute / step).to_base_units().magnitude,
        "communication": (communication / step).to_base_units().magnitude,
        "coordination": (coordination / step).to_base_units().magnitude,
    }
    dominant = max(shares, key=shares.get)
    return C3ScalingResult(
        workers=n,
        compute_time=compute,
        communication_time=communication,
        coordination_time=coordination,
        overlapped_time=overlapped,
        step_time=step,
        useful_work_per_step=work,
        useful_throughput=throughput,
        ideal_throughput=ideal,
        scaling_efficiency=efficiency,
        compute_fraction=shares["compute"],
        communication_fraction=shares["communication"],
        coordination_fraction=shares["coordination"],
        dominant_term=dominant,
    )


def compare_failure_semantics(
    *,
    nodes: int,
    component_mtbf: object,
    observation_horizon: object,
    recovery_time: object,
) -> ReliabilityComparisonResult:
    """Compare the same independent node failures under two workload semantics.

    Event probabilities answer whether a failure occurs during the explicit
    observation horizon, starting with all nodes operational. Independent-fleet
    capacity uses the stationary availability of an alternating exponential
    up/repair process, ``MTBF / (MTBF + repair time)``. A node event interrupts
    the coupled job; the same event only removes that node's capacity from an
    independent deployment.
    """

    n = _positive_int(nodes, "nodes")
    mtbf = _quantity(component_mtbf, ureg.hour, "component_mtbf")
    horizon = _quantity(observation_horizon, ureg.hour, "observation_horizon")
    recovery = _quantity(recovery_time, ureg.hour, "recovery_time", allow_zero=True)

    per_node_probability = calc_failure_probability(mtbf, horizon)
    fleet_mtbf = calc_mtbf_cluster(mtbf, n)
    any_probability = calc_failure_probability(fleet_mtbf, horizon)
    expected_events = (n * horizon / mtbf).to_base_units().magnitude
    expected_recovery = (expected_events * recovery).to(ureg.hour)
    steady_state_availability = (
        mtbf / (mtbf + recovery)
    ).to_base_units().magnitude
    mean_unavailable = n * (1.0 - steady_state_availability)
    unavailable_device_time = (mean_unavailable * horizon).to(ureg.hour)

    return ReliabilityComparisonResult(
        nodes=n,
        observation_horizon=horizon,
        component_mtbf=mtbf,
        recovery_time=recovery,
        per_node_event_probability=per_node_probability,
        probability_any_node_event=any_probability,
        expected_node_failure_events=expected_events,
        coupled_job_interruption_probability=any_probability,
        coupled_expected_interruptions=expected_events,
        coupled_expected_recovery_time=expected_recovery,
        independent_expected_affected_devices=n * per_node_probability,
        independent_expected_unavailable_device_time=unavailable_device_time,
        independent_mean_unavailable_devices=mean_unavailable,
        independent_available_capacity_fraction=steady_state_availability,
        independent_unavailable_capacity_fraction=1.0 - steady_state_availability,
    )


def evaluate_partition_policy(
    *,
    policy: PartitionPolicy,
    request_rate: object,
    observation_horizon: object,
    partition_duration: object,
    partitioned_fraction: float,
    last_confirmed_model_age: object = Q_(0, "second"),
) -> PartitionResult:
    """Count unavailable or stale requests caused by one partition.

    ``wait_for_fresh`` refuses affected requests until fresh state is reachable.
    ``serve_last_confirmed`` keeps those requests available but serves the last
    confirmed model, whose maximum age grows for the partition duration.
    """

    if policy not in ("wait_for_fresh", "serve_last_confirmed"):
        raise ValueError("policy must be 'wait_for_fresh' or 'serve_last_confirmed'")
    rate = _quantity(request_rate, ureg.count / ureg.second, "request_rate")
    horizon = _quantity(observation_horizon, ureg.second, "observation_horizon")
    duration = _quantity(partition_duration, ureg.second, "partition_duration", allow_zero=True)
    age = _quantity(last_confirmed_model_age, ureg.second, "last_confirmed_model_age", allow_zero=True)
    if duration > horizon:
        raise ValueError("partition_duration cannot exceed observation_horizon")
    if not 0.0 <= partitioned_fraction <= 1.0:
        raise ValueError("partitioned_fraction must be between 0 and 1")

    total = (rate * horizon).to(ureg.count)
    affected = (rate * duration * partitioned_fraction).to(ureg.count)
    if policy == "wait_for_fresh":
        unavailable = affected
        stale = Q_(0, "count")
        fresh = total - unavailable
        maximum_age = age
    else:
        unavailable = Q_(0, "count")
        stale = affected
        fresh = total - stale
        maximum_age = (age + duration).to(ureg.second)

    return PartitionResult(
        policy=policy,
        observation_horizon=horizon,
        partition_duration=duration,
        total_requests=total,
        affected_requests=affected,
        fresh_requests=fresh.to(ureg.count),
        stale_requests=stale,
        unavailable_requests=unavailable,
        maximum_model_age=maximum_age,
    )


__all__ = [
    "C3ScalingResult",
    "MachineBoundaryResult",
    "PartitionResult",
    "ReliabilityComparisonResult",
    "TRACK_PROFILES",
    "TRACK_SCENARIOS",
    "TrackProfile",
    "TrackScenario",
    "compare_failure_semantics",
    "evaluate_c3_scaling",
    "evaluate_machine_boundary",
    "evaluate_partition_policy",
    "get_track_profile",
    "get_track_scenario",
    "restore_quantities",
    "serialize_evaluation",
    "serialize_value",
]
