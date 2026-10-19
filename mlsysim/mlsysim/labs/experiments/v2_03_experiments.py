"""Chapter 3 network-fabric experiments.

The module models finite flows over explicit shared fabric lanes.  It is an
educational model rather than a packet-level simulator: flows are scheduled
FIFO on each lane, and each flow pays startup latency plus serialization time.
This is sufficient to expose the causal effects used by the lab without
inventing a tail-latency multiplier.

All non-registry values below are clearly identified as illustrative scenario
assumptions.  Device-oriented tracks still model a fleet: phones, wearables,
and vehicles communicate through regional gateways rather than pretending to
participate in accelerator collectives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Iterable, Mapping

from mlsysim.core.units import Q_, ureg
from mlsysim.hardware.registry import Hardware
from mlsysim.physics.communication import (
    calc_alpha_beta_crossover,
    calc_point_to_point_time,
)
from mlsysim.systems.registry import Systems


_TRACK_ALIASES = {
    "cloud": "cloud",
    "cloud_fleet": "cloud",
    "mobile": "mobile",
    "iphone": "mobile",
    "tinyml": "tinyml",
    "oura_ring": "tinyml",
    "edge": "edge",
    "robotaxi": "edge",
}

TRAFFIC_PATTERN_OPTIONS = {
    "Cross-fabric exchange": "all_to_all",
    "Synchronized fan-in": "incast",
    "Neighbor exchange": "neighbor",
    "Regional fan-in": "regional_fan_in",
    "Staged sync": "staged_sync",
    "Local then staged": "local_then_staged",
}

TRAFFIC_PATTERN_LABELS = {v: k for k, v in TRAFFIC_PATTERN_OPTIONS.items()}

OVERLAP_WINDOW_OPTIONS = {
    "None": "0_ms",
    "5 ms": "5_ms",
    "12 ms": "12_ms",
    "20 ms": "20_ms",
    "35 ms": "35_ms",
    "100 ms": "100_ms",
    "1 s": "1000_ms",
    "30 s": "30000_ms",
}

OVERLAP_WINDOW_QUANTITIES = {
    "0_ms": Q_(0, "ms"),
    "5_ms": Q_(5, "ms"),
    "12_ms": Q_(12, "ms"),
    "20_ms": Q_(20, "ms"),
    "35_ms": Q_(35, "ms"),
    "100_ms": Q_(100, "ms"),
    "1000_ms": Q_(1, "s"),
    "30000_ms": Q_(30, "s"),
}

PAYLOAD_OPTIONS = {
    "1 byte": "1_byte",
    "1 KB": "1_kb",
    "64 KB": "64_kb",
    "1 MB": "1_mb",
    "6 MB": "6_mb",
    "8 MB": "8_mb",
    "96 MB": "96_mb",
    "350 MB": "350_mb",
    "1 GB": "1_gb",
}

PAYLOAD_QUANTITIES = {
    "1_byte": Q_(1, "byte"), "1_kb": Q_(1, "KB"), "64_kb": Q_(64, "KB"),
    "1_mb": Q_(1, "MB"), "6_mb": Q_(6, "MB"), "8_mb": Q_(8, "MB"),
    "96_mb": Q_(96, "MB"), "350_mb": Q_(350, "MB"), "1_gb": Q_(1, "GB"),
}

@dataclass(frozen=True)
class LinkProfile:
    """A physical path used by a track scenario."""

    link_id: str
    label: str
    alpha: object
    bandwidth: object
    assumption: str


@dataclass(frozen=True)
class TrackScenario:
    """Fleet workload and guardrails for one teaching track."""

    track_id: str
    fleet_unit: str
    payload: object
    participants: int
    communication_budget: object
    utilization_limit: float
    default_link_id: str
    links: Mapping[str, LinkProfile]
    traffic_pattern: str
    overlap_window: object
    failure_mode: str


@dataclass(frozen=True)
class Flow:
    """One finite application transfer in a traffic matrix."""

    flow_id: str
    source: int
    destination: int
    payload: object
    ready_at: object = Q_(0, "ms")
    traffic_class: str = "foreground"


@dataclass(frozen=True)
class Topology:
    """A fabric topology represented by shared lane resources."""

    topology_id: str
    label: str
    endpoints: int
    cut_lanes: int
    local_lanes: int
    groups: int
    hop_count: int
    switches: int
    links: int
    link_cost: object
    switch_cost: object
    assumption: str
    role: str = ""

    @property
    def cost(self):
        return (self.links * self.link_cost + self.switches * self.switch_cost).to(ureg.USD)


@dataclass(frozen=True)
class FlowResult:
    flow_id: str
    lane_id: str
    start_ms: float
    finish_ms: float
    queue_ms: float
    service_ms: float


@dataclass(frozen=True)
class ScheduleResult:
    flows: tuple[FlowResult, ...]
    completion_ms: float
    p50_ms: float
    p95_ms: float
    max_queue_ms: float
    bottleneck_lane: str
    lane_busy_ms: Mapping[str, float]
    lane_bytes: Mapping[str, float]
    effective_bandwidth_mb_s: float


def _ms(value: object) -> float:
    return float(value.to(ureg.millisecond).magnitude)


def _mb(value: object) -> float:
    return float(value.to(ureg.MB).magnitude)


def _mb_s(value: object) -> float:
    return float(value.to(ureg.MB / ureg.second).magnitude)


def serialize_quantity(quantity: object) -> dict[str, object]:
    """Serialize a Pint quantity without discarding its unit."""

    return {
        "value": float(quantity.magnitude),
        "unit": str(quantity.units),
    }


def deserialize_quantity(serialized: Mapping[str, object]):
    """Reconstruct a quantity produced by :func:`serialize_quantity`."""

    if set(serialized) != {"value", "unit"}:
        raise ValueError("serialized quantity requires exactly value and unit")
    return Q_(serialized["value"], str(serialized["unit"]))


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def track_scenarios() -> dict[str, TrackScenario]:
    """Return the four fleet scenarios used by the lab.

    Cloud link facts reuse the MLSysIM registries.  The mobile, wearable, and
    vehicle values are illustrative workload/path assumptions, not hardware
    measurements.
    """

    nvlink = Hardware.Cloud.H100.nvlink
    cloud_links = {
        "ib_ndr": LinkProfile(
            "ib_ndr",
            "InfiniBand NDR",
            Systems.SwitchFabric.AlphaNdr,
            Systems.Fabrics.InfiniBand_NDR.bandwidth,
            "MLSysIM registry value",
        ),
        "roce_100g": LinkProfile(
            "roce_100g",
            "RoCE over 100G Ethernet",
            Systems.SwitchFabric.AlphaRoce,
            Systems.Fabrics.Ethernet_100G.bandwidth,
            "MLSysIM registry value",
        ),
        "nvlink_local": LinkProfile(
            "nvlink_local",
            "NVLink local domain",
            Q_(0.5, "us"),
            nvlink.bandwidth_per_direction,
            "bandwidth from MLSysIM; latency is an illustrative scenario assumption",
        ),
    }
    return {
        "cloud": TrackScenario(
            "cloud",
            "accelerators in a synchronous regional training pool",
            Q_(350, "MB"),
            64,
            Q_(120, "ms"),
            0.80,
            "ib_ndr",
            cloud_links,
            "all_to_all",
            Q_(35, "ms"),
            "step-time breach or idle accelerators",
        ),
        "mobile": TrackScenario(
            "mobile",
            "phones sending summarized telemetry through regional edge gateways",
            Q_(8, "MB"),
            80,
            Q_(180, "ms"),
            0.70,
            "wifi_edge",
            {
                "wifi_edge": LinkProfile(
                    "wifi_edge", "Wi-Fi to edge", Q_(8, "ms"), Q_(90, "MB/s"), "illustrative scenario assumption"
                ),
                "5g_edge": LinkProfile(
                    "5g_edge", "5G to edge", Q_(28, "ms"), Q_(18, "MB/s"), "illustrative scenario assumption"
                ),
                "cloud_wan": LinkProfile(
                    "cloud_wan", "Direct cloud WAN", Q_(70, "ms"), Q_(10, "MB/s"), "illustrative scenario assumption"
                ),
            },
            "regional_fan_in",
            Q_(20, "ms"),
            "responsiveness or regional-ingress backlog miss",
        ),
        "tinyml": TrackScenario(
            "tinyml",
            "wearables syncing through nearby phones and staged cloud gateways",
            Q_(6, "MB"),
            64,
            Q_(180, "s"),
            0.55,
            "ble_phone",
            {
                "ble_phone": LinkProfile(
                    "ble_phone", "BLE ring-to-phone", Q_(90, "ms"), Q_(0.18, "MB/s"), "illustrative scenario assumption"
                ),
                "phone_wifi": LinkProfile(
                    "phone_wifi", "Phone Wi-Fi relay", Q_(35, "ms"), Q_(12, "MB/s"), "illustrative scenario assumption"
                ),
                "cellular_relay": LinkProfile(
                    "cellular_relay",
                    "Phone cellular relay",
                    Q_(80, "ms"),
                    Q_(4, "MB/s"),
                    "illustrative scenario assumption",
                ),
            },
            "staged_sync",
            Q_(30, "s"),
            "sync-window or update backlog miss",
        ),
        "edge": TrackScenario(
            "edge",
            "vehicles keeping safety traffic local and staging fleet evidence",
            Q_(96, "MB"),
            32,
            Q_(90, "ms"),
            0.62,
            "vehicle_fabric",
            {
                "vehicle_fabric": LinkProfile(
                    "vehicle_fabric",
                    "Vehicle-local sensor fabric",
                    Q_(0.7, "ms"),
                    Q_(9500, "MB/s"),
                    "illustrative scenario assumption",
                ),
                "depot_wifi": LinkProfile(
                    "depot_wifi",
                    "Depot Wi-Fi offload",
                    Q_(18, "ms"),
                    Q_(75, "MB/s"),
                    "illustrative scenario assumption",
                ),
                "cellular_upload": LinkProfile(
                    "cellular_upload",
                    "Cellular fleet upload",
                    Q_(55, "ms"),
                    Q_(6, "MB/s"),
                    "illustrative scenario assumption",
                ),
            },
            "local_then_staged",
            Q_(12, "ms"),
            "safety-path latency or evidence-upload backlog miss",
        ),
    }


def get_track_scenario(track_id: str) -> TrackScenario:
    try:
        canonical_track_id = _TRACK_ALIASES[track_id]
        return track_scenarios()[canonical_track_id]
    except KeyError as exc:
        raise ValueError(f"unknown track_id: {track_id}") from exc

def default_traffic_pattern_label(track_id: str) -> str:
    scenario = get_track_scenario(track_id)
    return TRAFFIC_PATTERN_LABELS.get(scenario.traffic_pattern, "Synchronized fan-in")


def default_overlap_label(track_id: str) -> str:
    scenario = get_track_scenario(track_id)
    ms = _ms(scenario.overlap_window)
    for opt_key, qty in OVERLAP_WINDOW_QUANTITIES.items():
        if abs(_ms(qty) - ms) < 1e-4:
            for label, k in OVERLAP_WINDOW_OPTIONS.items():
                if k == opt_key:
                    return label
    return "None"


def default_payload_label(track_id: str) -> str:
    scenario = get_track_scenario(track_id)
    mb = _mb(scenario.payload)
    mapping = {350: "350 MB", 8: "8 MB", 6: "6 MB", 96: "96 MB"}
    for key_mb, label in mapping.items():
        if abs(mb - key_mb) < 1e-4:
            return label
    return "1 MB"


def burst_spacing_params(track_id: str) -> dict[str, object]:
    """Supply control parameters for burst spacing based on the track."""
    canonical = _TRACK_ALIASES.get(track_id, track_id)
    if canonical == "tinyml":
        return {
            "start": 0, "stop": 5000, "step": 250, "value": 0,
            "label": "Burst spacing (ms)",
        }
    return {
        "start": 0, "stop": 20, "step": 1, "value": 0,
        "label": "Burst spacing (ms)",
    }

def alpha_beta_experiment(track_id: str, link_id: str, payload: object) -> dict[str, object]:
    """Decompose one finite transfer into startup and serialization terms."""

    scenario = get_track_scenario(track_id)
    try:
        link = scenario.links[link_id]
    except KeyError as exc:
        raise ValueError(f"unknown link_id for {track_id}: {link_id}") from exc
    if payload.to(ureg.byte).magnitude < 0:
        raise ValueError("payload must be nonnegative")
    total = calc_point_to_point_time(payload, link.alpha, link.bandwidth)
    serialization = (payload / link.bandwidth).to(ureg.second)
    crossover = calc_alpha_beta_crossover(link.alpha, link.bandwidth)
    return {
        "inputs": {
            "track_id": scenario.track_id,
            "link_id": link_id,
            "payload": serialize_quantity(payload),
        },
        "track_id": scenario.track_id,
        "link_id": link_id,
        "alpha_ms": _ms(link.alpha),
        "serialization_ms": _ms(serialization),
        "total_ms": _ms(total),
        "crossover_kb": float(crossover.to(ureg.KB).magnitude),
        "binding_term": "startup" if link.alpha >= serialization else "serialization",
        "within_budget": total <= scenario.communication_budget,
        "assumption": link.assumption,
    }


def build_traffic_matrix(
    participants: int,
    payload: object,
    pattern: str,
    *,
    burst_spacing: object = Q_(0, "ms"),
) -> tuple[Flow, ...]:
    """Build a deterministic traffic matrix for comparison experiments."""

    if participants < 4:
        raise ValueError("participants must be at least 4")
    if payload.to(ureg.byte).magnitude <= 0:
        raise ValueError("payload must be positive")
    if burst_spacing.to(ureg.second).magnitude < 0:
        raise ValueError("burst_spacing must be nonnegative")
    flows: list[Flow] = []
    for source in range(participants):
        if pattern == "all_to_all":
            destination = (source + participants // 2) % participants
        elif pattern == "regional_fan_in":
            destination = (source // 8) * 8
        elif pattern == "staged_sync":
            destination = (source // 4) * 4
        elif pattern == "local_then_staged":
            destination = source ^ 1
        elif pattern == "incast":
            destination = 0
        elif pattern == "neighbor":
            destination = (source + 1) % participants
        else:
            raise ValueError(f"unknown traffic pattern: {pattern}")
        if destination == source:
            destination = (destination + 1) % participants
        flows.append(
            Flow(
                f"flow-{source}",
                source,
                destination,
                payload,
                source * burst_spacing,
            )
        )
    return tuple(flows)


def topology_options(
    participants: int,
    link_bandwidth: object,
    track_id: str = "cloud",
) -> dict[str, Topology]:
    """Construct topology options from explicit lane and switch counts.

    Costs and labels reflect the deployment track fleet entity (accelerators,
    phones, wearables with gateway pools, or vehicles with depot gateways).
    All non-datacenter costs are illustrative planning assumptions in USD.
    """

    if participants < 4:
        raise ValueError("participants must be at least 4")
    link_bandwidth.to(ureg.byte / ureg.second)
    groups = max(2, ceil(participants / 8))
    full_lanes = max(2, participants // 2)

    canonical = get_track_scenario(track_id).track_id
    if canonical == "tinyml":
        role = "wearable/phone sync gateway pool"
        link_cost = Q_(15, "USD")  # illustrative phone sync channel planning assumption
        switch_cost = Q_(180, "USD")  # illustrative sync gateway hub planning assumption
        labels = {
            "nonblocking": "Non-blocking gateway pool",
            "aligned": "Traffic-aligned phone relays",
            "grouped": "Grouped sync gateways",
            "oversubscribed": "4:1 oversubscribed gateway tier",
        }
        assumptions = {
            "nonblocking": "illustrative dedicated gateway pool planning assumption; MCUs stage via phone-gateway boundary without direct collectives",
            "aligned": "illustrative phone-aligned gateway relay planning assumption; sync traffic staged at phone tier",
            "grouped": "illustrative grouped sync gateway planning assumption with shared upstream staging tier",
            "oversubscribed": "illustrative oversubscribed staging gateway planning assumption with shared backend link",
        }
    elif canonical == "mobile":
        role = "phone ingress & regional edge gateway pool"
        link_cost = Q_(80, "USD")  # illustrative mobile ingress channel planning assumption
        switch_cost = Q_(1_200, "USD")  # illustrative regional ingress gateway planning assumption
        labels = {
            "nonblocking": "Non-blocking ingress pool",
            "aligned": "Traffic-aligned ingress relays",
            "grouped": "Grouped regional ingress",
            "oversubscribed": "4:1 oversubscribed edge ingress",
        }
        assumptions = {
            "nonblocking": "illustrative full-capacity regional ingress gateway planning assumption",
            "aligned": "illustrative traffic-aligned ingress relay planning assumption with partitioned edge pools",
            "grouped": "illustrative grouped regional ingress planning assumption with shared edge aggregation",
            "oversubscribed": "illustrative oversubscribed edge ingress planning assumption with shared backhaul",
        }
    elif canonical == "edge":
        role = "vehicle & depot gateway pool"
        link_cost = Q_(350, "USD")  # illustrative automotive/depot offload link planning assumption
        switch_cost = Q_(4_500, "USD")  # illustrative depot/vehicle gateway switch planning assumption
        labels = {
            "nonblocking": "Non-blocking depot fabric",
            "aligned": "Traffic-aligned vehicle rails",
            "grouped": "Grouped depot gateways",
            "oversubscribed": "4:1 oversubscribed depot spine",
        }
        assumptions = {
            "nonblocking": "illustrative non-blocking depot offload and vehicle gateway planning assumption; safety traffic kept local",
            "aligned": "illustrative vehicle-aligned sensor and depot relay planning assumption; safety traffic local and upload staged",
            "grouped": "illustrative grouped vehicle-depot gateway planning assumption with shared upload tier",
            "oversubscribed": "illustrative oversubscribed depot offload planning assumption with shared staging uplink",
        }
    else:
        role = "accelerator fabric switch & link pool"
        link_cost = Q_(2_400, "USD")  # illustrative installed-link assumption
        switch_cost = Q_(36_000, "USD")  # illustrative switch assumption
        labels = {
            "nonblocking": "Non-blocking fat-tree", "aligned": "Traffic-aligned rails",
            "grouped": "Grouped fabric", "oversubscribed": "4:1 oversubscribed spine",
        }
        assumptions = {
            "nonblocking": "illustrative two-tier full-bisection design",
            "aligned": "illustrative rail-local design with fewer cross-group lanes",
            "grouped": "illustrative group-local design with a shared global tier",
            "oversubscribed": "illustrative capacity-reduced shared fabric",
        }

    return {
        "nonblocking": Topology(
            "nonblocking",
            labels["nonblocking"],
            participants,
            full_lanes,
            full_lanes,
            groups,
            2,
            2 * groups + 2,
            2 * participants + 2 * full_lanes,
            link_cost,
            switch_cost,
            assumptions["nonblocking"],
            role=role,
        ),
        "aligned": Topology(
            "aligned",
            labels["aligned"],
            participants,
            max(2, groups),
            full_lanes,
            groups,
            1,
            groups + 2,
            2 * participants + groups,
            link_cost,
            switch_cost,
            assumptions["aligned"],
            role=role,
        ),
        "grouped": Topology(
            "grouped",
            labels["grouped"],
            participants,
            max(1, groups // 2),
            max(2, participants // 4),
            groups,
            2,
            groups + 1,
            participants + 2 * groups,
            link_cost,
            switch_cost,
            assumptions["grouped"],
            role=role,
        ),
        "oversubscribed": Topology(
            "oversubscribed",
            labels["oversubscribed"],
            participants,
            max(1, full_lanes // 4),
            max(1, participants // 8),
            groups,
            3,
            groups + 1,
            participants + max(1, full_lanes // 2),
            link_cost,
            switch_cost,
            assumptions["oversubscribed"],
            role=role,
        ),
    }


def _lane_for_flow(flow: Flow, topology: Topology) -> str:
    group_size = max(1, ceil(topology.endpoints / topology.groups))
    source_group = min(topology.groups - 1, flow.source // group_size)
    destination_group = min(topology.groups - 1, flow.destination // group_size)
    if source_group == destination_group:
        lane = (flow.source + flow.destination) % topology.local_lanes
        return f"local-{lane}"
    lane = (source_group * 17 + destination_group * 7) % topology.cut_lanes
    return f"cut-{lane}"


def simulate_schedule(
    flows: Iterable[Flow],
    topology: Topology,
    link: LinkProfile,
    *,
    capacity_fraction: float = 1.0,
) -> ScheduleResult:
    """FIFO-schedule finite flows over the topology's shared lanes."""

    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in (0, 1]")
    effective_bandwidth = link.bandwidth * capacity_fraction
    lane_free: dict[str, float] = {}
    lane_busy: dict[str, float] = {}
    lane_bytes: dict[str, float] = {}
    results: list[FlowResult] = []
    ordered = sorted(
        flows,
        key=lambda flow: (_ms(flow.ready_at), flow.source, flow.destination),
    )
    for flow in ordered:
        lane_id = _lane_for_flow(flow, topology)
        ready_ms = _ms(flow.ready_at)
        start_ms = max(ready_ms, lane_free.get(lane_id, 0.0))
        serialization_ms = _ms((flow.payload / effective_bandwidth).to(ureg.second))
        service_ms = topology.hop_count * _ms(link.alpha) + serialization_ms
        finish_ms = start_ms + service_ms
        lane_free[lane_id] = finish_ms
        lane_busy[lane_id] = lane_busy.get(lane_id, 0.0) + service_ms
        lane_bytes[lane_id] = lane_bytes.get(lane_id, 0.0) + _mb(flow.payload)
        results.append(FlowResult(flow.flow_id, lane_id, start_ms, finish_ms, start_ms - ready_ms, service_ms))
    if not results:
        raise ValueError("at least one flow is required")
    completion = max(result.finish_ms for result in results)
    finishes = [result.finish_ms for result in results]
    bottleneck = max(lane_busy, key=lane_busy.get)
    total_mb = sum(_mb(flow.payload) for flow in ordered)
    return ScheduleResult(
        tuple(results),
        completion,
        _percentile(finishes, 0.50),
        _percentile(finishes, 0.95),
        max(result.queue_ms for result in results),
        bottleneck,
        lane_busy,
        lane_bytes,
        total_mb / (completion / 1000),
    )


def topology_experiment(
    track_id: str,
    *,
    link_id: str | None = None,
    participants: int | None = None,
    payload: object | None = None,
    traffic_pattern: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Compare identical traffic over explicit topology resources."""

    scenario = get_track_scenario(track_id)
    n = participants or scenario.participants
    message = payload or scenario.payload
    link = scenario.links[link_id or scenario.default_link_id]
    pattern = traffic_pattern or scenario.traffic_pattern
    evaluator_inputs = {
        "track_id": scenario.track_id,
        "link_id": link.link_id,
        "participants": n,
        "payload": serialize_quantity(message),
        "traffic_pattern": pattern,
    }
    flows = build_traffic_matrix(n, message, pattern)
    rows = []
    for topology in topology_options(n, link.bandwidth, track_id=scenario.track_id).values():
        result = simulate_schedule(flows, topology, link)
        aggregate_cut_capacity = topology.cut_lanes * link.bandwidth
        utilization = result.lane_busy_ms[result.bottleneck_lane] / result.completion_ms
        rows.append(
            {
                "inputs": {**evaluator_inputs, "topology_id": topology.topology_id},
                "topology_id": topology.topology_id,
                "label": topology.label,
                "role": topology.role,
                "completion_ms": result.completion_ms,
                "p95_ms": result.p95_ms,
                "max_queue_ms": result.max_queue_ms,
                "bottleneck_lane": result.bottleneck_lane,
                "cut_capacity_mb_s": _mb_s(aggregate_cut_capacity),
                "utilization": utilization,
                "cost_usd": float(topology.cost.to(ureg.USD).magnitude),
                "within_budget": result.completion_ms <= _ms(scenario.communication_budget),
                "assumption": topology.assumption,
            }
        )
    return tuple(rows)


def congestion_experiment(
    track_id: str,
    topology_id: str,
    *,
    isolation_fraction: float = 0.0,
    burst_spacing: object = Q_(0, "ms"),
) -> dict[str, object]:
    """Compare shared traffic with a real foreground capacity reservation."""

    if not 0 <= isolation_fraction < 1:
        raise ValueError("isolation_fraction must be in [0, 1)")
    scenario = get_track_scenario(track_id)
    link = scenario.links[scenario.default_link_id]
    topology = topology_options(scenario.participants, link.bandwidth, track_id=scenario.track_id)[topology_id]
    foreground = build_traffic_matrix(scenario.participants, scenario.payload, "incast", burst_spacing=burst_spacing)
    background = tuple(
        replace(flow, flow_id=f"background-{index}", traffic_class="background")
        for index, flow in enumerate(
            build_traffic_matrix(scenario.participants, scenario.payload / 2, "incast", burst_spacing=burst_spacing)
        )
    )
    # Background traffic is already in flight when the foreground burst arrives.
    shared = simulate_schedule((*background, *foreground), topology, link)
    foreground_ids = {flow.flow_id for flow in foreground}
    shared_foreground_finishes = [flow.finish_ms for flow in shared.flows if flow.flow_id in foreground_ids]
    shared_foreground_p95 = _percentile(shared_foreground_finishes, 0.95)
    shared_foreground_completion = max(shared_foreground_finishes)
    if isolation_fraction:
        isolated = simulate_schedule(foreground, topology, link, capacity_fraction=isolation_fraction)
        reserved_mb_s = _mb_s(link.bandwidth * isolation_fraction) * topology.cut_lanes
        foreground_p95 = isolated.p95_ms
        foreground_completion = isolated.completion_ms
        foreground_max_queue = isolated.max_queue_ms
        bottleneck_lane = isolated.bottleneck_lane
    else:
        reserved_mb_s = 0.0
        foreground_p95 = shared_foreground_p95
        foreground_completion = shared_foreground_completion
        foreground_max_queue = max(flow.queue_ms for flow in shared.flows if flow.flow_id in foreground_ids)
        bottleneck_lane = shared.bottleneck_lane
    return {
        "inputs": {
            "track_id": scenario.track_id,
            "topology_id": topology_id,
            "isolation_fraction": isolation_fraction,
            "burst_spacing": serialize_quantity(burst_spacing),
        },
        "shared_p95_ms": shared.p95_ms,
        "shared_completion_ms": shared.completion_ms,
        "shared_foreground_p95_ms": shared_foreground_p95,
        "shared_foreground_completion_ms": shared_foreground_completion,
        "foreground_p95_ms": foreground_p95,
        "foreground_completion_ms": foreground_completion,
        "max_queue_ms": foreground_max_queue,
        "bottleneck_lane": bottleneck_lane,
        "reserved_capacity_mb_s": reserved_mb_s,
        "capacity_left_for_background_mb_s": _mb_s(link.bandwidth * (1 - isolation_fraction)) * topology.cut_lanes,
        "within_budget": foreground_completion <= _ms(scenario.communication_budget),
    }


def telemetry_experiment(
    track_id: str,
    topology_id: str,
    *,
    fault: str,
) -> dict[str, object]:
    """Test a diagnosis by applying one counterfactual intervention."""

    scenario = get_track_scenario(track_id)
    link = scenario.links[scenario.default_link_id]
    topology = topology_options(scenario.participants, link.bandwidth, track_id=scenario.track_id)[topology_id]
    flows = build_traffic_matrix(scenario.participants, scenario.payload, "incast")
    if fault == "capacity":
        impaired = replace(link, bandwidth=link.bandwidth / 2)
        baseline = simulate_schedule(flows, topology, impaired)
        counterfactual = simulate_schedule(flows, topology, link)
        expected_counter = "bottleneck lane busy time and queue delay"
    elif fault == "startup":
        impaired = replace(link, alpha=link.alpha * 8)
        baseline = simulate_schedule(flows, topology, impaired)
        counterfactual = simulate_schedule(flows, topology, link)
        expected_counter = "per-flow startup latency"
    elif fault == "traffic_hotspot":
        baseline = simulate_schedule(flows, topology, link)
        counterfactual_flows = build_traffic_matrix(scenario.participants, scenario.payload, "neighbor")
        counterfactual = simulate_schedule(counterfactual_flows, topology, link)
        expected_counter = "bytes and queue delay on one lane"
    else:
        raise ValueError(f"unknown fault: {fault}")
    improvement = baseline.p95_ms - counterfactual.p95_ms
    return {
        "inputs": {
            "track_id": scenario.track_id,
            "topology_id": topology_id,
            "fault": fault,
        },
        "fault": fault,
        "baseline_p95_ms": baseline.p95_ms,
        "counterfactual_p95_ms": counterfactual.p95_ms,
        "p95_improvement_ms": improvement,
        "supports_diagnosis": improvement > 0,
        "bottleneck_lane": baseline.bottleneck_lane,
        "counter_to_watch": expected_counter,
        "baseline_max_queue_ms": baseline.max_queue_ms,
        "counterfactual_max_queue_ms": counterfactual.max_queue_ms,
        "baseline": {
            "inputs": {
                "track_id": scenario.track_id,
                "topology_id": topology_id,
                "fault": fault,
                "intervention": "none",
            },
            "p95_ms": baseline.p95_ms,
            "max_queue_ms": baseline.max_queue_ms,
            "bottleneck_lane": baseline.bottleneck_lane,
        },
        "result": {
            "inputs": {
                "track_id": scenario.track_id,
                "topology_id": topology_id,
                "fault": fault,
                "intervention": "targeted_counterfactual",
            },
            "p95_ms": counterfactual.p95_ms,
            "max_queue_ms": counterfactual.max_queue_ms,
            "bottleneck_lane": counterfactual.bottleneck_lane,
        },
    }


def design_experiment(
    track_id: str,
    topology_id: str,
    *,
    traffic_pattern: str,
    overlap_window: object | None = None,
) -> dict[str, object]:
    """Evaluate one plan against a new traffic mix and bounded overlap."""

    scenario = get_track_scenario(track_id)
    link = scenario.links[scenario.default_link_id]
    topology = topology_options(scenario.participants, link.bandwidth, track_id=scenario.track_id)[topology_id]
    flows = build_traffic_matrix(scenario.participants, scenario.payload, traffic_pattern)
    schedule = simulate_schedule(flows, topology, link)
    available_overlap = overlap_window if overlap_window is not None else scenario.overlap_window
    if available_overlap.to(ureg.second).magnitude < 0:
        raise ValueError("overlap_window must be nonnegative")
    hidden_ms = min(schedule.completion_ms, _ms(available_overlap))
    exposed_ms = schedule.completion_ms - hidden_ms
    return {
        "inputs": {
            "track_id": scenario.track_id,
            "topology_id": topology_id,
            "traffic_pattern": traffic_pattern,
            "overlap_window": serialize_quantity(available_overlap),
        },
        "track_id": scenario.track_id,
        "topology_id": topology_id,
        "traffic_pattern": traffic_pattern,
        "completion_ms": schedule.completion_ms,
        "hidden_by_ready_compute_ms": hidden_ms,
        "exposed_ms": exposed_ms,
        "budget_ms": _ms(scenario.communication_budget),
        "valid_plan": exposed_ms <= _ms(scenario.communication_budget),
        "cost_usd": float(topology.cost.to(ureg.USD).magnitude),
        "bottleneck_lane": schedule.bottleneck_lane,
        "p95_ms": schedule.p95_ms,
        "remaining_limitation": scenario.failure_mode,
    }


def equal_budget_comparison(
    track_id: str,
    first_topology_id: str,
    second_topology_id: str,
    *,
    traffic_pattern: str,
    overlap_window: object | None = None,
) -> dict[str, object]:
    """Compare designs under the same illustrative capital budget.

    The common budget is the larger initial design cost.  The cheaper design
    spends its remaining budget on additional cross-cut lanes; any indivisible
    remainder is reported rather than silently treated as free capacity.
    """

    scenario = get_track_scenario(track_id)
    link = scenario.links[scenario.default_link_id]
    options = topology_options(scenario.participants, link.bandwidth, track_id=scenario.track_id)
    first = options[first_topology_id]
    second = options[second_topology_id]
    budget = max(first.cost, second.cost)

    def spend_to_budget(topology: Topology) -> Topology:
        extra_links = int(((budget - topology.cost) / topology.link_cost).to_base_units().magnitude)
        return replace(
            topology,
            cut_lanes=topology.cut_lanes + extra_links,
            links=topology.links + extra_links,
        )

    first_funded = spend_to_budget(first)
    second_funded = spend_to_budget(second)
    flows = build_traffic_matrix(scenario.participants, scenario.payload, traffic_pattern)
    first_schedule = simulate_schedule(flows, first_funded, link)
    second_schedule = simulate_schedule(flows, second_funded, link)
    available_overlap = overlap_window if overlap_window is not None else scenario.overlap_window
    if available_overlap.to(ureg.second).magnitude < 0:
        raise ValueError("overlap_window must be nonnegative")

    def summarize(topology: Topology, schedule: ScheduleResult) -> dict[str, object]:
        hidden_ms = min(schedule.completion_ms, _ms(available_overlap))
        return {
            "inputs": {
                "track_id": scenario.track_id,
                "topology_id": topology.topology_id,
                "traffic_pattern": traffic_pattern,
                "budget": serialize_quantity(budget),
                "overlap_window": serialize_quantity(available_overlap),
            },
            "topology_id": topology.topology_id,
            "label": topology.label,
            "role": topology.role,
            "cost_usd": float(topology.cost.magnitude),
            "unspent_usd": float((budget - topology.cost).magnitude),
            "cut_lanes": topology.cut_lanes,
            "completion_ms": schedule.completion_ms,
            "hidden_by_ready_compute_ms": hidden_ms,
            "exposed_ms": schedule.completion_ms - hidden_ms,
            "valid_plan": schedule.completion_ms - hidden_ms <= _ms(scenario.communication_budget),
        }

    first_result = summarize(first_funded, first_schedule)
    second_result = summarize(second_funded, second_schedule)
    return {
        "comparable": True,
        "inputs": {
            "track_id": scenario.track_id,
            "first_topology_id": first_topology_id,
            "second_topology_id": second_topology_id,
            "traffic_pattern": traffic_pattern,
            "overlap_window": serialize_quantity(available_overlap),
        },
        "budget_usd": float(budget.magnitude),
        "first": first_result,
        "second": second_result,
        "winner": first_topology_id if first_result["exposed_ms"] < second_result["exposed_ms"] else second_topology_id,
    }
