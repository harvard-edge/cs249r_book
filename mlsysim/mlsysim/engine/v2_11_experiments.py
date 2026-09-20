"""Chapter 11 experiments for resource-bounded fleet adaptation.

The fixtures in this module are deliberately small and illustrative.  Quality
values and rounds-to-target are supplied scenario evidence; none is inferred
from hardware speed, memory capacity, or a generic federation formula.
Physical arithmetic is performed with the MLSysIM Pint registry and concrete
hardware records.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from mlsysim.core.units import Q_, ureg
from mlsysim.hardware.registry import Hardware


METHODS = ("full", "adapter", "bias")
LOCAL_EPOCHS = (1, 2, 4, 8)
HETEROGENEITY_LEVELS = ("low", "moderate", "high")
MODEL_ID = "v2_11_experiments"


@dataclass(frozen=True)
class TrackProfile:
    """Illustrative workload layered on a registry-backed device."""

    label: str
    hardware: object
    model_parameters: object
    examples_per_update: int
    activation_bytes_per_example: object
    training_flops_per_example: object
    compute_utilization: float
    background_memory_fraction: float
    update_window: object
    update_energy_budget: object
    foreground_latency: object
    foreground_delay_budget: object
    replay_example_bytes: object
    client_uplink: object
    clients_per_round: int
    coordination_time: object
    raw_observations_per_client: object
    mains_powered: bool = False
    selection_assumptions: str = ""


TRACKS = {
    "tinyml": TrackProfile(
        "TinyML sensor fleet",
        Hardware.Tiny.ESP32_S3,
        Q_(180_000, "param"),
        24,
        Q_(18, "KiB"),
        Q_(5.4, "MFLOP"),
        0.28,
        0.55,
        Q_(18, "second"),
        Q_(4.0, "joule"),
        Q_(35, "ms"),
        Q_(4, "ms"),
        Q_(320, "byte"),
        Q_(0.25, "MB/s"),
        12,
        Q_(1.2, "second"),
        Q_(1.8, "MB"),
        mains_powered=False,
        selection_assumptions="MCU leaf sensors with gateway availability windows and duty cycles",
    ),
    "mobile": TrackProfile(
        "Mobile personalization fleet",
        Hardware.Mobile.Pixel8,
        Q_(24_000_000, "param"),
        64,
        Q_(1.5, "MiB"),
        Q_(1.6, "GFLOP"),
        0.18,
        0.12,
        Q_(45, "second"),
        Q_(75, "joule"),
        Q_(42, "ms"),
        Q_(5, "ms"),
        Q_(12, "KiB"),
        Q_(4, "MB/s"),
        24,
        Q_(0.8, "second"),
        Q_(48, "MB"),
        mains_powered=False,
        selection_assumptions="Battery-powered smartphones with Wi-Fi and charging requirements",
    ),
    "edge": TrackProfile(
        "Edge site fleet",
        Hardware.Edge.JetsonOrinNX,
        Q_(110_000_000, "param"),
        96,
        Q_(8, "MiB"),
        Q_(8.8, "GFLOP"),
        0.22,
        0.18,
        Q_(90, "second"),
        Q_(650, "joule"),
        Q_(18, "ms"),
        Q_(3, "ms"),
        Q_(28, "KiB"),
        Q_(12, "MB/s"),
        32,
        Q_(0.5, "second"),
        Q_(180, "MB"),
        mains_powered=True,
        selection_assumptions="Mains-powered on-site edge gateways and local compute nodes",
    ),
    # Cloud track clients are independent regional adaptation workers; the T4
    # memory below belongs to each worker and is never pooled across the fleet.
    "cloud": TrackProfile(
        "Regional cloud adaptation pool",
        Hardware.Cloud.T4,
        Q_(260_000_000, "param"),
        128,
        Q_(24, "MiB"),
        Q_(21, "GFLOP"),
        0.30,
        0.20,
        Q_(120, "second"),
        Q_(4_500, "joule"),
        Q_(12, "ms"),
        Q_(2, "ms"),
        Q_(64, "KiB"),
        Q_(100, "MB/s"),
        48,
        Q_(0.25, "second"),
        Q_(750, "MB"),
        mains_powered=True,
        selection_assumptions="Mains-powered regional datacenter workers and cloud service capacity",
    ),
}


@dataclass(frozen=True)
class MethodProfile:
    trainable_fraction: float
    optimizer_bytes_per_trainable_parameter: int
    activation_fraction: float
    compute_fraction: float
    active_power_fraction: float
    foreground_interference_fraction: float
    supplied_new_context_quality_pct: float


_METHOD_PROFILES = {
    "full": MethodProfile(1.0, 12, 1.0, 1.0, 0.92, 0.30, 91.5),
    "adapter": MethodProfile(0.018, 12, 0.62, 0.66, 0.68, 0.14, 90.8),
    "bias": MethodProfile(0.0015, 4, 0.38, 0.44, 0.46, 0.06, 87.9),
}


def _track(track_id: str) -> TrackProfile:
    try:
        return TRACKS[track_id]
    except KeyError as exc:
        raise ValueError(f"unknown track {track_id!r}; choose from {', '.join(TRACKS)}") from exc


def _method(method: str) -> MethodProfile:
    try:
        return _METHOD_PROFILES[method]
    except KeyError as exc:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}") from exc


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def evaluate_adaptation(
    track_id: str,
    method: str,
    *,
    batch_size: int = 1,
    concurrent_with_foreground: bool = True,
    memory_budget_scale: float = 1.0,
    energy_budget_scale: float = 1.0,
    window_scale: float = 1.0,
) -> dict:
    """Evaluate update state, energy, duration, and foreground admission.

    The four admission checks are independent.  In particular, a method that
    fits memory can still be rejected for update-window, energy, or foreground
    latency pressure.
    """

    track = _track(track_id)
    profile = _method(method)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be an integer of at least one")
    memory_budget_scale = _finite_nonnegative(memory_budget_scale, "memory_budget_scale")
    energy_budget_scale = _finite_nonnegative(energy_budget_scale, "energy_budget_scale")
    window_scale = _finite_nonnegative(window_scale, "window_scale")

    parameters = track.model_parameters.to(ureg.count).magnitude
    trainable_parameters = parameters * profile.trainable_fraction
    weights = parameters * 2 * ureg.byte
    gradients = trainable_parameters * 2 * ureg.byte
    optimizer = trainable_parameters * profile.optimizer_bytes_per_trainable_parameter * ureg.byte
    activations = track.activation_bytes_per_example * batch_size * profile.activation_fraction
    total_memory = (weights + gradients + optimizer + activations).to(ureg.MiB)
    memory_budget = (track.hardware.memory.capacity * track.background_memory_fraction * memory_budget_scale).to(
        ureg.MiB
    )

    operations = (track.training_flops_per_example * track.examples_per_update * profile.compute_fraction).to(ureg.flop)
    effective_rate = track.hardware.compute.peak_flops * track.compute_utilization
    duration = (operations / effective_rate).to(ureg.second)
    active_power = track.hardware.tdp * profile.active_power_fraction
    energy = (active_power * duration).to(ureg.joule)

    foreground_delay = Q_(0, "ms")
    if concurrent_with_foreground:
        foreground_delay = (track.foreground_latency * profile.foreground_interference_fraction).to(ureg.ms)

    memory_ok = total_memory <= memory_budget
    window_ok = duration <= track.update_window * window_scale
    energy_ok = energy <= track.update_energy_budget * energy_budget_scale
    foreground_ok = foreground_delay <= track.foreground_delay_budget
    violations = [
        name
        for name, passed in (
            ("memory", memory_ok),
            ("update window", window_ok),
            ("energy", energy_ok),
            ("foreground latency", foreground_ok),
        )
        if not passed
    ]

    return {
        "model_key": f"{MODEL_ID}.evaluate_adaptation",
        "inputs": {
            "track_id": track_id,
            "method": method,
            "batch_size": batch_size,
            "concurrent_with_foreground": concurrent_with_foreground,
            "memory_budget_scale": memory_budget_scale,
            "energy_budget_scale": energy_budget_scale,
            "window_scale": window_scale,
        },
        "track_id": track_id,
        "track_label": track.label,
        "method": method,
        "trainable_parameters": trainable_parameters,
        "weights_mib": weights.to(ureg.MiB).magnitude,
        "gradients_mib": gradients.to(ureg.MiB).magnitude,
        "optimizer_mib": optimizer.to(ureg.MiB).magnitude,
        "activations_mib": activations.to(ureg.MiB).magnitude,
        "total_memory_mib": total_memory.magnitude,
        "memory_budget_mib": memory_budget.magnitude,
        "operations_gflop": operations.to(ureg.GFLOP).magnitude,
        "duration_s": duration.magnitude,
        "energy_j": energy.magnitude,
        "foreground_delay_ms": foreground_delay.magnitude,
        "foreground_delay_budget_ms": track.foreground_delay_budget.to(ureg.ms).magnitude,
        "memory_ok": memory_ok,
        "window_ok": window_ok,
        "energy_ok": energy_ok,
        "foreground_ok": foreground_ok,
        "admitted": not violations,
        "violations": violations,
        "supplied_new_context_quality_pct": profile.supplied_new_context_quality_pct,
        "evidence_kind": "illustrative supplied outcome",
    }


_REPLAY_EVIDENCE = {
    0.0: (72.0, 92.0),
    0.25: (83.5, 91.2),
    0.5: (89.0, 89.5),
    0.75: (92.0, 85.0),
}


def evaluate_replay(track_id: str, method: str, replay_fraction: float) -> dict:
    """Allocate a finite background-memory budget between update state and replay."""

    track = _track(track_id)
    _method(method)
    replay_fraction = float(replay_fraction)
    if replay_fraction not in _REPLAY_EVIDENCE:
        choices = ", ".join(str(value) for value in _REPLAY_EVIDENCE)
        raise ValueError(f"replay_fraction must select supplied evidence: {choices}")

    adaptation = evaluate_adaptation(track_id, method, concurrent_with_foreground=False)
    total_budget = track.hardware.memory.capacity * track.background_memory_fraction
    replay_budget = (total_budget * replay_fraction).to(ureg.byte)
    adaptation_budget = (total_budget - replay_budget).to(ureg.MiB)
    retained_examples = int((replay_budget / track.replay_example_bytes).to_base_units().magnitude)
    old_quality, new_quality = _REPLAY_EVIDENCE[replay_fraction]
    update_fits_allocation = adaptation["total_memory_mib"] <= adaptation_budget.magnitude

    return {
        "model_key": f"{MODEL_ID}.evaluate_replay",
        "inputs": {
            "track_id": track_id,
            "method": method,
            "replay_fraction": replay_fraction,
        },
        "track_id": track_id,
        "method": method,
        "replay_fraction": replay_fraction,
        "total_budget_mib": total_budget.to(ureg.MiB).magnitude,
        "adaptation_budget_mib": adaptation_budget.magnitude,
        "adaptation_required_mib": adaptation["total_memory_mib"],
        "replay_budget_mib": replay_budget.to(ureg.MiB).magnitude,
        "retained_examples": retained_examples,
        "old_context_quality_pct": old_quality,
        "new_context_quality_pct": new_quality,
        "balanced_quality_pct": min(old_quality, new_quality),
        "feasible": update_fits_allocation,
        "evidence_kind": "illustrative supplied replay outcome",
    }


_ROUNDS_TO_TARGET = {
    "low": {1: 78, 2: 43, 4: 27, 8: 23},
    "moderate": {1: 92, 2: 54, 4: 39, 8: 44},
    "high": {1: 118, 2: 73, 4: 58, 8: 86},
}


def evaluate_federation(
    track_id: str,
    local_epochs: int,
    heterogeneity: str,
    *,
    method: str = "adapter",
    target_quality_pct: float = 90.0,
    client_count_scale: float = 1.0,
) -> dict:
    """Compute wall time and bytes to the same supplied quality target."""

    track = _track(track_id)
    method_profile = _method(method)
    if local_epochs not in LOCAL_EPOCHS:
        raise ValueError(f"local_epochs must be one of {LOCAL_EPOCHS}")
    if heterogeneity not in HETEROGENEITY_LEVELS:
        raise ValueError(f"heterogeneity must be one of {HETEROGENEITY_LEVELS}")
    if not math.isclose(float(target_quality_pct), 90.0):
        raise ValueError("the supplied convergence fixture supports only the 90% target")
    client_count_scale = _finite_nonnegative(client_count_scale, "client_count_scale")
    if client_count_scale == 0:
        raise ValueError("client_count_scale must be positive")

    rounds = _ROUNDS_TO_TARGET[heterogeneity][local_epochs]
    clients = max(1, math.ceil(track.clients_per_round * client_count_scale))
    parameters = track.model_parameters.to(ureg.count).magnitude
    update_bytes = (parameters * method_profile.trainable_fraction * 2 * ureg.byte).to(ureg.byte)
    one_epoch_ops = track.training_flops_per_example * track.examples_per_update * method_profile.compute_fraction
    local_epoch_time = (one_epoch_ops / (track.hardware.compute.peak_flops * track.compute_utilization)).to(ureg.second)
    heterogeneity_tail = {"low": 1.08, "moderate": 1.25, "high": 1.55}[heterogeneity]
    local_time_per_round = local_epoch_time * local_epochs * heterogeneity_tail
    # The round's aggregate uplink is finite, so simultaneous clients share it.
    upload_time_per_round = (update_bytes * clients / track.client_uplink).to(ureg.second)
    download_time_per_round = (update_bytes * clients / track.client_uplink).to(ureg.second)
    round_time = local_time_per_round + upload_time_per_round + download_time_per_round + track.coordination_time
    wall_time = (round_time * rounds).to(ureg.second)
    total_bytes = (2 * update_bytes * clients * rounds).to(ureg.MiB)

    return {
        "model_key": f"{MODEL_ID}.evaluate_federation",
        "inputs": {
            "track_id": track_id,
            "local_epochs": local_epochs,
            "heterogeneity": heterogeneity,
            "method": method,
            "target_quality_pct": float(target_quality_pct),
            "client_count_scale": client_count_scale,
        },
        "track_id": track_id,
        "method": method,
        "heterogeneity": heterogeneity,
        "local_epochs": local_epochs,
        "clients_per_round": clients,
        "target_quality_pct": 90.0,
        "rounds_to_target": rounds,
        "local_time_per_round_s": local_time_per_round.to(ureg.second).magnitude,
        "communication_time_per_round_s": (upload_time_per_round + download_time_per_round).to(ureg.second).magnitude,
        "coordination_time_per_round_s": track.coordination_time.to(ureg.second).magnitude,
        "wall_time_s": wall_time.magnitude,
        "total_communication_mib": total_bytes.magnitude,
        "update_mib_per_client": update_bytes.to(ureg.MiB).magnitude,
        "evidence_kind": "illustrative supplied convergence trajectory",
    }


def evaluate_architecture(track_id: str, architecture: str, *, local_epochs: int = 2) -> dict:
    """Evaluate one adaptation placement over its stated update horizon."""

    if architecture not in {"central", "local", "federated", "none"}:
        raise ValueError("architecture must be 'central', 'local', 'federated', or 'none'")
    track = _track(track_id)
    federation = evaluate_federation(track_id, local_epochs, "moderate")
    clients = track.clients_per_round
    raw_total = (track.raw_observations_per_client * clients).to(ureg.MiB).magnitude
    values = {
        "central": (True, raw_total, True, "one raw-observation collection"),
        "local": (False, 0.0, False, "one local update"),
        "federated": (
            False,
            federation["total_communication_mib"],
            True,
            "supplied trajectory to 90% target",
        ),
        "none": (False, 0.0, False, "no adaptation performed"),
    }
    raw_leaves, transfer, population_update, horizon = values[architecture]
    return {
        "model_key": f"{MODEL_ID}.evaluate_architecture",
        "inputs": {
            "track_id": track_id,
            "architecture": architecture,
            "local_epochs": local_epochs,
        },
        "track_id": track_id,
        "architecture": architecture,
        "raw_data_leaves_device": raw_leaves,
        "fleet_transfer_mib": transfer,
        "population_update": population_update,
        "comparison_horizon": horizon,
    }


def compare_architectures(track_id: str, *, local_epochs: int = 2) -> list[dict]:
    """Compare central, device-local, and federated data movement boundaries."""

    return [
        evaluate_architecture(track_id, architecture, local_epochs=local_epochs)
        for architecture in ("central", "local", "federated")
    ]


@dataclass(frozen=True)
class ClientRecord:
    client_id: str
    cohort: str
    failure_domain: str
    battery_pct: float
    charging: bool
    uplink_mbps: float
    evidence_age_hours: float
    available_seconds: float
    relative_compute_rate: float
    example_count: int


_CLIENTS_BY_TRACK = {
    "tinyml": (
        ClientRecord("c01", "industrial", "site-a", 88, True, 1.8, 2, 120, 1.15, 320),
        ClientRecord("c02", "industrial", "site-a", 54, False, 1.2, 5, 85, 0.95, 210),
        ClientRecord("c03", "ambient", "site-b", 92, True, 2.2, 3, 160, 1.05, 360),
        ClientRecord("c04", "ambient", "site-b", 28, False, 0.4, 26, 35, 0.65, 90),
        ClientRecord("c05", "acoustic", "site-c", 79, True, 1.5, 1, 110, 1.20, 390),
        ClientRecord("c06", "acoustic", "site-c", 61, False, 0.8, 8, 75, 0.85, 180),
        ClientRecord("c07", "structural", "site-d", 84, True, 2.0, 3.5, 130, 1.00, 280),
        ClientRecord("c08", "structural", "site-d", 42, False, 0.5, 16, 50, 0.70, 130),
    ),
    "mobile": (
        ClientRecord("c01", "urban", "site-a", 88, True, 18, 2, 150, 1.20, 420),
        ClientRecord("c02", "urban", "site-a", 52, False, 9, 5, 95, 0.95, 260),
        ClientRecord("c03", "rural", "site-b", 91, True, 3.5, 3, 180, 0.82, 310),
        ClientRecord("c04", "rural", "site-b", 34, False, 1.2, 28, 45, 0.65, 120),
        ClientRecord("c05", "night", "site-c", 78, True, 7, 1, 125, 1.10, 380),
        ClientRecord("c06", "night", "site-c", 63, False, 5, 9, 80, 0.88, 230),
        ClientRecord("c07", "assistive", "site-d", 82, True, 4.5, 4, 140, 0.76, 190),
        ClientRecord("c08", "assistive", "site-d", 46, False, 2.0, 18, 65, 0.70, 150),
    ),
    "edge": (
        ClientRecord("c01", "retail", "site-a", 100, True, 85, 2, 150, 1.15, 480),
        ClientRecord("c02", "retail", "site-a", 100, True, 45, 1.5, 100, 0.95, 320),
        ClientRecord("c03", "logistics", "site-b", 100, True, 60, 4, 180, 1.05, 510),
        ClientRecord("c04", "logistics", "site-b", 100, True, 15, 22, 55, 0.75, 210),
        ClientRecord("c05", "clinic", "site-c", 100, True, 95, 1, 140, 1.20, 590),
        ClientRecord("c06", "clinic", "site-c", 100, True, 35, 8, 85, 0.90, 380),
        ClientRecord("c07", "metro", "site-d", 100, True, 70, 3, 160, 1.00, 420),
        ClientRecord("c08", "metro", "site-d", 100, True, 20, 15, 60, 0.80, 260),
    ),
    "cloud": (
        ClientRecord("c01", "finance", "site-a", 100, True, 800, 2, 160, 1.25, 1200),
        ClientRecord("c02", "finance", "site-a", 100, True, 550, 6, 110, 0.95, 850),
        ClientRecord("c03", "health", "site-b", 100, True, 650, 3, 180, 1.10, 1400),
        ClientRecord("c04", "health", "site-b", 100, True, 350, 20, 50, 0.80, 600),
        ClientRecord("c05", "analytics", "site-c", 100, True, 900, 1, 200, 1.30, 1600),
        ClientRecord("c06", "analytics", "site-c", 100, True, 450, 8, 95, 0.90, 950),
        ClientRecord("c07", "public", "site-d", 100, True, 750, 4, 150, 1.05, 1100),
        ClientRecord("c08", "public", "site-d", 100, True, 400, 16, 70, 0.85, 700),
    ),
}

_CLIENTS = _CLIENTS_BY_TRACK["mobile"]


def client_records(track_id: str) -> tuple[ClientRecord, ...]:
    """Return the explicit cohort roster used by a track's selection activity."""

    _track(track_id)
    return _CLIENTS_BY_TRACK[track_id]


def evaluate_client_selection(
    track_id: str,
    *,
    min_battery_pct: float = 40,
    require_charging: bool = False,
    min_uplink_mbps: float = 2,
    max_evidence_age_hours: float = 12,
    deadline_seconds: float = 90,
    max_clients: int = 4,
    selection_policy: str = "freshest",
    unavailable_domains: Iterable[str] = (),
) -> dict:
    """Filter and select explicit clients, reporting coverage and each failure."""

    _track(track_id)
    min_battery_pct = _finite_nonnegative(min_battery_pct, "min_battery_pct")
    min_uplink_mbps = _finite_nonnegative(min_uplink_mbps, "min_uplink_mbps")
    max_evidence_age_hours = _finite_nonnegative(max_evidence_age_hours, "max_evidence_age_hours")
    deadline_seconds = _finite_nonnegative(deadline_seconds, "deadline_seconds")
    if not isinstance(max_clients, int) or max_clients < 1:
        raise ValueError("max_clients must be an integer of at least one")
    if selection_policy not in {"freshest", "coverage"}:
        raise ValueError("selection_policy must be 'freshest' or 'coverage'")
    unavailable_domains = frozenset(unavailable_domains)
    track = _track(track_id)
    adapter = _METHOD_PROFILES["adapter"]
    adapter_update = track.model_parameters.to(ureg.count).magnitude * adapter.trainable_fraction * 2 * ureg.byte
    reference_update_time = evaluate_adaptation(track_id, "adapter", concurrent_with_foreground=False)["duration_s"]

    failures: dict[str, tuple[str, ...]] = {}
    completion_seconds: dict[str, float] = {}
    eligible: list[ClientRecord] = []
    records = client_records(track_id)
    for client in records:
        reasons = []
        uplink = Q_(client.uplink_mbps, "megabit/second")
        transfer_time = (adapter_update / uplink).to(ureg.second).magnitude
        completion_time = reference_update_time / client.relative_compute_rate + transfer_time
        completion_seconds[client.client_id] = completion_time
        if client.failure_domain in unavailable_domains:
            reasons.append("failure domain unavailable")
        if not track.mains_powered:
            if client.battery_pct < min_battery_pct:
                reasons.append("battery")
            if require_charging and not client.charging:
                reasons.append("not charging")
        if client.uplink_mbps < min_uplink_mbps:
            reasons.append("uplink")
        if client.evidence_age_hours > max_evidence_age_hours:
            reasons.append("stale evidence")
        if completion_time > client.available_seconds:
            reasons.append("availability window")
        if deadline_seconds > 0 and completion_time > deadline_seconds:
            reasons.append("deadline")
        if reasons:
            failures[client.client_id] = tuple(reasons)
        else:
            eligible.append(client)

    if selection_policy == "freshest":
        ranked = sorted(
            eligible, key=lambda client: (client.evidence_age_hours, -client.example_count, client.client_id)
        )
        selected = ranked[:max_clients]
    else:
        by_cohort: dict[str, list[ClientRecord]] = {}
        for client in sorted(eligible, key=lambda item: (item.evidence_age_hours, item.client_id)):
            by_cohort.setdefault(client.cohort, []).append(client)
        selected = []
        while len(selected) < max_clients and any(by_cohort.values()):
            for cohort in sorted(by_cohort):
                if by_cohort[cohort] and len(selected) < max_clients:
                    selected.append(by_cohort[cohort].pop(0))

    all_cohorts = {client.cohort for client in records}
    selected_cohorts = {client.cohort for client in selected}
    ages = [client.evidence_age_hours for client in selected]
    selected_completion = [completion_seconds[client.client_id] for client in selected]
    failure_counts: dict[str, int] = {}
    for reasons in failures.values():
        for reason in reasons:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

    return {
        "model_key": f"{MODEL_ID}.evaluate_client_selection",
        "inputs": {
            "track_id": track_id,
            "min_battery_pct": min_battery_pct,
            "require_charging": require_charging,
            "min_uplink_mbps": min_uplink_mbps,
            "max_evidence_age_hours": max_evidence_age_hours,
            "deadline_seconds": deadline_seconds,
            "max_clients": max_clients,
            "selection_policy": selection_policy,
            "unavailable_domains": sorted(unavailable_domains),
        },
        "track_id": track_id,
        "mains_powered": track.mains_powered,
        "charging_applicable": not track.mains_powered,
        "eligible_count": len(eligible),
        "selected_ids": [client.client_id for client in selected],
        "selected_count": len(selected),
        "selection_status": "selected" if selected else "no eligible clients",
        "selected_cohorts": sorted(selected_cohorts),
        "selected_cohort_count": len(selected_cohorts),
        "cohort_coverage_fraction": len(selected_cohorts) / len(all_cohorts),
        "selected_examples": sum(client.example_count for client in selected),
        "mean_evidence_age_hours": sum(ages) / len(ages) if ages else None,
        "max_evidence_age_hours": max(ages) if ages else None,
        "max_completion_seconds": max(selected_completion) if selected_completion else None,
        "completion_seconds": completion_seconds,
        "failures": failures,
        "failure_counts": failure_counts,
        "unavailable_domains": sorted(unavailable_domains),
        "selection_policy": selection_policy,
    }


__all__ = [
    "HETEROGENEITY_LEVELS",
    "LOCAL_EPOCHS",
    "METHODS",
    "MODEL_ID",
    "TRACKS",
    "ClientRecord",
    "TrackProfile",
    "client_records",
    "compare_architectures",
    "evaluate_adaptation",
    "evaluate_architecture",
    "evaluate_client_selection",
    "evaluate_federation",
    "evaluate_replay",
]
