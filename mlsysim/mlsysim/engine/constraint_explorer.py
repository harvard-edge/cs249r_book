"""A small, bounded deployment-constraint explorer for introductory labs.

The values in :data:`TRACKS` are deliberately illustrative scenario assumptions,
not measurements of branded devices or production applications.  The model is
an additive per-request iron law: movement + compute + fixed runtime overhead.
It does not model queueing, tail latency, or thermal behavior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from mlsysim.core.units import Q_


_CANDIDATES = ("compact", "balanced", "large")
_INTERVENTIONS = ("none", "data", "model", "machine")

# Generic deployment envelopes.  These are scenario assumptions chosen for
# repeatable teaching comparisons; they are not claims about named products.
TRACKS: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML", "scenario": "A battery-powered local classifier",
        "provenance": "Illustrative teaching envelope; not an empirical device measurement.",
        "difficult_share": 0.30, "quality_floor_pct": 82.0,
        "memory_limit_mb": 2.5, "latency_limit_ms": 20.0, "energy_limit_mj": 1.50,
        "balanced_memory_mb": 1.20, "movement_volume_mb": 0.11, "bandwidth_mb_per_ms": 0.01,
        "ops_mflop": 3.0, "effective_rate_mflop_per_ms": 1.0,
        "overhead_ms": 2.0, "energy_per_ms": (0.040, 0.080, 0.020),
        "budget": 1.0, "monitor_interval_ms": 40.0, "monitor_effort_mj": 0.010,
    },
    "mobile": {
        "display": "Mobile", "scenario": "An interactive on-device assistant feature",
        "provenance": "Illustrative teaching envelope; not an empirical device measurement.",
        "difficult_share": 0.32, "quality_floor_pct": 85.0,
        "memory_limit_mb": 12.0, "latency_limit_ms": 30.0, "energy_limit_mj": 6.0,
        "balanced_memory_mb": 7.0, "movement_volume_mb": 40.0, "bandwidth_mb_per_ms": 10.0,
        "ops_mflop": 160.0, "effective_rate_mflop_per_ms": 20.0,
        "overhead_ms": 15.0, "energy_per_ms": (0.055, 0.150, 0.035),
        "budget": 1.0, "monitor_interval_ms": 80.0, "monitor_effort_mj": 0.030,
    },
    "edge": {
        "display": "Edge", "scenario": "A local inspection service near a sensor",
        "provenance": "Illustrative teaching envelope; not an empirical device measurement.",
        "difficult_share": 0.35, "quality_floor_pct": 85.0,
        "memory_limit_mb": 128.0, "latency_limit_ms": 35.0, "energy_limit_mj": 12.0,
        "balanced_memory_mb": 70.0, "movement_volume_mb": 56.0, "bandwidth_mb_per_ms": 10.0,
        "ops_mflop": 440.0, "effective_rate_mflop_per_ms": 20.0,
        "overhead_ms": 3.0, "energy_per_ms": (0.070, 0.280, 0.050),
        "budget": 1.0, "monitor_interval_ms": 120.0, "monitor_effort_mj": 0.080,
    },
    "cloud": {
        "display": "Cloud", "scenario": "A latency-bounded shared inference endpoint",
        "provenance": "Illustrative teaching envelope; not an empirical device measurement.",
        "difficult_share": 0.38, "quality_floor_pct": 84.0,
        "memory_limit_mb": 1000.0, "latency_limit_ms": 35.0, "energy_limit_mj": 30.0,
        "balanced_memory_mb": 600.0, "movement_volume_mb": 180.0, "bandwidth_mb_per_ms": 10.0,
        "ops_mflop": 320.0, "effective_rate_mflop_per_ms": 40.0,
        "overhead_ms": 3.0, "energy_per_ms": (0.100, 0.750, 0.080),
        "budget": 1.0, "monitor_interval_ms": 200.0, "monitor_effort_mj": 0.200,
    },
}

_MODEL = {
    "compact": {"scale": 0.60, "easy": 89.0, "difficult": 64.0},
    "balanced": {"scale": 1.00, "easy": 92.0, "difficult": 72.0},
    "large": {"scale": 1.60, "easy": 94.0, "difficult": 78.0},
}
_INTERVENTION_COST = {"none": 0.0, "data": 1.0, "model": 1.0, "machine": 1.0}


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _track(track_id: str) -> dict[str, Any]:
    if track_id not in TRACKS:
        raise ValueError(f"track_id must be one of {', '.join(TRACKS)}")
    return TRACKS[track_id]


def _rounded(value: float) -> float:
    return round(value, 4)


def margins(result: Mapping[str, Any]) -> dict[str, float]:
    """Return positive slack (or negative shortfall) for each deployment limit."""
    return {
        "quality_pct": _rounded(result["quality_pct"] - result["quality_floor_pct"]),
        "memory_mb": _rounded(result["memory_limit_mb"] - result["memory_mb"]),
        "latency_ms": _rounded(result["latency_limit_ms"] - result["latency_ms"]),
        "energy_mj": _rounded(result["energy_limit_mj"] - result["energy_mj"]),
    }


def speedup(baseline: Mapping[str, Any], result: Mapping[str, Any]) -> float:
    """Latency speedup for two evaluated outcomes (baseline latency / result latency)."""
    latency = _number(result["latency_ms"], "result latency_ms", minimum=0.0)
    if latency == 0:
        return float("inf")
    return _rounded(_number(baseline["latency_ms"], "baseline latency_ms", minimum=0.0) / latency)


def evaluate(
    track_id: str,
    candidate: str = "balanced",
    shift_pct: float = 0,
    compute_scale: float = 1,
    movement_scale: float = 1,
    overhead_scale: float = 1,
    intervention: str = "none",
    demand_scale: float = 1,
    budget_scale: float = 1,
) -> dict[str, Any]:
    """Evaluate one fixed-model deployment outcome.

    ``shift_pct`` is a percentage-point change in difficult-cohort share. The
    The timing equation is ``T = D_vol / BW + O / R_effective + L``.  The
    ``compute_scale``, ``movement_scale``, and ``overhead_scale`` multiply
    compute capability, bandwidth capability, and runtime-overhead capability
    respectively (so 2 means twice as capable). ``demand_scale`` multiplies
    serialized workload, and therefore represents repeated work in sequence,
    not a queue model. ``budget_scale`` scales the illustrative deployment
    envelope (memory, latency, energy, and intervention budget), not a
    hardware claim.
    """
    track = _track(track_id)
    if candidate not in _CANDIDATES:
        raise ValueError(f"candidate must be one of {', '.join(_CANDIDATES)}")
    if intervention not in _INTERVENTIONS:
        raise ValueError(f"intervention must be one of {', '.join(_INTERVENTIONS)}")
    shift_pct = _number(shift_pct, "shift_pct", minimum=-100.0)
    if shift_pct > 100.0:
        raise ValueError("shift_pct must be at most 100.0")
    compute_scale = _number(compute_scale, "compute_scale", minimum=0.0)
    movement_scale = _number(movement_scale, "movement_scale", minimum=0.0)
    overhead_scale = _number(overhead_scale, "overhead_scale", minimum=0.0)
    demand_scale = _number(demand_scale, "demand_scale", minimum=0.0)
    budget_scale = _number(budget_scale, "budget_scale", minimum=0.0)
    if min(compute_scale, movement_scale, overhead_scale, demand_scale, budget_scale) == 0:
        raise ValueError("work scales must be greater than 0")

    difficult_share = track["difficult_share"] + shift_pct / 100.0
    if not 0.0 <= difficult_share <= 1.0:
        raise ValueError("shift_pct moves the difficult-cohort share outside 0% to 100%")

    requested = candidate
    model_index = _CANDIDATES.index(candidate)
    effective_candidate = _CANDIDATES[min(model_index + 1, len(_CANDIDATES) - 1)] if intervention == "model" else candidate
    profile = _MODEL[effective_candidate]
    difficult_quality = profile["difficult"] + (7.0 if intervention == "data" else 0.0)
    quality = (1.0 - difficult_share) * profile["easy"] + difficult_share * difficult_quality

    work = profile["scale"] * demand_scale
    movement_volume_mb = track["movement_volume_mb"] * work
    ops_mflop = track["ops_mflop"] * work
    bandwidth_mb_per_ms = track["bandwidth_mb_per_ms"] * movement_scale
    effective_rate_mflop_per_ms = track["effective_rate_mflop_per_ms"] * compute_scale
    movement_ms = (Q_(movement_volume_mb, "megabyte") / Q_(bandwidth_mb_per_ms, "megabyte / millisecond")).to("millisecond").magnitude
    compute_ms = (Q_(ops_mflop, "megaflop") / Q_(effective_rate_mflop_per_ms, "megaflop / millisecond")).to("millisecond").magnitude
    overhead_ms = track["overhead_ms"] * demand_scale / overhead_scale
    machine_energy_multiplier = 1.0
    if intervention == "machine":
        compute_ms *= 0.60
        effective_rate_mflop_per_ms /= 0.60
        # Faster hardware has a deliberately explicit, assumed energy premium.
        machine_energy_multiplier = 2.0
    latency_ms = movement_ms + compute_ms + overhead_ms
    move_rate, compute_rate, overhead_rate = track["energy_per_ms"]
    energy_mj = movement_ms * move_rate + compute_ms * compute_rate * machine_energy_multiplier + overhead_ms * overhead_rate
    memory_mb = track["balanced_memory_mb"] * profile["scale"]

    terms = {"movement": movement_ms, "compute": compute_ms, "overhead": overhead_ms}
    violations: list[str] = []
    if quality < track["quality_floor_pct"]:
        violations.append("quality")
    memory_limit_mb = track["memory_limit_mb"] * budget_scale
    latency_limit_ms = track["latency_limit_ms"] * budget_scale
    energy_limit_mj = track["energy_limit_mj"] * budget_scale
    if memory_mb > memory_limit_mb:
        violations.append("memory")
    if latency_ms > latency_limit_ms:
        violations.append("latency")
    if energy_mj > energy_limit_mj:
        violations.append("energy")
    cost = _INTERVENTION_COST[intervention]
    budget = track["budget"] * budget_scale
    if cost > budget:
        violations.append("budget")

    operational_violations = [item for item in violations if item != "quality"]
    result = {
        "quality_pct": _rounded(quality), "quality_floor_pct": track["quality_floor_pct"],
        "memory_mb": _rounded(memory_mb), "memory_limit_mb": _rounded(memory_limit_mb),
        "movement_ms": _rounded(movement_ms), "compute_ms": _rounded(compute_ms),
        "overhead_ms": _rounded(overhead_ms), "latency_ms": _rounded(latency_ms),
        "latency_limit_ms": _rounded(latency_limit_ms), "energy_mj": _rounded(energy_mj),
        "energy_limit_mj": _rounded(energy_limit_mj), "feasible": not violations,
        "operational_ok": not operational_violations, "runtime_feasible": not operational_violations,
        "operational_violations": operational_violations,
        "violations": violations, "dominant_term": max(terms, key=terms.get),
        "intervention_cost": cost, "budget": budget, "candidate": requested,
        "track_id": track_id, "shift_pct": shift_pct,
        "effective_candidate": effective_candidate, "candidate_scale": profile["scale"],
        "difficult_share_pct": _rounded(difficult_share * 100.0),
        "easy_share_pct": _rounded((1.0 - difficult_share) * 100.0),
        "easy_quality_pct": profile["easy"], "difficult_quality_pct": _rounded(difficult_quality),
        "movement_volume_mb": _rounded(movement_volume_mb), "bandwidth_mb_per_ms": bandwidth_mb_per_ms,
        "ops_mflop": _rounded(ops_mflop), "effective_rate_mflop_per_ms": _rounded(effective_rate_mflop_per_ms),
        "provenance": track["provenance"],
        "quality_assumption": "Illustrative easy/difficult cohort rates; not an empirical accuracy measurement.",
        "intervention_assumption": {
            "none": "No change to the fixed model or execution path.",
            "data": "Assumed +7 percentage points on the difficult cohort only.",
            "model": "Assumed one-step candidate upgrade with its corresponding resource use.",
            "machine": "Assumed 40% compute-time reduction and a 2x compute-energy-rate premium.",
        }[intervention],
        "inputs": {
            "track_id": track_id, "candidate": requested, "shift_pct": shift_pct,
            "compute_scale": compute_scale, "movement_scale": movement_scale,
            "overhead_scale": overhead_scale, "intervention": intervention,
            "demand_scale": demand_scale, "budget_scale": budget_scale,
        },
    }
    return result


def compare(track_id: str, *, baseline: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Compare an action with the same baseline; no score surrogate is used."""
    action = evaluate(track_id, **kwargs)
    if baseline is None:
        baseline_kwargs = dict(kwargs)
        baseline_kwargs["intervention"] = "none"
        baseline = evaluate(track_id, **baseline_kwargs)
    else:
        baseline = dict(baseline)
    delta = {key: _rounded(action[key] - baseline[key]) for key in ("quality_pct", "memory_mb", "movement_ms", "compute_ms", "overhead_ms", "latency_ms", "energy_mj")}
    return {"baseline": baseline, "result": action, "delta": delta, "speedup": speedup(baseline, action), "margins": margins(action)}


def stress_compare(
    track_id: str,
    candidate: str = "balanced",
    shift_pct: float = 0,
    intervention: str = "none",
    condition: str = "demand",
    scale: float = 1,
    budget_scale: float = 1,
) -> dict[str, Any]:
    """Compare one capability or demand condition with its unchanged baseline.

    ``budget_scale`` is passed to both sides, so a capability counterfactual
    remains inside the same loose or tight deployment envelope.
    """
    if condition not in {"compute", "movement", "overhead", "demand"}:
        raise ValueError("condition must be one of compute, movement, overhead, demand")
    scale = _number(scale, "scale", minimum=0.0)
    if scale == 0:
        raise ValueError("scale must be greater than 0")
    kwargs: dict[str, Any] = {
        "candidate": candidate, "shift_pct": shift_pct, "intervention": intervention,
        "budget_scale": budget_scale,
    }
    kwargs[f"{condition}_scale"] = scale
    baseline = evaluate(
        track_id, candidate=candidate, shift_pct=shift_pct,
        intervention=intervention, budget_scale=budget_scale,
    )
    return compare(track_id, baseline=baseline, **kwargs)


def population_compare(
    track_id: str,
    candidate: str = "balanced",
    shift_pct: float = 0,
    shift_delta_pct: float = 0,
    intervention: str = "none",
    budget_scale: float = 1,
) -> dict[str, Any]:
    """Compare a changed cohort mix while holding model and runtime fixed.

    ``shift_delta_pct`` is an additional percentage-point difficult-cohort
    shift. Both outcomes preserve the supplied candidate, intervention, and
    deployment envelope; only cohort mix changes.
    """
    shift_pct = _number(shift_pct, "shift_pct", minimum=-100.0)
    shift_delta_pct = _number(shift_delta_pct, "shift_delta_pct", minimum=-100.0)
    if shift_pct > 100.0 or shift_delta_pct > 100.0:
        raise ValueError("population shifts must be at most 100.0")
    baseline = evaluate(
        track_id, candidate=candidate, shift_pct=shift_pct,
        intervention=intervention, budget_scale=budget_scale,
    )
    result = compare(
        track_id,
        baseline=baseline,
        candidate=candidate,
        shift_pct=shift_pct + shift_delta_pct,
        intervention=intervention,
        budget_scale=budget_scale,
    )
    result["shift_delta_pct"] = shift_delta_pct
    return result


def monitoring_plan(
    track_id: str,
    *,
    demand_scale: float = 1.0,
    check_interval_ms: float | None = None,
    window_ms: float = 1000.0,
) -> dict[str, Any]:
    """Describe monitoring work separately; it never changes model quality.

    Detection delay assumes failures arrive uniformly within an interval and a
    check observes them instantly: expected delay is interval/2, maximum delay
    is one interval. More frequent checks raise total inspection effort.
    """
    track = _track(track_id)
    demand_scale = _number(demand_scale, "demand_scale", minimum=0.0)
    if demand_scale == 0:
        raise ValueError("demand_scale must be greater than 0")
    interval = track["monitor_interval_ms"] if check_interval_ms is None else _number(check_interval_ms, "check_interval_ms", minimum=0.0)
    if interval == 0:
        raise ValueError("check_interval_ms must be greater than 0")
    window_ms = _number(window_ms, "window_ms", minimum=0.0)
    if window_ms == 0:
        raise ValueError("window_ms must be greater than 0")
    checks_per_window = window_ms / interval
    total_effort = track["monitor_effort_mj"] * checks_per_window * demand_scale
    return {
        "track_id": track_id, "check_interval_ms": _rounded(interval),
        "detection_delay_ms": _rounded(interval / 2.0),
        "maximum_detection_delay_ms": _rounded(interval), "window_ms": _rounded(window_ms),
        "checks_per_window": _rounded(checks_per_window),
        "check_effort_per_check_mj": _rounded(track["monitor_effort_mj"]),
        "check_effort_mj": _rounded(total_effort), "total_check_effort_mj": _rounded(total_effort),
        "detection_assumption": "Uniform failure arrival within an interval; instant check observation.",
        "provenance": track["provenance"],
    }
