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
    """Compute signed slack against each illustrative deployment limit.

    Parameters
    ----------
    result : mapping of str to object
        An evaluate result containing quality_pct, quality_floor_pct, memory_mb,
        memory_limit_mb, latency_ms, latency_limit_ms, energy_mj, and energy_limit_mj.

    Returns
    -------
    dict of str to float
        Quality slack in percentage points, memory slack in MB, latency slack in ms, and energy
        slack in mJ; positive values satisfy the corresponding limit. Values round to four
        decimal places.

    See Also
    --------
    evaluate : Evaluate one assumed deployment envelope.

    Notes
    -----
    Equation: quality_margin = quality - quality_floor; resource_margin = resource_limit -
    resource_use.

    Assumptions: The supplied result uses the units and constraint directions returned by
    evaluate.

    Excludes: Intervention-budget slack, uncertainty margins, and any inference that passing
    these illustrative limits certifies deployment.

    References
    ----------
    Derivation: A minimum-quality constraint permits quality minus floor; a maximum-resource
    constraint permits limit minus use. Their signs therefore share the same pass
    interpretation.

    Examples
    --------
    >>> r = evaluate("tinyml")
    >>> margins(r)["memory_mb"] == round(r["memory_limit_mb"] - r["memory_mb"], 4)
    True
    """
    return {
        "quality_pct": _rounded(result["quality_pct"] - result["quality_floor_pct"]),
        "memory_mb": _rounded(result["memory_limit_mb"] - result["memory_mb"]),
        "latency_ms": _rounded(result["latency_limit_ms"] - result["latency_ms"]),
        "energy_mj": _rounded(result["energy_limit_mj"] - result["energy_mj"]),
    }


def speedup(baseline: Mapping[str, Any], result: Mapping[str, Any]) -> float:
    """Calculate a latency ratio for two evaluated outcomes.

    Parameters
    ----------
    baseline : mapping of str to object
        Baseline with finite, nonnegative latency_ms in milliseconds.
    result : mapping of str to object
        Comparison with finite, nonnegative latency_ms in milliseconds.

    Returns
    -------
    float
        Baseline latency divided by comparison latency, rounded to four decimal places. Returns
        positive infinity whenever comparison latency is zero, including when both latencies are
        zero.

    Notes
    -----
    Equation: speedup = baseline[latency_ms] / result[latency_ms] for positive comparison
    latency.

    Assumptions: Both outcomes represent comparable work and have the same latency units;
    comparability is the caller's responsibility.

    Excludes: Quality, energy, memory, and statistical significance of a difference.

    References
    ----------
    Derivation: The ratio expresses how many comparison-duration intervals fit into one baseline
    duration. The zero-denominator infinity is an API convention, not a finite performance
    prediction.

    Examples
    --------
    >>> speedup({"latency_ms": 10}, {"latency_ms": 5})
    2.0
    """
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
    """Evaluate an illustrative deployment envelope using additive resource accounts.

    Parameters
    ----------
    track_id : str
        One of "tinyml", "mobile", "edge", or "cloud"; selects an assumed teaching envelope, not
        a branded device.
    candidate : str, optional
        One of "compact", "balanced" (default), or "large"; selects assumed work, footprint, and
        cohort-quality rates.
    shift_pct : float, optional
        Finite percentage-point shift in difficult-cohort share, default zero; the resulting
        share must remain in [0, 1].
    compute_scale : float, optional
        Positive finite compute-capability multiplier; default one. Twice the capability halves
        the compute term.
    movement_scale : float, optional
        Positive finite bandwidth multiplier; default one.
    overhead_scale : float, optional
        Positive finite overhead-capability multiplier; default one. The fixed runtime term
        divides by this value.
    intervention : str, optional
        "none" (default), "data", "model", or "machine"; selects the explicitly assumed cohort
        improvement, candidate upgrade, or compute-time/energy tradeoff.
    demand_scale : float, optional
        Positive finite serialized-work multiplier; default one. Scales movement, compute, and
        overhead but not resident model memory.
    budget_scale : float, optional
        Positive finite multiplier for memory, latency, energy, and intervention-cost limits;
        default one. Does not scale the quality floor.

    Returns
    -------
    dict of str to object
        Quality in percent, memory in MB, timing components and latency in ms, energy in mJ,
        limit violations, feasibility booleans, dominant timing term, and input/provenance
        records. Displayed numeric results round to four decimals; feasibility uses unrounded
        values.

    Notes
    -----
    Equation: T = movement_volume / bandwidth + operations / effective_compute_rate + overhead;
    E = sum(term_time * term_energy_rate); Q = ``(1-difficult_share)*easy_quality`` +
    ``difficult_share*difficult_quality``. Feasibility requires every quality, resource, and
    intervention-budget constraint.

    Assumptions: All times add without overlap. Candidate quality and interventions are
    constructed teaching assumptions. The data intervention adds seven percentage points to
    difficult-cohort quality; the model intervention advances one candidate; the machine
    intervention multiplies compute time by 0.60 and its energy rate by two.

    Excludes: Empirical accuracy prediction, queueing, concurrent demand, thermal behavior, and
    production hardware validation. operational_ok excludes quality violations but includes the
    intervention budget.

    References
    ----------
    Derivation: Work divided by capacity gives each service time. Adding serial terms yields
    latency, and weighting them by energy per millisecond yields energy. The quality mixture is
    a weighted mean of assumed cohort rates; it is not learned from measured outcomes.

    Examples
    --------
    >>> r = evaluate("mobile", demand_scale=2)
    >>> r["latency_ms"] == 2 * evaluate("mobile")["latency_ms"]
    True
    >>> r["memory_mb"] == evaluate("mobile")["memory_mb"]
    True
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
    """Compare one intervention outcome against an explicit or unchanged baseline.

    Parameters
    ----------
    track_id : str
        Illustrative deployment track accepted by evaluate.
    baseline : mapping of str to object or None, optional
        Previously evaluated baseline. None evaluates the same kwargs with intervention="none".
        Supplied baselines are copied but not checked for matching workloads.
    **kwargs : object
        Keyword arguments forwarded to evaluate for the action, including candidate,
        intervention, cohort shift, and positive capability/demand/budget scales.

    Returns
    -------
    dict of str to object
        Baseline and action dictionaries, rounded action-minus-baseline deltas in their original
        units, dimensionless latency speedup, and signed action margins.

    See Also
    --------
    evaluate : Compute each side of the comparison.
    margins : Interpret signed resource and quality slack.

    Notes
    -----
    Equation: delta[key] = action[key] - baseline[key]; speedup = baseline_latency /
    action_latency.

    Assumptions: The automatic baseline retains all action inputs except intervention. A
    supplied baseline is comparable only if the caller chooses matching work and units.

    Excludes: Automatic workload matching, uncertainty estimates, and a scalar score that trades
    quality against resource use.

    References
    ----------
    Derivation: Paired subtraction isolates changes in each reported metric without combining
    unlike units; the latency ratio separately reports relative execution time.

    Examples
    --------
    >>> r = compare("edge", intervention="none")
    >>> r["speedup"] == 1 and all(v == 0 for v in r["delta"].values())
    True
    """
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
    """Vary one capability or serialized-demand condition against a fixed baseline.

    Parameters
    ----------
    track_id : str
        Illustrative track accepted by evaluate.
    candidate : str, optional
        Assumed candidate on both sides; default "balanced".
    shift_pct : float, optional
        Difficult-cohort percentage-point shift on both sides; default zero.
    intervention : str, optional
        Intervention held fixed on both sides; default "none".
    condition : str, optional
        One of "compute", "movement", "overhead", or "demand" (default); selects the changed
        scale.
    scale : float, optional
        Positive finite multiplier of the selected capability or demand; default one.
    budget_scale : float, optional
        Positive deployment-limit multiplier shared by both sides; default one.

    Returns
    -------
    dict of str to object
        Comparison output with per-metric deltas, latency ratio, and margins. Times are ms, memory
        MB, energy mJ, and quality deltas percentage points.

    See Also
    --------
    compare : Return metric differences and margins.

    Notes
    -----
    Equation: baseline = evaluate with every capability and demand scale equal to one; action =
    evaluate with condition_scale = scale.

    Assumptions: Candidate, intervention, population mix, and deployment limits remain
    unchanged; only the named work/capability factor varies.

    Excludes: Concurrent request arrivals, changed model quality from faster hardware, and
    automatic changes in deployment limits.

    References
    ----------
    Derivation: Holding all other inputs fixed forms a one-factor counterfactual, allowing the
    changed additive timing term to be inspected separately.

    Examples
    --------
    >>> r = stress_compare("cloud", condition="compute", scale=2)
    >>> r["result"]["compute_ms"] == r["baseline"]["compute_ms"] / 2
    True
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
    """Vary difficult-cohort share while holding execution assumptions fixed.

    Parameters
    ----------
    track_id : str
        Illustrative track accepted by evaluate.
    candidate : str, optional
        Candidate shared by both populations; default "balanced".
    shift_pct : float, optional
        Baseline percentage-point shift from the track's difficult-cohort share; default zero.
    shift_delta_pct : float, optional
        Additional finite percentage-point shift for the comparison, default zero; both
        resulting shares must remain in [0, 1].
    intervention : str, optional
        Intervention shared by both populations; default "none".
    budget_scale : float, optional
        Positive deployment-limit multiplier on both sides; default one.

    Returns
    -------
    dict of str to object
        Paired outcomes, per-metric deltas, speedup and margins, plus shift_delta_pct. Quality
        deltas are percentage points; execution units match evaluate.

    See Also
    --------
    compare : Compare the baseline and shifted outcomes.

    Notes
    -----
    Equation: comparison_share = track_share + (shift_pct + shift_delta_pct)/100; quality =
    ``(1-share)*easy_quality`` + ``share*difficult_quality``.

    Assumptions: Only the cohort mixture changes. The same candidate has fixed easy/difficult
    quality and fixed resource work regardless of cohort.

    Excludes: Longer or more expensive difficult requests, distribution-dependent runtime, and
    empirical prediction of shifted accuracy.

    References
    ----------
    Derivation: A mixture of two fixed quality rates changes linearly with its mixing weight.
    Keeping the execution inputs fixed leaves the timing and resource accounts unchanged.

    Examples
    --------
    >>> r = population_compare("edge", shift_delta_pct=10)
    >>> r["delta"]["quality_pct"] < 0 and r["delta"]["latency_ms"] == 0
    True
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
    """Price periodic inspection effort and its ideal detection delay.

    Parameters
    ----------
    track_id : str
        Illustrative track selecting assumed interval and per-check effort.
    demand_scale : float, optional
        Positive finite multiplier for monitoring work per inspection; default one.
    check_interval_ms : float or None, optional
        Positive finite interval between checks in ms; None uses the track assumption.
    window_ms : float, optional
        Positive finite accounting-window duration in ms; default 1000.

    Returns
    -------
    dict of str to object
        Expected and maximum detection delays in ms, fractional checks/window, effort in mJ,
        chosen inputs, and assumption labels. Reported values round to four decimal places.

    Notes
    -----
    Equation: checks_per_window = window_ms / interval_ms; expected_delay = interval_ms/2;
    maximum_delay = interval_ms; effort = base_effort_per_check * checks_per_window *
    demand_scale.

    Assumptions: Failures arrive uniformly within a checking interval and checks observe them
    instantly. The fractional check count is a rate-based budget, not an integer schedule;
    per-check base effort is a teaching assumption.

    Excludes: False negatives, inspection execution delay, check jitter, corrective action, and
    changes in the classifier's quality.

    References
    ----------
    Derivation: The residual wait to the next periodic check is uniform over one interval, with
    mean half that interval. Multiplying check frequency, window duration, and per-check effort
    gives total inspection work.

    Examples
    --------
    >>> r = monitoring_plan("tinyml", check_interval_ms=20, window_ms=100)
    >>> r["checks_per_window"] == 5 and r["detection_delay_ms"] == 10
    True
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
