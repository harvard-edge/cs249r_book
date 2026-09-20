"""Fixture-driven robustness experiments for Volume II, Chapter 14.

The module deliberately separates four questions that are easy to conflate:

* whether a labeled deployment trace changed;
* whether a monitor detected the change, and when;
* how a defense performed inside its stated threat scope; and
* how a confidence threshold trades accepted-request risk for fallback load.

All outcomes are deterministic, illustrative scenario fixtures.  Drift metrics
and monitor settings never manufacture accuracy changes.  Monitoring only
changes which trace events are flagged and when those flags become available.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..core.types import Quantity
from ..core.units import Q_


TRACK_IDS = ("tinyml", "mobile", "edge", "cloud")
SCENARIO_IDS = ("covariate_shift", "concept_drift", "corruption", "attack")


@dataclass(frozen=True)
class TraceRecord:
    """One outcome-labeled request in an illustrative deployment trace."""

    period: int
    cohort: str
    feature_bin: str
    label: int
    prediction: int
    confidence: float
    event_active: bool


@dataclass(frozen=True)
class PeriodSummary:
    period: int
    requests: int
    errors: int
    error_rate: float
    feature_total_variation: float
    event_active: bool
    unit: str = "requests"


@dataclass(frozen=True)
class MonitorFlag:
    """A flag retains both the evidence period and its availability time."""

    signal: str
    evidence_period: int
    observed_period: int
    value: float


@dataclass(frozen=True)
class DetectionResult:
    summaries: tuple[PeriodSummary, ...]
    flags: tuple[MonitorFlag, ...]
    event_start: int
    first_detection_period: int | None
    detection_delay_periods: int | None
    false_alarms: int
    missed_event_periods: int


@dataclass(frozen=True)
class DefenseResult:
    track_id: str
    defense_id: str
    requested_threat: str
    threat_scope: tuple[str, ...]
    applicable: bool
    clean_accuracy: float
    stressed_accuracy: float | None
    mean_latency: Quantity
    mean_energy: Quantity
    sample_count_per_condition: int
    evidence_label: str
    cost_boundary: str = ""


@dataclass(frozen=True)
class SelectiveExample:
    cohort: str
    label: int
    prediction: int
    confidence: float


@dataclass(frozen=True)
class SelectivePredictionResult:
    threshold: float
    requests: int
    accepted: int
    accepted_errors: int
    coverage: float
    selective_risk: float | None
    fallback_arrivals: int
    fallback_capacity: int
    fallback_overflow: int
    cohort_fallback_arrivals: Mapping[str, int]
    unit: str = "requests"


@dataclass(frozen=True)
class ConformalResult:
    alpha: float
    calibration_count: int
    corrected_rank: int
    threshold: float
    full_set_required: bool
    empirical_coverage: float
    exchangeability_assumed: bool
    assumption_note: str


_TRACK_CONTEXT = {
    "tinyml": {
        "cohorts": ("routine", "loose_contact"),
        "unit": "sensor windows",
        "scenario_name": "Industrial Vibration Sensor Fleet",
        "device_role": "vibration sensor MCU",
        "gateway_role": "field gateway",
        "backend_role": "central telemetry service",
        "fallback_destination": "field gateway",
        "base_latency": Q_("8 ms"),
        "base_energy": Q_("0.42 mJ"),
        "severity": 1,
    },
    "mobile": {
        "cohorts": ("common_context", "accessibility_tail"),
        "unit": "sessions",
        "scenario_name": "On-Device Mobile Assistant Fleet",
        "device_role": "on-device mobile client",
        "gateway_role": "cellular edge aggregator",
        "backend_role": "cloud backend service",
        "fallback_destination": "cloud backend service",
        "base_latency": Q_("24 ms"),
        "base_energy": Q_("5.8 mJ"),
        "severity": 1,
    },
    "edge": {
        "cohorts": ("clear_weather", "rare_weather"),
        "unit": "scenes",
        "scenario_name": "Smart Intersection Camera Fleet",
        "device_role": "smart edge camera",
        "gateway_role": "on-prem edge appliance",
        "backend_role": "central analytics service",
        "fallback_destination": "on-prem edge appliance",
        "base_latency": Q_("38 ms"),
        "base_energy": Q_("92 mJ"),
        "severity": 2,
    },
    "cloud": {
        "cohorts": ("established_tenant", "new_tenant"),
        "unit": "requests",
        "scenario_name": "Cloud Multi-Tenant Inference Service",
        "device_role": "regional ingress proxy",
        "gateway_role": "cluster worker node",
        "backend_role": "central model server",
        "fallback_destination": "secondary model pool",
        "base_latency": Q_("72 ms"),
        "base_energy": Q_("31 mJ"),
        "severity": 2,
    },
}


def track_context(track_id: str) -> Mapping[str, object]:
    """Return immutable-by-convention teaching context for a deployment track."""

    _require_choice("track_id", track_id, TRACK_IDS)
    return dict(_TRACK_CONTEXT[track_id])


def _require_choice(name: str, value: str, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"unknown {name} {value!r}; expected one of {tuple(choices)!r}")


def _feature_pattern(scenario_id: str, event_active: bool) -> tuple[str, ...]:
    normal = ("common",) * 6 + ("secondary",) * 4 + ("tail",) * 2
    if not event_active or scenario_id == "concept_drift":
        return normal
    if scenario_id == "covariate_shift":
        return ("common",) * 2 + ("secondary",) * 4 + ("tail",) * 6
    if scenario_id == "corruption":
        return ("corrupt",) * 4 + normal[:8]
    return ("crafted",) * 3 + normal[:9]


def _error_indices(scenario_id: str, event_active: bool, severity: int) -> set[int]:
    if not event_active:
        return {10}
    base = {
        "covariate_shift": {5, 8, 10},
        "concept_drift": {1, 4, 7, 10},
        "corruption": {0, 1, 2, 3},
        "attack": {0, 1, 2, 8, 11},
    }[scenario_id]
    if severity == 2:
        return base | {6}
    return base


def labeled_cohort_trace(track_id: str, scenario_id: str) -> tuple[TraceRecord, ...]:
    """Return a fixed labeled trace for one named robustness challenge.

    Periods 0--1 form the reference window, periods 2--4 contain the event,
    and period 5 is recovery.  In the concept-drift fixture, feature-bin
    frequencies stay exactly unchanged while labels and errors change.
    """

    _require_choice("track_id", track_id, TRACK_IDS)
    _require_choice("scenario_id", scenario_id, SCENARIO_IDS)
    context = _TRACK_CONTEXT[track_id]
    common_cohort, tail_cohort = context["cohorts"]
    severity = int(context["severity"])
    records: list[TraceRecord] = []
    for period in range(6):
        event_active = 2 <= period <= 4
        features = _feature_pattern(scenario_id, event_active)
        errors = _error_indices(scenario_id, event_active, severity)
        for index, feature_bin in enumerate(features):
            cohort = common_cohort if index < 8 else tail_cohort
            # The labels are fixture outcomes, not outputs of a drift formula.
            label = (index + period) % 2
            prediction = 1 - label if index in errors else label
            confidence = 0.94 - 0.035 * (index % 5)
            if index in errors and event_active:
                confidence = 0.91 - 0.02 * (index % 3)
            records.append(
                TraceRecord(
                    period=period,
                    cohort=str(cohort),
                    feature_bin=feature_bin,
                    label=label,
                    prediction=prediction,
                    confidence=round(confidence, 3),
                    event_active=event_active,
                )
            )
    return tuple(records)


def _distribution(records: Iterable[TraceRecord]) -> dict[str, float]:
    counts = Counter(record.feature_bin for record in records)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("a feature distribution requires at least one record")
    return {key: count / total for key, count in counts.items()}


def total_variation(
    reference: Mapping[str, float], current: Mapping[str, float]
) -> float:
    """Compute total-variation distance over the union of category bins."""

    keys = set(reference) | set(current)
    return 0.5 * sum(abs(reference.get(key, 0.0) - current.get(key, 0.0)) for key in keys)


def summarize_trace(
    trace: Sequence[TraceRecord],
    unit: str = "requests",
) -> tuple[PeriodSummary, ...]:
    """Compute observed errors and feature movement from trace records."""

    if not trace:
        raise ValueError("trace must contain records")
    grouped: dict[int, list[TraceRecord]] = defaultdict(list)
    for record in trace:
        grouped[record.period].append(record)
    reference_records = grouped.get(0, []) + grouped.get(1, [])
    if not reference_records:
        raise ValueError("trace requires reference records in periods 0 or 1")
    reference = _distribution(reference_records)
    summaries = []
    for period in sorted(grouped):
        records = grouped[period]
        errors = sum(record.prediction != record.label for record in records)
        summaries.append(
            PeriodSummary(
                period=period,
                requests=len(records),
                errors=errors,
                error_rate=errors / len(records),
                feature_total_variation=total_variation(reference, _distribution(records)),
                event_active=any(record.event_active for record in records),
                unit=unit,
            )
        )
    return tuple(summaries)


def evaluate_monitor(
    trace: Sequence[TraceRecord],
    *,
    feature_threshold: float,
    error_threshold: float,
    label_delay_periods: int,
    unit: str = "requests",
) -> DetectionResult:
    """Evaluate feature and delayed-label flags against actual event periods.

    A delayed outcome flag keeps its evidence period, so a correct late alert is
    not mislabeled as a false alarm merely because it arrives after recovery.
    """

    if not 0.0 <= feature_threshold <= 1.0:
        raise ValueError("feature_threshold must be between zero and one")
    if not 0.0 <= error_threshold <= 1.0:
        raise ValueError("error_threshold must be between zero and one")
    if label_delay_periods < 0:
        raise ValueError("label_delay_periods must be nonnegative")
    summaries = summarize_trace(trace, unit=unit)
    flags: list[MonitorFlag] = []
    for summary in summaries:
        if summary.feature_total_variation >= feature_threshold:
            flags.append(
                MonitorFlag(
                    "feature", summary.period, summary.period, summary.feature_total_variation
                )
            )
        if summary.error_rate >= error_threshold:
            flags.append(
                MonitorFlag(
                    "delayed_label",
                    summary.period,
                    summary.period + label_delay_periods,
                    summary.error_rate,
                )
            )
    event_periods = {summary.period for summary in summaries if summary.event_active}
    if not event_periods:
        raise ValueError("trace must contain an active event")
    true_flags = [flag for flag in flags if flag.evidence_period in event_periods]
    detected_sources = {flag.evidence_period for flag in true_flags}
    first_detection = min((flag.observed_period for flag in true_flags), default=None)
    event_start = min(event_periods)
    return DetectionResult(
        summaries=summaries,
        flags=tuple(flags),
        event_start=event_start,
        first_detection_period=first_detection,
        detection_delay_periods=(
            None if first_detection is None else max(0, first_detection - event_start)
        ),
        false_alarms=sum(flag.evidence_period not in event_periods for flag in flags),
        missed_event_periods=len(event_periods - detected_sources),
    )


_DEFENSE_PROFILES = {
    "baseline": {
        "clean": (True,) * 11 + (False,),
        "stressed": {
            "covariate_shift": (True,) * 8 + (False,) * 4,
            "concept_drift": (True,) * 7 + (False,) * 5,
            "corruption": (True,) * 6 + (False,) * 6,
            "attack": (True,) * 6 + (False,) * 6,
        },
        "latency_factor": 1.00,
        "energy_factor": 1.00,
    },
    "input_filter": {
        "clean": (True,) * 10 + (False,) * 2,
        "stressed": {
            "corruption": (True,) * 10 + (False,) * 2,
            "attack": (True,) * 9 + (False,) * 3,
        },
        "latency_factor": 1.14,
        "energy_factor": 1.10,
    },
    "robust_training": {
        "clean": (True,) * 10 + (False,) * 2,
        "stressed": {
            "covariate_shift": (True,) * 10 + (False,) * 2,
            "corruption": (True,) * 10 + (False,) * 2,
            "attack": (True,) * 10 + (False,) * 2,
        },
        "latency_factor": 1.06,
        "energy_factor": 1.08,
    },
    "fallback_ensemble": {
        "clean": (True,) * 11 + (False,),
        "stressed": {
            "covariate_shift": (True,) * 11 + (False,),
            "concept_drift": (True,) * 10 + (False,) * 2,
            "corruption": (True,) * 11 + (False,),
            "attack": (True,) * 11 + (False,),
        },
        "latency_factor": 1.48,
        "energy_factor": 1.62,
    },
}


def defense_catalog() -> Mapping[str, Mapping[str, object]]:
    """Describe the supplied defense fixtures and their explicit threat scopes."""

    return {
        key: {
            "threat_scope": tuple(value["stressed"]),
            "evidence_label": "illustrative supplied outcome fixture",
        }
        for key, value in _DEFENSE_PROFILES.items()
    }


def _defense_boundary(track_id: str, defense_id: str) -> str:
    context = _TRACK_CONTEXT[track_id]
    if defense_id == "fallback_ensemble":
        if track_id == "tinyml":
            return f"Delegated to {context['gateway_role']}; MCU SRAM/storage unmodeled"
        if track_id == "mobile":
            return f"Delegated to {context['backend_role']}; client memory/thermal unmodeled"
        if track_id == "edge":
            return f"Hosted on {context['gateway_role']}; edge accelerator VRAM unmodeled"
        return f"Hosted on {context['backend_role']}; secondary pool concurrency unmodeled"
    if defense_id == "input_filter":
        return f"Evaluated on {context['device_role']}; filter compute modeled, buffer memory unmodeled"
    if defense_id == "robust_training":
        return f"Evaluated on {context['device_role']}; training cost offline; runtime memory cost is outside this fixture"
    return f"Evaluated on {context['device_role']}; nominal baseline model"

def evaluate_defense(track_id: str, defense_id: str, threat_id: str) -> DefenseResult:
    """Summarize supplied clean/stressed outcomes; do not extrapolate scope."""

    _require_choice("track_id", track_id, TRACK_IDS)
    _require_choice("defense_id", defense_id, tuple(_DEFENSE_PROFILES))
    _require_choice("threat_id", threat_id, SCENARIO_IDS)
    profile = _DEFENSE_PROFILES[defense_id]
    stressed_by_threat = profile["stressed"]
    scope = tuple(stressed_by_threat)
    clean = tuple(profile["clean"])
    stressed = (
        tuple(stressed_by_threat[threat_id]) if threat_id in stressed_by_threat else None
    )
    context = _TRACK_CONTEXT[track_id]
    return DefenseResult(
        track_id=track_id,
        defense_id=defense_id,
        requested_threat=threat_id,
        threat_scope=scope,
        applicable=threat_id in scope,
        clean_accuracy=sum(clean) / len(clean),
        stressed_accuracy=(sum(stressed) / len(stressed) if stressed is not None else None),
        mean_latency=context["base_latency"] * float(profile["latency_factor"]),
        mean_energy=context["base_energy"] * float(profile["energy_factor"]),
        sample_count_per_condition=len(clean),
        evidence_label="illustrative supplied outcome fixture",
        cost_boundary=_defense_boundary(track_id, defense_id),
    )


_SELECTIVE_PATTERN = (
    # confidence, correct, cohort index.  Wrong high-confidence examples are
    # intentional: confidence is a routing score, not a truth certificate.
    (0.99, True, 0),
    (0.97, True, 0),
    (0.95, False, 1),
    (0.93, True, 0),
    (0.90, True, 0),
    (0.88, False, 1),
    (0.84, True, 0),
    (0.80, True, 0),
    (0.76, False, 1),
    (0.72, True, 0),
    (0.68, True, 1),
    (0.64, False, 1),
    (0.60, True, 0),
    (0.56, False, 1),
    (0.52, True, 0),
    (0.48, False, 1),
    (0.43, True, 0),
    (0.38, False, 1),
    (0.31, False, 1),
    (0.24, False, 1),
)


def selective_fixture(track_id: str) -> tuple[SelectiveExample, ...]:
    """Return actual confidence, prediction, label, and cohort fixture rows."""

    _require_choice("track_id", track_id, TRACK_IDS)
    cohorts = _TRACK_CONTEXT[track_id]["cohorts"]
    examples = []
    for index, (confidence, correct, cohort_index) in enumerate(_SELECTIVE_PATTERN):
        label = index % 2
        prediction = label if correct else 1 - label
        examples.append(
            SelectiveExample(str(cohorts[cohort_index]), label, prediction, confidence)
        )
    return tuple(examples)


_CONFORMAL_FIXTURES = {
    "tinyml": ((0.06, 0.09, 0.12, 0.16, 0.21, 0.27, 0.31, 0.38, 0.46), (0.08, 0.14, 0.24, 0.40, 0.51)),
    "mobile": ((0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45), (0.09, 0.18, 0.29, 0.44, 0.56)),
    "edge": ((0.04, 0.08, 0.13, 0.18, 0.24, 0.29, 0.34, 0.41, 0.49), (0.07, 0.17, 0.28, 0.43, 0.61)),
    "cloud": ((0.07, 0.11, 0.14, 0.19, 0.23, 0.28, 0.33, 0.39, 0.47), (0.10, 0.16, 0.26, 0.42, 0.58)),
}


def conformal_fixture(track_id: str, calibration_count: int = 9) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return supplied calibration and true-label test nonconformity scores."""

    _require_choice("track_id", track_id, TRACK_IDS)
    if calibration_count not in (3, 9):
        raise ValueError("calibration_count must be 3 or 9 for this finite fixture")
    calibration, test = _CONFORMAL_FIXTURES[track_id]
    return calibration[:calibration_count], test


def evaluate_selective_prediction(
    examples: Sequence[SelectiveExample],
    *,
    threshold: float,
    fallback_capacity: int,
    unit: str = "requests",
) -> SelectivePredictionResult:
    """Compute empirical risk/coverage and the resulting fallback arrivals."""

    if not examples:
        raise ValueError("examples must not be empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if fallback_capacity < 0:
        raise ValueError("fallback_capacity must be nonnegative")
    accepted = [example for example in examples if example.confidence >= threshold]
    rejected = [example for example in examples if example.confidence < threshold]
    accepted_errors = sum(example.prediction != example.label for example in accepted)
    cohort_fallback = Counter(example.cohort for example in rejected)
    return SelectivePredictionResult(
        threshold=threshold,
        requests=len(examples),
        accepted=len(accepted),
        accepted_errors=accepted_errors,
        coverage=len(accepted) / len(examples),
        selective_risk=(accepted_errors / len(accepted) if accepted else None),
        fallback_arrivals=len(rejected),
        fallback_capacity=fallback_capacity,
        fallback_overflow=max(0, len(rejected) - fallback_capacity),
        cohort_fallback_arrivals=dict(cohort_fallback),
        unit=unit,
    )


def conformal_quantile(calibration_scores: Sequence[float], alpha: float) -> tuple[int, float]:
    """Return the corrected split-conformal rank and threshold.

    If ``ceil((n + 1) * (1 - alpha))`` exceeds ``n``, the mathematically
    correct finite-sample threshold is infinity, corresponding to a full set.
    """

    if not calibration_scores:
        raise ValueError("calibration_scores must not be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if any(not math.isfinite(score) or score < 0.0 for score in calibration_scores):
        raise ValueError("calibration scores must be finite and nonnegative")
    count = len(calibration_scores)
    rank = math.ceil((count + 1) * (1.0 - alpha))
    if rank > count:
        return rank, math.inf
    return rank, sorted(calibration_scores)[rank - 1]


def evaluate_conformal_fixture(
    calibration_scores: Sequence[float],
    test_true_label_scores: Sequence[float],
    *,
    alpha: float,
    exchangeability_assumed: bool,
) -> ConformalResult:
    """Evaluate fixture coverage while recording, never verifying, the assumption."""

    if not test_true_label_scores:
        raise ValueError("test_true_label_scores must not be empty")
    if any(not math.isfinite(score) or score < 0.0 for score in test_true_label_scores):
        raise ValueError("test scores must be finite and nonnegative")
    rank, threshold = conformal_quantile(calibration_scores, alpha)
    empirical = sum(score <= threshold for score in test_true_label_scores) / len(
        test_true_label_scores
    )
    note = (
        "Exchangeability is declared for this exercise, not verified by the calculation."
        if exchangeability_assumed
        else "Exchangeability is not assumed; report empirical fixture coverage only."
    )
    return ConformalResult(
        alpha=alpha,
        calibration_count=len(calibration_scores),
        corrected_rank=rank,
        threshold=threshold,
        full_set_required=math.isinf(threshold),
        empirical_coverage=empirical,
        exchangeability_assumed=exchangeability_assumed,
        assumption_note=note,
    )


def _quantity_payload(value: Quantity) -> dict[str, object]:
    return {"magnitude": float(value.magnitude), "unit": str(value.units)}


def trace_event_interpretation(scenario_id: str) -> str:
    """Return pedagogical interpretation for the selected trace event."""

    _require_choice("scenario_id", scenario_id, SCENARIO_IDS)
    return {
        "concept_drift": (
            "Concept drift changes target relations while leaving input distributions "
            "unchanged (TV = 0), leaving feature monitors blind until delayed labels arrive."
        ),
        "covariate_shift": (
            "Covariate shift alters feature distributions directly, allowing immediate "
            "detection by input monitors without waiting for delayed outcome labels."
        ),
        "corruption": (
            "Input corruption produces anomalous feature bins flagged immediately by feature "
            "monitors alongside a sharp increase in prediction error."
        ),
        "attack": (
            "Adversarial perturbations shift input feature statistics and drive elevated "
            "error rates during the active attack periods."
        ),
    }[scenario_id]

def monitor_experiment_payload(
    track_id: str,
    scenario_id: str,
    *,
    feature_threshold: float,
    error_threshold: float,
    label_delay_periods: int,
) -> dict[str, object]:
    """Run and serialize a monitor experiment with replayable arguments."""

    inputs = {
        "track_id": track_id,
        "scenario_id": scenario_id,
        "feature_threshold": feature_threshold,
        "error_threshold": error_threshold,
        "label_delay_periods": label_delay_periods,
    }
    unit = str(_TRACK_CONTEXT[track_id]["unit"])
    result = evaluate_monitor(
        labeled_cohort_trace(track_id, scenario_id),
        feature_threshold=feature_threshold,
        error_threshold=error_threshold,
        label_delay_periods=label_delay_periods,
        unit=unit,
    )
    return {
        "inputs": inputs,
        "result": {
            "unit": unit,
            "interpretation": trace_event_interpretation(scenario_id),
            "event_start": result.event_start,
            "first_detection_period": result.first_detection_period,
            "detection_delay_periods": result.detection_delay_periods,
            "false_alarms": result.false_alarms,
            "missed_event_periods": result.missed_event_periods,
            "summaries": [vars(item) for item in result.summaries],
            "flags": [vars(item) for item in result.flags],
        },
    }


def defense_experiment_payload(
    track_id: str, defense_id: str, threat_id: str
) -> dict[str, object]:
    """Run and serialize supplied defense evidence with unit-safe costs."""

    inputs = {"track_id": track_id, "defense_id": defense_id, "threat_id": threat_id}
    result = evaluate_defense(track_id, defense_id, threat_id)
    return {
        "inputs": inputs,
        "result": {
            "applicable": result.applicable,
            "threat_scope": list(result.threat_scope),
            "clean_accuracy": result.clean_accuracy,
            "stressed_accuracy": result.stressed_accuracy,
            "mean_latency": _quantity_payload(result.mean_latency),
            "mean_energy": _quantity_payload(result.mean_energy),
            "sample_count_per_condition": result.sample_count_per_condition,
            "evidence_label": result.evidence_label,
            "cost_boundary": result.cost_boundary,
        },
    }


def selective_experiment_payload(
    track_id: str, *, threshold: float, fallback_capacity: int
) -> dict[str, object]:
    """Run and serialize a confidence-threshold experiment."""

    inputs = {
        "track_id": track_id,
        "threshold": threshold,
        "fallback_capacity": fallback_capacity,
    }
    unit = str(_TRACK_CONTEXT[track_id]["unit"])
    result = evaluate_selective_prediction(
        selective_fixture(track_id),
        threshold=threshold,
        fallback_capacity=fallback_capacity,
        unit=unit,
    )
    return {"inputs": inputs, "result": vars(result)}


def conformal_experiment_payload(
    calibration_scores: Sequence[float],
    test_true_label_scores: Sequence[float],
    *,
    alpha: float,
    exchangeability_assumed: bool,
) -> dict[str, object]:
    """Run and strictly serialize finite conformal fixture evidence."""

    inputs = {
        "calibration_scores": list(calibration_scores),
        "test_true_label_scores": list(test_true_label_scores),
        "alpha": alpha,
        "exchangeability_assumed": exchangeability_assumed,
    }
    result = evaluate_conformal_fixture(
        calibration_scores,
        test_true_label_scores,
        alpha=alpha,
        exchangeability_assumed=exchangeability_assumed,
    )
    return {
        "inputs": inputs,
        "result": {
            "alpha": result.alpha,
            "calibration_count": result.calibration_count,
            "corrected_rank": result.corrected_rank,
            "threshold": None if result.full_set_required else result.threshold,
            "full_set_required": result.full_set_required,
            "empirical_coverage": result.empirical_coverage,
            "exchangeability_assumed": result.exchangeability_assumed,
            "assumption_note": result.assumption_note,
        },
    }
