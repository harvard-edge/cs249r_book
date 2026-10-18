"""Responsible-AI experiments built from observable fleet evidence.

The records and operating profiles in this module are deterministic teaching
fixtures, not measurements of any product or population.  They keep three
boundaries explicit:

* scored examples and their labels determine subgroup outcomes;
* sampling, label delay, and review capacity determine what can be observed;
* a learner's fairness criterion and release decision are normative choices.

Changing a policy tolerance or monitoring plan therefore cannot rewrite the
fixture's true labels.  Physical workload and service calculations use Pint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from mlsysim.core.units import Q_


@dataclass(frozen=True)
class ScoredCase:
    """One labeled decision case in an illustrative subgroup fixture."""

    case_id: str
    subgroup: str
    true_label: int
    score: float
    label_age_hours: float
    sample_rank: int


@dataclass(frozen=True)
class ConfusionCounts:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative


@dataclass(frozen=True)
class SubgroupMetrics:
    subgroup: str
    counts: ConfusionCounts
    selection_rate: float
    true_positive_rate: float | None
    false_positive_rate: float | None
    positive_predictive_value: float | None


@dataclass(frozen=True)
class FairnessAssessment:
    track_id: str
    threshold: float
    criterion: str
    allowed_gap: float
    criterion_gap: float | None
    criterion_passes: bool
    subgroup_metrics: tuple[SubgroupMetrics, ...]
    ground_truth_positive_cases: int
    predicted_positive_cases: int
    false_negative_cases: int


@dataclass(frozen=True)
class CapacityResult:
    track_id: str
    explanation_share: float
    review_share: float
    explanation_arrivals: Any
    explanation_capacity: Any
    explanation_backlog: Any
    review_arrivals: Any
    review_capacity: Any
    review_backlog: Any
    explanation_utilization: float
    review_utilization: float
    feasible: bool


@dataclass(frozen=True)
class AuditEvidence:
    track_id: str
    threshold: float
    sampling_fraction: float
    label_delay: Any
    eligible_cases: int
    sampled_cases: int
    sampled_subgroups: tuple[str, ...]
    observed_assessment: FairnessAssessment | None
    population_assessment: FairnessAssessment
    evidence_complete: bool
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class PrivacyProfile:
    key: str
    label: str
    retained_fields: tuple[str, ...]
    raw_content_retained: bool
    stable_subject_id_retained: bool
    cohort_attribute_retained: bool
    retention: Any
    deletion_window: Any


@dataclass(frozen=True)
class PrivacyCheck:
    """Evidence and data-handling checks, never a privacy guarantee."""

    profile: PrivacyProfile
    supports_subgroup_audit: bool
    supports_case_review: bool
    retention_ok: bool
    deletion_ok: bool
    passes_requirements: bool
    violations: tuple[str, ...]
    track_id: str = ""
    scope: str = ""


@dataclass(frozen=True)
class MechanicalGate:
    passes: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseDecision:
    choice: str
    rejected_alternative: str
    owner: str
    remedy: str
    reevaluation_trigger: str
    evidence_gate: MechanicalGate
    decision_record_complete: bool
    missing_fields: tuple[str, ...]


# These are scenario assumptions for four fleet shapes.  Device fleets send
# selected evidence to gateway or regional services; they are not represented
# as doing fleet-scale audit processing on an MCU or phone.
TRACK_PROFILES: dict[str, dict[str, Any]] = {
    "tinyml": {
        "display": "TinyML",
        "fleet_role": "TinyML endpoint (offloads to phone gateway and review service)",
        "subgroups": ("common sensor contact", "variable sensor contact"),
        "events_per_hour": Q_(3_600, "count/hour"),
        "explanation_service_time": Q_(0.08, "second/count"),
        "review_service_time": Q_(240, "second/count"),
        "explanation_workers": 1,
        "reviewers": 3,
        "remedies": {
            "Fall back to local rule heuristic": "fall back to local rule heuristic",
            "Disable endpoint path and route to gateway": "disable endpoint path and route to gateway",
            "Roll back on-device model and flag cases": "roll back on-device model and flag cases",
            "No remedy": "",
        },
    },
    "mobile": {
        "display": "Mobile",
        "fleet_role": "Phone endpoint (syncs to backend review service)",
        "subgroups": ("reference interaction", "accessibility interaction"),
        "events_per_hour": Q_(24_000, "count/hour"),
        "explanation_service_time": Q_(0.12, "second/count"),
        "review_service_time": Q_(150, "second/count"),
        "explanation_workers": 2,
        "reviewers": 8,
        "remedies": {
            "Prompt user confirmation and local fallback": "prompt user confirmation and local fallback",
            "Disable on-device path and route to backend": "disable on-device path and route to backend",
            "Roll back on-device model update": "roll back on-device model update",
            "No remedy": "",
        },
    },
    "edge": {
        "display": "Edge",
        "fleet_role": "Edge gateway / node (regional review)",
        "subgroups": ("routine scene", "rare road context"),
        "events_per_hour": Q_(72_000, "count/hour"),
        "explanation_service_time": Q_(0.18, "second/count"),
        "review_service_time": Q_(300, "second/count"),
        "explanation_workers": 6,
        "reviewers": 18,
        "remedies": {
            "Engage fail-safe safe stop / heuristic": "engage fail-safe safe stop / heuristic",
            "Switch to redundant sensor pipeline": "switch to redundant sensor pipeline",
            "Roll back edge gateway model": "roll back edge gateway model",
            "No remedy": "",
        },
    },
    "cloud": {
        "display": "Cloud",
        "fleet_role": "Central backend / cloud service",
        "subgroups": ("well represented cohort", "underserved cohort"),
        "events_per_hour": Q_(600_000, "count/hour"),
        "explanation_service_time": Q_(0.04, "second/count"),
        "review_service_time": Q_(120, "second/count"),
        "explanation_workers": 12,
        "reviewers": 40,
        "remedies": {
            "Disable path and offer appeal": "disable affected path and offer appeal",
            "Roll back and review cases": "roll back and review labeled cases",
            "Route cohort to human triage queue": "route cohort to human triage queue",
            "No remedy": "",
        },
    },
}

_TRACK_ALIASES = {
    "oura_ring": "tinyml",
    "iphone": "mobile",
    "robotaxi": "edge",
    "cloud_fleet": "cloud",
}

# (score, label, label age in hours).  Each track uses the same reference
# cohort and a distinct affected cohort.  The cases are intentionally small so
# students can inspect every decision behind the confusion matrix.
_REFERENCE_CASES = (
    (0.93, 1, 96), (0.86, 1, 72), (0.78, 1, 48), (0.69, 1, 30),
    (0.58, 1, 18), (0.47, 1, 10), (0.81, 0, 84), (0.62, 0, 60),
    (0.44, 0, 36), (0.33, 0, 24), (0.19, 0, 12), (0.08, 0, 4),
)

_AFFECTED_CASES = {
    "tinyml": (
        (0.88, 1, 96), (0.72, 1, 72), (0.61, 1, 48), (0.49, 1, 30),
        (0.38, 1, 18), (0.24, 1, 10), (0.76, 0, 84), (0.55, 0, 60),
        (0.51, 0, 36), (0.40, 0, 24), (0.22, 0, 12), (0.11, 0, 4),
    ),
    "mobile": (
        (0.90, 1, 96), (0.75, 1, 72), (0.57, 1, 48), (0.46, 1, 30),
        (0.35, 1, 18), (0.20, 1, 10), (0.79, 0, 84), (0.59, 0, 60),
        (0.48, 0, 36), (0.37, 0, 24), (0.17, 0, 12), (0.06, 0, 4),
    ),
    "edge": (
        (0.91, 1, 96), (0.73, 1, 72), (0.54, 1, 48), (0.43, 1, 30),
        (0.31, 1, 18), (0.16, 1, 10), (0.83, 0, 84), (0.64, 0, 60),
        (0.52, 0, 36), (0.39, 0, 24), (0.21, 0, 12), (0.09, 0, 4),
    ),
    "cloud": (
        (0.87, 1, 96), (0.70, 1, 72), (0.56, 1, 48), (0.45, 1, 30),
        (0.34, 1, 18), (0.18, 1, 10), (0.74, 0, 84), (0.57, 0, 60),
        (0.50, 0, 36), (0.36, 0, 24), (0.16, 0, 12), (0.05, 0, 4),
    ),
}

PRIVACY_PROFILES = {
    "aggregate_counts": PrivacyProfile(
        key="aggregate_counts",
        label="Cohort aggregate counts",
        retained_fields=("cohort", "score_bin", "label", "count", "window"),
        raw_content_retained=False,
        stable_subject_id_retained=False,
        cohort_attribute_retained=True,
        retention=Q_(30, "day"),
        deletion_window=Q_(7, "day"),
    ),
    "pseudonymous_events": PrivacyProfile(
        key="pseudonymous_events",
        label="Pseudonymous scored events",
        retained_fields=("pseudonymous_id", "cohort", "score", "label", "timestamp"),
        raw_content_retained=False,
        stable_subject_id_retained=True,
        cohort_attribute_retained=True,
        retention=Q_(14, "day"),
        deletion_window=Q_(3, "day"),
    ),
    "full_review_trace": PrivacyProfile(
        key="full_review_trace",
        label="Full case-review trace",
        retained_fields=("subject_id", "cohort", "input", "score", "label", "explanation", "timestamp"),
        raw_content_retained=True,
        stable_subject_id_retained=True,
        cohort_attribute_retained=True,
        retention=Q_(90, "day"),
        deletion_window=Q_(21, "day"),
    ),
}

FAIRNESS_CRITERIA = (
    "demographic_parity",
    "equal_opportunity",
    "predictive_parity",
    "equalized_odds",
)


def _canonical_track(track_id: str) -> str:
    canonical = _TRACK_ALIASES.get(track_id, track_id)
    if canonical not in TRACK_PROFILES:
        raise ValueError(f"track_id must be one of {', '.join(TRACK_PROFILES)}")
    return canonical


def _fraction(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _positive_quantity(value: Any, unit: str, name: str):
    try:
        quantity = value.to(unit)
    except AttributeError:
        quantity = Q_(value, unit)
    except Exception as exc:
        raise ValueError(f"{name} must be compatible with {unit}") from exc
    if not math.isfinite(float(quantity.magnitude)) or quantity.magnitude <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return quantity


def track_profile(track_id: str) -> dict[str, Any]:
    """Return a copy of an illustrative track profile."""
    canonical = _canonical_track(track_id)
    return {"track_id": canonical, **TRACK_PROFILES[canonical]}


def scored_fixture(track_id: str) -> tuple[ScoredCase, ...]:
    """Return coherent scores and labels for two named subgroups."""
    canonical = _canonical_track(track_id)
    subgroups = TRACK_PROFILES[canonical]["subgroups"]
    cases: list[ScoredCase] = []
    for group_index, (subgroup, rows) in enumerate(zip(subgroups, (_REFERENCE_CASES, _AFFECTED_CASES[canonical]))):
        for index, (score, label, age) in enumerate(rows):
            # Coprime steps give a deterministic, non-prefix sampling order.
            rank = ((index * 7 + group_index * 5) % len(rows)) + 1
            cases.append(ScoredCase(f"{canonical}-{group_index}-{index}", subgroup, label, score, age, rank))
    return tuple(cases)


def _rate(numerator: int, denominator: int) -> float | None:
    """Return an empirical rate, preserving an empty denominator as undefined."""
    return numerator / denominator if denominator else None


def _absolute_gap(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return abs(first - second)


def _metrics(cases: Iterable[ScoredCase], threshold: float) -> tuple[SubgroupMetrics, ...]:
    records = tuple(cases)
    groups = tuple(dict.fromkeys(case.subgroup for case in records))
    results = []
    for subgroup in groups:
        group = tuple(case for case in records if case.subgroup == subgroup)
        tp = sum(case.true_label == 1 and case.score >= threshold for case in group)
        fp = sum(case.true_label == 0 and case.score >= threshold for case in group)
        tn = sum(case.true_label == 0 and case.score < threshold for case in group)
        fn = sum(case.true_label == 1 and case.score < threshold for case in group)
        counts = ConfusionCounts(tp, fp, tn, fn)
        results.append(
            SubgroupMetrics(
                subgroup=subgroup,
                counts=counts,
                selection_rate=_rate(tp + fp, counts.total),
                true_positive_rate=_rate(tp, tp + fn),
                false_positive_rate=_rate(fp, fp + tn),
                positive_predictive_value=_rate(tp, tp + fp),
            )
        )
    return tuple(results)


def assess_fairness(
    track_id: str,
    *,
    threshold: float,
    criterion: str,
    allowed_gap: float,
    cases: Iterable[ScoredCase] | None = None,
) -> FairnessAssessment:
    """Calculate a chosen fairness criterion from scores and true labels.

    ``criterion`` and ``allowed_gap`` classify the observed metrics.  They do
    not change scores, labels, or any confusion count.
    """
    canonical = _canonical_track(track_id)
    threshold = _fraction(threshold, "threshold")
    allowed_gap = _fraction(allowed_gap, "allowed_gap")
    if criterion not in FAIRNESS_CRITERIA:
        raise ValueError(f"criterion must be one of {', '.join(FAIRNESS_CRITERIA)}")
    records = tuple(scored_fixture(canonical) if cases is None else cases)
    if not records:
        raise ValueError("cases must contain at least one scored example")
    metrics = _metrics(records, threshold)
    if len(metrics) != 2:
        raise ValueError("cases must contain exactly two subgroups")
    first, second = metrics
    opportunity_gap = _absolute_gap(first.true_positive_rate, second.true_positive_rate)
    false_positive_gap = _absolute_gap(first.false_positive_rate, second.false_positive_rate)
    equalized_odds_gap = (
        max(opportunity_gap, false_positive_gap)
        if opportunity_gap is not None and false_positive_gap is not None
        else None
    )
    gaps = {
        "demographic_parity": abs(first.selection_rate - second.selection_rate),
        "equal_opportunity": opportunity_gap,
        "predictive_parity": _absolute_gap(
            first.positive_predictive_value, second.positive_predictive_value
        ),
        "equalized_odds": equalized_odds_gap,
    }
    gap = gaps[criterion]
    return FairnessAssessment(
        track_id=canonical,
        threshold=threshold,
        criterion=criterion,
        allowed_gap=allowed_gap,
        criterion_gap=gap,
        criterion_passes=gap is not None and gap <= allowed_gap,
        subgroup_metrics=metrics,
        ground_truth_positive_cases=sum(case.true_label for case in records),
        predicted_positive_cases=sum(case.score >= threshold for case in records),
        false_negative_cases=sum(case.true_label == 1 and case.score < threshold for case in records),
    )


def evaluate_capacity(
    track_id: str,
    *,
    explanation_share: float,
    review_share: float,
    horizon=Q_(8, "hour"),
    explanation_workers: int | None = None,
    reviewers: int | None = None,
    explanation_service_time=None,
    review_service_time=None,
    target_utilization: float = 0.85,
) -> CapacityResult:
    """Size explanation and human-review queues from workload and service time."""
    profile = track_profile(track_id)
    explanation_share = _fraction(explanation_share, "explanation_share")
    review_share = _fraction(review_share, "review_share")
    target_utilization = _fraction(target_utilization, "target_utilization")
    if target_utilization == 0:
        raise ValueError("target_utilization must be positive")
    horizon = _positive_quantity(horizon, "hour", "horizon")
    explanation_workers = profile["explanation_workers"] if explanation_workers is None else explanation_workers
    reviewers = profile["reviewers"] if reviewers is None else reviewers
    if isinstance(explanation_workers, bool) or not isinstance(explanation_workers, int) or explanation_workers <= 0:
        raise ValueError("explanation_workers must be a positive integer")
    if isinstance(reviewers, bool) or not isinstance(reviewers, int) or reviewers <= 0:
        raise ValueError("reviewers must be a positive integer")
    explanation_service_time = _positive_quantity(
        profile["explanation_service_time"] if explanation_service_time is None else explanation_service_time,
        "second/count",
        "explanation_service_time",
    )
    review_service_time = _positive_quantity(
        profile["review_service_time"] if review_service_time is None else review_service_time,
        "second/count",
        "review_service_time",
    )

    workload = profile["events_per_hour"]
    explanation_arrivals = (workload * explanation_share).to("count/hour")
    review_arrivals = (workload * review_share).to("count/hour")
    explanation_capacity = (
        Q_(explanation_workers * target_utilization, "count") / explanation_service_time
    ).to("count/hour")
    review_capacity = (Q_(reviewers * target_utilization, "count") / review_service_time).to("count/hour")
    explanation_backlog = (
        max(0.0, (explanation_arrivals - explanation_capacity).to("count/hour").magnitude)
        * Q_(1, "count/hour")
        * horizon
    ).to("count")
    review_backlog = (
        max(0.0, (review_arrivals - review_capacity).to("count/hour").magnitude)
        * Q_(1, "count/hour")
        * horizon
    ).to("count")
    explanation_utilization = (
        explanation_arrivals / explanation_capacity
    ).to("").magnitude
    review_utilization = (review_arrivals / review_capacity).to("").magnitude
    return CapacityResult(
        track_id=profile["track_id"],
        explanation_share=explanation_share,
        review_share=review_share,
        explanation_arrivals=explanation_arrivals,
        explanation_capacity=explanation_capacity,
        explanation_backlog=explanation_backlog,
        review_arrivals=review_arrivals,
        review_capacity=review_capacity,
        review_backlog=review_backlog,
        explanation_utilization=explanation_utilization,
        review_utilization=review_utilization,
        feasible=explanation_utilization <= 1.0 and review_utilization <= 1.0,
    )


def audit_evidence(
    track_id: str,
    *,
    threshold: float,
    criterion: str,
    allowed_gap: float,
    sampling_fraction: float,
    label_delay=Q_(24, "hour"),
    cases: Iterable[ScoredCase] | None = None,
) -> AuditEvidence:
    """Apply delayed-label eligibility and deterministic cohort sampling.

    The returned population assessment is calculated from the entire fixture
    and is independent of the observation plan.
    """
    canonical = _canonical_track(track_id)
    records = tuple(scored_fixture(canonical) if cases is None else cases)
    sampling_fraction = _fraction(sampling_fraction, "sampling_fraction")
    delay = _positive_quantity(label_delay, "hour", "label_delay")
    population = assess_fairness(
        canonical, threshold=threshold, criterion=criterion, allowed_gap=allowed_gap, cases=records
    )
    eligible = tuple(case for case in records if case.label_age_hours >= delay.magnitude)
    rank_limit = math.floor(sampling_fraction * 12 + 1e-12)
    sampled = tuple(case for case in eligible if case.sample_rank <= rank_limit)
    sampled_subgroups = tuple(dict.fromkeys(case.subgroup for case in sampled))
    limitations: list[str] = []
    observed = None
    if len(sampled_subgroups) < 2:
        limitations.append("sample does not contain both subgroups")
    elif not all(any(case.true_label == 1 for case in sampled if case.subgroup == group) for group in sampled_subgroups):
        limitations.append("a sampled subgroup has no labeled positive case")
    else:
        observed = assess_fairness(
            canonical, threshold=threshold, criterion=criterion, allowed_gap=allowed_gap, cases=sampled
        )
        if observed.criterion_gap is None:
            limitations.append(
                "the chosen fairness criterion is undefined for the sampled class counts"
            )
    if len(eligible) < len(records):
        limitations.append(f"{len(records) - len(eligible)} labels are still delayed")
    if len(sampled) < len(eligible):
        limitations.append(f"{len(eligible) - len(sampled)} eligible cases were not sampled")
    return AuditEvidence(
        track_id=canonical,
        threshold=threshold,
        sampling_fraction=sampling_fraction,
        label_delay=delay,
        eligible_cases=len(eligible),
        sampled_cases=len(sampled),
        sampled_subgroups=sampled_subgroups,
        observed_assessment=observed,
        population_assessment=population,
        evidence_complete=len(sampled) == len(records),
        limitations=tuple(limitations),
    )


def check_privacy_profile(
    profile_key: str,
    *,
    track_id: str | None = None,
    require_case_review: bool,
    max_retention=Q_(30, "day"),
    max_deletion_window=Q_(7, "day"),
) -> PrivacyCheck:
    """Check explicit data-handling obligations and evidence availability.

    Passing these specified checks is not a privacy guarantee and carries no
    numerical privacy interpretation.
    """
    if profile_key not in PRIVACY_PROFILES:
        raise ValueError(f"profile_key must be one of {', '.join(PRIVACY_PROFILES)}")
    profile = PRIVACY_PROFILES[profile_key]
    max_retention = _positive_quantity(max_retention, "day", "max_retention")
    max_deletion_window = _positive_quantity(max_deletion_window, "day", "max_deletion_window")
    canonical_track = _canonical_track(track_id) if track_id is not None else ""
    fleet_scope = (
        TRACK_PROFILES[canonical_track]["fleet_role"]
        if canonical_track
        else "fleet telemetry"
    )
    subgroup = profile.cohort_attribute_retained and "label" in profile.retained_fields
    case_review = subgroup and "score" in profile.retained_fields and profile.stable_subject_id_retained
    retention_ok = profile.retention <= max_retention
    deletion_ok = profile.deletion_window <= max_deletion_window
    violations = []
    if not subgroup:
        violations.append("profile cannot reconstruct labeled subgroup evidence")
    if require_case_review and not case_review:
        violations.append("profile cannot support case-level review")
    if not retention_ok:
        violations.append("retention exceeds the specified maximum")
    if not deletion_ok:
        violations.append("deletion window exceeds the specified maximum")
    return PrivacyCheck(
        profile=profile,
        supports_subgroup_audit=subgroup,
        supports_case_review=case_review,
        retention_ok=retention_ok,
        deletion_ok=deletion_ok,
        passes_requirements=not violations,
        violations=tuple(violations),
        track_id=canonical_track,
        scope=fleet_scope,
    )


def mechanical_gate(
    assessment: FairnessAssessment,
    capacity: CapacityResult,
    audit: AuditEvidence,
    privacy: PrivacyCheck,
) -> MechanicalGate:
    """Combine observable requirements without making a release decision."""
    violations = []
    if assessment.criterion_gap is None:
        violations.append("chosen fairness criterion is undefined for the population class counts")
    elif not assessment.criterion_passes:
        violations.append("chosen fairness criterion exceeds its allowed gap")
    if not capacity.feasible:
        violations.append("explanation or review arrivals exceed service capacity")
    if audit.observed_assessment is None or audit.observed_assessment.criterion_gap is None:
        violations.append("audit sample cannot estimate the chosen subgroup metric")
    if not privacy.passes_requirements:
        violations.append("privacy evidence profile misses a specified requirement")
    return MechanicalGate(not violations, tuple(violations))


def record_release_decision(
    gate: MechanicalGate,
    *,
    choice: str,
    rejected_alternative: str,
    owner: str,
    remedy: str,
    reevaluation_trigger: str,
) -> ReleaseDecision:
    """Record learner governance choices without changing mechanical evidence."""
    allowed_choices = {"release", "restrict", "defer"}
    if choice not in allowed_choices or rejected_alternative not in allowed_choices:
        raise ValueError("choice and rejected_alternative must be release, restrict, or defer")
    missing = []
    if choice == rejected_alternative:
        missing.append("distinct rejected alternative")
    for label, value in (
        ("owner", owner),
        ("remedy", remedy),
        ("reevaluation trigger", reevaluation_trigger),
    ):
        if not str(value).strip():
            missing.append(label)
    return ReleaseDecision(
        choice=choice,
        rejected_alternative=rejected_alternative,
        owner=str(owner).strip(),
        remedy=str(remedy).strip(),
        reevaluation_trigger=str(reevaluation_trigger).strip(),
        evidence_gate=gate,
        decision_record_complete=not missing,
        missing_fields=tuple(missing),
    )


__all__ = [
    "AuditEvidence",
    "CapacityResult",
    "ConfusionCounts",
    "FAIRNESS_CRITERIA",
    "FairnessAssessment",
    "MechanicalGate",
    "PRIVACY_PROFILES",
    "PrivacyCheck",
    "PrivacyProfile",
    "ReleaseDecision",
    "ScoredCase",
    "SubgroupMetrics",
    "TRACK_PROFILES",
    "assess_fairness",
    "audit_evidence",
    "check_privacy_profile",
    "evaluate_capacity",
    "mechanical_gate",
    "record_release_decision",
    "scored_fixture",
    "track_profile",
]
