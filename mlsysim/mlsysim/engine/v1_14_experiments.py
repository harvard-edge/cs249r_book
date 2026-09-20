"""Evidence-driven operations experiments for Volume I, Chapter 14.

Every fixture in this module is an explicitly illustrative scenario, not a
measurement of a named production system.  Model outcomes, proxy observations,
and canary labels are supplied independently so that a monitoring threshold can
classify evidence but cannot change the underlying behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import json
from math import ceil, floor, sqrt
from typing import Any, Literal, Mapping, Sequence

from ..core.types import Quantity
from ..core.units import Q_


TrackId = Literal["tinyml", "mobile", "edge", "cloud"]
RetrainingPolicy = Literal["scheduled", "evidence_triggered"]
CanaryDecision = Literal["promote", "rollback", "continue"]

TRACKS: tuple[TrackId, ...] = (
    "tinyml",
    "mobile",
    "edge",
    "cloud",
)


def _hours(value: Quantity, name: str, *, allow_zero: bool = False) -> float:
    hours = value.to("hour").magnitude
    if hours < 0 or (hours == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return float(hours)


def _usd(value: Quantity, name: str, *, allow_zero: bool = True) -> float:
    amount = value.to("USD").magnitude
    if amount < 0 or (amount == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return float(amount)


def _validate_ratio(value: float, name: str, *, inclusive: bool = True) -> None:
    valid = 0 <= value <= 1 if inclusive else 0 < value < 1
    if not valid:
        interval = "[0, 1]" if inclusive else "(0, 1)"
        raise ValueError(f"{name} must be in {interval}")


@dataclass(frozen=True)
class TemporalObservation:
    """One proxy observation with its eventually available outcome label."""

    event_time: Quantity
    proxy_value: float
    outcome_quality: float


@dataclass(frozen=True)
class MonitoringScenario:
    observations: tuple[TemporalObservation, ...]
    label_delay: Quantity
    quality_floor: float
    cost_per_observation: Quantity
    cost_per_investigation: Quantity


@dataclass(frozen=True)
class MonitoringResult:
    sampled_event_times: tuple[Quantity, ...]
    alert_event_times: tuple[Quantity, ...]
    false_alert_event_times: tuple[Quantity, ...]
    missed_failure_event_times: tuple[Quantity, ...]
    failure_episode_starts: tuple[Quantity, ...]
    detection_delays: tuple[Quantity | None, ...]
    first_outcome_confirmation: Quantity | None
    observations_collected: int
    investigations: int
    telemetry_cost: Quantity

    @property
    def false_investigations(self) -> int:
        return len(self.false_alert_event_times)

    @property
    def missed_failures(self) -> int:
        return len(self.missed_failure_event_times)


def evaluate_monitoring(
    scenario: MonitoringScenario,
    *,
    proxy_threshold: float,
    sampling_interval: Quantity,
) -> MonitoringResult:
    """Compare proxy alerts with supplied, delayed outcome labels.

    An alert is one sampled observation at or above ``proxy_threshold``.  A
    false investigation is an alert whose eventual same-time outcome remains
    at or above the quality floor.  A miss is a failed outcome without an alert.
    Detection delay is computed from each contiguous failure episode's start to
    its first on-or-after alert; ``None`` records an undetected episode.
    """

    if proxy_threshold < 0:
        raise ValueError("proxy_threshold must be non-negative")
    interval_hours = _hours(sampling_interval, "sampling_interval")
    label_delay_hours = _hours(scenario.label_delay, "label_delay", allow_zero=True)
    _validate_ratio(scenario.quality_floor, "quality_floor")
    if not scenario.observations:
        raise ValueError("scenario must contain observations")

    ordered = tuple(sorted(scenario.observations, key=lambda item: item.event_time.to("hour").magnitude))
    times = [float(item.event_time.to("hour").magnitude) for item in ordered]
    if len(set(times)) != len(times):
        raise ValueError("observation event times must be unique")
    if any(item.proxy_value < 0 for item in ordered):
        raise ValueError("proxy values must be non-negative")
    if any(not 0 <= item.outcome_quality <= 1 for item in ordered):
        raise ValueError("outcome quality values must be in [0, 1]")

    first_time = times[0]
    sampled = [
        item
        for item, time_hours in zip(ordered, times)
        if abs(((time_hours - first_time) / interval_hours) - round((time_hours - first_time) / interval_hours))
        < 1e-9
    ]
    alerts = [item for item in sampled if item.proxy_value >= proxy_threshold]
    false_alerts = [item for item in alerts if item.outcome_quality >= scenario.quality_floor]
    misses = [
        item
        for item in sampled
        if item.outcome_quality < scenario.quality_floor and item.proxy_value < proxy_threshold
    ]

    failure_starts: list[TemporalObservation] = []
    failure_episodes: list[tuple[TemporalObservation, float | None]] = []
    previous_failed = False
    active_failure: TemporalObservation | None = None
    for item in ordered:
        failed = item.outcome_quality < scenario.quality_floor
        if failed and not previous_failed:
            failure_starts.append(item)
            active_failure = item
        elif not failed and previous_failed and active_failure is not None:
            failure_episodes.append(
                (active_failure, float(item.event_time.to("hour").magnitude))
            )
            active_failure = None
        previous_failed = failed
    if active_failure is not None:
        failure_episodes.append((active_failure, None))

    alert_hours = [float(item.event_time.to("hour").magnitude) for item in alerts]
    delays: list[Quantity | None] = []
    for failure, episode_end in failure_episodes:
        failure_hour = float(failure.event_time.to("hour").magnitude)
        next_alert = next(
            (
                hour
                for hour in alert_hours
                if hour >= failure_hour and (episode_end is None or hour < episode_end)
            ),
            None,
        )
        delays.append(None if next_alert is None else Q_(next_alert - failure_hour, "hour"))

    first_confirmation = None
    failed = [item for item in ordered if item.outcome_quality < scenario.quality_floor]
    if failed:
        first_confirmation = failed[0].event_time + Q_(label_delay_hours, "hour")

    observation_cost = scenario.cost_per_observation * len(sampled)
    investigation_cost = scenario.cost_per_investigation * len(alerts)
    return MonitoringResult(
        sampled_event_times=tuple(item.event_time for item in sampled),
        alert_event_times=tuple(item.event_time for item in alerts),
        false_alert_event_times=tuple(item.event_time for item in false_alerts),
        missed_failure_event_times=tuple(item.event_time for item in misses),
        failure_episode_starts=tuple(item.event_time for item in failure_starts),
        detection_delays=tuple(delays),
        first_outcome_confirmation=first_confirmation,
        observations_collected=len(sampled),
        investigations=len(alerts),
        telemetry_cost=(observation_cost + investigation_cost).to("USD"),
    )


@dataclass(frozen=True)
class RetrainingScenario:
    event_times: tuple[Quantity, ...]
    baseline_outcomes: tuple[float, ...]
    candidate_outcome_traces: tuple[tuple[float, ...], ...]
    label_delay: Quantity
    quality_floor: float
    candidate_promotion_floor: float
    minimum_candidate_gain: float
    training_duration: Quantity
    resource_count: int
    retraining_cost: Quantity
    requests_per_hour: float
    value_per_correct_outcome: Quantity
    target_quality: float
    staleness_loss_growth: Quantity


@dataclass(frozen=True)
class RetrainingJob:
    candidate_index: int
    started_at: Quantity
    completed_at: Quantity
    promoted_at: Quantity | None
    candidate_quality_at_completion: float
    deployed_quality_before_promotion: float


@dataclass(frozen=True)
class RetrainingResult:
    policy: RetrainingPolicy
    deployed_outcomes: tuple[float, ...]
    jobs: tuple[RetrainingJob, ...]
    promotions: int
    compute_resource_hours: Quantity
    retraining_cost: Quantity
    stale_outcome_loss: Quantity
    total_operating_cost: Quantity
    approximate_optimal_interval: Quantity


def approximate_retraining_interval(
    retraining_cost: Quantity,
    staleness_loss_growth: Quantity,
) -> Quantity:
    """Return ``sqrt(2 C / L)`` for the stated quadratic-loss approximation."""

    cost = _usd(retraining_cost, "retraining_cost", allow_zero=False)
    growth = staleness_loss_growth.to("USD/hour**2").magnitude
    if growth <= 0:
        raise ValueError("staleness_loss_growth must be positive")
    return Q_(sqrt(2 * cost / growth), "hour")


def _validate_retraining_scenario(scenario: RetrainingScenario) -> tuple[float, ...]:
    if not scenario.event_times:
        raise ValueError("event_times must not be empty")
    times = tuple(float(value.to("hour").magnitude) for value in scenario.event_times)
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("event_times must be strictly increasing")
    expected = len(times)
    traces = (scenario.baseline_outcomes,) + scenario.candidate_outcome_traces
    if any(len(trace) != expected for trace in traces):
        raise ValueError("every outcome trace must match event_times")
    if any(not 0 <= value <= 1 for trace in traces for value in trace):
        raise ValueError("outcome quality values must be in [0, 1]")
    for name in ("quality_floor", "candidate_promotion_floor", "target_quality"):
        _validate_ratio(getattr(scenario, name), name)
    if scenario.minimum_candidate_gain < 0:
        raise ValueError("minimum_candidate_gain must be non-negative")
    _hours(scenario.label_delay, "label_delay", allow_zero=True)
    _hours(scenario.training_duration, "training_duration")
    if scenario.resource_count <= 0:
        raise ValueError("resource_count must be positive")
    if scenario.requests_per_hour < 0:
        raise ValueError("requests_per_hour must be non-negative")
    _usd(scenario.retraining_cost, "retraining_cost", allow_zero=False)
    _usd(scenario.value_per_correct_outcome, "value_per_correct_outcome")
    return times


def evaluate_retraining(
    scenario: RetrainingScenario,
    *,
    policy: RetrainingPolicy,
    scheduled_interval: Quantity | None = None,
) -> RetrainingResult:
    """Replay scheduled or delayed-outcome-triggered retraining.

    A job consumes resources as soon as it starts.  It can change the deployed
    version only when training completes and the supplied candidate trace meets
    both promotion gates.  The gates never alter any supplied outcome.
    """

    if policy not in {"scheduled", "evidence_triggered"}:
        raise ValueError(f"unsupported retraining policy: {policy}")
    times = _validate_retraining_scenario(scenario)
    if policy == "scheduled" and scheduled_interval is None:
        raise ValueError("scheduled_interval is required for scheduled policy")
    interval_hours = (
        _hours(scheduled_interval, "scheduled_interval") if scheduled_interval is not None else None
    )
    duration_hours = _hours(scenario.training_duration, "training_duration")
    label_delay_hours = _hours(scenario.label_delay, "label_delay", allow_zero=True)

    traces = (scenario.baseline_outcomes,) + scenario.candidate_outcome_traces
    active_trace = 0
    deployed: list[float] = []
    jobs: list[RetrainingJob] = []
    pending: tuple[int, float] | None = None
    next_candidate = 1
    next_scheduled = times[0] + interval_hours if interval_hours is not None else None
    triggered_failure_starts: set[int] = set()

    for index, now in enumerate(times):
        if pending is not None and pending[1] <= now:
            candidate_index, completion = pending
            candidate_quality = traces[candidate_index][index]
            deployed_before = traces[active_trace][index]
            promoted = (
                candidate_quality >= scenario.candidate_promotion_floor
                and candidate_quality - deployed_before >= scenario.minimum_candidate_gain
            )
            if promoted:
                active_trace = candidate_index
            jobs.append(
                RetrainingJob(
                    candidate_index=candidate_index - 1,
                    started_at=Q_(completion - duration_hours, "hour"),
                    completed_at=Q_(now, "hour"),
                    promoted_at=Q_(now, "hour") if promoted else None,
                    candidate_quality_at_completion=candidate_quality,
                    deployed_quality_before_promotion=deployed_before,
                )
            )
            pending = None

        deployed.append(traces[active_trace][index])
        should_start = False
        if pending is None and next_candidate < len(traces):
            if policy == "scheduled" and next_scheduled is not None and now >= next_scheduled:
                should_start = True
                while next_scheduled <= now:
                    next_scheduled += interval_hours  # type: ignore[operator]
            elif policy == "evidence_triggered":
                available_count = sum(
                    failure_time + label_delay_hours <= now
                    for failure_time in times[: len(deployed)]
                )
                available_outcomes = deployed[:available_count]
                failure_starts = [
                    failure_index
                    for failure_index, quality in enumerate(available_outcomes)
                    if quality < scenario.quality_floor
                    and (
                        failure_index == 0
                        or available_outcomes[failure_index - 1] >= scenario.quality_floor
                    )
                    and failure_index not in triggered_failure_starts
                ]
                if failure_starts:
                    triggered_failure_starts.add(failure_starts[0])
                    should_start = True
            if should_start:
                pending = (next_candidate, now + duration_hours)
                next_candidate += 1

    if pending is not None:
        candidate_index, completion = pending
        completion_index = next((i for i, time in enumerate(times) if time >= completion), len(times) - 1)
        candidate_quality = traces[candidate_index][completion_index]
        deployed_before = traces[active_trace][completion_index]
        jobs.append(
            RetrainingJob(
                candidate_index=candidate_index - 1,
                started_at=Q_(completion - duration_hours, "hour"),
                completed_at=Q_(completion, "hour"),
                promoted_at=None,
                candidate_quality_at_completion=candidate_quality,
                deployed_quality_before_promotion=deployed_before,
            )
        )

    stale_loss = 0 * Q_(1, "USD")
    for index in range(len(times) - 1):
        hours = times[index + 1] - times[index]
        quality_shortfall = max(0.0, scenario.target_quality - deployed[index])
        affected_outcomes = quality_shortfall * scenario.requests_per_hour * hours
        stale_loss += scenario.value_per_correct_outcome * affected_outcomes
    job_cost = scenario.retraining_cost * len(jobs)
    resource_hours = Q_(scenario.resource_count * duration_hours * len(jobs), "hour")
    return RetrainingResult(
        policy=policy,
        deployed_outcomes=tuple(deployed),
        jobs=tuple(jobs),
        promotions=sum(job.promoted_at is not None for job in jobs),
        compute_resource_hours=resource_hours,
        retraining_cost=job_cost.to("USD"),
        stale_outcome_loss=stale_loss.to("USD"),
        total_operating_cost=(job_cost + stale_loss).to("USD"),
        approximate_optimal_interval=approximate_retraining_interval(
            scenario.retraining_cost, scenario.staleness_loss_growth
        ),
    )


@dataclass(frozen=True)
class CanaryObservation:
    event_time: Quantity
    cohort: str
    baseline_success: bool
    candidate_success: bool


@dataclass(frozen=True)
class CanaryEvidencePolicy:
    minimum_candidate_labels: int
    minimum_baseline_labels: int
    minimum_labels_per_cohort: int
    required_cohorts: tuple[str, ...]
    promotion_margin: float
    rollback_margin: float


@dataclass(frozen=True)
class CanaryResult:
    canary_fraction: float
    exposed_requests: int
    exposed_failures: int
    candidate_labeled: int
    baseline_labeled: int
    candidate_successes: int
    baseline_successes: int
    candidate_observed_rate: float | None
    baseline_observed_rate: float | None
    candidate_cohort_counts: tuple[tuple[str, int], ...]
    baseline_cohort_counts: tuple[tuple[str, int], ...]
    evidence_requirements_met: bool
    evidence_ready_at: Quantity | None
    decision: CanaryDecision
    decision_time: Quantity | None


def _canary_snapshot(
    observations: Sequence[CanaryObservation],
    *,
    canary_fraction: float,
    policy: CanaryEvidencePolicy,
) -> CanaryResult:
    candidate: list[CanaryObservation] = []
    baseline: list[CanaryObservation] = []
    grouped: dict[float, list[CanaryObservation]] = {}
    for item in observations:
        grouped.setdefault(float(item.event_time.to("hour").magnitude), []).append(item)
    for time in sorted(grouped):
        window = grouped[time]
        canary_count = floor(len(window) * canary_fraction)
        candidate.extend(window[:canary_count])
        baseline.extend(window[canary_count:])

    candidate_counts = {cohort: 0 for cohort in policy.required_cohorts}
    baseline_counts = {cohort: 0 for cohort in policy.required_cohorts}
    for item in candidate:
        candidate_counts[item.cohort] = candidate_counts.get(item.cohort, 0) + 1
    for item in baseline:
        baseline_counts[item.cohort] = baseline_counts.get(item.cohort, 0) + 1
    enough = (
        len(candidate) >= policy.minimum_candidate_labels
        and len(baseline) >= policy.minimum_baseline_labels
        and all(candidate_counts.get(cohort, 0) >= policy.minimum_labels_per_cohort for cohort in policy.required_cohorts)
        and all(baseline_counts.get(cohort, 0) >= policy.minimum_labels_per_cohort for cohort in policy.required_cohorts)
    )
    candidate_successes = sum(item.candidate_success for item in candidate)
    baseline_successes = sum(item.baseline_success for item in baseline)
    candidate_rate = candidate_successes / len(candidate) if candidate else None
    baseline_rate = baseline_successes / len(baseline) if baseline else None
    decision: CanaryDecision = "continue"
    if enough and candidate_rate is not None and baseline_rate is not None:
        difference = candidate_rate - baseline_rate
        if difference >= policy.promotion_margin:
            decision = "promote"
        elif difference <= -policy.rollback_margin:
            decision = "rollback"
    return CanaryResult(
        canary_fraction=canary_fraction,
        exposed_requests=len(candidate),
        exposed_failures=sum(not item.candidate_success for item in candidate),
        candidate_labeled=len(candidate),
        baseline_labeled=len(baseline),
        candidate_successes=candidate_successes,
        baseline_successes=baseline_successes,
        candidate_observed_rate=candidate_rate,
        baseline_observed_rate=baseline_rate,
        candidate_cohort_counts=tuple(sorted(candidate_counts.items())),
        baseline_cohort_counts=tuple(sorted(baseline_counts.items())),
        evidence_requirements_met=enough,
        evidence_ready_at=None,
        decision=decision,
        decision_time=None,
    )


def evaluate_canary(
    observations: Sequence[CanaryObservation],
    *,
    canary_fraction: float,
    label_delay: Quantity,
    evidence_policy: CanaryEvidencePolicy,
    stop_on_decision: bool = True,
) -> CanaryResult:
    """Apply a descriptive minimum-evidence gate to supplied canary outcomes.

    The gate is an operational policy, not a statistical-significance test.
    It reports observed rates and cohort coverage without assuming IID samples.
    """

    _validate_ratio(canary_fraction, "canary_fraction")
    delay_hours = _hours(label_delay, "label_delay", allow_zero=True)
    if evidence_policy.minimum_candidate_labels <= 0 or evidence_policy.minimum_baseline_labels <= 0:
        raise ValueError("minimum label requirements must be positive")
    if evidence_policy.minimum_labels_per_cohort <= 0:
        raise ValueError("minimum_labels_per_cohort must be positive")
    if not evidence_policy.required_cohorts:
        raise ValueError("required_cohorts must not be empty")
    if evidence_policy.promotion_margin < 0 or evidence_policy.rollback_margin < 0:
        raise ValueError("evidence margins must be non-negative")
    if not observations:
        raise ValueError("observations must not be empty")

    ordered = tuple(sorted(observations, key=lambda item: item.event_time.to("hour").magnitude))
    decision_result: CanaryResult | None = None
    evidence_ready_at: Quantity | None = None
    for event_time in sorted({float(item.event_time.to("hour").magnitude) for item in ordered}):
        available_at = event_time + delay_hours
        available = [
            item
            for item in ordered
            if float(item.event_time.to("hour").magnitude) + delay_hours <= available_at
        ]
        snapshot = _canary_snapshot(available, canary_fraction=canary_fraction, policy=evidence_policy)
        if snapshot.evidence_requirements_met and evidence_ready_at is None:
            evidence_ready_at = Q_(available_at, "hour")
        if snapshot.decision != "continue" and decision_result is None:
            decision_result = CanaryResult(
                **{
                    **snapshot.__dict__,
                    "evidence_ready_at": evidence_ready_at,
                    "decision_time": Q_(available_at, "hour"),
                }
            )
            if stop_on_decision:
                break

    final_snapshot = _canary_snapshot(ordered, canary_fraction=canary_fraction, policy=evidence_policy)
    if decision_result is not None:
        if stop_on_decision:
            return decision_result
        return CanaryResult(
            **{
                **final_snapshot.__dict__,
                "evidence_ready_at": evidence_ready_at,
                "decision": decision_result.decision,
                "decision_time": decision_result.decision_time,
            }
        )
    return CanaryResult(
        **{
            **final_snapshot.__dict__,
            "evidence_ready_at": evidence_ready_at,
            "decision_time": None,
        }
    )


@dataclass(frozen=True)
class IncidentPolicy:
    triage_duration: Quantity
    rollback_duration: Quantity
    validation_duration: Quantity
    recovery_objective: Quantity
    requests_per_hour: float
    affected_fraction: float
    harmful_request_cost: Quantity
    responder_hourly_cost: Quantity
    recovered_quality: float


@dataclass(frozen=True)
class IncidentReplay:
    incident_started_at: Quantity
    detected_at: Quantity | None
    triage_completed_at: Quantity | None
    rollback_completed_at: Quantity | None
    recovery_validated_at: Quantity | None
    mean_time_to_recovery: Quantity | None
    exposed_requests: int
    exposure_cost: Quantity
    response_labor_cost: Quantity
    total_incident_cost: Quantity
    recovered_quality: float
    verified_recovery: bool
    met_recovery_objective: bool


def replay_incident(
    monitoring_scenario: MonitoringScenario,
    *,
    proxy_threshold: float,
    sampling_interval: Quantity,
    policy: IncidentPolicy,
) -> IncidentReplay:
    """Replay detect, triage, rollback, and validation as sequential stages."""

    monitoring = evaluate_monitoring(
        monitoring_scenario,
        proxy_threshold=proxy_threshold,
        sampling_interval=sampling_interval,
    )
    if not monitoring.failure_episode_starts:
        raise ValueError("monitoring scenario has no incident to replay")
    incident_start = monitoring.failure_episode_starts[0]
    incident_hour = float(incident_start.to("hour").magnitude)
    incident_end = next(
        (
            float(item.event_time.to("hour").magnitude)
            for item in monitoring_scenario.observations
            if float(item.event_time.to("hour").magnitude) > incident_hour
            and item.outcome_quality >= monitoring_scenario.quality_floor
        ),
        None,
    )
    alerts = [
        value
        for value in monitoring.alert_event_times
        if float(value.to("hour").magnitude) >= incident_hour
        and (
            incident_end is None
            or float(value.to("hour").magnitude) < incident_end
        )
    ]
    detected = alerts[0] if alerts else None
    for name in ("triage_duration", "rollback_duration", "validation_duration", "recovery_objective"):
        _hours(getattr(policy, name), name, allow_zero=name != "recovery_objective")
    if policy.requests_per_hour < 0:
        raise ValueError("requests_per_hour must be non-negative")
    _validate_ratio(policy.affected_fraction, "affected_fraction")
    _validate_ratio(policy.recovered_quality, "recovered_quality")
    _usd(policy.harmful_request_cost, "harmful_request_cost")
    responder_rate = policy.responder_hourly_cost.to("USD/hour")
    if responder_rate.magnitude < 0:
        raise ValueError("responder_hourly_cost must be non-negative")

    if detected is None:
        final_hour = max(
            float(item.event_time.to("hour").magnitude)
            for item in monitoring_scenario.observations
        )
        exposed_hours = max(0.0, final_hour - incident_hour)
        exposed = ceil(exposed_hours * policy.requests_per_hour * policy.affected_fraction)
        exposure_cost = policy.harmful_request_cost * exposed
        zero = Q_(0, "USD")
        return IncidentReplay(
            incident_started_at=incident_start,
            detected_at=None,
            triage_completed_at=None,
            rollback_completed_at=None,
            recovery_validated_at=None,
            mean_time_to_recovery=None,
            exposed_requests=exposed,
            exposure_cost=exposure_cost.to("USD"),
            response_labor_cost=zero,
            total_incident_cost=exposure_cost.to("USD"),
            recovered_quality=policy.recovered_quality,
            verified_recovery=False,
            met_recovery_objective=False,
        )

    triage_complete = detected + policy.triage_duration
    rollback_complete = triage_complete + policy.rollback_duration
    recovery_validated = rollback_complete + policy.validation_duration
    mttr = recovery_validated - incident_start
    exposure_hours = max(0.0, (rollback_complete - incident_start).to("hour").magnitude)
    exposed = ceil(exposure_hours * policy.requests_per_hour * policy.affected_fraction)
    exposure_cost = policy.harmful_request_cost * exposed
    response_duration = policy.triage_duration + policy.rollback_duration + policy.validation_duration
    labor_cost = responder_rate * response_duration.to("hour")
    verified = policy.recovered_quality >= monitoring_scenario.quality_floor
    return IncidentReplay(
        incident_started_at=incident_start,
        detected_at=detected,
        triage_completed_at=triage_complete,
        rollback_completed_at=rollback_complete,
        recovery_validated_at=recovery_validated,
        mean_time_to_recovery=mttr.to("hour"),
        exposed_requests=exposed,
        exposure_cost=exposure_cost.to("USD"),
        response_labor_cost=labor_cost.to("USD"),
        total_incident_cost=(exposure_cost + labor_cost).to("USD"),
        recovered_quality=policy.recovered_quality,
        verified_recovery=verified,
        met_recovery_objective=verified and mttr <= policy.recovery_objective,
    )


@dataclass(frozen=True)
class OperationsScenario:
    track_id: TrackId
    label: str
    scenario_note: str
    monitoring: MonitoringScenario
    retraining: RetrainingScenario
    canary_observations: tuple[CanaryObservation, ...]
    canary_policy: CanaryEvidencePolicy
    incident_policy: IncidentPolicy


def _canary_fixture(scale: int, candidate_pattern: tuple[bool, ...]) -> tuple[CanaryObservation, ...]:
    cohorts = ("majority", "minority")
    baseline_pattern = (True, True, True, False, True, True, False, True)
    rows: list[CanaryObservation] = []
    for window in range(6):
        for index in range(20 * scale):
            pattern_index = window * 20 * scale + index
            rows.append(
                CanaryObservation(
                    event_time=Q_(window * 2, "hour"),
                    cohort=cohorts[index % len(cohorts)],
                    baseline_success=baseline_pattern[pattern_index % len(baseline_pattern)],
                    candidate_success=candidate_pattern[pattern_index % len(candidate_pattern)],
                )
            )
    return tuple(rows)


def _build_scenario(
    track_id: TrackId,
    label: str,
    *,
    time_step_hours: float,
    label_delay_hours: float,
    requests_per_hour: float,
    retraining_hours: float,
    resources: int,
    retraining_cost_usd: float,
    observation_cost_usd: float,
    investigation_cost_usd: float,
    canary_scale: int,
    candidate_pattern: tuple[bool, ...],
    response_minutes: tuple[float, float, float],
) -> OperationsScenario:
    times = tuple(Q_(index * time_step_hours, "hour") for index in range(12))
    proxy = (0.04, 0.08, 0.12, 0.18, 0.24, 0.33, 0.43, 0.39, 0.27, 0.16, 0.11, 0.08)
    baseline = (0.94, 0.94, 0.93, 0.92, 0.90, 0.87, 0.84, 0.86, 0.89, 0.91, 0.92, 0.92)
    candidate_one = (0.94, 0.94, 0.93, 0.92, 0.91, 0.91, 0.92, 0.93, 0.93, 0.94, 0.94, 0.94)
    candidate_two = (0.94, 0.94, 0.93, 0.92, 0.90, 0.89, 0.88, 0.88, 0.89, 0.90, 0.90, 0.91)
    monitoring = MonitoringScenario(
        observations=tuple(
            TemporalObservation(time, proxy_value, quality)
            for time, proxy_value, quality in zip(times, proxy, baseline)
        ),
        label_delay=Q_(label_delay_hours, "hour"),
        quality_floor=0.88,
        cost_per_observation=Q_(observation_cost_usd, "USD"),
        cost_per_investigation=Q_(investigation_cost_usd, "USD"),
    )
    retraining = RetrainingScenario(
        event_times=times,
        baseline_outcomes=baseline,
        candidate_outcome_traces=(candidate_one, candidate_two),
        label_delay=Q_(label_delay_hours, "hour"),
        quality_floor=0.88,
        candidate_promotion_floor=0.90,
        minimum_candidate_gain=0.02,
        training_duration=Q_(retraining_hours, "hour"),
        resource_count=resources,
        retraining_cost=Q_(retraining_cost_usd, "USD"),
        requests_per_hour=requests_per_hour,
        value_per_correct_outcome=Q_(0.02, "USD"),
        target_quality=0.93,
        staleness_loss_growth=Q_(max(0.01, requests_per_hour * 0.0001), "USD/hour**2"),
    )
    canary_policy = CanaryEvidencePolicy(
        minimum_candidate_labels=20 * canary_scale,
        minimum_baseline_labels=20 * canary_scale,
        minimum_labels_per_cohort=8 * canary_scale,
        required_cohorts=("majority", "minority"),
        promotion_margin=0.02,
        rollback_margin=0.02,
    )
    triage, rollback, validation = response_minutes
    incident_policy = IncidentPolicy(
        triage_duration=Q_(triage, "minute"),
        rollback_duration=Q_(rollback, "minute"),
        validation_duration=Q_(validation, "minute"),
        recovery_objective=Q_(2 * time_step_hours, "hour"),
        requests_per_hour=requests_per_hour,
        affected_fraction=0.10,
        harmful_request_cost=Q_(0.05, "USD"),
        responder_hourly_cost=Q_(90, "USD/hour"),
        recovered_quality=0.93,
    )
    return OperationsScenario(
        track_id=track_id,
        label=label,
        scenario_note="Illustrative outcome-labeled temporal fixture; not a production measurement.",
        monitoring=monitoring,
        retraining=retraining,
        canary_observations=_canary_fixture(canary_scale, candidate_pattern),
        canary_policy=canary_policy,
        incident_policy=incident_policy,
    )


SCENARIOS: dict[TrackId, OperationsScenario] = {
    "tinyml": _build_scenario(
        "tinyml", "TinyML wearable", time_step_hours=12, label_delay_hours=36,
        requests_per_hour=120, retraining_hours=6, resources=1, retraining_cost_usd=18,
        observation_cost_usd=0.002, investigation_cost_usd=12, canary_scale=1,
        candidate_pattern=(True, True, True, True, True, False, True, True),
        response_minutes=(20, 25, 30),
    ),
    "mobile": _build_scenario(
        "mobile", "Mobile on-device model", time_step_hours=8, label_delay_hours=24,
        requests_per_hour=600, retraining_hours=4, resources=1, retraining_cost_usd=55,
        observation_cost_usd=0.004, investigation_cost_usd=25, canary_scale=1,
        candidate_pattern=(True, True, True, True, True, False, True, True),
        response_minutes=(15, 20, 25),
    ),
    "edge": _build_scenario(
        "edge", "Edge perception node", time_step_hours=4, label_delay_hours=12,
        requests_per_hour=2_400, retraining_hours=3, resources=2, retraining_cost_usd=240,
        observation_cost_usd=0.008, investigation_cost_usd=80, canary_scale=2,
        candidate_pattern=(True, True, True, False, True, False, True, True),
        response_minutes=(8, 10, 20),
    ),
    "cloud": _build_scenario(
        "cloud", "Cloud recommendation service", time_step_hours=2,
        label_delay_hours=6, requests_per_hour=20_000, retraining_hours=2,
        resources=8, retraining_cost_usd=1_200, observation_cost_usd=0.02,
        investigation_cost_usd=180, canary_scale=4,
        candidate_pattern=(True, True, False, True, True, False, True, False),
        response_minutes=(5, 5, 10),
    ),
}


def get_track_scenario(track_id: str) -> OperationsScenario:
    """Return one of the four immutable illustrative operations scenarios."""

    try:
        return SCENARIOS[track_id]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"unknown track_id: {track_id}; choose one of {TRACKS}") from exc


def to_jsonable(value: Any) -> Any:
    """Convert experiment dataclasses and Pint quantities to JSON data."""

    if hasattr(value, "to") and hasattr(value, "magnitude") and hasattr(value, "units"):
        magnitude = value.magnitude
        if hasattr(magnitude, "item"):
            magnitude = magnitude.item()
        return {"value": magnitude, "unit": f"{value.units:~}"}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported experiment value: {type(value).__name__}")


def serialize_experiment(value: Any) -> str:
    """Serialize experiment inputs or results with stable JSON ordering."""

    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def restore_experiment(serialized: str) -> Any:
    """Restore JSON data for inspection or deterministic replay."""

    return json.loads(serialized)


def monitoring_experiment(
    track_id: str,
    *,
    proxy_threshold: float,
    sampling_interval_hours: float,
) -> dict[str, Any]:
    """Return a JSON-ready monitoring replay with chart and evidence fields."""

    scenario = get_track_scenario(track_id)
    inputs = {
        "track_id": track_id,
        "proxy_threshold": proxy_threshold,
        "sampling_interval_hours": sampling_interval_hours,
    }
    result = evaluate_monitoring(
        scenario.monitoring,
        proxy_threshold=proxy_threshold,
        sampling_interval=Q_(sampling_interval_hours, "hour"),
    )
    return {
        "inputs": inputs,
        "time_hours": [row.event_time.to("hour").magnitude for row in scenario.monitoring.observations],
        "proxy_values": [row.proxy_value for row in scenario.monitoring.observations],
        "outcome_quality_pct": [100 * row.outcome_quality for row in scenario.monitoring.observations],
        "quality_floor_pct": 100 * scenario.monitoring.quality_floor,
        "alert_times_hours": [value.to("hour").magnitude for value in result.alert_event_times],
        "false_alert_times_hours": [value.to("hour").magnitude for value in result.false_alert_event_times],
        "missed_failure_times_hours": [value.to("hour").magnitude for value in result.missed_failure_event_times],
        "failure_start_times_hours": [value.to("hour").magnitude for value in result.failure_episode_starts],
        "detection_delay_hours": [
            None if value is None else value.to("hour").magnitude
            for value in result.detection_delays
        ],
        "first_outcome_confirmation_hours": (
            None
            if result.first_outcome_confirmation is None
            else result.first_outcome_confirmation.to("hour").magnitude
        ),
        "observations_collected": result.observations_collected,
        "investigations": result.investigations,
        "false_investigations": result.false_investigations,
        "missed_failures": result.missed_failures,
        "telemetry_cost_usd": result.telemetry_cost.to("USD").magnitude,
    }


def experiment_options(track_id: str) -> dict[str, Any]:
    """Return discrete learner controls derived from the selected scenario."""

    scenario = get_track_scenario(track_id)
    step = (
        scenario.monitoring.observations[1].event_time
        - scenario.monitoring.observations[0].event_time
    ).to("hour").magnitude
    return {
        "step_hours": step,
        "monitor_thresholds": {
            "Sensitive · 0.15": 0.15,
            "Balanced · 0.30": 0.30,
            "Insensitive · 0.40": 0.40,
        },
        "monitor_intervals": {
            f"Every {step:g} h": step,
            f"Every {2 * step:g} h": 2 * step,
            f"Every {3 * step:g} h": 3 * step,
        },
        "scheduled_intervals": {
            f"Every {3 * step:g} h": 3 * step,
            f"Every {5 * step:g} h": 5 * step,
            f"Every {7 * step:g} h": 7 * step,
        },
        "canary_candidates": {
            "No canary traffic": 0.0,
            "10% canary": 0.10,
            "25% canary": 0.25,
            "50% canary": 0.50,
        },
        "rollback_options": {
            "2 minutes": 2.0,
            "5 minutes": 5.0,
            "15 minutes": 15.0,
            "30 minutes": 30.0,
        },
    }


def retraining_experiment(
    track_id: str,
    *,
    policy: RetrainingPolicy,
    scheduled_interval_hours: float | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready retraining-policy replay."""

    scenario = get_track_scenario(track_id)
    inputs = {
        "track_id": track_id,
        "policy": policy,
        "scheduled_interval_hours": scheduled_interval_hours,
    }
    result = evaluate_retraining(
        scenario.retraining,
        policy=policy,
        scheduled_interval=(
            None if scheduled_interval_hours is None else Q_(scheduled_interval_hours, "hour")
        ),
    )
    return {
        "inputs": inputs,
        "time_hours": [value.to("hour").magnitude for value in scenario.retraining.event_times],
        "baseline_quality_pct": [100 * value for value in scenario.retraining.baseline_outcomes],
        "deployed_quality_pct": [100 * value for value in result.deployed_outcomes],
        "quality_floor_pct": 100 * scenario.retraining.quality_floor,
        "jobs": to_jsonable(result.jobs),
        "job_count": len(result.jobs),
        "promotions": result.promotions,
        "compute_resource_hours": result.compute_resource_hours.to("hour").magnitude,
        "retraining_cost_usd": result.retraining_cost.to("USD").magnitude,
        "stale_outcome_loss_usd": result.stale_outcome_loss.to("USD").magnitude,
        "total_operating_cost_usd": result.total_operating_cost.to("USD").magnitude,
        "approximate_optimal_interval_hours": result.approximate_optimal_interval.to("hour").magnitude,
    }


def canary_experiment(
    track_id: str,
    *,
    canary_fraction: float,
    stop_on_decision: bool = True,
) -> dict[str, Any]:
    """Return a JSON-ready canary replay under the track's evidence policy."""

    scenario = get_track_scenario(track_id)
    inputs = {
        "track_id": track_id,
        "canary_fraction": canary_fraction,
        "stop_on_decision": stop_on_decision,
    }
    result = evaluate_canary(
        scenario.canary_observations,
        canary_fraction=canary_fraction,
        label_delay=scenario.monitoring.label_delay,
        evidence_policy=scenario.canary_policy,
        stop_on_decision=stop_on_decision,
    )
    return {
        "inputs": inputs,
        "canary_fraction_pct": 100 * canary_fraction,
        "exposed_requests": result.exposed_requests,
        "exposed_failures": result.exposed_failures,
        "candidate_labeled": result.candidate_labeled,
        "baseline_labeled": result.baseline_labeled,
        "candidate_successes": result.candidate_successes,
        "baseline_successes": result.baseline_successes,
        "candidate_observed_rate_pct": (
            None if result.candidate_observed_rate is None else 100 * result.candidate_observed_rate
        ),
        "baseline_observed_rate_pct": (
            None if result.baseline_observed_rate is None else 100 * result.baseline_observed_rate
        ),
        "candidate_cohort_counts": dict(result.candidate_cohort_counts),
        "baseline_cohort_counts": dict(result.baseline_cohort_counts),
        "minimum_candidate_labels": scenario.canary_policy.minimum_candidate_labels,
        "minimum_baseline_labels": scenario.canary_policy.minimum_baseline_labels,
        "minimum_labels_per_cohort": scenario.canary_policy.minimum_labels_per_cohort,
        "required_cohorts": list(scenario.canary_policy.required_cohorts),
        "evidence_requirements_met": result.evidence_requirements_met,
        "evidence_ready_hours": (
            None if result.evidence_ready_at is None else result.evidence_ready_at.to("hour").magnitude
        ),
        "decision": result.decision,
        "decision_time_hours": (
            None if result.decision_time is None else result.decision_time.to("hour").magnitude
        ),
    }


def incident_experiment(
    track_id: str,
    *,
    proxy_threshold: float,
    sampling_interval_hours: float,
    rollback_minutes: float,
    affected_fraction: float,
) -> dict[str, Any]:
    """Return a JSON-ready combined monitoring and recovery replay."""

    scenario = get_track_scenario(track_id)
    inputs = {
        "track_id": track_id,
        "proxy_threshold": proxy_threshold,
        "sampling_interval_hours": sampling_interval_hours,
        "rollback_minutes": rollback_minutes,
        "affected_fraction": affected_fraction,
    }
    policy = replace(
        scenario.incident_policy,
        rollback_duration=Q_(rollback_minutes, "minute"),
        affected_fraction=affected_fraction,
    )
    result = replay_incident(
        scenario.monitoring,
        proxy_threshold=proxy_threshold,
        sampling_interval=Q_(sampling_interval_hours, "hour"),
        policy=policy,
    )

    def time_or_none(value: Quantity | None) -> float | None:
        return None if value is None else value.to("hour").magnitude

    return {
        "inputs": inputs,
        "detection_status": "detected" if result.detected_at is not None else "unavailable",
        "recovery_status": "verified" if result.verified_recovery else "unavailable",
        "incident_started_hours": result.incident_started_at.to("hour").magnitude,
        "detected_hours": time_or_none(result.detected_at),
        "triage_completed_hours": time_or_none(result.triage_completed_at),
        "rollback_completed_hours": time_or_none(result.rollback_completed_at),
        "recovery_validated_hours": time_or_none(result.recovery_validated_at),
        "mean_time_to_recovery_hours": time_or_none(result.mean_time_to_recovery),
        "recovery_objective_hours": policy.recovery_objective.to("hour").magnitude,
        "exposed_requests": result.exposed_requests,
        "exposure_cost_usd": result.exposure_cost.to("USD").magnitude,
        "response_labor_cost_usd": result.response_labor_cost.to("USD").magnitude,
        "total_incident_cost_usd": result.total_incident_cost.to("USD").magnitude,
        "recovered_quality_pct": 100 * result.recovered_quality,
        "verified_recovery": result.verified_recovery,
        "met_recovery_objective": result.met_recovery_objective,
    }


def replay_experiment(model_key: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one captured experiment from its exact JSON input mapping.

    This is the stable handoff API for reports and the Volume I capstone.
    """

    kwargs = dict(inputs)
    dispatch = {
        "v1_14.monitoring": monitoring_experiment,
        "v1_14.retraining": retraining_experiment,
        "v1_14.canary": canary_experiment,
        "v1_14.incident": incident_experiment,
    }
    try:
        runner = dispatch[model_key]
    except KeyError as exc:
        raise ValueError(f"unknown model_key: {model_key}; choose one of {tuple(dispatch)}") from exc
    return runner(**kwargs)


def replay_capture(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Replay every simulator arm stored in one EvidenceCapture dictionary.

    ``chosen_result`` remains separate from ``result`` so a hold/no-feasible
    decision can preserve its unchanged baseline while ``result`` records the
    tested rejected alternative.  This is the capstone-facing capture API.
    """

    model_key = snapshot.get("model_key")
    if not isinstance(model_key, str):
        raise ValueError("capture model_key must be a string")

    def replay_arm(name: str, arm: Any) -> dict[str, Any] | None:
        if arm is None:
            return None
        if not isinstance(arm, Mapping) or not isinstance(arm.get("inputs"), Mapping):
            raise ValueError(f"capture {name} must contain an inputs mapping")
        return replay_experiment(model_key, arm["inputs"])

    alternatives = snapshot.get("alternatives", ())
    if not isinstance(alternatives, Sequence) or isinstance(alternatives, (str, bytes)):
        raise ValueError("capture alternatives must be a sequence")
    return {
        "baseline": replay_arm("baseline", snapshot.get("baseline")),
        "result": replay_arm("result", snapshot.get("result")),
        "chosen_result": replay_arm("chosen_result", snapshot.get("chosen_result")),
        "alternatives": [
            replay_arm(f"alternatives[{index}]", arm)
            for index, arm in enumerate(alternatives)
        ],
        "decision": snapshot.get("decision"),
        "result_role": snapshot.get("result_role"),
    }


__all__ = [
    "TRACKS",
    "CanaryEvidencePolicy",
    "CanaryObservation",
    "CanaryResult",
    "IncidentPolicy",
    "IncidentReplay",
    "MonitoringResult",
    "MonitoringScenario",
    "OperationsScenario",
    "RetrainingJob",
    "RetrainingResult",
    "RetrainingScenario",
    "TemporalObservation",
    "approximate_retraining_interval",
    "evaluate_canary",
    "evaluate_monitoring",
    "evaluate_retraining",
    "get_track_scenario",
    "experiment_options",
    "incident_experiment",
    "monitoring_experiment",
    "replay_experiment",
    "replay_capture",
    "replay_incident",
    "restore_experiment",
    "retraining_experiment",
    "canary_experiment",
    "serialize_experiment",
    "to_jsonable",
]
