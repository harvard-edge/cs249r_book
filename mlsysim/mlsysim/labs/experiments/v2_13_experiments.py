"""Security and privacy experiments for Volume II, Chapter 13.

The module keeps four questions separate: which adversary a control covers,
what differential-privacy guarantee a release has, what utility was observed,
and what latency or deletion cost the system pays.  Scenario fixtures are
illustrative teaching inputs, not measurements of a named production system.

Only a bounded scalar mean with fixed-size replace-one adjacency is implemented
here.  This is a pure-epsilon Laplace mechanism.  Seeded draws are reproducible
classroom demonstrations only: a deterministic public seed is not suitable for
a production privacy-preserving release.  The module intentionally does not
implement or approximate a DP-SGD accountant.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from random import Random
from typing import Literal, Mapping, Sequence


Track = Literal["tinyml", "mobile", "edge", "cloud"]
Threat = Literal[
    "network_observer",
    "malicious_client",
    "curious_aggregator",
    "host_operator",
    "artifact_thief",
    "physical_attacker",
]
PrivacyEventKind = Literal[
    "dp_release",
    "fixed_release_postprocessing",
    "fixed_model_query",
    "new_private_data_access",
]


@dataclass(frozen=True)
class ThreatControl:
    """A control and the explicit adversaries that it addresses."""

    name: str
    covered_threats: frozenset[Threat]
    protected_boundary: str
    execution_tier: str = ""


@dataclass(frozen=True)
class ThreatCoverageResult:
    adversary: Threat
    control_name: str
    protected_boundary: str
    covered: bool
    uncovered_capabilities: tuple[Threat, ...]
    execution_tier: str = ""


def evaluate_threat_coverage(
    adversary: Threat,
    control: ThreatControl,
    *,
    required_capabilities: Sequence[Threat] = (),
) -> ThreatCoverageResult:
    """Evaluate categorical coverage without using performance as a proxy."""

    capabilities = frozenset((adversary, *required_capabilities))
    uncovered = tuple(sorted(capabilities - control.covered_threats))
    return ThreatCoverageResult(
        adversary=adversary,
        control_name=control.name,
        protected_boundary=control.protected_boundary,
        covered=not uncovered,
        uncovered_capabilities=uncovered,
        execution_tier=control.execution_tier,
    )


@dataclass(frozen=True)
class BoundedMeanRelease:
    """One release from a bounded-mean Laplace mechanism.

    ``adjacency`` states the exact neighboring relation used by the guarantee:
    two fixed-size datasets are adjacent when one bounded record is replaced.
    The global L1 sensitivity is therefore ``(upper - lower) / n``.
    """

    release_id: str
    seed: int
    release_mode: str
    production_safe_randomness: bool
    record_count: int
    lower_bound: float
    upper_bound: float
    epsilon: float
    adjacency: str
    true_mean: float
    sensitivity: float
    laplace_scale: float
    noise: float
    raw_release: float
    published_release: float
    output_clipped: bool


def _laplace_draw(scale: float, rng: Random) -> float:
    """Draw Laplace(0, scale) via inverse transform using the standard library."""

    centered_uniform = rng.random() - 0.5
    if centered_uniform == 0:
        return 0.0
    sign = 1.0 if centered_uniform > 0 else -1.0
    return -scale * sign * log(1.0 - 2.0 * abs(centered_uniform))


def release_bounded_mean(
    values: Sequence[float],
    *,
    lower_bound: float,
    upper_bound: float,
    epsilon: float,
    release_id: str,
    seed: int,
    clip_output: bool = True,
) -> BoundedMeanRelease:
    """Demonstrate a bounded-mean Laplace release under replace-one adjacency.

    Input records must already lie in the declared public bounds.  Clipping the
    released value is optional postprocessing and does not change epsilon.  The
    explicit seed makes experiments replayable for teaching and tests.  A
    deterministic public seed makes the sampled noise predictable, so callers
    must not use this function as production release randomness.
    """

    if not release_id:
        raise ValueError("release_id must be non-empty")
    if not values:
        raise ValueError("values must be non-empty")
    if not isfinite(lower_bound) or not isfinite(upper_bound):
        raise ValueError("bounds must be finite")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if any(not isfinite(value) for value in values):
        raise ValueError("every value must be finite")
    if any(value < lower_bound or value > upper_bound for value in values):
        raise ValueError("every value must lie within the declared public bounds")

    true_mean = sum(values) / len(values)
    sensitivity = (upper_bound - lower_bound) / len(values)
    scale = sensitivity / epsilon
    noise = _laplace_draw(scale, Random(seed))
    raw_release = true_mean + noise
    published_release = (
        min(upper_bound, max(lower_bound, raw_release))
        if clip_output
        else raw_release
    )
    return BoundedMeanRelease(
        release_id=release_id,
        seed=seed,
        release_mode="seeded classroom demonstration",
        production_safe_randomness=False,
        record_count=len(values),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        epsilon=epsilon,
        adjacency="fixed-size replace-one over one bounded record",
        true_mean=true_mean,
        sensitivity=sensitivity,
        laplace_scale=scale,
        noise=noise,
        raw_release=raw_release,
        published_release=published_release,
        output_clipped=clip_output,
    )


@dataclass(frozen=True)
class PrivacyEvent:
    """An explicit operation in a privacy-accounting history."""

    event_id: str
    kind: PrivacyEventKind
    epsilon: float = 0.0
    source_release_id: str | None = None
    description: str = ""


@dataclass(frozen=True)
class PrivacyCompositionResult:
    total_epsilon: float
    accounted_release_ids: tuple[str, ...]
    accounted_release_count: int
    zero_additional_training_loss_ids: tuple[str, ...]
    zero_additional_training_loss_count: int
    unguaranteed_private_access_ids: tuple[str, ...]
    unguaranteed_private_access_count: int
    guarantee_complete: bool
    adjacency: str
    composition_rule: str


def compose_privacy_events(
    events: Sequence[PrivacyEvent],
) -> PrivacyCompositionResult:
    """Apply basic sequential composition to explicit DP releases.

    Postprocessing and queries to an already released fixed DP model add zero
    training privacy loss.  A new access to private data must itself name a
    valid DP release; otherwise this function reports that the composed DP
    guarantee is incomplete rather than inventing an epsilon.
    """

    seen: set[str] = set()
    released: set[str] = set()
    accounted: list[str] = []
    zero_loss: list[str] = []
    unguaranteed: list[str] = []
    total = 0.0

    for event in events:
        if not event.event_id or event.event_id in seen:
            raise ValueError("event_id values must be non-empty and unique")
        seen.add(event.event_id)
        if not isfinite(event.epsilon):
            raise ValueError("event epsilon must be finite")
        if event.kind == "dp_release":
            if event.epsilon <= 0:
                raise ValueError("a dp_release must have positive epsilon")
            if event.source_release_id is not None:
                raise ValueError("a dp_release must not reference another release")
            released.add(event.event_id)
            accounted.append(event.event_id)
            total += event.epsilon
        elif event.kind in {"fixed_release_postprocessing", "fixed_model_query"}:
            if event.epsilon != 0:
                raise ValueError("postprocessing and fixed-model queries add zero epsilon")
            if event.source_release_id not in released:
                raise ValueError("fixed-output operations must reference an earlier DP release")
            zero_loss.append(event.event_id)
        elif event.kind == "new_private_data_access":
            if event.epsilon != 0:
                raise ValueError("unaccounted private access cannot be assigned an epsilon")
            unguaranteed.append(event.event_id)
        else:
            raise ValueError(f"unsupported privacy event kind: {event.kind}")

    return PrivacyCompositionResult(
        total_epsilon=total,
        accounted_release_ids=tuple(accounted),
        accounted_release_count=len(accounted),
        zero_additional_training_loss_ids=tuple(zero_loss),
        zero_additional_training_loss_count=len(zero_loss),
        unguaranteed_private_access_ids=tuple(unguaranteed),
        unguaranteed_private_access_count=len(unguaranteed),
        guarantee_complete=not unguaranteed,
        adjacency="fixed-size replace-one over one bounded record per release",
        composition_rule="basic sequential composition: epsilon_total = sum(epsilon_i)",
    )


@dataclass(frozen=True)
class SuppliedQualityObservation:
    """An illustrative or empirical observation supplied independently of DP math."""

    candidate_id: str
    task: str
    correct: int
    evaluated: int
    evidence_label: str

    @property
    def accuracy(self) -> float:
        if self.evaluated <= 0:
            raise ValueError("evaluated must be positive")
        if self.correct < 0 or self.correct > self.evaluated:
            raise ValueError("correct must be between zero and evaluated")
        return self.correct / self.evaluated


@dataclass(frozen=True)
class QualityObservationResult:
    candidate_id: str
    task: str
    correct: int
    evaluated: int
    observed_accuracy: float
    evidence_label: str


def evaluate_supplied_quality(
    observation: SuppliedQualityObservation,
) -> QualityObservationResult:
    """Expose a supplied outcome without deriving it from privacy parameters."""

    return QualityObservationResult(
        candidate_id=observation.candidate_id,
        task=observation.task,
        correct=observation.correct,
        evaluated=observation.evaluated,
        observed_accuracy=observation.accuracy,
        evidence_label=observation.evidence_label,
    )


@dataclass(frozen=True)
class TrustMechanism:
    name: str
    protected_threats: frozenset[Threat]
    fixed_overhead_ms: float
    per_request_overhead_ms: float
    execution_tier: str = ""


@dataclass(frozen=True)
class TrustLatencyResult:
    mechanism_name: str
    protected: bool
    total_latency_ms: float
    overhead_ms: float
    meets_deadline: bool
    execution_tier: str = ""


def evaluate_trust_latency(
    mechanism: TrustMechanism,
    *,
    adversary: Threat,
    base_latency_ms: float,
    requests_per_session: int,
    deadline_ms: float,
) -> TrustLatencyResult:
    """Amortize setup overhead while keeping protection categorical."""

    if base_latency_ms < 0 or mechanism.fixed_overhead_ms < 0:
        raise ValueError("latencies must be non-negative")
    if mechanism.per_request_overhead_ms < 0:
        raise ValueError("latencies must be non-negative")
    if requests_per_session <= 0 or deadline_ms <= 0:
        raise ValueError("requests_per_session and deadline_ms must be positive")
    overhead = (
        mechanism.fixed_overhead_ms / requests_per_session
        + mechanism.per_request_overhead_ms
    )
    total = base_latency_ms + overhead
    return TrustLatencyResult(
        mechanism_name=mechanism.name,
        protected=adversary in mechanism.protected_threats,
        total_latency_ms=total,
        overhead_ms=overhead,
        meets_deadline=total <= deadline_ms,
        execution_tier=mechanism.execution_tier,
    )


@dataclass(frozen=True)
class ArtifactCopy:
    artifact_id: str
    copied_from: str | None
    deletion_time_hours: float
    tracked: bool = True
    incident_evidence_events: int = 0


@dataclass(frozen=True)
class ArtifactDeletionStatus:
    artifact_id: str
    deletion_completed_at_hours: float | None
    within_sla: bool
    tracked: bool
    incident_evidence_events_lost: int


@dataclass(frozen=True)
class DeletionPropagationResult:
    source_artifact_id: str
    statuses: tuple[ArtifactDeletionStatus, ...]
    complete: bool
    completion_time_hours: float | None
    untracked_artifact_ids: tuple[str, ...]
    incident_evidence_events_lost: int


@dataclass(frozen=True)
class IncidentEvidence:
    event_id: str
    artifact_id: str
    age_hours: float


@dataclass(frozen=True)
class RetentionResult:
    retention_hours: float
    retained_event_ids: tuple[str, ...]
    retained_event_count: int
    expired_event_ids: tuple[str, ...]
    expired_event_count: int
    retained_events_by_artifact: tuple[tuple[str, int], ...]


def evaluate_retention(
    evidence: Sequence[IncidentEvidence], *, retention_hours: float
) -> RetentionResult:
    """Apply a time-based retention window to explicit incident evidence."""

    if retention_hours < 0:
        raise ValueError("retention_hours must be non-negative")
    event_ids = [event.event_id for event in evidence]
    if any(not event_id for event_id in event_ids) or len(set(event_ids)) != len(event_ids):
        raise ValueError("event_id values must be non-empty and unique")
    if any(not event.artifact_id for event in evidence):
        raise ValueError("artifact_id values must be non-empty")
    if any(event.age_hours < 0 for event in evidence):
        raise ValueError("age_hours must be non-negative")

    retained = tuple(
        event.event_id for event in evidence if event.age_hours <= retention_hours
    )
    expired = tuple(
        event.event_id for event in evidence if event.age_hours > retention_hours
    )
    counts: dict[str, int] = {}
    for event in evidence:
        if event.age_hours <= retention_hours:
            counts[event.artifact_id] = counts.get(event.artifact_id, 0) + 1
    return RetentionResult(
        retention_hours=retention_hours,
        retained_event_ids=retained,
        retained_event_count=len(retained),
        expired_event_ids=expired,
        expired_event_count=len(expired),
        retained_events_by_artifact=tuple(sorted(counts.items())),
    )


def propagate_deletion(
    artifacts: Sequence[ArtifactCopy],
    *,
    source_artifact_id: str,
    deletion_sla_hours: float,
) -> DeletionPropagationResult:
    """Traverse every descendant copy and calculate its deletion completion."""

    if deletion_sla_hours <= 0:
        raise ValueError("deletion_sla_hours must be positive")
    by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    if len(by_id) != len(artifacts) or "" in by_id:
        raise ValueError("artifact_id values must be non-empty and unique")
    if source_artifact_id not in by_id:
        raise ValueError("source_artifact_id is unknown")
    for artifact in artifacts:
        if artifact.deletion_time_hours < 0:
            raise ValueError("deletion_time_hours must be non-negative")
        if artifact.incident_evidence_events < 0:
            raise ValueError("incident_evidence_events must be non-negative")
        if artifact.copied_from is not None and artifact.copied_from not in by_id:
            raise ValueError(f"unknown parent artifact: {artifact.copied_from}")

    children: dict[str, list[str]] = {artifact_id: [] for artifact_id in by_id}
    for artifact in artifacts:
        if artifact.copied_from is not None:
            children[artifact.copied_from].append(artifact.artifact_id)

    reachable: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(artifact_id: str) -> None:
        if artifact_id in visiting:
            raise ValueError("artifact copy graph must be acyclic")
        if artifact_id in visited:
            return
        visiting.add(artifact_id)
        reachable.append(artifact_id)
        for child_id in sorted(children[artifact_id]):
            visit(child_id)
        visiting.remove(artifact_id)
        visited.add(artifact_id)

    visit(source_artifact_id)

    completion: dict[str, float | None] = {}
    statuses: list[ArtifactDeletionStatus] = []
    for artifact_id in reachable:
        artifact = by_id[artifact_id]
        if not artifact.tracked:
            completed_at = None
        elif artifact.copied_from is None or artifact_id == source_artifact_id:
            completed_at = artifact.deletion_time_hours
        else:
            parent_completed = completion[artifact.copied_from]
            completed_at = (
                None
                if parent_completed is None
                else parent_completed + artifact.deletion_time_hours
            )
        completion[artifact_id] = completed_at
        statuses.append(
            ArtifactDeletionStatus(
                artifact_id=artifact_id,
                deletion_completed_at_hours=completed_at,
                within_sla=completed_at is not None and completed_at <= deletion_sla_hours,
                tracked=artifact.tracked,
                incident_evidence_events_lost=(
                    artifact.incident_evidence_events if completed_at is not None else 0
                ),
            )
        )

    untracked = tuple(
        status.artifact_id for status in statuses if not status.tracked
    )
    complete = all(status.within_sla for status in statuses)
    completion_times = [
        status.deletion_completed_at_hours
        for status in statuses
        if status.deletion_completed_at_hours is not None
    ]
    all_completion_times_known = len(completion_times) == len(statuses)
    return DeletionPropagationResult(
        source_artifact_id=source_artifact_id,
        statuses=tuple(statuses),
        complete=complete,
        completion_time_hours=(
            max(completion_times) if all_completion_times_known else None
        ),
        untracked_artifact_ids=untracked,
        incident_evidence_events_lost=sum(
            status.incident_evidence_events_lost for status in statuses
        ),
    )


@dataclass(frozen=True)
class SecurityPrivacyScenario:
    track: Track
    label: str
    adversary: Threat
    base_latency_ms: float
    deadline_ms: float
    bounded_values: tuple[float, ...]
    lower_bound: float
    upper_bound: float
    artifact_copies: tuple[ArtifactCopy, ...]
    incident_evidence: tuple[IncidentEvidence, ...]
    deletion_sla_hours: float
    fleet_roles: tuple[str, ...] = ()
    adversary_locus: str = ""
    metric_units: str = ""
    path_description: str = ""


TRACK_SCENARIOS: Mapping[Track, SecurityPrivacyScenario] = {
    "tinyml": SecurityPrivacyScenario(
        track="tinyml",
        label="Wearable biosignal fleet",
        adversary="physical_attacker",
        base_latency_ms=18.0,
        deadline_ms=25.0,
        bounded_values=(62.0, 68.0, 71.0, 75.0, 79.0, 83.0),
        lower_bound=40.0,
        upper_bound=120.0,
        artifact_copies=(
            ArtifactCopy("device-window", None, 1.0, incident_evidence_events=3),
            ArtifactCopy("gateway-cache", "device-window", 4.0, incident_evidence_events=5),
            ArtifactCopy("support-export", "gateway-cache", 10.0, incident_evidence_events=2),
        ),
        incident_evidence=(
            IncidentEvidence("sensor-alert-new", "gateway-cache", 6.0),
            IncidentEvidence("sensor-alert-old", "support-export", 60.0),
        ),
        deletion_sla_hours=24.0,
        fleet_roles=(
            "Wearable MCU (sensor capture & local inference)",
            "Mobile phone / BLE gateway (cache & upload)",
            "Clinical backend (model retraining & analytics)",
        ),
        adversary_locus="wearable sensor MCU in physical possession of attacker",
        metric_units="bpm",
        path_description="wearable MCU sensor to mobile BLE gateway",
    ),
    "mobile": SecurityPrivacyScenario(
        track="mobile",
        label="Keyboard personalization fleet",
        adversary="curious_aggregator",
        base_latency_ms=32.0,
        deadline_ms=50.0,
        bounded_values=(0.42, 0.48, 0.51, 0.57, 0.63, 0.66),
        lower_bound=0.0,
        upper_bound=1.0,
        artifact_copies=(
            ArtifactCopy("client-update", None, 2.0, incident_evidence_events=1),
            ArtifactCopy("regional-buffer", "client-update", 8.0, incident_evidence_events=4),
            ArtifactCopy("debug-snapshot", "regional-buffer", 20.0, incident_evidence_events=6),
        ),
        incident_evidence=(
            IncidentEvidence("abuse-signal-new", "regional-buffer", 12.0),
            IncidentEvidence("abuse-signal-old", "debug-snapshot", 96.0),
        ),
        deletion_sla_hours=24.0,
        fleet_roles=(
            "Smartphone client (on-device keyboard inference & gradient computation)",
            "Regional aggregator (federated buffer & secure aggregation)",
            "Federated learning service (global model coordinator & release)",
        ),
        adversary_locus="regional aggregation server observing individual client updates",
        metric_units="confidence",
        path_description="client keystroke predictor to regional aggregation buffer",
    ),
    "edge": SecurityPrivacyScenario(
        track="edge",
        label="Roadside safety fleet",
        adversary="host_operator",
        base_latency_ms=14.0,
        deadline_ms=20.0,
        bounded_values=(8.0, 11.0, 12.0, 15.0, 17.0, 19.0),
        lower_bound=0.0,
        upper_bound=30.0,
        artifact_copies=(
            ArtifactCopy("incident-window", None, 1.0, incident_evidence_events=8),
            ArtifactCopy("site-cache", "incident-window", 3.0, incident_evidence_events=8),
            ArtifactCopy("regional-replay", "site-cache", 9.0, incident_evidence_events=12),
        ),
        incident_evidence=(
            IncidentEvidence("near-miss-new", "regional-replay", 8.0),
            IncidentEvidence("near-miss-old", "regional-replay", 72.0),
        ),
        deletion_sla_hours=18.0,
        fleet_roles=(
            "Roadside camera & lidar (high-rate sensor ingest)",
            "Roadside compute unit (edge perception & event telemetry)",
            "Regional traffic operations center (fleet monitoring & storage)",
        ),
        adversary_locus="roadside compute unit enclosure inspected by hostile host operator",
        metric_units="m/s",
        path_description="roadside camera to roadside edge perception node",
    ),
    "cloud": SecurityPrivacyScenario(
        track="cloud",
        label="Multi-tenant analytics service",
        adversary="host_operator",
        base_latency_ms=45.0,
        deadline_ms=75.0,
        bounded_values=(12.0, 18.0, 21.0, 27.0, 31.0, 39.0),
        lower_bound=0.0,
        upper_bound=50.0,
        artifact_copies=(
            ArtifactCopy("training-shard", None, 2.0, incident_evidence_events=2),
            ArtifactCopy("feature-cache", "training-shard", 6.0, incident_evidence_events=4),
            ArtifactCopy("backup-copy", "training-shard", 30.0, incident_evidence_events=7),
        ),
        incident_evidence=(
            IncidentEvidence("tenant-alert-new", "feature-cache", 18.0),
            IncidentEvidence("tenant-alert-old", "backup-copy", 120.0),
        ),
        deletion_sla_hours=24.0,
        fleet_roles=(
            "API gateway (request authentication & routing)",
            "Multi-tenant compute container (isolated tenant worker process)",
            "Shared storage & backup pool (persistent training shards & archives)",
        ),
        adversary_locus="cloud hypervisor / host operator inspecting tenant container memory",
        metric_units="queries/s",
        path_description="API gateway to multi-tenant worker container",
    ),
}


THREAT_CONTROLS: Mapping[str, ThreatControl] = {
    "transport": ThreatControl(
        "authenticated encrypted transport",
        frozenset({"network_observer"}),
        "communication in transit",
        execution_tier="Network transit (device ↔ gateway / gateway ↔ backend)",
    ),
    "secure_aggregation": ThreatControl(
        "secure aggregation",
        frozenset({"network_observer", "curious_aggregator"}),
        "individual client updates before aggregation",
        execution_tier="Aggregation gateway / backend pool",
    ),
    "confidential_runtime": ThreatControl(
        "confidential runtime",
        frozenset({"network_observer", "host_operator", "artifact_thief"}),
        "runtime memory and model artifact",
        execution_tier="Gateway enclave / confidential cloud VM",
    ),
    "locked_device": ThreatControl(
        "locked device execution",
        frozenset({"network_observer", "artifact_thief", "physical_attacker"}),
        "device storage, boot, and local execution",
        execution_tier="Device hardware (MCU / client root of trust)",
    ),
}


TRUST_MECHANISMS: Mapping[str, TrustMechanism] = {
    "transport": TrustMechanism(
        "session transport protection", frozenset({"network_observer"}), 15.0, 0.2
        , execution_tier="Network transit"
    ),
    "secure_aggregation": TrustMechanism(
        "secure aggregation round",
        frozenset({"network_observer", "curious_aggregator"}),
        24.0,
        1.5,
        execution_tier="Aggregation gateway / backend",
    ),
    "confidential_runtime": TrustMechanism(
        "confidential runtime session",
        frozenset({"network_observer", "host_operator", "artifact_thief"}),
        30.0,
        3.0,
        execution_tier="Gateway / cloud confidential VM",
    ),
    "locked_device": TrustMechanism(
        "locked device verification",
        frozenset({"network_observer", "artifact_thief", "physical_attacker"}),
        8.0,
        1.0,
        execution_tier="Device hardware (MCU / client)",
    ),
}


PRIVACY_EPSILON_OPTIONS: Mapping[Track, tuple[float, ...]] = {
    "tinyml": (0.25, 0.5, 1.0, 2.0),
    "mobile": (0.25, 0.5, 1.0, 2.0),
    "edge": (0.5, 1.0, 2.0, 4.0),
    "cloud": (0.5, 1.0, 2.0, 4.0),
}


TASK_DESCRIPTIONS: Mapping[Track, str] = {
    "tinyml": "Arrhythmia detection from wearable biosignals (illustrative 100-record cohort)",
    "mobile": "Next-word keystroke suggestion (illustrative 100-phrase evaluation)",
    "edge": "Near-miss pedestrian collision detection (illustrative 100-event dataset)",
    "cloud": "Multi-tenant query anomaly detection (illustrative 100-query batch)",
}


QUALITY_OBSERVATIONS: Mapping[Track, Mapping[float, SuppliedQualityObservation]] = {
    track: {
        epsilon: SuppliedQualityObservation(
            candidate_id=f"{track}-epsilon-{epsilon:g}",
            task=TASK_DESCRIPTIONS[track],
            correct=correct,
            evaluated=100,
            evidence_label="illustrative scenario outcome; not a measured benchmark",
        )
        for epsilon, correct in zip(PRIVACY_EPSILON_OPTIONS[track], (72, 78, 83, 86))
    }
    for track, scenario in TRACK_SCENARIOS.items()
}


def get_track_scenario(track: Track) -> SecurityPrivacyScenario:
    """Return an immutable illustrative scenario for one teaching track."""

    try:
        return TRACK_SCENARIOS[track]
    except KeyError as error:
        raise ValueError(f"unsupported track: {track}") from error


def get_threat_control(control_id: str) -> ThreatControl:
    try:
        return THREAT_CONTROLS[control_id]
    except KeyError as error:
        raise ValueError(f"unsupported threat control: {control_id}") from error


def get_trust_mechanism(mechanism_id: str) -> TrustMechanism:
    try:
        return TRUST_MECHANISMS[mechanism_id]
    except KeyError as error:
        raise ValueError(f"unsupported trust mechanism: {mechanism_id}") from error


def get_privacy_epsilon_options(track: Track) -> tuple[float, ...]:
    try:
        return PRIVACY_EPSILON_OPTIONS[track]
    except KeyError as error:
        raise ValueError(f"unsupported track: {track}") from error


def get_quality_observation(
    track: Track, epsilon: float
) -> SuppliedQualityObservation:
    try:
        return QUALITY_OBSERVATIONS[track][epsilon]
    except KeyError as error:
        raise ValueError(
            f"no supplied quality observation for track={track}, epsilon={epsilon}"
        ) from error


def get_deletion_artifacts(
    track: Track, mode: Literal["complete", "slow", "untracked"]
) -> tuple[ArtifactCopy, ...]:
    """Return an explicit alternative copy graph for a deletion exercise."""

    scenario = get_track_scenario(track)
    if mode == "complete":
        return scenario.artifact_copies
    copies = list(scenario.artifact_copies)
    target = copies[-1]
    if mode == "slow":
        copies[-1] = ArtifactCopy(
            target.artifact_id,
            target.copied_from,
            scenario.deletion_sla_hours * 1.25,
            tracked=True,
            incident_evidence_events=target.incident_evidence_events,
        )
    elif mode == "untracked":
        copies[-1] = ArtifactCopy(
            target.artifact_id,
            target.copied_from,
            target.deletion_time_hours,
            tracked=False,
            incident_evidence_events=target.incident_evidence_events,
        )
    else:
        raise ValueError(f"unsupported deletion mode: {mode}")
    return tuple(copies)
