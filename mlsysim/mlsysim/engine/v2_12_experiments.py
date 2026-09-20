"""Quantitative experiments for fleet operations at scale.

The models in this module are deterministic teaching models.  Cost and traffic
inputs are scenario assumptions supplied by a lab; the functions make the
assumptions and conservation identities explicit instead of turning them into
dimensionless operational scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping, Sequence

from ..core.types import Quantity
from ..core.units import USD, count, hour, minute


def _fraction(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _nonnegative_quantity(name: str, value, unit):
    quantity = value.to(unit)
    if quantity.magnitude < 0:
        raise ValueError(f"{name} must be non-negative")
    return quantity


@dataclass(frozen=True)
class PortfolioCostResult:
    model_count: int
    adopted_models: float
    unadopted_models: float
    periods: int
    independent_total: Quantity
    platform_total: Quantity
    platform_fixed_cost: Quantity
    adopted_operation_cost: Quantity
    unadopted_duplication_cost: Quantity
    unadopted_toil_cost: Quantity
    savings: Quantity
    savings_fraction: float
    pays_back: bool
    break_even_model_count: int | None


def portfolio_costs(
    *,
    model_count: int,
    periods: int,
    adoption_fraction: float,
    independent_setup_cost_per_model,
    independent_operation_cost_per_model_period,
    shared_platform_fixed_cost,
    shared_operation_cost_per_adopted_model_period,
    migration_cost_per_adopted_model=0 * USD,
    unadopted_toil_cost_per_model_period=0 * USD,
) -> PortfolioCostResult:
    """Compare duplicated per-model operations with a partially adopted platform.

    ``periods`` is an explicit billing/planning-period count.  All cost inputs
    must be Pint currency quantities in USD-compatible units.
    """

    if model_count <= 0:
        raise ValueError("model_count must be positive")
    if periods <= 0:
        raise ValueError("periods must be positive")
    adoption = _fraction("adoption_fraction", adoption_fraction)

    independent_setup = _nonnegative_quantity(
        "independent_setup_cost_per_model", independent_setup_cost_per_model, USD
    )
    independent_operation = _nonnegative_quantity(
        "independent_operation_cost_per_model_period",
        independent_operation_cost_per_model_period,
        USD,
    )
    platform_fixed = _nonnegative_quantity(
        "shared_platform_fixed_cost", shared_platform_fixed_cost, USD
    )
    shared_operation = _nonnegative_quantity(
        "shared_operation_cost_per_adopted_model_period",
        shared_operation_cost_per_adopted_model_period,
        USD,
    )
    migration = _nonnegative_quantity(
        "migration_cost_per_adopted_model", migration_cost_per_adopted_model, USD
    )
    residual_toil = _nonnegative_quantity(
        "unadopted_toil_cost_per_model_period",
        unadopted_toil_cost_per_model_period,
        USD,
    )

    adopted = model_count * adoption
    unadopted = model_count - adopted
    independent_per_model = independent_setup + periods * independent_operation
    adopted_per_model = migration + periods * shared_operation
    unadopted_duplication_per_model = independent_per_model
    unadopted_toil_per_model = periods * residual_toil

    independent_total = model_count * independent_per_model
    adopted_operation_cost = adopted * adopted_per_model
    unadopted_duplication_cost = unadopted * unadopted_duplication_per_model
    unadopted_toil_cost = unadopted * unadopted_toil_per_model
    platform_total = (
        platform_fixed
        + adopted_operation_cost
        + unadopted_duplication_cost
        + unadopted_toil_cost
    )
    savings = independent_total - platform_total

    platform_variable_per_model = (
        adoption * adopted_per_model
        + (1 - adoption)
        * (unadopted_duplication_per_model + unadopted_toil_per_model)
    )
    margin_per_model = independent_per_model - platform_variable_per_model
    if margin_per_model.magnitude <= 0:
        break_even = None
    else:
        break_even = max(1, ceil((platform_fixed / margin_per_model).to_base_units().magnitude))

    return PortfolioCostResult(
        model_count=model_count,
        adopted_models=adopted,
        unadopted_models=unadopted,
        periods=periods,
        independent_total=independent_total,
        platform_total=platform_total,
        platform_fixed_cost=platform_fixed,
        adopted_operation_cost=adopted_operation_cost,
        unadopted_duplication_cost=unadopted_duplication_cost,
        unadopted_toil_cost=unadopted_toil_cost,
        savings=savings,
        savings_fraction=(savings / independent_total).to_base_units().magnitude,
        pays_back=savings.magnitude >= 0,
        break_even_model_count=break_even,
    )


@dataclass(frozen=True)
class DependencyContract:
    upstream_artifact: str
    consumer: str
    consumer_version: str
    accepted_upstream_versions: tuple[str, ...]


@dataclass(frozen=True)
class DependencyCheck:
    upstream_artifact: str
    upstream_version: str
    consumer: str
    consumer_version: str
    compatible: bool


@dataclass(frozen=True)
class DependencyTraceResult:
    checks: tuple[DependencyCheck, ...]
    reached_consumers: tuple[str, ...]
    blocked_consumers: tuple[str, ...]
    reached_consumer_count: int
    blocked_consumer_count: int


def trace_dependency_changes(
    changed_versions: Mapping[str, str],
    contracts: Sequence[DependencyContract],
) -> DependencyTraceResult:
    """Trace supplied artifact-version changes through consumer contracts.

    A consumer is traversed further only when its own candidate version appears
    in ``changed_versions``.  This prevents an incompatible consumer from being
    treated as though it had silently produced a new downstream artifact.
    """

    if not changed_versions:
        raise ValueError("changed_versions must contain at least one artifact")
    checks: list[DependencyCheck] = []
    reached: set[str] = set()
    blocked: set[str] = set()
    queue = list(changed_versions)
    visited_artifacts: set[str] = set()

    while queue:
        artifact = queue.pop(0)
        if artifact in visited_artifacts:
            continue
        visited_artifacts.add(artifact)
        version = changed_versions[artifact]
        for contract in contracts:
            if contract.upstream_artifact != artifact:
                continue
            compatible = version in contract.accepted_upstream_versions
            reached.add(contract.consumer)
            checks.append(
                DependencyCheck(
                    upstream_artifact=artifact,
                    upstream_version=version,
                    consumer=contract.consumer,
                    consumer_version=contract.consumer_version,
                    compatible=compatible,
                )
            )
            if not compatible:
                blocked.add(contract.consumer)
            if contract.consumer in changed_versions:
                queue.append(contract.consumer)

    reached_consumers = tuple(sorted(reached))
    blocked_consumers = tuple(sorted(blocked))
    return DependencyTraceResult(
        checks=tuple(checks),
        reached_consumers=reached_consumers,
        blocked_consumers=blocked_consumers,
        reached_consumer_count=len(reached_consumers),
        blocked_consumer_count=len(blocked_consumers),
    )


@dataclass(frozen=True)
class CanaryCohort:
    name: str
    traffic_rate: Quantity
    label_delay: Quantity
    required_labeled_samples: int


@dataclass(frozen=True)
class CanaryCohortEvidence:
    name: str
    exposed_requests: Quantity
    labeled_samples: Quantity
    pending_labels: Quantity
    required_labeled_samples: int
    requirement_met: bool


@dataclass(frozen=True)
class CanaryEvidenceResult:
    duration: Quantity
    canary_fraction: float
    cohorts: tuple[CanaryCohortEvidence, ...]
    total_exposed_requests: Quantity
    total_labeled_samples: Quantity
    evidence_requirements_met: bool
    establishes_statistical_significance: bool = False


def canary_evidence(
    *, duration, canary_fraction: float, cohorts: Sequence[CanaryCohort]
) -> CanaryEvidenceResult:
    """Accumulate cohort exposure and observable labels for a canary.

    Required counts are release-policy evidence thresholds.  They do not imply
    power, confidence, effect detection, or statistical significance.
    """

    duration_h = _nonnegative_quantity("duration", duration, hour)
    fraction = _fraction("canary_fraction", canary_fraction)
    if not cohorts:
        raise ValueError("cohorts must not be empty")

    evidence: list[CanaryCohortEvidence] = []
    total_exposed = 0 * count
    total_labeled = 0 * count
    for cohort in cohorts:
        if not cohort.name:
            raise ValueError("cohort name must not be empty")
        rate = _nonnegative_quantity("traffic_rate", cohort.traffic_rate, count / hour)
        delay = _nonnegative_quantity("label_delay", cohort.label_delay, hour)
        if cohort.required_labeled_samples < 0:
            raise ValueError("required_labeled_samples must be non-negative")
        exposed = (rate * duration_h * fraction).to(count)
        observable_duration = max(0.0, (duration_h - delay).to(hour).magnitude) * hour
        labeled = (rate * observable_duration * fraction).to(count)
        pending = exposed - labeled
        met = labeled.magnitude >= cohort.required_labeled_samples
        evidence.append(
            CanaryCohortEvidence(
                name=cohort.name,
                exposed_requests=exposed,
                labeled_samples=labeled,
                pending_labels=pending,
                required_labeled_samples=cohort.required_labeled_samples,
                requirement_met=met,
            )
        )
        total_exposed += exposed
        total_labeled += labeled

    return CanaryEvidenceResult(
        duration=duration_h,
        canary_fraction=fraction,
        cohorts=tuple(evidence),
        total_exposed_requests=total_exposed,
        total_labeled_samples=total_labeled,
        evidence_requirements_met=all(item.requirement_met for item in evidence),
    )


@dataclass(frozen=True)
class IncidentStage:
    name: str
    duration: Quantity
    affected_fraction: float


@dataclass(frozen=True)
class IncidentStageHarm:
    name: str
    duration: Quantity
    affected_fraction: float
    affected_requests: Quantity


@dataclass(frozen=True)
class IncidentHarmResult:
    request_rate: Quantity
    stages: tuple[IncidentStageHarm, ...]
    total_affected_requests: Quantity


def incident_harm(*, request_rate, stages: Sequence[IncidentStage]) -> IncidentHarmResult:
    """Sum affected requests across explicit response stages."""

    rate = _nonnegative_quantity("request_rate", request_rate, count / minute)
    if not stages:
        raise ValueError("stages must not be empty")
    stage_results: list[IncidentStageHarm] = []
    total = 0 * count
    for stage in stages:
        duration_min = _nonnegative_quantity("stage duration", stage.duration, minute)
        fraction = _fraction("affected_fraction", stage.affected_fraction)
        affected = (rate * duration_min * fraction).to(count)
        stage_results.append(
            IncidentStageHarm(
                name=stage.name,
                duration=duration_min,
                affected_fraction=fraction,
                affected_requests=affected,
            )
        )
        total += affected
    return IncidentHarmResult(rate, tuple(stage_results), total)


@dataclass(frozen=True)
class ArtifactRequirement:
    artifact: str
    required_version: str


@dataclass(frozen=True)
class EvidenceRequirement:
    evidence: str
    required_artifact_versions: Mapping[str, str]


@dataclass(frozen=True)
class EvidenceRecord:
    evidence: str
    artifact_versions: Mapping[str, str]


@dataclass(frozen=True)
class ReleaseContractResult:
    releasable: bool
    artifact_mismatches: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    stale_evidence: tuple[str, ...]


def validate_release_contract(
    *,
    current_artifact_versions: Mapping[str, str],
    artifact_requirements: Sequence[ArtifactRequirement],
    evidence_requirements: Sequence[EvidenceRequirement],
    evidence_records: Sequence[EvidenceRecord],
) -> ReleaseContractResult:
    """Check exact artifact and evidence lineage versions for a release."""

    artifact_mismatches = tuple(
        requirement.artifact
        for requirement in artifact_requirements
        if current_artifact_versions.get(requirement.artifact)
        != requirement.required_version
    )
    records = {record.evidence: record for record in evidence_records}
    missing: list[str] = []
    stale: list[str] = []
    for requirement in evidence_requirements:
        record = records.get(requirement.evidence)
        if record is None:
            missing.append(requirement.evidence)
            continue
        if any(
            record.artifact_versions.get(artifact) != required_version
            or current_artifact_versions.get(artifact) != required_version
            for artifact, required_version in requirement.required_artifact_versions.items()
        ):
            stale.append(requirement.evidence)

    return ReleaseContractResult(
        releasable=not artifact_mismatches and not missing and not stale,
        artifact_mismatches=artifact_mismatches,
        missing_evidence=tuple(missing),
        stale_evidence=tuple(stale),
    )


@dataclass(frozen=True)
class OperationsTrackProfile:
    """Illustrative fleet scenario used by the operations lab."""

    display: str
    scenario: str
    portfolio_arguments: Mapping[str, object]
    dependency_artifact: str
    dependency_baseline_version: str
    dependency_changed_version: str
    dependency_contracts: tuple[DependencyContract, ...]
    canary_cohorts: tuple[CanaryCohort, ...]
    incident_request_rate: Quantity
    incident_strategies: Mapping[str, tuple[IncidentStage, ...]]
    release_artifact_versions: Mapping[str, str]
    release_evidence_records: tuple[EvidenceRecord, ...]
    deployment_artifact: str = "serving-package"
    rollback_scope: str = "service instance"
    update_scope: str = "regional rollout"
    workload_unit: str = "requests"
    incident_context: str = "portfolio incidents"


def _profile(
    *,
    display: str,
    scenario: str,
    model_count: int,
    cost_scale: float,
    dependency_artifact: str,
    dependency_baseline_version: str,
    dependency_changed_version: str,
    dependency_contracts: tuple[DependencyContract, ...],
    canary_cohorts: tuple[CanaryCohort, ...],
    incident_request_rate: Quantity,
    incident_scale: float,
    deployment_artifact: str = "serving-package",
    rollback_scope: str = "service instance",
    update_scope: str = "regional rollout",
    workload_unit: str = "requests",
    incident_context: str = "portfolio incidents",
) -> OperationsTrackProfile:
    """Build one internally consistent illustrative track profile."""

    portfolio_arguments = {
        "model_count": model_count,
        "periods": 12,
        "independent_setup_cost_per_model": 4_000 * cost_scale * USD,
        "independent_operation_cost_per_model_period": 1_200 * cost_scale * USD,
        "shared_platform_fixed_cost": 120_000 * cost_scale * USD,
        "shared_operation_cost_per_adopted_model_period": 380 * cost_scale * USD,
        "migration_cost_per_adopted_model": 1_500 * cost_scale * USD,
        "unadopted_toil_cost_per_model_period": 220 * cost_scale * USD,
    }
    base_detection = 12 * incident_scale * minute
    base_diagnosis = 20 * incident_scale * minute
    base_mitigation = 8 * incident_scale * minute
    base_recovery = 30 * incident_scale * minute
    incident_strategies = {
        "baseline": (
            IncidentStage("Detection", base_detection, 0.12),
            IncidentStage("Diagnosis", base_diagnosis, 0.12),
            IncidentStage("Mitigation", base_mitigation, 0.05),
            IncidentStage("Recovery", base_recovery, 0.015),
        ),
        "detection": (
            IncidentStage("Detection", base_detection * 0.35, 0.12),
            IncidentStage("Diagnosis", base_diagnosis, 0.12),
            IncidentStage("Mitigation", base_mitigation, 0.05),
            IncidentStage("Recovery", base_recovery, 0.015),
        ),
        "diagnosis": (
            IncidentStage("Detection", base_detection, 0.12),
            IncidentStage("Diagnosis", base_diagnosis * 0.35, 0.12),
            IncidentStage("Mitigation", base_mitigation, 0.05),
            IncidentStage("Recovery", base_recovery, 0.015),
        ),
        "mitigation": (
            IncidentStage("Detection", base_detection, 0.12),
            IncidentStage("Diagnosis", base_diagnosis, 0.12),
            IncidentStage("Mitigation", base_mitigation, 0.015),
            IncidentStage("Recovery", base_recovery, 0.005),
        ),
        "recovery": (
            IncidentStage("Detection", base_detection, 0.12),
            IncidentStage("Diagnosis", base_diagnosis, 0.12),
            IncidentStage("Mitigation", base_mitigation, 0.05),
            IncidentStage("Recovery", base_recovery * 0.35, 0.015),
        ),
    }
    compat_evidence = f"{deployment_artifact.replace(' ', '-')}-compatibility"
    release_versions = {
        dependency_artifact: dependency_baseline_version,
        "model": "model-v3",
        deployment_artifact: f"{deployment_artifact.replace(' ', '-')}-v2",
    }
    evidence = (
        EvidenceRecord("offline-evaluation", dict(release_versions)),
        EvidenceRecord("canary-observation", dict(release_versions)),
        EvidenceRecord(
            compat_evidence,
            {"model": "model-v3", deployment_artifact: release_versions[deployment_artifact]},
        ),
    )
    return OperationsTrackProfile(
        display=display,
        scenario=scenario,
        portfolio_arguments=portfolio_arguments,
        dependency_artifact=dependency_artifact,
        dependency_baseline_version=dependency_baseline_version,
        dependency_changed_version=dependency_changed_version,
        dependency_contracts=dependency_contracts,
        canary_cohorts=canary_cohorts,
        incident_request_rate=incident_request_rate,
        incident_strategies=incident_strategies,
        release_artifact_versions=release_versions,
        release_evidence_records=evidence,
        deployment_artifact=deployment_artifact,
        rollback_scope=rollback_scope,
        update_scope=update_scope,
        workload_unit=workload_unit,
        incident_context=incident_context,
    )


OPERATIONS_TRACKS = {
    "tinyml": _profile(
        display="TinyML",
        scenario="OTA cohorts across a battery-powered device fleet",
        model_count=24,
        cost_scale=0.35,
        dependency_artifact="sensor-schema",
        dependency_baseline_version="sensor-v1",
        dependency_changed_version="sensor-v2",
        dependency_contracts=(
            DependencyContract("sensor-schema", "wake-word", "ww-v4", ("sensor-v1",)),
            DependencyContract("sensor-schema", "anomaly-detector", "ad-v2", ("sensor-v1", "sensor-v2")),
            DependencyContract("sensor-schema", "device-diagnostics", "diag-v3", ("sensor-v1",)),
        ),
        canary_cohorts=(
            CanaryCohort("frequently connected", 800 * count / hour, 6 * hour, 400),
            CanaryCohort("intermittently connected", 90 * count / hour, 30 * hour, 80),
        ),
        incident_request_rate=420 * count / minute,
        incident_scale=1.6,
        deployment_artifact="firmware",
        rollback_scope="device cohort OTA rollback",
        update_scope="device cohort OTA",
        workload_unit="inferences",
        incident_context="sensor drift across battery IoT nodes",
    ),
    "mobile": _profile(
        display="Mobile",
        scenario="app and model-version cohorts across a phone fleet",
        model_count=55,
        cost_scale=0.7,
        dependency_artifact="feature-schema",
        dependency_baseline_version="feature-v5",
        dependency_changed_version="feature-v6",
        dependency_contracts=(
            DependencyContract("feature-schema", "ranking-model", "rank-v8", ("feature-v5",)),
            DependencyContract("feature-schema", "search-model", "search-v6", ("feature-v5", "feature-v6")),
            DependencyContract("feature-schema", "safety-model", "safe-v3", ("feature-v5",)),
        ),
        canary_cohorts=(
            CanaryCohort("current app", 18_000 * count / hour, 2 * hour, 2_000),
            CanaryCohort("older app", 2_500 * count / hour, 12 * hour, 600),
        ),
        incident_request_rate=4_000 * count / minute,
        incident_scale=1.0,
        deployment_artifact="mobile bundle",
        rollback_scope="staged app store rollback",
        update_scope="staged app store release",
        workload_unit="queries",
        incident_context="feature drift across phone application cohorts",
    ),
    "edge": _profile(
        display="Edge",
        scenario="site cohorts using gateway-managed model releases",
        model_count=90,
        cost_scale=1.0,
        dependency_artifact="site-feature-view",
        dependency_baseline_version="view-v2",
        dependency_changed_version="view-v3",
        dependency_contracts=(
            DependencyContract("site-feature-view", "inspection", "inspect-v7", ("view-v2",)),
            DependencyContract("site-feature-view", "forecast", "forecast-v4", ("view-v2", "view-v3")),
            DependencyContract("site-feature-view", "routing", "route-v5", ("view-v2",)),
        ),
        canary_cohorts=(
            CanaryCohort("high-volume sites", 6_000 * count / hour, 1 * hour, 1_500),
            CanaryCohort("low-volume sites", 500 * count / hour, 8 * hour, 250),
        ),
        incident_request_rate=1_800 * count / minute,
        incident_scale=1.2,
        deployment_artifact="gateway package",
        rollback_scope="site gateway rollback",
        update_scope="site gateway rollout",
        workload_unit="frames",
        incident_context="inspection camera frame corruption on site gateways",
    ),
    "cloud": _profile(
        display="Cloud",
        scenario="regional service portfolio with shared release infrastructure",
        model_count=240,
        cost_scale=1.8,
        dependency_artifact="shared-feature-view",
        dependency_baseline_version="view-v11",
        dependency_changed_version="view-v12",
        dependency_contracts=(
            DependencyContract("shared-feature-view", "recommendation", "rec-v14", ("view-v11",)),
            DependencyContract("shared-feature-view", "fraud", "fraud-v9", ("view-v11", "view-v12")),
            DependencyContract("shared-feature-view", "moderation", "mod-v6", ("view-v11",)),
            DependencyContract("shared-feature-view", "search", "search-v12", ("view-v11", "view-v12")),
        ),
        canary_cohorts=(
            CanaryCohort("primary region", 160_000 * count / hour, 0.25 * hour, 10_000),
            CanaryCohort("small region", 12_000 * count / hour, 3 * hour, 2_000),
        ),
        incident_request_rate=42_000 * count / minute,
        incident_scale=0.7,
        deployment_artifact="cloud service",
        rollback_scope="regional service rollback",
        update_scope="regional service deployment",
        workload_unit="requests",
        incident_context="service degradation across regional endpoints",
    ),
}


def get_operations_track(track_id: str) -> OperationsTrackProfile:
    try:
        return OPERATIONS_TRACKS[track_id]
    except KeyError as exc:
        raise ValueError(f"unknown operations track: {track_id}") from exc


def evaluate_portfolio(track_id: str, adoption_fraction: float) -> PortfolioCostResult:
    profile = get_operations_track(track_id)
    return portfolio_costs(adoption_fraction=adoption_fraction, **profile.portfolio_arguments)


def evaluate_dependency(track_id: str, changed: bool) -> DependencyTraceResult:
    profile = get_operations_track(track_id)
    version = (
        profile.dependency_changed_version
        if changed
        else profile.dependency_baseline_version
    )
    return trace_dependency_changes(
        {profile.dependency_artifact: version}, profile.dependency_contracts
    )


def evaluate_canary(
    track_id: str, *, duration_hours: float, canary_fraction: float
) -> CanaryEvidenceResult:
    return canary_evidence(
        duration=duration_hours * hour,
        canary_fraction=canary_fraction,
        cohorts=get_operations_track(track_id).canary_cohorts,
    )


def evaluate_incident(track_id: str, strategy: str) -> IncidentHarmResult:
    profile = get_operations_track(track_id)
    try:
        stages = profile.incident_strategies[strategy]
    except KeyError as exc:
        raise ValueError(f"unknown incident strategy: {strategy}") from exc
    return incident_harm(request_rate=profile.incident_request_rate, stages=stages)


def evaluate_release_change(track_id: str, changed: bool) -> ReleaseContractResult:
    profile = get_operations_track(track_id)
    versions = dict(profile.release_artifact_versions)
    if changed:
        versions[profile.dependency_artifact] = profile.dependency_changed_version
    artifact_requirements = tuple(
        ArtifactRequirement(artifact, version) for artifact, version in versions.items()
    )
    compat_evidence = f"{profile.deployment_artifact.replace(' ', '-')}-compatibility"
    evidence_requirements = (
        EvidenceRequirement("offline-evaluation", dict(versions)),
        EvidenceRequirement("canary-observation", dict(versions)),
        EvidenceRequirement(
            compat_evidence,
            {
                "model": versions["model"],
                profile.deployment_artifact: versions[profile.deployment_artifact],
            },
        ),
    )
    return validate_release_contract(
        current_artifact_versions=versions,
        artifact_requirements=artifact_requirements,
        evidence_requirements=evidence_requirements,
        evidence_records=profile.release_evidence_records,
    )
