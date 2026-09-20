"""Deterministic experiments for the Volume I data-engineering lab.

The fixtures in this module are illustrative teaching scenarios.  They are not
measurements of a named production system.  Each experiment exposes observable
counts, times, bytes, rates, or supplied outcome labels; it deliberately avoids
turning those observations into a synthetic quality or readiness score.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Any, Literal, Mapping, Sequence

from ..core.types import Quantity
from ..core.units import Q_


Track = Literal["tinyml", "mobile", "edge", "cloud"]
RetentionPolicy = Literal["newest", "coverage", "deduplicate"]
ContractLevel = Literal["none", "schema", "semantic"]
ExperimentMethod = Literal["retention", "split", "pipeline", "freshness", "contract"]


@dataclass(frozen=True)
class CandidateRecord:
    """One candidate example with an explicit annotation cost."""

    record_id: str
    entity_id: str
    cohort: str
    captured_at: Quantity
    annotation_time: Quantity
    has_task_evidence: bool
    duplicate_of: str | None = None


@dataclass(frozen=True)
class RetentionResult:
    policy: RetentionPolicy
    retained_ids: tuple[str, ...]
    annotation_time: Quantity
    budget_remaining: Quantity
    represented_cohorts: tuple[str, ...]
    represented_cohort_count: int
    unique_entities: int
    duplicate_records: int
    records_with_task_evidence: int


def retain_for_annotation(
    records: Sequence[CandidateRecord],
    annotation_budget: Quantity,
    policy: RetentionPolicy,
) -> RetentionResult:
    """Select whole records without exceeding a fixed annotation-time budget."""

    budget_minutes = annotation_budget.to("minute").magnitude
    if budget_minutes < 0:
        raise ValueError("annotation_budget must be non-negative")
    if policy not in {"newest", "coverage", "deduplicate"}:
        raise ValueError(f"unsupported retention policy: {policy}")

    candidates = list(records)
    if policy == "newest":
        candidates.sort(key=lambda record: record.captured_at.to("hour").magnitude, reverse=True)
    elif policy == "deduplicate":
        candidates = [record for record in candidates if record.duplicate_of is None]
        candidates.sort(key=lambda record: record.captured_at.to("hour").magnitude, reverse=True)
    else:
        # Round-robin cohorts so scarce annotation time reaches distinct groups.
        remaining = sorted(
            candidates,
            key=lambda record: (record.cohort, -record.captured_at.to("hour").magnitude),
        )
        candidates = []
        represented: set[str] = set()
        while remaining:
            unseen = next(
                (record for record in remaining if record.cohort not in represented),
                remaining[0],
            )
            remaining.remove(unseen)
            candidates.append(unseen)
            represented.add(unseen.cohort)

    retained: list[CandidateRecord] = []
    spent_minutes = 0.0
    for record in candidates:
        cost = record.annotation_time.to("minute").magnitude
        if cost < 0:
            raise ValueError("record annotation_time must be non-negative")
        if spent_minutes + cost <= budget_minutes:
            retained.append(record)
            spent_minutes += cost

    return RetentionResult(
        policy=policy,
        retained_ids=tuple(record.record_id for record in retained),
        annotation_time=Q_(spent_minutes, "minute"),
        budget_remaining=Q_(budget_minutes - spent_minutes, "minute"),
        represented_cohorts=tuple(sorted({record.cohort for record in retained})),
        represented_cohort_count=len({record.cohort for record in retained}),
        unique_entities=len({record.entity_id for record in retained}),
        duplicate_records=sum(record.duplicate_of is not None for record in retained),
        records_with_task_evidence=sum(record.has_task_evidence for record in retained),
    )


@dataclass(frozen=True)
class LabeledRecord:
    record_id: str
    entity_id: str
    cohort: str
    observed_label: str


@dataclass(frozen=True)
class SplitResult:
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    overlapping_entity_keys: tuple[str, ...]
    overlapping_entity_count: int
    key_leakage_fraction: float
    key_leakage_percent: float
    preprocessing_test_records_seen: int
    correct_predictions: int
    evaluated_predictions: int
    observed_accuracy: float
    observed_accuracy_percent: float


def evaluate_split(
    records: Sequence[LabeledRecord],
    *,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    predicted_labels: Mapping[str, str],
    preprocessing_fit_ids: Sequence[str],
) -> SplitResult:
    """Measure key overlap and accuracy from supplied illustrative outcomes."""

    by_id = {record.record_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("record_id values must be unique")
    train_set, test_set = set(train_ids), set(test_ids)
    missing = (train_set | test_set | set(preprocessing_fit_ids)) - set(by_id)
    if missing:
        raise ValueError(f"unknown record ids: {sorted(missing)}")
    if train_set & test_set:
        raise ValueError("train_ids and test_ids must be disjoint")
    if set(predicted_labels) != test_set:
        raise ValueError("predicted_labels must contain exactly the test records")

    train_entities = {by_id[record_id].entity_id for record_id in train_set}
    test_entities = {by_id[record_id].entity_id for record_id in test_set}
    overlap = tuple(sorted(train_entities & test_entities))
    test_entity_count = len(test_entities)
    correct = sum(predicted_labels[record_id] == by_id[record_id].observed_label for record_id in test_set)
    return SplitResult(
        train_ids=tuple(train_ids),
        test_ids=tuple(test_ids),
        overlapping_entity_keys=overlap,
        overlapping_entity_count=len(overlap),
        key_leakage_fraction=len(overlap) / test_entity_count if test_entity_count else 0.0,
        key_leakage_percent=(100 * len(overlap) / test_entity_count if test_entity_count else 0.0),
        preprocessing_test_records_seen=len(set(preprocessing_fit_ids) & test_set),
        correct_predictions=correct,
        evaluated_predictions=len(test_set),
        observed_accuracy=correct / len(test_set) if test_set else 0.0,
        observed_accuracy_percent=(100 * correct / len(test_set) if test_set else 0.0),
    )


@dataclass(frozen=True)
class PipelineResult:
    compressed_sample_size: Quantity
    read_rate_per_second: float
    decode_rate_per_second: float
    transform_rate_per_second: float
    service_rate_per_second: float
    required_rate_per_second: float
    bottleneck_stage: str
    accelerator_wait_fraction: float
    accelerator_wait_percent: float
    meets_required_rate: bool


def evaluate_pipeline(
    *,
    sample_size: Quantity,
    compression_ratio: float,
    read_bandwidth: Quantity,
    base_decode_rate_per_second: float,
    compression_decode_work: float,
    transform_rate_per_second: float,
    required_rate_per_second: float,
) -> PipelineResult:
    """Compose independent overlapped stages into one steady-state service rate.

    The model assumes enough buffering to overlap read, decode, and transform
    work after pipeline warmup. Its minimum stage rate is therefore steady-state
    throughput, not the serial latency experienced by one record.
    """

    sample_bytes = sample_size.to("byte").magnitude
    bandwidth_bytes_per_second = read_bandwidth.to("byte/second").magnitude
    if not math.isfinite(sample_bytes) or sample_bytes <= 0:
        raise ValueError("sample_size must be positive")
    if not math.isfinite(bandwidth_bytes_per_second) or bandwidth_bytes_per_second <= 0:
        raise ValueError("read_bandwidth must be positive and finite")
    if not math.isfinite(compression_ratio) or compression_ratio < 1:
        raise ValueError("compression_ratio must be at least 1")
    rates = (
        base_decode_rate_per_second,
        compression_decode_work,
        transform_rate_per_second,
        required_rate_per_second,
    )
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("pipeline rates and decode work must be positive and finite")

    compressed_size = (sample_size / compression_ratio).to("byte")
    read_rate = (read_bandwidth.to("byte/second") / compressed_size).to_base_units().magnitude
    decode_rate = base_decode_rate_per_second / compression_decode_work
    stage_rates = {
        "read": read_rate,
        "decode": decode_rate,
        "transform": transform_rate_per_second,
    }
    bottleneck, service_rate = min(stage_rates.items(), key=lambda item: item[1])
    active_fraction = min(1.0, service_rate / required_rate_per_second)
    return PipelineResult(
        compressed_sample_size=compressed_size,
        read_rate_per_second=read_rate,
        decode_rate_per_second=decode_rate,
        transform_rate_per_second=transform_rate_per_second,
        service_rate_per_second=service_rate,
        required_rate_per_second=required_rate_per_second,
        bottleneck_stage=bottleneck,
        accelerator_wait_fraction=1.0 - active_fraction,
        accelerator_wait_percent=100 * (1.0 - active_fraction),
        meets_required_rate=service_rate >= required_rate_per_second,
    )


@dataclass(frozen=True)
class FreshnessPolicy:
    name: str
    collection_window: Quantity
    transport_time: Quantity
    feature_compute_time: Quantity
    bytes_per_event: Quantity
    annotation_time_per_event: Quantity


@dataclass(frozen=True)
class FreshnessResult:
    policy_name: str
    worst_case_age: Quantity
    traffic: Quantity
    traffic_megabytes: float
    annotation_time: Quantity
    events: int
    meets_freshness_sla: bool


def evaluate_freshness(
    policy: FreshnessPolicy,
    *,
    event_rate: Quantity,
    horizon: Quantity,
    freshness_sla: Quantity,
) -> FreshnessResult:
    """Calculate worst-case feature age and explicit traffic/labor costs."""

    if event_rate.to("1/second").magnitude < 0 or horizon.to("second").magnitude < 0:
        raise ValueError("event_rate and horizon must be non-negative")
    events = int((event_rate * horizon).to_base_units().magnitude)
    age = (policy.collection_window + policy.transport_time + policy.feature_compute_time).to("second")
    traffic = (events * policy.bytes_per_event).to("byte")
    annotation = (events * policy.annotation_time_per_event).to("minute")
    return FreshnessResult(
        policy_name=policy.name,
        worst_case_age=age,
        traffic=traffic,
        traffic_megabytes=traffic.to("megabyte").magnitude,
        annotation_time=annotation,
        events=events,
        meets_freshness_sla=age <= freshness_sla.to("second"),
    )


@dataclass(frozen=True)
class ProducerRecord:
    record_id: str
    schema_version: int
    feature_value: float
    semantic_unit: str


@dataclass(frozen=True)
class ContractResult:
    level: ContractLevel
    accepted_records: int
    rejected_records: int
    escaped_semantic_errors: int
    validation_time: Quantity
    recovery_time: Quantity


def evaluate_contract(
    records: Sequence[ProducerRecord],
    *,
    expected_schema_version: int,
    expected_semantic_unit: str,
    level: ContractLevel,
    schema_check_time_per_record: Quantity,
    semantic_check_time_per_record: Quantity,
    recovery_time_per_escaped_record: Quantity,
) -> ContractResult:
    """Apply an explicit producer contract and count escaped semantic changes."""

    if level not in {"none", "schema", "semantic"}:
        raise ValueError(f"unsupported contract level: {level}")
    accepted = rejected = escaped = 0
    for record in records:
        schema_valid = record.schema_version == expected_schema_version
        semantic_valid = schema_valid and record.semantic_unit == expected_semantic_unit
        contract_accepts = (
            level == "none" or (level == "schema" and schema_valid) or (level == "semantic" and semantic_valid)
        )
        if contract_accepts:
            accepted += 1
            escaped += not semantic_valid
        else:
            rejected += 1

    if level == "none":
        validation = Q_(0, "second")
    elif level == "schema":
        validation = len(records) * schema_check_time_per_record
    else:
        validation = len(records) * (schema_check_time_per_record + semantic_check_time_per_record)
    recovery = escaped * recovery_time_per_escaped_record
    return ContractResult(
        level=level,
        accepted_records=accepted,
        rejected_records=rejected,
        escaped_semantic_errors=escaped,
        validation_time=validation.to("second"),
        recovery_time=recovery.to("minute"),
    )


@dataclass(frozen=True)
class SplitFixture:
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    predictions: Mapping[str, str]


@dataclass(frozen=True)
class TrackScenario:
    """Illustrative, track-specific inputs shared by the five experiments."""

    track: Track
    candidates: tuple[CandidateRecord, ...]
    annotation_budget: Quantity
    labeled_records: tuple[LabeledRecord, ...]
    leaky_split: SplitFixture
    entity_disjoint_split: SplitFixture
    sample_size: Quantity
    read_bandwidth: Quantity
    base_decode_rate_per_second: float
    transform_rate_per_second: float
    required_rate_per_second: float
    freshness_policies: tuple[FreshnessPolicy, ...]
    event_rate: Quantity
    freshness_horizon: Quantity
    freshness_sla: Quantity
    semantic_field_name: str
    expected_semantic_unit: str
    producer_records: tuple[ProducerRecord, ...]
    schema_check_time_per_record: Quantity
    semantic_check_time_per_record: Quantity
    recovery_time_per_escaped_record: Quantity


_TRACK_PHYSICS = {
    "tinyml": (Q_(24, "kilobyte"), Q_(2, "megabyte/second"), 80.0, 48.0, 40.0, Q_(60, "second")),
    "mobile": (Q_(320, "kilobyte"), Q_(48, "megabyte/second"), 180.0, 135.0, 120.0, Q_(8, "second")),
    "edge": (Q_(4, "megabyte"), Q_(600, "megabyte/second"), 240.0, 150.0, 140.0, Q_(20, "second")),
    "cloud": (Q_(2, "megabyte"), Q_(1600, "megabyte/second"), 1200.0, 760.0, 700.0, Q_(15, "second")),
}

_TRACK_CONTEXT = {
    "tinyml": (
        ("typical-night", "sensor-dropout", "high-motion"),
        "sample_interval",
        "millisecond",
        "second",
    ),
    "mobile": (
        ("common-context", "private-context", "low-connectivity"),
        "location_radius",
        "meter",
        "foot",
    ),
    "edge": (
        ("ordinary-route", "rare-event", "night-weather"),
        "stopping_distance",
        "meter",
        "foot",
    ),
    "cloud": (
        ("common-producer", "late-arrival", "regional-source"),
        "event_time",
        "second",
        "millisecond",
    ),
}


def get_track_scenario(track: Track) -> TrackScenario:
    """Return a small representative fixture for one teaching track."""

    if track not in _TRACK_PHYSICS:
        raise ValueError(f"unsupported track: {track}")
    sample_size, bandwidth, decode_rate, transform_rate, required_rate, sla = _TRACK_PHYSICS[track]
    cohorts, semantic_field, expected_unit, changed_unit = _TRACK_CONTEXT[track]
    majority, rare, boundary = cohorts
    prefix = track[0]
    candidates = (
        CandidateRecord(f"{prefix}1", "entity-a", majority, Q_(1, "hour"), Q_(4, "minute"), True),
        CandidateRecord(f"{prefix}2", "entity-a", majority, Q_(6, "hour"), Q_(4, "minute"), True, f"{prefix}1"),
        CandidateRecord(f"{prefix}3", "entity-b", rare, Q_(4, "hour"), Q_(6, "minute"), True),
        CandidateRecord(f"{prefix}4", "entity-c", majority, Q_(5, "hour"), Q_(3, "minute"), False),
        CandidateRecord(f"{prefix}5", "entity-d", boundary, Q_(2, "hour"), Q_(5, "minute"), True),
        CandidateRecord(f"{prefix}6", "entity-e", rare, Q_(3, "hour"), Q_(5, "minute"), False),
    )
    labeled = (
        LabeledRecord(f"{prefix}a1", "entity-a", majority, "yes"),
        LabeledRecord(f"{prefix}a2", "entity-a", majority, "yes"),
        LabeledRecord(f"{prefix}b1", "entity-b", rare, "no"),
        LabeledRecord(f"{prefix}b2", "entity-b", rare, "no"),
        LabeledRecord(f"{prefix}c1", "entity-c", boundary, "yes"),
        LabeledRecord(f"{prefix}d1", "entity-d", majority, "no"),
    )
    leaky = SplitFixture(
        (f"{prefix}a1", f"{prefix}b1", f"{prefix}d1"),
        (f"{prefix}a2", f"{prefix}b2", f"{prefix}c1"),
        {f"{prefix}a2": "yes", f"{prefix}b2": "no", f"{prefix}c1": "yes"},
    )
    disjoint = SplitFixture(
        (f"{prefix}a1", f"{prefix}a2", f"{prefix}b1", f"{prefix}b2"),
        (f"{prefix}c1", f"{prefix}d1"),
        {f"{prefix}c1": "no", f"{prefix}d1": "no"},
    )
    raw_event = sample_size
    policies = (
        FreshnessPolicy("batch", Q_(30, "second"), Q_(3, "second"), Q_(2, "second"), raw_event, Q_(0.04, "minute")),
        FreshnessPolicy("stream", Q_(1, "second"), Q_(2, "second"), Q_(2, "second"), raw_event, Q_(0.04, "minute")),
        FreshnessPolicy(
            "local feature", Q_(1, "second"), Q_(0.5, "second"), Q_(4, "second"), raw_event / 8, Q_(0.06, "minute")
        ),
    )
    producer = (
        ProducerRecord(f"{prefix}-old", 1, 10.0, expected_unit),
        ProducerRecord(f"{prefix}-changed", 1, 10.0, changed_unit),
        ProducerRecord(f"{prefix}-versioned", 2, 3.0, expected_unit),
    )
    return TrackScenario(
        track=track,
        candidates=candidates,
        annotation_budget=Q_(15, "minute"),
        labeled_records=labeled,
        leaky_split=leaky,
        entity_disjoint_split=disjoint,
        sample_size=sample_size,
        read_bandwidth=bandwidth,
        base_decode_rate_per_second=decode_rate,
        transform_rate_per_second=transform_rate,
        required_rate_per_second=required_rate,
        freshness_policies=policies,
        event_rate=Q_(2, "1/second"),
        freshness_horizon=Q_(10, "minute"),
        freshness_sla=sla,
        semantic_field_name=semantic_field,
        expected_semantic_unit=expected_unit,
        producer_records=producer,
        schema_check_time_per_record=Q_(2, "millisecond"),
        semantic_check_time_per_record=Q_(3, "millisecond"),
        recovery_time_per_escaped_record=Q_(20, "minute"),
    )


def retention_inputs(scenario: TrackScenario, policy: RetentionPolicy) -> dict[str, Any]:
    """Return the complete argument set for a retention experiment."""

    return {
        "records": scenario.candidates,
        "annotation_budget": scenario.annotation_budget,
        "policy": policy,
    }


def split_inputs(
    scenario: TrackScenario,
    strategy: Literal["record", "entity"],
    preprocessing_scope: Literal["all", "train"],
) -> dict[str, Any]:
    """Return explicit split assignments, outcomes, and preprocessing fit IDs."""

    fixture = scenario.leaky_split if strategy == "record" else scenario.entity_disjoint_split
    fit_ids = fixture.train_ids + fixture.test_ids if preprocessing_scope == "all" else fixture.train_ids
    return {
        "records": scenario.labeled_records,
        "train_ids": fixture.train_ids,
        "test_ids": fixture.test_ids,
        "predicted_labels": fixture.predictions,
        "preprocessing_fit_ids": fit_ids,
    }


def pipeline_inputs(
    scenario: TrackScenario,
    *,
    storage_path: Literal["constrained", "native"],
    compression: Literal["raw", "balanced", "dense"],
    transform_lanes: Literal[1, 2, 4],
) -> dict[str, Any]:
    """Resolve discrete pipeline choices to complete physical model arguments."""

    bandwidth_divisor = {"constrained": 100, "native": 1}[storage_path]
    compression_ratio, decode_work = {
        "raw": (1.0, 1.0),
        "balanced": (4.0, 1.5),
        "dense": (8.0, 2.5),
    }[compression]
    transform_multiplier = {1: 1.0, 2: 1.6, 4: 2.4}[transform_lanes]
    return {
        "sample_size": scenario.sample_size,
        "compression_ratio": compression_ratio,
        "read_bandwidth": scenario.read_bandwidth / bandwidth_divisor,
        "base_decode_rate_per_second": scenario.base_decode_rate_per_second,
        "compression_decode_work": decode_work,
        "transform_rate_per_second": scenario.transform_rate_per_second * transform_multiplier,
        "required_rate_per_second": scenario.required_rate_per_second,
    }


def freshness_inputs(scenario: TrackScenario, policy_name: str) -> dict[str, Any]:
    """Return a named freshness policy and its complete evaluation context."""

    matches = [policy for policy in scenario.freshness_policies if policy.name == policy_name]
    if not matches:
        raise ValueError(f"unsupported freshness policy: {policy_name}")
    return {
        "policy": matches[0],
        "event_rate": scenario.event_rate,
        "horizon": scenario.freshness_horizon,
        "freshness_sla": scenario.freshness_sla,
    }


def contract_inputs(scenario: TrackScenario, level: ContractLevel) -> dict[str, Any]:
    """Return the complete argument set for one contract level."""

    return {
        "records": scenario.producer_records,
        "expected_schema_version": 1,
        "expected_semantic_unit": scenario.expected_semantic_unit,
        "level": level,
        "schema_check_time_per_record": scenario.schema_check_time_per_record,
        "semantic_check_time_per_record": scenario.semantic_check_time_per_record,
        "recovery_time_per_escaped_record": scenario.recovery_time_per_escaped_record,
    }


@dataclass(frozen=True)
class ExperimentSnapshot:
    """JSON-safe method, exact inputs, and result for immutable evidence."""

    method: ExperimentMethod
    inputs: dict[str, Any]
    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "inputs": self.inputs, "result": self.result}


def _quantity_to_dict(quantity: Quantity) -> dict[str, Any]:
    return {"magnitude": float(quantity.magnitude), "unit": str(quantity.units)}


def _quantity_from_dict(value: Mapping[str, Any]) -> Quantity:
    return Q_(float(value["magnitude"]), str(value["unit"]))


def _json_value(value: Any) -> Any:
    if hasattr(value, "units") and hasattr(value, "magnitude"):
        return _quantity_to_dict(value)
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize experiment value of type {type(value).__name__}")


def _candidate_from_dict(value: Mapping[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        record_id=str(value["record_id"]),
        entity_id=str(value["entity_id"]),
        cohort=str(value["cohort"]),
        captured_at=_quantity_from_dict(value["captured_at"]),
        annotation_time=_quantity_from_dict(value["annotation_time"]),
        has_task_evidence=bool(value["has_task_evidence"]),
        duplicate_of=(str(value["duplicate_of"]) if value["duplicate_of"] is not None else None),
    )


def _labeled_from_dict(value: Mapping[str, Any]) -> LabeledRecord:
    return LabeledRecord(
        record_id=str(value["record_id"]),
        entity_id=str(value["entity_id"]),
        cohort=str(value["cohort"]),
        observed_label=str(value["observed_label"]),
    )


def _freshness_policy_from_dict(value: Mapping[str, Any]) -> FreshnessPolicy:
    return FreshnessPolicy(
        name=str(value["name"]),
        collection_window=_quantity_from_dict(value["collection_window"]),
        transport_time=_quantity_from_dict(value["transport_time"]),
        feature_compute_time=_quantity_from_dict(value["feature_compute_time"]),
        bytes_per_event=_quantity_from_dict(value["bytes_per_event"]),
        annotation_time_per_event=_quantity_from_dict(value["annotation_time_per_event"]),
    )


def _producer_from_dict(value: Mapping[str, Any]) -> ProducerRecord:
    return ProducerRecord(
        record_id=str(value["record_id"]),
        schema_version=int(value["schema_version"]),
        feature_value=float(value["feature_value"]),
        semantic_unit=str(value["semantic_unit"]),
    )


def capture_experiment(method: ExperimentMethod, **inputs: Any) -> ExperimentSnapshot:
    """Evaluate one experiment and preserve its exact JSON-recoverable inputs."""

    if method == "retention":
        result = retain_for_annotation(**inputs)
    elif method == "split":
        result = evaluate_split(**inputs)
    elif method == "pipeline":
        result = evaluate_pipeline(**inputs)
    elif method == "freshness":
        result = evaluate_freshness(**inputs)
    elif method == "contract":
        result = evaluate_contract(**inputs)
    else:
        raise ValueError(f"unsupported experiment method: {method}")
    return ExperimentSnapshot(
        method=method,
        inputs=_json_value(inputs),
        result=_json_value(result),
    )


def replay_experiment(
    snapshot: ExperimentSnapshot | Mapping[str, Any],
) -> ExperimentSnapshot:
    """Re-evaluate a snapshot from its serialized inputs alone."""

    payload = snapshot.to_dict() if isinstance(snapshot, ExperimentSnapshot) else snapshot
    method = payload.get("method")
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ValueError("snapshot inputs must be a mapping")

    if method == "retention":
        inputs = {
            "records": tuple(_candidate_from_dict(item) for item in raw_inputs["records"]),
            "annotation_budget": _quantity_from_dict(raw_inputs["annotation_budget"]),
            "policy": raw_inputs["policy"],
        }
    elif method == "split":
        inputs = {
            "records": tuple(_labeled_from_dict(item) for item in raw_inputs["records"]),
            "train_ids": tuple(raw_inputs["train_ids"]),
            "test_ids": tuple(raw_inputs["test_ids"]),
            "predicted_labels": dict(raw_inputs["predicted_labels"]),
            "preprocessing_fit_ids": tuple(raw_inputs["preprocessing_fit_ids"]),
        }
    elif method == "pipeline":
        inputs = dict(raw_inputs)
        for key in ("sample_size", "read_bandwidth"):
            inputs[key] = _quantity_from_dict(raw_inputs[key])
    elif method == "freshness":
        inputs = {
            "policy": _freshness_policy_from_dict(raw_inputs["policy"]),
            "event_rate": _quantity_from_dict(raw_inputs["event_rate"]),
            "horizon": _quantity_from_dict(raw_inputs["horizon"]),
            "freshness_sla": _quantity_from_dict(raw_inputs["freshness_sla"]),
        }
    elif method == "contract":
        inputs = dict(raw_inputs)
        inputs["records"] = tuple(_producer_from_dict(item) for item in raw_inputs["records"])
        for key in (
            "schema_check_time_per_record",
            "semantic_check_time_per_record",
            "recovery_time_per_escaped_record",
        ):
            inputs[key] = _quantity_from_dict(raw_inputs[key])
    else:
        raise ValueError(f"unsupported experiment method: {method}")
    return capture_experiment(method, **inputs)


__all__ = [
    "CandidateRecord",
    "ContractResult",
    "ExperimentSnapshot",
    "FreshnessPolicy",
    "FreshnessResult",
    "LabeledRecord",
    "PipelineResult",
    "ProducerRecord",
    "RetentionResult",
    "SplitFixture",
    "SplitResult",
    "TrackScenario",
    "capture_experiment",
    "contract_inputs",
    "evaluate_contract",
    "evaluate_freshness",
    "evaluate_pipeline",
    "evaluate_split",
    "freshness_inputs",
    "get_track_scenario",
    "pipeline_inputs",
    "replay_experiment",
    "retention_inputs",
    "retain_for_annotation",
    "split_inputs",
]
