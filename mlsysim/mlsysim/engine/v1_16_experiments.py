"""Evidence replay and cross-layer composition for the Volume I capstone.

This module deliberately separates three things that an audit must not blur:

* the student's actual ledger history;
* an optional, explicitly labeled instructor scenario; and
* recalculated physical outcomes.

Ledger records are evidence, not inputs to a scoring formula.  A record can be
missing, incomplete, incompatible with the selected track, or usable.  Usable
records may still demonstrate a failed requirement.  Physical composition is
accepted only through :class:`DesignSnapshot`, whose quantities and task
identity are explicit.  Unknown historical payloads are never guessed into a
snapshot.

The prepared scenarios and intervention quality observations are illustrative
matched-task teaching fixtures.  They are not measurements of branded systems
or claims that quality follows from hardware performance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

from mlsysim.core.units import Q_
from mlsysim.physics.quantities import compute_time, energy_from_power, transfer_time


class EvidenceStatus(str, Enum):
    """Mutually exclusive states for one expected ledger record."""

    MISSING = "missing"
    INCOMPLETE = "incomplete"
    INCOMPATIBLE = "incompatible"
    SCHEMA_VALID_UNREPLAYED = "schema_valid_unreplayed"
    DEMONSTRATED = "demonstrated"
    FAILED = "failed"


TRACK_ALIASES = {
    "tinyml": "tinyml",
    "oura_ring": "tinyml",
    "mobile": "mobile",
    "iphone": "mobile",
    "edge": "edge",
    "robotaxi": "edge",
    "cloud": "cloud",
    "cloud_fleet": "cloud",
}

DEFAULT_REQUIRED_CHAPTERS = tuple(range(1, 16))
SUPPORTED_MODEL_IDS = frozenset(f"v1_{chapter:02d}_experiments" for chapter in DEFAULT_REQUIRED_CHAPTERS)
SUPPORTED_MODEL_IDS = SUPPORTED_MODEL_IDS | {"constraint_explorer"}
DIRECT_REPLAY_MODEL_IDS = frozenset(
    {
        "v1_04_experiments",
        "v1_03_experiments",
        "v1_02_experiments",
        "v1_05_experiments",
        "v1_06_experiments",
        "v1_07_experiments",
        "v1_08_experiments",
        "v1_09_experiments",
        "v1_10_experiments",
        "v1_11_experiments",
        "v1_12_experiments",
        "v1_13_experiments",
        "v1_14_experiments",
        "v1_15_experiments",
        "v1_16_experiments",
        "constraint_explorer",
    }
)

SCENARIO_ASSUMPTION = (
    "Prepared instructor teaching fixture; physical values are scenario inputs, "
    "not measurements of a named product or deployment."
)
QUALITY_ASSUMPTION = (
    "Supplied illustrative observations for the stated task and population; "
    "quality is not inferred from resource use or hardware capability."
)


@dataclass(frozen=True)
class LedgerRecordAudit:
    chapter: int
    status: EvidenceStatus
    track_id: str | None
    failed_requirements: tuple[str, ...] = ()
    reason: str = ""
    experiment_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class LedgerAudit:
    selected_track_id: str
    records: tuple[LedgerRecordAudit, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {
            status.value: sum(record.status == status for record in self.records)
            for status in EvidenceStatus
        }

    @property
    def usable_chapters(self) -> tuple[int, ...]:
        return tuple(
            record.chapter
            for record in self.records
            if record.status in {EvidenceStatus.DEMONSTRATED, EvidenceStatus.FAILED}
        )

    @property
    def release_evidence_complete(self) -> bool:
        return all(
            record.status in {EvidenceStatus.DEMONSTRATED, EvidenceStatus.FAILED}
            for record in self.records
        )


@dataclass(frozen=True)
class DesignSnapshot:
    """Explicit, composable observables for one single-system design."""

    track_id: str
    task_id: str
    population_id: str
    weights_mb: float
    activation_mb: float
    workspace_mb: float
    movement_mb: float
    operations_mflop: float
    memory_bandwidth_mb_per_ms: float
    compute_rate_mflop_per_ms: float
    fixed_overhead_ms: float
    movement_power_w: float
    compute_power_w: float
    overhead_power_w: float
    quality_pct: float | None
    quality_evidence_id: str | None
    source: str


@dataclass(frozen=True)
class DeploymentEnvelope:
    track_id: str
    task_id: str
    population_id: str
    memory_limit_mb: float
    latency_limit_ms: float
    energy_limit_mj: float
    quality_floor_pct: float
    source: str


_PREPARED: dict[str, dict[str, Any]] = {
    "tinyml": {
        "snapshot": (0.62, 0.18, 0.10, 0.34, 5.0, 0.16, 1.25, 1.2, 0.06, 0.28, 0.04, 88.0),
        "envelope": (1.10, 9.0, 1.50, 85.0),
    },
    "mobile": {
        "snapshot": (42.0, 18.0, 8.0, 76.0, 520.0, 12.0, 42.0, 4.0, 1.5, 4.8, 0.8, 90.0),
        "envelope": (96.0, 24.0, 90.0, 87.0),
    },
    "edge": {
        "snapshot": (680.0, 240.0, 160.0, 920.0, 18_000.0, 48.0, 1_200.0, 2.0, 8.0, 32.0, 5.0, 93.0),
        "envelope": (1_280.0, 40.0, 850.0, 91.0),
    },
    "cloud": {
        "snapshot": (14_000.0, 5_200.0, 2_400.0, 16_000.0, 780_000.0, 1_500.0, 55_000.0, 1.5, 180.0, 520.0, 95.0, 94.0),
        "envelope": (24_000.0, 30.0, 10_000.0, 92.0),
    },
}


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return result


def normalize_track_id(track_id: str) -> str:
    if not isinstance(track_id, str) or track_id not in TRACK_ALIASES:
        raise ValueError(f"track_id must be one of {', '.join(TRACK_ALIASES)}")
    return TRACK_ALIASES[track_id]


def _history_record(history: Mapping[Any, Any], chapter: int) -> Any:
    if chapter in history:
        return history[chapter]
    return history.get(str(chapter))


def _record_track(payload: Mapping[str, Any]) -> str | None:
    raw = payload.get("track_id", payload.get("track"))
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in TRACK_ALIASES:
        return "__invalid__"
    return normalize_track_id(raw)


def _failed_requirements(payload: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Validate and return explicitly failed requirements, if supplied."""
    results = payload.get("requirement_results")
    if results is None:
        violations = payload.get("violations", ())
        if not isinstance(violations, Sequence) or isinstance(violations, (str, bytes)):
            return None
        if not all(isinstance(item, str) and item for item in violations):
            return None
        return tuple(violations)
    if not isinstance(results, Mapping):
        return None
    if not all(isinstance(key, str) and key and isinstance(value, bool) for key, value in results.items()):
        return None
    return tuple(key for key, passed in results.items() if not passed)


def _capture_failures(capture: Mapping[str, Any]) -> tuple[str, ...] | None:
    result = capture.get("result")
    if not isinstance(result, Mapping):
        return None
    return _failed_requirements(result)


def _validate_capture(part: str, capture: Any) -> tuple[str, ...] | None:
    """Validate the immutable JSON capture contract and return failures."""
    if not isinstance(capture, Mapping):
        return None
    required = {
        "track", "part", "prediction", "inputs", "baseline", "result",
        "alternatives", "decision", "model_key", "upstream_fingerprint",
    }
    if not required.issubset(capture):
        return None
    if capture["part"] != part or not isinstance(capture["track"], str):
        return None
    if not isinstance(capture["inputs"], Mapping) or not capture["inputs"]:
        return None
    if not isinstance(capture["model_key"], str) or not capture["model_key"]:
        return None
    fingerprint = capture["upstream_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        return None
    arm_inputs = []
    for arm_name in ("baseline", "result"):
        arm = capture[arm_name]
        if (
            not isinstance(arm, Mapping)
            or not isinstance(arm.get("inputs"), Mapping)
            or not arm["inputs"]
        ):
            return None
        arm_inputs.append(arm["inputs"])
    if arm_inputs[0] == arm_inputs[1]:
        return None
    chosen = capture.get("chosen_result")
    if chosen is not None and (
        not isinstance(chosen, Mapping)
        or not isinstance(chosen.get("inputs"), Mapping)
        or not chosen["inputs"]
    ):
        return None
    if "result_role" in capture and not isinstance(capture["result_role"], str):
        return None
    return _capture_failures(capture)


def replay_evidence_arm(
    model_id: str,
    model_key: str,
    arm: Mapping[str, Any],
    *,
    track_id: str | None = None,
) -> Any:
    """Recompute one evidence arm through a recognized versioned evaluator."""
    if model_id not in DIRECT_REPLAY_MODEL_IDS:
        raise ValueError(f"no direct replay adapter for model_id {model_id!r}")
    if not isinstance(arm, Mapping) or not isinstance(arm.get("inputs"), Mapping):
        raise ValueError("evidence arm must contain an inputs mapping")
    if model_id == "v1_04_experiments":
        from mlsysim.engine.v1_04_experiments import replay_experiment

        return replay_experiment({"method": model_key, "inputs": arm["inputs"]})
    if model_id == "v1_02_experiments":
        from mlsysim.engine.v1_02_experiments import replay as replay_v1_02

        return replay_v1_02(model_key, arm["inputs"])
    if model_id == "v1_03_experiments":
        from mlsysim.engine.v1_03_experiments import replay as replay_v1_03

        return replay_v1_03(model_key, arm["inputs"])
    if model_id == "v1_05_experiments":
        from mlsysim.engine.v1_05_experiments import replay as replay_v1_05

        return replay_v1_05(model_key, arm["inputs"])
    if model_id == "constraint_explorer":
        from mlsysim.engine.constraint_explorer import evaluate

        if model_key != "constraint_explorer.evaluate":
            raise ValueError(f"unknown pilot model_key {model_key!r}")
        return evaluate(**arm["inputs"])
    if model_id == "v1_06_experiments":
        from mlsysim.engine.v1_06_experiments import replay_snapshot

        return replay_snapshot({"model_key": model_key, "inputs": arm["inputs"]})
    if model_id == "v1_07_experiments":
        from mlsysim.engine.v1_07_experiments import replay as replay_v1_07

        return replay_v1_07(model_key, arm["inputs"])
    if model_id == "v1_09_experiments":
        from mlsysim.engine.v1_09_experiments import replay

        return replay(model_key, arm["inputs"])
    if model_id == "v1_08_experiments":
        from mlsysim.engine.v1_08_experiments import replay as replay_v1_08

        return replay_v1_08(model_key, arm["inputs"])
    if model_id == "v1_10_experiments":
        from mlsysim.engine.v1_10_experiments import replay as replay_v1_10

        return replay_v1_10(model_key, arm["inputs"])
    if model_id == "v1_11_experiments":
        from mlsysim.engine.v1_11_experiments import replay_experiment

        if track_id is None:
            raise ValueError("track_id is required to replay Chapter 11 evidence")
        return replay_experiment(
            {"model_key": model_key, "track_id": normalize_track_id(track_id), "inputs": arm["inputs"]}
        )
    if model_id == "v1_12_experiments":
        from mlsysim.engine.v1_12_experiments import replay as replay_v1_12

        return replay_v1_12(model_key, arm["inputs"])
    if model_id == "v1_16_experiments":
        return replay(model_key, arm["inputs"])
    if model_id == "v1_15_experiments":
        from mlsysim.engine import v1_15_experiments as v1_15

        methods = {
            "v1_15_experiments.population_mix": v1_15.population_mix,
            "v1_15_experiments.threshold_consequences": v1_15.threshold_consequences,
            "v1_15_experiments.bounded_mean_release": v1_15.bounded_mean_release,
            "v1_15_experiments.lifecycle_footprint": v1_15.lifecycle_footprint,
            "v1_15_experiments.investigate_incident": v1_15.investigate_incident,
        }
        if model_key not in methods:
            raise ValueError(f"unknown Chapter 15 model_key {model_key!r}")
        return methods[model_key](**arm["inputs"])
    if model_id == "v1_14_experiments":
        from mlsysim.engine.v1_14_experiments import replay_experiment

        return replay_experiment(model_key, arm["inputs"])
    from mlsysim.engine.v1_13_experiments import REPLAY_METHOD, replay_serving_snapshot

    if model_key != REPLAY_METHOD:
        raise ValueError(f"unsupported Chapter 13 model_key {model_key!r}")
    return replay_serving_snapshot(arm["inputs"])


def replay_capture(model_id: str, capture: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute both immutable comparison arms from their saved inputs."""
    model_key = capture.get("model_key")
    if not isinstance(model_key, str) or not model_key:
        raise ValueError("capture model_key must be a nonempty string")
    arms = {
        arm_name: replay_evidence_arm(
            model_id,
            model_key,
            capture[arm_name],
            track_id=capture.get("track"),
        )
        for arm_name in ("baseline", "result")
    }
    if capture.get("chosen_result") is not None:
        arms["chosen_result"] = replay_evidence_arm(
            model_id, model_key, capture["chosen_result"], track_id=capture.get("track")
        )
    return arms


def _outcome_failures(outcome: Any) -> tuple[str, ...]:
    """Read explicit failures from a recomputed outcome, never from prose."""
    if hasattr(outcome, "result") and isinstance(outcome.result, Mapping):
        outcome = outcome.result
    if isinstance(outcome, Mapping):
        explicit = _failed_requirements(outcome)
        if explicit:
            return explicit
        failures = outcome.get("failures", ())
        if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)):
            return tuple(str(item) for item in failures)
        return ()
    violations = getattr(outcome, "violations", ())
    if isinstance(violations, Sequence) and not isinstance(violations, (str, bytes)):
        return tuple(str(item) for item in violations)
    return ()


def _validate_pilot_payload(
    payload: Mapping[str, Any], chapter: int, selected_track: str
) -> tuple[str, tuple[str, ...], tuple[str, ...], str | None]:
    """Return ``(state, failures, reason)`` for the stable pilot schema."""
    if payload.get("schema_version") != 1:
        return "incompatible", (), (), "schema_version must be 1"
    if payload.get("lab_id") != f"v1_{chapter:02d}":
        return "incompatible", (), (), "lab_id does not match the ledger chapter"
    model_id = payload.get("model_id")
    if model_id not in SUPPORTED_MODEL_IDS:
        return "incompatible", (), (), "model_id is not recognized"
    record_track = _record_track(payload)
    if record_track in {None, "__invalid__"}:
        return "incompatible", (), (), "track_id must be a canonical or supported legacy track"
    if record_track != selected_track:
        return "incompatible", (), (), "track differs from selected track"
    decision_fields = (
        "recommendation", "rejected_alternative", "reevaluation_trigger",
        "residual_risk", "rationale",
    )
    if any(
        not isinstance(payload.get(field), str) or not payload[field].strip()
        for field in decision_fields
    ):
        return "incomplete", (), (), "recommendation and synthesis fields must be completed"
    if payload["recommendation"] == payload["rejected_alternative"]:
        return "incomplete", (), (), "recommendation and rejected alternative must differ"
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        return "incomplete", (), (), "evidence must contain at least one captured part"
    stored_failures: list[str] = []
    replayed_failures: list[str] = []
    for part, capture in evidence.items():
        if not isinstance(part, str) or not part:
            return "incompatible", (), (), "evidence part keys must be nonempty strings"
        part_failures = _validate_capture(part, capture)
        if part_failures is None:
            return "incompatible", (), (), f"evidence part {part} has an invalid capture schema"
        if capture["track"] not in TRACK_ALIASES or normalize_track_id(capture["track"]) != selected_track:
            return "incompatible", (), (), f"evidence part {part} has a different track"
        if model_id in DIRECT_REPLAY_MODEL_IDS:
            try:
                replayed = replay_capture(model_id, capture)
            except (KeyError, TypeError, ValueError) as exc:
                return "incompatible", (), (), f"evidence part {part} cannot be replayed: {exc}"
            selected_name = "chosen_result" if "chosen_result" in replayed else "result"
            replayed_failures.extend(
                f"{part}:{selected_name}:{failure}"
                for failure in _outcome_failures(replayed[selected_name])
            )
            stored_failures.extend(
                f"{part}:result:{failure}"
                for failure in _outcome_failures(replayed["result"])
            )
        elif part_failures:
            stored_failures.extend(f"{part}:result:{failure}" for failure in part_failures)
    if model_id not in DIRECT_REPLAY_MODEL_IDS:
        return "schema_valid_unreplayed", (), tuple(stored_failures), "schema valid; no direct replay adapter is available"
    return "failed" if replayed_failures else "demonstrated", tuple(replayed_failures), tuple(stored_failures), None


def audit_ledger(
    history: Mapping[Any, Any],
    track_id: str,
    *,
    required_chapters: Sequence[int] = DEFAULT_REQUIRED_CHAPTERS,
) -> LedgerAudit:
    """Validate actual ledger records without filling gaps from a preset.

    Only the stable schema-version-1 payload is accepted.  Unknown historical
    shapes and unknown model identifiers are incompatible rather than guessed.
    Explicit requirement results inside captured outcomes distinguish a
    demonstrated failure from a missing record.
    """
    if not isinstance(history, Mapping):
        raise ValueError("history must be a mapping of chapter IDs to payloads")
    selected = normalize_track_id(track_id)
    chapters = tuple(required_chapters)
    if not chapters or any(isinstance(chapter, bool) or not isinstance(chapter, int) or chapter < 0 for chapter in chapters):
        raise ValueError("required_chapters must contain nonnegative integers")
    if len(set(chapters)) != len(chapters):
        raise ValueError("required_chapters must not contain duplicates")

    audits: list[LedgerRecordAudit] = []
    for chapter in chapters:
        payload = _history_record(history, chapter)
        if payload is None:
            audits.append(LedgerRecordAudit(chapter, EvidenceStatus.MISSING, None, reason="record absent"))
            continue
        if not isinstance(payload, Mapping):
            audits.append(LedgerRecordAudit(chapter, EvidenceStatus.INCOMPATIBLE, None, reason="payload is not a mapping"))
            continue
        state, failures, experiment_failures, reason = _validate_pilot_payload(payload, chapter, selected)
        status = EvidenceStatus(state)
        record_track = _record_track(payload)
        audits.append(LedgerRecordAudit(chapter, status, None if record_track == "__invalid__" else record_track, failures, reason or "", experiment_failures))
    return LedgerAudit(selected, tuple(audits))


def prepared_instructor_scenario(track_id: str) -> dict[str, Any]:
    """Return an explicitly labeled scenario, separate from ledger history."""
    track = normalize_track_id(track_id)
    values = _PREPARED[track]
    snapshot_values = values["snapshot"]
    snapshot = DesignSnapshot(
        track, f"{track}_event_classifier", f"{track}_reference_population",
        *snapshot_values[:-1], snapshot_values[-1], f"prepared_{track}_quality_v1",
        "prepared_instructor_scenario",
    )
    envelope_values = values["envelope"]
    envelope = DeploymentEnvelope(
        track, snapshot.task_id, snapshot.population_id, *envelope_values,
        "prepared_instructor_scenario",
    )
    return {
        "kind": "prepared_instructor_scenario",
        "is_student_history": False,
        "scenario_assumption": SCENARIO_ASSUMPTION,
        "quality_assumption": QUALITY_ASSUMPTION,
        "snapshot": snapshot,
        "envelope": envelope,
    }


def _validate_snapshot(snapshot: DesignSnapshot) -> DesignSnapshot:
    if not isinstance(snapshot, DesignSnapshot):
        raise ValueError("snapshot must be a DesignSnapshot")
    normalize_track_id(snapshot.track_id)
    if not snapshot.task_id or not snapshot.population_id or not snapshot.source:
        raise ValueError("snapshot task_id, population_id, and source are required")
    for name in (
        "weights_mb", "activation_mb", "workspace_mb", "movement_mb",
        "operations_mflop", "fixed_overhead_ms", "movement_power_w",
        "compute_power_w", "overhead_power_w",
    ):
        _finite(getattr(snapshot, name), name)
    _finite(snapshot.memory_bandwidth_mb_per_ms, "memory_bandwidth_mb_per_ms", positive=True)
    _finite(snapshot.compute_rate_mflop_per_ms, "compute_rate_mflop_per_ms", positive=True)
    if snapshot.quality_pct is not None:
        quality = _finite(snapshot.quality_pct, "quality_pct")
        if quality > 100:
            raise ValueError("quality_pct must be at most 100")
        if not snapshot.quality_evidence_id:
            raise ValueError("quality_evidence_id is required when quality_pct is supplied")
    return snapshot


def _validate_envelope(envelope: DeploymentEnvelope) -> DeploymentEnvelope:
    if not isinstance(envelope, DeploymentEnvelope):
        raise ValueError("envelope must be a DeploymentEnvelope")
    normalize_track_id(envelope.track_id)
    for name in ("memory_limit_mb", "latency_limit_ms", "energy_limit_mj"):
        _finite(getattr(envelope, name), name, positive=True)
    floor = _finite(envelope.quality_floor_pct, "quality_floor_pct")
    if floor > 100:
        raise ValueError("quality_floor_pct must be at most 100")
    return envelope


def evaluate_joint_design(snapshot: DesignSnapshot, envelope: DeploymentEnvelope) -> dict[str, Any]:
    """Compose memory, iron-law path time, energy, and matched quality."""
    snapshot = _validate_snapshot(snapshot)
    envelope = _validate_envelope(envelope)
    track_match = normalize_track_id(snapshot.track_id) == normalize_track_id(envelope.track_id)
    task_match = snapshot.task_id == envelope.task_id and snapshot.population_id == envelope.population_id

    memory_mb = snapshot.weights_mb + snapshot.activation_mb + snapshot.workspace_mb
    movement_time = transfer_time(
        Q_(snapshot.movement_mb, "megabyte"),
        Q_(snapshot.memory_bandwidth_mb_per_ms, "megabyte / millisecond"),
    )
    compute_duration = compute_time(
        Q_(snapshot.operations_mflop, "megaflop"),
        Q_(snapshot.compute_rate_mflop_per_ms, "megaflop / millisecond"),
    )
    overhead_duration = Q_(snapshot.fixed_overhead_ms, "millisecond")
    latency_ms = (movement_time + compute_duration + overhead_duration).m_as("millisecond")
    movement_energy = energy_from_power(Q_(snapshot.movement_power_w, "watt"), movement_time)
    compute_energy = energy_from_power(Q_(snapshot.compute_power_w, "watt"), compute_duration)
    overhead_energy = energy_from_power(Q_(snapshot.overhead_power_w, "watt"), overhead_duration)
    energy_mj = (movement_energy + compute_energy + overhead_energy).m_as("millijoule")

    requirement_results: dict[str, bool | None] = {
        "memory": memory_mb <= envelope.memory_limit_mb,
        "latency": latency_ms <= envelope.latency_limit_ms,
        "energy": energy_mj <= envelope.energy_limit_mj,
        "quality": (
            snapshot.quality_pct >= envelope.quality_floor_pct
            if task_match and snapshot.quality_pct is not None
            else None
        ),
    }
    failed = tuple(name for name, passed in requirement_results.items() if passed is False)
    unavailable = tuple(name for name, passed in requirement_results.items() if passed is None)
    return {
        "track_id": normalize_track_id(envelope.track_id),
        "source_track_id": normalize_track_id(snapshot.track_id),
        "track_match": track_match,
        "task_population_match": task_match,
        "memory_mb": round(memory_mb, 6),
        "movement_ms": round(movement_time.m_as("millisecond"), 6),
        "compute_ms": round(compute_duration.m_as("millisecond"), 6),
        "overhead_ms": round(snapshot.fixed_overhead_ms, 6),
        "latency_ms": round(latency_ms, 6),
        "energy_mj": round(energy_mj, 6),
        "quality_pct": snapshot.quality_pct if task_match else None,
        "limits": {
            "memory_mb": envelope.memory_limit_mb,
            "latency_ms": envelope.latency_limit_ms,
            "energy_mj": envelope.energy_limit_mj,
            "quality_floor_pct": envelope.quality_floor_pct,
        },
        "requirement_results": requirement_results,
        "failed_requirements": failed,
        "unavailable_requirements": unavailable,
        "feasible": not failed and not unavailable,
        "physical_requirements_pass": not any(
            requirement_results[name] is False for name in ("memory", "latency", "energy")
        ),
        "snapshot": asdict(snapshot),
        "envelope": asdict(envelope),
    }


_PREPARED_INTERVENTIONS: dict[str, dict[str, tuple[Any, ...]]] = {
    "tinyml": {
        "quantize": (0.31, 0.126, 0.10, 0.2108, 5.0, 0.16, 1.25, 1.55, 0.06, 0.28, 0.04, 86.8),
        "fuse_runtime": (0.62, 0.1476, 0.112, 0.2448, 5.0, 0.16, 1.25, 0.80, 0.06, 0.28, 0.04, 88.0),
        "larger_model": (0.837, 0.225, 0.11, 0.442, 7.0, 0.16, 1.25, 1.20, 0.06, 0.28, 0.04, 89.8),
    },
    "mobile": {
        "quantize": (21.0, 12.6, 8.0, 47.12, 520.0, 12.0, 42.0, 4.35, 1.5, 4.8, 0.8, 89.2),
        "fuse_runtime": (42.0, 14.76, 8.96, 54.72, 520.0, 12.0, 42.0, 3.60, 1.5, 4.8, 0.8, 90.0),
        "larger_model": (56.7, 22.5, 8.8, 98.8, 728.0, 12.0, 42.0, 4.0, 1.5, 4.8, 0.8, 91.5),
    },
    "edge": {
        "quantize": (340.0, 168.0, 160.0, 570.4, 18_000.0, 48.0, 1_200.0, 2.35, 8.0, 32.0, 5.0, 92.5),
        "fuse_runtime": (680.0, 196.8, 179.2, 662.4, 18_000.0, 48.0, 1_200.0, 1.60, 8.0, 32.0, 5.0, 93.0),
        "larger_model": (918.0, 300.0, 176.0, 1_196.0, 25_200.0, 48.0, 1_200.0, 2.0, 8.0, 32.0, 5.0, 94.1),
    },
    "cloud": {
        "quantize": (7_000.0, 3_640.0, 2_400.0, 9_920.0, 780_000.0, 1_500.0, 55_000.0, 1.85, 180.0, 520.0, 95.0, 93.3),
        "fuse_runtime": (14_000.0, 4_264.0, 2_688.0, 11_520.0, 780_000.0, 1_500.0, 55_000.0, 1.10, 180.0, 520.0, 95.0, 94.0),
        "larger_model": (18_900.0, 6_500.0, 2_640.0, 20_800.0, 1_092_000.0, 1_500.0, 55_000.0, 1.50, 180.0, 520.0, 95.0, 94.9),
    },
}


def apply_prepared_intervention(snapshot: DesignSnapshot, intervention_id: str) -> DesignSnapshot:
    """Apply one disclosed prepared-fixture intervention.

    Each alternative is a fully declared scenario snapshot, including its
    supplied matched-task quality observation.  No field is derived from a
    generic technique multiplier.
    """
    snapshot = _validate_snapshot(snapshot)
    if snapshot.source != "prepared_instructor_scenario":
        raise ValueError("prepared interventions apply only to the prepared instructor scenario")
    if intervention_id not in _PREPARED_INTERVENTIONS[normalize_track_id(snapshot.track_id)]:
        raise ValueError(f"intervention_id must be one of {', '.join(_PREPARED_INTERVENTIONS['tinyml'])}")
    track = normalize_track_id(snapshot.track_id)
    values = _PREPARED_INTERVENTIONS[track][intervention_id]
    return DesignSnapshot(
        track,
        snapshot.task_id,
        snapshot.population_id,
        *values[:-1],
        values[-1],
        f"prepared_{track}_{intervention_id}_quality_v1",
        f"prepared_instructor_scenario:{intervention_id}",
    )


def compare_prepared_intervention(track_id: str, intervention_id: str) -> dict[str, Any]:
    scenario = prepared_instructor_scenario(track_id)
    baseline = evaluate_joint_design(scenario["snapshot"], scenario["envelope"])
    # apply_prepared_intervention restricts the source before changing it.
    candidate_snapshot = apply_prepared_intervention(scenario["snapshot"], intervention_id)
    candidate = evaluate_joint_design(candidate_snapshot, scenario["envelope"])
    return {
        "track_id": normalize_track_id(track_id),
        "intervention_id": intervention_id,
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            key: round(candidate[key] - baseline[key], 6)
            for key in ("memory_mb", "movement_ms", "compute_ms", "overhead_ms", "latency_ms", "energy_mj", "quality_pct")
        },
        "quality_assumption": QUALITY_ASSUMPTION,
    }


def evaluate_envelope_change(
    snapshot: DesignSnapshot,
    envelope: DeploymentEnvelope,
    *,
    memory_limit_mb: float | None = None,
    latency_limit_ms: float | None = None,
    energy_limit_mj: float | None = None,
) -> dict[str, Any]:
    """Change explicit limits while holding the saved design fixed."""
    updates: dict[str, float] = {}
    for name, value in (
        ("memory_limit_mb", memory_limit_mb),
        ("latency_limit_ms", latency_limit_ms),
        ("energy_limit_mj", energy_limit_mj),
    ):
        if value is not None:
            updates[name] = _finite(value, name, positive=True)
    changed = replace(envelope, **updates)
    return {"baseline": evaluate_joint_design(snapshot, envelope), "changed": evaluate_joint_design(snapshot, changed)}


def evaluate_envelope_fraction(
    snapshot: DesignSnapshot,
    envelope: DeploymentEnvelope,
    *,
    requirement: str,
    fraction: float,
) -> dict[str, Any]:
    """Scale one explicit envelope limit while holding the design fixed."""
    if requirement not in {"memory", "latency", "energy"}:
        raise ValueError("requirement must be memory, latency, or energy")
    selected_fraction = _finite(fraction, "fraction", positive=True)
    fields = {
        "memory": ("memory_limit_mb", envelope.memory_limit_mb),
        "latency": ("latency_limit_ms", envelope.latency_limit_ms),
        "energy": ("energy_limit_mj", envelope.energy_limit_mj),
    }
    field, current_limit = fields[requirement]
    return evaluate_envelope_change(
        snapshot,
        envelope,
        **{field: current_limit * selected_fraction},
    )


def transfer_design(
    snapshot: DesignSnapshot,
    target_envelope: DeploymentEnvelope,
    *,
    matched_quality_pct: float | None = None,
    matched_quality_evidence_id: str | None = None,
) -> dict[str, Any]:
    """Reevaluate a saved physical design under another explicit envelope.

    The source quality observation is removed when task or population changes.
    A caller may supply a distinct matched observation for the target task and
    population; both its value and evidence identifier are required together.
    """
    snapshot = _validate_snapshot(snapshot)
    target_envelope = _validate_envelope(target_envelope)
    same_population = (
        snapshot.task_id == target_envelope.task_id
        and snapshot.population_id == target_envelope.population_id
    )
    if (matched_quality_pct is None) != (matched_quality_evidence_id is None):
        raise ValueError("matched quality value and evidence identifier must be supplied together")
    if matched_quality_pct is not None:
        quality = _finite(matched_quality_pct, "matched_quality_pct")
        if quality > 100:
            raise ValueError("matched_quality_pct must be at most 100")
        transferred = replace(
            snapshot,
            quality_pct=quality,
            quality_evidence_id=matched_quality_evidence_id,
            task_id=target_envelope.task_id,
            population_id=target_envelope.population_id,
        )
    elif same_population:
        transferred = snapshot
    else:
        transferred = replace(snapshot, quality_pct=None, quality_evidence_id=None)
    return evaluate_joint_design(transferred, target_envelope)


def replay(model_key: str, inputs: Mapping[str, Any]) -> Any:
    """Replay a Chapter 16 evidence arm from JSON-safe exact inputs."""
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping")
    if model_key == "v1_16.audit_ledger.v1":
        return audit_ledger(
            inputs["history"],
            str(inputs["track_id"]),
            required_chapters=tuple(inputs["required_chapters"]),
        )
    if model_key == "v1_16.evaluate_joint_design.v1":
        snapshot = DesignSnapshot(**inputs["snapshot"])
        envelope = DeploymentEnvelope(**inputs["envelope"])
        return evaluate_joint_design(snapshot, envelope)
    raise ValueError(f"unknown Chapter 16 model_key {model_key!r}")


__all__ = [
    "DEFAULT_REQUIRED_CHAPTERS", "DIRECT_REPLAY_MODEL_IDS", "DeploymentEnvelope", "DesignSnapshot",
    "EvidenceStatus", "LedgerAudit", "LedgerRecordAudit", "QUALITY_ASSUMPTION",
    "SCENARIO_ASSUMPTION", "SUPPORTED_MODEL_IDS", "TRACK_ALIASES", "apply_prepared_intervention",
    "audit_ledger", "compare_prepared_intervention", "evaluate_envelope_change",
    "evaluate_envelope_fraction",
    "evaluate_joint_design", "normalize_track_id", "prepared_instructor_scenario",
    "replay", "replay_capture", "replay_evidence_arm", "transfer_design",
]
