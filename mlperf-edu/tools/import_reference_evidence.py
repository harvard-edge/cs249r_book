#!/usr/bin/env python3
"""Import a complete, verified MLPerf EDU promotion-evidence set.

Raw attempts remain outside the repository because they contain checkpoints and
dataset-derived artifacts. This importer independently verifies every retained
run, then copies only the immutable evidence summaries and a source lock into the
repository and packaged-resource mirrors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mlperf.edu_cli import (  # noqa: E402
    run_comparison_fingerprint_sha256,
    verify_package_archive,
)
from mlperf.manifest import verify_provd  # noqa: E402
from mlperf.registry import Workload, load_registry  # noqa: E402
from tools import check_taxonomy, reference_source_lock  # noqa: E402


SUMMARY_SCHEMA = "mlperf-edu-reference-evidence/0.7"
INDEX_SCHEMA = "mlperf-edu-reference-index/0.3"
HOST_POWER_STATE_SCHEMA = "mlperf-edu-host-power-state/0.1"
POWER_STABILITY_POLICY_SCHEMA = "mlperf-edu-power-stability-policy/0.1"
SOURCE_LOCK_PATH = "reference_results/source_lock.json"
EMPTY_SHA256 = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
RESULT_ROLES = frozenset({"score-bearing", "performance-bearing"})
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
PREFIXED_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
OUTPUT_ROOTS = (
    ROOT / "reference_results",
    ROOT / "src" / "mlperf_edu" / "reference_results",
)
OUTER_EXECUTION_TIMING_SCOPE = (
    "The cooldown occurs before subprocess launch and is excluded from "
    "both subprocess wall time and within-run timing samples."
)
PRECONDITIONING_TIMING_SCOPE = (
    "Preconditioning completes before evidence repetitions. Its workload "
    "timings and quality results are retained for audit but excluded from "
    "all evidence aggregates, acceptance statistics, and repeatability."
)
POWER_STABILITY_POLICY = {
    "schema": POWER_STABILITY_POLICY_SCHEMA,
    "scope": "each complete preconditioning and measured subprocess execution",
    "promotion_requires_external_power": True,
    "promotion_requires_low_power_mode_disabled": True,
    "power_source_change_invalidates_execution": True,
    "power_mode_change_invalidates_execution": True,
    "sleep_or_wake_invalidates_execution": True,
    "unsupported_monitoring_invalidates_promotion": True,
}


@dataclass(frozen=True)
class EvidenceCase:
    """One independently measured workload/profile/mode/phase contract."""

    case_id: str
    workload: Workload
    profile: str
    mode: str
    phase: str | None
    result_role: str
    scenario: str
    data_mode: str
    measurement_protocol: Mapping[str, Any]
    gate: Mapping[str, Any]
    canonical_seed: int


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def case_id(workload: str, profile: str, mode: str, phase: str | None) -> str:
    parts = (workload, profile, mode, phase) if phase else (workload, profile, mode)
    value = "__".join(str(part) for part in parts)
    if not SAFE_COMPONENT_RE.fullmatch(value):
        raise ValueError(f"unsafe evidence case id {value!r}")
    return value


def _canonical_seed(workload: Workload) -> int:
    config = (workload.raw.get("canonical_max_contract") or {}).get("config") or {}
    seed = config.get("random_seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{workload.id}: canonical seed is not an integer")
    return seed


def expected_cases() -> dict[str, EvidenceCase]:
    """Derive the complete score and phase-level evidence closure."""
    cases: dict[str, EvidenceCase] = {}
    for workload in load_registry(ROOT / "registry").values():
        if workload.raw.get("promotion_scope", True) is not True:
            continue
        canonical = workload.raw.get("canonical_max_contract") or {}
        mode = canonical.get("mode")
        role = canonical.get("result_role")
        scenario = workload.scenario
        data_mode = canonical.get("data_mode")
        gate = canonical.get("quality") or {}
        protocol = workload.raw.get("measurement_protocol") or {}
        if (
            not isinstance(mode, str)
            or role not in RESULT_ROLES
            or not isinstance(scenario, str)
            or not isinstance(data_mode, str)
            or not isinstance(protocol, dict)
            or not isinstance(gate, dict)
        ):
            raise ValueError(
                f"{workload.id}: canonical evidence contract is incomplete"
            )
        identifier = case_id(workload.id, "max", mode, None)
        cases[identifier] = EvidenceCase(
            case_id=identifier,
            workload=workload,
            profile="max",
            mode=mode,
            phase=None,
            result_role=str(role),
            scenario=scenario,
            data_mode=data_mode,
            measurement_protocol=protocol,
            gate=gate,
            canonical_seed=_canonical_seed(workload),
        )

        inference = (workload.raw.get("mode_contracts") or {}).get("inference") or {}
        for phase, contract in (inference.get("phases") or {}).items():
            if not isinstance(contract, dict):
                raise ValueError(f"{workload.id}: phase {phase!r} is not a mapping")
            role = contract.get("result_role")
            scenario = contract.get("scenario")
            protocol = contract.get("measurement_protocol") or {}
            gate = contract.get("quality") or {}
            if (
                role not in RESULT_ROLES
                or not isinstance(scenario, str)
                or not isinstance(protocol, dict)
                or not isinstance(gate, dict)
            ):
                raise ValueError(
                    f"{workload.id}: phase {phase!r} evidence contract is incomplete"
                )
            identifier = case_id(workload.id, "max", "inference", str(phase))
            if identifier in cases:
                raise ValueError(f"duplicate evidence case {identifier}")
            cases[identifier] = EvidenceCase(
                case_id=identifier,
                workload=workload,
                profile="max",
                mode="inference",
                phase=str(phase),
                result_role=str(role),
                scenario=scenario,
                data_mode="checkpoint-backed",
                measurement_protocol=protocol,
                gate=gate,
                canonical_seed=_canonical_seed(workload),
            )
    return dict(sorted(cases.items()))


def git_repository_root() -> Path:
    try:
        return Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot locate Git repository for {ROOT}") from exc


def _source_blob(source_git_sha: str, relative_path: str) -> bytes:
    git_root = git_repository_root()
    project_prefix = ROOT.resolve().relative_to(git_root).as_posix()
    object_path = f"{source_git_sha}:{project_prefix}/{relative_path}"
    try:
        return subprocess.run(
            ["git", "show", object_path],
            cwd=git_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot read {object_path}") from exc


def source_sweep_tool_sha256(source_git_sha: str) -> str:
    return sha256_bytes(_source_blob(source_git_sha, "tools/run_reference_sweep.py"))


@contextmanager
def source_project_checkout(source_git_sha: str):
    """Yield a clean sparse checkout of the exact evidence source commit."""
    git_root = git_repository_root()
    project_prefix = ROOT.resolve().relative_to(git_root).as_posix()
    with tempfile.TemporaryDirectory(prefix="mlperf-edu-evidence-source-") as tmp:
        checkout = Path(tmp) / "repository"
        commands = (
            [
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                "--quiet",
                str(git_root),
                str(checkout),
            ],
            [
                "git",
                "-C",
                str(checkout),
                "sparse-checkout",
                "set",
                "--cone",
                project_prefix,
            ],
            [
                "git",
                "-C",
                str(checkout),
                "checkout",
                "--quiet",
                "--detach",
                source_git_sha,
            ],
        )
        try:
            for command in commands:
                subprocess.run(command, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                f"cannot create clean checkout for source commit {source_git_sha}"
            ) from exc
        project_root = checkout / Path(*project_prefix.split("/"))
        if not project_root.is_dir():
            raise ValueError(f"historical checkout lacks {project_prefix}")
        yield project_root


def _reject_duplicate_keys(label: str):
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    return hook


def load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    try:
        payload = json.loads(
            data,
            object_pairs_hook=_reject_duplicate_keys(label),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: JSON root must be an object")
    return payload, data


def _numeric(value: object, *, label: str, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{label}: expected a finite {qualifier}number")
    return float(value)


def aggregate(values: list[float]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot aggregate an empty reference series")
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _require_match(label: str, actual: object, expected: object) -> None:
    if not check_taxonomy.numbers_match(actual, expected):
        raise ValueError(f"{label}: {actual!r} does not match {expected!r}")


def validate_host_power_record(record: object, *, label: str) -> None:
    """Independently prove stable external power and disabled Low Power Mode."""
    if not isinstance(record, dict):
        raise ValueError(f"{label}: host power record is missing")
    if record.get("policy") != POWER_STABILITY_POLICY:
        raise ValueError(
            f"{label}: host power policy does not match the release contract"
        )
    if record.get("promotion_conditions_required") is not True:
        raise ValueError(f"{label}: promotion power conditions were not required")
    if record.get("stable") is not True or record.get("invalid_reasons") != []:
        raise ValueError(f"{label}: host power record is not stable")
    snapshots: dict[str, Mapping[str, Any]] = {}
    for boundary in ("before", "after"):
        snapshot = record.get(boundary)
        if not isinstance(snapshot, dict):
            raise ValueError(f"{label}: {boundary} host power snapshot is missing")
        snapshots[boundary] = snapshot
        if snapshot.get("schema") != HOST_POWER_STATE_SCHEMA:
            raise ValueError(f"{label}: {boundary} host power schema is unsupported")
        if snapshot.get("supported") is not True:
            raise ValueError(
                f"{label}: {boundary} host power monitoring is unsupported"
            )
        if snapshot.get("source") != "external":
            raise ValueError(f"{label}: {boundary} host power source is not external")
        if snapshot.get("low_power_mode") is not False:
            raise ValueError(f"{label}: {boundary} Low Power Mode is not disabled")
        if snapshot.get("query_errors") != []:
            raise ValueError(f"{label}: {boundary} host power query has errors")
        captured_at = snapshot.get("captured_at")
        if not isinstance(captured_at, str) or not captured_at:
            raise ValueError(f"{label}: {boundary} capture timestamp is missing")
    before = snapshots["before"]
    after = snapshots["after"]
    for field in ("provider", "platform", "source", "power_mode"):
        if before.get(field) != after.get(field):
            raise ValueError(f"{label}: host power {field} changed during execution")
    if before.get("provider") == "linux-sysfs-clock-boottime":
        before_offset = _numeric(
            before.get("suspend_clock_offset_seconds"),
            label=f"{label} before suspend clock offset",
        )
        after_offset = _numeric(
            after.get("suspend_clock_offset_seconds"),
            label=f"{label} after suspend clock offset",
        )
        if abs(after_offset - before_offset) > 1.0:
            raise ValueError(f"{label}: host suspend clock changed during execution")
    else:
        for field in ("last_sleep_epoch", "last_wake_epoch"):
            if before.get(field) != after.get(field):
                raise ValueError(f"{label}: host {field} changed during execution")


def _target_met(value: float, gate: Mapping[str, Any], tolerance: float) -> bool:
    target = _numeric(gate.get("target"), label="quality target")
    direction = gate.get("direction")
    if direction == "higher":
        return value + tolerance >= target
    if direction == "lower":
        return value - tolerance <= target
    if direction == "equal":
        return math.isclose(value, target, rel_tol=0.0, abs_tol=tolerance)
    raise ValueError(f"unsupported gate direction {direction!r}")


def _repeatability(values: list[float], case: EvidenceCase) -> dict[str, Any]:
    stats = aggregate(values)
    limit = _numeric(
        case.measurement_protocol.get("repeatability_limit"),
        label=f"{case.case_id} repeatability limit",
    )
    mean = float(stats["mean"])
    coefficient = float(stats["stdev"]) / mean
    return {
        "metric": case.measurement_protocol.get("repeatability_metric"),
        "coefficient_of_variation": coefficient,
        "limit": limit,
        "passed": coefficient <= limit,
    }


def validate_summary_structure(
    path: Path,
    payload: Mapping[str, Any],
    *,
    case: EvidenceCase,
    source_git_sha: str,
    sweep_tool_sha256: str,
) -> None:
    """Recompute aggregate and promotion semantics from compact run rows."""
    expected_fields = {
        "schema": SUMMARY_SCHEMA,
        "status": "valid",
        "evidence_tier": "promotion-candidate",
        "eligible_for_promotion": True,
        "eligible_for_public_baseline": False,
        "workload": case.workload.id,
        "canonical_workload": case.workload.id,
        "profile": case.profile,
        "mode": case.mode,
        "phase": case.phase,
        "result_role": case.result_role,
    }
    failures = [
        f"{field}={payload.get(field)!r}, expected {expected!r}"
        for field, expected in expected_fields.items()
        if payload.get(field) != expected
    ]
    evidence_id = payload.get("evidence_id")
    if not isinstance(evidence_id, str) or not check_taxonomy.EVIDENCE_ID_RE.fullmatch(
        evidence_id
    ):
        failures.append("evidence_id is missing or unsafe")
    if payload.get("invalid_reasons") != []:
        failures.append("invalid_reasons is not empty")
    if payload.get("power_stability_policy") != POWER_STABILITY_POLICY:
        failures.append("power_stability_policy does not match the release contract")
    source = payload.get("source") or {}
    expected_source = {
        "git_sha": source_git_sha,
        "git_dirty": False,
        "git_status_sha256": EMPTY_SHA256,
        "git_patch_sha256": EMPTY_SHA256,
        "tool_sha256": sweep_tool_sha256,
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            failures.append(f"source.{field} does not match the clean source commit")
    if failures:
        raise ValueError(f"{path}: " + "; ".join(failures))

    runs = payload.get("runs")
    # The registry declares how many timing runs a reference result needs. It
    # used to be fixed at five here, which meant lowering the protocol to match
    # the single-run acceptance rule broke the importer rather than relaxing it.
    expected_runs = case.measurement_protocol.get("outer_reference_runs")
    if not isinstance(expected_runs, int) or expected_runs < 1:
        raise ValueError(
            f"{case.case_id}: reference protocol must declare a positive "
            "outer_reference_runs"
        )
    # More repetitions than the protocol requires is stronger evidence, so the
    # floor is checked rather than an exact count. Records measured under the
    # earlier five-run protocol therefore remain importable unchanged.
    if not isinstance(runs, list) or len(runs) < expected_runs:
        raise ValueError(f"{path}: expected at least {expected_runs} runs")
    seeds_requested = payload.get("seeds_requested")
    if not isinstance(seeds_requested, list) or len(seeds_requested) != len(runs):
        raise ValueError(f"{path}: seeds_requested must cover every run")
    if any(seed != case.canonical_seed for seed in seeds_requested):
        raise ValueError(f"{path}: promotion runs must repeat the canonical seed")

    expected_cooldown_seconds = case.measurement_protocol.get(
        "outer_inter_execution_cooldown_seconds"
    )
    if (
        isinstance(expected_cooldown_seconds, bool)
        or not isinstance(expected_cooldown_seconds, (int, float))
        or not math.isfinite(float(expected_cooldown_seconds))
        or not 0 <= float(expected_cooldown_seconds) <= 300
    ):
        raise ValueError(
            f"{case.case_id}: outer inter-execution cooldown must be between 0 and 300"
        )
    expected_outer_executions = [
        {
            "execution_index": position,
            "seed": case.canonical_seed,
            "fresh_process": True,
            "cooldown_before_seconds": (
                0.0 if position == 1 else float(expected_cooldown_seconds)
            ),
        }
        # Built over the runs actually supplied, not the declared minimum, so a
        # record with more repetitions than the protocol requires still
        # validates every execution it contains.
        for position in range(1, len(runs) + 1)
    ]
    stabilization = payload.get("inter_execution_stabilization")
    if not isinstance(stabilization, dict):
        raise ValueError(f"{path}: inter-execution stabilization policy is missing")
    for field, expected in {
        "scope": "outer-process-executions",
        "applies": True,
        "applicability": (
            "all public-candidate score-bearing and performance-bearing workloads"
        ),
        "mode": "fixed-delay-between-fresh-processes",
        "execution_unit": "one fresh Python subprocess per repetition",
        "process_execution_count": len(runs),
        "configured_cooldown_seconds": float(expected_cooldown_seconds),
        "maximum_cooldown_seconds": 300.0,
        "first_execution_has_no_cooldown": True,
        "timing_scope": OUTER_EXECUTION_TIMING_SCOPE,
        "executions": expected_outer_executions,
    }.items():
        if stabilization.get(field) != expected:
            raise ValueError(
                f"{path}: inter_execution_stabilization.{field} does not match "
                "the protocol"
            )

    expected_preconditioning_runs = case.measurement_protocol.get(
        "outer_preconditioning_runs"
    )
    if (
        isinstance(expected_preconditioning_runs, bool)
        or not isinstance(expected_preconditioning_runs, int)
        or expected_preconditioning_runs < 0
    ):
        raise ValueError(
            f"{case.case_id}: outer_preconditioning_runs must be a nonnegative integer"
        )
    preconditioning = payload.get("preconditioning")
    if not isinstance(preconditioning, dict):
        raise ValueError(f"{path}: preconditioning policy is missing")
    expected_preconditioning_executions = [
        {
            "execution_index": position,
            "seed": case.canonical_seed,
            "fresh_process": True,
            "output_group": "preconditioning",
        }
        for position in range(1, expected_preconditioning_runs + 1)
    ]
    for field, expected in {
        "scope": "outer-process-preconditioning",
        "applies": bool(expected_preconditioning_runs),
        "mode": (
            "complete-canonical-executions"
            if expected_preconditioning_runs
            else "not-applied"
        ),
        "execution_unit": (
            "one complete canonical workload in a fresh Python subprocess"
        ),
        "process_execution_count": expected_preconditioning_runs,
        "cooldown_between_executions_seconds": 0.0,
        "cooldown_before_first_measured_execution_seconds": 0.0,
        "timing_scope": PRECONDITIONING_TIMING_SCOPE,
        "executions": expected_preconditioning_executions,
    }.items():
        if preconditioning.get(field) != expected:
            raise ValueError(
                f"{path}: preconditioning.{field} does not match the protocol"
            )
    preconditioning_rows = preconditioning.get("runs")
    if (
        not isinstance(preconditioning_rows, list)
        or len(preconditioning_rows) != expected_preconditioning_runs
    ):
        raise ValueError(
            f"{path}: expected exactly {expected_preconditioning_runs} "
            "preconditioning runs"
        )

    primary_values: list[float] = []
    quality_values: list[float] = []
    wall_values: list[float] = []
    fingerprints: set[str] = set()
    primary_metric = case.measurement_protocol.get("primary_metric")
    gate_metric = case.gate.get("metric")
    gate_key = case.gate.get("metric_key")
    tolerance = float(case.workload.quality_tolerance or 0.0)
    for position, run in enumerate(runs, start=1):
        label = f"{path} run {position}"
        if not isinstance(run, dict):
            raise ValueError(f"{label}: run row is not an object")
        required = {
            "execution_index": position,
            "requested_seed": case.canonical_seed,
            "report_recorded_seed": case.canonical_seed,
            "manifest_recorded_seed": case.canonical_seed,
            "seed_match": True,
            "status": "passed",
            "execution_ok": True,
            "evidence_valid": True,
            "timed_out": False,
            "manifest_verified": True,
            "data_mode": case.data_mode,
            "scenario": case.scenario,
            "manifest_scenario": case.scenario,
            "registry_scenario": case.scenario,
            "primary_metric_declared": primary_metric,
            "reference_metric_role": "performance",
            "result_role": case.result_role,
            "quality_target_met": True,
            "outer_process_execution": expected_outer_executions[position - 1],
        }
        for field, expected in required.items():
            if run.get(field) != expected:
                raise ValueError(
                    f"{label}: {field}={run.get(field)!r}, expected {expected!r}"
                )
        if run.get("invalid_reasons") != []:
            raise ValueError(f"{label}: invalid_reasons is not empty")
        validate_host_power_record(run.get("host_power"), label=label)
        reproduce_policy = (run.get("reproduce") or {}).get("reference_sweep") or {}
        if (
            reproduce_policy.get("outer_process_execution")
            != expected_outer_executions[position - 1]
        ):
            raise ValueError(f"{label}: reproduction execution policy mismatch")
        cooldown_record = reproduce_policy.get("inter_execution_cooldown") or {}
        expected_cooldown_record = {
            "cli_option": "--inter-execution-cooldown-seconds",
            "configured_seconds": float(expected_cooldown_seconds),
            "applied_before_this_execution_seconds": expected_outer_executions[
                position - 1
            ]["cooldown_before_seconds"],
            "applies": True,
        }
        if cooldown_record != expected_cooldown_record:
            raise ValueError(f"{label}: reproduction cooldown policy mismatch")
        if reproduce_policy.get("timing_scope") != OUTER_EXECUTION_TIMING_SCOPE:
            raise ValueError(f"{label}: reproduction timing scope mismatch")
        primary_values.append(
            _numeric(
                run.get("primary_metric_value"),
                label=f"{label} primary metric",
                positive=True,
            )
        )
        wall_values.append(
            _numeric(run.get("wall_seconds"), label=f"{label} wall time", positive=True)
        )
        fingerprint = run.get("comparison_fingerprint_sha256")
        if not isinstance(fingerprint, str) or not SHA256_HEX_RE.fullmatch(fingerprint):
            raise ValueError(f"{label}: comparison fingerprint is invalid")
        fingerprints.add(fingerprint)
        promotion = run.get("promotion_contract") or {}
        for field, expected in {
            "status": "passed",
            "promotion_eligible": True,
            "mode": case.mode,
            "phase": case.phase,
            "result_role": case.result_role,
        }.items():
            if promotion.get(field) != expected:
                raise ValueError(f"{label}: promotion_contract.{field} mismatch")
        if promotion.get("issues") != []:
            raise ValueError(f"{label}: promotion contract contains issues")
        grade = run.get("grade") or {}
        if grade.get("passed") is not True or grade.get("target_met") is not True:
            raise ValueError(f"{label}: grade did not pass")
        gate_value = _numeric(
            run.get("functional_metric_value"),
            label=f"{label} gate metric",
        )
        if case.result_role == "score-bearing":
            _require_match(
                f"{label} quality metric declaration",
                run.get("quality_metric_declared"),
                gate_metric,
            )
            _require_match(
                f"{label} quality metric key", run.get("quality_metric_key"), gate_key
            )
            _require_match(
                f"{label} quality value", run.get("quality_value"), gate_value
            )
            if not _target_met(gate_value, case.gate, tolerance):
                raise ValueError(f"{label}: score quality gate was not met")
            quality_values.append(gate_value)
        else:
            _require_match(
                f"{label} functional metric declaration",
                run.get("functional_metric_declared"),
                gate_metric,
            )
            _require_match(
                f"{label} functional metric key",
                run.get("functional_metric_key"),
                gate_key,
            )
            if not _target_met(gate_value, case.gate, 0.0):
                raise ValueError(f"{label}: functional gate was not met")

    if len(fingerprints) != 1:
        raise ValueError(f"{path}: runs do not share one comparison fingerprint")
    for position, (run, execution) in enumerate(
        zip(preconditioning_rows, expected_preconditioning_executions), start=1
    ):
        label = f"{path} preconditioning run {position}"
        if not isinstance(run, dict):
            raise ValueError(f"{label}: run row is not an object")
        required = {
            "execution_index": position,
            "requested_seed": case.canonical_seed,
            "report_recorded_seed": case.canonical_seed,
            "manifest_recorded_seed": case.canonical_seed,
            "seed_match": True,
            "status": "passed",
            "execution_ok": True,
            "evidence_valid": True,
            "timed_out": False,
            "manifest_verified": True,
            "data_mode": case.data_mode,
            "scenario": case.scenario,
            "manifest_scenario": case.scenario,
            "registry_scenario": case.scenario,
            "primary_metric_declared": primary_metric,
            "reference_metric_role": "performance",
            "result_role": case.result_role,
            "quality_target_met": True,
            "preconditioning_execution": execution,
        }
        for field, expected in required.items():
            if run.get(field) != expected:
                raise ValueError(
                    f"{label}: {field}={run.get(field)!r}, expected {expected!r}"
                )
        if run.get("invalid_reasons") != []:
            raise ValueError(f"{label}: invalid_reasons is not empty")
        validate_host_power_record(run.get("host_power"), label=label)
        reproduce_policy = (run.get("reproduce") or {}).get("reference_sweep") or {}
        expected_reproduce_policy = {
            "preconditioning_execution": execution,
            "aggregate_inclusion": "excluded",
            "timing_scope": PRECONDITIONING_TIMING_SCOPE,
        }
        if reproduce_policy != expected_reproduce_policy:
            raise ValueError(f"{label}: preparation reproduction policy mismatch")
        _require_match(
            f"{label} comparison fingerprint",
            run.get("comparison_fingerprint_sha256"),
            next(iter(fingerprints)),
        )
    _require_match(
        f"{path} comparison fingerprint",
        payload.get("comparison_fingerprint_sha256"),
        next(iter(fingerprints)),
    )
    _require_match(
        f"{path} primary aggregate",
        (payload.get("aggregate") or {}).get("primary_metric"),
        aggregate(primary_values),
    )
    _require_match(
        f"{path} wall aggregate",
        (payload.get("aggregate") or {}).get("wall_seconds"),
        aggregate(wall_values),
    )
    repeatability = _repeatability(primary_values, case)
    _require_match(
        f"{path} repeatability",
        payload.get("primary_metric_repeatability"),
        repeatability,
    )
    if repeatability["passed"] is not True:
        raise ValueError(f"{path}: primary timing repeatability exceeds its limit")

    if case.result_role == "score-bearing":
        _require_match(
            f"{path} quality aggregate",
            (payload.get("aggregate") or {}).get("quality"),
            aggregate(quality_values),
        )
        expected_gate = {
            "metric": gate_metric,
            "target": case.gate.get("target"),
            "direction": case.gate.get("direction"),
            "tolerance": tolerance,
            "all_runs_must_pass": True,
        }
        _require_match(
            f"{path} quality gate", payload.get("quality_gate"), expected_gate
        )
        if payload.get("functional_gate") is not None:
            raise ValueError(f"{path}: score-bearing summary has a functional gate")
        if (payload.get("acceptance") or {}).get("all_runs_passed") is not True:
            raise ValueError(f"{path}: not every score-bearing run passed")
    else:
        if (payload.get("aggregate") or {}).get("quality") is not None:
            raise ValueError(f"{path}: performance summary has a quality aggregate")
        if any(
            payload.get(field) is not None
            for field in (
                "quality_metric",
                "quality_target",
                "quality_direction",
                "quality_gate",
            )
        ):
            raise ValueError(
                f"{path}: performance summary exposes score-quality fields"
            )
        expected_functional = {
            "metric": gate_metric,
            "metric_key": gate_key,
            "condition": "Every run must pass the canonical functional gate.",
            "target": case.gate.get("target"),
        }
        _require_match(
            f"{path} functional gate",
            payload.get("functional_gate"),
            expected_functional,
        )
        acceptance = payload.get("acceptance") or {}
        if acceptance.get("passed") is not True or acceptance.get("value") != len(runs):
            raise ValueError(f"{path}: performance functional acceptance failed")


def resolve_indexed_file(root: Path, relative_path: object, *, label: str) -> Path:
    text = str(relative_path or "")
    if not check_taxonomy.is_safe_posix_relative_path(text):
        raise ValueError(f"{label}: path is unsafe or missing")
    unresolved = root / Path(*text.split("/"))
    if unresolved.is_symlink():
        raise ValueError(f"{label}: symlink artifacts are not allowed")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}: path escapes its evidence root") from exc
    if not resolved.is_file():
        raise ValueError(f"{label}: indexed file is missing")
    return resolved


def verify_file_claim(
    root: Path,
    claim: Mapping[str, Any],
    *,
    label: str,
    cache: dict[Path, tuple[int, str]],
) -> Path:
    path = resolve_indexed_file(root, claim.get("path"), label=label)
    if path not in cache:
        cache[path] = (path.stat().st_size, sha256_file(path))
    size, digest = cache[path]
    if claim.get("n_bytes") != size or claim.get("sha256") != digest:
        raise ValueError(f"{label}: indexed size or SHA-256 does not match")
    return path


_PROMOTION_SCRIPT = r"""
import json
import sys
from pathlib import Path

from mlperf.contracts import evaluate_promotion_contract
from mlperf.registry import load_registry

report = json.loads(Path(sys.argv[1]).read_text())
workload = load_registry()[sys.argv[2]]
print(json.dumps(evaluate_promotion_contract(workload, report), sort_keys=True))
"""


def recompute_promotion_contract(
    report_path: Path, case: EvidenceCase, *, source_project_root: Path
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_project_root / "src")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _PROMOTION_SCRIPT,
                str(report_path),
                case.workload.id,
            ],
            cwd=source_project_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{report_path}: cannot recompute the historical promotion contract"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"{report_path}: recomputed promotion contract is not an object"
        )
    return value


def _verify_lineage_stage(
    evidence_root: Path,
    attempt_root: Path,
    payload: Mapping[str, Any],
    *,
    source_project_root: Path,
    retained: set[Path],
    cache: dict[Path, tuple[int, str]],
) -> None:
    lineage = payload.get("nanogpt_training_lineage")
    if not isinstance(lineage, dict):
        return
    if lineage.get("required") is not True or lineage.get("status") != "staged":
        raise ValueError(f"{attempt_root}: required training lineage was not staged")
    digest = lineage.get("package_sha256")
    archives = [
        path
        for path in evidence_root.rglob("*.zip")
        if path.is_file() and not path.is_symlink() and sha256_file(path) == digest
    ]
    if len(archives) != 1:
        raise ValueError(
            f"{attempt_root}: expected one retained lineage archive, found {len(archives)}"
        )
    checks = verify_package_archive(archives[0], repo_root=source_project_root)
    failed = [name for name, passed, _detail in checks if not passed]
    if failed:
        raise ValueError(
            f"{archives[0]}: lineage package verification failed: {failed}"
        )
    stage = resolve_indexed_file(
        attempt_root,
        f"{lineage.get('staged_root')}/package_index.json",
        label=f"{attempt_root} staged package index",
    )
    retained.add(stage)
    package_index, _ = load_json_object(stage, label=str(stage))
    if package_index.get("schema") != "mlperf-edu-package/0.2":
        raise ValueError(f"{stage}: unsupported package schema")
    for position, claim in enumerate(package_index.get("included_files") or []):
        if not isinstance(claim, dict):
            raise ValueError(f"{stage}: included file {position} is not an object")
        retained.add(
            verify_file_claim(
                stage.parent,
                claim,
                label=f"{stage} included file {position}",
                cache=cache,
            )
        )


def verify_external_evidence(
    evidence_root: Path,
    summary_path: Path,
    payload: Mapping[str, Any],
    data: bytes,
    *,
    case: EvidenceCase,
    source_project_root: Path,
    cache: dict[Path, tuple[int, str]],
) -> None:
    """Verify raw files, manifests, reports, and historical contracts."""
    attempt_root = summary_path.parent.resolve()
    sidecar = summary_path.with_suffix(".json.sha256")
    expected_sidecar = f"{hashlib.sha256(data).hexdigest()}  {summary_path.name}\n"
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError(f"{summary_path}: adjacent digest sidecar is missing")
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise ValueError(f"{summary_path}: adjacent digest sidecar does not match")
    retained = {summary_path.resolve(), sidecar.resolve()}
    manifest_cache: set[Path] = set()

    execution_groups = (
        (
            "preconditioning run",
            (payload.get("preconditioning") or {}).get("runs") or [],
        ),
        ("run", payload.get("runs") or []),
    )
    for group_label, execution_rows in execution_groups:
        for position, run in enumerate(execution_rows, start=1):
            label = f"{case.case_id} {group_label} {position}"
            claims: dict[str, Mapping[str, Any]] = {}
            for claim in run.get("artifacts") or []:
                if not isinstance(claim, dict):
                    raise ValueError(f"{label}: artifact claim is not an object")
                role = str(claim.get("role") or "")
                if not role or role in claims:
                    raise ValueError(f"{label}: artifact role is missing or duplicated")
                claims[role] = claim
                retained.add(
                    verify_file_claim(
                        attempt_root,
                        claim,
                        label=f"{label} artifact {role}",
                        cache=cache,
                    )
                )
            if "report" not in claims or "provenance" not in claims:
                raise ValueError(f"{label}: report or provenance artifact is missing")
            report_path = resolve_indexed_file(
                attempt_root, claims["report"].get("path"), label=f"{label} report"
            )
            manifest_path = resolve_indexed_file(
                attempt_root,
                claims["provenance"].get("path"),
                label=f"{label} provenance",
            )
            _require_match(
                f"{label} report path",
                run.get("report_path"),
                claims["report"].get("path"),
            )
            _require_match(
                f"{label} manifest path",
                run.get("manifest_path"),
                claims["provenance"].get("path"),
            )
            report, _ = load_json_object(report_path, label=f"{label} report")
            manifest, _ = load_json_object(manifest_path, label=f"{label} provenance")
            expected_report = {
                "workload": case.workload.id,
                "profile": case.profile,
                "mode": case.mode,
                "phase": case.phase,
                "scenario": case.scenario,
                "status": "passed",
                "seed": case.canonical_seed,
                "backend": run.get("backend"),
                "data_mode": case.data_mode,
            }
            for field, expected in expected_report.items():
                _require_match(f"{label} report.{field}", report.get(field), expected)
            metrics = report.get("metrics") or {}
            _require_match(
                f"{label} primary metric bytes",
                metrics.get(run.get("primary_metric_key")),
                run.get("primary_metric_value"),
            )
            _require_match(
                f"{label} gate metric bytes",
                metrics.get(run.get("functional_metric_key")),
                run.get("functional_metric_value"),
            )
            recomputed = recompute_promotion_contract(
                report_path, case, source_project_root=source_project_root
            )
            _require_match(
                f"{label} recomputed promotion contract",
                report.get("promotion_contract"),
                recomputed,
            )
            _require_match(
                f"{label} summary promotion contract",
                run.get("promotion_contract"),
                recomputed,
            )
            fingerprint = report.get("run_fingerprint") or {}
            _require_match(
                f"{label} comparison fingerprint",
                fingerprint.get("comparison_fingerprint_sha256"),
                run_comparison_fingerprint_sha256(fingerprint),
            )
            _require_match(
                f"{label} summary comparison fingerprint",
                run.get("comparison_fingerprint_sha256"),
                fingerprint.get("comparison_fingerprint_sha256"),
            )
            _require_match(
                f"{label} manifest workload", manifest.get("workload"), case.workload.id
            )
            _require_match(
                f"{label} manifest scenario", manifest.get("scenario"), case.scenario
            )
            manifest_seed = ((manifest.get("leaves") or {}).get("rng") or {}).get(
                "seed"
            )
            _require_match(f"{label} manifest seed", manifest_seed, case.canonical_seed)
            measurement = (manifest.get("leaves") or {}).get("measurement") or {}
            _require_match(
                f"{label} manifest report digest",
                measurement.get("report_file_sha256"),
                sha256_file(report_path),
            )
            _require_match(
                f"{label} manifest report size",
                measurement.get("n_bytes"),
                report_path.stat().st_size,
            )
            if manifest_path not in manifest_cache:
                verification = verify_provd(
                    manifest_path, repo_root=source_project_root
                )
                failed = [
                    name for name, passed, _detail in verification.checks if not passed
                ]
                if failed:
                    raise ValueError(
                        f"{label}: provenance verification failed: {failed}"
                    )
                manifest_cache.add(manifest_path)

    _verify_lineage_stage(
        evidence_root,
        attempt_root,
        payload,
        source_project_root=source_project_root,
        retained=retained,
        cache=cache,
    )
    actual = {
        path.resolve()
        for path in attempt_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    unindexed = sorted(actual - retained)
    if unindexed:
        relative = [path.relative_to(attempt_root).as_posix() for path in unindexed]
        raise ValueError(
            f"{summary_path}: retained attempt has unindexed files {relative}"
        )


def rejected_attempt_reason(payload: Mapping[str, Any]) -> str | None:
    reasons = payload.get("invalid_reasons")
    if (
        payload.get("status") == "invalid"
        and payload.get("eligible_for_promotion") is False
        and isinstance(reasons, list)
        and reasons
        and all(isinstance(reason, str) and reason for reason in reasons)
    ):
        return "; ".join(reasons)
    return None


def _artifact_digest(run: Mapping[str, Any], role: str) -> str | None:
    values = {
        str(claim.get("sha256"))
        for claim in run.get("artifacts") or []
        if isinstance(claim, dict) and claim.get("role") == role
    }
    return next(iter(values)) if len(values) == 1 else None


def validate_lineage_closure(
    selected: Mapping[str, tuple[Path, Mapping[str, Any], bytes]],
) -> dict[str, dict[str, Any]]:
    """Bind every causal inference phase to one median-quality training run."""
    training_id = "causal-language-modeling__max__training"
    training_path, training, training_bytes = selected[training_id]
    training_runs = training.get("runs") or []
    quality_median = ((training.get("aggregate") or {}).get("quality") or {}).get(
        "median"
    )
    # The timing protocol declares one run, so the binding needs a training
    # record with at least one measurement rather than exactly five.
    if not isinstance(training_runs, list) or not training_runs:
        raise ValueError(f"{training_path}: causal training evidence has no runs")

    bindings: dict[str, dict[str, Any]] = {}
    shared_identity: tuple[str, str, str, str] | None = None
    for phase in ("full", "prefill", "decode"):
        identifier = f"causal-language-modeling__max__inference__{phase}"
        phase_path, payload, _data = selected[identifier]
        lineage = payload.get("nanogpt_training_lineage") or {}
        package_digest = lineage.get("package_sha256")
        if (
            lineage.get("required") is not True
            or lineage.get("status") != "staged"
            or not isinstance(package_digest, str)
            or not PREFIXED_SHA256_RE.fullmatch(package_digest)
        ):
            raise ValueError(f"{phase_path}: causal inference lineage is incomplete")
        phase_runs = payload.get("runs") or []
        source_digests: dict[str, str] = {}
        for phase_role, training_role in (
            ("checkpoint", "checkpoint"),
            ("source_training_report", "report"),
            ("source_training_provenance", "provenance"),
        ):
            values = {
                _artifact_digest(run, phase_role)
                for run in phase_runs
                if isinstance(run, dict)
            }
            if None in values or len(values) != 1:
                raise ValueError(
                    f"{phase_path}: phase runs do not share one {phase_role} digest"
                )
            source_digests[training_role] = str(next(iter(values)))
        matches = [
            run
            for run in training_runs
            if all(
                _artifact_digest(run, role) == digest
                for role, digest in source_digests.items()
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{phase_path}: source package does not select exactly one committed "
                "training run"
            )
        selected_run = matches[0]
        if not check_taxonomy.numbers_match(
            selected_run.get("quality_value"), quality_median
        ):
            raise ValueError(
                f"{phase_path}: source package must select the median-quality "
                "training run"
            )
        identity = (
            package_digest,
            source_digests["checkpoint"],
            source_digests["report"],
            source_digests["provenance"],
        )
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError(
                f"{phase_path}: causal inference phases do not share one training "
                "package"
            )
        bindings[identifier] = {
            "source_training_case_id": training_id,
            "source_training_evidence_id": training.get("evidence_id"),
            "source_training_evidence_sha256": sha256_bytes(training_bytes),
            "source_training_execution_index": selected_run.get("execution_index"),
            "source_training_checkpoint_sha256": source_digests["checkpoint"],
            "source_training_report_sha256": source_digests["report"],
            "source_training_provenance_sha256": source_digests["provenance"],
            "source_training_package_sha256": package_digest,
        }
    return bindings


def _case_for_payload(
    payload: Mapping[str, Any], cases: Mapping[str, EvidenceCase]
) -> EvidenceCase | None:
    workload = payload.get("workload")
    profile = payload.get("profile")
    mode = payload.get("mode")
    phase = payload.get("phase")
    if not all(isinstance(value, str) for value in (workload, profile, mode)):
        return None
    try:
        identifier = case_id(
            str(workload), str(profile), str(mode), str(phase) if phase else None
        )
    except ValueError:
        return None
    return cases.get(identifier)


def discover_summaries(
    evidence_root: Path,
    *,
    source_git_sha: str,
    sweep_tool_sha256: str,
    source_project_root: Path,
) -> tuple[
    dict[str, tuple[Path, dict[str, Any], bytes]],
    list[tuple[Path, str]],
    dict[str, dict[str, Any]],
]:
    cases = expected_cases()
    selected: dict[str, tuple[Path, dict[str, Any], bytes]] = {}
    rejected: list[tuple[Path, str]] = []
    cache: dict[Path, tuple[int, str]] = {}
    for path in sorted(evidence_root.rglob("evidence_summary.json")):
        if path.is_symlink():
            raise ValueError(f"{path}: summary may not be a symlink")
        resolved = path.resolve()
        try:
            resolved.relative_to(evidence_root.resolve())
        except ValueError as exc:
            raise ValueError(f"{path}: summary escapes the evidence root") from exc
        payload, data = load_json_object(resolved, label=str(resolved))
        case = _case_for_payload(payload, cases)
        if case is None:
            if payload.get("eligible_for_promotion") is True:
                raise ValueError(f"{path}: eligible summary names an unexpected case")
            continue
        rejection = rejected_attempt_reason(payload)
        if rejection is not None:
            rejected.append((resolved, rejection))
            continue
        validate_summary_structure(
            resolved,
            payload,
            case=case,
            source_git_sha=source_git_sha,
            sweep_tool_sha256=sweep_tool_sha256,
        )
        verify_external_evidence(
            evidence_root,
            resolved,
            payload,
            data,
            case=case,
            source_project_root=source_project_root,
            cache=cache,
        )
        if case.case_id in selected:
            raise ValueError(
                f"multiple eligible summaries found for {case.case_id}: "
                f"{selected[case.case_id][0]} and {resolved}"
            )
        selected[case.case_id] = (resolved, payload, data)
    missing = sorted(set(cases) - set(selected))
    if missing:
        raise ValueError("missing promotion evidence cases: " + ", ".join(missing))
    bindings = validate_lineage_closure(selected)
    return selected, rejected, bindings


def build_index(
    selected: Mapping[str, tuple[Path, Mapping[str, Any], bytes]],
    *,
    source_git_sha: str,
    source_lock: Mapping[str, Any],
    source_lock_bytes: bytes,
    lineage_bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    lineage_bindings = lineage_bindings or {}
    entries: list[dict[str, Any]] = []
    for identifier, (_path, payload, data) in sorted(selected.items()):
        destination = (
            Path("reference_results") / identifier / f"{payload['evidence_id']}.json"
        )
        entry = {
            "case_id": identifier,
            "workload": payload.get("workload"),
            "profile": payload.get("profile"),
            "mode": payload.get("mode"),
            "phase": payload.get("phase"),
            "result_role": payload.get("result_role"),
            "primary_metric": (payload.get("primary_metric") or {}).get("name"),
            "primary_metric_aggregate": (payload.get("aggregate") or {}).get(
                "primary_metric"
            ),
            "quality_metric": payload.get("quality_metric"),
            "quality_gate": payload.get("quality_gate"),
            "functional_gate": payload.get("functional_gate"),
            "acceptance": payload.get("acceptance"),
            "repeatability": payload.get("primary_metric_repeatability"),
            "device_requested": payload.get("device_requested"),
            "comparison_fingerprint_sha256": payload.get(
                "comparison_fingerprint_sha256"
            ),
            "evidence_id": payload.get("evidence_id"),
            "evidence_sha256": sha256_bytes(data),
            "path": destination.as_posix(),
        }
        if identifier in lineage_bindings:
            entry["source_training"] = dict(lineage_bindings[identifier])
        entries.append(entry)
    return {
        "schema": INDEX_SCHEMA,
        "source_git_sha": source_git_sha,
        "source_lock": {
            "path": SOURCE_LOCK_PATH,
            "schema": source_lock.get("schema"),
            "sha256": sha256_bytes(source_lock_bytes),
            "file_count": source_lock.get("file_count"),
            "contract_count": source_lock.get("contract_count"),
        },
        "workload_count": len({entry["workload"] for entry in entries}),
        "case_count": len(entries),
        "cases": entries,
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _require_safe_roots(evidence_root: Path) -> None:
    for output_root in OUTPUT_ROOTS:
        if _paths_overlap(evidence_root, output_root):
            raise ValueError(
                f"evidence source and destination overlap: {evidence_root} and {output_root}"
            )


def _sync_file(path: Path, expected: bytes, *, check: bool) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"generated destination escapes project root: {path}") from exc
    if path.is_symlink():
        raise ValueError(f"generated destination may not be a symlink: {path}")
    current = path.read_bytes() if path.is_file() else None
    if current == expected:
        return True
    if not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.evidence_root.is_dir():
        parser.error(f"evidence root is not a directory: {args.evidence_root}")
    if not SHA40_RE.fullmatch(args.source_git_sha):
        parser.error("--source-git-sha must be 40 lowercase hexadecimal characters")

    evidence_root = args.evidence_root.resolve()
    try:
        _require_safe_roots(evidence_root)
        sweep_tool_sha256 = source_sweep_tool_sha256(args.source_git_sha)
        source_lock = reference_source_lock.build_source_lock(
            args.source_git_sha, project_root=ROOT
        )
        source_lock_bytes = reference_source_lock.canonical_json_bytes(source_lock)
        with source_project_checkout(args.source_git_sha) as source_project_root:
            selected, rejected, lineage_bindings = discover_summaries(
                evidence_root,
                source_git_sha=args.source_git_sha,
                sweep_tool_sha256=sweep_tool_sha256,
                source_project_root=source_project_root,
            )
        index = build_index(
            selected,
            source_git_sha=args.source_git_sha,
            source_lock=source_lock,
            source_lock_bytes=source_lock_bytes,
            lineage_bindings=lineage_bindings,
        )
        index_bytes = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
        stale: list[Path] = []
        for output_root in OUTPUT_ROOTS:
            expected_paths = {
                output_root / "index.json",
                output_root / "source_lock.json",
            }
            for identifier, (_path, payload, data) in sorted(selected.items()):
                destination = (
                    output_root / identifier / f"{payload['evidence_id']}.json"
                )
                expected_paths.add(destination)
                if not _sync_file(destination, data, check=args.check):
                    stale.append(destination)
            for destination, data in (
                (output_root / "index.json", index_bytes),
                (output_root / "source_lock.json", source_lock_bytes),
            ):
                if not _sync_file(destination, data, check=args.check):
                    stale.append(destination)
            extra = (
                sorted(set(output_root.rglob("*.json")) - expected_paths)
                if output_root.is_dir()
                else []
            )
            if args.check:
                stale.extend(extra)
            else:
                for path in extra:
                    path.unlink()
                for path in sorted(output_root.rglob("*"), reverse=True):
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.check and stale:
        print("FAIL: imported reference evidence is out of date")
        for path in stale:
            print(f"  - {path}")
        return 1
    action = "verified" if args.check else "synchronized"
    print(f"PASS: {action} {len(selected)} promotion cases from {args.source_git_sha}.")
    if rejected:
        print(f"INFO: preserved and excluded {len(rejected)} rejected attempt(s).")
    for identifier, (_path, payload, data) in sorted(selected.items()):
        print(f"  {identifier}: {payload['evidence_id']} {sha256_bytes(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
