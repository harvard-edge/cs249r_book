#!/usr/bin/env python3
"""
Registry taxonomy and evidence linter for MLPerf EDU.

Enforces three invariants on workloads.yaml:

  (1) Every workload has a complete `regime` block with all three axes:
      working_set, arithmetic_intensity, dispatch.
  (2) Categorical `value` on each axis is one of the allowed strings.
  (3) When `value` is non-`unmeasured`, the numerical evidence supplied
      on that axis must be consistent with the classification thresholds
      declared below (drawn from Emer's iter-4 proposal).
  (4) Every measured classification or numerical observation names a
      committed roofline sidecar and its complete SHA-256 digest. The
      sidecar must match the digest and workload/axis claim, use measured
      content-addressed platform peaks, and record synchronized warmup and
      repeated measurement methodology.
  (5) Every declared quality asset or committed reference summary exists
      under the project root and matches its complete SHA-256 digest.

`unmeasured` is allowed as a value and tracked in the summary so pending
instrumentation remains visible without presenting estimates as measurements.

Run: python3 tools/check_taxonomy.py
Exit codes: 0 if all invariants hold, 1 if any error, 2 if no workloads
seen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path, PurePosixPath

import yaml

try:
    from tools import reference_source_lock
except ModuleNotFoundError:  # Direct `python tools/check_taxonomy.py` execution.
    import reference_source_lock  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOADS_YAML = REPO_ROOT / "workloads.yaml"

# Reference platform thresholds. These are *canonical* (textbook) M-series
# baseline numbers, intentionally chosen as the conservative lower bound
# of the Apple Silicon family so off-platform sidecars (M-Pro/Max/Ultra)
# rarely contradict the YAML. Per-machine actuals live in
# ~/.mlperf-edu/machine_caps_<hwfp>.json (bench/measure_peaks.py).
LLC_BYTES = 12 * 1024 * 1024  # M1 base LLC; M-Max class has more
RIDGE_FLOPS_PER_BYTE = 30  # M1 base fp32 ridge
PEAK_BW_GBPS = 68.25  # M1 base unified memory peak (informational)

VALID_VALUES = {
    "working_set": {"cache_resident", "dram_bound", "unmeasured"},
    "arithmetic_intensity": {"compute_bound", "bandwidth_bound", "unmeasured"},
    "dispatch": {"device_saturated", "dispatch_bound", "unmeasured"},
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREFIXED_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
EVIDENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f]")
REFERENCE_EVIDENCE_SCHEMA = "mlperf-edu-reference-evidence/0.4"
PROMOTION_REFERENCE_EVIDENCE_SCHEMA = "mlperf-edu-reference-evidence/0.7"
LEGACY_REFERENCE_EVIDENCE_SCHEMA = "mlperf-edu-reference-evidence/0.3"
SUPPORTED_REFERENCE_EVIDENCE_SCHEMAS = {
    LEGACY_REFERENCE_EVIDENCE_SCHEMA,
    REFERENCE_EVIDENCE_SCHEMA,
}
REFERENCE_INDEX_SCHEMA = "mlperf-edu-reference-index/0.2"
CASE_REFERENCE_INDEX_SCHEMA = "mlperf-edu-reference-index/0.3"
EMPTY_SHA256 = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SWEEP_TOOL_SHA256 = (
    "sha256:"
    + hashlib.sha256(
        (REPO_ROOT / "tools" / "run_reference_sweep.py").read_bytes()
    ).hexdigest()
)
MEASUREMENT_FIELDS = {
    "peak_bytes_per_step",
    "flops_per_byte",
    "utilization",
    "achieved_bw_gbps",
    "classification_rule",
    "observation_source",
    "measured_at",
    "platform_machine_class",
}


def numbers_match(actual: object, expected: object) -> bool:
    """Compare serialized benchmark numbers without hiding meaningful drift."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        )
    return actual == expected


def list_values_match(actual: object, expected: object) -> bool:
    if not isinstance(actual, list) or not isinstance(expected, list):
        return False
    return len(actual) == len(expected) and all(
        numbers_match(left, right) for left, right in zip(actual, expected, strict=True)
    )


def is_safe_posix_relative_path(value: object) -> bool:
    """Return whether *value* is a portable, strict POSIX relative path."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or CONTROL_CHARACTER_RE.search(value)
    ):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
        and ":" not in path.parts[0]
    )


def recompute_aggregate(values: list[float]) -> dict[str, int | float]:
    clean = [float(value) for value in values]
    return {
        "count": len(clean),
        "median": statistics.median(clean),
        "mean": statistics.fmean(clean),
        "min": min(clean),
        "max": max(clean),
        "stdev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


def check_regime(name: str, regime: dict) -> tuple[list[str], dict]:
    """Return (errors, axis_values) for one workload's regime block."""
    errors: list[str] = []
    values = {}

    for axis, allowed in VALID_VALUES.items():
        if axis not in regime:
            errors.append(f"{name}: missing axis '{axis}'")
            values[axis] = None
            continue
        block = regime[axis]
        if not isinstance(block, dict):
            errors.append(f"{name}.{axis}: expected dict, got {type(block).__name__}")
            values[axis] = None
            continue
        v = block.get("value")
        if v not in allowed:
            errors.append(f"{name}.{axis}: value '{v}' not in {sorted(allowed)}")
        values[axis] = v
        errors.extend(check_axis_evidence(name, axis, block))

    # Axis A — working_set numerical consistency.
    ws = regime.get("working_set", {})
    if ws.get("value") == "cache_resident":
        b = ws.get("peak_bytes_per_step")
        if b is not None and b > 0.5 * LLC_BYTES:
            errors.append(
                f"{name}.working_set: cache_resident but "
                f"peak_bytes_per_step={b:,} > 0.5 * LLC ({int(0.5 * LLC_BYTES):,})"
            )
    elif ws.get("value") == "dram_bound":
        b = ws.get("peak_bytes_per_step")
        if b is not None and b < 4 * LLC_BYTES:
            errors.append(
                f"{name}.working_set: dram_bound but "
                f"peak_bytes_per_step={b:,} < 4 * LLC ({int(4 * LLC_BYTES):,})"
            )

    # Axis B — arithmetic_intensity numerical consistency.
    ai = regime.get("arithmetic_intensity", {})
    fpb = ai.get("flops_per_byte")
    if ai.get("value") == "compute_bound" and fpb is not None:
        if fpb < 2 * RIDGE_FLOPS_PER_BYTE:
            errors.append(
                f"{name}.arithmetic_intensity: compute_bound but intensity "
                f"{fpb} < 2*ridge ({2 * RIDGE_FLOPS_PER_BYTE})"
            )
    if ai.get("value") == "bandwidth_bound" and fpb is not None:
        if fpb > 0.5 * RIDGE_FLOPS_PER_BYTE:
            errors.append(
                f"{name}.arithmetic_intensity: bandwidth_bound but intensity "
                f"{fpb} > 0.5*ridge ({0.5 * RIDGE_FLOPS_PER_BYTE})"
            )

    # Axis C — dispatch numerical consistency.
    d = regime.get("dispatch", {})
    util = d.get("utilization")
    if d.get("value") == "device_saturated" and util is not None and util < 0.50:
        errors.append(
            f"{name}.dispatch: device_saturated but utilization {util} < 0.50"
        )
    if d.get("value") == "dispatch_bound" and util is not None and util > 0.25:
        errors.append(f"{name}.dispatch: dispatch_bound but utilization {util} > 0.25")

    return errors, values


def check_axis_evidence(name: str, axis: str, block: dict) -> list[str]:
    """Require committed, content-addressed evidence for measured axis claims."""
    errors: list[str] = []
    value = block.get("value")
    sidecar = block.get("evidence_sidecar")
    digest = block.get("evidence_sha256")
    short_digest = block.get("evidence_sha256_short")
    has_measurement_fields = bool(MEASUREMENT_FIELDS.intersection(block))

    if short_digest is not None:
        errors.append(
            f"{name}.{axis}: evidence_sha256_short is not sufficient; declare the full evidence_sha256"
        )

    if value != "unmeasured" and not sidecar:
        errors.append(
            f"{name}.{axis}: measured classification '{value}' requires a committed evidence_sidecar"
        )
    if has_measurement_fields and not sidecar:
        errors.append(
            f"{name}.{axis}: numerical or observational evidence fields require a committed evidence_sidecar"
        )
    if sidecar and not digest:
        errors.append(
            f"{name}.{axis}: evidence_sidecar requires a full evidence_sha256"
        )
    if digest and not sidecar:
        errors.append(f"{name}.{axis}: evidence_sha256 requires evidence_sidecar")
    if not sidecar:
        return errors

    evidence_path = (REPO_ROOT / str(sidecar)).resolve()
    try:
        evidence_path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        errors.append(
            f"{name}.{axis}: evidence_sidecar escapes the project root: {sidecar}"
        )
        return errors
    if not evidence_path.is_file():
        errors.append(f"{name}.{axis}: evidence_sidecar does not exist: {sidecar}")
        return errors

    normalized_digest = str(digest or "").lower()
    if not SHA256_RE.fullmatch(normalized_digest):
        errors.append(
            f"{name}.{axis}: evidence_sha256 must be exactly 64 lowercase hexadecimal characters"
        )
        return errors
    actual_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if actual_digest != normalized_digest:
        errors.append(
            f"{name}.{axis}: evidence_sha256 mismatch for {sidecar} "
            f"(declared {normalized_digest}, actual {actual_digest})"
        )
        return errors

    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"{name}.{axis}: evidence_sidecar is not valid JSON: {exc}")
        return errors
    if payload.get("schema") != "mlperf-edu-roofline/1.0":
        errors.append(
            f"{name}.{axis}: evidence_sidecar has an unsupported roofline schema"
        )
    workload_id = name.split("/", 1)[-1]
    if payload.get("workload") != workload_id:
        errors.append(
            f"{name}.{axis}: evidence_sidecar workload is '{payload.get('workload')}', expected '{workload_id}'"
        )
    errors.extend(check_roofline_methodology(name, axis, payload))
    if value != "unmeasured":
        sidecar_axis = {
            "working_set": "axis_working_set",
            "arithmetic_intensity": "axis_arithmetic_intensity",
            "dispatch": "axis_dispatch",
        }[axis]
        inferred = (payload.get("regime_inference") or {}).get(sidecar_axis)
        if inferred != value:
            errors.append(
                f"{name}.{axis}: evidence_sidecar infers '{inferred}', but the registry claims '{value}'"
            )
    return errors


def check_roofline_methodology(name: str, axis: str, payload: dict) -> list[str]:
    """Reject sidecars that cannot prove platform peaks and measurement hygiene."""
    errors: list[str] = []
    label = f"{name}.{axis}"
    platform = payload.get("platform") or {}
    measurement = payload.get("measurement") or {}
    if platform.get("peak_source") != "measured":
        errors.append(f"{label}: roofline platform peaks are not marked as measured")
    errors.extend(
        check_declared_file(
            name,
            label=f"{axis}.platform.peak_evidence_file",
            relative_path=platform.get("peak_evidence_file"),
            digest=platform.get("peak_evidence_sha256"),
        )
    )
    if not platform.get("hardware_fingerprint"):
        errors.append(
            f"{label}: roofline platform lacks a complete hardware fingerprint"
        )
    if (
        not isinstance(measurement.get("n_iter"), int)
        or measurement.get("n_iter", 0) < 3
    ):
        errors.append(
            f"{label}: roofline measurement requires at least three timed iterations"
        )
    if (
        not isinstance(measurement.get("warmup_iterations"), int)
        or measurement.get("warmup_iterations", 0) < 1
    ):
        errors.append(
            f"{label}: roofline measurement requires at least one warmup iteration"
        )
    if measurement.get("synchronized") is not True:
        errors.append(
            f"{label}: roofline measurement is not marked as device-synchronized"
        )
    for field in ("operation_count_method", "byte_count_method"):
        if not measurement.get(field):
            errors.append(f"{label}: roofline measurement lacks {field}")
    for field in (
        "wall_time_s",
        "analytic_flops_total",
        "analytic_bytes_total",
        "achieved_FLOPS",
        "achieved_BW_GBps",
        "intensity_FLOPS_per_byte",
        "dispatch_utilization",
    ):
        value = measurement.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            errors.append(
                f"{label}: roofline measurement {field} is missing or non-finite"
            )
        elif field not in {"dispatch_utilization"} and float(value) <= 0:
            errors.append(f"{label}: roofline measurement {field} must be positive")
    return errors


def check_workload_evidence(name: str, body: dict) -> list[str]:
    """Verify non-taxonomy files explicitly declared as review evidence."""
    errors: list[str] = []
    if (body.get("public") or {}).get("status") == "performance-bearing":
        errors.extend(check_performance_reference_protocol(name, body))
    quality = body.get("quality_evaluation") or {}
    asset = quality.get("asset")
    asset_digest = quality.get("asset_sha256")
    if asset or asset_digest:
        errors.extend(
            check_declared_file(
                name,
                label="quality_evaluation.asset",
                relative_path=asset,
                digest=asset_digest,
            )
        )

    baseline = body.get("verified_baseline") or {}
    public_status = (body.get("public") or {}).get("status")
    if (
        public_status in {"score-bearing", "performance-bearing"}
        and baseline.get("evidence_status") != "committed-reference-summary"
    ):
        errors.append(
            f"{name}: {public_status} workload must cite a committed-reference-summary"
        )
    development_digest = baseline.get("development_summary_sha256")
    if development_digest is not None:
        if not SHA256_RE.fullmatch(str(development_digest)):
            errors.append(
                f"{name}: development_summary_sha256 must be a complete SHA-256 digest"
            )
        if not baseline.get("development_summary_id"):
            errors.append(
                f"{name}: development summary digest requires development_summary_id"
            )
        if baseline.get("development_summary_availability") not in {
            "local-handoff",
            "published",
        }:
            errors.append(
                f"{name}: development summary digest requires local-handoff or published availability"
            )
    if baseline.get("evidence_status") == "committed-reference-summary":
        lifecycle = (
            baseline.get("protocol_compatibility"),
            baseline.get("review_eligible"),
            baseline.get("replacement_required"),
        )
        superseded_lifecycle = ("superseded", False, True)
        current_lifecycle = ("current", True, False)
        lifecycle_declared = any(
            field in baseline
            for field in (
                "protocol_compatibility",
                "replacement_required",
            )
        )
        protocol_superseded = lifecycle == superseded_lifecycle
        if lifecycle_declared and not protocol_superseded:
            if lifecycle != current_lifecycle:
                errors.append(
                    f"{name}: current evidence lifecycle must be exactly "
                    "protocol_compatibility=current, review_eligible=true, "
                    "replacement_required=false"
                )
        file_errors = check_declared_file(
            name,
            label="verified_baseline.evidence_file",
            relative_path=baseline.get("evidence_file"),
            digest=baseline.get("evidence_sha256"),
        )
        errors.extend(file_errors)
        if not protocol_superseded and baseline.get("review_eligible") is not True:
            errors.append(
                f"{name}: committed-reference-summary must set verified_baseline.review_eligible to true"
            )
        availability = baseline.get("reference_package_availability")
        publication = baseline.get("external_publication_status")
        if availability not in {"local-handoff", "published"}:
            errors.append(
                f"{name}: committed-reference-summary must declare reference_package_availability as local-handoff or published"
            )
        if availability == "local-handoff" and publication != "pending":
            errors.append(
                f"{name}: local-handoff reference package must declare external_publication_status as pending"
            )
        if availability == "published":
            if publication != "published":
                errors.append(
                    f"{name}: published reference package must declare external_publication_status as published"
                )
            if not baseline.get("external_publication_url"):
                errors.append(
                    f"{name}: published reference package must declare external_publication_url"
                )
        if not file_errors:
            evidence_path = (REPO_ROOT / str(baseline.get("evidence_file"))).resolve()
            try:
                payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(
                    f"{name}: verified_baseline.evidence_file is not valid JSON: {exc}"
                )
            else:
                if protocol_superseded:
                    errors.extend(check_historical_reference_summary(name, payload))
                elif payload.get("schema") == PROMOTION_REFERENCE_EVIDENCE_SCHEMA:
                    errors.extend(
                        check_promoted_case_summary(name, body, baseline, payload)
                    )
                else:
                    errors.extend(check_reference_summary(name, body, payload))
    elif baseline.get("review_eligible") is True:
        errors.append(
            f"{name}: verified_baseline.review_eligible may be true only for a committed-reference-summary"
        )
    return errors


def check_performance_reference_protocol(name: str, body: dict) -> list[str]:
    errors: list[str] = []
    protocol = body.get("performance_reference_protocol")
    if not isinstance(protocol, dict):
        return [
            f"{name}: performance-bearing workload lacks performance_reference_protocol"
        ]
    required = (
        "profile",
        "reference_runs",
        "backend",
        "machine_class",
        "dataset_mode",
        "seeds",
        "aggregation",
        "repeatability_metric",
        "repeatability_limit",
        "repeatability_action",
        "functional_acceptance",
        "artifact_policy",
        "rerun_policy",
    )
    for field in required:
        if not protocol.get(field):
            errors.append(f"{name}: performance_reference_protocol.{field} is missing")
    seeds = protocol.get("seeds")
    reference_runs = protocol.get("reference_runs")
    if (
        isinstance(seeds, list)
        and isinstance(reference_runs, int)
        and len(seeds) != reference_runs
    ):
        errors.append(
            f"{name}: performance_reference_protocol seeds do not match reference_runs"
        )
    if isinstance(seeds, list) and len(set(seeds)) != len(seeds):
        errors.append(
            f"{name}: performance_reference_protocol seeds contain duplicates"
        )
    repeatability_limit = protocol.get("repeatability_limit")
    if (
        isinstance(repeatability_limit, bool)
        or not isinstance(repeatability_limit, (int, float))
        or not 0 < float(repeatability_limit) < 1
    ):
        errors.append(
            f"{name}: performance_reference_protocol.repeatability_limit must be between 0 and 1"
        )
    primary_metric = (body.get("measurement_protocol") or {}).get("primary_metric")
    if not primary_metric:
        errors.append(f"{name}: measurement_protocol.primary_metric is missing")
    elif primary_metric not in str(protocol.get("aggregation") or ""):
        errors.append(
            f"{name}: performance_reference_protocol.aggregation does not name primary metric {primary_metric}"
        )
    if "reference_runs" in (body.get("functional_check") or {}):
        errors.append(
            f"{name}: functional_check.reference_runs is ambiguous; declare performance_reference_protocol.reference_runs"
        )
    return errors


def check_summary_aggregate_integrity(
    name: str, payload: dict, runs: list[dict]
) -> list[str]:
    """Recompute the summary statistics from the indexed per-seed values."""
    errors: list[str] = []
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        return [f"{name}: reference summary aggregate is missing or not an object"]

    dual_metrics = payload.get("schema") == REFERENCE_EVIDENCE_SCHEMA
    series = (
        (
            ("primary_metric", "primary_metric_value"),
            ("quality", "quality_value"),
            ("wall_seconds", "wall_seconds"),
        )
        if dual_metrics
        else (
            ("quality", "quality_value"),
            ("wall_seconds", "wall_seconds"),
        )
    )
    for series_name, run_field in series:
        declared = aggregate.get(series_name)
        if dual_metrics and series_name == "quality" and declared is None:
            if payload.get("public_status") != "performance-bearing":
                errors.append(
                    f"{name}: score-bearing reference summary aggregate.quality is missing"
                )
            continue
        values: list[float] = []
        for index, run in enumerate(runs):
            value = run.get(run_field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                errors.append(
                    f"{name}: reference summary run {index} has invalid {run_field}"
                )
            else:
                values.append(float(value))
        if len(values) != len(runs):
            continue
        if not isinstance(declared, dict):
            errors.append(
                f"{name}: reference summary aggregate.{series_name} is missing or not an object"
            )
            continue
        computed = recompute_aggregate(values)
        for field, expected in computed.items():
            if not numbers_match(declared.get(field), expected):
                errors.append(
                    f"{name}: reference summary aggregate.{series_name}.{field} "
                    f"is {declared.get(field)!r}, recomputed value is {expected!r}"
                )
    return errors


def check_summary_acceptance(
    name: str, body: dict, payload: dict, runs: list[dict]
) -> list[str]:
    """Bind the summary's acceptance claim to the registry and raw run values."""
    if payload.get("schema") == REFERENCE_EVIDENCE_SCHEMA:
        return check_dual_metric_summary_acceptance(name, body, payload, runs)
    errors: list[str] = []
    acceptance = payload.get("acceptance") or {}
    values = [run.get("quality_value") for run in runs]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return [f"{name}: reference summary acceptance has invalid run values"]
    median = statistics.median(float(value) for value in values)
    public_status = (body.get("public") or {}).get("status")
    if public_status == "score-bearing":
        target = body.get("quality_target") or {}
        direction = target.get("direction")
        target_value = target.get("value")
        tolerance = float(target.get("tolerance") or 0.0)
        expected_operator = "<=" if direction == "lower" else ">="
        expected = {
            "statistic": "median",
            "operator": expected_operator,
            "target": target_value,
            "value": median,
        }
        if payload.get("reference_metric_role") != "quality":
            errors.append(
                f"{name}: score summary reference_metric_role must be quality"
            )
        if payload.get("quality_metric") != target.get("metric"):
            errors.append(
                f"{name}: score summary quality_metric does not match registry"
            )
        if payload.get("quality_direction") != target.get("direction"):
            errors.append(
                f"{name}: score summary quality_direction does not match registry"
            )
        if not numbers_match(payload.get("quality_target"), target.get("value")):
            errors.append(
                f"{name}: score summary quality_target does not match registry"
            )
        if payload.get("repeatability") is not None:
            errors.append(f"{name}: score summary repeatability must be null")
        if payload.get("functional_gate") is not None:
            errors.append(f"{name}: score summary functional_gate must be null")
        if isinstance(target_value, (int, float)):
            for index, value in enumerate(values):
                numeric = float(value)
                passed = (
                    numeric - tolerance <= float(target_value)
                    if direction == "lower"
                    else numeric + tolerance >= float(target_value)
                )
                if not passed:
                    errors.append(
                        f"{name}: reference summary run {index} does not satisfy the registry quality target"
                    )
    elif public_status == "performance-bearing":
        expected = {
            "statistic": "all_runs",
            "operator": "==",
            "target": len(runs),
            "value": len(runs),
            "condition": (body.get("functional_check") or {}).get("condition"),
        }
        if payload.get("reference_metric_role") != "performance":
            errors.append(
                f"{name}: performance summary reference_metric_role must be performance"
            )
        if payload.get("quality_metric") != (
            body.get("measurement_protocol") or {}
        ).get("primary_metric"):
            errors.append(
                f"{name}: performance summary metric does not match measurement protocol"
            )
        if payload.get("quality_target") is not None:
            errors.append(
                f"{name}: performance summary quality_target must be null; use functional_gate"
            )
        basis = payload.get("basis") or {}
        if basis.get("quality_target") is not None:
            errors.append(
                f"{name}: performance summary basis.quality_target must be null"
            )
        functional = basis.get("functional_check") or {}
        registry_functional = body.get("functional_check") or {}
        if functional.get("metric") != registry_functional.get("metric"):
            errors.append(
                f"{name}: performance summary functional metric does not match registry"
            )
        if functional.get("condition") != registry_functional.get("condition"):
            errors.append(
                f"{name}: performance summary functional condition does not match registry"
            )
        if payload.get("functional_gate") != functional:
            errors.append(
                f"{name}: performance summary functional_gate does not match its basis"
            )
        grade_targets = {
            json.dumps((run.get("grade") or {}).get("target"), sort_keys=True)
            for run in runs
        }
        if len(grade_targets) != 1 or functional.get("target") != json.loads(
            next(iter(grade_targets), "null")
        ):
            errors.append(
                f"{name}: performance summary functional target does not match raw grades"
            )
        for index, value in enumerate(values):
            if float(value) <= 0:
                errors.append(
                    f"{name}: performance summary run {index} has non-positive primary metric"
                )
        repeatability = payload.get("repeatability") or {}
        aggregate = (payload.get("aggregate") or {}).get("primary_metric") or {}
        mean = aggregate.get("mean")
        stdev = aggregate.get("stdev")
        limit = (body.get("performance_reference_protocol") or {}).get(
            "repeatability_limit"
        )
        computed = (
            float(stdev) / float(mean)
            if isinstance(mean, (int, float))
            and not isinstance(mean, bool)
            and float(mean) > 0
            and isinstance(stdev, (int, float))
            and not isinstance(stdev, bool)
            else None
        )
        expected_repeatability = {
            "metric": (body.get("performance_reference_protocol") or {}).get(
                "repeatability_metric"
            ),
            "coefficient_of_variation": computed,
            "limit": limit,
            "passed": computed is not None
            and isinstance(limit, (int, float))
            and computed <= float(limit),
        }
        if repeatability != expected_repeatability:
            errors.append(
                f"{name}: performance summary repeatability does not match recomputed values"
            )
    else:
        return errors

    for field, expected_value in expected.items():
        actual = acceptance.get(field)
        if not numbers_match(actual, expected_value):
            errors.append(
                f"{name}: reference summary acceptance.{field} is {actual!r}, "
                f"expected {expected_value!r}"
            )
    return errors


def check_dual_metric_summary_acceptance(
    name: str, body: dict, payload: dict, runs: list[dict]
) -> list[str]:
    """Validate schema 0.4 performance-primary and separate gate semantics."""
    errors: list[str] = []
    acceptance = payload.get("acceptance") or {}
    public_status = (body.get("public") or {}).get("status")
    primary_name = (body.get("measurement_protocol") or {}).get("primary_metric")
    if payload.get("reference_metric_role") != "performance":
        errors.append(f"{name}: schema 0.4 reference_metric_role must be performance")
    if payload.get("primary_metric") != {
        "name": primary_name,
        "role": "performance",
    }:
        errors.append(
            f"{name}: schema 0.4 primary metric does not match measurement protocol"
        )

    primary_values = [run.get("primary_metric_value") for run in runs]
    for index, value in enumerate(primary_values):
        run = runs[index]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            errors.append(
                f"{name}: reference summary run {index} has invalid primary performance value"
            )
        if run.get("primary_metric_declared") != primary_name:
            errors.append(
                f"{name}: reference summary run {index} primary metric declaration mismatches registry"
            )

    if public_status == "score-bearing":
        target = body.get("quality_target") or {}
        quality_metric = target.get("metric")
        direction = target.get("direction")
        target_value = target.get("value")
        tolerance = float(target.get("tolerance") or 0.0)
        values = [run.get("quality_value") for run in runs]
        numeric_values: list[float] = []
        for index, value in enumerate(values):
            run = runs[index]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                errors.append(
                    f"{name}: reference summary run {index} has invalid quality value"
                )
                continue
            numeric = float(value)
            numeric_values.append(numeric)
            passed = (
                numeric - tolerance <= float(target_value)
                if direction == "lower" and isinstance(target_value, (int, float))
                else numeric + tolerance >= float(target_value)
                if direction == "higher" and isinstance(target_value, (int, float))
                else False
            )
            if not passed or run.get("quality_target_met") is not True:
                errors.append(
                    f"{name}: reference summary run {index} does not satisfy the registry quality target"
                )
            if run.get("quality_metric_declared") != quality_metric:
                errors.append(
                    f"{name}: reference summary run {index} quality metric declaration mismatches registry"
                )
            if any(
                run.get(field) is not None
                for field in (
                    "functional_metric_declared",
                    "functional_metric_key",
                    "functional_metric_value",
                )
            ):
                errors.append(
                    f"{name}: score-bearing run {index} must not declare a functional gate metric"
                )
        median = statistics.median(numeric_values) if numeric_values else None
        expected_operator = "<=" if direction == "lower" else ">="
        expected = {
            "passed": True,
            "statistic": "median",
            "operator": expected_operator,
            "target": target_value,
            "value": median,
            "all_runs_passed": True,
            "passed_runs": len(runs),
            "run_count": len(runs),
            "tolerance": tolerance,
        }
        expected_quality_gate = {
            "metric": quality_metric,
            "target": target_value,
            "direction": direction,
            "tolerance": target.get("tolerance"),
            "all_runs_must_pass": True,
        }
        if payload.get("quality_metric") != quality_metric:
            errors.append(
                f"{name}: score summary quality_metric does not match registry"
            )
        if payload.get("quality_gate") != expected_quality_gate:
            errors.append(f"{name}: score summary quality_gate does not match registry")
        if (payload.get("basis") or {}).get("quality_target") != expected_quality_gate:
            errors.append(
                f"{name}: score summary basis.quality_target does not match registry"
            )
        if payload.get("quality_target") != target_value:
            errors.append(
                f"{name}: score summary quality_target does not match registry"
            )
        if payload.get("quality_direction") != direction:
            errors.append(
                f"{name}: score summary quality_direction does not match registry"
            )
        if payload.get("functional_gate") is not None:
            errors.append(f"{name}: score summary functional_gate must be null")
        if payload.get("repeatability") is not None:
            errors.append(f"{name}: score summary repeatability must be null")
        for index, run in enumerate(runs):
            grade = run.get("grade") or {}
            expected_grade = {
                "passed": True,
                "target_met": True,
                "metric": quality_metric,
                "value": run.get("quality_value"),
                "target": target_value,
            }
            for field, expected_value in expected_grade.items():
                if not numbers_match(grade.get(field), expected_value):
                    errors.append(
                        f"{name}: score run {index} grade.{field} is not bound to quality"
                    )
    elif public_status == "performance-bearing":
        functional = body.get("functional_check") or {}
        values = [run.get("functional_metric_value") for run in runs]
        if payload.get("quality_metric") is not None:
            errors.append(f"{name}: performance summary quality_metric must be null")
        if payload.get("quality_gate") is not None:
            errors.append(f"{name}: performance summary quality_gate must be null")
        if (payload.get("aggregate") or {}).get("quality") is not None:
            errors.append(f"{name}: performance summary aggregate.quality must be null")
        if payload.get("functional_gate") != (payload.get("basis") or {}).get(
            "functional_check"
        ):
            errors.append(
                f"{name}: performance summary functional gate differs from its basis"
            )
        if (payload.get("basis") or {}).get("quality_target") is not None:
            errors.append(
                f"{name}: performance summary basis.quality_target must be null"
            )
        if (payload.get("functional_gate") or {}).get("metric") != functional.get(
            "metric"
        ):
            errors.append(
                f"{name}: performance summary functional metric does not match registry"
            )
        for index, value in enumerate(values):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or runs[index].get("quality_target_met") is not True
            ):
                errors.append(
                    f"{name}: performance summary run {index} has an invalid or failed functional gate"
                )
            if runs[index].get("functional_metric_declared") != functional.get(
                "metric"
            ):
                errors.append(
                    f"{name}: performance summary run {index} functional metric declaration mismatches registry"
                )
            if any(
                runs[index].get(field) is not None
                for field in (
                    "quality_metric_declared",
                    "quality_metric_key",
                    "quality_value",
                )
            ):
                errors.append(
                    f"{name}: performance run {index} must not declare score-bearing quality fields"
                )
            grade = runs[index].get("grade") or {}
            expected_grade = {
                "passed": True,
                "target_met": True,
                "metric": functional.get("metric"),
                "value": value,
                "target": (payload.get("functional_gate") or {}).get("target"),
            }
            for field, expected_value in expected_grade.items():
                if not numbers_match(grade.get(field), expected_value):
                    errors.append(
                        f"{name}: performance run {index} grade.{field} is not bound to functional gate"
                    )
        expected = {
            "passed": True,
            "statistic": "all_runs",
            "operator": "==",
            "target": len(runs),
            "value": len(runs),
            "condition": functional.get("condition"),
        }
        repeatability = payload.get("repeatability") or {}
        aggregate = (payload.get("aggregate") or {}).get("primary_metric") or {}
        mean = aggregate.get("mean")
        stdev = aggregate.get("stdev")
        limit = (body.get("performance_reference_protocol") or {}).get(
            "repeatability_limit"
        )
        computed = (
            float(stdev) / float(mean)
            if isinstance(mean, (int, float))
            and not isinstance(mean, bool)
            and float(mean) > 0
            and isinstance(stdev, (int, float))
            and not isinstance(stdev, bool)
            else None
        )
        expected_repeatability = {
            "metric": (body.get("performance_reference_protocol") or {}).get(
                "repeatability_metric"
            ),
            "coefficient_of_variation": computed,
            "limit": limit,
            "passed": computed is not None
            and isinstance(limit, (int, float))
            and computed <= float(limit),
        }
        if repeatability != expected_repeatability:
            errors.append(
                f"{name}: performance summary repeatability does not match recomputed values"
            )
    else:
        return [f"{name}: unsupported public status {public_status!r}"]

    for field, expected_value in expected.items():
        if not numbers_match(acceptance.get(field), expected_value):
            errors.append(
                f"{name}: reference summary acceptance.{field} is "
                f"{acceptance.get(field)!r}, expected {expected_value!r}"
            )
    return errors


def check_registry_summary_alignment(name: str, body: dict, payload: dict) -> list[str]:
    """Bind every displayed baseline field to its committed evidence summary."""
    errors: list[str] = []
    baseline = body.get("verified_baseline") or {}
    runs = payload.get("runs") or []
    dual_metrics = payload.get("schema") == REFERENCE_EVIDENCE_SCHEMA
    primary_metric_name = (
        (payload.get("primary_metric") or {}).get("name")
        if dual_metrics
        else payload.get("quality_metric")
    )
    primary_values = [
        run.get("primary_metric_value") if dual_metrics else run.get("quality_value")
        for run in runs
    ]
    primary_aggregate = (payload.get("aggregate") or {}).get("primary_metric") or {}
    quality_aggregate = (payload.get("aggregate") or {}).get("quality") or {}
    baseline_primary_aggregate = (
        primary_aggregate if dual_metrics else quality_aggregate
    )
    wall_aggregate = (payload.get("aggregate") or {}).get("wall_seconds") or {}
    source = payload.get("source") or {}

    expected_fields = {
        "evidence_id": payload.get("evidence_id"),
        "evidence_tier": payload.get("evidence_tier"),
        "source_git_sha": source.get("git_sha"),
        "profile": payload.get("profile"),
        "device_requested": payload.get("device_requested"),
        "primary_metric": primary_metric_name,
        "accepted_runs": len(runs),
    }
    for field, expected in expected_fields.items():
        if not numbers_match(baseline.get(field), expected):
            errors.append(
                f"{name}: verified_baseline.{field} is {baseline.get(field)!r}, "
                f"reference summary value is {expected!r}"
            )

    expected_lists = {
        "seeds": payload.get("seeds_requested"),
        "metric_values_by_seed": primary_values,
    }
    for field, expected in expected_lists.items():
        if not list_values_match(baseline.get(field), expected):
            errors.append(
                f"{name}: verified_baseline.{field} does not match the reference summary"
            )

    for run_field, baseline_field in (
        ("backend", "execution_backend"),
        ("chip", "hardware_chip"),
        ("data_mode", "data_mode"),
    ):
        observed = [run.get(run_field) for run in runs]
        if any(not isinstance(value, str) or not value for value in observed):
            errors.append(
                f"{name}: reference summary has a missing or invalid {run_field}"
            )
            continue
        unique = set(observed)
        if len(unique) != 1:
            errors.append(
                f"{name}: reference summary does not have one stable {run_field} across seeds"
            )
        else:
            expected = next(iter(unique))
            if baseline.get(baseline_field) != expected:
                errors.append(
                    f"{name}: verified_baseline.{baseline_field} is "
                    f"{baseline.get(baseline_field)!r}, reference summary value is {expected!r}"
                )

    metric = primary_metric_name
    if not isinstance(metric, str) or not metric:
        errors.append(f"{name}: reference summary quality_metric is missing")
    elif not numbers_match(
        baseline.get(metric), baseline_primary_aggregate.get("median")
    ):
        errors.append(
            f"{name}: verified_baseline.{metric} does not match the committed median"
        )

    aggregate_fields = {
        "median": baseline_primary_aggregate.get("median"),
        "min": baseline_primary_aggregate.get("min"),
        "max": baseline_primary_aggregate.get("max"),
        "mean": baseline_primary_aggregate.get("mean"),
        "sample_stdev": baseline_primary_aggregate.get("stdev"),
        "wall_seconds_median": wall_aggregate.get("median"),
        "wall_seconds_min": wall_aggregate.get("min"),
        "wall_seconds_max": wall_aggregate.get("max"),
        "wall_seconds_mean": wall_aggregate.get("mean"),
        "wall_seconds_sample_stdev": wall_aggregate.get("stdev"),
    }
    for field, expected in aggregate_fields.items():
        if not numbers_match(baseline.get(field), expected):
            errors.append(
                f"{name}: verified_baseline.{field} does not match the reference summary"
            )

    public_status = (body.get("public") or {}).get("status")
    if public_status == "score-bearing":
        expected_role = "performance" if dual_metrics else "quality"
        if payload.get("reference_metric_role") != expected_role:
            errors.append(
                f"{name}: score-bearing reference summary metric role must be {expected_role}"
            )
        quality_target = body.get("quality_target") or {}
        if payload.get("quality_metric") != quality_target.get("metric"):
            errors.append(
                f"{name}: reference summary metric does not match quality_target.metric"
            )
        if not numbers_match(
            payload.get("quality_target"), quality_target.get("value")
        ):
            errors.append(
                f"{name}: reference summary target does not match quality_target.value"
            )
        if payload.get("quality_direction") != quality_target.get("direction"):
            errors.append(
                f"{name}: reference summary direction does not match quality_target.direction"
            )
        quality_metric = quality_target.get("metric")
        if dual_metrics and not numbers_match(
            baseline.get(quality_metric), quality_aggregate.get("median")
        ):
            errors.append(
                f"{name}: verified_baseline.{quality_metric} does not match committed quality median"
            )
        variance = quality_target.get("variance_summary") or {}
        variance_fields = {
            "runs": len(runs),
            "median": quality_aggregate.get("median"),
            "min": quality_aggregate.get("min"),
            "max": quality_aggregate.get("max"),
            "mean": quality_aggregate.get("mean"),
            "sample_stdev": quality_aggregate.get("stdev"),
        }
        for field, expected in variance_fields.items():
            if not numbers_match(variance.get(field), expected):
                errors.append(
                    f"{name}: quality_target.variance_summary.{field} does not match committed evidence"
                )
    elif public_status == "performance-bearing":
        if payload.get("reference_metric_role") != "performance":
            errors.append(
                f"{name}: performance-bearing reference summary metric role must be performance"
            )
        primary_metric = (body.get("measurement_protocol") or {}).get("primary_metric")
        if dual_metrics:
            if (payload.get("primary_metric") or {}).get("name") != primary_metric:
                errors.append(
                    f"{name}: reference summary metric does not match measurement_protocol.primary_metric"
                )
            if payload.get("quality_metric") is not None:
                errors.append(
                    f"{name}: performance summary quality_metric must be null"
                )
        elif payload.get("quality_metric") != primary_metric:
            errors.append(
                f"{name}: reference summary metric does not match measurement_protocol.primary_metric"
            )
        if baseline.get("functional_passes") != len(runs):
            errors.append(
                f"{name}: verified_baseline.functional_passes must equal the accepted run count"
            )
        mean = primary_aggregate.get("mean")
        stdev = primary_aggregate.get("stdev")
        repeatability_limit = (body.get("performance_reference_protocol") or {}).get(
            "repeatability_limit"
        )
        if (
            isinstance(mean, (int, float))
            and not isinstance(mean, bool)
            and float(mean) > 0
            and isinstance(stdev, (int, float))
            and not isinstance(stdev, bool)
            and isinstance(repeatability_limit, (int, float))
            and not isinstance(repeatability_limit, bool)
        ):
            coefficient_of_variation = float(stdev) / float(mean)
            if coefficient_of_variation > float(repeatability_limit):
                errors.append(
                    f"{name}: reference performance coefficient of variation "
                    f"{coefficient_of_variation:.6f} exceeds repeatability limit "
                    f"{float(repeatability_limit):.6f}"
                )
        else:
            errors.append(
                f"{name}: reference performance repeatability cannot be computed"
            )
    return errors


def check_promoted_case_summary(
    name: str,
    body: dict,
    baseline: dict,
    payload: dict,
) -> list[str]:
    """Validate a committed schema-0.6 case against its registry baseline."""
    errors: list[str] = []
    expected = {
        "schema": PROMOTION_REFERENCE_EVIDENCE_SCHEMA,
        "workload": name.split("/", 1)[-1],
        "profile": baseline.get("profile"),
        "mode": baseline.get("mode"),
        "phase": baseline.get("phase"),
        "result_role": baseline.get("result_role"),
        "status": "valid",
        "evidence_tier": "promotion-candidate",
        "eligible_for_promotion": True,
        "eligible_for_public_baseline": False,
        "evidence_id": baseline.get("evidence_id"),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            errors.append(
                f"{name}: promoted summary {field} is {payload.get(field)!r}, "
                f"expected {value!r}"
            )
    if payload.get("invalid_reasons") != []:
        errors.append(f"{name}: promoted summary invalid_reasons is not empty")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 5:
        errors.append(f"{name}: promoted summary must contain five runs")
    elif any(
        not isinstance(run, dict)
        or run.get("evidence_valid") is not True
        or (run.get("promotion_contract") or {}).get("status") != "passed"
        for run in runs
    ):
        errors.append(f"{name}: one or more promoted summary runs did not pass")
    primary = (payload.get("aggregate") or {}).get("primary_metric") or {}
    for field, baseline_field in (
        ("median", "median"),
        ("min", "min"),
        ("max", "max"),
        ("mean", "mean"),
        ("stdev", "sample_stdev"),
    ):
        if not numbers_match(primary.get(field), baseline.get(baseline_field)):
            errors.append(
                f"{name}: promoted primary aggregate {field} does not match baseline"
            )
    repeatability = payload.get("primary_metric_repeatability") or {}
    if not numbers_match(
        repeatability.get("coefficient_of_variation"),
        baseline.get("coefficient_of_variation"),
    ):
        errors.append(f"{name}: promoted timing CV does not match baseline")
    if repeatability.get("passed") is not True:
        errors.append(f"{name}: promoted timing repeatability did not pass")
    source = payload.get("source") or {}
    if (
        source.get("git_sha") != baseline.get("source_git_sha")
        or source.get("git_dirty") is not False
        or source.get("git_status_sha256") != EMPTY_SHA256
        or source.get("git_patch_sha256") != EMPTY_SHA256
    ):
        errors.append(
            f"{name}: promoted summary source is not the clean baseline source"
        )
    if baseline.get("result_role") == "score-bearing":
        quality = (payload.get("aggregate") or {}).get("quality") or {}
        if payload.get("quality_metric") != baseline.get("quality_metric"):
            errors.append(f"{name}: promoted quality metric does not match baseline")
        if not numbers_match(quality.get("median"), baseline.get("quality_median")):
            errors.append(f"{name}: promoted quality median does not match baseline")
        if payload.get("functional_gate") is not None:
            errors.append(
                f"{name}: score-bearing promoted summary has a functional gate"
            )
    else:
        if payload.get("quality_gate") is not None:
            errors.append(f"{name}: performance promoted summary has a quality gate")
        if not numbers_match(
            payload.get("functional_gate"), baseline.get("functional_gate")
        ):
            errors.append(f"{name}: promoted functional gate does not match baseline")
    canonical = body.get("verified_baseline") or {}
    case_map = body.get("verified_baselines") or {}
    if baseline is canonical and case_map.get(baseline.get("case_id")) != baseline:
        errors.append(f"{name}: canonical baseline is absent from verified_baselines")
    return errors


def check_reference_summary(name: str, body: dict, payload: dict) -> list[str]:
    """Validate a lightweight index for a clean, externally retained sweep package."""
    errors: list[str] = []
    workload_id = name.split("/", 1)[-1]
    quality_target = body.get("quality_target") or {}
    performance_protocol = body.get("performance_reference_protocol") or {}
    reference_protocol = (
        performance_protocol or quality_target.get("reference_protocol") or {}
    )
    expected_seeds = reference_protocol.get("seeds")
    expected_run_count = performance_protocol.get(
        "reference_runs"
    ) or quality_target.get("reference_runs")

    expected_values = {
        "workload": workload_id,
        "profile": reference_protocol.get("profile"),
        "status": "valid",
        "eligible_for_public_baseline": True,
        "evidence_tier": "public-candidate",
        "public_status": (body.get("public") or {}).get("status"),
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            errors.append(
                f"{name}: reference summary {field} is {payload.get(field)!r}, expected {expected!r}"
            )
    schema = payload.get("schema")
    if schema not in SUPPORTED_REFERENCE_EVIDENCE_SCHEMAS:
        errors.append(
            f"{name}: reference summary schema is {schema!r}, expected one of "
            f"{sorted(SUPPORTED_REFERENCE_EVIDENCE_SCHEMAS)!r}"
        )
    dual_metrics = schema == REFERENCE_EVIDENCE_SCHEMA
    if dual_metrics:
        expected_primary_metric = {
            "name": (body.get("measurement_protocol") or {}).get("primary_metric"),
            "role": "performance",
        }
    else:
        expected_metric_role = (
            "performance"
            if (body.get("public") or {}).get("status") == "performance-bearing"
            else "quality"
        )
        expected_primary_metric = {
            "name": payload.get("quality_metric"),
            "role": expected_metric_role,
        }
    if payload.get("primary_metric") != expected_primary_metric:
        errors.append(
            f"{name}: reference summary primary_metric does not match its metric role"
        )
    aggregate = payload.get("aggregate") or {}
    if not dual_metrics and aggregate.get("primary_metric") != aggregate.get("quality"):
        errors.append(
            f"{name}: reference summary aggregate.primary_metric does not match the legacy aggregate.quality mirror"
        )
    if payload.get("invalid_reasons"):
        errors.append(f"{name}: reference summary contains invalid_reasons")
    if (payload.get("acceptance") or {}).get("passed") is not True:
        errors.append(f"{name}: reference summary acceptance did not pass")
    source = payload.get("source") or {}
    if source.get("git_dirty") is not False:
        errors.append(
            f"{name}: reference summary was not produced from a clean Git worktree"
        )
    for field in ("git_status_sha256", "git_patch_sha256"):
        if source.get(field) != EMPTY_SHA256:
            errors.append(
                f"{name}: reference summary source.{field} does not prove an empty clean-source record"
            )
    if dual_metrics:
        if source.get("tool_sha256") != SWEEP_TOOL_SHA256:
            errors.append(
                f"{name}: reference summary source.tool_sha256 does not match the sweep tool"
            )
    elif not PREFIXED_SHA256_RE.fullmatch(str(source.get("tool_sha256") or "")):
        errors.append(f"{name}: legacy reference summary source.tool_sha256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_sha") or "")):
        errors.append(f"{name}: reference summary source.git_sha is missing or invalid")
    evidence_id = payload.get("evidence_id")
    if not isinstance(evidence_id, str) or not EVIDENCE_ID_RE.fullmatch(evidence_id):
        errors.append(f"{name}: reference summary evidence_id is not portable")

    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        return [*errors, f"{name}: reference summary has no run index"]
    observed_seeds = [
        run.get("requested_seed") for run in runs if isinstance(run, dict)
    ]
    if isinstance(expected_run_count, int) and len(runs) != expected_run_count:
        errors.append(
            f"{name}: reference summary contains {len(runs)} runs, expected {expected_run_count}"
        )
    if isinstance(expected_seeds, list) and observed_seeds != expected_seeds:
        errors.append(
            f"{name}: reference summary seeds {observed_seeds} do not match registry protocol {expected_seeds}"
        )
    if payload.get("seeds_requested") != observed_seeds:
        errors.append(
            f"{name}: reference summary seeds_requested does not match its run index"
        )
    basis_seeds = ((payload.get("basis") or {}).get("reference_protocol") or {}).get(
        "seeds"
    )
    if basis_seeds != observed_seeds:
        errors.append(
            f"{name}: reference summary basis.reference_protocol.seeds does not match its run index"
        )

    for index, run in enumerate(runs):
        label = f"{name}: reference summary run {index}"
        if not isinstance(run, dict):
            errors.append(f"{label} is not an object")
            continue
        for field in (
            "execution_ok",
            "evidence_valid",
            "seed_match",
            "manifest_verified",
        ):
            if run.get(field) is not True:
                errors.append(f"{label} does not set {field} to true")
        if run.get("quality_target_met") is not True:
            errors.append(f"{label} did not pass its quality or functional target")
        if run.get("timed_out") is True:
            errors.append(f"{label} timed out")
        if run.get("invalid_reasons"):
            errors.append(f"{label} contains invalid_reasons")
        if (run.get("grade") or {}).get("passed") is not True:
            errors.append(f"{label} did not pass grading")

        artifacts = run.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{label} has no artifact index")
            continue
        roles: set[str] = set()
        paths: set[str] = set()
        for artifact_index, artifact in enumerate(artifacts):
            artifact_label = f"{label} artifact {artifact_index}"
            if not isinstance(artifact, dict):
                errors.append(f"{artifact_label} is not an object")
                continue
            role = str(artifact.get("role") or "")
            path = str(artifact.get("path") or "")
            digest = str(artifact.get("sha256") or "")
            n_bytes = artifact.get("n_bytes")
            if not role:
                errors.append(f"{artifact_label} has no role")
            if not is_safe_posix_relative_path(path):
                errors.append(
                    f"{artifact_label} path is missing, absolute, or escapes its package"
                )
            if not PREFIXED_SHA256_RE.fullmatch(digest):
                errors.append(
                    f"{artifact_label} does not contain a full SHA-256 digest"
                )
            if not isinstance(n_bytes, int) or n_bytes <= 0:
                errors.append(f"{artifact_label} n_bytes must be a positive integer")
            if role in roles:
                errors.append(f"{artifact_label} duplicates role {role!r}")
            if path in paths:
                errors.append(f"{artifact_label} duplicates path {path!r}")
            roles.add(role)
            paths.add(path)
        missing_roles = {"report", "provenance"}.difference(roles)
        if missing_roles:
            errors.append(
                f"{label} artifact index is missing roles {sorted(missing_roles)}"
            )
        if run.get("report_path") not in paths:
            errors.append(f"{label} report_path is not present in the artifact index")
        if run.get("manifest_path") not in paths:
            errors.append(f"{label} manifest_path is not present in the artifact index")
    dict_runs = [run for run in runs if isinstance(run, dict)]
    if len(dict_runs) == len(runs):
        errors.extend(check_summary_aggregate_integrity(name, payload, dict_runs))
        errors.extend(check_summary_acceptance(name, body, payload, dict_runs))
    baseline = body.get("verified_baseline") or {}
    if baseline.get("evidence_status") == "committed-reference-summary" and len(
        dict_runs
    ) == len(runs):
        errors.extend(check_registry_summary_alignment(name, body, payload))
    return errors


def check_historical_reference_summary(name: str, payload: dict) -> list[str]:
    """Check pinned historical bytes internally without applying the new protocol."""
    errors: list[str] = []
    if payload.get("schema") not in SUPPORTED_REFERENCE_EVIDENCE_SCHEMAS:
        errors.append(f"{name}: historical reference summary schema is unsupported")
    if payload.get("status") != "valid":
        errors.append(f"{name}: historical reference summary status is not valid")
    if payload.get("workload") != name.split("/", 1)[-1]:
        errors.append(f"{name}: historical reference summary workload mismatches")
    if (payload.get("acceptance") or {}).get("passed") is not True:
        errors.append(f"{name}: historical reference summary acceptance did not pass")
    if payload.get("invalid_reasons"):
        errors.append(f"{name}: historical reference summary contains invalid reasons")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        return [*errors, f"{name}: historical reference summary has no runs"]
    dict_runs: list[dict] = []
    for index, run in enumerate(runs):
        label = f"{name}: historical run {index}"
        if not isinstance(run, dict):
            errors.append(f"{label} is not an object")
            continue
        dict_runs.append(run)
        for field in (
            "execution_ok",
            "evidence_valid",
            "seed_match",
            "manifest_verified",
        ):
            if run.get(field) is not True:
                errors.append(f"{label} does not set {field}=true")
        if run.get("invalid_reasons"):
            errors.append(f"{label} contains invalid reasons")
        artifacts = run.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{label} has no artifact index")
            continue
        roles: set[str] = set()
        paths: set[str] = set()
        for artifact_index, artifact in enumerate(artifacts):
            artifact_label = f"{label} artifact {artifact_index}"
            if not isinstance(artifact, dict):
                errors.append(f"{artifact_label} is not an object")
                continue
            role = artifact.get("role")
            path = artifact.get("path")
            if not isinstance(role, str) or not role or role in roles:
                errors.append(f"{artifact_label} has a missing or duplicate role")
            if not is_safe_posix_relative_path(path) or path in paths:
                errors.append(f"{artifact_label} has an unsafe or duplicate path")
            if not PREFIXED_SHA256_RE.fullmatch(str(artifact.get("sha256") or "")):
                errors.append(f"{artifact_label} has an invalid SHA-256")
            if not isinstance(artifact.get("n_bytes"), int) or artifact["n_bytes"] <= 0:
                errors.append(f"{artifact_label} has an invalid byte count")
            roles.add(str(role))
            paths.add(str(path))
        if not {"report", "provenance"}.issubset(roles):
            errors.append(f"{label} lacks report or provenance artifacts")
        if run.get("report_path") not in paths:
            errors.append(f"{label} report path is absent from its artifact index")
        if run.get("manifest_path") not in paths:
            errors.append(f"{label} manifest path is absent from its artifact index")
    if len(dict_runs) == len(runs):
        errors.extend(check_summary_aggregate_integrity(name, payload, dict_runs))
    return errors


def _case_id(workload: str, mode: object, phase: object = None) -> str:
    parts = [workload, "max", str(mode)]
    if phase:
        parts.append(str(phase))
    return "__".join(parts)


CAUSAL_TRAINING_CASE_ID = "causal-language-modeling__max__training"
CAUSAL_INFERENCE_CASE_IDS = tuple(
    f"causal-language-modeling__max__inference__{phase}"
    for phase in ("full", "prefill", "decode")
)
SOURCE_TRAINING_FIELDS = {
    "source_training_case_id",
    "source_training_evidence_id",
    "source_training_evidence_sha256",
    "source_training_execution_index",
    "source_training_checkpoint_sha256",
    "source_training_report_sha256",
    "source_training_provenance_sha256",
    "source_training_package_sha256",
}


def _run_artifact_digest(run: dict, role: str) -> object:
    digests = {
        artifact.get("sha256")
        for artifact in run.get("artifacts") or []
        if isinstance(artifact, dict) and artifact.get("role") == role
    }
    return next(iter(digests)) if len(digests) == 1 else None


def check_case_source_training_lineage(
    indexed: dict[str, dict], payloads: dict[str, dict]
) -> list[str]:
    """Recheck the committed causal checkpoint bindings after evidence import."""
    errors: list[str] = []
    for identifier, entry in indexed.items():
        if (
            identifier not in CAUSAL_INFERENCE_CASE_IDS
            and entry.get("source_training") is not None
        ):
            errors.append(f"{identifier}: unexpected source_training binding")

    present_phases = [
        identifier for identifier in CAUSAL_INFERENCE_CASE_IDS if identifier in indexed
    ]
    if not present_phases:
        return errors
    training_entry = indexed.get(CAUSAL_TRAINING_CASE_ID)
    training_payload = payloads.get(CAUSAL_TRAINING_CASE_ID)
    if not isinstance(training_entry, dict) or not isinstance(training_payload, dict):
        return [*errors, "causal inference cases lack their indexed training case"]

    bindings: dict[str, dict] = {}
    for identifier in present_phases:
        entry = indexed[identifier]
        binding = entry.get("source_training")
        if not isinstance(binding, dict):
            errors.append(f"{identifier}: source_training binding is missing")
            continue
        bindings[identifier] = binding
        if set(binding) != SOURCE_TRAINING_FIELDS:
            errors.append(
                f"{identifier}: source_training fields do not match the contract"
            )
        expected_identity = {
            "source_training_case_id": CAUSAL_TRAINING_CASE_ID,
            "source_training_evidence_id": training_entry.get("evidence_id"),
            "source_training_evidence_sha256": training_entry.get("evidence_sha256"),
        }
        for field, expected in expected_identity.items():
            if binding.get(field) != expected:
                errors.append(
                    f"{identifier}: source_training {field} does not match training evidence"
                )
        execution_index = binding.get("source_training_execution_index")
        if (
            isinstance(execution_index, bool)
            or not isinstance(execution_index, int)
            or execution_index < 1
        ):
            errors.append(
                f"{identifier}: source_training execution index is not a positive integer"
            )
        for field in SOURCE_TRAINING_FIELDS - {
            "source_training_case_id",
            "source_training_evidence_id",
            "source_training_execution_index",
        }:
            if not PREFIXED_SHA256_RE.fullmatch(str(binding.get(field) or "")):
                errors.append(
                    f"{identifier}: source_training {field} is not a full SHA-256 digest"
                )

        payload = payloads.get(identifier) or {}
        lineage = payload.get("nanogpt_training_lineage") or {}
        if lineage.get("package_sha256") != binding.get(
            "source_training_package_sha256"
        ):
            errors.append(
                f"{identifier}: staged package digest does not match source_training"
            )
        phase_role_fields = {
            "checkpoint": "source_training_checkpoint_sha256",
            "source_training_report": "source_training_report_sha256",
            "source_training_provenance": "source_training_provenance_sha256",
        }
        for run_index, run in enumerate(payload.get("runs") or [], start=1):
            if not isinstance(run, dict):
                continue
            for role, field in phase_role_fields.items():
                if _run_artifact_digest(run, role) != binding.get(field):
                    errors.append(
                        f"{identifier}: run {run_index} {role} digest does not match source_training"
                    )

    if len(bindings) > 1:
        first = next(iter(bindings.values()))
        if any(binding != first for binding in bindings.values()):
            errors.append(
                "causal inference phases do not share one source_training binding"
            )

    if not bindings:
        return errors
    binding = next(iter(bindings.values()))
    selected = [
        run
        for run in training_payload.get("runs") or []
        if isinstance(run, dict)
        and run.get("execution_index") == binding.get("source_training_execution_index")
    ]
    if len(selected) != 1:
        errors.append("causal source_training does not select exactly one training run")
        return errors
    selected_run = selected[0]
    training_role_fields = {
        "checkpoint": "source_training_checkpoint_sha256",
        "report": "source_training_report_sha256",
        "provenance": "source_training_provenance_sha256",
    }
    for role, field in training_role_fields.items():
        if _run_artifact_digest(selected_run, role) != binding.get(field):
            errors.append(
                f"causal source_training {role} digest does not match the selected training run"
            )
    quality_median = (
        (training_payload.get("aggregate") or {}).get("quality") or {}
    ).get("median")
    if not numbers_match(selected_run.get("quality_value"), quality_median):
        errors.append("causal source_training does not select the median-quality run")
    return errors


def check_case_reference_index(
    index: dict,
    workloads: dict[str, dict],
) -> list[str]:
    """Validate schema-0.3 case closure and registry bindings."""
    errors: list[str] = []
    source_git_sha = index.get("source_git_sha")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_git_sha or "")):
        errors.append("case reference index source_git_sha is missing or invalid")
    source_lock = index.get("source_lock") or {}
    lock_relative = source_lock.get("path")
    if not is_safe_posix_relative_path(lock_relative):
        errors.append("case reference index source_lock.path is not portable")
    else:
        lock_path = (REPO_ROOT / str(lock_relative)).resolve()
        try:
            lock_path.relative_to(REPO_ROOT.resolve())
            lock_bytes = lock_path.read_bytes()
        except (ValueError, OSError) as exc:
            errors.append(f"case reference source lock cannot be read: {exc}")
        else:
            if source_lock.get("sha256") != reference_source_lock.sha256_bytes(
                lock_bytes
            ):
                errors.append("case reference source-lock digest does not match")
            try:
                lock = reference_source_lock.load_source_lock(
                    lock_path,
                    project_root=REPO_ROOT,
                    expected_source_git_sha=str(source_git_sha),
                    verify_current=True,
                )
            except reference_source_lock.SourceLockError as exc:
                errors.append(f"case reference source lock is invalid: {exc}")
            else:
                for field in ("schema", "file_count", "contract_count"):
                    if source_lock.get(field) != lock.get(field):
                        errors.append(
                            f"case reference source_lock.{field} does not match lock"
                        )

    expected_cases: set[str] = set()
    expected_by_workload: dict[str, set[str]] = {}
    for workload_id, body in workloads.items():
        if (body.get("public") or {}).get("status") not in {
            "score-bearing",
            "performance-bearing",
        }:
            continue
        canonical = body.get("canonical_max_contract") or {}
        identifiers = {_case_id(workload_id, canonical.get("mode"))}
        phases = ((body.get("mode_contracts") or {}).get("inference") or {}).get(
            "phases", {}
        )
        identifiers.update(
            _case_id(workload_id, "inference", phase) for phase in phases
        )
        expected_by_workload[workload_id] = identifiers
        expected_cases.update(identifiers)

    entries = index.get("cases")
    if not isinstance(entries, list):
        return [*errors, "case reference index cases must be a list"]
    if index.get("case_count") != len(entries):
        errors.append("case reference index case_count does not match cases")
    if index.get("workload_count") != len(expected_by_workload):
        errors.append("case reference index workload_count does not match registry")
    indexed: dict[str, dict] = {}
    payloads: dict[str, dict] = {}
    for position, entry in enumerate(entries):
        label = f"case reference index cases[{position}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} is not an object")
            continue
        identifier = entry.get("case_id")
        if not isinstance(identifier, str) or identifier in indexed:
            errors.append(f"{label} has a missing or duplicate case_id")
            continue
        indexed[identifier] = entry
        relative_path = entry.get("path")
        if not is_safe_posix_relative_path(relative_path) or not str(
            relative_path
        ).startswith("reference_results/"):
            errors.append(f"{label}.path is unsafe")
            continue
        path = (REPO_ROOT / str(relative_path)).resolve()
        try:
            path.relative_to(REPO_ROOT.resolve())
            data = path.read_bytes()
            payload = json.loads(data)
        except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{label} cannot load its summary: {exc}")
            continue
        if entry.get("evidence_sha256") != "sha256:" + hashlib.sha256(data).hexdigest():
            errors.append(f"{label} evidence_sha256 does not match summary bytes")
        expected_identity = {
            "evidence_id": payload.get("evidence_id"),
            "workload": payload.get("workload"),
            "profile": payload.get("profile"),
            "mode": payload.get("mode"),
            "phase": payload.get("phase"),
            "result_role": payload.get("result_role"),
        }
        for field, expected in expected_identity.items():
            if entry.get(field) != expected:
                errors.append(f"{label}.{field} does not match summary")
        if payload.get("schema") != PROMOTION_REFERENCE_EVIDENCE_SCHEMA:
            errors.append(f"{label} does not cite schema-0.7 promotion evidence")
        if (payload.get("source") or {}).get("git_sha") != source_git_sha:
            errors.append(f"{label} source commit does not match index")
        payloads[identifier] = payload
        workload_id = entry.get("workload")
        body = workloads.get(str(workload_id)) or {}
        baseline = (body.get("verified_baselines") or {}).get(identifier)
        if not isinstance(baseline, dict):
            errors.append(f"{label} has no registry verified_baselines binding")
            continue
        expected_baseline = {
            "case_id": identifier,
            "evidence_id": entry.get("evidence_id"),
            "evidence_file": relative_path,
            "evidence_sha256": entry.get("evidence_sha256"),
            "source_git_sha": source_git_sha,
            "mode": entry.get("mode"),
            "phase": entry.get("phase"),
            "result_role": entry.get("result_role"),
        }
        for field, expected in expected_baseline.items():
            if baseline.get(field) != expected:
                errors.append(f"{label} registry baseline {field} does not match index")
        errors.extend(check_promoted_case_summary(label, body, baseline, payload))
        source_training = entry.get("source_training")
        if (
            source_training is not None
            and baseline.get("source_training") != source_training
        ):
            errors.append(f"{label} source_training does not match baseline")

    if set(indexed) != expected_cases:
        errors.append(
            "case reference index closure mismatch; "
            f"missing={sorted(expected_cases - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected_cases)}"
        )
    errors.extend(check_case_source_training_lineage(indexed, payloads))
    for workload_id, identifiers in expected_by_workload.items():
        body = workloads[workload_id]
        baselines = body.get("verified_baselines") or {}
        if set(baselines) != identifiers:
            errors.append(
                f"{workload_id}: verified_baselines closure mismatch; "
                f"missing={sorted(identifiers - set(baselines))}, "
                f"extra={sorted(set(baselines) - identifiers)}"
            )
        canonical = body.get("verified_baseline") or {}
        canonical_id = canonical.get("case_id")
        if canonical_id not in identifiers or baselines.get(canonical_id) != canonical:
            errors.append(
                f"{workload_id}: verified_baseline is not the canonical case binding"
            )
    return errors


def check_reference_index(workloads: dict[str, dict]) -> list[str]:
    """Validate index closure, source lock, and registry-to-summary bindings."""
    errors: list[str] = []
    index_path = REPO_ROOT / "reference_results" / "index.json"
    public_bodies = [
        body
        for body in workloads.values()
        if (body.get("public") or {}).get("status")
        in {"score-bearing", "performance-bearing"}
    ]
    if not public_bodies:
        if index_path.exists():
            return [
                "reference index must be absent when no score-bearing or "
                "performance-bearing workloads exist"
            ]
        return []
    try:
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"reference index cannot be read: {exc}"]
    if not isinstance(index, dict):
        return ["reference index root must be an object"]
    if index.get("schema") == CASE_REFERENCE_INDEX_SCHEMA:
        return check_case_reference_index(index, workloads)
    if index.get("schema") != REFERENCE_INDEX_SCHEMA:
        errors.append(
            f"reference index schema is {index.get('schema')!r}, "
            f"expected {REFERENCE_INDEX_SCHEMA!r}"
        )
    source_git_sha = index.get("source_git_sha")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_git_sha or "")):
        errors.append("reference index source_git_sha is missing or invalid")

    source_lock = index.get("source_lock") or {}
    historical_index = bool(public_bodies) and all(
        (
            (body.get("verified_baseline") or {}).get("protocol_compatibility"),
            (body.get("verified_baseline") or {}).get("review_eligible"),
            (body.get("verified_baseline") or {}).get("replacement_required"),
        )
        == ("superseded", False, True)
        for body in public_bodies
    )
    lock_relative = source_lock.get("path")
    if not is_safe_posix_relative_path(lock_relative):
        errors.append("reference index source_lock.path is not portable")
    else:
        lock_path = (REPO_ROOT / str(lock_relative)).resolve()
        try:
            lock_path.relative_to(REPO_ROOT.resolve())
            lock_bytes = lock_path.read_bytes()
        except (ValueError, OSError) as exc:
            errors.append(f"reference source lock cannot be read: {exc}")
        else:
            lock_digest = reference_source_lock.sha256_bytes(lock_bytes)
            if source_lock.get("sha256") != lock_digest:
                errors.append("reference index source-lock digest does not match")
            try:
                lock_payload = reference_source_lock.load_source_lock(
                    lock_path,
                    project_root=REPO_ROOT,
                    expected_source_git_sha=str(source_git_sha),
                    verify_current=not historical_index,
                )
            except reference_source_lock.SourceLockError as exc:
                errors.append(f"reference source lock is invalid: {exc}")
            else:
                expected_lock_fields = {
                    "schema": lock_payload.get("schema"),
                    "file_count": lock_payload.get("file_count"),
                    "contract_count": lock_payload.get("contract_count"),
                }
                for field, expected in expected_lock_fields.items():
                    if source_lock.get(field) != expected:
                        errors.append(
                            f"reference index source_lock.{field} does not match the lock"
                        )

    entries = index.get("summaries")
    if not isinstance(entries, list):
        return [*errors, "reference index summaries must be a list"]
    if index.get("summary_count") != len(entries):
        errors.append("reference index summary_count does not match its entries")
    expected_workloads = {
        workload_id
        for workload_id, body in workloads.items()
        if (body.get("public") or {}).get("status")
        in {"score-bearing", "performance-bearing"}
    }
    indexed_workloads: set[str] = set()
    indexed_paths: set[str] = set()
    for position, entry in enumerate(entries):
        label = f"reference index summaries[{position}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} is not an object")
            continue
        workload_id = entry.get("workload")
        relative_path = entry.get("path")
        if not isinstance(workload_id, str) or workload_id in indexed_workloads:
            errors.append(f"{label} has a missing or duplicate workload")
            continue
        indexed_workloads.add(workload_id)
        if (
            not is_safe_posix_relative_path(relative_path)
            or not str(relative_path).startswith("reference_results/")
            or relative_path in indexed_paths
        ):
            errors.append(f"{label}.path is unsafe or duplicated")
            continue
        indexed_paths.add(str(relative_path))
        path = (REPO_ROOT / str(relative_path)).resolve()
        try:
            path.relative_to(REPO_ROOT.resolve())
            data = path.read_bytes()
            payload = json.loads(data)
        except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{label} cannot load its summary: {exc}")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if entry.get("evidence_sha256") != digest:
            errors.append(f"{label} evidence_sha256 does not match summary bytes")
        if payload.get("evidence_id") != entry.get("evidence_id"):
            errors.append(f"{label} evidence_id does not match its summary")
        if payload.get("workload") != workload_id:
            errors.append(f"{label} workload does not match its summary")
        body = workloads.get(workload_id) or {}
        baseline = body.get("verified_baseline") or {}
        expected_baseline = {
            "evidence_id": entry.get("evidence_id"),
            "evidence_file": relative_path,
            "evidence_sha256": entry.get("evidence_sha256"),
            "source_git_sha": source_git_sha,
        }
        for field, expected in expected_baseline.items():
            if baseline.get(field) != expected:
                errors.append(
                    f"{workload_id}: verified_baseline.{field} does not match the reference index"
                )
    if indexed_workloads != expected_workloads:
        errors.append(
            "reference index workload closure mismatch; "
            f"missing={sorted(expected_workloads - indexed_workloads)}, "
            f"extra={sorted(indexed_workloads - expected_workloads)}"
        )
    return errors


def check_declared_file(
    name: str,
    *,
    label: str,
    relative_path: object,
    digest: object,
) -> list[str]:
    errors: list[str] = []
    if not relative_path:
        return [f"{name}: {label} path is missing"]
    if not digest:
        return [f"{name}: {label} requires a full SHA-256 digest"]
    normalized_digest = str(digest).lower()
    if PREFIXED_SHA256_RE.fullmatch(normalized_digest):
        normalized_digest = normalized_digest.removeprefix("sha256:")
    elif not SHA256_RE.fullmatch(normalized_digest):
        return [
            f"{name}: {label} SHA-256 must be 64 lowercase hexadecimal characters, "
            "optionally prefixed with sha256:"
        ]
    path = (REPO_ROOT / str(relative_path)).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return [f"{name}: {label} escapes the project root: {relative_path}"]
    if not path.is_file():
        return [f"{name}: {label} does not exist: {relative_path}"]
    actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_digest != normalized_digest:
        errors.append(
            f"{name}: {label} SHA-256 mismatch for {relative_path} "
            f"(declared {normalized_digest}, actual {actual_digest})"
        )
    return errors


def latest_sidecar_for(workload: str, sidecar_dir: Path) -> dict | None:
    """Return the most recent roofline sidecar for `workload`, or None."""
    if not sidecar_dir.exists():
        return None
    cands = sorted(
        sidecar_dir.glob(f"{workload}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        return None
    try:
        return json.loads(cands[0].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def verify_against_sidecar(name: str, regime: dict, sidecar: dict) -> list[str]:
    """Cross-check YAML regime claims against measured sidecar."""
    errs: list[str] = []
    measured = sidecar.get("regime_inference", {})
    for axis_yaml, axis_sidecar in [
        ("arithmetic_intensity", "axis_arithmetic_intensity"),
        ("dispatch", "axis_dispatch"),
    ]:
        yaml_value = regime.get(axis_yaml, {}).get("value")
        sidecar_value = measured.get(axis_sidecar)
        if yaml_value in (None, "unmeasured"):
            continue
        if sidecar_value in (None, "unmeasured"):
            continue
        if yaml_value != sidecar_value:
            errs.append(
                f"{name}.{axis_yaml}: YAML claims '{yaml_value}' but sidecar "
                f"measured '{sidecar_value}' "
                f"(rule: {measured.get('rule', 'no rule')})"
            )
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint MLPerf EDU workload taxonomy.")
    parser.add_argument(
        "--verify-against-sidecars",
        type=str,
        default=None,
        metavar="DIR",
        help="Cross-check YAML regime claims against the latest "
        "roofline sidecar in DIR for each workload.",
    )
    args = parser.parse_args()
    sidecar_dir = (
        Path(args.verify_against_sidecars) if args.verify_against_sidecars else None
    )

    if not WORKLOADS_YAML.exists():
        print(f"FAIL: {WORKLOADS_YAML} not found")
        return 2

    doc = yaml.safe_load(WORKLOADS_YAML.read_text())
    suites = doc.get("suites", {})
    if not suites:
        print("FAIL: no suites found")
        return 2

    all_errors: list[str] = []
    workload_bodies: dict[str, dict] = {}
    cell_counts: dict[tuple, list[str]] = {}
    unmeasured_axes: dict[str, list[str]] = {
        "working_set": [],
        "arithmetic_intensity": [],
        "dispatch": [],
    }
    n_workloads = 0

    for div, workloads in suites.items():
        for name, body in workloads.items():
            n_workloads += 1
            workload_bodies[name] = body
            full_name = f"{div}/{name}"
            if "regime" not in body:
                all_errors.append(f"{full_name}: no regime block")
                continue
            errs, values = check_regime(full_name, body["regime"])
            all_errors.extend(errs)
            all_errors.extend(check_workload_evidence(full_name, body))
            for axis, v in values.items():
                if v == "unmeasured":
                    unmeasured_axes[axis].append(full_name)
            if sidecar_dir is not None:
                sidecar = latest_sidecar_for(name, sidecar_dir)
                if sidecar is not None:
                    all_errors.extend(
                        verify_against_sidecar(full_name, body["regime"], sidecar)
                    )
            cell = (
                values.get("working_set"),
                values.get("arithmetic_intensity"),
                values.get("dispatch"),
            )
            cell_counts.setdefault(cell, []).append(full_name)

    all_errors.extend(check_reference_index(workload_bodies))

    print(f"Inspected {n_workloads} workloads.")
    print()

    # Cell occupancy report.
    print("Taxonomy cell occupancy (working_set, arithmetic_intensity, dispatch):")
    for cell, members in sorted(cell_counts.items(), key=lambda kv: -len(kv[1])):
        ws, ai, di = cell
        ws = "?" if ws is None else ws
        ai = "?" if ai is None else ai
        di = "?" if di is None else di
        print(f"  ({ws}, {ai}, {di}): {len(members)}")
        for m in members:
            print(f"      {m}")
    print()

    # Unmeasured tracker.
    for axis, names in unmeasured_axes.items():
        if names:
            print(f"Axis '{axis}' unmeasured on {len(names)} workload(s):")
            for n in names:
                print(f"  {n}")
            print()

    if all_errors:
        print(f"FAIL: {len(all_errors)} taxonomy violations:")
        for e in all_errors:
            print(f"  {e}")
        return 1
    print(f"PASS: {n_workloads} workloads consistent with taxonomy invariants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
