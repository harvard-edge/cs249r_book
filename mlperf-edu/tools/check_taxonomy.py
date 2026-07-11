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
LEGACY_REFERENCE_EVIDENCE_SCHEMA = "mlperf-edu-reference-evidence/0.3"
SUPPORTED_REFERENCE_EVIDENCE_SCHEMAS = {
    LEGACY_REFERENCE_EVIDENCE_SCHEMA,
    REFERENCE_EVIDENCE_SCHEMA,
}
REFERENCE_INDEX_SCHEMA = "mlperf-edu-reference-index/0.2"
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
        lifecycle_declared = any(
            field in baseline
            for field in (
                "protocol_compatibility",
                "replacement_required",
            )
        )
        protocol_superseded = lifecycle == superseded_lifecycle
        if lifecycle_declared and not protocol_superseded:
            errors.append(
                f"{name}: historical evidence lifecycle must be exactly "
                "protocol_compatibility=superseded, review_eligible=false, "
                "replacement_required=true"
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


def check_nanogpt_lineage(name: str, baseline: dict, payload: dict) -> list[str]:
    """Verify the portable training-to-inference lineage recorded by NanoGPT."""
    errors: list[str] = []
    lineage = payload.get("nanogpt_training_lineage")
    if not isinstance(lineage, dict):
        return [f"{name}: checkpoint-backed NanoGPT summary lacks training lineage"]
    expected = {
        "required": True,
        "status": "staged",
        "package_schema": "mlperf-edu-package/0.2",
        "source_workload": "nanogpt-train",
    }
    for field, value in expected.items():
        if lineage.get(field) != value:
            errors.append(
                f"{name}: NanoGPT lineage {field} is {lineage.get(field)!r}, expected {value!r}"
            )
    package_digest = str(lineage.get("package_sha256") or "")
    if not PREFIXED_SHA256_RE.fullmatch(package_digest):
        errors.append(f"{name}: NanoGPT lineage package_sha256 is missing or invalid")
    elif baseline and baseline.get(
        "source_training_package_sha256"
    ) != package_digest.removeprefix("sha256:"):
        errors.append(
            f"{name}: verified_baseline.source_training_package_sha256 does not match NanoGPT lineage"
        )

    role_paths = {
        "checkpoint": "source_training_checkpoint",
        "source_training_provenance": "source_training_manifest",
        "source_training_report": "source_training_report",
    }
    for role, lineage_field in role_paths.items():
        expected_path = lineage.get(lineage_field)
        if not is_safe_posix_relative_path(expected_path):
            errors.append(
                f"{name}: NanoGPT lineage {lineage_field} is not a safe relative path"
            )
            continue
        indexed: list[dict] = []
        for run_index, run in enumerate(payload.get("runs") or []):
            matches = [
                artifact
                for artifact in run.get("artifacts") or []
                if isinstance(artifact, dict) and artifact.get("role") == role
            ]
            if len(matches) != 1:
                errors.append(
                    f"{name}: reference summary run {run_index} must index exactly one {role} artifact"
                )
            else:
                indexed.append(matches[0])
        if len(indexed) != len(payload.get("runs") or []):
            continue
        if {artifact.get("path") for artifact in indexed} != {expected_path}:
            errors.append(
                f"{name}: {role} artifact paths do not match NanoGPT lineage {lineage_field}"
            )
        digests = {artifact.get("sha256") for artifact in indexed}
        if len(digests) != 1 or not PREFIXED_SHA256_RE.fullmatch(
            str(next(iter(digests), ""))
        ):
            errors.append(
                f"{name}: {role} artifact digest is missing or differs across seeds"
            )
        elif role == "checkpoint" and baseline:
            checkpoint_digest = str(next(iter(digests))).removeprefix("sha256:")
            if baseline.get("source_training_checkpoint_sha256") != checkpoint_digest:
                errors.append(
                    f"{name}: verified_baseline.source_training_checkpoint_sha256 "
                    "does not match NanoGPT lineage"
                )
    return errors


def check_reference_payload_roles(name: str, body: dict, payload: dict) -> list[str]:
    """Validate workload-specific payload roles independently of a baseline row."""
    errors: list[str] = []
    baseline = body.get("verified_baseline") or {}
    if body.get("shared_checkpoint") == "nanogpt-train":
        errors.extend(check_nanogpt_lineage(name, baseline, payload))
    if payload.get("workload") == "slm-decode":
        for index, run in enumerate(payload.get("runs") or []):
            roles = {
                artifact.get("role")
                for artifact in run.get("artifacts") or []
                if isinstance(artifact, dict)
            }
            if "model_metadata" not in roles:
                errors.append(
                    f"{name}: reference summary run {index} lacks model_metadata"
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
        if payload.get("quality_metric") != primary_metric:
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
        errors.extend(check_reference_payload_roles(name, body, payload))
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


def load_committed_summary(body: dict) -> dict | None:
    baseline = body.get("verified_baseline") or {}
    if baseline.get("evidence_status") != "committed-reference-summary":
        return None
    relative_path = baseline.get("evidence_file")
    if not relative_path:
        return None
    path = (REPO_ROOT / str(relative_path)).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def check_shared_checkpoint_evidence(workloads: dict[str, dict]) -> list[str]:
    """Bind dependent inference baselines to the committed training baseline."""
    errors: list[str] = []
    for workload_id, body in workloads.items():
        source_id = body.get("shared_checkpoint")
        if not source_id:
            continue
        name = f"{body.get('suite', 'unknown')}/{workload_id}"
        source = workloads.get(str(source_id))
        if source is None:
            errors.append(
                f"{name}: shared checkpoint source {source_id!r} is not in the registry"
            )
            continue
        baseline = body.get("verified_baseline") or {}
        source_baseline = source.get("verified_baseline") or {}
        if baseline.get("evidence_status") != "committed-reference-summary":
            continue
        if source_baseline.get("evidence_status") != "committed-reference-summary":
            errors.append(
                f"{name}: shared checkpoint source {source_id} lacks committed evidence"
            )
            continue
        expected_links = {
            "source_training_evidence_id": source_baseline.get("evidence_id"),
            "source_training_evidence_sha256": source_baseline.get("evidence_sha256"),
        }
        for field, expected in expected_links.items():
            if baseline.get(field) != expected:
                errors.append(
                    f"{name}: verified_baseline.{field} does not match shared checkpoint source {source_id}"
                )

        payload = load_committed_summary(body)
        source_payload = load_committed_summary(source)
        if payload is None or source_payload is None:
            continue
        lineage = payload.get("nanogpt_training_lineage") or {}
        if lineage.get("source_workload") != source_id:
            errors.append(
                f"{name}: evidence lineage source_workload does not match shared checkpoint {source_id}"
            )
            continue

        dependent_checkpoint_digests: set[str] = set()
        for run in payload.get("runs") or []:
            if not isinstance(run, dict):
                continue
            dependent_checkpoint_digests.update(
                str(artifact.get("sha256"))
                for artifact in run.get("artifacts") or []
                if isinstance(artifact, dict) and artifact.get("role") == "checkpoint"
            )
        selected_seed = baseline.get("source_training_seed")
        selected_checkpoint_digest = baseline.get("source_training_checkpoint_sha256")
        if (
            isinstance(selected_seed, bool)
            or not isinstance(selected_seed, int)
            or not SHA256_RE.fullmatch(str(selected_checkpoint_digest or ""))
        ):
            errors.append(
                f"{name}: verified_baseline must identify a source_training_seed "
                "and source_training_checkpoint_sha256"
            )
            continue
        source_median = (
            (source_payload.get("aggregate") or {}).get("quality") or {}
        ).get("median")
        selected_runs = [
            run
            for run in source_payload.get("runs") or []
            if isinstance(run, dict) and run.get("requested_seed") == selected_seed
        ]
        source_checkpoint_digests = {
            str(artifact.get("sha256"))
            for run in selected_runs
            for artifact in run.get("artifacts") or []
            if isinstance(artifact, dict) and artifact.get("role") == "checkpoint"
        }
        if len(selected_runs) != 1 or len(source_checkpoint_digests) != 1:
            errors.append(
                f"{name}: shared checkpoint source does not identify exactly one selected-seed checkpoint"
            )
            continue
        if not numbers_match(selected_runs[0].get("quality_value"), source_median):
            errors.append(
                f"{name}: source_training_seed does not select the committed median-quality training run"
            )
        expected_checkpoint = "sha256:" + str(selected_checkpoint_digest)
        if source_checkpoint_digests != {expected_checkpoint}:
            errors.append(
                f"{name}: source_training_checkpoint_sha256 does not match the selected training run"
            )
        if dependent_checkpoint_digests != {expected_checkpoint}:
            errors.append(
                f"{name}: inference evidence checkpoint digest does not match the selected training checkpoint"
            )
    return errors


def check_reference_index(workloads: dict[str, dict]) -> list[str]:
    """Validate index closure, source lock, and registry-to-summary bindings."""
    errors: list[str] = []
    index_path = REPO_ROOT / "reference_results" / "index.json"
    try:
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"reference index cannot be read: {exc}"]
    if not isinstance(index, dict):
        return ["reference index root must be an object"]
    if index.get("schema") != REFERENCE_INDEX_SCHEMA:
        errors.append(
            f"reference index schema is {index.get('schema')!r}, "
            f"expected {REFERENCE_INDEX_SCHEMA!r}"
        )
    source_git_sha = index.get("source_git_sha")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_git_sha or "")):
        errors.append("reference index source_git_sha is missing or invalid")

    source_lock = index.get("source_lock") or {}
    public_bodies = [
        body
        for body in workloads.values()
        if (body.get("public") or {}).get("status")
        in {"score-bearing", "performance-bearing"}
    ]
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
    if not SHA256_RE.fullmatch(normalized_digest):
        return [
            f"{name}: {label} SHA-256 must be exactly 64 lowercase hexadecimal characters"
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

    all_errors.extend(check_shared_checkpoint_evidence(workload_bodies))
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
