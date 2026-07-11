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
import sys
from pathlib import Path

import yaml

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
REFERENCE_EVIDENCE_SCHEMA = "mlperf-edu-reference-evidence/0.2"
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
        file_errors = check_declared_file(
            name,
            label="verified_baseline.evidence_file",
            relative_path=baseline.get("evidence_file"),
            digest=baseline.get("evidence_sha256"),
        )
        errors.extend(file_errors)
        if baseline.get("review_eligible") is not True:
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
        "schema": REFERENCE_EVIDENCE_SCHEMA,
        "workload": workload_id,
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
    if payload.get("invalid_reasons"):
        errors.append(f"{name}: reference summary contains invalid_reasons")
    if (payload.get("acceptance") or {}).get("passed") is not True:
        errors.append(f"{name}: reference summary acceptance did not pass")
    source = payload.get("source") or {}
    if source.get("git_dirty") is not False:
        errors.append(
            f"{name}: reference summary was not produced from a clean Git worktree"
        )
    if not PREFIXED_SHA256_RE.fullmatch(str(source.get("tool_sha256") or "")):
        errors.append(
            f"{name}: reference summary source.tool_sha256 is missing or invalid"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_sha") or "")):
        errors.append(f"{name}: reference summary source.git_sha is missing or invalid")

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
            if not path or Path(path).is_absolute() or ".." in Path(path).parts:
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
