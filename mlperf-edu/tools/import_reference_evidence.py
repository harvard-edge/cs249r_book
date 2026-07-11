#!/usr/bin/env python3
"""Import clean reference-sweep summaries without rewriting their bytes.

The raw sweep packages remain outside the repository because they can contain
large checkpoints and dataset-derived artifacts.  This tool promotes the
lightweight, content-addressed summaries that index those retained packages.

Examples:

    uv run python tools/import_reference_evidence.py \
        --evidence-root /path/to/reference_runs/review-SHA \
        --source-git-sha SHA

    uv run python tools/import_reference_evidence.py \
        --evidence-root /path/to/reference_runs/review-SHA \
        --source-git-sha SHA --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mlperf.contracts import evaluate_report_contract  # noqa: E402
from mlperf.edu_cli import (  # noqa: E402
    run_comparison_fingerprint_sha256,
    verify_package_archive,
)
from mlperf.manifest import verify_provd  # noqa: E402
from mlperf.registry import load_registry  # noqa: E402
from tools import check_taxonomy  # noqa: E402
from tools import reference_source_lock  # noqa: E402
from tools import run_reference_sweep  # noqa: E402

SUMMARY_SCHEMA = "mlperf-edu-reference-evidence/0.4"
LEGACY_SUMMARY_SCHEMA = "mlperf-edu-reference-evidence/0.3"
SUPPORTED_SUMMARY_SCHEMAS = {LEGACY_SUMMARY_SCHEMA, SUMMARY_SCHEMA}
INDEX_SCHEMA = "mlperf-edu-reference-index/0.2"
PUBLIC_CANDIDATE_STATUSES = {"score-bearing", "performance-bearing"}
PUBLIC_PRIMARY_METRIC_CV_LIMIT = 0.05
EMPTY_SHA256 = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
OUTPUT_ROOTS = (
    ROOT / "reference_results",
    ROOT / "src" / "mlperf_edu" / "reference_results",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def git_repository_root() -> Path:
    """Return the enclosing Git root or fail with an importer error."""
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


def source_sweep_tool_sha256(source_git_sha: str) -> str:
    """Hash the sweep tool bytes from the exact source Git object."""
    try:
        git_root = git_repository_root()
        tool_path = (ROOT / "tools" / "run_reference_sweep.py").relative_to(git_root)
        payload = subprocess.run(
            ["git", "show", f"{source_git_sha}:{tool_path.as_posix()}"],
            cwd=git_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"cannot read sweep tool from source Git object {source_git_sha}"
        ) from exc
    return "sha256:" + sha256_bytes(payload)


@contextmanager
def source_project_checkout(source_git_sha: str):
    """Yield a clean sparse checkout of the historical evidence source."""
    git_root = git_repository_root()
    try:
        project_prefix = ROOT.resolve().relative_to(git_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"project root is outside Git repository: {ROOT}") from exc
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
            raise ValueError(f"historical checkout lacks project path {project_prefix}")
        yield project_root


def expected_public_candidates() -> dict[str, Any]:
    """Return workload id -> native registry contract for public candidates."""
    registry = load_registry(ROOT / "registry")
    return {
        workload.id: workload
        for workload in registry.values()
        if workload.public_status in PUBLIC_CANDIDATE_STATUSES
    }


def load_summary(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: summary root must be an object")
    return payload, data


def rejected_attempt_reason(payload: dict[str, Any]) -> str | None:
    """Describe a sweep-rejected attempt that is safe to exclude from promotion."""
    reasons = payload.get("invalid_reasons")
    if (
        payload.get("status") == "invalid"
        and payload.get("eligible_for_public_baseline") is False
        and isinstance(reasons, list)
        and reasons
        and all(isinstance(reason, str) and reason for reason in reasons)
    ):
        return "; ".join(reasons)
    return None


def valid_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def declared_repeatability_protocol(workload_contract: Any) -> dict[str, Any]:
    raw = workload_contract.raw
    if workload_contract.public_status == "performance-bearing":
        protocol = raw.get("performance_reference_protocol")
    else:
        protocol = (raw.get("quality_target") or {}).get("reference_protocol")
    return protocol if isinstance(protocol, dict) else {}


def validate_primary_metric_repeatability(
    path: Path, payload: dict[str, Any], workload_contract: Any
) -> None:
    """Independently recompute the public packet's primary timing CV."""
    protocol = declared_repeatability_protocol(workload_contract)
    limit = protocol.get("repeatability_limit")
    if not check_taxonomy.numbers_match(limit, PUBLIC_PRIMARY_METRIC_CV_LIMIT):
        raise ValueError(
            f"{path}: registry public protocol must declare repeatability_limit="
            f"{PUBLIC_PRIMARY_METRIC_CV_LIMIT:g}"
        )
    metric = protocol.get("repeatability_metric")
    if not isinstance(metric, str) or not metric:
        raise ValueError(f"{path}: registry public protocol lacks repeatability_metric")
    values = [run.get("primary_metric_value") for run in payload.get("runs") or []]
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in values
    ):
        raise ValueError(
            f"{path}: primary performance repeatability values are invalid"
        )
    mean = statistics.fmean(float(value) for value in values)
    stdev = (
        statistics.stdev(float(value) for value in values) if len(values) > 1 else 0.0
    )
    coefficient_of_variation = stdev / mean
    expected = {
        "metric": metric,
        "coefficient_of_variation": coefficient_of_variation,
        "limit": float(limit),
        "passed": coefficient_of_variation <= float(limit),
    }
    actual = payload.get("primary_metric_repeatability")
    if not check_taxonomy.numbers_match(actual, expected):
        raise ValueError(
            f"{path}: primary_metric_repeatability does not match independently "
            "recomputed values"
        )
    if expected["passed"] is not True:
        raise ValueError(
            f"{path}: primary performance coefficient of variation "
            f"{coefficient_of_variation:.6f} exceeds limit {float(limit):.6f}"
        )
    if (
        workload_contract.public_status == "performance-bearing"
        and not check_taxonomy.numbers_match(payload.get("repeatability"), expected)
    ):
        raise ValueError(
            f"{path}: performance repeatability compatibility field does not match "
            "primary_metric_repeatability"
        )


def validate_summary_comparison_fingerprint(
    path: Path, payload: dict[str, Any]
) -> None:
    """Require one comparison context across every row in a new public packet."""
    digests = [
        run.get("comparison_fingerprint_sha256")
        for run in payload.get("runs") or []
        if isinstance(run, dict)
    ]
    if not digests or any(not valid_sha256_hex(digest) for digest in digests):
        raise ValueError(
            f"{path}: one or more runs lack a valid comparison_fingerprint_sha256"
        )
    distinct = set(digests)
    if len(distinct) != 1:
        raise ValueError(
            f"{path}: public-candidate runs have multiple comparison fingerprints"
        )
    only_digest = next(iter(distinct))
    if payload.get("comparison_fingerprint_sha256") != only_digest:
        raise ValueError(
            f"{path}: summary comparison_fingerprint_sha256 does not match its runs"
        )


def validate_summary(
    path: Path,
    payload: dict[str, Any],
    *,
    workload_contract: Any,
    source_git_sha: str,
    sweep_tool_sha256: str,
) -> None:
    workload = payload.get("workload")
    evidence_id = payload.get("evidence_id")
    failures: list[str] = []
    expected = {
        "status": "valid",
        "evidence_tier": "public-candidate",
        "eligible_for_public_baseline": True,
        "public_status": workload_contract.public_status,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            failures.append(f"{field}={payload.get(field)!r}, expected {value!r}")
    if payload.get("schema") not in SUPPORTED_SUMMARY_SCHEMAS:
        failures.append(
            f"schema={payload.get('schema')!r}, expected one of "
            f"{sorted(SUPPORTED_SUMMARY_SCHEMAS)!r}"
        )
    if not isinstance(workload, str) or not workload:
        failures.append("workload is missing")
    if not isinstance(evidence_id, str) or not check_taxonomy.EVIDENCE_ID_RE.fullmatch(
        evidence_id
    ):
        failures.append("evidence_id is missing or is not a safe path component")
    source = payload.get("source") or {}
    if source.get("git_dirty") is not False:
        failures.append("source.git_dirty is not false")
    for field in ("git_status_sha256", "git_patch_sha256"):
        if source.get(field) != EMPTY_SHA256:
            failures.append(f"source.{field} does not record an empty clean source")
    if source.get("git_sha") != source_git_sha:
        failures.append(
            f"source.git_sha={source.get('git_sha')!r}, expected {source_git_sha!r}"
        )
    if source.get("tool_sha256") != sweep_tool_sha256:
        failures.append("source.tool_sha256 does not match the exact source Git object")
    if (payload.get("acceptance") or {}).get("passed") is not True:
        failures.append("acceptance.passed is not true")
    if payload.get("invalid_reasons"):
        failures.append("invalid_reasons is not empty")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        failures.append("runs is empty")
    elif any(
        not isinstance(run, dict) or run.get("evidence_valid") is not True
        for run in runs
    ):
        failures.append("one or more runs are not evidence-valid")
    if failures:
        raise ValueError(f"{path}: " + "; ".join(failures))

    if payload.get("schema") == SUMMARY_SCHEMA:
        validate_summary_comparison_fingerprint(path, payload)
        validate_primary_metric_repeatability(path, payload, workload_contract)

    contract_body = dict(workload_contract.raw)
    contract_body.pop("verified_baseline", None)
    contract_errors = check_taxonomy.check_reference_summary(
        f"{workload_contract.suite}/{workload_contract.id}",
        contract_body,
        payload,
    )
    if contract_errors:
        raise ValueError(f"{path}: " + "; ".join(contract_errors))


def resolve_indexed_file(root: Path, relative_path: object, *, label: str) -> Path:
    text = str(relative_path or "")
    if not check_taxonomy.is_safe_posix_relative_path(text):
        raise ValueError(f"{label}: path is missing, absolute, or escapes its root")
    relative = Path(*text.split("/"))
    unresolved = root / relative
    if unresolved.is_symlink():
        raise ValueError(f"{label}: symlink artifacts are not allowed")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}: resolved path escapes its root") from exc
    if not resolved.is_file():
        raise ValueError(f"{label}: indexed artifact is missing: {text}")
    return resolved


def verify_file_claim(
    root: Path,
    claim: dict[str, Any],
    *,
    label: str,
    cache: dict[Path, tuple[int, str]],
) -> Path:
    path = resolve_indexed_file(root, claim.get("path"), label=label)
    if path not in cache:
        cache[path] = (path.stat().st_size, sha256_file(path))
    actual_size, actual_digest = cache[path]
    if claim.get("n_bytes") != actual_size:
        raise ValueError(
            f"{label}: n_bytes={claim.get('n_bytes')!r}, recomputed {actual_size}"
        )
    if claim.get("sha256") != actual_digest:
        raise ValueError(
            f"{label}: sha256={claim.get('sha256')!r}, recomputed {actual_digest}"
        )
    return path


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Load one UTF-8 JSON object and reject duplicate keys."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: JSON root must be an object")
    return payload


def require_semantic_match(label: str, actual: object, expected: object) -> None:
    if not check_taxonomy.numbers_match(actual, expected):
        raise ValueError(f"{label}: {actual!r} does not match {expected!r}")


def verify_manifest_against_source(
    manifest_path: Path,
    *,
    source_project_root: Path,
    cache: dict[Path, bool],
    label: str,
) -> None:
    if cache.get(manifest_path):
        return
    verification = verify_provd(manifest_path, repo_root=source_project_root)
    failures = [name for name, ok, _detail in verification.checks if not ok]
    if failures:
        raise ValueError(
            f"{label}: provenance verification against the historical source "
            f"failed: {failures}"
        )
    cache[manifest_path] = True


def verify_run_semantics(
    attempt_root: Path,
    payload: dict[str, Any],
    run: dict[str, Any],
    *,
    run_index: int,
    workload_contract: Any,
    source_project_root: Path,
    manifest_cache: dict[Path, bool],
) -> None:
    """Bind one compact summary row to its raw report and provenance."""
    label = f"{payload.get('workload')} run {run_index}"
    artifact_claims = {
        str(claim.get("role")): claim
        for claim in run.get("artifacts") or []
        if isinstance(claim, dict)
    }
    try:
        report_claim = artifact_claims["report"]
        manifest_claim = artifact_claims["provenance"]
    except KeyError as exc:
        raise ValueError(f"{label}: report or provenance artifact is missing") from exc
    require_semantic_match(
        f"{label} report path", report_claim.get("path"), run.get("report_path")
    )
    require_semantic_match(
        f"{label} manifest path",
        manifest_claim.get("path"),
        run.get("manifest_path"),
    )
    report_path = resolve_indexed_file(
        attempt_root, report_claim.get("path"), label=f"{label} report"
    )
    manifest_path = resolve_indexed_file(
        attempt_root, manifest_claim.get("path"), label=f"{label} manifest"
    )
    report = load_json_object(report_path, label=f"{label} report")
    manifest = load_json_object(manifest_path, label=f"{label} manifest")
    dual_metrics = payload.get("schema") == SUMMARY_SCHEMA
    if dual_metrics:
        expected_scenario = getattr(workload_contract, "scenario", None) or (
            workload_contract.raw.get("scenario")
        )
        if not isinstance(expected_scenario, str) or not expected_scenario:
            raise ValueError(f"{label}: registry scenario is missing")
        require_semantic_match(
            f"{label} report.scenario", report.get("scenario"), expected_scenario
        )
        require_semantic_match(
            f"{label} manifest.scenario", manifest.get("scenario"), expected_scenario
        )
        require_semantic_match(
            f"{label} summary scenario", run.get("scenario"), expected_scenario
        )
        require_semantic_match(
            f"{label} summary manifest scenario",
            run.get("manifest_scenario"),
            expected_scenario,
        )
        require_semantic_match(
            f"{label} summary registry scenario",
            run.get("registry_scenario"),
            expected_scenario,
        )

    for field, expected in (
        ("workload", payload.get("workload")),
        ("profile", payload.get("profile")),
        ("status", run.get("status")),
        ("seed", run.get("requested_seed")),
        ("backend", run.get("backend")),
        ("data_mode", run.get("data_mode")),
        ("variant", payload.get("variant")),
    ):
        require_semantic_match(f"{label} report.{field}", report.get(field), expected)
    if report.get("status") != "passed":
        raise ValueError(f"{label}: raw report did not pass")

    metrics = report.get("metrics") or {}
    if dual_metrics:
        primary = payload.get("primary_metric") or {}
        primary_name = primary.get("name")
        primary_key = run.get("primary_metric_key")
        primary_value = run.get("primary_metric_value")
        require_semantic_match(
            f"{label} declared primary metric",
            run.get("primary_metric_declared"),
            primary_name,
        )
        require_semantic_match(
            f"{label} primary metric value",
            metrics.get(primary_key),
            primary_value,
        )
        public_status = payload.get("public_status")
        if public_status == "score-bearing":
            gate_name = payload.get("quality_metric")
            gate_key = run.get("quality_metric_key")
            gate_value = run.get("quality_value")
            require_semantic_match(
                f"{label} declared quality metric",
                run.get("quality_metric_declared"),
                gate_name,
            )
            if run.get("functional_metric_declared") is not None:
                raise ValueError(
                    f"{label}: score-bearing run must not declare a functional metric"
                )
        elif public_status == "performance-bearing":
            gate_name = (payload.get("functional_gate") or {}).get("metric")
            gate_key = run.get("functional_metric_key")
            gate_value = run.get("functional_metric_value")
            require_semantic_match(
                f"{label} declared functional metric",
                run.get("functional_metric_declared"),
                gate_name,
            )
            if any(
                run.get(field) is not None
                for field in (
                    "quality_metric_declared",
                    "quality_metric_key",
                    "quality_value",
                )
            ):
                raise ValueError(
                    f"{label}: performance-bearing run must use its functional gate, "
                    "not score-bearing quality fields"
                )
        else:
            raise ValueError(f"{label}: unsupported public status {public_status!r}")
        require_semantic_match(
            f"{label} gate metric value", metrics.get(gate_key), gate_value
        )
    else:
        primary_name = payload.get("quality_metric")
        primary_key = run.get("quality_metric_key")
        primary_value = run.get("quality_value")
        gate_name = run.get("functional_metric_declared")
        gate_value = None
        require_semantic_match(
            f"{label} metric value", metrics.get(primary_key), primary_value
        )
        require_semantic_match(
            f"{label} declared metric",
            run.get("quality_metric_declared"),
            payload.get("quality_metric"),
        )

    review = report.get("review_contract") or {}
    recomputed_review = evaluate_report_contract(workload_contract, report)
    if recomputed_review.get("status") != "passed":
        raise ValueError(
            f"{label}: independently recomputed review contract failed: "
            f"{recomputed_review.get('issues') or []}"
        )
    for field in (
        "status",
        "review_eligible",
        "public_status",
        "profile",
        "data_mode",
        "metric",
        "metric_key",
        "metric_value",
        "functional_metric",
        "functional_metric_key",
        "functional_metric_value",
    ):
        require_semantic_match(
            f"{label} independently recomputed review_contract.{field}",
            review.get(field),
            recomputed_review.get(field),
        )
    require_semantic_match(
        f"{label} independently recomputed review_contract.issues",
        review.get("issues"),
        recomputed_review.get("issues"),
    )
    expected_review = {
        "status": "passed",
        "review_eligible": True,
        "public_status": payload.get("public_status"),
        "profile": payload.get("profile"),
        "data_mode": run.get("data_mode"),
        "metric": primary_name,
        "metric_key": primary_key,
        "metric_value": primary_value,
        "functional_metric": gate_name,
    }
    if dual_metrics:
        expected_review.update(
            {
                "functional_metric_key": gate_key,
                "functional_metric_value": gate_value,
            }
        )
    for field, expected in expected_review.items():
        require_semantic_match(
            f"{label} review_contract.{field}", review.get(field), expected
        )
    if review.get("issues"):
        raise ValueError(f"{label}: raw review contract contains issues")
    review_metric_key = review.get("metric_key")
    if not isinstance(review_metric_key, str) or not review_metric_key:
        raise ValueError(f"{label}: raw review contract metric_key is missing")
    require_semantic_match(
        f"{label} review metric bytes",
        metrics.get(review_metric_key),
        primary_value,
    )

    quality = report.get("quality") or {}
    grade = run.get("grade") or {}
    expected_grade = {
        "status": "passed",
        "passed": True,
        "target_met": True,
        "metric": gate_name,
        "value": gate_value if dual_metrics else review.get("functional_metric_value"),
        "target": quality.get("target"),
    }
    for field, expected in expected_grade.items():
        require_semantic_match(f"{label} grade.{field}", grade.get(field), expected)
    require_semantic_match(
        f"{label} quality.target_met",
        quality.get("target_met"),
        run.get("quality_target_met"),
    )

    fingerprint = report.get("run_fingerprint") or {}
    if dual_metrics:
        reported_comparison_fingerprint = fingerprint.get(
            "comparison_fingerprint_sha256"
        )
        if not valid_sha256_hex(reported_comparison_fingerprint):
            raise ValueError(
                f"{label}: raw report comparison_fingerprint_sha256 is malformed"
            )
        require_semantic_match(
            f"{label} raw report comparison fingerprint",
            reported_comparison_fingerprint,
            run_comparison_fingerprint_sha256(fingerprint),
        )
        require_semantic_match(
            f"{label} summary-row comparison fingerprint",
            run.get("comparison_fingerprint_sha256"),
            reported_comparison_fingerprint,
        )
        require_semantic_match(
            f"{label} summary comparison fingerprint",
            payload.get("comparison_fingerprint_sha256"),
            reported_comparison_fingerprint,
        )
    execution = fingerprint.get("execution") or {}
    hardware = fingerprint.get("hardware") or {}
    for field, expected in (
        ("workload", payload.get("workload")),
        ("profile", payload.get("profile")),
        ("seed", run.get("requested_seed")),
        ("status", run.get("status")),
    ):
        require_semantic_match(
            f"{label} fingerprint execution.{field}",
            execution.get(field),
            expected,
        )
    registry_scenario = getattr(workload_contract, "scenario", None)
    row_registry_scenario = run.get("registry_scenario")
    scenario_expected = registry_scenario or row_registry_scenario
    scenario_values = {
        "report": report.get("scenario"),
        "manifest": manifest.get("scenario"),
        "summary-row": run.get("scenario"),
        "summary-row manifest": run.get("manifest_scenario"),
        "summary-row registry": row_registry_scenario,
        "fingerprint execution": execution.get("scenario"),
    }
    if scenario_expected is not None or any(
        value is not None for value in scenario_values.values()
    ):
        scenario_expected = scenario_expected or report.get("scenario")
        for scenario_label, value in scenario_values.items():
            require_semantic_match(
                f"{label} {scenario_label} scenario",
                value,
                scenario_expected,
            )
        if execution.get("scenarios") is not None:
            require_semantic_match(
                f"{label} fingerprint execution.scenarios",
                execution.get("scenarios"),
                [scenario_expected],
            )
    require_semantic_match(
        f"{label} fingerprint backends",
        execution.get("backends"),
        run.get("fingerprint_backends"),
    )
    require_semantic_match(
        f"{label} fingerprint data mode",
        execution.get("data_modes"),
        [run.get("data_mode")],
    )
    require_semantic_match(
        f"{label} hardware chip", hardware.get("chip"), run.get("chip")
    )
    require_semantic_match(
        f"{label} hardware backend",
        hardware.get("backend"),
        run.get("hardware_backend"),
    )

    require_semantic_match(
        f"{label} manifest workload",
        manifest.get("workload"),
        payload.get("workload"),
    )
    manifest_seed = ((manifest.get("leaves") or {}).get("rng") or {}).get("seed")
    require_semantic_match(
        f"{label} manifest seed", manifest_seed, run.get("requested_seed")
    )
    require_semantic_match(
        f"{label} recorded report seed",
        run.get("report_recorded_seed"),
        run.get("requested_seed"),
    )
    require_semantic_match(
        f"{label} recorded manifest seed",
        run.get("manifest_recorded_seed"),
        run.get("requested_seed"),
    )
    measurement = (manifest.get("leaves") or {}).get("measurement") or {}
    require_semantic_match(
        f"{label} manifest report digest",
        measurement.get("report_file_sha256"),
        sha256_file(report_path),
    )
    require_semantic_match(
        f"{label} manifest report size",
        measurement.get("n_bytes"),
        report_path.stat().st_size,
    )
    try:
        recorded_report_path = Path(str(measurement.get("report_path"))).resolve()
    except OSError as exc:
        raise ValueError(f"{label}: invalid manifest report path") from exc
    if recorded_report_path != report_path:
        raise ValueError(f"{label}: manifest report path does not name the raw report")
    verify_manifest_against_source(
        manifest_path,
        source_project_root=source_project_root,
        cache=manifest_cache,
        label=label,
    )

    if payload.get("workload") == "slm-decode":
        evaluation = report.get("quality_evaluation") or {}
        result = evaluation.get("result") or {}
        contract = workload_contract.raw.get("quality_evaluation") or {}
        if evaluation.get("status") != "passed":
            raise ValueError(f"{label}: SLM quality evaluation did not pass")
        require_semantic_match(
            f"{label} SLM quality suite", evaluation.get("suite"), contract.get("suite")
        )
        require_semantic_match(
            f"{label} SLM fixture version",
            result.get("fixture_version"),
            contract.get("fixture_version"),
        )
        require_semantic_match(
            f"{label} SLM case count", result.get("cases"), contract.get("cases")
        )
        require_semantic_match(
            f"{label} SLM category count",
            result.get("categories"),
            contract.get("categories"),
        )
        require_semantic_match(
            f"{label} SLM aggregation",
            result.get("aggregation"),
            contract.get("aggregation"),
        )
        require_semantic_match(
            f"{label} SLM category guard",
            result.get("category_guard"),
            contract.get("category_guard"),
        )
        require_semantic_match(
            f"{label} SLM quality asset",
            result.get("suite_sha256"),
            "sha256:" + str(contract.get("asset_sha256")),
        )
        perplexity = result.get("perplexity")
        if (
            isinstance(perplexity, bool)
            or not isinstance(perplexity, (int, float))
            or float(perplexity) > float(contract.get("maximum"))
        ):
            raise ValueError(f"{label}: SLM continuation perplexity exceeds its gate")
        require_semantic_match(
            f"{label} SLM perplexity metric",
            metrics.get("quality_perplexity"),
            perplexity,
        )
        worst_category_perplexity = result.get("worst_category_perplexity")
        if (
            isinstance(worst_category_perplexity, bool)
            or not isinstance(worst_category_perplexity, (int, float))
            or float(worst_category_perplexity)
            > float(contract.get("worst_category_maximum"))
        ):
            raise ValueError(
                f"{label}: SLM worst-category continuation perplexity exceeds its gate"
            )
        require_semantic_match(
            f"{label} SLM worst-category perplexity metric",
            metrics.get("quality_worst_category_perplexity"),
            worst_category_perplexity,
        )
        require_semantic_match(
            f"{label} SLM continuation token count",
            metrics.get("quality_total_continuation_tokens"),
            result.get("total_continuation_tokens"),
        )
        gates = evaluation.get("gates") or {}
        if gates.get("passed") is not True:
            raise ValueError(f"{label}: SLM quality gate conjunction did not pass")
        for gate_name, target in (
            ("overall_perplexity", contract.get("maximum")),
            (
                "worst_category_perplexity",
                contract.get("worst_category_maximum"),
            ),
        ):
            gate = gates.get(gate_name) or {}
            require_semantic_match(
                f"{label} SLM {gate_name} target", gate.get("target"), target
            )
            require_semantic_match(
                f"{label} SLM {gate_name} direction",
                gate.get("direction"),
                "lower",
            )
            if gate.get("met") is not True:
                raise ValueError(f"{label}: SLM {gate_name} gate did not pass")
        require_semantic_match(
            f"{label} SLM NLL metric",
            metrics.get("quality_mean_nll"),
            result.get("mean_nll"),
        )

    if workload_contract.raw.get("shared_checkpoint") == "nanogpt-train":
        provenance = report.get("checkpoint_provenance") or {}
        role_paths = {
            "checkpoint": "checkpoint_path",
            "source_training_provenance": "source_manifest_path",
            "source_training_report": "source_report_path",
        }
        resolved_roles: dict[str, Path] = {}
        for role, provenance_field in role_paths.items():
            claim = artifact_claims.get(role)
            if not isinstance(claim, dict):
                raise ValueError(f"{label}: missing {role} artifact")
            path = resolve_indexed_file(
                attempt_root, claim.get("path"), label=f"{label} {role}"
            )
            resolved_roles[role] = path
            if Path(str(provenance.get(provenance_field))).resolve() != path:
                raise ValueError(
                    f"{label}: checkpoint provenance {provenance_field} path mismatch"
                )
        require_semantic_match(
            f"{label} checkpoint digest",
            provenance.get("checkpoint_sha256"),
            artifact_claims["checkpoint"].get("sha256"),
        )
        source_manifest = load_json_object(
            resolved_roles["source_training_provenance"],
            label=f"{label} source training manifest",
        )
        source_report = load_json_object(
            resolved_roles["source_training_report"],
            label=f"{label} source training report",
        )
        source_seed = ((source_manifest.get("leaves") or {}).get("rng") or {}).get(
            "seed"
        )
        require_semantic_match(
            f"{label} source training seed",
            provenance.get("source_seed"),
            source_seed,
        )
        require_semantic_match(
            f"{label} source report seed", source_report.get("seed"), source_seed
        )
        require_semantic_match(
            f"{label} source checkpoint digest",
            ((source_manifest.get("leaves") or {}).get("weights") or {}).get("sha256"),
            provenance.get("checkpoint_sha256"),
        )
        source_quality = source_report.get("quality") or {}
        source_metrics = source_report.get("metrics") or {}
        source_metric_key = source_quality.get("metric_key")
        require_semantic_match(
            f"{label} source quality value",
            provenance.get("source_quality_value"),
            source_metrics.get(source_metric_key),
        )
        if (
            provenance.get("source_manifest_verified") is not True
            or provenance.get("source_quality_target_met") is not True
            or source_quality.get("target_met") is not True
            or source_report.get("status") != "passed"
        ):
            raise ValueError(f"{label}: source NanoGPT training evidence did not pass")
        verify_manifest_against_source(
            resolved_roles["source_training_provenance"],
            source_project_root=source_project_root,
            cache=manifest_cache,
            label=f"{label} source training",
        )


def verify_lineage_package(
    evidence_root: Path,
    attempt_root: Path,
    payload: dict[str, Any],
    *,
    cache: dict[Path, tuple[int, str]],
    source_project_root: Path,
) -> set[Path]:
    lineage = payload.get("nanogpt_training_lineage")
    if not isinstance(lineage, dict):
        return set()
    expected_digest = str(lineage.get("package_sha256") or "")
    matching_archives: list[Path] = []
    for archive in sorted(evidence_root.rglob("*.zip")):
        if archive.is_symlink() or not archive.is_file():
            continue
        if archive not in cache:
            cache[archive] = (archive.stat().st_size, sha256_file(archive))
        if cache[archive][1] == expected_digest:
            matching_archives.append(archive)
    if len(matching_archives) != 1:
        raise ValueError(
            f"{attempt_root}: expected exactly one retained lineage archive with "
            f"digest {expected_digest}, found {len(matching_archives)}"
        )
    archive = matching_archives[0]
    try:
        package_index = run_reference_sweep._preflight_lineage_archive(archive)
        package_checks = verify_package_archive(archive, repo_root=source_project_root)
    except run_reference_sweep.LineagePackageError as exc:
        raise ValueError(
            f"{archive}: lineage package verification failed: {exc}"
        ) from exc
    failed_checks = [name for name, ok, _detail in package_checks if not ok]
    if failed_checks:
        raise ValueError(
            f"{archive}: lineage package failed historical clean-extraction "
            f"verification: {failed_checks}"
        )
    if sha256_file(archive) != expected_digest:
        raise ValueError(f"{archive}: package digest changed during verification")

    staged_text = str(lineage.get("staged_root") or "")
    if not check_taxonomy.is_safe_posix_relative_path(staged_text):
        raise ValueError(f"{attempt_root}: lineage staged_root is not a safe path")
    staged_relative = Path(*staged_text.split("/"))
    staged_root = (attempt_root / staged_relative).resolve()
    try:
        staged_root.relative_to(attempt_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{attempt_root}: lineage staged_root escapes attempt"
        ) from exc
    staged_index = resolve_indexed_file(
        staged_root, "package_index.json", label=f"{attempt_root}: staged package index"
    )
    index_bytes = staged_index.read_bytes()
    try:
        staged_package_index = json.loads(index_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{staged_index}: invalid package index: {exc}") from exc
    if staged_package_index.get("schema") != "mlperf-edu-package/0.2":
        raise ValueError(f"{staged_index}: unsupported package schema")
    verification = staged_package_index.get("verification")
    if (
        not isinstance(verification, list)
        or not verification
        or not all(
            isinstance(check, dict) and check.get("ok") is True
            for check in verification
        )
    ):
        raise ValueError(f"{staged_index}: source verification contains a failure")
    if (staged_package_index.get("source_verification") or {}).get(
        "passed"
    ) is not True:
        raise ValueError(f"{staged_index}: source verification did not pass")
    clean = staged_package_index.get("clean_extraction_verification") or {}
    if clean.get("required") is not True or clean.get("status") != "passed":
        raise ValueError(f"{staged_index}: clean extraction verification did not pass")

    included = staged_package_index.get("included_files")
    if not isinstance(included, list) or not included:
        raise ValueError(f"{staged_index}: package contains no indexed files")
    indexed_paths: set[str] = set()
    retained_paths = {staged_index.resolve()}
    for index, claim in enumerate(included):
        if not isinstance(claim, dict):
            raise ValueError(f"{staged_index}: included file {index} is not an object")
        relative = str(claim.get("path") or "")
        if relative in indexed_paths:
            raise ValueError(f"{staged_index}: duplicate indexed path {relative!r}")
        indexed_paths.add(relative)
        retained_paths.add(
            verify_file_claim(
                staged_root,
                claim,
                label=f"{staged_index}: included file {index}",
                cache=cache,
            )
        )

    if package_index != staged_package_index:
        raise ValueError(f"{archive}: verified archive index differs from staged index")
    with zipfile.ZipFile(archive) as package:
        if package.read("package_index.json") != index_bytes:
            raise ValueError(f"{archive}: staged package index differs from archive")
    return retained_paths


def verify_external_artifacts(
    evidence_root: Path,
    summary_path: Path,
    payload: dict[str, Any],
    data: bytes,
    *,
    cache: dict[Path, tuple[int, str]],
    source_project_root: Path | None = None,
) -> None:
    evidence_root = evidence_root.resolve()
    if summary_path.is_symlink():
        raise ValueError(f"{summary_path}: summary may not be a symlink")
    summary_path = summary_path.resolve()
    attempt_root = summary_path.parent
    try:
        summary_path.relative_to(evidence_root)
        attempt_root.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(
            f"{summary_path}: retained attempt escapes the evidence root"
        ) from exc
    sidecar = summary_path.with_suffix(".json.sha256")
    expected_sidecar = f"{sha256_bytes(data)}  {summary_path.name}\n"
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError(f"{summary_path}: adjacent SHA-256 sidecar is missing")
    if sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise ValueError(f"{summary_path}: adjacent SHA-256 sidecar does not match")

    retained_paths = {summary_path.resolve(), sidecar.resolve()}
    for run_index, run in enumerate(payload.get("runs") or []):
        for artifact_index, claim in enumerate(run.get("artifacts") or []):
            if not isinstance(claim, dict):
                raise ValueError(
                    f"{summary_path}: run {run_index} artifact {artifact_index} is not an object"
                )
            retained_paths.add(
                verify_file_claim(
                    attempt_root,
                    claim,
                    label=f"{summary_path}: run {run_index} artifact {artifact_index}",
                    cache=cache,
                )
            )
    if payload.get("nanogpt_training_lineage") and source_project_root is None:
        raise ValueError(
            f"{summary_path}: historical source checkout is required for lineage verification"
        )
    if source_project_root is not None:
        retained_paths.update(
            verify_lineage_package(
                evidence_root,
                attempt_root,
                payload,
                cache=cache,
                source_project_root=source_project_root,
            )
        )

    actual_paths: set[Path] = set()
    for path in attempt_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"{summary_path}: retained attempt contains symlink {path}"
            )
        if path.is_file():
            actual_paths.add(path.resolve())
    unindexed = sorted(actual_paths.difference(retained_paths))
    if unindexed:
        relative = [path.relative_to(attempt_root).as_posix() for path in unindexed]
        raise ValueError(
            f"{summary_path}: retained attempt has unindexed files {relative}"
        )


def discover_summaries(
    evidence_root: Path,
    *,
    source_git_sha: str,
    sweep_tool_sha256: str,
    source_project_root: Path,
    rejected_attempts: list[tuple[Path, str]] | None = None,
) -> dict[str, tuple[Path, dict[str, Any], bytes]]:
    """Discover exactly one valid summary for every public candidate."""
    expected = expected_public_candidates()
    selected: dict[str, tuple[Path, dict[str, Any], bytes]] = {}
    file_cache: dict[Path, tuple[int, str]] = {}
    manifest_cache: dict[Path, bool] = {}
    for path in sorted(evidence_root.rglob("evidence_summary.json")):
        if path.is_symlink():
            raise ValueError(f"{path}: summary may not be a symlink")
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(evidence_root.resolve())
            resolved_path.parent.relative_to(evidence_root.resolve())
        except ValueError as exc:
            raise ValueError(f"{path}: summary escapes the evidence root") from exc
        payload, data = load_summary(resolved_path)
        workload = payload.get("workload")
        if workload not in expected:
            if payload.get("eligible_for_public_baseline") is True:
                raise ValueError(
                    f"{path}: eligible summary names unexpected workload {workload!r}"
                )
            continue
        rejection = rejected_attempt_reason(payload)
        if rejection is not None:
            if rejected_attempts is not None:
                rejected_attempts.append((resolved_path, rejection))
            continue
        validate_summary(
            resolved_path,
            payload,
            workload_contract=expected[workload],
            source_git_sha=source_git_sha,
            sweep_tool_sha256=sweep_tool_sha256,
        )
        verify_external_artifacts(
            evidence_root,
            resolved_path,
            payload,
            data,
            cache=file_cache,
            source_project_root=source_project_root,
        )
        for run_index, run in enumerate(payload.get("runs") or []):
            verify_run_semantics(
                resolved_path.parent,
                payload,
                run,
                run_index=run_index,
                workload_contract=expected[workload],
                source_project_root=source_project_root,
                manifest_cache=manifest_cache,
            )
        if workload in selected:
            raise ValueError(
                f"multiple eligible summaries found for {workload}: "
                f"{selected[workload][0]} and {resolved_path}"
            )
        selected[workload] = (resolved_path, payload, data)

    missing = sorted(set(expected).difference(selected))
    if missing:
        raise ValueError(
            "missing eligible summaries for public candidates: " + ", ".join(missing)
        )
    return selected


def build_index(
    selected: dict[str, tuple[Path, dict[str, Any], bytes]],
    *,
    source_git_sha: str,
    source_lock: dict[str, Any],
    source_lock_bytes: bytes,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for workload, (_, payload, data) in sorted(selected.items()):
        evidence_id = str(payload["evidence_id"])
        relative_path = Path("reference_results") / workload / f"{evidence_id}.json"
        dual_metrics = payload.get("schema") == SUMMARY_SCHEMA
        primary_metric = (
            (payload.get("primary_metric") or {}).get("name")
            if dual_metrics
            else payload.get("quality_metric")
        )
        entry = {
            "acceptance": payload.get("acceptance"),
            "aggregate": payload.get("aggregate"),
            "evidence_id": evidence_id,
            "evidence_sha256": sha256_bytes(data),
            "metric": primary_metric,
            "path": relative_path.as_posix(),
            "profile": payload.get("profile"),
            "public_status": payload.get("public_status"),
            "reference_metric_role": payload.get("reference_metric_role"),
            "seeds": payload.get("seeds_requested"),
            "variant": payload.get("variant"),
            "workload": workload,
        }
        if dual_metrics:
            entry["quality_metric"] = payload.get("quality_metric")
            entry["quality_gate"] = payload.get("quality_gate")
        if payload.get("public_status") == "performance-bearing":
            entry["functional_gate"] = (payload.get("basis") or {}).get(
                "functional_check"
            )
            if not dual_metrics:
                entry["legacy_summary_semantics"] = {
                    "aggregate.quality": (
                        "primary performance metric samples; the field name is retained "
                        "from evidence schema 0.2"
                    ),
                    "quality_target": (
                        "not a speed threshold; acceptance is the all-runs functional "
                        "gate recorded above"
                    ),
                }
        entries.append(entry)
    return {
        "schema": INDEX_SCHEMA,
        "source_git_sha": source_git_sha,
        "source_lock": {
            "path": "reference_results/source_lock.json",
            "schema": source_lock.get("schema"),
            "sha256": reference_source_lock.sha256_bytes(source_lock_bytes),
            "file_count": source_lock.get("file_count"),
            "contract_count": source_lock.get("contract_count"),
        },
        "summary_count": len(entries),
        "summaries": entries,
    }


def sync_file(path: Path, expected: bytes, *, check: bool) -> bool:
    """Write or check one file. Return true when it was already current."""
    current = path.read_bytes() if path.is_file() else None
    if current == expected:
        return True
    if check:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)
    return False


def require_safe_destination(path: Path) -> None:
    """Refuse generated destinations that escape the project through symlinks."""
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"generated evidence destination escapes the project: {path}"
        ) from exc
    if path.is_symlink():
        raise ValueError(f"generated evidence destination may not be a symlink: {path}")


def paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def require_separate_source_and_destinations(evidence_root: Path) -> None:
    for output_root in OUTPUT_ROOTS:
        if paths_overlap(evidence_root, output_root):
            raise ValueError(
                "evidence source and generated destination may not contain one "
                f"another: {evidence_root} vs {output_root}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the imported summaries or index differ; do not write.",
    )
    args = parser.parse_args()

    if not args.evidence_root.is_dir():
        parser.error(f"evidence root is not a directory: {args.evidence_root}")
    if len(args.source_git_sha) != 40 or any(
        char not in "0123456789abcdef" for char in args.source_git_sha
    ):
        parser.error("--source-git-sha must be 40 lowercase hexadecimal characters")

    evidence_root = args.evidence_root.resolve()
    try:
        require_separate_source_and_destinations(evidence_root)
        sweep_tool_sha256 = source_sweep_tool_sha256(args.source_git_sha)
        source_lock = reference_source_lock.build_source_lock(
            args.source_git_sha, project_root=ROOT
        )
        source_lock_bytes = reference_source_lock.canonical_json_bytes(source_lock)
        rejected_attempts: list[tuple[Path, str]] = []
        with source_project_checkout(args.source_git_sha) as source_project_root:
            selected = discover_summaries(
                evidence_root,
                source_git_sha=args.source_git_sha,
                sweep_tool_sha256=sweep_tool_sha256,
                source_project_root=source_project_root,
                rejected_attempts=rejected_attempts,
            )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    stale: list[Path] = []
    index = build_index(
        selected,
        source_git_sha=args.source_git_sha,
        source_lock=source_lock,
        source_lock_bytes=source_lock_bytes,
    )
    index_bytes = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
    try:
        for output_root in OUTPUT_ROOTS:
            require_safe_destination(output_root)
            expected_paths = {
                output_root / "index.json",
                output_root / "source_lock.json",
            }
            for workload, (_, payload, data) in sorted(selected.items()):
                destination = output_root / workload / f"{payload['evidence_id']}.json"
                require_safe_destination(destination)
                expected_paths.add(destination)
                if not sync_file(destination, data, check=args.check):
                    stale.append(destination)
            index_path = output_root / "index.json"
            require_safe_destination(index_path)
            if not sync_file(index_path, index_bytes, check=args.check):
                stale.append(index_path)
            source_lock_path = output_root / "source_lock.json"
            require_safe_destination(source_lock_path)
            if not sync_file(source_lock_path, source_lock_bytes, check=args.check):
                stale.append(source_lock_path)
            extra_paths = (
                sorted(set(output_root.rglob("*.json")).difference(expected_paths))
                if output_root.is_dir()
                else []
            )
            if args.check:
                stale.extend(extra_paths)
            else:
                for path in extra_paths:
                    path.unlink()
                for path in sorted(output_root.rglob("*"), reverse=True):
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.check and stale:
        print("FAIL: imported reference evidence is out of date:")
        for path in stale:
            print(f"  - {path}")
        return 1

    action = "verified" if args.check else "synchronized"
    print(
        f"PASS: {action} {len(selected)} clean public-candidate summaries "
        f"from {args.source_git_sha}."
    )
    if rejected_attempts:
        print(
            f"INFO: excluded {len(rejected_attempts)} explicitly rejected "
            "create-once attempt(s)."
        )
    for workload, (_, payload, data) in sorted(selected.items()):
        print(f"  {workload}: {payload['evidence_id']} sha256:{sha256_bytes(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
