from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from mlperf.assets import sha256_file
from mlperf.registry import QUALITY_BASELINE_ALIASES, Workload


PUBLIC_DATA_MODES = {
    "score-bearing": {"real", "real-preprocessed-mlperf-tiny-accuracy-set"},
    "performance-bearing": {
        "checkpoint-backed",
        "local-prompt",
        "local-prompt-batch",
        "local-prompt-long-context",
    },
}


def _report_model_id(report: dict[str, Any]) -> str | None:
    model = report.get("model")
    if isinstance(model, str) and model:
        return model
    if isinstance(model, dict):
        value = model.get("id")
        if isinstance(value, str) and value:
            return value
    return None


def _report_model_revision(report: dict[str, Any]) -> str | None:
    model = report.get("model")
    if isinstance(model, dict):
        value = model.get("revision")
        if isinstance(value, str) and value:
            return value
    return None


def _report_model_n_params(report: dict[str, Any]) -> int | None:
    model = report.get("model")
    candidates: list[Any] = []
    if isinstance(model, dict):
        candidates.append(model.get("n_params"))
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        candidates.append(metrics.get("n_params"))
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _report_dataset_name(report: dict[str, Any]) -> str | None:
    dataset = report.get("dataset")
    if isinstance(dataset, str) and dataset:
        return dataset
    if isinstance(dataset, dict):
        value = dataset.get("name")
        if isinstance(value, str) and value:
            return value
    return None


def _report_dataset_sha256(report: dict[str, Any]) -> str | None:
    dataset = report.get("dataset")
    if isinstance(dataset, dict):
        value = dataset.get("sha256")
        if isinstance(value, str) and value:
            return value
    return None


def _mapping_contract_issues(
    *,
    label: str,
    expected: Any,
    actual: Any,
) -> list[str]:
    """Return exact, field-level mismatches for one canonical mapping."""
    if not isinstance(expected, dict):
        return [f"registry canonical_max_contract.{label} is missing"]
    if not isinstance(actual, dict):
        return [f"report {label} is missing"]

    issues: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            issues.append(f"report {label}.{key} is missing")
        elif actual[key] != expected_value:
            issues.append(
                f"report {label}.{key}={actual[key]!r} does not match "
                f"canonical value {expected_value!r}"
            )
    unexpected = sorted(str(key) for key in actual.keys() - expected.keys())
    if unexpected:
        issues.append(
            f"report {label} contains undeclared canonical fields: {unexpected}"
        )
    return issues


def report_metric_value(
    report: dict[str, Any],
    metric: str | None,
    *,
    use_quality_metric_key: bool = True,
) -> tuple[str | None, float | None]:
    """Resolve a declared quality/functional metric to its numeric report value."""
    metrics = report.get("metrics")
    quality = report.get("quality")
    if not isinstance(metrics, dict):
        return None, None
    quality = quality if isinstance(quality, dict) else {}
    candidates: list[str] = []
    metric_key = quality.get("metric_key")
    if use_quality_metric_key and metric_key:
        candidates.append(str(metric_key))
    if metric:
        candidates.append(str(metric))
        candidates.extend(QUALITY_BASELINE_ALIASES.get(str(metric), ()))
    for key in candidates:
        value = metrics.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return key, float(value)
    return None, None


def _numeric_target_met(
    value: float,
    target: float,
    direction: str,
    *,
    tolerance: float = 0.0,
) -> bool:
    """Apply one direction-aware numeric gate with absolute tolerance."""
    if direction == "higher":
        return value + tolerance >= target
    if direction == "lower":
        return value - tolerance <= target
    if direction == "equal":
        return math.isclose(value, target, rel_tol=0.0, abs_tol=tolerance)
    raise ValueError(f"unsupported quality direction {direction!r}")


def evaluate_report_contract(
    workload: Workload, report: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate whether one max report is eligible for MLCommons-facing review."""
    status = workload.public_status
    profile = str(report.get("profile", ""))
    result: dict[str, Any] = {
        "schema": "mlperf-edu-review-contract/0.2",
        "public_status": status,
        "profile": profile,
        "review_eligible": False,
        "issues": [],
    }
    if status not in {"score-bearing", "performance-bearing"}:
        result["status"] = "not-public-candidate"
        return result
    if profile != "max":
        result["status"] = "not-applicable"
        result["note"] = (
            "Public review eligibility is evaluated on max-profile evidence."
        )
        return result

    issues: list[str] = []
    if report.get("status") != "passed":
        issues.append(f"runner status is {report.get('status')!r}, not 'passed'")

    canonical = workload.raw.get("canonical_max_contract")
    if not isinstance(canonical, dict):
        canonical = {}
        issues.append("registry canonical_max_contract is missing")

    report_workload = report.get("workload")
    if report_workload != workload.id:
        issues.append(
            f"workload identity {report_workload!r} does not match canonical {workload.id!r}"
        )
    if report.get("id") != workload.id:
        issues.append(
            f"report id {report.get('id')!r} does not match canonical {workload.id!r}"
        )
    if report.get("suite") != workload.suite:
        issues.append(
            f"suite identity {report.get('suite')!r} does not match canonical {workload.suite!r}"
        )

    expected_canonical_workload = workload.canonical_workload or workload.id
    report_canonical_workload = report.get("canonical_workload") or report_workload
    if report_canonical_workload != expected_canonical_workload:
        issues.append(
            f"canonical workload identity {report_canonical_workload!r} does not "
            f"match {expected_canonical_workload!r}"
        )
    if report.get("variant") != workload.variant:
        issues.append(
            f"variant identity {report.get('variant')!r} does not match canonical "
            f"{workload.variant!r}"
        )
    if report.get("scenario") != workload.scenario:
        issues.append(
            f"scenario {report.get('scenario')!r} does not match canonical "
            f"{workload.scenario!r}"
        )

    expected_model_id = canonical.get("model_id")
    report_model_id = _report_model_id(report)
    if report_model_id != expected_model_id:
        issues.append(
            f"model identity {report_model_id!r} does not match canonical "
            f"{expected_model_id!r}"
        )
    expected_model_revision = canonical.get("model_revision")
    if expected_model_revision is not None:
        report_model_revision = _report_model_revision(report)
        if report_model_revision != expected_model_revision:
            issues.append(
                f"model revision {report_model_revision!r} does not match canonical "
                f"{expected_model_revision!r}"
            )
    expected_model_n_params = canonical.get("model_n_params")
    if expected_model_n_params is not None:
        report_model_n_params = _report_model_n_params(report)
        if report_model_n_params != expected_model_n_params:
            issues.append(
                f"model parameter count {report_model_n_params!r} does not match "
                f"canonical {expected_model_n_params!r}"
            )

    expected_dataset = canonical.get("dataset")
    report_dataset = _report_dataset_name(report)
    if report_dataset != expected_dataset:
        issues.append(
            f"dataset identity {report_dataset!r} does not match canonical "
            f"{expected_dataset!r}"
        )
    expected_dataset_sha256 = canonical.get("dataset_sha256")
    if expected_dataset_sha256 is not None:
        report_dataset_sha256 = _report_dataset_sha256(report)
        if report_dataset_sha256 != expected_dataset_sha256:
            issues.append(
                f"dataset SHA-256 {report_dataset_sha256!r} does not match canonical "
                f"{expected_dataset_sha256!r}"
            )

    data_mode = str(report.get("data_mode", ""))
    if data_mode not in PUBLIC_DATA_MODES[status]:
        issues.append(f"data_mode {data_mode!r} is not eligible for {status} review")
    if data_mode != canonical.get("data_mode"):
        issues.append(
            f"data_mode {data_mode!r} does not match canonical "
            f"{canonical.get('data_mode')!r}"
        )

    issues.extend(
        _mapping_contract_issues(
            label="config",
            expected=canonical.get("config"),
            actual=report.get("config"),
        )
    )

    seed = report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        issues.append("report does not record an integer seed")

    quality = report.get("quality")
    quality = quality if isinstance(quality, dict) else {}
    functional_metric = workload.quality_metric
    if status == "performance-bearing":
        functional = workload.raw.get("functional_check")
        if isinstance(functional, dict):
            functional_metric = str(functional.get("metric") or "") or None
    functional_key, functional_value = report_metric_value(report, functional_metric)
    if not functional_key:
        issues.append(
            f"declared functional metric {functional_metric!r} is absent from report metrics"
        )
    if quality.get("quality_required") is not True:
        issues.append("quality/functional enforcement is disabled")
    if quality.get("target_met") is not True:
        issues.append("quality/functional target was not met")
    if quality.get("override") is True:
        issues.append("quality target override is not canonical")

    issues.extend(
        _mapping_contract_issues(
            label="quality",
            expected=canonical.get("quality"),
            actual=(
                {key: quality.get(key) for key in canonical.get("quality", {})}
                if isinstance(canonical.get("quality"), dict)
                else quality
            ),
        )
    )

    canonical_quality = canonical.get("quality")
    canonical_quality = canonical_quality if isinstance(canonical_quality, dict) else {}
    canonical_target = canonical_quality.get("target")
    canonical_direction = canonical_quality.get("direction")
    canonical_tolerance_value = canonical_quality.get("tolerance", 0.0)
    canonical_tolerance = 0.0
    if (
        isinstance(canonical_tolerance_value, bool)
        or not isinstance(canonical_tolerance_value, (int, float))
        or not math.isfinite(float(canonical_tolerance_value))
        or float(canonical_tolerance_value) < 0
    ):
        issues.append(
            f"canonical quality tolerance {canonical_tolerance_value!r} is invalid"
        )
    else:
        canonical_tolerance = float(canonical_tolerance_value)
    if functional_value is not None and isinstance(canonical_target, (int, float)):
        try:
            functional_target_met = _numeric_target_met(
                functional_value,
                float(canonical_target),
                str(canonical_direction),
                tolerance=canonical_tolerance,
            )
        except ValueError:
            functional_target_met = False
            issues.append(
                f"canonical quality direction {canonical_direction!r} is unsupported"
            )
        if not functional_target_met:
            issues.append(
                f"functional metric {functional_key!r} value {functional_value!r} "
                f"does not satisfy canonical target {canonical_direction} "
                f"{canonical_target!r} with tolerance {canonical_tolerance!r}"
            )

    canonical_gates = canonical.get("quality_gates")
    if canonical_gates is not None:
        reported_gates = quality.get("gates")
        metrics = report.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        if not isinstance(canonical_gates, dict) or not canonical_gates:
            issues.append("registry canonical_max_contract.quality_gates is invalid")
        elif not isinstance(reported_gates, dict):
            issues.append("report quality.gates is missing")
        else:
            if reported_gates.get("passed") is not True:
                issues.append("report quality.gates conjunction did not pass")
            for gate_name, expected_gate in canonical_gates.items():
                actual_gate = reported_gates.get(gate_name)
                if not isinstance(expected_gate, dict):
                    issues.append(
                        f"registry canonical quality gate {gate_name!r} is invalid"
                    )
                    continue
                if not isinstance(actual_gate, dict):
                    issues.append(f"report quality gate {gate_name!r} is missing")
                    continue
                for field in ("target", "direction"):
                    if actual_gate.get(field) != expected_gate.get(field):
                        issues.append(
                            f"report quality gate {gate_name}.{field}="
                            f"{actual_gate.get(field)!r} does not match canonical value "
                            f"{expected_gate.get(field)!r}"
                        )
                if actual_gate.get("met") is not True:
                    issues.append(f"report quality gate {gate_name!r} did not pass")
                gate_value = actual_gate.get("value")
                gate_target = expected_gate.get("target")
                gate_direction = expected_gate.get("direction")
                gate_tolerance_value = expected_gate.get("tolerance", 0.0)
                valid_value = (
                    not isinstance(gate_value, bool)
                    and isinstance(gate_value, (int, float))
                    and math.isfinite(float(gate_value))
                )
                if not valid_value or not isinstance(gate_target, (int, float)):
                    issues.append(
                        f"report quality gate {gate_name!r} lacks a finite numeric value"
                    )
                    continue
                valid_tolerance = (
                    not isinstance(gate_tolerance_value, bool)
                    and isinstance(gate_tolerance_value, (int, float))
                    and math.isfinite(float(gate_tolerance_value))
                    and float(gate_tolerance_value) >= 0
                )
                if not valid_tolerance:
                    issues.append(
                        f"registry canonical quality gate {gate_name!r} has invalid "
                        f"tolerance {gate_tolerance_value!r}"
                    )
                    gate_passed = False
                else:
                    try:
                        gate_passed = _numeric_target_met(
                            float(gate_value),
                            float(gate_target),
                            str(gate_direction),
                            tolerance=float(gate_tolerance_value),
                        )
                    except ValueError:
                        gate_passed = False
                if not gate_passed:
                    issues.append(
                        f"report quality gate {gate_name!r} value {gate_value!r} "
                        f"does not satisfy canonical target {gate_direction} "
                        f"{gate_target!r} with tolerance {gate_tolerance_value!r}"
                    )
                metric_key = expected_gate.get("metric_key")
                metric_value = metrics.get(metric_key)
                if metric_key and metric_value != gate_value:
                    issues.append(
                        f"report quality gate {gate_name!r} value does not match "
                        f"metrics.{metric_key}"
                    )

    registry_measurement = workload.raw.get("measurement_protocol")
    registry_measurement = (
        registry_measurement if isinstance(registry_measurement, dict) else {}
    )
    primary_metric = str(registry_measurement.get("primary_metric") or "") or None
    primary_key, primary_value = report_metric_value(
        report,
        primary_metric,
        use_quality_metric_key=False,
    )
    if not primary_metric:
        issues.append("registry measurement_protocol.primary_metric is missing")
    elif not primary_key:
        issues.append(
            f"declared primary metric {primary_metric!r} is absent from report metrics"
        )
    elif primary_value is not None and primary_value <= 0:
        issues.append(f"declared primary metric {primary_metric!r} must be positive")

    protocol = report.get("measurement_protocol")
    if not isinstance(protocol, dict):
        issues.append("repeatable measurement_protocol is missing")
    issues.extend(
        _mapping_contract_issues(
            label="measurement_protocol",
            expected=registry_measurement,
            actual=protocol,
        )
    )

    if status == "performance-bearing":
        if isinstance(protocol, dict):
            warmup_runs = protocol.get("warmup_runs")
            measured_runs = protocol.get("measured_runs")
            if (
                isinstance(warmup_runs, bool)
                or not isinstance(warmup_runs, int)
                or warmup_runs < 1
            ):
                issues.append("measurement_protocol requires at least one warmup run")
            if (
                isinstance(measured_runs, bool)
                or not isinstance(measured_runs, int)
                or measured_runs < 3
            ):
                issues.append(
                    "measurement_protocol requires at least three measured runs"
                )
            if not protocol.get("latency_statistics"):
                issues.append("measurement_protocol must declare latency statistics")

        metrics = report.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        timing_sample_counts = canonical.get("timing_sample_counts")
        if not isinstance(timing_sample_counts, dict):
            issues.append(
                "registry canonical_max_contract.timing_sample_counts is missing"
            )
        else:
            for metric_name, expected_count in timing_sample_counts.items():
                samples = metrics.get(metric_name)
                if not isinstance(samples, list):
                    issues.append(
                        f"timing sample metric {metric_name!r} is missing from report metrics"
                    )
                elif len(samples) != expected_count:
                    issues.append(
                        f"timing sample metric {metric_name!r} has {len(samples)} samples; "
                        f"canonical protocol requires {expected_count}"
                    )
                elif any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                    for value in samples
                ):
                    issues.append(
                        f"timing sample metric {metric_name!r} must contain only "
                        "finite positive numbers"
                    )
        if workload.raw.get("shared_checkpoint"):
            checkpoint = report.get("checkpoint_provenance")
            if not isinstance(checkpoint, dict) or not checkpoint.get(
                "checkpoint_sha256"
            ):
                issues.append(
                    "checkpoint-backed result lacks a checkpoint SHA-256 digest"
                )
            else:
                checkpoint_path = Path(str(checkpoint.get("checkpoint_path") or ""))
                if not checkpoint_path.is_file():
                    issues.append(
                        "checkpoint-backed result lacks the checkpoint artifact"
                    )
                elif (
                    checkpoint.get("checkpoint_sha256")
                    != f"sha256:{sha256_file(checkpoint_path)}"
                ):
                    issues.append(
                        "checkpoint artifact does not match checkpoint_provenance SHA-256"
                    )
                for role in ("source_report", "source_manifest"):
                    path = Path(str(checkpoint.get(f"{role}_path") or ""))
                    digest = checkpoint.get(f"{role}_sha256")
                    if not path.is_file():
                        issues.append(f"checkpoint lineage {role} artifact is missing")
                    elif digest != f"sha256:{sha256_file(path)}":
                        issues.append(
                            f"checkpoint lineage {role} SHA-256 does not match"
                        )
                if checkpoint.get("source_manifest_verified") is not True:
                    issues.append("checkpoint lineage source manifest was not verified")
                if checkpoint.get("source_quality_target_met") is not True:
                    issues.append(
                        "checkpoint lineage source training quality did not pass"
                    )
        if isinstance(workload.raw.get("model_source"), dict):
            model_asset = report.get("model_asset")
            if not isinstance(model_asset, dict) or not model_asset.get("revision"):
                issues.append("external model result lacks a pinned model revision")

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        issues.append("report does not declare artifacts")
    else:
        for role in ("report", "provenance"):
            value = artifacts.get(role)
            if not value or not Path(str(value)).is_file():
                issues.append(f"{role} artifact is missing")

    result.update(
        {
            "status": "passed" if not issues else "failed",
            "review_eligible": not issues,
            "metric": primary_metric,
            "metric_key": primary_key,
            "metric_value": primary_value,
            "functional_metric": functional_metric,
            "functional_metric_key": functional_key,
            "functional_metric_value": functional_value,
            "data_mode": data_mode,
            "issues": issues,
        }
    )
    return result


def evaluate_promotion_contract(
    workload: Workload, report: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate a max report against its canonical pre-promotion contract."""
    mode = report.get("mode")
    phase = report.get("phase")
    candidate = workload
    result_role = "score-bearing"

    if mode == "inference" and phase:
        inference = (workload.raw.get("mode_contracts") or {}).get("inference") or {}
        phase_contract = (inference.get("phases") or {}).get(str(phase))
        if not isinstance(phase_contract, dict):
            return {
                "schema": "mlperf-edu-promotion-contract/0.1",
                "status": "failed",
                "promotion_eligible": False,
                "mode": mode,
                "phase": phase,
                "issues": [f"registry has no inference phase contract for {phase!r}"],
            }
        canonical_base = workload.raw.get("canonical_max_contract") or {}
        raw = copy.deepcopy(workload.raw)
        raw["canonical_max_contract"] = {
            "model_id": canonical_base.get("model_id"),
            "dataset": inference.get("dataset"),
            "data_mode": "checkpoint-backed",
            "config": copy.deepcopy(phase_contract.get("config") or {}),
            "quality": copy.deepcopy(phase_contract.get("quality") or {}),
            "timing_sample_counts": copy.deepcopy(
                phase_contract.get("timing_sample_counts") or {}
            ),
        }
        raw["measurement_protocol"] = copy.deepcopy(
            phase_contract.get("measurement_protocol") or {}
        )
        raw["functional_check"] = {
            "metric": (phase_contract.get("quality") or {}).get("metric"),
            "condition": "The phase must complete its canonical functional gate.",
        }
        raw["shared_checkpoint"] = "causal-language-modeling:training:max"
        raw.pop("model_source", None)
        candidate = replace(
            workload,
            public_status="performance-bearing",
            scenario=str(phase_contract.get("scenario") or ""),
            raw=raw,
        )
        result_role = "performance-bearing"
    else:
        candidate = replace(workload, public_status="score-bearing")

    result = evaluate_report_contract(candidate, report)
    if workload.raw.get("promotion_scope", True) is not True:
        result.setdefault("issues", []).append(
            f"workload {workload.id!r} is not eligible for score-bearing review "
            "until its authoritative quality runner is integrated"
        )
        result["status"] = "failed"
    result["schema"] = "mlperf-edu-promotion-contract/0.1"
    result["promotion_eligible"] = result.get("status") == "passed"
    result["result_role"] = result_role
    result["registry_public_status"] = workload.public_status
    result["mode"] = mode
    result["phase"] = phase
    return result


def aggregate_contract_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect failed public review contracts from an aggregate report."""
    failures: list[dict[str, Any]] = []
    for item in report.get("workloads", []):
        if not isinstance(item, dict):
            continue
        contract = item.get("review_contract")
        if isinstance(contract, dict) and contract.get("status") == "failed":
            failures.append(
                {
                    "workload": item.get("workload") or item.get("id"),
                    "issues": list(contract.get("issues") or []),
                }
            )
    return failures
