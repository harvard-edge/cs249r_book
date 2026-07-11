from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mlperf.assets import sha256_file
from mlperf.registry import QUALITY_BASELINE_ALIASES, Workload


PUBLIC_DATA_MODES = {
    "score-bearing": {"real"},
    "performance-bearing": {
        "checkpoint-backed",
        "local-prompt",
        "local-prompt-batch",
        "local-prompt-long-context",
    },
}


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


def evaluate_report_contract(
    workload: Workload, report: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate whether one max report is eligible for MLCommons-facing review."""
    status = workload.public_status
    profile = str(report.get("profile", ""))
    result: dict[str, Any] = {
        "schema": "mlperf-edu-review-contract/0.1",
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

    data_mode = str(report.get("data_mode", ""))
    if data_mode not in PUBLIC_DATA_MODES[status]:
        issues.append(f"data_mode {data_mode!r} is not eligible for {status} review")

    seed = report.get("seed")
    if not isinstance(seed, int):
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

    primary_metric = functional_metric
    primary_key = functional_key
    primary_value = functional_value
    if status == "performance-bearing":
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
                f"declared primary performance metric {primary_metric!r} is absent from report metrics"
            )
        elif primary_value is not None and primary_value <= 0:
            issues.append(
                f"declared primary performance metric {primary_metric!r} must be positive"
            )

        protocol = report.get("measurement_protocol")
        if not isinstance(protocol, dict):
            issues.append("repeatable measurement_protocol is missing")
        else:
            if int(protocol.get("warmup_runs", 0) or 0) < 1:
                issues.append("measurement_protocol requires at least one warmup run")
            if int(protocol.get("measured_runs", 0) or 0) < 3:
                issues.append(
                    "measurement_protocol requires at least three measured runs"
                )
            if not protocol.get("latency_statistics"):
                issues.append("measurement_protocol must declare latency statistics")
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
            quality_evaluation = report.get("quality_evaluation")
            if (
                not isinstance(quality_evaluation, dict)
                or quality_evaluation.get("status") != "passed"
            ):
                issues.append(
                    "external model result lacks a passing task-quality evaluation"
                )

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
