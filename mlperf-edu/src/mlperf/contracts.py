from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from importlib import resources
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


QUALITY_RESULT_BOUND_FIELDS = {
    "suite",
    "fixture_version",
    "suite_sha256",
    "cases",
    "categories",
    "aggregation",
    "category_guard",
}


@lru_cache(maxsize=1)
def _slm_fixture_contract() -> tuple[str, tuple[tuple[str, str, int], ...]]:
    """Load the packaged SLM case identities bound by the suite digest."""
    raw = (
        resources.files("mlperf_edu").joinpath("slm_quality_prompts.json").read_bytes()
    )
    try:
        fixture = json.loads(raw)
        cases = fixture["cases"]
        expected_tokens = fixture["expected_continuation_tokens"]
        rows = tuple(
            (
                str(case["id"]),
                str(case["category"]),
                int(expected_tokens[case["id"]]),
            )
            for case in cases
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("packaged SLM quality fixture contract is invalid") from exc
    return "sha256:" + hashlib.sha256(raw).hexdigest(), rows


def _quality_evaluation_contract_view(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, Any]:
    """Project report-owned values onto the canonical quality contract."""
    result = actual.get("result")
    result = result if isinstance(result, dict) else {}
    view: dict[str, Any] = {}
    for key, expected_value in expected.items():
        if key in QUALITY_RESULT_BOUND_FIELDS:
            view[key] = result.get(key)
        elif key == "gates" and isinstance(expected_value, dict):
            actual_gates = actual.get("gates")
            actual_gates = actual_gates if isinstance(actual_gates, dict) else {}
            view[key] = {
                gate_name: {
                    field: (actual_gates.get(gate_name) or {}).get(field)
                    for field in expected_gate
                }
                if isinstance(expected_gate, dict)
                and isinstance(actual_gates.get(gate_name), dict)
                else actual_gates.get(gate_name)
                for gate_name, expected_gate in expected_value.items()
            }
        else:
            view[key] = actual.get(key)
    return view


def _slm_quality_result_issues(
    expected: dict[str, Any], quality_evaluation: dict[str, Any]
) -> list[str]:
    """Recompute token-weighted SLM aggregates and enforce every declared gate."""
    issues: list[str] = []
    result = quality_evaluation.get("result")
    if not isinstance(result, dict):
        return ["quality evaluation result is missing"]
    case_results = result.get("case_results")
    expected_cases = expected.get("cases")
    if not isinstance(case_results, list) or len(case_results) != expected_cases:
        return [
            f"quality evaluation case count does not match canonical {expected_cases!r}"
        ]

    fixture_sha256, fixture_rows = _slm_fixture_contract()
    if expected.get("suite_sha256") != fixture_sha256:
        issues.append(
            "canonical quality suite SHA-256 does not match the packaged fixture"
        )
    if len(fixture_rows) != expected_cases:
        issues.append("packaged quality fixture case count does not match canonical")
    for index, (case, fixture_row) in enumerate(
        zip(case_results, fixture_rows, strict=False)
    ):
        if not isinstance(case, dict):
            continue
        expected_id, expected_category, expected_tokens = fixture_row
        if case.get("id") != expected_id:
            issues.append(
                f"quality evaluation case {index} id does not match the packaged fixture"
            )
        if case.get("category") != expected_category:
            issues.append(
                f"quality evaluation case {expected_id!r} category does not match the packaged fixture"
            )
        if case.get("continuation_tokens") != expected_tokens:
            issues.append(
                f"quality evaluation case {expected_id!r} token count does not match the pinned tokenizer contract"
            )

    rows: list[tuple[str, str, float, int]] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(case_results):
        if not isinstance(case, dict):
            issues.append(f"quality evaluation case result {index} is invalid")
            continue
        case_id = case.get("id")
        category = case.get("category")
        nll = case.get("mean_nll")
        tokens = case.get("continuation_tokens")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            issues.append(
                f"quality evaluation case result {index} has a missing or duplicate id"
            )
            continue
        seen_ids.add(case_id)
        if not isinstance(category, str) or not category:
            issues.append(f"quality evaluation case {case_id!r} has no category")
            continue
        if (
            isinstance(nll, bool)
            or not isinstance(nll, (int, float))
            or not math.isfinite(float(nll))
            or float(nll) < 0
        ):
            issues.append(f"quality evaluation case {case_id!r} has invalid NLL")
            continue
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            issues.append(
                f"quality evaluation case {case_id!r} has invalid token count"
            )
            continue
        rows.append((case_id, category, float(nll), tokens))
    if len(rows) != len(case_results):
        return issues

    total_tokens = sum(tokens for _, _, _, tokens in rows)
    mean_nll = math.fsum(nll * tokens for _, _, nll, tokens in rows) / total_tokens
    category_rows: dict[str, list[tuple[float, int]]] = {}
    for _, category, nll, tokens in rows:
        category_rows.setdefault(category, []).append((nll, tokens))
    if len(category_rows) != expected.get("categories"):
        issues.append(
            "quality evaluation category count does not match canonical "
            f"{expected.get('categories')!r}"
        )
    recomputed_categories: dict[str, dict[str, float | int]] = {}
    for category, values in category_rows.items():
        category_tokens = sum(tokens for _, tokens in values)
        category_nll = (
            math.fsum(nll * tokens for nll, tokens in values) / category_tokens
        )
        recomputed_categories[category] = {
            "cases": len(values),
            "continuation_tokens": category_tokens,
            "mean_nll": category_nll,
            "perplexity": math.exp(min(category_nll, 50.0)),
        }
    worst_category = max(
        recomputed_categories,
        key=lambda category: (
            float(recomputed_categories[category]["mean_nll"]),
            category,
        ),
    )
    recomputed = {
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 50.0)),
        "total_continuation_tokens": total_tokens,
        "category_results": recomputed_categories,
        "worst_category": worst_category,
        "worst_category_nll": recomputed_categories[worst_category]["mean_nll"],
        "worst_category_perplexity": recomputed_categories[worst_category][
            "perplexity"
        ],
    }
    for field, recomputed_value in recomputed.items():
        reported_value = result.get(field)
        if isinstance(recomputed_value, float):
            matches = (
                not isinstance(reported_value, bool)
                and isinstance(reported_value, (int, float))
                and math.isclose(
                    float(reported_value), recomputed_value, rel_tol=1e-9, abs_tol=1e-9
                )
            )
        else:
            matches = reported_value == recomputed_value
        if not matches:
            issues.append(
                f"quality evaluation {field} does not match token-weighted case results"
            )

    expected_gates = expected.get("gates")
    actual_gates = quality_evaluation.get("gates")
    if not isinstance(expected_gates, dict) or not isinstance(actual_gates, dict):
        return [*issues, "quality evaluation gates are missing"]
    if actual_gates.get("passed") is not True:
        issues.append("quality evaluation gate conjunction did not pass")
    for gate_name, expected_gate in expected_gates.items():
        actual_gate = actual_gates.get(gate_name)
        if not isinstance(expected_gate, dict) or not isinstance(actual_gate, dict):
            issues.append(f"quality evaluation gate {gate_name!r} is missing")
            continue
        metric_key = expected_gate.get("metric_key")
        value = actual_gate.get("value")
        if actual_gate.get("met") is not True:
            issues.append(f"quality evaluation gate {gate_name!r} did not pass")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or result.get(metric_key) != value
        ):
            issues.append(
                f"quality evaluation gate {gate_name!r} value is invalid or unbound"
            )
            continue
        target = expected_gate.get("target")
        if expected_gate.get("direction") != "lower" or not isinstance(
            target, (int, float)
        ):
            issues.append(f"canonical quality evaluation gate {gate_name!r} is invalid")
        elif float(value) > float(target):
            issues.append(
                f"quality evaluation gate {gate_name!r} value {value!r} exceeds {target!r}"
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
    if functional_value is not None and isinstance(canonical_target, (int, float)):
        if canonical_direction == "higher":
            functional_target_met = functional_value >= float(canonical_target)
        elif canonical_direction == "lower":
            functional_target_met = functional_value <= float(canonical_target)
        elif canonical_direction == "equal":
            functional_target_met = functional_value == float(canonical_target)
        else:
            functional_target_met = False
            issues.append(
                f"canonical quality direction {canonical_direction!r} is unsupported"
            )
        if not functional_target_met:
            issues.append(
                f"functional metric {functional_key!r} value {functional_value!r} "
                f"does not satisfy canonical target {canonical_direction} "
                f"{canonical_target!r}"
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
                if gate_direction == "higher":
                    gate_passed = float(gate_value) >= float(gate_target)
                elif gate_direction == "lower":
                    gate_passed = float(gate_value) <= float(gate_target)
                elif gate_direction == "equal":
                    gate_passed = float(gate_value) == float(gate_target)
                else:
                    gate_passed = False
                if not gate_passed:
                    issues.append(
                        f"report quality gate {gate_name!r} value {gate_value!r} "
                        f"does not satisfy canonical target {gate_direction} "
                        f"{gate_target!r}"
                    )
                metric_key = expected_gate.get("metric_key")
                metric_value = metrics.get(metric_key)
                if metric_key and metric_value != gate_value:
                    issues.append(
                        f"report quality gate {gate_name!r} value does not match "
                        f"metrics.{metric_key}"
                    )

    expected_quality_evaluation = canonical.get("quality_evaluation")
    if expected_quality_evaluation is not None:
        quality_evaluation = report.get("quality_evaluation")
        if isinstance(quality_evaluation, dict) and isinstance(
            expected_quality_evaluation, dict
        ):
            selected_quality_evaluation = _quality_evaluation_contract_view(
                expected_quality_evaluation, quality_evaluation
            )
        else:
            selected_quality_evaluation = quality_evaluation
        issues.extend(
            _mapping_contract_issues(
                label="quality_evaluation",
                expected=expected_quality_evaluation,
                actual=selected_quality_evaluation,
            )
        )
        if isinstance(quality_evaluation, dict) and isinstance(
            expected_quality_evaluation, dict
        ):
            issues.extend(
                _slm_quality_result_issues(
                    expected_quality_evaluation, quality_evaluation
                )
            )
            quality_result = quality_evaluation.get("result")
            metrics = report.get("metrics")
            if isinstance(quality_result, dict) and isinstance(metrics, dict):
                for metric_key, result_key in (
                    ("quality_perplexity", "perplexity"),
                    (
                        "quality_worst_category_perplexity",
                        "worst_category_perplexity",
                    ),
                    (
                        "quality_total_continuation_tokens",
                        "total_continuation_tokens",
                    ),
                ):
                    if metrics.get(metric_key) != quality_result.get(result_key):
                        issues.append(
                            f"metrics.{metric_key} does not match quality evaluation result"
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
