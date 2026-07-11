#!/usr/bin/env python3
"""Generate the paper's registry and committed-evidence snapshot tables."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mlperf.registry import load_registry


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from tools import check_taxonomy, reference_source_lock  # noqa: E402

REGISTRY = PROJECT / "registry"
DATASETS = PROJECT / "datasets.yaml"
REFERENCE_INDEX = PROJECT / "reference_results" / "index.json"
SNAPSHOT = HERE / "evidence_snapshot.json"
OUTPUT = HERE / "generated_registry.tex"

INDEX_SCHEMA = "mlperf-edu-reference-index/0.2"
SUMMARY_SCHEMA = "mlperf-edu-reference-evidence/0.3"
SNAPSHOT_SCHEMA = "mlperf-edu-paper-evidence-snapshot/0.2"
EXPECTED_SEEDS = [0, 1, 2, 3, 4]
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
EMPTY_SHA256 = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

METRIC_LABELS = {
    "accuracy": "accuracy",
    "anomaly_auroc": "anomaly AUROC",
    "cross_entropy_loss": "cross-entropy",
    "output_tokens_per_sec": "output tok/s",
    "prefill_tokens_per_sec": "prefill tok/s",
    "top1_accuracy": "top-1 accuracy",
}


def tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}\\%"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path.relative_to(PROJECT)}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path.relative_to(PROJECT)} must contain a JSON object")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def close(actual: object, expected: float, *, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} is not numeric: {actual!r}") from exc
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise SystemExit(f"{label} drift: {value!r} != {expected!r}")


def validate_summary(
    workload: Any,
    entry: dict[str, Any],
    payload: dict[str, Any],
    *,
    source_git_sha: str,
    relative_path: str,
    digest: str,
) -> dict[str, Any]:
    label = workload.id
    expected = {
        "schema": SUMMARY_SCHEMA,
        "workload": label,
        "public_status": workload.public_status,
        "evidence_tier": "public-candidate",
        "eligible_for_public_baseline": True,
        "status": "valid",
        "profile": "max",
    }
    for key, value in expected.items():
        require(
            payload.get(key) == value,
            f"{label}: summary {key}={payload.get(key)!r}, expected {value!r}",
        )
    require(not payload.get("invalid_reasons"), f"{label}: summary has invalid reasons")
    require(
        (payload.get("acceptance") or {}).get("passed") is True,
        f"{label}: summary acceptance did not pass",
    )
    require(
        payload.get("seeds_requested") == EXPECTED_SEEDS,
        f"{label}: summary must request seeds {EXPECTED_SEEDS}",
    )
    require(
        ((payload.get("basis") or {}).get("reference_protocol") or {}).get("seeds")
        == EXPECTED_SEEDS,
        f"{label}: summary protocol seeds drift",
    )

    source = payload.get("source") or {}
    require(source.get("git_dirty") is False, f"{label}: source is dirty")
    require(
        source.get("git_status_sha256") == EMPTY_SHA256
        and source.get("git_patch_sha256") == EMPTY_SHA256,
        f"{label}: clean-source status or patch digest is not empty",
    )
    require(
        source.get("git_sha") == source_git_sha,
        f"{label}: source Git SHA does not match the reference index",
    )
    require(
        re.fullmatch(r"sha256:[0-9a-f]{64}", str(source.get("tool_sha256") or ""))
        is not None,
        f"{label}: source tool digest is missing",
    )

    expected_variant = workload.variant
    require(
        payload.get("variant") == expected_variant,
        f"{label}: summary variant {payload.get('variant')!r} != {expected_variant!r}",
    )
    metric = payload.get("quality_metric")
    if workload.public_status == "score-bearing":
        require(metric == workload.quality_metric, f"{label}: quality metric drift")
        acceptance = payload.get("acceptance") or {}
        close(
            acceptance.get("target"),
            float(workload.quality_value),
            label=f"{label} target",
        )
        expected_operator = "<=" if workload.quality_direction == "lower" else ">="
        require(
            acceptance.get("operator") == expected_operator,
            f"{label}: acceptance operator drift",
        )
    else:
        expected_metric = (workload.raw.get("measurement_protocol") or {}).get(
            "primary_metric"
        )
        require(metric == expected_metric, f"{label}: performance metric drift")
        acceptance = payload.get("acceptance") or {}
        require(
            acceptance.get("statistic") == "all_runs"
            and acceptance.get("value") == len(EXPECTED_SEEDS)
            and acceptance.get("target") == len(EXPECTED_SEEDS),
            f"{label}: performance acceptance must record five passing runs",
        )

    runs = payload.get("runs")
    require(
        isinstance(runs, list) and len(runs) == len(EXPECTED_SEEDS),
        f"{label}: summary must index five runs",
    )
    values: list[float] = []
    wall_values: list[float] = []
    backends: set[str] = set()
    chips: set[str] = set()
    data_modes: set[str] = set()
    for seed, run in zip(EXPECTED_SEEDS, runs, strict=True):
        require(isinstance(run, dict), f"{label}: run {seed} is not an object")
        require(run.get("requested_seed") == seed, f"{label}: run seed order drift")
        for field in (
            "execution_ok",
            "evidence_valid",
            "seed_match",
            "manifest_verified",
            "quality_target_met",
        ):
            require(run.get(field) is True, f"{label}: seed {seed} failed {field}")
        require(run.get("timed_out") is False, f"{label}: seed {seed} timed out")
        require(
            not run.get("invalid_reasons"), f"{label}: seed {seed} has invalid reasons"
        )
        require(
            (run.get("grade") or {}).get("passed") is True,
            f"{label}: seed {seed} failed grading",
        )
        artifacts = run.get("artifacts")
        require(
            isinstance(artifacts, list) and artifacts,
            f"{label}: seed {seed} has no artifact index",
        )
        roles = {
            artifact.get("role") for artifact in artifacts if isinstance(artifact, dict)
        }
        require(
            {"report", "provenance"}.issubset(roles),
            f"{label}: seed {seed} lacks report or provenance evidence",
        )
        value = float(run["quality_value"])
        wall_value = float(run["wall_seconds"])
        require(
            math.isfinite(value) and math.isfinite(wall_value),
            f"{label}: seed {seed} contains a non-finite metric or wall time",
        )
        values.append(value)
        wall_values.append(wall_value)
        backends.add(str(run.get("backend") or "unknown"))
        chips.add(str(run.get("chip") or "unknown"))
        data_modes.add(str(run.get("data_mode") or "unknown"))

    aggregate = (payload.get("aggregate") or {}).get("quality") or {}
    require(aggregate.get("count") == 5, f"{label}: aggregate count is not five")
    expected_stats = {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values),
    }
    for key, value in expected_stats.items():
        close(aggregate.get(key), value, label=f"{label} aggregate {key}")

    wall = (payload.get("aggregate") or {}).get("wall_seconds") or {}
    require(wall.get("count") == 5, f"{label}: wall aggregate count is not five")
    expected_wall_stats = {
        "min": min(wall_values),
        "max": max(wall_values),
        "mean": statistics.mean(wall_values),
        "median": statistics.median(wall_values),
        "stdev": statistics.stdev(wall_values),
    }
    for key, value in expected_wall_stats.items():
        close(wall.get(key), value, label=f"{label} wall aggregate {key}")
    require(
        len(backends) == len(chips) == len(data_modes) == 1
        and "unknown" not in backends | chips | data_modes,
        f"{label}: backend, chip, and data mode must each be stable across seeds",
    )

    require(
        entry.get("evidence_id") == payload.get("evidence_id"),
        f"{label}: evidence ID drift",
    )
    require(entry.get("path") == relative_path, f"{label}: evidence path drift")
    require(entry.get("evidence_sha256") == digest, f"{label}: evidence digest drift")
    require(
        entry.get("public_status") == workload.public_status,
        f"{label}: index status drift",
    )
    require(entry.get("seeds") == EXPECTED_SEEDS, f"{label}: index seed drift")
    require(entry.get("metric") == metric, f"{label}: index metric drift")
    require(entry.get("profile") == "max", f"{label}: index profile drift")
    require(entry.get("variant") == expected_variant, f"{label}: index variant drift")
    require(
        entry.get("acceptance") == payload.get("acceptance"),
        f"{label}: index acceptance drift",
    )
    require(
        entry.get("aggregate") == payload.get("aggregate"),
        f"{label}: index aggregate drift",
    )

    baseline = workload.raw.get("verified_baseline") or {}
    require(
        baseline.get("evidence_status") == "committed-reference-summary"
        and baseline.get("review_eligible") is True,
        f"{label}: registry baseline is not committed and review eligible",
    )
    require(
        baseline.get("evidence_file") == relative_path
        and baseline.get("evidence_sha256") == digest,
        f"{label}: registry baseline does not cite the exact committed summary",
    )
    require(
        baseline.get("source_git_sha") == source_git_sha,
        f"{label}: registry source Git SHA drift",
    )
    availability = baseline.get("reference_package_availability")
    publication = baseline.get("external_publication_status")
    require(
        availability in {"local-handoff", "published"},
        f"{label}: reference package availability is invalid",
    )
    require(
        publication == ("published" if availability == "published" else "pending"),
        f"{label}: external publication status is inconsistent",
    )

    return {
        "acceptance": payload["acceptance"],
        "acceptance_passed": True,
        "aggregate": {
            key: float(aggregate[key])
            for key in ("min", "median", "max", "mean", "stdev")
        },
        "data_modes": sorted(data_modes),
        "evidence_file": relative_path,
        "evidence_id": payload["evidence_id"],
        "evidence_tier": "public-candidate",
        "execution_backends": sorted(backends),
        "external_publication_status": publication,
        "hardware_hosts": sorted(chips),
        "metric": metric,
        "profile": payload["profile"],
        "public_eligible": True,
        "public_status": workload.public_status,
        "reference_package_availability": availability,
        "seeds": EXPECTED_SEEDS,
        "source_git_sha": source_git_sha,
        "source_summary_sha256": digest,
        "variant": payload.get("variant"),
        "wall_seconds": {
            key: float(wall[key]) for key in ("min", "median", "max", "mean", "stdev")
        },
        "workload": label,
    }


def load_reference_records(
    registry: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = read_json(REFERENCE_INDEX)
    require(index.get("schema") == INDEX_SCHEMA, "reference index schema mismatch")
    source_git_sha = str(index.get("source_git_sha") or "")
    require(
        GIT_SHA_RE.fullmatch(source_git_sha) is not None,
        "reference index Git SHA is invalid",
    )
    entries = index.get("summaries")
    require(isinstance(entries, list), "reference index summaries must be a list")
    require(index.get("summary_count") == len(entries), "reference index count drift")
    source_lock_record = index.get("source_lock") or {}
    source_lock_path = PROJECT / str(source_lock_record.get("path") or "")
    try:
        source_lock_bytes = source_lock_path.read_bytes()
        source_lock = reference_source_lock.load_source_lock(
            source_lock_path,
            project_root=PROJECT,
            expected_source_git_sha=source_git_sha,
        )
    except (OSError, reference_source_lock.SourceLockError) as exc:
        raise SystemExit(f"reference source lock is invalid: {exc}") from exc
    require(
        reference_source_lock.sha256_bytes(source_lock_bytes)
        == source_lock_record.get("sha256"),
        "reference source-lock digest mismatch",
    )
    for field in ("schema", "file_count", "contract_count"):
        require(
            source_lock_record.get(field) == source_lock.get(field),
            f"reference source-lock {field} drift",
        )

    public_workloads = {
        workload.id: workload
        for workload in registry.values()
        if workload.public_status in {"score-bearing", "performance-bearing"}
    }
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        require(isinstance(entry, dict), "reference index entry is not an object")
        workload_id = str(entry.get("workload") or "")
        require(
            workload_id in public_workloads,
            f"unexpected reference workload {workload_id!r}",
        )
        require(workload_id not in by_id, f"duplicate reference workload {workload_id}")
        by_id[workload_id] = entry
    require(
        set(by_id) == set(public_workloads),
        "reference index candidate set drift; "
        f"missing={sorted(set(public_workloads) - set(by_id))}, "
        f"stale={sorted(set(by_id) - set(public_workloads))}",
    )

    records: list[dict[str, Any]] = []
    reference_root = (PROJECT / "reference_results").resolve()
    for workload_id in sorted(public_workloads):
        workload = public_workloads[workload_id]
        entry = by_id[workload_id]
        relative_path = str(entry.get("path") or "")
        candidate_path = (PROJECT / relative_path).resolve()
        require(
            relative_path
            and not Path(relative_path).is_absolute()
            and is_within(candidate_path, reference_root),
            f"{workload_id}: unsafe reference summary path {relative_path!r}",
        )
        try:
            data = candidate_path.read_bytes()
        except OSError as exc:
            raise SystemExit(
                f"{workload_id}: cannot read {relative_path}: {exc}"
            ) from exc
        digest = sha256_bytes(data)
        require(
            SHA256_RE.fullmatch(digest) is not None, f"{workload_id}: invalid digest"
        )
        require(
            digest == entry.get("evidence_sha256"),
            f"{workload_id}: summary SHA-256 does not match reference index",
        )
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{workload_id}: invalid summary JSON: {exc}") from exc
        require(
            isinstance(payload, dict), f"{workload_id}: summary root is not an object"
        )
        central_errors = check_taxonomy.check_reference_summary(
            f"{workload.suite}/{workload.id}", workload.raw, payload
        )
        require(
            not central_errors,
            f"{workload_id}: central evidence validation failed: {central_errors}",
        )
        records.append(
            validate_summary(
                workload,
                entry,
                payload,
                source_git_sha=source_git_sha,
                relative_path=relative_path,
                digest=digest,
            )
        )

    require(
        len(records) == 8,
        f"expected 8 committed reference summaries, found {len(records)}",
    )
    shared_errors = check_taxonomy.check_shared_checkpoint_evidence(
        {workload.id: workload.raw for workload in registry.values()}
    )
    require(
        not shared_errors,
        f"shared-checkpoint evidence validation failed: {shared_errors}",
    )
    return index, records


def snapshot_date(records: list[dict[str, Any]]) -> str:
    dates: list[datetime] = []
    for record in records:
        payload = read_json(PROJECT / record["evidence_file"])
        value = str(payload.get("finished_at") or payload.get("generated_at") or "")
        try:
            dates.append(datetime.fromisoformat(value))
        except ValueError as exc:
            raise SystemExit(
                f"{record['workload']}: invalid evidence timestamp {value!r}"
            ) from exc
    latest = max(dates)
    return f"{latest.strftime('%B')} {latest.day}, {latest.year}"


def build_snapshot(
    index: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "snapshot_date": snapshot_date(records),
        "source_git_sha": index["source_git_sha"],
        "summary_count": len(records),
        "publication_boundary": (
            "Committed summaries are project references, not MLCommons-verified results. "
            "Raw packages are retained for local reviewer handoff and do not yet have public URLs."
        ),
        "reference_summaries": records,
    }


def gate_for(workload: Any) -> str:
    if workload.public_status == "score-bearing":
        metric = METRIC_LABELS.get(workload.quality_metric, workload.quality_metric)
        operator = r"$\leq$" if workload.quality_direction == "lower" else r"$\geq$"
        value = float(workload.quality_value)
        formatted = (
            pct(value)
            if workload.quality_metric in {"accuracy", "anomaly_auroc", "top1_accuracy"}
            else f"{value:g}"
        )
        return f"{tex(metric)} {operator} {formatted}"
    if workload.id == "nanogpt-prefill":
        return "approved checkpoint; positive prefill throughput"
    if workload.id == "nanogpt-decode":
        steps = int(
            (workload.raw.get("measurement_protocol") or {}).get(
                "decode_steps_per_request"
            )
        )
        return f"approved checkpoint; {steps} decode steps; positive throughput"
    if workload.id == "slm-decode":
        token_floor = int((workload.raw.get("quality_target") or {}).get("value"))
        perplexity = float(
            (workload.raw.get("quality_evaluation") or {}).get("maximum")
        )
        return (
            f"at least {token_floor} tokens; continuation perplexity "
            rf"$\leq$ {perplexity:g}"
        )
    raise ValueError(f"Unrecognized public candidate {workload.id}")


def format_metric_value(metric: str, value: float) -> str:
    if metric in {"accuracy", "anomaly_auroc", "top1_accuracy"}:
        return pct(value)
    if metric == "cross_entropy_loss":
        return f"{value:.4f}"
    if metric == "prefill_tokens_per_sec":
        return f"{value / 1000.0:.1f}k"
    return f"{value:.2f}"


def acceptance_label(record: dict[str, Any]) -> str:
    if record["public_status"] == "performance-bearing":
        return "5/5 gates"
    acceptance = record["acceptance"]
    operator = r"$\leq$" if acceptance["operator"] == "<=" else r"$\geq$"
    return f"{operator} {format_metric_value(record['metric'], float(acceptance['target']))}"


def backend_label(record: dict[str, Any]) -> str:
    labels = {
        "pytorch-cpu": "PyTorch CPU",
        "pytorch-mps": "PyTorch MPS",
        "transformers-mps": "Transformers MPS",
    }
    values = [labels.get(value, value) for value in record["execution_backends"]]
    return ", ".join(values)


def build_tex(
    registry: dict[str, Any], datasets: dict[str, Any], snapshot: dict[str, Any]
) -> str:
    workloads = list(registry.values())
    records = {record["workload"]: record for record in snapshot["reference_summaries"]}
    status_counts = Counter(workload.public_status for workload in workloads)
    suite_counts = Counter(workload.suite for workload in workloads)
    public_workloads = sorted(
        (
            workload
            for workload in workloads
            if workload.public_status in {"score-bearing", "performance-bearing"}
        ),
        key=lambda workload: (workload.public_status, workload.suite, workload.id),
    )

    invalid_training = sorted(
        workload.id
        for workload in public_workloads
        if workload.public_status == "score-bearing" and workload.scenario != "training"
    )
    invalid_inference = sorted(
        workload.id
        for workload in public_workloads
        if workload.public_status == "performance-bearing"
        and workload.scenario not in {"single_stream", "offline", "server"}
    )
    require(
        not invalid_training and not invalid_inference,
        f"public scenario mismatch; training={invalid_training}, inference={invalid_inference}",
    )

    suite_rows = []
    for suite in sorted(suite_counts):
        suite_workloads = [
            workload for workload in workloads if workload.suite == suite
        ]
        public_count = sum(
            workload.public_status in {"score-bearing", "performance-bearing"}
            for workload in suite_workloads
        )
        suite_rows.append(
            f"{tex(suite)} & {len(suite_workloads)} & {public_count} \\\\"
        )

    candidate_rows = []
    for workload in public_workloads:
        record = records[workload.id]
        role = (
            "Performance"
            if workload.public_status == "performance-bearing"
            else "Score"
        )
        status = (
            "Committed 5-run functional pass; raw package local"
            if workload.public_status == "performance-bearing"
            else "Committed 5-seed pass; raw package local"
        )
        candidate_rows.append(
            f"\\texttt{{{tex(workload.id)}}} & {tex(workload.suite)} & {role} & "
            f"{gate_for(workload)} & {tex(status)} \\\\"
        )

    candidate_dataset_ids = sorted({workload.dataset for workload in public_workloads})
    dataset_rows = []
    for dataset_id in candidate_dataset_ids:
        require(
            dataset_id in datasets,
            f"public-candidate dataset {dataset_id!r} is missing",
        )
        entry = datasets[dataset_id]
        used_by = sum(workload.dataset == dataset_id for workload in public_workloads)
        dataset_rows.append(
            f"{tex(entry['display_name'])} & {used_by} & {tex(entry['license_status'])} & "
            f"{tex(entry['public_release_status'])} \\\\"
        )

    evidence_rows = []
    for workload in public_workloads:
        record = records[workload.id]
        aggregate = record["aggregate"]
        role = "Perf." if workload.public_status == "performance-bearing" else "Score"
        metric = METRIC_LABELS.get(record["metric"], record["metric"])
        evidence_rows.append(
            f"\\texttt{{{tex(workload.id)}}} & {role} & {tex(metric)} & "
            f"{acceptance_label(record)} & "
            f"{format_metric_value(record['metric'], aggregate['median'])} & "
            f"[{format_metric_value(record['metric'], aggregate['min'])}, "
            f"{format_metric_value(record['metric'], aggregate['max'])}] & "
            f"{tex(backend_label(record))} \\\\"
        )

    source_short = snapshot["source_git_sha"][:8]

    def margin_points(workload_id: str) -> str:
        workload = registry[workload_id]
        margin = records[workload_id]["aggregate"]["min"] - float(
            workload.quality_value
        )
        return f"{100.0 * margin:.2f}"

    prefill_protocol = registry["nanogpt-prefill"].raw["measurement_protocol"]
    decode_protocol = registry["nanogpt-decode"].raw["measurement_protocol"]
    slm_protocol = registry["slm-decode"].raw["measurement_protocol"]
    slm_quality = registry["slm-decode"].raw["quality_evaluation"]
    lines = [
        "% Generated by generate_registry_snapshot.py. Do not edit by hand.",
        f"\\newcommand{{\\PaperSnapshotDate}}{{{tex(snapshot['snapshot_date'])}}}",
        f"\\newcommand{{\\EvidenceSourceCommit}}{{{source_short}}}",
        f"\\newcommand{{\\RegistryRows}}{{{len(workloads)}}}",
        f"\\newcommand{{\\RegistrySuites}}{{{len(suite_counts)}}}",
        f"\\newcommand{{\\ScoreBearingRows}}{{{status_counts['score-bearing']}}}",
        f"\\newcommand{{\\PerformanceBearingRows}}{{{status_counts['performance-bearing']}}}",
        f"\\newcommand{{\\SystemsOnlyRows}}{{{status_counts['systems-only']}}}",
        f"\\newcommand{{\\ReferenceEvidenceRows}}{{{len(evidence_rows)}}}",
        f"\\newcommand{{\\AnomalyMinMarginPoints}}{{{margin_points('anomaly-ae-train')}}}",
        f"\\newcommand{{\\MobileNetMinMarginPoints}}{{{margin_points('mobilenetv2-train')}}}",
        f"\\newcommand{{\\ResNetMinMarginPoints}}{{{margin_points('resnet18-train')}}}",
        f"\\newcommand{{\\DLRMMinMarginPoints}}{{{margin_points('micro-dlrm-train')}}}",
        f"\\newcommand{{\\NanoGPTHighLoss}}{{{records['nanogpt-train']['aggregate']['max']:.3f}}}",
        f"\\newcommand{{\\NanoGPTPrefillMedianK}}{{{records['nanogpt-prefill']['aggregate']['median'] / 1000.0:.1f}}}",
        f"\\newcommand{{\\NanoGPTDecodeMedian}}{{{records['nanogpt-decode']['aggregate']['median']:.2f}}}",
        f"\\newcommand{{\\SLMDecodeMedian}}{{{records['slm-decode']['aggregate']['median']:.2f}}}",
        f"\\newcommand{{\\SLMDecodeMin}}{{{records['slm-decode']['aggregate']['min']:.2f}}}",
        f"\\newcommand{{\\SLMDecodeMax}}{{{records['slm-decode']['aggregate']['max']:.2f}}}",
        f"\\newcommand{{\\PrefillWarmups}}{{{int(prefill_protocol['warmup_runs'])}}}",
        f"\\newcommand{{\\PrefillMeasurements}}{{{int(prefill_protocol['measured_runs'])}}}",
        f"\\newcommand{{\\DecodeWarmups}}{{{int(decode_protocol['warmup_runs'])}}}",
        f"\\newcommand{{\\DecodeMeasurements}}{{{int(decode_protocol['measured_runs'])}}}",
        f"\\newcommand{{\\DecodeSteps}}{{{int(decode_protocol['decode_steps_per_request'])}}}",
        f"\\newcommand{{\\SLMWarmups}}{{{int(slm_protocol['warmup_runs'])}}}",
        f"\\newcommand{{\\SLMMeasurements}}{{{int(slm_protocol['measured_runs'])}}}",
        f"\\newcommand{{\\SLMTokenFloor}}{{{int(registry['slm-decode'].quality_value)}}}",
        f"\\newcommand{{\\SLMPerplexityLimit}}{{{float(slm_quality['maximum']):g}}}",
        r"\newcommand{\SuiteSnapshotRows}{%",
        *suite_rows,
        "}",
        r"\newcommand{\PublicCandidateRows}{%",
        *candidate_rows,
        "}",
        r"\newcommand{\CandidateDatasetRows}{%",
        *dataset_rows,
        "}",
        r"\newcommand{\ReferenceEvidenceTableRows}{%",
        *evidence_rows,
        "}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    registry = load_registry(REGISTRY)
    datasets = yaml.safe_load(DATASETS.read_text(encoding="utf-8"))["datasets"]
    index, records = load_reference_records(registry)
    snapshot = build_snapshot(index, records)
    SNAPSHOT.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    OUTPUT.write_text(build_tex(registry, datasets, snapshot), encoding="utf-8")
    print(
        f"wrote {SNAPSHOT.name} and {OUTPUT.name} from "
        f"{len(records)} committed summaries at {index['source_git_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
