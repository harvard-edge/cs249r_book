#!/usr/bin/env python3
"""Generate paper tables from the registry and draft reference-result index."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from mlperf.registry import Workload, load_registry  # noqa: E402
from tools import import_reference_evidence as evidence  # noqa: E402
from tools import reference_source_lock  # noqa: E402


REGISTRY = PROJECT / "registry"
DATASETS = PROJECT / "datasets.yaml"
REFERENCE_ROOT = PROJECT / "provisional_results"
REFERENCE_INDEX = REFERENCE_ROOT / "index.json"
SNAPSHOT = HERE / "evidence_snapshot.json"
OUTPUT = HERE / "generated_registry.tex"

SNAPSHOT_SCHEMA = "mlperf-edu-paper-evidence-snapshot/0.6"
INDEX_SCHEMA = "mlperf-edu-provisional-reference-index/0.1"
RESULT_SCHEMA = "mlperf-edu-provisional-reference-result/0.1"
SOURCE_LOCK_PATH = "provisional_results/source_lock.json"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

METRIC_LABELS = {
    "accuracy": "accuracy",
    "cross_entropy_loss": "cross-entropy",
    "inference_and_evaluation_seconds": "inference + eval s",
    "inference_seconds": "inference s",
    "mean_ndcg_at_10": "mean nDCG@10",
    "output_tokens_per_sec": "output tok/s",
    "prefill_tokens_per_sec": "prefill tok/s",
    "roc_auc": "ROC AUC",
    "test_accuracy": "test accuracy",
    "test_mse": "test MSE",
    "top1_accuracy": "top-1 accuracy",
    "train_and_eval_seconds": "train + eval s",
}

GATE_METRIC_LABELS = {
    "accuracy": "accuracy",
    "cross_entropy_loss": "loss",
    "fid": "FID",
    "humaneval_plus_pass_at_1": "HumanEval+ pass@1",
    "mean_ndcg_at_10": "mean nDCG@10",
    "non_live_ast_accuracy": "AST accuracy",
    "professional_move_prediction": "move prediction",
    "roc_auc": "ROC AUC",
    "test_accuracy": "test acc.",
    "test_mse": "test MSE",
    "top1_accuracy": "top-1",
}

WORKLOAD_PAPER_LABELS = {
    "anomaly-detection": "Anomaly detection",
    "causal-language-modeling": "Causal LM",
    "code-generation": "Code generation",
    "function-calling": "Function calling",
    "graph-node-classification": "Graph classification",
    "image-classification": "Image classification",
    "image-generation": "Image generation",
    "information-retrieval": "Retrieval",
    "keyword-spotting": "KWS",
    "recommendation": "Recommendation",
    "reinforcement-learning": "Reinforcement learning",
    "text-classification": "Text classification",
    "time-series-forecasting": "Time-series forecasting",
    "visual-wake-words": "Visual wake words",
}

DATASET_PAPER_LABELS = {
    "bfcl-v4-non-live-ast": "BFCL v4 non-live AST",
    "cifar10": "CIFAR-10",
    "criteo-terabyte": "Criteo Terabyte",
    "ettm1": "ETTm1",
    "humaneval-plus": "HumanEval+",
    "mlperf-tiny-anomaly-eval": "ToyCar (MLPerf Tiny)",
    "mlperf-tiny-kws-eval": "Keyword spotting (MLPerf Tiny)",
    "mlperf-tiny-vww-eval": "Visual wake words (MLPerf Tiny)",
    "minigo-self-play": "MiniGo self-play",
    "nanobeir-reranking": "NanoBEIR",
    "ogbn-arxiv": "ogbn-arxiv",
    "prompt-suite-local": "Deterministic prompt suite",
    "sst2": "SST-2",
    "tinyshakespeare": "Tiny Shakespeare",
}

DATASET_SPLIT_LABELS = {
    "bfcl-v4-non-live-ast": "official non-live AST split",
    "cifar10": "Tiny 200-example set",
    "criteo-terabyte": "official day split",
    "ettm1": "official 12/4/4-month split",
    "humaneval-plus": "official 164-task set",
    "mlperf-tiny-anomaly-eval": "Tiny 248-recording set",
    "mlperf-tiny-kws-eval": "Tiny 1,000-example set",
    "mlperf-tiny-vww-eval": "Tiny 1,000-example set",
    "minigo-self-play": "self-play trajectories",
    "nanobeir-reranking": "three English subsets",
    "ogbn-arxiv": "official time split",
    "prompt-suite-local": "deterministic prompts",
    "sst2": "train and validation",
    "tinyshakespeare": "90/10 train and validation",
}

DATASET_ACCESS_LABELS = {
    "fetch-instructions-only": "guide",
    "needs-release-decision": "fetch; review",
    "public-ok-bundled": "bundled",
    "public-ok-fetch-only": "fetch",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def tex(value: object) -> str:
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
    return "".join(replacements.get(char, char) for char in str(value))


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def finite_number(value: object, *, label: str) -> float:
    require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def format_number(value: float, metric: str) -> str:
    if "accuracy" in metric or metric in {"mean_ndcg_at_10"}:
        return f"{100.0 * value:.2f}\\%"
    if value >= 10_000:
        return f"{value / 1000.0:.1f}k"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def format_gate(gate: dict[str, Any], *, role: str) -> str:
    if role == "performance-bearing":
        return "functional pass"
    metric = str(gate.get("metric") or "quality")
    target = finite_number(gate.get("target"), label=f"{metric} target")
    direction = gate.get("direction")
    operator = r"$\leq$" if direction == "lower" else r"$\geq$"
    return (
        f"{tex(GATE_METRIC_LABELS.get(metric, metric))} "
        f"{operator} {format_number(target, metric)}"
    )


def canonical_gate(workload: Workload) -> dict[str, Any]:
    gate = (workload.raw.get("canonical_max_contract") or {}).get("quality") or {}
    require(isinstance(gate, dict) and gate, f"{workload.id} lacks a canonical gate")
    return gate


def canonical_mode(workload: Workload) -> str:
    canonical = workload.raw.get("canonical_max_contract") or {}
    mode = str(canonical.get("mode") or "")
    phases = ((workload.raw.get("mode_contracts") or {}).get("inference") or {}).get(
        "phases"
    ) or {}
    if phases:
        return f"{mode}; inference ({', '.join(phases)})"
    return mode


def upstream_authority(workload: Workload) -> str:
    provenance = workload.raw.get("provenance") or {}
    authority = provenance.get("authority")
    return str(authority or workload.model)


def workload_rows(workloads: dict[str, Workload]) -> str:
    rows = []
    for workload in sorted(workloads.values(), key=lambda item: item.id):
        gate = canonical_gate(workload)
        role = str(
            (workload.raw.get("canonical_max_contract") or {}).get("result_role")
        )
        rows.append(
            " & ".join(
                (
                    rf"\texttt{{{tex(workload.id)}}}",
                    tex(upstream_authority(workload)),
                    tex(canonical_mode(workload)),
                    format_gate(gate, role=role),
                )
            )
            + r" \\"
        )
    return "\n".join(rows)


def dataset_rows(workloads: dict[str, Workload]) -> str:
    catalog = yaml.safe_load(DATASETS.read_text(encoding="utf-8"))["datasets"]
    usage: Counter[str] = Counter()
    for workload in workloads.values():
        names = {workload.dataset} if workload.dataset else set()
        for contract in (workload.raw.get("mode_contracts") or {}).values():
            if isinstance(contract, dict) and contract.get("dataset"):
                names.add(str(contract["dataset"]))
        usage.update(name for name in names if name)
    require(set(catalog) == set(usage), "dataset catalog and registry usage differ")
    rows = []
    for identifier, entry in sorted(catalog.items()):
        release_status = str(entry["public_release_status"])
        size_mb = finite_number(
            entry.get("estimated_size_mb"),
            label=f"{identifier} estimated size",
        )
        if size_mb == 0.0:
            size_text = (
                "bundled" if release_status == "public-ok-bundled" else "generated"
            )
        elif size_mb >= 1_000_000:
            size_text = f"{size_mb / 1_000_000:.1f} TB"
        elif size_mb >= 1_000:
            size_text = f"{size_mb / 1_000:.1f} GB"
        else:
            size_text = f"{size_mb:.1f} MB"
        rows.append(
            " & ".join(
                (
                    tex(
                        DATASET_PAPER_LABELS.get(
                            identifier,
                            entry.get("display_name") or identifier,
                        )
                    ),
                    tex(DATASET_SPLIT_LABELS.get(identifier, entry["split"])),
                    tex(size_text),
                    tex(DATASET_ACCESS_LABELS.get(release_status, release_status)),
                )
            )
            + r" \\"
        )
    return "\n".join(rows)


def safe_reference_path(value: object, *, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path must be a string")
    relative = Path(value)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe path")
    path = (PROJECT / relative).resolve()
    require(
        path.is_relative_to(REFERENCE_ROOT.resolve()),
        f"evidence path escapes provisional_results: {value}",
    )
    return path


def load_evidence() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = read_json(REFERENCE_INDEX)
    require(
        index.get("schema") == INDEX_SCHEMA,
        "reference index schema mismatch",
    )
    source_git_sha = index.get("source_git_sha")
    require(
        isinstance(source_git_sha, str) and GIT_SHA_RE.fullmatch(source_git_sha),
        "evidence source SHA is invalid",
    )

    expected = evidence.expected_cases()
    expected_workloads = {case.workload.id for case in expected.values()}
    cases = index.get("cases")
    require(isinstance(cases, list), "reference index cases must be a list")
    require(
        index.get("workload_count") == len(expected_workloads),
        "workload count drift",
    )
    require(index.get("case_count") == len(expected), "case count drift")
    require(len(cases) == len(expected), "reference case closure is incomplete")

    source_lock_entry = index.get("source_lock")
    require(isinstance(source_lock_entry, dict), "reference source lock is missing")
    require(
        source_lock_entry.get("path") == SOURCE_LOCK_PATH,
        "reference source-lock path drift",
    )
    source_lock_path = safe_reference_path(
        source_lock_entry.get("path"), label="source lock"
    )
    source_lock_bytes = source_lock_path.read_bytes()
    require(
        sha256_bytes(source_lock_bytes) == source_lock_entry.get("sha256"),
        "reference source-lock digest drift",
    )
    source_lock = reference_source_lock.load_source_lock(
        source_lock_path,
        project_root=PROJECT,
        expected_source_git_sha=source_git_sha,
        # Draft evidence is an immutable historical snapshot. Validate the
        # lock's closure without requiring today's checkout to match old code.
        verify_current=False,
    )
    for field in ("schema", "file_count", "contract_count"):
        require(
            source_lock_entry.get(field) == source_lock.get(field),
            f"reference source-lock {field} drift",
        )

    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for entry in cases:
        require(isinstance(entry, dict), "reference case entry must be an object")
        case_id = str(entry.get("case_id") or "")
        require(case_id and case_id not in seen, f"duplicate case {case_id!r}")
        expected_case = expected.get(case_id)
        require(expected_case is not None, f"unexpected reference case {case_id!r}")
        seen.add(case_id)
        path = safe_reference_path(entry.get("path"), label=f"{case_id} evidence")
        data = path.read_bytes()
        digest = sha256_bytes(data)
        require(SHA256_RE.fullmatch(digest) is not None, "invalid evidence digest")
        require(digest == entry.get("sha256"), f"{case_id} digest drift")
        payload = json.loads(data)
        require(
            payload.get("schema") == RESULT_SCHEMA,
            f"{case_id} schema drift",
        )
        require(payload.get("case_id") == case_id, f"{case_id} identity drift")
        evidence_class = payload.get("evidence_class")
        require(
            evidence_class
            in {"five-run-verified", "single-run-provisional", "two-run-provisional"},
            f"{case_id} evidence class is invalid",
        )
        measurement = payload.get("measurement") or {}
        run_count = measurement.get("run_count")
        require(run_count in {1, 2, 5}, f"{case_id} run count is invalid")
        require(
            len(measurement.get("values") or []) == run_count,
            f"{case_id} measurement count drift",
        )
        quality = payload.get("quality")
        if isinstance(quality, dict):
            require(quality.get("all_runs_pass") is True, f"{case_id} quality failed")
        repeatability = payload.get("repeatability") or {}
        if evidence_class == "five-run-verified":
            require(run_count == 5, f"{case_id} verified evidence needs five runs")
            require(
                payload.get("eligible_for_promotion") is True,
                f"{case_id} verified evidence is not promotion eligible",
            )
            require(
                repeatability.get("passed") is True,
                f"{case_id} verified repeatability failed",
            )
        else:
            require(
                payload.get("eligible_for_promotion") is False,
                f"{case_id} provisional evidence cannot be promotion eligible",
            )
            require(
                payload.get("review_eligible") is False,
                f"{case_id} provisional evidence cannot be review eligible",
            )
        expected_fields = {
            "workload": expected_case.workload.id,
            "mode": expected_case.mode,
            "phase": expected_case.phase,
            "result_role": expected_case.result_role,
        }
        for field, expected_value in expected_fields.items():
            require(
                entry.get(field) == expected_value,
                f"{case_id} index {field} drift",
            )
            require(
                payload.get(field) == expected_value,
                f"{case_id} summary {field} drift",
            )
        require(
            payload.get("profile") == expected_case.profile,
            f"{case_id} summary profile drift",
        )
        records.append({"entry": entry, "result": payload})
    require(seen == set(expected), "reference case closure differs from the registry")
    return index, sorted(records, key=lambda item: item["entry"]["case_id"])


def case_display(entry: dict[str, Any]) -> str:
    workload = str(entry["workload"])
    mode = str(entry["mode"])
    phase = entry.get("phase")
    suffix = str(phase) if phase else mode
    return f"{tex(WORKLOAD_PAPER_LABELS.get(workload, workload))} ({tex(suffix)})"


def evidence_rows(records: list[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        entry = record["entry"]
        payload = record["result"]
        role = str(entry["result_role"])
        measurement = payload["measurement"]
        metric = str(measurement["primary_metric"])
        aggregate = measurement["aggregate"]
        minimum = finite_number(aggregate["min"], label=f"{metric} min")
        median = finite_number(aggregate["median"], label=f"{metric} median")
        maximum = finite_number(aggregate["max"], label=f"{metric} max")
        run_count = int(measurement["run_count"])
        repeatability = payload["repeatability"]
        cv_value = repeatability.get("coefficient_of_variation")
        if cv_value is None:
            repeatability_text = "not established"
        else:
            cv = finite_number(cv_value, label=f"{metric} CV")
            repeatability_text = f"{100.0 * cv:.2f}\\%"
            if repeatability.get("passed") is False:
                repeatability_text += " diagnostic"
        quality = payload.get("quality")
        gate = quality.get("gate") if isinstance(quality, dict) else {}
        gate_text = format_gate(gate or {}, role=role)
        if isinstance(quality, dict):
            quality_metric = str(quality["metric"])
            quality_median = finite_number(
                quality["aggregate"]["median"],
                label=f"{quality_metric} median",
            )
            observed_text = format_number(quality_median, quality_metric)
        else:
            observed_text = "pass"
        reference = format_number(median, metric)
        if run_count > 1:
            reference += (
                f" [{format_number(minimum, metric)}, {format_number(maximum, metric)}]"
            )
        measurement_text = f"{tex(METRIC_LABELS.get(metric, metric))} {reference}"
        evidence_label = {
            "five-run-verified": "verified",
            "single-run-provisional": "provisional",
            "two-run-provisional": "provisional",
        }[str(payload["evidence_class"])]
        evidence_label += f" ({run_count})"
        devices = ", ".join(
            (payload.get("execution") or {}).get("executed_devices") or []
        )
        rows.append(
            " & ".join(
                (
                    case_display(entry),
                    "Score" if role == "score-bearing" else "Perf.",
                    evidence_label,
                    gate_text,
                    observed_text,
                    measurement_text,
                    repeatability_text,
                    tex(devices),
                )
            )
            + r" \\"
        )
    return "\n".join(rows)


def snapshot_date(records: list[dict[str, Any]]) -> str:
    timestamps = []
    for record in records:
        source_summary = record["result"].get("source_summary") or {}
        match = re.search(r"_(\d{8})T", str(source_summary.get("evidence_id") or ""))
        if match is not None:
            timestamps.append(datetime.strptime(match.group(1), "%Y%m%d"))
    require(timestamps, "reference results lack a source date")
    latest = max(timestamps)
    return f"{latest.strftime('%B')} {latest.day}, {latest.year}"


def render_tex(
    workloads: dict[str, Workload],
    index: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    promotion_workloads = {
        workload.id
        for workload in workloads.values()
        if workload.raw.get("promotion_scope", True)
    }
    roles = Counter(record["entry"]["result_role"] for record in records)
    evidence_classes = Counter(record["result"]["evidence_class"] for record in records)
    score_medians = [
        finite_number(
            record["result"]["measurement"]["aggregate"]["median"],
            label=f"{record['entry']['case_id']} median",
        )
        for record in records
        if record["entry"]["result_role"] == "score-bearing"
    ]
    verified_cvs = [
        finite_number(
            record["result"]["repeatability"]["coefficient_of_variation"],
            label=f"{record['entry']['case_id']} CV",
        )
        for record in records
        if record["result"]["evidence_class"] == "five-run-verified"
    ]
    lines = [
        "% Generated by generate_registry_snapshot.py. Do not edit by hand.",
        rf"\newcommand{{\PaperSnapshotDate}}{{{tex(snapshot_date(records))}}}",
        rf"\newcommand{{\EvidenceSourceCommit}}{{{tex(index['source_git_sha'][:10])}}}",
        rf"\newcommand{{\RegistryRows}}{{{len(workloads)}}}",
        rf"\newcommand{{\RegistrySuites}}{{{len({w.suite for w in workloads.values()})}}}",
        rf"\newcommand{{\EvidenceWorkloads}}{{{len(promotion_workloads)}}}",
        rf"\newcommand{{\FunctionalSetupWorkloads}}{{{len(workloads) - len(promotion_workloads)}}}",
        rf"\newcommand{{\ScoreBearingCases}}{{{roles['score-bearing']}}}",
        rf"\newcommand{{\PerformanceBearingCases}}{{{roles['performance-bearing']}}}",
        rf"\newcommand{{\ReferenceEvidenceCases}}{{{len(records)}}}",
        rf"\newcommand{{\FiveRunEvidenceCases}}{{{evidence_classes['five-run-verified']}}}",
        rf"\newcommand{{\ProvisionalEvidenceCases}}{{{len(records) - evidence_classes['five-run-verified']}}}",
        rf"\newcommand{{\ScoreReferenceTotalMinutes}}{{{sum(score_medians) / 60.0:.1f}}}",
        rf"\newcommand{{\ScoreReferenceMinSeconds}}{{{min(score_medians):.3f}}}",
        rf"\newcommand{{\ScoreReferenceMaxMinutes}}{{{max(score_medians) / 60.0:.1f}}}",
        rf"\newcommand{{\FiveRunMinCV}}{{{100.0 * min(verified_cvs):.2f}\%}}",
        rf"\newcommand{{\FiveRunMaxCV}}{{{100.0 * max(verified_cvs):.2f}\%}}",
        r"\newcommand{\WorkloadContractRows}{%",
        workload_rows(workloads),
        "}",
        r"\newcommand{\DatasetSnapshotRows}{%",
        dataset_rows(workloads),
        "}",
        r"\newcommand{\ReferenceEvidenceTableRows}{%",
        evidence_rows(records),
        "}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    workloads = load_registry(REGISTRY)
    index, records = load_evidence()
    promotion_workloads = {
        workload.id
        for workload in workloads.values()
        if workload.raw.get("promotion_scope", True)
    }
    require(
        promotion_workloads == {record["entry"]["workload"] for record in records},
        "paper promotion-scope workloads differ from the evidence closure",
    )
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "snapshot_date": snapshot_date(records),
        "source_git_sha": index["source_git_sha"],
        "workload_count": len(workloads),
        "evidence_workload_count": len(promotion_workloads),
        "case_count": len(records),
        "cases": [record["entry"] for record in records],
        "publication_boundary": (
            "These are project reference measurements from one disclosed laptop. "
            "Five-run records establish project repeatability; provisional records "
            "establish execution and quality only. None are MLCommons-verified results."
        ),
    }
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    OUTPUT.write_text(render_tex(workloads, index, records))
    print(
        f"generated paper snapshot for {len(workloads)} workloads and {len(records)} cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
