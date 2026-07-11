#!/usr/bin/env python3
"""Generate the paper's registry and evidence tables from authoritative inputs."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import yaml

from mlperf.registry import load_registry


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REGISTRY = PROJECT / "registry"
DATASETS = PROJECT / "datasets.yaml"
EVIDENCE = HERE / "evidence_snapshot.json"
OUTPUT = HERE / "generated_registry.tex"


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
        "≥": r"$\geq$",
        "≤": r"$\leq$",
    }
    return "".join(replacements.get(char, char) for char in text)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}\\%"


def gate_for(workload) -> str:
    if workload.public_status == "score-bearing":
        labels = {
            "cross_entropy_loss": "cross-entropy loss",
            "accuracy": "accuracy",
            "anomaly_auroc": "anomaly AUROC",
            "top1_accuracy": "top-1 accuracy",
        }
        metric = labels.get(workload.quality_metric, workload.quality_metric)
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
        return "approved checkpoint; 64 decode steps; positive throughput"
    if workload.id == "slm-decode":
        return r"at least 8 tokens; continuation perplexity $\leq$ 10"
    raise ValueError(f"Unrecognized public candidate {workload.id}")


def short_status(value: str) -> str:
    replacements = {
        "Development five-seed calibration passed; clean-commit release bundle pending": "Dev 5-seed pass; release bundle pending",
        "Single-seed calibration passed; five-seed release bundle pending": "Single-seed pass; 5-seed bundle pending",
        "Checkpoint-backed functional run passed; release timing bundle pending": "Functional pass; timing bundle pending",
        "Five-seed calibration recorded; clean-commit raw bundle pending": "5-seed calibration; raw bundle pending",
        "Pinned-model quality and functional run passed; release timing bundle pending": "Quality/functional pass; timing bundle pending",
    }
    return replacements.get(value, value)


def build() -> str:
    registry = load_registry(REGISTRY)
    workloads = list(registry.values())
    evidence = json.loads(EVIDENCE.read_text())
    datasets = yaml.safe_load(DATASETS.read_text())["datasets"]

    status_counts = Counter(workload.public_status for workload in workloads)
    suite_counts = Counter(workload.suite for workload in workloads)
    public_workloads = sorted(
        (
            workload
            for workload in workloads
            if workload.public_status != "systems-only"
        ),
        key=lambda workload: (workload.public_status, workload.suite, workload.id),
    )
    invalid_training_scenarios = sorted(
        workload.id
        for workload in public_workloads
        if workload.public_status == "score-bearing" and workload.scenario != "training"
    )
    invalid_inference_scenarios = sorted(
        workload.id
        for workload in public_workloads
        if workload.public_status == "performance-bearing"
        and workload.scenario not in {"single_stream", "offline", "server"}
    )
    if invalid_training_scenarios or invalid_inference_scenarios:
        raise SystemExit(
            "public scenario mismatch; "
            f"training={invalid_training_scenarios}, inference={invalid_inference_scenarios}"
        )
    expected_ids = {workload.id for workload in public_workloads}
    recorded_ids = set(evidence["candidate_status"])
    if expected_ids != recorded_ids:
        missing = sorted(expected_ids - recorded_ids)
        stale = sorted(recorded_ids - expected_ids)
        raise SystemExit(f"candidate status mismatch; missing={missing}, stale={stale}")

    suite_rows = []
    for suite in sorted(suite_counts):
        suite_workloads = [
            workload for workload in workloads if workload.suite == suite
        ]
        public_count = sum(
            workload.public_status != "systems-only" for workload in suite_workloads
        )
        suite_rows.append(
            f"{tex(suite)} & {len(suite_workloads)} & {public_count} \\\\"
        )

    candidate_rows = []
    for workload in public_workloads:
        role = (
            "Performance"
            if workload.public_status == "performance-bearing"
            else "Score"
        )
        status = short_status(evidence["candidate_status"][workload.id])
        candidate_rows.append(
            f"\\texttt{{{tex(workload.id)}}} & {tex(workload.suite)} & {role} & "
            f"{gate_for(workload)} & {tex(status)} \\\\"
        )

    candidate_dataset_ids = sorted({workload.dataset for workload in public_workloads})
    dataset_rows = []
    for dataset_id in candidate_dataset_ids:
        if dataset_id not in datasets:
            raise SystemExit(
                f"public-candidate dataset {dataset_id!r} missing from datasets.yaml"
            )
        entry = datasets[dataset_id]
        used_by = sum(workload.dataset == dataset_id for workload in public_workloads)
        dataset_rows.append(
            f"{tex(entry['display_name'])} & {used_by} & {tex(entry['license_status'])} & "
            f"{tex(entry['public_release_status'])} \\\\"
        )

    calibration_rows = []
    calibration_ids: set[str] = set()
    for record in evidence["development_calibrations"]:
        record_id = record["workload"]
        if record_id in calibration_ids:
            raise SystemExit(f"duplicate calibration workload {record_id}")
        calibration_ids.add(record_id)
        workload = registry.get(record["workload"])
        if workload is None:
            raise SystemExit(f"unknown calibration workload {record['workload']}")
        if workload.public_status != "score-bearing":
            raise SystemExit(
                f"calibration workload is not score-bearing: {workload.id}"
            )
        if record["evidence_tier"] != "development" or record["public_eligible"]:
            raise SystemExit(
                "paper snapshot accepts development, non-public calibrations only"
            )
        if record["acceptance_passed"] is not True:
            raise SystemExit(
                f"calibration did not pass its declared gate: {workload.id}"
            )
        if record["metric"] != workload.quality_metric or float(
            record["target"]
        ) != float(workload.quality_value):
            raise SystemExit(f"calibration gate drift for {workload.id}")
        if record["seeds"] != [0, 1, 2, 3, 4] or not re.fullmatch(
            r"[0-9a-f]{64}", record["source_summary_sha256"]
        ):
            raise SystemExit(f"invalid calibration provenance for {workload.id}")
        values = [float(record[key]) for key in ("min", "median", "max", "stdev")]
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"non-finite calibration value for {workload.id}")
        minimum, median, maximum, stdev = values
        if minimum > median or median > maximum or stdev < 0:
            raise SystemExit(f"inconsistent calibration summary for {workload.id}")
        target = float(record["target"])
        passed = (
            median >= target
            if workload.quality_direction == "higher"
            else median <= target
        )
        if not passed:
            raise SystemExit(
                f"calibration median misses the registry target for {workload.id}"
            )

        registry_summary = workload.raw["quality_target"]["variance_summary"]
        for key in ("min", "median", "max"):
            if not math.isclose(
                float(record[key]),
                float(registry_summary[key]),
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise SystemExit(f"calibration {key} drift for {workload.id}")
        registry_baseline = workload.raw.get("verified_baseline", {})
        source_id = registry_baseline.get("development_summary_id")
        if source_id is not None and record["evidence_id"] != source_id:
            raise SystemExit(f"calibration evidence ID drift for {workload.id}")
        source_digest = registry_baseline.get("development_summary_sha256")
        if (
            source_digest is not None
            and record["source_summary_sha256"] != source_digest
        ):
            raise SystemExit(f"calibration evidence digest drift for {workload.id}")
        wall_median = record.get("wall_seconds_median")
        registry_wall_median = registry_baseline.get("duration_seconds_median")
        if (
            wall_median is not None
            and registry_wall_median is not None
            and not math.isclose(
                float(wall_median), float(registry_wall_median), rel_tol=0, abs_tol=1e-9
            )
        ):
            raise SystemExit(f"calibration wall-time drift for {workload.id}")
        values_as_pct = record["metric"] in {
            "accuracy",
            "anomaly_auroc",
            "top1_accuracy",
        }
        render = pct if values_as_pct else lambda value: f"{value:.4f}"
        calibration_rows.append(
            f"\\texttt{{{tex(workload.id)}}} & {tex(record['metric'])} & {render(record['target'])} & "
            f"{render(record['median'])} & [{render(record['min'])}, {render(record['max'])}] & "
            f"{tex(record['execution_backend'])} \\\\"
        )

    lines = [
        "% Generated by generate_registry_snapshot.py. Do not edit by hand.",
        f"\\newcommand{{\\PaperSnapshotDate}}{{{tex(evidence['snapshot_date'])}}}",
        f"\\newcommand{{\\RegistryRows}}{{{len(workloads)}}}",
        f"\\newcommand{{\\RegistrySuites}}{{{len(suite_counts)}}}",
        f"\\newcommand{{\\ScoreBearingRows}}{{{status_counts['score-bearing']}}}",
        f"\\newcommand{{\\PerformanceBearingRows}}{{{status_counts['performance-bearing']}}}",
        f"\\newcommand{{\\SystemsOnlyRows}}{{{status_counts['systems-only']}}}",
        f"\\newcommand{{\\DevelopmentCalibrationRows}}{{{len(calibration_rows)}}}",
        r"\newcommand{\SuiteSnapshotRows}{%",
        *suite_rows,
        "}",
        r"\newcommand{\PublicCandidateRows}{%",
        *candidate_rows,
        "}",
        r"\newcommand{\CandidateDatasetRows}{%",
        *dataset_rows,
        "}",
        r"\newcommand{\DevelopmentCalibrationTableRows}{%",
        *calibration_rows,
        "}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    OUTPUT.write_text(build())
    print(f"wrote {OUTPUT.relative_to(HERE)}")
