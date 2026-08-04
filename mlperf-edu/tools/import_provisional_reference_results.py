#!/usr/bin/env python3
"""Import a mixed verified/provisional MLPerf EDU reference snapshot.

The canonical promotion importer remains intentionally strict and accepts only
complete repeated-timing evidence. This companion importer supports a v0.1 draft
snapshot without weakening that contract. It records promotion-ready cases as
repeated-timing and development cases as single-measurement references.
Neither record class is exposed as an MLCommons-verified result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mlperf.edu_cli import verify_package_archive  # noqa: E402
from tools import import_reference_evidence as promotion  # noqa: E402
from tools import reference_source_lock  # noqa: E402


INDEX_SCHEMA = "mlperf-edu-provisional-reference-index/0.1"
RECORD_SCHEMA = "mlperf-edu-provisional-reference-result/0.1"
SOURCE_LOCK_PATH = "provisional_results/source_lock.json"
OUTPUT_ROOTS = (
    ROOT / "provisional_results",
    ROOT / "src" / "mlperf_edu" / "provisional_results",
)
CAUSAL_TRAINING_CASE = "causal-language-modeling__max__training"


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _artifact_index(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("evidence run has no artifact index")
    result: list[dict[str, Any]] = []
    roles: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("evidence artifact claim must be an object")
        role = artifact.get("role")
        digest = artifact.get("sha256")
        size = artifact.get("n_bytes")
        if not isinstance(role, str) or not role or role in roles:
            raise ValueError("evidence artifact role is missing or duplicated")
        if not isinstance(digest, str) or not promotion.PREFIXED_SHA256_RE.fullmatch(
            digest
        ):
            raise ValueError(f"evidence artifact {role} has an invalid digest")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"evidence artifact {role} has an invalid size")
        roles.add(role)
        result.append({"role": role, "sha256": digest, "n_bytes": size})
    return sorted(result, key=lambda item: item["role"])


def _summary_case(
    payload: Mapping[str, Any],
    cases: Mapping[str, promotion.EvidenceCase],
) -> promotion.EvidenceCase:
    workload = payload.get("workload")
    profile = payload.get("profile")
    mode = payload.get("mode")
    phase = payload.get("phase")
    if not all(isinstance(value, str) for value in (workload, profile, mode)):
        raise ValueError("evidence summary case identity is incomplete")
    identifier = promotion.case_id(
        str(workload), str(profile), str(mode), str(phase) if phase else None
    )
    if identifier not in cases:
        raise ValueError(f"unexpected evidence case {identifier}")
    return cases[identifier]


def validate_development_summary(
    path: Path,
    payload: Mapping[str, Any],
    *,
    case: promotion.EvidenceCase,
    source_git_sha: str,
    sweep_tool_sha256: str,
) -> None:
    expected = {
        "schema": promotion.SUMMARY_SCHEMA,
        "status": "valid",
        "evidence_tier": "development",
        "eligible_for_promotion": False,
        "eligible_for_public_baseline": False,
        "workload": case.workload.id,
        "canonical_workload": case.workload.id,
        "profile": case.profile,
        "mode": case.mode,
        "phase": case.phase,
        "result_role": case.result_role,
        "invalid_reasons": [],
        "seeds_requested": [case.canonical_seed],
    }
    failures = [
        f"{field}={payload.get(field)!r}, expected {value!r}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]
    source = payload.get("source") or {}
    for field, value in {
        "git_sha": source_git_sha,
        "git_dirty": False,
        "git_status_sha256": promotion.EMPTY_SHA256,
        "git_patch_sha256": promotion.EMPTY_SHA256,
        "tool_sha256": sweep_tool_sha256,
    }.items():
        if source.get(field) != value:
            failures.append(f"source.{field} does not match the frozen source")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        failures.append("development summary must contain exactly one run")
    else:
        run = runs[0]
        for field, value in {
            "execution_index": 1,
            "requested_seed": case.canonical_seed,
            "status": "passed",
            "execution_ok": True,
            "evidence_valid": True,
            "quality_target_met": True,
            "invalid_reasons": [],
        }.items():
            if run.get(field) != value:
                failures.append(f"run.{field}={run.get(field)!r}, expected {value!r}")
        host_power = run.get("host_power") or {}
        before = host_power.get("before") or {}
        after = host_power.get("after") or {}
        if host_power.get("stable") is not True:
            failures.append("run host power was not stable")
        if (
            before.get("source_raw") != "AC Power"
            or after.get("source_raw") != "AC Power"
        ):
            failures.append("run did not remain on AC power")
        if (
            before.get("low_power_mode") is not False
            or after.get("low_power_mode") is not False
        ):
            failures.append("run used Low Power Mode")
    if failures:
        raise ValueError(f"{path}: " + "; ".join(failures))


def build_summary_record(
    case: promotion.EvidenceCase,
    payload: Mapping[str, Any],
    data: bytes,
    *,
    evidence_class: str,
) -> dict[str, Any]:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{case.case_id}: summary has no runs")
    # Retained records carry the evidence-class strings from the era when the
    # protocol demanded five timing runs. The count is taken from the record
    # itself so those records keep validating, without the importer asserting a
    # run count the protocol no longer requires.
    minimum = 2 if evidence_class.startswith(("two-run", "repeated")) else 1
    if len(runs) < minimum:
        raise ValueError(
            f"{case.case_id}: {evidence_class} needs at least {minimum} run(s)"
        )
    primary_values = [
        _finite(run.get("primary_metric_value"), label="primary metric") for run in runs
    ]
    wall_values = [
        _finite(run.get("wall_seconds"), label="wall seconds") for run in runs
    ]
    result: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "case_id": case.case_id,
        "workload": case.workload.id,
        "profile": case.profile,
        "mode": case.mode,
        "phase": case.phase,
        "result_role": case.result_role,
        "evidence_class": evidence_class,
        "review_eligible": evidence_class == "five-run-verified",
        "eligible_for_promotion": payload.get("eligible_for_promotion") is True,
        "eligible_for_public_baseline": False,
        "source_git_sha": (payload.get("source") or {}).get("git_sha"),
        "source_summary": {
            "schema": payload.get("schema"),
            "evidence_id": payload.get("evidence_id"),
            "sha256": promotion.sha256_bytes(data),
        },
        "execution": {
            "requested_device": payload.get("device_requested"),
            "executed_devices": sorted(
                {str(run.get("device_executed")) for run in runs}
            ),
            "backends": sorted({str(run.get("backend")) for run in runs}),
            "hardware_chips": sorted({str(run.get("chip")) for run in runs}),
            "data_modes": sorted({str(run.get("data_mode")) for run in runs}),
            "canonical_seed": payload.get("canonical_seed"),
        },
        "measurement": {
            "run_count": len(runs),
            "primary_metric": (payload.get("primary_metric") or {}).get("name"),
            "values": primary_values,
            "aggregate": promotion.aggregate(primary_values),
            "wall_seconds": wall_values,
            "wall_aggregate": promotion.aggregate(wall_values),
        },
        "artifacts": [
            {"execution_index": index, "files": _artifact_index(run)}
            for index, run in enumerate(runs, start=1)
        ],
        "comparison_fingerprints": [
            str(run.get("comparison_fingerprint_sha256")) for run in runs
        ],
    }
    if case.result_role == "score-bearing":
        quality_values = [
            _finite(run.get("quality_value"), label="quality value") for run in runs
        ]
        result["quality"] = {
            "metric": payload.get("quality_metric"),
            "gate": payload.get("quality_gate"),
            "values": quality_values,
            "aggregate": promotion.aggregate(quality_values),
            "all_runs_pass": all(run.get("quality_target_met") is True for run in runs),
        }
    else:
        result["functional"] = {
            "gate": payload.get("functional_gate"),
            "all_runs_pass": all(run.get("quality_target_met") is True for run in runs),
        }
        lineage = payload.get("nanogpt_training_lineage")
        if isinstance(lineage, dict):
            result["source_training"] = {
                field: lineage.get(field)
                for field in (
                    "package_sha256",
                    "checkpoint_sha256",
                    "training_report_sha256",
                    "training_manifest_sha256",
                    "training_quality_metric",
                    "training_quality_value",
                    "training_quality_target",
                    "training_quality_passed",
                )
            }
    declared_repeatability = payload.get("primary_metric_repeatability") or {}
    if evidence_class == "five-run-verified":
        result["repeatability"] = {
            "observed": True,
            "metric": declared_repeatability.get("metric"),
            "coefficient_of_variation": declared_repeatability.get(
                "coefficient_of_variation"
            ),
            "limit": declared_repeatability.get("limit"),
            "passed": declared_repeatability.get("passed"),
        }
    else:
        result["repeatability"] = {
            "observed": False,
            "metric": declared_repeatability.get("metric"),
            "coefficient_of_variation": None,
            "limit": declared_repeatability.get("limit"),
            "passed": None,
            "note": "One execution cannot establish timing repeatability.",
        }
    return result


def _verify_causal_training_run(
    attempt_root: Path,
    execution_index: int,
    *,
    case: promotion.EvidenceCase,
    source_project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    run_root = attempt_root / f"run_{execution_index:03d}"
    report_path = run_root / "causal-language-modeling_training_max_report.json"
    manifest_path = run_root / "causal-language-modeling_training_max.provd.json"
    report, _ = promotion.load_json_object(report_path, label=str(report_path))
    manifest, _ = promotion.load_json_object(manifest_path, label=str(manifest_path))
    for field, value in {
        "workload": case.workload.id,
        "profile": case.profile,
        "mode": case.mode,
        "phase": case.phase,
        "status": "passed",
        "seed": case.canonical_seed,
        "data_mode": case.data_mode,
    }.items():
        if report.get(field) != value:
            raise ValueError(
                f"{report_path}: {field}={report.get(field)!r}, expected {value!r}"
            )
    contract = report.get("promotion_contract") or {}
    if (
        contract.get("status") != "passed"
        or contract.get("promotion_eligible") is not True
    ):
        raise ValueError(f"{report_path}: promotion contract did not pass")
    verification = promotion.verify_provd(manifest_path, repo_root=source_project_root)
    failures = [name for name, passed, _detail in verification.checks if not passed]
    if failures:
        raise ValueError(f"{manifest_path}: provenance verification failed: {failures}")
    return report, manifest, report_path, manifest_path


def build_causal_training_record(
    attempt_root: Path,
    package_path: Path,
    *,
    case: promotion.EvidenceCase,
    source_git_sha: str,
    source_project_root: Path,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    report_paths: list[Path] = []
    manifest_paths: list[Path] = []
    for execution_index in (1, 2):
        report, manifest, report_path, manifest_path = _verify_causal_training_run(
            attempt_root,
            execution_index,
            case=case,
            source_project_root=source_project_root,
        )
        reports.append(report)
        manifests.append(manifest)
        report_paths.append(report_path)
        manifest_paths.append(manifest_path)
    package_checks = verify_package_archive(package_path, repo_root=source_project_root)
    package_failures = [
        (name, detail) for name, passed, detail in package_checks if not passed
    ]
    if package_failures:
        raise ValueError(
            f"{package_path}: package verification failed: {package_failures}"
        )
    timing_values = [
        _finite(report["metrics"].get("train_and_eval_seconds"), label="training time")
        for report in reports
    ]
    quality_values = [
        _finite(report["metrics"].get("cross_entropy_loss"), label="training loss")
        for report in reports
    ]
    timing_stats = promotion.aggregate(timing_values)
    coefficient = float(timing_stats["stdev"]) / float(timing_stats["mean"])
    repeatability_limit = _finite(
        case.measurement_protocol.get("repeatability_limit"),
        label="causal training repeatability limit",
    )
    artifact_rows = []
    for execution_index, (report_path, manifest_path) in enumerate(
        zip(report_paths, manifest_paths, strict=True), start=1
    ):
        artifact_rows.append(
            {
                "execution_index": execution_index,
                "files": [
                    {
                        "role": "report",
                        "sha256": promotion.sha256_file(report_path),
                        "n_bytes": report_path.stat().st_size,
                    },
                    {
                        "role": "provenance",
                        "sha256": promotion.sha256_file(manifest_path),
                        "n_bytes": manifest_path.stat().st_size,
                    },
                ],
            }
        )
    fingerprints = [
        str((report.get("run_fingerprint") or {}).get("comparison_fingerprint_sha256"))
        for report in reports
    ]
    return {
        "schema": RECORD_SCHEMA,
        "case_id": case.case_id,
        "workload": case.workload.id,
        "profile": case.profile,
        "mode": case.mode,
        "phase": case.phase,
        "result_role": case.result_role,
        "evidence_class": "two-run-provisional",
        "review_eligible": False,
        "eligible_for_promotion": False,
        "eligible_for_public_baseline": False,
        "source_git_sha": source_git_sha,
        "selected_reference_execution": 1,
        "execution": {
            "requested_device": reports[0].get("device_requested"),
            "executed_devices": sorted(
                {str(report.get("device_executed")) for report in reports}
            ),
            "backends": sorted({str(report.get("backend")) for report in reports}),
            "hardware_chips": sorted(
                {
                    str(
                        (
                            (report.get("run_fingerprint") or {}).get("hardware") or {}
                        ).get("chip")
                    )
                    for report in reports
                }
            ),
            "data_modes": sorted({str(report.get("data_mode")) for report in reports}),
            "canonical_seed": case.canonical_seed,
        },
        "measurement": {
            "run_count": 2,
            "primary_metric": "train_and_eval_seconds",
            "values": timing_values,
            "aggregate": timing_stats,
        },
        "quality": {
            "metric": "cross_entropy_loss",
            "gate": dict(case.gate),
            "values": quality_values,
            "aggregate": promotion.aggregate(quality_values),
            "all_runs_pass": all(
                (report.get("promotion_contract") or {}).get("status") == "passed"
                for report in reports
            ),
        },
        "repeatability": {
            "observed": True,
            "metric": "train_and_eval_seconds",
            "coefficient_of_variation": coefficient,
            "limit": repeatability_limit,
            "passed": coefficient <= repeatability_limit,
            "note": "Two runs are diagnostic and do not establish a repeatability claim.",
        },
        "artifacts": artifact_rows,
        "comparison_fingerprints": fingerprints,
        "lineage_package": {
            "sha256": promotion.sha256_file(package_path),
            "n_bytes": package_path.stat().st_size,
            "verification_checks": len(package_checks),
            "selected_execution": 1,
        },
    }


def build_index(
    records: Mapping[str, Mapping[str, Any]],
    *,
    source_git_sha: str,
    source_lock: Mapping[str, Any],
    source_lock_bytes: bytes,
) -> dict[str, Any]:
    expected = promotion.expected_cases()
    if set(records) != set(expected):
        raise ValueError(
            "provisional result closure mismatch; "
            f"missing={sorted(set(expected) - set(records))}, "
            f"extra={sorted(set(records) - set(expected))}"
        )
    entries = []
    for identifier, record in sorted(records.items()):
        data = canonical_json_bytes(record)
        entries.append(
            {
                "case_id": identifier,
                "workload": record.get("workload"),
                "mode": record.get("mode"),
                "phase": record.get("phase"),
                "result_role": record.get("result_role"),
                "evidence_class": record.get("evidence_class"),
                "eligible_for_promotion": record.get("eligible_for_promotion"),
                "path": f"provisional_results/{identifier}.json",
                "sha256": promotion.sha256_bytes(data),
            }
        )
    return {
        "schema": INDEX_SCHEMA,
        "source_git_sha": source_git_sha,
        "source_lock": {
            "path": SOURCE_LOCK_PATH,
            "schema": source_lock.get("schema"),
            "sha256": promotion.sha256_bytes(source_lock_bytes),
            "file_count": source_lock.get("file_count"),
            "contract_count": source_lock.get("contract_count"),
        },
        "workload_count": len({entry["workload"] for entry in entries}),
        "case_count": len(entries),
        "five_run_verified_case_count": sum(
            entry["evidence_class"] == "five-run-verified" for entry in entries
        ),
        "provisional_case_count": sum(
            entry["evidence_class"] != "five-run-verified" for entry in entries
        ),
        "publication_status": "draft-provisional-not-mlcommons-verified",
        "cases": entries,
    }


def _sync_file(path: Path, expected: bytes, *, check: bool) -> bool:
    path.resolve().relative_to(ROOT.resolve())
    if path.is_symlink():
        raise ValueError(f"generated destination may not be a symlink: {path}")
    current = path.read_bytes() if path.is_file() else None
    if current == expected:
        return True
    if not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    return False


def _safe_external_root(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    for output_root in OUTPUT_ROOTS:
        if (
            resolved == output_root.resolve()
            or resolved in output_root.resolve().parents
        ):
            raise ValueError(f"{label} overlaps an output root: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promotion-evidence-root", type=Path, required=True)
    parser.add_argument("--provisional-evidence-root", type=Path, required=True)
    parser.add_argument("--causal-training-attempt-root", type=Path, required=True)
    parser.add_argument("--causal-training-package", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if not promotion.SHA40_RE.fullmatch(args.source_git_sha):
            raise ValueError(
                "--source-git-sha must be 40 lowercase hexadecimal characters"
            )
        promotion_root = _safe_external_root(
            args.promotion_evidence_root, label="promotion evidence root"
        )
        provisional_root = _safe_external_root(
            args.provisional_evidence_root, label="provisional evidence root"
        )
        causal_root = _safe_external_root(
            args.causal_training_attempt_root, label="causal training attempt root"
        )
        package_path = args.causal_training_package.resolve()
        if not package_path.is_file():
            raise ValueError(f"causal training package is missing: {package_path}")
        cases = promotion.expected_cases()
        sweep_tool_sha256 = promotion.source_sweep_tool_sha256(args.source_git_sha)
        records: dict[str, dict[str, Any]] = {}
        cache: dict[Path, tuple[int, str]] = {}
        with promotion.source_project_checkout(args.source_git_sha) as source_checkout:
            for path in sorted(promotion_root.rglob("evidence_summary.json")):
                payload, data = promotion.load_json_object(path, label=str(path))
                if (
                    payload.get("status") != "valid"
                    or payload.get("eligible_for_promotion") is not True
                ):
                    continue
                case = _summary_case(payload, cases)
                promotion.validate_summary_structure(
                    path,
                    payload,
                    case=case,
                    source_git_sha=args.source_git_sha,
                    sweep_tool_sha256=sweep_tool_sha256,
                )
                promotion.verify_external_evidence(
                    promotion_root,
                    path,
                    payload,
                    data,
                    case=case,
                    source_project_root=source_checkout,
                    cache=cache,
                )
                if case.case_id in records:
                    raise ValueError(f"duplicate promotion evidence for {case.case_id}")
                records[case.case_id] = build_summary_record(
                    case, payload, data, evidence_class="five-run-verified"
                )
            for path in sorted(provisional_root.rglob("evidence_summary.json")):
                payload, data = promotion.load_json_object(path, label=str(path))
                case = _summary_case(payload, cases)
                if case.case_id in records:
                    raise ValueError(
                        f"provisional evidence duplicates verified case {case.case_id}"
                    )
                validate_development_summary(
                    path,
                    payload,
                    case=case,
                    source_git_sha=args.source_git_sha,
                    sweep_tool_sha256=sweep_tool_sha256,
                )
                verification_payload = payload
                if isinstance(payload.get("nanogpt_training_lineage"), dict):
                    verification_payload = json.loads(json.dumps(payload))
                    verification_payload["nanogpt_training_lineage"]["required"] = True
                promotion.verify_external_evidence(
                    provisional_root,
                    path,
                    verification_payload,
                    data,
                    case=case,
                    source_project_root=source_checkout,
                    cache=cache,
                )
                records[case.case_id] = build_summary_record(
                    case, payload, data, evidence_class="single-run-provisional"
                )
            if CAUSAL_TRAINING_CASE in records:
                raise ValueError(
                    "causal training must come from the preserved partial sweep"
                )
            records[CAUSAL_TRAINING_CASE] = build_causal_training_record(
                causal_root,
                package_path,
                case=cases[CAUSAL_TRAINING_CASE],
                source_git_sha=args.source_git_sha,
                source_project_root=source_checkout,
            )
        source_lock = reference_source_lock.build_source_lock(
            args.source_git_sha, project_root=ROOT
        )
        source_lock_bytes = reference_source_lock.canonical_json_bytes(source_lock)
        index = build_index(
            records,
            source_git_sha=args.source_git_sha,
            source_lock=source_lock,
            source_lock_bytes=source_lock_bytes,
        )
        index_bytes = canonical_json_bytes(index)
        stale: list[Path] = []
        for output_root in OUTPUT_ROOTS:
            expected_paths = {
                output_root / "index.json",
                output_root / "source_lock.json",
            }
            for identifier, record in sorted(records.items()):
                destination = output_root / f"{identifier}.json"
                expected_paths.add(destination)
                if not _sync_file(
                    destination, canonical_json_bytes(record), check=args.check
                ):
                    stale.append(destination)
            for destination, data in (
                (output_root / "index.json", index_bytes),
                (output_root / "source_lock.json", source_lock_bytes),
            ):
                if not _sync_file(destination, data, check=args.check):
                    stale.append(destination)
            extras = (
                sorted(set(output_root.glob("*.json")) - expected_paths)
                if output_root.is_dir()
                else []
            )
            if args.check:
                stale.extend(extras)
            else:
                for extra in extras:
                    extra.unlink()
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.check and stale:
        print("FAIL: provisional reference results are out of date")
        for path in stale:
            print(f"  - {path}")
        return 1
    action = "verified" if args.check else "synchronized"
    print(
        f"PASS: {action} {index['case_count']} draft cases "
        f"({index['five_run_verified_case_count']} five-run verified, "
        f"{index['provisional_case_count']} provisional)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
