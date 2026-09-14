#!/usr/bin/env python3
"""Synchronize verified registry baselines from the case-level reference index."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools import import_reference_evidence as evidence  # noqa: E402
from tools import reference_source_lock  # noqa: E402


INDEX_PATH = ROOT / "reference_results" / "index.json"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _safe_index_path(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("reference index path must be a string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe reference index path {value!r}")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"reference index path escapes repository: {value!r}") from exc
    return path


def load_index() -> tuple[dict[str, Any], dict[str, tuple[dict, dict]]]:
    index = _load_json(INDEX_PATH)
    if index.get("schema") != evidence.INDEX_SCHEMA:
        raise ValueError(
            f"reference index schema is {index.get('schema')!r}, expected "
            f"{evidence.INDEX_SCHEMA!r}"
        )
    entries = index.get("cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError("reference index cases must be a nonempty list")
    if index.get("case_count") != len(entries):
        raise ValueError("reference index case_count does not match cases")
    records: dict[str, tuple[dict, dict]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("reference index case entries must be objects")
        identifier = entry.get("case_id")
        if not isinstance(identifier, str) or identifier in records:
            raise ValueError(f"missing or duplicate reference case {identifier!r}")
        path = _safe_index_path(entry.get("path"))
        payload = _load_json(path)
        if evidence.sha256_file(path) != entry.get("evidence_sha256"):
            raise ValueError(f"{identifier}: evidence digest does not match index")
        if payload.get("evidence_id") != entry.get("evidence_id"):
            raise ValueError(f"{identifier}: evidence id does not match index")
        records[identifier] = (entry, payload)
    expected = set(evidence.expected_cases())
    if set(records) != expected:
        raise ValueError(
            "reference index case closure mismatch; "
            f"missing={sorted(expected - set(records))}, "
            f"extra={sorted(set(records) - expected)}"
        )
    return index, records


def _stable_run_field(runs: list[dict], field: str) -> Any:
    values = {json.dumps(run.get(field), sort_keys=True) for run in runs}
    if len(values) != 1:
        raise ValueError(f"reference runs do not have one stable {field}")
    return json.loads(next(iter(values)))


def _metric_values(payload: Mapping[str, Any], field: str) -> list[float]:
    values: list[float] = []
    for run in payload.get("runs") or []:
        value = run.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"reference run has invalid {field}")
        values.append(float(value))
    if len(values) != 5:
        raise ValueError(f"reference evidence requires five {field} values")
    return values


def build_baseline(
    entry: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Create one registry baseline without weakening its measured contract."""
    runs = payload.get("runs") or []
    if not isinstance(runs, list) or len(runs) != 5:
        raise ValueError("verified baseline requires five runs")
    primary = payload["aggregate"]["primary_metric"]
    wall = payload["aggregate"]["wall_seconds"]
    primary_values = _metric_values(payload, "primary_metric_value")
    baseline: dict[str, Any] = {
        "evidence_status": "committed-reference-summary",
        "review_eligible": True,
        "protocol_compatibility": "current",
        "replacement_required": False,
        "case_id": entry["case_id"],
        "evidence_id": entry["evidence_id"],
        "evidence_file": entry["path"],
        "evidence_sha256": entry["evidence_sha256"],
        "source_git_sha": payload["source"]["git_sha"],
        "profile": payload["profile"],
        "mode": payload["mode"],
        "phase": payload.get("phase"),
        "result_role": payload["result_role"],
        "device_requested": payload["device_requested"],
        "data_mode": _stable_run_field(runs, "data_mode"),
        "execution_backend": _stable_run_field(runs, "backend"),
        "hardware_chip": _stable_run_field(runs, "chip"),
        "comparison_fingerprint_sha256": payload["comparison_fingerprint_sha256"],
        "accepted_runs": 5,
        "canonical_seed": payload["canonical_seed"],
        "primary_metric": payload["primary_metric"]["name"],
        "metric_values": primary_values,
        "median": primary["median"],
        "min": primary["min"],
        "max": primary["max"],
        "mean": primary["mean"],
        "sample_stdev": primary["stdev"],
        "coefficient_of_variation": payload["primary_metric_repeatability"][
            "coefficient_of_variation"
        ],
        "repeatability_limit": payload["primary_metric_repeatability"]["limit"],
        "wall_seconds_median": wall["median"],
        "wall_seconds_min": wall["min"],
        "wall_seconds_max": wall["max"],
        "wall_seconds_mean": wall["mean"],
        "wall_seconds_sample_stdev": wall["stdev"],
        "reference_package_availability": "local-handoff",
        "external_publication_status": "pending",
    }
    if payload["result_role"] == "score-bearing":
        quality = payload["aggregate"]["quality"]
        quality_values = _metric_values(payload, "quality_value")
        quality_metric = payload["quality_metric"]
        baseline.update(
            {
                "quality_metric": quality_metric,
                "quality_values": quality_values,
                quality_metric: quality["median"],
                "quality_median": quality["median"],
                "quality_min": quality["min"],
                "quality_max": quality["max"],
                "quality_mean": quality["mean"],
                "quality_sample_stdev": quality["stdev"],
                "quality_gate": payload["quality_gate"],
            }
        )
    else:
        baseline.update(
            {
                "functional_gate": payload["functional_gate"],
                "functional_passes": int(payload["acceptance"]["value"]),
            }
        )
        source_training = entry.get("source_training")
        if not isinstance(source_training, dict):
            raise ValueError(
                f"{entry.get('case_id')}: performance-bearing causal phase lacks "
                "source_training binding"
            )
        baseline["source_training"] = dict(source_training)
    baseline["baseline_note"] = (
        "Clean five-run reference from one exact source commit. Every run passed "
        "its declared gate and the primary timing coefficient of variation is "
        "within 5%. Raw content-addressed packages remain in the local review "
        "handoff. This is not an MLCommons-verified result."
    )
    return baseline


def _variance_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    quality = payload["aggregate"]["quality"]
    return {
        "runs": 5,
        "statistic": "median",
        "metric": payload["quality_metric"],
        "median": quality["median"],
        "min": quality["min"],
        "max": quality["max"],
        "mean": quality["mean"],
        "sample_stdev": quality["stdev"],
        "acceptance_rule": "every run and the median satisfy the declared quality target",
    }


def _reference_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = dict((payload.get("basis") or {}).get("reference_protocol") or {})
    protocol.update(
        {
            "outer_reference_runs": 5,
            "repeatability_metric": payload["primary_metric_repeatability"]["metric"],
            "repeatability_limit": payload["primary_metric_repeatability"]["limit"],
            "fresh_process_per_run": True,
        }
    )
    return protocol


def synchronized_contract(
    contract: Mapping[str, Any],
    records: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return a promoted workload contract with every case bound to evidence."""
    result = dict(contract)
    workload_id = str(result["id"])
    owned = {
        identifier: value
        for identifier, value in records.items()
        if value[0].get("workload") == workload_id
    }
    if not owned:
        raise ValueError(f"{workload_id}: reference index has no evidence cases")
    baselines = {
        identifier: build_baseline(entry, payload)
        for identifier, (entry, payload) in sorted(owned.items())
    }
    canonical = result.get("canonical_max_contract") or {}
    canonical_id = evidence.case_id(
        workload_id,
        "max",
        str(canonical.get("mode")),
        None,
    )
    if canonical_id not in baselines:
        raise ValueError(f"{workload_id}: canonical baseline {canonical_id} is missing")
    _canonical_entry, canonical_payload = owned[canonical_id]
    if canonical_payload.get("result_role") != canonical.get("result_role"):
        raise ValueError(
            f"{workload_id}: canonical evidence role does not match registry"
        )

    result["verified_baselines"] = baselines
    result["verified_baseline"] = baselines[canonical_id]
    public = dict(result.get("public") or {})
    public["status"] = canonical_payload["result_role"]
    original = str(public.get("rationale") or "").split(";", 1)[0].rstrip(".")
    public["rationale"] = (
        f"{original}. Five-run canonical reference evidence is committed, with "
        "additional mode and phase cases recorded under verified_baselines."
    )
    result["public"] = public

    quality_target = dict(result.get("quality_target") or {})
    quality_target["reference_runs"] = 5
    quality_target["variance_summary"] = _variance_summary(canonical_payload)
    quality_target["reference_protocol"] = _reference_protocol(canonical_payload)
    result["quality_target"] = quality_target
    return result


def _yaml_bytes(value: Mapping[str, Any]) -> bytes:
    text = yaml.safe_dump(
        dict(value),
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    return text.encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not INDEX_PATH.is_file():
        if not args.check:
            print(
                "FAIL: no promoted reference index exists; draft results must not "
                "be synchronized as verified baselines",
                file=sys.stderr,
            )
            return 1
        stale = []
        for _workload_id, relative_path in sorted(
            reference_source_lock.PROMOTED_CONTRACT_PATHS.items()
        ):
            path = ROOT / relative_path
            contract = yaml.safe_load(path.read_text(encoding="utf-8"))
            if contract.get("verified_baseline") or contract.get("verified_baselines"):
                stale.append(path)
        if stale:
            print("FAIL: registry contains baselines without a promoted index")
            for path in stale:
                print(f"  - {path.relative_to(ROOT)}")
            return 1
        print(
            "PASS: no promoted index exists and no draft result is exposed as a "
            "verified registry baseline"
        )
        return 0
    try:
        _index, records = load_index()
        writes: list[tuple[Path, bytes]] = []
        stale: list[Path] = []
        for _workload_id, relative_path in sorted(
            reference_source_lock.PROMOTED_CONTRACT_PATHS.items()
        ):
            path = ROOT / relative_path
            contract = yaml.safe_load(path.read_text(encoding="utf-8"))
            expected = _yaml_bytes(synchronized_contract(contract, records))
            if path.read_bytes() == expected:
                continue
            if args.check:
                stale.append(path)
            else:
                writes.append((path, expected))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if stale:
        print("FAIL: verified baselines are out of date")
        for path in stale:
            print(f"  - {path.relative_to(ROOT)}")
        return 1
    for path, expected in writes:
        path.write_bytes(expected)
    action = "verified" if args.check else "synchronized"
    print(f"PASS: {action} {len(records)} case-level verified baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
