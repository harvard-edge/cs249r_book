#!/usr/bin/env python3
"""Synchronize native verified-baseline rows from the reference index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import reference_source_lock  # noqa: E402


INDEX_PATH = ROOT / "reference_results" / "index.json"


def load_index() -> tuple[dict[str, Any], dict[str, tuple[dict, dict]]]:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    records: dict[str, tuple[dict, dict]] = {}
    for entry in index.get("summaries") or []:
        path = ROOT / str(entry["path"])
        records[str(entry["workload"])] = (
            entry,
            json.loads(path.read_text(encoding="utf-8")),
        )
    expected = set(reference_source_lock.PROMOTED_CONTRACT_PATHS)
    if set(records) != expected:
        raise ValueError(
            f"reference index closure mismatch; missing={sorted(expected - set(records))}, "
            f"extra={sorted(set(records) - expected)}"
        )
    return index, records


def stable_run_field(runs: list[dict], field: str) -> Any:
    values = {json.dumps(run.get(field), sort_keys=True) for run in runs}
    if len(values) != 1:
        raise ValueError(f"reference runs do not have one stable {field}")
    return json.loads(next(iter(values)))


def portable_note(workload_id: str) -> str:
    if workload_id == "micro-dlrm-train":
        return (
            "Raw attempts are retained on the source machine. Portable packaging "
            "is blocked by the current MovieLens redistribution policy, and no "
            "public package URL is recorded."
        )
    return (
        "Content-addressed portable run packages are retained for local review, "
        "but no public package URL is recorded."
    )


def baseline_note(workload_id: str, payload: dict, aggregate: dict) -> str:
    source_short = str((payload.get("source") or {}).get("git_sha"))[:8]
    common = (
        f"Clean five-run project reference from exact source commit {source_short}. "
        "Evidence semantics were recomputed from the raw reports and manifests "
        "during promotion. "
    )
    if payload.get("public_status") == "score-bearing":
        detail = "Every seed passed the declared quality gate. "
        if workload_id == "nanogpt-train":
            detail += (
                "The median-quality seed supplies the content-addressed checkpoint "
                "lineage used by the two NanoGPT performance references. "
            )
    else:
        repeatability = payload.get("repeatability") or {}
        cv = 100.0 * float(repeatability["coefficient_of_variation"])
        detail = (
            "Every run passed its functional gate. The primary performance metric "
            f"has {cv:.2f}% sample coefficient of variation across the five runs, "
            "within the 5% promotion limit. The speed is a machine observation, "
            "not a portable target. "
        )
    return (
        common
        + detail
        + portable_note(workload_id)
        + " This is not an MLCommons-verified result."
    )


def selected_training_lineage(source_payload: dict) -> tuple[int, str]:
    median = source_payload["aggregate"]["primary_metric"]["median"]
    selected = [
        run for run in source_payload["runs"] if run.get("quality_value") == median
    ]
    if len(selected) != 1:
        raise ValueError("NanoGPT training evidence lacks one median-quality run")
    checkpoints = [
        artifact
        for artifact in selected[0].get("artifacts") or []
        if artifact.get("role") == "checkpoint"
    ]
    if len(checkpoints) != 1:
        raise ValueError("selected NanoGPT training run lacks one checkpoint")
    return int(selected[0]["requested_seed"]), str(
        checkpoints[0]["sha256"]
    ).removeprefix("sha256:")


def build_baseline(
    workload_id: str,
    entry: dict,
    payload: dict,
    records: dict[str, tuple[dict, dict]],
) -> dict[str, Any]:
    runs = payload["runs"]
    aggregate = payload["aggregate"]["primary_metric"]
    wall = payload["aggregate"]["wall_seconds"]
    metric = str(payload["quality_metric"])
    baseline: dict[str, Any] = {
        "evidence_status": "committed-reference-summary",
        "review_eligible": True,
        "evidence_tier": "public-candidate",
        "evidence_id": payload["evidence_id"],
        "evidence_file": entry["path"],
        "evidence_sha256": entry["evidence_sha256"],
        "reference_package_availability": "local-handoff",
        "external_publication_status": "pending",
        "source_git_sha": payload["source"]["git_sha"],
        "profile": payload["profile"],
        "device_requested": payload["device_requested"],
        "data_mode": stable_run_field(runs, "data_mode"),
        "execution_backend": stable_run_field(runs, "backend"),
        "hardware_chip": stable_run_field(runs, "chip"),
    }
    if workload_id in {"nanogpt-prefill", "nanogpt-decode"}:
        source_entry, source_payload = records["nanogpt-train"]
        seed, checkpoint_digest = selected_training_lineage(source_payload)
        lineage = payload["nanogpt_training_lineage"]
        baseline.update(
            {
                "source_training_evidence_id": source_payload["evidence_id"],
                "source_training_evidence_sha256": source_entry["evidence_sha256"],
                "source_training_seed": seed,
                "source_training_checkpoint_sha256": checkpoint_digest,
                "source_training_package_sha256": str(
                    lineage["package_sha256"]
                ).removeprefix("sha256:"),
            }
        )
    baseline.update(
        {
            "seeds": payload["seeds_requested"],
            "primary_metric": metric,
            "metric_values_by_seed": [run["quality_value"] for run in runs],
            metric: aggregate["median"],
            "median": aggregate["median"],
            "min": aggregate["min"],
            "max": aggregate["max"],
            "mean": aggregate["mean"],
            "sample_stdev": aggregate["stdev"],
            "wall_seconds_median": wall["median"],
            "wall_seconds_min": wall["min"],
            "wall_seconds_max": wall["max"],
            "wall_seconds_mean": wall["mean"],
            "wall_seconds_sample_stdev": wall["stdev"],
            "accepted_runs": len(runs),
        }
    )
    if payload.get("public_status") == "performance-bearing":
        baseline["functional_passes"] = int(payload["acceptance"]["value"])
        baseline["coefficient_of_variation"] = payload["repeatability"][
            "coefficient_of_variation"
        ]
    baseline["baseline_note"] = baseline_note(workload_id, payload, aggregate)
    return baseline


def synchronized_contract(
    workload_id: str,
    path: Path,
    entry: dict,
    payload: dict,
    records: dict[str, tuple[dict, dict]],
) -> bytes:
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    contract["verified_baseline"] = build_baseline(workload_id, entry, payload, records)
    if payload.get("public_status") == "score-bearing":
        variance = contract["quality_target"].setdefault("variance_summary", {})
        aggregate = payload["aggregate"]["primary_metric"]
        variance.update(
            {
                "runs": len(payload["runs"]),
                "median": aggregate["median"],
                "min": aggregate["min"],
                "max": aggregate["max"],
                "mean": aggregate["mean"],
                "sample_stdev": aggregate["stdev"],
            }
        )
        if workload_id == "nanogpt-train":
            variance["spread_note"] = (
                "The clean five-seed public-candidate sweep produced validation "
                f"cross-entropy from {aggregate['min']:.10f} to "
                f"{aggregate['max']:.10f}. Every seed passed the 2.3 target."
            )
    text = yaml.safe_dump(
        contract,
        sort_keys=False,
        allow_unicode=True,
        width=88,
    )
    return text.encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _index, records = load_index()
    stale: list[Path] = []
    for workload_id, relative_path in sorted(
        reference_source_lock.PROMOTED_CONTRACT_PATHS.items()
    ):
        path = ROOT / relative_path
        expected = synchronized_contract(
            workload_id, path, *records[workload_id], records
        )
        if path.read_bytes() == expected:
            continue
        if args.check:
            stale.append(path)
        else:
            path.write_bytes(expected)
    if stale:
        print("verified baselines are out of date:")
        for path in stale:
            print(f"- {path.relative_to(ROOT)}")
        return 1
    action = "verified" if args.check else "synchronized"
    print(f"{action} {len(records)} verified baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
