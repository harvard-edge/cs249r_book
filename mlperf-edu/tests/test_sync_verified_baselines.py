from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from tools import reference_source_lock, sync_verified_baselines


ROOT = Path(__file__).resolve().parents[1]


def index_record(workload_id: str):
    _index, records = sync_verified_baselines.load_index()
    path = ROOT / reference_source_lock.PROMOTED_CONTRACT_PATHS[workload_id]
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    return contract, records[workload_id], records


def test_current_baseline_remains_bound_to_immutable_index_entry():
    contract, (entry, payload), records = index_record("nanogpt-train")

    expected = sync_verified_baselines.build_baseline(
        "nanogpt-train", entry, payload, records
    )
    assert contract["verified_baseline"] == expected


def test_current_baseline_identity_fail_closed():
    contract, (entry, payload), records = index_record("nanogpt-train")
    mutated = deepcopy(contract)
    mutated["verified_baseline"]["evidence_sha256"] = "0" * 64

    expected = sync_verified_baselines.build_baseline(
        "nanogpt-train", entry, payload, records
    )

    assert mutated["verified_baseline"] != expected
    assert (
        mutated["verified_baseline"]["evidence_sha256"] != expected["evidence_sha256"]
    )


def test_schema_04_baseline_keeps_timed_primary_separate_from_quality():
    runs = [
        {
            "requested_seed": seed,
            "primary_metric_value": 10.0 + seed,
            "quality_value": 0.86 + seed * 0.01,
            "data_mode": "real",
            "backend": "pytorch-cpu",
            "chip": "Test Chip",
        }
        for seed in range(5)
    ]
    payload = {
        "schema": "mlperf-edu-reference-evidence/0.4",
        "evidence_id": "resnet18-train_max_test",
        "public_status": "score-bearing",
        "primary_metric": {
            "name": "train_and_eval_seconds",
            "role": "performance",
        },
        "quality_metric": "top1_accuracy",
        "runs": runs,
        "aggregate": {
            "primary_metric": {
                "median": 12.0,
                "min": 10.0,
                "max": 14.0,
                "mean": 12.0,
                "stdev": 1.5811388300841898,
            },
            "quality": {
                "median": 0.88,
                "min": 0.86,
                "max": 0.90,
                "mean": 0.88,
                "stdev": 0.01581138830084191,
            },
            "wall_seconds": {
                "median": 13.0,
                "min": 11.0,
                "max": 15.0,
                "mean": 13.0,
                "stdev": 1.5811388300841898,
            },
        },
        "source": {"git_sha": "a" * 40},
        "profile": "max",
        "device_requested": "cpu",
        "seeds_requested": list(range(5)),
    }
    entry = {
        "path": "reference_results/resnet18-train/test.json",
        "evidence_sha256": "b" * 64,
    }

    baseline = sync_verified_baselines.build_baseline(
        "resnet18-train", entry, payload, {}
    )

    assert baseline["primary_metric"] == "train_and_eval_seconds"
    assert baseline["metric_values_by_seed"] == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert baseline["train_and_eval_seconds"] == 12.0
    assert baseline["median"] == 12.0
    assert baseline["quality_metric"] == "top1_accuracy"
    assert baseline["quality_values_by_seed"] == [0.86, 0.87, 0.88, 0.89, 0.90]
    assert baseline["top1_accuracy"] == 0.88
    assert baseline["quality_median"] == 0.88
