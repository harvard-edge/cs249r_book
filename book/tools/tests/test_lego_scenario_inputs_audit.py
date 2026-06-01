from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "book" / "tools" / "audit"))

from book_check_lego_scenario_inputs import _classify  # noqa: E402


def classify(name, rhs):
    return _classify(name, rhs, set())


def test_network_latency_is_workload_policy_not_fabric():
    target, confidence, reason = classify("network_latency", "15 * ms")

    assert target == "Scenarios.* or Ops.*"
    assert confidence == "medium"
    assert reason == "scenario/workload policy"


def test_allreduce_bucket_is_not_storage_system():
    target, confidence, reason = classify("bucket_size", "100 * MB")

    assert target == "Systems.Storage or Scenarios.*"
    assert confidence == "medium"
    assert reason == "storage-related scenario input"


def test_storage_bandwidth_remains_high_confidence():
    target, confidence, reason = classify("pfs_node_bw", "4.0 * (GB / second)")

    assert target == "Systems.Storage"
    assert confidence == "high"
    assert reason == "storage subsystem fact"


def test_human_hourly_rate_is_not_cloud_gpu_pricing():
    target, confidence, reason = classify("hourly_rate_low", "150")

    assert target == "Infrastructure.Pricing.* or Scenarios.*"
    assert confidence == "medium"
    assert reason == "economic input or scenario price"


def test_workload_flops_formula_is_not_hardware_spec():
    target, confidence, reason = classify(
        "decode_flops_val",
        "2 * decode_batch * decode_hidden * decode_hidden * flop",
    )

    assert target == "Models.* or Scenarios.TrainingRuns"
    assert confidence == "medium"
    assert reason == "workload compute requirement"


def test_amdahl_processor_count_is_not_fabric():
    target, confidence, reason = classify("processor_count", "8")

    assert target == "Scenarios.* or keep local"
    assert confidence == "low"
    assert reason == "bare numeric scenario input"


def test_rack_latency_is_workload_policy_not_topology():
    target, confidence, reason = classify("lat_rack_ms", "85")

    assert target == "Scenarios.* or Ops.*"
    assert confidence == "medium"
    assert reason == "scenario/workload policy"
