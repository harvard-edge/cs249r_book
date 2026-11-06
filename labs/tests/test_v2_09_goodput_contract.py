"""Fixed-boundary goodput accounting, comparability, and real trade reversals."""
from copy import deepcopy
import math

import pytest

from mlsysim.labs.experiments.v2_09_experiments import (
    TRACKS, compare_goodput_study, goodput_comparison_issues, goodput_study,
)


@pytest.mark.parametrize("track", TRACKS)
@pytest.mark.parametrize("compiled,fusion", [(False, False), (True, False), (False, True), (True, True)])
def test_losses_partition_offered_capacity_exactly_once(track, compiled, fusion):
    result = goodput_study(track, compiled=compiled, fusion=fusion)
    b = result["boundaries_s"]
    f = result["factors"]
    assert b["offered"] >= b["allocated"] >= b["productive"] >= b["ideal"] >= 0
    assert math.isclose(sum(result["losses"].values()), 1.0)
    assert math.isclose(f["scheduling"] * f["runtime"] * f["program"], f["total"])
    assert math.isclose(b["ideal"] / b["offered"], f["total"])
    physical_wall = sum(row["duration_s_per_unit"] for row in result["observations"])
    assert math.isclose(physical_wall, result["contract"]["window_s"])


@pytest.mark.parametrize("field", ["artifact", "workload", "population", "precision", "hardware_ideal", "window_s", "fleet_units", "warmup_policy", "sampling_policy", "quality_evidence", "measurement_unit", "retention_policy", "reference_fingerprint"])
def test_incompatible_contracts_cannot_claim_comparable_factors(field):
    baseline = goodput_study("cloud")
    candidate = deepcopy(baseline)
    candidate["contract"][field] = "different boundary"
    assert goodput_comparison_issues(baseline, candidate) == [field]


def test_missing_contract_is_rejected_not_normalized():
    assert goodput_comparison_issues(goodput_study("cloud"), {})


def test_compilation_improves_program_but_can_reduce_whole_fleet_progress():
    short = compare_goodput_study("cloud", compiled=True, window_runs=128)
    long = compare_goodput_study("cloud", compiled=True, window_runs=131072)
    assert short["comparable"] and long["comparable"]
    assert short["result"]["factors"]["program"] > short["baseline"]["factors"]["program"]
    assert short["total_delta"] < 0
    assert long["total_delta"] > 0
    assert short["result"]["compile_setup_s"] == long["result"]["compile_setup_s"]


@pytest.mark.parametrize("track", ["tinyml", "mobile", "edge"])
def test_device_aot_build_time_is_outside_endpoint_capacity_accounting(track):
    result = goodput_study(track, compiled=True)
    assert result["build_host_setup_s"] > 0
    assert result["compile_setup_s"] == 0
    assert result["contract"]["measurement_unit"].startswith("execution-unit")


def test_joint_policy_can_trade_program_gain_for_a_larger_runtime_loss():
    result = compare_goodput_study("mobile", fusion=True, startup_runs=90)
    assert result["result"]["factors"]["program"] > result["baseline"]["factors"]["program"]
    assert result["total_delta"] < 0
    assert result["changed_fields"] == ["startup_runs", "fusion"]


def test_no_allocation_and_no_retained_work_are_honest_zero_output_states():
    result = goodput_study("cloud", allocation_wait_runs=128)
    assert result["factors"]["total"] == 0
    assert result["failure"]
    assert not result["runtime_identifiable"]
    assert not result["program_identifiable"]
    assert result["losses"]["scheduling"] == 1


@pytest.mark.parametrize("field,value", [("allocation_wait_runs", -1), ("startup_runs", True), ("discarded_runs", 0.5), ("window_runs", 0), ("compiled", "yes")])
def test_invalid_observational_inputs_are_rejected(field, value):
    with pytest.raises(ValueError):
        goodput_study("cloud", **{field: value})


def test_saved_record_preserves_resolved_path_and_source_fingerprint():
    baseline = goodput_study("cloud")
    candidate = goodput_study("cloud", fusion=True)
    assert baseline["contract"]["reference_fingerprint"] == candidate["contract"]["reference_fingerprint"]
    assert baseline["resolved_reference"]["baseline_path"]["inputs"]["batch_size"] > 0
    assert candidate["resolved_execution"]["inputs"]["fusion"]
    assert "partial executions credit no useful output" in baseline["contract"]["retention_policy"]
    assert baseline["resolved_reference"]["hardware_supply"]["peak_rate"]["magnitude"] > 0


def test_truncated_jit_setup_is_not_claimed_as_fully_observed():
    result = goodput_study("cloud", window_runs=32, compiled=True)
    assert result["compile_setup_charged_s"] < result["compile_setup_s"]
    assert result["retained_runs_per_unit"] == 0
    initialization = next(r for r in result["observations"] if r["event"].startswith("initialization"))
    assert result["compile_setup_charged_s"] <= initialization["duration_s_per_unit"]
