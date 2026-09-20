import json
import math

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v2_09_experiments import (
    MODEL_ID,
    TRACKS,
    compare,
    compare_configs,
    comparison_to_dict,
    evaluate,
    get_scenario,
    quantity_from_dict,
    result_to_dict,
)


def _magnitude(quantity, unit):
    return quantity.to(unit).magnitude


def test_every_track_is_an_honest_fleet_scenario_with_feasible_default():
    for track_id in TRACKS:
        scenario = get_scenario(track_id)
        result = evaluate(track_id)

        assert "fleet" in scenario.workload or "pool" in scenario.workload
        assert "independent" in scenario.fleet_semantics or "replica" in scenario.fleet_semantics
        assert result.memory_feasible, track_id
        assert result.result_label == "simulated analytical bound"
        assert "not a hardware-derived" in result.quality_evidence


def test_phase_times_come_from_operations_traffic_and_hardware_supply():
    scenario = get_scenario("cloud")
    result = evaluate("cloud")
    evidence = next(item for item in scenario.precision_evidence if item.precision == result.precision)
    rate = evidence.peak_rate * evidence.attainable_fraction

    for phase in result.phases:
        assert _magnitude(phase.compute_time, "ms") == pytest.approx(
            _magnitude(phase.operations / rate, "ms")
        )
        assert _magnitude(phase.movement_time, "ms") == pytest.approx(
            _magnitude(phase.traffic / scenario.memory_bandwidth, "ms")
        )
        assert phase.exposed_time == max(phase.compute_time, phase.movement_time)

    expected = sum((phase.exposed_time + phase.launch_time for phase in result.phases), Q_("0 ms"))
    assert result.latency == expected


def test_fusion_reduces_intermediate_traffic_and_launches_on_same_work():
    baseline = evaluate("tinyml")
    fused = evaluate("tinyml", fusion=True)

    assert sum(phase.operations for phase in fused.phases) == sum(
        phase.operations for phase in baseline.phases
    )
    assert sum(phase.traffic for phase in fused.phases) < sum(
        phase.traffic for phase in baseline.phases
    )
    assert sum(phase.launches for phase in fused.phases) < sum(
        phase.launches for phase in baseline.phases
    )
    assert fused.latency < baseline.latency
    assert fused.memory_required > baseline.memory_required


def test_tiling_trades_more_workspace_for_less_off_chip_traffic():
    baseline = evaluate("tinyml")
    tiled = evaluate("tinyml", tiling=True)

    assert sum(phase.traffic for phase in tiled.phases) < sum(
        phase.traffic for phase in baseline.phases
    )
    assert tiled.memory_required > baseline.memory_required
    assert tiled.latency < baseline.latency


def test_batching_increases_live_state_and_amortizes_fixed_work():
    single = evaluate("mobile", batch_size=1)
    batched = evaluate("mobile", batch_size=8)

    assert batched.memory_required > single.memory_required
    assert batched.latency > single.latency
    assert batched.per_unit_throughput > single.per_unit_throughput
    assert batched.arithmetic_intensity > single.arithmetic_intensity


def test_each_track_has_a_reachable_per_unit_memory_failure():
    for track_id in TRACKS:
        failed = evaluate(track_id, batch_size=10_000)

        assert not failed.memory_feasible, track_id
        assert failed.memory_required > failed.memory_capacity
        assert failed.attainable_per_unit_throughput == Q_("0 1/s")
        assert failed.attainable_fleet_throughput == Q_("0 1/s")
        assert failed.per_unit_throughput == Q_("0 1/s")
        assert failed.fleet_throughput == Q_("0 1/s")
        assert failed.potential_per_unit_throughput > Q_("0 1/s")
        assert failed.potential_fleet_throughput > Q_("0 1/s")


def test_precision_uses_supported_rate_and_supplied_quality_evidence():
    fp16 = evaluate("cloud", precision="fp16")
    fp8 = evaluate("cloud", precision="fp8")

    assert sum(phase.traffic for phase in fp8.phases) < sum(
        phase.traffic for phase in fp16.phases
    )
    assert fp8.compute_time < fp16.compute_time
    assert fp8.latency < fp16.latency
    assert fp8.quality_pct < fp16.quality_pct
    assert compare("cloud", precision="fp8")["quality_delta_pct"] < 0


def test_every_supported_track_precision_endpoint_runs_with_finite_results():
    for track_id, scenario in TRACKS.items():
        for evidence in scenario.precision_evidence:
            result = evaluate(track_id, precision=evidence.precision)

            assert math.isfinite(_magnitude(result.latency, "ms"))
            assert _magnitude(result.latency, "ms") > 0
            assert result.quality_pct == evidence.quality_pct


def test_compile_setup_requires_reuse_before_it_pays_back():
    baseline = evaluate("mobile")
    cold_compile = evaluate("mobile", compiled=True, reuse_count=1)
    reused_compile = evaluate("mobile", compiled=True, reuse_count=100_000)

    assert cold_compile.latency > baseline.latency
    assert reused_compile.latency < baseline.latency
    assert cold_compile.compile_time_per_run > reused_compile.compile_time_per_run
    assert reused_compile.launch_time < baseline.launch_time


def test_combined_optimization_recomputes_shared_path_instead_of_multiplying_gains():
    fusion = compare("tinyml", fusion=True)["speedup"]
    tiling = compare("tinyml", tiling=True)["speedup"]
    combined = compare("tinyml", fusion=True, tiling=True)["speedup"]
    combined_result = evaluate("tinyml", fusion=True, tiling=True)

    assert combined > max(fusion, tiling)
    assert not math.isclose(combined, fusion * tiling, rel_tol=1e-3)
    assert combined_result.latency == sum(
        (phase.exposed_time + phase.launch_time for phase in combined_result.phases),
        Q_("0 ms"),
    )


def test_fleet_size_is_noncausal_for_per_unit_latency_memory_and_quality():
    small = evaluate("edge", fleet_units=1)
    large = evaluate("edge", fleet_units=5_000)

    assert small.latency == large.latency
    assert small.memory_required == large.memory_required
    assert small.memory_feasible == large.memory_feasible
    assert small.quality_pct == large.quality_pct
    assert _magnitude(large.fleet_throughput, "1/s") == pytest.approx(
        5_000 * _magnitude(small.fleet_throughput, "1/s")
    )


def test_results_retain_exact_resolved_evaluator_arguments():
    result = evaluate(
        "cloud",
        batch_size=7,
        fleet_units=11,
        precision="FP8",
        fusion=True,
        tiling=True,
        compiled=True,
        reuse_count=321,
    )

    assert result.inputs.track_id == "cloud"
    assert result.inputs.batch_size == 7
    assert result.inputs.fleet_units == 11
    assert result.inputs.precision == "fp8"
    assert result.inputs.fusion is True
    assert result.inputs.tiling is True
    assert result.inputs.compiled is True
    assert result.inputs.reuse_count == 321


def test_serialization_preserves_units_and_comparison_baseline_inputs():
    comparison = compare(
        "tinyml",
        batch_size=3,
        fleet_units=17,
        fusion=True,
        tiling=True,
        reuse_count=9,
    )
    payload = comparison_to_dict(comparison)

    assert payload["model_id"] == MODEL_ID
    assert payload["baseline"]["inputs"] == {
        "track_id": "tinyml",
        "batch_size": 3,
        "fleet_units": 17,
        "precision": "int8",
        "fusion": False,
        "tiling": False,
        "compiled": False,
        "reuse_count": 9,
    }
    assert payload["result"]["inputs"]["fusion"] is True
    assert payload["result"]["inputs"]["tiling"] is True
    assert quantity_from_dict(payload["result"]["latency"]) == comparison["result"].latency
    json.dumps(payload, allow_nan=False)

    standalone = result_to_dict(evaluate("edge"))
    assert standalone["model_id"] == MODEL_ID
    assert quantity_from_dict(standalone["latency"]).to("ms") == evaluate("edge").latency


def test_explicit_config_comparison_preserves_causal_baseline_and_finite_failure():
    baseline_options = {
        "batch_size": 4,
        "fleet_units": 21,
        "precision": "int8",
        "fusion": True,
        "tiling": False,
        "compiled": False,
        "reuse_count": 1,
    }
    result_options = {**baseline_options, "batch_size": 1_000}
    comparison = compare_configs(
        "mobile",
        baseline_options=baseline_options,
        result_options=result_options,
    )
    payload = comparison_to_dict(comparison)

    assert payload["baseline"]["inputs"] == baseline_options | {"track_id": "mobile"}
    assert payload["result"]["inputs"]["batch_size"] == 1_000
    assert not payload["result"]["memory_feasible"]
    assert payload["result"]["attainable_per_unit_throughput"]["magnitude"] == 0
    assert payload["result"]["per_unit_throughput"]["magnitude"] == 0
    assert payload["result"]["potential_per_unit_throughput"]["magnitude"] > 0
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"track_id": "unknown"},
        {"track_id": "tinyml", "batch_size": 0},
        {"track_id": "mobile", "fleet_units": -1},
        {"track_id": "edge", "reuse_count": True},
        {"track_id": "tinyml", "precision": "fp8"},
        {"track_id": "cloud", "precision": "made-up"},
    ],
)
def test_invalid_or_unsupported_inputs_fail_explicitly(kwargs):
    with pytest.raises((TypeError, ValueError)):
        evaluate(**kwargs)
