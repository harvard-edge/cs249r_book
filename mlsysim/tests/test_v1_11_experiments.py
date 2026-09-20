import json

import pytest

from mlsysim.core.units import Q_, resolve_precision
from mlsysim.engine.v1_11_experiments import (
    Candidate,
    ExecutionContract,
    TRACK_SCENARIOS,
    analyze_execution_path,
    analyze_roofline,
    analyze_tile_mapping,
    compare_application_speedup,
    compose_application_path,
    get_track_scenario,
    make_replay_packet,
    rank_accelerators,
    replay_experiment,
    scenario_budgets,
    scenario_candidates,
    scenario_gemm_demand,
    to_jsonable,
)
from mlsysim.hardware import Hardware


def _same_magnitude(left, right, unit="ms"):
    return left.to(unit).magnitude == pytest.approx(right.to(unit).magnitude)


@pytest.mark.parametrize("track", ("tinyml", "mobile", "edge", "cloud"))
def test_every_track_runs_each_chapter_experiment(track):
    scenario = get_track_scenario(track)
    m, n, k = scenario.dimensions
    tm, tn, tk = scenario.tile
    _, element_bytes = resolve_precision(scenario.precision)
    base_bytes = (m * k + k * n + m * n) * element_bytes
    operations = Q_(2 * m * n * k, "flop")

    roofline = analyze_roofline(
        hardware=scenario.hardware,
        operations=operations,
        base_bytes_moved=base_bytes,
        precision=scenario.precision,
    )
    tile = analyze_tile_mapping(
        hardware=scenario.hardware,
        m=m,
        n=n,
        k=k,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        precision=scenario.precision,
        fused=True,
        scratchpad_capacity=scenario.local_capacity,
    )
    execution = analyze_execution_path(
        hardware=scenario.hardware,
        fallback_hardware=scenario.fallback,
        contract=scenario.contract,
        operation="gemm",
        precision=scenario.precision,
        m=m,
        n=n,
        k=k,
    )
    transfer = Q_(1, "MiB") if scenario.hardware.interconnect else Q_(0, "byte")
    application = compose_application_path(
        hardware=scenario.hardware,
        kernel_time=execution.latency,
        host_time=Q_(1, "ms"),
        transfer_bytes=transfer,
        transfer_fixed_latency=Q_(0.05, "ms"),
        launches=2,
        postprocess_time=Q_(0.5, "ms"),
    )
    ranking = rank_accelerators(
        candidates=(
            Candidate("primary", scenario.hardware, scenario.contract, Q_(1, "dollar/hour")),
            Candidate("alternative", scenario.alternative, scenario.contract, Q_(0.5, "dollar/hour")),
        ),
        fallback_hardware=scenario.fallback,
        operation="gemm",
        precision=scenario.precision,
        m=m,
        n=n,
        k=k,
        host_time=Q_(1, "ms"),
        transfer_bytes=transfer,
        transfer_fixed_latency=Q_(0.05, "ms"),
        launches=2,
        postprocess_time=Q_(0.5, "ms"),
        objective="latency",
        latency_budget=Q_(1, "hour"),
        energy_budget=Q_(1, "MJ"),
        cost_budget=Q_(1, "dollar"),
    )

    assert roofline.latency > Q_(0, "ms")
    assert tile.fits and tile.movement_time is not None
    assert execution.path == "native"
    assert application.total_time > execution.latency
    assert len(ranking.rows) == 2
    assert ranking.recommendation is not None


def test_roofline_changes_only_when_the_binding_resource_changes():
    hardware = Hardware.Cloud.H100
    memory_bound = analyze_roofline(
        hardware=hardware,
        operations=Q_(1, "GFLOP"),
        base_bytes_moved=Q_(1, "GB"),
        precision="fp16",
    )
    more_compute = analyze_roofline(
        hardware=hardware,
        operations=Q_(1, "GFLOP"),
        base_bytes_moved=Q_(1, "GB"),
        precision="fp16",
        compute_scale=4,
    )
    more_bandwidth = analyze_roofline(
        hardware=hardware,
        operations=Q_(1, "GFLOP"),
        base_bytes_moved=Q_(1, "GB"),
        precision="fp16",
        bandwidth_scale=2,
    )
    more_reuse = analyze_roofline(
        hardware=hardware,
        operations=Q_(1, "GFLOP"),
        base_bytes_moved=Q_(1, "GB"),
        precision="fp16",
        reuse=2,
    )

    assert memory_bound.bottleneck == "memory"
    assert _same_magnitude(memory_bound.latency, more_compute.latency)
    assert more_bandwidth.latency < memory_bound.latency
    assert more_reuse.bytes_moved == Q_(0.5, "GB")


def test_tile_fit_and_fusion_are_explicit_not_penalty_factors():
    kwargs = dict(
        hardware=Hardware.Cloud.H100,
        m=256,
        n=256,
        k=256,
        precision="fp16",
        scratchpad_capacity=Q_(48, "KiB"),
    )
    fused = analyze_tile_mapping(**kwargs, tile_m=32, tile_n=32, tile_k=32, fused=True)
    unfused = analyze_tile_mapping(**kwargs, tile_m=32, tile_n=32, tile_k=32, fused=False)
    spill = analyze_tile_mapping(**kwargs, tile_m=128, tile_n=128, tile_k=128, fused=True)

    assert fused.fits
    assert fused.dram_bytes < unfused.dram_bytes
    assert fused.movement_energy < unfused.movement_energy
    assert not spill.fits
    assert spill.movement_time is None


def test_execution_contract_produces_native_padded_and_fallback_paths():
    contract = ExecutionContract(
        "test scenario contract",
        frozenset({"gemm"}),
        frozenset({"int8"}),
        16,
    )
    common = dict(
        hardware=Hardware.Edge.JetsonAGXOrin,
        fallback_hardware=Hardware.Edge.JetsonOrinNX,
        contract=contract,
        operation="gemm",
        precision="int8",
        k=64,
    )
    native = analyze_execution_path(**common, m=64, n=64)
    padded = analyze_execution_path(**common, m=65, n=64)
    fallback = analyze_execution_path(**{**common, "operation": "scatter"}, m=64, n=64)

    assert native.path == "native" and native.extra_operations == Q_(0, "flop")
    assert padded.path == "padded" and padded.extra_operations > Q_(0, "flop")
    assert fallback.path == "fallback"
    assert fallback.hardware_name == Hardware.Edge.JetsonOrinNX.name


def test_supplied_quality_observation_cannot_change_execution_time():
    scenario = TRACK_SCENARIOS["mobile"]
    kwargs = dict(
        hardware=scenario.hardware,
        fallback_hardware=scenario.fallback,
        contract=scenario.contract,
        operation="gemm",
        precision=scenario.precision,
        m=512,
        n=512,
        k=512,
    )
    accepted = analyze_execution_path(**kwargs, quality_observation="matched-task check passed")
    rejected = analyze_execution_path(**kwargs, quality_observation="matched-task check failed")

    assert accepted.quality_observation != rejected.quality_observation
    assert _same_magnitude(accepted.latency, rejected.latency)


def test_application_composition_preserves_unaccelerated_stages():
    kwargs = dict(
        hardware=Hardware.Cloud.H100,
        host_time=Q_(4, "ms"),
        transfer_bytes=Q_(64, "MB"),
        transfer_fixed_latency=Q_(0.1, "ms"),
        launches=10,
        postprocess_time=Q_(2, "ms"),
    )
    baseline = compose_application_path(**kwargs, kernel_time=Q_(20, "ms"))
    accelerated = compose_application_path(**kwargs, kernel_time=Q_(1, "ms"))

    assert baseline.kernel_time / accelerated.kernel_time == pytest.approx(20)
    assert baseline.total_time / accelerated.total_time < 20
    assert baseline.host_time == accelerated.host_time
    assert baseline.transfer_time == accelerated.transfer_time
    assert baseline.launch_time == accelerated.launch_time


def test_ranking_filters_then_optimizes_instead_of_selecting_first_pass():
    contract = ExecutionContract(
        "cloud comparison fixture",
        frozenset({"gemm"}),
        frozenset({"fp16"}),
        16,
    )
    result = rank_accelerators(
        candidates=(
            Candidate("expensive-first", Hardware.Cloud.H100, contract, Q_(12, "dollar/hour")),
            Candidate("cheap-second", Hardware.Cloud.A100, contract, Q_(2, "dollar/hour")),
        ),
        fallback_hardware=Hardware.Cloud.ReferenceCPU,
        operation="gemm",
        precision="fp16",
        m=1024,
        n=1024,
        k=1024,
        host_time=Q_(1, "ms"),
        transfer_bytes=Q_(1, "MB"),
        transfer_fixed_latency=Q_(0.05, "ms"),
        launches=2,
        postprocess_time=Q_(0.5, "ms"),
        objective="cost",
        latency_budget=Q_(1, "s"),
        energy_budget=Q_(1, "kJ"),
        cost_budget=Q_(1, "dollar"),
    )

    assert all(row.feasible for row in result.rows)
    assert result.recommendation is not None
    assert result.recommendation.candidate_id == "cheap-second"


def test_ranking_can_report_no_feasible_design():
    scenario = TRACK_SCENARIOS["tinyml"]
    result = rank_accelerators(
        candidates=(Candidate("tiny", scenario.hardware, scenario.contract, Q_(0.01, "dollar/hour")),),
        fallback_hardware=scenario.fallback,
        operation="gemm",
        precision="int8",
        m=64,
        n=64,
        k=64,
        host_time=Q_(0.1, "ms"),
        transfer_bytes=Q_(0, "byte"),
        transfer_fixed_latency=Q_(0, "ms"),
        launches=1,
        postprocess_time=Q_(0.1, "ms"),
        objective="energy",
        latency_budget=Q_(1, "ns"),
        energy_budget=Q_(1, "pJ"),
        cost_budget=Q_(1, "dollar"),
    )

    assert not result.rows[0].feasible
    assert {"latency", "energy"}.issubset(result.rows[0].violations)
    assert result.recommendation is None


def test_unsupported_candidate_is_not_recommended_via_its_fallback():
    unsupported = ExecutionContract(
        "unsupported fixture",
        frozenset({"conv"}),
        frozenset({"fp16"}),
        16,
    )
    result = rank_accelerators(
        candidates=(Candidate("unsupported", Hardware.Cloud.H100, unsupported, Q_(1, "dollar/hour")),),
        fallback_hardware=Hardware.Cloud.A100,
        operation="gemm",
        precision="fp16",
        m=256,
        n=256,
        k=256,
        host_time=Q_(1, "ms"),
        transfer_bytes=Q_(1, "MB"),
        transfer_fixed_latency=Q_(0.05, "ms"),
        launches=2,
        postprocess_time=Q_(0.5, "ms"),
        objective="latency",
        latency_budget=Q_(1, "hour"),
        energy_budget=Q_(1, "MJ"),
        cost_budget=Q_(1, "dollar"),
    )

    assert result.rows[0].execution_path == "fallback"
    assert "native execution support" in result.rows[0].violations
    assert result.recommendation is None


def test_ranking_excludes_candidate_when_application_latency_exceeds_budget():
    scenario = TRACK_SCENARIOS["cloud"]
    m, n, k = scenario.dimensions
    execution = analyze_execution_path(
        hardware=scenario.hardware,
        fallback_hardware=scenario.fallback,
        contract=scenario.contract,
        operation="gemm",
        precision=scenario.precision,
        m=m,
        n=n,
        k=k,
    )
    latency_budget = execution.latency + Q_(1, "ms")
    result = rank_accelerators(
        candidates=(Candidate("primary", scenario.hardware, scenario.contract, scenario.primary_hourly_cost),),
        fallback_hardware=scenario.fallback,
        operation="gemm",
        precision=scenario.precision,
        m=m,
        n=n,
        k=k,
        host_time=scenario.host_time,
        transfer_bytes=scenario.transfer_bytes,
        transfer_fixed_latency=scenario.transfer_fixed_latency,
        launches=scenario.launches,
        postprocess_time=scenario.postprocess_time,
        objective="latency",
        latency_budget=latency_budget,
        energy_budget=scenario.energy_budget,
        cost_budget=scenario.cost_budget,
    )
    row = result.rows[0]
    assert execution.latency < latency_budget
    assert row.application.total_time > latency_budget
    assert "latency" in row.violations
    assert not row.feasible
    assert row not in result.ranked_feasible
    assert result.recommendation is None


def test_replay_payload_preserves_quantity_value_and_unit():
    result = analyze_roofline(
        hardware=Hardware.Cloud.H100,
        operations=Q_(2, "GFLOP"),
        base_bytes_moved=Q_(32, "MB"),
        precision="fp16",
        reuse=4,
        compute_scale=2,
        bandwidth_scale=0.5,
    )
    payload = to_jsonable(result)

    json.dumps(payload)
    assert payload["base_bytes_moved"] == {"magnitude": 32_000_000, "unit": "B"}
    assert payload["reuse"] == 4
    assert payload["hardware_ref"] == "Hardware.Cloud.H100"
    restored = Q_(payload["latency"]["magnitude"], payload["latency"]["unit"])
    assert restored == result.latency


def test_candidate_replay_payload_retains_registry_reference():
    candidate = Candidate(
        "a100",
        Hardware.Cloud.A100,
        TRACK_SCENARIOS["cloud"].contract,
        Q_(2, "dollar/hour"),
    )
    payload = to_jsonable(candidate)

    assert payload["hardware"]["registry_ref"] == "Hardware.Cloud.A100"
    assert payload["hourly_operating_cost"]["unit"] == "dollar/h"


def test_json_roundtrip_replays_named_method_with_registry_and_units():
    inputs = {
        "hardware": Hardware.Cloud.H100,
        "operations": Q_(2, "GFLOP"),
        "base_bytes_moved": Q_(32, "MB"),
        "precision": "fp16",
        "reuse": 4,
        "compute_scale": 2,
        "bandwidth_scale": 0.5,
    }
    expected = analyze_roofline(**inputs)
    packet = make_replay_packet(track="cloud", experiment="roofline", inputs=inputs)
    stored = json.loads(json.dumps(packet))
    replayed = replay_experiment(stored)

    assert stored["track_id"] == "cloud"
    assert stored["model_key"] == "v1_11_experiments.roofline"
    assert stored["inputs"]["hardware"]["registry_ref"] == "Hardware.Cloud.H100"
    assert _same_magnitude(replayed.latency, expected.latency)
    assert replayed.bottleneck == expected.bottleneck


def test_every_named_experiment_replays_after_json_roundtrip():
    scenario = TRACK_SCENARIOS["cloud"]
    demand = scenario_gemm_demand("cloud")
    candidates = scenario_candidates("cloud")
    latency_budget, energy_budget, cost_budget = scenario_budgets("cloud")
    common_application = dict(
        hardware=scenario.hardware,
        host_time=scenario.host_time,
        transfer_bytes=scenario.transfer_bytes,
        transfer_fixed_latency=scenario.transfer_fixed_latency,
        launches=scenario.launches,
        postprocess_time=scenario.postprocess_time,
    )
    cases = {
        "roofline": dict(
            hardware=scenario.hardware,
            operations=demand.operations,
            base_bytes_moved=demand.bytes_moved,
            precision=scenario.precision,
            reuse=2,
            compute_scale=1,
            bandwidth_scale=1,
        ),
        "tile_mapping": dict(
            hardware=scenario.hardware,
            m=256, n=256, k=256, tile_m=32, tile_n=32, tile_k=32,
            precision=scenario.precision, fused=True,
            scratchpad_capacity=scenario.local_capacity,
        ),
        "execution_path": dict(
            hardware=scenario.hardware, fallback_hardware=scenario.fallback,
            contract=scenario.contract, operation="gemm", precision=scenario.precision,
            m=256, n=256, k=256,
            quality_observation="illustrative matched-task evidence held constant",
        ),
        "application_path": dict(common_application, kernel_time=Q_(2, "ms")),
        "application_comparison": dict(
            common_application, baseline_kernel_time=Q_(2, "ms"), local_speedup=4,
        ),
        "accelerator_ranking": dict(
            candidates=candidates, fallback_hardware=scenario.fallback,
            operation="gemm", precision=scenario.precision,
            m=256, n=256, k=256,
            host_time=scenario.host_time, transfer_bytes=scenario.transfer_bytes,
            transfer_fixed_latency=scenario.transfer_fixed_latency,
            launches=scenario.launches, postprocess_time=scenario.postprocess_time,
            objective="latency", latency_budget=latency_budget,
            energy_budget=energy_budget, cost_budget=cost_budget,
        ),
    }
    solvers = {
        "roofline": analyze_roofline,
        "tile_mapping": analyze_tile_mapping,
        "execution_path": analyze_execution_path,
        "application_path": compose_application_path,
        "application_comparison": compare_application_speedup,
        "accelerator_ranking": rank_accelerators,
    }

    for experiment, inputs in cases.items():
        expected = solvers[experiment](**inputs)
        packet = make_replay_packet(track="cloud", experiment=experiment, inputs=inputs)
        replayed = replay_experiment(json.loads(json.dumps(packet)))
        assert to_jsonable(replayed) == to_jsonable(expected)
