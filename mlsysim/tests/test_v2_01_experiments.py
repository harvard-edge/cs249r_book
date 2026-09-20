"""Causal tests for the Volume II introduction experiments."""

import json

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v2_01_experiments import (
    TRACK_PROFILES,
    compare_failure_semantics,
    evaluate_c3_scaling,
    evaluate_machine_boundary,
    evaluate_partition_policy,
    get_track_profile,
    get_track_scenario,
    restore_quantities,
    serialize_evaluation,
)


def test_four_tracks_preserve_deployment_and_backend_semantics():
    assert set(TRACK_PROFILES) == {"tinyml", "mobile", "edge", "cloud"}
    for track_id in TRACK_PROFILES:
        profile = get_track_profile(track_id)
        assert profile.scaling_coupling == "coupled_job"
        assert "training" in profile.scaling_system
    assert get_track_profile("tinyml").deployed_coupling == "independent_devices"
    assert get_track_profile("mobile").deployed_coupling == "independent_devices"
    assert get_track_profile("edge").deployed_coupling == "independent_devices"
    assert get_track_profile("cloud").deployed_coupling == "coupled_job"


@pytest.mark.parametrize("track_id", ("tinyml", "mobile", "edge", "cloud"))
def test_every_track_scenario_runs_all_four_experiments(track_id):
    scenario = get_track_scenario(track_id)
    boundary = evaluate_machine_boundary(
        model_state=scenario.model_state,
        memory_per_machine=scenario.memory_per_machine,
        required_rate=scenario.required_rate,
        rate_per_machine=scenario.rate_per_machine,
        coupling=scenario.profile.deployed_coupling,
        machine_count=scenario.machine_count_options[-1],
    )
    scaling = evaluate_c3_scaling(
        workers=scenario.worker_options[1],
        single_worker_compute_time=scenario.single_worker_compute_time,
        communication_payload=scenario.communication_payload,
        effective_bandwidth=scenario.bandwidth_options["planned"],
        communication_startup=scenario.communication_startup,
        coordination_base=scenario.coordination_base,
        coordination_per_added_worker=scenario.coordination_options["planned"],
        useful_work_per_step=scenario.useful_work_per_step,
        overlap_fraction=scenario.overlap_fraction,
    )
    reliability = compare_failure_semantics(
        nodes=scenario.fleet_size_options[1],
        component_mtbf=scenario.component_mtbf,
        observation_horizon=scenario.observation_horizon,
        recovery_time=scenario.recovery_options["planned"],
    )
    partition = evaluate_partition_policy(
        policy="serve_last_confirmed",
        request_rate=scenario.request_rate,
        observation_horizon=scenario.observation_horizon,
        partition_duration=scenario.partition_duration_options["extended"],
        partitioned_fraction=scenario.partitioned_fraction_options["regional cohort"],
        last_confirmed_model_age=scenario.last_confirmed_model_age,
    )

    assert boundary.memory_margin < Q_(0, "byte")
    assert scaling.useful_throughput > Q_(0, "count/second")
    assert reliability.probability_any_node_event > 0
    assert partition.stale_requests > Q_(0, "count")


def test_independent_device_memory_never_pools_across_the_fleet():
    inputs = dict(
        model_state=Q_(2, "GiB"),
        memory_per_machine=Q_(1, "GiB"),
        required_rate=Q_(4, "count/second"),
        rate_per_machine=Q_(5, "count/second"),
        coupling="independent_devices",
    )
    one_device = evaluate_machine_boundary(machine_count=1, **inputs)
    million_devices = evaluate_machine_boundary(machine_count=1_000_000, **inputs)

    assert not one_device.feasible
    assert not million_devices.feasible
    assert one_device.local_memory_fits == million_devices.local_memory_fits
    assert million_devices.minimum_coupled_machines is None
    assert million_devices.machines_for_memory is None
    assert million_devices.remedy == "reduce per-device state or demand"


def test_coupled_job_can_cross_memory_and_rate_boundaries_with_enough_workers():
    inputs = dict(
        model_state=Q_(36, "GiB"),
        memory_per_machine=Q_(16, "GiB"),
        required_rate=Q_(90, "count/second"),
        rate_per_machine=Q_(40, "count/second"),
        coupling="coupled_job",
    )
    too_small = evaluate_machine_boundary(machine_count=2, **inputs)
    feasible = evaluate_machine_boundary(machine_count=3, **inputs)

    assert too_small.minimum_coupled_machines == 3
    assert too_small.distribution_required
    assert not too_small.feasible
    assert feasible.feasible


def test_coupled_machine_count_changes_effective_properties_preserving_totals():
    inputs = dict(
        model_state=Q_(180, "GiB"),
        memory_per_machine=Q_(80, "GiB"),
        required_rate=Q_(120, "count/second"),
        rate_per_machine=Q_(50, "count/second"),
        coupling="coupled_job",
    )
    single = evaluate_machine_boundary(machine_count=1, **inputs)
    scaled = evaluate_machine_boundary(machine_count=3, **inputs)

    assert scaled.model_state == single.model_state
    assert scaled.required_rate == single.required_rate
    assert scaled.memory_per_machine == single.memory_per_machine
    assert scaled.rate_per_machine == single.rate_per_machine
    assert scaled.memory_margin == single.memory_margin

    assert single.effective_state_per_machine.to("GiB").magnitude == pytest.approx(180.0)
    assert scaled.effective_state_per_machine.to("GiB").magnitude == pytest.approx(60.0)

    assert single.effective_required_rate_per_machine.to("count/second").magnitude == pytest.approx(120.0)
    assert scaled.effective_required_rate_per_machine.to("count/second").magnitude == pytest.approx(40.0)

    assert single.effective_memory_margin.to("GiB").magnitude == pytest.approx(-100.0)
    assert scaled.effective_memory_margin.to("GiB").magnitude == pytest.approx(20.0)

    assert scaled.effective_state_per_machine.units == scaled.model_state.units
    assert scaled.effective_required_rate_per_machine.units == scaled.required_rate.units
    assert scaled.effective_memory_margin.units == scaled.memory_per_machine.units


def test_independent_device_count_does_not_change_effective_properties():
    inputs = dict(
        model_state=Q_(768, "KiB"),
        memory_per_machine=Q_(512, "KiB"),
        required_rate=Q_(10, "count/second"),
        rate_per_machine=Q_(20, "count/second"),
        coupling="independent_devices",
    )
    one = evaluate_machine_boundary(machine_count=1, **inputs)
    many = evaluate_machine_boundary(machine_count=16, **inputs)

    assert many.effective_state_per_machine == one.effective_state_per_machine == many.model_state
    assert many.effective_required_rate_per_machine == one.effective_required_rate_per_machine == many.required_rate
    assert many.effective_memory_margin == one.effective_memory_margin == many.memory_margin

    assert many.effective_state_per_machine.units == many.model_state.units
    assert many.effective_required_rate_per_machine.units == many.required_rate.units
    assert many.effective_memory_margin.units == many.memory_margin.units


def _c3(workers, **overrides):
    inputs = dict(
        workers=workers,
        single_worker_compute_time=Q_(64, "second"),
        communication_payload=Q_(2, "GB"),
        effective_bandwidth=Q_(4, "GB/second"),
        communication_startup=Q_(0.01, "second"),
        coordination_base=Q_(0.02, "second"),
        coordination_per_added_worker=Q_(0.01, "second"),
        useful_work_per_step=Q_(1024, "count"),
        overlap_fraction=0.25,
    )
    inputs.update(overrides)
    return evaluate_c3_scaling(**inputs)


def test_c3_scaling_exposes_benefit_and_coordination_knee():
    one = _c3(1)
    eight = _c3(8)
    huge = _c3(8192)

    assert eight.useful_throughput > one.useful_throughput
    assert huge.useful_throughput < eight.useful_throughput
    assert one.scaling_efficiency == pytest.approx(1.0)
    assert huge.dominant_term == "coordination"
    assert huge.scaling_efficiency < eight.scaling_efficiency


def test_slower_network_changes_communication_but_not_compute():
    fast = _c3(16, effective_bandwidth=Q_(8, "GB/second"))
    slow = _c3(16, effective_bandwidth=Q_(1, "GB/second"))

    assert slow.communication_time > fast.communication_time
    assert slow.useful_throughput < fast.useful_throughput
    assert slow.compute_time == fast.compute_time
    assert slow.coordination_time == fast.coordination_time


def test_failure_event_has_different_consequence_by_workload_coupling():
    result = compare_failure_semantics(
        nodes=1000,
        component_mtbf=Q_(100_000, "hour"),
        observation_horizon=Q_(24, "hour"),
        recovery_time=Q_(30, "minute"),
    )

    assert result.probability_any_node_event > result.per_node_event_probability
    assert result.coupled_job_interruption_probability == result.probability_any_node_event
    assert result.independent_expected_affected_devices > 0
    expected_availability = 100_000 / (100_000 + 0.5)
    expected_unavailable_device_hours = 1000 * (1 - expected_availability) * 24
    assert result.independent_expected_unavailable_device_time.m_as("hour") == pytest.approx(
        expected_unavailable_device_hours
    )
    assert result.independent_available_capacity_fraction == pytest.approx(expected_availability)
    assert result.independent_unavailable_capacity_fraction == pytest.approx(
        1 - expected_availability
    )
    assert (
        result.independent_available_capacity_fraction
        + result.independent_unavailable_capacity_fraction
    ) == pytest.approx(1.0)
    assert result.independent_available_capacity_fraction > 0.99
    assert result.coupled_expected_interruptions == pytest.approx(0.24)


def test_recovery_changes_capacity_loss_without_changing_failure_probability():
    inputs = dict(
        nodes=10_000,
        component_mtbf=Q_(50_000, "hour"),
        observation_horizon=Q_(8, "hour"),
    )
    quick = compare_failure_semantics(recovery_time=Q_(1, "minute"), **inputs)
    slow = compare_failure_semantics(recovery_time=Q_(1, "hour"), **inputs)

    assert quick.probability_any_node_event == slow.probability_any_node_event
    assert quick.expected_node_failure_events == slow.expected_node_failure_events
    assert slow.independent_available_capacity_fraction < quick.independent_available_capacity_fraction


def test_partition_policy_trades_unavailable_requests_for_stale_requests():
    inputs = dict(
        request_rate=Q_(100, "count/second"),
        observation_horizon=Q_(10, "minute"),
        partition_duration=Q_(2, "minute"),
        partitioned_fraction=0.25,
        last_confirmed_model_age=Q_(1, "minute"),
    )
    wait = evaluate_partition_policy(policy="wait_for_fresh", **inputs)
    stale = evaluate_partition_policy(policy="serve_last_confirmed", **inputs)

    assert wait.affected_requests.m_as("count") == pytest.approx(3000)
    assert wait.unavailable_requests == wait.affected_requests
    assert wait.stale_requests.m_as("count") == 0
    assert stale.stale_requests == stale.affected_requests
    assert stale.unavailable_requests.m_as("count") == 0
    assert stale.maximum_model_age.m_as("minute") == pytest.approx(3)
    assert wait.total_requests == stale.total_requests


def test_partition_duration_changes_consequences_but_not_total_demand():
    inputs = dict(
        policy="serve_last_confirmed",
        request_rate=Q_(20, "count/second"),
        observation_horizon=Q_(1, "hour"),
        partitioned_fraction=0.5,
    )
    short = evaluate_partition_policy(partition_duration=Q_(1, "minute"), **inputs)
    long = evaluate_partition_policy(partition_duration=Q_(10, "minute"), **inputs)

    assert short.total_requests == long.total_requests
    assert long.stale_requests > short.stale_requests
    assert long.maximum_model_age > short.maximum_model_age


def test_evaluation_serialization_preserves_exact_inputs_and_units():
    inputs = dict(
        policy="serve_last_confirmed",
        request_rate=Q_(20, "count/second"),
        observation_horizon=Q_(1, "hour"),
        partition_duration=Q_(10, "minute"),
        partitioned_fraction=0.5,
        last_confirmed_model_age=Q_(2, "minute"),
    )
    result = evaluate_partition_policy(**inputs)
    record = serialize_evaluation(inputs=inputs, result=result)
    frozen = json.loads(json.dumps(record, allow_nan=False))
    restored_inputs = restore_quantities(frozen["inputs"])

    assert set(frozen) == {"inputs", "outputs"}
    assert set(frozen["inputs"]) == set(inputs)
    assert frozen["inputs"]["request_rate"]["unit"] == "count / s"
    assert restored_inputs["observation_horizon"].to("minute").magnitude == pytest.approx(60)
    assert evaluate_partition_policy(**restored_inputs) == result


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: get_track_profile("wearable"), "unknown track_id"),
        (
            lambda: evaluate_partition_policy(
                policy="wait_for_fresh",
                request_rate=Q_(1, "count/second"),
                observation_horizon=Q_(1, "minute"),
                partition_duration=Q_(2, "minute"),
                partitioned_fraction=0.5,
            ),
            "cannot exceed",
        ),
        (
            lambda: _c3(0),
            "at least 1",
        ),
    ],
)
def test_invalid_scenarios_fail_loudly(call, message):
    with pytest.raises(ValueError, match=message):
        call()
