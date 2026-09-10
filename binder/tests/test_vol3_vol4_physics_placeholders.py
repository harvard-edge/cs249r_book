"""Tests for Volume 3 (Agents) and Volume 4 (Physical AI) physics functions."""

import pytest
from mlsysim.core.units import ureg, second, millisecond, meter, watt
from mlsysim.physics.agents import (
    calc_trajectory_step_time,
    calc_trajectory_reliability,
    calc_radix_cache_effective_latency,
    calc_test_time_compute_cost,
    calc_multi_agent_coordination_overhead,
)
from mlsysim.physics.robotics import (
    calc_sensor_to_actuator_latency,
    calc_safe_stopping_distance,
    calc_actuator_thermal_power,
    calc_action_chunk_cadence,
    calc_reflected_inertia,
)


def test_trajectory_step_time():
    thinking = 1.5 * second
    tool = 500 * millisecond
    verification = 200 * millisecond
    total = calc_trajectory_step_time(thinking, tool, verification)
    assert total.to(second).magnitude == pytest.approx(2.2)


def test_trajectory_reliability():
    # 99% step success over 10 steps without verifier
    prob = calc_trajectory_reliability(0.99, 10)
    assert prob == pytest.approx(0.99**10)

    # With verifier repairing 80% of errors
    prob_repaired = calc_trajectory_reliability(0.95, 10, verifier_recovery_rate=0.8)
    assert prob_repaired > 0.95**10

    # Invalid range should raise
    with pytest.raises(ValueError):
        calc_trajectory_reliability(1.5, 10)


def test_radix_cache_effective_latency():
    prompt_tokens = 4000
    shared_prefix = 3000
    prefill_rate = 1000 * (1 / second)
    decode_rate = 50 * (1 / second)

    latency = calc_radix_cache_effective_latency(
        prompt_tokens=prompt_tokens,
        shared_prefix_tokens=shared_prefix,
        prefill_rate=prefill_rate,
        decode_rate=decode_rate,
        output_tokens=100,
    )
    # Uncached: 1000 tokens @ 1000 tok/s = 1.0s
    # Decode: 100 tokens @ 50 tok/s = 2.0s
    # Total: 3.0s
    assert latency.to(second).magnitude == pytest.approx(3.0)


def test_test_time_compute_cost():
    base_cost = 10.0  # e.g., 10 FLOPs or units
    num_samples = 8
    verifier_cost = 2.0
    agg_cost = 5.0

    total = calc_test_time_compute_cost(base_cost, num_samples, verifier_cost, agg_cost)
    assert total == pytest.approx(8 * 10.0 + 8 * 2.0 + 5.0)


def test_multi_agent_coordination_overhead():
    res = calc_multi_agent_coordination_overhead(num_agents=5, avg_message_tokens=200)
    # Coordinator: (5-1) * 2 * 200 = 1600 tokens
    assert res["coordinator_tokens"] == 1600
    # Pairwise: 5 * 4 * 200 = 4000 tokens
    assert res["pairwise_tokens"] == 4000


def test_sensor_to_actuator_latency():
    sensor = 10 * millisecond
    infer = 40 * millisecond
    arb = 2 * millisecond
    actuator = 8 * millisecond

    total = calc_sensor_to_actuator_latency(sensor, infer, arb, actuator)
    assert total.to(millisecond).magnitude == pytest.approx(60.0)


def test_safe_stopping_distance():
    v = 2.0 * (meter / second)
    tau = 60 * millisecond
    a_max = 4.0 * (meter / (second**2))

    d_stop = calc_safe_stopping_distance(v, tau, a_max)
    # Reaction: 2.0 * 0.060 = 0.12m
    # Braking: (2.0^2) / (2 * 4.0) = 4 / 8 = 0.5m
    # Total: 0.62m
    assert d_stop.to(meter).magnitude == pytest.approx(0.62)


def test_actuator_thermal_power():
    i = 10.0 * ureg.ampere
    r = 0.5 * ureg.ohm
    power = calc_actuator_thermal_power(i, r, duty_cycle=0.5)
    # I^2 * R * 0.5 = 100 * 0.5 * 0.5 = 25 W
    assert power.to(watt).magnitude == pytest.approx(25.0)


def test_action_chunk_cadence():
    res = calc_action_chunk_cadence(
        chunk_horizon_steps=50,
        control_loop_hz=500 * (1 / second),
        brain_inference_hz=5 * (1 / second),
    )
    # Chunk duration: 50 / 500 = 0.1s = 100ms
    # Brain period: 1 / 5 = 0.2s = 200ms
    # Headroom: 0.1 / 0.2 = 0.5
    assert res["chunk_duration"].to(millisecond).magnitude == pytest.approx(100.0)
    assert res["brain_period"].to(millisecond).magnitude == pytest.approx(200.0)
    assert res["headroom_factor"] == pytest.approx(0.5)


def test_reflected_inertia():
    j_rotor = 0.001 * (ureg.kilogram * (meter**2))
    gear_ratio = 10.0
    j_ref = calc_reflected_inertia(gear_ratio, j_rotor)
    # N^2 * J = 100 * 0.001 = 0.1 kg*m^2
    assert j_ref.magnitude == pytest.approx(0.1)
