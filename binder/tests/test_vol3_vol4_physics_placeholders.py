"""Tests for Volume 3 (Agents) and Volume 4 (Physical AI) physics functions."""

import pytest
from mlsysim.core.units import ureg, second, millisecond, meter
from mlsysim.physics.agents import (
    calc_kv_cache_bytes_per_token,
)
from mlsysim.physics.robotics import (
    calc_safe_stopping_distance,
    calc_max_permitted_velocity,
    calc_kinetic_energy,
    calc_seam_acceleration_torque_jump,
    calc_inverted_pendulum_fall_time,
    calc_reflected_inertia,
)


def test_kv_cache_bytes_per_token():
    bpt = calc_kv_cache_bytes_per_token(n_layers=80, n_kv_heads=8, head_dim=128, bytes_per_elem=2)
    assert bpt.to("KiB").magnitude == pytest.approx(320.0)


def test_safe_stopping_distance():
    v = 2.0 * (meter / second)
    tau = 60 * millisecond
    a_max = 4.0 * (meter / (second**2))

    d_stop = calc_safe_stopping_distance(v, tau, a_max)
    # Reaction: 2.0 * 0.060 = 0.12m
    # Braking: (2.0^2) / (2 * 4.0) = 4 / 8 = 0.5m
    # Total: 0.62m
    assert d_stop.to(meter).magnitude == pytest.approx(0.62)


def test_reflected_inertia():
    j_rotor = 0.001 * (ureg.kilogram * (meter**2))
    gear_ratio = 10.0
    j_ref = calc_reflected_inertia(gear_ratio, j_rotor)
    # N^2 * J = 100 * 0.001 = 0.1 kg*m^2
    assert j_ref.magnitude == pytest.approx(0.1)


def test_safe_stopping_distance_with_margins():
    v = 1.8 * (meter / second)
    tau = 65 * millisecond
    a_max = 3.0 * (meter / (second**2))
    loc = 0.05 * meter
    margin = 0.10 * meter

    d_stop = calc_safe_stopping_distance(v, tau, a_max, loc_margin=loc, physical_margin=margin)
    # react = 0.117m, brake = 0.540m, loc = 0.05m, margin = 0.10m -> 0.807m
    assert d_stop.to(meter).magnitude == pytest.approx(0.807)


def test_max_permitted_velocity():
    d_clear = 0.807 * meter
    tau = 65 * millisecond
    a_max = 3.0 * (meter / (second**2))
    loc = 0.05 * meter
    margin = 0.10 * meter

    v_max = calc_max_permitted_velocity(d_clear, tau, a_max, loc_margin=loc, physical_margin=margin)
    assert v_max.to(meter / second).magnitude == pytest.approx(1.8)


def test_kinetic_energy():
    mass = 150 * ureg.kilogram
    velocity = 1.8 * (meter / second)
    ke = calc_kinetic_energy(mass, velocity)
    # 0.5 * 150 * 1.8^2 = 243 J
    assert ke.to(ureg.joule).magnitude == pytest.approx(243.0)


def test_seam_acceleration_torque_jump():
    gear_ratio = 50.0
    rotor_inertia = 1e-4 * (ureg.kilogram * (meter**2))
    delta_accel = 200.0 * (ureg.radian / (second**2))
    tau = calc_seam_acceleration_torque_jump(gear_ratio, rotor_inertia, delta_accel)
    # (50^2 * 1e-4) * 200 = 2500 * 1e-4 * 200 = 0.25 * 200 = 50 N*m
    assert tau.to(ureg.newton * meter).magnitude == pytest.approx(50.0)


def test_inverted_pendulum_fall_time():
    l_eff = 0.9 * meter
    tau_0 = calc_inverted_pendulum_fall_time(l_eff)
    # sqrt(0.9 / 9.80665) approx 0.3029 s = 302.9 ms
    assert tau_0.to(second).magnitude == pytest.approx(0.3029, rel=0.01)

