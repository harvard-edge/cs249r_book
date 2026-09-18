"""Unit tests for physical AI and embodied robotics physics formulas."""

from __future__ import annotations

import pytest
import pint

from mlsysim.core.units import ureg, Q_
from mlsysim.physics.robotics import (
    calc_sensor_to_actuator_latency,
    calc_safe_stopping_distance,
    calc_max_permitted_velocity,
    calc_kinetic_energy,
    calc_seam_acceleration_torque_jump,
    calc_inverted_pendulum_fall_time,
    calc_actuator_thermal_power,
    calc_action_chunk_cadence,
    calc_reflected_inertia,
    calc_spatial_information_age_displacement,
    calc_empirical_testing_exposure,
    calc_dense_voxel_grid_memory,
    calc_3dgs_memory_footprint,
    calc_voltage_droop,
    calc_cbf_safety_margin,
    calc_action_chunk_streaming_amortization,
)


def test_sensor_to_actuator_latency():
    lat = calc_sensor_to_actuator_latency(
        sensor_latency=Q_("16 ms"),
        inference_latency=Q_("25 ms"),
        arbitration_latency=Q_("1 ms"),
        actuator_latency=Q_("2 ms"),
    )
    assert lat.to("ms").magnitude == pytest.approx(44.0)


def test_safe_stopping_distance_tempe_golden():
    # Tempe crash golden numbers: v = 19.2 m/s, t_suppress = 1.2 s, a_max = 8.0 m/s^2
    d_stop = calc_safe_stopping_distance(
        velocity=Q_("19.2 m/s"),
        total_latency=Q_("1.2 s"),
        max_deceleration=Q_("8.0 m/s^2"),
    )
    # reaction = 19.2 * 1.2 = 23.04 m, braking = 19.2^2 / 16.0 = 23.04 m -> total = 46.08 m
    assert d_stop.to("m").magnitude == pytest.approx(46.08, rel=0.01)


def test_safe_stopping_distance_with_margins():
    # AMR golden numbers: v = 1.8 m/s, tau = 65 ms, a = 3.0 m/s^2, loc = 5 cm, margin = 10 cm
    d_stop = calc_safe_stopping_distance(
        velocity=Q_("1.8 m/s"),
        total_latency=Q_("65 ms"),
        max_deceleration=Q_("3.0 m/s^2"),
        loc_margin=Q_("5 cm"),
        physical_margin=Q_("10 cm"),
    )
    # react = 0.117 m, brake = 0.540 m, loc = 0.05 m, margin = 0.10 m -> 0.807 m
    assert d_stop.to("cm").magnitude == pytest.approx(80.7, rel=0.01)


def test_max_permitted_velocity_governor():
    # Given d_stop = 0.807 m with clearance = 0.807 m, max safe velocity should be 1.8 m/s
    v_max = calc_max_permitted_velocity(
        d_clear=Q_("0.807 m"),
        total_latency=Q_("65 ms"),
        max_deceleration=Q_("3.0 m/s^2"),
        loc_margin=Q_("5 cm"),
        physical_margin=Q_("10 cm"),
    )
    assert v_max.to("m/s").magnitude == pytest.approx(1.8, rel=0.01)

    # When clearance is less than margins, v_max should be zero
    v_zero = calc_max_permitted_velocity(
        d_clear=Q_("10 cm"),
        total_latency=Q_("65 ms"),
        max_deceleration=Q_("3.0 m/s^2"),
        loc_margin=Q_("5 cm"),
        physical_margin=Q_("10 cm"),
    )
    assert v_zero.to("m/s").magnitude == pytest.approx(0.0)


def test_kinetic_energy():
    # 150 kg AMR at 1.8 m/s: E_k = 0.5 * 150 * 1.8^2 = 243 J
    e = calc_kinetic_energy(mass=Q_("150 kg"), velocity=Q_("1.8 m/s"))
    assert e.to("J").magnitude == pytest.approx(243.0, rel=0.01)


def test_seam_acceleration_torque_jump():
    # N = 50, J_rotor = 1e-4 kg*m^2 -> J_refl = 2500 * 1e-4 = 0.25 kg*m^2
    # delta_accel = 200 rad/s^2 -> tau = 0.25 * 200 = 50 N*m
    tau = calc_seam_acceleration_torque_jump(
        gear_ratio=50.0,
        rotor_inertia=Q_("1e-4 kg * m^2"),
        delta_accel=Q_("200 rad / s^2"),
    )
    assert tau.to("N * m").magnitude == pytest.approx(50.0, rel=0.01)


def test_inverted_pendulum_fall_time():
    # Humanoid leg length L = 0.9 m: tau_0 = sqrt(0.9 / 9.80665) approx 0.3029 s
    tau_0 = calc_inverted_pendulum_fall_time(effective_length=Q_("0.9 m"))
    assert tau_0.to("ms").magnitude == pytest.approx(302.9, rel=0.01)

    # With theta_0 = 1 deg (0.01745 rad), theta_fall = 15 deg (0.2618 rad):
    # ln(2 * 15 / 1) = ln(30) approx 3.4012
    # t_fall = 0.3029 * 3.4012 approx 1.030 s
    t_fall = calc_inverted_pendulum_fall_time(
        effective_length=Q_("0.9 m"),
        theta_0=Q_("1 deg"),
        theta_fall=Q_("15 deg"),
    )
    assert t_fall.to("s").magnitude == pytest.approx(1.03, rel=0.02)


def test_actuator_thermal_power():
    p = calc_actuator_thermal_power(current=Q_("10 A"), resistance=Q_("0.5 ohm"), duty_cycle=0.8)
    # 100 * 0.5 * 0.8 = 40 W
    assert p.to("W").magnitude == pytest.approx(40.0)


def test_action_chunk_cadence():
    cadence = calc_action_chunk_cadence(
        chunk_horizon_steps=32,
        control_loop_hz=Q_("50 Hz"),
        brain_inference_hz=Q_("5 Hz"),
    )
    # step_period = 20 ms, chunk = 640 ms, brain_period = 200 ms -> headroom = 3.2
    assert cadence["chunk_duration"].to("ms").magnitude == pytest.approx(640.0)
    assert cadence["brain_period"].to("ms").magnitude == pytest.approx(200.0)
    assert cadence["headroom_factor"] == pytest.approx(3.2)


def test_spatial_information_age_displacement():
    dx = calc_spatial_information_age_displacement(
        velocity=Q_("1.5 m/s"),
        information_age=Q_("60 ms"),
    )
    assert dx.to("cm").magnitude == pytest.approx(9.0)


def test_empirical_testing_exposure():
    # Target failure rate: 10^-6 per hour
    # 95% confidence: T = -ln(0.05) / 10^-6 = 2.99573e6 hours ≈ 342.0 years
    hours = calc_empirical_testing_exposure(target_hazard_rate=Q_("1e-6 / hr"), confidence=0.95)
    assert hours.to("hr").magnitude == pytest.approx(2.99573e6, rel=0.01)
    years = hours.to("hr").magnitude / (24 * 365.25)
    assert years == pytest.approx(341.73, rel=0.01)


def test_dense_voxel_grid_memory():
    # 512^3 * 4 bytes = 536,870,912 bytes = 512 MiB = 536.87 MB
    mem = calc_dense_voxel_grid_memory(voxels_per_dim=512, bytes_per_voxel=4)
    assert mem.to("MiB").magnitude == pytest.approx(512.0)
    assert mem.to("MB").magnitude == pytest.approx(536.87, rel=0.01)


def test_3dgs_memory_footprint():
    # 1,000,000 gaussians * 56 bytes = 56,000,000 bytes = 56 MB
    mem = calc_3dgs_memory_footprint(num_gaussians=1_000_000, bytes_per_gaussian=56)
    assert mem.to("MB").magnitude == pytest.approx(56.0)


def test_voltage_droop():
    # 50 A draw through 40 mOhm internal resistance = 2.0 V droop
    droop = calc_voltage_droop(current=Q_("50 A"), internal_resistance=Q_("40 mohm"))
    assert droop.to("V").magnitude == pytest.approx(2.0)


def test_cbf_safety_margin():
    # Safe candidate
    res_safe = calc_cbf_safety_margin(h_val=1.0, l_f_h=-0.5, l_g_h=1.0, u_cmd=2.0, alpha_coeff=1.0)
    assert res_safe["psi"] == pytest.approx(2.5)
    assert res_safe["is_safe"] is True

    # Unsafe candidate
    res_unsafe = calc_cbf_safety_margin(h_val=1.0, l_f_h=-0.5, l_g_h=1.0, u_cmd=-3.0, alpha_coeff=1.0)
    assert res_unsafe["psi"] == pytest.approx(-2.5)
    assert res_unsafe["is_safe"] is False


def test_action_chunk_streaming_amortization():
    # OpenVLA-7B (14 GB) on Jetson AGX Orin (204.8 GB/s), eta=0.70, chunk H=16
    res = calc_action_chunk_streaming_amortization(
        model_weight_memory=Q_("14 GB"),
        memory_bandwidth=Q_("204.8 GB/s"),
        chunk_horizon=16,
        bus_efficiency=0.70,
    )
    # b_sustained = 143.36 GB/s
    assert res["b_sustained"].to("GB/s").magnitude == pytest.approx(143.36, rel=0.01)
    # t_stream = 14 / 143.36 = 0.097656 s = 97.66 ms
    assert res["t_stream"].to("ms").magnitude == pytest.approx(97.66, rel=0.01)
    # f_single = 10.24 Hz
    assert res["f_single"].to("Hz").magnitude == pytest.approx(10.24, rel=0.01)
    # tau_step = 6.10 ms
    assert res["tau_step"].to("ms").magnitude == pytest.approx(6.10, rel=0.01)
    # f_effective = 163.84 Hz
    assert res["f_effective"].to("Hz").magnitude == pytest.approx(163.84, rel=0.01)
