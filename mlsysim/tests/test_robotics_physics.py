"""Unit tests for physical AI and embodied robotics physics formulas."""

from __future__ import annotations

import math
import pytest
import pint

from mlsysim.core.units import ureg, Q_
from mlsysim.physics.robotics import (
    calc_sensor_to_actuator_latency,
    calc_axi_fifo_contention,
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
    calc_watchdog_lease_bound,
    calc_canfd_bus_utilization,
    calc_ethercat_cycle_time,
    calc_contact_force,
    calc_contact_yield_deadline,
    calc_optimal_gear_ratio,
    calc_geared_joint_acceleration,
    calc_inductive_voltage_droop,
    calc_alpha_power_gate_delay_stretch,
    calc_clopper_pearson_zero_failure_bound,
    calc_zero_failure_sample_size,
    calc_demonstration_collection_yield,
    calc_harmonic_drive_fatigue_consumption,
    calc_transient_impact_force,
    calc_tripwire_contact_force_accumulation,
    calc_crypto_auth_deadline,
    calc_coulomb_stiction_deadband,
    calc_shielded_system_hazard_rate,
    calc_cbf_qp_orthogonal_projection,
    calc_quintic_blend_duration,
    calc_thermal_cooling_recovery_time,
    calc_passive_compliance_stiffness,
    calc_process_containment_time_to_breach,
    calc_teleop_ingestion_budget,
    calc_tsdf_voxel_grid_budget,
    calc_target_evidence_horizon,
    calc_intent_drift_lease,
    calc_process_thermal_runaway_lease,
)


def test_axi_fifo_contention_distinguishes_tile_from_transaction_and_bypass():
    result = calc_axi_fifo_contention(
        tile_bytes=256 * 1024,
        camera_bytes=64 * 1024,
        telemetry_bytes=32 * 1024,
        safety_read_bytes=64 * 1024,
        transaction_bytes=4 * 1024,
        bandwidth=Q_("12.8 GB/s"),
        bank_switches=3,
        bank_switch_time=Q_("28 ns"),
        compute_time=Q_("95 us"),
    )
    assert result["tile_transactions"] == 64
    assert result["queued_transactions"] == 88
    assert result["queue_fifo"].to("us").magnitude == pytest.approx(28.244)
    assert result["own_read"].to("us").magnitude == pytest.approx(5.120)
    assert result["total_fifo"].to("us").magnitude == pytest.approx(128.364)
    assert result["total_priority_bypass"].to("us").magnitude == pytest.approx(100.44)
    assert result["total_priority_bypass"] < result["total_fifo"]
    with pytest.raises(ValueError):
        calc_axi_fifo_contention(
            tile_bytes=256 * 1024,
            camera_bytes=64 * 1024,
            telemetry_bytes=32 * 1024,
            safety_read_bytes=64 * 1024,
            transaction_bytes=4 * 1024 + 1,
            bandwidth=Q_("12.8 GB/s"),
            bank_switches=3,
            bank_switch_time=Q_("28 ns"),
            compute_time=Q_("95 us"),
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
    # Tempe-inspired illustration: rounded speed, NTSB one-second suppression, assumed 8.0 m/s^2 braking.
    d_stop = calc_safe_stopping_distance(
        velocity=Q_("19.2 m/s"),
        total_latency=Q_("1.0 s"),
        max_deceleration=Q_("8.0 m/s^2"),
    )
    # reaction = 19.2 * 1.0 = 19.2 m, braking = 19.2^2 / 16.0 = 23.04 m -> total = 42.24 m
    assert d_stop.to("m").magnitude == pytest.approx(42.24, rel=0.01)


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
    # 163.84 waypoint equivalents per second; fresh chunks remain bound by t_stream.
    assert res["f_waypoint_equivalent"].to("Hz").magnitude == pytest.approx(163.84, rel=0.01)
    assert res["f_effective"] == res["f_waypoint_equivalent"]


def test_watchdog_lease_bound():
    # Chapter 4 AMR golden: m = 300 kg, v0 = 1.5 m/s, a_max = 2.0 m/s^2, d_clear = 0.6939 m
    res = calc_watchdog_lease_bound(
        clearance_distance=Q_("0.6939 m"),
        initial_velocity=Q_("1.5 m/s"),
        max_deceleration=Q_("2.0 m/s^2"),
    )
    # d_brake = 1.5^2 / 4 = 0.5625 m
    assert res["d_brake"].to("m").magnitude == pytest.approx(0.5625, rel=1e-3)
    # d_drift = 0.6939 - 0.5625 = 0.1314 m
    assert res["d_drift_allowable"].to("m").magnitude == pytest.approx(0.1314, rel=1e-3)
    # t_lease = 0.1314 / 1.5 = 0.0876 s = 87.6 ms
    assert res["t_total_delay_max"].to("ms").magnitude == pytest.approx(87.6, rel=1e-2)
    assert res["t_lease_max"] == res["t_total_delay_max"]


def test_canfd_bus_utilization():
    # Chapter 4 CAN-FD 8 axes, 64-byte payload
    res = calc_canfd_bus_utilization(
        num_axes=8,
        payload_bytes=64,
        baud_arbitration=Q_("1.0 Mbps"),
        baud_data=Q_("5.0 Mbps"),
        cycle_time=Q_("1.0 ms"),
        interframe_spacing=Q_("6.0 us"),
        overhead_bits_arb=32,
    )
    # t_tx = 32 us + 102.4 us + 6.0 us = 140.4 us
    assert res["t_tx"].to("us").magnitude == pytest.approx(140.4, rel=1e-2)
    # t_total = 8 * 140.4 = 1123.2 us
    assert res["t_total_tx"].to("us").magnitude == pytest.approx(1123.2, rel=1e-2)
    # utilization = 1.1232 (>1.0 -> unschedulable)
    assert res["utilization"] == pytest.approx(1.1232, rel=1e-2)
    assert res["is_schedulable"] is False


def test_ethercat_cycle_time():
    # Chapter 4 EtherCAT 8 axes, 32 bytes/axis, 100 Mbps Fast Ethernet
    res = calc_ethercat_cycle_time(
        num_axes=8,
        bytes_per_axis=32,
        baud_rate=Q_("100 Mbps"),
        header_bytes=64,
        hop_delay=Q_("0.6 us"),
        cycle_time=Q_("1.0 ms"),
    )
    # total bytes = 64 + 256 = 320 bytes = 2560 bits -> t_tx = 25.6 us
    assert res["t_tx"].to("us").magnitude == pytest.approx(25.6, rel=1e-2)
    # t_forward = 8 * 0.6 = 4.8 us
    assert res["t_forward"].to("us").magnitude == pytest.approx(4.8, rel=1e-2)
    # total = 30.4 us
    assert res["t_cycle_total"].to("us").magnitude == pytest.approx(30.4, rel=1e-2)
    # margin = 1000 - 30.4 = 969.6 us
    assert res["t_margin"].to("us").magnitude == pytest.approx(969.6, rel=1e-2)
    # utilization = 0.0304 (3.04%)
    assert res["utilization"] == pytest.approx(0.0304, rel=1e-2)
    assert res["is_schedulable"] is True


def test_contact_force_and_yield_deadline():
    # Chapter 1 arm approaching steel fixture: k = 2.5e5 N/m, v = 0.20 m/s, F_yield = 1500 N
    t_deadline = calc_contact_yield_deadline(
        yield_force=Q_("1500 N"),
        stiffness=Q_("250000 N/m"),
        approach_velocity=Q_("0.20 m/s"),
    )
    # t_deadline = 1500 / (2.5e5 * 0.20) = 1500 / 50000 = 0.03 s = 30 ms
    assert t_deadline.to("ms").magnitude == pytest.approx(30.0, rel=1e-2)

    # Displacement of 15 mm -> F = 2.5e5 * 0.015 = 3750 N
    f_contact = calc_contact_force(stiffness=Q_("250000 N/m"), displacement=Q_("15 mm"))
    assert f_contact.to("N").magnitude == pytest.approx(3750.0, rel=1e-2)


def test_optimal_gear_ratio_and_acceleration():
    # Chapter 2 knee joint: J_load = 0.18 kg*m^2, J_rotor = 1.2e-4 kg*m^2, tau_motor = 4.0 N*m
    j_load = Q_("0.18 kg*m^2")
    j_rotor = Q_("1.2e-4 kg*m^2")
    tau_motor = Q_("4.0 N*m")

    n_opt = calc_optimal_gear_ratio(j_load, j_rotor)
    # sqrt(0.18 / 1.2e-4) = sqrt(1500) ≈ 38.73
    assert n_opt == pytest.approx(38.73, rel=1e-2)

    alpha_opt = calc_geared_joint_acceleration(38.73, tau_motor, j_load, j_rotor)
    # alpha = (38.73 * 4) / (0.18 + 38.73^2 * 1.2e-4) = 154.92 / 0.36 = 430.3 rad/s^2
    assert alpha_opt.to("rad/s^2").magnitude == pytest.approx(430.3, rel=1e-2)


def test_inductive_voltage_droop_and_gate_delay():
    # Chapter 1 humanoid DC bus droop: L = 25 uH, dI/dt = 500 A/ms = 5e5 A/s
    dv = calc_inductive_voltage_droop(Q_("25 uH"), Q_("500 A/ms"))
    assert dv.to("V").magnitude == pytest.approx(12.5, rel=1e-2)

    # Chapter 13 gate delay: V_nom = 0.85 V, V_drooped = 0.712 V, V_th = 0.42 V, alpha = 1.3
    res_gate = calc_alpha_power_gate_delay_stretch(
        nominal_voltage=Q_("0.85 V"),
        drooped_voltage=Q_("0.712 V"),
        threshold_voltage=Q_("0.42 V"),
        alpha=1.3,
    )
    assert res_gate["stretch_ratio"] == pytest.approx(1.38, rel=0.02)
    assert res_gate["delay_expansion_percent"] == pytest.approx(38.0, rel=0.05)


def test_clopper_pearson_and_sample_size():
    # Chapter 7: 20 successes in 20 trials, 95% confidence -> p_L = 0.05^(1/20) ≈ 86.09%
    p_lower = calc_clopper_pearson_zero_failure_bound(20, confidence=0.95)
    assert p_lower == pytest.approx(0.8609, rel=1e-2)

    # Certifying 95% requires 59 runs; 99% requires 299 runs; 99.9% requires 2995 runs
    assert calc_zero_failure_sample_size(0.95, 0.95) == 59
    assert calc_zero_failure_sample_size(0.99, 0.95) == 299
    assert calc_zero_failure_sample_size(0.999, 0.95) == 2995


def test_demonstration_yield_and_fatigue():
    # Chapter 5: 10,000 clean demos, 25% failure, 10% QA reject -> yield = 67.5%
    yield_res = calc_demonstration_collection_yield(
        target_clean_demos=10000,
        failure_rate=0.25,
        qa_rejection_rate=0.10,
        task_duration=Q_("45 s"),
        reset_duration=Q_("15 s"),
        daily_shift_hours=Q_("8 hr"),
        operator_efficiency=0.60,
    )
    assert yield_res["effective_yield"] == pytest.approx(0.675, rel=1e-3)
    assert yield_res["total_attempts"] == 14815
    assert yield_res["operator_hours"] == pytest.approx(411.5, rel=0.05)

    # Actuator wear: 50,000 demos * 30 s = 1.5e6 s, mean vel = 0.5 rad/s -> revs ≈ 1.19e5
    wear_res = calc_harmonic_drive_fatigue_consumption(
        total_active_duration=Q_("1500000 s"),
        mean_joint_velocity=Q_("0.5 rad/s"),
        rated_l10_life_revs=50_000_000.0,
    )
    assert wear_res["fatigue_percent"] < 1.0


def test_transient_impact_and_tripwire():
    # Chapter 16: m = 12 kg, v = 1.71 m/s, k = 15000 N/m -> F_peak = 1.71 * sqrt(15000 * 12) = 725.5 N
    f_impact = calc_transient_impact_force(
        contact_velocity=Q_("1.71 m/s"),
        effective_mass=Q_("12 kg"),
        contact_stiffness=Q_("15000 N/m"),
    )
    assert f_impact.to("N").magnitude == pytest.approx(725.5, rel=1e-2)

    # Tripwire: F_trip = 35 N, k = 4000 N/m, v = 0.4 m/s, t_lat = 10 ms
    f_accum = calc_tripwire_contact_force_accumulation(
        tripwire_force=Q_("35 N"),
        contact_stiffness=Q_("4000 N/m"),
        penetration_velocity=Q_("0.4 m/s"),
        loop_latency=Q_("10 ms"),
    )
    # F_lat = 35 + 4000 * 0.4 * 0.01 = 35 + 16 = 51 N
    assert f_accum.to("N").magnitude == pytest.approx(51.0, rel=1e-2)


def test_crypto_auth_deadline():
    # Chapter 14: d_clear = 1.2 m, v0 = 2.0 m/s, a = 4.0 m/s^2, t_mech = 80 ms, t_bus = 20 ms
    # d_brake = 4 / 8 = 0.5 m -> t_avail = 0.7 / 2 = 0.35 s = 350 ms -> T_auth = 350 - 100 = 250 ms
    t_deadline = calc_crypto_auth_deadline(
        clearance_distance=Q_("1.2 m"),
        initial_velocity=Q_("2.0 m/s"),
        emergency_deceleration=Q_("4.0 m/s^2"),
        mechanical_brake_lag=Q_("80 ms"),
        bus_arbitration_delay=Q_("20 ms"),
    )
    assert t_deadline.to("ms").magnitude == pytest.approx(250.0, rel=1e-2)


def test_coulomb_stiction_deadband():
    # Chapter 15: tau_lag = 5.0 ms, tau_c = 4.5 N*m, tau_cmd = 6.0 N*m
    # ratio = 6.0 / 1.5 = 4.0 -> t_dead = 5.0 * ln(4) ≈ 6.93 ms
    t_dead = calc_coulomb_stiction_deadband(
        commanded_torque=Q_("6.0 N*m"),
        coulomb_friction_torque=Q_("4.5 N*m"),
        stator_electrical_lag=Q_("5.0 ms"),
    )
    assert t_dead.to("ms").magnitude == pytest.approx(6.93, rel=1e-2)


def test_shielded_system_hazard_rate():
    # Chapter 17: p_brain = 1e-4 / hr, c_shield = 0.9999, p_hw = 1e-9 / hr
    p_sys = calc_shielded_system_hazard_rate(
        brain_hazard_rate=Q_("1e-4 / hr"),
        shield_coverage=0.9999,
        shield_hw_failure_rate=Q_("1e-9 / hr"),
    )
    # 1e-4 * 1e-4 + 1e-9 = 1.1e-8 / hr
    assert p_sys.to("1/hr").magnitude == pytest.approx(1.1e-8, rel=1e-2)


def test_cbf_qp_orthogonal_projection():
    # Chapter 12: u_nom = [3.5, 3.0], a = [2.0, 1.0], b = 5.0
    # a^T u_nom = 7.0 + 3.0 = 10.0 > 5.0 -> breach = 5.0, ||a||^2 = 5 -> lambda* = 1.0
    # u* = [3.5, 3.0] - 1.0 * [2.0, 1.0] = [1.5, 2.0]
    res = calc_cbf_qp_orthogonal_projection(u_nom=[3.5, 3.0], a_vec=[2.0, 1.0], b_scalar=5.0)
    assert res["was_modified"] is True
    assert res["lambda_star"] == pytest.approx(1.0)
    assert res["u_projected"][0] == pytest.approx(1.5)
    assert res["u_projected"][1] == pytest.approx(2.0)
    assert res["correction_norm"] == pytest.approx(math.sqrt(5.0), rel=1e-2)


def test_quintic_blend_duration():
    # Chapter 14: delta_tau = 25 N*m, j_max = 500 N*m/s -> tau = 1.875 * 25 / 500 = 0.09375 s = 93.75 ms
    tau = calc_quintic_blend_duration(delta_torque=Q_("25 N*m"), max_allowable_jerk=Q_("500 N*m/s"))
    assert tau.to("ms").magnitude == pytest.approx(93.75, rel=1e-2)


def test_thermal_cooling_recovery_time():
    # Chapter 13: tau_th = 8.0 s, T_start = 95 C, T_clear = 85 C, T_target = 53 C
    # t = 8 * ln((95 - 53)/(85 - 53)) = 8 * ln(42 / 32) = 8 * 0.2719 ≈ 2.175 s
    t_rec = calc_thermal_cooling_recovery_time(
        tau_thermal=Q_("8.0 s"),
        t_start=95.0,
        t_clear=85.0,
        t_target=53.0,
    )
    assert t_rec.to("s").magnitude == pytest.approx(2.18, rel=0.02)


def test_passive_compliance_stiffness():
    # Chapter 17: F_allow = 13.5 N, F_preload = 6.0 N, dx = 3.0 mm -> k = 7.5 / 0.003 = 2500 N/m
    k_spring = calc_passive_compliance_stiffness(
        allowable_force=Q_("13.5 N"),
        preload_force=Q_("6.0 N"),
        motor_stall_displacement=Q_("3.0 mm"),
    )
    assert k_spring.to("N/m").magnitude == pytest.approx(2500.0, rel=1e-2)


def test_process_containment_time_to_breach():
    # Chapter 15: V_max = 95 L, V_0 = 88 L, Q_in = 1.5 L/s, Q_out = 0.1 L/s -> Q_net = 1.4 L/s
    # t = 7.0 / 1.4 = 5.0 s
    t_b = calc_process_containment_time_to_breach(
        capacity_limit=Q_("95 L"),
        current_volume=Q_("88 L"),
        inflow_rate=Q_("1.5 L/s"),
        outflow_rate=Q_("0.1 L/s"),
    )
    assert t_b.to("s").magnitude == pytest.approx(5.0, rel=1e-2)


def test_teleop_ingestion_budget():
    # Chapter 5: 4 cameras (1080p RGB 30fps, 720p Depth 30fps), 1000Hz kin, 100Hz tac
    res = calc_teleop_ingestion_budget(
        num_cameras=4,
        rgb_width=1920,
        rgb_height=1080,
        rgb_fps=30.0,
        rgb_bytes_per_pixel=3,
        depth_width=1280,
        depth_height=720,
        depth_fps=30.0,
        depth_bytes_per_pixel=2,
        kinematics_bytes_per_sec=Q_("128 B") * 1000 / Q_("1 s"),
        tactile_bytes_per_sec=Q_("2048 B") * 100 / Q_("1 s"),
    )
    assert res["bw_rgb"].to("MB/s").magnitude == pytest.approx(746.50, rel=1e-3)
    assert res["bw_depth"].to("MB/s").magnitude == pytest.approx(221.18, rel=1e-3)
    assert res["bw_vision"].to("MB/s").magnitude == pytest.approx(967.68, rel=1e-3)
    assert res["bw_kinematics"].to("MB/s").magnitude == pytest.approx(0.128, rel=1e-2)
    assert res["bw_tactile"].to("MB/s").magnitude == pytest.approx(0.2048, rel=1e-2)
    assert res["bw_total"].to("MB/s").magnitude == pytest.approx(968.01, rel=1e-3)
    assert res["storage_per_hour"].to("TB").magnitude == pytest.approx(3.48, rel=1e-2)


def test_tsdf_voxel_grid_budget():
    # Chapter 9: 300 m^3, 1 cm vs 5 mm
    res_1cm = calc_tsdf_voxel_grid_budget(
        workspace_volume=Q_("300 m^3"),
        voxel_size=Q_("1.0 cm"),
        bytes_per_voxel=4,
        sparsity_ratio=0.03,
        ray_rate=Q_("18432000 / s"),
        dense_voxels_per_ray=300,
        sparse_voxels_per_ray=6,
        bytes_per_ray_voxel=8,
        cache_hit_rate=0.95,
    )
    assert res_1cm["dense_voxel_count"] == 300_000_000
    assert res_1cm["dense_memory"].to("GB").magnitude == pytest.approx(1.20, rel=1e-2)
    assert res_1cm["sparse_memory"].to("MB").magnitude == pytest.approx(36.56, rel=1e-2)
    assert res_1cm["dense_dram_bandwidth"].to("GB/s").magnitude == pytest.approx(44.24, rel=1e-2)
    assert res_1cm["traversed_voxels_per_ray"] == 300
    assert res_1cm["sparse_dram_bandwidth"].to("GB/s").magnitude == pytest.approx(2.21184, rel=1e-2)

    res_5mm = calc_tsdf_voxel_grid_budget(
        workspace_volume=Q_("300 m^3"),
        voxel_size=Q_("5.0 mm"),
        bytes_per_voxel=4,
        sparsity_ratio=0.03,
    )
    assert res_5mm["dense_voxel_count"] == 2_400_000_000
    assert res_5mm["dense_memory"].to("GB").magnitude == pytest.approx(9.60, rel=1e-2)
    assert res_5mm["sparse_memory"].to("MB").magnitude == pytest.approx(292.5, rel=1e-2)


@pytest.mark.parametrize("hit_rate", [-0.01, 1.01, float("nan")])
def test_tsdf_voxel_grid_budget_rejects_invalid_cache_hit_rate(hit_rate):
    with pytest.raises(ValueError, match="cache_hit_rate must be in"):
        calc_tsdf_voxel_grid_budget(
            workspace_volume=Q_("300 m^3"),
            voxel_size=Q_("1 cm"),
            cache_hit_rate=hit_rate,
        )


def test_intent_drift_lease():
    # Chapter 10: target error reaches tolerance in 60 ms; this is not a stopping budget.
    tau = calc_target_evidence_horizon(
        tolerance_radius=Q_("30.0 mm"),
        sensor_noise=Q_("6.0 mm"),
        drift_velocity=Q_("0.40 m/s"),
    )
    assert tau.to("ms").magnitude == pytest.approx(60.0, rel=1e-3)
    assert calc_intent_drift_lease(Q_("30.0 mm"), Q_("6.0 mm"), Q_("0.40 m/s")) == tau


def test_process_thermal_runaway_lease():
    # Chapter 10: P_in = 3600 W, C_th = 900 J/K, Delta_T = 2.0 K -> dT/dt = 4.0 K/s, tau = 500 ms
    res = calc_process_thermal_runaway_lease(
        heat_generation_rate=Q_("3600 W"),
        thermal_capacitance=Q_("900 J/K"),
        max_temp_overshoot=Q_("2.0 K"),
    )
    assert res["rate_of_rise"].to("K/s").magnitude == pytest.approx(4.0, rel=1e-3)
    assert res["tau_lease"].to("ms").magnitude == pytest.approx(500.0, rel=1e-3)
