"""Unit tests for physical AI and embodied robotics physics formulas."""

from __future__ import annotations

import math
import pytest
import pint

from mlsysim.core.units import ureg, Q_
from mlsysim.physics.robotics import (
    calc_axi_fifo_contention,
    calc_safe_stopping_distance,
    calc_max_permitted_velocity,
    calc_kinetic_energy,
    calc_seam_acceleration_torque_jump,
    calc_inverted_pendulum_fall_time,
    calc_reflected_inertia,
    calc_spatial_information_age_displacement,
    calc_empirical_testing_exposure,
    calc_3dgs_memory_footprint,
    calc_voltage_droop,
    calc_action_chunk_streaming_amortization,
    calc_watchdog_lease_bound,
    calc_phase_lag_total,
    calc_canfd_frame_bits,
    calc_canfd_bus_utilization,
    calc_ethercat_cycle_time,
    calc_contact_force,
    calc_contact_yield_deadline,
    calc_inductive_voltage_droop,
    calc_alpha_power_gate_delay_stretch,
    calc_clopper_pearson_zero_failure_bound,
    calc_zero_failure_sample_size,
    calc_transient_impact_force,
    calc_tripwire_contact_force_accumulation,
    calc_shielded_system_hazard_rate,
    calc_cbf_qp_orthogonal_projection,
    calc_quintic_blend_duration,
    calc_teleop_ingestion_budget,
    calc_tsdf_voxel_grid_budget,
    calc_target_evidence_horizon,
    calc_tracked_evidence_horizon,
    calc_latch_contact_stopping_profile,
    calc_c2_stop_suffix,
    calc_scurve_stop_distance,
    calc_scurve_stop_time,
    calc_scurve_max_permitted_velocity,
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




def test_3dgs_memory_footprint():
    # 1,000,000 gaussians * 56 bytes = 56,000,000 bytes = 56 MB
    mem = calc_3dgs_memory_footprint(num_gaussians=1_000_000, bytes_per_gaussian=56)
    assert mem.to("MB").magnitude == pytest.approx(56.0)


def test_voltage_droop():
    # 50 A draw through 40 mOhm internal resistance = 2.0 V droop
    droop = calc_voltage_droop(current=Q_("50 A"), internal_resistance=Q_("40 mohm"))
    assert droop.to("V").magnitude == pytest.approx(2.0)




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


def test_canfd_frame_bits_8_bytes():
    # ISO 11898-1:2015 FD base format, 8 data bytes, CRC-17.
    f = calc_canfd_frame_bits(8)
    assert f["data_field_bytes"] == 8
    assert f["crc_bits"] == 17 and f["fixed_stuff_bits"] == 6
    # Stuffed run SOF..data = 17 + 5 + 64 = 86 bits -> floor(85/4) = 21 stuff bits;
    # 4 of them fall in the 17 nominal-rate arbitration bits.
    assert f["stuff_bits_nominal"] == 4 and f["stuff_bits_data"] == 17
    # Nominal: 17 + 13 (CRC delim, ACK, ACK delim, EOF 7, IFS 3) + 4 = 34
    assert f["bits_nominal"] == 34
    # Data: ESI 1 + DLC 4 + 64 + stuff count 4 + CRC 17 + fixed stuff 6 + 17 = 113
    assert f["bits_data"] == 113


def test_canfd_frame_bits_crc21_padding_and_extended():
    f = calc_canfd_frame_bits(64)
    assert f["crc_bits"] == 21 and f["fixed_stuff_bits"] == 7
    assert f["bits_data"] == 5 + 512 + 4 + 21 + 7 + ((17 + 5 + 512 - 1) // 4 - 4)
    # 9 bytes pad up to the 12-byte data field; 17 < 20 keeps CRC-17.
    assert calc_canfd_frame_bits(9)["data_field_bytes"] == 12
    assert calc_canfd_frame_bits(17)["crc_bits"] == 21
    # Extended identifier: 36 nominal arbitration bits.
    ext = calc_canfd_frame_bits(8, extended_id=True, worst_case_stuffing=False)
    assert ext["bits_nominal"] == 36 + 13
    with pytest.raises(ValueError):
        calc_canfd_frame_bits(65)


def test_canfd_bus_utilization():
    # 9 axes, command + feedback frames, 8 bytes each, 1/5 Mbps, worst-case stuffing.
    res = calc_canfd_bus_utilization(
        num_axes=9,
        payload_bytes=8,
        baud_arbitration=Q_("1.0 Mbps"),
        baud_data=Q_("5.0 Mbps"),
        cycle_time=Q_("1.0 ms"),
    )
    # t_frame = 34 bits / 1 Mbps + 113 bits / 5 Mbps = 34 + 22.6 = 56.6 us
    assert res["t_tx"].to("us").magnitude == pytest.approx(56.6, rel=1e-9)
    assert res["n_frames"] == 18
    # 18 * 56.6 = 1018.8 us > 1 ms
    assert res["t_total_tx"].to("us").magnitude == pytest.approx(1018.8, rel=1e-9)
    assert res["utilization"] == pytest.approx(1.0188, rel=1e-9)
    assert res["is_schedulable"] is False
    # One bus per limb: the 7-joint arm bus fits the serialization check.
    arm = calc_canfd_bus_utilization(num_axes=7)
    assert arm["t_total_tx"].to("us").magnitude == pytest.approx(792.4, rel=1e-9)
    assert arm["is_schedulable"] is True
    # Without stuffing the frame is shorter (lower bound, not a WCRT input).
    lo = calc_canfd_bus_utilization(num_axes=9, worst_case_stuffing=False)
    assert lo["t_tx"] < res["t_tx"]


def test_phase_lag_total():
    # 20 Hz crossover, 1 ms tick, 0.25 ms compute, 1 ms bus, 5 deg allowance.
    r = calc_phase_lag_total(
        Q_("20 Hz"), Q_("1 ms"), compute_delay=Q_("0.25 ms"), transport_delay=Q_("1 ms"),
        phase_allowance=Q_("5 degree"),
    )
    # ZOH: 360 * 20 * 0.0005 = 3.6 deg; bus: 7.2 deg; compute: 1.8 deg; total 12.6 deg
    assert r["phase_zoh"].to("degree").magnitude == pytest.approx(3.6, rel=1e-9)
    assert r["phase_transport"].to("degree").magnitude == pytest.approx(7.2, rel=1e-9)
    assert r["phase_total"].to("degree").magnitude == pytest.approx(12.6, rel=1e-9)
    assert r["tau_total"].to("ms").magnitude == pytest.approx(1.75, rel=1e-9)
    # allowed delay = 5 / (360 * 20) s = 0.6944 ms; f_max = 5 / (360 * 1.75e-3) = 7.937 Hz
    assert r["tau_allowed"].to("ms").magnitude == pytest.approx(0.694444, rel=1e-5)
    assert r["f_crossover_max"].to("Hz").magnitude == pytest.approx(7.936508, rel=1e-5)
    assert r["within_allowance"] is False
    # Sample-and-hold alone fits the allowance.
    z = calc_phase_lag_total(Q_("20 Hz"), Q_("1 ms"), phase_allowance=Q_("5 degree"))
    assert z["within_allowance"] is True


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








def test_calc_latch_contact_stopping_profile():
    from mlsysim import Embodied
    AISLE = Embodied.Scenario.WarehouseAisle

    # Approach at guarded approach speed (0.03 m/s)
    res_lo = calc_latch_contact_stopping_profile(
        approach_velocity=AISLE.v_latch_approach,
        tripwire_force=AISLE.f_latch_tripwire,
        latch_stiffness=AISLE.k_latch,
        response_time=AISLE.t_contact_response,
        force_limit=AISLE.f_latch_limit,
    )
    assert res_lo["df_dt"].to("N/s").magnitude == pytest.approx(12000.0, rel=1e-3)
    assert res_lo["f_at_brake_onset"].to("N").magnitude == pytest.approx(39.0, rel=1e-6)
    assert res_lo["x_remaining"].to("mm").magnitude == pytest.approx(0.1525, rel=1e-4)
    assert res_lo["a_arm_needed"].to("m/s^2").magnitude == pytest.approx(2.9508, rel=1e-3)
    assert res_lo["t_arm_stop"].to("ms").magnitude == pytest.approx(10.167, rel=1e-3)

    # Inferring from scenario directly
    res_inferred = calc_latch_contact_stopping_profile(scenario=AISLE)
    assert res_inferred["f_at_brake_onset"] == res_lo["f_at_brake_onset"]

    # High approach speed (0.10 m/s)
    res_hi = calc_latch_contact_stopping_profile(
        approach_velocity=AISLE.v_latch_high,
        scenario=AISLE,
    )
    assert res_hi["f_at_brake_onset"].to("N").magnitude == pytest.approx(95.0, rel=1e-6)
    assert res_hi["x_remaining"].to("mm").magnitude == pytest.approx(0.0125, rel=1e-4)
    assert res_hi["a_arm_needed"].to("m/s^2").magnitude == pytest.approx(400.0, rel=1e-4)





# ── Jerk-limited S-curve stop (Biagiotti & Melchiorri 2008, ch. 3) ──────────

def test_c2_polynomial_peak_jerk():
    # T = 1.5 v / a = 1.125 s; j = 6 v / T^2 = (8/3) a^2 / v = 64/9 m/s^3.
    res = calc_c2_stop_suffix(Q_("1.5 m/s"), Q_("2.0 m/s^2"))
    assert res["peak_jerk"].to("m/s^3").magnitude == pytest.approx(64.0 / 9.0, rel=1e-12)


def test_scurve_stop_trapezoidal_regime_matches_closed_form():
    v, a, j = Q_("1.5 m/s"), Q_("2.0 m/s^2"), Q_(64.0 / 9.0, "m/s^3")
    # v >= a^2/j = 0.5625: d = (v/2)(v/a + a/j), T = v/a + a/j
    assert calc_scurve_stop_distance(v, a, j).to("mm").magnitude == pytest.approx(773.4375, rel=1e-12)
    assert calc_scurve_stop_time(v, a, j).to("ms").magnitude == pytest.approx(1031.25, rel=1e-12)
    # Equals constant braking plus cruise over half a jerk ramp.
    d_equiv = 1.5**2 / (2 * 2.0) + 1.5 * 2.0 / (2 * 64.0 / 9.0)
    assert calc_scurve_stop_distance(v, a, j).to("m").magnitude == pytest.approx(d_equiv, rel=1e-12)
    assert calc_scurve_stop_distance(Q_("1.3 m/s"), a, j).to("mm").magnitude == pytest.approx(605.3125, rel=1e-12)


def test_scurve_stop_triangular_regime_and_continuity():
    a, j = Q_("2.0 m/s^2"), Q_("8.0 m/s^3")  # threshold a^2/j = 0.5 m/s
    v = Q_("0.32 m/s")
    assert calc_scurve_stop_time(v, a, j).to("s").magnitude == pytest.approx(2 * math.sqrt(0.04), rel=1e-12)
    assert calc_scurve_stop_distance(v, a, j).to("m").magnitude == pytest.approx(0.32 * 0.2, rel=1e-12)
    # Both branches agree at the threshold speed.
    lo = calc_scurve_stop_distance(Q_(0.5 - 1e-12, "m/s"), a, j).to("m").magnitude
    hi = calc_scurve_stop_distance(Q_(0.5, "m/s"), a, j).to("m").magnitude
    assert lo == pytest.approx(hi, rel=1e-9)
    assert calc_scurve_stop_distance(Q_("0 m/s"), a, j).to("m").magnitude == 0.0


def test_scurve_is_shorter_than_polynomial_and_longer_than_constant_decel():
    for v_mps in (0.8, 1.0, 1.5, 2.5):
        v, a = Q_(v_mps, "m/s"), Q_("2.0 m/s^2")
        poly = calc_c2_stop_suffix(v, a)
        s_curve = calc_scurve_stop_distance(v, a, poly["peak_jerk"]).to("m").magnitude
        assert v_mps**2 / 4.0 < s_curve < poly["distance"].to("m").magnitude


def test_scurve_stop_rejects_invalid_limits():
    with pytest.raises(ValueError, match="max_jerk"):
        calc_scurve_stop_distance(Q_("1 m/s"), Q_("2 m/s^2"), Q_("0 m/s^3"))
    with pytest.raises(ValueError, match="max_deceleration"):
        calc_scurve_stop_time(Q_("1 m/s"), Q_("-2 m/s^2"), Q_("7 m/s^3"))


def test_scurve_ceiling_matches_closed_form_and_fills_clearance():
    d, tau, a, j = Q_("1.10 m"), Q_("133.6 ms"), Q_("2.0 m/s^2"), Q_(64.0 / 9.0, "m/s^3")
    loc, margin = Q_("0.05 m"), Q_("0.14 m")
    v = calc_scurve_max_permitted_velocity(d, tau, a, j, loc_margin=loc, physical_margin=margin)
    # Trapezoidal regime: constant braking after tau + a/(2j).
    closed = calc_max_permitted_velocity(d, tau + (a / (2 * j)).to("ms"), a, loc_margin=loc, physical_margin=margin)
    assert v.to("m/s").magnitude == pytest.approx(closed.to("m/s").magnitude, rel=1e-12)
    assert v.to("m/s").magnitude == pytest.approx(1.43669, abs=1e-5)
    used = (v * tau + calc_scurve_stop_distance(v, a, j) + loc + margin).to("m").magnitude
    assert used == pytest.approx(1.10, rel=1e-12)


def test_scurve_ceiling_with_approaching_obstacle_and_exhausted_clearance():
    d, tau, a, j = Q_("1.10 m"), Q_("133.6 ms"), Q_("2.0 m/s^2"), Q_(64.0 / 9.0, "m/s^3")
    vh = Q_("1.6 m/s")
    v = calc_scurve_max_permitted_velocity(d, tau, a, j, Q_("0.05 m"), Q_("0.14 m"), approach_velocity=vh)
    t = calc_scurve_stop_time(v, a, j)
    used = (v * tau + calc_scurve_stop_distance(v, a, j) + vh * (tau + t) + Q_("0.19 m")).to("m").magnitude
    assert used == pytest.approx(1.10, rel=1e-12)
    assert v < Q_("0.5625 m/s")  # triangular regime
    assert calc_scurve_max_permitted_velocity(Q_("0.1 m"), tau, a, j, Q_("0.05 m"), Q_("0.14 m")).magnitude == 0.0


# Chunk renewal on the setpoint grid (added 2026-09-28, M5).
from mlsysim.physics.robotics import calc_chunk_renewal_schedule  # noqa: E402


def test_chunk_renewal_schedule_hand_checked():
    s = calc_chunk_renewal_schedule(Q_("20 Hz"), Q_("25 ms"), 16, Q_("40 ms"))
    # 50 ms / 25 ms = 2; 50 + 40 = 90 ms; ceil(90/25) = 4; 16 x 25 = 400 ms; 16 // 2 = 8
    assert s["renewal_period"].to("ms").magnitude == pytest.approx(50.0)
    assert s["setpoints_per_renewal"] == 2
    assert s["open_loop_span"].to("ms").magnitude == pytest.approx(90.0)
    assert s["min_supply_setpoints"] == 4
    assert s["chunk_span"].to("ms").magnitude == pytest.approx(400.0)
    assert s["overlapping_chunks"] == 8
    assert s["renewal_slack"].to("ms").magnitude == pytest.approx(10.0)


def test_chunk_renewal_schedule_act_per_step_query():
    # ACT queries every step at 50 Hz with k = 100: n = 1, 100 overlapping predictions.
    s = calc_chunk_renewal_schedule(Q_("50 Hz"), Q_("20 ms"), 100, Q_("10 ms"))
    assert s["setpoints_per_renewal"] == 1
    assert s["overlapping_chunks"] == 100
    assert s["min_supply_setpoints"] == 2


def test_chunk_renewal_schedule_exact_multiple_h_min():
    # 40 ms renewal + 20 ms inference = 60 ms = exactly 3 x 20 ms steps.
    s = calc_chunk_renewal_schedule(Q_("25 Hz"), Q_("20 ms"), 8, Q_("20 ms"))
    assert s["min_supply_setpoints"] == 3


def test_chunk_renewal_schedule_rejects_off_grid():
    with pytest.raises(ValueError, match="not a whole number"):
        calc_chunk_renewal_schedule(Q_("20 Hz"), Q_("20 ms"), 16, Q_("40 ms"))


def test_chunk_renewal_schedule_rejects_overrun_and_short_chunk():
    with pytest.raises(ValueError, match="does not finish"):
        calc_chunk_renewal_schedule(Q_("25 Hz"), Q_("20 ms"), 16, Q_("40 ms"))
    with pytest.raises(ValueError, match="shorter than"):
        calc_chunk_renewal_schedule(Q_("20 Hz"), Q_("25 ms"), 3, Q_("40 ms"))


def test_tracked_evidence_horizon_constant_acceleration():
    # tau_ev = sqrt(2 (r_tol - e_0) / a_slip); 9 mm margin at 0.375 m/s^2 -> 219.09 ms
    tau = calc_tracked_evidence_horizon(Q_("15 mm"), Q_("6 mm"), Q_("0.375 m/s^2"))
    assert tau.to("ms").magnitude == pytest.approx(219.089, rel=1e-4)
    # the drift a_slip tau^2 / 2 consumes exactly the margin
    drift = (0.5 * Q_("0.375 m/s^2") * tau**2).to("mm")
    assert drift.magnitude == pytest.approx(9.0, rel=1e-9)
    with pytest.raises(ValueError):
        calc_tracked_evidence_horizon(Q_("6 mm"), Q_("6 mm"), Q_("0.375 m/s^2"))
    with pytest.raises(ValueError):
        calc_tracked_evidence_horizon(Q_("15 mm"), Q_("6 mm"), Q_("0 m/s^2"))


