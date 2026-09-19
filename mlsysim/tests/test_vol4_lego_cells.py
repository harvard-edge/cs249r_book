"""Unit tests verifying that Volume IV LEGO cells execute cleanly and their guards pass."""

import math
import pytest
from mlsysim import *
from mlsysim.core.units import *
from mlsysim.fmt import (
    fmt_qty,
    fmt_length,
    fmt_energy,
    fmt_time,
    fmt_velocity,
    fmt_acceleration,
    fmt_frequency,
    fmt_params,
    fmt_memory,
    fmt_bandwidth,
    fmt_latency,
    fmt_inertia,
    fmt_torque,
    fmt_multiple,
    fmt_int,
    fmt_sci,
    fmt,
    check,
)

millimeter = ureg.millimeter
newton = ureg.newton
joule = ureg.joule
watt = ureg.watt
kelvin = ureg.kelvin
kilogram = ureg.kilogram
hour = ureg.hour


def test_vol4_ch01_tempe_stopping_budget():
    vehicle = Embodied.Vehicle.UberATG_VolvoXC90
    v0 = 19.2 * (meter / second)
    mass = vehicle.mass
    t_suppress = 1.2 * second
    a_max = vehicle.max_acceleration
    d_remaining_at_request = 3.8 * meter

    d_lag = (v0 * t_suppress).to(meter)
    d_brake = ((v0**2) / (2 * a_max)).to(meter)
    d_stop = calc_safe_stopping_distance(v0, t_suppress, a_max)
    kinetic_energy = calc_kinetic_energy(mass, v0)

    check(abs(d_lag.to(meter).magnitude - 23.04) < 0.1, f"Unexpected lag displacement: {d_lag}")
    check(abs(d_brake.to(meter).magnitude - 23.04) < 0.1, f"Unexpected braking distance: {d_brake}")
    check(abs(d_stop.to(meter).magnitude - 46.08) < 0.1, f"Unexpected stopping distance: {d_stop}")
    check(
        d_stop.to(meter).magnitude > d_remaining_at_request.to(meter).magnitude * 12,
        "Remaining clearance should be less than 1/12 of stopping envelope",
    )


def test_vol4_ch02_amr_stopping_budget():
    amr = Embodied.AMR.LogisticsAMR
    mass = amr.mass
    v0 = amr.max_velocity
    tau_delay = 65 * millisecond
    a_brake = 3.0 * (meter / (second**2))
    delta_loc = 0.05 * meter
    delta_margin = 0.10 * meter

    d_react = (v0 * tau_delay).to(meter)
    d_brake = ((v0**2) / (2 * a_brake)).to(meter)
    ke = calc_kinetic_energy(mass, v0)
    d_stop = calc_safe_stopping_distance(v0, tau_delay, a_brake, loc_margin=delta_loc, physical_margin=delta_margin)

    v1 = 2.4 * (meter / second)
    d_react_v1 = (v1 * tau_delay).to(meter)
    d_brake_v1 = ((v1**2) / (2 * a_brake)).to(meter)
    d_stop_v1 = calc_safe_stopping_distance(v1, tau_delay, a_brake, loc_margin=delta_loc, physical_margin=delta_margin)
    d_stop_expansion = d_stop_v1 - d_stop

    check(abs(d_react.magnitude - 0.117) < 0.001, f"Unexpected reaction drift: {d_react}")
    check(abs(d_brake.magnitude - 0.540) < 0.001, f"Unexpected braking distance: {d_brake}")
    check(abs(ke.to(joule).magnitude - 243.0) < 0.1, f"Unexpected kinetic energy: {ke}")
    check(abs(d_stop.magnitude - 0.807) < 0.001, f"Unexpected stopping distance: {d_stop}")
    check(abs(d_react_v1.magnitude - 0.156) < 0.001, f"Unexpected scaled reaction drift: {d_react_v1}")
    check(abs(d_brake_v1.magnitude - 0.960) < 0.001, f"Unexpected scaled braking distance: {d_brake_v1}")
    check(abs(d_stop_v1.magnitude - 1.266) < 0.001, f"Unexpected scaled stopping distance: {d_stop_v1}")
    check(abs(d_stop_expansion.magnitude - 0.459) < 0.001, f"Unexpected clearance buffer loss: {d_stop_expansion}")


def test_vol4_ch03_action_chunking_lpddr5():
    model = Models.Embodied.OpenVLA_7B
    soc = Hardware.Edge.JetsonAGXOrin
    params = model.parameters
    bytes_per_param = 2 * (byte / param)
    b_peak = soc.memory.bandwidth
    bus_efficiency = 0.70
    chunk_horizon = 32

    weight_mem = memory_from_params(params, bytes_per_param)
    res = calc_action_chunk_streaming_amortization(
        model_weight_memory=weight_mem,
        memory_bandwidth=b_peak,
        chunk_horizon=chunk_horizon,
        bus_efficiency=bus_efficiency,
    )
    b_sustained = res["b_sustained"]
    t_stream = res["t_stream"]
    f_single = res["f_single"]
    tau_step = res["tau_step"]
    f_effective = res["f_effective"]

    check(abs(weight_mem.to(GB).magnitude - 14.0) < 0.1, f"Unexpected weight footprint: {weight_mem}")
    check(abs(b_sustained.to(GB / second).magnitude - 142.80) < 0.2, f"Unexpected sustained bandwidth: {b_sustained}")
    check(abs(t_stream.to(millisecond).magnitude - 98.04) < 0.2, f"Unexpected streaming time: {t_stream}")
    check(abs(f_single.magnitude - 10.20) < 0.2, f"Unexpected single frequency: {f_single}")
    check(abs(tau_step.to(millisecond).magnitude - 3.06) < 0.2, f"Unexpected step latency: {tau_step}")
    check(abs(f_effective.magnitude - 326.53) < 1.5, f"Unexpected effective frequency: {f_effective}")


def test_vol4_ch08_sensor_information_age():
    camera = Sensors.Camera.Sony_IMX477
    t_exp_half = (camera.exposure_time / 2).to(millisecond)
    t_readout = camera.readout_time
    t_transport = 0.8 * millisecond
    t_dma = 1.2 * millisecond
    t_isp = 2.5 * millisecond
    t_backbone = 22.0 * millisecond
    t_ipc = 0.5 * millisecond

    v_amr = 2.5 * (meter / second)
    v_arm = 1.8 * (meter / second)
    v_melt = 0.8 * (meter / second)

    tau_age = t_exp_half + t_readout + t_transport + t_dma + t_isp + t_backbone + t_ipc
    dx_amr = calc_spatial_information_age_displacement(v_amr, tau_age)
    dx_arm = calc_spatial_information_age_displacement(v_arm, tau_age)
    dx_melt = calc_spatial_information_age_displacement(v_melt, tau_age)

    check(abs(tau_age.to(millisecond).magnitude - 51.6) < 0.01, f"Unexpected tau_age: {tau_age}")
    check(abs(dx_amr.to(meter).magnitude - 0.129) < 0.001, f"Unexpected dx_amr: {dx_amr}")
    check(abs(dx_arm.to(meter).magnitude - 0.0929) < 0.001, f"Unexpected dx_arm: {dx_arm}")
    check(abs(dx_melt.to(meter).magnitude - 0.0413) < 0.001, f"Unexpected dx_melt: {dx_melt}")


def test_vol4_ch11_planning_seam_inertia():
    joint = Actuators.HarmonicDrive.CSG_25_50
    gear_ratio = joint.gear_ratio
    rotor_inertia = joint.rotor_inertia
    delta_omega = 0.2 * (ureg.radian / second)
    dt_tick = 1.0 * millisecond
    t_blend = 50.0 * millisecond

    j_reflected = joint.reflected_inertia
    alpha_unblended = (delta_omega / dt_tick).to(ureg.radian / (second**2))
    tau_unblended = calc_seam_acceleration_torque_jump(gear_ratio, rotor_inertia, alpha_unblended)
    alpha_blended = (delta_omega / t_blend).to(ureg.radian / (second**2))
    tau_blended = calc_seam_acceleration_torque_jump(gear_ratio, rotor_inertia, alpha_blended)
    torque_reduction_ratio = tau_unblended / tau_blended

    check(abs(j_reflected.magnitude - 0.25) < 0.001, f"Unexpected reflected inertia: {j_reflected}")
    check(abs(alpha_unblended.magnitude - 200.0) < 0.1, f"Unexpected unblended acceleration: {alpha_unblended}")
    check(abs(tau_unblended.magnitude - 50.0) < 0.1, f"Unexpected unblended torque: {tau_unblended}")
    check(abs(alpha_blended.magnitude - 4.0) < 0.01, f"Unexpected blended acceleration: {alpha_blended}")
    check(abs(tau_blended.magnitude - 1.0) < 0.01, f"Unexpected blended torque: {tau_blended}")
    check(abs(torque_reduction_ratio.magnitude - 50.0) < 0.1, f"Unexpected reduction ratio: {torque_reduction_ratio}")


def test_vol4_ch14_intervention_takeover_budget():
    v0 = 5.0 * (meter / second)
    a_max = 2.5 * (meter / (second**2))

    t_haptic = 150.0 * millisecond
    t_out_of_loop = 2000.0 * millisecond

    d_brake = ((v0**2) / (2 * a_max)).to(meter)
    d_drift_haptic = (v0 * t_haptic).to(meter)
    d_stop_haptic = calc_safe_stopping_distance(v0, t_haptic, a_max)

    d_drift_ool = (v0 * t_out_of_loop).to(meter)
    d_stop_ool = calc_safe_stopping_distance(v0, t_out_of_loop, a_max)

    drift_expansion_ratio = d_drift_ool / d_drift_haptic
    stop_expansion_ratio = d_stop_ool / d_stop_haptic

    check(abs(d_brake.magnitude - 5.0) < 0.01, f"Unexpected braking distance: {d_brake}")
    check(abs(d_drift_haptic.magnitude - 0.75) < 0.01, f"Unexpected haptic drift: {d_drift_haptic}")
    check(abs(d_stop_haptic.magnitude - 5.75) < 0.01, f"Unexpected haptic stop: {d_stop_haptic}")
    check(abs(d_drift_ool.magnitude - 10.0) < 0.01, f"Unexpected out-of-loop drift: {d_drift_ool}")
    check(abs(d_stop_ool.magnitude - 15.0) < 0.01, f"Unexpected out-of-loop stop: {d_stop_ool}")
    check(abs(drift_expansion_ratio.magnitude - 13.33) < 0.1, f"Unexpected drift expansion: {drift_expansion_ratio}")


def test_vol4_ch04_amr_watchdog_lease():
    amr = Embodied.AMR.WarehouseAMR
    v0 = amr.max_velocity
    a_max = amr.max_acceleration
    d_clear = 0.6939 * meter

    budget = calc_watchdog_lease_bound(d_clear, v0, a_max)
    d_brake = budget["d_brake"]
    d_drift = budget["d_drift_allowable"]
    t_lease = budget["t_lease_max"]

    check(abs(d_brake.to(meter).magnitude - 0.5625) < 0.001, f"Unexpected braking distance: {d_brake}")
    check(abs(d_drift.to(meter).magnitude - 0.1314) < 0.001, f"Unexpected drift allowance: {d_drift}")
    check(abs(t_lease.to(millisecond).magnitude - 87.6) < 0.1, f"Unexpected lease ceiling: {t_lease}")
    check(d_brake + d_drift <= d_clear, "Stopping envelope cannot exceed optical clearance")


def test_vol4_ch04_fieldbus_schedulability():
    num_axes = 8
    cycle_time = 1.0 * millisecond

    canfd_64 = calc_canfd_bus_utilization(
        num_axes=num_axes,
        payload_bytes=64,
        baud_arbitration=1.0 * (megabit / second),
        baud_data=5.0 * (megabit / second),
        cycle_time=cycle_time,
        interframe_spacing=6.0 * microsecond,
    )
    canfd_32 = calc_canfd_bus_utilization(
        num_axes=num_axes,
        payload_bytes=32,
        baud_arbitration=1.0 * (megabit / second),
        baud_data=5.0 * (megabit / second),
        cycle_time=cycle_time,
        interframe_spacing=6.0 * microsecond,
    )
    ethercat = calc_ethercat_cycle_time(
        num_axes=num_axes,
        bytes_per_axis=32,
        baud_rate=100.0 * (megabit / second),
        header_bytes=64,
        hop_delay=0.6 * microsecond,
        cycle_time=cycle_time,
    )

    check(abs(canfd_64["t_tx"].to(microsecond).magnitude - 140.4) < 0.5, f"Unexpected CAN-FD 64B frame time: {canfd_64['t_tx']}")
    check(abs(canfd_64["t_total_tx"].to(microsecond).magnitude - 1123.2) < 2.0, f"Unexpected CAN-FD 64B total: {canfd_64['t_total_tx']}")
    check(canfd_64["utilization"] > 1.0, "CAN-FD 64B must overrun 1.0 ms cycle time")
    check(abs(ethercat["t_cycle_total"].to(microsecond).magnitude - 30.4) < 0.2, f"Unexpected EtherCAT total: {ethercat['t_cycle_total']}")
    check(abs(ethercat["t_margin"].to(microsecond).magnitude - 969.6) < 0.5, f"Unexpected EtherCAT margin: {ethercat['t_margin']}")
    check(ethercat["utilization"] < 0.05, "EtherCAT utilization must be under 5%")


def test_vol4_ch05_multi_camera_ingestion_budget():
    res_1080p = calc_teleop_ingestion_budget(
        num_cameras=4,
        rgb_width=1920,
        rgb_height=1080,
        rgb_fps=30.0,
        rgb_bytes_per_pixel=3,
        depth_width=1280,
        depth_height=720,
        depth_fps=30.0,
        depth_bytes_per_pixel=2,
        kinematics_bytes_per_sec=128 * byte * 1000 / second,
        tactile_bytes_per_sec=2048 * byte * 100 / second,
    )
    check(abs(res_1080p["bw_rgb"].to(MB / second).magnitude - 746.50) < 0.2, f"Unexpected RGB: {res_1080p['bw_rgb']}")
    check(abs(res_1080p["bw_depth"].to(MB / second).magnitude - 221.18) < 0.2, f"Unexpected Depth: {res_1080p['bw_depth']}")
    check(abs(res_1080p["bw_total"].to(MB / second).magnitude - 968.01) < 0.2, f"Unexpected Total: {res_1080p['bw_total']}")
    check(abs(res_1080p["storage_per_hour"].to(TB).magnitude - 3.48) < 0.05, f"Unexpected Storage: {res_1080p['storage_per_hour']}")

    res_720p = calc_teleop_ingestion_budget(
        num_cameras=4,
        rgb_width=1280,
        rgb_height=720,
        rgb_fps=30.0,
        rgb_bytes_per_pixel=3,
        depth_width=1280,
        depth_height=720,
        depth_fps=30.0,
        depth_bytes_per_pixel=2,
        kinematics_bytes_per_sec=128 * byte * 1000 / second,
        tactile_bytes_per_sec=2048 * byte * 100 / second,
    )
    check(abs(res_720p["bw_rgb"].to(MB / second).magnitude - 331.78) < 0.2, f"Unexpected 720p RGB: {res_720p['bw_rgb']}")
    check(abs(res_720p["bw_total"].to(MB / second).magnitude - 553.29) < 0.2, f"Unexpected 720p Total: {res_720p['bw_total']}")
    check(abs(res_720p["storage_per_hour"].to(TB).magnitude - 1.99) < 0.05, f"Unexpected 720p Storage: {res_720p['storage_per_hour']}")


def test_vol4_ch05_demonstration_yield_fatigue():
    t_sched = 60.0 * minute
    t_motion = 35.0 * minute
    t_att = 25.0 * second
    t_rst = 15.0 * second
    t_cyc = t_att + t_rst

    n_raw_attempts = int(t_motion.to(second) // t_cyc.to(second))
    failure_rate = 0.20
    qa_filter_rate = 0.15
    retention_rate = (1.0 - failure_rate) * (1.0 - qa_filter_rate)
    n_usable = n_raw_attempts * retention_rate
    t_yield = (n_usable * t_att).to(minute)
    yield_fraction = t_yield / t_sched

    n_campaign_target = 5000
    n_total_attempts = int(math.ceil(n_campaign_target / retention_rate))
    t_sched_campaign = (n_total_attempts / n_raw_attempts) * (1.0 * hour)

    n_gear_reversals_per_attempt = 14
    n_cable_flex_per_attempt = 6 + 2
    l10_gear_life = 5_000_000
    l10_cable_life = 200_000

    gear_cycles = n_total_attempts * n_gear_reversals_per_attempt
    cable_cycles = n_total_attempts * n_cable_flex_per_attempt
    gear_fatigue_pct = (gear_cycles / l10_gear_life) * 100.0
    cable_fatigue_pct = (cable_cycles / l10_cable_life) * 100.0

    check(n_raw_attempts == 52, f"Unexpected raw attempts: {n_raw_attempts}")
    check(abs(retention_rate - 0.68) < 1e-4, f"Unexpected retention: {retention_rate}")
    check(abs(n_usable - 35.36) < 0.01, f"Unexpected usable episodes: {n_usable}")
    check(abs(t_yield.magnitude - 14.73) < 0.05, f"Unexpected yield time: {t_yield}")
    check(abs(yield_fraction.magnitude - 0.246) < 0.005, f"Unexpected yield fraction: {yield_fraction}")
    check(n_total_attempts == 7353, f"Unexpected campaign attempts: {n_total_attempts}")
    check(abs(t_sched_campaign.magnitude - 141.4) < 0.5, f"Unexpected scheduled hours: {t_sched_campaign}")
    check(gear_cycles == 102942, f"Unexpected gear cycles: {gear_cycles}")
    check(abs(gear_fatigue_pct - 2.06) < 0.05, f"Unexpected gear fatigue: {gear_fatigue_pct}")
    check(cable_cycles == 58824, f"Unexpected cable cycles: {cable_cycles}")
    check(abs(cable_fatigue_pct - 29.4) < 0.1, f"Unexpected cable fatigue: {cable_fatigue_pct}")


def test_vol4_ch07_clopper_pearson_zero_failure():
    # 20 consecutive successes
    n_trials = 20
    p_lower = calc_clopper_pearson_zero_failure_bound(n_trials, confidence=0.95)
    risk_bound = 1.0 - p_lower
    mtbf_trials = 1.0 / risk_bound

    check(abs(p_lower - 0.8609) < 0.001, f"Unexpected p_lower: {p_lower}")
    check(abs(risk_bound - 0.1391) < 0.001, f"Unexpected risk: {risk_bound}")
    check(abs(mtbf_trials - 7.19) < 0.05, f"Unexpected MTBF: {mtbf_trials}")

    # Certifying 99.9% reliability
    p_target = 0.999
    n_req = calc_zero_failure_sample_size(p_target, confidence=0.95)
    t_trial = 260.0 * second
    t_total = (n_req * t_trial).to(hour)
    t_days = t_total.to(day)
    t_operator_hours = t_total * 2.0
    shifts = (t_operator_hours / (8.0 * hour)).magnitude

    check(n_req == 2995, f"Unexpected required trials: {n_req}")
    check(abs(t_total.magnitude - 216.31) < 0.1, f"Unexpected total hours: {t_total}")
    check(abs(t_days.magnitude - 9.01) < 0.05, f"Unexpected total days: {t_days}")
    check(abs(t_operator_hours.magnitude - 432.61) < 0.2, f"Unexpected operator hours: {t_operator_hours}")
    check(abs(shifts - 54.08) < 0.1, f"Unexpected shifts: {shifts}")


def test_vol4_ch09_tsdf_voxel_grid_budget():
    workspace = 300.0 * (meter**3)
    res_1cm = calc_tsdf_voxel_grid_budget(
        workspace_volume=workspace,
        voxel_size=0.01 * meter,
        bytes_per_voxel=4,
        sparsity_ratio=0.03,
        ray_rate=18432000.0 / second,
        dense_voxels_per_ray=300,
        sparse_voxels_per_ray=6,
        bytes_per_ray_voxel=8,
        cache_hit_rate=0.95,
    )
    check(res_1cm["dense_voxel_count"] == 300_000_000, f"Unexpected 1cm voxels: {res_1cm['dense_voxel_count']}")
    check(abs(res_1cm["dense_memory"].to(GB).magnitude - 1.20) < 0.01, f"Unexpected 1cm dense mem: {res_1cm['dense_memory']}")
    check(res_1cm["sparse_block_count"] == 17579, f"Unexpected 1cm blocks: {res_1cm['sparse_block_count']}")
    check(abs(res_1cm["sparse_memory"].to(MB).magnitude - 36.56) < 0.1, f"Unexpected 1cm sparse mem: {res_1cm['sparse_memory']}")
    check(abs(res_1cm["dense_dram_bandwidth"].to(GB / second).magnitude - 44.24) < 0.1, f"Unexpected dense BW: {res_1cm['dense_dram_bandwidth']}")
    check(abs(res_1cm["sparse_dram_bandwidth"].to(MB / second).magnitude - 44.24) < 0.1, f"Unexpected sparse BW: {res_1cm['sparse_dram_bandwidth']}")

    res_5mm = calc_tsdf_voxel_grid_budget(
        workspace_volume=workspace,
        voxel_size=0.005 * meter,
        bytes_per_voxel=4,
        sparsity_ratio=0.03,
    )
    check(res_5mm["dense_voxel_count"] == 2_400_000_000, f"Unexpected 5mm voxels: {res_5mm['dense_voxel_count']}")
    check(abs(res_5mm["dense_memory"].to(GB).magnitude - 9.60) < 0.01, f"Unexpected 5mm dense mem: {res_5mm['dense_memory']}")
    check(abs(res_5mm["sparse_memory"].to(MB).magnitude - 292.5) < 0.5, f"Unexpected 5mm sparse mem: {res_5mm['sparse_memory']}")


def test_vol4_ch12_cbf_qp_orthogonal_projection():
    u_nom = [3.5, 3.0]
    a_vec = [2.0, 1.0]
    b_scalar = 5.0

    res = calc_cbf_qp_orthogonal_projection(u_nom, a_vec, b_scalar)
    check(res["was_modified"] is True, "Proposal must be modified")
    check(abs(res["lambda_star"] - 1.0) < 1e-4, f"Unexpected lambda*: {res['lambda_star']}")
    check(abs(res["u_projected"][0] - 1.5) < 1e-4, f"Unexpected u1*: {res['u_projected'][0]}")
    check(abs(res["u_projected"][1] - 2.0) < 1e-4, f"Unexpected u2*: {res['u_projected'][1]}")
    check(abs(res["correction_norm"] - math.sqrt(5.0)) < 1e-4, f"Unexpected norm: {res['correction_norm']}")

    # Heuristic per-axis clamping:
    u_clip = [2.0, 3.0]
    barrier_eval_clip = a_vec[0] * u_clip[0] + a_vec[1] * u_clip[1]
    check(barrier_eval_clip == 7.0, f"Unexpected barrier eval: {barrier_eval_clip}")
    check(barrier_eval_clip > b_scalar, "Heuristic clamp must violate the barrier")


def test_vol4_ch13_inductive_pdn_voltage_droop():
    v_core = 0.850 * ureg.volt
    l_pkg = 30.0 * ureg.picohenry
    r_pkg = 8.0 * ureg.milliohm
    delta_i = 6.0 * ureg.ampere
    t_rise = 2.0 * nanosecond
    di_dt = delta_i / t_rise

    v_ind = calc_inductive_voltage_droop(l_pkg, di_dt)
    v_res = calc_voltage_droop(delta_i, r_pkg)
    v_droop = v_ind + v_res
    v_drooped = v_core - v_droop
    droop_fraction = (v_droop / v_core).magnitude

    v_th = 0.420 * ureg.volt
    alpha = 1.3
    res_stretch = calc_alpha_power_gate_delay_stretch(
        nominal_voltage=v_core,
        drooped_voltage=v_drooped,
        threshold_voltage=v_th,
        alpha=alpha,
    )
    t_stretch_ratio = res_stretch["stretch_ratio"]
    delay_expansion_pct = res_stretch["delay_expansion_percent"]

    check(abs(v_ind.to(ureg.millivolt).magnitude - 90.0) < 0.1, f"Unexpected inductive droop: {v_ind}")
    check(abs(v_res.to(ureg.millivolt).magnitude - 48.0) < 0.1, f"Unexpected resistive droop: {v_res}")
    check(abs(v_droop.to(ureg.millivolt).magnitude - 138.0) < 0.1, f"Unexpected total droop: {v_droop}")
    check(abs(v_drooped.to(ureg.volt).magnitude - 0.712) < 0.001, f"Unexpected drooped rail: {v_drooped}")
    check(abs(droop_fraction - 0.1624) < 0.001, f"Unexpected droop fraction: {droop_fraction}")
    check(abs(t_stretch_ratio - 1.38) < 0.02, f"Unexpected delay stretch ratio: {t_stretch_ratio}")
    check(abs(delay_expansion_pct - 38.0) < 2.0, f"Unexpected delay expansion: {delay_expansion_pct}")


def test_vol4_ch15_coulomb_stiction_deadband():
    tau_cmd = 6.0 * (ureg.newton * meter)
    tau_c = 4.5 * (ureg.newton * meter)
    tau_lag = 5.0 * millisecond
    inertia = 0.050 * (kilogram * (meter**2))

    t_dead = calc_coulomb_stiction_deadband(
        commanded_torque=tau_cmd,
        coulomb_friction_torque=tau_c,
        stator_electrical_lag=tau_lag,
    )
    alpha_sim = (tau_cmd / inertia).to(ureg.radian / (second**2))
    tau_net_real = tau_cmd - tau_c
    alpha_real = (tau_net_real / inertia).to(ureg.radian / (second**2))

    t_eval = 50.0 * millisecond
    theta_sim = (0.5 * alpha_sim.magnitude * (t_eval.to(second).magnitude ** 2))
    t_active_real = (t_eval - t_dead).to(second).magnitude
    theta_real = (0.5 * alpha_real.magnitude * (t_active_real ** 2))
    delta_theta = abs(theta_sim - theta_real)

    check(abs(t_dead.to(millisecond).magnitude - 6.93) < 0.02, f"Unexpected deadband: {t_dead}")
    check(abs(alpha_sim.magnitude - 120.0) < 0.1, f"Unexpected sim accel: {alpha_sim}")
    check(abs(alpha_real.magnitude - 30.0) < 0.1, f"Unexpected real accel: {alpha_real}")
    check(abs(theta_sim - 0.1500) < 0.001, f"Unexpected sim angle: {theta_sim}")
    check(abs(theta_real - 0.0278) < 0.001, f"Unexpected real angle: {theta_real}")
    check(abs(delta_theta - 0.1222) < 0.001, f"Unexpected delta theta: {delta_theta}")


def test_vol4_ch06_covariate_drift_compounding():
    dt = 0.020 * second
    t_total = 10.0 * second
    n_steps = 500
    epsilon = 0.005
    a_bias = 0.05 * (meter / (second**2))
    tau_rec = 0.10 * second

    p_drift = 1.0 - ((1.0 - epsilon) ** n_steps)
    drift_open = (epsilon / dt.magnitude) * 0.5 * a_bias.magnitude * ((t_total.magnitude ** 3) / 3.0) * meter
    delta_y_rec = 0.5 * a_bias.magnitude * (tau_rec.magnitude ** 2) * meter
    drift_closed = (n_steps * epsilon) * delta_y_rec
    drift_ratio = (drift_open / drift_closed).to_base_units().magnitude

    check(abs(p_drift - 0.9184) < 0.001, f"Unexpected p_drift: {p_drift}")
    check(abs(drift_open.to(meter).magnitude - 2.0833) < 0.01, f"Unexpected open drift: {drift_open}")
    check(abs(drift_closed.to(ureg.millimeter).magnitude - 0.625) < 0.001, f"Unexpected closed drift: {drift_closed}")
    check(abs(drift_ratio - 3333.33) < 1.0, f"Unexpected drift ratio: {drift_ratio}")


def test_vol4_ch06_action_chunk_denoising_cadence():
    t_vis = 14.0 * millisecond
    t_step = 1.65 * millisecond

    k_ddim = 16
    t_denoise_ddim = k_ddim * t_step
    tau_inf_ddim = t_vis + t_denoise_ddim

    n_flow = 2
    t_flow = n_flow * t_step
    tau_inf_flow = t_vis + t_flow

    t_deadline = 20.0 * millisecond
    ddim_overrun_pct = ((tau_inf_ddim - t_deadline) / t_deadline).magnitude * 100.0
    flow_slack = t_deadline - tau_inf_flow
    flow_slack_pct = (flow_slack / t_deadline).magnitude * 100.0

    t_period_async = (1.0 / 15.0) * second
    async_slack = t_period_async.to(millisecond) - tau_inf_ddim

    check(abs(tau_inf_ddim.to(millisecond).magnitude - 40.4) < 0.01, f"Unexpected DDIM inf: {tau_inf_ddim}")
    check(abs(t_denoise_ddim.to(millisecond).magnitude - 26.4) < 0.01, f"Unexpected DDIM denoise: {t_denoise_ddim}")
    check(abs(tau_inf_flow.to(millisecond).magnitude - 17.3) < 0.01, f"Unexpected flow inf: {tau_inf_flow}")
    check(abs(t_flow.to(millisecond).magnitude - 3.3) < 0.01, f"Unexpected flow denoise: {t_flow}")
    check(abs(ddim_overrun_pct - 102.0) < 0.1, f"Unexpected DDIM overrun: {ddim_overrun_pct}")
    check(abs(flow_slack.to(millisecond).magnitude - 2.7) < 0.01, f"Unexpected flow slack: {flow_slack}")
    check(abs(flow_slack_pct - 13.5) < 0.1, f"Unexpected flow slack pct: {flow_slack_pct}")
    check(abs(async_slack.to(millisecond).magnitude - 26.27) < 0.1, f"Unexpected async slack: {async_slack}")


def test_vol4_ch10_intent_lease_archetypes():
    from mlsysim.physics.robotics import (
        calc_intent_drift_lease,
        calc_tripwire_contact_force_accumulation,
        calc_contact_force,
        calc_process_thermal_runaway_lease,
    )

    # Class 1: Logistics AMR
    r_tol_amr = 30.0 * millimeter
    sig_sensor_amr = 6.0 * millimeter
    v_drift_amr = 0.40 * (meter / second)
    v_cruise_amr = 1.5 * (meter / second)
    a_brake_amr = 3.0 * (meter / (second**2))
    delta_t_stall_amr = 200.0 * millisecond

    # Class 2: Manipulator
    k_fixture = 4.0e5 * (newton / meter)
    v_approach = 0.03 * (meter / second)
    f_max_ceiling = 15.0 * newton
    t_clamp_loop = 2.0 * millisecond
    delta_t_stall_arm = 200.0 * millisecond

    # Class 3: BESS Thermal
    c_thermal_bess = 900.0 * (joule / kelvin)
    p_heat_in = 3600.0 * watt
    delta_t_overshoot_max = 2.0 * kelvin
    delta_t_stall_bess = 10.0 * second

    # Class 1 Computations
    tau_amr = calc_intent_drift_lease(r_tol_amr, sig_sensor_amr, v_drift_amr)
    d_react_bounded = (v_cruise_amr * tau_amr).to(millimeter)
    d_brake_amr = ((v_cruise_amr**2) / (2.0 * a_brake_amr)).to(millimeter)
    d_stop_bounded = d_react_bounded + d_brake_amr

    d_react_unbounded = (v_cruise_amr * delta_t_stall_amr).to(millimeter)
    d_stop_unbounded = d_react_unbounded + d_brake_amr
    d_overrun_amr = d_stop_unbounded - d_stop_bounded

    # Class 2 Computations
    df_dt = (k_fixture * v_approach).to(newton / second)
    f_peak_bounded = calc_tripwire_contact_force_accumulation(f_max_ceiling, k_fixture, v_approach, t_clamp_loop)
    delta_x_unbounded = (v_approach * delta_t_stall_arm).to(millimeter)
    f_unbounded = calc_contact_force(k_fixture, delta_x_unbounded)

    # Class 3 Computations
    thermal_res = calc_process_thermal_runaway_lease(p_heat_in, c_thermal_bess, delta_t_overshoot_max)
    dt_dt_bess = thermal_res["rate_of_rise"]
    tau_bess = thermal_res["tau_lease"]
    delta_t_runaway = (dt_dt_bess * delta_t_stall_bess).to(kelvin)

    check(abs(tau_amr.to(millisecond).magnitude - 60.0) < 0.1, f"Unexpected AMR lease: {tau_amr}")
    check(abs(d_stop_bounded.to(millimeter).magnitude - 465.0) < 0.5, f"Unexpected bounded AMR stop: {d_stop_bounded}")
    check(abs(d_stop_unbounded.to(millimeter).magnitude - 675.0) < 0.5, f"Unexpected unbounded AMR stop: {d_stop_unbounded}")
    check(abs(d_overrun_amr.to(millimeter).magnitude - 210.0) < 0.5, f"Unexpected AMR overrun: {d_overrun_amr}")
    check(abs(f_peak_bounded.to(newton).magnitude - 39.0) < 0.1, f"Unexpected peak bounded force: {f_peak_bounded}")
    check(abs(f_unbounded.to(newton).magnitude - 2400.0) < 1.0, f"Unexpected unbounded force: {f_unbounded}")
    check(abs(dt_dt_bess.to(kelvin / second).magnitude - 4.0) < 0.1, f"Unexpected thermal rise rate: {dt_dt_bess}")
    check(abs(tau_bess.to(millisecond).magnitude - 500.0) < 0.1, f"Unexpected BESS lease: {tau_bess}")
    check(abs(delta_t_runaway.to(kelvin).magnitude - 40.0) < 0.1, f"Unexpected thermal runaway delta: {delta_t_runaway}")


def test_vol4_ch16_biomechanical_impact_envelope():
    from mlsysim.physics.robotics import (
        calc_safe_stopping_distance,
        calc_transient_impact_force,
        calc_kinetic_energy,
    )

    m_eff = 12.0 * kilogram
    v_max = 1.71 * (meter / second)
    k_tissue = 15000.0 * (newton / meter)
    f_iso_limit = 50.0 * newton

    t_detect_nominal = 2.0 * millisecond
    a_brake_spring = 45.0 * (meter / (second**2))
    delta_t_contamination = 12.0 * millisecond

    e_k = calc_kinetic_energy(m_eff, v_max)
    f_unbraked = calc_transient_impact_force(v_max, m_eff, k_tissue)
    f_ratio = (f_unbraked / f_iso_limit).magnitude

    d_drift_nom = (v_max * t_detect_nominal).to(millimeter)
    t_brake_nom = (v_max / a_brake_spring).to(millisecond)
    d_brake_nom = ((v_max**2) / (2.0 * a_brake_spring)).to(millimeter)
    t_stop_nom = t_detect_nominal + t_brake_nom
    d_stop_nom = d_drift_nom + d_brake_nom

    t_detect_delayed = t_detect_nominal + delta_t_contamination
    d_drift_delayed = (v_max * t_detect_delayed).to(millimeter)
    d_stop_delayed = d_drift_delayed + d_brake_nom
    t_stop_delayed = t_detect_delayed + t_brake_nom
    d_breach = d_stop_delayed - d_stop_nom
    expansion_pct = ((d_stop_delayed - d_stop_nom) / d_stop_nom).magnitude * 100.0

    check(abs(e_k.to(joule).magnitude - 17.54) < 0.02, f"Unexpected kinetic energy: {e_k}")
    check(abs(f_unbraked.to(newton).magnitude - 725.5) < 0.2, f"Unexpected unbraked force: {f_unbraked}")
    check(abs(f_ratio - 14.51) < 0.1, f"Unexpected force ratio: {f_ratio}")
    check(abs(d_drift_nom.to(millimeter).magnitude - 3.42) < 0.01, f"Unexpected nominal drift: {d_drift_nom}")
    check(abs(t_brake_nom.to(millisecond).magnitude - 38.0) < 0.1, f"Unexpected brake time: {t_brake_nom}")
    check(abs(d_brake_nom.to(millimeter).magnitude - 32.49) < 0.02, f"Unexpected brake distance: {d_brake_nom}")
    check(abs(t_stop_nom.to(millisecond).magnitude - 40.0) < 0.1, f"Unexpected total nominal stop time: {t_stop_nom}")
    check(abs(d_stop_nom.to(millimeter).magnitude - 35.91) < 0.02, f"Unexpected nominal stop distance: {d_stop_nom}")
    check(abs(d_drift_delayed.to(millimeter).magnitude - 23.94) < 0.01, f"Unexpected delayed drift: {d_drift_delayed}")
    check(abs(d_stop_delayed.to(millimeter).magnitude - 56.43) < 0.02, f"Unexpected delayed stop distance: {d_stop_delayed}")
    check(abs(t_stop_delayed.to(millisecond).magnitude - 52.0) < 0.1, f"Unexpected delayed stop time: {t_stop_delayed}")
    check(abs(d_breach.to(millimeter).magnitude - 20.52) < 0.1, f"Unexpected breach distance: {d_breach}")
    check(abs(expansion_pct - 57.14) < 0.2, f"Unexpected expansion pct: {expansion_pct}")


def test_vol4_ch17_architectural_shield_dilution():
    from mlsysim.physics.robotics import (
        calc_empirical_testing_exposure,
        calc_shielded_system_hazard_rate,
    )

    p_statutory_target = 1.0e-9  # failures / hour
    statutory_confidence = 0.95
    alpha_statutory = 1.0 - statutory_confidence

    fleet_units = 100
    t_cycle_days = 90.0
    hours_per_day = 24.0
    hours_per_year = 8766.0

    c_shield_spec = 0.9999

    n_astronomical_hours = calc_empirical_testing_exposure(p_statutory_target, confidence=statutory_confidence)
    machine_years_astronomical = n_astronomical_hours.magnitude / hours_per_year
    fleet_calendar_years = machine_years_astronomical / fleet_units

    t_cycle_hours = t_cycle_days * hours_per_day
    n_cycle_hours = fleet_units * t_cycle_hours
    p_emp_cycle = -math.log(alpha_statutory) / n_cycle_hours
    ratio_empirical_to_statutory = p_emp_cycle / p_statutory_target

    p_brain_relaxed = p_statutory_target / (1.0 - c_shield_spec)
    n_brain_hours = calc_empirical_testing_exposure(p_brain_relaxed, confidence=statutory_confidence)
    t_test_days = n_brain_hours.magnitude / (fleet_units * hours_per_day)
    t_test_months = t_test_days / (365.25 / 12.0)

    check(abs(n_astronomical_hours.magnitude - 2.9957e9) < 1e7, f"Unexpected astronomical exposure: {n_astronomical_hours}")
    check(abs(machine_years_astronomical - 341744.0) < 500.0, f"Unexpected machine years: {machine_years_astronomical}")
    check(abs(fleet_calendar_years - 3417.4) < 10.0, f"Unexpected fleet calendar years: {fleet_calendar_years}")
    check(abs(n_cycle_hours - 216000.0) < 1.0, f"Unexpected cycle hours: {n_cycle_hours}")
    check(abs(p_emp_cycle - 1.3869e-5) < 1e-8, f"Unexpected empirical p: {p_emp_cycle}")
    check(abs(ratio_empirical_to_statutory - 13869.0) < 10.0, f"Unexpected empirical ratio: {ratio_empirical_to_statutory}")
    check(abs(p_brain_relaxed - 1.0e-5) < 1e-9, f"Unexpected relaxed p_brain: {p_brain_relaxed}")
    check(abs(n_brain_hours.magnitude - 299573.0) < 100.0, f"Unexpected brain exposure: {n_brain_hours}")
    check(abs(t_test_days - 124.8) < 1.0, f"Unexpected test days: {t_test_days}")
    check(abs(t_test_months - 4.1) < 0.1, f"Unexpected test months: {t_test_months}")
