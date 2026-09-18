"""Unit tests verifying that Volume IV LEGO cells execute cleanly and their guards pass."""

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
    fmt,
    check,
)


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
