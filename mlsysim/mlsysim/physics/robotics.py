"""Physical AI and embodied ML systems physics and safety accounting formulas (Volume IV).

Domain scope:
- Sensor-to-actuator end-to-end latency budgets
- Dynamic stopping distance and kinetic inertia safety bounds
- Actuator stator thermal dissipation and power envelopes
- Action chunking rate adaptation and control loop timing
- Reflected rotor inertia and gear ratios
"""

from __future__ import annotations

import math
import numpy as np

from mlsysim.core.units import ureg
from mlsysim.core._validation import (
    validate_positive,
    validate_nonnegative,
    validate_at_least,
)


def calc_sensor_to_actuator_latency(
    sensor_latency,
    inference_latency,
    arbitration_latency,
    actuator_latency,
):
    """
    Calculate the total end-to-end perception-to-actuation loop latency.

    Source: modeling assumption. The stages are summed as a serial pipeline,
    not taken from a published model; sampling phase, jitter, and queueing
    between stages are ignored.

    Parameters
    ----------
    sensor_latency : Quantity
        Sensor capture and transmission delay (e.g. camera frame exposure + USB/CSI bus).
    inference_latency : Quantity
        Policy inference duration on neural accelerator (e.g. VLA, Diffusion Policy).
    arbitration_latency : Quantity
        Deterministic nervous system / Control Barrier Function arbitration delay.
    actuator_latency : Quantity
        Motor controller response time and CAN/EtherCAT bus transit delay.

    Returns
    -------
    Quantity
        Total closed-loop latency in milliseconds or seconds.
    """
    total = sensor_latency + inference_latency + arbitration_latency + actuator_latency
    return total.to(ureg.millisecond)


def calc_safe_stopping_distance(
    velocity,
    total_latency,
    max_deceleration,
    loc_margin=None,
    physical_margin=None,
):
    """
    Calculate dynamic stopping distance under non-zero control latency.

    Equation:
        d_stop = v * tau_latency + (v^2) / (2 * a_max) + delta_loc + delta_margin

    Source: reaction distance v * tau plus constant-deceleration braking
    distance v^2 / (2 * a), from v^2 = v0^2 + 2 * a * (x - x0) (OpenStax,
    University Physics Volume 1, Sec. 3.4, Eq. 3.14; Example 3.10 adds
    reaction-time travel to braking distance the same way). The additive
    localization and clearance margins are a modeling assumption.

    Parameters
    ----------
    velocity : Quantity
        Current operational velocity (e.g. m/s).
    total_latency : Quantity
        End-to-end perception-to-brake latency (e.g. ms or s).
    max_deceleration : Quantity
        Maximum braking deceleration available (e.g. m/s^2).
    loc_margin : Quantity, optional
        Localization uncertainty bound (e.g. cm or m).
    physical_margin : Quantity, optional
        Physical obstacle clearance safety margin (e.g. cm or m).

    Returns
    -------
    Quantity
        Total stopping distance in meters.
    """
    validate_nonnegative(velocity, "velocity")
    validate_positive(total_latency, "total_latency")
    validate_positive(max_deceleration, "max_deceleration")

    reaction_distance = velocity * total_latency
    braking_distance = (velocity ** 2) / (2 * max_deceleration)
    total_distance = reaction_distance + braking_distance

    if loc_margin is not None:
        validate_nonnegative(loc_margin, "loc_margin")
        total_distance = total_distance + loc_margin
    if physical_margin is not None:
        validate_nonnegative(physical_margin, "physical_margin")
        total_distance = total_distance + physical_margin

    return total_distance.to(ureg.meter)


def calc_max_permitted_velocity(
    d_clear,
    total_latency,
    max_deceleration,
    loc_margin=None,
    physical_margin=None,
):
    """
    Calculate maximum safe forward velocity admitted by physical stopping clearance.

    Solves the quadratic inequality:
        v * tau + v^2 / (2 * a) <= D_clear - delta_loc - delta_margin
    for the maximum non-negative velocity root:
        v_max = -a * tau + sqrt((a * tau)^2 + 2 * a * D_eff)
    where D_eff = D_clear - delta_loc - delta_margin. If D_eff <= 0, v_max = 0.

    Source: derived here as the positive root of the
    calc_safe_stopping_distance quadratic, so it rests on the same kinematics
    (OpenStax, University Physics Volume 1, Sec. 3.4, Eq. 3.14) and the same
    additive-margin modeling assumption.

    Parameters
    ----------
    d_clear : Quantity
        Detected clearance distance to obstacle along travel vector (e.g. m).
    total_latency : Quantity
        End-to-end sense-to-actuation latency tau_delay (e.g. s or ms).
    max_deceleration : Quantity
        Maximum braking deceleration capacity a_brake (e.g. m/s^2).
    loc_margin : Quantity, optional
        Localization uncertainty bound (e.g. cm or m).
    physical_margin : Quantity, optional
        Physical clearance safety buffer (e.g. cm or m).

    Returns
    -------
    Quantity
        Maximum safe forward velocity in meters/second.
    """
    validate_nonnegative(d_clear, "d_clear")
    validate_positive(total_latency, "total_latency")
    validate_positive(max_deceleration, "max_deceleration")

    d_eff = d_clear
    if loc_margin is not None:
        validate_nonnegative(loc_margin, "loc_margin")
        d_eff = d_eff - loc_margin
    if physical_margin is not None:
        validate_nonnegative(physical_margin, "physical_margin")
        d_eff = d_eff - physical_margin

    if d_eff.to(ureg.meter).magnitude <= 0.0:
        return (0.0 * (ureg.meter / ureg.second)).to(ureg.meter / ureg.second)

    a_tau = max_deceleration * total_latency
    discriminant = (a_tau ** 2) + 2.0 * max_deceleration * d_eff
    v_max = -a_tau + discriminant ** 0.5

    v_mps = max(0.0, v_max.to(ureg.meter / ureg.second).magnitude)
    return (v_mps * (ureg.meter / ureg.second)).to(ureg.meter / ureg.second)


def calc_kinetic_energy(mass, velocity):
    """
    Calculate kinetic energy of a translating body.

    Equation:
        E_k = 0.5 * m * v^2

    Source: OpenStax, University Physics Volume 1, Sec. 7.2, Eq. 7.6.

    Parameters
    ----------
    mass : Quantity
        Total mass of the body (e.g. kg).
    velocity : Quantity
        Translational velocity (e.g. m/s).

    Returns
    -------
    Quantity
        Kinetic energy in Joules.
    """
    validate_positive(mass, "mass")
    validate_nonnegative(velocity, "velocity")

    energy = 0.5 * mass * (velocity ** 2)
    return energy.to(ureg.joule)


def calc_seam_acceleration_torque_jump(gear_ratio: float, rotor_inertia, delta_accel):
    """
    Calculate the impulsive torque spike at an actuator seam transition.

    When an unblended trajectory transition injects an acceleration discontinuity
    delta_accel across a discrete boundary, the transmission's reflected rotor inertia
    amplifies the required inertial reaction torque at the output joint.

    Equation:
        tau_spike = J_reflected * delta_accel = (N^2 * J_rotor) * delta_accel

    Source: tau = I * alpha (OpenStax, University Physics Volume 1, Sec. 10.7,
    Eq. 10.25) applied to the rotor's apparent (reflected) inertia
    G^2 * I_rotor (Lynch and Park, Modern Robotics, Cambridge University
    Press, 2017, Sec. 8.9.2). Link inertia, load, friction, and gear
    compliance are omitted, so this is the reflected-rotor share of the spike;
    treating the seam as an instantaneous acceleration step is a modeling
    assumption.

    Parameters
    ----------
    gear_ratio : float
        Transmission reduction ratio N (>= 1.0).
    rotor_inertia : Quantity
        Unloaded rotor moment of inertia (e.g. kg * m^2).
    delta_accel : Quantity
        Discontinuous acceleration step across the seam (e.g. rad / s^2).

    Returns
    -------
    Quantity
        Peak reaction torque spike in Newton-meters.
    """
    validate_at_least(gear_ratio, 1.0, "gear_ratio")
    validate_positive(rotor_inertia, "rotor_inertia")
    validate_nonnegative(delta_accel, "delta_accel")

    reflected_inertia = calc_reflected_inertia(gear_ratio, rotor_inertia)
    torque_spike = reflected_inertia * delta_accel
    return torque_spike.to(ureg.newton * ureg.meter)


def calc_inverted_pendulum_fall_time(
    effective_length,
    gravity=None,
    theta_0=None,
    theta_fall=None,
):
    """
    Calculate characteristic instability timescale or fall duration for an inverted pendulum.

    Models bipedal/humanoid balance dynamics under gravity. The linearized dynamics
    d^2 theta / dt^2 = (g / L) * theta yield the characteristic natural timescale:
        tau_0 = sqrt(L / g)

    If initial perturbation theta_0 and fall threshold theta_fall are given:
        t_fall = tau_0 * ln(2 * theta_fall / theta_0)

    Source: the linear inverted pendulum, whose natural frequency is
    omega_0 = sqrt(g / h), so tau_0 = sqrt(L / g) (Caron, "Biped Stabilization
    by Linear Feedback of the Variable-Height Inverted Pendulum Model",
    arXiv:1909.07732, Sec. II). The t_fall expression is derived here, not
    taken from a published model. With zero initial angular velocity the
    linearized solution is theta(t) = theta_0 * cosh(t / tau_0), so
    t = tau_0 * acosh(theta_fall / theta_0), and ln(2x) approximates acosh(x)
    for x >> 1 (it overestimates by about 5% at x = 2). The small-angle
    linearization also loses accuracy as theta_fall grows.

    Parameters
    ----------
    effective_length : Quantity
        Center-of-mass height or pendulum leg length L (e.g. meter).
    gravity : Quantity, optional
        Gravitational acceleration g (defaults to 9.80665 m/s^2).
    theta_0 : Quantity or float, optional
        Initial angular perturbation from vertical (e.g. radians or degrees).
    theta_fall : Quantity or float, optional
        Critical tipping angle threshold beyond which recovery is kinematically impossible.

    Returns
    -------
    Quantity
        Characteristic timescale or fall duration in seconds.
    """
    import math

    validate_positive(effective_length, "effective_length")
    if gravity is None:
        gravity = 9.80665 * (ureg.meter / (ureg.second ** 2))
    else:
        validate_positive(gravity, "gravity")

    tau_0 = ((effective_length / gravity) ** 0.5).to(ureg.second)

    if theta_0 is not None and theta_fall is not None:
        th0 = theta_0.to(ureg.radian).magnitude if hasattr(theta_0, "to") else float(theta_0)
        th_f = theta_fall.to(ureg.radian).magnitude if hasattr(theta_fall, "to") else float(theta_fall)
        if th0 <= 0.0 or th_f <= th0:
            raise ValueError(f"theta_fall ({th_f}) must be greater than theta_0 ({th0}) > 0")
        t_fall = tau_0 * math.log(2.0 * th_f / th0)
        return t_fall.to(ureg.second)

    return tau_0.to(ureg.second)


def calc_actuator_thermal_power(current, resistance, duty_cycle: float = 1.0):
    """
    Calculate average ohmic stator heat dissipation in electromagnetic actuators.

    Equation:
        P_loss = (I^2 * R) * duty_cycle

    Source: Joule heating P = I^2 * R (OpenStax, University Physics Volume 2,
    Sec. 9.5, Eq. 9.13) for one winding at constant resistance. Scaling by
    duty_cycle is a modeling assumption; it takes the current as constant
    while on and zero while off, and ignores the rise of winding resistance
    with temperature.

    Parameters
    ----------
    current : Quantity
        Phase current flowing through actuator windings (e.g. Amperes).
    resistance : Quantity
        Phase winding resistance (e.g. Ohms).
    duty_cycle : float, optional
        Operating duty cycle fraction (0.0 to 1.0).

    Returns
    -------
    Quantity
        Thermal loss power in Watts.
    """
    validate_positive(resistance, "resistance")
    validate_nonnegative(current, "current")

    power = (current ** 2) * resistance * duty_cycle
    return power.to(ureg.watt)


def calc_action_chunk_cadence(chunk_horizon_steps: int, control_loop_hz, brain_inference_hz):
    """
    Evaluate the freshness and execution headroom of an action chunking policy.

    Source: action chunking, predicting a sequence of actions per policy query,
    is from Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
    Low-Cost Hardware" (ACT), arXiv:2304.13705. The chunk-duration and
    headroom arithmetic (chunk steps over control rate, divided by the
    inference period) is a modeling assumption of this package, not taken
    from that paper.

    Parameters
    ----------
    chunk_horizon_steps : int
        Number of consecutive actions predicted in a single model inference chunk.
    control_loop_hz : Quantity
        Execution frequency of the low-level nervous system / motor loop (e.g. Hz).
    brain_inference_hz : Quantity
        Inference rate of the high-level neural policy (e.g. Hz).

    Returns
    -------
    dict
        Execution window duration, inference budget, and headroom factor.
    """
    validate_at_least(chunk_horizon_steps, 1, "chunk_horizon_steps")
    validate_positive(control_loop_hz, "control_loop_hz")
    validate_positive(brain_inference_hz, "brain_inference_hz")

    step_period = (1.0 / control_loop_hz).to(ureg.second)
    chunk_duration = chunk_horizon_steps * step_period
    brain_period = (1.0 / brain_inference_hz).to(ureg.second)

    headroom_factor = (chunk_duration / brain_period).to_base_units().magnitude

    return {
        "chunk_duration": chunk_duration.to(ureg.millisecond),
        "brain_period": brain_period.to(ureg.millisecond),
        "headroom_factor": headroom_factor,
    }


def calc_reflected_inertia(gear_ratio: float, rotor_inertia):
    """
    Calculate reflected rotor inertia felt at the actuator output shaft.

    Equation:
        J_reflected = N^2 * J_rotor

    Source: Lynch and Park, Modern Robotics, Cambridge University Press, 2017,
    Sec. 8.9.2 ("Apparent Inertia"), where G^2 * I_rotor is the rotor's
    apparent (often called reflected) inertia for gear ratio G.

    Parameters
    ----------
    gear_ratio : float
        Transmission reduction ratio N (>= 1.0).
    rotor_inertia : Quantity
        Unloaded rotor moment of inertia (e.g. kg * m^2).

    Returns
    -------
    Quantity
        Reflected inertia in kg * m^2.
    """
    validate_at_least(gear_ratio, 1.0, "gear_ratio")
    reflected = (gear_ratio ** 2) * rotor_inertia
    return reflected


def calc_spatial_information_age_displacement(velocity, information_age):
    """
    Calculate unmodeled spatial displacement of a moving body across perception pipeline delay.

    Equation:
        Delta_x = v * tau_age

    Source: OpenStax, University Physics Volume 1, Sec. 3.1. Physical AI perception pipeline
    waterfall (exposure, readout, transport, DMA, ISP, backbone, IPC).

    Parameters
    ----------
    velocity : Quantity
        Operational forward or end-effector translational velocity (e.g. m/s).
    information_age : Quantity
        Total sensor-to-inference latency / information age tau_age (e.g. ms or s).

    Returns
    -------
    Quantity
        Unmodeled physical displacement in meters.
    """
    validate_nonnegative(velocity, "velocity")
    validate_positive(information_age, "information_age")
    dx = velocity * information_age
    return dx.to(ureg.meter)


def calc_empirical_testing_exposure(target_hazard_rate, confidence: float = 0.95):
    """
    Calculate required zero-failure operating exposure to statistically bound a hazard rate.

    Under a Poisson failure arrival process with constant rate lambda, the probability
    of observing zero failures over total exposure duration T is P(0) = exp(-lambda * T).
    To claim lambda <= lambda_target at statistical confidence C = 1 - alpha:
        exp(-lambda_target * T) <= 1 - C  ==>  T >= -ln(1 - C) / lambda_target

    For 95% confidence (C = 0.95):
        -ln(0.05) ≈ 2.99573  ==>  T ≈ 3.0 / lambda_target

    Source: Littlewood and Wright, "Some Conservative Stopping Rules for
    Testing Software", IEEE TSE, 1997; Kalbfleisch and Prentice, The Statistical
    Analysis of Failure Time Data, Wiley, 2002.

    Parameters
    ----------
    target_hazard_rate : Quantity or float
        Target acceptable failure rate (e.g. 1e-6 / hour or 1e-9 / hour).
        If float, interpreted as failures per hour.
    confidence : float, optional
        Statistical confidence level (default 0.95 for 95% two-sided/one-sided bound).

    Returns
    -------
    Quantity
        Required continuous operating hours.
    """
    import math

    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if hasattr(target_hazard_rate, "magnitude"):
        rate_per_hr = target_hazard_rate.to(1 / ureg.hour).magnitude
    else:
        rate_per_hr = float(target_hazard_rate)

    validate_positive(rate_per_hr, "target_hazard_rate")

    factor = -math.log(1.0 - confidence)
    hours = factor / rate_per_hr
    return hours * ureg.hour


def calc_dense_voxel_grid_memory(voxels_per_dim: int, bytes_per_voxel: int = 4):
    """
    Calculate volumetric memory footprint for a dense 3D cubic voxel grid.

    Equation:
        Memory = (N_voxels)^3 * bytes_per_voxel

    Source: standard cubic voxel discretization, O(N^3) spatial complexity.

    Parameters
    ----------
    voxels_per_dim : int
        Grid resolution along one dimension (e.g. 512 or 1024).
    bytes_per_voxel : int, optional
        Storage bytes per voxel state (default 4 bytes for float32 occupancy/TSDF).

    Returns
    -------
    Quantity
        Memory footprint in bytes (auto-scalable to MB / GB).
    """
    validate_at_least(voxels_per_dim, 1, "voxels_per_dim")
    validate_at_least(bytes_per_voxel, 1, "bytes_per_voxel")

    total_bytes = (voxels_per_dim ** 3) * bytes_per_voxel
    return total_bytes * ureg.byte


def calc_3dgs_memory_footprint(num_gaussians: int, bytes_per_gaussian: int = 56):
    """
    Calculate spatial state memory footprint for 3D Gaussian Splatting (3DGS).

    Equation:
        Memory = K_gaussians * bytes_per_gaussian

    Parameters
    ----------
    num_gaussians : int
        Number of 3D Gaussian ellipsoids in the spatial memory buffer.
    bytes_per_gaussian : int, optional
        Storage footprint per Gaussian (position 12B, covariance/scale/rotation 16B,
        opacity 4B, spherical harmonics color 24B = 56 bytes standard).

    Returns
    -------
    Quantity
        Memory footprint in bytes.
    """
    validate_at_least(num_gaussians, 1, "num_gaussians")
    validate_at_least(bytes_per_gaussian, 1, "bytes_per_gaussian")

    total_bytes = num_gaussians * bytes_per_gaussian
    return total_bytes * ureg.byte


def calc_voltage_droop(current, internal_resistance):
    """
    Calculate DC bus voltage droop across internal battery and cable resistance.

    Equation:
        Delta_V = I * R_internal

    Source: Ohm's law, V = I * R (OpenStax, University Physics Volume 2, Sec. 9.3).

    Parameters
    ----------
    current : Quantity
        Total instantaneous draw current across actuators and compute (e.g. Amperes).
    internal_resistance : Quantity
        Combined battery internal DC resistance and harness resistance (e.g. Ohms).

    Returns
    -------
    Quantity
        Voltage droop in Volts.
    """
    validate_nonnegative(current, "current")
    validate_positive(internal_resistance, "internal_resistance")

    droop = current * internal_resistance
    return droop.to(ureg.volt)


def calc_cbf_safety_margin(h_val, l_f_h, l_g_h, u_cmd, alpha_coeff: float = 1.0):
    """
    Evaluate the forward-invariance safety margin of a Control Barrier Function (CBF).

    Equation:
        psi = L_f h(x) + L_g h(x) * u + alpha * h(x)

    When psi >= 0, the candidate control command u guarantees that the system state
    remains forward-invariant within the safe set C = {x : h(x) >= 0}.

    Source: Ames et al., "Control Barrier Functions: Theory and Applications",
    IEEE ECC, 2019.

    Parameters
    ----------
    h_val : float or Quantity
        Current barrier function evaluation h(x) (dimensionless or margin).
    l_f_h : float or Quantity
        Lie derivative along uncontrolled drift vector field L_f h(x).
    l_g_h : float or Quantity
        Lie derivative along control input vector field L_g h(x).
    u_cmd : float or Quantity
        Candidate actuation setpoint command proposed by the neural policy.
    alpha_coeff : float, optional
        Class-K gain coefficient alpha (default 1.0).

    Returns
    -------
    dict
        Safety margin psi, forward-invariant boolean status, and barrier value.
    """
    margin = l_f_h + (l_g_h * u_cmd) + (alpha_coeff * h_val)
    is_safe = margin.magnitude >= 0 if hasattr(margin, "magnitude") else margin >= 0
    return {
        "psi": margin,
        "is_safe": bool(is_safe),
        "h_x": h_val,
    }


def calc_action_chunk_streaming_amortization(
    model_weight_memory,
    memory_bandwidth,
    chunk_horizon: int,
    bus_efficiency: float = 0.70,
):
    """
    Calculate memory streaming latency and amortized control rate for action chunking.

    Under memory-bandwidth bound autoregressive weight streaming:
        t_stream = M_weights / (BW_peak * eta_bus)
        f_single = 1 / t_stream
        tau_step = t_stream / H
        f_effective = 1 / tau_step = H * f_single

    Source: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware",
    arXiv:2304.13705; Volume IV Chapter 3.

    Parameters
    ----------
    model_weight_memory : Quantity
        Model parameter storage footprint (e.g. GB).
    memory_bandwidth : Quantity
        Peak memory bus bandwidth (e.g. GB/s).
    chunk_horizon : int
        Number of steps predicted per chunk H (>= 1).
    bus_efficiency : float, optional
        Sustained DRAM efficiency factor eta (default 0.70).

    Returns
    -------
    dict
        t_stream, f_single, tau_step, f_effective, b_sustained.
    """
    validate_at_least(chunk_horizon, 1, "chunk_horizon")
    validate_positive(model_weight_memory, "model_weight_memory")
    validate_positive(memory_bandwidth, "memory_bandwidth")

    b_sustained = memory_bandwidth * bus_efficiency
    t_stream = (model_weight_memory / b_sustained).to(ureg.second)
    f_single = (1.0 / t_stream).to(ureg.hertz)

    tau_step = (t_stream / chunk_horizon).to(ureg.second)
    f_effective = (1.0 / tau_step).to(ureg.hertz)

    return {
        "t_stream": t_stream.to(ureg.millisecond),
        "f_single": f_single,
        "tau_step": tau_step.to(ureg.millisecond),
        "f_effective": f_effective,
        "b_sustained": b_sustained.to(ureg.GB / ureg.second),
    }


def calc_watchdog_lease_bound(
    clearance_distance,
    initial_velocity,
    max_deceleration,
):
    """
    Calculate maximum permissible communication / control lease timeout before emergency brake.

    Under constant velocity v0 during uncommanded drift, followed by emergency braking
    at constant deceleration a_max:
        d_brake = (v0^2) / (2 * a_max)
        d_drift = d_clear - d_brake
        T_lease <= d_drift / v0

    Parameters
    ----------
    clearance_distance : Quantity
        Defended physical clearance / sensory safety field ahead of obstacle (e.g. m).
    initial_velocity : Quantity
        Cruise velocity of the vehicle / mobile base v0 (e.g. m/s).
    max_deceleration : Quantity
        Certified worst-case emergency braking deceleration a_max (e.g. m/s^2).

    Returns
    -------
    dict
        t_lease_max, d_brake, d_drift_allowable.
    """
    validate_positive(clearance_distance, "clearance_distance")
    validate_positive(initial_velocity, "initial_velocity")
    validate_positive(max_deceleration, "max_deceleration")

    v0 = initial_velocity.to(ureg.meter / ureg.second)
    a = max_deceleration.to(ureg.meter / (ureg.second**2))
    d_clear = clearance_distance.to(ureg.meter)

    d_brake = ((v0**2) / (2.0 * a)).to(ureg.meter)
    if d_clear < d_brake:
        raise ValueError(
            f"Clearance {d_clear} is insufficient to stop even with zero lease timeout "
            f"(braking distance requires {d_brake})."
        )

    d_drift = d_clear - d_brake
    t_lease = (d_drift / v0).to(ureg.second)

    return {
        "t_lease_max": t_lease.to(ureg.millisecond),
        "d_brake": d_brake.to(ureg.meter),
        "d_drift_allowable": d_drift.to(ureg.meter),
    }


def calc_canfd_bus_utilization(
    num_axes: int,
    payload_bytes: int = 64,
    baud_arbitration=None,
    baud_data=None,
    cycle_time=None,
    interframe_spacing=None,
    overhead_bits_arb: int = 32,
    overhead_bits_data: int = 0,
):
    """
    Calculate serialization latency and bus utilization for cyclic CAN-FD transmission.

    CAN-FD employs a dual-bitrate scheme: an arbitration phase at baud_arbitration
    (standard 1 Mbps) and a high-speed data phase at baud_data (e.g. 5 Mbps).

    Equation:
        t_arb = overhead_bits_arb / baud_arbitration
        t_data = (payload_bytes * 8 + overhead_bits_data) / baud_data
        t_tx = t_arb + t_data + interframe_spacing
        t_total_tx = num_axes * t_tx
        rho = t_total_tx / cycle_time

    Parameters
    ----------
    num_axes : int
        Number of cyclic node/axis messages scheduled per loop (>= 1).
    payload_bytes : int, optional
        Data field payload size in bytes (default 64 bytes for CAN-FD).
    baud_arbitration : Quantity, optional
        Nominal arbitration phase bitrate (default 1.0 Mbps).
    baud_data : Quantity, optional
        Data phase bitrate (default 5.0 Mbps).
    cycle_time : Quantity, optional
        Total control cycle period (default 1.0 ms).
    interframe_spacing : Quantity, optional
        Inter-frame spacing and bus idle interval (default 6.0 us).
    overhead_bits_arb : int, optional
        Arbitration, header, and CRC delimiter bit count (default 32 bits).
    overhead_bits_data : int, optional
        Data phase stuff bits and CRC bit overhead (default 0).

    Returns
    -------
    dict
        t_tx, t_total_tx, utilization, is_schedulable.
    """
    validate_at_least(num_axes, 1, "num_axes")
    validate_at_least(payload_bytes, 0, "payload_bytes")

    baud_arb = (1.0 * (ureg.megabit / ureg.second)) if baud_arbitration is None else baud_arbitration
    baud_dat = (5.0 * (ureg.megabit / ureg.second)) if baud_data is None else baud_data
    t_cycle = (1.0 * ureg.millisecond) if cycle_time is None else cycle_time
    t_ifs = (6.0 * ureg.microsecond) if interframe_spacing is None else interframe_spacing

    bits_arb = overhead_bits_arb * ureg.bit
    bits_data = (payload_bytes * 8 + overhead_bits_data) * ureg.bit

    t_arb = (bits_arb / baud_arb).to(ureg.microsecond)
    t_data = (bits_data / baud_dat).to(ureg.microsecond)
    t_tx = t_arb + t_data + t_ifs.to(ureg.microsecond)
    t_total_tx = num_axes * t_tx

    rho = (t_total_tx / t_cycle).to(ureg.dimensionless).magnitude

    return {
        "t_tx": t_tx.to(ureg.microsecond),
        "t_total_tx": t_total_tx.to(ureg.microsecond),
        "utilization": float(rho),
        "is_schedulable": bool(rho <= 1.0),
    }


def calc_ethercat_cycle_time(
    num_axes: int,
    bytes_per_axis: int = 32,
    baud_rate=None,
    header_bytes: int = 64,
    hop_delay=None,
    cycle_time=None,
):
    """
    Calculate on-the-fly hardware processing latency and bus utilization for EtherCAT daisy chain.

    In an EtherCAT daisy-chain ring, a single standard Ethernet frame passes through all
    slave ESCs (EtherCAT SubDevice Controllers). Each node processes and injects PDO data
    on-the-fly with hardware forwarding latency hop_delay (typically 0.6 us).

    Equation:
        total_frame_bytes = header_bytes + (num_axes * bytes_per_axis)
        t_tx = (total_frame_bytes * 8) / baud_rate
        t_forward = num_axes * hop_delay
        t_cycle_total = t_tx + t_forward
        rho = t_cycle_total / cycle_time
        t_margin = cycle_time - t_cycle_total

    Parameters
    ----------
    num_axes : int
        Number of slave joint nodes in the ring (>= 1).
    bytes_per_axis : int, optional
        Process data object (PDO) payload bytes per axis (default 32 bytes).
    baud_rate : Quantity, optional
        Fast Ethernet line rate (default 100.0 Mbps).
    header_bytes : int, optional
        Standard Ethernet preamble, SFD, MAC, EtherCAT header, FCS bytes (default 64 bytes).
    hop_delay : Quantity, optional
        Per-node ASIC hardware processing forwarding delay (default 0.6 us).
    cycle_time : Quantity, optional
        Total control cycle loop deadline (default 1.0 ms).

    Returns
    -------
    dict
        t_tx, t_forward, t_cycle_total, t_margin, utilization, is_schedulable.
    """
    validate_at_least(num_axes, 1, "num_axes")
    validate_at_least(bytes_per_axis, 1, "bytes_per_axis")

    baud = (100.0 * (ureg.megabit / ureg.second)) if baud_rate is None else baud_rate
    t_hop = (0.6 * ureg.microsecond) if hop_delay is None else hop_delay
    t_loop = (1.0 * ureg.millisecond) if cycle_time is None else cycle_time

    total_bytes = header_bytes + (num_axes * bytes_per_axis)
    bits = (total_bytes * 8) * ureg.bit

    t_tx = (bits / baud).to(ureg.microsecond)
    t_forward = (num_axes * t_hop).to(ureg.microsecond)
    t_total = t_tx + t_forward
    t_margin = (t_loop.to(ureg.microsecond) - t_total).to(ureg.microsecond)

    rho = (t_total / t_loop).to(ureg.dimensionless).magnitude

    return {
        "t_tx": t_tx.to(ureg.microsecond),
        "t_forward": t_forward.to(ureg.microsecond),
        "t_cycle_total": t_total.to(ureg.microsecond),
        "t_margin": t_margin.to(ureg.microsecond),
        "utilization": float(rho),
        "is_schedulable": bool(rho <= 1.0),
    }


def calc_contact_force(stiffness, displacement):
    """
    Calculate linear Hookean elastic contact force upon unconstrained impact.

    Equation:
        F = k * delta_x
    """
    validate_positive(stiffness, "stiffness")
    validate_nonnegative(displacement, "displacement")
    f = stiffness.to(ureg.newton / ureg.meter) * displacement.to(ureg.meter)
    return f.to(ureg.newton)


def calc_contact_yield_deadline(yield_force, stiffness, approach_velocity):
    """
    Calculate maximum allowable software reaction latency before impact force exceeds mechanical yield.

    For an arm approaching a rigid fixture of stiffness k at constant velocity v,
    contact force rises as F(t) = k * v * t. To prevent structural damage (F <= F_yield):
        t_deadline = F_yield / (k * v)

    Parameters
    ----------
    yield_force : Quantity
        Maximum permissible mechanical shock / yield force (e.g. N).
    stiffness : Quantity
        Contact environment stiffness (e.g. N/m).
    approach_velocity : Quantity
        Tool-center-point approach velocity (e.g. m/s).

    Returns
    -------
    Quantity
        Deadline duration in milliseconds.
    """
    validate_positive(yield_force, "yield_force")
    validate_positive(stiffness, "stiffness")
    validate_positive(approach_velocity, "approach_velocity")

    f = yield_force.to(ureg.newton)
    k = stiffness.to(ureg.newton / ureg.meter)
    v = approach_velocity.to(ureg.meter / ureg.second)

    t = f / (k * v)
    return t.to(ureg.millisecond)


def calc_optimal_gear_ratio(load_inertia, rotor_inertia) -> float:
    """
    Calculate inertia matching gear ratio maximizing joint output angular acceleration.

    Equation:
        N* = sqrt(J_load / J_rotor)
    """
    validate_positive(load_inertia, "load_inertia")
    validate_positive(rotor_inertia, "rotor_inertia")

    j_l = load_inertia.to(ureg.kg * (ureg.meter**2))
    j_r = rotor_inertia.to(ureg.kg * (ureg.meter**2))

    n_star = math.sqrt((j_l / j_r).magnitude)
    return float(n_star)


def calc_geared_joint_acceleration(gear_ratio: float, motor_torque, load_inertia, rotor_inertia):
    """
    Calculate joint output angular acceleration under geared transmission.

    Equation:
        ddot_q = (N * tau_motor) / (J_load + N^2 * J_rotor)
    """
    validate_positive(gear_ratio, "gear_ratio")
    validate_positive(motor_torque, "motor_torque")
    validate_positive(load_inertia, "load_inertia")
    validate_positive(rotor_inertia, "rotor_inertia")

    tau = motor_torque.to(ureg.newton * ureg.meter)
    j_l = load_inertia.to(ureg.kg * (ureg.meter**2))
    j_r = rotor_inertia.to(ureg.kg * (ureg.meter**2))

    j_total = j_l + (gear_ratio**2) * j_r
    alpha = (gear_ratio * tau) / j_total
    return alpha.to(ureg.radian / (ureg.second**2))


def calc_inductive_voltage_droop(loop_inductance, current_slew_rate):
    """
    Calculate inductive voltage drop across power distribution network (PDN) during current transients.

    Equation:
        Delta_V_ind = L * (dI / dt)
    """
    validate_positive(loop_inductance, "loop_inductance")
    validate_positive(current_slew_rate, "current_slew_rate")

    l = loop_inductance.to(ureg.henry)
    di_dt = current_slew_rate.to(ureg.ampere / ureg.second)
    dv = l * di_dt
    return dv.to(ureg.volt)


def calc_alpha_power_gate_delay_stretch(
    nominal_voltage,
    drooped_voltage,
    threshold_voltage=None,
    alpha: float = 1.3,
):
    """
    Calculate CMOS logic gate propagation delay expansion under power-rail voltage droop using the alpha-power law.

    Sakurai-Newton Alpha-Power Law:
        stretch_ratio = (V_droop / V_nom) * ((V_nom - V_th) / (V_droop - V_th))^alpha
    """
    validate_positive(nominal_voltage, "nominal_voltage")
    validate_positive(drooped_voltage, "drooped_voltage")

    v_th = (0.42 * ureg.volt) if threshold_voltage is None else threshold_voltage
    v_nom = nominal_voltage.to(ureg.volt).magnitude
    v_drop = drooped_voltage.to(ureg.volt).magnitude
    v_t = v_th.to(ureg.volt).magnitude

    if v_drop <= v_t:
        raise ValueError(f"Drooped voltage {v_drop} V is below or at threshold voltage {v_t} V (circuit stalls).")

    num = (v_drop / (v_drop - v_t)**alpha)
    den = (v_nom / (v_nom - v_t)**alpha)
    ratio = num / den
    expansion_pct = (ratio - 1.0) * 100.0

    return {
        "stretch_ratio": float(ratio),
        "delay_expansion_percent": float(expansion_pct),
    }


def calc_clopper_pearson_zero_failure_bound(n_trials: int, confidence: float = 0.95) -> float:
    """
    Calculate exact Clopper-Pearson binomial lower confidence bound for zero-failure trials.

    Equation:
        p_L = (1 - C)^(1 / n)
    """
    validate_at_least(n_trials, 1, "n_trials")
    alpha = 1.0 - confidence
    return float(alpha ** (1.0 / n_trials))


def calc_zero_failure_sample_size(target_lower_bound: float, confidence: float = 0.95) -> int:
    """
    Calculate required number of consecutive zero-failure trials to certify a target reliability lower bound.

    Equation:
        n = ceil(ln(1 - C) / ln(p_L))
    """
    if not (0.0 < target_lower_bound < 1.0):
        raise ValueError("target_lower_bound must be strictly between 0 and 1.")
    alpha = 1.0 - confidence
    n = math.ceil(math.log(alpha) / math.log(target_lower_bound))
    return int(n)


def calc_demonstration_collection_yield(
    target_clean_demos: int,
    failure_rate: float = 0.25,
    qa_rejection_rate: float = 0.10,
    task_duration=None,
    reset_duration=None,
    daily_shift_hours=None,
    operator_efficiency: float = 0.60,
):
    """
    Calculate physical demonstration collection yield, total attempts, and required labor hours.

    Equation:
        eta = (1 - failure_rate) * (1 - qa_rejection_rate)
        n_attempts = ceil(target_clean_demos / eta)
        t_total_attempt_sec = n_attempts * (task_duration + reset_duration)
        operator_hours = t_total_attempt_sec / (3600 * operator_efficiency)
    """
    validate_at_least(target_clean_demos, 1, "target_clean_demos")
    t_task = (45.0 * ureg.second) if task_duration is None else task_duration
    t_reset = (15.0 * ureg.second) if reset_duration is None else reset_duration
    shift_h = (8.0 * ureg.hour) if daily_shift_hours is None else daily_shift_hours

    eta = (1.0 - failure_rate) * (1.0 - qa_rejection_rate)
    n_attempts = math.ceil(target_clean_demos / eta)

    cycle_sec = (t_task + t_reset).to(ureg.second).magnitude
    total_sec = n_attempts * cycle_sec

    op_hours = total_sec / (3600.0 * operator_efficiency)
    shift_len = shift_h.to(ureg.hour).magnitude
    op_days = op_hours / shift_len

    return {
        "effective_yield": float(eta),
        "total_attempts": int(n_attempts),
        "operator_hours": float(op_hours),
        "operator_days": float(op_days),
    }


def calc_harmonic_drive_fatigue_consumption(
    total_active_duration,
    mean_joint_velocity,
    rated_l10_life_revs: float = 50_000_000.0,
):
    """
    Calculate cumulative shaft revolutions and L10 fatigue consumption for strain wave / harmonic gearboxes.

    Equation:
        revolutions = (omega_mean * T_active) / (2 * pi)
        fatigue_fraction = revolutions / L10_rated
    """
    validate_positive(total_active_duration, "total_active_duration")
    validate_positive(mean_joint_velocity, "mean_joint_velocity")

    t_sec = total_active_duration.to(ureg.second).magnitude
    omega = mean_joint_velocity.to(ureg.radian / ureg.second).magnitude

    revs = (omega * t_sec) / (2.0 * math.pi)
    fraction = revs / rated_l10_life_revs

    return {
        "cumulative_revolutions": float(revs),
        "fatigue_fraction": float(fraction),
        "fatigue_percent": float(fraction * 100.0),
    }


def calc_transient_impact_force(contact_velocity, effective_mass, contact_stiffness):
    """
    Calculate peak unbraked transient impact force on an elastic body.

    Equation:
        F_peak = v * sqrt(k * m_eff)
    """
    validate_positive(contact_velocity, "contact_velocity")
    validate_positive(effective_mass, "effective_mass")
    validate_positive(contact_stiffness, "contact_stiffness")

    v = contact_velocity.to(ureg.meter / ureg.second).magnitude
    m = effective_mass.to(ureg.kg).magnitude
    k = contact_stiffness.to(ureg.newton / ureg.meter).magnitude

    f = v * math.sqrt(k * m)
    return (f * ureg.newton).to(ureg.newton)


def calc_tripwire_contact_force_accumulation(
    tripwire_force,
    contact_stiffness,
    penetration_velocity,
    loop_latency,
):
    """
    Calculate total contact force reached before motor deceleration is commanded following a tripwire breach.

    Equation:
        F_lat = F_trip + k * v * t_lat
    """
    validate_positive(tripwire_force, "tripwire_force")
    validate_positive(contact_stiffness, "contact_stiffness")
    validate_positive(penetration_velocity, "penetration_velocity")
    validate_positive(loop_latency, "loop_latency")

    f_trip = tripwire_force.to(ureg.newton)
    k = contact_stiffness.to(ureg.newton / ureg.meter)
    v = penetration_velocity.to(ureg.meter / ureg.second)
    t = loop_latency.to(ureg.second)

    f_lat = f_trip + (k * v * t).to(ureg.newton)
    return f_lat.to(ureg.newton)


def calc_crypto_auth_deadline(
    clearance_distance,
    initial_velocity,
    emergency_deceleration,
    mechanical_brake_lag=None,
    bus_arbitration_delay=None,
):
    """
    Calculate maximum allowable cryptographic verification latency before kinematic stopping envelope breaches clearance.

    Equation:
        d_brake = v0^2 / (2 * a)
        T_auth_max = (d_clear - d_brake) / v0 - t_mech - t_bus
    """
    validate_positive(clearance_distance, "clearance_distance")
    validate_positive(initial_velocity, "initial_velocity")
    validate_positive(emergency_deceleration, "emergency_deceleration")

    t_mech = (0.0 * ureg.second) if mechanical_brake_lag is None else mechanical_brake_lag.to(ureg.second)
    t_bus = (0.0 * ureg.second) if bus_arbitration_delay is None else bus_arbitration_delay.to(ureg.second)

    v0 = initial_velocity.to(ureg.meter / ureg.second)
    a = emergency_deceleration.to(ureg.meter / (ureg.second**2))
    d_clear = clearance_distance.to(ureg.meter)

    d_brake = ((v0**2) / (2.0 * a)).to(ureg.meter)
    if d_clear < d_brake:
        raise ValueError(f"Clearance {d_clear} is insufficient even with zero auth latency (braking requires {d_brake}).")

    t_avail = (d_clear - d_brake) / v0
    t_deadline = t_avail - t_mech - t_bus
    return t_deadline.to(ureg.millisecond)


def calc_coulomb_stiction_deadband(commanded_torque, coulomb_friction_torque, stator_electrical_lag):
    """
    Calculate deadtime duration before shaft acceleration occurs under stator electrical lag and Coulomb stiction.

    Equation:
        t_dead = tau_lag * ln(tau_cmd / (tau_cmd - tau_c))
    """
    validate_positive(commanded_torque, "commanded_torque")
    validate_positive(coulomb_friction_torque, "coulomb_friction_torque")
    validate_positive(stator_electrical_lag, "stator_electrical_lag")

    cmd = commanded_torque.to(ureg.newton * ureg.meter).magnitude
    coulomb = coulomb_friction_torque.to(ureg.newton * ureg.meter).magnitude
    lag = stator_electrical_lag.to(ureg.millisecond)

    if cmd <= coulomb:
        return float("inf") * ureg.millisecond

    ratio = cmd / (cmd - coulomb)
    return (lag * math.log(ratio)).to(ureg.millisecond)


def calc_shielded_system_hazard_rate(brain_hazard_rate, shield_coverage: float, shield_hw_failure_rate=None):
    """
    Calculate end-to-end hazardous event rate for a physical AI system protected by a deterministic runtime safety shield.

    Equation:
        p_sys = p_brain * (1 - c_shield) + p_hw
    """
    validate_positive(brain_hazard_rate, "brain_hazard_rate")
    if not (0.0 <= shield_coverage <= 1.0):
        raise ValueError("shield_coverage must be between 0 and 1.")

    p_brain = brain_hazard_rate.to(1.0 / ureg.hour)
    p_hw = (0.0 / ureg.hour) if shield_hw_failure_rate is None else shield_hw_failure_rate.to(1.0 / ureg.hour)

    p_sys = p_brain * (1.0 - shield_coverage) + p_hw
    return p_sys.to(1.0 / ureg.hour)


def calc_cbf_qp_orthogonal_projection(u_nom, a_vec, b_scalar):
    """
    Calculate closed-form minimum-intervention Control Barrier Function (CBF) orthogonal projection.

    Solves the quadratic program:
        min  1/2 ||u - u_nom||^2
        s.t. a^T u <= b
    """
    u_n = np.asarray(u_nom, dtype=float)
    a = np.asarray(a_vec, dtype=float)
    b = float(b_scalar)

    breach = float(np.dot(a, u_n) - b)
    if breach <= 0.0:
        return {
            "u_projected": u_n.tolist(),
            "correction_norm": 0.0,
            "lambda_star": 0.0,
            "was_modified": False,
        }

    norm_sq = float(np.dot(a, a))
    lam = breach / norm_sq
    u_star = u_n - lam * a
    correction = float(np.linalg.norm(u_star - u_n))

    return {
        "u_projected": u_star.tolist(),
        "correction_norm": correction,
        "lambda_star": lam,
        "was_modified": True,
    }


def calc_quintic_blend_duration(delta_torque, max_allowable_jerk):
    """
    Compute minimum blend duration for quintic polynomial trajectory to bound torque jerk.

    Equation:
        tau_blend >= (1.875 * |Delta_tau|) / jerk_max
    """
    validate_positive(max_allowable_jerk, "max_allowable_jerk")
    d_tau = abs(delta_torque.to(ureg.newton * ureg.meter).magnitude) * (ureg.newton * ureg.meter)
    j_max = max_allowable_jerk.to(ureg.newton * ureg.meter / ureg.second)
    tau = (1.875 * d_tau) / j_max
    return tau.to(ureg.millisecond)


def calc_thermal_cooling_recovery_time(tau_thermal, t_start, t_clear, t_target):
    """
    Calculate thermal cooling duration back to safe recovery threshold under Newton cooling.

    Equation:
        t = tau_th * ln((T_start - T_target) / (T_clear - T_target))
    """
    validate_positive(tau_thermal, "tau_thermal")

    t_s = float(t_start.magnitude) if hasattr(t_start, "magnitude") else float(t_start)
    t_c = float(t_clear.magnitude) if hasattr(t_clear, "magnitude") else float(t_clear)
    t_tgt = float(t_target.magnitude) if hasattr(t_target, "magnitude") else float(t_target)

    num = t_s - t_tgt
    den = t_c - t_tgt
    if den <= 0 or num <= 0:
        raise ValueError("Invalid temperature gradient: target must be below start and clear thresholds.")

    t_recov = (tau_thermal * math.log(num / den)).to(ureg.second)
    return t_recov


def calc_passive_compliance_stiffness(allowable_force, preload_force, motor_stall_displacement):
    """
    Size series compliance spring stiffness to prevent brittle fracture within motor stall displacement.

    Equation:
        k_spring <= (F_allow - F_preload) / Delta_x_stall
    """
    validate_positive(allowable_force, "allowable_force")
    validate_nonnegative(preload_force, "preload_force")
    validate_positive(motor_stall_displacement, "motor_stall_displacement")

    delta_f = (allowable_force - preload_force).to(ureg.newton)
    if delta_f.magnitude <= 0:
        raise ValueError("Allowable force must exceed preload force.")

    dx = motor_stall_displacement.to(ureg.meter)
    k = delta_f / dx
    return k.to(ureg.newton / ureg.meter)


def calc_process_containment_time_to_breach(
    capacity_limit,
    current_volume,
    inflow_rate,
    outflow_rate=None,
):
    """
    Calculate time until liquid container or buffer capacity is breached under flow surge.

    Equation:
        t_breach = (V_max - V_0) / (Q_in - Q_out)
    """
    validate_positive(capacity_limit, "capacity_limit")
    validate_nonnegative(current_volume, "current_volume")
    validate_positive(inflow_rate, "inflow_rate")

    q_out = (0.0 * (ureg.liter / ureg.second)) if outflow_rate is None else outflow_rate.to(ureg.liter / ureg.second)
    v_max = capacity_limit.to(ureg.liter)
    v_0 = current_volume.to(ureg.liter)
    q_in = inflow_rate.to(ureg.liter / ureg.second)

    q_net = q_in - q_out
    if q_net.magnitude <= 0:
        return float("inf") * ureg.second

    headroom = v_max - v_0
    if headroom.magnitude <= 0:
        return 0.0 * ureg.second

    t_breach = (headroom / q_net).to(ureg.second)
    return t_breach


def calc_teleop_ingestion_budget(
    num_cameras: int,
    rgb_width: int,
    rgb_height: int,
    rgb_fps: float,
    rgb_bytes_per_pixel: int = 3,
    depth_width: int = 0,
    depth_height: int = 0,
    depth_fps: float = 0.0,
    depth_bytes_per_pixel: int = 2,
    kinematics_bytes_per_sec=None,
    tactile_bytes_per_sec=None,
):
    """
    Calculate sustained data ingestion bandwidth and hourly/shift storage accumulation
    for multi-camera robotic teleoperation rigs.

    Equation:
        BW_rgb = num_cameras * (W_rgb * H_rgb * Bpp_rgb) * FPS_rgb
        BW_depth = num_cameras * (W_depth * H_depth * Bpp_depth) * FPS_depth
        BW_kinematics = bytes_per_sec_kinematics
        BW_tactile = bytes_per_sec_tactile
        BW_total = BW_rgb + BW_depth + BW_kinematics + BW_tactile
        Storage_1hr = BW_total * 3600 s

    Parameters
    ----------
    num_cameras : int
        Number of active camera streams (e.g., 4).
    rgb_width, rgb_height : int
        Spatial resolution of RGB streams (e.g., 1920, 1080).
    rgb_fps : float
        Frame rate of RGB streams in Hz (e.g., 30.0).
    rgb_bytes_per_pixel : int
        Color encoding byte depth (default 3 bytes for 24-bit RGB).
    depth_width, depth_height : int
        Spatial resolution of depth streams (e.g., 1280, 720).
    depth_fps : float
        Frame rate of depth streams in Hz (e.g., 30.0).
    depth_bytes_per_pixel : int
        Depth encoding byte depth (default 2 bytes for uint16).
    kinematics_bytes_per_sec : Quantity, optional
        Data rate for joint positions, velocities, torques (e.g., 128 B * 1000 Hz).
    tactile_bytes_per_sec : Quantity, optional
        Data rate for tactile array telemetry (e.g., 2048 B * 100 Hz).

    Returns
    -------
    dict
        Breakdown of RGB, depth, vision total, kinematics, tactile, total bandwidth,
        and hourly uncompressed storage consumption.
    """
    validate_positive(num_cameras, "num_cameras")
    validate_positive(rgb_width, "rgb_width")
    validate_positive(rgb_height, "rgb_height")
    validate_positive(rgb_fps, "rgb_fps")

    bw_rgb_raw = num_cameras * (rgb_width * rgb_height * rgb_bytes_per_pixel) * rgb_fps
    bw_rgb = (bw_rgb_raw * (ureg.byte / ureg.second)).to(ureg.megabyte / ureg.second)

    if depth_width > 0 and depth_height > 0 and depth_fps > 0:
        bw_depth_raw = num_cameras * (depth_width * depth_height * depth_bytes_per_pixel) * depth_fps
        bw_depth = (bw_depth_raw * (ureg.byte / ureg.second)).to(ureg.megabyte / ureg.second)
    else:
        bw_depth = 0.0 * (ureg.megabyte / ureg.second)

    bw_vision = bw_rgb + bw_depth

    bw_kin = (0.0 * (ureg.megabyte / ureg.second)) if kinematics_bytes_per_sec is None else kinematics_bytes_per_sec.to(ureg.megabyte / ureg.second)
    bw_tac = (0.0 * (ureg.megabyte / ureg.second)) if tactile_bytes_per_sec is None else tactile_bytes_per_sec.to(ureg.megabyte / ureg.second)

    bw_total = bw_vision + bw_kin + bw_tac
    storage_1hr = (bw_total * (3600.0 * ureg.second)).to(ureg.terabyte)

    return {
        "bw_rgb": bw_rgb,
        "bw_depth": bw_depth,
        "bw_vision": bw_vision,
        "bw_kinematics": bw_kin,
        "bw_tactile": bw_tac,
        "bw_total": bw_total,
        "storage_per_hour": storage_1hr,
    }


def calc_tsdf_voxel_grid_budget(
    workspace_volume,
    voxel_size,
    bytes_per_voxel: int = 4,
    sparsity_ratio: float = 0.03,
    block_size_voxels: int = 512,
    block_header_bytes: int = 32,
    ray_rate=None,
    dense_voxels_per_ray: int = 300,
    sparse_voxels_per_ray: int = 6,
    bytes_per_ray_voxel: int = 8,
    cache_hit_rate: float = 0.95,
):
    """
    Calculate memory footprint and DRAM bus bandwidth demand for dense vs sparse TSDF voxel grids.

    Equation:
        N_dense = V_workspace / (Delta_v)^3
        M_dense = N_dense * B_voxel
        N_active = N_dense * S_sparsity
        N_blocks = ceil(N_active / B_block)
        M_sparse = N_blocks * (B_block * B_voxel + B_header)
        BW_dense_dram = R_rays * N_dense_lookups * B_lookup
        BW_sparse_dram = R_rays * N_sparse_lookups * B_lookup * (1 - Hit_cache)

    Parameters
    ----------
    workspace_volume : Quantity
        Total bounding volume of operational workcell (e.g. 300 m^3).
    voxel_size : Quantity
        Isotropic voxel side length (e.g. 1.0 cm or 5.0 mm).
    bytes_per_voxel : int
        Bytes per voxel state (default 4 bytes: 16-bit distance + 16-bit weight).
    sparsity_ratio : float
        Surface occupancy fraction within bounding volume (default 0.03 = 3%).
    block_size_voxels : int
        Voxel count per hierarchical leaf block (default 512 for 8x8x8).
    block_header_bytes : int
        Memory overhead per block metadata/pointer header (default 32 B).
    ray_rate : Quantity, optional
        Raycast integration query rate (e.g., 640x480x60 = 18.432e6 rays/s).
    dense_voxels_per_ray : int
        Raycast integration depth across dense volume (default 300 lookups).
    sparse_voxels_per_ray : int
        Lookups confined to narrow TSDF truncation margin (default 6 lookups).
    bytes_per_ray_voxel : int
        Read-modify-write memory traffic per ray step (default 8 B).
    cache_hit_rate : float
        SRAM/L2 cache hit rate for localized sparse blocks (default 0.95).

    Returns
    -------
    dict
        Voxel count, dense footprint, sparse blocks, sparse footprint,
        and DRAM bus bandwidths.
    """
    validate_positive(workspace_volume, "workspace_volume")
    validate_positive(voxel_size, "voxel_size")

    v_vol = workspace_volume.to(ureg.meter**3).magnitude
    v_vox = (voxel_size.to(ureg.meter).magnitude) ** 3
    n_dense = int(round(v_vol / v_vox))

    m_dense_bytes = n_dense * bytes_per_voxel * ureg.byte
    m_dense = m_dense_bytes.to(ureg.gigabyte)

    n_active = int(round(n_dense * sparsity_ratio))
    n_blocks = math.ceil(n_active / block_size_voxels)
    bytes_per_block = block_size_voxels * bytes_per_voxel + block_header_bytes
    m_sparse_bytes = n_blocks * bytes_per_block * ureg.byte
    m_sparse = m_sparse_bytes.to(ureg.megabyte)

    result = {
        "dense_voxel_count": n_dense,
        "dense_memory": m_dense,
        "active_voxel_count": n_active,
        "sparse_block_count": n_blocks,
        "sparse_memory": m_sparse,
    }

    if ray_rate is not None:
        r_rate = ray_rate.to(1 / ureg.second).magnitude if hasattr(ray_rate, "magnitude") else float(ray_rate)
        dense_dram_raw = r_rate * dense_voxels_per_ray * bytes_per_ray_voxel
        sparse_dram_raw = r_rate * sparse_voxels_per_ray * bytes_per_ray_voxel * (1.0 - cache_hit_rate)

        result["dense_dram_bandwidth"] = (dense_dram_raw * (ureg.byte / ureg.second)).to(ureg.gigabyte / ureg.second)
        result["sparse_dram_bandwidth"] = (sparse_dram_raw * (ureg.byte / ureg.second)).to(ureg.megabyte / ureg.second)

    return result


def calc_intent_drift_lease(
    tolerance_radius,
    sensor_noise,
    drift_velocity,
):
    """
    Calculate maximum permissible intent lease duration before drift breaches safety tolerance.

    Equation:
        tau = (r_tol - sigma_sensor) / v_drift

    Parameters
    ----------
    tolerance_radius : Quantity
        Defended task spatial tolerance radius (e.g. mm or m).
    sensor_noise : Quantity
        Uncertainty / sensor noise floor (e.g. mm or m). Must be < tolerance_radius.
    drift_velocity : Quantity
        Dynamic obstacle or unguided drift velocity (e.g. m/s).

    Returns
    -------
    Quantity
        Permissible lease duration in milliseconds.
    """
    validate_positive(tolerance_radius, "tolerance_radius")
    validate_positive(sensor_noise, "sensor_noise")
    validate_positive(drift_velocity, "drift_velocity")

    r_tol = tolerance_radius.to(ureg.meter)
    sig = sensor_noise.to(ureg.meter)
    if sig >= r_tol:
        raise ValueError("sensor_noise must be strictly less than tolerance_radius.")

    v_drift = drift_velocity.to(ureg.meter / ureg.second)
    tau = (r_tol - sig) / v_drift
    return tau.to(ureg.millisecond)


def calc_process_thermal_runaway_lease(
    heat_generation_rate,
    thermal_capacitance,
    max_temp_overshoot,
):
    """
    Calculate thermal rise rate and allowable control lease duration before thermal runaway.

    Equation:
        dT_dt = P_in / C_th
        tau = Delta_T_max / dT_dt

    Parameters
    ----------
    heat_generation_rate : Quantity
        Internal Joule/dissipative heat input rate P_in (e.g. W).
    thermal_capacitance : Quantity
        Lumped system thermal capacitance C_th (e.g. J/K).
    max_temp_overshoot : Quantity
        Permissible temperature rise Delta_T_max (e.g. K).

    Returns
    -------
    dict
        rate_of_rise, tau_lease.
    """
    validate_positive(heat_generation_rate, "heat_generation_rate")
    validate_positive(thermal_capacitance, "thermal_capacitance")
    validate_positive(max_temp_overshoot, "max_temp_overshoot")

    p = heat_generation_rate.to(ureg.watt)
    c_th = thermal_capacitance.to(ureg.joule / ureg.kelvin)
    delta_t = max_temp_overshoot.to(ureg.kelvin)

    dt_dt = p / c_th
    tau = delta_t / dt_dt

    return {
        "rate_of_rise": dt_dt.to(ureg.kelvin / ureg.second),
        "tau_lease": tau.to(ureg.millisecond),
    }
