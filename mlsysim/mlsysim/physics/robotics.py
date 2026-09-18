"""Physical AI and embodied ML systems physics and safety accounting formulas (Volume IV).

Domain scope:
- Sensor-to-actuator end-to-end latency budgets
- Dynamic stopping distance and kinetic inertia safety bounds
- Actuator stator thermal dissipation and power envelopes
- Action chunking rate adaptation and control loop timing
- Reflected rotor inertia and gear ratios
"""

from __future__ import annotations

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

