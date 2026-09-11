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
