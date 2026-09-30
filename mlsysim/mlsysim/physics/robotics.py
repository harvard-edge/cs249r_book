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


def calc_c2_stop_suffix(entry_velocity, peak_deceleration):
    """
    Calculate the duration and distance of a C2-continuous stop from cruise.

    A stop that begins and ends at zero acceleration, entered from constant
    velocity v_c, follows the minimum-order polynomial

        q(u) = q_c + v_c * T * (u - u^3 + u^4 / 2),  u = t / T in [0, 1],

    whose peak deceleration is 1.5 * v_c / T and whose distance is v_c * T / 2.
    Setting the peak equal to the credible deceleration a gives

        T = 1.5 * v / a,    d = 0.75 * v^2 / a,

    which is 1.5 times the constant-deceleration distance v^2 / (2 * a).
    The factor belongs to this polynomial, not to smooth stopping in general:
    its jerk q''' = -(6 v_c / T^2)(1 - 2u) peaks in magnitude at the ends,

        j_peak = 6 * v / T^2 = (8 / 3) * a^2 / v,

    and a jerk-limited S-curve with the same a and j_peak stops shorter
    (calc_scurve_stop_distance).

    Source: derived here. The polynomial satisfies q' = v_c, q'' = 0 at u = 0
    and q' = 0, q'' = 0 at u = 1; peak |q''| occurs at u = 1/2 (minimum of
    q''(u) = v_c / T * (-6u + 6u^2)); peak |q'''| occurs at u = 0 and u = 1.

    Parameters
    ----------
    entry_velocity : Quantity
        Cruise velocity at the start of the stop (e.g. m/s).
    peak_deceleration : Quantity
        Credible peak deceleration the stop may reach (e.g. m/s^2).

    Returns
    -------
    dict
        duration (ms), distance (m), and peak_jerk (m/s^3).
    """
    validate_positive(entry_velocity, "entry_velocity")
    validate_positive(peak_deceleration, "peak_deceleration")

    v = entry_velocity.to(ureg.meter / ureg.second)
    a = peak_deceleration.to(ureg.meter / ureg.second**2)
    duration = 1.5 * v / a
    return {
        "duration": duration.to(ureg.millisecond),
        "distance": (0.75 * v**2 / a).to(ureg.meter),
        "peak_jerk": (6.0 * v / duration**2).to(ureg.meter / ureg.second**3),
    }


def _scurve_stop_si(v, a, j):
    """Duration (s) and distance (m) of a jerk-limited stop, SI magnitudes."""
    if v <= 0.0:
        return 0.0, 0.0
    if v >= a * a / j:
        # Trapezoidal acceleration: ramp to a in a/j, hold, ramp back to zero.
        t = v / a + a / j
    else:
        # Triangular acceleration: the peak sqrt(v j) stays below a.
        t = 2.0 * math.sqrt(v / j)
    # The velocity profile is point-symmetric about its midpoint, so d = v T / 2.
    return t, 0.5 * v * t


def _scurve_si_inputs(entry_velocity, max_deceleration, max_jerk):
    validate_nonnegative(entry_velocity, "entry_velocity")
    validate_positive(max_deceleration, "max_deceleration")
    validate_positive(max_jerk, "max_jerk")
    return (
        entry_velocity.to(ureg.meter / ureg.second).magnitude,
        max_deceleration.to(ureg.meter / ureg.second**2).magnitude,
        max_jerk.to(ureg.meter / ureg.second**3).magnitude,
    )


def calc_scurve_stop_distance(entry_velocity, max_deceleration, max_jerk):
    """
    Distance of a jerk-limited S-curve stop entered from zero-acceleration cruise.

    The stop drives acceleration from zero to -a at the jerk limit j, holds
    -a, and returns acceleration to zero at the jerk limit as velocity reaches
    zero (trapezoidal acceleration, the "double S" velocity profile). The
    profile is C2: position, velocity, and acceleration are continuous and
    jerk is piecewise constant.

        if v >= a^2 / j:   d = (v / 2) * (v / a + a / j)
                             = v^2 / (2 a) + v * a / (2 j)
        else:              d = v * sqrt(v / j)     (a is never reached)

    In the first case the stop equals constant-deceleration braking at a plus
    cruise over half of one jerk ramp, a / (2 j). Under the bounds |a| <= a_max
    and |j| <= j_max this is the time-optimal stop, and for a stop that does not
    reverse it is also the shortest.

    Source: Biagiotti, L. and Melchiorri, C., Trajectory Planning for Automatic
    Machines and Robots, Springer (2008; DOI 10.1007/978-3-540-85629-0),
    ch. 3, trajectory with double S velocity profile, specialized here to a
    stop from v to rest with zero boundary accelerations. Formula re-derived and
    checked against a discretized minimum-distance linear program.

    Parameters
    ----------
    entry_velocity : Quantity
        Cruise velocity at stop onset (e.g. m/s); zero gives zero distance.
    max_deceleration : Quantity
        Deceleration bound a_max (e.g. m/s^2).
    max_jerk : Quantity
        Jerk bound j_max (e.g. m/s^3).

    Returns
    -------
    Quantity
        Stopping distance in meters.
    """
    v, a, j = _scurve_si_inputs(entry_velocity, max_deceleration, max_jerk)
    return (_scurve_stop_si(v, a, j)[1] * ureg.meter).to(ureg.meter)


def calc_scurve_stop_time(entry_velocity, max_deceleration, max_jerk):
    """
    Duration of the jerk-limited S-curve stop of calc_scurve_stop_distance.

        if v >= a^2 / j:   T = v / a + a / j
        else:              T = 2 * sqrt(v / j)

    Source: Biagiotti and Melchiorri (2008), ch. 3, double S velocity profile;
    see calc_scurve_stop_distance.

    Returns
    -------
    Quantity
        Stopping time in milliseconds.
    """
    v, a, j = _scurve_si_inputs(entry_velocity, max_deceleration, max_jerk)
    return (_scurve_stop_si(v, a, j)[0] * ureg.second).to(ureg.millisecond)


def calc_scurve_max_permitted_velocity(
    d_clear,
    total_latency,
    max_deceleration,
    max_jerk,
    loc_margin=None,
    physical_margin=None,
    approach_velocity=None,
):
    """
    Highest cruise speed whose delayed S-curve stop fits the clearance.

    Solves for the largest v with

        v * tau + d_S(v) + v_h * (tau + T_S(v)) <= D_clear - delta_loc - delta_margin,

    where d_S and T_S are the S-curve stop distance and time
    (calc_scurve_stop_distance, calc_scurve_stop_time) and v_h is an optional
    obstacle approach speed (zero for a static obstacle). The left side is
    continuous and strictly increasing in v, so bisection finds the root.
    Because d_S is not a single quadratic in v, no one closed form covers both
    stop regimes; for v_h = 0 in the trapezoidal regime the result equals
    calc_max_permitted_velocity with delay tau + a / (2 j).

    Source: derived here from calc_scurve_stop_distance (Biagiotti and
    Melchiorri 2008, ch. 3) and the additive delay-plus-braking budget of
    calc_safe_stopping_distance.

    Parameters
    ----------
    d_clear : Quantity
        Clear distance to the obstacle (e.g. m).
    total_latency : Quantity
        Pre-brake delay tau (e.g. ms).
    max_deceleration, max_jerk : Quantity
        Stop limits a_max (m/s^2) and j_max (m/s^3).
    loc_margin, physical_margin : Quantity, optional
        Localization bound and protective clearance subtracted from d_clear.
    approach_velocity : Quantity, optional
        Speed at which the obstacle closes during delay and stop (e.g. m/s).

    Returns
    -------
    Quantity
        Speed ceiling in m/s (zero if the margins consume the clearance).
    """
    validate_nonnegative(d_clear, "d_clear")
    validate_nonnegative(total_latency, "total_latency")
    a = max_deceleration.to(ureg.meter / ureg.second**2).magnitude
    j = max_jerk.to(ureg.meter / ureg.second**3).magnitude
    validate_positive(max_deceleration, "max_deceleration")
    validate_positive(max_jerk, "max_jerk")

    d_eff = d_clear.to(ureg.meter).magnitude
    for margin, name in ((loc_margin, "loc_margin"), (physical_margin, "physical_margin")):
        if margin is not None:
            validate_nonnegative(margin, name)
            d_eff -= margin.to(ureg.meter).magnitude
    tau = total_latency.to(ureg.second).magnitude
    v_h = 0.0
    if approach_velocity is not None:
        validate_nonnegative(approach_velocity, "approach_velocity")
        v_h = approach_velocity.to(ureg.meter / ureg.second).magnitude

    def excess(v):
        t_s, d_s = _scurve_stop_si(v, a, j)
        return v * tau + d_s + v_h * (tau + t_s) - d_eff

    zero = 0.0 * (ureg.meter / ureg.second)
    if excess(0.0) >= 0.0:
        return zero
    # d_S(v) >= v^2 / (2 a), so excess(sqrt(2 a d_eff)) >= 0: a valid upper bracket.
    lo, hi = 0.0, math.sqrt(2.0 * a * d_eff)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-15 * max(1.0, hi):
            break
    return (lo * (ureg.meter / ureg.second)).to(ureg.meter / ureg.second)


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
        opacity 4B, color coefficients 24B = 56 bytes in this assumed packing;
        actual storage varies with harmonic degree, precision, and implementation).

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
    Calculate a weight-stream lower bound and its amortized cost per waypoint.

    Under memory-bandwidth bound autoregressive weight streaming:
        t_stream = M_weights / (BW_peak * eta_bus)
        f_single = 1 / t_stream
        tau_step = t_stream / H
        f_waypoint_equivalent = 1 / tau_step = H * f_single

    The waypoint-equivalent rate is an accounting ratio, not the rate at which
    the model observes the world or generates a fresh action chunk. Decoder
    work and other inference costs are excluded. ``f_effective`` is retained
    as a legacy alias for callers of this function.

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
        t_stream, f_single, tau_step, f_waypoint_equivalent,
        f_effective (legacy alias), b_sustained.
    """
    validate_at_least(chunk_horizon, 1, "chunk_horizon")
    validate_positive(model_weight_memory, "model_weight_memory")
    validate_positive(memory_bandwidth, "memory_bandwidth")

    b_sustained = memory_bandwidth * bus_efficiency
    t_stream = (model_weight_memory / b_sustained).to(ureg.second)
    f_single = (1.0 / t_stream).to(ureg.hertz)

    tau_step = (t_stream / chunk_horizon).to(ureg.second)
    f_waypoint_equivalent = (1.0 / tau_step).to(ureg.hertz)

    return {
        "t_stream": t_stream.to(ureg.millisecond),
        "f_single": f_single,
        "tau_step": tau_step.to(ureg.millisecond),
        "f_waypoint_equivalent": f_waypoint_equivalent,
        "f_effective": f_waypoint_equivalent,
        "b_sustained": b_sustained.to(ureg.GB / ureg.second),
    }


def calc_chunk_renewal_schedule(
    chunk_rate,
    chunk_step,
    chunk_horizon: int,
    inference_latency,
):
    """
    Timing of a chunk policy that renews on a whole number of setpoint steps.

    A chunk policy begins a fresh evaluation every renewal period T_r = 1/f_r
    and emits H setpoints spaced Delta t apart. With the renewal on the setpoint
    grid (T_r = n Delta t for a whole number n), every renewal instant falls on
    a setpoint boundary, so a new chunk replaces its predecessor at a defined
    setpoint index and temporal ensembling can blend chunks step by step:

        n          = T_r / Delta t                  (setpoints per renewal, integer)
        T_open     = T_r + t_infer                  (worst evidence age at replacement)
        h_min      = ceil(T_open / Delta t)         (setpoints that bridge one renewal)
        T_span     = H * Delta t                    (motion one chunk supplies)
        n_overlap  = floor(H / n)                   (chunks that cover one step)

    T_open is the worst sampling-phase wait plus inference, before transport
    and admission delays. n_overlap is the number of predictions a temporal
    ensemble blends at a step in steady state.

    Source: action chunking and temporal ensembling from Zhao et al., "Learning
    Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), RSS 2023,
    arXiv:2304.13705, which queries its policy every step (n = 1). The
    renewal-on-the-grid requirement, the worst-case age, and the overlap count
    are arithmetic of this package for n > 1 (Volume IV, @sec-brain-action-chunking).

    Parameters
    ----------
    chunk_rate : Quantity
        Renewal rate f_r of the chunk policy (Hz).
    chunk_step : Quantity
        Setpoint spacing Delta t within a chunk (s).
    chunk_horizon : int
        Setpoints per chunk H (>= 1).
    inference_latency : Quantity
        Time from capture to an available chunk t_infer (s), for example a P99 target.

    Returns
    -------
    dict
        renewal_period, setpoints_per_renewal (int), open_loop_span,
        min_supply_setpoints (int), chunk_span, overlapping_chunks (int),
        renewal_slack (T_r - t_infer).

    Raises
    ------
    ValueError
        If the renewal period is not a whole number of setpoint steps, if the
        inference does not finish within one renewal period, or if one chunk
        cannot bridge a renewal (H < h_min).
    """
    validate_positive(chunk_rate, "chunk_rate")
    validate_positive(chunk_step, "chunk_step")
    validate_at_least(chunk_horizon, 1, "chunk_horizon")
    validate_positive(inference_latency, "inference_latency")

    period = (1.0 / chunk_rate).to(ureg.millisecond)
    step = chunk_step.to(ureg.millisecond)
    ratio = (period / step).to("dimensionless").magnitude
    n_per_renewal = round(ratio)
    if n_per_renewal < 1 or abs(ratio - n_per_renewal) > 1e-9:
        raise ValueError(
            f"renewal period {period:~P} is not a whole number of {step:~P} setpoint steps "
            f"(ratio {ratio:.4g})"
        )
    t_infer = inference_latency.to(ureg.millisecond)
    if t_infer >= period:
        raise ValueError(
            f"inference latency {t_infer:~P} does not finish within the {period:~P} renewal period"
        )

    open_loop = (period + t_infer).to(ureg.millisecond)
    h_min = math.ceil((open_loop / step).to("dimensionless").magnitude - 1e-12)
    if chunk_horizon < h_min:
        raise ValueError(
            f"chunk horizon {chunk_horizon} is shorter than the {h_min} setpoints one renewal needs"
        )
    span = (chunk_horizon * step).to(ureg.millisecond)

    return {
        "renewal_period": period,
        "setpoints_per_renewal": int(n_per_renewal),
        "open_loop_span": open_loop,
        "min_supply_setpoints": int(h_min),
        "chunk_span": span,
        "overlapping_chunks": int(chunk_horizon // n_per_renewal),
        "renewal_slack": (period - t_infer).to(ureg.millisecond),
    }


def calc_watchdog_lease_bound(
    clearance_distance,
    initial_velocity,
    max_deceleration,
):
    """
    Calculate the ideal total delay budget before emergency braking begins.

    Under constant velocity v0 during uncommanded drift, followed by emergency braking
    at constant deceleration a_max:
        d_brake = (v0^2) / (2 * a_max)
        d_drift = d_clear - d_brake
        T_lease + T_detection + T_brake_onset <= d_drift / v0

    The returned ceiling includes every pre-brake delay. A chosen watchdog
    lease must be shorter after reserving detection, command delivery, and
    brake-onset time. ``t_lease_max`` is retained as a legacy alias for the
    ideal total ceiling, not a safe standalone lease setting.

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
        t_total_delay_max, t_lease_max (legacy alias), d_brake,
        d_drift_allowable.
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
    t_total_delay = (d_drift / v0).to(ureg.millisecond)

    return {
        "t_total_delay_max": t_total_delay,
        "t_lease_max": t_total_delay,
        "d_brake": d_brake.to(ureg.meter),
        "d_drift_allowable": d_drift.to(ureg.meter),
    }


def calc_watchdog_feed_window(period, completion_min, completion_max, phase_jitter):
    """
    Calculate the feed-interval range a windowed watchdog must accept from a healthy tick.

    If a controller feeds once after each successful tick, the watchdog measures
    the interval between feeds. With the completion offset from tick start in
    [c_min, c_max] and the tick-start phase jitter bounded by J_phase,

    Equation:
        spread = (c_max - c_min) + J_phase
        T_feed in [T_period - spread, T_period + spread]

    The early (T_lower) and late (T_upper) trip points are configured around this
    range; the late-trip response must still fit a separately derived fault
    budget.

    Source: interval between successive completions of a periodic task whose
    completion offsets are bounded (difference of two bounded offsets plus the
    release jitter); derived here.

    Returns
    -------
    dict
        spread, t_lower, t_upper (all in microseconds).
    """
    validate_positive(period, "period")
    validate_nonnegative(completion_min, "completion_min")
    validate_nonnegative(phase_jitter, "phase_jitter")
    if completion_max < completion_min:
        raise ValueError("completion_max must be >= completion_min.")

    spread = (completion_max - completion_min + phase_jitter).to(ureg.microsecond)
    t_lower = (period - spread).to(ureg.microsecond)
    if t_lower.magnitude <= 0:
        raise ValueError(f"Feed spread {spread} consumes the whole period {period}.")
    return {
        "spread": spread,
        "t_lower": t_lower,
        "t_upper": (period + spread).to(ureg.microsecond),
    }


def calc_phase_lag_total(
    crossover_frequency,
    sample_period,
    compute_delay=None,
    transport_delay=None,
    phase_allowance=None,
):
    """
    Total loop delay and the phase it removes at crossover (sample-and-hold,
    compute, and transport terms together).

    A zero-order hold behaves in the baseband like a pure delay of half a
    sample period. A controller that outputs its command some time after it
    samples adds that compute delay, and a command that reaches the actuator
    one bus cycle later adds the transport delay. A pure delay tau removes
    phase omega * tau at every frequency, so near crossover:

        tau_total   = T_s / 2 + tau_comp + tau_bus
        dphi_total  = 2 * pi * f_c * tau_total              (radians)

    Given a phase allowance dphi_allow, the largest total delay the loop can
    absorb, and the highest crossover a given total delay supports, are

        tau_allow   = dphi_allow / (2 * pi * f_c)
        f_c,max     = dphi_allow / (2 * pi * tau_total)

    Source: Franklin, G. F., Powell, J. D. and Workman, M. L., Digital Control
    of Dynamic Systems, 3rd ed., Addison-Wesley, 1998 (zero-order hold as a
    T/2 delay; phase of a pure delay). The half-sample approximation holds
    only well below the sampling frequency; it is a budget, not a stability
    proof.

    Parameters
    ----------
    crossover_frequency : Quantity
        Loop crossover frequency f_c (Hz).
    sample_period : Quantity
        Controller sample period T_s.
    compute_delay : Quantity, optional
        Sample-to-output compute delay tau_comp (default 0).
    transport_delay : Quantity, optional
        Command transport delay to the actuator tau_bus (default 0).
    phase_allowance : Quantity, optional
        Phase erosion the design allows at crossover (angle). When given, the
        allowed delay, the delay margin, and the highest supported crossover
        are also returned.

    Returns
    -------
    dict
        tau_zoh, tau_total (ms); phase_zoh, phase_compute, phase_transport,
        phase_total (degrees); and, with phase_allowance, tau_allowed,
        tau_margin (negative when the budget fails), f_crossover_max,
        within_allowance.
    """
    validate_positive(crossover_frequency, "crossover_frequency")
    validate_positive(sample_period, "sample_period")

    f_c = crossover_frequency.to(ureg.hertz)
    t_s = sample_period.to(ureg.millisecond)
    t_comp = (0.0 * ureg.millisecond) if compute_delay is None else compute_delay.to(ureg.millisecond)
    t_bus = (0.0 * ureg.millisecond) if transport_delay is None else transport_delay.to(ureg.millisecond)
    validate_nonnegative(t_comp, "compute_delay")
    validate_nonnegative(t_bus, "transport_delay")

    omega = 2.0 * math.pi * f_c.magnitude  # rad/s

    def _deg(tau):
        return (omega * tau.to(ureg.second).magnitude * ureg.radian).to(ureg.degree)

    tau_zoh = (t_s / 2.0).to(ureg.millisecond)
    tau_total = (tau_zoh + t_comp + t_bus).to(ureg.millisecond)

    result = {
        "tau_zoh": tau_zoh,
        "tau_total": tau_total,
        "phase_zoh": _deg(tau_zoh),
        "phase_compute": _deg(t_comp),
        "phase_transport": _deg(t_bus),
        "phase_total": _deg(tau_total),
    }

    if phase_allowance is not None:
        validate_positive(phase_allowance, "phase_allowance")
        allow_rad = phase_allowance.to(ureg.radian).magnitude
        tau_allowed = (allow_rad / omega * ureg.second).to(ureg.millisecond)
        f_max = (allow_rad / (2.0 * math.pi * tau_total.to(ureg.second).magnitude)) * ureg.hertz
        result.update(
            {
                "tau_allowed": tau_allowed,
                "tau_margin": (tau_allowed - tau_total).to(ureg.millisecond),
                "f_crossover_max": f_max,
                "within_allowance": bool(tau_total <= tau_allowed),
            }
        )

    return result


# Data-field sizes a CAN FD DLC can encode (ISO 11898-1:2015). A payload that
# is not one of these is padded up to the next size on the wire.
CANFD_DATA_FIELD_SIZES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)


def calc_canfd_frame_bits(
    payload_bytes: int = 8,
    extended_id: bool = False,
    worst_case_stuffing: bool = True,
):
    """
    Bit count of one CAN FD data frame, split by the bit rate each bit uses.

    Frame layout (ISO 11898-1:2015 FD base format; extended format adds the
    SRR bit and the 18-bit identifier extension):

        nominal rate, before the switch:  SOF 1, ID 11, RRS 1, IDE 1, FDF 1,
                                          res 1, BRS 1          (17 bits; 36 extended)
        data rate, after BRS:             ESI 1, DLC 4, data 8n,
                                          stuff count 4, CRC 17 (n <= 16) or 21,
                                          fixed stuff bits 6 (CRC-17) or 7 (CRC-21)
        nominal rate, after the switch:   CRC delimiter 1, ACK slot 1,
                                          ACK delimiter 1, EOF 7, intermission 3
                                                                (13 bits)

    Dynamic stuff bits (one after five equal bits) apply from SOF through the
    end of the data field. The worst case over a stuffed run of L bits is
    floor((L - 1) / 4); the nominal-rate share is floor((L_nom - 1) / 4) and
    the rest fall in the data phase. The CRC field uses the fixed stuff bits
    above instead. BRS and the CRC delimiter straddle the rate switch; both
    are counted whole at the nominal rate, which slightly overstates the time.

    Source: ISO 11898-1:2015, Road vehicles -- Controller area network (CAN)
    -- Part 1: Data link layer and physical signalling (FD frame format, stuff
    count, fixed stuff bits); Robert Bosch GmbH, CAN with Flexible Data-Rate,
    Specification Version 1.0, 2012 (dual bit rate, CRC-17/21, fixed CRC stuff
    bits).

    Parameters
    ----------
    payload_bytes : int
        Application bytes to carry (0 to 64). Padded up to the next data-field
        size a DLC can encode.
    extended_id : bool
        29-bit identifier (extended format) instead of 11-bit.
    worst_case_stuffing : bool
        Include the worst-case dynamic stuff bits (the bound a response-time
        analysis needs). False gives the unstuffed length.

    Returns
    -------
    dict
        data_field_bytes, crc_bits, fixed_stuff_bits, stuff_bits_nominal,
        stuff_bits_data, bits_nominal, bits_data (all int).
    """
    validate_at_least(payload_bytes, 0, "payload_bytes")
    if payload_bytes > 64:
        raise ValueError(f"CAN FD carries at most 64 data bytes per frame, got {payload_bytes}.")

    data_field = next(s for s in CANFD_DATA_FIELD_SIZES if s >= payload_bytes)
    crc_bits = 17 if data_field <= 16 else 21
    fixed_stuff = 6 if crc_bits == 17 else 7

    arb_bits = 36 if extended_id else 17
    ctrl_data_bits = 1 + 4 + 8 * data_field  # ESI, DLC, data field
    trailer_nominal = 1 + 1 + 1 + 7 + 3  # CRC delimiter, ACK, ACK delimiter, EOF, intermission

    if worst_case_stuffing:
        stuffed_run = arb_bits + ctrl_data_bits
        stuff_total = (stuffed_run - 1) // 4
        stuff_nominal = (arb_bits - 1) // 4
        stuff_data = stuff_total - stuff_nominal
    else:
        stuff_nominal = 0
        stuff_data = 0

    bits_nominal = arb_bits + trailer_nominal + stuff_nominal
    bits_data = ctrl_data_bits + 4 + crc_bits + fixed_stuff + stuff_data

    return {
        "data_field_bytes": data_field,
        "crc_bits": crc_bits,
        "fixed_stuff_bits": fixed_stuff,
        "stuff_bits_nominal": stuff_nominal,
        "stuff_bits_data": stuff_data,
        "bits_nominal": bits_nominal,
        "bits_data": bits_data,
    }


def calc_canfd_bus_utilization(
    num_axes: int,
    payload_bytes: int = 8,
    baud_arbitration=None,
    baud_data=None,
    cycle_time=None,
    frames_per_axis: int = 2,
    extended_id: bool = False,
    worst_case_stuffing: bool = True,
):
    """
    Serialization time and bus occupancy of a cyclic CAN FD axis exchange.

    Each axis exchanges ``frames_per_axis`` frames per cycle (default 2: one
    command frame from the controller and one feedback frame from the drive;
    CAN carries them separately, unlike a summation-frame bus such as
    EtherCAT). Frame bits come from ``calc_canfd_frame_bits``:

        t_frame   = bits_nominal / R_nom + bits_data / R_data
        n_frames  = num_axes * frames_per_axis
        t_total   = n_frames * t_frame
        rho       = t_total / cycle_time

    rho <= 1 is only a necessary condition: non-preemptive blocking, competing
    frames, and error retransmissions still need a response-time analysis.

    Source: frame format per ISO 11898-1:2015 and Bosch CAN FD Specification
    1.0 (2012), via ``calc_canfd_frame_bits``.

    Parameters
    ----------
    num_axes : int
        Axes served on this bus (>= 1). Model one bus per call; parallel buses
        (for example one per limb) are separate calls.
    payload_bytes : int, optional
        Application bytes per frame (default 8: a position-mode setpoint or
        feedback record). Padded to a valid CAN FD data-field size.
    baud_arbitration : Quantity, optional
        Nominal (arbitration-phase) bit rate (default 1.0 Mbps).
    baud_data : Quantity, optional
        Data-phase bit rate (default 5.0 Mbps).
    cycle_time : Quantity, optional
        Cycle the exchange must fit (default 1.0 ms).
    frames_per_axis : int, optional
        Frames per axis per cycle (default 2, command plus feedback).
    extended_id : bool, optional
        29-bit identifiers (default False).
    worst_case_stuffing : bool, optional
        Charge worst-case dynamic stuff bits (default True).

    Returns
    -------
    dict
        t_tx (per frame), t_total_tx, n_frames, utilization, is_schedulable,
        plus the per-frame bit breakdown of ``calc_canfd_frame_bits``.
    """
    validate_at_least(num_axes, 1, "num_axes")
    validate_at_least(frames_per_axis, 1, "frames_per_axis")

    baud_arb = (1.0 * (ureg.megabit / ureg.second)) if baud_arbitration is None else baud_arbitration
    baud_dat = (5.0 * (ureg.megabit / ureg.second)) if baud_data is None else baud_data
    t_cycle = (1.0 * ureg.millisecond) if cycle_time is None else cycle_time

    frame = calc_canfd_frame_bits(
        payload_bytes=payload_bytes,
        extended_id=extended_id,
        worst_case_stuffing=worst_case_stuffing,
    )

    t_nom = ((frame["bits_nominal"] * ureg.bit) / baud_arb).to(ureg.microsecond)
    t_dat = ((frame["bits_data"] * ureg.bit) / baud_dat).to(ureg.microsecond)
    t_tx = (t_nom + t_dat).to(ureg.microsecond)
    n_frames = num_axes * frames_per_axis
    t_total_tx = (n_frames * t_tx).to(ureg.microsecond)

    rho = (t_total_tx / t_cycle).to(ureg.dimensionless).magnitude

    return {
        "t_tx": t_tx,
        "t_nominal": t_nom,
        "t_data": t_dat,
        "t_total_tx": t_total_tx,
        "n_frames": n_frames,
        "utilization": float(rho),
        "is_schedulable": bool(rho <= 1.0),
        **frame,
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


def calc_reduced_mass(mass_a, mass_b):
    """
    Calculate the two-body reduced mass of a collision.

    Equation:
        mu = (1/m_a + 1/m_b)^-1

    When one body cannot recoil (m_b -> infinity, a clamped contact), mu -> m_a.

    Source: ISO/TS 15066:2016 Annex A, transient-contact model (content carried
    into ISO 10218-2:2025), as reproduced in Kirschner et al. 2022, eq. 2
    (arXiv:2203.02706).
    """
    validate_positive(mass_a, "mass_a")
    validate_positive(mass_b, "mass_b")
    m_a = mass_a.to(ureg.kg).magnitude
    m_b = mass_b.to(ureg.kg).magnitude
    return (1.0 / (1.0 / m_a + 1.0 / m_b)) * ureg.kg


def calc_transient_impact_force(contact_velocity, effective_mass, contact_stiffness, human_mass=None):
    """
    Calculate peak unbraked transient impact force on an elastic body.

    Equation:
        F_peak = v * sqrt(k * m)

    With ``human_mass=None`` the robot-side ``effective_mass`` acts alone
    (m = m_eff), the clamped (quasi-static) case in which the body part cannot
    recoil. With ``human_mass`` given, the free body part recoils and the
    two-body reduced mass acts: m = mu = (1/m_eff + 1/m_H)^-1.

    Source: ISO/TS 15066:2016 Annex A energy model F = v * sqrt(mu * k)
    (content carried into ISO 10218-2:2025), as reproduced in Kirschner et al.
    2022, eqs. 1-2 (arXiv:2203.02706).
    """
    validate_positive(contact_velocity, "contact_velocity")
    validate_positive(effective_mass, "effective_mass")
    validate_positive(contact_stiffness, "contact_stiffness")

    mass = effective_mass if human_mass is None else calc_reduced_mass(effective_mass, human_mass)
    v = contact_velocity.to(ureg.meter / ureg.second).magnitude
    m = mass.to(ureg.kg).magnitude
    k = contact_stiffness.to(ureg.newton / ureg.meter).magnitude

    f = v * math.sqrt(k * m)
    return (f * ureg.newton).to(ureg.newton)


def calc_transient_contact_speed_limit(force_limit, effective_mass, contact_stiffness, human_mass=None):
    """
    Calculate the highest contact speed whose transient peak force stays at a force limit.

    Equation:
        v_max = F_max / sqrt(k * m)

    The inverse of ``calc_transient_impact_force``: ``human_mass=None`` gives the
    clamped case (m = m_eff); a ``human_mass`` gives the free, recoiling case
    (m = mu, the two-body reduced mass).

    Source: ISO/TS 15066:2016 Annex A (carried into ISO 10218-2:2025), as used
    for v_rel,max in Kirschner et al. 2022 (arXiv:2203.02706).
    """
    validate_positive(force_limit, "force_limit")
    validate_positive(effective_mass, "effective_mass")
    validate_positive(contact_stiffness, "contact_stiffness")

    mass = effective_mass if human_mass is None else calc_reduced_mass(effective_mass, human_mass)
    f = force_limit.to(ureg.newton).magnitude
    m = mass.to(ureg.kg).magnitude
    k = contact_stiffness.to(ureg.newton / ureg.meter).magnitude

    return (f / math.sqrt(k * m)) * (ureg.meter / ureg.second)


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
    Project onto a supplied abstract half-space in Euclidean control coordinates.

    Solves the quadratic program:
        min  1/2 ||u - u_nom||^2
        s.t. a^T u <= b

    The caller must derive ``a`` and ``b`` from its plant and barrier. This
    arithmetic alone proves neither forward invariance nor a physical stop.
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
    free_voxels_per_ray: int = 294,
    bytes_per_ray_voxel: int = 8,
    cache_hit_rate: float = 0.95,
):
    """
    Calculate partial TSDF storage and modeled DRAM traffic for dense vs sparse grids.

    The four-byte voxel stores distance and fusion weight only. The sparse result
    includes the stipulated leaf header, but neither result includes a separate
    occupancy layer, per-cell evidence epochs, lookup/allocator metadata, or
    other full-map state. The hit rate is an assumed workload input, not a
    property of sparse allocation.

    Equation:
        N_dense = V_workspace / (Delta_v)^3
        M_dense = N_dense * B_voxel
        N_active = N_dense * S_sparsity
        N_blocks = ceil(N_active / B_block)
        M_sparse = N_blocks * (B_block * B_voxel + B_header)
        BW_dense_dram = R_rays * N_dense_lookups * B_lookup
        BW_sparse_dram = R_rays * (N_free_occupancy + N_TSDF_band) * B_lookup * (1 - Hit_cache)

    Parameters
    ----------
    workspace_volume : Quantity
        Total bounding volume of operational workcell (e.g. 300 m^3).
    voxel_size : Quantity
        Isotropic voxel side length (e.g. 1.0 cm or 5.0 mm).
    bytes_per_voxel : int
        TSDF payload bytes per voxel (default 4: 16-bit distance + 16-bit weight).
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
        TSDF fusion updates in the narrow surface band (default 6 lookups).
    free_voxels_per_ray : int
        Separate occupancy free-ray updates before the hit (default 294). Sparse
        TSDF allocation does not remove this traversal or its occupancy work.
    bytes_per_ray_voxel : int
        Read-modify-write memory traffic per ray step (default 8 B).
    cache_hit_rate : float
        Assumed measured SRAM/L2 hit fraction for both occupancy and TSDF updates
        (default 0.95 for the illustrative scenario; not a sparse-map property).

    Returns
    -------
    dict
        Voxel count, dense TSDF payload, sparse blocks, sparse TSDF payload
        plus leaf headers, and modeled DRAM bus bandwidths. The legacy
        ``dense_memory`` and ``sparse_memory`` keys are not full-map totals.
    """
    validate_positive(workspace_volume, "workspace_volume")
    validate_positive(voxel_size, "voxel_size")
    if not (0.0 <= cache_hit_rate <= 1.0):
        raise ValueError(f"cache_hit_rate must be in [0, 1], got {cache_hit_rate}")

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
        validate_at_least(free_voxels_per_ray, 0, "free_voxels_per_ray")
        sparse_dram_raw = r_rate * (free_voxels_per_ray + sparse_voxels_per_ray) * bytes_per_ray_voxel * (1.0 - cache_hit_rate)

        result["dense_dram_bandwidth"] = (dense_dram_raw * (ureg.byte / ureg.second)).to(ureg.gigabyte / ureg.second)
        result["sparse_dram_bandwidth"] = (sparse_dram_raw * (ureg.byte / ureg.second)).to(ureg.megabyte / ureg.second)
        result["free_occupancy_updates_per_ray"] = free_voxels_per_ray
        result["tsdf_band_updates_per_ray"] = sparse_voxels_per_ray
        result["traversed_voxels_per_ray"] = free_voxels_per_ray + sparse_voxels_per_ray

    return result


def calc_target_evidence_horizon(
    tolerance_radius,
    sensor_noise,
    drift_velocity,
):
    """
    Calculate the target-evidence validity horizon under a bounded drift model.

    This is not a stopping-time or stopping-distance bound. Motion admission also
    needs current clearance, total pre-brake delay, achievable deceleration, and
    a feasible fallback after this evidence expires.

    Equation:
        tau = (r_tol - e_0) / v_drift

    Parameters
    ----------
    tolerance_radius : Quantity
        Defended task spatial tolerance radius (e.g. mm or m).
    sensor_noise : Quantity
        Initial target-error allowance e_0 (legacy parameter name). It is a hard
        bound only when the sensing model justifies one; a quantile gives a
        validity horizon at its declared tail risk. Must be < tolerance_radius.
    drift_velocity : Quantity
        Bounded relative target drift under the declared operating envelope
        (e.g. m/s), or a calibrated rate at a declared tail risk.

    Returns
    -------
    Quantity
        Maximum target-evidence age in milliseconds.
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


def calc_tracked_evidence_horizon(tolerance_radius, initial_error, slip_acceleration):
    """
    Calculate the tracked target-evidence horizon under residual slip acceleration.

    A tracker that follows a moving target removes its steady velocity, so the
    remaining error growth comes from a bounded residual acceleration a_slip.
    A segment executed open-loop for a time T then drifts by a_slip T^2 / 2;
    setting that drift equal to the error margin r_tol - e_0 gives

    Equation:
        tau_ev = sqrt(2 (r_tol - e_0) / a_slip)

    This is the constant-acceleration counterpart of
    ``calc_target_evidence_horizon`` (constant drift velocity, a_slip = 0).
    Like that function it is not a stopping-time bound.

    Source: first-passage time of a constant-acceleration displacement
    (d = a T^2 / 2, solved for T at d = r_tol - e_0); kinematics, derived here.

    Parameters
    ----------
    tolerance_radius : Quantity
        Defended task spatial tolerance radius r_tol (e.g. mm).
    initial_error : Quantity
        Initial target-error allowance e_0; must be >= 0 and < tolerance_radius.
    slip_acceleration : Quantity
        Bounded residual target acceleration a_slip after tracking (m/s^2).

    Returns
    -------
    Quantity
        Tracked evidence horizon in milliseconds.
    """
    validate_positive(tolerance_radius, "tolerance_radius")
    validate_nonnegative(initial_error, "initial_error")
    validate_positive(slip_acceleration, "slip_acceleration")

    margin = (tolerance_radius - initial_error).to(ureg.meter)
    if margin.magnitude <= 0:
        raise ValueError("initial_error must be strictly less than tolerance_radius.")
    a = slip_acceleration.to(ureg.meter / ureg.second**2)
    return ((2 * margin / a) ** 0.5).to(ureg.millisecond)


def calc_linear_invalidation_horizon(initial_error, error_growth_rate, error_limit):
    """
    Calculate the invalidation horizon of a belief whose error bound grows linearly.

    Equation:
        E(dt) = E_0 + r * dt ;  T_inv = max(0, (E_max - E_0) / r)

    A zero result means the belief never supported the task: its initial error
    already consumes the tolerance. Unlike ``calc_target_evidence_horizon`` this
    does not raise when E_0 >= E_max; it floors the deadline at zero.

    Source: first-passage time of a linear error bound (definition of the
    invalidation horizon, T_inv = inf{dt >= 0 : E(dt) >= E_max}); the
    reachable-set lineage for occluded free space is Orzechowski, Meyer, and
    Lauer 2018 (IEEE ITSC).
    """
    validate_positive(error_growth_rate, "error_growth_rate")
    e0 = initial_error.to(ureg.meter).magnitude
    e_max = error_limit.to(ureg.meter).magnitude
    r = error_growth_rate.to(ureg.meter / ureg.second).magnitude
    return (max(e_max - e0, 0.0) / r * ureg.second).to(ureg.millisecond)


def calc_free_space_expiry(free_depth, loc_bound, protective_clearance, approach_speed):
    """
    Calculate how long a free-space claim survives before an unseen person could breach the clearance.

    Equation:
        T_inv = max(0, (d_free - delta_loc - delta_margin) / v_h)

    The free region erodes from its unobserved boundary at the approach speed
    v_h; localization error delta_loc and the protective clearance delta_margin
    are spent up front. It is ``calc_linear_invalidation_horizon`` with
    E_0 = delta_loc, r = v_h, and E_max = d_free - delta_margin.

    Source: linear invalidation horizon applied to occluded free space; the
    walking-speed bound v_h is ISO 13855's K (1.6 m/s).
    """
    validate_positive(approach_speed, "approach_speed")
    return calc_linear_invalidation_horizon(loc_bound, approach_speed, free_depth - protective_clearance)


def calc_scurve_rest_to_rest_time(distance, max_acceleration, max_jerk):
    """
    Calculate the duration of a jerk-limited rest-to-rest move with no cruise phase.

    A symmetric double-S profile that accelerates, then immediately decelerates.
    With t_j = a/j and x the time from the start of the jerk ramp to the end of
    the constant-acceleration plateau minus t_j (peak speed a*x):

        plateau reached (x >= t_j):  D = a * x * (x + t_j),  T = 2 (x + t_j)
        no plateau (D < 2 a^3 / j^2): D = 2 j tau^3,           T = 4 tau

    Returns a dict with ``duration``, ``peak_velocity``, and ``reaches_max_acceleration``.
    Works for linear or angular moves (distance in m or rad, rates to match).

    Source: Biagiotti and Melchiorri, Trajectory Planning for Automatic Machines
    and Robots (Springer, 2008), ch. 3, double-S velocity profile.
    """
    validate_positive(distance, "distance")
    validate_positive(max_acceleration, "max_acceleration")
    validate_positive(max_jerk, "max_jerk")

    t_unit = ureg.second
    d_unit = distance.units
    d = distance.magnitude
    a = max_acceleration.to(d_unit / t_unit**2).magnitude
    j = max_jerk.to(d_unit / t_unit**3).magnitude

    t_j = a / j
    x = (-t_j + math.sqrt(t_j**2 + 4 * d / a)) / 2
    if x >= t_j:
        duration = 2 * (x + t_j)
        v_peak = a * x
        plateau = True
    else:
        tau = (d / (2 * j)) ** (1.0 / 3.0)
        duration = 4 * tau
        v_peak = j * tau**2
        plateau = False
    return {
        "duration": duration * t_unit,
        "peak_velocity": v_peak * (d_unit / t_unit),
        "reaches_max_acceleration": plateau,
    }


def calc_quasistatic_spill_acceleration(inner_radius, freeboard, gravity=None):
    """
    Calculate the steady horizontal acceleration at which an open container's liquid reaches its rim.

    Equation:
        tan(theta) = a / g ;  wall rise = R a / g ;  a_max = g h_f / R

    Under steady acceleration the free surface settles perpendicular to the
    effective gravity. Sloshing overshoots this steady tilt, so the bound is
    necessary, not sufficient.

    Source: hydrostatics of a uniformly accelerated liquid (free surface normal
    to effective gravity); container slosh context in Mayer and Krechetnikov
    2012, "Walking with coffee: why does it spill?", Phys. Rev. E 85, 046117.
    """
    validate_positive(inner_radius, "inner_radius")
    validate_positive(freeboard, "freeboard")
    g = (1 * ureg.standard_gravity) if gravity is None else gravity
    return (g * freeboard / inner_radius).to(ureg.meter / ureg.second**2)


def calc_intent_drift_lease(tolerance_radius, sensor_noise, drift_velocity):
    """Compatibility name for :func:`calc_target_evidence_horizon`.

    The result limits target-evidence age, not the time needed to stop.
    """
    return calc_target_evidence_horizon(tolerance_radius, sensor_noise, drift_velocity)


def calc_process_thermal_runaway_lease(
    heat_generation_rate,
    thermal_capacitance,
    max_temp_overshoot,
):
    """
    Calculate lumped thermal rise rate and an assumed overshoot horizon.

    This heat-capacity model does not predict thermal runaway or venting;
    those require chemistry, initial temperature, cooling, and validated thresholds.

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
        rate_of_rise and tau_lease, the latter an overshoot horizon only.
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


def calc_axi_fifo_contention(
    *,
    tile_bytes: int,
    camera_bytes: int,
    telemetry_bytes: int,
    safety_read_bytes: int,
    transaction_bytes: int,
    bandwidth,
    bank_switches: int,
    bank_switch_time,
    compute_time,
):
    """Compare a constructed FIFO backlog with priority bypass at AXI granularity.

    All transfers are aligned multiples of ``transaction_bytes``. The FIFO
    scenario admits the entire tile, camera transfer, and telemetry transfer
    ahead of the safety read, then allows no new arrivals. The priority-bypass
    scenario waits only for one already in-flight transaction. These are
    controller-policy assumptions, not AXI timing guarantees. Peak bandwidth
    ignores arbitration, refresh, and protocol overhead, so times are floors.
    """
    for name, count in (
        ("tile_bytes", tile_bytes),
        ("camera_bytes", camera_bytes),
        ("telemetry_bytes", telemetry_bytes),
        ("safety_read_bytes", safety_read_bytes),
        ("transaction_bytes", transaction_bytes),
    ):
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if any(count % transaction_bytes for count in (
        tile_bytes, camera_bytes, telemetry_bytes, safety_read_bytes
    )):
        raise ValueError("all transfers must align to transaction_bytes")
    if not isinstance(bank_switches, int) or bank_switches < 0:
        raise ValueError("bank_switches must be a nonnegative integer")
    validate_positive(bandwidth, "bandwidth")
    validate_nonnegative(bank_switch_time, "bank_switch_time")
    validate_nonnegative(compute_time, "compute_time")

    byte_rate = bandwidth.to(ureg.byte / ureg.second)
    switch_time = bank_switch_time.to(ureg.second)
    compute = compute_time.to(ureg.second)
    fifo_bytes = tile_bytes + camera_bytes + telemetry_bytes
    queue_fifo = fifo_bytes * ureg.byte / byte_rate + bank_switches * switch_time
    queue_bypass = transaction_bytes * ureg.byte / byte_rate
    own_read = safety_read_bytes * ureg.byte / byte_rate
    return {
        "queued_transactions": fifo_bytes // transaction_bytes,
        "tile_transactions": tile_bytes // transaction_bytes,
        "queue_fifo": queue_fifo.to(ureg.microsecond),
        "queue_priority_bypass": queue_bypass.to(ureg.microsecond),
        "own_read": own_read.to(ureg.microsecond),
        "total_fifo": (compute + queue_fifo + own_read).to(ureg.microsecond),
        "total_priority_bypass": (compute + queue_bypass + own_read).to(ureg.microsecond),
    }


def calc_c2_effective_deceleration(peak_deceleration):
    """
    Calculate the equivalent constant deceleration for a C2-continuous stop.

    A C2-continuous stop suffix entered from cruise velocity v with peak
    deceleration a_peak requires stopping distance d_C2 = 0.75 * v^2 / a_peak
    (derived in calc_c2_stop_suffix). The equivalent constant deceleration
    a_eff that achieves the exact same stopping distance (v^2 / (2 * a_eff) = d_C2) is:

        a_eff = a_peak / 1.5 = (2 / 3) * a_peak

    Parameters
    ----------
    peak_deceleration : Quantity
        Peak credible deceleration of the C2 stop profile (e.g. m/s^2).

    Returns
    -------
    Quantity
        Equivalent constant deceleration in m/s^2.
    """
    validate_positive(peak_deceleration, "peak_deceleration")
    a = peak_deceleration.to(ureg.meter / ureg.second**2)
    return (a / 1.5).to(ureg.meter / ureg.second**2)


def calc_latch_contact_stopping_profile(
    approach_velocity=None,
    tripwire_force=None,
    latch_stiffness=None,
    response_time=None,
    force_limit=None,
    *,
    scenario=None,
):
    """
    Calculate contact force rise, brake command force, and required arm deceleration upon latch contact.

    When a robot manipulator contacts a compliant latch or strike plate:
    1. Force accumulation rate: df/dt = k_latch * v_approach
    2. Force at brake command onset (tripwire force + accumulation during response time):
       f_cmd = f_tripwire + k_latch * v_approach * t_response
    3. Remaining allowable deflection before exceeding contact limit f_limit:
       x_remaining = (f_limit - f_cmd) / k_latch
    4. Required constant arm deceleration to halt within x_remaining:
       a_arm_needed = v_approach^2 / (2 * x_remaining)
    5. Duration of the arm stopping phase:
       t_arm_stop = v_approach / a_arm_needed

    Parameters
    ----------
    approach_velocity : Quantity, optional
        Arm approach speed (e.g. m/s). If omitted and scenario is given, defaults to scenario.v_latch_approach.
    tripwire_force : Quantity, optional
        Tripwire threshold force (e.g. N). Defaults to scenario.f_latch_tripwire if scenario is given.
    latch_stiffness : Quantity, optional
        Spring stiffness of the latch mechanism (e.g. N/m). Defaults to scenario.k_latch.
    response_time : Quantity, optional
        Contact-loop response latency from tripwire detection to brake onset (e.g. ms).
        Defaults to scenario.t_contact_response.
    force_limit : Quantity, optional
        Maximum permissible contact force limit (e.g. N). Defaults to scenario.f_latch_limit.
    scenario : EmbodiedSiteScenario, optional
        Site scenario supplying default latch physical parameters.

    Returns
    -------
    dict
        - "f_at_brake_onset": Force at brake command onset (Quantity in N).
        - "x_remaining": Remaining allowable displacement (Quantity in mm).
        - "a_arm_needed": Required arm deceleration to honor force limit (Quantity in m/s^2).
        - "t_arm_stop": Arm stopping duration (Quantity in ms).
        - "df_dt": Force rate of rise during contact (Quantity in N/s).
    """
    if scenario is not None:
        if approach_velocity is None:
            approach_velocity = scenario.v_latch_approach
        if tripwire_force is None:
            tripwire_force = scenario.f_latch_tripwire
        if latch_stiffness is None:
            latch_stiffness = scenario.k_latch
        if response_time is None:
            response_time = scenario.t_contact_response
        if force_limit is None:
            force_limit = scenario.f_latch_limit

    if approach_velocity is None or tripwire_force is None or latch_stiffness is None or response_time is None or force_limit is None:
        raise ValueError("All parameters must be provided directly or inferred from scenario.")

    validate_positive(approach_velocity, "approach_velocity")
    validate_positive(tripwire_force, "tripwire_force")
    validate_positive(latch_stiffness, "latch_stiffness")
    validate_positive(response_time, "response_time")
    validate_positive(force_limit, "force_limit")

    v = approach_velocity.to(ureg.meter / ureg.second)
    k = latch_stiffness.to(ureg.newton / ureg.meter)
    f_trip = tripwire_force.to(ureg.newton)
    t_resp = response_time.to(ureg.second)
    f_lim = force_limit.to(ureg.newton)

    df_dt = (k * v).to(ureg.newton / ureg.second)
    f_cmd = calc_tripwire_contact_force_accumulation(f_trip, k, v, t_resp).to(ureg.newton)
    if f_cmd >= f_lim:
        raise ValueError(f"Brake onset force {f_cmd} already exceeds or meets limit {f_lim}")

    x_rem = ((f_lim - f_cmd) / k).to(ureg.millimeter)
    x_rem_m = x_rem.to(ureg.meter)
    a_need = (v**2 / (2 * x_rem_m)).to(ureg.meter / ureg.second**2)
    t_stop = (v / a_need).to(ureg.millisecond)

    return {
        "f_at_brake_onset": f_cmd,
        "x_remaining": x_rem,
        "a_arm_needed": a_need,
        "t_arm_stop": t_stop,
        "df_dt": df_dt,
    }


def calc_scenario_stopping_budget(
    scenario,
    *,
    velocity=None,
    deceleration=None,
    c2_profile=True,
):
    """
    Calculate stopping distance budget, spare clearance, and speed ceiling for a scenario.

    Consolidates scenario-level stopping physics from an EmbodiedSiteScenario
    (e.g., WarehouseAisle):
    1. Pre-braking delay tau_delay is taken from scenario.tau_delay.
    2. Operating velocity defaults to scenario.v_aisle.
    3. Peak deceleration defaults to scenario.a_brake.
    4. When c2_profile=True, calculates effective constant deceleration
       a_eff = calc_c2_effective_deceleration(deceleration); otherwise a_eff = deceleration.
    5. Localization and tracking overhead is overhead_loc = scenario.delta_loc + scenario.eps_track.
    6. Physical margin is phys_margin = scenario.delta_margin.
    7. Total stopping distance budget:
       budget = calc_safe_stopping_distance(v, tau_delay, a_eff, loc_margin=overhead_loc, physical_margin=phys_margin)
    8. Residual spare clearance:
       spare = scenario.d_clear - budget
    9. Maximum permitted velocity ceiling:
       v_ceiling = calc_max_permitted_velocity(scenario.d_clear, tau_delay, a_eff, loc_margin=overhead_loc, physical_margin=phys_margin)

    Parameters
    ----------
    scenario : EmbodiedSiteScenario
        Site scenario instance containing d_clear, delta_loc, eps_track, delta_margin,
        tau_delay, and default speeds.
    velocity : Quantity, optional
        Operating velocity to evaluate (default: scenario.v_aisle).
    deceleration : Quantity, optional
        Loaded deceleration (default: scenario.a_brake).
    c2_profile : bool, optional
        Whether the stop follows a C2-continuous profile (default: True).

    Returns
    -------
    dict
        - "budget": Total safe stopping distance budget (Quantity in mm).
        - "spare": Residual clearance margin d_clear - budget (Quantity in mm).
        - "v_ceiling": Maximum safe operating velocity (Quantity in m/s).
        - "a_eff": Effective deceleration used (Quantity in m/s^2).
        - "tau_delay": Pre-braking delay (Quantity in ms).
        - "overhead_loc": Localization and tracking uncertainty (Quantity in m).
    """
    v = velocity if velocity is not None else scenario.v_aisle
    a_peak = deceleration if deceleration is not None else scenario.a_brake

    validate_positive(v, "velocity")
    validate_positive(a_peak, "deceleration")

    a_eff = calc_c2_effective_deceleration(a_peak) if c2_profile else a_peak
    tau = scenario.tau_delay
    overhead_loc = (scenario.delta_loc + scenario.eps_track).to(ureg.meter)
    phys_margin = scenario.delta_margin.to(ureg.meter)

    budget = calc_safe_stopping_distance(
        v, tau, a_eff, loc_margin=overhead_loc, physical_margin=phys_margin
    ).to(ureg.millimeter)

    spare = (scenario.d_clear - budget).to(ureg.millimeter)

    v_ceil = calc_max_permitted_velocity(
        scenario.d_clear, tau, a_eff, loc_margin=overhead_loc, physical_margin=phys_margin
    ).to(ureg.meter / ureg.second)

    return {
        "budget": budget,
        "spare": spare,
        "v_ceiling": v_ceil,
        "a_eff": a_eff,
        "tau_delay": tau,
        "overhead_loc": overhead_loc,
    }

