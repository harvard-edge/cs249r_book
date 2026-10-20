"""Joint mechanics, contact dynamics, and thermal/electrical actuation limits."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_at_least, validate_nonnegative, validate_positive


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


def calc_smoothstep_blend_window(change, rate_limit):
    """
    Shortest window of a fixed-endpoint quintic smoothstep whose peak rate stays at a limit.

    Equation:
        T >= 1.875 |Delta u| / rate_max

    The smoothstep alpha(s) = 10 s^3 - 15 s^4 + 6 s^5 has peak slope
    d alpha / ds = 30/16 = 1.875 at s = 1/2, so a change Delta u spread over T
    peaks at 1.875 Delta u / T. Unit-generic form of
    ``calc_quintic_blend_duration``: an acceleration change against a jerk
    bound, or a torque change against a torque-rate ceiling (added 2026-10-20).

    Returns
    -------
    Quantity
        Window in seconds.
    """
    validate_positive(rate_limit, "rate_limit")
    return (1.875 * abs(change) / rate_limit).to(ureg.second)


def calc_coasting_contact_peak_force(contact_velocity, effective_mass, contact_stiffness, preload_force):
    """
    Peak force when drive torque is removed at a contact tripwire and the mass coasts into a spring.

    Energy balance from the tripwire, where the spring already carries ``preload_force``:
        F0^2 / (2k) + m v^2 / 2 = F_peak^2 / (2k)  =>  F_peak = sqrt(F0^2 + k m v^2)

    With ``preload_force`` zero this is the unbraked transient force v sqrt(k m)
    (``calc_transient_impact_force``). Added 2026-10-20.
    """
    validate_positive(contact_velocity, "contact_velocity")
    validate_positive(effective_mass, "effective_mass")
    validate_positive(contact_stiffness, "contact_stiffness")
    f0 = preload_force.to(ureg.newton).magnitude
    v = contact_velocity.to(ureg.meter / ureg.second).magnitude
    m = effective_mass.to(ureg.kg).magnitude
    k = contact_stiffness.to(ureg.newton / ureg.meter).magnitude
    return math.sqrt(f0 ** 2 + k * m * v ** 2) * ureg.newton


def calc_passive_compliance_stiffness_bound(force_limit, tripwire_force, detection_speed, response_latency,
                                            arm_deceleration):
    """
    Largest series stiffness that keeps a driven contact under a force limit after the tripwire.

    After the tripwire the arm keeps driving for the response latency, then stops at
    ``arm_deceleration``; the stroke it drives is s = v t_r + v^2 / (2 a), and the force
    rises by k s on top of the tripwire force:
        F_trip + k s <= F_limit  <=>  k <= (F_limit - F_trip) / (v t_r + v^2 / (2 a))

    Returns (stiffness bound, driven stroke). Added 2026-10-20.
    """
    validate_positive(detection_speed, "detection_speed")
    validate_positive(arm_deceleration, "arm_deceleration")
    stroke = (detection_speed * response_latency + detection_speed ** 2 / (2 * arm_deceleration)).to(ureg.millimeter)
    k = ((force_limit - tripwire_force) / stroke).to(ureg.newton / ureg.meter)
    return k, stroke
