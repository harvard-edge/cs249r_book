"""Joint mechanics, contact dynamics, and thermal/electrical actuation limits."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_at_least, validate_nonnegative, validate_positive


def calc_kinetic_energy(mass, velocity):
    """Calculate kinetic energy of a translating body.

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

    Notes
    -----
    Equation:
        E_k = 0.5 * m * v^2

    Assumptions: A translating point mass has nonrelativistic speed; mass is positive and speed
    is nonnegative.

    Excludes: Rotation, deformation, and dissipated energy.

    References
    ----------
    OpenStax, University Physics Volume 1, Sec. 7.2, Eq. 7.6.

    Derivation: Integrating Newtonian work m v dv from rest gives m v squared divided by two.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_kinetic_energy
    >>> from mlsysim.core.units import ureg
    >>> calc_kinetic_energy(ureg('2 kg'), ureg('3 m/s')).check('[energy]')
    True
    """
    validate_positive(mass, "mass")
    validate_nonnegative(velocity, "velocity")

    energy = 0.5 * mass * (velocity**2)
    return energy.to(ureg.joule)


def calc_reflected_inertia(gear_ratio: float, rotor_inertia):
    """Calculate reflected rotor inertia felt at the actuator output shaft.

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

    Notes
    -----
    Equation:
        J_reflected = N^2 * J_rotor

    Assumptions: The ideal gear reduction has ratio at least one and a rigid rotor.

    Excludes: Gear compliance, efficiency losses, and load inertia.

    References
    ----------
    Lynch and Park, Modern Robotics, Cambridge University Press, 2017,
    Sec. 8.9.2 ("Apparent Inertia"), where G^2 * I_rotor is the rotor's
    apparent (often called reflected) inertia for gear ratio G.

    Derivation: Equate rotor kinetic energy to output-shaft kinetic energy with rotor speed N
    times output speed.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_reflected_inertia
    >>> from mlsysim.core.units import ureg
    >>> calc_reflected_inertia(2, ureg('1 kg*m**2')) == 4 * ureg('1 kg*m**2')
    True
    """
    validate_at_least(gear_ratio, 1.0, "gear_ratio")
    reflected = (gear_ratio**2) * rotor_inertia
    return reflected


def calc_seam_acceleration_torque_jump(gear_ratio: float, rotor_inertia, delta_accel):
    """Calculate the impulsive torque spike at an actuator seam transition.

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

    Notes
    -----
    When an unblended trajectory transition injects an acceleration discontinuity
    delta_accel across a discrete boundary, the transmission's reflected rotor inertia
    amplifies the required inertial reaction torque at the output joint.

    Equation:
        tau_spike = J_reflected * delta_accel = (N^2 * J_rotor) * delta_accel

    Assumptions: A positive rotor inertia is reflected through an ideal reduction; the
    acceleration jump is nonnegative.

    Excludes: Link/load inertia, friction, gear compliance, and finite controller response.

    References
    ----------
    The torque relation tau = I * alpha (OpenStax, University Physics Volume 1, Sec. 10.7,
    Eq. 10.25) applied to the rotor's apparent (reflected) inertia
    G^2 * I_rotor (Lynch and Park, Modern Robotics, Cambridge University
    Press, 2017, Sec. 8.9.2). Link inertia, load, friction, and gear
    compliance are omitted, so this is the reflected-rotor share of the spike;
    treating the seam as an instantaneous acceleration step is a modeling
    assumption.

    Derivation: Multiply the gear-reflected inertia N squared J by the acceleration jump.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_seam_acceleration_torque_jump
    >>> from mlsysim.core.units import ureg
    >>> calc_seam_acceleration_torque_jump(2, ureg('1 kg*m**2'), ureg('1 rad/s**2')).check('[force]*[length]')
    True
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
    """Calculate characteristic instability timescale or fall duration for an inverted pendulum.

    Parameters
    ----------
    effective_length : Quantity
        Center-of-mass height or pendulum leg length L (e.g. meter).
    gravity : Quantity, optional
        Gravitational acceleration g (defaults to 9.80665 m/s^2).
    theta_0 : Quantity or float, optional
        Initial angular perturbation from vertical (e.g. radians or degrees).
    theta_fall : Quantity or float, optional
        User-selected fall angle greater than theta_0; it is not a certified recovery threshold.

    Returns
    -------
    Quantity
        Characteristic timescale or fall duration in seconds.

    Notes
    -----
    Equation:

    Models bipedal/humanoid balance dynamics under gravity. The linearized dynamics
    d^2 theta / dt^2 = (g / L) * theta yield the characteristic natural timescale:
        tau_0 = sqrt(L / g)

    If initial perturbation theta_0 and fall threshold theta_fall are given:
        t_fall = tau_0 * ln(2 * theta_fall / theta_0)

    Assumptions: The pendulum has positive length and gravity. The optional fall estimate
    assumes a small initial angle, zero angular velocity, and a much larger final angle.

    Excludes: Nonlinear large-angle motion, active balancing, and a proof that the chosen angle
    implies irrecoverability.

    References
    ----------
    The linear inverted pendulum has natural frequency
    omega_0 = sqrt(g / h), so tau_0 = sqrt(L / g) (Caron, "Biped Stabilization
    by Linear Feedback of the Variable-Height Inverted Pendulum Model",
    arXiv:1909.07732, Sec. II). The t_fall expression is derived here, not
    taken from a published model. With zero initial angular velocity the
    linearized solution is theta(t) = theta_0 * cosh(t / tau_0), so
    t = tau_0 * acosh(theta_fall / theta_0), and ln(2x) approximates acosh(x)
    for x >> 1 (it overestimates by about 5% at x = 2). The small-angle
    linearization also loses accuracy as theta_fall grows.

    Derivation: The linearized equation has timescale sqrt(L/g); replacing
    acosh(theta_f/theta_0) by log(2 theta_f/theta_0) gives the optional approximation.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_inverted_pendulum_fall_time
    >>> from mlsysim.core.units import ureg
    >>> calc_inverted_pendulum_fall_time(ureg('1 m')).check('[time]')
    True
    """
    import math

    validate_positive(effective_length, "effective_length")
    if gravity is None:
        gravity = 9.80665 * (ureg.meter / (ureg.second**2))
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
    """Calculate DC bus voltage droop across internal battery and cable resistance.

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

    Notes
    -----
    Equation:
        Delta_V = I * R_internal

    Assumptions: Current is nonnegative and a positive lumped DC resistance describes the path.

    Excludes: Inductance, capacitance, and frequency-dependent impedance.

    References
    ----------
    Ohm's law, V = I * R (OpenStax, University Physics Volume 2, Sec. 9.3).

    Derivation: Ohm's law applied to the series resistance gives I R.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_voltage_droop
    >>> from mlsysim.core.units import ureg
    >>> calc_voltage_droop(ureg('2 A'), ureg('1 ohm')) == ureg('2 V')
    True
    """
    validate_nonnegative(current, "current")
    validate_positive(internal_resistance, "internal_resistance")

    droop = current * internal_resistance
    return droop.to(ureg.volt)


def calc_contact_force(stiffness, displacement):
    """Calculate linear Hookean elastic contact force upon unconstrained impact.

    Parameters
    ----------
    stiffness : Quantity
        Positive linear spring stiffness in N/m.
    displacement : Quantity
        Nonnegative compression in m.

    Returns
    -------
    Quantity
        Elastic force in N.

    Notes
    -----
    Equation:
        F = k * delta_x

    Assumptions: The spring is linear with positive stiffness and nonnegative compression.

    Excludes: Damping, nonlinear material response, and dynamic impact trajectory.

    References
    ----------
    Derivation: Hooke's law gives force as stiffness times displacement.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_contact_force
    >>> from mlsysim.core.units import ureg
    >>> calc_contact_force(ureg('100 N/m'), ureg('0.01 m')) == ureg('1 N')
    True
    """
    validate_positive(stiffness, "stiffness")
    validate_nonnegative(displacement, "displacement")
    f = stiffness.to(ureg.newton / ureg.meter) * displacement.to(ureg.meter)
    return f.to(ureg.newton)


def calc_contact_yield_deadline(yield_force, stiffness, approach_velocity):
    """Calculate the constant-speed contact deadline at a force threshold.

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

    Notes
    -----
    Equation:

    For an arm approaching a rigid fixture of stiffness k at constant velocity v,
    contact force rises as F(t) = k * v * t. To prevent structural damage (F <= F_yield):
        t_deadline = F_yield / (k * v)

    Assumptions: Positive approach speed stays constant while a positive linear stiffness
    accumulates force from zero.

    Excludes: Braking after detection, damping, initial preload, and safety certification.

    References
    ----------
    Derivation: Substitute compression v t into Hooke's law and solve k v t = F_yield.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_contact_yield_deadline
    >>> from mlsysim.core.units import ureg
    >>> calc_contact_yield_deadline(ureg('10 N'), ureg('100 N/m'), ureg('1 m/s')).check('[time]')
    True
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
    """Calculate inductive voltage drop during a current transient.

    Parameters
    ----------
    loop_inductance : Quantity
        Positive loop inductance in H.
    current_slew_rate : Quantity
        Positive current slew in A/s.

    Returns
    -------
    Quantity
        Inductive voltage drop in V.

    Notes
    -----
    Equation:
        Delta_V_ind = L * (dI / dt)

    Assumptions: A positive lumped loop inductance experiences a positive current slew.

    Excludes: Resistive droop, decoupling capacitance, and distributed PDN resonances.

    References
    ----------
    Derivation: The ideal inductor terminal relation is voltage L times the time derivative of
    current.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_inductive_voltage_droop
    >>> from mlsysim.core.units import ureg
    >>> calc_inductive_voltage_droop(ureg('1 H'), ureg('1 A/s')) == ureg('1 V')
    True
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
    """Estimate CMOS gate delay stretch using the alpha-power approximation.

    Parameters
    ----------
    nominal_voltage : Quantity
        Nominal rail voltage in V, above threshold in the intended regime.
    drooped_voltage : Quantity
        Drooped rail voltage in V; must exceed threshold.
    threshold_voltage : Quantity, optional
        Threshold in V; None selects the assumed 0.42 V default.
    alpha : float, optional
        Dimensionless alpha-power exponent; assumed default is 1.3.

    Returns
    -------
    dict
        Dimensionless stretch_ratio and delay_expansion_percent relative to nominal delay.

    Notes
    -----
    Equation:

    Sakurai-Newton Alpha-Power Law:
        stretch_ratio = (V_droop / V_nom) * ((V_nom - V_th) / (V_droop - V_th))^alpha

    Assumptions: Both rail voltages exceed the threshold in the intended regime; threshold
    defaults to an assumed 0.42 V and exponent alpha to an assumed 1.3.

    Excludes: Process-specific calibration, frequency control, temperature dependence, and
    behavior at or below threshold.

    References
    ----------
    Derivation: Under the alpha-power approximation delay is proportional to ``V/(V-V_th)**alpha``;
    divide drooped delay by nominal delay.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_alpha_power_gate_delay_stretch
    >>> from mlsysim.core.units import ureg
    >>> calc_alpha_power_gate_delay_stretch(ureg('1 V'), ureg('1 V'))['stretch_ratio'] == 1.0
    True
    """
    validate_positive(nominal_voltage, "nominal_voltage")
    validate_positive(drooped_voltage, "drooped_voltage")

    v_th = (0.42 * ureg.volt) if threshold_voltage is None else threshold_voltage
    v_nom = nominal_voltage.to(ureg.volt).magnitude
    v_drop = drooped_voltage.to(ureg.volt).magnitude
    v_t = v_th.to(ureg.volt).magnitude

    if v_drop <= v_t:
        raise ValueError(f"Drooped voltage {v_drop} V is below or at threshold voltage {v_t} V (circuit stalls).")

    num = v_drop / (v_drop - v_t) ** alpha
    den = v_nom / (v_nom - v_t) ** alpha
    ratio = num / den
    expansion_pct = (ratio - 1.0) * 100.0

    return {
        "stretch_ratio": float(ratio),
        "delay_expansion_percent": float(expansion_pct),
    }


def calc_reduced_mass(mass_a, mass_b):
    """Calculate the two-body reduced mass of a collision.

    Parameters
    ----------
    mass_a : Quantity
        Positive first-body mass in kg.
    mass_b : Quantity
        Positive second-body mass in kg.

    Returns
    -------
    Quantity
        Reduced mass in kg.

    Notes
    -----
    Equation:
        mu = (1/m_a + 1/m_b)^-1

    When one body cannot recoil (m_b -> infinity, a clamped contact), mu -> m_a.

    Assumptions: Both masses are positive and free to recoil in a one-dimensional two-body
    collision.

    Excludes: Rotational inertia coupling, constraints, and contact compliance.

    References
    ----------
    ISO/TS 15066:2016 Annex A, transient-contact model (content carried
    into ISO 10218-2:2025), as reproduced in Kirschner et al. 2022, eq. 2
    (arXiv:2203.02706).

    Derivation: In the relative coordinate, inverse effective mass is the sum of the two inverse
    masses.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_reduced_mass
    >>> from mlsysim.core.units import ureg
    >>> calc_reduced_mass(ureg('2 kg'), ureg('2 kg')) == ureg('1 kg')
    True
    """
    validate_positive(mass_a, "mass_a")
    validate_positive(mass_b, "mass_b")
    m_a = mass_a.to(ureg.kg).magnitude
    m_b = mass_b.to(ureg.kg).magnitude
    return (1.0 / (1.0 / m_a + 1.0 / m_b)) * ureg.kg


def calc_transient_impact_force(contact_velocity, effective_mass, contact_stiffness, human_mass=None):
    """Calculate peak unbraked transient impact force on an elastic body.

    Parameters
    ----------
    contact_velocity : Quantity
        Positive relative approach speed in m/s.
    effective_mass : Quantity
        Positive robot-side effective mass in kg.
    contact_stiffness : Quantity
        Positive linear contact stiffness in N/m.
    human_mass : Quantity, optional
        Positive recoiling-body mass in kg; None uses the robot mass alone.

    Returns
    -------
    Quantity
        Modeled peak elastic force in N.

    Notes
    -----
    Equation:
        F_peak = v * sqrt(k * m)

    With ``human_mass=None`` the robot-side ``effective_mass`` acts alone
    (m = m_eff), the clamped (quasi-static) case in which the body part cannot
    recoil. With ``human_mass`` given, the free body part recoils and the
    two-body reduced mass acts: m = mu = (1/m_eff + 1/m_H)^-1.

    Assumptions: A positive approach speed loads a positive linear stiffness without braking;
    optional human mass gives free recoil, otherwise the contacted body is clamped.

    Excludes: Damping, geometry-dependent stiffness, active control, injury prediction, and
    safety certification.

    References
    ----------
    ISO/TS 15066:2016 Annex A energy model F = v * sqrt(mu * k)
    (content carried into ISO 10218-2:2025), as reproduced in Kirschner et al.
    2022, eqs. 1-2 (arXiv:2203.02706).

    Derivation: Equate relative kinetic energy m v squared/2 with spring energy F squared/(2 k).

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_transient_impact_force
    >>> from mlsysim.core.units import ureg
    >>> calc_transient_impact_force(ureg('1 m/s'), ureg('1 kg'), ureg('1 N/m')) == ureg('1 N')
    True
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
    """Calculate the highest contact speed whose transient peak force stays at a force limit.

    Parameters
    ----------
    force_limit : Quantity
        Force ceiling in N.
    effective_mass : Quantity
        Positive robot-side effective mass in kg.
    contact_stiffness : Quantity
        Positive linear contact stiffness in N/m.
    human_mass : Quantity, optional
        Positive recoiling-body mass in kg; None uses the robot mass alone.

    Returns
    -------
    Quantity
        Modeled contact-speed ceiling in m/s.

    Notes
    -----
    Equation:
        v_max = F_max / sqrt(k * m)

    The inverse of ``calc_transient_impact_force``: ``human_mass=None`` gives the
    clamped case (m = m_eff); a ``human_mass`` gives the free, recoiling case
    (m = mu, the two-body reduced mass).

    Assumptions: Positive force limit, effective mass, and linear stiffness describe the same
    elastic model as calc_transient_impact_force.

    Excludes: Damping, control response, biological variability, and certification of a safe
    operating speed.

    References
    ----------
    ISO/TS 15066:2016 Annex A (carried into ISO 10218-2:2025), as used
    for v_rel,max in Kirschner et al. 2022 (arXiv:2203.02706).

    Derivation: Solve the elastic energy equation F = v sqrt(k m) for v.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_transient_contact_speed_limit
    >>> from mlsysim.core.units import ureg
    >>> calc_transient_contact_speed_limit(ureg('1 N'), ureg('1 kg'), ureg('1 N/m')) == ureg('1 m/s')
    True
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
    """Calculate contact force at brake-command onset after a tripwire breach.

    Parameters
    ----------
    tripwire_force : Quantity
        Force at tripwire detection in N.
    contact_stiffness : Quantity
        Positive linear contact stiffness in N/m.
    penetration_velocity : Quantity
        Positive continued penetration speed in m/s.
    loop_latency : Quantity
        Positive response delay in s.

    Returns
    -------
    Quantity
        Contact force at brake-command onset in N.

    Notes
    -----
    Equation:
        F_lat = F_trip + k * v * t_lat

    Assumptions: Positive speed remains constant during a positive loop delay after a positive
    tripwire threshold; stiffness is linear and positive.

    Excludes: Braking travel after command onset, damping, and contact safety certification.

    References
    ----------
    Derivation: Add the spring-force increment k times additional penetration v t to the
    tripwire force.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_tripwire_contact_force_accumulation
    >>> from mlsysim.core.units import ureg
    >>> calc_tripwire_contact_force_accumulation(ureg('1 N'), ureg('1 N/m'), ureg('1 m/s'), ureg('1 s')) == ureg('2 N')
    True
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
    """Compute minimum blend duration for quintic polynomial trajectory to bound torque jerk.

    Parameters
    ----------
    delta_torque : Quantity
        Signed torque change in N m; the magnitude sets duration.
    max_allowable_jerk : Quantity
        Positive torque-rate ceiling in N m/s, not angular jerk.

    Returns
    -------
    Quantity
        Required smoothstep duration in ms.

    Notes
    -----
    Equation:
        tau_blend >= (1.875 * |Delta_tau|) / jerk_max

    Assumptions: The torque command follows a fixed-endpoint quintic smoothstep; the positive
    rate limit is a torque rate, despite the legacy jerk parameter name.

    Excludes: Mechanical angular jerk, actuator tracking error, and time optimality among other
    trajectory families.

    References
    ----------
    Derivation: The peak derivative of 10 s cubed - 15 s fourth + 6 s fifth is 1.875; divide
    peak scaled torque slope by the allowed torque rate.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_quintic_blend_duration
    >>> from mlsysim.core.units import ureg
    >>> calc_quintic_blend_duration(ureg('0 N*m'), ureg('1 N*m/s')) == ureg('0 ms')
    True
    """
    validate_positive(max_allowable_jerk, "max_allowable_jerk")
    d_tau = abs(delta_torque.to(ureg.newton * ureg.meter).magnitude) * (ureg.newton * ureg.meter)
    j_max = max_allowable_jerk.to(ureg.newton * ureg.meter / ureg.second)
    tau = (1.875 * d_tau) / j_max
    return tau.to(ureg.millisecond)


def calc_quasistatic_spill_acceleration(inner_radius, freeboard, gravity=None):
    """Calculate the steady-tilt acceleration ceiling for an open vessel.

    Parameters
    ----------
    inner_radius : Quantity
        Positive vessel inner radius in m.
    freeboard : Quantity
        Positive liquid-to-rim height in m.
    gravity : Quantity, optional
        Positive gravitational acceleration in m/s^2; None uses standard gravity.

    Returns
    -------
    Quantity
        Steady-tilt acceleration ceiling in m/s^2.

    Notes
    -----
    Equation:
        tan(theta) = a / g ;  wall rise = R a / g ;  a_max = g h_f / R

    Under steady acceleration the free surface settles perpendicular to the
    effective gravity. Sloshing overshoots this steady tilt, so the bound is
    necessary, not sufficient.

    Assumptions: A positive-radius open vessel has positive freeboard and reaches a steady free-
    surface tilt under positive gravity.

    Excludes: Transient slosh, viscosity, vessel shape details, and a sufficient no-spill
    guarantee.

    References
    ----------
    Hydrostatics of a uniformly accelerated liquid gives a free surface normal
    to effective gravity. Container slosh is discussed by Mayer and Krechetnikov
    2012, "Walking with coffee: why does it spill?", Phys. Rev. E 85, 046117.

    Derivation: The free surface is perpendicular to effective gravity, so wall rise R a/g
    reaches freeboard h at a = g h/R.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_quasistatic_spill_acceleration
    >>> from mlsysim.core.units import ureg
    >>> calc_quasistatic_spill_acceleration(ureg('1 m'), ureg('1 m'), ureg('1 m/s**2')) == ureg('1 m/s**2')
    True
    """
    validate_positive(inner_radius, "inner_radius")
    validate_positive(freeboard, "freeboard")
    g = (1 * ureg.standard_gravity) if gravity is None else gravity
    return (g * freeboard / inner_radius).to(ureg.meter / ureg.second**2)


def calc_smoothstep_blend_window(change, rate_limit):
    """Shortest window of a fixed-endpoint quintic smoothstep whose peak rate stays at a limit.

    Parameters
    ----------
    change : Quantity
        Signed command change; its units must match rate_limit times seconds.
    rate_limit : Quantity
        Positive allowed command rate in change-units per second.

    Returns
    -------
    Quantity
        Window in seconds.

    Notes
    -----
    Equation:
        T >= 1.875 |Delta u| / rate_max

    The smoothstep alpha(s) = 10 s^3 - 15 s^4 + 6 s^5 has peak slope
    d alpha / ds = 30/16 = 1.875 at s = 1/2, so a change Delta u spread over T
    peaks at 1.875 Delta u / T. Unit-generic form of
    ``calc_quintic_blend_duration``: an acceleration change against a jerk
    bound, or a torque change against a torque-rate ceiling (added 2026-10-20).

    Assumptions: The change has units compatible with a positive rate limit times time; a
    quintic fixed-endpoint smoothstep is used.

    Excludes: Tracking error, other profile shapes, and dynamic feasibility beyond the rate
    ceiling.

    References
    ----------
    Derivation: The quintic smoothstep peak slope is 1.875; scaling by absolute change and
    dividing by rate gives the required window.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_smoothstep_blend_window
    >>> from mlsysim.core.units import ureg
    >>> calc_smoothstep_blend_window(ureg('0 m/s**2'), ureg('1 m/s**3')) == ureg('0 s')
    True
    """
    validate_positive(rate_limit, "rate_limit")
    return (1.875 * abs(change) / rate_limit).to(ureg.second)


def calc_coasting_contact_peak_force(contact_velocity, effective_mass, contact_stiffness, preload_force):
    """Calculate peak spring force after drive removal at a contact tripwire.

    Parameters
    ----------
    contact_velocity : Quantity
        Positive relative approach speed in m/s.
    effective_mass : Quantity
        Positive robot-side effective mass in kg.
    contact_stiffness : Quantity
        Positive linear contact stiffness in N/m.
    preload_force : Quantity
        Initial spring-force magnitude in N, nonnegative in the intended model.

    Returns
    -------
    Quantity
        Peak elastic contact force in N.

    Notes
    -----
    Equation:

    Energy balance from the tripwire, where the spring already carries ``preload_force``:
        F0^2 / (2k) + m v^2 / 2 = F_peak^2 / (2k)  =>  F_peak = sqrt(F0^2 + k m v^2)

    With ``preload_force`` zero this is the unbraked transient force v sqrt(k m)
    (``calc_transient_impact_force``). Added 2026-10-20.

    Assumptions: After drive removal, a positive effective mass and approach speed compress a
    positive linear spring with a nonnegative preload magnitude.

    Excludes: Damping, residual drive torque, nonlinear contact, and safety certification.

    References
    ----------
    Derivation: Conserve initial spring energy plus kinetic energy to obtain F_peak squared = F0
    squared + k m v squared.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_coasting_contact_peak_force
    >>> from mlsysim.core.units import ureg
    >>> calc_coasting_contact_peak_force(ureg('1 m/s'), ureg('1 kg'), ureg('1 N/m'), ureg('0 N')) == ureg('1 N')
    True
    """
    validate_positive(contact_velocity, "contact_velocity")
    validate_positive(effective_mass, "effective_mass")
    validate_positive(contact_stiffness, "contact_stiffness")
    f0 = preload_force.to(ureg.newton).magnitude
    v = contact_velocity.to(ureg.meter / ureg.second).magnitude
    m = effective_mass.to(ureg.kg).magnitude
    k = contact_stiffness.to(ureg.newton / ureg.meter).magnitude
    return math.sqrt(f0**2 + k * m * v**2) * ureg.newton


def calc_passive_compliance_stiffness_bound(
    force_limit, tripwire_force, detection_speed, response_latency, arm_deceleration
):
    """Largest series stiffness that keeps a driven contact under a force limit after the tripwire.

    Parameters
    ----------
    force_limit : Quantity
        Force ceiling in N.
    tripwire_force : Quantity
        Force at tripwire detection in N.
    detection_speed : Quantity
        Positive contact approach speed in m/s.
    response_latency : Quantity
        Nonnegative delay before braking in s.
    arm_deceleration : Quantity
        Positive constant braking deceleration in m/s^2.

    Returns
    -------
    tuple of Quantity
        Stiffness bound in N/m and driven stroke in mm; nonpositive stiffness indicates no
        positive force headroom.

    Notes
    -----
    Equation:

    After the tripwire the arm keeps driving for the response latency, then stops at
    ``arm_deceleration``; the stroke it drives is s = v t_r + v^2 / (2 a), and the force
    rises by k s on top of the tripwire force:
        F_trip + k s <= F_limit  <=>  k <= (F_limit - F_trip) / (v t_r + v^2 / (2 a))

    Returns (stiffness bound, driven stroke). Added 2026-10-20.

    Assumptions: The arm keeps a positive speed during a nonnegative delay, then brakes at
    positive constant deceleration; a usable bound requires force_limit greater than
    tripwire_force.

    Excludes: Jerk ramps, damping, controller tracking error, and certification. The
    implementation returns a nonpositive stiffness when no force headroom remains.

    References
    ----------
    Derivation: Add latency travel and constant-deceleration braking travel, then divide force
    headroom by that stroke.

    Examples
    --------
    >>> from mlsysim.physics.robotics.actuation import calc_passive_compliance_stiffness_bound
    >>> from mlsysim.core.units import ureg
    >>> calc_passive_compliance_stiffness_bound(ureg('10 N'), ureg('1 N'), ureg('1 m/s'), ureg('1 s'), ureg('1 m/s**2'))[0].check('[force]/[length]')
    True
    """
    validate_positive(detection_speed, "detection_speed")
    validate_positive(arm_deceleration, "arm_deceleration")
    stroke = (detection_speed * response_latency + detection_speed**2 / (2 * arm_deceleration)).to(ureg.millimeter)
    k = ((force_limit - tripwire_force) / stroke).to(ureg.newton / ureg.meter)
    return k, stroke
