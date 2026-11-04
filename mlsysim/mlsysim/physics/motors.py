"""Model electric-motor heat, voltage limits, and geared-joint inertia.

The equations cover winding heat, copper resistance, voltage-limited speed,
and the inertia match of a geared joint.

Sources: the first-order lumped thermal model and copper's linear resistance
coefficient are standard (Incropera and DeWitt, Fundamentals of Heat and Mass
Transfer, lumped capacitance; NBS Handbook 100, Copper Wire Tables); the
inertia match N* = sqrt(J_load / J_rotor) and the brushless terminal equation
follow Lynch and Park, Modern Robotics (2017), Sec. 8.9, and Krishnan,
Permanent Magnet Synchronous and Brushless DC Motor Drives (CRC Press, 2010).
"""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_positive, validate_nonnegative


def _k(t):
    """Convert a temperature to absolute kelvin, accepting degC offsets."""
    return t.to(ureg.kelvin)


def calc_copper_resistance_factor(temperature, alpha, reference_temperature):
    """Calculate winding resistance relative to its reference value.

    Parameters
    ----------
    temperature : Quantity
        Winding temperature, convertible to absolute kelvin.
    alpha : Quantity
        Linear temperature coefficient, in inverse kelvin.
    reference_temperature : Quantity
        Reference temperature of the resistance measurement, convertible to kelvin.

    Returns
    -------
    float
        Dimensionless resistance ratio R(T)/R(T_0).

    Notes
    -----
    Equation: R(T)/R(T_0) = 1 + alpha * (T - T_0).
    Assumptions: alpha is constant over the temperature interval and the reference
    resistance is positive. Offset temperatures are converted to kelvin first.
    Excludes: nonlinear resistivity, self-heating feedback, and frequency effects.
    The caller must keep the linear approximation in its physical regime.

    References
    ----------
    Derivation: first-order expansion of resistivity about the reference
    temperature, with alpha = (1/R_0) dR/dT at that temperature.

    Examples
    --------
    >>> calc_copper_resistance_factor(300 * ureg.K, 0.004 / ureg.K, 300 * ureg.K)
    1.0
    """
    return float((1 + alpha * (_k(temperature) - _k(reference_temperature))).to("").magnitude)


def calc_first_order_temperature(elapsed, start, asymptote, time_constant):
    """Evolve a lumped winding temperature under constant heating.

    Parameters
    ----------
    elapsed : Quantity
        Elapsed time, nonnegative, in seconds or compatible units.
    start : Quantity
        Initial absolute temperature, convertible to kelvin.
    asymptote : Quantity
        Constant equilibrium temperature, convertible to kelvin.
    time_constant : Quantity
        Positive thermal time constant, in seconds or compatible units.

    Returns
    -------
    Quantity
        Absolute temperature in kelvin.

    Notes
    -----
    Equation: T(t) = T_inf - (T_inf - T_0) exp(-t/tau).
    Assumptions: one uniform thermal state, constant thermal resistance and
    capacitance, and a constant heat input during the interval.
    Excludes: spatial gradients, changing cooling conditions, and temperature-dependent
    losses. Valid time and time-constant ranges are caller obligations.

    References
    ----------
    Derivation: solve the lumped heat balance dT/dt = (T_inf - T)/tau with
    initial condition T(0) = T_0.

    Examples
    --------
    >>> calc_first_order_temperature(0 * ureg.s, 300 * ureg.K, 350 * ureg.K, 10 * ureg.s) == 300 * ureg.K
    True
    """
    x = (elapsed / time_constant).to("").magnitude
    return (_k(asymptote) - (_k(asymptote) - _k(start)) * math.exp(-x)).to(ureg.kelvin)


def calc_first_order_time_to_temperature(start, target, asymptote, time_constant):
    """Invert the first-order heating or cooling trajectory.

    Parameters
    ----------
    start : Quantity
        Initial absolute temperature, convertible to kelvin.
    target : Quantity
        Target temperature between the initial state and asymptote; convertible to kelvin.
    asymptote : Quantity
        Constant equilibrium temperature, convertible to kelvin.
    time_constant : Quantity
        Positive thermal time constant, in seconds or compatible units.

    Returns
    -------
    Quantity or None
        Time in seconds, or None for the unreachable heating branch.

    Notes
    -----
    Equation: t = tau * ln((T_inf - T_0)/(T_inf - T_target)); the cooling
    branch uses the equivalent ratio (T_0 - T_inf)/(T_target - T_inf).
    Assumptions: constant forcing, a positive time constant, and a target on the
    trajectory between start and equilibrium. Cooling requires a target above the
    asymptote; passing the asymptote exactly can divide by zero.
    Excludes: varying heat input and validation of every out-of-regime temperature
    ordering. None is not a general reachability certificate for arbitrary inputs.

    References
    ----------
    Derivation: rearrange the exponential solution of
    :func:`calc_first_order_temperature` and take its natural logarithm.

    Examples
    --------
    >>> calc_first_order_time_to_temperature(300 * ureg.K, 300 * ureg.K, 350 * ureg.K, 10 * ureg.s).m_as('s')
    0.0
    """
    head = (_k(asymptote) - _k(target)).to(ureg.kelvin).magnitude
    if head < 0 and _k(target) <= _k(start):  # cooling toward a lower asymptote
        ratio = ((_k(start) - _k(asymptote)) / (_k(target) - _k(asymptote))).to("").magnitude
        return (time_constant * math.log(ratio)).to(ureg.second)
    if head <= 0:
        return None
    ratio = ((_k(asymptote) - _k(start)) / (_k(asymptote) - _k(target))).to("").magnitude
    return (time_constant * math.log(ratio)).to(ureg.second)


def calc_periodic_burst_peak_rise(power, thermal_resistance, on_time, period, time_constant):
    """Calculate steady-cycle peak temperature rise under periodic heating.

    Parameters
    ----------
    power : Quantity
        Nonnegative heat dissipation during the on interval, in watts.
    thermal_resistance : Quantity
        Positive thermal resistance to ambient, in kelvin/watt.
    on_time : Quantity
        On duration, from zero to period, in seconds.
    period : Quantity
        Positive cycle duration, in seconds.
    time_constant : Quantity
        Positive lumped thermal time constant, in seconds.

    Returns
    -------
    Quantity
        Peak rise above ambient in kelvin after many identical cycles.

    Notes
    -----
    Equation: delta_T_peak = P theta * (1 - exp(-t_on/tau)) /
    (1 - exp(-period/tau)).
    Assumptions: periodic square-wave heat input, zero dissipation in the off
    interval, and constant thermal properties; input ranges are caller obligations.
    Excludes: transient startup cycles, changing ambient temperature, and separate
    thermal states for windings, housing, and coolant.

    References
    ----------
    Derivation: equate the temperature at the start of successive heating and
    cooling cycles using the first-order exponential solution.

    Examples
    --------
    >>> rise = calc_periodic_burst_peak_rise(10 * ureg.W, 2 * ureg.K / ureg.W, 5 * ureg.s, 5 * ureg.s, 10 * ureg.s)
    >>> round(rise.m_as('K'), 8) == (10 * ureg.W * 2 * ureg.K / ureg.W).m_as('K')
    True
    """
    on = (on_time / time_constant).to("").magnitude
    cyc = (period / time_constant).to("").magnitude
    return (power * thermal_resistance * (1 - math.exp(-on)) / (1 - math.exp(-cyc))).to(ureg.kelvin)


def calc_voltage_limited_speed(torque, torque_constant, terminal_resistance, supply_voltage, no_load_speed):
    """Estimate brushless-motor speed at the available terminal voltage.

    Parameters
    ----------
    torque : Quantity
        Requested nonnegative output torque, in newton meters.
    torque_constant : Quantity
        Positive torque per amplitude-invariant phase current, in newton meters/ampere.
    terminal_resistance : Quantity
        Nonnegative line-to-line winding resistance, in ohms.
    supply_voltage : Quantity
        Positive DC supply voltage, in volts.
    no_load_speed : Quantity
        Positive measured no-load angular speed at the supply voltage, in rpm or compatible units.

    Returns
    -------
    Quantity
        Voltage-limited speed in revolutions/minute; can be negative if the resistive drop exceeds supply.

    Notes
    -----
    Equation: K_e,LL = V/omega_0 and omega =
    [V - (sqrt(3)/2) R_TT torque/K_t] / K_e,LL.
    Assumptions: amplitude-invariant sinusoidal phase current and a constant motor
    resistance; no-load speed supplies the line-to-line back-EMF coefficient.
    Excludes: inductive transients, inverter modulation limits, and saturation.
    A negative result indicates an infeasible requested operating point; it is
    not clipped into a feasible motor speed.

    References
    ----------
    Derivation: substitute phase current i = torque/K_t into the steady motor
    terminal balance V = K_e,LL omega + (sqrt(3)/2) R_TT i. The current convention
    must match the supplied torque constant.

    Examples
    --------
    >>> speed = calc_voltage_limited_speed(0 * ureg.N * ureg.m, 1 * ureg.N * ureg.m / ureg.A, 1 * ureg.ohm, 24 * ureg.V, 100 * ureg.rpm)
    >>> round(speed.m_as('rpm'), 8) == 100
    True
    """
    validate_nonnegative(torque, "torque")
    validate_positive(no_load_speed, "no_load_speed")
    k_e = (supply_voltage / no_load_speed.to(ureg.radian / ureg.second)).to(ureg.volt * ureg.second / ureg.radian)
    drop = (math.sqrt(3) / 2 * terminal_resistance * (torque / torque_constant)).to(ureg.volt)
    return ((supply_voltage - drop) / k_e).to(ureg.revolution / ureg.minute)


def calc_inertia_matched_ratio(load_inertia, rotor_inertia) -> float:
    """Find the ideal gear ratio that maximizes joint acceleration.

    Parameters
    ----------
    load_inertia : Quantity
        Positive load inertia at the output shaft, in kilogram meter squared.
    rotor_inertia : Quantity
        Positive rotor inertia at the motor shaft, in kilogram meter squared.

    Returns
    -------
    float
        Dimensionless ideal reduction ratio; no minimum gear ratio is imposed.

    Notes
    -----
    Equation: N_star = sqrt(J_load/J_rotor).
    Assumptions: constant motor torque and an ideal lossless transmission, with
    acceleration proportional to N/(J_load + N^2 J_rotor).
    Excludes: gear inertia, friction, motor speed limits, efficiency, and discrete
    available gear ratios.

    References
    ----------
    Derivation: differentiate N/(J_load + N^2 J_rotor) with respect to N;
    the stationary point satisfies N^2 J_rotor = J_load.

    Examples
    --------
    >>> calc_inertia_matched_ratio(1 * ureg.kg * ureg.m**2, 1 * ureg.kg * ureg.m**2)
    1.0
    """
    validate_positive(load_inertia, "load_inertia")
    validate_positive(rotor_inertia, "rotor_inertia")
    return math.sqrt((load_inertia / rotor_inertia).to("").magnitude)


def calc_peak_normalized_response(ratio: float) -> float:
    """Evaluate a response normalized to its peak at unit ratio.

    Parameters
    ----------
    ratio : float
        Dimensionless operating ratio x, normally nonnegative; e.g. gear ratio divided by its optimum.

    Returns
    -------
    float
        Dimensionless relative response; signed inputs produce signed outputs.

    Notes
    -----
    Equation: f(x) = 2x/(1 + x^2).
    Assumptions: the mechanism has the same linear-over-quadratic response shape;
    gear acceleration and ideal phase-short braking are possible applications.
    Excludes: mechanism-specific limits and parameter identification. This helper
    does not establish that a particular motor follows the assumed curve.

    References
    ----------
    Derivation: normalize x/(1 + x^2) by its value at x = 1; the denominator
    stays positive for finite real x.

    Examples
    --------
    >>> calc_peak_normalized_response(1.0)
    1.0
    """
    return 2 * ratio / (1 + ratio**2)


def calc_contact_force_rate(stiffness, approach_velocity):
    """Calculate contact force buildup against a linear elastic surface.

    Parameters
    ----------
    stiffness : Quantity
        Positive contact stiffness, in newton/meter.
    approach_velocity : Quantity
        Nonnegative penetration speed, in meter/second.

    Returns
    -------
    Quantity
        Force increase rate in newton/millisecond.

    Notes
    -----
    Equation: dF/dt = k v.
    Assumptions: Hookean contact F = k x and constant positive penetration speed.
    Excludes: damping, finite-mass impact dynamics, plastic deformation, and force
    controller response. This rate is not a contact-safety certification.

    References
    ----------
    Derivation: differentiate F = k x with constant stiffness and dx/dt = v.

    Examples
    --------
    >>> calc_contact_force_rate(1000 * ureg.N / ureg.m, 0 * ureg.m / ureg.s).m_as('N/ms')
    0.0
    """
    validate_positive(stiffness, "stiffness")
    validate_nonnegative(approach_velocity, "approach_velocity")
    return (stiffness * approach_velocity).to(ureg.newton / ureg.millisecond)
