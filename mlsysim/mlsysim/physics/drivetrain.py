"""Braking-drive physics for a wheeled mobile base (Volume IV, added 2026-09-28, M6).

A credible deceleration is admissible only if three independent ceilings
exceed it: tire-floor traction, tip-over about the leading support, and the
torque the drives can deliver at their configured current limit. A fourth
check asks whether the drive keeps current control at speed, which requires
the motor's back-EMF to stay below the bus.

Sources: Coulomb friction and the tipping moment balance are freshman
rigid-body equilibrium (OpenStax, University Physics Volume 1, Secs. 6.2 and 12.2); the
reflected rotating mass m_eff = m + sum(N^2 J / r^2) follows Lynch and Park,
Modern Robotics (2017), Sec. 8.9.2; the brushless DC terminal equation
V = e + I R is Krishnan, Permanent Magnet Synchronous and Brushless DC Motor
Drives (CRC Press, 2010).
"""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_positive, validate_nonnegative

_G = ureg.standard_gravity
_MPS2 = ureg.meter / ureg.second**2


def calc_traction_deceleration_limit(friction_coefficient: float, drive_load_share: float = 1.0):
    """Calculate the tire-friction braking ceiling.

    Parameters
    ----------
    friction_coefficient : float
        Positive dimensionless tire-floor friction coefficient mu.
    drive_load_share : float, optional
        Fraction phi of machine weight carried by braked wheels, in (0, 1].

    Returns
    -------
    Quantity
        Deceleration ceiling in meter/second squared.

    Notes
    -----
    Equation: a_traction = mu * g * phi.
    Assumptions: Coulomb friction with constant normal-load share on the braked
    wheels; unbraked casters contribute no retarding force.
    Excludes: transient load transfer, slip dynamics, surface variation, and
    traction-control behavior. The ceiling is conditional, not a safety guarantee.

    References
    ----------
    Derivation: the available horizontal force is mu times the braked-wheel
    normal load phi m g; divide by translating mass m. This is the Coulomb
    friction model stated in OpenStax, University Physics Volume 1, Section 6.2.

    Examples
    --------
    >>> a = calc_traction_deceleration_limit(0.5)
    >>> (a / ureg.standard_gravity).m_as('') == 0.5
    True
    """
    if not friction_coefficient > 0:
        raise ValueError(f"friction_coefficient must be positive; got {friction_coefficient}")
    if not 0.0 < drive_load_share <= 1.0:
        raise ValueError(f"drive_load_share must be in (0, 1]; got {drive_load_share}")
    return (friction_coefficient * drive_load_share * _G).to(_MPS2)


def calc_required_friction(deceleration, drive_load_share: float = 1.0) -> float:
    """Invert the traction ceiling to obtain required tire friction.

    Parameters
    ----------
    deceleration : Quantity
        Positive requested deceleration in meter/second squared.
    drive_load_share : float, optional
        Braked-wheel normal-load share phi, in (0, 1].

    Returns
    -------
    float
        Minimum dimensionless friction coefficient under the load-share model.

    Notes
    -----
    Equation: mu_required = a/(g * phi).
    Assumptions: the same fixed-load Coulomb friction model as
    :func:`calc_traction_deceleration_limit`.
    Excludes: drive torque, tip-over, dynamic load transfer, and uncertainty in
    surface friction. A computed coefficient does not establish available traction.

    References
    ----------
    Derivation: solve a = mu g phi for mu using the specified deceleration
    and braked-wheel load share.

    Examples
    --------
    >>> calc_required_friction(1 * ureg.standard_gravity, 1.0)
    1.0
    """
    validate_positive(deceleration, "deceleration")
    if not 0.0 < drive_load_share <= 1.0:
        raise ValueError(f"drive_load_share must be in (0, 1]; got {drive_load_share}")
    return float((deceleration / (_G * drive_load_share)).to("dimensionless").magnitude)


def calc_tipover_deceleration_limit(cg_height, tip_arm):
    """Calculate the rigid-body braking ceiling before forward tip-over.

    Parameters
    ----------
    cg_height : Quantity
        Positive center-of-gravity height h above the floor, in meters.
    tip_arm : Quantity
        Positive horizontal distance d from the center of gravity to the leading support line, in meters.

    Returns
    -------
    Quantity
        Deceleration ceiling in meter/second squared.

    Notes
    -----
    Equation: a_tip = g * d/h.
    Assumptions: planar rigid-body moment balance about the leading support line,
    with a fixed center of gravity for the arm pose being checked.
    Excludes: suspension motion, transient pitching, payload shifts, uneven floors,
    and dynamic stabilization. The result is not a tip-over safety certification.

    References
    ----------
    Derivation: balance the inertial pitching moment m a h against the restoring
    weight moment m g d. Static moment balance is described in OpenStax,
    University Physics Volume 1, Section 12.2.

    Examples
    --------
    >>> a = calc_tipover_deceleration_limit(1 * ureg.m, 1 * ureg.m)
    >>> a == 1 * ureg.standard_gravity
    True
    """
    validate_positive(cg_height, "cg_height")
    validate_positive(tip_arm, "tip_arm")
    return (_G * tip_arm / cg_height).to(_MPS2)


def calc_max_cg_height_ratio(deceleration) -> float:
    """Invert the tip-over ceiling to bound the height-to-arm ratio.

    Parameters
    ----------
    deceleration : Quantity
        Positive requested deceleration in meter/second squared.

    Returns
    -------
    float
        Largest dimensionless center-of-gravity height divided by tipping arm.

    Notes
    -----
    Equation: (h/d)_max = g/a.
    Assumptions: a fixed rigid-body center of gravity and planar static moment
    balance, as in :func:`calc_tipover_deceleration_limit`.
    Excludes: suspension and transient pitch dynamics; the ratio alone does not
    establish that a particular machine is stable.

    References
    ----------
    Derivation: rearrange a h <= g d to h/d <= g/a for positive deceleration.

    Examples
    --------
    >>> calc_max_cg_height_ratio(1 * ureg.standard_gravity)
    1.0
    """
    validate_positive(deceleration, "deceleration")
    return float((_G / deceleration).to("dimensionless").magnitude)


def calc_drive_torque_deceleration_limit(
    wheel_torque_max,
    wheel_radius,
    n_driven_wheels: int,
    mass,
    rotating_equivalent_mass=None,
):
    """Calculate the braking ceiling from available wheel torque.

    Parameters
    ----------
    wheel_torque_max : Quantity
        Positive delivered torque per braked wheel, in newton meters.
    wheel_radius : Quantity
        Positive rolling radius, in meters.
    n_driven_wheels : int
        Number of torque-producing braked wheels, at least one.
    mass : Quantity
        Positive translating mass, in kilograms.
    rotating_equivalent_mass : Quantity, optional
        Nonnegative sum of reflected rotational inertia divided by wheel radius squared, in kilograms; None means zero.

    Returns
    -------
    dict
        Force in newtons under force, effective mass in kilograms under m_eff, and deceleration in meter/second squared under a_max.

    Notes
    -----
    Equation: force = n * torque/r; m_eff = m + m_rot;
    a_max = force/m_eff, with m_rot = sum(J_reflected/r^2).
    Assumptions: equal torque and wheel radius at every driven wheel; the supplied
    torque already accounts for gearing, current limits, and transmission efficiency.
    Excludes: tire slip, aerodynamic drag, rolling resistance, and speed-dependent
    motor limits. A sourced upper bound on m_rot produces a lower torque ceiling.

    References
    ----------
    Derivation: Newton's law F = m_eff a with each wheel's torque converted
    through F = torque/r. Reflected inertia follows energy equivalence between
    rotation and translation; see Lynch and Park, Modern Robotics (2017), Section 8.9.2.

    Examples
    --------
    >>> result = calc_drive_torque_deceleration_limit(1 * ureg.N * ureg.m, 1 * ureg.m, 1, 1 * ureg.kg)
    >>> result['a_max'] == 1 * ureg.m / ureg.s**2
    True
    """
    validate_positive(wheel_torque_max, "wheel_torque_max")
    validate_positive(wheel_radius, "wheel_radius")
    validate_positive(mass, "mass")
    if n_driven_wheels < 1:
        raise ValueError(f"n_driven_wheels must be at least 1; got {n_driven_wheels}")
    m_rot = (0 * ureg.kg) if rotating_equivalent_mass is None else rotating_equivalent_mass
    validate_nonnegative(m_rot, "rotating_equivalent_mass")
    force = (n_driven_wheels * wheel_torque_max / wheel_radius).to(ureg.newton)
    m_eff = (mass + m_rot).to(ureg.kg)
    return {"force": force, "m_eff": m_eff, "a_max": (force / m_eff).to(_MPS2)}


def calc_drive_voltage_headroom(back_emf_constant, velocity, wheel_radius, bus_voltage, current, line_resistance):
    """Evaluate direct-drive motor terminal voltages during drive and braking.

    Parameters
    ----------
    back_emf_constant : Quantity
        Positive line-to-line back-EMF per shaft speed, e.g. volt/rpm.
    velocity : Quantity
        Nonnegative ground speed, in meter/second.
    wheel_radius : Quantity
        Positive rolling radius, in meters.
    bus_voltage : Quantity
        Positive available DC bus voltage, in volts.
    current : Quantity
        Nonnegative conducting phase current, in amperes.
    line_resistance : Quantity
        Positive line-to-line winding resistance, in ohms.

    Returns
    -------
    dict
        Wheel speed omega in rpm; e_bemf, v_drive, v_brake, and headroom in volts. Headroom is bus voltage minus back-EMF, not the loaded drive margin.

    Notes
    -----
    Equation: omega = v/r; e = K_e omega; V_drive = e + I R;
    V_brake = e - I R; headroom = V_bus - e.
    Assumptions: direct drive with two phases conducting and constant winding
    resistance; braking current opposes the back-EMF. If back-EMF exceeds the bus,
    inverter diode conduction can prevent controlled braking current.
    Excludes: inductive transients, inverter voltage drops, bus regeneration
    limits, and current-controller dynamics.

    References
    ----------
    Derivation: apply the brushless DC steady terminal balance V = e + I R,
    reversing current polarity for braking. Krishnan, Permanent Magnet
    Synchronous and Brushless DC Motor Drives (CRC Press, 2010), supplies the
    motor-circuit basis; the headroom metric here is the unloaded bus-minus-EMF difference.

    Examples
    --------
    >>> result = calc_drive_voltage_headroom(1 * ureg.V / ureg.rpm, 0 * ureg.m / ureg.s, 1 * ureg.m, 24 * ureg.V, 0 * ureg.A, 1 * ureg.ohm)
    >>> result['headroom'] == 24 * ureg.V
    True
    """
    validate_positive(back_emf_constant, "back_emf_constant")
    validate_nonnegative(velocity, "velocity")
    validate_positive(wheel_radius, "wheel_radius")
    validate_positive(bus_voltage, "bus_voltage")
    validate_nonnegative(current, "current")
    validate_positive(line_resistance, "line_resistance")
    omega = ((velocity / wheel_radius).to(1 / ureg.second) * ureg.radian).to(ureg.revolution / ureg.minute)
    e = (back_emf_constant * omega).to(ureg.volt)
    ir = (current * line_resistance).to(ureg.volt)
    return {
        "omega": omega,
        "e_bemf": e,
        "v_drive": (e + ir).to(ureg.volt),
        "v_brake": (e - ir).to(ureg.volt),
        "headroom": (bus_voltage - e).to(ureg.volt),
    }


def calc_braking_limits(
    mass,
    deceleration,
    *,
    wheel_torque_max,
    wheel_radius,
    n_driven_wheels: int,
    friction_coefficient: float,
    drive_load_share: float = 1.0,
    rotating_equivalent_mass=None,
    cg_height=None,
    tip_arm=None,
):
    """Compare a requested deceleration with independent braking ceilings.

    Parameters
    ----------
    mass : Quantity
        Positive translating mass in kilograms.
    deceleration : Quantity
        Positive requested deceleration in meter/second squared.
    wheel_torque_max : Quantity
        Positive available braking torque per driven wheel in newton meters.
    wheel_radius : Quantity
        Positive wheel rolling radius in meters.
    n_driven_wheels : int
        Number of driven braked wheels, at least one.
    friction_coefficient : float
        Positive dimensionless tire-floor friction coefficient.
    drive_load_share : float, optional
        Braked-wheel normal-load fraction in (0, 1].
    rotating_equivalent_mass : Quantity, optional
        Nonnegative reflected rotating mass in kilograms; None supplies zero.
    cg_height : Quantity, optional
        Positive center-of-gravity height in meters; required with tip_arm for a tip-over check.
    tip_arm : Quantity, optional
        Positive horizontal tipping arm in meters; required with cg_height.

    Returns
    -------
    dict
        Individual ceilings a_traction, a_torque, a_tip and a_admissible in meter/second squared; force in newtons; m_eff in kilograms; dimensionless mu_required and cg_ratio_max; binding label and Boolean passes.

    Notes
    -----
    Equation: a_admissible = min(a_traction, a_torque, a_tip) over the
    available ceilings. The request passes only if every available ceiling meets
    it and the tip-over geometry was supplied.
    Assumptions: the component traction, torque, and static moment models apply
    simultaneously. Missing center-of-gravity geometry makes a_tip None and
    passes False, while still reporting the available ceilings and h/d bound.
    Excludes: drive-voltage limits, reaction latency, dynamic stability, and
    uncertain friction. Passing these algebraic ceilings is not a safety certification.

    References
    ----------
    Derivation: intersect the admissible sets returned by
    :func:`calc_traction_deceleration_limit`, :func:`calc_tipover_deceleration_limit`,
    and :func:`calc_drive_torque_deceleration_limit`; the smallest ceiling binds.

    Examples
    --------
    >>> result = calc_braking_limits(1 * ureg.kg, 1 * ureg.m / ureg.s**2, wheel_torque_max=1 * ureg.N * ureg.m, wheel_radius=1 * ureg.m, n_driven_wheels=1, friction_coefficient=0.5)
    >>> result['a_tip'] is None and result['passes'] is False
    True
    """
    validate_positive(deceleration, "deceleration")
    a_trac = calc_traction_deceleration_limit(friction_coefficient, drive_load_share)
    torque = calc_drive_torque_deceleration_limit(
        wheel_torque_max, wheel_radius, n_driven_wheels, mass, rotating_equivalent_mass
    )
    a_tip = None
    if cg_height is not None and tip_arm is not None:
        a_tip = calc_tipover_deceleration_limit(cg_height, tip_arm)
    ceilings = {"traction": a_trac, "torque": torque["a_max"]}
    if a_tip is not None:
        ceilings["tip-over"] = a_tip
    binding = min(ceilings, key=lambda k: ceilings[k].to(_MPS2).magnitude)
    a_adm = ceilings[binding]
    passes = a_tip is not None and all(c >= deceleration for c in ceilings.values())
    return {
        "a_traction": a_trac,
        "a_torque": torque["a_max"],
        "a_tip": a_tip,
        "m_eff": torque["m_eff"],
        "force": torque["force"],
        "mu_required": calc_required_friction(deceleration, drive_load_share),
        "cg_ratio_max": calc_max_cg_height_ratio(deceleration),
        "a_admissible": a_adm,
        "binding": binding,
        "passes": passes,
    }
