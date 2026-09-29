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
    """
    Deceleration ceiling set by tire-floor friction on the braked wheels.

    Equation:
        a_traction = mu * g * phi

    where phi is the fraction of the machine's weight carried by the braked
    (driven) wheels. Unbraked casters carry the rest and contribute no
    retarding force.

    Source: Coulomb friction F <= mu N (OpenStax, University Physics Volume 1,
    Sec. 6.2, Eq. 6.2).

    Parameters
    ----------
    friction_coefficient : float
        Tire-floor friction coefficient mu (dimensionless, > 0).
    drive_load_share : float
        Fraction phi of the weight on the braked wheels, in (0, 1].

    Returns
    -------
    Quantity
        Deceleration ceiling in m/s^2.
    """
    if not friction_coefficient > 0:
        raise ValueError(f"friction_coefficient must be positive; got {friction_coefficient}")
    if not 0.0 < drive_load_share <= 1.0:
        raise ValueError(f"drive_load_share must be in (0, 1]; got {drive_load_share}")
    return (friction_coefficient * drive_load_share * _G).to(_MPS2)


def calc_required_friction(deceleration, drive_load_share: float = 1.0) -> float:
    """
    Minimum friction coefficient that supports a deceleration on the braked wheels.

    Equation:
        mu_min = a / (g * phi)

    Source: inverse of calc_traction_deceleration_limit.
    """
    validate_positive(deceleration, "deceleration")
    if not 0.0 < drive_load_share <= 1.0:
        raise ValueError(f"drive_load_share must be in (0, 1]; got {drive_load_share}")
    return float((deceleration / (_G * drive_load_share)).to("dimensionless").magnitude)


def calc_tipover_deceleration_limit(cg_height, tip_arm):
    """
    Deceleration at which a braking machine starts to pitch over its leading support.

    The retarding force acts at the floor while inertia acts at the center of
    gravity, so braking pitches the body forward. It stays on the ground while
    the inertial moment m a h stays below the restoring moment m g d.

    Equation:
        a_tip = g * d / h_cg

    Source: moment balance about the tipping edge for a rigid body in static
    equilibrium (OpenStax, University Physics Volume 1, Sec. 12.2).

    Parameters
    ----------
    cg_height : Quantity
        Height h_cg of the center of gravity above the floor (m).
    tip_arm : Quantity
        Horizontal distance d from the center of gravity to the leading
        support line (m), for the arm pose being checked.

    Returns
    -------
    Quantity
        Deceleration ceiling in m/s^2.
    """
    validate_positive(cg_height, "cg_height")
    validate_positive(tip_arm, "tip_arm")
    return (_G * tip_arm / cg_height).to(_MPS2)


def calc_max_cg_height_ratio(deceleration) -> float:
    """
    Largest ratio h_cg / d at which a deceleration does not tip the machine.

    Equation:
        (h_cg / d)_max = g / a

    Source: inverse of calc_tipover_deceleration_limit.
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
    """
    Deceleration ceiling set by the torque the drives deliver at their current limit.

    Equation:
        m_eff = m + sum_i J_i / r^2        (per wheel, reflected through N^2 if geared)
        a_torque = n * tau_wheel,max / (r * m_eff)

    with tau_wheel,max = eta * N * K_t * I_max for a geared drive, or the hub
    motor's torque at the current limit for direct drive (N = 1).

    Source: Newton's second law with the rotating parts reflected into the
    translational mass (Lynch and Park, Modern Robotics, 2017, Sec. 8.9.2).

    Parameters
    ----------
    wheel_torque_max : Quantity
        Torque each driven wheel delivers at the configured current limit (N m).
    wheel_radius : Quantity
        Rolling radius r (m).
    n_driven_wheels : int
        Number of driven (braked) wheels.
    mass : Quantity
        Translating mass m (kg).
    rotating_equivalent_mass : Quantity, optional
        Sum of N^2 J / r^2 over all rotating drive parts (kg). A sourced upper
        bound gives a lower bound on the deceleration.

    Returns
    -------
    dict
        - "force": total retarding force at the floor (N)
        - "m_eff": effective translational mass (kg)
        - "a_max": deceleration ceiling (m/s^2)
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
    """
    Terminal voltage a direct-drive wheel motor needs at speed, braking and driving.

    With two phases conducting, the line-to-line terminal equation is
    V = e + I R when driving and V = e - I R when braking (current opposes
    the back-EMF). The drive keeps current control while the voltage it must
    impose stays within the bus; if e itself exceeds the bus, the inverter's
    diodes conduct and braking current is no longer controlled.

    Equation:
        omega = v / r ;  e = K_e * omega
        V_drive = e + I R ;  V_brake = e - I R ;  headroom = V_bus - e

    Source: brushless DC motor terminal equation (Krishnan, Permanent Magnet
    Synchronous and Brushless DC Motor Drives, CRC Press, 2010).

    Parameters
    ----------
    back_emf_constant : Quantity
        Line-to-line back-EMF per unit shaft speed (e.g. V/rpm).
    velocity : Quantity
        Ground speed (m/s).
    wheel_radius : Quantity
        Rolling radius (m).
    bus_voltage : Quantity
        DC bus voltage available to the drive (V).
    current : Quantity
        Phase current at which the check is made (A).
    line_resistance : Quantity
        Line-to-line winding resistance (ohm).

    Returns
    -------
    dict
        - "omega": wheel speed (rpm)
        - "e_bemf": back-EMF (V)
        - "v_drive": terminal voltage to drive at this current (V)
        - "v_brake": terminal voltage to brake at this current (V)
        - "headroom": bus voltage minus back-EMF (V)
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
    """
    Check a credible deceleration against the traction, tip-over, and drive-torque ceilings.

    Each ceiling is computed by its own helper; the admissible deceleration is
    the smallest ceiling that could be computed. When the center-of-gravity
    geometry is unknown the tip-over ceiling is None and the result reports
    instead the largest h_cg / d the requested deceleration tolerates.

    Source: see calc_traction_deceleration_limit, calc_tipover_deceleration_limit,
    and calc_drive_torque_deceleration_limit.

    Returns
    -------
    dict
        - "a_traction", "a_torque", "a_tip" (None if geometry unknown), "m_eff",
          "force", "mu_required", "cg_ratio_max", "a_admissible", "binding",
          and "passes" (True only if every computed ceiling meets the request
          and the tip-over ceiling was computed).
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
