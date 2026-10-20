"""Electric-motor and actuator equations: winding heat, copper resistance,
voltage-limited speed, and the inertia match of a geared joint (added 2026-10-20).

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
    """A temperature as an absolute kelvin quantity (accepts degC offsets)."""
    return t.to(ureg.kelvin)


def calc_copper_resistance_factor(temperature, alpha, reference_temperature):
    """Winding resistance relative to its reference value: ``R(T)/R_0 = 1 + alpha (T - T_0)``."""
    return float((1 + alpha * (_k(temperature) - _k(reference_temperature))).to("").magnitude)


def calc_first_order_temperature(elapsed, start, asymptote, time_constant):
    """Lumped winding temperature after ``elapsed`` at constant power:
    ``T(t) = T_inf - (T_inf - T_0) exp(-t / tau)``."""
    x = (elapsed / time_constant).to("").magnitude
    return (_k(asymptote) - (_k(asymptote) - _k(start)) * math.exp(-x)).to(ureg.kelvin)


def calc_first_order_time_to_temperature(start, target, asymptote, time_constant):
    """Time for a first-order winding to rise from ``start`` to ``target``:
    ``t = tau ln((T_inf - T_0) / (T_inf - T_target))``; None if the asymptote never reaches it."""
    head = (_k(asymptote) - _k(target)).to(ureg.kelvin).magnitude
    if head <= 0:
        return None
    ratio = ((_k(asymptote) - _k(start)) / (_k(asymptote) - _k(target))).to("").magnitude
    return (time_constant * math.log(ratio)).to(ureg.second)


def calc_periodic_burst_peak_rise(power, thermal_resistance, on_time, period, time_constant):
    """Peak temperature rise of a first-order winding under a periodic burst at steady state:
    ``P theta (1 - e^{-t_on/tau}) / (1 - e^{-t_cycle/tau})``."""
    on = (on_time / time_constant).to("").magnitude
    cyc = (period / time_constant).to("").magnitude
    return (power * thermal_resistance * (1 - math.exp(-on)) / (1 - math.exp(-cyc))).to(ureg.kelvin)


def calc_voltage_limited_speed(torque, torque_constant, terminal_resistance, supply_voltage, no_load_speed):
    """Speed at which a brushless motor delivering ``torque`` runs out of supply voltage.

    With amplitude-invariant current i = tau / K_t, the line-to-line resistive
    drop is (sqrt(3)/2) R_TT i, and the back-EMF constant is K_e,LL = V / omega_0
    from the datasheet's no-load speed (inductance neglected):
    ``n = (V - (sqrt(3)/2) R_TT tau / K_t) / K_e,LL``.
    """
    validate_nonnegative(torque, "torque")
    validate_positive(no_load_speed, "no_load_speed")
    k_e = (supply_voltage / no_load_speed.to(ureg.radian / ureg.second)).to(ureg.volt * ureg.second / ureg.radian)
    drop = (math.sqrt(3) / 2 * terminal_resistance * (torque / torque_constant)).to(ureg.volt)
    return ((supply_voltage - drop) / k_e).to(ureg.revolution / ureg.minute)


def calc_inertia_matched_ratio(load_inertia, rotor_inertia) -> float:
    """Gear ratio that maximizes joint acceleration for a given motor torque: ``N* = sqrt(J_load / J_rotor)``."""
    validate_positive(load_inertia, "load_inertia")
    validate_positive(rotor_inertia, "rotor_inertia")
    return math.sqrt((load_inertia / rotor_inertia).to("").magnitude)


def calc_peak_normalized_response(ratio: float) -> float:
    """``2x / (1 + x^2)``: a response relative to its peak at x = 1.

    Joint acceleration against gear ratio (x = N / N*) and phase-short braking
    torque against speed (x = omega / omega_peak) both follow this curve.
    """
    return 2 * ratio / (1 + ratio ** 2)


def calc_contact_force_rate(stiffness, approach_velocity):
    """Rate at which contact force rises against a stiff surface at constant approach speed: ``dF/dt = k v``."""
    validate_positive(stiffness, "stiffness")
    validate_nonnegative(approach_velocity, "approach_velocity")
    return (stiffness * approach_velocity).to(ureg.newton / ureg.millisecond)
