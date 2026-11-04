"""Heat removal by a coolant stream."""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_positive

from ._units import _ensure_unit


def calc_coolant_mass_flow(heat_load, specific_heat, temperature_rise):
    """Calculate coolant mass flow for a steady heat-removal load.

    Parameters
    ----------
    heat_load : Quantity or float
        Heat transferred into coolant, nonnegative; bare numbers mean watts.
    specific_heat : Quantity or float
        Positive coolant specific heat; bare numbers mean joule/(kilogram kelvin).
    temperature_rise : Quantity or float
        Positive coolant temperature difference; bare numbers mean kelvin.

    Returns
    -------
    Quantity
        Required coolant mass flow in kilogram/second.

    Notes
    -----
    Equation: mass_flow = heat_load/(specific_heat * temperature_rise).
    Assumptions: steady single-phase flow, constant specific heat, and complete
    transfer of the stated heat load to the coolant. The caller supplies positive
    specific heat; only temperature rise is explicitly validated here.
    Excludes: pump power, pressure drop, phase change, and imperfect heat exchange.

    References
    ----------
    Derivation: divide the steady sensible-heat balance Q_dot = m_dot c_p delta_T
    by the heat absorbed per unit mass.

    Examples
    --------
    >>> calc_coolant_mass_flow(0, 4000, 10).m_as('kg/s')
    0.0
    """
    q = _ensure_unit(heat_load, ureg.watt, "heat_load")
    cp = _ensure_unit(specific_heat, ureg.joule / (ureg.kilogram * ureg.kelvin), "specific_heat")
    dt = _ensure_unit(temperature_rise, ureg.kelvin, "temperature_rise")
    validate_positive(dt.magnitude, "temperature_rise")
    return (q / (cp * dt)).to(ureg.kilogram / ureg.second)
