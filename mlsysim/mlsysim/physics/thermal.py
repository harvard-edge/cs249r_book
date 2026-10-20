"""Heat removal by a coolant stream."""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_positive

from ._units import _ensure_unit


def calc_coolant_mass_flow(heat_load, specific_heat, temperature_rise):
    """Coolant mass flow that carries a heat load: ``m_dot = Q / (c_p * dT)``.

    In steady state every watt a rack draws leaves as heat, so the coolant must
    carry ``Q`` with a temperature rise ``dT``. Returns kg/s.
    """
    q = _ensure_unit(heat_load, ureg.watt, "heat_load")
    cp = _ensure_unit(specific_heat, ureg.joule / (ureg.kilogram * ureg.kelvin), "specific_heat")
    dt = _ensure_unit(temperature_rise, ureg.kelvin, "temperature_rise")
    validate_positive(dt.magnitude, "temperature_rise")
    return (q / (cp * dt)).to(ureg.kilogram / ureg.second)
