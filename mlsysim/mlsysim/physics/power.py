"""Power delivery: load steps on a supply rail and their voltage droop."""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_positive

from ._units import _ensure_unit


def calc_rail_current_step(power_step, rail_voltage):
    """Current change on a supply rail for a power step: ``dI = dP / V``.

    ``power_step`` in watts, ``rail_voltage`` in volts. Returns amperes.
    """
    dp = _ensure_unit(power_step, ureg.watt, "power_step")
    v = _ensure_unit(rail_voltage, ureg.volt, "rail_voltage")
    validate_positive(v.magnitude, "rail_voltage")
    return (dp / v).to(ureg.ampere)


def calc_inductive_droop(inductance, current_step, rise_time):
    """Voltage droop across a loop inductance: ``dV = L * dI / dt``.

    The droop a supply loop of inductance ``L`` shows when its current changes
    by ``dI`` over ``dt``. The shorter the edge, the larger the droop, which is
    why on-package and on-die capacitance must answer the fastest edges.
    Returns volts.
    """
    l_loop = _ensure_unit(inductance, ureg.henry, "inductance")
    di = _ensure_unit(current_step, ureg.ampere, "current_step")
    dt = _ensure_unit(rise_time, ureg.second, "rise_time")
    validate_positive(dt.magnitude, "rise_time")
    return (l_loop * di / dt).to(ureg.volt)


# --- Dynamic CMOS power under clock scaling (added 2026-10-20) ---------------
def calc_dynamic_power_ratio(frequency_ratio, voltage_tracks_frequency=True):
    """Ratio of dynamic power after scaling the clock by ``frequency_ratio``.

    Dynamic CMOS power is ``P = C V^2 f``. When the supply voltage must rise in
    proportion to frequency (``V ~ f``), power grows with the cube of the
    frequency ratio; at fixed voltage it grows linearly. Capacitance is held
    constant. Dimensionless.
    """
    validate_positive(frequency_ratio, "frequency_ratio")
    voltage_ratio = frequency_ratio if voltage_tracks_frequency else 1.0
    return voltage_ratio ** 2 * frequency_ratio
