"""Power delivery: load steps on a supply rail and their voltage droop."""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_positive

from ._units import _ensure_unit


def calc_rail_current_step(power_step, rail_voltage):
    """Convert a power step into a rail-current step.

    Parameters
    ----------
    power_step : Quantity or float
        Signed load-power change; bare numbers mean watts.
    rail_voltage : Quantity or float
        Positive supply voltage; bare numbers mean volts.

    Returns
    -------
    Quantity
        Signed current change in amperes.

    Notes
    -----
    Equation: delta_I = delta_P/V.
    Assumptions: constant rail voltage during the calculation and power measured
    at that rail rather than upstream of conversion losses.
    Excludes: converter efficiency, voltage droop feedback, and transient storage.

    References
    ----------
    Derivation: subtract two operating points of P = V I at the same V.

    Examples
    --------
    >>> calc_rail_current_step(0, 12).m_as('A')
    0.0
    """
    dp = _ensure_unit(power_step, ureg.watt, "power_step")
    v = _ensure_unit(rail_voltage, ureg.volt, "rail_voltage")
    validate_positive(v.magnitude, "rail_voltage")
    return (dp / v).to(ureg.ampere)


def calc_inductive_droop(inductance, current_step, rise_time):
    """Estimate loop-inductance voltage during a linear current ramp.

    Parameters
    ----------
    inductance : Quantity or float
        Loop inductance; bare numbers mean henries; physical loop inductance is nonnegative.
    current_step : Quantity or float
        Signed current change; bare numbers mean amperes.
    rise_time : Quantity or float
        Positive edge duration; bare numbers mean seconds.

    Returns
    -------
    Quantity
        Signed inductive voltage in volts; a rising current gives positive droop.

    Notes
    -----
    Equation: delta_V = L delta_I/delta_t.
    Assumptions: constant lumped inductance and a linear current ramp.
    Excludes: resistance, decoupling capacitance, distributed parasitics, and
    converter response. The edge duration must be positive.

    References
    ----------
    Derivation: apply the inductor terminal relation V = L dI/dt to a linear ramp.

    Examples
    --------
    >>> calc_inductive_droop(1e-9, 0, 1e-6).m_as('V')
    0.0
    """
    l_loop = _ensure_unit(inductance, ureg.henry, "inductance")
    di = _ensure_unit(current_step, ureg.ampere, "current_step")
    dt = _ensure_unit(rise_time, ureg.second, "rise_time")
    validate_positive(dt.magnitude, "rise_time")
    return (l_loop * di / dt).to(ureg.volt)


# --- Dynamic CMOS power under clock scaling (added 2026-10-20) ---------------
def calc_dynamic_power_ratio(frequency_ratio, voltage_tracks_frequency=True):
    """Estimate the dynamic-power ratio after clock scaling.

    Parameters
    ----------
    frequency_ratio : float
        Positive new frequency divided by reference frequency.
    voltage_tracks_frequency : bool, optional
        If True, voltage scales by the same ratio; otherwise voltage stays fixed.

    Returns
    -------
    float
        Dimensionless new dynamic power divided by reference dynamic power.

    Notes
    -----
    Equation: P_new/P_old = (V_new/V_old)^2 (f_new/f_old), giving r^3
    when voltage tracks frequency and r at fixed voltage.
    Assumptions: identical switched capacitance and activity factor; voltage
    proportional to frequency is an optional modeling assumption.
    Excludes: leakage, clock gating changes, transition energy, and physical
    voltage-frequency operating limits.

    References
    ----------
    Derivation: take the ratio of the capacitive switching-power relation
    P = activity * capacitance * V^2 * f at two operating points.

    Examples
    --------
    >>> calc_dynamic_power_ratio(1.0)
    1.0
    >>> calc_dynamic_power_ratio(2.0, voltage_tracks_frequency=False) == 2.0
    True
    """
    validate_positive(frequency_ratio, "frequency_ratio")
    voltage_ratio = frequency_ratio if voltage_tracks_frequency else 1.0
    return voltage_ratio**2 * frequency_ratio
