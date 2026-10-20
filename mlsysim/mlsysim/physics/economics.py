"""Fleet economics and cloud cost models."""

from __future__ import annotations

from mlsysim.core.units import ureg, DAYS_PER_YEAR
from mlsysim.core._validation import validate_nonnegative

from ._units import _ensure_unit


def calc_monthly_egress_cost(bytes_per_sec, cost_per_gb):
    """
    Calculates the monthly cloud egress cost for sustained bandwidth usage.

    Parameters
    ----------
    bytes_per_sec : Quantity
        The sustained continuous data transfer rate (e.g., Bytes/s).
    cost_per_gb : Quantity
        The egress pricing tier from the cloud provider (e.g., $/GB).

    Returns
    -------
    float
        The total monthly cost in USD.

    Notes
    -----
    Uses a 30-day month convention (a common cloud-billing simplification),
    so a "month" here is exactly 2,592,000 seconds — not a calendar month
    and ~1.4% short of the 365/12 average.
    """
    b_s = _ensure_unit(bytes_per_sec, ureg.byte / ureg.second, "bytes_per_sec")
    validate_nonnegative(b_s, "bytes_per_sec")
    monthly_bytes = b_s * (30 * ureg.day)  # 30-day month convention (see Notes)
    cost_rate = _ensure_unit(cost_per_gb, ureg.dollar / ureg.gigabyte, "cost_per_gb")
    cost = monthly_bytes * cost_rate.to(ureg.dollar / ureg.byte)
    return cost.m_as(ureg.dollar)


def calc_fleet_tco(unit_cost, power_w, quantity, years, kwh_price):
    """
    Total Cost of Ownership (CapEx + energy OpEx) for a hardware fleet.

    Calculates the combined capital expenditure (hardware purchase) and 
    operational expenditure (electricity) over a given lifespan.

    Parameters
    ----------
    unit_cost : Quantity
        The purchase price of a single hardware unit (e.g., USD).
    power_w : Quantity
        The average active power consumption per unit (e.g., Watts).
    quantity : int
        The total number of units in the fleet.
    years : Quantity or float
        The total operating lifespan to cost out (NOT annualized — full CapEx
        is charged up front, no straight-line amortization). Bare floats are
        years of 365 days (8,760 hours, the convention every book LEGO cell
        uses).
    kwh_price : Quantity
        The local cost of electricity (e.g., $/kWh).

    Returns
    -------
    float
        The total cost of ownership in USD.

    Notes
    -----
    Device-level napkin model: energy is the device's own draw — no PUE /
    cooling overhead, no maintenance. For facility-level TCO with PUE-loaded
    energy, straight-line amortization, and maintenance, use
    ``mlsysim.solvers.EconomicsModel``.
    """
    u_cost = _ensure_unit(unit_cost, ureg.dollar, "unit_cost")
    p_w = _ensure_unit(power_w, ureg.watt, "power_w")
    validate_nonnegative(p_w, "power_w")
    validate_nonnegative(quantity, "quantity")
    price = _ensure_unit(kwh_price, ureg.dollar / ureg.kilowatt_hour, "kwh_price")
    time = _ensure_unit(years, ureg.year, "years")
    validate_nonnegative(time, "years")
    # Explicit 365-day year (8,760 h), matching DAYS_PER_YEAR/HOURS_PER_YEAR
    # and every book call site — NOT Pint's default Julian year (8,766 h).
    hours = time.m_as(ureg.year) * float(DAYS_PER_YEAR) * 24.0 * ureg.hour
    fleet_capex = u_cost * quantity
    total_energy = p_w * quantity * hours
    power_opex = total_energy * price
    return (fleet_capex + power_opex).m_as(ureg.dollar)


def calc_payback_period(upfront_cost, annual_savings):
    """Simple payback: the time for running-cost savings to repay an up-front cost.

    ``t = C_upfront / (savings per year)``, undiscounted (added 2026-10-20).

    Parameters
    ----------
    upfront_cost : Quantity
        One-time capital cost (USD).
    annual_savings : Quantity
        Running cost avoided per year (USD/year); must be positive.

    Returns
    -------
    Quantity
        Payback period in years.
    """
    cost = _ensure_unit(upfront_cost, ureg.dollar, "upfront_cost")
    validate_nonnegative(cost, "upfront_cost")
    savings = _ensure_unit(annual_savings, ureg.dollar / ureg.year, "annual_savings")
    if savings.magnitude <= 0:
        raise ValueError("annual_savings must be positive for the cost to pay back")
    return (cost / savings).to(ureg.year)


# ── Owning against renting accelerator capacity (added 2026-10-20) ───────────


def calc_rent_crossover(upfront_cost, owned_cost_rate, rent_cost_rate):
    """Time at which cumulative rent overtakes owning: ``t = C_up / (r_rent - r_own)``.

    Owning pays ``upfront_cost`` at time zero and then ``owned_cost_rate``;
    renting pays ``rent_cost_rate`` from zero. Both rates are costs per unit
    time in the same units. Returns ``None`` when rent never overtakes (the
    owned running cost is at least the rent).
    """
    gap = rent_cost_rate - owned_cost_rate
    if gap.magnitude <= 0:
        return None
    return upfront_cost / gap


def calc_ownership_breakeven_utilization(upfront_cost, fixed_cost_per_year, rent_per_year_full,
                                         years, use_cost_per_year_full=0.0):
    """Sustained utilization at which owning and renting cost the same over ``years``.

    Owning costs ``C_up + Y (F + u E)``, where ``F`` is running cost paid
    whether or not the fleet is busy and ``E`` is running cost that scales with
    use (energy drawn only while busy), both at full use. Renting the same
    capacity costs ``u Y R``, with ``R`` the rent for every hour of a year.
    Setting them equal gives

        u* = (C_up + Y F) / (Y (R - E)).
    """
    return (upfront_cost + years * fixed_cost_per_year) / (years * (rent_per_year_full - use_cost_per_year_full))


# ── Training-serving skew cost (added 2026-10-20) ───────────────────────────


def calc_skew_cost(skew_rate, query_rate, cost_per_error):
    """Cost rate of training-serving skew: ``Rate_skew * Q * C_error``.

    ``skew_rate`` is the fraction of queries whose prediction skew makes wrong,
    ``query_rate`` the query volume per unit time, and ``cost_per_error`` the
    business cost of one wrong prediction. Returns cost per unit time.
    """
    if not 0 <= skew_rate <= 1:
        raise ValueError("skew_rate must be a fraction in [0, 1].")
    return skew_rate * query_rate * cost_per_error
