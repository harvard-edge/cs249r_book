"""Fleet economics and cloud cost models."""

from __future__ import annotations

from mlsysim.core.constants import ureg

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
    years : Quantity
        The amortization or operational lifespan (e.g., years).
    kwh_price : Quantity
        The local cost of electricity (e.g., $/kWh).

    Returns
    -------
    float
        The total cost of ownership in USD.
    """
    u_cost = _ensure_unit(unit_cost, ureg.dollar, "unit_cost")
    p_w = _ensure_unit(power_w, ureg.watt, "power_w")
    price = _ensure_unit(kwh_price, ureg.dollar / ureg.kilowatt_hour, "kwh_price")
    time = _ensure_unit(years, ureg.year, "years")
    fleet_capex = u_cost * quantity
    total_energy = p_w * quantity * time
    power_opex = total_energy * price
    return (fleet_capex + power_opex).m_as(ureg.dollar)
