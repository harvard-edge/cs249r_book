"""Fleet economics and cloud cost models."""

from __future__ import annotations

from mlsysim.core.units import ureg, DAYS_PER_YEAR
from mlsysim.core._validation import validate_nonnegative

from ._units import _ensure_unit


def calc_monthly_egress_cost(bytes_per_sec, cost_per_gb):
    """Calculate the monthly cloud egress cost for sustained bandwidth usage.

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
    Equation: C = bandwidth * (30 days) * price per byte.

    Assumptions: Constant transfer rate and linear pricing for a 30-day month.

    Excludes: Calendar-month length, free tiers, and tiered discounts.

    References
    ----------
    Derivation: Multiplying transferred volume by a per-volume price gives cost.

    Examples
    --------
    >>> calc_monthly_egress_cost(0, 1)
    0.0
    """
    b_s = _ensure_unit(bytes_per_sec, ureg.byte / ureg.second, "bytes_per_sec")
    validate_nonnegative(b_s, "bytes_per_sec")
    monthly_bytes = b_s * (30 * ureg.day)  # 30-day month convention (see Notes)
    cost_rate = _ensure_unit(cost_per_gb, ureg.dollar / ureg.gigabyte, "cost_per_gb")
    cost = monthly_bytes * cost_rate.to(ureg.dollar / ureg.byte)
    return cost.m_as(ureg.dollar)


def calc_fleet_tco(unit_cost, power_w, quantity, years, kwh_price):
    """Total Cost of Ownership (CapEx + energy OpEx) for a hardware fleet.

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
    Equation: TCO = N * unit_cost + N * power * duration * electricity_price.

    Assumptions: All devices draw the supplied average power throughout the duration; a year has 365 days.

    Excludes: PUE, cooling, maintenance, discounting, and amortization. Use EconomicsModel for facility accounting.

    References
    ----------
    Derivation: Capital and energy expenditures add; energy is power integrated over time.

    Examples
    --------
    >>> calc_fleet_tco(100, 0, 2, 1, 0.1)
    200.0
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
    """Calculate the time for running-cost savings to repay an up-front cost.

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

    Notes
    -----
    Equation: t = upfront_cost / annual_savings.

    Assumptions: Positive savings continue at a constant annual rate.

    Excludes: Discounting, taxes, depreciation, and changes in savings.

    References
    ----------
    Derivation: Accumulated savings equal rate times time; solve that equality for time.

    Examples
    --------
    >>> calc_payback_period(100, 50).m_as("year")
    2.0
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

    Parameters
    ----------
    upfront_cost : Quantity
        Nonnegative initial purchase cost, in currency.
    owned_cost_rate : Quantity
        Owned running cost per time, in currency/time.
    rent_cost_rate : Quantity
        Rental cost per time, in currency/time.

    Returns
    -------
    Quantity or None
        Crossover time; None if rent does not overtake ownership.

    Notes
    -----
    Equation: t = upfront_cost / (rent_cost_rate - owned_cost_rate).

    Assumptions: Both cost rates are constant and use compatible time units.

    Excludes: Discounting and variable utilization; no crossover is returned when the rate gap is nonpositive.

    References
    ----------
    Derivation: Equate upfront cost plus owned running cost to accumulated rent.

    Examples
    --------
    >>> calc_rent_crossover(100 * ureg.dollar, 2 * ureg.dollar / ureg.hour, 12 * ureg.dollar / ureg.hour).m_as("hour")
    10.0
    """
    gap = rent_cost_rate - owned_cost_rate
    if gap.magnitude <= 0:
        return None
    return upfront_cost / gap


def calc_ownership_breakeven_utilization(
    upfront_cost, fixed_cost_per_year, rent_per_year_full, years, use_cost_per_year_full=0.0
):
    """Sustained utilization at which owning and renting cost the same over ``years``.

    Parameters
    ----------
    upfront_cost : float
        Initial cost in currency.
    fixed_cost_per_year : float
        Annual fixed ownership cost in the same currency.
    rent_per_year_full : float
        Annual rental cost at full utilization.
    years : float
        Positive duration in years.
    use_cost_per_year_full : float, optional
        Annual usage-dependent cost at full utilization; zero by default.

    Returns
    -------
    float
        Dimensionless utilization at equal cumulative cost.

    Notes
    -----
    Equation: u = (C + Y * F) / (Y * (R - E)).

    Assumptions: Constant sustained utilization, positive duration, and a nonzero rent-minus-use-cost gap.

    Excludes: Discounting and capacity constraints; a result outside [0, 1] is not clipped.

    References
    ----------
    Derivation: Equate C + Y(F + uE) with uYR and solve for u.

    Examples
    --------
    >>> calc_ownership_breakeven_utilization(100, 0, 100, 2)
    0.5
    """
    return (upfront_cost + years * fixed_cost_per_year) / (years * (rent_per_year_full - use_cost_per_year_full))


# ── Training-serving skew cost (added 2026-10-20) ───────────────────────────


def calc_skew_cost(skew_rate, query_rate, cost_per_error):
    """Cost rate of training-serving skew: ``Rate_skew * Q * C_error``.

    Parameters
    ----------
    skew_rate : float
        Fraction of queries made wrong by skew, in [0, 1].
    query_rate : float or Quantity
        Nonnegative queries per unit time.
    cost_per_error : float or Quantity
        Expected currency loss per erroneous query.

    Returns
    -------
    float or Quantity
        Expected currency loss per unit time.

    Notes
    -----
    Equation: loss_rate = skew_rate * query_rate * cost_per_error.

    Assumptions: Each affected query incurs the same supplied expected cost.

    Excludes: Correlated downstream losses and nonlinear business impact.

    References
    ----------
    Derivation: Multiply query volume by error fraction and loss per error.

    Examples
    --------
    >>> calc_skew_cost(0.5, 100, 2)
    100.0
    """
    if not 0 <= skew_rate <= 1:
        raise ValueError("skew_rate must be a fraction in [0, 1].")
    return skew_rate * query_rate * cost_per_error


# ── Detection delay and the retraining interval (added 2026-10-20) ───────────


def calc_detection_delay_loss(annual_value, quality_drop, delay, year_length=None):
    """Value lost while a quality drop goes undetected: ``V_year * drop * (delay / year)``.

    Parameters
    ----------
    annual_value : float or Quantity
        Value produced in one reference year, in currency.
    quality_drop : float
        Fraction of value lost, in [0, 1].
    delay : Quantity
        Nonnegative detection delay, in time units.
    year_length : Quantity, optional
        Positive reference-year duration; defaults to DAYS_PER_YEAR days.

    Returns
    -------
    float or Quantity
        Currency value lost during detection delay.

    Notes
    -----
    Equation: loss = annual_value * quality_drop * delay / year_length.

    Assumptions: Value accrues uniformly and the quality loss is constant until detection.

    Excludes: Recovery time after detection, seasonality, and nonlinear effects of quality.

    References
    ----------
    Derivation: Prorate annual value by elapsed time and multiply by the lost fraction.

    Examples
    --------
    >>> calc_detection_delay_loss(100, 0.5, 365 * ureg.day)
    50.0
    """
    if not 0 <= quality_drop <= 1:
        raise ValueError("quality_drop must be a fraction in [0, 1].")
    year = ureg.Quantity(DAYS_PER_YEAR, "day") if year_length is None else year_length
    return annual_value * quality_drop * (delay / year).to("").magnitude


def calc_staleness_slope(query_rate, value_per_query, initial_accuracy, decay_rate):
    """Growth of value lost per unit time since the last retraining run: ``Q V A0 gamma``.

    Parameters
    ----------
    query_rate : float or Quantity
        Queries per time, nonnegative.
    value_per_query : float or Quantity
        Currency value per query.
    initial_accuracy : float
        Reference correct fraction, in [0, 1].
    decay_rate : float or Quantity
        Nonnegative exponential decay rate in inverse time.

    Returns
    -------
    float or Quantity
        Loss-rate growth, in currency/time squared.

    Notes
    -----
    Equation: s = query_rate * value_per_query * initial_accuracy * decay_rate.

    Assumptions: Exponential accuracy decay is linearized near zero age, where decay_rate * age is small.

    Excludes: Long-horizon nonlinear decay and changing query volume.

    References
    ----------
    Derivation: The first-order expansion of A0 exp(-gamma t) gives an accuracy loss A0 gamma t.

    Examples
    --------
    >>> calc_staleness_slope(100, 2, 0.5, 0.1)
    10.0
    """
    return query_rate * value_per_query * initial_accuracy * decay_rate


def calc_retraining_cost_rate(interval, run_cost, staleness_slope):
    """Cost per unit time of retraining every ``interval``: ``C / T + (1/2) s T``.

    Parameters
    ----------
    interval : float or Quantity
        Positive time between retraining runs.
    run_cost : float or Quantity
        Nonnegative currency cost per run.
    staleness_slope : float or Quantity
        Nonnegative growth of loss rate, in currency/time squared.

    Returns
    -------
    float or Quantity
        Average total cost per time.

    Notes
    -----
    Equation: rate = run_cost / interval + staleness_slope * interval / 2.

    Assumptions: Positive cycle length, constant training cost, and linear staleness growth reset after each run.

    Excludes: Training downtime, discrete scheduling, and nonlinear drift.

    References
    ----------
    Derivation: Divide run cost plus the triangular integrated staleness loss by the cycle length.

    Examples
    --------
    >>> calc_retraining_cost_rate(2, 4, 2)
    4.0
    """
    return run_cost / interval + 0.5 * staleness_slope * interval


def calc_optimal_retraining_interval(run_cost, staleness_slope):
    """Interval that minimizes ``calc_retraining_cost_rate``: ``T* = sqrt(2 C / s)``.

    Parameters
    ----------
    run_cost : float or Quantity
        Positive currency cost of one retraining run.
    staleness_slope : float or Quantity
        Positive growth of loss rate, in currency/time squared.

    Returns
    -------
    float or Quantity
        Optimal cycle length, in the time units implied by the inputs.

    Notes
    -----
    Equation: T = sqrt(2 * run_cost / staleness_slope).

    Assumptions: Positive run cost and slope; training restores the reference quality each cycle.

    Excludes: Operational scheduling constraints, downtime, and nonlinear drift.

    References
    ----------
    Derivation: Set the derivative of C/T + sT/2 to zero; the positive stationary point minimizes the cost.

    Examples
    --------
    >>> calc_optimal_retraining_interval(4, 2)
    2.0
    """
    return (2 * run_cost / staleness_slope) ** 0.5
