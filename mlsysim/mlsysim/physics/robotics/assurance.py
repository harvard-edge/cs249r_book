"""Statistical release gates, Clopper-Pearson bounds, and fleet hazard rates."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_at_least, validate_positive


def calc_empirical_testing_exposure(target_hazard_rate, confidence: float = 0.95):
    """
    Calculate required zero-failure operating exposure to statistically bound a hazard rate.

    Under a Poisson failure arrival process with constant rate lambda, the probability
    of observing zero failures over total exposure duration T is P(0) = exp(-lambda * T).
    To claim lambda <= lambda_target at statistical confidence C = 1 - alpha:
        exp(-lambda_target * T) <= 1 - C  ==>  T >= -ln(1 - C) / lambda_target

    For 95% confidence (C = 0.95):
        -ln(0.05) ≈ 2.99573  ==>  T ≈ 3.0 / lambda_target

    Source: Littlewood and Wright, "Some Conservative Stopping Rules for
    Testing Software", IEEE TSE, 1997; Kalbfleisch and Prentice, The Statistical
    Analysis of Failure Time Data, Wiley, 2002.

    Parameters
    ----------
    target_hazard_rate : Quantity or float
        Target acceptable failure rate (e.g. 1e-6 / hour or 1e-9 / hour).
        If float, interpreted as failures per hour.
    confidence : float, optional
        Statistical confidence level (default 0.95 for 95% two-sided/one-sided bound).

    Returns
    -------
    Quantity
        Required continuous operating hours.
    """
    import math

    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if hasattr(target_hazard_rate, "magnitude"):
        rate_per_hr = target_hazard_rate.to(1 / ureg.hour).magnitude
    else:
        rate_per_hr = float(target_hazard_rate)

    validate_positive(rate_per_hr, "target_hazard_rate")

    factor = -math.log(1.0 - confidence)
    hours = factor / rate_per_hr
    return hours * ureg.hour


def calc_clopper_pearson_zero_failure_bound(n_trials: int, confidence: float = 0.95) -> float:
    """
    Calculate exact Clopper-Pearson binomial lower confidence bound for zero-failure trials.

    Equation:
        p_L = (1 - C)^(1 / n)
    """
    validate_at_least(n_trials, 1, "n_trials")
    alpha = 1.0 - confidence
    return float(alpha ** (1.0 / n_trials))


def calc_zero_failure_sample_size(target_lower_bound: float, confidence: float = 0.95) -> int:
    """
    Calculate required number of consecutive zero-failure trials to certify a target reliability lower bound.

    Equation:
        n = ceil(ln(1 - C) / ln(p_L))
    """
    if not (0.0 < target_lower_bound < 1.0):
        raise ValueError("target_lower_bound must be strictly between 0 and 1.")
    alpha = 1.0 - confidence
    n = math.ceil(math.log(alpha) / math.log(target_lower_bound))
    return int(n)


def calc_shielded_system_hazard_rate(brain_hazard_rate, shield_coverage: float, shield_hw_failure_rate=None):
    """
    Calculate end-to-end hazardous event rate for a physical AI system protected by a deterministic runtime safety shield.

    Equation:
        p_sys = p_brain * (1 - c_shield) + p_hw
    """
    validate_positive(brain_hazard_rate, "brain_hazard_rate")
    if not (0.0 <= shield_coverage <= 1.0):
        raise ValueError("shield_coverage must be between 0 and 1.")

    p_brain = brain_hazard_rate.to(1.0 / ureg.hour)
    p_hw = (0.0 / ureg.hour) if shield_hw_failure_rate is None else shield_hw_failure_rate.to(1.0 / ureg.hour)

    p_sys = p_brain * (1.0 - shield_coverage) + p_hw
    return p_sys.to(1.0 / ureg.hour)


def calc_zero_failure_rate_upper_bound(n_trials: int, confidence: float = 0.95) -> float:
    """
    Upper confidence bound on a per-trial failure rate after ``n_trials`` without failure.

    Under the Poisson approximation, P(0 failures) = exp(-p n); requiring it to
    stay at or above 1 - C gives p_upper = -ln(1 - C) / n. At 95 percent
    confidence -ln(0.05) = 2.996, the "rule of three": p_upper ~ 3 / n.

    Source: Hanley and Lippman-Hand, "If Nothing Goes Wrong, Is Everything All
    Right?", JAMA 249(13), 1983 (the rule of three); added 2026-10-20.
    """
    validate_at_least(n_trials, 1, "n_trials")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return -math.log(1.0 - confidence) / n_trials
