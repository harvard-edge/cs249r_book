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


# --- Closed-loop evaluation: exact bounds with failures, mission counts (added 2026-10-20) ---
RULE_OF_THREE_NUMERATOR = 3.0  # -ln(0.05) = 2.996, rounded (Hanley and Lippman-Hand 1983)


def _binomial_upper_tail(successes: int, trials: int, p: float) -> float:
    """P(X >= successes) for X ~ Binomial(trials, p)."""
    return sum(math.comb(trials, j) * p**j * (1.0 - p) ** (trials - j) for j in range(successes, trials + 1))


def calc_clopper_pearson_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """
    Exact one-sided Clopper-Pearson lower confidence limit on a success probability.

    Solves P(X >= k | n, p) = 1 - C for p by bisection, X ~ Binomial(n, p). At
    k = n it reduces to (1 - C)^(1/n), the zero-failure bound.

    Source: Clopper and Pearson, "The Use of Confidence or Fiducial Limits
    Illustrated in the Case of the Binomial," Biometrika 26(4), 1934.
    """
    validate_at_least(trials, 1, "trials")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between 0 and trials.")
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _binomial_upper_tail(successes, trials, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def calc_rule_of_three_bound(n_trials: int) -> float:
    """
    The rule of three: about 3/n bounds the failure rate after n clean trials at 95 percent.

    The rounded form of ``calc_zero_failure_rate_upper_bound``; it overstates
    the exact limit slightly. Source: Hanley and Lippman-Hand, JAMA 249(13), 1983.
    """
    validate_at_least(n_trials, 1, "n_trials")
    return RULE_OF_THREE_NUMERATOR / n_trials


def calc_rule_of_three_trials(target_lower_bound: float) -> float:
    """Clean trials the rule of three asks for a success lower limit: 3 / (1 - p_target)."""
    if not (0.0 < target_lower_bound < 1.0):
        raise ValueError("target_lower_bound must be strictly between 0 and 1.")
    return RULE_OF_THREE_NUMERATOR / (1.0 - target_lower_bound)


def calc_zero_failure_mission_count(target_hazard_rate, mission_time, confidence: float = 0.95) -> float:
    """
    Failure-free missions needed to bound a constant hazard rate, counted per mission.

    Each mission of length t fails with probability lambda t (small); n clean
    missions bound it when (1 - lambda t)^n = 1 - C:

        n = ln(1 - C) / ln(1 - lambda t)

    Unrounded; at small lambda t it approaches the exposure wall divided by t.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    p_mission = (target_hazard_rate * mission_time).to("").magnitude
    if not (0.0 < p_mission < 1.0):
        raise ValueError("target_hazard_rate * mission_time must be strictly between 0 and 1.")
    return math.log(1.0 - confidence) / math.log(1.0 - p_mission)
