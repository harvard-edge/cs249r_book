"""Statistical release gates, Clopper-Pearson bounds, and fleet hazard rates."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_at_least, validate_positive


def calc_empirical_testing_exposure(target_hazard_rate, confidence: float = 0.95):
    """Compute zero-event exposure for a one-sided Poisson rate bound.

    Parameters
    ----------
    target_hazard_rate : Quantity or float
        Positive target event rate; plain floats mean events per hour.
    confidence : float, optional
        One-sided confidence in (0, 1), default 0.95.

    Returns
    -------
    Quantity
        Required modeled failure-free operating exposure in hours.

    Notes
    -----
    Equation: exposure = -log(1 - confidence) / target_hazard_rate.

    Assumptions: A homogeneous Poisson process and complete event observation over
    accumulated exposure.

    Excludes: Nonstationary hazards, dependent exposure, unobserved failures, and safety
    certification.

    Under a Poisson failure arrival process with constant rate lambda, the probability
    of observing zero failures over total exposure duration T is P(0) = exp(-lambda * T).
    To claim lambda <= lambda_target at statistical confidence C = 1 - alpha:
        exp(-lambda_target * T) <= 1 - C  ==>  T >= -ln(1 - C) / lambda_target

    For 95% confidence (C = 0.95):
        -ln(0.05) ≈ 2.99573  ==>  T ≈ 3.0 / lambda_target

    Source: Littlewood and Wright, "Some Conservative Stopping Rules for
    Testing Software", IEEE TSE, 1997; Kalbfleisch and Prentice, The Statistical
    Analysis of Failure Time Data, Wiley, 2002.

    References
    ----------
    Derivation: Invert the zero-count likelihood ``exp(-rate*exposure)`` at the one-sided tail
    probability. Existing Littlewood-Wright and Kalbfleisch-Prentice citations give
    reliability-testing context.

    Examples
    --------
    >>> from mlsysim.physics.robotics.assurance import calc_empirical_testing_exposure
    >>> from mlsysim.core.units import ureg
    >>> a = calc_empirical_testing_exposure(1 / ureg.hour)
    >>> b = calc_empirical_testing_exposure(2 / ureg.hour)
    >>> assert a == 2 * b
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
    """Compute a one-sided success-probability bound after no failures.

    Parameters
    ----------
    n_trials : int
        Positive number of independent successful Bernoulli trials.
    confidence : float, optional
        One-sided confidence in (0, 1), default 0.95; caller must supply a valid
        probability.

    Returns
    -------
    float
        Dimensionless lower confidence limit on success probability.

    Notes
    -----
    Equation: success_lower = (1 - ``confidence)**(1`` / n_trials).

    Assumptions: Independent trials with a common success probability and no observed
    failures.

    Excludes: Distribution shift, trial dependence, and a deterministic reliability
    guarantee.

    Equation:
        p_L = (1 - C)^(1 / n)

    References
    ----------
    Derivation: For all-success binomial trials P(X=n) = ``p**n``; invert this tail at
    1-confidence, the Clopper-Pearson endpoint.

    Examples
    --------
    >>> from mlsysim.physics.robotics.assurance import calc_clopper_pearson_zero_failure_bound
    >>> from mlsysim.core.units import ureg
    >>> low = calc_clopper_pearson_zero_failure_bound(10)
    >>> high = calc_clopper_pearson_zero_failure_bound(20)
    >>> assert 0 < low < high < 1
    """
    validate_at_least(n_trials, 1, "n_trials")
    alpha = 1.0 - confidence
    return float(alpha ** (1.0 / n_trials))


def calc_zero_failure_sample_size(target_lower_bound: float, confidence: float = 0.95) -> int:
    """Size an all-success sample for a binomial lower confidence limit.

    Parameters
    ----------
    target_lower_bound : float
        Desired success-probability lower limit in (0, 1).
    confidence : float, optional
        One-sided confidence in (0, 1), default 0.95; caller must supply a valid
        probability.

    Returns
    -------
    int
        Minimum integer count of failure-free trials under the model.

    Notes
    -----
    Equation: n = ceil(log(1-confidence) / log(target_lower_bound)).

    Assumptions: Independent identically distributed Bernoulli trials, all successful.

    Excludes: Trials with failures, heterogeneous hazards, and certification of future
    operation.

    Equation:
        n = ceil(ln(1 - C) / ln(p_L))

    References
    ----------
    Derivation: Invert the all-success binomial endpoint p_lower = ``(1-confidence)**(1/n)``,
    then round upward to a trial count.

    Examples
    --------
    >>> from mlsysim.physics.robotics.assurance import calc_zero_failure_sample_size
    >>> from mlsysim.core.units import ureg
    >>> n = calc_zero_failure_sample_size(0.9)
    >>> assert isinstance(n, int) and n > 0
    """
    if not (0.0 < target_lower_bound < 1.0):
        raise ValueError("target_lower_bound must be strictly between 0 and 1.")
    alpha = 1.0 - confidence
    n = math.ceil(math.log(alpha) / math.log(target_lower_bound))
    return int(n)


def calc_shielded_system_hazard_rate(brain_hazard_rate, shield_coverage: float, shield_hw_failure_rate=None):
    """Combine residual brain hazards with a shield failure rate.

    Parameters
    ----------
    brain_hazard_rate : Quantity
        Positive unshielded event rate in inverse time.
    shield_coverage : float
        Fraction of brain hazards intercepted, in [0, 1].
    shield_hw_failure_rate : Quantity, optional
        Nonnegative additive hardware hazard rate; None means zero.

    Returns
    -------
    Quantity
        Modeled total hazardous-event rate per hour.

    Notes
    -----
    Equation: system_rate = ``brain_rate*(1-shield_coverage)`` + hardware_rate.

    Assumptions: Coverage is a stable fraction of hazardous events and the hardware
    contribution is additive.

    Excludes: Common-cause failures, dependence, hazard severity, and proof of shield
    coverage.

    Equation:
        p_sys = p_brain * (1 - c_shield) + p_hw

    References
    ----------
    Derivation: Partition brain events into intercepted and residual events, then add the
    stipulated hardware-event stream.

    Examples
    --------
    >>> from mlsysim.physics.robotics.assurance import calc_shielded_system_hazard_rate
    >>> from mlsysim.core.units import ureg
    >>> rate = calc_shielded_system_hazard_rate(1 / ureg.hour, 1.0)
    >>> assert rate.magnitude == 0
    """
    validate_positive(brain_hazard_rate, "brain_hazard_rate")
    if not (0.0 <= shield_coverage <= 1.0):
        raise ValueError("shield_coverage must be between 0 and 1.")

    p_brain = brain_hazard_rate.to(1.0 / ureg.hour)
    p_hw = (0.0 / ureg.hour) if shield_hw_failure_rate is None else shield_hw_failure_rate.to(1.0 / ureg.hour)

    p_sys = p_brain * (1.0 - shield_coverage) + p_hw
    return p_sys.to(1.0 / ureg.hour)


def calc_zero_failure_rate_upper_bound(n_trials: int, confidence: float = 0.95) -> float:
    """Compute a Poisson upper event-rate estimate per failure-free trial.

    Parameters
    ----------
    n_trials : int
        Positive number of independent failure-free trials.
    confidence : float, optional
        One-sided confidence in (0, 1), default 0.95.

    Returns
    -------
    float
        Dimensionless Poisson approximation per trial; not clipped to a Bernoulli
        probability of one.

    Notes
    -----
    Equation: upper = -log(1-confidence) / n_trials.

    Assumptions: Rare events modeled as Poisson counts over equal trial exposures.

    Excludes: Exact binomial bounds, event dependence, and distribution shift.

    Under the Poisson approximation, P(0 failures) = exp(-p n); requiring it to
    stay at or above 1 - C gives p_upper = -ln(1 - C) / n. At 95 percent
    confidence -ln(0.05) = 2.996, the "rule of three": p_upper ~ 3 / n.

    Source: Hanley and Lippman-Hand, "If Nothing Goes Wrong, Is Everything All
    Right?", JAMA 249(13), 1983 (the rule of three); added 2026-10-20.

    References
    ----------
    Derivation: Invert P(0) = ``exp(-rate*n_trials)``; Hanley and Lippman-Hand (1983) motivate
    the rounded rule-of-three approximation.

    Examples
    --------
    >>> from mlsysim.physics.robotics.assurance import calc_zero_failure_rate_upper_bound
    >>> from mlsysim.core.units import ureg
    >>> a = calc_zero_failure_rate_upper_bound(10)
    >>> b = calc_zero_failure_rate_upper_bound(20)
    >>> assert a == 2 * b
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
    """Invert a binomial tail for a one-sided success lower limit.

    Parameters
    ----------
    successes : int
        Observed successes from zero through trials inclusive.
    trials : int
        Positive total Bernoulli-trial count.
    confidence : float, optional
        One-sided confidence in (0, 1), default 0.95; caller must supply a valid
        probability.

    Returns
    -------
    float
        Lower success-probability limit; zero when no successes were observed.

    Notes
    -----
    Equation: Find p with sum(comb(n,``j)*p**j*(1-p)**(n-j)``, j=k..n) = 1-confidence.

    Assumptions: Independent trials with one common probability; the numerical solve uses
    fixed bisection iterations.

    Excludes: Dependence, changing success rates, numerical guarantees for arbitrarily
    large counts, and safety certification.

    Solves P(X >= k | n, p) = 1 - C for p by bisection, X ~ Binomial(n, p). At
    k = n it reduces to (1 - C)^(1/n), the zero-failure bound.

    Source: Clopper and Pearson, "The Use of Confidence or Fiducial Limits
    Illustrated in the Case of the Binomial," Biometrika 26(4), 1934.

    References
    ----------
    Derivation: Invert the binomial upper tail as in Clopper and Pearson, Biometrika
    26(4), 1934; this is a one-sided interval endpoint.

    Examples
    --------
    >>> from mlsysim.physics.robotics.assurance import calc_clopper_pearson_lower_bound
    >>> from mlsysim.core.units import ureg
    >>> assert calc_clopper_pearson_lower_bound(0, 10) == 0.0
    >>> assert 0 < calc_clopper_pearson_lower_bound(8, 10) < 0.8
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
    """Approximate a failure upper limit after failure-free trials.

    Parameters
    ----------
    n_trials : int
        Positive count of independent failure-free trials.

    Returns
    -------
    float
        Rounded Poisson upper limit 3/n, approximately at 95 percent one-sided confidence;
        not clipped to one.

    Notes
    -----
    Equation: upper = 3 / n_trials.

    Assumptions: Rare independent failures with equal trial exposure; the numerator rounds
    -log(0.05).

    Excludes: Exact binomial inference, small-sample probability clipping, and changing
    hazards.

    The rounded form of ``calc_zero_failure_rate_upper_bound``; it overstates
    the exact limit slightly. Source: Hanley and Lippman-Hand, JAMA 249(13), 1983.

    References
    ----------
    Derivation: Round the zero-event Poisson numerator -log(0.05) to three; Hanley and
    Lippman-Hand, JAMA 249(13), 1983.

    Examples
    --------
    >>> from mlsysim.physics.robotics.assurance import calc_rule_of_three_bound
    >>> from mlsysim.core.units import ureg
    >>> assert calc_rule_of_three_bound(30) == 3 / 30
    """
    validate_at_least(n_trials, 1, "n_trials")
    return RULE_OF_THREE_NUMERATOR / n_trials


def calc_rule_of_three_trials(target_lower_bound: float) -> float:
    """Invert the rounded rule of three for a success target.

    Parameters
    ----------
    target_lower_bound : float
        Desired lower success-probability limit in (0, 1).

    Returns
    -------
    float
        Unrounded trial-count estimate; callers needing an integer must round upward.

    Notes
    -----
    Equation: trials = 3 / (1-target_lower_bound).

    Assumptions: The rare-failure Poisson approximation and a rounded 95-percent
    numerator.

    Excludes: Exact binomial sizing, observed failures, and certification.

    References
    ----------
    Derivation: Set the rule-of-three failure upper limit to 1-target_lower_bound and
    solve for the number of trials.

    Examples
    --------
    >>> from mlsysim.physics.robotics.assurance import calc_rule_of_three_trials
    >>> from mlsysim.core.units import ureg
    >>> lo = calc_rule_of_three_trials(0.5)
    >>> hi = calc_rule_of_three_trials(0.75)
    >>> assert hi == 2 * lo
    """
    if not (0.0 < target_lower_bound < 1.0):
        raise ValueError("target_lower_bound must be strictly between 0 and 1.")
    return RULE_OF_THREE_NUMERATOR / (1.0 - target_lower_bound)


def calc_zero_failure_mission_count(target_hazard_rate, mission_time, confidence: float = 0.95) -> float:
    """Estimate failure-free missions needed for a hazard-rate target.

    Parameters
    ----------
    target_hazard_rate : Quantity
        Positive assumed constant hazard rate in inverse time.
    mission_time : Quantity
        Positive duration per mission; rate times duration must lie in (0, 1) and should
        be small.
    confidence : float, optional
        One-sided confidence in (0, 1), default 0.95.

    Returns
    -------
    float
        Unrounded failure-free mission count; round upward for a whole-mission plan.

    Notes
    -----
    Equation: missions = log(1-confidence) / ``log(1-target_hazard_rate*mission_time)``.

    Assumptions: Independent identical missions with Bernoulli failure probability
    approximated by rate times duration.

    Excludes: The exact Poisson mission probability ``1-exp(-rate*time)``, dependence,
    changing mission exposure, and safety certification.

    Each mission of length t fails with probability lambda t (small); n clean
    missions bound it when (1 - lambda t)^n = 1 - C:

        n = ln(1 - C) / ln(1 - lambda t)

    Unrounded; at small lambda t it approaches the exposure wall divided by t.

    References
    ----------
    Derivation: Invert ``(1-p_mission)**missions`` = 1-confidence after approximating
    p_mission by ``rate*duration``.

    Examples
    --------
    >>> from mlsysim.physics.robotics.assurance import calc_zero_failure_mission_count
    >>> from mlsysim.core.units import ureg
    >>> n = calc_zero_failure_mission_count(0.01 / ureg.hour, 1 * ureg.hour)
    >>> assert n > 0
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    p_mission = (target_hazard_rate * mission_time).to("").magnitude
    if not (0.0 < p_mission < 1.0):
        raise ValueError("target_hazard_rate * mission_time must be strictly between 0 and 1.")
    return math.log(1.0 - confidence) / math.log(1.0 - p_mission)
