"""Fairness arithmetic: error-rate identities, the bias feedback model, and gap sample sizes.

Added 2026-10-20. The closed forms a fairness release gate rests on:

- Chouldechova's identity, which fixes a group's false positive rate from its
  base rate, positive predictive value, and false negative rate;
- the bias feedback model, a disparity multiplied by a feedback factor at each
  retraining cycle, and the cycle at which it crosses a release gate;
- the per-group sample size that pins a gap between two groups' rates.
"""

from __future__ import annotations

import math


def calc_fpr_from_ppv(base_rate, ppv, fnr):
    """Infer a false-positive rate from prevalence, precision, and false negatives.

    Parameters
    ----------
    base_rate : float
        Positive-class prevalence p, dimensionless, in (0, 1).
    ppv : float
        Positive predictive value, dimensionless, in (0, 1].
    fnr : float
        False-negative fraction, dimensionless, expected in [0, 1].

    Returns
    -------
    float
        Implied false-positive fraction. An inconsistent input triple can yield
        a value above one; this function does not clamp the result.

    Notes
    -----
    Equation: FPR = p/(1-p) * (1-PPV)/PPV * (1-FNR).

    Assumptions: All rates describe the same population and classifier threshold.

    Excludes: Estimation uncertainty, threshold selection, and compatibility checks for
        the supplied rates.

    Equal PPV with unequal base rates prevents simultaneously equal false-positive
    and false-negative rates, except in degenerate cases.

    References
    ----------
    Chouldechova (2017), "Fair Prediction with Disparate Impact."
    https://arxiv.org/abs/1703.00056 (bibliography key: chouldechova2017).

    Examples
    --------
    >>> round(calc_fpr_from_ppv(0.2, 0.8, 0.1), 5)
    0.05625
    """
    p = float(base_rate)
    if not 0.0 < p < 1.0 or not 0.0 < ppv <= 1.0:
        raise ValueError("base_rate must be in (0, 1) and ppv in (0, 1]")
    return p / (1 - p) * (1 - ppv) / ppv * (1 - fnr)


def calc_feedback_disparity(launch_gap, factor, cycles):
    """Propagate a disparity through repeated multiplicative feedback.

    Parameters
    ----------
    launch_gap : float
        Initial rate difference, expressed as a fraction.
    factor : float
        Dimensionless multiplicative feedback factor per cycle, nonnegative.
    cycles : int
        Number of retraining cycles, nonnegative.

    Returns
    -------
    float
        Modeled rate difference after the specified cycles, in the same units
        as launch_gap; not clamped to a probability interval.

    Notes
    -----
    Equation: gap(k) = gap(0) * ``factor**k``.

    Assumptions: The same feedback factor applies each cycle; factors above one grow
        the disparity and factors below one reduce it.

    Excludes: Observed feedback dynamics, saturation, and sampling error.

    References
    ----------
    Derivation: apply the recurrence gap(k+1) = factor * gap(k) repeatedly.

    Examples
    --------
    >>> calc_feedback_disparity(0.02, 2, 3)
    0.16
    """
    return launch_gap * factor**cycles


def calc_cycles_to_gate(launch_gap, factor, gate):
    """Find the fractional cycle at which a growing disparity reaches a gate.

    Parameters
    ----------
    launch_gap : float
        Initial positive disparity, below gate, in the same units as gate.
    factor : float
        Dimensionless multiplicative growth per cycle, greater than one.
    gate : float
        Positive disparity threshold, greater than launch_gap.

    Returns
    -------
    float
        Continuous cycle count at equality. Failure under a strict greater-than
        gate occurs at the next whole cycle above this value.

    Notes
    -----
    Equation: k = ln(gate/launch_gap) / ln(factor).

    Assumptions: A positive disparity grows monotonically by a constant factor.

    Excludes: Discrete-cycle rounding and empirical estimation of feedback strength.

    References
    ----------
    Derivation: take logarithms of gate = launch_gap * ``factor**k``.

    Examples
    --------
    >>> calc_cycles_to_gate(0.01, 2, 0.08)
    3.0
    """
    if factor <= 1.0 or not 0.0 < launch_gap < gate:
        raise ValueError("need factor > 1 and 0 < launch_gap < gate")
    return math.log(gate / launch_gap) / math.log(factor)


def calc_gap_samples_per_group(rate, half_width, z_critical):
    """Plan equal group sizes for a normal-approximation rate-gap interval.

    Parameters
    ----------
    rate : float
        Assumed common Bernoulli probability p, dimensionless, in [0, 1].
    half_width : float
        Desired positive half-width of the difference in rates, as a fraction.
    z_critical : float
        Positive normal critical value for the chosen confidence level.

    Returns
    -------
    int
        Rounded-up number of independent observations required in each group.

    Notes
    -----
    Equation: n = ``ceil(2*z_critical**2*p*(1-p)/half_width**2)``.

    Assumptions: Independent Bernoulli observations, equal group sizes, the same
        planning rate in both groups, and an adequate normal approximation.

    Excludes: Exact finite-sample coverage, paired groups, and unequal allocation.

    References
    ----------
    Derivation: the independent difference has variance ``2*p*(1-p)/n``; solve
        ``z*sqrt(variance)`` = half_width for n and round up.

    Examples
    --------
    >>> calc_gap_samples_per_group(0.5, 0.1, 2)
    200
    """
    return math.ceil(2 * z_critical**2 * rate * (1 - rate) / half_width**2)
