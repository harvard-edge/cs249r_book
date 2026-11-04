"""Differential privacy and information-leakage formulas (added 2026-10-20)."""

from __future__ import annotations

import math


def calc_laplace_scale(sensitivity, epsilon):
    """Calculate the noise scale for a scalar Laplace mechanism.

    Parameters
    ----------
    sensitivity : float or Quantity
        Nonnegative global L1 sensitivity of the query, in its output units.
    epsilon : float
        Positive dimensionless pure-differential-privacy budget.

    Returns
    -------
    float or Quantity
        Laplace scale b in the same units as sensitivity; also the expected
        absolute error of one Laplace(0, b) draw.

    Notes
    -----
    Equation: b = sensitivity / epsilon.

    Assumptions: Sensitivity bounds every allowed neighboring-dataset change and
        the mechanism uses independent Laplace noise with this scale.

    Excludes: Sensitivity estimation, noise sampling, and approximate-DP accounting.

    References
    ----------
    Derivation: the Laplace density ratio is bounded by exp(sensitivity/b);
        choosing b=sensitivity/epsilon bounds that ratio by exp(epsilon).

    Examples
    --------
    >>> calc_laplace_scale(2, 1)
    2.0
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    return sensitivity / epsilon


def calc_basic_composition(epsilons):
    """Sum pure-DP budgets for sequential releases on the same data.

    Parameters
    ----------
    epsilons : iterable of float
        Nonnegative dimensionless privacy budgets for individual releases.

    Returns
    -------
    float
        Dimensionless basic sequential-composition budget; an empty iterable
        returns the additive identity zero.

    Notes
    -----
    Equation: epsilon_total = sum(epsilons).

    Assumptions: Each release satisfies its stated pure-DP guarantee under the same
        neighboring-dataset definition; releases may be adaptive.

    Excludes: Advanced composition, disjoint-data parallel composition, and delta terms.

    References
    ----------
    Derivation: conditional density-ratio bounds multiply across releases;
        their logarithmic epsilon bounds therefore add.

    Examples
    --------
    >>> round(calc_basic_composition([0.1, 0.2]), 2)
    0.3
    """
    return sum(epsilons)


def calc_membership_tpr_ceiling(fpr, epsilon):
    """Bound membership-test sensitivity under pure differential privacy.

    Parameters
    ----------
    fpr : float
        Membership-test false-positive rate, dimensionless, in [0, 1].
    epsilon : float
        Nonnegative dimensionless pure-DP budget, with delta equal to zero.

    Returns
    -------
    float
        Upper bound on the membership-test true-positive rate, as a fraction.

    Notes
    -----
    Equation: TPR <= ``min(exp(epsilon)*FPR``,
        ``1-exp(-epsilon)*(1-FPR))``.

    Assumptions: The member/nonmember hypotheses are neighboring datasets under
        the release's pure-DP definition.

    Excludes: Approximate-DP delta, prior membership probability, and construction
        of an attack achieving the ceiling.

    References
    ----------
    Derivation: apply the DP inequality to the event that a test flags a
        member, then apply the reverse inequality to its complement.

    Examples
    --------
    >>> round(calc_membership_tpr_ceiling(0.1, 0), 2)
    0.1
    """
    return min(math.exp(epsilon) * fpr, 1 - math.exp(-epsilon) * (1 - fpr))


def calc_level_bits(levels):
    """Calculate the information in an equiprobable finite choice.

    Parameters
    ----------
    levels : int
        Positive number of equally likely distinguishable values.

    Returns
    -------
    float
        Shannon information in bits.

    Notes
    -----
    Equation: bits = log2(levels).

    Assumptions: Each level is equally probable and observable without ambiguity.

    Excludes: Nonuniform distributions, side information, and measurement noise.

    References
    ----------
    Derivation: entropy of L equiprobable values is
        ``-sum((1/L)*log2(1/L))`` = log2(L).

    Examples
    --------
    >>> calc_level_bits(8)
    3.0
    """
    return math.log2(levels)
