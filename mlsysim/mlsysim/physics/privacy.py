"""Differential privacy and information-leakage formulas (added 2026-10-20)."""

from __future__ import annotations

import math


def calc_laplace_scale(sensitivity, epsilon):
    """Scale ``b = sensitivity / epsilon`` of the Laplace mechanism.

    Adding Laplace(0, b) noise to a query whose value one protected
    contribution can change by at most ``sensitivity`` gives
    ``epsilon``-differential privacy. The expected absolute error of one
    draw equals ``b``. ``sensitivity`` may carry units (dollars, counts).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    return sensitivity / epsilon


def calc_basic_composition(epsilons):
    """Basic sequential composition: the budgets of releases on the same data add."""
    return sum(epsilons)


def calc_membership_tpr_ceiling(fpr, epsilon):
    """Highest true-positive rate any membership test can reach at false-positive
    rate ``fpr`` against a pure ``epsilon``-DP release (delta = 0).

    Applying the DP inequality to the event "test flags member" gives
    ``TPR <= e^eps FPR``; applying it to the complement gives
    ``1 - TPR >= e^-eps (1 - FPR)``.
    """
    return min(math.exp(epsilon) * fpr, 1 - math.exp(-epsilon) * (1 - fpr))


def calc_level_bits(levels):
    """Information carried by choosing one of ``levels`` equally likely values, in bits."""
    return math.log2(levels)
