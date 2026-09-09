"""Statistical and workflow propagation helpers."""

from __future__ import annotations

import math


def calc_population_stability_index(expected, actual, epsilon=1e-12):
    """Population Stability Index (PSI) between two aligned distributions.

    Measures distribution drift between a reference ("expected") and an
    observed ("actual") binned distribution:
    ``PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)``
    (the symmetrized KL divergence). Interpretation thresholds live in
    ``Ops.Monitoring`` (the single source of truth the book follows:
    0.10 warn / 0.20 review / 0.25 critical) — do not restate bands here.

    Parameters
    ----------
    expected, actual : sequence of float
        Bin probabilities (dimensionless, each summing to ~1), aligned
        bin-for-bin and of equal length.
    epsilon : float
        Floor applied to every bin to avoid log(0) / division by zero when a
        bin is empty.

    Returns
    -------
    float
        PSI value (dimensionless, >= 0; 0 means identical distributions).
    """
    if len(expected) != len(actual):
        raise ValueError("expected and actual distributions must have the same length")

    total = 0.0
    for exp, act in zip(expected, actual):
        exp = max(float(exp), epsilon)
        act = max(float(act), epsilon)
        total += (act - exp) * math.log(act / exp)
    return total


def calc_two_proportion_sample_size(baseline_rate, detectable_lift, z_alpha=1.96, z_beta=0.84, pooled=False):
    """Per-arm sample size for a two-proportion A/B test.

    Standard normal-approximation power formula:
    ``n = 2 * (z_alpha + z_beta)^2 * p(1-p) / delta^2``, using the baseline
    rate's variance for both arms (a common simplification valid for small
    lifts). Defaults correspond to a two-sided 5% significance level
    (z = 1.96) and 80% power (z = 0.84).

    Parameters
    ----------
    baseline_rate : float
        Control-arm conversion rate p in (0, 1).
    detectable_lift : float
        Minimum absolute difference in rates to detect (same 0-1 scale as
        ``baseline_rate``, e.g. 0.01 for one percentage point). Must be > 0.
    z_alpha : float
        Normal quantile for the significance level (1.96 = two-sided 0.05).
    z_beta : float
        Normal quantile for power (0.84 = 80% power).
    pooled : bool
        When True, use the pooled two-proportion form, which carries the
        variance of both arms rather than reusing the baseline's for each:

        ``n = (z_alpha * sqrt(2 * p_bar * q_bar)
               + z_beta * sqrt(p1*q1 + p2*q2))^2 / delta^2``

        where ``p_bar`` is the mean of the two rates. This is the textbook
        form and the one the book's own appendix displays. It is not the
        default because the simplified form above is already in use and its
        results are quoted in print; pass ``pooled=True`` where the pooled
        form is the one being taught.

    Returns
    -------
    float
        Required sample size per arm (unrounded; callers should ceil).
    """
    p = float(baseline_rate)
    delta = float(detectable_lift)
    if not 0.0 < p < 1.0:
        raise ValueError(f"baseline_rate must be in (0, 1), got {p}")
    if delta <= 0.0:
        raise ValueError(f"detectable_lift must be > 0, got {delta}")
    if pooled:
        p2 = p + delta
        if not 0.0 < p2 < 1.0:
            raise ValueError(
                f"baseline_rate + detectable_lift must stay in (0, 1), got {p2}"
            )
        p_bar = (p + p2) / 2.0
        numerator = (
            float(z_alpha) * math.sqrt(2 * p_bar * (1 - p_bar))
            + float(z_beta) * math.sqrt(p * (1 - p) + p2 * (1 - p2))
        ) ** 2
        return numerator / delta ** 2
    variance = p * (1 - p)
    return 2 * (float(z_alpha) + float(z_beta)) ** 2 * variance / delta ** 2


def calc_constraint_propagation_factor(stage_from, stage_to, base=2):
    """Cost multiplier for finding a workflow constraint at a later lifecycle stage.

    Geometric cost-of-delay model: ``factor = base ** (stage_to - stage_from)``,
    the "cost of a defect doubles per stage" rule of thumb from software
    engineering economics (Boehm). Stages are integer lifecycle indices
    (e.g. 0 = requirements, ... N = production); the result is a
    dimensionless multiplier (1 when the stages are equal).
    """
    if stage_to < stage_from:
        raise ValueError("stage_to must be greater than or equal to stage_from")
    return int(base ** (stage_to - stage_from))
