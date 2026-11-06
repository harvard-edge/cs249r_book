"""Policy-gradient training on verifiable rewards: coefficients, forks, and clipping.

Closed forms behind reinforcement learning of a policy from rewards that an
automated check computes (added 2026-10-20). Each takes stated quantities: the
rewards of one group of rollouts, a trajectory's turns and the boundaries at
which continuations are forked, sampled continuation pass counts, or one token's
old probability under a ratio clip. None models a trained network's learning
dynamics. The solvers in engine/solvers/reinforcement.py compose them with the
shared helpers in physics/agents.py (calc_pass_posterior for the false-accept
share, calc_mixed_outcome_fraction for reward contrast, calc_independent_success
and calc_short_circuit_cost for repeated verification).
"""

from __future__ import annotations

import math

from mlsysim.core._validation import validate_at_least


def calc_group_relative_coefficients(rewards, stabilizer: float = 0.0, bessel: bool = False):
    """Normalize rollout rewards by their within-group dispersion.

    Parameters
    ----------
    rewards : iterable of float
        Dimensionless scalar rewards for one group; at least one value.
    stabilizer : float, optional
        Nonnegative denominator offset in reward units; defaults to zero.
    bessel : bool, optional
        Use the sample standard deviation with divisor G - 1; requires G >= 2.

    Returns
    -------
    tuple
        Mean reward, standard deviation, and list of dimensionless normalized coefficients.

    Notes
    -----
    Equation: A_i = (r_i - mean(r)) / (std(r) + stabilizer).

    Assumptions: Rewards share one scale; the population divisor is G unless bessel is true.

    Excludes: Policy updates, clipping, learning dynamics, and any claim that reward normalization improves quality.

    Tied rewards have zero dispersion. With zero stabilizer the division is undefined and raises ZeroDivisionError; a positive stabilizer produces zero coefficients.

    References
    ----------
    Derivation: Center each reward and divide by its group dispersion; the sample variant replaces G with G - 1 in the variance denominator.

    Examples
    --------
    >>> mean, std, coefficients = calc_group_relative_coefficients([0, 1])
    >>> coefficients == [-1.0, 1.0]
    True
    >>> calc_group_relative_coefficients([1, 1], stabilizer=1)[2] == [0.0, 0.0]
    True
    """
    values = [float(r) for r in rewards]
    group = len(values)
    validate_at_least(group, 2 if bessel else 1, "rollouts in the group")
    mean = sum(values) / group
    divisor = group - 1 if bessel else group
    std = math.sqrt(sum((r - mean) ** 2 for r in values) / divisor)
    return mean, std, [(r - mean) / (std + stabilizer) for r in values]


def calc_continuation_fork_tokens(turns: int, tokens_per_turn: int, branches: int, boundaries=None):
    """Count generated tokens for independent continuation forks.

    Parameters
    ----------
    turns : int
        Turns in the base trajectory; at least two.
    tokens_per_turn : int
        Generated tokens per turn, assumed fixed and nonnegative.
    branches : int
        Continuations sampled at each selected boundary; nonnegative.
    boundaries : iterable of int or None, optional
        Distinct turn indices strictly between zero and turns; None selects every interior boundary.

    Returns
    -------
    tuple
        Base generated-token count, total fork count, and total generated-token count.

    Notes
    -----
    Equation: total = ``H*T`` + ``m*T*sum_k(H`` - k); all boundaries give ``H*T*(1`` + ``m*(H`` - 1)/2).

    Assumptions: Each branch independently generates every remaining turn, and all turns have the same token cost.

    Excludes: Shared generated suffixes, early termination, input prefill, and caching.

    References
    ----------
    Derivation: Add the ``H*T`` base trajectory to m continuations of length H - k at every selected boundary k.

    Examples
    --------
    >>> base, forks, total = calc_continuation_fork_tokens(3, 10, 2)
    >>> (base, forks, total) == (3 * 10, 2 * 2, 3 * 10 + 2 * 10 * (2 + 1))
    True
    """
    validate_at_least(turns, 2, "turns")
    ks = list(range(1, turns)) if boundaries is None else list(boundaries)
    if any(not 0 < k < turns for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("fork boundaries must be distinct turns strictly inside the trajectory")
    base = turns * tokens_per_turn
    remaining = sum(turns - k for k in ks)
    return base, branches * len(ks), base + branches * tokens_per_turn * remaining


def calc_binary_mean_se_bound(samples: int) -> float:
    """Bound the standard error of an independent binary sample mean.

    Parameters
    ----------
    samples : int
        Number of independent Bernoulli samples; at least one.

    Returns
    -------
    float
        Maximum standard error in probability units.

    Notes
    -----
    Equation: SE <= 1 / ``(2*sqrt(samples))``.

    Assumptions: Samples are independent and identically distributed Bernoulli outcomes.

    Excludes: Dependence, selection bias, confidence-interval coverage, and finite-population correction.

    The difference of two independent means with equal sample counts has a standard-error bound sqrt(2) times this value.

    References
    ----------
    Derivation: Bernoulli variance ``p*(1`` - p) is maximized at p = 1/2, then divide that variance by the sample count.

    Examples
    --------
    >>> calc_binary_mean_se_bound(4) == calc_binary_mean_se_bound(1) / 2
    True
    """
    validate_at_least(samples, 1, "binary samples")
    return 1 / (2 * math.sqrt(samples))


def calc_ratio_clip_room(probability: float, epsilon: float):
    """Return the upper ratio-clip threshold and its absolute increment.

    Parameters
    ----------
    probability : float
        Old token probability between zero and one.
    epsilon : float
        Nonnegative dimensionless upper-ratio allowance.

    Returns
    -------
    tuple
        Unclamped probability threshold (1 + epsilon)*p and increment epsilon*p.

    Notes
    -----
    Equation: ceiling = (1 + ``epsilon)*probability``; rise = ``epsilon*probability``.

    Assumptions: The upper ratio threshold is supplied as a hypothetical bound on a probability update.

    Excludes: Renormalization, clipping at probability one, and a guarantee that PPO clipping constrains actual updates.

    A clipped policy objective is not a hard probability constraint. This function computes the ratio threshold only, and the threshold can exceed one.

    References
    ----------
    Derivation: Multiply the old probability by an allowed ratio and subtract the old value to express the increment.

    Examples
    --------
    >>> ceiling, rise = calc_ratio_clip_room(0.5, 0.25)
    >>> ceiling - 0.5 == rise
    True
    """
    return (1 + epsilon) * probability, epsilon * probability
