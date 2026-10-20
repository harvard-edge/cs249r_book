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
    """
    Group-relative (GRPO) coefficients of one group of rollout rewards.

        A_i = (r_i - mean(r)) / (std(r) + epsilon)

    ``std`` divides by ``G`` (the population dispersion) or, with ``bessel``,
    by ``G - 1``, as PyTorch-style ``nanstd`` does. A group whose rewards all
    tie has zero dispersion; every coefficient is then zero when ``epsilon`` is
    positive (and undefined when it is zero). Returns ``(mean, std, coefficients)``.
    """
    values = [float(r) for r in rewards]
    group = len(values)
    validate_at_least(group, 2 if bessel else 1, "rollouts in the group")
    mean = sum(values) / group
    divisor = group - 1 if bessel else group
    std = math.sqrt(sum((r - mean) ** 2 for r in values) / divisor)
    return mean, std, [(r - mean) / (std + stabilizer) for r in values]


def calc_continuation_fork_tokens(turns: int, tokens_per_turn: int, branches: int, boundaries=None):
    """
    Generated tokens when ``branches`` continuations are forked after each listed turn.

    A fork after turn ``k`` reruns the remaining ``H - k`` turns once per
    branch. ``boundaries=None`` forks at every boundary ``k = 1 .. H - 1``, so
    the total is ``H T (1 + m (H - 1) / 2)``: it grows with the square of the
    horizon. Returns ``(base_tokens, branch_count, total_tokens)``.
    """
    validate_at_least(turns, 2, "turns")
    ks = list(range(1, turns)) if boundaries is None else list(boundaries)
    if any(not 0 < k < turns for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("fork boundaries must be distinct turns strictly inside the trajectory")
    base = turns * tokens_per_turn
    remaining = sum(turns - k for k in ks)
    return base, branches * len(ks), base + branches * tokens_per_turn * remaining


def calc_binary_mean_se_bound(samples: int) -> float:
    """
    Largest standard error of the mean of ``k`` binary outcomes: ``1 / (2 sqrt k)``.

    A Bernoulli variance ``p (1 - p)`` is at most 1/4, reached at ``p = 1/2``.
    The difference of two independent such means is bounded by ``sqrt 2`` times it.
    """
    validate_at_least(samples, 1, "binary samples")
    return 1 / (2 * math.sqrt(samples))


def calc_ratio_clip_room(probability: float, epsilon: float):
    """
    How far one update may raise a token's probability under a symmetric ratio clip.

    The ratio of new to old probability is capped at ``1 + epsilon``, so the
    ceiling is ``(1 + epsilon) p`` and the absolute rise allowed is
    ``epsilon p``: large for a dominant token, tiny for a rare one. Returns
    ``(ceiling, rise)``.
    """
    return (1 + epsilon) * probability, epsilon * probability
