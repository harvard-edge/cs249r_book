"""Fine-tuning on serialized trajectories: weighting, packing, and preference pairs.

Closed forms behind supervised and preference fine-tuning of a policy on
recorded trajectories (added 2026-10-20). Each takes stated counts: selected
target tokens per example, stored example lengths against a fixed buffer, or
log-probability sums of two scored records. None models a trained network.
The solvers in engine/solvers/fine_tuning.py compose them.
"""

from __future__ import annotations

import math

from mlsysim.core._validation import validate_at_least


def calc_token_average_shares(target_counts) -> list[float]:
    """
    Each example's weight in a token-averaged loss: ``n_i / sum_j n_j``.

    Every selected target token counts once, so an example's weight grows with
    its target count.
    """
    counts = list(target_counts)
    total = sum(counts)
    validate_at_least(total, 1, "total target tokens")
    return [n / total for n in counts]


def calc_example_average_share(group_examples: int, total_examples: int) -> float:
    """
    A group's weight in an example-averaged loss: ``k / N``.

    Every example's mean loss counts once, so a trajectory split into ``k``
    records carries ``k`` times the weight of an unsplit example.
    """
    validate_at_least(total_examples, 1, "total examples")
    return group_examples / total_examples


def calc_first_fit_decreasing(lengths, capacity: int) -> list[int]:
    """
    Buffer loads when examples are packed longest first into the first buffer with room.

    Returns the stored tokens of each fixed-``capacity`` buffer. An example
    longer than ``capacity`` cannot be packed.
    """
    loads: list[int] = []
    for n in sorted(lengths, reverse=True):
        if n > capacity:
            raise ValueError("an example is longer than the buffer")
        for i, load in enumerate(loads):
            if load + n <= capacity:
                loads[i] = load + n
                break
        else:
            loads.append(n)
    return loads


def calc_preference_pair(policy_preferred: float, reference_preferred: float,
                         policy_dispreferred: float, reference_dispreferred: float,
                         beta: float):
    """
    One preference pair under the DPO loss (Rafailov et al. 2023).

    Each record's score is its policy log-probability sum minus the reference's,
    ``z = log pi_theta - log pi_ref``; the pair margin is ``beta (z_w - z_l)``;
    the logistic score is ``sigma(margin)``; the loss is
    ``-log sigma(margin) = log(1 + e^-margin)``; the gradient's saturation
    factor is ``sigma(-margin)``, and the full coefficient on the pair is
    ``beta sigma(-margin)``.

    Returns ``(score_w, score_l, margin, preference_score, loss, saturation,
    coefficient)``.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    score_w = policy_preferred - reference_preferred
    score_l = policy_dispreferred - reference_dispreferred
    margin = beta * (score_w - score_l)
    preference_score = 1 / (1 + math.exp(-margin))
    loss = math.log1p(math.exp(-margin))
    saturation = 1 / (1 + math.exp(margin))
    return score_w, score_l, margin, preference_score, loss, saturation, beta * saturation
