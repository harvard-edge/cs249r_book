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
    """Calculate each example's share of a token-averaged loss.

    Parameters
    ----------
    target_counts : iterable of int
        Selected target-token counts per example, nonnegative with positive sum.

    Returns
    -------
    list of float
        Dimensionless loss weights in input order; they sum to one for
        nonnegative counts.

    Notes
    -----
    Equation: weight_i = target_counts[i] / sum(target_counts).

    Assumptions: Every selected target token contributes equally to the loss.

    Excludes: Token-level masking decisions and differences in individual token losses.

    References
    ----------
    Derivation: divide each example's number of selected loss terms by the
        total number of selected terms.

    Examples
    --------
    >>> calc_token_average_shares([1, 3])
    [0.25, 0.75]
    """
    counts = list(target_counts)
    total = sum(counts)
    validate_at_least(total, 1, "total target tokens")
    return [n / total for n in counts]


def calc_example_average_share(group_examples: int, total_examples: int) -> float:
    """Calculate a group's share of an example-averaged loss.

    Parameters
    ----------
    group_examples : int
        Number of records in the group, between zero and total_examples.
    total_examples : int
        Positive total number of records.

    Returns
    -------
    float
        Dimensionless fraction of equal-weight example losses assigned to
        the group.

    Notes
    -----
    Equation: share = group_examples / total_examples.

    Assumptions: Each record contributes one mean loss, irrespective of its token count.

    Excludes: Within-record loss variation and validation of group membership.

    Splitting a trajectory into k records gives it k times the weight of one
    unsplit record when all record means receive equal weight.

    References
    ----------
    Derivation: a group of k equal-weight records contributes k of N terms
        to the example average.

    Examples
    --------
    >>> calc_example_average_share(2, 8)
    0.25
    """
    validate_at_least(total_examples, 1, "total examples")
    return group_examples / total_examples


def calc_first_fit_decreasing(lengths, capacity: int) -> list[int]:
    """Pack example lengths using the first-fit-decreasing heuristic.

    Parameters
    ----------
    lengths : iterable of int
        Nonnegative example lengths in tokens, each no larger than capacity.
    capacity : int
        Positive token capacity of each buffer.

    Returns
    -------
    list of int
        Token load of each allocated buffer, in buffer-creation order.

    Notes
    -----
    Equation: Sort lengths in descending order; place each length n in the first
        buffer whose current load plus n does not exceed capacity.

    Assumptions: Examples cannot be split and buffer capacities are identical.

    Excludes: Optimal bin packing, sequence-boundary overhead, and attention masking.

    References
    ----------
    Derivation: this is the explicit first-fit-decreasing packing procedure,
        a heuristic rather than a claim of optimal allocation.

    Examples
    --------
    >>> calc_first_fit_decreasing([6, 4, 3, 3], 10)
    [10, 6]
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


def calc_preference_pair(
    policy_preferred: float,
    reference_preferred: float,
    policy_dispreferred: float,
    reference_dispreferred: float,
    beta: float,
):
    """Evaluate the DPO loss and coefficients for one preference pair.

    Parameters
    ----------
    policy_preferred : float
        Policy log-probability sum for the preferred record, in natural-log units.
    reference_preferred : float
        Reference log-probability sum for the preferred record.
    policy_dispreferred : float
        Policy log-probability sum for the dispreferred record.
    reference_dispreferred : float
        Reference log-probability sum for the dispreferred record.
    beta : float
        Positive dimensionless scale of the preference margin.

    Returns
    -------
    tuple of float
        Preferred score, dispreferred score, scaled margin, logistic preference
        score, loss, saturation factor, and beta-scaled coefficient, in order.
        Scores and loss use natural-log conventions; all entries are dimensionless.

    Notes
    -----
    Equation: z_w = policy_preferred-reference_preferred and
        z_l = policy_dispreferred-reference_dispreferred. With m=``beta*(z_w-z_l)``,
        preference_score=sigmoid(m), loss=log(1+exp(-m)), saturation=sigmoid(-m),
        and coefficient=``beta*saturation``.

    Assumptions: Inputs are log-probability sums for the same pair under policy and
        reference models, with a supplied preference ordering.

    Excludes: Model training, preference-label validity, sequence normalization, and
        overflow protection for extreme margins.

    References
    ----------
    Rafailov et al. (2023), "Direct Preference Optimization: Your Language
    Model is Secretly a Reward Model."
    https://arxiv.org/abs/2305.18290 (bibliography key: rafailov2023dpo).

    Examples
    --------
    >>> pair = calc_preference_pair(-2, -2, -3, -3, 1)
    >>> pair[3], round(pair[4], 6), pair[6]
    (0.5, 0.693147, 0.5)
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
