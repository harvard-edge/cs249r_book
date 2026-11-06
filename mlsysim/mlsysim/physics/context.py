"""
Context engineering: the equations behind what one agent request carries.

Pure arithmetic on serialized token counts (added 2026-10-20). A request is
budgeted by role against the window, laid out in zones ordered by change rate
so a serving engine can reuse its prefix, compacted between two watermarks,
and edited when a write supersedes an observation. One attention head's
softmax gives the first-order cost of every admitted token. The solvers in
engine/solvers/context.py compose these.
"""

import math as _math


def calc_observation_budget(window: int, output_reserve: int, fixed_tokens: int) -> int:
    """Return the token budget remaining for observations.

    Parameters
    ----------
    window : int
        Total context-window capacity in tokens; nonnegative.
    output_reserve : int
        Tokens reserved for the generated reply; nonnegative.
    fixed_tokens : int
        Serialized system, tool, and persistent-state tokens; nonnegative.

    Returns
    -------
    int
        Available observation tokens; a negative result denotes overcommitment.

    Notes
    -----
    Equation: S_obs = window - output_reserve - fixed_tokens.

    Assumptions: All three counts refer to the same tokenizer and request.

    Excludes: Tokenizer estimation, KV capacity, and truncation; negative budgets are not clamped.

    References
    ----------
    Derivation: Subtract disjoint reservations from a finite token capacity.

    Examples
    --------
    >>> calc_observation_budget(100, 20, 30) == 100 - 20 - 30
    True
    """
    return window - output_reserve - fixed_tokens


def calc_turns_that_fit(observation_budget: int, tokens_per_turn: int) -> int:
    """Return the number of complete observation turns that fit.

    Parameters
    ----------
    observation_budget : int
        Available observation capacity in tokens; nonnegative.
    tokens_per_turn : int
        Fixed serialized tokens appended per turn; positive.

    Returns
    -------
    int
        Number of whole turns admitted by the token budget.

    Notes
    -----
    Equation: turns = floor(observation_budget / tokens_per_turn).

    Assumptions: Every turn contributes the same token count and existing state stays fixed.

    Excludes: Partial turns, variable-length observations, and context compaction.

    References
    ----------
    Derivation: Integer division counts complete equal-sized groups within a capacity.

    Examples
    --------
    >>> calc_turns_that_fit(100, 30) == 100 // 30
    True
    """
    return observation_budget // tokens_per_turn


def calc_softmax_target_weight(target_logit: float, competitors) -> float:
    """Return a target key's softmax weight against grouped competitors.

    Parameters
    ----------
    target_logit : float
        Dimensionless target score; its exponential must be representable.
    competitors : iterable of tuple
        Pairs of nonnegative key counts and dimensionless logits.

    Returns
    -------
    float
        Normalized target weight, between zero and one for valid inputs.

    Notes
    -----
    Equation: a = exp(z_t) / (exp(z_t) + sum_j n_j exp(z_j)).

    Assumptions: Competitor keys in each group share a fixed logit; there is one target key.

    Excludes: Learned changes to logits, multihead aggregation, and overflow-stable softmax.

    Adding keys at fixed logits increases the normalizer and reduces target weight.

    References
    ----------
    Derivation: Expand the softmax denominator by repeated equal-logit terms; each group contributes n_j exp(z_j).

    Examples
    --------
    >>> calc_softmax_target_weight(0.0, [(1, 0.0)]) == 0.5
    True
    """
    target = _math.exp(target_logit)
    return target / (target + sum(n * _math.exp(z) for n, z in competitors))


def calc_equivalent_key_count(count: int, logit: float, reference_logit: float) -> float:
    """Return an equal-normalizer count at a reference logit.

    Parameters
    ----------
    count : int
        Number of original keys; nonnegative.
    logit : float
        Dimensionless score shared by the original keys.
    reference_logit : float
        Dimensionless score assigned to equivalent keys.

    Returns
    -------
    float
        Equivalent key count; fractional counts represent normalizer mass.

    Notes
    -----
    Equation: n_equivalent = count * exp(logit - reference_logit).

    Assumptions: Logits are fixed and their exponential difference is representable.

    Excludes: Integer rounding and changes in attention scores induced by the input.

    References
    ----------
    Derivation: Solve n_equivalent exp(reference_logit) = count exp(logit) for the equivalent count.

    Examples
    --------
    >>> calc_equivalent_key_count(4, 2.0, 2.0) == 4
    True
    """
    return count * _math.exp(logit - reference_logit)


def calc_zoned_prefill_tokens(turns: int, root: int, trunk: int, leaf: int, trunk_rewrites: int) -> int:
    """Count uncached input tokens in a root-trunk-leaf trajectory.

    Parameters
    ----------
    turns : int
        Number of calls in the trajectory; at least one.
    root : int
        Invariant leading-prefix size in tokens; nonnegative.
    trunk : int
        Less frequently changed middle-prefix size in tokens; nonnegative.
    leaf : int
        Per-call changing suffix size in tokens; nonnegative.
    trunk_rewrites : int
        Calls after the first that rewrite the trunk; from zero to turns - 1.

    Returns
    -------
    int
        Total tokens prefilling uncached input across all calls.

    Notes
    -----
    Equation: total = root + trunk + leaf + ``R*(trunk`` + leaf) + (H - 1 - ``R)*leaf``.

    Assumptions: The first call is cold; matching root and trunk prefixes remain cached for subsequent calls.

    Excludes: Eviction, cache lookup overhead, output tokens, and partial trunk-prefix reuse.

    References
    ----------
    Derivation: Partition calls into one cold call, R trunk rewrites, and H - 1 - R leaf-only updates.

    Examples
    --------
    >>> calc_zoned_prefill_tokens(3, 10, 20, 5, 1) == (10 + 20 + 5) + (20 + 5) + 5
    True
    """
    warm = turns - 1 - trunk_rewrites
    return (root + trunk + leaf) + trunk_rewrites * (trunk + leaf) + warm * leaf


def calc_compaction_cycle(trigger: int, target: int, tokens_per_turn: int) -> int:
    """Return turns between two context-compaction watermarks.

    Parameters
    ----------
    trigger : int
        Token count at which compaction fires; greater than target.
    target : int
        Token count immediately after compaction; nonnegative.
    tokens_per_turn : int
        Fixed tokens appended per turn; positive.

    Returns
    -------
    int
        Turns until the first request reaches or exceeds the trigger.

    Notes
    -----
    Equation: n_cycle = ceil((trigger - target) / tokens_per_turn).

    Assumptions: Compaction resets exactly to target and observations append at a fixed rate.

    Excludes: Compaction latency, information loss, and variable-size observations.

    References
    ----------
    Derivation: Solve target + ``n*tokens_per_turn`` >= trigger for the smallest integer n.

    Examples
    --------
    >>> calc_compaction_cycle(100, 40, 20) == 3
    True
    """
    return _math.ceil((trigger - target) / tokens_per_turn)
