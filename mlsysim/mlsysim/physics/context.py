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
    """
    Tokens left for observations once the reply and the fixed roles are reserved.

        S_obs <= S_max - K_max - (S_system + S_tools + S_state)
    """
    return window - output_reserve - fixed_tokens


def calc_turns_that_fit(observation_budget: int, tokens_per_turn: int) -> int:
    """Whole turns of appended observations that fit in ``observation_budget``."""
    return observation_budget // tokens_per_turn


def calc_softmax_target_weight(target_logit: float, competitors) -> float:
    """
    One attention head's weight on a target key among fixed-logit competitors.

        a = e^{z_t} / (e^{z_t} + sum_j n_j e^{z_j})

    ``competitors`` is an iterable of ``(count, logit)`` pairs: ``count`` keys
    that each score ``logit``. Every key adds a positive term to the shared
    normalizer, so more keys at fixed logits always lower the target's weight.
    """
    target = _math.exp(target_logit)
    return target / (target + sum(n * _math.exp(z) for n, z in competitors))


def calc_equivalent_key_count(count: int, logit: float, reference_logit: float) -> float:
    """Keys at ``reference_logit`` that add as much to the normalizer as ``count`` keys at ``logit``."""
    return count * _math.exp(logit - reference_logit)


def calc_zoned_prefill_tokens(turns: int, root: int, trunk: int, leaf: int, trunk_rewrites: int) -> int:
    """
    Uncached input of a trajectory whose requests are ordered root, trunk, leaf.

    The first call is cold (root + trunk + leaf); each of ``trunk_rewrites``
    calls prefills trunk and leaf after the reused root; every other call
    prefills only the leaf after the reused root and trunk:

        S + R (S_trunk + S_leaf) + (H - 1 - R) S_leaf

    Assumes compatible cached state is retained for every matching prefix.
    """
    warm = turns - 1 - trunk_rewrites
    return (root + trunk + leaf) + trunk_rewrites * (trunk + leaf) + warm * leaf


def calc_compaction_cycle(trigger: int, target: int, tokens_per_turn: int) -> int:
    """
    Turns between compaction passes under two watermarks (hysteresis).

        n_c = ceil((S_trigger - S_target) / delta)

    A pass resets the request to ``target``; it fires on the first turn whose
    appended observation brings the request to ``trigger`` or beyond.
    """
    return _math.ceil((trigger - target) / tokens_per_turn)

