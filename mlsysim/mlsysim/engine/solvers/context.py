"""Context-engineering solvers: what one agent request carries, and what it costs.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each
model is arithmetic on serialized token counts: a budget by role against the
window, one attention head's softmax over fixed logits, a measured
comprehension sweep, the input bill of a zoned layout under prefix reuse,
compaction between two watermarks, the edits that retire a stale observation,
and the cost per accepted task of a protected-field check. None models
tokenization across joins, cache eviction, or a trained model's attention.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..results import (
    AttentionDilutionResult,
    CompactionCadenceResult,
    ComprehensionByLengthResult,
    ContextBudgetResult,
    ContextInvalidationResult,
    PrefixLayoutBillResult,
    ProtectedFieldCheckResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...core.units import USD as _USD, ureg as _ureg
from ...physics.agents import (
    calc_append_only_context,
    calc_cost_per_accepted,
    calc_first_horizon_below,
    calc_step_error_for_target,
    calc_unchecked_success,
)
from ...physics.context import (
    calc_compaction_cycle,
    calc_equivalent_key_count,
    calc_observation_budget,
    calc_softmax_target_weight,
    calc_turns_that_fit,
    calc_zoned_prefill_tokens,
)


_TOKEN = _ureg.token


def _running(values):
    total, out = 0, []
    for v in values:
        total += v
        out.append(total)
    return out


class ContextBudgetModel(ForwardModel):
    """
    A request budgeted by role: ``S_sys + S_tools + S_state + S_obs <= S_max - K_max``.

    Appending every tool result grows ``S_obs`` by ``delta`` per turn until the
    observation budget overflows, and resends the history, so the input bill
    grows quadratically. Selecting a fixed observation share keeps every
    request the same size and the bill linear. Bills assume no prefix reuse
    and, for the appending policy, a window large enough to admit every request.
    """
    requires = ()
    produces = ContextBudgetResult

    def solve(self, window: int, output_reserve: int, system_tokens: int, tool_tokens: int,
              state_tokens: int, observation_per_turn: int, selected_observation: int, turns: int,
              digest_tokens: int) -> ContextBudgetResult:
        fixed = system_tokens + tool_tokens + state_tokens
        limit = window - output_reserve
        budget = calc_observation_budget(window, output_reserve, fixed)
        fit = calc_turns_that_fit(budget, observation_per_turn)
        evidence = limit - system_tokens - tool_tokens
        selected = fixed + selected_observation
        # The first request already holds one result: turn t reads fixed + delta t.
        first = fixed + observation_per_turn
        appended = [first + observation_per_turn * (t - 1) for t in range(1, turns + 1)]
        last, append_bill = calc_append_only_context(first, observation_per_turn, turns)
        select_bill = turns * selected
        return ContextBudgetResult(
            input_limit=limit,
            fixed_tokens=fixed,
            evidence_tokens=evidence,
            evidence_share=evidence / window,
            observation_budget=budget,
            turns_fit=fit,
            overflow_turn=fit + 1,
            overflow_observations=(fit + 1) * observation_per_turn,
            digest_turns=calc_turns_that_fit(budget, digest_tokens),
            max_observation_per_turn=budget // turns,
            selected_request=selected,
            append_requests=appended,
            append_cumulative=_running(appended),
            select_cumulative=_running([selected] * turns),
            append_bill=append_bill,
            select_bill=select_bill,
            bill_ratio=append_bill / select_bill,
            final_append_request=last,
            final_append_with_reserve=last + output_reserve,
        )


class AttentionDilutionModel(ForwardModel):
    """
    One head's weight on a target key among keys whose logits stay fixed.

    ``short_keys`` and ``long_keys`` count every key, the target included, at
    ``background_logit``. The look-alike case counts competitors only:
    ``unrelated_keys`` at the background logit plus ``lookalike_keys`` at the
    higher ``lookalike_logit`` of text that shares the evidence's identifiers.
    A sufficient condition for dilution, not a forecast of a trained model.
    """
    requires = ()
    produces = AttentionDilutionResult

    def solve(self, target_logit: float, background_logit: float, short_keys: int, long_keys: int,
              unrelated_keys: Optional[int] = None, lookalike_keys: Optional[int] = None,
              lookalike_logit: Optional[float] = None) -> AttentionDilutionResult:
        out = dict(
            short_weight=calc_softmax_target_weight(target_logit, [(short_keys - 1, background_logit)]),
            long_weight=calc_softmax_target_weight(target_logit, [(long_keys - 1, background_logit)]),
        )
        if unrelated_keys is not None and lookalike_keys is not None:
            look = (lookalike_keys, lookalike_logit)
            out.update(
                weight_without_lookalikes=calc_softmax_target_weight(
                    target_logit, [(unrelated_keys, background_logit)]),
                weight_with_lookalikes=calc_softmax_target_weight(
                    target_logit, [(unrelated_keys, background_logit), look]),
                lookalike_equivalent_keys=calc_equivalent_key_count(
                    lookalike_keys, lookalike_logit, background_logit),
                weight_half_unrelated=calc_softmax_target_weight(
                    target_logit, [(unrelated_keys / 2, background_logit), look]),
                weight_no_unrelated=calc_softmax_target_weight(target_logit, [look]),
            )
        return AttentionDilutionResult(**out)


class ComprehensionByLengthModel(ForwardModel):
    """
    Scores of models measured at every length of a context-length sweep.

    Keeps the models scored at every listed length, then reports the median
    and quartiles per length and how many models hold ``hold_bar`` at the
    longest one. Scores are fractions, as the source series stores them.
    """
    requires = ()
    produces = ComprehensionByLengthResult

    def solve(self, rows: Sequence[dict], lengths: Sequence[str], hold_bar: float,
              column_prefix: str = "score_", model_column: str = "model") -> ComprehensionByLengthResult:
        cols = [column_prefix + n for n in lengths]
        scored = [r for r in rows if all(r[c] is not None for c in cols)]
        arr = np.array([[r[c] for c in cols] for r in scored])
        last = cols[-1]
        low = min(scored, key=lambda r: r[last])
        high = max(scored, key=lambda r: r[last])
        return ComprehensionByLengthResult(
            lengths=list(lengths),
            scored_models=len(scored),
            median={n: float(np.median(arr[:, i])) for i, n in enumerate(lengths)},
            q25={n: float(np.percentile(arr[:, i], 25)) for i, n in enumerate(lengths)},
            q75={n: float(np.percentile(arr[:, i], 75)) for i, n in enumerate(lengths)},
            held_count=sum(1 for r in scored if r[last] >= hold_bar),
            lowest_last=low[last],
            highest_last=high[last],
            lowest_model=str(low[model_column]),
            highest_model=str(high[model_column]),
        )


class PrefixLayoutBillModel(ForwardModel):
    """
    Input bill of ``turns`` calls whose requests are ordered root, trunk, leaf.

    With a volatile field first, no prefix is reused and every input token is
    billed at full price. Ordered by change rate, the first call is cold, each
    trunk rewrite prefills trunk and leaf, and every other call prefills only
    the leaf; reused tokens are billed at ``cached_fraction`` of the price.
    Assumes compatible cached state is retained; prices only input tokens.
    """
    requires = ()
    produces = PrefixLayoutBillResult

    def solve(self, turns: int, root_tokens: int, trunk_tokens: int, leaf_tokens: int,
              trunk_rewrites: int, input_price: Quantity, cached_fraction: float) -> PrefixLayoutBillResult:
        request = root_tokens + trunk_tokens + leaf_tokens
        total = turns * request
        uncached = calc_zoned_prefill_tokens(turns, root_tokens, trunk_tokens, leaf_tokens, trunk_rewrites)
        cached = total - uncached
        unstable = (total * _TOKEN * input_price).to(_USD)
        stable = ((uncached + cached_fraction * cached) * _TOKEN * input_price).to(_USD)
        rewrite = trunk_tokens + leaf_tokens
        return PrefixLayoutBillResult(
            request_tokens=request,
            total_input=total,
            warm_calls=turns - 1 - trunk_rewrites,
            uncached_tokens=uncached,
            cached_tokens=cached,
            hit_rate=cached / total,
            unstable_cost=unstable,
            stable_cost=stable,
            cost_ratio=(unstable / stable).to("").magnitude,
            prefill_ratio=total / uncached,
            unstable_cost_per_call=unstable / turns,
            stable_cost_per_call=stable / turns,
            rewrite_tokens=rewrite,
            rewrite_in_leaf_calls=rewrite / leaf_tokens,
        )


def _trace(target: int, trigger: int, per_turn: int, turns: int):
    """Request length after each turn's append; a pass resets it to ``target``."""
    length, out = target, []
    for _ in range(turns):
        length += per_turn
        out.append(length)
        if length >= trigger:
            length = target
    return out


class CompactionCadenceModel(ForwardModel):
    """
    Compaction between a trigger and a target watermark (hysteresis).

    Each turn appends ``delta`` tokens; a pass fires on the first turn that
    reaches the trigger and resets the request to the target, so passes come
    every ``ceil((trigger - target) / delta)`` turns. The trigger is soft: the
    request can overshoot it by less than one append, and the output reserve
    must still fit. An optional alternate target prices a narrower gap.
    """
    requires = ()
    produces = CompactionCadenceResult

    def solve(self, window: int, output_reserve: int, target: int, trigger: int, tokens_per_turn: int,
              turns: int, alternate_target: Optional[int] = None) -> CompactionCadenceResult:
        cycle = calc_compaction_cycle(trigger, target, tokens_per_turn)
        at_trigger = target + cycle * tokens_per_turn
        out = dict(
            input_limit=window - output_reserve,
            cycle_turns=cycle,
            trigger_length=at_trigger,
            overshoot=at_trigger - trigger,
            length_with_reserve=at_trigger + output_reserve,
            fits_window=at_trigger + output_reserve <= window,
            passes=turns // cycle,
            lengths=_trace(target, trigger, tokens_per_turn, turns),
        )
        if alternate_target is not None:
            alt = calc_compaction_cycle(trigger, alternate_target, tokens_per_turn)
            out.update(alternate_cycle_turns=alt, alternate_passes=turns // alt,
                       alternate_lengths=_trace(alternate_target, trigger, tokens_per_turn, turns),
                       frequency_ratio=cycle / alt)
        return CompactionCadenceResult(**out)


class ContextInvalidationModel(ForwardModel):
    """
    Three edits that retire a superseded excerpt at the start of the trunk.

    Removing it or replacing it in place with a short marker changes the
    request right after the reused root, so every later token is prefilled
    again; appending a notice keeps the prefix and prefills only the notice,
    but leaves the stale tokens resident on every later call.
    """
    requires = ()
    produces = ContextInvalidationResult

    def solve(self, request_tokens: int, prefix_tokens: int, superseded_tokens: int, marker_tokens: int,
              notice_tokens: int) -> ContextInvalidationResult:
        removed = request_tokens - superseded_tokens
        replaced = removed + marker_tokens
        return ContextInvalidationResult(
            removed_total=removed,
            removed_suffix=removed - prefix_tokens,
            replaced_total=replaced,
            replaced_suffix=replaced - prefix_tokens,
            appended_total=request_tokens + notice_tokens,
            appended_suffix=notice_tokens,
            suffix_ratio=(replaced - prefix_tokens) / notice_tokens,
        )


class ProtectedFieldCheckModel(ForwardModel):
    """
    Cost per accepted task with and without a protected-field check at each compaction.

    Each of ``constraints`` is lost independently with ``loss_per_round`` in
    each of ``rounds``; a lost constraint makes the task unacceptable, so
    acceptance is ``p_intact (1 - eps)^(n R)``. A check that detects
    ``detection`` of losses and repairs them leaves ``eps (1 - detection)`` per
    round and adds ``check_cost`` per round to the attempt.
    """
    requires = ()
    produces = ProtectedFieldCheckResult

    def solve(self, constraints: int, loss_per_round: float, rounds: int, acceptance_if_intact: float,
              attempt_cost: Quantity, detection: float, check_cost_per_round: Quantity,
              survival_floor: float, target_survival: Optional[float] = None) -> ProtectedFieldCheckResult:
        n = constraints * rounds
        attempt = attempt_cost.to(_USD).magnitude
        s_no = calc_unchecked_success(1 - loss_per_round, n)
        p_no = acceptance_if_intact * s_no
        undetected = loss_per_round * (1 - detection)
        s_chk = calc_unchecked_success(1 - undetected, n)
        p_chk = acceptance_if_intact * s_chk
        attempt_chk = attempt + rounds * check_cost_per_round.to(_USD).magnitude
        out = dict(
            exposures=n,
            survival_unchecked=s_no,
            acceptance_unchecked=p_no,
            cost_per_accepted_unchecked=calc_cost_per_accepted(attempt, p_no),
            undetected_loss=undetected,
            survival_checked=s_chk,
            acceptance_checked=p_chk,
            attempt_cost_checked=attempt_chk,
            cost_per_accepted_checked=calc_cost_per_accepted(attempt_chk, p_chk),
            checked_rounds_below_floor=calc_first_horizon_below(
                calc_unchecked_success(1 - undetected, constraints), survival_floor),
        )
        if target_survival is not None:
            out.update(max_loss_for_target=calc_step_error_for_target(target_survival, n))
        return ProtectedFieldCheckResult(**out)

