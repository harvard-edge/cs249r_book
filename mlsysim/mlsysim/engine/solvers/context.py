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

from typing import Optional


from ..results import (
    AttentionDilutionResult,
    CompactionCadenceResult,
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

    def solve(
        self,
        window: int,
        output_reserve: int,
        system_tokens: int,
        tool_tokens: int,
        state_tokens: int,
        observation_per_turn: int,
        selected_observation: int,
        turns: int,
        digest_tokens: int,
    ) -> ContextBudgetResult:
        """Compare appending and selecting observations within a token budget.

        Parameters
        ----------
        window : int
            Positive total context-window capacity, in tokens.
        output_reserve : int
            Nonnegative reply reservation, in tokens, no larger than window.
        system_tokens : int
            Nonnegative serialized system-instruction count, in tokens.
        tool_tokens : int
            Nonnegative serialized tool-descriptor count, in tokens.
        state_tokens : int
            Nonnegative retained-state count, in tokens.
        observation_per_turn : int
            Positive count of observation tokens appended each turn.
        selected_observation : int
            Nonnegative observation tokens selected for each fixed-size request.
        turns : int
            Positive number of requests to account for.
        digest_tokens : int
            Positive tokens per digest, used for an alternative capacity calculation.

        Returns
        -------
        ContextBudgetResult
            Token budgets and per-turn request traces, integer overflow turns,
            cumulative token bills, and dimensionless ratios.

        Notes
        -----
        Equation: Observation budget is window - output_reserve - system_tokens
        - tool_tokens - state_tokens. Appending sums fixed + ``t*observation_per_turn``
        for t=1..turns; selection sends ``turns*(fixed`` + selected_observation).

        Assumptions: Counts are already serialized tokens. Fixed roles fit inside
        the input budget; selected requests are nonempty. The append bill counts
        all requested turns even when their sizes exceed the supplied window.

        Excludes: Cache reuse, tokenization across joins, retrieval quality,
        compaction cost, and automatic rejection or truncation of oversized requests.

        References
        ----------
        Derivation: Subtract reserved roles from a finite capacity, then sum an
        arithmetic progression for appending and a constant sequence for selection.
        The helpers in physics.context compute remaining capacity and integer fit.

        Examples
        --------
        >>> from mlsysim.solvers import ContextBudgetModel
        >>> result = ContextBudgetModel().solve(100, 10, 5, 5, 10, 10, 10, 3, 5)
        >>> result.append_bill > result.select_bill
        True
        """
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

    def solve(
        self,
        target_logit: float,
        background_logit: float,
        short_keys: int,
        long_keys: int,
        unrelated_keys: Optional[int] = None,
        lookalike_keys: Optional[int] = None,
        lookalike_logit: Optional[float] = None,
    ) -> AttentionDilutionResult:
        """Compare target attention mass under fixed-logit key populations.

        Parameters
        ----------
        target_logit : float
            Finite dimensionless logit of the target key.
        background_logit : float
            Finite dimensionless logit shared by ordinary competing keys.
        short_keys : int
            Positive total key count in the short case, including the target.
        long_keys : int
            Positive total key count in the long case, including the target.
        unrelated_keys : int, optional
            Nonnegative competitor count, excluding the target, for the look-alike case.
        lookalike_keys : int, optional
            Nonnegative look-alike competitor count; provide with unrelated_keys.
        lookalike_logit : float, optional
            Finite logit of each look-alike; required when both competitor counts are supplied.

        Returns
        -------
        AttentionDilutionResult
            Dimensionless target weights and optional equivalent-key counts
            under the supplied logit populations.

        Notes
        -----
        Equation: Target mass is exp(z_target) / (exp(z_target) + sum_j
        ``n_j*exp(z_j))``. Equivalent background keys equal ``n_look*exp(z_look-z_background)``.

        Assumptions: The logits stay fixed when competitors are added. Each member
        of a population has the same logit; counts are not learned.

        Excludes: Queries changing with context, positional effects, multiple heads,
        value aggregation, and any prediction of answer accuracy.

        References
        ----------
        Derivation: The softmax denominator sums one exponential per key. Grouping
        identical logits gives ``n_j*exp(z_j)``; equating denominator contributions
        yields equivalent-key count. Helpers live in physics.context.

        Examples
        --------
        >>> from mlsysim.solvers import AttentionDilutionModel
        >>> result = AttentionDilutionModel().solve(1.0, 0.0, 2, 8)
        >>> result.long_weight < result.short_weight
        True
        """
        out = dict(
            short_weight=calc_softmax_target_weight(target_logit, [(short_keys - 1, background_logit)]),
            long_weight=calc_softmax_target_weight(target_logit, [(long_keys - 1, background_logit)]),
        )
        if unrelated_keys is not None and lookalike_keys is not None:
            look = (lookalike_keys, lookalike_logit)
            out.update(
                weight_without_lookalikes=calc_softmax_target_weight(
                    target_logit, [(unrelated_keys, background_logit)]
                ),
                weight_with_lookalikes=calc_softmax_target_weight(
                    target_logit, [(unrelated_keys, background_logit), look]
                ),
                lookalike_equivalent_keys=calc_equivalent_key_count(lookalike_keys, lookalike_logit, background_logit),
                weight_half_unrelated=calc_softmax_target_weight(
                    target_logit, [(unrelated_keys / 2, background_logit), look]
                ),
                weight_no_unrelated=calc_softmax_target_weight(target_logit, [look]),
            )
        return AttentionDilutionResult(**out)


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

    def solve(
        self,
        turns: int,
        root_tokens: int,
        trunk_tokens: int,
        leaf_tokens: int,
        trunk_rewrites: int,
        input_price: Quantity,
        cached_fraction: float,
    ) -> PrefixLayoutBillResult:
        """Price repeated requests with root, trunk, and leaf cache zones.

        Parameters
        ----------
        turns : int
            Positive number of calls.
        root_tokens : int
            Nonnegative tokens in the unchanged root prefix.
        trunk_tokens : int
            Nonnegative tokens in the occasionally rewritten trunk.
        leaf_tokens : int
            Positive tokens in the per-call leaf.
        trunk_rewrites : int
            Number of trunk changes after the cold call, from zero through turns - 1.
        input_price : Quantity
            Nonnegative uncached price, convertible to USD/token; positive for cost ratios.
        cached_fraction : float
            Cached-input price as a fraction of uncached price, in [0, 1].

        Returns
        -------
        PrefixLayoutBillResult
            Token counts, dimensionless reuse and cost ratios, and USD-valued
            total and per-call bills for the two layouts.

        Notes
        -----
        Equation: Total input is ``turns*(root`` + trunk + leaf). Fresh prefill is
        root + trunk + leaf + ``trunk_rewrites*(trunk`` + leaf)
        + (turns - 1 - ``trunk_rewrites)*leaf``; cached input is the remainder.

        Assumptions: The first call is cold. Compatible cached state remains
        available, and each trunk rewrite invalidates all following tokens.

        Excludes: Cache eviction, output-token bills, latency, and differences in
        answer quality between layouts.

        References
        ----------
        Derivation: Partition calls into cold, trunk-rewrite, and leaf-only cases.
        Sum their newly computed tokens, then charge cached tokens at the stated
        price fraction. physics.context.calc_zoned_prefill_tokens owns the count.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import PrefixLayoutBillModel
        >>> result = PrefixLayoutBillModel().solve(3, 20, 10, 5, 0, Q_(1, "USD/token"), 0.5)
        >>> result.stable_cost < result.unstable_cost
        True
        """
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

    def solve(
        self,
        window: int,
        output_reserve: int,
        target: int,
        trigger: int,
        tokens_per_turn: int,
        turns: int,
        alternate_target: Optional[int] = None,
    ) -> CompactionCadenceResult:
        """Trace fixed-growth context between compaction watermarks.

        Parameters
        ----------
        window : int
            Positive context-window capacity, in tokens.
        output_reserve : int
            Nonnegative output-token reservation.
        target : int
            Nonnegative token count immediately after each compaction.
        trigger : int
            Token watermark above target that triggers a compaction.
        tokens_per_turn : int
            Positive token increment appended on every turn.
        turns : int
            Nonnegative number of appended turns to trace.
        alternate_target : int, optional
            Alternative nonnegative reset length below trigger; None omits its trace.

        Returns
        -------
        CompactionCadenceResult
            Integer cycle length and token traces, pass counts, overshoot,
            and whether the peak request plus reservation fits the window.

        Notes
        -----
        Equation: Cycle length is ceil((trigger - target)/tokens_per_turn).
        Peak length is target + ``cycle*tokens_per_turn``; passes equal turns//cycle.

        Assumptions: Compaction instantly resets length to target after the turn
        that reaches the trigger. The trigger is soft and can be overshot.

        Excludes: Summarizer runtime, fidelity loss, variable observation sizes,
        and interruptions while compaction runs.

        References
        ----------
        Derivation: The first integer increment count reaching the trigger is the
        ceiling of its distance from target divided by tokens_per_turn. Repeating
        that reset yields the trace in _trace and the cycle helper in physics.context.

        Examples
        --------
        >>> from mlsysim.solvers import CompactionCadenceModel
        >>> result = CompactionCadenceModel().solve(100, 10, 20, 60, 10, 8)
        >>> result.fits_window and result.passes > 0
        True
        """
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
            out.update(
                alternate_cycle_turns=alt,
                alternate_passes=turns // alt,
                alternate_lengths=_trace(alternate_target, trigger, tokens_per_turn, turns),
                frequency_ratio=cycle / alt,
            )
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

    def solve(
        self, request_tokens: int, prefix_tokens: int, superseded_tokens: int, marker_tokens: int, notice_tokens: int
    ) -> ContextInvalidationResult:
        """Count cache suffixes for three ways to retire stale text.

        Parameters
        ----------
        request_tokens : int
            Positive original request length, in tokens.
        prefix_tokens : int
            Nonnegative reusable prefix ending before the superseded passage.
        superseded_tokens : int
            Nonnegative passage length; prefix plus passage must fit the request.
        marker_tokens : int
            Nonnegative replacement-marker length.
        notice_tokens : int
            Positive length of a notice appended after the original request.

        Returns
        -------
        ContextInvalidationResult
            Total and newly prefilling suffix lengths in tokens for deletion,
            replacement, and append, plus a dimensionless suffix ratio.

        Notes
        -----
        Equation: Deletion leaves request_tokens - superseded_tokens; replacement
        adds marker_tokens. Their fresh suffix starts at prefix_tokens. Appending
        adds notice_tokens, all of which are new.

        Assumptions: Cached prefix validity ends exactly at the first edited token.
        The appended notice leaves every original token resident.

        Excludes: Semantic compliance with the notice, cache eviction, tokenization
        changes at edit boundaries, and the cost of editing.

        References
        ----------
        Derivation: Subtract removed spans and add inserted spans to obtain total
        length. Subtract the unchanged prefix to obtain work requiring fresh prefill.

        Examples
        --------
        >>> from mlsysim.solvers import ContextInvalidationModel
        >>> result = ContextInvalidationModel().solve(100, 20, 30, 5, 5)
        >>> result.removed_total < result.replaced_total < result.appended_total
        True
        """
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

    def solve(
        self,
        constraints: int,
        loss_per_round: float,
        rounds: int,
        acceptance_if_intact: float,
        attempt_cost: Quantity,
        detection: float,
        check_cost_per_round: Quantity,
        survival_floor: float,
        target_survival: Optional[float] = None,
    ) -> ProtectedFieldCheckResult:
        """Compare acceptance cost with checks at context compactions.

        Parameters
        ----------
        constraints : int
            Positive number of protected constraints exposed at each compaction.
        loss_per_round : float
            Independent loss probability per constraint per round, in (0, 1).
        rounds : int
            Positive number of compaction rounds.
        acceptance_if_intact : float
            Task acceptance probability conditional on intact constraints, in (0, 1].
        attempt_cost : Quantity
            Positive unchecked attempt cost, convertible to USD.
        detection : float
            Fraction of losses detected and repaired, in [0, 1); exact one is outside the horizon-ratio calculation.
        check_cost_per_round : Quantity
            Nonnegative cost of each check in USD.
        survival_floor : float
            Probability in (0, 1) used to locate the first unacceptable round count.
        target_survival : float, optional
            Desired intact probability in (0, 1); None omits the inverse loss budget.

        Returns
        -------
        ProtectedFieldCheckResult
            Dimensionless survival and acceptance probabilities, USD-valued
            scalar costs, exposure counts, and optional maximum loss probability.

        Notes
        -----
        Equation: With N=``constraints*rounds`` and loss e, survival is ``(1-e)**N``.
        Checking changes e to ``e*(1-detection)`` and adds ``rounds*check_cost_per_round``
        to attempt cost. Effective cost divides attempt cost by acceptance.

        Assumptions: Losses are independent; all detected losses are repaired.
        Every constraint must remain intact, and checking has no other side effects.

        Excludes: Correlated loss, imperfect repairs, false alarms, and measured
        quality improvements; detection and conditional acceptance are inputs.

        References
        ----------
        Derivation: Multiply survival probabilities across constraint-round
        exposures, then multiply by conditional acceptance. Expected repeated-attempt
        cost follows the geometric identity implemented in physics.agents.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import ProtectedFieldCheckModel
        >>> result = ProtectedFieldCheckModel().solve(2, 0.1, 3, 0.9, Q_(1, "USD"), 0.5, Q_(0.1, "USD"), 0.8)
        >>> result.survival_checked > result.survival_unchecked
        True
        """
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
                calc_unchecked_success(1 - undetected, constraints), survival_floor
            ),
        )
        if target_survival is not None:
            out.update(max_loss_for_target=calc_step_error_for_target(target_survival, n))
        return ProtectedFieldCheckResult(**out)
