"""Agent cost-accounting solvers: what one accepted task costs, and what each lever buys (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. Each model is arithmetic on
stated inputs: the fitted fall of a fixed capability's price, cost against
score on long-horizon trials, a cohort of one-attempt tasks under a tariff,
prefix-cache bills and the payback of one compaction, retry policies under
independent attempts, a serial episode under Amdahl's law and speculative
decoding, a tiered model cascade under one gate, admission against a token
quota, Kingman's queue price, renting tokens against owning capacity, a
conserved budget across children, architectures with a human fallback, the
payback of a one-time adaptation, per-workload levers on cost per accepted
task, measured coverage against independent trials, and one server swept
across concurrency. None models a particular provider's billing system.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence


from ..results import (
    AdaptationPaybackResult,
    ArchitectureChoiceResult,
    CohortAttemptCostResult,
    CompactionPaybackResult,
    ConservedBudgetResult,
    DedicatedBreakEvenResult,
    KingmanQueueResult,
    ModelCascadeResult,
    PrefixCacheBillResult,
    RetryPolicyResult,
    SerialEpisodeResult,
    SpeculativeEpisodeResult,
    TokenPriceComparisonResult,
    TrajectoryAdmissionResult,
    WorkloadLeverResult,
    WorkloadLeverRow,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...core.units import USD as _USD, ureg
from ...physics.agents import calc_independent_success
from ...physics.serving import calc_kingman_wait_ratio, calc_speculative_round_speedup


_MTOK = 1_000_000


def _usd(q) -> float:
    return float(q.to("USD").magnitude)


def _per_mtok(price: Quantity) -> float:
    """Dollars per million tokens, as a plain number."""
    return float(price.to("USD / Mtoken").magnitude)


class CohortAttemptCostModel(ForwardModel):
    """
    One configuration's cohort: one bounded attempt per task, all charges settled.

    The attempt costs its turns of model tokens and environment plus one
    assessment; the cohort cost per accepted task divides every attempt's
    cost by the accepted count, and triage charges each rejected attempt.
    """

    requires = ()
    produces = CohortAttemptCostResult

    def solve(
        self,
        tasks: int,
        accepted: int,
        input_price: Quantity,
        output_price: Quantity,
        turns: int,
        input_tokens_per_turn: int,
        output_tokens_per_turn: int,
        environment_per_turn: Quantity,
        assessment_per_attempt: Quantity,
        triage_per_rejected: Quantity,
    ) -> CohortAttemptCostResult:
        """Calculate settled cohort spend per accepted task.

        Parameters
        ----------
        tasks : int
            Number of initiated tasks; positive integer.
        accepted : int
            Accepted tasks, with 0 < accepted <= tasks.
        input_price : Quantity
            Fresh input tariff, as a quantity convertible to USD/Mtoken.
        output_price : Quantity
            Output tariff, as a quantity convertible to USD/Mtoken.
        turns : int
            Number of model turns; positive integer.
        input_tokens_per_turn : int
            Fresh input count per turn, in tokens.
        output_tokens_per_turn : int
            Generated count per turn, in tokens.
        environment_per_turn : Quantity
            Environment charge per turn, convertible to USD.
        assessment_per_attempt : Quantity
            Assessment charge per attempt, convertible to USD.
        triage_per_rejected : Quantity
            Human triage charge per rejected attempt, convertible to USD.

        Returns
        -------
        CohortAttemptCostResult
            Per-attempt and cohort USD amounts as floats, rejected-task counts,
            USD per accepted task, and rejected-to-accepted count ratio.

        Notes
        -----
        Equation: C = ``turns*(input_tokens*p_in`` + ``output_tokens*p_out`` + environment) + assessment; cohort cost per accepted = ``tasks*C/accepted``.

        Assumptions: Every task makes one bounded attempt with the same token budget; accepted is positive and no greater than tasks.

        Excludes: Taxes, volume discounts, correlated retry costs, and time-varying tariffs.

        References
        ----------
        Derivation: Sum token charges and environment charges over turns, then add the
        once-per-attempt assessment. Multiply by initiated tasks before dividing
        by accepted tasks so rejected attempts remain in the numerator.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = CohortAttemptCostModel().solve(10, 5, Q_('1 USD/Mtoken'), Q_('2 USD/Mtoken'), 1, 100, 10, Q_('1 USD'), Q_('1 USD'), Q_('1 USD'))
        >>> r.rejected == 5
        True
        """
        p_in, p_out = _per_mtok(input_price), _per_mtok(output_price)
        env, assess, triage = _usd(environment_per_turn), _usd(assessment_per_attempt), _usd(triage_per_rejected)
        model = turns * (input_tokens_per_turn * p_in + output_tokens_per_turn * p_out) / 1e6
        attempt = model + turns * env + assess
        total = tasks * attempt
        rejected = tasks - accepted
        total_h = total + rejected * triage
        return CohortAttemptCostResult(
            model_cost=model,
            environment_cost=turns * env,
            attempt_cost=attempt,
            cohort_cost=total,
            rejected=rejected,
            cost_per_accepted=total / accepted,
            rejected_per_accepted=rejected / accepted,
            cohort_cost_with_triage=total_h,
            cost_per_accepted_with_triage=total_h / accepted,
        )

    @staticmethod
    def compare(
        baseline: CohortAttemptCostResult,
        alternative: CohortAttemptCostResult,
        baseline_accepted: int,
        alternative_accepted: int,
    ) -> TokenPriceComparisonResult:
        """Compare cost and acceptance changes between equivalent cohorts.

        Parameters
        ----------
        baseline : CohortAttemptCostResult
            Baseline USD costs; denominators must be positive.
        alternative : CohortAttemptCostResult
            Alternative USD costs under equivalent task definitions.
        baseline_accepted : int
            Positive baseline accepted count.
        alternative_accepted : int
            Alternative accepted count over the same cohort size.

        Returns
        -------
        TokenPriceComparisonResult
            Original cohort results and dimensionless alternative/baseline cost
            ratios and relative model, attempt, and acceptance reductions.

        Notes
        -----
        Equation: cost_ratio = alternative.cost_per_accepted / baseline.cost_per_accepted; cuts = 1-alternative/baseline.

        Assumptions: Equivalent task mix, acceptance gate, and cohort sizes.

        Excludes: Statistical uncertainty and causal attribution.

        References
        ----------
        Derivation: Normalize each alternative cost or count by its corresponding baseline. Subtract from one for a relative reduction.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = CohortAttemptCostModel().solve(10, 5, Q_('1 USD/Mtoken'), Q_('2 USD/Mtoken'), 1, 100, 10, Q_('1 USD'), Q_('1 USD'), Q_('1 USD'))
        >>> CohortAttemptCostModel.compare(r, r, 5, 5).cost_ratio == 1
        True
        """
        return TokenPriceComparisonResult(
            baseline=baseline,
            alternative=alternative,
            cost_ratio=alternative.cost_per_accepted / baseline.cost_per_accepted,
            cost_ratio_with_triage=alternative.cost_per_accepted_with_triage / baseline.cost_per_accepted_with_triage,
            model_cut=1 - alternative.model_cost / baseline.model_cost,
            attempt_cut=1 - alternative.attempt_cost / baseline.attempt_cost,
            acceptance_cut=1 - alternative_accepted / baseline_accepted,
        )


class PrefixCacheBillModel(ForwardModel):
    """
    An append-only trajectory billed at the fresh rate, and with ideal prefix reuse.

    The logical input is ``H s0 + delta H (H - 1) / 2``; ideal reuse bills only
    ``s0 + (H - 1) delta`` at the input rate and the rest at the cached rate,
    so caching scales the quadratic term down without removing it. A miss on
    ``miss_fraction`` of the eligible prefix bills it at the full rate.
    """

    requires = ()
    produces = PrefixCacheBillResult

    def solve(
        self,
        turns: int,
        first_prompt: int,
        appended: int,
        output_per_turn: int,
        input_price: Quantity,
        cached_fraction: float,
        output_price: Quantity,
        miss_fraction: float,
    ) -> PrefixCacheBillResult:
        """Price append-only prompts with ideal prefix reuse.

        Parameters
        ----------
        turns : int
            Number of model turns; positive integer.
        first_prompt : int
            Initial prompt length, in tokens.
        appended : int
            Tokens appended between consecutive calls.
        output_per_turn : int
            Generated tokens per turn.
        input_price : Quantity
            Fresh input tariff, as a quantity convertible to USD/Mtoken.
        cached_fraction : float
            Cached-input price divided by fresh-input price, in [0,1].
        output_price : Quantity
            Output tariff, as a quantity convertible to USD/Mtoken.
        miss_fraction : float
            Fraction of reusable prefix charged fresh, in [0,1].

        Returns
        -------
        PrefixCacheBillResult
            Sent and fresh token counts, cached and uncached USD quantity bills,
            dimensionless cost ratios, and per-turn USD amounts as floats.

        Notes
        -----
        Equation: sent = ``H*s0`` + ``delta*H*(H-1)/2``; fresh = s0 + ``(H-1)*delta``. Cached input costs ``p_in*(fresh`` + ``cached_fraction*(sent-fresh))``.

        Assumptions: Positive turn count; cached_fraction and miss_fraction lie in [0,1]; only unchanged prefixes are eligible.

        Excludes: Cache capacity, eviction, cache-write charges, and changes in generation latency.

        References
        ----------
        Derivation: Sum the arithmetic progression of prompt lengths. Each token is
        fresh on its first appearance; subsequent appearances form the cached
        portion. Apply the two tariffs separately, then add generated-token cost.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = PrefixCacheBillModel().solve(2, 100, 10, 5, Q_('1 USD/Mtoken'), 0.1, Q_('2 USD/Mtoken'), 0.0)
        >>> r.sent_tokens == 210 and r.fresh_tokens == 110
        True
        """
        p_in, p_out = _per_mtok(input_price) * _USD, _per_mtok(output_price) * _USD
        s0, delta, k, cs = first_prompt, appended, output_per_turn, cached_fraction
        sent = turns * s0 + delta * turns * (turns - 1) // 2
        fresh = s0 + delta * (turns - 1)
        in_nocache = sent * p_in / 1_000_000
        in_cache = (fresh + cs * (sent - fresh)) * p_in / 1_000_000
        out_cost = turns * k * p_out / 1_000_000
        total_nocache, total_cache = in_nocache + out_cost, in_cache + out_cost
        cached_reads = cs * (sent - fresh) * p_in / 1_000_000
        miss = miss_fraction * (sent - fresh) * (1 - cs) * p_in / 1_000_000
        pin, pc, pout = _per_mtok(input_price) / 1e6, cs * _per_mtok(input_price) / 1e6, _per_mtok(output_price) / 1e6
        lengths = [s0 + (t - 1) * delta for t in range(1, turns + 1)]
        per_no = [L * pin + k * pout for L in lengths]
        per_re = [
            (s0 if t == 1 else delta) * pin + (0 if t == 1 else L - delta) * pc + k * pout
            for t, L in enumerate(lengths, 1)
        ]
        return PrefixCacheBillResult(
            sent_tokens=sent,
            fresh_tokens=fresh,
            input_uncached=in_nocache,
            input_cached=in_cache,
            output_cost=out_cost,
            total_uncached=total_nocache,
            total_cached=total_cache,
            cache_ratio=float((total_nocache / total_cache).magnitude),
            cached_reread_cost=cached_reads,
            reread_share=float((cached_reads / total_cache).to(_USD / _USD).magnitude),
            miss_cost=miss,
            miss_share=float((miss / total_cache).magnitude),
            per_turn_uncached=per_no,
            per_turn_cached=per_re,
        )


class CompactionPaybackModel(ForwardModel):
    """
    One compaction priced against keeping the transcript, under one tariff.

    Compacting at call ``n_c`` replaces ``S`` tokens of context with a
    ``S_c``-token summary that costs ``K`` output tokens and is resent fresh,
    so the compacting call costs ``p_out K + p_in S_c`` more; every later call
    reads ``S - S_c`` fewer cached tokens. Compaction has paid after
    ``n* = 1 + ceil(upfront / per-call saving)`` calls.
    """

    requires = ()
    produces = CompactionPaybackResult

    def solve(
        self,
        turns: int,
        first_prompt: int,
        appended: int,
        input_price: Quantity,
        cached_fraction: float,
        output_price: Quantity,
        compact_turn: int,
        compact_to: int,
        summary_tokens: int,
    ) -> CompactionPaybackResult:
        """Compute when a context summary repays its token bill.

        Parameters
        ----------
        turns : int
            Number of model turns; positive integer.
        first_prompt : int
            Initial prompt length, in tokens.
        appended : int
            Tokens appended between consecutive calls.
        input_price : Quantity
            Fresh input tariff, as a quantity convertible to USD/Mtoken.
        cached_fraction : float
            Cached-input price divided by fresh-input price, in [0,1].
        output_price : Quantity
            Output tariff, as a quantity convertible to USD/Mtoken.
        compact_turn : int
            One-based compaction call index within the trajectory.
        compact_to : int
            Summary context length in tokens, smaller than the replaced context.
        summary_tokens : int
            Generated summary tokens billed at the output tariff.

        Returns
        -------
        CompactionPaybackResult
            Context and remaining-call counts, USD quantity costs and savings,
            integer break-even call count, and per-horizon net USD floats.

        Notes
        -----
        Equation: upfront = ``p_out*summary_tokens`` + ``p_in*compact_to``; saving = ``(context_at_compaction-compact_to)*cached_fraction*p_in``; crossing = 1+ceil(upfront/saving).

        Assumptions: The summary preserves the required task state; compact_turn is within the trajectory; positive cached price and positive token reduction.

        Excludes: Changes in answer quality, latency, and later summary regeneration.

        References
        ----------
        Derivation: The summary generation and first fresh summary read are upfront
        charges. Each later call avoids the same cached-context bill. Divide
        upfront cost by this saving, round up, and include the compaction call.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = CompactionPaybackModel().solve(10, 1000, 10, Q_('1 USD/Mtoken'), 0.5, Q_('2 USD/Mtoken'), 2, 100, 10)
        >>> r.per_call_saving > Q_('0 USD')
        True
        """
        p_in, p_out = _per_mtok(input_price) * _USD, _per_mtok(output_price) * _USD
        s0, delta, cs = first_prompt, appended, cached_fraction
        ctx_at = s0 + (compact_turn - 1) * delta
        rest = turns - compact_turn + 1
        first_keep = (delta + (ctx_at - delta) * cs) * p_in / 1_000_000
        first_comp = compact_to * p_in / 1_000_000
        summary_cost = first_keep + summary_tokens * p_out / 1_000_000
        upfront = summary_cost + first_comp - first_keep
        saving = (ctx_at - compact_to) * cs * p_in / 1_000_000
        n_star = 1 + math.ceil((upfront / saving).magnitude)
        keep_sum = rest * ctx_at + delta * rest * (rest - 1) // 2
        keep_cost = (rest * delta + cs * (keep_sum - rest * delta)) * p_in / 1_000_000
        comp_sum = rest * compact_to + delta * rest * (rest - 1) // 2
        comp_fresh = compact_to + (rest - 1) * delta
        comp_cost = (comp_fresh + cs * (comp_sum - comp_fresh)) * p_in / 1_000_000
        net = {n: float((upfront - (n - 1) * saving).to(_USD).magnitude) for n in range(1, turns + 1)}
        return CompactionPaybackResult(
            context_at_compaction=ctx_at,
            remaining_calls=rest,
            summary_cost=summary_cost,
            upfront=upfront,
            per_call_saving=saving,
            breakeven_calls=n_star,
            keep_cost=keep_cost,
            compact_cost=comp_cost,
            compact_total=comp_cost + summary_cost,
            gap_before=upfront - (n_star - 2) * saving,
            saving_at=(n_star - 1) * saving - upfront,
            net_by_calls=net,
        )


class RetryPolicyModel(ForwardModel):
    """
    Bounded attempt policies under independent attempts at a fixed all-in cost.

    Sequential retry with early stopping covers ``1 - (1 - p)^k`` of tasks in
    ``coverage / p`` attempts on average, so its cost per accepted task is
    ``c / p`` whatever the cap; a batch of ``k`` pays for all ``k``.
    """

    requires = ()
    produces = RetryPolicyResult

    def solve(
        self,
        pass_probability: float,
        attempt_cost: Quantity,
        max_attempts: int,
        strong_pass_probability: float,
        strong_attempt_cost: Quantity,
    ) -> RetryPolicyResult:
        """Compare bounded sequential retries with a paid batch.

        Parameters
        ----------
        pass_probability : float
            Independent per-attempt success probability in (0,1].
        attempt_cost : Quantity
            All-in attempt charge, convertible to USD.
        max_attempts : int
            Positive integer cap on attempts.
        strong_pass_probability : float
            Comparison configuration success probability in (0,1].
        strong_attempt_cost : Quantity
            Comparison configuration charge, convertible to USD.

        Returns
        -------
        RetryPolicyResult
            Coverage probability, expected attempt count, USD quantity spends
            and costs per accepted task, and dimensionless batch premium.

        Notes
        -----
        Equation: coverage = ``1-(1-p)**k``; expected attempts = ``sum((1-p)**j`` for j in range(k)); sequential spend = ``cost*expected`` attempts.

        Assumptions: Independent trials with constant success probability in (0,1]; positive max_attempts and positive strong_pass_probability.

        Excludes: Correlation across retries, changing attempt difficulty, cancellation, and latency.

        References
        ----------
        Derivation: Trial j is reached only if the preceding trials fail. Sum those
        reach probabilities for expected sequential attempts. A batch pays for
        every attempt, even when an earlier member would have succeeded.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = RetryPolicyModel().solve(0.5, Q_('1 USD'), 2, 0.8, Q_('2 USD'))
        >>> r.expected_attempts == 1.5
        True
        """
        p, c, k = pass_probability, attempt_cost.to(_USD), max_attempts
        coverage = calc_independent_success(p, k)
        attempts = 0.0
        for j in range(k):
            attempts += (1 - p) ** j
        seq_cost, par_cost = c * attempts, k * c
        return RetryPolicyResult(
            coverage=coverage,
            expected_attempts=attempts,
            sequential_cost=seq_cost,
            batch_cost=par_cost,
            sequential_per_accepted=seq_cost / coverage,
            batch_per_accepted=par_cost / coverage,
            strong_per_accepted=strong_attempt_cost.to(_USD) / strong_pass_probability,
            batch_premium=k / attempts,
        )


class SerialEpisodeModel(ForwardModel):
    """
    One serial episode of disjoint intervals; a faster model shortens only its own.

    ``T' = T_other + T_model / s`` (Amdahl's law); instant model calls bound
    the episode speedup at ``1 / (1 - f)``.
    """

    requires = ()
    produces = SerialEpisodeResult

    def solve(
        self,
        model_intervals: Sequence[float],
        tests: float,
        lifecycle_intervals: Sequence[float],
        runtime_intervals: Sequence[float],
        model_speedup: float,
    ) -> SerialEpisodeResult:
        """Apply a model-only speedup to a serial episode.

        Parameters
        ----------
        model_intervals : Sequence[float]
            Disjoint model execution durations in seconds.
        tests : float
            Test duration in seconds.
        lifecycle_intervals : Sequence[float]
            Disjoint startup and shutdown durations in seconds.
        runtime_intervals : Sequence[float]
            Other disjoint runtime durations in seconds.
        model_speedup : float
            Positive multiplicative acceleration of model intervals.

        Returns
        -------
        SerialEpisodeResult
            Component, total, changed, and saved durations as second quantities;
            model fraction and achieved/limiting speedups are dimensionless.

        Notes
        -----
        Equation: T_new = T_other + T_model/model_speedup; speedup = T_total/T_new.

        Assumptions: Nonnegative disjoint intervals; positive speedup and positive non-model time for the finite limiting ratio.

        Excludes: Overlap between intervals, queueing, and changes to non-model work.

        References
        ----------
        Derivation: Partition elapsed time into model and non-model intervals. Divide
        only the model sum by the supplied acceleration; leave other intervals
        unchanged. Zero model time gives the limiting episode speedup.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = SerialEpisodeModel().solve([2.0], 1.0, [1.0], [], 2.0)
        >>> r.changed_time == Q_('3 s')
        True
        """
        s = ureg.second
        model = sum(model_intervals)
        lifecycle = sum(lifecycle_intervals)
        runtime = sum(runtime_intervals)
        other = tests + lifecycle + runtime
        total = model + other
        changed = other + model / model_speedup
        return SerialEpisodeResult(
            model_time=model * s,
            tests_time=tests * s,
            lifecycle_time=lifecycle * s,
            runtime_time=runtime * s,
            other_time=other * s,
            total_time=total * s,
            model_fraction=model / total,
            changed_time=changed * s,
            saving=(total - changed) * s,
            speedup=total / changed,
            limiting_speedup=total / other,
            tests_half_saving=(tests / 2) * s,
        )


class SpeculativeEpisodeModel(ForwardModel):
    """
    Speculative decoding's decode-stream speedup, composed with a serial episode.

    Treats the episode's model time as pure decode (an optimistic bound):
    ``T' = T [(1 - f) + f / S]`` with ``S = (m + 1) / (gamma r + 1)``.
    """

    requires = ()
    produces = SpeculativeEpisodeResult

    def solve(
        self,
        total_time: float,
        decode_fraction: float,
        draft_length: int,
        draft_cost_ratio: float,
        accepted_high: float,
        accepted_low: float,
    ) -> SpeculativeEpisodeResult:
        """Propagate decode speedup into serial episode time.

        Parameters
        ----------
        total_time : float
            Baseline episode duration in seconds.
        decode_fraction : float
            Fraction of episode spent decoding, in [0,1).
        draft_length : int
            Positive number of drafted tokens per round.
        draft_cost_ratio : float
            Draft-token cost relative to one target step, nonnegative.
        accepted_high : float
            Higher assumed mean accepted draft tokens per round.
        accepted_low : float
            Lower assumed mean accepted draft tokens per round.

        Returns
        -------
        SpeculativeEpisodeResult
            Decode speedups, accepted-token break-even count, episode times
            as second quantities, and dimensionless episode speedup bounds.

        Notes
        -----
        Equation: S = ``(accepted+1)/(draft_length*draft_cost_ratio+1)``; T_new = ``T*((1-f)+f/S)``.

        Assumptions: Constant draft costs and accepted-token means; decode_fraction in [0,1); accepted counts between zero and draft_length.

        Excludes: Prefill acceleration, variable-context memory costs, and overlap with other work.

        References
        ----------
        Derivation: A speculative round spends one target step plus the draft-step
        budget and returns accepted drafts plus a bonus token. Divide returned
        tokens by normalized round cost, then accelerate only the decode share.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = SpeculativeEpisodeModel().solve(10.0, 0.5, 4, 0.1, 3.0, 1.0)
        >>> r.episode_time_high < r.episode_time_low
        True
        """
        s = ureg.second
        f = decode_fraction
        s_high = calc_speculative_round_speedup(accepted_high, draft_length, draft_cost_ratio)
        s_low = calc_speculative_round_speedup(accepted_low, draft_length, draft_cost_ratio)
        t_high = total_time * ((1 - f) + f / s_high)
        t_low = total_time * ((1 - f) + f / s_low)
        return SpeculativeEpisodeResult(
            break_even_accepted=draft_length * draft_cost_ratio,
            speedup_high=s_high,
            speedup_low=s_low,
            episode_time_high=t_high * s,
            episode_time_low=t_low * s,
            episode_speedup_high=total_time / t_high,
            episode_speedup_bound=1 / (1 - f),
        )


class ModelCascadeModel(ForwardModel):
    """
    A tiered cascade under one acceptance gate, priced over a completed cohort.

    Every reached tier pays its call and the gate; the cascade's cost per
    accepted task divides the whole spend by every tier's exits. Optionally a
    harm per false accept at the first tier, whose break-even rate erases the
    cascade's saving over the top tier alone.
    """

    requires = ()
    produces = ModelCascadeResult

    def solve(
        self,
        tasks: int,
        call_costs: Sequence[float],
        gate_cost: float,
        call_times: Sequence[float],
        gate_time: float,
        accepted_by_tier: Sequence[int],
        direct_accepted: int,
        false_accept_rate: Optional[float] = None,
        harm_per_false_accept: Optional[float] = None,
    ) -> ModelCascadeResult:
        """Price a completed cohort through sequential model tiers.

        Parameters
        ----------
        tasks : int
            Number of initiated tasks; positive integer.
        call_costs : Sequence[float]
            Aligned tier call costs in USD per call.
        gate_cost : float
            Acceptance-check cost in USD per reached tier.
        call_times : Sequence[float]
            Aligned tier call durations in seconds.
        gate_time : float
            Check duration in seconds per reached tier.
        accepted_by_tier : Sequence[int]
            Counts exiting each tier, bounded by the cohort reaching it.
        direct_accepted : int
            Positive number accepted when using the final tier alone.
        false_accept_rate : Optional[float]
            Optional fraction of first-tier accepts that are false, in [0,1].
        harm_per_false_accept : Optional[float]
            Optional harm cost in USD per false first-tier accept.

        Returns
        -------
        ModelCascadeResult
            Reached/accepted counts, conditional acceptance fractions, USD
            cost floats, mean and worst-path seconds, and optional USD harm costs.

        Notes
        -----
        Equation: reached[j+1] = reached[j]-accepted[j]; spend = sum(reached[j]*(call_cost[j]+gate_cost)); cost per accepted = spend/sum(accepted).

        Assumptions: Each acceptance exits the cascade; aligned tier sequences; positive reached and accepted denominators.

        Excludes: Unobserved acceptance quality, correlated latency, and asynchronous tier overlap.

        References
        ----------
        Derivation: Only unresolved tasks reach the next tier. Multiply each tier
        charge and duration by its reached population, then normalize spend by
        all accepted exits and elapsed work by all initiated tasks.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = ModelCascadeModel().solve(10, [1.0, 3.0], 0.1, [1.0, 2.0], 0.1, [5, 4], 9)
        >>> r.reached == [10, 5]
        True
        """
        n = len(call_costs)
        reached = [tasks]
        for j in range(n - 1):
            reached.append(reached[-1] - accepted_by_tier[j])
        spend, elapsed = 0.0, 0.0
        conditional, reaches = [], []
        for j in range(n):
            conditional.append(accepted_by_tier[j] / reached[j])
            reaches.append(reached[j] / tasks)
            spend += reached[j] * (call_costs[j] + gate_cost)
            elapsed += reached[j] * (call_times[j] + gate_time)
        accepted = sum(accepted_by_tier)
        per_accepted = spend / accepted
        direct_spend = tasks * (call_costs[-1] + gate_cost)
        direct_per = direct_spend / direct_accepted
        toll = 0.0
        for j in range(n - 1):
            toll += call_costs[j]
        toll += (n - 1) * gate_cost
        out = dict(
            reached=reached,
            conditional_acceptance=conditional,
            reach_shares=reaches,
            accepted=accepted,
            unresolved=tasks - accepted,
            spend=spend,
            per_initiated=spend / tasks,
            per_accepted=per_accepted,
            direct_spend=direct_spend,
            direct_per_accepted=direct_per,
            mean_time=elapsed / tasks,
            all_tiers_time=sum(call_times) + n * gate_time,
            direct_time=call_times[-1] + gate_time,
            direct_ratio=direct_per / per_accepted,
            lower_tier_toll=toll,
            saving_per_accepted=direct_per - per_accepted,
        )
        if harm_per_false_accept is not None:
            saving = direct_per - per_accepted
            out["false_accept_breakeven"] = saving * accepted / (accepted_by_tier[0] * harm_per_false_accept)
            if false_accept_rate is not None:
                harm = accepted_by_tier[0] * false_accept_rate * harm_per_false_accept
                out.update(harm_cost=harm, per_accepted_with_harm=(spend + harm) / accepted)
        return ModelCascadeResult(**out)


class TrajectoryAdmissionModel(ForwardModel):
    """
    Admitted trajectories against a token quota, in the mean.

    Little's law gives the trajectories present; the call and token rates
    scale with admissions. At fixed goodput, admissions scale as ``1 / alpha``,
    so a relative acceptance drop of ``1 - quota_share`` fills the quota.
    """

    requires = ()
    produces = TrajectoryAdmissionResult

    def solve(
        self,
        arrival: Quantity,
        residence: Quantity,
        calls: int,
        tokens_per_call: int,
        quota: Quantity,
        acceptance_drop: Optional[float] = None,
    ) -> TrajectoryAdmissionResult:
        """Compare mean trajectory traffic with a token quota.

        Parameters
        ----------
        arrival : Quantity
            Mean trajectory arrival rate, convertible to inverse seconds.
        residence : Quantity
            Mean trajectory residence time, convertible to seconds.
        calls : int
            Positive calls per trajectory.
        tokens_per_call : int
            Positive token count per call.
        quota : Quantity
            Positive token-count rate in inverse seconds; token counts are dimensionless here.
        acceptance_drop : Optional[float]
            Optional relative acceptance reduction, in [0,1).

        Returns
        -------
        TrajectoryAdmissionResult
            Mean concurrent trajectories, call and demand rates in the supplied
            arrival units, tokens per trajectory, and dimensionless quota shares.

        Notes
        -----
        Equation: concurrent = ``arrival*residence``; demand = ``arrival*calls*tokens_per_call``; quota share = demand/quota.

        Assumptions: Stationary mean arrivals and fixed work per trajectory; positive quota and tokens per trajectory; acceptance_drop in [0,1).

        Excludes: Burst admission, service-time variability, and enforcement of provider quotas.

        References
        ----------
        Derivation: Little's-law occupancy multiplies mean admission rate by residence
        time. Each admission induces a fixed number of calls and tokens; their
        product with admission rate is the mean quota demand.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = TrajectoryAdmissionModel().solve(Q_('1 / s'), Q_('2 s'), 2, 10, Q_('100 / s'))
        >>> r.concurrent == 2
        True
        """
        concurrent = (arrival * residence).to("").magnitude
        call_rate = arrival * calls
        tokens = calls * tokens_per_call
        demand = arrival * tokens
        share = (demand / quota).to("").magnitude
        return TrajectoryAdmissionResult(
            concurrent=concurrent,
            call_rate=call_rate,
            tokens_per_trajectory=tokens,
            demand=demand,
            quota_share=share,
            arrival_ceiling=quota / tokens,
            acceptance_margin=1 - share,
            quota_share_after_drop=None if acceptance_drop is None else share / (1 - acceptance_drop),
        )


class KingmanQueueModel(ForwardModel):
    """
    Kingman's mean wait for one server at two utilizations, and its holding price.

    ``E[W] = (c_a^2 + c_s^2) / 2 * rho / (1 - rho) * E[S]``; each second of
    wait holds resources at ``holding_rate`` dollars.
    """

    requires = ()
    produces = KingmanQueueResult

    def solve(
        self, service: Quantity, arrival_scv: float, service_scv: float, low: float, high: float, holding_rate: float
    ) -> KingmanQueueResult:
        """Estimate mean queue wait and its resource holding cost.

        Parameters
        ----------
        service : Quantity
            Positive mean service duration, convertible to seconds.
        arrival_scv : float
            Squared coefficient of variation of interarrival times, nonnegative.
        service_scv : float
            Squared coefficient of variation of service times, nonnegative.
        low : float
            Lower utilization in (0,1).
        high : float
            Higher utilization in (0,1).
        holding_rate : float
            Holding cost in USD per second of waiting.

        Returns
        -------
        KingmanQueueResult
            Mean wait quantities in service-time units, holding costs as USD
            floats, and dimensionless variability, utilization, and wait ratios.

        Notes
        -----
        Equation: W = ``service*(arrival_scv+service_scv)/2*rho/(1-rho)``; holding cost = ``W_seconds*holding_rate``.

        Assumptions: Single-server stationary renewal approximation; positive service and variability; 0 < low, high < 1.

        Excludes: Tail latency, transient overload, and multiple-server scheduling.

        References
        ----------
        Derivation: The Kingman single-server approximation multiplies mean service
        time by a variability factor and rho/(1-rho). Convert the resulting wait
        to seconds before applying the holding tariff. This is an approximation,
        not an exact queueing identity; see calc_kingman_wait_ratio.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = KingmanQueueModel().solve(Q_('1 s'), 1.0, 1.0, 0.5, 0.75, 1.0)
        >>> r.wait_high > r.wait_low
        True
        """
        v = (arrival_scv + service_scv) / 2
        wait_low = calc_kingman_wait_ratio(low, arrival_scv, service_scv) * service
        wait_high = calc_kingman_wait_ratio(high, arrival_scv, service_scv) * service
        return KingmanQueueResult(
            variability=v,
            wait_low=wait_low,
            wait_high=wait_high,
            holding_low=wait_low.to("second").magnitude * holding_rate,
            holding_high=wait_high.to("second").magnitude * holding_rate,
            wait_ratio=(wait_high / wait_low).to("").magnitude,
            throughput_gain=high / low - 1,
            unit_wait_utilization=1 / (1 + v),
        )


class DedicatedBreakEvenModel(ForwardModel):
    """
    Renting tokens from an API against owning capacity with a fixed hourly cost.

    The API's price per output token carries its input mix; owning pays a
    fixed ``F`` per hour plus ``v`` per output token. The costs tie at load
    ``u* = F / (3600 R (P_eff - v))`` of the reference rate ``R``.
    """

    requires = ()
    produces = DedicatedBreakEvenResult

    @staticmethod
    def hourly_costs(load: float, reference_rate: Quantity, api_price: float, fixed: float, variable: float):
        """Price one hour of API usage and dedicated capacity.

        Parameters
        ----------
        load : float
            Delivered reference-capacity fraction, in [0,1].
        reference_rate : Quantity
            Positive token-count rate, convertible to inverse seconds.
        api_price : float
            Effective API price in USD per million output tokens.
        fixed : float
            Fixed ownership charge in USD per hour.
        variable : float
            Variable ownership price in USD per million output tokens.

        Returns
        -------
        tuple of float
            API bill followed by dedicated-capacity bill, both in USD for one hour.

        Notes
        -----
        Equation: tokens = ``load*reference_rate*one_hour``; API = ``tokens*api_price/1e6``; dedicated = ``fixed+tokens*variable/1e6``.

        Assumptions: Constant load and tariff for one hour.

        Excludes: Quality differences, autoscaling delays, and extra usage charges.

        References
        ----------
        Derivation: Integrate a constant count rate over an hour, multiply by per-token price, then add fixed ownership charges.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> DedicatedBreakEvenModel.hourly_costs(0.0, Q_('100 / s'), 2.0, 1.0, 0.1)
        (0.0, 1.0)
        """
        tokens = load * (reference_rate * ureg.hour).to("").magnitude
        return tokens * api_price / 1_000_000, fixed + tokens * variable / 1_000_000

    def solve(
        self,
        input_per_output: float,
        cached_fraction: float,
        input_price: float,
        cached_price: float,
        output_price: float,
        fixed_per_hour: float,
        variable_price: float,
        reference_rate: Quantity,
        load: float,
    ) -> DedicatedBreakEvenResult:
        """Find the load where owned capacity matches API cost.

        Parameters
        ----------
        input_per_output : float
            Nonnegative input-token count per output token.
        cached_fraction : float
            Fraction of input tokens served from cache, in [0,1].
        input_price : float
            Nonnegative input price in USD per million tokens.
        cached_price : float
            Cached input price in USD per million tokens.
        output_price : float
            Nonnegative output price in USD per million tokens.
        fixed_per_hour : float
            Fixed ownership charge in USD per hour.
        variable_price : float
            Variable ownership charge in USD per million output tokens.
        reference_rate : Quantity
            Reference output-token count rate, convertible to inverse seconds.
        load : float
            Delivered fraction of reference capacity, in [0,1].

        Returns
        -------
        DedicatedBreakEvenResult
            Effective prices in USD per million tokens, hourly token counts,
            hourly USD cost floats, and dimensionless crossing loads.

        Notes
        -----
        Equation: API effective price = output_price + ``input_per_output*((1-cached_fraction)*input_price+cached_fraction*cached_price)``; crossing = ``fixed_per_hour/(reference_hour_tokens*(API-variable_price)/1e6)``.

        Assumptions: Fixed hourly ownership bill; proportional token cost; cached_fraction in [0,1]; effective API price exceeds variable_price.

        Excludes: Autoscaling delays, capacity acquisition, utilization variance, and quality differences.

        References
        ----------
        Derivation: Weight fresh and cached input tariffs by their traffic fractions
        and input/output mix. Equate linear API spend with fixed-plus-variable
        ownership spend and solve for the delivered fraction of reference capacity.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = DedicatedBreakEvenModel().solve(1.0, 0.5, 2.0, 1.0, 3.0, 1.0, 0.1, Q_('100 / s'), 0.5)
        >>> r.delivered_hour_tokens == r.reference_hour_tokens * 0.5
        True
        """
        weighted = (1 - cached_fraction) * input_price + cached_fraction * cached_price
        api = output_price + input_per_output * weighted
        ref = (reference_rate * ureg.hour).to("").magnitude
        delivered = load * ref
        api_hour = delivered * api / 1_000_000
        ded_hour = fixed_per_hour + delivered * variable_price / 1_000_000
        crossing = fixed_per_hour / (ref * (api - variable_price) / 1_000_000)
        api_nc = output_price + input_per_output * input_price
        return DedicatedBreakEvenResult(
            weighted_input_price=weighted,
            api_price=api,
            reference_hour_tokens=ref,
            delivered_hour_tokens=delivered,
            api_hour=api_hour,
            dedicated_hour=ded_hour,
            crossing=crossing,
            saving_hour=api_hour - ded_hour,
            api_price_no_cache=api_nc,
            crossing_no_cache=fixed_per_hour / (ref * (api_nc - variable_price) / 1_000_000),
        )


class ConservedBudgetModel(ForwardModel):
    """
    A root budget kept as settled + reserved + free across two children.

    Granting moves free to reserved; a child that closes settles its charge
    and returns the rest. A child stopped with a remote operation pending
    keeps that operation's enforceable bound reserved until its final bill
    settles, once per settlement identity however often it is delivered.
    """

    requires = ()
    produces = ConservedBudgetResult

    def solve(
        self,
        budget: int,
        planning: int,
        allocations: Sequence[int],
        settled_children: Sequence[int],
        pending: int,
        late_charge: int,
        deliveries: int = 2,
    ) -> ConservedBudgetResult:
        """Track settled, reserved, and free budget through child completion.

        Parameters
        ----------
        budget : int
            Total nonnegative integer budget in milli-USD.
        planning : int
            Initial settled planning charge in milli-USD.
        allocations : Sequence[int]
            Two nonnegative child reservations in milli-USD.
        settled_children : Sequence[int]
            Two child settled charges in milli-USD.
        pending : int
            Second child pending reservation in milli-USD.
        late_charge : int
            Final charge for the pending operation in milli-USD.
        deliveries : int
            Positive count of repeated deliveries of the final bill; default 2.

        Returns
        -------
        ConservedBudgetResult
            Ledger states and child balances in integer milli-USD, with
            settled, reserved, and free components at each transition.

        Notes
        -----
        Equation: budget = settled + reserved + free at every transition; duplicate delivery of the same settlement identity is applied once.

        Assumptions: Two child allocations in common integer budget units; charges fit reservations; nonnegative conserved state.

        Excludes: Distributed durability, concurrent ledger races, and remote billing enforcement.

        References
        ----------
        Derivation: A grant transfers free funds to reservations. Settlement transfers
        a charge from reserved to settled and refunds the unused reservation.
        Deduplicating the final bill prevents repeated delivery from spending twice.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = ConservedBudgetModel().solve(1000, 100, [300, 400], [200, 100], 100, 80)
        >>> r.settled_final + r.reserved_final + r.free_final == r.budget
        True
        """
        b_first, b_second = allocations
        e_first, e_second = settled_children
        f_plan = budget - planning
        r_deleg = b_first + b_second
        f_deleg = f_plan - r_deleg
        refund = b_first - e_first
        e_after = planning + e_first
        r_after = r_deleg - b_first
        f_after = f_deleg + refund
        free_at_stop = b_second - e_second - pending
        apparent = b_second - e_second
        applied: Dict[str, int] = {}
        settled_second = e_second
        for key, charge in [("final_bill", late_charge)] * deliveries:
            if key in applied:
                if applied[key] != charge:
                    raise ValueError("conflicting payload for one settlement identity")
            else:
                applied[key] = charge
                settled_second += charge
        free_settled = free_at_stop + pending - late_charge
        e_final = e_after + settled_second
        r_final = r_after - b_second
        f_final = f_after + free_settled
        states = [
            (planning, 0, f_plan),
            (planning, r_deleg, f_deleg),
            (e_after, r_after, f_after),
            (e_after, r_after, f_after),
            (e_final, r_final, f_final),
        ]
        for s_, r_, f_ in states:
            if s_ + r_ + f_ != budget or min(s_, r_, f_) < 0:
                raise ValueError("budget not conserved")
        return ConservedBudgetResult(
            units_per_dollar=1000,
            budget=budget,
            planning=planning,
            allocations=(b_first, b_second),
            settled_children=(e_first, e_second),
            pending=pending,
            late_charge=late_charge,
            free_after_planning=f_plan,
            reserved_delegated=r_deleg,
            free_after_delegation=f_deleg,
            refund_first=refund,
            settled_after_first=e_after,
            reserved_after_first=r_after,
            free_after_first=f_after,
            child_free_at_stop=free_at_stop,
            child_apparent_free=apparent,
            child_free_settled=free_settled,
            settled_final=e_final,
            reserved_final=r_final,
            free_final=f_final,
            states=states,
        )


class ArchitectureChoiceModel(ForwardModel):
    """
    Designs priced by automated spend alone and with a human fallback.

    A bounded loop stops at its first accepted turn, so turn ``j`` is reached
    by the share that every earlier turn rejected. Rejected tickets go to a
    person who resolves a fraction of them; cost per resolved ticket divides
    the combined spend by combined resolution.
    """

    requires = ()
    produces = ArchitectureChoiceResult

    def solve(
        self,
        tickets: int,
        single_costs: Mapping[str, Quantity],
        single_acceptance: Mapping[str, float],
        loop_name: str,
        turn_cost: Quantity,
        turn_acceptance: Sequence[float],
        check_cost: Quantity,
        human_cost: Quantity,
        human_resolution: float,
        point: float,
    ) -> ArchitectureChoiceResult:
        """Compare automated attempts and human fallback per resolved ticket.

        Parameters
        ----------
        tickets : int
            Nonnegative number of initiated tickets in the reporting period.
        single_costs : Mapping[str, Quantity]
            Configuration names mapped to per-attempt USD quantities.
        single_acceptance : Mapping[str, float]
            Configuration names mapped to acceptance probabilities in (0,1].
        loop_name : str
            Name for the iterative configuration, distinct from single configurations.
        turn_cost : Quantity
            Charge per reached loop turn, convertible to USD.
        turn_acceptance : Sequence[float]
            Conditional probabilities of acceptance on each reached turn.
        check_cost : Quantity
            Charge for checking each attempt, convertible to USD.
        human_cost : Quantity
            Fallback human charge per unresolved automated ticket, convertible to USD.
        human_resolution : float
            Fraction of fallback tickets resolved by a person, in [0,1].
        point : float
            Dimensionless fraction of human cost used for point_value.

        Returns
        -------
        ArchitectureChoiceResult
            Per-configuration USD quantity costs, acceptance/resolution
            fractions, expected loop turns, and period spend and resolved counts.

        Notes
        -----
        Equation: reach[j] = product(1-turn_acceptance[:j]); total cost = automated + ``(1-acceptance)*human_cost``; resolved = ``acceptance+(1-acceptance)*human_resolution``.

        Assumptions: Conditional turn acceptance probabilities in [0,1]; positive acceptance and resolution denominators; constant fallback cost.

        Excludes: Quality comparability, correlated human/model mistakes, and queueing delays.

        References
        ----------
        Derivation: Multiply conditional rejection probabilities to obtain each turn
        reach share. Weight automated charges by reach, then add human fallback
        cost for rejected tickets and divide by total automated-plus-human resolution.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = ArchitectureChoiceModel().solve(10, {'single': Q_('1 USD')}, {'single': 0.5}, 'loop', Q_('1 USD'), [0.5, 0.5], Q_('0.1 USD'), Q_('5 USD'), 1.0, 0.01)
        >>> r.resolved['loop'] == 1.0
        True
        """
        turn_cost, check_cost, human_cost = turn_cost.to(_USD), check_cost.to(_USD), human_cost.to(_USD)
        single_costs = {k: v.to(_USD) for k, v in single_costs.items()}
        reach = [1.0]
        for r in turn_acceptance[:-1]:
            reach.append(reach[-1] * (1 - r))
        p_loop = 1 - reach[-1] * (1 - turn_acceptance[-1])
        turns = 0
        for x in reach:
            turns = turns + x
        auto, accept = {}, {}
        for name, c in single_costs.items():
            auto[name] = c + check_cost
            accept[name] = single_acceptance[name]
        auto[loop_name] = turns * (turn_cost + check_cost)
        accept[loop_name] = p_loop
        per_auto = {k: auto[k] / accept[k] for k in auto}
        combined = {k: auto[k] + (1 - accept[k]) * human_cost for k in auto}
        resolved = {k: accept[k] + (1 - accept[k]) * human_resolution for k in auto}
        per_res = {k: combined[k] / resolved[k] for k in auto}
        return ArchitectureChoiceResult(
            loop_acceptance=p_loop,
            loop_turns=turns,
            reach=reach,
            automated=auto,
            automated_per_accepted=per_auto,
            combined=combined,
            resolved=resolved,
            per_resolved=per_res,
            month_spend=tickets * combined[loop_name],
            month_resolved=tickets * resolved[loop_name],
            point_value=point * human_cost,
        )


class AdaptationPaybackModel(ForwardModel):
    """
    Tasks until a one-time adaptation repays its data, training, and evaluation.

    Each task's model usage is steps times (input tokens x input price +
    output tokens x output price); the break-even count is the investment
    over the per-task saving, and the first whole task that crosses it.
    """

    requires = ()
    produces = AdaptationPaybackResult

    def solve(
        self,
        base: Mapping[str, float],
        adapted: Mapping[str, float],
        data_cost: Quantity,
        training_cost: Quantity,
        evaluation_cost: Quantity,
        monthly_volume: int,
        low_volume: int,
    ) -> AdaptationPaybackResult:
        """Calculate task volume needed to recover adaptation investment.

        Parameters
        ----------
        base : Mapping[str, float]
            Baseline mapping: steps, input_tokens, output_tokens, and USD/Mtoken input_price/output_price quantities.
        adapted : Mapping[str, float]
            Adapted mapping with the same keys and units as base.
        data_cost : Quantity
            One-time data investment, convertible to USD.
        training_cost : Quantity
            One-time training investment, convertible to USD.
        evaluation_cost : Quantity
            One-time evaluation investment, convertible to USD.
        monthly_volume : int
            Positive tasks per month.
        low_volume : int
            Positive lower-volume tasks per month.

        Returns
        -------
        AdaptationPaybackResult
            Step, task, investment, and saving USD quantities; fractional and
            rounded task break-even counts; payback months and cost shares.

        Notes
        -----
        Equation: task cost = ``steps*(input_tokens*input_price`` + ``output_tokens*output_price)``; crossing = investment/(base_task-adapted_task).

        Assumptions: Fixed token usage and tariffs; adaptation lowers cost per task; positive volume and investment.

        Excludes: Quality changes, model drift, recurring retraining, and discounting.

        References
        ----------
        Derivation: Sum data, training, and evaluation investment. Each completed
        task repays the difference between baseline and adapted usage bills.
        Divide investment by this difference, then divide tasks by monthly volume.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> base = dict(steps=2, input_tokens=100, output_tokens=10, input_price=Q_('1 USD/Mtoken'), output_price=Q_('2 USD/Mtoken'))
        >>> adapted = dict(base, steps=1)
        >>> r = AdaptationPaybackModel().solve(base, adapted, Q_('1 USD'), Q_('1 USD'), Q_('1 USD'), 1000, 100)
        >>> r.payback_months_low > r.payback_months
        True
        """

        def price(q):
            return _per_mtok(q) * _USD / 1_000_000

        b_pin, b_pout = price(base["input_price"]), price(base["output_price"])
        a_pin, a_pout = price(adapted["input_price"]), price(adapted["output_price"])
        base_step = base["input_tokens"] * b_pin + base["output_tokens"] * b_pout
        base_task = base["steps"] * base_step
        adapt_step = adapted["input_tokens"] * a_pin + adapted["output_tokens"] * a_pout
        adapt_task = adapted["steps"] * adapt_step
        data_cost, training_cost, evaluation_cost = (x.to(_USD) for x in (data_cost, training_cost, evaluation_cost))
        invest = data_cost + training_cost + evaluation_cost
        delta = base_task - adapt_task
        n_star = (invest / delta).to("").magnitude
        input_saving = base["steps"] * base["input_tokens"] * b_pin - adapted["steps"] * adapted["input_tokens"] * a_pin
        return AdaptationPaybackResult(
            base_step=base_step,
            base_task=base_task,
            adapted_step=adapt_step,
            adapted_task=adapt_task,
            investment=invest,
            saving_per_task=delta,
            breakeven_tasks=n_star,
            breakeven_whole=math.ceil(n_star),
            payback_months=n_star / monthly_volume,
            payback_months_low=n_star / low_volume,
            input_saving_share=(input_saving / delta).to("").magnitude,
            evaluation_share=(evaluation_cost / invest).to("").magnitude,
            training_share=(training_cost / invest).to("").magnitude,
        )


class WorkloadLeverModel(ForwardModel):
    """
    Cost per accepted task by workload, ``C_eff = C_attempt / p_acc``, and three levers.

    Halving the token price saves half the model share; halving the largest
    term saves half its share; raising acceptance by ``dp`` saves
    ``dp / (p + dp)`` of every term, and also removes the handoffs charged per
    rejected attempt. Acceptance beats halving tokens whenever the model share
    is below ``2 dp / (p + dp)``.
    """

    requires = ()
    produces = WorkloadLeverResult

    def solve(self, workloads: Mapping[str, Mapping[str, float]], acceptance_gain: float) -> WorkloadLeverResult:
        """Compare token, dominant-cost, and acceptance improvements.

        Parameters
        ----------
        workloads : Mapping[str, Mapping[str, float]]
            Names mapped to model, environment, assessment, human costs in USD and acceptance in (0,1]; optional handoff replaces human with (1-acceptance)*handoff.
        acceptance_gain : float
            Absolute increase in acceptance probability; keeps each probability at most one.

        Returns
        -------
        WorkloadLeverResult
            Rows with USD attempt and accepted-task costs as floats, the
            dominant cost-term name, and dimensionless lever savings and shares.

        Notes
        -----
        Equation: cost per accepted = sum(attempt costs)/acceptance; each lever recomputes this ratio after changing its specified term.

        Assumptions: Positive acceptance and total cost; acceptance_gain keeps acceptance at most one; costs use a common currency.

        Excludes: Interactions among levers, causal predictions of acceptance, and changed task mix.

        References
        ----------
        Derivation: Recompute cost per accepted task after halving model cost, halving
        the largest cost term, or raising acceptance. If human cost is charged
        per rejection, recompute that term under the changed rejection fraction.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> w = {'x': dict(model=1.0, environment=2.0, assessment=1.0, human=1.0, acceptance=0.5)}
        >>> r = WorkloadLeverModel().solve(w, 0.1)
        >>> r.rows['x'].largest_term == 'environment'
        True
        """
        dp = acceptance_gain
        rows = {}
        for name, w in workloads.items():
            p = w["acceptance"]
            handoff = w.get("handoff")
            human = (1 - p) * handoff if handoff is not None else w["human"]
            terms = {
                "model": w["model"],
                "environment": w["environment"],
                "assessment": w["assessment"],
                "human": human,
            }
            a = terms["model"] + terms["environment"] + terms["assessment"] + terms["human"]
            c = a / p
            big_name = max(terms, key=terms.get)
            big = max(terms["model"], terms["environment"], terms["assessment"], terms["human"])
            p2 = p + dp
            h2 = (1 - p2) * handoff if handoff is not None else human
            rows[name] = WorkloadLeverRow(
                model=terms["model"],
                environment=terms["environment"],
                assessment=terms["assessment"],
                human=human,
                acceptance=p,
                attempt=a,
                cost_per_accepted=c,
                model_share=terms["model"] / a,
                largest_term=big_name,
                halve_tokens=1 - (a - terms["model"] / 2) / p / c,
                halve_largest=1 - (a - big / 2) / p / c,
                acceptance_gain=1 - (a - human + h2) / p2 / c,
                token_threshold=2 * dp / (p + dp),
            )
        return WorkloadLeverResult(rows=rows)
