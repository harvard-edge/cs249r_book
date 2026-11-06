"""Reinforcement learning from verifiable rewards: what a reward measures and what the loop costs.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each
model is arithmetic on stated rates, counts, and durations: the share of an
update's accepted rollouts that failed, binary rewards centered on a fixed
baseline, a shaped reward's ordering condition, a turn valued by forked
continuations and the tokens forking costs, group-relative coefficients, the
per-token push of length normalization, the rollouts one informative group
costs, an all-pass rule over repeated trials, the room a ratio clip leaves, a
sandbox slot's episode cycle, a group barrier, and a verifier pool. None
models a trained network's learning dynamics.
"""

from __future__ import annotations

import math

from ..results import (
    CenteredRewardResult,
    ContinuationForkCostResult,
    ForkValueResult,
    GroupBarrierResult,
    GroupCoefficientsResult,
    LengthNormalizationResult,
    RatioClipResult,
    RepeatedVerificationResult,
    RewardCompositionResult,
    RolloutSlotCycleResult,
    RolloutStepBudgetResult,
    VerifierPoolResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...physics.agents import (
    calc_independent_success,
    calc_mixed_outcome_fraction,
    calc_pass_posterior,
    calc_short_circuit_cost,
)
from ...physics.policy_gradient import (
    calc_binary_mean_se_bound,
    calc_continuation_fork_tokens,
    calc_group_relative_coefficients,
    calc_ratio_clip_room,
)
from ...physics.serving import calc_littles_law_occupancy


def _ratio(a, b) -> float:
    r = a / b
    return float(r.to("").magnitude) if hasattr(r, "to") else float(r)


class RewardCompositionModel(ForwardModel):
    """
    True and false accepts among one update's rollouts under an imperfect verifier.

    ``tasks x group`` rollouts pass with probability ``p``; the verifier accepts
    a pass with probability ``valid_accept_rate`` and a failure with
    ``false_accept_rate``. Expected counts carry no rounding. The false share
    is Bayes' rule over the accepts, so it moves with the policy's pass rate as
    well as with the verifier.
    """

    requires = ()
    produces = RewardCompositionResult

    def solve(
        self, tasks: int, group: int, pass_rate: float, false_accept_rate: float, valid_accept_rate: float
    ) -> RewardCompositionResult:
        """Separate true and false verifier acceptances in a rollout batch.

        Parameters
        ----------
        tasks : int
            Number of tasks in the update, a positive count.
        group : int
            Number of candidates per group, a positive count.
        pass_rate : float
            Underlying binary success probability in [0, 1].
        false_accept_rate : float
            Probability of accepting an incorrect rollout, in [0, 1].
        valid_accept_rate : float
            Probability of accepting a correct rollout, in [0, 1].

        Returns
        -------
        RewardCompositionResult
            Expected rollout and acceptance counts, plus the dimensionless false-accept share.

        Notes
        -----
        Equation: n=``tasks*group``; true=``n*p*v``; false=``n*(1-p)*f``.

        Assumptions: Stationary conditional acceptance rates and a nonempty accepted population.

        Excludes: Correlated verifier errors and changes in policy quality.

        References
        ----------
        Derivation: Multiply rollout counts by correctness prevalence and conditional acceptance probabilities. The false-accept share divides the false joint count by all accepts, which is Bayes conditioning on acceptance.

        Examples
        --------
        >>> result = RewardCompositionModel().solve(2, 4, 0.5, 0.1, 0.9)
        >>> result.accepts == result.true_accepts + result.false_accepts
        True
        """
        n = tasks * group
        true_acc = n * pass_rate * valid_accept_rate
        false_acc = n * (1 - pass_rate) * false_accept_rate
        return RewardCompositionResult(
            rollouts=n,
            true_accepts=true_acc,
            false_accepts=false_acc,
            accepts=true_acc + false_acc,
            false_share=calc_pass_posterior(1 - pass_rate, false_accept_rate, valid_accept_rate),
        )


class CenteredRewardModel(ForwardModel):
    """
    Binary terminal rewards centered on a fixed baseline: ``r - b``.

    A positive baseline makes every failure's centered score negative, so the
    update pushes every turn of a failed rollout down alike.
    """

    requires = ()
    produces = CenteredRewardResult

    def solve(self, baseline: float, passing_reward: float, failing_reward: float) -> CenteredRewardResult:
        """Subtract a shared baseline from passing and failing rewards.

        Parameters
        ----------
        baseline : float
            Fixed dimensionless reward baseline.
        passing_reward : float
            Dimensionless reward assigned to a passing rollout.
        failing_reward : float
            Dimensionless reward assigned to a failing rollout.

        Returns
        -------
        CenteredRewardResult
            Dimensionless centered scores for passing and failing outcomes.

        Notes
        -----
        Equation: score = reward - baseline for each outcome.

        Assumptions: A fixed baseline shared by both outcomes.

        Excludes: Gradient dynamics, temporal credit assignment, and learned baselines.

        References
        ----------
        Derivation: Centering translates each reward by the same baseline; it changes signs relative to that baseline without changing the difference between rewards.

        Examples
        --------
        >>> result = CenteredRewardModel().solve(0.5, 1.0, 0.0)
        >>> result.passing_score == -result.failing_score
        True
        """
        return CenteredRewardResult(passing_score=passing_reward - baseline, failing_score=failing_reward - baseline)


class ForkValueModel(ForwardModel):
    """
    One turn valued by forking ``k`` binary continuations before and after it.

    Each fork's sampled mean has standard error at most ``1 / (2 sqrt k)``;
    the difference of two independent means at most ``sqrt 2`` times that. The
    difference is resolved when it exceeds two difference bounds.
    """

    requires = ()
    produces = ForkValueResult

    def solve(self, continuations: int, passes_before: int, passes_after: int) -> ForkValueResult:
        """Compare continuation pass fractions against sampling uncertainty.

        Parameters
        ----------
        continuations : int
            Positive number of independent continuations at each fork.
        passes_before : int
            Successful continuations before the turn, between zero and continuations.
        passes_after : int
            Successful continuations after the turn, between zero and continuations.

        Returns
        -------
        ForkValueResult
            Empirical success fractions, standard-error bounds, and the resolution decision.

        Notes
        -----
        Equation: v=passes/k; SE<=``1/(2*sqrt(k))``; threshold=``2*sqrt(2)*SE``.

        Assumptions: Independent Bernoulli continuations in the before and after samples.

        Excludes: Paired-sample covariance and a formal confidence-interval decision rule.

        References
        ----------
        Derivation: A Bernoulli variance is at most one quarter; dividing by continuation count bounds each mean variance. Adding the two mean variances gives the comparison standard error.

        Examples
        --------
        >>> result = ForkValueModel().solve(100, 20, 80)
        >>> result.value_after > result.value_before
        True
        """
        if not 0 <= passes_before <= continuations or not 0 <= passes_after <= continuations:
            raise ValueError("pass counts must lie between 0 and the continuations per fork")
        v0, v1 = passes_before / continuations, passes_after / continuations
        se = calc_binary_mean_se_bound(continuations)
        se_diff = math.sqrt(2) * se
        threshold = 2 * se_diff
        return ForkValueResult(
            value_before=v0,
            value_after=v1,
            difference=v1 - v0,
            se_bound=se,
            difference_se_bound=se_diff,
            resolution_threshold=threshold,
            resolved=abs(v1 - v0) > threshold,
        )


class ContinuationForkCostModel(ForwardModel):
    """
    Tokens generated when ``branches`` continuations are forked at chosen turns.

    Forking at every boundary costs ``1 + m (H - 1) / 2`` times the base
    trajectory, which grows with the square of the horizon in tokens.
    """

    requires = ()
    produces = ContinuationForkCostResult

    def solve(self, turns: int, tokens_per_turn: int, branches: int, boundaries=None) -> ContinuationForkCostResult:
        """Count tokens generated by continuations from selected turn boundaries.

        Parameters
        ----------
        turns : int
            Positive number of turns in each base trajectory.
        tokens_per_turn : int
            Positive generated-token count per turn.
        branches : int
            Nonnegative continuation count at each selected boundary.
        boundaries : sequence, optional
            Selected turn boundaries; None selects every internal boundary.

        Returns
        -------
        ContinuationForkCostResult
            Generated-token counts and the dimensionless expansion over the base trajectory.

        Notes
        -----
        Equation: base=``H*a``; total=``base+sum(m*a*(H-b)`` for b in boundaries).

        Assumptions: Equal token cost per turn and fixed branch count at selected boundaries.

        Excludes: Prompt rereads, caching, and variable-length continuations.

        References
        ----------
        Derivation: Each boundary retains its prefix and generates only the remaining turns for each branch. Adding those suffix lengths to the base transcript counts generated work without resending prefixes.

        Examples
        --------
        >>> result = ContinuationForkCostModel().solve(4, 10, 2)
        >>> result.total_tokens >= result.base_tokens
        True
        """
        base, count, total = calc_continuation_fork_tokens(turns, tokens_per_turn, branches, boundaries)
        return ContinuationForkCostResult(base_tokens=base, branches=count, total_tokens=total, multiple=total / base)


class GroupCoefficientsModel(ForwardModel):
    """
    Group-relative (GRPO) coefficients ``(r_i - mean) / (std + epsilon)`` of one group.

    Every sampled token of rollout ``i`` carries coefficient ``i``. The
    dispersion divisor (``G`` or ``G - 1``) and the stabilizer are part of the
    objective; a lone pass in a group of ``G`` binary rewards gets
    ``sqrt(G - 1)`` as epsilon goes to zero.
    """

    requires = ()
    produces = GroupCoefficientsResult

    def solve(self, rewards, stabilizer: float, bessel: bool = False) -> GroupCoefficientsResult:
        """Center and scale rewards within a rollout group.

        Parameters
        ----------
        rewards : sequence of float
            Nonempty sequence of dimensionless sampled rewards.
        stabilizer : float
            Nonnegative dimensionless denominator stabilizer.
        bessel : bool
            Use sample standard deviation when True; population deviation otherwise.

        Returns
        -------
        GroupCoefficientsResult
            Group size, reward mean and standard deviation, and dimensionless coefficients.

        Notes
        -----
        Equation: coefficient_i=(reward_i-mean)/(std+stabilizer).

        Assumptions: One finite reward group; Bessel correction follows the supplied flag.

        Excludes: Policy updates, clipping, and the effect of these coefficients on learning.

        References
        ----------
        Derivation: Subtracting the group mean yields centered rewards. Dividing by the chosen standard deviation plus stabilizer defines the normalized update coefficients.

        Examples
        --------
        >>> result = GroupCoefficientsModel().solve([0.0, 1.0], 0.1)
        >>> abs(result.coefficient_sum) < 1e-12
        True
        """
        mean, std, coefs = calc_group_relative_coefficients(rewards, stabilizer, bessel)
        return GroupCoefficientsResult(
            group=len(coefs), mean=mean, std=std, coefficients=coefs, coefficient_sum=sum(coefs)
        )


class LengthNormalizationModel(ForwardModel):
    """
    Per-token push of two rollouts with the same coefficient under per-rollout averaging.

    Each rollout's loss is divided by its own length, so the shorter rollout's
    tokens each move ``long / short`` times as far. A failure is pushed down
    hardest when it is short, so failures drift long.
    """

    requires = ()
    produces = LengthNormalizationResult

    def solve(self, short_tokens: int, long_tokens: int) -> LengthNormalizationResult:
        """Compare per-token weights under equal trajectory weighting.

        Parameters
        ----------
        short_tokens : int
            Positive token count of the shorter rollout.
        long_tokens : int
            Positive token count of the longer rollout.

        Returns
        -------
        LengthNormalizationResult
            Dimensionless ratio of the shorter rollout's per-token coefficient to the longer one.

        Notes
        -----
        Equation: per_token_ratio=long_tokens/short_tokens.

        Assumptions: Equal rollout coefficients and separate per-rollout mean losses.

        Excludes: Actual gradients, reward differences, and predicted sequence-length drift.

        References
        ----------
        Derivation: Equal trajectory weight is divided among its tokens; the ratio of per-token weights is therefore the inverse ratio of trajectory lengths.

        Examples
        --------
        >>> result = LengthNormalizationModel().solve(10, 20)
        >>> result.per_token_ratio == 20 / 10
        True
        """
        return LengthNormalizationResult(per_token_ratio=long_tokens / short_tokens)


class RolloutStepBudgetModel(ForwardModel):
    """
    One update's generated tokens and the rollouts spent per informative group.

    A group of ``G`` independent binary attempts at pass rate ``p`` carries
    reward contrast with probability ``1 - p^G - (1 - p)^G``; a group without
    it moves no weights, so each informative group costs ``G / P(contrast)``
    rollouts (infinite at ``p = 0`` or ``p = 1``, where no group carries contrast).
    """

    requires = ()
    produces = RolloutStepBudgetResult

    def solve(
        self, tasks: int, group: int, turns: int, tokens_per_turn: int, pass_rate: float
    ) -> RolloutStepBudgetResult:
        """Count rollout tokens and groups containing both outcome classes.

        Parameters
        ----------
        tasks : int
            Number of tasks in the update, a positive count.
        group : int
            Number of candidates per group, a positive count.
        turns : int
            Positive number of turns in each base trajectory.
        tokens_per_turn : int
            Positive generated-token count per turn.
        pass_rate : float
            Underlying binary success probability in [0, 1].

        Returns
        -------
        RolloutStepBudgetResult
            Rollout and token counts, contrast probability, and rollouts per informative group.

        Notes
        -----
        Equation: tokens=``tasks*group*turns*tokens_per_turn``; contrast=``1-p**G-(1-p)**G``.

        Assumptions: Independent binary outcomes with one shared pass probability.

        Excludes: Correlated outcomes, token-length variance, and update effectiveness.

        References
        ----------
        Derivation: Token volume multiplies tasks, rollouts, turns, and tokens per turn. Subtracting all-pass and all-fail group probabilities from one leaves groups with both outcomes.

        Examples
        --------
        >>> result = RolloutStepBudgetModel().solve(2, 4, 3, 10, 0.5)
        >>> result.step_tokens == result.rollouts * result.rollout_tokens
        True
        """
        contrast = calc_mixed_outcome_fraction(pass_rate, group)
        rollouts = tasks * group
        per_rollout = turns * tokens_per_turn
        return RolloutStepBudgetResult(
            rollouts=rollouts,
            rollout_tokens=per_rollout,
            step_tokens=rollouts * per_rollout,
            contrast=contrast,
            silent_share=1 - contrast,
            rollouts_per_informative_group=group / contrast if contrast > 0 else math.inf,
        )


class RepeatedVerificationModel(ForwardModel):
    """
    An all-pass rule over ``n`` independent trials of one candidate.

    Graded once, a group of ``G`` incorrect candidates holds at least one false
    pass with probability ``1 - (1 - q_bad)^G``. Requiring all ``n`` trials to
    pass accepts an incorrect candidate with ``q_bad^n`` and rejects a correct
    one with ``1 - q_good^n``. Stopping at the first failed trial runs
    ``sum_{j<n} q^j`` trials on average.
    """

    requires = ()
    produces = RepeatedVerificationResult

    def solve(self, incorrect_pass: float, correct_pass: float, trials: int, group: int) -> RepeatedVerificationResult:
        """Compute acceptance errors and trial cost for an all-pass rule.

        Parameters
        ----------
        incorrect_pass : float
            Probability an incorrect candidate passes one trial, in [0, 1].
        correct_pass : float
            Probability a correct candidate passes one trial, in [0, 1].
        trials : int
            Positive count of verification trials per candidate.
        group : int
            Number of candidates per group, a positive count.

        Returns
        -------
        RepeatedVerificationResult
            Acceptance and rejection probabilities and expected counts of executed trials.

        Notes
        -----
        Equation: false_accept=``q_bad**n``; true_reject=``1-q_good**n``; E[trials]=``sum(q**j)``.

        Assumptions: Independent identically distributed trials and first-failure termination.

        Excludes: Persistent verifier bias and correlated repeated judgments.

        References
        ----------
        Derivation: Independent pass probabilities multiply across required trials. For early rejection, trial j is reached only after the preceding trials pass, so summing those survival probabilities gives expected work.

        Examples
        --------
        >>> result = RepeatedVerificationModel().solve(0.1, 0.9, 2, 4)
        >>> result.incorrect_accepted < 0.1
        True
        """
        ones = [1.0] * trials
        e_bad, _ = calc_short_circuit_cost([incorrect_pass] * trials, ones)
        e_good, _ = calc_short_circuit_cost([correct_pass] * trials, ones)
        return RepeatedVerificationResult(
            group_false_pass=calc_independent_success(incorrect_pass, group),
            incorrect_accepted=incorrect_pass**trials,
            correct_rejected=1 - correct_pass**trials,
            expected_trials_incorrect=e_bad,
            expected_trials_correct=e_good,
        )


class RatioClipModel(ForwardModel):
    """
    The probability ceiling a symmetric ratio clip ``1 +/- epsilon`` sets on one token.

    A dominant token's ceiling can exceed 1 (the clip does not bind), while a
    rare token may rise only ``epsilon p`` per update.
    """

    requires = ()
    produces = RatioClipResult

    def solve(self, epsilon: float, probability: float) -> RatioClipResult:
        """Translate a probability-ratio clip into absolute probability headroom.

        Parameters
        ----------
        epsilon : float
            Nonnegative dimensionless symmetric ratio-clip width.
        probability : float
            Old token probability in [0, 1].

        Returns
        -------
        RatioClipResult
            Dimensionless probability ceiling and rise, plus whether the ceiling lies below one.

        Notes
        -----
        Equation: ceiling=``p*(1+epsilon)``; rise=``p*epsilon``; binds=(ceiling<1).

        Assumptions: A symmetric probability-ratio upper clip for one token.

        Excludes: Probability renormalization across tokens and optimizer dynamics.

        References
        ----------
        Derivation: Multiplying the original probability by the upper ratio clip gives its clipped ceiling; subtracting the original probability gives the permitted absolute rise.

        Examples
        --------
        >>> result = RatioClipModel().solve(0.2, 0.5)
        >>> result.binds
        True
        """
        ceiling, rise = calc_ratio_clip_room(probability, epsilon)
        return RatioClipResult(ceiling=ceiling, rise=rise, binds=ceiling < 1)


class RolloutSlotCycleModel(ForwardModel):
    """
    A sandbox slot running one episode at a time, under a cold and a snapshot reset.

    Each episode is ``decisions x (model + tool)`` plus an assessment, then a
    reset. Capacity is ``slots / cycle``. Faster resets raise capacity only by
    the reset's share of the cycle (Amdahl's law); the work floor remains.
    """

    requires = ()
    produces = RolloutSlotCycleResult

    def solve(
        self,
        slots: int,
        decisions: int,
        model_time: Quantity,
        tool_time: Quantity,
        assessment_time: Quantity,
        cold_reset: Quantity,
        snapshot_reset: Quantity,
    ) -> RolloutSlotCycleResult:
        """Estimate slot occupancy and throughput over rollout-reset cycles.

        Parameters
        ----------
        slots : int
            Positive count of independently occupied sandbox slots.
        decisions : int
            Nonnegative number of model/tool decisions per episode.
        model_time : Quantity
            Time quantity per model invocation, nonnegative.
        tool_time : Quantity
            Time quantity per tool invocation, nonnegative.
        assessment_time : Quantity
            Nonnegative time quantity for final assessment.
        cold_reset : Quantity
            Positive time quantity for a cold reset.
        snapshot_reset : Quantity
            Positive time quantity for a snapshot reset.

        Returns
        -------
        RolloutSlotCycleResult
            Work and cycle times in seconds, episodes per hour, and dimensionless capacity ratios.

        Notes
        -----
        Equation: work=``D*(model+tool)+assessment``; capacity=slots/(work+reset).

        Assumptions: Serial fixed-duration stages and one episode per slot.

        Excludes: Queueing, overlapping stages, startup variance, and resource contention.

        References
        ----------
        Derivation: Serial model and tool stages accumulate over turns, then assessment and reset extend the cycle. Dividing available slots by cycle duration gives the capacity bound.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> result = RolloutSlotCycleModel().solve(2, 3, Q_("1 s"), Q_("2 s"), Q_("1 s"), Q_("2 s"), Q_("1 s"))
        >>> result.snapshot_cycle < result.cold_cycle
        True
        """
        work = (decisions * (model_time + tool_time) + assessment_time).to("s")
        cold = (work + cold_reset).to("s")
        snap = (work + snapshot_reset).to("s")
        cold_rate = (slots / cold).to("1/hour").magnitude
        snap_rate = (slots / snap).to("1/hour").magnitude
        return RolloutSlotCycleResult(
            work=work,
            cold_cycle=cold,
            snapshot_cycle=snap,
            cold_work_share=_ratio(work, cold),
            snapshot_work_share=_ratio(work, snap),
            cold_episodes_per_hour=cold_rate,
            snapshot_episodes_per_hour=snap_rate,
            capacity_ratio=snap_rate / cold_rate,
            reset_ratio=_ratio(cold_reset, snapshot_reset),
        )


class GroupBarrierModel(ForwardModel):
    """
    A synchronous group whose step waits for its slowest member.

    Every slot is held until the barrier, so the reserved slot time is
    ``G x max`` and the busy share is ``mean / max``.
    """

    requires = ()
    produces = GroupBarrierResult

    def solve(self, latencies) -> GroupBarrierResult:
        """Measure synchronous group delay and idle slot time.

        Parameters
        ----------
        latencies : sequence of Quantity
            Nonempty sequence of positive, mutually compatible time quantities.

        Returns
        -------
        GroupBarrierResult
            Group count, time-valued extrema and mean, and dimensionless idle-slot fraction.

        Notes
        -----
        Equation: barrier=max(latencies); idle_share=1-mean(latencies)/barrier.

        Assumptions: All group members begin together and reserve their slots until the barrier.

        Excludes: Staggered starts, slot reuse, and distributed synchronization traffic.

        References
        ----------
        Derivation: A barrier releases at the longest latency. Each shorter member idles for the difference; summing those differences and dividing by total reserved slot-time gives idle share.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> result = GroupBarrierModel().solve([Q_("1 s"), Q_("2 s")])
        >>> result.fastest <= result.mean <= result.barrier
        True
        """
        group = len(latencies)
        barrier = max(latencies)
        mean = sum(latencies, 0 * latencies[0]) / group
        return GroupBarrierResult(
            group=group, fastest=min(latencies), barrier=barrier, mean=mean, idle_share=1 - _ratio(mean, barrier)
        )


class VerifierPoolModel(ForwardModel):
    """
    Verification workers for a stationary stream of completed trajectories.

    By Little's law the busy workers are the arrival rate times the service
    time per trajectory (every trial executes in this sizing case); the pool
    is that demand over the target utilization, rounded up.
    """

    requires = ()
    produces = VerifierPoolResult

    def solve(
        self, trajectories: int, interval: Quantity, trials: int, trial_time: Quantity, target_utilization: float
    ) -> VerifierPoolResult:
        """Size a verifier pool from arrival load and utilization target.

        Parameters
        ----------
        trajectories : int
            Positive completed-trajectory count in the observation interval.
        interval : Quantity
            Positive observation interval as a time quantity.
        trials : int
            Positive count of verification trials per candidate.
        trial_time : Quantity
            Positive service time quantity per verification trial.
        target_utilization : float
            Target busy fraction in (0, 1].

        Returns
        -------
        VerifierPoolResult
            Arrival rate in inverse seconds, mean busy-worker counts, integer slots, and utilization.

        Notes
        -----
        Equation: busy=``arrival_rate*trials*trial_time``; slots=ceil(busy/target_utilization).

        Assumptions: Stationary arrivals, every trial executed, and identical workers.

        Excludes: Queueing-tail guarantees, early trial termination, and burst arrivals.

        References
        ----------
        Derivation: Arrival rate times trials per arrival times trial duration gives offered busy-slot demand. Divide by the utilization target and round up to obtain an integer pool size.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> result = VerifierPoolModel().solve(10, Q_("1 s"), 2, Q_("0.1 s"), 0.8)
        >>> result.utilization <= 0.8
        True
        """
        rate = (trajectories / interval).to("1/s")
        one = float(calc_littles_law_occupancy(rate, trial_time).to("").magnitude)
        busy = float(calc_littles_law_occupancy(rate, trials * trial_time).to("").magnitude)
        slots = math.ceil(busy / target_utilization)
        return VerifierPoolResult(
            arrival_rate=rate, one_trial_busy=one, busy=busy, slots=slots, utilization=busy / slots
        )
