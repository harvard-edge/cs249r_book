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
    RewardShapingResult,
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

    def solve(self, tasks: int, group: int, pass_rate: float, false_accept_rate: float,
              valid_accept_rate: float) -> RewardCompositionResult:
        n = tasks * group
        true_acc = n * pass_rate * valid_accept_rate
        false_acc = n * (1 - pass_rate) * false_accept_rate
        return RewardCompositionResult(
            rollouts=n, true_accepts=true_acc, false_accepts=false_acc, accepts=true_acc + false_acc,
            false_share=calc_pass_posterior(1 - pass_rate, false_accept_rate, valid_accept_rate))


class CenteredRewardModel(ForwardModel):
    """
    Binary terminal rewards centered on a fixed baseline: ``r - b``.

    A positive baseline makes every failure's centered score negative, so the
    update pushes every turn of a failed rollout down alike.
    """
    requires = ()
    produces = CenteredRewardResult

    def solve(self, baseline: float, passing_reward: float, failing_reward: float) -> CenteredRewardResult:
        return CenteredRewardResult(passing_score=passing_reward - baseline,
                                    failing_score=failing_reward - baseline)


class RewardShapingModel(ForwardModel):
    """
    A shaped terminal score ``R = c + alpha f - lambda len`` with ``c`` in {0, 1}.

    With ``f`` and ``len = tokens / cap`` in [0, 1], a pass scores at least
    ``1 - lambda`` and a failure at most ``alpha``, so every pass outranks
    every failure when ``alpha + lambda < 1``. The runtime must enforce the
    cap, or ``len`` can exceed 1 and the bound fails.
    """
    requires = ()
    produces = RewardShapingResult

    def solve(self, format_weight: float, length_weight: float, length_cap: int,
              pass_format: float, pass_tokens: int, failure_format: float,
              failure_tokens: int) -> RewardShapingResult:
        def score(correct, fmt, tokens):
            return correct + format_weight * fmt - length_weight * tokens / length_cap

        total = format_weight + length_weight
        return RewardShapingResult(
            weight_sum=total, lowest_pass=1 - length_weight, highest_failure=format_weight,
            order_holds=total < 1,
            example_pass=score(1, pass_format, pass_tokens),
            example_failure=score(0, failure_format, failure_tokens))


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
        if not 0 <= passes_before <= continuations or not 0 <= passes_after <= continuations:
            raise ValueError("pass counts must lie between 0 and the continuations per fork")
        v0, v1 = passes_before / continuations, passes_after / continuations
        se = calc_binary_mean_se_bound(continuations)
        se_diff = math.sqrt(2) * se
        threshold = 2 * se_diff
        return ForkValueResult(value_before=v0, value_after=v1, difference=v1 - v0, se_bound=se,
                               difference_se_bound=se_diff, resolution_threshold=threshold,
                               resolved=abs(v1 - v0) > threshold)


class ContinuationForkCostModel(ForwardModel):
    """
    Tokens generated when ``branches`` continuations are forked at chosen turns.

    Forking at every boundary costs ``1 + m (H - 1) / 2`` times the base
    trajectory, which grows with the square of the horizon in tokens.
    """
    requires = ()
    produces = ContinuationForkCostResult

    def solve(self, turns: int, tokens_per_turn: int, branches: int,
              boundaries=None) -> ContinuationForkCostResult:
        base, count, total = calc_continuation_fork_tokens(turns, tokens_per_turn, branches, boundaries)
        return ContinuationForkCostResult(base_tokens=base, branches=count, total_tokens=total,
                                          multiple=total / base)


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
        mean, std, coefs = calc_group_relative_coefficients(rewards, stabilizer, bessel)
        return GroupCoefficientsResult(group=len(coefs), mean=mean, std=std, coefficients=coefs,
                                       coefficient_sum=sum(coefs))


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

    def solve(self, tasks: int, group: int, turns: int, tokens_per_turn: int,
              pass_rate: float) -> RolloutStepBudgetResult:
        contrast = calc_mixed_outcome_fraction(pass_rate, group)
        rollouts = tasks * group
        per_rollout = turns * tokens_per_turn
        return RolloutStepBudgetResult(
            rollouts=rollouts, rollout_tokens=per_rollout, step_tokens=rollouts * per_rollout,
            contrast=contrast, silent_share=1 - contrast,
            rollouts_per_informative_group=group / contrast if contrast > 0 else math.inf)


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

    def solve(self, incorrect_pass: float, correct_pass: float, trials: int,
              group: int) -> RepeatedVerificationResult:
        ones = [1.0] * trials
        e_bad, _ = calc_short_circuit_cost([incorrect_pass] * trials, ones)
        e_good, _ = calc_short_circuit_cost([correct_pass] * trials, ones)
        return RepeatedVerificationResult(
            group_false_pass=calc_independent_success(incorrect_pass, group),
            incorrect_accepted=incorrect_pass ** trials, correct_rejected=1 - correct_pass ** trials,
            expected_trials_incorrect=e_bad, expected_trials_correct=e_good)


class RatioClipModel(ForwardModel):
    """
    The probability ceiling a symmetric ratio clip ``1 +/- epsilon`` sets on one token.

    A dominant token's ceiling can exceed 1 (the clip does not bind), while a
    rare token may rise only ``epsilon p`` per update.
    """
    requires = ()
    produces = RatioClipResult

    def solve(self, epsilon: float, probability: float) -> RatioClipResult:
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

    def solve(self, slots: int, decisions: int, model_time: Quantity, tool_time: Quantity,
              assessment_time: Quantity, cold_reset: Quantity,
              snapshot_reset: Quantity) -> RolloutSlotCycleResult:
        work = (decisions * (model_time + tool_time) + assessment_time).to("s")
        cold = (work + cold_reset).to("s")
        snap = (work + snapshot_reset).to("s")
        cold_rate = (slots / cold).to("1/hour").magnitude
        snap_rate = (slots / snap).to("1/hour").magnitude
        return RolloutSlotCycleResult(
            work=work, cold_cycle=cold, snapshot_cycle=snap,
            cold_work_share=_ratio(work, cold), snapshot_work_share=_ratio(work, snap),
            cold_episodes_per_hour=cold_rate, snapshot_episodes_per_hour=snap_rate,
            capacity_ratio=snap_rate / cold_rate, reset_ratio=_ratio(cold_reset, snapshot_reset))


class GroupBarrierModel(ForwardModel):
    """
    A synchronous group whose step waits for its slowest member.

    Every slot is held until the barrier, so the reserved slot time is
    ``G x max`` and the busy share is ``mean / max``.
    """
    requires = ()
    produces = GroupBarrierResult

    def solve(self, latencies) -> GroupBarrierResult:
        group = len(latencies)
        barrier = max(latencies)
        mean = sum(latencies, 0 * latencies[0]) / group
        return GroupBarrierResult(group=group, fastest=min(latencies), barrier=barrier, mean=mean,
                                  idle_share=1 - _ratio(mean, barrier))


class VerifierPoolModel(ForwardModel):
    """
    Verification workers for a stationary stream of completed trajectories.

    By Little's law the busy workers are the arrival rate times the service
    time per trajectory (every trial executes in this sizing case); the pool
    is that demand over the target utilization, rounded up.
    """
    requires = ()
    produces = VerifierPoolResult

    def solve(self, trajectories: int, interval: Quantity, trials: int, trial_time: Quantity,
              target_utilization: float) -> VerifierPoolResult:
        rate = (trajectories / interval).to("1/s")
        one = float(calc_littles_law_occupancy(rate, trial_time).to("").magnitude)
        busy = float(calc_littles_law_occupancy(rate, trials * trial_time).to("").magnitude)
        slots = math.ceil(busy / target_utilization)
        return VerifierPoolResult(arrival_rate=rate, one_trial_busy=one, busy=busy, slots=slots,
                                  utilization=busy / slots)
