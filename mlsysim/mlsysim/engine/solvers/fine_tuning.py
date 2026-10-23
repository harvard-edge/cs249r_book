"""Fine-tuning solvers: what a serialized trajectory teaches, and what the update costs.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each
model is arithmetic on stated counts: one serialized example's positions under
a target mask, one example per model call against one transcript, recorded
transcripts split by role, a long trajectory's weight under token and example
averaging, padding against packing, one preference pair under the DPO loss, a
distilled per-turn prefix against its bill, and a candidate policy's release
comparison. None models a trained network's learning dynamics.
"""

from __future__ import annotations

from ..results import (
    LossNormalizationResult,
    PerInvocationSerializationResult,
    PreferencePairResult,
    SequencePackingResult,
    TargetMaskResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...core.units import ureg as _ureg
from ...physics.agents import calc_append_only_context, calc_cost_per_accepted
from ...physics.fine_tuning import (
    calc_example_average_share,
    calc_first_fit_decreasing,
    calc_preference_pair,
    calc_token_average_shares,
)
from ...physics.statistics import calc_proportion_gap_se


class TargetMaskModel(ForwardModel):
    """
    One serialized example's positions under two target masks.

    The example stores a prompt, a task, an optional rationale, the policy's
    responses, and the observations. Targets are the responses minus the
    ``unselected`` ones (kept as context), with or without the rationale. The
    first stored position has no preceding position, so an unmasked token
    average divides by ``stored - 1`` prediction terms; each target's weight
    shrinks by ``(stored - 1) / targets``.
    """
    requires = ()
    produces = TargetMaskResult

    def solve(self, prompt: int, task: int, rationale: int, response: int, observation: int,
              unselected_response: int) -> TargetMaskResult:
        if not 0 <= unselected_response < response:
            raise ValueError("unselected responses must be fewer than all responses")
        stored = prompt + task + rationale + response + observation
        predicted = stored - 1
        selected = response - unselected_response
        with_rat = selected + rationale
        return TargetMaskResult(
            stored=stored,
            predicted=predicted,
            selected_response=selected,
            targets_with_rationale=with_rat,
            targets_without_rationale=selected,
            observation_share=observation / stored,
            share_with_rationale=with_rat / stored,
            share_without_rationale=selected / stored,
            context_with_rationale=stored - with_rat,
            context_without_rationale=stored - selected,
            other_context_share=(stored - observation - selected) / stored,
            dilution_without_rationale=predicted / selected,
            dilution_with_rationale=predicted / with_rat,
            unselected_vs_selected=unselected_response / selected,
            unselected_share_of_responses=unselected_response / response,
        )


class PerInvocationSerializationModel(ForwardModel):
    """
    One training example per model call against one stored transcript.

    When each of ``H`` turns adds about ``a`` tokens, call ``t`` carries ``t a``
    tokens, so separate examples store ``a H (H + 1) / 2`` tokens against
    ``a H`` for one transcript: the append-only resend triangle
    (``calc_append_only_context`` with ``s0 = delta = a``), an expansion of
    ``(H + 1) / 2``.
    """
    requires = ()
    produces = PerInvocationSerializationResult

    def solve(self, turns: int, tokens_per_turn: int) -> PerInvocationSerializationResult:
        last, total = calc_append_only_context(tokens_per_turn, tokens_per_turn, turns)
        return PerInvocationSerializationResult(
            per_invocation_tokens=total, transcript_tokens=last, expansion=total / last)




class LossNormalizationModel(ForwardModel):
    """
    One long trajectory's weight in a batch under token and example averaging.

    The batch holds the trajectory, split into ``focus_pieces`` records that
    each select every turn's response once, and ``other_examples`` examples of
    ``other_targets`` selected targets each. Token averaging weights each
    example by its target count, so splitting changes nothing; example
    averaging weights each record once, so the trajectory's share rises from
    ``1/B`` to ``k/(B - 1 + k)``.
    """
    requires = ()
    produces = LossNormalizationResult

    def solve(self, focus_targets: int, other_examples: int, other_targets: int,
              focus_pieces: int = 1) -> LossNormalizationResult:
        shares = calc_token_average_shares([focus_targets] + [other_targets] * other_examples)
        records = focus_pieces + other_examples
        return LossNormalizationResult(
            examples=records,
            total_targets=focus_targets + other_examples * other_targets,
            focus_token_share=shares[0],
            other_token_share=shares[1],
            focus_example_share=calc_example_average_share(focus_pieces, records),
            other_example_share=calc_example_average_share(1, records),
            focus_over_other=focus_targets / other_targets,
        )


class SequencePackingModel(ForwardModel):
    """
    Variable-length examples padded to the longest against packed into fixed buffers.

    Padding allocates ``N`` rows of the longest length, so occupancy is the
    mean length over the maximum. Packing places examples longest first into
    the first ``capacity``-token buffer with room (first-fit decreasing).
    Occupancy counts context and targets alike; it is not measured throughput.
    """
    requires = ()
    produces = SequencePackingResult

    def solve(self, lengths, capacity: int) -> SequencePackingResult:
        lengths = list(lengths)
        stored = sum(lengths)
        longest = max(lengths)
        padded = len(lengths) * longest
        loads = calc_first_fit_decreasing(lengths, capacity)
        packed = len(loads) * capacity
        return SequencePackingResult(
            examples=len(lengths), stored=stored, longest=longest,
            mean_length=stored / len(lengths), padded_positions=padded,
            buffers=len(loads), buffer_loads=loads, packed_positions=packed,
            padded_occupancy=stored / padded, packed_occupancy=stored / packed,
            position_ratio=padded / packed)


class PreferencePairModel(ForwardModel):
    """
    One preference pair under the DPO loss, against its value at a zero margin.

    At initialization the policy equals the reference, so both scores are zero,
    the loss is ``ln 2``, the saturation factor is 1/2, and the coefficient is
    ``beta / 2``. The logistic score depends on ``beta``, so it measures fit to
    this comparison, not a success probability.
    """
    requires = ()
    produces = PreferencePairResult

    def solve(self, policy_preferred: float, reference_preferred: float, policy_dispreferred: float,
              reference_dispreferred: float, beta: float) -> PreferencePairResult:
        zw, zl, margin, score, loss, sat, coef = calc_preference_pair(
            policy_preferred, reference_preferred, policy_dispreferred, reference_dispreferred, beta)
        _, _, _, _, loss0, sat0, coef0 = calc_preference_pair(0.0, 0.0, 0.0, 0.0, beta)
        return PreferencePairResult(
            score_preferred=zw, score_dispreferred=zl, margin=margin, preference_score=score,
            loss=loss, initial_loss=loss0, saturation_factor=sat, initial_saturation_factor=sat0,
            gradient_coefficient=coef, initial_gradient_coefficient=coef0)




