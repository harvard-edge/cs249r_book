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
from ...physics.agents import calc_append_only_context
from ...physics.fine_tuning import (
    calc_example_average_share,
    calc_first_fit_decreasing,
    calc_preference_pair,
    calc_token_average_shares,
)


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

    def solve(
        self, prompt: int, task: int, rationale: int, response: int, observation: int, unselected_response: int
    ) -> TargetMaskResult:
        """Count stored context and selected prediction targets.

        Parameters
        ----------
        prompt : int
            Prompt token count, nonnegative.
        task : int
            Task-description token count, nonnegative.
        rationale : int
            Rationale token count, nonnegative.
        response : int
            Response token count, positive.
        observation : int
            Observation token count, nonnegative.
        unselected_response : int
            Response tokens retained as context; from zero to response minus one.

        Returns
        -------
        TargetMaskResult
            Token counts and dimensionless target shares and dilution ratios.

        Notes
        -----
        Equation: stored = prompt + task + rationale + response + observation; selected = response - unselected_response; predicted = stored - 1.

        Assumptions: Counts refer to a single serialized sequence; the first position has no preceding predictor.

        Excludes: Tokenizer behavior, model quality, and loss values.

        References
        ----------
        Derivation: Subtracting excluded responses counts targets; dividing by stored positions or prediction terms gives the reported shares and dilution.

        Examples
        --------
        >>> TargetMaskModel().solve(2, 1, 1, 4, 2, 1).selected_response
        3
        """
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
        """Compare repeated call prefixes with one stored transcript.

        Parameters
        ----------
        turns : int
            Positive number of turns.
        tokens_per_turn : int
            Positive number of tokens appended per turn.

        Returns
        -------
        PerInvocationSerializationResult
            Token counts and a dimensionless storage expansion.

        Notes
        -----
        Equation: transcript = a H; per_invocation = a H (H + 1) / 2.

        Assumptions: Every turn appends the same a tokens and stores its full prefix.

        Excludes: Compression, deduplication, tokenization, and model execution.

        References
        ----------
        Derivation: Summing prefix lengths a, 2a, ..., Ha yields the arithmetic-series total.

        Examples
        --------
        >>> PerInvocationSerializationModel().solve(3, 10).expansion
        2.0
        """
        last, total = calc_append_only_context(tokens_per_turn, tokens_per_turn, turns)
        return PerInvocationSerializationResult(
            per_invocation_tokens=total, transcript_tokens=last, expansion=total / last
        )


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

    def solve(
        self, focus_targets: int, other_examples: int, other_targets: int, focus_pieces: int = 1
    ) -> LossNormalizationResult:
        """Compare target-token weighting with equal-record weighting.

        Parameters
        ----------
        focus_targets : int
            Positive selected-target count of the focus trajectory.
        other_examples : int
            Positive count of other records.
        other_targets : int
            Positive target count in each other record.
        focus_pieces : int, optional
            Positive number of focus records after splitting; default one.

        Returns
        -------
        LossNormalizationResult
            Counts and dimensionless weighting shares.

        Notes
        -----
        Equation: focus_token_share = focus_targets / (focus_targets + other_examples * other_targets); focus_example_share = focus_pieces / (focus_pieces + other_examples).

        Assumptions: Splitting preserves all focus targets; other records have equal target counts.

        Excludes: Gradient magnitudes, optimization dynamics, and changes in training quality.

        References
        ----------
        Derivation: Token averaging normalizes by target counts; equal-record averaging normalizes by the count of records.

        Examples
        --------
        >>> LossNormalizationModel().solve(20, 2, 10).focus_token_share
        0.5
        """
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
        """Compare padding with first-fit-decreasing sequence packing.

        Parameters
        ----------
        lengths : sequence of int
            Nonempty sequence of positive token counts, each no larger than capacity.
        capacity : int
            Positive token capacity of each packing buffer.

        Returns
        -------
        SequencePackingResult
            Token-position counts, buffer loads, and dimensionless occupancies.

        Notes
        -----
        Equation: padded = N * max(lengths); packed = number_of_buffers * capacity; occupancy = sum(lengths) / allocated_positions.

        Assumptions: Examples remain whole; first-fit decreasing determines the buffer placement.

        Excludes: Attention masking costs, runtime throughput, and optimal bin packing.

        References
        ----------
        Derivation: Allocated positions count rectangular padding or fixed buffers; occupancy is occupied divided by allocated positions.

        Examples
        --------
        >>> SequencePackingModel().solve([2, 2], 4).packed_occupancy
        1.0
        """
        lengths = list(lengths)
        stored = sum(lengths)
        longest = max(lengths)
        padded = len(lengths) * longest
        loads = calc_first_fit_decreasing(lengths, capacity)
        packed = len(loads) * capacity
        return SequencePackingResult(
            examples=len(lengths),
            stored=stored,
            longest=longest,
            mean_length=stored / len(lengths),
            padded_positions=padded,
            buffers=len(loads),
            buffer_loads=loads,
            packed_positions=packed,
            padded_occupancy=stored / padded,
            packed_occupancy=stored / packed,
            position_ratio=padded / packed,
        )


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

    def solve(
        self,
        policy_preferred: float,
        reference_preferred: float,
        policy_dispreferred: float,
        reference_dispreferred: float,
        beta: float,
    ) -> PreferencePairResult:
        """Evaluate one DPO preference margin and logistic loss.

        Parameters
        ----------
        policy_preferred : float
            Log probability of the preferred response under the policy, dimensionless.
        reference_preferred : float
            Log probability of the preferred response under the reference, dimensionless.
        policy_dispreferred : float
            Log probability of the dispreferred response under the policy, dimensionless.
        reference_dispreferred : float
            Log probability of the dispreferred response under the reference, dimensionless.
        beta : float
            Positive dimensionless scale of the policy/reference log-ratio margin.

        Returns
        -------
        PreferencePairResult
            Dimensionless margin, logistic score, loss, saturation, and gradient coefficient.

        Notes
        -----
        Equation: margin = beta * ((policy_preferred - reference_preferred) - (policy_dispreferred - reference_dispreferred)); loss = -log(sigmoid(margin)).

        Assumptions: The supplied log probabilities describe one preferred/dispreferred pair; initialization has zero log-ratio margin.

        Excludes: Sequence generation, empirical success probability, and training dynamics.

        References
        ----------
        Derivation: The negative log-sigmoid gives pairwise logistic loss; differentiating the margin yields the saturation factor and beta-scaled coefficient. This wrapper delegates the calculation to calc_preference_pair.

        Examples
        --------
        >>> PreferencePairModel().solve(0.0, 0.0, 0.0, 0.0, 0.2).preference_score
        0.5
        """
        zw, zl, margin, score, loss, sat, coef = calc_preference_pair(
            policy_preferred, reference_preferred, policy_dispreferred, reference_dispreferred, beta
        )
        _, _, _, _, loss0, sat0, coef0 = calc_preference_pair(0.0, 0.0, 0.0, 0.0, beta)
        return PreferencePairResult(
            score_preferred=zw,
            score_dispreferred=zl,
            margin=margin,
            preference_score=score,
            loss=loss,
            initial_loss=loss0,
            saturation_factor=sat,
            initial_saturation_factor=sat0,
            gradient_coefficient=coef,
            initial_gradient_coefficient=coef0,
        )
