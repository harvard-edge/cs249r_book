"""Model-call solvers: what one call admits, samples, and costs, and what a trajectory resends.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each model
is arithmetic on stated token counts and per-token floors: admission against a
context window, temperature and top-p on a fixed logit vector, Bayes' rule for
a fallible check, and prefill plus decode time at the weight-transfer floors.
None models queueing, batching, attention, or serving overhead.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from ..results import (
    CheckerPosteriorResult,
    ContextAdmissionResult,
    ModelCallResult,
    SamplingResult,
    TrajectoryResendResult,
)
from ...core.types import Quantity
from ...physics.agents import (
    calc_nucleus,
    calc_pass_posterior,
    calc_append_only_context,
    calc_reused_prefix_prompt_tokens,
    calc_tempered_softmax,
)
from .base import ForwardModel


def _ratio(a, b) -> float:
    return (a / b).to("").magnitude


class ContextAdmissionModel(ForwardModel):
    """
    Token-measured admission: a request fits when ``S <= S_max - K_max``.

    The serialized prompt (text plus any image charge) must fit in the context
    window with ``output_reserve`` tokens held back for the reply.
    """

    requires = ()
    produces = ContextAdmissionResult

    def solve(
        self, window: int, output_reserve: int, text_tokens: int, image_tokens: int = 0
    ) -> ContextAdmissionResult:
        """Check a serialized prompt against a reserved context capacity.

        Parameters
        ----------
        window : int
            Positive total context-window capacity, in tokens.
        output_reserve : int
            Nonnegative tokens reserved for output, at most window.
        text_tokens : int
            Nonnegative serialized text-token count.
        image_tokens : int, optional
            Nonnegative image charge expressed in tokens; defaults to zero.

        Returns
        -------
        ContextAdmissionResult
            Token-valued scalar capacity, demand and excess, with an admission
            boolean. Headroom is measured before the image charge.

        Notes
        -----
        Equation: Input ceiling is window - output_reserve. Total demand is
        text_tokens + image_tokens + output_reserve; admission requires demand <= window.

        Assumptions: All counts use the serving model's already serialized token
        accounting. A negative excess denotes spare capacity.

        Excludes: Tokenization, KV-memory feasibility, attention work, and any
        claim that an admitted request can meet a latency target.

        References
        ----------
        Derivation: The request must fit the sum of its text, image, and output
        reservation in a finite token capacity; this is a capacity-accounting identity.

        Examples
        --------
        >>> from mlsysim.solvers import ContextAdmissionModel
        >>> result = ContextAdmissionModel().solve(100, 10, 80, 20)
        >>> result.admitted
        False
        """
        ceiling = window - output_reserve
        demand = text_tokens + image_tokens + output_reserve
        return ContextAdmissionResult(
            input_ceiling=ceiling,
            headroom=ceiling - text_tokens,
            total_demand=demand,
            excess=demand - window,
            admitted=demand <= window,
        )


class SamplingModel(ForwardModel):
    """
    Temperature and top-p on one fixed logit vector.

    Softmax at the base and the sharpened temperature, the probability left to
    every non-top token (the tail), and the top-p nucleus at the base
    temperature, renormalized.
    """

    requires = ()
    produces = SamplingResult

    def solve(
        self, logits: Sequence[float], base_temperature: float, sharp_temperature: float, top_p: float
    ) -> SamplingResult:
        """Compare temperature and nucleus filtering for a fixed logit vector.

        Parameters
        ----------
        logits : Sequence[float]
            Finite dimensionless logits for at least two possible tokens.
        base_temperature : float
            Positive temperature used for the base distribution and top-p filter.
        sharp_temperature : float
            Positive comparison temperature; smaller values sharpen the distribution.
        top_p : float
            Cumulative probability threshold in (0, 1] for retaining the largest base probabilities.

        Returns
        -------
        SamplingResult
            Dimensionless probability vectors, partition sum, tail ratio,
            nucleus size and its retained probability mass.

        Notes
        -----
        Equation: At temperature T, p_i=exp(z_i/T)/sum_j exp(z_j/T). The
        nucleus is the smallest descending-probability prefix whose mass reaches top_p.

        Assumptions: Both distributions use identical fixed logits. Inputs must
        avoid overflow in the reported direct exponential partition sum and a
        zero comparison tail in the tail ratio.

        Excludes: Random token draws, changes in future logits, and a prediction
        of quality, factuality, or diversity over a generated sequence.

        References
        ----------
        Derivation: Softmax normalizes positive exponential weights. Sorting
        probabilities and accumulating to the chosen threshold defines the nucleus;
        physics.agents.calc_tempered_softmax and calc_nucleus implement these steps.

        Examples
        --------
        >>> from mlsysim.solvers import SamplingModel
        >>> result = SamplingModel().solve([0.0, 1.0, 2.0], 1.0, 0.5, 0.8)
        >>> result.sharp_tail < result.base_tail
        True
        """
        import math

        base = calc_tempered_softmax(logits, base_temperature)
        sharp = calc_tempered_softmax(logits, sharp_temperature)
        kept, mass, renorm = calc_nucleus(base, top_p)
        base_tail, sharp_tail = 1 - max(base), 1 - max(sharp)
        return SamplingResult(
            partition_sum=sum(math.exp(z / base_temperature) for z in logits),
            base_probabilities=base,
            sharp_probabilities=sharp,
            base_tail=base_tail,
            sharp_tail=sharp_tail,
            tail_ratio=base_tail / sharp_tail,
            nucleus_size=kept,
            nucleus_mass=mass,
            nucleus_probabilities=renorm,
        )


class CheckerPosteriorModel(ForwardModel):
    """
    Defects that survive a fallible check, by Bayes' rule.

    ``false_accept_rate`` is P(PASS | defective) and ``valid_accept_rate`` is
    P(PASS | valid). Rerunning the same check is not modeled: its errors are
    correlated with the first pass.
    """

    requires = ()
    produces = CheckerPosteriorResult

    def solve(self, defect_rate: float, false_accept_rate: float, valid_accept_rate: float) -> CheckerPosteriorResult:
        """Compute defect prevalence among items that pass a fallible check.

        Parameters
        ----------
        defect_rate : float
            Prior defective-item fraction, in [0, 1].
        false_accept_rate : float
            Conditional probability of passing a defective item, in [0, 1].
        valid_accept_rate : float
            Conditional probability of passing a valid item, in [0, 1].

        Returns
        -------
        CheckerPosteriorResult
            Dimensionless joint defective-pass rate, overall pass share,
            posterior defect probability, catch rate, and valid rejection rate.

        Notes
        -----
        Equation: P(defect|pass) = ``defect_rate*false_accept_rate`` /
        ``(defect_rate*false_accept_rate`` + ``(1-defect_rate)*valid_accept_rate)``.

        Assumptions: The prior and conditional pass rates describe the same
        population; total pass probability must be positive.

        Excludes: Correlated repeated checks, prevalence drift, and estimating
        checker accuracy from data.

        References
        ----------
        Derivation: Partition passing items into defective and valid populations
        and normalize the defective joint probability by their sum (Bayes' rule).
        physics.agents.calc_pass_posterior owns the normalization.

        Examples
        --------
        >>> from mlsysim.solvers import CheckerPosteriorModel
        >>> result = CheckerPosteriorModel().solve(0.2, 0.1, 0.9)
        >>> result.posterior < 0.2
        True
        """
        defective_pass = false_accept_rate * defect_rate
        return CheckerPosteriorResult(
            defective_pass=defective_pass,
            pass_share=defective_pass + valid_accept_rate * (1 - defect_rate),
            posterior=calc_pass_posterior(defect_rate, false_accept_rate, valid_accept_rate),
            catch_rate=1 - false_accept_rate,
            valid_reject_rate=1 - valid_accept_rate,
        )


class ModelCallModel(ForwardModel):
    """
    One model call at per-token floors: ``S * tau_prefill + K * tau_decode``.

    ``prefill_per_token`` is the compute floor ``2P / R_peak`` and
    ``decode_per_token`` the weight-transfer floor ``P b / BW`` (see
    ``DecodeBoundModel``). A shared prefix reused from cache leaves only the
    remaining prompt for fresh prefill. Optional: a deadline, and the success
    rates of the stages a reply must clear to be accepted.
    """

    requires = ()
    produces = ModelCallResult

    def solve(
        self,
        prefill_per_token: Quantity,
        decode_per_token: Quantity,
        prompt_tokens: int,
        output_tokens: int,
        shared_prefix_tokens: int = 0,
        deadline: Optional[Quantity] = None,
        stage_success: Optional[Dict[str, float]] = None,
    ) -> ModelCallResult:
        """Account for one model call at supplied per-token timing floors.

        Parameters
        ----------
        prefill_per_token : Quantity
            Nonnegative time per counted prompt token, convertible to seconds.
        decode_per_token : Quantity
            Positive time per counted output token, convertible to seconds.
        prompt_tokens : int
            Nonnegative prompt-token count.
        output_tokens : int
            Nonnegative generated-token count; prompt plus output must be positive.
        shared_prefix_tokens : int, optional
            Reused prompt-token count from zero through prompt_tokens; defaults to zero.
        deadline : Quantity, optional
            Nonnegative response-time target in seconds; None omits its ratio.
        stage_success : Dict[str, float], optional
            Named stage-success probabilities in [0, 1]; multiplied when nonempty.

        Returns
        -------
        ModelCallResult
            Durations in seconds, decode rate in inverse seconds, token
            counts and dimensionless shares. Acceptance is None without stages.

        Notes
        -----
        Equation: Total cold time is ``prompt_tokens*prefill_per_token`` +
        ``output_tokens*decode_per_token``. Fresh prefill subtracts shared_prefix_tokens
        from prompt_tokens; it does not replace the reported cold total.

        Assumptions: Per-token floors stay constant and phases execute serially.
        Total time is positive; supplied stage probabilities multiply independently.

        Excludes: Queueing, attention-context growth, batching, framework overhead,
        cache maintenance, and any inference of acceptance from timing.

        References
        ----------
        Derivation: Multiply per-token time by its token count, then sum serial
        phases. Independent stage success probabilities multiply. Timing bounds
        are supplied inputs, not measurements produced by this model.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import ModelCallModel
        >>> result = ModelCallModel().solve(Q_(1, "ms"), Q_(2, "ms"), 10, 2, 5)
        >>> result.fresh_prefill_time < result.prefill_time
        True
        """
        pre = (prompt_tokens * prefill_per_token).to("second")
        dec = (output_tokens * decode_per_token).to("second")
        total = pre + dec
        acceptance = None
        if stage_success:
            acceptance = 1.0
            for p in stage_success.values():
                acceptance *= p
        return ModelCallResult(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            prefill_time=pre,
            decode_time=dec,
            total_time=total,
            output_token_share=output_tokens / (prompt_tokens + output_tokens),
            output_time_share=_ratio(dec, total),
            fresh_prefill_time=((prompt_tokens - shared_prefix_tokens) * prefill_per_token).to("second"),
            decode_rate=(1 / decode_per_token).to("1/second"),
            deadline_margin=_ratio(deadline, total) if deadline is not None else None,
            acceptance=acceptance,
        )


class TrajectoryResendModel(ForwardModel):
    """
    Input a trajectory resends when every call carries the whole history.

    Turn ``t`` sends the first prompt ``S0`` plus ``t - 1`` appended blocks of
    ``D`` tokens (model output plus tool observation). Without reuse the total
    is ``H S0 + D H (H - 1) / 2``; reusing each earlier prompt prefix leaves
    ``S0 + (H - 1) D`` of fresh prefill. Times are at the stated floors.
    """

    requires = ()
    produces = TrajectoryResendResult

    def solve(
        self,
        prefill_per_token: Quantity,
        decode_per_token: Quantity,
        first_prompt: int,
        appended: int,
        output_per_turn: int,
        turns: int,
    ) -> TrajectoryResendResult:
        """Count repeated prompt work across an append-only trajectory.

        Parameters
        ----------
        prefill_per_token : Quantity
            Nonnegative time per input token, convertible to seconds.
        decode_per_token : Quantity
            Nonnegative time per generated token, convertible to seconds.
        first_prompt : int
            Positive number of tokens in the first request.
        appended : int
            Nonnegative output-plus-observation tokens added between calls.
        output_per_turn : int
            Nonnegative output tokens generated on each turn.
        turns : int
            Positive number of calls.

        Returns
        -------
        TrajectoryResendResult
            Token counts, dimensionless reuse reduction and input share,
            and prefill/decode durations in seconds.

        Notes
        -----
        Equation: Without prefix reuse, input is ``H*S`` + ``D*H*(H-1)/2``. Perfect
        reuse computes ``S+(H-1)*D`` fresh tokens; output is ``H*output_per_turn``.

        Assumptions: Each turn appends the same token count. Reuse retains every
        earlier compatible prompt prefix, and per-token timing floors stay fixed.

        Excludes: Cache eviction, context-window admission, cache-transfer time,
        and the context dependence of attention or decode work.

        References
        ----------
        Derivation: The resend total is an arithmetic-series sum. Counting the
        first prompt and each later increment only once gives fresh work under
        prefix reuse; helpers in physics.agents own both token identities.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import TrajectoryResendModel
        >>> result = TrajectoryResendModel().solve(Q_(1, "ms"), Q_(2, "ms"), 10, 2, 1, 3)
        >>> result.reused_prefill_time < result.prefill_time
        True
        """
        resent = calc_append_only_context(first_prompt, appended, turns)[1]
        reused = calc_reused_prefix_prompt_tokens(turns, first_prompt, appended)
        out = turns * output_per_turn
        return TrajectoryResendResult(
            turns=turns,
            first_prompt_term=turns * first_prompt,
            appended_term=resent - turns * first_prompt,
            resent_tokens=resent,
            reused_prefix_tokens=reused,
            reuse_reduction=resent / reused,
            output_tokens=out,
            input_share=resent / (resent + out),
            prefill_time=(resent * prefill_per_token).to("second"),
            reused_prefill_time=(reused * prefill_per_token).to("second"),
            decode_time=(out * decode_per_token).to("second"),
        )
