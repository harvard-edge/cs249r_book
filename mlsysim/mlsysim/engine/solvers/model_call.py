"""Model-call solvers: what one call admits, samples, and costs, and what a trajectory resends.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each model
is arithmetic on stated token counts and per-token floors: admission against a
context window, temperature and top-p on a fixed logit vector, Bayes' rule for
a fallible check, and prefill plus decode time at the weight-transfer floors.
None models queueing, batching, attention, or serving overhead.
"""

from __future__ import annotations

import statistics
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

    def solve(self, window: int, output_reserve: int, text_tokens: int,
              image_tokens: int = 0) -> ContextAdmissionResult:
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

    def solve(self, logits: Sequence[float], base_temperature: float, sharp_temperature: float,
              top_p: float) -> SamplingResult:
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

    def solve(self, defect_rate: float, false_accept_rate: float,
              valid_accept_rate: float) -> CheckerPosteriorResult:
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

    def solve(self, prefill_per_token: Quantity, decode_per_token: Quantity, prompt_tokens: int,
              output_tokens: int, shared_prefix_tokens: int = 0, deadline: Optional[Quantity] = None,
              stage_success: Optional[Dict[str, float]] = None) -> ModelCallResult:
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

    def solve(self, prefill_per_token: Quantity, decode_per_token: Quantity, first_prompt: int,
              appended: int, output_per_turn: int, turns: int) -> TrajectoryResendResult:
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


