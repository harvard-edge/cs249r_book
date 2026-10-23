"""Test-time deliberation solvers: spending more work on one decision with the weights fixed.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each
model is arithmetic on stated token counts, pass rates, check error rates, and
the per-token decode floor of a ``DecodeScenario``: how one output ceiling
turns into different latencies and bills as depth, breadth, or feedback; what a
pool of independent candidates holds and what a fallible check lets through;
how argmax selection concentrates scoring error; how a step check bounds a
search tree; and what flat sampling, beam search, and feedback policies spend
under one deadline. None models queueing, batching, prefill, or serving
overhead unless a scenario states it.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional, Sequence

from ..results import (
    AllocationAxesResult,
    ArgmaxNoiseResult,
    BestOfNPoolResult,
    CoverageVsVoteResult,
    DeadlinePolicyResult,
    ProbeVsAnswerResult,
    PrunedSearchTreeResult,
    ScopedRevisionResult,
    SearchAccountsResult,
    VerifiedPoolResult,
)
from ...core.types import Quantity
from ...core.units import USD
from ...physics.agents import (
    calc_append_only_context,
    calc_expected_max_std_normal,
    calc_independent_success,
    calc_pass_at_k_unbiased,
    calc_pruned_tree_nodes,
)
from .base import ForwardModel


def _per_token(price: Quantity) -> float:
    """Dollars per token from a price such as ``Q_(3.00, "USD / Mtoken")``."""
    return price.to("USD/token").magnitude


def _usd(x: float) -> Quantity:
    return x * USD


def _ratio(a, b) -> float:
    return (a / b).to("").magnitude


class ProbeVsAnswerModel(ForwardModel):
    """
    A generated explanation of ``K`` tokens against a prepared test of ``T_probe``.

    Generation time is ``K * tau_decode`` at the weight-transfer floor; the
    charge is ``K`` output tokens at the output price. Prefill, queueing, and
    test preparation are excluded.
    """
    requires = ()
    produces = ProbeVsAnswerResult

    def solve(self, decode_per_token: Quantity, answer_tokens: int, probe_time: Quantity,
              output_price: Quantity) -> ProbeVsAnswerResult:
        token_time = decode_per_token.to("second")
        answer_time = answer_tokens * token_time
        return ProbeVsAnswerResult(
            token_time=token_time,
            answer_time=answer_time,
            answer_cost=_usd(answer_tokens * _per_token(output_price)),
            time_ratio=_ratio(answer_time, probe_time),
        )


class AllocationAxesModel(ForwardModel):
    """
    One output ceiling spent three ways: depth, breadth, and feedback.

    Depth generates the whole ceiling in one chain. Breadth runs ``N``
    concurrent candidates, each billed its own copy of the ``S``-token prompt,
    so it re-reads ``(N - 1) S`` tokens. Feedback runs ``H`` rounds of a
    candidate plus a check; each later request appends the earlier candidates
    and reports, so input is ``H S + (K + R) H (H - 1) / 2``. Times are
    generation at the decode floor plus checks; bills count model tokens only.
    """
    requires = ()
    produces = AllocationAxesResult

    def solve(self, decode_per_token: Quantity, prompt: int, ceiling: int, candidates: int,
              candidate_tokens: int, rounds: int, report_tokens: int, check_time: Quantity,
              input_price: Quantity, output_price: Quantity,
              context_growth: Optional[int] = None) -> AllocationAxesResult:
        t = decode_per_token.to("second")
        p_in, p_out = _per_token(input_price), _per_token(output_price)
        depth_in, depth_out = prompt, ceiling
        breadth_in, breadth_out = candidates * prompt, candidates * candidate_tokens
        feedback_in = calc_append_only_context(prompt, candidate_tokens + report_tokens, rounds)[1]
        feedback_out = rounds * candidate_tokens
        depth_time = depth_out * t
        breadth_time = candidate_tokens * t
        feedback_time = rounds * (candidate_tokens * t + check_time)
        depth_cost = _usd(depth_in * p_in + depth_out * p_out)
        breadth_cost = _usd(breadth_in * p_in + breadth_out * p_out)
        feedback_cost = _usd(feedback_in * p_in + feedback_out * p_out)
        premium = breadth_in - depth_in
        grown = (candidates - 1) * prompt * context_growth if context_growth else None
        return AllocationAxesResult(
            token_time=t,
            depth_input=depth_in, depth_output=depth_out, depth_time=depth_time, depth_cost=depth_cost,
            breadth_input=breadth_in, breadth_output=breadth_out, breadth_time=breadth_time,
            breadth_cost=breadth_cost,
            feedback_input=feedback_in, feedback_output=feedback_out, feedback_time=feedback_time,
            feedback_cost=feedback_cost,
            path_ratio=_ratio(depth_time, breadth_time),
            bill_ratio=_ratio(breadth_cost, depth_cost),
            breadth_premium=premium,
            serial_breadth_time=candidates * candidate_tokens * t,
            grown_premium=grown,
            grown_premium_cost=_usd(grown * p_in) if grown is not None else None,
        )


class BestOfNPoolModel(ForwardModel):
    """
    A parallel pool of ``N`` independent candidates against sequential retry.

    pass@N is ``1 - (1 - p)^N``. A parallel pool always generates ``N``
    candidates; retrying one at a time and stopping at the first pass (capped
    at ``N``) generates ``[1 - (1 - p)^N] / p`` on average, ``1 / p`` per
    success. A mixed workload whose tasks pass at different rates must apply
    pass@N per task before averaging; averaging ``p`` first overstates it.
    The selector is assumed to return a passing member whenever one exists.
    """
    requires = ()
    produces = BestOfNPoolResult

    def solve(self, pass_rate: float, pool_sizes: Sequence[int], easy_pass_rate: float,
              hard_pass_rate: float) -> BestOfNPoolResult:
        p = pass_rate
        pass_at = {n: calc_independent_success(p, n) for n in pool_sizes}
        n_max = pool_sizes[-1]
        pass_max = pass_at[n_max]
        parallel = n_max / pass_max
        seq_candidates = pass_max / p
        seq_per_success = seq_candidates / pass_max
        mixed = (easy_pass_rate + hard_pass_rate) / 2
        easy = calc_independent_success(easy_pass_rate, n_max)
        return BestOfNPoolResult(
            pass_at=pass_at,
            last_gain=p * (1 - p) ** (n_max - 1),
            parallel_per_success=parallel,
            sequential_candidates=seq_candidates,
            sequential_per_success=seq_per_success,
            cost_ratio=parallel / seq_per_success,
            mixed_average_rate=mixed,
            mixed_plug_in=calc_independent_success(mixed, n_max),
            mixed_true=(easy + calc_independent_success(hard_pass_rate, n_max)) / 2,
            easy_pass_at_max=easy,
        )




class CoverageVsVoteModel(ForwardModel):
    """
    Pool coverage against what majority voting returns, from per-sample labels.

    Coverage (pass@k averaged over problems) at one sample and at the largest
    pool, for a small and a large model; majority-vote accuracy at the largest
    pool; the gains of each from a knee to the largest pool; and repository
    repair coverage at one sample and at its pool size.
    """
    requires = ("series",)
    produces = CoverageVsVoteResult

    @staticmethod
    def _at(rows, model, dataset, k, field):
        return next(r[field] for r in rows if r["model"] == model and r["dataset"] == dataset and r["k"] == k)

    def solve(self, majority_rows, coverage_rows, small_model: str, large_model: str, dataset: str,
              k_max: int, k_knee: int, repair_system: str, repair_dataset: str,
              repair_k: int) -> CoverageVsVoteResult:
        at = self._at
        cov, vote = "coverage", "majority_vote_accuracy"
        gains = [at(majority_rows, m, dataset, k_max, vote) - at(majority_rows, m, dataset, k_knee, vote)
                 for m in (small_model, large_model)]
        cov_small = at(majority_rows, small_model, dataset, k_max, cov)
        return CoverageVsVoteResult(
            coverage_small=cov_small,
            coverage_large=at(majority_rows, large_model, dataset, k_max, cov),
            coverage_small_first=at(majority_rows, small_model, dataset, 1, cov),
            coverage_large_first=at(majority_rows, large_model, dataset, 1, cov),
            vote_small=at(majority_rows, small_model, dataset, k_max, vote),
            vote_large=at(majority_rows, large_model, dataset, k_max, vote),
            vote_gain_small=gains[0],
            vote_gain_after_knee=max(0.0, *gains),
            coverage_gain_small=cov_small - at(majority_rows, small_model, dataset, k_knee, cov),
            repair_first=at(coverage_rows, repair_system, repair_dataset, 1, cov),
            repair_max=at(coverage_rows, repair_system, repair_dataset, repair_k, cov),
        )


class VerifiedPoolModel(ForwardModel):
    """
    First-passing selection from a fixed pool of ``n`` under a fallible check.

    Each candidate is valid with rate ``v``; the check rejects valid ones with
    ``eps_FR`` and passes invalid ones with ``eps_FA``, so a candidate passes
    with ``q = v (1 - eps_FR) + (1 - v) eps_FA``. Passing-set precision is
    ``v (1 - eps_FR) / q``, one minus the Bayes posterior of a defect given a
    pass (``CheckerPosteriorModel`` with ``defect_rate = 1 - v``). The pool
    selects with ``1 - (1 - q)^n`` and otherwise abstains; a selection is
    defective with ``1 - precision``. An incident loss prices
    each defective selection; the output bill prices the whole pool.
    """
    requires = ()
    produces = VerifiedPoolResult

    def solve(self, pool_size: int, valid_rate: float, false_reject_rate: float,
              false_accept_rate: float, tokens_per_candidate: Quantity, output_price: Quantity,
              incident_loss: Quantity) -> VerifiedPoolResult:
        n, v = pool_size, valid_rate
        valid_pass = v * (1 - false_reject_rate)
        invalid_pass = (1 - v) * false_accept_rate
        q = valid_pass + invalid_pass
        precision = valid_pass / q
        defect = 1 - precision
        any_sel = 1 - (1 - q) ** n
        wrong = any_sel * defect
        bill = (n * tokens_per_candidate * output_price).to(USD)
        loss = incident_loss.to(USD)
        task_loss = wrong * loss
        return VerifiedPoolResult(
            valid_pass=valid_pass,
            invalid_pass=invalid_pass,
            pass_probability=q,
            precision=precision,
            defect_given_selection=defect,
            any_selection=any_sel,
            no_selection=1 - any_sel,
            wrong_selection=wrong,
            any_false_pass=1 - (1 - invalid_pass) ** n,
            expected_valid_passes=n * valid_pass,
            expected_invalid_passes=n * invalid_pass,
            expected_passes=n * q,
            conditional_loss=defect * loss,
            task_loss=task_loss,
            output_bill=bill,
            loss_to_bill=_ratio(task_loss, bill),
        )


class ArgmaxNoiseModel(ForwardModel):
    """
    Argmax over ``N`` equally good candidates whose scores carry independent noise.

    The winner's expected scoring error is the expected maximum of ``N``
    standard normals (in units of the noise's spread), below the large-``N``
    asymptote ``sqrt(2 ln N)``. A loophole present in a fraction ``g`` of
    single samples appears in a pool of ``N`` with ``1 - (1 - g)^N``.
    """
    requires = ()
    produces = ArgmaxNoiseResult

    def solve(self, pool_sizes: Sequence[int], large_pool: int, exploit_rate: float) -> ArgmaxNoiseResult:
        return ArgmaxNoiseResult(
            expected_max={n: calc_expected_max_std_normal(n) for n in pool_sizes},
            asymptote={n: math.sqrt(2 * math.log(n)) for n in pool_sizes if n > 1},
            exploit_presence=calc_independent_success(exploit_rate, large_pool),
        )


class ScopedRevisionModel(ForwardModel):
    """
    Output of regenerating a whole candidate plan against only its affected nodes.

    Equal-sized nodes; only replaced nodes are generated. Input processing,
    checks, and physical effects are excluded.
    """
    requires = ()
    produces = ScopedRevisionResult

    def solve(self, total_nodes: int, replaced_nodes: int, tokens_per_node: int) -> ScopedRevisionResult:
        full, scoped = total_nodes * tokens_per_node, replaced_nodes * tokens_per_node
        share = scoped / full
        return ScopedRevisionResult(full_output=full, scoped_output=scoped, output_share=share,
                                    avoided_share=1 - share)


class PrunedSearchTreeModel(ForwardModel):
    """
    Size of a search tree whose step check rejects each child with probability ``r``.

    The effective branching factor is ``b (1 - r)``; above ``r = 1 - 1/b`` the
    expected frontier shrinks and the tree dies out on its own. Rejecting a
    parent removes its whole subtree, so the node saving exceeds ``r``.
    """
    requires = ()
    produces = PrunedSearchTreeResult

    def solve(self, branching: int, rejection_rate: float, depth: int,
              wide_branching: Optional[int] = None, wide_depth: Optional[int] = None) -> PrunedSearchTreeResult:
        b_eff = branching * (1 - rejection_rate)
        unpruned, pruned = calc_pruned_tree_nodes(branching, rejection_rate, depth)
        return PrunedSearchTreeResult(
            effective_branching=b_eff,
            critical_rejection=1 - 1 / branching,
            unpruned_nodes=unpruned,
            pruned_nodes=pruned,
            unpruned_leaves=branching ** depth,
            pruned_leaves=b_eff ** depth,
            node_saving=1 - pruned / unpruned,
            flat_cut_nodes=unpruned * (1 - rejection_rate),
            wide_leaves=wide_branching ** wide_depth if wide_branching and wide_depth else None,
        )


class SearchAccountsModel(ForwardModel):
    """
    Flat best-of-N against beam search, kept as separate accounts.

    Flat: ``N`` complete candidates of ``L`` segments of ``s`` tokens, each
    request billed the task prompt. Beam: each retained prefix proposes ``e``
    extensions per stage and the best ``w`` survive; every extension request
    resends the prompt plus its prefix, and the scorer reads each full prefix.
    Generation runs in waves of the stated concurrency at the decode floor;
    scoring and final tests add fixed allowances. No prefix-reuse discount.
    The break-even input share is the input price fraction at which the beam's
    output saving, net of scoring, pays for its repeated input.
    """
    requires = ()
    produces = SearchAccountsResult

    def solve(self, decode_per_token: Quantity, segments: int, segment_tokens: int, prompt_tokens: int,
              flat_candidates: int, beam_width: int, extensions: int, generation_slots: int,
              test_slots: int, score_time: Quantity, test_time: Quantity, input_price: Quantity,
              output_price: Quantity, scorer_price: Quantity) -> SearchAccountsResult:
        t = decode_per_token.to("second")
        p_in, p_out, p_sc = _per_token(input_price), _per_token(output_price), _per_token(scorer_price)
        retained, frontier, beam_prompt, scorer, beam_waves = 1, [], 0, 0, 0
        for depth in range(1, segments + 1):
            expanded = retained * extensions
            frontier.append(expanded)
            beam_prompt += expanded * (prompt_tokens + (depth - 1) * segment_tokens)
            scorer += expanded * depth * segment_tokens
            beam_waves += math.ceil(expanded / generation_slots)
            retained = min(beam_width, expanded)
        beam_segments = sum(frontier)
        beam_out = beam_segments * segment_tokens
        flat_length = segments * segment_tokens
        flat_out = flat_candidates * flat_length
        flat_prompt = flat_candidates * prompt_tokens
        flat_waves = math.ceil(flat_candidates / generation_slots)
        flat_test_waves = math.ceil(flat_candidates / test_slots)
        beam_test_waves = math.ceil(retained / test_slots)
        beam_scoring = beam_segments * score_time
        flat_elapsed = flat_waves * flat_length * t + flat_test_waves * test_time
        beam_elapsed = beam_waves * segment_tokens * t + beam_scoring + beam_test_waves * test_time
        out_saving_cost = _usd((flat_out - beam_out) * p_out)
        scorer_cost = _usd(scorer * p_sc)
        premium = _usd((beam_prompt - flat_prompt) * p_in)
        return SearchAccountsResult(
            token_time=t,
            frontier_counts=frontier,
            beam_segments=beam_segments,
            retained=retained,
            flat_output_tokens=flat_out,
            beam_output_tokens=beam_out,
            flat_prompt_tokens=flat_prompt,
            beam_prompt_tokens=beam_prompt,
            beam_common_prompt_tokens=beam_segments * prompt_tokens,
            scorer_tokens=scorer,
            flat_waves=flat_waves,
            beam_waves=beam_waves,
            flat_test_waves=flat_test_waves,
            beam_test_waves=beam_test_waves,
            beam_scoring_time=beam_scoring.to("second"),
            flat_elapsed=flat_elapsed.to("second"),
            beam_elapsed=beam_elapsed.to("second"),
            flat_token_cost=_usd(flat_prompt * p_in + flat_out * p_out),
            beam_token_cost=_usd(beam_prompt * p_in + beam_out * p_out + scorer * p_sc),
            output_saving=1 - beam_out / flat_out,
            output_saving_cost=out_saving_cost,
            scorer_cost=scorer_cost,
            input_premium_cost=premium,
            breakeven_input_share=_ratio(out_saving_cost - scorer_cost, premium),
        )


class DeadlinePolicyModel(ForwardModel):
    """
    Three deliberation policies for one task under one deadline.

    A: one candidate and one check. B: ``N`` independent candidates generated
    and checked in parallel, pass@N. C: feedback; a failed first candidate
    plus its failure report feeds one revision that passes with the
    conditional rate, as many rounds as fit the deadline (the model assumes
    two). A round is a candidate at the decode floor plus a check. Bills count
    model tokens; per-pass figures divide by the pass probability.
    """
    requires = ()
    produces = DeadlinePolicyResult

    def solve(self, decode_per_token: Quantity, prompt: int, output: int, diagnostic: int, breadth: int,
              max_rounds: int, deadline: Quantity, check_time: Quantity, initial_pass: float,
              revision_pass: float, input_price: Quantity, output_price: Quantity) -> DeadlinePolicyResult:
        t = decode_per_token.to("second")
        p_in, p_out = _per_token(input_price), _per_token(output_price)
        round_time = output * t + check_time
        rounds = min(max_rounds, math.floor(_ratio(deadline, round_time)))
        first_cost = _usd(prompt * p_in + output * p_out)
        revision_input = prompt + output + diagnostic
        revision_cost = _usd(revision_input * p_in + output * p_out)
        a_pass = initial_pass
        b_pass = calc_independent_success(initial_pass, breadth)
        c_pass = initial_pass + (1 - initial_pass) * revision_pass
        c_rounds = 1 + (1 - initial_pass)
        a_out, b_out, c_out = output, breadth * output, c_rounds * output
        a_cost, b_cost = first_cost, breadth * first_cost
        c_cost = first_cost + (1 - initial_pass) * revision_cost
        a_pp, b_pp = a_cost / a_pass, b_cost / b_pass
        return DeadlinePolicyResult(
            token_time=t,
            round_time=round_time.to("second"),
            rounds=rounds,
            requested_time=(max_rounds * round_time).to("second"),
            revision_input=revision_input,
            a_pass=a_pass, b_pass=b_pass, c_pass=c_pass, c_rounds=c_rounds,
            a_input=prompt, b_input=breadth * prompt, c_input=prompt + (1 - initial_pass) * revision_input,
            a_output=a_out, b_output=b_out, c_output=c_out,
            a_cost=a_cost, b_cost=b_cost, c_cost=c_cost,
            c_time=(c_rounds * round_time).to("second"),
            max_c_time=(rounds * round_time).to("second"),
            a_cost_per_pass=a_pp, b_cost_per_pass=b_pp, c_cost_per_pass=c_cost / c_pass,
            a_output_per_pass=a_out / a_pass, b_output_per_pass=b_out / b_pass,
            c_output_per_pass=c_out / c_pass,
            b_over_a=_ratio(b_pp, a_pp),
        )


