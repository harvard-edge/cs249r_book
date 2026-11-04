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
from fractions import Fraction
from typing import Optional, Sequence

from ..results import (
    AllocationAxesResult,
    ArgmaxNoiseResult,
    BestOfNPoolResult,
    DeadlinePolicyResult,
    ProbeVsAnswerResult,
    PrunedSearchTreeResult,
    ScopedRevisionResult,
    SearchAccountsResult,
    VerifiedPoolResult,
)
from ...core._validation import validate_at_least, validate_nonnegative, validate_positive, validate_range
from ...core.types import Quantity
from ...core.units import Q_, USD
from ...physics.agents import (
    calc_append_only_context,
    calc_expected_max_std_normal,
    calc_independent_success,
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


def _time_fraction_seconds(duration: Quantity) -> Fraction:
    """Interpret a time's decimal magnitude and unit scale without rounding its sum.

    Admission compares budgets, so binary rounding in ``0.1 * 3`` must not
    reject a declared ``0.3 s`` deadline. Using the original decimal magnitude
    preserves smaller declared budgets as smaller; no epsilon admits extra work.
    """
    seconds_per_unit = Q_(1, duration.units).to("second").magnitude
    return Fraction(str(duration.magnitude)) * Fraction(str(seconds_per_unit))


class ProbeVsAnswerModel(ForwardModel):
    """
    A generated explanation of ``K`` tokens against a prepared test of ``T_probe``.

    Generation time is ``K * tau_decode`` at the weight-transfer floor; the
    charge is ``K`` output tokens at the output price. Prefill, queueing, and
    test preparation are excluded.
    """

    requires = ()
    produces = ProbeVsAnswerResult

    def solve(
        self, decode_per_token: Quantity, answer_tokens: int, probe_time: Quantity, output_price: Quantity
    ) -> ProbeVsAnswerResult:
        """A generated explanation of ``K`` tokens against a prepared test of ``T_probe``.

        Parameters
        ----------
        decode_per_token : Quantity
            Duration of one generated token, convertible to seconds; positive.
        answer_tokens : int
            Positive generated explanation length, in tokens.
        probe_time : Quantity
            Positive duration of the prepared test, convertible to seconds.
        output_price : Quantity
            Positive output-token price, convertible to USD/token.

        Returns
        -------
        ProbeVsAnswerResult
            Answer and token duration in seconds, output charge in USD, and dimensionless answer/probe time ratio.

        Notes
        -----
        Equation: T = K tau; cost = K p_out; time ratio = T / T_probe.

        Assumptions: Fixed decode duration and token price; probe preparation already complete.

        Excludes: Prefill, queueing, test preparation, and answer quality.

        References
        ----------
        Derivation: Fixed-duration token generation adds one decode interval per
        output token. The corresponding output bill multiplies the same token
        count by the tariff; dividing duration by probe time compares the two
        operations without assigning either a quality advantage.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> ProbeVsAnswerModel().solve(Q_("1 ms"), 10, Q_("1 s"), Q_("1 USD/token")).answer_time.m_as("ms") == 10
        True
        """
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

    def solve(
        self,
        decode_per_token: Quantity,
        prompt: int,
        ceiling: int,
        candidates: int,
        candidate_tokens: int,
        rounds: int,
        report_tokens: int,
        check_time: Quantity,
        input_price: Quantity,
        output_price: Quantity,
        context_growth: Optional[int] = None,
    ) -> AllocationAxesResult:
        """One output ceiling spent three ways: depth, breadth, and feedback.

        Parameters
        ----------
        decode_per_token : Quantity
            Duration of one generated token, convertible to seconds; positive.
        prompt : int
            Initial prompt length, in tokens; nonnegative.
        ceiling : int
            Depth-policy output length, in tokens; positive.
        candidates : int
            Positive count of concurrently generated candidates.
        candidate_tokens : int
            Positive generated token count per candidate.
        rounds : int
            Positive count of feedback rounds.
        report_tokens : int
            Nonnegative tokens appended by each check report.
        check_time : Quantity
            Duration of one check, convertible to seconds; nonnegative.
        input_price : Quantity
            Positive input-token price, convertible to USD/token.
        output_price : Quantity
            Positive output-token price, convertible to USD/token.
        context_growth : Optional[int]
            Optional positive multiplier for the extra breadth prompt charge; None omits it.

        Returns
        -------
        AllocationAxesResult
            Per-policy input/output token counts, duration in seconds, USD bills, and dimensionless latency/cost ratios.

        Notes
        -----
        Equation: Depth output = ceiling; breadth output = N K; feedback input = H S + (K + R) H (H - 1) / 2.

        Assumptions: Breadth executes concurrently; feedback executes serially; each request pays its full prompt.

        Excludes: Prefill, queueing, resource contention, and effectiveness of reasoning.

        References
        ----------
        Derivation: Parallel candidates each pay for a separate prompt. Feedback
        round j resends j earlier increments, so repeated context is an arithmetic
        series. Adding the generated tokens and check durations gives the serial
        feedback time; breadth instead charges concurrent candidates separately.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> AllocationAxesModel().solve(Q_("1 ms"), 10, 20, 2, 10, 2, 2, Q_("1 ms"), Q_("1 USD/token"), Q_("1 USD/token")).breadth_output == 20
        True
        """
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
            depth_input=depth_in,
            depth_output=depth_out,
            depth_time=depth_time,
            depth_cost=depth_cost,
            breadth_input=breadth_in,
            breadth_output=breadth_out,
            breadth_time=breadth_time,
            breadth_cost=breadth_cost,
            feedback_input=feedback_in,
            feedback_output=feedback_out,
            feedback_time=feedback_time,
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

    def solve(
        self, pass_rate: float, pool_sizes: Sequence[int], easy_pass_rate: float, hard_pass_rate: float
    ) -> BestOfNPoolResult:
        """A parallel pool of ``N`` independent candidates against sequential retry.

        Parameters
        ----------
        pass_rate : float
            Independent candidate success probability, in (0, 1].
        pool_sizes : Sequence[int]
            Nonempty positive candidate counts; the final entry is the comparison pool.
        easy_pass_rate : float
            Success probability for the equally weighted easy-task population, in [0, 1].
        hard_pass_rate : float
            Success probability for the equally weighted hard-task population, in [0, 1].

        Returns
        -------
        BestOfNPoolResult
            Pass probabilities by pool size, candidate counts per success, and equally weighted mixture comparisons.

        Notes
        -----
        Equation: pass@n = 1 - (1 - p)^n; capped retry work = pass@n / p.

        Assumptions: Independent trials and a selector that recognizes any passing member; equal easy/hard mixture.

        Excludes: Correlated candidates, selector errors, and generation latency.

        References
        ----------
        Derivation: Complement the event that every independent candidate fails
        to obtain pool availability. Capped sequential work sums the geometric
        probabilities of reaching each candidate. Dividing work by availability
        charges failed attempts to successful tasks. For heterogeneous tasks,
        compute each task type's availability before averaging.

        Examples
        --------
        >>> BestOfNPoolModel().solve(0.5, [1, 2], 0.75, 0.25).pass_at[2] == 0.75
        True
        """
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

    def solve(
        self,
        pool_size: int,
        valid_rate: float,
        false_reject_rate: float,
        false_accept_rate: float,
        tokens_per_candidate: Quantity,
        output_price: Quantity,
        incident_loss: Quantity,
    ) -> VerifiedPoolResult:
        """First-passing selection from a fixed pool of ``n`` under a fallible check.

        Parameters
        ----------
        pool_size : int
            Positive number of generated candidates.
        valid_rate : float
            Prior probability that a candidate is valid, in [0, 1].
        false_reject_rate : float
            Probability of rejecting a valid candidate, in [0, 1].
        false_accept_rate : float
            Probability of passing an invalid candidate, in [0, 1].
        tokens_per_candidate : Quantity
            Positive token quantity per candidate, convertible to token.
        output_price : Quantity
            Positive output-token price, convertible to USD/token.
        incident_loss : Quantity
            Loss per defective selection, convertible to USD; nonnegative.

        Returns
        -------
        VerifiedPoolResult
            Selection/defect probabilities, expected passing counts, USD incident loss and output bill, and their ratio.

        Notes
        -----
        Equation: q = v (1 - false_reject_rate) + (1 - v) false_accept_rate; precision = v (1 - false_reject_rate) / q.

        Assumptions: Independent candidates, constant checker error rates, and q > 0; the entire pool is billed.

        Excludes: Checker runtime, correlated failures, and adaptive stopping charges.

        References
        ----------
        Derivation: Partition checker passes into valid-pass and invalid-pass
        joint events. Bayes conditioning gives their shares among passes. The
        complement of an all-rejected pool gives selection probability; multiply
        it by the invalid share and incident loss to price wrong selections.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> VerifiedPoolModel().solve(2, 0.5, 0.0, 0.0, Q_("10 token"), Q_("1 USD/token"), Q_("10 USD")).precision == 1.0
        True
        """
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
        """Argmax over ``N`` equally good candidates whose scores carry independent noise.

        Parameters
        ----------
        pool_sizes : Sequence[int]
            Nonempty positive candidate counts; the final entry is the comparison pool.
        large_pool : int
            Positive candidate count for the exploit-presence calculation.
        exploit_rate : float
            Independent per-candidate exploit probability, in [0, 1].

        Returns
        -------
        ArgmaxNoiseResult
            Expected winner noise in standard-deviation units and dimensionless exploit-presence probabilities.

        Notes
        -----
        Equation: Expected winner noise = E[max(Z_1,...,Z_n)]; exploit presence = 1 - (1 - g)^N.

        Assumptions: Candidates have equal underlying quality and independent standard-normal score noise.

        Excludes: Systematic scoring bias, dependent errors, and unequal candidate quality.

        References
        ----------
        Derivation: The maximum of n independent normal scores has cumulative
        probability Phi(x)^n, which the expected-maximum helper integrates. The
        loophole calculation instead complements the event that no candidate
        contains a loophole; it does not assume normal loophole probabilities.

        Examples
        --------
        >>> ArgmaxNoiseModel().solve([1], 2, 0.5).exploit_presence == 0.75
        True
        """
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
        """Output of regenerating a whole candidate plan against only its affected nodes.

        Parameters
        ----------
        total_nodes : int
            Positive count of equal-sized plan nodes.
        replaced_nodes : int
            Count of regenerated nodes, between zero and total_nodes.
        tokens_per_node : int
            Positive generated tokens per node.

        Returns
        -------
        ScopedRevisionResult
            Full and revised output token counts, with dimensionless regenerated and avoided shares.

        Notes
        -----
        Equation: Full output = total_nodes times tokens_per_node; scoped output = replaced_nodes times tokens_per_node.

        Assumptions: Nodes have equal generation length and unchanged nodes need no regeneration.

        Excludes: Input processing, checks, and physical effects of plan execution.

        References
        ----------
        Derivation: Multiplying equal tokens per node by the affected-node count
        gives the scoped output. Comparing that count with the full node count
        isolates generation savings without assuming a saving in input processing.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> ScopedRevisionModel().solve(4, 1, 10).output_share == 0.25
        True
        """
        full, scoped = total_nodes * tokens_per_node, replaced_nodes * tokens_per_node
        share = scoped / full
        return ScopedRevisionResult(full_output=full, scoped_output=scoped, output_share=share, avoided_share=1 - share)


class PrunedSearchTreeModel(ForwardModel):
    """
    Size of a search tree whose step check rejects each child with probability ``r``.

    The effective branching factor is ``b (1 - r)``; above ``r = 1 - 1/b`` the
    expected frontier shrinks and the tree dies out on its own. Rejecting a
    parent removes its whole subtree, so the node saving exceeds ``r``.
    """

    requires = ()
    produces = PrunedSearchTreeResult

    def solve(
        self,
        branching: int,
        rejection_rate: float,
        depth: int,
        wide_branching: Optional[int] = None,
        wide_depth: Optional[int] = None,
    ) -> PrunedSearchTreeResult:
        """Size of a search tree whose step check rejects each child with probability ``r``.

        Parameters
        ----------
        branching : int
            Positive children per expanded search node.
        rejection_rate : float
            Independent rejection probability per child, in [0, 1].
        depth : int
            Nonnegative count of tree expansion levels.
        wide_branching : Optional[int]
            Optional positive branching factor for an unpruned comparison tree.
        wide_depth : Optional[int]
            Optional positive depth for the comparison tree.

        Returns
        -------
        PrunedSearchTreeResult
            Expected frontier and node counts, critical rejection probability, and node-saving fraction.

        Notes
        -----
        Equation: Effective branching = b (1 - r); frontier at depth d = [b (1 - r)]^d.

        Assumptions: Independent child rejection and uniform branching; expected counts may be fractional.

        Excludes: Checker cost, mistaken rejection, and correctness of surviving nodes.

        References
        ----------
        Derivation: A parent contributes b children, each retained with
        probability 1-r, giving expected branching b(1-r). Summing expected level
        sizes yields explored nodes; comparing with the unpruned geometric series
        also counts descendants removed with each rejected parent.

        Examples
        --------
        >>> PrunedSearchTreeModel().solve(2, 0.5, 2).effective_branching == 1.0
        True
        """
        b_eff = branching * (1 - rejection_rate)
        unpruned, pruned = calc_pruned_tree_nodes(branching, rejection_rate, depth)
        return PrunedSearchTreeResult(
            effective_branching=b_eff,
            critical_rejection=1 - 1 / branching,
            unpruned_nodes=unpruned,
            pruned_nodes=pruned,
            unpruned_leaves=branching**depth,
            pruned_leaves=b_eff**depth,
            node_saving=1 - pruned / unpruned,
            flat_cut_nodes=unpruned * (1 - rejection_rate),
            wide_leaves=wide_branching**wide_depth if wide_branching and wide_depth else None,
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

    def solve(
        self,
        decode_per_token: Quantity,
        segments: int,
        segment_tokens: int,
        prompt_tokens: int,
        flat_candidates: int,
        beam_width: int,
        extensions: int,
        generation_slots: int,
        test_slots: int,
        score_time: Quantity,
        test_time: Quantity,
        input_price: Quantity,
        output_price: Quantity,
        scorer_price: Quantity,
    ) -> SearchAccountsResult:
        """Flat best-of-N against beam search, kept as separate accounts.

        Parameters
        ----------
        decode_per_token : Quantity
            Duration of one generated token, convertible to seconds; positive.
        segments : int
            Positive segment count in a complete candidate.
        segment_tokens : int
            Positive generated tokens per segment.
        prompt_tokens : int
            Nonnegative task-prompt tokens resent by each request.
        flat_candidates : int
            Positive complete-candidate count for flat sampling.
        beam_width : int
            Positive maximum retained prefixes at each search depth.
        extensions : int
            Positive proposed children per retained prefix.
        generation_slots : int
            Positive number of concurrent generation requests.
        test_slots : int
            Positive number of concurrent final tests.
        score_time : Quantity
            Duration to score each extension, convertible to seconds; nonnegative.
        test_time : Quantity
            Duration of each final test, convertible to seconds; nonnegative.
        input_price : Quantity
            Positive input-token price, convertible to USD/token.
        output_price : Quantity
            Positive output-token price, convertible to USD/token.
        scorer_price : Quantity
            Scorer token price, convertible to USD/token; nonnegative.

        Returns
        -------
        SearchAccountsResult
            Frontier counts, generated/scored token totals, elapsed seconds, USD charges, and the input-price break-even ratio.

        Notes
        -----
        Equation: Each beam frontier is retained times extensions; generation waves = ceil(frontier / slots); bills sum input, output, and scorer tokens.

        Assumptions: Fixed-length segments and fixed concurrency; scoring is serially additive and all prefixes are resent.

        Excludes: Prefix reuse, queueing, prefill, candidate quality, and contention; input-premium denominator must be nonzero.

        References
        ----------
        Derivation: Count complete outputs for flat sampling and retained-prefix
        extensions at each beam stage. Each extension resends its prompt and prefix,
        and the scorer reads that prefix again. Convert concurrent generation to
        waves before adding scorer and final-test durations. The input-price
        break-even equates extra prompt charges to net output savings.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> SearchAccountsModel().solve(Q_("1 ms"), 2, 10, 5, 2, 2, 2, 2, 2, Q_("1 ms"), Q_("1 ms"), Q_("1 USD/token"), Q_("1 USD/token"), Q_("1 USD/token")).beam_segments == 6
        True
        """
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
    conditional rate. The model admits at most two rounds, and only starts a
    round if it can finish by the deadline. A round is a candidate at the
    decode floor plus a check. Bills count
    model tokens; per-pass figures divide by the pass probability.
    """

    requires = ()
    produces = DeadlinePolicyResult

    def solve(
        self,
        decode_per_token: Quantity,
        prompt: int,
        output: int,
        diagnostic: int,
        breadth: int,
        max_rounds: int,
        deadline: Quantity,
        check_time: Quantity,
        initial_pass: float,
        revision_pass: float,
        input_price: Quantity,
        output_price: Quantity,
    ) -> DeadlinePolicyResult:
        """Three deliberation policies for one task under one deadline.

        Parameters
        ----------
        decode_per_token : Quantity
            Duration of one generated token, convertible to seconds; positive.
        prompt : int
            Initial prompt length, in tokens; nonnegative.
        output : int
            Positive generated tokens per candidate or revision.
        diagnostic : int
            Nonnegative tokens in the failure report.
        breadth : int
            Positive parallel candidate count.
        max_rounds : int
            Positive requested feedback-round cap. This model supports at most one
            revision, so the admitted count is additionally capped at two.
        deadline : Quantity
            Available duration, convertible to seconds; nonnegative.
        check_time : Quantity
            Duration of one check, convertible to seconds; nonnegative.
        initial_pass : float
            First-candidate success probability, in [0, 1].
        revision_pass : float
            Conditional success probability after first failure, in [0, 1].
        input_price : Quantity
            Nonnegative input-token price, convertible to USD/token.
        output_price : Quantity
            Nonnegative output-token price, convertible to USD/token.

        Returns
        -------
        DeadlinePolicyResult
            Policy success probabilities, expected input/output tokens, durations in seconds, USD bills, and cost per pass.

        Notes
        -----
        Equation: admitted rounds = min(2, cap, floor(deadline / round_time)).
        Feedback success is zero, p, or p + ``(1-p)*p_revision`` for zero, one,
        or two admitted rounds. Expected rounds are zero, one, or 1+(1-p).

        Assumptions: Parallel breadth and a single conditional revision. Requests
        are admitted only when one complete generation/check round fits.
        No partial-round work is launched or billed. A failed first attempt
        triggers a revision only when a second complete round fits. Admission
        interprets original time magnitudes and unit scales as decimal values;
        it adds them before conversion to floating-point output durations, so
        binary rounding cannot turn 3 * 100 ms into more than a 300 ms budget.

        Excludes: Queueing, prefill, and arbitrary multi-round feedback. Per-pass
        costs and output counts are infinite when success probability is zero;
        the B/A cost-per-pass ratio is NaN when its denominator is zero or
        either per-pass cost is infinite.

        References
        ----------
        Derivation: Divide the deadline by the fixed generation-plus-check duration
        to admit whole rounds, capped at the model's one revision. A second round is
        reached only after first-round failure, so its probability weights both
        success and billed tokens. Parallel breadth uses the complement of all
        independent candidates failing; unaffordable rounds contribute no work.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> DeadlinePolicyModel().solve(Q_("1 ms"), 10, 10, 2, 2, 2, Q_("1 s"), Q_("1 ms"), 0.5, 1.0, Q_("1 USD/token"), Q_("1 USD/token")).c_pass == 1.0
        True
        """
        for name, value, minimum in (
            ("prompt", prompt, 0), ("output", output, 1), ("diagnostic", diagnostic, 0),
            ("breadth", breadth, 1), ("max_rounds", max_rounds, 1),
        ):
            validate_at_least(value, minimum, name)
            if int(value) != value:
                raise ValueError(f"{name} must be an integer")
        validate_positive(decode_per_token, "decode_per_token")
        validate_nonnegative(check_time, "check_time")
        validate_nonnegative(deadline, "deadline")
        validate_range(initial_pass, 0, 1, "initial_pass")
        validate_range(revision_pass, 0, 1, "revision_pass")
        validate_nonnegative(input_price, "input_price")
        validate_nonnegative(output_price, "output_price")

        t = decode_per_token.to("second")
        p_in, p_out = _per_token(input_price), _per_token(output_price)
        round_seconds = output * _time_fraction_seconds(decode_per_token) + _time_fraction_seconds(check_time)
        deadline_seconds = _time_fraction_seconds(deadline)
        rounds = min(2, max_rounds, deadline_seconds // round_seconds)
        round_time = Q_(float(round_seconds), "second")
        first_cost = _usd(prompt * p_in + output * p_out)
        revision_input = prompt + output + diagnostic
        revision_cost = _usd(revision_input * p_in + output * p_out)
        attempted = rounds > 0
        revision_probability = (1 - initial_pass) if rounds == 2 else 0.0
        a_pass = initial_pass if attempted else 0.0
        b_pass = calc_independent_success(initial_pass, breadth) if attempted else 0.0
        c_pass = a_pass + revision_probability * revision_pass
        c_rounds = int(attempted) + revision_probability
        a_out = output if attempted else 0
        b_out, c_out = breadth * a_out, c_rounds * output
        a_cost = first_cost if attempted else _usd(0)
        b_cost = breadth * a_cost
        c_cost = a_cost + revision_probability * revision_cost
        a_pp = a_cost / a_pass if a_pass else _usd(math.inf)
        b_pp = b_cost / b_pass if b_pass else _usd(math.inf)
        finite_costs = math.isfinite(a_pp.m_as("USD")) and math.isfinite(b_pp.m_as("USD"))
        b_over_a = _ratio(b_pp, a_pp) if finite_costs and a_pp.m_as("USD") > 0 else math.nan
        return DeadlinePolicyResult(
            token_time=t,
            round_time=round_time.to("second"),
            rounds=rounds,
            requested_time=(max_rounds * round_time).to("second"),
            revision_input=revision_input,
            a_pass=a_pass,
            b_pass=b_pass,
            c_pass=c_pass,
            c_rounds=c_rounds,
            a_input=prompt if attempted else 0,
            b_input=breadth * prompt if attempted else 0,
            c_input=(prompt if attempted else 0) + revision_probability * revision_input,
            a_output=a_out,
            b_output=b_out,
            c_output=c_out,
            a_cost=a_cost,
            b_cost=b_cost,
            c_cost=c_cost,
            c_time=(c_rounds * round_time).to("second"),
            max_c_time=(rounds * round_time).to("second"),
            a_cost_per_pass=a_pp,
            b_cost_per_pass=b_pp,
            c_cost_per_pass=c_cost / c_pass if c_pass else _usd(math.inf),
            a_output_per_pass=a_out / a_pass if a_pass else math.inf,
            b_output_per_pass=b_out / b_pass if b_pass else math.inf,
            c_output_per_pass=c_out / c_pass if c_pass else math.inf,
            b_over_a=b_over_a,
        )
