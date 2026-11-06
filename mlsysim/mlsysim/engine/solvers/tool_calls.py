"""Tool-calling solvers: what a retry duplicates, an observation carries, a poll costs, a catalog stages.

Domain implementations behind ``mlsysim.solvers`` (added 2026-10-20). Each
model is arithmetic on stated probabilities, counts, and prices: duplicate
effects under one blind retry with independent request and reply loss, a raw
log against the shaped view later calls carry, the status calls a model makes
while a job is pending, descriptor tokens staged per call with and without
selection, cost per accepted task for structured tools against one shell
tool, and adapters with and without a shared protocol. None models delayed
packets, a tokenizer, cache eviction, or a provider's price schedule.
"""

from __future__ import annotations

from ..results import (
    BlindRetryResult,
    ModelPollingResult,
    ObservationCarryResult,
    ToolCatalogSelectionResult,
    ToolkitGranularityResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...core.units import USD as _USD, ureg as _ureg
from ...physics.agents import calc_cost_per_accepted
from ...physics.tool_calls import (
    calc_blind_retry_duplicate_probability,
    calc_pending_polls,
    calc_selection_break_even,
    calc_timeout_probability,
)

_TOKEN = _ureg.token


class BlindRetryModel(ForwardModel):
    """
    Timeouts and duplicate effects of ``calls`` mutating calls under one blind retry.

    Requests and replies are lost independently and never arrive later; every
    delivered request executes; each timeout triggers exactly one retry of a
    non-idempotent call with no deduplication.
    """

    requires = ()
    produces = BlindRetryResult

    def solve(self, request_loss: float, reply_loss: float, calls: int) -> BlindRetryResult:
        """Estimate timeouts and duplicate effects from one blind retry.

        Parameters
        ----------
        request_loss : float
            Probability a request is permanently lost, in [0, 1].
        reply_loss : float
            Probability an execution reply is permanently lost, in [0, 1].
        calls : int
            Nonnegative original mutating-call count.

        Returns
        -------
        BlindRetryResult
            Dimensionless per-call timeout and duplicate probabilities,
            original call count, and expected timeout and duplicate counts.

        Notes
        -----
        Equation: Timeout probability is request_loss + ``(1-request_loss)*reply_loss``.
        Duplicate execution probability is ``(1-request_loss)**2`` * reply_loss.

        Assumptions: Request and reply losses are independent, including across
        attempts. Every timeout triggers one retry; every delivered request executes.

        Excludes: Idempotency keys, deduplication, late packets, additional retries,
        and a prediction of workload-specific harm from duplicate execution.

        References
        ----------
        Derivation: A timeout means either a lost request or a delivered request
        with lost reply. Duplication requires the first request to execute, its
        reply to vanish, and the retry to execute; multiply those probabilities.

        Examples
        --------
        >>> from mlsysim.solvers import BlindRetryModel
        >>> result = BlindRetryModel().solve(0.1, 0.2, 100)
        >>> result.duplicate_probability < result.timeout_probability
        True
        """
        timeout = calc_timeout_probability(request_loss, reply_loss)
        duplicate = calc_blind_retry_duplicate_probability(request_loss, reply_loss)
        return BlindRetryResult(
            timeout_probability=timeout,
            duplicate_probability=duplicate,
            calls=calls,
            timeouts=calls * timeout,
            duplicates=calls * duplicate,
        )


class ObservationCarryModel(ForwardModel):
    """
    A raw tool output in windows, and the input bill of a shaped view carried by later calls.

    The raw byte count is converted at an assumed bytes-per-token ratio; the
    shaped observation is billed on each of ``carry_calls`` later requests, at
    the full input price or at ``cached_fraction`` of it.
    """

    requires = ()
    produces = ObservationCarryResult

    def solve(
        self,
        raw_bytes: Quantity,
        bytes_per_token: Quantity,
        window: int,
        shaped_tokens: int,
        carry_calls: int,
        input_price: Quantity,
        cached_fraction: float,
    ) -> ObservationCarryResult:
        """Price a shaped tool observation carried into later prompts.

        Parameters
        ----------
        raw_bytes : Quantity
            Nonnegative raw-output size, convertible to bytes.
        bytes_per_token : Quantity
            Positive assumed conversion ratio, convertible to bytes/token.
        window : int
            Positive context-window size in tokens.
        shaped_tokens : int
            Nonnegative size of the retained view, in tokens.
        carry_calls : int
            Nonnegative number of later calls carrying the view.
        input_price : Quantity
            Nonnegative input price, convertible to USD/token.
        cached_fraction : float
            Cached price divided by uncached price, in [0, 1].

        Returns
        -------
        ObservationCarryResult
            Raw and carried sizes in tokens, dimensionless window shares,
            and cached/uncached carry costs in USD.

        Notes
        -----
        Equation: Raw tokens equal raw_bytes/bytes_per_token. Carried tokens
        equal ``shaped_tokens*carry_calls``; uncached bill multiplies by input_price
        and cached bill multiplies that cost by cached_fraction.

        Assumptions: The conversion ratio and retained view stay fixed. The cached
        comparison assumes every carried token receives the cached price.

        Excludes: Tokenizer-specific expansion, shaping computation, retained
        information quality, eviction, and the first tool-call cost.

        References
        ----------
        Derivation: Convert bytes using the specified ratio, multiply retained
        tokens by their reuse count, and apply the supplied token price.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import ObservationCarryModel
        >>> result = ObservationCarryModel().solve(Q_(100, "byte"), Q_(4, "byte/token"), 100, 10, 3, Q_(1, "USD/token"), 0.5)
        >>> result.carried_cost_cached < result.carried_cost_uncached
        True
        """
        raw = (raw_bytes / bytes_per_token).to(_TOKEN)
        carried = shaped_tokens * carry_calls * _TOKEN
        uncached = (carried * input_price).to(_USD)
        return ObservationCarryResult(
            raw_tokens=raw,
            raw_windows=raw.to(_TOKEN).magnitude / window,
            shaped_share=shaped_tokens / window,
            carried_tokens=carried,
            carried_cost_uncached=uncached,
            carried_cost_cached=uncached * cached_fraction,
        )


class ModelPollingModel(ForwardModel):
    """
    Incremental cost of a model that checks a pending job every ``poll_interval``.

    Checks start one interval after dispatch and stop at completion; each reads
    ``context_tokens`` (a ``cache_hit`` share billed at ``cached_fraction`` of
    the input price) and generates ``output_tokens``. The worst detection delay
    is one interval. Useful calls common to every design are excluded.
    """

    requires = ()
    produces = ModelPollingResult

    def solve(
        self,
        job_duration: Quantity,
        poll_interval: Quantity,
        context_tokens: int,
        output_tokens: int,
        cache_hit: float,
        input_price: Quantity,
        cached_fraction: float,
        output_price: Quantity,
    ) -> ModelPollingResult:
        """Account for extra model calls while a dispatched job is pending.

        Parameters
        ----------
        job_duration : Quantity
            Positive job duration, convertible to seconds.
        poll_interval : Quantity
            Positive duration between checks, convertible to seconds.
        context_tokens : int
            Nonnegative input tokens read on each poll.
        output_tokens : int
            Nonnegative output tokens generated on each poll.
        cache_hit : float
            Share of input tokens receiving cached pricing, in [0, 1].
        input_price : Quantity
            Nonnegative uncached price, convertible to USD/token.
        cached_fraction : float
            Cached price as a fraction of the uncached price, in [0, 1].
        output_price : Quantity
            Nonnegative generated-token price, convertible to USD/token.

        Returns
        -------
        ModelPollingResult
            Poll and token counts, token prices in USD/token, costs in USD,
            and an interval-valued worst detection delay.

        Notes
        -----
        Equation: Pending polls equal ceil(job_duration/poll_interval)-1.
        Input price is ``input_price*((1-cache_hit)+cache_hit*cached_fraction)``;
        total cost sums poll input and output charges.

        Assumptions: Polling begins one interval after dispatch and counts only
        checks before completion. Context size, output size, and prices stay fixed.

        Excludes: The completion-handling call, useful calls common to alternatives,
        poll execution duration, variable job completion, and queueing.

        References
        ----------
        Derivation: Count positive interval multiples strictly below completion,
        then multiply by the token counts and weighted token prices.
        physics.tool_calls.calc_pending_polls implements the endpoint convention.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import ModelPollingModel
        >>> result = ModelPollingModel().solve(Q_(10, "s"), Q_(2, "s"), 10, 1, 0.5, Q_(1, "USD/token"), 0.5, Q_(1, "USD/token"))
        >>> result.cost == result.input_cost + result.output_cost
        True
        """
        polls = calc_pending_polls(job_duration, poll_interval)
        tin, tout = polls * context_tokens, polls * output_tokens
        cached_price = input_price * cached_fraction
        blended = (1 - cache_hit) * input_price + cache_hit * cached_price
        cin = (tin * _TOKEN * blended).to(_USD)
        cout = (tout * _TOKEN * output_price).to(_USD)
        return ModelPollingResult(
            polls=polls,
            input_tokens=tin,
            output_tokens=tout,
            cached_price=cached_price,
            blended_input_price=blended,
            input_cost=cin,
            output_cost=cout,
            cost=cin + cout,
            worst_delay=poll_interval,
        )


class ToolCatalogSelectionModel(ForwardModel):
    """
    Descriptor tokens on every call: the whole catalog against a selected subset plus overhead.

    Other request content is held fixed; selection runs without an extra model
    call. Selection pays once the descriptors it leaves out exceed its overhead.
    """

    requires = ()
    produces = ToolCatalogSelectionResult

    def solve(
        self, registered: int, selected: int, descriptor_tokens: int, selection_overhead: int
    ) -> ToolCatalogSelectionResult:
        """Compare staged descriptor tokens with and without tool selection.

        Parameters
        ----------
        registered : int
            Nonnegative total number of registered tools.
        selected : int
            Selected tool count, from zero through registered.
        descriptor_tokens : int
            Positive descriptor-token count per tool.
        selection_overhead : int
            Nonnegative additional request tokens introduced by selection.

        Returns
        -------
        ToolCatalogSelectionResult
            Full and selected token counts, released and unused tokens,
            and the unused-tool count at equal token cost; savings require a larger count.

        Notes
        -----
        Equation: Full tokens are ``registered*descriptor_tokens``; selected tokens
        are ``selected*descriptor_tokens`` + selection_overhead. Savings subtract the latter.

        Assumptions: Tools have equal descriptor length, selection adds no model
        call, and other request content remains fixed.

        Excludes: Selection computation, missed-tool quality loss, heterogeneous
        descriptors, and any prediction of serving latency.

        References
        ----------
        Derivation: Count descriptors in each layout and subtract. Selection saves
        tokens once omitted descriptor tokens exceed its added overhead;
        physics.tool_calls.calc_selection_break_even returns the unrounded equality threshold.

        Examples
        --------
        >>> from mlsysim.solvers import ToolCatalogSelectionModel
        >>> result = ToolCatalogSelectionModel().solve(10, 2, 20, 10)
        >>> result.selected_tokens < result.full_tokens
        True
        """
        full = registered * descriptor_tokens
        chosen = selected * descriptor_tokens
        staged = chosen + selection_overhead
        return ToolCatalogSelectionResult(
            full_tokens=full,
            selected_descriptor_tokens=chosen,
            selected_tokens=staged,
            released_tokens=full - staged,
            unused_descriptor_tokens=(registered - selected) * descriptor_tokens,
            break_even_unused=calc_selection_break_even(selection_overhead, descriptor_tokens),
        )


class ToolkitGranularityModel(ForwardModel):
    """
    Cost per accepted task for a structured toolkit against one general shell tool.

    The shell's attempt is the structured attempt minus the descriptor tokens
    it saves over ``calls``; each design then adds the expected loss of
    unconfined changes, ``p_change x change_cost``, to its attempt cost.
    """

    requires = ()
    produces = ToolkitGranularityResult

    def solve(
        self,
        tools: int,
        tool_tokens: int,
        shell_tokens: int,
        calls: int,
        input_price: Quantity,
        structured_attempt: Quantity,
        structured_acceptance: float,
        structured_change: float,
        shell_acceptance: float,
        shell_change: float,
        change_cost: Quantity,
    ) -> ToolkitGranularityResult:
        """Compare toolkit costs with acceptance and expected change losses.

        Parameters
        ----------
        tools : int
            Nonnegative structured-tool count.
        tool_tokens : int
            Nonnegative descriptor tokens per structured tool.
        shell_tokens : int
            Nonnegative descriptor tokens for the general shell tool.
        calls : int
            Nonnegative calls carrying descriptors per attempt.
        input_price : Quantity
            Nonnegative input-token price, convertible to USD/token.
        structured_attempt : Quantity
            Positive structured-design attempt cost in USD; shell descriptor savings must not exceed it.
        structured_acceptance : float
            Structured-design acceptance probability in (0, 1].
        structured_change : float
            Probability of a costly unconfined change under structured tools, in [0, 1].
        shell_acceptance : float
            Shell-design acceptance probability in (0, 1].
        shell_change : float
            Probability of a costly unconfined change under the shell, in [0, 1].
        change_cost : Quantity
            Positive USD loss conditional on an unconfined change.

        Returns
        -------
        ToolkitGranularityResult
            Descriptor token counts, USD costs with and without expected
            change loss, and a dimensionless break-even change probability.

        Notes
        -----
        Equation: Shell attempt cost subtracts ``(tools*tool_tokens-shell_tokens)``
        *``calls*input_price`` from structured_attempt. Each effective cost is
        (attempt_cost + ``change_probability*change_cost)/acceptance``.

        Assumptions: Descriptor price is the only baseline attempt-cost difference.
        Acceptance and change probabilities are supplied, constant rates; the same
        conditional loss applies to both designs.

        Excludes: Correlated retry losses, shell execution overhead, measured
        safety or acceptance improvements, and enforcement of probability bounds
        on the derived break-even value.

        References
        ----------
        Derivation: Expected loss equals event probability times its conditional
        cost. Add it to each attempt cost and divide by acceptance; equating the
        two effective costs and solving for shell change probability gives break-even.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import ToolkitGranularityModel
        >>> result = ToolkitGranularityModel().solve(3, 10, 10, 2, Q_(0.01, "USD/token"), Q_(2, "USD"), 0.8, 0.1, 0.8, 0.2, Q_(5, "USD"))
        >>> result.shell_attempt_cost < Q_(2, "USD")
        True
        """
        catalog = tools * tool_tokens
        saved_tokens = (catalog - shell_tokens) * calls
        saved = (saved_tokens * _TOKEN * input_price).to(_USD)
        attempt = structured_attempt.to(_USD)
        shell_attempt = attempt - saved
        loss = change_cost.to(_USD)
        structured_loss = calc_cost_per_accepted(attempt + structured_change * loss, structured_acceptance)
        shell_loss = calc_cost_per_accepted(shell_attempt + shell_change * loss, shell_acceptance)
        p_break = ((structured_loss * shell_acceptance - shell_attempt) / loss).to("").magnitude
        return ToolkitGranularityResult(
            structured_descriptor_tokens=catalog,
            saved_tokens=saved_tokens,
            saved_cost=saved,
            shell_attempt_cost=shell_attempt,
            structured_cost_per_accepted=calc_cost_per_accepted(attempt, structured_acceptance),
            shell_cost_per_accepted=calc_cost_per_accepted(shell_attempt, shell_acceptance),
            structured_cost_per_accepted_with_loss=structured_loss,
            shell_cost_per_accepted_with_loss=shell_loss,
            break_even_change_probability=p_break,
        )
