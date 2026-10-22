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
    IntegrationAdaptersResult,
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
    calc_integration_adapters,
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

    def solve(self, raw_bytes: Quantity, bytes_per_token: Quantity, window: int, shaped_tokens: int,
              carry_calls: int, input_price: Quantity, cached_fraction: float) -> ObservationCarryResult:
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

    def solve(self, job_duration: Quantity, poll_interval: Quantity, context_tokens: int, output_tokens: int,
              cache_hit: float, input_price: Quantity, cached_fraction: float,
              output_price: Quantity) -> ModelPollingResult:
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

    def solve(self, registered: int, selected: int, descriptor_tokens: int,
              selection_overhead: int) -> ToolCatalogSelectionResult:
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

    def solve(self, tools: int, tool_tokens: int, shell_tokens: int, calls: int, input_price: Quantity,
              structured_attempt: Quantity, structured_acceptance: float, structured_change: float,
              shell_acceptance: float, shell_change: float, change_cost: Quantity) -> ToolkitGranularityResult:
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


class IntegrationAdaptersModel(ForwardModel):
    """Adapters connecting every host to every service, pairwise and through one shared protocol."""
    requires = ()
    produces = IntegrationAdaptersResult

    def solve(self, hosts: int, services: int) -> IntegrationAdaptersResult:
        pairwise, shared = calc_integration_adapters(hosts, services)
        return IntegrationAdaptersResult(pairwise=pairwise, shared=shared)
