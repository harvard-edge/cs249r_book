"""Failure-recovery solvers: what the runtime does once an effect has failed (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. Each model is arithmetic on
stated inputs: one bounded repair priced against unwinding and rebuilding to
the same endpoint, the expected irreversible loss of a plan whose checks run
before or after its pivot, an append-only loop run to its turn ceiling or
stopped early, retries multiplied through nested layers, a capped dependency
under one burst of blocking calls, a retry cohort synchronized or jittered,
a cleanup graph against a reverse stack, a breaker window classified two ways,
and the incidents a published table lists as sustained by retries. None models
a particular framework, provider, or queueing burst beyond what it states.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from ..results import (
    BreakerWindowResult,
    BulkheadAdmissionResult,
    CleanupGraphResult,
    IrreversibleLossResult,
    RepairBreakEvenResult,
    RetryAmplificationResult,
    RetryJitterResult,
    RetrySustainedOutagesResult,
    UndetectedLoopResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...core.units import Q_, USD
from ...physics.agents import (
    calc_append_only_context,
    calc_cleanup_critical_path,
    calc_expected_irreversible_loss,
    calc_repair_break_even,
    calc_repair_first_expected_cost,
)
from ...physics.tool_calls import (
    calc_bulkhead_admission,
    calc_jitter_expected_arrivals,
    calc_layered_retry_attempts,
)


def _tokens_usd(tokens: int, price: Quantity) -> Quantity:
    return (Q_(tokens, "token") * price).to(USD)


def _ratio(q) -> float:
    return float(q.to("").magnitude) if hasattr(q, "to") else float(q)


class RepairBreakEvenModel(ForwardModel):
    """
    One bounded repair attempt against compensating and rebuilding to the same endpoint.

    The fallback costs the compensation and rebuild times and the rebuild's
    model tokens; one repair costs its pre-output startup, its output at the
    decode rate, local validation, and its whole isolated request. If a failed
    attempt leaves the fallback unchanged, ``E[C] = C_repair + (1 - p) C_fallback``
    and repair first pays when ``p > p* = C_repair / C_fallback``, per axis.
    """
    requires = ()
    produces = RepairBreakEvenResult

    def solve(self, unwind_time: Quantity, rebuild_time: Quantity, rebuild_input: int,
              rebuild_output: int, repair_input: int, repair_output: int,
              startup_time: Quantity, output_rate: Quantity, validation_time: Quantity,
              input_price: Quantity, output_price: Quantity,
              assumed_success: float) -> RepairBreakEvenResult:
        back_t = (unwind_time + rebuild_time).to("second")
        back_usd = _tokens_usd(rebuild_input, input_price) + _tokens_usd(rebuild_output, output_price)
        rep_t = (startup_time + Q_(repair_output, "token") / output_rate + validation_time).to("second")
        rep_usd = _tokens_usd(repair_input, input_price) + _tokens_usd(repair_output, output_price)
        p_t = calc_repair_break_even(rep_t, back_t)
        p_usd = calc_repair_break_even(rep_usd, back_usd)
        exp_t = calc_repair_first_expected_cost(rep_t, back_t, assumed_success)
        exp_usd = calc_repair_first_expected_cost(rep_usd, back_usd, assumed_success)
        return RepairBreakEvenResult(
            fallback_time=back_t, fallback_cost=back_usd, repair_time=rep_t, repair_cost=rep_usd,
            break_even_time=p_t, break_even_cost=p_usd, expected_time=exp_t, expected_cost=exp_usd,
            time_saved_fraction=1 - _ratio(exp_t / back_t),
            cost_saved_fraction=1 - _ratio(exp_usd / back_usd),
            one_in=math.floor(1 / max(p_t, p_usd)),
        )


class IrreversibleLossModel(ForwardModel):
    """
    Expected irreversible loss of a plan with its check after, or before, the pivot.

    Without a check ahead of the pivot nothing recovers: ``L = P(bad) I``. A
    check that catches a fraction ``v`` and reverts with fidelity ``rho`` leaves
    ``P(bad) v I (1 - rho)`` from caught changes and ``P(bad) (1 - v) I`` from
    missed ones, which sum to ``P(bad) I (1 - v rho)``. Raising ``v`` by one step
    ``dv`` removes ``P(bad) I rho dv``.
    """
    requires = ()
    produces = IrreversibleLossResult

    def solve(self, defect_probability: float, consequence: Quantity, catch_rate: float,
              revert_fidelity: float, attempt_cost: Quantity,
              catch_step: float) -> IrreversibleLossResult:
        p, impact = defect_probability, consequence.to(USD)
        first = calc_expected_irreversible_loss(p, impact, 0)

        def split(v):
            caught = p * v * impact * (1 - revert_fidelity)
            missed = calc_expected_irreversible_loss(p * (1 - v), impact, 0)
            return caught, missed, caught + missed

        caught, missed, total = split(catch_rate)
        raised = catch_rate + catch_step
        caught_r, missed_r, total_r = split(raised)
        per_step = p * impact * revert_fidelity * catch_step
        attempt = attempt_cost.to(USD)
        return IrreversibleLossResult(
            loss_without_check=first, loss_caught=caught, loss_missed=missed, loss_with_check=total,
            reduction=_ratio(first / total), missed_share=_ratio(missed / total),
            loss_per_catch_step=per_step, raised_catch_rate=raised,
            loss_caught_raised=caught_r, loss_missed_raised=missed_r, loss_with_check_raised=total_r,
            attempt_cost=attempt, loss_vs_attempt=_ratio(total / attempt),
            step_vs_attempt=_ratio(per_step / attempt),
        )


class UndetectedLoopModel(ForwardModel):
    """
    An append-only loop run to its turn ceiling, or stopped after an earlier turn.

    Each turn appends its output and an observation and resends the whole
    history, so ``h`` turns send ``h s0 + delta h (h - 1) / 2`` input tokens
    (``calc_append_only_context``) and ``h k_out`` output tokens; every turn
    takes the same time. The avoided charge is gross: detector cost, false
    alarms, and cleanup are not debited.
    """
    requires = ()
    produces = UndetectedLoopResult

    def solve(self, first_input: int, output_per_turn: int, observation_per_turn: int,
              turn_time: Quantity, ceiling: int, stop_after: int, context_window: int,
              input_price: Quantity, output_price: Quantity) -> UndetectedLoopResult:
        grow = output_per_turn + observation_per_turn

        def sent(h):
            return calc_append_only_context(first_input, grow, h)[1]

        def charge(h):
            return _tokens_usd(sent(h), input_price) + _tokens_usd(h * output_per_turn, output_price)

        last, ceiling_in = calc_append_only_context(first_input, grow, ceiling)
        peak = last + output_per_turn
        half = ceiling // 2
        c_cap, c_stop = charge(ceiling), charge(stop_after)
        return UndetectedLoopResult(
            appended_per_turn=grow, ceiling_input=ceiling_in, stop_input=sent(stop_after),
            ceiling_cost=c_cap, stop_cost=c_stop,
            ceiling_time=(ceiling * turn_time).to("second"), stop_time=(stop_after * turn_time).to("second"),
            cost_ratio=_ratio(c_cap / c_stop), avoided_cost=c_cap - c_stop,
            peak_request=peak, fits_context=peak <= context_window,
            half_turns=half, half_ratio=ceiling_in / sent(half),
            input_by_turn=[sent(h) for h in range(ceiling + 1)],
        )


class RetryAmplificationModel(ForwardModel):
    """
    Attempts one failing logical operation sends when each layer retries on its own allowance.

    Every attempt of a layer drives a full set of attempts in the layer below,
    so ``a`` attempts per layer over ``L`` layers reach the dependency ``a^L``
    times; one retry owner with the same allowance sends ``a`` (Brooker 2019).
    """
    requires = ()
    produces = RetryAmplificationResult

    def solve(self, attempts_per_layer: int, layers: int) -> RetryAmplificationResult:
        return RetryAmplificationResult(
            by_layer=[calc_layered_retry_attempts(attempts_per_layer, k) for k in range(layers + 1)],
            layered=calc_layered_retry_attempts(attempts_per_layer, layers),
            coordinated=attempts_per_layer,
        )


class BulkheadAdmissionModel(ForwardModel):
    """
    One burst of blocking calls to a stalled dependency, before any completes.

    Uncapped, every call takes a worker until the pool is full. Capped, at most
    ``worker_cap`` occupy workers and ``queue_cap`` wait without one; the rest
    are rejected at admission, and the cap protects the remaining workers
    however long the stall lasts. Worker slots only: not CPU, storage, or
    remote jobs that outlive a timeout.
    """
    requires = ()
    produces = BulkheadAdmissionResult

    def solve(self, arrivals: int, workers: int, worker_cap: int, queue_cap: int) -> BulkheadAdmissionResult:
        u_occ, _, _, u_free = calc_bulkhead_admission(arrivals, workers, workers, 0)
        occ, queued, rejected, free = calc_bulkhead_admission(arrivals, workers, worker_cap, queue_cap)
        return BulkheadAdmissionResult(uncapped_occupied=u_occ, uncapped_free=u_free, occupied=occ,
                                       queued=queued, rejected=rejected, protected_workers=free)


class RetryJitterModel(ForwardModel):
    """
    A cohort of retries released together: all inside a burst window, or jittered.

    Synchronized, the window average is ``N / w``; each drawing its arrival
    uniformly over the span ``J``, the expected rate is ``N / J`` and a window
    of length ``w`` expects ``N w / J`` arrivals. Expectations only: one window
    can still receive every retry, so only an admission rate limiter bounds the peak.
    """
    requires = ()
    produces = RetryJitterResult

    def solve(self, retries: int, burst_window: Quantity, jitter_span: Quantity) -> RetryJitterResult:
        sync = (retries / burst_window).to("1/second")
        mean = (retries / jitter_span).to("1/second")
        return RetryJitterResult(synchronized_rate=sync, jittered_mean_rate=mean,
                                 spread_ratio=_ratio(sync / mean),
                                 expected_in_window=calc_jitter_expected_arrivals(retries, burst_window, jitter_span))


class CleanupGraphModel(ForwardModel):
    """
    Registered cleanup run as a reverse stack, one at a time, against a dependency graph.

    The stack takes the sum of the durations; the graph, with unlimited
    parallelism, the longest path (``calc_cleanup_critical_path``). When one
    operation's outcome stays unresolved, only its transitive dependents are
    blocked; every other operation can still settle.
    """
    requires = ()
    produces = CleanupGraphResult

    def solve(self, durations: Mapping[str, Quantity], depends_on: Mapping[str, Sequence[str]],
              unresolved: str) -> CleanupGraphResult:
        stack = sum(durations.values(), 0 * next(iter(durations.values())))
        graph = calc_cleanup_critical_path(durations, depends_on)
        blocked, frontier = [], {unresolved}
        while frontier:
            nxt = {op for op, deps in depends_on.items() if set(deps) & frontier and op not in blocked}
            blocked += sorted(nxt)
            frontier = nxt
        settleable = [op for op in durations if op != unresolved and op not in blocked]
        return CleanupGraphResult(stack_time=stack.to("second"), graph_time=graph.to("second"),
                                  saved_time=(stack - graph).to("second"),
                                  blocked=[op for op in durations if op in blocked], settleable=settleable)


class BreakerWindowModel(ForwardModel):
    """
    One circuit-breaker window: failure fraction under a classification rule, and a naive one.

    The classified rule counts only outcomes that show the dependency cannot
    do work (``failure_classes``); answers from a working service count as
    observations, not failures. The naive rule counts every outcome other than
    success. The breaker trips at ``f >= threshold`` with at least ``min_outcomes``.
    """
    requires = ()
    produces = BreakerWindowResult

    def solve(self, outcomes: Mapping[str, int], failure_classes: Iterable[str], success_class: str,
              threshold: float, min_outcomes: int) -> BreakerWindowResult:
        n = sum(outcomes.values())
        classified = sum(outcomes[c] for c in failure_classes)
        naive = n - outcomes[success_class]
        f_c, f_n = classified / n, naive / n
        enough = n >= min_outcomes
        return BreakerWindowResult(outcomes=n, classified_failures=classified, classified_fraction=f_c,
                                   classified_trips=enough and f_c >= threshold, naive_failures=naive,
                                   naive_fraction=f_n, naive_trips=enough and f_n >= threshold)


class RetrySustainedOutagesModel(ForwardModel):
    """
    Incidents a published table lists with retry as a sustaining effect.

    Counts the table's rows and those whose sustaining effects include retries,
    and lists the retry-sustained durations, longest first; rows with no
    reported duration are listed separately.
    """
    requires = ()
    produces = RetrySustainedOutagesResult

    def solve(self, rows: Sequence[Mapping], id_column: str, flag_column: str, duration_column: str,
              threshold_hours: float) -> RetrySustainedOutagesResult:
        hit = [r for r in rows if r[flag_column]]
        timed = sorted(((r[id_column], r[duration_column]) for r in hit if r[duration_column] is not None),
                       key=lambda x: -x[1])
        return RetrySustainedOutagesResult(
            incidents=len(rows), retry_sustained=len(hit), share=len(hit) / len(rows),
            with_duration=timed,
            without_duration=[r[id_column] for r in hit if r[duration_column] is None],
            over_hours=sum(1 for _, h in timed if h > threshold_hours),
        )
