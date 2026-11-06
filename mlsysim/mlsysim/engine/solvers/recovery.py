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

from ..results import (
    BulkheadAdmissionResult,
    IrreversibleLossResult,
    RepairBreakEvenResult,
    RetryAmplificationResult,
    RetryJitterResult,
    UndetectedLoopResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...core.units import Q_, USD
from ...physics.agents import (
    calc_append_only_context,
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

    def solve(
        self,
        unwind_time: Quantity,
        rebuild_time: Quantity,
        rebuild_input: int,
        rebuild_output: int,
        repair_input: int,
        repair_output: int,
        startup_time: Quantity,
        output_rate: Quantity,
        validation_time: Quantity,
        input_price: Quantity,
        output_price: Quantity,
        assumed_success: float,
    ) -> RepairBreakEvenResult:
        """Price one repair attempt against rebuilding.

        Parameters
        ----------
        unwind_time : Quantity
            Nonnegative compensation time.
        rebuild_time : Quantity
            Positive rebuild time.
        rebuild_input : int
            Nonnegative rebuild input-token count.
        rebuild_output : int
            Nonnegative rebuild output-token count.
        repair_input : int
            Nonnegative repair input-token count.
        repair_output : int
            Nonnegative repair output-token count.
        startup_time : Quantity
            Nonnegative startup time.
        output_rate : Quantity
            Positive output-token generation rate, tokens per second.
        validation_time : Quantity
            Nonnegative local-validation time.
        input_price : Quantity
            Positive input price, USD per token.
        output_price : Quantity
            Positive output price, USD per token.
        assumed_success : float
            Assumed repair success probability in [0, 1].

        Returns
        -------
        RepairBreakEvenResult
            Repair and fallback costs in USD and seconds, break-even success
            probabilities per axis, expected costs, and fractional savings.

        Notes
        -----
        Equation: E[C]=C_repair+(1-p)C_fallback; p_star=C_repair/C_fallback.

        Assumptions: A failed repair leaves the same fallback available; costs are additive.

        Excludes: Success-rate estimation, retries after repair, and correlated failures.

        References
        ----------
        Derivation: a repair charge is paid on every trial, whereas fallback
        is paid only on the failure branch with probability 1 - assumed_success.
        Comparing this expected charge with unconditional fallback yields the
        repair-to-fallback cost ratio as the success threshold, independently
        for time and money.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = RepairBreakEvenModel().solve(Q_(1,"s"), Q_(10,"s"), 100, 10, 10, 2, Q_(1,"s"), Q_(2,"token/s"), Q_(1,"s"), Q_(1,"USD/token"), Q_(1,"USD/token"), 0.9)
        >>> r.repair_time < r.fallback_time
        True
        """
        back_t = (unwind_time + rebuild_time).to("second")
        back_usd = _tokens_usd(rebuild_input, input_price) + _tokens_usd(rebuild_output, output_price)
        rep_t = (startup_time + Q_(repair_output, "token") / output_rate + validation_time).to("second")
        rep_usd = _tokens_usd(repair_input, input_price) + _tokens_usd(repair_output, output_price)
        p_t = calc_repair_break_even(rep_t, back_t)
        p_usd = calc_repair_break_even(rep_usd, back_usd)
        exp_t = calc_repair_first_expected_cost(rep_t, back_t, assumed_success)
        exp_usd = calc_repair_first_expected_cost(rep_usd, back_usd, assumed_success)
        return RepairBreakEvenResult(
            fallback_time=back_t,
            fallback_cost=back_usd,
            repair_time=rep_t,
            repair_cost=rep_usd,
            break_even_time=p_t,
            break_even_cost=p_usd,
            expected_time=exp_t,
            expected_cost=exp_usd,
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

    def solve(
        self,
        defect_probability: float,
        consequence: Quantity,
        catch_rate: float,
        revert_fidelity: float,
        attempt_cost: Quantity,
        catch_step: float,
    ) -> IrreversibleLossResult:
        """Split expected loss into caught and missed defects.

        Parameters
        ----------
        defect_probability : float
            Defect probability in [0, 1].
        consequence : Quantity
            Positive consequence per defect, USD.
        catch_rate : float
            Check catch probability in [0, 1].
        revert_fidelity : float
            Fraction recovered after a caught defect, in [0, 1].
        attempt_cost : Quantity
            Positive per-attempt cost, USD.
        catch_step : float
            Nonnegative catch-rate increment; catch_rate + catch_step <= 1.

        Returns
        -------
        IrreversibleLossResult
            Caught and missed expected losses in USD, recovery benefit ratios,
            and marginal loss reduction for the supplied catch-rate increment.

        Notes
        -----
        Equation: L=p I(1-v rho); delta_L=p I rho delta_v.

        Assumptions: Consequence and recovery fidelity are fixed conditional on a defect.

        Excludes: Estimated detection accuracy, correlated harms, and risk certification.

        References
        ----------
        Derivation: partition defective attempts into caught and missed events.
        Caught defects retain fraction 1 - revert_fidelity of their consequence;
        missed defects retain all of it. Summing the two mutually exclusive
        contributions gives defect_probability * consequence *
        (1 - catch_rate * revert_fidelity).

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = IrreversibleLossModel().solve(0.1, Q_(100,"USD"), 0.5, 0.5, Q_(1,"USD"), 0.1)
        >>> r.loss_with_check < r.loss_without_check
        True
        """
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
            loss_without_check=first,
            loss_caught=caught,
            loss_missed=missed,
            loss_with_check=total,
            reduction=_ratio(first / total),
            missed_share=_ratio(missed / total),
            loss_per_catch_step=per_step,
            raised_catch_rate=raised,
            loss_caught_raised=caught_r,
            loss_missed_raised=missed_r,
            loss_with_check_raised=total_r,
            attempt_cost=attempt,
            loss_vs_attempt=_ratio(total / attempt),
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

    def solve(
        self,
        first_input: int,
        output_per_turn: int,
        observation_per_turn: int,
        turn_time: Quantity,
        ceiling: int,
        stop_after: int,
        context_window: int,
        input_price: Quantity,
        output_price: Quantity,
    ) -> UndetectedLoopResult:
        """Account for an append-only loop stopped early.

        Parameters
        ----------
        first_input : int
            Positive initial prompt-token count.
        output_per_turn : int
            Nonnegative output tokens per turn.
        observation_per_turn : int
            Nonnegative observation tokens per turn.
        turn_time : Quantity
            Positive fixed duration per turn.
        ceiling : int
            Turn ceiling, integer at least two.
        stop_after : int
            Positive stop turn no larger than the ceiling.
        context_window : int
            Positive context capacity in tokens.
        input_price : Quantity
            Positive input price, USD per token.
        output_price : Quantity
            Positive output price, USD per token.

        Returns
        -------
        UndetectedLoopResult
            Input/output token accounting, costs in USD, durations in seconds,
            context-fit flag, and avoided gross spend from early stopping.

        Notes
        -----
        Equation: I(h)=h s0+delta h(h-1)/2; C(h)=I(h)p_in+h k_out p_out.

        Assumptions: Each turn appends the same token counts and resends all history.

        Excludes: Detector overhead, false alarms, cleanup, and prefix-cache discounts.

        References
        ----------
        Derivation: prompt lengths form an arithmetic progression beginning
        with first_input and increasing by output_per_turn + observation_per_turn.
        Summing that progression prices all resent input; multiplying the fixed
        output count and fixed duration by turns gives output spend and elapsed
        time. The stopped trajectory truncates the same progression.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = UndetectedLoopModel().solve(10, 2, 3, Q_(1,"s"), 4, 2, 100, Q_(1,"USD/token"), Q_(1,"USD/token"))
        >>> r.stop_cost < r.ceiling_cost
        True
        """
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
            appended_per_turn=grow,
            ceiling_input=ceiling_in,
            stop_input=sent(stop_after),
            ceiling_cost=c_cap,
            stop_cost=c_stop,
            ceiling_time=(ceiling * turn_time).to("second"),
            stop_time=(stop_after * turn_time).to("second"),
            cost_ratio=_ratio(c_cap / c_stop),
            avoided_cost=c_cap - c_stop,
            peak_request=peak,
            fits_context=peak <= context_window,
            half_turns=half,
            half_ratio=ceiling_in / sent(half),
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
        """Count attempts amplified through nested retry layers.

        Parameters
        ----------
        attempts_per_layer : int
            Positive integer attempt allowance per layer.
        layers : int
            Nonnegative number of nested layers.

        Returns
        -------
        RetryAmplificationResult
            Attempt counts by depth, total layered attempts, and coordinated
            attempts; these are counts rather than latency predictions.

        Notes
        -----
        Equation: A_layered=``a**L``; A_coordinated=a.

        Assumptions: Every attempt fails and consumes the full downstream allowance.

        Excludes: Early success, backoff timing, and overlapping requests.

        References
        ----------
        Derivation: every parent attempt invokes attempts_per_layer child
        attempts. Repeating this branching for layers levels multiplies the
        allowance once per level. A single retry owner invokes the dependency
        directly and therefore spends only one allowance.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = RetryAmplificationModel().solve(2, 3)
        >>> r.layered > r.coordinated
        True
        """
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
        """Partition a burst among occupied, queued, and rejected calls.

        Parameters
        ----------
        arrivals : int
            Nonnegative burst arrival count.
        workers : int
            Positive total worker count.
        worker_cap : int
            Dependency worker cap between zero and workers.
        queue_cap : int
            Nonnegative waiting-queue capacity.

        Returns
        -------
        BulkheadAdmissionResult
            Occupied, queued, rejected, and protected-worker counts for the
            burst, together with the uncapped occupancy comparison.

        Notes
        -----
        Equation: occupied=min(arrivals,cap); queued=min(arrivals-occupied,queue_cap).

        Assumptions: All arrivals precede completion and queued calls hold no worker.

        Excludes: CPU load, remote jobs surviving timeouts, and service-time distributions.

        References
        ----------
        Derivation: assign burst calls first to dependency worker slots, then
        to waiting slots, then reject the remainder. Subtract occupied workers
        from the whole pool to count protected workers. The comparison repeats
        this partition with the entire pool available and no waiting queue.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = BulkheadAdmissionModel().solve(10, 8, 3, 2)
        >>> r.occupied + r.queued + r.rejected == 10
        True
        """
        u_occ, _, _, u_free = calc_bulkhead_admission(arrivals, workers, workers, 0)
        occ, queued, rejected, free = calc_bulkhead_admission(arrivals, workers, worker_cap, queue_cap)
        return BulkheadAdmissionResult(
            uncapped_occupied=u_occ,
            uncapped_free=u_free,
            occupied=occ,
            queued=queued,
            rejected=rejected,
            protected_workers=free,
        )


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
        """Compare synchronized and uniformly jittered retry rates.

        Parameters
        ----------
        retries : int
            Nonnegative retry count.
        burst_window : Quantity
            Positive observation-window duration.
        jitter_span : Quantity
            Positive uniform-jitter span, at least the window duration.

        Returns
        -------
        RetryJitterResult
            Synchronized and jittered mean rates in inverse seconds, their
            ratio, and the expected count in the observation window.

        Notes
        -----
        Equation: rate_sync=N/w; rate_jitter=N/J; E[arrivals_in_window]=N w/J.

        Assumptions: Each retry chooses an independent uniform arrival within the jitter span.

        Excludes: A deterministic peak-rate guarantee or admission rate limiting.

        References
        ----------
        Derivation: a uniform arrival has probability burst_window / jitter_span
        of falling in the observation window. Linearity of expectation multiplies
        that probability by retries. Dividing the cohort by its release interval
        gives each mean rate; this calculation does not bound a realized burst.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = RetryJitterModel().solve(10, Q_(1,"s"), Q_(10,"s"))
        >>> r.jittered_mean_rate < r.synchronized_rate
        True
        """
        sync = (retries / burst_window).to("1/second")
        mean = (retries / jitter_span).to("1/second")
        return RetryJitterResult(
            synchronized_rate=sync,
            jittered_mean_rate=mean,
            spread_ratio=_ratio(sync / mean),
            expected_in_window=calc_jitter_expected_arrivals(retries, burst_window, jitter_span),
        )
