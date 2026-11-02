"""Agent-operations solvers: traces, retention, and staged release (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. Each model is arithmetic on
stated inputs: the elapsed time one trial's spans explain along its critical
path, and what shortening one span saves; the diagnostic trace bodies head
sampling and verdict-aware selection keep; and a Wald sequential test run as a
canary monitor, in labeled tasks, calendar time, and excess failures. None
models a particular tracing system, collector, or rollout controller.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..results import SequentialCanaryResult, SpanCriticalPathResult, TraceRetentionResult
from .base import ForwardModel
from ...core.types import Quantity
from ...physics.agents import calc_cleanup_critical_path
from ...physics.statistics import calc_sprt_boundaries, calc_sprt_expected_samples, calc_sprt_increments


def _ratio(q) -> float:
    return float(q.to("").magnitude) if hasattr(q, "to") else float(q)


class SpanCriticalPathModel(ForwardModel):
    """
    One trial's spans on a shared clock: what they explain, and what one shorter span saves.

    Each span has an observed interval and the spans it waits for. Summing
    every span counts overlapping work twice; the critical path
    (``calc_cleanup_critical_path``) counts the longest dependent chain once.
    The elapsed segment less that path is time no span explains. For an
    intervention, every span keeps its observed wait after its predecessors
    (the unexplained gap stays where it was) and takes its new duration.
    """
    requires = ()
    produces = SpanCriticalPathResult

    def solve(self, intervals: Mapping[str, Sequence[Quantity]], depends_on: Mapping[str, Sequence[str]],
              intervention: Mapping[str, Quantity]) -> SpanCriticalPathResult:
        start = {k: iv[0] for k, iv in intervals.items()}
        end = {k: iv[1] for k, iv in intervals.items()}
        dur = {k: end[k] - start[k] for k in intervals}
        t0 = min(start.values())
        elapsed = max(end.values()) - t0
        leaf_sum = sum(dur.values(), 0 * elapsed)
        path = calc_cleanup_critical_path(dur, depends_on)
        wait = {k: start[k] - max((end[d] for d in depends_on.get(k, ())), default=t0) for k in intervals}
        new_dur = {k: intervention.get(k, dur[k]) for k in intervals}
        finish, via = {}, {}

        def _finish(k):
            if k not in finish:
                preds = list(depends_on.get(k, ()))
                ready = max((_finish(d) for d in preds), default=0 * elapsed)
                via[k] = max(preds, key=lambda d: finish[d].to("s").magnitude) if preds else None
                finish[k] = ready + wait[k] + new_dur[k]
            return finish[k]

        changed = max(_finish(k) for k in intervals)
        last = max(finish, key=lambda k: finish[k].to("s").magnitude)
        chain = []
        while last is not None:
            chain.append(last)
            last = via[last]
        return SpanCriticalPathResult(
            leaf_sum=leaf_sum.to("s"), critical_path=path.to("s"), elapsed=elapsed.to("s"),
            unexplained=(elapsed - path).to("s"), changed_elapsed=changed.to("s"),
            saving=(elapsed - changed).to("s"), changed_critical=chain[::-1])


class TraceRetentionModel(ForwardModel):
    """
    Diagnostic trace bodies kept over a window under three selections.

    Keeping every body stores the whole cohort. Head sampling keeps each trial
    with one probability, decided before its outcome, so it keeps that share
    of the failures too. Verdict-aware (tail) selection keeps every failed and
    unresolved trial and a random control share of the ordinary ones; its
    volume is a floor set by the failure-plus-unresolved rate plus a term
    linear in the control share. Classes are disjoint and every body has the
    same size.
    """
    requires = ()
    produces = TraceRetentionResult

    def solve(self, failed: float, unresolved: float, ordinary: float, cohort_period: Quantity,
              trace_size: Quantity, window: Quantity, head_probability: float,
              control_probability: float, control_step: float) -> TraceRetentionResult:
        periods = _ratio(window / cohort_period)
        trials = failed + unresolved + ordinary
        head_count = trials * head_probability
        tail_control = ordinary * control_probability
        tail_count = failed + unresolved + tail_control

        def _vol(count):
            return (count * periods * trace_size).to("GB")

        full, tail = _vol(trials), _vol(tail_count)
        return TraceRetentionResult(
            head_count=head_count, head_failed=failed * head_probability,
            head_unresolved=unresolved * head_probability, tail_control=tail_control,
            tail_count=tail_count, full_volume=full, head_volume=_vol(head_count), tail_volume=tail,
            full_over_tail=_ratio(full / tail), floor_volume=_vol(failed + unresolved),
            per_point_volume=_vol(ordinary * control_step))


class SequentialCanaryModel(ForwardModel):
    """
    A Wald SPRT on task acceptance run as a canary monitor.

    The log-likelihood ratio favors the degraded acceptance ``p_degraded`` over
    ``p_base``: a failure adds ``ln((1 - p1)/(1 - p0))`` and a success
    ``ln(p1/p0)`` (the negated increments of ``calc_sprt_increments`` on the
    failure probabilities). Wald's boundaries come from the false-hold target
    ``alpha`` and the missed-degradation target ``beta``
    (``calc_sprt_boundaries``). Under the degraded hypothesis the walk ends at
    the upper boundary with probability ``1 - beta``; neglecting overshoot,
    ``calc_sprt_expected_samples`` gives the expected labeled tasks. Labels
    arrive ``label_delay`` after admission at ``fraction`` of the eligible
    ``arrival_rate``, so calendar time scales as ``1/fraction`` while the
    measured regression's excess failures change only through the tasks
    admitted while labels are pending.
    """
    requires = ()
    produces = SequentialCanaryResult

    def solve(self, p_base: float, p_degraded: float, alpha: float, beta: float, arrival_rate: Quantity,
              fraction: float, label_delay: Quantity) -> SequentialCanaryResult:
        success_inc, failure_inc = calc_sprt_increments(1 - p_degraded, 1 - p_base)
        success_step, failure_step = -success_inc, -failure_inc
        upper, lower = calc_sprt_boundaries(alpha, beta)
        drift = p_degraded * success_step + (1 - p_degraded) * failure_step
        terminal = (1 - beta) * upper + beta * lower
        samples = calc_sprt_expected_samples(upper, lower, drift, 1 - beta)
        rate = (arrival_rate * fraction).to("1/hour")
        admission = (samples / rate).to("hour")
        gap = p_base - p_degraded
        pending = _ratio(rate * label_delay)
        full_pending = _ratio(arrival_rate * label_delay)
        return SequentialCanaryResult(
            upper=upper, lower=lower, failure_step=failure_step, success_step=success_step,
            drift=drift, terminal=terminal, expected_samples=samples,
            failures_to_hold=math.ceil(upper / failure_step), candidate_rate=rate,
            admission_time=admission, decision_time=(admission + label_delay).to("hour"),
            pending=pending, excess_failures=(samples + pending) * gap, full_pending=full_pending,
            full_excess_failures=(samples + full_pending) * gap, exposure_share=fraction)
