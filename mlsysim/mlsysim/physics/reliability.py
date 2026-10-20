"""Reliability, checkpointing, and availability models."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import (
    validate_positive,
    validate_nonnegative,
    validate_at_least,
)

from ._units import _ensure_unit


def calc_young_daly_interval(checkpoint_cost_s, mtbf_s):
    """Optimal checkpoint interval: ``tau = sqrt(2 * delta * MTBF)``.

    Young (1974) first-order approximation, popularized with the Daly (2006)
    refinement; this implements the Young form, which balances time lost to
    checkpointing against expected rework after a failure. Valid when
    ``delta << MTBF`` (the usual regime; otherwise use Daly's higher-order
    form).

    Parameters
    ----------
    checkpoint_cost_s : Quantity or float
        Time to write one checkpoint (delta); bare floats are seconds.
    mtbf_s : Quantity or float
        Mean time between failures of the system; bare floats are seconds.

    Returns
    -------
    Quantity
        Optimal interval between checkpoints, in seconds.
    """
    delta = _ensure_unit(checkpoint_cost_s, ureg.second, "checkpoint_cost_s")
    mtbf = _ensure_unit(mtbf_s, ureg.second, "mtbf_s")
    validate_positive(delta, "checkpoint_cost_s")
    validate_positive(mtbf, "mtbf_s")
    if delta.m_as(ureg.second) >= 2 * mtbf.m_as(ureg.second):
        import warnings
        warnings.warn(
            "calc_young_daly_interval: checkpoint cost delta >= 2*MTBF — the "
            "Young first-order form is out of regime here (it returns tau < "
            "delta, which is physically impossible). Daly (2006) Sec. 5 "
            "prescribes tau = MTBF in this regime.",
            stacklevel=2,
        )
    seconds = math.sqrt(2 * delta.m_as(ureg.second) * mtbf.m_as(ureg.second))
    return seconds * ureg.second


def calc_mtbf_cluster(component_mtbf_hours, n_components, correlation_factor=1.0):
    """Cluster MTBF from identical independent components.

    ``MTBF_cluster = MTBF_component / N`` — exact when failures are
    independent and exponentially distributed (the minimum of N i.i.d.
    exponentials is exponential with N times the rate).

    ``correlation_factor`` scales the result: values **below 1 model
    correlated/clustered failures making the effective cluster MTBF worse**
    (0.5 halves it); 1.0 (default) is the independent case.
    """
    validate_at_least(n_components, 1, "n_components")
    mtbf = _ensure_unit(component_mtbf_hours, ureg.hour, "component_mtbf_hours")
    return (mtbf / n_components * correlation_factor).to(ureg.hour)


def calc_mtbf_node(
    gpu_mtbf_h,
    n_gpus,
    nic_mtbf_h,
    n_nics,
    psu_mtbf_h,
    n_psus,
    other_mtbf_h=None,
    n_other=0,
):
    """Compound node MTBF from heterogeneous components (series system).

    Assumes independent, exponentially distributed failures where any single
    component failure fails the node, so failure *rates* add:
    ``MTBF_node = 1 / (n_gpu/MTBF_gpu + n_nic/MTBF_nic + n_psu/MTBF_psu + ...)``.
    Redundant PSUs (N+M, M >= 1) do not fail the node singly; pass
    ``Node.series_psus`` (0 for a redundant supply) rather than the physical
    PSU count.

    Parameters
    ----------
    gpu_mtbf_h, nic_mtbf_h, psu_mtbf_h, other_mtbf_h : Quantity or float
        Per-component MTBF; bare floats are hours.
    n_gpus, n_nics, n_psus, n_other : int
        Component counts in the node.

    Returns
    -------
    Quantity
        Node MTBF in hours (always below the weakest component's MTBF).
    """
    gpu = _ensure_unit(gpu_mtbf_h, ureg.hour, "gpu_mtbf_h")
    nic = _ensure_unit(nic_mtbf_h, ureg.hour, "nic_mtbf_h")
    psu = _ensure_unit(psu_mtbf_h, ureg.hour, "psu_mtbf_h")
    rate = n_gpus / gpu + n_nics / nic + n_psus / psu
    if other_mtbf_h is not None and n_other > 0:
        rate += n_other / _ensure_unit(other_mtbf_h, ureg.hour, "other_mtbf_h")
    return (1.0 / rate).to(ureg.hour)


def calc_availability_stacked(single_availability, n_replicas):
    """System availability with k independent replicas."""
    return 1.0 - (1.0 - single_availability) ** n_replicas


def calc_failure_probability(mtbf, job_duration):
    """Probability of at least one failure during a job.

    Poisson/exponential failure model: ``P = 1 - exp(-T / MTBF)``. For short
    jobs (``T << MTBF``) this approaches ``T / MTBF``; as ``T`` grows it
    saturates at 1.

    Parameters
    ----------
    mtbf : Quantity or float
        Mean time between failures (> 0).
    job_duration : Quantity or float
        Job length. Both arguments must be Quantities (any time units) or
        both raw numbers in the *same* unit — mixing raises ``TypeError``
        because the implied unit would be ambiguous.

    Returns
    -------
    float
        Dimensionless probability in [0, 1).
    """
    validate_positive(mtbf, "mtbf")
    validate_nonnegative(job_duration, "job_duration")
    both_qty = isinstance(mtbf, ureg.Quantity) and isinstance(job_duration, ureg.Quantity)
    either_qty = isinstance(mtbf, ureg.Quantity) or isinstance(job_duration, ureg.Quantity)
    if either_qty and not both_qty:
        raise TypeError(
            "calc_failure_probability: both arguments must be Quantities or both raw numbers. "
            "Mixed types are ambiguous — attach units to the raw number first."
        )
    if both_qty:
        ratio = job_duration.to(ureg.second).magnitude / mtbf.to(ureg.second).magnitude
    else:
        ratio = job_duration / mtbf
    return 1.0 - math.exp(-ratio)


def calc_all_up_availability(availability, n_components):
    """Probability that all of N independent components are up at once: ``a**N``.

    A synchronous job that needs every one of N components is available only
    when all are; with independent failures that is the product of the
    per-component availabilities. Added 2026-10-20.
    """
    validate_at_least(n_components, 1, "n_components")
    return float(availability) ** n_components


def calc_downtime(availability, interval):
    """Expected time down in ``interval`` at a steady-state ``availability``."""
    return ((1 - float(availability)) * interval)


def calc_checkpoint_waste(write_time, interval, mtbf):
    """First-order share of wall time lost to checkpointing at ``interval``.

    ``W = write_time / interval + interval / (2 * mtbf)``: each save pauses
    training for ``write_time`` once per interval, and each failure (one per
    ``mtbf``) discards half an interval of work on average (Young 1974). At
    the Young-Daly interval the two terms are equal and ``W = sqrt(2 * write_time
    / mtbf)``. Added 2026-10-20.
    """
    delta = _ensure_unit(write_time, ureg.second, "write_time")
    tau = _ensure_unit(interval, ureg.second, "interval")
    mtbf_q = _ensure_unit(mtbf, ureg.second, "mtbf")
    validate_positive(tau, "interval")
    validate_positive(mtbf_q, "mtbf")
    return float((delta / tau + tau / (2 * mtbf_q)).to("").magnitude)


def calc_step_event_probability(rate_per_device_hour, n_devices, step_time):
    """Probability that at least one of ``n_devices`` meets an event during one step.

    Each device meets events at ``rate_per_device_hour`` (a probability per
    device-hour); one step exposes ``n_devices * step_time`` device-hours, so
    ``p = 1 - (1 - rate) ** exposure``. Used for silent-data-corruption
    exposure across a synchronous job. Added 2026-10-20.
    """
    step = _ensure_unit(step_time, ureg.second, "step_time")
    exposure = n_devices * step.m_as(ureg.hour)
    return 1 - (1 - float(rate_per_device_hour)) ** exposure


def calc_probe_detection_window(interval, timeout, failure_threshold):
    """Earliest and latest time from a crash to removal by a health probe.

    A replica leaves rotation after ``failure_threshold`` consecutive failed
    probes, each failing ``timeout`` after it starts. A crash just before a
    probe is removed after ``(threshold - 1) * interval + timeout``; a crash
    just after a successful probe waits one more interval. Added 2026-10-20.
    """
    period = _ensure_unit(interval, ureg.second, "interval")
    wait = _ensure_unit(timeout, ureg.second, "timeout")
    validate_at_least(failure_threshold, 1, "failure_threshold")
    earliest = (failure_threshold - 1) * period + wait
    return earliest.to(ureg.second), (earliest + period).to(ureg.second)


def calc_error_budget_spent(burn_rate, window, budget_period):
    """Share of an SLO's error budget spent by a sustained burn rate (added 2026-10-20).

    A burn rate of 1 spends exactly the budget over ``budget_period``; a burn
    rate ``b`` sustained for ``window`` spends ``b * window / budget_period``
    (Google SRE Workbook, "Alerting on SLOs").
    """
    return float(burn_rate * (window / budget_period).to("").magnitude)


# --- Durable execution: logs, checkpoints, and recovery data (added 2026-10-20) ---

def calc_group_commit_waits(batch_timer, flush_time, batch_size):
    """Storage work per record and the waits of the first and last arrivals in one group commit.

    One flush of ``flush_time`` acknowledges ``batch_size`` queued records
    (DeWitt et al. 1984). The work charged to each record is
    ``flush_time / batch_size``, but every record waits for the whole flush:
    the first arrival also waits out the ``batch_timer`` that gathers the
    batch, the last waits the flush alone. Returns
    ``(work_per_record, first_wait, last_wait)``.
    """
    timer = _ensure_unit(batch_timer, ureg.second, "batch_timer")
    flush = _ensure_unit(flush_time, ureg.second, "flush_time")
    validate_at_least(batch_size, 1, "batch_size")
    return (flush / batch_size).to(ureg.ms), (timer + flush).to(ureg.ms), flush.to(ureg.ms)


def calc_snapshot_write_volume(state_growth, stride, checkpoints):
    """Bytes written by full snapshots and by deltas over ``checkpoints`` checkpoints.

    State grows by ``state_growth`` per turn and a checkpoint is taken every
    ``stride`` turns. Checkpoint ``m`` of a full-snapshot policy rewrites all
    ``m * stride * state_growth`` accumulated so far, so the total is
    ``state_growth * stride * n (n + 1) / 2``; deltas write each turn's growth
    once, ``state_growth * stride * n``. Returns ``(full, delta)``.
    """
    growth = _ensure_unit(state_growth, ureg.byte, "state_growth")
    validate_at_least(stride, 1, "stride")
    validate_at_least(checkpoints, 1, "checkpoints")
    full = growth * stride * checkpoints * (checkpoints + 1) / 2
    delta = growth * stride * checkpoints
    return full.to(ureg.byte), delta.to(ureg.byte)


def calc_checkpoint_stride(pause, crash_rate, lost_turn_cost):
    """Checkpoint stride, in turns, that balances pauses against work lost to crashes.

    Young's (1974) balance counted in turns: a checkpoint every ``k`` turns
    costs ``pause / k`` per turn, and a crash (``crash_rate`` per turn) loses
    ``k / 2`` turns on average, each costing ``lost_turn_cost`` to recover.
    The per-turn overhead ``pause / k + crash_rate * lost_turn_cost * k / 2``
    is least at ``k* = sqrt(2 pause / (crash_rate * lost_turn_cost))``.
    """
    delta = _ensure_unit(pause, ureg.second, "pause")
    cost = _ensure_unit(lost_turn_cost, ureg.second, "lost_turn_cost")
    validate_positive(crash_rate, "crash_rate")
    validate_positive(cost, "lost_turn_cost")
    return math.sqrt(2 * delta.m_as(ureg.second) / (crash_rate * cost.m_as(ureg.second)))


def calc_stride_overhead(pause, stride, crash_rate, lost_turn_cost):
    """Expected overhead per turn at checkpoint ``stride``: ``pause / k + rate * cost * k / 2`` (seconds)."""
    delta = _ensure_unit(pause, ureg.second, "pause")
    cost = _ensure_unit(lost_turn_cost, ureg.second, "lost_turn_cost")
    validate_positive(stride, "stride")
    return (delta / stride + crash_rate * cost * stride / 2).to(ureg.second)
