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
    """Estimate the first-order checkpoint interval that minimizes lost work.

    Parameters
    ----------
    checkpoint_cost_s : Quantity or float
        Positive write time delta; bare numbers are seconds.
    mtbf_s : Quantity or float
        Positive system mean time between failures; bare numbers are seconds.

    Returns
    -------
    Quantity
        Interval in seconds; not rounded to a scheduling quantum.

    Notes
    -----
    Equation: tau = sqrt(2 * delta * MTBF), from minimizing delta/tau + ``tau/(2*MTBF)``.

    Assumptions: Failures arrive independently and rework averages half an interval;
    delta is small relative to MTBF.

    Excludes: Restart time, correlated failures, asynchronous checkpointing, and
    higher-order terms.

    This implements the Young form, despite the historical function name.
    When delta >= ``2*MTBF`` it warns but still returns the first-order result;
    the caller must not treat that out-of-regime value as an achievable schedule.

    References
    ----------
    Young (1974), "A first order approximation to the optimum checkpoint interval."
    https://doi.org/10.1145/361147.361115 (bibliography key: young1974).
    Daly (2006), "A higher order estimate of the optimum checkpoint interval for restart
    dumps."
    https://doi.org/10.1016/j.future.2004.11.016 (bibliography key: daly2006).

    Examples
    --------
    >>> baseline = calc_young_daly_interval(10, 10000)
    >>> longer = calc_young_daly_interval(40, 10000)
    >>> math.isclose(longer.m_as("s"), 2 * baseline.m_as("s"))
    True
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
    """Combine component MTBFs for a homogeneous series system.

    Parameters
    ----------
    component_mtbf_hours : Quantity or float
        Positive per-component mean time between failures; bare numbers are hours.
    n_components : int
        Number of failure-critical components, at least one.
    correlation_factor : float, optional
        Positive dimensionless multiplier, default one; below one reduces effective
        MTBF.

    Returns
    -------
    Quantity
        Effective cluster MTBF in hours.

    Notes
    -----
    Equation: MTBF_cluster = correlation_factor * MTBF_component / n_components.

    Assumptions: Identical independent exponential lifetimes for the unadjusted formula;
    any component failure interrupts the cluster.

    Excludes: Repair times, redundancy, and an explicit joint model of correlated
    failures.

    A factor below one represents a shorter effective MTBF. It does not itself
    specify the structure or timing of correlated failures.

    References
    ----------
    Derivation: Independent exponential survival probabilities multiply, so their
    hazards add.
    The correlation multiplier is a declared modeling adjustment, not derived from that
    independence.

    Examples
    --------
    >>> one = calc_mtbf_cluster(1000, 1)
    >>> two = calc_mtbf_cluster(1000, 2)
    >>> two * 2 == one
    True
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
    """Combine heterogeneous component hazards into node MTBF.

    Parameters
    ----------
    gpu_mtbf_h : Quantity or float
        Positive GPU mean time between failures; bare numbers are hours.
    n_gpus : int
        Nonnegative count of failure-critical GPUs.
    nic_mtbf_h : Quantity or float
        Positive NIC mean time between failures; bare numbers are hours.
    n_nics : int
        Nonnegative count of failure-critical NICs.
    psu_mtbf_h : Quantity or float
        Positive PSU mean time between failures; bare numbers are hours.
    n_psus : int
        Nonnegative count of failure-critical power supplies.
    other_mtbf_h : Quantity or float, optional
        Positive MTBF for another component class, in hours; None omits this class.
    n_other : int, optional
        Nonnegative count in the optional class; default zero.

    Returns
    -------
    Quantity
        Node MTBF in hours, given a positive summed hazard.

    Notes
    -----
    Equation: MTBF_node = 1 / (n_gpus/gpu_mtbf_h + n_nics/nic_mtbf_h + n_psus/psu_mtbf_h
    + n_other/other_mtbf_h).

    Assumptions: Independent exponential failures; each counted component can interrupt
    the node. The optional class contributes only when supplied and its count is
    positive.

    Excludes: Redundant-system state transitions, common-cause failures, and repair
    behavior.

    For redundant power supplies, pass the number that individually fail the
    node (for example Node.series_psus), rather than the physical supply count.

    References
    ----------
    Derivation: The minimum of independent exponential lifetimes has a hazard equal
    to the sum of the component hazards.

    Examples
    --------
    >>> two = calc_mtbf_node(1000, 2, 1000, 0, 1000, 0)
    >>> one = calc_mtbf_node(1000, 1, 1000, 0, 1000, 0)
    >>> two * 2 == one
    True
    """
    gpu = _ensure_unit(gpu_mtbf_h, ureg.hour, "gpu_mtbf_h")
    nic = _ensure_unit(nic_mtbf_h, ureg.hour, "nic_mtbf_h")
    psu = _ensure_unit(psu_mtbf_h, ureg.hour, "psu_mtbf_h")
    rate = n_gpus / gpu + n_nics / nic + n_psus / psu
    if other_mtbf_h is not None and n_other > 0:
        rate += n_other / _ensure_unit(other_mtbf_h, ureg.hour, "other_mtbf_h")
    return (1.0 / rate).to(ureg.hour)


def calc_availability_stacked(single_availability, n_replicas):
    """Calculate availability when any replica can serve.

    Parameters
    ----------
    single_availability : float
        Per-replica steady-state availability in [0, 1].
    n_replicas : int
        Positive count of interchangeable replicas.

    Returns
    -------
    float
        Dimensionless probability that at least one replica is available.

    Notes
    -----
    Equation: A = 1 - (1 - ``single_availability)**n_replicas``.

    Assumptions: Replica availability events are independent and any available replica
    suffices.

    Excludes: Common-cause outages, load capacity, failover delay, and replication lag.

    References
    ----------
    Derivation: All replicas are unavailable with probability ``(1-a)**n``; complementing
    that event gives the at-least-one-up probability.

    Examples
    --------
    >>> calc_availability_stacked(0.9, 2) > calc_availability_stacked(0.9, 1)
    True
    """
    return 1.0 - (1.0 - single_availability) ** n_replicas


def calc_failure_probability(mtbf, job_duration):
    """Calculate the chance of at least one failure during a job.

    Parameters
    ----------
    mtbf : Quantity or float
        Positive mean time between failures; supply a time quantity or a bare number.
    job_duration : Quantity or float
        Nonnegative duration; supply a time quantity if MTBF has units, otherwise use
        the same implicit time unit.

    Returns
    -------
    float
        Failure probability in [0, 1], subject to floating-point rounding.

    Notes
    -----
    Equation: P = 1 - exp(-job_duration / mtbf).

    Assumptions: Constant Poisson failure rate 1/MTBF, independent of job age. Both
    inputs are quantities or both are bare numbers.

    Excludes: Aging hazards, correlated failures, and recovery or retry costs.

    Mixing a bare number with a quantity raises TypeError because its implied
    unit would be ambiguous. For short jobs, the probability approaches T/MTBF.

    References
    ----------
    Derivation: A constant-hazard lifetime survives duration T with probability
    exp(-T/MTBF); the complement is the probability of at least one failure.

    Examples
    --------
    >>> calc_failure_probability(1000, 0)
    0.0
    >>> calc_failure_probability(1000, 20) > calc_failure_probability(1000, 10)
    True
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
    """Calculate availability when every component must be up.

    Parameters
    ----------
    availability : float
        Per-component steady-state availability in [0, 1].
    n_components : int
        Number of required components, at least one.

    Returns
    -------
    float
        Dimensionless joint availability.

    Notes
    -----
    Equation: A_all = ``availability**n_components``.

    Assumptions: Independent and identically available components; every component is
    required.

    Excludes: Redundancy, common-cause failures, and partial-service operation.

    References
    ----------
    Derivation: Independence turns the intersection of all component-up events
    into the product of their individual probabilities.

    Examples
    --------
    >>> calc_all_up_availability(0.9, 2) < calc_all_up_availability(0.9, 1)
    True
    """
    validate_at_least(n_components, 1, "n_components")
    return float(availability) ** n_components


def calc_downtime(availability, interval):
    """Calculate expected downtime over a steady-state interval.

    Parameters
    ----------
    availability : float
        Steady-state fraction of time available in [0, 1].
    interval : Quantity or float
        Nonnegative observation duration; returned time retains its units or implicit
        unit.

    Returns
    -------
    Quantity or float
        Expected downtime in the same time unit as interval.

    Notes
    -----
    Equation: downtime = (1 - availability) * interval.

    Assumptions: The availability fraction applies throughout the observation interval.

    Excludes: Outage clustering, recovery trajectories, and downtime quantiles.

    References
    ----------
    Derivation: Multiplying total observation time by its unavailable fraction
    gives expected time down.

    Examples
    --------
    >>> calc_downtime(1.0, 24 * ureg.hour).m_as("hour")
    0.0
    """
    return (1 - float(availability)) * interval


def calc_checkpoint_waste(write_time, interval, mtbf):
    """Estimate the wall-time fraction lost to saves and rework.

    Parameters
    ----------
    write_time : Quantity or float
        Nonnegative duration of one blocking checkpoint; bare numbers are seconds.
    interval : Quantity or float
        Positive spacing between checkpoints; bare numbers are seconds.
    mtbf : Quantity or float
        Positive mean time between failures; bare numbers are seconds.

    Returns
    -------
    float
        Dimensionless first-order waste; values above one signal an unsuitable regime,
        not a capped fraction.

    Notes
    -----
    Equation: W = write_time / interval + interval / (2 * mtbf).

    Assumptions: Rare independent failures and uniformly distributed failure position
    within an interval. Saves block useful work.

    Excludes: Restart overhead, lost checkpoint writes, overlapping saves, and
    higher-order interactions.

    At the minimizing Young interval, the save and rework contributions are
    equal. This is an approximation to wasted wall time rather than a
    simulation of failure and restart events.

    References
    ----------
    Young (1974), "A first order approximation to the optimum checkpoint interval."
    https://doi.org/10.1145/361147.361115 (bibliography key: young1974).

    Examples
    --------
    >>> tau = calc_young_daly_interval(10, 10000)
    >>> optimum = calc_checkpoint_waste(10, tau, 10000)
    >>> optimum < calc_checkpoint_waste(10, tau * 2, 10000)
    True
    """
    delta = _ensure_unit(write_time, ureg.second, "write_time")
    tau = _ensure_unit(interval, ureg.second, "interval")
    mtbf_q = _ensure_unit(mtbf, ureg.second, "mtbf")
    validate_positive(tau, "interval")
    validate_positive(mtbf_q, "mtbf")
    return float((delta / tau + tau / (2 * mtbf_q)).to("").magnitude)


def calc_step_event_probability(rate_per_device_hour, n_devices, step_time):
    """Calculate event exposure across device-hours in one step.

    Parameters
    ----------
    rate_per_device_hour : float
        Probability of an event in one device-hour, in [0, 1]; this is not a continuous
        Poisson rate.
    n_devices : int
        Nonnegative device count exposed to the event.
    step_time : Quantity or float
        Nonnegative step duration; bare numbers are seconds.

    Returns
    -------
    float
        Dimensionless probability of one or more events across the exposure.

    Notes
    -----
    Equation: p = 1 - (1 - ``rate_per_device_hour)**(n_devices`` * step_time_hours).

    Assumptions: Independent devices and multiplicative hourly survival, extended to
    fractional hours by exponentiation.

    Excludes: Common-cause events, varying hazards, and event severity.

    References
    ----------
    Derivation: Raise hourly survival to total device-hours, then complement it.
    The implied constant hazard is -log(1-rate_per_device_hour), not the supplied
    probability itself.

    Examples
    --------
    >>> calc_step_event_probability(0.01, 1, 0)
    0.0
    >>> calc_step_event_probability(0.01, 2, 3600) > calc_step_event_probability(0.01, 1, 3600)
    True
    """
    step = _ensure_unit(step_time, ureg.second, "step_time")
    exposure = n_devices * step.m_as(ureg.hour)
    return 1 - (1 - float(rate_per_device_hour)) ** exposure


def calc_probe_detection_window(interval, timeout, failure_threshold):
    """Bound crash-to-removal delay for periodic health probes.

    Parameters
    ----------
    interval : Quantity or float
        Positive time between probe starts; bare numbers are seconds.
    timeout : Quantity or float
        Nonnegative delay for a failed probe to time out; bare numbers are seconds.
    failure_threshold : int
        Number of consecutive failed probes required for removal, at least one.

    Returns
    -------
    tuple of Quantity
        Earliest and latest removal delays, both in seconds.

    Notes
    -----
    Equation: earliest = (failure_threshold - ``1)*interval`` + timeout; latest = earliest +
    interval.

    Assumptions: The replica previously passed probes; all subsequent probes fail, with
    fixed start cadence and timeout.

    Excludes: Probe scheduler jitter, transport variation, and removal propagation
    delay.

    References
    ----------
    Derivation: A crash immediately before a probe waits for threshold-1 more
    probe periods plus one timeout; a crash just after a successful probe adds
    one full period.

    Examples
    --------
    >>> earliest, latest = calc_probe_detection_window(5, 1, 3)
    >>> latest - earliest == 5 * ureg.second
    True
    """
    period = _ensure_unit(interval, ureg.second, "interval")
    wait = _ensure_unit(timeout, ureg.second, "timeout")
    validate_at_least(failure_threshold, 1, "failure_threshold")
    earliest = (failure_threshold - 1) * period + wait
    return earliest.to(ureg.second), (earliest + period).to(ureg.second)


def calc_error_budget_spent(burn_rate, window, budget_period):
    """Calculate the share of an error budget consumed by a burn.

    Parameters
    ----------
    burn_rate : float
        Nonnegative rate relative to the rate that exhausts the budget over
        budget_period.
    window : Quantity
        Nonnegative duration of sustained burn, with time units.
    budget_period : Quantity
        Positive duration of the complete error-budget period.

    Returns
    -------
    float
        Dimensionless budget share; may exceed one after overspending.

    Notes
    -----
    Equation: share = burn_rate * window / budget_period.

    Assumptions: Constant burn rate across the window and a fixed budget-period
    definition.

    Excludes: Traffic weighting changes, rolling-window edge effects, and alerting
    policy.

    References
    ----------
    Google, "Alerting on SLOs," The Site Reliability Workbook.
    https://sre.google/workbook/alerting-on-slos/

    Examples
    --------
    >>> calc_error_budget_spent(1, 1 * ureg.day, 1 * ureg.day)
    1.0
    """
    return float(burn_rate * (window / budget_period).to("").magnitude)


# --- Durable execution: logs, checkpoints, and recovery data (added 2026-10-20) ---


def calc_group_commit_waits(batch_timer, flush_time, batch_size):
    """Separate per-record flush work from group-commit wait time.

    Parameters
    ----------
    batch_timer : Quantity or float
        Nonnegative collection window after the first arrival; bare numbers are seconds.
    flush_time : Quantity or float
        Nonnegative duration of the single batch flush; bare numbers are seconds.
    batch_size : int
        Number of records sharing the flush, at least one.

    Returns
    -------
    tuple of Quantity
        Per-record flush work, first-arrival wait, and last-arrival wait, all in
        milliseconds.

    Notes
    -----
    Equation: work = flush_time/batch_size; first_wait = batch_timer + flush_time;
    last_wait = flush_time.

    Assumptions: One flush acknowledges the entire batch, the first arrival starts the
    timer, and the last arrives when the timer expires.

    Excludes: Prior queued batches, encoding costs, size-dependent flush cost, and
    arrival-distribution averages.

    References
    ----------
    DeWitt et al. (1984), "Implementation Techniques for Main Memory Database Systems"
    (bibliography key: dewitt1984).
    Derivation: Divide one shared flush by record count for charged work; add
    timer and flush durations on each arrival's waiting path.

    Examples
    --------
    >>> work, first, last = calc_group_commit_waits(0.01, 0.002, 4)
    >>> work * 4 == last and first > last
    True
    """
    timer = _ensure_unit(batch_timer, ureg.second, "batch_timer")
    flush = _ensure_unit(flush_time, ureg.second, "flush_time")
    validate_at_least(batch_size, 1, "batch_size")
    return (flush / batch_size).to(ureg.ms), (timer + flush).to(ureg.ms), flush.to(ureg.ms)


def calc_snapshot_write_volume(state_growth, stride, checkpoints):
    """Compare accumulated bytes in full snapshots and deltas.

    Parameters
    ----------
    state_growth : Quantity or float
        Nonnegative state growth per turn; bare numbers are bytes.
    stride : int
        Turns between checkpoints, at least one.
    checkpoints : int
        Number of checkpoints, at least one.

    Returns
    -------
    tuple of Quantity
        Total bytes written by full snapshots and deltas, respectively.

    Notes
    -----
    Equation: full = ``state_growth*stride*n*(n+1)/2``; delta = ``state_growth*stride*n``, with
    n = checkpoints.

    Assumptions: Zero initial state and constant append-only growth; a full snapshot
    rewrites all accumulated state.

    Excludes: Compression, deleted state, metadata, replication, and storage write
    amplification.

    References
    ----------
    Derivation: Sum snapshot sizes proportional to 1+2+...+n; deltas each
    write only one stride of new state.

    Examples
    --------
    >>> full, delta = calc_snapshot_write_volume(100, 2, 3)
    >>> full > delta
    True
    >>> calc_snapshot_write_volume(100, 2, 1)[0] == calc_snapshot_write_volume(100, 2, 1)[1]
    True
    """
    growth = _ensure_unit(state_growth, ureg.byte, "state_growth")
    validate_at_least(stride, 1, "stride")
    validate_at_least(checkpoints, 1, "checkpoints")
    full = growth * stride * checkpoints * (checkpoints + 1) / 2
    delta = growth * stride * checkpoints
    return full.to(ureg.byte), delta.to(ureg.byte)


def calc_checkpoint_stride(pause, crash_rate, lost_turn_cost):
    """Find a continuous checkpoint stride balancing pause and rework.

    Parameters
    ----------
    pause : Quantity or float
        Nonnegative checkpoint pause; bare numbers are seconds.
    crash_rate : float
        Positive expected crashes per turn.
    lost_turn_cost : Quantity or float
        Positive recovery cost per lost turn; bare numbers are seconds.

    Returns
    -------
    float
        Optimal continuous stride in turns; not rounded to an integer policy.

    Notes
    -----
    Equation: k_star = ``sqrt(2*pause/(crash_rate*lost_turn_cost))``.

    Assumptions: A checkpoint every k turns costs pause/k per turn; crashes lose k/2
    turns on average.

    Excludes: Integer scheduling, restart costs independent of lost turns, and
    correlated crashes.

    References
    ----------
    Young (1974), "A first order approximation to the optimum checkpoint interval."
    https://doi.org/10.1145/361147.361115 (bibliography key: young1974).
    Derivation: Differentiate pause/k + ``crash_rate*lost_turn_cost*k/2`` with respect
    to the continuous stride k and set the derivative to zero.

    Examples
    --------
    >>> stride = calc_checkpoint_stride(1, 0.01, 1)
    >>> optimum = calc_stride_overhead(1, stride, 0.01, 1)
    >>> optimum < calc_stride_overhead(1, stride * 2, 0.01, 1)
    True
    """
    delta = _ensure_unit(pause, ureg.second, "pause")
    cost = _ensure_unit(lost_turn_cost, ureg.second, "lost_turn_cost")
    validate_positive(crash_rate, "crash_rate")
    validate_positive(cost, "lost_turn_cost")
    return math.sqrt(2 * delta.m_as(ureg.second) / (crash_rate * cost.m_as(ureg.second)))


def calc_stride_overhead(pause, stride, crash_rate, lost_turn_cost):
    """Estimate checkpoint and expected-rework overhead per turn.

    Parameters
    ----------
    pause : Quantity or float
        Nonnegative pause per checkpoint; bare numbers are seconds.
    stride : float
        Positive number of turns between checkpoints.
    crash_rate : float
        Nonnegative expected crashes per turn.
    lost_turn_cost : Quantity or float
        Nonnegative recovery cost per lost turn; bare numbers are seconds.

    Returns
    -------
    Quantity
        Expected overhead per turn in seconds.

    Notes
    -----
    Equation: overhead = pause/stride + ``crash_rate*lost_turn_cost*stride/2``.

    Assumptions: Each checkpoint blocks work and a crash loses half a checkpoint stride
    on average.

    Excludes: Recovery costs unrelated to lost work, correlated failures, and
    interactions between failures.

    References
    ----------
    Young (1974), "A first order approximation to the optimum checkpoint interval."
    https://doi.org/10.1145/361147.361115 (bibliography key: young1974).
    Derivation: Amortize one checkpoint over k turns and add the expected crash
    cost per turn.

    Examples
    --------
    >>> calc_stride_overhead(2, 4, 0, 1) * 4 == 2 * ureg.second
    True
    """
    delta = _ensure_unit(pause, ureg.second, "pause")
    cost = _ensure_unit(lost_turn_cost, ureg.second, "lost_turn_cost")
    validate_positive(stride, "stride")
    return (delta / stride + crash_rate * cost * stride / 2).to(ureg.second)
