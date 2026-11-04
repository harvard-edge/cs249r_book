"""
Tool calling: the equations behind a mediated call, its retry, and its catalog.

Pure arithmetic (added 2026-10-20). A call whose request or reply may be lost
times out with one probability and, under one blind retry, executes twice
with another; a model that polls a pending job makes one status call per
interval before completion; a tool catalog staged on every call costs its
descriptors, and selection pays once the descriptors it leaves out exceed its
own overhead; a shared protocol turns pairwise adapters into one per party.
The solvers in engine/solvers/tool_calls.py compose these.
"""

import math as _math


def calc_timeout_probability(request_loss: float, reply_loss: float) -> float:
    """Return timeout probability for independently lost request and reply.

    Parameters
    ----------
    request_loss : float
        Probability that a request is lost; between zero and one.
    reply_loss : float
        Probability that the reply is lost after delivery; between zero and one.

    Returns
    -------
    float
        Probability of a timeout on one attempt.

    Notes
    -----
    Equation: P(timeout) = 1 - (1 - ``request_loss)*(1`` - reply_loss).

    Assumptions: Delivery succeeds only if both messages arrive; message losses are independent.

    Excludes: Finite timeout thresholds, delayed arrival, server failures, and retries.

    References
    ----------
    Derivation: Take the complement of joint request-and-reply delivery under independence.

    Examples
    --------
    >>> calc_timeout_probability(0.0, 0.5) == 0.5
    True
    """
    return 1 - (1 - request_loss) * (1 - reply_loss)


def calc_blind_retry_duplicate_probability(request_loss: float, reply_loss: float) -> float:
    """Return the unconditional probability of a duplicate effect after one retry.

    Parameters
    ----------
    request_loss : float
        Loss probability of each request transmission; between zero and one.
    reply_loss : float
        Loss probability of the original reply; between zero and one.

    Returns
    -------
    float
        Probability that the original and retry both execute the effect.

    Notes
    -----
    Equation: P(duplicate) = (1 - ``request_loss)*reply_loss*(1`` - request_loss).

    Assumptions: A timeout causes one blind retry; requests are independent and lost messages never arrive later.

    Excludes: Deduplication, idempotent effects, delayed messages, and further retries.

    References
    ----------
    Derivation: Multiply the probabilities of original delivery, lost original reply, and retry delivery.

    Examples
    --------
    >>> calc_blind_retry_duplicate_probability(0.0, 0.5) == 0.5
    True
    >>> calc_blind_retry_duplicate_probability(1.0, 0.5) == 0.0
    True
    """
    return (1 - request_loss) * reply_loss * (1 - request_loss)


def calc_pending_polls(job_duration, poll_interval) -> int:
    """Count polling calls that occur before job completion.

    Parameters
    ----------
    job_duration : Quantity
        Positive duration of the job, with time units.
    poll_interval : Quantity
        Positive interval between polls, with compatible time units.

    Returns
    -------
    int
        Number of polls returning a pending status before completion.

    Notes
    -----
    Equation: pending = ceil(job_duration / poll_interval) - 1.

    Assumptions: Polling starts one interval after dispatch; a poll at completion sees a completed result.

    Excludes: Polling latency, timer jitter, adaptive intervals, and zero-duration jobs.

    References
    ----------
    Derivation: Count positive interval multiples strictly less than the completion time.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> calc_pending_polls(Q_("3 s"), Q_("1 s")) == 2
    True
    """
    ratio = (job_duration / poll_interval).to("").magnitude
    return _math.ceil(ratio) - 1


def calc_selection_break_even(selection_overhead: float, descriptor_tokens: float) -> float:
    """Return the number of descriptors that offsets tool-selection overhead.

    Parameters
    ----------
    selection_overhead : float
        Nonnegative token overhead of selecting tools.
    descriptor_tokens : float
        Positive serialized token count per omitted tool descriptor.

    Returns
    -------
    float
        Omitted-descriptor count at equal token cost; savings require a larger count.

    Notes
    -----
    Equation: break_even = selection_overhead / descriptor_tokens.

    Assumptions: Descriptor sizes are uniform and selection overhead is paid once per request.

    Excludes: Selection accuracy, invocation latency, heterogeneous descriptor sizes, and integer rounding.

    References
    ----------
    Derivation: Solve ``omitted_count*descriptor_tokens`` = selection_overhead for omitted_count.

    Examples
    --------
    >>> calc_selection_break_even(100, 20) == 100 / 20
    True
    """
    return selection_overhead / descriptor_tokens


def calc_integration_adapters(hosts: int, services: int):
    """Count pairwise and shared-protocol integration adapters.

    Parameters
    ----------
    hosts : int
        Number of client hosts; nonnegative.
    services : int
        Number of services reached by every host; nonnegative.

    Returns
    -------
    tuple
        Pairwise adapter count and shared-protocol endpoint adapter count.

    Notes
    -----
    Equation: pairwise = ``hosts*services``; shared = hosts + services.

    Assumptions: Every host connects to every service; each protocol endpoint needs one adapter.

    Excludes: Adapter complexity, protocol gateways, compatibility testing, and partial connectivity.

    References
    ----------
    Derivation: Count edges in a complete bipartite integration graph, then count its endpoint adapters.

    Examples
    --------
    >>> calc_integration_adapters(2, 3) == (2 * 3, 2 + 3)
    True
    """
    return hosts * services, hosts + services


# --- Containing shared failures: retries, jitter, bulkheads (added 2026-10-20) ---


def calc_layered_retry_attempts(attempts_per_layer: int, layers: int) -> int:
    """Count downstream attempts under nested retry allowances.

    Parameters
    ----------
    attempts_per_layer : int
        Attempts allowed per invocation at each layer, including the initial attempt; positive.
    layers : int
        Number of independently retrying layers; nonnegative.

    Returns
    -------
    int
        Worst-case downstream attempt count for one logical operation.

    Notes
    -----
    Equation: attempts = attempts_per_layer ** layers.

    Assumptions: Every attempt fails and exhausts all lower-layer allowances before an outer retry.

    Excludes: Early success, shared retry budgets, deadlines, and deduplication.

    References
    ----------
    Derivation: Multiply equal allowances across each nested layer of the retry tree.

    Examples
    --------
    >>> calc_layered_retry_attempts(3, 2) == 3 * 3
    True
    """
    return attempts_per_layer**layers


def calc_jitter_expected_arrivals(retries: float, window, jitter_span) -> float:
    """Estimate retries arriving inside a uniform-jitter window.

    Parameters
    ----------
    retries : float
        Nonnegative number of independently jittered retries.
    window : Quantity or float
        Observation-window duration, from zero to jitter_span; use compatible time units.
    jitter_span : Quantity or float
        Positive width of the uniform retry-time distribution.

    Returns
    -------
    float
        Expected retry count in the specified window.

    Notes
    -----
    Equation: E[arrivals] = ``retries*window/jitter_span``.

    Assumptions: The window lies entirely inside the uniform jitter support; float durations share one unit.

    Excludes: Realized burst size, nonuniform jitter, dependence, and windows outside the support.

    An expectation is not a capacity bound: a realized window can contain every retry.

    References
    ----------
    Derivation: Multiply retry count by the fraction of uniform timing mass within the observation window.

    Examples
    --------
    >>> calc_jitter_expected_arrivals(10, 1, 2) == 10 / 2
    True
    """
    ratio = retries * window / jitter_span
    return float(ratio.to("").magnitude) if hasattr(ratio, "to") else float(ratio)


def calc_bulkhead_admission(arrivals: int, workers: int, worker_cap: int, queue_cap: int):
    """Partition a burst into active, queued, and rejected calls.

    Parameters
    ----------
    arrivals : int
        Number of calls arriving in one burst; nonnegative.
    workers : int
        Total worker count; nonnegative.
    worker_cap : int
        Maximum workers allocated to this dependency; nonnegative.
    queue_cap : int
        Queue slots that hold calls without occupying workers; nonnegative.

    Returns
    -------
    tuple
        Occupied workers, queued calls, rejected calls, and remaining free workers.

    Notes
    -----
    Equation: occupied = min(arrivals, worker_cap, workers); queued = min(arrivals - occupied, queue_cap).

    Assumptions: No call finishes during admission; each active call occupies one worker.

    Excludes: Service duration, queue waiting time, cancellations, and future arrivals.

    References
    ----------
    Derivation: Allocate the worker budget first, then queue capacity, and reject the unallocated remainder; subtract occupied workers from total workers.

    Examples
    --------
    >>> occupied, queued, rejected, free = calc_bulkhead_admission(8, 4, 2, 3)
    >>> occupied + queued + rejected == 8 and occupied + free == 4
    True
    """
    occupied = min(arrivals, worker_cap, workers)
    queued = min(arrivals - occupied, queue_cap)
    rejected = arrivals - occupied - queued
    return occupied, queued, rejected, workers - occupied
