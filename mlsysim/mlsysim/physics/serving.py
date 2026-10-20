"""Inference serving and queueing models."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg

from ._units import _ensure_unit


def calc_queue_latency_mmc(arrival_rate_hz, service_rate_hz, num_servers):
    """
    M/M/c queueing model for inference tail latency (Erlang C).

    Calculates the expected queueing delays (P50 and P99) for a system with 
    Markovian arrivals, Markovian service times, and `c` parallel servers.

    This implementation uses the Log-Sum-Exp trick to calculate the Erlang C 
    formula. This prevents floating-point overflow (`math.inf`) or underflow 
    to `0.0` when dealing with large-scale clusters (e.g., c > 100).

    Parameters
    ----------
    arrival_rate_hz : Quantity or float
        The average rate of incoming requests (λ) in requests per second (Hz).
    service_rate_hz : Quantity or float
        The average rate at which a single server completes requests (μ) in Hz.
    num_servers : int
        The number of active parallel serving replicas (c).

    Returns
    -------
    tuple
        A 3-tuple containing:
        - rho (float): Server utilization (λ / (c * μ)).
        - p50_wait (Quantity): The 50th percentile queueing wait time.
        - p99_wait (Quantity): The 99th percentile queueing wait time.
    """
    lam = _ensure_unit(arrival_rate_hz, ureg.hertz, "arrival_rate_hz").magnitude
    mu = _ensure_unit(service_rate_hz, ureg.hertz, "service_rate_hz").magnitude
    c = max(1, int(num_servers))

    # Unstable regime: arrivals meet or exceed total service capacity, so the
    # queue grows without bound — report saturation and infinite waits.
    if lam >= c * mu or mu == 0:
        return 1.0, float("inf") * ureg.second, float("inf") * ureg.second

    rho = lam / (c * mu)
    a = c * rho  # offered load in Erlangs (mean number of busy servers)
    try:
        # Erlang C, computed in the log domain. The probability an arrival must
        # wait is  P_wait = T_c / (sum_{i<c} a^i/i! + T_c)  with the "all servers
        # busy" term  T_c = a^c / (c! * (1 - rho)).  At cluster scale (c > ~100)
        # a^c and c! overflow doubles, so each term is carried as its logarithm
        # (lgamma(i+1) = log i!) and combined via log-sum-exp.
        log_last = c * math.log(a) - math.lgamma(c + 1) - math.log(1 - rho)
        # Series terms a^i/i! for i < c: the Poisson-shaped "i servers busy,
        # no queue" states. a == 0 means no load: only the i=0 term survives.
        log_terms = [
            i * math.log(a) - math.lgamma(i + 1) if a > 0 else (-math.inf if i > 0 else 0.0)
            for i in range(c)
        ]
        # Log-sum-exp: factor out the largest exponent so every exp() argument
        # is <= 0 and nothing overflows; the common factor cancels in the ratio.
        max_log = max(max(log_terms) if log_terms else -math.inf, log_last)
        sum_exp = sum(math.exp(t - max_log) for t in log_terms) + math.exp(log_last - max_log)
        p_wait = math.exp(log_last - max_log) / sum_exp
    except (OverflowError, ValueError, ZeroDivisionError):
        # Numerical breakdown despite the log-domain guard: fall back to rho,
        # a crude but monotone proxy for the waiting probability.
        p_wait = rho

    if math.isnan(p_wait) or math.isinf(p_wait):
        p_wait = rho
    p_wait = max(0.0, min(1.0, p_wait))  # clamp: it is a probability

    # Conditional on waiting, M/M/c wait time is exponential with rate
    # c*mu*(1-rho), so the unconditional tail is P(W > t) = p_wait * e^(-rate*t).
    # Inverting at quantile q gives t_q = -ln(q / p_wait) / rate; when p_wait is
    # already below the tail mass q, that percentile of requests never queues.
    rate_param = c * mu * (1 - rho)
    # max(0.0, ...) also normalizes the -0.0 produced at the exact
    # p_wait == quantile boundary (e.g. M/M/1 at rho = 0.5).
    p50_wait = 0.0 if p_wait < 0.5 else max(0.0, -math.log(0.5 / p_wait) / rate_param)
    p99_wait = 0.0 if p_wait < 0.01 else max(0.0, -math.log(0.01 / p_wait) / rate_param)
    return rho, p50_wait * ureg.second, p99_wait * ureg.second


def calc_mg1_wait_ratio(utilization, service_cv):
    """Mean M/G/1 queue wait as a multiple of the mean service time.

    The Pollaczek-Khinchine formula for one server with Poisson arrivals and a
    general service-time distribution:

        W_q / (1/mu) = rho / (1 - rho) * (1 + C_s^2) / 2

    The first factor (load) grows without bound as ``utilization`` approaches 1;
    the second (variability) scales the whole curve by the spread of service
    times, ``service_cv`` being their coefficient of variation (1 for
    exponential service). Added 2026-10-20 for scheduler queueing, where a
    fleet is modeled as one server working through jobs.

    Parameters
    ----------
    utilization : float
        Offered load rho, in [0, 1).
    service_cv : float
        Coefficient of variation of service time, C_s >= 0.

    Returns
    -------
    float
        Mean queue wait divided by the mean service time (``inf`` at rho >= 1).
    """
    rho = float(utilization)
    cv = float(service_cv)
    if rho < 0 or cv < 0:
        raise ValueError("utilization and service_cv must be nonnegative")
    if rho >= 1:
        return float("inf")
    return rho / (1 - rho) * (1 + cv * cv) / 2


# --- Serving at scale: occupancy, batching, speculation (added 2026-10-20) -----

def calc_littles_law_occupancy(arrival_rate, time_in_system):
    """
    Mean requests in a stable system, by Little's law.

        L = lambda W

    ``arrival_rate`` (requests per unit time) times ``time_in_system`` (the
    mean time each request spends inside). The law needs only stability, not
    a distribution, so it gives a mean and says nothing about the tail.
    """
    return arrival_rate * time_in_system


def calc_batch_knee(fixed_time, per_request_time):
    """
    Batch size at which per-request cost equals the per-batch fixed cost.

        B_knee = t_0 / t_req

    With batch service time S(B) = t_0 + B t_req, throughput B / S(B) reaches
    half its ceiling 1 / t_req at this batch.
    """
    return (fixed_time / per_request_time).to("").magnitude


def calc_batch_throughput_share(batch_size, fixed_time, per_request_time) -> float:
    """Throughput at ``batch_size`` as a share of the ceiling 1 / t_req: B t_req / (t_0 + B t_req)."""
    work = batch_size * per_request_time
    return (work / (fixed_time + work)).to("").magnitude


def calc_padding_waste_fraction(lengths) -> float:
    """
    Share of slot time a fixed-membership batch leaves idle.

        W = 1 - mean(L) / max(L)

    Every member holds its slot until the longest member finishes, so slot
    time is ``n max(L)`` while useful work is ``sum(L)``.
    """
    lengths = [float(x) for x in lengths]
    if not lengths or max(lengths) <= 0:
        raise ValueError("lengths must be nonempty and positive")
    return 1 - (sum(lengths) / len(lengths)) / max(lengths)


def calc_speculative_expected_tokens(acceptance: float, draft_length: int) -> float:
    """
    Expected tokens one speculative round emits.

        E[tokens] = (1 - alpha^(K+1)) / (1 - alpha)

    A draft proposes ``draft_length`` (K) tokens, each accepted independently
    with probability ``acceptance`` (alpha) until the first rejection; the
    target's verification pass always contributes one token.
    """
    a = float(acceptance)
    if not 0 <= a < 1:
        raise ValueError("acceptance must be in [0, 1)")
    return (1 - a ** (draft_length + 1)) / (1 - a)
# --- Residence time and turn rate (added 2026-10-20) --------------------------

def calc_littles_law_rate(occupied_slots, residence_time):
    """
    Turns per unit time that ``occupied_slots`` sustain, by Little's law.

        L = lambda W   =>   lambda = L / W

    ``residence_time`` is the time each turn holds its slot. The law assumes
    only a stable system, so the rate is an average, not a peak.
    """
    return occupied_slots / residence_time


def calc_residency_factor(generation_time, wait_time) -> float:
    """
    How much holding a slot through a wait divides the turn rate.

        1 + T_wait / T_gen

    Keeping state through a wait stretches each turn's residence from
    ``generation_time`` to ``generation_time + wait_time``; for the same slots,
    Little's law divides the rate by this factor (the residency tax).
    """
    return ((generation_time + wait_time) / generation_time).to("").magnitude


# --- One-server queueing and batch formation (added 2026-10-20) ---------------

def calc_mm1_time_in_system(service_time, utilization):
    """
    Mean time a request spends in an M/M/1 server, waiting plus service.

        W = S / (1 - rho)

    ``service_time`` (S) is the mean service time and ``utilization`` (rho)
    the fraction of time the server is busy. W grows without bound as rho
    approaches 1: the queueing knee.
    """
    rho = float(utilization)
    if not 0 <= rho < 1:
        raise ValueError("utilization must be in [0, 1)")
    return service_time / (1 - rho)


def calc_exponential_quantile_factor(quantile: float) -> float:
    """
    A quantile of an exponential distribution as a multiple of its mean.

        k_q = ln(1 / (1 - q))

    Time in an M/M/1 system is exponential, so its p99 is ln(100) = 4.605
    times its mean at every load.
    """
    q = float(quantile)
    if not 0 <= q < 1:
        raise ValueError("quantile must be in [0, 1)")
    return math.log(1 / (1 - q))


def calc_mm1_max_utilization(service_time, tail_budget, tail_factor: float) -> float:
    """
    Highest utilization at which an M/M/1 tail stays within ``tail_budget``.

        k S / (1 - rho) <= T   =>   rho <= 1 - k S / T

    ``tail_factor`` (k) is the tail as a multiple of the mean time in system
    (``calc_exponential_quantile_factor``).
    """
    return 1 - (tail_factor * service_time / tail_budget).to("").magnitude


def calc_batch_formation_wait(batch_size: int, arrival_rate):
    """
    Mean time a request waits for a batch of ``batch_size`` to fill.

        W_form = (B - 1) / (2 lambda)

    With arrivals at a steady ``arrival_rate`` (lambda), the first request of
    a batch waits for B - 1 more and the last for none; the mean is half.
    """
    return (batch_size - 1) / (2 * arrival_rate)
