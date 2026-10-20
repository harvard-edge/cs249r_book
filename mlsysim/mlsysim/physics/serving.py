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
