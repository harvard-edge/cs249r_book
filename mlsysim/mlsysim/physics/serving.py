"""Inference serving and queueing models."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg

from ._units import _ensure_unit


def calc_queue_latency_mmc(arrival_rate_hz, service_rate_hz, num_servers):
    """Calculate M/M/c utilization and queue-wait percentiles.

    Parameters
    ----------
    arrival_rate_hz : Quantity or float
        Nonnegative request arrival rate lambda; bare numbers mean Hz.
    service_rate_hz : Quantity or float
        Positive completion rate mu per replica; bare numbers mean Hz.
    num_servers : int
            Replica count c; implementation truncates to an integer and clamps to at
        least one.

    Returns
    -------
    tuple
            Dimensionless utilization, median queue wait, and p99 queue wait; waits are
        second quantities.

    Notes
    -----
    Equation: rho = lambda/(c mu); P(W > t) = ErlangC(c, rho) exp[-c mu (1-rho)t]. The
    returned waits invert this tail at tail probabilities 0.5 and 0.01.

    Assumptions: Poisson arrivals, independent exponential service, identical replicas,
    and a shared first-come queue. Stable operation requires lambda < c mu.

    Excludes: Service duration in the returned waits, batching, admission limits,
    heterogeneous replicas, and burst correlations.

    The Erlang C sum is evaluated in the log domain to limit overflow. Numerical
    breakdown falls back to rho as a proxy waiting probability, which is an
    approximation. At saturation or zero service rate the function returns
    utilization one and infinite waits; these are not finite latency estimates.

    References
    ----------
    Harchol-Balter (2013), Performance Modeling and Design of Computer
    Systems: Queueing Theory in Action.
    https://www.cs.cmu.edu/~harchol/PerformanceModeling/book.html
    (bibliography key: harchol-balter2013).

    Examples
    --------
    >>> rho, median, tail = calc_queue_latency_mmc(1, 2, 1)
    >>> rho == 0.5 and median.magnitude == 0 and tail.magnitude > 0
    True
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
        log_terms = [i * math.log(a) - math.lgamma(i + 1) if a > 0 else (-math.inf if i > 0 else 0.0) for i in range(c)]
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
    """Calculate mean M/G/1 queue wait relative to mean service time.

    Parameters
    ----------
    utilization : float
        Nonnegative busy fraction rho; values at or above one return infinity.
    service_cv : float
        Nonnegative dimensionless coefficient of variation of service duration.

    Returns
    -------
    float
        Mean waiting time divided by mean service time; dimensionless.

    Notes
    -----
    Equation: Wq/E[S] = rho (1 + ``Cs**2)`` / [2 (1-rho)].

    Assumptions: Poisson arrivals, one first-come server, independent service times
    with finite second moment.

    Excludes: Tail percentiles, correlated service, finite queue capacity, and multiple
    servers.

    References
    ----------
    Harchol-Balter (2013), Performance Modeling and Design of Computer
    Systems: Queueing Theory in Action.
    https://www.cs.cmu.edu/~harchol/PerformanceModeling/book.html
    (bibliography key: harchol-balter2013).

    Examples
    --------
    >>> calc_mg1_wait_ratio(0.5, 1) == 1
    True
    >>> calc_mg1_wait_ratio(0.5, 0) < calc_mg1_wait_ratio(0.5, 1)
    True
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
    """Calculate mean occupancy from arrival rate and residence time.

    Parameters
    ----------
    arrival_rate : Quantity or float
            Nonnegative average arrivals per unit time; use units reciprocal to
        time_in_system.
    time_in_system : Quantity or float
        Nonnegative mean total residence duration, including waiting and service.

    Returns
    -------
    Quantity or float
        Mean resident request count; units follow the product of the inputs.

    Notes
    -----
    Equation: L = lambda W.

    Assumptions: A stable system and consistently defined boundaries for arrivals,
    occupancy, and residence.

    Excludes: Tail occupancy, burst reserves, and admission guarantees.

    References
    ----------
    Little (1961), A Proof for the Queuing Formula: L = lambda W.
    https://doi.org/10.1287/opre.9.3.383 (bibliography key: little1961).

    Examples
    --------
    >>> calc_littles_law_occupancy(2 / ureg.second, 3 * ureg.second).dimensionless
    True
    """
    return arrival_rate * time_in_system


def calc_batch_knee(fixed_time, per_request_time):
    """Calculate where variable batch work equals fixed work.

    Parameters
    ----------
    fixed_time : Quantity
        Nonnegative fixed duration per batch.
    per_request_time : Quantity
        Positive incremental duration per request; compatible time units.

    Returns
    -------
    float
        Continuous batch-size knee; an integer scheduler may need rounding.

    Notes
    -----
    Equation: Bknee = t0/treq; throughput B/(t0+B treq) is half its asymptotic ceiling
    at this point.

    Assumptions: Batch service duration is affine in batch size with fixed
    coefficients.

    Excludes: Queueing, capacity limits, nonlinear kernels, and discrete batch-size
    rounding.

    References
    ----------
    Derivation: equate the fixed term t0 and variable term B treq in the
    affine batch-service model and solve for B.

    Examples
    --------
    >>> calc_batch_knee(4 * ureg.ms, 2 * ureg.ms) == 2
    True
    """
    return (fixed_time / per_request_time).to("").magnitude


def calc_batch_throughput_share(batch_size, fixed_time, per_request_time) -> float:
    """Calculate batch throughput as a fraction of its affine-model ceiling.

    Parameters
    ----------
    batch_size : int or float
        Nonnegative request count B, with a positive total service duration.
    fixed_time : Quantity
        Nonnegative fixed per-batch duration t0.
    per_request_time : Quantity
        Positive incremental duration treq per request.

    Returns
    -------
    float
        Dimensionless share of the ceiling throughput 1/treq.

    Notes
    -----
    Equation: share = B treq / (t0 + B treq).

    Assumptions: The same affine service model holds across batch sizes.

    Excludes: Queue formation, memory-capacity limits, and changes in per-request
    execution efficiency.

    References
    ----------
    Derivation: divide B/(t0+B treq) by its large-batch limit 1/treq.

    Examples
    --------
    >>> calc_batch_throughput_share(2, 4 * ureg.ms, 2 * ureg.ms) == 0.5
    True
    """
    work = batch_size * per_request_time
    return (work / (fixed_time + work)).to("").magnitude


def calc_padding_waste_fraction(lengths) -> float:
    """Calculate idle slot share when a batch waits for its longest member.

    Parameters
    ----------
    lengths : iterable of float
            Nonempty sequence of nonnegative work lengths in common units, with a
        positive maximum.

    Returns
    -------
    float
        Dimensionless fraction of reserved slot-time unused.

    Notes
    -----
    Equation: waste = 1 - sum(L)/(n max(L)).

    Assumptions: Each member reserves one slot until the longest member finishes, and
    length is proportional to slot occupancy.

    Excludes: Continuous replacement, variable work per token, and non-slot resource
    bottlenecks.

    References
    ----------
    Derivation: useful slot-time is sum(L), while fixed-membership reservation
    consumes n max(L). Divide the unused difference by the reservation.

    Examples
    --------
    >>> calc_padding_waste_fraction([2, 2]) == 0
    True
    >>> calc_padding_waste_fraction([1, 2]) > 0
    True
    """
    lengths = [float(x) for x in lengths]
    if not lengths or max(lengths) <= 0:
        raise ValueError("lengths must be nonempty and positive")
    return 1 - (sum(lengths) / len(lengths)) / max(lengths)


def calc_speculative_expected_tokens(acceptance: float, draft_length: int) -> float:
    """Calculate expected emitted tokens per speculative round.

    Parameters
    ----------
    acceptance : float
            Independent per-token acceptance probability alpha in [0, 1); the alpha=1
        limit is not accepted by this implementation.
    draft_length : int
        Nonnegative number K of draft proposals; caller must supply an integer count.

    Returns
    -------
    float
        Expected total emitted tokens, including the target bonus or correction token.

    Notes
    -----
    Equation: E[tokens] = ``sum(alpha**i``, i=0..K) = ``(1-alpha**(K+1))/(1-alpha)``.

    Assumptions: Identical independent acceptance probabilities; verification stops at
    the first rejected draft, and the target supplies one additional token.

    Excludes: Context-dependent acceptance, verification compute, draft cost, and
    output quality.

    References
    ----------
    Leviathan, Kalman, and Matias (2023), Fast Inference from Transformers
    via Speculative Decoding. https://proceedings.mlr.press/v202/leviathan23a.html
    (bibliography key: leviathan2023fast).

    Examples
    --------
    >>> calc_speculative_expected_tokens(0, 4) == 1
    True
    >>> calc_speculative_expected_tokens(0.5, 4) > calc_speculative_expected_tokens(0.5, 2)
    True
    """
    a = float(acceptance)
    if not 0 <= a < 1:
        raise ValueError("acceptance must be in [0, 1)")
    return (1 - a ** (draft_length + 1)) / (1 - a)


# --- Residence time and turn rate (added 2026-10-20) --------------------------


def calc_littles_law_rate(occupied_slots, residence_time):
    """Calculate average turnover rate from occupied slots and residence time.

    Parameters
    ----------
    occupied_slots : Quantity or float
        Nonnegative mean count L of occupied slots.
    residence_time : Quantity or float
        Positive mean duration W for which each turn holds a slot.

    Returns
    -------
    Quantity or float
        Average turnover rate L/W, reciprocal to the residence-time unit.

    Notes
    -----
    Equation: lambda = L/W.

    Assumptions: Stable operation and a common definition of a turn across slot and
    duration measurements.

    Excludes: Peak throughput, tail delays, and idle capacity not represented by
    occupied_slots.

    References
    ----------
    Little (1961), A Proof for the Queuing Formula: L = lambda W.
    https://doi.org/10.1287/opre.9.3.383 (bibliography key: little1961).

    Examples
    --------
    >>> calc_littles_law_rate(6, 3 * ureg.second) == 2 / ureg.second
    True
    """
    return occupied_slots / residence_time


def calc_residency_factor(generation_time, wait_time) -> float:
    """Calculate how retaining state during a wait expands residence time.

    Parameters
    ----------
    generation_time : Quantity
        Positive active generation duration.
    wait_time : Quantity
        Nonnegative additional duration for which state retains its slot.

    Returns
    -------
    float
            Dimensionless residence multiplier; fixed-slot turnover is divided by this
        factor.

    Notes
    -----
    Equation: factor = (Tgen + Twait)/Tgen.

    Assumptions: Generation and waiting are sequential, and both reserve the same slot.

    Excludes: Releasing state during waits, overlapping work, and different memory
    sizes across phases.

    References
    ----------
    Derivation: compare total reserved duration Tgen+Twait with generation-only
    duration Tgen; Little's law converts this to the inverse turnover ratio.

    Examples
    --------
    >>> calc_residency_factor(2 * ureg.second, 0 * ureg.second) == 1
    True
    """
    return ((generation_time + wait_time) / generation_time).to("").magnitude


# --- One-server queueing and batch formation (added 2026-10-20) ---------------


def calc_mm1_time_in_system(service_time, utilization):
    """Calculate mean M/M/1 residence time including service.

    Parameters
    ----------
    service_time : Quantity or float
        Positive mean service duration S; output preserves its unit.
    utilization : float
        Dimensionless busy fraction rho in [0, 1).

    Returns
    -------
    Quantity or float
        Mean total residence time, in the service-time unit.

    Notes
    -----
    Equation: W = S/(1-rho).

    Assumptions: One first-come server, Poisson arrivals, independent exponential
    service, and steady state.

    Excludes: Finite buffer loss, nonexponential service, and tail percentiles.

    References
    ----------
    Harchol-Balter (2013), Performance Modeling and Design of Computer
    Systems: Queueing Theory in Action.
    https://www.cs.cmu.edu/~harchol/PerformanceModeling/book.html
    (bibliography key: harchol-balter2013).

    Examples
    --------
    >>> calc_mm1_time_in_system(2 * ureg.ms, 0) == 2 * ureg.ms
    True
    """
    rho = float(utilization)
    if not 0 <= rho < 1:
        raise ValueError("utilization must be in [0, 1)")
    return service_time / (1 - rho)


def calc_exponential_quantile_factor(quantile: float) -> float:
    """Calculate an exponential quantile relative to its mean.

    Parameters
    ----------
    quantile : float
        Cumulative probability q in [0, 1).

    Returns
    -------
    float
        Dimensionless quantile-to-mean ratio.

    Notes
    -----
    Equation: kq = -log(1-q).

    Assumptions: An exponential distribution with positive finite mean.

    Excludes: Other service-time distributions and mixtures with a point mass at zero.

    References
    ----------
    Derivation: invert F(t)=1-exp(-t/mean) at F(t)=q, then divide by mean.

    Examples
    --------
    >>> calc_exponential_quantile_factor(0) == 0
    True
    >>> calc_exponential_quantile_factor(0.99) > calc_exponential_quantile_factor(0.5)
    True
    """
    q = float(quantile)
    if not 0 <= q < 1:
        raise ValueError("quantile must be in [0, 1)")
    return math.log(1 / (1 - q))


def calc_mm1_max_utilization(service_time, tail_budget, tail_factor: float) -> float:
    """Calculate the utilization ceiling implied by an M/M/1 tail budget.

    Parameters
    ----------
    service_time : Quantity
        Positive mean service duration S.
    tail_budget : Quantity
        Positive allowed tail duration T, in compatible units.
    tail_factor : float
        Positive tail-to-mean multiplier k, normally from the exponential quantile.

    Returns
    -------
    float
            Unclamped dimensionless ceiling; a negative result means even idle service
        misses the budget.

    Notes
    -----
    Equation: rho_max = 1 - k S/T.

    Assumptions: M/M/1 response time is exponential, and the tail factor matches the
    chosen quantile.

    Excludes: Admission enforcement, finite buffers, or nonexponential latency tails.

    References
    ----------
    Derivation: rearrange k S/(1-rho) <= T for rho. The underlying M/M/1
    mean is documented in calc_mm1_time_in_system.

    Examples
    --------
    >>> calc_mm1_max_utilization(2 * ureg.ms, 2 * ureg.ms, 1) == 0
    True
    """
    return 1 - (tail_factor * service_time / tail_budget).to("").magnitude


def calc_batch_formation_wait(batch_size: int, arrival_rate):
    """Calculate mean delay while a fixed-size batch fills.

    Parameters
    ----------
    batch_size : int
        Positive batch count B; no timeout or partial-batch release.
    arrival_rate : Quantity or float
        Positive stationary arrival rate lambda; output has reciprocal-rate units.

    Returns
    -------
    Quantity or float
        Mean formation delay, excluding execution and downstream queueing.

    Notes
    -----
    Equation: Wform = (B-1)/(2 lambda).

    Assumptions: Average interarrival gap is 1/lambda and each request position within
    a batch is equally represented.

    Excludes: Arrival bursts correlated with batch position, timeouts, priorities, and
    execution delay.

    References
    ----------
    Derivation: batch positions wait for B-1, B-2, ..., 0 subsequent arrivals;
    averaging those counts and multiplying by the mean gap gives the formula.

    Examples
    --------
    >>> calc_batch_formation_wait(1, 2 / ureg.second) == 0 * ureg.second
    True
    """
    return (batch_size - 1) / (2 * arrival_rate)


# --- Agent serving economics: queue price and speculative rounds (added 2026-10-20) ---


def calc_kingman_wait_ratio(utilization: float, arrival_scv: float, service_scv: float) -> float:
    """Approximate mean G/G/1 wait relative to mean service time.

    Parameters
    ----------
    utilization : float
        Busy fraction rho in [0, 1).
    arrival_scv : float
        Nonnegative squared coefficient of variation of interarrival gaps.
    service_scv : float
        Nonnegative squared coefficient of variation of service durations.

    Returns
    -------
    float
        Dimensionless approximate mean-wait to mean-service ratio.

    Notes
    -----
    Equation: Wq/E[S] approximately ``((Ca**2`` + ``Cs**2)/2)`` rho/(1-rho).

    Assumptions: A single first-come server with independent renewal arrivals and
    service durations; the approximation is motivated by heavy traffic.

    Excludes: Tail percentiles, correlated service/arrivals, finite buffers, and
    multiple-server routing.

    References
    ----------
    Kingman (1961), The Single Server Queue in Heavy Traffic.
    https://doi.org/10.1017/S0305004100036094
    (bibliography key: kingman1961single).

    Examples
    --------
    >>> calc_kingman_wait_ratio(0.5, 1, 1) == calc_mg1_wait_ratio(0.5, 1)
    True
    """
    rho = float(utilization)
    if not 0 <= rho < 1:
        raise ValueError("utilization must be in [0, 1)")
    if arrival_scv < 0 or service_scv < 0:
        raise ValueError("squared coefficients of variation must be nonnegative")
    variability = (arrival_scv + service_scv) / 2
    return variability * rho / (1 - rho)


def calc_speculative_round_speedup(accepted_per_round: float, draft_length: int, draft_cost_ratio: float) -> float:
    """Calculate a speculative-round decode speedup under fixed step costs.

    Parameters
    ----------
    accepted_per_round : float
        Nonnegative mean number m of accepted draft tokens, excluding the target token.
    draft_length : int
        Nonnegative number gamma of draft steps per round.
    draft_cost_ratio : float
        Nonnegative draft-step duration divided by one target-step duration.

    Returns
    -------
    float
        Dimensionless time-per-token speedup against serial target-only decoding.

    Notes
    -----
    Equation: speedup = (m+1)/(gamma r + 1).

    Assumptions: Draft steps execute serially and one target verification costs one
    target step regardless of gamma; one target token is always emitted.

    Excludes: Verification compute growth, KV-cache memory, communication, and changing
    context or acceptance.

    References
    ----------
    Leviathan, Kalman, and Matias (2023), Fast Inference from Transformers
    via Speculative Decoding. https://proceedings.mlr.press/v202/leviathan23a.html
    (bibliography key: leviathan2023fast).

    Examples
    --------
    >>> calc_speculative_round_speedup(0, 0, 0) == 1
    True
    >>> calc_speculative_round_speedup(1, 4, 1) < 1
    True
    """
    if accepted_per_round < 0 or draft_length < 0 or draft_cost_ratio < 0:
        raise ValueError("inputs must be nonnegative")
    return (accepted_per_round + 1) / (draft_length * draft_cost_ratio + 1)
