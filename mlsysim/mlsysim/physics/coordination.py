"""Coordination of several agents: partitions, schedules, shared writes, and correlated votes.

Added 2026-10-20. Each function is the arithmetic of one coordination
relation: the elapsed time of a work partition with a per-worker overhead, the
deliveries of a full mesh against a supervisor, the earliest schedule of a
dependency graph, the chance that independent write sets stay disjoint, the
binomial and beta-binomial tails of a vote, and the beta mass above a
threshold that floors a correlated majority.
"""

from __future__ import annotations

import math
from numbers import Integral


def calc_partition_elapsed(workers: int, parallel_fraction: float, overhead_per_extra_worker: float) -> float:
    """Estimate partition elapsed time relative to serial execution.

    Parameters
    ----------
    workers : int
        Number N of workers, at least one.
    parallel_fraction : float
        Fraction p of serial work that divides evenly, between zero and one.
    overhead_per_extra_worker : float
        Nonnegative coordination cost c per added worker, divided by serial time.

    Returns
    -------
    float
        Dimensionless elapsed-time ratio T(N) / T(1).

    Notes
    -----
    Equation: T(N)/T(1) = (1-p) + p/N + c(N-1). The serial remainder,
    parallel work, and coordination cost add. With sigma = 1-p and kappa = c,
    this is the reciprocal of universal-scalability-law capacity.

    Assumptions: fixed total work, equal partitions, and linear added-worker
    coordination cost. Input ranges are caller preconditions.

    Excludes: load imbalance, communication overlap, and worker heterogeneity.
    The overhead coefficient is an input assumption, not a fitted prediction.

    References
    ----------
    Derivation: split normalized serial work into a serial share 1-p and
    N equal shares of p; add the stipulated overhead for N-1 added workers.

    Examples
    --------
    >>> calc_partition_elapsed(2, 1.0, 0.0)
    0.5
    """
    return (1 - parallel_fraction) + parallel_fraction / workers + overhead_per_extra_worker * (workers - 1)


def calc_full_mesh_deliveries(workers: int) -> int:
    """Count directed deliveries in one full-mesh exchange.

    Parameters
    ----------
    workers : int
        Nonnegative number N of participants.

    Returns
    -------
    int
        Number of directed message deliveries, excluding self-deliveries.

    Notes
    -----
    Equation: deliveries = N(N-1).

    Assumptions: every participant sends exactly one message to every other
    participant. The caller supplies a nonnegative count.

    Excludes: acknowledgments, retries, broadcasts, and message byte sizes.

    References
    ----------
    Derivation: each of N senders has N-1 distinct recipients; count ordered
    sender-recipient pairs.

    Examples
    --------
    >>> calc_full_mesh_deliveries(3)
    6
    """
    return workers * (workers - 1)


def calc_supervisor_deliveries(workers: int) -> int:
    """Count deliveries in one supervisor request-and-receipt round.

    Parameters
    ----------
    workers : int
        Nonnegative number N of workers, excluding the supervisor.

    Returns
    -------
    int
        Count of request and receipt deliveries.

    Notes
    -----
    Equation: deliveries = 2N.

    Assumptions: the supervisor sends one envelope to each worker and receives
    one receipt from each. The caller supplies a nonnegative count.

    Excludes: worker-to-worker messages, retries, and message byte sizes.

    References
    ----------
    Derivation: add N outbound envelopes and N inbound receipts.

    Examples
    --------
    >>> calc_supervisor_deliveries(3)
    6
    """
    return 2 * workers


def calc_earliest_schedule(durations, predecessors):
    """Compute earliest starts and finishes for an acyclic dependency graph.

    Parameters
    ----------
    durations : mapping
        Node identifiers mapped to nonnegative durations. All durations must
        use compatible time units; bare numbers share one caller-selected unit.
    predecessors : mapping
        Node identifiers mapped to iterables of predecessor identifiers.
        Every referenced predecessor must occur in durations. Missing entries
        denote nodes without predecessors.

    Returns
    -------
    start : dict
        Earliest start time for each node, in durations insertion order and units.
    finish : dict
        Earliest finish time for each node, in durations insertion order and units.

    Raises
    ------
    ValueError
        If a dependency path contains a cycle.
    KeyError
        If a referenced predecessor is absent from durations.

    Notes
    -----
    Equation: start(v) = max(finish(u) for predecessors u), with zero for a
    source node; finish(v) = start(v) + duration(v). The latest finish is the
    makespan and equals the duration of the critical path.

    Assumptions: unlimited execution slots, deterministic nonnegative durations,
    and all predecessor results available at their finish times.

    Excludes: resource contention, transfer delays, preemption, and scheduling
    policy. This is a dependency-only lower bound for resource-limited execution.

    References
    ----------
    Derivation: a node can begin after its last predecessor finishes. Applying
    that recurrence in dependency order yields the earliest feasible schedule.

    Examples
    --------
    >>> calc_earliest_schedule({'read': 2, 'write': 3}, {'write': ['read']})
    ({'read': 0, 'write': 2}, {'read': 2, 'write': 5})
    """
    start, finish = {}, {}

    def _finish(node, trail=()):
        if node in trail:
            raise ValueError(f"dependency graph has a cycle through {node!r}")
        if node not in finish:
            ready = 0 * durations[node]
            for parent in predecessors.get(node, ()):
                ready = max(ready, _finish(parent, trail + (node,)))
            start[node] = ready
            finish[node] = ready + durations[node]
        return finish[node]

    for node in durations:
        _finish(node)
    return {n: start[n] for n in durations}, {n: finish[n] for n in durations}


def calc_disjoint_writes(workers: int, files: int, writes_per_worker: int) -> float:
    """Calculate the probability that uniform write sets are disjoint.

    Parameters
    ----------
    workers : int
        Number N of independent workers, at least one.
    files : int
        Positive number F of available files.
    writes_per_worker : int
        Number w of distinct files chosen by each worker, with 0 <= w <= F.
        If N*w exceeds F, disjoint placement is impossible and probability is zero.

    Returns
    -------
    float
        Dimensionless probability that all worker write sets are pairwise disjoint.

    Notes
    -----
    Equation: product over k=1,...,N-1 of C(F-kw,w)/C(F,w).
    Worker k+1 must avoid the kw files already selected by earlier workers.

    Assumptions: each worker independently chooses a uniformly distributed
    w-element subset; selections within a worker are without replacement.
    Counts must be integers in the documented ranges.

    Excludes: hot-file skew, coordinated partitioning, and write ordering.

    References
    ----------
    Derivation: condition each successive worker on all earlier write sets
    being disjoint, count allowed subsets, and multiply conditional probabilities.

    Examples
    --------
    >>> calc_disjoint_writes(2, 4, 1)
    0.75
    """
    for name, value, minimum in (("workers", workers, 1), ("files", files, 1),
                                 ("writes_per_worker", writes_per_worker, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if writes_per_worker > files:
        raise ValueError("writes_per_worker cannot exceed files")
    if workers * writes_per_worker > files:
        return 0.0

    disjoint = 1.0
    for earlier in range(1, workers):
        disjoint *= math.comb(files - earlier * writes_per_worker, writes_per_worker) / math.comb(
            files, writes_per_worker
        )
    return disjoint


def calc_disjoint_writes_approx(workers: int, files: int, writes_per_worker: int) -> float:
    """Approximate disjoint write-set probability in the sparse regime.

    Parameters
    ----------
    workers : int
        Number N of workers, at least one.
    files : int
        Positive number F of available files.
    writes_per_worker : int
        Nonnegative number w of distinct files selected per worker, at most F.

    Returns
    -------
    float
        Dimensionless approximate probability of no cross-worker collision.

    Notes
    -----
    Equation: probability approximately ``exp(-N(N-1)w**2/(2F))``.

    Assumptions: independent uniform file selection and sparse collisions,
    so a Poisson approximation to the number of pairwise collisions is useful.

    Excludes: hot-file skew and higher-order dependence among collisions.
    It need not match the exact probability near capacity; use
    calc_disjoint_writes for its supported exact-combinatorial domain.

    References
    ----------
    Derivation: N(N-1)/2 worker pairs each have approximately ``w**2/F`` shared
    files in expectation; approximate the zero-collision probability by the
    zero mass of a Poisson variable with that mean.

    Examples
    --------
    >>> round(calc_disjoint_writes_approx(2, 100, 1), 6)
    0.99005
    """
    return math.exp(-workers * (workers - 1) * writes_per_worker**2 / (2 * files))


def calc_binomial_tail(trials: int, probability: float, at_least: int) -> float:
    """Calculate a binomial upper-tail probability by summing masses.

    Parameters
    ----------
    trials : int
        Nonnegative number n of independent trials.
    probability : float
        Common success probability p, between zero and one.
    at_least : int
        Inclusive success threshold k, from zero through n+1.

    Returns
    -------
    float
        Dimensionless probability of at least k successes.

    Notes
    -----
    Equation: P(X >= k) = sum from j=k to n of C(n,``j)p**j(1-p)**(n-j)``.
    A threshold of n+1 produces an empty sum, which is zero.

    Assumptions: independent Bernoulli trials with a common fixed probability.
    Valid input ranges are caller preconditions.

    Excludes: correlated or unequal-probability trials. Direct summation is
    not a numerically stabilized large-n survival-function implementation.

    References
    ----------
    Derivation: there are C(n,j) success-position subsets for j successes;
    each has probability ``p**j(1-p)**(n-j)``. Sum the disjoint events above k.

    Examples
    --------
    >>> calc_binomial_tail(2, 0.5, 1)
    0.75
    """
    return sum(
        math.comb(trials, k) * probability**k * (1 - probability) ** (trials - k) for k in range(at_least, trials + 1)
    )


def calc_beta_mixture_shape(mean: float, correlation: float):
    """Recover beta shape parameters from Bernoulli mean and correlation.

    Parameters
    ----------
    mean : float
        Marginal success probability mu, greater than zero and less than one.
    correlation : float
        Intraclass correlation rho, greater than zero and less than one.

    Returns
    -------
    a : float
        Positive dimensionless first beta shape parameter.
    b : float
        Positive dimensionless second beta shape parameter.

    Notes
    -----
    Equation: a = mu(1/rho-1) and b = (1-mu)(1/rho-1).

    Assumptions: Bernoulli outcomes share one beta-distributed propensity and
    are independent conditional on that propensity. Input ranges are caller
    preconditions; independent trials require the rho -> 0 limiting model.

    Excludes: negative correlation, deterministic propensities at the domain
    boundaries, and estimation uncertainty in the supplied mean and correlation.

    References
    ----------
    Derivation: a beta mixture has mean a/(a+b) and Bernoulli intraclass
    correlation 1/(a+b+1). Solve these two equations for a and b.

    Examples
    --------
    >>> calc_beta_mixture_shape(0.5, 0.5)
    (0.5, 0.5)
    """
    scale = 1 / correlation - 1
    return mean * scale, (1 - mean) * scale


def calc_beta_binomial_pmf(trials: int, mean: float, correlation: float) -> list:
    """Calculate beta-binomial masses for a correlated Bernoulli panel.

    Parameters
    ----------
    trials : int
        Nonnegative number n of Bernoulli outcomes.
    mean : float
        Marginal success probability, greater than zero and less than one.
    correlation : float
        Pairwise intraclass correlation, greater than zero and less than one.

    Returns
    -------
    list of float
        Dimensionless masses ordered by success count k=0,...,n.

    Notes
    -----
    Equation: P(X=k) = C(n,k) B(k+a,n-k+b)/B(a,b), where a and b come from
    calc_beta_mixture_shape. Beta-function ratios are evaluated with log gamma.

    Assumptions: draw one propensity per panel from the beta law, then sample
    conditionally independent outcomes using that same propensity. Input
    ranges are caller preconditions.

    Excludes: arbitrary dependence structures and uncertainty in fitted mixture
    parameters. Large binomial coefficients can still limit numerical range.

    References
    ----------
    Derivation: integrate the conditional binomial mass against the beta
    density; the resulting integral is B(k+a,n-k+b)/B(a,b).

    Examples
    --------
    >>> [round(p, 6) for p in calc_beta_binomial_pmf(2, 0.5, 0.5)]
    [0.375, 0.25, 0.375]
    """
    a, b = calc_beta_mixture_shape(mean, correlation)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) - math.lgamma(trials + a + b)
    return [
        math.comb(trials, k) * math.exp(math.lgamma(k + a) + math.lgamma(trials - k + b) + log_norm)
        for k in range(trials + 1)
    ]


def calc_beta_tail(mean: float, correlation: float, threshold: float) -> float:
    """Calculate beta propensity mass above a threshold.

    Parameters
    ----------
    mean : float
        Mean propensity, greater than zero and less than one.
    correlation : float
        Bernoulli intraclass correlation, greater than zero and less than one.
    threshold : float
        Propensity threshold between zero and one.

    Returns
    -------
    float
        Dimensionless beta probability mass above the threshold.

    Notes
    -----
    Equation: P(theta > t) = 1 - I_t(a,b), where I is the regularized
    incomplete beta function and shapes come from calc_beta_mixture_shape.

    Assumptions: one beta-distributed propensity is shared by all conditionally
    independent members of a panel. As the panel grows, its success share
    converges to that propensity. If success denotes an error and t=1/2, the
    returned mass is the limiting majority-error probability.

    Excludes: finite-panel voting corrections, arbitrary correlations, and
    parameter-estimation uncertainty. Subtracting from one can lose precision
    in very small upper tails. SciPy is required at call time.

    References
    ----------
    Derivation: integrate the beta density above t, or subtract its cumulative
    mass I_t(a,b) from one. The conditional law of large numbers gives the
    infinite-panel interpretation.

    Examples
    --------
    >>> round(calc_beta_tail(0.5, 0.5, 0.5), 6)
    0.5
    """
    from scipy.special import betainc

    a, b = calc_beta_mixture_shape(mean, correlation)
    return 1 - float(betainc(a, b, threshold))
