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


def calc_partition_elapsed(workers: int, parallel_fraction: float, overhead_per_extra_worker: float) -> float:
    """
    Elapsed time of a fixed partition, as a fraction of the serial reference.

        T(N) / T(1) = (1 - p) + p / N + c (N - 1)

    ``p`` is the share of the work that divides among ``N`` workers and ``c``
    the coordination overhead each added worker costs, both normalized by the
    serial time. With ``sigma = 1 - p`` and ``kappa = c`` this is the reciprocal
    of the universal scalability law's capacity.
    """
    return (1 - parallel_fraction) + parallel_fraction / workers + overhead_per_extra_worker * (workers - 1)


def calc_full_mesh_deliveries(workers: int) -> int:
    """Directed deliveries when each of ``N`` participants sends to every other once: N (N - 1)."""
    return workers * (workers - 1)


def calc_supervisor_deliveries(workers: int) -> int:
    """Deliveries when a supervisor sends one envelope to and receives one receipt from each worker: 2 N."""
    return 2 * workers


def calc_earliest_schedule(durations, predecessors):
    """
    Earliest start and finish of every node of an acyclic dependency graph with unlimited slots.

    ``durations`` maps each node to its time; ``predecessors`` maps it to the
    nodes that must finish first. A node starts when its last predecessor
    finishes. Returns ``(start, finish)`` dictionaries in the order of
    ``durations``; the makespan is the latest finish, the critical path.
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
    """
    Probability that ``N`` write sets of ``w`` files each, drawn uniformly from ``F``, are pairwise disjoint.

        prod_{k=1}^{N-1} C(F - k w, w) / C(F, w)

    Worker ``k + 1`` must avoid the ``k w`` files already written.
    """
    disjoint = 1.0
    for earlier in range(1, workers):
        disjoint *= math.comb(files - earlier * writes_per_worker, writes_per_worker) / math.comb(files, writes_per_worker)
    return disjoint


def calc_disjoint_writes_approx(workers: int, files: int, writes_per_worker: int) -> float:
    """The birthday approximation of :func:`calc_disjoint_writes`: exp(-N (N - 1) w^2 / (2 F))."""
    return math.exp(-workers * (workers - 1) * writes_per_worker ** 2 / (2 * files))


def calc_binomial_tail(trials: int, probability: float, at_least: int) -> float:
    """Probability of at least ``at_least`` successes in ``trials`` independent trials."""
    return sum(math.comb(trials, k) * probability ** k * (1 - probability) ** (trials - k)
               for k in range(at_least, trials + 1))


def calc_beta_mixture_shape(mean: float, correlation: float):
    """
    Beta shape ``(a, b)`` with the given mean and pairwise (intraclass) correlation.

        a = mean (1/rho - 1),  b = (1 - mean)(1/rho - 1)
    """
    scale = 1 / correlation - 1
    return mean * scale, (1 - mean) * scale


def calc_beta_binomial_pmf(trials: int, mean: float, correlation: float) -> list:
    """
    Beta-binomial masses for 0 to ``trials`` successes.

    Each trial succeeds with a propensity drawn once from a beta law of the
    given mean and pairwise correlation; the masses use log gamma functions.
    """
    a, b = calc_beta_mixture_shape(mean, correlation)
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) - math.lgamma(trials + a + b)
    return [math.comb(trials, k) * math.exp(math.lgamma(k + a) + math.lgamma(trials - k + b) + log_norm)
            for k in range(trials + 1)]


def calc_beta_tail(mean: float, correlation: float, threshold: float) -> float:
    """
    Mass of the beta law (given mean and correlation) above ``threshold``.

    As a correlated panel grows, the share of members that err converges to
    the item's propensity, so a majority errs on exactly the items whose
    propensity exceeds one half: this mass is the floor under majority error.
    """
    from scipy.special import betainc

    a, b = calc_beta_mixture_shape(mean, correlation)
    return 1 - float(betainc(a, b, threshold))
