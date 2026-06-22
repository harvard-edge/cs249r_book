"""Collective communication time models (α–β).

All collectives here are pure communication models: the local reduction
compute term (γ in Thakur et al. 2005) is deliberately omitted, matching the
book's α–β pedagogical treatment.
"""

from __future__ import annotations

import math

from mlsysim.core.units import ureg, Q_
from mlsysim.core._validation import (
    validate_positive,
    validate_nonnegative,
    validate_range,
    validate_at_least,
)

from ._units import _ensure_unit


def calc_ring_allreduce_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s):
    """
    Calculates communication time for a Ring AllReduce collective.

    The Ring AllReduce algorithm operates in two phases (Reduce-Scatter and
    All-Gather). Each phase takes (N-1) steps. At each step, a chunk of size
    M/N is transmitted. Therefore, the total volume transmitted by any single 
    node is 2*(N-1)/N * M.

    Parameters
    ----------
    message_bytes : Quantity
        Total message size (M) to be reduced (e.g., model gradients).
    n_gpus : int
        Number of participating GPUs (N) in the ring.
    bandwidth_bytes_s : Quantity
        Link bandwidth between nodes in the ring (β).
    latency_s : Quantity
        Per-hop network latency (α).

    Returns
    -------
    Quantity
        Total time required for the collective operation in seconds.
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return Q_("0 second")
    msg = _ensure_unit(message_bytes, ureg.byte, "message_bytes")
    validate_nonnegative(msg, "message_bytes")
    bw = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s")
    validate_positive(bw, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
    validate_nonnegative(lat, "latency_s")
    n = n_gpus
    bw_term = 2 * (n - 1) / n * msg / bw
    lat_term = 2 * (n - 1) * lat
    return (bw_term + lat_term).to(ureg.second)


def ring_allreduce_data_factor_latex(n_gpus_symbol="N"):
    """LaTeX form for the Ring AllReduce data factor, e.g. ``2 \times (N-1)/N``."""
    return f"2 \\times ({n_gpus_symbol}-1)/{n_gpus_symbol}"


def calc_ring_allreduce_data_factor(n_gpus):
    """Ring AllReduce bandwidth factor: 2*(N-1)/N.

    This multiplier appears in the bandwidth term of the ring AllReduce
    alpha-beta model:

    ``T_ring = 2*(N-1)/N * M/beta + 2*(N-1)*alpha``
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return 0.0
    return 2 * (n_gpus - 1) / n_gpus


def calc_ring_collective_data_factor(n_gpus):
    """Ring single-stage collective factor: (N-1)/N.

    This multiplier appears in ring AllGather/ReduceScatter/AllToAll bandwidth terms.
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return 0.0
    return (n_gpus - 1) / n_gpus


def calc_ring_allreduce_latency_steps(n_gpus):
    """Number of startup latency terms in ring AllReduce: 2(N-1)."""
    validate_at_least(n_gpus, 1, "n_gpus")
    return 2 * (n_gpus - 1)


def calc_ring_allreduce_latency_time(n_gpus, latency_s):
    """Ring AllReduce latency term only: 2(N-1) × alpha."""
    validate_at_least(n_gpus, 1, "n_gpus")
    alpha = _ensure_unit(latency_s, ureg.second, "latency_s")
    validate_nonnegative(alpha, "latency_s")
    return (calc_ring_allreduce_latency_steps(n_gpus) * alpha).to(ureg.second)


def calc_alpha_beta_crossover(alpha_s, bandwidth_bytes_s):
    """Crossover size for the α-β model where α = n/β."""

    alpha = _ensure_unit(alpha_s, ureg.second, "alpha_s")
    bandwidth = _ensure_unit(
        bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s"
    )
    validate_nonnegative(alpha, "alpha_s")
    validate_positive(bandwidth, "bandwidth_bytes_s")
    return (alpha * bandwidth).to(ureg.byte)


def calc_point_to_point_time(message_bytes, alpha_s, bandwidth_bytes_s):
    """Point-to-point transfer time under the α-β model: α + n/β."""

    alpha = _ensure_unit(alpha_s, ureg.second, "alpha_s")
    bandwidth = _ensure_unit(
        bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s"
    )
    message = _ensure_unit(message_bytes, ureg.byte, "message_bytes")
    validate_nonnegative(alpha, "alpha_s")
    validate_positive(bandwidth, "bandwidth_bytes_s")
    validate_nonnegative(message, "message_bytes")
    return (alpha + message / bandwidth).to(ureg.second)


def calc_oversubscription_effect(comm_fraction, oversubscription_ratio):
    """Compute throughput penalty from oversubscription.

    For an idealized workload with baseline step time T_base, oversubscribing
    the fabric stretches only the communication fraction::

        T_new = T_base * [(1 - f) + f * r]
        relative_throughput = T_base / T_new = 1 / ((1 - f) + f * r)

    where:
    - ``f`` = communication fraction of baseline step time
    - ``r`` = N:1 oversubscription ratio (e.g., 2.0, 4.0)

    Returns
    -------
    tuple
        ``(relative_throughput, throughput_loss)`` where both are dimensionless
        fractions in [0,1].
    """
    validate_range(comm_fraction, 0.0, 1.0, "comm_fraction")
    validate_positive(oversubscription_ratio, "oversubscription_ratio")

    relative_throughput = 1 / ((1 - comm_fraction) + (comm_fraction * oversubscription_ratio))
    throughput_loss = 1 - relative_throughput
    return relative_throughput, throughput_loss


def calc_bisection_bandwidth(num_ports, link_bandwidth, oversubscription_ratio=1.0):
    """Bisection bandwidth for a single partition cut.

    Uses ``BW_bisect = num_ports * link_bw / oversubscription_ratio``.

    ``num_ports`` is the number of links CROSSING the bisection cut — N/2 for
    a non-blocking fabric of N endpoints — not the total port count of the
    fabric. Passing total ports silently overstates bisection bandwidth 2x.
    """
    validate_at_least(num_ports, 1, "num_ports")
    validate_positive(oversubscription_ratio, "oversubscription_ratio")

    link_bw = _ensure_unit(
        link_bandwidth, ureg.byte / ureg.second, "link_bandwidth"
    )
    validate_positive(link_bw, "link_bandwidth")
    return (num_ports * link_bw / oversubscription_ratio).to(ureg.byte / ureg.second)


def calc_hop_latency(hops, hop_latency_s):
    """Total latency across ``hops`` identical hops."""
    validate_nonnegative(hops, "hops")
    latency = _ensure_unit(hop_latency_s, ureg.second, "hop_latency_s")
    validate_nonnegative(latency, "hop_latency_s")
    return (hops * latency).to(ureg.second)


def calc_tree_allreduce_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s):
    """
    Calculates communication time for a Tree AllReduce collective.

    Tree AllReduce operates in O(log N) steps. It is often used in hierarchical
    topologies where latency dominates over bandwidth.

    Parameters
    ----------
    message_bytes : Quantity
        Total message size (M) to be reduced.
    n_gpus : int
        Number of participating GPUs (N) in the tree.
    bandwidth_bytes_s : Quantity
        Link bandwidth between nodes (β).
    latency_s : Quantity
        Per-hop network latency (α).

    Returns
    -------
    Quantity
        Total time required for the collective operation in seconds.
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return Q_("0 second")
    msg = _ensure_unit(message_bytes, ureg.byte, "message_bytes")
    validate_nonnegative(msg, "message_bytes")
    bw = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s")
    validate_positive(bw, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
    validate_nonnegative(lat, "latency_s")
    log_n = math.ceil(math.log2(max(n_gpus, 2)))
    bw_term = 2 * log_n * msg / bw
    lat_term = 2 * log_n * lat
    return (bw_term + lat_term).to(ureg.second)


def calc_double_binary_tree_allreduce_time(
    message_bytes,
    n_gpus,
    bandwidth_bytes_s,
    latency_s,
    latency_factor=1.2,
    bandwidth_factor=1.05,
):
    """
    Double Binary Tree AllReduce approximation.

    The model keeps logarithmic startup cost while using near-bandwidth-optimal
    behavior (NCCL 2.4 double binary trees: "full bandwidth and logarithmic
    latency"). The default 1.2/1.05 factors are package-chosen pedagogical
    constants, not published measurements — they encode "slightly worse than
    ideal log latency, slightly above ring-optimal bandwidth cost." Note the
    latency term uses fractional ``log2(N)`` (a smooth approximation), unlike
    ``calc_tree_allreduce_time`` which counts discrete ``ceil(log2 N)`` steps.

    Parameters
    ----------
    message_bytes : Quantity
        Total message size (M) to be reduced.
    n_gpus : int
        Number of participating GPUs (N).
    bandwidth_bytes_s : Quantity
        Link bandwidth between nodes (beta).
    latency_s : Quantity
        Per-hop network latency (alpha).
    latency_factor : float, optional
        Multiplier applied to the pure logarithmic latency term (default 1.2).
    bandwidth_factor : float, optional
        Multiplier applied to the near-bandwidth term (default 1.05).

    Returns
    -------
    Quantity
        Total time for the approximate Double Binary Tree collective in seconds.
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return Q_("0 second")

    msg = _ensure_unit(message_bytes, ureg.byte, "message_bytes")
    bw = _ensure_unit(
        bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s"
    )
    validate_positive(bw, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
    validate_nonnegative(lat, "latency_s")
    if latency_factor < 0:
        raise ValueError(f"latency_factor must be non-negative, got {latency_factor}")
    if bandwidth_factor < 0:
        raise ValueError(
            f"bandwidth_factor must be non-negative, got {bandwidth_factor}"
        )

    log_n = math.log2(max(n_gpus, 2))
    lat_term = latency_factor * (2 * log_n) * lat
    bw_term = bandwidth_factor * (2 * (n_gpus - 1) / n_gpus) * msg / bw
    return (lat_term + bw_term).to(ureg.second)


def calc_ring_tree_crossover_size(n_gpus, alpha_s, bandwidth_bytes_s):
    """
    Crossover size where the approximate ring/tree model costs intersect.

    Uses the log-aware estimate:
      M_crossover = N * alpha * beta / log2(N)

    This is a large-N estimate of the exact intersection of this package's
    ring and tree cost models: good to ~10% for N >= 32, overestimates ~40%
    at N = 8, and is undefined at N = 2 (where the ring model dominates for
    every message size, so no crossover exists).
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return Q_("0 byte")
    alpha = _ensure_unit(alpha_s, ureg.second, "alpha_s")
    validate_nonnegative(alpha, "alpha_s")
    bandwidth = _ensure_unit(
        bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s"
    )
    validate_positive(bandwidth, "bandwidth_bytes_s")
    log_n = math.log2(n_gpus)
    return (n_gpus * alpha * bandwidth / log_n).to(ureg.byte)


def calc_all_to_all_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s):
    """
    Calculates communication time for an All-to-All collective.

    In an All-to-All operation (used heavily in Mixture of Experts routing), 
    every node sends a distinct message to every other node.

    Parameters
    ----------
    message_bytes : Quantity
        Per-node buffer size (M) held by each participant; the ring exchanges
        (N-1)/N of it per node.
    n_gpus : int
        Number of participating GPUs (N).
    bandwidth_bytes_s : Quantity
        Link bandwidth between nodes (β).
    latency_s : Quantity
        Per-hop network latency (α).

    Returns
    -------
    Quantity
        Total time required for the collective operation in seconds.
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    msg = _ensure_unit(message_bytes, ureg.byte, "message_bytes")
    validate_nonnegative(msg, "message_bytes")
    bw = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s")
    validate_positive(bw, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
    validate_nonnegative(lat, "latency_s")
    n = n_gpus
    bw_term = (n - 1) / n * msg / bw
    lat_term = (n - 1) * lat
    return (bw_term + lat_term).to(ureg.second)


def calc_hierarchical_allreduce_time(
    message_bytes,
    n_nodes,
    gpus_per_node,
    intra_node_bw,
    inter_node_bw,
    intra_node_lat=Q_("500 ns"),
    inter_node_lat=Q_("5 us"),
):
    """
    Calculates communication time for a Hierarchical AllReduce collective.

    This models the standard topology of modern GPU clusters (e.g., NVLink 
    within a node, InfiniBand across nodes). It operates in three phases:
    1. Reduce-Scatter locally within each node (using fast intra-node BW).
    2. AllReduce the smaller, partially reduced chunks globally across nodes 
       (using slower inter-node BW).
    3. AllGather the fully reduced chunks locally within each node.

    Parameters
    ----------
    message_bytes : Quantity
        Total message size (M) to be reduced.
    n_nodes : int
        Number of physical server nodes.
    gpus_per_node : int
        Number of accelerators per node.
    intra_node_bw : Quantity
        Bandwidth between GPUs on the same node (e.g., NVLink).
    inter_node_bw : Quantity
        PER-GPU NIC bandwidth between nodes (e.g., one InfiniBand HCA port).
        The model assumes one NIC per GPU (DGX-style), so the g concurrent
        inter-node rings in phase 2 do not share a link. With a single NIC
        shared by all g local ranks, the inter phase costs
        ``2(n-1)/n * M / beta`` — g times this model's phase-2 term — and
        this function understates the total accordingly.
    intra_node_lat : Quantity, optional
        Latency within the node. Defaults to 500 ns.
    inter_node_lat : Quantity, optional
        Latency across nodes. Defaults to 5 us.

    Returns
    -------
    Quantity
        Total time required for the hierarchical operation in seconds.
    """
    validate_at_least(gpus_per_node, 1, "gpus_per_node")
    msg = _ensure_unit(message_bytes, ureg.byte, "message_bytes")
    intra_bw = _ensure_unit(intra_node_bw, ureg.byte / ureg.second, "intra_node_bw")
    validate_positive(intra_bw, "intra_node_bw")
    intra_lat = _ensure_unit(intra_node_lat, ureg.second, "intra_node_lat")
    g = gpus_per_node

    # Phase 1 — intra-node Reduce-Scatter: each of the g ranks ends up owning a
    # fully reduced 1/g chunk. Ring cost: (g-1)/g * M/beta + (g-1) * alpha.
    t_reduce_scatter = (g - 1) / g * msg / intra_bw + (g - 1) * intra_lat

    # Phase 2 — inter-node ring AllReduce over the M/g chunk each node owns.
    t_allreduce_inter = calc_ring_allreduce_time(
        msg / g, n_nodes, inter_node_bw, inter_node_lat
    )

    # Phase 3 — intra-node AllGather: redistribute the reduced chunks so every
    # rank holds the full result. Same ring cost as the Reduce-Scatter.
    t_all_gather = (g - 1) / g * msg / intra_bw + (g - 1) * intra_lat

    # (Audit fix 2026-06-06: the previous implementation ran a FULL intra-node
    # AllReduce of M — already equal to RS+AG combined — plus a second intra-node
    # AllReduce of M/g labeled "broadcast", inflating the intra-node term by
    # (1 + 1/g) and contradicting the 3-phase algorithm documented above.)
    return (t_reduce_scatter + t_allreduce_inter + t_all_gather).to(ureg.second)
