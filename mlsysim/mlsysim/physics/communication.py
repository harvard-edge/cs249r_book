"""Collective communication time models (α–β)."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg, Q_
from mlsysim.core._validation import validate_positive, validate_at_least

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
    bw = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s")
    validate_positive(bw, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
    n = n_gpus
    bw_term = 2 * (n - 1) / n * msg / bw
    lat_term = 2 * (n - 1) * lat
    return (bw_term + lat_term).to(ureg.second)


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
    bw = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s")
    validate_positive(bw, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
    log_n = math.ceil(math.log2(max(n_gpus, 2)))
    bw_term = 2 * log_n * msg / bw
    lat_term = 2 * log_n * lat
    return (bw_term + lat_term).to(ureg.second)


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
    bw = _ensure_unit(bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s")
    validate_positive(bw, "bandwidth_bytes_s")
    lat = _ensure_unit(latency_s, ureg.second, "latency_s")
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
        Bandwidth between nodes (e.g., InfiniBand / Ethernet).
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
