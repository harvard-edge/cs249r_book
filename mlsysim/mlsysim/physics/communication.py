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
    """Estimate ring AllReduce communication time.

    Parameters
    ----------
    message_bytes : Quantity or float
        Nonnegative message size M; bare values are bytes.
    n_gpus : int
        Participant count N, at least one; intended to be integral.
    bandwidth_bytes_s : Quantity or float
        Positive unidirectional bandwidth B; bare values are bytes/second.
    latency_s : Quantity or float
        Nonnegative startup latency alpha; bare values are seconds.

    Returns
    -------
    Quantity
        Collective time in seconds; zero for one participant.

    Notes
    -----
    Equation: T = 2(N - 1) M / (N B) + 2(N - 1) alpha.

    Assumptions: Equal chunks and identical links; ReduceScatter followed by AllGather.

    Excludes: Local reduction compute, congestion, overlap with computation, and
    protocol costs.

    Bandwidth is one direction, not the sum of both directions of a full-duplex link.

    References
    ----------
    Derivation: Each of the two phases sends M/N bytes in each of N - 1 rounds; sum
    serialization and startup costs.

    Examples
    --------
    >>> calc_ring_allreduce_time(1024, 1, 1024, 0).m_as("s") == 0
    True
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
    """Format the ring AllReduce data factor as LaTeX.

    Parameters
    ----------
    n_gpus_symbol : str, optional
        Symbol substituted for participant count; default N.

    Returns
    -------
    str
        LaTeX expression for the dimensionless traffic multiplier.

    Notes
    -----
    Equation: The displayed expression is 2(N - 1)/N.

    Assumptions: The supplied symbol is already suitable for LaTeX.

    Excludes: Numeric evaluation and LaTeX escaping of arbitrary input.

    References
    ----------
    Derivation: ReduceScatter and AllGather each move (N - 1)/N message equivalents per
    rank.

    Examples
    --------
    >>> "P" in ring_allreduce_data_factor_latex("P")
    True
    """
    return f"2 \\times ({n_gpus_symbol}-1)/{n_gpus_symbol}"


def calc_ring_allreduce_data_factor(n_gpus):
    """Return per-rank ring AllReduce traffic in message equivalents.

    Parameters
    ----------
    n_gpus : int
        Participant count N, at least one; intended to be integral.

    Returns
    -------
    float
        Dimensionless total traffic multiplier; zero for N = 1.

    Notes
    -----
    Equation: f = 2(N - 1)/N.

    Assumptions: Equal chunks with two complete ring phases.

    Excludes: Startup latency and local reduction work.

    References
    ----------
    Derivation: Each phase moves N - 1 chunks of size M/N; divide two-phase traffic by
    M.

    Examples
    --------
    >>> calc_ring_allreduce_data_factor(4) == 2 * calc_ring_collective_data_factor(4)
    True
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return 0.0
    return 2 * (n_gpus - 1) / n_gpus


def calc_ring_collective_data_factor(n_gpus):
    """Return a single ring-phase traffic multiplier.

    Parameters
    ----------
    n_gpus : int
        Participant count N, at least one; intended to be integral.

    Returns
    -------
    float
        Dimensionless fraction of a per-rank message transferred.

    Notes
    -----
    Equation: f = (N - 1)/N.

    Assumptions: One ring AllGather, ReduceScatter, or equal-partition exchange phase.

    Excludes: Startup latency and repeated phases.

    References
    ----------
    Derivation: The rank transfers N - 1 of the N equal message chunks.

    Examples
    --------
    >>> calc_ring_collective_data_factor(1) == 0
    True
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    if n_gpus == 1:
        return 0.0
    return (n_gpus - 1) / n_gpus


def calc_ring_allreduce_latency_steps(n_gpus):
    """Count startup terms in ring AllReduce.

    Parameters
    ----------
    n_gpus : int
        Participant count N, at least one; intended to be integral.

    Returns
    -------
    int
        Startup count for integral participant input.

    Notes
    -----
    Equation: s = 2(N - 1).

    Assumptions: Two sequential ring phases with N - 1 rounds each.

    Excludes: Pipelining across rounds and per-byte transmission time.

    References
    ----------
    Derivation: Add the ReduceScatter and AllGather round counts.

    Examples
    --------
    >>> calc_ring_allreduce_latency_steps(4) == 2 * (4 - 1)
    True
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    return 2 * (n_gpus - 1)


def calc_ring_allreduce_latency_time(n_gpus, latency_s):
    """Compute the startup-only component of ring AllReduce.

    Parameters
    ----------
    n_gpus : int
        Participant count N, at least one; intended to be integral.
    latency_s : Quantity or float
        Nonnegative startup latency alpha; bare values are seconds.

    Returns
    -------
    Quantity
        Aggregate startup time in seconds.

    Notes
    -----
    Equation: T_startup = 2(N - 1) alpha.

    Assumptions: Every ring round incurs the same startup cost.

    Excludes: Payload serialization and reduction compute.

    References
    ----------
    Derivation: Multiply the two-phase round count by the per-round startup latency.

    Examples
    --------
    >>> t = calc_ring_allreduce_latency_time(4, Q_("2 us"))
    >>> t == calc_ring_allreduce_latency_steps(4) * Q_("2 us")
    True
    """
    validate_at_least(n_gpus, 1, "n_gpus")
    alpha = _ensure_unit(latency_s, ureg.second, "latency_s")
    validate_nonnegative(alpha, "latency_s")
    return (calc_ring_allreduce_latency_steps(n_gpus) * alpha).to(ureg.second)


def calc_ring_attention_min_chunk(element_bytes, sustained_flops, link_bandwidth):
    """Estimate the token chunk needed to hide a ring-attention transfer.

    Parameters
    ----------
    element_bytes : float
        Positive byte count per key/value element s.
    sustained_flops : Quantity or float
        Positive sustained attention compute rate R; bare values are FLOP/second.
    link_bandwidth : Quantity or float
        Positive unidirectional link bandwidth B; bare values are bytes/second.

    Returns
    -------
    float
        Continuous chunk-size estimate in tokens; not rounded to an integer.

    Notes
    -----
    Equation: n_min = s R / (2 B).

    Assumptions: Tile compute is 4 n^2 d FLOPs and key/value transfer is 2 n d s bytes,
    with perfect overlap; d cancels.

    Excludes: Kernel launch costs, softmax overhead, link startup, and discrete tile
    constraints.

    References
    ----------
    Derivation: Set tile compute time 4 n^2 d/R greater than or equal to tile transfer
    time 2 n d s/B and cancel n d.

    Liu et al. (2023), "Ring Attention with Blockwise Transformers for Near-Infinite
    Context." https://arxiv.org/abs/2310.01889 (bibliography key: liu2023ringAttention). The threshold above is the stated
    ideal-overlap tile accounting, not an empirical latency fit.

    Examples
    --------
    >>> base = calc_ring_attention_min_chunk(2, Q_("1 TFLOP/s"), Q_("1 GB/s"))
    >>> calc_ring_attention_min_chunk(2, Q_("2 TFLOP/s"), Q_("1 GB/s")) == 2 * base
    True
    """
    rate = _ensure_unit(sustained_flops, ureg.flop / ureg.second, "sustained_flops")
    bw = _ensure_unit(link_bandwidth, ureg.byte / ureg.second, "link_bandwidth")
    validate_positive(bw, "link_bandwidth")
    return (float(element_bytes) * rate / (2 * bw)).to(ureg.flop / ureg.byte).magnitude


def calc_alpha_beta_crossover(alpha_s, bandwidth_bytes_s):
    """Find the message size where startup and transfer costs are equal.

    Parameters
    ----------
    alpha_s : Quantity or float
        Nonnegative startup latency alpha; bare values are seconds.
    bandwidth_bytes_s : Quantity or float
        Positive unidirectional bandwidth B; bare values are bytes/second.

    Returns
    -------
    Quantity
        Crossover message size in bytes.

    Notes
    -----
    Equation: M_cross = alpha B.

    Assumptions: Constant startup latency and size-independent bandwidth.

    Excludes: Protocol thresholds, contention, and collective round counts.

    References
    ----------
    Derivation: Solve alpha = M/B for M in the point-to-point alpha-beta model.

    Examples
    --------
    >>> m = calc_alpha_beta_crossover(Q_("1 us"), Q_("1 GB/s"))
    >>> (m / Q_("1 GB/s")).to("us") == Q_("1 us")
    True
    """
    alpha = _ensure_unit(alpha_s, ureg.second, "alpha_s")
    bandwidth = _ensure_unit(
        bandwidth_bytes_s, ureg.byte / ureg.second, "bandwidth_bytes_s"
    )
    validate_nonnegative(alpha, "alpha_s")
    validate_positive(bandwidth, "bandwidth_bytes_s")
    return (alpha * bandwidth).to(ureg.byte)


def calc_point_to_point_time(message_bytes, alpha_s, bandwidth_bytes_s):
    """Estimate startup plus serialization time for one transfer.

    Parameters
    ----------
    message_bytes : Quantity or float
        Nonnegative message size M; bare values are bytes.
    alpha_s : Quantity or float
        Nonnegative startup latency alpha; bare values are seconds.
    bandwidth_bytes_s : Quantity or float
        Positive unidirectional bandwidth B; bare values are bytes/second.

    Returns
    -------
    Quantity
        Transfer time in seconds, including startup even for zero bytes.

    Notes
    -----
    Equation: T = alpha + M/B.

    Assumptions: One transfer on an otherwise idle link at the supplied effective
    bandwidth.

    Excludes: Retransmission, queueing, and receiver processing.

    References
    ----------
    Derivation: Serialization takes bytes divided by bytes per second; add the fixed
    startup delay.

    Examples
    --------
    >>> calc_point_to_point_time(0, Q_("2 us"), Q_("1 GB/s")) == Q_("2 us")
    True
    """
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
    """Estimate throughput after stretching the communication portion of a step.

    Parameters
    ----------
    comm_fraction : float
        Baseline communication time fraction f in [0, 1].
    oversubscription_ratio : float
        Positive stretch ratio r; oversubscription normally means r >= 1.

    Returns
    -------
    tuple of float
        Relative throughput and throughput loss, both dimensionless; r < 1 produces a
        gain and negative loss.

    Notes
    -----
    Equation: relative = 1/((1 - f) + f r); loss = 1 - relative.

    Assumptions: Only communication time changes; compute and communication are serial.

    Excludes: Overlap changes, congestion nonlinearities, and altered workload size.

    References
    ----------
    Derivation: Normalize baseline step time to one, stretch its communication share by
    r, then invert the time ratio.

    Examples
    --------
    >>> relative, loss = calc_oversubscription_effect(0.5, 2)
    >>> relative + loss == 1
    True
    """
    validate_range(comm_fraction, 0.0, 1.0, "comm_fraction")
    validate_positive(oversubscription_ratio, "oversubscription_ratio")

    relative_throughput = 1 / ((1 - comm_fraction) + (comm_fraction * oversubscription_ratio))
    throughput_loss = 1 - relative_throughput
    return relative_throughput, throughput_loss


def calc_bisection_bandwidth(num_ports, link_bandwidth, oversubscription_ratio=1.0):
    """Sum effective link capacity across a specified partition cut.

    Parameters
    ----------
    num_ports : int
        Count of links crossing the cut, at least one; not all fabric ports.
    link_bandwidth : Quantity or float
        Positive capacity per crossing link; bare values are bytes/second.
    oversubscription_ratio : float, optional
        Positive capacity divisor; default one means no reduction.

    Returns
    -------
    Quantity
        Effective cut bandwidth in bytes/second.

    Notes
    -----
    Equation: B_cut = num_ports B_link / r.

    Assumptions: Identical crossing links; the specified ratio captures the shared
    upstream capacity.

    Excludes: Routing imbalance and traffic not crossing the selected cut.

    For a nonblocking fabric with N endpoints, N/2 links may cross the chosen cut;
    counting all ports would overstate this cut capacity.

    References
    ----------
    Derivation: Add parallel crossing-link capacities and divide by the supplied
    sharing factor.

    Examples
    --------
    >>> calc_bisection_bandwidth(4, Q_("1 GB/s"), 2) == 2 * Q_("1 GB/s")
    True
    """
    validate_at_least(num_ports, 1, "num_ports")
    validate_positive(oversubscription_ratio, "oversubscription_ratio")

    link_bw = _ensure_unit(
        link_bandwidth, ureg.byte / ureg.second, "link_bandwidth"
    )
    validate_positive(link_bw, "link_bandwidth")
    return (num_ports * link_bw / oversubscription_ratio).to(ureg.byte / ureg.second)


def calc_hop_latency(hops, hop_latency_s):
    """Sum fixed latency over identical sequential hops.

    Parameters
    ----------
    hops : int
        Nonnegative number of sequential hops.
    hop_latency_s : Quantity or float
        Nonnegative per-hop latency; bare values are seconds.

    Returns
    -------
    Quantity
        Total hop latency in seconds.

    Notes
    -----
    Equation: T = hops alpha.

    Assumptions: Each hop contributes the same serial delay.

    Excludes: Packet serialization, queueing, and overlapping hop processing.

    References
    ----------
    Derivation: Add one identical latency contribution per traversed hop.

    Examples
    --------
    >>> calc_hop_latency(0, Q_("1 us")).m_as("s") == 0
    True
    """
    validate_nonnegative(hops, "hops")
    latency = _ensure_unit(hop_latency_s, ureg.second, "hop_latency_s")
    validate_nonnegative(latency, "hop_latency_s")
    return (hops * latency).to(ureg.second)


def calc_fat_tree_hosts(radix, tiers):
    """Count hosts in the modeled nonblocking fat-tree.

    Parameters
    ----------
    radix : int
        Switch port count k, at least two; intended to be even. Odd inputs use
        floor(k/2).
    tiers : int
        Positive number of switching tiers L.

    Returns
    -------
    int
        Host count under the equal up/down port split.

    Notes
    -----
    Equation: H = 2 floor(k/2)^L.

    Assumptions: Identical even-radix switches, full population, and half the non-top
    ports facing each direction.

    Excludes: Oversubscription, spare ports, and incomplete pods.

    References
    ----------
    Derivation: Starting with a two-way split, multiply host reach by k/2 for each
    tier. For odd radix the implementation explicitly uses integer division.

    Al-Fares, Loukissas, and Vahdat (2008), "A Scalable, Commodity Data Center Network
    Architecture." https://doi.org/10.1145/1402958.1402967 (bibliography key: al-fares2008). The tier-general formula
    above extends the symmetric port-count construction.

    Examples
    --------
    >>> calc_fat_tree_hosts(4, 3) == calc_fat_tree_hosts(4, 2) * (4 // 2)
    True
    """
    validate_at_least(radix, 2, "radix")
    validate_at_least(tiers, 1, "tiers")
    half = radix // 2
    return 2 * half**tiers


def calc_fat_tree_switches(radix, tiers):
    """Count switches in the modeled full fat-tree.

    Parameters
    ----------
    radix : int
        Switch port count k, at least two; intended to be even. Odd inputs use
        floor(k/2).
    tiers : int
        Positive number of switching tiers L.

    Returns
    -------
    int
        Total switch count for the assumed topology.

    Notes
    -----
    Equation: S = (2L - 1) floor(k/2)^(L - 1).

    Assumptions: Full symmetric fabric with even radix and equal up/down splits.

    Excludes: Management switches, spares, and partially populated tiers.

    References
    ----------
    Derivation: Count one top-tier group and two groups per remaining tier, each with
    (k/2)^(L - 1) switches under this regular construction.

    Al-Fares, Loukissas, and Vahdat (2008), "A Scalable, Commodity Data Center Network
    Architecture." https://doi.org/10.1145/1402958.1402967 (bibliography key: al-fares2008). The tier-general expression
    follows the regular construction described above.

    Examples
    --------
    >>> calc_fat_tree_switches(4, 1) == 1
    True
    """
    validate_at_least(radix, 2, "radix")
    validate_at_least(tiers, 1, "tiers")
    half = radix // 2
    return (2 * tiers - 1) * half ** (tiers - 1)


def calc_fat_tree_path_switches(tiers):
    """Count switch traversals on the longest fat-tree path.

    Parameters
    ----------
    tiers : int
        Positive number of switching tiers L.

    Returns
    -------
    int
        Number of switches traversed between hosts in different top-level branches.

    Notes
    -----
    Equation: S_path = 2L - 1.

    Assumptions: Route ascends from the source leaf to the top tier and descends to the
    destination leaf.

    Excludes: Host NICs, retries, and alternate routing detours.

    References
    ----------
    Derivation: Count L ascending switches and L - 1 descending switches without
    counting the top switch twice.

    Examples
    --------
    >>> calc_fat_tree_path_switches(3) == 2 * 3 - 1
    True
    """
    validate_at_least(tiers, 1, "tiers")
    return 2 * tiers - 1


def calc_torus_bisection_links(side, dims=3):
    """Count links in a planar cut through a regular torus.

    Parameters
    ----------
    side : int
        Nodes per dimension, at least two for a bisection and at least one for
        diameter.
    dims : int, optional
        Positive dimension count; default three.

    Returns
    -------
    int
        Crossing-link count for the modeled planar cut.

    Notes
    -----
    Equation: links = 2 side^(dims - 1).

    Assumptions: Equal side lengths and wraparound links; an even side gives the usual
    equal planar bisection.

    Excludes: Irregular dimensions and claims that the odd-side planar cut is an equal
    partition.

    References
    ----------
    Derivation: There are side^(dims - 1) parallel rings and each contributes two
    crossings.

    Examples
    --------
    >>> calc_torus_bisection_links(4, 2) == 2 * 4
    True
    """
    validate_at_least(side, 2, "side")
    validate_at_least(dims, 1, "dims")
    return 2 * side ** (dims - 1)


def calc_torus_diameter(side, dims=3):
    """Count hops in the longest shortest path of a regular torus.

    Parameters
    ----------
    side : int
        Nodes per dimension, at least two for a bisection and at least one for
        diameter.
    dims : int, optional
        Positive dimension count; default three.

    Returns
    -------
    int
        Maximum shortest-path hop count.

    Notes
    -----
    Equation: diameter = dims floor(side/2).

    Assumptions: Bidirectional wraparound links with identical lengths in each
    dimension.

    Excludes: Routing restrictions, failed links, and hop timing.

    References
    ----------
    Derivation: The farthest coordinate is floor(side/2) hops away in each dimension;
    add the independent coordinate distances.

    Examples
    --------
    >>> calc_torus_diameter(4, 3) == 3 * (4 // 2)
    True
    """
    validate_at_least(side, 1, "side")
    validate_at_least(dims, 1, "dims")
    return dims * (side // 2)


def calc_tree_allreduce_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s):
    """Estimate reduction and broadcast over a binary tree.

    Parameters
    ----------
    message_bytes : Quantity or float
        Nonnegative message size M; bare values are bytes.
    n_gpus : int
        Participant count N, at least one; intended to be integral.
    bandwidth_bytes_s : Quantity or float
        Positive unidirectional bandwidth B; bare values are bytes/second.
    latency_s : Quantity or float
        Nonnegative startup latency alpha; bare values are seconds.

    Returns
    -------
    Quantity
        Two-phase tree communication time in seconds; zero for one participant.

    Notes
    -----
    Equation: T = 2 ceil(log2 N) (M/B + alpha).

    Assumptions: Each level serializes the full message; balanced tree with enough
    levels for N ranks.

    Excludes: Pipelining, local reduction compute, contention, and bandwidth-optimal
    double trees.

    References
    ----------
    Derivation: A reduction and a broadcast each traverse ceil(log2 N) levels, paying
    startup and full-message transfer at every level.

    Examples
    --------
    >>> calc_tree_allreduce_time(1024, 1, 1024, 0).m_as("s") == 0
    True
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
    """Approximate a double-tree collective with declared overhead factors.

    Parameters
    ----------
    message_bytes : Quantity or float
        Nonnegative message size M; bare values are bytes.
    n_gpus : int
        Participant count N, at least one; intended to be integral.
    bandwidth_bytes_s : Quantity or float
        Positive unidirectional bandwidth B; bare values are bytes/second.
    latency_s : Quantity or float
        Nonnegative startup latency alpha; bare values are seconds.
    latency_factor : float, optional
        Nonnegative startup multiplier; package assumption defaults to 1.2.
    bandwidth_factor : float, optional
        Nonnegative payload multiplier; package assumption defaults to 1.05.

    Returns
    -------
    Quantity
        Approximate collective time in seconds; zero for one participant.

    Notes
    -----
    Equation: T = f_latency 2 log2(N) alpha + f_bandwidth 2(N - 1) M/(N B).

    Assumptions: Logarithmic startup and near-ring traffic volume; the factors are
    modeling assumptions, not measurements.

    Excludes: Exact NCCL scheduling, reduction compute, and topology-specific routing.

    The startup term uses fractional log2(N), unlike the discrete level count in
    calc_tree_allreduce_time. Defaults do not establish implementation accuracy.

    References
    ----------
    Derivation: Combine a smooth logarithmic startup term with the ring-optimal traffic
    term; the user-supplied factors perturb these idealized costs.

    Examples
    --------
    >>> calc_double_binary_tree_allreduce_time(1024, 1, 1024, 0).m_as("s") == 0
    True
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
    """Estimate a large-rank ring/tree crossover payload.

    Parameters
    ----------
    n_gpus : int
        Participant count N, at least one; intended to be integral.
    alpha_s : Quantity or float
        Nonnegative startup latency alpha; bare values are seconds.
    bandwidth_bytes_s : Quantity or float
        Positive unidirectional bandwidth B; bare values are bytes/second.

    Returns
    -------
    Quantity
        Heuristic crossover size in bytes; zero for one participant.

    Notes
    -----
    Equation: M_est = N alpha B / log2(N).

    Assumptions: Large N; this is an approximation to the crossover of the ring and
    full-message tree models.

    Excludes: Exact crossover solving and small-N guarantees; N = 2 returns a finite
    estimate although those two cost models have no positive crossover.

    References
    ----------
    Derivation: For the exact intersection, the coefficient is ((N - 1) - ceil(log2
    N))/(ceil(log2 N) - (N - 1)/N). Replacing its leading terms by N/log2(N) yields
    this estimate.

    Examples
    --------
    >>> calc_ring_tree_crossover_size(1, Q_("1 us"), Q_("1 GB/s")).m_as("byte") == 0
    True
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
    """Estimate an equal-partition ring-style All-to-All exchange.

    Parameters
    ----------
    message_bytes : Quantity or float
        Nonnegative message size M; bare values are bytes.
    n_gpus : int
        Participant count N, at least one; intended to be integral.
    bandwidth_bytes_s : Quantity or float
        Positive unidirectional bandwidth B; bare values are bytes/second.
    latency_s : Quantity or float
        Nonnegative startup latency alpha; bare values are seconds.

    Returns
    -------
    Quantity
        Exchange time in seconds; zero for one participant.

    Notes
    -----
    Equation: T = (N - 1) M/(N B) + (N - 1) alpha.

    Assumptions: M is the per-rank total send buffer, divided into N equal destination
    chunks; one chunk stays local.

    Excludes: Unequal expert routing, fabric contention, local packing, and reduction
    compute.

    References
    ----------
    Derivation: Each rank sends N - 1 chunks of size M/N in sequential exchange rounds.

    Examples
    --------
    >>> calc_all_to_all_time(1024, 1, 1024, 0).m_as("s") == 0
    True
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
    """Compose local scatter, inter-node reduction, and local gather.

    Parameters
    ----------
    message_bytes : Quantity or float
        Nonnegative message size M; bare values are bytes.
    n_nodes : int
        Positive physical node count n.
    gpus_per_node : int
        Positive ranks per node g.
    intra_node_bw : Quantity or float
        Positive unidirectional local bandwidth; bare values are bytes/second.
    inter_node_bw : Quantity or float
        Positive per-GPU NIC bandwidth; bare values are bytes/second.
    intra_node_lat : Quantity or float, optional
        Nonnegative local round startup; default 500 ns; bare values are seconds.
    inter_node_lat : Quantity or float, optional
        Nonnegative inter-node round startup; default 5 us; bare values are seconds.

    Returns
    -------
    Quantity
        Sum of all three sequential phase costs in seconds.

    Notes
    -----
    Equation: T = 2[(g - 1) M/(g B_local) + (g - 1) alpha_local] + T_ring(M/g, n,
    B_network, alpha_network).

    Assumptions: Local ReduceScatter and AllGather surround g concurrent inter-node
    rings; each local GPU has its own NIC capacity.

    Excludes: Shared-NIC contention, phase overlap, packing, and local reduction
    arithmetic.

    If all local GPUs share one NIC, the inter-node traffic term needs an
    aggregate-capacity treatment; this independent-NIC model understates that
    contention.

    References
    ----------
    Derivation: Each local ring phase moves (g - 1)/g of M. Each GPU then reduces its
    M/g shard across nodes before local gathering.

    Examples
    --------
    >>> calc_hierarchical_allreduce_time(1024, 1, 1, 1024, 1024).m_as("s") == 0
    True
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
