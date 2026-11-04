"""Typed result models for communication calculations."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "TransferLadderResult",
    "TensorParallelCommResult",
    "GradientSyncResult",
    "CommunicationIntensityResult",
    "AlphaBetaResult",
    "BisectionResult",
    "IncastResult",
    "AllReduceAlgorithmResult",
    "HierarchicalAllReduceResult",
    "AllToAllDispatchResult",
    "RingTraceResult",
    "ErrorFeedbackResult",
    "CompressionRunResult",
    "OverlapBudgetResult",
]


class TransferLadderResult(SolverResult):
    """Result from TransferLadderModel: one payload moved over each link tier."""

    __module__ = "mlsysim.engine.results"

    payload: Quantity
    bandwidths: Dict[str, Quantity]
    times: Dict[str, Quantity]


class TensorParallelCommResult(SolverResult):
    """Result from TensorParallelCommModel: activation AllReduces in one TP step."""

    __module__ = "mlsysim.engine.results"

    payload: Quantity  # one AllReduce's logical tensor
    ring_factor: float  # 2(N-1)/N
    traffic_per_device: Quantity
    collectives_per_step: int
    per_collective_time: Quantity
    communication_time: Quantity
    compute_time: Quantity
    step_time: Quantity  # compute + communication, no overlap
    compute_fraction: float
    communication_fraction: float


class GradientSyncResult(SolverResult):
    """Result from GradientSyncModel: one data-parallel gradient exchange per link."""

    __module__ = "mlsysim.engine.results"

    sync_bytes: Quantity
    compute_time: Quantity
    transfer_times: Dict[str, Quantity]
    step_times: Dict[str, Quantity]
    compute_fractions: Dict[str, float]
    energy_per_step: Quantity  # on the energy link, per worker
    energy_total: Quantity  # over all steps, per worker


class CommunicationIntensityResult(SolverResult):
    """Result from CommunicationIntensityModel: bytes sent per local FLOP vs. fabric balance."""

    __module__ = "mlsysim.engine.results"

    local_tokens: int
    bytes_per_step: Quantity
    flops_per_step: Quantity
    intensity: float  # bytes per FLOP
    balance: float  # link bytes/s per peak FLOP/s
    compute_time_at_peak: Quantity


class AlphaBetaResult(SolverResult):
    """Result from AlphaBetaModel: alpha + n/beta for each message on each path."""

    __module__ = "mlsysim.engine.results"

    alphas: Dict[str, Quantity]
    bandwidths: Dict[str, Quantity]
    messages: Dict[str, Quantity]
    times: Dict[str, Dict[str, Quantity]]  # times[path][message]
    transmission_times: Dict[str, Dict[str, Quantity]]  # the n/beta term alone
    crossovers: Dict[str, Quantity]  # n* = alpha * beta, per path


class BisectionResult(SolverResult):
    """Result from BisectionModel: a ring AllReduce across a full and an oversubscribed cut."""

    __module__ = "mlsysim.engine.results"

    devices_per_side: int
    bisection_full: Quantity
    bisection_oversubscribed: Quantity
    ring_factor: float
    crossing_bytes: Quantity
    time_full: Quantity
    time_oversubscribed: Quantity
    slowdown: float
    relative_throughput: float
    throughput_loss: float
    idle_cost: Quantity


class IncastResult(SolverResult):
    """Result from IncastModel: many senders into one egress port with a finite buffer."""

    __module__ = "mlsysim.engine.results"

    offered_load: Quantity
    oversubscription: float
    net_fill_rate: Quantity
    overflow_time: Quantity  # buffer fills from empty
    drain_time: Quantity  # a full buffer drained by the egress port alone
    bandwidth_delay_product: Quantity


class AllReduceAlgorithmResult(SolverResult):
    """Result from AllReduceAlgorithmModel: ring, simple tree, and star schedules per group size.

    Every dict is keyed by the group size N. Byte times exclude startup; startup
    times are the round count times alpha, kept in alpha's own unit.
    """

    __module__ = "mlsysim.engine.results"

    message: Quantity
    alpha: Quantity
    bandwidth: Quantity
    chunk: Dict[int, Quantity]  # M / N, the bytes one ring round moves
    ring_factor: Dict[int, float]  # 2(N-1)/N
    ring_rounds: Dict[int, int]  # 2(N-1)
    tree_rounds: Dict[int, int]  # 2 ceil(log2 N)
    ring_byte_time: Dict[int, Quantity]
    ring_startup_time: Dict[int, Quantity]
    ring_time: Dict[int, Quantity]
    tree_byte_time: Dict[int, Quantity]
    tree_startup_time: Dict[int, Quantity]
    tree_time: Dict[int, Quantity]
    star_byte_time: Dict[int, Quantity]  # one server port carries N x M each way
    star_ring_limit: Dict[int, float]  # N/2, the large-N limit of star over ring byte time
    crossover_estimate: Dict[int, Quantity]  # N alpha beta / log2 N
    crossover_exact: Dict[int, Quantity]  # where the two models are equal


class HierarchicalAllReduceResult(SolverResult):
    """Result from HierarchicalAllReduceModel: three phases against one flat ring."""

    __module__ = "mlsysim.engine.results"

    intra_bytes: Quantity  # bytes each GPU sends in the in-node ReduceScatter
    reduce_scatter_time: Quantity
    shard: Quantity  # M / G, the payload of each cross-node ring
    inter_byte_time: Quantity
    inter_startup_time: Quantity
    inter_time: Quantity
    all_gather_time: Quantity
    total_time: Quantity
    flat_ring_time: Quantity  # one ring over every GPU at the inter-node rate, with startup
    flat_bound_time: Quantity  # the large-N byte bound 2M / beta of that ring
    speedup_vs_flat_ring: float
    speedup_vs_flat_bound: float


class AllToAllDispatchResult(SolverResult):
    """Result from AllToAllDispatchModel: one expert-parallel dispatch under uniform routing."""

    __module__ = "mlsysim.engine.results"

    tokens_per_device: int
    experts_per_device: int
    tokens_per_expert: int
    tokens_per_pair: int  # tokens one device sends one peer
    remote_peers: int
    bytes_per_token: Quantity
    pair_bytes: Quantity
    sent_bytes: Quantity  # per device, to all remote peers
    injection_bound: Quantity  # alpha + sent / beta, the peers sharing one port
    uniform_remote_tokens: int
    hot_remote_tokens: int
    uniform_receive_bytes: Quantity
    hot_receive_bytes: Quantity
    uniform_receive_bound: Quantity
    hot_receive_bound: Quantity


class RingTraceResult(SolverResult):
    """Result from RingTraceModel: every cell of a ring AllReduce on small integer vectors.

    ``scatter_reduce[s][g]`` and ``all_gather[s][g]`` are (chunk index, value)
    received by GPU g in step s.
    """

    __module__ = "mlsysim.engine.results"

    scatter_reduce: List[List[Tuple[int, int]]]
    owned: List[Tuple[int, int]]  # the fully reduced chunk each GPU holds after ReduceScatter
    totals: List[int]
    all_gather: List[List[Tuple[int, int]]]
    held: List[Set[int]]  # chunks each GPU holds after AllGather
    steps: int


class ErrorFeedbackResult(SolverResult):
    """Result from ErrorFeedbackModel: a rounding compressor with and without a carried residual."""

    __module__ = "mlsysim.engine.results"

    errors: List[float]  # e_1 .. e_{T+1}; errors[t] enters step t + 1
    corrected: List[float]  # g_t + e_t
    sent: List[int]  # v_t
    cumulative_true: List[float]
    cumulative_sent: List[int]
    total_sent: int
    final_error: float
    true_total: float
    naive_total_sent: int  # every value rounded alone, no residual


class CompressionRunResult(SolverResult):
    """Result from CompressionRunModel: time to a quality target with and without compression."""

    __module__ = "mlsysim.engine.results"

    steps_compressed: int
    step_times: Dict[str, Quantity]
    compressed_step_times: Dict[str, Quantity]
    totals: Dict[str, Quantity]
    compressed_totals: Dict[str, Quantity]
    changes: Dict[str, float]  # compressed / baseline - 1 (negative is faster)


class OverlapBudgetResult(SolverResult):
    """Result from OverlapBudgetModel: layer-by-layer AllReduce, sequential against overlapped."""

    __module__ = "mlsysim.engine.results"

    layers: int
    gradient_per_layer: Quantity
    buckets_per_layer: int
    communication_per_layer: Quantity
    total_backward: Quantity
    sequential_time: Quantity
    exposed_per_layer: Quantity
    overlapped_time: Quantity
    savings_fraction: float
