"""Collective-communication solvers: AllReduce schedules, hierarchy, AllToAll
dispatch, gradient compression, and overlap with the backward pass.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
Every model is an alpha-beta bound: a startup ``alpha`` per round or message
plus bytes over a stated bandwidth ``beta``, with contention, protocol
overhead, and the reduction arithmetic left out.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Sequence

from ..results import (AllReduceAlgorithmResult, AllToAllDispatchResult, CompressionRunResult,
                       ErrorFeedbackResult, HierarchicalAllReduceResult, OverlapBudgetResult,
                       RingTraceResult)
from ...core.types import Quantity
from ...physics.communication import (calc_ring_allreduce_data_factor, calc_ring_allreduce_latency_steps,
                                      calc_ring_allreduce_latency_time, calc_ring_allreduce_time,
                                      calc_ring_collective_data_factor, calc_ring_tree_crossover_size,
                                      calc_tree_allreduce_time)
from .base import ForwardModel


class AllReduceAlgorithmModel(ForwardModel):
    """
    Ring, simple-tree, and parameter-server (star) AllReduce of one buffer at several group sizes.

    A ring takes ``2(N-1)`` rounds, each paying one startup ``alpha`` and moving
    one chunk of ``M/N`` bytes, so each rank sends ``2(N-1)/N`` times the
    buffer. The simple tree reduces up and broadcasts down in ``2 log2 N``
    rounds, forwarding the whole buffer each round. A single parameter server
    receives and sends ``N`` buffers through one port. The crossover is the
    buffer at which ring and tree cost the same: the log-aware estimate
    ``N alpha beta / log2 N`` and the exact intersection of the two models.

    Literature Source:
    1. Patarasuk and Yuan (2009), bandwidth-optimal ring AllReduce.
    2. Thakur, Rabenseifner, and Gropp (2005), tree and ring collectives.
    """
    requires = ("message", "group_sizes", "alpha", "bandwidth")
    produces = AllReduceAlgorithmResult

    def solve(self, message: Quantity, group_sizes: Sequence[int], alpha: Quantity,
              bandwidth: Quantity) -> AllReduceAlgorithmResult:
        out = {k: {} for k in ("chunk", "ring_factor", "ring_rounds", "tree_rounds", "ring_byte_time",
                               "ring_startup_time", "ring_time", "tree_byte_time", "tree_startup_time",
                               "tree_time", "star_byte_time", "star_ring_limit", "crossover_estimate",
                               "crossover_exact")}
        zero = 0 * alpha
        for n in group_sizes:
            if n < 2:
                raise ValueError(f"group size {n}: an AllReduce needs at least two ranks")
            ring_rounds = calc_ring_allreduce_latency_steps(n)
            tree_rounds = 2 * math.ceil(math.log2(n))
            factor = calc_ring_allreduce_data_factor(n)
            out["chunk"][n] = message / n
            out["ring_factor"][n] = factor
            out["ring_rounds"][n] = ring_rounds
            out["tree_rounds"][n] = tree_rounds
            out["ring_byte_time"][n] = calc_ring_allreduce_time(message, n, bandwidth, zero)
            # Startup terms stay in alpha's unit: a round trip through seconds leaves float residue.
            out["ring_startup_time"][n] = ring_rounds * alpha
            out["ring_time"][n] = calc_ring_allreduce_time(message, n, bandwidth, alpha)
            out["tree_byte_time"][n] = calc_tree_allreduce_time(message, n, bandwidth, zero)
            out["tree_startup_time"][n] = tree_rounds * alpha
            out["tree_time"][n] = calc_tree_allreduce_time(message, n, bandwidth, alpha)
            out["star_byte_time"][n] = (n * message / bandwidth).to("second")
            out["star_ring_limit"][n] = n / 2
            out["crossover_estimate"][n] = calc_ring_tree_crossover_size(n, alpha, bandwidth)
            # ring = tree:  (R - T) alpha = (T - f) M / beta, with R, T the round counts and f the ring factor
            out["crossover_exact"][n] = ((ring_rounds - tree_rounds) * alpha * bandwidth
                                         / (tree_rounds - factor)).to("byte")
        return AllReduceAlgorithmResult(message=message, alpha=alpha, bandwidth=bandwidth, **out)


class HierarchicalAllReduceModel(ForwardModel):
    """
    Three-phase hierarchical AllReduce against one flat ring across every GPU.

    Inside each node of ``G`` GPUs a ReduceScatter over the fast link leaves
    each GPU ``1/G`` of the node's partial sum, sending ``(G-1)/G`` of the
    buffer. GPUs in the same position on every node then ring-AllReduce their
    ``M/G`` shard across nodes, one ring per GPU position, each on its own
    port; an AllGather inside the node rebuilds the sum. The in-node phases
    carry no startup term here. The flat ring runs at the slowest edge, the
    inter-node port; ``flat_bound_time`` is its large-N byte bound ``2M/beta``.

    Literature Source: Sergeev and Del Balso (2018); NCCL hierarchical rings.
    """
    requires = ("message", "n_nodes", "gpus_per_node", "intra_bandwidth", "inter_bandwidth", "inter_alpha")
    produces = HierarchicalAllReduceResult

    def solve(self, message: Quantity, n_nodes: int, gpus_per_node: int, intra_bandwidth: Quantity,
              inter_bandwidth: Quantity, inter_alpha: Quantity) -> HierarchicalAllReduceResult:
        from ...literature.registry import Literature
        g = gpus_per_node
        intra = message * calc_ring_collective_data_factor(g)
        rs = (intra / intra_bandwidth).to("second")
        shard = message / g
        inter_bytes = calc_ring_allreduce_time(shard, n_nodes, inter_bandwidth, 0 * inter_alpha)
        inter_startup = calc_ring_allreduce_latency_time(n_nodes, inter_alpha)
        inter = (inter_bytes + inter_startup).to("second")
        total = (rs + inter + rs).to("second")
        flat = calc_ring_allreduce_time(message, n_nodes * g, inter_bandwidth, inter_alpha)
        bound = (float(Literature.Communication.RingAllreduceFactor) * message / inter_bandwidth).to("second")
        return HierarchicalAllReduceResult(
            intra_bytes=intra, reduce_scatter_time=rs, shard=shard, inter_byte_time=inter_bytes,
            inter_startup_time=inter_startup, inter_time=inter, all_gather_time=rs, total_time=total,
            flat_ring_time=flat, flat_bound_time=bound,
            speedup_vs_flat_ring=(flat / total).to("").magnitude,
            speedup_vs_flat_bound=(bound / total).to("").magnitude,
        )


class AllToAllDispatchModel(ForwardModel):
    """
    One mixture-of-experts dispatch AllToAll under uniform routing, and one hot receiver.

    Each device holds ``tokens / N`` tokens and sends an equal share to every
    device, keeping its own; a token is ``hidden_dim`` values. The concurrent
    peer streams share one injection port, so dispatch takes at least
    ``alpha + sent / beta``. On the receive side the bound is the receiving
    port: a device whose experts draw ``skew`` times the uniform remote load
    takes ``skew`` times longer, and every device waiting on its outputs waits
    with it.
    """
    requires = ("total_tokens", "n_devices", "n_experts", "hidden_dim", "bytes_per_element", "alpha",
                "bandwidth")
    produces = AllToAllDispatchResult

    def solve(self, total_tokens: int, n_devices: int, n_experts: int, hidden_dim: int,
              bytes_per_element: Quantity, alpha: Quantity, bandwidth: Quantity,
              receive_skew: float = 1.0) -> AllToAllDispatchResult:
        per_device = total_tokens // n_devices
        per_pair = per_device // n_devices
        remote = n_devices - 1
        token_bytes = hidden_dim * bytes_per_element
        pair_bytes = per_pair * token_bytes
        sent = remote * pair_bytes
        uniform_tokens = remote * per_pair
        hot_tokens = round(receive_skew * uniform_tokens)
        uniform_rx = uniform_tokens * token_bytes
        hot_rx = hot_tokens * token_bytes
        return AllToAllDispatchResult(
            tokens_per_device=per_device, experts_per_device=n_experts // n_devices,
            tokens_per_expert=total_tokens // n_experts, tokens_per_pair=per_pair, remote_peers=remote,
            bytes_per_token=token_bytes, pair_bytes=pair_bytes, sent_bytes=sent,
            injection_bound=alpha + sent / bandwidth,
            uniform_remote_tokens=uniform_tokens, hot_remote_tokens=hot_tokens,
            uniform_receive_bytes=uniform_rx, hot_receive_bytes=hot_rx,
            uniform_receive_bound=(uniform_rx / bandwidth).to("microsecond"),
            hot_receive_bound=(hot_rx / bandwidth).to("microsecond"),
        )


class RingTraceModel(ForwardModel):
    """
    Every step of a ring AllReduce on one small vector per GPU, the vectors split into N chunks.

    In ReduceScatter step s, GPU i sends chunk (i - s) mod N to GPU i + 1,
    which adds it into its own copy; after N - 1 steps GPU i owns the fully
    reduced chunk (i + 1) mod N. In AllGather step s (from 1), GPU i forwards
    chunk (i - s + 2) mod N, the one it completed or received most recently,
    so after N - 1 more steps every GPU holds every chunk.
    """
    requires = ("vectors",)
    produces = RingTraceResult

    def solve(self, vectors: Sequence[Sequence[int]]) -> RingTraceResult:
        n = len(vectors)
        partial = [list(row) for row in vectors]
        scatter = []
        for s in range(n - 1):
            sent = [(i, (i - s) % n, partial[i][(i - s) % n]) for i in range(n)]
            row = [None] * n
            for src, chunk, value in sent:
                dst = (src + 1) % n
                partial[dst][chunk] += value
                row[dst] = (chunk, partial[dst][chunk])
            scatter.append(row)
        owned = [((i + 1) % n, partial[i][(i + 1) % n]) for i in range(n)]
        totals = [sum(v[j] for v in vectors) for j in range(len(vectors[0]))]
        held = [{(i + 1) % n} for i in range(n)]
        gather = []
        for s in range(1, n):
            row = [None] * n
            for i in range(n):
                chunk = (i - s + 2) % n
                dst = (i + 1) % n
                held[dst].add(chunk)
                row[dst] = (chunk, totals[chunk])
            gather.append(row)
        return RingTraceResult(scatter_reduce=scatter, owned=owned, totals=totals, all_gather=gather,
                               held=held, steps=2 * (n - 1))


def _nearest(x: float) -> int:
    """Nearest whole unit, ties away from zero, after removing float noise."""
    return int(Decimal(repr(round(x, 9))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


class ErrorFeedbackModel(ForwardModel):
    """
    A compressor that sends each value rounded to a whole unit, with error feedback.

    Without a residual, every value below half a unit is lost. With error
    feedback the worker adds the carried residual ``e_t`` to the gradient,
    sends the rounded ``v_t``, and keeps ``e_{t+1} = g_t + e_t - v_t``; what
    is sent plus the last residual equals the true sum at every step.

    Literature Source: Karimireddy et al. (2019), "Error Feedback Fixes SignSGD".
    """
    requires = ("gradients",)
    produces = ErrorFeedbackResult

    def solve(self, gradients: Sequence[float]) -> ErrorFeedbackResult:
        errors, corrected, sent = [0.0], [], []
        for t, g in enumerate(gradients):
            s = g + errors[t]
            v = _nearest(s)
            corrected.append(s)
            sent.append(v)
            errors.append(s - v)
        cum_true, acc = [], 0.0
        for g in gradients:
            acc += g
            cum_true.append(acc)
        cum_sent, acc_v = [], 0
        for v in sent:
            acc_v += v
            cum_sent.append(acc_v)
        return ErrorFeedbackResult(
            errors=errors, corrected=corrected, sent=sent, cumulative_true=cum_true, cumulative_sent=cum_sent,
            total_sent=int(sum(sent)), final_error=round(errors[-1], 4), true_total=sum(gradients),
            naive_total_sent=sum(_nearest(g) for g in gradients),
        )


class CompressionRunModel(ForwardModel):
    """
    Time to a quality target with and without gradient compression, per network.

    Each step is compute plus exposed communication, with nothing overlapped.
    Compression divides only the communication term by ``ratio`` and needs
    ``1 + step_penalty`` times the steps to reach the same quality, so it
    pays only where communication is a large share of the step.
    """
    requires = ("steps", "compute_time", "communication_times", "ratio", "step_penalty")
    produces = CompressionRunResult

    def solve(self, steps: int, compute_time: Quantity, communication_times: Dict[str, Quantity],
              ratio: float, step_penalty: float) -> CompressionRunResult:
        steps_c = round(steps * (1 + step_penalty))
        step, step_c, total, total_c, change = {}, {}, {}, {}, {}
        for k, comm in communication_times.items():
            step[k] = compute_time + comm
            step_c[k] = compute_time + comm / ratio
            total[k] = (steps * step[k]).to("second")
            total_c[k] = (steps_c * step_c[k]).to("second")
            change[k] = (total_c[k] / total[k]).to("").magnitude - 1
        return CompressionRunResult(steps_compressed=steps_c, step_times=step, compressed_step_times=step_c,
                                    totals=total, compressed_totals=total_c, changes=change)


class OverlapBudgetModel(ForwardModel):
    """
    Layer-by-layer gradient AllReduce, run after the backward pass or overlapped with it.

    Each of ``L`` equal layers yields its gradient, reduced in fixed-size
    buckets that each take ``time_per_bucket``. Sequentially, the step pays
    the whole backward pass and then every reduction. Overlapped, a layer's
    reduction starts when its backward finishes; when a reduction outlasts the
    next layer's backward, the excess accumulates, and the last reduction has
    no compute left to hide under.
    """
    requires = ("workload", "bytes_per_value", "backward_per_layer", "bucket_size", "time_per_bucket")
    produces = OverlapBudgetResult

    def solve(self, workload, bytes_per_value: Quantity, backward_per_layer: Quantity, bucket_size: Quantity,
              time_per_bucket: Quantity) -> OverlapBudgetResult:
        layers = workload.layers
        grad = (workload.size_in_bytes(bytes_per_value) / layers).to("MB")
        buckets = math.ceil((grad / bucket_size).to_base_units().magnitude)
        comm = buckets * time_per_bucket
        backward = layers * backward_per_layer
        sequential = backward + layers * comm
        exposed = max(0 * comm, comm - backward_per_layer)
        overlapped = backward + comm + (layers - 1) * exposed
        return OverlapBudgetResult(
            layers=layers, gradient_per_layer=grad, buckets_per_layer=buckets, communication_per_layer=comm,
            total_backward=backward, sequential_time=sequential, exposed_per_layer=exposed,
            overlapped_time=overlapped, savings_fraction=1 - (overlapped / sequential).to_base_units().magnitude,
        )
