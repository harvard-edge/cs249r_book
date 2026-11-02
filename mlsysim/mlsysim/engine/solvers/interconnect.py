"""Interconnect solvers: transfer bounds across link tiers, alpha-beta messages,
bisection cuts, incast, and fits to measured collectives.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

from typing import Dict

from ..results import (AlphaBetaResult, BisectionResult, 
                       CommunicationIntensityResult, GradientSyncResult, IncastResult,
                       TensorParallelCommResult, TransferLadderResult)
from ...core.types import Quantity
from ...core.units import BYTES_FP16, Q_
from ...models.types import TransformerWorkload
from ...physics.communication import calc_ring_allreduce_data_factor
from ...physics.quantities import transfer_time
from .base import ForwardModel


class TransferLadderModel(ForwardModel):
    """
    Moves one payload over each tier of a bandwidth hierarchy.

    The time on each tier is the bandwidth-only lower bound, ``bytes /
    bandwidth``; per-message startup latency (the alpha term) is excluded.
    The ratios between tiers are the bandwidth staircase that decides where
    each kind of parallel traffic may run.
    """
    requires = ("payload", "links")
    produces = TransferLadderResult

    def solve(self, payload: Quantity, links: Dict[str, Quantity]) -> TransferLadderResult:
        times = {name: transfer_time(payload, bw).to("millisecond") for name, bw in links.items()}
        return TransferLadderResult(payload=payload, bandwidths=dict(links), times=times)


class TensorParallelCommModel(ForwardModel):
    """
    Activation AllReduces of one tensor-parallel training step.

    Each Transformer layer sums an activation-sized partial result across the
    tensor-parallel group once in the forward pass and once in the backward
    pass. A ring AllReduce over N devices moves ``2(N-1)/N`` times the payload
    per device. The bound is bandwidth-only and assumes no overlap with the
    stated compute time, so the fractions are a diagnostic, not a forecast.

    Literature Source:
    1. Shoeybi et al. (2019), "Megatron-LM" (two AllReduces per layer per pass).
    2. Patarasuk and Yuan (2009), ring AllReduce bandwidth factor.
    """
    requires = ("workload", "tp_size", "link_bandwidth")
    produces = TensorParallelCommResult

    def solve(self, model: TransformerWorkload, tp_size: int, microbatch: int,
              link_bandwidth: Quantity, compute_time: Quantity, seq_len: int = None,
              collectives_per_layer: int = 2) -> TensorParallelCommResult:
        seq = seq_len if seq_len is not None else model.context_length
        payload = (microbatch * seq * model.hidden_dim * BYTES_FP16).to("byte")
        factor = calc_ring_allreduce_data_factor(tp_size)
        traffic = factor * payload
        n = collectives_per_layer * model.layers
        per = transfer_time(traffic, link_bandwidth).to("millisecond")
        comm = (n * per).to("millisecond")
        comp = compute_time.to("millisecond")
        step = comp + comm
        return TensorParallelCommResult(
            payload=payload,
            ring_factor=factor,
            traffic_per_device=traffic,
            collectives_per_step=n,
            per_collective_time=per,
            communication_time=comm,
            compute_time=comp,
            step_time=step,
            compute_fraction=(comp / step).to("").magnitude,
            communication_fraction=(comm / step).to("").magnitude,
        )


class GradientSyncModel(ForwardModel):
    """
    One data-parallel step: local compute, then a ring AllReduce of the gradients.

    Each worker sends about ``2x`` the gradient bytes per step (the ring factor
    for large N). With no overlap the step is compute plus transfer, so the
    compute fraction shows how much of the step the accelerator works. Energy
    charges each transferred bit once at the fabric's energy per bit.
    """
    requires = ("workload", "links")
    produces = GradientSyncResult

    def solve(self, model, precision, links, compute_time, energy_per_bit=None,
              steps=1, compute_speedup=1.0) -> GradientSyncResult:
        from ...literature.registry import Literature
        sync = (float(Literature.Communication.RingAllreduceFactor) * model.size_in_bytes(precision)).to("byte")
        comp = (compute_time / compute_speedup).to("second")
        transfer = {k: transfer_time(sync, bw).to("second") for k, bw in links.items()}
        step = {k: (comp + t).to("second") for k, t in transfer.items()}
        frac = {k: (comp / step[k]).to("").magnitude for k in step}
        e_step = (sync.to("bit") * energy_per_bit).to("joule") if energy_per_bit is not None else Q_(0, "joule")
        return GradientSyncResult(
            sync_bytes=sync, compute_time=comp, transfer_times=transfer, step_times=step,
            compute_fractions=frac, energy_per_step=e_step, energy_total=(e_step * steps),
        )


class CommunicationIntensityModel(ForwardModel):
    """
    Communication intensity of one data-parallel worker step against fabric balance.

    CI is bytes sent over the network per FLOP done locally. The fabric's
    balance point is link bandwidth over peak arithmetic rate: below it the
    step is set by compute, above it by the network.
    """
    requires = ("workload", "link_bandwidth", "peak_flops")
    produces = CommunicationIntensityResult

    def solve(self, model, sequences_per_worker, link_bandwidth, peak_flops,
              precision="fp16") -> CommunicationIntensityResult:
        from ...literature.registry import Literature
        from ...physics.transformer import calc_transformer_training_flops
        tokens = sequences_per_worker * model.context_length
        sent = (float(Literature.Communication.RingAllreduceFactor) * model.size_in_bytes(precision)).to("byte")
        flops = calc_transformer_training_flops(model.parameters, tokens)
        return CommunicationIntensityResult(
            local_tokens=tokens, bytes_per_step=sent, flops_per_step=flops,
            intensity=(sent.to("byte").magnitude / flops.to("flop").magnitude),
            balance=(link_bandwidth.to("byte/second").magnitude / peak_flops.to("flop/second").magnitude),
            compute_time_at_peak=(flops / peak_flops).to("second"),
        )


class AlphaBetaModel(ForwardModel):
    """
    Point-to-point message time under the alpha-beta model on several paths.

    Each message of ``n`` bytes pays a fixed startup ``alpha`` (NIC, switch
    hops, error correction, software) plus ``n / beta`` on the wire. The two
    terms are equal at ``n* = alpha * beta``: below it a lower alpha helps,
    above it only a higher beta does.

    Literature Source: Hockney (1994), "The communication challenge for MPP".
    """
    requires = ("alphas", "bandwidths", "messages")
    produces = AlphaBetaResult

    def solve(self, alphas: Dict[str, Quantity], bandwidths: Dict[str, Quantity],
              messages: Dict[str, Quantity]) -> AlphaBetaResult:
        from ...physics.communication import calc_alpha_beta_crossover, calc_point_to_point_time
        times, wire = {}, {}
        for path, alpha in alphas.items():
            beta = bandwidths[path]
            times[path] = {m: calc_point_to_point_time(n, alpha, beta) for m, n in messages.items()}
            wire[path] = {m: (n / beta).to("second") for m, n in messages.items()}
        cross = {path: calc_alpha_beta_crossover(alpha, bandwidths[path]) for path, alpha in alphas.items()}
        return AlphaBetaResult(alphas=dict(alphas), bandwidths=dict(bandwidths), messages=dict(messages),
                               times=times, transmission_times=wire, crossovers=cross)


class BisectionModel(ForwardModel):
    """
    A ring AllReduce whose every edge crosses the bisection, on a full and an oversubscribed cut.

    With ranks placed alternately in the two halves, each of the ``N/2`` ring
    edges in one direction crosses the cut and carries ``2(N-1)/N`` times the
    per-device payload (ReduceScatter then AllGather). The cut carries ``N/2``
    links, divided by the oversubscription ratio. Only the communication
    fraction of the step stretches, so relative throughput is
    ``1 / ((1 - f) + f r)``; the lost share of a cluster's capital cost is its
    equivalent idle capacity. This placement is the worst case.
    """
    requires = ("n_devices", "link_bandwidth", "oversubscription_ratio")
    produces = BisectionResult

    def solve(self, n_devices: int, link_bandwidth: Quantity, oversubscription_ratio: float,
              payload_per_device: Quantity, communication_fraction: float,
              cluster_cost: Quantity) -> BisectionResult:
        from ...physics.communication import calc_bisection_bandwidth, calc_oversubscription_effect
        side = n_devices // 2
        full = calc_bisection_bandwidth(side, link_bandwidth)
        over = calc_bisection_bandwidth(side, link_bandwidth, oversubscription_ratio=oversubscription_ratio)
        factor = calc_ring_allreduce_data_factor(n_devices)
        crossing = side * payload_per_device * factor
        t_full = (crossing / full).to("second")
        t_over = (crossing / over).to("second")
        rel, loss = calc_oversubscription_effect(communication_fraction, oversubscription_ratio)
        return BisectionResult(
            devices_per_side=side, bisection_full=full, bisection_oversubscribed=over,
            ring_factor=factor, crossing_bytes=crossing.to("GB"), time_full=t_full,
            time_oversubscribed=t_over, slowdown=(t_over / t_full).to("").magnitude,
            relative_throughput=rel, throughput_loss=loss, idle_cost=loss * cluster_cost,
        )


class IncastModel(ForwardModel):
    """
    Many senders converge on one egress port (incast).

    The offered load is the senders' combined rate; the buffer fills at the
    offered load less what the egress drains, and a full buffer holds
    ``buffer / egress`` of waiting. The bandwidth-delay product is the data in
    flight on the egress link during one feedback round trip, which is what
    end-to-end congestion control must absorb before it can react.
    """
    requires = ("n_senders", "sender_bandwidth", "egress_bandwidth", "buffer")
    produces = IncastResult

    def solve(self, n_senders: int, sender_bandwidth: Quantity, egress_bandwidth: Quantity,
              buffer: Quantity, feedback_rtt: Quantity) -> IncastResult:
        offered = n_senders * sender_bandwidth
        net = offered - egress_bandwidth
        return IncastResult(
            offered_load=offered, oversubscription=(offered / egress_bandwidth).to("").magnitude,
            net_fill_rate=net, overflow_time=(buffer / net).to("microsecond"),
            drain_time=(buffer / egress_bandwidth).to("microsecond"),
            bandwidth_delay_product=(egress_bandwidth * feedback_rtt).to("KB"),
        )


