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
        """Evaluate bandwidth-only transfer time on each named link.

        Parameters
        ----------
        payload : Quantity
            Nonnegative payload size in bytes.
        links : Dict[str, Quantity]
            Mapping of path names to positive bandwidth quantities in byte/s.

        Returns
        -------
        TransferLadderResult
            Payload bytes, bandwidths, and transfer times in milliseconds per link.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: t_i = payload / bandwidth_i.

        Assumptions: Each link carries the same complete payload at constant bandwidth.

        Excludes: Startup, contention, protocol overhead, and overlap.

        References
        ----------
        Derivation: t_i = payload / bandwidth_i. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import TransferLadderModel
        >>> TransferLadderModel().solve(Q_('1 GB'), {'link': Q_('1 GB/s')}).times['link'].to('s').magnitude == 1
        True
        """
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
        """Account for repeated activation collectives in a tensor-parallel step.

        Parameters
        ----------
        model : TransformerWorkload
            Workload with parameter count, dimensions, and weight size.
        tp_size : int
            Positive number of tensor-parallel ranks.
        microbatch : int
            Positive sequences per microbatch.
        link_bandwidth : Quantity
            Positive link bandwidth in byte/s.
        compute_time : Quantity
            Nonnegative local compute time, convertible to seconds.
        seq_len : int
            Positive tokens per sequence; None uses model context length.
        collectives_per_layer : int
            Nonnegative collectives per layer per modeled step.

        Returns
        -------
        TensorParallelCommResult
            Payload and per-rank traffic in bytes; per-collective, communication,
            compute, and total step times in milliseconds; dimensionless time shares.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: payload = microbatch * sequence * hidden_dim * FP16_bytes; traffic = 2(N-1)/N * payload; step = compute + layers * collectives_per_layer * traffic / bandwidth.

        Assumptions: FP16 activations and serialized bandwidth-only ring collectives; the caller chooses collective count.

        Excludes: Startup latency, collective overlap, contention, and model residency.

        References
        ----------
        Derivation: payload = microbatch * sequence * hidden_dim * FP16_bytes; traffic = 2(N-1)/N * payload; step = compute + layers * collectives_per_layer * traffic / bandwidth. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import TensorParallelCommModel
        >>> TensorParallelCommModel().solve(Models.Language.Llama3_8B, 2, 1, Q_('100 GB/s'), Q_('1 s'), seq_len=8).communication_time.magnitude > 0
        True
        """
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
        """Compare gradient synchronization time and energy across fabrics.

        Parameters
        ----------
        model : object
            Workload with parameter count, dimensions, and weight size.
        precision : str
            Weight or KV precision name; gradient synchronization also accepts bytes per element.
        links : object
            Mapping of path names to positive bandwidth quantities in byte/s.
        compute_time : object
            Nonnegative local compute time, convertible to seconds.
        energy_per_bit : object
            Optional nonnegative joules per transferred bit.
        steps : object
            Nonnegative number of steps for total energy.
        compute_speedup : object
            Positive dimensionless local compute multiplier.

        Returns
        -------
        GradientSyncResult
            Synchronization bytes, transfer and step times in seconds per link,
            compute fractions, and per-step and total energy in joules.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: sync_bytes = ring_factor * model_bytes; step_i = compute_time / speedup + sync_bytes / bandwidth_i; energy = sync_bits * energy_per_bit.

        Assumptions: The registry ring factor approximates large groups; transfers serialize after compute.

        Excludes: Finite-rank correction, overlap, reduction compute, and network retransmission.

        References
        ----------
        Derivation: sync_bytes = ring_factor * model_bytes; step_i = compute_time / speedup + sync_bytes / bandwidth_i; energy = sync_bits * energy_per_bit. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import GradientSyncModel
        >>> GradientSyncModel().solve(Models.Language.Llama3_8B, 'fp16', {'link': Q_('100 GB/s')}, Q_('1 s')).transfer_times['link'].magnitude > 0
        True
        """
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
        """Compare gradient bytes per FLOP with the fabric balance.

        Parameters
        ----------
        model : object
            Workload with parameter count, dimensions, and weight size.
        sequences_per_worker : object
            Positive sequences processed per worker per step.
        link_bandwidth : object
            Positive link bandwidth in byte/s.
        peak_flops : object
            Positive arithmetic rate in FLOP/s.
        precision : str
            Weight or KV precision name; gradient synchronization also accepts bytes per element.

        Returns
        -------
        CommunicationIntensityResult
            Local tokens, sent bytes, training FLOPs, byte/FLOP intensity and
            balance magnitudes, and ideal compute time in seconds.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: tokens = sequences_per_worker * context_length; intensity = ring_factor * model_bytes / training_FLOPs; balance = bandwidth / peak_FLOPs.

        Assumptions: Parameter-dominated dense training and the registry ring factor.

        Excludes: Attention work, collective startup, overlap, and achieved compute efficiency.

        References
        ----------
        Derivation: tokens = sequences_per_worker * context_length; intensity = ring_factor * model_bytes / training_FLOPs; balance = bandwidth / peak_FLOPs. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import CommunicationIntensityModel
        >>> CommunicationIntensityModel().solve(Models.Language.Llama3_8B, 1, Q_('100 GB/s'), Q_('100 TFLOP/s')).intensity > 0
        True
        """
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
        """Evaluate startup and wire time for named messages and paths.

        Parameters
        ----------
        alphas : Dict[str, Quantity]
            Mapping of path names to nonnegative startup times.
        bandwidths : Dict[str, Quantity]
            Mapping of path names to positive bandwidths in byte/s.
        messages : Dict[str, Quantity]
            Mapping of message names to nonnegative byte quantities.

        Returns
        -------
        AlphaBetaResult
            Per-path message times and wire times in seconds, with crossover bytes.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: time = alpha + message_bytes / bandwidth; crossover_bytes = alpha * bandwidth.

        Assumptions: Every alpha path has a bandwidth entry; both terms are constant per path.

        Excludes: Congestion, packetization, and throughput variation.

        References
        ----------
        Derivation: time = alpha + message_bytes / bandwidth; crossover_bytes = alpha * bandwidth. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import AlphaBetaModel
        >>> AlphaBetaModel().solve({'p': Q_('1 ms')}, {'p': Q_('1 GB/s')}, {'m': Q_('1 MB')}).times['p']['m'].to('ms').magnitude == 2
        True
        """
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
        """Bound collective traffic through a worst-case bisection placement.

        Parameters
        ----------
        n_devices : int
            Even device count at least two.
        link_bandwidth : Quantity
            Positive link bandwidth in byte/s.
        oversubscription_ratio : float
            Positive full-cut to available-cut bandwidth ratio.
        payload_per_device : Quantity
            Positive collective payload per device in bytes.
        communication_fraction : float
            Fraction of baseline step spent communicating, between zero and one.
        cluster_cost : Quantity
            Nonnegative capital cost in currency units.

        Returns
        -------
        BisectionResult
            Cut bandwidths, crossing GB, communication seconds, slowdown and
            throughput ratios, and idle-equivalent capital cost.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: crossing = floor(N/2) * payload * ring_factor; cut_bandwidth = floor(N/2) * link_bandwidth / oversubscription; throughput = 1 / ``(1-f+f*r)``.

        Assumptions: Even device groups with alternating placement across the cut and positive traffic.

        Excludes: Alternative rank placement, overlap changes, and monetary accounting beyond idle-capacity equivalence.

        References
        ----------
        Derivation: crossing = floor(N/2) * payload * ring_factor; cut_bandwidth = floor(N/2) * link_bandwidth / oversubscription; throughput = 1 / ``(1-f+f*r)``. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import BisectionModel
        >>> BisectionModel().solve(4, Q_('100 GB/s'), 2, Q_('1 GB'), 0.5, Q_('100 USD')).slowdown == 2
        True
        """
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
        """Estimate buffer fill and drain under sustained incast.

        Parameters
        ----------
        n_senders : int
            Positive number of simultaneous senders.
        sender_bandwidth : Quantity
            Positive bandwidth offered by each sender in byte/s.
        egress_bandwidth : Quantity
            Positive shared egress bandwidth in byte/s.
        buffer : Quantity
            Positive buffer capacity in bytes.
        feedback_rtt : Quantity
            Nonnegative feedback round-trip time.

        Returns
        -------
        IncastResult
            Offered and net fill bandwidths, dimensionless oversubscription,
            fill/drain microseconds, and bandwidth-delay product in KB.
            Bounds are conditional on the stated assumptions, not deployment guarantees.

        Notes
        -----
        Equation: offered = senders * sender_bandwidth; fill_time = buffer / (offered-egress); drain_time = buffer / egress; BDP = egress * RTT.

        Assumptions: Offered load exceeds positive egress bandwidth; constant rates during the burst.

        Excludes: Congestion control reactions, dropped packets, and finite burst termination.

        References
        ----------
        Derivation: offered = senders * sender_bandwidth; fill_time = buffer / (offered-egress); drain_time = buffer / egress; BDP = egress * RTT. This combines the supplied quantities by conservation
        of traffic or residence time; it is an analytical accounting model.

        Examples
        --------
        >>> from mlsysim import Models
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.solvers import IncastModel
        >>> IncastModel().solve(2, Q_('1 GB/s'), Q_('1 GB/s'), Q_('1 MB'), Q_('1 ms')).oversubscription == 2
        True
        """
        offered = n_senders * sender_bandwidth
        net = offered - egress_bandwidth
        return IncastResult(
            offered_load=offered, oversubscription=(offered / egress_bandwidth).to("").magnitude,
            net_fill_rate=net, overflow_time=(buffer / net).to("microsecond"),
            drain_time=(buffer / egress_bandwidth).to("microsecond"),
            bandwidth_delay_product=(egress_bandwidth * feedback_rtt).to("KB"),
        )


