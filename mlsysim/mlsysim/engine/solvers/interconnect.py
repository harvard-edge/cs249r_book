"""Interconnect solvers: bandwidth-only transfer bounds across link tiers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

from typing import Dict

from ..results import (CommunicationIntensityResult, GradientSyncResult,
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
