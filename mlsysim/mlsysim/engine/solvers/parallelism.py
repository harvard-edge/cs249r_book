"""Distributed-training solvers: parallel layouts, ZeRO sharding, data-parallel
scale-out, tensor-parallel layer exchanges, and alignment-training state.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). Added 2026-10-20.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from ..results import (AlignmentMemoryResult, DataParallelScaleOutResult, ParallelLayoutResult,
                       TensorParallelLayerResult, ZeroShardingResult)
from ...core.types import Quantity
from ...core.units import BYTES_FP16, Q_, param
from ...models.types import TransformerWorkload, Workload
from ...physics.communication import calc_ring_allreduce_data_factor, calc_ring_allreduce_time
from ...physics.performance import calc_pipeline_bubble
from ...physics.quantities import memory_from_params, transfer_time
from .base import ForwardModel


def _adam_bytes():
    """Mixed-precision Adam bytes per parameter: (all persistent state, optimizer only)."""
    from ...reference_stats.registry import ReferenceStats
    adam = ReferenceStats.MixedPrecisionAdam
    return adam.TrainingStateBytesPerParameter, adam.OptimizerStateBytesPerParameter


def _ring_factor() -> float:
    """The large-N ring AllReduce volume, about 2x the payload per device."""
    from ...literature.registry import Literature
    return float(Literature.Communication.RingAllreduceFactor)


def _round_to(t: Quantity, resolution: Optional[Quantity]) -> Quantity:
    """``t`` carried at ``resolution`` (a power of ten), the napkin-math convention."""
    if resolution is None:
        return t
    res = resolution.to(t.units).magnitude
    digits = round(-math.log10(res))
    if abs(10 ** -digits - res) < 1e-12:
        return round(t.magnitude, digits) * t.units
    return round(t.magnitude / res) * res * t.units


class ParallelLayoutModel(ForwardModel):
    """
    One training step of a TP x PP x DP layout, read through the Fleet Law.

    Compute: the step's ``6 P D`` FLOPs (``flops_per_param_token`` per
    parameter per token, ``D`` tokens per update) spread over every device at
    a local share of peak. Communication: each data-parallel rank reduces its
    gradient shard (the full gradient over TP x PP) with a ring AllReduce of
    about twice the shard; up to ``overlap_fraction`` of that transfer hides
    under the compute window, never more than the window itself. Pipeline
    stages idle ``(p-1)/(m+p-1)`` of the step, which stretches the compute
    window by ``1/(1 - bubble)``. State: 16 B per parameter of mixed-precision
    Adam split over TP x PP; ZeRO-1 also shards the 12 B of optimizer state
    across the data-parallel group.

    Literature Source:
    1. Narayanan et al. (2021), Megatron-LM 3D parallelism.
    2. Rajbhandari et al. (2020), ZeRO (16 B/param, optimizer sharding).
    3. Huang et al. (2019), GPipe (bubble fraction).
    """
    requires = ("workload", "tp_size", "pp_size", "dp_size")
    produces = ParallelLayoutResult

    def solve(self, model: Workload, tp_size: int, pp_size: int, dp_size: int, microbatches: int,
              batch_tokens: float, flops_per_param_token: float, peak_flops: Quantity,
              local_efficiency: float, link_bandwidth: Quantity, overlap_fraction: float,
              gradient_precision="fp16", window_resolution: Optional[Quantity] = None,
              device_memory: Optional[Quantity] = None) -> ParallelLayoutResult:
        n = tp_size * pp_size * dp_size
        p = model.parameters.to(param).magnitude
        step_flops = Q_(flops_per_param_token * p * batch_tokens, "flop")
        compute = _round_to((step_flops / (n * peak_flops * local_efficiency)).to("second"),
                            window_resolution)
        grad = model.size_in_bytes(gradient_precision).to("GB")
        shard = grad / (tp_size * pp_size)
        ring = (calc_ring_allreduce_data_factor(dp_size) * shard).to("GB")
        comm = (_ring_factor() * shard / link_bandwidth).to("second")
        hidden = min(comm * overlap_fraction, compute)
        exposed = comm - hidden
        bubble = calc_pipeline_bubble(pp_size, microbatches)
        stretched = compute / (1 - bubble)
        step = stretched + exposed
        eff = (compute / step).to("").magnitude
        eff_raw = (compute / (stretched + comm)).to("").magnitude
        state_bpp, opt_bpp = _adam_bytes()
        state = memory_from_params(model.parameters, state_bpp).to("GB")
        wg = memory_from_params(model.parameters, 2 * BYTES_FP16 / param)
        opt = memory_from_params(model.parameters, opt_bpp)
        zero1 = (wg / (tp_size * pp_size) + opt / (tp_size * pp_size * dp_size)).to("GB")
        min_dev = None
        if device_memory is not None:
            min_dev = math.ceil((state / device_memory).to("").magnitude)
        return ParallelLayoutResult(
            n_devices=n, step_flops=step_flops, compute_time=compute, gradient_bytes=grad,
            dp_shard=shard, ring_bytes=ring, allreduce_time=comm, hidden_time=hidden,
            exposed_time=exposed, bubble=bubble, efficiency=eff, efficiency_no_overlap=eff_raw,
            overlap_share=(hidden / comm).to("").magnitude, step_time=step.to("second"),
            state_total=state, state_per_device=state / (tp_size * pp_size),
            zero1_state_per_device=zero1, fleet_mfu=local_efficiency * eff,
            double_counted_mfu=local_efficiency * eff * eff, min_devices_for_state=min_dev,
        )


class ZeroShardingModel(ForwardModel):
    """
    Per-device training state at each ZeRO stage, and what full sharding sends.

    DDP replicates weights (W), gradients (G), and optimizer state (O) on every
    device. ZeRO-1 shards O over N devices, ZeRO-2 also G, and ZeRO-3 (FSDP)
    also W. Stages 1 and 2 move DDP's bytes (a ring AllReduce, about 2W per
    device); ZeRO-3 gathers every layer's weights in forward and again in
    backward and reduce-scatters the gradients, about 3W, 1.5x as much.

    Literature Source:
    1. Rajbhandari et al. (2020), "ZeRO" (stages and communication volume).
    """
    requires = ("workload", "n_devices")
    produces = ZeroShardingResult

    def solve(self, model: Workload, n_devices: int, activations: Optional[Quantity] = None,
              link_bandwidth: Optional[Quantity] = None) -> ZeroShardingResult:
        _, opt_bpp = _adam_bytes()
        w = model.size_in_bytes(BYTES_FP16).to("GB")
        g = w
        o = memory_from_params(model.parameters, opt_bpp).to("GB")
        n = n_devices
        stages = {
            "DDP": w + g + o,
            "ZeRO-1": w + g + o / n,
            "ZeRO-2": w + (g + o) / n,
            "ZeRO-3": (w + g + o) / n,
        }
        ddp_bytes = _ring_factor() * w
        gather = 3 * w  # forward gather, backward regather, gradient ReduceScatter
        multiple = (gather / ddp_bytes).to("").magnitude
        act = (activations if activations is not None else Q_(0, "GB")).to("GB")
        return ZeroShardingResult(
            weights=w, gradients=g, optimizer_state=o, stage_state=stages,
            traffic_multiple={"DDP": 1.0, "ZeRO-1": 1.0, "ZeRO-2": 1.0, "ZeRO-3": multiple},
            activations=act, zero3_total_per_device=stages["ZeRO-3"] + act, gather_traffic=gather,
            gather_time=None if link_bandwidth is None else transfer_time(gather, link_bandwidth).to("second"),
        )


class DataParallelScaleOutModel(ForwardModel):
    """
    A data-parallel job inside one node and across nodes, from its one-device profile.

    Every update reduces the full gradient: a ring AllReduce inside each node
    on the fast link, and, with more than one node, a ring across all devices
    on the inter-node link. Gradient accumulation runs ``accumulation_steps``
    local steps per update, so the same exchange serves more compute and the
    global batch grows. Speedup is the device count times the compute share of
    each update; training time and cost follow from the one-device run.
    """
    requires = ("workload", "devices_per_node", "nodes")
    produces = DataParallelScaleOutResult

    def solve(self, model: Workload, precision, step_time: Quantity, training_time: Quantity,
              per_device_batch: int, devices_per_node: int, nodes: int, intra_bandwidth: Quantity,
              inter_bandwidth: Optional[Quantity] = None, accumulation_steps: int = 1,
              node_hour_rate: Optional[Quantity] = None) -> DataParallelScaleOutResult:
        n = devices_per_node * nodes
        grad = model.size_in_bytes(precision)
        zero = Q_(0, "ms")
        intra_bytes = grad * calc_ring_allreduce_data_factor(devices_per_node)
        intra = calc_ring_allreduce_time(grad, devices_per_node, intra_bandwidth, zero).to("ms")
        if nodes > 1:
            inter_bytes = grad * calc_ring_allreduce_data_factor(n)
            inter = calc_ring_allreduce_time(grad, n, inter_bandwidth, zero).to("ms")
        else:
            inter_bytes, inter = 0 * grad, zero
        comm = intra + inter
        compute = (accumulation_steps * step_time).to("ms")
        update = compute + comm
        speedup = n * (compute / update).to("").magnitude
        hours = (training_time / speedup).to("hour")
        cost = None
        if node_hour_rate is not None:
            cost = (hours * nodes * node_hour_rate).to("USD").magnitude
        return DataParallelScaleOutResult(
            n_devices=n, gradient_bytes=grad, intra_bytes=intra_bytes, inter_bytes=inter_bytes,
            intra_time=intra, inter_time=inter, communication_time=comm, compute_time=compute,
            update_time=update, communication_overhead=(comm / compute).to("").magnitude,
            compute_share=(compute / update).to("").magnitude,
            communication_share=(comm / update).to("").magnitude,
            speedup=speedup, efficiency=speedup / n,
            global_batch=per_device_batch * n * accumulation_steps, training_time=hours, cost=cost,
        )


class TensorParallelLayerModel(ForwardModel):
    """
    One Transformer layer's forward compute against its tensor-parallel exchange.

    The layer's forward pass is ``2 (P/L)`` FLOPs per token, split over the TP
    group and run at a sustained share of peak. Megatron-style tensor
    parallelism sums an activation of ``tokens x d_model`` once after attention
    and once after the MLP, each a ring AllReduce of ``2(t-1)/t`` times the
    payload per device. The exchange must fit inside the compute on the link
    that carries it.

    Literature Source:
    1. Shoeybi et al. (2019), "Megatron-LM" (two AllReduces per layer per pass).
    """
    requires = ("workload", "tp_size")
    produces = TensorParallelLayerResult

    def solve(self, model: TransformerWorkload, tp_size: int, tokens: int, peak_flops: Quantity,
              local_efficiency: float, links: Dict[str, Quantity],
              collectives_per_layer: int = 2) -> TensorParallelLayerResult:
        flops = Q_(2 * (model.parameters.to(param).magnitude / model.layers) * tokens, "flop")
        compute = (flops / tp_size / (peak_flops * local_efficiency)).to("ms")
        layer_bytes = (collectives_per_layer * tokens * model.hidden_dim * BYTES_FP16
                       * calc_ring_allreduce_data_factor(tp_size))
        return TensorParallelLayerResult(
            layer_flops=flops, compute_time=compute, layer_bytes=layer_bytes,
            transfer_times={k: (layer_bytes / bw).to("ms") for k, bw in links.items()},
        )


class AlignmentMemoryModel(ForwardModel):
    """
    Model state of PPO-style RLHF against DPO, and its split over a model-parallel group.

    PPO trains a policy and a value model (16 B per parameter of mixed-precision
    Adam each) and runs a frozen reference copy of the policy and a reward model
    forward only (16-bit weights). DPO keeps only the policy and the reference.
    During rollout the policy also holds a KV cache for every generated sequence.
    """
    requires = ("policy", "critic")
    produces = AlignmentMemoryResult

    def solve(self, policy: TransformerWorkload, critic: Workload, tp_size: int, pp_size: int,
              rollout_sequences: int, rollout_tokens: int,
              device_memory: Optional[Quantity] = None) -> AlignmentMemoryResult:
        state_bpp, _ = _adam_bytes()
        infer = BYTES_FP16 / param
        pol = memory_from_params(policy.parameters, state_bpp)
        ref = memory_from_params(policy.parameters, infer)
        rew = memory_from_params(critic.parameters, infer)
        val = memory_from_params(critic.parameters, state_bpp)
        ppo = pol + ref + rew + val
        dpo = pol + ref
        n = tp_size * pp_size
        kv = policy.get_kv_cache_size(rollout_tokens, rollout_sequences, BYTES_FP16)
        min_dev = None
        if device_memory is not None:
            min_dev = math.ceil((ppo / device_memory).to("").magnitude)
        return AlignmentMemoryResult(
            policy_state=pol, reference_weights=ref, reward_weights=rew, value_state=val,
            ppo_total=ppo, dpo_total=dpo, dpo_reduction=((ppo - dpo) / ppo).to("").magnitude,
            model_parallel_devices=n, policy_per_device=pol / n, reference_per_device=ref / n,
            kv_cache=kv, kv_per_device=kv / n, min_devices_for_ppo=min_dev,
        )
