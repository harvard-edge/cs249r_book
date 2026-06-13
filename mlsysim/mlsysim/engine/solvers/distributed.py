"""Distributed training, routing, topology, and parallelism search solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

import math
from typing import Optional

from ..engine import Engine
from ..results import (
    DistributedResult,
    MoERoutingResult,
    TopologyResult,
    ParallelismOptimizerResult,
)
from ...physics import (
    calc_ring_allreduce_time,
    calc_hierarchical_allreduce_time,
    calc_all_to_all_time,
    calc_pipeline_bubble,
)
from ...core.units import ureg, Q_, resolve_precision
from ...models.types import Workload, SparseTransformerWorkload
from ...systems.types import Fleet, NetworkFabric
from .base import BaseOptimizer, ForwardModel
from .utils import _inter_node_latency, _intra_node_latency

class DistributedModel(ForwardModel):
    """
    Resolves fleet-wide communication, synchronization, and pipelining constraints.

    This model analyzes the constraints of distributed scale for distributed training. It
    decomposes a workload across a cluster using 3D Parallelism (DP, TP, PP)
    and calculates the resulting communication overheads and idle times
    (bubbles) that determine the Model FLOPs Utilization (MFU).

    Literature Source:
    1. Shoeybi et al. (2019), "Megatron-LM: Training Multi-Billion Parameter
       Language Models Using Model Parallelism." (3D Parallelism Framework)
    2. Narayanan et al. (2019), "PipeDream: Efficient Pipeline Parallelism for
       Training Large Models." (1F1B Pipeline Bubble Model)
    3. Patarasuk & Mueller (2009), "Bandwidth-Optimal All-Reduce Algorithms
       for Clusters of Workstations." (Ring All-Reduce)

    Formula contract:
    - split total accelerators into TP * PP * EP model-parallel groups and DP
      replicas; the split must be exact.
    - local step time comes from Engine.solve on the per-DP local batch.
    - exposed communication = DP gradient collective + TP activation collective
      + EP token all-to-all, optionally reduced by overlap.
    - total step latency = local compute + exposed communication + pipeline
      bubble, adjusted by explicit congestion and straggler multipliers.
    """
    requires = ("workload", "fleet")
    produces = DistributedResult
    _fallacies = {
        "Doubling GPUs halves training time": "Reality: communication overhead grows with N. Amdahl's Law always applies — there is a serial fraction (AllReduce, pipeline bubble) that limits speedup.",
        "Tensor parallelism is free within a node": "Reality: TP requires 2 AllReduce operations per layer on activations over NVLink. At TP=8 on H100, this can consume 10-20% of step time.",
        "More data parallelism is always better": "Reality: beyond the critical batch size (McCandlish et al. 2018), additional DP provides diminishing returns in convergence per step.",
    }

    def solve(self,
              model: Workload,
              fleet: Fleet,
              batch_size: int = 1,
              precision: str = "fp16",
              efficiency: float = 0.5,
              tp_size: int = 1,
              pp_size: int = 1,
              ep_size: int = 1,
              v_stages: int = 1,
              microbatch_count: int = 1,
              topology_override: Optional[str] = None,
              zero_stage: int = 0,
              is_lora: bool = False,
              activation_recomputation: bool = False,
              overlap_comm: bool = False,
              overlap_efficiency: float = 0.85,
              congestion_factor: float = 1.0,
              straggler_factor: float = 1.0,
              moe_routing_imbalance_factor: float = 1.0,
              gradient_accumulation_steps: int = 1,
              seq_len: int = 2048) -> DistributedResult:
        """
        Calculates distributed training performance using the 3D/4D Parallelism model.

        Parameters
        ----------
        model : Workload
            The model architecture to analyze.
        fleet : Fleet
            The hardware cluster and network topology.
        batch_size : int
            Global batch size.
        precision : str
            Numerical precision (fp16, fp32, int8).
        efficiency : float
            Achieved compute efficiency (0.0 to 1.0).
        tp_size : int
            Tensor Parallelism degree. Splits individual layers across GPUs,
            usually within a single node over high-speed NVLink.
        pp_size : int
            Pipeline Parallelism degree. Chains model layers across multiple
            nodes, introducing 'pipeline bubbles' while saving memory.
        ep_size : int
            Expert Parallelism degree for MoE models. Introduces All-to-All
            communication overhead across nodes.
        v_stages : int
            Number of virtual stages for interleaved pipeline schedules.
        microbatch_count : int
            Number of microbatches (M). Increasing M reduces the pipeline
            bubble but increases synchronization overhead.
        topology_override : str, optional
            Force a specific topology (ring, tree).
        zero_stage : int
            ZeRO optimization stage (0, 1, 2, 3) for sharding memory and altering DP comms.
        is_lora : bool
            Whether using Low-Rank Adaptation (PEFT).
        activation_recomputation : bool
            Whether to trade FLOPS (+33%) for activation memory savings.
        overlap_comm : bool
            Whether to overlap DP communication with backward pass compute.
        overlap_efficiency : float
            Fraction of communication hidden behind compute (0.0-1.0).
            Default 0.85 reflects typical Megatron-LM overlap efficiency.
        congestion_factor : float
            Multiplicative factor on communication time to account for
            network congestion (1.0 = ideal, 1.5-2.0 = shared fabric,
            2.0-3.0 = oversubscribed multi-tenant).
        moe_routing_imbalance_factor : float
            Multiplier on routed MoE token traffic. A value of 1.0 is perfectly
            balanced routing; values above 1.0 approximate hot experts.
        seq_len : int
            Sequence length for memory calculation.

        Returns
        -------
        DistributedResult
            Metrics including DP/TP/EP latency, the Pipeline Bubble penalty,
            and the final Scaling Efficiency.
        """
        # 0. Input validation
        from ...core._validation import validate_at_least, validate_range
        validate_at_least(tp_size, 1, "tp_size")
        validate_at_least(pp_size, 1, "pp_size")
        validate_at_least(ep_size, 1, "ep_size")
        validate_at_least(batch_size, 1, "batch_size")
        validate_at_least(gradient_accumulation_steps, 1, "gradient_accumulation_steps")
        validate_range(efficiency, 1e-9, 1.0, "efficiency")
        validate_range(overlap_efficiency, 0.0, 1.0, "overlap_efficiency")
        validate_at_least(moe_routing_imbalance_factor, 1.0, "moe_routing_imbalance_factor")

        # 1. 3D/4D Parallelism Decomposition
        n_accelerators = fleet.total_accelerators
        parallel_group_size = tp_size * pp_size * ep_size

        # The remaining accelerators become data-parallel replicas. Require an
        # exact split so invalid TP/PP/EP choices do not silently drop hardware.
        if parallel_group_size > n_accelerators:
            raise ValueError(f"Infeasible 4D Parallelism: TP({tp_size}) * PP({pp_size}) * EP({ep_size}) > Total({n_accelerators})")
        if n_accelerators % parallel_group_size != 0:
            raise ValueError(
                f"Infeasible 4D Parallelism: TP({tp_size}) * PP({pp_size}) * EP({ep_size}) "
                f"must divide total accelerators ({n_accelerators}) exactly."
            )
        dp_size = n_accelerators // parallel_group_size

        # 2. Single Node Performance (Computation)
        # Global batch is divided by DP size (TP/PP/EP split the model, not the batch).
        local_batch = max(1, batch_size // dp_size)
        if batch_size < dp_size:
            import warnings
            warnings.warn(
                f"batch_size ({batch_size}) < dp_size ({dp_size}): "
                f"some ranks will be idle. Using local_batch=1.",
                stacklevel=2,
            )
        node_perf = Engine.solve(
            model, fleet.node.accelerator, batch_size=local_batch,
            precision=precision, efficiency=efficiency,
            is_training=True, seq_len=seq_len, zero_stage=zero_stage,
            dp_size=dp_size, is_lora=is_lora,
            activation_recomputation=activation_recomputation
        )

        # 3. Communication Overhead (Network)
        # DP AllReduce exchanges gradients, which equal model size in the active precision.
        # With TP, each rank holds 1/tp_size of the model, so gradient buffer is smaller.
        # With ZeRO-1/2, AllReduce is replaced by Reduce-Scatter and All-Gather (same total volume but different patterns).
        # Resolve the precision after Engine.solve so communication uses the
        # same canonical key and byte width as the local compute pass.
        precision, precision_bytes = resolve_precision(precision)
        gradient_size = model.size_in_bytes(precision_bytes) / tp_size
        if is_lora:
            gradient_size = gradient_size * 0.01

        # DP Communication
        if dp_size > 1:
            if fleet.node.accelerators_per_node > 1 and dp_size > fleet.node.accelerators_per_node:
                # Hierarchical: Ring within node, then Ring across nodes
                t_comm_dp = calc_hierarchical_allreduce_time(
                    message_bytes=gradient_size,
                    # ceil, not floor: 12 ranks on 8-GPU nodes span 2 nodes;
                    # flooring to 1 modeled ZERO inter-node traffic (audit
                    # fix 2026-06-06).
                    n_nodes=math.ceil(dp_size / fleet.node.accelerators_per_node),
                    gpus_per_node=fleet.node.accelerators_per_node,
                    intra_node_bw=fleet.node.intra_node_bw,
                    inter_node_bw=fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio,
                    inter_node_lat=_inter_node_latency(fleet.fabric)
                )
            else:
                # Single node or small DP: Intra-node only
                t_comm_dp = calc_ring_allreduce_time(
                    gradient_size,
                    dp_size,
                    fleet.node.intra_node_bw,
                    _intra_node_latency(fleet.node)
                )
        else:
            t_comm_dp = Q_("0 ms")

        # TP Communication (activation AllReduce, intra-node NVLink)
        # TP requires 2 AllReduce ops per transformer layer on activations:
        # one after the column-parallel MLP, one after the row-parallel attention.
        # Volume per AllReduce = batch_size * seq_len * hidden_dim * precision_bytes
        # Source: Shoeybi et al. (2019), "Megatron-LM"
        # NOTE: This analytical model evaluates TP AllReduces as sequential to establish
        # a conservative upper bound. In practice, modern frameworks overlap this
        # communication with the next layer's computation to reduce exposed latency.
        if tp_size > 1:
            _, bpp = resolve_precision(precision)
            hidden_dim = getattr(model, 'hidden_dim', 4096) or 4096
            n_layers = getattr(model, 'layers', 1) or 1
            activation_bytes_per_allreduce = local_batch * seq_len * hidden_dim * bpp.magnitude
            # 2 AllReduces per layer (attention + MLP)
            tp_volume = 2 * n_layers * activation_bytes_per_allreduce * ureg.byte
            # Select bandwidth: NVLink if TP fits within a node, IB if it spans nodes
            if tp_size <= fleet.node.accelerators_per_node:
                tp_bw = fleet.node.intra_node_bw
                tp_lat = _intra_node_latency(fleet.node)
            else:
                tp_bw = fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio
                tp_lat = _inter_node_latency(fleet.fabric)
            t_comm_tp = calc_ring_allreduce_time(
                tp_volume / n_layers,  # per-layer volume
                tp_size,
                tp_bw,
                tp_lat
            ) * n_layers  # total across all layers (sequential, conservative upper bound)
        else:
            t_comm_tp = Q_("0 ms")

        # EP Communication (All-to-All token routing for MoE)
        # EP exchanges routed tokens, not the full model. Volume depends on
        # batch_size, seq_len, hidden_dim, and routing factor (top_k / num_experts).
        # Source: Fedus et al. (2022), "Switch Transformers"
        if ep_size > 1:
            _, bpp = resolve_precision(precision)
            hidden_dim = getattr(model, 'hidden_dim', 4096) or 4096
            top_k = getattr(model, 'active_experts_per_token', 1) or 1
            # Each token sends hidden_dim activations to its assigned expert(s)
            # Approximate: each GPU sends (ep_size-1)/ep_size of its tokens to others
            token_volume = (
                local_batch
                * seq_len
                * hidden_dim
                * bpp.magnitude
                * top_k
                * moe_routing_imbalance_factor
                * ureg.byte
            )
            t_comm_ep = calc_all_to_all_time(
                message_bytes=token_volume,
                n_gpus=ep_size,
                bandwidth_bytes_s=fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio,
                latency_s=_inter_node_latency(fleet.fabric)
            )
        else:
            t_comm_ep = Q_("0 ms")

        # 4. Pipeline Parallelism (PP) Bubble
        # Source: Narayanan et al. (2019), "PipeDream: Efficient Pipeline Parallelism"
        # Supports interleaved 1F1B schedules via v_stages
        bubble_fraction = calc_pipeline_bubble(pp_size, microbatch_count, v_stages=v_stages)
        t_bubble = (node_perf.latency * bubble_fraction) if pp_size > 1 else Q_("0 ms")

        # 5. Total Latency and Scaling Efficiency
        # Apply congestion factor to all communication
        t_comm_dp = t_comm_dp * congestion_factor
        t_comm_tp = t_comm_tp * congestion_factor
        t_comm_ep = t_comm_ep * congestion_factor
        total_comm_latency = t_comm_dp + t_comm_tp + t_comm_ep

        # Gradient accumulation: communication happens once per N micro-steps,
        # so amortize DP communication cost across accumulation steps.
        if gradient_accumulation_steps > 1:
            t_comm_dp = t_comm_dp / gradient_accumulation_steps
            # Update total_comm_latency to reflect the amortized DP cost
            total_comm_latency = t_comm_dp + t_comm_tp + t_comm_ep

        if overlap_comm:
            # Overlap DP communication with backward pass compute.
            # overlap_efficiency=1.0 means perfect overlap (theoretical best).
            # Default 0.85 reflects real framework behavior (Megatron-LM).
            exposed_dp_latency = t_comm_dp * (1 - overlap_efficiency)
            step_latency_total = node_perf.latency + exposed_dp_latency + t_comm_tp + t_comm_ep + t_bubble
        else:
            step_latency_total = node_perf.latency + total_comm_latency + t_bubble

        # Straggler effect: BSP training is gated by the slowest worker.
        # Source: Dean & Barroso (2013), "The Tail at Scale"
        if straggler_factor > 1.0:
            step_latency_total = step_latency_total * straggler_factor

        scaling_efficiency = (node_perf.latency / step_latency_total).magnitude

        constraint_trace = [
            (
                f"Distributed Wall: dp={dp_size}, tp={tp_size}, pp={pp_size}, ep={ep_size}; "
                f"step latency {step_latency_total.to('ms'):~P}."
            )
        ]
        if ep_size > 1 and moe_routing_imbalance_factor > 1.0:
            constraint_trace.append(
                f"MoE Routing: expert-parallel traffic inflated by {moe_routing_imbalance_factor:.2f}x for hot-expert imbalance."
            )

        return DistributedResult(
            constraint_trace=constraint_trace,
            node_profile=node_perf,
            dp_communication_latency=t_comm_dp,
            tp_communication_latency=t_comm_tp,
            ep_communication_latency=t_comm_ep,
            communication_latency=total_comm_latency,
            pipeline_bubble_latency=t_bubble,
            bubble_fraction=bubble_fraction,
            step_latency_total=step_latency_total,
            scaling_efficiency=scaling_efficiency,
            effective_throughput=(n_accelerators * node_perf.throughput * scaling_efficiency),
            parallelism={"dp": dp_size, "tp": tp_size, "pp": pp_size, "ep": ep_size},
        )

class MoERoutingModel(ForwardModel):
    """
    Models first-order MoE routing imbalance and expert-parallel all-to-all cost.

    Sparse models decouple memory from compute, but routing is rarely perfectly
    balanced. This model keeps the abstraction small: a single imbalance factor
    inflates the effective active experts and the routed-token communication
    volume. It does not simulate a router or token dispatcher.
    """
    requires = ("workload",)
    produces = MoERoutingResult

    def solve(
        self,
        model: SparseTransformerWorkload,
        batch_size: int,
        seq_len: int,
        precision: str = "fp16",
        ep_size: int = 1,
        routing_imbalance_factor: float = 1.0,
        fleet: Optional[Fleet] = None,
    ) -> MoERoutingResult:
        """Estimate effective active parameters and optional EP all-to-all latency.

        Single-factor imbalance model: a router that overloads hot experts
        behaves as if each token activated
        ``top_k * routing_imbalance_factor`` experts (capped at the model's
        expert count), inflating active parameters and routed traffic by the
        same multiplier. The routed payload per step is
        ``batch * seq_len * hidden_dim * bytes_per_param * top_k *
        multiplier`` bytes, of which the fraction ``(ep_size - 1) / ep_size``
        crosses ranks (a token's target expert is local with probability
        1/ep_size under uniform placement).

        Parameters
        ----------
        model : SparseTransformerWorkload
            MoE workload (experts, active_experts_per_token, hidden_dim,
            active_parameters).
        batch_size : int
            Sequences per step (>= 1).
        seq_len : int
            Tokens per sequence (>= 1).
        precision : str
            Numerical precision; sets bytes per routed activation element.
        ep_size : int
            Expert-parallel group size (1 = no expert parallelism, no
            all-to-all cost).
        routing_imbalance_factor : float
            >= 1.0; 1.0 is perfectly balanced routing, larger values
            approximate hot experts.
        fleet : Fleet, optional
            If given (and ep_size > 1), prices the all-to-all exchange over
            the fleet fabric (bandwidth deflated by oversubscription).

        Returns
        -------
        MoERoutingResult
            Effective active experts/parameters, the dispatch volume (MB),
            and the all-to-all latency (ms) when a fleet is provided.
        """
        from ...core._validation import validate_at_least, validate_range

        precision, precision_bytes = resolve_precision(precision)
        validate_at_least(batch_size, 1, "batch_size")
        validate_at_least(seq_len, 1, "seq_len")
        validate_at_least(ep_size, 1, "ep_size")
        validate_range(routing_imbalance_factor, 1.0, float("inf"), "routing_imbalance_factor")

        top_k = max(1, model.active_experts_per_token)
        effective_active_experts = min(float(model.experts), top_k * routing_imbalance_factor)
        active_parameter_multiplier = effective_active_experts / top_k
        effective_active_parameters = model.active_parameters * active_parameter_multiplier

        bpp = precision_bytes.to(ureg.byte).magnitude
        hidden_dim = model.hidden_dim or 4096
        routed_fraction = (ep_size - 1) / ep_size if ep_size > 1 else 0.0
        routing_payload_bytes = batch_size * seq_len * hidden_dim * bpp * top_k * active_parameter_multiplier * ureg.byte
        token_dispatch_bytes = routing_payload_bytes * routed_fraction

        all_to_all_latency = None
        if fleet is not None and ep_size > 1:
            all_to_all_latency = calc_all_to_all_time(
                message_bytes=routing_payload_bytes,
                n_gpus=ep_size,
                bandwidth_bytes_s=fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio,
                latency_s=_inter_node_latency(fleet.fabric),
            )

        trace = [
            (
                f"MoE Routing: top-{top_k} routing with imbalance {routing_imbalance_factor:.2f} "
                f"acts like {effective_active_experts:.2f} active experts per token."
            )
        ]
        if all_to_all_latency is not None:
            trace.append(f"Expert Parallelism: all-to-all latency is {all_to_all_latency.to('ms'):~P}.")

        return MoERoutingResult(
            constraint_trace=trace,
            effective_active_experts=effective_active_experts,
            effective_active_parameters=effective_active_parameters.to(ureg.count),
            active_parameter_multiplier=active_parameter_multiplier,
            routing_imbalance_factor=routing_imbalance_factor,
            token_dispatch_bytes=token_dispatch_bytes.to("MB"),
            all_to_all_latency=all_to_all_latency.to("ms") if all_to_all_latency is not None else None,
            ep_size=ep_size,
        )

class TopologyModel(ForwardModel):
    """
    Models bisection bandwidth for different network topologies (Wall 10).

    This model calculates the effective bandwidth available to collective
    communication operations based on the physical network topology. Different
    topologies trade cost against bisection bandwidth — the minimum bandwidth
    across any cut that divides the network in half.

    Literature Source:
    1. Leiserson (1985), "Fat-Trees: Universal Networks for Hardware-Efficient
       Supercomputing." (Fat-tree bisection bandwidth.)
    2. Kim et al. (2008), "Technology-Driven, Highly-Scalable Dragonfly Topology."
       (Dragonfly topology model.)
    3. Dally & Towles (2003), "Principles and Practices of Interconnection
       Networks." (Torus and ring analysis.)
    """
    requires = ("fabric",)
    produces = TopologyResult

    # Bisection bandwidth fractions (β) relative to full fat-tree.
    # Ring and torus_3d are computed dynamically from num_nodes in solve().
    TOPOLOGY_BETA = {
        "fat_tree": 1.0,
        "dragonfly": 0.85,
    }

    # Average hop counts (normalized to diameter)
    TOPOLOGY_HOPS = {
        "fat_tree": 4,       # 2 up + 2 down in 3-level fat-tree
        "dragonfly": 3,      # Local + global + local
        "torus_3d": None,    # Computed from num_nodes
        "ring": None,         # Computed from num_nodes
    }

    def solve(self, fabric: NetworkFabric, topology: str = "fat_tree",
              num_nodes: int = 64) -> TopologyResult:
        """
        Solves for effective network bandwidth under a given topology.

        Parameters
        ----------
        fabric : NetworkFabric
            The network fabric specification (link bandwidth, oversubscription).
        topology : str
            Network topology ('fat_tree', 'dragonfly', 'torus_3d', 'ring').
        num_nodes : int
            Number of nodes in the network.

        Returns
        -------
        TopologyResult
            Effective bandwidth, bisection fraction, and average hops.
        """
        # Compute bisection bandwidth fraction β.
        # Ring: bisection cuts 2 links out of N, so β = 2/N.
        # 3D torus: bisection cuts 2*N^(2/3) links, so β ≈ 2*N^(-1/3).
        if topology == "ring":
            beta = 2.0 / max(num_nodes, 1)
        elif topology == "torus_3d":
            beta = 2.0 * (num_nodes ** (-1.0 / 3.0))
        else:
            beta = self.TOPOLOGY_BETA.get(topology, 1.0)

        link_bw = fabric.bandwidth
        oversubscription = fabric.oversubscription_ratio

        effective_bw = (link_bw * beta / oversubscription).to("GB/s")
        # bisection_bw is reported relative to a non-blocking fat-tree reference
        # cut of N/2 links (beta=1.0): it is `beta * (N/2)` link-equivalents.
        # So a ring (beta=2/N) reports 1 link-equivalent here even though its
        # physical mid-cut severs 2 links, and a 3D torus reports N^(2/3) vs its
        # 2*N^(2/3) physical cut. The dimensionless `beta` (bisection_bw_fraction)
        # is the apples-to-apples cross-topology metric; the absolute figure
        # follows this fat-tree-normalized convention by design.
        bisection_bw = (link_bw * beta * num_nodes / 2 / oversubscription).to("GB/s")

        # Average hops
        fixed_hops = self.TOPOLOGY_HOPS.get(topology)
        if fixed_hops is not None:
            hops_avg = fixed_hops
        elif topology == "torus_3d":
            # 3D torus: avg hops ≈ 3/4 × (N^(1/3))
            hops_avg = 0.75 * (num_nodes ** (1.0 / 3.0))
        elif topology == "ring":
            # Ring: avg hops ≈ N/4
            hops_avg = num_nodes / 4.0
        else:
            hops_avg = 4  # Default to fat-tree

        return TopologyResult(
            effective_bw=effective_bw,
            bisection_bw=bisection_bw,
            bisection_bw_fraction=beta,
            hops_avg=hops_avg,
            topology=topology,
            num_nodes=num_nodes,
        )

class ParallelismOptimizer(BaseOptimizer):
    """
    Searches for the optimal 3D/4D parallelism split (DP, TP, PP, EP).

    Given a model architecture and a cluster size, this optimizer sweeps
    the integer design space of parallelism degrees to find the
    configuration that maximizes Model FLOPs Utilization (MFU).

    Literature Source:
    1. Narayanan et al. (2021), "Efficient Large-Scale Language Model
       Training on GPU Clusters Using Megatron-LM."
    """
    requires = ("workload", "fleet")
    produces = ParallelismOptimizerResult

    def solve(self, model: Workload, fleet: Fleet, batch_size: int,
              precision: str = "fp16", efficiency: float = 0.5,
              max_tp: Optional[int] = None, max_pp: Optional[int] = None,
              overlap_comm: bool = True) -> ParallelismOptimizerResult:
        """
        Search for the parallelism split (TP x PP x DP) that maximizes MFU.

        Exhaustive sweep over power-of-two TP and PP degrees with
        ``TP * PP * DP = total_accelerators``. TP is capped at the node size
        by default (tensor parallelism needs NVLink-class bandwidth).
        Candidates are pre-screened for memory feasibility — per-GPU weights
        plus same-size gradients must fit in 90% of HBM capacity, with
        weights sharded by TP*PP — then each survivor is priced with
        ``DistributedModel`` and ranked by
        ``MFU = scaling_efficiency * efficiency``.

        Parameters
        ----------
        model : Workload
            The model to train.
        fleet : Fleet
            The cluster (total accelerators, node size, fabric).
        batch_size : int
            Global batch size in samples.
        precision : str
            Numerical precision.
        efficiency : float
            Single-node software efficiency in (0, 1]; multiplied by the
            distributed scaling efficiency to form the MFU objective.
        max_tp : int, optional
            Cap on tensor-parallel degree (default: accelerators per node).
        max_pp : int, optional
            Cap on pipeline-parallel degree (default: cluster size).
        overlap_comm : bool
            Whether DP communication overlaps the backward pass.

        Returns
        -------
        ParallelismOptimizerResult
            Best {tp, pp, dp} config with its MFU, throughput, step time,
            and the top-5 candidate list.

        Raises
        ------
        ValueError
            If no factorization is memory-feasible for this cluster.
        """
        n_gpus = fleet.total_accelerators
        gpus_per_node = fleet.node.accelerators_per_node

        # 1. Generate search space (all valid factorizations of n_gpus)
        # We simplify to: TP * PP * DP = total_gpus (EP=1 for now)
        candidates = []

        # Heuristic constraints
        limit_tp = max_tp or gpus_per_node
        limit_pp = max_pp or n_gpus

        dist_model = DistributedModel()

        # Discrete search
        tp_options = [2**i for i in range(10) if 2**i <= limit_tp]
        pp_options = [2**i for i in range(10) if 2**i <= limit_pp]

        for tp in tp_options:
            for pp in pp_options:
                if (tp * pp) > n_gpus: continue
                if n_gpus % (tp * pp) != 0: continue

                dp = n_gpus // (tp * pp)

                try:
                    # Memory feasibility: per-GPU training state must fit in HBM.
                    # TP shards weights across tp GPUs; PP shards layers across pp
                    # stages. The state is weights + gradients (same width) + Adam
                    # optimizer state (FP32 master + moments, ~12 B/param =
                    # ~6x the FP16 weight bytes), sharded by model parallelism to
                    # match the zero_stage=0 convention DistributedModel.solve uses
                    # below. Activations are intentionally EXCLUDED: this optimizer
                    # searches the parallelism split, not the microbatch /
                    # gradient-accumulation that sizes activations, so it cannot
                    # bound that term without exploding at the full global batch.
                    # (The previous gate counted only weights+gradients and so
                    # admitted configs that OOM the instant optimizer state is
                    # allocated — the dominant training-memory term for Adam.)
                    per_gpu_weights = model.size_in_bytes() / tp / pp
                    per_gpu_grads = per_gpu_weights  # gradients same width as weights
                    try:
                        from .. import calibration as cal
                        n_params = model.parameters.to(ureg.count).magnitude
                        per_gpu_opt = (n_params * cal.TRAINING_OPTIMIZER_BYTES_ADAM / (tp * pp)) * ureg.byte
                    except Exception:
                        per_gpu_opt = per_gpu_weights * 6.0  # Adam ~6x FP16 weight bytes
                    per_gpu_mem = per_gpu_weights + per_gpu_grads + per_gpu_opt
                    gpu_capacity = fleet.node.accelerator.memory.capacity
                    if per_gpu_mem > gpu_capacity * 0.9:
                        continue  # Infeasible: training state does not fit in GPU memory

                    # Evaluate this config
                    res = dist_model.solve(
                        model, fleet, batch_size=batch_size,
                        precision=precision, efficiency=efficiency,
                        tp_size=tp, pp_size=pp, overlap_comm=overlap_comm
                    )

                    # Store candidate
                    candidates.append({
                        "config": {"tp": tp, "pp": pp, "dp": dp},
                        "mfu": (res.scaling_efficiency * efficiency),
                        "throughput": res.effective_throughput,
                        "step_time": res.step_latency_total
                    })
                except Exception:
                    continue

        if not candidates:
            raise ValueError("No valid parallelism configurations found for this cluster size.")

        # 2. Find best
        best = max(candidates, key=lambda x: x["mfu"])

        # Sort candidates for top list
        top_n = sorted(candidates, key=lambda x: x["mfu"], reverse=True)[:5]

        return ParallelismOptimizerResult(
            objective_value=best["mfu"],
            best_config=best["config"],
            best_mfu=best["mfu"],
            best_throughput=best["throughput"],
            best_step_time=best["step_time"],
            total_searched=len(candidates),
            search_space_size=len(candidates),
            top_candidates=top_n
        )

__all__ = [
    "DistributedModel",
    "MoERoutingModel",
    "TopologyModel",
    "ParallelismOptimizer",
]
