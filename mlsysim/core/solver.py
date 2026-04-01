import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ConfigDict
from .engine import Engine, PerformanceProfile
from .results import (
    SolverResult,
    DistributedResult, ReliabilityResult, CheckpointResult, SustainabilityResult,
    ServingResult, ContinuousBatchingResult, WeightStreamingResult, TailLatencyResult, 
    EconomicsResult, DataResult, TopologyResult,
    EfficiencyResult, TransformationResult, ScalingResult,
    CompressionResult, SynthesisResult, OrchestrationResult,
    InferenceScalingResult, SensitivityResult, ResponsibleEngineeringResult,
    OptimizerResult, ParallelismOptimizerResult,
    BatchingOptimizerResult, PlacementOptimizerResult
)
from .formulas import (
    calc_ring_allreduce_time,
    calc_tree_allreduce_time,
    calc_hierarchical_allreduce_time,
    calc_all_to_all_time,
    calc_bottleneck,
    calc_mtbf_cluster,
    calc_mtbf_node,
    calc_young_daly_interval,
    calc_failure_probability,
    calc_pipeline_bubble
)
from .constants import (
    ureg, Q_, PRECISION_MAP,
    BYTES_FP16, BYTES_FP32, BYTES_INT8, BYTES_INT4,
    LATENCY_INFINIBAND, LATENCY_NVLINK,
    GPU_MTTF_HOURS, H100_TDP,
    CHINCHILLA_TOKENS_PER_PARAM, CHINCHILLA_COMPUTE_CONSTANT,
    H100_FLOPS_FP16_TENSOR,
    CLOUD_ELECTRICITY_PER_KWH,
)
from .defaults import (
    GPU_UNIT_COST_H100, ANNUAL_MAINTENANCE_RATIO,
    MFU_TRAINING_LOW, MFU_TRAINING_HIGH, MFU_INFERENCE_BATCH1, MFU_INFERENCE_BATCHED,
    QUANT_ACCURACY_DELTA_INT8, QUANT_ACCURACY_DELTA_INT4, QUANT_ACCURACY_DELTA_FP8,
    PRUNING_ACCURACY_THRESHOLD, PRUNING_MILD_DELTA, PRUNING_STEEP_COEFFICIENT, PRUNING_STEEP_EXPONENT,
    NIC_MTTF_HOURS, PSU_MTTF_HOURS,
    MFU_FLASH_ATTENTION, MFU_FLASH_ATTENTION_CAP, MFU_FFN_CAP, MFU_CONV_CAP, HFU_MFU_RATIO,
    TOKENS_PER_REASONING_STEP, REFERENCE_MFU_SUSTAINED, DP_SGD_SLOWDOWN_COEFFICIENT,
)
from .types import Quantity
from ..models.types import Workload, TransformerWorkload
from ..hardware.types import HardwareNode
from ..systems.types import Fleet, NetworkFabric
from ..infra.types import Datacenter, GridProfile

class BaseResolver(ABC):
    """Base class for all mlsysim analytical components (Models, Solvers, Optimizers).

    Each resolver declares its input requirements and output type.
    Taxonomic classification lives in ``core/walls.py``.
    """
    requires: tuple = ()
    produces: Optional[Type[SolverResult]] = None

    @abstractmethod
    def solve(self, *args, **kwargs) -> Any:
        pass

    @classmethod
    def schema(cls) -> dict:
        """Return a summary of this resolver's interface."""
        from .walls import walls_for_resolver
        wall_info = [
            {"number": w.number, "name": w.name, "domain": w.domain.value}
            for w in walls_for_resolver(cls.__name__)
        ]
        return {
            "resolver": cls.__name__,
            "type": cls.resolver_type(),
            "walls": wall_info,
            "requires": cls.requires,
            "produces": cls.produces.__name__ if cls.produces else "Any",
        }

    @classmethod
    def resolver_type(cls) -> str:
        if issubclass(cls, ForwardModel): return "model"
        if issubclass(cls, BaseSolver): return "solver"
        if issubclass(cls, BaseOptimizer): return "optimizer"
        return "unknown"

    # ── Fallacies and Pitfalls (Patterson & Hennessy tradition) ──
    _fallacies: Dict[str, str] = {}

    @classmethod
    def fallacies(cls) -> Dict[str, str]:
        """Return common fallacies and pitfalls for this solver.

        Following the Hennessy & Patterson tradition, each solver declares
        the misconceptions students most commonly hold about its domain.
        Call ``solver.fallacies()`` in notebooks for pedagogical discussion.
        """
        return cls._fallacies

class ForwardModel(BaseResolver):
    """Forward-evaluating mechanistic engine (Y = f(X))."""
    pass

# Backward-compatible alias (avoid Pydantic BaseModel name collision)
BaseModel = ForwardModel

class BaseSolver(BaseResolver):
    """Inverse-design or diagnostic engine (X = f^-1(Y) or grad f)."""
    pass

class BaseOptimizer(BaseResolver):
    """Design-space search engine (max f(X) s.t. g(X) < c)."""
    pass

class SingleNodeModel(BaseModel):
    """
    Resolves single-node hardware Roofline bounds and feasibility.

    This model handles the 'Iron Law' of machine learning systems,
    calculating whether a model fits in memory and predicting its
    throughput based on arithmetic intensity.

    Literature Source: Williams et al. (2009), "Roofline: An Insightful Visual
    Performance Model for Floating-Point Programs and Multicore Architectures."
    """
    requires = ("workload", "hardware")
    produces = PerformanceProfile
    _fallacies = {
        "Peak FLOPS determines training speed": "Reality: most ML workloads are memory-bandwidth-bound at small batch sizes. A GPU with 2x FLOPS but the same memory bandwidth gives < 2x speedup.",
        "Bigger GPU always means faster inference": "Reality: if the workload is memory-bound (batch=1 LLM decode), memory bandwidth matters more than compute. H100 has 1.7x the bandwidth of A100, not 3.2x the speedup.",
        "Model fits in memory = it will run well": "Reality: fitting in memory is necessary but not sufficient. The bottleneck may be bandwidth, compute, or framework overhead.",
    }

    def solve(self, model: Workload, hardware: HardwareNode, batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5, raise_errors: bool = False, **kwargs) -> PerformanceProfile:
        """
        Calculates the performance profile for a single hardware node.
        """
        return Engine.solve(model, hardware, batch_size=batch_size, precision=precision, efficiency=efficiency, raise_errors=raise_errors, **kwargs)

class DistributedModel(BaseModel):
    """
    Resolves fleet-wide communication, synchronization, and pipelining constraints.
    
    This model simulates the constraints of distributed scale for distributed training. It
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
              gradient_accumulation_steps: int = 1,
              seq_len: int = 2048) -> DistributedResult:
        """
        Calculates distributed training performance using the 3D/4D Parallelism model.

        Parameters
        ----------
        model : Workload
            The model architecture to simulate.
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
        seq_len : int
            Sequence length for memory calculation.

        Returns
        -------
        Dict[str, Any]
            Metrics including DP/TP/EP latency, the Pipeline Bubble penalty, 
            and the final Scaling Efficiency.
        """
        # 0. Input validation
        from ._validation import validate_at_least, validate_range
        validate_at_least(tp_size, 1, "tp_size")
        validate_at_least(pp_size, 1, "pp_size")
        validate_at_least(ep_size, 1, "ep_size")
        validate_at_least(batch_size, 1, "batch_size")
        validate_at_least(gradient_accumulation_steps, 1, "gradient_accumulation_steps")
        validate_range(efficiency, 1e-9, 1.0, "efficiency")
        validate_range(overlap_efficiency, 0.0, 1.0, "overlap_efficiency")

        # 1. 3D/4D Parallelism Decomposition
        n_accelerators = fleet.total_accelerators
        dp_size = n_accelerators // (tp_size * pp_size * ep_size)
        
        if dp_size < 1:
            raise ValueError(f"Infeasible 4D Parallelism: TP({tp_size}) * PP({pp_size}) * EP({ep_size}) > Total({n_accelerators})")

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
        gradient_size = model.size_in_bytes() / tp_size
        if is_lora:
            gradient_size = gradient_size * 0.01
        
        # DP Communication
        if dp_size > 1:
            if fleet.node.accelerators_per_node > 1 and dp_size > fleet.node.accelerators_per_node:
                # Hierarchical: Ring within node, then Ring across nodes
                t_comm_dp = calc_hierarchical_allreduce_time(
                    message_bytes=gradient_size,
                    n_nodes=dp_size // fleet.node.accelerators_per_node,
                    gpus_per_node=fleet.node.accelerators_per_node,
                    intra_node_bw=fleet.node.intra_node_bw,
                    inter_node_bw=fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio,
                    inter_node_lat=fleet.fabric.latency or LATENCY_INFINIBAND
                )
            else:
                # Single node or small DP: Intra-node only
                t_comm_dp = calc_ring_allreduce_time(
                    gradient_size,
                    dp_size,
                    fleet.node.intra_node_bw,
                    LATENCY_NVLINK
                )
        else:
            t_comm_dp = Q_("0 ms")

        # TP Communication (activation AllReduce, intra-node NVLink)
        # TP requires 2 AllReduce ops per transformer layer on activations:
        # one after the column-parallel MLP, one after the row-parallel attention.
        # Volume per AllReduce = batch_size * seq_len * hidden_dim * precision_bytes
        # Source: Shoeybi et al. (2019), "Megatron-LM"
        # NOTE: This models TP AllReduces as sequential (conservative upper bound).
        # Real frameworks pipeline TP communication with the next layer's compute,
        # reducing exposed latency. For v0.2.0: add tp_overlap_efficiency parameter.
        if tp_size > 1:
            bpp = PRECISION_MAP.get(precision, BYTES_FP16)
            hidden_dim = getattr(model, 'hidden_dim', 4096) or 4096
            n_layers = getattr(model, 'layers', 1) or 1
            activation_bytes_per_allreduce = local_batch * seq_len * hidden_dim * bpp.magnitude
            # 2 AllReduces per layer (attention + MLP)
            tp_volume = 2 * n_layers * activation_bytes_per_allreduce * ureg.byte
            # Select bandwidth: NVLink if TP fits within a node, IB if it spans nodes
            if tp_size <= fleet.node.accelerators_per_node:
                tp_bw = fleet.node.intra_node_bw
                tp_lat = LATENCY_NVLINK
            else:
                tp_bw = fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio
                tp_lat = fleet.fabric.latency or LATENCY_INFINIBAND
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
            bpp = PRECISION_MAP.get(precision, BYTES_FP16)
            hidden_dim = getattr(model, 'hidden_dim', 4096) or 4096
            # Each token sends hidden_dim activations to its assigned expert(s)
            # Approximate: each GPU sends (ep_size-1)/ep_size of its tokens to others
            token_volume = local_batch * seq_len * hidden_dim * bpp.magnitude * ureg.byte
            t_comm_ep = calc_all_to_all_time(
                message_bytes=token_volume,
                n_gpus=ep_size,
                bandwidth_bytes_s=fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio,
                latency_s=fleet.fabric.latency or LATENCY_INFINIBAND
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

        
        return DistributedResult(
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

class NetworkRooflineModel(BaseModel):
    """
    Analyzes the Distributed Performance Bounds (The Network Wall).
    
    This model elevates the single-node Roofline analysis to the fleet scale. 
    It calculates the Communication Intensity (CI) of a workload and identifies 
    if the fleet is Compute-Bound (Wall 1) or Network-Bound (Wall 2).

    Literature Source:
    1. Reddi et al. (2025), "Machine Learning Systems," Volume 2, Chapter 1.
    2. Williams et al. (2009), "Roofline Model." (Theoretical basis)
    3. Ghose et al. (2019), "A Survey of Communication-Efficient Distributed 
       Training."
    """
    requires = ("workload", "fleet")
    produces = PerformanceProfile # Reusing PerformanceProfile for consistency

    def solve(self, model: Workload, fleet: Fleet, precision: str = "fp16", efficiency: float = 0.5) -> PerformanceProfile:
        """
        Solves for the distributed performance bound.
        """
        # 1. Supply (Fleet Capacity)
        total_flops = (fleet.node.accelerator.compute.precision_flops.get(precision, fleet.node.accelerator.compute.peak_flops) * fleet.total_accelerators).to("TFLOPs/s")
        # Bisection Bandwidth (total data movement capacity per step)
        bisection_bw = (fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio).to("GB/s")
        
        # 2. Demand (Workload Characteristics)
        # For training, total ops = 6 * parameters * dataset_size_per_step
        # To normalize per-step, we use 1 sample.
        # But CI is better defined as Total Ops / Total Sync Bytes.
        # For Data Parallel sync: Bytes = 2 * Parameters * Precision_Bytes
        prec_bytes = PRECISION_MAP.get(precision, BYTES_FP16)
        sync_bytes = (2 * model.parameters.to(ureg.count).magnitude * prec_bytes.to(ureg.byte).magnitude) * ureg.byte
        
        # Total OPS for one training step (approx 6 * P FLOPs per sample)
        # We assume CI is independent of batch size for basic All-Reduce sync.
        training_ops = (6 * model.parameters.to(ureg.count).magnitude) * ureg.flop
        
        # CI = FLOPs / Byte
        ci = (training_ops / sync_bytes).to("flop/byte")
        
        # 3. Physics (The Ridge)
        # Ridge = Peak_FLOPs / Bisection_BW
        # Convert BW to Byte/s for units to cancel to FLOP/Byte
        network_ridge = (total_flops / bisection_bw.to("byte/s")).to("flop/byte")
        
        # 4. Results
        is_network_bound = ci < network_ridge
        achievable_flops = min(total_flops, bisection_bw.to("byte/s") * ci) * efficiency
        
        # Return results in a standardized PerformanceProfile
        return PerformanceProfile(
            latency=(training_ops / achievable_flops).to("ms") if achievable_flops.magnitude > 0 else Q_("0 ms"),
            throughput=(achievable_flops / training_ops).to("1/s"),
            arithmetic_intensity=ci,
            peak_flops=total_flops,
            achievable_flops=achievable_flops,
            bottleneck="Network-Bound" if is_network_bound else "Compute-Bound"
        )

class ReliabilityModel(BaseModel):
    """
    Calculates Mean Time Between Failures (MTBF) and optimal checkpointing intervals.
    
    This model handles the reliability modeling of massive clusters, helping
    determine the 'Goodput' of long-running training jobs. It identifies 
    the probability of a job failure before completion and calculates the 
    Young-Daly optimal interval to minimize wasted compute time.

    Literature Source:
    1. Young (1974), "A First-Order Approximation to the Optimum Checkpoint 
       Interval."
    2. Daly (2006), "A Higher Order Estimate of the Optimum Checkpoint 
       Interval for Restart-Dump Strategy."
    """
    requires = ("fleet",)
    produces = ReliabilityResult

    def solve(self, fleet: Fleet, job_duration_hours: float, checkpoint_time_s: float = 60.0,
              avg_recovery_time_s: float = 300.0) -> ReliabilityResult:
        """
        Calculates reliability and checkpointing metrics for a fleet.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        job_duration_hours : float
            Total job duration in hours.
        checkpoint_time_s : float
            Time to write one checkpoint in seconds (default 60s).
        avg_recovery_time_s : float
            Average time to recover from a failure in seconds (default 300s).
            Includes checkpoint reload, process restart, and re-warmup.
        """
        # Use compound node MTBF accounting for GPUs, NICs, and PSUs
        node_mtbf = calc_mtbf_node(
            gpu_mtbf_h=GPU_MTTF_HOURS, n_gpus=fleet.node.accelerators_per_node,
            nic_mtbf_h=NIC_MTTF_HOURS, n_nics=fleet.node.nics_per_node,
            psu_mtbf_h=PSU_MTTF_HOURS, n_psus=fleet.node.psus_per_node,
        )
        fleet_mtbf = calc_mtbf_cluster(node_mtbf, fleet.count)

        job_dur_q = Q_(job_duration_hours, "hour")
        prob_fail = calc_failure_probability(fleet_mtbf, job_dur_q)

        ckpt_time_q = Q_(checkpoint_time_s, "second")
        optimal_interval = calc_young_daly_interval(ckpt_time_q, fleet_mtbf.to("second"))

        # Goodput ratio: fraction of rawput that produces useful training progress.
        # Lost compute = P(failure) * (avg_recovery_time / checkpoint_interval)
        # Source: Narayanan et al. (2021), "Efficient Large-Scale Language Model Training"
        interval_s = optimal_interval.m_as(ureg.second)
        if interval_s > 0:
            goodput_ratio = max(0.0, 1.0 - prob_fail * (avg_recovery_time_s / interval_s))
        else:
            goodput_ratio = 0.0

        return ReliabilityResult(
            fleet_mtbf=fleet_mtbf,
            failure_probability=prob_fail,
            optimal_checkpoint_interval=optimal_interval,
            expected_failures=(job_dur_q / fleet_mtbf).magnitude,
            goodput_ratio=goodput_ratio,
        )

class CheckpointModel(BaseModel):
    """
    Analyzes the storage constraints and I/O burst penalties of saving model states.
    
    Training massive models requires saving hundreds of gigabytes (Weights + 
    Optimizer States) to persistent storage. This model calculates the time 
    spent blocked on I/O, subtracting from the cluster's Model FLOPs Utilization.

    Literature Source:
    1. Eisenman et al. (2022), "Check-N-Run: A Checkpointing System for 
       Training Large Language Models."
    """
    requires = ("workload", "hardware")
    produces = CheckpointResult

    def solve(self, model: Workload, hardware: HardwareNode, optimizer: str = "adam",
              checkpoint_interval_hours: float = 4.0, n_writers: int = 1,
              filesystem_limit_gbs: float = 500.0) -> CheckpointResult:
        """Solves for checkpoint size, write time, and resulting MFU penalty.

        Parameters
        ----------
        n_writers : int
            Number of parallel checkpoint writers (default 1). Distributed
            checkpointing (e.g., FSDP) shards the write across workers.
        filesystem_limit_gbs : float
            Maximum aggregate filesystem write bandwidth in GB/s (default 500).
            Prevents over-optimistic scaling when n_writers is large.
        """
        from .formulas import calc_checkpoint_size

        # Calculate size based on optimizer states
        # Mixed-precision Adam: 14 bytes/param (FP32 master + FP32 momentum + FP32 variance + FP16 weights)
        # Gradients are ephemeral and not checkpointed.
        if optimizer.lower() == "adam":
            bytes_per_param = 14
        else:
            bytes_per_param = 4  # e.g., SGD

        ckpt_size = calc_checkpoint_size(model.parameters, bytes_per_param=bytes_per_param)

        storage_bw = getattr(hardware.storage, 'bandwidth', Q_("0 GB/s")) if hardware.storage else Q_("0 GB/s")
        # Fallback to network or standard disk speed if undefined
        if storage_bw.magnitude == 0:
            storage_bw = Q_("1 GB/s")

        # Distributed writing: scale bandwidth by n_writers, capped by filesystem limit
        fs_limit = Q_(filesystem_limit_gbs, "GB/s")
        effective_write_bw = min(storage_bw * n_writers, fs_limit)

        t_write = (ckpt_size / effective_write_bw).to("second")
        
        # Calculate penalty to MFU
        interval_s = Q_(checkpoint_interval_hours, "hour").to("second")
        if interval_s.magnitude > 0:
            penalty_pct = (t_write / interval_s).magnitude
        else:
            penalty_pct = 1.0
            
        return CheckpointResult(
            checkpoint_size=ckpt_size.to("GB"),
            write_time_seconds=t_write,
            max_bandwidth_required=storage_bw,
            storage_bottleneck=t_write.m_as("second") > 60.0, # Flag if checkpoint takes > 1 min
            mfu_penalty_pct=penalty_pct
        )

class SustainabilityModel(BaseModel):
    """
    Calculates Datacenter-scale Sustainability metrics.
    
    Handles Power Usage Effectiveness (PUE), Carbon Intensity, 
    and Water Usage Effectiveness (WUE) across different regional grids.
    This model simulates the 'Infrastructure Tax' — the energy spent on 
    cooling and power delivery rather than on neural computation.

    Literature Source:
    1. Patterson et al. (2021), "Carbon Emissions and Large Neural Network 
       Training."
    2. Belkhir & Elmeligi (2018), "Assessing ICT Global Emissions Footprint."
    3. Wu et al. (2022), "Sustainable AI: Environmental Implications, 
       Challenges and Opportunities."
    """
    requires = ("fleet",)
    produces = SustainabilityResult

    def solve(self, fleet: Fleet, duration_days: float, datacenter: Optional[Datacenter] = None,
              mfu: float = 1.0, embodied_carbon_per_device: float = 0.0) -> SustainabilityResult:
        """
        Calculates energy, carbon, and water footprint for a fleet operation.
        """
        # 1. Resolve Environment
        dc = datacenter or fleet.datacenter
        
        # Flexibly handle if dc is already a GridProfile or a Datacenter
        if hasattr(dc, 'grid'):
            region = dc.grid
        else:
            region = dc or fleet.region
        
        if not region:
             from ..infra.registry import Grids
             region = Grids.US_Avg

        from ._validation import validate_range, validate_nonnegative
        validate_range(mfu, 0.0, 1.0, "mfu")
        validate_nonnegative(embodied_carbon_per_device, "embodied_carbon_per_device")

        duration_hours = duration_days * 24
        
        # 2. Power
        base_tdp = fleet.node.accelerator.tdp if fleet.node.accelerator.tdp else H100_TDP
        # Energy proportionality: Idle power is ~30% of TDP. Dynamic power scales with compute utilization (MFU).
        idle_power = base_tdp * 0.3
        dynamic_power = base_tdp * 0.7 * mfu
        effective_power_per_chip = idle_power + dynamic_power
        it_power_w = effective_power_per_chip * fleet.total_accelerators
            
        # 3. Energy Consumption
        it_energy_kwh = (it_power_w * Q_(duration_hours, "hour")).to("kWh")
        
        # Apply PUE
        pue = getattr(dc, 'pue', fleet.effective_pue)
        total_energy_kwh = it_energy_kwh * pue
        
        # 4. Carbon Footprint (use total facility energy, PUE already applied)
        carbon_kg = region.carbon_kg(total_energy_kwh.magnitude) if hasattr(region, 'carbon_kg') else total_energy_kwh.magnitude * (region.carbon_intensity_g_kwh / 1000.0)
        
        # 5. Water Usage
        # Resolve WUE from dc.grid, dc, or region
        if hasattr(dc, 'grid') and dc.grid:
            wue = dc.grid.wue
        elif hasattr(dc, 'wue'):
            wue = dc.wue
        else:
            wue = region.wue
            
        water_liters = total_energy_kwh.magnitude * wue

        # 6. Embodied Carbon (manufacturing, shipping, end-of-life)
        # Source: Gupta et al. (2022), "ACT: Designing Sustainable Computer Systems
        #         with an Architectural Carbon Modeling Tool"
        n_devices = fleet.total_accelerators
        embodied_kg = embodied_carbon_per_device * n_devices
        total_carbon_kg = carbon_kg + embodied_kg

        return SustainabilityResult(
            it_energy_kwh=it_energy_kwh,
            total_energy_kwh=total_energy_kwh,
            carbon_footprint_kg=total_carbon_kg,
            water_usage_liters=water_liters,
            pue=pue,
            region_name=region.name,
            embodied_carbon_kg=embodied_kg,
        )

class ServingModel(BaseModel):
    """
    Analyzes the two-phase LLM serving lifecycle: Pre-fill vs. Decoding.
    
    LLM inference is not a single mathematical operation; it is a stateful 
    process with two distinct physical regimes (Compute-bound Pre-fill and 
    Memory-bound Decoding).

    Literature Source:
    1. Pope et al. (2023), "LLM.int8(): 8-bit Matrix Multiplication for 
       Transformers at Scale" (Inference Bottlenecks)
    2. Aminabadi et al. (2022), "DeepSpeed-Inference: Enabling Efficient 
       Inference of Transformer Models at Unprecedented Scale."
    3. Yu et al. (2022), "ORCA: A Distributed Serving System for 
       Transformer-Based Generative Models."
    """
    requires = ("workload", "hardware")
    produces = ServingResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode, seq_len: int, batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5,
              decode_hardware: Optional[HardwareNode] = None, network_bandwidth: Quantity = Q_("100 GB/s"),
              draft_model: Optional[TransformerWorkload] = None, draft_acceptance_rate: float = 0.7,
              cached_prefix_len: int = 0) -> ServingResult:
        """
        Solves for LLM serving performance.

        Parameters
        ----------
        model : TransformerWorkload
            The primary model to be served.
        hardware : HardwareNode
            The hardware node for serving (or pre-fill node in disaggregated serving).
        seq_len : int
            Sequence length (context window).
        batch_size : int
            Batch size.
        precision : str
            Numerical precision.
        efficiency : float
            Compute efficiency.
        decode_hardware : HardwareNode, optional
            If provided, models Disaggregated Serving where 'hardware' does pre-fill
            and 'decode_hardware' does decoding. KV-cache is transferred over the network.
        network_bandwidth : Quantity
            Network bandwidth between pre-fill and decode nodes.
        draft_model : TransformerWorkload, optional
            If provided, models Speculative Decoding using this smaller draft model.
        draft_acceptance_rate : float
            Expected acceptance rate (0.0 to 1.0) of draft tokens per step.
        cached_prefix_len : int
            Number of tokens with pre-computed KV-cache (prompt caching / prefix caching).
            When > 0, the prefill phase only processes (seq_len - cached_prefix_len) new
            tokens, reducing TTFT proportionally. The full KV-cache (including cached prefix)
            still occupies memory. Must be < seq_len.

        Returns
        -------
        ServingResult
            Serving performance metrics.
        """
        prec_map = PRECISION_MAP
        bpp = prec_map.get(precision, BYTES_FP16)

        # 0. Input validation
        if cached_prefix_len >= seq_len:
            raise ValueError(f"cached_prefix_len ({cached_prefix_len}) must be < seq_len ({seq_len})")

        # 1. Pre-fill Phase (with optional prompt caching)
        peak_flops_prefill = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)
        # Prompt caching: only new tokens need prefill computation
        new_tokens = max(1, seq_len - cached_prefix_len)
        # Linear layer FLOPs: 2 * params * tokens * batch (standard 2P approximation)
        linear_flops = 2 * model.parameters.to(ureg.count).magnitude * new_tokens * batch_size
        # Attention FLOPs: 2 * n_layers * n_heads * head_dim * seq_len^2 * batch_size
        # This captures the O(S^2) self-attention cost, which dominates for long contexts.
        n_layers = getattr(model, 'layers', 1) or 1
        n_heads = getattr(model, 'heads', 32) or 32
        head_dim = (getattr(model, 'hidden_dim', 4096) or 4096) // n_heads
        # New tokens attend to ALL tokens (cached + new), not just each other.
        # When cached_prefix_len=0, new_tokens == seq_len so this simplifies to S^2.
        attention_flops = 4 * n_layers * n_heads * head_dim * new_tokens * seq_len * batch_size
        prefill_ops = (linear_flops + attention_flops) * ureg.flop
        t_prefill = (prefill_ops / (peak_flops_prefill * efficiency)).to("ms") + hardware.dispatch_tax

        # KV-cache covers the full sequence (cached prefix + new tokens)
        kv_cache_bytes = model.get_kv_cache_size(seq_len=seq_len, batch_size=batch_size, precision=bpp)
        
        # 2. Disaggregated Serving (KV-Cache Transfer)
        if decode_hardware:
            t_transfer = (kv_cache_bytes / network_bandwidth).to("ms")
            t_prefill += t_transfer
            decode_hw = decode_hardware
        else:
            decode_hw = hardware

        # 3. Decode Phase
        # Per decode step: load weights once (W) + load KV cache for all B requests.
        # kv_cache_bytes already includes batch_size from get_kv_cache_size().
        # Per-token ITL = step_time (each step produces 1 token per request).
        model_weights_bytes = model.size_in_bytes(bpp)
        t_decode_per_token = ((model_weights_bytes + kv_cache_bytes) / decode_hw.memory.bandwidth).to("ms")
        
        # 4. Framework Tax (Per-token decode also incurs launch overhead)
        from .defaults import FRAMEWORK_LAYER_TAX_MS
        layer_tax = Q_(model.layers * FRAMEWORK_LAYER_TAX_MS, "ms")
        t_decode_per_token += layer_tax

        # 5. Speculative Decoding
        if draft_model:
            # Time to generate 1 token with draft model
            draft_weights_bytes = draft_model.size_in_bytes(bpp)
            t_draft_token = ((draft_weights_bytes + kv_cache_bytes) / decode_hw.memory.bandwidth).to("ms")
            
            # Speculative decoding batches K draft tokens (e.g., K=4)
            K = 4
            t_draft_phase = t_draft_token * K
            
            # Verification phase: Target model verifies K tokens in parallel (compute-bound or memory-bound)
            # Simplification: verifying K tokens is roughly 1 forward pass of target model + small compute overhead
            t_verify = t_decode_per_token # essentially memory bound by loading target weights once
            
            # Expected tokens per step via geometric series (Leviathan et al., 2023):
            # E[accepted] = alpha*(1 - alpha^K)/(1 - alpha), plus 1 bonus token
            alpha = draft_acceptance_rate
            if alpha < 1.0:
                expected_tokens = 1 + alpha * (1 - alpha**K) / (1 - alpha)
            else:
                expected_tokens = 1 + K  # perfect acceptance
            
            # Effective ITL
            t_decode_per_token = (t_draft_phase + t_verify) / expected_tokens
        
        total_memory_required = model_weights_bytes + kv_cache_bytes
        if draft_model:
            total_memory_required += draft_model.size_in_bytes(bpp)
        feasible = total_memory_required <= decode_hw.memory.capacity
        
        constraint_trace = []
        if feasible:
            constraint_trace.append(f"Memory Wall: Passed. Required {total_memory_required.to('GB'):~P} (Weights: {model_weights_bytes.to('GB'):~P}, KV Cache: {kv_cache_bytes.to('GB'):~P}) <= Available {decode_hw.memory.capacity.to('GB'):~P} on {decode_hw.name}.")
        else:
            constraint_trace.append(f"Memory Wall: FAILED. Required {total_memory_required.to('GB'):~P} (Weights: {model_weights_bytes.to('GB'):~P}, KV Cache: {kv_cache_bytes.to('GB'):~P}) > Available {decode_hw.memory.capacity.to('GB'):~P} on {decode_hw.name}.")

        cache_hit_ratio = cached_prefix_len / seq_len if seq_len > 0 else 0.0

        return ServingResult(
            feasible=feasible,
            constraint_trace=constraint_trace,
            ttft=t_prefill,
            itl=t_decode_per_token,
            kv_cache_size=kv_cache_bytes.to("GB"),
            model_weights_size=model_weights_bytes.to("GB"),
            total_memory_required=total_memory_required.to("GB"),
            memory_utilization=(total_memory_required / decode_hw.memory.capacity).to_base_units().magnitude,
            prompt_cache_hit_ratio=cache_hit_ratio,
        )


class ContinuousBatchingModel(BaseModel):
    """
    Analyzes production LLM serving with Continuous Batching and PagedAttention.
    
    Traditional static batching suffers from severe memory fragmentation and 
    padding waste. This model simulates the throughput improvements achieved by 
    iteration-level scheduling and non-contiguous KV cache allocation.

    Literature Source:
    1. Kwon et al. (2023), "Efficient Memory Management for Large Language Model
       Serving with PagedAttention."
    2. Yu et al. (2022), "ORCA: A Distributed Serving System for 
       Transformer-Based Generative Models."
    """
    requires = ("workload", "hardware")
    produces = ContinuousBatchingResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode, seq_len: int, max_batch_size: int = 1, page_size: int = 16, precision: str = "fp16", efficiency: float = 0.5) -> ContinuousBatchingResult:
        """Calculates continuous batching throughput and PagedAttention memory."""
        prec_map = PRECISION_MAP
        bpp = prec_map.get(precision, BYTES_FP16)
        peak_flops = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)
        
        # Base latency metrics (including attention O(S²) term)
        n_layers = getattr(model, 'layers', 1) or 1
        n_heads = getattr(model, 'heads', 32) or 32
        head_dim = (getattr(model, 'hidden_dim', 4096) or 4096) // n_heads
        linear_flops = 2 * model.parameters.to(ureg.count).magnitude * seq_len * max_batch_size
        attention_flops = 4 * n_layers * n_heads * head_dim * seq_len**2 * max_batch_size
        prefill_ops = (linear_flops + attention_flops) * ureg.flop
        t_prefill = (prefill_ops / (peak_flops * efficiency)).to("ms") + hardware.dispatch_tax
        
        model_weights_bytes = model.size_in_bytes(bpp)
        max_memory_for_kv = hardware.memory.capacity - model_weights_bytes
        
        if max_memory_for_kv.magnitude <= 0:
            return ContinuousBatchingResult(
                feasible=False, 
                constraint_trace=[f"Memory Wall: FAILED. Weights ({model_weights_bytes.to('GB'):~P}) exceed available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}."],
                throughput_tokens_per_sec=0.0, max_active_requests=0,
                memory_fragmentation_pct=0.0, paged_kv_cache_size=Q_("0 GB"),
                ttft=t_prefill, itl=Q_("1000000 ms"), speedup_vs_static=1.0
            )

        # Calculate memory using PagedAttention formulas
        from .formulas import calc_paged_kv_cache_size
        n_heads = model.kv_heads or model.heads or 32
        h_dim = model.hidden_dim or 4096
        head_dim = h_dim // (model.heads or 32)
        
        bytes_per_seq, frag_pct = calc_paged_kv_cache_size(
            model.layers, n_heads, 
            head_dim,
            seq_len, batch_size=1, page_size_tokens=page_size, bytes_per_elem=bpp
        )
        
        max_possible_requests = int((max_memory_for_kv / bytes_per_seq).to_base_units().magnitude)
        active_requests = min(max_possible_requests, max_batch_size)
        
        constraint_trace = []
        if active_requests > 0:
            constraint_trace.append(f"Memory Wall: Passed. Can fit {active_requests} concurrent requests (Weights: {model_weights_bytes.to('GB'):~P}, Max KV available: {max_memory_for_kv.to('GB'):~P}) on {hardware.name}.")
        else:
            constraint_trace.append(f"Memory Wall: FAILED. Cannot fit even 1 request. Weights ({model_weights_bytes.to('GB'):~P}) exceed or leave no room for KV cache in available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}.")

        total_kv_cache = bytes_per_seq * active_requests
        
        # Throughput
        t_decode_per_token = ((model_weights_bytes + total_kv_cache) / hardware.memory.bandwidth).to("ms")
        
        if active_requests == 0 or t_decode_per_token.magnitude == 0:
            throughput = 0.0
            speedup = 1.0
        else:
            throughput = (active_requests / t_decode_per_token).to("1/s").magnitude
            # Static batching comparison:
            # With contiguous KV allocation, memory fragmentation limits the max batch.
            # Kwon et al. (2023) measured 20-40% external fragmentation in static allocation.
            # We model static batching as achieving ~60% of continuous batching's batch size
            # due to fragmentation waste, with no internal fragmentation savings.
            static_frag_factor = 0.6  # effective batch = 60% of continuous (Kwon et al. 2023)
            static_effective_batch = max(1, int(active_requests * static_frag_factor))
            static_t_decode = ((model_weights_bytes + (bytes_per_seq * static_effective_batch)) / hardware.memory.bandwidth).to("ms")
            static_throughput = (static_effective_batch / static_t_decode).to("1/s").magnitude
            speedup = throughput / static_throughput if static_throughput > 0 else 1.0
            
        return ContinuousBatchingResult(
            feasible=active_requests > 0,
            constraint_trace=constraint_trace,
            throughput_tokens_per_sec=throughput,
            max_active_requests=active_requests,
            memory_fragmentation_pct=frag_pct,
            paged_kv_cache_size=total_kv_cache.to("GB"),
            ttft=t_prefill,
            itl=t_decode_per_token,
            speedup_vs_static=speedup
        )


class WeightStreamingModel(BaseModel):
    """
    Analyzes Wafer-Scale inference (e.g., Cerebras CS-3) using Weight Streaming.
    
    Instead of holding weights in HBM and streaming activations (the GPU Memory Wall),
    this architecture holds massive activation batches on-wafer (SRAM) and streams 
    the model weights from external MemoryX nodes.
    
    The bottleneck shifts from Memory Bandwidth to Injection Interconnect Bandwidth.

    Literature Source:
    1. Lie et al. (2022), "Cerebras Architecture Deep Dive: First Look Inside 
       the Hardware/Software Co-Design for Deep Learning."
    """
    requires = ("workload", "hardware")
    produces = WeightStreamingResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode, seq_len: int,
              batch_size: int = 1, precision: str = "fp16", efficiency: float = 0.5,
              phase: str = "decode") -> WeightStreamingResult:
        """Simulates Weight Streaming throughput and SRAM feasibility.

        Parameters
        ----------
        phase : str
            Inference phase: 'prefill' or 'decode' (default 'decode').
            - prefill: processes all S tokens in parallel (compute-heavy, O(S^2) attention)
            - decode: processes one token at a time per request (memory-bound)
        """
        bpp = PRECISION_MAP.get(precision, BYTES_FP16)
        peak_flops = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)

        # 1. SRAM Capacity Constraint
        # Entire batch's KV Cache must fit in the 44GB on-wafer SRAM
        # Working set (activations for the current layer) must also fit, but KV cache dominates
        n_heads = model.kv_heads or model.heads or 32
        h_dim = model.hidden_dim or 4096
        head_dim = h_dim // (model.heads or 32)

        # KV dimensions per layer per sequence
        bytes_per_seq_per_layer = seq_len * n_heads * head_dim * 2 * bpp.magnitude
        total_kv_bytes = bytes_per_seq_per_layer * model.layers * batch_size * ureg.byte

        # Let's add 10% overhead for working memory
        total_memory_required = (total_kv_bytes * 1.1).to("GB")

        feasible = total_memory_required <= hardware.memory.capacity
        utilization = (total_memory_required / hardware.memory.capacity).magnitude if hardware.memory.capacity.magnitude > 0 else 1.0

        constraint_trace = []
        if feasible:
            constraint_trace.append(f"SRAM Wall: Passed. Required {total_memory_required.to('GB'):~P} (KV + 10% Overhead) <= Available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}.")
        else:
            constraint_trace.append(f"SRAM Wall: FAILED. Required {total_memory_required.to('GB'):~P} (KV + 10% Overhead) > Available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}.")

        # 2. Injection Bottleneck vs Compute Bottleneck per Layer
        layer_params = model.parameters / model.layers
        layer_weight_bytes = layer_params.to(ureg.count).magnitude * bpp.magnitude * ureg.byte

        # Injection time (MemoryX -> WSE)
        inj_bw = hardware.interconnect.bandwidth if hardware.interconnect else Q_("100 GB/s")
        if inj_bw.magnitude == 0:
            inj_bw = Q_("100 GB/s")

        layer_injection_time = (layer_weight_bytes / inj_bw).to("ms")

        # Compute time depends on phase
        if phase == "prefill":
            # Prefill: process all S tokens in parallel.
            # Linear FLOPs: 2 * params * S * batch_size
            # Attention FLOPs: 2 * n_heads * S^2 * head_dim * n_layers (per layer already factored below)
            layer_linear_flops = 2 * layer_params.to(ureg.count).magnitude * seq_len * batch_size
            # Attention for one layer: 2 * n_heads * seq_len^2 * head_dim
            layer_attn_flops = 2 * n_heads * seq_len * seq_len * head_dim
            layer_total_flops = (layer_linear_flops + layer_attn_flops * batch_size) * ureg.flop
            layer_decode_flops = layer_total_flops
        else:
            # Decode: one new token per request across the batch
            # 2 * parameters * batch_size
            layer_decode_flops = 2 * layer_params.to(ureg.count).magnitude * batch_size * ureg.flop
        layer_compute_time = (layer_decode_flops / (peak_flops * efficiency)).to("ms") + hardware.dispatch_tax
        
        # True layer time is bounded by the slowest process
        if layer_compute_time >= layer_injection_time:
            bottleneck = "Compute-Bound"
            layer_time = layer_compute_time
        else:
            bottleneck = "Interconnect-Bandwidth-Bound"
            layer_time = layer_injection_time
            
        total_token_time = layer_time * model.layers
        # In prefill, we process batch_size * seq_len tokens; in decode, batch_size tokens per step
        tokens_produced = batch_size * seq_len if phase == "prefill" else batch_size
        tps = (tokens_produced / total_token_time).to("1/s").magnitude if total_token_time.magnitude > 0 else 0.0
        
        # 3. Optimal Batch Size Analysis
        # What batch size perfectly overlaps injection and compute?
        # Solve: t_inject = (2 * layer_params * B) / (peak_flops * efficiency)
        # => B = t_inject * peak_flops * efficiency / (2 * layer_params)
        optimal_batch = (layer_injection_time * peak_flops * efficiency) / (2 * layer_params.to(ureg.count).magnitude * ureg.flop)
        optimal_batch_int = max(1, int(optimal_batch.to_base_units().magnitude))

        # Short-circuit: if infeasible (SRAM overflow), report zero throughput
        if not feasible:
            tps = 0.0

        return WeightStreamingResult(
            feasible=feasible,
            constraint_trace=constraint_trace,
            throughput_tokens_per_sec=tps,
            bottleneck=bottleneck,
            layer_compute_time=layer_compute_time,
            layer_injection_time=layer_injection_time,
            optimal_batch_size=optimal_batch_int,
            wafer_memory_utilization=min(utilization, 1.0) if feasible else utilization,
        )

class TailLatencyModel(BaseModel):
    """
    Analyzes queueing delays and P99 tail latency for deployed inference models.
    
    Models inference servers as M/M/c queues to determine if the deployment 
    can sustain the target arrival rate while meeting strict SLA latency bounds.

    Literature Source:
    1. Dean & Barroso (2013), "The Tail at Scale."
    """
    requires = ("hardware",)
    produces = TailLatencyResult

    def solve(self, arrival_rate_qps: float, service_latency_ms: float, num_replicas: int = 1, service_time_cv: float = 1.0) -> TailLatencyResult:
        """Solves for P50 and P99 tail latencies under variable load.

        Parameters
        ----------
        arrival_rate_qps : float
            Request arrival rate in queries per second.
        service_latency_ms : float
            Mean service time per request in milliseconds.
        num_replicas : int
            Number of server replicas (c in M/M/c).
        service_time_cv : float
            Coefficient of variation of service time (default 1.0 = exponential).
            When CV != 1, applies Kingman's M/G/1 correction factor
            (cv^2 + 1) / 2 to queue wait times, approximating M/G/c behavior.
        """
        from ._validation import validate_nonnegative
        validate_nonnegative(service_time_cv, "service_time_cv")
        from .formulas import calc_queue_latency_mmc

        service_rate_hz = 1000.0 / service_latency_ms if service_latency_ms > 0 else 0.0

        rho, p50_w, p99_w = calc_queue_latency_mmc(arrival_rate_qps, service_rate_hz, num_replicas)

        # Kingman's formula correction for M/G/c approximation:
        # W_q(M/G/c) ≈ W_q(M/M/c) * (cv² + 1) / 2
        # When cv=1 (exponential), the factor is 1.0 (no correction).
        if service_time_cv != 1.0:
            kingman_factor = (service_time_cv ** 2 + 1) / 2
            p50_w = p50_w * kingman_factor
            p99_w = p99_w * kingman_factor

        is_stable = rho < 1.0

        # P99 wait time exceeding 5x service latency signifies severe SLA risk
        slo_threshold = service_latency_ms * 5

        p99_w_ms = p99_w.m_as(ureg.millisecond)
        p50_w_ms = p50_w.m_as(ureg.millisecond)

        # SLO headroom: ratio of P99 wait to SLO threshold.
        # Values > 1.0 indicate SLO violation. This is a ratio, not a probability.
        slo_headroom_ratio = 1.0 if not is_stable else (p99_w_ms / slo_threshold if slo_threshold > 0 else 0)
        slo_headroom_ratio = max(0.0, slo_headroom_ratio)  # No upper clamp: values > 1.0 indicate SLO violation

        return TailLatencyResult(
            p50_latency=Q_(p50_w_ms + service_latency_ms, "ms"),
            p99_latency=Q_(p99_w_ms + service_latency_ms, "ms"),
            queue_utilization=rho,
            is_stable=is_stable,
            slo_violation_probability=slo_headroom_ratio  # legacy field name; semantically a ratio
        )

class EconomicsModel(BaseModel):
    """
    Calculates Total Cost of Ownership (TCO) including Capex and Opex.
    
    Combines hardware costs, energy consumption, and maintenance 
    into a single financial model for the fleet.

    Literature Source:
    1. Barroso et al. (2018), "The Datacenter as a Computer: An Introduction 
       to the Design of Warehouse-Scale Machines."
    2. Patterson (2004), "Latent Bugs in Common-Case Software." (TCO Foundations)
    3. Meta (2024), "Sustainable AI Infrastructure at Meta Scale."
    """
    requires = ("fleet",)
    produces = EconomicsResult
    _fallacies = {
        "Cheaper hardware is always more cost-effective": "Reality: slower hardware may cost more in electricity and total time than expensive hardware. TCO = CapEx + OpEx; a 2x cheaper GPU that takes 3x longer has higher TCO.",
        "GPU cost is the dominant expense": "Reality: networking, cooling, facility, and staff costs are 50-150% of GPU CapEx. Use infrastructure_multiplier=2.0-2.5 for realistic TCO.",
        "Cloud is always more expensive than on-prem": "Reality: for bursty or short-duration workloads, cloud spot instances can be 3-10x cheaper than amortized on-prem hardware sitting idle.",
    }

    def solve(self, fleet: Fleet, duration_days: float, kwh_price: Optional[float] = None, datacenter: Optional[Any] = None, grid: Optional[Any] = None, mfu: float = 1.0, amortization_years: float = 3.0, infrastructure_multiplier: float = 1.0) -> EconomicsResult:
        """
        Calculates the TCO for a fleet over a specified duration.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        duration_days : float
            Operation duration in days.
        kwh_price : float, optional
            Price of electricity per kWh.
        datacenter : Datacenter, optional
            A specific datacenter profile.
        grid : GridProfile, optional
            A specific grid profile.
        mfu : float, optional
            Model FLOPs Utilization (0.0 to 1.0) impacting energy footprint.

        Returns
        -------
        Dict[str, Any]
            Financial metrics including CapEx, OpEx, and total TCO.
        """
        sust_model = SustainabilityModel()
        energy_result = sust_model.solve(fleet, duration_days, datacenter=datacenter or grid, mfu=mfu)
        
        price = kwh_price
        if price is None:
            # Try to resolve from grid/datacenter or default
            target = grid or datacenter or fleet.datacenter or fleet.region
            price = getattr(target, 'kwh_price', None)
            if price is None:
                price = CLOUD_ELECTRICITY_PER_KWH.magnitude  # 0.12 USD/kWh
            
        opex_energy = energy_result.total_energy_kwh.magnitude * price
        
        unit_cost = fleet.node.accelerator.unit_cost
        if unit_cost is None:
            unit_cost = GPU_UNIT_COST_H100
        total_capex_hardware = unit_cost.magnitude * fleet.total_accelerators
        # Apply infrastructure multiplier for networking, cooling, facility, staff costs
        # Default 1.0 (hardware only). Set 2.0-2.5x for full datacenter TCO.
        total_capex = total_capex_hardware * infrastructure_multiplier
        # Amortize CapEx over deployment period (default 3-year depreciation schedule)
        capex_for_period = (total_capex / amortization_years) * (duration_days / 365.0)

        annual_maintenance_ratio = ANNUAL_MAINTENANCE_RATIO
        opex_maintenance = total_capex * annual_maintenance_ratio * (duration_days / 365.0)

        # Compose economics + sustainability into single result
        return EconomicsResult(
            capex_usd=capex_for_period,
            opex_energy_usd=opex_energy,
            opex_maintenance_usd=opex_maintenance,
            total_opex_usd=opex_energy + opex_maintenance,
            tco_usd=capex_for_period + opex_energy + opex_maintenance,
            it_energy_kwh=energy_result.it_energy_kwh,
            total_energy_kwh=energy_result.total_energy_kwh,
            carbon_footprint_kg=energy_result.carbon_footprint_kg,
            water_usage_liters=energy_result.water_usage_liters,
            pue=energy_result.pue,
            region_name=energy_result.region_name,
        )

class DataModel(BaseModel):
    """
    Analyzes the 'Data Wall' — the throughput bottleneck between storage and compute.
    
    This model simulates the data pipeline constraints, comparing the data demand 
    of a workload (e.g., training tokens or high-resolution video frames) 
    against the physical bandwidth of the storage hierarchy and IO interconnects.

    Literature Source:
    1. Janapa Reddi et al. (2025), "Machine Learning Systems," Chapter 4 (Data Engineering).
    2. Beitzel et al. (2024), "The Data Wall: Scaling Laws for Data Ingestion in AI."
    3. Mohan et al. (2022), "Analyzing and Mitigating Data Bottlenecks in Deep Learning Training."
    """
    requires = ("workload", "hardware")
    produces = DataResult

    def solve(self, workload_data_rate: Quantity, hardware: HardwareNode) -> DataResult:
        """
        Solves for data pipeline feasibility.

        Parameters
        ----------
        workload_data_rate : Quantity
            The required data ingestion rate (e.g., TB/hour or GB/s).
        hardware : HardwareNode
            The hardware node with storage and interconnect specs.

        Returns
        -------
        Dict[str, Any]
            Pipeline metrics including utilization and stall probability.
        """
        # 1. Resolve Hardware Supply
        storage_bw = getattr(hardware.storage, 'bandwidth', Q_("0 GB/s")) if hardware.storage else Q_("0 GB/s")
        io_bw = getattr(hardware.interconnect, 'bandwidth', Q_("0 GB/s")) if hardware.interconnect else Q_("0 GB/s")
        
        # The pipeline is limited by the minimum of storage and interconnect BW
        supply_bw = min(storage_bw.to("GB/s"), io_bw.to("GB/s"))
        demand_bw = workload_data_rate.to("GB/s")
        
        utilization = (demand_bw / supply_bw).magnitude if supply_bw.magnitude > 0 else float('inf')
        is_stalled = utilization > 1.0
        
        return DataResult(
            is_stalled=is_stalled,
            utilization=utilization,
            demand_bw=demand_bw,
            supply_bw=supply_bw,
            bottleneck="Storage" if (storage_bw < io_bw and storage_bw.magnitude > 0) else "Interconnect",
            margin=(supply_bw - demand_bw).to("GB/s"),
        )

class ScalingModel(BaseModel):
    """
    Analyzes the 'Scaling Physics' of model training (Chinchilla Laws).
    
    This model determines the optimal model size (P) and dataset size (D) 
    given a compute budget (C), following the compute-optimal training 
    regime where D ≈ 20P.

    Literature Source:
    1. Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models."
    2. Kaplan et al. (2020), "Scaling Laws for Neural Language Models."
    3. McCandlish et al. (2018), "An Empirical Model of Large-Batch Training."
    """
    requires = ("compute_budget",)
    produces = ScalingResult

    def solve(self, compute_budget: Quantity, target_model_size: Optional[Quantity] = None) -> ScalingResult:
        """
        Solves for compute-optimal model and dataset parameters.

        Parameters
        ----------
        compute_budget : Quantity
            Total training budget (e.g., in TFLOPs or H100-GPU-days).
        target_model_size : Quantity, optional
            If provided, calculates the required tokens for this specific model size.

        Returns
        -------
        Dict[str, Any]
            Optimal parameters, token count, and training duration estimates.
        """
        # C = 6 * P * D
        # Chinchilla: D = 20 * P
        # C = 120 * P^2  => P = sqrt(C / 120)
        
        # Convert H100-days to FLOPs if necessary (simplified approximation)
        c_flops = compute_budget
        if compute_budget.dimensionality == ureg.day.dimensionality:
            # Convert GPU-days to FLOPs using H100 SXM reference
            # Source: NVIDIA H100 datasheet (989 TFLOPs FP16 dense)
            c_flops = (compute_budget * H100_FLOPS_FP16_TENSOR * REFERENCE_MFU_SUSTAINED).to(ureg.flop)

        if target_model_size:
            p_opt = target_model_size.to(ureg.count).magnitude
            d_opt = (c_flops.magnitude / (CHINCHILLA_COMPUTE_CONSTANT * p_opt))
        else:
            p_opt = math.sqrt(c_flops.magnitude / (CHINCHILLA_COMPUTE_CONSTANT * CHINCHILLA_TOKENS_PER_PARAM))
            d_opt = CHINCHILLA_TOKENS_PER_PARAM * p_opt

        return ScalingResult(
            optimal_parameters=Q_(p_opt, ureg.count),
            optimal_tokens=Q_(d_opt, ureg.count),
            compute_budget_flops=c_flops,
            tokens_per_parameter=d_opt / p_opt if p_opt > 0 else 0,
        )

class OrchestrationModel(BaseModel):
    """
    Analyzes Cluster Orchestration and Queueing (Little's Law).

    **Caveat:** This model uses an M/D/1 queue (single server, deterministic
    service) which assumes one job at a time on the entire cluster. For
    multi-tenant clusters with job packing and preemption, an M/G/c model
    is more appropriate — planned for v0.2.0.

    This model simulates the 'Wait Wall' in shared research clusters,
    calculating job completion times and researcher wait times based on
    cluster utilization and arrival rates.

    Literature Source:
    1. Little (1961), "A Proof for the Queuing Formula: L = λW."
    2. Barroso et al. (2018), "The Datacenter as a Computer" (Cluster Mgmt).
    3. Jeon et al. (2019), "Analysis of Large-Scale Multi-Tenant GPU Clusters."
    """
    requires = ("fleet",)
    produces = OrchestrationResult

    def solve(self, fleet: Fleet, arrival_rate_jobs_per_day: float, avg_job_duration_days: float) -> OrchestrationResult:
        """
        Solves for cluster wait times and utilization.

        Parameters
        ----------
        fleet : Fleet
            The hardware cluster configuration.
        arrival_rate_jobs_per_day : float
            λ: Rate at which new training jobs are submitted.
        avg_job_duration_days : float
            The average time a job takes to run if it has the whole cluster.

        Returns
        -------
        Dict[str, Any]
            Wait time, system length, and utilization metrics.
        """
        # ρ = λ / μ  (Utilization)
        # μ = 1 / avg_duration
        
        lambda_rate = arrival_rate_jobs_per_day
        mu_rate = 1.0 / avg_job_duration_days
        
        utilization = lambda_rate / mu_rate
        
        # M/D/1 Queue approximation for wait time (Fixed duration jobs)
        # T_wait = ρ / (2μ(1-ρ))
        if utilization < 1.0:
            wait_time_days = utilization / (2 * mu_rate * (1 - utilization))
        else:
            wait_time_days = float('inf')
            
        return OrchestrationResult(
            cluster_utilization=utilization,
            avg_wait_time_days=Q_(wait_time_days, ureg.day),
            avg_queue_length=utilization**2 / (2 * (1 - utilization)) if utilization < 1.0 else float('inf'),
            is_stable=utilization < 1.0,
        )

class CompressionModel(BaseModel):
    """
    Analyzes model compression trade-offs (Accuracy vs. Efficiency).
    
    This model simulates the 'Compression Tax' — the accuracy degradation 
    that occurs when reducing model size via quantization or pruning, 
    balanced against the gains in memory footprint and inference latency.

    Literature Source:
    1. Han et al. (2015), "Deep Compression: Compressing Deep Neural Networks 
       with Pruning, Trained Quantization and Huffman Coding."
    2. Gholami et al. (2021), "A Survey of Quantization Methods for 
       Efficient Neural Network Inference."
    3. Blalock et al. (2020), "What is the State of Neural Network Pruning?"
    """
    requires = ("workload", "hardware")
    produces = CompressionResult

    def solve(self, model: Workload, hardware: HardwareNode, method: str = "quantization",
              target_bitwidth: int = 8, sparsity: float = 0.0,
              sparsity_type: str = "unstructured") -> CompressionResult:
        """
        Solves for compression gains and estimated accuracy impact.

        Parameters
        ----------
        model : Workload
            The model to be compressed.
        hardware : HardwareNode
            The target execution hardware.
        method : str
            The compression method ('quantization', 'pruning', 'distillation').
        target_bitwidth : int
            Target numerical precision in bits (e.g., 8 for INT8/FP8, 4 for INT4).
            At 8-bit, accuracy delta uses the FP8 estimate (near-lossless) by default.
        sparsity : float
            Target sparsity ratio (0.0 to 1.0) for pruning.
        sparsity_type : str
            Type of sparsity pattern: 'unstructured', 'structured', or 'n_m' (2:4).
            - unstructured: storage savings only, no inference speedup
            - structured: both storage and compute savings
            - n_m: hardware 2:4 sparsity with 2x speedup at 50% sparsity (Ampere+)

        Returns
        -------
        CompressionResult
            Compression metrics including memory savings, inference speedup,
            and estimated accuracy delta.
        """
        from ._validation import validate_at_least, validate_range
        validate_at_least(target_bitwidth, 1, "target_bitwidth")
        validate_range(sparsity, 0.0, 1.0, "sparsity")
        original_size = model.size_in_bytes(Q_("4 byte")) # FP32 baseline
        inference_speedup = 1.0

        if method == "quantization":
            compression_ratio = 32 / target_bitwidth
            # Source: Gholami et al. (2021), "A Survey of Quantization Methods"
            # Conservative estimates: <1% for FP8, <1% for INT8, 2-5% for INT4
            if target_bitwidth >= 16:
                # FP16/BF16/FP32: no meaningful compression from FP32 baseline
                accuracy_delta = 0.0
                compression_ratio = 32 / target_bitwidth  # 2x for FP16, 1x for FP32
            elif target_bitwidth == 8:
                # FP8/INT8: use FP8 accuracy delta (near-lossless, -0.2%)
                accuracy_delta = QUANT_ACCURACY_DELTA_FP8
            elif target_bitwidth >= 4:
                accuracy_delta = QUANT_ACCURACY_DELTA_INT4
            else:
                accuracy_delta = -0.05   # Sub-INT4: significant degradation

            # Inference speedup depends on compute vs memory boundedness
            # Memory-bound workloads: speedup ≈ compression_ratio (less data to move)
            # Compute-bound workloads: speedup depends on hardware low-precision support
            graph = model.lower(Q_("4 byte"))  # FP32 baseline graph
            roofline = calc_bottleneck(
                graph.total_ops, graph.weight_bytes,
                hardware.compute.peak_flops, hardware.memory.bandwidth
            )
            if roofline["bottleneck"] == "Memory":
                inference_speedup = compression_ratio
            else:
                # Compute-bound: check if hardware has accelerated low-precision paths
                prec_key = f"int{target_bitwidth}" if target_bitwidth <= 8 else f"fp{target_bitwidth}"
                if prec_key in hardware.compute.precision_flops:
                    hw_speedup = (hardware.compute.precision_flops[prec_key] / hardware.compute.peak_flops).magnitude
                    inference_speedup = min(hw_speedup, compression_ratio)
                else:
                    inference_speedup = 1.0  # No hardware support → no compute speedup

        elif method == "pruning":
            compression_ratio = 1.0 / (1.0 - sparsity) if sparsity < 1.0 else 100.0
            # Source: Blalock et al. (2020), "What is the State of Neural Network Pruning?"
            # Log-linear degradation accelerates after 50% sparsity
            if sparsity <= PRUNING_ACCURACY_THRESHOLD:
                accuracy_delta = PRUNING_MILD_DELTA
            else:
                accuracy_delta = -PRUNING_STEEP_COEFFICIENT * math.exp(sparsity * PRUNING_STEEP_EXPONENT)

            # Inference speedup depends on sparsity type
            if sparsity_type == "structured":
                # Structured pruning removes entire rows/columns → direct compute savings
                inference_speedup = compression_ratio
            elif sparsity_type == "n_m":
                # N:M sparsity (2:4): hardware-accelerated 2x speedup at exactly 50% sparsity
                # Source: NVIDIA Ampere Architecture Whitepaper (2020)
                if abs(sparsity - 0.5) < 0.05:
                    inference_speedup = 2.0
                else:
                    inference_speedup = 1.0  # N:M only works at 50%
            else:
                # Unstructured: irregular access patterns → storage savings only
                inference_speedup = 1.0
        else:
            compression_ratio = 1.0
            accuracy_delta = 0.0

        compressed_size = original_size / compression_ratio

        return CompressionResult(
            original_size_gb=original_size.to("GB"),
            compressed_size_gb=compressed_size.to("GB"),
            compression_ratio=compression_ratio,
            estimated_accuracy_delta=accuracy_delta,
            memory_savings_pct=(1.0 - 1.0/compression_ratio) * 100,
            inference_speedup=inference_speedup,
        )

class EfficiencyModel(BaseModel):
    """
    Models the gap between peak and achieved FLOPS (Wall 3: Software Efficiency).

    This model quantifies the software efficiency of a workload — the fraction
    of peak hardware FLOPS that the software stack actually converts into useful
    computation. It decomposes Model FLOPs Utilization (MFU) by workload type,
    accounting for kernel fusion efficiency, SM occupancy, and memory access
    patterns.

    Literature Source:
    1. Chowdhery et al. (2022), "PaLM: Scaling Language Modeling with Pathways."
       (First systematic MFU reporting for large Transformers.)
    2. Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact
       Attention with IO-Awareness." (FlashAttention MFU improvement.)
    3. NVIDIA (2023), "Hopper Architecture Tuning Guide." (SM Occupancy model.)
    """
    requires = ("workload", "hardware")
    produces = EfficiencyResult

    def solve(self, model: Workload, hardware: HardwareNode,
              workload_type: str = "ffn", use_flash_attention: bool = False,
              precision: str = "fp16", efficiency: float = 0.5) -> EfficiencyResult:
        """
        Estimates achievable MFU and FLOPS for a given workload type.

        Parameters
        ----------
        model : Workload
            The model architecture to simulate.
        hardware : HardwareNode
            The target hardware node.
        workload_type : str
            The dominant kernel type ('attention', 'ffn', 'conv').
        use_flash_attention : bool
            Whether FlashAttention is enabled (only applies to 'attention').
        precision : str
            Numerical precision ('fp16', 'fp32', 'int8', 'int4').
        efficiency : float
            Base compute efficiency factor (0.0 to 1.0).

        Returns
        -------
        Dict[str, Any]
            MFU estimate, achievable FLOPS, and overhead breakdown.
        """
        peak_flops = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)

        # Base MFU range from defaults (training regime)
        mfu_low = MFU_TRAINING_LOW
        mfu_high = MFU_TRAINING_HIGH

        # Workload-type MFU adjustment (heuristic calibrations — see defaults.py)
        # All values scale linearly with the efficiency parameter relative to 0.5.
        scale = efficiency / 0.5
        if workload_type == "attention":
            if use_flash_attention:
                eta = min(MFU_FLASH_ATTENTION * scale, MFU_FLASH_ATTENTION_CAP)
            else:
                # Standard attention is memory-bound due to O(S²) reads
                eta = mfu_low * scale
        elif workload_type == "ffn":
            # FFN layers are compute-dense GEMM — best MFU
            eta = min(mfu_high * scale, MFU_FFN_CAP)
        elif workload_type == "conv":
            # Convolutions via im2col + GEMM — moderate MFU
            eta = min((mfu_low + mfu_high) / 2.0 * scale, MFU_CONV_CAP)
        else:
            eta = mfu_low * scale

        eta = max(0.0, min(eta, 1.0))  # Clamp to [0, 1]

        achievable_flops = peak_flops * eta

        # Overhead breakdown: decompose (1 - eta) into meaningful components.
        # Total overhead = 1 - eta, split into occupancy loss and memory stall.
        total_overhead = 1.0 - eta
        occupancy_loss = 1.0 - min(efficiency * HFU_MFU_RATIO, 1.0)  # SM occupancy overhead
        # Memory stall absorbs the remainder: time spent waiting on data movement.
        memory_stall = max(0.0, total_overhead - occupancy_loss)

        return EfficiencyResult(
            mfu=eta,
            achievable_flops=achievable_flops,
            peak_flops=peak_flops,
            workload_type=workload_type,
            use_flash_attention=use_flash_attention,
            overhead_breakdown={
                "occupancy_loss": occupancy_loss,
                "memory_stall": memory_stall,
            },
        )

class TransformationModel(BaseModel):
    """
    Quantifies the CPU preprocessing bottleneck (Wall 9: Transformation).

    This model simulates the 'Transformation Wall' — the gap between CPU-bound
    data preprocessing (JPEG decode, tokenization, augmentation) and
    accelerator step time. When preprocessing cannot keep up, the accelerator
    starves and utilization drops.

    Literature Source:
    1. Mohan et al. (2022), "Analyzing and Mitigating Data Bottlenecks in
       Deep Learning Training."
    2. Murray et al. (2021), "tf.data: A Machine Learning Data Processing
       Framework." (Pipeline stall analysis.)
    3. NVIDIA DALI Documentation (2024). (GPU-accelerated preprocessing.)
    """
    requires = ("hardware",)
    produces = TransformationResult

    def solve(self, batch_size: int, sample_size_bytes: Quantity,
              cpu_throughput: Quantity, accelerator_step_time: Quantity) -> TransformationResult:
        """
        Solves for CPU preprocessing bottleneck.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.
        sample_size_bytes : Quantity
            Size of one sample in bytes (e.g., Q_("500 KB")).
        cpu_throughput : Quantity
            CPU preprocessing throughput (e.g., Q_("2 GB/s")).
        accelerator_step_time : Quantity
            Time for one accelerator training step (e.g., Q_("50 ms")).

        Returns
        -------
        Dict[str, Any]
            Transform time, bottleneck status, and accelerator utilization.
        """
        # T_transform = (B × S_sample) / C_throughput
        batch_data = batch_size * sample_size_bytes.to("byte")
        transform_time = (batch_data / cpu_throughput.to("byte/s")).to("ms")

        accel_time = accelerator_step_time.to("ms")
        is_bottleneck = transform_time.magnitude > accel_time.magnitude

        # Accelerator utilization: fraction of time the accelerator is active
        total_step_time = max(transform_time.magnitude, accel_time.magnitude)
        accelerator_utilization = accel_time.magnitude / total_step_time if total_step_time > 0 else 0.0

        return TransformationResult(
            transform_time=transform_time,
            accelerator_step_time=accel_time,
            is_bottleneck=is_bottleneck,
            accelerator_utilization=accelerator_utilization,
            slowdown_factor=total_step_time / accel_time.magnitude if accel_time.magnitude > 0 else float('inf'),
        )

class TopologyModel(BaseModel):
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
        Dict[str, Any]
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

class InferenceScalingModel(BaseModel):
    """
    Models inference-time compute scaling (Wall 12: Reasoning/CoT Cost).

    This model quantifies the cost of 'System-2 thinking' — inference-time
    compute scaling via chain-of-thought (CoT) reasoning, where the model
    generates K intermediate reasoning steps before producing the final answer.
    Each step incurs the full cost of autoregressive decoding.

    Literature Source:
    1. Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning in
       Large Language Models."
    2. Snell et al. (2024), "Scaling LLM Test-Time Compute Optimally Can Be
       More Effective Than Scaling Model Parameters."
    3. OpenAI (2024), "Learning to Reason with LLMs." (o1 reasoning model.)
    """
    requires = ("workload", "hardware")
    produces = InferenceScalingResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode,
              reasoning_steps: int = 8, context_length: int = 2048,
              precision: str = "fp16", efficiency: float = 0.5) -> InferenceScalingResult:
        """
        Solves for inference-time reasoning cost.

        Parameters
        ----------
        model : TransformerWorkload
            The language model used for reasoning.
        hardware : HardwareNode
            The target hardware node.
        reasoning_steps : int
            Number of reasoning steps K (each generates tokens).
        context_length : int
            Input context length in tokens.
        precision : str
            Numerical precision.
        efficiency : float
            Compute efficiency factor (0.0 to 1.0).

        Returns
        -------
        Dict[str, Any]
            Total reasoning time, cost per query, and token counts.
        """
        # Use ServingModel internally to get per-step latency
        serving = ServingModel()
        serving_result = serving.solve(
            model=model, hardware=hardware, seq_len=context_length,
            batch_size=1, precision=precision, efficiency=efficiency
        )

        # Inter-token latency (ITL) is the cost per generated token
        itl = serving_result.itl

        # Average tokens per reasoning step (heuristic — see defaults.py)
        tokens_per_step = TOKENS_PER_REASONING_STEP
        total_tokens = reasoning_steps * tokens_per_step

        # T_reason = K × tokens_per_step × ITL + TTFT (initial prefill)
        ttft = serving_result.ttft
        t_step = tokens_per_step * itl
        total_reasoning_time = ttft + reasoning_steps * t_step

        # Cost estimate (based on hardware TDP and time)
        if hardware.tdp is not None:
            energy_j = (hardware.tdp * total_reasoning_time.to("s")).to("J")
        else:
            energy_j = Q_("0 J")

        return InferenceScalingResult(
            total_reasoning_time=total_reasoning_time.to("ms"),
            ttft=ttft,
            itl=itl,
            tokens_generated=total_tokens,
            reasoning_steps=reasoning_steps,
            time_per_step=t_step.to("ms"),
            energy_per_query=energy_j,
            feasible=serving_result.feasible,
            serving_detail=serving_result,
        )

class SensitivitySolver(BaseSolver):
    """
    Identifies the binding constraint via numerical sensitivity analysis (Wall 21).

    This solver computes numerical partial derivatives of performance
    with respect to hardware parameters to identify the 'binding constraint.'

    Literature Source:
    1. Williams et al. (2009), "Roofline Model."
    2. Ofenbeck et al. (2014), "Applying the Roofline Model."
    """
    requires = ("workload", "hardware")
    produces = SensitivityResult

    def solve(self, model: Workload, hardware: HardwareNode,
              precision: str = "fp16", perturbation_pct: float = 10.0,
              efficiency: float = 0.5) -> SensitivityResult:
        """
        Solves for sensitivities and identifies the binding constraint.
        """
        from copy import deepcopy
        from ..hardware.types import HardwareNode, ComputeCore, MemoryHierarchy

        baseline = Engine.solve(model, hardware, precision=precision, efficiency=efficiency)
        t_base = baseline.latency.to("ms").magnitude
        factor = 1.0 + perturbation_pct / 100.0
        sensitivities = {}

        hw_flops = deepcopy(hardware)
        hw_flops.compute = ComputeCore(
            peak_flops=hardware.compute.peak_flops * factor,
            precision_flops={k: v * factor for k, v in hardware.compute.precision_flops.items()}
        )
        t_flops = Engine.solve(model, hw_flops, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["peak_flops"] = (t_flops - t_base) / t_base if t_base > 0 else 0.0

        hw_bw = deepcopy(hardware)
        hw_bw.memory = MemoryHierarchy(
            capacity=hardware.memory.capacity,
            bandwidth=hardware.memory.bandwidth * factor
        )
        t_bw = Engine.solve(model, hw_bw, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["memory_bandwidth"] = (t_bw - t_base) / t_base if t_base > 0 else 0.0

        hw_mem = deepcopy(hardware)
        hw_mem.memory = MemoryHierarchy(
            capacity=hardware.memory.capacity * factor,
            bandwidth=hardware.memory.bandwidth
        )
        t_mem = Engine.solve(model, hw_mem, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["memory_capacity"] = (t_mem - t_base) / t_base if t_base > 0 else 0.0

        binding = max(sensitivities, key=lambda k: abs(sensitivities[k]))

        return SensitivityResult(
            sensitivities=sensitivities,
            binding_constraint=binding,
            baseline_latency=baseline.latency,
            perturbation_pct=perturbation_pct,
        )

class SynthesisSolver(BaseSolver):
    """
    Given an SLA, synthesizes the required hardware specs (Wall 22: Inverse Solve).

    This solver inverts the Roofline model to derive minimum hardware 
    specifications required to meet a target latency SLA.

    Literature Source:
    1. Williams et al. (2009), "Roofline Model."
    2. Jouppi et al. (2017), "In-Datacenter Performance Analysis of a TPU."
    """
    requires = ("workload", "target_latency")
    produces = SynthesisResult

    def solve(self, model: Workload, target_latency: Quantity,
              precision: str = "fp16", efficiency: float = 0.5) -> SynthesisResult:
        """
        Synthesizes hardware requirements from an SLA target.
        """
        prec_map = PRECISION_MAP
        bpp = prec_map.get(precision, BYTES_FP16)
        weight_bytes = model.size_in_bytes(bpp)
        graph = model.lower(bpp)
        total_ops = graph.total_ops
        t_target = target_latency.to("s")

        required_bw = (weight_bytes / t_target).to("GB/s")
        required_flops = (total_ops / (t_target * efficiency)).to("TFLOP/s")
        required_memory = weight_bytes.to("GB")
        compute_memory_ratio = (required_flops / required_bw).to("flop/byte")

        return SynthesisResult(
            required_bw=required_bw,
            required_flops=required_flops,
            required_memory=required_memory,
            compute_memory_ratio=compute_memory_ratio,
            target_latency=target_latency,
            model_size=weight_bytes.to("GB"),
            total_ops=total_ops,
        )

class ResponsibleEngineeringModel(BaseModel):
    """
    Models the computational cost of responsible AI practices (Wall 20: Safety).

    This model quantifies the 'Safety Tax' — the additional compute and data
    required for differential privacy or fairness guarantees.

    Literature Source:
    1. Abadi et al. (2016), "Deep Learning with Differential Privacy."
    2. Anil et al. (2022), "Large-Scale Differentially Private BERT."
    """
    requires = ("training_time",)
    produces = ResponsibleEngineeringResult

    def solve(self, base_training_time: Quantity,
              epsilon: float = 1.0, delta: float = 1e-5,
              min_subgroup_prevalence: float = 0.01) -> ResponsibleEngineeringResult:
        """
        Calculates the overhead of responsible engineering practices.
        """
        dp_slowdown = 1.0 + (DP_SGD_SLOWDOWN_COEFFICIENT / max(epsilon, 0.01))
        additional_data_factor = 1.0 / max(min_subgroup_prevalence, 1e-6)
        effective_time = base_training_time * dp_slowdown

        return ResponsibleEngineeringResult(
            dp_slowdown_factor=dp_slowdown,
            effective_training_time=effective_time.to(base_training_time.units),
            additional_data_requirement=additional_data_factor,
            epsilon=epsilon,
            delta=delta,
            min_subgroup_prevalence=min_subgroup_prevalence,
            privacy_cost_ratio=dp_slowdown,
            fairness_data_ratio=additional_data_factor,
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
        Searches for the optimal parallelism split.
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
                    # Memory feasibility: per-GPU model shard must fit in HBM.
                    # TP shards weights across tp GPUs; PP shards layers across pp stages.
                    # Gradients and optimizer states are sharded across DP ranks (ZeRO-1+).
                    # Conservative check: weights + gradients per GPU < 90% HBM capacity.
                    per_gpu_weights = model.size_in_bytes() / tp / pp
                    per_gpu_grads = per_gpu_weights  # gradients same size as weights
                    per_gpu_mem = per_gpu_weights + per_gpu_grads
                    gpu_capacity = fleet.node.accelerator.memory.capacity
                    if per_gpu_mem > gpu_capacity * 0.9:
                        continue  # Infeasible: model shard doesn't fit in GPU memory

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


class BatchingOptimizer(BaseOptimizer):
    """
    Finds the maximum batch size that satisfies a P99 latency SLA.

    Searches the continuous batching design space using an M/M/c queueing
    model to find the optimal balance between throughput and tail latency.
    """
    requires = ("workload", "hardware", "sla_latency_ms")
    produces = BatchingOptimizerResult

    def solve(self, model: TransformerWorkload, hardware: HardwareNode, 
              seq_len: int, sla_latency_ms: float, arrival_rate_qps: float,
              num_replicas: int = 1, precision: str = "fp16", 
              efficiency: float = 0.5, max_search_batch: int = 256) -> BatchingOptimizerResult:
        
        from .optimization.registry import OptimizationRegistry
        serving_model = ServingModel()
        tail_model = TailLatencyModel()
        
        def objective(b_array):
            b = int(b_array[0])
            res = serving_model.solve(model, hardware, seq_len=seq_len, batch_size=b, precision=precision, efficiency=efficiency)
            if not res.feasible:
                return 1e12 # Infeasible due to memory
                
            service_latency = res.ttft + (res.itl * seq_len)
            tail_res = tail_model.solve(arrival_rate_qps / b, service_latency.m_as("ms"), num_replicas)
            
            if not tail_res.is_stable or tail_res.p99_latency.m_as("ms") > sla_latency_ms:
                return 1e12 # Infeasible due to queueing instability or SLA violation
                
            # We want to maximize batch size (which maximizes valid throughput), so minimize -b
            return -b
            
        backend = OptimizationRegistry.get_backend("exhaustive")
        backend.compile(objective_fn=objective, ranges=[(1, max_search_batch)], grid_size=max_search_batch)
        opt_res = backend.solve()

        if not opt_res.feasible:
            return BatchingOptimizerResult(
                objective_value=0.0,
                best_config={"max_batch_size": 0},
                best_batch_size=0,
                max_throughput=0.0,
                p99_latency=Q_("0 ms"),
                slo_violation_probability=0.0,
                is_feasible=False,
                total_searched=max_search_batch
            )
            
        best_b = int(opt_res.best_configuration["optimal_variable"])
        max_tps = arrival_rate_qps * seq_len
        
        # Resolve final exact metrics for the optimal batch size
        res = serving_model.solve(model, hardware, seq_len=seq_len, batch_size=best_b, precision=precision, efficiency=efficiency)
        service_latency = res.ttft + (res.itl * seq_len)
        tail_res = tail_model.solve(arrival_rate_qps / best_b, service_latency.m_as("ms"), num_replicas)

        return BatchingOptimizerResult(
            objective_value=max_tps,
            best_config={"max_batch_size": best_b},
            best_batch_size=best_b,
            max_throughput=max_tps,
            p99_latency=tail_res.p99_latency,
            slo_violation_probability=tail_res.slo_violation_probability,
            is_feasible=True,
            total_searched=max_search_batch
        )


class PlacementOptimizer(BaseOptimizer):
    """
    Finds the optimal datacenter location to minimize TCO and Carbon.
    """
    requires = ("fleet", "duration_days")
    produces = PlacementOptimizerResult

    def solve(self, fleet: Fleet, duration_days: float, 
              regions: List[str] = ["US_Avg", "Quebec", "Iowa"], 
              carbon_tax_per_ton: float = 100.0, mfu: float = 1.0) -> PlacementOptimizerResult:
        
        from ..infra.registry import Infra
        econ_model = EconomicsModel()
        
        candidates = []
        
        for region_name in regions:
            grid = getattr(Infra.Grids, region_name, None)
            if not grid: continue
                
            res = econ_model.solve(fleet, duration_days=duration_days, grid=grid, mfu=mfu)
            
            # Objective: TCO + Carbon Tax
            carbon_tons = res.carbon_footprint_kg / 1000.0
            total_cost = res.tco_usd + (carbon_tons * carbon_tax_per_ton)
            
            candidates.append({
                "region": region_name,
                "tco": res.tco_usd,
                "carbon": carbon_tons,
                "pue": res.pue,
                "objective": total_cost
            })
            
        if not candidates:
            raise ValueError("No valid regions found for optimization.")
            
        best = min(candidates, key=lambda x: x["objective"])
        top_n = sorted(candidates, key=lambda x: x["objective"])
        
        return PlacementOptimizerResult(
            objective_value=best["objective"],
            best_config={"region": best["region"]},
            best_region=best["region"],
            lowest_tco=best["tco"],
            carbon_footprint=best["carbon"],
            pue=best["pue"],
            total_searched=len(candidates),
            top_candidates=top_n
        )

