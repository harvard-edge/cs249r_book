"""Training memory, checkpointing, and scaling-law solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations

import math
from typing import Optional

from ..results import (
    CheckpointResult,
    TrainingMemoryResult,
    ScalingResult,
)
from ...core.units import ureg, Q_, resolve_precision
from ...literature.registry import Literature
from .. import calibration as cal
from ...core.types import Quantity
from ...models.types import Workload, TransformerWorkload, SparseTransformerWorkload
from ...hardware.types import HardwareNode
from .base import ForwardModel

class CheckpointModel(ForwardModel):
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
        from ...physics import calc_checkpoint_size

        # Calculate size based on optimizer states
        # Mixed-precision Adam: 14 bytes/param (FP32 master + FP32 momentum + FP32 variance + FP16 weights)
        # Gradients are ephemeral and not checkpointed.
        if optimizer.lower() == "adam":
            bytes_per_param = cal.CHECKPOINT_BYTES_PER_PARAM_ADAM
        else:
            bytes_per_param = cal.CHECKPOINT_BYTES_PER_PARAM_SGD  # e.g., SGD

        ckpt_size = calc_checkpoint_size(model.parameters, bytes_per_param=bytes_per_param)

        storage_bw = getattr(hardware.storage, 'bandwidth', Q_(cal.FALLBACK_STORAGE_BANDWIDTH_GB_S, "GB/s")) if hardware.storage else Q_(cal.FALLBACK_STORAGE_BANDWIDTH_GB_S, "GB/s")
        # Fallback to network or standard disk speed if undefined — a zero
        # bandwidth would make the write time below divide by zero.
        if storage_bw.magnitude == 0:
            storage_bw = Q_(cal.FALLBACK_STORAGE_BANDWIDTH_GB_S, "GB/s")

        # Distributed writing: per-writer bandwidth scales linearly until the
        # shared filesystem's aggregate ceiling binds (FSDP-style sharded writes
        # do not get to ignore the backend's total ingest limit).
        fs_limit = Q_(filesystem_limit_gbs, "GB/s")
        effective_write_bw = min(storage_bw * n_writers, fs_limit)

        t_write = (ckpt_size / effective_write_bw).to("second")

        # MFU penalty: the fraction of each checkpoint interval the cluster spends
        # blocked on the synchronous write (training stalls while state drains).
        interval_s = Q_(checkpoint_interval_hours, "hour").to("second")
        if interval_s.magnitude > 0:
            penalty_pct = (t_write / interval_s).magnitude
        else:
            # Degenerate zero interval: checkpointing continuously, no useful work.
            penalty_pct = 1.0

        return CheckpointResult(
            checkpoint_size=ckpt_size.to("GB"),
            write_time_seconds=t_write,
            max_bandwidth_required=storage_bw,
            storage_bottleneck=t_write.m_as("second") > 60.0, # Flag if checkpoint takes > 1 min
            mfu_penalty_pct=penalty_pct
        )

class TrainingMemoryModel(ForwardModel):
    """
    Decomposes per-accelerator training memory into teachable components.

    This model answers a different question than ``SingleNodeModel``. Roofline
    feasibility asks whether a workload's inference weights fit; training
    feasibility must also account for gradients, optimizer state, activations,
    and communication buffers. The accounting follows the common mixed-precision
    state breakdown used by Megatron-LM and ZeRO.

    Literature Source:
    1. Shoeybi et al. (2019), "Megatron-LM" (tensor/pipeline parallel state).
    2. Rajbhandari et al. (2020), "ZeRO" (data-parallel state sharding).
    3. Korthikanti et al. (2023), activation recomputation accounting.

    Formula contract:
    - parameter states are sharded by TP * PP * EP before ZeRO sharding.
    - ZeRO-1 shards optimizer state, ZeRO-2 also shards gradients, and ZeRO-3
      also shards weights across DP.
    - activation memory is based on local microbatch, layers per pipeline stage,
      hidden dimension, and the actual bytes per element for the precision.
    """
    requires = ("workload", "hardware")
    produces = TrainingMemoryResult

    def _get_optimizer_bytes(self, optimizer_key: str) -> float:
        """
        Retrieves the byte multiplier for the requested optimizer state.

        Parameters
        ----------
        optimizer_key : str
            The name of the optimizer (e.g., 'adam', 'sgd', 'none').

        Returns
        -------
        float
            The number of bytes required per parameter for the optimizer's state.

        Raises
        ------
        ValueError
            If the requested optimizer is not supported.
        """
        mapping = {
            "adam": cal.TRAINING_OPTIMIZER_BYTES_ADAM,   # FP32 master + first and second moments
            "adamw": cal.TRAINING_OPTIMIZER_BYTES_ADAM,
            "sgd": cal.TRAINING_OPTIMIZER_BYTES_SGD,     # FP32 master weights
            "none": 0.0,
        }
        if optimizer_key not in mapping:
            supported = ", ".join(sorted(mapping))
            raise ValueError(f"Unknown optimizer '{optimizer_key}'. Supported: {supported}")
        return mapping[optimizer_key]

    def solve(
        self,
        model: TransformerWorkload,
        hardware: HardwareNode,
        batch_size: int,
        seq_len: int = 2048,
        precision: str = "fp16",
        optimizer: str = "adam",
        activation_checkpointing: str = "selective",
        tp_size: int = 1,
        pp_size: int = 1,
        dp_size: int = 1,
        ep_size: int = 1,
        zero_stage: int = 0,
        gradient_accumulation_steps: int = 1,
        trainable_fraction: float = 1.0,
        communication_buffer_fraction: float = 0.05,
    ) -> TrainingMemoryResult:
        """Estimate per-accelerator training memory.

        ``batch_size`` is the global batch. The activation term uses the local
        microbatch implied by data parallelism and gradient accumulation. Model
        states are sharded by tensor, pipeline, and expert parallelism first;
        ZeRO then shards optimizer, gradient, and parameter states across the
        data-parallel group according to its stage.
        """
        from ...core._validation import validate_at_least, validate_range
        from ...physics import calc_activation_memory

        precision, precision_bytes = resolve_precision(precision)
        optimizer_key = optimizer.lower()
        opt_bytes = self._get_optimizer_bytes(optimizer_key)

        if activation_checkpointing not in {"none", "selective", "full"}:
            raise ValueError("activation_checkpointing must be 'none', 'selective', or 'full'")
        validate_at_least(batch_size, 1, "batch_size")
        validate_at_least(seq_len, 1, "seq_len")
        validate_at_least(tp_size, 1, "tp_size")
        validate_at_least(pp_size, 1, "pp_size")
        validate_at_least(dp_size, 1, "dp_size")
        validate_at_least(ep_size, 1, "ep_size")
        validate_at_least(gradient_accumulation_steps, 1, "gradient_accumulation_steps")
        validate_range(trainable_fraction, 0.0, 1.0, "trainable_fraction")
        validate_range(communication_buffer_fraction, 0.0, 1.0, "communication_buffer_fraction")
        if zero_stage not in {0, 1, 2, 3}:
            raise ValueError("zero_stage must be 0, 1, 2, or 3")

        # Step 1: shard model state across model-parallel dimensions before
        # applying ZeRO's data-parallel sharding rules.
        bpp = precision_bytes.to(ureg.byte).magnitude
        total_params = model.parameters.to(ureg.count).magnitude
        model_parallel_shards = tp_size * pp_size * ep_size
        params_per_rank = total_params / model_parallel_shards
        trainable_params_per_rank = params_per_rank * trainable_fraction

        weights = params_per_rank * bpp * ureg.byte
        gradients = trainable_params_per_rank * bpp * ureg.byte
        optimizer_state = trainable_params_per_rank * opt_bytes * ureg.byte

        # ZeRO stages are cumulative: each stage shards one more state class
        # across the DP group. Stage 1 = optimizer state only (the biggest win at
        # 8-12 bytes/param for Adam), stage 2 adds gradients, stage 3 adds the
        # weights themselves (at the cost of gather traffic every layer).
        if zero_stage >= 1:
            optimizer_state = optimizer_state / dp_size
        if zero_stage >= 2:
            gradients = gradients / dp_size
        if zero_stage >= 3:
            weights = weights / dp_size

        # Step 2: activations scale with the local microbatch and layers owned by
        # this pipeline stage. bpp is bytes/element; the Korthikanti constants
        # inside calc_activation_memory are FP16-based and scale by bpp/2.
        # Global batch -> what one rank holds at once: divide by DP replicas, then
        # by accumulation steps (each micro-step's activations are freed before
        # the next), and never below one sample.
        local_microbatch = max(1, math.ceil(batch_size / (dp_size * gradient_accumulation_steps)))
        layers_per_stage = max(1, math.ceil(model.layers / pp_size))
        hidden_dim = model.hidden_dim or 4096
        activations = calc_activation_memory(
            n_layers=layers_per_stage,
            seq_len=seq_len,
            batch_size=local_microbatch,
            hidden_dim=hidden_dim,
            n_heads=getattr(model, "heads", None),
            precision_bytes=bpp,
            strategy=activation_checkpointing,
        )

        # Gradient bucket: the staging buffer the DP allreduce drains from,
        # modeled as a small fraction of the gradient footprint.
        grad_bucket = gradients * communication_buffer_fraction
        pipeline_buffer = Q_("0 byte")
        if pp_size > 1:
            # Pipeline boundary activations: one send + one receive buffer, each
            # holding a microbatch's worth of hidden states at the stage boundary.
            pipeline_buffer = 2 * local_microbatch * seq_len * hidden_dim * bpp * ureg.byte
        communication_buffers = grad_bucket + pipeline_buffer

        total_memory = (weights + gradients + optimizer_state + activations + communication_buffers).to("GB")
        available_memory = hardware.memory.capacity.to("GB")
        memory_utilization = (total_memory / available_memory).to_base_units().magnitude if available_memory.magnitude > 0 else float("inf")
        feasible = total_memory <= available_memory

        trace = [
            (
                "Training Memory: "
                f"weights={weights.to('GB'):~P}, gradients={gradients.to('GB'):~P}, "
                f"optimizer={optimizer_state.to('GB'):~P}, activations={activations.to('GB'):~P}, "
                f"buffers={communication_buffers.to('GB'):~P}."
            )
        ]
        if zero_stage > 0:
            trace.append(f"ZeRO-{zero_stage}: data-parallel state sharding applied across {dp_size} ranks.")
        if isinstance(model, SparseTransformerWorkload) and ep_size > 1:
            trace.append(
                "MoE memory approximation: total expert parameters are evenly sharded by expert parallelism; "
                "routing imbalance is modeled separately by MoERoutingModel."
            )

        return TrainingMemoryResult(
            feasible=feasible,
            constraint_trace=trace,
            total_memory=total_memory,
            available_memory=available_memory,
            memory_utilization=memory_utilization,
            weights=weights.to("GB"),
            gradients=gradients.to("GB"),
            optimizer_state=optimizer_state.to("GB"),
            activations=activations.to("GB"),
            communication_buffers=communication_buffers.to("GB"),
            precision=precision,
            optimizer=optimizer_key,
            activation_checkpointing=activation_checkpointing,
            parallelism={"dp": dp_size, "tp": tp_size, "pp": pp_size, "ep": ep_size},
        )

class ScalingModel(ForwardModel):
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
        ScalingResult
            Optimal parameters, token count, and training duration estimates.
        """
        # C = 6 * P * D
        # Chinchilla: D = 20 * P
        # C = 120 * P^2  => P = sqrt(C / 120)

        # Convert H100-days to FLOPs if necessary (simplified approximation)
        c_flops = compute_budget
        if compute_budget.dimensionality == ureg.day.dimensionality:
            # Convert GPU-days to FLOPs using H100 SXM reference, derated by a
            # sustained-MFU factor — clusters never run at datasheet peak.
            # Source: NVIDIA H100 datasheet (989 TFLOPs FP16 dense)
            c_flops = (compute_budget * (cal.REFERENCE_HARDWARE_TFLOPS * ureg.TFLOPs / ureg.second) * cal.REFERENCE_MFU_SUSTAINED).to(ureg.flop)

        if target_model_size:
            # Pinned model size: spend the whole budget on tokens via C = 6PD,
            # i.e. D = C / (6P). The result may be over- or under-trained
            # relative to the Chinchilla-optimal D = 20P.
            p_opt = target_model_size.to(ureg.count).magnitude
            d_opt = (c_flops.magnitude / (Literature.Chinchilla.ComputeConstant * p_opt))
        else:
            # Compute-optimal point: substituting D = 20P into C = 6PD gives
            # C = 120 P^2, so P = sqrt(C / 120).
            p_opt = math.sqrt(c_flops.magnitude / (Literature.Chinchilla.ComputeConstant * Literature.Chinchilla.TokensPerParam))
            d_opt = Literature.Chinchilla.TokensPerParam * p_opt

        return ScalingResult(
            optimal_parameters=Q_(p_opt, ureg.count),
            optimal_tokens=Q_(d_opt, ureg.count),
            compute_budget_flops=c_flops,
            tokens_per_parameter=d_opt / p_opt if p_opt > 0 else 0,
        )

__all__ = [
    "CheckpointModel",
    "TrainingMemoryModel",
    "ScalingModel",
]
