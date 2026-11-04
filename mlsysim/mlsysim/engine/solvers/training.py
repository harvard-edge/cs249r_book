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
from ...core._validation import validate_positive
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
        """Estimate synchronous checkpoint size, write time, and stall fraction.

        Parameters
        ----------
        model : Workload
            Workload whose parameter count sets the checkpoint payload.
        hardware : HardwareNode
            Device with storage bandwidth in bytes/second. Missing or zero storage
            bandwidth uses the declared fallback in ``engine.calibration``.
        optimizer : str, optional
            ``"adam"`` selects mixed-precision Adam checkpoint bytes; other names
            use the SGD convention. Gradients are not persisted.
        checkpoint_interval_hours : float, optional
            Hours between checkpoints. A nonpositive interval returns a stall
            fraction of one; positive intervals use write time divided by interval.
        n_writers : int, optional
            Positive number of writers sharing the checkpoint payload.
        filesystem_limit_gbs : float, optional
            Positive aggregate filesystem ceiling in decimal GB/second.

        Returns
        -------
        CheckpointResult
            Checkpoint size in GB, write duration in seconds, and a dimensionless
            stall fraction in ``mfu_penalty_pct`` (not multiplied by 100).
            ``storage_bottleneck`` flags writes longer than one minute; it does
            not certify that a checkpoint schedule fits the interval.

        See Also
        --------
        mlsysim.physics.memory.calc_checkpoint_size : Persisted state accounting.

        Notes
        -----
        Equation: M = P b; bandwidth = min(n_writers * device_bandwidth,
        filesystem_limit); write_time = M / bandwidth. The penalty is
        write_time / checkpoint_interval for a positive interval.

        Assumptions: writers partition one payload without duplicated writes;
        checkpointing synchronously blocks useful training. Byte conventions
        come from ``engine.calibration`` and are separate from storage rates.

        Excludes: asynchronous overlap, filesystem metadata, serialization CPU
        work, network contention, and checkpoint recovery. The penalty can
        exceed one when a write takes longer than its configured interval.

        References
        ----------
        Derivation: payload size is parameter count times persisted bytes per
        parameter; elapsed transfer time is payload divided by available rate.
        The writer minimum is a shared-bandwidth capacity constraint.

        Examples
        --------
        >>> from mlsysim import Models, Hardware
        >>> result = CheckpointModel().solve(Models.Language.Llama3_8B, Hardware.Cloud.H100)
        >>> result.checkpoint_size.to("GB").magnitude > 0
        True
        >>> result.write_time_seconds.to("second").magnitude > 0
        True
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
        from ...physics.memory import optimizer_state_bytes
        return optimizer_state_bytes(optimizer_key).m_as("byte")

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
        sequence_parallel: bool = True,
    ) -> TrainingMemoryResult:
        """Estimate per-accelerator training memory from local state and activations.

        Parameters
        ----------
        model : TransformerWorkload
            Parameter count, layer count, hidden width, and attention-head geometry.
        hardware : HardwareNode
            Device memory capacity used for the residency comparison.
        batch_size : int
            Global sequence count per update; must be at least one.
        seq_len : int, optional
            Tokens per sequence; must be at least one.
        precision : str, optional
            Parameter and gradient storage precision, resolved to bytes/element.
        optimizer : str, optional
            Optimizer storage convention supported by ``optimizer_state_bytes``.
        activation_checkpointing : {'none', 'selective', 'full'}, optional
            Activation retention strategy passed to the activation-memory model.
        tp_size : int, optional
            Tensor-parallel degree, at least one.
        pp_size : int, optional
            Pipeline-parallel degree, at least one.
        dp_size : int, optional
            Data-parallel degree, at least one.
        ep_size : int, optional
            Expert-parallel degree, at least one.
        zero_stage : {0, 1, 2, 3}, optional
            Cumulative DP sharding: optimizer state, then gradients, then weights.
        gradient_accumulation_steps : int, optional
            Positive number of sequential microbatches per update.
        trainable_fraction : float, optional
            Fraction in [0, 1] receiving gradients and optimizer state.
        communication_buffer_fraction : float, optional
            Gradient staging buffer as a fraction in [0, 1] of local gradients.
        sequence_parallel : bool, optional
            If true, split stored activations across the TP group. If false,
            retain the unsharded activation estimate as a conservative bound.

        Returns
        -------
        TrainingMemoryResult
            Weights, gradients, optimizer state, activations, buffers, and total
            memory in GB. ``feasible`` compares that modeled total with device
            capacity; it excludes unmodeled runtime reserves.

        See Also
        --------
        mlsysim.physics.memory.calc_activation_memory : Activation retention model.
        mlsysim.physics.memory.optimizer_state_bytes : Optimizer storage convention.

        Notes
        -----
        Equation: total = weights + gradients + optimizer + activations +
        communication_buffers. Parameter state first divides by TP * PP * EP;
        ZeRO then divides selected terms by DP. Local microbatch is
        max(1, ceil(global_batch / (DP * accumulation_steps))). Layers per stage
        are max(1, ceil(layers / PP)). Activation bytes follow
        ``calc_activation_memory`` and divide by TP when sequence parallelism
        is enabled. Buffers contain a gradient fraction plus send/receive
        pipeline boundary activations.

        Assumptions: balanced parameter sharding and uniform layer geometry.
        The precision byte count rescales the activation model's FP16 baseline.
        ZeRO stages are cumulative. Trainable fraction affects gradients and
        optimizer state but not resident weights.

        Excludes: allocator reserves, implementation workspaces, fragmentation,
        uneven expert placement, and detailed interleaved pipeline lifetimes.
        A capacity pass does not guarantee a runnable deployment.

        References
        ----------
        Rajbhandari et al. (2020), "ZeRO: Memory Optimizations Toward Training
        Trillion Parameter Models." https://arxiv.org/abs/1910.02054 (bibliography key: rajbhandari2020).
        Korthikanti et al. (2023), "Reducing Activation Recomputation in Large
        Transformer Models." https://arxiv.org/abs/2205.05198 (bibliography key: korthikanti2023).

        Examples
        --------
        >>> from mlsysim import Models, Hardware
        >>> result = TrainingMemoryModel().solve(Models.Language.Llama3_8B,
        ...     Hardware.Cloud.H100, batch_size=1, seq_len=128)
        >>> result.total_memory.to("GB").magnitude > 0
        True
        >>> result.feasible == (result.total_memory <= result.available_memory)
        True
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
        if tp_size > 1 and sequence_parallel:
            # (Added 2026-09-15: activations were never sharded by TP, so a
            # TP=8 Llama-3 70B step reported ~8x its stored activations.)
            activations = activations / tp_size

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
        """Allocate a training compute budget between parameters and tokens.

        Parameters
        ----------
        compute_budget : Quantity
            Finite positive training work in FLOPs, or a finite positive duration interpreted
            as reference-H100 compute time at the declared sustained efficiency.
            Any compatible FLOP prefix is accepted and normalized before calculation.
        target_model_size : Quantity, optional
            Finite positive fixed parameter count. If omitted, use the registered
            compute-optimal tokens-per-parameter ratio.

        Returns
        -------
        ScalingResult
            Modeled parameter and token counts, compute budget, and dimensionless
            tokens per parameter. The result does not include a quality estimate
            or a memory-feasibility decision.

        Notes
        -----
        Equation: C = k P D, where k is the registered compute coefficient.
        With D = r P, P = sqrt(C / (k r)); when P is fixed, D = C / (k P).
        A time input becomes C = duration * reference_peak * reference_MFU.

        Assumptions: parameter-dominated dense training work and a constant
        registered optimal token ratio. The time conversion uses a declared
        reference accelerator rather than an arbitrary fleet supplied by a user.

        Excludes: inference cost, data-quality variation, architecture-dependent
        loss prediction, attention work, and hardware memory capacity.

        References
        ----------
        Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models."
        https://arxiv.org/abs/2203.15556 (bibliography key: hoffmann2022chinchilla).
        The fixed-ratio allocation is the package's simplified accounting form.

        Examples
        --------
        >>> result = ScalingModel().solve(Q_(1200000, "flop"))
        >>> result.optimal_parameters.to("count").magnitude > 0
        True
        >>> result.optimal_tokens.to("count").magnitude > 0
        True
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

        # Prefixes describe the same work: normalize before extracting magnitude.
        c_flops = c_flops.to(ureg.flop)
        validate_positive(c_flops, "compute_budget")

        if target_model_size is not None:
            # Pinned model size: spend the whole budget on tokens via C = 6PD,
            # i.e. D = C / (6P). The result may be over- or under-trained
            # relative to the Chinchilla-optimal D = 20P.
            p_opt = target_model_size.to(ureg.count).magnitude
            validate_positive(p_opt, "target_model_size")
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
    "TrainingRunModel",
    "TrainingStateModel",
]


class TrainingRunModel(ForwardModel):
    """
    Wall-clock time of a training run, from peak throughput to delivered time.

    The chain starts from the ``6PD`` training FLOPs at every device's peak,
    divides by the local compute efficiency (the compute-only baseline), then
    by the scaling efficiency (communication and pipeline bubbles), then
    multiplies by each operational overhead in order (checkpointing, failures
    and restarts, maintenance), each relative to the subtotal before it.

    Literature Source:
    1. Kaplan et al. (2020) and Hoffmann et al. (2022), the 6PD rule.
    2. Narayanan et al. (2021), scaling efficiency at cluster scale.
    """
    requires = ("workload", "fleet")
    produces = None

    def solve(self, model: Workload, fleet, local_efficiency: float, scaling_efficiency: float,
              overheads: dict, precision: str = "fp8", tokens: Quantity = None):
        """Compose compute, scaling, and operational factors into training time.

        Parameters
        ----------
        model : Workload
            Parameter count and, when tokens is omitted, training-token count.
        fleet : Fleet
            Accelerator count and per-device peak compute at the chosen precision.
        local_efficiency : float
            Positive dimensionless fraction of local peak throughput, usually
            at most one. It acts before cluster scaling losses.
        scaling_efficiency : float
            Positive fraction of compute throughput retained after communication
            and pipeline effects, usually at most one.
        overheads : dict
            Ordered mapping from stage name to nonnegative fractional time
            overhead. Each fraction applies to the preceding subtotal.
        precision : str, optional
            Compute precision key; unsupported keys use the hardware's base peak.
        tokens : Quantity, optional
            Positive total training-token count overriding model.training_tokens.

        Returns
        -------
        TrainingRunResult
            Work in FLOPs and peak, compute, scaled, intermediate, and total times
            in hours. Delivered fraction is peak time divided by total time;
            this calculation does not check model residency.

        See Also
        --------
        mlsysim.physics.transformer.calc_transformer_training_flops : Training work.
        mlsysim.physics.performance.dTime : Work divided by effective compute rate.

        Notes
        -----
        Equation: F = 6 P D; T_peak = F / (N peak); T_compute = T_peak / eta;
        T_scaled = T_compute / scaling_efficiency. Each overhead f multiplies
        its preceding subtotal by (1 + f).

        Assumptions: homogeneous accelerator throughput and supplied efficiencies
        that already summarize kernel and cluster behavior. The overhead mapping
        is an ordered accounting scenario, not a measured trace.

        Excludes: overlap among overhead stages, explicit communication schedules,
        activation residency, data starvation, and changing efficiency during a run.

        References
        ----------
        Derivation: work divided by aggregate rate gives elapsed compute time;
        sequential fractional overheads multiply the running subtotal.
        Parameter-dominated training work follows the cited contract in
        ``mlsysim.physics.transformer.calc_transformer_training_flops``.

        Examples
        --------
        >>> from mlsysim import Models, Systems
        >>> result = TrainingRunModel().solve(Models.Language.Llama3_8B,
        ...     Systems.Clusters.Research_256, local_efficiency=0.5,
        ...     scaling_efficiency=0.8, overheads={"checkpoint": 0.1},
        ...     tokens=Q_(1000, "count"))
        >>> result.total_time > result.scaled_time > result.peak_time
        True
        """
        from ..results import TrainingRunResult
        from ...physics.performance import dTime
        from ...physics.transformer import calc_transformer_training_flops

        hardware = fleet.node.accelerator
        peak = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)
        n_tokens = tokens if tokens is not None else model.training_tokens
        flops = calc_transformer_training_flops(model.parameters, n_tokens)
        n = fleet.total_accelerators
        t_peak = dTime(flops, n, peak, 1.0).to("hour")
        t_compute = dTime(flops, n, peak, local_efficiency).to("hour")
        t = t_compute / scaling_efficiency
        t_scaled = t
        stages = {}
        for name, fraction in overheads.items():
            t = t * (1 + fraction)
            stages[name] = t
        return TrainingRunResult(
            total_flops=flops,
            peak_time=t_peak,
            compute_time=t_compute,
            scaled_time=t_scaled,
            stage_times=stages,
            total_time=t,
            delivered_fraction=(t_peak / t).to("").magnitude,
        )


class TrainingStateModel(ForwardModel):
    """
    Persistent mixed-precision Adam state and how parallelism shards it.

    Per parameter: FP16 weights and gradients (2 B each), FP32 master weights
    (4 B), and two FP32 Adam moments (4 B each), 16 B in all. Tensor
    parallelism splits every term across the TP group; ZeRO-style sharding
    further splits the optimizer moments across the data-parallel replicas.
    Activations are a stated total, not derived (they depend on sequence
    length, microbatch, and recomputation).

    Literature Source:
    1. Rajbhandari et al. (2020), "ZeRO" (16 B/param accounting, sharding).
    2. Shoeybi et al. (2019), "Megatron-LM" (tensor parallelism).
    """
    requires = ("workload",)
    produces = None

    def solve(self, model: Workload, tp_size: int, dp_size: int, activations: Quantity):
        """Account for mixed-precision Adam state under selected sharding choices.

        Parameters
        ----------
        model : Workload
            Model parameter count; this accounting uses FP16 weights and gradients.
        tp_size : int
            Positive tensor-parallel degree splitting each state term.
        dp_size : int
            Positive data-parallel degree additionally splitting Adam moments in
            the reported moments-sharded comparison.
        activations : Quantity
            Nonnegative total activation storage in bytes, supplied by the caller
            rather than derived from a batch shape.

        Returns
        -------
        TrainingStateResult
            Full and per-device state components in GB, including TP-only,
            offloaded, and moments-sharded comparisons. These totals do not imply
            device feasibility because no capacity is supplied.

        See Also
        --------
        mlsysim.engine.solvers.training.TrainingMemoryModel.solve : Full state budget.

        Notes
        -----
        Equation: W = G = 2 P bytes; master = 4 P bytes; moments = 8 P bytes.
        Persistent state is W + G + master + moments. TP-only total is
        (persistent + activations) / TP. The moments-sharded term is
        moments / (TP * DP); weights remain W / TP in that comparison.

        Assumptions: uniform dense parameter storage and even sharding. The
        moments-only comparison does not shard the master copy and must not be
        read as a complete implementation of every ZeRO state transition.

        Excludes: communication buffers, activation recomputation, allocator
        reserve, offload transfer latency, and a hardware capacity check.

        References
        ----------
        Rajbhandari et al. (2020), "ZeRO: Memory Optimizations Toward Training
        Trillion Parameter Models." https://arxiv.org/abs/1910.02054 (bibliography key: rajbhandari2020).
        The moments-only output is a declared partial-sharding comparison;
        ``TrainingMemoryModel`` provides cumulative ZeRO-stage accounting.

        Examples
        --------
        >>> from mlsysim import Models
        >>> result = TrainingStateModel().solve(Models.Language.Llama3_8B,
        ...     tp_size=2, dp_size=4, activations=Q_(1, "GB"))
        >>> result.zero_moments_per_device < result.tp_moments_per_device
        True
        """
        from ..results import TrainingStateResult
        from ...core.units import BYTES_FP16, BYTES_FP32

        p = model.parameters
        weights = model.size_in_bytes(BYTES_FP16).to("GB")
        gradients = weights
        master = (p.to("count").magnitude * BYTES_FP32).to("GB")
        moment = master
        moments = 2 * moment
        persistent = weights + gradients + master + moments
        act = activations.to("GB")
        tp = tp_size
        zero = tp_size * dp_size
        return TrainingStateResult(
            weights=weights, gradients=gradients, master_weights=master,
            adam_moment=moment, optimizer_moments=moments, persistent_state=persistent,
            activations=act,
            tp_weights_per_device=weights / tp, tp_gradients_per_device=gradients / tp,
            tp_master_per_device=master / tp, tp_moments_per_device=moments / tp,
            tp_activations_per_device=act / tp,
            tp_total_per_device=(persistent + act) / tp,
            tp_offloaded_per_device=(persistent + act - master - moments) / tp,
            zero_moments_per_device=moments / zero,
            zero_moments_per_tp_group=moments / dp_size,
            zero_weights_and_moments_per_device=weights / tp + moments / zero,
        )
