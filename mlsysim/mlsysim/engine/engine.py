from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List
from ..core.units import ureg, Q_, resolve_precision
from . import calibration as cal
from ..physics import calc_bottleneck
from ..core.exceptions import OOMError
from ..core._validation import validate_range, validate_at_least
from ..models.types import Workload, TransformerWorkload
from ..hardware.types import HardwareNode, Quantity

__all__ = ["PerformanceProfile", "Engine"]

class PerformanceProfile(BaseModel):
    """The resulting performance metrics bounded by hardware limits."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    latency: Quantity
    latency_compute: Quantity
    latency_memory: Quantity
    latency_overhead: Quantity
    throughput: Quantity
    bottleneck: str
    arithmetic_intensity: Quantity
    energy: Quantity
    memory_footprint: Quantity
    peak_flops_actual: Quantity
    peak_bw_actual: Quantity
    mfu: float # Model FLOPs Utilization
    hfu: float # Hardware FLOPs Utilization
    feasible: bool
    constraint_trace: Optional[List[str]] = Field(
        default=None,
        description="A trace of physical constraints evaluated during the solve. If feasible=False, this explains why."
    )
    # Offload fields — populated when model spills beyond HBM to host memory
    offload_spill_bytes: Optional[Quantity] = None
    offload_effective_bw: Optional[Quantity] = None
    # True when dispatch + framework layer tax exceeds both the compute and
    # memory roofline terms; the bottleneck label reports the larger roofline
    # ceiling, so small models at low batch can be overhead-bound in practice.
    overhead_dominated: bool = False

    @property
    def energy_per_inference(self) -> Quantity:
        """Return energy for the batch-latency window.

        Returns
        -------
        Quantity
            Stored batch energy in joules. Despite this legacy property name,
            the value is not divided by batch size.

        Notes
        -----
        Equation: energy_per_inference = energy. Per-sample energy is energy / B,
        where B is the batch size supplied to Engine.solve.

        Assumptions: the caller retains B; PerformanceProfile does not store it.
        The underlying energy estimate uses average power over the batch time.

        Excludes: a separate measurement of individual sample energy, device
        power-meter traces, and division by the number of samples.

        References
        ----------
        Derivation: this compatibility property returns the stored batch energy
        without arithmetic. Dividing total energy by the sample count gives
        an average per sample, not a separately simulated quantity.

        Examples
        --------
        >>> from mlsysim import Engine, Hardware, Models
        >>> profile = Engine.solve(Models.Vision.ResNet50, Hardware.Cloud.H100)
        >>> profile.energy_per_inference == profile.energy
        True
        """
        return self.energy

    def summary(self) -> str:
        """Returns a human-readable summary of the performance profile."""
        lines = [
            f"Feasible: {self.feasible}",
            f"Bottleneck: {self.bottleneck}",
            f"Latency: {self.latency:~P}",
            f"  Compute: {self.latency_compute:~P}",
            f"  Memory: {self.latency_memory:~P}",
            f"  Overhead: {self.latency_overhead:~P}",
            f"Throughput: {self.throughput:~P}",
            f"MFU: {self.mfu:.3f}",
            f"HFU: {self.hfu:.3f}",
            f"Memory Footprint: {self.memory_footprint:~P}",
        ]
        return "\n".join(lines)


class Engine:
    """
    The core analytical engine for single-node Roofline performance modeling.

    Maps a (Workload, HardwareNode) pair to a PerformanceProfile by applying
    the Roofline model: the workload is "lowered" into a hardware-agnostic
    ComputationGraph, then bounded by the hardware's compute ceiling and
    memory bandwidth ceiling.

    The Iron Law of ML Training
    ---------------------------
    Compute-dominated training time can be organized as:

        Time = FLOPs / (N × Peak_FLOPS × MFU × η_scaling × Goodput)

    where N is device count, MFU is Model FLOPs Utilization, η_scaling is
    scaling efficiency, and Goodput accounts for failures and checkpointing.
    This accounting view requires mutually consistent definitions of the
    efficiency factors. Engine.solve evaluates compute and memory bounds
    on one device; DistributedModel adds fleet coordination terms. Neither
    this expression nor a single efficiency factor replaces those bounds.

    Source: Williams et al. (2009), "Roofline: An Insightful Visual
    Performance Model for Floating-Point Programs and Multicore Architectures."
    """

    @staticmethod
    def solve(model: Workload, hardware: HardwareNode, batch_size: int = 1,
              precision: str = "fp16", efficiency: float = 0.5,
              raise_errors: bool = False, is_training: bool = False,
              seq_len: int = 2048, zero_stage: int = 0, dp_size: int = 1,
              is_lora: bool = False, activation_recomputation: bool = False) -> PerformanceProfile:
        """Estimate one device's compute, memory, and capacity limits.

        Parameters
        ----------
        model : Workload
            Workload whose lower method supplies FLOPs and weight bytes. A dense
            autoregressive Transformer graph represents one token; a vision
            graph represents one input. Training multiplies Transformer work
            by seq_len before forming a batch.
        hardware : HardwareNode
            Device specification with FLOP/s, byte/s, byte capacity, dispatch
            time, and optional TDP in watts and alternate memory tiers.
        batch_size : int, default: 1
            Positive number of samples per batch, or sequences during training.
        precision : str, default: 'fp16'
            Registered numeric format controlling bytes per weight and the
            precision-specific compute ceiling. Missing device precision
            throughput falls back to the device's default peak.
        efficiency : float, default: 0.5
            Declared compute efficiency in [1e-9, 1]. It derates FLOP/s only;
            bandwidth remains the selected tier's peak specification. This
            input is an assumption, not a fitted bandwidth efficiency.
        raise_errors : bool, default: False
            Raise OOMError for a footprint above device capacity. Otherwise
            retain feasible=False and estimate offload-limited latency.
        is_training : bool, default: False
            Include backward work and training state. Inference capacity
            checks weights only; use ServingModel for KV-aware capacity.
        seq_len : int, default: 2048
            Positive tokens per training sequence. Scales Transformer work
            and activation storage; generic inference remains one forward step.
        zero_stage : int, default: 0
            ZeRO stage 0, 1, 2, or 3, passed to Transformer training memory.
        dp_size : int, default: 1
            Positive data-parallel count used for ZeRO state sharding. Does
            not add distributed execution or communication to this solver.
        is_lora : bool, default: False
            Use the training-memory adapter approximation. The generic
            non-Transformer fallback uses a separate 1.1-times-weights heuristic.
        activation_recomputation : bool, default: False
            Recompute a forward pass during training: four rather than three
            forward-equivalent work units, paired with full activation recompute.

        Returns
        -------
        PerformanceProfile
            Modeled batch latency in milliseconds, throughput in samples/s,
            energy in joules, byte footprint, utilization fractions, selected
            resource bound, and feasibility trace. A false capacity flag remains
            false even when an offload-latency estimate is returned.

        See Also
        --------
        mlsysim.physics.performance.calc_bottleneck : Compute and memory bounds.
        mlsysim.solvers.ServingModel : KV-aware prefill and decode estimates.
        mlsysim.models.types.TransformerWorkload.training_memory : Training state.

        Notes
        -----
        Equation: T = max(F_batch / (eta * F_peak), W * (1 + 0.1 * B) / BW)
        + T_dispatch + L * T_layer * launch_multiplier. Here W is weight bytes,
        B is batch size, and launch_multiplier is three for training and one
        for inference. Training work is three forward passes, or four with full
        recomputation. Throughput is B / T; energy is average_power * T.

        Assumptions: activation traffic is the declared 0.1 * W per-sample
        heuristic, not an architecture trace. Compute and memory bounds overlap
        ideally. The bandwidth path selects host offload, resident SRAM, flash,
        or primary memory. Offload charges all traffic at the limiting link
        rate, not only the spilled fraction. Power is TDP times an idle-plus-HFU
        model with fractions in engine.calibration; missing TDP yields zero
        estimated energy. Fixed dispatch and layer taxes are modeling defaults.

        Excludes: KV-cache capacity, attention-context work in generic inference,
        allocator reserve, kernel workspaces, queueing, serving concurrency,
        and communication between devices. Feasibility is a footprint check,
        not a deployment guarantee. The bottleneck label compares compute and
        memory terms; overhead_dominated separately flags a larger additive tax.

        References
        ----------
        Williams et al. (2009), "Roofline: An Insightful Visual Performance Model
        for Multicore Architectures."
        https://doi.org/10.1145/1498765.1498785 (bibliography key: williams2009).
        The activation traffic, layer tax, and power coefficients are separate
        implementation assumptions, not constants established by Roofline.

        Examples
        --------
        >>> from mlsysim import Engine, Hardware, Models
        >>> result = Engine.solve(Models.Vision.ResNet50, Hardware.Cloud.H100)
        >>> result.feasible
        True
        >>> result.latency >= max(result.latency_compute, result.latency_memory)
        True
        """
        # 0. Input validation
        validate_range(efficiency, 1e-9, 1.0, "efficiency")  # >0 to prevent division by zero
        validate_at_least(batch_size, 1, "batch_size")
        validate_at_least(seq_len, 1, "seq_len")

        # 1. Canonicalize precision and map it to bytes per parameter. Unknown
        # precision names fail here instead of falling back to FP16 storage.
        precision, bpp = resolve_precision(precision)

        # 2. Resolve peak FLOPS for the requested precision
        if precision in hardware.compute.precision_flops:
            peak_flops = hardware.compute.precision_flops[precision]
        else:
            peak_flops = hardware.compute.peak_flops
            if precision not in ("fp16", "bf16") and hardware.compute.precision_flops:
                import warnings
                warnings.warn(
                    f"Precision '{precision}' not in {hardware.name} precision_flops "
                    f"(available: {list(hardware.compute.precision_flops.keys())}); "
                    f"using default peak_flops (FP16). Results may be inaccurate.",
                    stacklevel=2,
                )

        # 3. Lower the workload to a ComputationGraph
        graph = model.lower(bpp)
        # Transformer lowering is per token; training batches contain sequences.
        # CNN and other workload graphs already describe one complete sample.
        tokens_per_sample = seq_len if is_training and isinstance(model, TransformerWorkload) else 1
        sample_ops = graph.total_ops * tokens_per_sample

        # 4. Resolve memory footprint and FLOPs based on training vs inference
        if is_training:
            # Training is roughly 3x FLOPs of inference (1 forward, 2 backward passes)
            # With activation recomputation, it adds another forward pass (+33% total training ops) -> 4x base ops
            ops_multiplier = 4 if activation_recomputation else 3
            base_ops = sample_ops * ops_multiplier
            
            # Calculate training memory footprint
            if hasattr(model, 'training_memory'):
                # Pair FULL recomputation memory with the 4x ops multiplier above:
                # full recompute stores only layer inputs and re-runs the forward
                # pass (+1 forward), which is exactly the 4/3 ops convention.
                # Selective checkpointing would save less memory AND cost fewer
                # extra FLOPs; mixing the two charges the worst of both.
                strategy = "full" if activation_recomputation else "none"
                memory_footprint = model.training_memory(
                    batch_size=batch_size, seq_len=seq_len, precision=precision, 
                    strategy=strategy, zero_stage=zero_stage, dp_size=dp_size, is_lora=is_lora
                )
            else:
                # Fallback for non-Transformer workloads (no training_memory
                # method): weights + gradients (same width) + Adam
                # mixed-precision optimizer state (FP32 master + moments =
                # 12 B/param, cal.TRAINING_OPTIMIZER_BYTES_ADAM). The old 3x
                # weight-bytes heuristic understated Adam accounting ~2.7x
                # at fp16. (Activations still excluded, as documented.)
                if is_lora:
                    memory_footprint = graph.weight_bytes * 1.1
                else:
                    n_params = (graph.weight_bytes / bpp).to_base_units()
                    opt_state = n_params * Q_(cal.TRAINING_OPTIMIZER_BYTES_ADAM, "byte")
                    memory_footprint = graph.weight_bytes * 2 + opt_state
        else:
            base_ops = graph.total_ops
            memory_footprint = graph.weight_bytes

        # 5. Feasibility check — does the model fit in memory?
        feasible = memory_footprint <= hardware.memory.capacity
        offload_spill = None
        offload_bw = None
        constraint_trace = []

        if feasible:
            constraint_trace.append(f"Memory Wall: Passed. Required {memory_footprint.to('GB'):~P} <= Available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}.")
        else:
            constraint_trace.append(f"Memory Wall: FAILED. Required {memory_footprint.to('GB'):~P} > Available {hardware.memory.capacity.to('GB'):~P} on {hardware.name}.")

        if not feasible and raise_errors:
            raise OOMError(
                f"Model '{model.name}' requires {memory_footprint.to('GB'):~P} "
                f"but hardware '{hardware.name}' has only "
                f"{hardware.memory.capacity.to('GB'):~P}.",
                required_bytes=memory_footprint,
                available_bytes=hardware.memory.capacity,
            )

        # 6. Roofline bottleneck analysis
        # Scale compute by batch_size: processing B samples requires B × ops_per_sample.
        # Note: if is_training, base_ops already represents the full training step cost per sample.
        effective_flops = peak_flops * efficiency

        if not feasible:
            # Offload path: model spills beyond primary memory → bandwidth degrades
            offload_spill = memory_footprint - hardware.memory.capacity
            # Whole-traffic convention: ALL memory traffic is charged at the
            # offload-limited bandwidth, not just the spilled fraction — a
            # deliberate pessimistic simplification of host-offload paging.
            pcie_fallback = Q_(cal.FALLBACK_PCIE_BANDWIDTH_GB_S, "GB/s")
            pcie_bw = (
                getattr(
                    hardware.interconnect,
                    'bandwidth_per_direction',
                    getattr(hardware.interconnect, 'bandwidth', pcie_fallback),
                )
                if hardware.interconnect else pcie_fallback
            )
            effective_bw = min(hardware.memory.bandwidth, pcie_bw)
            offload_bw = effective_bw
        elif (hardware.memory.sram_capacity is not None
              and hardware.memory.sram_bandwidth is not None
              and graph.weight_bytes <= hardware.memory.sram_capacity):
            # SRAM-resident path: working set fits in on-chip SRAM.
            # For GPUs: captures FlashAttention-style tiled execution.
            # For TinyML: small models that fit entirely in SRAM bypass flash.
            effective_bw = hardware.memory.sram_bandwidth
        elif hardware.memory.flash_bandwidth is not None:
            # TinyML flash path: weights live in flash, activations in SRAM.
            # Use flash bandwidth for weight loading (the typical bottleneck
            # when the model is too large for SRAM).
            effective_bw = hardware.memory.flash_bandwidth
        else:
            effective_bw = hardware.memory.bandwidth
        batch_ops = base_ops * batch_size

        # Guard against zero ops or zero bandwidth
        if batch_ops.magnitude == 0:
            compute_time = Q_("0 ms")
        else:
            compute_time = (batch_ops / effective_flops).to("ms")

        # Memory traffic for batched workloads: weights are loaded once from HBM,
        # but activations scale with batch_size. For batch=1, traffic ≈ weight_bytes.
        # For larger batches, activation I/O becomes significant.
        # This captures the Roofline regime transition as batch_size grows.
        # Computed ONCE and shared by the latency term and the bottleneck
        # classifier so the two can never silently diverge.
        activation_bytes = graph.weight_bytes * 0.1 * batch_size  # ~10% of weights per sample (heuristic)
        actual_memory_traffic = graph.weight_bytes + activation_bytes

        if effective_bw.magnitude == 0:
            memory_time = Q_("0 ms")
        else:
            memory_time = (actual_memory_traffic / effective_bw).to("ms")

        roofline = calc_bottleneck(
            batch_ops, actual_memory_traffic, effective_flops, effective_bw
        )
        bottleneck = roofline["bottleneck"]

        # 6. Latency = max(compute, memory) + dispatch overhead + layer tax
        dispatch_tax = hardware.dispatch_tax
        
        # Calculate layer-wise software tax
        # Source: Reddi et al. (2025), Wall 7 (Framework Overhead)
        num_layers = getattr(model, 'layers', 1) or 1
        # Training has ~3x the launches of inference (Fwd, GradW, GradA)
        launch_multiplier = 3 if is_training else 1
        layer_tax = Q_(num_layers * cal.FRAMEWORK_LAYER_TAX_MS * launch_multiplier, "ms")
        
        latency = max(compute_time.to("ms").magnitude, memory_time.to("ms").magnitude) * ureg.ms + dispatch_tax + layer_tax

        # 7. Throughput
        if latency.magnitude > 0:
            throughput = Q_(batch_size / latency.to("s").magnitude, "1/s")
        else:
            throughput = Q_(0, "1/s")

        # 8-9. MFU, HFU, and Energy (energy depends on utilization, so compute MFU/HFU first)
        # MFU = model FLOPs / (step_time × peak): only the FLOPs the model
        # requires (1 fwd + 2 bwd for training), EXCLUDING recomputation.
        # HFU = hardware FLOPs / (step_time × peak): what the silicon actually
        # executed, INCLUDING the recompute forward pass. Without activation
        # recomputation the two are identical by definition.
        # Source: Chowdhery et al. (2022), "PaLM", Appendix B;
        #         Korthikanti et al. (2022), "Reducing Activation Recomputation".
        # IMPORTANT: Use Pint unit-aware division to handle TFLOPS/s vs flop correctly.
        latency_s = latency.to("s")
        model_ops = sample_ops * (3 if is_training else 1) * batch_size
        if latency_s.magnitude > 0 and peak_flops.magnitude > 0:
            mfu = ((model_ops / latency_s).to(peak_flops.units) / peak_flops).to_base_units().magnitude
            mfu = max(0.0, min(mfu, 1.0))  # Clamp to [0, 1]
            # batch_ops carries the 4x multiplier when recomputation is on,
            # so hfu/mfu = 4/3 under recompute and hfu == mfu otherwise.
            hfu = ((batch_ops / latency_s).to(peak_flops.units) / peak_flops).to_base_units().magnitude
            hfu = max(0.0, min(hfu, 1.0))  # Clamp to [0, 1]
        else:
            mfu = 0.0
            hfu = 0.0

        # Energy estimate (energy-proportional model)
        # Consistent with SustainabilityModel: 30% idle + 70% * utilization.
        # Utilization is HFU (the silicon's actual busy fraction, including
        # recompute work); identical to MFU when recomputation is off.
        # Source: Barroso & Hölzle (2007), "The Case for Energy-Proportional Computing"
        if hardware.tdp is not None:
            avg_power = hardware.tdp * (cal.ENERGY_IDLE_FRACTION + cal.ENERGY_DYNAMIC_FRACTION * hfu)
            energy = (avg_power * latency.to("s")).to("J")
        else:
            energy = Q_("0 J")

        return PerformanceProfile(
            latency=latency,
            latency_compute=compute_time,
            latency_memory=memory_time,
            latency_overhead=dispatch_tax + layer_tax,
            throughput=throughput,
            bottleneck=bottleneck,
            arithmetic_intensity=Q_(roofline["intensity"], "flop/byte"),
            energy=energy,
            memory_footprint=memory_footprint,
            peak_flops_actual=effective_flops,
            peak_bw_actual=effective_bw,
            mfu=mfu,
            hfu=hfu,
            feasible=feasible,
            constraint_trace=constraint_trace,
            offload_spill_bytes=offload_spill,
            offload_effective_bw=offload_bw,
            overhead_dominated=(
                (dispatch_tax + layer_tax).to("ms").magnitude
                > max(compute_time.to("ms").magnitude, memory_time.to("ms").magnitude)
            ),
        )

    @staticmethod
    def sweep(model: Workload, hardware_list: list, batch_sizes: list = None,
              precisions: list = None, efficiency: float = 0.5) -> list:
        """Evaluate feasible inference configurations over a Cartesian product.

        Parameters
        ----------
        model : Workload
            Fixed workload to lower for each device and precision.
        hardware_list : list of HardwareNode
            Devices to evaluate in their supplied order; an empty list gives
            no results.
        batch_sizes : list of int, optional
            Positive sample counts. None or an empty list uses [1].
        precisions : list of str, optional
            Registered numeric formats. None or an empty list uses ['fp16'].
        efficiency : float, default: 0.5
            Declared compute efficiency in [1e-9, 1], shared by all configurations.

        Returns
        -------
        list of dict
            One entry per capacity-feasible combination, containing hardware
            name, batch_size, precision, and a PerformanceProfile with units.
            Capacity failures are omitted rather than returned as failed rows.

        See Also
        --------
        mlsysim.engine.engine.Engine.solve : Per-configuration model and limits.

        Notes
        -----
        Equation: results = [solve(model, h, b, p) for h, b, p in
        hardware_list x batch_sizes x precisions if the footprint fits]. The
        latency and energy equations are those of Engine.solve.

        Assumptions: every point uses generic inference and the same efficiency;
        input order determines result order. OOMError alone is caught; invalid
        parameters and other errors still propagate.

        Excludes: training configuration sweeps, optimization, queueing, measured
        hardware execution, and a record of omitted infeasible configurations.
        No runtime bound follows from the number of configurations alone.

        References
        ----------
        Derivation: enumeration of the Cartesian product delegates each point
        to Engine.solve and filters its capacity failures. There is no separate
        performance equation or cross-configuration scheduling model.

        Examples
        --------
        >>> from mlsysim import Engine, Hardware, Models
        >>> rows = Engine.sweep(Models.Vision.ResNet50, [Hardware.Cloud.H100], [1, 2])
        >>> [row['batch_size'] for row in rows]
        [1, 2]
        >>> all(row['profile'].feasible for row in rows)
        True
        """
        results = []
        batch_sizes = batch_sizes or [1]
        precisions = precisions or ["fp16"]
        for hw in hardware_list:
            for bs in batch_sizes:
                for prec in precisions:
                    try:
                        profile = Engine.solve(
                            model, hw, batch_size=bs, precision=prec,
                            efficiency=efficiency, raise_errors=True,
                        )
                        results.append({
                            "hardware": hw.name,
                            "batch_size": bs,
                            "precision": prec,
                            "profile": profile,
                        })
                    except OOMError:
                        # Skip infeasible configurations (model doesn't fit in memory)
                        pass
        return results
