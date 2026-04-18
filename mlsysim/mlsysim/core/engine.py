from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List
from .constants import ureg, Q_, BYTES_FP16, PRECISION_MAP
from .defaults import HFU_MFU_RATIO
from .formulas import calc_bottleneck
from .exceptions import OOMError
from ._validation import validate_range, validate_at_least
from ..models.types import Workload
from ..hardware.types import HardwareNode, Quantity

__all__ = ["PerformanceProfile", "Engine"]

class PerformanceProfile(BaseModel):
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

    @property
    def energy_per_inference(self) -> Quantity:
        """Energy cost for a single inference pass.

        This is the total energy for the batch-latency window. For per-sample
        energy, divide by the batch size used in Engine.solve(). The engine
        does not store batch_size on the profile to keep it hardware-centric,
        so the caller must apply the division when needed.
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
    Every solver in mlsysim ultimately decomposes into this equation:

        Time = FLOPs / (N × Peak_FLOPS × MFU × η_scaling × Goodput)

    where N is device count, MFU is Model FLOPs Utilization, η_scaling is
    scaling efficiency (communication overhead), and Goodput accounts for
    failures and checkpointing. This engine evaluates the single-node case
    (N=1, η_scaling=1, Goodput=1); DistributedModel handles the fleet case.

    Source: Williams et al. (2009), "Roofline: An Insightful Visual
    Performance Model for Floating-Point Programs and Multicore Architectures."
    """

    @staticmethod
    def solve(model: Workload, hardware: HardwareNode, batch_size: int = 1,
              precision: str = "fp16", efficiency: float = 0.5,
              raise_errors: bool = False, is_training: bool = False,
              seq_len: int = 2048, zero_stage: int = 0, dp_size: int = 1,
              is_lora: bool = False, activation_recomputation: bool = False) -> PerformanceProfile:
        """
        Solves the Roofline performance profile for a single hardware node.

        Parameters
        ----------
        model : Workload
            The model architecture to simulate.
        hardware : HardwareNode
            The target hardware node.
        batch_size : int
            Batch size for throughput calculation.
        precision : str
            Numerical precision ('fp32', 'fp16', 'int8', 'int4').
        efficiency : float
            Achieved compute efficiency as a fraction of peak (0.0 to 1.0).
        raise_errors : bool
            If True, raise OOMError when the model does not fit in memory.
        is_training : bool
            Whether to model training (impacts memory and FLOPs).
        seq_len : int
            Sequence length (used for training memory footprint).
        zero_stage : int
            ZeRO optimization stage (0, 1, 2, 3) for memory sharding.
        dp_size : int
            Data parallel size for ZeRO sharding.
        is_lora : bool
            Whether to use Low-Rank Adaptation (reduces optimizer memory).
        activation_recomputation : bool
            Whether to use activation recomputation (saves memory, adds 33% FLOPs).

        Returns
        -------
        PerformanceProfile
            Complete performance characterization of the workload on the hardware.
        """
        # 0. Input validation
        validate_range(efficiency, 1e-9, 1.0, "efficiency")  # >0 to prevent division by zero
        validate_at_least(batch_size, 1, "batch_size")

        # 1. Map precision to bytes per parameter
        bpp = PRECISION_MAP.get(precision, BYTES_FP16)

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

        # 4. Resolve memory footprint and FLOPs based on training vs inference
        if is_training:
            # Training is roughly 3x FLOPs of inference (1 forward, 2 backward passes)
            # With activation recomputation, it adds another forward pass (+33% total training ops) -> 4x base ops
            ops_multiplier = 4 if activation_recomputation else 3
            base_ops = graph.total_ops * ops_multiplier
            
            # Calculate training memory footprint
            if hasattr(model, 'training_memory'):
                strategy = "selective" if activation_recomputation else "none"
                memory_footprint = model.training_memory(
                    batch_size=batch_size, seq_len=seq_len, precision=precision, 
                    strategy=strategy, zero_stage=zero_stage, dp_size=dp_size, is_lora=is_lora
                )
            else:
                # Fallback for non-Transformer workloads: 3x weights + gradients + states
                memory_footprint = graph.weight_bytes * (3 if not is_lora else 1.1)
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
            pcie_bw = (
                getattr(hardware.interconnect, 'bandwidth', Q_("64 GB/s"))
                if hardware.interconnect else Q_("64 GB/s")
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

        if effective_bw.magnitude == 0:
            memory_time = Q_("0 ms")
        else:
            # Memory traffic for batched workloads: weights are loaded once from HBM,
            # but activations scale with batch_size. For batch=1, traffic ≈ weight_bytes.
            # For larger batches, activation I/O becomes significant.
            # This captures the Roofline regime transition as batch_size grows.
            activation_bytes = graph.weight_bytes * 0.1 * batch_size  # ~10% of weights per sample (heuristic)
            actual_memory_traffic = graph.weight_bytes + activation_bytes
            memory_time = (actual_memory_traffic / effective_bw).to("ms")

        roofline = calc_bottleneck(
            batch_ops, graph.weight_bytes + graph.weight_bytes * 0.1 * batch_size, effective_flops, effective_bw
        )
        bottleneck = roofline["bottleneck"]

        # 6. Latency = max(compute, memory) + dispatch overhead + layer tax
        dispatch_tax = hardware.dispatch_tax
        
        # Calculate layer-wise software tax
        # Source: Reddi et al. (2025), Wall 7 (Framework Overhead)
        num_layers = getattr(model, 'layers', 1) or 1
        from .defaults import FRAMEWORK_LAYER_TAX_MS
        # Training has ~3x the launches of inference (Fwd, GradW, GradA)
        launch_multiplier = 3 if is_training else 1
        layer_tax = Q_(num_layers * FRAMEWORK_LAYER_TAX_MS * launch_multiplier, "ms")
        
        latency = max(compute_time.to("ms").magnitude, memory_time.to("ms").magnitude) * ureg.ms + dispatch_tax + layer_tax

        # 7. Throughput
        if latency.magnitude > 0:
            throughput = Q_(batch_size / latency.to("s").magnitude, "1/s")
        else:
            throughput = Q_(0, "1/s")

        # 8-9. MFU, HFU, and Energy (energy depends on MFU, so compute MFU first)
        # MFU = achieved_flops / peak_flops = (model_flops / step_time) / peak_flops
        # Source: Chowdhery et al. (2022), "PaLM: Scaling Language Modeling with Pathways"
        # IMPORTANT: Use Pint unit-aware division to handle TFLOPS/s vs flop correctly.
        latency_s = latency.to("s")
        if latency_s.magnitude > 0 and peak_flops.magnitude > 0:
            achieved_flops_rate = (batch_ops / latency_s).to(peak_flops.units)
            mfu = (achieved_flops_rate / peak_flops).to_base_units().magnitude
            mfu = max(0.0, min(mfu, 1.0))  # Clamp to [0, 1]
        else:
            mfu = 0.0
        hfu = min(mfu * HFU_MFU_RATIO, 1.0)  # Source: PaLM (Chowdhery et al. 2022)

        # Energy estimate (energy-proportional model)
        # Consistent with SustainabilityModel: 30% idle + 70% * utilization
        # Source: Barroso & Hölzle (2007), "The Case for Energy-Proportional Computing"
        if hardware.tdp is not None:
            avg_power = hardware.tdp * (0.3 + 0.7 * mfu)
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
        )

    @staticmethod
    def sweep(model: Workload, hardware_list: list, batch_sizes: list = None,
              precisions: list = None, efficiency: float = 0.5) -> list:
        """
        Sweeps over hardware, batch sizes, and precisions to produce a list of PerformanceProfiles.

        Enables rapid design-space exploration: evaluate thousands of configurations in sub-second time.

        Parameters
        ----------
        model : Workload
            The model architecture to evaluate.
        hardware_list : list[HardwareNode]
            List of hardware nodes to evaluate against.
        batch_sizes : list[int], optional
            List of batch sizes (default: [1]).
        precisions : list[str], optional
            List of precisions (default: ["fp16"]).
        efficiency : float
            Achieved compute efficiency.

        Returns
        -------
        list[dict]
            List of dicts with 'hardware', 'batch_size', 'precision', and 'profile' keys.
        """
        results = []
        batch_sizes = batch_sizes or [1]
        precisions = precisions or ["fp16"]
        for hw in hardware_list:
            for bs in batch_sizes:
                for prec in precisions:
                    try:
                        profile = Engine.solve(model, hw, batch_size=bs, precision=prec, efficiency=efficiency)
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
