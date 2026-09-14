"""Single-node, network-roofline, efficiency, and inverse-design solvers.

Domain implementations behind ``mlsysim.solvers`` (the public import
path, derived from ``engine.solvers.__init__``); kept per-domain so the logic stays reviewable.
"""

from __future__ import annotations


from ..engine import Engine, PerformanceProfile
from ..results import (
    EfficiencyResult,
    SynthesisResult,
    SensitivityResult,
)
from ...core.units import ureg, Q_, resolve_precision
from ...literature.registry import Literature
from .. import calibration as cal
from ...core.types import Quantity
from ...models.types import Workload
from ...hardware.types import HardwareNode
from ...systems.types import Fleet
from .base import BaseSolver, ForwardModel

class SingleNodeModel(ForwardModel):
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
        Calculate the Roofline performance profile for a single hardware node.

        Thin wrapper that delegates to ``Engine.solve`` (the core Roofline
        implementation in ``engine/engine.py``), exposing it through the
        resolver interface so it can participate in pipelines.

        Parameters
        ----------
        model : Workload
            The workload to profile.
        hardware : HardwareNode
            The accelerator node (compute, memory, interconnect specs).
        batch_size : int
            Samples per forward pass (>= 1).
        precision : str
            Numerical precision ('fp32', 'fp16', 'int8', ...); sets bytes
            per parameter and selects the hardware's precision FLOPS entry.
        efficiency : float
            Software efficiency factor in (0, 1] applied to peak FLOPS.
        raise_errors : bool
            If True, raise ``OOMError`` on memory infeasibility instead of
            returning a profile with ``feasible=False``.

        Returns
        -------
        PerformanceProfile
            Latency breakdown, throughput, bottleneck, MFU, and memory footprint.
        """
        return Engine.solve(model, hardware, batch_size=batch_size, precision=precision, efficiency=efficiency, raise_errors=raise_errors, **kwargs)

class NetworkRooflineModel(ForwardModel):
    """
    Analyzes the Distributed Performance Bounds (The Network Wall).

    This model elevates the single-node Roofline analysis to the fleet scale.
    It calculates the Communication Intensity (CI) of a workload and identifies
    if the fleet is Compute-Bound (Wall 1) or Network-Bound (Wall 2).

    Literature Source:
    1. Williams et al. (2009), "Roofline Model." (Theoretical basis)
    2. Ghose et al. (2019), "A Survey of Communication-Efficient Distributed
       Training."
    """
    requires = ("workload", "fleet")
    produces = PerformanceProfile # Reusing PerformanceProfile for consistency

    def solve(self, model: Workload, fleet: Fleet, precision: str = "fp16", efficiency: float = 0.5) -> PerformanceProfile:
        """
        Solve for the fleet-scale performance bound (network roofline).

        Models one data-parallel training step as two independent ceilings —
        compute and gradient synchronization — and takes the slower as the
        binding latency (no overlap is assumed, so this is a lower-bound
        model). The conventions used:

        - communication intensity ``CI = training_ops / sync_bytes``
          [FLOP/byte], with ``training_ops = 6 * P`` FLOPs per sample
          (Kaplan/Chinchilla forward+backward convention) and
          ``sync_bytes = 2 * P * bytes_per_param`` (ring all-reduce moves
          roughly 2x the gradient volume);
        - network ridge ``= fleet_peak_FLOPS / bisection_BW`` [FLOP/byte],
          the CI above which the fleet is compute-bound.

        Parameters
        ----------
        model : Workload
            Workload with a parameter count (required).
        fleet : Fleet
            The cluster; supplies total accelerators, per-accelerator FLOPS,
            and fabric bandwidth / oversubscription ratio.
        precision : str
            Numerical precision; sets bytes per parameter and the FLOPS entry.
        efficiency : float
            Software efficiency factor in (0, 1] applied to fleet peak FLOPS.

        Returns
        -------
        PerformanceProfile
            ``latency_compute`` holds the compute ceiling, ``latency_memory``
            the network ceiling (both in ms); ``bottleneck`` is 'Network' or
            'Compute'; ``arithmetic_intensity`` carries CI.
        """
        from ...core._validation import validate_range
        validate_range(efficiency, 1e-9, 1.0, "efficiency")
        if model.parameters is None:
            raise ValueError("NetworkRooflineModel requires a workload with a parameter count.")

        # 1. Supply (Fleet Capacity)
        total_flops = (
            fleet.node.accelerator.compute.precision_flops.get(
                precision, fleet.node.accelerator.compute.peak_flops
            ) * fleet.total_accelerators
        ).to("TFLOPs/s")
        # Bisection Bandwidth (total data movement capacity per step)
        bisection_bw = (fleet.fabric.bandwidth / fleet.fabric.oversubscription_ratio).to("GB/s")

        # 2. Demand (Workload Characteristics)
        # For training, total ops = 6 * parameters * dataset_size_per_step
        # To normalize per-step, we use 1 sample.
        # But CI is better defined as Total Ops / Total Sync Bytes.
        # For Data Parallel sync: Bytes = 2 * Parameters * Precision_Bytes
        precision, prec_bytes = resolve_precision(precision)
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

        # 4. Results. Treat this as a one-step training lower-bound model:
        # compute and network are independent ceilings; the slower one binds.
        effective_flops = total_flops * efficiency
        compute_time = (training_ops / effective_flops).to("ms")
        network_time = (sync_bytes / bisection_bw).to("ms")
        is_network_bound = network_time > compute_time
        latency = max(compute_time.to("ms").magnitude, network_time.to("ms").magnitude) * ureg.ms

        # MFU follows from the binding ceiling: useful FLOPs delivered per step
        # over fleet peak. This model counts only model FLOPs (no recomputation
        # is modeled), so HFU equals MFU by definition (PaLM App. B).
        achieved_rate = (training_ops / latency.to("s")).to(total_flops.units)
        mfu = min((achieved_rate / total_flops).to_base_units().magnitude, 1.0)
        hfu = mfu
        throughput = (1 / latency.to("s")).to("1/s")

        return PerformanceProfile(
            latency=latency,
            latency_compute=compute_time,
            latency_memory=network_time,
            latency_overhead=Q_("0 ms"),
            throughput=throughput,
            bottleneck="Network" if is_network_bound else "Compute",
            arithmetic_intensity=ci,
            energy=Q_("0 J"),
            memory_footprint=sync_bytes,
            peak_flops_actual=effective_flops,
            peak_bw_actual=bisection_bw,
            mfu=mfu,
            hfu=hfu,
            feasible=True,
            constraint_trace=[
                f"Network Roofline: CI {ci:~P} vs ridge {network_ridge:~P}; "
                f"binding ceiling is {'network' if is_network_bound else 'compute'}."
            ],
        )

class EfficiencyModel(ForwardModel):
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
        EfficiencyResult
            MFU estimate, achievable FLOPS, and overhead breakdown.
        """
        peak_flops = hardware.compute.precision_flops.get(precision, hardware.compute.peak_flops)

        # Base MFU range from Literature.Training (training regime)
        mfu_low = Literature.Training.MfuLow
        mfu_high = Literature.Training.MfuHigh

        # Workload-type MFU adjustment (heuristic calibrations — see core/calibration.py)
        # All values scale linearly with the efficiency parameter relative to 0.5.
        scale = efficiency / 0.5
        if workload_type == "attention":
            if use_flash_attention:
                eta = min(cal.MFU_FLASH_ATTENTION * scale, cal.MFU_FLASH_ATTENTION_CAP)
            else:
                # Standard attention is memory-bound due to O(S²) reads
                eta = mfu_low * scale
        elif workload_type == "ffn":
            # FFN layers are compute-dense GEMM — best MFU
            eta = min(mfu_high * scale, cal.MFU_FFN_CAP)
        elif workload_type == "conv":
            # Convolutions via im2col + GEMM — moderate MFU
            eta = min((mfu_low + mfu_high) / 2.0 * scale, cal.MFU_CONV_CAP)
        else:
            eta = mfu_low * scale

        eta = max(0.0, min(eta, 1.0))  # Clamp to [0, 1]

        achievable_flops = peak_flops * eta

        # Overhead breakdown: decompose (1 - eta) into meaningful components.
        # Total overhead = 1 - eta, split into occupancy loss and memory stall.
        total_overhead = 1.0 - eta
        occupancy_loss = 1.0 - min(efficiency * cal.SM_OCCUPANCY_HEADROOM, 1.0)  # SM occupancy overhead
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
        Solve for parameter sensitivities and identify the binding constraint.

        Numerically perturbs each hardware parameter (peak FLOPS, memory
        bandwidth, memory capacity) by ``perturbation_pct`` and re-solves the
        single-node Roofline. Each sensitivity is the normalized latency
        response ``(T_perturbed - T_base) / T_base`` (dimensionless; negative
        means the upgrade helps). The binding constraint is the parameter with
        the largest absolute sensitivity — the resource whose improvement
        moves latency the most.

        Parameters
        ----------
        model : Workload
            The workload to analyze.
        hardware : HardwareNode
            The baseline hardware configuration.
        precision : str
            Numerical precision used for both baseline and perturbed solves.
        perturbation_pct : float
            Perturbation size in percent (default 10.0, i.e. a 1.1x upgrade).
        efficiency : float
            Software efficiency factor in (0, 1].

        Returns
        -------
        SensitivityResult
            Per-parameter sensitivities, the binding constraint name, and the
            baseline latency.
        """
        baseline = Engine.solve(model, hardware, precision=precision, efficiency=efficiency)
        t_base = baseline.latency.to("ms").magnitude
        factor = 1.0 + perturbation_pct / 100.0
        sensitivities = {}

        # Registry hardware entries are frozen shared singletons, so each
        # perturbation builds a modified COPY via model_copy(update=...) rather
        # than assigning to the node (2026-06-06; the old deepcopy-then-assign
        # pattern relied on mutability that frozen=True now forbids).
        hw_flops = hardware.model_copy(update={
            "compute": hardware.compute.model_copy(update={
                "peak_flops": hardware.compute.peak_flops * factor,
                "precision_flops": {k: v * factor for k, v in hardware.compute.precision_flops.items()},
            })
        })
        t_flops = Engine.solve(model, hw_flops, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["peak_flops"] = (t_flops - t_base) / t_base if t_base > 0 else 0.0

        hw_bw = hardware.model_copy(update={
            "memory": hardware.memory.model_copy(update={"bandwidth": hardware.memory.bandwidth * factor})
        })
        t_bw = Engine.solve(model, hw_bw, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["memory_bandwidth"] = (t_bw - t_base) / t_base if t_base > 0 else 0.0

        hw_mem = hardware.model_copy(update={
            "memory": hardware.memory.model_copy(update={"capacity": hardware.memory.capacity * factor})
        })
        t_mem = Engine.solve(model, hw_mem, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["memory_capacity"] = (t_mem - t_base) / t_base if t_base > 0 else 0.0

        # The binding constraint is the parameter whose upgrade moves latency
        # most (abs() because improvements show up as negative sensitivities).
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
        Synthesize minimum hardware requirements from a latency SLA.

        Inverts the Roofline model: instead of asking "how fast does this
        hardware run the model?", asks "what hardware would run the model in
        ``target_latency``?". Assumes a single full pass that streams every
        weight once (batch size 1, no weight reuse), so:

        - ``required_bw = weight_bytes / T_target`` [GB/s] — the bandwidth
          needed if the workload is memory-bound;
        - ``required_flops = total_ops / (T_target * efficiency)`` [TFLOP/s]
          — the compute rate needed if it is compute-bound;
        - ``compute_memory_ratio = required_flops / required_bw`` [FLOP/byte]
          — the hardware balance point a matching part should sit near.

        Parameters
        ----------
        model : Workload
            The workload to serve; sets weight bytes and operation count.
        target_latency : Quantity
            The SLA latency target (any time unit; converted to seconds).
        precision : str
            Numerical precision; sets bytes per parameter.
        efficiency : float
            Expected software efficiency in (0, 1]; lower values inflate the
            required FLOPS to compensate.

        Returns
        -------
        SynthesisResult
            Required bandwidth, FLOPS, memory capacity, and the
            compute:memory balance ratio.
        """
        precision, bpp = resolve_precision(precision)
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

__all__ = [
    "SingleNodeModel",
    "NetworkRooflineModel",
    "EfficiencyModel",
    "SensitivitySolver",
    "SynthesisSolver",
]
