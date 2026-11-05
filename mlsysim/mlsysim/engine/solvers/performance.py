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

    def solve(
        self,
        model: Workload,
        hardware: HardwareNode,
        batch_size: int = 1,
        precision: str = "fp16",
        efficiency: float = 0.5,
        raise_errors: bool = False,
        **kwargs,
    ) -> PerformanceProfile:
        """Calculate the Roofline performance profile for a single hardware node.

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

        **kwargs : dict
            Additional Engine.solve options such as training mode or sequence length.

        Returns
        -------
        PerformanceProfile
            Latency breakdown, throughput, bottleneck, MFU, and memory footprint.

        See Also
        --------
        mlsysim.engine.engine.Engine.solve : Generic node calculation.

        Notes
        -----
        Equation: T = max(F / (eta * peak), traffic / bandwidth) + overhead.
        This wrapper delegates the workload lowering, memory accounting, and
        constraint traces to ``Engine.solve``.

        Assumptions: the generic engine's activation-traffic convention and
        one forward step unless additional engine options select training.
        Efficiency scales compute capacity, not peak memory bandwidth.
        Excludes: explicit KV-aware serving and distributed collectives.

        References
        ----------
        Williams, Waterman, and Patterson (2009), "Roofline."
        https://doi.org/10.1145/1498765.1498785 (bibliography key: williams2009).
        Activation traffic and runtime taxes are separate modeling conventions.

        Examples
        --------
        >>> from mlsysim import Hardware, Models
        >>> from mlsysim.solvers import SingleNodeModel
        >>> r = SingleNodeModel().solve(Models.Language.Llama3_8B, Hardware.Cloud.H100)
        >>> bool(r.feasible and r.latency.m_as("ms") > 0)
        True
        """
        return Engine.solve(
            model,
            hardware,
            batch_size=batch_size,
            precision=precision,
            efficiency=efficiency,
            raise_errors=raise_errors,
            **kwargs,
        )


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
    produces = PerformanceProfile  # Reusing PerformanceProfile for consistency

    def solve(
        self, model: Workload, fleet: Fleet, precision: str = "fp16", efficiency: float = 0.5
    ) -> PerformanceProfile:
        """Solve for the fleet-scale performance bound (network roofline).

        Models one data-parallel training step as two independent ceilings —
        compute and gradient synchronization — and takes the slower as the
        binding latency (ideal overlap is assumed, so this is a lower-bound
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

        Notes
        -----
        Equation: T = max(6P / (eta * fleet_peak), ``2P*b`` / bisection_BW),
        where P is parameter count and b is bytes per gradient element.
        CI = 6P / ``(2P*b)``; the reported ridge uses fleet peak without eta.

        Assumptions: one normalized training sample, uniform data-parallel
        replicas, and ideal overlap between the two independent ceilings.
        Excludes: batch/sequence scaling, ring startup latency, optimizer state,
        capacity checks, and pipeline bubbles. ``feasible=True`` does not check
        whether training state fits; the stored footprint is sync traffic.

        References
        ----------
        Williams, Waterman, and Patterson (2009), "Roofline."
        https://doi.org/10.1145/1498765.1498785 (bibliography key: williams2009).
        Derivation: replace memory traffic and its rate by gradient traffic and
        fabric bisection rate; the ``2P*b`` traffic is an approximate ring volume.

        Examples
        --------
        >>> from mlsysim import Models, Systems
        >>> from mlsysim.solvers import NetworkRooflineModel
        >>> r = NetworkRooflineModel().solve(Models.Language.Llama3_8B, Systems.Clusters.Research_256)
        >>> bool(r.latency >= r.latency_compute and r.latency >= r.latency_memory)
        True
        """
        from ...core._validation import validate_range

        validate_range(efficiency, 1e-9, 1.0, "efficiency")
        if model.parameters is None:
            raise ValueError("NetworkRooflineModel requires a workload with a parameter count.")

        # 1. Supply (Fleet Capacity)
        total_flops = (
            fleet.node.accelerator.compute.precision_flops.get(precision, fleet.node.accelerator.compute.peak_flops)
            * fleet.total_accelerators
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

    def solve(
        self,
        model: Workload,
        hardware: HardwareNode,
        workload_type: str = "ffn",
        use_flash_attention: bool = False,
        precision: str = "fp16",
        efficiency: float = 0.5,
    ) -> EfficiencyResult:
        """Estimate achievable MFU and FLOPS for a given workload type.

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

        Notes
        -----
        Equation: achieved_rate = peak_rate * eta, where eta is a workload
        lookup scaled by efficiency / 0.5 and clipped by the configured cap.
        FFN uses the literature registry's upper MFU; ordinary attention uses
        its lower MFU; convolution uses their average. Flash attention uses
        its own declared coefficient and cap.

        Assumptions: these are illustrative workload-class efficiency factors,
        not a kernel-level occupancy or memory-access simulation.
        Excludes: model shape, context length, actual kernel selection, and
        memory feasibility. The returned occupancy/stall split is heuristic;
        its components need not sum to the clipped utilization loss.

        References
        ----------
        Derivation: throughput is the supplied peak multiplied by an assumed
        utilization fraction. Coefficients are in ``engine.calibration`` and
        ``Literature.Training``; this function does not estimate them from traces.

        Examples
        --------
        >>> from mlsysim import Hardware, Models
        >>> from mlsysim.solvers import EfficiencyModel
        >>> r = EfficiencyModel().solve(Models.Language.Llama3_8B, Hardware.Cloud.H100)
        >>> bool(0 <= r.mfu <= 1 and r.achievable_flops <= r.peak_flops)
        True
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

    def solve(
        self,
        model: Workload,
        hardware: HardwareNode,
        precision: str = "fp16",
        perturbation_pct: float = 10.0,
        efficiency: float = 0.5,
    ) -> SensitivityResult:
        """Solve for parameter sensitivities and identify the binding constraint.

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
            Per-parameter sensitivities, the binding constraint name, the
            baseline latency, and ``feasible``. If the baseline does not fit
            in memory, ``feasible`` is False, ``binding_constraint`` is
            ``"memory_capacity"``, and ``constraint_trace`` carries the Memory
            Wall failure: the offload path pins latency, so the sensitivities
            (typically all zero) do not rank hardware levers.

        See Also
        --------
        mlsysim.engine.engine.Engine.solve : Baseline and perturbed evaluator.

        Notes
        -----
        Equation: response_j = ``(T(x_j*(1+p/100))`` - T(x)) / T(x), where p is
        ``perturbation_pct``. The reported response is a finite fractional
        change, not a derivative divided by the perturbation size.

        Assumptions: each hardware field changes alone; all other inputs and
        the generic engine's assumptions remain fixed. Use positive p for an
        upgrade experiment. Capacity-infeasible baselines are labeled separately.
        Excludes: coupled hardware costs, changes in kernel efficiency, and
        uncertainty intervals. Near a regime boundary the finite step matters.

        References
        ----------
        Derivation: compute a finite difference in latency and normalize by the
        baseline; rank the absolute responses of compute, bandwidth, and capacity.

        Examples
        --------
        >>> from mlsysim import Hardware, Models
        >>> from mlsysim.solvers import SensitivitySolver
        >>> r = SensitivitySolver().solve(Models.Language.Llama3_8B, Hardware.Cloud.H100)
        >>> sorted(r.sensitivities)
        ['memory_bandwidth', 'memory_capacity', 'peak_flops']
        """
        baseline = Engine.solve(model, hardware, precision=precision, efficiency=efficiency)
        t_base = baseline.latency.to("ms").magnitude
        factor = 1.0 + perturbation_pct / 100.0
        sensitivities = {}

        # Registry hardware entries are frozen shared singletons, so each
        # perturbation builds a modified COPY via model_copy(update=...) rather
        # than assigning to the node (2026-06-06; the old deepcopy-then-assign
        # pattern relied on mutability that frozen=True now forbids).
        hw_flops = hardware.model_copy(
            update={
                "compute": hardware.compute.model_copy(
                    update={
                        "peak_flops": hardware.compute.peak_flops * factor,
                        "precision_flops": {k: v * factor for k, v in hardware.compute.precision_flops.items()},
                    }
                )
            }
        )
        t_flops = Engine.solve(model, hw_flops, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["peak_flops"] = (t_flops - t_base) / t_base if t_base > 0 else 0.0

        hw_bw = hardware.model_copy(
            update={"memory": hardware.memory.model_copy(update={"bandwidth": hardware.memory.bandwidth * factor})}
        )
        t_bw = Engine.solve(model, hw_bw, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["memory_bandwidth"] = (t_bw - t_base) / t_base if t_base > 0 else 0.0

        hw_mem = hardware.model_copy(
            update={"memory": hardware.memory.model_copy(update={"capacity": hardware.memory.capacity * factor})}
        )
        t_mem = Engine.solve(model, hw_mem, precision=precision, efficiency=efficiency).latency.to("ms").magnitude
        sensitivities["memory_capacity"] = (t_mem - t_base) / t_base if t_base > 0 else 0.0

        if baseline.feasible:
            # The binding constraint is the parameter whose upgrade moves latency
            # most (abs() because improvements show up as negative sensitivities).
            binding = max(sensitivities, key=lambda k: abs(sensitivities[k]))
            trace = [
                f"Sensitivity: {binding} is binding "
                f"({sensitivities[binding]:+.4f} latency change per {perturbation_pct:g}% upgrade)."
            ]
        else:
            # Infeasible baseline (fixed 2026-09-15): the model spills out of
            # HBM, Engine.solve prices every byte at the offload bandwidth, and
            # a small perturbation of FLOPs, HBM bandwidth, or capacity leaves
            # latency unchanged. max() over those all-zero sensitivities used
            # to name "peak_flops" as binding. The real constraint is capacity.
            binding = "memory_capacity"
            trace = list(baseline.constraint_trace or []) + [
                "Sensitivity: infeasible baseline. The model does not fit in memory, so "
                "memory_capacity is binding and the perturbation sensitivities are not a "
                "ranking of hardware levers until the configuration fits."
            ]

        return SensitivityResult(
            constraint_trace=trace,
            feasible=baseline.feasible,
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

    def solve(
        self, model: Workload, target_latency: Quantity, precision: str = "fp16", efficiency: float = 0.5
    ) -> SynthesisResult:
        """Synthesize minimum hardware requirements from a latency SLA.

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

        Notes
        -----
        Equation: BW_required = weight_bytes / T; peak_required = F / ``(T*eta)``.
        Required capacity is weight storage alone and the reported balance is
        peak_required / BW_required.

        Assumptions: positive target time and efficiency in (0, 1], batch one,
        one full weight read, and the workload's lowered forward operation count.
        Excludes: activations, KV cache, dispatch overhead, and communication.
        These are screening requirements, not a sufficient deployment specification.

        References
        ----------
        Williams, Waterman, and Patterson (2009), "Roofline."
        https://doi.org/10.1145/1498765.1498785 (bibliography key: williams2009).
        Derivation: solve each resource-time inequality for its required rate.

        Examples
        --------
        >>> from mlsysim import Models, Q_
        >>> from mlsysim.solvers import SynthesisSolver
        >>> r = SynthesisSolver().solve(Models.Language.Llama3_8B, Q_("50 ms"))
        >>> bool(r.required_bw.m_as("GB/s") > 0 and r.required_memory.m_as("GB") > 0)
        True
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
    "DecodeBoundModel",
    "IronLawModel",
    "AmdahlModel",
]


class DecodeBoundModel(ForwardModel):
    """
    Weight-streaming bound on one autoregressive decode step.

    At small batch every new token multiplies every weight once and must read
    every weight from memory, so a decode step needs ``2P`` FLOPs per request
    and ``P x bytes_per_param`` of weight traffic. Comparing the two times on
    one device (one accelerator-equivalent path, capacity assumed) shows which
    resource binds; the weight-only arithmetic intensity places the step on the
    roofline. KV-cache reads and launch overhead are excluded (see
    ``ServingModel`` for those).

    Literature Source:
    1. Williams, Waterman, and Patterson (2009), "Roofline."
    2. Kaplan et al. (2020), the 2P forward-FLOP rule.
    """

    requires = ("workload", "hardware")

    def solve(
        self,
        model: Workload,
        hardware: HardwareNode,
        precision: str = "fp16",
        batch_size: int = 1,
        peak_flops: Quantity = None,
        tp_size: int = 1,
    ):
        """Compute a weight-only decode Roofline bound per tensor-parallel shard.

        Parameters
        ----------
        model : Workload
            Model supplying parameter count and weight storage.
        hardware : HardwareNode
            Device supplying peak FLOP/s and memory bandwidth in bytes/s.
        precision : str, optional
            Storage precision; default FP16. Does not select compute peak here.
        batch_size : int, optional
            Positive requests decoded together; default one.
        peak_flops : Quantity, optional
            Positive FLOP/s override; otherwise uses the device's generic peak.
        tp_size : int, optional
            Positive count of ideal equal weight/work shards; default one.

        Returns
        -------
        DecodeBoundResult
            Compute and memory times, weight bytes, work in FLOPs, intensity
            in FLOP/byte, attainable FLOP/s, and the binding resource. No
            residency or deployment-feasibility check is performed.

        See Also
        --------
        mlsysim.physics.performance.calc_bottleneck : Compare resource times.
        mlsysim.physics.transformer.calc_transformer_decode_flops : Linear work.

        Notes
        -----
        Equation: F = ``2P*B/TP``, W = ``P*b/TP``; T_compute = F/peak and
        T_memory = W/BW. Attainable rate is min(peak, ``BW*F/W)``.

        Assumptions: equal sharding, one weight read shared across the batch,
        full use of supplied peak rates, and parameter-dominated forward work.
        Excludes: KV-cache reads, attention over context, communication,
        activations, runtime overhead, and memory-capacity checks.

        References
        ----------
        Williams, Waterman, and Patterson (2009), "Roofline."
        https://doi.org/10.1145/1498765.1498785 (bibliography key: williams2009).
        Kaplan et al. (2020), "Scaling Laws for Neural Language Models."
        https://arxiv.org/abs/2001.08361 (bibliography key: kaplan2020scaling).

        Examples
        --------
        >>> from mlsysim import Hardware, Models
        >>> from mlsysim.solvers import DecodeBoundModel
        >>> r = DecodeBoundModel().solve(Models.Language.Llama3_8B, Hardware.Cloud.H100)
        >>> bool(r.memory_time.m_as("ms") > 0 and r.compute_time.m_as("ms") > 0)
        True
        """
        from ..results import DecodeBoundResult
        from ...physics.performance import calc_bottleneck
        from ...physics.transformer import calc_transformer_decode_flops

        precision, _ = resolve_precision(precision)
        # tp_size (added 2026-10-20): each device holds and multiplies a 1/tp_size weight shard.
        weights = model.size_in_bytes(precision) / tp_size
        flops = calc_transformer_decode_flops(model.parameters) * batch_size / tp_size
        peak = peak_flops if peak_flops is not None else hardware.compute.peak_flops
        bw = hardware.memory.bandwidth
        b = calc_bottleneck(flops, weights, peak, bw)
        compute_time = Q_(b["compute_ms"], "millisecond")
        memory_time = Q_(b["memory_ms"], "millisecond")
        intensity = (flops / weights).to("flop/byte")
        attainable = min(peak.to("flop/second"), (bw * intensity).to("flop/second"))
        return DecodeBoundResult(
            weight_bytes=weights,
            flops_per_step=flops,
            compute_time=compute_time,
            memory_time=memory_time,
            memory_share=(memory_time / (memory_time + compute_time)).to("").magnitude,
            bound_ratio=(memory_time / compute_time).to("").magnitude,
            intensity=intensity,
            attainable_flops=attainable,
            peak_fraction=(attainable / peak).to("").magnitude,
            bottleneck=b["bottleneck"],
        )


# ── Single-machine iron law and Amdahl (added 2026-10-20) ─────────────────────

from ..results import AmdahlResult, IronLawResult  # noqa: E402
from ...physics.performance import calc_amdahls_speedup  # noqa: E402


class IronLawModel(ForwardModel):
    """
    The iron law of ML systems on one machine, in its two forms.

    Sequential: T = memory + compute + overhead, the safe estimate when the
    stages run one after another. Overlapped: T >= max(memory, compute) +
    overhead, the best case when loading and computing overlap perfectly. An
    upgrade divides the memory term by ``memory_speedup`` (more bandwidth or
    fewer bytes) and the compute term by ``compute_speedup``; the overhead
    stays, which is why the sum falls by less than the upgrade.
    """

    requires = ("memory_time", "compute_time", "overhead")
    produces = IronLawResult

    def solve(
        self,
        memory_time: Quantity,
        compute_time: Quantity,
        overhead: Quantity,
        memory_speedup: float = 1.0,
        compute_speedup: float = 1.0,
    ) -> IronLawResult:
        """Compare sequential and ideally overlapped memory and compute times.

        Parameters
        ----------
        memory_time : Quantity
            Nonnegative memory-transfer duration, convertible to overhead's unit.
        compute_time : Quantity
            Nonnegative compute duration, convertible to overhead's unit.
        overhead : Quantity
            Nonnegative fixed time unaffected by either upgrade; total must be positive.
        memory_speedup : float, optional
            Positive divisor of memory time; default one leaves it unchanged.
        compute_speedup : float, optional
            Positive divisor of compute time; default one leaves it unchanged.

        Returns
        -------
        IronLawResult
            Original and upgraded times in overhead's units and the dimensionless
            sequential-time reduction. These are bounds, with no capacity check.

        Notes
        -----
        Equation: T_serial = M+C+O; T_overlap = max(M,C)+O. Upgrade terms are
        M/s_memory and C/s_compute while O remains fixed.

        Assumptions: distinct memory, compute, and overhead components; either
        serial execution or complete overlap between memory and compute.
        Excludes: partial overlap schedules, resource contention, and workload changes.

        References
        ----------
        Derivation: add disjoint durations for serial execution and take the
        longer duration for simultaneous memory/compute execution.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import IronLawModel
        >>> r = IronLawModel().solve(Q_("2 ms"), Q_("3 ms"), Q_("1 ms"), memory_speedup=2)
        >>> bool(r.upgraded_sequential_time < r.sequential_time)
        True
        """
        unit = overhead.units
        m, c, o = memory_time.to(unit), compute_time.to(unit), overhead.to(unit)
        m2, c2 = m / memory_speedup, c / compute_speedup
        seq, seq2 = m + c + o, m2 + c2 + o
        return IronLawResult(
            memory_time=m,
            compute_time=c,
            overhead=o,
            sequential_time=seq,
            overlapped_time=max(m, c) + o,
            upgraded_memory_time=m2,
            upgraded_compute_time=c2,
            upgraded_sequential_time=seq2,
            upgraded_overlapped_time=max(m2, c2) + o,
            naive_upgraded_time=seq / (memory_speedup * compute_speedup),
            sequential_reduction=(1 - seq2 / seq).to("").magnitude,
        )


class AmdahlModel(ForwardModel):
    """
    Amdahl's law for a sequential request path: one stage made ``speedup``
    times faster, the others untouched. The untouched time caps the overall
    speedup at total / untouched, however fast the stage becomes.
    """

    requires = ("stages", "stage", "speedup")
    produces = AmdahlResult

    def solve(self, stages: dict, stage: str, speedup: float) -> AmdahlResult:
        """Evaluate the end-to-end effect of accelerating one sequential stage.

        Parameters
        ----------
        stages : dict
            Stage names mapped to nonnegative time quantities in compatible units.
            Total and untouched time must both be positive for the reported bound.
        stage : str
            Existing key identifying the stage to accelerate.
        speedup : float
            Positive stage speedup; positive infinity represents its zero-time limit.

        Returns
        -------
        AmdahlResult
            Original and upgraded times in the selected stage's units, fractional
            reductions, overall speedup, and the untouched-time speedup ceiling.

        See Also
        --------
        mlsysim.physics.performance.calc_amdahls_speedup : Fractional form.

        Notes
        -----
        Equation: T_new = T_other + T_stage/s; speedup_total = T_total/T_new.
        With p = T_stage/T_total, this is 1/(1-p+p/s).

        Assumptions: sequential stages and only the named stage changes.
        Excludes: stage overlap, workload growth, and acceleration overhead.

        References
        ----------
        Derivation: partition total duration into the accelerated stage and
        untouched stages, divide only the former by its speedup, and sum.

        Examples
        --------
        >>> from mlsysim import Q_
        >>> from mlsysim.solvers import AmdahlModel
        >>> r = AmdahlModel().solve({"compute": Q_("3 ms"), "io": Q_("1 ms")}, "compute", 2)
        >>> bool(r.upgraded_total_time < r.total_time and r.overall_speedup < r.speedup_bound)
        True
        """
        unit = stages[stage].units
        total = sum((t.to(unit) for t in stages.values()), Q_(0, unit))
        part = stages[stage].to(unit)
        rest = total - part
        p = (part / total).to("").magnitude
        overall = calc_amdahls_speedup(p, speedup)
        return AmdahlResult(
            total_time=total,
            stage_time=part,
            upgraded_stage_time=part / speedup,
            upgraded_total_time=rest + part / speedup,
            untouched_time=rest,
            stage_fraction=p,
            overall_speedup=overall,
            end_to_end_reduction=1 - 1 / overall,
            stage_reduction=1 - 1 / speedup,
            speedup_bound=(total / rest).to("").magnitude,
        )
