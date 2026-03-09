from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Any, Annotated
from .constants import ureg, Q_, BYTES_FP32, BYTES_FP16, BYTES_INT8, BYTES_INT4, PRECISION_MAP
from .defaults import HFU_MFU_RATIO
from .formulas import calc_bottleneck
from .exceptions import OOMError
from ..models.types import Workload, TransformerWorkload, CNNWorkload
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

    def plot(self, mode="latency"):
        """Generates a visualization of this profile.
        
        Args:
            mode (str): 'latency' for breakdown, 'roofline' for roofline plot.
        """
        from ..viz import plot_latency_breakdown, plot_roofline
        if mode == "latency":
            return plot_latency_breakdown(self)
        elif mode == "roofline":
            return plot_roofline(self)
        else:
            raise ValueError(f"Unknown plot mode: {mode}")


class Engine:
    """
    The core analytical engine for single-node Roofline performance modeling.

    Maps a (Workload, HardwareNode) pair to a PerformanceProfile by applying
    the Roofline model: the workload is "lowered" into a hardware-agnostic
    ComputationGraph, then bounded by the hardware's compute ceiling and
    memory bandwidth ceiling.

    Source: Williams et al. (2009), "Roofline: An Insightful Visual
    Performance Model for Floating-Point Programs and Multicore Architectures."
    """

    @staticmethod
    def solve(model: Workload, hardware: HardwareNode, batch_size: int = 1,
              precision: str = "fp16", efficiency: float = 0.5,
              raise_errors: bool = False) -> PerformanceProfile:
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

        Returns
        -------
        PerformanceProfile
            Complete performance characterization of the workload on the hardware.
        """
        # 1. Map precision to bytes per parameter
        bpp = PRECISION_MAP.get(precision, BYTES_FP16)

        # 2. Resolve peak FLOPS for the requested precision
        peak_flops = hardware.compute.precision_flops.get(
            precision, hardware.compute.peak_flops
        )

        # 3. Lower the workload to a ComputationGraph
        graph = model.lower(bpp)

        # 4. Feasibility check — does the model fit in memory?
        feasible = graph.weight_bytes <= hardware.memory.capacity
        if not feasible and raise_errors:
            raise OOMError(
                f"Model '{model.name}' requires {graph.weight_bytes.to('GB'):~P} "
                f"but hardware '{hardware.name}' has only "
                f"{hardware.memory.capacity.to('GB'):~P}.",
                required_bytes=graph.weight_bytes,
                available_bytes=hardware.memory.capacity,
            )

        # 5. Roofline bottleneck analysis
        # Scale compute by batch_size: processing B samples requires B × ops_per_sample.
        # Weights are loaded once regardless of batch size (weight-streaming model).
        effective_flops = peak_flops * efficiency
        effective_bw = hardware.memory.bandwidth
        batch_ops = graph.total_ops * batch_size

        # Guard against zero ops or zero bandwidth
        if batch_ops.magnitude == 0:
            compute_time = Q_("0 ms")
        else:
            compute_time = (batch_ops / effective_flops).to("ms")

        if effective_bw.magnitude == 0:
            memory_time = Q_("0 ms")
        else:
            memory_time = (graph.weight_bytes / effective_bw).to("ms")

        roofline = calc_bottleneck(
            batch_ops, graph.weight_bytes, effective_flops, effective_bw
        )
        bottleneck = roofline["bottleneck"]

        # 6. Latency = max(compute, memory) + dispatch overhead
        dispatch_tax = hardware.dispatch_tax
        latency = max(compute_time.to("ms").magnitude, memory_time.to("ms").magnitude) * ureg.ms + dispatch_tax

        # 7. Throughput
        if latency.magnitude > 0:
            throughput = Q_(batch_size / latency.to("s").magnitude, "1/s")
        else:
            throughput = Q_(0, "1/s")

        # 8. Energy estimate (TDP * wall-clock time)
        if hardware.tdp is not None:
            energy = (hardware.tdp * latency.to("s")).to("J")
        else:
            energy = Q_("0 J")

        # 9. MFU and HFU
        latency_mag = latency.to("ms").magnitude
        if latency_mag > 0:
            mfu = compute_time.to("ms").magnitude / latency_mag
        else:
            mfu = 0.0
        hfu = min(mfu * HFU_MFU_RATIO, 1.0)  # Source: PaLM (Chowdhery et al. 2022)

        return PerformanceProfile(
            latency=latency,
            latency_compute=compute_time,
            latency_memory=memory_time,
            latency_overhead=dispatch_tax,
            throughput=throughput,
            bottleneck=bottleneck,
            arithmetic_intensity=graph.arithmetic_intensity,
            energy=energy,
            memory_footprint=graph.weight_bytes,
            peak_flops_actual=effective_flops,
            peak_bw_actual=effective_bw,
            mfu=mfu,
            hfu=hfu,
            feasible=feasible,
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
                    except Exception:
                        # Skip infeasible configurations silently
                        pass
        return results
