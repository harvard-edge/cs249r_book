from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Any, Annotated
from .constants import ureg, Q_, BYTES_FP32, BYTES_FP16, BYTES_INT8, BYTES_INT4
from .formulas import calc_bottleneck
from .exceptions import OOMError
from ..models.types import Workload, TransformerWorkload, CNNWorkload
from ..hardware.types import HardwareNode, Quantity

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
    Unified solver for ML Systems trade-offs.
    
    This engine implements the 'Roofline Performance Model' (Williams et al. 2009)
    to identify whether a workload is compute-bound or memory-bound.
    """
    
    @staticmethod
    def solve(model: Workload, hardware: HardwareNode, batch_size=1, precision="fp16", efficiency=0.5, raise_errors=False) -> PerformanceProfile:
        # 1. Map Precision
        if precision == "fp32":
            bpp = BYTES_FP32
            peak_flops = hardware.compute.precision_flops.get("fp32", hardware.compute.peak_flops)
        elif precision == "int8":
            bpp = BYTES_INT8
            peak_flops = hardware.compute.precision_flops.get("int8", hardware.compute.peak_flops)
        elif precision == "int4":
            bpp = BYTES_INT4
            peak_flops = hardware.compute.precision_flops.get("int4", hardware.compute.peak_flops)
        else: # Default fp16
            bpp = BYTES_FP16
            peak_flops = hardware.compute.peak_flops

        # 2. Workload
        if hasattr(model, "inference_flops") and model.inference_flops:
            ops_per_inference = model.inference_flops
        else:
            # Fallback for transformers: 2 * Params
            if hasattr(model, "parameters") and model.parameters:
                ops_per_inference = 2 * model.parameters.to(ureg.count).magnitude * ureg.flop
            else:
                ops_per_inference = 0 * ureg.flop

        total_ops = ops_per_inference * batch_size
        memory_bytes = model.size_in_bytes(bpp)
        
        # 3. Iron Law (Roofline)
        results = calc_bottleneck(
            ops=total_ops, 
            model_bytes=memory_bytes, 
            device_flops=peak_flops * efficiency, 
            device_bw=hardware.memory.bandwidth
        )
        
        t_comp = results["compute_ms"] * ureg.ms
        t_mem = results["memory_ms"] * ureg.ms
        t_overhead = hardware.dispatch_tax
        
        # Total Latency (Pipelined Assumption: overlapping data and compute)
        latency = max(t_comp, t_mem) + t_overhead
        
        # 4. Feasibility Check (Simple memory check)
        feasible = memory_bytes <= hardware.memory.capacity
        
        if raise_errors and not feasible:
            raise OOMError(
                f"OOM: {model.name} requires {memory_bytes.to('GB')} but {hardware.name} only has {hardware.memory.capacity.to('GB')}.",
                required_bytes=memory_bytes,
                available_bytes=hardware.memory.capacity
            )
            
        # 5. Utilization Metrics
        # MFU: Model FLOPs Utilization (Actual / Peak)
        # HFU: Hardware FLOPs Utilization
        throughput_samples_per_sec = (batch_size / latency).to(1/ureg.second).magnitude
        actual_flops_delivered = ops_per_inference.magnitude * throughput_samples_per_sec
        
        mfu = actual_flops_delivered / peak_flops.magnitude if peak_flops.magnitude > 0 else 0.0
        hfu = mfu / efficiency if efficiency > 0 else 0.0 # HFU is normalized by achieved compute efficiency
            
        return PerformanceProfile(
            latency=latency,
            latency_compute=t_comp,
            latency_memory=t_mem,
            latency_overhead=t_overhead,
            throughput=(batch_size / latency).to(1/ureg.second),
            bottleneck=results["bottleneck"],
            arithmetic_intensity=results["intensity"] * (ureg.flop / ureg.byte),
            energy=(hardware.tdp * latency).to(ureg.joule) if hardware.tdp else 0 * ureg.joule,
            memory_footprint=memory_bytes,
            peak_flops_actual=peak_flops * efficiency,
            peak_bw_actual=hardware.memory.bandwidth,
            mfu=mfu,
            hfu=hfu,
            feasible=feasible
        )
