# engine.py
# The central computational engine for ML Systems analysis.
# Ties Models, Systems, and Formulas into a single "Solver".

from dataclasses import dataclass
from .models import ModelSpec
from .systems import SystemArchetype
from .constants import ureg, Q_, BYTES_FP32, BYTES_FP16, BYTES_INT8
from .formulas import calc_bottleneck

@dataclass(frozen=True)
class PerformanceProfile:
    """The result of a system simulation."""
    latency: Q_
    latency_compute: Q_
    latency_memory: Q_
    latency_overhead: Q_
    throughput: Q_
    bottleneck: str
    arithmetic_intensity: Q_
    energy: Q_
    memory_footprint: Q_
    peak_flops_actual: Q_
    peak_bw_actual: Q_
    feasible: bool

class Engine:
    """
    Unified solver for ML Systems trade-offs.
    """
    
    @staticmethod
    def solve(model: ModelSpec, system: SystemArchetype, batch_size=1, precision="fp16", efficiency=0.5) -> PerformanceProfile:
        hw = system.hardware
        
        # 1. Map Precision
        if precision == "fp32":
            bpp = BYTES_FP32
            peak_flops = hw.peak_flops_fp32 or hw.peak_flops
        elif precision == "int8":
            bpp = BYTES_INT8
            peak_flops = hw.int8_flops or hw.peak_flops
        else: # Default fp16
            bpp = BYTES_FP16
            peak_flops = hw.peak_flops

        # 2. Workload
        ops_per_inference = model.inference_flops or (2 * model.parameters.to(ureg.count).magnitude * ureg.flop)
        total_ops = ops_per_inference * batch_size
        memory_bytes = model.size_in_bytes(bpp)
        
        # 3. Physics (Iron Law)
        # Note: We use the hardware's memory bandwidth directly.
        results = calc_bottleneck(
            ops=total_ops, 
            model_bytes=memory_bytes, 
            device_flops=peak_flops * efficiency, 
            device_bw=hw.memory_bw
        )
        
        t_comp = results["compute_ms"] * ureg.ms
        t_mem = results["memory_ms"] * ureg.ms
        t_overhead = hw.dispatch_tax
        
        # Total Latency (Pipelined Assumption: overlapping data and compute)
        latency = max(t_comp, t_mem) + t_overhead
        
        # 4. Feasibility Check
        feasible = memory_bytes <= system.ram
        
        return PerformanceProfile(
            latency=latency,
            latency_compute=t_comp,
            latency_memory=t_mem,
            latency_overhead=t_overhead,
            throughput=(batch_size / latency).to(1/ureg.second),
            bottleneck=results["bottleneck"],
            arithmetic_intensity=results["intensity"] * (ureg.flop / ureg.byte),
            energy=(hw.tdp * latency).to(ureg.joule) if hw.tdp else 0 * ureg.joule,
            memory_footprint=memory_bytes,
            peak_flops_actual=peak_flops * efficiency,
            peak_bw_actual=hw.memory_bw,
            feasible=feasible
        )
