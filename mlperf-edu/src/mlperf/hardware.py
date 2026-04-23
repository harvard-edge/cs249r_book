import time
import torch
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

class HardwareEfficiencyWarning(UserWarning):
    """Raised when measured hardware performance deviates significantly from theoretical peaks."""
    pass

@dataclass
class HardwareSpec:
    name: str
    peak_tflops: Dict[str, float]  # Precision -> TFLOPS
    peak_bandwidth_gb_s: float
    
# Theoretical database of common MLPerf hardware
THEORETICAL_DATABASE = {
    "NVIDIA A100-SXM4-80GB": HardwareSpec(
        name="NVIDIA A100-SXM4-80GB",
        peak_tflops={"fp64": 9.7, "fp32": 19.5, "tf32": 156.0, "fp16": 312.0, "bf16": 312.0, "int8": 624.0},
        peak_bandwidth_gb_s=2039.0
    ),
    "NVIDIA A100-SXM4-40GB": HardwareSpec(
        name="NVIDIA A100-SXM4-40GB",
        peak_tflops={"fp64": 9.7, "fp32": 19.5, "tf32": 156.0, "fp16": 312.0, "bf16": 312.0, "int8": 624.0},
        peak_bandwidth_gb_s=1555.0
    ),
    "NVIDIA H100-SXM5": HardwareSpec(
        name="NVIDIA H100-SXM5",
        peak_tflops={"fp64": 34.0, "fp32": 67.0, "tf32": 494.0, "fp16": 989.0, "bf16": 989.0, "int8": 1979.0},
        peak_bandwidth_gb_s=3350.0
    ),
    "NVIDIA GeForce RTX 4090": HardwareSpec(
        name="NVIDIA GeForce RTX 4090",
        peak_tflops={"fp32": 82.6, "fp16": 330.0, "bf16": 330.0, "int8": 661.0},
        peak_bandwidth_gb_s=1008.0
    ),
    "NVIDIA GeForce RTX 3090": HardwareSpec(
        name="NVIDIA GeForce RTX 3090",
        peak_tflops={"fp32": 35.6, "fp16": 142.0, "bf16": 142.0},
        peak_bandwidth_gb_s=936.0
    ),
}

class RooflineHardwareNormalizer:
    """
    Profiles hardware to calculate the machine-specific Roofline curve.
    Adheres to MLPerf EDU's strict architectural and benchmarking standards.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "Generic CPU"
        self.theoretical_spec = self._lookup_spec()
        
    def _lookup_spec(self) -> Optional[HardwareSpec]:
        for key, spec in THEORETICAL_DATABASE.items():
            if key in self.device_name:
                return spec
        return None

    def verify_steady_state(self, duration_secs: float = 5.0):
        """
        Executes a 'Warm-up and Stabilize' phase to reach thermal steady-state.
        Prevents recording 'Turbo Boost' transients that are unsustainable.
        """
        print(f"[Roofline] 🔥 Warming up {self.device_name} for {duration_secs}s...")
        end_time = time.time() + duration_secs
        size = 2048
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)
        
        latencies = []
        while time.time() < end_time:
            t0 = time.time()
            torch.mm(a, b)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.time() - t0)
            
        # Check stability: variance of the last 10% of measurements
        tail = latencies[-(len(latencies)//10):]
        if len(tail) > 5:
            cv = np.std(tail) / np.mean(tail)
            if cv > 0.05:
                warnings.warn(f"Thermal jitter detected (CV={cv:.4f}). Results may be unstable.", HardwareEfficiencyWarning)

    def profile_compute_peak(self, precision: str = "fp32") -> float:
        """
        Measures Peak TFLOPs using a Compute-Only kernel.
        Optimizes for register reuse to ensure the 'Compute Roof' is a true hardware limit.
        """
        dtype = self._parse_precision(precision)
        # Size chosen to maximize Tensor Core utilization and register tiling
        # 8192^2 @ 8192^2 is large enough to hide launch overhead but fits in VRAM
        size = 8192
        a = torch.randn(size, size, device=self.device, dtype=dtype)
        b = torch.randn(size, size, device=self.device, dtype=dtype)
        
        # Pure Compute Kernel: Matrix Multiply with minimal memory movement relative to FMA
        # Intensity = (2 * N^3) / (3 * N^2 * bytes) = (2/3) * (N / bytes)
        # For N=8192 and FP32 (4 bytes), Intensity = (2/3) * (8192 / 4) = 1365 FLOPs/byte
        # This is well above the ridge point of any modern hardware.
        
        iterations = 20
        # Warmup
        for _ in range(5): torch.mm(a, b)
        if self.device.type == "cuda": torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(iterations):
            torch.mm(a, b)
        if self.device.type == "cuda": torch.cuda.synchronize()
        t1 = time.time()
        
        tflops = (iterations * 2 * size**3) / (t1 - t0) / 1e12
        
        # Guard against divergence
        if self.theoretical_spec and precision in self.theoretical_spec.peak_tflops:
            theo = self.theoretical_spec.peak_tflops[precision]
            efficiency = tflops / theo
            if efficiency > 1.02:
                raise HardwareEfficiencyWarning(f"Measured {precision} TFLOPs ({tflops:.2f}) exceeds theoretical peak ({theo:.2f}) by {efficiency:.1%}. Measurement error likely.")
            if efficiency < 0.80:
                warnings.warn(f"Low compute efficiency detected: {efficiency:.1%}. Check for background processes or thermal throttling.", HardwareEfficiencyWarning)
                
        return tflops

    def profile_memory_bandwidth(self) -> float:
        """
        Measures Peak Memory Bandwidth (GB/s) using a Memory-Only kernel.
        Minimizes compute to ensure the 'Memory Roof' is a true hardware limit.
        """
        # 512MB tensors to bypass caches (L2/L3)
        size = 128 * 1024 * 1024 # 128M elements * 4 bytes = 512MB
        a = torch.randn(size, device=self.device)
        b = torch.empty_like(a)
        
        # Memory-Only Kernel: Large vector copy/add with zero data reuse
        # Intensity = (N FLOPs) / (3 * N * 4 bytes) = 1/12 FLOPs/byte (Very low)
        
        iterations = 50
        # Warmup
        for _ in range(10): b.copy_(a)
        if self.device.type == "cuda": torch.cuda.synchronize()
        
        t0 = time.time()
        for _ in range(iterations):
            b.copy_(a)
        if self.device.type == "cuda": torch.cuda.synchronize()
        t1 = time.time()
        
        # Read + Write = 2 * size * 4 bytes
        gb_s = (iterations * 2 * size * 4) / (t1 - t0) / 1e9
        
        # Guard against divergence
        if self.theoretical_spec:
            theo = self.theoretical_spec.peak_bandwidth_gb_s
            efficiency = gb_s / theo
            if efficiency > 1.02:
                raise HardwareEfficiencyWarning(f"Measured Bandwidth ({gb_s:.2f} GB/s) exceeds theoretical peak ({theo:.2f}) by {efficiency:.1%}.")
            if efficiency < 0.80:
                warnings.warn(f"Low memory efficiency detected: {efficiency:.1%}. Verify memory frequency and NUMA affinity.", HardwareEfficiencyWarning)
                
        return gb_s

    def calculate_arithmetic_intensity(self, flops: int, bytes_accessed: int) -> float:
        """
        Calculates the Arithmetic Intensity (AI) of a workload.
        MLPerf EDU Standard: Uses 'Effective Bandwidth' accounting for actual data movement.
        """
        if bytes_accessed == 0:
            return float('inf')
        return flops / bytes_accessed

    def get_roofline_status(self, workload_ai: float, precision: str = "fp32") -> Dict[str, Any]:
        """
        Determines if a workload is Compute-Bound or Memory-Bound on this machine.
        """
        peak_tflops = self.profile_compute_peak(precision)
        peak_bw = self.profile_memory_bandwidth()
        
        ridge_point = (peak_tflops * 1e12) / (peak_bw * 1e9) # FLOPs/Byte
        
        is_compute_bound = workload_ai > ridge_point
        achievable_tflops = peak_tflops if is_compute_bound else (workload_ai * peak_bw * 1e9) / 1e12
        
        return {
            "device": self.device_name,
            "precision": precision,
            "peak_tflops": peak_tflops,
            "peak_bw_gb_s": peak_bw,
            "ridge_point_ai": ridge_point,
            "workload_ai": workload_ai,
            "bound": "compute" if is_compute_bound else "memory",
            "max_achievable_tflops": achievable_tflops
        }

    def _parse_precision(self, precision: str):
        mapping = {
            "fp64": torch.float64,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int8": torch.int8,
        }
        if precision not in mapping:
            raise ValueError(f"Unsupported precision: {precision}")
        return mapping[precision]

def profile_hardware() -> Dict[str, Any]:
    """
    Backwards-compatible convenience function for the Referee.
    Performs a full Roofline analysis.
    """
    normalizer = RooflineHardwareNormalizer()
    normalizer.verify_steady_state()
    
    # We profile fp32 by default for general normalizing
    res = normalizer.get_roofline_status(workload_ai=0.0) # AI=0 just to get peaks
    
    return {
        "device": res["device"],
        "peak_flops": res["peak_tflops"] * 1e12,
        "peak_bandwidth": res["peak_bw_gb_s"] * 1e9,
        "ridge_point": res["ridge_point_ai"]
    }

if __name__ == "__main__":
    # Diagnostic test
    try:
        norm = RooflineHardwareNormalizer()
        print(f"Profiling {norm.device_name}...")
        norm.verify_steady_state(duration_secs=2)
        stats = norm.get_roofline_status(workload_ai=50.0)
        print(f"Roofline Analysis: {stats}")
    except Exception as e:
        print(f"Profiling failed: {e}")
