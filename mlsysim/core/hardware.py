# hardware.py
# Hierarchical Hardware Definitions for MLSys Textbook

import pint
from dataclasses import dataclass
from typing import Optional, Tuple
from .constants import (
    ureg, Q_,
    V100_MEM_BW, V100_FLOPS_FP16_TENSOR, V100_MEM_CAPACITY, V100_TDP, V100_FLOPS_FP32,
    A100_MEM_BW, A100_FLOPS_FP16_TENSOR, A100_MEM_CAPACITY, A100_TDP, A100_FLOPS_FP32, A100_FLOPS_TF32, A100_FLOPS_INT8,
    H100_MEM_BW, H100_FLOPS_FP16_TENSOR, H100_MEM_CAPACITY, H100_TDP, H100_FLOPS_TF32, H100_FLOPS_FP8_TENSOR, H100_FLOPS_INT8,
    B200_MEM_BW, B200_FLOPS_FP16_TENSOR, B200_MEM_CAPACITY, B200_TDP, B200_FLOPS_FP8_TENSOR, B200_FLOPS_INT4,
    MI300X_MEM_BW, MI300X_FLOPS_FP16_TENSOR, MI300X_MEM_CAPACITY, MI300X_TDP,
    T4_MEM_BW, T4_FLOPS_FP16_TENSOR, T4_TDP, T4_FLOPS_INT8,
    TPUV4_MEM_BW, TPUV4_FLOPS_BF16,
    TPUV6_MEM_BW, TPUV6_FLOPS_BF16, TPUV6_MEM_CAPACITY,
    MOBILE_NPU_MEM_BW, MOBILE_NPU_TOPS_INT8,
    ESP32_RAM, ESP32_FLASH, ESP32_POWER_MAX,
    MCU_RAM_KIB,
    NETWORK_10G_BW, NETWORK_100G_BW
)

@dataclass(frozen=True)
class HardwareSpec:
    name: str
    release_year: int
    memory_bw: Q_
    peak_flops: Q_ # Usually FP16 Tensor for AI accelerators
    memory_capacity: Q_
    tdp: Optional[Q_] = None
    battery_capacity: Optional[Q_] = None
    dispatch_tax: Q_ = 0.01 * ureg.ms # Default 10us
    
    # Precision-specific FLOPS
    peak_flops_fp32: Optional[Q_] = None
    tf32_flops: Optional[Q_] = None
    fp8_flops: Optional[Q_] = None
    int8_flops: Optional[Q_] = None
    
    def __post_init__(self):
        """Validate hardware specs: correct dimension type first, then positive value."""
        from .constants import ureg

        def _validate(qty, name, target_unit, dim_desc):
            if not qty.is_compatible_with(target_unit):
                raise pint.DimensionalityError(
                    qty.units, target_unit,
                    extra_msg=f" — {self.name}.{name} must be {dim_desc}, got {qty.units}"
                )
            if qty.magnitude <= 0:
                raise ValueError(f"{self.name}.{name} must be positive, got {qty}")

        _validate(self.memory_bw,       "memory_bw",       ureg.byte/ureg.second, "data/time (e.g. GB/s)")
        _validate(self.peak_flops,      "peak_flops",      ureg.flop/ureg.second, "compute rate (e.g. TFLOPs/s)")
        _validate(self.memory_capacity, "memory_capacity", ureg.byte,             "data size (e.g. GiB)")
        _validate(self.dispatch_tax,    "dispatch_tax",    ureg.second,           "time (e.g. ms)")
        if self.tdp:
            _validate(self.tdp,         "tdp",             ureg.watt,             "power (e.g. W)")
        if self.battery_capacity:
            _validate(self.battery_capacity, "battery_capacity", ureg.joule,      "energy (e.g. Wh or J)")

    def ridge_point(self) -> Q_:
        """Calculates the Roofline ridge point (Intensity threshold)."""
        # FLOPS / BW = Ops/Byte
        return (self.peak_flops / self.memory_bw).to('flop/byte')

    def __repr__(self):
        return f"Hardware({self.name}, {self.release_year})"

@dataclass(frozen=True)
class NetworkSpec:
    name: str
    bandwidth: Q_

class Networks:
    Ethernet_10G = NetworkSpec("10GbE", NETWORK_10G_BW)
    Ethernet_100G = NetworkSpec("100GbE", NETWORK_100G_BW)

class Cloud:
    """Datacenter-scale Accelerators."""
    V100 = HardwareSpec("NVIDIA V100", 2017, V100_MEM_BW, V100_FLOPS_FP16_TENSOR, V100_MEM_CAPACITY, V100_TDP, 
                        peak_flops_fp32=V100_FLOPS_FP32, dispatch_tax=0.02 * ureg.ms)
    A100 = HardwareSpec("NVIDIA A100", 2020, A100_MEM_BW, A100_FLOPS_FP16_TENSOR, A100_MEM_CAPACITY, A100_TDP,
                        peak_flops_fp32=A100_FLOPS_FP32, tf32_flops=A100_FLOPS_TF32, int8_flops=A100_FLOPS_INT8,
                        dispatch_tax=0.015 * ureg.ms)
    H100 = HardwareSpec("NVIDIA H100", 2022, H100_MEM_BW, H100_FLOPS_FP16_TENSOR, H100_MEM_CAPACITY, H100_TDP,
                        tf32_flops=H100_FLOPS_TF32, fp8_flops=H100_FLOPS_FP8_TENSOR, int8_flops=H100_FLOPS_INT8,
                        dispatch_tax=0.01 * ureg.ms)
    B200 = HardwareSpec("NVIDIA B200", 2024, B200_MEM_BW, B200_FLOPS_FP16_TENSOR, B200_MEM_CAPACITY, B200_TDP,
                        fp8_flops=B200_FLOPS_FP8_TENSOR, int8_flops=B200_FLOPS_INT4, dispatch_tax=0.008 * ureg.ms)
    
    MI300X = HardwareSpec("AMD MI300X", 2023, MI300X_MEM_BW, MI300X_FLOPS_FP16_TENSOR, MI300X_MEM_CAPACITY, MI300X_TDP,
                          dispatch_tax=0.012 * ureg.ms)
    
    T4   = HardwareSpec("NVIDIA T4",   2018, T4_MEM_BW,   T4_FLOPS_FP16_TENSOR,   16 * ureg.GiB,     T4_TDP,
                        int8_flops=T4_FLOPS_INT8, dispatch_tax=0.03 * ureg.ms)
    
    TPUv4 = HardwareSpec("Google TPU v4", 2021, TPUV4_MEM_BW, TPUV4_FLOPS_BF16, 32 * ureg.GiB, dispatch_tax=0.05 * ureg.ms)
    TPUv6 = HardwareSpec("Google TPU v6", 2025, TPUV6_MEM_BW, TPUV6_FLOPS_BF16, TPUV6_MEM_CAPACITY, dispatch_tax=0.04 * ureg.ms)

class Edge:
    """Mobile and Robotics Hardware."""
    Generic_Phone = HardwareSpec("Smartphone", 2024, MOBILE_NPU_MEM_BW, MOBILE_NPU_TOPS_INT8, 8 * ureg.GiB, 
                                 battery_capacity=15 * ureg.Wh, dispatch_tax=1.0 * ureg.ms) # High OS overhead
    
    # Specific Edge Devices
    Coral = HardwareSpec("Google Coral Dev", 2019, 25 * ureg.GB/ureg.s, 4 * ureg.TFLOPs/ureg.s, 1 * ureg.GB, 2 * ureg.W, 
                         dispatch_tax=0.5 * ureg.ms)
    JetsonOrinNX = HardwareSpec("NVIDIA Jetson Orin NX", 2023, 102 * ureg.GB/ureg.s, 100 * ureg.TFLOPs/ureg.s, 16 * ureg.GB, 25 * ureg.W, 
                                dispatch_tax=0.2 * ureg.ms)
    NUC_Movidius = HardwareSpec("Intel NUC + Movidius", 2020, 50 * ureg.GB/ureg.s, 4 * ureg.TFLOPs/ureg.s, 16 * ureg.GB, 15 * ureg.W, 
                                dispatch_tax=2.0 * ureg.ms)
    
    # Servers
    GenericServer = HardwareSpec("Edge Server", 2024, 100 * ureg.GB/ureg.s, 1 * ureg.TFLOPs/ureg.s, 128 * ureg.GB, 300 * ureg.W, 
                                 dispatch_tax=0.1 * ureg.ms)

class Tiny:
    """Microcontrollers and Embedded Systems."""
    # ESP32 at 240MHz is ~240 MIPS, for AI math without FPU it's roughly 100-200 MFLOPS (0.0001-0.0002 TFLOPS)
    ESP32 = HardwareSpec("ESP32-CAM", 2019, 0.1 * ureg.GB/ureg.second, 0.0002 * ureg.TFLOPs/ureg.second, ESP32_RAM, ESP32_POWER_MAX, 
                         dispatch_tax=5.0 * ureg.ms) # Very high overhead relative to math
    Generic_MCU = HardwareSpec("Cortex-M7", 2020, 0.05 * ureg.GB/ureg.second, 0.001 * ureg.TFLOPs/ureg.second, MCU_RAM_KIB, 
                               dispatch_tax=2.0 * ureg.ms)

class Hardware:
    Cloud = Cloud
    Edge = Edge
    Tiny = Tiny
    Networks = Networks
    
    # Aliases for the most common ones
    V100 = Cloud.V100
    A100 = Cloud.A100
    H100 = Cloud.H100
    B200 = Cloud.B200
    TPUv4 = Cloud.TPUv4
    ESP32 = Tiny.ESP32
