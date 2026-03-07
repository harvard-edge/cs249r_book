from .types import HardwareNode, ComputeCore, MemoryHierarchy
from ..core.registry import Registry
from ..core.constants import (
    ureg,
    V100_MEM_BW, V100_FLOPS_FP16_TENSOR, V100_MEM_CAPACITY, V100_TDP, V100_FLOPS_FP32,
    A100_MEM_BW, A100_FLOPS_FP16_TENSOR, A100_MEM_CAPACITY, A100_TDP, A100_FLOPS_FP32, A100_FLOPS_TF32, A100_FLOPS_INT8,
    H100_MEM_BW, H100_FLOPS_FP16_TENSOR, H100_MEM_CAPACITY, H100_TDP, H100_FLOPS_TF32, H100_FLOPS_FP8_TENSOR, H100_FLOPS_INT8,
    B200_MEM_BW, B200_FLOPS_FP16_TENSOR, B200_MEM_CAPACITY, B200_TDP, B200_FLOPS_FP8_TENSOR, B200_FLOPS_INT4,
    MI300X_MEM_BW, MI300X_FLOPS_FP16_TENSOR, MI300X_MEM_CAPACITY, MI300X_TDP,
    TPUV5P_MEM_BW, TPUV5P_FLOPS_BF16, TPUV5P_MEM_CAPACITY,
    T4_MEM_BW, T4_FLOPS_FP16_TENSOR, T4_TDP, T4_FLOPS_INT8
)

class CloudHardware(Registry):
    """Datacenter-scale accelerators (Volume II)."""
    V100 = HardwareNode(
        name="NVIDIA V100",
        release_year=2017,
        compute=ComputeCore(peak_flops=V100_FLOPS_FP16_TENSOR, precision_flops={"fp32": V100_FLOPS_FP32}),
        memory=MemoryHierarchy(capacity=V100_MEM_CAPACITY, bandwidth=V100_MEM_BW),
        tdp=V100_TDP,
        dispatch_tax=0.02 * ureg.ms
    )

    A100 = HardwareNode(
        name="NVIDIA A100",
        release_year=2020,
        compute=ComputeCore(peak_flops=A100_FLOPS_FP16_TENSOR, precision_flops={"fp32": A100_FLOPS_FP32, "tf32": A100_FLOPS_TF32, "int8": A100_FLOPS_INT8}),
        memory=MemoryHierarchy(capacity=A100_MEM_CAPACITY, bandwidth=A100_MEM_BW),
        tdp=A100_TDP,
        dispatch_tax=0.015 * ureg.ms,
        metadata={"source_url": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf", "last_verified": "2025-03-06"}
    )

    H100 = HardwareNode(
        name="NVIDIA H100",
        release_year=2022,
        compute=ComputeCore(peak_flops=H100_FLOPS_FP16_TENSOR, precision_flops={"tf32": H100_FLOPS_TF32, "fp8": H100_FLOPS_FP8_TENSOR, "int8": H100_FLOPS_INT8}),
        memory=MemoryHierarchy(capacity=H100_MEM_CAPACITY, bandwidth=H100_MEM_BW),
        tdp=H100_TDP,
        dispatch_tax=0.01 * ureg.ms,
        metadata={"source_url": "https://resources.nvidia.com/en-us-tensor-core/nvidia-h100-tensor-core-gpu-datasheet", "last_verified": "2025-03-06"}
    )
    
    H200 = HardwareNode(
        name="NVIDIA H200",
        release_year=2023,
        compute=ComputeCore(peak_flops=H100_FLOPS_FP16_TENSOR),
        memory=MemoryHierarchy(capacity=141 * ureg.GB, bandwidth=4.8 * ureg.TB/ureg.s),
        tdp=700 * ureg.W,
        dispatch_tax=0.01 * ureg.ms
    )

    B200 = HardwareNode(
        name="NVIDIA B200",
        release_year=2024,
        compute=ComputeCore(peak_flops=B200_FLOPS_FP16_TENSOR, precision_flops={"fp8": B200_FLOPS_FP8_TENSOR, "int4": B200_FLOPS_INT4}),
        memory=MemoryHierarchy(capacity=B200_MEM_CAPACITY, bandwidth=B200_MEM_BW),
        tdp=1000 * ureg.W,
        dispatch_tax=0.008 * ureg.ms
    )

    MI300X = HardwareNode(
        name="AMD MI300X",
        release_year=2023,
        compute=ComputeCore(peak_flops=1300 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=192 * ureg.GB, bandwidth=5.3 * ureg.TB/ureg.s),
        tdp=750 * ureg.W,
        dispatch_tax=0.012 * ureg.ms
    )
    
    TPUv5p = HardwareNode(
        name="Google TPU v5p",
        release_year=2023,
        compute=ComputeCore(peak_flops=TPUV5P_FLOPS_BF16),
        memory=MemoryHierarchy(capacity=TPUV5P_MEM_CAPACITY, bandwidth=TPUV5P_MEM_BW),
        tdp=300 * ureg.W,
        dispatch_tax=0.04 * ureg.ms
    )

    T4 = HardwareNode(
        name="NVIDIA T4",
        release_year=2018,
        compute=ComputeCore(peak_flops=T4_FLOPS_FP16_TENSOR, precision_flops={"int8": T4_FLOPS_INT8}),
        memory=MemoryHierarchy(capacity=16 * ureg.GiB, bandwidth=T4_MEM_BW),
        tdp=T4_TDP,
        dispatch_tax=0.03 * ureg.ms
    )

class WorkstationHardware(Registry):
    """Personal computing systems used for local development."""
    DGX_Spark = HardwareNode(
        name="NVIDIA DGX Spark (GB10)",
        release_year=2024,
        compute=ComputeCore(
            peak_flops=250 * ureg.TFLOPs/ureg.s, 
            precision_flops={"fp8": 500 * ureg.TFLOPs/ureg.s, "fp4": 1000 * ureg.TFLOPs/ureg.s}
        ),
        memory=MemoryHierarchy(capacity=128 * ureg.GB, bandwidth=500 * ureg.GB/ureg.s),
        tdp=250 * ureg.W,
        dispatch_tax=0.01 * ureg.ms
    )

    MacBookM3Max = HardwareNode(
        name="MacBook Pro (M3 Max)",
        release_year=2023,
        compute=ComputeCore(peak_flops=14.2 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=128 * ureg.GB, bandwidth=400 * ureg.GB/ureg.s),
        tdp=100 * ureg.W,
        dispatch_tax=0.05 * ureg.ms
    )

class MobileHardware(Registry):
    """Smartphone and handheld devices (Volume I)."""
    iPhone15Pro = HardwareNode(
        name="iPhone 15 Pro (A17 Pro)",
        release_year=2023,
        compute=ComputeCore(peak_flops=35 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=8 * ureg.GB, bandwidth=100 * ureg.GB/ureg.s),
        tdp=5 * ureg.W,
        battery_capacity=15 * ureg.Wh,
        dispatch_tax=1.0 * ureg.ms
    )
    
    Pixel8 = HardwareNode(
        name="Google Pixel 8 (Tensor G3)",
        release_year=2023,
        compute=ComputeCore(peak_flops=15 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=8 * ureg.GB, bandwidth=60 * ureg.GB/ureg.s),
        tdp=5 * ureg.W,
        dispatch_tax=1.2 * ureg.ms
    )

    Snapdragon8Gen3 = HardwareNode(
        name="Snapdragon 8 Gen 3",
        release_year=2023,
        compute=ComputeCore(peak_flops=45 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=12 * ureg.GB, bandwidth=77 * ureg.GB/ureg.s),
        tdp=5 * ureg.W,
        dispatch_tax=1.5 * ureg.ms
    )

class EdgeHardware(Registry):
    """Robotics and Industrial Edge (Volume I)."""
    JetsonOrinNX = HardwareNode(
        name="NVIDIA Jetson Orin NX",
        release_year=2023,
        compute=ComputeCore(peak_flops=25 * ureg.TFLOPs/ureg.s, precision_flops={"int8": 100 * ureg.TFLOPs/ureg.s}),
        memory=MemoryHierarchy(capacity=16 * ureg.GB, bandwidth=102 * ureg.GB/ureg.s),
        tdp=25 * ureg.W,
        dispatch_tax=0.2 * ureg.ms
    )
    
    Coral = HardwareNode(
        name="Google Coral Edge TPU",
        release_year=2019,
        compute=ComputeCore(peak_flops=4 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=1 * ureg.GB, bandwidth=8 * ureg.GB/ureg.s),
        tdp=2 * ureg.W,
        dispatch_tax=1.0 * ureg.ms
    )
    
    NUC_Movidius = HardwareNode(
        name="Intel NUC + Movidius",
        release_year=2020,
        compute=ComputeCore(peak_flops=1 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=16 * ureg.GB, bandwidth=25 * ureg.GB/ureg.s),
        tdp=15 * ureg.W,
        dispatch_tax=2.0 * ureg.ms
    )
    
    GenericServer = HardwareNode(
        name="Edge Server",
        release_year=2024,
        compute=ComputeCore(peak_flops=1 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=128 * ureg.GB, bandwidth=100 * ureg.GB/ureg.s),
        tdp=300 * ureg.W,
        dispatch_tax=0.1 * ureg.ms
    )

class TinyHardware(Registry):
    """Microcontrollers and sub-watt devices."""
    ESP32_S3 = HardwareNode(
        name="ESP32-S3 (AI)",
        release_year=2022,
        compute=ComputeCore(peak_flops=0.0005 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=512 * ureg.KiB, bandwidth=0.2 * ureg.GB/ureg.s),
        tdp=1.2 * ureg.W,
        dispatch_tax=5.0 * ureg.ms
    )
    ESP32 = ESP32_S3 # Alias for backward compatibility
    
    HimaxWE1 = HardwareNode(
        name="Himax WE-I Plus",
        release_year=2020,
        compute=ComputeCore(peak_flops=0.0002 * ureg.TFLOPs/ureg.s),
        memory=MemoryHierarchy(capacity=2 * ureg.MB, bandwidth=0.1 * ureg.GB/ureg.s),
        tdp=0.005 * ureg.W,
        dispatch_tax=2.0 * ureg.ms
    )

class Hardware(Registry):
    Cloud = CloudHardware
    Workstation = WorkstationHardware
    Mobile = MobileHardware
    Edge = EdgeHardware
    Tiny = TinyHardware
    
    # Common Aliases (Vetted only)
    V100 = CloudHardware.V100
    A100 = CloudHardware.A100
    H100 = CloudHardware.H100
    H200 = CloudHardware.H200
    B200 = CloudHardware.B200
    MI300X = CloudHardware.MI300X
    TPUv5p = CloudHardware.TPUv5p
    T4 = CloudHardware.T4
    
    DGXSpark = WorkstationHardware.DGX_Spark
    MacBook = WorkstationHardware.MacBookM3Max
    
    iPhone = MobileHardware.iPhone15Pro
    Snapdragon = MobileHardware.Snapdragon8Gen3
    Jetson = EdgeHardware.JetsonOrinNX
    ESP32 = TinyHardware.ESP32_S3
    Himax = TinyHardware.HimaxWE1
