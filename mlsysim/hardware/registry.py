from .types import HardwareNode, ComputeCore, MemoryHierarchy, StorageHierarchy, IOInterconnect
from ..core.registry import Registry
from ..core.constants import (
    ureg, USD,
    V100_MEM_BW, V100_FLOPS_FP16_TENSOR, V100_MEM_CAPACITY, V100_TDP, V100_FLOPS_FP32, V100_UNIT_COST,
    A100_MEM_BW, A100_FLOPS_FP16_TENSOR, A100_MEM_CAPACITY, A100_TDP, A100_FLOPS_FP32, A100_FLOPS_TF32, A100_FLOPS_INT8, A100_UNIT_COST,
    H100_MEM_BW, H100_FLOPS_FP16_TENSOR, H100_MEM_CAPACITY, H100_TDP, H100_FLOPS_TF32, H100_FLOPS_FP8_TENSOR, H100_FLOPS_INT8, H100_UNIT_COST,
    H200_MEM_BW, H200_MEM_CAPACITY, H200_TDP, H200_UNIT_COST,
    B200_MEM_BW, B200_FLOPS_FP16_TENSOR, B200_MEM_CAPACITY, B200_TDP, B200_FLOPS_FP8_TENSOR, B200_FLOPS_INT4, B200_UNIT_COST,
    NVL72_FLOPS_FP8_TENSOR, NVL72_MEM_CAPACITY, NVL72_MEM_BW, NVL72_NVLINK_BW, NVL72_TDP, NVL72_UNIT_COST,
    MI300X_MEM_BW, MI300X_FLOPS_FP16_TENSOR, MI300X_MEM_CAPACITY, MI300X_TDP, MI300X_UNIT_COST,
    MI300X_FLOPS_FP8, MI300X_FLOPS_INT8, MI300X_FLOPS_FP32,
    TPUV5P_MEM_BW, TPUV5P_FLOPS_BF16, TPUV5P_MEM_CAPACITY, TPUV5P_TDP, TPUV5P_FLOPS_INT8,
    T4_MEM_BW, T4_FLOPS_FP16_TENSOR, T4_TDP, T4_FLOPS_INT8, T4_UNIT_COST,
    WSE3_FLOPS_FP16, WSE3_MEM_CAPACITY, WSE3_MEM_BW, WSE3_TDP, WSE3_CORES, CEREBRAS_CS3_UNIT_COST,
    PCIE_GEN3_BW, PCIE_GEN4_BW, PCIE_GEN5_BW, NVME_SEQUENTIAL_BW
)

class CloudHardware(Registry):
    """Datacenter-scale accelerators (Volume II)."""
    V100 = HardwareNode(
        name="NVIDIA V100",
        release_year=2017,
        compute=ComputeCore(peak_flops=V100_FLOPS_FP16_TENSOR, precision_flops={"fp32": V100_FLOPS_FP32}),
        memory=MemoryHierarchy(capacity=V100_MEM_CAPACITY, bandwidth=V100_MEM_BW),
        interconnect=IOInterconnect(name="PCIe Gen3 x16", bandwidth=PCIE_GEN3_BW),
        tdp=V100_TDP,
        unit_cost=V100_UNIT_COST,
        dispatch_tax=0.02 * ureg.ms
    )

    A100 = HardwareNode(
        name="NVIDIA A100",
        release_year=2020,
        compute=ComputeCore(peak_flops=A100_FLOPS_FP16_TENSOR, precision_flops={"fp32": A100_FLOPS_FP32, "tf32": A100_FLOPS_TF32, "int8": A100_FLOPS_INT8}),
        memory=MemoryHierarchy(capacity=A100_MEM_CAPACITY, bandwidth=A100_MEM_BW),
        interconnect=IOInterconnect(name="PCIe Gen4 x16", bandwidth=PCIE_GEN4_BW),
        tdp=A100_TDP,
        unit_cost=A100_UNIT_COST,
        dispatch_tax=0.015 * ureg.ms,
        metadata={"source_url": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf", "last_verified": "2025-03-06"}
    )

    H100 = HardwareNode(
        name="NVIDIA H100",
        release_year=2022,
        compute=ComputeCore(peak_flops=H100_FLOPS_FP16_TENSOR, precision_flops={"tf32": H100_FLOPS_TF32, "fp8": H100_FLOPS_FP8_TENSOR, "int8": H100_FLOPS_INT8}),
        memory=MemoryHierarchy(capacity=H100_MEM_CAPACITY, bandwidth=H100_MEM_BW),
        storage=StorageHierarchy(capacity=2 * ureg.TB, bandwidth=NVME_SEQUENTIAL_BW),
        interconnect=IOInterconnect(name="PCIe Gen5 x16", bandwidth=PCIE_GEN5_BW),
        tdp=H100_TDP,
        unit_cost=H100_UNIT_COST,
        dispatch_tax=0.01 * ureg.ms,
        metadata={"source_url": "https://resources.nvidia.com/en-us-tensor-core/nvidia-h100-tensor-core-gpu-datasheet", "last_verified": "2025-03-06"}
    )

    H200 = HardwareNode(
        name="NVIDIA H200",
        release_year=2023,
        compute=ComputeCore(peak_flops=H100_FLOPS_FP16_TENSOR, precision_flops={"tf32": H100_FLOPS_TF32, "fp8": H100_FLOPS_FP8_TENSOR, "int8": H100_FLOPS_INT8}),
        memory=MemoryHierarchy(capacity=H200_MEM_CAPACITY, bandwidth=H200_MEM_BW),
        storage=StorageHierarchy(capacity=4 * ureg.TB, bandwidth=NVME_SEQUENTIAL_BW),
        interconnect=IOInterconnect(name="PCIe Gen5 x16", bandwidth=PCIE_GEN5_BW),
        tdp=H200_TDP,
        unit_cost=H200_UNIT_COST,
        dispatch_tax=0.01 * ureg.ms
    )

    B200 = HardwareNode(
        name="NVIDIA B200",
        release_year=2024,
        compute=ComputeCore(peak_flops=B200_FLOPS_FP16_TENSOR, precision_flops={"fp8": B200_FLOPS_FP8_TENSOR, "int4": B200_FLOPS_INT4}),
        memory=MemoryHierarchy(capacity=B200_MEM_CAPACITY, bandwidth=B200_MEM_BW),
        tdp=B200_TDP,
        unit_cost=B200_UNIT_COST,
        dispatch_tax=0.008 * ureg.ms,
        metadata={"source_url": "https://www.nvidia.com/en-us/data-center/blackwell/", "last_verified": "2025-03-06"}
    )

    GB200_NVL72 = HardwareNode(
        name="NVIDIA GB200 NVL72",
        release_year=2024,
        compute=ComputeCore(peak_flops=NVL72_FLOPS_FP8_TENSOR),
        memory=MemoryHierarchy(capacity=NVL72_MEM_CAPACITY, bandwidth=NVL72_MEM_BW),
        interconnect=IOInterconnect(name="NVLink Switch (Bisection)", bandwidth=NVL72_NVLINK_BW),
        tdp=NVL72_TDP,
        unit_cost=NVL72_UNIT_COST,
        dispatch_tax=0.005 * ureg.ms,
        metadata={"source_url": "https://www.nvidia.com/en-us/data-center/gb200-nvl72/"}
    )

    MI300X = HardwareNode(
        name="AMD MI300X",
        release_year=2023,
        compute=ComputeCore(peak_flops=MI300X_FLOPS_FP16_TENSOR, precision_flops={"fp8": MI300X_FLOPS_FP8, "int8": MI300X_FLOPS_INT8, "fp32": MI300X_FLOPS_FP32}),
        memory=MemoryHierarchy(capacity=MI300X_MEM_CAPACITY, bandwidth=MI300X_MEM_BW),
        tdp=MI300X_TDP,
        unit_cost=MI300X_UNIT_COST,
        dispatch_tax=0.012 * ureg.ms
    )

    TPUv5p = HardwareNode(
        name="Google TPU v5p",
        release_year=2023,
        compute=ComputeCore(peak_flops=TPUV5P_FLOPS_BF16, precision_flops={"int8": TPUV5P_FLOPS_INT8}),
        memory=MemoryHierarchy(capacity=TPUV5P_MEM_CAPACITY, bandwidth=TPUV5P_MEM_BW),
        tdp=TPUV5P_TDP,
        dispatch_tax=0.04 * ureg.ms
    )

    T4 = HardwareNode(
        name="NVIDIA T4",
        release_year=2018,
        compute=ComputeCore(peak_flops=T4_FLOPS_FP16_TENSOR, precision_flops={"int8": T4_FLOPS_INT8}),
        memory=MemoryHierarchy(capacity=16 * ureg.GiB, bandwidth=T4_MEM_BW),
        tdp=T4_TDP,
        unit_cost=T4_UNIT_COST,
        dispatch_tax=0.03 * ureg.ms
    )

    Cerebras_CS3 = HardwareNode(
        name="Cerebras CS-3 (WSE-3)",
        release_year=2024,
        # A single WSE acts as a gigantic compute core with minimal dispatch tax
        compute=ComputeCore(peak_flops=WSE3_FLOPS_FP16),
        # Memory reflects the 44GB on-wafer SRAM, meaning large weights must stream from MemoryX
        memory=MemoryHierarchy(capacity=WSE3_MEM_CAPACITY, bandwidth=WSE3_MEM_BW),
        # Injection bandwidth from MemoryX (approx 1.2 TB/s per WSE)
        interconnect=IOInterconnect(name="SwarmX / MemoryX", bandwidth=1.2 * ureg.TB / ureg.second),
        tdp=WSE3_TDP,
        unit_cost=CEREBRAS_CS3_UNIT_COST,
        dispatch_tax=0.001 * ureg.ms,
        metadata={"source_url": "https://www.cerebras.net/product-system/"}
    )

    # Backward-compatible alias
    TPUv4 = TPUv5p

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

    # Backward-compatible alias
    Generic_Phone = MobileHardware.iPhone15Pro

class TinyHardware(Registry):
    """Microcontrollers and sub-watt devices."""
    ESP32_S3 = HardwareNode(
        name="ESP32-S3 (AI)",
        release_year=2022,
        # ESP32-S3 has no FP16 hardware; INT8 via vector extensions.
        # ~2.4 GOPS INT8 (240 MHz dual-core, vector instructions)
        compute=ComputeCore(
            peak_flops=0.0005 * ureg.TFLOPs/ureg.s,
            precision_flops={"int8": 0.0024 * ureg.TFLOPs/ureg.s}
        ),
        # 512 KiB SRAM (usable ~256 KiB after RTOS/stack); 8 MB flash for weights
        memory=MemoryHierarchy(
            capacity=512 * ureg.KiB,                  # SRAM (activations + runtime)
            bandwidth=0.96 * ureg.GB/ureg.s,          # SRAM bandwidth @ 240 MHz
            flash_capacity=8 * ureg.MB,               # SPI flash (weight storage)
            flash_bandwidth=0.08 * ureg.GB/ureg.s,    # Flash read ~80 MB/s (XIP)
        ),
        tdp=0.4 * ureg.W,              # Inference-only power (not WiFi-on 1.2W)
        dispatch_tax=1.0 * ureg.ms     # TFLite Micro interpreter overhead
    )
    ESP32 = ESP32_S3 # Alias for backward compatibility

    nRF52840 = HardwareNode(
        name="Nordic nRF52840 (Cortex-M4F)",
        release_year=2018,
        # MLPerf Tiny reference platform. Cortex-M4F @ 64 MHz.
        compute=ComputeCore(
            peak_flops=0.000064 * ureg.TFLOPs/ureg.s,  # ~64 MFLOPS (single-precision)
            precision_flops={"int8": 0.000128 * ureg.TFLOPs/ureg.s}  # ~128 MOPS INT8
        ),
        memory=MemoryHierarchy(
            capacity=256 * ureg.KiB,                  # 256 KiB RAM
            bandwidth=0.256 * ureg.GB/ureg.s,         # SRAM bandwidth @ 64 MHz
            flash_capacity=1 * ureg.MB,               # 1 MB internal flash
            flash_bandwidth=0.064 * ureg.GB/ureg.s,   # Flash read ~64 MB/s
        ),
        tdp=0.015 * ureg.W,            # ~15 mW active inference
        dispatch_tax=0.5 * ureg.ms
    )

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
    NVL72 = CloudHardware.GB200_NVL72
    MI300X = CloudHardware.MI300X
    TPUv5p = CloudHardware.TPUv5p
    TPUv4 = CloudHardware.TPUv5p
    T4 = CloudHardware.T4
    CerebrasCS3 = CloudHardware.Cerebras_CS3

    DGXSpark = WorkstationHardware.DGX_Spark
    MacBook = WorkstationHardware.MacBookM3Max

    iPhone = MobileHardware.iPhone15Pro
    Snapdragon = MobileHardware.Snapdragon8Gen3
    Jetson = EdgeHardware.JetsonOrinNX
    ESP32 = TinyHardware.ESP32_S3
    Himax = TinyHardware.HimaxWE1

from ..systems.registry import Fabrics
Hardware.Networks = Fabrics