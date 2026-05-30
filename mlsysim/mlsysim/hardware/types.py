from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict
from ..core.constants import Q_
from ..core.types import Quantity, Metadata

class ComputeCore(BaseModel):
    """
    Represents the processing units of a hardware accelerator.
    
    Contains the theoretical peak throughput capabilities of the silicon,
    differentiated by numerical precision.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    peak_flops: Quantity
    precision_flops: Dict[str, Quantity] = Field(default_factory=dict)
    # Streaming-multiprocessor count. Present for GPUs; absent (None) for device
    # classes that have no SM concept (MCUs, TPUs). Pairs with the per-SM memory
    # fields below to derive chip-level totals (e.g. register file across the die).
    sm_count: Optional[int] = None

class MemoryHierarchy(BaseModel):
    """
    Represents the memory subsystem of a hardware accelerator.
    
    Captures capacity and bandwidth across different levels of the memory
    hierarchy, from primary HBM/DRAM down to on-chip SRAM or Flash storage.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    capacity: Quantity           # Primary memory (HBM for GPUs, SRAM for MCUs)
    bandwidth: Quantity          # Primary memory bandwidth
    # On-chip SRAM (optional — for modeling FlashAttention tiling on GPUs,
    # or the fast scratchpad on microcontrollers)
    sram_capacity: Optional[Quantity] = None
    sram_bandwidth: Optional[Quantity] = None
    # Flash storage (optional — for TinyML devices where weights live in flash)
    flash_capacity: Optional[Quantity] = None
    flash_bandwidth: Optional[Quantity] = None
    l2_cache: Optional[Quantity] = None
    # Per-SM on-chip memory. Present for GPUs; absent (None) for device classes
    # without SMs. Multiply by ComputeCore.sm_count to derive chip-level totals.
    register_file_per_sm: Optional[Quantity] = None
    shared_memory_per_sm: Optional[Quantity] = None

class StorageHierarchy(BaseModel):
    """
    Represents the persistent storage subsystem connected to the hardware.
    
    Crucial for analyzing checkpointing overheads and the Data Wall (data
    ingestion rates for training).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    capacity: Quantity
    bandwidth: Quantity
    latency: Optional[Quantity] = None

class IOInterconnect(BaseModel):
    """
    Represents a point-to-point interconnect link.
    
    Used to model PCIe links (host to device) or NVLink/ICI connections
    (device to device within a node).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str # e.g., "PCIe Gen4 x16"
    bandwidth: Quantity
    latency: Optional[Quantity] = None

class HardwareNode(BaseModel):
    """
    Layer B (Hardware Supply): Represents a complete hardware accelerator.
    
    This is the primary object that solvers evaluate against. It encapsulates
    the compute, memory, IO, and power constraints of a single physical device 
    (e.g., an NVIDIA H100 GPU or a Jetson Orin Nano).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    release_year: int
    compute: ComputeCore
    memory: MemoryHierarchy
    storage: Optional[StorageHierarchy] = None
    interconnect: Optional[IOInterconnect] = None  # Host I/O (e.g., PCIe)
    nvlink: Optional[IOInterconnect] = None        # Intra-node GPU-GPU (NVLink/ICI)
    tdp: Optional[Quantity] = None
    tdp_min: Optional[Quantity] = None
    tdp_max: Optional[Quantity] = None
    battery_capacity: Optional[Quantity] = None
    unit_cost: Optional[Quantity] = None
    unit_cost_max: Optional[Quantity] = None
    accelerator_count: Optional[int] = None
    embodied_carbon_kg: Optional[float] = Field(
        default=None,
        description="Embodied CO2e in kg from manufacturing, packaging, and transport (Gupta et al. 2022)."
    )
    dispatch_tax: Quantity = Field(default_factory=lambda: Q_("0.01 ms"))
    metadata: Metadata = Field(default_factory=Metadata)

    def ridge_point(self, precision: Optional[str] = None) -> Quantity:
        """
        Calculates the Roofline ridge point (Intensity threshold).
        
        Args:
            precision: Optional precision string (e.g., 'fp16', 'fp8', 'int8').
                If provided, uses the corresponding precision flops from ComputeCore.
                If None (default), uses the default peak_flops.
        """
        flops = self.compute.peak_flops
        if precision:
            if precision.lower() in self.compute.precision_flops:
                flops = self.compute.precision_flops[precision.lower()]
            else:
                import warnings
                warnings.warn(
                    f"Precision '{precision}' not found in hardware '{self.name}'; "
                    f"using default peak_flops.",
                    stacklevel=2
                )
        return (flops / self.memory.bandwidth).to('flop/byte')

    def __repr__(self):
        return f"HardwareNode({self.name}, {self.release_year})"
