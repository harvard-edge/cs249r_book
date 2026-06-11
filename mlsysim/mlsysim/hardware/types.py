from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Dict, Literal
from ..core.units import Q_, ureg
from ..core.types import Quantity, Metadata, require_dimensionality, require_unit_family

class ComputeCore(BaseModel):
    """
    Represents the processing units of a hardware accelerator.
    
    Contains the theoretical peak throughput capabilities of the silicon,
    differentiated by numerical precision.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    peak_flops: Quantity
    precision_flops: Dict[str, Quantity] = Field(default_factory=dict)
    # Streaming-multiprocessor count. Present for GPUs; absent (None) for device
    # classes that have no SM concept (MCUs, TPUs). Pairs with the per-SM memory
    # fields below to derive chip-level totals (e.g. register file across the die).
    sm_count: Optional[int] = None

    @field_validator("peak_flops", mode="after")
    @classmethod
    def _validate_peak_flops(cls, v):
        return require_unit_family(v, ureg.flop / ureg.second, "peak_flops", "operation")

    @field_validator("precision_flops", mode="after")
    @classmethod
    def _validate_precision_flops(cls, v):
        return {
            key: require_unit_family(val, ureg.flop / ureg.second, f"precision_flops[{key!r}]", "operation")
            for key, val in v.items()
        }

class MemoryHierarchy(BaseModel):
    """
    Represents the memory subsystem of a hardware accelerator.
    
    Captures capacity and bandwidth across different levels of the memory
    hierarchy, from primary HBM/DRAM down to on-chip SRAM or Flash storage.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
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

    @field_validator("capacity", "sram_capacity", "flash_capacity", "l2_cache", "register_file_per_sm", "shared_memory_per_sm", mode="after")
    @classmethod
    def _validate_capacity_fields(cls, v, info):
        return require_unit_family(v, ureg.byte, info.field_name, "data")

    @field_validator("bandwidth", "sram_bandwidth", "flash_bandwidth", mode="after")
    @classmethod
    def _validate_bandwidth_fields(cls, v, info):
        return require_unit_family(v, ureg.byte / ureg.second, info.field_name, "data")

class StorageHierarchy(BaseModel):
    """
    Represents the persistent storage subsystem connected to the hardware.
    
    Crucial for analyzing checkpointing overheads and the Data Wall (data
    ingestion rates for training).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    capacity: Quantity
    bandwidth: Quantity
    latency: Optional[Quantity] = None

    @field_validator("capacity", mode="after")
    @classmethod
    def _validate_capacity(cls, v):
        return require_unit_family(v, ureg.byte, "capacity", "data")

    @field_validator("bandwidth", mode="after")
    @classmethod
    def _validate_bandwidth(cls, v):
        return require_unit_family(v, ureg.byte / ureg.second, "bandwidth", "data")

    @field_validator("latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        return require_dimensionality(v, ureg.second, "latency")

class IOInterconnect(BaseModel):
    """
    Represents a point-to-point interconnect link.

    Used to model PCIe links (host to device) or NVLink/ICI connections
    (device to device within a node).

    Direction convention (2026-06-10 audit). ``bandwidth`` stores the figure
    exactly as the vendor datasheet quotes it, and ``direction`` declares
    which convention that figure uses:

    - ``per_direction`` (default): the one-way deliverable rate. PCIe entries
      (64 GB/s Gen5 x16) and Ethernet/RoCE entries use this.
    - ``bidirectional_total``: send + receive summed. NVIDIA quotes NVLink
      this way (H100 "900 GB/s" = 450 GB/s each direction); Google quotes
      TPU ICI the same way.

    Formulas that divide bytes by a link rate (transfer time, ring-collective
    beta terms) need the ONE-WAY rate: always consume
    ``bandwidth_per_direction``, never raw ``bandwidth``. Feeding the
    bidirectional total into a beta term renders every intra-node collective
    ~2x optimistic — the exact bug this field exists to prevent.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str # e.g., "PCIe Gen4 x16"
    bandwidth: Quantity
    latency: Optional[Quantity] = None
    direction: Literal["per_direction", "bidirectional_total"] = "per_direction"

    @property
    def bandwidth_per_direction(self) -> Quantity:
        """One-way deliverable bandwidth — the beta for transfer-time math."""
        if self.direction == "bidirectional_total":
            return self.bandwidth / 2
        return self.bandwidth

    @field_validator("bandwidth", mode="after")
    @classmethod
    def _validate_bandwidth(cls, v):
        return require_unit_family(v, ureg.bit / ureg.second, "bandwidth", "data")

    @field_validator("latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        return require_dimensionality(v, ureg.second, "latency")

class MigProfile(BaseModel):
    """A single NVIDIA Multi-Instance GPU partition profile (e.g. ``1g.10gb``).

    MIG carves one physical GPU into hardware-isolated instances, each with a fixed
    slice of HBM and a fixed number of streaming multiprocessors. The profiles are a
    frozen per-product spec (NVIDIA MIG User Guide), so they live on the device node.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str                     # NVIDIA profile name, e.g. "1g.10gb"
    gpu_memory: Quantity          # dedicated HBM per instance
    sm_count: int                 # streaming multiprocessors per instance
    typical_workload: str = ""

    @field_validator("gpu_memory", mode="after")
    @classmethod
    def _validate_gpu_memory(cls, v):
        return require_unit_family(v, ureg.byte, "gpu_memory", "data")


class HardwareNode(BaseModel):
    """
    Layer B (Hardware Supply): Represents a complete hardware accelerator.
    
    This is the primary object that solvers evaluate against. It encapsulates
    the compute, memory, IO, and power constraints of a single physical device 
    (e.g., an NVIDIA H100 GPU or a Jetson Orin Nano).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
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
    die_area: Optional[Quantity] = None
    battery_capacity: Optional[Quantity] = None
    unit_cost: Optional[Quantity] = None
    unit_cost_max: Optional[Quantity] = None
    accelerator_count: Optional[int] = None
    embodied_carbon_kg: Optional[float] = Field(
        default=None,
        description="Embodied CO2e in kg from manufacturing, packaging, and transport (Gupta et al. 2022)."
    )
    dispatch_tax: Quantity = Field(default_factory=lambda: Q_("0.01 ms"))
    # Multi-Instance GPU partition profiles, when the device supports MIG (A100+).
    # None for devices without MIG. Frozen per-product spec (NVIDIA MIG User Guide).
    mig_profiles: Optional[tuple[MigProfile, ...]] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("tdp", "tdp_min", "tdp_max", mode="after")
    @classmethod
    def _validate_power_fields(cls, v, info):
        return require_dimensionality(v, ureg.watt, info.field_name)

    @field_validator("die_area", mode="after")
    @classmethod
    def _validate_die_area(cls, v):
        return require_dimensionality(v, ureg.meter**2, "die_area")

    @field_validator("battery_capacity", mode="after")
    @classmethod
    def _validate_battery_capacity(cls, v):
        return require_dimensionality(v, ureg.joule, "battery_capacity")

    @field_validator("unit_cost", "unit_cost_max", mode="after")
    @classmethod
    def _validate_cost_fields(cls, v, info):
        return require_unit_family(v, ureg.dollar, info.field_name, "currency")

    @field_validator("dispatch_tax", mode="after")
    @classmethod
    def _validate_dispatch_tax(cls, v):
        return require_dimensionality(v, ureg.second, "dispatch_tax")

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
