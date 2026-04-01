from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, Annotated, Union
from ..core.constants import Q_, ureg
from ..core.types import Quantity, Metadata
from typing import Optional, Dict, Any, Annotated, Union, ClassVar

class ComputeCore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    peak_flops: Quantity
    precision_flops: Dict[str, Quantity] = Field(default_factory=dict)

class MemoryHierarchy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    capacity: Quantity           # Primary memory (HBM for GPUs, SRAM for MCUs)
    bandwidth: Quantity          # Primary memory bandwidth
    # On-chip SRAM (optional — for modeling FlashAttention tiling on GPUs,
    # or the fast scratchpad on microcontrollers)
    sram_capacity: Optional[Quantity] = None
    sram_bandwidth: Optional[Quantity] = None
    # Flash storage (optional — for TinyML devices where weights live in flash)
    flash_capacity: Optional[Quantity] = None
    flash_bandwidth: Optional[Quantity] = None

class StorageHierarchy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    capacity: Quantity
    bandwidth: Quantity
    latency: Optional[Quantity] = None

class IOInterconnect(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str # e.g., "PCIe Gen4 x16"
    bandwidth: Quantity
    latency: Optional[Quantity] = None

class HardwareNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    release_year: int
    compute: ComputeCore
    memory: MemoryHierarchy
    storage: Optional[StorageHierarchy] = None
    interconnect: Optional[IOInterconnect] = None
    tdp: Optional[Quantity] = None
    battery_capacity: Optional[Quantity] = None
    unit_cost: Optional[Quantity] = None
    dispatch_tax: Quantity = Field(default_factory=lambda: Q_("0.01 ms"))
    metadata: Metadata = Field(default_factory=Metadata)

    # Backward-compatible flat access properties (chapters use these)
    @property
    def peak_flops(self) -> Quantity:
        return self.compute.peak_flops

    @property
    def peak_flops_fp32(self) -> Optional[Quantity]:
        return self.compute.precision_flops.get('fp32')

    @property
    def memory_bw(self) -> Quantity:
        return self.memory.bandwidth

    @property
    def memory_capacity(self) -> Quantity:
        return self.memory.capacity

    @property
    def ram(self) -> Quantity:
        return self.memory.capacity

    @property
    def power_budget(self) -> Optional[Quantity]:
        return self.tdp

    def ridge_point(self) -> Quantity:
        """Calculates the Roofline ridge point (Intensity threshold)."""
        return (self.compute.peak_flops / self.memory.bandwidth).to('flop/byte')

    def __repr__(self):
        return f"HardwareNode({self.name}, {self.release_year})"

    # ─────────────────────────────────────────────────────────────
    # Backward-compatibility aliases (used by older textbook code)
    # ─────────────────────────────────────────────────────────────
    _legacy_aliases: ClassVar[dict[str, Any]] = {
        "peak_flops": lambda self: self.compute.peak_flops,
        "peak_flops_fp32": lambda self: self.compute.precision_flops.get("fp32"),
        "tf32_flops": lambda self: self.compute.precision_flops.get("tf32"),
        "fp8_flops": lambda self: self.compute.precision_flops.get("fp8"),
        "int8_flops": lambda self: self.compute.precision_flops.get("int8"),
        "int4_flops": lambda self: self.compute.precision_flops.get("int4"),
        "memory_bw": lambda self: self.memory.bandwidth,
        "memory_capacity": lambda self: self.memory.capacity,
        "ram": lambda self: self.memory.capacity,
        "power_budget": lambda self: self.tdp,
    }

    def __getattr__(self, name):
        aliases = type(self)._legacy_aliases
        if name in aliases:
            value = aliases[name](self)
            if value is not None:
                return value
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")