from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, Annotated, Union
from ..core.constants import Q_, ureg
from ..core.types import Quantity, Metadata

class ComputeCore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    peak_flops: Quantity
    precision_flops: Dict[str, Quantity] = Field(default_factory=dict)

class MemoryHierarchy(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    capacity: Quantity
    bandwidth: Quantity

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
    def memory_bw(self) -> Quantity:
        return self.memory.bandwidth

    @property
    def memory_capacity(self) -> Quantity:
        return self.memory.capacity

    @property
    def peak_flops_fp32(self) -> Optional[Quantity]:
        return self.compute.precision_flops.get('fp32')

    @property
    def tf32_flops(self) -> Optional[Quantity]:
        return self.compute.precision_flops.get('tf32')

    @property
    def fp8_flops(self) -> Optional[Quantity]:
        return self.compute.precision_flops.get('fp8')

    @property
    def int8_flops(self) -> Optional[Quantity]:
        return self.compute.precision_flops.get('int8')

    def ridge_point(self) -> Quantity:
        """Calculates the Roofline ridge point (Intensity threshold)."""
        return (self.compute.peak_flops / self.memory.bandwidth).to('flop/byte')

    def __repr__(self):
        return f"HardwareNode({self.name}, {self.release_year})"
