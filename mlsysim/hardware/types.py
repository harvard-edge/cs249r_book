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

class HardwareNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    release_year: int
    compute: ComputeCore
    memory: MemoryHierarchy
    tdp: Optional[Quantity] = None
    battery_capacity: Optional[Quantity] = None
    unit_cost: Optional[Quantity] = None
    dispatch_tax: Quantity = Field(default_factory=lambda: Q_("0.01 ms"))
    metadata: Metadata = Field(default_factory=Metadata)

    def ridge_point(self) -> Quantity:
        """Calculates the Roofline ridge point (Intensity threshold)."""
        return (self.compute.peak_flops / self.memory.bandwidth).to('flop/byte')

    def __repr__(self):
        return f"HardwareNode({self.name}, {self.release_year})"
