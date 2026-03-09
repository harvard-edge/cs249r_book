from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Any, Annotated, List, Union
from ..core.constants import Q_, ureg, PUE_BEST_AIR
from ..hardware.types import HardwareNode
from ..infra.types import Datacenter, GridProfile
from ..core.types import Quantity

class DeploymentTier(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    ram: Quantity
    storage: Quantity
    typical_latency_budget: Quantity

class NetworkFabric(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    topology: str = "fat-tree"
    bandwidth: Quantity
    latency: Optional[Quantity] = None
    oversubscription_ratio: float = 1.0  # 1.0 = Non-blocking, 3.0 = 3:1 blocking

class Node(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    accelerator: HardwareNode
    accelerators_per_node: int
    intra_node_bw: Quantity
    nics_per_node: int = 1
    psus_per_node: int = 2

class Fleet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    node: Node
    count: int # total nodes
    fabric: NetworkFabric
    
    # Environment Linkage
    region: Optional[GridProfile] = None
    datacenter: Optional[Datacenter] = None
    
    mtbf_hours: Optional[Quantity] = None
    
    @property
    def total_accelerators(self) -> int:
        return self.count * self.node.accelerators_per_node
    
    @property
    def effective_pue(self) -> float:
        """Returns the PUE of the datacenter, or a default if not specified."""
        if self.datacenter:
            return self.datacenter.pue
        if self.region:
            return self.region.pue
        return PUE_BEST_AIR  # Default Hyperscale PUE
