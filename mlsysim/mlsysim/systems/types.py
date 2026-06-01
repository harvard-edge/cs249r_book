from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional
from ..hardware.types import HardwareNode
from ..infrastructure.types import Datacenter, GridProfile
from ..core.types import Quantity, Metadata

class DeploymentTier(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    ram: Quantity
    storage: Quantity
    typical_latency_budget: Quantity


class StorageSubsystem(BaseModel):
    """Reusable storage tier or service in a system design."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    storage_tech: Any | None = None
    capacity: Optional[Quantity] = None
    bandwidth: Quantity
    latency: Optional[Quantity] = None
    iops: Optional[float] = None
    media: Optional[str] = None
    interface: Optional[str] = None
    durability: Optional[str] = None
    metadata: Metadata = Field(default_factory=Metadata)


class NodeStorageConfig(BaseModel):
    """Per-node storage composition, such as four local NVMe drives."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    device: StorageSubsystem
    devices_per_node: int = 1
    metadata: Metadata = Field(default_factory=Metadata)

    @property
    def aggregate_bandwidth(self) -> Quantity:
        return self.devices_per_node * self.device.bandwidth


class CheckpointStoragePath(BaseModel):
    """Local staging plus durable checkpoint destination for training fleets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    local_stage: Optional[NodeStorageConfig] = None
    durable_store: Optional[StorageSubsystem] = None
    write_bandwidth: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)


class NetworkFabric(BaseModel):
    """
    Represents the inter-node network interconnect.
    
    Captures the topology, raw bandwidth, latency, and oversubscription ratio 
    of the cluster's switching fabric (e.g., InfiniBand NDR or Ethernet).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    topology: str = "fat-tree"
    bandwidth: Quantity
    latency: Optional[Quantity] = None
    oversubscription_ratio: float = 1.0
    metadata: Metadata = Field(default_factory=Metadata)

    @property
    def bandwidth_gbs(self) -> float:
        from ..core.constants import ureg
        from ..core.units import GB
        return float(self.bandwidth.to(GB / ureg.second).magnitude)

    @property
    def latency_us(self) -> float:
        if self.latency is None:
            return 0.0
        return float(self.latency.m_as("microsecond"))

class Node(BaseModel):
    """
    Represents a physical server chassis containing multiple accelerators.
    
    Essential for modeling the bandwidth gap between fast intra-node 
    communication (e.g., NVLink) and slower inter-node communication.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    accelerator: HardwareNode
    accelerators_per_node: int
    intra_node_bw: Quantity
    nics_per_node: int = 1
    psus_per_node: int = 2
    metadata: Metadata = Field(default_factory=Metadata)

class PodEnvelope(BaseModel):
    """Reference TPU/accelerator pod envelope (not a K8s Pod)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    chips: int
    memory: Quantity
    power: Quantity
    metadata: Metadata = Field(default_factory=Metadata)

class Fleet(BaseModel):
    """
    Layer D (Systems/Topology): The complete distributed cluster environment.
    
    A Fleet composes a `Node` configuration, a `NetworkFabric`, and a total 
    count of nodes. It can optionally be situated in a specific `Datacenter` 
    to enable unified calculations of distributed performance and sustainability.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    node: Node
    count: int
    fabric: NetworkFabric
    region: Optional[GridProfile] = None
    datacenter: Optional[Datacenter] = None
    mtbf_hours: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @property
    def total_accelerators(self) -> int:
        return self.count * self.node.accelerators_per_node

    @property
    def effective_pue(self) -> float:
        if self.datacenter:
            return self.datacenter.pue
        if self.region:
            return self.region.pue
        from ..infrastructure.registry import FacilityCooling
        return float(FacilityCooling.BestAir.pue)
