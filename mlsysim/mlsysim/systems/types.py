from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Any, Optional
from ..hardware.types import HardwareNode
from ..infrastructure.types import Datacenter, GridProfile
from ..core.units import ureg
from ..core.types import (
    Quantity,
    Metadata,
    require_dimensionality,
    require_unit_family,
)

class DeploymentTier(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    ram: Quantity
    storage: Quantity
    typical_latency_budget: Quantity

    @field_validator("ram", "storage", mode="after")
    @classmethod
    def _validate_capacity_fields(cls, v, info):
        return require_unit_family(v, ureg.byte, info.field_name, "data")

    @field_validator("typical_latency_budget", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        return require_dimensionality(v, ureg.second, "typical_latency_budget")


class StorageSubsystem(BaseModel):
    """Reusable storage tier or service in a system design."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
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

    @field_validator("capacity", mode="after")
    @classmethod
    def _validate_capacity(cls, v):
        return require_unit_family(v, ureg.byte, "capacity", "data")

    @field_validator("bandwidth", mode="after")
    @classmethod
    def _validate_bandwidth(cls, v):
        return require_unit_family(v, ureg.bit / ureg.second, "bandwidth", "data")

    @field_validator("latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        return require_dimensionality(v, ureg.second, "latency")


class NodeStorageConfig(BaseModel):
    """Per-node storage composition, such as four local NVMe drives."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    device: StorageSubsystem
    devices_per_node: int = 1
    metadata: Metadata = Field(default_factory=Metadata)

    @property
    def aggregate_bandwidth(self) -> Quantity:
        return self.devices_per_node * self.device.bandwidth


class CheckpointStoragePath(BaseModel):
    """Local staging plus durable checkpoint destination for training fleets."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    local_stage: Optional[NodeStorageConfig] = None
    durable_store: Optional[StorageSubsystem] = None
    write_bandwidth: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("write_bandwidth", mode="after")
    @classmethod
    def _validate_write_bandwidth(cls, v):
        return require_unit_family(v, ureg.bit / ureg.second, "write_bandwidth", "data")


class NetworkFabric(BaseModel):
    """
    Represents the inter-node network interconnect.
    
    Captures the topology, raw bandwidth, latency, and oversubscription ratio 
    of the cluster's switching fabric (e.g., InfiniBand NDR or Ethernet).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    topology: str = "fat-tree"
    bandwidth: Quantity
    latency: Optional[Quantity] = None
    oversubscription_ratio: float = 1.0
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("bandwidth", mode="after")
    @classmethod
    def _validate_bandwidth(cls, v):
        return require_unit_family(v, ureg.bit / ureg.second, "bandwidth", "data")

    @field_validator("latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        return require_dimensionality(v, ureg.second, "latency")

    @property
    def bandwidth_gbs(self) -> float:
        from ..core.units import ureg
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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    accelerator: HardwareNode
    accelerators_per_node: int
    intra_node_bw: Quantity
    nics_per_node: int = 1
    psus_per_node: int = 2
    host_memory: Optional[Quantity] = None
    host_cpu: Optional[str] = None
    host_cpu_cores: Optional[int] = None
    host_memory_bw: Optional[Quantity] = None
    host_storage: Optional[Any] = None
    # Physical composition (added 2026-09-28; optional because only some node
    # datasheets publish them): NVSwitch chips on the baseboard, NVLink links
    # per accelerator, and chassis mass.
    nvswitch_count: Optional[int] = None
    nvlink_links_per_accelerator: Optional[int] = None
    mass: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("mass", mode="after")
    @classmethod
    def _validate_mass(cls, v):
        return require_dimensionality(v, ureg.kilogram, "mass")

    @field_validator("intra_node_bw", mode="after")
    @classmethod
    def _validate_intra_node_bw(cls, v):
        return require_unit_family(v, ureg.bit / ureg.second, "intra_node_bw", "data")

    @field_validator("host_memory", mode="after")
    @classmethod
    def _validate_host_memory(cls, v):
        if v is None:
            return v
        return require_unit_family(v, ureg.byte, "host_memory", "data")

    @field_validator("host_memory_bw", mode="after")
    @classmethod
    def _validate_host_memory_bw(cls, v):
        if v is None:
            return v
        return require_unit_family(v, ureg.bit / ureg.second, "host_memory_bw", "data")


class RackProfile(BaseModel):
    """Physical rack composition for node-level system profiles."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    node: Node
    nodes_per_rack: int
    non_accelerator_power: Optional[Quantity] = None
    # Optional breakdown of non_accelerator_power (added 2026-09-28). When all
    # five are set, their sum must equal non_accelerator_power, so a table that
    # explains the aggregate cannot drift from it.
    host_power: Optional[Quantity] = None
    switch_power: Optional[Quantity] = None
    network_power: Optional[Quantity] = None
    conversion_loss_power: Optional[Quantity] = None
    cooling_allocation_power: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator(
        "non_accelerator_power", "host_power", "switch_power", "network_power",
        "conversion_loss_power", "cooling_allocation_power", mode="after",
    )
    @classmethod
    def _validate_power_fields(cls, v, info):
        return require_dimensionality(v, ureg.watt, info.field_name)

    @model_validator(mode="after")
    def _components_sum_to_aggregate(self):
        parts = self.power_components
        if parts is not None:
            if self.non_accelerator_power is None:
                raise ValueError(f"{self.name}: power components set without non_accelerator_power")
            total = sum(q.m_as("watt") for q in parts.values())
            if abs(total - self.non_accelerator_power.m_as("watt")) > 1e-6:
                raise ValueError(
                    f"{self.name}: power components sum to {total} W, "
                    f"not non_accelerator_power {self.non_accelerator_power.m_as('watt')} W"
                )
        return self

    @property
    def power_components(self) -> Optional[dict]:
        """The five-way breakdown of non_accelerator_power, or None if any part is unset."""
        parts = {
            "host": self.host_power,
            "switch": self.switch_power,
            "network": self.network_power,
            "conversion_loss": self.conversion_loss_power,
            "cooling_allocation": self.cooling_allocation_power,
        }
        if any(v is None for v in parts.values()):
            return None
        return parts

    @property
    def it_power(self) -> Quantity:
        """Rack IT power: total_power minus the cooling allocation (requires the breakdown)."""
        if self.cooling_allocation_power is None:
            raise ValueError(f"{self.name}: cooling_allocation_power is not set")
        return self.total_power - self.cooling_allocation_power

    @property
    def accelerator_count(self) -> int:
        return self.nodes_per_rack * self.node.accelerators_per_node

    @property
    def accelerator_power(self) -> Quantity:
        return self.accelerator_count * self.node.accelerator.tdp

    @property
    def total_power(self) -> Quantity:
        if self.non_accelerator_power is None:
            return self.accelerator_power
        return self.accelerator_power + self.non_accelerator_power


class PodEnvelope(BaseModel):
    """Reference TPU/accelerator pod envelope (not a K8s Pod)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    chips: int
    memory: Quantity
    power: Quantity
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("memory", mode="after")
    @classmethod
    def _validate_memory(cls, v):
        return require_unit_family(v, ureg.byte, "memory", "data")

    @field_validator("power", mode="after")
    @classmethod
    def _validate_power(cls, v):
        return require_dimensionality(v, ureg.watt, "power")

class Fleet(BaseModel):
    """
    Layer D (Systems/Topology): The complete distributed cluster environment.
    
    A Fleet composes a `Node` configuration, a `NetworkFabric`, and a total 
    count of nodes. It can optionally be situated in a specific `Datacenter` 
    to enable unified calculations of distributed performance and sustainability.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    node: Node
    count: int
    fabric: NetworkFabric
    region: Optional[GridProfile] = None
    datacenter: Optional[Datacenter] = None
    mtbf_hours: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("mtbf_hours", mode="after")
    @classmethod
    def _validate_mtbf(cls, v):
        return require_dimensionality(v, ureg.hour, "mtbf_hours")

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
