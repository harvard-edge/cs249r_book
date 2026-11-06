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
        """Sum the nominal bandwidth of identical local storage devices.

        Returns
        -------
        Quantity
            Aggregate data rate in the storage device bandwidth units.

        Notes
        -----
        Equation: BW_node = devices_per_node * device.bandwidth.

        Assumptions: all devices transfer concurrently and their bandwidths add without a shared bottleneck.

        Excludes: PCIe/root-complex limits, RAID overhead, workload imbalance, IOPS limits, and access latency.

        References
        ----------
        Derivation: add the independent equal device rates for an ideal striped transfer. This is a supply ceiling, not measured node throughput.

        Examples
        --------
        >>> from mlsysim.systems.types import NodeStorageConfig, StorageSubsystem
        >>> from mlsysim.core.units import Q_
        >>> device = StorageSubsystem(name='Drive', bandwidth=Q_('1 GB/s'))
        >>> storage = NodeStorageConfig(name='Pair', device=device, devices_per_node=2)
        >>> storage.aggregate_bandwidth == 2 * device.bandwidth
        True
        """
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
        """Express the fabric rate as a scalar in decimal gigabytes per second.

        Returns
        -------
        float
            Data rate in GB/s, where GB uses the package decimal-byte unit.

        Notes
        -----
        Equation: bandwidth_gbs = bandwidth converted to GB/s.

        Assumptions: bandwidth describes the intended fabric link or port; conversion preserves its existing direction convention.

        Excludes: oversubscription, protocol efficiency, topology, and conversion to GiB/s.

        References
        ----------
        Derivation: unit conversion changes only the representation of the stored data rate; it does not add a network model.

        Examples
        --------
        >>> from mlsysim import Systems
        >>> fabric = Systems.Fabrics.InfiniBand_NDR
        >>> fabric.bandwidth_gbs == fabric.bandwidth.m_as('GB/s')
        True
        """
        from ..core.units import ureg
        from ..core.units import GB
        return float(self.bandwidth.to(GB / ureg.second).magnitude)

    @property
    def latency_us(self) -> float:
        """Express the configured fabric latency in microseconds.

        Returns
        -------
        float
            Latency in microseconds; zero if latency is missing.

        Notes
        -----
        Equation: latency_us = latency converted to microsecond, or 0 when unset.

        Assumptions: missing latency is treated as zero for compatibility rather than inferred from topology or bandwidth.

        Excludes: queueing, serialization time, hop counts, and a measured latency distribution.

        References
        ----------
        Derivation: convert the stored time unit; the zero fallback is an explicit implementation convention for unspecified latency.

        Examples
        --------
        >>> from mlsysim.systems.types import NetworkFabric
        >>> from mlsysim.core.units import Q_
        >>> fabric = NetworkFabric(name='Unspecified latency', bandwidth=Q_('1 GB/s'))
        >>> fabric.latency_us == 0
        True
        """
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
    # PSU redundancy (added 2026-09-29). psu_redundancy is M in an N+M supply: the
    # PSUs that may fail with the node still at full performance. The default 0 makes
    # every PSU a single point of failure (the series model's historical behavior).
    psu_redundancy: int = 0
    psu_rating: Optional[Quantity] = None   # per-PSU output rating
    max_power: Optional[Quantity] = None    # maximum system power draw
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

    @field_validator("psu_rating", "max_power", mode="after")
    @classmethod
    def _validate_power_fields(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.watt, info.field_name)

    @model_validator(mode="after")
    def _validate_psu_redundancy(self):
        if not 0 <= self.psu_redundancy < self.psus_per_node:
            raise ValueError(
                f"psu_redundancy ({self.psu_redundancy}) must satisfy "
                f"0 <= psu_redundancy < psus_per_node ({self.psus_per_node})"
            )
        if self.psu_redundancy > 0 and self.psu_rating is not None and self.max_power is not None:
            n_required = self.psus_per_node - self.psu_redundancy
            if n_required * self.psu_rating < self.max_power:
                raise ValueError(
                    f"{n_required} required PSUs x {self.psu_rating} cannot carry max_power "
                    f"{self.max_power}; the {n_required}+{self.psu_redundancy} redundancy claim is inconsistent"
                )
        return self

    @property
    def series_psus(self) -> int:
        """Count power supplies treated as single points of failure.

        Returns
        -------
        int
            PSU count included in the simple series-failure-rate model.

        Notes
        -----
        Equation: N_series = 0 if psu_redundancy > 0, else psus_per_node.

        Assumptions: any declared hot-replaceable redundancy removes the first-order single-PSU failure term; the node validator checks count and stated power-rating consistency.

        Excludes: concurrent failures during repair, common-cause failure, redundancy controller failure, and a full N+M reliability calculation.

        References
        ----------
        Derivation: a series model counts components whose individual failure stops service. The redundant case drops that single-failure term rather than claiming zero outage risk.

        Examples
        --------
        >>> from mlsysim import Systems
        >>> node = Systems.Nodes.DGX_H100.model_copy(update={'psu_redundancy': 0})
        >>> node.series_psus == node.psus_per_node
        True
        """
        return 0 if self.psu_redundancy > 0 else self.psus_per_node


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
        """Return a complete breakdown of the non-accelerator power allocation.

        Returns
        -------
        dict or None
            Host, switch, network, conversion-loss, and cooling-allocation powers with units; None if any component is absent.

        Notes
        -----
        Equation: parts = {host, switch, network, conversion_loss, cooling_allocation}; sum(parts) must equal non_accelerator_power when all are present.

        Assumptions: each category describes a non-overlapping portion of the same rack accounting boundary. Model validation checks the complete sum.

        Excludes: imputation of absent categories, accelerator TDP, and dynamic utilization scaling.

        References
        ----------
        Derivation: this property assembles stated components, while the model validator compares their additive total with the stated aggregate.

        Examples
        --------
        >>> from mlsysim.systems.types import RackProfile
        >>> from mlsysim import Systems
        >>> rack = RackProfile(name='Partial', node=Systems.Nodes.DGX_H100, nodes_per_rack=1)
        >>> rack.power_components is None
        True
        """
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
        """Remove the stated cooling allocation from total rack power.

        Returns
        -------
        Quantity
            Rack IT power in power units; raises ValueError when cooling allocation is absent.

        Notes
        -----
        Equation: P_IT = total_power - cooling_allocation_power.

        Assumptions: the cooling allocation is included once in total_power; this accounting boundary separates IT from cooling.

        Excludes: inference of cooling from PUE, utility power upstream of the rack, and utilization-dependent power.

        References
        ----------
        Derivation: subtract the identified cooling component from the complete rack-power sum; the result depends on consistent component boundaries.

        Examples
        --------
        >>> from mlsysim.systems.types import RackProfile
        >>> from mlsysim import Systems
        >>> from mlsysim.core.units import Q_
        >>> rack = RackProfile(name='Example rack', node=Systems.Nodes.DGX_H100, nodes_per_rack=2, non_accelerator_power=Q_('1 kW'), cooling_allocation_power=Q_('0.2 kW'))
        >>> rack.it_power + rack.cooling_allocation_power == rack.total_power
        True
        """
        if self.cooling_allocation_power is None:
            raise ValueError(f"{self.name}: cooling_allocation_power is not set")
        return self.total_power - self.cooling_allocation_power

    @property
    def accelerator_count(self) -> int:
        """Count the accelerators installed in a homogeneous rack.

        Returns
        -------
        int
            Number of accelerator devices in the rack.

        Notes
        -----
        Equation: N_accelerators = nodes_per_rack * node.accelerators_per_node.

        Assumptions: every occupied node slot has the same accelerator configuration.

        Excludes: spare devices, failed devices, partially populated nodes, and scheduling availability.

        References
        ----------
        Derivation: multiply equal per-node device inventory by the rack node count.

        Examples
        --------
        >>> from mlsysim.systems.types import RackProfile
        >>> from mlsysim import Systems
        >>> from mlsysim.core.units import Q_
        >>> rack = RackProfile(name='Example rack', node=Systems.Nodes.DGX_H100, nodes_per_rack=2, non_accelerator_power=Q_('1 kW'), cooling_allocation_power=Q_('0.2 kW'))
        >>> rack.accelerator_count == rack.nodes_per_rack * rack.node.accelerators_per_node
        True
        """
        return self.nodes_per_rack * self.node.accelerators_per_node

    @property
    def accelerator_power(self) -> Quantity:
        """Sum accelerator TDP across the rack.

        Returns
        -------
        Quantity
            Aggregate accelerator power envelope; raises ValueError if device TDP is missing.

        Notes
        -----
        Equation: P_accelerators = accelerator_count * node.accelerator.tdp.

        Assumptions: all accelerators use the same stated TDP, summed concurrently as a design envelope.

        Excludes: host and network power, cooling, conversion losses, utilization, and measured workload power.

        References
        ----------
        Derivation: add the rated power of identical devices. TDP aggregation does not establish actual electrical draw during a workload.

        Examples
        --------
        >>> from mlsysim.systems.types import RackProfile
        >>> from mlsysim import Systems
        >>> from mlsysim.core.units import Q_
        >>> rack = RackProfile(name='Example rack', node=Systems.Nodes.DGX_H100, nodes_per_rack=2, non_accelerator_power=Q_('1 kW'), cooling_allocation_power=Q_('0.2 kW'))
        >>> rack.accelerator_power == rack.accelerator_count * rack.node.accelerator.tdp
        True
        """
        if self.node.accelerator.tdp is None:
            raise ValueError(f"{self.name}: accelerator '{self.node.accelerator.name}' has no tdp set")
        return self.accelerator_count * self.node.accelerator.tdp

    @property
    def total_power(self) -> Quantity:
        """Add non-accelerator allocation to the accelerator power envelope.

        Returns
        -------
        Quantity
            Rack power including the stated non-accelerator allocation, if any.

        Notes
        -----
        Equation: P_total = accelerator_power + non_accelerator_power; an absent allocation contributes zero.

        Assumptions: non-accelerator power is a rack total, not a per-node value, and does not overlap accelerator TDP.

        Excludes: unknown overhead when non_accelerator_power is absent, facility-wide PUE multiplication, and time-varying load.

        References
        ----------
        Derivation: add disjoint rack power categories; omitting a missing category is an implementation fallback, not evidence that the category consumes no power.

        Examples
        --------
        >>> from mlsysim.systems.types import RackProfile
        >>> from mlsysim import Systems
        >>> from mlsysim.core.units import Q_
        >>> rack = RackProfile(name='Example rack', node=Systems.Nodes.DGX_H100, nodes_per_rack=2, non_accelerator_power=Q_('1 kW'), cooling_allocation_power=Q_('0.2 kW'))
        >>> rack.total_power == rack.accelerator_power + rack.non_accelerator_power
        True
        """
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
        """Count accelerator devices across homogeneous fleet nodes.

        Returns
        -------
        int
            Total installed accelerator count.

        Notes
        -----
        Equation: N_accelerators = count * node.accelerators_per_node.

        Assumptions: all counted nodes have the declared accelerator configuration.

        Excludes: faults, spare inventory, job placement, and utilization.

        References
        ----------
        Derivation: multiply equal per-node inventory by the number of fleet nodes.

        Examples
        --------
        >>> from mlsysim import Systems
        >>> fleet = Systems.Clusters.Research_256
        >>> fleet.total_accelerators == fleet.count * fleet.node.accelerators_per_node
        True
        """
        return self.count * self.node.accelerators_per_node

    @property
    def effective_pue(self) -> float:
        """Select the facility-overhead factor for fleet energy accounting.

        Returns
        -------
        float
            Dimensionless PUE: total facility energy divided by IT energy.

        Notes
        -----
        Equation: PUE = datacenter.pue if present, else region.pue if present, else FacilityCooling.BestAir.pue.

        Assumptions: a named facility overrides a regional default; a fleet with neither uses the declared BestAir registry assumption.

        Excludes: dynamic cooling efficiency, per-rack differences, and deriving PUE from power measurements.

        References
        ----------
        Derivation: this is an ordered selection of contextual assumptions, not an additional physical equation. Consumers multiply IT energy by the selected PUE.

        Examples
        --------
        >>> from mlsysim import Systems, Infrastructure
        >>> fleet = Systems.Clusters.Research_256.model_copy(update={'datacenter': Infrastructure.Datacenters.Quebec_Hydro})
        >>> fleet.effective_pue == fleet.datacenter.pue
        True
        """
        if self.datacenter:
            return self.datacenter.pue
        if self.region:
            return self.region.pue
        from ..infrastructure.registry import FacilityCooling
        return float(FacilityCooling.BestAir.pue)
