from .types import (
    CheckpointStoragePath,
    Fleet,
    NetworkFabric,
    Node,
    NodeStorageConfig,
    PodEnvelope,
    RackProfile,
    StorageSubsystem,
)
from .reliability import Reliability
from .orchestration import Orchestration as OrchestrationProfile
from ..core.units import ureg, Q_, Gbps, GB, TB, TiB, watt, MB, kilowatt, pJ, bit
from ..core.provenance import sourced, sourced_qty
from ..hardware.registry import Hardware
from ..core.registry import Registry
from ..core.types import Metadata
from ..core import provenance_catalog as pc

# Fabric α latencies for α-β communication models.
IB_NDR_LATENCY_US = sourced(5, pc.FABRIC_LATENCY_ASSUMPTIONS, name="InfiniBand NDR latency (μs)", description="One-way α latency for NDR fabrics.")
IB_HDR_LATENCY_US = sourced(7, pc.FABRIC_LATENCY_ASSUMPTIONS, name="InfiniBand HDR latency (μs)", description="One-way α latency for HDR fabrics.")
ROCE_LATENCY_US = sourced(10, pc.FABRIC_LATENCY_ASSUMPTIONS, name="RoCE latency (μs)", description="One-way α latency for RoCE v2.")
TCP_LATENCY_US = sourced(50, pc.FABRIC_LATENCY_ASSUMPTIONS, name="TCP latency (μs)", description="One-way α latency for TCP over Ethernet.")

_INFINIBAND_HDR_BW = 200 * Gbps
_INFINIBAND_NDR_BW = 400 * Gbps
_INFINIBAND_XDR_BW = 800 * Gbps
_INFINIBAND_GXDR_BW = 1600 * Gbps

_FABRIC_PROV = {
    "hdr": pc.INFINIBAND_HDR_GBS,
    "ndr": pc.INFINIBAND_NDR_GBS,
    "xdr": pc.INFINIBAND_XDR_GBS,
    "eth400": pc.ETHERNET_400G_GBS,
    "eth800": pc.ETHERNET_800G_GBS,
    "roce": pc.ROCE_100G_GBS,
}


class Nodes(Registry):
    """Vetted reference nodes."""
    DGX_H100 = Node(
        name="DGX H100",
        accelerator=Hardware.Cloud.H100,
        accelerators_per_node=8,
        intra_node_bw=Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        nics_per_node=8,
        host_memory=2 * TiB,
        metadata=Metadata(provenance=pc.DGX_GPUS_PER_HOST),
    )
    DGX_A100 = Node(
        name="DGX A100",
        accelerator=Hardware.Cloud.A100,
        accelerators_per_node=8,
        intra_node_bw=Hardware.Cloud.A100.nvlink.bandwidth_per_direction,
        nics_per_node=8,
        metadata=Metadata(provenance=pc.DGX_GPUS_PER_HOST),
    )
    DGX_B200 = Node(
        name="DGX B200",
        accelerator=Hardware.Cloud.B200,
        accelerators_per_node=8,
        intra_node_bw=Hardware.Cloud.B200.nvlink.bandwidth_per_direction,
        nics_per_node=8,
        metadata=Metadata(provenance=pc.DGX_GPUS_PER_HOST),
    )
    Kempner_H100_4GPU = Node(
        name="Kempner H100 4-GPU Server",
        accelerator=Hardware.Cloud.H100,
        accelerators_per_node=4,
        intra_node_bw=Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        nics_per_node=4,
        metadata=Metadata(
            provenance=pc.KEMPNER_AI_CLUSTER_H100,
            description="Kempner H100 partition server envelope: four H100 80GB GPUs per server.",
        ),
    )


class Fabrics(Registry):
    """Vetted network fabrics (canonical bandwidth + latency)."""
    Ethernet_10G = NetworkFabric(
        name="10GbE", bandwidth=10 * Gbps, latency=Q_(TCP_LATENCY_US, "us"),
        metadata=Metadata(provenance=pc.FABRIC_LATENCY_ASSUMPTIONS),
    )
    Ethernet_100G = NetworkFabric(
        name="100GbE", bandwidth=100 * Gbps, latency=Q_(TCP_LATENCY_US, "us"),
        metadata=Metadata(provenance=pc.FABRIC_LATENCY_ASSUMPTIONS),
    )
    Ethernet_400G = NetworkFabric(
        name="400GbE", bandwidth=400 * Gbps, latency=Q_(TCP_LATENCY_US, "us"),
        metadata=Metadata(provenance=_FABRIC_PROV["eth400"]),
    )
    Ethernet_800G = NetworkFabric(
        name="800GbE", bandwidth=800 * Gbps, latency=Q_(TCP_LATENCY_US, "us"),
        metadata=Metadata(provenance=_FABRIC_PROV["eth800"]),
    )
    Ethernet_1P6T = NetworkFabric(
        name="1.6TbE", bandwidth=1600 * Gbps, latency=Q_(TCP_LATENCY_US, "us"),
        metadata=Metadata(provenance=_FABRIC_PROV["eth800"]),
    )
    RoCE_100G = NetworkFabric(
        name="100GbE RoCE", bandwidth=100 * Gbps, latency=Q_(ROCE_LATENCY_US, "us"),
        metadata=Metadata(provenance=_FABRIC_PROV["roce"]),
    )
    InfiniBand_HDR = NetworkFabric(
        name="IB HDR", bandwidth=_INFINIBAND_HDR_BW, latency=Q_(IB_HDR_LATENCY_US, "us"),
        metadata=Metadata(provenance=_FABRIC_PROV["hdr"]),
    )
    InfiniBand_NDR = NetworkFabric(
        name="IB NDR", bandwidth=_INFINIBAND_NDR_BW, latency=Q_(IB_NDR_LATENCY_US, "us"),
        metadata=Metadata(provenance=_FABRIC_PROV["ndr"]),
    )
    InfiniBand_XDR = NetworkFabric(
        name="IB XDR", bandwidth=_INFINIBAND_XDR_BW, latency=Q_(IB_NDR_LATENCY_US, "us"),
        metadata=Metadata(provenance=_FABRIC_PROV["xdr"]),
    )
    InfiniBand_GXDR = NetworkFabric(
        name="IB GXDR", bandwidth=_INFINIBAND_GXDR_BW, latency=Q_(IB_NDR_LATENCY_US, "us"),
        metadata=Metadata(provenance=_FABRIC_PROV["xdr"]),
    )


_TIER_PROV = pc.CLUSTER_TIER_CONVENTIONS


class Clusters(Registry):
    """Reference fleet tiers (256 / 1k / 2k / 8k / 10k / 100k GPUs)."""
    Research_256 = Fleet(
        name="Research Cluster (256 GPUs)",
        node=Nodes.DGX_H100,
        count=32,
        fabric=Fabrics.Ethernet_100G,
        metadata=Metadata(provenance=_TIER_PROV, description="Small cluster tier (256 GPUs)."),
    )
    Lab_64_H100 = Fleet(
        name="Lab Cluster (64 H100 GPUs)",
        node=Nodes.DGX_H100,
        count=8,
        fabric=Fabrics.InfiniBand_HDR,
        metadata=Metadata(provenance=_TIER_PROV, description="Small lab tier (64 H100 GPUs)."),
    )
    Training_512_H100 = Fleet(
        name="Training Cluster (512 H100 GPUs)",
        node=Nodes.DGX_H100,
        count=64,
        fabric=Fabrics.InfiniBand_HDR,
        metadata=Metadata(provenance=_TIER_PROV, description="Mid-small training tier (512 H100 GPUs)."),
    )
    Production_2K = Fleet(
        name="Production Cluster (2048 GPUs)",
        node=Nodes.DGX_H100,
        count=256,
        fabric=Fabrics.InfiniBand_HDR,
        metadata=Metadata(provenance=_TIER_PROV, description="Medium cluster tier (2048 GPUs)."),
    )
    Training_1K = Fleet(
        name="Training Cluster (1024 GPUs)",
        node=Nodes.DGX_H100,
        count=128,
        fabric=Fabrics.InfiniBand_HDR,
        metadata=Metadata(provenance=_TIER_PROV, description="Mid cluster tier (1024 GPUs)."),
    )
    Training_1K_A100 = Fleet(
        name="Training Cluster (1024 A100 GPUs)",
        node=Nodes.DGX_A100,
        count=128,
        fabric=Fabrics.InfiniBand_HDR,
        metadata=Metadata(
            provenance=_TIER_PROV,
            description="A100/HDR mid cluster tier (1024 GPUs).",
        ),
    )
    Frontier_8K = Fleet(
        name="Frontier Cluster (8192 GPUs)",
        node=Nodes.DGX_H100,
        count=1024,
        fabric=Fabrics.InfiniBand_NDR,
        metadata=Metadata(provenance=_TIER_PROV, description="Large cluster tier (8192 GPUs)."),
    )
    Training_10K = Fleet(
        name="Training Cluster (10000 GPUs)",
        node=Nodes.DGX_H100,
        count=1250,
        fabric=Fabrics.InfiniBand_NDR,
        metadata=Metadata(provenance=_TIER_PROV, description="Large training tier (10000 GPUs)."),
    )
    Reference_25K_H100 = Fleet(
        name="Reference H100 Cluster (25000 GPUs)",
        node=Nodes.DGX_H100,
        count=3125,
        fabric=Fabrics.InfiniBand_NDR,
        metadata=Metadata(
            provenance=_TIER_PROV,
            description="Round-number reference H100 fleet tier (25000 GPUs).",
        ),
    )
    Kempner_H100_384 = Fleet(
        name="Kempner AI Cluster H100 Partition (384 H100 GPUs)",
        node=Nodes.Kempner_H100_4GPU,
        count=96,
        fabric=Fabrics.InfiniBand_NDR,
        metadata=Metadata(
            provenance=pc.KEMPNER_AI_CLUSTER_H100,
            description="Published Kempner Institute H100 partition before the 2026 heterogeneous expansion.",
        ),
    )
    Mega_100K = Fleet(
        name="Mega Cluster (100000 GPUs)",
        node=Nodes.DGX_H100,
        count=12500,
        fabric=Fabrics.InfiniBand_NDR,
        metadata=Metadata(provenance=_TIER_PROV, description="Mega cluster tier (100k GPUs)."),
    )


class Pods(Registry):
    """Reference accelerator pod envelopes (not K8s runtime)."""
    TPUv4_4096 = PodEnvelope(
        name="Cloud TPU v4 Pod (4096 chips)",
        chips=4096,
        memory=128 * TiB,
        power=3 * ureg.megawatt,
        metadata=Metadata(
            provenance=pc.CLUSTER_TIER_CONVENTIONS,
            description="Reference TPU pod envelope for cloud-scale system models.",
        ),
    )


class Racks(Registry):
    """Reusable physical rack profiles."""

    DGX_H100_4Node = RackProfile(
        name="DGX H100 4-node rack",
        node=Nodes.DGX_H100,
        nodes_per_rack=4,
        non_accelerator_power=11.1 * kilowatt,
        metadata=Metadata(
            provenance=pc.DGX_H100_RACK_REFERENCE,
            description="Reference 32-H100 rack profile for rack-level power and cooling models.",
        ),
    )



class SwitchFabric(Registry):
    """Switch-ASIC capacity, 400G optics power, and α-β / FEC / hop latency reference
    figures for datacenter network-fabric models."""

    NdrSwitchPorts = sourced(64, pc.NVIDIA_QUANTUM2_QM97XX_SWITCH,
        name="NDR switch ports", description="Reference 64-port NDR switch used for two-tier fabric sizing.")
    NdrLeafDownlinkPorts = sourced(32, pc.NDR_LEAF_SPINE_64_PORT_SPLIT,
        name="NDR leaf downlink ports", description="Reference downlinks per 64-port leaf switch.")
    NdrLeafUplinkPorts = sourced(32, pc.NDR_LEAF_SPINE_64_PORT_SPLIT,
        name="NDR leaf uplink ports", description="Reference uplinks per 64-port leaf switch.")
    SwitchAsic51T2 = sourced_qty(51_200 * Gbps, pc.NVIDIA_QUANTUM2_QM97XX_SWITCH,
        name="Switch ASIC 51.2T", description="Single-ASIC switch capacity (51.2 Tb/s class).")
    SwitchAsic102T4 = sourced_qty(102_400 * Gbps, pc.SWITCH_OPTICS_REFERENCE,
        name="Switch ASIC 102.4T", description="Single-ASIC switch capacity (102.4 Tb/s class).")
    OpticsPluggable400G = sourced_qty(20 * watt, pc.SWITCH_OPTICS_REFERENCE,
        name="400G pluggable optics power", description="Per-module power for a 400G pluggable transceiver.")
    OpticsCpo400G = sourced_qty(10 * watt, pc.SWITCH_OPTICS_REFERENCE,
        name="400G co-packaged optics power", description="Per-port power for 400G co-packaged optics.")
    # α-β model latency anchors. NOTE: AlphaNdr (1.5 µs) is the per-message base latency used in
    # the ring-allreduce model; it is DISTINCT from the end-to-end
    # Fabrics.InfiniBand_NDR.latency (5 µs). Both values are preserved as-is here — reconciling
    # which is canonical is a separate editorial decision, not a taxonomy move.
    AlphaNdr = sourced_qty(1.5 * ureg.microsecond, pc.FABRIC_LATENCY_ASSUMPTIONS,
        name="α (NDR base latency)", description="Per-message base latency for the NDR α-β model.")
    AlphaRoce = sourced_qty(5.0 * ureg.microsecond, pc.FABRIC_LATENCY_ASSUMPTIONS,
        name="α (RoCE base latency)", description="Per-message base latency for the RoCE α-β model.")
    HopLatency = sourced_qty(0.6 * ureg.microsecond, pc.FABRIC_LATENCY_ASSUMPTIONS,
        name="Per-hop switch latency", description="Approximate per-hop store-and-forward switch latency.")
    FecLatencyLow = sourced_qty(100 * ureg.nanosecond, pc.FABRIC_LATENCY_ASSUMPTIONS,
        name="FEC latency (low)", description="Lower-bound forward-error-correction added latency.")
    FecLatencyHigh = sourced_qty(200 * ureg.nanosecond, pc.FABRIC_LATENCY_ASSUMPTIONS,
        name="FEC latency (high)", description="Upper-bound forward-error-correction added latency.")


class NetworkEnergy(Registry):
    """Network data-transfer energy anchors (order-of-magnitude intuition)."""

    Per5gMb = sourced_qty(100 * ureg.millijoule / MB, pc.NETWORK_ENERGY_ANCHORS,
        name="5G transfer energy per MB", description="Approximate radio-access energy to move 1 MB over 5G.")
    # (2026-06-10 audit: Per1Kb deleted — zero consumers, and its 1,000
    # pJ/byte contradicted Hardware.Tech.Movement.Network's canonical
    # 10,000 pJ/byte 10x. Datacenter-network movement energy lives in
    # Hardware.Tech.Movement; do not re-add a competing anchor here.)
    NvlinkEnergyPerBit = sourced_qty(7.5 * pJ / bit, pc.NETWORK_ENERGY_ANCHORS,
        name="NVLink energy per bit", description="Representative midpoint of a 5--10 pJ/bit NVLink transfer envelope.")
    InfiniBandEnergyPerBit = sourced_qty(35 * pJ / bit, pc.NETWORK_ENERGY_ANCHORS,
        name="InfiniBand energy per bit", description="Representative midpoint of a 20--50 pJ/bit switched-network transfer envelope.")


class Storage(Registry):
    """Reusable storage-system profiles for data and checkpoint paths."""

    TraditionalGpuIoLatency = sourced_qty(
        120 * ureg.microsecond,
        pc.STORAGE_ACCESS_PATH_REFERENCE,
        name="Traditional GPU storage path latency",
        description="Reference per-transfer latency for CPU-mediated storage-to-GPU I/O.",
    )
    GpuDirectStorageLatency = sourced_qty(
        30 * ureg.microsecond,
        pc.STORAGE_ACCESS_PATH_REFERENCE,
        name="GPU Direct Storage path latency",
        description="Reference per-transfer latency for GPU Direct Storage bypass I/O.",
    )

    LocalNvmeGen4 = StorageSubsystem(
        name="Local NVMe SSD (Gen4)",
        storage_tech=Hardware.Tech.Storage.NvmeGen4,
        bandwidth=Hardware.Tech.Storage.NvmeGen4.bandwidth,
        latency=Hardware.Tech.Storage.NvmeGen4.latency,
        media="NVMe SSD",
        interface="PCIe Gen4",
        durability="local",
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    LocalNvmeGen3 = StorageSubsystem(
        name="Local NVMe SSD (Gen3)",
        storage_tech=Hardware.Tech.Storage.NvmeGen3,
        bandwidth=Hardware.Tech.Storage.NvmeGen3.bandwidth,
        iops=500_000,
        media="NVMe SSD",
        interface="PCIe Gen3",
        durability="local",
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    SataSsd = StorageSubsystem(
        name="SATA SSD",
        bandwidth=550 * (MB / ureg.second),
        iops=10_000,
        media="SATA SSD",
        interface="SATA",
        durability="local",
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    SataSsdEffective = StorageSubsystem(
        name="SATA SSD (effective sequential)",
        bandwidth=500 * (MB / ureg.second),
        media="SATA SSD",
        interface="SATA",
        durability="local",
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    Hdd7200Rpm = StorageSubsystem(
        name="7.2k RPM HDD",
        bandwidth=150 * (MB / ureg.second),
        iops=100,
        media="HDD",
        interface="SATA",
        durability="local",
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    CloudBlockStorageBaseline = StorageSubsystem(
        name="Cloud block storage baseline",
        bandwidth=250 * (MB / ureg.second),
        media="cloud block volume",
        durability="durable",
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    ObjectStoreSingleStream = StorageSubsystem(
        name="Object storage single-stream read",
        bandwidth=100 * (MB / ureg.second),
        media="object storage",
        interface="HTTP range request",
        durability="durable",
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    LocalNvmeTrainingLow = StorageSubsystem(
        name="Local NVMe training tier (low)",
        bandwidth=3 * (GB / ureg.second),
        media="NVMe SSD",
        interface="PCIe",
        durability="local",
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    LocalNvmeTrainingTypical = StorageSubsystem(
        name="Local NVMe training tier (typical)",
        bandwidth=5 * (GB / ureg.second),
        media="NVMe SSD",
        interface="PCIe",
        durability="local",
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    LocalNvmeGen4x4 = NodeStorageConfig(
        name="4x local Gen4 NVMe drives per node",
        device=LocalNvmeGen4,
        devices_per_node=4,
        metadata=Metadata(provenance=pc.STORAGE_TIER_CONVENTIONS),
    )
    PfsOneTbPerSecond = StorageSubsystem(
        name="Reference parallel file system (1 TB/s)",
        bandwidth=1 * (TB / ureg.second),
        media="parallel file system",
        durability="durable",
        metadata=Metadata(
            provenance=pc.STORAGE_TIER_CONVENTIONS,
            description="Reference aggregate PFS bandwidth for checkpoint-staging examples.",
        ),
    )
    Production2KCheckpointPath = CheckpointStoragePath(
        name="Production 2K checkpoint path",
        local_stage=LocalNvmeGen4x4,
        durable_store=PfsOneTbPerSecond,
        write_bandwidth=PfsOneTbPerSecond.bandwidth,
        metadata=Metadata(
            provenance=pc.STORAGE_TIER_CONVENTIONS,
            description="Reference local-NVMe-to-PFS checkpoint path for a 2K H100 fleet.",
        ),
    )


class Systems(Registry):
    """Registry namespace for Systems."""
    Nodes = Nodes
    Clusters = Clusters
    Fabrics = Fabrics
    Storage = Storage
    SwitchFabric = SwitchFabric
    NetworkEnergy = NetworkEnergy
    Pods = Pods
    Racks = Racks
    Reliability = Reliability
    Orchestration = OrchestrationProfile(
        metadata=Metadata(provenance=pc.ORCHESTRATION_ASSUMPTIONS)
    )
