from .types import Node, Fleet, NetworkFabric, PodEnvelope
from .reliability import Reliability
from .orchestration import Orchestration
from ..core.units import ureg, Q_, Gbps, TB, watt, MB
from ..core.provenance import sourced, sourced_qty
from ..hardware.registry import Hardware
from ..core.registry import Registry
from ..core.types import Metadata
from ..core import provenance_catalog as pc

# Fabric α latencies (appendix α–β model)
IB_NDR_LATENCY_US = sourced(5, pc.BOOK_FABRIC_LATENCY, name="InfiniBand NDR latency (μs)", description="One-way α latency for NDR fabrics.")
IB_HDR_LATENCY_US = sourced(7, pc.BOOK_FABRIC_LATENCY, name="InfiniBand HDR latency (μs)", description="One-way α latency for HDR fabrics.")
ROCE_LATENCY_US = sourced(10, pc.BOOK_FABRIC_LATENCY, name="RoCE latency (μs)", description="One-way α latency for RoCE v2.")
TCP_LATENCY_US = sourced(50, pc.BOOK_FABRIC_LATENCY, name="TCP latency (μs)", description="One-way α latency for TCP over Ethernet.")

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
        intra_node_bw=Hardware.Cloud.H100.nvlink.bandwidth,
        nics_per_node=8,
        metadata=Metadata(provenance=pc.BOOK_DGX_GPUS_PER_HOST),
    )
    DGX_A100 = Node(
        name="DGX A100",
        accelerator=Hardware.Cloud.A100,
        accelerators_per_node=8,
        intra_node_bw=Hardware.Cloud.A100.nvlink.bandwidth,
        nics_per_node=8,
    )
    DGX_B200 = Node(
        name="DGX B200",
        accelerator=Hardware.Cloud.B200,
        accelerators_per_node=8,
        intra_node_bw=Hardware.Cloud.B200.nvlink.bandwidth,
        nics_per_node=8,
    )


class Fabrics(Registry):
    """Vetted network fabrics (canonical bandwidth + latency)."""
    Ethernet_10G = NetworkFabric(
        name="10GbE", bandwidth=10 * Gbps, latency=Q_(TCP_LATENCY_US, "us"),
        metadata=Metadata(provenance=pc.BOOK_FABRIC_LATENCY),
    )
    Ethernet_100G = NetworkFabric(
        name="100GbE", bandwidth=100 * Gbps, latency=Q_(TCP_LATENCY_US, "us"),
        metadata=Metadata(provenance=pc.BOOK_FABRIC_LATENCY),
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


_TIER_PROV = pc.BOOK_CLUSTER_TIERS


class Clusters(Registry):
    """Book reference fleet tiers (256 / 1k / 2k / 8k / 10k / 100k GPUs)."""
    Research_256 = Fleet(
        name="Research Cluster (256 GPUs)",
        node=Nodes.DGX_H100,
        count=32,
        fabric=Fabrics.Ethernet_100G,
        metadata=Metadata(provenance=_TIER_PROV, description="Small cluster tier (256 GPUs)."),
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
        memory=131 * TB,
        power=3 * ureg.megawatt,
        metadata=Metadata(
            provenance=pc.BOOK_CLUSTER_TIERS,
            description="Reference TPU pod envelope for Volume I cloud-scale examples.",
        ),
    )



class SwitchFabric(Registry):
    """Switch-ASIC capacity, 400G optics power, and α-β / FEC / hop latency reference
    figures for the network-fabrics worked examples (Volume II)."""

    SwitchAsic51T2 = sourced_qty(51_200 * Gbps, pc.BOOK_SWITCH_OPTICS,
        name="Switch ASIC 51.2T", description="Single-ASIC switch capacity (51.2 Tb/s class).")
    SwitchAsic102T4 = sourced_qty(102_400 * Gbps, pc.BOOK_SWITCH_OPTICS,
        name="Switch ASIC 102.4T", description="Single-ASIC switch capacity (102.4 Tb/s class).")
    OpticsPluggable400G = sourced_qty(20 * watt, pc.BOOK_SWITCH_OPTICS,
        name="400G pluggable optics power", description="Per-module power for a 400G pluggable transceiver.")
    OpticsCpo400G = sourced_qty(10 * watt, pc.BOOK_SWITCH_OPTICS,
        name="400G co-packaged optics power", description="Per-port power for 400G co-packaged optics.")
    # α-β model latency anchors. NOTE: AlphaNdr (1.5 µs) is the per-message base latency used in
    # the ring-allreduce worked example; it is DISTINCT from the end-to-end
    # Fabrics.InfiniBand_NDR.latency (5 µs). Both values are preserved as-is here — reconciling
    # which is canonical is a separate editorial decision, not a taxonomy move.
    AlphaNdr = sourced_qty(1.5 * ureg.microsecond, pc.BOOK_FABRIC_LATENCY,
        name="α (NDR base latency)", description="Per-message base latency for the NDR α-β model worked example.")
    AlphaRoce = sourced_qty(5.0 * ureg.microsecond, pc.BOOK_FABRIC_LATENCY,
        name="α (RoCE base latency)", description="Per-message base latency for the RoCE α-β model worked example.")
    HopLatency = sourced_qty(0.6 * ureg.microsecond, pc.BOOK_FABRIC_LATENCY,
        name="Per-hop switch latency", description="Approximate per-hop store-and-forward switch latency.")
    FecLatencyLow = sourced_qty(100 * ureg.nanosecond, pc.BOOK_FABRIC_LATENCY,
        name="FEC latency (low)", description="Lower-bound forward-error-correction added latency.")
    FecLatencyHigh = sourced_qty(200 * ureg.nanosecond, pc.BOOK_FABRIC_LATENCY,
        name="FEC latency (high)", description="Upper-bound forward-error-correction added latency.")


class NetworkEnergy(Registry):
    """Network data-transfer energy anchors (order-of-magnitude intuition)."""

    Per5gMb = sourced_qty(100 * ureg.millijoule / MB, pc.BOOK_NETWORK_ENERGY,
        name="5G transfer energy per MB", description="Approximate radio-access energy to move 1 MB over 5G.")
    Per1Kb = sourced_qty(1_000_000 * ureg.picojoule, pc.BOOK_NETWORK_ENERGY,
        name="Network energy per KB", description="Approximate energy to move 1 KB across a datacenter network (~1 µJ).")


class Systems(Registry):
    """Registry namespace for Systems."""
    Nodes = Nodes
    Clusters = Clusters
    Fabrics = Fabrics
    SwitchFabric = SwitchFabric
    NetworkEnergy = NetworkEnergy
    Pods = Pods
    Reliability = Reliability
    Orchestration = Orchestration()
