from .types import DeploymentTier, Node, Fleet, NetworkFabric
from ..core.constants import (
    ureg, Q_,
    SMARTPHONE_RAM_GB, MCU_RAM_KIB, INFINIBAND_NDR_BW, INFINIBAND_HDR_BW, NETWORK_10G_BW, NETWORK_100G_BW,
    IB_NDR_LATENCY_US, IB_HDR_LATENCY_US, TCP_LATENCY_US
)
from ..hardware.registry import Hardware

class Tiers:
    """Vetted Deployment Tiers."""
    Cloud = DeploymentTier(
        name="Cloud",
        ram=512 * ureg.GB,
        storage=10 * ureg.TB,
        typical_latency_budget=200 * ureg.ms
    )
    Edge = DeploymentTier(
        name="Edge",
        ram=32 * ureg.GB,
        storage=1 * ureg.TB,
        typical_latency_budget=50 * ureg.ms
    )
    Mobile = DeploymentTier(
        name="Mobile",
        ram=SMARTPHONE_RAM_GB,
        storage=256 * ureg.GB,
        typical_latency_budget=30 * ureg.ms
    )
    Tiny = DeploymentTier(
        name="TinyML",
        ram=MCU_RAM_KIB,
        storage=4 * ureg.MB,
        typical_latency_budget=100 * ureg.ms
    )

class Nodes:
    """Vetted Reference Nodes."""
    DGX_H100 = Node(
        name="DGX H100",
        accelerator=Hardware.H100,
        accelerators_per_node=8,
        intra_node_bw=900 * ureg.GB / ureg.second,
        nics_per_node=8
    )
    DGX_A100 = Node(
        name="DGX A100",
        accelerator=Hardware.A100,
        accelerators_per_node=8,
        intra_node_bw=600 * ureg.GB / ureg.second,
        nics_per_node=8
    )
    DGX_B200 = Node(
        name="DGX B200",
        accelerator=Hardware.B200,
        accelerators_per_node=8,
        intra_node_bw=1800 * ureg.GB / ureg.second,
        nics_per_node=8
    )

class Fabrics:
    """Vetted Network Fabrics."""
    Ethernet_10G = NetworkFabric(name="10GbE", bandwidth=NETWORK_10G_BW, latency=Q_(TCP_LATENCY_US, "us"))
    Ethernet_100G = NetworkFabric(name="100GbE", bandwidth=NETWORK_100G_BW, latency=Q_(TCP_LATENCY_US, "us"))
    InfiniBand_HDR = NetworkFabric(name="IB HDR", bandwidth=INFINIBAND_HDR_BW, latency=Q_(IB_HDR_LATENCY_US, "us"))
    InfiniBand_NDR = NetworkFabric(name="IB NDR", bandwidth=INFINIBAND_NDR_BW, latency=Q_(IB_NDR_LATENCY_US, "us"))

class Clusters:
    """Vetted Production Clusters."""
    Research_256 = Fleet(
        name="Research Cluster (256 GPUs)",
        node=Nodes.DGX_H100,
        count=32, # 32 nodes * 8 GPUs = 256
        fabric=Fabrics.Ethernet_100G
    )
    Frontier_8K = Fleet(
        name="Frontier Cluster (8192 GPUs)",
        node=Nodes.DGX_H100,
        count=1024, # 1024 nodes * 8 GPUs = 8192
        fabric=Fabrics.InfiniBand_NDR
    )
    Production_2K = Fleet(
        name="Production Cluster (2048 GPUs)",
        node=Nodes.DGX_H100,
        count=256,
        fabric=Fabrics.InfiniBand_HDR
    )
    Mega_100K = Fleet(
        name="Mega Cluster (100000 GPUs)",
        node=Nodes.DGX_H100,
        count=12500,
        fabric=Fabrics.InfiniBand_NDR
    )

class Systems:
    Tiers = Tiers
    Nodes = Nodes
    Clusters = Clusters
    Fabrics = Fabrics

    # Backward-compatible aliases expected by older book content
    Cloud = Hardware.H100
    Edge = Hardware.Edge.JetsonOrinNX
    Mobile = Hardware.iPhone
    Tiny = Hardware.Tiny.ESP32_S3
    