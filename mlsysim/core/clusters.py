# clusters.py
# Multi-Node Cluster Definitions for MLSys Textbook (Volume II)
# Extends systems.py (single-node) to multi-node distributed configurations.
#
# Hierarchy:
#   HardwareSpec  (hardware.py)   — one accelerator
#   NetworkSpec   (hardware.py)   — one inter-node fabric
#   NodeSpec      (this file)     — one node (N accelerators + fabric port)
#   ClusterSpec   (this file)     — M nodes over a fabric
#   Clusters      (this file)     — named reference clusters for prose

import math
from dataclasses import dataclass, field
from typing import Optional
from .hardware import HardwareSpec, NetworkSpec, Hardware, Networks
from .constants import (
    ureg, Q_,
    GPU_MTTF_HOURS, NIC_MTTF_HOURS, PSU_MTTF_HOURS,
    CLUSTER_SMALL_GPUS, CLUSTER_MEDIUM_GPUS, CLUSTER_LARGE_GPUS, CLUSTER_MEGA_GPUS,
    INFINIBAND_NDR_BW, INFINIBAND_HDR_BW,
    MFU_TRAINING_LOW, MFU_TRAINING_HIGH,
    SCALING_EFF_32GPU, SCALING_EFF_256GPU, SCALING_EFF_1024GPU, SCALING_EFF_8192GPU,
    OVERHEAD_FAILURE_RECOVERY, OVERHEAD_CHECKPOINT, OVERHEAD_PIPELINE_BUBBLE,
    hour, TFLOPs, second,
)
from .formulas import calc_mtbf_cluster, calc_effective_flops


# ─────────────────────────────────────────────────────────────────────────────
# NodeSpec: one physical server in the cluster
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NodeSpec:
    """
    One physical training node: N accelerators sharing a host interconnect.

    Examples: DGX H100 (8× H100 + NVLink), DGX A100 (8× A100 + NVLink).
    """
    name: str
    accelerator: HardwareSpec
    accelerators_per_node: int              # GPUs / TPUs per physical host
    intra_node_bw: Q_                       # NVLink / XBar bandwidth per GPU (GB/s)
    nics_per_node: int = 1                  # High-speed NICs per host
    psus_per_node: int = 2                  # Redundant PSUs per host

    @property
    def node_peak_flops(self) -> Q_:
        """Aggregate peak FLOPS for one node."""
        return self.accelerator.peak_flops * self.accelerators_per_node

    @property
    def node_memory_capacity(self) -> Q_:
        """Total HBM across all accelerators in one node."""
        return self.accelerator.memory_capacity * self.accelerators_per_node

    @property
    def node_tdp(self) -> Optional[Q_]:
        """Total TDP for one node (accelerators only)."""
        if self.accelerator.tdp:
            return self.accelerator.tdp * self.accelerators_per_node
        return None

    @property
    def node_mtbf(self) -> Q_:
        """
        Node MTBF from heterogeneous component failure rates.
        Uses calc_mtbf_node approximation: 1/MTBF = Σ(n_i / MTBF_i).
        """
        gpu_rate  = self.accelerators_per_node / (GPU_MTTF_HOURS * ureg.hour)
        nic_rate  = self.nics_per_node          / (NIC_MTTF_HOURS * ureg.hour)
        psu_rate  = self.psus_per_node          / (PSU_MTTF_HOURS * ureg.hour)
        total_rate = gpu_rate + nic_rate + psu_rate
        return (1.0 / total_rate).to(ureg.hour)

    def __repr__(self):
        return f"Node({self.name}, {self.accelerators_per_node}× {self.accelerator.name})"


# ─────────────────────────────────────────────────────────────────────────────
# ClusterSpec: M nodes over an inter-node fabric
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClusterSpec:
    """
    A homogeneous training cluster: N_nodes × NodeSpec over a fabric.

    Key derived quantities:
      - total_gpus            accelerator count (the headline number)
      - peak_flops            aggregate hardware peak (no efficiency)
      - aggregate_memory_bw   total HBM bandwidth available to the cluster
      - cluster_mtbf          expected hours between any node failure
      - effective_flops       delivered FLOPS after MFU × scaling_eff × goodput

    Efficiency fractions:
      - mfu              Model FLOPS Utilization (kernel efficiency)
      - scaling_eff      Multi-node scaling efficiency η (communication overhead)
      - goodput          Goodput ratio = 1 - failure/checkpoint/maintenance overhead
    """
    name: str
    node: NodeSpec
    n_nodes: int
    fabric: NetworkSpec                     # Inter-node fabric (IB, RoCE, etc.)
    mfu: float = MFU_TRAINING_HIGH          # Default: well-optimized training
    scaling_eff: float = SCALING_EFF_1024GPU  # Conservative default
    goodput: float = 1.0 - OVERHEAD_FAILURE_RECOVERY - OVERHEAD_CHECKPOINT

    # ── Structural properties ──────────────────────────────────────────────

    @property
    def total_gpus(self) -> int:
        return self.n_nodes * self.node.accelerators_per_node

    @property
    def peak_flops(self) -> Q_:
        """Aggregate hardware peak FLOPS (no efficiency applied)."""
        return self.node.node_peak_flops * self.n_nodes

    @property
    def aggregate_memory_bw(self) -> Q_:
        """Sum of all HBM bandwidth across all accelerators."""
        return self.node.accelerator.memory_bw * self.total_gpus

    @property
    def total_memory_capacity(self) -> Q_:
        """Total HBM across the entire cluster."""
        return self.node.node_memory_capacity * self.n_nodes

    @property
    def total_tdp(self) -> Optional[Q_]:
        """Aggregate TDP for all accelerators in the cluster."""
        if self.node.node_tdp:
            return self.node.node_tdp * self.n_nodes
        return None

    # ── Reliability ───────────────────────────────────────────────────────

    @property
    def cluster_mtbf(self) -> Q_:
        """
        Expected hours between any node failure in the cluster.
        MTBF_cluster = MTBF_node / N_nodes  (independent failure model).
        """
        return calc_mtbf_cluster(self.node.node_mtbf, self.n_nodes)

    @property
    def cluster_mtbf_days(self) -> float:
        """Cluster MTBF in days (convenience for prose)."""
        return self.cluster_mtbf.m_as(ureg.day)

    # ── Delivered performance ─────────────────────────────────────────────

    @property
    def effective_flops(self) -> Q_:
        """
        Delivered FLOPS after MFU × scaling efficiency × goodput.
        This is the number that determines training wall-clock time.
        """
        return calc_effective_flops(self.peak_flops, self.mfu, self.scaling_eff, self.goodput)

    @property
    def peak_flops_pflops(self) -> float:
        """Peak FLOPS in PFLOPs/s (convenient for prose)."""
        return self.peak_flops.m_as(ureg.petaflop / ureg.second)

    @property
    def effective_flops_pflops(self) -> float:
        """Effective FLOPS in PFLOPs/s (convenient for prose)."""
        return self.effective_flops.m_as(ureg.petaflop / ureg.second)

    def __repr__(self):
        return f"Cluster({self.name}, {self.total_gpus} GPUs, {self.peak_flops_pflops:.0f} PFLOPs peak)"


# ─────────────────────────────────────────────────────────────────────────────
# Named Nodes — canonical server configs used across Vol2 prose
# ─────────────────────────────────────────────────────────────────────────────

class Nodes:
    """Reference node configurations."""

    DGX_H100 = NodeSpec(
        name="DGX H100",
        accelerator=Hardware.H100,
        accelerators_per_node=8,
        intra_node_bw=900 * ureg.GB / ureg.second,  # NVLink 4.0 per GPU
        nics_per_node=8,   # 8× ConnectX-7 (one per GPU)
        psus_per_node=2,
    )

    DGX_A100 = NodeSpec(
        name="DGX A100",
        accelerator=Hardware.A100,
        accelerators_per_node=8,
        intra_node_bw=600 * ureg.GB / ureg.second,  # NVLink 3.0 per GPU
        nics_per_node=8,
        psus_per_node=2,
    )

    DGX_B200 = NodeSpec(
        name="DGX B200",
        accelerator=Hardware.B200,
        accelerators_per_node=8,
        intra_node_bw=1800 * ureg.GB / ureg.second,  # NVLink 5.0 per GPU
        nics_per_node=8,
        psus_per_node=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Named Clusters — canonical reference clusters for Vol2 worked examples
# ─────────────────────────────────────────────────────────────────────────────

class Clusters:
    """
    Named reference clusters for Vol2 prose and LEGO blocks.

    Usage in QMD cells:
        from mlsysim import Clusters
        c = Clusters.Research_512
        print(f"{c.total_gpus} GPUs, {c.peak_flops_pflops:.0f} PFLOPs peak")
    """

    # --- Research-scale (256–512 GPUs) ---
    # Typical university or mid-tier cloud research cluster.
    Research_256 = ClusterSpec(
        name="Research Cluster (256 GPUs)",
        node=Nodes.DGX_H100,
        n_nodes=32,                     # 32 nodes × 8 GPUs = 256 GPUs
        fabric=Networks.Ethernet_100G,
        mfu=MFU_TRAINING_HIGH,
        scaling_eff=SCALING_EFF_256GPU,
        goodput=1.0 - OVERHEAD_CHECKPOINT,
    )

    # --- Production-scale (2 K GPUs) ---
    # Hyperscaler fine-tuning / mid-scale pre-training cluster.
    Production_2K = ClusterSpec(
        name="Production Cluster (2 048 GPUs)",
        node=Nodes.DGX_H100,
        n_nodes=256,                    # 256 nodes × 8 GPUs = 2 048 GPUs
        fabric=NetworkSpec("IB HDR", INFINIBAND_HDR_BW),
        mfu=MFU_TRAINING_HIGH,
        scaling_eff=SCALING_EFF_1024GPU,
        goodput=1.0 - OVERHEAD_CHECKPOINT - OVERHEAD_FAILURE_RECOVERY / 2,
    )

    # --- Frontier-scale (8 K GPUs) ---
    # Large-scale pre-training (LLaMA-scale, Falcon-scale).
    Frontier_8K = ClusterSpec(
        name="Frontier Cluster (8 192 GPUs)",
        node=Nodes.DGX_H100,
        n_nodes=1024,                   # 1 024 nodes × 8 GPUs = 8 192 GPUs
        fabric=NetworkSpec("IB NDR", INFINIBAND_NDR_BW),
        mfu=MFU_TRAINING_HIGH,
        scaling_eff=SCALING_EFF_8192GPU,
        goodput=1.0 - OVERHEAD_CHECKPOINT - OVERHEAD_FAILURE_RECOVERY,
    )

    # --- Mega-scale (100 K GPUs) ---
    # Frontier model training (GPT-4-scale, Gemini-scale).
    Mega_100K = ClusterSpec(
        name="Mega Cluster (100 000 GPUs)",
        node=Nodes.DGX_H100,
        n_nodes=12500,                  # 12 500 nodes × 8 GPUs = 100 000 GPUs
        fabric=NetworkSpec("IB NDR", INFINIBAND_NDR_BW),
        mfu=MFU_TRAINING_LOW,           # Lower MFU at mega-scale
        scaling_eff=SCALING_EFF_8192GPU * 0.80,  # Extra degradation beyond 8K
        goodput=1.0 - OVERHEAD_CHECKPOINT - OVERHEAD_FAILURE_RECOVERY - OVERHEAD_PIPELINE_BUBBLE,
    )
