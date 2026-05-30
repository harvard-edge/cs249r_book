"""CI gate: registry entries must not duplicate hardware interconnect specs."""

from __future__ import annotations

from mlsysim.hardware.registry import Hardware
from mlsysim.systems.registry import Systems


def _qty_equal(a, b) -> bool:
    return abs(a.m_as("byte/second") - b.m_as("byte/second")) < 1e-6


def test_nodes_intra_node_bw_matches_accelerator_nvlink() -> None:
    pairs = (
        (Systems.Nodes.DGX_H100, Hardware.Cloud.H100),
        (Systems.Nodes.DGX_A100, Hardware.Cloud.A100),
        (Systems.Nodes.DGX_B200, Hardware.Cloud.B200),
    )
    violations: list[str] = []
    for node, accel in pairs:
        expected = accel.nvlink.bandwidth
        actual = node.intra_node_bw
        if not _qty_equal(actual, expected):
            violations.append(
                f"{node.name}: intra_node_bw={actual:~P} != {accel.name} nvlink={expected:~P}"
            )
    assert not violations, "Nodes must source intra_node_bw from Hardware nvlink:\n  " + "\n  ".join(
        violations
    )


def test_fabrics_bandwidth_matches_canonical_constants() -> None:
    # Ethernet bandwidth now lives ON the fabric: Systems.Fabrics is the source of
    # truth (the old NETWORK_*/ETHERNET_* constants were retired in the taxonomy
    # refactor). Guard the canonical Gbps figures against accidental drift.
    pairs = (
        (Systems.Fabrics.Ethernet_10G, 10),
        (Systems.Fabrics.Ethernet_100G, 100),
        (Systems.Fabrics.Ethernet_400G, 400),
        (Systems.Fabrics.Ethernet_800G, 800),
        (Systems.Fabrics.Ethernet_1P6T, 1600),
    )
    violations: list[str] = []
    for fabric, expected_gbps in pairs:
        actual = fabric.bandwidth.m_as("gigabit/second")
        if abs(actual - expected_gbps) > 1e-6:
            violations.append(f"{fabric.name}: {actual} Gbps != canonical {expected_gbps} Gbps")
    assert not violations, "Fabrics bandwidth drifted from canonical Gbps figures:\n  " + "\n  ".join(
        violations
    )
