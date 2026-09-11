from __future__ import annotations

import pytest

from mlsysim import Systems
from mlsysim.core.units import GB, MW, TB, TiB, bit, kilowatt, pJ, second


def test_reference_25k_h100_cluster_totals():
    fleet = Systems.Clusters.Reference_25K_H100

    assert fleet.count == 3125
    assert fleet.node.accelerators_per_node == 8
    assert fleet.total_accelerators == 25_000
    assert fleet.fabric is Systems.Fabrics.InfiniBand_NDR


def test_reference_25k_h100_cluster_tdp_power():
    fleet = Systems.Clusters.Reference_25K_H100

    power = fleet.total_accelerators * fleet.node.accelerator.tdp
    assert power.to(MW).magnitude == pytest.approx(17.5)


def test_training_1k_a100_cluster_matches_debug_examples():
    fleet = Systems.Clusters.Training_1K_A100

    assert fleet.count == 128
    assert fleet.node is Systems.Nodes.DGX_A100
    assert fleet.total_accelerators == 1_024
    assert fleet.fabric is Systems.Fabrics.InfiniBand_HDR


def test_small_reference_cluster_tiers():
    assert Systems.Clusters.Lab_64_H100.count == 8
    assert Systems.Clusters.Lab_64_H100.total_accelerators == 64
    assert Systems.Clusters.Training_512_H100.count == 64
    assert Systems.Clusters.Training_512_H100.total_accelerators == 512


def test_kempner_h100_partition_profile():
    fleet = Systems.Clusters.Kempner_H100_384

    assert fleet.count == 96
    assert fleet.node is Systems.Nodes.Kempner_H100_4GPU
    assert fleet.node.accelerators_per_node == 4
    assert fleet.total_accelerators == 384
    assert fleet.fabric is Systems.Fabrics.InfiniBand_NDR


def test_reliability_node_composite_profile():
    assert Systems.Reliability.DgxNodeComposite.mttf_hours == pytest.approx(1_000)
    assert Systems.Reliability.NodeRecoveryLowMin == pytest.approx(10)
    assert Systems.Reliability.NodeRecoveryHighMin == pytest.approx(30)
    recovery = Systems.Reliability.Recovery
    assert float(recovery.detection_time_s) == pytest.approx(60)
    assert float(recovery.restart_time_s) == pytest.approx(180)
    assert float(recovery.warmup_time_s) == pytest.approx(120)
    assert float(recovery.checkpoint_write_bw_gbs) == pytest.approx(100)


def test_storage_random_access_profiles():
    assert Systems.Storage.Hdd7200Rpm.bandwidth.to(GB / second).magnitude == pytest.approx(0.15)
    assert Systems.Storage.Hdd7200Rpm.iops == pytest.approx(100)
    assert Systems.Storage.SataSsd.bandwidth.to(GB / second).magnitude == pytest.approx(0.55)
    assert Systems.Storage.SataSsd.iops == pytest.approx(10_000)
    assert Systems.Storage.LocalNvmeGen3.bandwidth.to(GB / second).magnitude == pytest.approx(3.5)
    assert Systems.Storage.LocalNvmeGen3.iops == pytest.approx(500_000)


def test_switch_fabric_port_anchors():
    assert int(Systems.SwitchFabric.NdrSwitchPorts) == 64
    assert int(Systems.SwitchFabric.NdrLeafDownlinkPorts) == 32
    assert int(Systems.SwitchFabric.NdrLeafUplinkPorts) == 32


def test_network_energy_link_anchors():
    assert Systems.NetworkEnergy.NvlinkEnergyPerBit.to(pJ / bit).magnitude == pytest.approx(7.5)
    assert Systems.NetworkEnergy.InfiniBandEnergyPerBit.to(pJ / bit).magnitude == pytest.approx(35.0)


def test_rack_profiles():
    rack = Systems.Racks.DGX_H100_4Node
    assert rack.nodes_per_rack == 4
    assert rack.node is Systems.Nodes.DGX_H100
    assert rack.accelerator_count == 32
    assert rack.node.host_memory.to(TiB).magnitude == pytest.approx(2.0)
    assert rack.non_accelerator_power.to(kilowatt).magnitude == pytest.approx(11.1)
    assert rack.accelerator_power.to(kilowatt).magnitude == pytest.approx(22.4)
    assert rack.total_power.to(kilowatt).magnitude == pytest.approx(33.5)


def test_storage_training_effective_profiles():
    assert Systems.Storage.CloudBlockStorageBaseline.bandwidth.to(GB / second).magnitude == pytest.approx(0.25)
    assert Systems.Storage.ObjectStoreSingleStream.bandwidth.to(GB / second).magnitude == pytest.approx(0.1)
    assert Systems.Storage.SataSsdEffective.bandwidth.to(GB / second).magnitude == pytest.approx(0.5)
    assert Systems.Storage.LocalNvmeTrainingLow.bandwidth.to(GB / second).magnitude == pytest.approx(3.0)
    assert Systems.Storage.LocalNvmeTrainingTypical.bandwidth.to(GB / second).magnitude == pytest.approx(5.0)


def test_storage_access_path_latency_profiles():
    assert Systems.Storage.TraditionalGpuIoLatency.to("microsecond").magnitude == pytest.approx(120)
    assert Systems.Storage.GpuDirectStorageLatency.to("microsecond").magnitude == pytest.approx(30)


def test_production_2k_checkpoint_storage_path():
    path = Systems.Storage.Production2KCheckpointPath

    assert path.local_stage.devices_per_node == 4
    assert path.local_stage.device is Systems.Storage.LocalNvmeGen4
    assert path.local_stage.aggregate_bandwidth.to(GB / second).magnitude == pytest.approx(
        28.0
    )
    assert path.durable_store is Systems.Storage.PfsOneTbPerSecond
    assert path.write_bandwidth.to(TB / second).magnitude == pytest.approx(1.0)
