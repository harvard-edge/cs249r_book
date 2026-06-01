from __future__ import annotations

import pytest

from mlsysim import Systems
from mlsysim.core.units import GB, MW, TB, second


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


def test_production_2k_checkpoint_storage_path():
    path = Systems.Storage.Production2KCheckpointPath

    assert path.local_stage.devices_per_node == 4
    assert path.local_stage.device is Systems.Storage.LocalNvmeGen4
    assert path.local_stage.aggregate_bandwidth.to(GB / second).magnitude == pytest.approx(
        28.0
    )
    assert path.durable_store is Systems.Storage.PfsOneTbPerSecond
    assert path.write_bandwidth.to(TB / second).magnitude == pytest.approx(1.0)
