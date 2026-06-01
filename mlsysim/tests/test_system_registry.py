from __future__ import annotations

import pytest

from mlsysim import Systems
from mlsysim.core.units import MW


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
