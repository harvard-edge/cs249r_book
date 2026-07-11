import torch

from mlperf.reference.cloud.micro_dlrm_dram import MicroDLRMDRAM
from mlperf.registry import load_registry
from mlperf.runners.dlrm import _dlrm_dram_collate_fn


def test_dlrm_dram_max_geometry_matches_registry_claims():
    workload = load_registry()["micro-dlrm-dram-train"]
    model = MicroDLRMDRAM(
        m_spa=256,
        virtual_table_size=65_536,
        sparse_grad=True,
    )

    assert workload.scenario == "training"
    assert sum(parameter.numel() for parameter in model.parameters()) == 16_802_833
    assert model.working_set_bytes() == 64 * 1024 * 1024
    assert workload.raw["params"] == 16_802_833
    assert workload.raw["virtual_table_bytes"] == model.working_set_bytes()
    assert workload.raw["llc_capacity_factor"] == "unmeasured"


def test_dlrm_dram_collate_addresses_user_item_cross_not_small_occupation_vocab():
    batch = [
        (
            torch.zeros(16),
            [torch.tensor(1), torch.tensor(10), torch.tensor(3)],
            torch.tensor([1.0]),
        ),
        (
            torch.ones(16),
            [torch.tensor(20), torch.tensor(30), torch.tensor(4)],
            torch.tensor([0.0]),
        ),
    ]

    _dense, sparse_indices, sparse_offsets, _labels = _dlrm_dram_collate_fn(batch)

    assert sparse_indices[0].tolist() == [1, 20]
    assert sparse_indices[1].tolist() == [10, 30]
    assert sparse_indices[2].tolist() == [1 * 1682 + 10, 20 * 1682 + 30]
    assert [offsets.tolist() for offsets in sparse_offsets] == [[0, 1]] * 3
