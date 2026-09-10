import torch

from mlperf.runners.nanogpt import _NonOverlappingTextDataset


def test_nanogpt_quality_contexts_cover_targets_once_in_order():
    tokens = torch.arange(21)
    dataset = _NonOverlappingTextDataset(tokens, seq_len=4)

    targets = torch.cat([dataset[index][1] for index in range(len(dataset))])

    assert len(dataset) == 5
    assert dataset.total_target_tokens == 20
    assert targets.tolist() == list(range(1, 21))


def test_nanogpt_quality_contexts_drop_only_incomplete_tail():
    tokens = torch.arange(24)
    dataset = _NonOverlappingTextDataset(tokens, seq_len=5)

    targets = torch.cat([dataset[index][1] for index in range(len(dataset))])

    assert dataset.total_target_tokens == 23
    assert len(targets) == 20
    assert len(set(targets.tolist())) == len(targets)
