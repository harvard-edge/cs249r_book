"""
Boundary-value tests for DataLoader batching.

Systematic sweep of batch-size/dataset-size edge cases: a dataset size
not evenly divisible by batch_size (the last batch is partial), a single
sample, and a batch_size equal to the whole dataset. None of these were
previously exercised, only "typical" evenly-divisible cases were.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.dataloader import TensorDataset, DataLoader


class TestDataLoaderBoundaryValues:
    def test_partial_last_batch_has_correct_size_and_content(self):
        """5 samples, batch_size=2 -> batches of [2, 2, 1], not silently
        dropped or padded."""
        features = Tensor(np.arange(5).reshape(5, 1).astype(np.float32))
        labels = Tensor(np.arange(5).astype(np.float32))
        dataset = TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        batches = list(loader)

        assert len(batches) == 3
        assert batches[0][0].shape == (2, 1)
        assert batches[1][0].shape == (2, 1)
        assert batches[2][0].shape == (1, 1)

        # Nothing lost or duplicated: concatenating batches recovers the
        # full dataset in order.
        all_labels = np.concatenate([b[1].data for b in batches])
        np.testing.assert_array_equal(all_labels, np.arange(5))

    def test_len_matches_actual_number_of_batches_yielded_for_partial_batch(self):
        features = Tensor(np.arange(7).reshape(7, 1).astype(np.float32))
        dataset = TensorDataset(features)
        loader = DataLoader(dataset, batch_size=3, shuffle=False)

        assert len(loader) == 3
        assert len(list(loader)) == len(loader)

    def test_batch_size_one_yields_one_sample_per_batch(self):
        features = Tensor(np.arange(4).reshape(4, 1).astype(np.float32))
        dataset = TensorDataset(features)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        batches = list(loader)

        assert len(batches) == 4
        for i, batch in enumerate(batches):
            assert batch[0].shape == (1, 1)
            assert batch[0].data[0, 0] == i

    def test_batch_size_equal_to_dataset_size_yields_single_batch(self):
        features = Tensor(np.arange(6).reshape(6, 1).astype(np.float32))
        dataset = TensorDataset(features)
        loader = DataLoader(dataset, batch_size=6, shuffle=False)

        batches = list(loader)

        assert len(batches) == 1
        assert batches[0][0].shape == (6, 1)

    def test_batch_size_larger_than_dataset_yields_one_partial_batch(self):
        """batch_size exceeding the dataset size must not crash or hang,
        it should just yield everything as one (smaller than requested)
        batch."""
        features = Tensor(np.arange(3).reshape(3, 1).astype(np.float32))
        dataset = TensorDataset(features)
        loader = DataLoader(dataset, batch_size=100, shuffle=False)

        batches = list(loader)

        assert len(batches) == 1
        assert batches[0][0].shape == (3, 1)

    def test_single_sample_dataset(self):
        features = Tensor(np.array([[42.0]]))
        dataset = TensorDataset(features)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        batches = list(loader)

        assert len(batches) == 1
        assert batches[0][0].shape == (1, 1)
        assert batches[0][0].data[0, 0] == 42.0
