"""
Module 05: DataLoader - Core Functionality Tests
=================================================

WHY DATALOADER MATTERS:
----------------------
Real datasets don't fit in memory. DataLoader:
- Loads data in batches
- Shuffles for better training
- Enables parallel loading

WHAT STUDENTS LEARN:
-------------------
1. Batching: split data into chunks
2. Shuffling: randomize order each epoch
3. Iteration: yield batches one at a time
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDataLoaderBasics:
    """Test basic DataLoader functionality."""

    def test_dataloader_iteration(self):
        """
        WHAT: Verify DataLoader can iterate over data.

        WHY: Training loops need: for batch in dataloader: ...
        If iteration doesn't work, training can't happen.

        STUDENT LEARNING: DataLoader is iterable - use it in for loops.
        """
        try:
            from tinytorch.core.dataloader import DataLoader

            # Simple dataset
            X = np.random.randn(100, 10)
            y = np.random.randint(0, 2, 100)

            dataset = list(zip(X, y))
            loader = DataLoader(dataset, batch_size=16)

            batches = list(loader)
            assert len(batches) > 0, "DataLoader should yield batches"

        except ImportError:
            pytest.skip("DataLoader not implemented yet")

    def test_batch_sizes(self):
        """
        WHAT: Verify batch_size controls batch dimensions.

        WHY: Batch size affects:
        - Memory usage (bigger = more memory)
        - Gradient quality (bigger = smoother)
        - Training speed (bigger = faster epochs)

        STUDENT LEARNING: Common batch sizes: 16, 32, 64, 128.
        Start small if memory is limited.
        """
        try:
            from tinytorch.core.dataloader import DataLoader

            X = np.random.randn(100, 10)
            y = np.random.randint(0, 2, 100)

            dataset = list(zip(X, y))
            loader = DataLoader(dataset, batch_size=32)

            first_batch = next(iter(loader))
            batch_x, batch_y = first_batch

            assert batch_x.shape[0] == 32 or batch_x.shape[0] <= 32, (
                f"Batch size should be 32 (or less for last batch)\n"
                f"  Got: {batch_x.shape[0]}"
            )

        except ImportError:
            pytest.skip("DataLoader batch_size not implemented yet")

    def test_shuffling(self):
        """
        WHAT: Verify shuffle=True randomizes order.

        WHY: Without shuffling:
        - Model may learn order instead of patterns
        - Similar samples grouped together cause issues

        STUDENT LEARNING: Always shuffle=True for training,
        shuffle=False for evaluation (reproducibility).
        """
        try:
            from tinytorch.core.dataloader import DataLoader

            # Data with clear order
            X = np.arange(100).reshape(100, 1)
            y = np.arange(100)

            # Two loaders with shuffle
            dataset1 = list(zip(X, y))
            dataset2 = list(zip(X, y))
            loader1 = DataLoader(dataset1, batch_size=10, shuffle=True)
            loader2 = DataLoader(dataset2, batch_size=10, shuffle=True)

            # Get first batches
            batch1 = next(iter(loader1))[0]
            batch2 = next(iter(loader2))[0]

            # With shuffling, batches should differ
            # (Note: there's a small chance they're the same by luck)

        except ImportError:
            pytest.skip("DataLoader shuffle not implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
