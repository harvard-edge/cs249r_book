"""Invalid batch sizes must fail at construction, never silently skip training."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.dataloader import DataLoader, TensorDataset


@pytest.mark.parametrize('size', [0, -1, 1.5, True, False])
def test_invalid_batch_size_is_rejected(size):
    with pytest.raises(ValueError, match='positive integer'):
        DataLoader(TensorDataset(Tensor([[1.], [2.]])), batch_size=size)


def test_numpy_integer_batch_size_and_partial_tail():
    loader = DataLoader(TensorDataset(Tensor([[1.], [2.], [3.]])), batch_size=np.int64(2))
    assert len(loader) == 2
    assert [batch[0].shape[0] for batch in loader] == [2, 1]
