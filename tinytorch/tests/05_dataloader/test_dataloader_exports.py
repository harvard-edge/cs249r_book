"""The exported Module 05 loader must deliver real batches to earlier layers."""
import numpy as np


def test_public_loader_is_the_source_export():
    from tinytorch import DataLoader
    from tinytorch.core.dataloader import DataLoader as SourceDataLoader

    assert DataLoader is SourceDataLoader


def test_exported_loader_preserves_samples_through_a_layer():
    from tinytorch import DataLoader, Tensor, Linear
    from tinytorch.core.dataloader import TensorDataset

    x = Tensor([[1., 2.], [3., 4.], [5., 6.]])
    labels = Tensor([0, 1, 2])
    loader = DataLoader(TensorDataset(x, labels), batch_size=2, shuffle=False)
    layer = Linear(2, 1, bias=False)
    layer.weight.data[:] = [[1.], [2.]]
    outputs, observed_labels, sizes = [], [], []
    for batch, target in loader:
        outputs.extend(layer(batch).data[:, 0])
        observed_labels.extend(target.data)
        sizes.append(batch.shape[0])
    np.testing.assert_array_equal(outputs, [5., 11., 17.])
    np.testing.assert_array_equal(observed_labels, [0, 1, 2])
    assert sizes == [2, 1]
