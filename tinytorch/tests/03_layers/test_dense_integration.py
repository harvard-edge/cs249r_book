"""Module 03: the actual Linear/activation/Sequential exports compose correctly."""

import numpy as np

from tinytorch.core.activations import ReLU
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.tensor import Tensor


def test_dense_module_integration():
    """A hand-calculated two-layer network checks values and parameter ownership."""
    hidden = Linear(2, 2)
    output = Linear(2, 1)
    hidden.weight.data[:] = [[1, -1], [2, 1]]
    hidden.bias.data[:] = [0, -1]
    output.weight.data[:] = [[2], [-1]]
    output.bias.data[:] = [0.5]
    model = Sequential(hidden, ReLU(), output)

    result = model(Tensor([[1, 2], [-2, 1]]))
    # Hidden preactivations are [5,0] and [0,2]; ReLU preserves both.
    np.testing.assert_allclose(result.data, [[10.5], [-1.5]])
    assert model.parameters() == [hidden.weight, hidden.bias, output.weight, output.bias]


if __name__ == "__main__":
    test_dense_module_integration()
    print("Linear/Sequential integration passed")


def test_sequential_collects_shared_parameters_once():
    """Reusing a layer must not ask the optimizer to update its weights twice."""
    shared = Linear(2, 2)
    model = Sequential(shared, ReLU(), Sequential(shared))
    assert model.parameters() == [shared.weight, shared.bias]
