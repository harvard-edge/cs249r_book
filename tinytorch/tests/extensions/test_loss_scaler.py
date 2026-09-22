import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import Optimizer
from tinytorch.extensions.loss_scaler import LossScaler

class MockOptimizer(Optimizer):
    def __init__(self, params):
        self.params = params
    def step(self):
        pass
    def zero_grad(self):
        pass

def test_loss_scaler_unscales_gradients():
    # Mock a parameter with a gradient
    p = Tensor(np.array([1.0]), requires_grad=True)
    p.grad = np.array([65536.0])

    opt = MockOptimizer([p])
    scaler = LossScaler(scale=65536.0)

    scaler.unscale_(opt)
    assert np.allclose(p.grad, [1.0])
