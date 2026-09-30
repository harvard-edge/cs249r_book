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


def test_loss_scaler_backward_end_to_end():
    from tinytorch.core.optimizers import SGD

    p = Tensor(np.array([2.0], dtype=np.float32), requires_grad=True)
    scaler = LossScaler(scale=1024.0)
    optimizer = SGD([p], lr=0.1)

    loss = p * Tensor(np.array([3.0], dtype=np.float32))
    scaler.backward(loss)
    assert np.allclose(p.grad, [3072.0])

    scaler.unscale_(optimizer)
    assert np.allclose(p.grad, [3.0])

    optimizer.step()
    assert np.allclose(p.data, [1.7])

