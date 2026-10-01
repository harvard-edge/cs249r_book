import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.activations import ReLU
from tinytorch.core.losses import MSELoss
from tinytorch.core.optimizers import SGD
from tinytorch.extensions.lora import LoRALinear

def test_lora_linear_forward():
    layer = LoRALinear(128, 64, rank=4)
    x = Tensor(np.random.randn(32, 128).astype(np.float32))
    out = layer(x)
    assert out.shape == (32, 64)
    assert not layer.weight.requires_grad
    assert not layer.bias.requires_grad
    assert layer.A.requires_grad
    assert layer.B.requires_grad

def test_lora_learning_xor():
    np.random.seed(42)
    X = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32))
    Y = Tensor(np.array([[0],[1],[1],[0]], dtype=np.float32))

    class XORModel:
        def __init__(self):
            self.l1 = Linear(2, 16)
            self.relu = ReLU()
            self.l2 = LoRALinear(16, 1, rank=4)
            
        def parameters(self):
            return self.l1.parameters() + self.relu.parameters() + self.l2.parameters()
            
        def __call__(self, x):
            x = self.l1(x)
            x = self.relu(x)
            x = self.l2(x)
            return x

    model = XORModel()
    optimizer = SGD(model.parameters(), lr=0.1)
    loss_fn = MSELoss()

    # Train for a bit
    out = model(X)
    initial_loss = loss_fn(out, Y).data.item()

    for _ in range(100):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()
        
    final_loss = loss.data.item()
    assert final_loss < initial_loss


def test_lora_parameters_are_exactly_the_adapter_and_they_train():
    """An optimizer built from parameters() must update A and B, never W or b."""
    np.random.seed(0)
    layer = LoRALinear(8, 4, rank=2)
    params = layer.parameters()
    assert len(params) == 2
    assert params[0] is layer.A and params[1] is layer.B

    w_before = layer.weight.data.copy()
    b_before = layer.B.data.copy()
    optimizer = SGD(layer.parameters(), lr=0.1)
    x = Tensor(np.random.randn(5, 8).astype(np.float32))
    y = Tensor(np.random.randn(5, 4).astype(np.float32))
    loss = MSELoss()(layer(x), y)
    loss.backward()
    optimizer.step()

    assert not np.allclose(layer.B.data, b_before), "adapter B did not train"
    assert np.array_equal(layer.weight.data, w_before), "frozen base weight changed"
