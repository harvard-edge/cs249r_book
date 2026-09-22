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
