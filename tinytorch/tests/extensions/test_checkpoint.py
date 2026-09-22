import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.extensions.checkpoint import checkpoint

def test_activation_checkpointing():
    def my_block(x):
        a = x * Tensor(np.array([2.0], dtype=np.float32))
        b = a + Tensor(np.array([3.0], dtype=np.float32))
        return b
        
    x = Tensor(np.array([1.5], dtype=np.float32), requires_grad=True)
    out = checkpoint(my_block, x)
    
    # Forward check
    assert np.allclose(out.data, [6.0])
    
    # Backward check
    out.backward(Tensor(np.array([1.0], dtype=np.float32)))
    assert np.allclose(x.grad, [2.0])
