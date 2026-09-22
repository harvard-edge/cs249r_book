from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Layer
import numpy as np

class LoRALinear(Layer):
    """
    Low-Rank Adaptation (LoRA) for a Linear layer.
    Replaces a dense weight matrix with a frozen weight matrix + a low-rank adapter.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        # 1. The frozen pre-trained weights
        self.weight = Tensor(np.random.randn(in_features, out_features) / np.sqrt(in_features), requires_grad=False)
        self.bias = Tensor(np.zeros(out_features), requires_grad=False)
        
        # 2. The low-rank adapter matrices (these are trained!)
        # A maps from in_features down to the bottleneck rank
        self.A = Tensor(np.random.randn(in_features, rank) / np.sqrt(in_features), requires_grad=True)
        # B maps from the bottleneck rank back up to out_features
        self.B = Tensor(np.zeros((rank, out_features)), requires_grad=True)
        
    def forward(self, x: Tensor) -> Tensor:
        # Standard forward pass: x @ W + b
        base = x.matmul(self.weight) + self.bias
        
        # Adapter forward pass: (x @ A) @ B
        adapter = x.matmul(self.A).matmul(self.B)
        
        # The autograd tape automatically merges gradients where they sum!
        return base + adapter
