# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the LoRA example of the layer extension
# point; no src/ module or student notebook
# exports to it.
"""Low-Rank Adaptation (LoRA) for TinyTorch: an example extension.

What it is: a Linear layer whose dense weight is frozen and whose update is
learned as the product of two thin matrices. It is the example of extending
through a Layer: a new module that reuses the framework's own autograd with
no backward pass of its own. Hand-written and tracked in git as-is, not
generated from src/.

Why it exists: fine-tuning a dense in x out weight trains in x out numbers.
LoRA freezes that matrix and trains A (in x rank) and B (rank x out) instead.
At rank 4 on a 4096 x 4096 layer that is 4 x (4096 + 4096) = 32,768 trainable
numbers instead of 16,777,216, about 0.2%. What shrinks is everything that
scales with the trainable count: the gradients and the optimizer state (Adam
keeps two more numbers per trainable parameter). What does not shrink is the
frozen weight itself or the activations of the forward pass.

How it is triggered: you put it in a model where a Linear layer would go.
After that, nothing new happens: your normal training loop drives it.

    from tinytorch.extensions import LoRALinear
    from tinytorch.core.optimizers import SGD
    layer = LoRALinear(512, 512, rank=4)          # instead of Linear(512, 512)
    optimizer = SGD(layer.parameters(), lr=0.01)  # sees only A and B
    y = layer(x)                                  # calls forward()
    loss.backward()                               # autograd fills A.grad, B.grad
    optimizer.step()                              # updates A and B only

    On construction: allocates the frozen weight and bias and the two
                     adapter matrices.
    On every call:   forward() runs the base path and the adapter path.
    On backward:     ordinary autograd; there is no LoRA-specific code.

How it works:

1. The base weight and bias are created with requires_grad=False, so autograd
   never computes or stores gradients for them.
2. B starts at zero, so A @ B is zero and the layer begins exactly equal to
   the frozen base layer. Training moves it away only as far as it needs to.
3. forward() computes x @ W + b and (x @ A) @ B and adds them. The addition is
   an ordinary Tensor op, so autograd routes the gradient into both A and B.
4. parameters() returns only A and B, so an optimizer built from it updates
   the adapter and nothing else.

What it leaves out: loading real weights. Here the frozen weight is random;
in real fine-tuning it comes from a pretrained model.

Where to go next: merge the adapter after training (W + A @ B is an ordinary
dense matrix, so inference pays nothing extra), or write a hook that swaps
every Linear layer in a model for a LoRALinear.
"""

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

    def parameters(self):
        """Only the adapter matrices train; the base weight and bias stay frozen.

        Without this, an optimizer built from model.parameters() never sees A
        and B, so fine-tuning silently updates nothing in this layer.
        """
        return [self.A, self.B]
