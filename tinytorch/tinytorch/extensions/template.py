"""TinyTorch Extension Starter Template.

Use this file as a canonical starting point when authoring a new extension for TinyTorch.

Extensions live in `tinytorch/extensions/` alongside the 20 core modules. An extension
can introduce new autograd functions, custom neural network layers, hardware acceleration
kernels (via C++ / Triton / MPS), or compilation passes.

THE EXTENSION CONTRACT:
1. Systems Framing: Document the concrete systems bottleneck (memory bandwidth, FLOPs,
   underflow, cache locality) that this extension addresses.
2. Tensor Interoperability: Functions should accept either `Tensor` or `np.ndarray`.
   If given `Tensor`, return `Tensor`. If given `np.ndarray`, return `np.ndarray`.
3. Autograd Transparency: Differentiable operations should inherit from `Function` and
   implement both `forward` and `backward`. Custom neural network blocks should subclass `Layer`.
4. Graceful Fallbacks: If custom hardware, compilers, or accelerator drivers are missing,
   always fall back cleanly to a pure-Python or NumPy reference implementation.
"""

from typing import Optional, Tuple, Union
import numpy as np

from tinytorch.core.tensor import Tensor, Function
from tinytorch.core.layers import Layer


# =====================================================================
# 1. Custom Differentiable Autograd Operator (subclassing Function)
# =====================================================================

class CustomScaledShift(Function):
    """Example autograd operation: y = alpha * x + beta.

    Demonstrates the Function lifecycle:
    - forward() receives raw NumPy arrays (unwrapped by apply()).
    - backward() receives the incoming gradient (as a NumPy array or Tensor)
      and returns downstream gradients matching each input Tensor.
    - Hyperparameters (e.g. alpha, beta) are passed as keyword arguments to apply()
      and stored on the node instance.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Hyperparameters passed to apply() are bound as attributes on self
        alpha = getattr(self, "alpha", 2.0)
        beta = getattr(self, "beta", 1.0)
        return alpha * x + beta

    def backward(self, grad_output: Union[np.ndarray, Tensor]) -> Tuple[np.ndarray]:
        # Unwrap grad_output to a NumPy array if needed
        g = grad_output.data if isinstance(grad_output, Tensor) else grad_output
        alpha = getattr(self, "alpha", 2.0)
        # Derivative w.r.t x is alpha * grad_output
        dx = alpha * g
        # Return a tuple of gradients corresponding to each input Tensor
        return (dx,)


def custom_scaled_shift(
    x: Union[Tensor, np.ndarray],
    alpha: float = 2.0,
    beta: float = 1.0,
) -> Union[Tensor, np.ndarray]:
    """Public functional interface with Tensor / NumPy interoperability."""
    if isinstance(x, Tensor):
        return CustomScaledShift.apply(x, alpha=alpha, beta=beta)
    return alpha * np.asarray(x, dtype=np.float32) + beta


# =====================================================================
# 2. Custom Neural Network Module (subclassing Layer)
# =====================================================================

class CustomResidualBlock(Layer):
    """Example custom neural network module.

    Demonstrates how to register parameters and compose forward operations
    using TinyTorch primitives while maintaining clean parameter tracking.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Initialized trainable parameters registered on the layer
        self.weight = Tensor(
            np.random.randn(dim, dim).astype(np.float32) / np.sqrt(dim),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True)

    def parameters(self):
        """Return list of learnable parameters for optimizer traversal."""
        return [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        # Residual branch: x + Linear(x)
        linear_out = x.matmul(self.weight) + self.bias
        return x + linear_out


# =====================================================================
# 3. Hardware / Kernel Hook with Automated Fallback
# =====================================================================

def has_custom_hardware() -> bool:
    """Detect whether custom hardware runtime or compiled C++/CUDA kernel is available."""
    # Replace with real hardware check (e.g., ctypes library load or torch.cuda.is_available())
    return False


def custom_accelerated_op(
    x: Union[Tensor, np.ndarray],
) -> Union[Tensor, np.ndarray]:
    """Example hardware-accelerated operator demonstrating safe fallback.

    If custom acceleration is detected, dispatches to the hardware path;
    otherwise falls back transparently to NumPy without raising an error.
    """
    is_tensor = isinstance(x, Tensor)
    x_data = x.data if is_tensor else np.asarray(x)

    if has_custom_hardware():
        # Dispatch to compiled C++ shared library, Triton kernel, or GPU runtime
        out_data = x_data  # placeholder for hardware execution
    else:
        # Transparent fallback to CPU reference implementation (e.g., ReLU)
        out_data = np.maximum(0, x_data)

    return Tensor(out_data) if is_tensor else out_data
