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
    """Example autograd operation: y = alpha * x + beta."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        alpha = getattr(self, "alpha", 2.0)
        beta = getattr(self, "beta", 1.0)
        return alpha * x + beta

    def backward(self, grad_output):
        # Unwrap grad_output to NumPy array
        is_t = isinstance(grad_output, Tensor)
        g = grad_output.data if is_t else grad_output
        alpha = getattr(self, "alpha", 2.0)
        # Derivative w.r.t x is alpha * grad
        dx = alpha * g
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
    """Example custom neural network module."""

    def __init__(self, dim: int):
        super().__init__()
        std = 1.0 / np.sqrt(dim)
        w_init = np.random.randn(dim, dim) * std
        self.weight = Tensor(
            w_init.astype(np.float32),
            requires_grad=True,
        )
        self.bias = Tensor(
            np.zeros(dim, dtype=np.float32),
            requires_grad=True,
        )

    def parameters(self):
        """Return list of learnable parameters."""
        return [self.weight, self.bias]

    def forward(self, x: Tensor) -> Tensor:
        # Residual branch: x + Linear(x)
        linear_out = x.matmul(self.weight)
        return x + linear_out + self.bias


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
