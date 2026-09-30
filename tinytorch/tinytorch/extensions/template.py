"""TinyTorch Extension Starter Template.

Use this file as a canonical starting point when authoring a new extension for TinyTorch.

Extensions live in `tinytorch/extensions/` alongside the 20 core modules. An extension
can introduce new autograd functions, custom neural network layers, training hooks,
or hardware acceleration kernels (via C++ / Triton / MPS).

=============================================================================
HOW TO BUILD YOUR OWN EXTENSION IN 5 STEPS:
=============================================================================
1. IDENTIFY THE BOUNDARY:
   - Need custom autograd math or activation recomputation? -> Subclass `Function` (Section 1)
   - Need new layer architectures, adapters, or parameter freezing? -> Subclass `Layer` (Section 2)
   - Need custom gradient scaling, clipping, or optimizer rules? -> Wrap `Optimizer` (Section 3)
   - Need bare-metal C++, CUDA, Triton, or Apple MPS speed? -> Use C-ABI FFI (Section 4)

2. COPY THIS TEMPLATE:
   cp tinytorch/extensions/template.py tinytorch/extensions/my_extension.py

3. IMPLEMENT THE EXTENSION CONTRACT:
   - Systems Framing: Document the concrete hardware bottleneck (bandwidth, underflow, FLOPS).
   - Tensor Interoperability: If given `Tensor`, return `Tensor`. If given `ndarray`, return `ndarray`.
   - Autograd Transparency: Differentiable ops must provide forward & backward passes.
   - Defensive Fallback: If custom compilers or GPUs are missing, fall back cleanly to NumPy.

4. CREATE UNIT TESTS:
   cp tests/extensions/test_template.py tests/extensions/test_my_extension.py
   pytest tinytorch/tests/extensions/test_my_extension.py

5. EXPORT PUBLIC SYMBOLS:
   Add your classes/functions to `tinytorch/extensions/__init__.py`.
=============================================================================
"""

from typing import Optional, Tuple, Union, List
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
# 3. Custom Training & Gradient Transformation (wrapping Optimizer)
# =====================================================================

class CustomGradientTransform:
    """Example training dynamics hook: scales gradients or clips thresholds."""

    def __init__(self, clip_value: float = 1.0):
        self.clip_value = clip_value

    def transform_gradients(self, parameters: List[Tensor]) -> None:
        """In-place gradient transformation before optimizer.step()."""
        for p in parameters:
            if p.grad is not None:
                p.grad = np.clip(p.grad, -self.clip_value, self.clip_value)


# =====================================================================
# 4. Hardware / Kernel Hook with Automated Fallback (Silicon Boundary)
# =====================================================================

def has_custom_hardware() -> bool:
    """Detect whether custom hardware runtime or compiled C++/CUDA kernel is available."""
    # Replace with real hardware check (e.g., ctypes library load or torch.cuda.is_available())
    return False


def _hardware_relu_kernel(x_data: np.ndarray) -> np.ndarray:
    """Stand-in for a compiled ReLU kernel (C++ / CUDA / Triton / MPS).

    Replace the body with the native call (e.g., a ctypes function that writes
    into a preallocated output buffer). Whatever runs here must compute the
    same function as the NumPy fallback, so parity tests can compare them.
    """
    out = np.empty_like(x_data)
    np.maximum(x_data, 0, out=out)
    return out


def custom_accelerated_op(
    x: Union[Tensor, np.ndarray],
) -> Union[Tensor, np.ndarray]:
    """Example hardware-accelerated ReLU demonstrating safe fallback.

    Both paths compute y = max(0, x). If custom acceleration is detected,
    dispatches to the hardware kernel; otherwise falls back transparently
    to NumPy without raising an error.
    """
    is_tensor = isinstance(x, Tensor)
    x_data = x.data if is_tensor else np.asarray(x)

    if has_custom_hardware():
        # Dispatch to compiled C++ shared library, Triton kernel, or GPU runtime
        out_data = _hardware_relu_kernel(x_data)
    else:
        # Transparent fallback to the CPU reference implementation (ReLU)
        out_data = np.maximum(0, x_data)

    return Tensor(out_data) if is_tensor else out_data
