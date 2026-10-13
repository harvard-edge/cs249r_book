'''TinyTorch Extension Template.

Here is the shape every extension in this folder follows. To write one:
copy this file, fill in the skeleton below, delete the code sections you do
not need, and read the example that matches your kind of extension first.

=============================================================================
READ AN EXAMPLE FIRST
=============================================================================
Pick the boundary your idea lives at; each has a finished example to copy:

  Autograd (a new op, or what the tape keeps)  -> checkpoint.py   (Section 1)
  Layer (new parameters or architecture)        -> lora.py         (Section 2)
  Around the optimizer (gradient handling)      -> loss_scaler.py  (Section 3)
  Compiler (capturing a computation)            -> compile.py
  Silicon (C++, CUDA, Triton, Metal)            -> simd_ops.py + cpp_simd_gemm.cpp,
                                                   then triton_gelu.py, mps_ops.py
                                                   (Section 4)

=============================================================================
FILL THIS OUT
=============================================================================
Replace this whole docstring in your copy with the skeleton below, and replace
every <...>. Keep the sections and their order; readers rely on finding
"How it is triggered" in the same place in every file.

    # Hand-maintained extension, NOT generated. This file is not produced by
    # `tito dev export` or `tito module complete`; it is tracked in git and
    # edited directly here. It is <the role it plays in the course>; no src/
    # module or student notebook exports to it.
    """<Name> for TinyTorch: an example extension.

    What it is: <one or two sentences: what it does, and which boundary it
    extends (autograd, layer, optimizer, compiler, or silicon)>.

    Why it exists: <the bottleneck it addresses, as a number a reader could
    recompute: bytes moved, FLOP/byte, parameters trained, the smallest
    float16 value>.

    How it is triggered: <who calls it and when. Nothing in TinyTorch calls an
    extension for you, so say exactly what the reader writes, and where:
    a direct call, a layer placed in a model, a wrapper inside forward(), or
    a call from the training loop>.

        <3 to 8 lines: the real call site, run and checked before you commit>

        On import:     <what runs when the module is imported; ideally
                       nothing expensive>
        On first call: <one-time work, such as compiling a kernel; omit if
                       none>
        On every call: <what each call does, including the fallback path>
        On backward:   <for anything that takes part in training; omit if not>

    How it works:

    1. <numbered steps a reader can follow through the code, from the call to
       the result>

    What it leaves out: <the simplification you made on purpose, and what a
    production version does instead. If the result has no gradient history,
    say so here>.

    Where to go next: <the natural follow-on project>.
    """

=============================================================================
HOW TO BUILD YOUR OWN EXTENSION IN 5 STEPS
=============================================================================
1. IDENTIFY THE BOUNDARY, and read its example (the table above).

2. COPY THIS TEMPLATE:
   cp tinytorch/extensions/template.py tinytorch/extensions/my_extension.py
   Fill in the skeleton, keep the code section for your boundary, delete the
   rest.

3. IMPLEMENT THE EXTENSION CONTRACT:
   - Systems Framing: State the concrete hardware bottleneck (bandwidth,
     underflow, FLOPS) as a number a reader could recompute, not an adjective.
   - Tensor Interoperability: If given `Tensor`, return `Tensor`. If given
     `ndarray`, return `ndarray`.
   - Autograd Transparency: Differentiable ops must provide forward & backward
     passes. Say which kind yours is. A hardware kernel that returns a plain
     Tensor(out) cuts the graph: that is fine for a benchmark path
     (simd_matmul, mps_matmul), but say so in "What it leaves out" rather than
     letting a reader find out in training.
   - Defensive Fallback: If custom compilers or GPUs are missing, fall back
     cleanly to NumPy, and make it possible to tell which path ran: a has_*()
     check, a *_build_info() that says what was built and why not, and a
     warning (not silence) if a native call fails at run time. A NumPy timing
     mistaken for a GPU one teaches the wrong lesson.

4. CREATE UNIT TESTS:
   cp tests/extensions/test_template.py tests/extensions/test_my_extension.py
   pytest tinytorch/tests/extensions/test_my_extension.py
   Tests must pass on a machine without your hardware: cover the fallback
   path, and check the native path against NumPy whenever it is available.

5. EXPORT PUBLIC SYMBOLS:
   Add your classes/functions to `tinytorch/extensions/__init__.py`.

Beyond the docstring, put short comments at the lines where the systems idea
lives (the atomic file replace, the coalesced load, the no_grad block),
saying why, not what.
=============================================================================
'''

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
# This section shows the shape of the dispatch with a stand-in kernel. For the
# real mechanics, build on simd_ops.py: compile on first use, write the library
# with an atomic os.replace, load it with ctypes and declare argtypes, pass raw
# pointers to contiguous float32 buffers, and report what was built through a
# *_build_info() function so a reader can tell which path actually ran.

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
