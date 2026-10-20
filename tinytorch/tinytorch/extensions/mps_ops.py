# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the Apple Metal/MPS matmul bridge for Milestone 07,
# a bundled reference point students time their own Module 17 kernels
# against; no src/ module or student notebook exports to it.
"""Apple Silicon Metal / MPS Matrix Multiply for TinyTorch: an example extension.

What it is: a matrix multiply on the GPU of an Apple M-series chip, through
PyTorch's MPS (Metal Performance Shaders) backend. It is the contrast case
among the examples: simd_ops.py and triton_gelu.py run a kernel you can read,
while this one calls a vendor library, so it teaches where offloading pays,
not how a GPU kernel works. Hand-written and tracked in git as-is, not
generated from src/.

Why it exists: a product of two n x n float32 matrices does 2n^3 FLOPs and
moves 3n^2 floats (12n^2 bytes), so it does n/6 FLOPs per byte moved. At
n = 256 that is about 43 FLOP/byte and the copies to and from the GPU
dominate, so the GPU loses to the CPU; at n = 4096 it is about 683 and the GPU
wins even with both copies. The extensions chapter measures both ends.

How it is triggered: only when you call it. Nothing in TinyTorch calls this
file for you; Tensor.matmul never routes here.

    from tinytorch.extensions import mps_matmul, has_mps_support
    C = mps_matmul(A, B)          # A, B: NumPy arrays or Tensors
    print(has_mps_support())      # True if that ran on the GPU

    `tito milestone run 07` makes the same call for you and times it
    against your own Module 17 kernels.

    On import:     tries `import torch` and asks whether an MPS device
                   exists. tinytorch/extensions/__init__.py imports this
                   file, so this happens as soon as you import
                   tinytorch.extensions, and PyTorch is a large import.
    On every call: copy in, multiply on the GPU, copy out; or np.matmul when
                   PyTorch or an Apple GPU is missing.

How it works:

1. Detect. _HAS_MPS is set once, at import.
2. Copy in. torch.from_numpy wraps the array without copying, and .to("mps")
   copies it into a GPU-owned buffer. Apple silicon shares one memory between
   CPU and GPU, so there is no PCIe transfer, but this is still a copy.
3. Run. torch.matmul queues the product on the Apple GPU.
4. Copy out. .cpu() waits for the GPU and copies C back, and .numpy() hands
   it to TinyTorch. Both copies are part of what you measure when you time it.

What it leaves out: the kernel itself, which is PyTorch's, and gradients:
the result is a new Tensor with no gradient history, so this is a benchmark
path, not a drop-in replacement for Tensor.matmul in training. It needs
PyTorch, which TinyTorch itself never imports; install it separately to try
this path.

Where to go next: time it yourself at n = 256 and n = 4096 to find the
crossover, then write the kernel yourself in Metal and call it through the
same kind of C bridge as simd_ops.py.
"""

import numpy as np

_HAS_MPS = False
try:
    import torch
    if torch.backends.mps.is_available():
        _HAS_MPS = True
except ImportError:
    _HAS_MPS = False


def has_mps_support() -> bool:
    """Returns True if PyTorch is installed and reports an MPS device."""
    return _HAS_MPS


def _is_tensor(obj):
    return obj is not None and type(obj).__name__ == "Tensor"


def mps_matmul(a, b):
    """Matrix multiply on the Apple GPU via MPS (the GPU, not the Neural Engine).

    Args:
        a: 2D array or Tensor [M, K]
        b: 2D array or Tensor [K, N]

    Returns:
        2D float32 array or Tensor [M, N]. Falls back to np.matmul without MPS.
    """
    is_tensor = _is_tensor(a) or _is_tensor(b)
    a_arr = a.data if _is_tensor(a) else a
    b_arr = b.data if _is_tensor(b) else b

    if not _HAS_MPS:
        out = np.matmul(a_arr, b_arr).astype(np.float32)
        if is_tensor:
            from tinytorch.core.tensor import Tensor
            return Tensor(out)
        return out

    a_c = np.ascontiguousarray(a_arr, dtype=np.float32)
    b_c = np.ascontiguousarray(b_arr, dtype=np.float32)
    M, K = a_c.shape
    K_b, N = b_c.shape
    if K != K_b:
        raise ValueError(f"Incompatible matrix dimensions: ({M}, {K}) x ({K_b}, {N})")

    a_t = torch.from_numpy(a_c).to("mps")
    b_t = torch.from_numpy(b_c).to("mps")
    c_t = torch.matmul(a_t, b_t)
    out = c_t.cpu().numpy().astype(np.float32)

    if is_tensor:
        from tinytorch.core.tensor import Tensor
        return Tensor(out)
    return out
