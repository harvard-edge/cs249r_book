# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the Apple Metal/MPS matmul bridge for Milestone 07,
# a bundled reference point students time their own Module 17 kernels
# against; no src/ module or student notebook exports to it.
"""Apple Silicon Metal / MPS Acceleration Bridge for TinyTorch.

Runs a matrix multiply on the GPU of an Apple M-series chip through PyTorch's
MPS (Metal Performance Shaders) backend. This is an optional, hand-written
extension: it is tracked in git as-is and is not generated from src/.

It needs PyTorch, which TinyTorch itself never imports; install it separately
(pip install torch) to try this path. Without it, or on a machine without an
Apple GPU, mps_matmul falls back to np.matmul.

Apple silicon shares one memory between CPU and GPU, so there is no PCIe copy,
but .to("mps") and .cpu() still copy the data into and out of GPU-owned
buffers. Those copies are part of what you measure when you time this.
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
