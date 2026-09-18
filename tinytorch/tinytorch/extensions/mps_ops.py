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


def mps_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply on the Apple GPU via MPS (the GPU, not the Neural Engine).

    Args:
        a: 2D array [M, K]; converted to contiguous float32
        b: 2D array [K, N]; converted to contiguous float32

    Returns:
        c: 2D float32 array [M, N]. Falls back to np.matmul without MPS.
    """
    if not _HAS_MPS:
        return np.matmul(a, b).astype(np.float32)

    a_t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to("mps")
    b_t = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to("mps")
    c_t = torch.matmul(a_t, b_t)
    return c_t.cpu().numpy()
