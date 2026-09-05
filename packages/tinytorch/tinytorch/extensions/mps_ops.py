"""Apple Silicon Metal / MPS Acceleration Bridge for TinyTorch.

Demonstrates unified memory zero-copy dispatch on Apple M-series chips.
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
    """Returns True if running on Apple Silicon with MPS support."""
    return _HAS_MPS


def mps_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dispatches matrix multiplication to Apple Silicon GPU / Neural Engine via MPS.
    
    Args:
        a: 2D numpy array [M, K]
        b: 2D numpy array [K, N]
        
    Returns:
        c: 2D numpy array [M, N]
    """
    if not _HAS_MPS:
        return np.matmul(a, b)

    a_t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to("mps")
    b_t = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)).to("mps")
    c_t = torch.matmul(a_t, b_t)
    return c_t.cpu().numpy()
