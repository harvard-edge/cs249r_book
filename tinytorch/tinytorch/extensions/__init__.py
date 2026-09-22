"""TinyTorch Hardware Extensions and Accelerators.

Optional, hand-written kernels that take TinyTorch past NumPy onto the
hardware underneath. None of them is needed by the 20 modules.
1. C++ SIMD kernels through ctypes (vectorized for AVX2/NEON, OpenMP if available)
2. A Triton GPU kernel that fuses bias + GELU (needs PyTorch, Triton, NVIDIA GPU)
3. A matrix multiply on the Apple GPU through MPS (needs PyTorch)
Each falls back to NumPy when its hardware or library is missing.
"""

from .simd_ops import simd_matmul, simd_fused_bias_gelu, has_simd_support, simd_build_info
from .triton_gelu import triton_fused_gelu, has_triton_support
from .mps_ops import mps_matmul, has_mps_support

__all__ = [
    "simd_matmul",
    "simd_fused_bias_gelu",
    "has_simd_support",
    "simd_build_info",
    "triton_fused_gelu",
    "has_triton_support",
    "mps_matmul",
    "has_mps_support",
]

# Ecosystem Extensions (Chapter 22)
try:
    from .lora import LoRALinear
    from .loss_scaler import LossScaler
except ImportError:
    pass
