"""TinyTorch Hardware Extensions and Accelerators.

This module provides plug-and-play accelerated backends and custom kernels
that demonstrate how to extend TinyTorch beyond pure Python/NumPy to real hardware:
1. C++ SIMD Vectorization (AVX2/NEON + OpenMP)
2. OpenAI Triton GPU Kernel JIT (Block-Level SRAM Tiling)
3. Apple Metal / MPS Zero-Copy Unified Memory Dispatch
"""

from .simd_ops import simd_matmul, has_simd_support
from .triton_gelu import triton_fused_gelu, has_triton_support
from .mps_ops import mps_matmul, has_mps_support

__all__ = [
    "simd_matmul",
    "has_simd_support",
    "triton_fused_gelu",
    "has_triton_support",
    "mps_matmul",
    "has_mps_support",
]
