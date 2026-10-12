"""TinyTorch Extensions Ecosystem.

Optional, modular extensions that demonstrate how TinyTorch interfaces with physical
hardware, memory optimization, fine-tuning, and compilation beyond the 20 core modules:

1. LoRALinear: Low-Rank Adaptation for parameter-efficient fine-tuning
2. LossScaler: Mixed-precision loss scaling for FP16 training stability
3. checkpoint: Activation checkpointing / gradient recomputation for memory efficiency
4. compile_graph: Graph capture and kernel fusion compiler
5. simd_matmul, simd_fused_bias_gelu: C++ SIMD kernels through ctypes (AVX2/NEON, OpenMP)
6. triton_fused_gelu: Triton GPU kernel that fuses bias + GELU (NVIDIA GPUs)
7. mps_matmul: Apple Silicon GPU matrix multiply via Metal Performance Shaders
8. cuda_matmul: CUDA SGEMM in three teaching stages (naive, coalesced, tiled) through ctypes

"""

from .simd_ops import simd_matmul, simd_fused_bias_gelu, has_simd_support, simd_build_info
from .triton_gelu import triton_fused_gelu, has_triton_support
from .mps_ops import mps_matmul, has_mps_support
from .cuda_ops import cuda_matmul, has_cuda_support, cuda_build_info
from .lora import LoRALinear
from .loss_scaler import LossScaler
from .checkpoint import checkpoint
from .compile import compile_graph

__all__ = [
    # Systems & Modeling
    "LoRALinear",
    "LossScaler",
    "checkpoint",
    "compile_graph",
    # Hardware Acceleration
    "simd_matmul",
    "simd_fused_bias_gelu",
    "has_simd_support",
    "simd_build_info",
    "triton_fused_gelu",
    "has_triton_support",
    "mps_matmul",
    "has_mps_support",
    "cuda_matmul",
    "has_cuda_support",
    "cuda_build_info",
]
