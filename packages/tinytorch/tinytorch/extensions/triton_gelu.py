"""OpenAI Triton Fused Bias + GELU Kernel for TinyTorch.

Demonstrates block-level GPU programming with automatic SRAM tiling.
Falls back seamlessly to CPU if Triton / CUDA GPU is unavailable.
"""

import numpy as np

_HAS_TRITON = False
try:
    import torch
    import triton
    import triton.language as tl
    if torch.cuda.is_available():
        _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _fused_bias_gelu_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        total_elements,
        inner_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        # Compute bias offset (modulo inner_dim)
        bias_offsets = offsets % inner_dim

        # Zero-overhead coalesced DRAM -> SRAM load
        x = tl.load(x_ptr + offsets, mask=mask)
        bias = tl.load(bias_ptr + bias_offsets, mask=mask)

        # Fused arithmetic inside register file
        val = x + bias
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        inner = sqrt_2_over_pi * (val + coeff * val * val * val)
        tanh_val = tl.extra.cuda.libdevice.tanh(inner)
        out = 0.5 * val * (1.0 + tanh_val)

        # Write back directly to HBM without intermediate activation traffic
        tl.store(out_ptr + offsets, out, mask=mask)


def has_triton_support() -> bool:
    """Returns True if Triton and an NVIDIA GPU are available."""
    return _HAS_TRITON


def triton_fused_gelu(x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Computes Fused Bias + GELU on GPU via Triton JIT.
    
    Args:
        x: Input numpy array of shape [..., D]
        bias: Bias vector of shape [D]
        
    Returns:
        Numpy array of identical shape with fused GELU applied.
    """
    if not _HAS_TRITON:
        # Fallback to pure numpy CPU implementation
        val = x + bias
        inner = np.sqrt(2.0 / np.pi) * (val + 0.044715 * np.power(val, 3))
        return 0.5 * val * (1.0 + np.tanh(inner))

    x_torch = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).cuda()
    bias_torch = torch.from_numpy(np.ascontiguousarray(bias, dtype=np.float32)).cuda()
    out_torch = torch.empty_like(x_torch)

    total_elements = x_torch.numel()
    inner_dim = bias_torch.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _fused_bias_gelu_kernel[grid](
        x_torch,
        bias_torch,
        out_torch,
        total_elements,
        inner_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out_torch.cpu().numpy()
