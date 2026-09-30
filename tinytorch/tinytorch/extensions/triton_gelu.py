# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the Triton fused bias+GELU kernel for Milestone 07,
# a bundled reference point students time their own Module 17 kernels
# against; no src/ module or student notebook exports to it.
"""OpenAI Triton Fused Bias + GELU Kernel for TinyTorch.

Adds a bias and applies GELU in one GPU kernel, so the intermediate x + bias
never makes a round trip through GPU memory. Each Triton program handles one
block of BLOCK_SIZE elements. This is an optional, hand-written extension: it
is tracked in git as-is and is not generated from src/.

It needs PyTorch, Triton, and an NVIDIA GPU; TinyTorch itself never imports
PyTorch. Without them, triton_fused_gelu computes the same formula in NumPy.
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

        # Adjacent programs read adjacent addresses, so the loads coalesce
        x = tl.load(x_ptr + offsets, mask=mask)
        bias = tl.load(bias_ptr + bias_offsets, mask=mask)

        # The whole formula stays in registers
        val = x + bias
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        inner = sqrt_2_over_pi * (val + coeff * val * val * val)
        # tanh(z) = 2*sigmoid(2z) - 1; tl.sigmoid exists in every Triton
        # release, while the libdevice tanh import path has moved between them.
        tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        out = 0.5 * val * (1.0 + tanh_val)

        # One write of the result; x + bias is never stored
        tl.store(out_ptr + offsets, out, mask=mask)


def has_triton_support() -> bool:
    """Returns True if Triton and an NVIDIA GPU are available."""
    return _HAS_TRITON


def _is_tensor(obj):
    return obj is not None and type(obj).__name__ == "Tensor"


def triton_fused_gelu(x, bias=None):
    """GELU(x + bias) in one Triton kernel on an NVIDIA GPU (tanh approximation).

    Args:
        x: array or Tensor [..., D]
        bias: optional vector or Tensor [D]

    Returns:
        float32 array or Tensor with the shape of x. Computed in NumPy without Triton.
    """
    is_tensor = _is_tensor(x) or (bias is not None and _is_tensor(bias))
    x_arr = x.data if _is_tensor(x) else x
    bias_arr = bias.data if (bias is not None and _is_tensor(bias)) else bias

    if not _HAS_TRITON:
        val = np.asarray(x_arr, dtype=np.float32)
        if bias_arr is not None:
            val = val + np.asarray(bias_arr, dtype=np.float32)
        inner = np.sqrt(2.0 / np.pi) * (val + 0.044715 * np.power(val, 3))
        out = (0.5 * val * (1.0 + np.tanh(inner))).astype(np.float32)
        if is_tensor:
            from tinytorch.core.tensor import Tensor
            return Tensor(out)
        return out

    x_torch = torch.from_numpy(np.ascontiguousarray(x_arr, dtype=np.float32)).cuda()
    if bias_arr is not None:
        bias_torch = torch.from_numpy(np.ascontiguousarray(bias_arr, dtype=np.float32)).cuda()
    else:
        bias_torch = torch.zeros(x_torch.shape[-1], device=x_torch.device, dtype=torch.float32)
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
    out = out_torch.cpu().numpy()
    if is_tensor:
        from tinytorch.core.tensor import Tensor
        return Tensor(out)
    return out
