# Hand-maintained extension, NOT generated. This file is not produced by
# `tito dev export` or `tito module complete`; it is tracked in git and
# edited directly here. It is the Triton fused bias+GELU kernel for Milestone 07,
# a bundled reference point students time their own Module 17 kernels
# against; no src/ module or student notebook exports to it.
"""OpenAI Triton Fused Bias + GELU for TinyTorch: an example extension.

What it is: one GPU kernel, written in Triton, that adds a bias and applies
GELU in a single pass, so the intermediate x + bias never makes a round trip
through GPU memory. Hand-written and tracked in git as-is, not generated from
src/.

Why it exists: computed as two operations, x + bias is written to GPU memory
as a full intermediate tensor and read back for the GELU, so each element
crosses the memory bus four times (read x, write t, read t, write y) plus the
small bias reads. Fused, it crosses twice (read x, write y): about half the
traffic for an operation that is purely memory-bound, since a GELU does only
a handful of FLOPs per element.

How it is triggered: only when you call it. Nothing in TinyTorch calls this
file for you; no activation layer routes here.

    from tinytorch.extensions import triton_fused_gelu, has_triton_support
    y = triton_fused_gelu(x, bias)  # x: [..., D], bias: [D]; arrays or Tensors
    print(has_triton_support())     # True if that ran on the GPU

    `tito milestone run 07` makes the same call for you and times it
    against your own Module 17 fused_gelu.

    On import:     tries `import torch` and `import triton` and asks whether
                   a CUDA device exists; the kernel is defined only if all
                   three are there. tinytorch/extensions/__init__.py imports
                   this file, so this happens as soon as you import
                   tinytorch.extensions.
    On first call: Triton compiles the kernel for this GPU.
    On every call: copy in, launch, copy out; or the same formula in NumPy
                   when Triton or an NVIDIA GPU is missing.

How it works:

1. Copy in. The inputs are moved to the GPU as torch tensors, and an output
   buffer of the same shape is allocated there.
2. Launch. Triton runs one program per block of BLOCK_SIZE = 1024 elements.
   Each program loads its block of x (adjacent programs read adjacent
   addresses, so the loads coalesce) and the matching bias entries, keeps
   x + bias and the whole GELU formula in registers, and stores only y.
3. Copy out. .cpu() waits for the kernel and copies y back.

The kernel computes tanh as 2 * sigmoid(2z) - 1, an exact identity, because
tl.sigmoid is stable across Triton releases and the tanh import path is not.

What it leaves out: gradients. The result is a new Tensor with no gradient
history, so this is a benchmark path, not a drop-in activation for training.
It needs PyTorch, Triton, and an NVIDIA GPU; TinyTorch itself never imports
PyTorch.

Where to go next: fuse more of the layer, such as the matmul that feeds this
activation, which is the idea behind FlashAttention's tiling.
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
