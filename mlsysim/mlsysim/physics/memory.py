"""Model and activation memory accounting."""

from __future__ import annotations

import math
import pint

from mlsysim.core.units import ureg, MB
from mlsysim.core._validation import validate_at_least, validate_nonnegative

from ._units import _ensure_unit


def model_memory(params, bytes_per_param, unit=MB):
    """
    Calculates the memory footprint to store model weights in a requested unit.

    Parameters
    ----------
    params : Quantity or int
        Total number of parameters in the model.
    bytes_per_param : Quantity or int
        Size of each parameter in bytes (e.g., 2 bytes for FP16).
    unit : pint.Unit, optional
        The desired output unit (defaults to MB).

    Returns
    -------
    float
        The calculated memory footprint magnitude in the requested unit.
    """
    if isinstance(params, ureg.Quantity):
        try:
            param_count = params.to(ureg.count).magnitude
        except pint.DimensionalityError:
            raise pint.DimensionalityError(
                params.units,
                ureg.count,
                extra_msg=(
                    f" in model_memory() — params must be in param/count units, "
                    f"got {params.units}"
                ),
            )
    else:
        param_count = params

    if isinstance(bytes_per_param, ureg.Quantity):
        try:
            bpp = bytes_per_param.to(ureg.byte).magnitude
        except pint.DimensionalityError:
            raise pint.DimensionalityError(
                bytes_per_param.units,
                ureg.byte,
                extra_msg=(
                    f" in model_memory() — bytes_per_param must be byte units, "
                    f"got {bytes_per_param.units}"
                ),
            )
    else:
        bpp = bytes_per_param

    total_bytes = param_count * bpp * ureg.byte
    return total_bytes.to(unit).magnitude


def calc_activation_memory(
    n_layers,
    seq_len,
    batch_size,
    hidden_dim,
    n_heads=None,
    precision_bytes=2,
    strategy="selective",
):
    """
    Estimates the activation memory required for a Transformer model during training.

    Implements the per-layer analytical bounds of Korthikanti et al. (2023),
    "Reducing Activation Recomputation in Large Transformer Models", Sec. 4.1,
    whose constants are expressed in BYTES with FP16 (2-byte) activations baked in:

    - ``none``      (no recomputation):  ``s*b*h*34 + 5*a*s^2*b``
    - ``selective`` (recompute attention matrices): ``s*b*h*34``
    - ``full``      (recompute everything; keep layer inputs): ``s*b*h*2``

    where ``s`` = sequence length, ``b`` = microbatch size, ``h`` = hidden
    dimension, and ``a`` = attention heads. The quadratic ``5*a*s^2*b`` term is
    the attention softmax/dropout/score storage — dominant at long sequence
    lengths — and is exactly what selective recomputation discards. Activations
    stored at other precisions scale the FP16-based constants by
    ``precision_bytes / 2``.

    (Audit fix 2026-06-06: the previous implementation used 34/10/2 as
    precision-free coefficients AND multiplied by ``precision_bytes``, double-
    counting the FP16 width at every strategy; its ``selective=10`` matched no
    published convention; and the attention term was missing entirely.)

    Parameters
    ----------
    n_layers : int
        Number of transformer layers per device (after pipeline parallelism).
    seq_len : int
        Sequence length.
    batch_size : int
        Microbatch size per device.
    hidden_dim : int
        Hidden dimension of the model.
    n_heads : int, optional
        Number of attention heads. REQUIRED for ``strategy='none'`` (the
        quadratic attention term needs it); unused otherwise.
    precision_bytes : float, optional
        Bytes per activation element (default 2, i.e. FP16 — the paper's
        convention).
    strategy : str, optional
        Recomputation strategy: 'none', 'selective' (default), or 'full'.

    Returns
    -------
    Quantity
        The total estimated activation memory in bytes.
    """
    validate_at_least(n_layers, 1, "n_layers")
    s, b, h = seq_len, batch_size, hidden_dim
    precision_scale = precision_bytes / 2  # Korthikanti constants are FP16 bytes
    if strategy == "full":
        bytes_per_layer = 2 * s * b * h * precision_scale
    elif strategy == "selective":
        bytes_per_layer = 34 * s * b * h * precision_scale
    elif strategy == "none":
        if n_heads is None:
            raise ValueError(
                "strategy='none' requires n_heads: the dominant 5*a*s^2*b "
                "attention term scales with the head count (Korthikanti et al. "
                "2023, Sec. 4.1)."
            )
        bytes_per_layer = (34 * s * b * h + 5 * n_heads * s * s * b) * precision_scale
    else:
        raise ValueError(
            f"Unknown activation strategy {strategy!r}; expected 'none', "
            "'selective', or 'full'."
        )
    return (n_layers * bytes_per_layer) * ureg.byte


def calc_checkpoint_size(n_params, bytes_per_param=14):
    """
    Calculates the total storage size required for a training checkpoint.

    Parameters
    ----------
    n_params : Quantity
        Total number of parameters in the model.
    bytes_per_param : Quantity or int, optional
        Bytes required per parameter for optimizer state + weights.
        For mixed-precision Adam, this is typically 14 bytes (FP32 master weights,
        FP32 momentum, FP32 variance, FP16 parameters). Defaults to 14.

    Returns
    -------
    Quantity
        Total checkpoint size in bytes.
    """
    bpp = _ensure_unit(bytes_per_param, ureg.byte, "bytes_per_param")
    return (n_params * bpp).to(ureg.byte)


def calc_kv_cache_size(
    n_layers,
    n_heads,
    head_dim,
    seq_len,
    batch_size,
    bytes_per_elem=2,
    kv_precision_bytes=None,
):
    """
    Calculates the KV cache memory size for autoregressive inference.

    The KV cache stores Key and Value tensors for all previous tokens to avoid
    recomputing them. The size is strictly: 2 * L * H * d * S * B * precision.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers.
    n_heads : int
        Number of Key/Value attention heads (accounts for MQA/GQA).
    head_dim : int
        Dimension of a single attention head.
    seq_len : int
        Total sequence length (context + generated tokens).
    batch_size : int
        Number of parallel requests.
    bytes_per_elem : Quantity or int, optional
        Numerical precision of the cache (defaults to 2 for FP16/BF16).
    kv_precision_bytes : Quantity or int, optional
        Override for specific KV cache quantization (e.g., INT8 KV cache).

    Returns
    -------
    Quantity
        Total KV cache size in bytes.
    """
    validate_at_least(n_layers, 1, "n_layers")
    validate_at_least(n_heads, 1, "n_heads")
    validate_at_least(head_dim, 1, "head_dim")
    validate_nonnegative(seq_len, "seq_len")
    validate_at_least(batch_size, 1, "batch_size")
    effective_bpe = kv_precision_bytes if kv_precision_bytes is not None else bytes_per_elem
    bpe = _ensure_unit(
        effective_bpe,
        ureg.byte,
        "kv_precision_bytes" if kv_precision_bytes is not None else "bytes_per_elem",
    )
    # Leading 2 = the separate K and V tensors cached per layer per head.
    return (2 * n_layers * n_heads * head_dim * seq_len * batch_size * bpe).to(ureg.byte)


def calc_paged_kv_cache_size(
    n_layers,
    n_heads,
    head_dim,
    seq_len,
    batch_size,
    page_size_tokens=16,
    bytes_per_elem=2,
):
    """
    Calculates KV cache size accounting for PagedAttention fragmentation.

    PagedAttention (Kwon et al., 2023) allocates KV cache in fixed-size blocks
    (pages). This eliminates external fragmentation but introduces internal
    fragmentation in the final allocated page.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers.
    n_heads : int
        Number of Key/Value attention heads.
    head_dim : int
        Dimension of a single attention head.
    seq_len : int
        Current sequence length.
    batch_size : int
        Number of parallel requests.
    page_size_tokens : int, optional
        Number of tokens per allocated page block (defaults to 16).
    bytes_per_elem : Quantity or int, optional
        Numerical precision (defaults to 2).

    Returns
    -------
    tuple
        A 2-tuple containing:
        - size (Quantity): Total allocated KV cache size in bytes.
        - frag_pct (float): Internal memory fragmentation (0.0 to 1.0).
    """
    validate_at_least(n_layers, 1, "n_layers")
    validate_at_least(n_heads, 1, "n_heads")
    validate_at_least(head_dim, 1, "head_dim")
    validate_nonnegative(seq_len, "seq_len")
    validate_at_least(batch_size, 1, "batch_size")
    validate_at_least(page_size_tokens, 1, "page_size_tokens")
    bpe = _ensure_unit(bytes_per_elem, ureg.byte, "bytes_per_elem")
    # Allocation is page-granular: round the sequence up to whole pages. The
    # slack in the final partially-filled page is the only waste PagedAttention
    # leaves (internal fragmentation), bounded by one page per sequence.
    padded_seq_len = math.ceil(seq_len / page_size_tokens) * page_size_tokens
    internal_frag = max(0, padded_seq_len - seq_len)
    frag_pct = internal_frag / padded_seq_len if padded_seq_len > 0 else 0.0
    size = (
        2 * n_layers * n_heads * head_dim * padded_seq_len * batch_size * bpe
    ).to(ureg.byte)
    return size, frag_pct
