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


def calc_mla_cache_size(
    n_layers,
    kv_lora_rank,
    qk_rope_head_dim,
    seq_len,
    batch_size,
    bytes_per_elem=2,
):
    """
    Calculates the cache size for Multi-Head Latent Attention (MLA).

    MLA compresses keys and values into a single low-rank latent vector per
    token per layer, and carries a decoupled rotary position key alongside it.
    Only those two are resident, so the cache is:

        n_layers * (kv_lora_rank + qk_rope_head_dim) * S * B * precision

    There is no leading factor of two: unlike standard attention, MLA does not
    keep a separate K and V tensor per key-value head. That absence is the whole
    point of the scheme, and it is why an MLA model's resident cache is an order
    of magnitude smaller than a grouped-query model of comparable depth.

    Source: DeepSeek-AI, "DeepSeek-V2/V3 Technical Report" (Multi-Head Latent
    Attention), where the compressed latent has dimension ``kv_lora_rank`` and
    the decoupled rotary key has dimension ``qk_rope_head_dim``.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers.
    kv_lora_rank : int
        Dimension of the compressed key-value latent held per token per layer.
    qk_rope_head_dim : int
        Dimension of the decoupled rotary position key held alongside it.
    seq_len : int
        Total sequence length (context + generated tokens).
    batch_size : int
        Number of parallel requests.
    bytes_per_elem : Quantity or int, optional
        Numerical precision of the cache (defaults to 2 for FP16/BF16).

    Returns
    -------
    Quantity
        Total latent cache size in bytes.
    """
    validate_at_least(n_layers, 1, "n_layers")
    validate_at_least(kv_lora_rank, 1, "kv_lora_rank")
    validate_at_least(qk_rope_head_dim, 1, "qk_rope_head_dim")
    validate_nonnegative(seq_len, "seq_len")
    validate_at_least(batch_size, 1, "batch_size")
    bpe = _ensure_unit(bytes_per_elem, ureg.byte, "bytes_per_elem")
    latent_dim = kv_lora_rank + qk_rope_head_dim
    return (n_layers * latent_dim * seq_len * batch_size * bpe).to(ureg.byte)


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

    This sizes one known sequence length. When ``seq_len`` is a multiple of the
    page size the waste is zero for every page size, so a single length cannot
    show the page-size trade-off. ``calc_expected_paged_kv_tokens`` averages the
    last-page tail over a distribution of request lengths, which is what makes
    page size matter for capacity planning.

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


def calc_capped_exponential_scale(mean_tokens, max_tokens):
    """
    Solves for the exponential scale whose context-capped mean is ``mean_tokens``.

    Request lengths are modeled as ``S = min(X, S_max)`` with
    ``X ~ Exponential(mu)``: short requests are common, long ones are rare, and
    a request that would run past the context window is cut off at ``S_max``.
    The capped mean is ``E[S] = mu * (1 - exp(-S_max / mu))``, which rises
    monotonically from 0 toward ``S_max`` as ``mu`` grows, so exactly one scale
    matches any ``0 < mean_tokens < S_max``. At ``mean_tokens == S_max`` every
    request fills the window (the fixed-length limit) and the scale is infinite.

    The exponential shape is a modeling assumption, chosen because it is the
    maximum-entropy distribution for a positive length with a known mean. It
    is not fitted to a serving trace.

    Parameters
    ----------
    mean_tokens : float
        Mean tokens per request after the cap, ``0 < mean_tokens <= max_tokens``.
    max_tokens : float
        Context-window cap ``S_max`` in tokens (> 0).

    Returns
    -------
    float
        Exponential scale ``mu`` in tokens (``math.inf`` in the fixed-length limit).
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens ({max_tokens}) must be positive")
    if not 0 < mean_tokens <= max_tokens:
        raise ValueError(
            f"mean_tokens ({mean_tokens}) must be in (0, max_tokens={max_tokens}]"
        )
    rho = mean_tokens / max_tokens
    if rho >= 1.0:
        return math.inf
    # Substitute t = S_max / mu, so rho = (1 - e^-t) / t, which decreases from
    # 1 to 0. Because (1 - e^-t) / t < 1 / t, the root lies below t = 1 / rho.
    lo, hi = 0.0, 1.0 / rho
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if -math.expm1(-mid) / mid > rho:
            lo = mid
        else:
            hi = mid
    return max_tokens / (0.5 * (lo + hi))


def calc_expected_paged_kv_tokens(mean_request_tokens, max_seq_len, page_size_tokens=16):
    """
    Expected KV tokens a paged allocator holds per request, and the waste inside.

    PagedAttention (Kwon et al., 2023, Sec. 4.2) hands a request fixed-size
    blocks on demand, so a request of ``S`` tokens holds ``ceil(S / p)`` blocks
    and wastes at most the unused tail of its last block. With request lengths
    ``S = min(X, S_max)`` and ``X`` exponential (``calc_capped_exponential_scale``),
    the expected block count has a closed form from summing tail probabilities:

        E[ceil(S/p)] = sum_{j=0}^{K-1} P(S > j p) = (1 - r^K) / (1 - r)
        r = exp(-p / mu),  K = ceil(S_max / p)

    The expected allocation is ``p * E[ceil(S/p)]`` and the expected internal
    fragmentation is ``1 - mean / allocation``. When ``mu >> p`` the tail
    averages about ``p / 2`` tokens, so larger pages waste proportionally more
    even when ``S_max`` divides evenly by ``p``. In the fixed-length limit
    (``mean_request_tokens == max_seq_len``) the allocation is
    ``ceil(S_max / p) * p``, matching ``calc_paged_kv_cache_size``.

    Parameters
    ----------
    mean_request_tokens : float
        Mean tokens a request fills, ``0 < mean_request_tokens <= max_seq_len``.
    max_seq_len : int
        Longest context a request may reach, in tokens.
    page_size_tokens : int, optional
        Tokens per block (defaults to 16, the vLLM default).

    Returns
    -------
    tuple
        A 2-tuple containing:
        - allocated_tokens (float): Expected KV tokens allocated per request.
        - internal_frag (float): Expected unused fraction of that allocation (0.0 to 1.0).
    """
    validate_at_least(page_size_tokens, 1, "page_size_tokens")
    mu = calc_capped_exponential_scale(mean_request_tokens, max_seq_len)
    max_blocks = math.ceil(max_seq_len / page_size_tokens)
    if math.isinf(mu):
        expected_blocks = float(max_blocks)
    else:
        s = page_size_tokens / mu
        # expm1 keeps the geometric series accurate when p << mu (r close to 1).
        expected_blocks = math.expm1(-max_blocks * s) / math.expm1(-s)
    allocated_tokens = page_size_tokens * expected_blocks
    internal_frag = max(0.0, 1.0 - mean_request_tokens / allocated_tokens)
    return allocated_tokens, internal_frag


def calc_speculative_branch_capacity(
    c_kv_memory,
    l_prefix: int,
    l_branch: int,
    bytes_per_token,
    cow: bool = True,
    page_block_tokens: int = 16,
) -> int:
    """
    Calculate maximum concurrent speculative branches admitted by KV cache memory.

    Under naive replication, each branch duplicates the entire (prefix + branch) context:
        M_per_branch = (l_prefix + l_branch) * bytes_per_token
        K = floor(c_kv_memory / M_per_branch)

    Under Copy-on-Write (PagedAttention), the parent prefix is stored exactly once,
    and each branch allocates branch tokens plus a tail block buffer for internal fragmentation:
        M_prefix = l_prefix * bytes_per_token
        M_per_branch = (l_branch + page_block_tokens) * bytes_per_token
        K = floor((c_kv_memory - M_prefix) / M_per_branch)

    Parameters
    ----------
    c_kv_memory : Quantity
        Total dedicated KV cache memory capacity (e.g. GB).
    l_prefix : int
        Shared parent context length in tokens.
    l_branch : int
        Speculative branch rollout length in tokens.
    bytes_per_token : Quantity
        KV cache memory consumed per token across all layers.
    cow : bool, optional
        Whether Copy-on-Write prefix sharing is enabled (default True).
    page_block_tokens : int, optional
        Number of tokens in a PagedAttention block allocated as tail buffer (default 16).

    Returns
    -------
    int
        Maximum number of concurrent speculative branches.
    """
    import math

    mem_bytes = c_kv_memory.to(ureg.byte).magnitude
    token_bytes = bytes_per_token.to(ureg.byte).magnitude

    if not cow:
        mem_per_branch = (l_prefix + l_branch) * token_bytes
        return math.floor(mem_bytes / mem_per_branch)
    else:
        m_prefix = l_prefix * token_bytes
        m_per_branch = (l_branch + page_block_tokens) * token_bytes
        c_avail = mem_bytes - m_prefix
        if c_avail <= 0:
            return 0
        return math.floor(c_avail / m_per_branch)

