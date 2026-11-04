"""Model and activation memory accounting."""

from __future__ import annotations

import math
import pint

from mlsysim.core.units import ureg, MB, BYTES_FP32
from mlsysim.core._validation import validate_at_least, validate_nonnegative, validate_positive

from ._units import _ensure_unit


def optimizer_state_bytes(optimizer: str):
    """Return optimizer storage per trainable parameter, excluding gradients.

    Parameters
    ----------
    optimizer : str
        Case-insensitive storage convention: adam, adamw, sgd, or none.

    Returns
    -------
    Quantity
        Bytes per parameter; multiply by a parameter count to obtain total storage.

    Notes
    -----
    Equation: B = c * b_FP32, with c = 3 for Adam/AdamW, 1 for SGD, and 0 for none.

    Assumptions: Adam retains an FP32 master weight and two FP32 moments. SGD retains a
    master weight but no momentum; this is an explicit mixed-precision convention.

    Excludes: Model weights at execution precision, gradients, activations, allocator
    reserve, and optimizer workspaces.

    References
    ----------
    Derivation: count the stored FP32 tensors per parameter. The Adam convention follows
    the model-state accounting in Rajbhandari et al. (2020), "ZeRO: Memory Optimizations
    Toward Training Trillion Parameter Models," https://arxiv.org/abs/1910.02054
    (bibliography key: rajbhandari2020). The SGD convention is declared here, not
    inferred from every SGD implementation.

    Examples
    --------
    >>> optimizer_state_bytes("adamw") == 3 * BYTES_FP32
    True
    """
    copies = {"adam": 3, "adamw": 3, "sgd": 1, "none": 0}
    key = optimizer.lower()
    if key not in copies:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Supported: {', '.join(sorted(copies))}")
    return copies[key] * BYTES_FP32


def model_memory(params, bytes_per_param, unit=MB):
    """Calculate weight storage in a requested memory unit.

    Parameters
    ----------
    params : Quantity or int
        Nonnegative parameter count, dimensionless or in param/count units.
    bytes_per_param : Quantity or float
        Nonnegative bytes per stored parameter; quantities must have byte units.
    unit : pint.Unit, optional
        Memory unit for the returned magnitude; defaults to decimal MB.

    Returns
    -------
    float
        Weight-memory magnitude in unit; unlike most physics helpers, this API returns
        no attached unit.

    Notes
    -----
    Equation: M = P * b, converted from bytes to unit.

    Assumptions: Each parameter occupies the same storage width; the caller supplies any
    quantized effective width.

    Excludes: Activations, gradients, optimizer state, KV cache, metadata, padding, and
    packing overhead.

    References
    ----------
    Derivation: summing P equal-width stored elements gives P times the bytes per
    element.

    Examples
    --------
    >>> model_memory(1000, 2, unit=ureg.byte) == 1000 * 2
    True
    """
    if isinstance(params, ureg.Quantity):
        try:
            param_count = params.to(ureg.count).magnitude
        except pint.DimensionalityError:
            raise pint.DimensionalityError(
                params.units,
                ureg.count,
                extra_msg=(f" in model_memory() — params must be in param/count units, got {params.units}"),
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
                extra_msg=(f" in model_memory() — bytes_per_param must be byte units, got {bytes_per_param.units}"),
            )
    else:
        bpp = bytes_per_param

    total_bytes = param_count * bpp * ureg.byte
    return total_bytes.to(unit).magnitude


# Korthikanti et al. (2023), Table 2: bytes one transformer layer saves for the
# backward pass, per token and per unit of width (s*b*h), in 16-bit storage.
# The score term saves three copies of every attention entry: the softmax
# probabilities (2 B), the dropout mask (1 B), and the probabilities after
# dropout (2 B), 5 B per entry, so it is 5*a*s^2*b. Full recomputation keeps
# only each layer's 16-bit input, 2 B per element. (Named 2026-10-20.)
KORTHIKANTI_PER_TOKEN_BYTES = 34
SCORE_SOFTMAX_BYTES = 2
SCORE_DROPOUT_MASK_BYTES = 1
SCORE_DROPPED_BYTES = 2
KORTHIKANTI_SCORE_BYTES = SCORE_SOFTMAX_BYTES + SCORE_DROPOUT_MASK_BYTES + SCORE_DROPPED_BYTES
KORTHIKANTI_LAYER_INPUT_BYTES = 2


def calc_activation_memory(
    n_layers,
    seq_len,
    batch_size,
    hidden_dim,
    n_heads=None,
    precision_bytes=2,
    strategy="selective",
):
    """Estimate saved Transformer training activations per device.

    Parameters
    ----------
    n_layers : int
        Positive number of layers resident on the device after pipeline partitioning.
    seq_len : int
        Positive tokens per sequence.
    batch_size : int
        Positive microbatch sequence count per device.
    hidden_dim : int
        Positive hidden width in elements.
    n_heads : int, optional
        Positive attention-head count; required for strategy="none" and otherwise
        unused.
    precision_bytes : float, optional
        Positive bytes per activation element; default 2 scales the FP16 accounting.
    strategy : str, optional
        Saved-state policy: none, selective (default), or full recomputation.

    Returns
    -------
    Quantity
        Total saved activation storage in bytes.

    Notes
    -----
    Equation: M = L * (b/2) * A. At FP16, A = ``34*S*B*H`` + ``5*N_heads*S**2*B`` for none,
    ``34*S*B*H`` for selective, and ``2*S*B*H`` for full.

    Assumptions: Equal layer shapes and the FP16 saved-tensor convention of Korthikanti
    et al. Selective recomputation discards attention matrices; full recomputation
    retains layer inputs. Other precisions scale all coefficients uniformly.

    Excludes: Weights, gradients, optimizer state, kernel scratch space, allocator
    reserve, sequence/tensor-parallel sharding, and schedule-dependent simultaneous
    microbatch residency.

    The attention coefficient counts softmax probabilities, a dropout mask, and dropped
    probabilities. The FP16 byte widths are already included in the coefficients;
    multiplying them by precision_bytes without dividing by two would count precision
    twice.

    References
    ----------
    Korthikanti et al. (2023), "Reducing Activation Recomputation in Large Transformer
    Models," https://arxiv.org/abs/2205.05198 (bibliography key: korthikanti2023). The
    linear precision rescaling is a modeling convention.

    Examples
    --------
    >>> full = calc_activation_memory(2, 8, 1, 16, strategy="full")
    >>> selective = calc_activation_memory(2, 8, 1, 16, strategy="selective")
    >>> bool(full < selective)
    True
    """
    validate_at_least(n_layers, 1, "n_layers")
    validate_at_least(seq_len, 1, "seq_len")
    validate_at_least(batch_size, 1, "batch_size")
    validate_at_least(hidden_dim, 1, "hidden_dim")
    validate_positive(precision_bytes, "precision_bytes")
    if n_heads is not None:
        validate_at_least(n_heads, 1, "n_heads")
    s, b, h = seq_len, batch_size, hidden_dim
    precision_scale = precision_bytes / 2  # Korthikanti constants are FP16 bytes
    if strategy == "full":
        bytes_per_layer = KORTHIKANTI_LAYER_INPUT_BYTES * s * b * h * precision_scale
    elif strategy == "selective":
        bytes_per_layer = KORTHIKANTI_PER_TOKEN_BYTES * s * b * h * precision_scale
    elif strategy == "none":
        if n_heads is None:
            raise ValueError(
                "strategy='none' requires n_heads: the dominant 5*a*s^2*b "
                "attention term scales with the head count (Korthikanti et al. "
                "2023, Sec. 4.1)."
            )
        bytes_per_layer = (
            KORTHIKANTI_PER_TOKEN_BYTES * s * b * h + KORTHIKANTI_SCORE_BYTES * n_heads * s * s * b
        ) * precision_scale
    else:
        raise ValueError(f"Unknown activation strategy {strategy!r}; expected 'none', 'selective', or 'full'.")
    return (n_layers * bytes_per_layer) * ureg.byte


def calc_checkpoint_memory_units(n_layers: int, checkpoints: int) -> float:
    """Estimate evenly spaced checkpoint memory in layer-equivalent units.

    Parameters
    ----------
    n_layers : int
        Positive depth L of an equal-sized layer chain.
    checkpoints : int
        Positive checkpoint count k; use k <= L for a physical partition.

    Returns
    -------
    float
        Activation footprint measured in units of one layer, not bytes.

    Notes
    -----
    Equation: M_layers = k + L/k.

    Assumptions: The forward pass keeps k checkpoints while backward recomputes one
    segment of L/k equal-sized layers at a time.

    Excludes: Integer segment rounding, unequal layer sizes, tensor parallelism, and
    recomputation runtime.

    References
    ----------
    Chen et al. (2016), "Training Deep Nets with Sublinear Memory Cost,"
    https://arxiv.org/abs/1604.06174 (bibliography key: chen2016training). The
    equal-segment approximation yields k + L/k.

    Examples
    --------
    >>> calc_checkpoint_memory_units(16, 4) == 4 + 16 / 4
    True
    """
    validate_at_least(n_layers, 1, "n_layers")
    validate_at_least(checkpoints, 1, "checkpoints")
    return checkpoints + n_layers / checkpoints


def calc_checkpoint_size(n_params, bytes_per_param=14):
    """Calculate persistent training-checkpoint storage.

    Parameters
    ----------
    n_params : Quantity or int
        Nonnegative parameter count; quantities must be dimensionless/count-compatible.
    bytes_per_param : Quantity or float, optional
        Stored bytes per parameter, default 14: an FP16 weight plus an FP32 master
        weight and two FP32 moments.

    Returns
    -------
    Quantity
        Checkpoint storage in bytes.

    Notes
    -----
    Equation: M_checkpoint = P * b_checkpoint.

    Assumptions: Every parameter has the same saved-state footprint; the default stores
    mixed-precision Adam state.

    Excludes: Gradients, activations, dataset position, random-number state,
    serialization metadata, filesystem replication, and compression.

    References
    ----------
    Derivation: multiply parameter count by the sum of persisted tensor widths. The
    default is 2 + 4 + 4 + 4 bytes per parameter, not a measured file size.

    Examples
    --------
    >>> calc_checkpoint_size(1000).m_as("byte") == 1000 * 14
    True
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
    """Calculate resident key/value cache storage.

    Parameters
    ----------
    n_layers : int
        Positive number of decoder layers.
    n_heads : int
        Positive key/value-head count, not query-head count; use the GQA/MQA count.
    head_dim : int
        Positive elements per key/value head.
    seq_len : int
        Nonnegative resident token count per request, including prompt and generated
        tokens.
    batch_size : int
        Positive number of requests with this same resident length.
    bytes_per_elem : Quantity or float, optional
        Bytes per cache element; default 2 for FP16/BF16.
    kv_precision_bytes : Quantity or float, optional
        Cache-specific byte width overriding bytes_per_elem when supplied.

    Returns
    -------
    Quantity
        Combined K and V storage in bytes.

    Notes
    -----
    Equation: M_KV = 2 * L * H_KV * d_head * S * B * b.

    Assumptions: Each layer stores separate K and V vectors for every resident token and
    KV head. Both vectors use the effective byte width.

    Excludes: Page rounding, quantization scales, shared-prefix deduplication, allocator
    reserve, and runtime workspaces.

    References
    ----------
    Derivation: multiply the tensor dimensions for K and V and add their equal
    footprints. The leading two counts tensors, not bytes of precision.

    Examples
    --------
    >>> small = calc_kv_cache_size(2, 4, 8, 16, 1)
    >>> calc_kv_cache_size(2, 4, 8, 32, 1) == 2 * small
    True
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
    """Calculate resident Multi-Head Latent Attention cache storage.

    Parameters
    ----------
    n_layers : int
        Positive decoder-layer count.
    kv_lora_rank : int
        Positive elements in the compressed key/value latent per token and layer.
    qk_rope_head_dim : int
        Positive elements in the decoupled rotary key per token and layer.
    seq_len : int
        Nonnegative resident tokens per request.
    batch_size : int
        Positive request count with the same resident length.
    bytes_per_elem : Quantity or float, optional
        Bytes per cached element, default 2.

    Returns
    -------
    Quantity
        Compressed latent plus rotary-key cache in bytes.

    Notes
    -----
    Equation: M_MLA = L * (r_KV + d_RoPE) * S * B * b.

    Assumptions: The latent and rotary key are the only resident cache vectors. K and V
    are reconstructed/absorbed at use time, so no separate leading factor of two
    applies.

    Excludes: Reconstruction computation, quantization metadata, page rounding, prefix
    sharing, and attention workspace.

    References
    ----------
    DeepSeek-AI (2024), "DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model," https://arxiv.org/abs/2405.04434 (bibliography
    key: deepseek2024v2).

    Examples
    --------
    >>> calc_mla_cache_size(2, 8, 4, 16, 1).m_as("byte") == 2 * (8 + 4) * 16 * 2
    True
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
    """Calculate page-rounded KV storage and internal fragmentation.

    Parameters
    ----------
    n_layers : int
        Positive decoder-layer count.
    n_heads : int
        Positive key/value-head count.
    head_dim : int
        Positive elements per key/value head.
    seq_len : int
        Nonnegative resident tokens per request.
    batch_size : int
        Positive request count, each with the same length.
    page_size_tokens : int, optional
        Positive tokens per physical page, default 16.
    bytes_per_elem : Quantity or float, optional
        Bytes per cache element, default 2.

    Returns
    -------
    tuple
        Allocated bytes as a Quantity and unused fraction of allocated bytes as a float;
        empty sequences return zero for both.

    Notes
    -----
    Equation: S_alloc = ``ceil(S/p)*p``; M = ``2*L*H_KV*d*S_alloc*B*b``; f = (S_alloc-S)/S_alloc
    for nonempty allocation.

    Assumptions: Each request receives whole pages and its final page is the only source
    of internal slack.

    Excludes: External fragmentation, block-table metadata, prefix sharing, and reserve
    for future output tokens.

    A single length divisible by the page size has no internal waste. Use
    calc_expected_paged_kv_tokens when the question concerns a distribution of lengths.

    References
    ----------
    Kwon et al. (2023), "Efficient Memory Management for Large Language Model Serving
    with PagedAttention," https://arxiv.org/abs/2309.06180 (bibliography key: kwon2023).
    Page rounding here is deterministic tensor accounting.

    Examples
    --------
    >>> size, waste = calc_paged_kv_cache_size(2, 4, 8, 16, 1, page_size_tokens=8)
    >>> size == calc_kv_cache_size(2, 4, 8, 16, 1) and waste == 0
    True
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
    size = calc_kv_cache_size(n_layers, n_heads, head_dim, padded_seq_len, batch_size, bpe)
    return size, frag_pct


def calc_capped_exponential_scale(mean_tokens, max_tokens):
    """Solve for the exponential scale of a context-capped length model.

    Parameters
    ----------
    mean_tokens : float
        Mean resident tokens after capping, in (0, max_tokens].
    max_tokens : float
        Positive context cap in tokens.

    Returns
    -------
    float
        Uncapped exponential scale in tokens; infinity when every request reaches the
        cap.

    Notes
    -----
    Equation: mean_tokens = mu * (1 - exp(-max_tokens/mu)).

    Assumptions: Length is min(X, max_tokens), where X is a continuous exponential
    variable. Bisection solves the monotone mean relation; the shape is assumed, not
    fitted to a trace.

    Excludes: Discrete token effects and measured request-length distributions.

    References
    ----------
    Derivation: integrate the survival function P(X > s) = exp(-s/mu) from zero to
    max_tokens. The capped mean rises monotonically from zero to the cap.

    Examples
    --------
    >>> math.isinf(calc_capped_exponential_scale(16, 16))
    True
    >>> mu = calc_capped_exponential_scale(8, 32)
    >>> math.isclose(mu * (1 - math.exp(-32 / mu)), 8)
    True
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens ({max_tokens}) must be positive")
    if not 0 < mean_tokens <= max_tokens:
        raise ValueError(f"mean_tokens ({mean_tokens}) must be in (0, max_tokens={max_tokens}]")
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
    """Estimate page allocation under a capped exponential request model.

    Parameters
    ----------
    mean_request_tokens : float
        Positive mean resident tokens, at most max_seq_len.
    max_seq_len : int
        Positive cap on resident tokens per request.
    page_size_tokens : int, optional
        Positive tokens per page; default 16.

    Returns
    -------
    tuple
        Expected allocated tokens per request and expected unused fraction of that
        allocation, both floats.

    Notes
    -----
    Equation: E[pages] = ``sum(exp(-j*p/mu)``, j=0..K-1) = ``(1-r**K)/(1-r)``, with
    r=exp(-p/mu), K=ceil(S_max/p). Allocation is ``p*E``[pages], and waste is
    1-mean/allocation.

    Assumptions: Resident length is min(X, S_max), with an exponential X calibrated only
    to the supplied capped mean. At the fixed-length limit, allocate ``ceil(S_max/p)*p``
    tokens.

    Excludes: External fragmentation, block metadata, time-varying occupancy, prefix
    sharing, and departures from the assumed length distribution.

    References
    ----------
    Derivation: the tail-sum identity for the integer page count gives the finite
    geometric series. Kwon et al. (2023), "Efficient Memory Management for Large
    Language Model Serving with PagedAttention," https://arxiv.org/abs/2309.06180
    (bibliography key: kwon2023), motivates page allocation; the exponential request
    model is an additional assumption.

    Examples
    --------
    >>> allocation, waste = calc_expected_paged_kv_tokens(16, 16, page_size_tokens=8)
    >>> allocation == 16 and waste == 0
    True
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


# --- Decoder weight matrices (added 2026-10-20) ------------------------------


def calc_decoder_matmul_weights(layers, hidden_dim, vocab_size, mlp_ratio=4):
    """Count GPT-style decoder matrix weights by location.

    Parameters
    ----------
    layers : int
        Positive decoder-block count L.
    hidden_dim : int
        Positive hidden width d.
    vocab_size : int
        Positive output-vocabulary size V.
    mlp_ratio : float, optional
        Positive intermediate-width multiple r; default 4.

    Returns
    -------
    dict
        Counts: block weights across all layers, block_matrices across all layers, and
        output projection weights.

    Notes
    -----
    Equation: block = ``L*(4*d**2`` + ``2*r*d**2)``; block_matrices = ``4*L``; output = ``V*d``.

    Assumptions: Each block has one fused QKV matrix, one attention output matrix, and
    two MLP matrices. The output projection is reported separately.

    Excludes: Biases, normalization scales, position embeddings, GQA/MLA geometry, and
    gated-MLP extra matrices.

    References
    ----------
    Derivation: sum the rectangular matrix element counts ``d*(3*d)``, ``d*d``, ``d*(r*d)``,
    ``(r*d)*d``, and ``V*d``. Fusing QKV changes matrix count, not parameter count.

    Examples
    --------
    >>> result = calc_decoder_matmul_weights(2, 8, 16)
    >>> result["block"] == 2 * (4 + 2 * 4) * 8**2
    True
    """
    d = hidden_dim
    from .transformer import calc_transformer_block_weights

    _, _, per_block = calc_transformer_block_weights(d, ffn_multiple=mlp_ratio)
    return {"block": layers * per_block, "block_matrices": 4 * layers, "output": vocab_size * d}


# --- KV cache residency: fit, blocks, sharing (added 2026-10-20) -------------


def calc_kv_attention_design(heads, kv_heads=None, kv_lora_rank=None) -> str:
    """Classify a cache geometry as MLA, GQA, or MHA.

    Parameters
    ----------
    heads : int
        Positive query-head count for a physical attention configuration.
    kv_heads : int, optional
        Key/value-head count; a positive count below heads selects GQA.
    kv_lora_rank : int, optional
        Compressed KV latent width; any nonzero value selects MLA before checking head
        counts.

    Returns
    -------
    str
        MLA, GQA, or MHA; absent geometry defaults to MHA.

    Notes
    -----
    Equation: design = MLA if kv_lora_rank is truthy, else GQA if 0 < kv_heads < heads,
    else MHA.

    Assumptions: This is a routing convention based on declared geometry, not an
    inference from weights. Inputs describe an already valid model.

    Excludes: Architectural validity checks, cache sizing, and distinctions between
    single-head MQA and other GQA configurations.

    References
    ----------
    Derivation: classify whether cache storage uses a compressed latent, grouped
    key/value heads, or one key/value head per query head. The precedence is a local API
    convention.

    Examples
    --------
    >>> calc_kv_attention_design(8, kv_heads=2)
    'GQA'
    """
    if kv_lora_rank:
        return "MLA"
    if kv_heads and heads and kv_heads < heads:
        return "GQA"
    return "MHA"


def calc_kv_resident_fit(pool, bytes_per_sequence) -> int:
    """Count complete equal-sized sequences that fit a memory pool.

    Parameters
    ----------
    pool : Quantity
        Nonnegative available memory in byte-compatible units.
    bytes_per_sequence : Quantity
        Positive total cache bytes per sequence, including any caller-selected reserve.

    Returns
    -------
    int
        Whole sequence count; integer conversion equals floor for nonnegative ratios.

    Notes
    -----
    Equation: B = floor(M_pool / M_sequence).

    Assumptions: Equal resident length and storage per request; the supplied pool is
    available exclusively for these caches.

    Excludes: Block rounding, allocator headroom, other tensors, and output growth
    unless the caller includes them in the inputs.

    References
    ----------
    Derivation: the capacity inequality ``B*M_sequence`` <= M_pool is solved for the largest
    nonnegative integer B.

    Examples
    --------
    >>> calc_kv_resident_fit(10 * ureg.byte, 3 * ureg.byte) == 10 // 3
    True
    """
    return int((pool / bytes_per_sequence).to("").magnitude)


def calc_kv_block_count(tokens: int, block_tokens: int) -> int:
    """Count fixed-size blocks needed by a token sequence.

    Parameters
    ----------
    tokens : int
        Nonnegative resident sequence length in tokens.
    block_tokens : int
        Positive tokens per allocation block.

    Returns
    -------
    int
        Number of whole blocks, with zero blocks for an empty sequence.

    Notes
    -----
    Equation: N_blocks = ceil(tokens / block_tokens).

    Assumptions: A block has identical token capacity for every sequence.

    Excludes: Metadata bytes, sharing, and future token reservations.

    References
    ----------
    Derivation: round the fractional block requirement upward because a partial block
    still occupies one block.

    Examples
    --------
    >>> calc_kv_block_count(17, 8) == math.ceil(17 / 8)
    True
    """
    validate_at_least(block_tokens, 1, "block_tokens")
    return math.ceil(tokens / block_tokens)


def calc_shared_prefix_fit(pool_blocks: int, shared_blocks: int, private_blocks: int) -> int:
    """Count requests that share one physical prefix allocation.

    Parameters
    ----------
    pool_blocks : int
        Total nonnegative physical block capacity; must be at least shared_blocks for a
        feasible pool.
    shared_blocks : int
        Nonnegative prefix blocks stored once for the group.
    private_blocks : int
        Positive unique suffix blocks needed by each request.

    Returns
    -------
    int
        Whole request count; a negative value signals that the supplied prefix already
        exceeds the pool.

    Notes
    -----
    Equation: B = floor((pool_blocks - shared_blocks) / private_blocks).

    Assumptions: Every admitted request reuses the same prefix and has the same private
    suffix footprint.

    Excludes: Copy-on-write duplication, reference-count metadata, and unequal suffix
    lengths.

    References
    ----------
    Derivation: solve shared_blocks + ``B*private_blocks`` <= pool_blocks for integer B.

    Examples
    --------
    >>> calc_shared_prefix_fit(20, 4, 3) == (20 - 4) // 3
    True
    """
    return (pool_blocks - shared_blocks) // private_blocks


def calc_sharing_gain_limit(shared_fraction: float) -> float:
    """Calculate the asymptotic residency gain from prefix sharing.

    Parameters
    ----------
    shared_fraction : float
        Fraction f of each request footprint shared across requests, in [0, 1).

    Returns
    -------
    float
        Dimensionless large-pool request-capacity multiplier.

    Notes
    -----
    Equation: gain = 1 / (1 - f).

    Assumptions: The shared prefix is stored once and becomes negligible as pool/request
    count grows.

    Excludes: Finite-pool prefix cost, integer block rounding, and metadata overhead.

    References
    ----------
    Derivation: the per-request private footprint is (1-f) times the unshared footprint;
    capacity scales inversely with footprint.

    Examples
    --------
    >>> calc_sharing_gain_limit(0) == 1
    True
    """
    if not 0 <= shared_fraction < 1:
        raise ValueError("shared_fraction must be in [0, 1)")
    return 1 / (1 - shared_fraction)


# --- On-device training: update traffic and parameter-efficient updates (added 2026-10-20) ---


def calc_update_step_traffic(weights, gradients, optimizer_state, activations):
    """Account for tensor traffic in one full-update training step.

    Parameters
    ----------
    weights : Quantity or float
        Nonnegative weight footprint; use bytes or byte-compatible quantities
        consistently.
    gradients : Quantity or float
        Nonnegative gradient footprint in the same units as weights.
    optimizer_state : Quantity or float
        Nonnegative persistent optimizer-state footprint in the same units.
    activations : Quantity or float
        Nonnegative saved-activation footprint in the same units.

    Returns
    -------
    tuple
        Total step traffic and traffic from the three weight reads; both preserve the
        input memory units.

    Notes
    -----
    Equation: V = (W+A) + (A+W+G) + (W+G+S) + (W+S) = ``4*W+2*A+2*G+2*S``; weight_reads =
    ``3*W``.

    Assumptions: Forward reads weights and writes saved activations. Backward reads both
    and writes gradients. The optimizer reads weights, gradients, state, then writes
    weights and state.

    Excludes: Cache reuse, fused-kernel savings, temporary workspace, communication, and
    activation recomputation.

    References
    ----------
    Derivation: sum each read and write at the memory boundary exactly once for the
    declared unfused execution sequence.

    Examples
    --------
    >>> total, reads = calc_update_step_traffic(2, 3, 4, 5)
    >>> total == 4*2 + 2*3 + 2*4 + 2*5 and reads == 3*2
    True
    """
    forward = weights + activations
    backward = activations + weights + gradients
    optimizer = (weights + gradients + optimizer_state) + (weights + optimizer_state)
    return forward + backward + optimizer, 3 * weights


def calc_low_rank_update_params(in_dim: int, out_dim: int, rank: int) -> int:
    """Count trainable weights in a low-rank matrix update.

    Parameters
    ----------
    in_dim : int
        Positive input width d_in of the frozen matrix.
    out_dim : int
        Positive output width d_out of the frozen matrix.
    rank : int
        Positive factor rank r; a low-rank use normally has r below both matrix
        dimensions.

    Returns
    -------
    int
        Trainable parameter count in the two update factors.

    Notes
    -----
    Equation: P = ``r*d_in`` + ``d_out*r``.

    Assumptions: The update is a product of an r-by-d_in down projection and a
    d_out-by-r up projection.

    Excludes: Frozen base weights, biases, optimizer state, and any scale parameter.

    References
    ----------
    Derivation: add the element counts of the two rectangular factors. This is the
    factor-size accounting used by LoRA; it does not predict adaptation quality.

    Examples
    --------
    >>> calc_low_rank_update_params(8, 16, 2) == 2 * (8 + 16)
    True
    """
    validate_at_least(rank, 1, "rank")
    return rank * (in_dim + out_dim)


def calc_residual_adapter_params(width: int, bottleneck: int) -> int:
    """Count weights and biases in a bottleneck residual adapter.

    Parameters
    ----------
    width : int
        Positive input/output width d.
    bottleneck : int
        Positive hidden width r of the adapter.

    Returns
    -------
    int
        Trainable parameter count for both affine projections.

    Notes
    -----
    Equation: P = ``d*r`` + r + ``r*d`` + d.

    Assumptions: A down projection maps d to r and an up projection maps r to d; each
    has a bias.

    Excludes: Layer-normalization parameters, gating, activation storage, and the frozen
    surrounding model.

    References
    ----------
    Derivation: count both projection matrices and their output biases. Residual
    addition introduces no learned parameter.

    Examples
    --------
    >>> calc_residual_adapter_params(8, 2) == 2 * 8 * 2 + 2 + 8
    True
    """
    validate_at_least(bottleneck, 1, "bottleneck")
    return 2 * width * bottleneck + bottleneck + width


# --- Sparse storage formats (added 2026-10-20) --------------------------------


def calc_sparse_bytes_per_nonzero(value_bytes, index_bytes, indices: int = 1):
    """Calculate sparse value-plus-index storage per nonzero.

    Parameters
    ----------
    value_bytes : Quantity or float
        Nonnegative bytes per stored value; use the same units as index_bytes.
    index_bytes : Quantity or float
        Nonnegative bytes per index.
    indices : int, optional
        Nonnegative indices stored per nonzero: default 1 for CSR, 2 for COO.

    Returns
    -------
    Quantity or float
        Per-nonzero storage in the input byte units.

    Notes
    -----
    Equation: b_nonzero = b_value + ``indices*b_index``.

    Assumptions: Every stored nonzero carries its value and the declared fixed number of
    indices.

    Excludes: CSR row pointers, headers, padding, and compressed/variable-width index
    coding.

    References
    ----------
    Derivation: add the storage widths of the fields in one sparse entry.

    Examples
    --------
    >>> calc_sparse_bytes_per_nonzero(2, 4, indices=2) == 2 + 2 * 4
    True
    """
    validate_at_least(indices, 0, "indices")
    return value_bytes + indices * index_bytes


def calc_sparse_breakeven_zero_share(dense_bytes_per_value, bytes_per_nonzero) -> float:
    """Find the zero fraction at which sparse and dense storage tie.

    Parameters
    ----------
    dense_bytes_per_value : Quantity or float
        Positive bytes per dense entry.
    bytes_per_nonzero : Quantity or float
        Positive bytes per sparse nonzero, in units compatible with
        dense_bytes_per_value.

    Returns
    -------
    float
        Dimensionless zero-share threshold; sparse wins above it. A negative threshold
        means sparse entries are already cheaper before sparsity.

    Notes
    -----
    Equation: z_star = 1 - b_dense/b_nonzero.

    Assumptions: Only surviving nonzeros consume sparse storage; both representations
    cover the same logical element count.

    Excludes: Per-row pointers, format headers, sparse compute overhead, and alignment.

    References
    ----------
    Derivation: equate ``N*b_dense`` with ``N*(1-z)*b_nonzero`` and solve for z.

    Examples
    --------
    >>> calc_sparse_breakeven_zero_share(2, 6) == 1 - 2 / 6
    True
    """
    return 1 - float(dense_bytes_per_value / bytes_per_nonzero)


def calc_csr_bytes(nonzeros: int, rows: int, value_bytes, index_bytes):
    """Calculate compressed sparse row matrix storage.

    Parameters
    ----------
    nonzeros : int
        Nonnegative number of stored matrix entries, including any explicitly stored
        zeros.
    rows : int
        Nonnegative matrix row count.
    value_bytes : Quantity or float
        Nonnegative bytes per value, in units compatible with index_bytes.
    index_bytes : Quantity or float
        Nonnegative bytes per column index and row pointer.

    Returns
    -------
    Quantity or float
        Total value, column-index, and row-pointer storage, preserving input byte units.

    Notes
    -----
    Equation: M = ``nnz*(b_value+b_index)`` + ``(rows+1)*b_index``.

    Assumptions: Each stored entry has one value and one column index; row boundaries
    use rows+1 pointers of the same width as indices.

    Excludes: Object headers, alignment, capacity over-allocation, and separate
    indexing-width choices for columns and row offsets.

    References
    ----------
    Derivation: sum the three CSR arrays: nnz values, nnz column indices, and rows+1 row
    pointers.

    Examples
    --------
    >>> calc_csr_bytes(5, 3, 2, 4) == 5 * (2 + 4) + (3 + 1) * 4
    True
    """
    return nonzeros * (value_bytes + index_bytes) + (rows + 1) * index_bytes
