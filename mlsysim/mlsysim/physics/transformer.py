"""Transformer FLOP accounting identities."""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.literature.registry import Literature

from ._units import _ensure_unit


def calc_transformer_training_flops(n_params, n_tokens, activation_checkpointing: bool = False):
    """Estimate parameter-dominated Transformer training work.

    Parameters
    ----------
    n_params : Quantity or int
        Parameter count P, as a count or a quantity in param.
    n_tokens : Quantity or int
        Total training tokens D, including repeated passes through the data.
    activation_checkpointing : bool, optional
        Include one recomputed forward pass when True; default False.

    Returns
    -------
    Quantity
        Analytical training work in flop, not an execution-time measurement.

    Notes
    -----
    Equation: C = 6 P D without recomputation; C = 8 P D with full
    recomputation. Forward work contributes 2 P per token and backward work
    contributes 4 P. Full recomputation adds another 2 P forward pass.

    Assumptions: dense, parameter-dominated work.
    Excludes: attention over the context, embedding details, optimizer updates,
    communication, and runtime overhead. The recomputation increment is explicit accounting, not a
    fitted correction to the scaling-law paper. A serving solver must add
    context-dependent attention work separately.

    References
    ----------
    Kaplan et al. (2020), "Scaling Laws for Neural Language Models."
    https://arxiv.org/abs/2001.08361 (bibliography key: kaplan2020scaling).
    Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models."
    https://arxiv.org/abs/2203.15556 (bibliography key: hoffmann2022chinchilla).

    Examples
    --------
    Doubling processed tokens doubles work under identical assumptions.

    >>> first = calc_transformer_training_flops(1000, 100)
    >>> second = calc_transformer_training_flops(1000, 200)
    >>> second == first * 2
    True
    """
    p = _ensure_unit(n_params, ureg.param, "n_params").to(ureg.count).magnitude
    d = _ensure_unit(n_tokens, ureg.count, "n_tokens").magnitude
    factor = 8.0 if activation_checkpointing else float(Literature.Chinchilla.ComputeConstant)
    return (factor * p * d) * ureg.flop


def calc_transformer_decode_flops(n_params, n_tokens=1):
    """Estimate parameter-dominated forward work for generated tokens.

    Parameters
    ----------
    n_params : Quantity or int
        Nonnegative model parameter count P, in param or as a count.
    n_tokens : Quantity or int, optional
        Nonnegative number N of token positions; default one token.

    Returns
    -------
    Quantity
        Linear projection and feed-forward work in flop.

    Notes
    -----
    Equation: F = 2 P N, using the registered Chinchilla decode coefficient.
    Assumptions: dense parameter-dominated execution; each parameter contributes
    one multiply and one addition per token position.
    Excludes: attention products over the KV cache, softmax, normalization,
    communication, and runtime overhead. At long contexts, callers must add
    attention work separately; this is not a complete decode FLOP count.

    References
    ----------
    Kaplan et al. (2020), "Scaling Laws for Neural Language Models."
    https://arxiv.org/abs/2001.08361 (bibliography key: kaplan2020scaling).

    Examples
    --------
    >>> single = calc_transformer_decode_flops(1000)
    >>> calc_transformer_decode_flops(1000, 2) == 2 * single
    True
    """
    p = _ensure_unit(n_params, ureg.param, "n_params").to(ureg.count).magnitude
    t = _ensure_unit(n_tokens, ureg.count, "n_tokens").magnitude
    return (Literature.Chinchilla.DecodeConstant * p * t) * ureg.flop


def calc_attention_matmul_flops(n_layers, hidden_dim, query_tokens, key_tokens, batch_size=1):
    """Count work in both rectangular attention matrix products.

    Parameters
    ----------
    n_layers : int
        Number of transformer layers evaluated.
    hidden_dim : int
        Query-head count times head width, in elements.
    query_tokens : int or float
        Query positions evaluated per request.
    key_tokens : int or float
        Key/value positions visited by each query; may be an expected context.
    batch_size : int, optional
        Independent requests, default one.

    Returns
    -------
    Quantity
        Work in flop for QK-transpose and attention-weighted V.

    Notes
    -----
    Equation: F = 4 L H Q S B. Each matrix product costs two FLOPs per
    multiply-accumulate; the two products have the same operation count.
    Assumptions: every query visits every supplied key position. H is the
    query width, including grouped-query attention's query heads. Key and value
    head widths are equal.
    Excludes: projections, softmax, masking, data movement, and overhead.
    A causal prefill that skips the masked triangle applies its own reduction;
    this helper does not silently choose that execution convention.

    References
    ----------
    Vaswani et al. (2017), "Attention Is All You Need," Section 3.2.1,
    https://arxiv.org/abs/1706.03762 (bibliography key: vaswani2017attention).
    The FLOP count follows the two matrix products in the attention equation.

    Examples
    --------
    Partitioning queries into chunks preserves total matrix-product work.

    >>> whole = calc_attention_matmul_flops(2, 16, 8, 32)
    >>> chunk = calc_attention_matmul_flops(2, 16, 4, 32)
    >>> bool(whole == 2 * chunk)
    True
    """
    return (4 * n_layers * hidden_dim * query_tokens * key_tokens * batch_size) * ureg.flop


def calc_transformer_prefill_flops(
    n_params,
    seq_len,
    n_layers=None,
    hidden_dim=None,
    causal: bool = True,
):
    """Estimate projection and attention work for one prompt.

    Parameters
    ----------
    n_params : Quantity or int
        Nonnegative model parameter count P, in param or as a count.
    seq_len : Quantity or int
        Nonnegative prompt length S in tokens, represented as a count.
    n_layers : int, optional
        Nonnegative layer count L. Attention work is omitted if this or
        hidden_dim is None.
    hidden_dim : int, optional
        Nonnegative model width H, summed across attention heads.
    causal : bool, optional
        Use a half-square causal attention approximation when True; default True.

    Returns
    -------
    Quantity
        Projection plus attention matrix-product work in flop for one prompt.

    Notes
    -----
    Equation: F = 2 P S + a L H ``S**2``, where a is 2 for causal attention and
    4 for bidirectional attention. If either attention dimension is omitted,
    only 2 P S is returned.
    Assumptions: two matrix products for attention, two FLOPs per multiply-add,
    and head widths summing to H. Causal attention uses half of the square
    matrix rather than the exact triangular S(S+1)/2 count.
    Excludes: softmax, normalization, embedding lookup, vision encoders,
    communication, and kernel overhead. A batch of prompts requires summing
    or multiplying this per-prompt work.

    References
    ----------
    Derivation: QK-transpose and attention-times-V each require ``S**2`` H
    multiply-adds per layer. Counting two FLOPs per multiply-add yields
    4 L H ``S**2``; the causal approximation halves that term. Parameter products
    contribute 2 P S independently.

    Examples
    --------
    >>> dense = calc_transformer_prefill_flops(1000, 8)
    >>> causal = calc_transformer_prefill_flops(1000, 8, 2, 4)
    >>> full = calc_transformer_prefill_flops(1000, 8, 2, 4, causal=False)
    >>> full - dense == 2 * (causal - dense)
    True
    """
    p = _ensure_unit(n_params, ureg.param, "n_params").to(ureg.count).magnitude
    s = _ensure_unit(seq_len, ureg.count, "seq_len").magnitude

    proj_flops = calc_transformer_decode_flops(p, s)
    if n_layers is not None and hidden_dim is not None:
        full_attention = calc_attention_matmul_flops(n_layers, hidden_dim, s, s)
        attn_flops = full_attention / 2 if causal else full_attention
    else:
        attn_flops = 0.0 * ureg.flop

    return proj_flops + attn_flops


def calc_vla_inference_latency(
    n_params,
    weight_memory,
    input_tokens,
    action_tokens,
    peak_flops,
    memory_bandwidth,
    *,
    bus_efficiency: float = 1.0,
    compute_utilization: float = 1.0,
    parallel_decode: bool = False,
    n_layers=None,
    hidden_dim=None,
):
    """Bound one vision-language-action evaluation by compute and weight traffic.

    Parameters
    ----------
    n_params : Quantity or int
        Nonnegative decoder parameter count P, in param or as a count.
    weight_memory : Quantity or float
        Resident decoder weights in bytes; bare values are interpreted as bytes.
    input_tokens : int
        Number S of image-patch and prompt tokens, at least one.
    action_tokens : int
        Number D of generated action tokens, at least one.
    peak_flops : Quantity or float
        Positive compute capacity in flop/second at the execution precision.
    memory_bandwidth : Quantity or float
        Positive weight-memory bandwidth in byte/second.
    bus_efficiency : float, optional
        Sustained fraction of memory bandwidth in (0, 1]; default one.
    compute_utilization : float, optional
        Sustained fraction of peak compute in (0, 1]; default one.
    parallel_decode : bool, optional
        Model all action placeholders in one bidirectional pass if True;
        otherwise prefill emits the first token and D-1 serial passes follow.
    n_layers : int, optional
        Layer count for prefill attention work; omitted with hidden_dim by default.
    hidden_dim : int, optional
        Model width for prefill attention work; both dimensions must be supplied.

    Returns
    -------
    dict
        FLOPs in prefill_flops; millisecond quantities t_prefill_compute,
        t_weight_sweep, t_prefill, t_decode_step, t_decode, and t_total;
        frequency f_max in hertz; integer prefill_tokens and decode_steps;
        and prefill_bound identifying the larger prefill resource time.

    Notes
    -----
    Equation: T_prefill = max(F_prefill / (peak_flops * compute_utilization),
    weight_memory / (memory_bandwidth * bus_efficiency)). Serial decoding adds
    (D-1) * max(2 P / effective_compute, weight_memory / effective_bandwidth).
    Parallel decoding instead evaluates S+D prompt positions with bidirectional
    attention and no serial decode phase.
    Assumptions: ideal compute/memory overlap within each pass, serial passes,
    and one full weight sweep per pass. Efficiency factors are caller inputs,
    not predictions. Weight-only quantization changes weight_memory; callers
    must still supply compute capacity at the activation execution precision.
    Excludes: a separately timed vision encoder, KV-cache traffic and decode
    attention products, launch/framework overhead, and other bus clients.
    A weight-dominated pass often binds on bandwidth, but the maximum also
    permits compute to bind; this is not a measured robot-control latency.

    References
    ----------
    Williams et al. (2009), "Roofline: An Insightful Visual Performance Model
    for Multicore Architectures." https://doi.org/10.1145/1498765.1498785
    (bibliography key: williams2009). The pass composition is a modeling choice.
    Kim, Finn, and Liang (2025), "Fine-Tuning Vision-Language-Action Models:
    Optimizing Speed and Success." https://arxiv.org/abs/2502.19645
    (bibliography key: kim2025openvlaoft).
    Parallel action decoding motivates the parallel_decode scenario; this
    function does not reproduce that paper's measured speedup.
    Kim et al. (2024), "OpenVLA: An Open-Source Vision-Language-Action Model."
    https://arxiv.org/abs/2406.09246 (bibliography key: kim2024openvla).
    Autoregressive action-token generation motivates the serial scenario.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> result = calc_vla_inference_latency(
    ...     1000, Q_(2000, "byte"), 8, 1,
    ...     Q_(1, "TFLOP/s"), Q_(1, "GB/s"))
    >>> result["decode_steps"] == 0
    True
    >>> result["t_total"] == result["t_prefill"]
    True
    """
    n = _ensure_unit(n_params, ureg.param, "n_params")
    m_w = _ensure_unit(weight_memory, ureg.byte, "weight_memory")
    pi = _ensure_unit(peak_flops, ureg.flop / ureg.second, "peak_flops")
    bw = _ensure_unit(memory_bandwidth, ureg.byte / ureg.second, "memory_bandwidth")
    s_in = int(input_tokens)
    d = int(action_tokens)
    if s_in < 1 or d < 1:
        raise ValueError(f"input_tokens ({s_in}) and action_tokens ({d}) must be >= 1")
    if not 0.0 < compute_utilization <= 1.0:
        raise ValueError(f"compute_utilization ({compute_utilization}) must be in (0, 1]")
    if not 0.0 < bus_efficiency <= 1.0:
        raise ValueError(f"bus_efficiency ({bus_efficiency}) must be in (0, 1]")

    pi_eff = pi * compute_utilization
    b_eff = bw * bus_efficiency

    prefill_tokens = s_in + d if parallel_decode else s_in
    decode_steps = 0 if parallel_decode else d - 1

    # Bidirectional attention under parallel decoding doubles the S^2 term.
    prefill_flops = calc_transformer_prefill_flops(
        n,
        prefill_tokens,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        causal=not parallel_decode,
    )
    t_prefill_compute = (prefill_flops / pi_eff).to(ureg.millisecond)
    t_weight_sweep = (m_w / b_eff).to(ureg.millisecond)
    t_prefill = max(t_prefill_compute, t_weight_sweep)

    step_compute = (calc_transformer_decode_flops(n, 1) / pi_eff).to(ureg.millisecond)
    t_decode_step = max(step_compute, t_weight_sweep)
    t_decode = (decode_steps * t_decode_step).to(ureg.millisecond)

    t_total = (t_prefill + t_decode).to(ureg.millisecond)
    return {
        "prefill_tokens": prefill_tokens,
        "decode_steps": decode_steps,
        "prefill_flops": prefill_flops,
        "t_prefill_compute": t_prefill_compute,
        "t_weight_sweep": t_weight_sweep,
        "t_prefill": t_prefill.to(ureg.millisecond),
        "prefill_bound": "compute" if t_prefill_compute >= t_weight_sweep else "memory",
        "t_decode_step": t_decode_step.to(ureg.millisecond),
        "t_decode": t_decode,
        "t_total": t_total,
        "f_max": (1 / t_total).to(ureg.hertz),
    }


def calc_weight_only_step_time(n_params, n_tokens, sustained_rate, weight_read_time):
    """Bound forward-pass time using weight traffic and linear computation.

    Parameters
    ----------
    n_params : Quantity or int
        Model parameter count P, in param or as a count.
    n_tokens : Quantity or int
        Token positions N processed in this pass.
    sustained_rate : Quantity
        Effective computation rate R in flop/second, including any efficiency.
    weight_read_time : Quantity
        Time required to read the weights once, in time units.

    Returns
    -------
    Quantity
        Step-time bound in the same units as weight_read_time.

    Notes
    -----
    Equation: T = max(T_weights, 2 P N / R).
    Assumptions: one weight read and ideal compute/memory overlap.
    Excludes: attention traffic and attention matrix products, KV-cache reads,
    runtime overhead, and communication. This is a weight-only bound, not a
    complete serving-latency prediction. R is supplied, not fitted here.

    References
    ----------
    Williams et al. (2009), "Roofline: An Insightful Visual Performance Model
    for Multicore Architectures." https://doi.org/10.1145/1498765.1498785
    (bibliography key: williams2009). Roofline motivates the maximum;
    the weight-only workload is an explicit simplification.

    Examples
    --------
    The step cannot finish before its weight read completes.

    >>> from mlsysim.core.units import Q_
    >>> read_time = Q_(1, "ms")
    >>> step = calc_weight_only_step_time(
    ...     1000, 1, Q_(1, "TFLOP/s"), read_time)
    >>> bool(step >= read_time)
    True
    """
    ops = calc_transformer_decode_flops(n_params, n_tokens)
    compute = (ops / sustained_rate).to(weight_read_time.units)
    return max(weight_read_time, compute)


# --- Decoder-block counting rules (added 2026-10-20) --------------------------


def calc_transformer_block_weights(width, ffn_multiple=4, attention_projections=4):
    """Count attention and feed-forward weights in a dense decoder block.

    Parameters
    ----------
    width : int
        Positive hidden width H in scalar features.
    ffn_multiple : float, optional
        Positive expansion ratio R of the feed-forward hidden width; default four.
    attention_projections : int, optional
        Nonnegative number A of square H-by-H projections; default four
        represents query, key, value, and output projections.

    Returns
    -------
    tuple
        Attention, feed-forward, and total scalar weight counts, respectively.

    Notes
    -----
    Equation: W_attention = A ``H**2``; W_ffn = 2 R ``H**2``;
    W_total = W_attention + W_ffn.
    Assumptions: square attention projections and two dense feed-forward matrices
    with widths H to ``R*H`` to H.
    Excludes: biases, normalization parameters, embeddings, gated feed-forward
    branches, grouped-query dimensions, and sparse experts.

    References
    ----------
    Derivation: a dense matrix contributes its row count times its column count.
    The attention matrices each have ``H**2`` entries; each of the two feed-forward
    matrices has R ``H**2`` entries. Different architectures need different shapes.

    Examples
    --------
    >>> attention, feed_forward, total = calc_transformer_block_weights(8)
    >>> total == attention + feed_forward
    True
    >>> calc_transformer_block_weights(16)[2] == 4 * total
    True
    """
    attention = attention_projections * width**2
    mlp = 2 * ffn_multiple * width**2
    return attention, mlp, attention + mlp


def calc_attention_score_macs(seq_len, width):
    """Count multiply-adds in both full attention matrix products.

    Parameters
    ----------
    seq_len : int
        Nonnegative sequence length S in token positions.
    width : int
        Nonnegative hidden width H, summed across attention heads.

    Returns
    -------
    int
        Multiply-add count for QK-transpose and attention-times-V in one layer.

    Notes
    -----
    Equation: MACs = 2 ``S**2`` H.
    Assumptions: full square attention, equal query/key/value sequence lengths,
    and head widths summing to H. A multiply-add counts as one MAC here.
    Excludes: projection matrices, softmax, causal triangular savings,
    normalization, and conversion from MACs to FLOPs.

    References
    ----------
    Derivation: QK-transpose computes ``S**2`` dot products of width H across all
    heads. Multiplying the resulting attention matrix by V requires the same
    number of multiply-adds, giving two equal terms.

    Examples
    --------
    >>> work = calc_attention_score_macs(8, 4)
    >>> calc_attention_score_macs(16, 4) == 4 * work
    True
    """
    return 2 * seq_len**2 * width


def calc_attention_crossover(width, ffn_multiple=4, attention_projections=4):
    """Find where full attention and dense block multiply-add counts match.

    Parameters
    ----------
    width : int
        Positive hidden width H in scalar features.
    ffn_multiple : float, optional
        Positive feed-forward width expansion ratio; default four.
    attention_projections : int, optional
        Nonnegative number of square attention projections; default four.

    Returns
    -------
    float
        Sequence length in tokens at the algebraic crossover; not rounded to an
        integer token boundary.

    Notes
    -----
    Equation: S_cross = W_block / (2 H), obtained from ``W_block*S`` = 2 ``S**2`` H.
    Assumptions: the dense block shape in calc_transformer_block_weights and
    full square attention. This compares operation counts, not execution time.
    Excludes: different kernel efficiencies, memory bandwidth, causal masking,
    biases, embeddings, and grouped-query or sparse-expert architectures.

    References
    ----------
    Derivation: divide the equality of dense and attention MAC counts by the
    positive sequence length and solve for S. Both sides use MACs, so no
    FLOP conversion factor is needed.

    Examples
    --------
    >>> width = 8
    >>> context = calc_attention_crossover(width)
    >>> weights = calc_transformer_block_weights(width)[2]
    >>> weights * context == calc_attention_score_macs(context, width)
    True
    """
    _, _, total = calc_transformer_block_weights(width, ffn_multiple, attention_projections)
    return total / (2 * width)


def calc_online_softmax(scores, values, block_size: int):
    """Accumulate softmax normalization and weighted values in score blocks.

    Parameters
    ----------
    scores : sequence of float
        Finite, dimensionless logits; empty input returns an empty trace.
    values : sequence of float
        Scalar values paired with scores, of the same length and shared units.
    block_size : int
        Positive number of score/value pairs processed per block.

    Returns
    -------
    list of dict
        One record per block with running max, rescale factor, exponential sum,
        and unnormalized weighted output. Divide output by sum to obtain the
        normalized weighted mean. The output retains the values' numeric scale.

    Notes
    -----
    Equation: for new maximum m, rescale old sums by exp(m_old-m), then add
    sum(exp(x-m)) and ``sum(exp(x-m)*v)`` for the current block.
    Assumptions: finite scalar logits and scalar values; each block uses its
    maximum to avoid overflow from exponentiating large positive logits.
    Excludes: vector values, attention masking, normalization of the returned
    output, GPU execution, and memory-traffic timing. In exact arithmetic,
    output/sum agrees with full-row softmax; floating-point rounding can differ.

    References
    ----------
    Milakov and Gimelshein (2018), "Online Normalizer Calculation for Softmax."
    https://arxiv.org/abs/1805.02867. The running weighted numerator follows the
    same exponential rescaling as the normalizer.

    Examples
    --------
    >>> trace = calc_online_softmax([0.0, 0.0], [2.0, 4.0], 1)
    >>> last = trace[-1]
    >>> last["output"] / last["sum"] == sum([2.0, 4.0]) / len([2.0, 4.0])
    True
    """
    import math as _m

    if len(scores) != len(values):
        raise ValueError("scores and values must have the same length")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    rows = []
    m, ell, out = -_m.inf, 0.0, 0.0
    for start in range(0, len(scores), block_size):
        xs = scores[start : start + block_size]
        vs = values[start : start + block_size]
        m_new = max(m, max(xs))
        rescale = _m.exp(m - m_new) if m != -_m.inf else 0.0
        ell = ell * rescale + sum(_m.exp(x - m_new) for x in xs)
        out = out * rescale + sum(_m.exp(x - m_new) * v for x, v in zip(xs, vs))
        m = m_new
        rows.append(dict(max=m, rescale=rescale, sum=ell, output=out))
    return rows


def calc_score_matrix_intensity(head_dim, bytes_per_value):
    """Compute arithmetic intensity for materialized attention scores.

    Parameters
    ----------
    head_dim : int
        Positive dot-product width H of one attention head.
    bytes_per_value : Quantity or float
        Positive storage size b of one score in bytes; bare values mean bytes.

    Returns
    -------
    Quantity
        FLOPs per byte for score formation with one write and one subsequent read.

    Notes
    -----
    Equation: intensity = (2 H) / (2 b) = H / b FLOP/byte.
    Assumptions: two FLOPs per dot-product element and each score is written
    once to memory and later read once.
    Excludes: Q/K input traffic, softmax, value products, cache reuse, and fused
    attention kernels that avoid materializing the score matrix. This local
    intensity is not the whole layer's arithmetic intensity.

    References
    ----------
    Derivation: each score is one length-H dot product, costing 2 H FLOPs.
    Writing and reading its b-byte representation costs 2 b bytes; divide work
    by traffic to obtain the score-materialization intensity.

    Examples
    --------
    >>> intensity = calc_score_matrix_intensity(8, 2)
    >>> calc_score_matrix_intensity(16, 2) == 2 * intensity
    True
    """
    bpv = _ensure_unit(bytes_per_value, ureg.byte).to(ureg.byte).magnitude
    return (2 * head_dim / (2 * bpv)) * ureg.flop / ureg.byte
