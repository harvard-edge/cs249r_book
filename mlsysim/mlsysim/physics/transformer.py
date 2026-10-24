"""Transformer FLOP accounting identities."""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.literature.registry import Literature

from ._units import _ensure_unit


def calc_transformer_training_flops(n_params, n_tokens, activation_checkpointing: bool = False):
    """Training FLOPs for a Transformer.

    By default, applies the standard dense baseline 6PD rule (Kaplan et al. 2020;
    Chinchilla: 2P forward + 4P backward). When activation_checkpointing=True,
    accounts for the additional forward pass recomputed during backpropagation
    (2P forward + 2P recompute + 4P backward = 8PD rule).

    Parameters
    ----------
    n_params : Quantity or int
        Total parameter count of the model (P).
    n_tokens : Quantity or int
        Training tokens processed (D).
    activation_checkpointing : bool, optional
        Whether full activation recomputation is enabled (default: False -> 6PD; True -> 8PD).

    Returns
    -------
    Quantity
        Total training compute in FLOPs.
    """
    p = _ensure_unit(n_params, ureg.param, "n_params").to(ureg.count).magnitude
    d = _ensure_unit(n_tokens, ureg.count, "n_tokens").magnitude
    factor = 8.0 if activation_checkpointing else float(Literature.Chinchilla.ComputeConstant)
    return (factor * p * d) * ureg.flop


def calc_transformer_decode_flops(n_params, n_tokens=1):
    """Autoregressive decode FLOPs using the 2P rule.

    Forward pass ~ 2 FLOPs per parameter per token (Kaplan et al. 2020,
    Sec. 2.1). Deliberately EXCLUDES per-token attention-over-KV FLOPs
    (~ 2 * layers * context * kv_width per token, linear in context length),
    so it holds in the parameter-dominated regime (short/moderate context);
    do not apply it unmodified at very long contexts (e.g. 128K).
    """
    p = _ensure_unit(n_params, ureg.param, "n_params").to(ureg.count).magnitude
    t = _ensure_unit(n_tokens, ureg.count, "n_tokens").magnitude
    return (Literature.Chinchilla.DecodeConstant * p * t) * ureg.flop


def calc_transformer_prefill_flops(
    n_params,
    seq_len,
    n_layers=None,
    hidden_dim=None,
    causal: bool = True,
):
    """
    Calculate prompt prefill FLOPs for a Transformer.

    Prefill computation comprises:
    1. Linear projections and FFN layers: 2 * n_params * seq_len
    2. Self-attention matrix multiplications (QK^T and Softmax * V):
       - Causal (lower-triangular): 2 * n_layers * hidden_dim * seq_len^2
       - Bidirectional (full):      4 * n_layers * hidden_dim * seq_len^2

    Parameters
    ----------
    n_params : Quantity or int
        Total parameter count of the model.
    seq_len : Quantity or int
        Prompt sequence length in tokens.
    n_layers : int, optional
        Number of transformer layers.
    hidden_dim : int, optional
        Model hidden dimension (d_model).
    causal : bool, optional
        Whether attention is causal / autoregressive masked (default: True).

    Returns
    -------
    Quantity
        Total prefill FLOPs.
    """
    p = _ensure_unit(n_params, ureg.param, "n_params").to(ureg.count).magnitude
    s = _ensure_unit(seq_len, ureg.count, "seq_len").magnitude

    proj_flops = 2.0 * p * s
    if n_layers is not None and hidden_dim is not None:
        attn_factor = 2.0 if causal else 4.0
        attn_flops = attn_factor * n_layers * hidden_dim * (s ** 2)
    else:
        attn_flops = 0.0

    return (proj_flops + attn_flops) * ureg.flop


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
    """
    Roofline latency of one fresh vision-language-action (VLA) evaluation.

    One evaluation of a decoder-only VLA backbone has two phases that bind on
    different resources:

    1. **Prefill.** The input tokens (image patches plus prompt) pass through
       the backbone together. Each weight fetched from memory is reused across
       all S tokens, so at 16-bit activations the arithmetic intensity is about
       S FLOP/byte and the pass is compute-bound once S exceeds the ridge
       point. The pass cannot finish before its own weight sweep either:
       ``t_prefill = max(F_prefill / (Pi * u), M_weights / (B * eta))``, with
       ``F_prefill`` from :func:`calc_transformer_prefill_flops`.
    2. **Decode.** An autoregressive action decoder emits one action token per
       step (OpenVLA emits one token per action dimension). The prefill pass
       emits the first token; each of the remaining ``action_tokens - 1`` steps
       is a matrix-vector pass that re-reads every weight, so it is
       bandwidth-bound: ``t_step = max(2 N / (Pi * u), M_weights / (B * eta))``.

    With ``parallel_decode=True`` (OpenVLA-OFT) the action tokens enter the
    same forward pass as empty placeholder embeddings under bidirectional
    attention: the prefill grows by ``action_tokens`` and the decode phase
    vanishes.

    Weight-only quantization (for example INT4 weights with 16-bit
    activations) shrinks ``weight_memory`` and therefore the decode term, but
    not the prefill FLOPs, which still execute at the 16-bit ``peak_flops``
    the caller passes in.

    Source: Pope et al., "Efficiently Scaling Transformer Inference," MLSys
    2023 (prefill compute-bound, decode memory-bound); Williams, Waterman and
    Patterson, "Roofline," CACM 52(4), 2009; Kim et al., "OpenVLA,"
    arXiv:2406.09246, 2024 (one autoregressive token per action dimension);
    Kim, Finn and Liang, "Fine-Tuning Vision-Language-Action Models:
    Optimizing Speed and Success," RSS 2025 (parallel decoding).

    Excludes a separately timed vision encoder, KV-cache traffic, kernel
    launch and framework overhead, and contention from other bus clients.
    ``compute_utilization`` (achieved fraction of ``peak_flops``) and
    ``bus_efficiency`` (sustained fraction of ``memory_bandwidth``) are the
    two calibration knobs.

    Returns
    -------
    dict
        Quantities ``prefill_flops``, ``t_prefill_compute``,
        ``t_weight_sweep``, ``t_prefill``, ``t_decode_step``, ``t_decode``,
        ``t_total``, ``f_max``; ints ``prefill_tokens`` and ``decode_steps``;
        and ``prefill_bound`` ("compute" or "memory").
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
        n, prefill_tokens, n_layers=n_layers, hidden_dim=hidden_dim, causal=not parallel_decode,
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
    """
    Time for one forward pass over ``n_tokens`` positions, weight work only.

        T(N) = max(T_weights, 2 P N / R)

    The pass either waits to read every weight once (``weight_read_time``) or
    does ``2 P N`` weight operations at ``sustained_rate``, whichever is
    longer. Attention traffic is left out, so long contexts take longer
    than this bound (added 2026-10-20).
    """
    ops = calc_transformer_decode_flops(n_params, n_tokens)
    compute = (ops / sustained_rate).to(weight_read_time.units)
    return max(weight_read_time, compute)


# --- Decoder-block counting rules (added 2026-10-20) --------------------------

def calc_transformer_block_weights(width, ffn_multiple=4, attention_projections=4):
    """Weights of one classic decoder block, biases and norms aside.

    Attention holds ``attention_projections`` square projections (query, key,
    value, output: 4 d^2); the feed-forward network goes d -> ffn_multiple d
    -> d (8 d^2 at the usual multiple of four). Returns (attention, mlp,
    total); the classic block is 12 d^2 (Vaswani et al. 2017; Radford et al.
    2019).
    """
    attention = attention_projections * width ** 2
    mlp = 2 * ffn_multiple * width ** 2
    return attention, mlp, attention + mlp


def calc_attention_score_macs(seq_len, width):
    """Multiply-adds of one block's two attention products, Q K^T and the
    weighted sum of V: S^2 d each, 2 S^2 d in all (summed over heads, which
    split d without changing the total)."""
    return 2 * seq_len ** 2 * width


def calc_attention_crossover(width, ffn_multiple=4, attention_projections=4):
    """Context length at which a block's attention products equal its dense
    products: (weights per block) S = 2 S^2 d gives S = (weights per block) / 2d,
    six times the width for the classic block."""
    _, _, total = calc_transformer_block_weights(width, ffn_multiple, attention_projections)
    return total / (2 * width)


def calc_online_softmax(scores, values, block_size: int):
    """
    Softmax-weighted sum of ``values`` over ``scores``, one block of scores at a time.

    Each block updates a running maximum ``m``, a running sum
    ``l = sum exp(x - m)``, and a running output ``O = sum exp(x - m) v``;
    when the maximum rises, the old ``l`` and ``O`` are rescaled by
    ``exp(m_old - m_new)`` (Milakov and Gimelshein 2018; the inner loop of
    FlashAttention, Dao et al. 2022). The result after the last block equals
    the whole-row computation exactly.

    Returns a list with one dict per block: ``max``, ``rescale`` (the factor
    applied to the previous ``l`` and ``O``), ``sum``, and ``output``
    (added 2026-10-20).
    """
    import math as _m
    if len(scores) != len(values):
        raise ValueError("scores and values must have the same length")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    rows = []
    m, ell, out = -_m.inf, 0.0, 0.0
    for start in range(0, len(scores), block_size):
        xs = scores[start:start + block_size]
        vs = values[start:start + block_size]
        m_new = max(m, max(xs))
        rescale = _m.exp(m - m_new) if m != -_m.inf else 0.0
        ell = ell * rescale + sum(_m.exp(x - m_new) for x in xs)
        out = out * rescale + sum(_m.exp(x - m_new) * v for x, v in zip(xs, vs))
        m = m_new
        rows.append(dict(max=m, rescale=rescale, sum=ell, output=out))
    return rows


def calc_score_matrix_intensity(head_dim, bytes_per_value):
    """
    Arithmetic intensity of materializing one head's attention scores in memory.

    Each score costs ``2 d_head`` FLOPs (a dot product of length ``d_head``) and
    is written once and read back once, ``2 x bytes_per_value`` bytes, so the
    intensity is ``d_head / bytes_per_value`` FLOP per byte: 32 for 64-wide
    heads in FP16, far below an accelerator's ridge point (added 2026-10-20).
    """
    bpv = _ensure_unit(bytes_per_value, ureg.byte).to(ureg.byte).magnitude
    return (2 * head_dim / (2 * bpv)) * ureg.flop / ureg.byte
