"""Transformer FLOP accounting identities."""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.literature.registry import Literature

from ._units import _ensure_unit


def calc_transformer_training_flops(n_params, n_tokens):
    """Training FLOPs for a Transformer (6PD rule, Kaplan et al. 2020)."""
    p = _ensure_unit(n_params, ureg.param, "n_params").to(ureg.count).magnitude
    d = _ensure_unit(n_tokens, ureg.count, "n_tokens").magnitude
    return (Literature.Chinchilla.ComputeConstant * p * d) * ureg.flop


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
