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
    """Autoregressive decode FLOPs using the 2P rule."""
    p = _ensure_unit(n_params, ureg.param, "n_params").to(ureg.count).magnitude
    t = _ensure_unit(n_tokens, ureg.count, "n_tokens").magnitude
    return (Literature.Chinchilla.DecodeConstant * p * t) * ureg.flop
