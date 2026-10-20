"""Scaling-law arithmetic: power-law price lists and compute-optimal allocation."""

from __future__ import annotations

import math

from mlsysim.core._validation import validate_positive


def calc_halving_multiple(exponent):
    """Resource multiple that halves a power-law term ``A R^-alpha``: ``2**(1/alpha)``."""
    validate_positive(exponent, "exponent")
    return 2 ** (1 / exponent)


def calc_compute_optimal_allocation(budget_flops, tokens_per_param, flops_per_param_token):
    """Parameters and tokens that spend ``budget_flops`` at a fixed tokens-per-parameter ratio.

    With ``C = k N D`` and ``D = r N``, ``N = sqrt(C / (k r))`` and ``D = r N``
    (Hoffmann et al. 2022). Plain numbers in, plain numbers out.
    """
    validate_positive(budget_flops, "budget_flops")
    n = math.sqrt(budget_flops / (flops_per_param_token * tokens_per_param))
    return n, tokens_per_param * n


def calc_allocation_growth(compute_multiple, exponent):
    """Fractional growth of a compute-optimal allocation ``X_opt ~ C^exponent`` when
    compute grows by ``compute_multiple``: ``multiple^exponent - 1`` (added 2026-10-20)."""
    validate_positive(compute_multiple, "compute_multiple")
    return compute_multiple ** exponent - 1
def calc_loss_reduction_compute_multiple(exponent, loss_reduction):
    """Compute multiple that lowers a power-law loss ``L = A C^-alpha`` by the fraction ``loss_reduction``.

    ``(1 / (1 - r)) ** (1 / alpha)``: with ``alpha = 0.05`` a 10 percent lower loss
    costs about 8.2 times the compute (added 2026-10-20).
    """
    validate_positive(exponent, "exponent")
    if not 0 < loss_reduction < 1:
        raise ValueError(f"loss_reduction must lie in (0, 1), got {loss_reduction}")
    return (1 / (1 - loss_reduction)) ** (1 / exponent)
