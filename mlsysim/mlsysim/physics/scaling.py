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
