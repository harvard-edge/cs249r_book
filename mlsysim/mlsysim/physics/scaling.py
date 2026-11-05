"""Scaling-law arithmetic: power-law price lists and compute-optimal allocation."""

from __future__ import annotations

import math

from mlsysim.core._validation import validate_positive


def calc_halving_multiple(exponent):
    """Find the resource multiple that halves a power-law loss term.

    Parameters
    ----------
    exponent : float
        Positive dimensionless power-law exponent alpha.

    Returns
    -------
    float
        Dimensionless resource multiplier required to halve the modeled term.

    Notes
    -----
    Equation: multiple = ``2**(1/exponent)``.

    Assumptions: The reducible term follows ``A*R**(-alpha)`` with fixed A and alpha.

    Excludes: Irreducible loss, regime changes, and estimation of the exponent.

    References
    ----------
    Derivation: solve ``(R_new/R_old)**(-alpha)`` = 1/2 for the resource ratio.

    Examples
    --------
    >>> calc_halving_multiple(0.5)
    4.0
    """
    validate_positive(exponent, "exponent")
    return 2 ** (1 / exponent)


def calc_compute_optimal_allocation(budget_flops, tokens_per_param, flops_per_param_token):
    """Allocate a compute budget at a fixed tokens-per-parameter ratio.

    Parameters
    ----------
    budget_flops : float
        Positive training-work budget C, in FLOPs as a plain number.
    tokens_per_param : float
        Positive ratio r of training tokens to model parameters.
    flops_per_param_token : float
        Positive accounting coefficient k, FLOPs per parameter per token.

    Returns
    -------
    tuple of float
        Parameter count N and training-token count D as continuous plain numbers;
        neither value is rounded to an implementable model shape.

    Notes
    -----
    Equation: N = ``sqrt(C/(k*r))``; D = ``r*N``, from C=``k*N*D``.

    Assumptions: The supplied ratio is the chosen allocation rule and work is
        parameter dominated. The function does not estimate the optimal ratio.

    Excludes: Attention work, communication, architecture constraints, and fitting
        a loss model to determine r.

    References
    ----------
    Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models."
    https://arxiv.org/abs/2203.15556 (bibliography key: hoffmann2022chinchilla).
    Derivation: substitute the assumed D=``r*N`` into the work budget C=``k*N*D``.

    Examples
    --------
    >>> calc_compute_optimal_allocation(600, 1, 6)
    (10.0, 10.0)
    """
    validate_positive(budget_flops, "budget_flops")
    n = math.sqrt(budget_flops / (flops_per_param_token * tokens_per_param))
    return n, tokens_per_param * n


def calc_allocation_growth(compute_multiple, exponent):
    """Calculate fractional allocation growth under a compute power law.

    Parameters
    ----------
    compute_multiple : float
        Positive dimensionless ratio of new to old compute budget.
    exponent : float
        Dimensionless exponent relating the allocation to compute.

    Returns
    -------
    float
        Fractional change in allocation; multiply by 100 only when presenting
        a percentage.

    Notes
    -----
    Equation: growth = ``compute_multiple**exponent`` - 1.

    Assumptions: Allocation follows X_opt = ``A*C**exponent`` with fixed A.

    Excludes: Fitting the exponent, integer rounding, and changes in optimal regime.

    References
    ----------
    Derivation: divide X_new=``A*C_new**exponent`` by X_old=``A*C_old**exponent``
        and subtract one.

    Examples
    --------
    >>> calc_allocation_growth(4, 0.5)
    1.0
    """
    validate_positive(compute_multiple, "compute_multiple")
    return compute_multiple**exponent - 1


def calc_power_law_factor(multiple, exponent):
    """Calculate the remaining fraction of a resource-dependent power-law term.

    Parameters
    ----------
    multiple : float
        Positive dimensionless ratio of new to old resource.
    exponent : float
        Dimensionless alpha in the assumed law A*R**(-alpha).

    Returns
    -------
    float
        New term divided by old term, dimensionless; this is the remaining
        fraction rather than the reduction fraction.

    Notes
    -----
    Equation: factor = ``multiple**(-exponent)``.

    Assumptions: The same coefficient and exponent hold before and after the change.

    Excludes: Irreducible offsets, empirical fitting, and saturation.

    References
    ----------
    Derivation: cancel A in the ratio ``(A*R_new**(-alpha))/(A*R_old**(-alpha))``.

    Examples
    --------
    >>> calc_power_law_factor(4, 0.5)
    0.5
    """
    validate_positive(multiple, "multiple")
    return multiple ** (-exponent)


def calc_loss_reduction_compute_multiple(exponent, loss_reduction):
    """Find the compute multiple for a fractional power-law loss reduction.

    Parameters
    ----------
    exponent : float
        Positive dimensionless alpha in L=A*C**(-alpha).
    loss_reduction : float
        Desired fractional reduction r in the modeled loss, in (0, 1).

    Returns
    -------
    float
        Dimensionless new-to-old compute ratio.

    Notes
    -----
    Equation: multiple = ``(1/(1-loss_reduction))**(1/exponent)``.

    Assumptions: The entire loss being reduced is the power-law term with fixed
        coefficient and exponent.

    Excludes: Irreducible loss floors, model-size constraints, and fitted uncertainty.

    References
    ----------
    Derivation: solve ``multiple**(-alpha)`` = 1-r for multiple.
        Small positive exponents can make modest loss reductions expensive.

    Examples
    --------
    >>> calc_loss_reduction_compute_multiple(0.5, 0.5)
    4.0
    """
    validate_positive(exponent, "exponent")
    if not 0 < loss_reduction < 1:
        raise ValueError(f"loss_reduction must lie in (0, 1), got {loss_reduction}")
    return (1 / (1 - loss_reduction)) ** (1 / exponent)
