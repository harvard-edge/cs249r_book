"""Layer-level counting laws for neural network architectures (added 2026-10-20).

Weight, multiply-add, and byte counts for one layer are products of its
dimensions and live on the calculation scenarios (``engine/scenario_types.py``:
``DenseLayerScenario``, ``ConvLayerScenario``, ``RecurrentStepScenario``).
This module holds the few laws those counts compose with: how a stack of
convolutions widens its view, how a depthwise separable factoring shrinks a
convolution, and how a product of per-layer gradient gains decays with depth.
"""

import math


def calc_receptive_field(layers: int, kernel: int) -> int:
    """Side of the input window one output sees after ``layers`` stride-1 ``kernel``-wide convolutions.

    Each layer adds ``kernel - 1`` pixels to the one-pixel start.
    """
    return 1 + layers * (kernel - 1)


def calc_layers_for_receptive_field(field: int, kernel: int) -> int:
    """Fewest stride-1 ``kernel``-wide convolutions whose receptive field reaches ``field`` pixels."""
    return math.ceil((field - 1) / (kernel - 1))


def calc_separable_cost_fraction(kernel: int, out_channels: int) -> float:
    """Depthwise separable cost over a standard ``kernel``-by-``kernel`` convolution (Howard et al. 2017).

    A depthwise step (one filter per input channel) and a pointwise 1x1 step
    cost ``1/out_channels + 1/kernel**2`` of the standard convolution's
    multiply-adds or weights.
    """
    return 1 / out_channels + 1 / kernel ** 2


def calc_gradient_product(gain: float, depth: int) -> float:
    """Product of ``depth`` per-layer gradient factors of average ``gain`` (plain stack, no skips)."""
    return gain ** depth
