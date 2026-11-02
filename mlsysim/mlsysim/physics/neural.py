"""Neural computation: what a dense network costs, and what its activations and loss compute.

A dense (fully connected) network is a list of layer widths ``[n_0, n_1, ...,
n_L]``: layer ``l`` maps ``n_{l-1}`` inputs to ``n_l`` outputs with an
``n_{l-1}`` by ``n_l`` weight matrix and one bias per output. Every count here
follows from the widths and the batch size alone (added 2026-10-20).
"""

from __future__ import annotations

import math

from mlsysim.core._validation import validate_at_least


def _pairs(widths):
    widths = list(widths)
    validate_at_least(len(widths), 2, "number of layer widths")
    return list(zip(widths[:-1], widths[1:]))


def calc_dense_weights(widths) -> list[int]:
    """Weights per layer: one per connection, ``n_in * n_out``."""
    return [a * b for a, b in _pairs(widths)]


def calc_dense_biases(widths) -> list[int]:
    """Biases per layer: one per output unit, ``n_out``."""
    return [b for _, b in _pairs(widths)]


def calc_dense_params(widths) -> int:
    """Parameters of a dense network: ``sum((n_in + 1) * n_out)``."""
    return sum(calc_dense_weights(widths)) + sum(calc_dense_biases(widths))


def calc_dense_macs(widths) -> int:
    """Multiply-accumulates per example: one per weight."""
    return sum(calc_dense_weights(widths))


def calc_dense_forward_flops(widths, batch: int = 1):
    """
    Forward-pass operations at batch ``B``, per layer.

        matrix multiply: 2 B n_in n_out      (one MAC = 2 FLOPs)
        elementwise:     2 B n_out           (bias add and activation)

    Returns ``(matmul, elementwise)``, two lists with one entry per layer.
    """
    validate_at_least(batch, 1, "batch")
    matmul = [2 * batch * a * b for a, b in _pairs(widths)]
    elementwise = [2 * batch * b for _, b in _pairs(widths)]
    return matmul, elementwise


def calc_dense_backward_flops(widths, batch: int = 1):
    """
    Backward-pass matrix work at batch ``B``.

    Each layer computes its weight gradient (activations transposed times the
    error, ``2 B n_in n_out``) and, except the first layer, the error sent to
    the layer before it (error times weights transposed, the same size). The
    first layer has no earlier layer to send a signal to.

    Returns ``(weight_gradient, input_gradient)`` totals.
    """
    matmul, _ = calc_dense_forward_flops(widths, batch)
    return sum(matmul), sum(matmul[1:])


def calc_dense_traffic_values(widths, batch: int = 1) -> int:
    """
    Activation values that cross main memory in one forward pass.

    Each layer runs as its own kernel: the input is read once, the output is
    written once, and every hidden activation is written by one layer and read
    back by the next, ``B (n_0 + 2 sum(hidden) + n_L)``. The weights are read
    once on top of this.
    """
    widths = list(widths)
    return batch * (widths[0] + 2 * sum(widths[1:-1]) + widths[-1])


def calc_dense_saved_activations(widths, batch: int = 1) -> int:
    """Values training keeps for the backward pass: the input and every layer output, ``B sum(n_l)``."""
    return batch * sum(widths)


def calc_dense_inference_activations(widths, batch: int = 1) -> int:
    """
    Values inference must hold at once with rolling buffers: the largest
    adjacent pair, ``B max(n_{l-1} + n_l)``, since each output can overwrite
    its input once the next layer has read it.
    """
    return batch * max(a + b for a, b in _pairs(widths))


def calc_training_step_ops(layers: int) -> int:
    """
    Operations launched in one training step of a dense ReLU network of ``layers`` layers.

        forward       L   (one matrix multiply each, bias and ReLU fused)
        loss          1
        backward      2L - 1   (a weight gradient per layer, an input gradient for all but the first)
        ReLU masks    L - 1    (one per hidden layer)
        update        1   (one fused optimizer step)

    Total ``4L``.
    """
    validate_at_least(layers, 1, "layers")
    return layers + 1 + (2 * layers - 1) + (layers - 1) + 1


def calc_training_state(parameters: int, bytes_per_value, optimizer_bytes_per_parameter):
    """
    Persistent training state of ``parameters`` values, one precision throughout.

    Weights and gradients take ``bytes_per_value`` each per parameter; the
    optimizer keeps ``optimizer_bytes_per_parameter`` more (two FP32 values for
    Adam, one for SGD with momentum). Returns ``(weights, gradients, optimizer)``.
    """
    weights = parameters * bytes_per_value
    return weights, parameters * bytes_per_value, parameters * optimizer_bytes_per_parameter


def calc_sigmoid(x: float) -> float:
    """The logistic sigmoid, ``1 / (1 + e^-x)``."""
    return 1.0 / (1.0 + math.exp(-x))


def calc_sigmoid_slope(x: float) -> float:
    """The sigmoid's derivative, ``s(x)(1 - s(x))``; its peak, at ``x = 0``, is one quarter."""
    s = calc_sigmoid(x)
    return s * (1.0 - s)


def calc_cross_entropy(p_correct: float) -> float:
    """Cross-entropy loss with a one-hot label: ``-ln p`` of the correct class."""
    if not 0 < p_correct <= 1:
        raise ValueError(f"p_correct must lie in (0, 1], got {p_correct}")
    return -math.log(p_correct)


def calc_relu(x: float) -> float:
    """Rectified linear unit, ``max(0, x)``."""
    return max(0.0, x)
