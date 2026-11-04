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
    """Count the dense weight matrix entries in each layer.

    Parameters
    ----------
    widths : sequence of int
        Positive layer widths [input, hidden..., output]; at least two entries. Counts are
        dimensionless.

    Returns
    -------
    list of int
        Parameter counts, one per layer.

    Notes
    -----
    Equation: W_l = n_(l-1) n_l.

    Assumptions: Every input connects to every output in each layer.

    Excludes: Biases, sparsity, shared parameters, and convolutional structure.

    References
    ----------
    Derivation: A fully connected mapping from n_in inputs to n_out outputs has one
    independently stored weight for every input-output pair, giving n_in times n_out
    entries.

    Examples
    --------
    >>> calc_dense_weights([2, 3]) == [2 * 3]
    True
    """
    return [a * b for a, b in _pairs(widths)]


def calc_dense_biases(widths) -> list[int]:
    """Count one bias for each output unit in every layer.

    Parameters
    ----------
    widths : sequence of int
        Positive layer widths [input, hidden..., output]; at least two entries. Counts are
        dimensionless.

    Returns
    -------
    list of int
        Bias parameter counts, one per layer.

    Notes
    -----
    Equation: b_l = n_l.

    Assumptions: Every dense layer has a bias.

    Excludes: Bias-free layers, normalization, and weight matrices.

    References
    ----------
    Derivation: Each output unit adds one independently stored scalar bias; therefore a
    layer with n_out units contains n_out bias parameters.

    Examples
    --------
    >>> calc_dense_biases([2, 3, 1])
    [3, 1]
    """
    return [b for _, b in _pairs(widths)]


def calc_dense_params(widths) -> int:
    """Count weights and biases in a dense network.

    Parameters
    ----------
    widths : sequence of int
        Positive layer widths [input, hidden..., output]; at least two entries. Counts are
        dimensionless.

    Returns
    -------
    int
        Total trainable parameter count.

    Notes
    -----
    Equation: P = sum_l (n_(l-1) + 1) n_l.

    Assumptions: Every dense connection has a weight and every output has a bias.

    Excludes: Normalization parameters, shared weights, and sparsity.

    References
    ----------
    Derivation: P = sum_l (n_(l-1) + 1) n_l.

    Examples
    --------
    >>> calc_dense_params([2, 3]) == (2 + 1) * 3
    True
    """
    return sum(calc_dense_weights(widths)) + sum(calc_dense_biases(widths))


def calc_dense_macs(widths) -> int:
    """Count dense multiply-accumulates for one example.

    Parameters
    ----------
    widths : sequence of int
        Positive layer widths [input, hidden..., output]; at least two entries. Counts are
        dimensionless.

    Returns
    -------
    int
        Multiply-accumulate count per example.

    Notes
    -----
    Equation: MACs = sum_l n_(l-1) n_l.

    Assumptions: One matrix-vector product is evaluated per layer.

    Excludes: Bias additions, nonlinearities, and backward computation.

    References
    ----------
    Derivation: Evaluating a dense layer uses each weight once in a multiply-accumulate for
    one example; summing the matrix entry counts gives the network MAC count.

    Examples
    --------
    >>> calc_dense_macs([2, 3]) == 2 * 3
    True
    """
    return sum(calc_dense_weights(widths))


def calc_dense_forward_flops(widths, batch: int = 1):
    """Count forward matrix and elementwise work by layer.

    Parameters
    ----------
    widths : sequence of int
        Positive layer widths [input, hidden..., output]; at least two entries. Counts are
        dimensionless.
    batch : int, optional
        Positive number of examples processed together; default 1.

    Returns
    -------
    tuple of list of int
        Matrix FLOPs and elementwise FLOPs, each indexed by layer.

    Notes
    -----
    Equation: F_matrix,l = 2 B n_(l-1) n_l; F_element,l = 2 B n_l.

    Assumptions: A multiply-accumulate counts as two FLOPs; bias and activation cost one
    operation each.

    Excludes: Architecture-specific activation costs, fused-kernel savings, and backward
    work.

    References
    ----------
    Derivation: F_matrix,l = 2 B n_(l-1) n_l; F_element,l = 2 B n_l.

    Examples
    --------
    >>> calc_dense_forward_flops([2, 3], batch=2) == ([2 * 2 * 2 * 3], [2 * 2 * 3])
    True
    """
    validate_at_least(batch, 1, "batch")
    matmul = [2 * batch * a * b for a, b in _pairs(widths)]
    elementwise = [2 * batch * b for _, b in _pairs(widths)]
    return matmul, elementwise


def calc_dense_backward_flops(widths, batch: int = 1):
    """Count dense weight-gradient and input-gradient matrix work.

    Parameters
    ----------
    widths : sequence of int
        Positive layer widths [input, hidden..., output]; at least two entries. Counts are
        dimensionless.
    batch : int, optional
        Positive number of examples processed together; default 1.

    Returns
    -------
    tuple of int
        Total weight-gradient FLOPs and total input-gradient FLOPs.

    Notes
    -----
    Equation: F_weight = sum_l 2 B n_(l-1) n_l; F_input sums the same work except the first
    layer.

    Assumptions: Training does not request a gradient with respect to the original network
    input.

    Excludes: Activation derivatives, bias gradients, loss work, and optimizer updates.

    References
    ----------
    Derivation: F_weight = sum_l 2 B n_(l-1) n_l; F_input sums the same work except the first layer.

    Examples
    --------
    >>> calc_dense_backward_flops([2, 3], batch=2) == (2 * 2 * 2 * 3, 0)
    True
    """
    matmul, _ = calc_dense_forward_flops(widths, batch)
    return sum(matmul), sum(matmul[1:])


def calc_dense_traffic_values(widths, batch: int = 1) -> int:
    """Count activation values transferred by unfused dense layers.

    Parameters
    ----------
    widths : sequence of int
        Positive layer widths [input, hidden..., output]; at least two entries. Counts are
        dimensionless.
    batch : int, optional
        Positive number of examples processed together; default 1.

    Returns
    -------
    int
        Activation values moved per forward pass, not bytes.

    Notes
    -----
    Equation: V = B [n_0 + 2 sum(hidden widths) + n_L].

    Assumptions: Each layer reads its input and writes its output to main memory.

    Excludes: Weight reads, cache reuse, fusion, and conversion from values to bytes.

    References
    ----------
    Derivation: V = B [n_0 + 2 sum(hidden widths) + n_L].

    Examples
    --------
    >>> calc_dense_traffic_values([2, 3, 1]) == 2 + 2 * 3 + 1
    True
    """
    widths = list(widths)
    return batch * (widths[0] + 2 * sum(widths[1:-1]) + widths[-1])


def calc_dense_saved_activations(widths, batch: int = 1) -> int:
    """Count input and layer-output values retained for backward work.

    Parameters
    ----------
    widths : sequence of int
        Positive layer widths [input, hidden..., output]; at least two entries. Counts are
        dimensionless.
    batch : int, optional
        Positive number of examples processed together; default 1.

    Returns
    -------
    int
        Saved activation value count.

    Notes
    -----
    Equation: V = B sum_l n_l, including input width.

    Assumptions: Every layer output and the original input remain resident.

    Excludes: Recomputation, masks, workspace, and weight or optimizer storage.

    References
    ----------
    Derivation: V = B sum_l n_l, including input width.

    Examples
    --------
    >>> calc_dense_saved_activations([2, 3, 1], batch=2) == 2 * (2 + 3 + 1)
    True
    """
    return batch * sum(widths)


def calc_dense_inference_activations(widths, batch: int = 1) -> int:
    """Count peak activation values using adjacent rolling buffers.

    Parameters
    ----------
    widths : sequence of int
        Positive layer widths [input, hidden..., output]; at least two entries. Counts are
        dimensionless.
    batch : int, optional
        Positive number of examples processed together; default 1.

    Returns
    -------
    int
        Peak simultaneously resident activation values.

    Notes
    -----
    Equation: V_peak = B max_l(n_(l-1) + n_l).

    Assumptions: Layers execute serially; consumed buffers can be reused.

    Excludes: Skip connections, concurrent execution, weights, and runtime workspace.

    References
    ----------
    Derivation: V_peak = B max_l(n_(l-1) + n_l).

    Examples
    --------
    >>> calc_dense_inference_activations([2, 3, 1]) == 2 + 3
    True
    """
    return batch * max(a + b for a, b in _pairs(widths))


def calc_training_step_ops(layers: int) -> int:
    """Count launches in the assumed dense ReLU training schedule.

    Parameters
    ----------
    layers : int
        Positive number of dense layers.

    Returns
    -------
    int
        Operation-launch count per training step, not FLOPs.

    Notes
    -----
    Equation: N = L + 1 + (2L - 1) + (L - 1) + 1 = 4L.

    Assumptions: Each forward matrix operation fuses bias and ReLU; loss and optimizer each
    use one launch. Backward computes all weight gradients and all but the first input
    gradient.

    Excludes: Compiler fusion across layers, communication, and optimizer implementations
    with multiple launches.

    References
    ----------
    Derivation: N = L + 1 + (2L - 1) + (L - 1) + 1 = 4L.

    Examples
    --------
    >>> calc_training_step_ops(3) == 4 * 3
    True
    """
    validate_at_least(layers, 1, "layers")
    return layers + 1 + (2 * layers - 1) + (layers - 1) + 1


def calc_training_state(parameters: int, bytes_per_value, optimizer_bytes_per_parameter):
    """Separate weight, gradient, and optimizer storage.

    Parameters
    ----------
    parameters : int
        Nonnegative trainable parameter count.
    bytes_per_value : float or Quantity
        Bytes per weight and gradient; use matching storage units across both storage
        inputs.
    optimizer_bytes_per_parameter : float or Quantity
        Additional optimizer bytes per parameter, supplied explicitly.

    Returns
    -------
    tuple
        Weights, gradients, and optimizer storage; unit-bearing inputs retain their units.

    Notes
    -----
    Equation: M_weights = M_gradients = P b; M_optimizer = P o.

    Assumptions: One common weight/gradient precision; optimizer storage is supplied
    separately.

    Excludes: Activation memory, runtime buffers, and any master weights not included in the
    optimizer input.

    References
    ----------
    Derivation: M_weights = M_gradients = P b; M_optimizer = P o.

    Examples
    --------
    >>> calc_training_state(3, 2, 8) == (3 * 2, 3 * 2, 3 * 8)
    True
    """
    weights = parameters * bytes_per_value
    return weights, parameters * bytes_per_value, parameters * optimizer_bytes_per_parameter


def calc_sigmoid(x: float) -> float:
    """Evaluate the logistic sigmoid.

    Parameters
    ----------
    x : float
        Dimensionless finite activation; extreme negative inputs may overflow the
        exponential.

    Returns
    -------
    float
        Dimensionless sigmoid activation between zero and one.

    Notes
    -----
    Equation: s(x) = 1 / (1 + exp(-x)).

    Assumptions: Scalar arithmetic using the standard exponential implementation.

    Excludes: Vectorization and overflow-stabilized evaluation.

    References
    ----------
    Derivation: s(x) = 1 / (1 + exp(-x)).

    Examples
    --------
    >>> calc_sigmoid(0.0)
    0.5
    """
    return 1.0 / (1.0 + math.exp(-x))


def calc_sigmoid_slope(x: float) -> float:
    """Evaluate the derivative of the logistic sigmoid.

    Parameters
    ----------
    x : float
        Dimensionless finite activation; inherits sigmoid exponential limits.

    Returns
    -------
    float
        Dimensionless local derivative.

    Notes
    -----
    Equation: s_prime(x) = s(x) [1 - s(x)].

    Assumptions: The derivative is evaluated at a single scalar activation.

    Excludes: Products across layers and numerical overflow stabilization.

    References
    ----------
    Derivation: Differentiate s(x) = 1 / (1 + exp(-x)) by the chain rule.

    Examples
    --------
    >>> calc_sigmoid_slope(0.0) == 0.5 * (1 - 0.5)
    True
    """
    s = calc_sigmoid(x)
    return s * (1.0 - s)


def calc_cross_entropy(p_correct: float) -> float:
    """Evaluate scalar one-hot cross-entropy in natural-log units.

    Parameters
    ----------
    p_correct : float
        Probability assigned to the correct class, in (0, 1].

    Returns
    -------
    float
        Loss in nats.

    Notes
    -----
    Equation: H = -ln(p_correct).

    Assumptions: One-hot target with the correct-class probability already normalized.

    Excludes: Label smoothing, class weights, batch reduction, and softmax normalization.

    References
    ----------
    Derivation: In -sum_i y_i ln(p_i), a one-hot target leaves only the correct-class term.

    Examples
    --------
    >>> calc_cross_entropy(1.0) == 0.0
    True
    """
    if not 0 < p_correct <= 1:
        raise ValueError(f"p_correct must lie in (0, 1], got {p_correct}")
    return -math.log(p_correct)


def calc_relu(x: float) -> float:
    """Apply the scalar rectified linear activation.

    Parameters
    ----------
    x : float
        Dimensionless scalar activation.

    Returns
    -------
    float
        Zero for a negative input; otherwise the unchanged input.

    Notes
    -----
    Equation: ReLU(x) = max(0, x).

    Assumptions: Scalar input.

    Excludes: Leaky variants, derivatives, and tensor execution costs.

    References
    ----------
    Derivation: Rectification preserves a nonnegative scalar and replaces a negative scalar
    with zero; the maximum with zero implements this piecewise definition.

    Examples
    --------
    >>> calc_relu(-2.0)
    0.0
    """
    return max(0.0, x)
