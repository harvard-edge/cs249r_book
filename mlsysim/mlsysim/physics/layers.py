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
    """Calculate receptive-field width for stride-one convolutions.

    Parameters
    ----------
    layers : int
        Nonnegative convolution layer count.
    kernel : int
        Positive kernel width in pixels.

    Returns
    -------
    int
        Receptive-field width in input pixels.

    Notes
    -----
    Equation: R = 1 + L (K - 1).

    Assumptions: Every convolution has the same kernel width, unit stride, and unit
    dilation.

    Excludes: Pooling, strided layers, dilation, and boundary clipping.

    References
    ----------
    Derivation: R = 1 + L (K - 1).

    Examples
    --------
    >>> calc_receptive_field(3, 3) == 1 + 3 * (3 - 1)
    True
    """
    return 1 + layers * (kernel - 1)


def calc_layers_for_receptive_field(field: int, kernel: int) -> int:
    """Find the layer count needed to reach a receptive-field width.

    Parameters
    ----------
    field : int
        Target width in input pixels, at least one.
    kernel : int
        Kernel width in pixels, greater than one.

    Returns
    -------
    int
        Minimum nonnegative count of stride-one convolution layers.

    Notes
    -----
    Equation: L = ceil((R - 1) / (K - 1)).

    Assumptions: Each layer increases receptive-field width by K - 1.

    Excludes: Strides, dilation, and a kernel of width one, which cannot enlarge the field.

    References
    ----------
    Derivation: Solve R = 1 + L(K - 1) for L and round upward to a whole layer.

    Examples
    --------
    >>> calc_layers_for_receptive_field(7, 3) == 3
    True
    """
    return math.ceil((field - 1) / (kernel - 1))


def calc_separable_cost_fraction(kernel: int, out_channels: int) -> float:
    """Compare depthwise-separable and dense convolution work.

    Parameters
    ----------
    kernel : int
        Positive square kernel width.
    out_channels : int
        Positive output channel count.

    Returns
    -------
    float
        Separable-to-dense MAC or weight-count ratio.

    Notes
    -----
    Equation: rho = 1 / C_out + 1 / ``K**2``.

    Assumptions: A depthwise K-by-K convolution is followed by a dense pointwise
    convolution; spatial shape is identical.

    Excludes: Biases, nonlinearities, memory movement, and achieved kernel speed.

    References
    ----------
    Derivation: Divide ``K**2`` C_in + C_in C_out by the dense cost ``K**2`` C_in C_out. This is the counting argument used for depthwise-separable convolutions.

    Examples
    --------
    >>> calc_separable_cost_fraction(3, 8) == 1 / 8 + 1 / 3**2
    True
    """
    return 1 / out_channels + 1 / kernel**2


def calc_gradient_product(gain: float, depth: int) -> float:
    """Multiply equal gradient gains through a plain layer stack.

    Parameters
    ----------
    gain : float
        Dimensionless per-layer scalar derivative factor.
    depth : int
        Nonnegative number of multiplied factors.

    Returns
    -------
    float
        Dimensionless product of local derivative factors.

    Notes
    -----
    Equation: g_total = ``gain**depth``.

    Assumptions: All scalar gains are identical and the computation follows one serial path.

    Excludes: Residual paths, Jacobian spectra, and variation among layers.

    References
    ----------
    Derivation: Apply the scalar chain rule repeatedly across depth layers.

    Examples
    --------
    >>> calc_gradient_product(0.5, 3) == 0.5**3
    True
    """
    return gain**depth


def calc_resnet_bottleneck_macs(
    stages=((64, 3), (128, 4), (256, 6), (512, 3)),
    input_size: int = 224,
    num_classes: int = 1000,
    stride_on_3x3: bool = False,
) -> dict:
    """Count convolution and classifier MACs in a bottleneck ResNet.

    Parameters
    ----------
    stages : sequence of tuple of int, optional
        Positive (bottleneck width, block count) pairs; default describes ResNet-50.
    input_size : int, optional
        Square image side in pixels; default 224. Spatial downsampling uses integer
        division.
    num_classes : int, optional
        Positive classifier output width; default 1000.
    stride_on_3x3 : bool, optional
        Put downsampling stride on the 3-by-3 convolution for v1.5; default False puts it on
        the first 1-by-1.

    Returns
    -------
    dict
        MAC counts under keys 7x7, 3x3, 1x1, fc, and total, per image.

    Notes
    -----
    Equation: MAC_conv = ``H_out**2`` C_in C_out ``K**2``; MAC_fc = C_in C_out; total sums all
    counted layers.

    Assumptions: A 7-by-7 stride-two stem, stride-two pooling, bottleneck expansion by four,
    and projection at each stage's first block.

    Excludes: Bias, batch normalization, pooling, nonlinearities, and residual-add
    operations.

    The original bottleneck places a downsampling stride in its first 1-by-1 convolution.
    Moving that stride to the 3-by-3 convolution retains more first-projection work, so the
    two variants need not have identical MAC counts.

    References
    ----------
    Derivation: Enumerate the stem, three convolutions per bottleneck, stage-entry projection, and classifier; multiply each output element by its convolution reduction length. The architecture follows He et al. (2016), "Deep Residual Learning for Image Recognition," Table 1. https://arxiv.org/abs/1512.03385

    Examples
    --------
    >>> work = calc_resnet_bottleneck_macs(stages=((4, 1),), input_size=32, num_classes=2)
    >>> work["total"] == sum(work[k] for k in ("7x7", "3x3", "1x1", "fc"))
    True
    """

    def conv(h, cin, cout, k, s):
        ho = h // s
        return ho, ho * ho * cout * cin * k * k

    macs = {"7x7": 0, "3x3": 0, "1x1": 0, "fc": 0}
    h, m = conv(input_size, 3, 64, 7, 2)
    macs["7x7"] += m
    h //= 2  # 3x3/2 max pool
    cin = 64
    for i, (width, blocks) in enumerate(stages):
        for b in range(blocks):
            s = 2 if (b == 0 and i > 0) else 1
            h1, m = conv(h, cin, width, 1, 1 if stride_on_3x3 else s)
            macs["1x1"] += m
            h2, m = conv(h1, width, width, 3, s if stride_on_3x3 else 1)
            macs["3x3"] += m
            _, m = conv(h2, width, 4 * width, 1, 1)
            macs["1x1"] += m
            if b == 0:
                _, m = conv(h, cin, 4 * width, 1, s)
                macs["1x1"] += m
            h, cin = h2, 4 * width
    macs["fc"] = cin * num_classes
    macs["total"] = sum(macs.values())
    return macs
