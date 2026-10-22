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


def calc_resnet_bottleneck_macs(stages=((64, 3), (128, 4), (256, 6), (512, 3)), input_size: int = 224,
                                num_classes: int = 1000, stride_on_3x3: bool = False) -> dict:
    """
    Multiply-adds per image of a bottleneck ResNet, by kernel size (added 2026-10-21).

    The architecture of He et al. (2016), Table 1: a 7x7/2 stem, a 3x3/2 max
    pool, then stages of bottleneck blocks (1x1 to ``width``, 3x3, 1x1 to
    4 x ``width``), a 1x1 projection on each stage's first block, global
    pooling, and a fully connected classifier. The default ResNet-50 stages
    give 1.850 G of 3x3 and 3.858 G of total multiply-adds at 224x224. The
    original network strides the first 1x1 of a downsampling block
    (``stride_on_3x3=False``); torchvision's v1.5 strides the 3x3 (about
    4.09 G in total).

    Returns ``{"7x7", "3x3", "1x1", "fc", "total"}`` in multiply-adds.
    """
    def conv(h, cin, cout, k, s):
        ho = h // s
        return ho, ho * ho * cout * cin * k * k

    macs = {"7x7": 0, "3x3": 0, "1x1": 0, "fc": 0}
    h, m = conv(input_size, 3, 64, 7, 2)
    macs["7x7"] += m
    h //= 2                                     # 3x3/2 max pool
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
