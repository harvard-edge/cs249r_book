"""Named training scenarios for layers.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....datasets.registry import Datasets as _Datasets
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32
from ... import scenario_types as _st
from ....physics.layers import calc_resnet_bottleneck_macs as _resnet_macs

_RESNET50_MACS = _resnet_macs()


# --- Network architectures: dense, convolutional, and recurrent layers (added 2026-10-20) ---
MnistMlpFirstLayer = _st.DenseLayerScenario(
    name="The first dense layer of the 784-128-64-10 MNIST multilayer perceptron",
    description="Flattened 28x28 digits to 128 hidden units; FP32 weights at batch one.",
    in_features=_Datasets.MNIST.image_width * _Datasets.MNIST.image_height,
    out_features=128,  # scenario assumption: the running network's first hidden width
    bytes_per_weight=_BYTES_FP32,
)


MnistMlpLayerStack = _st.LayerStackScenario(
    name="The 784-128-64-10 MNIST multilayer perceptron, weights",
    description="Hidden widths 128 and 64, ten classes; FP32 weights, biases omitted.",
    dense_layers=[
        MnistMlpFirstLayer,
        MnistMlpFirstLayer.variant(
            name="128 to 64", in_features=MnistMlpFirstLayer.out_features, out_features=64
        ),  # scenario assumption: second hidden width
        MnistMlpFirstLayer.variant(name="64 to 10", in_features=64, out_features=_Datasets.MNIST.num_classes),
    ],
)


Dense2048Layer = _st.DenseLayerScenario(
    name="A production-width dense layer, 2,048 to 2,048, FP32",
    description="Weights grow as the product of the widths.",
    in_features=2048,  # scenario assumption: production-width layer
    out_features=2048,
    bytes_per_weight=_BYTES_FP32,
)


Dense1024LayerFp16 = _st.DenseLayerScenario(
    name="A 1,024-to-1,024 dense layer in FP16",
    description="Quadratic scaling of dense weights at a modest width.",
    in_features=1024,  # scenario assumption: illustrative width
    out_features=1024,
    bytes_per_weight=_BYTES_FP16,
)


MnistFirstConvLayer = _st.ConvLayerScenario(
    name="32 feature detectors on a 28x28 MNIST digit: 3x3 filters against dense units",
    description="One grayscale channel, same padding, FP32 values.",
    height=_Datasets.MNIST.image_height,
    width=_Datasets.MNIST.image_width,
    kernel=3,  # scenario assumption: 3x3 filters
    in_channels=1,  # grayscale input
    out_channels=32,  # scenario assumption: first-layer features
    bytes_per_value=_BYTES_FP32,
)


MnistSmallCnn = _st.LayerStackScenario(
    name="A small MNIST CNN: Conv(32), pool, Conv(64), pool, dense 128, 10 classes",
    description="Two 3x3 convolutions, each followed by a 2x2 pool (28 to 14 to 7), then a dense head.",
    conv_layers=[
        MnistFirstConvLayer,
        MnistFirstConvLayer.variant(
            name="Second convolution, 32 to 64 channels at 14x14",
            height=_Datasets.MNIST.image_height // 2,  # after one 2x2 pool
            width=_Datasets.MNIST.image_width // 2,
            in_channels=MnistFirstConvLayer.out_channels,
            out_channels=64,
        ),
    ],  # scenario assumption
    dense_layers=[
        MnistMlpFirstLayer.variant(
            name="Head: 64 maps of 7x7 to 128",
            in_features=64 * (_Datasets.MNIST.image_width // 4) * (_Datasets.MNIST.image_height // 4),
            out_features=128,
        ),  # scenario assumption: head width
        MnistMlpFirstLayer.variant(name="Head: 128 to 10", in_features=128, out_features=_Datasets.MNIST.num_classes),
    ],
)


ImageNetRgbFilterVersusDense = _st.ConvLayerScenario(
    name="One feature detector on a 224x224 RGB image: a 3x3 filter against a dense unit",
    description="Weights for one output feature; same padding.",
    height=_Datasets.ImageNet.image_height,
    width=_Datasets.ImageNet.image_width,
    kernel=3,  # scenario assumption: 3x3 filter
    in_channels=3,  # RGB
    out_channels=1,  # one feature detector
    bytes_per_value=_BYTES_FP32,
)


ImageNet64ChannelConv = _st.ConvLayerScenario(
    name="A 3x3 convolution on a 224x224x64 feature map to 64 features, FP32",
    description="Same padding; the reference layer for weight sharing, reuse, activations, and factoring.",
    height=_Datasets.ImageNet.image_height,
    width=_Datasets.ImageNet.image_width,
    kernel=3,  # scenario assumption: 3x3 filter
    in_channels=64,  # scenario assumption: early-layer channel count
    out_channels=64,
    bytes_per_value=_BYTES_FP32,
)


ResidualAddFp32 = _st.ElementwiseIntensityScenario(
    name="An unfused FP32 residual addition",
    description="One FLOP per element; two operands read and one result written.",
    flops_per_element=1,
    operands_moved=3,
    bytes_per_value=_BYTES_FP32,
)


ResNet50ConvKernelShare = _st.KernelShareScenario(
    name="ResNet-50's 3x3 convolutions as a share of its multiply-adds",
    description="Per 224x224 image, derived from the architecture of He et al. (2016), Table 1 "
    "(original network: stride on the first 1x1 of a downsampling block).",
    workload=_Models.Vision.ResNet50,
    kernel_macs=Q_(_RESNET50_MACS["3x3"] / 1e9, "GFLOP"),  # multiply-adds, 1.850 G
    total_macs=Q_(_RESNET50_MACS["total"] / 1e9, "GFLOP"),  # multiply-adds, 3.858 G
)


ResNet50ThirtyFpsStream = _st.FrameRateHeadroomScenario(
    name="ResNet-50 on a 30 frame-per-second stream on a 10 TFLOP/s GPU",
    description="Planning utilization of 50 to 60 percent of peak; a 100 GFLOP detector as the heavier model.",
    workload=_Models.Vision.ResNet50,
    frame_rate=Q_(30, "Hz"),  # scenario assumption: video frame rate
    peak_rate=Q_(10, "TFLOP/s"),  # scenario assumption: mid-range GPU peak
    utilization_low=0.50,  # scenario assumption: planning utilization range
    utilization_high=0.60,
    alternative_flops=Q_(100, "GFLOP"),  # scenario assumption: object detector per frame
)


ElmanRnnStep = _st.RecurrentStepScenario(
    name="A plain recurrent layer: 100 inputs, a 128-wide state, a 1,000-step sequence",
    description="One affine map per step for the state and one for the input; FP32.",
    input_dim=100,  # scenario assumption: input width
    hidden_dim=128,  # scenario assumption: state width
    bytes_per_value=_BYTES_FP32,
    gates=1,
    steps=1000,  # scenario assumption: sequence length
)


RecurrentState512Fp32 = ElmanRnnStep.variant(
    name="A 512-wide FP32 recurrent state",
    description="The fixed state inference keeps at any stream length.",
    hidden_dim=512,  # scenario assumption: illustrative state width
)
