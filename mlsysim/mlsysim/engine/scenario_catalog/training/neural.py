"""Named training scenarios for neural.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....datasets.registry import Datasets as _Datasets
from ....core.units import BYTES_FP32 as _BYTES_FP32, BYTES_INT8 as _BYTES_INT8
from ....ops import Ops as _Ops
from ... import scenario_types as _st
from ....core.units import BYTES_ADAM_MOMENTS as _BYTES_ADAM_MOMENTS


MnistDenseNetwork = _st.DenseNetworkScenario(
    name="MNIST digit classifier, 784-128-64-10, batch 32, FP32, Adam",
    description="A fully connected network from the 784 pixels of a 28 x 28 MNIST digit through hidden "
    "layers of 128 and 64 units to the 10 class scores, trained in batches of 32.",
    layer_widths=[
        _Datasets.MNIST.image_width * _Datasets.MNIST.image_height,
        128,
        64,  # scenario assumption: hidden widths
        _Datasets.MNIST.num_classes,
    ],
    batch_size=32,  # scenario assumption: mini-batch size
    bytes_per_value=_BYTES_FP32,
    optimizer_bytes_per_parameter=_BYTES_ADAM_MOMENTS,
    dataset=_Datasets.MNIST,
)


MnistDenseNetworkWiderHidden = MnistDenseNetwork.variant(
    name="MNIST digit classifier with both hidden widths doubled, 784-256-128-10",
    layer_widths=(
        MnistDenseNetwork.layer_widths[:1]
        + [2 * w for w in MnistDenseNetwork.layer_widths[1:-1]]
        + MnistDenseNetwork.layer_widths[-1:]
    ),
)


MnistDenseNetworkDoubledInputSide = MnistDenseNetwork.variant(
    name="MNIST digit classifier on 56 x 56 inputs (the image side doubled)",
    layer_widths=[(2 * _Datasets.MNIST.image_width) * (2 * _Datasets.MNIST.image_height)]
    + MnistDenseNetwork.layer_widths[1:],
)


MnistDenseNetworkLargeBatch = MnistDenseNetwork.variant(
    name="MNIST digit classifier at batch 1,024",
    batch_size=1_024,  # scenario assumption: a large batch, 32 times the running one
)


DenseWidthDepthSweep = _st.DenseScalingSweepScenario(
    name="Dense networks on MNIST swept in hidden width and in depth",
    description="Two hidden layers of width 64 to 4,096; and 1 to 32 hidden layers of width 512; "
    "784 inputs and 10 outputs throughout.",
    input_dim=MnistDenseNetwork.layer_widths[0],
    output_dim=MnistDenseNetwork.layer_widths[-1],
    hidden_layers=2,  # scenario assumption: matches the running network
    widths=[2**k for k in range(6, 13)],  # scenario assumption: 64 to 4,096
    fixed_width=512,  # scenario assumption
    max_depth=32,  # scenario assumption
)


MnistEpochCpuVersusH100 = _st.EpochOverheadScenario(
    name="One MNIST training epoch on a desktop CPU and on an H100, FP32",
    description="1,875 steps of the 784-128-64-10 network at batch 32; each launched operation pays "
    "Python dispatch, and on the accelerator also a kernel launch.",
    network=MnistDenseNetwork,
    host=_Hardware.Cloud.ReferenceCPU,
    accelerator=_Hardware.Cloud.H100,
    precision="fp32",
    dispatch=_Ops.RuntimeOverheads.PythonDispatch,
    launch=_Ops.RuntimeOverheads.KernelLaunch,
    dram_energy_per_byte=_Hardware.Tech.Memory.DRAM.energy_per_byte,
    # One FLOP is half a multiply-accumulate: the mean of the multiply and add energies.
    flop_energy=(_Hardware.Tech.Op.FlopFp32.energy + _Hardware.Tech.Op.AddFp32.energy) / 2,
)


DigitRecognitionParadigms = _st.ParadigmCostScenario(
    name="One MNIST digit by rules, by HOG features, and by the dense network",
    description="Illustrative implementations: about 100 pixel comparisons; a 7 x 7 grid of 9-bin "
    "HOG cells in about 8,000 operations and 2 KB; the 784-128-64-10 network.",
    image_pixels=_Datasets.MNIST.image_width * _Datasets.MNIST.image_height,
    bytes_per_pixel=_BYTES_INT8,  # 8-bit grayscale pixels
    rule_comparisons=100,  # scenario assumption: illustrative rule tree
    feature_grid=7,  # scenario assumption: HOG cells per side
    feature_bins=9,  # scenario assumption: orientation bins per cell
    feature_operations=8_000,  # scenario assumption: HOG plus a linear classifier
    feature_memory=Q_(2, "KB"),  # scenario assumption: HOG working memory
    network=MnistDenseNetwork,
)


ReluVersusSigmoidLogic = _st.ActivationLogicCostScenario(
    name="Logic cost of a ReLU and of a sigmoid datapath",
    description="A stylized teaching budget in transistor equivalents, not a measured constant: a "
    "sign-bit comparator and multiplexer against an exponential approximation.",
    cheap_label="ReLU",
    cheap_transistors=50,  # scenario assumption: comparator plus selection
    costly_label="sigmoid",
    costly_transistors=2_500,  # scenario assumption: exponential approximation
)


TwentySigmoidLayers = _st.GradientChainScenario(
    name="Gradient through 20 sigmoid layers, at the sigmoid's steepest slope",
    description="The activation-slope factor alone, before the weights multiply in.",
    depth=20,  # scenario assumption
    activation="sigmoid",
)


DigitCrossEntropyExamples = _st.CrossEntropyScenario(
    name="Cross-entropy for one MNIST digit: confident, correct, and misread",
    description="Probabilities on the true class of 0.9 and 0.8, and a seven read as a three "
    "(0.6 on three, 0.1 on seven).",
    classes=_Datasets.MNIST.num_classes,
    probabilities={"confident": 0.9, "correct": 0.8},  # scenario assumption
    misread={"true": 0.1, "wrong": 0.6},  # scenario assumption
)


Gpt2Fp32AdamOnH100 = _st.TrainingStateFitScenario(
    name="GPT-2 (1.5B) trained in FP32 with Adam, against one H100's memory",
    description="Weights, gradients, and two Adam values per parameter, all FP32, before activations.",
    workload=_Models.Language.GPT2,
    hardware=_Hardware.Cloud.H100,
    bytes_per_value=_BYTES_FP32,
    optimizer_bytes_per_parameter=_BYTES_ADAM_MOMENTS,
)


ResNet50Fp32SgdMomentum = _st.TrainingStateFitScenario(
    name="ResNet-50 trained in FP32 with SGD and momentum",
    description="Weights, gradients, and one momentum value per parameter, all FP32, before activations.",
    workload=_Models.Vision.ResNet50,
    bytes_per_value=_BYTES_FP32,
    optimizer_bytes_per_parameter=_BYTES_FP32,  # one momentum value per parameter
)
