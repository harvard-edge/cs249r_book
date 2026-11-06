"""Named optimization scenarios for framework.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....datasets.registry import Datasets as _Datasets
from ....core.units import BYTES_FP32 as _BYTES_FP32, BYTES_INT8 as _BYTES_INT8
from ....core.units import BYTES_INT64 as _BYTES_INT64
from ....ops import Ops as _Ops
from ... import scenario_types as _st
from ....core.units import BYTES_ADAM_MOMENTS as _BYTES_ADAM_MOMENTS
from ..training.neural import ResNet50Fp32SgdMomentum


# --- Framework execution: dispatch, fusion, compilation, saved state, copies (added 2026-10-20) ---
Gpt2XlFp16DecodeOnH100 = _st.DecodeScenario(
    name="GPT-2 XL batch-one decode step on one H100, FP16",
    description="Every FP16 weight read once from HBM for each generated token.",
    workload=_Models.Language.GPT2,
    hardware=_Hardware.Cloud.H100,
    precision="fp16",
    batch_size=1,
)


Gpt2XlEagerDecodeDispatch = _st.HostDispatchScenario(
    name="Eager host work to issue one GPT-2 XL decode step on an H100",
    description="One eager kernel per operation in each transformer block, each paying Python dispatch "
    "and a kernel launch, against the step's HBM weight read.",
    # scenario assumption: one eager kernel each for two layer norms, the fused QKV projection,
    # attention scores, softmax, weighted sum, output projection, two feed-forward matmuls,
    # the activation, and two residual adds
    ops_per_block=12,
    blocks=_Models.Language.GPT2.layers,
    dispatch=_Ops.RuntimeOverheads.PythonDispatch,
    launch=_Ops.RuntimeOverheads.KernelLaunch,
    device_time=Gpt2XlFp16DecodeOnH100.result.memory_time,
)


SmallKernelDispatch = _st.HostDispatchScenario(
    name="One small eager operation: a short element-wise kernel",
    description="Python dispatch and one kernel launch against a 1 us kernel (a ReLU on a small vector).",
    ops_per_block=1,
    dispatch=_Ops.RuntimeOverheads.PythonDispatch,
    launch=_Ops.RuntimeOverheads.KernelLaunch,
    device_time=Q_(1, "microsecond"),  # scenario assumption: ReLU on a small vector
)


LargeKernelDispatch = SmallKernelDispatch.variant(
    name="One large eager operation: a long matrix multiply",
    description="The same host cost against a 100 us kernel (a large matrix multiply).",
    device_time=Q_(100, "microsecond"),  # scenario assumption: large matrix multiply
)


SmallMlpDispatch = SmallKernelDispatch.variant(
    name="A small MLP's eager step: six operations, 2.6 us of device work",
    description="Six dispatched operations against the small network's device time.",
    ops_per_block=6,  # scenario assumption: operations in the small MLP's step
    device_time=Q_(2.6, "microsecond"),  # scenario assumption: small MLP device time
)


Gpt3BlockDispatch = SmallKernelDispatch.variant(
    name="One GPT-3-scale transformer block over a large batch, eager",
    description="One block's eager kernels against 100 ms of device time.",
    ops_per_block=Gpt2XlEagerDecodeDispatch.ops_per_block,
    device_time=Q_(100, "ms"),  # scenario assumption: one GPT-3-scale layer over a large batch
)


ElementwisePairFusion = _st.KernelFusionScenario(
    name="Two element-wise kernels against one fused kernel",
    description="Launch overhead and device-memory trips of a two-operation chain, unfused and fused.",
    chain_length=2,  # scenario assumption
    dispatch=_Ops.RuntimeOverheads.PythonDispatch,
    launch=_Ops.RuntimeOverheads.KernelLaunch,
)


ResNet50CompileBreakeven = _st.CompileBreakevenScenario(
    name="ResNet-50 inference, eager against compiled (hypothetical rates)",
    description="Images that repay a compile, and a prototyping run that recompiles on every code change.",
    eager_rate=Q_(1450, "1/second"),  # scenario assumption: hypothetical eager images per second
    compiled_rate=Q_(2150, "1/second"),  # scenario assumption: hypothetical compiled images per second
    compile_time=Q_(30, "second"),  # scenario assumption: compile per code change
    experiment_items=10_000,  # scenario assumption: images in one prototyping run
    recompiles=10,  # scenario assumption: code changes in that run
)


Dense100kLayerJacobianFp32 = _st.JacobianScenario(
    name="Full Jacobian of one 100,000-wide layer, FP32",
    description="The matrix reverse mode never builds: every output against every input.",
    input_width=100_000,  # scenario assumption
    output_width=100_000,  # scenario assumption
    bytes_per_value=_BYTES_FP32,
)


ResNet50Fp32AdamBatch64 = ResNet50Fp32SgdMomentum.variant(
    name="ResNet-50 trained in FP32 with Adam at batch 64, low activation estimate",
    description="Weights, gradients, two Adam values per parameter, and 8 GB of saved activations "
    "for 224x224 images, against one H100's memory.",
    optimizer_bytes_per_parameter=_BYTES_ADAM_MOMENTS,
    hardware=_Hardware.Cloud.H100,
    batch_size=64,  # scenario assumption
    activations=Q_(8, "GB"),  # scenario assumption: saved-activation estimate, low end
)


ResNet50Fp32AdamBatch64HighActivations = ResNet50Fp32AdamBatch64.variant(
    name="ResNet-50 trained in FP32 with Adam at batch 64, high activation estimate",
    description="The same training state with 12 GB of saved activations.",
    activations=Q_(12, "GB"),  # scenario assumption: saved-activation estimate, high end
)


ResNet50Fp32AdamBatch512 = ResNet50Fp32AdamBatch64.variant(
    name="ResNet-50 trained in FP32 with Adam at batch 512, low activation estimate",
    description="The batch-64 state with its saved activations scaled linearly to batch 512.",
    batch_size=512,  # scenario assumption
    activations=ResNet50Fp32AdamBatch64.activations_at(512),  # the batch above
)


Fp32MillionElementTensorAcrossH100Links = _st.TransferScenario(
    name="A 1,000 by 1,000 FP32 tensor across an H100's memory and links",
    description="One 4 MB tensor over HBM, NVLink (one direction), and the PCIe host link. Bandwidth-only bounds.",
    payload=1000 * 1000 * _BYTES_FP32,  # scenario assumption: activation tensor shape
    links={
        "HBM": _Hardware.Cloud.H100.memory.bandwidth,
        "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        "PCIe": _Hardware.Cloud.H100.interconnect.bandwidth,
    },
)


ImageNetBatch64HostFeed = _st.HostInputPipelineScenario(
    name="A data loader feeding 224x224 RGB images to one H100 at batch 64",
    description="Four loader workers prefetching two batches each; uint8 pixels decoded on the host, "
    "FP32 tensors copied over PCIe.",
    batch_size=64,  # scenario assumption
    workers=4,  # scenario assumption
    prefetch_factor=2,  # scenario assumption
    raw_sample=(
        _Datasets.ImageNet.image_width
        * _Datasets.ImageNet.image_height
        * _Datasets.ImageNet.image_channels
        * _BYTES_INT8
    ),
    model_sample=(
        _Datasets.ImageNet.image_width
        * _Datasets.ImageNet.image_height
        * _Datasets.ImageNet.image_channels
        * _BYTES_FP32
    ),
    consume_rate=Q_(1000, "1/second"),  # scenario assumption: accelerator consumption rate
    host_link=_Hardware.Cloud.H100.interconnect.bandwidth,
    device_memory=_Hardware.Cloud.H100.memory.bandwidth,
    step_times={
        "large model": Q_(5, "ms"),  # scenario assumption
        "small model": Q_(0.5, "ms"),  # scenario assumption
    },
)


MnistTwoLayerTrainingStepOnH100 = _st.MlpTrainingStepScenario(
    name="One eager training step of a 784-128-10 MNIST MLP at batch 32 on an H100, FP32",
    description="Per-operator FLOPs and bytes, ideal compute and memory times, and the host's "
    "Python entries and kernel launches.",
    batch_size=32,  # scenario assumption
    input_width=(_Datasets.MNIST.image_width * _Datasets.MNIST.image_height * _Datasets.MNIST.image_channels),
    hidden_width=128,  # scenario assumption: two-layer version of the 784-128-64-10 network
    output_width=_Datasets.MNIST.num_classes,
    bytes_per_value=_BYTES_FP32,
    label_bytes=_BYTES_INT64,
    loss_flops_per_logit=3,  # scenario assumption: one exp, accumulation, and normalization per logit
    optimizer_values_moved=7,  # FP32 Adam reads p, g, m, v and writes p, m, v
    hardware=_Hardware.Cloud.H100,
    precision="fp32",
    forward_ops=4,  # GEMM 1, ReLU, GEMM 2, combined loss
    backward_ops=5,  # three GEMMs, ReLU derivative, loss derivative
    dispatch=_Ops.RuntimeOverheads.PythonDispatch,
    launch=_Ops.RuntimeOverheads.KernelLaunch,
)
