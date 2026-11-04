"""Named optimization scenarios for acceleration.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_INT8 as _BYTES_INT8
from ... import scenario_types as _st


# --- Hardware acceleration: reuse, rooflines, and limits on one accelerator (added 2026-10-20) ---
H100OverReferenceCpu = _st.PeakRatioRangeScenario(
    name="H100 dense FP16 peak against a 1 to 2 TFLOP/s reference CPU",
    description="The reference CPU's registry peak is the low end of the range.",
    rate=_Hardware.Cloud.H100.compute.peak_flops,
    baseline_low=_Hardware.Cloud.ReferenceCPU.compute.peak_flops,
    baseline_high=Q_(2, "TFLOPs / s"),  # scenario assumption: upper end of the reference-CPU range
)


SystolicArrayReuseEnergy = _st.SystolicReuseEnergyScenario(
    name="Energy per MAC: every operand from DRAM against a 128-wide systolic array",
    description="A deliberately naive streaming baseline makes four DRAM accesses per MAC; the array "
    "loads two operands at its edges for every 128 MACs and keeps partial sums inside.",
    access_energy=_Hardware.Tech.Memory.DRAM.energy_per_access,
    mac_energy=Q_(1, "pJ"),  # scenario assumption: about 1 pJ per MAC
    naive_accesses=4,  # scenario assumption: load both operands and the sum, write the sum
    edge_loads=2,  # two operands enter at the array's edges
    array_dim=128,  # scenario assumption: a 128-wide array
)


ResNet50InferenceH100OverServerCpu = _st.AcceleratedFractionScenario(
    name="ResNet-50 inference moved to an H100's INT8 path",
    description="95 percent of baseline time runs on the matrix path; the H100's INT8 peak against "
    "an assumed 8 TOPS server CPU without matrix extensions sets the path gain.",
    fraction=0.95,  # scenario assumption: accelerated share of time
    accelerated_rate=_Hardware.Cloud.H100.compute.precision_flops["int8"],
    baseline_rate=Q_(8, "TOPS"),  # scenario assumption: server-CPU INT8 rate
)


Gpt2DecodeH100OverServerCpu = ResNet50InferenceH100OverServerCpu.variant(
    name="GPT-2 XL token generation moved to an H100's INT8 path",
    description="80 percent of each token's time on the matrix path; host overhead and sampling stay.",
    fraction=0.80,  # scenario assumption: accelerated share of time
)


Gpt2DecodeH100OverServerCpuBandwidth = Gpt2DecodeH100OverServerCpu.variant(
    name="GPT-2 XL token generation with a bandwidth-set path gain",
    description="Decode is bandwidth bound, so the path gain is the H100's memory bandwidth over the "
    "reference CPU socket's.",
    accelerated_rate=_Hardware.Cloud.H100.memory.bandwidth,
    baseline_rate=_Hardware.Cloud.ReferenceCPU.memory.bandwidth,
)


ResNet50Batch1H100Int8PeakRate = _st.SingleNodeInferenceScenario(
    name="ResNet-50, one image, on an H100 at INT8 and peak rates",
    description="Operations over the INT8 tensor peak: an arithmetic floor, not a latency.",
    workload=_Models.Vision.ResNet50,
    hardware=_Hardware.Cloud.H100,
    precision="int8",
    efficiency=1.0,  # scenario assumption: peak rates (lower bounds)
)


ResNet50Batch1H100Fp16PeakRate = ResNet50Batch1H100Int8PeakRate.variant(
    name="ResNet-50, one image, on an H100 at FP16 and peak rates",
    precision="fp16",
)


# ResNet-50 layer shapes (He et al. 2016): stage 2 is 56 x 56 with 64-channel
# bottlenecks expanding to 256; stage 3 is 28 x 28 with 128 channels; the
# classifier maps 2,048 pooled features to 1,000 classes.
ResNet50Stage3ConvBatch32 = _st.ConvLayerScenario(
    name="ResNet-50 stage-3 3x3 convolution, batch 32, FP16",
    description="28 x 28 x 128 to 128 channels; input and weights read once, output written once.",
    height=28,
    width=28,
    kernel=3,
    in_channels=128,
    out_channels=128,
    bytes_per_value=_BYTES_FP16,
    batch_size=32,  # scenario assumption: inference batch
)


ResNet50Stage3ConvBatch32Int8 = ResNet50Stage3ConvBatch32.variant(
    name="ResNet-50 stage-3 3x3 convolution, batch 32, INT8", bytes_per_value=_BYTES_INT8
)


ResNet50Stage3ConvBatch32Int8Weights = ResNet50Stage3ConvBatch32.variant(
    name="ResNet-50 stage-3 3x3 convolution, batch 32, INT8 weights and FP16 activations",
    weight_bytes_per_value=_BYTES_INT8,
)


ResNet50Stage2ConvBatch32 = ResNet50Stage3ConvBatch32.variant(
    name="ResNet-50 stage-2 3x3 convolution, batch 32, FP16",
    description="56 x 56 x 64 to 64 channels.",
    height=56,
    width=56,
    in_channels=64,
    out_channels=64,
)


ResNet50Stage2ConvBatch32Int8 = ResNet50Stage2ConvBatch32.variant(
    name="ResNet-50 stage-2 3x3 convolution, batch 32, INT8", bytes_per_value=_BYTES_INT8
)


ResNet50Stage2ExpansionBatch32 = ResNet50Stage2ConvBatch32.variant(
    name="ResNet-50 stage-2 1x1 expansion, batch 32, FP16",
    description="56 x 56 x 64 to 256 channels.",
    kernel=1,
    out_channels=256,
)


ResNet50Stage2ExpansionBatch32Int8 = ResNet50Stage2ExpansionBatch32.variant(
    name="ResNet-50 stage-2 1x1 expansion, batch 32, INT8", bytes_per_value=_BYTES_INT8
)


ResNet50ClassifierHeadBatch32 = _st.DenseLayerScenario(
    name="ResNet-50 classifier head, batch 32, FP16",
    description="2,048 pooled features to 1,000 classes; each weight is reused only across the batch.",
    in_features=2048,
    out_features=1000,
    bytes_per_weight=_BYTES_FP16,
    batch_size=32,  # scenario assumption: inference batch
)


Gpt2XlQkvProjectionPrefill = _st.DenseLayerScenario(
    name="GPT-2 XL QKV projection at prefill, batch 8 full-context sequences, FP16",
    description="Hidden width to three projections; every token of the batch reuses each weight.",
    in_features=_Models.Language.GPT2.hidden_dim,
    out_features=3 * _Models.Language.GPT2.hidden_dim,
    bytes_per_weight=_BYTES_FP16,
    batch_size=8 * _Models.Language.GPT2.context_length,  # scenario assumption: 8 sequences
)


Gpt2XlLayerNormPrefill = _st.ElementwiseOpScenario(
    name="GPT-2 XL LayerNorm at prefill, batch 8 full-context sequences, FP16",
    description="About eight operations per element read and written; scale and shift read once.",
    elements=8 * _Models.Language.GPT2.context_length * _Models.Language.GPT2.hidden_dim,
    flops_per_element=8,  # scenario assumption: mean, variance, normalize, scale, shift
    bytes_per_value=_BYTES_FP16,
    parameter_values=2 * _Models.Language.GPT2.hidden_dim,
)


Gpt2XlSoftmaxPrefill = _st.ElementwiseOpScenario(
    name="GPT-2 XL attention softmax at prefill, batch 8 full-context sequences, FP16",
    description="One score per head per query-key pair; exponential, sum, and divide per score.",
    elements=8 * _Models.Language.GPT2.heads * _Models.Language.GPT2.context_length**2,
    flops_per_element=3,  # scenario assumption: exp, sum, divide
    bytes_per_value=_BYTES_FP16,
)


Gpt2XlDecodeH100Roofline = _st.RooflinePointScenario(
    name="GPT-2 XL decoding one token at batch 1 on the H100 FP16 roofline",
    description="Two operations per parameter against every FP16 weight read once.",
    hardware=_Hardware.Cloud.H100,
    intensity=(_Models.Language.GPT2.inference_flops / _Models.Language.GPT2.size_in_bytes(_BYTES_FP16)).to(
        "flop/byte"
    ),
)


Gpt2XlLayerNormH100Roofline = Gpt2XlDecodeH100Roofline.variant(
    name="GPT-2 XL LayerNorm on the H100 FP16 roofline",
    description="The unfused LayerNorm's intensity against the H100's ceilings.",
    intensity=Q_(Gpt2XlLayerNormPrefill.intensity, "flop/byte"),
)


ResNet50ClassifierHeadH100Roofline = Gpt2XlDecodeH100Roofline.variant(
    name="ResNet-50 classifier head on the H100 FP16 roofline",
    description="The batch-32 head's intensity against the H100's ceilings.",
    intensity=Q_(ResNet50ClassifierHeadBatch32.traffic_intensity, "flop/byte"),
)


KernelAt200FlopPerByteA100 = _st.RooflinePointScenario(
    name="A 200 FLOP/byte kernel on the A100 FP16 roofline",
    description="Above the A100 ridge: compute bound at its dense FP16 peak.",
    hardware=_Hardware.Cloud.A100,
    intensity=Q_(200, "flop/byte"),  # scenario assumption: a kernel tuned for the A100
)


KernelAt200FlopPerByteH100 = KernelAt200FlopPerByteA100.variant(
    name="The same 200 FLOP/byte kernel on the H100 FP16 roofline",
    description="Below the H100 ridge: bandwidth bound.",
    hardware=_Hardware.Cloud.H100,
)


Dense2048Fp16Batch1 = _st.DenseLayerScenario(
    name="A 2,048-to-2,048 dense layer in FP16 at batch 1",
    description="Each weight read serves one example; intensity is about 1 FLOP/byte.",
    in_features=2048,
    out_features=2048,  # scenario assumption: production-width layer
    bytes_per_weight=_BYTES_FP16,
    batch_size=1,
)


Dense2048Fp16Batch32 = Dense2048Fp16Batch1.variant(
    name="A 2,048-to-2,048 dense layer in FP16 at batch 32", batch_size=32
)  # scenario assumption


Dense2048Fp16Batch512 = Dense2048Fp16Batch1.variant(
    name="A 2,048-to-2,048 dense layer in FP16 at batch 512", batch_size=512
)  # scenario assumption


DenseParametersVsBandwidthSince2014 = _st.ParameterBandwidthGrowthScenario(
    name="Dense model parameters against per-device memory bandwidth since 2014",
    description="VGG-16 and one GPU of a K80 board are the 2014 baselines; DeepSeek-V3 is a sparse "
    "expert model, reported but kept out of the growth figure.",
    accelerators=[
        _Hardware.Cloud.K80,
        _Hardware.Cloud.TPUv2,
        _Hardware.Cloud.V100,
        _Hardware.Cloud.A100,
        _Hardware.Cloud.TPUv4,
        _Hardware.Cloud.H100,
        _Hardware.Cloud.MI300X,
        _Hardware.Cloud.TPUv6,
        _Hardware.Cloud.MI325X,
        _Hardware.Cloud.B200,
    ],
    dense_models=[
        _Models.Vision.AlexNet,
        _Models.Vision.VGG16,
        _Models.Vision.ResNet50,
        _Models.Language.BERT_Large,
        _Models.Language.GPT3,
        _Models.Language.PaLM_540B,
        _Models.Language.Llama3_405B,
    ],
    sparse_models=[_Models.Language.DeepSeek_V3],
    baseline_accelerator=_Hardware.Cloud.K80,
    baseline_model=_Models.Vision.VGG16,
)


NvidiaGenerationsV100ToB200 = _st.AcceleratorGenerationsScenario(
    name="Dense FP16 peak and memory bandwidth, V100 through B200",
    description="Five NVIDIA datacenter generations; the H200 keeps the H100's compute with more bandwidth.",
    accelerators=[
        _Hardware.Cloud.V100,
        _Hardware.Cloud.A100,
        _Hardware.Cloud.H100,
        _Hardware.Cloud.H200,
        _Hardware.Cloud.B200,
    ],
)
