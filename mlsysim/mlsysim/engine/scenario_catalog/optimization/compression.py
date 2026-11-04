"""Named optimization scenarios for compression.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....core.units import BYTES_FP16 as _BYTES_FP16
from ....core.units import BYTES_FP32 as _BYTES_FP32, BYTES_INT8 as _BYTES_INT8
from ....core.units import BYTES_INT4 as _BYTES_INT4
from ....core.units import BYTES_INT16 as _BYTES_INT16
from ... import scenario_types as _st


# --- Model compression (added 2026-10-20) ---------------------------------
Llama2_7BTokenOnFlagshipPhone = _st.WeightReadScenario(
    name="Llama 2 7B, FP16 weights read once per token on a flagship phone",
    description="Batch-one decoding reads every weight over the phone's LPDDR bus; DRAM energy per "
    "byte prices the reads.",
    workload=_Models.Language.Llama2_7B,
    hardware=_Hardware.Mobile.iPhone15Pro,
    bytes_per_weight=_BYTES_FP16,
    energy_per_byte=_Hardware.Tech.Memory.DRAM.energy_per_byte,
)


MobileNetV2Fp32WeightReadPhone = Llama2_7BTokenOnFlagshipPhone.variant(
    name="MobileNetV2, FP32 weights read once per frame on a flagship phone",
    description="The weight-reading floor of one batch-one frame.",
    workload=_Models.Vision.MobileNetV2,
    bytes_per_weight=_BYTES_FP32,
    energy_per_byte=None,
)


MobileNetV2Int8WeightReadPhone = MobileNetV2Fp32WeightReadPhone.variant(
    name="MobileNetV2, INT8 weights read once per frame on a flagship phone",
    bytes_per_weight=_BYTES_INT8,
)


ResNet50Fp32WeightReadPhone = MobileNetV2Fp32WeightReadPhone.variant(
    name="ResNet-50, FP32 weights read once per image on a flagship phone",
    workload=_Models.Vision.ResNet50,
)


Gpt2XlFp16WeightReadPhone = Llama2_7BTokenOnFlagshipPhone.variant(
    name="GPT-2 XL, FP16 weights read once per token on a flagship phone",
    description="Batch-one decoding reads every weight over the phone's LPDDR bus.",
    workload=_Models.Language.GPT2,
    energy_per_byte=None,
)


Gpt2XlInt4WeightReadPhone = Gpt2XlFp16WeightReadPhone.variant(
    name="GPT-2 XL, INT4 weights read once per token on a flagship phone",
    bytes_per_weight=_BYTES_INT4,
)


MobileNetV2BlurPipelinePhone = _st.FrameCompressionScenario(
    name="Real-time background blur with MobileNetV2 on a mid-tier phone",
    description="A labeled composite: FP32 on the CPU, INT8 on the integer accelerator, and the "
    "width-0.75 network as the channel-pruned model, against a 30 FPS camera.",
    full=_Models.Vision.MobileNetV2,
    pruned=_Models.Vision.MobileNetV2_Alpha0_75,  # scenario assumption: pruning keeps 75 percent of every layer's channels
    frame_rate=Q_(30, "1/s"),  # scenario assumption: camera frame rate
    cpu_rate=Q_(8, "1/s"),  # scenario assumption: FP32 on the phone CPU (illustrative)
    accelerator_rate=Q_(35, "1/s"),  # scenario assumption: INT8 on the integer accelerator (illustrative)
    float_bytes=_BYTES_FP32,
    int_bytes=_BYTES_INT8,
)


Rank128Factorization4096 = _st.LowRankUpdateScenario(
    name="A 4,096 by 4,096 FP32 layer factored at rank 128",
    description="Two thin factors (4,096 by 128 and 128 by 4,096) replace the dense matrix.",
    in_dim=4096,  # scenario assumption
    out_dim=4096,  # scenario assumption
    rank=128,  # scenario assumption: captures nearly all of the layer
    bytes_per_param=_BYTES_FP32,
)


ResNet50Conv256FusionH100 = _st.ConvBnReluFusionScenario(
    name="One ResNet-50 3x3 layer with batch norm and ReLU, fused or not, on an H100",
    description="256 channels at 28 by 28 in FP32; traffic over HBM bandwidth, compute ignored.",
    channels=256,
    height=28,
    width=28,  # a ResNet-50 stage-2 layer
    kernel_size=3,
    bytes_per_value=_BYTES_FP32,
    norm_params_per_channel=2,  # batch norm's folded scale and shift
    hardware=_Hardware.Cloud.H100,
)


WearableMobileNetV2SparseFlash = _st.SparseFlashFitScenario(
    name="MobileNetV2 pruned to 2 MB of FP32 values for a wearable's 2 MB of flash",
    description="Unstructured pruning; each surviving value carries a 16-bit position index.",
    workload=_Models.Vision.MobileNetV2,
    value_bytes=_BYTES_FP32,
    index_bytes=_BYTES_INT16,
    surviving_bytes=Q_(2, "MB"),  # scenario assumption: surviving FP32 values after pruning
    flash_budget=Q_(2, "MB"),  # scenario assumption: wearable microcontroller flash
)


PrunedInt8PaperVsMeasured = _st.SpeedupCompositionScenario(
    name="Half the weights pruned and the rest quantized to INT8: paper against measured",
    description="Structured pruning of half the weights and FP32-to-INT8 storage, multiplied on "
    "paper, against a hypothetical realized speedup.",
    factors={
        "pruning": 1 / (1 - 0.5),  # scenario assumption: 50 percent structured pruning
        "quantization": (_BYTES_FP32 / _BYTES_INT8).to("").magnitude,
    },
    measured=1.5,  # scenario assumption: hypothetical realized speedup
)


IrregularSparsityLatencyTrace = _st.WorkLatencyGapScenario(
    name="A pruned model whose irregular sparsity barely moves latency",
    description="A representative trace: parameters fall far more than latency.",
    work_reduction=0.40,  # scenario assumption: share of parameters removed
    latency_reduction=0.12,  # scenario assumption: share of latency removed
)


BertBaseMobileCompression = _st.StagedCompressionScenario(
    name="BERT-Base pruned, distilled, and quantized for a mobile CPU (modeled)",
    description="A stipulated compact student; accuracies after each stage and with quantization first.",
    teacher=_Models.Language.BERT_Base,
    student_params=Q_(27_500_000, "param"),  # scenario assumption: final pruned graph, the distillation student
    float_bytes=_BYTES_FP32,
    int_bytes=_BYTES_INT8,
    baseline_accuracy=0.762,  # scenario assumption: before compression
    pruned_accuracy=0.751,  # scenario assumption: after pruning
    distilled_accuracy=0.759,  # scenario assumption: after distillation
    final_accuracy=0.756,  # scenario assumption: after INT8 quantization
    mobile_speedup=12,  # scenario assumption: mobile CPU speedup
    quantize_first_loss=0.021,  # scenario assumption: loss when quantization precedes pruning
)


ModelFifthOfRequestLatency = _st.AmdahlScenario(
    name="A request path whose model is 20 percent of latency, made infinitely fast",
    description="Decoding, resizing, and display hold the other 80 percent.",
    stages={"model": Q_(0.2, ""), "rest": Q_(0.8, "")},  # scenario assumption: shares of latency
    stage="model",
    speedup=float("inf"),
)
